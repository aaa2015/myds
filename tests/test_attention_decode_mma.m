/* tests/test_attention_decode_mma.m
 *
 * [D-A1] Single-Pass MQA-64 MMA Decode Attention with Fused RoPE
 * Comprehensive verification & M4 microbenchmark:
 *   1. Strict fail-closed MYDS_ATTN_DECODE_MMA parsing + /debug/state counters.
 *   2. PSO residency inspection (maxTotalThreadsPerThreadgroup, staticThreadgroupMemoryLength)
 *      for baseline register_rope_fuse and all D_SPLIT in {2, 4, 8} x {FP32-P, FP16-P}.
 *   3. Numerical accuracy (max_abs_err, max_rel_err, cosine similarity) vs
 *      kernel_myds_attention_register_rope_fuse_f16_512 across:
 *      - keys in {1, 7, 31, 32, 64, 127, 128, 129, 256, 512, 639, 640}
 *      - RoPE dimensions 448..511 (separate check vs 0..447) at multiple pos & compressed={0,1}
 *      - stale-tail NaN/+Inf poisoning beyond keys (proves zero OOB / 0*NaN hazard)
 *      - exact-sized unpadded K/V buffers (proves zero out-of-bounds read)
 *      - extreme logits (ascending & descending +/-500), all-equal logits, all-masked,
 *        pattern mask, causal prefix mask, disabled sinks (-FLT_MAX/2), -INFINITY mask
 *   4. End-to-end C runtime dispatch (myds_v41_gpu_attention_direct_rope_fuse &
 *      myds_v41_gpu_attention_rope_fuse) with MYDS_ATTN_DECODE_MMA in {1, 2, 4, 8}
 *      (both with MYDS_ATTN_ROPE_FUSE=1 and standalone MYDS_ATTN_ROPE_FUSE=0).
 *   5. (--bench) Batched 40-layer-in-1-CB GPU-timestamp M4 microbenchmark comparing
 *      baseline register_rope_fuse vs D_SPLIT in {2, 4, 8} (FP32-P & FP16-P) across
 *      short & long keys (1, 7, 16, 31, 32, 64, 128, 256, 512, 640).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_metal.h"
#include "myds_v41_gpu.h"

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

/* Exact Metal ABI mirror of ds4_metal_args_flash_attn_ext_vec + myds_attention_rope_fuse_args */
struct test_flash_vec_args {
    int32_t  ne01, ne02, ne03;
    uint64_t nb01, nb02, nb03;
    int32_t  ne11, ne_12_2, ne_12_3, ns10;
    uint64_t nb11, nb12, nb13;
    int32_t  ns20;
    uint64_t nb21, nb22, nb23;
    int32_t  ne31, ne32, ne33;
    uint64_t nb31, nb32, nb33;
    int32_t  ne1, ne2, ne3;
    float    scale, max_bias, m0, m1;
    int32_t  n_head_log2;
    float    logit_softcap;
    uint64_t nb_kvq;
};
_Static_assert(sizeof(struct test_flash_vec_args) == 200u, "flash_vec_args ABI mismatch");

struct test_rope_fuse_args {
    struct test_flash_vec_args attention;
    uint32_t pos;
    uint32_t inverse;
    float    frequencies[32];
};
_Static_assert(sizeof(struct test_rope_fuse_args) == 336u, "rope_fuse_args ABI mismatch");

struct test_pad_args {
    int32_t  ne11, ne_12_2, ne_12_3;
    uint64_t nb11, nb12, nb13;
    uint64_t nb21, nb22, nb23;
    int32_t  ne31, ne32, ne33;
    uint64_t nb31, nb32, nb33;
};

static uint32_t g_rng = 0x2468ace1u;
static uint32_t xorshift32(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}
static float rand_f(float lo, float hi) {
    return lo + (hi - lo) * ((float)(xorshift32() & 0xffffffu) / (float)0x1000000u);
}

static uint16_t f32_to_f16_bits(float v) {
    __fp16 h = (__fp16)v;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static uint16_t f32_to_bf16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16u) & 1u);
    return (uint16_t)(u >> 16u);
}

static float g_rope_freqs[2][32];
static void init_test_rope_freqs(void) {
    const double pi = 3.14159265358979323846;
    for (int kind = 0; kind < 2; kind++) {
        float base = kind ? 160000.0f : 10000.0f;
        float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * pi)) / (2.0 * log(base)));
        float high = (float)ceil(64.0 * log(65536.0 / (2.0 * pi)) / (2.0 * log(base)));
        for (int i = 0; i < 32; i++) {
            float denom = powf(base, (float)i / 32.0f);
            float f = 1.0f / denom;
            if (kind) {
                float ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / (high - low)));
                float smooth = 1.0f - ramp;
                f = (f / 16.0f) * (1.0f - smooth) + f * smooth;
            }
            g_rope_freqs[kind][i] = f;
        }
    }
}

static struct test_rope_fuse_args make_fused_args(uint32_t keys, uint32_t pos, bool compressed) {
    const uint64_t kvbytes = (uint64_t)keys * 1024u;
    const uint64_t maskbytes = (uint64_t)keys * 2u;
    struct test_rope_fuse_args f;
    memset(&f, 0, sizeof(f));
    f.attention = (struct test_flash_vec_args){
        .ne01 = 1, .ne02 = 64, .ne03 = 1,
        .nb01 = 64u * 2048u, .nb02 = 2048u, .nb03 = 64u * 2048u,
        .ne11 = (int32_t)keys, .ne_12_2 = 1, .ne_12_3 = 1, .ns10 = 512,
        .nb11 = 1024u, .nb12 = kvbytes, .nb13 = kvbytes,
        .ns20 = 512, .nb21 = 1024u, .nb22 = kvbytes, .nb23 = kvbytes,
        .ne31 = 1, .ne32 = 1, .ne33 = 1,
        .nb31 = maskbytes, .nb32 = maskbytes, .nb33 = maskbytes,
        .ne1 = 64, .ne2 = 1, .ne3 = 1,
        .scale = 1.0f / sqrtf(512.0f),
    };
    f.pos = pos;
    f.inverse = 1u;
    memcpy(f.frequencies, g_rope_freqs[compressed ? 1 : 0], sizeof(f.frequencies));
    return f;
}

typedef struct {
    double max_abs;
    double max_rel;
    double l2_rel;
    double cosine;
    double rope_max_abs;
    double rope_max_rel;
    double rope_cosine;
    uint32_t max_bf16_ulps;
    uint32_t non_finite;
    float worst_ref, worst_got;
} err_metrics_t;

static uint32_t bf16_ulp_dist(float a, float b) {
    uint16_t ua = f32_to_bf16_bits(a), ub = f32_to_bf16_bits(b);
    if (ua == ub) return 0u;
    int32_t ia = (ua & 0x8000u) ? -(int32_t)(ua & 0x7fffu) : (int32_t)(ua & 0x7fffu);
    int32_t ib = (ub & 0x8000u) ? -(int32_t)(ub & 0x7fffu) : (int32_t)(ub & 0x7fffu);
    int32_t d = ia - ib;
    return (uint32_t)(d < 0 ? -d : d);
}

static err_metrics_t compute_metrics(const float *ref, const float *got) {
    err_metrics_t m = {0};
    double dot_all = 0.0, na_all = 0.0, nb_all = 0.0, err2 = 0.0;
    double dot_rp = 0.0, na_rp = 0.0, nb_rp = 0.0;
    for (uint32_t h = 0; h < 64u; h++) {
        for (uint32_t d = 0; d < 512u; d++) {
            const float a = ref[(size_t)h * 512u + d];
            const float b = got[(size_t)h * 512u + d];
            if (!isfinite(a) || !isfinite(b)) {
                m.non_finite++;
                continue;
            }
            const double diff = fabs((double)a - (double)b);
            const double denom = fabs((double)a) > 1e-1 ? fabs((double)a) : 1e-1;
            const double rel = diff / denom;
            if (diff > m.max_abs) m.max_abs = diff;
            if (rel > m.max_rel) {
                m.max_rel = rel;
                m.worst_ref = a;
                m.worst_got = b;
            }
            if (fabsf(a) >= 0.05f && fabsf(b) >= 0.05f && ((a > 0.0f) == (b > 0.0f))) {
                const uint32_t ulps = bf16_ulp_dist(a, b);
                if (ulps > m.max_bf16_ulps) m.max_bf16_ulps = ulps;
            }
            err2    += diff * diff;
            dot_all += (double)a * (double)b;
            na_all  += (double)a * (double)a;
            nb_all  += (double)b * (double)b;
            if (d >= 448u) {
                if (diff > m.rope_max_abs) m.rope_max_abs = diff;
                if (rel > m.rope_max_rel) m.rope_max_rel = rel;
                dot_rp += (double)a * (double)b;
                na_rp  += (double)a * (double)a;
                nb_rp  += (double)b * (double)b;
            }
        }
    }
    m.l2_rel = (na_all > 0.0) ? sqrt(err2 / na_all) : sqrt(err2);
    m.cosine = (na_all == 0.0 && nb_all == 0.0) ? 1.0 :
               (na_all > 0.0 && nb_all > 0.0) ? (dot_all / sqrt(na_all * nb_all)) : 0.0;
    m.rope_cosine = (na_rp == 0.0 && nb_rp == 0.0) ? 1.0 :
                    (na_rp > 0.0 && nb_rp > 0.0) ? (dot_rp / sqrt(na_rp * nb_rp)) : 0.0;
    return m;
}

typedef struct {
    const char *label;
    const char *kernel;
    uint32_t d_split;
    uint32_t nsg;
    bool fp16_p;
} variant_t;

enum { kNumBaseVariants = 6, kNumSplitKVariants = 4, kNumVariants = 10 };

static const variant_t kVariants[kNumVariants] = {
    {"D2-FP32P", "kernel_myds_attention_decode_mma_rope_fuse_f16_512_d2",     2u,  4u, false},
    {"D4-FP32P", "kernel_myds_attention_decode_mma_rope_fuse_f16_512_d4",     4u,  4u, false},
    {"D8-FP32P", "kernel_myds_attention_decode_mma_rope_fuse_f16_512_d8",     8u,  4u, false},
    {"D2-FP16P", "kernel_myds_attention_decode_mma_rope_fuse_f16_512_d2_p16", 2u,  4u, true},
    {"D4-FP16P", "kernel_myds_attention_decode_mma_rope_fuse_f16_512_d4_p16", 4u,  4u, true},
    {"D8-FP16P", "kernel_myds_attention_decode_mma_rope_fuse_f16_512_d8_p16", 8u,  4u, true},
    {"SK4-D8",   "kernel_myds_attention_decode_mma_sk4_f16_512_d8",           8u,  4u, false},
    {"SK8-D8",   "kernel_myds_attention_decode_mma_sk8_f16_512_d8",           8u,  8u, false},
    {"SK10-D8",  "kernel_myds_attention_decode_mma_sk10_f16_512_d8",          8u, 10u, false},
    {"SK16-D8",  "kernel_myds_attention_decode_mma_sk16_f16_512_d8",          8u, 16u, false},
};

static id<MTLComputePipelineState> build_base_pso(id<MTLDevice> dev, id<MTLLibrary> lib, bool has_kvpad) {
    MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
    bool t = true, f = false, kp = has_kvpad;
    int32_t i512 = 512, i1 = 1, i32 = 32;
    [fc setConstantValue:&t    type:MTLDataTypeBool atIndex:400];
    [fc setConstantValue:&t    type:MTLDataTypeBool atIndex:401];
    [fc setConstantValue:&f    type:MTLDataTypeBool atIndex:402];
    [fc setConstantValue:&f    type:MTLDataTypeBool atIndex:403];
    [fc setConstantValue:&kp   type:MTLDataTypeBool atIndex:404];
    [fc setConstantValue:&f    type:MTLDataTypeBool atIndex:405];
    [fc setConstantValue:&i512 type:MTLDataTypeInt  atIndex:420];
    [fc setConstantValue:&i512 type:MTLDataTypeInt  atIndex:421];
    [fc setConstantValue:&i1   type:MTLDataTypeInt  atIndex:422];
    [fc setConstantValue:&i32  type:MTLDataTypeInt  atIndex:423];
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_myds_attention_register_rope_fuse_f16_512"
                                   constantValues:fc error:&err];
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    return pso;
}

static id<MTLComputePipelineState> build_pad_pso(id<MTLDevice> dev, id<MTLLibrary> lib) {
    MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
    bool t = true;
    int32_t i32 = 32;
    [fc setConstantValue:&t   type:MTLDataTypeBool atIndex:100];
    [fc setConstantValue:&i32 type:MTLDataTypeInt  atIndex:125];
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                   constantValues:fc error:&err];
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    return pso;
}

static id<MTLComputePipelineState> build_plain_pso(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    return pso;
}

static void run_pad(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pad_pso,
                    id<MTLBuffer> kbuf, id<MTLBuffer> vbuf, id<MTLBuffer> mbuf,
                    id<MTLBuffer> padbuf, uint32_t keys) {
    if (keys % 32u == 0u) return;
    const uint64_t kvb = (uint64_t)keys * 1024u;
    const uint64_t mb = (uint64_t)keys * 2u;
    struct test_pad_args pa = {
        .ne11 = (int32_t)keys, .ne_12_2 = 1, .ne_12_3 = 1,
        .nb11 = 1024u, .nb12 = kvb, .nb13 = kvb,
        .nb21 = 1024u, .nb22 = kvb, .nb23 = kvb,
        .ne31 = 1, .ne32 = 1, .ne33 = 1,
        .nb31 = mb, .nb32 = mb, .nb33 = mb
    };
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pad_pso];
    [enc setBytes:&pa length:sizeof(pa) atIndex:0];
    [enc setBuffer:kbuf offset:0 atIndex:1];
    [enc setBuffer:vbuf offset:0 atIndex:2];
    [enc setBuffer:mbuf offset:0 atIndex:3];
    [enc setBuffer:padbuf offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
}

static void run_baseline(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> base_pso,
                         const struct test_rope_fuse_args *args,
                         id<MTLBuffer> qbuf, id<MTLBuffer> kbuf, id<MTLBuffer> vbuf,
                         id<MTLBuffer> mbuf, id<MTLBuffer> sbuf, id<MTLBuffer> padbuf,
                         id<MTLBuffer> dstbuf) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:base_pso];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:qbuf offset:0 atIndex:1];
    [enc setBuffer:kbuf offset:0 atIndex:2];
    [enc setBuffer:vbuf offset:0 atIndex:3];
    [enc setBuffer:mbuf offset:0 atIndex:4];
    [enc setBuffer:sbuf offset:0 atIndex:5];
    [enc setBuffer:padbuf offset:0 atIndex:6];
    [enc setBuffer:dstbuf offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:22400 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(448, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
}

static void run_mma(id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> mma_pso, uint32_t d_split, uint32_t nsg,
                    const struct test_rope_fuse_args *args,
                    id<MTLBuffer> qbuf, id<MTLBuffer> kbuf, id<MTLBuffer> vbuf,
                    id<MTLBuffer> mbuf, id<MTLBuffer> sbuf, id<MTLBuffer> dstbuf) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:mma_pso];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:qbuf offset:0 atIndex:1];
    [enc setBuffer:kbuf offset:0 atIndex:2];
    [enc setBuffer:vbuf offset:0 atIndex:3];
    [enc setBuffer:mbuf offset:0 atIndex:4];
    [enc setBuffer:sbuf offset:0 atIndex:5];
    [enc setBuffer:dstbuf offset:0 atIndex:6];
    [enc setThreadgroupMemoryLength:10272 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(8, d_split, 1) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
}

/* 1. Fail-closed knob parsing & counter unit tests */
static void test_knob_parsing(void) {
    printf("[1/5] Testing MYDS_ATTN_DECODE_MMA fail-closed parsing...\n");
    unsetenv("MYDS_ATTN_DECODE_MMA");
    REQUIRE(myds_v41_gpu_attention_decode_mma_enabled() == 0);
    myds_v41_gpu_attention_decode_mma_counters_t cnt;
    myds_v41_gpu_attention_decode_mma_counters(&cnt, true);
    REQUIRE(cnt.d_split == 0u);

    setenv("MYDS_ATTN_DECODE_MMA", "0", 1);
    REQUIRE(myds_v41_gpu_attention_decode_mma_enabled() == 0);
    myds_v41_gpu_attention_decode_mma_counters(&cnt, false);
    REQUIRE(cnt.d_split == 0u);

    const char *valid_vals[] = {"1", "2", "4", "8"};
    const int expected_ret[] = {1, 2, 4, 8};
    const uint32_t expected_ds[] = {8u, 2u, 4u, 8u};
    for (size_t i = 0; i < 4; i++) {
        setenv("MYDS_ATTN_DECODE_MMA", valid_vals[i], 1);
        REQUIRE(myds_v41_gpu_attention_decode_mma_enabled() == expected_ret[i]);
        myds_v41_gpu_attention_decode_mma_counters(&cnt, false);
        REQUIRE(cnt.d_split == expected_ds[i]);
    }

    const char *bad_vals[] = {"", "3", "5", "6", "16", "-1", "01", "04", " 4", "4 ", "abc", "true"};
    for (size_t i = 0; i < sizeof(bad_vals) / sizeof(bad_vals[0]); i++) {
        setenv("MYDS_ATTN_DECODE_MMA", bad_vals[i], 1);
        REQUIRE(myds_v41_gpu_attention_decode_mma_enabled() == -1);
        REQUIRE(myds_v41_gpu_scratch_create(128u) == NULL);
    }

    unsetenv("MYDS_ATTN_DECODE_MMA");
    REQUIRE(myds_v41_gpu_attention_decode_mma_enabled() == 0);
    printf("  PASS: absent/0->0, 1->1(d_split=8), 2/4/8->2/4/8, 12 invalid values fail-closed (-1).\n");
}

/* 2. Print PSO residency for all variants */
static void inspect_psos(id<MTLDevice> dev, id<MTLLibrary> lib,
                         id<MTLComputePipelineState> __strong mma_psos[kNumVariants],
                         id<MTLComputePipelineState> base_pso) {
    printf("[2/5] PSO Residency on %s (maxTGMem=%lu B):\n",
           dev.name.UTF8String, (unsigned long)dev.maxThreadgroupMemoryLength);
    printf("  %-58s | maxThr | staticSmem | dynSmem | TGs/core\n", "Kernel");
    printf("  -----------------------------------------------------------+--------+------------+---------+---------\n");
    const NSUInteger max_mem = dev.maxThreadgroupMemoryLength;
    printf("  %-58s | %6lu | %10lu | %7u | %8lu\n",
           "kernel_myds_attention_register_rope_fuse_f16_512",
           (unsigned long)base_pso.maxTotalThreadsPerThreadgroup,
           (unsigned long)base_pso.staticThreadgroupMemoryLength,
           22400u,
           (unsigned long)(max_mem / (22400u + base_pso.staticThreadgroupMemoryLength)));

    id<MTLComputePipelineState> canon = build_plain_pso(dev, lib, "kernel_myds_attention_decode_mma_rope_fuse_f16_512");
    REQUIRE(canon.maxTotalThreadsPerThreadgroup >= 128u);
    REQUIRE(canon.staticThreadgroupMemoryLength == 0u);
    printf("  %-58s | %6lu | %10lu | %7u | %8lu\n",
           "kernel_myds_attention_decode_mma_rope_fuse_f16_512",
           (unsigned long)canon.maxTotalThreadsPerThreadgroup,
           (unsigned long)canon.staticThreadgroupMemoryLength,
           10272u,
           (unsigned long)(max_mem / (10272u + canon.staticThreadgroupMemoryLength)));

    for (int v = 0; v < kNumVariants; v++) {
        id<MTLComputePipelineState> p = mma_psos[v];
        REQUIRE(p.maxTotalThreadsPerThreadgroup >= 32u * kVariants[v].nsg);
        REQUIRE(p.threadExecutionWidth == 32u);
        REQUIRE(p.staticThreadgroupMemoryLength + 10272u <= max_mem);
        NSUInteger tgs = max_mem / (10272u + p.staticThreadgroupMemoryLength);
        REQUIRE(tgs >= 3u);
        printf("  %-58s | %6lu | %10lu | %7u | %8lu\n",
               kVariants[v].kernel,
               (unsigned long)p.maxTotalThreadsPerThreadgroup,
               (unsigned long)p.staticThreadgroupMemoryLength,
               10272u,
               (unsigned long)tgs);
    }
}

/* 3. Accuracy tests across keys, RoPE dims, masks, sinks, extreme/equal logits, NaN tail */
static void test_accuracy_suite(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                                id<MTLComputePipelineState> pad_pso,
                                id<MTLComputePipelineState> base_pso_nopad,
                                id<MTLComputePipelineState> base_pso_pad,
                                id<MTLComputePipelineState> __strong mma_psos[kNumVariants]) {
    printf("[3/5] Running numerical accuracy & edge-case suite vs register_rope_fuse...\n");

    const uint32_t keys_list[] = {1, 7, 15, 16, 17, 31, 32, 64, 127, 128, 129, 192, 256, 512, 639, 640};
    const uint32_t pos_list[]  = {0, 1, 127, 847, 16384};

    id<MTLBuffer> qbuf   = [dev newBufferWithLength:64u * 512u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> kbuf   = [dev newBufferWithLength:640u * 512u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> vbuf   = [dev newBufferWithLength:640u * 512u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> mbuf   = [dev newBufferWithLength:640u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> sbuf   = [dev newBufferWithLength:64u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> padbuf = [dev newBufferWithLength:(2u * 32u * 512u * 2u + 32u * 2u) options:MTLResourceStorageModeShared];
    id<MTLBuffer> refbuf = [dev newBufferWithLength:64u * 512u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> outbuf = [dev newBufferWithLength:64u * 512u * 4u options:MTLResourceStorageModeShared];

    double worst_abs[kNumVariants] = {0}, worst_rel[kNumVariants] = {0}, worst_l2[kNumVariants] = {0};
    double min_cos[kNumVariants], worst_rp_abs[kNumVariants] = {0}, worst_rp_rel[kNumVariants] = {0}, min_rp_cos[kNumVariants];
    uint32_t worst_ulps[kNumVariants] = {0};
    for (int v = 0; v < kNumVariants; v++) { min_cos[v] = 1.0; min_rp_cos[v] = 1.0; }

    /* 3a. Sweep all required key counts + stale-tail NaN/Inf poisoning + RoPE dims */
    for (size_t ki = 0; ki < sizeof(keys_list) / sizeof(keys_list[0]); ki++) {
        const uint32_t keys = keys_list[ki];
        const uint32_t pos = pos_list[ki % 5];
        const bool compressed = (ki & 1u) != 0;

        float *qp = (float *)qbuf.contents;
        uint16_t *kp = (uint16_t *)kbuf.contents;
        uint16_t *vp = (uint16_t *)vbuf.contents;
        uint16_t *mp = (uint16_t *)mbuf.contents;
        float *sp = (float *)sbuf.contents;

        for (uint32_t i = 0; i < 64u * 512u; i++) qp[i] = rand_f(-1.5f, 1.5f);
        for (uint32_t i = 0; i < keys * 512u; i++) {
            kp[i] = f32_to_f16_bits(rand_f(-1.5f, 1.5f));
            vp[i] = f32_to_f16_bits(rand_f(-1.5f, 1.5f));
        }
        /* Poison rows [keys, 640) with half NaN (0x7e00) and +Inf (0x7c00) */
        for (uint32_t i = keys * 512u; i < 640u * 512u; i++) {
            kp[i] = (i & 1u) ? 0x7e00u : 0x7c00u;
            vp[i] = (i & 1u) ? 0x7e00u : 0x7c00u;
        }
        for (uint32_t i = 0; i < keys; i++) mp[i] = 0u;
        for (uint32_t i = keys; i < 640u; i++) mp[i] = 0x7e00u;
        for (uint32_t h = 0; h < 64u; h++) sp[h] = ((int)h - 31) / 8.0f;

        struct test_rope_fuse_args fargs = make_fused_args(keys, pos, compressed);
        run_pad(queue, pad_pso, kbuf, vbuf, mbuf, padbuf, keys);
        run_baseline(queue, (keys % 32u) ? base_pso_pad : base_pso_nopad,
                     &fargs, qbuf, kbuf, vbuf, mbuf, sbuf, padbuf, refbuf);

        /* Also test exact-sized unpadded K/V buffers on tail sizes */
        id<MTLBuffer> exact_k = (keys % 64u != 0)
            ? [dev newBufferWithBytes:kp length:(size_t)keys * 1024u options:MTLResourceStorageModeShared]
            : kbuf;
        id<MTLBuffer> exact_v = (keys % 64u != 0)
            ? [dev newBufferWithBytes:vp length:(size_t)keys * 1024u options:MTLResourceStorageModeShared]
            : vbuf;
        id<MTLBuffer> exact_m = (keys % 64u != 0)
            ? [dev newBufferWithBytes:mp length:(size_t)keys * 2u options:MTLResourceStorageModeShared]
            : mbuf;

        for (int v = 0; v < kNumVariants; v++) {
            memset(outbuf.contents, 0xa5, 64u * 512u * 4u);
            run_mma(queue, mma_psos[v], kVariants[v].d_split, kVariants[v].nsg, &fargs,
                    qbuf, kbuf, vbuf, mbuf, sbuf, outbuf);
            err_metrics_t m = compute_metrics((const float *)refbuf.contents, (const float *)outbuf.contents);
            REQUIRE(m.non_finite == 0u);
            if (m.max_abs > worst_abs[v]) worst_abs[v] = m.max_abs;
            if (m.max_rel > worst_rel[v]) worst_rel[v] = m.max_rel;
            if (m.l2_rel > worst_l2[v]) worst_l2[v] = m.l2_rel;
            if (m.max_bf16_ulps > worst_ulps[v]) worst_ulps[v] = m.max_bf16_ulps;
            if (m.cosine < min_cos[v]) min_cos[v] = m.cosine;
            if (m.rope_max_abs > worst_rp_abs[v]) worst_rp_abs[v] = m.rope_max_abs;
            if (m.rope_max_rel > worst_rp_rel[v]) worst_rp_rel[v] = m.rope_max_rel;
            if (m.rope_cosine < min_rp_cos[v]) min_rp_cos[v] = m.rope_cosine;

            /* Epilogue rounds to BF16 (1 ULP = 1/128 = 0.0078 relative).
             * Gate strictly on L2 relative error (< 2e-3), cosine (> 0.99999), and BF16 ULP <= 2! */
            const uint32_t ulp_gate = kVariants[v].fp16_p ? 8u : 2u;
            if (m.l2_rel > 2e-3 || m.cosine < 0.99999 || m.rope_cosine < 0.99999 || m.max_bf16_ulps > ulp_gate) {
                fprintf(stderr, "FAIL %s keys=%u pos=%u: max_abs=%.3e max_rel=%.3e l2_rel=%.3e ulps=%u cos=%.8f rope_cos=%.8f (ref=%.6f got=%.6f)\n",
                        kVariants[v].label, keys, pos, m.max_abs, m.max_rel, m.l2_rel, m.max_bf16_ulps,
                        m.cosine, m.rope_cosine, m.worst_ref, m.worst_got);
                exit(1);
            }

            /* Verify exact unpadded buffers match poisoned-tail buffers bit-for-bit */
            if (keys % 64u != 0) {
                id<MTLBuffer> out2 = [dev newBufferWithLength:64u * 512u * 4u options:MTLResourceStorageModeShared];
                run_mma(queue, mma_psos[v], kVariants[v].d_split, kVariants[v].nsg, &fargs,
                        qbuf, exact_k, exact_v, exact_m, sbuf, out2);
                REQUIRE(memcmp(outbuf.contents, out2.contents, 64u * 512u * 4u) == 0);
            }
        }
    }

    printf("  Sweep keys={1,7,15,16,17,31,32,64,127,128,129,192,256,512,639,640} + NaN-tail + exact-bounds + RoPE(448..511):\n");
    for (int v = 0; v < kNumVariants; v++) {
        printf("    %-8s | max_abs=%.3e max_rel=%.3e l2_rel=%.3e max_bf16_ulps=%u min_cos=%.8f | RoPE: max_abs=%.3e min_cos=%.8f\n",
               kVariants[v].label, worst_abs[v], worst_rel[v], worst_l2[v], worst_ulps[v],
               min_cos[v], worst_rp_abs[v], min_rp_cos[v]);
    }

    /* 3b. Torture cases: extreme logits (+/-500), all-equal logits, all-masked, pattern mask,
     *     disabled sinks (-FLT_MAX/2), and -INFINITY mask entries! */
    const char *case_names[] = {
        "extreme_asc", "extreme_desc", "all_equal_pos", "all_equal_zero",
        "all_masked_finite_sink", "all_masked_neg_sink", "pattern_mask", "causal_prefix_mask", "neginf_mask"
    };
    for (int tc = 0; tc < 9; tc++) {
        const uint32_t keys = (tc == 0 || tc == 1) ? 193u : 131u;
        float *qp = (float *)qbuf.contents;
        uint16_t *kp = (uint16_t *)kbuf.contents;
        uint16_t *vp = (uint16_t *)vbuf.contents;
        uint16_t *mp = (uint16_t *)mbuf.contents;
        float *sp = (float *)sbuf.contents;

        for (uint32_t i = 0; i < 64u * 512u; i++) qp[i] = rand_f(-1.0f, 1.0f);
        for (uint32_t i = 0; i < keys * 512u; i++) {
            kp[i] = f32_to_f16_bits(rand_f(-1.0f, 1.0f));
            vp[i] = f32_to_f16_bits(rand_f(-2.0f, 2.0f));
        }
        for (uint32_t i = 0; i < keys; i++) mp[i] = 0u;
        for (uint32_t h = 0; h < 64u; h++) sp[h] = ((int)h - 31) / 8.0f;

        if (tc == 0 || tc == 1) {
            /* Extreme logits across tiles: +/- 450 scale */
            for (uint32_t h = 0; h < 64u; h++) {
                for (uint32_t d = 0; d < 512u; d++) qp[h * 512u + d] = 1.0f;
            }
            for (uint32_t r = 0; r < keys; r++) {
                float kval = (tc == 0)
                    ? (-20.0f + 40.0f * ((float)r / (float)keys))   /* -450 .. +450 after /sqrt(512) */
                    : ( 20.0f - 40.0f * ((float)r / (float)keys));  /* +450 .. -450 */
                uint16_t kh = f32_to_f16_bits(kval);
                for (uint32_t d = 0; d < 512u; d++) kp[r * 512u + d] = kh;
            }
        } else if (tc == 2 || tc == 3) {
            /* All-equal logits (+100 or 0) */
            float qv = (tc == 2) ? 2.0f : 0.0f;
            uint16_t kv = f32_to_f16_bits((tc == 2) ? 2.2f : 0.0f);
            for (uint32_t i = 0; i < 64u * 512u; i++) qp[i] = qv;
            for (uint32_t i = 0; i < keys * 512u; i++) kp[i] = kv;
        } else if (tc == 4) {
            for (uint32_t i = 0; i < keys; i++) mp[i] = 0xfbffu; /* -MAXHALF */
        } else if (tc == 5) {
            for (uint32_t i = 0; i < keys; i++) mp[i] = 0xfbffu;
            for (uint32_t h = 0; h < 64u; h++) sp[h] = -FLT_MAX / 2.0f;
        } else if (tc == 6) {
            for (uint32_t i = 0; i < keys; i++) mp[i] = (i % 3u == 1u) ? 0xfbffu : 0xbc00u; /* -MAXHALF / -1 */
        } else if (tc == 7) {
            for (uint32_t i = 0; i < keys; i++) mp[i] = (i < 64u) ? 0xfbffu : 0u; /* entire tile 0 masked */
            for (uint32_t h = 0; h < 64u; h++) sp[h] = -FLT_MAX / 2.0f;
        } else if (tc == 8) {
            for (uint32_t i = 0; i < keys; i++) mp[i] = (i % 2u == 0u) ? 0xfc00u : 0u; /* -INFINITY half */
        }

        struct test_rope_fuse_args fargs = make_fused_args(keys, 321u, tc % 2 == 1);
        run_pad(queue, pad_pso, kbuf, vbuf, mbuf, padbuf, keys);
        run_baseline(queue, (keys % 32u) ? base_pso_pad : base_pso_nopad,
                     &fargs, qbuf, kbuf, vbuf, mbuf, sbuf, padbuf, refbuf);

        for (int v = 0; v < kNumVariants; v++) {
            run_mma(queue, mma_psos[v], kVariants[v].d_split, kVariants[v].nsg, &fargs,
                    qbuf, kbuf, vbuf, mbuf, sbuf, outbuf);
            err_metrics_t m = compute_metrics((const float *)refbuf.contents, (const float *)outbuf.contents);
            REQUIRE(m.non_finite == 0u);
            if (tc == 4 || tc == 5) {
                /* All masked -> exact zeros! */
                REQUIRE(m.max_abs == 0.0);
            } else {
                const uint32_t ulp_gate = kVariants[v].fp16_p ? 8u : 2u;
                if (m.l2_rel > 2e-3 || m.cosine < 0.99999 || m.max_bf16_ulps > ulp_gate) {
                    fprintf(stderr, "FAIL torture=%s %s: max_abs=%.3e max_rel=%.3e l2_rel=%.3e ulps=%u cos=%.8f (ref=%.6f got=%.6f)\n",
                            case_names[tc], kVariants[v].label, m.max_abs, m.max_rel, m.l2_rel, m.max_bf16_ulps,
                            m.cosine, m.worst_ref, m.worst_got);
                    exit(1);
                }
            }
        }
    }
    printf("  PASS: All 9 torture cases (extreme logits, equal logits, all-masked, -INF mask, sinks) passed!\n");
}

/* 4. End-to-end C runtime dispatch & /debug/state counters verification */
static void test_c_runtime_integration(void) {
    printf("[4/5] Testing C runtime wiring (direct + staged, RF=1 & RF=0, trace bypass, counters)...\n");
    REQUIRE(myds_metal_init("bin"));
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(2048u);
    REQUIRE(s != NULL);

    myds_metal_buffer_t *q      = myds_metal_buffer_alloc(64u * 512u * 4u);
    myds_metal_buffer_t *win    = myds_metal_buffer_alloc(128u * 512u * 4u);
    myds_metal_buffer_t *comp   = myds_metal_buffer_alloc(600u * 512u * 2u);
    myds_metal_buffer_t *sel_f  = myds_metal_buffer_alloc(512u * 512u * 4u);
    myds_metal_buffer_t *ids    = myds_metal_buffer_alloc(512u * 4u);
    myds_metal_buffer_t *sinks  = myds_metal_buffer_alloc(64u * 4u);
    myds_metal_buffer_t *out_ref = myds_metal_buffer_alloc(64u * 512u * 4u);
    myds_metal_buffer_t *out_mma = myds_metal_buffer_alloc(64u * 512u * 4u);
    REQUIRE(q && win && comp && sel_f && ids && sinks && out_ref && out_mma);

    float *qp = (float *)myds_metal_buffer_data(q);
    float *wp = (float *)myds_metal_buffer_data(win);
    uint16_t *cp = (uint16_t *)myds_metal_buffer_data(comp);
    float *sfp = (float *)myds_metal_buffer_data(sel_f);
    int32_t *idp = (int32_t *)myds_metal_buffer_data(ids);
    float *sp = (float *)myds_metal_buffer_data(sinks);

    for (uint32_t i = 0; i < 64u * 512u; i++) qp[i] = rand_f(-1.0f, 1.0f);
    for (uint32_t i = 0; i < 128u * 512u; i++) wp[i] = rand_f(-1.0f, 1.0f);
    for (uint32_t i = 0; i < 600u * 512u; i++) cp[i] = f32_to_bf16_bits(rand_f(-1.0f, 1.0f));
    for (uint32_t r = 0; r < 512u; r++) {
        idp[r] = (int32_t)((r * 37u + 11u) % 600u);
        for (uint32_t d = 0; d < 512u; d++) {
            uint32_t bits = (uint32_t)cp[(size_t)idp[r] * 512u + d] << 16u;
            memcpy(&sfp[(size_t)r * 512u + d], &bits, 4);
        }
    }
    for (uint32_t h = 0; h < 64u; h++) sp[h] = ((int)h - 31) / 8.0f;

    /* Golden reference via MYDS_ATTN_PACKED_REGISTER=1, MYDS_ATTN_ROPE_FUSE=1, MYDS_ATTN_DECODE_MMA=0 */
    setenv("MYDS_ATTN_PACKED_REGISTER", "1", 1);
    setenv("MYDS_ATTN_ROPE_FUSE", "1", 1);
    setenv("MYDS_ATTN_DECODE_MMA", "0", 1);

    const uint32_t test_nraw[] = {7u, 128u, 128u, 128u, 127u, 128u};
    const uint32_t test_sel[]  = {0u,   0u,  63u, 111u, 512u, 512u}; /* keys = 7, 128, 191, 239, 639, 640 */
    const uint32_t test_pos[]  = {6u, 127u, 190u, 513u, 4094u, 4095u};

    for (size_t ci = 0; ci < sizeof(test_nraw) / sizeof(test_nraw[0]); ci++) {
        bool ref_consumed = false;
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_attention_direct_rope_fuse(
            out_ref, sinks, q, win, test_nraw[ci], 13u, comp, ids, 600u, test_sel[ci],
            s, test_pos[ci], (ci & 1u) != 0, true, false, &ref_consumed));
        REQUIRE(myds_metal_commit_batch());
        REQUIRE(ref_consumed);

        /* Test all modes {1, 2, 4, 8} under RF=1 (production config) and RF=0 (standalone) */
        const char *modes[] = {"1", "2", "4", "8"};
        const uint32_t exp_ds[] = {8u, 2u, 4u, 8u};
        for (int rf = 1; rf >= 0; rf--) {
            setenv("MYDS_ATTN_ROPE_FUSE", rf ? "1" : "0", 1);
            for (size_t mi = 0; mi < 4; mi++) {
                setenv("MYDS_ATTN_DECODE_MMA", modes[mi], 1);
                myds_v41_gpu_attention_decode_mma_counters(NULL, true);

                /* Direct path */
                bool consumed = false;
                REQUIRE(myds_metal_begin_batch());
                REQUIRE(myds_v41_gpu_attention_direct_rope_fuse(
                    out_mma, sinks, q, win, test_nraw[ci], 13u, comp, ids, 600u, test_sel[ci],
                    s, test_pos[ci], (ci & 1u) != 0, true, false, &consumed));
                REQUIRE(myds_metal_commit_batch());
                REQUIRE(consumed);

                myds_v41_gpu_attention_decode_mma_counters_t cnt;
                myds_v41_gpu_attention_decode_mma_counters(&cnt, false);
                REQUIRE(cnt.requests == 1u);
                REQUIRE(cnt.encoded == 1u);
                REQUIRE(cnt.fallbacks == 0u);
                REQUIRE(cnt.d_split == exp_ds[mi]);

                err_metrics_t m = compute_metrics((const float *)myds_metal_buffer_data(out_ref),
                                                  (const float *)myds_metal_buffer_data(out_mma));
                REQUIRE(m.non_finite == 0u && m.l2_rel < 2e-3 && m.cosine > 0.99999 && m.rope_cosine > 0.99999);

                /* Staged path */
                consumed = false;
                REQUIRE(myds_metal_begin_batch());
                REQUIRE(myds_v41_gpu_attention_rope_fuse(
                    out_mma, sinks, q, win, test_nraw[ci], 13u, sel_f, test_sel[ci],
                    s, test_pos[ci], (ci & 1u) != 0, true, false, &consumed));
                REQUIRE(myds_metal_commit_batch());
                REQUIRE(consumed);

                myds_v41_gpu_attention_decode_mma_counters(&cnt, false);
                REQUIRE(cnt.requests == 2u && cnt.encoded == 2u && cnt.fallbacks == 0u);
                m = compute_metrics((const float *)myds_metal_buffer_data(out_ref),
                                    (const float *)myds_metal_buffer_data(out_mma));
                REQUIRE(m.non_finite == 0u && m.l2_rel < 2e-3 && m.cosine > 0.99999);
            }
        }
    }

    /* Verify trace_on=true bypasses MMA (0 requests, rope_consumed=false) */
    setenv("MYDS_ATTN_ROPE_FUSE", "1", 1);
    setenv("MYDS_ATTN_DECODE_MMA", "4", 1);
    myds_v41_gpu_attention_decode_mma_counters(NULL, true);
    bool trace_consumed = true;
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_attention_direct_rope_fuse(
        out_mma, sinks, q, win, 128u, 0u, comp, ids, 600u, 512u,
        s, 1024u, false, true, true, &trace_consumed));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(!trace_consumed);
    myds_v41_gpu_attention_decode_mma_counters_t tcnt;
    myds_v41_gpu_attention_decode_mma_counters(&tcnt, false);
    REQUIRE(tcnt.requests == 0u && tcnt.encoded == 0u);

    unsetenv("MYDS_ATTN_DECODE_MMA");
    unsetenv("MYDS_ATTN_ROPE_FUSE");
    unsetenv("MYDS_ATTN_PACKED_REGISTER");
    myds_metal_buffer_free(q);
    myds_metal_buffer_free(win);
    myds_metal_buffer_free(comp);
    myds_metal_buffer_free(sel_f);
    myds_metal_buffer_free(ids);
    myds_metal_buffer_free(sinks);
    myds_metal_buffer_free(out_ref);
    myds_metal_buffer_free(out_mma);
    myds_v41_gpu_scratch_free(s);
    myds_metal_cleanup();
    printf("  PASS: C runtime direct/staged + RF={0,1} + modes={1,2,4,8} + trace bypass + counters verified.\n");
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

/* 5. Batched 40-layer-in-1-CB GPU microbenchmark (Serial-K + Single-Pass In-TG Split-K) */
static void run_m4_benchmark(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                             id<MTLComputePipelineState> pad_pso,
                             id<MTLComputePipelineState> base_pso_nopad,
                             id<MTLComputePipelineState> base_pso_pad,
                             id<MTLComputePipelineState> __strong mma_psos[kNumVariants]) {
    printf("\n[5/5] Decode Attention Microbenchmark (40 layers batched in 1 CB, median of 7 runs, us/layer):\n");
    printf("  Device: %s\n", dev.name.UTF8String);
    printf("  %-6s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | Best\n",
           "keys", "Base(us)", "D2-F32P", "D4-F32P", "D8-F32P", "D8-F16P",
           "SK4-D8", "SK8-D8", "SK10-D8", "SK16-D8", "D2-F16P");
    printf("  -------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------------------\n");

    const uint32_t bench_keys[] = {1, 7, 16, 31, 32, 64, 128, 192, 256, 512, 639, 640};
    enum { kBenchLayers = 40, kBenchRuns = 7 };
    const int n_layers = kBenchLayers;
    const int n_runs = kBenchRuns;

    id<MTLBuffer> qbuf   = [dev newBufferWithLength:64u * 512u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> kbuf   = [dev newBufferWithLength:640u * 512u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> mbuf   = [dev newBufferWithLength:640u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> sbuf   = [dev newBufferWithLength:64u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> padbuf = [dev newBufferWithLength:(2u * 32u * 512u * 2u + 32u * 2u) options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstbuf = [dev newBufferWithLength:64u * 512u * 4u options:MTLResourceStorageModeShared];

    float *qp = (float *)qbuf.contents;
    uint16_t *kp = (uint16_t *)kbuf.contents;
    float *sp = (float *)sbuf.contents;
    for (uint32_t i = 0; i < 64u * 512u; i++) qp[i] = rand_f(-1.0f, 1.0f);
    for (uint32_t i = 0; i < 640u * 512u; i++) kp[i] = f32_to_f16_bits(rand_f(-1.0f, 1.0f));
    memset(mbuf.contents, 0, 640u * 2u);
    for (uint32_t h = 0; h < 64u; h++) sp[h] = ((int)h - 31) / 8.0f;

    /* Warmup GPU P-state before benchmarking to prevent cold-start DVFS jitter. */
    {
        struct test_rope_fuse_args wargs = make_fused_args(640u, 1024u, false);
        for (int w = 0; w < 8; w++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:mma_psos[kNumVariants - 1]];
            [enc setBytes:&wargs length:sizeof(wargs) atIndex:0];
            [enc setBuffer:qbuf offset:0 atIndex:1];
            [enc setBuffer:kbuf offset:0 atIndex:2];
            [enc setBuffer:kbuf offset:0 atIndex:3];
            [enc setBuffer:mbuf offset:0 atIndex:4];
            [enc setBuffer:sbuf offset:0 atIndex:5];
            [enc setBuffer:dstbuf offset:0 atIndex:6];
            [enc setThreadgroupMemoryLength:10272 atIndex:0];
            for (int l = 0; l < n_layers; l++) {
                [enc dispatchThreadgroups:MTLSizeMake(8, kVariants[kNumVariants - 1].d_split, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, kVariants[kNumVariants - 1].nsg, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
        }
    }

    for (size_t ki = 0; ki < sizeof(bench_keys) / sizeof(bench_keys[0]); ki++) {
        const uint32_t keys = bench_keys[ki];
        struct test_rope_fuse_args fargs = make_fused_args(keys, 1024u, false);
        run_pad(queue, pad_pso, kbuf, kbuf, mbuf, padbuf, keys);

        id<MTLComputePipelineState> bpso = (keys % 32u) ? base_pso_pad : base_pso_nopad;

        double base_samples[kBenchRuns];
        for (int run = -2; run < n_runs; run++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:bpso];
            [enc setBytes:&fargs length:sizeof(fargs) atIndex:0];
            [enc setBuffer:qbuf offset:0 atIndex:1];
            [enc setBuffer:kbuf offset:0 atIndex:2];
            [enc setBuffer:kbuf offset:0 atIndex:3];
            [enc setBuffer:mbuf offset:0 atIndex:4];
            [enc setBuffer:sbuf offset:0 atIndex:5];
            [enc setBuffer:padbuf offset:0 atIndex:6];
            [enc setBuffer:dstbuf offset:0 atIndex:7];
            [enc setThreadgroupMemoryLength:22400 atIndex:0];
            for (int l = 0; l < n_layers; l++) {
                [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(448, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
            if (run >= 0) {
                base_samples[run] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / (double)n_layers;
            }
        }
        qsort(base_samples, n_runs, sizeof(double), cmp_double);
        const double base_us = base_samples[n_runs / 2];

        double var_us[kNumVariants];
        int best_idx = -1;
        double best_us = base_us;
        for (int v = 0; v < kNumVariants; v++) {
            double samples[kBenchRuns];
            for (int run = -2; run < n_runs; run++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:mma_psos[v]];
                [enc setBytes:&fargs length:sizeof(fargs) atIndex:0];
                [enc setBuffer:qbuf offset:0 atIndex:1];
                [enc setBuffer:kbuf offset:0 atIndex:2];
                [enc setBuffer:kbuf offset:0 atIndex:3];
                [enc setBuffer:mbuf offset:0 atIndex:4];
                [enc setBuffer:sbuf offset:0 atIndex:5];
                [enc setBuffer:dstbuf offset:0 atIndex:6];
                [enc setThreadgroupMemoryLength:10272 atIndex:0];
                for (int l = 0; l < n_layers; l++) {
                    [enc dispatchThreadgroups:MTLSizeMake(8, kVariants[v].d_split, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, kVariants[v].nsg, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
                if (run >= 0) {
                    samples[run] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / (double)n_layers;
                }
            }
            qsort(samples, n_runs, sizeof(double), cmp_double);
            var_us[v] = samples[n_runs / 2];
            if (var_us[v] < best_us) {
                best_us = var_us[v];
                best_idx = v;
            }
        }

        const char *best_name = (best_idx >= 0) ? kVariants[best_idx].label : "Baseline";
        const double speedup = base_us / best_us;
        printf("  %-6u | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %-8s (%.2fx)\n",
               keys, base_us,
               var_us[0], var_us[1], var_us[2], var_us[5],
               var_us[6], var_us[7], var_us[8], var_us[9], var_us[3],
               best_name, speedup);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        bool do_bench = false;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--bench") == 0) do_bench = true;
        }

        init_test_rope_freqs();

        /* Step 1: Knob parsing (no GPU init required) */
        test_knob_parsing();

        /* Direct Metal setup for PSO inspection, exhaustive accuracy, and 40L CB benchmark */
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        REQUIRE(dev != nil);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"bin/myds.metallib"] error:&err];
        REQUIRE(lib != nil);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        REQUIRE(queue != nil);

        id<MTLComputePipelineState> pad_pso = build_pad_pso(dev, lib);
        id<MTLComputePipelineState> base_pso_nopad = build_base_pso(dev, lib, false);
        id<MTLComputePipelineState> base_pso_pad   = build_base_pso(dev, lib, true);
        id<MTLComputePipelineState> mma_psos[kNumVariants];
        for (int v = 0; v < kNumVariants; v++) {
            mma_psos[v] = build_plain_pso(dev, lib, kVariants[v].kernel);
        }

        inspect_psos(dev, lib, mma_psos, base_pso_nopad);
        test_accuracy_suite(dev, queue, pad_pso, base_pso_nopad, base_pso_pad, mma_psos);
        test_c_runtime_integration();

        if (do_bench) {
            run_m4_benchmark(dev, queue, pad_pso, base_pso_nopad, base_pso_pad, mma_psos);
        } else {
            printf("[5/5] Skipping M4 benchmark (pass --bench to run 40-layer-batched microbench).\n");
        }

        printf("ALL [D-A1] DECODE MMA ATTENTION TESTS PASSED!\n");
        return 0;
    }
}
