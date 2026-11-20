/* tests/test_kv_comp_nvfp4.m
 *
 * Apple Silicon GPU 0-ULP mathematical bijection, adversarial quantization,
 * Prefill & Decode attention consumer parity, strict env parsing, PSO
 * occupancy parity, lowkv SSD dual-hash isolation + round-trip, and
 * microbenchmark suite for [CAND-KV1] MYDS_KV_COMP_NVFP4
 * (4.5-bit E2M1FN + E4M3FN, 288 B/row lossless Compressed KV Cache).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "myds_debugstate.h"
#include "myds_internal.h"
#include "myds_kvdsk.h"
#include "myds_metal.h"
#include "myds_v41.h"
#include "myds_v41_gpu.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint32_t f32_bits(float f) {
    uint32_t u = 0;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static float bits_f32(uint32_t u) {
    float f = 0.0f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint16_t cpu_f32_to_f16_bits(float f) {
    __fp16 h = (__fp16)f;
    uint16_t u = 0;
    memcpy(&u, &h, sizeof(u));
    return u;
}

static float cpu_f16_bits_to_f32(uint16_t u) {
    __fp16 h = 0;
    memcpy(&h, &u, sizeof(h));
    return (float)h;
}

static float cpu_e4m3fn_to_f32(uint32_t s) {
    s &= 0x7fu;
    if (s < 8u) return (float)s * (1.0f / 512.0f);
    const int e = (int)(s >> 3u) - 7;
    const uint32_t m = s & 7u;
    return ldexpf(1.0f + (float)m * 0.125f, e);
}

static float cpu_nvfp4_dequant_f32(uint32_t code4, uint32_t sc8) {
    static const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint32_t sign_u32 = (code4 & 8u) << 28u;
    const float mag = mags[code4 & 7u];
    const float sc = cpu_e4m3fn_to_f32(sc8);
    const float prod = mag * sc;
    return bits_f32(f32_bits(prod) | sign_u32);
}

/* Kernel arg structs mirroring metal/dsv41.metal and metal/myds_chunk.metal */
typedef struct {
    uint32_t source_rows;
    uint32_t selected_rows;
    uint32_t dst_row_offset;
    uint32_t pad;
} test_get_rows_f16_args_t;

typedef struct {
    uint32_t n_raw;
    uint32_t raw_start;
    uint32_t source_rows;
    uint32_t selected_rows;
} test_direct_pack_args_t;

typedef struct {
    uint32_t rows;
    uint32_t top_k;
    uint32_t n_comp;
    uint32_t valid_from;
    uint32_t mask_stride;
    uint32_t c_start;
    uint32_t base_pos;
    uint32_t ratio;
} test_gather_comp_args_t;

static int dispatch_get_rows_f32(bool nvfp4, myds_metal_buffer_t *dst_f32,
    const myds_metal_buffer_t *src, const myds_metal_buffer_t *ids,
    uint32_t source_rows, uint32_t selected_rows) {
    const uint32_t args[2] = { source_rows, selected_rows };
    const myds_metal_buffer_t *bufs[3] = { src, ids, dst_f32 };
    if (nvfp4) {
        const uint32_t groups[3] = { 1u, selected_rows, 1u };
        const uint32_t threads[3] = { 128u, 1u, 1u };
        return myds_metal_dispatch_kernel("kernel_dsv41_get_rows_nvfp4", args, sizeof(args), bufs, 3, groups, threads, 0);
    }
    const uint32_t groups[3] = { 2u, selected_rows, 1u };
    const uint32_t threads[3] = { 256u, 1u, 1u };
    return myds_metal_dispatch_kernel("kernel_dsv41_get_rows_bf16", args, sizeof(args), bufs, 3, groups, threads, 0);
}

static int dispatch_stage_bf16_f16(myds_metal_buffer_t *dst_f16, const myds_metal_buffer_t *src_bf16, uint32_t n_rows) {
    const uint32_t count = n_rows * 512u;
    const myds_metal_buffer_t *bufs[2] = { src_bf16, dst_f16 };
    const uint32_t groups[3] = { (count + 255u) / 256u, 1u, 1u };
    const uint32_t threads[3] = { 256u, 1u, 1u };
    return myds_metal_dispatch_kernel("kernel_dsv41_stage_bf16_f16", &count, sizeof(count), bufs, 2, groups, threads, 0);
}

static int dispatch_stage_nvfp4_f16(myds_metal_buffer_t *dst_f16, const myds_metal_buffer_t *src_nvfp4, uint32_t n_rows) {
    const myds_metal_buffer_t *bufs[2] = { src_nvfp4, dst_f16 };
    const uint32_t groups[3] = { 1u, n_rows, 1u };
    const uint32_t threads[3] = { 128u, 1u, 1u };
    return myds_metal_dispatch_kernel("kernel_dsv41_stage_nvfp4_f16", &n_rows, sizeof(n_rows), bufs, 2, groups, threads, 0);
}

static int dispatch_get_rows_to_f16(const char *kernel, myds_metal_buffer_t *dst_f16,
    const myds_metal_buffer_t *src, const myds_metal_buffer_t *ids,
    uint32_t source_rows, uint32_t selected_rows, uint32_t dst_row_offset) {
    const test_get_rows_f16_args_t args = { source_rows, selected_rows, dst_row_offset, 0u };
    const myds_metal_buffer_t *bufs[3] = { src, ids, dst_f16 };
    const uint32_t groups[3] = { 1u, selected_rows, 1u };
    const uint32_t threads[3] = { 128u, 1u, 1u };
    return myds_metal_dispatch_kernel(kernel, &args, sizeof(args), bufs, 3, groups, threads, 0);
}

static int dispatch_direct_pack_fuse(const char *kernel, myds_metal_buffer_t *dst_f16,
    const myds_metal_buffer_t *window, const myds_metal_buffer_t *src,
    const myds_metal_buffer_t *ids, uint32_t n_raw, uint32_t raw_start,
    uint32_t source_rows, uint32_t selected_rows) {
    const test_direct_pack_args_t args = { n_raw, raw_start, source_rows, selected_rows };
    const myds_metal_buffer_t *bufs[4] = { window, src, ids, dst_f16 };
    const uint32_t groups[3] = { 1u, n_raw + selected_rows, 1u };
    const uint32_t threads[3] = { 128u, 1u, 1u };
    return myds_metal_dispatch_kernel(kernel, &args, sizeof(args), bufs, 4, groups, threads, 0);
}

static int dispatch_gather_comp(const char *kernel, myds_metal_buffer_t *dst_f16,
    myds_metal_buffer_t *mask, const myds_metal_buffer_t *comp,
    const myds_metal_buffer_t *ids, uint32_t rows, uint32_t top_k, uint32_t n_comp,
    uint32_t base_pos, uint32_t c_start, uint32_t ratio) {
    const test_gather_comp_args_t args = { rows, top_k, n_comp, 0u, 640u, c_start, base_pos, ratio };
    const myds_metal_buffer_t *bufs[4] = { dst_f16, mask, comp, ids };
    const uint32_t groups[3] = { top_k, rows, 1u };
    const uint32_t threads[3] = { 128u, 1u, 1u };
    return myds_metal_dispatch_kernel(kernel, &args, sizeof(args), bufs, 4, groups, threads, 0);
}

/* 1. Strict MYDS_KV_COMP_NVFP4 environment parsing contract (default MUST be 0, fail-closed) */
static void test_env_parsing(void) {
    const char *saved = getenv("MYDS_KV_COMP_NVFP4");
    char saved_buf[64] = {0};
    const bool had_saved = (saved != NULL);
    if (had_saved) snprintf(saved_buf, sizeof(saved_buf), "%s", saved);

    unsetenv("MYDS_KV_COMP_NVFP4");
    CHECK(myds_v41_gpu_kv_comp_nvfp4_enabled() == 0);
    CHECK(myds_v41_gpu_kv_comp_cache_mode() == MYDS_KV_COMP_CACHE_BF16);
    CHECK(myds_v41_gpu_kv_comp_row_bytes() == 1024u);

    setenv("MYDS_KV_COMP_NVFP4", "0", 1);
    CHECK(myds_v41_gpu_kv_comp_nvfp4_enabled() == 0);
    CHECK(myds_v41_gpu_kv_comp_cache_mode() == MYDS_KV_COMP_CACHE_BF16);
    CHECK(myds_v41_gpu_kv_comp_row_bytes() == 1024u);

    setenv("MYDS_KV_COMP_NVFP4", "1", 1);
    CHECK(myds_v41_gpu_kv_comp_nvfp4_enabled() == 1);
    CHECK(myds_v41_gpu_kv_comp_cache_mode() == MYDS_KV_COMP_CACHE_NVFP4);
    CHECK(myds_v41_gpu_kv_comp_row_bytes() == 288u);
    CHECK((MYDS_KV_COMP_NVFP4_ROW_BYTES % 32u) == 0u);

    CHECK(myds_v41_kv_comp_cache_mode_row_bytes(MYDS_KV_COMP_CACHE_F32) == 2048u);
    CHECK(myds_v41_kv_comp_cache_mode_row_bytes(MYDS_KV_COMP_CACHE_BF16) == 1024u);
    CHECK(myds_v41_kv_comp_cache_mode_row_bytes(MYDS_KV_COMP_CACHE_NVFP4) == 288u);
    CHECK(myds_v41_kv_comp_cache_mode_row_bytes(99u) == 0u);

    const char *bad[] = {"", "2", "-1", "01", "true", "false", " 1", "1 "};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        setenv("MYDS_KV_COMP_NVFP4", bad[i], 1);
        CHECK(myds_v41_gpu_kv_comp_nvfp4_enabled() == -1);
        CHECK(myds_v41_gpu_scratch_create(1024u) == NULL);
    }

    /* Check /debug/state metadata reflects kv_comp_cache for both BF16 and NVFP4 */
    char state[16384];
    setenv("MYDS_KV_COMP_NVFP4", "0", 1);
    myds_debug_state_json(state, sizeof(state), NULL);
    CHECK(strstr(state, "\"kv_comp_cache\":{\"mode\":2,\"format\":\"BF16\",\"row_bytes\":1024}") != NULL);

    setenv("MYDS_KV_COMP_NVFP4", "1", 1);
    myds_debug_state_json(state, sizeof(state), NULL);
    CHECK(strstr(state, "\"kv_comp_cache\":{\"mode\":3,\"format\":\"NVFP4-E2M1+E4M3\",\"row_bytes\":288}") != NULL);

    if (had_saved) setenv("MYDS_KV_COMP_NVFP4", saved_buf, 1);
    else unsetenv("MYDS_KV_COMP_NVFP4");
    printf("  [PASS] strict MYDS_KV_COMP_NVFP4 env parsing (default=0, fail-closed), row-byte invariants & /debug/state metadata\n");
}

/* 2. PSO occupancy & static threadgroup memory parity across all 8 NVFP4 kernels */
static void verify_pso_occupancy(void) {
    @autoreleasepool {
        const char *lib_path = myds_metal_loaded_library_path();
        CHECK(lib_path != NULL && lib_path[0] != '\0');
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        CHECK(dev != nil);
        NSError *err = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:lib_path]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        CHECK(lib != nil && err == nil);

        struct {
            const char *name;
            const char *bf16_peer;
            NSUInteger min_max_threads;
            NSUInteger max_static_smem;
        } specs[] = {
            {"kernel_dsv41_quant_pack_kv_nvfp4",            NULL,                                    1024u, 16u},
            {"kernel_dsv41_kv_rope_quant_batch",            NULL,                                    1024u,  0u},
            {"kernel_dsv41_stage_nvfp4_f16",                "kernel_dsv41_stage_bf16_f16",           1024u,  0u},
            {"kernel_dsv41_get_rows_nvfp4",                 "kernel_dsv41_get_rows_bf16",            1024u,  0u},
            {"kernel_dsv41_get_rows_nvfp4_to_f16",          "kernel_dsv41_get_rows_bf16_to_f16",     1024u,  0u},
            {"kernel_myds_dsv41_direct_pack_fuse_nvfp4",    "kernel_myds_dsv41_direct_pack_fuse",    1024u,  0u},
            {"kernel_myds_attn_gather_comp_nvfp4",          "kernel_myds_attn_gather_comp_bf16",     1024u,  0u},
            {"kernel_myds_attn_gather_comp_512_nvfp4",      "kernel_myds_attn_gather_comp_512_bf16", 1024u,  0u},
        };

        printf("  Verifying NVFP4 Metal PSO occupancy & static threadgroup memory:\n");
        for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); ++i) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:specs[i].name]];
            CHECK(fn != nil);
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            CHECK(pso != nil && err == nil);
            NSUInteger peer_threads = 0u;
            if (specs[i].bf16_peer) {
                id<MTLFunction> pfn = [lib newFunctionWithName:[NSString stringWithUTF8String:specs[i].bf16_peer]];
                CHECK(pfn != nil);
                id<MTLComputePipelineState> ppso = [dev newComputePipelineStateWithFunction:pfn error:&err];
                CHECK(ppso != nil && err == nil);
                peer_threads = ppso.maxTotalThreadsPerThreadgroup;
                CHECK(pso.maxTotalThreadsPerThreadgroup >= peer_threads);
            }
            CHECK(pso.threadExecutionWidth == 32u);
            CHECK(pso.maxTotalThreadsPerThreadgroup >= specs[i].min_max_threads);
            CHECK(pso.staticThreadgroupMemoryLength <= specs[i].max_static_smem);
            printf("    %-46s width=%lu max_threads=%4lu (bf16=%4lu) static_smem=%2lu B\n",
                   specs[i].name,
                   (unsigned long)pso.threadExecutionWidth,
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)peer_threads,
                   (unsigned long)pso.staticThreadgroupMemoryLength);
        }
    }
}

/* 3. Exhaustive 127 x 16 = 2,032 valid (code4, e4m3_sc8) states:
 *    verify mathematical 0-ULP bijection in BOTH BF16/FP32 AND FP16 across ALL GPU reader kernels! */
static void test_exhaustive_2032_states(void) {
    /* 127 scales x 16 codes = 2032 states = 127 blocks of 16 elements.
     * Pack into rows 0..3 (128 blocks; block 127 is zero padding); row 4 is spare so n_sel=5 <= n_rows. */
    const uint32_t n_rows = 5u;
    const uint32_t n_sel = 5u; /* 4 valid rows (0..3) + 1 out-of-range (-1) row */
    myds_metal_buffer_t *nvfp4_buf = myds_metal_buffer_alloc((size_t)n_rows * 288u);
    myds_metal_buffer_t *bf16_buf  = myds_metal_buffer_alloc((size_t)n_rows * 1024u);
    myds_metal_buffer_t *ids_buf   = myds_metal_buffer_alloc((size_t)n_sel * sizeof(int32_t));
    myds_metal_buffer_t *win_buf   = myds_metal_buffer_alloc(128u * 512u * sizeof(float));
    myds_metal_buffer_t *f32_nv    = myds_metal_buffer_alloc((size_t)n_sel * 512u * sizeof(float));
    myds_metal_buffer_t *f32_bf    = myds_metal_buffer_alloc((size_t)n_sel * 512u * sizeof(float));
    myds_metal_buffer_t *f16_nv    = myds_metal_buffer_alloc((size_t)(2u + n_sel) * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *f16_bf    = myds_metal_buffer_alloc((size_t)(2u + n_sel) * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *gath_nv   = myds_metal_buffer_alloc(640u * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *gath_bf   = myds_metal_buffer_alloc(640u * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *mask_nv   = myds_metal_buffer_alloc(640u * sizeof(uint16_t));
    myds_metal_buffer_t *mask_bf   = myds_metal_buffer_alloc(640u * sizeof(uint16_t));
    CHECK(nvfp4_buf && bf16_buf && ids_buf && win_buf && f32_nv && f32_bf &&
          f16_nv && f16_bf && gath_nv && gath_bf && mask_nv && mask_bf);

    uint8_t  *nv_bytes = (uint8_t  *)myds_metal_buffer_data(nvfp4_buf);
    uint16_t *bf_u16   = (uint16_t *)myds_metal_buffer_data(bf16_buf);
    memset(nv_bytes, 0, (size_t)n_rows * 288u);
    memset(bf_u16,   0, (size_t)n_rows * 1024u);

    float    ref_f32[4 * 512];
    uint16_t ref_f16[4 * 512];
    memset(ref_f32, 0, sizeof(ref_f32));
    memset(ref_f16, 0, sizeof(ref_f16));

    float min_pos = FLT_MAX;
    float max_val = 0.0f;
    uint32_t neg_zero_count = 0u;

    for (uint32_t s = 0; s < 127u; ++s) {
        const uint32_t row = s / 32u;
        const uint32_t blk = s % 32u;
        uint8_t *row288 = nv_bytes + (size_t)row * 288u;
        row288[256u + blk] = (uint8_t)s;
        for (uint32_t lane = 0; lane < 16u; ++lane) {
            const uint32_t code4 = lane;
            const uint32_t col = blk * 16u + lane;
            if ((lane & 1u) == 0u) row288[col >> 1u] = (uint8_t)code4;
            else                   row288[col >> 1u] |= (uint8_t)(code4 << 4u);

            const float v = cpu_nvfp4_dequant_f32(code4, s);
            const uint32_t bits = f32_bits(v);
            /* Every valid NVFP4 value must be finite, have zero lower 16 bits (exact BF16),
             * and convert losslessly to normal/zero FP16 (zero subnormals, zero Inf/NaN)! */
            CHECK(isfinite(v));
            CHECK((bits & 0xffffu) == 0u);
            const uint16_t hbits = cpu_f32_to_f16_bits(v);
            const uint16_t hexp = (uint16_t)((hbits >> 10u) & 0x1fu);
            const uint16_t hman = (uint16_t)(hbits & 0x3ffu);
            CHECK(hexp != 31u);                 /* no Inf/NaN */
            CHECK(!(hexp == 0u && hman != 0u)); /* no FP16 subnormals */
            CHECK(f32_bits(cpu_f16_bits_to_f32(hbits)) == bits); /* exact FP16 round-trip incl. -0.0 */

            if (v > 0.0f) {
                if (v < min_pos) min_pos = v;
                if (v > max_val) max_val = v;
            }
            if (code4 == 8u) {
                CHECK(bits == 0x80000000u && hbits == 0x8000u);
                neg_zero_count++;
            }

            ref_f32[row * 512u + col] = v;
            ref_f16[row * 512u + col] = hbits;
            bf_u16[row * 512u + col]  = (uint16_t)(bits >> 16u);
        }
    }
    CHECK(min_pos == (1.0f / 1024.0f));
    CHECK(max_val == 2688.0f);
    CHECK(neg_zero_count == 127u);

    int32_t *ids = (int32_t *)myds_metal_buffer_data(ids_buf);
    ids[0] = 3; ids[1] = 0; ids[2] = 2; ids[3] = 1; ids[4] = -1;

    float *win = (float *)myds_metal_buffer_data(win_buf);
    for (uint32_t i = 0; i < 128u * 512u; ++i) win[i] = (float)((int)(i % 33u) - 16) * 0.25f;

    /* 3A. FP32/BF16 reader: myds_v41_gpu_gather_mode (kernel_dsv41_get_rows_nvfp4 vs _bf16) */
    CHECK(myds_v41_gpu_gather_mode(f32_bf, bf16_buf, ids_buf, n_rows, n_sel, MYDS_KV_COMP_CACHE_BF16));
    CHECK(myds_v41_gpu_gather_mode(f32_nv, nvfp4_buf, ids_buf, n_rows, n_sel, MYDS_KV_COMP_CACHE_NVFP4));
    CHECK(memcmp(myds_metal_buffer_data(f32_bf), myds_metal_buffer_data(f32_nv), (size_t)n_sel * 512u * sizeof(float)) == 0);
    for (uint32_t i = 0; i < 4u; ++i) {
        const float *got = (const float *)myds_metal_buffer_data(f32_nv) + (size_t)i * 512u;
        CHECK(memcmp(got, ref_f32 + (size_t)ids[i] * 512u, 512u * sizeof(float)) == 0);
    }

    /* 3B. FP16 slab reader: kernel_dsv41_stage_nvfp4_f16 vs _bf16_f16 */
    CHECK(dispatch_stage_bf16_f16(f16_bf, bf16_buf, 4u));
    CHECK(dispatch_stage_nvfp4_f16(f16_nv, nvfp4_buf, 4u));
    CHECK(memcmp(myds_metal_buffer_data(f16_bf), myds_metal_buffer_data(f16_nv), 4u * 512u * sizeof(uint16_t)) == 0);
    CHECK(memcmp(myds_metal_buffer_data(f16_nv), ref_f16, 4u * 512u * sizeof(uint16_t)) == 0);

    /* 3C. FP16 decode direct reader: kernel_dsv41_get_rows_nvfp4_to_f16 vs _bf16_to_f16 */
    memset(myds_metal_buffer_data(f16_bf), 0x5a, (size_t)(2u + n_sel) * 512u * sizeof(uint16_t));
    memset(myds_metal_buffer_data(f16_nv), 0x5a, (size_t)(2u + n_sel) * 512u * sizeof(uint16_t));
    CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_bf16_to_f16", f16_bf, bf16_buf, ids_buf, n_rows, n_sel, 2u));
    CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_nvfp4_to_f16", f16_nv, nvfp4_buf, ids_buf, n_rows, n_sel, 2u));
    CHECK(memcmp(myds_metal_buffer_data(f16_bf), myds_metal_buffer_data(f16_nv), (size_t)(2u + n_sel) * 512u * sizeof(uint16_t)) == 0);

    /* 3D. FP16 decode direct pack fuse: kernel_myds_dsv41_direct_pack_fuse_nvfp4 vs BF16 */
    CHECK(dispatch_direct_pack_fuse("kernel_myds_dsv41_direct_pack_fuse", f16_bf, win_buf, bf16_buf, ids_buf, 2u, 127u, n_rows, n_sel));
    CHECK(dispatch_direct_pack_fuse("kernel_myds_dsv41_direct_pack_fuse_nvfp4", f16_nv, win_buf, nvfp4_buf, ids_buf, 2u, 127u, n_rows, n_sel));
    CHECK(memcmp(myds_metal_buffer_data(f16_bf), myds_metal_buffer_data(f16_nv), (size_t)(2u + n_sel) * 512u * sizeof(uint16_t)) == 0);

    /* 3E. FP16 prefill gather_comp & gather_comp_512 */
    memset(myds_metal_buffer_data(gath_bf), 0x33, 640u * 512u * sizeof(uint16_t));
    memset(myds_metal_buffer_data(gath_nv), 0x33, 640u * 512u * sizeof(uint16_t));
    memset(myds_metal_buffer_data(mask_bf), 0x00, 640u * sizeof(uint16_t));
    memset(myds_metal_buffer_data(mask_nv), 0x00, 640u * sizeof(uint16_t));
    CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_bf16", gath_bf, mask_bf, bf16_buf, ids_buf, 1u, n_sel, n_rows, 100u, 0u, 1u));
    CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_nvfp4", gath_nv, mask_nv, nvfp4_buf, ids_buf, 1u, n_sel, n_rows, 100u, 0u, 1u));
    CHECK(memcmp(myds_metal_buffer_data(gath_bf), myds_metal_buffer_data(gath_nv), 640u * 512u * sizeof(uint16_t)) == 0);
    CHECK(memcmp(myds_metal_buffer_data(mask_bf), myds_metal_buffer_data(mask_nv), 640u * sizeof(uint16_t)) == 0);

    memset(myds_metal_buffer_data(gath_bf), 0x77, 512u * 512u * sizeof(uint16_t));
    memset(myds_metal_buffer_data(gath_nv), 0x77, 512u * 512u * sizeof(uint16_t));
    CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_512_bf16", gath_bf, mask_bf, bf16_buf, ids_buf, 1u, n_sel, n_rows, 100u, 0u, 1u));
    CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_512_nvfp4", gath_nv, mask_nv, nvfp4_buf, ids_buf, 1u, n_sel, n_rows, 100u, 0u, 1u));
    CHECK(memcmp(myds_metal_buffer_data(gath_bf), myds_metal_buffer_data(gath_nv), 512u * 512u * sizeof(uint16_t)) == 0);
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint16_t *got = (const uint16_t *)myds_metal_buffer_data(gath_nv) + (size_t)i * 512u;
        CHECK(memcmp(got, ref_f16 + (size_t)ids[i] * 512u, 512u * sizeof(uint16_t)) == 0);
    }

    myds_metal_buffer_free(nvfp4_buf);
    myds_metal_buffer_free(bf16_buf);
    myds_metal_buffer_free(ids_buf);
    myds_metal_buffer_free(win_buf);
    myds_metal_buffer_free(f32_nv);
    myds_metal_buffer_free(f32_bf);
    myds_metal_buffer_free(f16_nv);
    myds_metal_buffer_free(f16_bf);
    myds_metal_buffer_free(gath_nv);
    myds_metal_buffer_free(gath_bf);
    myds_metal_buffer_free(mask_nv);
    myds_metal_buffer_free(mask_bf);
    printf("  [PASS] exhaustive 127 x 16 = 2,032 states: 0-ULP (memcmp==0) in BOTH BF16/FP32 & FP16 across all 6 GPU reader kernels (min_pos=2^-10, max=2688.0, 127x -0.0, 0 subnormals/Inf/NaN)\n");
}

/* 4. 50,000+ random & pathological rows through BOTH writers:
 *    - Prefill fused kv_rope_quant_batch(..., mode=3u, NVFP4) vs BF16
 *    - Decode fused quant_pack_kv_nvfp4 vs quant(..., mode=3u) + pack_bf16
 *    Including the subnormal E4M3FN 9/512 trap, -0.0f, exact E2M1 midpoint ties,
 *    overflow > 2688, underflow, and non-zero dst/src row offsets. */
static void test_50k_rows_both_writers(void) {
    const uint32_t n_rows = 53248u; /* > 50,000 rows = 27,262,976 elements */
    myds_metal_buffer_t *x_a     = myds_metal_buffer_alloc((size_t)n_rows * 512u * sizeof(float));
    myds_metal_buffer_t *x_b     = myds_metal_buffer_alloc((size_t)n_rows * 512u * sizeof(float));
    myds_metal_buffer_t *bf16_d  = myds_metal_buffer_alloc((size_t)n_rows * 1024u);
    myds_metal_buffer_t *nv_d    = myds_metal_buffer_alloc((size_t)n_rows * 288u);
    myds_metal_buffer_t *f16_bf  = myds_metal_buffer_alloc((size_t)n_rows * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *f16_nv  = myds_metal_buffer_alloc((size_t)n_rows * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *ids_all = myds_metal_buffer_alloc((size_t)n_rows * sizeof(int32_t));
    CHECK(x_a && x_b && bf16_d && nv_d && f16_bf && f16_nv && ids_all);

    float *pa = (float *)myds_metal_buffer_data(x_a);
    int32_t *ids = (int32_t *)myds_metal_buffer_data(ids_all);
    uint32_t rng = 0x243f6a88u;
    static const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

    for (uint32_t r = 0; r < n_rows; ++r) {
        ids[r] = (int32_t)r;
        float *row = pa + (size_t)r * 512u;
        const uint32_t bucket = r % 8u;
        if (bucket == 0u) {
            for (uint32_t i = 0; i < 512u; ++i) row[i] = ((r + i) & 1u) ? -0.0f : +0.0f;
        } else if (bucket == 1u) {
            /* Subnormal E4M3FN trap (s in 1..7, e.g. 9/512 = 0.017578125f) */
            for (uint32_t i = 0; i < 512u; ++i) {
                const float sgn = ((r + i) & 1u) ? -1.0f : 1.0f;
                const uint32_t k = 1u + (xorshift32(&rng) % 24u);
                row[i] = sgn * ((float)k / 512.0f);
            }
            row[0] = 9.0f / 512.0f;
            row[16] = -9.0f / 512.0f;
        } else if (bucket == 2u) {
            /* Tiny magnitudes down to underflow */
            for (uint32_t i = 0; i < 512u; ++i) {
                const float sgn = (xorshift32(&rng) & 1u) ? -1.0f : 1.0f;
                const float u = (float)(xorshift32(&rng) & 0xffffu) / 65535.0f;
                row[i] = sgn * u * 1.0e-5f;
            }
        } else if (bucket == 3u) {
            /* Overflow clamp > 2688.0 (amax/6 > 448.0) */
            for (uint32_t i = 0; i < 512u; ++i) {
                const float sgn = (xorshift32(&rng) & 1u) ? -1.0f : 1.0f;
                const float u = (float)(xorshift32(&rng) & 0xffffu) / 65535.0f;
                row[i] = sgn * u * 1.0e5f;
            }
        } else if (bucket == 4u) {
            /* Exact E2M1 midpoint ties across E4M3FN scales */
            for (uint32_t b = 0; b < 32u; ++b) {
                const float sc = cpu_e4m3fn_to_f32(1u + ((r * 7u + b * 3u) % 126u));
                for (uint32_t l = 0; l < 16u; ++l) {
                    const float sgn = ((l + b) & 1u) ? -1.0f : 1.0f;
                    row[b * 16u + l] = sgn * mids[(r + l) % 7u] * sc;
                }
            }
        } else if (bucket == 5u) {
            /* Wide per-16-element block dynamic range */
            for (uint32_t b = 0; b < 32u; ++b) {
                const int sh = (int)(xorshift32(&rng) % 22u) - 12;
                const float amp = ldexpf(1.0f, sh);
                for (uint32_t l = 0; l < 16u; ++l) {
                    const float u = ((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f;
                    row[b * 16u + l] = u * amp;
                }
            }
        } else {
            /* Normal activation range */
            for (uint32_t i = 0; i < 512u; ++i) {
                const float u = ((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f;
                row[i] = u * 4.0f;
            }
        }
    }
    memcpy(myds_metal_buffer_data(x_b), pa, (size_t)n_rows * 512u * sizeof(float));

    /* 4A. Writer 1: kv_rope_quant_batch_mode (mode=3u, stride=2, compressed=true) */
    CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(x_a, bf16_d, 0u, 100u, n_rows, 2u, true, 3u, MYDS_KV_COMP_CACHE_BF16));
    CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(x_b, nv_d,   0u, 100u, n_rows, 2u, true, 3u, MYDS_KV_COMP_CACHE_NVFP4));
    CHECK(memcmp(myds_metal_buffer_data(x_a), myds_metal_buffer_data(x_b), (size_t)n_rows * 512u * sizeof(float)) == 0);

    /* Verify FP32/BF16 GPU unpack across ALL 53,248 rows (reusing x_a/x_b as dst buffers!) */
    CHECK(dispatch_get_rows_f32(false, x_a, bf16_d, ids_all, n_rows, n_rows));
    CHECK(dispatch_get_rows_f32(true,  x_b, nv_d,   ids_all, n_rows, n_rows));
    CHECK(memcmp(myds_metal_buffer_data(x_a), myds_metal_buffer_data(x_b), (size_t)n_rows * 512u * sizeof(float)) == 0);

    /* Verify FP16 GPU unpack across all 53,248 rows */
    CHECK(dispatch_stage_bf16_f16(f16_bf, bf16_d, n_rows));
    CHECK(dispatch_stage_nvfp4_f16(f16_nv, nv_d, n_rows));
    CHECK(memcmp(myds_metal_buffer_data(f16_bf), myds_metal_buffer_data(f16_nv), (size_t)n_rows * 512u * sizeof(uint16_t)) == 0);

    /* 4B. Writer 2: quant_pack_kv_nvfp4 (non-RoPE, direct subnormal 9/512 + non-zero dst_row/src_row) */
    const uint32_t n_dec = 8192u;
    const size_t src_off = 37u;
    const size_t dst_off = 113u;
    for (uint32_t r = 0; r < n_dec + src_off; ++r) {
        float *row = pa + (size_t)r * 512u;
        const uint32_t bucket = r % 5u;
        if (bucket == 0u) {
            /* Direct 9/512 subnormal trap & all subnormal scales s in 1..7 */
            for (uint32_t b = 0; b < 32u; ++b) {
                for (uint32_t l = 0; l < 16u; ++l) {
                    const float sgn = ((r + b + l) & 1u) ? -1.0f : 1.0f;
                    row[b * 16u + l] = sgn * ((float)((b + l) % 19u) / 512.0f);
                }
                row[b * 16u] = ((r + b) & 1u) ? (-9.0f / 512.0f) : (9.0f / 512.0f);
            }
        } else if (bucket == 1u) {
            for (uint32_t i = 0; i < 512u; ++i) row[i] = ((r + i) & 1u) ? -0.0f : +0.0f;
        } else if (bucket == 2u) {
            for (uint32_t b = 0; b < 32u; ++b) {
                const float sc = cpu_e4m3fn_to_f32(1u + ((r + b) % 126u));
                for (uint32_t l = 0; l < 16u; ++l) {
                    row[b * 16u + l] = ((l & 1u) ? -1.0f : 1.0f) * mids[(r + l) % 7u] * sc;
                }
            }
        } else if (bucket == 3u) {
            for (uint32_t i = 0; i < 512u; ++i) {
                const float sgn = (xorshift32(&rng) & 1u) ? -1.0f : 1.0f;
                row[i] = sgn * (float)(xorshift32(&rng) & 0xffffu);
            }
        } else {
            for (uint32_t i = 0; i < 512u; ++i) {
                const float u = ((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f;
                row[i] = u * 6.0f;
            }
        }
    }
    memcpy(myds_metal_buffer_data(x_b), pa, (size_t)(n_dec + src_off) * 512u * sizeof(float));

    /* Reference BF16 path: quant(mode=3) on slice + pack_bf16 at dst_off */
    myds_metal_buffer_t *x_slice = myds_metal_buffer_alloc((size_t)n_dec * 512u * sizeof(float));
    CHECK(x_slice != NULL);
    memcpy(myds_metal_buffer_data(x_slice), pa + src_off * 512u, (size_t)n_dec * 512u * sizeof(float));
    CHECK(myds_v41_gpu_quant(x_slice, 512u, n_dec, 3u));
    CHECK(myds_v41_gpu_pack_bf16(bf16_d, dst_off * 512u, x_slice, 0u, n_dec * 512u));

    /* NVFP4 fused path with non-zero src_off & dst_off */
    CHECK(myds_v41_gpu_quant_pack_kv_nvfp4(nv_d, dst_off, x_b, src_off, n_dec));
    CHECK(memcmp(myds_metal_buffer_data(x_slice),
                 (const float *)myds_metal_buffer_data(x_b) + src_off * 512u,
                 (size_t)n_dec * 512u * sizeof(float)) == 0);

    /* Unpack slice [dst_off .. dst_off + n_dec) in FP32 AND FP16 -> 0-ULP */
    for (uint32_t i = 0; i < n_dec; ++i) ids[i] = (int32_t)(dst_off + i);
    CHECK(dispatch_get_rows_f32(false, x_a, bf16_d, ids_all, (uint32_t)(dst_off + n_dec), n_dec));
    CHECK(dispatch_get_rows_f32(true,  x_b, nv_d,   ids_all, (uint32_t)(dst_off + n_dec), n_dec));
    CHECK(memcmp(myds_metal_buffer_data(x_a), myds_metal_buffer_data(x_b), (size_t)n_dec * 512u * sizeof(float)) == 0);
    /* Also test myds_v41_gpu_gather_mode wrapper on top-512 slice */
    CHECK(myds_v41_gpu_gather_mode(x_a, bf16_d, ids_all, (uint32_t)(dst_off + n_dec), 512u, MYDS_KV_COMP_CACHE_BF16));
    CHECK(myds_v41_gpu_gather_mode(x_b, nv_d,   ids_all, (uint32_t)(dst_off + n_dec), 512u, MYDS_KV_COMP_CACHE_NVFP4));
    CHECK(memcmp(myds_metal_buffer_data(x_a), myds_metal_buffer_data(x_b), 512u * 512u * sizeof(float)) == 0);

    CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_bf16_to_f16", f16_bf, bf16_d, ids_all, (uint32_t)(dst_off + n_dec), n_dec, 0u));
    CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_nvfp4_to_f16", f16_nv, nv_d,   ids_all, (uint32_t)(dst_off + n_dec), n_dec, 0u));
    CHECK(memcmp(myds_metal_buffer_data(f16_bf), myds_metal_buffer_data(f16_nv), (size_t)n_dec * 512u * sizeof(uint16_t)) == 0);

    myds_metal_buffer_free(x_slice);
    myds_metal_buffer_free(x_a);
    myds_metal_buffer_free(x_b);
    myds_metal_buffer_free(bf16_d);
    myds_metal_buffer_free(nv_d);
    myds_metal_buffer_free(f16_bf);
    myds_metal_buffer_free(f16_nv);
    myds_metal_buffer_free(ids_all);
    printf("  [PASS] 53,248 prefill rows (27.3M elems) + 8,192 decode/subnormal-trap rows: 0-ULP (memcmp==0) in BOTH BF16/FP32 & FP16\n");
}

/* 5. End-to-end Prefill (gather_comp_512 fast-matrix, gather_comp matrix/vec, non-gather slab)
 *    and Decode (attention_direct, direct_rope_fuse, staged gather+attention) 0-ULP parity */
static void test_e2e_prefill_and_decode_attn_0ulp(bool pack_fuse_only) {
    const uint32_t n_comp = 4096u;
    const uint32_t c_len = 64u;
    const uint32_t top_k = 512u;
    const uint32_t base_pos = 1024u;

    myds_metal_buffer_t *raw_kv   = myds_metal_buffer_alloc((size_t)n_comp * 512u * sizeof(float));
    myds_metal_buffer_t *raw_kv2  = myds_metal_buffer_alloc((size_t)n_comp * 512u * sizeof(float));
    myds_metal_buffer_t *comp_bf  = myds_metal_buffer_alloc((size_t)n_comp * 1024u);
    myds_metal_buffer_t *comp_nv  = myds_metal_buffer_alloc((size_t)n_comp * 288u);
    myds_metal_buffer_t *q_buf    = myds_metal_buffer_alloc((size_t)c_len * 64u * 512u * sizeof(float));
    myds_metal_buffer_t *chunk_kv = myds_metal_buffer_alloc((size_t)c_len * 512u * sizeof(float));
    myds_metal_buffer_t *win_buf  = myds_metal_buffer_alloc(128u * 512u * sizeof(float));
    myds_metal_buffer_t *sinks    = myds_metal_buffer_alloc(64u * sizeof(float));
    myds_metal_buffer_t *ids_buf  = myds_metal_buffer_alloc((size_t)c_len * top_k * sizeof(int32_t));
    myds_metal_buffer_t *gathered = myds_metal_buffer_alloc((size_t)c_len * 640u * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *out_bf   = myds_metal_buffer_alloc((size_t)c_len * 64u * 512u * sizeof(float));
    myds_metal_buffer_t *out_nv   = myds_metal_buffer_alloc((size_t)c_len * 64u * 512u * sizeof(float));
    myds_metal_buffer_t *sel_f32  = myds_metal_buffer_alloc((size_t)top_k * 512u * sizeof(float));
    CHECK(raw_kv && raw_kv2 && comp_bf && comp_nv && q_buf && chunk_kv && win_buf &&
          sinks && ids_buf && gathered && out_bf && out_nv && sel_f32);

    uint32_t rng = 0x9e3779b9u;
    float *rkv = (float *)myds_metal_buffer_data(raw_kv);
    for (size_t i = 0; i < (size_t)n_comp * 512u; ++i) {
        rkv[i] = (((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f) * 2.0f;
    }
    memcpy(myds_metal_buffer_data(raw_kv2), rkv, (size_t)n_comp * 512u * sizeof(float));
    CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(raw_kv,  comp_bf, 0u, 0u, n_comp, 1u, true, 3u, MYDS_KV_COMP_CACHE_BF16));
    CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(raw_kv2, comp_nv, 0u, 0u, n_comp, 1u, true, 3u, MYDS_KV_COMP_CACHE_NVFP4));

    float *qp = (float *)myds_metal_buffer_data(q_buf);
    for (size_t i = 0; i < (size_t)c_len * 64u * 512u; ++i) {
        qp[i] = (((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f) * 0.5f;
    }
    float *ckp = (float *)myds_metal_buffer_data(chunk_kv);
    for (size_t i = 0; i < (size_t)c_len * 512u; ++i) {
        ckp[i] = (((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f) * 1.5f;
    }
    float *wp = (float *)myds_metal_buffer_data(win_buf);
    for (size_t i = 0; i < 128u * 512u; ++i) {
        wp[i] = (((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f) * 1.5f;
    }
    float *sp = (float *)myds_metal_buffer_data(sinks);
    for (uint32_t h = 0; h < 64u; ++h) sp[h] = -0.5f + 0.02f * (float)h;

    int32_t *ip = (int32_t *)myds_metal_buffer_data(ids_buf);
    for (uint32_t q = 0; q < c_len; ++q) {
        for (uint32_t j = 0; j < top_k; ++j) {
            /* Mix valid scattered indices with a few -1 entries at the tail */
            ip[(size_t)q * top_k + j] = (j >= 500u && (q & 1u)) ? -1 : (int32_t)(xorshift32(&rng) % n_comp);
        }
    }

    if (!pack_fuse_only) {
        /* 5A. Prefill: fast-matrix (gather_comp_512_nvfp4) + standard matrix (gather_comp_nvfp4) */
        setenv("MYDS_PREFILL_ATTN", "matrix", 1);
        setenv("MYDS_CHUNK_KEY_MAX", "16384", 1);
        myds_v41_gpu_scratch_t *s_mat = myds_v41_gpu_scratch_create(16384u);
        CHECK(s_mat != NULL);
        CHECK(myds_v41_gpu_carry_stage(s_mat, win_buf, base_pos));

        /* Fast matrix (MYDS_PREFILL_ATTN_FAST=1) -> gather_comp_512 + reuse cache invalidation across BF16->NVFP4 */
        setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
        myds_v41_gpu_prefill_attn_fast_counters_t fast_c0;
        myds_v41_gpu_prefill_attn_fast_counters(&fast_c0, true);
        for (uint32_t ratio = 1u; ratio <= 2u; ++ratio) {
            CHECK(myds_v41_gpu_carry_stage(s_mat, win_buf, base_pos));
            CHECK(myds_v41_gpu_attention_chunk_mode(out_bf, sinks, q_buf, chunk_kv, comp_bf, ids_buf, gathered,
                c_len, n_comp, top_k, base_pos, 0u, ratio, s_mat, MYDS_KV_COMP_CACHE_BF16));
            CHECK(myds_v41_gpu_carry_stage(s_mat, win_buf, base_pos));
            CHECK(myds_v41_gpu_attention_chunk_mode(out_nv, sinks, q_buf, chunk_kv, comp_nv, ids_buf, gathered,
                c_len, n_comp, top_k, base_pos, 0u, ratio, s_mat, MYDS_KV_COMP_CACHE_NVFP4));
            CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                         (size_t)c_len * 64u * 512u * sizeof(float)) == 0);
        }
        /* Second call with same comp_nv & ids_buf tests fast gather_comp cache HIT! */
        CHECK(myds_v41_gpu_carry_stage(s_mat, win_buf, base_pos));
        CHECK(myds_v41_gpu_attention_chunk_mode(out_nv, sinks, q_buf, chunk_kv, comp_nv, ids_buf, gathered,
            c_len, n_comp, top_k, base_pos, 0u, 2u, s_mat, MYDS_KV_COMP_CACHE_NVFP4));
        CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                     (size_t)c_len * 64u * 512u * sizeof(float)) == 0);
        myds_v41_gpu_prefill_attn_fast_counters_t fast_c1;
        myds_v41_gpu_prefill_attn_fast_counters(&fast_c1, false);
        CHECK(fast_c1.gather_comp_512_encoded >= 2u);
        CHECK(fast_c1.gather_comp_hits >= 1u);

        /* Standard gather_comp matrix (MYDS_PREFILL_ATTN_FAST=0) */
        setenv("MYDS_PREFILL_ATTN_FAST", "0", 1);
        CHECK(myds_v41_gpu_carry_stage(s_mat, win_buf, base_pos));
        CHECK(myds_v41_gpu_attention_chunk_mode(out_bf, sinks, q_buf, chunk_kv, comp_bf, ids_buf, gathered,
            c_len, n_comp, top_k, base_pos, 0u, 1u, s_mat, MYDS_KV_COMP_CACHE_BF16));
        CHECK(myds_v41_gpu_carry_stage(s_mat, win_buf, base_pos));
        CHECK(myds_v41_gpu_attention_chunk_mode(out_nv, sinks, q_buf, chunk_kv, comp_nv, ids_buf, gathered,
            c_len, n_comp, top_k, base_pos, 0u, 1u, s_mat, MYDS_KV_COMP_CACHE_NVFP4));
        CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                     (size_t)c_len * 64u * 512u * sizeof(float)) == 0);
        myds_v41_gpu_scratch_free(s_mat);

        /* 5B. Prefill vec gather_comp */
        setenv("MYDS_PREFILL_ATTN", "vec", 1);
        myds_v41_gpu_scratch_t *s_vec = myds_v41_gpu_scratch_create(16384u);
        CHECK(s_vec != NULL);
        CHECK(myds_v41_gpu_carry_stage(s_vec, win_buf, base_pos));
        CHECK(myds_v41_gpu_attention_chunk_mode(out_bf, sinks, q_buf, chunk_kv, comp_bf, ids_buf, gathered,
            c_len, n_comp, top_k, base_pos, 0u, 1u, s_vec, MYDS_KV_COMP_CACHE_BF16));
        CHECK(myds_v41_gpu_carry_stage(s_vec, win_buf, base_pos));
        CHECK(myds_v41_gpu_attention_chunk_mode(out_nv, sinks, q_buf, chunk_kv, comp_nv, ids_buf, gathered,
            c_len, n_comp, top_k, base_pos, 0u, 1u, s_vec, MYDS_KV_COMP_CACHE_NVFP4));
        CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                     (size_t)c_len * 64u * 512u * sizeof(float)) == 0);
        myds_v41_gpu_scratch_free(s_vec);

        /* 5C. Prefill non-gather slab path (stage_nvfp4_f16 vs stage_bf16_f16) */
        setenv("MYDS_ATTN_GATHER_MIN_KEYS", "16384", 1);
        setenv("MYDS_PREFILL_ATTN", "matrix", 1);
        myds_v41_gpu_scratch_t *s_slab = myds_v41_gpu_scratch_create(16384u);
        CHECK(s_slab != NULL);
        CHECK(myds_v41_gpu_carry_stage(s_slab, win_buf, base_pos));
        CHECK(myds_v41_gpu_attention_chunk_mode(out_bf, sinks, q_buf, chunk_kv, comp_bf, ids_buf, gathered,
            32u, 1024u, top_k, base_pos, 0u, 1u, s_slab, MYDS_KV_COMP_CACHE_BF16));
        CHECK(myds_v41_gpu_carry_stage(s_slab, win_buf, base_pos));
        CHECK(myds_v41_gpu_attention_chunk_mode(out_nv, sinks, q_buf, chunk_kv, comp_nv, ids_buf, gathered,
            32u, 1024u, top_k, base_pos, 0u, 1u, s_slab, MYDS_KV_COMP_CACHE_NVFP4));
        CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                     (size_t)32u * 64u * 512u * sizeof(float)) == 0);
        myds_v41_gpu_scratch_free(s_slab);
        unsetenv("MYDS_ATTN_GATHER_MIN_KEYS");
        unsetenv("MYDS_PREFILL_ATTN");
        unsetenv("MYDS_PREFILL_ATTN_FAST");
        unsetenv("MYDS_CHUNK_KEY_MAX");
    }

    /* 5D. Decode attention paths (direct, direct_rope_fuse, staged gather+attention) */
    myds_v41_gpu_scratch_t *s_dec = myds_v41_gpu_scratch_create(16384u);
    CHECK(s_dec != NULL);
    const uint32_t sel_counts[2] = { 128u, 512u };
    for (size_t si = 0; si < 2u; ++si) {
        const uint32_t sel = sel_counts[si];
        if (!pack_fuse_only) {
            /* Direct (batch2 / non-RF) */
            CHECK(myds_v41_gpu_attention_direct_mode(out_bf, sinks, q_buf, win_buf, 128u, 37u,
                comp_bf, ids_buf, n_comp, sel, s_dec, MYDS_KV_COMP_CACHE_BF16));
            CHECK(myds_v41_gpu_attention_direct_mode(out_nv, sinks, q_buf, win_buf, 128u, 37u,
                comp_nv, ids_buf, n_comp, sel, s_dec, MYDS_KV_COMP_CACHE_NVFP4));
            CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                         64u * 512u * sizeof(float)) == 0);

            /* Staged gather_mode + attention */
            CHECK(myds_v41_gpu_gather_mode(sel_f32, comp_bf, ids_buf, n_comp, sel, MYDS_KV_COMP_CACHE_BF16));
            CHECK(myds_v41_gpu_attention(out_bf, sinks, q_buf, win_buf, 128u, 37u, sel_f32, sel, s_dec));
            CHECK(myds_v41_gpu_gather_mode(sel_f32, comp_nv, ids_buf, n_comp, sel, MYDS_KV_COMP_CACHE_NVFP4));
            CHECK(myds_v41_gpu_attention(out_nv, sinks, q_buf, win_buf, 128u, 37u, sel_f32, sel, s_dec));
            CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                         64u * 512u * sizeof(float)) == 0);
        }

        /* Direct + RoPE fuse (exercises PACK_FUSE=0 get_rows_nvfp4_to_f16 OR PACK_FUSE=1 direct_pack_fuse_nvfp4) */
        bool rc_bf = false, rc_nv = false;
        myds_metal_decode_candidate_counters_t pf0, pf1;
        myds_v41_gpu_attention_direct_pack_fuse_counters(&pf0, true);
        CHECK(myds_v41_gpu_attention_direct_rope_fuse_mode(out_bf, sinks, q_buf, win_buf, 128u, 37u,
            comp_bf, ids_buf, n_comp, sel, s_dec, 1088u, true, true, false, &rc_bf, MYDS_KV_COMP_CACHE_BF16));
        CHECK(myds_v41_gpu_attention_direct_rope_fuse_mode(out_nv, sinks, q_buf, win_buf, 128u, 37u,
            comp_nv, ids_buf, n_comp, sel, s_dec, 1088u, true, true, false, &rc_nv, MYDS_KV_COMP_CACHE_NVFP4));
        CHECK(rc_bf == rc_nv);
        if (pack_fuse_only) {
            CHECK(rc_nv == true);
            myds_v41_gpu_attention_direct_pack_fuse_counters(&pf1, false);
            CHECK(pf1.encoded == 2u && pf1.failures == 0u);
        }
        CHECK(memcmp(myds_metal_buffer_data(out_bf), myds_metal_buffer_data(out_nv),
                     64u * 512u * sizeof(float)) == 0);
    }

    myds_v41_gpu_scratch_free(s_dec);
    myds_metal_buffer_free(raw_kv);
    myds_metal_buffer_free(raw_kv2);
    myds_metal_buffer_free(comp_bf);
    myds_metal_buffer_free(comp_nv);
    myds_metal_buffer_free(q_buf);
    myds_metal_buffer_free(chunk_kv);
    myds_metal_buffer_free(win_buf);
    myds_metal_buffer_free(sinks);
    myds_metal_buffer_free(ids_buf);
    myds_metal_buffer_free(gathered);
    myds_metal_buffer_free(out_bf);
    myds_metal_buffer_free(out_nv);
    myds_metal_buffer_free(sel_f32);
    if (pack_fuse_only) {
        printf("  [PASS] Decode direct_rope_fuse with MYDS_ATTN_DIRECT_PACK_FUSE=1 (direct_pack_fuse_nvfp4): 0-ULP (memcmp==0)\n");
    } else {
        printf("  [PASS] End-to-end Prefill (gather_comp_512 fast-matrix, gather_comp matrix/vec, slab) & Decode (direct, direct_rope_fuse, staged) attention: 0-ULP (memcmp==0)\n");
    }
}

/* 6. kvdisk dual-hash isolation (4 distinct model_identity hashes) + SSD 288 B/row round-trip */
static void test_lowkv_nvfp4_roundtrip_and_isolation(void) {
    const char *saved_kv  = getenv("MYDS_KV_COMP_NVFP4");
    const char *saved_idx = getenv("MYDS_INDEXER_KEY_FP4");
    char saved_kv_buf[64] = {0}, saved_idx_buf[64] = {0};
    const bool had_kv  = (saved_kv != NULL);
    const bool had_idx = (saved_idx != NULL);
    if (had_kv)  snprintf(saved_kv_buf,  sizeof(saved_kv_buf),  "%s", saved_kv);
    if (had_idx) snprintf(saved_idx_buf, sizeof(saved_idx_buf), "%s", saved_idx);

    /* 6A. Verify myds_v41_model_identity() yields 4 distinct non-zero hashes across (idx_fp4 x kv_nvfp4) */
    myds_model_t model = {0};
    gguf_file_t gf = {0};
    snprintf(model.arch, sizeof(model.arch), "deepseek41");
    model.gf = &gf;
    uint64_t ids[4];
    for (int i = 0; i < 4; ++i) {
        setenv("MYDS_INDEXER_KEY_FP4", (i & 1) ? "1" : "0", 1);
        setenv("MYDS_KV_COMP_NVFP4",   (i & 2) ? "1" : "0", 1);
        ids[i] = myds_v41_model_identity(&model);
        CHECK(ids[i] != 0ull);
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            CHECK(ids[i] != ids[j]);
        }
    }

    /* Verify end_stride shrinks under NVFP4=1 vs NVFP4=0 at default 65536 max_tokens */
    setenv("MYDS_INDEXER_KEY_FP4", "0", 1);
    setenv("MYDS_KV_COMP_NVFP4", "0", 1);
    const uint64_t end_stride_bf16 = myds_v41_end_stride_for(65536u);
    setenv("MYDS_KV_COMP_NVFP4", "1", 1);
    const uint64_t end_stride_nvfp4 = myds_v41_end_stride_for(65536u);
    CHECK(end_stride_nvfp4 < end_stride_bf16);

    /* 6B. SSD segment + prompt-end round-trip at 288 B/row + cross-format isolation vs BF16 */
    char dir[] = "/tmp/test_myds_nvfp4_lowkv_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char path[512];
    snprintf(path, sizeof(path), "%s/snap.bin", dir);

    const uint32_t ntok = 1600u; /* 3 pinned segments (512, 1024, 1536) + prompt-end tail at 1599 */
    int32_t *tokens = (int32_t *)malloc(ntok * sizeof(int32_t));
    CHECK(tokens != NULL);
    for (uint32_t i = 0; i < ntok; ++i) tokens[i] = (int32_t)(i * 43u + 13u);

    setenv("MYDS_KV_COMP_NVFP4", "1", 1);
    myds_v41_context_t *g1 = myds_v41_context_create_snapshot_fixture();
    CHECK(g1 != NULL);
    CHECK(myds_v41_low_kv_fixture_enable(g1, path, 0x87654321ull, 1, false, ntok + 64u));

    /* Verify compressed KV buffer sizes are 288 B/row (-71.88% vs 1024 B/row)! */
    CHECK(myds_metal_buffer_size(myds_v41_snapshot_get_compressed(g1, 3u)) == (ntok + 64u + 1u) * 288u);
    CHECK(myds_metal_buffer_size(myds_v41_snapshot_get_compressed(g1, 0u)) == (((ntok + 64u) / 2u) + 1u) * 288u);

    const uint32_t pin_pos[2] = {1024u, 1536u};
    for (uint32_t i = 0; i < 2u; ++i) {
        const uint32_t s = myds_snap_slot_for(pin_pos[i], 0);
        myds_metal_buffer_t *w = myds_v41_snapshot_get_snap_window(g1, s);
        float *wf = (float *)myds_metal_buffer_data(w);
        const size_t nw = myds_metal_buffer_size(w) / sizeof(float);
        for (size_t k = 0; k < nw; ++k) wf[k] = (float)((s * 997u + k) % 16384u) * 0.125f;
        myds_v41_snapshot_fixture_publish(g1, s, pin_pos[i]);
    }
    for (uint32_t l = 0; l < 40u; ++l) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g1, l));
        const size_t nw = myds_metal_buffer_size(myds_v41_snapshot_get_window(g1, l)) / sizeof(float);
        for (size_t k = 0; k < nw; ++k) w[k] = (float)((l * 503u + k) % 8192u) * 0.5f;
    }
    for (uint32_t o = 0; o < 4u; ++o) {
        uint8_t *c = (uint8_t *)myds_metal_buffer_data(myds_v41_snapshot_get_compressed(g1, o));
        const size_t nc = myds_metal_buffer_size(myds_v41_snapshot_get_compressed(g1, o));
        for (size_t k = 0; k < nc; ++k) c[k] = (uint8_t)((o * 97u + k * 23u + 11u) & 0xffu);

        uint8_t *x = (uint8_t *)myds_metal_buffer_data(myds_v41_snapshot_get_index_cache(g1, o));
        const size_t nx = myds_metal_buffer_size(myds_v41_snapshot_get_index_cache(g1, o));
        for (size_t k = 0; k < nx; ++k) x[k] = (uint8_t)((o * 59u + k * 17u + 3u) & 0xffu);
    }
    myds_v41_snapshot_set_engram_history(g1, 101u, 202u, 303u);
    CHECK(myds_v41_snapshot_store_now(g1, ntok - 1u));
    CHECK(myds_v41_low_kv_save(g1, tokens, ntok));
    myds_v41_context_free(g1);

    /* Reload in fresh MYDS_KV_COMP_NVFP4=1 context -> bit-exact restore of all 288 B/row bytes */
    myds_v41_context_t *g2 = myds_v41_context_create_snapshot_fixture();
    CHECK(g2 != NULL);
    CHECK(myds_v41_low_kv_fixture_enable(g2, path, 0x87654321ull, 1, false, ntok + 64u));
    CHECK(myds_v41_low_kv_load_deeper(g2, tokens, 0u, ntok) == ntok - 1u);
    CHECK(myds_v41_snapshot_restore(g2, ntok - 1u));
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t r = (o == 3u) ? 1u : 2u;
        const size_t valid_bytes = (size_t)((ntok - 1u) / r) * 288u;
        const uint8_t *c = (const uint8_t *)myds_metal_buffer_data(myds_v41_snapshot_get_compressed(g2, o));
        for (size_t k = 0; k < valid_bytes; ++k) {
            CHECK(c[k] == (uint8_t)((o * 97u + k * 23u + 11u) & 0xffu));
        }
    }
    myds_v41_context_free(g2);

    /* Cross-format isolation: opening same path under MYDS_KV_COMP_NVFP4=0 (BF16) must miss/reject */
    setenv("MYDS_KV_COMP_NVFP4", "0", 1);
    myds_v41_context_t *g3 = myds_v41_context_create_snapshot_fixture();
    CHECK(g3 != NULL);
    if (myds_v41_low_kv_fixture_enable(g3, path, 0x87654321ull, 1, false, ntok + 64u)) {
        CHECK(myds_v41_low_kv_probe(g3, tokens, ntok) == 0u);
    }
    myds_v41_context_free(g3);

    /* Cleanup temp directory */
    NSArray<NSString *> *items = [[NSFileManager defaultManager]
        contentsOfDirectoryAtPath:[NSString stringWithUTF8String:dir] error:nil];
    for (NSString *item in items) {
        NSString *full = [[NSString stringWithUTF8String:dir] stringByAppendingPathComponent:item];
        unlink([full UTF8String]);
    }
    rmdir(dir);
    free(tokens);

    if (had_kv)  setenv("MYDS_KV_COMP_NVFP4",   saved_kv_buf,  1); else unsetenv("MYDS_KV_COMP_NVFP4");
    if (had_idx) setenv("MYDS_INDEXER_KEY_FP4", saved_idx_buf, 1); else unsetenv("MYDS_INDEXER_KEY_FP4");
    printf("  [PASS] kvdisk 4-way dual-hash isolation (idx_fp4 x kv_nvfp4) & SSD 288 B/row round-trip vs BF16\n");
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da < db) ? -1 : (da > db ? 1 : 0);
}

/* 7. 7-run median Apple M4 microbenchmarks: BF16 (1024 B/row) vs NVFP4 (288 B/row, -71.88%) */
static void run_nvfp4_microbenchmarks(void) {
    printf("=== Running 7-run median GPU microbenchmarks: BF16 (1024 B/row) vs NVFP4 (288 B/row, -71.88%%) ===\n");
    const uint32_t max_nc = 262144u; /* 256 MiB BF16, 72 MiB NVFP4 */
    const uint32_t max_c  = 256u;
    const uint32_t top_k  = 512u;

    myds_metal_buffer_t *comp_bf  = myds_metal_buffer_alloc((size_t)max_nc * 1024u);
    myds_metal_buffer_t *comp_nv  = myds_metal_buffer_alloc((size_t)max_nc * 288u);
    myds_metal_buffer_t *raw_w    = myds_metal_buffer_alloc(4096u * 512u * sizeof(float));
    myds_metal_buffer_t *raw_w2   = myds_metal_buffer_alloc(4096u * 512u * sizeof(float));
    myds_metal_buffer_t *ids_buf  = myds_metal_buffer_alloc((size_t)max_c * top_k * sizeof(int32_t));
    myds_metal_buffer_t *gath_bf  = myds_metal_buffer_alloc((size_t)max_c * top_k * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *gath_nv  = myds_metal_buffer_alloc((size_t)max_c * top_k * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *mask_dum = myds_metal_buffer_alloc((size_t)max_c * 640u * sizeof(uint16_t));
    myds_metal_buffer_t *win_buf  = myds_metal_buffer_alloc(128u * 512u * sizeof(float));
    myds_metal_buffer_t *dec_f16  = myds_metal_buffer_alloc(640u * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *dec_f16b = myds_metal_buffer_alloc(640u * 512u * sizeof(uint16_t));
    CHECK(comp_bf && comp_nv && raw_w && raw_w2 && ids_buf && gath_bf && gath_nv &&
          mask_dum && win_buf && dec_f16 && dec_f16b);

    uint32_t rng = 0x517cc1b7u;
    /* Populate 4096 realistic quantized rows and tile them across max_nc (saves startup time while exercising full 256K footprint) */
    float *rwp = (float *)myds_metal_buffer_data(raw_w);
    for (size_t i = 0; i < 4096u * 512u; ++i) {
        rwp[i] = ((float)((int32_t)(xorshift32(&rng) % 2001u) - 1000)) * 0.01f;
    }
    memcpy(myds_metal_buffer_data(raw_w2), rwp, 4096u * 512u * sizeof(float));
    CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(raw_w,  comp_bf, 0u, 0u, 4096u, 1u, true, 3u, MYDS_KV_COMP_CACHE_BF16));
    CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(raw_w2, comp_nv, 0u, 0u, 4096u, 1u, true, 3u, MYDS_KV_COMP_CACHE_NVFP4));
    uint8_t *bfp = (uint8_t *)myds_metal_buffer_data(comp_bf);
    uint8_t *nvp = (uint8_t *)myds_metal_buffer_data(comp_nv);
    for (uint32_t r = 4096u; r < max_nc; r += 4096u) {
        memcpy(bfp + (size_t)r * 1024u, bfp, 4096u * 1024u);
        memcpy(nvp + (size_t)r * 288u,  nvp, 4096u * 288u);
    }

    float *wp = (float *)myds_metal_buffer_data(win_buf);
    for (size_t i = 0; i < 128u * 512u; ++i) wp[i] = 0.25f;

    /* 1. Prefill gather_comp_512 (BF16 vs NVFP4, scattered top-512 across nc) */
    struct { const char *name; uint32_t nc; uint32_t rows; } gath_cases[] = {
        {"GATH512_16K_c256",   16384u, 256u},
        {"GATH512_32K_c256",   32768u, 256u},
        {"GATH512_64K_c256",   65536u, 256u},
        {"GATH512_128K_c256", 131072u, 256u},
        {"GATH512_256K_c256", 262144u, 256u},
    };
    printf("  --- Prefill gather_comp_512 (BF16 1024 B/row vs NVFP4 288 B/row) ---\n");
    for (size_t ci = 0; ci < sizeof(gath_cases) / sizeof(gath_cases[0]); ++ci) {
        const uint32_t nc = gath_cases[ci].nc, rows = gath_cases[ci].rows;
        int32_t *ip = (int32_t *)myds_metal_buffer_data(ids_buf);
        for (size_t i = 0; i < (size_t)rows * top_k; ++i) ip[i] = (int32_t)(xorshift32(&rng) % nc);

        CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_512_bf16",  gath_bf, mask_dum, comp_bf, ids_buf, rows, top_k, nc, nc, 0u, 1u));
        CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_512_nvfp4", gath_nv, mask_dum, comp_nv, ids_buf, rows, top_k, nc, nc, 0u, 1u));
        CHECK(memcmp(myds_metal_buffer_data(gath_bf), myds_metal_buffer_data(gath_nv), (size_t)rows * top_k * 512u * sizeof(uint16_t)) == 0);

        double t_bf[7], t_nv[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_512_bf16", gath_bf, mask_dum, comp_bf, ids_buf, rows, top_k, nc, nc, 0u, 1u));
            CHECK(myds_metal_commit_batch());
            t_bf[r] = monotonic_ms() - s0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(dispatch_gather_comp("kernel_myds_attn_gather_comp_512_nvfp4", gath_nv, mask_dum, comp_nv, ids_buf, rows, top_k, nc, nc, 0u, 1u));
            CHECK(myds_metal_commit_batch());
            t_nv[r] = monotonic_ms() - s1;
        }
        qsort(t_bf, 7, sizeof(double), cmp_double);
        qsort(t_nv, 7, sizeof(double), cmp_double);
        printf("    %-18s (nc=%6u, c=%3u): BF16 %6.3f ms | NVFP4 %6.3f ms | %.2fx\n",
               gath_cases[ci].name, nc, rows, t_bf[3], t_nv[3], t_bf[3] / t_nv[3]);
    }

    /* 2. Decode get_rows_to_f16 & direct_pack_fuse (top_k=512, 16x distinct cold id sets per CB) */
    const uint32_t dec_ncs[] = {32768u, 65536u, 131072u, 262144u};
    myds_metal_buffer_t *dec_ids[16];
    for (int k = 0; k < 16; ++k) {
        dec_ids[k] = myds_metal_buffer_alloc((size_t)top_k * sizeof(int32_t));
        CHECK(dec_ids[k] != NULL);
    }
    printf("  --- Decode get_rows_to_f16 (top_k=512, 16x distinct cold id sets, per call & x40 layers) ---\n");
    for (size_t ci = 0; ci < sizeof(dec_ncs) / sizeof(dec_ncs[0]); ++ci) {
        const uint32_t nc = dec_ncs[ci];
        for (int k = 0; k < 16; ++k) {
            int32_t *ip = (int32_t *)myds_metal_buffer_data(dec_ids[k]);
            for (uint32_t j = 0; j < top_k; ++j) ip[j] = (int32_t)(xorshift32(&rng) % nc);
        }

        CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_bf16_to_f16",  dec_f16b, comp_bf, dec_ids[0], nc, top_k, 128u));
        CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_nvfp4_to_f16", dec_f16,  comp_nv, dec_ids[0], nc, top_k, 128u));
        CHECK(memcmp((const uint16_t *)myds_metal_buffer_data(dec_f16b) + 128u * 512u,
                     (const uint16_t *)myds_metal_buffer_data(dec_f16)  + 128u * 512u,
                     (size_t)top_k * 512u * sizeof(uint16_t)) == 0);

        double t_bf[7], t_nv[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 16; ++k) {
                CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_bf16_to_f16", dec_f16b, comp_bf, dec_ids[k], nc, top_k, 128u));
            }
            CHECK(myds_metal_commit_batch());
            t_bf[r] = (monotonic_ms() - s0) / 16.0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 16; ++k) {
                CHECK(dispatch_get_rows_to_f16("kernel_dsv41_get_rows_nvfp4_to_f16", dec_f16, comp_nv, dec_ids[k], nc, top_k, 128u));
            }
            CHECK(myds_metal_commit_batch());
            t_nv[r] = (monotonic_ms() - s1) / 16.0;
        }
        qsort(t_bf, 7, sizeof(double), cmp_double);
        qsort(t_nv, 7, sizeof(double), cmp_double);
        printf("    DEC_getrows_%3uK   (nc=%6u): BF16 %6.4f ms (x40=%5.2f ms) | NVFP4 %6.4f ms (x40=%5.2f ms) | %.2fx\n",
               nc / 1024u, nc, t_bf[3], t_bf[3] * 40.0, t_nv[3], t_nv[3] * 40.0, t_bf[3] / t_nv[3]);
    }

    printf("  --- Decode direct_pack_fuse (n_raw=128 + top_k=512, 16x distinct cold id sets) ---\n");
    for (size_t ci = 0; ci < sizeof(dec_ncs) / sizeof(dec_ncs[0]); ++ci) {
        const uint32_t nc = dec_ncs[ci];
        for (int k = 0; k < 16; ++k) {
            int32_t *ip = (int32_t *)myds_metal_buffer_data(dec_ids[k]);
            for (uint32_t j = 0; j < top_k; ++j) ip[j] = (int32_t)(xorshift32(&rng) % nc);
        }

        double t_bf[7], t_nv[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 16; ++k) {
                CHECK(dispatch_direct_pack_fuse("kernel_myds_dsv41_direct_pack_fuse", dec_f16b, win_buf, comp_bf, dec_ids[k], 128u, 0u, nc, top_k));
            }
            CHECK(myds_metal_commit_batch());
            t_bf[r] = (monotonic_ms() - s0) / 16.0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 16; ++k) {
                CHECK(dispatch_direct_pack_fuse("kernel_myds_dsv41_direct_pack_fuse_nvfp4", dec_f16, win_buf, comp_nv, dec_ids[k], 128u, 0u, nc, top_k));
            }
            CHECK(myds_metal_commit_batch());
            t_nv[r] = (monotonic_ms() - s1) / 16.0;
        }
        qsort(t_bf, 7, sizeof(double), cmp_double);
        qsort(t_nv, 7, sizeof(double), cmp_double);
        printf("    DEC_packfuse_%3uK  (nc=%6u): BF16 %6.4f ms (x40=%5.2f ms) | NVFP4 %6.4f ms (x40=%5.2f ms) | %.2fx\n",
               nc / 1024u, nc, t_bf[3], t_bf[3] * 40.0, t_nv[3], t_nv[3] * 40.0, t_bf[3] / t_nv[3]);
    }
    for (int k = 0; k < 16; ++k) myds_metal_buffer_free(dec_ids[k]);

    /* 3. Writers: Prefill kv_rope_quant_batch (rows=2048) & Decode quant+pack_bf16 vs quant_pack_kv_nvfp4 */
    printf("  --- Writers: Prefill kv_rope_quant_batch (2048 rows) & Decode commit (1 row, 10x amortized) ---\n");
    {
        double t_bf[7], t_nv[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(raw_w, comp_bf, 0u, 0u, 2048u, 1u, true, 3u, MYDS_KV_COMP_CACHE_BF16));
            CHECK(myds_metal_commit_batch());
            t_bf[r] = monotonic_ms() - s0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_kv_rope_quant_batch_mode(raw_w2, comp_nv, 0u, 0u, 2048u, 1u, true, 3u, MYDS_KV_COMP_CACHE_NVFP4));
            CHECK(myds_metal_commit_batch());
            t_nv[r] = monotonic_ms() - s1;
        }
        qsort(t_bf, 7, sizeof(double), cmp_double);
        qsort(t_nv, 7, sizeof(double), cmp_double);
        printf("    WR_prefill_2048    (rows=2048): BF16 %6.3f ms | NVFP4 %6.3f ms | %.2fx\n",
               t_bf[3], t_nv[3], t_bf[3] / t_nv[3]);
    }
    {
        double t_bf[7], t_nv[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 10; ++k) {
                CHECK(myds_v41_gpu_quant(raw_w, 512u, 1u, 3u));
                CHECK(myds_v41_gpu_pack_bf16(comp_bf, 0u, raw_w, 0u, 512u));
            }
            CHECK(myds_metal_commit_batch());
            t_bf[r] = (monotonic_ms() - s0) / 10.0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 10; ++k) {
                CHECK(myds_v41_gpu_quant_pack_kv_nvfp4(comp_nv, 0u, raw_w2, 0u, 1u));
            }
            CHECK(myds_metal_commit_batch());
            t_nv[r] = (monotonic_ms() - s1) / 10.0;
        }
        qsort(t_bf, 7, sizeof(double), cmp_double);
        qsort(t_nv, 7, sizeof(double), cmp_double);
        printf("    WR_decode_1row     (1 row x4) : BF16 %6.4f ms (x4=%5.3f ms) | NVFP4 %6.4f ms (x4=%5.3f ms) | %.2fx\n",
               t_bf[3], t_bf[3] * 4.0, t_nv[3], t_nv[3] * 4.0, t_bf[3] / t_nv[3]);
    }

    myds_metal_buffer_free(comp_bf);
    myds_metal_buffer_free(comp_nv);
    myds_metal_buffer_free(raw_w);
    myds_metal_buffer_free(raw_w2);
    myds_metal_buffer_free(ids_buf);
    myds_metal_buffer_free(gath_bf);
    myds_metal_buffer_free(gath_nv);
    myds_metal_buffer_free(mask_dum);
    myds_metal_buffer_free(win_buf);
    myds_metal_buffer_free(dec_f16);
    myds_metal_buffer_free(dec_f16b);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        bool run_bench = false;
        bool pack_fuse_only = false;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--bench") == 0) run_bench = true;
            if (strcmp(argv[i], "--pack-fuse-only") == 0) pack_fuse_only = true;
        }
        printf("=== test_kv_comp_nvfp4 (4.5-bit E2M1FN + E4M3FN, 288 B/row, -71.88%%, 0-ULP) ===\n");
        if (!pack_fuse_only) {
            test_env_parsing();
        }
        setenv("MYDS_ATTN_PACKED_REGISTER", "1", 1);
        setenv("MYDS_ATTN_ROPE_FUSE", "1", 1);
        if (!myds_metal_init(NULL) && !myds_metal_init("bin/myds.metallib")) {
            fprintf(stderr, "Failed to initialize Metal\n");
            return 1;
        }
        if (pack_fuse_only) {
            test_e2e_prefill_and_decode_attn_0ulp(true);
        } else {
            verify_pso_occupancy();
            test_exhaustive_2032_states();
            test_50k_rows_both_writers();
            test_e2e_prefill_and_decode_attn_0ulp(false);
            test_lowkv_nvfp4_roundtrip_and_isolation();
            if (run_bench) {
                run_nvfp4_microbenchmarks();
            }
        }
        myds_metal_cleanup();
        printf("ALL test_kv_comp_nvfp4 CHECKS PASSED!\n");
    }
    return 0;
}
