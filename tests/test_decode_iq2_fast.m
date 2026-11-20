/* tests/test_decode_iq2_fast.m
 *
 * Strict 0-ULP verification + PSO occupancy check + M4 benchmark for [D-M1]
 * (MYDS_DECODE_IQ2_FAST, kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_fast_f32):
 *   1. Strict knob parsing (NULL/"0"->0, "1"->1, invalid->-1 fail-closed) and
 *      fail-closed init on invalid or conflicting configs.
 *   2. Direct Metal PSO occupancy check (maxTotalThreadsPerThreadgroup,
 *      threadExecutionWidth, staticThreadgroupMemoryLength, dynamic smem 2048 vs 2176 B)
 *      for both NSG=2 and NSG=4.
 *   3. Strict 0-ULP (memcmp == 0) on dst_gate, dst_up, and dst_mid vs baseline
 *      kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32 AND reference
 *      kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 across:
 *        - arbitrary random 24-bit FP32 activations
 *        - BF16-rounded activations
 *        - signed zeros (-0.0f / +0.0f)
 *        - FP32 subnormals (positive & negative)
 *        - extreme FP16 block scales (subnormals, large scales, negative, zero)
 *          covering all 256 grid codes, all 128 sign codes, and all 16 sub-block scales
 *        - k=3 and k=6, NSG=2 and NSG=4, clamp=0 and clamp=7.0f
 *   4. End-to-end host API parity via myds_metal_routed_moe + /debug/state counters.
 *   5. Optional --bench on M4 comparing popcnt (2176 B) vs popcnt_fast (2048 B).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "myds_gguf.h"
#include "myds_metal.h"

static int cmp_dbl(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_addend;
    int32_t  tp_expert_base;
};

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

static uint32_t g_rng = 0x51a7c3d9u;
static uint32_t xorshift32(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static float bits_to_f32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint32_t f32_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static float round_bf16(float f) {
    uint32_t u = f32_to_bits(f);
    if ((u & 0x7f800000u) != 0x7f800000u) {
        u = (u + 0x7fffu + ((u >> 16u) & 1u)) & 0xffff0000u;
    }
    return bits_to_f32(u);
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

/* Fill IQ2_XXS blocks (66 bytes each: fp16 d + 32 x uint16 qs).
 * Covers all 256 grid codes, all 128 sign codes, all 16 scale nibbles. */
static void fill_iq2_weights(uint8_t *dst, size_t n_blocks, bool extreme_scales, uint32_t seed) {
    g_rng = seed;
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = dst + b * 66u;
        uint16_t d;
        if (extreme_scales) {
            static const uint16_t ext_d[] = {
                0x0000u, /* +0.0 */
                0x8000u, /* -0.0 */
                0x0001u, /* smallest +fp16 subnormal */
                0x8001u, /* smallest -fp16 subnormal */
                0x03ffu, /* largest +fp16 subnormal */
                0x0400u, /* smallest +fp16 normal */
                0x3c00u, /* 1.0 */
                0xbc00u, /* -1.0 */
                0x4800u, /* 8.0 */
                0xc800u, /* -8.0 */
                0x5800u, /* 128.0 */
                0x2400u, /* 2^-6 */
            };
            d = ext_d[b % (sizeof(ext_d) / sizeof(ext_d[0]))];
        } else {
            /* Moderate realistic scales (~2^-9 .. 2^-4, positive and negative) */
            uint16_t sign = (uint16_t)((xorshift32() & 1u) << 15u);
            uint16_t exp = (uint16_t)((6u + (xorshift32() % 6u)) << 10u);
            uint16_t mant = (uint16_t)(xorshift32() & 0x3ffu);
            d = sign | exp | mant;
        }
        memcpy(blk, &d, 2u);
        uint16_t *qs = (uint16_t *)(blk + 2u);
        for (size_t ib = 0; ib < 8u; ib++) {
            uint32_t g0 = (uint32_t)((b * 17u + ib * 31u + (xorshift32() & 0xffu)) & 0xffu);
            uint32_t g1 = (uint32_t)((b * 23u + ib * 13u + (xorshift32() & 0xffu)) & 0xffu);
            uint32_t g2 = (uint32_t)((b * 29u + ib * 7u  + (xorshift32() & 0xffu)) & 0xffu);
            uint32_t g3 = (uint32_t)((b * 37u + ib * 5u  + (xorshift32() & 0xffu)) & 0xffu);
            qs[4u * ib + 0u] = (uint16_t)(g0 | (g1 << 8u));
            qs[4u * ib + 1u] = (uint16_t)(g2 | (g3 << 8u));
            uint32_t s0 = (uint32_t)((b + ib * 4u + 0u) & 127u);
            uint32_t s1 = (uint32_t)((b + ib * 4u + 1u) & 127u);
            uint32_t s2 = (uint32_t)((b + ib * 4u + 2u) & 127u);
            uint32_t s3 = (uint32_t)((b + ib * 4u + 3u) & 127u);
            uint32_t sc = (uint32_t)((b + ib) & 15u);
            uint32_t aux32 = s0 | (s1 << 7u) | (s2 << 14u) | (s3 << 21u) | (sc << 28u);
            qs[4u * ib + 2u] = (uint16_t)(aux32 & 0xffffu);
            qs[4u * ib + 3u] = (uint16_t)(aux32 >> 16u);
        }
    }
}

enum act_pattern {
    ACT_RANDOM_FP32 = 0,
    ACT_BF16,
    ACT_SIGNED_ZEROS,
    ACT_SUBNORMALS,
    ACT_EXTREME_MIX,
    ACT_COUNT
};

static const char *act_name(enum act_pattern p) {
    switch (p) {
        case ACT_RANDOM_FP32: return "random_fp32";
        case ACT_BF16: return "bf16";
        case ACT_SIGNED_ZEROS: return "signed_zeros(-0/+0)";
        case ACT_SUBNORMALS: return "subnormals";
        case ACT_EXTREME_MIX: return "extreme_scales+fp32";
        default: return "?";
    }
}

static void fill_activations(float *x, uint32_t dim, enum act_pattern pat, uint32_t seed) {
    g_rng = seed;
    switch (pat) {
        case ACT_RANDOM_FP32:
            for (uint32_t i = 0; i < dim; i++) {
                /* Full 24-bit mantissa arbitrary FP32 in [-2, +2] */
                uint32_t r = xorshift32();
                float v = ((float)(int32_t)(r & 0xffffffu) - 8388608.0f) * (2.0f / 8388608.0f);
                x[i] = v;
            }
            break;
        case ACT_BF16:
            for (uint32_t i = 0; i < dim; i++) {
                uint32_t r = xorshift32();
                float v = ((float)(int32_t)(r & 0xffffffu) - 8388608.0f) * (2.0f / 8388608.0f);
                x[i] = round_bf16(v);
            }
            break;
        case ACT_SIGNED_ZEROS:
            for (uint32_t i = 0; i < dim; i++) {
                x[i] = (xorshift32() & 1u) ? bits_to_f32(0x80000000u) : 0.0f;
            }
            break;
        case ACT_SUBNORMALS:
            for (uint32_t i = 0; i < dim; i++) {
                uint32_t sign = (xorshift32() & 1u) << 31u;
                uint32_t mant = 1u + (xorshift32() & 0x7fffffu); /* 0x00000001 .. 0x007fffff */
                x[i] = bits_to_f32(sign | mant);
            }
            break;
        case ACT_EXTREME_MIX:
            for (uint32_t i = 0; i < dim; i++) {
                uint32_t kind = i % 5u;
                if (kind == 0u) x[i] = bits_to_f32(0x80000000u);
                else if (kind == 1u) x[i] = bits_to_f32(((xorshift32() & 1u) << 31u) | (1u + (xorshift32() & 0x7ffu)));
                else if (kind == 2u) x[i] = ((xorshift32() & 1u) ? -1.0f : 1.0f) * 1.0e-3f * (float)(1u + (xorshift32() & 0x3ffu));
                else if (kind == 3u) x[i] = ((xorshift32() & 1u) ? -1.0f : 1.0f) * (float)(1u + (xorshift32() & 0x7fffu)) * 1.0e-4f;
                else x[i] = ((float)(int32_t)(xorshift32() & 0xffffffu) - 8388608.0f) * 1.0e-7f;
            }
            break;
        default:
            break;
    }
}

static int test_knob_and_fail_closed(void) {
    printf("[1/4] Verifying MYDS_DECODE_IQ2_FAST strict parsing & fail-closed init...\n");
    unsetenv("MYDS_DECODE_IQ2_FAST");
    if (myds_metal_decode_iq2_fast_enabled() != 0) {
        fprintf(stderr, "FAIL: absent MYDS_DECODE_IQ2_FAST must default to 0\n");
        return 1;
    }
    setenv("MYDS_DECODE_IQ2_FAST", "0", 1);
    if (myds_metal_decode_iq2_fast_enabled() != 0) return 1;
    setenv("MYDS_DECODE_IQ2_FAST", "1", 1);
    if (myds_metal_decode_iq2_fast_enabled() != 1) return 1;

    const char *bad[] = {"", "2", "-1", "true", "yes", "01", "1 "};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        setenv("MYDS_DECODE_IQ2_FAST", bad[i], 1);
        if (myds_metal_decode_iq2_fast_enabled() != -1) {
            fprintf(stderr, "FAIL: invalid MYDS_DECODE_IQ2_FAST='%s' must return -1\n", bad[i]);
            return 1;
        }
        if (myds_metal_init("bin")) {
            fprintf(stderr, "FAIL: myds_metal_init must fail closed on MYDS_DECODE_IQ2_FAST='%s'\n", bad[i]);
            myds_metal_cleanup();
            return 1;
        }
    }
    /* Conflicting combination: MYDS_DECODE_IQ2_FAST=1 + MYDS_MOE_IQ2_CONSTANT_LUT=1 */
    setenv("MYDS_DECODE_IQ2_FAST", "1", 1);
    setenv("MYDS_MOE_IQ2_CONSTANT_LUT", "1", 1);
    if (myds_metal_init("bin")) {
        fprintf(stderr, "FAIL: MYDS_DECODE_IQ2_FAST=1 + MYDS_MOE_IQ2_CONSTANT_LUT=1 must fail closed\n");
        myds_metal_cleanup();
        return 1;
    }
    unsetenv("MYDS_MOE_IQ2_CONSTANT_LUT");
    setenv("MYDS_DECODE_IQ2_FAST", "0", 1);
    printf("  PASS: strict knob parsing and fail-closed init checks.\n");
    return 0;
}

static int compare_f32_bits(const float *ref, const float *got, size_t n, const char *tag) {
    if (memcmp(ref, got, n * sizeof(float)) == 0) return 0;
    size_t diffs = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t br = f32_to_bits(ref[i]);
        uint32_t bg = f32_to_bits(got[i]);
        if (br != bg) {
            if (diffs < 5u) {
                fprintf(stderr, "FAIL [%s] idx=%zu: ref=%g (0x%08x) got=%g (0x%08x)\n",
                        tag, i, ref[i], br, got[i], bg);
            }
            diffs++;
        }
    }
    fprintf(stderr, "FAIL [%s]: total %zu / %zu bit differences!\n", tag, diffs, n);
    return 1;
}

static int test_direct_metal_parity_and_bench(bool do_bench) {
    printf("[2/4] Testing PSO occupancy & direct Metal 0-ULP parity (popcnt vs popcnt_fast vs ref)...\n");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "FAIL: no Metal device\n"); return 1; }
    NSError *err = nil;
    NSURL *url = [NSURL fileURLWithPath:@"bin/myds.metallib"];
    id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
    if (!lib) {
        fprintf(stderr, "FAIL: cannot load bin/myds.metallib: %s\n", [[err localizedDescription] UTF8String]);
        return 1;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (!queue) return 1;

    const uint32_t dim = 5120u;
    const uint32_t expert_dim = 2304u;
    const uint32_t n_experts = 8u;
    const uint64_t row_bytes = (uint64_t)(dim / 256u) * 66u;
    const uint64_t exp_bytes = (uint64_t)expert_dim * row_bytes;

    id<MTLBuffer> buf_gate = [dev newBufferWithLength:(NSUInteger)(n_experts * exp_bytes) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_up   = [dev newBufferWithLength:(NSUInteger)(n_experts * exp_bytes) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_x    = [dev newBufferWithLength:(NSUInteger)(dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_ids  = [dev newBufferWithLength:(NSUInteger)(6u * sizeof(int32_t)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_w    = [dev newBufferWithLength:(NSUInteger)(6u * sizeof(float)) options:MTLResourceStorageModeShared];

    id<MTLBuffer> ref_g = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ref_u = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ref_m = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pop_g = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pop_u = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pop_m = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> fst_g = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> fst_u = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> fst_m = [dev newBufferWithLength:(NSUInteger)(6u * expert_dim * sizeof(float)) options:MTLResourceStorageModeShared];

    int32_t *ids_p = (int32_t *)buf_ids.contents;
    float *w_p = (float *)buf_w.contents;
    const int32_t exp_list[6] = {0, 3, 7, 1, 5, 2};
    const float w_list[6] = {0.37f, 0.29f, 0.19f, 0.08f, 0.05f, 0.02f};
    memcpy(ids_p, exp_list, sizeof(exp_list));
    memcpy(w_p, w_list, sizeof(w_list));

    const short nsgs[2] = {2, 4};
    for (int ni = 0; ni < 2; ni++) {
        short nsg = nsgs[ni];
        MTLFunctionConstantValues *fc = [[MTLFunctionConstantValues alloc] init];
        [fc setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

        id<MTLFunction> fn_ref = [lib newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32" constantValues:fc error:&err];
        id<MTLFunction> fn_pop = [lib newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32" constantValues:fc error:&err];
        id<MTLFunction> fn_fst = [lib newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_fast_f32" constantValues:fc error:&err];
        if (!fn_ref || !fn_pop || !fn_fst) {
            fprintf(stderr, "FAIL: missing kernel function for NSG=%d\n", (int)nsg);
            return 1;
        }
        id<MTLComputePipelineState> pso_ref = [dev newComputePipelineStateWithFunction:fn_ref error:&err];
        id<MTLComputePipelineState> pso_pop = [dev newComputePipelineStateWithFunction:fn_pop error:&err];
        id<MTLComputePipelineState> pso_fst = [dev newComputePipelineStateWithFunction:fn_fst error:&err];
        if (!pso_ref || !pso_pop || !pso_fst) {
            fprintf(stderr, "FAIL: PSO creation failed for NSG=%d\n", (int)nsg);
            return 1;
        }
        printf("  PSO (NSG=%d):\n"
               "    popcnt     : maxThreads=%4lu  width=%2lu  staticSmem=%lu  dynSmem=2176 B\n"
               "    popcnt_fast: maxThreads=%4lu  width=%2lu  staticSmem=%lu  dynSmem=2048 B\n",
               (int)nsg,
               (unsigned long)pso_pop.maxTotalThreadsPerThreadgroup,
               (unsigned long)pso_pop.threadExecutionWidth,
               (unsigned long)pso_pop.staticThreadgroupMemoryLength,
               (unsigned long)pso_fst.maxTotalThreadsPerThreadgroup,
               (unsigned long)pso_fst.threadExecutionWidth,
               (unsigned long)pso_fst.staticThreadgroupMemoryLength);
        if (pso_fst.maxTotalThreadsPerThreadgroup < (NSUInteger)(32 * nsg) ||
            pso_fst.maxTotalThreadsPerThreadgroup < pso_pop.maxTotalThreadsPerThreadgroup ||
            pso_fst.staticThreadgroupMemoryLength > 0u) {
            fprintf(stderr, "FAIL: PSO occupancy regression for popcnt_fast (NSG=%d)\n", (int)nsg);
            return 1;
        }

        const uint32_t ks[3] = {1u, 3u, 6u};
        const float clamps[3] = {0.0f, 7.0f, 7.0f};
        for (int ki = 0; ki < 3; ki++) {
            const uint32_t k_act = ks[ki];
            const float clamp_v = clamps[ki];
            const uint32_t write_clamped = (k_act == 1u) ? 1u : 0u;
            for (int pat = 0; pat < ACT_COUNT; pat++) {
                const bool extreme = (pat == ACT_EXTREME_MIX);
                fill_iq2_weights((uint8_t *)buf_gate.contents, (size_t)n_experts * expert_dim * (dim / 256u),
                                 extreme, 0x1111u + (uint32_t)(ni * 100 + ki * 20 + pat));
                fill_iq2_weights((uint8_t *)buf_up.contents, (size_t)n_experts * expert_dim * (dim / 256u),
                                 extreme, 0x2222u + (uint32_t)(ni * 100 + ki * 20 + pat));
                fill_activations((float *)buf_x.contents, dim, (enum act_pattern)pat,
                                 0x3333u + (uint32_t)(ni * 100 + ki * 20 + pat));

                struct ds4_metal_args_mul_mv_id ga = {
                    .nei0 = (int32_t)k_act, .nei1 = 1, .nbi1 = (uint64_t)k_act * sizeof(int32_t),
                    .ne00 = (int32_t)dim, .ne01 = (int32_t)expert_dim, .ne02 = (int32_t)n_experts,
                    .nb00 = 66u, .nb01 = row_bytes, .nb02 = exp_bytes,
                    .ne10 = (int32_t)dim, .ne11 = 1, .ne12 = 1, .ne13 = 1,
                    .nb10 = sizeof(float), .nb11 = (uint64_t)dim * sizeof(float), .nb12 = (uint64_t)dim * sizeof(float),
                    .ne0 = (int32_t)expert_dim, .ne1 = (int32_t)k_act, .nb1 = (uint64_t)expert_dim * sizeof(float),
                    .nr0 = 4, .tp_rank = 0, .tp_world = 0, .tp_addend = 0, .tp_expert_base = 0
                };
                struct ds4_metal_dsv4_moe_swiglu_weight_args sa = {
                    .width = expert_dim, .rows = k_act,
                    .gate_row_stride = (uint64_t)expert_dim * sizeof(float),
                    .up_row_stride = (uint64_t)expert_dim * sizeof(float),
                    .mid_row_stride = (uint64_t)expert_dim * sizeof(float),
                    .weight_stride = sizeof(float),
                    .write_clamped = write_clamped, .clamp_value = clamp_v
                };

                const NSUInteger row_groups = (expert_dim + (NSUInteger)(4 * nsg) - 1u) / (NSUInteger)(4 * nsg);
                memset(ref_g.contents, 0x55, k_act * expert_dim * sizeof(float));
                memset(ref_u.contents, 0x55, k_act * expert_dim * sizeof(float));
                memset(ref_m.contents, 0x55, k_act * expert_dim * sizeof(float));
                memset(pop_g.contents, 0x66, k_act * expert_dim * sizeof(float));
                memset(pop_u.contents, 0x66, k_act * expert_dim * sizeof(float));
                memset(pop_m.contents, 0x66, k_act * expert_dim * sizeof(float));
                memset(fst_g.contents, 0x77, k_act * expert_dim * sizeof(float));
                memset(fst_u.contents, 0x77, k_act * expert_dim * sizeof(float));
                memset(fst_m.contents, 0x77, k_act * expert_dim * sizeof(float));

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                id<MTLComputePipelineState> psos[3] = {pso_ref, pso_pop, pso_fst};
                NSUInteger smems[3] = {2176u, 2176u, 2048u};
                id<MTLBuffer> og[3] = {ref_g, pop_g, fst_g};
                id<MTLBuffer> ou[3] = {ref_u, pop_u, fst_u};
                id<MTLBuffer> om[3] = {ref_m, pop_m, fst_m};
                for (int a = 0; a < 3; a++) {
                    [enc setComputePipelineState:psos[a]];
                    [enc setBytes:&ga length:sizeof(ga) atIndex:0];
                    [enc setBytes:&sa length:sizeof(sa) atIndex:1];
                    [enc setBuffer:buf_gate offset:0 atIndex:2];
                    [enc setBuffer:buf_up offset:0 atIndex:3];
                    [enc setBuffer:buf_x offset:0 atIndex:4];
                    [enc setBuffer:og[a] offset:0 atIndex:5];
                    [enc setBuffer:ou[a] offset:0 atIndex:6];
                    [enc setBuffer:om[a] offset:0 atIndex:7];
                    [enc setBuffer:buf_ids offset:0 atIndex:8];
                    [enc setBuffer:buf_w offset:0 atIndex:9];
                    [enc setThreadgroupMemoryLength:smems[a] atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, k_act)
                        threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    fprintf(stderr, "FAIL: Metal CB failed\n");
                    return 1;
                }

                char tag[128];
                snprintf(tag, sizeof(tag), "NSG=%d k=%u pat=%s popcnt_vs_fast mid", (int)nsg, k_act, act_name((enum act_pattern)pat));
                if (compare_f32_bits((const float *)pop_m.contents, (const float *)fst_m.contents, k_act * expert_dim, tag) != 0) return 1;
                if (write_clamped) {
                    snprintf(tag, sizeof(tag), "NSG=%d k=%u pat=%s popcnt_vs_fast gate", (int)nsg, k_act, act_name((enum act_pattern)pat));
                    if (compare_f32_bits((const float *)pop_g.contents, (const float *)fst_g.contents, expert_dim, tag) != 0) return 1;
                    snprintf(tag, sizeof(tag), "NSG=%d k=%u pat=%s popcnt_vs_fast up", (int)nsg, k_act, act_name((enum act_pattern)pat));
                    if (compare_f32_bits((const float *)pop_u.contents, (const float *)fst_u.contents, expert_dim, tag) != 0) return 1;
                }
                snprintf(tag, sizeof(tag), "NSG=%d k=%u pat=%s ref_vs_fast mid", (int)nsg, k_act, act_name((enum act_pattern)pat));
                if (compare_f32_bits((const float *)ref_m.contents, (const float *)fst_m.contents, k_act * expert_dim, tag) != 0) return 1;
            }
        }
    }
    printf("  PASS: memcmp == 0 across random FP32, BF16, -0/+0, subnormals, extreme scales (NSG=2/4, k=3/6).\n");

    if (do_bench) {
        printf("\n[4/4] M4 GPU Benchmark (--bench, steady-state interleaved median-of-11): IQ2_XXS Decode Gate+Up+SwiGLU (dim=5120, exp=2304)\n");
        fill_iq2_weights((uint8_t *)buf_gate.contents, (size_t)n_experts * expert_dim * (dim / 256u), false, 0x9001u);
        fill_iq2_weights((uint8_t *)buf_up.contents, (size_t)n_experts * expert_dim * (dim / 256u), false, 0x9002u);
        fill_activations((float *)buf_x.contents, dim, ACT_BF16, 0x9003u);

        const int rounds = 11, iters = 40;
        for (int ni = 0; ni < 2; ni++) {
            short nsg = nsgs[ni];
            MTLFunctionConstantValues *fc = [[MTLFunctionConstantValues alloc] init];
            [fc setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
            id<MTLComputePipelineState> pso_pop = [dev newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32" constantValues:fc error:nil] error:nil];
            id<MTLComputePipelineState> pso_fst = [dev newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_fast_f32" constantValues:fc error:nil] error:nil];
            const NSUInteger row_groups = (expert_dim + (NSUInteger)(4 * nsg) - 1u) / (NSUInteger)(4 * nsg);

            const uint32_t ks[2] = {3u, 6u};
            for (int ki = 0; ki < 2; ki++) {
                const uint32_t k_act = ks[ki];
                struct ds4_metal_args_mul_mv_id ga = {
                    .nei0 = (int32_t)k_act, .nei1 = 1, .nbi1 = (uint64_t)k_act * sizeof(int32_t),
                    .ne00 = (int32_t)dim, .ne01 = (int32_t)expert_dim, .ne02 = (int32_t)n_experts,
                    .nb00 = 66u, .nb01 = row_bytes, .nb02 = exp_bytes,
                    .ne10 = (int32_t)dim, .ne11 = 1, .ne12 = 1, .ne13 = 1,
                    .nb10 = sizeof(float), .nb11 = (uint64_t)dim * sizeof(float), .nb12 = (uint64_t)dim * sizeof(float),
                    .ne0 = (int32_t)expert_dim, .ne1 = (int32_t)k_act, .nb1 = (uint64_t)expert_dim * sizeof(float),
                    .nr0 = 4, .tp_rank = 0, .tp_world = 0, .tp_addend = 0, .tp_expert_base = 0
                };
                struct ds4_metal_dsv4_moe_swiglu_weight_args sa = {
                    .width = expert_dim, .rows = k_act,
                    .gate_row_stride = (uint64_t)expert_dim * sizeof(float),
                    .up_row_stride = (uint64_t)expert_dim * sizeof(float),
                    .mid_row_stride = (uint64_t)expert_dim * sizeof(float),
                    .weight_stride = sizeof(float),
                    .write_clamped = 0u, .clamp_value = 7.0f
                };

                /* Bring Apple GPU clocks to steady state before interleaving A/B */
                for (int w = 0; w < 6; w++) {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int it = 0; it < 60; it++) {
                        [enc setComputePipelineState:pso_pop];
                        [enc setBytes:&ga length:sizeof(ga) atIndex:0];
                        [enc setBytes:&sa length:sizeof(sa) atIndex:1];
                        [enc setBuffer:buf_gate offset:0 atIndex:2];
                        [enc setBuffer:buf_up offset:0 atIndex:3];
                        [enc setBuffer:buf_x offset:0 atIndex:4];
                        [enc setBuffer:fst_g offset:0 atIndex:5];
                        [enc setBuffer:fst_u offset:0 atIndex:6];
                        [enc setBuffer:fst_m offset:0 atIndex:7];
                        [enc setBuffer:buf_ids offset:0 atIndex:8];
                        [enc setBuffer:buf_w offset:0 atIndex:9];
                        [enc setThreadgroupMemoryLength:2176u atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, k_act)
                            threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }

                double pop_samples[11], fst_samples[11];
                for (int r = 0; r < rounds; r++) {
                    for (int arm = 0; arm < 2; arm++) {
                        const int run_fast = (r & 1) ? (1 - arm) : arm;
                        id<MTLComputePipelineState> pso = run_fast ? pso_fst : pso_pop;
                        NSUInteger smem = run_fast ? 2048u : 2176u;
                        double t0 = now_us();
                        id<MTLCommandBuffer> cb = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        for (int it = 0; it < iters; it++) {
                            [enc setComputePipelineState:pso];
                            [enc setBytes:&ga length:sizeof(ga) atIndex:0];
                            [enc setBytes:&sa length:sizeof(sa) atIndex:1];
                            [enc setBuffer:buf_gate offset:0 atIndex:2];
                            [enc setBuffer:buf_up offset:0 atIndex:3];
                            [enc setBuffer:buf_x offset:0 atIndex:4];
                            [enc setBuffer:fst_g offset:0 atIndex:5];
                            [enc setBuffer:fst_u offset:0 atIndex:6];
                            [enc setBuffer:fst_m offset:0 atIndex:7];
                            [enc setBuffer:buf_ids offset:0 atIndex:8];
                            [enc setBuffer:buf_w offset:0 atIndex:9];
                            [enc setThreadgroupMemoryLength:smem atIndex:0];
                            [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, k_act)
                                threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
                        }
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        double avg = (cb.GPUEndTime > cb.GPUStartTime)
                            ? ((cb.GPUEndTime - cb.GPUStartTime) * 1e6 / (double)iters)
                            : ((now_us() - t0) / (double)iters);
                        if (run_fast) fst_samples[r] = avg; else pop_samples[r] = avg;
                    }
                }
                qsort(pop_samples, (size_t)rounds, sizeof(double), cmp_dbl);
                qsort(fst_samples, (size_t)rounds, sizeof(double), cmp_dbl);
                const double pop_us = pop_samples[rounds / 2];
                const double fst_us = fst_samples[rounds / 2];
                const double bytes_rd = (double)k_act * (double)exp_bytes * 2.0;
                printf("  NSG=%d k=%u (%.1f MB/L): popcnt=%6.1f us (%.1f GB/s)  fast=%6.1f us (%.1f GB/s)  delta=%+5.1f us (%.2fx)\n",
                       (int)nsg, k_act, bytes_rd / 1e6,
                       pop_us, bytes_rd / (pop_us * 1e3),
                       fst_us, bytes_rd / (fst_us * 1e3),
                       fst_us - pop_us, pop_us / fst_us);
            }
        }
    }
    return 0;
}

static int test_host_api_and_counters(void) {
    printf("[3/4] Testing host API myds_metal_routed_moe parity & /debug/state counters...\n");
    const uint32_t dim = 5120u, expert_dim = 2304u, n_exp = 6u, k_act = 3u;
    const uint64_t gate_bytes = (uint64_t)n_exp * expert_dim * (dim / 256u) * 66u;
    const uint64_t down_bytes = (uint64_t)n_exp * dim * (expert_dim / 256u) * 84u;

    uint8_t *host_gate = (uint8_t *)malloc((size_t)gate_bytes);
    uint8_t *host_up   = (uint8_t *)malloc((size_t)gate_bytes);
    uint8_t *host_down = (uint8_t *)calloc((size_t)down_bytes, 1u);
    float   *host_x    = (float *)malloc(dim * sizeof(float));
    float   *out_ref   = (float *)malloc(dim * sizeof(float));
    float   *out_fast  = (float *)malloc(dim * sizeof(float));
    if (!host_gate || !host_up || !host_down || !host_x || !out_ref || !out_fast) return 1;

    fill_iq2_weights(host_gate, (size_t)n_exp * expert_dim * (dim / 256u), false, 0x4141u);
    fill_iq2_weights(host_up,   (size_t)n_exp * expert_dim * (dim / 256u), false, 0x4242u);
    /* Populate valid Q2_K down weights so full MoE output is non-trivial */
    for (size_t b = 0; b < (size_t)n_exp * dim * (expert_dim / 256u); b++) {
        uint8_t *blk = host_down + b * 84u;
        for (size_t i = 0; i < 16u; i++) blk[i] = (uint8_t)((b + i) & 0xffu);
        for (size_t i = 16u; i < 80u; i++) blk[i] = (uint8_t)((b * 7u + i) & 0xffu);
        uint16_t d = 0x2400u, dmin = 0x2000u;
        memcpy(blk + 80u, &d, 2u);
        memcpy(blk + 82u, &dmin, 2u);
    }
    fill_activations(host_x, dim, ACT_RANDOM_FP32, 0x4343u);

    const int32_t sel_ids[3] = {1, 4, 2};
    const float sel_w[3] = {0.5f, 0.3f, 0.2f};

    for (int arm = 0; arm < 2; arm++) {
        setenv("MYDS_MOE_IQ2_POPCNT_SIGN", "1", 1);
        setenv("MYDS_DECODE_IQ2_FAST", arm ? "1" : "0", 1);
        if (!myds_metal_init("bin")) {
            fprintf(stderr, "FAIL: myds_metal_init failed arm=%d\n", arm);
            return 1;
        }
        myds_metal_decode_iq2_fast_counters(NULL, true);

        myds_metal_buffer_t *b_out  = myds_metal_buffer_alloc(dim * sizeof(float));
        myds_metal_buffer_t *b_x    = myds_metal_buffer_alloc(dim * sizeof(float));
        myds_metal_buffer_t *b_gate = myds_metal_buffer_alloc((size_t)gate_bytes);
        myds_metal_buffer_t *b_up   = myds_metal_buffer_alloc((size_t)gate_bytes);
        myds_metal_buffer_t *b_down = myds_metal_buffer_alloc((size_t)down_bytes);
        myds_metal_buffer_t *b_ids  = myds_metal_buffer_alloc(k_act * sizeof(int32_t));
        myds_metal_buffer_t *b_w    = myds_metal_buffer_alloc(k_act * sizeof(float));
        myds_metal_buffer_t *s_g    = myds_metal_buffer_alloc(k_act * expert_dim * sizeof(float));
        myds_metal_buffer_t *s_u    = myds_metal_buffer_alloc(k_act * expert_dim * sizeof(float));
        myds_metal_buffer_t *s_m    = myds_metal_buffer_alloc(k_act * expert_dim * sizeof(float));
        myds_metal_buffer_t *s_d    = myds_metal_buffer_alloc(k_act * dim * sizeof(float));

        memcpy(myds_metal_buffer_data(b_x), host_x, dim * sizeof(float));
        memcpy(myds_metal_buffer_data(b_gate), host_gate, (size_t)gate_bytes);
        memcpy(myds_metal_buffer_data(b_up), host_up, (size_t)gate_bytes);
        memcpy(myds_metal_buffer_data(b_down), host_down, (size_t)down_bytes);
        memcpy(myds_metal_buffer_data(b_ids), sel_ids, sizeof(sel_ids));
        memcpy(myds_metal_buffer_data(b_w), sel_w, sizeof(sel_w));

        if (!myds_metal_routed_moe(b_out, b_x, b_gate, b_up, b_down, b_ids, b_w,
                s_g, s_u, s_m, s_d, dim, expert_dim, n_exp, k_act,
                GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 7.0f)) {
            fprintf(stderr, "FAIL: myds_metal_routed_moe failed arm=%d\n", arm);
            return 1;
        }
        memcpy(arm ? out_fast : out_ref, myds_metal_buffer_data(b_out), dim * sizeof(float));

        myds_metal_decode_iq2_fast_counters_t cnt;
        myds_metal_decode_iq2_fast_counters(&cnt, false);
        if (arm == 0 && (cnt.requests != 0u || cnt.encoded != 0u || cnt.failures != 0u)) {
            fprintf(stderr, "FAIL: expected zero counters when MYDS_DECODE_IQ2_FAST=0\n");
            return 1;
        }
        if (arm == 1 && (cnt.requests != 1u || cnt.encoded != 1u || cnt.failures != 0u)) {
            fprintf(stderr, "FAIL: expected requests=1 encoded=1 failures=0, got %llu/%llu/%llu\n",
                    (unsigned long long)cnt.requests, (unsigned long long)cnt.encoded,
                    (unsigned long long)cnt.failures);
            return 1;
        }

        myds_metal_buffer_free(b_out); myds_metal_buffer_free(b_x);
        myds_metal_buffer_free(b_gate); myds_metal_buffer_free(b_up); myds_metal_buffer_free(b_down);
        myds_metal_buffer_free(b_ids); myds_metal_buffer_free(b_w);
        myds_metal_buffer_free(s_g); myds_metal_buffer_free(s_u);
        myds_metal_buffer_free(s_m); myds_metal_buffer_free(s_d);
        myds_metal_cleanup();
    }

    int cmp = compare_f32_bits(out_ref, out_fast, dim, "routed_moe_e2e");
    free(host_gate); free(host_up); free(host_down);
    free(host_x); free(out_ref); free(out_fast);
    unsetenv("MYDS_DECODE_IQ2_FAST");
    unsetenv("MYDS_MOE_IQ2_POPCNT_SIGN");
    if (cmp != 0) return 1;
    printf("  PASS: end-to-end myds_metal_routed_moe bit-exact (memcmp == 0) + counters verified.\n");
    return 0;
}

int main(int argc, char **argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) do_bench = true;
    }
    @autoreleasepool {
        if (test_knob_and_fail_closed() != 0) return 1;
        if (test_direct_metal_parity_and_bench(do_bench) != 0) return 1;
        if (test_host_api_and_counters() != 0) return 1;
        printf("\n[ALL PASS] test_decode_iq2_fast: strict 0-ULP + PSO occupancy verified!\n");
        return 0;
    }
}
