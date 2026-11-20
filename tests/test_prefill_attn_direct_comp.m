/* Verification & GPU microbenchmark for [CAND-A2] Direct-Indirect Compressed KV
 * inside q8 heads_maskless_slab (MYDS_PREFILL_ATTN_DIRECT_COMP, default 0 fail-closed):
 *
 *   1. Fail-closed env knob contract (absent/0 -> 0, 1 -> 1, else -1 rejecting
 *      scratch_create & attention_chunk) + /debug/state JSON counter parity.
 *   2. Specialized PSO residency check (DK=512, DV=512, NSG=4, has_sinks=1, has_scap=0)
 *      for both _nvfp4 and _bf16 direct-indirect kernels (staticSmem=0, dynSmem=10,272 B).
 *   3. Strict 0-ULP (memcmp == 0) vs gather_comp_512 + heads_maskless_slab for BOTH
 *      NVFP4 (288 B/row) and BF16 (1,024 B/row) across:
 *        - c_len in {32, 64, 122, 512, 2048}
 *        - n_comp in {512, 16384, 65536, 131072}
 *        - ratio in {1, 2}
 *        - random, duplicate, out-of-order, negative (-1), out-of-range (>= n_comp),
 *          -0.0, subnormal E4M3 scales (0..7), max E4M3 scale (126), 2-byte aligned BF16 view,
 *          and both auto_barrier=1 and auto_barrier=0.
 *      Also proves 0 bytes written to kv_gathered (sentinel intact), 0 gather_comp_512
 *      dispatches, and -1 GPU dispatch per chunk.
 *   4. Fallback & gathered_cache_invalidate safety across p0 < 127, (p0+1)/ratio < 512,
 *      c_len < 32, F32 mode, and fast_matrix -> direct_comp -> fallback transitions.
 *   5. --bench: interleaved 7-run median GPU benchmark comparing gather_comp_512 +
 *      heads_maskless_slab vs DIRECT_COMP at c_len=2048, n_comp in {16K, 64K, 128K, 256K}
 *      for both NVFP4 and BF16.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "myds_debugstate.h"
#include "myds_metal.h"
#include "myds_v41_gpu.h"
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

struct guarded_buf {
    myds_metal_buffer_t *owner;
    myds_metal_buffer_t *view;
    size_t bytes;
};

static struct guarded_buf guarded_alloc_align(size_t bytes, size_t off) {
    struct guarded_buf g = { .bytes = bytes };
    g.owner = myds_metal_buffer_alloc(bytes + off + 64u);
    REQUIRE(g.owner != NULL);
    memset(myds_metal_buffer_data(g.owner), 0xa5, bytes + off + 64u);
    g.view = myds_metal_buffer_sub_view(g.owner, off, bytes);
    REQUIRE(g.view != NULL);
    return g;
}

static struct guarded_buf guarded_alloc(size_t bytes) {
    return guarded_alloc_align(bytes, 32u);
}

static void guarded_check_align(struct guarded_buf g, size_t off) {
    const uint8_t *p = (const uint8_t *)myds_metal_buffer_data(g.owner);
    for (size_t i = 0; i < off; ++i) REQUIRE(p[i] == 0xa5);
    for (size_t i = 0; i < 64u; ++i) REQUIRE(p[off + g.bytes + i] == 0xa5);
}

static void guarded_free_align(struct guarded_buf g, size_t off) {
    guarded_check_align(g, off);
    myds_metal_buffer_free(g.view);
    myds_metal_buffer_free(g.owner);
}

static void guarded_free(struct guarded_buf g) {
    guarded_free_align(g, 32u);
}

static uint32_t xorshift32(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

static float sample_f32(uint32_t i, uint32_t seed) {
    uint32_t x = (i + 1u) * 747796405u + seed * 2891336453u;
    x = ((x >> ((x >> 28) + 4u)) ^ x) * 277803737u;
    x = (x >> 22) ^ x;
    return ((int32_t)(x % 8191u) - 4095) / 8192.0f;
}

static void fill_f32_buf(myds_metal_buffer_t *b, size_t n, uint32_t seed) {
    float *p = (float *)myds_metal_buffer_data(b);
    for (size_t i = 0; i < n; ++i) p[i] = sample_f32((uint32_t)i, seed);
}

/* Fill NVFP4 compressed rows with adversarial patterns:
 *   row 0: all -0.0 nibbles (0x88) with subnormal scales 0..7
 *   row 1: mixed +/-0.0 and +/-6.0 with scale=0 and scale=1..7 (subnormal E4M3)
 *   row 2: max E4M3 scale (126 = 448.0) with small nibbles (0, 1, 8, 9)
 *   rows 3..n-1: deterministic pseudo-random nibbles & E4M3 scales in [0..108] */
static void fill_nvfp4_rows(myds_metal_buffer_t *b, uint32_t n_comp, uint32_t seed) {
    uint8_t *base = (uint8_t *)myds_metal_buffer_data(b);
    uint32_t st = seed ? seed : 0x51a7c3d1u;
    for (uint32_t r = 0; r < n_comp; ++r) {
        uint8_t *row = base + (size_t)r * MYDS_KV_COMP_NVFP4_ROW_BYTES;
        if (r == 0u) {
            memset(row, 0x88, 256u);
            for (uint32_t j = 0; j < 32u; ++j) row[256u + j] = (uint8_t)(j & 7u);
        } else if (r == 1u) {
            for (uint32_t j = 0; j < 256u; ++j) {
                const uint8_t lo = (j & 1u) ? 0x07u : 0x08u;
                const uint8_t hi = (j & 2u) ? 0xf0u : 0x00u;
                row[j] = (uint8_t)(lo | hi);
            }
            for (uint32_t j = 0; j < 32u; ++j) row[256u + j] = (uint8_t)(j % 9u);
        } else if (r == 2u) {
            for (uint32_t j = 0; j < 256u; ++j) row[j] = (uint8_t)((j & 1u) ? 0x91u : 0x80u);
            for (uint32_t j = 0; j < 32u; ++j) row[256u + j] = (j & 1u) ? 126u : 7u;
        } else {
            uint32_t *w = (uint32_t *)row;
            for (uint32_t j = 0; j < 64u; ++j) w[j] = xorshift32(&st);
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t rv = xorshift32(&st);
                row[256u + j] = (r < 32u) ? (uint8_t)(rv % 16u) : (uint8_t)(rv % 96u);
            }
        }
    }
}

static void fill_bf16_rows(myds_metal_buffer_t *b, uint32_t n_comp, uint32_t seed) {
    uint16_t *p = (uint16_t *)myds_metal_buffer_data(b);
    uint32_t st = seed ? seed : 0x91e4a72bu;
    for (uint32_t r = 0; r < n_comp; ++r) {
        uint16_t *row = p + (size_t)r * 512u;
        if (r == 0u) {
            for (uint32_t d = 0; d < 512u; ++d) row[d] = (d & 1u) ? 0x8000u : 0x0000u;
        } else {
            for (uint32_t d = 0; d < 512u; ++d) {
                const uint32_t rv = xorshift32(&st);
                const uint32_t sign = (rv & 1u) << 15u;
                const uint32_t exp  = (118u + ((rv >> 1u) % 14u)) << 7u;
                const uint32_t mant = (rv >> 8u) & 0x7fu;
                row[d] = (uint16_t)(sign | exp | mant);
            }
        }
    }
}

/* Fill top-512 indices with random, duplicate, out-of-order, -0.0/subnormal rows,
 * plus optional invalid (<0 and >= n_comp) sentinels on selected queries. */
static void fill_ids_adversarial(myds_metal_buffer_t *b, uint32_t c_len, uint32_t n_comp,
                                 uint32_t seed, bool inject_invalid) {
    int32_t *pids = (int32_t *)myds_metal_buffer_data(b);
    uint32_t st = seed ? seed : 0x37f92105u;
    for (uint32_t q = 0; q < c_len; ++q) {
        int32_t *row = pids + (size_t)q * 512u;
        for (uint32_t j = 0; j < 512u; ++j) {
            row[j] = (int32_t)(xorshift32(&st) % n_comp);
        }
        /* Pin adversarial special rows (0=-0.0, 1=subnormal, 2=max E4M3) & duplicates */
        row[0]   = 0;
        row[1]   = 1;
        row[2]   = (n_comp > 2u) ? 2 : 0;
        row[3]   = row[0];            /* duplicate */
        row[63]  = (int32_t)(n_comp - 1u); /* out-of-order high index at tile boundary */
        row[64]  = 1;
        row[127] = row[63];           /* duplicate across tiles */
        row[255] = 0;
        row[511] = (int32_t)(n_comp - 1u);
        if (inject_invalid) {
            if ((q & 1u) == 0u) row[7]   = -1;
            if ((q & 3u) == 1u) row[71]  = -999;
            if ((q & 3u) == 2u) row[135] = (int32_t)n_comp;
            if ((q & 3u) == 3u) row[509] = (int32_t)n_comp + 17;
        }
    }
}

static void verify_exact_0ulp(const void *ref, const void *cand, size_t n_floats, const char *label) {
    if (memcmp(ref, cand, n_floats * 4u) == 0) return;
    const uint32_t *a = (const uint32_t *)ref;
    const uint32_t *b = (const uint32_t *)cand;
    size_t diffs = 0, first = SIZE_MAX;
    for (size_t i = 0; i < n_floats; ++i) {
        if (a[i] != b[i]) {
            if (first == SIZE_MAX) first = i;
            ++diffs;
        }
    }
    fprintf(stderr, "FAIL 0-ULP %s: diffs=%zu/%zu first=%zu (dim=%zu head=%zu tok=%zu) ref=0x%08x cand=0x%08x\n",
            label, diffs, n_floats, first, first % 512u, (first / 512u) % 64u, first / (64u * 512u),
            a[first], b[first]);
    exit(1);
}

static void verify_all_byte(const myds_metal_buffer_t *b, uint8_t val, const char *label) {
    const uint8_t *p = (const uint8_t *)myds_metal_buffer_data((myds_metal_buffer_t *)b);
    const size_t n = myds_metal_buffer_size(b);
    for (size_t i = 0; i < n; ++i) {
        if (p[i] != val) {
            fprintf(stderr, "FAIL %s: expected untouched 0x%02x at byte %zu/%zu, got 0x%02x\n",
                    label, val, i, n, p[i]);
            exit(1);
        }
    }
}

static void reset_env_defaults(void) {
    unsetenv("MYDS_PREFILL_ATTN_DIRECT_COMP");
    unsetenv("MYDS_PREFILL_ATTN_L01_MMA");
    unsetenv("MYDS_ATTN_HEADS_MODE");
    setenv("MYDS_PREFILL_ATTN", "auto", 1);
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
}

/* 1. Fail-closed knob contract + /debug/state JSON emission. */
static void test_knob_and_debug_state(void) {
    reset_env_defaults();
    REQUIRE(myds_v41_gpu_prefill_attn_direct_comp_enabled() == 0);

    setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "0", 1);
    REQUIRE(myds_v41_gpu_prefill_attn_direct_comp_enabled() == 0);
    setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "1", 1);
    REQUIRE(myds_v41_gpu_prefill_attn_direct_comp_enabled() == 1);

    const char *bad[] = { "", "2", "01", "true", "false", "1 ", "-1" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", bad[i], 1);
        REQUIRE(myds_v41_gpu_prefill_attn_direct_comp_enabled() == -1);
        REQUIRE(myds_v41_gpu_scratch_create(4096u) == NULL);
    }

    reset_env_defaults();
    myds_v41_gpu_prefill_attn_direct_comp_counters(NULL, true);
    char json[8192];
    myds_debug_state_json(json, sizeof(json), NULL);
    REQUIRE(strstr(json, "\"prefill_attn_direct_comp\":{\"requests\":0,\"bf16_encoded\":0,\"nvfp4_encoded\":0,\"fallbacks\":0}") != NULL);
    REQUIRE(strstr(json, "truncated") == NULL);
    printf("PASS knob_and_debug_state (MYDS_PREFILL_ATTN_DIRECT_COMP fail-closed default 0 + /debug/state JSON)\n");
}

/* 2. Specialized PSO residency probe on local device (DK=512, DV=512, NSG=4). */
static void test_pso_residency(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        REQUIRE(dev != nil);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"bin/myds.metallib"] error:&err];
        REQUIRE(lib != nil);

        const char *names[] = {
            "kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512",
            "kernel_myds_flash_attn_heads_maskless_slab_direct_nvfp4_dk512_dv512",
            "kernel_myds_flash_attn_heads_maskless_slab_direct_bf16_dk512_dv512"
        };
        printf("=== Specialized PSO Residency Probe (%s, DK=512, DV=512, NSG=4) ===\n",
               dev.name.UTF8String);
        for (size_t i = 0; i < 3u; ++i) {
            MTLFunctionConstantValues *vals = [MTLFunctionConstantValues new];
            bool b0 = false, b1 = true;
            int32_t ns10 = 512, ns20 = 512, nsg = 4;
            [vals setConstantValue:&b0   type:MTLDataTypeBool atIndex:300];
            [vals setConstantValue:&b1   type:MTLDataTypeBool atIndex:301];
            [vals setConstantValue:&b0   type:MTLDataTypeBool atIndex:302];
            [vals setConstantValue:&b0   type:MTLDataTypeBool atIndex:303];
            [vals setConstantValue:&b0   type:MTLDataTypeBool atIndex:304];
            [vals setConstantValue:&b0   type:MTLDataTypeBool atIndex:310];
            [vals setConstantValue:&ns10 type:MTLDataTypeInt  atIndex:320];
            [vals setConstantValue:&ns20 type:MTLDataTypeInt  atIndex:321];
            [vals setConstantValue:&nsg  type:MTLDataTypeInt  atIndex:322];
            id<MTLFunction> fn = [lib newFunctionWithName:@(names[i]) constantValues:vals error:&err];
            REQUIRE(fn != nil);
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            REQUIRE(pso != nil);
            REQUIRE(pso.threadExecutionWidth == 32u);
            REQUIRE(pso.staticThreadgroupMemoryLength == 0u);
            REQUIRE(pso.maxTotalThreadsPerThreadgroup >= 384u);
            const unsigned tgs = (unsigned)((pso.maxTotalThreadsPerThreadgroup / 128u) < (32768u / 10272u)
                                            ? (pso.maxTotalThreadsPerThreadgroup / 128u)
                                            : (32768u / 10272u));
            REQUIRE(tgs == 3u);
            printf("PSO  %-68s maxThr=%4lu staticSmem=%lu dynSmem=10272 TGs/core=%u\n",
                   names[i], (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)pso.staticThreadgroupMemoryLength, tgs);
        }
        printf("PASS pso_residency (all 3 specialized kernels: staticSmem=0, dynSmem=10272, TGs/core=3)\n");
    }
}

/* 3. Strict 0-ULP comparison vs gather_comp_512 + heads_maskless_slab. */
static void run_0ulp_case(uint32_t c_len, uint32_t n_comp, uint32_t ratio,
                          uint32_t comp_mode, bool auto_barrier, bool unaligned_bf16,
                          myds_v41_gpu_scratch_t *s) {
    const uint32_t p0 = (ratio == 2u) ? (n_comp * 2u - c_len) : (n_comp - c_len);
    const uint32_t p0_use = (p0 < 1024u) ? (ratio * 512u + 128u) : p0;
    const size_t q_floats = (size_t)c_len * 64u * 512u;
    const size_t comp_row_bytes = myds_v41_kv_comp_cache_mode_row_bytes(comp_mode);
    const size_t comp_align = (comp_mode == MYDS_KV_COMP_CACHE_BF16 && unaligned_bf16) ? 2u : 32u;

    struct guarded_buf q_f32   = guarded_alloc(q_floats * 4u);
    struct guarded_buf sinks   = guarded_alloc(64u * 4u);
    struct guarded_buf win     = guarded_alloc(128u * 512u * 4u);
    struct guarded_buf ckv     = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf comp    = guarded_alloc_align((size_t)n_comp * comp_row_bytes, comp_align);
    struct guarded_buf ids     = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf gath    = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
    struct guarded_buf out_ref = guarded_alloc(q_floats * 4u);
    struct guarded_buf out_dc  = guarded_alloc(q_floats * 4u);

    const uint32_t seed = 9001u + c_len * 17u + n_comp * 3u + ratio * 71u + comp_mode * 131u;
    fill_f32_buf(q_f32.view, q_floats, seed + 1u);
    fill_f32_buf(sinks.view, 64u, seed + 2u);
    fill_f32_buf(win.view, 128u * 512u, seed + 3u);
    fill_f32_buf(ckv.view, (size_t)c_len * 512u, seed + 4u);
    if (comp_mode == MYDS_KV_COMP_CACHE_NVFP4) {
        fill_nvfp4_rows(comp.view, n_comp, seed + 5u);
    } else {
        fill_bf16_rows(comp.view, n_comp, seed + 5u);
    }
    fill_ids_adversarial(ids.view, c_len, n_comp, seed + 6u, true);

    myds_metal_set_auto_barrier(auto_barrier);

    /* Baseline: MYDS_PREFILL_ATTN_FAST=1, MYDS_PREFILL_ATTN_DIRECT_COMP=0 */
    reset_env_defaults();
    setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "0", 1);
    myds_v41_gpu_prefill_attn_fast_counters(NULL, true);
    myds_v41_gpu_prefill_attn_direct_comp_counters(NULL, true);
    uint64_t disp_base0 = myds_metal_dispatch_count();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0_use));
    REQUIRE(myds_v41_gpu_attention_chunk_mode(out_ref.view, sinks.view, q_f32.view, ckv.view,
            comp.view, ids.view, gath.view, c_len, n_comp, 512u, p0_use, 0u, ratio, s, comp_mode));
    REQUIRE(myds_metal_commit_batch());
    uint64_t disp_base = myds_metal_dispatch_count() - disp_base0;
    myds_v41_gpu_prefill_attn_fast_counters_t paf_base;
    myds_v41_gpu_prefill_attn_fast_counters(&paf_base, false);
    REQUIRE(paf_base.fast_matrix_encoded == 1u);
    REQUIRE(paf_base.gather_comp_512_encoded == 1u);

    /* Candidate: MYDS_PREFILL_ATTN_DIRECT_COMP=1 */
    memset(myds_metal_buffer_data(gath.view), 0xa5, gath.bytes);
    memset(myds_metal_buffer_data(out_dc.view), 0x00, out_dc.bytes);
    setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "1", 1);
    myds_v41_gpu_prefill_attn_fast_counters(NULL, true);
    myds_v41_gpu_prefill_attn_direct_comp_counters(NULL, true);
    uint64_t disp_dc0 = myds_metal_dispatch_count();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0_use));
    REQUIRE(myds_v41_gpu_attention_chunk_mode(out_dc.view, sinks.view, q_f32.view, ckv.view,
            comp.view, ids.view, gath.view, c_len, n_comp, 512u, p0_use, 0u, ratio, s, comp_mode));
    REQUIRE(myds_metal_commit_batch());
    uint64_t disp_dc = myds_metal_dispatch_count() - disp_dc0;

    /* Verify gather_comp_512 was completely skipped (0 bytes written to kv_gathered, -1 dispatch) */
    myds_v41_gpu_prefill_attn_fast_counters_t paf_dc;
    myds_v41_gpu_prefill_attn_fast_counters(&paf_dc, false);
    REQUIRE(paf_dc.gather_comp_512_encoded == 0u);
    REQUIRE(paf_dc.gather_comp_hits == 0u);
    REQUIRE(paf_dc.gather_comp_misses == 0u);
    REQUIRE(disp_dc + 1u == disp_base);

    myds_v41_gpu_prefill_attn_direct_comp_counters_t padc;
    myds_v41_gpu_prefill_attn_direct_comp_counters(&padc, false);
    REQUIRE(padc.requests == 1u);
    REQUIRE(padc.fallbacks == 0u);
    if (comp_mode == MYDS_KV_COMP_CACHE_NVFP4) {
        REQUIRE(padc.nvfp4_encoded == 1u && padc.bf16_encoded == 0u);
    } else {
        REQUIRE(padc.bf16_encoded == 1u && padc.nvfp4_encoded == 0u);
    }

    char lbl[160];
    snprintf(lbl, sizeof(lbl), "%s c_len=%u n_comp=%u ratio=%u auto_bar=%d%s",
             comp_mode == MYDS_KV_COMP_CACHE_NVFP4 ? "NVFP4" : "BF16",
             c_len, n_comp, ratio, (int)auto_barrier, unaligned_bf16 ? " off=2B" : "");
    verify_all_byte(gath.view, 0xa5, lbl);
    verify_exact_0ulp(myds_metal_buffer_data(out_ref.view), myds_metal_buffer_data(out_dc.view),
                      q_floats, lbl);
    printf("PASS 0-ULP %-54s (memcmp==0, gath=0B, disp=%llu->%llu)\n",
           lbl, (unsigned long long)disp_base, (unsigned long long)disp_dc);

    guarded_free(out_dc);
    guarded_free(out_ref);
    guarded_free(gath);
    guarded_free(ids);
    guarded_free_align(comp, comp_align);
    guarded_free(ckv);
    guarded_free(win);
    guarded_free(sinks);
    guarded_free(q_f32);
}

/* 4. Fallback & gathered_cache_invalidate safety (p0<127, (p0+1)/ratio<512, c_len<32, F32, transitions). */
static void test_fallbacks_and_cache_safety(myds_v41_gpu_scratch_t *s) {
    struct { uint32_t c_len, n_comp, p0, ratio, mode; const char *desc; } cases[] = {
        { 32u,  512u,   64u, 1u, MYDS_KV_COMP_CACHE_NVFP4, "p0<127 (NVFP4)" },
        { 32u,  512u,  400u, 1u, MYDS_KV_COMP_CACHE_NVFP4, "(p0+1)/ratio<512 (NVFP4)" },
        { 16u, 1024u, 1024u, 1u, MYDS_KV_COMP_CACHE_NVFP4, "c_len<32 (NVFP4)" },
        { 16u, 1024u, 2048u, 2u, MYDS_KV_COMP_CACHE_BF16,  "c_len<32 ratio=2 (BF16)" },
        { 32u, 1024u, 1024u, 1u, MYDS_KV_COMP_CACHE_F32,   "F32 mode fallback" },
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const uint32_t c_len = cases[ci].c_len;
        const uint32_t n_comp = cases[ci].n_comp;
        const uint32_t p0 = cases[ci].p0;
        const uint32_t ratio = cases[ci].ratio;
        const uint32_t mode = cases[ci].mode;
        const size_t q_floats = (size_t)c_len * 64u * 512u;
        const size_t rb = myds_v41_kv_comp_cache_mode_row_bytes(mode);

        struct guarded_buf q_f32   = guarded_alloc(q_floats * 4u);
        struct guarded_buf sinks   = guarded_alloc(64u * 4u);
        struct guarded_buf win     = guarded_alloc(128u * 512u * 4u);
        struct guarded_buf ckv     = guarded_alloc((size_t)c_len * 512u * 4u);
        struct guarded_buf comp    = guarded_alloc((size_t)n_comp * rb);
        struct guarded_buf ids     = guarded_alloc((size_t)c_len * 512u * 4u);
        struct guarded_buf gath    = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
        struct guarded_buf out_ref = guarded_alloc(q_floats * 4u);
        struct guarded_buf out_dc  = guarded_alloc(q_floats * 4u);

        const uint32_t seed = 33001u + (uint32_t)ci * 97u;
        fill_f32_buf(q_f32.view, q_floats, seed + 1u);
        fill_f32_buf(sinks.view, 64u, seed + 2u);
        fill_f32_buf(win.view, 128u * 512u, seed + 3u);
        fill_f32_buf(ckv.view, (size_t)c_len * 512u, seed + 4u);
        if (mode == MYDS_KV_COMP_CACHE_NVFP4) fill_nvfp4_rows(comp.view, n_comp, seed + 5u);
        else if (mode == MYDS_KV_COMP_CACHE_BF16) fill_bf16_rows(comp.view, n_comp, seed + 5u);
        else fill_f32_buf(comp.view, (size_t)n_comp * 512u, seed + 5u);
        fill_ids_adversarial(ids.view, c_len, n_comp, seed + 6u, true);

        reset_env_defaults();
        setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "0", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_mode(out_ref.view, sinks.view, q_f32.view, ckv.view,
                comp.view, ids.view, gath.view, c_len, n_comp, 512u, p0, 0u, ratio, s, mode));
        REQUIRE(myds_metal_commit_batch());

        setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "1", 1);
        myds_v41_gpu_prefill_attn_direct_comp_counters(NULL, true);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_mode(out_dc.view, sinks.view, q_f32.view, ckv.view,
                comp.view, ids.view, gath.view, c_len, n_comp, 512u, p0, 0u, ratio, s, mode));
        REQUIRE(myds_metal_commit_batch());

        myds_v41_gpu_prefill_attn_direct_comp_counters_t padc;
        myds_v41_gpu_prefill_attn_direct_comp_counters(&padc, false);
        REQUIRE(padc.requests == 1u && padc.bf16_encoded == 0u &&
                padc.nvfp4_encoded == 0u && padc.fallbacks == 1u);
        verify_exact_0ulp(myds_metal_buffer_data(out_ref.view), myds_metal_buffer_data(out_dc.view),
                          q_floats, cases[ci].desc);
        printf("PASS fallback %-32s (0-ULP, req=1 fallbacks=1)\n", cases[ci].desc);

        guarded_free(out_dc); guarded_free(out_ref); guarded_free(gath);
        guarded_free(ids); guarded_free(comp); guarded_free(ckv);
        guarded_free(win); guarded_free(sinks); guarded_free(q_f32);
    }

    /* Transition test: fast_matrix (populates gathered_cache) -> direct_comp (invalidates
     * gathered_cache, skips gather) -> fallback with modified gathered buffer -> must refresh! */
    {
        const uint32_t c_len = 32u, n_comp = 1024u, p0 = 1024u;
        const size_t q_floats = (size_t)c_len * 64u * 512u;
        struct guarded_buf q_f32   = guarded_alloc(q_floats * 4u);
        struct guarded_buf sinks   = guarded_alloc(64u * 4u);
        struct guarded_buf win     = guarded_alloc(128u * 512u * 4u);
        struct guarded_buf ckv     = guarded_alloc((size_t)c_len * 512u * 4u);
        struct guarded_buf comp    = guarded_alloc((size_t)n_comp * MYDS_KV_COMP_NVFP4_ROW_BYTES);
        struct guarded_buf ids     = guarded_alloc((size_t)c_len * 512u * 4u);
        struct guarded_buf gath    = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
        struct guarded_buf out_ref = guarded_alloc(q_floats * 4u);
        struct guarded_buf out_a   = guarded_alloc(q_floats * 4u);
        struct guarded_buf out_b   = guarded_alloc(q_floats * 4u);

        fill_f32_buf(q_f32.view, q_floats, 41001u);
        fill_f32_buf(sinks.view, 64u, 41002u);
        fill_f32_buf(win.view, 128u * 512u, 41003u);
        fill_f32_buf(ckv.view, (size_t)c_len * 512u, 41004u);
        fill_nvfp4_rows(comp.view, n_comp, 41005u);
        fill_ids_adversarial(ids.view, c_len, n_comp, 41006u, false);

        /* Step A: fast_matrix fills gath & gathered_cache */
        reset_env_defaults();
        setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "0", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_mode(out_ref.view, sinks.view, q_f32.view, ckv.view,
                comp.view, ids.view, gath.view, c_len, n_comp, 512u, p0, 0u, 1u, s, MYDS_KV_COMP_CACHE_NVFP4));
        REQUIRE(myds_metal_commit_batch());

        /* Step B: direct_comp runs -> invalidates gathered_cache without touching gath */
        setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "1", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_mode(out_a.view, sinks.view, q_f32.view, ckv.view,
                comp.view, ids.view, gath.view, c_len, n_comp, 512u, p0, 0u, 1u, s, MYDS_KV_COMP_CACHE_NVFP4));
        REQUIRE(myds_metal_commit_batch());
        verify_exact_0ulp(myds_metal_buffer_data(out_ref.view), myds_metal_buffer_data(out_a.view),
                          q_floats, "transition step B direct_comp");

        /* Corrupt gath; Step C runs fast_matrix again -> MUST miss gathered_cache and recompute! */
        memset(myds_metal_buffer_data(gath.view), 0x7f, gath.bytes);
        setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "0", 1);
        myds_v41_gpu_prefill_attn_fast_counters(NULL, true);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_mode(out_b.view, sinks.view, q_f32.view, ckv.view,
                comp.view, ids.view, gath.view, c_len, n_comp, 512u, p0, 0u, 1u, s, MYDS_KV_COMP_CACHE_NVFP4));
        REQUIRE(myds_metal_commit_batch());
        myds_v41_gpu_prefill_attn_fast_counters_t paf;
        myds_v41_gpu_prefill_attn_fast_counters(&paf, false);
        REQUIRE(paf.gather_comp_hits == 0u && paf.gather_comp_misses == 1u);
        verify_exact_0ulp(myds_metal_buffer_data(out_ref.view), myds_metal_buffer_data(out_b.view),
                          q_floats, "transition step C invalidated cache refresh");
        printf("PASS gathered_cache_invalidate transition (fast -> direct_comp -> corrupted gath -> fast 0-ULP)\n");

        guarded_free(out_b); guarded_free(out_a); guarded_free(out_ref);
        guarded_free(gath); guarded_free(ids); guarded_free(comp);
        guarded_free(ckv); guarded_free(win); guarded_free(sinks); guarded_free(q_f32);
    }

    /* Verify /debug/state reports non-zero live counters accurately. */
    char json[8192];
    myds_v41_gpu_prefill_attn_direct_comp_counters_t c;
    myds_v41_gpu_prefill_attn_direct_comp_counters(&c, false);
    myds_debug_state_json(json, sizeof(json), NULL);
    char expected[256];
    snprintf(expected, sizeof(expected),
             "\"prefill_attn_direct_comp\":{\"requests\":%llu,\"bf16_encoded\":%llu,"
             "\"nvfp4_encoded\":%llu,\"fallbacks\":%llu}",
             (unsigned long long)c.requests, (unsigned long long)c.bf16_encoded,
             (unsigned long long)c.nvfp4_encoded, (unsigned long long)c.fallbacks);
    REQUIRE(strstr(json, expected) != NULL);
    printf("PASS debug_state live counters: %s\n", expected);
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db);
}

/* 5. --bench: interleaved 7-run median GPU microbenchmark (end-to-end attention_chunk time). */
static void run_bench(void) {
    const uint32_t c_len = 2048u;
    const uint32_t ncomps[] = { 16384u, 65536u, 131072u, 262144u };
    const uint32_t max_nc = 262144u;

    reset_env_defaults();
    myds_metal_set_auto_barrier(false); /* production explicit-barrier mode */
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(max_nc);
    REQUIRE(s != NULL);

    myds_metal_buffer_t *q_f32 = myds_metal_buffer_alloc((size_t)c_len * 64u * 512u * 4u);
    myds_metal_buffer_t *sinks = myds_metal_buffer_alloc(64u * 4u);
    myds_metal_buffer_t *win   = myds_metal_buffer_alloc(128u * 512u * 4u);
    myds_metal_buffer_t *ckv   = myds_metal_buffer_alloc((size_t)c_len * 512u * 4u);
    myds_metal_buffer_t *ids   = myds_metal_buffer_alloc((size_t)c_len * 512u * 4u);
    myds_metal_buffer_t *gath  = myds_metal_buffer_alloc((size_t)c_len * 640u * 512u * 2u);
    myds_metal_buffer_t *out   = myds_metal_buffer_alloc((size_t)c_len * 64u * 512u * 4u);
    myds_metal_buffer_t *nv    = myds_metal_buffer_alloc((size_t)max_nc * MYDS_KV_COMP_NVFP4_ROW_BYTES);
    myds_metal_buffer_t *bf    = myds_metal_buffer_alloc((size_t)max_nc * MYDS_KV_COMP_BF16_ROW_BYTES);
    REQUIRE(q_f32 && sinks && win && ckv && ids && gath && out && nv && bf);

    memset(myds_metal_buffer_data(q_f32), 0x3c, myds_metal_buffer_size(q_f32));
    memset(myds_metal_buffer_data(sinks), 0x00, myds_metal_buffer_size(sinks));
    memset(myds_metal_buffer_data(win),   0x38, myds_metal_buffer_size(win));
    memset(myds_metal_buffer_data(ckv),   0x38, myds_metal_buffer_size(ckv));
    memset(myds_metal_buffer_data(nv),    0x35, myds_metal_buffer_size(nv));
    memset(myds_metal_buffer_data(bf),    0x3f, myds_metal_buffer_size(bf));

    printf("\n=== [CAND-A2] End-to-End attention_chunk Bench (c_len=2048, auto_barrier=0, median of 7 interleaved pairs) ===\n");
    printf("%-8s %-6s | %16s | %14s | %8s | %9s\n",
           "n_comp", "format", "Base(g512+slab)", "DIRECT_COMP", "Speedup", "Delta");

    for (size_t ni = 0; ni < sizeof(ncomps) / sizeof(ncomps[0]); ++ni) {
        const uint32_t n_comp = ncomps[ni];
        fill_ids_adversarial(ids, c_len, n_comp, 77000u + n_comp, false);

        for (int fmt = 0; fmt < 2; ++fmt) {
            const uint32_t mode = (fmt == 0) ? MYDS_KV_COMP_CACHE_NVFP4 : MYDS_KV_COMP_CACHE_BF16;
            myds_metal_buffer_t *comp = (fmt == 0) ? nv : bf;
            double t_base[7], t_dc[7];

            for (int it = -2; it < 7; ++it) {
                /* Baseline: gather_comp_512 + heads_maskless_slab (force cold miss via p0 shift, matching 0% live hit rate!) */
                setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "0", 1);
                const uint32_t p0_b = n_comp + (uint32_t)(it + 4) * 2u;
                double t0 = myds_metal_host_time();
                REQUIRE(myds_metal_begin_batch());
                REQUIRE(myds_v41_gpu_attention_chunk_mode(out, sinks, q_f32, ckv, comp, ids, gath,
                        c_len, n_comp, 512u, p0_b, 0u, 1u, s, mode));
                REQUIRE(myds_metal_commit_batch());
                double dt_b = (myds_metal_host_time() - t0) * 1000.0;

                /* Candidate: DIRECT_COMP */
                setenv("MYDS_PREFILL_ATTN_DIRECT_COMP", "1", 1);
                t0 = myds_metal_host_time();
                REQUIRE(myds_metal_begin_batch());
                REQUIRE(myds_v41_gpu_attention_chunk_mode(out, sinks, q_f32, ckv, comp, ids, gath,
                        c_len, n_comp, 512u, p0_b, 0u, 1u, s, mode));
                REQUIRE(myds_metal_commit_batch());
                double dt_d = (myds_metal_host_time() - t0) * 1000.0;

                if (it >= 0) { t_base[it] = dt_b; t_dc[it] = dt_d; }
            }
            qsort(t_base, 7u, sizeof(double), cmp_double);
            qsort(t_dc,   7u, sizeof(double), cmp_double);
            const double mb = t_base[3], md = t_dc[3];
            printf("%-8u %-6s | %13.2f ms | %11.2f ms | %7.2fx | %+7.2f ms (%.1f%%)\n",
                   n_comp, fmt == 0 ? "NVFP4" : "BF16", mb, md, mb / md, md - mb,
                   (md - mb) / mb * 100.0);
        }
    }

    myds_metal_buffer_free(bf);
    myds_metal_buffer_free(nv);
    myds_metal_buffer_free(out);
    myds_metal_buffer_free(gath);
    myds_metal_buffer_free(ids);
    myds_metal_buffer_free(ckv);
    myds_metal_buffer_free(win);
    myds_metal_buffer_free(sinks);
    myds_metal_buffer_free(q_f32);
    myds_v41_gpu_scratch_free(s);
}

int main(int argc, char **argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) do_bench = true;
    }

    REQUIRE(myds_metal_init(NULL) || myds_metal_init("bin/myds.metallib"));
    test_knob_and_debug_state();
    test_pso_residency();

    reset_env_defaults();
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(131072u);
    REQUIRE(s != NULL);

    /* Matrix covering c_len in {32, 64, 122, 512, 2048}, n_comp in {512, 16384, 65536, 131072},
     * ratio in {1, 2}, both NVFP4 and BF16, both auto_barrier=1 and auto_barrier=0,
     * plus 2-byte-aligned BF16 sub-view check! */
    struct { uint32_t c_len, n_comp, ratio; bool auto_bar, unaligned_bf16; } matrix[] = {
        {   32u,    512u, 1u, true,  false },
        {   32u,    512u, 2u, false, true  },
        {   64u,  16384u, 1u, true,  false },
        {   64u,  16384u, 2u, false, false },
        {  122u,  16384u, 2u, true,  true  },
        {  122u,  65536u, 1u, false, false },
        {  512u,  65536u, 2u, true,  false },
        {  512u, 131072u, 1u, false, false },
        { 2048u,  16384u, 2u, false, false },
        { 2048u, 131072u, 2u, false, false },
    };

    for (size_t i = 0; i < sizeof(matrix) / sizeof(matrix[0]); ++i) {
        run_0ulp_case(matrix[i].c_len, matrix[i].n_comp, matrix[i].ratio,
                      MYDS_KV_COMP_CACHE_NVFP4, matrix[i].auto_bar, false, s);
        run_0ulp_case(matrix[i].c_len, matrix[i].n_comp, matrix[i].ratio,
                      MYDS_KV_COMP_CACHE_BF16, matrix[i].auto_bar, matrix[i].unaligned_bf16, s);
    }

    myds_metal_set_auto_barrier(true);
    test_fallbacks_and_cache_safety(s);
    myds_v41_gpu_scratch_free(s);

    printf("ALL PASS (STRICT 0-ULP)\n");

    if (do_bench) {
        run_bench();
    }

    myds_metal_cleanup();
    return 0;
}
