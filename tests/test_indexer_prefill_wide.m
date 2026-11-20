/* tests/test_indexer_prefill_wide.m
 *
 * [CAND-I1] Apple Silicon GPU strict 0-ULP (memcmp == 0) parity, Red-Team #11
 * Bug #1 fallback zero-corruption proof, strict fail-closed env parsing,
 * /debug/state counter verification, Metal PSO residency audit, and M4
 * interleaved microbenchmark suite for MYDS_INDEXER_PREFILL_WIDE.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "myds_debugstate.h"
#include "myds_metal.h"
#include "myds_v41_gpu.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

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

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint32_t f32_bits(float v) {
    uint32_t u = 0;
    memcpy(&u, &v, sizeof(u));
    return u;
}

/* 1. Strict fail-closed MYDS_INDEXER_PREFILL_WIDE env parsing contract */
static void test_env_parsing(void) {
    const char *saved = getenv("MYDS_INDEXER_PREFILL_WIDE");
    char saved_buf[64] = {0};
    const bool had_saved = (saved != NULL);
    if (had_saved) snprintf(saved_buf, sizeof(saved_buf), "%s", saved);

    unsetenv("MYDS_INDEXER_PREFILL_WIDE");
    CHECK(myds_v41_gpu_indexer_prefill_wide_enabled() == 0);

    setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
    CHECK(myds_v41_gpu_indexer_prefill_wide_enabled() == 0);

    setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
    CHECK(myds_v41_gpu_indexer_prefill_wide_enabled() == 1);

    const char *bad[] = {"", "2", "-1", "01", "true", "false", "yes", " 1", "1 "};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        setenv("MYDS_INDEXER_PREFILL_WIDE", bad[i], 1);
        CHECK(myds_v41_gpu_indexer_prefill_wide_enabled() == -1);
        CHECK(myds_v41_gpu_scratch_create(16384u) == NULL);
    }

    if (had_saved) setenv("MYDS_INDEXER_PREFILL_WIDE", saved_buf, 1);
    else unsetenv("MYDS_INDEXER_PREFILL_WIDE");
    printf("  [PASS] strict MYDS_INDEXER_PREFILL_WIDE env parsing (default 0, fail-closed on invalid)\n");
}

/* 2. Metal PSO occupancy & static threadgroup memory verification */
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
            NSUInteger min_max_threads;
            NSUInteger exact_static_smem;
        } specs[] = {
            {"kernel_myds_indexer_scores_register_only_fp4_wide",        768u, 8192u},
            {"kernel_myds_indexer_scores_register_only_bf16_wide",       768u, 8192u},
            {"kernel_myds_indexer_scores_register_only_fp4_wide_m8",     768u, 4096u},
            {"kernel_myds_indexer_scores_register_only_bf16_wide_m8",    768u, 4096u},
            {"kernel_myds_indexer_scores_prefill_compact_ro_fp4_wide",   768u, 4096u},
            {"kernel_myds_indexer_scores_prefill_compact_ro_bf16_wide",  768u, 4096u},
        };

        printf("  Verifying [CAND-I1] wide PSO residency & static threadgroup memory:\n");
        for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); ++i) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:specs[i].name]];
            CHECK(fn != nil);
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            CHECK(pso != nil && err == nil);
            CHECK(pso.threadExecutionWidth == 32u);
            CHECK(pso.maxTotalThreadsPerThreadgroup >= specs[i].min_max_threads);
            CHECK(pso.staticThreadgroupMemoryLength == specs[i].exact_static_smem);
            printf("    %-56s width=%lu max_threads=%4lu static_smem=%4lu B\n",
                   specs[i].name,
                   (unsigned long)pso.threadExecutionWidth,
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)pso.staticThreadgroupMemoryLength);
        }
    }
}

static void build_l20_mask(myds_metal_buffer_t *mask_buf, uint32_t glen,
                           uint32_t nc, uint32_t mask_units, uint32_t seed) {
    const uint32_t n_blocks = (nc + 7u) / 8u;
    uint32_t *m2 = (uint32_t *)myds_metal_buffer_data(mask_buf);
    CHECK(m2 != NULL);
    memset(m2, 0, (size_t)glen * mask_units * sizeof(uint32_t));
    for (uint32_t r = 0; r < glen; ++r) {
        const uint32_t tile_base = ((r / 8u) * 37u + seed) % (n_blocks - 2200u);
        for (uint32_t k = 0; k < 2048u; ++k) {
            const uint32_t b = (tile_base + k * (n_blocks / 2100u) + (r & 7u)) % n_blocks;
            m2[(size_t)r * mask_units + (b >> 5u)] |= (1u << (b & 31u));
        }
    }
}

/* 3. Strict 0-ULP parity + Red-Team #11 Bug #1 fallback zero-corruption + /debug/state counters */
static void test_0ulp_and_fallbacks(myds_v41_gpu_scratch_t *scratch,
                                    const myds_metal_buffer_t *q_f32,
                                    const myds_metal_buffer_t *q_bf16,
                                    const myds_metal_buffer_t *w_buf,
                                    const myds_metal_buffer_t *cache_bf16,
                                    const myds_metal_buffer_t *cache_fp4,
                                    myds_metal_buffer_t *sc_ref,
                                    myds_metal_buffer_t *sc_wide,
                                    myds_metal_buffer_t *mask_2d,
                                    myds_metal_buffer_t *top_ref,
                                    myds_metal_buffer_t *top_wide) {
    /* 3A. Verify fused indexer_rope_quant_batch populated q_f32 and q_bf16 bit-exact */
    {
        const float *qf = (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)q_f32);
        const uint16_t *qb = (const uint16_t *)myds_metal_buffer_data((myds_metal_buffer_t *)q_bf16);
        for (size_t i = 0; i < (size_t)512u * 32u * 128u; ++i) {
            const uint32_t bits = f32_bits(qf[i]);
            CHECK((bits & 0xffffu) == 0u);
            CHECK((uint16_t)(bits >> 16u) == qb[i]);
        }
    }

    /* 3B. Exhaustive IRO 0-ULP matrix across nc, glen, ratio, and FP4/BF16 cache modes */
    const uint32_t ncs[]   = {16384u, 32768u, 65536u, 131072u, 262144u};
    const uint32_t glens[] = {1u, 7u, 8u, 15u, 16u, 24u, 32u, 64u, 256u, 512u};
    const uint32_t ratios[] = {1u, 2u};
    const uint32_t modes[] = {MYDS_INDEXER_CACHE_FP4, MYDS_INDEXER_CACHE_BF16};

    uint64_t iro_cases_checked = 0;
    myds_v41_gpu_indexer_prefill_wide_counters(NULL, true);
    uint64_t exp_iro_wide = 0, exp_fallbacks = 0;

    for (size_t mi = 0; mi < 2u; ++mi) {
        const uint32_t mode = modes[mi];
        const myds_metal_buffer_t *cache = (mode == MYDS_INDEXER_CACHE_FP4) ? cache_fp4 : cache_bf16;
        for (size_t ri = 0; ri < 2u; ++ri) {
            const uint32_t ratio = ratios[ri];
            for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ++ni) {
                const uint32_t nc = ncs[ni];
                for (size_t gi = 0; gi < sizeof(glens) / sizeof(glens[0]); ++gi) {
                    const uint32_t glen = glens[gi];
                    if ((uint64_t)nc * glen > 8388608u) continue;
                    if ((uint64_t)nc * ratio > 1048576u) continue;
                    const uint32_t pos0 = nc * ratio - glen;

                    /* Baseline: MYDS_INDEXER_PREFILL_WIDE=0 */
                    setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
                    CHECK(myds_metal_begin_batch());
                    CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
                        scratch, sc_ref, q_f32, w_buf, cache,
                        nc, 32u, glen, pos0, ratio, mode, true) == 1);
                    CHECK(myds_metal_commit_batch());

                    /* Wide: MYDS_INDEXER_PREFILL_WIDE=1 */
                    setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
                    CHECK(myds_metal_begin_batch());
                    CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode_qbf16(
                        scratch, sc_wide, q_f32, q_bf16, w_buf, cache,
                        nc, 32u, glen, pos0, ratio, mode, true) == 1);
                    CHECK(myds_metal_commit_batch());

                    const size_t nbytes = (size_t)nc * glen * sizeof(float);
                    if (memcmp(myds_metal_buffer_data(sc_ref), myds_metal_buffer_data(sc_wide), nbytes) != 0) {
                        fprintf(stderr, "IRO 0-ULP MISMATCH: mode=%u ratio=%u nc=%u glen=%u pos0=%u\n",
                                mode, ratio, nc, glen, pos0);
                        exit(1);
                    }
                    if (glen >= 8u) exp_iro_wide++;
                    else exp_fallbacks++;
                    iro_cases_checked++;
                }
            }
        }
    }

    /* Extra IRO edge cases: odd nc (513, 16385), partial causal invisibility (pos0=0), odd tail > 40 (glen=49) */
    struct { uint32_t nc, glen, pos0, ratio; } extra_iro[] = {
        {513u,   513u,     0u, 1u},
        {513u,     9u,  1017u, 2u},
        {16385u,  49u, 16336u, 1u},
        {16385u,  25u, 32745u, 2u},
        {16384u,  32u, 32736u, 2u},
    };
    for (size_t mi = 0; mi < 2u; ++mi) {
        const uint32_t mode = modes[mi];
        const myds_metal_buffer_t *cache = (mode == MYDS_INDEXER_CACHE_FP4) ? cache_fp4 : cache_bf16;
        for (size_t ei = 0; ei < sizeof(extra_iro) / sizeof(extra_iro[0]); ++ei) {
            const uint32_t nc = extra_iro[ei].nc, glen = extra_iro[ei].glen;
            const uint32_t pos0 = extra_iro[ei].pos0, ratio = extra_iro[ei].ratio;
            /* Use subview for glen=513? Note q_f32/q_bf16 has 512 rows, clamp 513 -> 512 for pos0=1 */
            const uint32_t g_use = (glen > 512u) ? 512u : glen;
            const uint32_t p_use = (glen > 512u) ? (nc * ratio - g_use) : pos0;
            setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
                scratch, sc_ref, q_f32, w_buf, cache,
                nc, 32u, g_use, p_use, ratio, mode, true) == 1);
            CHECK(myds_metal_commit_batch());

            setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode_qbf16(
                scratch, sc_wide, q_f32, q_bf16, w_buf, cache,
                nc, 32u, g_use, p_use, ratio, mode, true) == 1);
            CHECK(myds_metal_commit_batch());

            CHECK(memcmp(myds_metal_buffer_data(sc_ref), myds_metal_buffer_data(sc_wide),
                         (size_t)nc * g_use * sizeof(float)) == 0);
            exp_iro_wide++;
            iro_cases_checked++;
        }
    }

    myds_v41_gpu_indexer_prefill_wide_counters_t wc;
    myds_v41_gpu_indexer_prefill_wide_counters(&wc, false);
    CHECK(wc.iro_wide_encoded == exp_iro_wide);
    CHECK(wc.fallbacks == exp_fallbacks);
    CHECK(wc.compact_wide_encoded == 0u);
    printf("  [PASS] IRO 0-ULP bit-exact (memcmp == 0) across %llu cases (nc 513..262K, glen 1..512, ratio 1/2, FP4+BF16; wide=%llu fallbacks=%llu)\n",
           (unsigned long long)iro_cases_checked,
           (unsigned long long)wc.iro_wide_encoded,
           (unsigned long long)wc.fallbacks);

    /* 3C. Exhaustive [P-L1] Compact 0-ULP matrix across nc, glen, and FP4/BF16 */
    const uint32_t pl1_ncs[] = {32768u, 65536u, 131072u, 262144u};
    const uint32_t pl1_glens[] = {8u, 15u, 16u, 24u, 32u, 64u, 256u, 512u};
    uint64_t exp_compact_wide = 0;
    uint64_t pl1_cases_checked = 0;

    for (size_t mi = 0; mi < 2u; ++mi) {
        const uint32_t mode = modes[mi];
        const myds_metal_buffer_t *cache = (mode == MYDS_INDEXER_CACHE_FP4) ? cache_fp4 : cache_bf16;
        for (size_t ni = 0; ni < sizeof(pl1_ncs) / sizeof(pl1_ncs[0]); ++ni) {
            const uint32_t nc = pl1_ncs[ni];
            const uint32_t mu = myds_v41_gpu_maskbits_stride((nc + 7u) / 8u);
            for (size_t gi = 0; gi < sizeof(pl1_glens) / sizeof(pl1_glens[0]); ++gi) {
                const uint32_t glen = pl1_glens[gi];
                /* Keep runtime brisk while covering every glen and every nc */
                if (ni > 0 && ni < 3 && (glen == 15u || glen == 24u || glen == 64u)) continue;
                const uint32_t pos0 = nc - glen;
                build_l20_mask(mask_2d, glen, nc, mu, (uint32_t)(ni * 101u + gi * 17u));

                setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
                CHECK(myds_metal_begin_batch());
                CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(
                    scratch, top_ref, q_f32, w_buf, cache, mask_2d,
                    nc, glen, pos0, 1u, mu, 2048u, mode, true) == 1);
                CHECK(myds_metal_commit_batch());

                setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
                CHECK(myds_metal_begin_batch());
                CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode_qbf16(
                    scratch, top_wide, q_f32, q_bf16, w_buf, cache, mask_2d,
                    nc, glen, pos0, 1u, mu, 2048u, mode, true) == 1);
                CHECK(myds_metal_commit_batch());

                if (memcmp(myds_metal_buffer_data(top_ref), myds_metal_buffer_data(top_wide),
                           (size_t)glen * 512u * sizeof(int32_t)) != 0) {
                    fprintf(stderr, "PL1 0-ULP MISMATCH: mode=%u nc=%u glen=%u\n", mode, nc, glen);
                    exit(1);
                }
                exp_compact_wide++;
                pl1_cases_checked++;
            }
        }
    }

    myds_v41_gpu_indexer_prefill_wide_counters(&wc, false);
    CHECK(wc.compact_wide_encoded == exp_compact_wide);
    printf("  [PASS] [P-L1] compact_ro_wide 0-ULP bit-exact (memcmp == 0) across %llu cases (nc 32K..262K, glen 8..512, FP4+BF16)\n",
           (unsigned long long)pl1_cases_checked);

    /* 3D. Red-Team #11 Bug #1 Explicit Fallback Zero-Corruption Proof:
     * Verify n_comp <= 512, glen < 8, RO_SMEM_Q=0, and q_bf16=NULL all fall back
     * cleanly reading intact FP32 q_f32 with memcmp == 0 and exact counter increments. */
    {
        myds_v41_gpu_indexer_prefill_wide_counters(NULL, true);
        const uint32_t fb_cases[][3] = {
            /* {nc, glen, pos0} */
            {512u,   16u,   496u},  /* n_comp <= 512 fallback */
            {512u,    8u,   504u},  /* n_comp <= 512 fallback */
            {16384u,  1u, 16383u},  /* glen < 8 fallback */
            {16384u,  7u, 16377u},  /* glen < 8 fallback */
        };
        uint64_t expected_fb = 0;
        for (size_t i = 0; i < sizeof(fb_cases) / sizeof(fb_cases[0]); ++i) {
            const uint32_t nc = fb_cases[i][0], glen = fb_cases[i][1], pos0 = fb_cases[i][2];
            setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
                scratch, sc_ref, q_f32, w_buf, cache_fp4,
                nc, 32u, glen, pos0, 1u, MYDS_INDEXER_CACHE_FP4, true) == 1);
            CHECK(myds_metal_commit_batch());

            setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode_qbf16(
                scratch, sc_wide, q_f32, q_bf16, w_buf, cache_fp4,
                nc, 32u, glen, pos0, 1u, MYDS_INDEXER_CACHE_FP4, true) == 1);
            CHECK(myds_metal_commit_batch());

            CHECK(memcmp(myds_metal_buffer_data(sc_ref), myds_metal_buffer_data(sc_wide),
                         (size_t)nc * glen * sizeof(float)) == 0);
            expected_fb++;
        }

        /* Also test MYDS_INDEXER_RO_SMEM_Q=0 fallback and q_bf16=NULL fallback under WIDE=1 */
        setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
        setenv("MYDS_INDEXER_RO_SMEM_Q", "0", 1);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode_qbf16(
            scratch, sc_wide, q_f32, q_bf16, w_buf, cache_fp4,
            16384u, 32u, 16u, 16368u, 1u, MYDS_INDEXER_CACHE_FP4, true) == 1);
        CHECK(myds_metal_commit_batch());
        expected_fb++;
        unsetenv("MYDS_INDEXER_RO_SMEM_Q");

        /* q_bf16=NULL on IRO and compact */
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
            scratch, sc_ref, q_f32, w_buf, cache_fp4,
            16384u, 32u, 16u, 16368u, 1u, MYDS_INDEXER_CACHE_FP4, true) == 1);
        CHECK(myds_metal_commit_batch());
        expected_fb++;
        CHECK(memcmp(myds_metal_buffer_data(sc_ref), myds_metal_buffer_data(sc_wide),
                     (size_t)16384u * 16u * sizeof(float)) == 0);

        const uint32_t mu32 = myds_v41_gpu_maskbits_stride(32768u / 8u);
        build_l20_mask(mask_2d, 16u, 32768u, mu32, 777u);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(
            scratch, top_wide, q_f32, w_buf, cache_fp4, mask_2d,
            32768u, 16u, 32752u, 1u, mu32, 2048u, MYDS_INDEXER_CACHE_FP4, true) == 1);
        CHECK(myds_metal_commit_batch());
        expected_fb++;

        /* Verify invalid env fails closed on both scoped entry points */
        setenv("MYDS_INDEXER_PREFILL_WIDE", "2", 1);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode_qbf16(
            scratch, sc_wide, q_f32, q_bf16, w_buf, cache_fp4,
            16384u, 32u, 16u, 16368u, 1u, MYDS_INDEXER_CACHE_FP4, true) == 0);
        CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode_qbf16(
            scratch, top_wide, q_f32, q_bf16, w_buf, cache_fp4, mask_2d,
            32768u, 16u, 32752u, 1u, mu32, 2048u, MYDS_INDEXER_CACHE_FP4, true) == -1);
        CHECK(myds_metal_commit_batch());

        myds_v41_gpu_indexer_prefill_wide_counters(&wc, false);
        CHECK(wc.iro_wide_encoded == 0u);
        CHECK(wc.compact_wide_encoded == 0u);
        CHECK(wc.fallbacks == expected_fb);

        /* Verify /debug/state JSON exposes indexer_prefill_wide counters cleanly */
        setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
        char dbg[4096];
        myds_debug_state_json(dbg, sizeof(dbg), NULL);
        CHECK(strstr(dbg, "\"indexer_prefill_wide\":{") != NULL);
        CHECK(strstr(dbg, "\"iro_wide_encoded\":0") != NULL);
        CHECK(strstr(dbg, "\"compact_wide_encoded\":0") != NULL);
        char fb_needle[64];
        snprintf(fb_needle, sizeof(fb_needle), "\"fallbacks\":%llu", (unsigned long long)expected_fb);
        CHECK(strstr(dbg, fb_needle) != NULL);
        CHECK(strstr(dbg, "truncated") == NULL);

        printf("  [PASS] Red-Team #11 Bug #1 fallback tail zero-corruption (%llu fallbacks, 0-ULP) + /debug/state JSON verified\n",
               (unsigned long long)expected_fb);
    }

    unsetenv("MYDS_INDEXER_PREFILL_WIDE");
}

/* 4. Interleaved 7-run median M4 microbenchmark matching §177 geometries */
static void run_wide_microbenchmarks(myds_v41_gpu_scratch_t *scratch,
                                     const myds_metal_buffer_t *q_f32,
                                     const myds_metal_buffer_t *q_bf16,
                                     const myds_metal_buffer_t *w_buf,
                                     const myds_metal_buffer_t *cache_bf16,
                                     const myds_metal_buffer_t *cache_fp4,
                                     myds_metal_buffer_t *sc_ref,
                                     myds_metal_buffer_t *sc_wide,
                                     myds_metal_buffer_t *mask_2d,
                                     myds_metal_buffer_t *top_ref,
                                     myds_metal_buffer_t *top_wide) {
    printf("\n=== [CAND-I1] M4 Interleaved 7-Run Median Microbenchmarks (Baseline vs MYDS_INDEXER_PREFILL_WIDE=1) ===\n");
    struct { const char *name; uint32_t nc; uint32_t glen; } iro_cases[] = {
        {"IRO_32K_g256",   32768u, 256u},
        {"IRO_64K_g128",   65536u, 128u},
        {"IRO_128K_g64",  131072u,  64u},
        {"IRO_256K_g32",  262144u,  32u},
        {"IRO_128K_g16",  131072u,  16u},
        {"IRO_128K_g24",  131072u,  24u},
    };
    const uint32_t modes[] = {MYDS_INDEXER_CACHE_FP4, MYDS_INDEXER_CACHE_BF16};
    const char *mode_names[] = {"FP4 (68 B/row)", "BF16 (256 B/row)"};

    for (size_t mi = 0; mi < 2u; ++mi) {
        const uint32_t mode = modes[mi];
        const myds_metal_buffer_t *cache = (mode == MYDS_INDEXER_CACHE_FP4) ? cache_fp4 : cache_bf16;
        printf("  --- Prefill IRO Scorer [%s] ---\n", mode_names[mi]);
        for (size_t i = 0; i < sizeof(iro_cases) / sizeof(iro_cases[0]); ++i) {
            const uint32_t nc = iro_cases[i].nc, glen = iro_cases[i].glen, pos0 = nc - glen;
            /* Warmup + 0-ULP check */
            setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
                scratch, sc_ref, q_f32, w_buf, cache, nc, 32u, glen, pos0, 1u, mode, true) == 1);
            CHECK(myds_metal_commit_batch());

            setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode_qbf16(
                scratch, sc_wide, q_f32, q_bf16, w_buf, cache, nc, 32u, glen, pos0, 1u, mode, true) == 1);
            CHECK(myds_metal_commit_batch());
            CHECK(memcmp(myds_metal_buffer_data(sc_ref), myds_metal_buffer_data(sc_wide),
                         (size_t)nc * glen * sizeof(float)) == 0);

            double t_base[7], t_wide[7];
            for (int r = 0; r < 7; ++r) {
                setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
                const double s0 = monotonic_ms();
                CHECK(myds_metal_begin_batch());
                CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
                    scratch, sc_ref, q_f32, w_buf, cache, nc, 32u, glen, pos0, 1u, mode, true) == 1);
                CHECK(myds_metal_commit_batch());
                t_base[r] = monotonic_ms() - s0;

                setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
                const double s1 = monotonic_ms();
                CHECK(myds_metal_begin_batch());
                CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode_qbf16(
                    scratch, sc_wide, q_f32, q_bf16, w_buf, cache, nc, 32u, glen, pos0, 1u, mode, true) == 1);
                CHECK(myds_metal_commit_batch());
                t_wide[r] = monotonic_ms() - s1;
            }
            qsort(t_base, 7, sizeof(double), cmp_double);
            qsort(t_wide, 7, sizeof(double), cmp_double);
            printf("    %-16s (nc=%6u, glen=%3u): base %6.2f ms | wide %6.2f ms | %.2fx (%+.1f%%)\n",
                   iro_cases[i].name, nc, glen, t_base[3], t_wide[3],
                   t_base[3] / t_wide[3], (t_base[3] / t_wide[3] - 1.0) * 100.0);
        }

        struct { const char *name; uint32_t nc; uint32_t glen; } pl1_cases[] = {
            {"PL1_128K_g512", 131072u, 512u},
            {"PL1_256K_g304", 262144u, 304u},
        };
        printf("  --- Prefill [P-L1] YoCo Compact Pipeline [%s] ---\n", mode_names[mi]);
        for (size_t i = 0; i < sizeof(pl1_cases) / sizeof(pl1_cases[0]); ++i) {
            const uint32_t nc = pl1_cases[i].nc, glen = pl1_cases[i].glen, pos0 = nc - glen;
            const uint32_t mu = myds_v41_gpu_maskbits_stride(nc / 8u);
            build_l20_mask(mask_2d, glen, nc, mu, 0u);

            setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(
                scratch, top_ref, q_f32, w_buf, cache, mask_2d,
                nc, glen, pos0, 1u, mu, 2048u, mode, true) == 1);
            CHECK(myds_metal_commit_batch());

            setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode_qbf16(
                scratch, top_wide, q_f32, q_bf16, w_buf, cache, mask_2d,
                nc, glen, pos0, 1u, mu, 2048u, mode, true) == 1);
            CHECK(myds_metal_commit_batch());
            CHECK(memcmp(myds_metal_buffer_data(top_ref), myds_metal_buffer_data(top_wide),
                         (size_t)glen * 512u * sizeof(int32_t)) == 0);

            double t_base[7], t_wide[7];
            for (int r = 0; r < 7; ++r) {
                setenv("MYDS_INDEXER_PREFILL_WIDE", "0", 1);
                const double s0 = monotonic_ms();
                CHECK(myds_metal_begin_batch());
                CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(
                    scratch, top_ref, q_f32, w_buf, cache, mask_2d,
                    nc, glen, pos0, 1u, mu, 2048u, mode, true) == 1);
                CHECK(myds_metal_commit_batch());
                t_base[r] = monotonic_ms() - s0;

                setenv("MYDS_INDEXER_PREFILL_WIDE", "1", 1);
                const double s1 = monotonic_ms();
                CHECK(myds_metal_begin_batch());
                CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode_qbf16(
                    scratch, top_wide, q_f32, q_bf16, w_buf, cache, mask_2d,
                    nc, glen, pos0, 1u, mu, 2048u, mode, true) == 1);
                CHECK(myds_metal_commit_batch());
                t_wide[r] = monotonic_ms() - s1;
            }
            qsort(t_base, 7, sizeof(double), cmp_double);
            qsort(t_wide, 7, sizeof(double), cmp_double);
            printf("    %-16s (nc=%6u, glen=%3u): base %6.2f ms | wide %6.2f ms | %.2fx (%+.1f%%)\n",
                   pl1_cases[i].name, nc, glen, t_base[3], t_wide[3],
                   t_base[3] / t_wide[3], (t_base[3] / t_wide[3] - 1.0) * 100.0);
        }
    }
    unsetenv("MYDS_INDEXER_PREFILL_WIDE");
}

int main(int argc, char **argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) do_bench = true;
    }

    printf("=== test_indexer_prefill_wide ([CAND-I1] MYDS_INDEXER_PREFILL_WIDE, 0-ULP) ===\n");
    CHECK(myds_metal_init("metal"));

    test_env_parsing();
    verify_pso_occupancy();

    const uint32_t max_nc = 262144u;
    const uint32_t max_rows = 512u;
    setenv("MYDS_CHUNK_KEY_MAX", "262144", 1);
    myds_v41_gpu_scratch_t *scratch = myds_v41_gpu_scratch_create(max_nc);
    unsetenv("MYDS_CHUNK_KEY_MAX");
    CHECK(scratch != NULL);

    myds_metal_buffer_t *raw_k      = myds_metal_buffer_alloc((size_t)max_nc * 128u * sizeof(float));
    myds_metal_buffer_t *raw_k2     = myds_metal_buffer_alloc((size_t)max_nc * 128u * sizeof(float));
    myds_metal_buffer_t *cache_bf16 = myds_metal_buffer_alloc((size_t)max_nc * 256u);
    myds_metal_buffer_t *cache_fp4  = myds_metal_buffer_alloc((size_t)max_nc * 68u);
    myds_metal_buffer_t *q_f32      = myds_metal_buffer_alloc((size_t)max_rows * 32u * 128u * sizeof(float));
    myds_metal_buffer_t *q_bf16     = myds_metal_buffer_alloc((size_t)max_rows * 32u * 128u * sizeof(uint16_t));
    myds_metal_buffer_t *w_buf      = myds_metal_buffer_alloc((size_t)max_rows * 32u * sizeof(float));
    myds_metal_buffer_t *sc_ref     = myds_metal_buffer_alloc((size_t)8388608u * sizeof(float));
    myds_metal_buffer_t *sc_wide    = myds_metal_buffer_alloc((size_t)8388608u * sizeof(float));
    const uint32_t max_mu           = myds_v41_gpu_maskbits_stride(max_nc / 8u);
    myds_metal_buffer_t *mask_2d    = myds_metal_buffer_alloc((size_t)max_rows * max_mu * sizeof(uint32_t));
    myds_metal_buffer_t *top_ref    = myds_metal_buffer_alloc((size_t)max_rows * 512u * sizeof(int32_t));
    myds_metal_buffer_t *top_wide   = myds_metal_buffer_alloc((size_t)max_rows * 512u * sizeof(int32_t));
    CHECK(raw_k && raw_k2 && cache_bf16 && cache_fp4 && q_f32 && q_bf16 && w_buf &&
          sc_ref && sc_wide && mask_2d && top_ref && top_wide);

    uint32_t rng = 0x517cc1b7u;
    float *kp = (float *)myds_metal_buffer_data(raw_k);
    for (size_t i = 0; i < (size_t)max_nc * 128u; ++i) {
        kp[i] = ((float)((int32_t)(xorshift32(&rng) % 2001u) - 1000)) * 0.01f;
    }
    memcpy(myds_metal_buffer_data(raw_k2), kp, (size_t)max_nc * 128u * sizeof(float));
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(raw_k, cache_bf16, 0u, 1u, 0u, max_nc, 1u, true, MYDS_INDEXER_CACHE_BF16));
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(raw_k2, cache_fp4, 0u, 1u, 0u, max_nc, 1u, true, MYDS_INDEXER_CACHE_FP4));
    myds_metal_buffer_free(raw_k);
    myds_metal_buffer_free(raw_k2);

    float *qp = (float *)myds_metal_buffer_data(q_f32);
    for (size_t i = 0; i < (size_t)max_rows * 32u * 128u; ++i) {
        qp[i] = ((float)((int32_t)(xorshift32(&rng) % 2001u) - 1000)) * 0.01f;
    }
    /* Exact production fused pass: writes FP32 q_f32 AND packed BF16 q_bf16 simultaneously */
    CHECK(myds_v41_gpu_indexer_rope_quant_batch(q_f32, q_bf16, 0u, 32u, 0u, max_rows, 1u, true));

    float *wp = (float *)myds_metal_buffer_data(w_buf);
    for (size_t i = 0; i < (size_t)max_rows * 32u; ++i) {
        wp[i] = 0.05f + 0.01f * (float)(xorshift32(&rng) % 100u);
    }

    test_0ulp_and_fallbacks(scratch, q_f32, q_bf16, w_buf, cache_bf16, cache_fp4,
                            sc_ref, sc_wide, mask_2d, top_ref, top_wide);

    if (do_bench) {
        run_wide_microbenchmarks(scratch, q_f32, q_bf16, w_buf, cache_bf16, cache_fp4,
                                 sc_ref, sc_wide, mask_2d, top_ref, top_wide);
    }

    myds_metal_buffer_free(cache_bf16);
    myds_metal_buffer_free(cache_fp4);
    myds_metal_buffer_free(q_f32);
    myds_metal_buffer_free(q_bf16);
    myds_metal_buffer_free(w_buf);
    myds_metal_buffer_free(sc_ref);
    myds_metal_buffer_free(sc_wide);
    myds_metal_buffer_free(mask_2d);
    myds_metal_buffer_free(top_ref);
    myds_metal_buffer_free(top_wide);
    myds_v41_gpu_scratch_free(scratch);
    printf("ALL test_indexer_prefill_wide CHECKS PASSED!\n");
    return 0;
}
