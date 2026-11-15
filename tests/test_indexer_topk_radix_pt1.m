/* tests/test_indexer_topk_radix_pt1.m
 *
 * [P-T1] Apple Silicon GPU 0-ULP parity, PSO occupancy, fail-closed/fallback
 * contract, and 7-run median microbenchmark suite for:
 *   Part 1: Batched 0-ULP 2-Pass Radix/Histogram Causal Top-K + 1024-Partition
 *           Bitonic Tie-Replay (myds_v41_gpu_topk_radix_causal_scoped)
 *   Part 2: L20 Candidate + Top-2048 + Block-Mask + 16,384-Slot Self-Compact
 *           Score Gather & Compact Top-512 (myds_v41_gpu_indexer_l20_compact_topk_scoped)
 *
 * All parity checks assert memcmp == 0 on output int32 indices (and L20 masks)
 * against the production baseline (MYDS_INDEXER_TOPK_PREFIX_MERGE=1 +
 * MYDS_INDEXER_TOPK_NAN_FUSE=1 + MYDS_INDEXER_TOPK_RADIX_CAUSAL=0).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_signed(uint32_t *state, float amp) {
    const float u = (float)(xorshift32(state) & 0x00ffffffu) / (float)0x00800000u - 1.0f;
    return u * amp;
}

static float bits_to_f32(uint32_t u) {
    float f = 0.0f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void set_prod_baseline_knobs(void) {
    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);
    unsetenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE");
    unsetenv("MYDS_TOPK_THRESHOLD");
    unsetenv("MYDS_BLOCK_BUDGET");
}

static void fill_realistic_relu_scores(myds_metal_buffer_t *scores, uint32_t rows,
                                       uint32_t cols, uint32_t *rng) {
    float *dst = (float *)myds_metal_buffer_data(scores);
    CHECK(dst != NULL);
    const size_t total = (size_t)rows * cols;
    for (size_t i = 0; i < total; ++i) {
        /* ~35% strictly positive ReLU tail, ~65% exact 0.0f (realistic Indexer score profile) */
        const float v = rand_signed(rng, 4.0f) - 1.2f;
        dst[i] = (v > 0.0f) ? v : 0.0f;
    }
}

static void assert_ids_0ulp(const char *label, const myds_metal_buffer_t *ref_buf,
                            const myds_metal_buffer_t *got_buf, uint32_t rows, uint32_t top) {
    const int32_t *ref = (const int32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)ref_buf);
    const int32_t *got = (const int32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)got_buf);
    const size_t bytes = (size_t)rows * top * sizeof(int32_t);
    if (memcmp(ref, got, bytes) != 0) {
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t k = 0; k < top; ++k) {
                const int32_t a = ref[(size_t)r * top + k];
                const int32_t b = got[(size_t)r * top + k];
                if (a != b) {
                    fprintf(stderr, "0-ULP ID MISMATCH [%s] at row=%u/%u rank=%u/%u: ref=%d (0x%08x) got=%d (0x%08x)\n",
                            label, r, rows, k, top, a, (uint32_t)a, b, (uint32_t)b);
                    exit(1);
                }
            }
        }
    }
}

static void assert_bytes_0ulp(const char *label, const myds_metal_buffer_t *ref_buf,
                              const myds_metal_buffer_t *got_buf, size_t bytes) {
    const uint32_t *ref = (const uint32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)ref_buf);
    const uint32_t *got = (const uint32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)got_buf);
    if (memcmp(ref, got, bytes) != 0) {
        const size_t words = bytes / sizeof(uint32_t);
        for (size_t i = 0; i < words; ++i) {
            if (ref[i] != got[i]) {
                fprintf(stderr, "0-ULP MASK MISMATCH [%s] at word=%zu/%zu: ref=0x%08x got=0x%08x\n",
                        label, i, words, ref[i], got[i]);
                exit(1);
            }
        }
    }
}

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
            NSUInteger max_static_smem;
        } specs[] = {
            {"kernel_myds_topk_radix_clear",          256u,     0u},
            {"kernel_myds_topk_radix_pass1",          256u,  4096u},
            {"kernel_myds_topk_radix_pivot",          256u,  4224u},
            {"kernel_myds_topk_radix_gather",         256u,   256u},
            {"kernel_myds_topk_radix_final",         1024u, 26624u},
            {"kernel_myds_topk_radix_nan_fallback",  1024u,  8320u},
            {"kernel_myds_l20_candidate_blocks_nan",  256u,     0u},
            {"kernel_myds_l20_sort_blocks_2048",     1024u,  8192u},
            {"kernel_myds_l20_gather_compact_scores", 256u,     0u},
            {"kernel_myds_l20_compact_topk_final",   1024u, 22528u},
        };

        printf("[1/6] Verifying [P-T1] Metal PSO occupancy & static threadgroup memory...\n");
        for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); ++i) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:specs[i].name]];
            CHECK(fn != nil);
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            CHECK(pso != nil && err == nil);
            CHECK(pso.threadExecutionWidth == 32u);
            CHECK(pso.maxTotalThreadsPerThreadgroup >= specs[i].min_max_threads);
            CHECK(pso.staticThreadgroupMemoryLength <= specs[i].max_static_smem);
            printf("  %-44s width=%lu max_threads=%4lu static_smem=%5lu B\n",
                   specs[i].name,
                   (unsigned long)pso.threadExecutionWidth,
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)pso.staticThreadgroupMemoryLength);
        }
    }
}

static void run_part1_pair(const char *label, myds_v41_gpu_scratch_t *s,
                           const myds_metal_buffer_t *scores,
                           myds_metal_buffer_t *sel_ref,
                           myds_metal_buffer_t *sel_got,
                           uint32_t n_cols, uint32_t top, uint32_t n_rows,
                           uint32_t causal_start, uint32_t causal_ratio,
                           bool expect_radix_encoded) {
    const size_t out_bytes = (size_t)n_rows * top * sizeof(int32_t);
    const uint32_t min_vis = (causal_start + 1u) / causal_ratio;
    const uint32_t got_fill = (min_vis >= top) ? 0x5a : 0xa5;

    memset(myds_metal_buffer_data(sel_ref), 0xa5, out_bytes);
    memset(myds_metal_buffer_data(sel_got), (int)got_fill, out_bytes);

    setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "0", 1);
    myds_v41_gpu_topk_radix_causal_counters(NULL, true);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_topk_radix_causal_scoped(
        s, sel_ref, scores, n_cols, top, n_rows, causal_start, causal_ratio, true) == 1);
    CHECK(myds_metal_commit_batch());
    myds_v41_gpu_topk_radix_causal_counters_t c_off;
    myds_v41_gpu_topk_radix_causal_counters(&c_off, true);
    CHECK(c_off.requests == 1u && c_off.encoded == 0u && c_off.fallbacks == 0u && c_off.failures == 0u);

    setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "1", 1);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_topk_radix_causal_scoped(
        s, sel_got, scores, n_cols, top, n_rows, causal_start, causal_ratio, true) == 1);
    CHECK(myds_metal_commit_batch());
    myds_v41_gpu_topk_radix_causal_counters_t c_on;
    myds_v41_gpu_topk_radix_causal_counters(&c_on, true);
    CHECK(c_on.requests == 1u && c_on.fallbacks == 0u && c_on.failures == 0u);
    if (expect_radix_encoded) {
        CHECK(c_on.encoded == 1u && c_on.rows == n_rows);
    } else {
        CHECK(c_on.encoded == 0u);
    }

    assert_ids_0ulp(label, sel_ref, sel_got, n_rows, top);
}

static void run_l20_reference(myds_v41_gpu_scratch_t *s,
                              const myds_metal_buffer_t *scores_buf,
                              myds_metal_buffer_t *mask_out,
                              myds_metal_buffer_t *sel_out,
                              uint32_t nc, uint32_t glen,
                              uint32_t g_start, uint32_t ratio,
                              uint32_t mask_units, uint32_t mask_stride,
                              bool mask_bits) {
    const uint32_t blocks = (nc + 7u) / 8u;
    const uint32_t top_blocks = blocks < 2048u ? blocks : 2048u;
    myds_metal_buffer_t *blocks_buf = myds_v41_gpu_indexer_blocks(s);
    myds_metal_buffer_t *block_sel_buf = myds_v41_gpu_indexer_block_selected(s);

    CHECK(myds_metal_begin_batch());
    if (!myds_metal_get_auto_barrier()) {
        const myds_metal_buffer_t *deps[] = {scores_buf, blocks_buf, block_sel_buf, mask_out, sel_out};
        myds_metal_barrier_buffers(deps, 5u);
    }
    CHECK(myds_v41_gpu_candidates_rows(blocks_buf, scores_buf, nc, glen, g_start, ratio, mask_stride));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(blocks_buf);
    CHECK(myds_v41_gpu_topk_causal(block_sel_buf, blocks_buf, blocks, top_blocks, glen, g_start, ratio, s));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(block_sel_buf);
    if (mask_bits) {
        CHECK(myds_v41_gpu_maskbits_rows(mask_out, block_sel_buf, blocks, top_blocks, glen, mask_units));
    } else {
        CHECK(myds_v41_gpu_topkmask_rows(mask_out, block_sel_buf, blocks, top_blocks, glen, mask_stride));
    }
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(mask_out);
    CHECK(myds_v41_gpu_topk_causal(sel_out, scores_buf, nc, nc < 512u ? nc : 512u, glen, g_start, ratio, s));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(sel_out);
    CHECK(myds_metal_commit_batch());
}

static void run_l20_candidate(myds_v41_gpu_scratch_t *s,
                              const myds_metal_buffer_t *scores_buf,
                              myds_metal_buffer_t *mask_out,
                              myds_metal_buffer_t *sel_out,
                              uint32_t nc, uint32_t glen,
                              uint32_t g_start, uint32_t ratio,
                              uint32_t mask_units, uint32_t mask_stride,
                              bool mask_bits, bool pt1_enabled) {
    const uint32_t blocks = (nc + 7u) / 8u;
    const uint32_t top_blocks = blocks < 2048u ? blocks : 2048u;
    myds_metal_buffer_t *blocks_buf = myds_v41_gpu_indexer_blocks(s);
    myds_metal_buffer_t *block_sel_buf = myds_v41_gpu_indexer_block_selected(s);

    setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", pt1_enabled ? "1" : "0", 1);
    CHECK(myds_metal_begin_batch());
    if (nc > 16384u && top_blocks == 2048u) {
        const int rc = myds_v41_gpu_indexer_l20_compact_topk_scoped(
            s, sel_out, mask_out, scores_buf, nc, glen, g_start, ratio,
            mask_bits ? mask_units : mask_stride, mask_bits, top_blocks, true);
        CHECK(rc >= 0);
        if (rc > 0) {
            CHECK(myds_metal_commit_batch());
            return;
        }
    }
    if (!myds_metal_get_auto_barrier()) {
        const myds_metal_buffer_t *deps[] = {scores_buf, blocks_buf, block_sel_buf, mask_out, sel_out};
        myds_metal_barrier_buffers(deps, 5u);
    }
    CHECK(myds_v41_gpu_candidates_rows(blocks_buf, scores_buf, nc, glen, g_start, ratio, mask_stride));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(blocks_buf);
    CHECK(myds_v41_gpu_topk_radix_causal_scoped(
        s, block_sel_buf, blocks_buf, blocks, top_blocks, glen, g_start, ratio, true) == 1);
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(block_sel_buf);
    if (mask_bits) {
        CHECK(myds_v41_gpu_maskbits_rows(mask_out, block_sel_buf, blocks, top_blocks, glen, mask_units));
    } else {
        CHECK(myds_v41_gpu_topkmask_rows(mask_out, block_sel_buf, blocks, top_blocks, glen, mask_stride));
    }
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(mask_out);
    CHECK(myds_v41_gpu_topk_radix_causal_scoped(
        s, sel_out, scores_buf, nc, nc < 512u ? nc : 512u, glen, g_start, ratio, true) == 1);
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(sel_out);
    CHECK(myds_metal_commit_batch());
}

static void verify_knob_and_contracts(myds_v41_gpu_scratch_t *s,
                                      myds_metal_buffer_t *scores,
                                      myds_metal_buffer_t *sel_ref,
                                      myds_metal_buffer_t *sel_got,
                                      myds_metal_buffer_t *mask_ref,
                                      myds_metal_buffer_t *mask_got,
                                      uint32_t mask_units, uint32_t mask_stride) {
    printf("[2/6] Verifying MYDS_INDEXER_TOPK_RADIX_CAUSAL knob parsing, fail-closed & fallback...\n");
    unsetenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL");
    CHECK(myds_v41_gpu_topk_radix_causal_enabled() == 0);
    setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "0", 1);
    CHECK(myds_v41_gpu_topk_radix_causal_enabled() == 0);
    setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "1", 1);
    CHECK(myds_v41_gpu_topk_radix_causal_enabled() == 1);

    const char *bad_vals[] = {"", "2", "true", "false", "01", " 1", "1 "};
    for (size_t i = 0; i < sizeof(bad_vals) / sizeof(bad_vals[0]); ++i) {
        setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", bad_vals[i], 1);
        CHECK(myds_v41_gpu_topk_radix_causal_enabled() == -1);
        CHECK(myds_v41_gpu_scratch_create(65536u) == NULL);
    }

    /* Fail-closed in both scoped selectors on invalid env */
    myds_v41_gpu_topk_radix_causal_counters(NULL, true);
    setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "bad", 1);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_topk_radix_causal_scoped(
        s, sel_got, scores, 32768u, 512u, 8u, 32760u, 1u, true) == 0);
    CHECK(myds_v41_gpu_indexer_l20_compact_topk_scoped(
        s, sel_got, mask_got, scores, 32768u, 8u, 32760u, 1u, mask_units, true, 2048u, true) == -1);
    CHECK(myds_metal_commit_batch());
    myds_v41_gpu_topk_radix_causal_counters_t ctr;
    myds_v41_gpu_topk_radix_causal_counters(&ctr, true);
    CHECK(ctr.requests == 1u && ctr.l20_compact_requests == 1u && ctr.encoded == 0u && ctr.failures == 2u);

    /* Diagnostic unavailable injection -> graceful fallback + 0-ULP parity */
    setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "1", 1);
    run_l20_reference(s, scores, mask_ref, sel_ref, 32768u, 8u, 32760u, 1u, mask_units, mask_stride, true);
    myds_v41_gpu_topk_radix_causal_counters(NULL, true);
    myds_v41_gpu_inject_topk_radix_causal_unavailable(
        MYDS_TOPK_RADIX_CAUSAL_UNAVAILABLE_RADIX | MYDS_TOPK_RADIX_CAUSAL_UNAVAILABLE_L20);
    run_l20_candidate(s, scores, mask_got, sel_got, 32768u, 8u, 32760u, 1u, mask_units, mask_stride, true, true);
    myds_v41_gpu_inject_topk_radix_causal_unavailable(0u);
    myds_v41_gpu_topk_radix_causal_counters(&ctr, true);
    CHECK(ctr.l20_compact_requests == 1u && ctr.l20_compact_encoded == 0u && ctr.l20_compact_fallbacks == 1u);
    CHECK(ctr.requests == 2u && ctr.encoded == 0u && ctr.fallbacks == 2u && ctr.failures == 0u);
    assert_bytes_0ulp("fallback_l20_mask", mask_ref, mask_got, 8u * mask_units * sizeof(uint32_t));
    assert_ids_0ulp("fallback_l20_ids", sel_ref, sel_got, 8u, 512u);

    /* Out-of-scope shapes cleanly bypass without incrementing fallbacks/failures */
    run_part1_pair("bypass_nc8192_top512", s, scores, sel_ref, sel_got, 8192u, 512u, 16u, 8176u, 1u, false);
    run_part1_pair("bypass_nc2048_top2048", s, scores, sel_ref, sel_got, 2048u, 2048u, 8u, 2040u, 1u, false);
    unsetenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL");
}

static void verify_part1_matrix_and_torture(myds_v41_gpu_scratch_t *s,
                                            myds_metal_buffer_t *scores,
                                            myds_metal_buffer_t *sel_ref,
                                            myds_metal_buffer_t *sel_got,
                                            uint32_t max_rows,
                                            uint32_t *rng) {
    printf("[3/6] Running Part 1 Radix Top-K 0-ULP matrix (top={512,2048}, ratio={1,2}, nc={4K..256K})...\n");
    const uint32_t batch_floats = myds_v41_gpu_indexer_batch_floats(s);

    /* A. Top-512 across nc={16384..262144}, glen={1,7,8,9,64,128,392,512}, ratio={1,2} */
    const uint32_t ncs_512[] = {16384u, 16385u, 32768u, 65536u, 131072u, 262144u};
    const uint32_t glens_512[] = {1u, 7u, 8u, 9u, 64u, 128u, 392u, 512u};
    for (size_t ni = 0; ni < sizeof(ncs_512) / sizeof(ncs_512[0]); ++ni) {
        const uint32_t nc = ncs_512[ni];
        for (size_t gi = 0; gi < sizeof(glens_512) / sizeof(glens_512[0]); ++gi) {
            const uint32_t glen = glens_512[gi];
            if (glen > max_rows || (uint64_t)glen * nc > (uint64_t)batch_floats) continue;
            fill_realistic_relu_scores(scores, glen, nc, rng);
            for (uint32_t ratio = 1u; ratio <= 2u; ++ratio) {
                const uint32_t pos0 = (nc * ratio > glen) ? (nc * ratio - glen) : 0u;
                char lbl[128];
                snprintf(lbl, sizeof(lbl), "top512 nc=%u glen=%u r=%u", nc, glen, ratio);
                run_part1_pair(lbl, s, scores, sel_ref, sel_got, nc, 512u, glen, pos0, ratio, true);
            }
        }
        printf("  top=512  nc=%6u: all in-budget glens x ratio={1,2} -> PASS (0-ULP)\n", nc);
    }
    if (max_rows >= 2048u && (uint64_t)2048u * 16384u <= (uint64_t)batch_floats) {
        fill_realistic_relu_scores(scores, 2048u, 16384u, rng);
        run_part1_pair("top512_nc16384_glen2048", s, scores, sel_ref, sel_got,
                       16384u, 512u, 2048u, 16384u - 2048u, 1u, true);
    }

    /* B. Top-2048 across blocks={4096..65536}, glen={1,8,9,64,128,512}, ratio={1,2} */
    const uint32_t ncs_2048[] = {4096u, 8192u, 16384u, 16385u, 32768u, 65536u};
    const uint32_t glens_2048[] = {1u, 8u, 9u, 64u, 128u, 512u};
    for (size_t ni = 0; ni < sizeof(ncs_2048) / sizeof(ncs_2048[0]); ++ni) {
        const uint32_t nc = ncs_2048[ni];
        for (size_t gi = 0; gi < sizeof(glens_2048) / sizeof(glens_2048[0]); ++gi) {
            const uint32_t glen = glens_2048[gi];
            if (glen > max_rows || (uint64_t)glen * nc > (uint64_t)batch_floats) continue;
            fill_realistic_relu_scores(scores, glen, nc, rng);
            for (uint32_t ratio = 1u; ratio <= 2u; ++ratio) {
                const uint32_t pos0 = (nc * ratio > glen) ? (nc * ratio - glen) : 0u;
                char lbl[128];
                snprintf(lbl, sizeof(lbl), "top2048 nc=%u glen=%u r=%u", nc, glen, ratio);
                run_part1_pair(lbl, s, scores, sel_ref, sel_got, nc, 2048u, glen, pos0, ratio, true);
            }
        }
        printf("  top=2048 nc=%6u: all in-budget glens x ratio={1,2} -> PASS (0-ULP)\n", nc);
    }

    printf("[4/6] Running IEEE-754 & Tie Torture Suite (-0/+0, subnormals, 256K 0.0f ties, zoom, NaN)...\n");
    float *sp = (float *)myds_metal_buffer_data(scores);

    /* C. Causal width edge cases: width=0 (pos0=0,r=2), width<512, width=1024/1025, width=2047/2048 */
    fill_realistic_relu_scores(scores, 16u, 32768u, rng);
    const uint32_t edge_starts[] = {0u, 508u, 1020u, 2044u, 16380u};
    for (size_t ei = 0; ei < sizeof(edge_starts) / sizeof(edge_starts[0]); ++ei) {
        const uint32_t cs = edge_starts[ei];
        run_part1_pair("causal_edge_t512_r1", s, scores, sel_ref, sel_got, 32768u, 512u, 16u, cs, 1u, true);
        run_part1_pair("causal_edge_t512_r2", s, scores, sel_ref, sel_got, 32768u, 512u, 16u, cs, 2u, true);
        run_part1_pair("causal_edge_t2048_r1", s, scores, sel_ref, sel_got, 32768u, 2048u, 16u, cs, 1u, true);
        run_part1_pair("causal_edge_t2048_r2", s, scores, sel_ref, sel_got, 32768u, 2048u, 16u, cs, 2u, true);
    }

    /* D. All 0.0f across 262,144 cols (MODE_HEAVY_TIE with c_gt == 0) */
    memset(sp, 0, 8u * 262144u * sizeof(float));
    run_part1_pair("all_zero_256k_t512", s, scores, sel_ref, sel_got, 262144u, 512u, 8u, 262136u, 1u, true);
    run_part1_pair("all_zero_64k_t2048", s, scores, sel_ref, sel_got, 65536u, 2048u, 8u, 65528u, 1u, true);

    /* E. Sparse <512 strictly positive scores + ~262,000 0.0f ReLU ties (MODE_HEAVY_TIE with 0 < c_gt < top) */
    memset(sp, 0, 8u * 262144u * sizeof(float));
    for (uint32_t r = 0; r < 8u; ++r) {
        const uint32_t n_pos = (r == 0u) ? 137u : ((r == 1u) ? 511u : (r * 73u + 19u));
        for (uint32_t k = 0; k < n_pos; ++k) {
            const uint32_t col = (k * 1999u + r * 313u) % 262100u;
            sp[(size_t)r * 262144u + col] = 0.25f + 0.01f * (float)(k % 17u);
        }
    }
    run_part1_pair("sparse_lt512_zero_ties_256k", s, scores, sel_ref, sel_got, 262144u, 512u, 8u, 262136u, 1u, true);

    /* F. Mixed -0.0f / +0.0f + positive/negative subnormals + small normals across 131,072 cols */
    for (uint32_t r = 0; r < 16u; ++r) {
        float *row = sp + (size_t)r * 131072u;
        for (uint32_t c = 0; c < 131072u; ++c) {
            const uint32_t pick = xorshift32(rng) & 7u;
            switch (pick) {
                case 0: row[c] = bits_to_f32(0x00000000u); break; /* +0.0f */
                case 1: row[c] = bits_to_f32(0x80000000u); break; /* -0.0f */
                case 2: row[c] = bits_to_f32(0x00000001u); break; /* +min subnormal */
                case 3: row[c] = bits_to_f32(0x80000001u); break; /* -min subnormal */
                case 4: row[c] = bits_to_f32(0x007fffffu); break; /* +max subnormal */
                case 5: row[c] = bits_to_f32(0x807fffffu); break; /* -max subnormal */
                case 6: row[c] = -1.0e-30f; break;
                default: row[c] = (c % 257u == 0u) ? (1.0e-20f * (float)((c % 13u) + 1u)) : bits_to_f32(0x80000000u); break;
            }
        }
    }
    run_part1_pair("signed_zero_and_subnormals_128k_t512", s, scores, sel_ref, sel_got, 131072u, 512u, 16u, 131056u, 1u, true);
    run_part1_pair("signed_zero_and_subnormals_64k_t2048", s, scores, sel_ref, sel_got, 65536u, 2048u, 16u, 65520u, 1u, true);

    /* G. 16-value quantized heavy duplicates across & within 1024-partitions + boundary ties */
    for (uint32_t r = 0; r < 16u; ++r) {
        float *row = sp + (size_t)r * 131072u;
        for (uint32_t c = 0; c < 131072u; ++c) {
            row[c] = -2.0f + 0.25f * (float)(xorshift32(rng) & 15u);
        }
    }
    run_part1_pair("quantized_16val_128k_t512", s, scores, sel_ref, sel_got, 131072u, 512u, 16u, 131056u, 1u, true);
    run_part1_pair("quantized_16val_64k_t2048", s, scores, sel_ref, sel_got, 65536u, 2048u, 16u, 65520u, 1u, true);

    /* H. Single coarse-bin cluster (>8192 keys within 1 ULP..8000 ULP of 1.0f -> exercises multi-pass
     *    10-bit zoom into BOTH MODE_FAST_GATHER (2048 < surv_cnt <= 4096 2-run merge!) and MODE_HEAVY_TIE!) */
    for (uint32_t r = 0; r < 8u; ++r) {
        float *row = sp + (size_t)r * 65536u;
        for (uint32_t c = 0; c < 65536u; ++c) {
            if (r < 4u) {
                /* 16,384 keys packed in [1.0f, 1.001f) -> same coarse 2^18 bin, ~3000 survivors after zoom */
                const uint32_t ulp = (c < 16384u) ? (xorshift32(rng) % 5u + (c / 3000u) * 8u) : 0u;
                row[c] = (c < 16384u) ? bits_to_f32(0x3f800000u + ulp) : 0.0f;
            } else {
                /* 300 keys slightly > 1.0f + 10,000 exact 1.0f ties in same coarse bin -> zoom to shift==0 HEAVY_TIE */
                if (c % 211u == 0u) row[c] = bits_to_f32(0x3f800010u + (c & 7u));
                else if (c < 12000u) row[c] = 1.0f;
                else row[c] = 0.0f;
            }
        }
    }
    run_part1_pair("single_coarse_bin_zoom_t512", s, scores, sel_ref, sel_got, 65536u, 512u, 8u, 65528u, 1u, true);
    run_part1_pair("single_coarse_bin_zoom_2run_merge_t2048", s, scores, sel_ref, sel_got, 65536u, 2048u, 8u, 65528u, 1u, true);

    /* I. -INFINITY (<512 finite) & +INFINITY multi-partition ties + exact rank-511/512 non-adjacent ties */
    for (uint32_t r = 0; r < 8u; ++r) {
        float *row = sp + (size_t)r * 65536u;
        for (uint32_t c = 0; c < 65536u; ++c) row[c] = -INFINITY;
        if (r < 4u) {
            /* Only 93 finite scores -> <512 finite + -INF partition-0 bitonic tail */
            for (uint32_t k = 0; k < 93u; ++k) row[(k * 677u + r * 41u) % 65500u] = -5.0f + 0.1f * (float)(k % 11u);
        } else {
            /* +INFINITY ties + non-adjacent same-partition ties right around rank 511/512 */
            for (uint32_t k = 0; k < 40u; ++k) row[(k * 1031u + r * 7u) % 65500u] = INFINITY;
            for (uint32_t k = 0; k < 470u; ++k) row[2048u + k * 2u] = 100.0f - 0.05f * (float)k;
            /* 12 exact boundary ties inside partition p=5 separated by smaller values */
            for (uint32_t k = 0; k < 12u; ++k) row[5u * 1024u + k * 79u] = 3.1415927f;
        }
    }
    run_part1_pair("inf_ties_and_rank512_boundary", s, scores, sel_ref, sel_got, 65536u, 512u, 8u, 65528u, 1u, true);

    /* J. Visible IEEE NaN injection (quiet/signaling-pattern NaNs in single & multiple partitions,
     *    width <= 1024 and width = 65536, mixed with healthy non-NaN rows in the same batch!) */
    fill_realistic_relu_scores(scores, 16u, 65536u, rng);
    sp[0u * 65536u + 17u] = bits_to_f32(0x7fc00001u);
    sp[2u * 65536u + 0u] = bits_to_f32(0xffc00013u);
    sp[2u * 65536u + 5u * 1024u + 333u] = bits_to_f32(0x7f800001u);
    sp[2u * 65536u + 63u * 1024u + 999u] = bits_to_f32(0x7fc12345u);
    sp[7u * 65536u + 37u * 1024u + 512u] = NAN;
    run_part1_pair("nan_mixed_batch_64k_t512", s, scores, sel_ref, sel_got, 65536u, 512u, 16u, 65520u, 1u, true);
    run_part1_pair("nan_mixed_batch_64k_t2048", s, scores, sel_ref, sel_got, 65536u, 2048u, 16u, 65520u, 1u, true);
    run_part1_pair("nan_single_part_width777", s, scores, sel_ref, sel_got, 65536u, 512u, 8u, 770u, 1u, true);

    printf("  width=0..2049, 256K all-0.0f, -0/+0/subnormals, 16-val ties, coarse-bin zoom, +-INF, NaN -> PASS\n");
}

static void verify_part2_l20_matrix_and_torture(myds_v41_gpu_scratch_t *s,
                                                myds_metal_buffer_t *scores,
                                                myds_metal_buffer_t *sel_ref,
                                                myds_metal_buffer_t *sel_got,
                                                myds_metal_buffer_t *mask_ref,
                                                myds_metal_buffer_t *mask_got,
                                                myds_metal_buffer_t *mask_f_ref,
                                                myds_metal_buffer_t *mask_f_got,
                                                uint32_t max_rows,
                                                uint32_t mask_units,
                                                uint32_t mask_stride,
                                                uint32_t *rng) {
    printf("[5/6] Running Part 2 L20 Self-Compact Score Gather + Top-512 0-ULP matrix...\n");
    const uint32_t batch_floats = myds_v41_gpu_indexer_batch_floats(s);
    const uint32_t ncs[] = {16385u, 32768u, 65536u, 131072u, 262144u};
    const uint32_t glens[] = {1u, 8u, 9u, 32u, 64u, 128u, 256u};

    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ++ni) {
        const uint32_t nc = ncs[ni];
        for (size_t gi = 0; gi < sizeof(glens) / sizeof(glens[0]); ++gi) {
            const uint32_t glen = glens[gi];
            if (glen > max_rows || (uint64_t)glen * nc > (uint64_t)batch_floats) continue;
            fill_realistic_relu_scores(scores, glen, nc, rng);
            const uint32_t pos0 = nc - glen;

            memset(myds_metal_buffer_data(mask_ref), 0, (size_t)glen * mask_units * sizeof(uint32_t));
            memset(myds_metal_buffer_data(mask_got), 0xff, (size_t)glen * mask_units * sizeof(uint32_t));
            memset(myds_metal_buffer_data(sel_ref), 0xa5, (size_t)glen * 512u * sizeof(int32_t));
            memset(myds_metal_buffer_data(sel_got), 0x5a, (size_t)glen * 512u * sizeof(int32_t));

            run_l20_reference(s, scores, mask_ref, sel_ref, nc, glen, pos0, 1u, mask_units, mask_stride, true);
            myds_v41_gpu_topk_radix_causal_counters(NULL, true);
            run_l20_candidate(s, scores, mask_got, sel_got, nc, glen, pos0, 1u, mask_units, mask_stride, true, true);
            myds_v41_gpu_topk_radix_causal_counters_t ctr;
            myds_v41_gpu_topk_radix_causal_counters(&ctr, true);
            CHECK(ctr.l20_compact_encoded == 1u && ctr.l20_compact_rows == glen &&
                  ctr.l20_compact_fallbacks == 0u && ctr.failures == 0u);

            char lbl[128];
            snprintf(lbl, sizeof(lbl), "l20_bits nc=%u glen=%u", nc, glen);
            /* Note: maskbits_rows writes exact ceil(blocks/32) words per row at stride mask_units */
            const uint32_t active_words = (((nc + 7u) / 8u) + 31u) / 32u;
            for (uint32_t r = 0; r < glen; ++r) {
                const uint32_t *mr = (const uint32_t *)myds_metal_buffer_data(mask_ref) + (size_t)r * mask_units;
                const uint32_t *mg = (const uint32_t *)myds_metal_buffer_data(mask_got) + (size_t)r * mask_units;
                CHECK(memcmp(mr, mg, (size_t)active_words * sizeof(uint32_t)) == 0);
            }
            assert_ids_0ulp(lbl, sel_ref, sel_got, glen, 512u);

            /* Also verify mask_bits=false (float topkmask_rows) on representative glen=8 */
            if (glen == 8u && nc <= 65536u) {
                const uint32_t blocks = (nc + 7u) / 8u;
                run_l20_reference(s, scores, mask_f_ref, sel_ref, nc, glen, pos0, 1u, mask_units, mask_stride, false);
                run_l20_candidate(s, scores, mask_f_got, sel_got, nc, glen, pos0, 1u, mask_units, mask_stride, false, true);
                for (uint32_t r = 0; r < glen; ++r) {
                    const float *fr = (const float *)myds_metal_buffer_data(mask_f_ref) + (size_t)r * mask_stride;
                    const float *fg = (const float *)myds_metal_buffer_data(mask_f_got) + (size_t)r * mask_stride;
                    CHECK(memcmp(fr, fg, (size_t)blocks * sizeof(float)) == 0);
                }
                assert_ids_0ulp("l20_float_mask", sel_ref, sel_got, glen, 512u);
            }
        }
        printf("  L20 nc=%6u: mask_view + sel_view memcmp==0 across all glens -> PASS\n", nc);
    }

    /* L20 Adversarial Torture Cases */
    float *sp = (float *)myds_metal_buffer_data(scores);
    const uint32_t nc = 65536u, glen = 16u;

    /* 1) Unselected-block finite-score same-1024-partition bitonic tie + ReLU 0.0f heavy tie +
     *    hidden NaN inside an unselected low-max 8-key block + <512 finite -INF row + early pos0<16K */
    for (uint32_t r = 0; r < glen; ++r) {
        float *row = sp + (size_t)r * nc;
        if (r < 4u) {
            /* All 0.0f ReLU heavy tie across 65,536 cols */
            memset(row, 0, nc * sizeof(float));
        } else if (r < 8u) {
            /* Partition p=3 has unselected blocks with finite 0.37f..0.91f AND selected blocks tying at 9.5f */
            for (uint32_t c = 0; c < nc; ++c) {
                row[c] = 0.1f + 0.01f * (float)((c * 13u + r) % 73u);
            }
            for (uint32_t b = 0; b < 2048u; ++b) {
                row[(b * 3u + 1u) * 8u + (b & 7u)] = 5.0f + 0.001f * (float)(b % 997u);
            }
            /* Inject 6 same-partition ties at 9.5f inside p=3 separated by unselected finite blocks */
            for (uint32_t k = 0; k < 6u; ++k) {
                row[3u * 1024u + (k * 21u) * 8u + 3u] = 9.5f;
            }
        } else if (r < 12u) {
            /* Hidden NaN inside block b=5000 where max(best, NaN) in Metal returns finite best=0.2f
             * (so block 5000 would NOT even enter top-2048 blocks without candidate_blocks_nan's raw NaN check!) */
            fill_realistic_relu_scores(scores, 1u, nc, rng); /* temp into row 0 if r==0, copy safely */
            for (uint32_t c = 0; c < nc; ++c) row[c] = 1.0f + 0.01f * (float)(c % 500u);
            const uint32_t nan_blk = 5000u + r;
            for (uint32_t j = 0; j < 8u; ++j) row[nan_blk * 8u + j] = 0.05f;
            row[nan_blk * 8u + 5u] = bits_to_f32(0x7fc00077u);
        } else {
            /* <512 finite scores + -INFINITY everywhere else */
            for (uint32_t c = 0; c < nc; ++c) row[c] = -INFINITY;
            for (uint32_t k = 0; k < 111u; ++k) row[(k * 521u + r * 17u) % 65000u] = 2.0f - 0.01f * (float)k;
        }
    }
    const uint32_t active_words = ((nc / 8u) + 31u) / 32u;
    run_l20_reference(s, scores, mask_ref, sel_ref, nc, glen, nc - glen, 1u, mask_units, mask_stride, true);
    run_l20_candidate(s, scores, mask_got, sel_got, nc, glen, nc - glen, 1u, mask_units, mask_stride, true, true);
    for (uint32_t r = 0; r < glen; ++r) {
        /* On NaN rows (8..11), verify both mask_view AND sel_view! Wait: on NaN rows, topk_radix_causal_scoped
         * ran nan_fallback for block_sel_buf, so mask_got ALSO matches mask_ref 100% bit-for-bit! */
        const uint32_t *mr = (const uint32_t *)myds_metal_buffer_data(mask_ref) + (size_t)r * mask_units;
        const uint32_t *mg = (const uint32_t *)myds_metal_buffer_data(mask_got) + (size_t)r * mask_units;
        CHECK(memcmp(mr, mg, (size_t)active_words * sizeof(uint32_t)) == 0);
    }
    assert_ids_0ulp("l20_adversarial_zero_dense_replay_hidden_nan_inf", sel_ref, sel_got, glen, 512u);

    /* 2) Early causal rows (pos0=500 < 2048 bypasses compact cleanly, pos0=15000 < 16384 runs compact!) */
    fill_realistic_relu_scores(scores, 16u, 32768u, rng);
    run_l20_reference(s, scores, mask_ref, sel_ref, 32768u, 16u, 500u, 1u, mask_units, mask_stride, true);
    run_l20_candidate(s, scores, mask_got, sel_got, 32768u, 16u, 500u, 1u, mask_units, mask_stride, true, true);
    assert_ids_0ulp("l20_early_pos500", sel_ref, sel_got, 16u, 512u);

    run_l20_reference(s, scores, mask_ref, sel_ref, 32768u, 16u, 15000u, 1u, mask_units, mask_stride, true);
    run_l20_candidate(s, scores, mask_got, sel_got, 32768u, 16u, 15000u, 1u, mask_units, mask_stride, true, true);
    assert_ids_0ulp("l20_straddle_pos15000", sel_ref, sel_got, 16u, 512u);

    /* 3) Concurrent encoder scratch-reuse stress: 4 back-to-back sub-batches in ONE command buffer
     *    under BOTH (concurrent=1, auto_barrier=1) AND (concurrent=1, auto_barrier=0)! */
    const int prev_conc = myds_metal_get_concurrent();
    const int prev_ab = myds_metal_get_auto_barrier();
    fill_realistic_relu_scores(scores, 64u, 65536u, rng);
    myds_metal_buffer_t *sv_ref = myds_metal_buffer_create_view();
    myds_metal_buffer_t *sv_got = myds_metal_buffer_create_view();
    myds_metal_buffer_t *mv_ref = myds_metal_buffer_create_view();
    myds_metal_buffer_t *mv_got = myds_metal_buffer_create_view();
    myds_metal_buffer_t *sc_v   = myds_metal_buffer_create_view();
    CHECK(sv_ref && sv_got && mv_ref && mv_got && sc_v);

    for (int ab = 1; ab >= 0; --ab) {
        myds_metal_set_concurrent(1);
        myds_metal_set_auto_barrier(ab);

        setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "0", 1);
        CHECK(myds_metal_begin_batch());
        for (uint32_t b = 0; b < 4u; ++b) {
            CHECK(myds_metal_buffer_update_sub_view(sc_v, scores,
                (size_t)b * 16u * 65536u * sizeof(float), 16u * 65536u * sizeof(float)));
            CHECK(myds_metal_buffer_update_sub_view(sv_ref, sel_ref,
                (size_t)b * 16u * 512u * sizeof(int32_t), 16u * 512u * sizeof(int32_t)));
            CHECK(myds_metal_buffer_update_sub_view(mv_ref, mask_ref,
                (size_t)b * 16u * mask_units * sizeof(uint32_t), 16u * mask_units * sizeof(uint32_t)));
            if (b & 1u) {
                CHECK(myds_v41_gpu_topk_radix_causal_scoped(
                    s, sv_ref, sc_v, 65536u, 512u, 16u, 65520u, 2u, true) == 1);
            } else {
                myds_metal_buffer_t *bb = myds_v41_gpu_indexer_blocks(s);
                myds_metal_buffer_t *bs = myds_v41_gpu_indexer_block_selected(s);
                if (!ab) {
                    const myds_metal_buffer_t *deps[] = {sc_v, bb, bs, mv_ref, sv_ref};
                    myds_metal_barrier_buffers(deps, 5u);
                }
                CHECK(myds_v41_gpu_candidates_rows(bb, sc_v, 65536u, 16u, 65520u, 1u, mask_stride));
                if (!ab) myds_metal_barrier_buffer(bb);
                CHECK(myds_v41_gpu_topk_causal(bs, bb, 8192u, 2048u, 16u, 65520u, 1u, s));
                if (!ab) myds_metal_barrier_buffer(bs);
                CHECK(myds_v41_gpu_maskbits_rows(mv_ref, bs, 8192u, 2048u, 16u, mask_units));
                if (!ab) myds_metal_barrier_buffer(mv_ref);
                CHECK(myds_v41_gpu_topk_causal(sv_ref, sc_v, 65536u, 512u, 16u, 65520u, 1u, s));
                if (!ab) myds_metal_barrier_buffer(sv_ref);
            }
        }
        CHECK(myds_metal_commit_batch());

        setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "1", 1);
        CHECK(myds_metal_begin_batch());
        for (uint32_t b = 0; b < 4u; ++b) {
            CHECK(myds_metal_buffer_update_sub_view(sc_v, scores,
                (size_t)b * 16u * 65536u * sizeof(float), 16u * 65536u * sizeof(float)));
            CHECK(myds_metal_buffer_update_sub_view(sv_got, sel_got,
                (size_t)b * 16u * 512u * sizeof(int32_t), 16u * 512u * sizeof(int32_t)));
            CHECK(myds_metal_buffer_update_sub_view(mv_got, mask_got,
                (size_t)b * 16u * mask_units * sizeof(uint32_t), 16u * mask_units * sizeof(uint32_t)));
            if (b & 1u) {
                CHECK(myds_v41_gpu_topk_radix_causal_scoped(
                    s, sv_got, sc_v, 65536u, 512u, 16u, 65520u, 2u, true) == 1);
            } else {
                CHECK(myds_v41_gpu_indexer_l20_compact_topk_scoped(
                    s, sv_got, mv_got, sc_v, 65536u, 16u, 65520u, 1u, mask_units, true, 2048u, true) == 1);
            }
        }
        CHECK(myds_metal_commit_batch());

        char clbl[64];
        snprintf(clbl, sizeof(clbl), "concurrent_4batch_reuse_ab%d", ab);
        assert_ids_0ulp(clbl, sel_ref, sel_got, 64u, 512u);

        /* 4) Concurrent WAR regression at nc=24,576 (blocks=3,072 < 4,096, top_blocks=2,048):
         *    Step 3 delegates to prefix=false topk_rows_causal while sub-batch 0 has a NaN row
         *    touching chunk_tmp_a/chunk_tmp_b and sub-batches 1..3 are normal L20 rows! */
        fill_realistic_relu_scores(scores, 64u, 24576u, rng);
        sp[3u * 24576u + 111u] = bits_to_f32(0x7fc00123u); /* NaN row in sub-batch 0 */

        myds_metal_set_auto_barrier(1);
        setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "0", 1);
        CHECK(myds_metal_begin_batch());
        for (uint32_t b = 0; b < 4u; ++b) {
            CHECK(myds_metal_buffer_update_sub_view(sc_v, scores,
                (size_t)b * 16u * 24576u * sizeof(float), 16u * 24576u * sizeof(float)));
            CHECK(myds_metal_buffer_update_sub_view(sv_ref, sel_ref,
                (size_t)b * 16u * 512u * sizeof(int32_t), 16u * 512u * sizeof(int32_t)));
            CHECK(myds_metal_buffer_update_sub_view(mv_ref, mask_ref,
                (size_t)b * 16u * mask_units * sizeof(uint32_t), 16u * mask_units * sizeof(uint32_t)));
            myds_metal_buffer_t *bb = myds_v41_gpu_indexer_blocks(s);
            myds_metal_buffer_t *bs = myds_v41_gpu_indexer_block_selected(s);
            CHECK(myds_v41_gpu_candidates_rows(bb, sc_v, 24576u, 16u, 24560u, 1u, mask_stride));
            CHECK(myds_v41_gpu_topk_causal(bs, bb, 3072u, 2048u, 16u, 24560u, 1u, s));
            CHECK(myds_v41_gpu_maskbits_rows(mv_ref, bs, 3072u, 2048u, 16u, mask_units));
            CHECK(myds_v41_gpu_topk_causal(sv_ref, sc_v, 24576u, 512u, 16u, 24560u, 1u, s));
        }
        CHECK(myds_metal_commit_batch());

        myds_metal_set_auto_barrier(ab);
        memset(myds_metal_buffer_data(sel_got), 0x5a, 64u * 512u * sizeof(int32_t));
        memset(myds_metal_buffer_data(mask_got), 0xff, 64u * mask_units * sizeof(uint32_t));
        setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "1", 1);
        CHECK(myds_metal_begin_batch());
        for (uint32_t b = 0; b < 4u; ++b) {
            CHECK(myds_metal_buffer_update_sub_view(sc_v, scores,
                (size_t)b * 16u * 24576u * sizeof(float), 16u * 24576u * sizeof(float)));
            CHECK(myds_metal_buffer_update_sub_view(sv_got, sel_got,
                (size_t)b * 16u * 512u * sizeof(int32_t), 16u * 512u * sizeof(int32_t)));
            CHECK(myds_metal_buffer_update_sub_view(mv_got, mask_got,
                (size_t)b * 16u * mask_units * sizeof(uint32_t), 16u * mask_units * sizeof(uint32_t)));
            CHECK(myds_v41_gpu_indexer_l20_compact_topk_scoped(
                s, sv_got, mv_got, sc_v, 24576u, 16u, 24560u, 1u, mask_units, true, 2048u, true) == 1);
        }
        CHECK(myds_metal_commit_batch());

        const uint32_t w24k = (3072u + 31u) / 32u;
        for (uint32_t r = 0; r < 64u; ++r) {
            const uint32_t *mr = (const uint32_t *)myds_metal_buffer_data(mask_ref) + (size_t)r * mask_units;
            const uint32_t *mg = (const uint32_t *)myds_metal_buffer_data(mask_got) + (size_t)r * mask_units;
            CHECK(memcmp(mr, mg, (size_t)w24k * sizeof(uint32_t)) == 0);
        }
        snprintf(clbl, sizeof(clbl), "concurrent_4batch_nc24576_war_ab%d", ab);
        assert_ids_0ulp(clbl, sel_ref, sel_got, 64u, 512u);
    }

    myds_metal_buffer_free(sc_v);
    myds_metal_buffer_free(mv_got);
    myds_metal_buffer_free(mv_ref);
    myds_metal_buffer_free(sv_got);
    myds_metal_buffer_free(sv_ref);
    myds_metal_set_concurrent(prev_conc);
    myds_metal_set_auto_barrier(prev_ab);
    unsetenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL");
    printf("  L20 dense-replay ties, hidden-NaN block, <512 -INF, straddle, 4-batch concurrent ab={0,1} -> PASS\n");
}

static void run_benchmarks(myds_v41_gpu_scratch_t *s,
                           myds_metal_buffer_t *scores,
                           myds_metal_buffer_t *sel_ref,
                           myds_metal_buffer_t *sel_got,
                           myds_metal_buffer_t *mask_ref,
                           myds_metal_buffer_t *mask_got,
                           uint32_t mask_units,
                           uint32_t mask_stride,
                           uint32_t *rng) {
    printf("[6/6] Running 7-run median GPU microbenchmark vs Production Baseline (PREFIX_MERGE=1, NAN_FUSE=1)...\n");
    const int prev_conc = myds_metal_get_concurrent();
    const int prev_ab = myds_metal_get_auto_barrier();
    myds_metal_set_concurrent(1);
    myds_metal_set_auto_barrier(0);
    set_prod_baseline_knobs();

    struct {
        const char *name;
        uint32_t nc;
        uint32_t glen;
        uint32_t ratio;
    } p1_cases[] = {
        {"L2/8/14_32K_g256",   32768u, 256u, 2u},
        {"L2/8/14_64K_g128",   65536u, 128u, 2u},
        {"L2/8/14_128K_g64",  131072u,  64u, 2u},
        {"L2/8/14_256K_g32",  262144u,  32u, 2u},
        {"L2/8/14_128K_warm16", 131072u, 16u, 2u},
    };

    printf("  --- Part 1: Causal Top-512 (per sub-batch & x3 root layers L2/8/14) ---\n");
    for (size_t i = 0; i < sizeof(p1_cases) / sizeof(p1_cases[0]); ++i) {
        const uint32_t nc = p1_cases[i].nc;
        const uint32_t glen = p1_cases[i].glen;
        const uint32_t ratio = p1_cases[i].ratio;
        const uint32_t pos0 = nc * ratio - glen;
        fill_realistic_relu_scores(scores, glen, nc, rng);

        /* Warmup + 0-ULP check */
        run_part1_pair(p1_cases[i].name, s, scores, sel_ref, sel_got, nc, 512u, glen, pos0, ratio, true);

        double t_off[7], t_on[7];
        for (int r = 0; r < 7; ++r) {
            setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "0", 1);
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_topk_radix_causal_scoped(
                s, sel_ref, scores, nc, 512u, glen, pos0, ratio, true) == 1);
            CHECK(myds_metal_commit_batch());
            t_off[r] = monotonic_ms() - s0;

            setenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL", "1", 1);
            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_topk_radix_causal_scoped(
                s, sel_got, scores, nc, 512u, glen, pos0, ratio, true) == 1);
            CHECK(myds_metal_commit_batch());
            t_on[r] = monotonic_ms() - s1;
        }
        qsort(t_off, 7, sizeof(double), cmp_double);
        qsort(t_on,  7, sizeof(double), cmp_double);
        printf("  %-20s (nc=%6u, glen=%3u): PT1=0 %6.2f ms (x3=%6.2f ms) | PT1=1 %6.2f ms (x3=%6.2f ms) | %.2fx speedup\n",
               p1_cases[i].name, nc, glen, t_off[3], t_off[3] * 3.0, t_on[3], t_on[3] * 3.0, t_off[3] / t_on[3]);
    }

    struct {
        const char *name;
        uint32_t nc;
        uint32_t glen;
    } l20_cases[] = {
        {"L20_32K_g256",   32768u, 256u},
        {"L20_64K_g128",   65536u, 128u},
        {"L20_128K_g64",  131072u,  64u},
        {"L20_256K_g32",  262144u,  32u},
        {"L20_128K_warm16", 131072u, 16u},
    };

    printf("  --- Part 2: Full L20 Post-Scorer (candidates + top_blocks_2048 + maskbits + top_512) ---\n");
    for (size_t i = 0; i < sizeof(l20_cases) / sizeof(l20_cases[0]); ++i) {
        const uint32_t nc = l20_cases[i].nc;
        const uint32_t glen = l20_cases[i].glen;
        const uint32_t pos0 = nc - glen;
        fill_realistic_relu_scores(scores, glen, nc, rng);

        /* Warmup + 0-ULP check */
        run_l20_candidate(s, scores, mask_ref, sel_ref, nc, glen, pos0, 1u, mask_units, mask_stride, true, false);
        run_l20_candidate(s, scores, mask_got, sel_got, nc, glen, pos0, 1u, mask_units, mask_stride, true, true);
        assert_ids_0ulp(l20_cases[i].name, sel_ref, sel_got, glen, 512u);

        double t_off[7], t_on[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            run_l20_candidate(s, scores, mask_ref, sel_ref, nc, glen, pos0, 1u, mask_units, mask_stride, true, false);
            t_off[r] = monotonic_ms() - s0;

            const double s1 = monotonic_ms();
            run_l20_candidate(s, scores, mask_got, sel_got, nc, glen, pos0, 1u, mask_units, mask_stride, true, true);
            t_on[r] = monotonic_ms() - s1;
        }
        qsort(t_off, 7, sizeof(double), cmp_double);
        qsort(t_on,  7, sizeof(double), cmp_double);
        printf("  %-20s (nc=%6u, glen=%3u): PT1=0 %6.2f ms | PT1=1 %6.2f ms | %.2fx speedup\n",
               l20_cases[i].name, nc, glen, t_off[3], t_on[3], t_off[3] / t_on[3]);
    }

    unsetenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL");
    myds_metal_set_concurrent(prev_conc);
    myds_metal_set_auto_barrier(prev_ab);
}

int main(int argc, char **argv) {
    bool run_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) run_bench = true;
    }

    unsetenv("MYDS_INDEXER_TOPK_RADIX_CAUSAL");
    set_prod_baseline_knobs();

    CHECK(myds_metal_init(NULL));
    verify_pso_occupancy();

    const uint32_t max_nc = 262144u;
    const uint32_t max_rows = run_bench ? 2048u : 512u;
    const uint32_t max_blocks = (max_nc + 7u) / 8u;
    const uint32_t mask_units = myds_v41_gpu_maskbits_stride(max_blocks);
    const uint32_t mask_stride = (max_blocks + 3u) & ~3u;

    setenv("MYDS_CHUNK_KEY_MAX", "262144", 1);
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(max_nc);
    unsetenv("MYDS_CHUNK_KEY_MAX");
    CHECK(s != NULL);

    myds_metal_buffer_t *scores = myds_v41_gpu_indexer_scores(s);
    myds_metal_buffer_t *sel_ref = myds_metal_buffer_alloc((size_t)max_rows * 2048u * sizeof(int32_t));
    myds_metal_buffer_t *sel_got = myds_metal_buffer_alloc((size_t)max_rows * 2048u * sizeof(int32_t));
    myds_metal_buffer_t *mask_ref = myds_metal_buffer_alloc((size_t)max_rows * mask_units * sizeof(uint32_t));
    myds_metal_buffer_t *mask_got = myds_metal_buffer_alloc((size_t)max_rows * mask_units * sizeof(uint32_t));
    myds_metal_buffer_t *mask_f_ref = myds_metal_buffer_alloc(16u * (size_t)mask_stride * sizeof(float));
    myds_metal_buffer_t *mask_f_got = myds_metal_buffer_alloc(16u * (size_t)mask_stride * sizeof(float));
    CHECK(scores && sel_ref && sel_got && mask_ref && mask_got && mask_f_ref && mask_f_got);

    uint32_t rng = 0x20261115u;
    fill_realistic_relu_scores(scores, 16u, 32768u, &rng);

    verify_knob_and_contracts(s, scores, sel_ref, sel_got, mask_ref, mask_got, mask_units, mask_stride);
    verify_part1_matrix_and_torture(s, scores, sel_ref, sel_got, max_rows, &rng);
    verify_part2_l20_matrix_and_torture(s, scores, sel_ref, sel_got, mask_ref, mask_got,
                                        mask_f_ref, mask_f_got, max_rows, mask_units, mask_stride, &rng);

    if (run_bench) {
        run_benchmarks(s, scores, sel_ref, sel_got, mask_ref, mask_got, mask_units, mask_stride, &rng);
    } else {
        printf("[6/6] Microbenchmark skipped (pass --bench to run 32K..256K 7-run medians).\n");
    }

    myds_metal_buffer_free(mask_f_got);
    myds_metal_buffer_free(mask_f_ref);
    myds_metal_buffer_free(mask_got);
    myds_metal_buffer_free(mask_ref);
    myds_metal_buffer_free(sel_got);
    myds_metal_buffer_free(sel_ref);
    myds_v41_gpu_scratch_free(s);
    myds_metal_cleanup();
    printf("PASS: test_indexer_topk_radix_pt1 completed with 100%% 0-ULP parity.\n");
    return 0;
}
