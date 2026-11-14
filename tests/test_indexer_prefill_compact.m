/* tests/test_indexer_prefill_compact.m
 *
 * [P-L1] Apple Silicon GPU 0-ULP parity, PSO occupancy, fail-closed/fallback
 * contract, and microbenchmark suite for Prefill YoCo (il > 20, ratio == 1,
 * mask_bits, nc > 16384) 8-Row Tile-Union Register-Only Scorer + Batched
 * <=16,384 Compact Top-512.
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

static uint16_t f32_to_bf16_bits(float v) {
    uint32_t u = 0;
    memcpy(&u, &v, sizeof(u));
    return (uint16_t)(u >> 16u);
}

static void fill_bf16_cache(myds_metal_buffer_t *cache, uint32_t rows,
                            uint32_t *rng, bool all_positive) {
    uint16_t *dst = (uint16_t *)myds_metal_buffer_data(cache);
    CHECK(dst != NULL);
    for (size_t i = 0; i < (size_t)rows * 128u; ++i) {
        float v = rand_signed(rng, 1.0f);
        if (all_positive) v = fabsf(v) + 0.05f;
        dst[i] = f32_to_bf16_bits(v);
    }
}

static void stamp_heavy_tie_cache(myds_metal_buffer_t *cache, uint32_t rows) {
    uint16_t *pk = (uint16_t *)myds_metal_buffer_data(cache);
    const uint32_t palette_len = 16u;
    CHECK(rows >= palette_len);
    for (uint32_t r = palette_len; r < rows; ++r) {
        const uint32_t src = (r * 7u + (r / 8u) * 3u + (r / 1024u) * 5u) % palette_len;
        memcpy(pk + (size_t)r * 128u, pk + (size_t)src * 128u, 128u * sizeof(uint16_t));
    }
}

static void fill_q_weights(myds_metal_buffer_t *q, myds_metal_buffer_t *w,
                           uint32_t rows, uint32_t *rng, bool force_negative_q) {
    float *qp = (float *)myds_metal_buffer_data(q);
    float *wp = (float *)myds_metal_buffer_data(w);
    CHECK(qp != NULL && wp != NULL);
    for (size_t i = 0; i < (size_t)rows * 32u * 128u; ++i) {
        float v = rand_signed(rng, 0.9f);
        if (force_negative_q) v = -fabsf(v) - 0.05f;
        qp[i] = v;
    }
    for (size_t i = 0; i < (size_t)rows * 32u; ++i) {
        wp[i] = 0.05f + 0.95f * fabsf(rand_signed(rng, 1.0f));
    }
}

typedef enum {
    MASK_GPU_L20 = 0,
    MASK_DISJOINT_8ROW,
    MASK_IDENTICAL_8ROW,
    MASK_SPARSE_LT512
} mask_mode_t;

static void build_synthetic_mask(myds_metal_buffer_t *mask_buf, uint32_t rows,
                                 uint32_t pos0, uint32_t n_comp,
                                 uint32_t mask_units, mask_mode_t mode,
                                 uint32_t *rng) {
    uint32_t *mw = (uint32_t *)myds_metal_buffer_data(mask_buf);
    CHECK(mw != NULL);
    memset(mw, 0, (size_t)rows * mask_units * sizeof(uint32_t));

    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t vis = pos0 + r + 1u;
        if (vis > n_comp) vis = n_comp;
        const uint32_t n_blks = (vis + 7u) / 8u;
        uint32_t *row_w = mw + (size_t)r * mask_units;

        if (mode == MASK_SPARSE_LT512) {
            /* Only 17 blocks = 136 < 512 finite keys -> exercises Partition-0 -INF tail replay. */
            const uint32_t keep = n_blks < 17u ? n_blks : 17u;
            for (uint32_t k = 0; k < keep; ++k) {
                const uint32_t blk = (k * 97u + r * 13u) % n_blks;
                row_w[blk >> 5u] |= (1u << (blk & 31u));
            }
            continue;
        }

        const uint32_t want = n_blks < 2048u ? n_blks : 2048u;
        if (mode == MASK_DISJOINT_8ROW && n_blks >= 16384u) {
            /* Each row j in an 8-row tile picks a completely disjoint 2048-block set -> 16384 union! */
            const uint32_t j = r & 7u;
            const uint32_t stride = n_blks / 2048u; /* >= 8 */
            for (uint32_t k = 0; k < 2048u; ++k) {
                const uint32_t blk = k * stride + j;
                CHECK(blk < n_blks);
                row_w[blk >> 5u] |= (1u << (blk & 31u));
            }
        } else if (mode == MASK_IDENTICAL_8ROW) {
            /* All rows in an 8-row tile share the exact same 2048 blocks -> 2048 union! */
            const uint32_t tile_vis = (pos0 + (r & ~7u) + 1u) < n_comp ? (pos0 + (r & ~7u) + 1u) : n_comp;
            const uint32_t tile_blks = (tile_vis + 7u) / 8u;
            const uint32_t t_want = tile_blks < 2048u ? tile_blks : 2048u;
            uint32_t tile_seed = 0x9e3779b9u ^ ((r >> 3u) * 0x85ebca6bu);
            uint32_t picked = 0u;
            for (uint32_t b = 0; b < tile_blks && picked < t_want; ++b) {
                const uint32_t rem_b = tile_blks - b;
                const uint32_t need = t_want - picked;
                if ((xorshift32(&tile_seed) % rem_b) < need) {
                    row_w[b >> 5u] |= (1u << (b & 31u));
                    picked++;
                }
            }
            CHECK(picked == t_want);
        } else {
            /* Reservoir/Bernoulli exact `want` blocks per row. */
            uint32_t picked = 0u;
            for (uint32_t b = 0; b < n_blks && picked < want; ++b) {
                const uint32_t rem_b = n_blks - b;
                const uint32_t need = want - picked;
                if ((xorshift32(rng) % rem_b) < need) {
                    row_w[b >> 5u] |= (1u << (b & 31u));
                    picked++;
                }
            }
            CHECK(picked == want);
        }
    }
}

static uint32_t host_dense_group_len(uint32_t c_len, uint32_t off, uint32_t base_pos,
                                     uint32_t batch_floats) {
    const uint32_t rem = c_len - off;
    uint32_t glen = rem;
    for (;;) {
        const uint32_t nc = base_pos + off + glen;
        const uint32_t width = nc < 512u ? 512u : nc;
        if ((uint64_t)glen * width <= (uint64_t)batch_floats) {
            if (glen > 8u && glen < rem) {
                const uint32_t max_w = (base_pos + c_len) < 512u ? 512u : (base_pos + c_len);
                const uint32_t min_g = (uint32_t)((uint64_t)batch_floats / max_w);
                if (min_g >= 8u && (rem + min_g - 1u) / min_g <= rem / 8u) {
                    while (glen > 8u) {
                        const uint32_t r = rem - glen;
                        if ((r + min_g - 1u) / min_g <= r / 8u) break;
                        --glen;
                    }
                }
            }
            return glen;
        }
        const uint32_t next = glen > 1u ? (uint32_t)((uint64_t)batch_floats / width) : 0u;
        CHECK(next > 0u && next < glen);
        glen = next;
    }
}

static void generate_gpu_l20_mask(myds_v41_gpu_scratch_t *s,
                                  const myds_metal_buffer_t *q20,
                                  const myds_metal_buffer_t *w20,
                                  const myds_metal_buffer_t *cache,
                                  myds_metal_buffer_t *mask_buf,
                                  myds_metal_buffer_t *q_view,
                                  myds_metal_buffer_t *w_view,
                                  myds_metal_buffer_t *m_view,
                                  uint32_t c_len, uint32_t pos0,
                                  uint32_t mask_units) {
    const uint32_t batch_floats = myds_v41_gpu_indexer_batch_floats(s);
    myds_metal_buffer_t *scores_buf = myds_v41_gpu_indexer_scores(s);
    myds_metal_buffer_t *blocks_buf = myds_v41_gpu_indexer_blocks(s);
    myds_metal_buffer_t *block_sel_buf = myds_v41_gpu_indexer_block_selected(s);

    CHECK(myds_metal_begin_batch());
    for (uint32_t off = 0; off < c_len; ) {
        const uint32_t glen = host_dense_group_len(c_len, off, pos0, batch_floats);
        const uint32_t g_start = pos0 + off;
        const uint32_t nc = (g_start + glen) < 512u ? 512u : (g_start + glen);
        const uint32_t blocks = (nc + 7u) / 8u;
        const uint32_t top_blocks = blocks < 2048u ? blocks : 2048u;

        CHECK(myds_metal_buffer_update_sub_view(q_view, q20,
            (size_t)off * 32u * 128u * sizeof(float),
            (size_t)glen * 32u * 128u * sizeof(float)));
        CHECK(myds_metal_buffer_update_sub_view(w_view, w20,
            (size_t)off * 32u * sizeof(float),
            (size_t)glen * 32u * sizeof(float)));
        CHECK(myds_metal_buffer_update_sub_view(m_view, mask_buf,
            (size_t)off * mask_units * sizeof(uint32_t),
            (size_t)glen * mask_units * sizeof(uint32_t)));

        if (!myds_metal_get_auto_barrier()) {
            const myds_metal_buffer_t *deps[] = {q_view, w_view, cache, scores_buf};
            myds_metal_barrier_buffers(deps, 4u);
        }
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(
            s, scores_buf, q_view, w_view, cache, nc, 32u, glen, g_start, 1u, true, true));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(scores_buf);
        CHECK(myds_v41_gpu_candidates_rows(blocks_buf, scores_buf, nc, glen, g_start, 1u, mask_units));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(blocks_buf);
        CHECK(myds_v41_gpu_topk_causal(block_sel_buf, blocks_buf, blocks, top_blocks, glen, g_start, 1u, s));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(block_sel_buf);
        CHECK(myds_v41_gpu_maskbits_rows(m_view, block_sel_buf, blocks, top_blocks, glen, mask_units));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(m_view);
        off += glen;
    }
    CHECK(myds_metal_commit_batch());
}

/* Replicates publish_indexer_chunk's exact post-RoPE il > 20 sub-batch loop. */
static void run_consumer_chunk(myds_v41_gpu_scratch_t *s,
                               const myds_metal_buffer_t *q_buf,
                               const myds_metal_buffer_t *w_buf,
                               const myds_metal_buffer_t *cache,
                               const myds_metal_buffer_t *mask_buf,
                               myds_metal_buffer_t *sel_buf,
                               myds_metal_buffer_t *q_view,
                               myds_metal_buffer_t *w_view,
                               myds_metal_buffer_t *s_view,
                               myds_metal_buffer_t *m_view,
                               uint32_t c_len, uint32_t pos0,
                               uint32_t mask_units, bool compact_enabled) {
    setenv("MYDS_INDEXER_PREFILL_COMPACT", compact_enabled ? "1" : "0", 1);
    const uint32_t batch_floats = myds_v41_gpu_indexer_batch_floats(s);
    const uint32_t max_compact_glen = myds_v41_gpu_indexer_prefill_compact_max_glen(s);
    myds_metal_buffer_t *scores_buf = myds_v41_gpu_indexer_scores(s);
    bool allow_compact = true;

    CHECK(myds_metal_begin_batch());
    for (uint32_t off = 0; off < c_len; ) {
        const uint32_t rem = c_len - off;
        const uint32_t g_start = pos0 + off;
        if (allow_compact && max_compact_glen >= 8u && rem >= 8u &&
            g_start >= 16384u && (uint64_t)g_start + rem <= 1048576u) {
            uint32_t c_glen = rem <= max_compact_glen ? rem : max_compact_glen;
            if (c_glen < rem && rem - c_glen < 8u && c_glen >= 16u) c_glen -= 8u;
            const uint32_t c_nc = g_start + c_glen;

            CHECK(myds_metal_buffer_update_sub_view(q_view, q_buf,
                (size_t)off * 32u * 128u * sizeof(float),
                (size_t)c_glen * 32u * 128u * sizeof(float)));
            CHECK(myds_metal_buffer_update_sub_view(w_view, w_buf,
                (size_t)off * 32u * sizeof(float),
                (size_t)c_glen * 32u * sizeof(float)));
            CHECK(myds_metal_buffer_update_sub_view(s_view, sel_buf,
                (size_t)off * 512u * sizeof(int32_t),
                (size_t)c_glen * 512u * sizeof(int32_t)));
            CHECK(myds_metal_buffer_update_sub_view(m_view, mask_buf,
                (size_t)off * mask_units * sizeof(uint32_t),
                (size_t)c_glen * mask_units * sizeof(uint32_t)));

            const int crc = myds_v41_gpu_indexer_prefill_compact_scoped(
                s, s_view, q_view, w_view, cache, m_view,
                c_nc, c_glen, g_start, 1u, mask_units, 2048u, true);
            CHECK(crc >= 0);
            if (crc > 0) {
                off += c_glen;
                continue;
            }
            allow_compact = false;
        }

        const uint32_t glen = host_dense_group_len(c_len, off, pos0, batch_floats);
        const uint32_t nc = (g_start + glen) < 512u ? 512u : (g_start + glen);
        CHECK(myds_metal_buffer_update_sub_view(q_view, q_buf,
            (size_t)off * 32u * 128u * sizeof(float),
            (size_t)glen * 32u * 128u * sizeof(float)));
        CHECK(myds_metal_buffer_update_sub_view(w_view, w_buf,
            (size_t)off * 32u * sizeof(float),
            (size_t)glen * 32u * sizeof(float)));
        CHECK(myds_metal_buffer_update_sub_view(s_view, sel_buf,
            (size_t)off * 512u * sizeof(int32_t),
            (size_t)glen * 512u * sizeof(int32_t)));
        CHECK(myds_metal_buffer_update_sub_view(m_view, mask_buf,
            (size_t)off * mask_units * sizeof(uint32_t),
            (size_t)glen * mask_units * sizeof(uint32_t)));

        if (!myds_metal_get_auto_barrier()) {
            const myds_metal_buffer_t *deps[] = {q_view, w_view, cache, m_view, scores_buf};
            myds_metal_barrier_buffers(deps, 5u);
        }
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(
            s, scores_buf, q_view, w_view, cache, nc, 32u, glen, g_start, 1u, true, true));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(scores_buf);
        CHECK(myds_v41_gpu_filter_rows_bits(scores_buf, m_view, nc, glen, g_start, 1u, mask_units));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(scores_buf);
        CHECK(myds_v41_gpu_topk_causal(s_view, scores_buf, nc, 512u, glen, g_start, 1u, s));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(s_view);
        off += glen;
    }
    CHECK(myds_metal_commit_batch());
    unsetenv("MYDS_INDEXER_PREFILL_COMPACT");
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
            NSUInteger exact_smem_or_max;
            bool exact_smem;
        } specs[] = {
            {"kernel_myds_indexer_prefill_compact_prep",          1024u, 2048u,  false},
            {"kernel_myds_indexer_scores_prefill_compact_ro_bf16", 1024u, 4096u,  true},
            {"kernel_myds_indexer_scores_prefill_compact_ro_0b_bf16", 1024u, 0u,  true},
            {"kernel_myds_prefill_compact_topk_stage1",            512u, 8192u,  true},
            {"kernel_myds_prefill_compact_topk_stage2",            512u, 8256u,  false},
            {"kernel_myds_prefill_compact_topk_final",            1024u, 24576u, false},
        };

        printf("[1/5] Verifying [P-L1] Metal PSO occupancy & static threadgroup memory...\n");
        for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); ++i) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:specs[i].name]];
            CHECK(fn != nil);
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            CHECK(pso != nil && err == nil);
            CHECK(pso.threadExecutionWidth == 32u);
            CHECK(pso.maxTotalThreadsPerThreadgroup >= specs[i].min_max_threads);
            if (specs[i].exact_smem) {
                CHECK(pso.staticThreadgroupMemoryLength == specs[i].exact_smem_or_max);
            } else {
                CHECK(pso.staticThreadgroupMemoryLength <= specs[i].exact_smem_or_max);
            }
            printf("  %-52s width=%lu max_threads=%4lu static_smem=%5lu B\n",
                   specs[i].name,
                   (unsigned long)pso.threadExecutionWidth,
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)pso.staticThreadgroupMemoryLength);
        }
        /* 4096 B smem with 128 threads/TG -> 32768 / 4096 = 8 TGs/core = 1024 threads/core (100% occupancy). */
    }
}

static void verify_knob_and_contracts(myds_v41_gpu_scratch_t *s,
                                      const myds_metal_buffer_t *q,
                                      const myds_metal_buffer_t *w,
                                      const myds_metal_buffer_t *cache,
                                      const myds_metal_buffer_t *mask,
                                      myds_metal_buffer_t *sel,
                                      uint32_t mask_units) {
    printf("[2/5] Verifying MYDS_INDEXER_PREFILL_COMPACT knob parsing, fail-closed & fallback...\n");
    unsetenv("MYDS_INDEXER_PREFILL_COMPACT");
    CHECK(myds_v41_gpu_indexer_prefill_compact_enabled() == 1);
    setenv("MYDS_INDEXER_PREFILL_COMPACT", "1", 1);
    CHECK(myds_v41_gpu_indexer_prefill_compact_enabled() == 1);
    setenv("MYDS_INDEXER_PREFILL_COMPACT", "0", 1);
    CHECK(myds_v41_gpu_indexer_prefill_compact_enabled() == 0);

    const char *bad_vals[] = {"", "2", "true", "false", "01", " 1", "1 "};
    for (size_t i = 0; i < sizeof(bad_vals) / sizeof(bad_vals[0]); ++i) {
        setenv("MYDS_INDEXER_PREFILL_COMPACT", bad_vals[i], 1);
        CHECK(myds_v41_gpu_indexer_prefill_compact_enabled() == -1);
        CHECK(myds_v41_gpu_scratch_create(65536u) == NULL);
    }

    /* Fail-closed in scoped selector on invalid env */
    myds_v41_gpu_indexer_prefill_compact_counters(NULL, true);
    setenv("MYDS_INDEXER_PREFILL_COMPACT", "2", 1);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_indexer_prefill_compact_scoped(
        s, sel, q, w, cache, mask, 32768u, 8u, 32760u, 1u, mask_units, 2048u, true) == -1);
    CHECK(myds_metal_commit_batch());
    myds_v41_gpu_indexer_prefill_compact_counters_t ctr;
    myds_v41_gpu_indexer_prefill_compact_counters(&ctr, true);
    CHECK(ctr.requests == 1u && ctr.encoded == 0u && ctr.fallbacks == 0u && ctr.failures == 1u);

    /* Disabled mode: requests=1, encoded=0, fallbacks=0, failures=0 */
    setenv("MYDS_INDEXER_PREFILL_COMPACT", "0", 1);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_indexer_prefill_compact_scoped(
        s, sel, q, w, cache, mask, 32768u, 8u, 32760u, 1u, mask_units, 2048u, true) == 0);
    CHECK(myds_metal_commit_batch());
    myds_v41_gpu_indexer_prefill_compact_counters(&ctr, true);
    CHECK(ctr.requests == 1u && ctr.encoded == 0u && ctr.fallbacks == 0u && ctr.failures == 0u);

    /* Diagnostic unavailable injection -> graceful fallback */
    setenv("MYDS_INDEXER_PREFILL_COMPACT", "1", 1);
    myds_v41_gpu_inject_indexer_prefill_compact_unavailable(true);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_indexer_prefill_compact_scoped(
        s, sel, q, w, cache, mask, 32768u, 8u, 32760u, 1u, mask_units, 2048u, true) == 0);
    CHECK(myds_metal_commit_batch());
    myds_v41_gpu_inject_indexer_prefill_compact_unavailable(false);
    myds_v41_gpu_indexer_prefill_compact_counters(&ctr, true);
    CHECK(ctr.requests == 1u && ctr.encoded == 0u && ctr.fallbacks == 1u && ctr.failures == 0u);

    /* Invalid MYDS_INDEXER_RO_SMEM_Q fails closed */
    setenv("MYDS_INDEXER_RO_SMEM_Q", "2", 1);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_indexer_prefill_compact_scoped(
        s, sel, q, w, cache, mask, 32768u, 8u, 32760u, 1u, mask_units, 2048u, true) == -1);
    CHECK(myds_metal_commit_batch());
    unsetenv("MYDS_INDEXER_RO_SMEM_Q");
    unsetenv("MYDS_INDEXER_PREFILL_COMPACT");
    myds_v41_gpu_indexer_prefill_compact_counters(&ctr, true);
    CHECK(ctr.requests == 1u && ctr.encoded == 0u && ctr.fallbacks == 0u && ctr.failures == 1u);
}

static void assert_ids_0ulp(const char *label, const myds_metal_buffer_t *ref_buf,
                            const myds_metal_buffer_t *got_buf, uint32_t rows) {
    const int32_t *ref = (const int32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)ref_buf);
    const int32_t *got = (const int32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)got_buf);
    const size_t bytes = (size_t)rows * 512u * sizeof(int32_t);
    if (memcmp(ref, got, bytes) != 0) {
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t k = 0; k < 512u; ++k) {
                const int32_t a = ref[(size_t)r * 512u + k];
                const int32_t b = got[(size_t)r * 512u + k];
                if (a != b) {
                    fprintf(stderr, "0-ULP MISMATCH [%s] at row=%u rank=%u: ref=%d got=%d\n",
                            label, r, k, a, b);
                    exit(1);
                }
            }
        }
    }
}

/* Direct <8-row (glen=1, 7) hardware partial-tile 0-ULP check against 8-row register_only. */
static void verify_sub8_direct_0ulp(myds_v41_gpu_scratch_t *s,
                                    const myds_metal_buffer_t *q_buf,
                                    const myds_metal_buffer_t *w_buf,
                                    const myds_metal_buffer_t *cache,
                                    myds_metal_buffer_t *mask_buf,
                                    myds_metal_buffer_t *sel_ref,
                                    myds_metal_buffer_t *sel_got,
                                    uint32_t nc, uint32_t glen,
                                    uint32_t mask_units, uint32_t *rng) {
    CHECK(glen >= 1u && glen < 8u && nc >= 16385u);
    const uint32_t pos0 = nc - glen;
    const uint32_t ref_nc = pos0 + 8u; /* 8-row reference so register_only runs MMA */
    build_synthetic_mask(mask_buf, 8u, pos0, ref_nc, mask_units, MASK_GPU_L20, rng);

    myds_metal_buffer_t *scores_buf = myds_v41_gpu_indexer_scores(s);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(
        s, scores_buf, q_buf, w_buf, cache, ref_nc, 32u, 8u, pos0, 1u, true, true));
    CHECK(myds_v41_gpu_filter_rows_bits(scores_buf, mask_buf, ref_nc, 8u, pos0, 1u, mask_units));
    CHECK(myds_v41_gpu_topk_causal(sel_ref, scores_buf, ref_nc, 512u, 8u, pos0, 1u, s));
    CHECK(myds_metal_commit_batch());

    memset(myds_metal_buffer_data(sel_got), 0x7f, (size_t)glen * 512u * sizeof(int32_t));
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_indexer_prefill_compact_bf16(
        s, sel_got, q_buf, w_buf, cache, mask_buf, nc, glen, pos0, 1u, mask_units) == 1);
    CHECK(myds_metal_commit_batch());

    char label[128];
    snprintf(label, sizeof(label), "direct_sub8 nc=%u glen=%u", nc, glen);
    assert_ids_0ulp(label, sel_ref, sel_got, glen);
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    bool run_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) run_bench = true;
    }

    unsetenv("MYDS_INDEXER_PREFILL_COMPACT");
    unsetenv("MYDS_INDEXER_REGISTER_ONLY");
    unsetenv("MYDS_INDEXER_RO_SMEM_Q");
    unsetenv("MYDS_INDEXER_KEY_WINDOW");
    unsetenv("MYDS_INDEXER_SGBAR");
    unsetenv("MYDS_TOPK_THRESHOLD");
    unsetenv("MYDS_BLOCK_BUDGET");

    CHECK(myds_metal_init(NULL));
    verify_pso_occupancy();

    const uint32_t max_nc = 262144u + 16u;
    const uint32_t max_rows = run_bench ? 2048u : 512u;
    const uint32_t max_blocks = (max_nc + 7u) / 8u;
    const uint32_t mask_units = myds_v41_gpu_maskbits_stride(max_blocks);

    setenv("MYDS_CHUNK_KEY_MAX", "262200", 1);
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(max_nc);
    unsetenv("MYDS_CHUNK_KEY_MAX");
    CHECK(s != NULL);
    CHECK(myds_v41_gpu_indexer_prefill_compact_max_glen(s) == 512u);

    myds_metal_buffer_t *cache = myds_metal_buffer_alloc((size_t)max_nc * 128u * sizeof(uint16_t));
    myds_metal_buffer_t *q20   = myds_metal_buffer_alloc((size_t)max_rows * 32u * 128u * sizeof(float));
    myds_metal_buffer_t *w20   = myds_metal_buffer_alloc((size_t)max_rows * 32u * sizeof(float));
    myds_metal_buffer_t *q24   = myds_metal_buffer_alloc((size_t)max_rows * 32u * 128u * sizeof(float));
    myds_metal_buffer_t *w24   = myds_metal_buffer_alloc((size_t)max_rows * 32u * sizeof(float));
    myds_metal_buffer_t *mask  = myds_metal_buffer_alloc((size_t)max_rows * mask_units * sizeof(uint32_t));
    myds_metal_buffer_t *sel0  = myds_metal_buffer_alloc((size_t)max_rows * 512u * sizeof(int32_t));
    myds_metal_buffer_t *sel1  = myds_metal_buffer_alloc((size_t)max_rows * 512u * sizeof(int32_t));
    myds_metal_buffer_t *qv    = myds_metal_buffer_create_view();
    myds_metal_buffer_t *wv    = myds_metal_buffer_create_view();
    myds_metal_buffer_t *sv    = myds_metal_buffer_create_view();
    myds_metal_buffer_t *mv    = myds_metal_buffer_create_view();
    CHECK(cache && q20 && w20 && q24 && w24 && mask && sel0 && sel1 && qv && wv && sv && mv);

    uint32_t rng = 0x20261114u;
    fill_bf16_cache(cache, max_nc, &rng, false);
    fill_q_weights(q20, w20, max_rows, &rng, false);
    fill_q_weights(q24, w24, max_rows, &rng, false);

    verify_knob_and_contracts(s, q24, w24, cache, mask, sel0, mask_units);

    printf("[3/5] Running 0-ULP parity matrix across nc={16385..262144} x glen={1,7,8,9,64,392,512}...\n");
    const uint32_t ncs[] = {16385u, 32768u, 65536u, 131072u, 262144u};
    const uint32_t glens[] = {1u, 7u, 8u, 9u, 64u, 392u, 512u};

    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ++ni) {
        const uint32_t nc_target = ncs[ni];
        for (size_t gi = 0; gi < sizeof(glens) / sizeof(glens[0]); ++gi) {
            const uint32_t glen = glens[gi];
            if (glen < 8u) {
                /* 1) Direct hardware partial-tile (<8 rows) 0-ULP check */
                verify_sub8_direct_0ulp(s, q24, w24, cache, mask, sel0, sel1,
                                        nc_target, glen, mask_units, &rng);
                /* 2) Scoped publisher COMPACT=0 vs COMPACT=1 parity */
                const uint32_t pos0 = nc_target - glen;
                memset(myds_metal_buffer_data(sel0), 0xaa, (size_t)glen * 512u * sizeof(int32_t));
                memset(myds_metal_buffer_data(sel1), 0x55, (size_t)glen * 512u * sizeof(int32_t));
                run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv,
                                   glen, pos0, mask_units, false);
                run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv,
                                   glen, pos0, mask_units, true);
                char label[128];
                snprintf(label, sizeof(label), "scoped_sub8 nc=%u glen=%u", nc_target, glen);
                assert_ids_0ulp(label, sel0, sel1, glen);
                continue;
            }

            /* For nc=16385 and glen>=8, test BOTH:
             * (a) straddling 16K (pos0 = 16385 - glen < 16384), AND
             * (b) active compact above 16K with odd partial-block tail (pos0 = 16385, nc = 16385 + glen). */
            const uint32_t pos0 = (nc_target >= 16384u + glen) ? (nc_target - glen) : 16385u;
            const uint32_t nc_eff = pos0 + glen;

            /* Use real GPU L20 YoCo masks for representative shapes, synthetic Bernoulli for fast sweep */
            if ((nc_target == 32768u && glen == 64u) ||
                (nc_target == 131072u && glen == 392u) ||
                (nc_target == 262144u && glen == 64u)) {
                generate_gpu_l20_mask(s, q20, w20, cache, mask, qv, wv, mv, glen, pos0, mask_units);
            } else {
                build_synthetic_mask(mask, glen, pos0, nc_eff, mask_units, MASK_GPU_L20, &rng);
            }

            memset(myds_metal_buffer_data(sel0), 0xaa, (size_t)glen * 512u * sizeof(int32_t));
            memset(myds_metal_buffer_data(sel1), 0x55, (size_t)glen * 512u * sizeof(int32_t));
            myds_v41_gpu_indexer_prefill_compact_counters(NULL, true);

            run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv,
                               glen, pos0, mask_units, false);
            myds_v41_gpu_indexer_prefill_compact_counters_t c_off;
            myds_v41_gpu_indexer_prefill_compact_counters(&c_off, true);
            CHECK(c_off.requests == 1u && c_off.encoded == 0u && c_off.fallbacks == 0u && c_off.failures == 0u);

            run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv,
                               glen, pos0, mask_units, true);
            myds_v41_gpu_indexer_prefill_compact_counters_t c_on;
            myds_v41_gpu_indexer_prefill_compact_counters(&c_on, true);
            CHECK(c_on.encoded == 1u && c_on.fallbacks == 0u && c_on.failures == 0u);
            CHECK(c_on.rows_scored == glen);

            char label[128];
            snprintf(label, sizeof(label), "matrix nc=%u pos0=%u glen=%u", nc_eff, pos0, glen);
            assert_ids_0ulp(label, sel0, sel1, glen);

            if (nc_target == 16385u) {
                /* Also test exact nc=16385 straddling 16K boundary + direct kernel partial <2048 blocks */
                const uint32_t straddle_pos0 = (glen <= 16385u) ? (16385u - glen) : 0u;
                build_synthetic_mask(mask, glen, straddle_pos0, 16385u, mask_units, MASK_GPU_L20, &rng);
                run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv,
                                   glen, straddle_pos0, mask_units, false);
                run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv,
                                   glen, straddle_pos0, mask_units, true);
                snprintf(label, sizeof(label), "straddle_16k nc=16385 glen=%u", glen);
                assert_ids_0ulp(label, sel0, sel1, glen);

                memset(myds_metal_buffer_data(sel1), 0x33, (size_t)glen * 512u * sizeof(int32_t));
                CHECK(myds_metal_begin_batch());
                CHECK(myds_v41_gpu_indexer_prefill_compact_bf16(
                    s, sel1, q24, w24, cache, mask, 16385u, glen, straddle_pos0, 1u, mask_units) == 1);
                CHECK(myds_metal_commit_batch());
                snprintf(label, sizeof(label), "direct_partial_lt2048 nc=16385 glen=%u", glen);
                assert_ids_0ulp(label, sel0, sel1, glen);
            }
        }
        printf("  nc=%6u: glen={1, 7, 8, 9, 64, 392, 512} -> PASS (0-ULP memcmp==0)\n", nc_target);
    }

    printf("[4/5] Running adversarial 0-ULP stress cases (ties, masks, SMEM_Q=0, concurrent)...\n");
    {
        /* A. Disjoint 8-row YoCo masks (maximum 16384 union blocks per 8-row tile) */
        const uint32_t nc = 131077u, glen = 64u, pos0 = nc - glen;
        build_synthetic_mask(mask, glen, pos0, nc, mask_units, MASK_DISJOINT_8ROW, &rng);
        run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, glen, pos0, mask_units, false);
        run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, glen, pos0, mask_units, true);
        assert_ids_0ulp("disjoint_8row_union16384", sel0, sel1, glen);

        /* B. Identical 8-row YoCo masks (minimum 2048 union blocks per 8-row tile) */
        build_synthetic_mask(mask, glen, pos0, nc, mask_units, MASK_IDENTICAL_8ROW, &rng);
        run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, glen, pos0, mask_units, false);
        run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, glen, pos0, mask_units, true);
        assert_ids_0ulp("identical_8row_union2048", sel0, sel1, glen);

        /* C. Sparse <512 finite unmasked rows (-INFINITY Partition-0 bitonic tail replay) */
        build_synthetic_mask(mask, 16u, 32768u, 32768u + 16u, mask_units, MASK_SPARSE_LT512, &rng);
        run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, 16u, 32768u, mask_units, false);
        run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, 16u, 32768u, mask_units, true);
        assert_ids_0ulp("sparse_lt512_inf_tail", sel0, sel1, 16u);

        /* D. MYDS_INDEXER_RO_SMEM_Q=0 (0B smem variant) */
        setenv("MYDS_INDEXER_RO_SMEM_Q", "0", 1);
        build_synthetic_mask(mask, 64u, 65539u - 64u, 65539u, mask_units, MASK_GPU_L20, &rng);
        run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, 64u, 65539u - 64u, mask_units, false);
        run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, 64u, 65539u - 64u, mask_units, true);
        assert_ids_0ulp("ro_smem_q_0", sel0, sel1, 64u);
        unsetenv("MYDS_INDEXER_RO_SMEM_Q");

        /* E. Concurrent encoder modes: (concurrent=1, auto_barrier=1) AND (concurrent=1, auto_barrier=0) */
        const int prev_conc = myds_metal_get_concurrent();
        const int prev_ab   = myds_metal_get_auto_barrier();
        for (int ab = 1; ab >= 0; --ab) {
            myds_metal_set_concurrent(1);
            myds_metal_set_auto_barrier(ab);
            run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, 128u, 65536u, mask_units, false);
            run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, 128u, 65536u, mask_units, true);
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "concurrent_ab%d", ab);
            assert_ids_0ulp(lbl, sel0, sel1, 128u);
        }
        myds_metal_set_concurrent(prev_conc);
        myds_metal_set_auto_barrier(prev_ab);

        /* F. Small scratch batch_floats=262144 (max_glen=16 multi-subbatch slicing over glen=64) */
        setenv("MYDS_INDEXER_BATCH_FLOATS", "262144", 1);
        myds_v41_gpu_scratch_t *small_s = myds_v41_gpu_scratch_create(65536u);
        unsetenv("MYDS_INDEXER_BATCH_FLOATS");
        CHECK(small_s != NULL && myds_v41_gpu_indexer_prefill_compact_max_glen(small_s) == 16u);
        myds_v41_gpu_indexer_prefill_compact_counters(NULL, true);
        run_consumer_chunk(small_s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, 64u, 32768u - 64u, mask_units, false);
        run_consumer_chunk(small_s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, 64u, 32768u - 64u, mask_units, true);
        myds_v41_gpu_indexer_prefill_compact_counters_t sc_ctr;
        myds_v41_gpu_indexer_prefill_compact_counters(&sc_ctr, true);
        CHECK(sc_ctr.encoded == 4u && sc_ctr.rows_scored == 64u && sc_ctr.fallbacks == 0u);
        assert_ids_0ulp("small_batch_floats_262144_in_scope", sel0, sel1, 64u);
        /* Above 262144/8 = 32768, dense COMPACT=0 drops <8 rows to chunk_exact, so scoped compact cleanly bypasses for 0-ULP */
        run_consumer_chunk(small_s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, 64u, 32768u, mask_units, false);
        run_consumer_chunk(small_s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, 64u, 32768u, mask_units, true);
        myds_v41_gpu_indexer_prefill_compact_counters(&sc_ctr, true);
        CHECK(sc_ctr.encoded == 0u && sc_ctr.fallbacks == 0u && sc_ctr.failures == 0u);
        assert_ids_0ulp("small_batch_floats_262144_bypass_gt32k", sel0, sel1, 64u);
        myds_v41_gpu_scratch_free(small_s);

        /* G. Heavy duplicate-row ties across & within 1024-partitions + boundary ties */
        stamp_heavy_tie_cache(cache, 65600u);
        build_synthetic_mask(mask, 64u, 65536u - 64u, 65536u, mask_units, MASK_GPU_L20, &rng);
        run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, 64u, 65536u - 64u, mask_units, false);
        run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, 64u, 65536u - 64u, mask_units, true);
        assert_ids_0ulp("heavy_duplicate_ties_64k", sel0, sel1, 64u);

        /* H. All-0.0f ReLU ties (all negative Q * all positive K -> 16384-way 0.0f ties!) */
        fill_bf16_cache(cache, 65600u, &rng, true);
        fill_q_weights(q24, w24, 32u, &rng, true);
        build_synthetic_mask(mask, 32u, 65536u - 32u, 65536u, mask_units, MASK_GPU_L20, &rng);
        run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, 32u, 65536u - 32u, mask_units, false);
        run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, 32u, 65536u - 32u, mask_units, true);
        assert_ids_0ulp("all_zero_relu_ties_64k", sel0, sel1, 32u);
        printf("  disjoint/identical masks, <512 -INF tail, SMEM_Q=0, concurrent, multi-way & 0.0f ties -> PASS\n");
    }

    if (run_bench) {
        printf("[5/5] Running 7-run median GPU microbenchmark (COMPACT=0 vs COMPACT=1, GPU L20 YoCo masks)...\n");
        fill_bf16_cache(cache, max_nc, &rng, false);
        fill_q_weights(q20, w20, max_rows, &rng, false);
        fill_q_weights(q24, w24, max_rows, &rng, false);

        struct {
            const char *name;
            uint32_t pos0;
            uint32_t c_len;
        } benches[] = {
            {"128K_chunk2048", 129024u, 2048u},
            {"128K_tail392",   131072u,  392u},
            {"256K_tail304",   261840u,  304u},
        };

        for (size_t bi = 0; bi < sizeof(benches) / sizeof(benches[0]); ++bi) {
            const uint32_t pos0 = benches[bi].pos0;
            const uint32_t c_len = benches[bi].c_len;
            generate_gpu_l20_mask(s, q20, w20, cache, mask, qv, wv, mv, c_len, pos0, mask_units);

            /* Warmup + 0-ULP check on benchmarked workload */
            run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, c_len, pos0, mask_units, false);
            run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, c_len, pos0, mask_units, true);
            assert_ids_0ulp(benches[bi].name, sel0, sel1, c_len);

            double t_off[7], t_on[7];
            for (int r = 0; r < 7; ++r) {
                const double s0 = monotonic_ms();
                run_consumer_chunk(s, q24, w24, cache, mask, sel0, qv, wv, sv, mv, c_len, pos0, mask_units, false);
                t_off[r] = monotonic_ms() - s0;

                const double s1 = monotonic_ms();
                run_consumer_chunk(s, q24, w24, cache, mask, sel1, qv, wv, sv, mv, c_len, pos0, mask_units, true);
                t_on[r] = monotonic_ms() - s1;
            }
            qsort(t_off, 7, sizeof(double), cmp_double);
            qsort(t_on,  7, sizeof(double), cmp_double);
            const double m_off = t_off[3];
            const double m_on  = t_on[3];
            printf("  %-16s (pos0=%6u, c_len=%4u): COMPACT=0 %7.2f ms (x4 layers=%7.2f ms) | COMPACT=1 %7.2f ms (x4=%7.2f ms) | %.2fx speedup\n",
                   benches[bi].name, pos0, c_len,
                   m_off, m_off * 4.0, m_on, m_on * 4.0, m_off / m_on);
        }
    } else {
        printf("[5/5] Microbenchmark skipped (pass --bench to run 128K/256K 7-run medians).\n");
    }

    myds_metal_buffer_free(mv);
    myds_metal_buffer_free(sv);
    myds_metal_buffer_free(wv);
    myds_metal_buffer_free(qv);
    myds_metal_buffer_free(sel1);
    myds_metal_buffer_free(sel0);
    myds_metal_buffer_free(mask);
    myds_metal_buffer_free(w24);
    myds_metal_buffer_free(q24);
    myds_metal_buffer_free(w20);
    myds_metal_buffer_free(q20);
    myds_metal_buffer_free(cache);
    myds_v41_gpu_scratch_free(s);
    myds_metal_cleanup();
    printf("PASS: test_indexer_prefill_compact completed with 100%% 0-ULP parity.\n");
    return 0;
}
