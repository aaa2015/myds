/* Model-free M4 GPU acceptance & hardware-timestamp microbenchmark for [D-L1]:
 * Long-context single-step Decode (il > 20, count > 16384) compact <=16K scoring +
 * compact <=16K Top-512 selection (0-ULP vs dense rows4 + filter + full topk, and
 * 0-ULP compatible with INDEXER_DECODE_OVERLAP defer_q_rope + bind_topk_qb). */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include "myds_metal.h"
#include "myds_gguf.h"
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { \
    if (!(c)) { \
        fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #c); \
        return 0; \
    } \
} while (0)

static uint32_t rng_u32(uint32_t i, uint32_t seed) {
    uint32_t z = i * 747796405u + seed * 2891336453u + 1013904223u;
    z = ((z >> ((z >> 28u) + 4u)) ^ z) * 277803737u;
    return (z >> 22u) ^ z;
}

typedef struct {
    myds_metal_buffer_t *owner;
    myds_metal_buffer_t *view;
    size_t bytes;
} guarded_buf_t;

static int guarded_alloc(guarded_buf_t *g, size_t bytes) {
    memset(g, 0, sizeof(*g));
    g->bytes = bytes;
    g->owner = myds_metal_buffer_alloc(bytes + 64u);
    CHECK(g->owner != NULL);
    memset(myds_metal_buffer_data(g->owner), 0xa5, bytes + 64u);
    g->view = myds_metal_buffer_sub_view(g->owner, 32u, bytes);
    CHECK(g->view != NULL);
    return 1;
}

static int guarded_check(const guarded_buf_t *g) {
    const uint8_t *p = (const uint8_t *)myds_metal_buffer_data(g->owner);
    for (size_t i = 0; i < 32u; ++i) {
        CHECK(p[i] == 0xa5u);
        CHECK(p[32u + g->bytes + i] == 0xa5u);
    }
    return 1;
}

static void guarded_poison(const guarded_buf_t *g) {
    memset(myds_metal_buffer_data(g->view), 0x5a, g->bytes);
}

static void guarded_free(guarded_buf_t *g) {
    myds_metal_buffer_free(g->view);
    myds_metal_buffer_free(g->owner);
    memset(g, 0, sizeof(*g));
}

static int fill_quantized_q_and_w(myds_metal_buffer_t *q, myds_metal_buffer_t *w, uint32_t seed) {
    float *qp = (float *)myds_metal_buffer_data(q);
    float *wp = (float *)myds_metal_buffer_data(w);
    for (uint32_t i = 0; i < 32u * 128u; ++i) {
        qp[i] = ((int)(rng_u32(i, seed) % 1023u) - 511) / 512.0f;
    }
    for (uint32_t h = 0; h < 32u; ++h) {
        wp[h] = 0.0625f + (float)(rng_u32(h, seed + 19u) % 127u) / 128.0f;
    }
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_quant(q, 128u, 32u, 2u));
    CHECK(myds_metal_commit_batch());
    return 1;
}

static int populate_bf16_cache_random(myds_metal_buffer_t *packed, myds_metal_buffer_t *tmp_f32,
                                      uint32_t total_rows, uint32_t chunk_rows, uint32_t seed) {
    for (uint32_t r0 = 0; r0 < total_rows; r0 += chunk_rows) {
        const uint32_t nr = (total_rows - r0 < chunk_rows) ? (total_rows - r0) : chunk_rows;
        float *fp = (float *)myds_metal_buffer_data(tmp_f32);
        for (size_t i = 0; i < (size_t)nr * 128u; ++i) {
            fp[i] = ((int)(rng_u32((uint32_t)(r0 * 128u + i), seed) % 1023u) - 511) / 256.0f;
        }
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_quant(tmp_f32, 128u, nr, 2u));
        CHECK(myds_v41_gpu_pack_bf16(packed, (size_t)r0 * 128u, tmp_f32, 0u, nr * 128u));
        CHECK(myds_metal_commit_batch());
    }
    return 1;
}

/* Duplicate a small palette of quantized 128-dim BF16 rows across blocks so both
 * intra-1024-partition and cross-1024-partition equal-score ties occur heavily,
 * including multi-way ties spanning rank 511 / rank 512. */
static void stamp_heavy_tie_cache(myds_metal_buffer_t *packed, uint32_t rows, uint32_t palette_len) {
    uint16_t *pk = (uint16_t *)myds_metal_buffer_data(packed);
    for (uint32_t r = palette_len; r < rows; ++r) {
        const uint32_t src = (r * 7u + (r / 8u) * 3u + (r / 1024u) * 5u) % palette_len;
        memcpy(pk + (size_t)r * 128u, pk + (size_t)src * 128u, 128u * sizeof(uint16_t));
    }
}

static int run_dense_l20_l24(myds_v41_gpu_scratch_t *s,
                             const myds_metal_buffer_t *q20, const myds_metal_buffer_t *w20,
                             const myds_metal_buffer_t *q24, const myds_metal_buffer_t *w24,
                             const myds_metal_buffer_t *cache,
                             myds_metal_buffer_t *scores, myds_metal_buffer_t *block_scores,
                             myds_metal_buffer_t *block_selected, myds_metal_buffer_t *block_mask,
                             myds_metal_buffer_t *selected_out,
                             uint32_t count, uint32_t first_row, bool run_l20) {
    const uint32_t blocks = (count + 7u) / 8u;
    const uint32_t top_blocks = blocks < 2048u ? blocks : 2048u;
    const uint32_t top_k = count < 512u ? count : 512u;
    const uint32_t pos = count - 1u;

    CHECK(myds_metal_begin_batch());
    if (run_l20) {
        CHECK(myds_v41_gpu_indexscores_head_slots(scores, q20, w20, cache, NULL, count, 32u, true, first_row));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(scores);
        CHECK(myds_v41_gpu_candidates(block_scores, scores, count, pos, 1u));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(block_scores);
        CHECK(myds_v41_gpu_topk(block_selected, block_scores, blocks, top_blocks, s));
    }
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(block_selected);
    CHECK(myds_v41_gpu_topkmask(block_mask, block_selected, blocks, top_blocks));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(block_mask);

    CHECK(myds_v41_gpu_indexscores_head_slots(scores, q24, w24, cache, block_mask, count, 32u, true, first_row));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(scores);
    CHECK(myds_v41_gpu_filter(scores, block_mask, count, pos, 1u));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(scores);
    CHECK(myds_v41_gpu_topk(selected_out, scores, count, top_k, s));
    CHECK(myds_metal_commit_batch());
    return 1;
}

static int run_compact_l20_l24(myds_v41_gpu_scratch_t *s,
                               const myds_metal_buffer_t *q20, const myds_metal_buffer_t *w20,
                               const myds_metal_buffer_t *q24, const myds_metal_buffer_t *w24,
                               const myds_metal_buffer_t *cache,
                               myds_metal_buffer_t *scores, myds_metal_buffer_t *block_scores,
                               myds_metal_buffer_t *block_selected,
                               myds_metal_buffer_t *selected_out,
                               uint32_t count, uint32_t first_row, bool run_l20) {
    const uint32_t blocks = (count + 7u) / 8u;
    const uint32_t top_blocks = blocks < 2048u ? blocks : 2048u;
    const uint32_t pos = count - 1u;

    CHECK(myds_metal_begin_batch());
    if (run_l20) {
        CHECK(myds_v41_gpu_indexscores_head_slots(scores, q20, w20, cache, NULL, count, 32u, true, first_row));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(scores);
        CHECK(myds_v41_gpu_candidates(block_scores, scores, count, pos, 1u));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(block_scores);
        CHECK(myds_v41_gpu_topk(block_selected, block_scores, blocks, top_blocks, s));
    }
    CHECK(myds_v41_gpu_indexer_prepare_compact_blocks(s, block_selected, blocks, top_blocks, count));
    CHECK(myds_v41_gpu_indexer_decode_compact_ready(s, count));
    if (!myds_metal_get_auto_barrier()) {
        const myds_metal_buffer_t *deps[] = {
            q24, w24, cache,
            myds_v41_gpu_scratch_compact_blocks(s),
            myds_v41_gpu_scratch_compact_scores(s),
            myds_v41_gpu_scratch_compact_rows(s),
        };
        myds_metal_barrier_buffers(deps, 6u);
    }
    CHECK(myds_v41_gpu_indexscores_compact_decode_bf16(s, q24, w24, cache, NULL, count, first_row));
    if (!myds_metal_get_auto_barrier()) {
        const myds_metal_buffer_t *c_deps[] = {
            myds_v41_gpu_scratch_compact_scores(s),
            myds_v41_gpu_scratch_compact_rows(s),
            myds_v41_gpu_scratch_compact_stage1(s),
        };
        myds_metal_barrier_buffers(c_deps, 3u);
    }
    CHECK(myds_v41_gpu_topk_compact_decode(selected_out, count, s));
    CHECK(myds_metal_commit_batch());
    return 1;
}

static int verify_ids_exact(const guarded_buf_t *ref, const guarded_buf_t *cand, const char *label) {
    CHECK(guarded_check(ref));
    CHECK(guarded_check(cand));
    const int32_t *rp = (const int32_t *)myds_metal_buffer_data(ref->view);
    const int32_t *cp = (const int32_t *)myds_metal_buffer_data(cand->view);
    if (memcmp(rp, cp, 512u * sizeof(int32_t)) != 0) {
        for (uint32_t i = 0; i < 512u; ++i) {
            if (rp[i] != cp[i]) {
                fprintf(stderr, "[MISMATCH] %s rank=%u ref=%d cand=%d\n", label, i, rp[i], cp[i]);
                break;
            }
        }
        return 0;
    }
    return 1;
}

static int test_env_and_fail_closed_guards(void) {
    const char *bad[] = {"", "2", "01", "true", "-1", " 1"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        CHECK(!setenv("MYDS_INDEXER_DECODE_COMPACT", bad[i], 1));
        CHECK(myds_v41_gpu_indexer_decode_compact_enabled() == -1);
    }
    unsetenv("MYDS_INDEXER_DECODE_COMPACT");
    CHECK(myds_v41_gpu_indexer_decode_compact_enabled() == 1);

    myds_v41_gpu_scratch_t *small_s = myds_v41_gpu_scratch_create(16384u);
    CHECK(small_s != NULL);
    CHECK(!myds_v41_gpu_indexer_decode_compact_eligible(small_s, 16384u));
    myds_v41_gpu_scratch_free(small_s);

    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(65536u);
    CHECK(s != NULL);
    CHECK(!myds_v41_gpu_indexer_decode_compact_eligible(s, 16384u));
    CHECK(myds_v41_gpu_indexer_decode_compact_eligible(s, 16385u));
    CHECK(myds_v41_gpu_indexer_decode_compact_eligible(s, 32768u));
    CHECK(!myds_v41_gpu_indexer_decode_compact_ready(s, 32768u));

    CHECK(!setenv("MYDS_INDEXER_DECODE_COMPACT", "0", 1));
    CHECK(myds_v41_gpu_indexer_decode_compact_enabled() == 0);
    CHECK(!myds_v41_gpu_indexer_decode_compact_eligible(s, 32768u));
    unsetenv("MYDS_INDEXER_DECODE_COMPACT");

    myds_metal_buffer_t *q = myds_metal_buffer_alloc(32u * 128u * 4u);
    myds_metal_buffer_t *w = myds_metal_buffer_alloc(32u * 4u);
    myds_metal_buffer_t *cache = myds_metal_buffer_alloc(32768u * 128u * 2u);
    myds_metal_buffer_t *ids = myds_metal_buffer_alloc(512u * 4u);
    CHECK(q && w && cache && ids);

    /* Calling compact scorer or compact topk before prepare_compact_blocks must fail closed. */
    myds_v41_gpu_indexer_decode_compact_counters(NULL, true);
    CHECK(!myds_v41_gpu_indexscores_compact_decode_bf16(s, q, w, cache, NULL, 32768u, 0u));
    CHECK(!myds_v41_gpu_topk_compact_decode(ids, 32768u, s));
    myds_v41_gpu_indexer_decode_compact_counters_t ctr;
    myds_v41_gpu_indexer_decode_compact_counters(&ctr, false);
    CHECK(ctr.score_requests == 1u && ctr.score_encoded == 0u &&
          ctr.topk_requests == 1u && ctr.topk_encoded == 0u && ctr.failures == 2u);

    myds_metal_buffer_free(q);
    myds_metal_buffer_free(w);
    myds_metal_buffer_free(cache);
    myds_metal_buffer_free(ids);
    myds_v41_gpu_scratch_free(s);
    printf("  [PASS] [D-L1] strict env parsing, count>16384 eligibility, and fail-closed readiness guards\n");
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static int test_gpu_parity_ties_overlap_and_bench(bool full_bench) {
    const uint32_t max_rows = 524288u;
    const uint32_t max_blocks = (max_rows + 7u) / 8u;
    const uint32_t chunk_rows = 65536u;

    myds_v41_gpu_scratch_t *s_ref = myds_v41_gpu_scratch_create(max_rows);
    myds_v41_gpu_scratch_t *s_cand = myds_v41_gpu_scratch_create(max_rows);
    CHECK(s_ref && s_cand);

    myds_metal_buffer_t *cache = myds_metal_buffer_alloc((size_t)max_rows * 128u * 2u);
    myds_metal_buffer_t *tmp_f32 = myds_metal_buffer_alloc((size_t)chunk_rows * 128u * 4u);
    myds_metal_buffer_t *q20 = myds_metal_buffer_alloc(32u * 128u * 4u);
    myds_metal_buffer_t *w20 = myds_metal_buffer_alloc(32u * 4u);
    myds_metal_buffer_t *q24 = myds_metal_buffer_alloc(32u * 128u * 4u);
    myds_metal_buffer_t *w24 = myds_metal_buffer_alloc(32u * 4u);
    myds_metal_buffer_t *scores = myds_metal_buffer_alloc((size_t)max_rows * 4u);
    myds_metal_buffer_t *block_scores = myds_metal_buffer_alloc((size_t)max_blocks * 4u);
    myds_metal_buffer_t *block_selected = myds_metal_buffer_alloc(2048u * 4u);
    myds_metal_buffer_t *block_mask = myds_metal_buffer_alloc((size_t)max_blocks * 4u);
    guarded_buf_t sel_ref, sel_cand;
    CHECK(cache && tmp_f32 && q20 && w20 && q24 && w24 && scores && block_scores && block_selected && block_mask);
    CHECK(guarded_alloc(&sel_ref, 512u * sizeof(int32_t)));
    CHECK(guarded_alloc(&sel_cand, 512u * sizeof(int32_t)));

    CHECK(fill_quantized_q_and_w(q20, w20, 101u));
    CHECK(fill_quantized_q_and_w(q24, w24, 211u));
    CHECK(populate_bf16_cache_random(cache, tmp_f32, max_rows, chunk_rows, 307u));

    /* 1. Random BF16 parity across 16385, 32768, 131072, 131077 (partial tail block), 524288
     *    in both serial (auto_barrier=1, concurrent=0) and concurrent (auto_barrier=0, concurrent=1) modes. */
    const uint32_t counts[] = {16385u, 32768u, 131072u, 131077u, 524288u};
    for (int conc = 0; conc < 2; ++conc) {
        myds_metal_set_concurrent(conc);
        myds_metal_set_auto_barrier(conc ? 0 : 1);
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ++ci) {
            const uint32_t count = counts[ci];
            guarded_poison(&sel_ref);
            guarded_poison(&sel_cand);
            CHECK(run_dense_l20_l24(s_ref, q20, w20, q24, w24, cache,
                                    scores, block_scores, block_selected, block_mask,
                                    sel_ref.view, count, 0u, true));
            CHECK(run_compact_l20_l24(s_cand, q20, w20, q24, w24, cache,
                                      scores, block_scores, block_selected,
                                      sel_cand.view, count, 0u, true));
            char label[96];
            snprintf(label, sizeof(label), "random conc=%d count=%u", conc, count);
            CHECK(verify_ids_exact(&sel_ref, &sel_cand, label));
        }
    }
    myds_metal_set_concurrent(0);
    myds_metal_set_auto_barrier(1);
    printf("  [PASS] [D-L1] 0-ULP selected[512] parity across count=16385,32768,131072,131077,524288 (serial & concurrent)\n");

    /* 2. Distance-window first_row parity at unaligned count=131077, including <512 finite rows (300, 37, 1, 0). */
    const uint32_t win_firsts[] = {98300u, 130777u, 131040u, 131076u, 131077u};
    for (size_t wi = 0; wi < sizeof(win_firsts) / sizeof(win_firsts[0]); ++wi) {
        guarded_poison(&sel_ref);
        guarded_poison(&sel_cand);
        CHECK(run_dense_l20_l24(s_ref, q20, w20, q24, w24, cache,
                                scores, block_scores, block_selected, block_mask,
                                sel_ref.view, 131077u, win_firsts[wi], true));
        CHECK(run_compact_l20_l24(s_cand, q20, w20, q24, w24, cache,
                                  scores, block_scores, block_selected,
                                  sel_cand.view, 131077u, win_firsts[wi], true));
        char label[96];
        snprintf(label, sizeof(label), "window first_row=%u", win_firsts[wi]);
        CHECK(verify_ids_exact(&sel_ref, &sel_cand, label));
    }
    printf("  [PASS] [D-L1] 0-ULP window first_row parity (including <512 finite rows: 300, 37, 1, 0)\n");

    /* 3. Adversarial block_selected: negative (-1, -999), OOB (>= blocks, INT32_MAX),
     *    duplicate block IDs, reverse order, and partial final block at count=131077. */
    {
        const uint32_t count = 131077u;
        const uint32_t blocks = (count + 7u) / 8u; /* 16385, block 16384 has 5 valid rows */
        int32_t *bs = (int32_t *)myds_metal_buffer_data(block_selected);
        bs[0] = -1;
        bs[1] = -999;
        bs[2] = (int32_t)blocks;
        bs[3] = INT32_MAX;
        bs[4] = (int32_t)(blocks - 1u); /* partial tail block! */
        bs[5] = (int32_t)(blocks - 1u); /* duplicate of partial tail block! */
        bs[6] = 0;
        bs[7] = 0;                      /* duplicate of block 0! */
        for (uint32_t i = 8u; i < 2048u; ++i) {
            if (i % 19u == 0u) {
                bs[i] = -1;
            } else if (i % 23u == 0u) {
                bs[i] = (int32_t)(blocks + i);
            } else {
                /* Reverse-strided valid blocks with deliberate duplicates */
                bs[i] = (int32_t)((blocks - 1u - ((i * 7u) % blocks)));
            }
        }
        guarded_poison(&sel_ref);
        guarded_poison(&sel_cand);
        CHECK(run_dense_l20_l24(s_ref, q20, w20, q24, w24, cache,
                                scores, block_scores, block_selected, block_mask,
                                sel_ref.view, count, 0u, false));
        CHECK(run_compact_l20_l24(s_cand, q20, w20, q24, w24, cache,
                                  scores, block_scores, block_selected,
                                  sel_cand.view, count, 0u, false));
        CHECK(verify_ids_exact(&sel_ref, &sel_cand, "adversarial block_selected (-1/OOB/dups/partial-tail)"));
    }
    printf("  [PASS] [D-L1] 0-ULP adversarial block_selected (-1, OOB, duplicates, reverse order, partial tail block)\n");

    /* 4. INDEXER_DECODE_OVERLAP (defer_q_rope + bind_topk_qb) M4 GPU parity at count=131072:
     *    verifies both selected[512] AND 64-head Q_B + RoPE output q[32768] are 0-ULP identical! */
    {
        const uint32_t count = 131072u;
        const uint32_t pos = count - 1u;
        /* Q4_K weight for 64*512=32768 rows x 1280 cols (5 super-blocks of 256 per row, 144 B each) */
        const size_t qb_bytes = (size_t)64u * 512u * 5u * 144u;
        myds_metal_buffer_t *qb_weight = myds_metal_buffer_alloc(qb_bytes);
        myds_metal_buffer_t *qr_norm = myds_metal_buffer_alloc(1280u * 4u);
        guarded_buf_t q_ref, q_cand;
        CHECK(qb_weight && qr_norm);
        CHECK(guarded_alloc(&q_ref, 64u * 512u * 4u));
        CHECK(guarded_alloc(&q_cand, 64u * 512u * 4u));

        /* Populate deterministic finite Q4_K blocks (fp16 d=0x3000 (0.00048828125), dmin=0, pseudo-random qs/scales) */
        uint8_t *qbp = (uint8_t *)myds_metal_buffer_data(qb_weight);
        for (size_t sb = 0; sb < (size_t)64u * 512u * 5u; ++sb) {
            uint8_t *blk = qbp + sb * 144u;
            const uint16_t d_f16 = (uint16_t)(0x2c00u + (sb % 5u) * 0x0100u);
            const uint16_t dmin_f16 = 0x2400u;
            memcpy(blk + 0, &d_f16, 2u);
            memcpy(blk + 2, &dmin_f16, 2u);
            for (uint32_t j = 4u; j < 144u; ++j) {
                blk[j] = (uint8_t)(rng_u32((uint32_t)(sb * 31u + j), 991u) & 0x3fu);
            }
        }
        float *qrn = (float *)myds_metal_buffer_data(qr_norm);
        for (uint32_t i = 0; i < 1280u; ++i) {
            qrn[i] = ((int)(rng_u32(i, 557u) % 511u) - 255) / 256.0f;
        }
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_v41_bf16(qr_norm, 1280u));
        CHECK(myds_metal_commit_batch());

        /* Prepare L20 blocks on both scratches */
        CHECK(run_dense_l20_l24(s_ref, q20, w20, q24, w24, cache,
                                scores, block_scores, block_selected, block_mask,
                                sel_ref.view, count, 0u, true));
        CHECK(run_compact_l20_l24(s_cand, q20, w20, q24, w24, cache,
                                  scores, block_scores, block_selected,
                                  sel_cand.view, count, 0u, true));

        guarded_poison(&sel_ref);
        guarded_poison(&sel_cand);
        guarded_poison(&q_ref);
        guarded_poison(&q_cand);

        /* Reference L24: non-overlapped full dense_q4_v41 + rope + dense score/filter/topk */
        myds_metal_set_concurrent(0);
        myds_metal_set_auto_barrier(1);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_dense_q4_v41(q_ref.view, qb_weight, qr_norm, 64u * 512u, 1280u, true));
        CHECK(myds_v41_gpu_rope(q_ref.view, 512u, 64u, pos, true, false));
        CHECK(myds_v41_gpu_indexscores_head_slots(scores, q24, w24, cache, block_mask, count, 32u, true, 0u));
        CHECK(myds_v41_gpu_filter(scores, block_mask, count, pos, 1u));
        CHECK(myds_v41_gpu_topk(sel_ref.view, scores, count, 512u, s_ref));
        CHECK(myds_metal_commit_batch());

        /* Candidate L24: concurrent INDEXER_DECODE_OVERLAP schedule (qb_slice + defer_q_rope + bind_topk_qb + compact_topk) */
        myds_metal_set_concurrent(1);
        myds_metal_set_auto_barrier(0);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_dense_q4_v41_qb_slice(q_cand.view, qb_weight, qr_norm, 0u, 4u * 512u, true));
        const myds_metal_buffer_t *s_deps[] = {
            q24, w24, cache,
            myds_v41_gpu_scratch_compact_blocks(s_cand),
            myds_v41_gpu_scratch_compact_scores(s_cand),
            myds_v41_gpu_scratch_compact_rows(s_cand),
        };
        myds_metal_barrier_buffers(s_deps, 6u);
        CHECK(myds_v41_gpu_indexscores_compact_decode_bf16(s_cand, q24, w24, cache, NULL, count, 0u));
        CHECK(myds_metal_dense_q4_v41_qb_slice(q_cand.view, qb_weight, qr_norm, 4u * 512u, 28u * 512u, false));
        const myds_metal_buffer_t *c6_deps[] = {
            myds_v41_gpu_scratch_compact_scores(s_cand),
            myds_v41_gpu_scratch_compact_rows(s_cand),
            myds_v41_gpu_scratch_compact_stage1(s_cand),
            q_cand.view,
        };
        myds_metal_barrier_buffers(c6_deps, 4u);
        myds_v41_gpu_attn_kv_reuse_defer_q_rope(s_cand, q_cand.view, true);
        myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s_cand, qb_weight, qr_norm, pos, 32u);
        CHECK(myds_v41_gpu_topk_compact_decode(sel_cand.view, count, s_cand));
        CHECK(myds_v41_gpu_attn_kv_reuse_qb_done(s_cand) == 64u);
        const myds_metal_buffer_t *end_deps[] = {sel_cand.view, q_cand.view};
        myds_metal_barrier_buffers(end_deps, 2u);
        CHECK(myds_metal_commit_batch());

        myds_metal_set_concurrent(0);
        myds_metal_set_auto_barrier(1);
        myds_v41_gpu_attn_kv_reuse_reset(s_cand);

        CHECK(verify_ids_exact(&sel_ref, &sel_cand, "INDEXER_DECODE_OVERLAP selected[512]"));
        CHECK(guarded_check(&q_ref) && guarded_check(&q_cand));
        CHECK(memcmp(myds_metal_buffer_data(q_ref.view),
                     myds_metal_buffer_data(q_cand.view),
                     64u * 512u * 4u) == 0);

        guarded_free(&q_ref);
        guarded_free(&q_cand);
        myds_metal_buffer_free(qb_weight);
        myds_metal_buffer_free(qr_norm);
    }
    printf("  [PASS] [D-L1] 0-ULP INDEXER_DECODE_OVERLAP parity (selected[512] & 64-head Q_B+RoPE memcmp == 0)\n");

    /* 5. M4 GPU Hardware Timestamp Microbenchmark (GPUEndTime - GPUStartTime) on random cache
     *    at count = 32768, 131072, 524288 (single-layer L24: dense rows4+filter+topk vs compact [D-L1]). */
    {
        const uint32_t bench_counts[] = {32768u, 131072u, 524288u};
        const unsigned warmup = full_bench ? 12u : 4u;
        const unsigned rounds = full_bench ? 25u : 9u;
        const unsigned reps_per_cb = 4u;
        double dense_times[32], compact_times[32];

        for (size_t bi = 0; bi < sizeof(bench_counts) / sizeof(bench_counts[0]); ++bi) {
            const uint32_t count = bench_counts[bi];
            const uint32_t pos = count - 1u;

            myds_metal_set_concurrent(0);
            myds_metal_set_auto_barrier(1);
            CHECK(run_dense_l20_l24(s_ref, q20, w20, q24, w24, cache,
                                    scores, block_scores, block_selected, block_mask,
                                    sel_ref.view, count, 0u, true));
            CHECK(run_compact_l20_l24(s_cand, q20, w20, q24, w24, cache,
                                      scores, block_scores, block_selected,
                                      sel_cand.view, count, 0u, true));
            CHECK(verify_ids_exact(&sel_ref, &sel_cand, "pre-bench parity"));

            myds_metal_set_concurrent(1);
            myds_metal_set_auto_barrier(0);
            for (unsigned w = 0; w < warmup; ++w) {
                CHECK(myds_metal_begin_batch());
                for (unsigned r = 0; r < reps_per_cb; ++r) {
                    CHECK(myds_v41_gpu_indexscores_head_slots(scores, q24, w24, cache, block_mask, count, 32u, true, 0u));
                    myds_metal_barrier_buffer(scores);
                    CHECK(myds_v41_gpu_filter(scores, block_mask, count, pos, 1u));
                    myds_metal_barrier_buffer(scores);
                    CHECK(myds_v41_gpu_topk(sel_ref.view, scores, count, 512u, s_ref));
                }
                CHECK(myds_metal_commit_batch());

                CHECK(myds_metal_begin_batch());
                for (unsigned r = 0; r < reps_per_cb; ++r) {
                    const myds_metal_buffer_t *s_deps[] = {
                        myds_v41_gpu_scratch_compact_scores(s_cand),
                        myds_v41_gpu_scratch_compact_rows(s_cand),
                        sel_cand.view,
                    };
                    myds_metal_barrier_buffers(s_deps, 3u);
                    CHECK(myds_v41_gpu_indexscores_compact_decode_bf16(s_cand, q24, w24, cache, NULL, count, 0u));
                    const myds_metal_buffer_t *c_deps[] = {
                        myds_v41_gpu_scratch_compact_scores(s_cand),
                        myds_v41_gpu_scratch_compact_rows(s_cand),
                        myds_v41_gpu_scratch_compact_stage1(s_cand),
                    };
                    myds_metal_barrier_buffers(c_deps, 3u);
                    CHECK(myds_v41_gpu_topk_compact_decode(sel_cand.view, count, s_cand));
                }
                CHECK(myds_metal_commit_batch());
            }

            for (unsigned it = 0; it < rounds; ++it) {
                for (unsigned step = 0; step < 2u; ++step) {
                    const unsigned arm = (it % 2u == 0u) ? step : (1u - step);
                    double t0 = 0.0, t1 = 0.0;
                    CHECK(myds_metal_begin_batch());
                    if (arm == 0u) {
                        for (unsigned r = 0; r < reps_per_cb; ++r) {
                            CHECK(myds_v41_gpu_indexscores_head_slots(scores, q24, w24, cache, block_mask, count, 32u, true, 0u));
                            myds_metal_barrier_buffer(scores);
                            CHECK(myds_v41_gpu_filter(scores, block_mask, count, pos, 1u));
                            myds_metal_barrier_buffer(scores);
                            CHECK(myds_v41_gpu_topk(sel_ref.view, scores, count, 512u, s_ref));
                        }
                    } else {
                        for (unsigned r = 0; r < reps_per_cb; ++r) {
                            const myds_metal_buffer_t *s_deps[] = {
                                myds_v41_gpu_scratch_compact_scores(s_cand),
                                myds_v41_gpu_scratch_compact_rows(s_cand),
                                sel_cand.view,
                            };
                            myds_metal_barrier_buffers(s_deps, 3u);
                            CHECK(myds_v41_gpu_indexscores_compact_decode_bf16(s_cand, q24, w24, cache, NULL, count, 0u));
                            const myds_metal_buffer_t *c_deps[] = {
                                myds_v41_gpu_scratch_compact_scores(s_cand),
                                myds_v41_gpu_scratch_compact_rows(s_cand),
                                myds_v41_gpu_scratch_compact_stage1(s_cand),
                            };
                            myds_metal_barrier_buffers(c_deps, 3u);
                            CHECK(myds_v41_gpu_topk_compact_decode(sel_cand.view, count, s_cand));
                        }
                    }
                    CHECK(myds_metal_commit_batch());
                    CHECK(myds_metal_last_completed_gpu_span(&t0, &t1));
                    const double us = (t1 - t0) * 1e6 / (double)reps_per_cb;
                    CHECK(isfinite(us) && us > 0.0);
                    if (arm == 0u) dense_times[it] = us;
                    else compact_times[it] = us;
                }
            }
            myds_metal_set_concurrent(0);
            myds_metal_set_auto_barrier(1);
            CHECK(verify_ids_exact(&sel_ref, &sel_cand, "post-bench parity"));

            qsort(dense_times, rounds, sizeof(double), cmp_double);
            qsort(compact_times, rounds, sizeof(double), cmp_double);
            const double med_dense = dense_times[rounds / 2u];
            const double med_compact = compact_times[rounds / 2u];
            printf("  [GPU-SPAN] [D-L1] L24 count=%-6u | dense(rows4+filter+topk)=%7.2f us | compact(16K+topk)=%7.2f us | speedup=%.2fx (-%.1f%%)\n",
                   count, med_dense, med_compact, med_dense / med_compact,
                   100.0 * (med_dense - med_compact) / med_dense);
            if (count >= 131072u) {
                CHECK(med_compact < med_dense);
            }
        }
    }

    /* 6. Heavy duplicate equal-score ties across count = 32768, 131072, 524288:
     *    - 19-entry quantized palette (thousands of identical scores intra- & cross-1024-partitions,
     *      guaranteed ties spanning rank 511 == rank 512)
     *    - All-zero cache (all 16,384 candidate rows tied at exact 0.0f) */
    stamp_heavy_tie_cache(cache, max_rows, 19u);
    const uint32_t tie_counts[] = {32768u, 131072u, 524288u};
    for (size_t ti = 0; ti < sizeof(tie_counts) / sizeof(tie_counts[0]); ++ti) {
        const uint32_t count = tie_counts[ti];
        guarded_poison(&sel_ref);
        guarded_poison(&sel_cand);
        CHECK(run_dense_l20_l24(s_ref, q20, w20, q24, w24, cache,
                                scores, block_scores, block_selected, block_mask,
                                sel_ref.view, count, 0u, true));
        CHECK(run_compact_l20_l24(s_cand, q20, w20, q24, w24, cache,
                                  scores, block_scores, block_selected,
                                  sel_cand.view, count, 0u, true));
        /* Confirm rank 511 and rank 512 in scores actually tied! */
        const int32_t *rp = (const int32_t *)myds_metal_buffer_data(sel_ref.view);
        const float *sp = (const float *)myds_metal_buffer_data(scores);
        CHECK(sp[rp[0]] == sp[rp[10]] && sp[rp[500]] == sp[rp[511]]);
        char label[96];
        snprintf(label, sizeof(label), "19-palette heavy ties count=%u", count);
        CHECK(verify_ids_exact(&sel_ref, &sel_cand, label));
    }

    memset(myds_metal_buffer_data(cache), 0, (size_t)max_rows * 128u * 2u);
    for (size_t ti = 0; ti < sizeof(tie_counts) / sizeof(tie_counts[0]); ++ti) {
        const uint32_t count = tie_counts[ti];
        guarded_poison(&sel_ref);
        guarded_poison(&sel_cand);
        CHECK(run_dense_l20_l24(s_ref, q20, w20, q24, w24, cache,
                                scores, block_scores, block_selected, block_mask,
                                sel_ref.view, count, 0u, true));
        CHECK(run_compact_l20_l24(s_cand, q20, w20, q24, w24, cache,
                                  scores, block_scores, block_selected,
                                  sel_cand.view, count, 0u, true));
        char label[96];
        snprintf(label, sizeof(label), "all-zero 16384-way ties count=%u", count);
        CHECK(verify_ids_exact(&sel_ref, &sel_cand, label));
    }
    printf("  [PASS] [D-L1] 0-ULP heavy equal-score tie-breaking (intra/cross-1024 & 16384-way 0.0f ties at 32K/131K/524K)\n");

    guarded_free(&sel_ref);
    guarded_free(&sel_cand);
    myds_metal_buffer_free(cache);
    myds_metal_buffer_free(tmp_f32);
    myds_metal_buffer_free(q20);
    myds_metal_buffer_free(w20);
    myds_metal_buffer_free(q24);
    myds_metal_buffer_free(w24);
    myds_metal_buffer_free(scores);
    myds_metal_buffer_free(block_scores);
    myds_metal_buffer_free(block_selected);
    myds_metal_buffer_free(block_mask);
    myds_v41_gpu_scratch_free(s_ref);
    myds_v41_gpu_scratch_free(s_cand);
    return 1;
}

int main(int argc, char **argv) {
    const bool full_bench = (argc > 1 && strcmp(argv[1], "--bench") == 0);
    unsetenv("MYDS_INDEXER_DECODE_COMPACT");
    unsetenv("MYDS_INDEXER_COMPACT_SCORER");
    unsetenv("MYDS_INDEXER_TOPK_THRESHOLD");
    unsetenv("MYDS_BLOCK_BUDGET");
    setenv("MYDS_INDEXER_HEAD_SLOTS", "1", 1);
    setenv("MYDS_DENSE_Q4_V41", "1", 1);
    if (!myds_metal_init("metal")) {
        fprintf(stderr, "[FAIL] myds_metal_init failed\n");
        return 1;
    }
    const int ok = test_env_and_fail_closed_guards() &&
                   test_gpu_parity_ties_overlap_and_bench(full_bench);
    myds_metal_cleanup();
    if (ok) {
        printf("ALL [D-L1] COMPACT DECODE INDEXER GPU TESTS PASSED (0-ULP)\n");
    }
    return ok ? 0 : 1;
}
