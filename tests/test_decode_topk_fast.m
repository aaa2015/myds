/* tests/test_decode_topk_fast.m
 *
 * Strict 0-ULP verification + M4 benchmark for [D-T2] (MYDS_DECODE_TOPK_FAST):
 *   (a) overlap_topk_qb stays enabled under MYDS_INDEXER_TOPK_PREFIX_MERGE=1
 *       (both NAN_FUSE=0 and NAN_FUSE=1) with concurrent auto_barrier=0 safety;
 *   (b) overlap_topk_qb safely overlaps Q_B + Q RoPE on L20;
 *   (c) prefix_merge cap raised from top<=512 to top<=2048 bit-identically;
 *   (d) L20 self-compact Top-512 over compact_sorted_blocks[0..2047] matches
 *       baseline full-length argsort bit-for-bit (memcmp == 0) across
 *       nc in {16384, 32768, 65536, 131072, 262144}, ties, boundary ties,
 *       all-zeros, -0.0f/+0.0f, <512 valid, +INF, visible & hidden NaNs, and
 *       concurrent auto_barrier=0.
 *
 * Run:
 *   ./bin/test_decode_topk_fast           (full 0-ULP correctness suite)
 *   ./bin/test_decode_topk_fast --bench   (correctness + M4 GPU latency bench)
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

#include "myds_metal.h"
#include "myds_v41_gpu.h"

static uint32_t g_rng = 0x2468ace1u;
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

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static int test_knob_parsing(void) {
    printf("[1/5] Verifying MYDS_DECODE_TOPK_FAST strict fail-closed parsing...\n");
    unsetenv("MYDS_DECODE_TOPK_FAST");
    if (myds_v41_gpu_decode_topk_fast_enabled() != 0) {
        fprintf(stderr, "FAIL: absent MYDS_DECODE_TOPK_FAST must default to 0\n");
        return 1;
    }
    setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
    if (myds_v41_gpu_decode_topk_fast_enabled() != 0) {
        fprintf(stderr, "FAIL: MYDS_DECODE_TOPK_FAST=0 must return 0\n");
        return 1;
    }
    setenv("MYDS_DECODE_TOPK_FAST", "1", 1);
    if (myds_v41_gpu_decode_topk_fast_enabled() != 1) {
        fprintf(stderr, "FAIL: MYDS_DECODE_TOPK_FAST=1 must return 1\n");
        return 1;
    }
    const char *bad[] = {"", "2", "-1", "true", "yes", "01", "1 "};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        setenv("MYDS_DECODE_TOPK_FAST", bad[i], 1);
        if (myds_v41_gpu_decode_topk_fast_enabled() != -1) {
            fprintf(stderr, "FAIL: invalid MYDS_DECODE_TOPK_FAST='%s' must return -1\n", bad[i]);
            return 1;
        }
    }
    setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
    printf("  PASS: strict knob parsing (NULL/0->0, 1->1, invalid->-1).\n");
    return 0;
}

enum score_pattern {
    PAT_RANDOM = 0,
    PAT_HEAVY_TIES,
    PAT_BOUNDARY_TIES,
    PAT_ALL_ZEROS,
    PAT_SIGNED_ZEROS,
    PAT_LT512_VALID_0,
    PAT_LT512_VALID_17,
    PAT_LT512_VALID_511,
    PAT_ALL_POS_INF,
    PAT_VISIBLE_NAN,
    PAT_HIDDEN_NAN_UNSELECTED_BLOCK,
    PAT_HIDDEN_NAN_LAST_BLOCK,
    PAT_COUNT
};

static const char *pattern_name(enum score_pattern p) {
    switch (p) {
        case PAT_RANDOM: return "random";
        case PAT_HEAVY_TIES: return "heavy_ties";
        case PAT_BOUNDARY_TIES: return "boundary_ties";
        case PAT_ALL_ZEROS: return "all_zeros";
        case PAT_SIGNED_ZEROS: return "signed_zeros(-0/+0)";
        case PAT_LT512_VALID_0: return "lt512_valid_0";
        case PAT_LT512_VALID_17: return "lt512_valid_17";
        case PAT_LT512_VALID_511: return "lt512_valid_511";
        case PAT_ALL_POS_INF: return "all_pos_inf";
        case PAT_VISIBLE_NAN: return "visible_nan";
        case PAT_HIDDEN_NAN_UNSELECTED_BLOCK: return "hidden_nan_unselected";
        case PAT_HIDDEN_NAN_LAST_BLOCK: return "hidden_nan_last_block";
        default: return "?";
    }
}

static void fill_scores(float *dst, uint32_t nc, enum score_pattern pat, uint32_t seed) {
    g_rng = seed ? seed : 0x12345678u;
    switch (pat) {
        case PAT_RANDOM:
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = ((float)(int32_t)(xorshift32() & 0xffffu) - 32768.0f) * 0.001f;
            }
            break;
        case PAT_HEAVY_TIES:
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = (float)(int32_t)(xorshift32() % 9u) * 0.25f;
            }
            break;
        case PAT_BOUNDARY_TIES: {
            /* 300 strictly above tau=5.0f, thousands tied at tau=5.0f across many partitions */
            for (uint32_t i = 0; i < nc; i++) dst[i] = -1.0f;
            for (uint32_t k = 0; k < 300u; k++) {
                uint32_t idx = (k * 97u + 13u) % nc;
                dst[idx] = 10.0f + (float)(k % 7u);
            }
            for (uint32_t i = 1024u; i < nc; i += 19u) {
                if (dst[i] < 0.0f) dst[i] = 5.0f;
            }
            /* Also inject intra-partition ties inside partition 3 and 7 */
            for (uint32_t j = 0; j < 20u; j++) {
                dst[3072u + j * 3u] = 12.0f;
                dst[7168u + j * 5u] = 5.0f;
            }
            break;
        }
        case PAT_ALL_ZEROS:
            for (uint32_t i = 0; i < nc; i++) dst[i] = 0.0f;
            break;
        case PAT_SIGNED_ZEROS:
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = (xorshift32() & 1u) ? bits_to_f32(0x80000000u) : 0.0f;
            }
            /* Sprinkle a few strictly positive scores so both interior & boundary -0/+0 ties happen */
            for (uint32_t k = 0; k < 120u; k++) {
                dst[(k * 131u + 5u) % nc] = 1.0f + (float)(k % 5u);
            }
            break;
        case PAT_LT512_VALID_0:
            for (uint32_t i = 0; i < nc; i++) dst[i] = -INFINITY;
            break;
        case PAT_LT512_VALID_17:
            for (uint32_t i = 0; i < nc; i++) dst[i] = -INFINITY;
            for (uint32_t k = 0; k < 17u; k++) {
                dst[(k * 733u + 11u) % nc] = (float)(k % 4u) * 0.5f;
            }
            break;
        case PAT_LT512_VALID_511:
            for (uint32_t i = 0; i < nc; i++) dst[i] = -INFINITY;
            for (uint32_t k = 0; k < 511u; k++) {
                dst[(k * 29u + 7u) % nc] = 2.0f + (float)(k % 11u) * 0.25f;
            }
            break;
        case PAT_ALL_POS_INF:
            for (uint32_t i = 0; i < nc; i++) dst[i] = INFINITY;
            break;
        case PAT_VISIBLE_NAN:
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = ((float)(int32_t)(xorshift32() & 0xfffu) - 2048.0f) * 0.01f;
            }
            dst[17] = bits_to_f32(0x7fc00001u); /* quiet NaN in partition 0 */
            dst[2049] = bits_to_f32(0x7f800001u); /* signaling NaN bit pattern */
            break;
        case PAT_HIDDEN_NAN_UNSELECTED_BLOCK:
            /* Construct high scores in first 2200 blocks so block 2500 is NOT selected,
             * then hide a NaN inside block 2500 alongside a finite number so max(x, NaN) == x! */
            for (uint32_t i = 0; i < nc; i++) dst[i] = -100.0f + (float)(i % 13u);
            for (uint32_t b = 0; b < 2200u && b * 8u < nc; b++) {
                dst[b * 8u] = 100.0f + (float)(b % 50u);
            }
            if (nc > 2500u * 8u + 5u) {
                dst[2500u * 8u + 0u] = -50.0f;
                dst[2500u * 8u + 3u] = bits_to_f32(0x7fc01234u); /* masked by max(-50, NaN) in unselected block */
            } else {
                dst[nc - 10u] = bits_to_f32(0x7fc01234u);
            }
            break;
        case PAT_HIDDEN_NAN_LAST_BLOCK:
            /* Hide a NaN in the very last block where candidate_blocks overwrites best = +INFINITY */
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = ((float)(int32_t)(xorshift32() & 0xfffu) - 2048.0f) * 0.01f;
            }
            dst[nc - 1u] = bits_to_f32(0x7fc05678u);
            break;
        default:
            break;
    }
}

/* Fill a valid deterministic Q4_K weight matrix for Q_B [32768 x 1280] (5 blocks/row, 144 B/block). */
static void fill_qb_weights_and_input(myds_metal_buffer_t *qb_w, myds_metal_buffer_t *qb_in) {
    uint8_t *w = (uint8_t *)myds_metal_buffer_data(qb_w);
    const size_t total_blocks = 32768u * 5u;
    /* fp16 0.015625 (2^-6) = 0x2400, fp16 0.0078125 (2^-7) = 0x2000 */
    for (size_t b = 0; b < total_blocks; b++) {
        uint8_t *blk = w + b * 144u;
        uint16_t d = (uint16_t)(0x2400u + (b & 0x1fu));
        uint16_t dmin = (uint16_t)(0x2000u + ((b >> 5u) & 0x1fu));
        memcpy(blk + 0, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (size_t i = 4; i < 16; i++) blk[i] = (uint8_t)((b * 13u + i * 7u) & 0x3fu);
        for (size_t i = 16; i < 144; i++) blk[i] = (uint8_t)((b * 29u + i * 11u) & 0xffu);
    }
    float *in = (float *)myds_metal_buffer_data(qb_in);
    for (uint32_t i = 0; i < 1280u; i++) {
        in[i] = ((float)((int)(i % 31u) - 15)) * 0.05f;
    }
}

/* Reference sequential Q_B (64 heads) + Q RoPE (64 heads) with auto_barrier=1 */
static int compute_reference_q(myds_v41_gpu_scratch_t *s, myds_metal_buffer_t *q_ref,
                               const myds_metal_buffer_t *qb_w, const myds_metal_buffer_t *qb_in,
                               uint32_t pos, bool compressed) {
    myds_metal_set_auto_barrier(1);
    myds_metal_set_concurrent(0);
    if (!myds_metal_begin_batch()) return 0;
    if (!myds_metal_dense_q4_v41_qb_slice(q_ref, qb_w, qb_in, 0u, 32u * 512u, false)) return 0;
    if (!myds_metal_dense_q4_v41_qb_slice(q_ref, qb_w, qb_in, 32u * 512u, 32u * 512u, false)) return 0;
    if (!myds_v41_gpu_rope_heads_slice(s, q_ref, 0u, 64u, pos, compressed)) return 0;
    if (!myds_metal_commit_batch()) return 0;
    return 1;
}

static int compare_i32_exact(const int32_t *ref, const int32_t *got, uint32_t n,
                             const float *scores, const char *label) {
    if (memcmp(ref, got, (size_t)n * sizeof(int32_t)) == 0) return 0;
    for (uint32_t i = 0; i < n; i++) {
        if (ref[i] != got[i]) {
            float sr = (scores && ref[i] >= 0) ? scores[ref[i]] : NAN;
            float sg = (scores && got[i] >= 0) ? scores[got[i]] : NAN;
            fprintf(stderr, "FAIL [%s] mismatch at slot %u: ref=%d (score=%g 0x%08x) got=%d (score=%g 0x%08x)\n",
                    label, i, ref[i], sr, f32_to_bits(sr), got[i], sg, f32_to_bits(sg));
            break;
        }
    }
    return 1;
}

/* [2/5] Test Sub-requirement (a): overlap_topk_qb with PREFIX_MERGE=1 (NAN_FUSE=0 & 1) */
static int test_overlap_prefix_root(myds_v41_gpu_scratch_t *s,
                                    myds_metal_buffer_t *scores_buf,
                                    myds_metal_buffer_t *ids_ref,
                                    myds_metal_buffer_t *ids_got,
                                    myds_metal_buffer_t *q_ref,
                                    myds_metal_buffer_t *q_got,
                                    const myds_metal_buffer_t *qb_w,
                                    const myds_metal_buffer_t *qb_in) {
    printf("[2/5] Testing [D-T2](a) overlap_topk_qb + PREFIX_MERGE=1 (concurrent auto_barrier=0)...\n");
    const uint32_t ncs[] = {16384u, 32768u, 65536u, 131072u, 262144u};
    const uint32_t pos = 131071u;
    if (!compute_reference_q(s, q_ref, qb_w, qb_in, pos, true)) {
        fprintf(stderr, "FAIL: compute_reference_q failed\n");
        return 1;
    }

    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    for (int nan_fuse = 0; nan_fuse <= 1; nan_fuse++) {
        setenv("MYDS_INDEXER_TOPK_NAN_FUSE", nan_fuse ? "1" : "0", 1);
        for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
            const uint32_t nc = ncs[ni];
            float *sc = (float *)myds_metal_buffer_data(scores_buf);
            enum score_pattern pats[] = {PAT_RANDOM, PAT_BOUNDARY_TIES, PAT_SIGNED_ZEROS, PAT_VISIBLE_NAN};
            for (size_t pi = 0; pi < sizeof(pats) / sizeof(pats[0]); pi++) {
                fill_scores(sc, nc, pats[pi], 0x1000u + (uint32_t)(ni * 37u + pi * 13u + (size_t)nan_fuse));

                /* Baseline: MYDS_DECODE_TOPK_FAST=0, auto_barrier=1 */
                setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
                myds_metal_set_auto_barrier(1);
                myds_metal_set_concurrent(0);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_topk(ids_ref, scores_buf, nc, 512u, s) ||
                    !myds_metal_commit_batch()) {
                    fprintf(stderr, "FAIL: baseline topk failed nc=%u\n", nc);
                    return 1;
                }

                /* Candidate: MYDS_DECODE_TOPK_FAST=1, concurrent=1, auto_barrier=0, overlap_topk_qb */
                setenv("MYDS_DECODE_TOPK_FAST", "1", 1);
                myds_v41_gpu_decode_topk_fast_counters_t c0, c1;
                myds_v41_gpu_decode_topk_fast_counters(&c0, true);
                memset(myds_metal_buffer_data(q_got), 0x5a, 64u * 512u * sizeof(float));
                memset(myds_metal_buffer_data(ids_got), 0xa5, 512u * sizeof(int32_t));

                myds_metal_set_auto_barrier(0);
                myds_metal_set_concurrent(1);
                if (!myds_metal_begin_batch()) return 1;
                /* Pre-compute heads 0..32 (like c5+c6) then fence {scores, q} */
                if (!myds_metal_dense_q4_v41_qb_slice(q_got, qb_w, qb_in, 0u, 32u * 512u, false)) return 1;
                const myds_metal_buffer_t *c6_deps[] = {scores_buf, q_got};
                myds_metal_barrier_buffers(c6_deps, 2u);
                myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_got, true);
                myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                if (!myds_v41_gpu_topk(ids_got, scores_buf, nc, 512u, s)) return 1;
                const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                if (qb_done < 64u) {
                    if (!myds_metal_dense_q4_v41_qb_slice(q_got, qb_w, qb_in,
                            qb_done * 512u, (64u - qb_done) * 512u, false)) return 1;
                }
                const myds_metal_buffer_t *fin_deps[] = {ids_got, q_got};
                myds_metal_barrier_buffers(fin_deps, 2u);
                if (rope_done < 64u) {
                    if (!myds_v41_gpu_rope_heads_slice(s, q_got, rope_done, 64u - rope_done, pos, true)) return 1;
                    myds_metal_barrier_buffer(q_got);
                }
                myds_v41_gpu_attn_kv_reuse_reset(s);
                if (!myds_metal_commit_batch()) return 1;
                myds_metal_set_auto_barrier(1);
                myds_metal_set_concurrent(0);

                myds_v41_gpu_decode_topk_fast_counters(&c1, false);
                if (c1.overlap_prefix_encoded != 1u) {
                    fprintf(stderr, "FAIL: expected overlap_prefix_encoded==1 for nc=%u nan_fuse=%d, got %llu\n",
                            nc, nan_fuse, (unsigned long long)c1.overlap_prefix_encoded);
                    return 1;
                }
                char lbl[128];
                snprintf(lbl, sizeof(lbl), "overlap_prefix nc=%u nf=%d pat=%s", nc, nan_fuse, pattern_name(pats[pi]));
                if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                      (const int32_t *)myds_metal_buffer_data(ids_got),
                                      512u, sc, lbl) != 0) return 1;
                if (memcmp(myds_metal_buffer_data(q_ref), myds_metal_buffer_data(q_got),
                           64u * 512u * sizeof(float)) != 0) {
                    fprintf(stderr, "FAIL [%s]: Q_B + Q RoPE output mismatch under concurrent overlap!\n", lbl);
                    return 1;
                }
            }
        }
    }
    printf("  PASS: overlap_topk_qb + PREFIX_MERGE=1 (NAN_FUSE=0/1) bit-exact on IDs & Q across 16K..256K.\n");
    return 0;
}

/* [3/5] Test Sub-requirement (c): prefix_merge cap raised to top<=2048 */
static int test_prefix_top2048(myds_v41_gpu_scratch_t *s,
                               myds_metal_buffer_t *scores_buf,
                               myds_metal_buffer_t *ids_ref,
                               myds_metal_buffer_t *ids_got) {
    printf("[3/5] Testing [D-T2](c) prefix_merge top==2048 (blocks=16384 & 32768)...\n");
    const uint32_t block_counts[] = {16384u, 32768u};
    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    for (int nan_fuse = 0; nan_fuse <= 1; nan_fuse++) {
        setenv("MYDS_INDEXER_TOPK_NAN_FUSE", nan_fuse ? "1" : "0", 1);
        for (size_t bi = 0; bi < sizeof(block_counts) / sizeof(block_counts[0]); bi++) {
            const uint32_t nb = block_counts[bi];
            float *sc = (float *)myds_metal_buffer_data(scores_buf);
            enum score_pattern pats[] = {
                PAT_RANDOM, PAT_HEAVY_TIES, PAT_BOUNDARY_TIES, PAT_ALL_ZEROS,
                PAT_SIGNED_ZEROS, PAT_LT512_VALID_17, PAT_ALL_POS_INF, PAT_VISIBLE_NAN
            };
            for (size_t pi = 0; pi < sizeof(pats) / sizeof(pats[0]); pi++) {
                fill_scores(sc, nb, pats[pi], 0x3333u + (uint32_t)(bi * 53u + pi * 17u + (size_t)nan_fuse));

                /* Baseline: MYDS_DECODE_TOPK_FAST=0 (prefix disabled because top=2048 > 512) */
                setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
                myds_metal_set_auto_barrier(1);
                myds_metal_set_concurrent(0);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_topk(ids_ref, scores_buf, nb, 2048u, s) ||
                    !myds_metal_commit_batch()) return 1;

                /* Candidate: MYDS_DECODE_TOPK_FAST=1 (prefix enabled for top=2048) */
                setenv("MYDS_DECODE_TOPK_FAST", "1", 1);
                myds_v41_gpu_decode_topk_fast_counters_t c1;
                myds_v41_gpu_decode_topk_fast_counters(NULL, true);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_topk(ids_got, scores_buf, nb, 2048u, s) ||
                    !myds_metal_commit_batch()) return 1;
                myds_v41_gpu_decode_topk_fast_counters(&c1, false);
                if (c1.prefix_top2048_encoded != 1u) {
                    fprintf(stderr, "FAIL: expected prefix_top2048_encoded==1 for blocks=%u nf=%d\n", nb, nan_fuse);
                    return 1;
                }
                char lbl[128];
                snprintf(lbl, sizeof(lbl), "prefix_top2048 blocks=%u nf=%d pat=%s", nb, nan_fuse, pattern_name(pats[pi]));
                if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                      (const int32_t *)myds_metal_buffer_data(ids_got),
                                      2048u, sc, lbl) != 0) return 1;
            }
        }
    }
    printf("  PASS: prefix_merge top==2048 (NAN_FUSE=0/1) bit-exact across ties, zeros, -0/+0, <2048, +INF, NaN.\n");
    return 0;
}

/* [4/5] Test Sub-requirements (b) & (d): L20 self-compact Top-512 + L20 overlap_topk_qb */
static int test_l20_self_compact(myds_v41_gpu_scratch_t *s,
                                 myds_metal_buffer_t *scores_buf,
                                 myds_metal_buffer_t *block_scores,
                                 myds_metal_buffer_t *block_sel_ref,
                                 myds_metal_buffer_t *block_sel_got,
                                 myds_metal_buffer_t *ids_ref,
                                 myds_metal_buffer_t *ids_got,
                                 myds_metal_buffer_t *q_ref,
                                 myds_metal_buffer_t *q_got,
                                 const myds_metal_buffer_t *qb_w,
                                 const myds_metal_buffer_t *qb_in) {
    printf("[4/5] Testing [D-T2](b)+(d) L20 self-compact Top-512 + L20 overlap_topk_qb...\n");
    const uint32_t ncs[] = {16392u, 32768u, 65536u, 131072u, 262144u};
    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);

    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u; /* ratio = 1, visible == nc */
        if (!compute_reference_q(s, q_ref, qb_w, qb_in, pos, true)) return 1;

        for (int p = 0; p < PAT_COUNT; p++) {
            float *sc = (float *)myds_metal_buffer_data(scores_buf);
            fill_scores(sc, nc, (enum score_pattern)p, 0x7777u + (uint32_t)(ni * 97u + (size_t)p * 23u));

            /* Baseline: full candidates + block_selected Top-2048 + 2nd full nc argsort */
            setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
            myds_metal_set_auto_barrier(1);
            myds_metal_set_concurrent(0);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_candidates(block_scores, scores_buf, nc, pos, 1u) ||
                !myds_v41_gpu_topk(block_sel_ref, block_scores, blocks, 2048u, s) ||
                !myds_v41_gpu_topk(ids_ref, scores_buf, nc, 512u, s) ||
                !myds_metal_commit_batch()) {
                fprintf(stderr, "FAIL: baseline L20 failed nc=%u pat=%s\n", nc, pattern_name((enum score_pattern)p));
                return 1;
            }

            /* Candidate: MYDS_DECODE_TOPK_FAST=1, concurrent auto_barrier=0 + L20 overlap_topk_qb + self-compact */
            setenv("MYDS_DECODE_TOPK_FAST", "1", 1);
            if (!myds_v41_gpu_topk_l20_self_compact_eligible(s, nc)) {
                fprintf(stderr, "FAIL: l20_self_compact not eligible for nc=%u\n", nc);
                return 1;
            }
            myds_v41_gpu_decode_topk_fast_counters_t c1;
            myds_v41_gpu_decode_topk_fast_counters(NULL, true);
            memset(myds_metal_buffer_data(q_got), 0x3c, 64u * 512u * sizeof(float));
            memset(myds_metal_buffer_data(ids_got), 0xc3, 512u * sizeof(int32_t));
            memset(myds_metal_buffer_data(block_sel_got), 0xc3, 2048u * sizeof(int32_t));

            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            if (!myds_metal_begin_batch()) return 1;
            if (!myds_metal_dense_q4_v41_qb_slice(q_got, qb_w, qb_in, 0u, 32u * 512u, false)) return 1;
            const myds_metal_buffer_t *c6_deps[] = {scores_buf, q_got};
            myds_metal_barrier_buffers(c6_deps, 2u);

            myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_got, true);
            myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
            if (!myds_v41_gpu_l20_decode_candidates_nan(block_scores, scores_buf, nc, pos, 1u, s)) return 1;
            myds_metal_barrier_buffer(block_scores);
            if (!myds_v41_gpu_topk(block_sel_got, block_scores, blocks, 2048u, s)) return 1;
            const myds_metal_buffer_t *blk_q_deps[] = {block_sel_got, q_got};
            myds_metal_barrier_buffers(blk_q_deps, 2u);
            if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel_got, blocks, 2048u, nc)) return 1;

            myds_v41_gpu_attn_kv_reuse_rebind_topk_qb(s, qb_w, qb_in, pos);
            if (!myds_v41_gpu_topk_l20_self_compact_decode(ids_got, scores_buf, nc, s)) return 1;

            const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
            const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
            if (qb_done < 64u) {
                if (!myds_metal_dense_q4_v41_qb_slice(q_got, qb_w, qb_in,
                        qb_done * 512u, (64u - qb_done) * 512u, false)) return 1;
            }
            const myds_metal_buffer_t *fin_deps[] = {ids_got, q_got};
            myds_metal_barrier_buffers(fin_deps, 2u);
            if (rope_done < 64u) {
                if (!myds_v41_gpu_rope_heads_slice(s, q_got, rope_done, 64u - rope_done, pos, true)) return 1;
                myds_metal_barrier_buffer(q_got);
            }
            myds_v41_gpu_attn_kv_reuse_reset(s);
            if (!myds_metal_commit_batch()) return 1;
            myds_metal_set_auto_barrier(1);
            myds_metal_set_concurrent(0);

            myds_v41_gpu_decode_topk_fast_counters(&c1, false);
            if (c1.l20_self_compact_encoded != 1u || c1.l20_self_compact_failures != 0u) {
                fprintf(stderr, "FAIL: unexpected L20 counters nc=%u pat=%s enc=%llu fail=%llu\n",
                        nc, pattern_name((enum score_pattern)p),
                        (unsigned long long)c1.l20_self_compact_encoded,
                        (unsigned long long)c1.l20_self_compact_failures);
                return 1;
            }
            char lbl[128];
            snprintf(lbl, sizeof(lbl), "L20_blk nc=%u pat=%s", nc, pattern_name((enum score_pattern)p));
            if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(block_sel_ref),
                                  (const int32_t *)myds_metal_buffer_data(block_sel_got),
                                  2048u, NULL, lbl) != 0) return 1;
            snprintf(lbl, sizeof(lbl), "L20_sel nc=%u pat=%s", nc, pattern_name((enum score_pattern)p));
            if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                  (const int32_t *)myds_metal_buffer_data(ids_got),
                                  512u, sc, lbl) != 0) return 1;
            if (memcmp(myds_metal_buffer_data(q_ref), myds_metal_buffer_data(q_got),
                       64u * 512u * sizeof(float)) != 0) {
                fprintf(stderr, "FAIL [%s]: L20 overlapped Q_B + Q RoPE mismatch!\n", lbl);
                return 1;
            }
        }
    }
    printf("  PASS: L20 self-compact + L20 overlap_topk_qb bit-exact across ALL %d patterns x 5 sizes (16K..256K).\n",
           (int)PAT_COUNT);
    return 0;
}

/* [5/5] Optional --bench: measure L20 and Root (L2/8/14) latency on M4 */
static void fill_realistic_f32_scores(float *dst, uint32_t nc, uint32_t seed) {
    g_rng = seed ? seed : 0xabcdef01u;
    for (uint32_t i = 0; i < nc; i++) {
        dst[i] = ((float)(int32_t)(xorshift32() & 0xffffffu) - 8388608.0f) * 1.0e-5f;
    }
}

static void run_bench(myds_v41_gpu_scratch_t *s,
                      myds_metal_buffer_t *scores_buf,
                      myds_metal_buffer_t *block_scores,
                      myds_metal_buffer_t *block_sel,
                      myds_metal_buffer_t *ids_buf,
                      myds_metal_buffer_t *q_buf,
                      const myds_metal_buffer_t *qb_w,
                      const myds_metal_buffer_t *qb_in) {
    printf("\n[5/5] M4 GPU Benchmark (--bench): Baseline (MYDS_DECODE_TOPK_FAST=0) vs [D-T2] (=1)\n");
    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);
    const uint32_t ncs[] = {16384u, 32768u, 65536u, 131072u, 262144u};
    const int warmup = 5, iters = 40;

    printf("  -- Pure L20 Top-K (candidates + block_sel(2048) + prepare + Top-512, batched GPU CB) --\n");
    for (size_t ni = 1; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u;
        fill_realistic_f32_scores((float *)myds_metal_buffer_data(scores_buf), nc, 0xbbb0u + nc);

        double base_us = 0.0, fast_us = 0.0;
        for (int arm = 0; arm < 2; arm++) {
            setenv("MYDS_DECODE_TOPK_FAST", arm ? "1" : "0", 1);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            for (int w = 0; w < warmup; w++) {
                if (!myds_metal_begin_batch()) return;
                if (arm == 0) {
                    if (!myds_v41_gpu_candidates(block_scores, scores_buf, nc, pos, 1u)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                } else {
                    if (!myds_v41_gpu_l20_decode_candidates_nan(block_scores, scores_buf, nc, pos, 1u, s)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    if (!myds_v41_gpu_topk_l20_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                }
                if (!myds_metal_commit_batch()) return;
            }
            const double t0 = now_us();
            if (!myds_metal_begin_batch()) return;
            for (int it = 0; it < iters; it++) {
                if (arm == 0) {
                    if (!myds_v41_gpu_candidates(block_scores, scores_buf, nc, pos, 1u)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                } else {
                    if (!myds_v41_gpu_l20_decode_candidates_nan(block_scores, scores_buf, nc, pos, 1u, s)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    if (!myds_v41_gpu_topk_l20_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                }
            }
            if (!myds_metal_commit_batch()) return;
            double avg = (now_us() - t0) / (double)iters;
            if (arm == 0) base_us = avg; else fast_us = avg;
        }
        printf("    nc=%6u: base=%7.1f us  fast=%7.1f us  delta=%+6.1f us (%.2fx)\n",
               nc, base_us, fast_us, fast_us - base_us, base_us / fast_us);
    }

    printf("  -- Root Layer (L2/8/14): Top-512 + trailing 32 Q_B heads + 64 Q RoPE (batched GPU CB) --\n");
    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t pos = nc - 1u;
        fill_realistic_f32_scores((float *)myds_metal_buffer_data(scores_buf), nc, 0x9999u + nc);

        double base_us = 0.0, fast_us = 0.0;
        for (int arm = 0; arm < 2; arm++) {
            setenv("MYDS_DECODE_TOPK_FAST", arm ? "1" : "0", 1);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            for (int w = 0; w < warmup; w++) {
                if (!myds_metal_begin_batch()) return;
                myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                if (qb_done < 64u) {
                    if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in,
                            qb_done * 512u, (64u - qb_done) * 512u, false)) return;
                }
                const myds_metal_buffer_t *fin_deps[] = {ids_buf, q_buf};
                myds_metal_barrier_buffers(fin_deps, 2u);
                if (rope_done < 64u) {
                    if (!myds_v41_gpu_rope_heads_slice(s, q_buf, rope_done, 64u - rope_done, pos, true)) return;
                    myds_metal_barrier_buffer(q_buf);
                }
                myds_v41_gpu_attn_kv_reuse_reset(s);
                if (!myds_metal_commit_batch()) return;
            }
            const double t0 = now_us();
            if (!myds_metal_begin_batch()) return;
            for (int it = 0; it < iters; it++) {
                myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                if (qb_done < 64u) {
                    if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in,
                            qb_done * 512u, (64u - qb_done) * 512u, false)) return;
                }
                const myds_metal_buffer_t *fin_deps[] = {ids_buf, q_buf};
                myds_metal_barrier_buffers(fin_deps, 2u);
                if (rope_done < 64u) {
                    if (!myds_v41_gpu_rope_heads_slice(s, q_buf, rope_done, 64u - rope_done, pos, true)) return;
                    myds_metal_barrier_buffer(q_buf);
                }
                myds_v41_gpu_attn_kv_reuse_reset(s);
            }
            if (!myds_metal_commit_batch()) return;
            double avg = (now_us() - t0) / (double)iters;
            if (arm == 0) base_us = avg; else fast_us = avg;
        }
        printf("    nc=%6u: base=%7.1f us  fast=%7.1f us  delta=%+6.1f us (%.2fx)\n",
               nc, base_us, fast_us, fast_us - base_us, base_us / fast_us);
    }

    printf("  -- L20 Full Pipeline + trailing 32 Q_B + 64 Q RoPE (batched GPU CB) --\n");
    for (size_t ni = 1; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u;
        fill_realistic_f32_scores((float *)myds_metal_buffer_data(scores_buf), nc, 0xaaaau + nc);

        double base_us = 0.0, fast_us = 0.0;
        for (int arm = 0; arm < 2; arm++) {
            setenv("MYDS_DECODE_TOPK_FAST", arm ? "1" : "0", 1);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            for (int w = 0; w < warmup; w++) {
                if (!myds_metal_begin_batch()) return;
                if (arm == 0) {
                    if (!myds_v41_gpu_candidates(block_scores, scores_buf, nc, pos, 1u)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                    if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                    if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in, 32u * 512u, 32u * 512u, false)) return;
                    const myds_metal_buffer_t *fin_deps[] = {ids_buf, q_buf};
                    myds_metal_barrier_buffers(fin_deps, 2u);
                    if (!myds_v41_gpu_rope_heads_slice(s, q_buf, 0u, 64u, pos, true)) return;
                    myds_metal_barrier_buffer(q_buf);
                } else {
                    myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                    myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                    if (!myds_v41_gpu_l20_decode_candidates_nan(block_scores, scores_buf, nc, pos, 1u, s)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    const myds_metal_buffer_t *blk_q_deps[] = {block_sel, q_buf};
                    myds_metal_barrier_buffers(blk_q_deps, 2u);
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    myds_v41_gpu_attn_kv_reuse_rebind_topk_qb(s, qb_w, qb_in, pos);
                    if (!myds_v41_gpu_topk_l20_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                    const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                    const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                    if (qb_done < 64u) {
                        if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in,
                                qb_done * 512u, (64u - qb_done) * 512u, false)) return;
                    }
                    const myds_metal_buffer_t *fin_deps[] = {ids_buf, q_buf};
                    myds_metal_barrier_buffers(fin_deps, 2u);
                    if (rope_done < 64u) {
                        if (!myds_v41_gpu_rope_heads_slice(s, q_buf, rope_done, 64u - rope_done, pos, true)) return;
                        myds_metal_barrier_buffer(q_buf);
                    }
                }
                myds_v41_gpu_attn_kv_reuse_reset(s);
                if (!myds_metal_commit_batch()) return;
            }
            const double t0 = now_us();
            if (!myds_metal_begin_batch()) return;
            for (int it = 0; it < iters; it++) {
                if (arm == 0) {
                    if (!myds_v41_gpu_candidates(block_scores, scores_buf, nc, pos, 1u)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                    if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                    if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in, 32u * 512u, 32u * 512u, false)) return;
                    const myds_metal_buffer_t *fin_deps[] = {ids_buf, q_buf};
                    myds_metal_barrier_buffers(fin_deps, 2u);
                    if (!myds_v41_gpu_rope_heads_slice(s, q_buf, 0u, 64u, pos, true)) return;
                    myds_metal_barrier_buffer(q_buf);
                } else {
                    myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                    myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                    if (!myds_v41_gpu_l20_decode_candidates_nan(block_scores, scores_buf, nc, pos, 1u, s)) return;
                    myds_metal_barrier_buffer(block_scores);
                    if (!myds_v41_gpu_topk(block_sel, block_scores, blocks, 2048u, s)) return;
                    const myds_metal_buffer_t *blk_q_deps[] = {block_sel, q_buf};
                    myds_metal_barrier_buffers(blk_q_deps, 2u);
                    if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, block_sel, blocks, 2048u, nc)) return;
                    myds_v41_gpu_attn_kv_reuse_rebind_topk_qb(s, qb_w, qb_in, pos);
                    if (!myds_v41_gpu_topk_l20_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                    const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                    const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                    if (qb_done < 64u) {
                        if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in,
                                qb_done * 512u, (64u - qb_done) * 512u, false)) return;
                    }
                    const myds_metal_buffer_t *fin_deps[] = {ids_buf, q_buf};
                    myds_metal_barrier_buffers(fin_deps, 2u);
                    if (rope_done < 64u) {
                        if (!myds_v41_gpu_rope_heads_slice(s, q_buf, rope_done, 64u - rope_done, pos, true)) return;
                        myds_metal_barrier_buffer(q_buf);
                    }
                }
                myds_v41_gpu_attn_kv_reuse_reset(s);
            }
            if (!myds_metal_commit_batch()) return;
            double avg = (now_us() - t0) / (double)iters;
            if (arm == 0) base_us = avg; else fast_us = avg;
        }
        printf("    nc=%6u: base=%7.1f us  fast=%7.1f us  delta=%+6.1f us (%.2fx)\n",
               nc, base_us, fast_us, fast_us - base_us, base_us / fast_us);
    }
    myds_metal_set_auto_barrier(1);
    myds_metal_set_concurrent(0);
}

int main(int argc, char **argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) do_bench = true;
    }

    @autoreleasepool {
        if (test_knob_parsing() != 0) return 1;

        setenv("MYDS_DENSE_Q4_V41", "1", 1);
        setenv("MYDS_INDEXER_DECODE_COMPACT", "1", 1);
        setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
        setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);

        if (!myds_metal_init("bin")) {
            fprintf(stderr, "FAIL: myds_metal_init failed\n");
            return 1;
        }
        myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(262144u);
        if (!s) {
            fprintf(stderr, "FAIL: myds_v41_gpu_scratch_create failed\n");
            myds_metal_cleanup();
            return 1;
        }

        myds_metal_buffer_t *scores_buf = myds_metal_buffer_alloc(262144u * sizeof(float));
        myds_metal_buffer_t *block_scores = myds_metal_buffer_alloc(32768u * sizeof(float));
        myds_metal_buffer_t *block_sel_ref = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *block_sel_got = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *ids_ref = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *ids_got = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *q_ref = myds_metal_buffer_alloc(64u * 512u * sizeof(float));
        myds_metal_buffer_t *q_got = myds_metal_buffer_alloc(64u * 512u * sizeof(float));
        myds_metal_buffer_t *qb_w = myds_metal_buffer_alloc(32768u * 5u * 144u);
        myds_metal_buffer_t *qb_in = myds_metal_buffer_alloc(1280u * sizeof(float));
        if (!scores_buf || !block_scores || !block_sel_ref || !block_sel_got ||
            !ids_ref || !ids_got || !q_ref || !q_got || !qb_w || !qb_in) {
            fprintf(stderr, "FAIL: buffer allocation failed\n");
            return 1;
        }
        fill_qb_weights_and_input(qb_w, qb_in);

        int rc = 0;
        if (!rc) rc = test_overlap_prefix_root(s, scores_buf, ids_ref, ids_got, q_ref, q_got, qb_w, qb_in);
        if (!rc) rc = test_prefix_top2048(s, scores_buf, ids_ref, ids_got);
        if (!rc) rc = test_l20_self_compact(s, scores_buf, block_scores, block_sel_ref, block_sel_got,
                                            ids_ref, ids_got, q_ref, q_got, qb_w, qb_in);
        if (!rc && do_bench) {
            run_bench(s, scores_buf, block_scores, block_sel_got, ids_got, q_got, qb_w, qb_in);
        }

        myds_metal_buffer_free(scores_buf);
        myds_metal_buffer_free(block_scores);
        myds_metal_buffer_free(block_sel_ref);
        myds_metal_buffer_free(block_sel_got);
        myds_metal_buffer_free(ids_ref);
        myds_metal_buffer_free(ids_got);
        myds_metal_buffer_free(q_ref);
        myds_metal_buffer_free(q_got);
        myds_metal_buffer_free(qb_w);
        myds_metal_buffer_free(qb_in);
        myds_v41_gpu_scratch_free(s);
        myds_metal_cleanup();
        unsetenv("MYDS_DECODE_TOPK_FAST");

        if (rc == 0) {
            printf("\n[ALL PASS] test_decode_topk_fast: strict 0-ULP verified!\n");
        }
        return rc;
    }
}
