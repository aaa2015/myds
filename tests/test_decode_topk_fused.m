/* tests/test_decode_topk_fused.m
 *
 * Strict 0-ULP verification + M4 benchmark for [D-I2] (MYDS_INDEXER_DECODE_TOPK_FUSED):
 *   - kernel_myds_indexer_score_one_mma_v2_{fp4,bf16,f32} (Root L2/8/14)
 *   - kernel_myds_indexer_score_one_mma_v2_l20_{fp4,bf16} (L20 block_max)
 *   - kernel_myds_indexer_score_compact_mma_v2_{fp4,bf16} (L24..36 compact)
 *   - kernel_myds_decode_topk_pivot_gather + kernel_myds_decode_topk_final_and_clear
 *
 * Proves strict 0-ULP (memcmp == 0) vs Arm C [D-I1] (score_one_mma / score_compact_mma)
 * + baseline argsort across nc in {16384, 32768, 65536, 131072, 262144}, FP4 & BF16,
 * all 15 torture patterns (incl. MODE_ZOOM dense & sub-bin overflow, ties, all-zeros,
 * -0/+0, <512 valid, +INF, visible & hidden NaNs), L2/8/14, L20, L24..36 compact,
 * and concurrent=1, auto_barrier=0 + overlap_topk_qb (Q_B + Q RoPE).
 *
 * Run:
 *   ./bin/test_decode_topk_fused           (full 0-ULP correctness suite)
 *   ./bin/test_decode_topk_fused --bench   (correctness + M4 GPU latency bench)
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

static uint32_t g_rng = 0x13572468u;
static uint32_t xorshift32(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static float rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((float)(xorshift32() & 0xffffffu) / (float)0x1000000u);
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

static uint16_t f32_to_bf16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16u) & 1u);
    return (uint16_t)(u >> 16u);
}

static float fp4_dequant_cpu(uint32_t code4, uint8_t exp_u8) {
    uint32_t sign = (code4 & 8u) << 28u;
    uint32_t mag  = code4 & 7u;
    if (mag == 0u || exp_u8 == 0u) return 0.0f;
    uint32_t off = (mag == 1u ? 0u : mag) << 22u;
    uint32_t vbits = sign | (0x3f000000u + off);
    uint32_t sbits = (uint32_t)exp_u8 << 23u;
    float v, s;
    memcpy(&v, &vbits, 4);
    memcpy(&s, &sbits, 4);
    return v * s;
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static void fill_k_caches(uint8_t *fp4_buf, uint16_t *bf16_buf, float *f32_buf,
                          uint32_t n_rows, uint8_t exp_lo, uint8_t exp_hi, bool inject_extreme) {
    for (uint32_t r = 0; r < n_rows; r++) {
        uint8_t *row = fp4_buf + (size_t)r * 68u;
        uint8_t exps[4];
        const uint32_t span = (uint32_t)(exp_hi - exp_lo + 1u);
        for (uint32_t b = 0; b < 4u; b++) {
            exps[b] = (uint8_t)(exp_lo + (xorshift32() % span));
        }
        if (inject_extreme && (r % 19u == 0u)) {
            static const uint8_t extremes[8] = {0u, 1u, 60u, 113u, 114u, 138u, 140u, 148u};
            exps[0] = extremes[(r / 19u) % 8u];
            exps[2] = extremes[(r / 19u + 3u) % 8u];
        }
        for (uint32_t j = 0; j < 64u; j++) {
            uint32_t c0 = xorshift32() & 0xfu;
            uint32_t c1 = xorshift32() & 0xfu;
            uint8_t eb = exps[j >> 4u];
            if (eb == 0u || (eb == 1u && (c0 & 7u) == 1u)) c0 = 0u;
            if (eb == 0u || (eb == 1u && (c1 & 7u) == 1u)) c1 = 0u;
            row[j] = (uint8_t)(c0 | (c1 << 4u));
            if (bf16_buf || f32_buf) {
                float v0 = fp4_dequant_cpu(c0, eb);
                float v1 = fp4_dequant_cpu(c1, eb);
                if (bf16_buf) {
                    bf16_buf[(size_t)r * 128u + j * 2u + 0u] = f32_to_bf16_bits(v0);
                    bf16_buf[(size_t)r * 128u + j * 2u + 1u] = f32_to_bf16_bits(v1);
                }
                if (f32_buf) {
                    f32_buf[(size_t)r * 128u + j * 2u + 0u] = v0;
                    f32_buf[(size_t)r * 128u + j * 2u + 1u] = v1;
                }
            }
        }
        memcpy(row + 64u, exps, 4u);
    }
}

static void fill_qb_weights_and_input(myds_metal_buffer_t *qb_w, myds_metal_buffer_t *qb_in) {
    uint8_t *w = (uint8_t *)myds_metal_buffer_data(qb_w);
    const size_t total_blocks = 32768u * 5u;
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

static int compare_f32_exact(const float *ref, const float *got, uint32_t n, const char *label) {
    if (memcmp(ref, got, (size_t)n * sizeof(float)) == 0) return 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rb = f32_to_bits(ref[i]), gb = f32_to_bits(got[i]);
        if (rb != gb) {
            fprintf(stderr, "FAIL [%s]: f32 mismatch at i=%u: ref=%.8g (0x%08x) got=%.8g (0x%08x)\n",
                    label, i, ref[i], rb, got[i], gb);
            return 1;
        }
    }
    return 1;
}

static int compare_i32_exact(const int32_t *ref, const int32_t *got, uint32_t n,
                             const float *scores, const char *label) {
    if (memcmp(ref, got, (size_t)n * sizeof(int32_t)) == 0) return 0;
    for (uint32_t i = 0; i < n; i++) {
        if (ref[i] != got[i]) {
            fprintf(stderr, "FAIL [%s]: i32 mismatch at rank %u: ref=%d", label, i, ref[i]);
            if (scores && ref[i] >= 0) {
                fprintf(stderr, " (s=%.8g bits=0x%08x)", scores[ref[i]], f32_to_bits(scores[ref[i]]));
            }
            fprintf(stderr, " vs got=%d", got[i]);
            if (scores && got[i] >= 0) {
                fprintf(stderr, " (s=%.8g bits=0x%08x)", scores[got[i]], f32_to_bits(scores[got[i]]));
            }
            fprintf(stderr, "\n");
            return 1;
        }
    }
    return 1;
}

static int verify_scratch_zeroed(const myds_v41_gpu_scratch_t *s, const char *label) {
    const uint32_t *h = (const uint32_t *)myds_metal_buffer_data(
        (myds_metal_buffer_t *)myds_v41_gpu_scratch_decode_topk_hist(s));
    const uint32_t *m = (const uint32_t *)myds_metal_buffer_data(
        (myds_metal_buffer_t *)myds_v41_gpu_scratch_decode_topk_meta(s));
    const uint32_t *sh = (const uint32_t *)myds_metal_buffer_data(
        (myds_metal_buffer_t *)myds_v41_gpu_scratch_decode_topk_subhist(s));
    for (uint32_t i = 0; i < 256u; i++) {
        if (h[i] != 0u) {
            fprintf(stderr, "FAIL [%s]: hist[%u]=%u not self-cleared!\n", label, i, h[i]);
            return 1;
        }
    }
    for (uint32_t i = 0; i < 16u; i++) {
        if (m[i] != 0u) {
            fprintf(stderr, "FAIL [%s]: meta[%u]=%u not self-cleared!\n", label, i, m[i]);
            return 1;
        }
    }
    for (uint32_t i = 0; i < 1024u; i++) {
        if (sh[i] != 0u) {
            fprintf(stderr, "FAIL [%s]: subhist[%u]=%u not self-cleared!\n", label, i, sh[i]);
            return 1;
        }
    }
    return 0;
}

/* [1/5] Strict fail-closed knob parsing & scratch_create rejection */
static int test_knob_parsing(void) {
    printf("[1/5] Verifying MYDS_INDEXER_DECODE_TOPK_FUSED strict fail-closed parsing...\n");
    unsetenv("MYDS_INDEXER_DECODE_TOPK_FUSED");
    if (myds_v41_gpu_decode_topk_fused_enabled() != 0) {
        fprintf(stderr, "FAIL: absent MYDS_INDEXER_DECODE_TOPK_FUSED must default to 0\n");
        return 1;
    }
    setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
    if (myds_v41_gpu_decode_topk_fused_enabled() != 0) {
        fprintf(stderr, "FAIL: MYDS_INDEXER_DECODE_TOPK_FUSED=0 must return 0\n");
        return 1;
    }
    setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
    if (myds_v41_gpu_decode_topk_fused_enabled() != 1) {
        fprintf(stderr, "FAIL: MYDS_INDEXER_DECODE_TOPK_FUSED=1 must return 1\n");
        return 1;
    }
    const char *bad[] = {"", "2", "-1", "true", "yes", "01", "1 "};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", bad[i], 1);
        if (myds_v41_gpu_decode_topk_fused_enabled() != -1) {
            fprintf(stderr, "FAIL: invalid MYDS_INDEXER_DECODE_TOPK_FUSED='%s' must return -1\n", bad[i]);
            return 1;
        }
    }
    setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
    printf("  PASS: strict knob parsing (NULL/0->0, 1->1, invalid->-1).\n");
    return 0;
}

/* [2/5] Strict 0-ULP (memcmp == 0) score_one_mma_v2 & score_compact_mma_v2 vs Arm C [D-I1] */
static int test_scorer_0ulp(myds_v41_gpu_scratch_t *s,
                            const myds_metal_buffer_t *q,
                            const myds_metal_buffer_t *w,
                            const myds_metal_buffer_t *k_fp4,
                            const myds_metal_buffer_t *k_bf16,
                            const myds_metal_buffer_t *k_f32,
                            const myds_metal_buffer_t *mask,
                            myds_metal_buffer_t *sc_ref,
                            myds_metal_buffer_t *sc_got,
                            myds_metal_buffer_t *blk_ref,
                            myds_metal_buffer_t *blk_got,
                            myds_metal_buffer_t *bsel_ref,
                            myds_metal_buffer_t *bsel_got,
                            myds_metal_buffer_t *ids_ref,
                            myds_metal_buffer_t *ids_got) {
    printf("[2/5] Testing [D-I2] 0-ULP vs Arm C [D-I1] (Root L2/8/14, L20, L24..36 compact, FP4 & BF16 & F32)...\n");
    myds_v41_gpu_decode_topk_fused_counters(NULL, true);
    uint64_t expected_root_enc = 0u, expected_l20_enc = 0u;
    const uint32_t root_ncs[] = {16384u, 16389u, 32768u, 65536u, 131072u, 262144u};
    const uint32_t l20_ncs[]  = {16392u, 32768u, 65536u, 131072u, 262144u};

    /* 2a. Root L2/8/14: FP4 & BF16 across all nc, mask=0/1, first_row=0/137 */
    for (size_t ni = 0; ni < sizeof(root_ncs) / sizeof(root_ncs[0]); ni++) {
        const uint32_t nc = root_ncs[ni];
        for (int is_fp4 = 1; is_fp4 >= 0; is_fp4--) {
            const uint32_t mode = is_fp4 ? MYDS_INDEXER_CACHE_FP4 : MYDS_INDEXER_CACHE_BF16;
            const myds_metal_buffer_t *kbuf = is_fp4 ? k_fp4 : k_bf16;
            for (int use_mask = 0; use_mask <= 1; use_mask++) {
                /* Test mask=1 on 16K/32K/262K to keep runtime snappy while hitting all edge sizes */
                if (use_mask && nc != 16384u && nc != 16389u && nc != 32768u && nc != 262144u) continue;
                const myds_metal_buffer_t *mbuf = use_mask ? mask : NULL;
                const uint32_t first_row = use_mask ? 137u : 0u;

                /* Reference: Arm C [D-I1] score_one_mma + baseline argsort(top=512) */
                setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
                setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
                setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
                myds_metal_set_auto_barrier(1);
                myds_metal_set_concurrent(0);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_indexscores_head_slots_mode(sc_ref, q, w, kbuf, mbuf, nc, 32u, mode, first_row) ||
                    !myds_v41_gpu_topk(ids_ref, sc_ref, nc, 512u, s) ||
                    !myds_metal_commit_batch()) {
                    fprintf(stderr, "FAIL: Arm C reference Root nc=%u failed\n", nc);
                    return 1;
                }

                /* Candidate: [D-I2] score_one_mma_v2 + 2-pass topk_fused_decode(top=512) */
                setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_indexscores_fused_decode_mode(s, sc_got, NULL, q, w, kbuf, mbuf,
                                                                 nc, mode, first_row, nc, false) ||
                    !myds_v41_gpu_topk_fused_decode(s, ids_got, sc_got, nc, 512u, false, 0u) ||
                    !myds_metal_commit_batch()) {
                    fprintf(stderr, "FAIL: [D-I2] Root nc=%u failed\n", nc);
                    return 1;
                }

                char lbl[128];
                snprintf(lbl, sizeof(lbl), "Root %s mask=%d nc=%u", is_fp4 ? "FP4" : "BF16", use_mask, nc);
                if (compare_f32_exact((const float *)myds_metal_buffer_data(sc_ref),
                                      (const float *)myds_metal_buffer_data(sc_got), nc, lbl) != 0) return 1;
                if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                      (const int32_t *)myds_metal_buffer_data(ids_got),
                                      512u, (const float *)myds_metal_buffer_data(sc_ref), lbl) != 0) return 1;
                if (verify_scratch_zeroed(s, lbl) != 0) return 1;
                expected_root_enc++;
            }
        }
    }

    /* Also verify F32 fallback kernel_myds_indexer_score_one_mma_v2_f32 at 16384 */
    {
        const uint32_t nc = 16384u;
        setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
        setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
        if (!myds_metal_begin_batch() ||
            !myds_v41_gpu_indexscores_head_slots_mode(sc_ref, q, w, k_f32, NULL, nc, 32u, MYDS_INDEXER_CACHE_F32, 0u) ||
            !myds_v41_gpu_topk(ids_ref, sc_ref, nc, 512u, s) ||
            !myds_metal_commit_batch()) return 1;

        setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
        if (!myds_metal_begin_batch() ||
            !myds_v41_gpu_indexscores_fused_decode_mode(s, sc_got, NULL, q, w, k_f32, NULL,
                                                         nc, MYDS_INDEXER_CACHE_F32, 0u, nc, false) ||
            !myds_v41_gpu_topk_fused_decode(s, ids_got, sc_got, nc, 512u, false, 0u) ||
            !myds_metal_commit_batch()) return 1;
        if (compare_f32_exact((const float *)myds_metal_buffer_data(sc_ref),
                              (const float *)myds_metal_buffer_data(sc_got), nc, "Root F32 16384") != 0) return 1;
        if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                              (const int32_t *)myds_metal_buffer_data(ids_got),
                              512u, (const float *)myds_metal_buffer_data(sc_ref), "Root F32 16384") != 0) return 1;
        if (verify_scratch_zeroed(s, "Root F32 16384") != 0) return 1;
        expected_root_enc++;
    }

    /* 2b. L20: FP4 & BF16 across all nc > 16384 (incl. 16392, 32768..262144) */
    int32_t *cblk_ref = (int32_t *)malloc(2048u * sizeof(int32_t));
    if (!cblk_ref) return 1;
    for (size_t ni = 0; ni < sizeof(l20_ncs) / sizeof(l20_ncs[0]); ni++) {
        const uint32_t nc = l20_ncs[ni];
        const uint32_t n_blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u;
        for (int is_fp4 = 1; is_fp4 >= 0; is_fp4--) {
            const uint32_t mode = is_fp4 ? MYDS_INDEXER_CACHE_FP4 : MYDS_INDEXER_CACHE_BF16;
            const myds_metal_buffer_t *kbuf = is_fp4 ? k_fp4 : k_bf16;

            /* Arm C [D-I1] L20 reference */
            setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
            setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
            myds_metal_set_auto_barrier(1);
            myds_metal_set_concurrent(0);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_indexscores_head_slots_mode(sc_ref, q, w, kbuf, NULL, nc, 32u, mode, 0u) ||
                !myds_v41_gpu_candidates(blk_ref, sc_ref, nc, pos, 1u) ||
                !myds_v41_gpu_topk(bsel_ref, blk_ref, n_blocks, 2048u, s) ||
                !myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_ref, n_blocks, 2048u, nc) ||
                !myds_v41_gpu_topk(ids_ref, sc_ref, nc, 512u, s) ||
                !myds_metal_commit_batch()) {
                fprintf(stderr, "FAIL: Arm C L20 reference nc=%u failed\n", nc);
                free(cblk_ref);
                return 1;
            }
            memcpy(cblk_ref,
                   myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_blocks(s)),
                   2048u * sizeof(int32_t));

            /* [D-I2] Fused L20: score_one_mma_v2_l20 + 2-pass block Top-2048 (w/ fused prepare) + self-compact Top-512 */
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_indexscores_fused_decode_mode(s, sc_got, blk_got, q, w, kbuf, NULL,
                                                             nc, mode, 0u, nc, true) ||
                !myds_v41_gpu_topk_fused_decode(s, bsel_got, blk_got, n_blocks, 2048u, true, nc) ||
                !myds_v41_gpu_topk_l20_fused_self_compact_decode(ids_got, sc_got, nc, s) ||
                !myds_metal_commit_batch()) {
                fprintf(stderr, "FAIL: [D-I2] L20 nc=%u failed\n", nc);
                free(cblk_ref);
                return 1;
            }

            char lbl[128];
            snprintf(lbl, sizeof(lbl), "L20 %s nc=%u", is_fp4 ? "FP4" : "BF16", nc);
            if (compare_f32_exact((const float *)myds_metal_buffer_data(sc_ref),
                                  (const float *)myds_metal_buffer_data(sc_got), nc, lbl) != 0) {
                free(cblk_ref); return 1;
            }
            if (compare_f32_exact((const float *)myds_metal_buffer_data(blk_ref),
                                  (const float *)myds_metal_buffer_data(blk_got), n_blocks, lbl) != 0) {
                free(cblk_ref); return 1;
            }
            if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(bsel_ref),
                                  (const int32_t *)myds_metal_buffer_data(bsel_got),
                                  2048u, (const float *)myds_metal_buffer_data(blk_ref), lbl) != 0) {
                free(cblk_ref); return 1;
            }
            if (compare_i32_exact(cblk_ref,
                                  (const int32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_blocks(s)),
                                  2048u, NULL, lbl) != 0) {
                free(cblk_ref); return 1;
            }
            if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                  (const int32_t *)myds_metal_buffer_data(ids_got),
                                  512u, (const float *)myds_metal_buffer_data(sc_ref), lbl) != 0) {
                free(cblk_ref); return 1;
            }
            if (verify_scratch_zeroed(s, lbl) != 0) {
                free(cblk_ref); return 1;
            }
            expected_l20_enc++;
        }
    }
    free(cblk_ref);
    {
        myds_v41_gpu_decode_topk_fused_counters_t cnt;
        myds_v41_gpu_decode_topk_fused_counters(&cnt, false);
        if (cnt.requests != expected_root_enc + expected_l20_enc ||
            cnt.root_encoded != expected_root_enc ||
            cnt.l20_encoded != expected_l20_enc ||
            cnt.failures != 0u) {
            fprintf(stderr, "FAIL: scorer counters mismatch req=%llu (exp %llu) root=%llu l20=%llu fail=%llu\n",
                    (unsigned long long)cnt.requests,
                    (unsigned long long)(expected_root_enc + expected_l20_enc),
                    (unsigned long long)cnt.root_encoded,
                    (unsigned long long)cnt.l20_encoded,
                    (unsigned long long)cnt.failures);
            return 1;
        }
    }

    /* 2c. L24..36 Compact Decode MMA v2 vs Arm C score_compact_mma (FP4 & BF16, mask=0/1, -1 padded slots) */
    float *c_sc_ref = (float *)malloc(16384u * sizeof(float));
    int32_t *c_rw_ref = (int32_t *)malloc(16384u * sizeof(int32_t));
    if (!c_sc_ref || !c_rw_ref) return 1;
    const uint32_t nc_c = 65536u;
    const uint32_t nb_c = nc_c / 8u;
    int32_t *bsel_p = (int32_t *)myds_metal_buffer_data(bsel_ref);
    /* Include duplicates & out-of-range slots to exercise -1 padding in compact_blocks */
    for (uint32_t i = 0; i < 2048u; i++) {
        bsel_p[i] = (i < 1900u) ? (int32_t)((i * 3u + 1u) % nb_c) : (int32_t)(nb_c - 1u);
    }
    myds_metal_set_auto_barrier(1);
    myds_metal_set_concurrent(0);
    if (!myds_metal_begin_batch() ||
        !myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_ref, nb_c, 2048u, nc_c) ||
        !myds_metal_commit_batch()) {
        free(c_sc_ref); free(c_rw_ref); return 1;
    }

    for (int is_fp4 = 1; is_fp4 >= 0; is_fp4--) {
        const uint32_t mode = is_fp4 ? MYDS_INDEXER_CACHE_FP4 : MYDS_INDEXER_CACHE_BF16;
        const myds_metal_buffer_t *kbuf = is_fp4 ? k_fp4 : k_bf16;
        for (int use_mask = 0; use_mask <= 1; use_mask++) {
            const myds_metal_buffer_t *mbuf = use_mask ? mask : NULL;
            const uint32_t first_row = use_mask ? 137u : 0u;

            setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_indexscores_compact_decode_mode(s, q, w, kbuf, mbuf, nc_c, mode, first_row) ||
                !myds_metal_commit_batch()) {
                free(c_sc_ref); free(c_rw_ref); return 1;
            }
            memcpy(c_sc_ref,
                   myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_scores(s)),
                   16384u * sizeof(float));
            memcpy(c_rw_ref,
                   myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_rows(s)),
                   16384u * sizeof(int32_t));

            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_indexscores_compact_decode_mode(s, q, w, kbuf, mbuf, nc_c, mode, first_row) ||
                !myds_metal_commit_batch()) {
                free(c_sc_ref); free(c_rw_ref); return 1;
            }
            char lbl[128];
            snprintf(lbl, sizeof(lbl), "L24..36 Compact %s mask=%d", is_fp4 ? "FP4" : "BF16", use_mask);
            if (compare_f32_exact(c_sc_ref,
                                  (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_scores(s)),
                                  16384u, lbl) != 0) {
                free(c_sc_ref); free(c_rw_ref); return 1;
            }
            if (compare_i32_exact(c_rw_ref,
                                  (const int32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_rows(s)),
                                  16384u, NULL, lbl) != 0) {
                free(c_sc_ref); free(c_rw_ref); return 1;
            }
        }
    }
    free(c_sc_ref);
    free(c_rw_ref);
    printf("  PASS: strict 0-ULP (memcmp == 0) across Root (16K..262K + tail), L20 (16K+8..262K), L24..36 compact, FP4 & BF16 & F32.\n");
    return 0;
}

/* [3/5] 15 Torture Patterns across all 5 sizes (16K..262K) for BOTH Root Top-512 AND L20 Top-2048 + Top-512 */
enum score_pattern {
    PAT_RANDOM = 0,
    PAT_HEAVY_TIES,
    PAT_BOUNDARY_TIES,
    PAT_ALL_ZEROS,
    PAT_SIGNED_ZEROS,
    PAT_LT512_VALID_0,
    PAT_LT512_VALID_17,
    PAT_LT512_VALID_511,
    PAT_MODE_ZOOM_DENSE,
    PAT_MODE_ZOOM_SUBBIN_OVERFLOW,
    PAT_MODE_ZOOM_TINY_POS,
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
        case PAT_MODE_ZOOM_DENSE: return "mode_zoom_dense";
        case PAT_MODE_ZOOM_SUBBIN_OVERFLOW: return "mode_zoom_subbin_ovf";
        case PAT_MODE_ZOOM_TINY_POS: return "mode_zoom_tiny_pos";
        case PAT_ALL_POS_INF: return "all_pos_inf";
        case PAT_VISIBLE_NAN: return "visible_nan";
        case PAT_HIDDEN_NAN_UNSELECTED_BLOCK: return "hidden_nan_unselected";
        case PAT_HIDDEN_NAN_LAST_BLOCK: return "hidden_nan_last_block";
        default: return "?";
    }
}

static void fill_torture_scores(float *dst, uint32_t nc, enum score_pattern pat, uint32_t seed) {
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
        case PAT_BOUNDARY_TIES:
            for (uint32_t i = 0; i < nc; i++) dst[i] = -1.0f;
            for (uint32_t k = 0; k < 300u; k++) {
                dst[(k * 97u + 13u) % nc] = 10.0f + (float)(k % 7u);
            }
            for (uint32_t i = 1024u; i < nc; i += 19u) {
                if (dst[i] < 0.0f) dst[i] = 5.0f;
            }
            for (uint32_t j = 0; j < 20u; j++) {
                dst[3072u + j * 3u] = 12.0f;
                dst[7168u + j * 5u] = 5.0f;
            }
            break;
        case PAT_ALL_ZEROS:
            for (uint32_t i = 0; i < nc; i++) dst[i] = 0.0f;
            break;
        case PAT_SIGNED_ZEROS:
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = (xorshift32() & 1u) ? bits_to_f32(0x80000000u) : 0.0f;
            }
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
        case PAT_MODE_ZOOM_DENSE:
            /* Pack >6000 distinct float values inside single coarse bin [1.0f, 1.125f)
             * (raw 0x3f800000..0x3f8fffff, 20-bit span) + 150 strictly above (2.0f..3.0f). */
            for (uint32_t i = 0; i < nc; i++) {
                uint32_t frac = xorshift32() & 0x000fffffu;
                dst[i] = bits_to_f32(0x3f800000u | frac);
            }
            for (uint32_t k = 0; k < 150u; k++) {
                dst[(k * 101u + 7u) % nc] = 2.0f + (float)(k % 17u) * 0.05f;
            }
            break;
        case PAT_MODE_ZOOM_SUBBIN_OVERFLOW:
            /* Pack ALL values inside a SINGLE 10-bit sub-bin of coarse bin [1.0f, 1.125f):
             * raw 0x3f880000..0x3f8803ff (1024 distinct ULPs, repeated across nc elements!)
             * Forces multi-pass zoom refinement AND tie resolution at 1-ULP precision! */
            for (uint32_t i = 0; i < nc; i++) {
                uint32_t u = 0x3f880000u | (xorshift32() & 0x000003ffu);
                dst[i] = bits_to_f32(u);
            }
            /* Add 200 elements strictly above in the same coarse bin (0x3f8c0000..) */
            for (uint32_t k = 0; k < 200u; k++) {
                dst[(k * 73u + 19u) % nc] = bits_to_f32(0x3f8c0000u | (k & 0xffu));
            }
            /* Add 50 elements above the coarse bin */
            for (uint32_t k = 0; k < 50u; k++) {
                dst[(k * 211u + 3u) % nc] = 4.0f + (float)k * 0.01f;
            }
            break;
        case PAT_MODE_ZOOM_TINY_POS:
            /* Pack ALL values inside bin 52 (+0, 2^-11) which has a 30-bit span!
             * Exercises 3-pass zoom loop (30 -> 20 -> 10 -> 0 bits). */
            for (uint32_t i = 0; i < nc; i++) {
                /* Tiny positive floats raw 0x00800000..0x39ffffff (normal tiny floats) */
                uint32_t r = 0x30000000u + (xorshift32() & 0x07ffffffu);
                dst[i] = bits_to_f32(r);
            }
            break;
        case PAT_ALL_POS_INF:
            for (uint32_t i = 0; i < nc; i++) dst[i] = INFINITY;
            break;
        case PAT_VISIBLE_NAN:
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = ((float)(int32_t)(xorshift32() & 0xfffu) - 2048.0f) * 0.01f;
            }
            dst[17] = bits_to_f32(0x7fc00001u);
            dst[2049] = bits_to_f32(0x7f800001u);
            break;
        case PAT_HIDDEN_NAN_UNSELECTED_BLOCK:
            for (uint32_t i = 0; i < nc; i++) dst[i] = -100.0f + (float)(i % 13u);
            for (uint32_t b = 0; b < 2200u && b * 8u < nc; b++) {
                dst[b * 8u] = 100.0f + (float)(b % 50u);
            }
            if (nc > 2500u * 8u + 5u) {
                dst[2500u * 8u + 0u] = -50.0f;
                dst[2500u * 8u + 3u] = bits_to_f32(0x7fc01234u);
            } else {
                dst[nc - 10u] = bits_to_f32(0x7fc01234u);
            }
            break;
        case PAT_HIDDEN_NAN_LAST_BLOCK:
            for (uint32_t i = 0; i < nc; i++) {
                dst[i] = ((float)(int32_t)(xorshift32() & 0xfffu) - 2048.0f) * 0.01f;
            }
            dst[nc - 1u] = bits_to_f32(0x7fc05678u);
            break;
        default:
            break;
    }
}

static int test_torture_suite(myds_v41_gpu_scratch_t *s,
                              myds_metal_buffer_t *scores_buf,
                              myds_metal_buffer_t *blk_ref,
                              myds_metal_buffer_t *blk_got,
                              myds_metal_buffer_t *bsel_ref,
                              myds_metal_buffer_t *bsel_got,
                              myds_metal_buffer_t *ids_ref,
                              myds_metal_buffer_t *ids_got) {
    printf("[3/5] Testing all %d torture patterns x 5 sizes (16K..262K) for Root Top-512 & L20 Top-2048+Top-512...\n",
           (int)PAT_COUNT);
    const uint32_t ncs[] = {16384u, 32768u, 65536u, 131072u, 262144u};
    int32_t *cblk_ref = (int32_t *)malloc(2048u * sizeof(int32_t));
    if (!cblk_ref) return 1;

    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t n_blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u;

        for (int p = 0; p < (int)PAT_COUNT; p++) {
            float *sc = (float *)myds_metal_buffer_data(scores_buf);
            fill_torture_scores(sc, nc, (enum score_pattern)p, 0x55aa0000u + nc + (uint32_t)p * 37u);

            /* 3a. Root Top-512 */
            setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
            myds_metal_set_auto_barrier(1);
            myds_metal_set_concurrent(0);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_topk(ids_ref, scores_buf, nc, 512u, s) ||
                !myds_metal_commit_batch()) {
                free(cblk_ref); return 1;
            }

            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_decode_topk_hist_only(s, scores_buf, NULL, nc, nc, false) ||
                !myds_v41_gpu_topk_fused_decode(s, ids_got, scores_buf, nc, 512u, false, 0u) ||
                !myds_metal_commit_batch()) {
                free(cblk_ref); return 1;
            }
            char lbl[128];
            snprintf(lbl, sizeof(lbl), "Torture Root nc=%u pat=%s", nc, pattern_name((enum score_pattern)p));
            if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                  (const int32_t *)myds_metal_buffer_data(ids_got),
                                  512u, sc, lbl) != 0) {
                free(cblk_ref); return 1;
            }
            if (verify_scratch_zeroed(s, lbl) != 0) {
                free(cblk_ref); return 1;
            }

            /* 3b. L20 block_max Top-2048 + compact_sorted_blocks + self-compact Top-512 (nc > 16384) */
            if (nc > 16384u) {
                setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
                setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_candidates(blk_ref, scores_buf, nc, pos, 1u) ||
                    !myds_v41_gpu_topk(bsel_ref, blk_ref, n_blocks, 2048u, s) ||
                    !myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_ref, n_blocks, 2048u, nc) ||
                    !myds_metal_commit_batch()) {
                    free(cblk_ref); return 1;
                }
                memcpy(cblk_ref,
                       myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_blocks(s)),
                       2048u * sizeof(int32_t));

                setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_decode_topk_hist_only(s, scores_buf, blk_got, nc, nc, true) ||
                    !myds_v41_gpu_topk_fused_decode(s, bsel_got, blk_got, n_blocks, 2048u, true, nc) ||
                    !myds_v41_gpu_topk_l20_fused_self_compact_decode(ids_got, scores_buf, nc, s) ||
                    !myds_metal_commit_batch()) {
                    free(cblk_ref); return 1;
                }

                snprintf(lbl, sizeof(lbl), "Torture L20 nc=%u pat=%s", nc, pattern_name((enum score_pattern)p));
                if (compare_f32_exact((const float *)myds_metal_buffer_data(blk_ref),
                                      (const float *)myds_metal_buffer_data(blk_got),
                                      n_blocks, lbl) != 0) {
                    free(cblk_ref); return 1;
                }
                if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(bsel_ref),
                                      (const int32_t *)myds_metal_buffer_data(bsel_got),
                                      2048u, (const float *)myds_metal_buffer_data(blk_ref), lbl) != 0) {
                    free(cblk_ref); return 1;
                }
                if (compare_i32_exact(cblk_ref,
                                      (const int32_t *)myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_blocks(s)),
                                      2048u, NULL, lbl) != 0) {
                    free(cblk_ref); return 1;
                }
                if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                      (const int32_t *)myds_metal_buffer_data(ids_got),
                                      512u, sc, lbl) != 0) {
                    free(cblk_ref); return 1;
                }
                if (verify_scratch_zeroed(s, lbl) != 0) {
                    free(cblk_ref); return 1;
                }
            }
        }
    }
    free(cblk_ref);
    printf("  PASS: all %d torture patterns bit-exact (memcmp == 0) & self-cleared across 16K..262K!\n",
           (int)PAT_COUNT);
    return 0;
}

/* [4/5] Concurrent auto_barrier=0 + overlap_topk_qb (Q_B + Q RoPE) + /debug/state counters */
static int test_concurrent_overlap(myds_v41_gpu_scratch_t *s,
                                   myds_metal_buffer_t *scores_buf,
                                   myds_metal_buffer_t *blk_ref,
                                   myds_metal_buffer_t *blk_got,
                                   myds_metal_buffer_t *bsel_ref,
                                   myds_metal_buffer_t *bsel_got,
                                   myds_metal_buffer_t *ids_ref,
                                   myds_metal_buffer_t *ids_got,
                                   myds_metal_buffer_t *q_ref,
                                   myds_metal_buffer_t *q_got,
                                   const myds_metal_buffer_t *qb_w,
                                   const myds_metal_buffer_t *qb_in) {
    printf("[4/5] Testing concurrent=1, auto_barrier=0 + overlap_topk_qb (Root & L20) + counters...\n");
    const uint32_t ncs[] = {16384u, 32768u, 65536u, 131072u, 262144u};
    const enum score_pattern pats[] = {
        PAT_RANDOM, PAT_MODE_ZOOM_DENSE, PAT_MODE_ZOOM_SUBBIN_OVERFLOW,
        PAT_ALL_ZEROS, PAT_VISIBLE_NAN, PAT_HIDDEN_NAN_UNSELECTED_BLOCK
    };

    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t n_blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u;

        if (!compute_reference_q(s, q_ref, qb_w, qb_in, pos, true)) return 1;

        for (size_t pi = 0; pi < sizeof(pats) / sizeof(pats[0]); pi++) {
            const enum score_pattern pat = pats[pi];
            float *sc = (float *)myds_metal_buffer_data(scores_buf);
            fill_torture_scores(sc, nc, pat, 0x99cc0000u + nc + (uint32_t)pi * 53u);

            /* Baseline IDs */
            setenv("MYDS_DECODE_TOPK_FAST", "0", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);
            myds_metal_set_auto_barrier(1);
            myds_metal_set_concurrent(0);
            if (!myds_metal_begin_batch() ||
                !myds_v41_gpu_topk(ids_ref, scores_buf, nc, 512u, s) ||
                (nc > 16384u && (!myds_v41_gpu_candidates(blk_ref, scores_buf, nc, pos, 1u) ||
                                 !myds_v41_gpu_topk(bsel_ref, blk_ref, n_blocks, 2048u, s))) ||
                !myds_metal_commit_batch()) return 1;

            /* 4a. Root concurrent overlap */
            memset(myds_metal_buffer_data(q_got), 0x5a, 64u * 512u * sizeof(float));
            memset(myds_metal_buffer_data(ids_got), 0xa5, 512u * sizeof(int32_t));
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "1", 1);
            myds_v41_gpu_decode_topk_fused_counters(NULL, true);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            if (!myds_metal_begin_batch()) return 1;
            if (!myds_metal_dense_q4_v41_qb_slice(q_got, qb_w, qb_in, 0u, 32u * 512u, false)) return 1;
            if (!myds_v41_gpu_decode_topk_hist_only(s, scores_buf, NULL, nc, nc, false)) return 1;
            const myds_metal_buffer_t *h_deps[] = {
                scores_buf, q_got,
                myds_v41_gpu_scratch_decode_topk_hist(s),
                myds_v41_gpu_scratch_decode_topk_meta(s)
            };
            myds_metal_barrier_buffers(h_deps, 4u);
            myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_got, true);
            myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
            if (!myds_v41_gpu_topk_fused_decode(s, ids_got, scores_buf, nc, 512u, false, 0u)) return 1;
            {
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
            }
            myds_v41_gpu_attn_kv_reuse_reset(s);
            if (!myds_metal_commit_batch()) return 1;

            myds_v41_gpu_decode_topk_fused_counters_t cnt;
            myds_v41_gpu_decode_topk_fused_counters(&cnt, false);
            if (cnt.root_encoded != 1u || cnt.l20_encoded != 0u || cnt.failures != 0u) {
                fprintf(stderr, "FAIL: unexpected Root counters req=%llu root=%llu l20=%llu fail=%llu\n",
                        (unsigned long long)cnt.requests, (unsigned long long)cnt.root_encoded,
                        (unsigned long long)cnt.l20_encoded, (unsigned long long)cnt.failures);
                return 1;
            }
            char lbl[128];
            snprintf(lbl, sizeof(lbl), "ConcRoot nc=%u pat=%s", nc, pattern_name(pat));
            if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                  (const int32_t *)myds_metal_buffer_data(ids_got), 512u, sc, lbl) != 0) return 1;
            if (memcmp(myds_metal_buffer_data(q_ref), myds_metal_buffer_data(q_got), 64u * 512u * sizeof(float)) != 0) {
                fprintf(stderr, "FAIL [%s]: overlapped Q_B + Q RoPE mismatch!\n", lbl);
                return 1;
            }
            if (verify_scratch_zeroed(s, lbl) != 0) return 1;

            /* 4b. L20 concurrent overlap (nc > 16384) */
            if (nc > 16384u) {
                memset(myds_metal_buffer_data(q_got), 0x5a, 64u * 512u * sizeof(float));
                memset(myds_metal_buffer_data(ids_got), 0xa5, 512u * sizeof(int32_t));
                memset(myds_metal_buffer_data(bsel_got), 0xa5, 2048u * sizeof(int32_t));
                myds_v41_gpu_decode_topk_fused_counters(NULL, true);
                myds_metal_set_auto_barrier(0);
                myds_metal_set_concurrent(1);
                if (!myds_metal_begin_batch()) return 1;
                if (!myds_metal_dense_q4_v41_qb_slice(q_got, qb_w, qb_in, 0u, 32u * 512u, false)) return 1;
                if (!myds_v41_gpu_decode_topk_hist_only(s, scores_buf, blk_got, nc, nc, true)) return 1;
                const myds_metal_buffer_t *h2_deps[] = {
                    scores_buf, blk_got, q_got,
                    myds_v41_gpu_scratch_decode_topk_hist(s),
                    myds_v41_gpu_scratch_decode_topk_meta(s)
                };
                myds_metal_barrier_buffers(h2_deps, 5u);
                myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_got, true);
                myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                if (!myds_v41_gpu_topk_fused_decode(s, bsel_got, blk_got, n_blocks, 2048u, true, nc)) return 1;
                const myds_metal_buffer_t *blk_q_deps[] = {
                    bsel_got,
                    myds_v41_gpu_scratch_compact_blocks(s),
                    myds_v41_gpu_scratch_l20_nan_flags(s),
                    q_got
                };
                myds_metal_barrier_buffers(blk_q_deps, 4u);
                myds_v41_gpu_attn_kv_reuse_rebind_topk_qb(s, qb_w, qb_in, pos);
                if (!myds_v41_gpu_topk_l20_fused_self_compact_decode(ids_got, scores_buf, nc, s)) return 1;
                {
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
                }
                myds_v41_gpu_attn_kv_reuse_reset(s);
                if (!myds_metal_commit_batch()) return 1;

                myds_v41_gpu_decode_topk_fused_counters(&cnt, false);
                if (cnt.root_encoded != 0u || cnt.l20_encoded != 1u || cnt.failures != 0u) {
                    fprintf(stderr, "FAIL: unexpected L20 counters req=%llu root=%llu l20=%llu fail=%llu\n",
                            (unsigned long long)cnt.requests, (unsigned long long)cnt.root_encoded,
                            (unsigned long long)cnt.l20_encoded, (unsigned long long)cnt.failures);
                    return 1;
                }
                snprintf(lbl, sizeof(lbl), "ConcL20 nc=%u pat=%s", nc, pattern_name(pat));
                if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(bsel_ref),
                                      (const int32_t *)myds_metal_buffer_data(bsel_got),
                                      2048u, (const float *)myds_metal_buffer_data(blk_ref), lbl) != 0) return 1;
                if (compare_i32_exact((const int32_t *)myds_metal_buffer_data(ids_ref),
                                      (const int32_t *)myds_metal_buffer_data(ids_got), 512u, sc, lbl) != 0) return 1;
                if (memcmp(myds_metal_buffer_data(q_ref), myds_metal_buffer_data(q_got), 64u * 512u * sizeof(float)) != 0) {
                    fprintf(stderr, "FAIL [%s]: L20 overlapped Q_B + Q RoPE mismatch!\n", lbl);
                    return 1;
                }
                if (verify_scratch_zeroed(s, lbl) != 0) return 1;
            }
        }
    }
    myds_metal_set_auto_barrier(1);
    myds_metal_set_concurrent(0);
    printf("  PASS: concurrent=1 auto_barrier=0 + overlap_topk_qb bit-exact on IDs, blocks & Q + counters verified.\n");
    return 0;
}

/* [5/5] M4 GPU Benchmark: Arm C [D-I1] vs Wave-1 [D-T2] vs Wave-3 [D-I2] */
static void run_bench(myds_v41_gpu_scratch_t *s,
                      const myds_metal_buffer_t *q,
                      const myds_metal_buffer_t *w,
                      const myds_metal_buffer_t *k_fp4,
                      const myds_metal_buffer_t *k_bf16,
                      myds_metal_buffer_t *scores_buf,
                      myds_metal_buffer_t *blk_buf,
                      myds_metal_buffer_t *bsel_buf,
                      myds_metal_buffer_t *ids_buf,
                      myds_metal_buffer_t *q_buf,
                      const myds_metal_buffer_t *qb_w,
                      const myds_metal_buffer_t *qb_in) {
    printf("\n[5/5] M4 GPU Benchmark (--bench): Arm C [D-I1] vs Wave-1 [D-T2] vs Wave-3 [D-I2] (FP4)\n");
    setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);
    const uint32_t ncs[] = {16384u, 32768u, 65536u, 131072u, 262144u};
    const int warmup = 4, iters = 25;

    /* 0. Stage breakdown at 65536 & 262144: Score-only vs TopK-only */
    printf("  -- Stage Breakdown (FP4, batched GPU CB) --\n");
    const uint32_t b_ncs[] = {32768u, 65536u, 262144u};
    for (size_t bi = 0; bi < sizeof(b_ncs) / sizeof(b_ncs[0]); bi++) {
        const uint32_t nc = b_ncs[bi];
        double sc_armc = 0, sc_v2_root = 0, sc_v2_l20 = 0;
        myds_metal_set_auto_barrier(0);
        myds_metal_set_concurrent(1);
        for (int arm = 0; arm < 3; arm++) {
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", (arm > 0) ? "1" : "0", 1);
            for (int r = 0; r < warmup + 1; r++) {
                const int n_rep = (r == warmup) ? iters : 1;
                const double t0 = now_us();
                if (!myds_metal_begin_batch()) return;
                for (int it = 0; it < n_rep; it++) {
                    if (arm == 0) {
                        if (!myds_v41_gpu_indexscores_head_slots_mode(scores_buf, q, w, k_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u)) return;
                        myds_metal_barrier_buffer(scores_buf);
                    } else if (arm == 1) {
                        if (!myds_v41_gpu_indexscores_fused_decode_mode(s, scores_buf, NULL, q, w, k_fp4, NULL, nc, MYDS_INDEXER_CACHE_FP4, 0u, nc, false)) return;
                        const myds_metal_buffer_t *d[] = {scores_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                        myds_metal_barrier_buffers(d, 3u);
                        if (!myds_v41_gpu_topk_fused_decode(s, ids_buf, scores_buf, nc, 512u, false, 0u)) return;
                        const myds_metal_buffer_t *e[] = {ids_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                        myds_metal_barrier_buffers(e, 3u);
                    } else {
                        /* Score-only v2_l20 without topk (clear hist each iter only via dirty flag -> no, run 100% score-only on dirty-safe: just clear? Better: measure score_one_mma_v2 directly by calling it with auto_barrier=1, hist accumulates harmlessly in bench then clear once!) */
                        if (!myds_v41_gpu_indexscores_fused_decode_mode(s, scores_buf, blk_buf, q, w, k_fp4, NULL, nc, MYDS_INDEXER_CACHE_FP4, 0u, nc, true)) return;
                        myds_metal_barrier_buffer(scores_buf);
                    }
                }
                if (!myds_metal_commit_batch()) return;
                if (r == warmup) {
                    double avg = (now_us() - t0) / (double)iters;
                    if (arm == 0) sc_armc = avg;
                    else if (arm == 1) sc_v2_root = avg;
                    else sc_v2_l20 = avg;
                }
            }
            /* Clean dirty flag after arm==2 */
            if (arm == 2) {
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_topk_fused_decode(s, bsel_buf, blk_buf, (nc + 7u) / 8u, 2048u, true, nc) ||
                    !myds_metal_commit_batch()) return;
            }
        }
        printf("    nc=%6u: score_ArmC=%7.1f us  (score_v2+2passTop512)=%7.1f us  score_v2_l20_only(+clear)=%7.1f us\n",
               nc, sc_armc, sc_v2_root, sc_v2_l20);
    }

    /* A. Root L2/8/14 Pure: Score FP4 + Top-512 */
    printf("  -- Root L2/8/14 Pure: Score FP4 + Top-512 (batched GPU CB) --\n");
    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        double us[3] = {0.0, 0.0, 0.0};
        for (int arm = 0; arm < 3; arm++) {
            setenv("MYDS_DECODE_TOPK_FAST", (arm >= 1) ? "1" : "0", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", (arm == 2) ? "1" : "0", 1);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            for (int r = 0; r < warmup + 1; r++) {
                const int n_rep = (r == warmup) ? iters : 1;
                const double t0 = now_us();
                if (!myds_metal_begin_batch()) return;
                for (int it = 0; it < n_rep; it++) {
                    if (arm < 2) {
                        if (!myds_v41_gpu_indexscores_head_slots_mode(scores_buf, q, w, k_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u)) return;
                        myds_metal_barrier_buffer(scores_buf);
                        if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                        myds_metal_barrier_buffer(ids_buf);
                    } else {
                        if (!myds_v41_gpu_indexscores_fused_decode_mode(s, scores_buf, NULL, q, w, k_fp4, NULL, nc, MYDS_INDEXER_CACHE_FP4, 0u, nc, false)) return;
                        const myds_metal_buffer_t *d[] = {scores_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                        myds_metal_barrier_buffers(d, 3u);
                        if (!myds_v41_gpu_topk_fused_decode(s, ids_buf, scores_buf, nc, 512u, false, 0u)) return;
                        const myds_metal_buffer_t *e[] = {ids_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s), myds_v41_gpu_scratch_decode_topk_subhist(s)};
                        myds_metal_barrier_buffers(e, 4u);
                    }
                }
                if (!myds_metal_commit_batch()) return;
                if (r == warmup) us[arm] = (now_us() - t0) / (double)iters;
            }
        }
        printf("    nc=%6u: ArmC=%7.1f us  DT2=%7.1f us  DI2=%7.1f us  vsArmC=%+.1f%% (%.2fx)  vsDT2=%+.1f%% (%.2fx)\n",
               nc, us[0], us[1], us[2],
               (us[2] - us[0]) / us[0] * 100.0, us[0] / us[2],
               (us[2] - us[1]) / us[1] * 100.0, us[1] / us[2]);
    }

    /* B. Root L2/8/14 Full: Score FP4 + Top-512 + trailing 32 Q_B + 64 Q RoPE */
    printf("  -- Root L2/8/14 Full: Score FP4 + Top-512 + 32 Q_B + 64 Q RoPE --\n");
    for (size_t ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t pos = nc - 1u;
        double us[3] = {0.0, 0.0, 0.0};
        for (int arm = 0; arm < 3; arm++) {
            setenv("MYDS_DECODE_TOPK_FAST", (arm >= 1) ? "1" : "0", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", (arm == 2) ? "1" : "0", 1);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            for (int r = 0; r < warmup + 1; r++) {
                const int n_rep = (r == warmup) ? iters : 1;
                const double t0 = now_us();
                if (!myds_metal_begin_batch()) return;
                for (int it = 0; it < n_rep; it++) {
                    if (arm < 2) {
                        if (!myds_v41_gpu_indexscores_head_slots_mode(scores_buf, q, w, k_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u)) return;
                        myds_metal_barrier_buffer(scores_buf);
                        myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                        myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                        if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                    } else {
                        if (!myds_v41_gpu_indexscores_fused_decode_mode(s, scores_buf, NULL, q, w, k_fp4, NULL, nc, MYDS_INDEXER_CACHE_FP4, 0u, nc, false)) return;
                        const myds_metal_buffer_t *d[] = {scores_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                        myds_metal_barrier_buffers(d, 3u);
                        myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                        myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                        if (!myds_v41_gpu_topk_fused_decode(s, ids_buf, scores_buf, nc, 512u, false, 0u)) return;
                    }
                    const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                    const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                    if (qb_done < 64u) {
                        if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in, qb_done * 512u, (64u - qb_done) * 512u, false)) return;
                    }
                    const myds_metal_buffer_t *fin[] = {ids_buf, q_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                    myds_metal_barrier_buffers(fin, (arm == 2) ? 4u : 2u);
                    if (rope_done < 64u) {
                        if (!myds_v41_gpu_rope_heads_slice(s, q_buf, rope_done, 64u - rope_done, pos, true)) return;
                        myds_metal_barrier_buffer(q_buf);
                    }
                    myds_v41_gpu_attn_kv_reuse_reset(s);
                }
                if (!myds_metal_commit_batch()) return;
                if (r == warmup) us[arm] = (now_us() - t0) / (double)iters;
            }
        }
        printf("    nc=%6u: ArmC=%7.1f us  DT2=%7.1f us  DI2=%7.1f us  vsArmC=%+.1f%% (%.2fx)  vsDT2=%+.1f%% (%.2fx)\n",
               nc, us[0], us[1], us[2],
               (us[2] - us[0]) / us[0] * 100.0, us[0] / us[2],
               (us[2] - us[1]) / us[1] * 100.0, us[1] / us[2]);
    }

    /* C. L20 Pure: Score FP4 + block_max + Top-2048 + compact_blocks + Top-512 */
    printf("  -- L20 Pure: Score FP4 + block_max + Top-2048 + compact_blocks + Top-512 --\n");
    for (size_t ni = 1; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t n_blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u;
        double us[3] = {0.0, 0.0, 0.0};
        for (int arm = 0; arm < 3; arm++) {
            setenv("MYDS_DECODE_TOPK_FAST", (arm >= 1) ? "1" : "0", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", (arm == 2) ? "1" : "0", 1);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            for (int r = 0; r < warmup + 1; r++) {
                const int n_rep = (r == warmup) ? iters : 1;
                const double t0 = now_us();
                if (!myds_metal_begin_batch()) return;
                for (int it = 0; it < n_rep; it++) {
                    if (arm == 0) {
                        if (!myds_v41_gpu_indexscores_head_slots_mode(scores_buf, q, w, k_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u)) return;
                        myds_metal_barrier_buffer(scores_buf);
                        if (!myds_v41_gpu_candidates(blk_buf, scores_buf, nc, pos, 1u)) return;
                        myds_metal_barrier_buffer(blk_buf);
                        if (!myds_v41_gpu_topk(bsel_buf, blk_buf, n_blocks, 2048u, s)) return;
                        if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_buf, n_blocks, 2048u, nc)) return;
                        if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                        myds_metal_barrier_buffer(ids_buf);
                    } else if (arm == 1) {
                        if (!myds_v41_gpu_indexscores_head_slots_mode(scores_buf, q, w, k_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u)) return;
                        myds_metal_barrier_buffer(scores_buf);
                        if (!myds_v41_gpu_l20_decode_candidates_nan(blk_buf, scores_buf, nc, pos, 1u, s)) return;
                        myds_metal_barrier_buffer(blk_buf);
                        if (!myds_v41_gpu_topk(bsel_buf, blk_buf, n_blocks, 2048u, s)) return;
                        if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_buf, n_blocks, 2048u, nc)) return;
                        if (!myds_v41_gpu_topk_l20_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                        myds_metal_barrier_buffer(ids_buf);
                    } else {
                        if (!myds_v41_gpu_indexscores_fused_decode_mode(s, scores_buf, blk_buf, q, w, k_fp4, NULL, nc, MYDS_INDEXER_CACHE_FP4, 0u, nc, true)) return;
                        const myds_metal_buffer_t *d[] = {scores_buf, blk_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                        myds_metal_barrier_buffers(d, 4u);
                        if (!myds_v41_gpu_topk_fused_decode(s, bsel_buf, blk_buf, n_blocks, 2048u, true, nc)) return;
                        const myds_metal_buffer_t *e[] = {bsel_buf, myds_v41_gpu_scratch_compact_blocks(s), myds_v41_gpu_scratch_l20_nan_flags(s), myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                        myds_metal_barrier_buffers(e, 5u);
                        if (!myds_v41_gpu_topk_l20_fused_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                        myds_metal_barrier_buffer(ids_buf);
                    }
                }
                if (!myds_metal_commit_batch()) return;
                if (r == warmup) us[arm] = (now_us() - t0) / (double)iters;
            }
        }
        printf("    nc=%6u: ArmC=%7.1f us  DT2=%7.1f us  DI2=%7.1f us  vsArmC=%+.1f%% (%.2fx)  vsDT2=%+.1f%% (%.2fx)\n",
               nc, us[0], us[1], us[2],
               (us[2] - us[0]) / us[0] * 100.0, us[0] / us[2],
               (us[2] - us[1]) / us[1] * 100.0, us[1] / us[2]);
    }

    /* D. L20 Full: Score FP4 + block_max + Top-2048 + compact + Top-512 + 32 Q_B + 64 Q RoPE */
    printf("  -- L20 Full: Score FP4 + Top-2048 + Top-512 + 32 Q_B + 64 Q RoPE --\n");
    for (size_t ni = 1; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
        const uint32_t nc = ncs[ni];
        const uint32_t n_blocks = (nc + 7u) / 8u;
        const uint32_t pos = nc - 1u;
        double us[3] = {0.0, 0.0, 0.0};
        for (int arm = 0; arm < 3; arm++) {
            setenv("MYDS_DECODE_TOPK_FAST", (arm >= 1) ? "1" : "0", 1);
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", (arm == 2) ? "1" : "0", 1);
            myds_metal_set_auto_barrier(0);
            myds_metal_set_concurrent(1);
            for (int r = 0; r < warmup + 1; r++) {
                const int n_rep = (r == warmup) ? iters : 1;
                const double t0 = now_us();
                if (!myds_metal_begin_batch()) return;
                for (int it = 0; it < n_rep; it++) {
                    if (arm == 0) {
                        if (!myds_v41_gpu_indexscores_head_slots_mode(scores_buf, q, w, k_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u)) return;
                        myds_metal_barrier_buffer(scores_buf);
                        if (!myds_v41_gpu_candidates(blk_buf, scores_buf, nc, pos, 1u)) return;
                        myds_metal_barrier_buffer(blk_buf);
                        if (!myds_v41_gpu_topk(bsel_buf, blk_buf, n_blocks, 2048u, s)) return;
                        if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_buf, n_blocks, 2048u, nc)) return;
                        myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                        if (!myds_v41_gpu_topk(ids_buf, scores_buf, nc, 512u, s)) return;
                        if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in, 32u * 512u, 32u * 512u, false)) return;
                        const myds_metal_buffer_t *fin[] = {ids_buf, q_buf};
                        myds_metal_barrier_buffers(fin, 2u);
                        if (!myds_v41_gpu_rope_heads_slice(s, q_buf, 0u, 64u, pos, true)) return;
                        myds_metal_barrier_buffer(q_buf);
                    } else if (arm == 1) {
                        if (!myds_v41_gpu_indexscores_head_slots_mode(scores_buf, q, w, k_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u)) return;
                        myds_metal_barrier_buffer(scores_buf);
                        myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                        myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                        if (!myds_v41_gpu_l20_decode_candidates_nan(blk_buf, scores_buf, nc, pos, 1u, s)) return;
                        myds_metal_barrier_buffer(blk_buf);
                        if (!myds_v41_gpu_topk(bsel_buf, blk_buf, n_blocks, 2048u, s)) return;
                        const myds_metal_buffer_t *bq[] = {bsel_buf, q_buf};
                        myds_metal_barrier_buffers(bq, 2u);
                        if (!myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_buf, n_blocks, 2048u, nc)) return;
                        myds_v41_gpu_attn_kv_reuse_rebind_topk_qb(s, qb_w, qb_in, pos);
                        if (!myds_v41_gpu_topk_l20_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                        const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                        const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                        if (qb_done < 64u) {
                            if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in, qb_done * 512u, (64u - qb_done) * 512u, false)) return;
                        }
                        const myds_metal_buffer_t *fin[] = {ids_buf, q_buf};
                        myds_metal_barrier_buffers(fin, 2u);
                        if (rope_done < 64u) {
                            if (!myds_v41_gpu_rope_heads_slice(s, q_buf, rope_done, 64u - rope_done, pos, true)) return;
                            myds_metal_barrier_buffer(q_buf);
                        }
                    } else {
                        if (!myds_v41_gpu_indexscores_fused_decode_mode(s, scores_buf, blk_buf, q, w, k_fp4, NULL, nc, MYDS_INDEXER_CACHE_FP4, 0u, nc, true)) return;
                        const myds_metal_buffer_t *d[] = {scores_buf, blk_buf, myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s)};
                        myds_metal_barrier_buffers(d, 4u);
                        myds_v41_gpu_attn_kv_reuse_defer_q_rope(s, q_buf, true);
                        myds_v41_gpu_attn_kv_reuse_bind_topk_qb(s, qb_w, qb_in, pos, 32u);
                        if (!myds_v41_gpu_topk_fused_decode(s, bsel_buf, blk_buf, n_blocks, 2048u, true, nc)) return;
                        const myds_metal_buffer_t *bq[] = {bsel_buf, myds_v41_gpu_scratch_compact_blocks(s), myds_v41_gpu_scratch_l20_nan_flags(s), myds_v41_gpu_scratch_decode_topk_hist(s), myds_v41_gpu_scratch_decode_topk_meta(s), q_buf};
                        myds_metal_barrier_buffers(bq, 6u);
                        myds_v41_gpu_attn_kv_reuse_rebind_topk_qb(s, qb_w, qb_in, pos);
                        if (!myds_v41_gpu_topk_l20_fused_self_compact_decode(ids_buf, scores_buf, nc, s)) return;
                        const uint32_t qb_done = myds_v41_gpu_attn_kv_reuse_qb_done(s);
                        const uint32_t rope_done = myds_v41_gpu_attn_kv_reuse_rope_done(s);
                        if (qb_done < 64u) {
                            if (!myds_metal_dense_q4_v41_qb_slice(q_buf, qb_w, qb_in, qb_done * 512u, (64u - qb_done) * 512u, false)) return;
                        }
                        const myds_metal_buffer_t *fin[] = {ids_buf, q_buf};
                        myds_metal_barrier_buffers(fin, 2u);
                        if (rope_done < 64u) {
                            if (!myds_v41_gpu_rope_heads_slice(s, q_buf, rope_done, 64u - rope_done, pos, true)) return;
                            myds_metal_barrier_buffer(q_buf);
                        }
                    }
                    myds_v41_gpu_attn_kv_reuse_reset(s);
                }
                if (!myds_metal_commit_batch()) return;
                if (r == warmup) us[arm] = (now_us() - t0) / (double)iters;
            }
        }
        printf("    nc=%6u: ArmC=%7.1f us  DT2=%7.1f us  DI2=%7.1f us  vsArmC=%+.1f%% (%.2fx)  vsDT2=%+.1f%% (%.2fx)\n",
               nc, us[0], us[1], us[2],
               (us[2] - us[0]) / us[0] * 100.0, us[0] / us[2],
               (us[2] - us[1]) / us[1] * 100.0, us[1] / us[2]);
    }

    /* E. L24..36 Compact Score (16,384 keys): Arm C score_compact_mma vs [D-I2] score_compact_mma_v2 */
    printf("  -- L24..36 Compact Score (2048 blocks = 16,384 keys): Arm C vs [D-I2] v2 --\n");
    const uint32_t nc_c = 262144u;
    int32_t *bp = (int32_t *)myds_metal_buffer_data(bsel_buf);
    for (uint32_t i = 0; i < 2048u; i++) bp[i] = (int32_t)(i * 15u);
    myds_metal_set_auto_barrier(1);
    myds_metal_set_concurrent(0);
    if (!myds_metal_begin_batch() ||
        !myds_v41_gpu_indexer_prepare_compact_blocks(s, bsel_buf, nc_c / 8u, 2048u, nc_c) ||
        !myds_metal_commit_batch()) return;
    for (int is_fp4 = 1; is_fp4 >= 0; is_fp4--) {
        const uint32_t mode = is_fp4 ? MYDS_INDEXER_CACHE_FP4 : MYDS_INDEXER_CACHE_BF16;
        const myds_metal_buffer_t *kbuf = is_fp4 ? k_fp4 : k_bf16;
        double us[2] = {0.0, 0.0};
        for (int arm = 0; arm < 2; arm++) {
            setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", arm ? "1" : "0", 1);
            for (int r = 0; r < warmup + 1; r++) {
                const int n_rep = (r == warmup) ? 60 : 1;
                const double t0 = now_us();
                if (!myds_metal_begin_batch()) return;
                for (int it = 0; it < n_rep; it++) {
                    if (!myds_v41_gpu_indexscores_compact_decode_mode(s, q, w, kbuf, NULL, nc_c, mode, 0u)) return;
                }
                if (!myds_metal_commit_batch()) return;
                if (r == warmup) us[arm] = (now_us() - t0) / 60.0;
            }
        }
        printf("    %s: ArmC=%6.1f us  DI2_v2=%6.1f us  delta=%+.1f%% (%.2fx)\n",
               is_fp4 ? "FP4 " : "BF16", us[0], us[1], (us[1] - us[0]) / us[0] * 100.0, us[0] / us[1]);
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
        setenv("MYDS_INDEXER_HEAD_SLOTS", "1", 1);
        setenv("MYDS_INDEXER_SCORE_ROWS4", "auto", 1);
        setenv("MYDS_INDEXER_DECODE_COMPACT", "1", 1);
        setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
        setenv("MYDS_INDEXER_KEY_FP4", "1", 1);
        setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
        setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);

        if (!myds_metal_init("bin")) {
            fprintf(stderr, "FAIL: myds_metal_init failed\n");
            return 1;
        }

        /* Verify fail-closed scratch creation on invalid MYDS_INDEXER_DECODE_TOPK_FUSED */
        setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "2", 1);
        if (myds_v41_gpu_scratch_create(262144u) != NULL) {
            fprintf(stderr, "FAIL: scratch_create must fail closed on MYDS_INDEXER_DECODE_TOPK_FUSED=2\n");
            myds_metal_cleanup();
            return 1;
        }
        setenv("MYDS_INDEXER_DECODE_TOPK_FUSED", "0", 1);

        myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(262144u);
        if (!s) {
            fprintf(stderr, "FAIL: myds_v41_gpu_scratch_create failed\n");
            myds_metal_cleanup();
            return 1;
        }

        const uint32_t max_nc = 262144u;
        const uint32_t max_nb = max_nc / 8u;
        myds_metal_buffer_t *q = myds_metal_buffer_alloc(32u * 128u * sizeof(float));
        myds_metal_buffer_t *w = myds_metal_buffer_alloc(32u * sizeof(float));
        myds_metal_buffer_t *k_fp4 = myds_metal_buffer_alloc((size_t)max_nc * 68u);
        myds_metal_buffer_t *k_bf16 = myds_metal_buffer_alloc((size_t)max_nc * 256u);
        myds_metal_buffer_t *k_f32 = myds_metal_buffer_alloc(16384u * 512u);
        myds_metal_buffer_t *mask = myds_metal_buffer_alloc((size_t)max_nb * sizeof(float));
        myds_metal_buffer_t *sc_ref = myds_metal_buffer_alloc((size_t)max_nc * sizeof(float));
        myds_metal_buffer_t *sc_got = myds_metal_buffer_alloc((size_t)max_nc * sizeof(float));
        myds_metal_buffer_t *blk_ref = myds_metal_buffer_alloc((size_t)max_nb * sizeof(float));
        myds_metal_buffer_t *blk_got = myds_metal_buffer_alloc((size_t)max_nb * sizeof(float));
        myds_metal_buffer_t *bsel_ref = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *bsel_got = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *ids_ref = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *ids_got = myds_metal_buffer_alloc(2048u * sizeof(int32_t));
        myds_metal_buffer_t *q_ref = myds_metal_buffer_alloc(64u * 512u * sizeof(float));
        myds_metal_buffer_t *q_got = myds_metal_buffer_alloc(64u * 512u * sizeof(float));
        myds_metal_buffer_t *qb_w = myds_metal_buffer_alloc(32768u * 5u * 144u);
        myds_metal_buffer_t *qb_in = myds_metal_buffer_alloc(1280u * sizeof(float));

        if (!q || !w || !k_fp4 || !k_bf16 || !k_f32 || !mask || !sc_ref || !sc_got ||
            !blk_ref || !blk_got || !bsel_ref || !bsel_got || !ids_ref || !ids_got ||
            !q_ref || !q_got || !qb_w || !qb_in) {
            fprintf(stderr, "FAIL: buffer allocation failed\n");
            return 1;
        }

        float *qp = (float *)myds_metal_buffer_data(q);
        float *wp = (float *)myds_metal_buffer_data(w);
        float *mp = (float *)myds_metal_buffer_data(mask);
        for (uint32_t i = 0; i < 32u * 128u; i++) qp[i] = rand_uniform(-2.0f, 2.0f);
        for (uint32_t i = 0; i < 32u; i++) wp[i] = rand_uniform(0.1f, 1.5f);
        for (uint32_t b = 0; b < max_nb; b++) mp[b] = (b % 7u == 0u) ? 1.0f : 0.0f;
        fill_k_caches((uint8_t *)myds_metal_buffer_data(k_fp4),
                      (uint16_t *)myds_metal_buffer_data(k_bf16),
                      (float *)myds_metal_buffer_data(k_f32),
                      16384u, 118u, 134u, true);
        fill_k_caches((uint8_t *)myds_metal_buffer_data(k_fp4) + 16384u * 68u,
                      (uint16_t *)myds_metal_buffer_data(k_bf16) + 16384u * 128u,
                      NULL, max_nc - 16384u, 120u, 132u, false);
        fill_qb_weights_and_input(qb_w, qb_in);

        int rc = 0;
        if (!rc) rc = test_scorer_0ulp(s, q, w, k_fp4, k_bf16, k_f32, mask,
                                       sc_ref, sc_got, blk_ref, blk_got,
                                       bsel_ref, bsel_got, ids_ref, ids_got);
        if (!rc) rc = test_torture_suite(s, sc_ref, blk_ref, blk_got,
                                         bsel_ref, bsel_got, ids_ref, ids_got);
        if (!rc) rc = test_concurrent_overlap(s, sc_ref, blk_ref, blk_got,
                                              bsel_ref, bsel_got, ids_ref, ids_got,
                                              q_ref, q_got, qb_w, qb_in);
        if (!rc && do_bench) {
            run_bench(s, q, w, k_fp4, k_bf16, sc_got, blk_got, bsel_got, ids_got, q_got, qb_w, qb_in);
        }

        myds_metal_buffer_free(q);
        myds_metal_buffer_free(w);
        myds_metal_buffer_free(k_fp4);
        myds_metal_buffer_free(k_bf16);
        myds_metal_buffer_free(k_f32);
        myds_metal_buffer_free(mask);
        myds_metal_buffer_free(sc_ref);
        myds_metal_buffer_free(sc_got);
        myds_metal_buffer_free(blk_ref);
        myds_metal_buffer_free(blk_got);
        myds_metal_buffer_free(bsel_ref);
        myds_metal_buffer_free(bsel_got);
        myds_metal_buffer_free(ids_ref);
        myds_metal_buffer_free(ids_got);
        myds_metal_buffer_free(q_ref);
        myds_metal_buffer_free(q_got);
        myds_metal_buffer_free(qb_w);
        myds_metal_buffer_free(qb_in);
        myds_v41_gpu_scratch_free(s);
        myds_metal_cleanup();
        unsetenv("MYDS_INDEXER_DECODE_TOPK_FUSED");

        if (rc == 0) {
            printf("\n[ALL PASS] test_decode_topk_fused: strict 0-ULP verified!\n");
        }
        return rc;
    }
}
