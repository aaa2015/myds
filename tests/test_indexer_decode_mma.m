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

static uint32_t g_rng = 0x13579bdfu;
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

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
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

static void fill_fp4_k_cache(uint8_t *fp4_buf, uint16_t *bf16_buf, uint32_t n_rows,
                             uint8_t exp_lo, uint8_t exp_hi, bool inject_extreme) {
    for (uint32_t r = 0; r < n_rows; r++) {
        uint8_t *row = fp4_buf + (size_t)r * 68u;
        uint8_t exps[4];
        for (uint32_t b = 0; b < 4u; b++) {
            uint32_t span = (uint32_t)(exp_hi - exp_lo + 1u);
            exps[b] = (uint8_t)(exp_lo + (xorshift32() % span));
        }
        if (inject_extreme && (r % 17u == 0u)) {
            static const uint8_t extremes[8] = {0u, 1u, 60u, 113u, 114u, 140u, 141u, 155u};
            exps[0] = extremes[(r / 17u) % 8u];
            exps[2] = extremes[(r / 17u + 3u) % 8u];
        }
        for (uint32_t j = 0; j < 64u; j++) {
            uint32_t c0 = xorshift32() & 0xfu;
            uint32_t c1 = xorshift32() & 0xfu;
            uint8_t eb = exps[j >> 4u];
            if (eb == 0u || (eb == 1u && (c0 & 7u) == 1u)) c0 = 0u;
            if (eb == 0u || (eb == 1u && (c1 & 7u) == 1u)) c1 = 0u;
            row[j] = (uint8_t)(c0 | (c1 << 4u));
            if (bf16_buf) {
                bf16_buf[(size_t)r * 128u + j * 2u + 0u] = f32_to_bf16_bits(fp4_dequant_cpu(c0, eb));
                bf16_buf[(size_t)r * 128u + j * 2u + 1u] = f32_to_bf16_bits(fp4_dequant_cpu(c1, eb));
            }
        }
        memcpy(row + 64u, exps, 4u);
    }
}

static uint32_t topk_overlap_512(const float *ref, const float *got, uint32_t n) {
    typedef struct { float s; uint32_t idx; } pair_t;
    pair_t *a = (pair_t *)malloc((size_t)n * sizeof(pair_t));
    pair_t *b = (pair_t *)malloc((size_t)n * sizeof(pair_t));
    for (uint32_t i = 0; i < n; i++) { a[i] = (pair_t){ref[i], i}; b[i] = (pair_t){got[i], i}; }
    int (^cmp)(const void *, const void *) = ^(const void *p1, const void *p2) {
        const pair_t *x = (const pair_t *)p1, *y = (const pair_t *)p2;
        if (x->s != y->s) return (x->s > y->s) ? -1 : 1;
        return (x->idx < y->idx) ? -1 : 1;
    };
    qsort_b(a, n, sizeof(pair_t), cmp);
    qsort_b(b, n, sizeof(pair_t), cmp);
    uint32_t k = n < 512u ? n : 512u;
    uint8_t *seen = (uint8_t *)calloc(n, 1);
    for (uint32_t i = 0; i < k; i++) seen[a[i].idx] = 1;
    uint32_t hit = 0;
    for (uint32_t i = 0; i < k; i++) if (seen[b[i].idx]) hit++;
    free(seen); free(a); free(b);
    return hit;
}

static int test_knob(void) {
    printf("[1/3] Testing MYDS_INDEXER_DECODE_MMA fail-closed parsing...\n");
    unsetenv("MYDS_INDEXER_DECODE_MMA");
    if (myds_v41_gpu_indexer_decode_mma_enabled() != 0) {
        fprintf(stderr, "FAIL: default must be 0\n"); return 1;
    }
    setenv("MYDS_INDEXER_DECODE_MMA", "2", 1);
    if (myds_v41_gpu_indexer_decode_mma_enabled() != -1) {
        fprintf(stderr, "FAIL: invalid value '2' must return -1\n"); return 1;
    }
    setenv("MYDS_INDEXER_DECODE_MMA", "abc", 1);
    if (myds_v41_gpu_indexer_decode_mma_enabled() != -1) {
        fprintf(stderr, "FAIL: invalid value 'abc' must return -1\n"); return 1;
    }
    setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
    if (myds_v41_gpu_indexer_decode_mma_enabled() != 1) {
        fprintf(stderr, "FAIL: '1' must return 1\n"); return 1;
    }
    setenv("MYDS_INDEXER_DECODE_MMA", "0", 1);
    printf("  PASS: MYDS_INDEXER_DECODE_MMA defaults 0 and fails closed on invalid values.\n");
    return 0;
}

int main(void) {
    @autoreleasepool {
        setenv("MYDS_INDEXER_HEAD_SLOTS", "1", 1);
        setenv("MYDS_INDEXER_SCORE_ROWS4", "auto", 1);
        setenv("MYDS_INDEXER_DECODE_COMPACT", "1", 1);
        setenv("MYDS_INDEXER_KEY_FP4", "1", 1);

        if (test_knob() != 0) return 1;
        if (!myds_metal_init("bin")) {
            fprintf(stderr, "FAIL: myds_metal_init failed\n"); return 1;
        }
        myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(262144u);
        if (!s) {
            fprintf(stderr, "FAIL: myds_v41_gpu_scratch_create failed\n");
            myds_metal_cleanup(); return 1;
        }

        printf("[2/3] Testing [D-I1] score_one_mma & score_compact_mma (FP4 & BF16, mask & first_row)...\n");
        const uint32_t n_comp = 32768u;
        const uint32_t n_blocks = (n_comp + 7u) / 8u;
        myds_metal_buffer_t *q = myds_metal_buffer_alloc(32u * 128u * 4u);
        myds_metal_buffer_t *w = myds_metal_buffer_alloc(32u * 4u);
        myds_metal_buffer_t *k_fp4 = myds_metal_buffer_alloc((size_t)n_comp * 68u);
        myds_metal_buffer_t *k_bf16 = myds_metal_buffer_alloc((size_t)n_comp * 256u);
        myds_metal_buffer_t *mask = myds_metal_buffer_alloc((size_t)n_blocks * 4u);
        myds_metal_buffer_t *sc_ref = myds_metal_buffer_alloc((size_t)n_comp * 4u);
        myds_metal_buffer_t *sc_opt = myds_metal_buffer_alloc((size_t)n_comp * 4u);
        myds_metal_buffer_t *blk_sel = myds_metal_buffer_alloc(2048u * 4u);

        float *qp = (float *)myds_metal_buffer_data(q);
        float *wp = (float *)myds_metal_buffer_data(w);
        float *mp = (float *)myds_metal_buffer_data(mask);
        for (uint32_t i = 0; i < 32u * 128u; i++) qp[i] = rand_uniform(-2.0f, 2.0f);
        for (uint32_t i = 0; i < 32u; i++) wp[i] = rand_uniform(0.1f, 1.5f);
        for (uint32_t b = 0; b < n_blocks; b++) mp[b] = (b % 5u == 0u) ? 1.0f : 0.0f;
        fill_fp4_k_cache((uint8_t *)myds_metal_buffer_data(k_fp4),
                         (uint16_t *)myds_metal_buffer_data(k_bf16),
                         n_comp, 118u, 134u, true);

        for (int use_mask = 0; use_mask <= 1; use_mask++) {
            for (int is_fp4 = 1; is_fp4 >= 0; is_fp4--) {
                const uint32_t mode = is_fp4 ? MYDS_INDEXER_CACHE_FP4 : MYDS_INDEXER_CACHE_BF16;
                const myds_metal_buffer_t *kbuf = is_fp4 ? k_fp4 : k_bf16;
                const myds_metal_buffer_t *mbuf = use_mask ? mask : NULL;
                const uint32_t first_row = use_mask ? 256u : 128u;

                setenv("MYDS_INDEXER_DECODE_MMA", "0", 1);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_indexscores_head_slots_mode(sc_ref, q, w, kbuf, mbuf, n_comp, 32u, mode, first_row) ||
                    !myds_metal_commit_batch()) return 1;

                setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
                if (!myds_metal_begin_batch() ||
                    !myds_v41_gpu_indexscores_head_slots_mode(sc_opt, q, w, kbuf, mbuf, n_comp, 32u, mode, first_row) ||
                    !myds_metal_commit_batch()) return 1;

                const float *r = (const float *)myds_metal_buffer_data(sc_ref);
                const float *o = (const float *)myds_metal_buffer_data(sc_opt);
                double max_rel = 0.0;
                uint64_t nan_cnt = 0;
                for (uint32_t i = 0; i < n_comp; i++) {
                    bool masked = (i < first_row) || (use_mask && mp[i >> 3u] != 0.0f);
                    if (masked) {
                        if (!isinf(o[i]) || o[i] > 0.0f) {
                            fprintf(stderr, "FAIL: mask/first_row mismatch at i=%u\n", i); return 1;
                        }
                        continue;
                    }
                    if (isnan(o[i]) || isinf(o[i])) { nan_cnt++; continue; }
                    double denom = fabs((double)r[i]) > 1e-3 ? fabs((double)r[i]) : 1e-3;
                    double rel = fabs((double)o[i] - (double)r[i]) / denom;
                    if (rel > max_rel) max_rel = rel;
                }
                uint32_t ov = topk_overlap_512(r, o, n_comp);
                printf("  score_one %s mask=%d (32K): nan=%llu max_rel=%.2e top512_overlap=%u/512\n",
                       is_fp4 ? "FP4 " : "BF16", use_mask, (unsigned long long)nan_cnt, max_rel, ov);
                if (nan_cnt > 0 || max_rel > 1e-3 || ov < 511u) {
                    fprintf(stderr, "FAIL: score_one accuracy gate!\n"); return 1;
                }
            }
        }

        /* Compact decode MMA (FP4 & BF16) */
        int32_t *bsel = (int32_t *)myds_metal_buffer_data(blk_sel);
        for (uint32_t i = 0; i < 2048u; i++) bsel[i] = (int32_t)(i * 2u);
        myds_metal_begin_batch();
        myds_v41_gpu_indexer_prepare_compact_blocks(s, blk_sel, n_blocks, 2048u, n_comp);
        myds_metal_commit_batch();

        float *c_ref = (float *)malloc(16384u * sizeof(float));
        for (int is_fp4 = 1; is_fp4 >= 0; is_fp4--) {
            const uint32_t mode = is_fp4 ? MYDS_INDEXER_CACHE_FP4 : MYDS_INDEXER_CACHE_BF16;
            const myds_metal_buffer_t *kbuf = is_fp4 ? k_fp4 : k_bf16;
            setenv("MYDS_INDEXER_DECODE_MMA", "0", 1);
            myds_metal_begin_batch();
            if (!myds_v41_gpu_indexscores_compact_decode_mode(s, q, w, kbuf, NULL, n_comp, mode, 0u)) return 1;
            myds_metal_commit_batch();
            memcpy(c_ref, myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_scores(s)), 16384u * sizeof(float));

            setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
            myds_metal_begin_batch();
            if (!myds_v41_gpu_indexscores_compact_decode_mode(s, q, w, kbuf, NULL, n_comp, mode, 0u)) return 1;
            myds_metal_commit_batch();
            const float *c_opt = (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)myds_v41_gpu_scratch_compact_scores(s));
            double max_rel = 0.0;
            for (uint32_t i = 0; i < 16384u; i++) {
                if (isnan(c_opt[i]) || isinf(c_opt[i])) { fprintf(stderr, "FAIL: compact NaN/Inf\n"); return 1; }
                double denom = fabs((double)c_ref[i]) > 1e-3 ? fabs((double)c_ref[i]) : 1e-3;
                double rel = fabs((double)c_opt[i] - (double)c_ref[i]) / denom;
                if (rel > max_rel) max_rel = rel;
            }
            uint32_t cov = topk_overlap_512(c_ref, c_opt, 16384u);
            printf("  compact_decode %s (16K): max_rel=%.2e top512_overlap=%u/512\n",
                   is_fp4 ? "FP4 " : "BF16", max_rel, cov);
            if (max_rel > 1e-3 || cov < 511u) {
                fprintf(stderr, "FAIL: compact_decode accuracy gate!\n"); return 1;
            }
        }
        free(c_ref);

        printf("[3/3] Microbenchmarking [D-I1] FP4 262K on Apple M4...\n");
        const uint32_t bn = 262144u;
        myds_metal_buffer_t *bk = myds_metal_buffer_alloc((size_t)bn * 68u);
        myds_metal_buffer_t *bsc = myds_metal_buffer_alloc((size_t)bn * 4u);
        fill_fp4_k_cache((uint8_t *)myds_metal_buffer_data(bk), NULL, bn, 120u, 132u, false);
        const int iters = 15;
        setenv("MYDS_INDEXER_DECODE_MMA", "0", 1);
        for (int i = 0; i < 3; i++) { myds_metal_begin_batch(); myds_v41_gpu_indexscores_head_slots_mode(bsc, q, w, bk, NULL, bn, 32u, MYDS_INDEXER_CACHE_FP4, 0u); myds_metal_commit_batch(); }
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) { myds_metal_begin_batch(); myds_v41_gpu_indexscores_head_slots_mode(bsc, q, w, bk, NULL, bn, 32u, MYDS_INDEXER_CACHE_FP4, 0u); myds_metal_commit_batch(); }
        double ref_ms = (now_ms() - t0) / iters;

        setenv("MYDS_INDEXER_DECODE_MMA", "1", 1);
        for (int i = 0; i < 3; i++) { myds_metal_begin_batch(); myds_v41_gpu_indexscores_head_slots_mode(bsc, q, w, bk, NULL, bn, 32u, MYDS_INDEXER_CACHE_FP4, 0u); myds_metal_commit_batch(); }
        double t1 = now_ms();
        for (int i = 0; i < iters; i++) { myds_metal_begin_batch(); myds_v41_gpu_indexscores_head_slots_mode(bsc, q, w, bk, NULL, bn, 32u, MYDS_INDEXER_CACHE_FP4, 0u); myds_metal_commit_batch(); }
        double opt_ms = (now_ms() - t1) / iters;
        printf("  PASS: M4 Decode score_one FP4 (262K): %.3f -> %.3f ms (%.2fx)\n",
               ref_ms, opt_ms, ref_ms / opt_ms);

        setenv("MYDS_INDEXER_DECODE_MMA", "0", 1);
        myds_metal_buffer_free(bk); myds_metal_buffer_free(bsc);
        myds_metal_buffer_free(q); myds_metal_buffer_free(w);
        myds_metal_buffer_free(k_fp4); myds_metal_buffer_free(k_bf16);
        myds_metal_buffer_free(mask); myds_metal_buffer_free(sc_ref);
        myds_metal_buffer_free(sc_opt); myds_metal_buffer_free(blk_sel);
        myds_v41_gpu_scratch_free(s);
        myds_metal_cleanup();
        printf("ALL [D-I1] DECODE MMA TESTS PASSED!\n");
        return 0;
    }
}
