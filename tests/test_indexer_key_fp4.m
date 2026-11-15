/* tests/test_indexer_key_fp4.m
 *
 * Apple Silicon GPU 0-ULP mathematical bit-identity, edge-case quantization,
 * multi-scorer parity, strict env parsing, and lowkv SSD round-trip suite for
 * MYDS_INDEXER_KEY_FP4 (4.25-bit E2M1FN + UE8M0, 68 B/row lossless Indexer Key Cache).
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

static float cpu_bf16_trunc(float f) {
    return bits_f32(f32_bits(f) & 0xffff0000u);
}

static float cpu_fp4_dequant_ref(uint32_t code4, uint32_t exp_u8) {
    static const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint32_t mag = code4 & 7u;
    const uint32_t sign_u32 = (code4 & 8u) << 28u;
    if (mag == 0u) return bits_f32(sign_u32);
    const float scale = bits_f32((exp_u8 & 0xffu) << 23u);
    const float prod = mags[mag] * scale;
    const uint32_t abs_bits = f32_bits(prod) & 0xffff0000u;
    return bits_f32(sign_u32 | abs_bits);
}

static void unpack_fp4_row_cpu(const uint8_t *row68, float out128[128]) {
    const uint8_t *nibbles = row68;
    const uint8_t *exps = row68 + 64;
    for (uint32_t blk = 0; blk < 4u; ++blk) {
        const uint32_t exp_u8 = exps[blk];
        for (uint32_t lane = 0; lane < 32u; ++lane) {
            const uint32_t idx = blk * 32u + lane;
            const uint8_t byte_v = nibbles[idx >> 1u];
            const uint32_t code4 = (idx & 1u) ? (byte_v >> 4u) : (byte_v & 0x0fu);
            out128[idx] = cpu_fp4_dequant_ref(code4, exp_u8);
        }
    }
}

static void unpack_bf16_row_cpu(const uint16_t *row128, float out128[128]) {
    for (uint32_t i = 0; i < 128u; ++i) {
        out128[i] = bits_f32((uint32_t)row128[i] << 16u);
    }
}

/* 1. Strict MYDS_INDEXER_KEY_FP4 environment parsing contract */
static void test_env_parsing(void) {
    const char *saved = getenv("MYDS_INDEXER_KEY_FP4");
    char saved_buf[64] = {0};
    const bool had_saved = (saved != NULL);
    if (had_saved) snprintf(saved_buf, sizeof(saved_buf), "%s", saved);

    unsetenv("MYDS_INDEXER_KEY_FP4");
    const int def_mode = myds_v41_gpu_indexer_key_fp4_enabled();
    CHECK(def_mode == 0 || def_mode == 1);

    setenv("MYDS_INDEXER_KEY_FP4", "0", 1);
    CHECK(myds_v41_gpu_indexer_key_fp4_enabled() == 0);
    CHECK(myds_v41_gpu_indexer_key_cache_mode() == MYDS_INDEXER_CACHE_BF16);
    CHECK(myds_v41_gpu_indexer_key_row_bytes() == 256u);

    setenv("MYDS_INDEXER_KEY_FP4", "1", 1);
    CHECK(myds_v41_gpu_indexer_key_fp4_enabled() == 1);
    CHECK(myds_v41_gpu_indexer_key_cache_mode() == MYDS_INDEXER_CACHE_FP4);
    CHECK(myds_v41_gpu_indexer_key_row_bytes() == 68u);

    const char *bad[] = {"", "2", "-1", "01", "true", "false", " 1", "1 "};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        setenv("MYDS_INDEXER_KEY_FP4", bad[i], 1);
        CHECK(myds_v41_gpu_indexer_key_fp4_enabled() == -1);
    }

    if (had_saved) setenv("MYDS_INDEXER_KEY_FP4", saved_buf, 1);
    else unsetenv("MYDS_INDEXER_KEY_FP4");
    printf("  [PASS] strict MYDS_INDEXER_KEY_FP4 env parsing & row-byte invariants\n");
}

/* 2. Exhaustive 16 x 256 = 4,096 (code4, exp_u8) GPU dequantization states vs CPU IEEE-754 */
static void test_exhaustive_4096_states(void) {
    const uint32_t n_rows = 4096u;
    myds_metal_buffer_t *fp4_cache = myds_metal_buffer_alloc((size_t)n_rows * 68u);
    myds_metal_buffer_t *bf16_cache = myds_metal_buffer_alloc((size_t)n_rows * 256u);
    myds_metal_buffer_t *q_buf = myds_metal_buffer_alloc(128u * sizeof(float));
    myds_metal_buffer_t *w_buf = myds_metal_buffer_alloc(sizeof(float));
    myds_metal_buffer_t *sc_fp4 = myds_metal_buffer_alloc((size_t)n_rows * sizeof(float));
    myds_metal_buffer_t *sc_bf16 = myds_metal_buffer_alloc((size_t)n_rows * sizeof(float));
    CHECK(fp4_cache && bf16_cache && q_buf && w_buf && sc_fp4 && sc_bf16);

    uint8_t *fp4_bytes = (uint8_t *)myds_metal_buffer_data(fp4_cache);
    uint16_t *bf16_u16 = (uint16_t *)myds_metal_buffer_data(bf16_cache);
    memset(fp4_bytes, 0, (size_t)n_rows * 68u);
    memset(bf16_u16, 0, (size_t)n_rows * 256u);

    for (uint32_t exp_u8 = 0; exp_u8 < 256u; ++exp_u8) {
        for (uint32_t code4 = 0; code4 < 16u; ++code4) {
            const uint32_t r = exp_u8 * 16u + code4;
            uint8_t *row68 = fp4_bytes + (size_t)r * 68u;
            row68[0] = (uint8_t)code4;
            row68[64] = (uint8_t)exp_u8;
            const float ref = cpu_fp4_dequant_ref(code4, exp_u8);
            bf16_u16[(size_t)r * 128u + 0u] = (uint16_t)(f32_bits(ref) >> 16u);
        }
    }

    float *qp = (float *)myds_metal_buffer_data(q_buf);
    float *wp = (float *)myds_metal_buffer_data(w_buf);
    const float signs[2] = {+1.0f, -1.0f};
    const float q_scales[2] = {64.0f, 1.0f / 64.0f};
    const float w_scales[2] = {1.0f, 1.0f};

    for (int wi = 0; wi < 2; ++wi) {
        *wp = w_scales[wi];
        for (int si = 0; si < 2; ++si) {
            memset(qp, 0, 128u * sizeof(float));
            qp[0] = signs[si] * q_scales[wi];
            CHECK(myds_v41_gpu_indexscores_mode(sc_fp4, q_buf, w_buf, fp4_cache,
                                                n_rows, 1u, MYDS_INDEXER_CACHE_FP4, 0u));
            CHECK(myds_v41_gpu_indexscores_mode(sc_bf16, q_buf, w_buf, bf16_cache,
                                                n_rows, 1u, MYDS_INDEXER_CACHE_BF16, 0u));
            const float *a = (const float *)myds_metal_buffer_data(sc_fp4);
            const float *b = (const float *)myds_metal_buffer_data(sc_bf16);
            for (uint32_t exp_u8 = 0; exp_u8 < 255u; ++exp_u8) {
                /* wi=0 covers exp_u8 < 246 (q=64 cancels args.scale=1/64 without overflow);
                 * wi=1 covers exp_u8 >= 246 (q=1/64 prevents 6*2^120 overflow, score = ref/4096). */
                if (wi == 0 && exp_u8 >= 246u) continue;
                if (wi == 1 && exp_u8 < 246u) continue;
                for (uint32_t code4 = 0; code4 < 16u; ++code4) {
                    const uint32_t r = exp_u8 * 16u + code4;
                    CHECK(f32_bits(a[r]) == f32_bits(b[r]));
                    const float ref = cpu_fp4_dequant_ref(code4, exp_u8);
                    const float gpu_in = ((f32_bits(ref) & 0x7f800000u) == 0u) ? 0.0f : ref;
                    const float dot = (signs[si] * q_scales[wi]) * gpu_in;
                    const float scaled = dot * (1.0f / 64.0f);
                    const float gpu_sc = ((f32_bits(scaled) & 0x7f800000u) == 0u) ? 0.0f : scaled;
                    const float exp_score = (gpu_sc > 0.0f ? gpu_sc : 0.0f) * w_scales[wi];
                    if (f32_bits(a[r]) != f32_bits(exp_score)) {
                        fprintf(stderr, "4096 MISMATCH exp=%u code=%u wi=%d si=%d a=0x%08x b=0x%08x exp_sc=0x%08x\n",
                                exp_u8, code4, wi, si, f32_bits(a[r]), f32_bits(b[r]), f32_bits(exp_score));
                        exit(1);
                    }
                }
            }
            /* For exp_u8 == 255 (+INF scale), verify mag==0 produces exact +0.0f (never NaN!) */
            for (uint32_t code4 = 0; code4 < 16u; ++code4) {
                const uint32_t r = 255u * 16u + code4;
                if ((code4 & 7u) == 0u) {
                    CHECK(!isnan(a[r]) && f32_bits(a[r]) == 0u);
                }
            }
        }
    }

    myds_metal_buffer_free(fp4_cache);
    myds_metal_buffer_free(bf16_cache);
    myds_metal_buffer_free(q_buf);
    myds_metal_buffer_free(w_buf);
    myds_metal_buffer_free(sc_fp4);
    myds_metal_buffer_free(sc_bf16);
    printf("  [PASS] exhaustive 16 x 256 = 4,096 (code4, exp_u8) states: 0-ULP GPU == CPU IEEE-754 (no NaN at 0*INF)\n");
}

/* 3. 100,000+ rows across BOTH GPU writers (fused rope_quant_batch & standalone pack_indexer_fp4)
 *    covering normal, exact E2M1 midpoint ties, closed-subset (3.0 < amax/scale <= 3.5) blocks,
 *    signed zeros (-0.0f), subnormals, and overflow (> 5.0 * 2^126). */
static void test_100k_rows_both_writers(void) {
    const uint32_t n_rows = 102400u; /* > 100K rows = 13,107,200 elements */
    myds_metal_buffer_t *src_a = myds_metal_buffer_alloc((size_t)n_rows * 128u * sizeof(float));
    myds_metal_buffer_t *src_b = myds_metal_buffer_alloc((size_t)n_rows * 128u * sizeof(float));
    myds_metal_buffer_t *bf16_dst = myds_metal_buffer_alloc((size_t)n_rows * 256u);
    myds_metal_buffer_t *fp4_fused = myds_metal_buffer_alloc((size_t)n_rows * 68u);
    myds_metal_buffer_t *fp4_pack = myds_metal_buffer_alloc((size_t)n_rows * 68u);
    CHECK(src_a && src_b && bf16_dst && fp4_fused && fp4_pack);

    float *pa = (float *)myds_metal_buffer_data(src_a);
    uint32_t rng = 0x243f6a88u;

    /* Populate diverse adversarial & random rows BEFORE RoPE+Hadamard+Quant */
    for (uint32_t r = 0; r < n_rows; ++r) {
        float *row = pa + (size_t)r * 128u;
        const uint32_t bucket = r % 8u;
        if (bucket == 0u) {
            /* All +0.0f and -0.0f mix */
            for (uint32_t i = 0; i < 128u; ++i) {
                row[i] = ((r + i) & 1u) ? -0.0f : +0.0f;
            }
        } else if (bucket == 1u) {
            /* Tiny subnormal / near-floor magnitudes around 7.05e-38 */
            for (uint32_t i = 0; i < 128u; ++i) {
                const float s = (xorshift32(&rng) & 1u) ? -1.0f : 1.0f;
                const float u = (float)(xorshift32(&rng) & 0xffffu) / 65535.0f;
                row[i] = s * u * 1.0e-36f;
            }
        } else if (bucket == 2u) {
            /* Huge finite magnitudes (up to ~3.1e38 post-Hadamard < FLT_MAX) to trigger E2M1 6*2^126 -> +INF overflow */
            for (uint32_t i = 0; i < 128u; ++i) {
                const float s = (xorshift32(&rng) & 1u) ? -1.0f : 1.0f;
                const float u = (float)(xorshift32(&rng) & 0xffffu) / 65535.0f;
                row[i] = s * u * 5.5e37f;
            }
        } else if (bucket == 3u) {
            /* Wide dynamic range per 32-element block */
            for (uint32_t b = 0; b < 4u; ++b) {
                const int exp_shift = (int)((xorshift32(&rng) % 160u)) - 80;
                const float amp = ldexpf(1.0f, exp_shift);
                for (uint32_t l = 0; l < 32u; ++l) {
                    const float u = ((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f;
                    row[b * 32u + l] = u * amp;
                }
            }
        } else {
            /* Standard normal-like activation distribution */
            for (uint32_t i = 0; i < 128u; ++i) {
                const float u = ((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f;
                row[i] = u * 3.5f;
            }
        }
    }
    memcpy(myds_metal_buffer_data(src_b), pa, (size_t)n_rows * 128u * sizeof(float));

    /* Path A: Fused kernel_dsv41_indexer_rope_quant_batch into BF16 vs FP4 */
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(
        src_a, bf16_dst, 0u, 1u, 100u, n_rows, 1u, true, MYDS_INDEXER_CACHE_BF16));
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(
        src_b, fp4_fused, 0u, 1u, 100u, n_rows, 1u, true, MYDS_INDEXER_CACHE_FP4));
    /* Path B: Standalone kernel_dsv41_pack_indexer_fp4 from already quantized src_a */
    CHECK(myds_v41_gpu_pack_indexer_fp4(fp4_pack, 0u, src_a, 0u, n_rows));

    const uint16_t *bf16_p = (const uint16_t *)myds_metal_buffer_data(bf16_dst);
    const uint8_t *fused_p = (const uint8_t *)myds_metal_buffer_data(fp4_fused);
    const uint8_t *pack_p = (const uint8_t *)myds_metal_buffer_data(fp4_pack);

    /* Both writers must produce byte-for-byte identical 68-byte FP4 rows, AND
     * every dequantized float across all 102,400 x 128 = 13,107,200 elements
     * must be 0-ULP bitwise identical (including sign of -0.0f) to BF16! */
    CHECK(memcmp(fused_p, pack_p, (size_t)n_rows * 68u) == 0);

    float row_bf16[128], row_fp4[128];
    for (uint32_t r = 0; r < n_rows; ++r) {
        unpack_bf16_row_cpu(bf16_p + (size_t)r * 128u, row_bf16);
        unpack_fp4_row_cpu(fused_p + (size_t)r * 68u, row_fp4);
        for (uint32_t i = 0; i < 128u; ++i) {
            if (f32_bits(row_bf16[i]) != f32_bits(row_fp4[i])) {
                fprintf(stderr, "MISMATCH row=%u col=%u bf16=0x%08x (%g) fp4=0x%08x (%g)\n",
                        r, i, f32_bits(row_bf16[i]), row_bf16[i], f32_bits(row_fp4[i]), row_fp4[i]);
                exit(1);
            }
        }
    }

    /* Direct post-Hadamard adversarial test: exact midpoint ties {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}*2^E
     * AND closed-subset blocks where max quantized value is 3.0*2^E (amax in (3.0, 3.5]*2^E)
     * so max_abits / 6.0 lands in [0.5, 0.5833]*2^E while pow2_ceil still recovers exact 2^E! */
    const uint32_t n_adv = 4096u;
    float *adv = pa;
    static const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    for (uint32_t r = 0; r < n_adv; ++r) {
        const int e = (int)(r % 200u) - 100;
        const float sc = ldexpf(1.0f, e);
        float *row = adv + (size_t)r * 128u;
        for (uint32_t b = 0; b < 4u; ++b) {
            for (uint32_t l = 0; l < 32u; ++l) {
                if ((r & 1u) && b == 0u) {
                    /* Closed-subset block: max value is 3.25 * sc -> rounds to 3.0 * sc, NO 4.0 or 6.0! */
                    const float v = (l == 0u) ? (3.25f * sc) : (mids[l % 5u] * sc);
                    row[b * 32u + l] = cpu_bf16_trunc(((l & 1u) ? -v : v));
                } else {
                    const float v = (l == 0u) ? (5.0f * sc) : (mids[(b * 32u + l) % 7u] * sc);
                    row[b * 32u + l] = cpu_bf16_trunc(((l & 2u) ? -v : v));
                }
            }
        }
    }
    CHECK(myds_v41_gpu_quant(src_a, 128u, n_adv, 2u));
    CHECK(myds_v41_gpu_pack_bf16(bf16_dst, 0u, src_a, 0u, n_adv * 128u));
    CHECK(myds_v41_gpu_pack_indexer_fp4(fp4_pack, 0u, src_a, 0u, n_adv));
    for (uint32_t r = 0; r < n_adv; ++r) {
        unpack_bf16_row_cpu(bf16_p + (size_t)r * 128u, row_bf16);
        unpack_fp4_row_cpu(pack_p + (size_t)r * 68u, row_fp4);
        for (uint32_t i = 0; i < 128u; ++i) {
            CHECK(f32_bits(row_bf16[i]) == f32_bits(row_fp4[i]));
        }
    }

    myds_metal_buffer_free(src_a);
    myds_metal_buffer_free(src_b);
    myds_metal_buffer_free(bf16_dst);
    myds_metal_buffer_free(fp4_fused);
    myds_metal_buffer_free(fp4_pack);
    printf("  [PASS] 102,400+ rows (13.1M elems) + 4,096 midpoint/closed-subset rows: 0-ULP bit-exact across both GPU writers\n");
}

/* 4. 0-ULP score & Top-512 bit-identity across ALL GPU scorer kernels (Prefill & Decode) */
static void test_all_gpu_scorers_0ulp(void) {
    const uint32_t n_comp = 20480u; /* > 16,384 so both [D-L1] and [P-L1] compact paths activate! */
    const uint32_t n_tok = 64u;
    const uint32_t pos0 = n_comp - n_tok;
    const uint32_t n_blocks = (n_comp + 7u) / 8u; /* 2560 blocks */
    const uint32_t mask_units = (n_blocks + 31u) / 32u; /* 80 uint32 words */
    const uint32_t top_blocks = 2048u;

    setenv("MYDS_CHUNK_KEY_MAX", "65536", 1);
    myds_v41_gpu_scratch_t *scratch = myds_v41_gpu_scratch_create(65536u);
    unsetenv("MYDS_CHUNK_KEY_MAX");
    CHECK(scratch != NULL);

    myds_metal_buffer_t *raw_k = myds_metal_buffer_alloc((size_t)n_comp * 128u * sizeof(float));
    myds_metal_buffer_t *raw_k2 = myds_metal_buffer_alloc((size_t)n_comp * 128u * sizeof(float));
    myds_metal_buffer_t *cache_bf16 = myds_metal_buffer_alloc((size_t)n_comp * 256u);
    myds_metal_buffer_t *cache_fp4 = myds_metal_buffer_alloc((size_t)n_comp * 68u);
    myds_metal_buffer_t *q_buf = myds_metal_buffer_alloc((size_t)n_tok * 32u * 128u * sizeof(float));
    myds_metal_buffer_t *w_buf = myds_metal_buffer_alloc((size_t)n_tok * 32u * sizeof(float));
    myds_metal_buffer_t *sc_bf16 = myds_metal_buffer_alloc((size_t)n_tok * n_comp * sizeof(float));
    myds_metal_buffer_t *sc_fp4 = myds_metal_buffer_alloc((size_t)n_tok * n_comp * sizeof(float));
    myds_metal_buffer_t *mask_1d = myds_metal_buffer_alloc((size_t)n_blocks * sizeof(float));
    myds_metal_buffer_t *mask_2d = myds_metal_buffer_alloc((size_t)n_tok * mask_units * sizeof(uint32_t));
    myds_metal_buffer_t *blk_sel = myds_metal_buffer_alloc((size_t)top_blocks * sizeof(int32_t));
    myds_metal_buffer_t *top_bf16 = myds_metal_buffer_alloc((size_t)n_tok * 512u * sizeof(int32_t));
    myds_metal_buffer_t *top_fp4 = myds_metal_buffer_alloc((size_t)n_tok * 512u * sizeof(int32_t));
    CHECK(raw_k && raw_k2 && cache_bf16 && cache_fp4 && q_buf && w_buf &&
          sc_bf16 && sc_fp4 && mask_1d && mask_2d && blk_sel && top_bf16 && top_fp4);

    uint32_t rng = 0x9e3779b9u;
    float *kp = (float *)myds_metal_buffer_data(raw_k);
    for (size_t i = 0; i < (size_t)n_comp * 128u; ++i) {
        kp[i] = (((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f) * 2.5f;
    }
    memcpy(myds_metal_buffer_data(raw_k2), kp, (size_t)n_comp * 128u * sizeof(float));
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(
        raw_k, cache_bf16, 0u, 1u, 0u, n_comp, 1u, true, MYDS_INDEXER_CACHE_BF16));
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(
        raw_k2, cache_fp4, 0u, 1u, 0u, n_comp, 1u, true, MYDS_INDEXER_CACHE_FP4));

    float *qp = (float *)myds_metal_buffer_data(q_buf);
    float *wp = (float *)myds_metal_buffer_data(w_buf);
    for (size_t i = 0; i < (size_t)n_tok * 32u * 128u; ++i) {
        qp[i] = (((float)(xorshift32(&rng) & 0xffffffu) / 8388608.0f) - 1.0f) * 1.5f;
    }
    CHECK(myds_v41_gpu_indexer_rope_quant_batch(q_buf, NULL, 0u, 32u, pos0, n_tok, 1u, true));
    for (size_t i = 0; i < (size_t)n_tok * 32u; ++i) {
        wp[i] = 0.05f + 0.95f * ((float)(xorshift32(&rng) & 0xffffu) / 65535.0f);
    }

    /* Build deterministic block_selected + mask_1d (float per block) + mask_2d (bitmask per row) */
    int32_t *bs = (int32_t *)myds_metal_buffer_data(blk_sel);
    float *m1 = (float *)myds_metal_buffer_data(mask_1d);
    uint32_t *m2 = (uint32_t *)myds_metal_buffer_data(mask_2d);
    for (uint32_t b = 0; b < n_blocks; ++b) m1[b] = 1.0f; /* 1.0f = masked out */
    memset(m2, 0, (size_t)n_tok * mask_units * sizeof(uint32_t));
    for (uint32_t i = 0; i < top_blocks; ++i) {
        const uint32_t blk = (i * 2557u) % n_blocks; /* 2557 coprime with 2560 -> injection */
        bs[i] = (int32_t)blk;
        m1[blk] = 0.0f; /* 0.0f = unmasked / selected */
    }
    for (uint32_t r = 0; r < n_tok; ++r) {
        const uint32_t vis_blks = (pos0 + r + 1u + 7u) / 8u;
        uint32_t *rw = m2 + (size_t)r * mask_units;
        uint32_t picked = 0u;
        for (uint32_t b = 0; b < vis_blks && picked < top_blocks; ++b) {
            if ((xorshift32(&rng) % (vis_blks - b)) < (top_blocks - picked)) {
                rw[b >> 5u] |= (1u << (b & 31u));
                picked++;
            }
        }
    }

    /* 4A. Decode scorers: indexscores_mode (1, 200 direct, 20480 rows4), masked, first_row, head_slots, selected_blocks, compact_decode */
    const uint32_t decode_counts[] = {1u, 200u, n_comp};
    for (size_t ci = 0; ci < 3u; ++ci) {
        const uint32_t nc = decode_counts[ci];
        const uint32_t f_rows[] = {0u, nc > 64u ? 37u : 0u};
        for (size_t fi = 0; fi < 2u; ++fi) {
            const uint32_t fr = f_rows[fi];
            CHECK(myds_v41_gpu_indexscores_mode(sc_bf16, q_buf, w_buf, cache_bf16, nc, 32u, MYDS_INDEXER_CACHE_BF16, fr));
            CHECK(myds_v41_gpu_indexscores_mode(sc_fp4, q_buf, w_buf, cache_fp4, nc, 32u, MYDS_INDEXER_CACHE_FP4, fr));
            CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)nc * sizeof(float)) == 0);

            CHECK(myds_v41_gpu_indexscores_masked_mode(sc_bf16, q_buf, w_buf, cache_bf16, mask_1d, nc, 32u, MYDS_INDEXER_CACHE_BF16, fr));
            CHECK(myds_v41_gpu_indexscores_masked_mode(sc_fp4, q_buf, w_buf, cache_fp4, mask_1d, nc, 32u, MYDS_INDEXER_CACHE_FP4, fr));
            CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)nc * sizeof(float)) == 0);

            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_head_slots_mode(sc_bf16, q_buf, w_buf, cache_bf16, NULL, nc, 32u, MYDS_INDEXER_CACHE_BF16, fr));
            CHECK(myds_v41_gpu_indexscores_head_slots_mode(sc_fp4, q_buf, w_buf, cache_fp4, NULL, nc, 32u, MYDS_INDEXER_CACHE_FP4, fr));
            CHECK(myds_metal_commit_batch());
            CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)nc * sizeof(float)) == 0);

            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_head_slots_mode(sc_bf16, q_buf, w_buf, cache_bf16, mask_1d, nc, 32u, MYDS_INDEXER_CACHE_BF16, fr));
            CHECK(myds_v41_gpu_indexscores_head_slots_mode(sc_fp4, q_buf, w_buf, cache_fp4, mask_1d, nc, 32u, MYDS_INDEXER_CACHE_FP4, fr));
            CHECK(myds_metal_commit_batch());
            CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)nc * sizeof(float)) == 0);
        }
    }

    /* selected_blocks: both small (32 blocks -> non-rows4) and full (2048 blocks -> rows4) */
    const uint32_t sb_counts[] = {32u, top_blocks};
    for (size_t bi = 0; bi < 2u; ++bi) {
        const uint32_t nb = sb_counts[bi];
        for (uint32_t fr = 0u; fr <= 40u; fr += 40u) {
            memset(myds_metal_buffer_data(sc_bf16), 0xaa, (size_t)n_comp * sizeof(float));
            memset(myds_metal_buffer_data(sc_fp4), 0xaa, (size_t)n_comp * sizeof(float));
            CHECK(myds_v41_gpu_indexscores_selected_blocks_mode(
                sc_bf16, q_buf, w_buf, cache_bf16, mask_1d, blk_sel, n_comp, nb, MYDS_INDEXER_CACHE_BF16, fr));
            CHECK(myds_v41_gpu_indexscores_selected_blocks_mode(
                sc_fp4, q_buf, w_buf, cache_fp4, mask_1d, blk_sel, n_comp, nb, MYDS_INDEXER_CACHE_FP4, fr));
            CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)n_comp * sizeof(float)) == 0);
        }
    }

    /* [D-L1] compact_decode_mode (with and without mask_1d, with and without first_row) */
    CHECK(myds_v41_gpu_indexer_prepare_compact_blocks(scratch, blk_sel, n_blocks, top_blocks, n_comp));
    for (int m_on = 0; m_on < 2; ++m_on) {
        const myds_metal_buffer_t *mp = m_on ? mask_1d : NULL;
        for (uint32_t fr = 0u; fr <= 53u; fr += 53u) {
            CHECK(myds_v41_gpu_indexscores_compact_decode_mode(
                scratch, q_buf, w_buf, cache_bf16, mp, n_comp, MYDS_INDEXER_CACHE_BF16, fr));
            CHECK(myds_v41_gpu_topk_compact_decode(top_bf16, n_comp, scratch));

            CHECK(myds_v41_gpu_indexscores_compact_decode_mode(
                scratch, q_buf, w_buf, cache_fp4, mp, n_comp, MYDS_INDEXER_CACHE_FP4, fr));
            CHECK(myds_v41_gpu_topk_compact_decode(top_fp4, n_comp, scratch));

            CHECK(memcmp(myds_metal_buffer_data(top_bf16), myds_metal_buffer_data(top_fp4), 512u * sizeof(int32_t)) == 0);
        }
    }

    /* 4B. Prefill chunk scorers: chunk_exact, chunk_direct, tiled_fp4, tiled_sg_fp4 */
    /* Test chunk_exact (default) */
    unsetenv("MYDS_INDEXER_TILED");
    unsetenv("MYDS_INDEXER_SGBAR");
    unsetenv("MYDS_INDEXER_CHUNK_EXACT");
    CHECK(myds_v41_gpu_indexscores_chunk_mode(sc_bf16, q_buf, w_buf, cache_bf16, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_BF16));
    CHECK(myds_v41_gpu_indexscores_chunk_mode(sc_fp4, q_buf, w_buf, cache_fp4, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_FP4));
    CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)n_tok * n_comp * sizeof(float)) == 0);

    /* Test chunk_direct */
    setenv("MYDS_INDEXER_CHUNK_EXACT", "0", 1);
    CHECK(myds_v41_gpu_indexscores_chunk_mode(sc_bf16, q_buf, w_buf, cache_bf16, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_BF16));
    CHECK(myds_v41_gpu_indexscores_chunk_mode(sc_fp4, q_buf, w_buf, cache_fp4, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_FP4));
    CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)n_tok * n_comp * sizeof(float)) == 0);
    unsetenv("MYDS_INDEXER_CHUNK_EXACT");

    /* Test tiled_fp4 */
    setenv("MYDS_INDEXER_TILED", "1", 1);
    CHECK(myds_v41_gpu_indexscores_chunk_mode(sc_bf16, q_buf, w_buf, cache_bf16, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_BF16));
    CHECK(myds_v41_gpu_indexscores_chunk_mode(sc_fp4, q_buf, w_buf, cache_fp4, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_FP4));
    CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)n_tok * n_comp * sizeof(float)) == 0);

    /* Test tiled_sg_fp4 against race-free tiled_bf16 reference */
    setenv("MYDS_INDEXER_SGBAR", "1", 1);
    CHECK(myds_v41_gpu_indexscores_chunk_mode(sc_fp4, q_buf, w_buf, cache_fp4, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_FP4));
    CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)n_tok * n_comp * sizeof(float)) == 0);
    unsetenv("MYDS_INDEXER_TILED");
    unsetenv("MYDS_INDEXER_SGBAR");

    /* 4C. Prefill IRO (register_only_fp4 & register_only_0b_fp4) + [P-L1] prefill_compact_ro_fp4 / 0b_fp4 */
    const char *smem_modes[2] = {"1", "0"};
    for (int sm = 0; sm < 2; ++sm) {
        setenv("MYDS_INDEXER_RO_SMEM_Q", smem_modes[sm], 1);

        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
            scratch, sc_bf16, q_buf, w_buf, cache_bf16, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_BF16, true));
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(
            scratch, sc_fp4, q_buf, w_buf, cache_fp4, n_comp, 32u, n_tok, pos0, 1u, MYDS_INDEXER_CACHE_FP4, true));
        CHECK(myds_metal_commit_batch());
        CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)n_tok * n_comp * sizeof(float)) == 0);

        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(
            scratch, top_bf16, q_buf, w_buf, cache_bf16, mask_2d,
            n_comp, n_tok, pos0, 1u, mask_units, top_blocks, MYDS_INDEXER_CACHE_BF16, true) == 1);
        CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(
            scratch, top_fp4, q_buf, w_buf, cache_fp4, mask_2d,
            n_comp, n_tok, pos0, 1u, mask_units, top_blocks, MYDS_INDEXER_CACHE_FP4, true) == 1);
        CHECK(myds_metal_commit_batch());
        CHECK(memcmp(myds_metal_buffer_data(top_bf16), myds_metal_buffer_data(top_fp4), (size_t)n_tok * 512u * sizeof(int32_t)) == 0);
    }
    unsetenv("MYDS_INDEXER_RO_SMEM_Q");

    myds_metal_buffer_free(raw_k);
    myds_metal_buffer_free(raw_k2);
    myds_metal_buffer_free(cache_bf16);
    myds_metal_buffer_free(cache_fp4);
    myds_metal_buffer_free(q_buf);
    myds_metal_buffer_free(w_buf);
    myds_metal_buffer_free(sc_bf16);
    myds_metal_buffer_free(sc_fp4);
    myds_metal_buffer_free(mask_1d);
    myds_metal_buffer_free(mask_2d);
    myds_metal_buffer_free(blk_sel);
    myds_metal_buffer_free(top_bf16);
    myds_metal_buffer_free(top_fp4);
    myds_v41_gpu_scratch_free(scratch);
    printf("  [PASS] 0-ULP bit-exact score & Top-512 parity across all 14 Decode + Prefill GPU scorer paths\n");
}

/* 5. lowkv SSD round-trip under MYDS_INDEXER_KEY_FP4=1 + cross-format isolation against FP4=0 */
static void test_lowkv_fp4_roundtrip_and_isolation(void) {
    const char *saved = getenv("MYDS_INDEXER_KEY_FP4");
    char saved_buf[64] = {0};
    const bool had_saved = (saved != NULL);
    if (had_saved) snprintf(saved_buf, sizeof(saved_buf), "%s", saved);

    char dir[] = "/tmp/test_myds_fp4_lowkv_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char path[512];
    snprintf(path, sizeof(path), "%s/snap.bin", dir);

    const uint32_t ntok = 1600u; /* 3 pinned segments (512, 1024, 1536) + prompt-end tail at 1599 */
    int32_t *tokens = (int32_t *)malloc(ntok * sizeof(int32_t));
    CHECK(tokens != NULL);
    for (uint32_t i = 0; i < ntok; ++i) tokens[i] = (int32_t)(i * 41u + 7u);

    setenv("MYDS_INDEXER_KEY_FP4", "1", 1);
    myds_v41_context_t *g1 = myds_v41_context_create_snapshot_fixture();
    CHECK(g1 != NULL);
    CHECK(myds_v41_low_kv_fixture_enable(g1, path, 0x12345678ull, 1, false, ntok + 64u));

    /* Verify index_cache buffer size in fixture is 68 B/row, not 256 B/row! */
    CHECK(myds_metal_buffer_size(myds_v41_snapshot_get_index_cache(g1, 3u)) == (ntok + 64u + 1u) * 68u);
    CHECK(myds_metal_buffer_size(myds_v41_snapshot_get_index_cache(g1, 0u)) == (((ntok + 64u) / 2u) + 1u) * 68u);

    /* Populate pinned windows (1024, 1536), prompt-end window (1599), compressed_kv, and FP4 index_cache */
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
        uint16_t *c = (uint16_t *)myds_metal_buffer_data(myds_v41_snapshot_get_compressed(g1, o));
        const size_t nc = myds_metal_buffer_size(myds_v41_snapshot_get_compressed(g1, o)) / sizeof(uint16_t);
        for (size_t k = 0; k < nc; ++k) c[k] = (uint16_t)(0x3c00u + ((o * 131u + k) & 0x3ffu));

        uint8_t *x = (uint8_t *)myds_metal_buffer_data(myds_v41_snapshot_get_index_cache(g1, o));
        const size_t nx = myds_metal_buffer_size(myds_v41_snapshot_get_index_cache(g1, o));
        for (size_t k = 0; k < nx; ++k) x[k] = (uint8_t)((o * 73u + k * 19u + 5u) & 0xffu);
    }
    myds_v41_snapshot_set_engram_history(g1, 11u, 22u, 33u);
    CHECK(myds_v41_snapshot_store_now(g1, ntok - 1u));
    CHECK(myds_v41_low_kv_save(g1, tokens, ntok));
    myds_v41_context_free(g1);

    /* Reload in a fresh FP4=1 context -> must hit prompt-end 1599 and restore all FP4 bytes identically */
    myds_v41_context_t *g2 = myds_v41_context_create_snapshot_fixture();
    CHECK(g2 != NULL);
    CHECK(myds_v41_low_kv_fixture_enable(g2, path, 0x12345678ull, 1, false, ntok + 64u));
    CHECK(myds_v41_low_kv_load_deeper(g2, tokens, 0u, ntok) == ntok - 1u);
    CHECK(myds_v41_snapshot_restore(g2, ntok - 1u));
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t r = (o == 3u) ? 1u : 2u;
        const size_t valid_bytes = (size_t)((ntok - 1u) / r) * 68u;
        const uint8_t *x = (const uint8_t *)myds_metal_buffer_data(myds_v41_snapshot_get_index_cache(g2, o));
        for (size_t k = 0; k < valid_bytes; ++k) {
            CHECK(x[k] == (uint8_t)((o * 73u + k * 19u + 5u) & 0xffu));
        }
    }
    myds_v41_context_free(g2);

    /* Cross-format isolation: opening the same path under MYDS_INDEXER_KEY_FP4=0 (BF16 256 B/row)
     * must cleanly reject/reinitialize or miss due to distinct lowkv_identity_mode & row stride! */
    setenv("MYDS_INDEXER_KEY_FP4", "0", 1);
    myds_v41_context_t *g3 = myds_v41_context_create_snapshot_fixture();
    CHECK(g3 != NULL);
    if (myds_v41_low_kv_fixture_enable(g3, path, 0x12345678ull, 1, false, ntok + 64u)) {
        CHECK(myds_v41_low_kv_probe(g3, tokens, ntok) == 0u);
    }
    myds_v41_context_free(g3);

    /* Cleanup temp directory */
    char seg_path[540], end_path[540];
    snprintf(seg_path, sizeof(seg_path), "%s.segments", path);
    snprintf(end_path, sizeof(end_path), "%s.ends", path);
    unlink(path);
    unlink(seg_path);
    unlink(end_path);
    /* Remove any kvanchor-v1-* in dir */
    NSArray<NSString *> *items = [[NSFileManager defaultManager]
        contentsOfDirectoryAtPath:[NSString stringWithUTF8String:dir] error:nil];
    for (NSString *item in items) {
        NSString *full = [[NSString stringWithUTF8String:dir] stringByAppendingPathComponent:item];
        unlink([full UTF8String]);
    }
    rmdir(dir);
    free(tokens);

    if (had_saved) setenv("MYDS_INDEXER_KEY_FP4", saved_buf, 1);
    else unsetenv("MYDS_INDEXER_KEY_FP4");
    printf("  [PASS] lowkv SSD segment + prompt-end round-trip (68 B/row) & cross-format isolation vs BF16\n");
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
            const char *bf16_peer;
            NSUInteger min_max_threads;
            NSUInteger max_static_smem;
        } specs[] = {
            {"kernel_dsv41_pack_indexer_fp4",                        NULL,                                                   1024u,    0u},
            {"kernel_myds_indexer_scores_register_only_fp4",         "kernel_myds_indexer_scores_register_only_bf16",         832u, 4096u},
            {"kernel_myds_indexer_scores_register_only_0b_fp4",      "kernel_myds_indexer_scores_register_only_0b_bf16",     1024u,    0u},
            {"kernel_myds_indexer_scores_prefill_compact_ro_fp4",    "kernel_myds_indexer_scores_prefill_compact_ro_bf16",    768u, 4096u},
            {"kernel_myds_indexer_scores_prefill_compact_ro_0b_fp4", "kernel_myds_indexer_scores_prefill_compact_ro_0b_bf16", 896u,    0u},
            {"kernel_dsv4_indexer_scores_tiled_fp4",                 "kernel_dsv4_indexer_scores_tiled_bf16",                1024u, 4096u},
            {"kernel_dsv4_indexer_scores_tiled_sg_fp4",              "kernel_dsv4_indexer_scores_tiled_sg_bf16",             1024u, 4096u},
        };

        printf("  Verifying FP4 Metal PSO occupancy & static threadgroup memory:\n");
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
            printf("    %-52s width=%lu max_threads=%4lu (bf16=%4lu) static_smem=%4lu B\n",
                   specs[i].name,
                   (unsigned long)pso.threadExecutionWidth,
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)peer_threads,
                   (unsigned long)pso.staticThreadgroupMemoryLength);
        }
    }
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

static void run_fp4_microbenchmarks(void) {
    printf("=== Running 7-run median GPU microbenchmarks: BF16 (256 B/row) vs FP4 (68 B/row, -73.44%%) ===\n");
    const uint32_t max_nc = 262144u;
    const uint32_t max_rows = 512u;
    setenv("MYDS_CHUNK_KEY_MAX", "262144", 1);
    myds_v41_gpu_scratch_t *scratch = myds_v41_gpu_scratch_create(max_nc);
    unsetenv("MYDS_CHUNK_KEY_MAX");
    CHECK(scratch != NULL);

    myds_metal_buffer_t *raw_k = myds_metal_buffer_alloc((size_t)max_nc * 128u * sizeof(float));
    myds_metal_buffer_t *cache_bf16 = myds_metal_buffer_alloc((size_t)max_nc * 256u);
    myds_metal_buffer_t *cache_fp4  = myds_metal_buffer_alloc((size_t)max_nc * 68u);
    myds_metal_buffer_t *q_buf      = myds_metal_buffer_alloc((size_t)max_rows * 32u * 128u * sizeof(float));
    myds_metal_buffer_t *w_buf      = myds_metal_buffer_alloc((size_t)max_rows * 32u * sizeof(float));
    myds_metal_buffer_t *sc_bf16    = myds_metal_buffer_alloc((size_t)8388608u * sizeof(float));
    myds_metal_buffer_t *sc_fp4     = myds_metal_buffer_alloc((size_t)8388608u * sizeof(float));
    const uint32_t max_blocks = max_nc / 8u;
    const uint32_t mask_units = myds_v41_gpu_maskbits_stride(max_blocks);
    myds_metal_buffer_t *mask_2d    = myds_metal_buffer_alloc((size_t)max_rows * mask_units * sizeof(uint32_t));
    myds_metal_buffer_t *blk_sel    = myds_metal_buffer_alloc(16384u * sizeof(int32_t));
    myds_metal_buffer_t *top_bf16   = myds_metal_buffer_alloc((size_t)max_rows * 512u * sizeof(int32_t));
    myds_metal_buffer_t *top_fp4    = myds_metal_buffer_alloc((size_t)max_rows * 512u * sizeof(int32_t));
    CHECK(raw_k && cache_bf16 && cache_fp4 && q_buf && w_buf && sc_bf16 && sc_fp4 && mask_2d && blk_sel && top_bf16 && top_fp4);

    uint32_t rng = 0x517cc1b7u;
    float *kp = (float *)myds_metal_buffer_data(raw_k);
    for (size_t i = 0; i < (size_t)max_nc * 128u; ++i) {
        kp[i] = ((float)((int32_t)(xorshift32(&rng) % 2001u) - 1000)) * 0.01f;
    }
    myds_metal_buffer_t *raw_k2 = myds_metal_buffer_alloc((size_t)max_nc * 128u * sizeof(float));
    CHECK(raw_k2 != NULL);
    memcpy(myds_metal_buffer_data(raw_k2), kp, (size_t)max_nc * 128u * sizeof(float));
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(raw_k, cache_bf16, 0u, 1u, 0u, max_nc, 1u, true, MYDS_INDEXER_CACHE_BF16));
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(raw_k2, cache_fp4, 0u, 1u, 0u, max_nc, 1u, true, MYDS_INDEXER_CACHE_FP4));
    myds_metal_buffer_free(raw_k2);

    float *qp = (float *)myds_metal_buffer_data(q_buf);
    for (size_t i = 0; i < (size_t)max_rows * 32u * 128u; ++i) {
        qp[i] = ((float)((int32_t)(xorshift32(&rng) % 2001u) - 1000)) * 0.01f;
    }
    CHECK(myds_v41_gpu_indexer_rope_quant_batch_mode(q_buf, NULL, 0u, 32u, 0u, max_rows, 1u, true, MYDS_INDEXER_CACHE_F32));

    float *wp = (float *)myds_metal_buffer_data(w_buf);
    for (size_t i = 0; i < (size_t)max_rows * 32u; ++i) {
        wp[i] = 0.05f + 0.01f * (float)(xorshift32(&rng) % 100u);
    }

    int32_t *bsp = (int32_t *)myds_metal_buffer_data(blk_sel);
    for (uint32_t i = 0; i < 2048u; ++i) bsp[i] = (int32_t)((i * 15u + 3u) % max_blocks);

    /* 1. Prefill register_only (IRO) */
    struct { const char *name; uint32_t nc; uint32_t glen; } iro_cases[] = {
        {"IRO_32K_g256",   32768u, 256u},
        {"IRO_64K_g128",   65536u, 128u},
        {"IRO_128K_g64",  131072u,  64u},
        {"IRO_256K_g32",  262144u,  32u},
        {"IRO_128K_warm16", 131072u, 16u},
    };
    printf("  --- Prefill IRO Scorer (register_only_bf16 vs register_only_fp4) ---\n");
    for (size_t i = 0; i < sizeof(iro_cases) / sizeof(iro_cases[0]); ++i) {
        const uint32_t nc = iro_cases[i].nc, glen = iro_cases[i].glen, pos0 = nc - glen;
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(scratch, sc_bf16, q_buf, w_buf, cache_bf16, nc, 32u, glen, pos0, 1u, MYDS_INDEXER_CACHE_BF16, true));
        CHECK(myds_metal_commit_batch());
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(scratch, sc_fp4, q_buf, w_buf, cache_fp4, nc, 32u, glen, pos0, 1u, MYDS_INDEXER_CACHE_FP4, true));
        CHECK(myds_metal_commit_batch());
        CHECK(memcmp(myds_metal_buffer_data(sc_bf16), myds_metal_buffer_data(sc_fp4), (size_t)nc * glen * sizeof(float)) == 0);

        double t_bf16[7], t_fp4[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(scratch, sc_bf16, q_buf, w_buf, cache_bf16, nc, 32u, glen, pos0, 1u, MYDS_INDEXER_CACHE_BF16, true));
            CHECK(myds_metal_commit_batch());
            t_bf16[r] = monotonic_ms() - s0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped_mode(scratch, sc_fp4, q_buf, w_buf, cache_fp4, nc, 32u, glen, pos0, 1u, MYDS_INDEXER_CACHE_FP4, true));
            CHECK(myds_metal_commit_batch());
            t_fp4[r] = monotonic_ms() - s1;
        }
        qsort(t_bf16, 7, sizeof(double), cmp_double);
        qsort(t_fp4,  7, sizeof(double), cmp_double);
        printf("    %-18s (nc=%6u, glen=%3u): BF16 %6.2f ms | FP4 %6.2f ms | %.2fx\n",
               iro_cases[i].name, nc, glen, t_bf16[3], t_fp4[3], t_bf16[3] / t_fp4[3]);
    }

    /* 2. Prefill [P-L1] Compact YoCo Pipeline */
    struct { const char *name; uint32_t nc; uint32_t glen; } pl1_cases[] = {
        {"PL1_128K_g512", 131072u, 512u},
        {"PL1_256K_g304", 262144u, 304u},
    };
    printf("  --- Prefill [P-L1] YoCo Compact Pipeline (BF16 vs FP4) ---\n");
    for (size_t i = 0; i < sizeof(pl1_cases) / sizeof(pl1_cases[0]); ++i) {
        const uint32_t nc = pl1_cases[i].nc, glen = pl1_cases[i].glen, pos0 = nc - glen;
        const uint32_t n_blocks = nc / 8u;
        const uint32_t mu = myds_v41_gpu_maskbits_stride(n_blocks);
        uint32_t *m2 = (uint32_t *)myds_metal_buffer_data(mask_2d);
        memset(m2, 0, (size_t)glen * mu * sizeof(uint32_t));
        for (uint32_t r = 0; r < glen; ++r) {
            /* Realistic overlapping L20 2048-block mask (~12.6% union at 128K, ~6.4% at 256K) */
            const uint32_t tile_base = ((r / 8u) * 37u) % (n_blocks - 2200u);
            for (uint32_t k = 0; k < 2048u; ++k) {
                const uint32_t b = (tile_base + k * (n_blocks / 2100u) + (r & 7u)) % n_blocks;
                m2[(size_t)r * mu + (b >> 5u)] |= (1u << (b & 31u));
            }
        }
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(scratch, top_bf16, q_buf, w_buf, cache_bf16, mask_2d, nc, glen, pos0, 1u, mu, 2048u, MYDS_INDEXER_CACHE_BF16, true) == 1);
        CHECK(myds_metal_commit_batch());
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(scratch, top_fp4, q_buf, w_buf, cache_fp4, mask_2d, nc, glen, pos0, 1u, mu, 2048u, MYDS_INDEXER_CACHE_FP4, true) == 1);
        CHECK(myds_metal_commit_batch());
        CHECK(memcmp(myds_metal_buffer_data(top_bf16), myds_metal_buffer_data(top_fp4), (size_t)glen * 512u * sizeof(int32_t)) == 0);

        double t_bf16[7], t_fp4[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(scratch, top_bf16, q_buf, w_buf, cache_bf16, mask_2d, nc, glen, pos0, 1u, mu, 2048u, MYDS_INDEXER_CACHE_BF16, true) == 1);
            CHECK(myds_metal_commit_batch());
            t_bf16[r] = monotonic_ms() - s0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            CHECK(myds_v41_gpu_indexer_prefill_compact_scoped_mode(scratch, top_fp4, q_buf, w_buf, cache_fp4, mask_2d, nc, glen, pos0, 1u, mu, 2048u, MYDS_INDEXER_CACHE_FP4, true) == 1);
            CHECK(myds_metal_commit_batch());
            t_fp4[r] = monotonic_ms() - s1;
        }
        qsort(t_bf16, 7, sizeof(double), cmp_double);
        qsort(t_fp4,  7, sizeof(double), cmp_double);
        printf("    %-18s (nc=%6u, glen=%3u): BF16 %6.2f ms | FP4 %6.2f ms | %.2fx\n",
               pl1_cases[i].name, nc, glen, t_bf16[3], t_fp4[3], t_bf16[3] / t_fp4[3]);
    }

    /* 3. Single-step Decode Scorer (rows4 x 4 root/L20 layers per token, 10x amortized per CB) */
    const uint32_t dec_ncs[] = {32768u, 65536u, 131072u, 262144u};
    printf("  --- Single-step Decode rows4 Scorer (per call & x4 layers, 10x amortized) ---\n");
    for (size_t i = 0; i < sizeof(dec_ncs) / sizeof(dec_ncs[0]); ++i) {
        const uint32_t nc = dec_ncs[i];
        double t_bf16[7], t_fp4[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 10; ++k) {
                CHECK(myds_v41_gpu_indexscores_mode(sc_bf16, q_buf, w_buf, cache_bf16, nc, 32u, MYDS_INDEXER_CACHE_BF16, 0u));
            }
            CHECK(myds_metal_commit_batch());
            t_bf16[r] = (monotonic_ms() - s0) / 10.0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 10; ++k) {
                CHECK(myds_v41_gpu_indexscores_mode(sc_fp4, q_buf, w_buf, cache_fp4, nc, 32u, MYDS_INDEXER_CACHE_FP4, 0u));
            }
            CHECK(myds_metal_commit_batch());
            t_fp4[r] = (monotonic_ms() - s1) / 10.0;
        }
        qsort(t_bf16, 7, sizeof(double), cmp_double);
        qsort(t_fp4,  7, sizeof(double), cmp_double);
        printf("    DEC_rows4_%3uK     (nc=%6u): BF16 %6.3f ms (x4=%6.3f ms) | FP4 %6.3f ms (x4=%6.3f ms) | %.2fx\n",
               nc / 1024u, nc, t_bf16[3], t_bf16[3] * 4.0, t_fp4[3], t_fp4[3] * 4.0, t_bf16[3] / t_fp4[3]);
    }

    /* 4. Single-step Decode [D-L1] Compact 16K (4 YoCo layers per token, 10x amortized) */
    {
        CHECK(myds_v41_gpu_indexer_prepare_compact_blocks(scratch, blk_sel, max_blocks, 2048u, max_nc));
        double t_bf16[7], t_fp4[7];
        for (int r = 0; r < 7; ++r) {
            const double s0 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 10; ++k) {
                CHECK(myds_v41_gpu_indexscores_compact_decode_mode(scratch, q_buf, w_buf, cache_bf16, NULL, max_nc, MYDS_INDEXER_CACHE_BF16, 0u));
                CHECK(myds_v41_gpu_topk_compact_decode(top_bf16, max_nc, scratch));
            }
            CHECK(myds_metal_commit_batch());
            t_bf16[r] = (monotonic_ms() - s0) / 10.0;

            const double s1 = monotonic_ms();
            CHECK(myds_metal_begin_batch());
            for (int k = 0; k < 10; ++k) {
                CHECK(myds_v41_gpu_indexscores_compact_decode_mode(scratch, q_buf, w_buf, cache_fp4, NULL, max_nc, MYDS_INDEXER_CACHE_FP4, 0u));
                CHECK(myds_v41_gpu_topk_compact_decode(top_fp4, max_nc, scratch));
            }
            CHECK(myds_metal_commit_batch());
            t_fp4[r] = (monotonic_ms() - s1) / 10.0;
        }
        qsort(t_bf16, 7, sizeof(double), cmp_double);
        qsort(t_fp4,  7, sizeof(double), cmp_double);
        printf("    DEC_DL1_256K_16K   (nc=262144): BF16 %6.3f ms (x4=%6.3f ms) | FP4 %6.3f ms (x4=%6.3f ms) | %.2fx\n",
               t_bf16[3], t_bf16[3] * 4.0, t_fp4[3], t_fp4[3] * 4.0, t_bf16[3] / t_fp4[3]);
    }

    myds_metal_buffer_free(raw_k);
    myds_metal_buffer_free(cache_bf16);
    myds_metal_buffer_free(cache_fp4);
    myds_metal_buffer_free(q_buf);
    myds_metal_buffer_free(w_buf);
    myds_metal_buffer_free(sc_bf16);
    myds_metal_buffer_free(sc_fp4);
    myds_metal_buffer_free(mask_2d);
    myds_metal_buffer_free(blk_sel);
    myds_metal_buffer_free(top_bf16);
    myds_metal_buffer_free(top_fp4);
    myds_v41_gpu_scratch_free(scratch);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        bool run_bench = false;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--bench") == 0) run_bench = true;
        }
        printf("=== test_indexer_key_fp4 (4.25-bit E2M1FN + UE8M0, 68 B/row, 0-ULP) ===\n");
        test_env_parsing();
        setenv("MYDS_INDEXER_HEAD_SLOTS", "1", 1);
        setenv("MYDS_INDEXER_ROPE_QUANT_FUSE", "1", 1);
        if (!myds_metal_init(NULL) && !myds_metal_init("bin/myds.metallib")) {
            fprintf(stderr, "Failed to initialize Metal\n");
            return 1;
        }
        verify_pso_occupancy();
        test_exhaustive_4096_states();
        test_100k_rows_both_writers();
        test_all_gpu_scorers_0ulp();
        test_lowkv_fp4_roundtrip_and_isolation();
        if (run_bench) {
            run_fp4_microbenchmarks();
        }
        myds_metal_cleanup();
        printf("ALL test_indexer_key_fp4 CHECKS PASSED!\n");
    }
    return 0;
}
