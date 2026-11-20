#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "myds_metal.h"
#include "myds_v41_gpu.h"
#include "myds_v41_moe.h"

static uint32_t rng_state = 0x13572468u;
static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static float rand_f32(float lo, float hi) {
    return lo + (hi - lo) * ((float)(xorshift32() & 0x00ffffffu) / (float)0x01000000u);
}

static uint16_t f32_to_f16_bits(float f) {
    __fp16 h = (__fp16)f;
    uint16_t bits = 0;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

/* Fill random valid Q4_K blocks (144 bytes = d:f16, dmin:f16, scales[12], qs[128]). */
static void fill_q4_k(uint8_t *buf, size_t n_blocks) {
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *p = buf + b * 144u;
        uint16_t d    = f32_to_f16_bits(rand_f32(0.005f, 0.15f));
        uint16_t dmin = f32_to_f16_bits(rand_f32(0.002f, 0.08f));
        memcpy(p + 0u, &d, 2u);
        memcpy(p + 2u, &dmin, 2u);
        for (size_t i = 4u; i < 144u; i++) {
            p[i] = (uint8_t)(xorshift32() & 0xffu);
        }
    }
}

static void fill_f32(float *buf, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) {
        buf[i] = rand_f32(lo, hi);
    }
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int compare_memcmp_exact(const char *label, const float *ref, const float *got, size_t n) {
    if (memcmp(ref, got, n * sizeof(float)) == 0) {
        printf("PASS %s: 0-ULP memcmp==0 (%zu floats)\n", label, n);
        return 1;
    }
    size_t diffs = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        uint32_t ua = 0, ub = 0;
        memcpy(&ua, ref + i, 4u);
        memcpy(&ub, got + i, 4u);
        if (ua != ub) {
            diffs++;
            float d = fabsf(ref[i] - got[i]);
            if (d > max_abs) max_abs = d;
            if (diffs <= 4u) {
                fprintf(stderr, "  [DIFF] %s idx=%zu ref=%g (0x%08x) got=%g (0x%08x)\n",
                        label, i, ref[i], ua, got[i], ub);
            }
        }
    }
    fprintf(stderr, "FAIL %s: %zu / %zu elements differ (max_abs=%g)\n", label, diffs, n, max_abs);
    return 0;
}

/* 1. Strict fail-closed knob parsing & default-off check. */
static int test_knob_policy(const char *metallib) {
    const char *bad_vals[] = {"2", "-1", "true", "yes", "", "01", "1 "};
    for (size_t i = 0; i < sizeof(bad_vals) / sizeof(bad_vals[0]); i++) {
        setenv("MYDS_DENSE_PREFILL_WIDE", bad_vals[i], 1);
        if (myds_metal_init(metallib) != 0) {
            fprintf(stderr, "FAIL: MYDS_DENSE_PREFILL_WIDE='%s' should fail closed\n", bad_vals[i]);
            myds_metal_cleanup();
            return 0;
        }
    }
    unsetenv("MYDS_DENSE_PREFILL_WIDE");
    if (!myds_metal_init(metallib)) {
        fprintf(stderr, "FAIL: default unset init failed\n");
        return 0;
    }
    if (myds_metal_dense_prefill_wide_enabled() != 0) {
        fprintf(stderr, "FAIL: default unset should have enabled==0\n");
        myds_metal_cleanup();
        return 0;
    }
    myds_metal_cleanup();

    setenv("MYDS_DENSE_PREFILL_WIDE", "0", 1);
    if (!myds_metal_init(metallib) || myds_metal_dense_prefill_wide_enabled() != 0) {
        fprintf(stderr, "FAIL: MYDS_DENSE_PREFILL_WIDE=0 should succeed with enabled==0\n");
        myds_metal_cleanup();
        return 0;
    }
    myds_metal_cleanup();

    setenv("MYDS_DENSE_PREFILL_WIDE", "1", 1);
    if (!myds_metal_init(metallib) || myds_metal_dense_prefill_wide_enabled() != 1) {
        fprintf(stderr, "FAIL: MYDS_DENSE_PREFILL_WIDE=1 should succeed with enabled==1\n");
        myds_metal_cleanup();
        return 0;
    }
    myds_metal_cleanup();
    unsetenv("MYDS_DENSE_PREFILL_WIDE");
    printf("PASS knob policy: default=0, 0->0, 1->1, 7 invalid values fail-closed\n");
    return 1;
}

/* 2. Safe fallback on unaligned Q4_K weight offset (offset % 16 != 0). */
static int test_unaligned_fallback(const char *metallib) {
    const uint32_t out_dim = 128u, in_dim = 256u, n_tok = 64u;
    const size_t w_bytes = (size_t)out_dim * (in_dim / 256u) * 144u;
    const size_t x_bytes = (size_t)n_tok * in_dim * sizeof(float);
    const size_t y_bytes = (size_t)n_tok * out_dim * sizeof(float);

    uint8_t *h_w = (uint8_t *)malloc(w_bytes);
    float   *h_x = (float *)malloc(x_bytes);
    float   *ref = (float *)malloc(y_bytes);
    float   *got = (float *)malloc(y_bytes);
    if (!h_w || !h_x || !ref || !got) return 0;

    rng_state = 0xfeedcafeu;
    fill_q4_k(h_w, (size_t)out_dim * (in_dim / 256u));
    fill_f32(h_x, (size_t)n_tok * in_dim, -1.0f, 1.0f);

    setenv("MYDS_DENSE_PREFILL_WIDE", "0", 1);
    if (!myds_metal_init(metallib)) return 0;
    myds_metal_buffer_t *bw = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *bx = myds_metal_buffer_alloc(x_bytes);
    myds_metal_buffer_t *by = myds_metal_buffer_alloc(y_bytes);
    memcpy(myds_metal_buffer_data(bw), h_w, w_bytes);
    memcpy(myds_metal_buffer_data(bx), h_x, x_bytes);
    if (!myds_metal_mul_mm(by, bw, bx, out_dim, in_dim, n_tok, GGML_TYPE_Q4_K)) return 0;
    memcpy(ref, myds_metal_buffer_data(by), y_bytes);
    myds_metal_buffer_free(bw); myds_metal_buffer_free(bx); myds_metal_buffer_free(by);
    myds_metal_cleanup();

    setenv("MYDS_DENSE_PREFILL_WIDE", "1", 1);
    if (!myds_metal_init(metallib)) return 0;
    myds_metal_dense_prefill_wide_counters_t c0, c1;
    myds_metal_dense_prefill_wide_counters(&c0, true);

    /* Allocate +16B and create an intentionally 4-byte-unaligned sub-view. */
    myds_metal_buffer_t *bw_parent = myds_metal_buffer_alloc(w_bytes + 16u);
    memcpy((uint8_t *)myds_metal_buffer_data(bw_parent) + 4u, h_w, w_bytes);
    myds_metal_buffer_t *bw_unaligned = myds_metal_buffer_sub_view(bw_parent, 4u, w_bytes);
    bx = myds_metal_buffer_alloc(x_bytes);
    by = myds_metal_buffer_alloc(y_bytes);
    memcpy(myds_metal_buffer_data(bx), h_x, x_bytes);

    /* mul_mm_bf16 and pair_swiglu must safely refuse (return 0) and count fallback. */
    if (myds_metal_mul_mm_bf16(by, bw_unaligned, bx, out_dim, in_dim, n_tok, GGML_TYPE_Q4_K) != 0) {
        fprintf(stderr, "FAIL: mul_mm_bf16 should refuse unaligned W\n");
        return 0;
    }
    if (myds_metal_mul_mm_q4_k_pair_swiglu_bf16(by, bw_unaligned, bw_unaligned, bx, out_dim, in_dim, n_tok, 7.0f) != 0) {
        fprintf(stderr, "FAIL: pair_swiglu should refuse unaligned W\n");
        return 0;
    }
    /* mul_mm must seamlessly fall back to baseline scalar kernel and match bit-for-bit. */
    if (!myds_metal_mul_mm(by, bw_unaligned, bx, out_dim, in_dim, n_tok, GGML_TYPE_Q4_K)) {
        fprintf(stderr, "FAIL: mul_mm unaligned fallback failed\n");
        return 0;
    }
    memcpy(got, myds_metal_buffer_data(by), y_bytes);
    myds_metal_dense_prefill_wide_counters(&c1, false);

    myds_metal_buffer_free(bw_unaligned);
    myds_metal_buffer_free(bw_parent);
    myds_metal_buffer_free(bx);
    myds_metal_buffer_free(by);
    myds_metal_cleanup();
    unsetenv("MYDS_DENSE_PREFILL_WIDE");

    int ok = compare_memcmp_exact("unaligned Q4_K fallback", ref, got, (size_t)n_tok * out_dim);
    if (c1.fallbacks < 3u) {
        fprintf(stderr, "FAIL: expected >=3 unaligned fallbacks, got %llu\n", (unsigned long long)c1.fallbacks);
        ok = 0;
    }
    free(h_w); free(h_x); free(ref); free(got);
    return ok;
}

typedef struct {
    const char *name;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t n_groups; /* 1 for regular mul_mm, >1 for grouped */
    bool     test_f32;
    bool     test_bf16;
} mm_shape_t;

/* 3. Part (a): Strict 0-ULP for Q4_K wide + uint4 + fused BF16 on Q_A, KV, Q_B,
 *    AttnOutB, SharedDown, non-aligned boundary shapes, and grouped AttnOutA
 *    across c_len in {32, 48, 64, 122, 128, 312, 513, 2048}. */
static int test_part_a_mul_mm(const char *metallib) {
    const mm_shape_t shapes[] = {
        {"Q_A",        1280u, 5120u, 1u, true,  true},
        {"KV",          512u, 5120u, 1u, false, true},
        {"Q_B",       32768u, 1280u, 1u, true,  true},
        {"AttnOutB",   5120u, 8192u, 1u, false, true},
        {"SharedDown", 5120u, 2304u, 1u, false, true},
        {"Tail70",       70u,  256u, 1u, true,  true}, /* non-64 AND non-4 aligned out_dim */
        {"Tail100",     100u,  512u, 1u, true,  true}, /* non-64, 4-aligned out_dim */
        {"AttnOutA",   1024u, 4096u, 8u, true,  true}, /* grouped 8x1024x4096 */
    };
    const uint32_t c_lens[] = {32u, 48u, 64u, 122u, 128u, 312u, 513u, 2048u};
    const size_t n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    const size_t n_clens  = sizeof(c_lens) / sizeof(c_lens[0]);

    int ok = 1;
    for (size_t si = 0; si < n_shapes && ok; si++) {
        const mm_shape_t *s = &shapes[si];
        const uint32_t max_tok = 2048u;
        const size_t row_bytes = (size_t)(s->in_dim / 256u) * 144u;
        const size_t w_bytes   = (size_t)s->n_groups * s->out_dim * row_bytes;
        const size_t x_max_b   = (size_t)max_tok * s->n_groups * s->in_dim * sizeof(float);
        const size_t y_max_b   = (size_t)max_tok * s->n_groups * s->out_dim * sizeof(float);

        uint8_t *h_w = (uint8_t *)malloc(w_bytes);
        float   *h_x = (float *)malloc(x_max_b);
        float   *ref = (float *)malloc(y_max_b);
        float   *got = (float *)malloc(y_max_b);
        if (!h_w || !h_x || !ref || !got) {
            fprintf(stderr, "FAIL: host OOM for %s\n", s->name);
            return 0;
        }
        rng_state = 0x10203040u + (uint32_t)si * 997u;
        fill_q4_k(h_w, (size_t)s->n_groups * s->out_dim * (s->in_dim / 256u));
        fill_f32(h_x, (size_t)max_tok * s->n_groups * s->in_dim, -1.5f, 1.5f);

        for (size_t ci = 0; ci < n_clens && ok; ci++) {
            const uint32_t n_tok = c_lens[ci];
            const size_t n_out = (size_t)n_tok * s->n_groups * s->out_dim;
            const size_t x_b   = (size_t)n_tok * s->n_groups * s->in_dim * sizeof(float);
            const size_t y_b   = n_out * sizeof(float);

            for (int mode = 0; mode < 2 && ok; mode++) {
                const bool round_bf16 = (mode == 1);
                if (!round_bf16 && !s->test_f32) continue;
                if (round_bf16 && !s->test_bf16) continue;

                /* Arm 0: Baseline MYDS_DENSE_PREFILL_WIDE=0 */
                setenv("MYDS_DENSE_PREFILL_WIDE", "0", 1);
                if (!myds_metal_init(metallib)) { ok = 0; break; }
                myds_metal_buffer_t *bw = myds_metal_buffer_alloc(w_bytes);
                myds_metal_buffer_t *bx = myds_metal_buffer_alloc(x_b);
                myds_metal_buffer_t *by = myds_metal_buffer_alloc(y_b);
                memcpy(myds_metal_buffer_data(bw), h_w, w_bytes);
                memcpy(myds_metal_buffer_data(bx), h_x, x_b);
                int r0;
                if (s->n_groups > 1u) {
                    r0 = myds_metal_mul_mm_grouped(by, bw, bx, s->out_dim, s->in_dim, s->n_groups, n_tok, GGML_TYPE_Q4_K);
                } else {
                    r0 = myds_metal_mul_mm(by, bw, bx, s->out_dim, s->in_dim, n_tok, GGML_TYPE_Q4_K);
                }
                if (r0 && round_bf16) {
                    r0 = myds_metal_v41_bf16(by, (uint32_t)n_out);
                }
                if (!r0) {
                    fprintf(stderr, "FAIL: arm0 %s c_len=%u bf16=%d\n", s->name, n_tok, (int)round_bf16);
                    ok = 0;
                } else {
                    memcpy(ref, myds_metal_buffer_data(by), y_b);
                }
                myds_metal_buffer_free(bw); myds_metal_buffer_free(bx); myds_metal_buffer_free(by);
                myds_metal_cleanup();
                if (!ok) break;

                /* Arm 1: Candidate MYDS_DENSE_PREFILL_WIDE=1 */
                setenv("MYDS_DENSE_PREFILL_WIDE", "1", 1);
                if (!myds_metal_init(metallib)) { ok = 0; break; }
                bw = myds_metal_buffer_alloc(w_bytes);
                bx = myds_metal_buffer_alloc(x_b);
                by = myds_metal_buffer_alloc(y_b);
                memcpy(myds_metal_buffer_data(bw), h_w, w_bytes);
                memcpy(myds_metal_buffer_data(bx), h_x, x_b);
                int r1;
                if (s->n_groups > 1u) {
                    r1 = round_bf16
                        ? myds_metal_mul_mm_grouped_bf16(by, bw, bx, s->out_dim, s->in_dim, s->n_groups, n_tok, GGML_TYPE_Q4_K)
                        : myds_metal_mul_mm_grouped(by, bw, bx, s->out_dim, s->in_dim, s->n_groups, n_tok, GGML_TYPE_Q4_K);
                } else {
                    r1 = round_bf16
                        ? myds_metal_mul_mm_bf16(by, bw, bx, s->out_dim, s->in_dim, n_tok, GGML_TYPE_Q4_K)
                        : myds_metal_mul_mm(by, bw, bx, s->out_dim, s->in_dim, n_tok, GGML_TYPE_Q4_K);
                }
                if (!r1) {
                    fprintf(stderr, "FAIL: arm1 %s c_len=%u bf16=%d\n", s->name, n_tok, (int)round_bf16);
                    ok = 0;
                } else {
                    memcpy(got, myds_metal_buffer_data(by), y_b);
                }
                myds_metal_buffer_free(bw); myds_metal_buffer_free(bx); myds_metal_buffer_free(by);
                myds_metal_cleanup();
                if (!ok) break;

                char label[128];
                snprintf(label, sizeof(label), "[P-D1] %s (%ux%ux%u) c_len=%u %s",
                         s->name, s->n_groups, s->out_dim, s->in_dim, n_tok,
                         round_bf16 ? "bf16" : "f32");
                if (!compare_memcmp_exact(label, ref, got, n_out)) ok = 0;
            }
        }
        free(h_w); free(h_x); free(ref); free(got);
    }
    unsetenv("MYDS_DENSE_PREFILL_WIDE");
    return ok;
}

/* 4. Part (b): Strict 0-ULP for fused shared_gate+up+SwiGLU vs
 *    separate mul_mm + v41_bf16 + swiglu_flat. */
static int test_part_b_shared_pair(const char *metallib) {
    struct { const char *name; uint32_t out_dim; uint32_t in_dim; float clamp_val; } cases[] = {
        {"SharedPair_V41",   2304u, 5120u, 7.0f},
        {"SharedPair_NoClp", 2304u, 5120u, 0.0f},
        {"SharedPair_Tail50",  50u,  256u, 7.0f},
    };
    const uint32_t c_lens[] = {32u, 48u, 64u, 122u, 128u, 312u, 513u, 2048u};
    int ok = 1;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]) && ok; ci++) {
        const uint32_t out_dim = cases[ci].out_dim;
        const uint32_t in_dim  = cases[ci].in_dim;
        const float clamp_val  = cases[ci].clamp_val;
        const uint32_t max_tok = (ci == 0) ? 2048u : 122u;
        const size_t w_bytes   = (size_t)out_dim * (in_dim / 256u) * 144u;
        const size_t x_max_b   = (size_t)max_tok * in_dim * sizeof(float);
        const size_t y_max_b   = (size_t)max_tok * out_dim * sizeof(float);

        uint8_t *h_wg = (uint8_t *)malloc(w_bytes);
        uint8_t *h_wu = (uint8_t *)malloc(w_bytes);
        float   *h_x  = (float *)malloc(x_max_b);
        float   *ref  = (float *)malloc(y_max_b);
        float   *got  = (float *)malloc(y_max_b);
        if (!h_wg || !h_wu || !h_x || !ref || !got) return 0;

        rng_state = 0xabcdef01u + (uint32_t)ci * 313u;
        fill_q4_k(h_wg, (size_t)out_dim * (in_dim / 256u));
        fill_q4_k(h_wu, (size_t)out_dim * (in_dim / 256u));
        fill_f32(h_x, (size_t)max_tok * in_dim, -2.5f, 2.5f); /* wide range exercises clamp */

        for (size_t ti = 0; ti < sizeof(c_lens) / sizeof(c_lens[0]) && ok; ti++) {
            const uint32_t n_tok = c_lens[ti];
            if (n_tok > max_tok) continue;
            const size_t n_out = (size_t)n_tok * out_dim;
            const size_t x_b   = (size_t)n_tok * in_dim * sizeof(float);
            const size_t y_b   = n_out * sizeof(float);

            /* Arm 0: separate mul_mm(gate) + bf16 + mul_mm(up) + bf16 + swiglu_flat */
            setenv("MYDS_DENSE_PREFILL_WIDE", "0", 1);
            if (!myds_metal_init(metallib)) { ok = 0; break; }
            myds_metal_buffer_t *bwg = myds_metal_buffer_alloc(w_bytes);
            myds_metal_buffer_t *bwu = myds_metal_buffer_alloc(w_bytes);
            myds_metal_buffer_t *bx  = myds_metal_buffer_alloc(x_b);
            myds_metal_buffer_t *bg  = myds_metal_buffer_alloc(y_b);
            myds_metal_buffer_t *bu  = myds_metal_buffer_alloc(y_b);
            myds_metal_buffer_t *bm  = myds_metal_buffer_alloc(y_b);
            memcpy(myds_metal_buffer_data(bwg), h_wg, w_bytes);
            memcpy(myds_metal_buffer_data(bwu), h_wu, w_bytes);
            memcpy(myds_metal_buffer_data(bx),  h_x,  x_b);
            if (!myds_metal_mul_mm(bg, bwg, bx, out_dim, in_dim, n_tok, GGML_TYPE_Q4_K) ||
                !myds_metal_v41_bf16(bg, (uint32_t)n_out) ||
                !myds_metal_mul_mm(bu, bwu, bx, out_dim, in_dim, n_tok, GGML_TYPE_Q4_K) ||
                !myds_metal_v41_bf16(bu, (uint32_t)n_out) ||
                !myds_v41_shared_swiglu(bm, bg, bu, (uint32_t)n_out, clamp_val)) {
                fprintf(stderr, "FAIL: arm0 %s c_len=%u\n", cases[ci].name, n_tok);
                ok = 0;
            } else {
                memcpy(ref, myds_metal_buffer_data(bm), y_b);
            }
            myds_metal_buffer_free(bwg); myds_metal_buffer_free(bwu); myds_metal_buffer_free(bx);
            myds_metal_buffer_free(bg);  myds_metal_buffer_free(bu);  myds_metal_buffer_free(bm);
            myds_metal_cleanup();
            if (!ok) break;

            /* Arm 1: single fused kernel_mul_mm_q4_K_pair_swiglu_bf16 */
            setenv("MYDS_DENSE_PREFILL_WIDE", "1", 1);
            if (!myds_metal_init(metallib)) { ok = 0; break; }
            bwg = myds_metal_buffer_alloc(w_bytes);
            bwu = myds_metal_buffer_alloc(w_bytes);
            bx  = myds_metal_buffer_alloc(x_b);
            bm  = myds_metal_buffer_alloc(y_b);
            memcpy(myds_metal_buffer_data(bwg), h_wg, w_bytes);
            memcpy(myds_metal_buffer_data(bwu), h_wu, w_bytes);
            memcpy(myds_metal_buffer_data(bx),  h_x,  x_b);
            if (!myds_metal_mul_mm_q4_k_pair_swiglu_bf16(bm, bwg, bwu, bx, out_dim, in_dim, n_tok, clamp_val)) {
                fprintf(stderr, "FAIL: arm1 %s c_len=%u\n", cases[ci].name, n_tok);
                ok = 0;
            } else {
                memcpy(got, myds_metal_buffer_data(bm), y_b);
            }
            myds_metal_buffer_free(bwg); myds_metal_buffer_free(bwu);
            myds_metal_buffer_free(bx);  myds_metal_buffer_free(bm);
            myds_metal_cleanup();
            if (!ok) break;

            char label[128];
            snprintf(label, sizeof(label), "[P-D1] %s (%ux%u clamp=%.1f) c_len=%u",
                     cases[ci].name, out_dim, in_dim, clamp_val, n_tok);
            if (!compare_memcmp_exact(label, ref, got, n_out)) ok = 0;
        }
        free(h_wg); free(h_wu); free(h_x); free(ref); free(got);
    }
    unsetenv("MYDS_DENSE_PREFILL_WIDE");
    return ok;
}

/* 5. Part (c): Strict 0-ULP for [P-R1] Site 1: Q_B fused BF16 + 64-dim rope
 *    vs Q_B + rope_round across c_len and compressed={false,true}. */
static int test_part_c_qb_rope(const char *metallib) {
    const uint32_t heads = 64u, head_dim = 512u, out_dim = heads * head_dim, in_dim = 1280u;
    const uint32_t max_tok = 2048u;
    const uint32_t c_lens[] = {32u, 48u, 64u, 122u, 128u, 312u, 513u, 2048u};
    const size_t w_bytes = (size_t)out_dim * (in_dim / 256u) * 144u;
    const size_t x_max_b = (size_t)max_tok * in_dim * sizeof(float);
    const size_t y_max_b = (size_t)max_tok * out_dim * sizeof(float);

    uint8_t *h_w = (uint8_t *)malloc(w_bytes);
    float   *h_x = (float *)malloc(x_max_b);
    float   *ref = (float *)malloc(y_max_b);
    float   *got = (float *)malloc(y_max_b);
    if (!h_w || !h_x || !ref || !got) return 0;

    rng_state = 0x55aa33ccu;
    fill_q4_k(h_w, (size_t)out_dim * (in_dim / 256u));
    fill_f32(h_x, (size_t)max_tok * in_dim, -1.5f, 1.5f);

    int ok = 1;
    for (size_t ti = 0; ti < sizeof(c_lens) / sizeof(c_lens[0]) && ok; ti++) {
        const uint32_t n_tok = c_lens[ti];
        const bool compressed = (ti & 1u) != 0;
        const uint32_t start_pos = compressed ? 126976u : 0u;
        const size_t x_b = (size_t)n_tok * in_dim * sizeof(float);
        const size_t y_b = (size_t)n_tok * out_dim * sizeof(float);

        /* Arm 0: Q_B f32 + full-512 rope_batch_round */
        setenv("MYDS_DENSE_PREFILL_WIDE", "0", 1);
        if (!myds_metal_init(metallib)) { ok = 0; break; }
        myds_metal_buffer_t *bw = myds_metal_buffer_alloc(w_bytes);
        myds_metal_buffer_t *bx = myds_metal_buffer_alloc(x_b);
        myds_metal_buffer_t *bq = myds_metal_buffer_alloc(y_b);
        memcpy(myds_metal_buffer_data(bw), h_w, w_bytes);
        memcpy(myds_metal_buffer_data(bx), h_x, x_b);
        if (!myds_metal_mul_mm(bq, bw, bx, out_dim, in_dim, n_tok, GGML_TYPE_Q4_K) ||
            !myds_v41_gpu_rope_batch_round(bq, head_dim, heads, start_pos, n_tok, compressed, false)) {
            fprintf(stderr, "FAIL: arm0 Q_B+rope_round c_len=%u\n", n_tok);
            ok = 0;
        } else {
            memcpy(ref, myds_metal_buffer_data(bq), y_b);
        }
        myds_metal_buffer_free(bw); myds_metal_buffer_free(bx); myds_metal_buffer_free(bq);
        myds_metal_cleanup();
        if (!ok) break;

        /* Arm 1: Q_B fused BF16 + 64-dim rope_batch */
        setenv("MYDS_DENSE_PREFILL_WIDE", "1", 1);
        if (!myds_metal_init(metallib)) { ok = 0; break; }
        bw = myds_metal_buffer_alloc(w_bytes);
        bx = myds_metal_buffer_alloc(x_b);
        bq = myds_metal_buffer_alloc(y_b);
        memcpy(myds_metal_buffer_data(bw), h_w, w_bytes);
        memcpy(myds_metal_buffer_data(bx), h_x, x_b);
        if (!myds_metal_mul_mm_bf16(bq, bw, bx, out_dim, in_dim, n_tok, GGML_TYPE_Q4_K) ||
            !myds_v41_gpu_rope_batch(bq, head_dim, heads, start_pos, n_tok, compressed, false)) {
            fprintf(stderr, "FAIL: arm1 Q_B_bf16+rope64 c_len=%u\n", n_tok);
            ok = 0;
        } else {
            myds_metal_dense_prefill_wide_note_qb_rope64();
            memcpy(got, myds_metal_buffer_data(bq), y_b);
        }
        myds_metal_buffer_free(bw); myds_metal_buffer_free(bx); myds_metal_buffer_free(bq);
        myds_metal_cleanup();
        if (!ok) break;

        char label[128];
        snprintf(label, sizeof(label), "[P-R1] Q_B_bf16+rope64 c_len=%u comp=%d pos=%u",
                 n_tok, (int)compressed, start_pos);
        if (!compare_memcmp_exact(label, ref, got, (size_t)n_tok * out_dim)) ok = 0;
    }

    myds_metal_dense_prefill_wide_counters_t c;
    myds_metal_dense_prefill_wide_counters(&c, false);
    if (c.requests == 0 || c.wide_encoded == 0 || c.narrow_bf16_encoded == 0 ||
        c.grouped_wide_encoded == 0 || c.shared_pair_encoded == 0 ||
        c.qb_rope64_encoded == 0 || c.failures != 0) {
        fprintf(stderr, "FAIL: counters mismatch req=%llu wide=%llu narrow=%llu grp=%llu pair=%llu rope64=%llu fb=%llu fail=%llu\n",
                (unsigned long long)c.requests, (unsigned long long)c.wide_encoded,
                (unsigned long long)c.narrow_bf16_encoded, (unsigned long long)c.grouped_wide_encoded,
                (unsigned long long)c.shared_pair_encoded, (unsigned long long)c.qb_rope64_encoded,
                (unsigned long long)c.fallbacks, (unsigned long long)c.failures);
        ok = 0;
    } else {
        printf("PASS counters: req=%llu wide=%llu narrow_bf16=%llu grp_wide=%llu shared_pair=%llu qb_rope64=%llu fallbacks=%llu failures=%llu\n",
               (unsigned long long)c.requests, (unsigned long long)c.wide_encoded,
               (unsigned long long)c.narrow_bf16_encoded, (unsigned long long)c.grouped_wide_encoded,
               (unsigned long long)c.shared_pair_encoded, (unsigned long long)c.qb_rope64_encoded,
               (unsigned long long)c.fallbacks, (unsigned long long)c.failures);
    }

    free(h_w); free(h_x); free(ref); free(got);
    unsetenv("MYDS_DENSE_PREFILL_WIDE");
    return ok;
}

/* M4 microbenchmark comparing MYDS_DENSE_PREFILL_WIDE=0 vs 1 across V4.1 sites. */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static double median_ms(double *arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    return arr[n / 2];
}

typedef enum {
    SITE_MM_BF16 = 0,
    SITE_QB_ROPE,
    SITE_ATTN_OUT_A,
    SITE_SHARED_PAIR,
} bench_site_kind_t;

typedef struct {
    const char       *name;
    bench_site_kind_t kind;
    uint32_t          out_dim;
    uint32_t          in_dim;
    uint32_t          n_groups;
} bench_site_t;

static double run_site_one_sample_ms(const char *metallib, const bench_site_t *site,
                                     uint32_t n_tok, int wide_on,
                                     const uint8_t *h_w, const uint8_t *h_w2, const float *h_x) {
    setenv("MYDS_DENSE_PREFILL_WIDE", wide_on ? "1" : "0", 1);
    if (!myds_metal_init(metallib)) return -1.0;

    const size_t row_bytes = (size_t)(site->in_dim / 256u) * 144u;
    const size_t w_bytes   = (size_t)site->n_groups * site->out_dim * row_bytes;
    const size_t x_bytes   = (size_t)n_tok * site->n_groups * site->in_dim * sizeof(float);
    const size_t y_bytes   = (size_t)n_tok * site->n_groups * site->out_dim * sizeof(float);
    const uint32_t n_out   = n_tok * site->n_groups * site->out_dim;

    myds_metal_buffer_t *bw  = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *bw2 = (site->kind == SITE_SHARED_PAIR) ? myds_metal_buffer_alloc(w_bytes) : NULL;
    myds_metal_buffer_t *bx  = myds_metal_buffer_alloc(x_bytes);
    myds_metal_buffer_t *by  = myds_metal_buffer_alloc(y_bytes);
    myds_metal_buffer_t *bg  = (site->kind == SITE_SHARED_PAIR) ? myds_metal_buffer_alloc(y_bytes) : NULL;
    myds_metal_buffer_t *bu  = (site->kind == SITE_SHARED_PAIR) ? myds_metal_buffer_alloc(y_bytes) : NULL;
    memcpy(myds_metal_buffer_data(bw), h_w, w_bytes);
    if (bw2) memcpy(myds_metal_buffer_data(bw2), h_w2, w_bytes);
    memcpy(myds_metal_buffer_data(bx), h_x, x_bytes);

    const int n_iters = (n_tok >= 2048u) ? 4 : (n_tok >= 312u ? 8 : 16);
    double res_ms = 0.0;

    for (int phase = 0; phase < 2; phase++) {
        const int runs = (phase == 0) ? 1 : n_iters;
        myds_metal_phase_reset();
        double t0 = now_ms();
        myds_metal_begin_batch();
        for (int it = 0; it < runs; it++) {
            switch (site->kind) {
            case SITE_MM_BF16:
                if (wide_on) {
                    myds_metal_mul_mm_bf16(by, bw, bx, site->out_dim, site->in_dim, n_tok, GGML_TYPE_Q4_K);
                } else {
                    myds_metal_mul_mm(by, bw, bx, site->out_dim, site->in_dim, n_tok, GGML_TYPE_Q4_K);
                    myds_metal_v41_bf16(by, n_out);
                }
                break;
            case SITE_QB_ROPE:
                if (wide_on) {
                    myds_metal_mul_mm_bf16(by, bw, bx, site->out_dim, site->in_dim, n_tok, GGML_TYPE_Q4_K);
                    myds_v41_gpu_rope_batch(by, 512u, 64u, 0u, n_tok, true, false);
                } else {
                    myds_metal_mul_mm(by, bw, bx, site->out_dim, site->in_dim, n_tok, GGML_TYPE_Q4_K);
                    myds_v41_gpu_rope_batch_round(by, 512u, 64u, 0u, n_tok, true, false);
                }
                break;
            case SITE_ATTN_OUT_A:
                if (wide_on) {
                    myds_metal_mul_mm_grouped_bf16(by, bw, bx, site->out_dim, site->in_dim, site->n_groups, n_tok, GGML_TYPE_Q4_K);
                } else {
                    myds_metal_mul_mm_grouped(by, bw, bx, site->out_dim, site->in_dim, site->n_groups, n_tok, GGML_TYPE_Q4_K);
                    myds_metal_v41_bf16(by, n_out);
                }
                break;
            case SITE_SHARED_PAIR:
                if (wide_on) {
                    myds_metal_mul_mm_q4_k_pair_swiglu_bf16(by, bw, bw2, bx, site->out_dim, site->in_dim, n_tok, 7.0f);
                } else {
                    myds_metal_mul_mm(bg, bw, bx, site->out_dim, site->in_dim, n_tok, GGML_TYPE_Q4_K);
                    myds_metal_v41_bf16(bg, n_out);
                    myds_metal_mul_mm(bu, bw2, bx, site->out_dim, site->in_dim, n_tok, GGML_TYPE_Q4_K);
                    myds_metal_v41_bf16(bu, n_out);
                    myds_v41_shared_swiglu(by, bg, bu, n_out, 7.0f);
                }
                break;
            }
        }
        myds_metal_commit_batch();
        double wall_ms = (now_ms() - t0) / (double)runs;
        double gpu_s = 0.0, wait_s = 0.0;
        uint64_t cbs = 0;
        myds_metal_phase_take(&gpu_s, &wait_s, &cbs);
        if (phase == 1) {
            res_ms = (gpu_s > 0.0) ? (gpu_s * 1000.0 / (double)runs) : wall_ms;
        }
    }

    myds_metal_buffer_free(bw);
    if (bw2) myds_metal_buffer_free(bw2);
    myds_metal_buffer_free(bx);
    myds_metal_buffer_free(by);
    if (bg) myds_metal_buffer_free(bg);
    if (bu) myds_metal_buffer_free(bu);
    myds_metal_cleanup();
    unsetenv("MYDS_DENSE_PREFILL_WIDE");
    return res_ms;
}

static void run_site_interleaved_ms(const char *metallib, const bench_site_t *site,
                                    uint32_t n_tok,
                                    const uint8_t *h_w, const uint8_t *h_w2, const float *h_x,
                                    double *out_off_ms, double *out_on_ms) {
    const int n_pairs = 5;
    double s_off[5], s_on[5];
    for (int p = 0; p < n_pairs; p++) {
        if ((p & 1) == 0) {
            s_off[p] = run_site_one_sample_ms(metallib, site, n_tok, 0, h_w, h_w2, h_x);
            s_on[p]  = run_site_one_sample_ms(metallib, site, n_tok, 1, h_w, h_w2, h_x);
        } else {
            s_on[p]  = run_site_one_sample_ms(metallib, site, n_tok, 1, h_w, h_w2, h_x);
            s_off[p] = run_site_one_sample_ms(metallib, site, n_tok, 0, h_w, h_w2, h_x);
        }
    }
    *out_off_ms = median_ms(s_off, n_pairs);
    *out_on_ms  = median_ms(s_on, n_pairs);
}

static void run_bench(const char *metallib) {
    const bench_site_t sites[] = {
        {"Q_A (1280x5120)+bf16",        SITE_MM_BF16,     1280u, 5120u, 1u},
        {"KV  (512x5120)+bf16",         SITE_MM_BF16,      512u, 5120u, 1u},
        {"Q_B (32768x1280)+RoPE",       SITE_QB_ROPE,    32768u, 1280u, 1u},
        {"AttnOutA (8x1024x4096)+bf16", SITE_ATTN_OUT_A,  1024u, 4096u, 8u},
        {"AttnOutB (5120x8192)+bf16",   SITE_MM_BF16,     5120u, 8192u, 1u},
        {"Shared GU+SwiGLU (2x2304x5120)", SITE_SHARED_PAIR, 2304u, 5120u, 1u},
        {"SharedDown (5120x2304)+bf16", SITE_MM_BF16,     5120u, 2304u, 1u},
    };
    const uint32_t bench_clens[] = {122u, 312u, 513u, 2048u};
    const size_t n_sites = sizeof(sites) / sizeof(sites[0]);

    /* Pre-allocate max host buffers once (max site weight = 163,840 Q4_K blocks = 23.6 MB). */
    const size_t max_w_blocks = (size_t)5120u * (8192u / 256u); /* == 32768 * (1280/256) == 163840 */
    const size_t max_w = max_w_blocks * 144u;
    const size_t max_x = (size_t)2048u * 8u * 4096u * sizeof(float);
    uint8_t *h_w  = (uint8_t *)malloc(max_w);
    uint8_t *h_w2 = (uint8_t *)malloc((size_t)2304u * (5120u / 256u) * 144u);
    float   *h_x  = (float *)malloc(max_x);
    if (!h_w || !h_w2 || !h_x) { fprintf(stderr, "bench OOM\n"); return; }

    rng_state = 0x20261120u;
    fill_q4_k(h_w,  max_w_blocks);
    fill_q4_k(h_w2, (size_t)2304u * (5120u / 256u));
    fill_f32(h_x, (size_t)2048u * 8u * 4096u, -1.5f, 1.5f);

    printf("\n=== M4 Dense Q4_K Prefill Wide Microbenchmark (GPU time, median of 5 trials) ===\n");
    for (size_t ci = 0; ci < sizeof(bench_clens) / sizeof(bench_clens[0]); ci++) {
        const uint32_t n_tok = bench_clens[ci];
        double sum_off = 0.0, sum_on = 0.0, sum_gflops = 0.0;
        printf("\n--- c_len = %u ---\n", n_tok);
        printf("%-32s | %9s | %9s | %8s | %7s -> %7s\n",
               "Site", "Off (ms)", "On (ms)", "Speedup", "Off TF/s", "On TF/s");
        printf("---------------------------------+-----------+-----------+----------+--------------------\n");
        for (size_t si = 0; si < n_sites; si++) {
            const bench_site_t *s = &sites[si];
            const double mult = (s->kind == SITE_SHARED_PAIR) ? 2.0 : 1.0;
            const double gflops = 2.0 * mult * (double)s->n_groups * (double)s->out_dim *
                                  (double)s->in_dim * (double)n_tok * 1e-9;
            double ms_off = 0.0, ms_on = 0.0;
            run_site_interleaved_ms(metallib, s, n_tok, h_w, h_w2, h_x, &ms_off, &ms_on);
            double sp = (ms_on > 0.0) ? (ms_off / ms_on) : 0.0;
            double tf_off = (ms_off > 0.0) ? (gflops / ms_off) : 0.0;
            double tf_on  = (ms_on  > 0.0) ? (gflops / ms_on)  : 0.0;
            printf("%-32s | %9.3f | %9.3f | %7.2fx | %7.2f -> %7.2f\n",
                   s->name, ms_off, ms_on, sp, tf_off, tf_on);
            sum_off += ms_off;
            sum_on  += ms_on;
            sum_gflops += gflops;
        }
        printf("---------------------------------+-----------+-----------+----------+--------------------\n");
        printf("%-32s | %9.3f | %9.3f | %7.2fx | %7.2f -> %7.2f (delta %.1f%%)\n",
               "TOTAL (all 7 V4.1 Q4_K sites)",
               sum_off, sum_on, sum_off / sum_on,
               sum_gflops / sum_off, sum_gflops / sum_on,
               (sum_on - sum_off) / sum_off * 100.0);
    }
    free(h_w); free(h_w2); free(h_x);
}

int main(int argc, char **argv) {
    const char *metallib = "bin/myds.metallib";
    int do_bench = 0;
    int bench_only = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) do_bench = 1;
        else if (strcmp(argv[i], "--bench-only") == 0) { do_bench = 1; bench_only = 1; }
        else metallib = argv[i];
    }

    int ok = 1;
    if (!bench_only) {
        if (!test_knob_policy(metallib))              ok = 0;
        if (ok && !test_unaligned_fallback(metallib)) ok = 0;
        if (ok && !test_part_a_mul_mm(metallib))      ok = 0;
        if (ok && !test_part_b_shared_pair(metallib)) ok = 0;
        if (ok && !test_part_c_qb_rope(metallib))     ok = 0;

        printf("\n=== SUMMARY: %s ===\n", ok ? "ALL PASS (STRICT 0-ULP)" : "FAILED");
    }
    if (ok && do_bench) {
        run_bench(metallib);
    }
    return ok ? 0 : 1;
}
