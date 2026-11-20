#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "myds_metal.h"

static uint32_t g_rng = 0x2468ace1u;
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

static float make_special_or_rand(size_t idx, float scale) {
    switch (idx % 257u) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return 1.40129846e-45f;   /* min positive subnormal */
        case 3: return -1.40129846e-45f;  /* min negative subnormal */
        case 4: return 5.87747175e-39f;   /* subnormal */
        case 5: return -5.87747175e-39f;
        case 6: return 1.0e-20f;
        case 7: return -1.0e-20f;
        case 8: return 128.5f * scale;
        case 9: return -128.5f * scale;
        case 10: {
            /* BF16 tie-to-even boundary */
            uint32_t bits = (idx & 1u) ? 0x3f818000u : 0x3f808000u;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        case 11: {
            uint32_t bits = 0xbf808000u;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        default:
            return rand_uniform(-2.5f * scale, 2.5f * scale);
    }
}

static int compare_exact(const char *label, const float *ref, const float *got, size_t n) {
    if (memcmp(ref, got, n * sizeof(float)) == 0) {
        return 1;
    }
    size_t diffs = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        uint32_t ua = 0, ub = 0;
        memcpy(&ua, ref + i, 4);
        memcpy(&ub, got + i, 4);
        if (ua != ub) {
            diffs++;
            float d = fabsf(ref[i] - got[i]);
            if (d > max_abs) max_abs = d;
            if (diffs <= 5) {
                fprintf(stderr, "  [DIFF] %s idx=%zu ref=%.9g (0x%08x) got=%.9g (0x%08x)\n",
                        label, i, ref[i], ua, got[i], ub);
            }
        }
    }
    fprintf(stderr, "FAIL %s: %zu / %zu floats differ (max_abs=%g)\n", label, diffs, n, max_abs);
    return 0;
}

static int test_knob_parsing(void) {
    printf("[1/4] Testing MYDS_HC_MIX_FAST fail-closed parsing...\n");

    unsetenv("MYDS_HC_MIX_FAST");
    if (myds_metal_hc_mix_fast_parse_mode() != 0) {
        fprintf(stderr, "FAIL: unset MYDS_HC_MIX_FAST must return 0\n");
        return 0;
    }
    setenv("MYDS_HC_MIX_FAST", "0", 1);
    if (myds_metal_hc_mix_fast_parse_mode() != 0) {
        fprintf(stderr, "FAIL: MYDS_HC_MIX_FAST=0 must return 0\n");
        return 0;
    }
    setenv("MYDS_HC_MIX_FAST", "1", 1);
    if (myds_metal_hc_mix_fast_parse_mode() != 1) {
        fprintf(stderr, "FAIL: MYDS_HC_MIX_FAST=1 must return 1\n");
        return 0;
    }

    const char *bad_vals[] = {"", "2", "-1", "true", "auto", "01", "1 "};
    for (size_t i = 0; i < sizeof(bad_vals) / sizeof(bad_vals[0]); i++) {
        setenv("MYDS_HC_MIX_FAST", bad_vals[i], 1);
        if (myds_metal_hc_mix_fast_parse_mode() != -1) {
            fprintf(stderr, "FAIL: MYDS_HC_MIX_FAST='%s' must return -1\n", bad_vals[i]);
            return 0;
        }
        if (myds_metal_init("bin") != 0) {
            fprintf(stderr, "FAIL: myds_metal_init must fail closed for MYDS_HC_MIX_FAST='%s'\n", bad_vals[i]);
            myds_metal_cleanup();
            return 0;
        }
    }

    /* Default unset must initialize with MYDS_HC_MIX_FAST disabled */
    unsetenv("MYDS_HC_MIX_FAST");
    if (!myds_metal_init("bin")) {
        fprintf(stderr, "FAIL: myds_metal_init failed with default MYDS_HC_MIX_FAST\n");
        return 0;
    }
    if (myds_metal_hc_mix_fast_enabled() != 0) {
        fprintf(stderr, "FAIL: MYDS_HC_MIX_FAST must default to 0\n");
        myds_metal_cleanup();
        return 0;
    }
    myds_metal_cleanup();
    printf("  PASS: MYDS_HC_MIX_FAST defaults to 0 and fails closed on all invalid values.\n");
    return 1;
}

typedef struct {
    float *after_attn;
    float *x_norm_s1;
    float *x_norm_s6;
    float *sum_s6;
    float *flat_scale_s1;
    float *flat_scale_s6;
    float *hc_mix_s1;
    float *hc_mix_s6;
    float *hc_split_s6;
    float *hc_mix_tail;
    float *x_norm_4096;
    float *flat_scale_4096;
} run_outputs_t;

static void alloc_outputs(run_outputs_t *o, uint32_t c_len, uint32_t dim) {
    o->after_attn      = (float *)calloc((size_t)c_len * 4u * dim, sizeof(float));
    o->x_norm_s1       = (float *)calloc((size_t)c_len * dim, sizeof(float));
    o->x_norm_s6       = (float *)calloc((size_t)c_len * dim, sizeof(float));
    o->sum_s6          = (float *)calloc((size_t)c_len * dim, sizeof(float));
    o->flat_scale_s1   = (float *)calloc((size_t)c_len, sizeof(float));
    o->flat_scale_s6   = (float *)calloc((size_t)c_len, sizeof(float));
    o->hc_mix_s1       = (float *)calloc((size_t)c_len * 24u, sizeof(float));
    o->hc_mix_s6       = (float *)calloc((size_t)c_len * 24u, sizeof(float));
    o->hc_split_s6     = (float *)calloc((size_t)c_len * 24u, sizeof(float));
    o->hc_mix_tail     = (float *)calloc((size_t)c_len * 24u, sizeof(float));
    o->x_norm_4096     = (float *)calloc((size_t)c_len * 4096u, sizeof(float));
    o->flat_scale_4096 = (float *)calloc((size_t)c_len, sizeof(float));
}

static void free_outputs(run_outputs_t *o) {
    free(o->after_attn);
    free(o->x_norm_s1);
    free(o->x_norm_s6);
    free(o->sum_s6);
    free(o->flat_scale_s1);
    free(o->flat_scale_s6);
    free(o->hc_mix_s1);
    free(o->hc_mix_s6);
    free(o->hc_split_s6);
    free(o->hc_mix_tail);
    free(o->x_norm_4096);
    free(o->flat_scale_4096);
}

static int run_arm(
    int fast_mode,
    uint32_t c_len,
    uint32_t dim,
    const float *h_res,
    const float *h_block,
    const float *h_pre16,
    const float *h_split96,
    const float *h_norm_w,
    const _Float16 *h_w_f16,
    const float *h_hc_scale,
    const float *h_hc_base,
    const float *h_res_tail,
    const _Float16 *h_w_tail_f16,
    const float *h_res_4096,
    const float *h_norm_w_4096,
    run_outputs_t *out,
    myds_metal_hc_mix_fast_counters_t *counters_out)
{
    const uint32_t flat_dim = 4u * dim;
    const uint32_t tail_in_dim = flat_dim + 4u; /* 20484: exercises non-32-aligned tail loop */
    setenv("MYDS_HC_MIX_FAST", fast_mode ? "1" : "0", 1);
    if (!myds_metal_init("bin")) {
        fprintf(stderr, "FAIL: myds_metal_init(fast=%d) failed\n", fast_mode);
        return 0;
    }
    if (myds_metal_hc_mix_fast_enabled() != fast_mode) {
        fprintf(stderr, "FAIL: enabled mismatch (expected %d, got %d)\n",
                fast_mode, myds_metal_hc_mix_fast_enabled());
        myds_metal_cleanup();
        return 0;
    }

    myds_metal_hc_mix_fast_counters_t dummy_reset;
    myds_metal_hc_mix_fast_counters(&dummy_reset, true);

    myds_metal_buffer_t *b_res      = myds_metal_buffer_alloc((size_t)c_len * flat_dim * sizeof(float));
    myds_metal_buffer_t *b_block    = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_pre16    = myds_metal_buffer_alloc((size_t)c_len * 4u * sizeof(float));
    myds_metal_buffer_t *b_split96  = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));
    myds_metal_buffer_t *b_norm_w   = myds_metal_buffer_alloc((size_t)dim * sizeof(float));
    myds_metal_buffer_t *b_w_f16    = myds_metal_buffer_alloc((size_t)24u * flat_dim * sizeof(_Float16));
    myds_metal_buffer_t *b_hc_scale = myds_metal_buffer_alloc(3u * sizeof(float));
    myds_metal_buffer_t *b_hc_base  = myds_metal_buffer_alloc(24u * sizeof(float));

    myds_metal_buffer_t *b_after    = myds_metal_buffer_alloc((size_t)c_len * flat_dim * sizeof(float));
    myds_metal_buffer_t *b_xnorm_s1 = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_xnorm_s6 = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_sum_s6   = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_fsc_s1   = myds_metal_buffer_alloc((size_t)c_len * sizeof(float));
    myds_metal_buffer_t *b_fsc_s6   = myds_metal_buffer_alloc((size_t)c_len * sizeof(float));
    myds_metal_buffer_t *b_mix_s1   = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));
    myds_metal_buffer_t *b_mix_s6   = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));
    myds_metal_buffer_t *b_split_s6 = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));

    myds_metal_buffer_t *b_res_tail = myds_metal_buffer_alloc((size_t)c_len * tail_in_dim * sizeof(float));
    myds_metal_buffer_t *b_w_tail   = myds_metal_buffer_alloc((size_t)24u * tail_in_dim * sizeof(_Float16));
    myds_metal_buffer_t *b_mix_tail = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));

    myds_metal_buffer_t *b_res_4096   = myds_metal_buffer_alloc((size_t)c_len * 4u * 4096u * sizeof(float));
    myds_metal_buffer_t *b_normw_4096 = myds_metal_buffer_alloc(4096u * sizeof(float));
    myds_metal_buffer_t *b_xnorm_4096 = myds_metal_buffer_alloc((size_t)c_len * 4096u * sizeof(float));
    myds_metal_buffer_t *b_fsc_4096   = myds_metal_buffer_alloc((size_t)c_len * sizeof(float));

    int ok = b_res && b_block && b_pre16 && b_split96 && b_norm_w && b_w_f16 &&
             b_hc_scale && b_hc_base && b_after && b_xnorm_s1 && b_xnorm_s6 &&
             b_sum_s6 && b_fsc_s1 && b_fsc_s6 && b_mix_s1 && b_mix_s6 && b_split_s6 &&
             b_res_tail && b_w_tail && b_mix_tail &&
             b_res_4096 && b_normw_4096 && b_xnorm_4096 && b_fsc_4096;
    if (!ok) {
        fprintf(stderr, "FAIL: GPU buffer alloc failed (c_len=%u)\n", c_len);
        myds_metal_cleanup();
        return 0;
    }

    memcpy(myds_metal_buffer_data(b_res), h_res, (size_t)c_len * flat_dim * sizeof(float));
    memcpy(myds_metal_buffer_data(b_block), h_block, (size_t)c_len * dim * sizeof(float));
    memcpy(myds_metal_buffer_data(b_pre16), h_pre16, (size_t)c_len * 4u * sizeof(float));
    memcpy(myds_metal_buffer_data(b_split96), h_split96, (size_t)c_len * 24u * sizeof(float));
    memcpy(myds_metal_buffer_data(b_norm_w), h_norm_w, (size_t)dim * sizeof(float));
    memcpy(myds_metal_buffer_data(b_w_f16), h_w_f16, (size_t)24u * flat_dim * sizeof(_Float16));
    memcpy(myds_metal_buffer_data(b_hc_scale), h_hc_scale, 3u * sizeof(float));
    memcpy(myds_metal_buffer_data(b_hc_base), h_hc_base, 24u * sizeof(float));
    memcpy(myds_metal_buffer_data(b_res_tail), h_res_tail, (size_t)c_len * tail_in_dim * sizeof(float));
    memcpy(myds_metal_buffer_data(b_w_tail), h_w_tail_f16, (size_t)24u * tail_in_dim * sizeof(_Float16));
    memcpy(myds_metal_buffer_data(b_res_4096), h_res_4096, (size_t)c_len * 4u * 4096u * sizeof(float));
    memcpy(myds_metal_buffer_data(b_normw_4096), h_norm_w_4096, 4096u * sizeof(float));

    ok = myds_metal_begin_batch();

    /* Step 1 pattern: stride=16, sum_out=NULL, round_bf16=true -> gemv_batch_scaled(24, 20480, F16) */
    if (ok) ok = myds_metal_v41_hc_sum_rmsnorm_scale_batch(
        b_xnorm_s1, NULL, b_fsc_s1, b_res, b_pre16, b_norm_w, dim, c_len, 16u, 1e-6f, true);
    if (ok) ok = myds_metal_v41_gemv_batch_scaled(
        b_mix_s1, b_w_f16, b_res, b_fsc_s1, 24u, flat_dim, c_len, GGML_TYPE_F16);

    /* Step 6 combined chain: hc_expand_batch -> hc_sum_rmsnorm_scale_batch (stride=96, sum_out!=NULL) -> gemv_batch_scaled -> hc_split_batch */
    if (ok) ok = myds_metal_v41_hc_expand_batch(b_after, b_block, b_res, b_split96, dim, c_len);
    if (ok) ok = myds_metal_v41_hc_sum_rmsnorm_scale_batch(
        b_xnorm_s6, b_sum_s6, b_fsc_s6, b_after, b_split96, b_norm_w, dim, c_len, 96u, 1e-6f, true);
    if (ok) ok = myds_metal_v41_gemv_batch_scaled(
        b_mix_s6, b_w_f16, b_after, b_fsc_s6, 24u, flat_dim, c_len, GGML_TYPE_F16);
    if (ok) ok = myds_metal_v41_hc_split_batch(b_split_s6, b_mix_s6, b_hc_scale, b_hc_base, c_len, 1e-6f);

    /* Tail in_dim % 32 != 0 GEMV scaled test */
    if (ok) ok = myds_metal_v41_gemv_batch_scaled(
        b_mix_tail, b_w_tail, b_res_tail, b_fsc_s1, 24u, tail_in_dim, c_len, GGML_TYPE_F16);

    /* Non-5120 dim=4096 automatic fallback test for hc_sum_rmsnorm_scale_batch */
    if (ok) ok = myds_metal_v41_hc_sum_rmsnorm_scale_batch(
        b_xnorm_4096, NULL, b_fsc_4096, b_res_4096, b_pre16, b_normw_4096, 4096u, c_len, 16u, 1e-6f, false);

    if (ok) ok = myds_metal_commit_batch();

    if (ok) {
        memcpy(out->after_attn, myds_metal_buffer_data(b_after), (size_t)c_len * flat_dim * sizeof(float));
        memcpy(out->x_norm_s1, myds_metal_buffer_data(b_xnorm_s1), (size_t)c_len * dim * sizeof(float));
        memcpy(out->x_norm_s6, myds_metal_buffer_data(b_xnorm_s6), (size_t)c_len * dim * sizeof(float));
        memcpy(out->sum_s6, myds_metal_buffer_data(b_sum_s6), (size_t)c_len * dim * sizeof(float));
        memcpy(out->flat_scale_s1, myds_metal_buffer_data(b_fsc_s1), (size_t)c_len * sizeof(float));
        memcpy(out->flat_scale_s6, myds_metal_buffer_data(b_fsc_s6), (size_t)c_len * sizeof(float));
        memcpy(out->hc_mix_s1, myds_metal_buffer_data(b_mix_s1), (size_t)c_len * 24u * sizeof(float));
        memcpy(out->hc_mix_s6, myds_metal_buffer_data(b_mix_s6), (size_t)c_len * 24u * sizeof(float));
        memcpy(out->hc_split_s6, myds_metal_buffer_data(b_split_s6), (size_t)c_len * 24u * sizeof(float));
        memcpy(out->hc_mix_tail, myds_metal_buffer_data(b_mix_tail), (size_t)c_len * 24u * sizeof(float));
        memcpy(out->x_norm_4096, myds_metal_buffer_data(b_xnorm_4096), (size_t)c_len * 4096u * sizeof(float));
        memcpy(out->flat_scale_4096, myds_metal_buffer_data(b_fsc_4096), (size_t)c_len * sizeof(float));
    }

    myds_metal_hc_mix_fast_counters(counters_out, false);

    myds_metal_buffer_free(b_res);
    myds_metal_buffer_free(b_block);
    myds_metal_buffer_free(b_pre16);
    myds_metal_buffer_free(b_split96);
    myds_metal_buffer_free(b_norm_w);
    myds_metal_buffer_free(b_w_f16);
    myds_metal_buffer_free(b_hc_scale);
    myds_metal_buffer_free(b_hc_base);
    myds_metal_buffer_free(b_after);
    myds_metal_buffer_free(b_xnorm_s1);
    myds_metal_buffer_free(b_xnorm_s6);
    myds_metal_buffer_free(b_sum_s6);
    myds_metal_buffer_free(b_fsc_s1);
    myds_metal_buffer_free(b_fsc_s6);
    myds_metal_buffer_free(b_mix_s1);
    myds_metal_buffer_free(b_mix_s6);
    myds_metal_buffer_free(b_split_s6);
    myds_metal_buffer_free(b_res_tail);
    myds_metal_buffer_free(b_w_tail);
    myds_metal_buffer_free(b_mix_tail);
    myds_metal_buffer_free(b_res_4096);
    myds_metal_buffer_free(b_normw_4096);
    myds_metal_buffer_free(b_xnorm_4096);
    myds_metal_buffer_free(b_fsc_4096);
    myds_metal_cleanup();
    return ok;
}

static int test_zero_ulp_all_shapes(void) {
    printf("[2/4] Testing strict 0-ULP (memcmp == 0) across c_len in {1, 7, 122, 123, 312, 513, 2048}...\n");
    const uint32_t dim = 5120u;
    const uint32_t flat_dim = 4u * dim;
    const uint32_t tail_in_dim = flat_dim + 4u;
    const uint32_t c_lens[] = {1u, 7u, 122u, 123u, 312u, 513u, 2048u};
    const uint32_t max_c = 2048u;

    float *h_res        = (float *)malloc((size_t)max_c * flat_dim * sizeof(float));
    float *h_block      = (float *)malloc((size_t)max_c * dim * sizeof(float));
    float *h_pre16      = (float *)malloc((size_t)max_c * 4u * sizeof(float));
    float *h_split96    = (float *)malloc((size_t)max_c * 24u * sizeof(float));
    float *h_norm_w     = (float *)malloc((size_t)dim * sizeof(float));
    _Float16 *h_w_f16   = (_Float16 *)malloc((size_t)24u * flat_dim * sizeof(_Float16));
    float h_hc_scale[3] = {0.11f, -0.07f, 0.19f};
    float h_hc_base[24];
    float *h_res_tail   = (float *)malloc((size_t)max_c * tail_in_dim * sizeof(float));
    _Float16 *h_w_tail  = (_Float16 *)malloc((size_t)24u * tail_in_dim * sizeof(_Float16));
    float *h_res_4096   = (float *)malloc((size_t)max_c * 4u * 4096u * sizeof(float));
    float *h_normw_4096 = (float *)malloc(4096u * sizeof(float));

    if (!h_res || !h_block || !h_pre16 || !h_split96 || !h_norm_w || !h_w_f16 ||
        !h_res_tail || !h_w_tail || !h_res_4096 || !h_normw_4096) {
        fprintf(stderr, "FAIL: host OOM\n");
        return 0;
    }

    for (size_t i = 0; i < (size_t)max_c * flat_dim; i++) h_res[i] = make_special_or_rand(i, 1.0f);
    for (size_t i = 0; i < (size_t)max_c * dim; i++) h_block[i] = make_special_or_rand(i + 17u, 1.5f);
    for (size_t i = 0; i < (size_t)max_c * 4u; i++) h_pre16[i] = make_special_or_rand(i + 31u, 0.5f);
    for (size_t i = 0; i < (size_t)max_c * 24u; i++) h_split96[i] = make_special_or_rand(i + 53u, 0.5f);
    for (size_t i = 0; i < (size_t)dim; i++) h_norm_w[i] = rand_uniform(-2.0f, 2.0f);
    for (size_t i = 0; i < (size_t)24u * flat_dim; i++) h_w_f16[i] = (_Float16)make_special_or_rand(i + 71u, 0.25f);
    for (size_t i = 0; i < 24u; i++) h_hc_base[i] = rand_uniform(-1.0f, 1.0f);
    for (size_t i = 0; i < (size_t)max_c * tail_in_dim; i++) h_res_tail[i] = make_special_or_rand(i + 97u, 1.0f);
    for (size_t i = 0; i < (size_t)24u * tail_in_dim; i++) h_w_tail[i] = (_Float16)make_special_or_rand(i + 101u, 0.25f);
    for (size_t i = 0; i < (size_t)max_c * 4u * 4096u; i++) h_res_4096[i] = make_special_or_rand(i + 131u, 1.0f);
    for (size_t i = 0; i < 4096u; i++) h_normw_4096[i] = rand_uniform(-2.0f, 2.0f);

    int all_ok = 1;
    for (size_t ci = 0; ci < sizeof(c_lens) / sizeof(c_lens[0]) && all_ok; ci++) {
        const uint32_t c_len = c_lens[ci];
        run_outputs_t ref, got;
        alloc_outputs(&ref, c_len, dim);
        alloc_outputs(&got, c_len, dim);

        myds_metal_hc_mix_fast_counters_t cnt0 = {0}, cnt1 = {0};
        if (!run_arm(0, c_len, dim, h_res, h_block, h_pre16, h_split96, h_norm_w, h_w_f16,
                     h_hc_scale, h_hc_base, h_res_tail, h_w_tail, h_res_4096, h_normw_4096, &ref, &cnt0) ||
            !run_arm(1, c_len, dim, h_res, h_block, h_pre16, h_split96, h_norm_w, h_w_f16,
                     h_hc_scale, h_hc_base, h_res_tail, h_w_tail, h_res_4096, h_normw_4096, &got, &cnt1)) {
            all_ok = 0;
        }

        if (all_ok) {
            if (cnt0.gemv_nr24_encoded != 0 || cnt0.rmsnorm_single_read_encoded != 0 || cnt0.expand4_vec_encoded != 0) {
                fprintf(stderr, "FAIL c_len=%u: arm0 counters non-zero (%llu/%llu/%llu)\n",
                        c_len, (unsigned long long)cnt0.gemv_nr24_encoded,
                        (unsigned long long)cnt0.rmsnorm_single_read_encoded,
                        (unsigned long long)cnt0.expand4_vec_encoded);
                all_ok = 0;
            }
            /* Expect: 2 gemv_nr24 (s1 + s6; NOT 20484 non-32-aligned fallback!),
             * 2 rmsnorm_single_read (s1 + s6; NOT 4096 fallback!), 1 expand4_vec */
            if (cnt1.gemv_nr24_encoded != 2u || cnt1.rmsnorm_single_read_encoded != 2u || cnt1.expand4_vec_encoded != 1u) {
                fprintf(stderr, "FAIL c_len=%u: arm1 counters mismatch (got %llu/%llu/%llu, expected 2/2/1)\n",
                        c_len, (unsigned long long)cnt1.gemv_nr24_encoded,
                        (unsigned long long)cnt1.rmsnorm_single_read_encoded,
                        (unsigned long long)cnt1.expand4_vec_encoded);
                all_ok = 0;
            }
        }

        if (all_ok) {
            all_ok &= compare_exact("hc_expand4_batch_vec (after_attn)", ref.after_attn, got.after_attn, (size_t)c_len * flat_dim);
            all_ok &= compare_exact("rmsnorm_scale_s1 (x_norm)", ref.x_norm_s1, got.x_norm_s1, (size_t)c_len * dim);
            all_ok &= compare_exact("rmsnorm_scale_s1 (flat_scale)", ref.flat_scale_s1, got.flat_scale_s1, (size_t)c_len);
            all_ok &= compare_exact("gemv_nr24_s1 (hc_mix)", ref.hc_mix_s1, got.hc_mix_s1, (size_t)c_len * 24u);
            all_ok &= compare_exact("combined_s6 (x_norm)", ref.x_norm_s6, got.x_norm_s6, (size_t)c_len * dim);
            all_ok &= compare_exact("combined_s6 (sum_out)", ref.sum_s6, got.sum_s6, (size_t)c_len * dim);
            all_ok &= compare_exact("combined_s6 (flat_scale)", ref.flat_scale_s6, got.flat_scale_s6, (size_t)c_len);
            all_ok &= compare_exact("combined_s6 (hc_mix)", ref.hc_mix_s6, got.hc_mix_s6, (size_t)c_len * 24u);
            all_ok &= compare_exact("combined_s6 (hc_split)", ref.hc_split_s6, got.hc_split_s6, (size_t)c_len * 24u);
            all_ok &= compare_exact("gemv_nr24_tail_20484 (hc_mix)", ref.hc_mix_tail, got.hc_mix_tail, (size_t)c_len * 24u);
            all_ok &= compare_exact("rmsnorm_fallback_4096 (x_norm)", ref.x_norm_4096, got.x_norm_4096, (size_t)c_len * 4096u);
            all_ok &= compare_exact("rmsnorm_fallback_4096 (flat_scale)", ref.flat_scale_4096, got.flat_scale_4096, (size_t)c_len);
        }

        if (all_ok) {
            printf("  PASS c_len=%-4u: strict 0-ULP (memcmp == 0) all 3 kernels + combined chain + fallbacks + counters (2/2/1)\n", c_len);
        }

        free_outputs(&ref);
        free_outputs(&got);
    }

    free(h_res);
    free(h_block);
    free(h_pre16);
    free(h_split96);
    free(h_norm_w);
    free(h_w_f16);
    free(h_res_tail);
    free(h_w_tail);
    free(h_res_4096);
    free(h_normw_4096);
    return all_ok;
}

static int test_reference_vs_separate_degraded(void) {
    printf("[3/4] Testing MYDS_HC_MIX_FAST=1 strict 0-ULP vs separate non-fused reference (rmsnorm_batch nth=256 + gemv_batch)...\n");
    const uint32_t dim = 5120u;
    const uint32_t flat_dim = 4u * dim;
    const uint32_t c_len = 123u;

    /* Align rmsnorm_batch reduction width (nth=256) with hc_sum_rmsnorm_scale_batch (nth=256)
     * so both paths execute the exact same 256-thread simd_sum reduction tree. */
    setenv("MYDS_RMS_NTH", "256", 1);
    setenv("MYDS_HC_MIX_FAST", "1", 1);
    if (!myds_metal_init("bin")) return 0;

    myds_metal_buffer_t *b_res     = myds_metal_buffer_alloc((size_t)c_len * flat_dim * sizeof(float));
    myds_metal_buffer_t *b_pre     = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));
    myds_metal_buffer_t *b_norm_w  = myds_metal_buffer_alloc((size_t)dim * sizeof(float));
    myds_metal_buffer_t *b_w_f16   = myds_metal_buffer_alloc((size_t)24u * flat_dim * sizeof(_Float16));
    myds_metal_buffer_t *b_flat_n  = myds_metal_buffer_alloc((size_t)c_len * flat_dim * sizeof(float));
    myds_metal_buffer_t *b_xn_ref  = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_mix_ref = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));
    myds_metal_buffer_t *b_xn_got  = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_scales  = myds_metal_buffer_alloc((size_t)c_len * sizeof(float));
    myds_metal_buffer_t *b_mix_got = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));

    float *p_res = (float *)myds_metal_buffer_data(b_res);
    float *p_pre = (float *)myds_metal_buffer_data(b_pre);
    float *p_nw  = (float *)myds_metal_buffer_data(b_norm_w);
    _Float16 *p_w = (_Float16 *)myds_metal_buffer_data(b_w_f16);
    for (size_t i = 0; i < (size_t)c_len * flat_dim; i++) p_res[i] = make_special_or_rand(i + 3u, 1.0f);
    for (size_t i = 0; i < (size_t)c_len * 24u; i++) p_pre[i] = make_special_or_rand(i + 11u, 0.5f);
    for (size_t i = 0; i < (size_t)dim; i++) p_nw[i] = rand_uniform(-1.5f, 1.5f);
    for (size_t i = 0; i < (size_t)24u * flat_dim; i++) p_w[i] = (_Float16)make_special_or_rand(i + 19u, 0.25f);

    int ok = myds_metal_begin_batch();
    if (ok) ok = myds_metal_v41_hc_sum_rmsnorm_batch(b_xn_ref, NULL, b_res, b_pre, b_norm_w, dim, c_len, 96u, 1e-6f, true);
    if (ok) ok = myds_metal_v41_rmsnorm_batch(b_flat_n, b_res, NULL, flat_dim, c_len, 1e-6f, false);
    if (ok) ok = myds_metal_v41_gemv_batch(b_mix_ref, b_w_f16, b_flat_n, 24u, flat_dim, c_len, GGML_TYPE_F16);

    if (ok) ok = myds_metal_v41_hc_sum_rmsnorm_scale_batch(b_xn_got, NULL, b_scales, b_res, b_pre, b_norm_w, dim, c_len, 96u, 1e-6f, true);
    if (ok) ok = myds_metal_v41_gemv_batch_scaled(b_mix_got, b_w_f16, b_res, b_scales, 24u, flat_dim, c_len, GGML_TYPE_F16);
    if (ok) ok = myds_metal_commit_batch();

    if (ok) {
        ok &= compare_exact("fast_vs_separate (x_norm)", (const float *)myds_metal_buffer_data(b_xn_ref),
                            (const float *)myds_metal_buffer_data(b_xn_got), (size_t)c_len * dim);
        ok &= compare_exact("fast_vs_separate (hc_mix)", (const float *)myds_metal_buffer_data(b_mix_ref),
                            (const float *)myds_metal_buffer_data(b_mix_got), (size_t)c_len * 24u);
    }
    if (ok) {
        printf("  PASS: MYDS_HC_MIX_FAST=1 matches separate rmsnorm_batch + gemv_batch at strict 0-ULP (c_len=%u).\n", c_len);
    }

    myds_metal_buffer_free(b_res);
    myds_metal_buffer_free(b_pre);
    myds_metal_buffer_free(b_norm_w);
    myds_metal_buffer_free(b_w_f16);
    myds_metal_buffer_free(b_flat_n);
    myds_metal_buffer_free(b_xn_ref);
    myds_metal_buffer_free(b_mix_ref);
    myds_metal_buffer_free(b_xn_got);
    myds_metal_buffer_free(b_scales);
    myds_metal_buffer_free(b_mix_got);
    myds_metal_cleanup();
    return ok;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double bench_kernel_ms(int fast_mode, uint32_t c_len, int mode_kind, int disps_per_cb, int runs) {
    /* mode_kind: 0=gemv_scaled, 1=rmsnorm_scale, 2=expand4, 3=step1_chain, 4=step6_chain */
    const uint32_t dim = 5120u;
    const uint32_t flat_dim = 4u * dim;

    setenv("MYDS_HC_MIX_FAST", fast_mode ? "1" : "0", 1);
    if (!myds_metal_init("bin")) return -1.0;

    myds_metal_buffer_t *b_res     = myds_metal_buffer_alloc((size_t)c_len * flat_dim * sizeof(float));
    myds_metal_buffer_t *b_block   = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_split96 = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));
    myds_metal_buffer_t *b_norm_w  = myds_metal_buffer_alloc((size_t)dim * sizeof(float));
    myds_metal_buffer_t *b_w_f16   = myds_metal_buffer_alloc((size_t)24u * flat_dim * sizeof(_Float16));
    myds_metal_buffer_t *b_after   = myds_metal_buffer_alloc((size_t)c_len * flat_dim * sizeof(float));
    myds_metal_buffer_t *b_xnorm   = myds_metal_buffer_alloc((size_t)c_len * dim * sizeof(float));
    myds_metal_buffer_t *b_scales  = myds_metal_buffer_alloc((size_t)c_len * sizeof(float));
    myds_metal_buffer_t *b_mix     = myds_metal_buffer_alloc((size_t)c_len * 24u * sizeof(float));

    float *p_res = (float *)myds_metal_buffer_data(b_res);
    float *p_blk = (float *)myds_metal_buffer_data(b_block);
    float *p_sp  = (float *)myds_metal_buffer_data(b_split96);
    float *p_nw  = (float *)myds_metal_buffer_data(b_norm_w);
    float *p_sc  = (float *)myds_metal_buffer_data(b_scales);
    _Float16 *p_w = (_Float16 *)myds_metal_buffer_data(b_w_f16);
    for (size_t i = 0; i < (size_t)c_len * flat_dim; i++) p_res[i] = 0.01f * (float)((int)(i % 19u) - 9);
    for (size_t i = 0; i < (size_t)c_len * dim; i++) p_blk[i] = 0.02f * (float)((int)(i % 13u) - 6);
    for (size_t i = 0; i < (size_t)c_len * 24u; i++) p_sp[i] = 0.25f;
    for (size_t i = 0; i < (size_t)dim; i++) p_nw[i] = 1.0f;
    for (size_t i = 0; i < (size_t)c_len; i++) p_sc[i] = 0.5f;
    for (size_t i = 0; i < (size_t)24u * flat_dim; i++) p_w[i] = (_Float16)(0.01f * (float)((int)(i % 11u) - 5));

    double samples[32];
    if (runs > 32) runs = 32;

    /* 2 warmup CBs + `runs` measured CBs */
    for (int r = -2; r < runs; r++) {
        double t0 = now_ms();
        myds_metal_begin_batch();
        for (int d = 0; d < disps_per_cb; d++) {
            if (mode_kind == 0) {
                myds_metal_v41_gemv_batch_scaled(b_mix, b_w_f16, b_res, b_scales, 24u, flat_dim, c_len, GGML_TYPE_F16);
            } else if (mode_kind == 1) {
                myds_metal_v41_hc_sum_rmsnorm_scale_batch(b_xnorm, NULL, b_scales, b_res, b_split96, b_norm_w, dim, c_len, 96u, 1e-6f, true);
            } else if (mode_kind == 2) {
                myds_metal_v41_hc_expand_batch(b_after, b_block, b_res, b_split96, dim, c_len);
            } else if (mode_kind == 3) {
                myds_metal_v41_hc_sum_rmsnorm_scale_batch(b_xnorm, NULL, b_scales, b_res, b_split96, b_norm_w, dim, c_len, 96u, 1e-6f, true);
                myds_metal_v41_gemv_batch_scaled(b_mix, b_w_f16, b_res, b_scales, 24u, flat_dim, c_len, GGML_TYPE_F16);
            } else {
                myds_metal_v41_hc_expand_batch(b_after, b_block, b_res, b_split96, dim, c_len);
                myds_metal_v41_hc_sum_rmsnorm_scale_batch(b_xnorm, NULL, b_scales, b_after, b_split96, b_norm_w, dim, c_len, 96u, 1e-6f, true);
                myds_metal_v41_gemv_batch_scaled(b_mix, b_w_f16, b_after, b_scales, 24u, flat_dim, c_len, GGML_TYPE_F16);
            }
        }
        myds_metal_commit_batch();
        double dt_wall = now_ms() - t0;
        double gs = 0.0, ge = 0.0;
        double dt_ms = dt_wall;
        if (myds_metal_last_completed_gpu_span(&gs, &ge) && ge > gs) {
            dt_ms = (ge - gs) * 1000.0;
        }
        if (r >= 0) samples[r] = dt_ms / (double)disps_per_cb;
    }

    qsort(samples, (size_t)runs, sizeof(double), cmp_double);

    myds_metal_buffer_free(b_res);
    myds_metal_buffer_free(b_block);
    myds_metal_buffer_free(b_split96);
    myds_metal_buffer_free(b_norm_w);
    myds_metal_buffer_free(b_w_f16);
    myds_metal_buffer_free(b_after);
    myds_metal_buffer_free(b_xnorm);
    myds_metal_buffer_free(b_scales);
    myds_metal_buffer_free(b_mix);
    myds_metal_cleanup();
    return samples[runs / 2];
}

static void run_benchmarks(void) {
    printf("[4/4] Running M4 GPU microbenchmarks (baseline MYDS_HC_MIX_FAST=0 vs 1, median of 7 runs)...\n");
    const uint32_t c_lens[] = {122u, 312u, 2048u};
    const char *names[] = {
        "gemv_scaled (24x20480 F16)",
        "hc_sum_rmsnorm_scale (5120)",
        "hc_expand4_batch (5120)",
        "Step1 chain (rmsnorm+gemv)",
        "Step6 chain (exp+rms+gemv)"
    };
    const double bytes_per_tok[] = {
        20480.0 * 4.0 + 24.0 * 4.0,                          /* gemv 1x read + write (excl 983KB weights) */
        20480.0 * 4.0 + 5120.0 * 4.0,                        /* rmsnorm_scale 1x read + norm write */
        (5120.0 + 20480.0 + 20480.0) * 4.0,                  /* expand4 block+res read + dst write */
        (20480.0 + 5120.0 + 20480.0) * 4.0,                  /* Step1 net */
        (5120.0 + 20480.0 + 20480.0 + 20480.0 + 5120.0 + 20480.0) * 4.0, /* Step6 net */
    };

    for (size_t ci = 0; ci < sizeof(c_lens) / sizeof(c_lens[0]); ci++) {
        const uint32_t c_len = c_lens[ci];
        const int disps = (c_len >= 2048u) ? 4 : 16;
        printf("\n  --- c_len = %u (%d disps/CB) ---\n", c_len, disps);
        printf("  %-30s | %9s | %9s | %7s | %10s | %9s\n",
               "Kernel / Chain", "Base (ms)", "Fast (ms)", "Speedup", "Delta (ms)", "Fast GB/s");
        for (int k = 0; k < 5; k++) {
            double t_base = bench_kernel_ms(0, c_len, k, disps, 7);
            double t_fast = bench_kernel_ms(1, c_len, k, disps, 7);
            double total_bytes = bytes_per_tok[k] * (double)c_len + ((k == 0 || k >= 3) ? (24.0 * 20480.0 * 2.0) : 0.0);
            double gbs = (t_fast > 0.0) ? (total_bytes / (t_fast * 1e6)) : 0.0;
            printf("  %-30s | %9.3f | %9.3f | %6.2fx | %+10.3f | %9.1f\n",
                   names[k], t_base, t_fast, t_base / t_fast, t_fast - t_base, gbs);
        }
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        int do_bench = 0;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--bench") == 0) do_bench = 1;
        }
        if (!test_knob_parsing()) return 1;
        if (!test_zero_ulp_all_shapes()) return 1;
        if (!test_reference_vs_separate_degraded()) return 1;
        if (do_bench) {
            run_benchmarks();
        } else {
            printf("[4/4] Skipping --bench (pass --bench to run M4 timing comparison).\n");
        }
        printf("\nALL test_hc_mix_fast CHECKS PASSED!\n");
        return 0;
    }
}
