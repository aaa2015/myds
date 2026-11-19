#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_v41.h"

extern uint64_t myds_v41_model_identity(const myds_model_t *m);

static uint32_t rng_state = 0x12345678u;
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

static void fill_iq2_xxs(uint8_t *buf, size_t n_blocks) {
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *p = buf + b * 66u;
        uint16_t d = f32_to_f16_bits(rand_f32(0.01f, 0.25f));
        memcpy(p, &d, 2u);
        for (size_t i = 2u; i < 66u; i++) {
            p[i] = (uint8_t)(xorshift32() & 0xffu);
        }
    }
}

static void fill_q2_k(uint8_t *buf, size_t n_blocks) {
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *p = buf + b * 84u;
        for (size_t i = 0; i < 80u; i++) {
            p[i] = (uint8_t)(xorshift32() & 0xffu);
        }
        uint16_t d = f32_to_f16_bits(rand_f32(0.01f, 0.25f));
        uint16_t dmin = f32_to_f16_bits(rand_f32(0.005f, 0.12f));
        memcpy(p + 80u, &d, 2u);
        memcpy(p + 82u, &dmin, 2u);
    }
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int compare_f32_exact(const char *label, const float *a, const float *b, size_t n) {
    size_t diffs = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        uint32_t ua = 0, ub = 0;
        memcpy(&ua, a + i, 4u);
        memcpy(&ub, b + i, 4u);
        if (ua != ub) {
            diffs++;
            float d = fabsf(a[i] - b[i]);
            if (d > max_abs) max_abs = d;
            if (diffs <= 3u) {
                fprintf(stderr, "  [DIFF] %s idx=%zu ref=%g (0x%08x) got=%g (0x%08x)\n",
                        label, i, a[i], ua, b[i], ub);
            }
        }
    }
    if (diffs != 0) {
        fprintf(stderr, "FAIL %s: %zu / %zu elements differ (max_abs=%g)\n", label, diffs, n, max_abs);
        return 0;
    }
    printf("PASS %s: 0-ULP bit-identical (%zu floats)\n", label, n);
    return 1;
}

static int test_kvdisk_identity_isolation(void) {
    gguf_file_t dummy_gf;
    memset(&dummy_gf, 0, sizeof(dummy_gf));
    dummy_gf.fd = -1;

    myds_model_t dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.gf = &dummy_gf;
    snprintf(dummy.arch, sizeof(dummy.arch), "deepseek41");
    dummy.n_activated_experts = 6u;
    dummy.n_routed_experts = 384u;
    dummy.n_layers = 40u;

    for (int fp4 = 0; fp4 <= 1; fp4++) {
        setenv("MYDS_INDEXER_KEY_FP4", fp4 ? "1" : "0", 1);

        unsetenv("MYDS_MOE_PREFILL_TOPK");
        const uint64_t id_unset = myds_v41_model_identity(&dummy);
        setenv("MYDS_MOE_PREFILL_TOPK", "0", 1);
        const uint64_t id_zero = myds_v41_model_identity(&dummy);
        setenv("MYDS_MOE_PREFILL_TOPK", "6", 1);
        const uint64_t id_six = myds_v41_model_identity(&dummy);
        setenv("MYDS_MOE_PREFILL_TOPK", "4", 1);
        const uint64_t id_four = myds_v41_model_identity(&dummy);
        setenv("MYDS_MOE_PREFILL_TOPK", "3", 1);
        const uint64_t id_three = myds_v41_model_identity(&dummy);
        unsetenv("MYDS_MOE_PREFILL_TOPK");

        if (id_unset == 0ull || id_four == 0ull || id_three == 0ull) {
            fprintf(stderr, "FAIL [P-NEW-4] fp4=%d: model_identity returned 0 (unset=%llx top4=%llx top3=%llx)\n",
                    fp4, (unsigned long long)id_unset, (unsigned long long)id_four, (unsigned long long)id_three);
            return 0;
        }
        if (id_unset != id_zero || id_unset != id_six) {
            fprintf(stderr, "FAIL [P-NEW-4] fp4=%d: default/0/6 mismatch (%llx vs %llx vs %llx)\n",
                    fp4, (unsigned long long)id_unset, (unsigned long long)id_zero, (unsigned long long)id_six);
            return 0;
        }
        if (id_four == id_six || id_three == id_six || id_four == id_three) {
            fprintf(stderr, "FAIL [P-NEW-4] fp4=%d: kvdisk hash not isolated (k6=%llx k4=%llx k3=%llx)\n",
                    fp4, (unsigned long long)id_six, (unsigned long long)id_four, (unsigned long long)id_three);
            return 0;
        }
        printf("PASS [P-NEW-4] kvdisk/model_identity isolation (fp4=%d): k6=0x%016llx k4=0x%016llx k3=0x%016llx\n",
               fp4, (unsigned long long)id_six, (unsigned long long)id_four, (unsigned long long)id_three);
    }
    unsetenv("MYDS_INDEXER_KEY_FP4");
    return 1;
}

static int test_p_new_1_and_4(const char *metallib_path) {
    const uint32_t dim = 5120u;
    const uint32_t expert_dim = 2304u;
    const uint32_t n_experts = 32u;
    const float clamp_val = 7.0f;
    const uint32_t test_tokens[] = {32u, 64u, 304u, 512u};
    const uint32_t test_ks[] = {6u, 4u, 3u};

    setenv("MYDS_MOE_PREFILL_WIDE", "2", 1);
    if (myds_metal_init(metallib_path) != 0) {
        fprintf(stderr, "FAIL: MYDS_MOE_PREFILL_WIDE=2 should fail closed\n");
        myds_metal_cleanup();
        return 0;
    }
    printf("PASS MYDS_MOE_PREFILL_WIDE=2 fail-closed\n");

    const size_t gate_exp_bytes = (size_t)expert_dim * (dim / 256u) * 66u;
    const size_t down_exp_bytes = (size_t)dim * (expert_dim / 256u) * 84u;
    const uint32_t max_tok = 512u;

    uint8_t *h_gate = (uint8_t *)malloc(n_experts * gate_exp_bytes);
    uint8_t *h_up   = (uint8_t *)malloc(n_experts * gate_exp_bytes);
    uint8_t *h_down = (uint8_t *)malloc(n_experts * down_exp_bytes);
    float   *h_x    = (float *)malloc((size_t)max_tok * dim * sizeof(float));
    int32_t *h_ids  = (int32_t *)malloc((size_t)max_tok * 6u * sizeof(int32_t));
    float   *h_w    = (float *)malloc((size_t)max_tok * 6u * sizeof(float));
    float   *ref_out = (float *)malloc((size_t)max_tok * dim * sizeof(float));
    float   *got_out = (float *)malloc((size_t)max_tok * dim * sizeof(float));
    if (!h_gate || !h_up || !h_down || !h_x || !h_ids || !h_w || !ref_out || !got_out) {
        fprintf(stderr, "FAIL: host OOM\n");
        return 0;
    }

    rng_state = 0x51a7beefu;
    fill_iq2_xxs(h_gate, (size_t)n_experts * expert_dim * (dim / 256u));
    fill_iq2_xxs(h_up,   (size_t)n_experts * expert_dim * (dim / 256u));
    fill_q2_k(h_down,    (size_t)n_experts * dim * (expert_dim / 256u));
    for (size_t i = 0; i < (size_t)max_tok * dim; i++) h_x[i] = rand_f32(-1.5f, 1.5f);

    int ok = 1;
    for (size_t ki = 0; ki < sizeof(test_ks)/sizeof(test_ks[0]) && ok; ki++) {
        const uint32_t k = test_ks[ki];
        for (size_t ti = 0; ti < sizeof(test_tokens)/sizeof(test_tokens[0]) && ok; ti++) {
            const uint32_t n_tok = test_tokens[ti];
            for (uint32_t t = 0; t < n_tok; t++) {
                uint32_t used = 0u;
                for (uint32_t s = 0; s < k; s++) {
                    int32_t e = 0;
                    for (;;) {
                        uint32_t r = xorshift32() % 100u;
                        if (r < 45u) e = (int32_t)(xorshift32() % 4u);
                        else if (r < 80u) e = (int32_t)(4u + (xorshift32() % 8u));
                        else e = (int32_t)(12u + (xorshift32() % (n_experts - 12u)));
                        if (!(used & (1u << (uint32_t)e))) { used |= (1u << (uint32_t)e); break; }
                    }
                    h_ids[t * k + s] = e;
                    h_w[t * k + s] = rand_f32(0.05f, 0.5f);
                }
            }

            setenv("MYDS_MOE_PREFILL_WIDE", "0", 1);
            if (!myds_metal_init(metallib_path)) { ok = 0; break; }
            myds_metal_buffer_t *b_gate = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
            myds_metal_buffer_t *b_up   = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
            myds_metal_buffer_t *b_down = myds_metal_buffer_alloc(n_experts * down_exp_bytes);
            myds_metal_buffer_t *b_x    = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
            myds_metal_buffer_t *b_ids  = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(int32_t));
            myds_metal_buffer_t *b_w    = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(float));
            myds_metal_buffer_t *b_sg   = myds_metal_buffer_alloc((size_t)n_tok * k * expert_dim * sizeof(float));
            myds_metal_buffer_t *b_su   = myds_metal_buffer_alloc((size_t)n_tok * k * expert_dim * sizeof(float));
            myds_metal_buffer_t *b_sm   = myds_metal_buffer_alloc((size_t)n_tok * k * expert_dim * sizeof(float));
            myds_metal_buffer_t *b_sd   = myds_metal_buffer_alloc((size_t)n_tok * k * dim * sizeof(float));
            myds_metal_buffer_t *b_out  = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
            memcpy(myds_metal_buffer_data(b_gate), h_gate, n_experts * gate_exp_bytes);
            memcpy(myds_metal_buffer_data(b_up),   h_up,   n_experts * gate_exp_bytes);
            memcpy(myds_metal_buffer_data(b_down), h_down, n_experts * down_exp_bytes);
            memcpy(myds_metal_buffer_data(b_x),    h_x,    (size_t)n_tok * dim * sizeof(float));
            memcpy(myds_metal_buffer_data(b_ids),  h_ids,  (size_t)n_tok * k * sizeof(int32_t));
            memcpy(myds_metal_buffer_data(b_w),    h_w,    (size_t)n_tok * k * sizeof(float));

            double t0_off = now_ms();
            if (!myds_metal_routed_moe_batch(b_out, b_x, b_gate, b_up, b_down, b_ids, b_w,
                    b_sg, b_su, b_sm, b_sd, dim, expert_dim, n_experts, k, n_tok,
                    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, clamp_val)) {
                fprintf(stderr, "FAIL: arm0 routed_moe_batch failed (k=%u n_tok=%u)\n", k, n_tok);
                ok = 0;
            }
            double dt_off = now_ms() - t0_off;
            memcpy(ref_out, myds_metal_buffer_data(b_out), (size_t)n_tok * dim * sizeof(float));
            myds_metal_buffer_free(b_gate); myds_metal_buffer_free(b_up); myds_metal_buffer_free(b_down);
            myds_metal_buffer_free(b_x); myds_metal_buffer_free(b_ids); myds_metal_buffer_free(b_w);
            myds_metal_buffer_free(b_sg); myds_metal_buffer_free(b_su); myds_metal_buffer_free(b_sm);
            myds_metal_buffer_free(b_sd); myds_metal_buffer_free(b_out);
            myds_metal_cleanup();
            if (!ok) break;

            setenv("MYDS_MOE_PREFILL_WIDE", "1", 1);
            if (!myds_metal_init(metallib_path)) { ok = 0; break; }
            b_gate = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
            b_up   = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
            b_down = myds_metal_buffer_alloc(n_experts * down_exp_bytes);
            b_x    = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
            b_ids  = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(int32_t));
            b_w    = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(float));
            b_sg   = myds_metal_buffer_alloc((size_t)n_tok * k * expert_dim * sizeof(float));
            b_su   = myds_metal_buffer_alloc((size_t)n_tok * k * expert_dim * sizeof(float));
            b_sm   = myds_metal_buffer_alloc((size_t)n_tok * k * expert_dim * sizeof(float));
            b_sd   = myds_metal_buffer_alloc((size_t)n_tok * k * dim * sizeof(float));
            b_out  = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
            memcpy(myds_metal_buffer_data(b_gate), h_gate, n_experts * gate_exp_bytes);
            memcpy(myds_metal_buffer_data(b_up),   h_up,   n_experts * gate_exp_bytes);
            memcpy(myds_metal_buffer_data(b_down), h_down, n_experts * down_exp_bytes);
            memcpy(myds_metal_buffer_data(b_x),    h_x,    (size_t)n_tok * dim * sizeof(float));
            memcpy(myds_metal_buffer_data(b_ids),  h_ids,  (size_t)n_tok * k * sizeof(int32_t));
            memcpy(myds_metal_buffer_data(b_w),    h_w,    (size_t)n_tok * k * sizeof(float));

            double t0_on = now_ms();
            if (!myds_metal_routed_moe_batch(b_out, b_x, b_gate, b_up, b_down, b_ids, b_w,
                    b_sg, b_su, b_sm, b_sd, dim, expert_dim, n_experts, k, n_tok,
                    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, clamp_val)) {
                fprintf(stderr, "FAIL: arm1 routed_moe_batch failed (k=%u n_tok=%u)\n", k, n_tok);
                ok = 0;
            }
            double dt_on = now_ms() - t0_on;
            memcpy(got_out, myds_metal_buffer_data(b_out), (size_t)n_tok * dim * sizeof(float));
            myds_metal_buffer_free(b_gate); myds_metal_buffer_free(b_up); myds_metal_buffer_free(b_down);
            myds_metal_buffer_free(b_x); myds_metal_buffer_free(b_ids); myds_metal_buffer_free(b_w);
            myds_metal_buffer_free(b_sg); myds_metal_buffer_free(b_su); myds_metal_buffer_free(b_sm);
            myds_metal_buffer_free(b_sd); myds_metal_buffer_free(b_out);
            myds_metal_cleanup();

            char label[128];
            snprintf(label, sizeof(label), "[P-NEW-1/4] MoE MM k=%u n_tok=%u (off=%.2fms on=%.2fms)",
                     k, n_tok, dt_off, dt_on);
            if (!compare_f32_exact(label, ref_out, got_out, (size_t)n_tok * dim)) ok = 0;
        }
    }

    free(h_gate); free(h_up); free(h_down); free(h_x);
    free(h_ids); free(h_w); free(ref_out); free(got_out);
    return ok;
}

int main(int argc, char **argv) {
    const char *metallib = (argc > 1) ? argv[1] : "bin/myds.metallib";
    int ok = 1;
    if (!test_kvdisk_identity_isolation()) ok = 0;
    if (!test_p_new_1_and_4(metallib)) ok = 0;
    printf("\n=== SUMMARY: %s ===\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
