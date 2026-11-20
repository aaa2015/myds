/* [P-M1] Strict 0-ULP unit test, PSO occupancy check, and M4 microbenchmark
 * for MYDS_MOE_PREFILL_M1 (active when MYDS_MOE_PREFILL_WIDE=1 && MYDS_MOE_PREFILL_M1=1). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "myds_gguf.h"
#include "myds_metal.h"

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static uint32_t rng_state = 0x2468ace1u;
static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
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
        /* Include tiny subnormal-boundary scales occasionally to exercise exact dl order. */
        float dv = (b % 37u == 0u) ? 6.103515625e-5f : rand_f32(0.01f, 0.25f);
        float mv = (b % 41u == 0u) ? 3.0517578125e-5f : rand_f32(0.005f, 0.12f);
        uint16_t d = f32_to_f16_bits(dv);
        uint16_t dmin = f32_to_f16_bits(mv);
        memcpy(p + 80u, &d, 2u);
        memcpy(p + 82u, &dmin, 2u);
    }
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db);
}

static void check_fail_closed_env(const char *metallib_path) {
    const char *bad[] = {"", "2", "-1", "true", " 1", "01"};
    for (int w = 0; w <= 1; ++w) {
        setenv("MYDS_MOE_PREFILL_WIDE", w ? "1" : "0", 1);
        for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
            setenv("MYDS_MOE_PREFILL_M1", bad[i], 1);
            REQUIRE(myds_metal_init(metallib_path) == 0);
        }
    }
    /* WIDE=0 && M1=1 must succeed cleanly without enabling M1. */
    setenv("MYDS_MOE_PREFILL_WIDE", "0", 1);
    setenv("MYDS_MOE_PREFILL_M1", "1", 1);
    REQUIRE(myds_metal_init(metallib_path) == 1);
    REQUIRE(myds_metal_moe_prefill_m1_enabled() == 0);
    myds_metal_cleanup();

    /* WIDE=1 && M1=0 must succeed cleanly without enabling M1. */
    setenv("MYDS_MOE_PREFILL_WIDE", "1", 1);
    setenv("MYDS_MOE_PREFILL_M1", "0", 1);
    REQUIRE(myds_metal_init(metallib_path) == 1);
    REQUIRE(myds_metal_moe_prefill_m1_enabled() == 0);
    myds_metal_cleanup();

    /* Default (unset) must disable M1. */
    unsetenv("MYDS_MOE_PREFILL_M1");
    setenv("MYDS_MOE_PREFILL_WIDE", "1", 1);
    REQUIRE(myds_metal_init(metallib_path) == 1);
    REQUIRE(myds_metal_moe_prefill_m1_enabled() == 0);
    myds_metal_cleanup();

    printf("[PASS] MYDS_MOE_PREFILL_M1 fail-closed env parsing & WIDE gating\n");
}

static void check_pso_occupancy(const char *metallib_path) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        REQUIRE(dev != nil);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path)] error:&err];
        REQUIRE(lib != nil);

        struct { const char *name; int needs_fc; NSUInteger min_thr; NSUInteger dyn_smem; } specs[] = {
            {"kernel_mul_mm_id_map0_wide_m1_ne20_3", 0, 384u, 1792u},
            {"kernel_mul_mm_id_map0_wide_m1_ne20_4", 0, 384u, 1792u},
            {"kernel_mul_mm_id_map0_wide_m1_ne20_6", 0, 384u, 1792u},
            {"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_compact_tail_fast", 0, 64u, 8192u},
            {"kernel_mul_mm_id_q2_K_f16_wide_m1", 0, 128u, 8192u},
            {"kernel_mul_mm_id_q2_K_f16_tail_cull_fast", 1, 128u, 8192u},
        };
        MTLFunctionConstantValues *fc0 = [[MTLFunctionConstantValues alloc] init];
        bool b_false = false;
        [fc0 setConstantValue:&b_false type:MTLDataTypeBool atIndex:700];

        printf("[PSO] Occupancy verification on %s:\n", dev.name.UTF8String);
        for (size_t i = 0; i < sizeof(specs) / sizeof(*specs); ++i) {
            id<MTLFunction> fn = specs[i].needs_fc
                ? [lib newFunctionWithName:@(specs[i].name) constantValues:fc0 error:&err]
                : [lib newFunctionWithName:@(specs[i].name)];
            REQUIRE(fn != nil);
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            REQUIRE(pso != nil);
            REQUIRE(pso.maxTotalThreadsPerThreadgroup >= specs[i].min_thr);
            REQUIRE(pso.staticThreadgroupMemoryLength + specs[i].dyn_smem <= dev.maxThreadgroupMemoryLength);
            const NSUInteger tgs_by_smem = (specs[i].dyn_smem + pso.staticThreadgroupMemoryLength) > 0
                ? (dev.maxThreadgroupMemoryLength / (specs[i].dyn_smem + pso.staticThreadgroupMemoryLength)) : 32u;
            printf("  %-62s maxThr=%4lu staticSmem=%4lu dynSmem=%5lu (~%lu TGs/core by smem)\n",
                   specs[i].name,
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)pso.staticThreadgroupMemoryLength,
                   (unsigned long)specs[i].dyn_smem,
                   (unsigned long)tgs_by_smem);
        }
    }
}

static void generate_routing(int32_t *h_ids, float *h_w, uint32_t n_tok, uint32_t k,
                             uint32_t n_experts, int skewed) {
    for (uint32_t t = 0; t < n_tok; ++t) {
        for (uint32_t s = 0; s < k; ++s) {
            int32_t e = 0;
            for (;;) {
                if (skewed && n_experts >= 16u) {
                    uint32_t r = xorshift32() % 100u;
                    if (r < 45u) e = (int32_t)(xorshift32() % 4u);
                    else if (r < 80u) e = (int32_t)(4u + (xorshift32() % 8u));
                    else e = (int32_t)(12u + (xorshift32() % (n_experts - 12u)));
                } else {
                    e = (int32_t)(xorshift32() % n_experts);
                }
                int dup = 0;
                for (uint32_t p = 0; p < s; ++p) {
                    if (h_ids[t * k + p] == e) { dup = 1; break; }
                }
                if (!dup) break;
            }
            h_ids[t * k + s] = e;
            h_w[t * k + s] = rand_f32(0.05f, 0.5f);
        }
    }
}

static void verify_case_0ulp(const char *metallib_path, const char *tag,
                             uint32_t dim, uint32_t expert_dim,
                             uint32_t n_experts, uint32_t k, uint32_t n_tok,
                             const uint8_t *h_gate, const uint8_t *h_up, const uint8_t *h_down,
                             const float *h_x, const int32_t *h_ids, const float *h_w,
                             int check_counters) {
    const size_t gate_exp_bytes = (size_t)expert_dim * (dim / 256u) * 66u;
    const size_t down_exp_bytes = (size_t)dim * (expert_dim / 256u) * 84u;
    const size_t out_bytes = (size_t)n_tok * dim * sizeof(float);
    const size_t mid_bytes = (size_t)n_tok * k * expert_dim * sizeof(uint16_t);
    const size_t sd_bytes  = (size_t)n_tok * k * dim * sizeof(float);

    float *ref_out = (float *)malloc(out_bytes);
    float *got_out = (float *)malloc(out_bytes);
    REQUIRE(ref_out && got_out);

    /* Arm 0: baseline (WIDE=0, M1=0) */
    setenv("MYDS_MOE_PREFILL_WIDE", "0", 1);
    setenv("MYDS_MOE_PREFILL_M1", "0", 1);
    REQUIRE(myds_metal_init(metallib_path) == 1);
    {
        myds_metal_buffer_t *b_gate = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
        myds_metal_buffer_t *b_up   = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
        myds_metal_buffer_t *b_down = myds_metal_buffer_alloc(n_experts * down_exp_bytes);
        myds_metal_buffer_t *b_x    = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
        myds_metal_buffer_t *b_ids  = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(int32_t));
        myds_metal_buffer_t *b_w    = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(float));
        myds_metal_buffer_t *b_sg   = myds_metal_buffer_alloc(64u);
        myds_metal_buffer_t *b_su   = myds_metal_buffer_alloc(64u);
        myds_metal_buffer_t *b_sm   = myds_metal_buffer_alloc(mid_bytes);
        myds_metal_buffer_t *b_sd   = myds_metal_buffer_alloc(sd_bytes);
        myds_metal_buffer_t *b_out  = myds_metal_buffer_alloc(out_bytes);
        REQUIRE(b_gate && b_up && b_down && b_x && b_ids && b_w && b_sg && b_su && b_sm && b_sd && b_out);

        memcpy(myds_metal_buffer_data(b_gate), h_gate, n_experts * gate_exp_bytes);
        memcpy(myds_metal_buffer_data(b_up),   h_up,   n_experts * gate_exp_bytes);
        memcpy(myds_metal_buffer_data(b_down), h_down, n_experts * down_exp_bytes);
        memcpy(myds_metal_buffer_data(b_x),    h_x,    (size_t)n_tok * dim * sizeof(float));
        memcpy(myds_metal_buffer_data(b_ids),  h_ids,  (size_t)n_tok * k * sizeof(int32_t));
        memcpy(myds_metal_buffer_data(b_w),    h_w,    (size_t)n_tok * k * sizeof(float));

        REQUIRE(myds_metal_routed_moe_batch(b_out, b_x, b_gate, b_up, b_down, b_ids, b_w,
                b_sg, b_su, b_sm, b_sd, dim, expert_dim, n_experts, k, n_tok,
                GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 7.0f) == 1);
        memcpy(ref_out, myds_metal_buffer_data(b_out), out_bytes);

        myds_metal_buffer_free(b_gate); myds_metal_buffer_free(b_up); myds_metal_buffer_free(b_down);
        myds_metal_buffer_free(b_x); myds_metal_buffer_free(b_ids); myds_metal_buffer_free(b_w);
        myds_metal_buffer_free(b_sg); myds_metal_buffer_free(b_su); myds_metal_buffer_free(b_sm);
        myds_metal_buffer_free(b_sd); myds_metal_buffer_free(b_out);
    }
    myds_metal_cleanup();

    /* Arm 1: [P-M1] (WIDE=1, M1=1) */
    setenv("MYDS_MOE_PREFILL_WIDE", "1", 1);
    setenv("MYDS_MOE_PREFILL_M1", "1", 1);
    REQUIRE(myds_metal_init(metallib_path) == 1);
    REQUIRE(myds_metal_moe_prefill_m1_enabled() == 1);
    if (check_counters) {
        myds_metal_moe_prefill_m1_counters(NULL, true);
    }
    {
        myds_metal_buffer_t *b_gate = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
        myds_metal_buffer_t *b_up   = myds_metal_buffer_alloc(n_experts * gate_exp_bytes);
        myds_metal_buffer_t *b_down = myds_metal_buffer_alloc(n_experts * down_exp_bytes);
        myds_metal_buffer_t *b_x    = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
        myds_metal_buffer_t *b_ids  = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(int32_t));
        myds_metal_buffer_t *b_w    = myds_metal_buffer_alloc((size_t)n_tok * k * sizeof(float));
        myds_metal_buffer_t *b_sg   = myds_metal_buffer_alloc(64u);
        myds_metal_buffer_t *b_su   = myds_metal_buffer_alloc(64u);
        myds_metal_buffer_t *b_sm   = myds_metal_buffer_alloc(mid_bytes);
        myds_metal_buffer_t *b_sd   = myds_metal_buffer_alloc(sd_bytes);
        myds_metal_buffer_t *b_out  = myds_metal_buffer_alloc(out_bytes);
        REQUIRE(b_gate && b_up && b_down && b_x && b_ids && b_w && b_sg && b_su && b_sm && b_sd && b_out);

        memcpy(myds_metal_buffer_data(b_gate), h_gate, n_experts * gate_exp_bytes);
        memcpy(myds_metal_buffer_data(b_up),   h_up,   n_experts * gate_exp_bytes);
        memcpy(myds_metal_buffer_data(b_down), h_down, n_experts * down_exp_bytes);
        memcpy(myds_metal_buffer_data(b_x),    h_x,    (size_t)n_tok * dim * sizeof(float));
        memcpy(myds_metal_buffer_data(b_ids),  h_ids,  (size_t)n_tok * k * sizeof(int32_t));
        memcpy(myds_metal_buffer_data(b_w),    h_w,    (size_t)n_tok * k * sizeof(float));

        REQUIRE(myds_metal_routed_moe_batch(b_out, b_x, b_gate, b_up, b_down, b_ids, b_w,
                b_sg, b_su, b_sm, b_sd, dim, expert_dim, n_experts, k, n_tok,
                GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 7.0f) == 1);
        memcpy(got_out, myds_metal_buffer_data(b_out), out_bytes);

        if (check_counters) {
            myds_metal_moe_prefill_m1_counters_t c;
            myds_metal_moe_prefill_m1_counters(&c, false);
            REQUIRE(c.calls == 1u);
            REQUIRE(c.map0_m1_encoded == 1u);
            REQUIRE(c.narrow_compact_encoded == 1u);
            /* Wide down stays on prod q2_K_f16_wide (Apple8 maxThr=704 vs wide_m1 576, #54). */
            REQUIRE(c.down_wide_m1_encoded == 0u);
            REQUIRE(c.down_narrow_m1_encoded == 1u);
        }

        myds_metal_buffer_free(b_gate); myds_metal_buffer_free(b_up); myds_metal_buffer_free(b_down);
        myds_metal_buffer_free(b_x); myds_metal_buffer_free(b_ids); myds_metal_buffer_free(b_w);
        myds_metal_buffer_free(b_sg); myds_metal_buffer_free(b_su); myds_metal_buffer_free(b_sm);
        myds_metal_buffer_free(b_sd); myds_metal_buffer_free(b_out);
    }
    myds_metal_cleanup();

    if (memcmp(ref_out, got_out, out_bytes) != 0) {
        size_t diffs = 0;
        for (size_t i = 0; i < (size_t)n_tok * dim; ++i) {
            uint32_t ua = 0, ub = 0;
            memcpy(&ua, ref_out + i, 4u);
            memcpy(&ub, got_out + i, 4u);
            if (ua != ub) {
                if (++diffs <= 5u) {
                    fprintf(stderr, "  [DIFF] %s idx=%zu ref=%g (0x%08x) got=%g (0x%08x)\n",
                            tag, i, ref_out[i], ua, got_out[i], ub);
                }
            }
        }
        fprintf(stderr, "FAIL %s: %zu diffs\n", tag, diffs);
        exit(1);
    }
    printf("[PASS] %s: strict 0-ULP (memcmp==0, %u tok x %u dim, E=%u, k=%u)\n",
           tag, n_tok, dim, n_experts, k);

    free(ref_out);
    free(got_out);
}

static void run_0ulp_suite(const char *metallib_path) {
    /* Force MM path down to n_tok=1 so n_tok in {1, 15, 16, 17} exercise compact_tail_fast culling! */
    setenv("MYDS_MOE_MM_MIN_TOKENS", "1", 1);

    /* 1. Full DeepSeek-V4.1 dims (5120 x 2304), n_experts=32, across n_tok in {1,15,16,17,32,312,2170}. */
    {
        const uint32_t dim = 5120u, edim = 2304u, nexp = 32u, max_tok = 2170u;
        const size_t gate_exp_bytes = (size_t)edim * (dim / 256u) * 66u;
        const size_t down_exp_bytes = (size_t)dim * (edim / 256u) * 84u;
        uint8_t *h_gate = (uint8_t *)malloc(nexp * gate_exp_bytes);
        uint8_t *h_up   = (uint8_t *)malloc(nexp * gate_exp_bytes);
        uint8_t *h_down = (uint8_t *)malloc(nexp * down_exp_bytes);
        float   *h_x    = (float *)malloc((size_t)max_tok * dim * sizeof(float));
        int32_t *h_ids  = (int32_t *)malloc((size_t)max_tok * 6u * sizeof(int32_t));
        float   *h_w    = (float *)malloc((size_t)max_tok * 6u * sizeof(float));
        REQUIRE(h_gate && h_up && h_down && h_x && h_ids && h_w);

        rng_state = 0x51a7beefu;
        fill_iq2_xxs(h_gate, (size_t)nexp * edim * (dim / 256u));
        fill_iq2_xxs(h_up,   (size_t)nexp * edim * (dim / 256u));
        fill_q2_k(h_down,    (size_t)nexp * dim * (edim / 256u));
        for (size_t i = 0; i < (size_t)max_tok * dim; ++i) h_x[i] = rand_f32(-1.5f, 1.5f);

        /* Exact tail-cull boundary cases (force expert 0 to receive exactly nr1 = 1, 15, 16, 17, 32 rows!) */
        const uint32_t tail_toks[] = {1u, 15u, 16u, 17u, 32u};
        for (size_t ti = 0; ti < sizeof(tail_toks) / sizeof(*tail_toks); ++ti) {
            const uint32_t nt = tail_toks[ti];
            for (uint32_t t = 0; t < nt; ++t) {
                for (uint32_t s = 0; s < 6u; ++s) {
                    h_ids[t * 6u + s] = (int32_t)s; /* experts 0..5 each get EXACTLY nt rows! */
                    h_w[t * 6u + s] = rand_f32(0.05f, 0.5f);
                }
            }
            char tag[64];
            snprintf(tag, sizeof(tag), "full5120x2304_exact_nr1_%u", nt);
            verify_case_0ulp(metallib_path, tag, dim, edim, nexp, 6u, nt,
                             h_gate, h_up, h_down, h_x, h_ids, h_w, ti == 0 ? 1 : 0);
        }

        /* Realistic skewed routing on 312 & 2170 tokens (mixed wide + narrow tiles), k in {6, 4, 3}. */
        const uint32_t med_toks[] = {312u, 2170u};
        for (size_t ti = 0; ti < sizeof(med_toks) / sizeof(*med_toks); ++ti) {
            const uint32_t nt = med_toks[ti];
            generate_routing(h_ids, h_w, nt, 6u, nexp, 1);
            char tag[64];
            snprintf(tag, sizeof(tag), "full5120x2304_skewed_k6_n%u", nt);
            verify_case_0ulp(metallib_path, tag, dim, edim, nexp, 6u, nt,
                             h_gate, h_up, h_down, h_x, h_ids, h_w, 0);
        }
        const uint32_t alt_ks[] = {4u, 3u};
        for (size_t ki = 0; ki < 2u; ++ki) {
            generate_routing(h_ids, h_w, 312u, alt_ks[ki], nexp, 1);
            char tag[64];
            snprintf(tag, sizeof(tag), "full5120x2304_skewed_k%u_n312", alt_ks[ki]);
            verify_case_0ulp(metallib_path, tag, dim, edim, nexp, alt_ks[ki], 312u,
                             h_gate, h_up, h_down, h_x, h_ids, h_w, 0);
        }

        free(h_gate); free(h_up); free(h_down); free(h_x); free(h_ids); free(h_w);
    }

    /* 2. Full 384-expert production topology (exercises 12-16 SG SIMD prefix sum in map0_wide_m1)
     *    across ALL 8 required token counts: {1, 15, 16, 17, 32, 312, 2170, 16384}! */
    {
        const uint32_t dim = 512u, edim = 256u, nexp = 384u, max_tok = 16384u;
        const size_t gate_exp_bytes = (size_t)edim * (dim / 256u) * 66u;
        const size_t down_exp_bytes = (size_t)dim * (edim / 256u) * 84u;
        uint8_t *h_gate = (uint8_t *)malloc(nexp * gate_exp_bytes);
        uint8_t *h_up   = (uint8_t *)malloc(nexp * gate_exp_bytes);
        uint8_t *h_down = (uint8_t *)malloc(nexp * down_exp_bytes);
        float   *h_x    = (float *)malloc((size_t)max_tok * dim * sizeof(float));
        int32_t *h_ids  = (int32_t *)malloc((size_t)max_tok * 6u * sizeof(int32_t));
        float   *h_w    = (float *)malloc((size_t)max_tok * 6u * sizeof(float));
        REQUIRE(h_gate && h_up && h_down && h_x && h_ids && h_w);

        rng_state = 0x13579bdfu;
        fill_iq2_xxs(h_gate, (size_t)nexp * edim * (dim / 256u));
        fill_iq2_xxs(h_up,   (size_t)nexp * edim * (dim / 256u));
        fill_q2_k(h_down,    (size_t)nexp * dim * (edim / 256u));
        for (size_t i = 0; i < (size_t)max_tok * dim; ++i) h_x[i] = rand_f32(-1.5f, 1.5f);

        const uint32_t all_toks[] = {1u, 15u, 16u, 17u, 32u, 312u, 2170u, 16384u};
        for (size_t ti = 0; ti < sizeof(all_toks) / sizeof(*all_toks); ++ti) {
            const uint32_t nt = all_toks[ti];
            generate_routing(h_ids, h_w, nt, 6u, nexp, 0);
            char tag[64];
            snprintf(tag, sizeof(tag), "exp384_512x256_k6_n%u", nt);
            verify_case_0ulp(metallib_path, tag, dim, edim, nexp, 6u, nt,
                             h_gate, h_up, h_down, h_x, h_ids, h_w, ti == 7u ? 1 : 0);
        }

        free(h_gate); free(h_up); free(h_down); free(h_x); free(h_ids); free(h_w);
    }
}

struct m1_bench_map0_args {
    int32_t ne02, ne10, ne11;
    uint64_t nb11, nb12;
    int32_t ne21, ne20;
    uint64_t nb21;
};
struct m1_bench_mm_args {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne11;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne20, ne21, ne0, ne1, r2, r3, tp_rank, tp_world, tp_expert_base;
};
struct m1_bench_swiglu_args {
    uint32_t width;
    uint64_t rows, gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

static void run_m4_per_kernel_bench(const char *metallib_path) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path)] error:&err];
        REQUIRE(dev && lib);
        id<MTLCommandQueue> q = [dev newCommandQueue];

        id<MTLComputePipelineState> p_map_old = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_mul_mm_id_map0_wide_ne20_6"] error:&err];
        id<MTLComputePipelineState> p_map_m1  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_mul_mm_id_map0_wide_m1_ne20_6"] error:&err];
        id<MTLComputePipelineState> p_nar_old = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_tail_cull"] error:&err];
        id<MTLComputePipelineState> p_nar_m1  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_compact_tail_fast"] error:&err];
        id<MTLComputePipelineState> p_dn_old  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_mul_mm_id_q2_K_f16_wide"] error:&err];
        id<MTLComputePipelineState> p_dn_m1   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_mul_mm_id_q2_K_f16_wide_m1"] error:&err];
        REQUIRE(p_map_old && p_map_m1 && p_nar_old && p_nar_m1 && p_dn_old && p_dn_m1);

        printf("\n=== [P-M1] Interleaved GPU-Timed Per-Kernel Microbenchmark (M4) ===\n");

        /* 1. map0_wide vs map0_wide_m1 (E=384, k=6, N in {2170, 16384}) */
        const uint32_t map_ns[] = {2170u, 16384u};
        for (size_t mi = 0; mi < 2u; ++mi) {
            const uint32_t ne = 384u, k = 6u, nt = map_ns[mi];
            id<MTLBuffer> b_ids = [dev newBufferWithLength:(size_t)nt * k * 4u options:MTLResourceStorageModeShared];
            int32_t *ip = b_ids.contents;
            for (uint32_t t = 0; t < nt; ++t)
                for (uint32_t s = 0; s < k; ++s)
                    ip[t * k + s] = (int32_t)((t * 7u + s * 61u) % ne);
            const size_t tpe = ne * 4u, hids = (size_t)ne * nt * 4u;
            const size_t wo = (tpe + hids + 7u) & ~7u;
            const uint64_t wc = ((uint64_t)nt * k / 64u) + ne;
            const size_t wno = wo + 8u + wc * 8u;
            id<MTLBuffer> b_map = [dev newBufferWithLength:wno + 8u + ne * 8u options:MTLResourceStorageModeShared];
            struct m1_bench_map0_args ma = {
                .ne02=(int32_t)ne, .ne10=5120, .ne11=1, .nb11=5120*4u, .nb12=5120*4u,
                .ne21=(int32_t)nt, .ne20=(int32_t)k, .nb21=k*4u
            };
            double ts[2][10];
            for (int r = 0; r < 10; ++r) {
                for (int o = 0; o < 2; ++o) {
                    int v = (r + o) & 1;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int rep = 0; rep < 10; ++rep) {
                        [enc setComputePipelineState:v ? p_map_m1 : p_map_old];
                        [enc setBytes:&ma length:sizeof(ma) atIndex:0];
                        [enc setBuffer:b_ids offset:0 atIndex:1];
                        [enc setBuffer:b_map offset:0 atIndex:2];
                        [enc setBuffer:b_map offset:tpe atIndex:3];
                        [enc setBuffer:b_map offset:wo atIndex:4];
                        [enc setBuffer:b_map offset:wno atIndex:5];
                        [enc setThreadgroupMemoryLength:v ? ((((ne+3u)&~3u)+64u)*4u) : (ne*k*2u) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(v ? 512u : ne, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    ts[v][r] = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 / 10.0;
                }
            }
            qsort(ts[0], 10, sizeof(double), cmp_dbl);
            qsort(ts[1], 10, sizeof(double), cmp_dbl);
            printf("  map0 (E=384, k=6, N=%5u)       : old=%.3f ms  m1=%.3f ms  (%.2fx)\n",
                   nt, ts[0][5], ts[1][5], ts[0][5] / ts[1][5]);
        }

        /* 2. pair_swiglu_narrow: tail_cull (16KB, 2 TGs/core, legacy IQ2) vs compact_tail_fast (8KB, 4 TGs/core, branchless IQ2) */
        {
            const uint32_t dim = 5120u, edim = 2304u, nexp = 64u, k = 6u, nt = 200u; /* ~18.8 rows/exp -> narrow-only */
            const uint64_t g_rb = (dim / 256u) * 66u, g_eb = edim * g_rb;
            const uint64_t mid_rb = edim * 2u;
            id<MTLBuffer> bg = [dev newBufferWithLength:nexp * g_eb options:MTLResourceStorageModeShared];
            id<MTLBuffer> bu = [dev newBufferWithLength:nexp * g_eb options:MTLResourceStorageModeShared];
            id<MTLBuffer> bx = [dev newBufferWithLength:(size_t)nt * dim * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bi = [dev newBufferWithLength:(size_t)nt * k * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bw = [dev newBufferWithLength:(size_t)nt * k * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bm = [dev newBufferWithLength:(size_t)nt * k * edim * 2u options:MTLResourceStorageModeShared];
            fill_iq2_xxs(bg.contents, (size_t)nexp * edim * (dim / 256u));
            fill_iq2_xxs(bu.contents, (size_t)nexp * edim * (dim / 256u));
            float *xp = bx.contents; for (size_t i = 0; i < (size_t)nt * dim; ++i) xp[i] = rand_f32(-1.f, 1.f);
            generate_routing(bi.contents, bw.contents, nt, k, nexp, 0);
            const size_t tpe = nexp * 4u, hids = (size_t)nexp * nt * 4u, wo = (tpe + hids + 7u) & ~7u;
            const uint64_t wc = ((uint64_t)nt * k / 64u) + nexp;
            const size_t wno = wo + 8u + wc * 8u;
            id<MTLBuffer> bmap = [dev newBufferWithLength:wno + 8u + nexp * 8u options:MTLResourceStorageModeShared];
            struct m1_bench_map0_args ma = { (int32_t)nexp, (int32_t)dim, 1, dim*4u, dim*4u, (int32_t)nt, (int32_t)k, k*4u };
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:p_map_m1];
                [enc setBytes:&ma length:sizeof(ma) atIndex:0];
                [enc setBuffer:bi offset:0 atIndex:1];
                [enc setBuffer:bmap offset:0 atIndex:2];
                [enc setBuffer:bmap offset:tpe atIndex:3];
                [enc setBuffer:bmap offset:wo atIndex:4];
                [enc setBuffer:bmap offset:wno atIndex:5];
                [enc setThreadgroupMemoryLength:((((nexp+3u)&~3u)+64u)*4u) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            struct m1_bench_mm_args ga = {
                (int32_t)dim, (int32_t)nexp, g_rb, g_eb, nexp * g_eb, 1, 4u, dim*4u, dim*4u, (uint64_t)nt*dim*4u,
                (int32_t)k, (int32_t)nt, (int32_t)edim, (int32_t)k, 1, 1, 0, 1, 0
            };
            struct m1_bench_swiglu_args sa = { edim, (uint64_t)nt*k, edim*4u, edim*4u, mid_rb, 4u, 0u, 7.0f };
            double ts[2][10];
            for (int r = 0; r < 10; ++r) {
                for (int o = 0; o < 2; ++o) {
                    int v = (r + o) & 1;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int rep = 0; rep < 3; ++rep) {
                        [enc setComputePipelineState:v ? p_nar_m1 : p_nar_old];
                        [enc setBytes:&ga length:sizeof(ga) atIndex:0];
                        [enc setBytes:&sa length:sizeof(sa) atIndex:1];
                        [enc setBuffer:bg offset:0 atIndex:2];
                        [enc setBuffer:bu offset:0 atIndex:3];
                        [enc setBuffer:bx offset:0 atIndex:4];
                        [enc setBuffer:bmap offset:0 atIndex:5];
                        [enc setBuffer:bmap offset:tpe atIndex:6];
                        [enc setBuffer:bm offset:0 atIndex:7];
                        [enc setBuffer:bw offset:0 atIndex:8];
                        [enc setBuffer:bmap offset:wno atIndex:9];
                        [enc setThreadgroupMemoryLength:v ? 8192u : 16384u atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(nexp, v ? ((edim+31)/32) : ((edim+63)/64), 1)
                            threadsPerThreadgroup:MTLSizeMake(v ? 64 : 128, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    ts[v][r] = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 / 3.0;
                }
            }
            qsort(ts[0], 10, sizeof(double), cmp_dbl);
            qsort(ts[1], 10, sizeof(double), cmp_dbl);
            printf("  pair_swiglu_narrow (5120x2304)   : old(16KB)=%.3f ms  m1(8KB)=%.3f ms  (%+.1f%%)\n",
                   ts[0][5], ts[1][5], 100.0 * (ts[0][5] - ts[1][5]) / ts[0][5]);
        }

        /* 3. down_q2_K_wide: baseline vs m1 (u32 dequant + 20-reg streaming + float4 wb) */
        {
            const uint32_t dim = 5120u, edim = 2304u, nexp = 32u, k = 6u, nt = 512u;
            const uint64_t d_rb = (edim / 256u) * 84u, d_eb = dim * d_rb, mid_rb = edim * 2u;
            id<MTLBuffer> bd = [dev newBufferWithLength:nexp * d_eb options:MTLResourceStorageModeShared];
            id<MTLBuffer> bm = [dev newBufferWithLength:(size_t)nt * k * edim * 2u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bi = [dev newBufferWithLength:(size_t)nt * k * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bw = [dev newBufferWithLength:(size_t)nt * k * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bo = [dev newBufferWithLength:(size_t)nt * k * dim * 4u options:MTLResourceStorageModeShared];
            fill_q2_k(bd.contents, (size_t)nexp * dim * (edim / 256u));
            __fp16 *mp = bm.contents; for (size_t i = 0; i < (size_t)nt * k * edim; ++i) mp[i] = (__fp16)rand_f32(-1.f, 1.f);
            generate_routing(bi.contents, bw.contents, nt, k, nexp, 0);
            const size_t tpe = nexp * 4u, hids = (size_t)nexp * nt * 4u, wo = (tpe + hids + 7u) & ~7u;
            const uint64_t wc = ((uint64_t)nt * k / 64u) + nexp;
            const size_t wno = wo + 8u + wc * 8u;
            id<MTLBuffer> bmap = [dev newBufferWithLength:wno + 8u + nexp * 8u options:MTLResourceStorageModeShared];
            struct m1_bench_map0_args ma = { (int32_t)nexp, (int32_t)dim, 1, dim*4u, dim*4u, (int32_t)nt, (int32_t)k, k*4u };
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:p_map_m1];
                [enc setBytes:&ma length:sizeof(ma) atIndex:0];
                [enc setBuffer:bi offset:0 atIndex:1];
                [enc setBuffer:bmap offset:0 atIndex:2];
                [enc setBuffer:bmap offset:tpe atIndex:3];
                [enc setBuffer:bmap offset:wo atIndex:4];
                [enc setBuffer:bmap offset:wno atIndex:5];
                [enc setThreadgroupMemoryLength:((((nexp+3u)&~3u)+64u)*4u) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            struct m1_bench_mm_args da = {
                (int32_t)edim, (int32_t)nexp, d_rb, d_eb, nexp * d_eb, (int32_t)k, 2u, mid_rb,
                k * mid_rb, (uint64_t)nt * k * mid_rb, (int32_t)k, (int32_t)nt, (int32_t)dim, (int32_t)k, 1, 1, 0, 1, 0
            };
            double ts[2][10];
            for (int r = 0; r < 10; ++r) {
                for (int o = 0; o < 2; ++o) {
                    int v = (r + o) & 1;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int rep = 0; rep < 3; ++rep) {
                        [enc setComputePipelineState:v ? p_dn_m1 : p_dn_old];
                        [enc setBytes:&da length:sizeof(da) atIndex:0];
                        [enc setBuffer:bd offset:0 atIndex:1];
                        [enc setBuffer:bm offset:0 atIndex:2];
                        [enc setBuffer:bmap offset:0 atIndex:3];
                        [enc setBuffer:bmap offset:tpe atIndex:4];
                        [enc setBuffer:bo offset:0 atIndex:5];
                        [enc setBuffer:bmap offset:wo atIndex:6];
                        [enc setThreadgroupMemoryLength:8192u atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(wc, (dim+63)/64, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    ts[v][r] = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 / 3.0;
                }
            }
            qsort(ts[0], 10, sizeof(double), cmp_dbl);
            qsort(ts[1], 10, sizeof(double), cmp_dbl);
            printf("  down_q2_K_wide (5120x2304,N=512) : old=%.3f ms  m1=%.3f ms  (%+.1f%%)\n",
                   ts[0][5], ts[1][5], 100.0 * (ts[0][5] - ts[1][5]) / ts[0][5]);
        }
    }
}

static void run_m4_bench(const char *metallib_path) {
    run_m4_per_kernel_bench(metallib_path);
    printf("\n=== [P-M1] M4 End-to-End Microbenchmark (Full DeepSeek-V4.1 5120x2304, k=6) ===\n");
    const uint32_t dim = 5120u, edim = 2304u, k = 6u;
    struct { const char *name; uint32_t nexp; uint32_t ntok; int skewed; int iters; } cases[] = {
        /* 362*6/64 = 33.9 rows/expert -> exact warm_128k+2k (2170 tok / 384 exp) narrow-heavy mix! */
        {"warm2k_mix (33.9 rows/exp, E=64, N=362)",  64u, 362u, 0, 6},
        /* 200*6/64 = 18.8 rows/expert -> ~90% narrow tiles, heavily exercises 8KB compact_tail_fast! */
        {"narrow_dom (18.8 rows/exp, E=64, N=200)",  64u, 200u, 0, 8},
        /* 1024*6/32 = 192 rows/expert -> cold-prefill wide-dominated mix! */
        {"cold_wide  (192 rows/exp,  E=32, N=1024)", 32u, 1024u, 0, 4},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(*cases); ++ci) {
        const uint32_t nexp = cases[ci].nexp;
        const uint32_t ntok = cases[ci].ntok;
        const size_t gate_exp_bytes = (size_t)edim * (dim / 256u) * 66u;
        const size_t down_exp_bytes = (size_t)dim * (edim / 256u) * 84u;

        uint8_t *h_gate = (uint8_t *)malloc(nexp * gate_exp_bytes);
        uint8_t *h_up   = (uint8_t *)malloc(nexp * gate_exp_bytes);
        uint8_t *h_down = (uint8_t *)malloc(nexp * down_exp_bytes);
        float   *h_x    = (float *)malloc((size_t)ntok * dim * sizeof(float));
        int32_t *h_ids  = (int32_t *)malloc((size_t)ntok * k * sizeof(int32_t));
        float   *h_w    = (float *)malloc((size_t)ntok * k * sizeof(float));
        REQUIRE(h_gate && h_up && h_down && h_x && h_ids && h_w);

        rng_state = 0xabcdef01u + (uint32_t)ci;
        fill_iq2_xxs(h_gate, (size_t)nexp * edim * (dim / 256u));
        fill_iq2_xxs(h_up,   (size_t)nexp * edim * (dim / 256u));
        fill_q2_k(h_down,    (size_t)nexp * dim * (edim / 256u));
        for (size_t i = 0; i < (size_t)ntok * dim; ++i) h_x[i] = rand_f32(-1.5f, 1.5f);
        generate_routing(h_ids, h_w, ntok, k, nexp, cases[ci].skewed);

        double med_ms[3] = {0, 0, 0};
        const char *arm_names[3] = {"base (W=0,M1=0)", "wide (W=1,M1=0)", "M1   (W=1,M1=1)"};
        const char *w_vals[3] = {"0", "1", "1"};
        const char *m_vals[3] = {"0", "0", "1"};

        for (int arm = 0; arm < 3; ++arm) {
            setenv("MYDS_MOE_PREFILL_WIDE", w_vals[arm], 1);
            setenv("MYDS_MOE_PREFILL_M1",   m_vals[arm], 1);
            REQUIRE(myds_metal_init(metallib_path) == 1);

            myds_metal_buffer_t *b_gate = myds_metal_buffer_alloc(nexp * gate_exp_bytes);
            myds_metal_buffer_t *b_up   = myds_metal_buffer_alloc(nexp * gate_exp_bytes);
            myds_metal_buffer_t *b_down = myds_metal_buffer_alloc(nexp * down_exp_bytes);
            myds_metal_buffer_t *b_x    = myds_metal_buffer_alloc((size_t)ntok * dim * sizeof(float));
            myds_metal_buffer_t *b_ids  = myds_metal_buffer_alloc((size_t)ntok * k * sizeof(int32_t));
            myds_metal_buffer_t *b_w    = myds_metal_buffer_alloc((size_t)ntok * k * sizeof(float));
            myds_metal_buffer_t *b_sg   = myds_metal_buffer_alloc(64u);
            myds_metal_buffer_t *b_su   = myds_metal_buffer_alloc(64u);
            myds_metal_buffer_t *b_sm   = myds_metal_buffer_alloc((size_t)ntok * k * edim * sizeof(uint16_t));
            myds_metal_buffer_t *b_sd   = myds_metal_buffer_alloc((size_t)ntok * k * dim * sizeof(float));
            myds_metal_buffer_t *b_out  = myds_metal_buffer_alloc((size_t)ntok * dim * sizeof(float));
            REQUIRE(b_gate && b_up && b_down && b_x && b_ids && b_w && b_sg && b_su && b_sm && b_sd && b_out);

            memcpy(myds_metal_buffer_data(b_gate), h_gate, nexp * gate_exp_bytes);
            memcpy(myds_metal_buffer_data(b_up),   h_up,   nexp * gate_exp_bytes);
            memcpy(myds_metal_buffer_data(b_down), h_down, nexp * down_exp_bytes);
            memcpy(myds_metal_buffer_data(b_x),    h_x,    (size_t)ntok * dim * sizeof(float));
            memcpy(myds_metal_buffer_data(b_ids),  h_ids,  (size_t)ntok * k * sizeof(int32_t));
            memcpy(myds_metal_buffer_data(b_w),    h_w,    (size_t)ntok * k * sizeof(float));

            /* Warmup */
            for (int w = 0; w < 2; ++w) {
                REQUIRE(myds_metal_routed_moe_batch(b_out, b_x, b_gate, b_up, b_down, b_ids, b_w,
                        b_sg, b_su, b_sm, b_sd, dim, edim, nexp, k, ntok,
                        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 7.0f) == 1);
            }
            double samples[16];
            int iters = cases[ci].iters;
            for (int it = 0; it < iters; ++it) {
                double t0 = now_ms();
                REQUIRE(myds_metal_routed_moe_batch(b_out, b_x, b_gate, b_up, b_down, b_ids, b_w,
                        b_sg, b_su, b_sm, b_sd, dim, edim, nexp, k, ntok,
                        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 7.0f) == 1);
                samples[it] = now_ms() - t0;
            }
            qsort(samples, (size_t)iters, sizeof(double), cmp_dbl);
            med_ms[arm] = samples[iters / 2];

            myds_metal_buffer_free(b_gate); myds_metal_buffer_free(b_up); myds_metal_buffer_free(b_down);
            myds_metal_buffer_free(b_x); myds_metal_buffer_free(b_ids); myds_metal_buffer_free(b_w);
            myds_metal_buffer_free(b_sg); myds_metal_buffer_free(b_su); myds_metal_buffer_free(b_sm);
            myds_metal_buffer_free(b_sd); myds_metal_buffer_free(b_out);
            myds_metal_cleanup();
        }
        printf("  %-40s | %s=%.2f ms | %s=%.2f ms | %s=%.2f ms (vs wide: %+.1f%%, vs base: %+.1f%%)\n",
               cases[ci].name,
               arm_names[0], med_ms[0],
               arm_names[1], med_ms[1],
               arm_names[2], med_ms[2],
               100.0 * (med_ms[1] - med_ms[2]) / med_ms[1],
               100.0 * (med_ms[0] - med_ms[2]) / med_ms[0]);

        free(h_gate); free(h_up); free(h_down); free(h_x); free(h_ids); free(h_w);
    }
}

int main(int argc, char **argv) {
    const char *metallib = "bin/myds.metallib";
    int do_bench = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) do_bench = 1;
        else metallib = argv[i];
    }

    check_fail_closed_env(metallib);
    check_pso_occupancy(metallib);
    run_0ulp_suite(metallib);
    if (do_bench) {
        run_m4_bench(metallib);
    }
    printf("\n=== [P-M1] ALL TESTS PASSED ===\n");
    return 0;
}
