/* [P-E1] Strict 0-ULP unit test, concurrent-encoder barrier stress test,
 * PSO occupancy check, and M4 microbenchmark for MYDS_MOE_EPILOGUE_FUSE. */
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

static uint32_t rng_state = 0x76543219u;
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
        for (size_t i = 2u; i < 66u; i++) p[i] = (uint8_t)(xorshift32() & 0xffu);
    }
}

static void fill_q2_k(uint8_t *buf, size_t n_blocks) {
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *p = buf + b * 84u;
        for (size_t i = 0; i < 80u; i++) p[i] = (uint8_t)(xorshift32() & 0xffu);
        uint16_t d = f32_to_f16_bits(rand_f32(0.01f, 0.25f));
        uint16_t dmin = f32_to_f16_bits(rand_f32(0.005f, 0.12f));
        memcpy(p + 80u, &d, 2u);
        memcpy(p + 82u, &dmin, 2u);
    }
}

static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db);
}

static void check_fail_closed_env(const char *metallib_path) {
    const char *bad[] = {"", "2", "-1", "true", " 1", "01"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
        setenv("MYDS_MOE_EPILOGUE_FUSE", bad[i], 1);
        REQUIRE(myds_metal_init(metallib_path) == 0);
    }
    unsetenv("MYDS_MOE_EPILOGUE_FUSE");
    REQUIRE(myds_metal_init(metallib_path) == 1);
    REQUIRE(myds_metal_moe_epilogue_fuse_enabled() == 0);
    myds_metal_cleanup();

    setenv("MYDS_MOE_EPILOGUE_FUSE", "0", 1);
    REQUIRE(myds_metal_init(metallib_path) == 1);
    REQUIRE(myds_metal_moe_epilogue_fuse_enabled() == 0);
    myds_metal_cleanup();

    setenv("MYDS_MOE_EPILOGUE_FUSE", "1", 1);
    REQUIRE(myds_metal_init(metallib_path) == 1);
    REQUIRE(myds_metal_moe_epilogue_fuse_enabled() == 1);
    myds_metal_cleanup();

    printf("[PASS] MYDS_MOE_EPILOGUE_FUSE fail-closed env parsing\n");
}

static void check_pso_occupancy(const char *metallib_path) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        REQUIRE(dev != nil);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path)] error:&err];
        REQUIRE(lib != nil);
        const char *names[] = {
            "kernel_dsv4_moe_sum1_add_hc_expand4_f4",
            "kernel_dsv4_moe_sum3_add_hc_expand4_f4",
            "kernel_dsv4_moe_sum4_add_hc_expand4_f4",
            "kernel_dsv4_moe_sum6_add_hc_expand4_f4",
        };
        printf("[PSO] Epilogue fuse occupancy on %s:\n", dev.name.UTF8String);
        for (size_t i = 0; i < 4u; ++i) {
            id<MTLFunction> fn = [lib newFunctionWithName:@(names[i])];
            REQUIRE(fn != nil);
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            REQUIRE(pso != nil);
            REQUIRE(pso.maxTotalThreadsPerThreadgroup >= 256u);
            REQUIRE(pso.staticThreadgroupMemoryLength == 0u);
            printf("  %-44s maxThr=%4lu staticSmem=%4lu\n",
                   names[i],
                   (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                   (unsigned long)pso.staticThreadgroupMemoryLength);
        }
    }
}

/* Baseline reference: exact kernel_dsv4_moe_sum{3,4,6}_f32 + chunked (2048)
 * myds_metal_v41_moe_add_hc_expand_batch -> layer_out_buf -> copy to prompt_res + prompt_pre. */
struct moe_sumk_args {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
};

static void fill_epilogue_inputs(
    float *sd, float *sh, float *res, float *sp,
    uint32_t dim, uint32_t top_k, uint32_t n_tok) {
    const size_t n_sd = (size_t)n_tok * top_k * dim;
    const size_t n_sh = (size_t)n_tok * dim;
    const size_t n_res = (size_t)n_tok * 4u * dim;
    const size_t n_sp = (size_t)n_tok * 24u;
    for (size_t i = 0; i < n_sd; ++i) sd[i] = rand_f32(-2.0f, 2.0f);
    for (size_t i = 0; i < n_sh; ++i) sh[i] = rand_f32(-2.0f, 2.0f);
    for (size_t i = 0; i < n_res; ++i) res[i] = rand_f32(-3.0f, 3.0f);
    for (size_t i = 0; i < n_sp; ++i) sp[i] = rand_f32(-1.0f, 1.0f);
    /* Inject BF16 tie-to-even boundary cases in token 0! */
    if (n_tok > 0 && dim >= 8u) {
        uint32_t tie0 = 0x3f808000u; /* 1.0 + 0.5 ulp(bf16) -> round to even */
        uint32_t tie1 = 0x3f818000u; /* odd -> round up */
        uint32_t neg0 = 0x80000000u; /* -0.0f */
        memcpy(&sd[0], &tie0, 4u);
        memcpy(&sd[1], &tie1, 4u);
        memcpy(&sd[2], &neg0, 4u);
        sh[0] = 0.0f; sh[1] = 0.0f; sh[2] = -0.0f;
    }
}

static void test_epilogue_0ulp_case(uint32_t dim, uint32_t top_k, uint32_t mb_moe_len,
                                    uint32_t pad_prefix, uint32_t pad_suffix) {
    const uint32_t chunk_size = 2048u;
    const uint32_t total_prompt_toks = pad_prefix + mb_moe_len + pad_suffix;
    const size_t res_tok_bytes = (size_t)4u * dim * sizeof(float);
    const size_t pre_tok_bytes = (size_t)24u * sizeof(float);

    myds_metal_buffer_t *b_sd = myds_metal_buffer_alloc((size_t)mb_moe_len * top_k * dim * sizeof(float));
    myds_metal_buffer_t *b_sh = myds_metal_buffer_alloc((size_t)mb_moe_len * dim * sizeof(float));
    myds_metal_buffer_t *b_af = myds_metal_buffer_alloc((size_t)mb_moe_len * res_tok_bytes);
    myds_metal_buffer_t *b_sp = myds_metal_buffer_alloc((size_t)mb_moe_len * pre_tok_bytes);
    myds_metal_buffer_t *b_moe_acc = myds_metal_buffer_alloc((size_t)mb_moe_len * dim * sizeof(float));
    const uint32_t lo_cap = mb_moe_len < chunk_size ? mb_moe_len : chunk_size;
    myds_metal_buffer_t *b_layer_out = myds_metal_buffer_alloc((size_t)lo_cap * res_tok_bytes);
    myds_metal_buffer_t *b_res_ref = myds_metal_buffer_alloc((size_t)total_prompt_toks * res_tok_bytes);
    myds_metal_buffer_t *b_res_got = myds_metal_buffer_alloc((size_t)total_prompt_toks * res_tok_bytes);
    myds_metal_buffer_t *b_pre_ref = myds_metal_buffer_alloc((size_t)total_prompt_toks * pre_tok_bytes);
    myds_metal_buffer_t *b_pre_got = myds_metal_buffer_alloc((size_t)total_prompt_toks * pre_tok_bytes);
    REQUIRE(b_sd && b_sh && b_af && b_sp && b_moe_acc && b_layer_out &&
            b_res_ref && b_res_got && b_pre_ref && b_pre_got);

    fill_epilogue_inputs(
        (float *)myds_metal_buffer_data(b_sd),
        (float *)myds_metal_buffer_data(b_sh),
        (float *)myds_metal_buffer_data(b_af),
        (float *)myds_metal_buffer_data(b_sp),
        dim, top_k, mb_moe_len);

    /* Fill both prompt_res/prompt_pre with canary pattern to verify zero OOB writes. */
    memset(myds_metal_buffer_data(b_res_ref), 0xA5, (size_t)total_prompt_toks * res_tok_bytes);
    memset(myds_metal_buffer_data(b_res_got), 0xA5, (size_t)total_prompt_toks * res_tok_bytes);
    memset(myds_metal_buffer_data(b_pre_ref), 0x5A, (size_t)total_prompt_toks * pre_tok_bytes);
    memset(myds_metal_buffer_data(b_pre_got), 0x5A, (size_t)total_prompt_toks * pre_tok_bytes);

    /* ---- Baseline path (sumk + 8x chunked moe_add_hc_expand4_batch + layer_out->prompt_res + prompt_pre copy) ---- */
    REQUIRE(myds_metal_begin_batch() == 1);
    {
        const char *sum_kname = (top_k == 6u) ? "kernel_dsv4_moe_sum6_f32" :
                                (top_k == 4u) ? "kernel_dsv4_moe_sum4_f32" :
                                                "kernel_dsv4_moe_sum3_f32";
        struct moe_sumk_args sa = {
            .width = dim, .tokens = mb_moe_len,
            .src_token_stride = (uint64_t)top_k * dim * sizeof(float),
            .dst_token_stride = (uint64_t)dim * sizeof(float),
        };
        const myds_metal_buffer_t *sbufs[2] = {b_sd, b_moe_acc};
        uint32_t gs[3] = {mb_moe_len, 1u, 1u}, ts[3] = {256u, 1u, 1u};
        REQUIRE(myds_metal_dispatch_kernel(sum_kname, &sa, sizeof(sa), sbufs, 2u, gs, ts, 0u) == 1);
        myds_metal_barrier();

        myds_metal_buffer_t *v_moe = myds_metal_buffer_create_view();
        myds_metal_buffer_t *v_sh  = myds_metal_buffer_create_view();
        myds_metal_buffer_t *v_af  = myds_metal_buffer_create_view();
        myds_metal_buffer_t *v_sp  = myds_metal_buffer_create_view();
        REQUIRE(v_moe && v_sh && v_af && v_sp);
        for (uint32_t e0 = 0; e0 < mb_moe_len; e0 += chunk_size) {
            uint32_t e_len = mb_moe_len - e0;
            if (e_len > chunk_size) e_len = chunk_size;
            const uint32_t e_start = pad_prefix + e0;
            REQUIRE(myds_metal_buffer_update_sub_view(v_moe, b_moe_acc, (size_t)e0 * dim * 4u, (size_t)e_len * dim * 4u) == 1);
            REQUIRE(myds_metal_buffer_update_sub_view(v_sh,  b_sh,      (size_t)e0 * dim * 4u, (size_t)e_len * dim * 4u) == 1);
            REQUIRE(myds_metal_buffer_update_sub_view(v_af,  b_af,      (size_t)e0 * res_tok_bytes, (size_t)e_len * res_tok_bytes) == 1);
            REQUIRE(myds_metal_buffer_update_sub_view(v_sp,  b_sp,      (size_t)e0 * pre_tok_bytes, (size_t)e_len * pre_tok_bytes) == 1);
            REQUIRE(myds_metal_v41_moe_add_hc_expand_batch(b_layer_out, NULL, v_moe, v_sh, v_af, v_sp, dim, e_len) == 1);
            myds_metal_barrier();
            REQUIRE(myds_metal_buffer_copy(b_res_ref, (size_t)e_start * res_tok_bytes, b_layer_out, 0u, (size_t)e_len * res_tok_bytes) == 1);
            REQUIRE(myds_metal_buffer_copy(b_pre_ref, (size_t)e_start * pre_tok_bytes, v_sp, 0u, (size_t)e_len * pre_tok_bytes) == 1);
            myds_metal_barrier();
        }
        myds_metal_buffer_free(v_moe); myds_metal_buffer_free(v_sh);
        myds_metal_buffer_free(v_af);  myds_metal_buffer_free(v_sp);
    }
    REQUIRE(myds_metal_commit_batch() == 1);

    /* ---- [P-E1] Fused path under MYDS_METAL_CONCURRENT=1, auto_barrier=0 ---- */
    REQUIRE(myds_metal_begin_batch() == 1);
    {
        myds_metal_buffer_t *v_res_dst = myds_metal_buffer_create_view();
        REQUIRE(v_res_dst != NULL);
        REQUIRE(myds_metal_buffer_update_sub_view(v_res_dst, b_res_got,
                (size_t)pad_prefix * res_tok_bytes, (size_t)mb_moe_len * res_tok_bytes) == 1);
        REQUIRE(myds_metal_v41_moe_sumk_add_hc_expand_batch(v_res_dst, b_sd, b_sh, b_af, b_sp,
                dim, top_k, mb_moe_len) == 1);
        REQUIRE(myds_metal_buffer_copy(b_pre_got, (size_t)pad_prefix * pre_tok_bytes,
                b_sp, 0u, (size_t)mb_moe_len * pre_tok_bytes) == 1);
        myds_metal_barrier();
        myds_metal_buffer_free(v_res_dst);
    }
    REQUIRE(myds_metal_commit_batch() == 1);

    const size_t total_res_bytes = (size_t)total_prompt_toks * res_tok_bytes;
    const size_t total_pre_bytes = (size_t)total_prompt_toks * pre_tok_bytes;
    if (memcmp(myds_metal_buffer_data(b_res_ref), myds_metal_buffer_data(b_res_got), total_res_bytes) != 0) {
        const float *a = (const float *)myds_metal_buffer_data(b_res_ref);
        const float *b = (const float *)myds_metal_buffer_data(b_res_got);
        size_t diffs = 0;
        for (size_t i = 0; i < total_res_bytes / 4u; ++i) {
            uint32_t ua, ub; memcpy(&ua, a + i, 4); memcpy(&ub, b + i, 4);
            if (ua != ub && ++diffs <= 5u) {
                fprintf(stderr, "  [DIFF] k=%u len=%u idx=%zu ref=%g (0x%08x) got=%g (0x%08x)\n",
                        top_k, mb_moe_len, i, a[i], ua, b[i], ub);
            }
        }
        fprintf(stderr, "FAIL epilogue prompt_res k=%u len=%u (%zu diffs)\n", top_k, mb_moe_len, diffs);
        exit(1);
    }
    REQUIRE(memcmp(myds_metal_buffer_data(b_pre_ref), myds_metal_buffer_data(b_pre_got), total_pre_bytes) == 0);
    printf("[PASS] epilogue_fuse k=%u mb_moe_len=%5u (offset=%u, concurrent=1 auto_barrier=0): strict 0-ULP (memcmp==0)\n",
           top_k, mb_moe_len, pad_prefix);

    myds_metal_buffer_free(b_sd); myds_metal_buffer_free(b_sh); myds_metal_buffer_free(b_af);
    myds_metal_buffer_free(b_sp); myds_metal_buffer_free(b_moe_acc); myds_metal_buffer_free(b_layer_out);
    myds_metal_buffer_free(b_res_ref); myds_metal_buffer_free(b_res_got);
    myds_metal_buffer_free(b_pre_ref); myds_metal_buffer_free(b_pre_got);
}

/* End-to-end MoE + Fused Epilogue parity test covering BOTH MM (n_tok=312, k in {3,4,6})
 * and GEMV (n_tok=1, k=6) paths under MYDS_METAL_CONCURRENT=1, auto_barrier=0! */
static void test_end_to_end_moe_epilogue(void) {
    const uint32_t dim = 5120u, edim = 2304u, nexp = 32u;
    const size_t gate_exp_bytes = (size_t)edim * (dim / 256u) * 66u;
    const size_t down_exp_bytes = (size_t)dim * (edim / 256u) * 84u;
    const size_t res_tok_bytes = (size_t)4u * dim * sizeof(float);
    const size_t pre_tok_bytes = (size_t)24u * sizeof(float);

    myds_metal_buffer_t *b_gate = myds_metal_buffer_alloc(nexp * gate_exp_bytes);
    myds_metal_buffer_t *b_up   = myds_metal_buffer_alloc(nexp * gate_exp_bytes);
    myds_metal_buffer_t *b_down = myds_metal_buffer_alloc(nexp * down_exp_bytes);
    REQUIRE(b_gate && b_up && b_down);
    fill_iq2_xxs(myds_metal_buffer_data(b_gate), (size_t)nexp * edim * (dim / 256u));
    fill_iq2_xxs(myds_metal_buffer_data(b_up),   (size_t)nexp * edim * (dim / 256u));
    fill_q2_k(myds_metal_buffer_data(b_down),    (size_t)nexp * dim * (edim / 256u));

    myds_metal_moe_epilogue_fuse_counters(NULL, true);

    struct { uint32_t k; uint32_t n_tok; } cases[] = {
        {6u, 1u},   /* GEMV path -> fused_mv_encoded */
        {6u, 312u}, /* MM path -> fused_mm_encoded */
        {4u, 312u},
        {3u, 312u},
    };
    uint64_t exp_mm = 0, exp_mv = 0, exp_toks = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(*cases); ++ci) {
        const uint32_t k = cases[ci].k;
        const uint32_t n_tok = cases[ci].n_tok;
        const uint32_t pad = 3u;
        const uint32_t tot = pad + n_tok + 2u;

        myds_metal_buffer_t *b_x   = myds_metal_buffer_alloc((size_t)n_tok * dim * 4u);
        myds_metal_buffer_t *b_ids = myds_metal_buffer_alloc((size_t)n_tok * k * 4u);
        myds_metal_buffer_t *b_w   = myds_metal_buffer_alloc((size_t)n_tok * k * 4u);
        myds_metal_buffer_t *b_sh  = myds_metal_buffer_alloc((size_t)n_tok * dim * 4u);
        myds_metal_buffer_t *b_af  = myds_metal_buffer_alloc((size_t)n_tok * res_tok_bytes);
        myds_metal_buffer_t *b_sp  = myds_metal_buffer_alloc((size_t)n_tok * pre_tok_bytes);
        myds_metal_buffer_t *b_sg  = myds_metal_buffer_alloc((size_t)n_tok * k * edim * 4u);
        myds_metal_buffer_t *b_su  = myds_metal_buffer_alloc((size_t)n_tok * k * edim * 4u);
        myds_metal_buffer_t *b_sm  = myds_metal_buffer_alloc((size_t)n_tok * k * edim * 4u);
        myds_metal_buffer_t *b_sd  = myds_metal_buffer_alloc((size_t)n_tok * k * dim * 4u);
        myds_metal_buffer_t *b_moe = myds_metal_buffer_alloc((size_t)n_tok * dim * 4u);
        myds_metal_buffer_t *b_lo  = myds_metal_buffer_alloc((size_t)n_tok * res_tok_bytes);
        myds_metal_buffer_t *b_ref = myds_metal_buffer_alloc((size_t)tot * res_tok_bytes);
        myds_metal_buffer_t *b_got = myds_metal_buffer_alloc((size_t)tot * res_tok_bytes);
        REQUIRE(b_x && b_ids && b_w && b_sh && b_af && b_sp && b_sg && b_su &&
                b_sm && b_sd && b_moe && b_lo && b_ref && b_got);

        float *xp = myds_metal_buffer_data(b_x);
        for (size_t i = 0; i < (size_t)n_tok * dim; ++i) xp[i] = rand_f32(-1.5f, 1.5f);
        int32_t *ip = myds_metal_buffer_data(b_ids);
        float *wp = myds_metal_buffer_data(b_w);
        for (uint32_t t = 0; t < n_tok; ++t) {
            for (uint32_t s = 0; s < k; ++s) {
                ip[t * k + s] = (int32_t)((t * 5u + s * 3u) % nexp);
                wp[t * k + s] = rand_f32(0.05f, 0.5f);
            }
        }
        float *shp = myds_metal_buffer_data(b_sh);
        for (size_t i = 0; i < (size_t)n_tok * dim; ++i) shp[i] = rand_f32(-1.0f, 1.0f);
        float *afp = myds_metal_buffer_data(b_af);
        for (size_t i = 0; i < (size_t)n_tok * 4u * dim; ++i) afp[i] = rand_f32(-2.0f, 2.0f);
        float *spp = myds_metal_buffer_data(b_sp);
        for (size_t i = 0; i < (size_t)n_tok * 24u; ++i) spp[i] = rand_f32(-0.5f, 0.5f);

        memset(myds_metal_buffer_data(b_ref), 0xCC, (size_t)tot * res_tok_bytes);
        memset(myds_metal_buffer_data(b_got), 0xCC, (size_t)tot * res_tok_bytes);

        /* Baseline inside concurrent batch with auto_barrier=0 */
        REQUIRE(myds_metal_begin_batch() == 1);
        REQUIRE(myds_metal_routed_moe_batch(b_moe, b_x, b_gate, b_up, b_down, b_ids, b_w,
                b_sg, b_su, b_sm, b_sd, dim, edim, nexp, k, n_tok,
                GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 7.0f) == 1);
        myds_metal_barrier();
        REQUIRE(myds_metal_v41_moe_add_hc_expand_batch(b_lo, NULL, b_moe, b_sh, b_af, b_sp, dim, n_tok) == 1);
        myds_metal_barrier();
        REQUIRE(myds_metal_buffer_copy(b_ref, (size_t)pad * res_tok_bytes, b_lo, 0u, (size_t)n_tok * res_tok_bytes) == 1);
        REQUIRE(myds_metal_commit_batch() == 1);

        /* Fused [P-E1] inside concurrent batch with auto_barrier=0 */
        myds_metal_buffer_t *v_dst = myds_metal_buffer_create_view();
        REQUIRE(v_dst && myds_metal_buffer_update_sub_view(v_dst, b_got,
                (size_t)pad * res_tok_bytes, (size_t)n_tok * res_tok_bytes) == 1);
        /* Poison b_moe to prove MM path never reads/writes b_moe! */
        memset(myds_metal_buffer_data(b_moe), 0xFF, (size_t)n_tok * dim * 4u);
        REQUIRE(myds_metal_begin_batch() == 1);
        REQUIRE(myds_metal_routed_moe_batch_epilogue_fused(v_dst, b_moe, b_x, b_gate, b_up, b_down,
                b_ids, b_w, b_sh, b_af, b_sp, b_sg, b_su, b_sm, b_sd,
                dim, edim, nexp, k, n_tok,
                GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 7.0f) == 1);
        REQUIRE(myds_metal_commit_batch() == 1);
        myds_metal_buffer_free(v_dst);

        REQUIRE(memcmp(myds_metal_buffer_data(b_ref), myds_metal_buffer_data(b_got), (size_t)tot * res_tok_bytes) == 0);
        if (n_tok >= 32u) {
            /* Prove b_moe was untouched (zero-write!) on MM path */
            const uint8_t *mp = (const uint8_t *)myds_metal_buffer_data(b_moe);
            REQUIRE(mp[0] == 0xFFu && mp[(size_t)n_tok * dim * 4u - 1u] == 0xFFu);
            exp_mm++;
        } else {
            exp_mv++;
        }
        exp_toks += n_tok;

        printf("[PASS] e2e_moe_epilogue_fused k=%u n_tok=%3u (%s): strict 0-ULP (memcmp==0)\n",
               k, n_tok, n_tok >= 32u ? "MM zero-moe_out" : "GEMV k=1");

        myds_metal_buffer_free(b_x); myds_metal_buffer_free(b_ids); myds_metal_buffer_free(b_w);
        myds_metal_buffer_free(b_sh); myds_metal_buffer_free(b_af); myds_metal_buffer_free(b_sp);
        myds_metal_buffer_free(b_sg); myds_metal_buffer_free(b_su); myds_metal_buffer_free(b_sm);
        myds_metal_buffer_free(b_sd); myds_metal_buffer_free(b_moe); myds_metal_buffer_free(b_lo);
        myds_metal_buffer_free(b_ref); myds_metal_buffer_free(b_got);
    }

    myds_metal_moe_epilogue_fuse_record_fallback();
    myds_metal_moe_epilogue_fuse_counters_t cnt;
    myds_metal_moe_epilogue_fuse_counters(&cnt, false);
    REQUIRE(cnt.calls == exp_mm + exp_mv);
    REQUIRE(cnt.fused_mm_encoded == exp_mm);
    REQUIRE(cnt.fused_mv_encoded == exp_mv);
    REQUIRE(cnt.fallbacks == 1u);
    REQUIRE(cnt.tokens_fused == exp_toks);
    printf("[PASS] /debug/state counters verified (calls=%llu mm=%llu mv=%llu fb=%llu toks=%llu)\n",
           (unsigned long long)cnt.calls, (unsigned long long)cnt.fused_mm_encoded,
           (unsigned long long)cnt.fused_mv_encoded, (unsigned long long)cnt.fallbacks,
           (unsigned long long)cnt.tokens_fused);

    myds_metal_buffer_free(b_gate); myds_metal_buffer_free(b_up); myds_metal_buffer_free(b_down);
}

static void run_m4_bench(const char *metallib_path) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path)] error:&err];
        REQUIRE(dev && lib);
        id<MTLCommandQueue> q = [dev newCommandQueue];

        id<MTLComputePipelineState> p_sum3 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_dsv4_moe_sum3_f32"] error:&err];
        id<MTLComputePipelineState> p_sum4 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_dsv4_moe_sum4_f32"] error:&err];
        id<MTLComputePipelineState> p_sum6 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_dsv4_moe_sum6_f32"] error:&err];
        id<MTLComputePipelineState> p_exp  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_dsv4_moe_add_hc_expand4_batch"] error:&err];
        id<MTLComputePipelineState> p_cp4  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_compute_copy_u32x4"] error:&err];
        id<MTLComputePipelineState> p_f3   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_dsv4_moe_sum3_add_hc_expand4_f4"] error:&err];
        id<MTLComputePipelineState> p_f4   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_dsv4_moe_sum4_add_hc_expand4_f4"] error:&err];
        id<MTLComputePipelineState> p_f6   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_dsv4_moe_sum6_add_hc_expand4_f4"] error:&err];
        REQUIRE(p_sum3 && p_sum4 && p_sum6 && p_exp && p_cp4 && p_f3 && p_f4 && p_f6);

        printf("\n=== [P-E1] Interleaved GPU-Timed M4 Epilogue Microbenchmark (dim=5120, concurrent=1) ===\n");
        const uint32_t dim = 5120u, chunk = 2048u;
        const size_t res_b = 4u * dim * 4u, pre_b = 24u * 4u;
        const uint32_t ks[] = {6u, 4u, 3u};
        const uint32_t lens[] = {312u, 2048u, 2170u, 16384u};

        for (size_t li = 0; li < sizeof(lens) / sizeof(*lens); ++li) {
            const uint32_t nt = lens[li];
            const uint32_t lo_cap = nt < chunk ? nt : chunk;
            id<MTLBuffer> b_sd  = [dev newBufferWithLength:(size_t)nt * 6u * dim * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sh  = [dev newBufferWithLength:(size_t)nt * dim * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_af  = [dev newBufferWithLength:(size_t)nt * res_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sp  = [dev newBufferWithLength:(size_t)nt * pre_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_moe = [dev newBufferWithLength:(size_t)nt * dim * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_lo  = [dev newBufferWithLength:(size_t)lo_cap * res_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_res = [dev newBufferWithLength:(size_t)nt * res_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pre = [dev newBufferWithLength:(size_t)nt * pre_b options:MTLResourceStorageModeShared];
            REQUIRE(b_sd && b_sh && b_af && b_sp && b_moe && b_lo && b_res && b_pre);
            /* Populate with non-zero incompressible values so Apple GPU lossless DRAM
             * compression does not mask baseline's redundant moe_out + layer_out traffic. */
            {
                uint32_t *p = (uint32_t *)b_sd.contents;
                for (size_t i = 0; i < (size_t)nt * 6u * dim; ++i) p[i] = 0x3f000000u ^ (xorshift32() & 0x007fffffu);
                p = (uint32_t *)b_sh.contents;
                for (size_t i = 0; i < (size_t)nt * dim; ++i) p[i] = 0x3f000000u ^ (xorshift32() & 0x007fffffu);
                p = (uint32_t *)b_af.contents;
                for (size_t i = 0; i < (size_t)nt * 4u * dim; ++i) p[i] = 0x3f000000u ^ (xorshift32() & 0x007fffffu);
                float *sp = (float *)b_sp.contents;
                for (size_t i = 0; i < (size_t)nt * 24u; ++i) sp[i] = rand_f32(-0.5f, 0.5f);
            }

            for (size_t ki = 0; ki < 3u; ++ki) {
                const uint32_t k = ks[ki];
                id<MTLComputePipelineState> p_sum = (k == 6u) ? p_sum6 : (k == 4u ? p_sum4 : p_sum3);
                id<MTLComputePipelineState> p_fus = (k == 6u) ? p_f6   : (k == 4u ? p_f4   : p_f3);

                const int ROUNDS = (nt >= 16384u) ? 5 : 9;
                const int REPS   = (nt <= 312u) ? 5 : 1;
                double ts[2][16];

                for (int r = 0; r < ROUNDS; ++r) {
                    for (int o = 0; o < 2; ++o) {
                        int arm = (r + o) & 1;
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                        for (int rep = 0; rep < REPS; ++rep) {
                            if (arm == 0) {
                                /* Baseline: sumk + chunked expand4 + chunked copy layer_out->res + copy pre */
                                struct moe_sumk_args sa = { dim, nt, (uint64_t)k*dim*4u, (uint64_t)dim*4u };
                                [enc setComputePipelineState:p_sum];
                                [enc setBytes:&sa length:sizeof(sa) atIndex:0];
                                [enc setBuffer:b_sd offset:0 atIndex:1];
                                [enc setBuffer:b_moe offset:0 atIndex:2];
                                [enc dispatchThreadgroups:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                                id<MTLResource> rm = b_moe;
                                [enc memoryBarrierWithResources:&rm count:1];

                                for (uint32_t e0 = 0; e0 < nt; e0 += chunk) {
                                    uint32_t el = nt - e0; if (el > chunk) el = chunk;
                                    struct {
                                        int64_t n_embd, n_hc, n_tokens;
                                        uint64_t nb_block0, nb_block1, nb_add0, nb_add1;
                                        uint64_t nb_res0, nb_res1, nb_res2;
                                        uint64_t nb_post0, nb_post1, nb_comb0, nb_comb1, nb_comb2;
                                        uint64_t nb0, nb1, nb2;
                                        uint32_t has_add, round_bf16;
                                    } ea = { dim, 4, el, 4, 4ull*dim, 4, 4ull*dim, 4, 4ull*dim, 16ull*dim,
                                             4, 96, 4, 16, 96, 4, 4ull*dim, 16ull*dim, 0, 1 };
                                    [enc setComputePipelineState:p_exp];
                                    [enc setBytes:&ea length:sizeof(ea) atIndex:0];
                                    [enc setBuffer:b_moe offset:(NSUInteger)e0*dim*4u atIndex:1];
                                    [enc setBuffer:b_sh  offset:(NSUInteger)e0*dim*4u atIndex:2];
                                    [enc setBuffer:b_af  offset:(NSUInteger)e0*res_b atIndex:3];
                                    [enc setBuffer:b_sp  offset:(NSUInteger)e0*pre_b+16u atIndex:4];
                                    [enc setBuffer:b_sp  offset:(NSUInteger)e0*pre_b+32u atIndex:5];
                                    [enc setBuffer:b_lo  offset:0 atIndex:6];
                                    [enc setBuffer:b_lo  offset:0 atIndex:7];
                                    [enc dispatchThreadgroups:MTLSizeMake((dim+255)/256, el, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                                    id<MTLResource> rlo = b_lo;
                                    [enc memoryBarrierWithResources:&rlo count:1];

                                    struct { uint64_t so, doff, cnt; } ca1 = { 0, (uint64_t)e0*res_b, ((uint64_t)el*res_b)/16u };
                                    [enc setComputePipelineState:p_cp4];
                                    [enc setBytes:&ca1 length:sizeof(ca1) atIndex:0];
                                    [enc setBuffer:b_lo offset:0 atIndex:1];
                                    [enc setBuffer:b_res offset:0 atIndex:2];
                                    [enc dispatchThreadgroups:MTLSizeMake((ca1.cnt+255)/256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

                                    struct { uint64_t so, doff, cnt; } ca2 = { (uint64_t)e0*pre_b, (uint64_t)e0*pre_b, ((uint64_t)el*pre_b)/16u };
                                    [enc setBytes:&ca2 length:sizeof(ca2) atIndex:0];
                                    [enc setBuffer:b_sp offset:0 atIndex:1];
                                    [enc setBuffer:b_pre offset:0 atIndex:2];
                                    [enc dispatchThreadgroups:MTLSizeMake((ca2.cnt+255)/256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                                    [enc memoryBarrierWithResources:&rlo count:1];
                                }
                            } else {
                                /* [P-E1] Fused: 1x sumk_add_hc_expand4_f4 direct to prompt_res + 1x prompt_pre copy */
                                struct { uint32_t dim, n_tokens, top_k, round_bf16; } fa = { dim, nt, k, 1u };
                                NSUInteger gx = (dim / 4u + 255u) / 256u;
                                [enc setComputePipelineState:p_fus];
                                [enc setBytes:&fa length:sizeof(fa) atIndex:0];
                                [enc setBuffer:b_sd offset:0 atIndex:1];
                                [enc setBuffer:b_sh offset:0 atIndex:2];
                                [enc setBuffer:b_af offset:0 atIndex:3];
                                [enc setBuffer:b_sp offset:0 atIndex:4];
                                [enc setBuffer:b_res offset:0 atIndex:5];
                                [enc dispatchThreadgroups:MTLSizeMake(gx, nt, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                                struct { uint64_t so, doff, cnt; } ca2 = { 0, 0, ((uint64_t)nt*pre_b)/16u };
                                [enc setComputePipelineState:p_cp4];
                                [enc setBytes:&ca2 length:sizeof(ca2) atIndex:0];
                                [enc setBuffer:b_sp offset:0 atIndex:1];
                                [enc setBuffer:b_pre offset:0 atIndex:2];
                                [enc dispatchThreadgroups:MTLSizeMake((ca2.cnt+255)/256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                                id<MTLResource> rout[2] = {b_res, b_pre};
                                [enc memoryBarrierWithResources:rout count:2];
                            }
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        ts[arm][r] = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3 / (double)REPS;
                    }
                }
                qsort(ts[0], (size_t)ROUNDS, sizeof(double), cmp_dbl);
                qsort(ts[1], (size_t)ROUNDS, sizeof(double), cmp_dbl);
                const double base_ms = ts[0][ROUNDS / 2];
                const double fuse_ms = ts[1][ROUNDS / 2];
                const double fused_bytes_gb = (double)nt * ((double)(k + 1u + 4u + 4u) * dim * 4.0 + 192.0) / 1e9;
                printf("  k=%u  mb_moe_len=%5u | base=%7.3f ms | fused=%7.3f ms | speedup=%5.2fx (%+.1f%%) | effBW=%5.1f GB/s\n",
                       k, nt, base_ms, fuse_ms, base_ms / fuse_ms,
                       100.0 * (base_ms - fuse_ms) / base_ms,
                       fused_bytes_gb / (fuse_ms * 1e-3));
            }
        }
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

    /* Run all 0-ULP verification tests under MYDS_METAL_CONCURRENT=1, auto_barrier=0! */
    setenv("MYDS_METAL_CONCURRENT", "1", 1);
    setenv("MYDS_MOE_EPILOGUE_FUSE", "1", 1);
    REQUIRE(myds_metal_init(metallib) == 1);
    myds_metal_set_auto_barrier(0);
    REQUIRE(myds_metal_get_auto_barrier() == 0);

    const uint32_t ks[] = {3u, 4u, 6u};
    const uint32_t lens[] = {1u, 312u, 2048u, 2170u, 16384u};
    for (size_t li = 0; li < sizeof(lens) / sizeof(*lens); ++li) {
        const uint32_t nt = lens[li];
        const uint32_t pad_pre = (nt >= 16384u) ? 4u : 7u;
        const uint32_t pad_suf = (nt >= 16384u) ? 4u : 5u;
        for (size_t ki = 0; ki < sizeof(ks) / sizeof(*ks); ++ki) {
            test_epilogue_0ulp_case(5120u, ks[ki], nt, pad_pre, pad_suf);
        }
    }

    test_end_to_end_moe_epilogue();
    myds_metal_cleanup();

    if (do_bench) {
        run_m4_bench(metallib);
    }

    printf("\n=== [P-E1] ALL TESTS PASSED ===\n");
    return 0;
}
