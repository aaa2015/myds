/* Verification & GPU microbenchmark for Wave 2 Prefill Attention [CAND-A3]:
 *   MYDS_PREFILL_ATTN_L01_MMA (absent/0=off, 1=on, else fail-closed -1; Track 2 num-equiv):
 *     1. Fail-closed knob contract & counter reporting.
 *     2. PSO residency probe for kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512
 *        (Apple8- and Apple9-safe: staticSmem == 0, maxThr >= 128 for NSG=4).
 *     3. Numerical parity (max_abs_err < 1e-4, l2_rel < 1e-4, min_cos > 0.999999) for L0/L1
 *        window-only direct-slab MMA vs baseline scalar vec across all 9 shapes:
 *        p0 in {127, 256, 2048} x c_len in {32, 122, 2048}; verifies mask template, mask fill,
 *        and scalar vec are all skipped.
 *     4. Fallback safety & carry roll: c_len < 32 and p0 < 127 stay bit-exact 0-ULP on vec,
 *        and chunk0 -> carry roll -> chunk1 L01_MMA matches vec within tight MMA tolerance.
 *     5. --bench GPU microbenchmark comparing L0/L1 vec vs L01_MMA.
 */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

struct guarded_buf {
    myds_metal_buffer_t *owner;
    myds_metal_buffer_t *view;
    size_t bytes;
};

static struct guarded_buf guarded_alloc(size_t bytes) {
    struct guarded_buf g = { .bytes = bytes };
    g.owner = myds_metal_buffer_alloc(bytes + 64u);
    REQUIRE(g.owner != NULL);
    memset(myds_metal_buffer_data(g.owner), 0xa5, bytes + 64u);
    g.view = myds_metal_buffer_sub_view(g.owner, 32u, bytes);
    REQUIRE(g.view != NULL);
    return g;
}

static void guarded_check(struct guarded_buf g) {
    const uint8_t *p = (const uint8_t *)myds_metal_buffer_data(g.owner);
    for (size_t i = 0; i < 32u; ++i) {
        REQUIRE(p[i] == 0xa5);
        REQUIRE(p[32u + g.bytes + i] == 0xa5);
    }
}

static void guarded_free(struct guarded_buf g) {
    guarded_check(g);
    myds_metal_buffer_free(g.view);
    myds_metal_buffer_free(g.owner);
}

static float sample_f32(uint32_t i, uint32_t seed) {
    uint32_t x = (i + 1u) * 747796405u + seed * 2891336453u;
    x = ((x >> ((x >> 28) + 4u)) ^ x) * 277803737u;
    x = (x >> 22) ^ x;
    return ((int32_t)(x % 8191u) - 4095) / 8192.0f;
}

static uint64_t hist_count(const char *name) {
    for (int i = 0; i < g_hist_used; ++i) {
        if (strcmp(g_hist[i].name, name) == 0) return g_hist[i].n;
    }
    return 0;
}

static void hist_reset(void) {
    g_hist_used = 0;
}

static void fill_f32_buf(myds_metal_buffer_t *b, size_t n, uint32_t seed) {
    float *p = (float *)myds_metal_buffer_data(b);
    for (size_t i = 0; i < n; ++i) p[i] = sample_f32((uint32_t)i, seed);
}

static void reset_env_defaults(void) {
    unsetenv("MYDS_PREFILL_ATTN_FAST");
    unsetenv("MYDS_PREFILL_ATTN_L01_MMA");
    unsetenv("MYDS_ATTN_HEADS_MODE");
    setenv("MYDS_PREFILL_ATTN", "auto", 1);
}

/* 1. Fail-closed knob validation contract for MYDS_PREFILL_ATTN_L01_MMA. */
static void test_knob_contracts(void) {
    reset_env_defaults();
    REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == 0);

    setenv("MYDS_PREFILL_ATTN_L01_MMA", "0", 1);
    REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == 0);
    setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1);
    REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == 1);

    const char *bad_bool[] = {"", "2", "01", "true", "1 ", "-1"};
    for (size_t i = 0; i < sizeof(bad_bool)/sizeof(bad_bool[0]); ++i) {
        setenv("MYDS_PREFILL_ATTN_L01_MMA", bad_bool[i], 1);
        REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == -1);
        REQUIRE(myds_v41_gpu_scratch_create(4096u) == NULL);
    }
    reset_env_defaults();
    printf("PASS knob_contracts (MYDS_PREFILL_ATTN_L01_MMA=0/1, default 0 fail-closed)\n");
}

/* 2. PSO residency check for retained maskless-slab kernel (Apple8 & Apple9 safe). */
static void test_pso_residency(void) {
    printf("=== PSO Residency Probe (%s) ===\n", g_device.name.UTF8String);
    const char *name = "kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512";
    const int32_t nsg = 4;
    const size_t dyn_smem = 10272u;

    NSError *err = nil;
    MTLFunctionConstantValues *vals = [MTLFunctionConstantValues new];
    bool b_mask = false, b_sinks = true, b_bias = false, b_scap = false, b_kvpad = false, b_bc = false;
    int32_t ns10 = 512, ns20 = 512;
    [vals setConstantValue:&b_mask  type:MTLDataTypeBool atIndex:300];
    [vals setConstantValue:&b_sinks type:MTLDataTypeBool atIndex:301];
    [vals setConstantValue:&b_bias  type:MTLDataTypeBool atIndex:302];
    [vals setConstantValue:&b_scap  type:MTLDataTypeBool atIndex:303];
    [vals setConstantValue:&b_kvpad type:MTLDataTypeBool atIndex:304];
    [vals setConstantValue:&b_bc    type:MTLDataTypeBool atIndex:310];
    [vals setConstantValue:&ns10    type:MTLDataTypeInt  atIndex:320];
    [vals setConstantValue:&ns20    type:MTLDataTypeInt  atIndex:321];
    [vals setConstantValue:&nsg     type:MTLDataTypeInt  atIndex:322];
    id<MTLFunction> fn = [g_library newFunctionWithName:@(name) constantValues:vals error:&err];
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    REQUIRE(pso.threadExecutionWidth == 32u);
    REQUIRE(pso.staticThreadgroupMemoryLength == 0u);
    const unsigned tpg = (unsigned)(nsg * 32);
    REQUIRE(pso.maxTotalThreadsPerThreadgroup >= tpg);
    const unsigned tgs_by_thr = (unsigned)(pso.maxTotalThreadsPerThreadgroup / tpg);
    const unsigned tgs_by_smem = (unsigned)(32768u / dyn_smem);
    const unsigned tgs_core = tgs_by_thr < tgs_by_smem ? tgs_by_thr : tgs_by_smem;
    printf("PSO  %-64s nsg=%2d maxThr=%4lu staticSmem=%lu dynSmem=%5zu TGs/core=%u\n",
           name, nsg, (unsigned long)pso.maxTotalThreadsPerThreadgroup,
           (unsigned long)pso.staticThreadgroupMemoryLength, dyn_smem, tgs_core);
    printf("PASS pso_residency (staticSmem=0, maxThr=%lu >= %u)\n",
           (unsigned long)pso.maxTotalThreadsPerThreadgroup, tpg);
}

static void verify_exact_0ulp(const void *ref, const void *cand, size_t n_floats, const char *label) {
    if (memcmp(ref, cand, n_floats * 4u) == 0) return;
    const uint32_t *a = (const uint32_t *)ref;
    const uint32_t *b = (const uint32_t *)cand;
    size_t diffs = 0, first = SIZE_MAX;
    for (size_t i = 0; i < n_floats; ++i) {
        if (a[i] != b[i]) {
            if (first == SIZE_MAX) first = i;
            ++diffs;
        }
    }
    fprintf(stderr, "FAIL 0-ULP %s: diffs=%zu/%zu first=%zu (dim=%zu head=%zu row=%zu) ref=0x%08x cand=0x%08x\n",
            label, diffs, n_floats, first, first % 512u, (first / 512u) % 64u, first / (64u * 512u),
            a[first], b[first]);
    exit(1);
}

struct parity_stats {
    double max_abs_err;
    double l2_rel;
    double min_cos;
};

static struct parity_stats compute_parity(const float *ref, const float *cand, uint32_t n_rows, uint32_t dim) {
    double max_abs = 0.0, sum_diff2 = 0.0, sum_ref2 = 0.0, min_cos = 1.0;
    for (uint32_t r = 0; r < n_rows; ++r) {
        const float *a = ref + (size_t)r * dim;
        const float *b = cand + (size_t)r * dim;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            const double va = (double)a[d], vb = (double)b[d];
            const double diff = fabs(va - vb);
            if (diff > max_abs) max_abs = diff;
            sum_diff2 += diff * diff;
            sum_ref2 += va * va;
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
        }
        if (na > 1e-12 && nb > 1e-12) {
            double c = dot / (sqrt(na) * sqrt(nb));
            if (c < min_cos) min_cos = c;
        }
    }
    struct parity_stats st = {
        .max_abs_err = max_abs,
        .l2_rel = (sum_ref2 > 0.0) ? sqrt(sum_diff2 / sum_ref2) : 0.0,
        .min_cos = min_cos,
    };
    return st;
}

/* 3. [CAND-A3] L0/L1 Window-Only Direct-Slab MMA parity & dispatch checks. */
static void test_cand_a3_l01_mma(uint32_t p0, uint32_t c_len) {
    const size_t q_floats = (size_t)c_len * 64u * 512u;
    struct guarded_buf q_f32   = guarded_alloc(q_floats * 4u);
    struct guarded_buf sinks   = guarded_alloc(64u * 4u);
    struct guarded_buf win     = guarded_alloc(128u * 512u * 4u);
    struct guarded_buf ckv     = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf out_vec = guarded_alloc(q_floats * 4u);
    struct guarded_buf out_mma = guarded_alloc(q_floats * 4u);

    const uint32_t seed = 7000u + p0 * 5u + c_len * 19u;
    fill_f32_buf(q_f32.view, q_floats, seed + 1u);
    fill_f32_buf(sinks.view, 64u, seed + 2u);
    fill_f32_buf(win.view, 128u * 512u, seed + 3u);
    fill_f32_buf(ckv.view, (size_t)c_len * 512u, seed + 4u);

    reset_env_defaults();
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(4096u);
    REQUIRE(s != NULL);

    /* Baseline: MYDS_PREFILL_ATTN_L01_MMA=0 -> scalar vec */
    setenv("MYDS_PREFILL_ATTN_L01_MMA", "0", 1);
    myds_v41_gpu_prefill_attn_l01_mma_counters(NULL, true);
    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_rope_batch_round(q_f32.view, 512u, 64u, p0, c_len, false, false));
    REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out_vec.view, sinks.view, q_f32.view, ckv.view,
            NULL, NULL, NULL, c_len, 0u, 512u, p0, 0u, 0u, s));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(hist_count("kernel_flash_attn_ext_vec_f16_dk512_dv512") == 1u);
    myds_v41_gpu_prefill_attn_l01_mma_counters_t lc;
    myds_v41_gpu_prefill_attn_l01_mma_counters(&lc, false);
    REQUIRE(lc.encoded == 0u);

    /* Candidate: MYDS_PREFILL_ATTN_L01_MMA=1 -> maskless-slab MMA */
    setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1);
    myds_v41_gpu_prefill_attn_l01_mma_counters(NULL, true);
    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out_mma.view, sinks.view, q_f32.view, ckv.view,
            NULL, NULL, NULL, c_len, 0u, 512u, p0, 0u, 0u, s));
    REQUIRE(myds_metal_commit_batch());

    /* Verify mask template, mask fill, and scalar vec are ALL skipped */
    REQUIRE(hist_count("kernel_myds_mask_chunk_window") == 0u);
    REQUIRE(hist_count("kernel_myds_mask_chunk_fill") == 0u);
    REQUIRE(hist_count("kernel_flash_attn_ext_vec_f16_dk512_dv512") == 0u);
    REQUIRE(hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512") == 1u);
    myds_v41_gpu_prefill_attn_l01_mma_counters(&lc, false);
    REQUIRE(lc.encoded == 1u);

    struct parity_stats st = compute_parity((const float *)myds_metal_buffer_data(out_vec.view),
                                            (const float *)myds_metal_buffer_data(out_mma.view),
                                            c_len * 64u, 512u);
    REQUIRE(st.max_abs_err < 1e-4);
    REQUIRE(st.l2_rel < 1e-4);
    REQUIRE(st.min_cos > 0.999999);

    printf("PASS [CAND-A3] L0/L1 parity p0=%4u c_len=%4u vs vec: max_abs=%.3e l2_rel=%.3e min_cos=%.9f\n",
           p0, c_len, st.max_abs_err, st.l2_rel, st.min_cos);

    myds_v41_gpu_scratch_free(s);
    guarded_free(out_mma); guarded_free(out_vec);
    guarded_free(ckv); guarded_free(win); guarded_free(sinks); guarded_free(q_f32);
}

/* 4. Fallback safety (c_len < 32, p0 < 127) and chunk0 -> carry roll -> chunk1 L01_MMA. */
static void test_cand_a3_fallback_and_carry_roll(void) {
    /* 4a: c_len = 31 (< 32) at p0 = 2048 must stay on scalar vec (0-ULP vs baseline). */
    {
        const uint32_t p0 = 2048u, c_len = 31u;
        const size_t q_floats = (size_t)c_len * 64u * 512u;
        struct guarded_buf q     = guarded_alloc(q_floats * 4u);
        struct guarded_buf sinks = guarded_alloc(64u * 4u);
        struct guarded_buf win   = guarded_alloc(128u * 512u * 4u);
        struct guarded_buf ckv   = guarded_alloc((size_t)c_len * 512u * 4u);
        struct guarded_buf o_ref = guarded_alloc(q_floats * 4u);
        struct guarded_buf o_new = guarded_alloc(q_floats * 4u);
        fill_f32_buf(q.view, q_floats, 901u);
        fill_f32_buf(sinks.view, 64u, 902u);
        fill_f32_buf(win.view, 128u * 512u, 903u);
        fill_f32_buf(ckv.view, (size_t)c_len * 512u, 904u);

        reset_env_defaults();
        myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(4096u);
        REQUIRE(s != NULL);

        setenv("MYDS_PREFILL_ATTN_L01_MMA", "0", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(o_ref.view, sinks.view, q.view, ckv.view,
                NULL, NULL, NULL, c_len, 0u, 512u, p0, 0u, 0u, s));
        REQUIRE(myds_metal_commit_batch());

        setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1);
        myds_v41_gpu_prefill_attn_l01_mma_counters(NULL, true);
        hist_reset();
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(o_new.view, sinks.view, q.view, ckv.view,
                NULL, NULL, NULL, c_len, 0u, 512u, p0, 0u, 0u, s));
        REQUIRE(myds_metal_commit_batch());

        REQUIRE(hist_count("kernel_flash_attn_ext_vec_f16_dk512_dv512") == 1u);
        REQUIRE(hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512") == 0u);
        myds_v41_gpu_prefill_attn_l01_mma_counters_t lc;
        myds_v41_gpu_prefill_attn_l01_mma_counters(&lc, false);
        REQUIRE(lc.encoded == 0u);
        verify_exact_0ulp(myds_metal_buffer_data(o_ref.view), myds_metal_buffer_data(o_new.view),
                          q_floats, "L01 c_len<32 vec fallback");
        printf("PASS [CAND-A3] c_len=31 (<32) stays bit-exact 0-ULP on scalar vec\n");

        myds_v41_gpu_scratch_free(s);
        guarded_free(o_new); guarded_free(o_ref);
        guarded_free(ckv); guarded_free(win); guarded_free(sinks); guarded_free(q);
    }

    /* 4b: chunk0 (p0=0 < 127) stays vec -> carry roll -> chunk1 (p0=256) uses L01_MMA. */
    {
        const uint32_t c0 = 256u, c1 = 128u;
        struct guarded_buf q0     = guarded_alloc((size_t)c0 * 64u * 512u * 4u);
        struct guarded_buf q1     = guarded_alloc((size_t)c1 * 64u * 512u * 4u);
        struct guarded_buf sinks  = guarded_alloc(64u * 4u);
        struct guarded_buf ckv0   = guarded_alloc((size_t)c0 * 512u * 4u);
        struct guarded_buf ckv1   = guarded_alloc((size_t)c1 * 512u * 4u);
        struct guarded_buf o0_ref = guarded_alloc((size_t)c0 * 64u * 512u * 4u);
        struct guarded_buf o1_ref = guarded_alloc((size_t)c1 * 64u * 512u * 4u);
        struct guarded_buf o0_new = guarded_alloc((size_t)c0 * 64u * 512u * 4u);
        struct guarded_buf o1_new = guarded_alloc((size_t)c1 * 64u * 512u * 4u);

        fill_f32_buf(q0.view, (size_t)c0 * 64u * 512u, 3101u);
        fill_f32_buf(q1.view, (size_t)c1 * 64u * 512u, 3102u);
        fill_f32_buf(sinks.view, 64u, 3103u);
        fill_f32_buf(ckv0.view, (size_t)c0 * 512u, 3104u);
        fill_f32_buf(ckv1.view, (size_t)c1 * 512u, 3105u);

        reset_env_defaults();
        myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(4096u);
        REQUIRE(s != NULL);

        /* Reference: vec -> carry roll -> vec */
        setenv("MYDS_PREFILL_ATTN_L01_MMA", "0", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(o0_ref.view, sinks.view, q0.view, ckv0.view,
                NULL, NULL, NULL, c0, 0u, 512u, 0u, 0u, 0u, s));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(o1_ref.view, sinks.view, q1.view, ckv1.view,
                NULL, NULL, NULL, c1, 0u, 512u, 0u, c0, 0u, s));
        REQUIRE(myds_metal_commit_batch());

        /* Candidate: chunk0 (p0=0) MUST use vec, chunk1 (p0=256) uses L01_MMA */
        setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1);
        myds_v41_gpu_prefill_attn_l01_mma_counters(NULL, true);
        hist_reset();
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(o0_new.view, sinks.view, q0.view, ckv0.view,
                NULL, NULL, NULL, c0, 0u, 512u, 0u, 0u, 0u, s));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(o1_new.view, sinks.view, q1.view, ckv1.view,
                NULL, NULL, NULL, c1, 0u, 512u, 0u, c0, 0u, s));
        REQUIRE(myds_metal_commit_batch());

        REQUIRE(hist_count("kernel_flash_attn_ext_vec_f16_dk512_dv512") == 1u);
        REQUIRE(hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512") == 1u);
        myds_v41_gpu_prefill_attn_l01_mma_counters_t lc;
        myds_v41_gpu_prefill_attn_l01_mma_counters(&lc, false);
        REQUIRE(lc.encoded == 1u);

        verify_exact_0ulp(myds_metal_buffer_data(o0_ref.view), myds_metal_buffer_data(o0_new.view),
                          (size_t)c0 * 64u * 512u, "L01 chunk0 vec");
        struct parity_stats st = compute_parity((const float *)myds_metal_buffer_data(o1_ref.view),
                                                (const float *)myds_metal_buffer_data(o1_new.view),
                                                c1 * 64u, 512u);
        REQUIRE(st.max_abs_err < 1e-4 && st.l2_rel < 1e-4 && st.min_cos > 0.999999);
        printf("PASS [CAND-A3] chunk0_vec(0-ULP) + carry_roll -> chunk1_l01_mma (max_abs=%.3e l2_rel=%.3e min_cos=%.9f)\n",
               st.max_abs_err, st.l2_rel, st.min_cos);

        myds_v41_gpu_scratch_free(s);
        guarded_free(o1_new); guarded_free(o0_new); guarded_free(o1_ref); guarded_free(o0_ref);
        guarded_free(ckv1); guarded_free(ckv0); guarded_free(sinks); guarded_free(q1); guarded_free(q0);
    }
}

/* 5. GPU microbenchmark comparing L0/L1 vec vs L01_MMA. */
static double median_of(double *arr, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (arr[j] < arr[i]) { double t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    return arr[n / 2];
}

static void run_bench(void) {
    printf("\n=== GPU Microbenchmark (%s) ===\n", g_device.name.UTF8String);
    reset_env_defaults();
    const uint32_t c_len = 2048u;
    const size_t q_floats = (size_t)c_len * 64u * 512u;

    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(4096u);
    REQUIRE(s != NULL);
    myds_metal_buffer_t *q_f32 = myds_metal_buffer_alloc(q_floats * 4u);
    myds_metal_buffer_t *sinks = myds_metal_buffer_alloc(64u * 4u);
    myds_metal_buffer_t *win   = myds_metal_buffer_alloc(128u * 512u * 4u);
    myds_metal_buffer_t *ckv   = myds_metal_buffer_alloc((size_t)c_len * 512u * 4u);
    myds_metal_buffer_t *out   = myds_metal_buffer_alloc(q_floats * 4u);

    fill_f32_buf(q_f32, q_floats, 41u);
    fill_f32_buf(sinks, 64u, 42u);
    fill_f32_buf(win, 128u * 512u, 43u);
    fill_f32_buf(ckv, (size_t)c_len * 512u, 44u);

    const int n_runs = 5;
    struct { const char *label; const char *l01; } arms[] = {
        { "L0/L1 baseline vec (rope_Q + mask+vec + rope_Out)", "0" },
        { "L0/L1 MMA q8       (rope_Q + slab_q8  + rope_Out)", "1" },
    };
    double times[2][5];
    for (int it = -1; it < n_runs; ++it) {
        for (int ai = 0; ai < 2; ++ai) {
            setenv("MYDS_PREFILL_ATTN_L01_MMA", arms[ai].l01, 1);
            double t0 = myds_metal_host_time();
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_v41_gpu_rope_batch_round(q_f32, 512u, 64u, 2048u, c_len, false, false));
            REQUIRE(myds_v41_gpu_carry_stage(s, win, 2048u));
            REQUIRE(myds_v41_gpu_attention_chunk_bf16(out, sinks, q_f32, ckv, NULL, NULL, NULL,
                    c_len, 0u, 512u, 2048u, 0u, 0u, s));
            REQUIRE(myds_v41_gpu_rope_batch_round(out, 512u, 64u, 2048u, c_len, false, true));
            REQUIRE(myds_metal_commit_batch());
            double dt = (myds_metal_host_time() - t0) * 1000.0;
            if (it >= 0) times[ai][it] = dt;
        }
    }
    printf("\n--- L0/L1 Window-Only Pipeline (c_len=2048, p0=2048, 128 keys, interleaved median of %d) ---\n", n_runs);
    double base = median_of(times[0], n_runs);
    for (int ai = 0; ai < 2; ++ai) {
        double med = median_of(times[ai], n_runs);
        printf("  %s : %7.2f ms  (%5.2fx, %+6.2f ms)\n",
               arms[ai].label, med, base / med, med - base);
    }

    myds_metal_buffer_free(out); myds_metal_buffer_free(ckv); myds_metal_buffer_free(win);
    myds_metal_buffer_free(sinks); myds_metal_buffer_free(q_f32);
    myds_v41_gpu_scratch_free(s);
}

int main(int argc, char **argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) do_bench = true;
    }

    g_hist_enabled = 1;
    REQUIRE(myds_metal_init("metal"));

    test_knob_contracts();
    test_pso_residency();

    /* [CAND-A3] L0/L1 parity across p0 in {127, 256, 2048}, c_len in {32, 122, 2048} */
    const uint32_t l01_p0s[3]   = { 127u, 256u, 2048u };
    const uint32_t l01_clens[3] = {  32u, 122u, 2048u };
    for (int pi = 0; pi < 3; ++pi) {
        for (int ci = 0; ci < 3; ++ci) {
            test_cand_a3_l01_mma(l01_p0s[pi], l01_clens[ci]);
        }
    }
    test_cand_a3_fallback_and_carry_roll();

    if (do_bench) {
        g_hist_enabled = 0;
        run_bench();
    }

    printf("\nALL WAVE 2 PREFILL ATTENTION [CAND-A3] TESTS PASSED!\n");
    myds_metal_cleanup();
    return 0;
}
