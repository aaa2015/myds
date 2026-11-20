/* Comprehensive verification & M4 GPU microbenchmark for Wave 2 Prefill Attention:
 *   1. [CAND-A1 S1-S2] MYDS_PREFILL_ATTN_Q16 (0/1/16/32):
 *      - Fail-closed knob contract & /debug/state counters.
 *      - Strict 0-ULP (memcmp == 0) for Q=16 (4,160 B / zero-sq) & Q=32 (8,320 B / zero-sq)
 *        with kernel_dsv41_rope_round_f16 vs q8 heads_maskless_slab + kernel_dsv41_rope_round
 *        across c_len in {32, 64, 122, 512, 2048}.
 *   2. [P-R1 Site 2] MYDS_PREFILL_ATTN_ROPE_FUSE (0/1):
 *      - Fail-closed knob contract & /debug/state counters.
 *      - Strict 0-ULP (memcmp == 0) for q8, q16, q32 fused epilogue vs
 *        heads_maskless_slab + rope_batch_round across arbitrary p0 and c_len,
 *        explicitly checking RoPE dims [448..511] & non-RoPE dims [0..447],
 *        both compressed=true (YaRN, L2..39) and compressed=false (L0/L1).
 *   3. [CAND-A3] MYDS_PREFILL_ATTN_L01_MMA (0/1):
 *      - Fail-closed knob contract & /debug/state counters.
 *      - Numerical parity (max_abs_err, l2_rel, min_cos) for L0/L1 window_only_slab
 *        (q8/q16/q32 + fused RoPE) vs baseline vec across p0 in {127, 256, 2048},
 *        c_len in {32, 122, 2048}; verifies mask/fill/vec skipped, p0<127 stays vec,
 *        carry roll chunk0->chunk1, and gathered_cache invalidation.
 *   4. PSO residency probe & --bench M4 GPU microbenchmark table.
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

static uint16_t f32_to_bf16(float v) {
    uint32_t b;
    memcpy(&b, &v, 4);
    b = (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
    return (uint16_t)b;
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

static void fill_bf16_buf(myds_metal_buffer_t *b, size_t n, uint32_t seed) {
    uint16_t *p = (uint16_t *)myds_metal_buffer_data(b);
    for (size_t i = 0; i < n; ++i) p[i] = f32_to_bf16(sample_f32((uint32_t)i, seed));
}

static void fill_ids_buf(myds_metal_buffer_t *b, uint32_t c_len, uint32_t top_k,
                         uint32_t base_pos, uint32_t c_start, uint32_t ratio,
                         uint32_t n_comp, uint32_t seed) {
    int32_t *ids = (int32_t *)myds_metal_buffer_data(b);
    for (uint32_t r = 0; r < c_len; ++r) {
        uint32_t count = (base_pos + c_start + r + 1u) / ratio;
        if (count > n_comp) count = n_comp;
        for (uint32_t j = 0; j < top_k; ++j) {
            uint32_t h = (r * 1315423911u + j * 2654435761u + seed * 97u);
            ids[(size_t)r * top_k + j] = (count > 0u) ? (int32_t)(h % count) : -1;
        }
    }
}

static void reset_env_defaults(void) {
    unsetenv("MYDS_PREFILL_ATTN_FAST");
    unsetenv("MYDS_PREFILL_ATTN_Q16");
    unsetenv("MYDS_PREFILL_ATTN_ROPE_FUSE");
    unsetenv("MYDS_PREFILL_ATTN_L01_MMA");
    unsetenv("MYDS_ATTN_HEADS_MODE");
    setenv("MYDS_PREFILL_ATTN", "auto", 1);
}

/* 1. Fail-closed knob validation contract for all 3 new knobs. */
static void test_knob_contracts(void) {
    reset_env_defaults();
    REQUIRE(myds_v41_gpu_prefill_attn_q16_mode() == 0);
    REQUIRE(myds_v41_gpu_prefill_attn_rope_fuse_enabled() == 0);
    REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == 0);

    setenv("MYDS_PREFILL_ATTN_Q16", "0", 1);  REQUIRE(myds_v41_gpu_prefill_attn_q16_mode() == 0);
    setenv("MYDS_PREFILL_ATTN_Q16", "1", 1);  REQUIRE(myds_v41_gpu_prefill_attn_q16_mode() == 16);
    setenv("MYDS_PREFILL_ATTN_Q16", "16", 1); REQUIRE(myds_v41_gpu_prefill_attn_q16_mode() == 16);
    setenv("MYDS_PREFILL_ATTN_Q16", "32", 1); REQUIRE(myds_v41_gpu_prefill_attn_q16_mode() == 32);
    const char *bad_q16[] = {"", "2", "8", "64", "true", "16 ", "-1"};
    for (size_t i = 0; i < sizeof(bad_q16)/sizeof(bad_q16[0]); ++i) {
        setenv("MYDS_PREFILL_ATTN_Q16", bad_q16[i], 1);
        REQUIRE(myds_v41_gpu_prefill_attn_q16_mode() == -1);
        REQUIRE(myds_v41_gpu_scratch_create(4096u) == NULL);
    }
    unsetenv("MYDS_PREFILL_ATTN_Q16");

    setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "0", 1); REQUIRE(myds_v41_gpu_prefill_attn_rope_fuse_enabled() == 0);
    setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "1", 1); REQUIRE(myds_v41_gpu_prefill_attn_rope_fuse_enabled() == 1);
    const char *bad_bool[] = {"", "2", "01", "true", "1 ", "-1"};
    for (size_t i = 0; i < sizeof(bad_bool)/sizeof(bad_bool[0]); ++i) {
        setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", bad_bool[i], 1);
        REQUIRE(myds_v41_gpu_prefill_attn_rope_fuse_enabled() == -1);
        REQUIRE(myds_v41_gpu_scratch_create(4096u) == NULL);
    }
    unsetenv("MYDS_PREFILL_ATTN_ROPE_FUSE");

    setenv("MYDS_PREFILL_ATTN_L01_MMA", "0", 1); REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == 0);
    setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1); REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == 1);
    for (size_t i = 0; i < sizeof(bad_bool)/sizeof(bad_bool[0]); ++i) {
        setenv("MYDS_PREFILL_ATTN_L01_MMA", bad_bool[i], 1);
        REQUIRE(myds_v41_gpu_prefill_attn_l01_mma_enabled() == -1);
        REQUIRE(myds_v41_gpu_scratch_create(4096u) == NULL);
    }
    reset_env_defaults();
    printf("PASS knob_contracts (Q16=0/1/16/32, ROPE_FUSE=0/1, L01_MMA=0/1, all fail-closed)\n");
}

/* 2. PSO residency check for all Wave 2 kernels. */
static void probe_one_pso(const char *name, bool specialized, int32_t nsg, size_t dyn_smem) {
    NSError *err = nil;
    id<MTLFunction> fn = nil;
    if (specialized) {
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
        fn = [g_library newFunctionWithName:@(name) constantValues:vals error:&err];
    } else {
        fn = [g_library newFunctionWithName:@(name)];
    }
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    REQUIRE(pso.threadExecutionWidth == 32u);
    REQUIRE(pso.staticThreadgroupMemoryLength == 0u);
    const unsigned tpg = (unsigned)(nsg * 32);
    REQUIRE(pso.maxTotalThreadsPerThreadgroup >= tpg);
    if (specialized) {
        REQUIRE(pso.maxTotalThreadsPerThreadgroup == 1024u);
    }
    const unsigned tgs_by_thr = (unsigned)(pso.maxTotalThreadsPerThreadgroup / tpg);
    const unsigned tgs_by_smem = dyn_smem ? (unsigned)(32768u / dyn_smem) : 32u;
    const unsigned tgs_core = tgs_by_thr < tgs_by_smem ? tgs_by_thr : tgs_by_smem;
    printf("PSO  %-64s nsg=%2d maxThr=%4lu staticSmem=%lu dynSmem=%5zu TGs/core=%u\n",
           name, nsg, (unsigned long)pso.maxTotalThreadsPerThreadgroup,
           (unsigned long)pso.staticThreadgroupMemoryLength, dyn_smem, tgs_core);
}

static void test_pso_residency(void) {
    printf("=== PSO Residency Probe (%s) ===\n", g_device.name.UTF8String);
    probe_one_pso("kernel_dsv41_rope_round_f16", false, 1, 0u);
    probe_one_pso("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512", true, 4, 10272u);
    probe_one_pso("kernel_myds_flash_attn_heads_maskless_slab_q16_f16_dk512_dv512", true, 8, 4160u);
    probe_one_pso("kernel_myds_flash_attn_heads_maskless_slab_q32_f16_dk512_dv512", true, 16, 8320u);
    probe_one_pso("kernel_myds_flash_attn_heads_maskless_slab_rope_f16_dk512_dv512", true, 4, 10272u);
    probe_one_pso("kernel_myds_fa_maskless_slab_q16_rope_f16_dk512_dv512", true, 8, 4416u);
    probe_one_pso("kernel_myds_fa_maskless_slab_q32_rope_f16_dk512_dv512", true, 16, 8576u);
    printf("PASS pso_residency (all 7 kernels compiled, maxThr=1024, staticSmem=0)\n");
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

/* 3. [CAND-A1 S1-S2] + [P-R1 Site 2] Strict 0-ULP tests across c_len in {32, 64, 122, 512, 2048}. */
static void test_cand_a1_and_pr1_0ulp(uint32_t c_len, uint32_t base_pos, uint32_t c_start,
                                      uint32_t ratio, bool rope_compressed) {
    const uint32_t p0 = base_pos + c_start;
    const uint32_t top_k = 512u;
    const uint32_t n_comp = (p0 + c_len) / ratio;
    REQUIRE(n_comp >= 512u);

    const size_t q_floats = (size_t)c_len * 64u * 512u;
    struct guarded_buf q_raw   = guarded_alloc(q_floats * 4u);
    struct guarded_buf q_f32   = guarded_alloc(q_floats * 4u);
    struct guarded_buf q_f16   = guarded_alloc(q_floats * 2u);
    struct guarded_buf sinks   = guarded_alloc(64u * 4u);
    struct guarded_buf win     = guarded_alloc(128u * 512u * 4u);
    struct guarded_buf ckv     = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf comp    = guarded_alloc((size_t)n_comp * 512u * 2u);
    struct guarded_buf ids     = guarded_alloc((size_t)c_len * top_k * 4u);
    struct guarded_buf gath    = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
    struct guarded_buf out_q8  = guarded_alloc(q_floats * 4u);
    struct guarded_buf out_q8r = guarded_alloc(q_floats * 4u);
    struct guarded_buf out_cand = guarded_alloc(q_floats * 4u);

    const uint32_t seed = c_len * 31u + p0 * 7u + ratio * 13u + (rope_compressed ? 3u : 0u);
    fill_f32_buf(q_raw.view, q_floats, seed + 1u);
    memcpy(myds_metal_buffer_data(q_f32.view), myds_metal_buffer_data(q_raw.view), q_floats * 4u);
    fill_f32_buf(sinks.view, 64u, seed + 2u);
    fill_f32_buf(win.view, 128u * 512u, seed + 3u);
    fill_f32_buf(ckv.view, (size_t)c_len * 512u, seed + 4u);
    fill_bf16_buf(comp.view, (size_t)n_comp * 512u, seed + 5u);
    fill_ids_buf(ids.view, c_len, top_k, base_pos, c_start, ratio, n_comp, seed + 6u);

    reset_env_defaults();
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(n_comp + 2048u);
    REQUIRE(s != NULL);

    /* Reference A: rope_batch_round(q_f32) + q8 heads_maskless_slab -> out_q8 */
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
    setenv("MYDS_PREFILL_ATTN_Q16", "0", 1);
    setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "0", 1);
    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_rope_batch_round(q_f32.view, 512u, 64u, p0, c_len, rope_compressed, false));
    REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out_q8.view, sinks.view, q_f32.view, ckv.view,
            comp.view, ids.view, gath.view, c_len, n_comp, top_k, base_pos, c_start, ratio, s));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512") == 1u);

    /* Reference B: out_q8 + separate rope_batch_round(inverse=true) -> out_q8r */
    memcpy(myds_metal_buffer_data(out_q8r.view), myds_metal_buffer_data(out_q8.view), q_floats * 4u);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_rope_batch_round(out_q8r.view, 512u, 64u, p0, c_len, rope_compressed, true));
    REQUIRE(myds_metal_commit_batch());

    /* Verify RoPE actually modified dims 448..511 (non-trivial test!) */
    {
        const float *a = (const float *)myds_metal_buffer_data(out_q8.view);
        const float *b = (const float *)myds_metal_buffer_data(out_q8r.view);
        size_t rope_changed = 0;
        for (size_t d = 448u; d < 512u; ++d) {
            if (fabsf(a[d] - b[d]) > 1e-6f) ++rope_changed;
        }
        REQUIRE(rope_changed > 32u);
    }

    /* Candidate 1: [CAND-A1] Q=16 (env="1" and "16") & Q=32 (env="32") without RoPE fusion -> 0-ULP vs out_q8 */
    const char *q_envs[2] = { (c_len == 32u) ? "1" : "16", "32" };
    const char *q_kernels[2] = {
        "kernel_myds_flash_attn_heads_maskless_slab_q16_f16_dk512_dv512",
        "kernel_myds_flash_attn_heads_maskless_slab_q32_f16_dk512_dv512"
    };
    for (int qi = 0; qi < 2; ++qi) {
        setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
        setenv("MYDS_PREFILL_ATTN_Q16", q_envs[qi], 1);
        setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "0", 1);
        REQUIRE(myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, c_len, n_comp, top_k, base_pos, c_start, ratio));

        myds_v41_gpu_prefill_attn_q16_counters(NULL, true);
        hist_reset();
        bool rope_consumed = true;
        const myds_v41_gpu_attention_chunk_opts_t opts = {
            .q_f16 = q_f16.view,
            .fuse_rope = false,
            .rope_compressed = rope_compressed,
            .rope_consumed = &rope_consumed,
        };
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_rope_batch_round_f16(q_f16.view, q_raw.view, 512u, 64u, p0, c_len, rope_compressed, false));
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_ex(out_cand.view, sinks.view, q_raw.view, ckv.view,
                comp.view, ids.view, gath.view, c_len, n_comp, top_k, base_pos, c_start, ratio, s, true, &opts));
        REQUIRE(myds_metal_commit_batch());

        REQUIRE(!rope_consumed);
        REQUIRE(hist_count("kernel_dsv41_rope_round_f16") == 1u);
        REQUIRE(hist_count(q_kernels[qi]) == 1u);
        myds_v41_gpu_prefill_attn_q16_counters_t qc;
        myds_v41_gpu_prefill_attn_q16_counters(&qc, false);
        REQUIRE(qc.q_rope_f16_encoded == 1u);
        REQUIRE((qi == 0 ? qc.q16_encoded : qc.q32_encoded) == 1u);
        verify_exact_0ulp(myds_metal_buffer_data(out_q8.view), myds_metal_buffer_data(out_cand.view),
                          q_floats, qi == 0 ? "CAND-A1 Q16" : "CAND-A1 Q32");
    }

    /* Candidate 2: [P-R1 Site 2] Fused epilogue (q8, q16, q32) -> strict 0-ULP vs out_q8r */
    const char *rf_envs[3] = { "0", "16", "32" };
    const char *rf_kernels[3] = {
        "kernel_myds_flash_attn_heads_maskless_slab_rope_f16_dk512_dv512",
        "kernel_myds_fa_maskless_slab_q16_rope_f16_dk512_dv512",
        "kernel_myds_fa_maskless_slab_q32_rope_f16_dk512_dv512"
    };
    for (int ri = 0; ri < 3; ++ri) {
        setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
        setenv("MYDS_PREFILL_ATTN_Q16", rf_envs[ri], 1);
        setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "1", 1);
        myds_v41_gpu_prefill_attn_rope_fuse_counters(NULL, true);
        hist_reset();
        bool rope_consumed = false;
        const myds_v41_gpu_attention_chunk_opts_t opts = {
            .q_f16 = (ri == 0) ? NULL : q_f16.view,
            .fuse_rope = true,
            .rope_compressed = rope_compressed,
            .rope_consumed = &rope_consumed,
        };
        REQUIRE(myds_metal_begin_batch());
        if (ri > 0) {
            REQUIRE(myds_v41_gpu_rope_batch_round_f16(q_f16.view, q_raw.view, 512u, 64u, p0, c_len, rope_compressed, false));
        }
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_ex(out_cand.view, sinks.view, q_f32.view, ckv.view,
                comp.view, ids.view, gath.view, c_len, n_comp, top_k, base_pos, c_start, ratio, s, true, &opts));
        REQUIRE(myds_metal_commit_batch());

        REQUIRE(rope_consumed);
        REQUIRE(hist_count(rf_kernels[ri]) == 1u);
        REQUIRE(hist_count("kernel_dsv41_rope_round") == 0u);
        myds_v41_gpu_prefill_attn_rope_fuse_counters_t rfc;
        myds_v41_gpu_prefill_attn_rope_fuse_counters(&rfc, false);
        REQUIRE((ri == 0 ? rfc.q8_encoded : ri == 1 ? rfc.q16_encoded : rfc.q32_encoded) == 1u);

        /* Explicitly verify non-RoPE [0..447], RoPE [448..511], and exact BF16 lower-16 zero bits. */
        const uint32_t *ref_u = (const uint32_t *)myds_metal_buffer_data(out_q8r.view);
        const uint32_t *got_u = (const uint32_t *)myds_metal_buffer_data(out_cand.view);
        for (size_t i = 0; i < q_floats; ++i) {
            REQUIRE((got_u[i] & 0xffffu) == 0u);
        }
        verify_exact_0ulp(ref_u, got_u, q_floats,
                          ri == 0 ? "P-R1 q8_rope" : ri == 1 ? "P-R1 q16_rope" : "P-R1 q32_rope");
    }

    printf("PASS 0-ULP [CAND-A1 Q16/Q32 + P-R1 q8/q16/q32] c_len=%4u p0=%6u ratio=%u comp_rope=%d (memcmp==0, %zu floats)\n",
           c_len, p0, ratio, (int)rope_compressed, q_floats);

    myds_v41_gpu_scratch_free(s);
    guarded_free(out_cand); guarded_free(out_q8r); guarded_free(out_q8);
    guarded_free(gath); guarded_free(ids); guarded_free(comp); guarded_free(ckv);
    guarded_free(win); guarded_free(sinks); guarded_free(q_f16); guarded_free(q_f32); guarded_free(q_raw);
}

/* Verify non-maskless fallback chunks keep FP32 and rope_consumed=false. */
static void test_non_maskless_fallback_safety(void) {
    reset_env_defaults();
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
    setenv("MYDS_PREFILL_ATTN_Q16", "16", 1);
    setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "1", 1);
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(8192u);
    REQUIRE(s != NULL);
    struct guarded_buf gath = guarded_alloc(64u * 640u * 512u * 2u);

    /* c_len < 32 -> not eligible for Q16/Q32 */
    REQUIRE(!myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, 31u, 1024u, 512u, 2048u, 0u, 2u));
    /* (p0+1)/ratio < 512 -> not all visible */
    REQUIRE(!myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, 64u, 600u, 512u, 1022u, 0u, 2u));
    /* p0 < 127 -> carry incomplete */
    REQUIRE(!myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, 64u, 600u, 512u, 126u, 0u, 1u));
    /* L0/L1 with MYDS_PREFILL_ATTN_L01_MMA=0 -> false */
    REQUIRE(!myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, 64u, 0u, 512u, 2048u, 0u, 0u));
    /* L0/L1 with MYDS_PREFILL_ATTN_L01_MMA=1 & p0>=127 & c_len>=32 -> true; p0<127 -> false */
    setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1);
    REQUIRE(myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, 64u, 0u, 512u, 127u, 0u, 0u));
    REQUIRE(!myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, 64u, 0u, 512u, 126u, 0u, 0u));
    REQUIRE(!myds_v41_gpu_prefill_attn_wants_q_f16(s, gath.view, 31u, 0u, 512u, 2048u, 0u, 0u));

    /* Actual vec fallback chunk (c_len=31) must leave rope_consumed=false */
    const uint32_t c_len = 31u, base_pos = 2048u, ratio = 2u, n_comp = (base_pos + c_len) / ratio;
    struct guarded_buf q     = guarded_alloc((size_t)c_len * 64u * 512u * 4u);
    struct guarded_buf sinks = guarded_alloc(64u * 4u);
    struct guarded_buf win   = guarded_alloc(128u * 512u * 4u);
    struct guarded_buf ckv   = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf comp  = guarded_alloc((size_t)n_comp * 512u * 2u);
    struct guarded_buf ids   = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf out   = guarded_alloc((size_t)c_len * 64u * 512u * 4u);
    fill_f32_buf(q.view, (size_t)c_len * 64u * 512u, 901u);
    fill_f32_buf(sinks.view, 64u, 902u);
    fill_f32_buf(win.view, 128u * 512u, 903u);
    fill_f32_buf(ckv.view, (size_t)c_len * 512u, 904u);
    fill_bf16_buf(comp.view, (size_t)n_comp * 512u, 905u);
    fill_ids_buf(ids.view, c_len, 512u, base_pos, 0u, ratio, n_comp, 906u);

    bool rope_consumed = true;
    const myds_v41_gpu_attention_chunk_opts_t opts = {
        .q_f16 = NULL, .fuse_rope = true, .rope_compressed = true, .rope_consumed = &rope_consumed
    };
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win.view, base_pos));
    REQUIRE(myds_v41_gpu_attention_chunk_ex(out.view, sinks.view, q.view, ckv.view,
            comp.view, ids.view, gath.view, c_len, n_comp, 512u, base_pos, 0u, ratio, s, true, &opts));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(!rope_consumed);

    printf("PASS non_maskless_fallback_safety (wants_q_f16=false, rope_consumed=false on vec/early chunks)\n");
    guarded_free(out); guarded_free(ids); guarded_free(comp); guarded_free(ckv);
    guarded_free(win); guarded_free(sinks); guarded_free(q); guarded_free(gath);
    myds_v41_gpu_scratch_free(s);
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

/* 4. [CAND-A3] L0/L1 Window-Only Direct-Slab MMA parity & dispatch checks. */
static void test_cand_a3_l01_mma(uint32_t p0, uint32_t c_len) {
    const size_t q_floats = (size_t)c_len * 64u * 512u;
    struct guarded_buf q_raw     = guarded_alloc(q_floats * 4u);
    struct guarded_buf q_f32     = guarded_alloc(q_floats * 4u);
    struct guarded_buf q_f16     = guarded_alloc(q_floats * 2u);
    struct guarded_buf sinks     = guarded_alloc(64u * 4u);
    struct guarded_buf win       = guarded_alloc(128u * 512u * 4u);
    struct guarded_buf ckv       = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf out_vec   = guarded_alloc(q_floats * 4u);
    struct guarded_buf out_mma   = guarded_alloc(q_floats * 4u);
    struct guarded_buf out_mmar  = guarded_alloc(q_floats * 4u);
    struct guarded_buf out_fuse  = guarded_alloc(q_floats * 4u);

    const uint32_t seed = 7000u + p0 * 5u + c_len * 19u;
    fill_f32_buf(q_raw.view, q_floats, seed + 1u);
    memcpy(myds_metal_buffer_data(q_f32.view), myds_metal_buffer_data(q_raw.view), q_floats * 4u);
    fill_f32_buf(sinks.view, 64u, seed + 2u);
    fill_f32_buf(win.view, 128u * 512u, seed + 3u);
    fill_f32_buf(ckv.view, (size_t)c_len * 512u, seed + 4u);

    reset_env_defaults();
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(4096u);
    REQUIRE(s != NULL);

    /* Baseline: MYDS_PREFILL_ATTN_L01_MMA=0 -> scalar vec */
    setenv("MYDS_PREFILL_ATTN_L01_MMA", "0", 1);
    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_rope_batch_round(q_f32.view, 512u, 64u, p0, c_len, false, false));
    REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out_vec.view, sinks.view, q_f32.view, ckv.view,
            NULL, NULL, NULL, c_len, 0u, 512u, p0, 0u, 0u, s));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(hist_count("kernel_flash_attn_ext_vec_f16_dk512_dv512") == 1u);

    /* Candidate A: L01_MMA=1 (q8) */
    setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1);
    setenv("MYDS_PREFILL_ATTN_Q16", "0", 1);
    setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "0", 1);
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
    myds_v41_gpu_prefill_attn_l01_mma_counters_t lc;
    myds_v41_gpu_prefill_attn_l01_mma_counters(&lc, false);
    REQUIRE(lc.encoded == 1u && lc.q16_encoded == 0u && lc.rope_fuse_encoded == 0u);

    struct parity_stats st = compute_parity((const float *)myds_metal_buffer_data(out_vec.view),
                                            (const float *)myds_metal_buffer_data(out_mma.view),
                                            c_len * 64u, 512u);
    REQUIRE(st.max_abs_err < 1e-4);
    REQUIRE(st.l2_rel < 1e-4);
    REQUIRE(st.min_cos > 0.999999);

    /* Candidate B: L01_MMA=1 + Q16 + ROPE_FUSE=1 (must be strict 0-ULP vs L01_MMA(q8) + rope_batch_round(compressed=false)) */
    memcpy(myds_metal_buffer_data(out_mmar.view), myds_metal_buffer_data(out_mma.view), q_floats * 4u);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_rope_batch_round(out_mmar.view, 512u, 64u, p0, c_len, false, true));
    REQUIRE(myds_metal_commit_batch());

    const char *l01_qmodes[3] = { "0", "16", "32" };
    for (int qi = 0; qi < 3; ++qi) {
        setenv("MYDS_PREFILL_ATTN_L01_MMA", "1", 1);
        setenv("MYDS_PREFILL_ATTN_Q16", l01_qmodes[qi], 1);
        setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "1", 1);
        myds_v41_gpu_prefill_attn_l01_mma_counters(NULL, true);
        myds_v41_gpu_prefill_attn_rope_fuse_counters(NULL, true);
        hist_reset();
        bool rope_consumed = false;
        const myds_v41_gpu_attention_chunk_opts_t opts = {
            .q_f16 = (qi == 0) ? NULL : q_f16.view,
            .fuse_rope = true,
            .rope_compressed = false,
            .rope_consumed = &rope_consumed,
        };
        REQUIRE(myds_metal_begin_batch());
        if (qi > 0) {
            REQUIRE(myds_v41_gpu_rope_batch_round_f16(q_f16.view, q_raw.view, 512u, 64u, p0, c_len, false, false));
        }
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
        REQUIRE(myds_v41_gpu_attention_chunk_ex(out_fuse.view, sinks.view, q_f32.view, ckv.view,
                NULL, NULL, NULL, c_len, 0u, 512u, p0, 0u, 0u, s, true, &opts));
        REQUIRE(myds_metal_commit_batch());

        REQUIRE(rope_consumed);
        REQUIRE(hist_count("kernel_myds_mask_chunk_window") == 0u);
        REQUIRE(hist_count("kernel_myds_mask_chunk_fill") == 0u);
        REQUIRE(hist_count("kernel_flash_attn_ext_vec_f16_dk512_dv512") == 0u);
        myds_v41_gpu_prefill_attn_l01_mma_counters(&lc, false);
        REQUIRE(lc.encoded == 1u && lc.rope_fuse_encoded == 1u);
        if (qi == 1) REQUIRE(lc.q16_encoded == 1u);
        if (qi == 2) REQUIRE(lc.q32_encoded == 1u);
        myds_v41_gpu_prefill_attn_rope_fuse_counters_t rfc;
        myds_v41_gpu_prefill_attn_rope_fuse_counters(&rfc, false);
        REQUIRE(rfc.l01_encoded == 1u);

        verify_exact_0ulp(myds_metal_buffer_data(out_mmar.view), myds_metal_buffer_data(out_fuse.view),
                          q_floats, qi == 0 ? "L01 q8_rope" : qi == 1 ? "L01 q16_rope" : "L01 q32_rope");
    }

    printf("PASS [CAND-A3] L0/L1 parity p0=%4u c_len=%4u vs vec: max_abs=%.3e l2_rel=%.3e min_cos=%.9f (q8/q16/q32+rope 0-ULP)\n",
           p0, c_len, st.max_abs_err, st.l2_rel, st.min_cos);

    myds_v41_gpu_scratch_free(s);
    guarded_free(out_fuse); guarded_free(out_mmar); guarded_free(out_mma); guarded_free(out_vec);
    guarded_free(ckv); guarded_free(win); guarded_free(sinks);
    guarded_free(q_f16); guarded_free(q_f32); guarded_free(q_raw);
}

/* Verify chunk 0 (p0 < 127) stays on vec and carry-rolls cleanly into chunk 1 L01_MMA. */
static void test_cand_a3_chunk0_vec_and_carry_roll(void) {
    const uint32_t c0 = 256u, c1 = 128u;
    struct guarded_buf q0    = guarded_alloc((size_t)c0 * 64u * 512u * 4u);
    struct guarded_buf q1    = guarded_alloc((size_t)c1 * 64u * 512u * 4u);
    struct guarded_buf sinks = guarded_alloc(64u * 4u);
    struct guarded_buf ckv0  = guarded_alloc((size_t)c0 * 512u * 4u);
    struct guarded_buf ckv1  = guarded_alloc((size_t)c1 * 512u * 4u);
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
    setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "1", 1);
    hist_reset();
    bool rc0 = true, rc1 = false;
    const myds_v41_gpu_attention_chunk_opts_t opts0 = { .q_f16 = NULL, .fuse_rope = true, .rope_compressed = false, .rope_consumed = &rc0 };
    const myds_v41_gpu_attention_chunk_opts_t opts1 = { .q_f16 = NULL, .fuse_rope = false, .rope_compressed = false, .rope_consumed = &rc1 };
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_attention_chunk_ex(o0_new.view, sinks.view, q0.view, ckv0.view,
            NULL, NULL, NULL, c0, 0u, 512u, 0u, 0u, 0u, s, true, &opts0));
    REQUIRE(myds_v41_gpu_attention_chunk_ex(o1_new.view, sinks.view, q1.view, ckv1.view,
            NULL, NULL, NULL, c1, 0u, 512u, 0u, c0, 0u, s, true, &opts1));
    REQUIRE(myds_metal_commit_batch());

    REQUIRE(!rc0 && !rc1);
    REQUIRE(hist_count("kernel_flash_attn_ext_vec_f16_dk512_dv512") == 1u);
    REQUIRE(hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512") == 1u);
    /* Chunk 0 must be bit-exact 0-ULP vs baseline */
    verify_exact_0ulp(myds_metal_buffer_data(o0_ref.view), myds_metal_buffer_data(o0_new.view),
                      (size_t)c0 * 64u * 512u, "L01 chunk0 vec");
    /* Chunk 1 must match vec within tight MMA tolerance */
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

/* 5. M4 GPU microbenchmark table. */
static double median_of(double *arr, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (arr[j] < arr[i]) { double t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    return arr[n / 2];
}

static void run_m4_bench(void) {
    printf("\n=== M4 GPU Microbenchmark (%s) ===\n", g_device.name.UTF8String);
    reset_env_defaults();
    const uint32_t c_len = 2048u;
    const uint32_t base_pos = 16384u;
    const uint32_t ratio = 1u;
    const uint32_t n_comp = base_pos + c_len;
    const size_t q_floats = (size_t)c_len * 64u * 512u;

    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(n_comp + 2048u);
    REQUIRE(s != NULL);
    myds_metal_buffer_t *q_raw = myds_metal_buffer_alloc(q_floats * 4u);
    myds_metal_buffer_t *q_f32 = myds_metal_buffer_alloc(q_floats * 4u);
    myds_metal_buffer_t *q_f16 = myds_metal_buffer_alloc(q_floats * 2u);
    myds_metal_buffer_t *sinks = myds_metal_buffer_alloc(64u * 4u);
    myds_metal_buffer_t *win   = myds_metal_buffer_alloc(128u * 512u * 4u);
    myds_metal_buffer_t *ckv   = myds_metal_buffer_alloc((size_t)c_len * 512u * 4u);
    myds_metal_buffer_t *comp  = myds_metal_buffer_alloc((size_t)n_comp * 512u * 2u);
    myds_metal_buffer_t *ids   = myds_metal_buffer_alloc((size_t)c_len * 512u * 4u);
    myds_metal_buffer_t *gath  = myds_metal_buffer_alloc((size_t)c_len * 640u * 512u * 2u);
    myds_metal_buffer_t *out   = myds_metal_buffer_alloc(q_floats * 4u);

    fill_f32_buf(q_raw, q_floats, 41u);
    fill_f32_buf(q_f32, q_floats, 41u);
    fill_f32_buf(sinks, 64u, 42u);
    fill_f32_buf(win, 128u * 512u, 43u);
    fill_f32_buf(ckv, (size_t)c_len * 512u, 44u);
    fill_bf16_buf(comp, (size_t)n_comp * 512u, 45u);
    fill_ids_buf(ids, c_len, 512u, base_pos, 0u, ratio, n_comp, 46u);

    /* Pre-populate q_f16 and compact-512 gathered once for isolated kernel-only bench */
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
    setenv("MYDS_PREFILL_ATTN_Q16", "16", 1);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_rope_batch_round_f16(q_f16, q_raw, 512u, 64u, base_pos, c_len, true, false));
    REQUIRE(myds_v41_gpu_carry_stage(s, win, base_pos));
    {
        bool rc = false;
        const myds_v41_gpu_attention_chunk_opts_t opts = { .q_f16 = q_f16, .fuse_rope = false, .rope_compressed = true, .rope_consumed = &rc };
        REQUIRE(myds_v41_gpu_attention_chunk_ex(out, sinks, q_f32, ckv, comp, ids, gath,
                c_len, n_comp, 512u, base_pos, 0u, ratio, s, true, &opts));
    }
    REQUIRE(myds_metal_commit_batch());

    const int n_runs = 5;

    /* Part A: End-to-end Layer-Chunk Attn Pipeline (interleaved across all 6 arms) */
    struct { const char *label; const char *q16; const char *rf; } e2e_arms[] = {
        { "q8  baseline (rope_Q + slab_q8  + rope_Out)", "0",  "0" },
        { "q8  + P-R1   (rope_Q + slab_q8_rope)       ", "0",  "1" },
        { "q16 zero-sq  (rope_f16 + q16  + rope_Out)  ", "16", "0" },
        { "q16 + P-R1   (rope_f16 + q16_rope)         ", "16", "1" },
        { "q32 zero-sq  (rope_f16 + q32  + rope_Out)  ", "32", "0" },
        { "q32 + P-R1   (rope_f16 + q32_rope)         ", "32", "1" },
    };
    double e2e_times[6][5];
    for (int it = -1; it < n_runs; ++it) {
        for (int ai = 0; ai < 6; ++ai) {
            setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
            setenv("MYDS_PREFILL_ATTN_Q16", e2e_arms[ai].q16, 1);
            setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", e2e_arms[ai].rf, 1);
            const int use_q16 = strcmp(e2e_arms[ai].q16, "0") != 0;
            const int use_rf  = strcmp(e2e_arms[ai].rf, "1") == 0;
            gathered_cache_invalidate(s);
            double t0 = myds_metal_host_time();
            bool rc = false;
            const myds_v41_gpu_attention_chunk_opts_t opts = {
                .q_f16 = use_q16 ? q_f16 : NULL,
                .fuse_rope = use_rf != 0,
                .rope_compressed = true,
                .rope_consumed = &rc,
            };
            REQUIRE(myds_metal_begin_batch());
            if (use_q16) {
                REQUIRE(myds_v41_gpu_rope_batch_round_f16(q_f16, q_raw, 512u, 64u, base_pos, c_len, true, false));
            } else {
                REQUIRE(myds_v41_gpu_rope_batch_round(q_f32, 512u, 64u, base_pos, c_len, true, false));
            }
            REQUIRE(myds_v41_gpu_carry_stage(s, win, base_pos));
            REQUIRE(myds_v41_gpu_attention_chunk_ex(out, sinks, q_f32, ckv, comp, ids, gath,
                    c_len, n_comp, 512u, base_pos, 0u, ratio, s, true, &opts));
            if (!rc) {
                REQUIRE(myds_v41_gpu_rope_batch_round(out, 512u, 64u, base_pos, c_len, true, true));
            }
            REQUIRE(myds_metal_commit_batch());
            double dt = (myds_metal_host_time() - t0) * 1000.0;
            if (it >= 0) e2e_times[ai][it] = dt;
        }
    }
    printf("\n--- [A] L2-39 Full Attn Pipeline (c_len=2048, 640 keys, interleaved median of %d: Q-RoPE + gather + Core + Out-RoPE) ---\n", n_runs);
    double base_e2e = median_of(e2e_times[0], n_runs);
    for (int ai = 0; ai < 6; ++ai) {
        double med = median_of(e2e_times[ai], n_runs);
        printf("  %s : %7.2f ms  (%5.2fx, %+6.2f ms)\n",
               e2e_arms[ai].label, med, base_e2e / med, med - base_e2e);
    }

    /* Part B: Isolated Attention Core Kernel Only (gather cached, interleaved across all 6 arms) */
    /* Prime compact-512 gather cache once before timed loop */
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
    setenv("MYDS_PREFILL_ATTN_Q16", "0", 1);
    setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", "0", 1);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win, base_pos));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out, sinks, q_f32, ckv, comp, ids, gath,
            c_len, n_comp, 512u, base_pos, 0u, ratio, s));
    REQUIRE(myds_metal_commit_batch());

    double core_times[6][5];
    for (int it = -1; it < n_runs; ++it) {
        for (int ai = 0; ai < 6; ++ai) {
            setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
            setenv("MYDS_PREFILL_ATTN_Q16", e2e_arms[ai].q16, 1);
            setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", e2e_arms[ai].rf, 1);
            const int use_q16 = strcmp(e2e_arms[ai].q16, "0") != 0;
            const int use_rf  = strcmp(e2e_arms[ai].rf, "1") == 0;
            bool rc = false;
            const myds_v41_gpu_attention_chunk_opts_t opts = {
                .q_f16 = use_q16 ? q_f16 : NULL,
                .fuse_rope = use_rf != 0,
                .rope_compressed = true,
                .rope_consumed = &rc,
            };
            double t0 = myds_metal_host_time();
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_v41_gpu_attention_chunk_ex(out, sinks, q_f32, ckv, comp, ids, gath,
                    c_len, n_comp, 512u, base_pos, 0u, ratio, s, true, &opts));
            REQUIRE(myds_metal_commit_batch());
            double dt = (myds_metal_host_time() - t0) * 1000.0;
            if (it >= 0) core_times[ai][it] = dt;
        }
    }
    printf("\n--- [B] L2-39 Core Attention Kernel Only (c_len=2048, 640 keys, interleaved median of %d, gather cached) ---\n", n_runs);
    double base_core = median_of(core_times[0], n_runs);
    for (int ai = 0; ai < 6; ++ai) {
        double med = median_of(core_times[ai], n_runs);
        printf("  %-14s (Q16=%2s RF=%s) : %7.2f ms  (%5.2fx, %+6.2f ms)\n",
               ai == 0 ? "q8_slab" : ai == 1 ? "q8_slab_rope" :
               ai == 2 ? "q16_zerosq" : ai == 3 ? "q16_rope" :
               ai == 4 ? "q32_zerosq" : "q32_rope",
               e2e_arms[ai].q16, e2e_arms[ai].rf, med, base_core / med, med - base_core);
    }

    /* Part C: L0/L1 Window-Only Attention (c_len=2048, p0=2048, 128 keys, interleaved) */
    struct { const char *label; const char *l01; const char *q16; const char *rf; } l01_arms[] = {
        { "L0/L1 baseline vec (rope_Q + mask+vec + rope_Out)", "0", "0",  "0" },
        { "L0/L1 MMA q8       (rope_Q + slab_q8  + rope_Out)", "1", "0",  "0" },
        { "L0/L1 MMA q8 + P-R1(rope_Q + slab_q8_rope)      ", "1", "0",  "1" },
        { "L0/L1 MMA q16+ P-R1(rope_f16 + q16_rope)        ", "1", "16", "1" },
        { "L0/L1 MMA q32+ P-R1(rope_f16 + q32_rope)        ", "1", "32", "1" },
    };
    double l01_times[5][5];
    for (int it = -1; it < n_runs; ++it) {
        for (int ai = 0; ai < 5; ++ai) {
            setenv("MYDS_PREFILL_ATTN_L01_MMA", l01_arms[ai].l01, 1);
            setenv("MYDS_PREFILL_ATTN_Q16", l01_arms[ai].q16, 1);
            setenv("MYDS_PREFILL_ATTN_ROPE_FUSE", l01_arms[ai].rf, 1);
            const int use_q16 = strcmp(l01_arms[ai].q16, "0") != 0;
            const int use_rf  = strcmp(l01_arms[ai].rf, "1") == 0;
            double t0 = myds_metal_host_time();
            bool rc = false;
            const myds_v41_gpu_attention_chunk_opts_t opts = {
                .q_f16 = use_q16 ? q_f16 : NULL,
                .fuse_rope = use_rf != 0,
                .rope_compressed = false,
                .rope_consumed = &rc,
            };
            REQUIRE(myds_metal_begin_batch());
            if (use_q16) {
                REQUIRE(myds_v41_gpu_rope_batch_round_f16(q_f16, q_raw, 512u, 64u, 2048u, c_len, false, false));
            } else {
                REQUIRE(myds_v41_gpu_rope_batch_round(q_f32, 512u, 64u, 2048u, c_len, false, false));
            }
            REQUIRE(myds_v41_gpu_carry_stage(s, win, 2048u));
            REQUIRE(myds_v41_gpu_attention_chunk_ex(out, sinks, q_f32, ckv, NULL, NULL, NULL,
                    c_len, 0u, 512u, 2048u, 0u, 0u, s, true, &opts));
            if (!rc) {
                REQUIRE(myds_v41_gpu_rope_batch_round(out, 512u, 64u, 2048u, c_len, false, true));
            }
            REQUIRE(myds_metal_commit_batch());
            double dt = (myds_metal_host_time() - t0) * 1000.0;
            if (it >= 0) l01_times[ai][it] = dt;
        }
    }
    printf("\n--- [C] L0/L1 Window-Only Pipeline (c_len=2048, p0=2048, 128 keys, interleaved median of %d) ---\n", n_runs);
    double base_l01 = median_of(l01_times[0], n_runs);
    for (int ai = 0; ai < 5; ++ai) {
        double med = median_of(l01_times[ai], n_runs);
        printf("  %s : %7.2f ms  (%5.2fx, %+6.2f ms)\n",
               l01_arms[ai].label, med, base_l01 / med, med - base_l01);
    }

    myds_metal_buffer_free(out); myds_metal_buffer_free(gath); myds_metal_buffer_free(ids);
    myds_metal_buffer_free(comp); myds_metal_buffer_free(ckv); myds_metal_buffer_free(win);
    myds_metal_buffer_free(sinks); myds_metal_buffer_free(q_f16); myds_metal_buffer_free(q_f32);
    myds_metal_buffer_free(q_raw);
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

    /* [CAND-A1 S1-S2] + [P-R1 Site 2] strict 0-ULP across c_len in {32, 64, 122, 512, 2048} */
    test_cand_a1_and_pr1_0ulp(  32u,   1023u, 0u, 2u, true);
    test_cand_a1_and_pr1_0ulp(  64u,   1536u, 0u, 2u, true);
    test_cand_a1_and_pr1_0ulp( 122u, 126976u, 2048u, 1u, true);
    test_cand_a1_and_pr1_0ulp( 512u,   4096u, 0u, 2u, true);
    test_cand_a1_and_pr1_0ulp(2048u,  16384u, 0u, 1u, true);
    /* Also verify compressed=false (base=10000 RoPE) on fused epilogue */
    test_cand_a1_and_pr1_0ulp(  64u,    511u, 0u, 1u, false);

    test_non_maskless_fallback_safety();

    /* [CAND-A3] L0/L1 parity across p0 in {127, 256, 2048}, c_len in {32, 122, 2048} */
    const uint32_t l01_p0s[3]   = { 127u, 256u, 2048u };
    const uint32_t l01_clens[3] = {  32u, 122u, 2048u };
    for (int pi = 0; pi < 3; ++pi) {
        for (int ci = 0; ci < 3; ++ci) {
            test_cand_a3_l01_mma(l01_p0s[pi], l01_clens[ci]);
        }
    }
    test_cand_a3_chunk0_vec_and_carry_roll();

    if (do_bench) {
        g_hist_enabled = 0;
        run_m4_bench();
    }

    printf("\nALL WAVE 2 PREFILL ATTENTION TESTS PASSED!\n");
    myds_metal_cleanup();
    return 0;
}
