/* Strict 0-ULP verification and M4 GPU microbenchmark for [P-NEW-3]
 * (MYDS_PREFILL_ATTN_FAST=1):
 *   1. Maskless + Direct Window Slab Reg-O heads kernel (10,272 B smem -> 3 TGs/core,
 *      skips kernel_flash_attn_ext_blk and kernel_myds_attn_gather_window).
 *   2. Compact-512 compressed KV gather (20% smaller gathered footprint, no mask write).
 *   3. Cross-layer gathered compressed KV reuse on single-chunk batches, with strict
 *      invalidation on topk/indexer-select, comp_kv writes, and multi-chunk transitions.
 */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <math.h>

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

/* 1. Verify Metal PSO geometry for maskless+slab kernel. */
static void test_pso_geometry(void) {
    MTLFunctionConstantValues *vals = [MTLFunctionConstantValues new];
    bool b_mask = false, b_sinks = true, b_bias = false, b_scap = false, b_kvpad = false, b_bc = false;
    int32_t ns10 = 512, ns20 = 512, nsg = 4;
    [vals setConstantValue:&b_mask  type:MTLDataTypeBool atIndex:300];
    [vals setConstantValue:&b_sinks type:MTLDataTypeBool atIndex:301];
    [vals setConstantValue:&b_bias  type:MTLDataTypeBool atIndex:302];
    [vals setConstantValue:&b_scap  type:MTLDataTypeBool atIndex:303];
    [vals setConstantValue:&b_kvpad type:MTLDataTypeBool atIndex:304];
    [vals setConstantValue:&b_bc    type:MTLDataTypeBool atIndex:310];
    [vals setConstantValue:&ns10    type:MTLDataTypeInt  atIndex:320];
    [vals setConstantValue:&ns20    type:MTLDataTypeInt  atIndex:321];
    [vals setConstantValue:&nsg     type:MTLDataTypeInt  atIndex:322];
    NSError *err = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512"
                                         constantValues:vals error:&err];
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    REQUIRE(pso.threadExecutionWidth == 32u);
    REQUIRE(pso.maxTotalThreadsPerThreadgroup >= 128u);
    REQUIRE(pso.staticThreadgroupMemoryLength == 0u);
    printf("PASS pso_geometry kernel=maskless_slab tew=%lu maxThreads=%lu staticSmem=%lu dynSmem=10272 (3 TGs/core)\n",
           (unsigned long)pso.threadExecutionWidth,
           (unsigned long)pso.maxTotalThreadsPerThreadgroup,
           (unsigned long)pso.staticThreadgroupMemoryLength);
}

/* 2. Strict 0-ULP parity across shapes, BF16/F32, visible/boundary cases. */
static void test_single_chunk_0ulp(uint32_t c_len, uint32_t base_pos, uint32_t c_start,
                                   uint32_t ratio, bool comp_bf16, bool expect_fast_matrix) {
    const uint32_t p0 = base_pos + c_start;
    const uint32_t top_k = 512u;
    const uint32_t n_comp = (p0 + c_len) / ratio;
    REQUIRE(n_comp >= 512u);

    struct guarded_buf q      = guarded_alloc((size_t)c_len * 64u * 512u * 4u);
    struct guarded_buf sinks  = guarded_alloc(64u * 4u);
    struct guarded_buf win    = guarded_alloc(128u * 512u * 4u);
    struct guarded_buf ckv    = guarded_alloc((size_t)c_len * 512u * 4u);
    struct guarded_buf comp   = guarded_alloc((size_t)n_comp * 512u * (comp_bf16 ? 2u : 4u));
    struct guarded_buf ids    = guarded_alloc((size_t)c_len * top_k * 4u);
    struct guarded_buf gath0  = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
    struct guarded_buf gath1  = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
    struct guarded_buf out0   = guarded_alloc((size_t)c_len * 64u * 512u * 4u);
    struct guarded_buf out1   = guarded_alloc((size_t)c_len * 64u * 512u * 4u);

    const uint32_t seed = c_len * 17u + p0 * 3u + ratio * 11u + (comp_bf16 ? 1u : 0u);
    fill_f32_buf(q.view, (size_t)c_len * 64u * 512u, seed + 1u);
    fill_f32_buf(sinks.view, 64u, seed + 2u);
    fill_f32_buf(win.view, 128u * 512u, seed + 3u);
    fill_f32_buf(ckv.view, (size_t)c_len * 512u, seed + 4u);
    if (comp_bf16) fill_bf16_buf(comp.view, (size_t)n_comp * 512u, seed + 5u);
    else           fill_f32_buf(comp.view, (size_t)n_comp * 512u, seed + 5u);
    fill_ids_buf(ids.view, c_len, top_k, base_pos, c_start, ratio, n_comp, seed + 6u);

    setenv("MYDS_PREFILL_ATTN", "auto", 1);
    unsetenv("MYDS_ATTN_HEADS_MODE");
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(n_comp + 2048u);
    REQUIRE(s != NULL);

    /* Reference: MYDS_PREFILL_ATTN_FAST=0 */
    setenv("MYDS_PREFILL_ATTN_FAST", "0", 1);
    REQUIRE(myds_metal_begin_batch());
    if (p0 > 0u) REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
    int ok0 = comp_bf16
        ? myds_v41_gpu_attention_chunk_bf16(out0.view, sinks.view, q.view, ckv.view,
              comp.view, ids.view, gath0.view, c_len, n_comp, top_k, base_pos, c_start, ratio, s)
        : myds_v41_gpu_attention_chunk(out0.view, sinks.view, q.view, ckv.view,
              comp.view, ids.view, gath0.view, c_len, n_comp, top_k, base_pos, c_start, ratio, s);
    REQUIRE(ok0);
    REQUIRE(myds_metal_commit_batch());

    /* Candidate: MYDS_PREFILL_ATTN_FAST=1 */
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    if (p0 > 0u) REQUIRE(myds_v41_gpu_carry_stage(s, win.view, p0));
    int ok1 = comp_bf16
        ? myds_v41_gpu_attention_chunk_bf16(out1.view, sinks.view, q.view, ckv.view,
              comp.view, ids.view, gath1.view, c_len, n_comp, top_k, base_pos, c_start, ratio, s)
        : myds_v41_gpu_attention_chunk(out1.view, sinks.view, q.view, ckv.view,
              comp.view, ids.view, gath1.view, c_len, n_comp, top_k, base_pos, c_start, ratio, s);
    REQUIRE(ok1);
    REQUIRE(myds_metal_commit_batch());

    const uint64_t n_fast = hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512");
    const uint64_t n_win  = hist_count("kernel_myds_attn_gather_window");
    const uint64_t n_blk  = hist_count("kernel_flash_attn_ext_blk");
    const uint64_t n_c512 = hist_count(comp_bf16 ? "kernel_myds_attn_gather_comp_512_bf16"
                                                 : "kernel_myds_attn_gather_comp_512");
    if (expect_fast_matrix) {
        REQUIRE(n_fast == 1u);
        REQUIRE(n_win == 0u);
        REQUIRE(n_blk == 0u);
        REQUIRE(n_c512 == 1u);
    } else {
        REQUIRE(n_fast == 0u);
        REQUIRE(n_c512 == 0u);
    }

    const size_t out_bytes = (size_t)c_len * 64u * 512u * 4u;
    REQUIRE(memcmp(myds_metal_buffer_data(out0.view), myds_metal_buffer_data(out1.view), out_bytes) == 0);

    printf("PASS 0-ULP c_len=%4u p0=%6u ratio=%u %s fast=%d (maskless_slab=%" PRIu64 " win=%" PRIu64 " blk=%" PRIu64 " c512=%" PRIu64 ")\n",
           c_len, p0, ratio, comp_bf16 ? "bf16" : "f32 ", (int)expect_fast_matrix,
           n_fast, n_win, n_blk, n_c512);

    myds_v41_gpu_scratch_free(s);
    guarded_free(out1); guarded_free(out0); guarded_free(gath1); guarded_free(gath0);
    guarded_free(ids); guarded_free(comp); guarded_free(ckv); guarded_free(win);
    guarded_free(sinks); guarded_free(q);
}

/* 3. Cross-layer reuse (6-layer group, L2 leader + 5 followers) + invalidation tests. */
static void test_cross_layer_reuse_and_invalidation(void) {
    const uint32_t c_len = 304u;
    const uint32_t base_pos = 16384u;
    const uint32_t c_start = 0u;
    const uint32_t ratio = 1u;
    const uint32_t top_k = 512u;
    const uint32_t n_comp = base_pos + c_len;
    const int n_layers = 6;

    struct guarded_buf comp  = guarded_alloc((size_t)n_comp * 512u * 2u);
    struct guarded_buf ids   = guarded_alloc((size_t)c_len * top_k * 4u);
    struct guarded_buf gath0 = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
    struct guarded_buf gath1 = guarded_alloc((size_t)c_len * 640u * 512u * 2u);
    fill_bf16_buf(comp.view, (size_t)n_comp * 512u, 701u);
    fill_ids_buf(ids.view, c_len, top_k, base_pos, c_start, ratio, n_comp, 702u);

    struct guarded_buf q[6], sinks[6], win[6], ckv[6], out0[6], out1[6];
    for (int l = 0; l < n_layers; ++l) {
        q[l]     = guarded_alloc((size_t)c_len * 64u * 512u * 4u);
        sinks[l] = guarded_alloc(64u * 4u);
        win[l]   = guarded_alloc(128u * 512u * 4u);
        ckv[l]   = guarded_alloc((size_t)c_len * 512u * 4u);
        out0[l]  = guarded_alloc((size_t)c_len * 64u * 512u * 4u);
        out1[l]  = guarded_alloc((size_t)c_len * 64u * 512u * 4u);
        fill_f32_buf(q[l].view, (size_t)c_len * 64u * 512u, 800u + (uint32_t)l * 10u);
        fill_f32_buf(sinks[l].view, 64u, 801u + (uint32_t)l * 10u);
        fill_f32_buf(win[l].view, 128u * 512u, 802u + (uint32_t)l * 10u);
        fill_f32_buf(ckv[l].view, (size_t)c_len * 512u, 803u + (uint32_t)l * 10u);
    }

    setenv("MYDS_PREFILL_ATTN", "auto", 1);
    unsetenv("MYDS_ATTN_HEADS_MODE");
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(n_comp + 2048u);
    REQUIRE(s != NULL);

    /* Reference FAST=0 across all 6 layers */
    setenv("MYDS_PREFILL_ATTN_FAST", "0", 1);
    REQUIRE(myds_metal_begin_batch());
    for (int l = 0; l < n_layers; ++l) {
        REQUIRE(myds_v41_gpu_carry_stage(s, win[l].view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out0[l].view, sinks[l].view, q[l].view,
                ckv[l].view, comp.view, ids.view, gath0.view, c_len, n_comp, top_k,
                base_pos, c_start, ratio, s));
    }
    REQUIRE(myds_metal_commit_batch());

    /* Candidate FAST=1 across all 6 layers */
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    for (int l = 0; l < n_layers; ++l) {
        REQUIRE(myds_v41_gpu_carry_stage(s, win[l].view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1[l].view, sinks[l].view, q[l].view,
                ckv[l].view, comp.view, ids.view, gath1.view, c_len, n_comp, top_k,
                base_pos, c_start, ratio, s));
    }
    REQUIRE(myds_metal_commit_batch());

    REQUIRE(hist_count("kernel_myds_attn_gather_comp_512_bf16") == 1u);
    REQUIRE(hist_count("kernel_myds_attn_gather_window") == 0u);
    REQUIRE(hist_count("kernel_flash_attn_ext_blk") == 0u);
    REQUIRE(hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512") == (uint64_t)n_layers);

    const size_t out_bytes = (size_t)c_len * 64u * 512u * 4u;
    for (int l = 0; l < n_layers; ++l) {
        REQUIRE(memcmp(myds_metal_buffer_data(out0[l].view),
                       myds_metal_buffer_data(out1[l].view), out_bytes) == 0);
    }
    printf("PASS cross_layer_reuse_6layers c_len=%u gather_comp_dispatches=1/%d 0-ULP=exact\n",
           c_len, n_layers);

    /* Invalidation test A: comp_kv mutation via myds_v41_gpu_pack_bf16 */
    struct guarded_buf new_row = guarded_alloc(512u * 4u);
    fill_f32_buf(new_row.view, 512u, 9999u);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_pack_bf16(comp.view, 0u, new_row.view, 0u, 512u));
    REQUIRE(myds_metal_commit_batch());

    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win[0].view, base_pos));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1[0].view, sinks[0].view, q[0].view,
            ckv[0].view, comp.view, ids.view, gath1.view, c_len, n_comp, top_k,
            base_pos, c_start, ratio, s));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(hist_count("kernel_myds_attn_gather_comp_512_bf16") == 1u);

    setenv("MYDS_PREFILL_ATTN_FAST", "0", 1);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win[0].view, base_pos));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out0[0].view, sinks[0].view, q[0].view,
            ckv[0].view, comp.view, ids.view, gath0.view, c_len, n_comp, top_k,
            base_pos, c_start, ratio, s));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(memcmp(myds_metal_buffer_data(out0[0].view),
                   myds_metal_buffer_data(out1[0].view), out_bytes) == 0);
    printf("PASS invalidation_pack_bf16 gather_retriggered=1 0-ULP=exact\n");

    /* Invalidation test B: topk_causal invalidates gathered cache */
    setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win[0].view, base_pos));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1[0].view, sinks[0].view, q[0].view,
            ckv[0].view, comp.view, ids.view, gath1.view, c_len, n_comp, top_k,
            base_pos, c_start, ratio, s));
    REQUIRE(myds_metal_commit_batch());

    struct guarded_buf dummy_scores = guarded_alloc(1024u * 4u);
    struct guarded_buf dummy_ids    = guarded_alloc(512u * 4u);
    fill_f32_buf(dummy_scores.view, 1024u, 1234u);
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_topk_causal(dummy_ids.view, dummy_scores.view, 1024u, 512u, 1u, 1024u, 1u, s));
    REQUIRE(myds_metal_commit_batch());

    hist_reset();
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win[0].view, base_pos));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1[0].view, sinks[0].view, q[0].view,
            ckv[0].view, comp.view, ids.view, gath1.view, c_len, n_comp, top_k,
            base_pos, c_start, ratio, s));
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(hist_count("kernel_myds_attn_gather_comp_512_bf16") == 1u);
    printf("PASS invalidation_topk_causal gather_retriggered=1\n");

    /* Short c_len=16 (vec path) cross-layer reuse + 0-ULP */
    {
        const uint32_t sc = 16u;
        struct guarded_buf sq0 = guarded_alloc((size_t)sc * 64u * 512u * 4u);
        struct guarded_buf sq1 = guarded_alloc((size_t)sc * 64u * 512u * 4u);
        struct guarded_buf sck0 = guarded_alloc((size_t)sc * 512u * 4u);
        struct guarded_buf sck1 = guarded_alloc((size_t)sc * 512u * 512u * 4u);
        struct guarded_buf sids = guarded_alloc((size_t)sc * top_k * 4u);
        struct guarded_buf sout0_a = guarded_alloc((size_t)sc * 64u * 512u * 4u);
        struct guarded_buf sout0_b = guarded_alloc((size_t)sc * 64u * 512u * 4u);
        struct guarded_buf sout1_a = guarded_alloc((size_t)sc * 64u * 512u * 4u);
        struct guarded_buf sout1_b = guarded_alloc((size_t)sc * 64u * 512u * 4u);
        fill_f32_buf(sq0.view, (size_t)sc * 64u * 512u, 4001u);
        fill_f32_buf(sq1.view, (size_t)sc * 64u * 512u, 4002u);
        fill_f32_buf(sck0.view, (size_t)sc * 512u, 4003u);
        fill_f32_buf(sck1.view, (size_t)sc * 512u, 4004u);
        fill_ids_buf(sids.view, sc, top_k, base_pos, 0u, ratio, n_comp, 4005u);

        setenv("MYDS_PREFILL_ATTN_FAST", "0", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win[0].view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(sout0_a.view, sinks[0].view, sq0.view, sck0.view,
                comp.view, sids.view, gath0.view, sc, n_comp, top_k, base_pos, 0u, ratio, s));
        REQUIRE(myds_v41_gpu_carry_stage(s, win[1].view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(sout0_b.view, sinks[1].view, sq1.view, sck1.view,
                comp.view, sids.view, gath0.view, sc, n_comp, top_k, base_pos, 0u, ratio, s));
        REQUIRE(myds_metal_commit_batch());

        setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
        hist_reset();
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win[0].view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(sout1_a.view, sinks[0].view, sq0.view, sck0.view,
                comp.view, sids.view, gath1.view, sc, n_comp, top_k, base_pos, 0u, ratio, s));
        REQUIRE(myds_v41_gpu_carry_stage(s, win[1].view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(sout1_b.view, sinks[1].view, sq1.view, sck1.view,
                comp.view, sids.view, gath1.view, sc, n_comp, top_k, base_pos, 0u, ratio, s));
        REQUIRE(myds_metal_commit_batch());

        REQUIRE(hist_count("kernel_myds_attn_gather_comp_bf16") == 1u);
        REQUIRE(memcmp(myds_metal_buffer_data(sout0_a.view), myds_metal_buffer_data(sout1_a.view), (size_t)sc*64u*512u*4u) == 0);
        REQUIRE(memcmp(myds_metal_buffer_data(sout0_b.view), myds_metal_buffer_data(sout1_b.view), (size_t)sc*64u*512u*4u) == 0);
        printf("PASS vec_path_cross_layer_reuse c_len=%u gather_comp_dispatches=1/2 0-ULP=exact\n", sc);

        guarded_free(sout1_b); guarded_free(sout1_a); guarded_free(sout0_b); guarded_free(sout0_a);
        guarded_free(sids); guarded_free(sck1); guarded_free(sck0); guarded_free(sq1); guarded_free(sq0);
    }

    guarded_free(dummy_ids); guarded_free(dummy_scores); guarded_free(new_row);
    myds_v41_gpu_scratch_free(s);
    for (int l = n_layers - 1; l >= 0; --l) {
        guarded_free(out1[l]); guarded_free(out0[l]); guarded_free(ckv[l]);
        guarded_free(win[l]); guarded_free(sinks[l]); guarded_free(q[l]);
    }
    guarded_free(gath1); guarded_free(gath0); guarded_free(ids); guarded_free(comp);
}

/* 4. Warm-append 128K+2K (base_pos=126976, 2048+122 multi-chunk + carry roll +
 *    kvdisk memcpy restore -> L0/L1 invalidation + cross-layer follower reuse). */
static void test_warm_append_128k_plus_2k(void) {
    const uint32_t base_pos = 126976u;
    const uint32_t c0 = 2048u, c1 = 122u;
    const uint32_t total_len = c0 + c1; /* 2170 */
    const uint32_t top_k = 512u;

    for (int ri = 0; ri < 2; ++ri) {
        const uint32_t ratio = (ri == 0) ? 2u : 1u;
        const uint32_t n_comp0 = (base_pos + c0) / ratio;
        const uint32_t n_comp1 = (base_pos + total_len) / ratio;

        struct guarded_buf comp  = guarded_alloc((size_t)n_comp1 * 512u * 2u);
        struct guarded_buf ids0  = guarded_alloc((size_t)c0 * top_k * 4u);
        struct guarded_buf ids1  = guarded_alloc((size_t)c1 * top_k * 4u);
        struct guarded_buf win   = guarded_alloc(128u * 512u * 4u);
        struct guarded_buf q0    = guarded_alloc((size_t)c0 * 64u * 512u * 4u);
        struct guarded_buf q1    = guarded_alloc((size_t)c1 * 64u * 512u * 4u);
        struct guarded_buf ckv0  = guarded_alloc((size_t)c0 * 512u * 4u);
        struct guarded_buf ckv1  = guarded_alloc((size_t)c1 * 512u * 4u);
        struct guarded_buf sinks = guarded_alloc(64u * 4u);
        struct guarded_buf gath0 = guarded_alloc((size_t)c0 * 640u * 512u * 2u);
        struct guarded_buf gath1 = guarded_alloc((size_t)c0 * 640u * 512u * 2u);
        struct guarded_buf out0_a = guarded_alloc((size_t)c0 * 64u * 512u * 4u);
        struct guarded_buf out0_b = guarded_alloc((size_t)c1 * 64u * 512u * 4u);
        struct guarded_buf out1_a = guarded_alloc((size_t)c0 * 64u * 512u * 4u);
        struct guarded_buf out1_b = guarded_alloc((size_t)c1 * 64u * 512u * 4u);

        const uint32_t seed = 5000u + ratio * 100u;
        fill_bf16_buf(comp.view, (size_t)n_comp1 * 512u, seed + 1u);
        fill_ids_buf(ids0.view, c0, top_k, base_pos, 0u, ratio, n_comp0, seed + 2u);
        fill_ids_buf(ids1.view, c1, top_k, base_pos, c0, ratio, n_comp1, seed + 3u);
        fill_f32_buf(win.view, 128u * 512u, seed + 4u);
        fill_f32_buf(q0.view, (size_t)c0 * 64u * 512u, seed + 5u);
        fill_f32_buf(q1.view, (size_t)c1 * 64u * 512u, seed + 6u);
        fill_f32_buf(ckv0.view, (size_t)c0 * 512u, seed + 7u);
        fill_f32_buf(ckv1.view, (size_t)c1 * 512u, seed + 8u);
        fill_f32_buf(sinks.view, 64u, seed + 9u);

        setenv("MYDS_PREFILL_ATTN", "auto", 1);
        unsetenv("MYDS_ATTN_HEADS_MODE");
        myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(n_comp1 + 2048u);
        REQUIRE(s != NULL);

        /* Reference FAST=0: carry_stage -> chunk0 (2048) -> carry roll -> chunk1 (122) */
        setenv("MYDS_PREFILL_ATTN_FAST", "0", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out0_a.view, sinks.view, q0.view, ckv0.view,
                comp.view, ids0.view, gath0.view, c0, n_comp0, top_k, base_pos, 0u, ratio, s));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out0_b.view, sinks.view, q1.view, ckv1.view,
                comp.view, ids1.view, gath0.view, c1, n_comp1, top_k, base_pos, c0, ratio, s));
        REQUIRE(myds_metal_commit_batch());

        /* Candidate FAST=1: same two chunks with direct slab + carry roll */
        setenv("MYDS_PREFILL_ATTN_FAST", "1", 1);
        hist_reset();
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, base_pos));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1_a.view, sinks.view, q0.view, ckv0.view,
                comp.view, ids0.view, gath1.view, c0, n_comp0, top_k, base_pos, 0u, ratio, s));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1_b.view, sinks.view, q1.view, ckv1.view,
                comp.view, ids1.view, gath1.view, c1, n_comp1, top_k, base_pos, c0, ratio, s));
        REQUIRE(myds_metal_commit_batch());

        REQUIRE(hist_count("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512") == 2u);
        REQUIRE(hist_count("kernel_myds_attn_gather_window") == 0u);
        REQUIRE(hist_count("kernel_flash_attn_ext_blk") == 0u);
        REQUIRE(memcmp(myds_metal_buffer_data(out0_a.view), myds_metal_buffer_data(out1_a.view),
                       (size_t)c0 * 64u * 512u * 4u) == 0);
        REQUIRE(memcmp(myds_metal_buffer_data(out0_b.view), myds_metal_buffer_data(out1_b.view),
                       (size_t)c1 * 64u * 512u * 4u) == 0);

        /* Simulate kvdisk memcpy restore (CPU overwrites comp_kv without GPU write),
         * followed by Layer 0 (n_comp=0, window-only) which MUST invalidate gathered_cache! */
        fill_bf16_buf(comp.view, (size_t)n_comp1 * 512u, seed + 99u);
        hist_reset();
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, base_pos));
        /* Layer 0 (n_comp=0) invalidates gathered cache */
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1_b.view, sinks.view, q1.view, ckv1.view,
                NULL, NULL, gath1.view, c1, 0u, top_k, base_pos, c0, 0u, s));
        /* Layer 2 (same c1/c0 params as last cached chunk) MUST re-gather! */
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, base_pos + c0));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out1_b.view, sinks.view, q1.view, ckv1.view,
                comp.view, ids1.view, gath1.view, c1, n_comp1, top_k, base_pos, c0, ratio, s));
        REQUIRE(myds_metal_commit_batch());
        REQUIRE(hist_count("kernel_myds_attn_gather_comp_512_bf16") == 1u);

        setenv("MYDS_PREFILL_ATTN_FAST", "0", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, win.view, base_pos + c0));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(out0_b.view, sinks.view, q1.view, ckv1.view,
                comp.view, ids1.view, gath0.view, c1, n_comp1, top_k, base_pos, c0, ratio, s));
        REQUIRE(myds_metal_commit_batch());
        REQUIRE(memcmp(myds_metal_buffer_data(out0_b.view), myds_metal_buffer_data(out1_b.view),
                       (size_t)c1 * 64u * 512u * 4u) == 0);

        printf("PASS warm_append_128k+2k base_pos=%u ratio=%u (2048+122 carry_roll + L0_kvdisk_invalidation) 0-ULP=exact\n",
               base_pos, ratio);

        myds_v41_gpu_scratch_free(s);
        guarded_free(out1_b); guarded_free(out1_a); guarded_free(out0_b); guarded_free(out0_a);
        guarded_free(gath1); guarded_free(gath0); guarded_free(sinks);
        guarded_free(ckv1); guarded_free(ckv0); guarded_free(q1); guarded_free(q0);
        guarded_free(win); guarded_free(ids1); guarded_free(ids0); guarded_free(comp);
    }
}

/* 5. Hardware GPU microbenchmark on local Apple Silicon. */
static double bench_mode(myds_v41_gpu_scratch_t *s, myds_metal_buffer_t *out,
                         myds_metal_buffer_t *sinks, myds_metal_buffer_t *q,
                         myds_metal_buffer_t *ckv, myds_metal_buffer_t *comp,
                         myds_metal_buffer_t *ids, myds_metal_buffer_t *gath,
                         myds_metal_buffer_t *win, uint32_t c_len, uint32_t n_comp,
                         uint32_t base_pos, uint32_t ratio, int fast, int iters, int repeat_layers) {
    setenv("MYDS_PREFILL_ATTN_FAST", fast ? "1" : "0", 1);
    /* Warmup */
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_carry_stage(s, win, base_pos));
    REQUIRE(myds_v41_gpu_attention_chunk_bf16(out, sinks, q, ckv, comp, ids, gath,
            c_len, n_comp, 512u, base_pos, 0u, ratio, s));
    REQUIRE(myds_metal_commit_batch());

    double best_ms = 1e9;
    for (int it = 0; it < iters; ++it) {
        gathered_cache_invalidate(s);
        double t0 = myds_metal_host_time();
        REQUIRE(myds_metal_begin_batch());
        for (int l = 0; l < repeat_layers; ++l) {
            REQUIRE(myds_v41_gpu_carry_stage(s, win, base_pos));
            REQUIRE(myds_v41_gpu_attention_chunk_bf16(out, sinks, q, ckv, comp, ids, gath,
                    c_len, n_comp, 512u, base_pos, 0u, ratio, s));
        }
        REQUIRE(myds_metal_commit_batch());
        double dt_ms = (myds_metal_host_time() - t0) * 1000.0;
        if (dt_ms < best_ms) best_ms = dt_ms;
    }
    return best_ms;
}

static void run_microbench(void) {
    setenv("MYDS_PREFILL_ATTN", "auto", 1);
    unsetenv("MYDS_ATTN_HEADS_MODE");
    const uint32_t max_c = 2048u;
    const uint32_t base_pos = 16384u;
    const uint32_t n_comp = base_pos + max_c;

    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(n_comp + 2048u);
    REQUIRE(s != NULL);
    myds_metal_buffer_t *q     = myds_metal_buffer_alloc((size_t)max_c * 64u * 512u * 4u);
    myds_metal_buffer_t *sinks = myds_metal_buffer_alloc(64u * 4u);
    myds_metal_buffer_t *win   = myds_metal_buffer_alloc(128u * 512u * 4u);
    myds_metal_buffer_t *ckv   = myds_metal_buffer_alloc((size_t)max_c * 512u * 4u);
    myds_metal_buffer_t *comp  = myds_metal_buffer_alloc((size_t)n_comp * 512u * 2u);
    myds_metal_buffer_t *ids   = myds_metal_buffer_alloc((size_t)max_c * 512u * 4u);
    myds_metal_buffer_t *gath  = myds_metal_buffer_alloc((size_t)max_c * 640u * 512u * 2u);
    myds_metal_buffer_t *out   = myds_metal_buffer_alloc((size_t)max_c * 64u * 512u * 4u);
    fill_f32_buf(q, (size_t)max_c * 64u * 512u, 11u);
    fill_f32_buf(sinks, 64u, 12u);
    fill_f32_buf(win, 128u * 512u, 13u);
    fill_f32_buf(ckv, (size_t)max_c * 512u, 14u);
    fill_bf16_buf(comp, (size_t)n_comp * 512u, 15u);
    fill_ids_buf(ids, max_c, 512u, base_pos, 0u, 1u, n_comp, 16u);

    double ms0_2k = bench_mode(s, out, sinks, q, ckv, comp, ids, gath, win, 2048u, n_comp, base_pos, 1u, 0, 5, 1);
    double ms1_2k = bench_mode(s, out, sinks, q, ckv, comp, ids, gath, win, 2048u, n_comp, base_pos, 1u, 1, 5, 1);
    printf("BENCH c_len=2048 1-layer  FAST=0: %.2f ms  FAST=1: %.2f ms  speedup=%.2fx\n",
           ms0_2k, ms1_2k, ms0_2k / ms1_2k);

    double ms0_6l = bench_mode(s, out, sinks, q, ckv, comp, ids, gath, win, 304u, base_pos + 304u, base_pos, 1u, 0, 5, 6);
    double ms1_6l = bench_mode(s, out, sinks, q, ckv, comp, ids, gath, win, 304u, base_pos + 304u, base_pos, 1u, 1, 5, 6);
    printf("BENCH c_len=304  6-layers FAST=0: %.2f ms  FAST=1: %.2f ms  speedup=%.2fx\n",
           ms0_6l, ms1_6l, ms0_6l / ms1_6l);

    myds_metal_buffer_free(out); myds_metal_buffer_free(gath); myds_metal_buffer_free(ids);
    myds_metal_buffer_free(comp); myds_metal_buffer_free(ckv); myds_metal_buffer_free(win);
    myds_metal_buffer_free(sinks); myds_metal_buffer_free(q);
    myds_v41_gpu_scratch_free(s);
}

int main(void) {
    g_hist_enabled = 1;
    REQUIRE(myds_metal_init("metal"));
    test_pso_geometry();

    /* Eligible fast-matrix shapes (all_visible=1, c_len>=32, BF16 & F32, ratio=1 & 2) */
    test_single_chunk_0ulp(  32u,  1023u, 0u, 2u, true,  true);
    test_single_chunk_0ulp(  33u,  1024u, 0u, 2u, true,  true);
    test_single_chunk_0ulp(  63u,  1536u, 0u, 2u, false, true);
    test_single_chunk_0ulp( 127u,   511u, 0u, 1u, true,  true);
    test_single_chunk_0ulp( 128u,   512u, 0u, 1u, false, true);
    test_single_chunk_0ulp( 294u,  2048u, 0u, 2u, true,  true);
    test_single_chunk_0ulp( 304u, 16384u, 0u, 1u, true,  true);
    test_single_chunk_0ulp( 512u,  4096u, 0u, 2u, true,  true);
    test_single_chunk_0ulp(2048u,  2048u, 0u, 2u, true,  true);

    /* Boundary / non-eligible shapes (must stay on masked path and remain 0-ULP) */
    test_single_chunk_0ulp(  31u,  2048u, 0u, 2u, true,  false); /* c_len < 32 -> vec */
    test_single_chunk_0ulp( 128u,  1022u, 0u, 2u, true,  false); /* (1022+1)/2 = 511 < 512 */
    test_single_chunk_0ulp( 128u,   510u, 0u, 1u, true,  false); /* (510+1)/1 = 511 < 512 */

    /* Cross-layer reuse & invalidation */
    test_cross_layer_reuse_and_invalidation();

    /* Warm-append 128K+2K (pos0=126976, 2048+122 carry roll + L0 kvdisk invalidation) */
    test_warm_append_128k_plus_2k();

    /* Local M4 microbenchmark */
    g_hist_enabled = 0;
    run_microbench();

    printf("ALL [P-NEW-3] PREFILL ATTN FAST TESTS PASSED (STRICT 0-ULP)\n");
    myds_metal_cleanup();
    return 0;
}
