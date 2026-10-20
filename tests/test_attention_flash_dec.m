/* Standalone Component Unit Test and Benchmark for D1 Attention Flash-Decoding.
 * Verifies 0-ULP bit-exact equivalence between:
 *   - Golden reference: kernel_myds_attention_register_rope_fuse_f16_512 (single-TG 14-NSG)
 *   - D1 candidate: kernel_myds_attention_flash_dec_split_f16_512 (4 TGs per head, 8 NSG)
 *                 + kernel_myds_attention_flash_dec_reduce_rope_f16_512 (1 TG, 4 NSG)
 *
 * Microbenchmarks GPU kernel execution time across context lengths.
 */
#define main original_attention_packed_main
#include "test_attention_packed.m"
#undef main

static int encode_baseline(struct fixture *f, myds_v41_gpu_scratch_t *s, unsigned out_idx, uint32_t pos) {
    struct flash_vec_args a = vector_args(f->keys);
    myds_metal_constant_t c[10];
    constants(f->keys, c);
    const myds_metal_buffer_t *b[] = {f->q, s->kv_f16, s->kv_f16, s->mask, f->sinks, s->pad, f->out[out_idx].view};
    struct attention_rope_fuse_args fused = {.attention = a, .pos = pos, .inverse = 1u};
    memcpy(fused.frequencies, rope_frequencies[0], sizeof(fused.frequencies));
    return specialized("kernel_myds_attention_register_rope_fuse_f16_512",
                       &fused, sizeof(fused), b, 7, 64, 1, 1, 448, 1, 1, 22400, c, 10);
}

static int encode_flash_dec(struct fixture *f, myds_v41_gpu_scratch_t *s, unsigned out_idx, uint32_t pos) {
    struct flash_vec_args a = vector_args(f->keys);
    myds_metal_constant_t c[10];
    constants(f->keys, c);

    // Pass 1: Grid (64, 4, 1) = 256 threadgroups across 76 cores; TG = 256 threads (8 simdgroups)
    const myds_metal_buffer_t *b_split[] = {f->q, s->kv_f16, s->kv_f16, s->mask, f->sinks, s->pad, s->partial};
    if (!specialized("kernel_myds_attention_flash_dec_split_f16_512",
                     &a, sizeof(a), b_split, 7, 64, 4, 1, 256, 1, 1, 2048, c, 10)) {
        return 0;
    }

    // Intra-encoder buffer barrier
    if (!myds_metal_get_auto_barrier()) {
        myds_metal_barrier_buffer(s->partial);
    }

    // Pass 2: Grid (64, 1, 1); TG = 128 threads (4 simdgroups, one per quadrant)
    struct attention_rope_fuse_args fused = {.attention = a, .pos = pos, .inverse = 1u};
    memcpy(fused.frequencies, rope_frequencies[0], sizeof(fused.frequencies));
    const myds_metal_buffer_t *b_reduce[] = {s->partial, f->out[out_idx].view};
    if (!specialized("kernel_myds_attention_flash_dec_reduce_rope_f16_512",
                     &fused, sizeof(fused), b_reduce, 2, 64, 1, 1, 128, 1, 1, 0, c, 10)) {
        return 0;
    }
    return 1;
}

static void test_flash_dec_parity(uint32_t keys, uint32_t start, uint32_t mode, uint32_t pos) {
    struct fixture f;
    fixture_init(&f, keys, start, mode);
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(2048);
    REQUIRE(s);
    prepare_mask(s, &f);

    // Stage KV data into s->kv_f16 and pad
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(call(&f, s, 0, 1));
    REQUIRE(myds_metal_commit_batch());

    // Run baseline into out[0]
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(encode_baseline(&f, s, 0, pos));
    REQUIRE(myds_metal_commit_batch());

    // Run D1 flash-dec into out[1]
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(encode_flash_dec(&f, s, 1, pos));
    REQUIRE(myds_metal_commit_batch());

    // Verify bit-exact equality (0-ULP)
    compare(&f, 1);

    printf("  [PASS 0-ULP] keys=%-4u start=%-3u mode=%u pos=%-4u -> exact bit match (memcmp==0)\n",
           keys, start, mode, pos);

    myds_v41_gpu_scratch_free(s);
    fixture_free(&f);
}

static void bench_flash_dec(uint32_t keys, uint32_t pos, unsigned reps) {
    struct fixture f;
    fixture_init(&f, keys, 0, 0);
    myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(2048);
    REQUIRE(s);
    prepare_mask(s, &f);

    // Stage KV data into s->kv_f16 and pad
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(call(&f, s, 0, 1));
    REQUIRE(myds_metal_commit_batch());

    // Warmup
    REQUIRE(myds_metal_begin_batch());
    REQUIRE(encode_baseline(&f, s, 0, pos));
    REQUIRE(myds_metal_commit_batch());

    REQUIRE(myds_metal_begin_batch());
    REQUIRE(encode_flash_dec(&f, s, 1, pos));
    REQUIRE(myds_metal_commit_batch());
    compare(&f, 1);

    // Benchmark Baseline
    REQUIRE(myds_metal_begin_batch());
    id<MTLCommandBuffer> cb_base = g_batch_cb;
    for (unsigned r = 0; r < reps; r++) {
        REQUIRE(encode_baseline(&f, s, 0, pos));
        myds_metal_barrier();
    }
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(cb_base.status == MTLCommandBufferStatusCompleted);
    double us_base = (cb_base.GPUEndTime - cb_base.GPUStartTime) * 1e6 / reps;

    // Benchmark D1 Flash-Dec
    REQUIRE(myds_metal_begin_batch());
    id<MTLCommandBuffer> cb_d1 = g_batch_cb;
    for (unsigned r = 0; r < reps; r++) {
        REQUIRE(encode_flash_dec(&f, s, 1, pos));
        myds_metal_barrier();
    }
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(cb_d1.status == MTLCommandBufferStatusCompleted);
    double us_d1 = (cb_d1.GPUEndTime - cb_d1.GPUStartTime) * 1e6 / reps;

    compare(&f, 1);

    double delta = us_base - us_d1;
    double speedup_pct = (delta / us_base) * 100.0;

    printf("  [BENCH] keys=%-4u | Baseline (1-TG 14-NSG): %5.2f µs | D1 Flash-Dec (4-TG): %5.2f µs | Δ: %+5.2f µs (%+5.1f%%) [0-ULP PASS]\n",
           keys, us_base, us_d1, delta, speedup_pct);

    myds_v41_gpu_scratch_free(s);
    fixture_free(&f);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    setenv("MYDS_ATTN_VEC_HEAD_FIRST", "0", 1);
    setenv("MYDS_DEBUG_DISPATCH", "0", 1);
    setenv("MYDS_METAL_CONCURRENT", "0", 1);
    g_hist_enabled = 0;

    REQUIRE(myds_metal_init("bin"));
    init_rope();

    printf("======================================================================\n");
    printf("  D1 Decode Attention Flash-Decoding 0-ULP Parity & Benchmark\n");
    printf("  Target Device: %s\n", g_device.name.UTF8String);
    printf("======================================================================\n\n");

    printf("--- Phase 1: 0-ULP Bit-Exact Functional Equivalence Tests ---\n");
    const uint32_t test_keys[] = {1, 16, 31, 32, 64, 96, 127, 128, 192, 256, 384, 512, 576, 640};
    const uint32_t test_positions[] = {0, 1, 127, 847, 1536};
    const uint32_t test_modes[] = {0, 1, 3, 4}; // Normal, pattern mask, causal prefix, sinks only

    for (size_t k = 0; k < sizeof(test_keys)/sizeof(test_keys[0]); k++) {
        uint32_t keys = test_keys[k];
        uint32_t pos = test_positions[k % (sizeof(test_positions)/sizeof(test_positions[0]))];
        uint32_t mode = test_modes[k % (sizeof(test_modes)/sizeof(test_modes[0]))];
        test_flash_dec_parity(keys, 0, mode, pos);
    }

    printf("\n--- Phase 2: GPU Kernel Latency Benchmark (50 repetitions) ---\n");
    for (size_t k = 0; k < sizeof(test_keys)/sizeof(test_keys[0]); k++) {
        uint32_t keys = test_keys[k];
        bench_flash_dec(keys, 127, 50);
    }

    printf("\n======================================================================\n");
    printf("  D1 Flash-Decoding Verification Complete: 0-ULP Parity Confirmed!\n");
    printf("======================================================================\n");

    myds_metal_cleanup();
    return 0;
}
