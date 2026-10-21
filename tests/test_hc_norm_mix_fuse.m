/* Standalone Unit Test and Microbenchmark for D2: RMSNorm -> GEMV Pre-fusion.
 * Verifies 0-ULP bit-exact equivalence between:
 *   - Baseline: v41_rmsnorm (1 TG, 1024 threads) + buffer barrier + v41_gemv (12 TGs, 256 threads)
 *   - Candidate 1: kernel_dsv4_hc_rms_norm_mix_f16 (12 TGs, 256 threads)
 *   - Candidate 2: kernel_dsv4_hc_rms_norm_mix_f16_cluster2 (6 TGs, 512 threads)
 *
 * Microbenchmarks latency on MacStudio M2 Ultra (76 cores).
 */
#include "myds_metal.m"
#include <mach/mach_time.h>
#include <math.h>

#define REQUIRE(c) do { \
    if (!(c)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
        exit(1); \
    } \
} while (0)

static double time_in_us(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)(end - start) * (double)tb.numer / (double)tb.denom / 1000.0;
}

static void fill_f32(myds_metal_buffer_t *b, unsigned seed) {
    float *p = (float *)myds_metal_buffer_data(b);
    size_t count = b->size / sizeof(float);
    for (size_t i = 0; i < count; i++) {
        // Diverse values in range [-10.0, 10.0]
        int v = (int)((i * 17u + seed * 31u) % 2001u) - 1000;
        p[i] = (float)v / 100.0f;
    }
}

static void fill_f16(myds_metal_buffer_t *b, unsigned seed) {
    uint16_t *p = (uint16_t *)myds_metal_buffer_data(b);
    size_t count = b->size / sizeof(uint16_t);
    for (size_t i = 0; i < count; i++) {
        // Generate valid half-precision floats
        int v = (int)((i * 23u + seed * 47u) % 401u) - 200;
        float f = (float)v / 100.0f;
        // Convert float to half bits
        _Float16 h = (_Float16)f;
        memcpy(&p[i], &h, sizeof(uint16_t));
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *shader_dir = argc > 1 ? argv[1] : "bin";
        printf("=== D2: RMSNorm -> GEMV Pre-fusion Unit Test & Benchmark ===\n");
        printf("Loading Metal runtime from '%s'...\n", shader_dir);
        REQUIRE(myds_metal_init(shader_dir));
        REQUIRE(g_initialized && g_device && g_library);
        myds_metal_set_auto_barrier(0);
        myds_metal_set_concurrent(1);

        // Fetch candidate PSOs
        id<MTLFunction> fn1 = [g_library newFunctionWithName:@"kernel_dsv4_hc_rms_norm_mix_f16"];
        REQUIRE(fn1 != nil);
        id<MTLComputePipelineState> pso_cand1 = [g_device newComputePipelineStateWithFunction:fn1 error:nil];
        REQUIRE(pso_cand1 != nil);

        id<MTLFunction> fn2 = [g_library newFunctionWithName:@"kernel_dsv4_hc_rms_norm_mix_f16_cluster2"];
        REQUIRE(fn2 != nil);
        id<MTLComputePipelineState> pso_cand2 = [g_device newComputePipelineStateWithFunction:fn2 error:nil];
        REQUIRE(pso_cand2 != nil);

        const uint32_t dim = 5120u;
        const uint32_t in_dim = 4u * dim; // 20480
        const uint32_t out_dim = 24u;
        const float eps = 1e-5f;

        const size_t res_bytes = (size_t)in_dim * sizeof(float);       // 81,920 bytes
        const size_t fn_bytes = (size_t)out_dim * in_dim * sizeof(uint16_t); // 983,040 bytes
        const size_t out_bytes = (size_t)out_dim * sizeof(float);      // 96 bytes

        myds_metal_buffer_t *buf_res = myds_metal_buffer_alloc(res_bytes);
        myds_metal_buffer_t *buf_fn = myds_metal_buffer_alloc(fn_bytes);
        myds_metal_buffer_t *buf_flat_norm = myds_metal_buffer_alloc(res_bytes);
        myds_metal_buffer_t *buf_mix_base = myds_metal_buffer_alloc(out_bytes);
        myds_metal_buffer_t *buf_mix_cand1 = myds_metal_buffer_alloc(out_bytes);
        myds_metal_buffer_t *buf_mix_cand2 = myds_metal_buffer_alloc(out_bytes);

        REQUIRE(buf_res && buf_fn && buf_flat_norm && buf_mix_base && buf_mix_cand1 && buf_mix_cand2);

        // -------------------------------------------------------------
        // Phase 1: 0-ULP Parity Suite (10 random seeds)
        // -------------------------------------------------------------
        printf("\n--- Running 0-ULP Parity Suite across 10 random seeds ---\n");
        for (unsigned seed = 1; seed <= 10; seed++) {
            fill_f32(buf_res, seed);
            fill_f16(buf_fn, seed + 100);
            memset(myds_metal_buffer_data(buf_mix_base), 0, out_bytes);
            memset(myds_metal_buffer_data(buf_mix_cand1), 0, out_bytes);
            memset(myds_metal_buffer_data(buf_mix_cand2), 0, out_bytes);

            // 1. Run Baseline: v41_rmsnorm -> barrier -> v41_gemv
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_metal_v41_hc_mix_norm_stage(buf_flat_norm, buf_res, dim, eps));
            myds_metal_barrier_buffer(buf_flat_norm);
            REQUIRE(myds_metal_v41_hc_mix_project_stage(buf_mix_base, buf_fn, buf_flat_norm, dim));
            REQUIRE(myds_metal_commit_batch());

            // 2. Run Candidate 1: kernel_dsv4_hc_rms_norm_mix_f16 (12 TGs x 256 threads)
            REQUIRE(myds_metal_begin_batch());
            {
                id<MTLCommandBuffer> cb = g_batch_cb;
                id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
                REQUIRE(enc != nil);
                struct ds4_metal_args_hc_norm_mix args = {
                    .n = (int32_t)in_dim,
                    .out_dim = (int32_t)out_dim,
                    .eps = eps,
                };
                [enc setComputePipelineState:pso_cand1];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_res->mtl_buf offset:buf_res->offset atIndex:1];
                [enc setBuffer:buf_fn->mtl_buf offset:buf_fn->offset atIndex:2];
                [enc setBuffer:buf_mix_cand1->mtl_buf offset:buf_mix_cand1->offset atIndex:3];
                [enc setThreadgroupMemoryLength:(32 + 32 * 2) * sizeof(float) atIndex:0];
                metal_dispatch(enc, "cand1_fused_nr2", MTLSizeMake(12, 1, 1), MTLSizeMake(32, 8, 1));
                v41_finish(cb, enc);
            }
            REQUIRE(myds_metal_commit_batch());

            // 3. Run Candidate 2: kernel_dsv4_hc_rms_norm_mix_f16_cluster2 (6 TGs x 512 threads)
            REQUIRE(myds_metal_begin_batch());
            {
                id<MTLCommandBuffer> cb = g_batch_cb;
                id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
                REQUIRE(enc != nil);
                struct ds4_metal_args_hc_norm_mix args = {
                    .n = (int32_t)in_dim,
                    .out_dim = (int32_t)out_dim,
                    .eps = eps,
                };
                [enc setComputePipelineState:pso_cand2];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_res->mtl_buf offset:buf_res->offset atIndex:1];
                [enc setBuffer:buf_fn->mtl_buf offset:buf_fn->offset atIndex:2];
                [enc setBuffer:buf_mix_cand2->mtl_buf offset:buf_mix_cand2->offset atIndex:3];
                [enc setThreadgroupMemoryLength:(32 + 2 * 2 * 32) * sizeof(float) atIndex:0];
                metal_dispatch(enc, "cand2_fused_cluster2", MTLSizeMake(6, 1, 1), MTLSizeMake(32, 16, 1));
                v41_finish(cb, enc);
            }
            REQUIRE(myds_metal_commit_batch());

            const float *p_base = (const float *)myds_metal_buffer_data(buf_mix_base);
            const float *p_c1 = (const float *)myds_metal_buffer_data(buf_mix_cand1);
            const float *p_c2 = (const float *)myds_metal_buffer_data(buf_mix_cand2);

            int diff_c1 = 0, diff_c2 = 0;
            for (uint32_t i = 0; i < out_dim; i++) {
                uint32_t b_bits, c1_bits, c2_bits;
                memcpy(&b_bits, &p_base[i], 4);
                memcpy(&c1_bits, &p_c1[i], 4);
                memcpy(&c2_bits, &p_c2[i], 4);
                if (b_bits != c1_bits) diff_c1++;
                if (b_bits != c2_bits) diff_c2++;
            }

            if (diff_c1 != 0 || diff_c2 != 0) {
                printf("[FAIL] seed %u: diff_cand1=%d, diff_cand2=%d\n", seed, diff_c1, diff_c2);
                for (uint32_t i = 0; i < out_dim; i++) {
                    uint32_t b_bits, c1_bits, c2_bits;
                    memcpy(&b_bits, &p_base[i], 4);
                    memcpy(&c1_bits, &p_c1[i], 4);
                    memcpy(&c2_bits, &p_c2[i], 4);
                    printf("  [%2u] base=%+.6f (0x%08x) | cand1=%+.6f (0x%08x) %s | cand2=%+.6f (0x%08x) %s\n",
                           i, p_base[i], b_bits,
                           p_c1[i], c1_bits, (b_bits == c1_bits) ? "MATCH" : "DIFF",
                           p_c2[i], c2_bits, (b_bits == c2_bits) ? "MATCH" : "DIFF");
                }
                exit(1);
            }
            printf("  Seed %2u: PASS 0-ULP (memcmp == 0, bit-exact match for Cand1 and Cand2)\n", seed);
        }
        printf(">>> 0-ULP PARITY VERIFICATION PASSED 100%%! <<<\n");

        // -------------------------------------------------------------
        // Phase 2: Microbenchmark (3000 iterations per arm)
        // -------------------------------------------------------------
        const int ITERS = 3000;
        printf("\n--- Microbenchmark: Measuring %d iterations per arm ---\n", ITERS);

        // Warmup
        for (int i = 0; i < 100; i++) {
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_metal_v41_hc_mix_norm_stage(buf_flat_norm, buf_res, dim, eps));
            myds_metal_barrier_buffer(buf_flat_norm);
            REQUIRE(myds_metal_v41_hc_mix_project_stage(buf_mix_base, buf_fn, buf_flat_norm, dim));
            REQUIRE(myds_metal_commit_batch());
        }

        // Arm 0: Baseline (v41_rmsnorm + barrier + v41_gemv)
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++) {
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_metal_v41_hc_mix_norm_stage(buf_flat_norm, buf_res, dim, eps));
            myds_metal_barrier_buffer(buf_flat_norm);
            REQUIRE(myds_metal_v41_hc_mix_project_stage(buf_mix_base, buf_fn, buf_flat_norm, dim));
            REQUIRE(myds_metal_commit_batch());
        }
        uint64_t t1 = mach_absolute_time();
        double base_us = time_in_us(t0, t1) / ITERS;

        // Arm 1: Candidate 1 (kernel_dsv4_hc_rms_norm_mix_f16, grid 12)
        uint64_t t2 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++) {
            REQUIRE(myds_metal_begin_batch());
            id<MTLCommandBuffer> cb = g_batch_cb;
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            struct ds4_metal_args_hc_norm_mix args = {
                .n = (int32_t)in_dim,
                .out_dim = (int32_t)out_dim,
                .eps = eps,
            };
            [enc setComputePipelineState:pso_cand1];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:buf_res->mtl_buf offset:buf_res->offset atIndex:1];
            [enc setBuffer:buf_fn->mtl_buf offset:buf_fn->offset atIndex:2];
            [enc setBuffer:buf_mix_cand1->mtl_buf offset:buf_mix_cand1->offset atIndex:3];
            [enc setThreadgroupMemoryLength:(32 + 32 * 2) * sizeof(float) atIndex:0];
            metal_dispatch(enc, "cand1_fused_nr2", MTLSizeMake(12, 1, 1), MTLSizeMake(32, 8, 1));
            v41_finish(cb, enc);
            REQUIRE(myds_metal_commit_batch());
        }
        uint64_t t3 = mach_absolute_time();
        double cand1_us = time_in_us(t2, t3) / ITERS;

        // Arm 2: Candidate 2 (kernel_dsv4_hc_rms_norm_mix_f16_cluster2, grid 6)
        uint64_t t4 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++) {
            REQUIRE(myds_metal_begin_batch());
            id<MTLCommandBuffer> cb = g_batch_cb;
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            struct ds4_metal_args_hc_norm_mix args = {
                .n = (int32_t)in_dim,
                .out_dim = (int32_t)out_dim,
                .eps = eps,
            };
            [enc setComputePipelineState:pso_cand2];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:buf_res->mtl_buf offset:buf_res->offset atIndex:1];
            [enc setBuffer:buf_fn->mtl_buf offset:buf_fn->offset atIndex:2];
            [enc setBuffer:buf_mix_cand2->mtl_buf offset:buf_mix_cand2->offset atIndex:3];
            [enc setThreadgroupMemoryLength:(32 + 2 * 2 * 32) * sizeof(float) atIndex:0];
            metal_dispatch(enc, "cand2_fused_cluster2", MTLSizeMake(6, 1, 1), MTLSizeMake(32, 16, 1));
            v41_finish(cb, enc);
            REQUIRE(myds_metal_commit_batch());
        }
        uint64_t t5 = mach_absolute_time();
        double cand2_us = time_in_us(t4, t5) / ITERS;

        printf("\n=======================================================\n");
        printf("RESULTS SUMMARY (%d calls averaged):\n", ITERS);
        printf("  Baseline (v41_norm + barrier + v41_gemv) : %6.3f us/call\n", base_us);
        printf("  Candidate 1 (fused NR0=2, grid=12)       : %6.3f us/call  (Delta: %+.3f us, %+.1f%%)\n",
               cand1_us, cand1_us - base_us, (cand1_us - base_us) / base_us * 100.0);
        printf("  Candidate 2 (fused cluster2, grid=6)     : %6.3f us/call  (Delta: %+.3f us, %+.1f%%)\n",
               cand2_us, cand2_us - base_us, (cand2_us - base_us) / base_us * 100.0);
        printf("-------------------------------------------------------\n");
        double best_delta = base_us - fmin(cand1_us, cand2_us);
        printf("  Best savings per call                    :  %.3f us\n", best_delta);
        printf("  Projected savings per token (80 calls)   : %6.1f us (~%.2f ms/token)\n",
               best_delta * 80.0, (best_delta * 80.0) / 1000.0);
        printf("=======================================================\n");

        // -------------------------------------------------------------
        // Phase 3: Pipelined / Batched Benchmark (40 calls per token batch)
        // -------------------------------------------------------------
        const int BATCH_RUNS = 200;
        const int CALLS_PER_BATCH = 40;
        printf("\n--- Pipelined Batch Benchmark: %d runs of %d dispatches per command buffer ---\n",
               BATCH_RUNS, CALLS_PER_BATCH);

        // Warmup
        for (int r = 0; r < 20; r++) {
            REQUIRE(myds_metal_begin_batch());
            for (int c = 0; c < CALLS_PER_BATCH; c++) {
                REQUIRE(myds_metal_v41_hc_mix_norm_stage(buf_flat_norm, buf_res, dim, eps));
                myds_metal_barrier_buffer(buf_flat_norm);
                REQUIRE(myds_metal_v41_hc_mix_project_stage(buf_mix_base, buf_fn, buf_flat_norm, dim));
            }
            REQUIRE(myds_metal_commit_batch());
        }

        // Arm 0: Pipelined Baseline
        uint64_t bt0 = mach_absolute_time();
        for (int r = 0; r < BATCH_RUNS; r++) {
            REQUIRE(myds_metal_begin_batch());
            for (int c = 0; c < CALLS_PER_BATCH; c++) {
                REQUIRE(myds_metal_v41_hc_mix_norm_stage(buf_flat_norm, buf_res, dim, eps));
                myds_metal_barrier_buffer(buf_flat_norm);
                REQUIRE(myds_metal_v41_hc_mix_project_stage(buf_mix_base, buf_fn, buf_flat_norm, dim));
            }
            REQUIRE(myds_metal_commit_batch());
        }
        uint64_t bt1 = mach_absolute_time();
        double pipe_base_us = time_in_us(bt0, bt1) / (BATCH_RUNS * CALLS_PER_BATCH);

        // Arm 1: Pipelined Candidate 1 (fused NR0=2, grid=12)
        uint64_t bt2 = mach_absolute_time();
        for (int r = 0; r < BATCH_RUNS; r++) {
            REQUIRE(myds_metal_begin_batch());
            id<MTLCommandBuffer> cb = g_batch_cb;
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            struct ds4_metal_args_hc_norm_mix args = {
                .n = (int32_t)in_dim,
                .out_dim = (int32_t)out_dim,
                .eps = eps,
            };
            [enc setComputePipelineState:pso_cand1];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:buf_res->mtl_buf offset:buf_res->offset atIndex:1];
            [enc setBuffer:buf_fn->mtl_buf offset:buf_fn->offset atIndex:2];
            [enc setBuffer:buf_mix_cand1->mtl_buf offset:buf_mix_cand1->offset atIndex:3];
            [enc setThreadgroupMemoryLength:(32 + 32 * 2) * sizeof(float) atIndex:0];
            for (int c = 0; c < CALLS_PER_BATCH; c++) {
                metal_dispatch(enc, "cand1_fused_nr2", MTLSizeMake(12, 1, 1), MTLSizeMake(32, 8, 1));
            }
            v41_finish(cb, enc);
            REQUIRE(myds_metal_commit_batch());
        }
        uint64_t bt3 = mach_absolute_time();
        double pipe_cand1_us = time_in_us(bt2, bt3) / (BATCH_RUNS * CALLS_PER_BATCH);

        // Arm 2: Pipelined Candidate 2 (fused cluster2, grid=6)
        uint64_t bt4 = mach_absolute_time();
        for (int r = 0; r < BATCH_RUNS; r++) {
            REQUIRE(myds_metal_begin_batch());
            id<MTLCommandBuffer> cb = g_batch_cb;
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            struct ds4_metal_args_hc_norm_mix args = {
                .n = (int32_t)in_dim,
                .out_dim = (int32_t)out_dim,
                .eps = eps,
            };
            [enc setComputePipelineState:pso_cand2];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:buf_res->mtl_buf offset:buf_res->offset atIndex:1];
            [enc setBuffer:buf_fn->mtl_buf offset:buf_fn->offset atIndex:2];
            [enc setBuffer:buf_mix_cand2->mtl_buf offset:buf_mix_cand2->offset atIndex:3];
            [enc setThreadgroupMemoryLength:(32 + 2 * 2 * 32) * sizeof(float) atIndex:0];
            for (int c = 0; c < CALLS_PER_BATCH; c++) {
                metal_dispatch(enc, "cand2_fused_cluster2", MTLSizeMake(6, 1, 1), MTLSizeMake(32, 16, 1));
            }
            v41_finish(cb, enc);
            REQUIRE(myds_metal_commit_batch());
        }
        uint64_t bt5 = mach_absolute_time();
        double pipe_cand2_us = time_in_us(bt4, bt5) / (BATCH_RUNS * CALLS_PER_BATCH);

        printf("\n=======================================================\n");
        printf("PIPELINED BATCH RESULTS (%d total dispatches per arm):\n", BATCH_RUNS * CALLS_PER_BATCH);
        printf("  Baseline (v41_norm + barrier + v41_gemv) : %6.3f us/call\n", pipe_base_us);
        printf("  Candidate 1 (fused NR0=2, grid=12)       : %6.3f us/call  (Delta: %+.3f us, %+.1f%%)\n",
               pipe_cand1_us, pipe_cand1_us - pipe_base_us, (pipe_cand1_us - pipe_base_us) / pipe_base_us * 100.0);
        printf("  Candidate 2 (fused cluster2, grid=6)     : %6.3f us/call  (Delta: %+.3f us, %+.1f%%)\n",
               pipe_cand2_us, pipe_cand2_us - pipe_base_us, (pipe_cand2_us - pipe_base_us) / pipe_base_us * 100.0);
        printf("-------------------------------------------------------\n");
        double pipe_best_delta = pipe_base_us - fmin(pipe_cand1_us, pipe_cand2_us);
        printf("  Net savings per call                     :  %.3f us\n", pipe_best_delta);
        printf("  Projected savings per token (80 calls)   : %6.1f us (~%.3f ms/token)\n",
               pipe_best_delta * 80.0, (pipe_best_delta * 80.0) / 1000.0);
        printf("=======================================================\n");

        myds_metal_buffer_free(buf_res);
        myds_metal_buffer_free(buf_fn);

        myds_metal_buffer_free(buf_flat_norm);
        myds_metal_buffer_free(buf_mix_base);
        myds_metal_buffer_free(buf_mix_cand1);
        myds_metal_buffer_free(buf_mix_cand2);
        myds_metal_cleanup();
    }
    return 0;
}
