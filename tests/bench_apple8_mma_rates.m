/* tests/bench_apple8_mma_rates.m
 *
 * Standalone Apple Silicon (M2 Ultra Apple8 / M4 Apple9) hardware-gate probe:
 * measures sustained peak TFLOP/s of Metal simdgroup_multiply_accumulate across:
 *   1. simdgroup_half8x8  x half8x8  -> half8x8   (pure FP16)
 *   2. simdgroup_half8x8  x half8x8  -> float8x8  (FP16 inputs, FP32 accum)
 *   3. simdgroup_float8x8 x float8x8 -> float8x8  (pure FP32)
 *   4. simdgroup_float8x8 x half8x8  -> float8x8  (mixed P*V style)
 *
 * Design:
 * - Pure compute inner loop: 8 independent 8x8 accumulators (C[0..7] += A[0..1] * B[0..3])
 *   unrolled 2x per outer iteration (16 MMAs = 16,384 FLOPs / iter / simdgroup),
 *   zero barriers and zero memory traffic inside the timed loop.
 * - Runtime anti-folding check: verifies checksum(2*k) ~= 2 * checksum(k) and finite
 *   non-zero outputs so compiler DCE / loop-folding cannot inflate numbers.
 * - Batches multiple dispatches inside ONE MTLCommandBuffer (single compute encoder)
 *   to amortize the ~190 us commit tax per ledger §B.3, timing via GPUStartTime..GPUEndTime
 *   with 2 warmup CBs + 7 measured runs (reporting min / median / max TFLOP/s).
 * - Prints PSO maxTotalThreadsPerThreadgroup, threadExecutionWidth, and GPU envelope.
 *
 * Usage:
 *   bin/bench_apple8_mma_rates [--tgs N] [--nsg N] [--iters N] [--batch N] [--runs N]
 */
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t query_gpu_core_count(void) {
    io_iterator_t iter = IO_OBJECT_NULL;
    kern_return_t kr = IOServiceGetMatchingServices(
        kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iter);
    if (kr != KERN_SUCCESS || iter == IO_OBJECT_NULL) return 0;
    uint32_t cores = 0;
    io_object_t service = IO_OBJECT_NULL;
    while ((service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        CFTypeRef prop = IORegistryEntryCreateCFProperty(
            service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (prop) {
            if (CFGetTypeID(prop) == CFNumberGetTypeID()) {
                int64_t v = 0;
                if (CFNumberGetValue((CFNumberRef)prop, kCFNumberSInt64Type, &v) && v > 0) {
                    cores = (uint32_t)v;
                }
            }
            CFRelease(prop);
        }
        IOObjectRelease(service);
        if (cores > 0) break;
    }
    IOObjectRelease(iter);
    return cores;
}

static int detect_apple_gpu_family(id<MTLDevice> dev) {
    int best = 0;
    for (int fam = 1; fam <= 14; ++fam) {
        MTLGPUFamily f = (MTLGPUFamily)(1000 + fam);
        @try {
            if ([dev supportsFamily:f]) best = fam;
        } @catch (__unused NSException *ex) {
            break;
        }
    }
    return best;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da < db) ? -1 : (da > db ? 1 : 0);
}

struct mma_params {
    uint32_t iters;
    uint32_t out_stride;
};

static NSString *const kMMAShaderSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct mma_params {\n"
"    uint iters;\n"
"    uint out_stride;\n"
"};\n"
"\n"
"template <typename MatA, typename MatB, typename MatC, typename ElemA, typename ElemB, typename ElemC>\n"
"static inline void mma_rate_body(\n"
"    constant mma_params &p,\n"
"    device const ElemA *in_a,\n"
"    device const ElemB *in_b,\n"
"    device float *out_sum,\n"
"    uint tg_id,\n"
"    ushort sgitg,\n"
"    ushort nsg,\n"
"    ushort tiisg,\n"
"    threadgroup ElemC *scratch)\n"
"{\n"
"    MatA a0, a1;\n"
"    MatB b0, b1, b2, b3;\n"
"    simdgroup_load(a0, in_a + 0,  8);\n"
"    simdgroup_load(a1, in_a + 64, 8);\n"
"    simdgroup_load(b0, in_b + 0,   8);\n"
"    simdgroup_load(b1, in_b + 64,  8);\n"
"    simdgroup_load(b2, in_b + 128, 8);\n"
"    simdgroup_load(b3, in_b + 192, 8);\n"
"\n"
"    MatC c0 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"    MatC c1 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"    MatC c2 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"    MatC c3 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"    MatC c4 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"    MatC c5 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"    MatC c6 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"    MatC c7 = make_filled_simdgroup_matrix<ElemC, 8, 8>(ElemC(0));\n"
"\n"
"    #pragma clang loop unroll(disable)\n"
"    for (uint it = 0; it < p.iters; ++it) {\n"
"        simdgroup_multiply_accumulate(c0, a0, b0, c0);\n"
"        simdgroup_multiply_accumulate(c1, a0, b1, c1);\n"
"        simdgroup_multiply_accumulate(c2, a0, b2, c2);\n"
"        simdgroup_multiply_accumulate(c3, a0, b3, c3);\n"
"        simdgroup_multiply_accumulate(c4, a1, b0, c4);\n"
"        simdgroup_multiply_accumulate(c5, a1, b1, c5);\n"
"        simdgroup_multiply_accumulate(c6, a1, b2, c6);\n"
"        simdgroup_multiply_accumulate(c7, a1, b3, c7);\n"
"\n"
"        simdgroup_multiply_accumulate(c0, a0, b0, c0);\n"
"        simdgroup_multiply_accumulate(c1, a0, b1, c1);\n"
"        simdgroup_multiply_accumulate(c2, a0, b2, c2);\n"
"        simdgroup_multiply_accumulate(c3, a0, b3, c3);\n"
"        simdgroup_multiply_accumulate(c4, a1, b0, c4);\n"
"        simdgroup_multiply_accumulate(c5, a1, b1, c5);\n"
"        simdgroup_multiply_accumulate(c6, a1, b2, c6);\n"
"        simdgroup_multiply_accumulate(c7, a1, b3, c7);\n"
"    }\n"
"\n"
"    threadgroup ElemC *sg_buf = scratch + (uint)sgitg * 64u;\n"
"    float acc = 0.0f;\n"
"    #define ACC_MAT(M) \\\n"
"        simdgroup_store(M, sg_buf, 8); \\\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup); \\\n"
"        acc += (float)sg_buf[tiisg] + (float)sg_buf[tiisg + 32u]; \\\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"    ACC_MAT(c0) ACC_MAT(c1) ACC_MAT(c2) ACC_MAT(c3)\n"
"    ACC_MAT(c4) ACC_MAT(c5) ACC_MAT(c6) ACC_MAT(c7)\n"
"    #undef ACC_MAT\n"
"    acc = simd_sum(acc);\n"
"    if (tiisg == 0) {\n"
"        out_sum[tg_id * (uint)nsg + (uint)sgitg] = acc;\n"
"    }\n"
"}\n"
"\n"
"kernel void kernel_mma_h_h_h(\n"
"    constant mma_params &p          [[buffer(0)]],\n"
"    device const half *in_a         [[buffer(1)]],\n"
"    device const half *in_b         [[buffer(2)]],\n"
"    device float *out_sum           [[buffer(3)]],\n"
"    threadgroup half *scratch       [[threadgroup(0)]],\n"
"    uint tg_id                      [[threadgroup_position_in_grid]],\n"
"    ushort sgitg                    [[simdgroup_index_in_threadgroup]],\n"
"    ushort nsg                      [[simdgroups_per_threadgroup]],\n"
"    ushort tiisg                    [[thread_index_in_simdgroup]])\n"
"{\n"
"    mma_rate_body<simdgroup_half8x8, simdgroup_half8x8, simdgroup_half8x8, half, half, half>(\n"
"        p, in_a, in_b, out_sum, tg_id, sgitg, nsg, tiisg, scratch);\n"
"}\n"
"\n"
"kernel void kernel_mma_h_h_f(\n"
"    constant mma_params &p          [[buffer(0)]],\n"
"    device const half *in_a         [[buffer(1)]],\n"
"    device const half *in_b         [[buffer(2)]],\n"
"    device float *out_sum           [[buffer(3)]],\n"
"    threadgroup float *scratch      [[threadgroup(0)]],\n"
"    uint tg_id                      [[threadgroup_position_in_grid]],\n"
"    ushort sgitg                    [[simdgroup_index_in_threadgroup]],\n"
"    ushort nsg                      [[simdgroups_per_threadgroup]],\n"
"    ushort tiisg                    [[thread_index_in_simdgroup]])\n"
"{\n"
"    mma_rate_body<simdgroup_half8x8, simdgroup_half8x8, simdgroup_float8x8, half, half, float>(\n"
"        p, in_a, in_b, out_sum, tg_id, sgitg, nsg, tiisg, scratch);\n"
"}\n"
"\n"
"kernel void kernel_mma_f_f_f(\n"
"    constant mma_params &p          [[buffer(0)]],\n"
"    device const float *in_a        [[buffer(1)]],\n"
"    device const float *in_b        [[buffer(2)]],\n"
"    device float *out_sum           [[buffer(3)]],\n"
"    threadgroup float *scratch      [[threadgroup(0)]],\n"
"    uint tg_id                      [[threadgroup_position_in_grid]],\n"
"    ushort sgitg                    [[simdgroup_index_in_threadgroup]],\n"
"    ushort nsg                      [[simdgroups_per_threadgroup]],\n"
"    ushort tiisg                    [[thread_index_in_simdgroup]])\n"
"{\n"
"    mma_rate_body<simdgroup_float8x8, simdgroup_float8x8, simdgroup_float8x8, float, float, float>(\n"
"        p, in_a, in_b, out_sum, tg_id, sgitg, nsg, tiisg, scratch);\n"
"}\n"
"\n"
"kernel void kernel_mma_f_h_f(\n"
"    constant mma_params &p          [[buffer(0)]],\n"
"    device const float *in_a        [[buffer(1)]],\n"
"    device const half  *in_b        [[buffer(2)]],\n"
"    device float *out_sum           [[buffer(3)]],\n"
"    threadgroup float *scratch      [[threadgroup(0)]],\n"
"    uint tg_id                      [[threadgroup_position_in_grid]],\n"
"    ushort sgitg                    [[simdgroup_index_in_threadgroup]],\n"
"    ushort nsg                      [[simdgroups_per_threadgroup]],\n"
"    ushort tiisg                    [[thread_index_in_simdgroup]])\n"
"{\n"
"    mma_rate_body<simdgroup_float8x8, simdgroup_half8x8, simdgroup_float8x8, float, half, float>(\n"
"        p, in_a, in_b, out_sum, tg_id, sgitg, nsg, tiisg, scratch);\n"
"}\n";

typedef struct {
    const char *label;
    const char *kernel_name;
    BOOL a_is_half;
    BOOL b_is_half;
    size_t scratch_elem_bytes;
} mma_case_t;

static double run_single_cb(
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pso,
    id<MTLBuffer> buf_a,
    id<MTLBuffer> buf_b,
    id<MTLBuffer> buf_out,
    uint32_t n_tgs,
    uint32_t nsg,
    uint32_t iters,
    uint32_t n_batch,
    size_t scratch_bytes,
    double *wall_ms_out)
{
    struct mma_params params = { .iters = iters, .out_stride = n_tgs * nsg };
    @autoreleasepool {
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBytes:&params length:sizeof(params) atIndex:0];
        [enc setBuffer:buf_a offset:0 atIndex:1];
        [enc setBuffer:buf_b offset:0 atIndex:2];
        [enc setBuffer:buf_out offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        MTLSize grid = MTLSizeMake(n_tgs, 1, 1);
        MTLSize tpg  = MTLSizeMake(32, nsg, 1);
        for (uint32_t b = 0; b < n_batch; ++b) {
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "ERROR: command buffer failed: %s\n",
                    cb.error.localizedDescription.UTF8String ?: "unknown");
            exit(1);
        }
        if (wall_ms_out) *wall_ms_out = (t1 - t0) * 1000.0;
        double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
        return (gpu_s > 0.0) ? (gpu_s * 1000.0) : ((t1 - t0) * 1000.0);
    }
}

int main(int argc, char **argv) {
    uint32_t n_tgs   = 1024;
    uint32_t nsg     = 4;
    uint32_t iters   = 256;
    uint32_t n_batch = 8;
    uint32_t n_runs  = 7;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tgs") == 0 && i + 1 < argc) n_tgs = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--nsg") == 0 && i + 1 < argc) nsg = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) n_batch = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) n_runs = (uint32_t)atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--tgs 1024] [--nsg 4] [--iters 256] [--batch 8] [--runs 7]\n", argv[0]);
            return 2;
        }
    }
    if (!n_tgs || !nsg || !iters || !n_batch || !n_runs || n_runs > 101) {
        fprintf(stderr, "invalid parameters\n");
        return 2;
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "ERROR: no Metal device\n");
            return 1;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            fprintf(stderr, "ERROR: cannot create command queue\n");
            return 1;
        }

        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeFast;
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kMMAShaderSource options:opts error:&err];
        if (!lib) {
            fprintf(stderr, "ERROR: shader compile failed: %s\n",
                    err.localizedDescription.UTF8String ?: "unknown");
            return 1;
        }

        const uint32_t gpu_cores = query_gpu_core_count();
        const int apple_fam = detect_apple_gpu_family(dev);

        /* Prepare deterministic small-magnitude A[2 x 64] and B[4 x 64] so sums stay
         * finite and normal in FP16 even across large iteration counts. */
        id<MTLBuffer> buf_a_f32 = [dev newBufferWithLength:2 * 64 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b_f32 = [dev newBufferWithLength:4 * 64 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_a_f16 = [dev newBufferWithLength:2 * 64 * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b_f16 = [dev newBufferWithLength:4 * 64 * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out   = [dev newBufferWithLength:(size_t)n_tgs * nsg * sizeof(float) options:MTLResourceStorageModeShared];

        float  *af32 = (float  *)buf_a_f32.contents;
        float  *bf32 = (float  *)buf_b_f32.contents;
        __fp16 *af16 = (__fp16 *)buf_a_f16.contents;
        __fp16 *bf16 = (__fp16 *)buf_b_f16.contents;
        for (int m = 0; m < 2; ++m) {
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    float v = 0.025f * (1.0f + 0.04f * (float)r + 0.03f * (float)(m + 1));
                    af16[m * 64 + r * 8 + c] = (__fp16)v;
                    af32[m * 64 + r * 8 + c] = (float)af16[m * 64 + r * 8 + c];
                }
            }
        }
        for (int m = 0; m < 4; ++m) {
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    float v = 0.025f * (1.0f + 0.04f * (float)c + 0.02f * (float)(m + 1));
                    bf16[m * 64 + r * 8 + c] = (__fp16)v;
                    bf32[m * 64 + r * 8 + c] = (float)bf16[m * 64 + r * 8 + c];
                }
            }
        }

        /* 16 MMAs per outer iter * (8*8*8*2 = 1024 FLOPs per MMA) = 16,384 FLOPs/iter/SG */
        const double flops_per_cb = (double)n_batch * (double)n_tgs * (double)nsg *
                                    (double)iters * 16.0 * 1024.0;

        printf("=== Apple Silicon simdgroup_multiply_accumulate Sustained Rate Probe ===\n");
        printf("device=%s  apple_family=Apple%d  gpu_cores=%u\n",
               dev.name.UTF8String, apple_fam, gpu_cores);
        printf("config: grid=(%u,1,1) threads=(32,%u,1) iters=%u (16 MMAs/iter) batch=%u/CB runs=%u (%.2f GFLOP/CB)\n\n",
               n_tgs, nsg, iters, n_batch, n_runs, flops_per_cb / 1e9);

        const mma_case_t cases[] = {
            { "half8x8  x half8x8  -> half8x8 ", "kernel_mma_h_h_h", YES, YES, sizeof(__fp16) },
            { "half8x8  x half8x8  -> float8x8", "kernel_mma_h_h_f", YES, YES, sizeof(float)  },
            { "float8x8 x float8x8 -> float8x8", "kernel_mma_f_f_f", NO,  NO,  sizeof(float)  },
            { "float8x8 x half8x8  -> float8x8", "kernel_mma_f_h_f", NO,  YES, sizeof(float)  },
        };
        const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
        id<MTLComputePipelineState> psos[4];
        id<MTLBuffer> bas[4], bbs[4];
        size_t scratch_arr[4];
        NSUInteger max_thr_arr[4] = {0};
        double gpu_ms[4][101];
        double tflops[4][101];
        float  chk_arr[4] = {0};

        /* Build all PSOs and run anti-folding check before timed runs. */
        for (size_t ci = 0; ci < n_cases; ++ci) {
            const mma_case_t *tc = &cases[ci];
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:tc->kernel_name]];
            if (!fn) {
                fprintf(stderr, "ERROR: missing kernel %s\n", tc->kernel_name);
                return 1;
            }
            NSError *perr = nil;
            psos[ci] = [dev newComputePipelineStateWithFunction:fn error:&perr];
            if (!psos[ci]) {
                fprintf(stderr, "ERROR: PSO failed for %s: %s\n",
                        tc->kernel_name, perr.localizedDescription.UTF8String ?: "unknown");
                return 1;
            }
            max_thr_arr[ci] = psos[ci].maxTotalThreadsPerThreadgroup;
            if (32u * nsg > max_thr_arr[ci]) {
                fprintf(stderr, "ERROR: %s maxTotalThreadsPerThreadgroup=%lu < %u\n",
                        tc->kernel_name, (unsigned long)max_thr_arr[ci], 32u * nsg);
                return 1;
            }
            bas[ci] = tc->a_is_half ? buf_a_f16 : buf_a_f32;
            bbs[ci] = tc->b_is_half ? buf_b_f16 : buf_b_f32;
            scratch_arr[ci] = (size_t)nsg * 64u * tc->scratch_elem_bytes;

            /* Anti-folding verification: run iters=8 vs iters=16 (1 dispatch) and verify linear scaling. */
            (void)run_single_cb(queue, psos[ci], bas[ci], bbs[ci], buf_out, 16, nsg, 8, 1, scratch_arr[ci], NULL);
            float chk8 = ((const float *)buf_out.contents)[0];
            (void)run_single_cb(queue, psos[ci], bas[ci], bbs[ci], buf_out, 16, nsg, 16, 1, scratch_arr[ci], NULL);
            float chk16 = ((const float *)buf_out.contents)[0];
            if (!isfinite(chk8) || !isfinite(chk16) || fabsf(chk8) < 1e-6f ||
                fabsf(chk16 / chk8 - 2.0f) > 0.05f) {
                fprintf(stderr, "ERROR: anti-folding check failed for %s (chk8=%g chk16=%g)\n",
                        tc->kernel_name, chk8, chk16);
                return 1;
            }
        }

        /* 2 interleaved warmup rounds across all variants (stabilizes DVFS/boost). */
        for (int w = 0; w < 2; ++w) {
            for (size_t ci = 0; ci < n_cases; ++ci) {
                (void)run_single_cb(queue, psos[ci], bas[ci], bbs[ci], buf_out,
                                    n_tgs, nsg, iters, n_batch, scratch_arr[ci], NULL);
            }
        }

        /* n_runs interleaved timed rounds across all variants. */
        for (uint32_t r = 0; r < n_runs; ++r) {
            for (size_t ci = 0; ci < n_cases; ++ci) {
                double wall_ms = 0.0;
                gpu_ms[ci][r] = run_single_cb(queue, psos[ci], bas[ci], bbs[ci], buf_out,
                                              n_tgs, nsg, iters, n_batch, scratch_arr[ci], &wall_ms);
                tflops[ci][r] = (flops_per_cb / (gpu_ms[ci][r] * 1e-3)) / 1e12;
                if (r + 1 == n_runs) {
                    chk_arr[ci] = ((const float *)buf_out.contents)[0];
                    if (!isfinite(chk_arr[ci]) || fabsf(chk_arr[ci]) < 1e-6f) {
                        fprintf(stderr, "ERROR: invalid final checksum %g for %s\n",
                                chk_arr[ci], cases[ci].kernel_name);
                        return 1;
                    }
                }
            }
        }

        double med_ms_arr[4], med_tf_arr[4], min_tf_arr[4], max_tf_arr[4];
        for (size_t ci = 0; ci < n_cases; ++ci) {
            qsort(gpu_ms[ci], n_runs, sizeof(double), cmp_double);
            qsort(tflops[ci], n_runs, sizeof(double), cmp_double);
            med_ms_arr[ci] = gpu_ms[ci][n_runs / 2];
            med_tf_arr[ci] = tflops[ci][n_runs / 2];
            min_tf_arr[ci] = tflops[ci][0];
            max_tf_arr[ci] = tflops[ci][n_runs - 1];
        }

        const double f32_tf = med_tf_arr[2] > 0.0 ? med_tf_arr[2] : 1.0;
        printf("%-32s  %7s  %11s  %10s  %10s  %10s  %8s  %s\n",
               "MMA Variant", "maxThr", "med_ms(GPU)", "med_TF/s", "min_TF/s", "max_TF/s", "vs_f32", "chk");
        printf("--------------------------------------------------------------------------------------------------------------\n");
        for (size_t ci = 0; ci < n_cases; ++ci) {
            printf("%-32s  %7lu  %11.3f  %10.3f  %10.3f  %10.3f  %7.3fx  %.3f\n",
                   cases[ci].label, (unsigned long)max_thr_arr[ci],
                   med_ms_arr[ci], med_tf_arr[ci], min_tf_arr[ci], max_tf_arr[ci],
                   med_tf_arr[ci] / f32_tf, chk_arr[ci]);
        }
        printf("\nPASS: all 4 MMA variants verified (anti-folding + finite output).\n");
    }
    return 0;
}
