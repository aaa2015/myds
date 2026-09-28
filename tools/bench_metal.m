/* Reproducible operator benchmarks. Run from the repository root:
 *   make bin/bench_metal && ./bin/bench_metal
 * --quick reduces repetitions; --shader-dir PATH locates glu.metal.
 * No model weights are needed. Outputs must agree before timings are reported.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "myds_metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct bench_glu_args {
    int32_t ne00;
    uint64_t nb01;
    int32_t ne10;
    uint64_t nb11;
    int32_t ne0;
    uint64_t nb1;
    int32_t i00;
    int32_t i10;
    float alpha;
    float limit;
    int32_t round_bf16;
};

static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_metal: %s\n", what); exit(1); }
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *values, unsigned n) {
    qsort(values, n, sizeof(*values), compare_double);
    return values[n / 2];
}

static double measure_swiglu(id<MTLCommandQueue> queue,
                            id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> gate, id<MTLBuffer> up, id<MTLBuffer> out,
                            uint32_t dim, unsigned repetitions, bool flat) {
    @autoreleasepool {
        struct bench_glu_args args = {
            .ne00 = (int32_t)dim, .nb01 = (uint64_t)dim * sizeof(float),
            .ne10 = (int32_t)dim, .nb11 = (uint64_t)dim * sizeof(float),
            .ne0 = (int32_t)dim, .nb1 = (uint64_t)dim * sizeof(float),
            .alpha = 1.0f,
        };
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate offset:0 atIndex:1];
        [enc setBuffer:up offset:0 atIndex:2];
        [enc setBuffer:out offset:0 atIndex:3];
        NSUInteger threads = MIN((NSUInteger)dim, MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup));
        for (unsigned i = 0; i < repetitions; i++) {
            if (flat) {
                [enc dispatchThreads:MTLSizeMake(dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            } else {
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "bench_metal: %s\n", [[cb.error localizedDescription] UTF8String]);
            exit(1);
        }
        require(cb.GPUEndTime > cb.GPUStartTime, "GPU timestamps are unavailable");
        return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / repetitions;
    }
}

static void bench_swiglu(const char *shader_dir, unsigned samples, unsigned repetitions) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "Metal device unavailable");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSString *path = [NSString stringWithFormat:@"%s/glu.metal", shader_dir];
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    require(source != nil, "cannot read glu.metal");
    source = [@"#include <metal_stdlib>\nusing namespace metal;\n" stringByAppendingString:source];
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeFast;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) fprintf(stderr, "%s\n", [[error localizedDescription] UTF8String]);
    require(library != nil, "cannot compile SwiGLU shaders");
    id<MTLComputePipelineState> legacy = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"kernel_swiglu_f32"] error:&error];
    id<MTLComputePipelineState> flat = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"kernel_swiglu_flat_f32"] error:&error];
    require(legacy && flat && queue, "cannot create SwiGLU pipelines");
    printf("# GPU: %s; median of %u alternating samples, %u dispatches/sample\n",
           [[device name] UTF8String], samples, repetitions);
    const uint32_t dims[] = { 1024, 2048, 8192, 32768, 131072 };
    for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++) {
        @autoreleasepool {
            uint32_t dim = dims[d];
            size_t bytes = (size_t)dim * sizeof(float);
            id<MTLBuffer> gate = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> up = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> old_out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> new_out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            require(gate && up && old_out && new_out, "SwiGLU buffer allocation failed");
            float *g = gate.contents, *u = up.contents;
            for (uint32_t i = 0; i < dim; i++) {
                g[i] = ((int)(i % 101) - 50) * 0.125f;
                u[i] = ((int)(i % 37) - 18) * 0.0625f;
            }
            /* Warm both paths and validate before measuring. */
            (void)measure_swiglu(queue, legacy, gate, up, old_out, dim, 32, false);
            (void)measure_swiglu(queue, flat, gate, up, new_out, dim, 32, true);
            require(memcmp(old_out.contents, new_out.contents, bytes) == 0,
                    "SwiGLU old/new outputs are not bit-exact");
            float *result = new_out.contents;
            for (uint32_t i = 0; i < dim; i++) {
                float expected = g[i] / (1.0f + expf(-g[i])) * u[i];
                require(isfinite(result[i]) && fabsf(result[i] - expected) < 1e-5f,
                        "SwiGLU differs from CPU reference");
            }
            double before[15], after[15];
            for (unsigned s = 0; s < samples; s++) {
                if (s % 2 == 0) {
                    before[s] = measure_swiglu(queue, legacy, gate, up, old_out, dim, repetitions, false);
                    after[s] = measure_swiglu(queue, flat, gate, up, new_out, dim, repetitions, true);
                } else {
                    after[s] = measure_swiglu(queue, flat, gate, up, new_out, dim, repetitions, true);
                    before[s] = measure_swiglu(queue, legacy, gate, up, old_out, dim, repetitions, false);
                }
            }
            double old_us = median(before, samples), new_us = median(after, samples);
            printf("swiglu,%u,%.3f,%.3f,%.3f,gpu_us_per_dispatch\n", dim, old_us, new_us, old_us / new_us);
        }
    }
}

static double measure_gemv(myds_metal_buffer_t *out, myds_metal_buffer_t *weights,
                           myds_metal_buffer_t *input, myds_metal_buffer_t **out_rows,
                           myds_metal_buffer_t **in_rows, uint32_t out_dim,
                           uint32_t in_dim, uint32_t n_tok, ggml_type type, bool batch) {
    double start = now_seconds();
    require(myds_metal_begin_batch(), "begin_batch failed");
    if (batch) require(myds_metal_gemv_batch(out, weights, input, out_dim, in_dim, n_tok, type), "batched GEMV failed");
    else {
        for (uint32_t t = 0; t < n_tok; t++) {
            require(myds_metal_gemv(out_rows[t], weights, in_rows[t], out_dim, in_dim, type), "single-token GEMV failed");
        }
    }
    require(myds_metal_commit_batch(), "GEMV completion failed");
    return (now_seconds() - start) * 1e6;
}

static void bench_gemv(const char *shader_dir, unsigned samples) {
    require(myds_metal_init(shader_dir), "runtime initialization failed");
    const ggml_type types[] = { GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32 };
    const uint32_t tokens[] = { 8, 32 };
    const uint32_t in_dim = 2048, out_dim = 2048;
    printf("# GEMV: wall time includes CPU encoding and GPU completion; row views allocated before timing\n");
    for (size_t k = 0; k < 3; k++) {
        for (size_t n = 0; n < 2; n++) {
            ggml_type type = types[k];
            uint32_t n_tok = tokens[n];
            size_t row_bytes = ggml_type_row_size(type, in_dim);
            size_t w_bytes = row_bytes * out_dim;
            size_t x_bytes = (size_t)n_tok * in_dim * sizeof(float);
            size_t y_bytes = (size_t)n_tok * out_dim * sizeof(float);
            myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
            myds_metal_buffer_t *x = myds_metal_buffer_alloc(x_bytes);
            myds_metal_buffer_t *old_out = myds_metal_buffer_alloc(y_bytes);
            myds_metal_buffer_t *new_out = myds_metal_buffer_alloc(y_bytes);
            require(w && x && old_out && new_out, "GEMV buffer allocation failed");
            float *xv = myds_metal_buffer_data(x);
            for (size_t i = 0; i < x_bytes / sizeof(float); i++) xv[i] = sinf((float)i * 0.17f);
            if (type == GGML_TYPE_Q8_0) {
                typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0_t;
                block_q8_0_t *blocks = myds_metal_buffer_data(w);
                for (size_t b = 0; b < w_bytes / sizeof(*blocks); b++) {
                    blocks[b].d = 0x3000;
                    for (int i = 0; i < 32; i++) blocks[b].qs[i] = (int8_t)((b * 7 + i) % 15) - 7;
                }
            } else {
                for (size_t i = 0; i < (size_t)out_dim * in_dim; i++) {
                    float value = ((int)(i % 17) - 8) * 0.125f;
                    if (type == GGML_TYPE_F32) ((float *)myds_metal_buffer_data(w))[i] = value;
                    else ((_Float16 *)myds_metal_buffer_data(w))[i] = (_Float16)value;
                }
            }
            myds_metal_buffer_t *x_rows[32], *y_rows[32];
            for (uint32_t t = 0; t < n_tok; t++) {
                x_rows[t] = myds_metal_buffer_sub_view(x, (size_t)t * in_dim * sizeof(float), in_dim * sizeof(float));
                y_rows[t] = myds_metal_buffer_sub_view(old_out, (size_t)t * out_dim * sizeof(float), out_dim * sizeof(float));
                require(x_rows[t] && y_rows[t], "row view allocation failed");
            }
            for (unsigned warm = 0; warm < 3; warm++) {
                (void)measure_gemv(old_out, w, x, y_rows, x_rows, out_dim, in_dim, n_tok, type, false);
                (void)measure_gemv(new_out, w, x, y_rows, x_rows, out_dim, in_dim, n_tok, type, true);
            }
            require(memcmp(myds_metal_buffer_data(old_out), myds_metal_buffer_data(new_out), y_bytes) == 0,
                    "GEMV old/new outputs are not bit-exact");
            double before[15], after[15];
            for (unsigned s = 0; s < samples; s++) {
                if (s % 2 == 0) {
                    before[s] = measure_gemv(old_out, w, x, y_rows, x_rows, out_dim, in_dim, n_tok, type, false);
                    after[s] = measure_gemv(new_out, w, x, y_rows, x_rows, out_dim, in_dim, n_tok, type, true);
                } else {
                    after[s] = measure_gemv(new_out, w, x, y_rows, x_rows, out_dim, in_dim, n_tok, type, true);
                    before[s] = measure_gemv(old_out, w, x, y_rows, x_rows, out_dim, in_dim, n_tok, type, false);
                }
            }
            double old_us = median(before, samples), new_us = median(after, samples);
            printf("gemv_%s,%ux%ux%u,%.3f,%.3f,%.3f,wall_us_per_batch\n",
                   ggml_type_name(type), out_dim, in_dim, n_tok, old_us, new_us, old_us / new_us);
            for (uint32_t t = 0; t < n_tok; t++) {
                myds_metal_buffer_free(x_rows[t]);
                myds_metal_buffer_free(y_rows[t]);
            }
            myds_metal_buffer_free(w);
            myds_metal_buffer_free(x);
            myds_metal_buffer_free(old_out);
            myds_metal_buffer_free(new_out);
        }
    }
    myds_metal_cleanup();
}

int main(int argc, char **argv) {
    const char *shader_dir = "metal";
    unsigned samples = 9, repetitions = 512;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) { samples = 5; repetitions = 128; }
        else if (strcmp(argv[i], "--shader-dir") == 0 && i + 1 < argc) shader_dir = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--quick] [--shader-dir PATH]\n", argv[0]);
            return 1;
        }
    }
    @autoreleasepool {
        printf("kind,shape,baseline_us,optimized_us,speedup,metric\n");
        bench_swiglu(shader_dir, samples, repetitions);
        bench_gemv(shader_dir, samples);
    }
    return 0;
}
