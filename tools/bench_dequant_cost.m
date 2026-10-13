// 反量化代价对照 harness（2026-10-13 参谋长）。仅诊断，不进生产。
//   cc -O2 -fobjc-arc -framework Foundation -framework Metal tools/bench_dequant_cost.m -o /tmp/bdq
//   /tmp/bdq <probe.metallib> [rows=64] [trials=24] [reps=64]
//
// 度量：每行单次派发的 GPU 时间（µs）。三种反量化路径处理同样 5120 权重/行，
// 且字节几乎相同（IQ2 1320 B vs 码本 1280 B）⇒ 差异来自指令而非带宽。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct dq_args { int32_t rows, reps; };

static uint64_t rng_s = 88172645463325252ull;
static uint64_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: %s probe.metallib [rows] [trials] [reps]\n", argv[0]); return 2; }
        const int ROWS   = argc > 2 ? atoi(argv[2]) : 64;
        const int trials = argc > 3 ? atoi(argv[3]) : 24;
        const int reps   = argc > 4 ? atoi(argv[4]) : 64;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
        if (!lib) { fprintf(stderr, "metallib: %s\n", err.localizedDescription.UTF8String); return 1; }

        NSArray<NSString *> *names = @[@"bench_dq_ref_iq2", @"bench_dq_vec_iq2", @"bench_dq_q4code", @"bench_dq_cb_table"];
        NSMutableArray<id<MTLComputePipelineState>> *psos = [NSMutableArray new];
        for (NSString *n in names) {
            id<MTLFunction> fn = [lib newFunctionWithName:n];
            id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
            if (!p) { fprintf(stderr, "PSO %s 不可用: %s\n", n.UTF8String, err.localizedDescription.UTF8String); return 1; }
            [psos addObject:p];
        }

        id<MTLBuffer> b_w   = [dev newBufferWithLength:(size_t)1320 * ROWS options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_q   = [dev newBufferWithLength:(size_t)1280 * ROWS options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_x   = [dev newBufferWithLength:5120 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_out = [dev newBufferWithLength:(size_t)ROWS * 4 options:MTLResourceStorageModeShared];
        if (!b_w || !b_q || !b_x || !b_out) { fprintf(stderr, "alloc failed\n"); return 1; }
        for (size_t i = 0; i < b_w.length / 8; ++i) ((uint64_t *)b_w.contents)[i] = rnd();
        for (size_t i = 0; i < b_q.length / 8; ++i) ((uint64_t *)b_q.contents)[i] = rnd();
        { float *xp = b_x.contents; for (int i = 0; i < 5120; ++i) xp[i] = ((int32_t)(rnd() % 2000001) - 1000000) / 1.0e6f; }
        // IQ2 每个 66B 块的 d(half) 设为有限正值，避免 NaN 影响耗时
        { uint8_t *p = b_w.contents; for (size_t b = 0; b < b_w.length / 66; ++b) { uint16_t e = (uint16_t)(0x2800 + (rnd() % 0x400)); memcpy(p + b * 66, &e, 2); } }
        // 码本尺度设为有限正值
        { float *sp = (float *)((uint8_t *)b_q.contents); for (size_t r = 0; r < (size_t)ROWS; ++r) for (int g = 0; g < 160; ++g) sp[r * 320 + g] = 0.01f + (rnd() % 100) / 1000.0f; }

        struct dq_args a = { ROWS, reps };
        const NSUInteger smem = 256 * 8 + 128;
        id<MTLCommandQueue> q = [dev newCommandQueue];

        void (^encode)(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer>) =
            ^(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> w) {
                [enc setComputePipelineState:p];
                [enc setBytes:&a length:sizeof(a) atIndex:0];
                [enc setBuffer:w offset:0 atIndex:1];
                [enc setBuffer:b_x offset:0 atIndex:2];
                [enc setBuffer:b_out offset:0 atIndex:3];
                [enc setThreadgroupMemoryLength:smem atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ROWS, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            };

        printf("device=%s rows=%d reps=%d  (每行 5120 权重；IQ2 1320B / 码本 1280B)\n", dev.name.UTF8String, ROWS, reps);
        double med[8][16]; int nmed[8] = {0};
        for (int t = 0; t < trials; ++t) {
            for (int c = 0; c < 4; ++c) {
                id<MTLBuffer> w = (c >= 2) ? b_q : b_w;
                id<MTLCommandBuffer> cb0 = [q commandBuffer];
                id<MTLComputeCommandEncoder> e0 = [cb0 computeCommandEncoder];
                encode(e0, psos[c], w); [e0 endEncoding]; [cb0 commit]; [cb0 waitUntilCompleted];
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int r = 0; r < reps; ++r) encode(enc, psos[c], w);
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / reps;
                if (nmed[c] < 16) med[c][nmed[c]++] = ms;
            }
        }
        double tmp[16]; memcpy(tmp, med[0], sizeof(double) * nmed[0]);
        qsort(tmp, nmed[0], sizeof(double), cmpd);
        const double base = tmp[nmed[0] / 2];
        printf("\n每行单次派发（中位）：\n");
        for (int c = 0; c < 4; ++c) {
            memcpy(tmp, med[c], sizeof(double) * nmed[c]);
            qsort(tmp, nmed[c], sizeof(double), cmpd);
            const double m = tmp[nmed[c] / 2];
            const double bytes = (c == 2) ? 1280.0 : 1320.0;
            printf("  %-18s %8.3f µs   行字节等效 %7.1f GB/s   vs ref %+7.2f%%\n",
                   names[c].UTF8String, m, bytes * ROWS / (m / 1000.0) / 1e9, (m / base - 1.0) * 100.0);
        }
        return 0;
    }
}
