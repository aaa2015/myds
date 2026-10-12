// Bit-exactness + timing harness for IQ2_XXS gate/up/SwiGLU decode kernel candidates.
// No model needed: random IQ2_XXS weights with valid half scales, real decode geometry
// (384 experts, 6 active, dim 5120, expert_dim 2304, row 1320 B, expert 3041280 B).
//
//   cc -O2 -fobjc-arc -framework Foundation -framework Metal tools/bench_iq2_pair.m -o /tmp/bench_iq2_pair
//   /tmp/bench_iq2_pair bin/myds.metallib [nsg=4] [trials=24] [reps=9] [cand1,cand2,...]
//
// For each candidate: compares the MID output (the only thing the release path consumes)
// bit-for-bit against kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32, then times 40 back-to-back
// dispatches with different cold experts (one token's worth), A-B-A.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>

struct args_mul_mv_id {
    int32_t nei0, nei1; uint64_t nbi1; int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13; uint64_t nb10, nb11, nb12; int32_t ne0, ne1; uint64_t nb1;
    int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base;
};
struct swiglu_args { uint32_t width, rows; uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride; uint32_t write_clamped; float clamp_value; };

static uint64_t rng_s = 88172645463325252ull;
static uint64_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void fill_iq2(uint8_t *p, size_t bytes) {
    for (size_t i = 0; i < bytes / 8; ++i) ((uint64_t *)p)[i] = rnd();
    // d (half) at the start of every 66-byte block: finite, small, varied.
    const size_t nblk = bytes / 66;
    for (size_t b = 0; b < nblk; ++b) {
        const uint16_t e = (uint16_t)(0x2800 + (rnd() % 0x400));   // ~0.0004..0.03, positive
        memcpy(p + b * 66, &e, 2);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: %s metallib [nsg] [trials] [reps] [cands]\n", argv[0]); return 2; }
        const short nsg = argc > 2 ? (short)atoi(argv[2]) : 4;
        const int trials = argc > 3 ? atoi(argv[3]) : 24;
        const int reps = argc > 4 ? atoi(argv[4]) : 9;
        NSArray<NSString *> *cands = @[@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_vec_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_vecfma_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_vtab_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_pf_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl0_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl1_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl2_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl3_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl4_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl6_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl7_f32",
                                       @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_abl8_f32"];
        if (argc > 5) {
            NSMutableArray *m = [NSMutableArray new];
            for (NSString *s in [[NSString stringWithUTF8String:argv[5]] componentsSeparatedByString:@","]) [m addObject:s];
            cands = m;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
        if (!lib) { fprintf(stderr, "metallib: %s\n", err.localizedDescription.UTF8String); return 1; }
        MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
        [cv setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
        NSMutableArray<NSString *> *names = [NSMutableArray arrayWithObject:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"];
        [names addObjectsFromArray:cands];
        NSMutableArray<id<MTLComputePipelineState>> *psos = [NSMutableArray new];
        for (NSString *n in names) {
            id<MTLFunction> fn = [lib newFunctionWithName:n constantValues:cv error:&err];
            id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
            if (!p) { fprintf(stderr, "PSO %s: %s\n", n.UTF8String, err.localizedDescription.UTF8String); return 1; }
            [psos addObject:p];
            printf("PSO %-52s maxThreads=%lu staticTG=%lu\n", n.UTF8String, (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)p.staticThreadgroupMemoryLength);
        }
        const uint32_t DIM = 5120, W = 2304, E = 384, ACT = 6, LAYERS = 40;
        const uint64_t ROW = 1320, EXP = ROW * W, TENSOR = EXP * E;
        id<MTLBuffer> gate = [dev newBufferWithLength:TENSOR options:MTLResourceStorageModeShared];
        id<MTLBuffer> up = [dev newBufferWithLength:TENSOR options:MTLResourceStorageModeShared];
        if (!gate || !up) { fprintf(stderr, "alloc failed\n"); return 1; }
        fill_iq2(gate.contents, TENSOR); fill_iq2(up.contents, TENSOR);
        id<MTLBuffer> x = [dev newBufferWithLength:DIM * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids = [dev newBufferWithLength:LAYERS * ACT * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> wts = [dev newBufferWithLength:LAYERS * ACT * 4 options:MTLResourceStorageModeShared];
        const size_t mid_bytes = (size_t)ACT * W * 4;
        NSMutableArray<id<MTLBuffer>> *mids = [NSMutableArray new];
        for (int i = 0; i < 8; ++i) [mids addObject:[dev newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared]];
        id<MTLBuffer> scratch = [dev newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        struct args_mul_mv_id a = {.nei0 = ACT, .nei1 = 1, .nbi1 = ACT * 4, .ne00 = DIM, .ne01 = W, .ne02 = E, .nb00 = 66,
            .nb01 = ROW, .nb02 = EXP, .ne10 = DIM, .ne11 = 1, .ne12 = 1, .ne13 = 1, .nb10 = 4, .nb11 = DIM * 4ull,
            .nb12 = DIM * 4ull, .ne0 = W, .ne1 = ACT, .nb1 = W * 4ull, .nr0 = 4};
        struct swiglu_args s = {.width = W, .rows = ACT, .gate_row_stride = W * 4ull, .up_row_stride = W * 4ull,
            .mid_row_stride = W * 4ull, .weight_stride = 4, .write_clamped = 0, .clamp_value = 10.0f};
        const NSUInteger rows_per_tg = 4 * (NSUInteger)nsg;
        const MTLSize grid = MTLSizeMake((W + rows_per_tg - 1) / rows_per_tg, 1, ACT);
        const MTLSize tgs = MTLSizeMake(32, (NSUInteger)nsg, 1);
        const NSUInteger smem = 256 * 8 + 128;

        void (^encode)(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, NSUInteger, id<MTLBuffer>, NSUInteger) =
            ^(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger layer, id<MTLBuffer> mid, NSUInteger smem_len) {
                [enc setComputePipelineState:p];
                [enc setBytes:&a length:sizeof(a) atIndex:0];
                [enc setBytes:&s length:sizeof(s) atIndex:1];
                [enc setBuffer:gate offset:0 atIndex:2];
                [enc setBuffer:up offset:0 atIndex:3];
                [enc setBuffer:x offset:0 atIndex:4];
                [enc setBuffer:scratch offset:0 atIndex:5];
                [enc setBuffer:scratch offset:0 atIndex:6];
                [enc setBuffer:mid offset:0 atIndex:7];
                [enc setBuffer:ids offset:layer * ACT * 4 atIndex:8];
                [enc setBuffer:wts offset:layer * ACT * 4 atIndex:9];
                [enc setThreadgroupMemoryLength:smem_len atIndex:0];
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
            };

        printf("device=%s nsg=%d trials=%d reps=%d\n", dev.name.UTF8String, nsg, trials, reps);
        int all_exact[8] = {0};
        for (NSUInteger c = 1; c < names.count; ++c) all_exact[c] = 1;
        for (int t = 0; t < trials; ++t) {
            float *xp = x.contents;
            for (uint32_t i = 0; i < DIM; ++i) xp[i] = ((int32_t)(rnd() % 2000001) - 1000000) / 1.0e6f * (1.0f + (t % 4));
            int32_t *ip = ids.contents; float *wp = wts.contents;
            for (uint32_t l = 0; l < LAYERS; ++l) for (uint32_t k = 0; k < ACT; ++k) {
                int32_t e; int dup;
                do { e = (int32_t)(rnd() % E); dup = 0; for (uint32_t j = 0; j < k; ++j) dup |= ip[l * ACT + j] == e; } while (dup);
                ip[l * ACT + k] = e; wp[l * ACT + k] = 0.1f + (rnd() % 1000) / 1000.0f;
            }
            s.clamp_value = (t % 3 == 0) ? 0.0f : 10.0f;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (NSUInteger c = 0; c < names.count; ++c) { memset(mids[c].contents, 0xa5, mid_bytes); encode(enc, psos[c], 0, mids[c], [names[c] containsString:@"vtab"] ? 12288 : smem); }
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            for (NSUInteger c = 1; c < names.count; ++c) {
                const uint32_t *r = mids[0].contents, *d = mids[c].contents;
                size_t bad = 0, maxbad = 0; for (size_t i = 0; i < mid_bytes / 4; ++i) if (r[i] != d[i]) { if (!bad) maxbad = i; ++bad; }
                if (bad) { all_exact[c] = 0; if (t < 3) printf("  trial %d %s: %zu/%zu words differ (first at %zu: ref %08x cand %08x)\n", t, names[c].UTF8String, bad, mid_bytes / 4, maxbad, r[maxbad], d[maxbad]); }
            }
        }
        for (NSUInteger c = 1; c < names.count; ++c)
            printf("bit-exact vs reference over %d trials: %-52s %s\n", trials, names[c].UTF8String, all_exact[c] ? "YES" : "NO");

        // Timing, A-B-A per candidate. 36.5 MB per dispatch.
        const double bytes = (double)ACT * 2 * EXP;
        s.clamp_value = 10.0f;
        double (^run)(NSUInteger) = ^double(NSUInteger c) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t l = 0; l < LAYERS; ++l) encode(enc, psos[c], l, mids[c], [names[c] containsString:@"vtab"] ? 12288 : smem);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / LAYERS;
        };
        for (NSUInteger c = 1; c < names.count; ++c) {
            double a1[64], b[64], a2[64]; int n = 0;
            (void)run(0); (void)run(c);
            for (int r = 0; r < reps && r < 64; ++r) { a1[n] = run(0); b[n] = run(c); a2[n] = run(0); ++n; }
            qsort(a1, n, 8, cmp); qsort(b, n, 8, cmp); qsort(a2, n, 8, cmp);
            const double ref = (a1[n / 2] + a2[n / 2]) / 2;
            printf("%-52s ref %.1f us (%.0f GB/s)  cand %.1f us (%.0f GB/s)  speedup %.2fx  = %.2f ms/token saved\n",
                   names[c].UTF8String, ref, bytes / ref / 1e3, b[n / 2], bytes / b[n / 2] / 1e3, ref / b[n / 2],
                   (ref - b[n / 2]) * LAYERS / 1e3);
        }
    }
    return 0;
}
