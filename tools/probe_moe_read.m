// Memory-access probe for the decode routed-MoE kernels (diagnostic, no model needed).
//
//   cc -O2 -fobjc-arc -framework Foundation -framework Metal tools/probe_moe_read.m -o /tmp/probe_moe_read
//   /tmp/probe_moe_read [reps=7]
//
// Geometry mirrors DeepSeek-V4.1-Flash Q2 on M2 Ultra decode (384 experts, 6 active, dim 5120,
// expert_dim 2304): gate/up = IQ2_XXS (66 B/256, row 1320 B, expert 3041280 B), down = Q2_K
// (84 B/256, row 756 B, expert 3870720 B). Each dispatch reads 6 random experts, like one layer;
// 40 dispatches run back to back (one "token") so weights are cold, as in real decode.
//
// Modes (gate/up, 36.5 MB per dispatch; down, 23.2 MB):
//   lane  : exact load pattern of kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 (2-byte loads), gate/up only
//   coal  : same rows/bytes, lane-coalesced 8-byte loads, gate/up only
//   flat  : same expert regions, contiguous 16-byte loads per threadgroup slab (ideal within expert)
//   seq   : same bytes from one contiguous region (LM-head-like streaming ceiling)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>

static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
struct P { ulong expert_bytes; ulong up_off; ulong seq_base; uint row_bytes; uint nb32; uint ids_off;
           uint slab_u4; uint n_ten; uint n_exp; };

kernel void k_lane(device const uchar *base [[buffer(0)]], device const int *ids [[buffer(1)]],
                   device uint *out [[buffer(2)]], constant P &p [[buffer(3)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int e = ids[p.ids_off + tg.z];
    const uint first_row = (tg.x * 4 + sg) * 4;
    device const uchar *g = base + (ulong)e * p.expert_bytes + (ulong)first_row * p.row_bytes;
    device const uchar *u = g + p.up_off;
    uint acc = 0;
    for (uint ib32 = lane; ib32 < p.nb32; ib32 += 32) {
        const uint ibl = ib32 / 8, ib = ib32 % 8;
        for (uint r = 0; r < 4; ++r) {
            device const uchar *gb = g + (ulong)r * p.row_bytes + ibl * 66;
            device const uchar *ub = u + (ulong)r * p.row_bytes + ibl * 66;
            device const ushort *qg = (device const ushort *)(gb + 2) + 4 * ib;
            device const ushort *qu = (device const ushort *)(ub + 2) + 4 * ib;
            acc += (uint)(*(device const ushort *)gb) + (uint)(*(device const ushort *)ub);
            acc += (uint)qg[0] + qg[1] + qg[2] + qg[3] + (uint)qu[0] + qu[1] + qu[2] + qu[3];
        }
    }
    if (acc == 0xdeadbeefu) out[0] = acc;
}

kernel void k_coal(device const uchar *base [[buffer(0)]], device const int *ids [[buffer(1)]],
                   device uint *out [[buffer(2)]], constant P &p [[buffer(3)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int e = ids[p.ids_off + tg.z];
    const uint first_row = (tg.x * 4 + sg) * 4;
    uint acc = 0;
    for (uint t = 0; t < 2; ++t) {
        device const uchar *g = base + t * p.up_off + (ulong)e * p.expert_bytes + (ulong)first_row * p.row_bytes;
        for (uint r = 0; r < 4; ++r) {
            device const uint2 *row = (device const uint2 *)(g + (ulong)r * p.row_bytes);
            for (uint i = lane; i < p.row_bytes / 8; i += 32) { uint2 v = row[i]; acc += v.x ^ v.y; }
        }
    }
    if (acc == 0xdeadbeefu) out[0] = acc;
}

static inline uint slab_read(device const uchar *ptr, uint n_u4, ushort t) {
    device const uint4 *v = (device const uint4 *)ptr;
    uint acc = 0;
    for (uint i = t; i < n_u4; i += 128) { uint4 x = v[i]; acc += x.x ^ x.y ^ x.z ^ x.w; }
    return acc;
}

// One slab per threadgroup per (expert slot z); n_exp experts looped inside when grid z == 1.
kernel void k_flat(device const uchar *base [[buffer(0)]], device const int *ids [[buffer(1)]],
                   device uint *out [[buffer(2)]], constant P &p [[buffer(3)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    uint acc = 0;
    for (uint z = 0; z < p.n_exp; ++z) {
        const int e = ids[p.ids_off + (p.n_exp == 1 ? tg.z : z)];
        for (uint t = 0; t < p.n_ten; ++t)
            acc += slab_read(base + t * p.up_off + (ulong)e * p.expert_bytes + (ulong)tg.x * p.slab_u4 * 16, p.slab_u4, tid);
    }
    if (acc == 0xdeadbeefu) out[0] = acc;
}

kernel void k_seq(device const uchar *base [[buffer(0)]], device const int *ids [[buffer(1)]],
                  device uint *out [[buffer(2)]], constant P &p [[buffer(3)]],
                  uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    uint acc = 0;
    for (uint t = 0; t < p.n_ten; ++t)
        acc += slab_read(base + t * p.up_off + p.seq_base + (ulong)tg.x * p.slab_u4 * 16, p.slab_u4, tid);
    if (acc == 0xdeadbeefu) out[0] = acc;
}
)MSL";

typedef struct { uint64_t expert_bytes, up_off, seq_base; uint32_t row_bytes, nb32, ids_off, slab_u4, n_ten, n_exp; } P;

static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    @autoreleasepool {
        const int reps = argc > 1 ? atoi(argv[1]) : 7;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSDictionary *pso = nil;
        NSMutableDictionary *m = [NSMutableDictionary new];
        for (NSString *n in @[@"k_lane", @"k_coal", @"k_flat", @"k_seq"])
            m[n] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&err];
        pso = m;

        const uint64_t GU_TENSOR = 1167851520ull;    // 384 * 3041280
        const uint64_t GU_EXP = 3041280ull, DN_EXP = 3870720ull;
        const size_t total = 2 * GU_TENSOR;
        id<MTLBuffer> buf = [dev newBufferWithLength:total options:MTLResourceStorageModeShared];
        if (!buf) { fprintf(stderr, "alloc %zu failed\n", total); return 1; }
        { uint64_t *p = buf.contents; uint64_t s = 88172645463325252ull;
          for (size_t i = 0; i < total / 8; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; p[i] = s; } }
        id<MTLBuffer> outb = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        const int LAYERS = 40;
        id<MTLBuffer> idsb = [dev newBufferWithLength:LAYERS * 6 * 4 options:MTLResourceStorageModeShared];
        printf("device=%s buffer=%.2f GB reps=%d (40 dispatches per rep = one token)\n",
               dev.name.UTF8String, total / 1e9, reps);

        struct Case { const char *name; const char *kern; MTLSize grid; P p; double bytes; } cases[6];
        int nc = 0;
        P g = {.expert_bytes = GU_EXP, .up_off = GU_TENSOR, .row_bytes = 1320, .nb32 = 160, .slab_u4 = 1320, .n_ten = 2, .n_exp = 1};
        const double GU_BYTES = 6.0 * 2 * GU_EXP, DN_BYTES = 6.0 * DN_EXP;
        cases[nc++] = (struct Case){"gate/up lane (real load pattern)", "k_lane", MTLSizeMake(144, 1, 6), g, GU_BYTES};
        cases[nc++] = (struct Case){"gate/up coalesced 8B loads", "k_coal", MTLSizeMake(144, 1, 6), g, GU_BYTES};
        cases[nc++] = (struct Case){"gate/up flat 16B slabs", "k_flat", MTLSizeMake(144, 1, 6), g, GU_BYTES};
        P d = {.expert_bytes = DN_EXP, .up_off = 0, .row_bytes = 756, .slab_u4 = 378, .n_ten = 1, .n_exp = 6};
        cases[nc++] = (struct Case){"down flat 16B slabs (6 experts/TG)", "k_flat", MTLSizeMake(640, 1, 1), d, DN_BYTES};
        P sg = g; sg.seq_base = 0;
        cases[nc++] = (struct Case){"gate/up-size contiguous seq", "k_seq", MTLSizeMake(864, 1, 1), sg, GU_BYTES};
        P sd = d; sd.n_exp = 1; sd.seq_base = 0;
        cases[nc++] = (struct Case){"down-size contiguous seq", "k_seq", MTLSizeMake(3840, 1, 1), sd, DN_BYTES};

        for (int c = 0; c < nc; ++c) {
            double us[64]; int nus = 0;
            for (int rep = -1; rep < reps; ++rep) {   // rep -1 = warm-up
                int *ids = idsb.contents;
                for (int l = 0; l < LAYERS; ++l) for (int k = 0; k < 6; ++k) {
                    int e; do { e = (int)(arc4random() % 384); int dup = 0;
                                for (int j = 0; j < k; ++j) dup |= (ids[l * 6 + j] == e); if (!dup) break; } while (1);
                    ids[l * 6 + k] = e;
                }
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso[@(cases[c].kern)]];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc setBuffer:idsb offset:0 atIndex:1];
                [enc setBuffer:outb offset:0 atIndex:2];
                for (int l = 0; l < LAYERS; ++l) {
                    P p = cases[c].p; p.ids_off = (uint32_t)(l * 6);
                    if (!strcmp(cases[c].kern, "k_seq")) p.seq_base = ((uint64_t)l * 61000000ull / 16) * 16 % (GU_TENSOR - 70000000ull);
                    [enc setBytes:&p length:sizeof(p) atIndex:3];
                    [enc dispatchThreadgroups:cases[c].grid threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (rep >= 0 && nus < 64) us[nus++] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / LAYERS;
            }
            qsort(us, nus, sizeof(double), cmp);
            const double med = us[nus / 2];
            printf("%-38s %7.1f us/dispatch  %6.0f GB/s  (min %.1f, max %.1f)  %.2f ms/token\n", cases[c].name, med,
                   cases[c].bytes / med / 1e3, us[0], us[nus - 1], med * LAYERS / 1e3);
        }
    }
    return 0;
}
