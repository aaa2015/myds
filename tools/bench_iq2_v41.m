// Standalone IQ2_XXS shape/address screening. No myds/server initialization.
// Build: cc -O2 -fobjc-arc -framework Foundation -framework Metal tools/bench_iq2_v41.m -o
// /tmp/bench_iq2_v41 Shader: xcrun metal -mmacosx-version-min=15.0 -O3 -ffast-math -c
// metal/myds_iq2_v41.metal -o /tmp/iq2-v41.air
//         xcrun metallib /tmp/iq2-v41.air -o /tmp/iq2-v41.metallib
// Run: bench_iq2_v41 probe.metallib production.metallib NEW_OUTPUT_DIR [nsg=2] [trials=8]
// [samples=9] [candidate CSV] [pair=0|1] Actual popcount PSO is mandatory. Frozen clone must match
// it before candidates are timed. Pair mode appends actual production V41 Q2 down with NSG2. Seeded
// 384-expert tensor; forty disjoint expert sets, NOT a full-model run.
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(v)                                                                                 \
    do {                                                                                           \
        if (!(v)) {                                                                                \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #v);                           \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
struct args_mul_mv_id {
    int32_t nei0, nei1;
    uint64_t nbi1;
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13;
    uint64_t nb10, nb11, nb12;
    int32_t ne0, ne1;
    uint64_t nb1;
    int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base;
};
struct swiglu_args {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};
_Static_assert(sizeof(struct args_mul_mv_id) == 136 &&
                   offsetof(struct args_mul_mv_id, tp_expert_base) == 128,
               "Metal args ABI changed");
_Static_assert(sizeof(struct swiglu_args) == 48 && offsetof(struct swiglu_args, clamp_value) == 44,
               "Metal activation ABI changed");
enum { DIM = 5120, WIDTH = 2304, EXPERTS = 384, ACTIVE = 6, LAYERS = 40, GUARD = 64 };
static uint64_t rng_s = 88172645463325252ull;
static uint64_t rnd(void) {
    rng_s ^= rng_s << 13;
    rng_s ^= rng_s >> 7;
    rng_s ^= rng_s << 17;
    return rng_s;
}
static NSString *sha256(const void *p, size_t n) {
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    const uint8_t *s = p;
    while (n) {
        CC_LONG k = (CC_LONG)(n > 0x10000000u ? 0x10000000u : n);
        CC_SHA256_Update(&c, s, k);
        s += k;
        n -= k;
    }
    uint8_t d[32];
    CC_SHA256_Final(d, &c);
    char out[65];
    for (unsigned i = 0; i < 32; ++i)
        snprintf(out + 2 * i, 3, "%02x", d[i]);
    return [NSString stringWithUTF8String:out];
}
static void emit(NSDictionary *d) {
    NSError *e = nil;
    NSData *x = [NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:&e];
    REQUIRE(x);
    fwrite(x.bytes, 1, x.length, stdout);
    putchar('\n');
    fflush(stdout);
}
static void save(NSString *dir, NSMutableArray *m, NSString *n, const void *p, size_t bytes) {
    NSError *e = nil;
    NSData *d = [NSData dataWithBytes:p length:bytes];
    REQUIRE([d writeToFile:[dir stringByAppendingPathComponent:n]
                   options:NSDataWritingWithoutOverwriting
                     error:&e]);
    [m addObject:@{@"path" : n, @"bytes" : @(bytes), @"sha256" : sha256(p, bytes)}];
}
static void save_guards(NSString *dir, NSMutableArray *m, NSString *n, id<MTLBuffer> b,
                        size_t bytes) {
    uint8_t g[2 * GUARD];
    memcpy(g, b.contents, GUARD);
    memcpy(g + GUARD, (const uint8_t *)b.contents + GUARD + bytes, GUARD);
    save(dir, m, n, g, sizeof(g));
}
static void finish_manifest(NSString *dir, NSArray *m, NSDictionary *identity) {
    NSError *e = nil;
    NSData *d =
        [NSJSONSerialization dataWithJSONObject:@{
            @"schema_version" : @1,
            @"files" : m,
            @"identity" : identity
        }
                                        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                          error:&e];
    REQUIRE(d);
    REQUIRE([d writeToFile:[dir stringByAppendingPathComponent:@"artifact-manifest.json"]
                   options:NSDataWritingWithoutOverwriting
                     error:&e]);
}
static unsigned number(const char *s, unsigned lo, unsigned hi) {
    char *e = NULL;
    unsigned long n = strtoul(s, &e, 10);
    REQUIRE(*s && e && !*e && n >= lo && n <= hi);
    return (unsigned)n;
}
static BOOL guards(id<MTLBuffer> b, size_t n) {
    const uint8_t *p = b.contents;
    for (unsigned i = 0; i < GUARD; ++i)
        if (p[i] != 0xa5 || p[GUARD + n + i] != 0xa5)
            return NO;
    return YES;
}
static BOOL poison(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    for (size_t i = 0; i < b.length; ++i)
        if (p[i] != 0xa5)
            return NO;
    return YES;
}
static BOOL finite_f32(const void *p, size_t n) {
    const float *f = p;
    for (size_t i = 0; i < n / 4; ++i)
        if (!isfinite(f[i]))
            return NO;
    return YES;
}
static size_t mismatch(const void *a, const void *b, size_t n, size_t *first) {
    const uint32_t *x = a, *y = b;
    size_t bad = 0;
    *first = SIZE_MAX;
    for (size_t i = 0; i < n / 4; ++i)
        if (x[i] != y[i]) {
            if (!bad)
                *first = i;
            ++bad;
        }
    return bad;
}
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, unsigned n) {
    double a[64];
    REQUIRE(n <= 64);
    memcpy(a, v, n * sizeof(double));
    qsort(a, n, sizeof(double), cmpd);
    return n % 2 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
}
static void fill_iq2(void *ptr, size_t n) {
    uint8_t *p = ptr;
    REQUIRE(n % 66 == 0);
    for (size_t i = 0; i < n / 66; ++i) {
        uint8_t *b = p + i * 66;
        uint16_t d = (uint16_t)(0x2000u + rnd() % 0x1000u);
        if (i % 257u == 0)
            d = 0;
        memcpy(b, &d, 2);
        for (unsigned j = 2; j < 66; j += 8) {
            uint64_t v = rnd();
            memcpy(b + j, &v, 8);
        }
    }
}
static void fill_q2(void *ptr, size_t n) {
    uint8_t *p = ptr;
    REQUIRE(n % 84 == 0);
    for (size_t i = 0; i < n / 84; ++i) {
        uint8_t *b = p + i * 84;
        for (unsigned j = 0; j < 80; j += 8) {
            uint64_t v = rnd();
            memcpy(b + j, &v, 8);
        }
        uint16_t d = (uint16_t)(0x1800u + rnd() % 0x800u), m = (uint16_t)(0x1400u + rnd() % 0x800u);
        memcpy(b + 80, &d, 2);
        memcpy(b + 82, &m, 2);
    }
}
static void fill_inputs(id<MTLBuffer> x, id<MTLBuffer> ids, id<MTLBuffer> w, unsigned trial) {
    float *p = x.contents, *wp = w.contents;
    for (unsigned i = 0; i < LAYERS * DIM; ++i) {
        float v = ((int32_t)(rnd() % 2000001) - 1000000) / 1000000.f;
        if (trial % 4 == 1)
            v = ldexpf(v, (int)(i % 17) - 8);
        if (trial % 4 == 2)
            v = (i % 2 ? 1.f : -1.f) * (i % 23) / 23.f;
        if (trial % 4 == 3)
            v *= 16.f;
        if (i % 127 == 0)
            v = (i % 2 ? -0.f : 0.f);
        p[i] = v;
    }
    int32_t order[EXPERTS];
    for (unsigned i = 0; i < EXPERTS; ++i)
        order[i] = i;
    for (unsigned i = EXPERTS - 1; i > 0; --i) {
        unsigned j = rnd() % (i + 1);
        int32_t t = order[i];
        order[i] = order[j];
        order[j] = t;
    }
    memcpy(ids.contents, order, LAYERS * ACTIVE * 4u);
    for (unsigned i = 0; i < LAYERS * ACTIVE; ++i)
        wp[i] = (i % 61 == 0) ? 0.f : (0.01f + (rnd() % 1000) / 1000.f);
}
static BOOL eligible(const struct args_mul_mv_id *a, const struct swiglu_args *s, short nsg) {
    return (nsg == 2 || nsg == 4) && a->nei0 == 6 && a->nei1 == 1 && a->nbi1 == 24 &&
           a->ne00 == 5120 && a->ne01 == 2304 && a->ne02 == 384 && a->nb00 == 66 &&
           a->nb01 == 1320 && a->nb02 == 3041280 && a->ne10 == 5120 && a->ne11 == 1 &&
           a->ne12 == 1 && a->ne13 == 1 && a->nb10 == 4 && a->nb11 == 20480 && a->nb12 == 20480 &&
           a->ne0 == 2304 && a->ne1 == 6 && a->nb1 == 9216 && a->nr0 == 4 && a->tp_rank == 0 &&
           a->tp_world <= 1 && a->tp_world >= 0 && a->tp_addend == 0 && a->tp_expert_base == 0 &&
           s->width == 2304 && s->rows == 6 && s->gate_row_stride == 9216 &&
           s->up_row_stride == 9216 && s->mid_row_stride == 9216 && s->weight_stride == 4 &&
           s->write_clamped == 0 && isfinite(s->clamp_value) && s->clamp_value >= 0;
}
static int cpu_self_test(void) {
    struct args_mul_mv_id a = {.nei0 = 6,
                               .nei1 = 1,
                               .nbi1 = 24,
                               .ne00 = 5120,
                               .ne01 = 2304,
                               .ne02 = 384,
                               .nb00 = 66,
                               .nb01 = 1320,
                               .nb02 = 3041280,
                               .ne10 = 5120,
                               .ne11 = 1,
                               .ne12 = 1,
                               .ne13 = 1,
                               .nb10 = 4,
                               .nb11 = 20480,
                               .nb12 = 20480,
                               .ne0 = 2304,
                               .ne1 = 6,
                               .nb1 = 9216,
                               .nr0 = 4};
    struct swiglu_args act = {.width = 2304,
                              .rows = 6,
                              .gate_row_stride = 9216,
                              .up_row_stride = 9216,
                              .mid_row_stride = 9216,
                              .weight_stride = 4,
                              .clamp_value = 10};
    REQUIRE(eligible(&a, &act, 2) && eligible(&a, &act, 4));
    a.tp_world = 1;
    REQUIRE(eligible(&a, &act, 2));
    a.tp_world = 2;
    REQUIRE(!eligible(&a, &act, 2));
    a.tp_world = 0;
    a.nb02++;
    REQUIRE(!eligible(&a, &act, 2));
    a.nb02--;
    act.write_clamped = 1;
    REQUIRE(!eligible(&a, &act, 2));
    act.write_clamped = 0;
    uint8_t blocks[1320 + 2 * GUARD];
    memset(blocks, 0xa5, sizeof(blocks));
    fill_iq2(blocks + GUARD, 1320);
    for (unsigned i = 0; i < GUARD; ++i)
        REQUIRE(blocks[i] == 0xa5 && blocks[GUARD + 1320 + i] == 0xa5);
    for (unsigned b = 0; b < 20; ++b) {
        uint16_t h;
        memcpy(&h, blocks + GUARD + 66 * b, 2);
        REQUIRE((h & 0x7c00u) != 0x7c00u);
    }
    REQUIRE((uint64_t)383 * 3041280 + 2303 * 1320 + 1319 == 1167851519ull);
    emit(@{
        @"kind" : @"cpu_self_test",
        @"passed" : @YES,
        @"gpu_executed" : @NO,
        @"metal_device_created" : @NO
    });
    return 0;
}
static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, NSString *n, short nsg) {
    NSError *e = nil;
    MTLFunctionConstantValues *c = [MTLFunctionConstantValues new];
    [c setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> f = [l newFunctionWithName:n constantValues:c error:&e];
    if (!f)
        fprintf(stderr, "PSO %s: %s\n", n.UTF8String, e.localizedDescription.UTF8String);
    REQUIRE(f);
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e];
    REQUIRE(p && p.threadExecutionWidth == 32 &&
            p.maxTotalThreadsPerThreadgroup >= 32u * (unsigned)nsg);
    return p;
}
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc == 2 && !strcmp(argv[1], "--cpu-self-test"))
            return cpu_self_test();
        if (argc < 4 || argc > 9) {
            fprintf(stderr,
                    "usage: %s probe.metallib production.metallib NEW_OUTPUT_DIR [nsg 2|4] [trials "
                    "1..64] [samples 3..64] [candidate CSV] [pair 0|1]\n",
                    argv[0]);
            return 2;
        }
        const short nsg = argc > 4 ? (short)number(argv[4], 2, 4) : 2;
        REQUIRE(nsg == 2 || nsg == 4);
        const unsigned trials = argc > 5 ? number(argv[5], 1, 64) : 8,
                       samples = argc > 6 ? number(argv[6], 3, 64) : 9;
        const BOOL pair = argc > 8 ? number(argv[8], 0, 1) : NO;
        NSArray *candidates =
            argc > 7 ? [[NSString stringWithUTF8String:argv[7]] componentsSeparatedByString:@","]
                     : @[
                           @"kernel_myds_iq2_v41_shape64_f32", @"kernel_myds_iq2_v41_addr32_f32",
                           @"kernel_myds_iq2_v41_shape32_f32"
                       ];
        NSSet *allowed = [NSSet setWithArray:@[
            @"kernel_myds_iq2_v41_shape64_f32", @"kernel_myds_iq2_v41_addr32_f32",
            @"kernel_myds_iq2_v41_shape32_f32"
        ]];
        REQUIRE(candidates.count && candidates.count <= 3 &&
                [NSSet setWithArray:candidates].count == candidates.count);
        for (NSString *n in candidates)
            REQUIRE([allowed containsObject:n]);
        NSString *dir = [NSString stringWithUTF8String:argv[3]];
        NSError *error = nil;
        REQUIRE(![[NSFileManager defaultManager] fileExistsAtPath:dir]);
        REQUIRE([[NSFileManager defaultManager] createDirectoryAtPath:dir
                                          withIntermediateDirectories:YES
                                                           attributes:nil
                                                                error:&error]);
        NSString *probe_path = [NSString stringWithUTF8String:argv[1]],
                 *golden_path = [NSString stringWithUTF8String:argv[2]];
        NSData *probe_data = [NSData dataWithContentsOfFile:probe_path],
               *golden_data = [NSData dataWithContentsOfFile:golden_path];
        REQUIRE(probe_data && golden_data);
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        REQUIRE(d);
        id<MTLLibrary> probe = [d newLibraryWithURL:[NSURL fileURLWithPath:probe_path]
                                              error:&error],
                       golden = [d newLibraryWithURL:[NSURL fileURLWithPath:golden_path]
                                               error:&error];
        REQUIRE(probe && golden);
        NSString *actual = @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32",
                 *frozen = @"probe_iq2_v41_frozen_f32";
        NSMutableArray *names = [NSMutableArray arrayWithObjects:actual, frozen, nil];
        [names addObjectsFromArray:candidates];
        NSMutableArray *psos = [NSMutableArray new], *pso_identity = [NSMutableArray new];
        for (NSUInteger c = 0; c < names.count; ++c) {
            id<MTLComputePipelineState> p = pso(d, c == 0 ? golden : probe, names[c], nsg);
            [psos addObject:p];
            [pso_identity addObject:@{
                @"kernel" : names[c],
                @"nsg" : @(nsg),
                @"nr0" : @4,
                @"groups" : @(WIDTH / (4 * nsg)),
                @"pso_max_threads" : @(p.maxTotalThreadsPerThreadgroup),
                @"static_tg_bytes" : @(p.staticThreadgroupMemoryLength),
                @"actual_production_pso" : @(c == 0)
            }];
        }
        const size_t row = 1320, expert = 3041280, tensor = 1167851520, x_bytes = LAYERS * DIM * 4u,
                     id_bytes = LAYERS * ACTIVE * 4u, mid_layer = ACTIVE * WIDTH * 4u,
                     mid_bytes = LAYERS * mid_layer, out_layer = DIM * 4u,
                     out_bytes = LAYERS * out_layer;
        REQUIRE(tensor <= d.maxBufferLength);
        id<MTLBuffer> gate = [d newBufferWithLength:tensor options:MTLResourceStorageModeShared],
                      up = [d newBufferWithLength:tensor options:MTLResourceStorageModeShared],
                      x = [d newBufferWithLength:x_bytes options:MTLResourceStorageModeShared],
                      ids = [d newBufferWithLength:id_bytes options:MTLResourceStorageModeShared],
                      weights = [d newBufferWithLength:id_bytes
                                               options:MTLResourceStorageModeShared];
        REQUIRE(gate && up && x && ids && weights);
        fill_iq2(gate.contents, tensor);
        NSString *gate_sha = sha256(gate.contents, tensor);
        fill_iq2(up.contents, tensor);
        NSString *up_sha = sha256(up.contents, tensor);
        NSMutableArray *mid_outputs = [NSMutableArray new], *gate_outputs = [NSMutableArray new],
                       *up_outputs = [NSMutableArray new], *down_outputs = [NSMutableArray new];
        for (NSUInteger c = 0; c < names.count; ++c) {
            for (NSMutableArray *a in @[ mid_outputs, gate_outputs, up_outputs ]) {
                id<MTLBuffer> b = [d newBufferWithLength:mid_bytes + 2 * GUARD
                                                 options:MTLResourceStorageModeShared];
                REQUIRE(b);
                memset(b.contents, 0xa5, b.length);
                [a addObject:b];
            }
            if (pair) {
                id<MTLBuffer> b = [d newBufferWithLength:out_bytes + 2 * GUARD
                                                 options:MTLResourceStorageModeShared];
                REQUIRE(b);
                memset(b.contents, 0xa5, b.length);
                [down_outputs addObject:b];
            }
        }
        id<MTLBuffer> down_weights = nil;
        id<MTLComputePipelineState> down_pso = nil;
        NSString *down_sha = @"", *down_symbol = @"kernel_mul_mv_id_q2_K_sum6_v41_f32";
        const short down_nsg = 2;
        if (pair) {
            down_pso = pso(d, golden, down_symbol, down_nsg);
            const size_t n = (size_t)EXPERTS * DIM * 9u * 84u;
            REQUIRE(n <= d.maxBufferLength);
            down_weights = [d newBufferWithLength:n options:MTLResourceStorageModeShared];
            REQUIRE(down_weights);
            fill_q2(down_weights.contents, n);
            down_sha = sha256(down_weights.contents, n);
        }
        __block struct args_mul_mv_id a = {.nei0 = 6,
                                           .nei1 = 1,
                                           .nbi1 = 24,
                                           .ne00 = 5120,
                                           .ne01 = 2304,
                                           .ne02 = 384,
                                           .nb00 = 66,
                                           .nb01 = row,
                                           .nb02 = expert,
                                           .ne10 = 5120,
                                           .ne11 = 1,
                                           .ne12 = 1,
                                           .ne13 = 1,
                                           .nb10 = 4,
                                           .nb11 = 20480,
                                           .nb12 = 20480,
                                           .ne0 = 2304,
                                           .ne1 = 6,
                                           .nb1 = 9216,
                                           .nr0 = 4};
        __block struct swiglu_args act = {.width = 2304,
                                          .rows = 6,
                                          .gate_row_stride = 9216,
                                          .up_row_stride = 9216,
                                          .mid_row_stride = 9216,
                                          .weight_stride = 4,
                                          .write_clamped = 0,
                                          .clamp_value = 10.f};
        REQUIRE(eligible(&a, &act, nsg));
        // CPU-only host eligibility negative controls; no dispatch is allowed for them.
        unsigned host_rejects = 0;
        struct args_mul_mv_id bad = a;
        bad.ne00 = 4096;
        host_rejects += !eligible(&bad, &act, nsg);
        bad = a;
        bad.nei1 = 2;
        host_rejects += !eligible(&bad, &act, nsg);
        bad = a;
        bad.nei0 = 5;
        host_rejects += !eligible(&bad, &act, nsg);
        bad = a;
        bad.tp_world = 2;
        host_rejects += !eligible(&bad, &act, nsg);
        bad = a;
        bad.tp_expert_base = 1;
        host_rejects += !eligible(&bad, &act, nsg);
        bad = a;
        bad.nb01++;
        host_rejects += !eligible(&bad, &act, nsg);
        struct swiglu_args badact = act;
        badact.write_clamped = 1;
        host_rejects += !eligible(&a, &badact, nsg);
        badact = act;
        badact.mid_row_stride++;
        host_rejects += !eligible(&a, &badact, nsg);
        REQUIRE(host_rejects == 8);
        id<MTLCommandQueue> q = [d newCommandQueue];
        REQUIRE(q);
        id<MTLCommandBuffer> (^run)(NSUInteger, BOOL) =
            ^id<MTLCommandBuffer>(NSUInteger c, BOOL do_pair) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            REQUIRE(cb);
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            REQUIRE(e);
            for (unsigned l = 0; l < LAYERS; ++l) {
                [e setComputePipelineState:psos[c]];
                [e setBytes:&a length:sizeof(a) atIndex:0];
                [e setBytes:&act length:sizeof(act) atIndex:1];
                [e setBuffer:gate offset:0 atIndex:2];
                [e setBuffer:up offset:0 atIndex:3];
                [e setBuffer:x offset:l * DIM * 4u atIndex:4];
                [e setBuffer:gate_outputs[c] offset:GUARD + l * mid_layer atIndex:5];
                [e setBuffer:up_outputs[c] offset:GUARD + l * mid_layer atIndex:6];
                [e setBuffer:mid_outputs[c] offset:GUARD + l * mid_layer atIndex:7];
                [e setBuffer:ids offset:l * ACTIVE * 4u atIndex:8];
                [e setBuffer:weights offset:l * ACTIVE * 4u atIndex:9];
                [e setThreadgroupMemoryLength:2176 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(WIDTH / (4 * nsg), 1, ACTIVE)
                    threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                if (do_pair) {
                    struct args_mul_mv_id da = {.nei0 = 6,
                                                .nei1 = 1,
                                                .nbi1 = 24,
                                                .ne00 = 2304,
                                                .ne01 = 5120,
                                                .ne02 = 384,
                                                .nb00 = 84,
                                                .nb01 = 756,
                                                .nb02 = 3870720,
                                                .ne10 = 2304,
                                                .ne11 = 6,
                                                .ne12 = 1,
                                                .ne13 = 1,
                                                .nb10 = 4,
                                                .nb11 = 9216,
                                                .nb12 = 55296,
                                                .ne0 = 5120,
                                                .ne1 = 1,
                                                .nb1 = 20480,
                                                .nr0 = 2,
                                                .tp_world = 1};
                    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [e setComputePipelineState:down_pso];
                    [e setBytes:&da length:sizeof(da) atIndex:0];
                    [e setBuffer:down_weights offset:0 atIndex:1];
                    [e setBuffer:mid_outputs[c] offset:GUARD + l * mid_layer atIndex:2];
                    [e setBuffer:down_outputs[c] offset:GUARD + l * out_layer atIndex:3];
                    [e setBuffer:ids offset:l * ACTIVE * 4u atIndex:4];
                    [e setBuffer:down_outputs[c] offset:GUARD + l * out_layer atIndex:5];
                    [e setThreadgroupMemoryLength:16 atIndex:0];
                    [e dispatchThreadgroups:MTLSizeMake(DIM / (2 * down_nsg), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, down_nsg, 1)];
                }
            }
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted)
                fprintf(stderr, "GPU failure: %s\n", cb.error.localizedDescription.UTF8String);
            REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
            return cb;
        };
        NSDictionary *identity = @{
            @"kind" : @"identity",
            @"device" : d.name,
            @"nsg" : @(nsg),
            @"fc600" : @(nsg),
            @"dim" : @(DIM),
            @"width" : @(WIDTH),
            @"experts" : @(EXPERTS),
            @"active" : @(ACTIVE),
            @"layers" : @(LAYERS),
            @"nr0" : @4,
            @"tensor_bytes" : @(tensor),
            @"working_set_bytes_per_arm" : @(LAYERS * ACTIVE * expert * 2u),
            @"gate_sha256" : gate_sha,
            @"up_sha256" : up_sha,
            @"probe_library_sha256" : sha256(probe_data.bytes, probe_data.length),
            @"golden_library_sha256" : sha256(golden_data.bytes, golden_data.length),
            @"actual_production_kernel" : actual,
            @"frozen_symbol" : frozen,
            @"frozen_helper_sha256" :
                @"8344379eda8be1389fb0722ff8c4e7e1eb7a0667afeeda4208bed5c7bea27a75",
            @"psos" : pso_identity,
            @"candidates" : candidates,
            @"pair" : @(pair),
            @"down_actual_symbol" : pair ? (id)down_symbol : [NSNull null],
            @"down_nsg" : @(down_nsg),
            @"down_weights_sha256" : pair ? (id)down_sha : [NSNull null],
            @"trials" : @(trials),
            @"samples" : @(samples),
            @"host_rejected_cases_without_dispatch" : @(host_rejects),
            @"seed" : @88172645463325252ull,
            @"seeded_fixture_not_gguf" : @YES,
            @"fullmodel" : @NO,
            @"cold_dram_proven" : @NO,
            @"disjoint_experts_within_arm" : @YES,
            @"encoder_dispatch_type" : @"serial"
        };
        emit(identity);
        NSMutableArray *manifest = [NSMutableArray new];
        BOOL exact = YES;
        for (unsigned t = 0; t < trials; ++t) {
            fill_inputs(x, ids, weights, t);
            a.tp_world = t % 2;
            act.clamp_value = t % 3 == 0 ? 0.f : 10.f;
            REQUIRE(eligible(&a, &act, nsg));
            NSString *prefix = [NSString stringWithFormat:@"parity-%02u", t];
            save(dir, manifest, [prefix stringByAppendingString:@"-x.f32"], x.contents, x_bytes);
            save(dir, manifest, [prefix stringByAppendingString:@"-ids.i32"], ids.contents,
                 id_bytes);
            save(dir, manifest, [prefix stringByAppendingString:@"-weights.f32"], weights.contents,
                 id_bytes);
            for (NSUInteger c = 0; c < names.count; ++c) {
                id<MTLBuffer> mid = mid_outputs[c];
                memset(mid.contents, 0xa5, mid.length);
                if (pair) {
                    id<MTLBuffer> out = down_outputs[c];
                    memset(out.contents, 0xa5, out.length);
                }
                (void)run(c, pair);
                const uint8_t *p = (const uint8_t *)mid.contents + GUARD;
                id<MTLBuffer> ref = mid_outputs[0];
                size_t first,
                    bads = mismatch((const uint8_t *)ref.contents + GUARD, p, mid_bytes, &first);
                BOOL good = guards(mid, mid_bytes) && finite_f32(p, mid_bytes) &&
                            poison(gate_outputs[c]) && poison(up_outputs[c]) && !bads;
                save(dir, manifest, [NSString stringWithFormat:@"%@-%@-mid.f32", prefix, names[c]],
                     p, mid_bytes);
                save_guards(dir, manifest,
                            [NSString stringWithFormat:@"%@-%@-mid.guards", prefix, names[c]], mid,
                            mid_bytes);
                size_t pair_bad = 0;
                BOOL pair_good = YES;
                if (pair) {
                    id<MTLBuffer> out = down_outputs[c], r = down_outputs[0];
                    const uint8_t *op = (const uint8_t *)out.contents + GUARD;
                    size_t pf;
                    pair_bad = mismatch((const uint8_t *)r.contents + GUARD, op, out_bytes, &pf);
                    pair_good = guards(out, out_bytes) && finite_f32(op, out_bytes) && !pair_bad;
                    save(dir, manifest,
                         [NSString stringWithFormat:@"%@-%@-out.f32", prefix, names[c]], op,
                         out_bytes);
                    save_guards(dir, manifest,
                                [NSString stringWithFormat:@"%@-%@-out.guards", prefix, names[c]],
                                out, out_bytes);
                }
                exact &= good && pair_good;
                emit(@{
                    @"kind" : @"parity",
                    @"trial" : @(t),
                    @"kernel" : names[c],
                    @"tp_world" : @(a.tp_world),
                    @"clamp" : @(act.clamp_value),
                    @"layers" : @(LAYERS),
                    @"f32_per_layer" : @(mid_layer / 4),
                    @"f32_total" : @(mid_bytes / 4),
                    @"mismatches" : @(bads),
                    @"first_mismatch" : first == SIZE_MAX ? (id)[NSNull null] : @(first),
                    @"full_f32_bit_exact" : @(good),
                    @"finite" : @(finite_f32(p, mid_bytes)),
                    @"guards" : @(guards(mid, mid_bytes)),
                    @"release_sg_su_poison" : @(poison(gate_outputs[c]) && poison(up_outputs[c])),
                    @"actual_production_pso" : @(c == 0),
                    @"comparison_reference_symbol" : actual,
                    @"pair_mismatches" : @(pair_bad),
                    @"pair_full_f32_bit_exact" : @(pair_good),
                    @"completed_cb" : @YES,
                    @"sha256" : sha256(p, mid_bytes)
                });
                if (!good || !pair_good)
                    break;
            }
            if (!exact)
                break;
        }
        if (!exact) {
            finish_manifest(dir, manifest, identity);
            return 1;
        }
        // Never run the production oracle on invalid IDs: its historical TP<=1
        // helper does not check them. Candidate guards must leave the whole MID poison.
        const int32_t invalid[] = {-1, 384, INT32_MIN, INT32_MAX, -2, 385};
        for (unsigned l = 0; l < LAYERS; ++l)
            memcpy((uint8_t *)ids.contents + l * 24u, invalid, 24u);
        save(dir, manifest, @"negative-ids.i32", ids.contents, id_bytes);
        for (NSUInteger c = 2; c < names.count; ++c) {
            id<MTLBuffer> b = mid_outputs[c];
            memset(b.contents, 0xa5, b.length);
            (void)run(c, NO);
            REQUIRE(poison(b));
            save(dir, manifest, [NSString stringWithFormat:@"negative-ids-%@-mid.raw", names[c]],
                 b.contents, b.length);
            emit(@{
                @"kind" : @"negative",
                @"case" : @"invalid_ids",
                @"kernel" : names[c],
                @"all_bytes_poison" : @YES,
                @"completed_cb" : @YES
            });
        }
        a.tp_world = 2;
        for (NSUInteger c = 2; c < names.count; ++c) {
            id<MTLBuffer> b = mid_outputs[c];
            memset(b.contents, 0xa5, b.length);
            (void)run(c, NO);
            REQUIRE(poison(b));
            save(dir, manifest, [NSString stringWithFormat:@"negative-tp-%@-mid.raw", names[c]],
                 b.contents, b.length);
            emit(@{
                @"kind" : @"negative",
                @"case" : @"split_tp",
                @"kernel" : names[c],
                @"all_bytes_poison" : @YES,
                @"completed_cb" : @YES
            });
        }
        a.tp_world = 0;
        act.clamp_value = 10.f;
        fill_inputs(x, ids, weights, 0);
        REQUIRE(eligible(&a, &act, nsg));
        (void)run(0, pair);
        id<MTLBuffer> gm = mid_outputs[0];
        NSData *timing_ref = [NSData dataWithBytes:(const uint8_t *)gm.contents + GUARD
                                            length:mid_bytes];
        NSData *pair_ref = nil;
        if (pair) {
            id<MTLBuffer> o = down_outputs[0];
            pair_ref = [NSData dataWithBytes:(const uint8_t *)o.contents + GUARD length:out_bytes];
        }
        save(dir, manifest, @"timing-x.f32", x.contents, x_bytes);
        save(dir, manifest, @"timing-ids.i32", ids.contents, id_bytes);
        save(dir, manifest, @"timing-weights.f32", weights.contents, id_bytes);
        for (NSUInteger c = 2; c < names.count; ++c) {
            double med[3];
            for (unsigned arm = 0; arm < 3; ++arm) {
                NSUInteger idx = arm == 1 ? c : 0;
                NSString *label = arm == 0 ? @"A1" : arm == 1 ? @"B" : @"A2";
                double times[64];
                for (unsigned sample = 0; sample <= samples; ++sample) {
                    id<MTLCommandBuffer> cb = run(idx, pair);
                    REQUIRE(isfinite(cb.GPUStartTime) && isfinite(cb.GPUEndTime) &&
                            cb.GPUEndTime > cb.GPUStartTime);
                    id<MTLBuffer> b = mid_outputs[idx];
                    const uint8_t *p = (const uint8_t *)b.contents + GUARD;
                    REQUIRE(guards(b, mid_bytes) && finite_f32(p, mid_bytes) &&
                            !memcmp(timing_ref.bytes, p, mid_bytes) && poison(gate_outputs[idx]) &&
                            poison(up_outputs[idx]));
                    if (pair) {
                        id<MTLBuffer> o = down_outputs[idx];
                        const uint8_t *op = (const uint8_t *)o.contents + GUARD;
                        REQUIRE(guards(o, out_bytes) && finite_f32(op, out_bytes) &&
                                !memcmp(pair_ref.bytes, op, out_bytes));
                    }
                    double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / LAYERS;
                    emit(@{
                        @"kind" : @"timing",
                        @"candidate" : names[c],
                        @"kernel" : names[idx],
                        @"arm" : label,
                        @"sample" : @(sample),
                        @"warmup" : @(sample == 0),
                        @"gpu_start_s" : @(cb.GPUStartTime),
                        @"gpu_end_s" : @(cb.GPUEndTime),
                        @"us_per_layer" : @(us),
                        @"layers" : @(LAYERS),
                        @"dispatches" : @(LAYERS * (pair ? 2 : 1)),
                        @"pair" : @(pair),
                        @"actual_production_pso" : @(idx == 0),
                        @"completed_cb" : @YES,
                        @"full_f32_bit_exact" : @YES,
                        @"production_math_oracle_validated" : @YES
                    });
                    if (sample)
                        times[sample - 1] = us;
                }
                med[arm] = median(times, samples);
                id<MTLBuffer> b = mid_outputs[idx];
                save(dir, manifest,
                     [NSString stringWithFormat:@"timing-%@-%@-mid.f32", names[c], label],
                     (const uint8_t *)b.contents + GUARD, mid_bytes);
                save_guards(dir, manifest,
                            [NSString stringWithFormat:@"timing-%@-%@-mid.guards", names[c], label],
                            b, mid_bytes);
                if (pair) {
                    id<MTLBuffer> o = down_outputs[idx];
                    save(dir, manifest,
                         [NSString stringWithFormat:@"timing-%@-%@-out.f32", names[c], label],
                         (const uint8_t *)o.contents + GUARD, out_bytes);
                    save_guards(
                        dir, manifest,
                        [NSString stringWithFormat:@"timing-%@-%@-out.guards", names[c], label], o,
                        out_bytes);
                }
            }
            double control = (med[0] + med[2]) / 2;
            emit(@{
                @"kind" : @"summary",
                @"candidate" : names[c],
                @"a1_us" : @(med[0]),
                @"b_us" : @(med[1]),
                @"a2_us" : @(med[2]),
                @"control_drift_percent" : @((med[2] / med[0] - 1) * 100),
                @"time_saved_percent" : @((1 - med[1] / control) * 100),
                @"conservative_time_saved_percent" : @((1 - med[1] / fmin(med[0], med[2])) * 100),
                @"estimated_ms_saved_per_40_layers" : @((control - med[1]) * LAYERS / 1000),
                @"fullmodel_speedup_proven" : @NO,
                @"pair" : @(pair),
                @"control_kernel" : actual,
                @"production_math_oracle_validated" : @YES
            });
        }
        finish_manifest(dir, manifest, identity);
        emit(@{
            @"kind" : @"complete",
            @"full_f32_bit_exact" : @YES,
            @"raw_files" : @(manifest.count),
            @"production_math_oracle_validated" : @YES,
            @"fullmodel" : @NO
        });
    }
    return 0;
}
