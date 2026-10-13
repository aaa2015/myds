// Native full SharedDown + IQ2 GU + Q2 Down + BF16 add component, real GGUF.
// Independent process per arm: overlap0 A1, overlap1 B, overlap0 A2.
// CLI: bench_moe_down_overlap MODEL.gguf LIBRARY_DIR NEW_DIR MODE[0|1] [samples=5] [cases=8]
// [repetitions=5] LIBRARY_DIR/myds.metallib is the required actual precompiled library. No full
// model/context/server is initialized; only four layer0 tensors map. Build against src/myds_metal.m
// src/myds_cpu.c src/myds_gguf.c src/myds_log.c.
#include "myds_gguf.h"
#include "myds_metal.h"
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
// Root owns the native implementations; these getters never submit work.
extern int myds_metal_last_completed_gpu_span(double *start_s, double *end_s);
extern const char *myds_metal_loaded_library_path(void);
#define REQUIRE(v)                                                                                 \
    do {                                                                                           \
        if (!(v)) {                                                                                \
            fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #v);                            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
enum { DIM = 5120, WIDTH = 2304, EXPERTS = 384, ACTIVE = 6, GUARD = 64 };
static uint64_t rng = 88172645463325252ull;
static uint64_t rnd(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}
static double monotonic(void) {
    struct timespec t;
    REQUIRE(!clock_gettime(CLOCK_MONOTONIC, &t));
    return t.tv_sec + t.tv_nsec * 1e-9;
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
    for (unsigned i = 0; i < 32; i++)
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
static unsigned number(const char *s, unsigned lo, unsigned hi) {
    char *e = NULL;
    unsigned long n = strtoul(s, &e, 10);
    REQUIRE(*s && e && !*e && n >= lo && n <= hi);
    return (unsigned)n;
}
static void env(const char *n, const char *v) { REQUIRE(!setenv(n, v, 1)); }
static BOOL finite_f32(const void *p, size_t n) {
    const float *f = p;
    for (size_t i = 0; i < n / 4; i++)
        if (!isfinite(f[i]))
            return NO;
    return YES;
}
static BOOL valid_ids(const int32_t *p) {
    for (unsigned i = 0; i < 6; i++) {
        if (p[i] < 0 || p[i] >= 384)
            return NO;
        for (unsigned j = 0; j < i; j++)
            if (p[i] == p[j])
                return NO;
    }
    return YES;
}
static void save(NSString *dir, NSMutableArray *m, NSString *name, const void *p, size_t n) {
    NSError *e = nil;
    NSData *d = [NSData dataWithBytes:p length:n];
    REQUIRE([d writeToFile:[dir stringByAppendingPathComponent:name]
                   options:NSDataWritingWithoutOverwriting
                     error:&e]);
    [m addObject:@{@"path" : name, @"bytes" : @(n), @"sha256" : sha256(p, n)}];
}
static NSDictionary *q2_counts(myds_metal_moe_q2_v41_counters_t c) {
    return @{@"requests" : @(c.requests), @"encoded" : @(c.encoded), @"failures" : @(c.failures)};
}
static NSDictionary *counts(myds_metal_decode_candidate_counters_t c) {
    return @{@"requests" : @(c.requests), @"encoded" : @(c.encoded), @"failures" : @(c.failures)};
}
typedef struct {
    myds_metal_buffer_t *owner, *view;
    size_t bytes;
} guarded;
static guarded alloc_guarded(size_t n) {
    guarded g = {.owner = myds_metal_buffer_alloc(n + 2 * GUARD), .bytes = n};
    REQUIRE(g.owner);
    memset(myds_metal_buffer_data(g.owner), 0xa5, n + 2 * GUARD);
    g.view = myds_metal_buffer_sub_view(g.owner, GUARD, n);
    REQUIRE(g.view);
    return g;
}
static BOOL guard_ok(guarded g) {
    const uint8_t *p = myds_metal_buffer_data(g.owner);
    for (unsigned i = 0; i < GUARD; i++)
        if (p[i] != 0xa5 || p[GUARD + g.bytes + i] != 0xa5)
            return NO;
    return YES;
}
static BOOL all_poison(guarded g) {
    const uint8_t *p = myds_metal_buffer_data(g.owner);
    for (size_t i = 0; i < g.bytes + 2 * GUARD; i++)
        if (p[i] != 0xa5)
            return NO;
    return YES;
}
static void free_guard(guarded g) {
    myds_metal_buffer_free(g.view);
    myds_metal_buffer_free(g.owner);
}
typedef struct {
    myds_metal_buffer_t *x, *shared_input, *ids, *weights;
    guarded mid, moe_out, shared_out, sum, sg, su, down;
} fixture;
static fixture make_fixture(unsigned index) {
    fixture c = {0};
    c.x = myds_metal_buffer_alloc(DIM * 4u);
    c.shared_input = myds_metal_buffer_alloc(WIDTH * 4u);
    c.ids = myds_metal_buffer_alloc(24);
    c.weights = myds_metal_buffer_alloc(24);
    REQUIRE(c.x && c.shared_input && c.ids && c.weights);
    float *x = myds_metal_buffer_data(c.x), *sh = myds_metal_buffer_data(c.shared_input),
          *w = myds_metal_buffer_data(c.weights);
    int32_t *ids = myds_metal_buffer_data(c.ids);
    for (unsigned i = 0; i < DIM; i++)
        x[i] = ((int32_t)(rnd() % 2000001) - 1000000) / 1000000.f * (index % 3 + 1);
    for (unsigned i = 0; i < WIDTH; i++) {
        float f = ((int32_t)(rnd() % 2000001) - 1000000) / 1000000.f;
        uint32_t u;
        memcpy(&u, &f, 4);
        u = (u + 0x7fffu + ((u >> 16) & 1)) & 0xffff0000u;
        memcpy(sh + i, &u, 4);
    }
    ids[0] = 0;
    ids[1] = 383;
    for (unsigned i = 2; i < 6; i++) {
        BOOL dup;
        do {
            ids[i] = rnd() % 384;
            dup = NO;
            for (unsigned j = 0; j < i; j++)
                dup |= ids[i] == ids[j];
        } while (dup);
    }
    for (unsigned i = 0; i < 6; i++)
        w[i] = 0.1f + (rnd() % 1000) / 1000.f;
    REQUIRE(valid_ids(ids) && finite_f32(x, DIM * 4u) && finite_f32(sh, WIDTH * 4u));
    c.mid = alloc_guarded(6 * WIDTH * 4u);
    c.moe_out = alloc_guarded(DIM * 4u);
    c.shared_out = alloc_guarded(DIM * 4u);
    c.sum = alloc_guarded(DIM * 4u);
    c.sg = alloc_guarded(6 * WIDTH * 4u);
    c.su = alloc_guarded(6 * WIDTH * 4u);
    c.down = alloc_guarded(6 * DIM * 4u);
    return c;
}
static void free_fixture(fixture c) {
    myds_metal_buffer_free(c.x);
    myds_metal_buffer_free(c.shared_input);
    myds_metal_buffer_free(c.ids);
    myds_metal_buffer_free(c.weights);
    free_guard(c.mid);
    free_guard(c.moe_out);
    free_guard(c.shared_out);
    free_guard(c.sum);
    free_guard(c.sg);
    free_guard(c.su);
    free_guard(c.down);
}
static int overlap_call(fixture *c, myds_metal_buffer_t **t, myds_metal_buffer_t *shared) {
    return myds_metal_routed_moe_down_overlap_scoped(
        c->moe_out.view, c->x, t[0], t[1], t[2], c->ids, c->weights, c->sg.view, c->su.view,
        c->mid.view, c->down.view, DIM, WIDTH, EXPERTS, ACTIVE, GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.f, true, shared, t[3], c->shared_input,
        GGML_TYPE_Q4_K);
}
static BOOL native_case(fixture *c, myds_metal_buffer_t **t, unsigned mode) {
    if (mode) {
        if (!overlap_call(c, t, c->shared_out.view))
            return NO;
    } else {
        if (!myds_metal_v41_gemv_bf16(c->shared_out.view, t[3], c->shared_input, DIM, WIDTH,
                                      GGML_TYPE_Q4_K))
            return NO;
        myds_metal_barrier_buffer(c->shared_out.view);
        if (!myds_metal_routed_moe_scoped(
                c->moe_out.view, c->x, t[0], t[1], t[2], c->ids, c->weights, c->sg.view, c->su.view,
                c->mid.view, c->down.view, DIM, WIDTH, EXPERTS, ACTIVE, GGML_TYPE_IQ2_XXS,
                GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.f, true))
            return NO;
    }
    const myds_metal_buffer_t *deps[] = {c->moe_out.view, c->shared_out.view};
    myds_metal_barrier_buffers(deps, 2);
    if (!myds_metal_v41_add_bf16(c->sum.view, c->moe_out.view, c->shared_out.view, DIM))
        return NO;
    myds_metal_barrier_buffer(c->sum.view);
    return YES;
}
static void verify_case(fixture *c) {
    guarded g[] = {c->mid, c->moe_out, c->shared_out, c->sum};
    for (unsigned i = 0; i < 4; i++)
        REQUIRE(guard_ok(g[i]) && finite_f32(myds_metal_buffer_data(g[i].view), g[i].bytes));
    REQUIRE(all_poison(c->sg) && all_poison(c->su) && all_poison(c->down));
}
static void dump_case(NSString *dir, NSMutableArray *m, fixture *c, unsigned index,
                      NSString *prefix) {
    NSString *name = [NSString stringWithFormat:@"%@-case-%02u", prefix, index];
    guarded g[] = {c->mid, c->moe_out, c->shared_out, c->sum};
    NSArray *fields = @[ @"mid", @"moe", @"shared", @"sum_bf16" ];
    for (unsigned i = 0; i < 4; i++) {
        save(dir, m, [NSString stringWithFormat:@"%@-%@.f32", name, fields[i]],
             myds_metal_buffer_data(g[i].view), g[i].bytes);
        uint8_t guard[128];
        memcpy(guard, myds_metal_buffer_data(g[i].owner), 64);
        memcpy(guard + 64, (const uint8_t *)myds_metal_buffer_data(g[i].owner) + 64 + g[i].bytes,
               64);
        save(dir, m, [NSString stringWithFormat:@"%@-%@.guards", name, fields[i]], guard, 128);
    }
}
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 5 || argc > 8) {
            fprintf(stderr,
                    "usage: %s MODEL.gguf LIBRARY_DIR NEW_DIR MODE[0|1] [samples=5] [cases=8] "
                    "[repetitions=5]\n",
                    argv[0]);
            return 2;
        }
        unsigned mode = number(argv[4], 0, 1), samples = argc > 5 ? number(argv[5], 3, 31) : 5,
                 cases = argc > 6 ? number(argv[6], 1, 32) : 8,
                 reps = argc > 7 ? number(argv[7], 1, 32) : 5;
        NSString *model = [NSString stringWithUTF8String:argv[1]],
                 *libdir = [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath],
                 *dir = [NSString stringWithUTF8String:argv[3]];
        REQUIRE([libdir isAbsolutePath]);
        NSString *expected_lib = [libdir stringByAppendingPathComponent:@"myds.metallib"];
        NSData *library = [NSData dataWithContentsOfFile:expected_lib];
        REQUIRE(library);
        NSString *required_sha = sha256(library.bytes, library.length);
        NSError *e = nil;
        REQUIRE(![[NSFileManager defaultManager] fileExistsAtPath:dir]);
        REQUIRE([[NSFileManager defaultManager] createDirectoryAtPath:dir
                                          withIntermediateDirectories:YES
                                                           attributes:nil
                                                                error:&e]);
        env("MYDS_MOE_DOWN_OVERLAP", mode ? "1" : "0");
        env("MYDS_MOE_IQ2_V41", "0");
        env("MYDS_DENSE_Q4_V41", "0");
        env("MYDS_MOE_Q2_V41", "1");
        env("MYDS_ROUTE_SIMD_SHUFFLE", "1");
        env("MYDS_ATTN_PACKED_REGISTER", "1");
        env("MYDS_MOE_IQ2_POPCNT_SIGN", "1");
        env("MYDS_MOE_IQ2_CONSTANT_LUT", "0");
        env("MYDS_MOE_NSG", "2");
        env("MYDS_MOE_DOWN_NSG", "2");
        env("MYDS_DENSE_Q4_NR0", "2");
        env("MYDS_DENSE_Q4_NSG", "4");
        env("MYDS_MOE_Q2_SUM_CACHE", "0");
        env("MYDS_MOE_Q2_SUM_CACHE_FUSED", "0");
        env("MYDS_MOE_MM_MIN_TOKENS", "0");
        env("MYDS_METAL_CONCURRENT", "1");
        env("MYDS_KPROF", "0");
        env("MYDS_MOE_STEP_PROF", "0");
        env("MYDS_NO_RESIDENCY_HB", "1");
        struct stat st;
        REQUIRE(!stat(model.fileSystemRepresentation, &st));
        gguf_file_t *gf = gguf_open(model.fileSystemRepresentation);
        REQUIRE(gf);
        NSArray *names = @[
            @"blk.0.ffn_gate_exps.weight", @"blk.0.ffn_up_exps.weight",
            @"blk.0.ffn_down_exps.weight", @"blk.0.ffn_down_shexp.weight"
        ];
        const uint64_t dims[4][3] = {
            {5120, 2304, 384}, {5120, 2304, 384}, {2304, 5120, 384}, {2304, 5120, 1}};
        const size_t sizes[] = {1167851520ull, 1167851520ull, 1486356480ull, 6635520ull};
        const ggml_type types[] = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K,
                                   GGML_TYPE_Q4_K};
        const gguf_tensor_t *td[4];
        NSMutableArray *descriptors = [NSMutableArray new];
        for (unsigned i = 0; i < 4; i++) {
            td[i] = gguf_find_tensor(gf, [names[i] UTF8String]);
            REQUIRE(td[i] && td[i]->type == types[i] && td[i]->n_dims == (i == 3 ? 2 : 3) &&
                    td[i]->size_bytes == sizes[i]);
            for (unsigned j = 0; j < td[i]->n_dims; j++)
                REQUIRE(td[i]->dims[j] == dims[i][j]);
            [descriptors addObject:@{
                @"name" : names[i],
                @"type" : @(types[i]),
                @"bytes" : @(sizes[i]),
                @"absolute_offset" : @(gf->tensor_data_offset + td[i]->offset),
                @"sha256" : sha256(td[i]->data, sizes[i])
            }];
        }
        REQUIRE(myds_metal_init(libdir.fileSystemRepresentation));
        REQUIRE(myds_metal_moe_down_overlap_enabled() == (int)mode &&
                myds_metal_moe_q2_v41_enabled() == 1 && myds_metal_moe_iq2_v41_enabled() == 0 &&
                myds_metal_dense_q4_v41_enabled() == 0 && !myds_metal_kprof_enabled());
        const char *actualpath = myds_metal_loaded_library_path();
        REQUIRE(actualpath);
        NSString *actual_path = [NSString stringWithUTF8String:actualpath];
        NSData *actual_data = [NSData dataWithContentsOfFile:actual_path];
        REQUIRE(actual_data &&
                [sha256(actual_data.bytes, actual_data.length) isEqual:required_sha]);
        myds_metal_set_concurrent(1);
        myds_metal_set_auto_barrier(0);
        myds_metal_buffer_t *t[4];
        for (unsigned i = 0; i < 4; i++) {
            t[i] = myds_metal_buffer_from_ptr(td[i]->data, sizes[i]);
            REQUIRE(t[i]);
        }
        fixture *c = calloc(cases, sizeof(fixture));
        REQUIRE(c);
        for (unsigned i = 0; i < cases; i++)
            c[i] = make_fixture(i);
        NSMutableArray *manifest = [NSMutableArray new];
        for (unsigned i = 0; i < cases; i++) {
            save(dir, manifest, [NSString stringWithFormat:@"input-case-%02u-x.f32", i],
                 myds_metal_buffer_data(c[i].x), DIM * 4u);
            save(dir, manifest, [NSString stringWithFormat:@"input-case-%02u-shared.f32", i],
                 myds_metal_buffer_data(c[i].shared_input), WIDTH * 4u);
            save(dir, manifest, [NSString stringWithFormat:@"input-case-%02u-ids.i32", i],
                 myds_metal_buffer_data(c[i].ids), 24);
            save(dir, manifest, [NSString stringWithFormat:@"input-case-%02u-weights.f32", i],
                 myds_metal_buffer_data(c[i].weights), 24);
        }
        NSMutableDictionary *environment = [NSMutableDictionary new];
        NSArray *env_names = @[
            @"MYDS_MOE_DOWN_OVERLAP", @"MYDS_MOE_IQ2_V41", @"MYDS_DENSE_Q4_V41", @"MYDS_MOE_Q2_V41",
            @"MYDS_ROUTE_SIMD_SHUFFLE", @"MYDS_ATTN_PACKED_REGISTER", @"MYDS_MOE_IQ2_POPCNT_SIGN",
            @"MYDS_MOE_IQ2_CONSTANT_LUT", @"MYDS_MOE_NSG", @"MYDS_MOE_DOWN_NSG",
            @"MYDS_DENSE_Q4_NR0", @"MYDS_DENSE_Q4_NSG", @"MYDS_MOE_Q2_SUM_CACHE",
            @"MYDS_MOE_Q2_SUM_CACHE_FUSED", @"MYDS_MOE_MM_MIN_TOKENS", @"MYDS_METAL_CONCURRENT",
            @"MYDS_KPROF", @"MYDS_MOE_STEP_PROF", @"MYDS_NO_RESIDENCY_HB"
        ];
        for (NSString *key in env_names) {
            const char *value = getenv(key.UTF8String);
            environment[key] = value ? (id)[NSString stringWithUTF8String:value] : [NSNull null];
        }
        char exe_path[4096];
        uint32_t exe_size = sizeof(exe_path);
        REQUIRE(!_NSGetExecutablePath(exe_path, &exe_size));
        NSData *binary = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:exe_path]];
        REQUIRE(binary);
        NSDictionary *identity = @{
            @"kind" : @"identity",
            @"argv" : [NSProcessInfo processInfo].arguments,
            @"environment" : environment,
            @"timestamp_utc" : [[NSDate date] description],
            @"pid" : @(getpid()),
            @"binary_sha256" : sha256(binary.bytes, binary.length),
            @"mode" : @(mode),
            @"actual_mode" : @(myds_metal_moe_down_overlap_enabled()),
            @"samples" : @(samples),
            @"cases" : @(cases),
            @"repetitions" : @(reps),
            @"seed" : @88172645463325252ull,
            @"model_path" : model,
            @"model_bytes" : @(st.st_size),
            @"model_mtime_s" : @(st.st_mtime),
            @"model_mtime_ns" :
                @((uint64_t)st.st_mtimespec.tv_sec * 1000000000ull + st.st_mtimespec.tv_nsec),
            @"model_inode" : @(st.st_ino),
            @"model_device" : @(st.st_dev),
            @"tensors" : descriptors,
            @"tensor_bytes_total" : @(sizes[0] + sizes[1] + sizes[2] + sizes[3]),
            @"expected_library_path" : expected_lib,
            @"actual_library_path" : actual_path,
            @"actual_library_sha256" : required_sha,
            @"q2_v41_actual_mode" : @(myds_metal_moe_q2_v41_enabled()),
            @"iq2_v41_actual_mode" : @(myds_metal_moe_iq2_v41_enabled()),
            @"dense_q4_v41_actual_mode" : @(myds_metal_dense_q4_v41_enabled()),
            @"fullmodel" : @NO,
            @"gpu_overlap_proven" : @NO,
            @"physical_cold_dram_proven" : @NO,
            @"weight_source" : @"read-only real GGUF layer0",
            @"input_source" : @"seeded F32; shared MID exact BF16 grid",
            @"timed_work" :
                @"SharedDown + IQ2 gate/up + Q2 down/sum6 + BF16 add in one native concurrent CB",
            @"encoder_dispatch_type" : @"concurrent",
            @"kprof" : @NO
        };
        emit(identity);
        // Invalid numeric IDs belong to a CPU-owned probe preflight. Never submit
        // them to the historical Q2 provider, which trusts GPU router legal IDs.
        int32_t *ip = myds_metal_buffer_data(c[0].ids);
        int32_t keep = ip[0];
        uint64_t dc = myds_metal_dispatch_count();
        ip[0] = -1;
        REQUIRE(!valid_ids(ip));
        REQUIRE(myds_metal_dispatch_count() == dc);
        ip[0] = keep;
        emit(@{
            @"kind" : @"negative",
            @"case" : @"invalid_cpu_supplied_id",
            @"gate_scope" : @"probe input preflight; not a native GPU router ID check",
            @"native_api_calls" : @0,
            @"dispatch_delta" : @0
        });
        if (mode) {
            myds_metal_buffer_t *shortout =
                myds_metal_buffer_sub_view(c[0].shared_out.owner, GUARD, DIM * 4u - 4);
            REQUIRE(shortout);
            for (unsigned i = 0; i < 2; i++) {
                REQUIRE(myds_metal_begin_batch());
                uint64_t before = myds_metal_dispatch_count();
                myds_metal_decode_candidate_counters_t cb, ca;
                myds_metal_moe_down_overlap_counters(&cb, false);
                REQUIRE(!overlap_call(&c[0], t, i ? c[0].moe_out.view : shortout));
                myds_metal_moe_down_overlap_counters(&ca, false);
                REQUIRE(myds_metal_dispatch_count() == before && ca.requests == cb.requests + 1 &&
                        ca.failures == cb.failures + 1 && ca.encoded == cb.encoded);
                REQUIRE(all_poison(c[0].mid) && all_poison(c[0].moe_out) &&
                        all_poison(c[0].shared_out));
                REQUIRE(myds_metal_commit_batch());
                guarded rejected[] = {c[0].mid, c[0].moe_out, c[0].shared_out};
                NSArray *rejected_names = @[ @"mid", @"moe", @"shared" ];
                for (unsigned j = 0; j < 3; ++j)
                    save(dir, manifest,
                         [NSString stringWithFormat:@"negative-%u-%@.raw", i, rejected_names[j]],
                         myds_metal_buffer_data(rejected[j].owner), rejected[j].bytes + 2 * GUARD);
                emit(@{
                    @"kind" : @"negative",
                    @"case" : i ? @"shared_output_alias_moe_output" : @"short_shared_output",
                    @"gate_scope" : @"actual native wrapper",
                    @"dispatch_delta" : @0,
                    @"requests_delta" : @1,
                    @"failures_delta" : @1,
                    @"encoded_delta" : @0,
                    @"buffers_all_poison" : @YES
                });
            }
            myds_metal_buffer_free(shortout);
        }
        myds_metal_moe_down_overlap_counters(NULL, true);
        myds_metal_moe_q2_v41_counters(NULL, true);
        myds_metal_dispatch_count_reset();
        NSMutableArray *gpu_values = [NSMutableArray new], *wall_values = [NSMutableArray new];
        NSMutableArray *reference = [NSMutableArray new];
        for (unsigned sample = 0; sample <= samples; sample++) {
            myds_metal_decode_candidate_counters_t before, after;
            myds_metal_moe_down_overlap_counters(&before, false);
            myds_metal_moe_q2_v41_counters_t qb, qa;
            myds_metal_moe_q2_v41_counters(&qb, false);
            uint64_t dispatch_before = myds_metal_dispatch_count(),
                     commits = myds_metal_get_commit_count();
            double wall0 = monotonic();
            REQUIRE(myds_metal_begin_batch());
            for (unsigned r = 0; r < reps; r++)
                for (unsigned i = 0; i < cases; i++)
                    REQUIRE(native_case(&c[i], t, mode));
            REQUIRE(myds_metal_commit_batch());
            double wall = monotonic() - wall0, start = 0, end = 0;
            REQUIRE(myds_metal_last_completed_gpu_span(&start, &end) && isfinite(start) &&
                    isfinite(end) && end > start);
            myds_metal_moe_down_overlap_counters(&after, false);
            myds_metal_moe_q2_v41_counters(&qa, false);
            uint64_t pairs = (uint64_t)reps * cases;
            REQUIRE(myds_metal_get_commit_count() == commits + 1 &&
                    myds_metal_dispatch_count() - dispatch_before == 4 * pairs &&
                    qa.requests - qb.requests == pairs && qa.encoded - qb.encoded == pairs &&
                    qa.failures == qb.failures);
            REQUIRE(after.requests - before.requests == (mode ? pairs : 0) &&
                    after.encoded - before.encoded == (mode ? pairs : 0) &&
                    after.failures == before.failures);
            for (unsigned i = 0; i < cases; i++) {
                verify_case(&c[i]);
                guarded gs[] = {c[i].mid, c[i].moe_out, c[i].shared_out, c[i].sum};
                for (unsigned f = 0; f < 4; f++) {
                    NSData *data = [NSData dataWithBytes:myds_metal_buffer_data(gs[f].view)
                                                  length:gs[f].bytes];
                    if (sample == 0)
                        [reference addObject:data];
                    else
                        REQUIRE([data isEqual:reference[i * 4 + f]]);
                }
                dump_case(dir, manifest, &c[i], i,
                          [NSString stringWithFormat:@"sample-%02u", sample]);
            }
            double gpu_us = (end - start) * 1e6 / pairs, wall_us = wall * 1e6 / pairs;
            if (sample) {
                [gpu_values addObject:@(gpu_us)];
                [wall_values addObject:@(wall_us)];
            }
            emit(@{
                @"kind" : @"sample",
                @"sample" : @(sample),
                @"warmup" : @(sample == 0),
                @"gpu_start_s" : @(start),
                @"gpu_end_s" : @(end),
                @"gpu_us_per_pair" : @(gpu_us),
                @"wall_us_per_pair" : @(wall_us),
                @"pairs" : @(pairs),
                @"dispatch_delta" : @(4 * pairs),
                @"commit_delta" : @1,
                @"actual_mode" : @(myds_metal_moe_down_overlap_enabled()),
                @"overlap_before" : counts(before),
                @"overlap_after" : counts(after),
                @"q2_before" : q2_counts(qb),
                @"q2_after" : q2_counts(qa),
                @"all_outputs_finite" : @YES,
                @"all_guards" : @YES,
                @"release_scratch_poison" : @YES,
                @"full_outputs_match_warmup" : @YES,
                @"completed_cb" : @YES
            });
        }
        NSArray *sorted_gpu = [gpu_values sortedArrayUsingSelector:@selector(compare:)],
                *sorted_wall = [wall_values sortedArrayUsingSelector:@selector(compare:)];
        double gm = samples % 2 ? [sorted_gpu[samples / 2] doubleValue]
                                : ([sorted_gpu[samples / 2 - 1] doubleValue] +
                                   [sorted_gpu[samples / 2] doubleValue]) /
                                      2,
               wm = samples % 2 ? [sorted_wall[samples / 2] doubleValue]
                                : ([sorted_wall[samples / 2 - 1] doubleValue] +
                                   [sorted_wall[samples / 2] doubleValue]) /
                                      2;
        emit(@{
            @"kind" : @"summary",
            @"gpu_median_us_per_pair" : @(gm),
            @"wall_median_us_per_pair" : @(wm),
            @"mode" : @(mode),
            @"actual_mode" : @(myds_metal_moe_down_overlap_enabled()),
            @"formal_samples" : @(samples),
            @"gpu_overlap_proven" : @NO,
            @"fullmodel_speedup_proven" : @NO
        });
        struct stat final_st;
        REQUIRE(!stat(model.fileSystemRepresentation, &final_st));
        REQUIRE(final_st.st_size == st.st_size && final_st.st_ino == st.st_ino &&
                final_st.st_mtimespec.tv_sec == st.st_mtimespec.tv_sec &&
                final_st.st_mtimespec.tv_nsec == st.st_mtimespec.tv_nsec);
        NSData *final_lib = [NSData dataWithContentsOfFile:actual_path];
        REQUIRE(final_lib && [sha256(final_lib.bytes, final_lib.length) isEqual:required_sha]);
        NSString *histpath = [dir stringByAppendingPathComponent:@"dispatch-hist.txt"];
        FILE *hist = fopen(histpath.fileSystemRepresentation, "wx");
        REQUIRE(hist);
        REQUIRE(myds_metal_dispatch_hist_dump(hist));
        REQUIRE(!fclose(hist));
        NSData *histdata = [NSData dataWithContentsOfFile:histpath];
        REQUIRE(histdata);
        [manifest addObject:@{
            @"path" : @"dispatch-hist.txt",
            @"bytes" : @(histdata.length),
            @"sha256" : sha256(histdata.bytes, histdata.length)
        }];
        for (unsigned i = 0; i < cases; i++)
            free_fixture(c[i]);
        free(c);
        for (unsigned i = 0; i < 4; i++)
            myds_metal_buffer_free(t[i]);
        myds_metal_cleanup();
        REQUIRE(myds_metal_moe_down_overlap_enabled() == 0 &&
                myds_metal_moe_iq2_v41_enabled() == 0 && myds_metal_dense_q4_v41_enabled() == 0);
        gguf_close(gf);
        NSData *md = [NSJSONSerialization
            dataWithJSONObject:@{
                @"schema_version" : @1,
                @"identity" : identity,
                @"files" : manifest
            }
                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                         error:&e];
        REQUIRE(md);
        REQUIRE([md writeToFile:[dir stringByAppendingPathComponent:@"artifact-manifest.json"]
                        options:NSDataWritingWithoutOverwriting
                          error:&e]);
        emit(@{
            @"kind" : @"complete",
            @"cleanup_modes_reset" : @YES,
            @"raw_files" : @(manifest.count),
            @"fullmodel" : @NO
        });
    }
    return 0;
}
