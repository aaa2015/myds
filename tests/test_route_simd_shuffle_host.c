/* CPU-only failure-path checks against the actual C route entry. Metal calls
 * are explicit stubs; this fixture cannot prove shader correctness or speed.
 * Build: clang -O2 -Wall -Wextra -std=c11 -Iinclude -Ithird_party
 *   -Wl,-dead_strip tests/test_route_simd_shuffle_host.c src/myds_v41_moe.c
 *   -o /tmp/test_route_simd_shuffle_host
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_moe.h"
#include "myds_log.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct myds_metal_buffer { size_t bytes; };
static unsigned calls, barriers;
static int fail_candidate, fail_reference;
static char last_kernel[128];
static uint32_t last_groups, last_threads;
size_t myds_metal_buffer_size(const myds_metal_buffer_t *b) { return b ? b->bytes : 0; }
void myds_log(myds_log_level_t level, const char *fmt, ...) { (void)level; (void)fmt; }
int myds_metal_dispatch_kernel(const char *name, const void *args, size_t size,
    const myds_metal_buffer_t *const *buffers, size_t n_buffers,
    const uint32_t groups[3], const uint32_t threads[3], size_t scratch) {
    (void)args; (void)buffers; ++calls;
    assert(strlen(name) < sizeof(last_kernel)); strcpy(last_kernel, name);
    last_groups = groups[0]; last_threads = threads[0];
    if (!strcmp(name, "kernel_myds_v41_route_simd_shuffle")) {
        assert(size == 28 && n_buffers == 6 && scratch == 5760 && threads[0] == 512);
        return !fail_candidate;
    }
    if (!strcmp(name, "kernel_myds_v41_route_fused")) return !fail_reference;
    return 1;
}
void myds_metal_barrier_buffers(const myds_metal_buffer_t *const *buffers, size_t count) {
    (void)buffers; assert(count > 0); ++barriers;
}
void myds_metal_barrier_buffer(const myds_metal_buffer_t *b) { assert(b); ++barriers; }
static void reset(void) {
    calls = barriers = 0; last_kernel[0] = 0; fail_candidate = fail_reference = 0;
    myds_v41_route_simd_shuffle_counters(NULL, true);
}
static void counters(uint64_t requests, uint64_t encoded, uint64_t failures) {
    myds_v41_route_simd_shuffle_counters_t c;
    myds_v41_route_simd_shuffle_counters(&c, false);
    assert(c.requests == requests && c.encoded == encoded && c.failures == failures);
}
static struct myds_metal_buffer selected = {6*5*4}, weights = {6*5*4};
static struct myds_metal_buffer probs = {384*5*4}, scores = {384*5*4};
static struct myds_metal_buffer logits = {384*5*4}, bias = {384*4};
static int route(uint32_t rows) {
    return myds_v41_route_batch(&selected, &weights, &probs, &scores, &logits, &bias,
                               384, 6, 2.5f, rows);
}
int main(void) {
    assert(!setenv("MYDS_MOE_ROUTE_FUSE", "2", 1));
    for (unsigned mode=0; mode<2; mode++) {
        reset(); if (mode) assert(!setenv("MYDS_ROUTE_SIMD_SHUFFLE", "0", 1));
        else assert(!unsetenv("MYDS_ROUTE_SIMD_SHUFFLE"));
        assert(myds_v41_route_simd_shuffle_mode() == 0 && route(1));
        assert(calls == 1 && !strcmp(last_kernel, "kernel_myds_v41_route_fused"));
        assert(barriers == 1); counters(0, 0, 0);
    }
    reset(); assert(!setenv("MYDS_ROUTE_SIMD_SHUFFLE", "1", 1));
    assert(myds_v41_route_simd_shuffle_mode() == 1 && route(5));
    assert(calls == 1 && !strcmp(last_kernel, "kernel_myds_v41_route_simd_shuffle"));
    assert(last_groups == 5 && last_threads == 512 && barriers == 1); counters(1, 1, 0);
    reset(); fail_candidate = 1; assert(!route(1));
    assert(calls == 1 && barriers == 0 && !strcmp(last_kernel, "kernel_myds_v41_route_simd_shuffle"));
    counters(1, 0, 1); /* No original or staged fallback after candidate failure. */
    const char *bad[] = {"", "01", "true", "2", " 1", "1 "};
    for (unsigned i=0; i<sizeof(bad)/sizeof(*bad); i++) {
        reset(); assert(!setenv("MYDS_ROUTE_SIMD_SHUFFLE", bad[i], 1));
        assert(myds_v41_route_simd_shuffle_mode() == -1 && !route(1));
        assert(calls == 0 && barriers == 0); counters(0, 0, 0);
    }
    reset(); assert(!setenv("MYDS_ROUTE_SIMD_SHUFFLE", "1", 1));
    assert(!route(0)); size_t saved = logits.bytes; logits.bytes = 4;
    assert(!route(1)); logits.bytes = saved;
    assert(calls == 0 && barriers == 0); counters(0, 0, 0);
    /* Disabled candidate preserves both the old fused fallback and fuse0/1. */
    reset(); assert(!setenv("MYDS_ROUTE_SIMD_SHUFFLE", "0", 1)); fail_reference = 1;
    assert(route(1) && calls == 4); counters(0, 0, 0);
    for (unsigned level=0; level<2; level++) {
        reset(); assert(!setenv("MYDS_MOE_ROUTE_FUSE", level ? "1" : "0", 1));
        assert(route(1) && calls == (level ? 3u : 7u)); counters(0, 0, 0);
    }
    puts("PASS: native router host selection, counters, guards, strict env, candidate failure, original fallback; CPU stubs only");
    return 0;
}
