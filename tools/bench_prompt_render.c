/* CPU-only worker for bench_prompt_render.py. No model or GPU is opened. */
#define _POSIX_C_SOURCE 200809L
#include "myds_server_internal.h"
#include "myds_vjob.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int myds_vjob_sink_add(myds_vjob_sink_t *sink, const uint8_t *encoded, size_t length,
                       char *error, size_t error_cap) {
    (void)sink; (void)encoded; (void)length; (void)error; (void)error_cap;
    return 0; /* Corpus is text-only. */
}

static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) exit(2);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f || fseek(f, 0, SEEK_END)) return 2;
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) return 2;
    char *body = malloc((size_t)n + 1u);
    if (!body || fread(body, 1, (size_t)n, f) != (size_t)n || fclose(f)) return 2;
    body[n] = 0;
    size_t cap = strtoull(argv[2], NULL, 10);
    unsigned reps = (unsigned)strtoul(argv[4], NULL, 10);
    if (cap > SIZE_MAX - 2u || !reps) return 2;
    unsigned char *guarded = malloc(cap + 2u);
    if (!guarded) return 2;
    memset(guarded, 0xa5, cap + 2u);
    char *out = (char *)guarded + 1;
    const char *reason = "";

    if (!strcmp(argv[3], "check")) {
        const bool ok = parse_chat_prompt_ex(body, out, cap, &reason);
        size_t why = reason ? strlen(reason) : 0;
        printf("%d %zu %d\n", ok, why, guarded[0] == 0xa5 && guarded[cap + 1u] == 0xa5);
        if (why) fwrite(reason, 1, why, stdout);
        if (cap) fwrite(out, 1, cap, stdout);
    } else if (!strcmp(argv[3], "time")) {
        for (unsigned i = 0; i < 8u; ++i)
            if (!parse_chat_prompt_ex(body, out, cap, &reason)) return 3;
        const uint64_t start = now_ns();
        for (unsigned i = 0; i < reps; ++i)
            if (!parse_chat_prompt_ex(body, out, cap, &reason)) return 3;
        const uint64_t elapsed = now_ns() - start;
        if (guarded[0] != 0xa5 || guarded[cap + 1u] != 0xa5) return 3;
        printf("%llu %zu\n", (unsigned long long)elapsed, strlen(out));
    } else return 2;
    free(guarded); free(body);
    return 0;
}
