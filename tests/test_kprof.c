/* Per-kernel GPU timing (MYDS_KPROF): armed batches must produce the same
 * results as unarmed ones and yield one tagged, valid timestamp pair per
 * dispatch. Skips (exit 0) if the device has no stage-boundary sampling. */
#include "myds_metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4096u
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

static int run_graph(myds_metal_buffer_t *out, myds_metal_buffer_t *g, myds_metal_buffer_t *u,
                     myds_metal_buffer_t *t) {
    if (!myds_metal_begin_batch()) return 0;
    for (int i = 0; i < 3; ++i) {
        if (!myds_metal_swiglu(t, g, u, N)) return 0;
        if (!myds_metal_add(out, t, u, N)) return 0;
        myds_metal_barrier();   /* must be harmless while armed */
    }
    return myds_metal_commit_batch();
}

int main(void) {
    setenv("MYDS_KPROF", "1", 1);
    CHECK(myds_metal_init("metal"));
    if (!myds_metal_kprof_enabled()) { puts("SKIP: no stage-boundary counter sampling"); return 0; }

    myds_metal_buffer_t *g = myds_metal_buffer_alloc(N * 4), *u = myds_metal_buffer_alloc(N * 4);
    myds_metal_buffer_t *t = myds_metal_buffer_alloc(N * 4);
    myds_metal_buffer_t *ref = myds_metal_buffer_alloc(N * 4), *got = myds_metal_buffer_alloc(N * 4);
    CHECK(g && u && t && ref && got);
    float *gp = myds_metal_buffer_data(g), *up = myds_metal_buffer_data(u);
    for (uint32_t i = 0; i < N; ++i) { gp[i] = sinf((float)i) * 3.0f; up[i] = cosf((float)i * 0.5f); }

    CHECK(run_graph(ref, g, u, t));                 /* unarmed reference */

    myds_metal_kprof_arm(1);
    CHECK(run_graph(got, g, u, t));                 /* armed */
    myds_metal_kprof_arm(0);
    CHECK(!memcmp(myds_metal_buffer_data(ref), myds_metal_buffer_data(got), N * 4));

    double floor_us = 0;
    CHECK(myds_metal_kprof_calibrate(&floor_us));   /* must not disturb the captured 6 records */
    CHECK(floor_us > 0.0 && floor_us < 5000.0);

    FILE *f = tmpfile();
    CHECK(f);
    CHECK(myds_metal_kprof_report(f, "test", 1));
    rewind(f);
    char line[4096];
    int kernels = 0, summaries = 0;
    unsigned long long total_calls = 0;
    while (fgets(line, sizeof line, f)) {
        if (getenv("KPROF_DUMP")) fputs(line, stderr);
        if (strstr(line, "\"kind\":\"kprof_kernel\"")) {
            ++kernels;
            const char *c = strstr(line, "\"calls\":");
            CHECK(c);
            total_calls += strtoull(c + 8, NULL, 10);
            CHECK(!strstr(line, "(untagged)"));
        } else if (strstr(line, "\"kind\":\"kprof_summary\"")) {
            ++summaries;
            CHECK(strstr(line, "\"encoders\":6,"));      /* 3 x (swiglu + add), one encoder each */
            CHECK(strstr(line, "\"bad_samples\":0,"));
            CHECK(strstr(line, "\"untagged_encoders\":0,"));
            CHECK(strstr(line, "\"merged_encoders\":0,"));
            CHECK(strstr(line, "\"buffer_overflow\":0,"));
            const char *fl = strstr(line, "\"floor_us\":");
            CHECK(fl && atof(fl + 11) > 0.0);
            const char *b = strstr(line, "\"busy_ms\":");
            CHECK(b && atof(b + 10) > 0.0);
        }
    }
    CHECK(kernels >= 2 && summaries == 1 && total_calls == 6);
    fclose(f);

    /* After disarm a normal batch still works (no stale sample attachments). */
    CHECK(run_graph(got, g, u, t));
    CHECK(!memcmp(myds_metal_buffer_data(ref), myds_metal_buffer_data(got), N * 4));
    puts("test_kprof: OK");
    return 0;
}
