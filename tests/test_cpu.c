#include "myds_cpu.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Frozen head-major reference from the original inference path. */
static void reference_rope(float *x, uint32_t heads, uint32_t dim,
                           uint32_t rot, uint32_t pos, float base,
                           float scale, bool inverse) {
    if (rot == 0 || rot > dim) return;
    float theta_scale = powf(base, -2.0f / (float)rot);
    float sign = inverse ? -1.0f : 1.0f;
    for (uint32_t h = 0; h < heads; h++) {
        float *tail = x + (size_t)h * dim + dim - rot;
        float angle = (float)pos;
        for (uint32_t i = 0; i < rot; i += 2) {
            float theta = scale * angle;
            float c = cosf(theta), s = sign * sinf(theta);
            float x0 = tail[i], x1 = tail[i + 1];
            tail[i] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
            angle *= theta_scale;
        }
    }
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const uint32_t dims[] = {16, 64, 512};
    const uint32_t heads[] = {1, 2, 64};
    const uint32_t positions[] = {0, 1, 4095, 262143};
    const float scales[] = {1.0f, 0.125f};
    const size_t count = 64 * 512;
    float *input = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    assert(input && expected && actual);
    for (size_t i = 0; i < count; i++) input[i] = sinf((float)i * 0.13f);

    unsigned cases = 0;
    for (size_t d = 0; d < 3; d++) {
        for (size_t h = 0; h < 3; h++) {
            for (size_t p = 0; p < 4; p++) {
                for (size_t s = 0; s < 2; s++) {
                    for (unsigned inv = 0; inv < 2; inv++) {
                        for (unsigned full = 0; full < 2; full++) {
                            uint32_t rot = full ? dims[d] : dims[d] / 2;
                            memcpy(expected, input, count * sizeof(float));
                            memcpy(actual, input, count * sizeof(float));
                            reference_rope(expected, heads[h], dims[d], rot,
                                           positions[p], 10000.0f, scales[s], inv);
                            myds_rope_tail_inplace(actual, heads[h], dims[d], rot,
                                                   positions[p], 10000.0f, scales[s], inv);
                            assert(memcmp(expected, actual, count * sizeof(float)) == 0);
                            myds_rope_tail_inplace(actual, heads[h], dims[d], rot,
                                                   positions[p], 10000.0f, scales[s], !inv);
                            for (size_t i = 0; i < count; i++) {
                                assert(isfinite(actual[i]));
                                assert(fabsf(actual[i] - input[i]) < 4e-7f);
                            }
                            cases++;
                        }
                    }
                }
            }
        }
    }
    memcpy(actual, input, count * sizeof(float));
    myds_rope_tail_inplace(actual, 64, 512, 513, 10, 10000, 1, false);
    myds_rope_tail_inplace(actual, 64, 512, 63, 10, 10000, 1, false);
    myds_rope_tail_inplace(actual, 64, 512, 0, 10, 10000, 1, false);
    assert(memcmp(actual, input, count * sizeof(float)) == 0);
    printf("CPU RoPE: %u bit-exact cases, inverse and invalid-shape checks passed\n", cases);

    if (argc == 2 && strcmp(argv[1], "--bench") == 0) {
        double legacy[7], optimized[7];
        const unsigned iterations = 1000;
        for (unsigned round = 0; round < 8; round++) {
            double times[2];
            for (unsigned run = 0; run < 2; run++) {
                unsigned variant = (run + round) % 2;
                memcpy(actual, input, count * sizeof(float));
                double begin = now_s();
                for (unsigned i = 0; i < iterations; i++) {
                    if (variant)
                        myds_rope_tail_inplace(actual, 64, 512, 64, 4095, 10000, 1, i & 1);
                    else
                        reference_rope(actual, 64, 512, 64, 4095, 10000, 1, i & 1);
                }
                times[variant] = (now_s() - begin) * 1e6 / iterations;
                assert(isfinite(actual[count - 1]));
            }
            if (round) {
                legacy[round - 1] = times[0];
                optimized[round - 1] = times[1];
            }
        }
        qsort(legacy, 7, sizeof(double), compare_double);
        qsort(optimized, 7, sizeof(double), compare_double);
        printf("{\"benchmark\":\"rope\",\"heads\":64,\"head_dim\":512,\"rope_dim\":64,"
               "\"legacy_us\":%.3f,\"optimized_us\":%.3f,\"speedup\":%.3f}\n",
               legacy[3], optimized[3], legacy[3] / optimized[3]);
    }
    free(actual);
    free(expected);
    free(input);
    return 0;
}
