/* DSpark's 128-expert router contract against an independent scalar oracle.
 * Bias selects experts; unbiased sqrt(softplus(logit)) supplies weights. */
#include "myds_metal.h"
#include "myds_v41_moe.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

enum { EXPERTS = 128, GUARD_BYTES = 16 };
struct guarded { myds_metal_buffer_t *owner, *view; size_t bytes; };

static struct guarded alloc_guarded(size_t bytes) {
    struct guarded g = {0};
    g.bytes = bytes;
    g.owner = myds_metal_buffer_alloc(bytes + 2 * GUARD_BYTES);
    CHECK(g.owner);
    memset(myds_metal_buffer_data(g.owner), 0xa5, bytes + 2 * GUARD_BYTES);
    g.view = myds_metal_buffer_sub_view(g.owner, GUARD_BYTES, bytes);
    CHECK(g.view);
    return g;
}

static void free_guarded(struct guarded g) {
    const uint8_t *p = myds_metal_buffer_data(g.owner);
    for (size_t i = 0; i < GUARD_BYTES; ++i)
        CHECK(p[i] == 0xa5 && p[GUARD_BYTES + g.bytes + i] == 0xa5);
    myds_metal_buffer_free(g.view);
    myds_metal_buffer_free(g.owner);
}

static double probability(double x) {
    /* Stable exact softplus in double; no shader polynomial or CPU router. */
    return sqrt(fmax(x, 0.0) + log1p(exp(-fabs(x))));
}

static void oracle(const float *logits, const float *bias, unsigned top_k,
                   float scale, int32_t *ids, double *weights) {
    double probabilities[EXPERTS], scores[EXPERTS];
    int picked[EXPERTS] = {0};
    for (unsigned e = 0; e < EXPERTS; ++e) {
        probabilities[e] = probability(logits[e]);
        scores[e] = probabilities[e] + (bias ? bias[e] : 0.0);
    }
    double sum = 0.0;
    for (unsigned k = 0; k < top_k; ++k) {
        unsigned best = EXPERTS;
        for (unsigned e = 0; e < EXPERTS; ++e)
            if (!picked[e] && (best == EXPERTS || scores[e] > scores[best])) best = e;
        CHECK(best < EXPERTS);
        picked[best] = 1;
        ids[k] = (int32_t)best;
        weights[k] = probabilities[best];
        sum += weights[k];
    }
    sum = fmax(sum, 0x1p-14);
    for (unsigned k = 0; k < top_k; ++k) weights[k] = weights[k] / sum * scale;
}

static uint32_t rng_state = 42;
static float random_float(float low, float high) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return low + (high - low) * (float)(rng_state >> 8) / 16777216.0f;
}

static void run_case(unsigned rows, unsigned top_k, float scale, unsigned mode,
                     int use_bias) {
    struct guarded logits = alloc_guarded((size_t)rows * EXPERTS * sizeof(float));
    struct guarded bias = alloc_guarded(EXPERTS * sizeof(float));
    struct guarded selected = alloc_guarded((size_t)rows * top_k * sizeof(int32_t));
    struct guarded weights = alloc_guarded((size_t)rows * top_k * sizeof(float));
    float *x = myds_metal_buffer_data(logits.view);
    float *b = myds_metal_buffer_data(bias.view);
    for (unsigned e = 0; e < EXPERTS; ++e) b[e] = random_float(-2.0f, 2.0f);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned e = 0; e < EXPERTS; ++e) {
            if (mode == 0) x[r * EXPERTS + e] = random_float(-35.0f, 100.0f);
            else if (mode == 1) x[r * EXPERTS + e] = 0.0f; /* Exact ties. */
            else if (mode == 2) x[r * EXPERTS + e] = -30.0f - e * 0.1f;
            else if (mode == 3) x[r * EXPERTS + e] = -50.0f;
            else x[r * EXPERTS + e] = -200.0f; /* Zero underflow weights. */
        }
        if (mode == 3) {
            x[r * EXPERTS + 0] = 4.0f;
            x[r * EXPERTS + 1] = 0.0f;
            x[r * EXPERTS + 2] = -4.0f;
        }
    }
    if (mode == 1) memset(b, 0, EXPERTS * sizeof(float));
    if (mode == 3) {
        for (unsigned e = 0; e < EXPERTS; ++e) b[e] = -10.0f;
        b[0] = 0.0f; b[1] = 2.0f; b[2] = 6.0f;
    }
    float *input_copy = malloc(logits.bytes);
    float *bias_copy = malloc(bias.bytes);
    CHECK(input_copy && bias_copy);
    memcpy(input_copy, x, logits.bytes);
    memcpy(bias_copy, b, bias.bytes);

    CHECK(myds_metal_begin_batch());
    CHECK(myds_dspark_route_fused(selected.view, weights.view, logits.view,
                                 use_bias ? bias.view : NULL, top_k, scale, rows));
    CHECK(myds_metal_commit_batch());
    const int32_t *gpu_ids = myds_metal_buffer_data(selected.view);
    const float *gpu_weights = myds_metal_buffer_data(weights.view);
    for (unsigned r = 0; r < rows; ++r) {
        int32_t ids[32]; double expected[32];
        oracle(x + r * EXPERTS, use_bias ? b : NULL, top_k, scale, ids, expected);
        if (mode == 3) CHECK(ids[0] == 2 && ids[1] == 1 && ids[2] == 0);
        double sum = 0.0;
        for (unsigned k = 0; k < top_k; ++k) {
            const size_t i = r * top_k + k;
            if (gpu_ids[i] != ids[k] || !isfinite(gpu_weights[i]) ||
                fabs(gpu_weights[i] - expected[k]) > 2e-6 + 2e-5 * fabs(expected[k])) {
                fprintf(stderr, "row=%u k=%u mode=%u: id=%d want=%d weight=%.9g want=%.9g\n",
                        r, k, mode, gpu_ids[i], ids[k], gpu_weights[i], expected[k]);
                exit(1);
            }
            sum += gpu_weights[i];
        }
        /* Extremely small probabilities intentionally sum below scale after
         * the 2^-14 denominator floor. This is not a softmax. */
        if (mode == 2 || mode == 4) CHECK(sum < scale * 0.05);
        else CHECK(fabs(sum - scale) < 2e-5 * scale);
    }
    CHECK(memcmp(input_copy, x, logits.bytes) == 0);
    CHECK(memcmp(bias_copy, b, bias.bytes) == 0);
    free(input_copy); free(bias_copy);
    free_guarded(logits); free_guarded(bias);
    free_guarded(selected); free_guarded(weights);
}

static void test_invalid_contract(void) {
    struct guarded logits = alloc_guarded(EXPERTS * sizeof(float));
    struct guarded bias = alloc_guarded(EXPERTS * sizeof(float));
    struct guarded selected = alloc_guarded(3 * sizeof(int32_t));
    struct guarded weights = alloc_guarded(3 * sizeof(float));
    myds_metal_buffer_t *short_logits = myds_metal_buffer_sub_view(logits.view, 0, logits.bytes - 4);
    myds_metal_buffer_t *short_bias = myds_metal_buffer_sub_view(bias.view, 0, bias.bytes - 4);
    myds_metal_buffer_t *short_ids = myds_metal_buffer_sub_view(selected.view, 0, selected.bytes - 4);
    myds_metal_buffer_t *short_weights = myds_metal_buffer_sub_view(weights.view, 0, weights.bytes - 4);
    CHECK(short_logits && short_bias && short_ids && short_weights);
    const uint64_t before = myds_metal_dispatch_count();
#define REJECT(s, w, x, b, k, scale, rows) \
    CHECK(!myds_dspark_route_fused(s, w, x, b, k, scale, rows))
    REJECT(selected.view, weights.view, short_logits, bias.view, 3, 1.5f, 1);
    REJECT(selected.view, weights.view, logits.view, short_bias, 3, 1.5f, 1);
    REJECT(short_ids, weights.view, logits.view, bias.view, 3, 1.5f, 1);
    REJECT(selected.view, short_weights, logits.view, bias.view, 3, 1.5f, 1);
    REJECT(selected.view, weights.view, logits.view, bias.view, 3, 1.5f, 2);
    REJECT(selected.view, weights.view, logits.view, bias.view, 3, 1.5f, UINT32_MAX / EXPERTS + 1u);
    REJECT(selected.view, weights.view, logits.view, bias.view, 0, 1.5f, 1);
    REJECT(selected.view, weights.view, logits.view, bias.view, 33, 1.5f, 1);
    REJECT(selected.view, weights.view, logits.view, bias.view, 3, 1.5f, 0);
    REJECT(selected.view, weights.view, logits.view, bias.view, 3, NAN, 1);
    REJECT(selected.view, weights.view, logits.view, bias.view, 3, INFINITY, 1);
    REJECT(selected.view, weights.view, logits.view, bias.view, 3, 0.0f, 1);
    REJECT(selected.view, weights.view, logits.view, bias.view, 3, -1.0f, 1);
#undef REJECT
    CHECK(myds_metal_dispatch_count() == before);
    const uint8_t *ids = myds_metal_buffer_data(selected.view);
    const uint8_t *w = myds_metal_buffer_data(weights.view);
    for (size_t i = 0; i < selected.bytes; ++i) CHECK(ids[i] == 0xa5 && w[i] == 0xa5);
    myds_metal_buffer_free(short_logits); myds_metal_buffer_free(short_bias);
    myds_metal_buffer_free(short_ids); myds_metal_buffer_free(short_weights);
    free_guarded(logits); free_guarded(bias);
    free_guarded(selected); free_guarded(weights);
}

int main(void) {
    CHECK(myds_metal_init(NULL));
    test_invalid_contract();
    for (unsigned trial = 0; trial < 64; ++trial)
        run_case(5, 3, trial & 1 ? 1.5f : 2.5f, 0, trial & 1);
    run_case(5, 3, 1.5f, 3, 1);
    run_case(5, 3, 1.5f, 1, 0);
    run_case(5, 32, 2.5f, 1, 1);
    run_case(5, 3, 1.5f, 2, 0);
    run_case(5, 3, 1.5f, 4, 0);
    puts("DSpark router: 320 random rows, bias order, normalization, ties, top-32, underflow, view guards and rejection contract passed");
    myds_metal_cleanup();
    return 0;
}
