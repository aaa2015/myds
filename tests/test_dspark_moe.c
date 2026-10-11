/* Independent scalar dequantization/matvec oracle for the DSpark Q2_K/top-3
 * expert graph. Exercises real production dimensions and GPU-produced IDs. */
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
enum { ACTIVE = 3, BLOCK = 256, Q2_BYTES = 84, GUARD = 16 };
struct guarded { myds_metal_buffer_t *owner, *view; size_t bytes; };

static struct guarded alloc_guarded(size_t bytes) {
    struct guarded g = {0}; g.bytes = bytes;
    g.owner = myds_metal_buffer_alloc(bytes + 2u * GUARD); CHECK(g.owner);
    uint8_t *p = myds_metal_buffer_data(g.owner);
    memset(p, 0xa5, GUARD); memset(p + GUARD + bytes, 0xa5, GUARD);
    g.view = myds_metal_buffer_sub_view(g.owner, GUARD, bytes); CHECK(g.view);
    return g;
}
static void free_guarded(struct guarded g) {
    const uint8_t *p = myds_metal_buffer_data(g.owner);
    for (size_t i = 0; i < GUARD; ++i)
        CHECK(p[i] == 0xa5 && p[GUARD + g.bytes + i] == 0xa5);
    myds_metal_buffer_free(g.view); myds_metal_buffer_free(g.owner);
}

static float half_to_float(uint16_t h) {
    const unsigned exponent = (h >> 10) & 31u, fraction = h & 1023u;
    double value = exponent ? ldexp(1.0 + (double)fraction / 1024.0, (int)exponent - 15)
                            : ldexp((double)fraction, -24);
    return (float)((h & 0x8000u) ? -value : value);
}

static float q2_value(const uint8_t *row, unsigned k) {
    const uint8_t *b = row + (size_t)(k / BLOCK) * Q2_BYTES;
    const unsigned within = k % BLOCK, group = within / 16u;
    /* GGML Q2_K stores two 128-value regions. Each packed byte supplies
     * four values spaced by 32, each with its own 16-value scale/min. */
    const unsigned byte = (within / 128u) * 32u + within % 32u;
    const unsigned shift = ((within % 128u) / 32u) * 2u;
    const unsigned q = (b[16u + byte] >> shift) & 3u;
    uint16_t d, m; memcpy(&d, b + 80u, 2); memcpy(&m, b + 82u, 2);
    return half_to_float(d) * (float)(b[group] & 15u) * (float)q -
           half_to_float(m) * (float)(b[group] >> 4u);
}

static void fill_expert(uint8_t *p, unsigned in_dim, unsigned rows,
                        unsigned expert, unsigned seed) {
    const size_t row_bytes = (size_t)(in_dim / BLOCK) * Q2_BYTES;
    for (unsigned r = 0; r < rows; ++r) {
        const unsigned mode = (r + seed) % 4u;
        for (unsigned block = 0; block < in_dim / BLOCK; ++block) {
            uint8_t *b = p + r * row_bytes + block * Q2_BYTES;
            for (unsigned g = 0; g < 16; ++g) {
                unsigned sc = (expert * 13u + r * 7u + g * 5u + block * 3u + seed) % 15u + 1u;
                unsigned mn = (expert * 7u + r * 3u + g * 11u + block + seed) % 15u + 1u;
                if (mode == 0) mn = 0;
                if (mode == 1) sc = 0;
                b[g] = (uint8_t)(sc | (mn << 4u));
            }
            for (unsigned q = 0; q < 64; ++q)
                b[16 + q] = (uint8_t)(expert * 97u + r * 37u + block * 67u + q * 29u + seed * 113u);
            uint16_t d = mode == 3 ? 0x1000u : 0x2000u;
            uint16_t m = mode == 3 ? 0x0c00u : 0x1800u;
            memcpy(b + 80, &d, 2); memcpy(b + 82, &m, 2);
        }
    }
}

static uint64_t hash_bytes(const void *data, size_t count) {
    const uint8_t *p = data; uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < count; ++i) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}

static void compare(const char *label, const float *gpu, const double *reference,
                    size_t count, double absolute, double relative) {
    double worst = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double error = fabs(gpu[i] - reference[i]);
        if (!isfinite(gpu[i]) || error > absolute + relative * fabs(reference[i])) {
            fprintf(stderr, "%s[%zu]: gpu=%.9g scalar=%.9g abs_error=%.9g\n",
                    label, i, gpu[i], reference[i], error); exit(1);
        }
        if (error > worst) worst = error;
    }
    printf("  %s: %zu values checked, maximum absolute error %.9g\n", label, count, worst);
}

static void run_case(unsigned dim, unsigned width, unsigned experts,
                     float clamp_val, int gpu_router) {
    printf("[TEST] DSpark MoE dim=%u width=%u experts=%u clamp=%.1f GPU-router=%d\n",
           dim, width, experts, clamp_val, gpu_router);
    const size_t gate_expert = (size_t)(dim / BLOCK) * Q2_BYTES * width;
    const size_t down_expert = (size_t)(width / BLOCK) * Q2_BYTES * dim;
    struct guarded gate = alloc_guarded(gate_expert * experts);
    struct guarded up = alloc_guarded(gate_expert * experts);
    struct guarded down = alloc_guarded(down_expert * experts);
    struct guarded x = alloc_guarded(dim * sizeof(float));
    struct guarded ids = alloc_guarded(ACTIVE * sizeof(int32_t));
    struct guarded weights = alloc_guarded(ACTIVE * sizeof(float));
    struct guarded mid = alloc_guarded(ACTIVE * width * sizeof(float));
    struct guarded out = alloc_guarded(dim * sizeof(float));
    int32_t selected[ACTIVE] = {(int32_t)experts - 1, 0, (int32_t)experts / 2};
    float expected_weights[ACTIVE] = {0.15f, 0.9f, 0.45f};
    for (unsigned k = 0; k < ACTIVE; ++k) {
        const unsigned e = (unsigned)selected[k];
        fill_expert((uint8_t *)myds_metal_buffer_data(gate.view) + e * gate_expert, dim, width, e, 0);
        fill_expert((uint8_t *)myds_metal_buffer_data(up.view) + e * gate_expert, dim, width, e, 1);
        fill_expert((uint8_t *)myds_metal_buffer_data(down.view) + e * down_expert, width, dim, e, 2);
    }
    float *input = myds_metal_buffer_data(x.view);
    for (unsigned i = 0; i < dim; ++i)
        input[i] = 3.5f + (float)((int)(i * 19u % 127u) - 63) / 128.0f;
    memcpy(myds_metal_buffer_data(ids.view), selected, sizeof(selected));
    memcpy(myds_metal_buffer_data(weights.view), expected_weights, sizeof(expected_weights));
    memset(myds_metal_buffer_data(mid.view), 0xa5, mid.bytes);
    memset(myds_metal_buffer_data(out.view), 0xa5, out.bytes);
    struct guarded logits = {0}, bias = {0};
    if (gpu_router) {
        CHECK(experts == 128);
        logits = alloc_guarded(experts * sizeof(float)); bias = alloc_guarded(experts * sizeof(float));
        float *l = myds_metal_buffer_data(logits.view), *b = myds_metal_buffer_data(bias.view);
        for (unsigned i = 0; i < experts; ++i) { l[i] = -20.0f; b[i] = 0.0f; }
        double sum = 0.0, probabilities[ACTIVE];
        for (unsigned k = 0; k < ACTIVE; ++k) {
            l[selected[k]] = 12.0f - 4.0f * k;
            probabilities[k] = sqrt(log1p(exp(l[selected[k]]))); sum += probabilities[k];
        }
        for (unsigned k = 0; k < ACTIVE; ++k) expected_weights[k] = (float)(probabilities[k] / sum * 1.5);
        /* Poison IDs to prove downstream experts consume the preceding GPU
         * router results, without a host read or a command-buffer split. */
        memset(myds_metal_buffer_data(ids.view), 0xff, ids.bytes);
    }
    uint64_t hashes[ACTIVE][3];
    for (unsigned k = 0; k < ACTIVE; ++k) {
        hashes[k][0] = hash_bytes((uint8_t *)myds_metal_buffer_data(gate.view) + selected[k] * gate_expert, gate_expert);
        hashes[k][1] = hash_bytes((uint8_t *)myds_metal_buffer_data(up.view) + selected[k] * gate_expert, gate_expert);
        hashes[k][2] = hash_bytes((uint8_t *)myds_metal_buffer_data(down.view) + selected[k] * down_expert, down_expert);
    }
    const uint64_t input_hash = hash_bytes(input, x.bytes);
    CHECK(myds_metal_begin_batch());
    if (gpu_router) CHECK(myds_dspark_route_fused(ids.view, weights.view, logits.view, bias.view, 3, 1.5f, 1));
    CHECK(myds_dspark_moe(out.view, x.view, gate.view, up.view, down.view, ids.view, weights.view,
                          NULL, NULL, mid.view, NULL, dim, width, experts, ACTIVE,
                          GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, clamp_val));
    CHECK(myds_metal_commit_batch());
    CHECK(memcmp(myds_metal_buffer_data(ids.view), selected, sizeof(selected)) == 0);

    double *reference_mid = calloc(ACTIVE * width, sizeof(double));
    double *reference_out = calloc(dim, sizeof(double)); CHECK(reference_mid && reference_out);
    unsigned gate_high = 0, gate_low = 0, up_high = 0, up_low = 0;
    for (unsigned k = 0; k < ACTIVE; ++k) {
        const uint8_t *g = (uint8_t *)myds_metal_buffer_data(gate.view) + selected[k] * gate_expert;
        const uint8_t *u = (uint8_t *)myds_metal_buffer_data(up.view) + selected[k] * gate_expert;
        for (unsigned r = 0; r < width; ++r) {
            double vg = 0.0, vu = 0.0;
            const size_t row_bytes = (size_t)(dim / BLOCK) * Q2_BYTES;
            for (unsigned i = 0; i < dim; ++i) {
                vg += (double)q2_value(g + r * row_bytes, i) * input[i];
                vu += (double)q2_value(u + r * row_bytes, i) * input[i];
            }
            if (clamp_val > 1e-6f) {
                gate_high += vg > clamp_val; gate_low += vg < -clamp_val;
                up_high += vu > clamp_val; up_low += vu < -clamp_val;
                vg = fmin(vg, clamp_val); vu = fmax(-clamp_val, fmin(vu, clamp_val));
            }
            reference_mid[k * width + r] = vg / (1.0 + exp(-vg)) * vu * expected_weights[k];
        }
        const uint8_t *d = (uint8_t *)myds_metal_buffer_data(down.view) + selected[k] * down_expert;
        for (unsigned r = 0; r < dim; ++r) {
            const size_t row_bytes = (size_t)(width / BLOCK) * Q2_BYTES;
            double v = 0.0;
            for (unsigned i = 0; i < width; ++i)
                v += (double)q2_value(d + r * row_bytes, i) * reference_mid[k * width + i];
            reference_out[r] += v;
        }
    }
    if (clamp_val > 1e-6f) CHECK(gate_high && gate_low && up_high && up_low);
    printf("  Clamp coverage gate>limit=%u gate<-limit=%u up>limit=%u up<-limit=%u\n",
           gate_high, gate_low, up_high, up_low);
    compare("weighted SwiGLU", myds_metal_buffer_data(mid.view), reference_mid, ACTIVE * width, 2e-6, 3e-5);
    compare("sum3 output", myds_metal_buffer_data(out.view), reference_out, dim, 2e-4, 1e-4);
    CHECK(hash_bytes(input, x.bytes) == input_hash);
    for (unsigned k = 0; k < ACTIVE; ++k) {
        CHECK(hash_bytes((uint8_t *)myds_metal_buffer_data(gate.view) + selected[k] * gate_expert, gate_expert) == hashes[k][0]);
        CHECK(hash_bytes((uint8_t *)myds_metal_buffer_data(up.view) + selected[k] * gate_expert, gate_expert) == hashes[k][1]);
        CHECK(hash_bytes((uint8_t *)myds_metal_buffer_data(down.view) + selected[k] * down_expert, down_expert) == hashes[k][2]);
    }
    free(reference_mid); free(reference_out);
    free_guarded(gate); free_guarded(up); free_guarded(down); free_guarded(x);
    free_guarded(ids); free_guarded(weights); free_guarded(mid); free_guarded(out);
    if (gpu_router) { free_guarded(logits); free_guarded(bias); }
}

static void reject_contract(void) {
    struct guarded matrix = alloc_guarded(8u * 256u * Q2_BYTES);
    struct guarded x = alloc_guarded(256u * sizeof(float));
    struct guarded ids = alloc_guarded(6u * sizeof(int32_t));
    struct guarded weights = alloc_guarded(6u * sizeof(float));
    struct guarded mid = alloc_guarded(3u * 256u * sizeof(float));
    struct guarded out = alloc_guarded(256u * sizeof(float));
    memset(myds_metal_buffer_data(mid.view), 0xa5, mid.bytes);
    memset(myds_metal_buffer_data(out.view), 0xa5, out.bytes);
    myds_metal_buffer_t *short_matrix = myds_metal_buffer_sub_view(matrix.view, 0, matrix.bytes - 1u);
    myds_metal_buffer_t *short_mid = myds_metal_buffer_sub_view(mid.view, 0, mid.bytes - 4u);
    CHECK(short_matrix && short_mid);
    const uint64_t before = myds_metal_dispatch_count();
#define REJECT(g, u, d, m, dim, width, count, k, gt, ut, dt, clamp) \
    CHECK(!myds_dspark_moe(out.view, x.view, g, u, d, ids.view, weights.view, NULL, NULL, m, NULL, \
                           dim, width, count, k, gt, ut, dt, clamp))
    REJECT(short_matrix, matrix.view, matrix.view, mid.view, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, short_matrix, matrix.view, mid.view, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, short_matrix, mid.view, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, short_mid, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 255, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 256, 255, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 256, 256, 8, 6, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 256, 256, 8, 3, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, 10);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, NAN);
    REJECT(matrix.view, matrix.view, matrix.view, mid.view, 256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, -1);
#undef REJECT
    /* The target entry point must reject Q2_K gate/up and top-3 before its
     * hardcoded IQ2/sum6 graph can touch these tensors. */
    CHECK(!myds_metal_routed_moe(out.view, x.view, matrix.view, matrix.view, matrix.view,
                                ids.view, weights.view, mid.view, mid.view, mid.view, mid.view,
                                256, 256, 8, 3, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10));
    CHECK(!myds_metal_routed_moe(out.view, x.view, matrix.view, matrix.view, matrix.view,
                                ids.view, weights.view, mid.view, mid.view, mid.view, mid.view,
                                256, 256, 8, 6, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 10));
    CHECK(!myds_metal_routed_moe(out.view, x.view, matrix.view, matrix.view, matrix.view,
                                ids.view, weights.view, mid.view, mid.view, mid.view, mid.view,
                                256, 256, 8, 3, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10));
    CHECK(myds_metal_dispatch_count() == before);
    const uint8_t *mp = myds_metal_buffer_data(mid.view), *op = myds_metal_buffer_data(out.view);
    for (size_t i = 0; i < mid.bytes; ++i) CHECK(mp[i] == 0xa5);
    for (size_t i = 0; i < out.bytes; ++i) CHECK(op[i] == 0xa5);
    myds_metal_buffer_free(short_matrix); myds_metal_buffer_free(short_mid);
    free_guarded(matrix); free_guarded(x); free_guarded(ids);
    free_guarded(weights); free_guarded(mid); free_guarded(out);
}

static void test_shared_swiglu(unsigned width, float limit) {
    printf("[TEST] DSpark shared SwiGLU width=%u limit=%.1f F32 output\n", width, limit);
    struct guarded gate = alloc_guarded(width * sizeof(float));
    struct guarded up = alloc_guarded(width * sizeof(float));
    struct guarded out = alloc_guarded(width * sizeof(float));
    const float gate_cases[] = {100.0f, 12.0f, -12.0f, -100.0f, 0.333f, 0.0f, 9.95f, -1.25f};
    const float up_cases[] = {-100.0f, 100.0f, -100.0f, 3.12f, 0.173f, 20.0f, -9.95f, -0.173f};
    float *g = myds_metal_buffer_data(gate.view), *u = myds_metal_buffer_data(up.view);
    double *reference = calloc(width, sizeof(double)); CHECK(reference);
    for (unsigned i = 0; i < width; ++i) {
        g[i] = gate_cases[i % 8u]; u[i] = up_cases[i % 8u];
        double gc = g[i], uc = u[i];
        if (limit > 1e-6f) { gc = fmin(gc, limit); uc = fmax(-limit, fmin(uc, limit)); }
        reference[i] = gc / (1.0 + exp(-gc)) * uc;
    }
    const uint64_t gh = hash_bytes(g, gate.bytes), uh = hash_bytes(u, up.bytes);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_dspark_shared_swiglu(out.view, gate.view, up.view, width, limit));
    CHECK(myds_metal_commit_batch());
    compare("shared SwiGLU", myds_metal_buffer_data(out.view), reference, width, 2e-6, 2e-6);
    const float *actual = myds_metal_buffer_data(out.view); unsigned full_precision = 0;
    for (unsigned i = 0; i < width; ++i) {
        uint32_t bits; memcpy(&bits, actual + i, 4);
        full_precision += (bits & 0xffffu) != 0;
    }
    CHECK(full_precision > 0);
    CHECK(hash_bytes(g, gate.bytes) == gh && hash_bytes(u, up.bytes) == uh);
    myds_metal_buffer_t *short_out = myds_metal_buffer_sub_view(out.view, 0, out.bytes - 4u);
    myds_metal_buffer_t *short_gate = myds_metal_buffer_sub_view(gate.view, 0, gate.bytes - 4u);
    myds_metal_buffer_t *short_up = myds_metal_buffer_sub_view(up.view, 0, up.bytes - 4u);
    CHECK(short_out && short_gate && short_up);
    const uint64_t before = myds_metal_dispatch_count();
    CHECK(!myds_dspark_shared_swiglu(short_out, gate.view, up.view, width, limit));
    CHECK(!myds_dspark_shared_swiglu(out.view, short_gate, up.view, width, limit));
    CHECK(!myds_dspark_shared_swiglu(out.view, gate.view, short_up, width, limit));
    CHECK(!myds_dspark_shared_swiglu(out.view, gate.view, up.view, width, NAN));
    CHECK(!myds_dspark_shared_swiglu(out.view, gate.view, up.view, width, -1));
    CHECK(myds_metal_dispatch_count() == before);
    myds_metal_buffer_free(short_out); myds_metal_buffer_free(short_gate); myds_metal_buffer_free(short_up);
    free(reference); free_guarded(gate); free_guarded(up); free_guarded(out);
}

int main(void) {
    CHECK(myds_metal_init(NULL));
    reject_contract();
    test_shared_swiglu(13, 10.0f);
    test_shared_swiglu(2304, 10.0f);
    test_shared_swiglu(13, 0.0f);
    run_case(256, 256, 8, 10.0f, 0);
    run_case(512, 512, 8, 0.0f, 0);
    const int previous_auto_barrier = myds_metal_get_auto_barrier();
    const int previous_concurrent = myds_metal_get_concurrent();
    myds_metal_set_auto_barrier(0);
    myds_metal_set_concurrent(1);
    run_case(5120, 2304, 128, 10.0f, 1);
    myds_metal_set_auto_barrier(previous_auto_barrier);
    myds_metal_set_concurrent(previous_concurrent);
    myds_metal_cleanup();
    puts("DSpark Q2_K/top-3 MoE: scalar dequant/matvec, clamp, GPU router dependency, production shape, guards and rejection contract passed");
    return 0;
}
