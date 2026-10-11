/* Real Metal contract test for DSpark's five scalar causal-prefix queries.
 * The CPU oracle is independent of the GPU attention implementation. Every
 * input is exactly representable in F16, matching the attention staging
 * contract without hiding errors behind an approximate F16 conversion.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DRAFT 5u
#define HEADS 64u
#define WIDTH 512u
#define KEYS 6u
#define RING 128u
#define BLOCKS 3u
#define GUARD 16u
#define ROW_FLOATS (HEADS * WIDTH)
#define OUTPUT_FLOATS (DRAFT * ROW_FLOATS)
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return 0; } } while (0)

struct block {
    myds_metal_buffer_t *q, *window, *sinks, *storage, *ref_storage;
    myds_metal_buffer_t *out[DRAFT], *ref[DRAFT], *query[DRAFT];
    uint32_t raw_start;
};

static int create_block(struct block *b, uint32_t index) {
    memset(b, 0, sizeof(*b));
    b->raw_start = index == 0u ? 0u : index == 1u ? 127u : 124u;
    b->q = myds_metal_buffer_alloc((size_t)OUTPUT_FLOATS * sizeof(float));
    b->window = myds_metal_buffer_alloc((size_t)RING * WIDTH * sizeof(float));
    b->sinks = myds_metal_buffer_alloc(HEADS * sizeof(float));
    b->storage = myds_metal_buffer_alloc((size_t)(OUTPUT_FLOATS + 2u * GUARD) * sizeof(float));
    b->ref_storage = myds_metal_buffer_alloc((size_t)(OUTPUT_FLOATS + 2u * GUARD) * sizeof(float));
    CHECK(b->q && b->window && b->sinks && b->storage && b->ref_storage);
    memset(myds_metal_buffer_data(b->storage), 0xa5, myds_metal_buffer_size(b->storage));
    memset(myds_metal_buffer_data(b->ref_storage), 0xa5, myds_metal_buffer_size(b->ref_storage));
    float *q = myds_metal_buffer_data(b->q);
    float *window = myds_metal_buffer_data(b->window);
    float *sinks = myds_metal_buffer_data(b->sinks);
    /* Unused ring rows carry poison values, so reading a physical prefix or
     * a larger prefix accidentally cannot agree with the logical oracle. */
    for (size_t i = 0; i < RING * WIDTH; i++) window[i] = 8.0f;
    for (uint32_t k = 0; k < KEYS; k++) {
        float *row = window + (size_t)((b->raw_start + k) % RING) * WIDTH;
        for (uint32_t dim = 0; dim < WIDTH; dim++) {
            row[dim] = (float)((int)((dim + 3u * k) % 11u) - 5) / 32.0f + (float)k / 8.0f;
            if (index == 0u) row[dim] *= -0.5f;
            if (index == 2u && k == KEYS - 1u) row[dim] += 3.0625f;
        }
    }
    for (uint32_t head = 0; head < HEADS; head++) {
        sinks[head] = (float)((int)(head % 9u) - 4) / 4.0f;
        for (uint32_t slot = 0; slot < DRAFT; slot++) {
            for (uint32_t dim = 0; dim < WIDTH; dim++) {
                q[((size_t)slot * HEADS + head) * WIDTH + dim] = index == 0u ? 0.0f :
                    (float)((int)((dim + head + slot) % 13u) - 6) / 64.0f;
            }
        }
    }
    for (uint32_t slot = 0; slot < DRAFT; slot++) {
        const size_t offset = (size_t)slot * ROW_FLOATS * sizeof(float);
        b->query[slot] = myds_metal_buffer_sub_view(b->q, offset, ROW_FLOATS * sizeof(float));
        b->out[slot] = myds_metal_buffer_sub_view(b->storage, GUARD * sizeof(float) + offset, ROW_FLOATS * sizeof(float));
        b->ref[slot] = myds_metal_buffer_sub_view(b->ref_storage, GUARD * sizeof(float) + offset, ROW_FLOATS * sizeof(float));
        CHECK(b->query[slot] && b->out[slot] && b->ref[slot]);
    }
    return 1;
}

static void free_block(struct block *b) {
    for (uint32_t slot = 0; slot < DRAFT; slot++) {
        myds_metal_buffer_free(b->query[slot]);
        myds_metal_buffer_free(b->out[slot]);
        myds_metal_buffer_free(b->ref[slot]);
    }
    myds_metal_buffer_free(b->q);
    myds_metal_buffer_free(b->window);
    myds_metal_buffer_free(b->sinks);
    myds_metal_buffer_free(b->storage);
    myds_metal_buffer_free(b->ref_storage);
}

static int guards_intact(myds_metal_buffer_t *storage) {
    const unsigned char *p = myds_metal_buffer_data(storage);
    const size_t bytes = GUARD * sizeof(float);
    for (size_t i = 0; i < bytes; i++) {
        CHECK(p[i] == 0xa5u);
        CHECK(p[bytes + OUTPUT_FLOATS * sizeof(float) + i] == 0xa5u);
    }
    return 1;
}

static int compare_oracle(struct block *b, double *max_error) {
    const float *q = myds_metal_buffer_data(b->q);
    const float *window = myds_metal_buffer_data(b->window);
    const float *sinks = myds_metal_buffer_data(b->sinks);
    for (uint32_t slot = 0; slot < DRAFT; slot++) {
        const float *out = myds_metal_buffer_data(b->out[slot]);
        const float *reference = myds_metal_buffer_data(b->ref[slot]);
        CHECK(!memcmp(out, reference, ROW_FLOATS * sizeof(float)));
        for (uint32_t head = 0; head < HEADS; head++) {
            /* k <= slot + 1: exactly the 5 x 6 block-causal mask, including
             * the committed target feature and query's own draft row. */
            const uint32_t visible = slot + 2u;
            double logits[KEYS], weights[KEYS], maximum = sinks[head];
            for (uint32_t k = 0; k < visible; k++) {
                const float *kv = window + (size_t)((b->raw_start + k) % RING) * WIDTH;
                double dot = 0.0;
                for (uint32_t dim = 0; dim < WIDTH; dim++)
                    dot += (double)q[((size_t)slot * HEADS + head) * WIDTH + dim] * kv[dim];
                logits[k] = dot / sqrt((double)WIDTH);
                if (logits[k] > maximum) maximum = logits[k];
            }
            double denominator = exp((double)sinks[head] - maximum);
            for (uint32_t k = 0; k < visible; k++) {
                weights[k] = exp(logits[k] - maximum);
                denominator += weights[k];
            }
            for (uint32_t dim = 0; dim < WIDTH; dim++) {
                double numerator = 0.0;
                for (uint32_t k = 0; k < visible; k++)
                    numerator += weights[k] * window[(size_t)((b->raw_start + k) % RING) * WIDTH + dim];
                const double expected = numerator / denominator;
                const float got = out[(size_t)head * WIDTH + dim];
                const double error = fabs((double)got - expected);
                if (error > *max_error) *max_error = error;
                if (!isfinite(got) || error > 3e-5 * (1.0 + fabs(expected))) {
                    fprintf(stderr, "oracle slot=%u head=%u dim=%u got=%.9g expected=%.12g\n", slot, head, dim, got, expected);
                    return 0;
                }
            }
        }
    }
    CHECK(guards_intact(b->storage));
    CHECK(guards_intact(b->ref_storage));
    return 1;
}

static int attention(struct block *b, uint32_t slot, int reference, myds_v41_gpu_scratch_t *scratch) {
    return myds_v41_gpu_attention_direct(reference ? b->ref[slot] : b->out[slot], b->sinks,
        b->query[slot], b->window, slot + 2u, b->raw_start,
        NULL, NULL, 0u, 0u, scratch);
}

static int test_case(int packed, int head_first, int concurrent, int automatic) {
    CHECK(!setenv("MYDS_ATTN_PACKED_REDUCE", packed ? "1" : "0", 1));
    CHECK(!setenv("MYDS_ATTN_VEC_HEAD_FIRST", head_first ? "1" : "0", 1));
    struct block blocks[BLOCKS];
    for (uint32_t block = 0; block < BLOCKS; block++) CHECK(create_block(&blocks[block], block));
    myds_v41_gpu_scratch_t *scratch = myds_v41_gpu_scratch_create(2u);
    myds_v41_gpu_scratch_t *reference = myds_v41_gpu_scratch_create(2u);
    CHECK(scratch && reference);
    /* Separate completed dispatches are the ordering reference. */
    myds_metal_set_concurrent(0);
    myds_metal_set_auto_barrier(1);
    for (uint32_t block = 0; block < BLOCKS; block++) {
        for (uint32_t slot = 0; slot < DRAFT; slot++) CHECK(attention(&blocks[block], slot, 1, reference));
    }
    myds_metal_set_concurrent(concurrent);
    myds_metal_set_auto_barrier(automatic);
    CHECK(myds_metal_begin_batch());
    const uint64_t before = myds_metal_dispatch_count();
    /* Fifteen queries share kv_f16, pad and partial. There is deliberately
     * no caller barrier between queries: scratch reuse belongs to the API.
     * CPU writes finished before encoding, so no CPU/GPU ownership race is
     * introduced by the test fixture itself. */
    for (uint32_t block = 0; block < BLOCKS; block++) {
        for (uint32_t slot = 0; slot < DRAFT; slot++) CHECK(attention(&blocks[block], slot, 0, scratch));
    }
    CHECK(myds_metal_dispatch_count() > before);
    CHECK(myds_metal_commit_batch());
    double max_error = 0.0;
    for (uint32_t block = 0; block < BLOCKS; block++) CHECK(compare_oracle(&blocks[block], &max_error));
    /* Only future key row five differs. Earlier slots must be byte-identical;
     * the last slot must actually consume the changed key/value. */
    for (uint32_t slot = 0; slot + 1u < DRAFT; slot++)
        CHECK(!memcmp(myds_metal_buffer_data(blocks[1].out[slot]), myds_metal_buffer_data(blocks[2].out[slot]), ROW_FLOATS * sizeof(float)));
    CHECK(memcmp(myds_metal_buffer_data(blocks[1].out[DRAFT - 1u]), myds_metal_buffer_data(blocks[2].out[DRAFT - 1u]), ROW_FLOATS * sizeof(float)));
    printf("[PASS] DSpark scalar-prefix attention packed=%d head_first=%d concurrent=%d auto=%d: 15 same-CB queries, CPU oracle max_abs=%.3g, serial byte-exact, wrap/future mask/canaries\n",
           packed, head_first, concurrent, automatic, max_error);
    myds_v41_gpu_scratch_free(scratch);
    myds_v41_gpu_scratch_free(reference);
    for (uint32_t block = 0; block < BLOCKS; block++) free_block(&blocks[block]);
    return 1;
}

static float e4m3_value(unsigned code) {
    const unsigned exponent = code >> 3u, mantissa = code & 7u;
    return exponent == 0u ? (float)mantissa / 512.0f :
        ldexpf(1.0f + (float)mantissa / 8.0f, (int)exponent - 7);
}

/* Exhaustive nearest-value search, independent of the shader's binary
 * search. Exactly halfway cases choose the even E4M3FN encoding. */
static float quant_cpu(float value, float scale) {
    const float target = fminf(fabsf(value / scale), 448.0f);
    unsigned best = 0u;
    float distance = INFINITY;
    for (unsigned code = 0u; code <= 126u; code++) {
        const float error = fabsf(target - e4m3_value(code));
        if (error < distance || (error == distance && !(code & 1u))) {
            distance = error;
            best = code;
        }
    }
    return (value < 0.0f ? -1.0f : 1.0f) * e4m3_value(best) * scale;
}

static int precision_case(int concurrent, int automatic) {
    const uint32_t heads = 3u, rows = 3u, position = 257u;
    const size_t rope_floats = heads * WIDTH, quant_floats = rows * WIDTH;
    myds_metal_buffer_t *rope_storage = myds_metal_buffer_alloc((rope_floats + 2u * GUARD) * sizeof(float));
    myds_metal_buffer_t *quant_storage = myds_metal_buffer_alloc((quant_floats + 2u * GUARD) * sizeof(float));
    myds_metal_buffer_t *norm_in = myds_metal_buffer_alloc(WIDTH * sizeof(float));
    myds_metal_buffer_t *norm_out = myds_metal_buffer_alloc(WIDTH * sizeof(float));
    myds_metal_buffer_t *norm_weight = myds_metal_buffer_alloc(WIDTH * sizeof(float));
    CHECK(rope_storage && quant_storage && norm_in && norm_out && norm_weight);
    memset(myds_metal_buffer_data(rope_storage), 0xa5, myds_metal_buffer_size(rope_storage));
    memset(myds_metal_buffer_data(quant_storage), 0xa5, myds_metal_buffer_size(quant_storage));
    myds_metal_buffer_t *rope = myds_metal_buffer_sub_view(rope_storage, GUARD * sizeof(float), rope_floats * sizeof(float));
    myds_metal_buffer_t *quant = myds_metal_buffer_sub_view(quant_storage, GUARD * sizeof(float), quant_floats * sizeof(float));
    CHECK(rope && quant);
    float *rp = myds_metal_buffer_data(rope), *qp = myds_metal_buffer_data(quant);
    float *ni = myds_metal_buffer_data(norm_in), *nw = myds_metal_buffer_data(norm_weight);
    float *original = malloc(rope_floats * sizeof(float));
    float *quant_reference = malloc(quant_floats * sizeof(float));
    CHECK(original && quant_reference);
    for (size_t i = 0; i < rope_floats; i++) rp[i] = (float)((int)(i % 71u) - 35) * 0.01371f;
    memcpy(original, rp, rope_floats * sizeof(float));
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t block = 0; block < 7u; block++) {
            const float scale = ldexpf(1.0f, (int)block - 12 + (int)row);
            for (uint32_t col = 0; col < 64u; col++) {
                const unsigned code = (block * 63u + col + row * 17u) % 126u;
                const float midpoint = (e4m3_value(code) + e4m3_value(code + 1u)) * 0.5f;
                const float value = (col & 1u ? -midpoint : midpoint) * scale;
                qp[(size_t)row * WIDTH + block * 64u + col] = value;
            }
            /* The anchor fixes this 64-column scale. Its second-half
             * placement catches accidental 32-column quantization blocks. */
            qp[(size_t)row * WIDTH + block * 64u + 63u] = 448.0f * scale;
            for (uint32_t col = 0; col < 64u; col++) {
                const size_t i = (size_t)row * WIDTH + block * 64u + col;
                quant_reference[i] = quant_cpu(qp[i], scale);
            }
        }
        for (uint32_t col = 448u; col < WIDTH; col++) {
            const size_t i = (size_t)row * WIDTH + col;
            qp[i] = (float)((int)col - 477) * 0.01731f;
            quant_reference[i] = qp[i];
        }
    }
    double square_sum = 0.0;
    for (uint32_t i = 0; i < WIDTH; i++) {
        ni[i] = (float)((int)(i % 41u) - 20) * 0.02139f;
        nw[i] = 0.93731f + (float)(i % 5u) * 0.03971f;
        square_sum += (double)ni[i] * ni[i];
    }
    myds_metal_set_concurrent(concurrent);
    myds_metal_set_auto_barrier(automatic);
    CHECK(myds_metal_begin_batch());
    const uint64_t before = myds_metal_dispatch_count();
    CHECK(!myds_dspark_gpu_rope(rope, 63u, heads, position, false));
    CHECK(!myds_dspark_gpu_rope(rope, WIDTH, heads, 1048576u, false));
    CHECK(!myds_dspark_gpu_quant_kv(quant, rows + 1u));
    CHECK(myds_metal_dispatch_count() == before);
    CHECK(!memcmp(rp, original, rope_floats * sizeof(float)));
    CHECK(myds_dspark_gpu_rope(rope, WIDTH, heads, position, false));
    CHECK(myds_dspark_gpu_quant_kv(quant, rows));
    CHECK(myds_metal_v41_rmsnorm(norm_out, norm_in, norm_weight, WIDTH, 1e-6f, false));
    CHECK(myds_metal_commit_batch());
    unsigned f32_norm_values = 0u;
    const float *no = myds_metal_buffer_data(norm_out);
    const double norm_scale = 1.0 / sqrt(square_sum / WIDTH + 1e-6);
    for (uint32_t i = 0; i < WIDTH; i++) {
        const double expected = ni[i] * norm_scale * nw[i];
        CHECK(isfinite(no[i]) && fabs((double)no[i] - expected) < 3e-6 * (1.0 + fabs(expected)));
        uint32_t bits; memcpy(&bits, no + i, sizeof(bits));
        f32_norm_values += (bits & 65535u) != 0u;
    }
    CHECK(f32_norm_values > WIDTH / 2u);
    for (uint32_t head = 0; head < heads; head++) {
        CHECK(!memcmp(rp + head * WIDTH, original + head * WIDTH, 448u * sizeof(float)));
        for (uint32_t lane = 0; lane < 32u; lane++) {
            const size_t i = (size_t)head * WIDTH + 448u + 2u * lane;
            const float frequency = 1.0f / powf(10000.0f, (float)lane / 32.0f);
            const float theta = (float)position * frequency;
            const float c = cosf(theta), s = sinf(theta);
            const float re = original[i], im = original[i + 1u];
            CHECK(fabsf(rp[i] - (re * c - im * s)) < 4e-6f);
            CHECK(fabsf(rp[i + 1u] - (re * s + im * c)) < 4e-6f);
        }
    }
    CHECK(!memcmp(qp, quant_reference, quant_floats * sizeof(float)));
    /* The in-place API must join its forward dispatch before inverse writes
     * even when the caller uses a concurrent encoder with auto0. */
    CHECK(myds_metal_begin_batch());
    CHECK(myds_dspark_gpu_rope(rope, WIDTH, heads, position, true));
    CHECK(myds_dspark_gpu_rope(rope, WIDTH, heads, 0u, false));
    CHECK(myds_metal_commit_batch());
    for (size_t i = 0; i < rope_floats; i++) CHECK(isfinite(rp[i]) && fabsf(rp[i] - original[i]) < 4e-6f);
    const unsigned char *rg = myds_metal_buffer_data(rope_storage), *qg = myds_metal_buffer_data(quant_storage);
    for (size_t i = 0; i < GUARD * sizeof(float); i++) {
        CHECK(rg[i] == 0xa5u && rg[(rope_floats + GUARD) * sizeof(float) + i] == 0xa5u);
        CHECK(qg[i] == 0xa5u && qg[(quant_floats + GUARD) * sizeof(float) + i] == 0xa5u);
    }
    printf("[PASS] DSpark F32 precision concurrent=%d auto=%d: weighted norm, forward/inverse RoPE, 64-wide E4M3 ties/scales, unchanged tail, invalid preflight, views/canaries\n", concurrent, automatic);
    free(original); free(quant_reference);
    myds_metal_buffer_free(rope); myds_metal_buffer_free(quant);
    myds_metal_buffer_free(rope_storage); myds_metal_buffer_free(quant_storage);
    myds_metal_buffer_free(norm_in); myds_metal_buffer_free(norm_out); myds_metal_buffer_free(norm_weight);
    return 1;
}

int main(void) {
    /* Control the environment before Metal caches its concurrent policy. */
    if (unsetenv("MYDS_METAL_CONCURRENT") || setenv("MYDS_ATTN_PACKED_NSG", "8", 1)) return 1;
    if (!myds_metal_init("metal")) return 1;
    int ok = 1;
    for (int path = 0; path < 3 && ok; path++) {
        for (int concurrent = 0; concurrent <= 1 && ok; concurrent++) {
            for (int automatic = 0; automatic <= 1 && ok; automatic++)
                ok = test_case(path == 2, path == 1, concurrent, automatic);
        }
    }
    for (int concurrent = 0; concurrent <= 1 && ok; concurrent++) {
        for (int automatic = 0; automatic <= 1 && ok; automatic++)
            ok = precision_case(concurrent, automatic);
    }
    myds_metal_set_concurrent(0);
    myds_metal_set_auto_barrier(1);
    myds_metal_cleanup();
    if (ok) puts("ALL DSPARK ATTENTION GPU CONTRACTS PASSED (12 attention cases, 180 queries, 4 precision cases)");
    return ok ? 0 : 1;
}
