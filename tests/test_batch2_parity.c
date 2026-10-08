/* =================================================================
 *  test_batch2_parity.c: Phase 3 Bit-Exact Verification Suite
 *  Verifies that Per-Layer Batch-2 GEMV & MoE produces 100% bit-exact
 *  parity (memcmp == 0) against sequential single-token decoding.
 * ================================================================= */

#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include "myds_v41_moe.h"
#include "myds_metal.h"
#include "myds_log.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define CHECK(c) do { \
    if (!(c)) { \
        fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
        return 0; \
    } \
} while (0)

struct guarded {
    myds_metal_buffer_t *owner;
    myds_metal_buffer_t *view;
    size_t bytes;
};

static int guarded_create(struct guarded *b, size_t bytes) {
    b->bytes = bytes;
    b->owner = myds_metal_buffer_alloc(bytes + 32);
    CHECK(b->owner);
    memset(myds_metal_buffer_data(b->owner), 0xa5, bytes + 32);
    b->view = myds_metal_buffer_sub_view(b->owner, 16, bytes);
    CHECK(b->view);
    return 1;
}

static int guarded_ok(struct guarded *b) {
    const uint8_t *p = (const uint8_t *)myds_metal_buffer_data(b->owner);
    for (size_t i = 0; i < 16; i++) {
        CHECK(p[i] == 0xa5 && p[16 + b->bytes + i] == 0xa5);
    }
    return 1;
}

static void guarded_free(struct guarded *b) {
    if (b->view) myds_metal_buffer_free(b->view);
    if (b->owner) myds_metal_buffer_free(b->owner);
}

static void fill_act(myds_metal_buffer_t *b, unsigned seed) {
    float *p = (float *)myds_metal_buffer_data(b);
    size_t count = myds_metal_buffer_size(b) / sizeof(float);
    for (size_t i = 0; i < count; i++) {
        p[i] = ((int)((i * 19u + seed) % 257u) - 128) / 512.0f;
    }
}

static void fill_q8_weights(myds_metal_buffer_t *b) {
    uint8_t *p = (uint8_t *)myds_metal_buffer_data(b);
    size_t n_blocks = myds_metal_buffer_size(b) / 34;
    for (size_t i = 0; i < n_blocks; i++) {
        uint16_t d = (uint16_t)(0x2400u + (i % 4) * 0x100u);
        memcpy(p + i * 34, &d, 2);
        for (size_t j = 0; j < 32; j++) {
            p[i * 34 + 2 + j] = (uint8_t)(int8_t)((int)((i * 13 + j * 29) % 255) - 127);
        }
    }
}

static int verify_exact(const myds_metal_buffer_t *a, const myds_metal_buffer_t *b, size_t count) {
    const float *x = (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)a);
    const float *y = (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)b);
    size_t nonzero = 0;
    for (size_t i = 0; i < count; i++) {
        if (memcmp(x + i, y + i, sizeof(float)) != 0) {
            uint32_t u, v;
            memcpy(&u, x + i, 4);
            memcpy(&v, y + i, 4);
            fprintf(stderr, "Bit mismatch at index %zu: seq=%08x (%.7f) batch2=%08x (%.7f)\n",
                    i, u, x[i], v, y[i]);
            return 0;
        }
        CHECK(isfinite(x[i]));
        if (x[i] != 0.0f) nonzero++;
    }
    CHECK(nonzero > 0);
    return 1;
}

/* Test 1: Verify QKV and Output GEMV projections (Batch-2 vs Sequential) */
static int test_batch2_gemv_projections(uint32_t rows, uint32_t width) {
    printf("[TEST] Batch-2 GEMV Projections (rows=%u, width=%u)...\n", rows, width);
    struct guarded w, x_seq0, x_seq1, x_batch2, y_seq, y_batch2;

    CHECK(guarded_create(&w, (size_t)rows * width / 32 * 34));
    CHECK(guarded_create(&x_seq0, (size_t)width * sizeof(float)));
    CHECK(guarded_create(&x_seq1, (size_t)width * sizeof(float)));
    CHECK(guarded_create(&x_batch2, 2 * (size_t)width * sizeof(float)));
    CHECK(guarded_create(&y_seq, 2 * (size_t)rows * sizeof(float)));
    CHECK(guarded_create(&y_batch2, 2 * (size_t)rows * sizeof(float)));

    fill_q8_weights(w.view);
    fill_act(x_seq0.view, 101);
    fill_act(x_seq1.view, 202);

    // Pack into x_batch2
    float *xb = (float *)myds_metal_buffer_data(x_batch2.view);
    memcpy(xb, myds_metal_buffer_data(x_seq0.view), width * sizeof(float));
    memcpy(xb + width, myds_metal_buffer_data(x_seq1.view), width * sizeof(float));

    // 1. Sequential execution
    CHECK(myds_metal_begin_batch());
    myds_metal_buffer_t *y_seq0 = myds_metal_buffer_sub_view(y_seq.view, 0, rows * sizeof(float));
    myds_metal_buffer_t *y_seq1 = myds_metal_buffer_sub_view(y_seq.view, rows * sizeof(float), rows * sizeof(float));
    CHECK(myds_metal_gemv(y_seq0, w.view, x_seq0.view, rows, width, GGML_TYPE_Q8_0));
    CHECK(myds_metal_gemv(y_seq1, w.view, x_seq1.view, rows, width, GGML_TYPE_Q8_0));
    CHECK(myds_metal_commit_batch());
    myds_metal_buffer_free(y_seq0);
    myds_metal_buffer_free(y_seq1);

    // 2. Batch-2 execution
    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_gemv_batch(y_batch2.view, w.view, x_batch2.view, rows, width, 2, GGML_TYPE_Q8_0));
    CHECK(myds_metal_commit_batch());

    // 3. Bit-exact comparison (memcmp == 0) across all 2 * rows outputs
    CHECK(verify_exact(y_seq.view, y_batch2.view, 2 * (size_t)rows));
    CHECK(guarded_ok(&w) && guarded_ok(&x_seq0) && guarded_ok(&x_seq1) &&
          guarded_ok(&x_batch2) && guarded_ok(&y_seq) && guarded_ok(&y_batch2));

    guarded_free(&w);
    guarded_free(&x_seq0);
    guarded_free(&x_seq1);
    guarded_free(&x_batch2);
    guarded_free(&y_seq);
    guarded_free(&y_batch2);

    printf("  [PASS] Batch-2 GEMV (rows=%u, width=%u) 100%% bit-exact!\n", rows, width);
    return 1;
}

/* Test 2: MoE Route Batch-2 vs Sequential */
static int test_batch2_moe_route(void) {
    printf("[TEST] Batch-2 MoE Fused Router (rows=2 vs rows=1x2)...\n");
    const uint32_t experts = 384;
    const uint32_t top_k = 6;
    const float scale = 1.50f;

    struct guarded logits_seq0, logits_seq1, logits_batch2, bias;
    struct guarded sel_seq, w_seq, p_seq, sc_seq;
    struct guarded sel_b2, w_b2, p_b2, sc_b2;

    CHECK(guarded_create(&logits_seq0, experts * sizeof(float)));
    CHECK(guarded_create(&logits_seq1, experts * sizeof(float)));
    CHECK(guarded_create(&logits_batch2, 2 * experts * sizeof(float)));
    CHECK(guarded_create(&bias, experts * sizeof(float)));

    CHECK(guarded_create(&sel_seq, 2 * top_k * sizeof(int32_t)));
    CHECK(guarded_create(&w_seq, 2 * top_k * sizeof(float)));
    CHECK(guarded_create(&p_seq, 2 * experts * sizeof(float)));
    CHECK(guarded_create(&sc_seq, 2 * experts * sizeof(float)));

    CHECK(guarded_create(&sel_b2, 2 * top_k * sizeof(int32_t)));
    CHECK(guarded_create(&w_b2, 2 * top_k * sizeof(float)));
    CHECK(guarded_create(&p_b2, 2 * experts * sizeof(float)));
    CHECK(guarded_create(&sc_b2, 2 * experts * sizeof(float)));

    fill_act(logits_seq0.view, 333);
    fill_act(logits_seq1.view, 444);
    fill_act(bias.view, 555);

    float *lb = (float *)myds_metal_buffer_data(logits_batch2.view);
    memcpy(lb, myds_metal_buffer_data(logits_seq0.view), experts * sizeof(float));
    memcpy(lb + experts, myds_metal_buffer_data(logits_seq1.view), experts * sizeof(float));

    // 1. Sequential route Token 0 and Token 1
    myds_metal_buffer_t *s0 = myds_metal_buffer_sub_view(sel_seq.view, 0, top_k * sizeof(int32_t));
    myds_metal_buffer_t *w0 = myds_metal_buffer_sub_view(w_seq.view, 0, top_k * sizeof(float));
    myds_metal_buffer_t *p0 = myds_metal_buffer_sub_view(p_seq.view, 0, experts * sizeof(float));
    myds_metal_buffer_t *sc0 = myds_metal_buffer_sub_view(sc_seq.view, 0, experts * sizeof(float));

    myds_metal_buffer_t *s1 = myds_metal_buffer_sub_view(sel_seq.view, top_k * sizeof(int32_t), top_k * sizeof(int32_t));
    myds_metal_buffer_t *w1 = myds_metal_buffer_sub_view(w_seq.view, top_k * sizeof(float), top_k * sizeof(float));
    myds_metal_buffer_t *p1 = myds_metal_buffer_sub_view(p_seq.view, experts * sizeof(float), experts * sizeof(float));
    myds_metal_buffer_t *sc1 = myds_metal_buffer_sub_view(sc_seq.view, experts * sizeof(float), experts * sizeof(float));

    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_route_batch(s0, w0, p0, sc0, logits_seq0.view, bias.view, experts, top_k, scale, 1));
    CHECK(myds_v41_route_batch(s1, w1, p1, sc1, logits_seq1.view, bias.view, experts, top_k, scale, 1));
    CHECK(myds_metal_commit_batch());

    myds_metal_buffer_free(s0); myds_metal_buffer_free(w0); myds_metal_buffer_free(p0); myds_metal_buffer_free(sc0);
    myds_metal_buffer_free(s1); myds_metal_buffer_free(w1); myds_metal_buffer_free(p1); myds_metal_buffer_free(sc1);

    // 2. Batch-2 route (rows = 2)
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_route_batch(sel_b2.view, w_b2.view, p_b2.view, sc_b2.view,
                               logits_batch2.view, bias.view, experts, top_k, scale, 2));
    CHECK(myds_metal_commit_batch());

    // 3. Compare selected IDs & weights
    const int32_t *ids_seq = (const int32_t *)myds_metal_buffer_data(sel_seq.view);
    const int32_t *ids_b2 = (const int32_t *)myds_metal_buffer_data(sel_b2.view);
    for (size_t i = 0; i < 2 * top_k; i++) {
        CHECK(ids_seq[i] == ids_b2[i]);
    }
    CHECK(verify_exact(w_seq.view, w_b2.view, 2 * top_k));
    CHECK(verify_exact(p_seq.view, p_b2.view, 2 * experts));

    guarded_free(&logits_seq0); guarded_free(&logits_seq1); guarded_free(&logits_batch2); guarded_free(&bias);
    guarded_free(&sel_seq); guarded_free(&w_seq); guarded_free(&p_seq); guarded_free(&sc_seq);
    guarded_free(&sel_b2); guarded_free(&w_b2); guarded_free(&p_b2); guarded_free(&sc_b2);

    printf("  [PASS] Batch-2 MoE Fused Router 100%% bit-exact!\n");
    return 1;
}

int main(void) {
    printf("\n=======================================================\n");
    printf("   Phase 3: Metal Per-Layer Batch-2 Parity Test Suite  \n");
    printf("=======================================================\n");

    if (!myds_metal_init(NULL)) {
        fprintf(stderr, "Failed to initialize Metal\n");
        return 1;
    }

    // 1. Attention Q_A projection (1280 x 5120)
    CHECK(test_batch2_gemv_projections(1280, 5120));

    // 2. Attention KV_A projection (512 x 5120)
    CHECK(test_batch2_gemv_projections(512, 5120));

    // 3. Attention Q_B projection (32768 x 1280)
    CHECK(test_batch2_gemv_projections(32768, 1280));

    // 4. Attention Output B projection (5120 x 8192)
    CHECK(test_batch2_gemv_projections(5120, 8192));

    // 5. MoE Shared Expert Gate/Up (2304 x 5120)
    CHECK(test_batch2_gemv_projections(2304, 5120));

    // 6. MoE Shared Expert Down (5120 x 2304)
    CHECK(test_batch2_gemv_projections(5120, 2304));

    // 7. MoE Router Gate (384 x 5120)
    CHECK(test_batch2_gemv_projections(384, 5120));

    // 8. MoE Fused Router (rows = 2)
    CHECK(test_batch2_moe_route());

    printf("\n=== ALL PHASE 3 BATCH-2 PARITY TESTS PASSED CLEANLY! ===\n\n");
    return 0;
}
