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
#include "myds_v41.h"
#include "myds_v41_decode.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

static void fill_q4_k_weights(myds_metal_buffer_t *b) {
    uint8_t *p = (uint8_t *)myds_metal_buffer_data(b);
    size_t bytes = myds_metal_buffer_size(b);
    for (size_t i = 0; i < bytes; i++) {
        p[i] = (uint8_t)((i * 37 + 101) & 0xff);
    }
    size_t n_blocks = bytes / 144;
    for (size_t blk = 0; blk < n_blocks; blk++) {
        uint16_t *d_ptr = (uint16_t *)(p + blk * 144);
        uint16_t *dmin_ptr = (uint16_t *)(p + blk * 144 + 2);
        *d_ptr = 0x3c00; // 1.0 in FP16
        *dmin_ptr = 0x3800; // 0.5 in FP16
    }
}

static int test_authoritative_projection(uint32_t rows, uint32_t width,
    ggml_type type, myds_v41_batch2_projection_kind_t kind) {
    printf("[TEST] Production decode projection boundary kind=%u rows=%u width=%u type=%u...\n",
           kind, rows, width, type);
    const size_t w_bytes = type == GGML_TYPE_Q8_0 ? (size_t)rows * width / 32u * 34u
                         : (type == GGML_TYPE_Q4_K ? (size_t)rows * width / 256u * 144u
                         : (type == GGML_TYPE_F16 ? (size_t)rows * width * sizeof(uint16_t)
                                                  : (size_t)rows * width * sizeof(float)));
    myds_metal_buffer_t *weight = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *input[2], *reference[2], *candidate[2], *raw[2];
    CHECK(weight);
    if (type == GGML_TYPE_Q8_0) {
        fill_q8_weights(weight);
    } else if (type == GGML_TYPE_Q4_K) {
        fill_q4_k_weights(weight);
    } else if (type == GGML_TYPE_F16) {
        uint16_t *p = (uint16_t *)myds_metal_buffer_data(weight);
        for (size_t i = 0; i < (size_t)rows * width; i++) p[i] = (uint16_t)(0x3000u + (i % 17));
    } else {
        fill_act(weight, 17);
    }
    for (uint32_t t = 0; t < 2; ++t) {
        input[t] = myds_metal_buffer_alloc(width * sizeof(float));
        reference[t] = myds_metal_buffer_alloc(rows * sizeof(float));
        candidate[t] = myds_metal_buffer_alloc(rows * sizeof(float));
        raw[t] = myds_metal_buffer_alloc(rows * sizeof(float));
        CHECK(input[t] && reference[t] && candidate[t] && raw[t]);
        fill_act(input[t], 101u + 101u * t);
    }
    CHECK(myds_metal_begin_batch());
    for (uint32_t t = 0; t < 2; ++t) {
        if (kind == MYDS_V41_BATCH2_QKV)
            CHECK(myds_metal_v41_gemv_bf16(reference[t], weight, input[t], rows, width, type));
        else if (kind == MYDS_V41_BATCH2_ROUTER || kind == MYDS_V41_BATCH2_LM_HEAD)
            CHECK(myds_metal_gemv(reference[t], weight, input[t], rows, width, type));
        else
            CHECK(myds_v41_decode_gemv_bf16(reference[t], weight, input[t], rows, width, type, false));
        CHECK(myds_metal_gemv(raw[t], weight, input[t], rows, width, type));
    }
    CHECK(myds_metal_commit_batch());
    myds_metal_batch2_q8_shared_counters_t before, after;
    myds_metal_batch2_q8_shared_counters(&before, false);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_batch2_projection_probe(candidate, weight, input, rows, width, type, kind));
    CHECK(myds_metal_commit_batch());
    myds_metal_batch2_q8_shared_counters(&after, false);
    if ((type == GGML_TYPE_Q8_0 && myds_metal_batch2_q8_shared_mode() > 0 &&
         (kind == MYDS_V41_BATCH2_LM_HEAD ||
          ((kind == MYDS_V41_BATCH2_QKV || kind == MYDS_V41_BATCH2_TAIL || kind == MYDS_V41_BATCH2_SHARED) &&
           myds_v41_decode_q8_fuse_bf16_enabled()))) ||
        (type == GGML_TYPE_Q4_K && myds_metal_batch2_q4_shared_mode() > 0 &&
         (kind == MYDS_V41_BATCH2_QKV || kind == MYDS_V41_BATCH2_TAIL || kind == MYDS_V41_BATCH2_SHARED || kind == MYDS_V41_BATCH2_LM_HEAD))) {
        CHECK(after.encoded > before.encoded);
    }
    for (uint32_t t = 0; t < 2; ++t) {
        CHECK(verify_exact(reference[t], candidate[t], rows));
        if (kind != MYDS_V41_BATCH2_ROUTER && kind != MYDS_V41_BATCH2_LM_HEAD) {
            CHECK(memcmp(myds_metal_buffer_data(raw[t]), myds_metal_buffer_data(candidate[t]), rows * sizeof(float)) != 0);
            const uint32_t *bits = myds_metal_buffer_data(candidate[t]);
            for (uint32_t r = 0; r < rows; ++r) CHECK((bits[r] & 0xffffu) == 0);
        }
        myds_metal_buffer_free(input[t]); myds_metal_buffer_free(reference[t]);
        myds_metal_buffer_free(candidate[t]); myds_metal_buffer_free(raw[t]);
    }
    myds_metal_buffer_free(weight);
    return 1;
}

static int test_batch2_attn_output_a(void) {
    printf("[TEST] Batch-2 Attention Output A (8 groups x 1024 rank x 4096 dim)...\n");
    const uint32_t n_groups = 8;
    const uint32_t group_dim = 4096;
    const uint32_t rank = 1024;
    const size_t out_a_bytes = (size_t)n_groups * rank * (group_dim / 256u) * 144u;
    const size_t heads_per_tok = (size_t)n_groups * group_dim; // 32768 floats
    const size_t low_per_tok = (size_t)n_groups * rank;         // 8192 floats

    myds_metal_buffer_t *out_a = myds_metal_buffer_alloc(out_a_bytes);
    myds_metal_buffer_t *heads_seq[2], *low_seq[2];
    myds_metal_buffer_t *heads_b2 = myds_metal_buffer_alloc(2 * heads_per_tok * sizeof(float));
    myds_metal_buffer_t *low_b2 = myds_metal_buffer_alloc(2 * low_per_tok * sizeof(float));
    CHECK(out_a && heads_b2 && low_b2);

    fill_q4_k_weights(out_a);

    for (uint32_t t = 0; t < 2; ++t) {
        heads_seq[t] = myds_metal_buffer_alloc(heads_per_tok * sizeof(float));
        low_seq[t] = myds_metal_buffer_alloc(low_per_tok * sizeof(float));
        CHECK(heads_seq[t] && low_seq[t]);
        fill_act(heads_seq[t], 77u + 88u * t);
        memcpy((float *)myds_metal_buffer_data(heads_b2) + t * heads_per_tok,
               myds_metal_buffer_data(heads_seq[t]), heads_per_tok * sizeof(float));
    }

    // 1. Sequential execution
    CHECK(myds_metal_begin_batch());
    for (uint32_t t = 0; t < 2; ++t) {
        CHECK(myds_metal_attn_output_a_bf16(low_seq[t], out_a, heads_seq[t], n_groups, group_dim, rank, GGML_TYPE_Q4_K));
    }
    CHECK(myds_metal_commit_batch());

    // 2. Batch-2 execution
    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_attn_output_a_bf16_batch(low_b2, out_a, heads_b2, n_groups, group_dim, rank, 2, GGML_TYPE_Q4_K));
    CHECK(myds_metal_commit_batch());

    // 3. Bit-exact comparison
    for (uint32_t t = 0; t < 2; ++t) {
        myds_metal_buffer_t *low_sub = myds_metal_buffer_sub_view(low_b2, t * low_per_tok * sizeof(float), low_per_tok * sizeof(float));
        CHECK(verify_exact(low_seq[t], low_sub, low_per_tok));
        myds_metal_buffer_free(low_sub);
        myds_metal_buffer_free(heads_seq[t]);
        myds_metal_buffer_free(low_seq[t]);
    }
    myds_metal_buffer_free(heads_b2);
    myds_metal_buffer_free(low_b2);
    myds_metal_buffer_free(out_a);

    // Also test Q8_0 shared2 Attention Output A
    const size_t out_a_bytes_q8 = (size_t)n_groups * rank * (group_dim / 32u) * 34u;
    myds_metal_buffer_t *out_a_q8 = myds_metal_buffer_alloc(out_a_bytes_q8);
    heads_b2 = myds_metal_buffer_alloc(2 * heads_per_tok * sizeof(float));
    low_b2 = myds_metal_buffer_alloc(2 * low_per_tok * sizeof(float));
    CHECK(out_a_q8 && heads_b2 && low_b2);
    fill_q8_weights(out_a_q8);

    for (uint32_t t = 0; t < 2; ++t) {
        heads_seq[t] = myds_metal_buffer_alloc(heads_per_tok * sizeof(float));
        low_seq[t] = myds_metal_buffer_alloc(low_per_tok * sizeof(float));
        CHECK(heads_seq[t] && low_seq[t]);
        fill_act(heads_seq[t], 123u + 456u * t);
        memcpy((float *)myds_metal_buffer_data(heads_b2) + t * heads_per_tok,
               myds_metal_buffer_data(heads_seq[t]), heads_per_tok * sizeof(float));
    }

    CHECK(myds_metal_begin_batch());
    for (uint32_t t = 0; t < 2; ++t) {
        CHECK(myds_metal_attn_output_a_bf16(low_seq[t], out_a_q8, heads_seq[t], n_groups, group_dim, rank, GGML_TYPE_Q8_0));
    }
    CHECK(myds_metal_commit_batch());

    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_attn_output_a_bf16_batch(low_b2, out_a_q8, heads_b2, n_groups, group_dim, rank, 2, GGML_TYPE_Q8_0));
    CHECK(myds_metal_commit_batch());

    for (uint32_t t = 0; t < 2; ++t) {
        myds_metal_buffer_t *low_sub = myds_metal_buffer_sub_view(low_b2, t * low_per_tok * sizeof(float), low_per_tok * sizeof(float));
        CHECK(verify_exact(low_seq[t], low_sub, low_per_tok));
        myds_metal_buffer_free(low_sub);
        myds_metal_buffer_free(heads_seq[t]);
        myds_metal_buffer_free(low_seq[t]);
    }
    myds_metal_buffer_free(heads_b2);
    myds_metal_buffer_free(low_b2);
    myds_metal_buffer_free(out_a_q8);

    printf("  [PASS] Batch-2 Attention Output A (Q4_K and Q8_0) 100%% bit-exact!\n");
    return 1;
}

static uint32_t outa_random(uint32_t value) {
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

/* Full production shape, canonical BF16 heads, non-power-of-two F16 block
 * scales, and explicit even/odd BF16 midpoint outputs. Compare the public
 * batch entry with two completed scalar CBs in all four dependency modes. */
static int test_batch2_outa_precision(void) {
    const uint32_t groups=8, width=4096, rank=1024;
    const size_t heads_count=(size_t)groups*width, low_count=(size_t)groups*rank;
    const char *shared_env=getenv("MYDS_ATTN_OUT_A_SHARED2");
    const int diagnostic=shared_env && !strcmp(shared_env,"1");
    size_t comparisons=0;
    for(unsigned format=0;format<2;++format){
        const ggml_type type=format?GGML_TYPE_Q8_0:GGML_TYPE_Q4_K;
        const uint32_t block_width=format?32u:256u, block_bytes=format?34u:144u;
        const size_t blocks=low_count*(width/block_width);
        struct guarded weight,heads,scalar,paired;
        CHECK(guarded_create(&weight,blocks*block_bytes));
        CHECK(guarded_create(&heads,2u*heads_count*4u));
        CHECK(guarded_create(&scalar,2u*low_count*4u));
        CHECK(guarded_create(&paired,2u*low_count*4u));
        myds_metal_buffer_t *head_rows[2],*low_rows[2];
        for(unsigned t=0;t<2;++t){
            head_rows[t]=myds_metal_buffer_sub_view(heads.view,t*heads_count*4u,heads_count*4u);
            low_rows[t]=myds_metal_buffer_sub_view(scalar.view,t*low_count*4u,low_count*4u);
            CHECK(head_rows[t]&&low_rows[t]);
        }
        for(unsigned sample=0;sample<3;++sample){
            uint8_t *w=myds_metal_buffer_data(weight.view);
            float *h=myds_metal_buffer_data(heads.view);
            if(!sample){
                memset(w,0,weight.bytes);memset(h,0,heads.bytes);
                for(size_t row=0;row<low_count;++row)for(unsigned b=0;b<2;++b){
                    uint8_t *p=w+(row*(width/block_width)+b)*block_bytes;
                    uint16_t scale=b?0x1c00u:0x3c00u; /* 2^-8 and 1 */
                    memcpy(p,&scale,2u);
                    if(format)p[2]=1u;
                    else {p[4]=1u;p[16]=1u;}
                }
                for(unsigned t=0;t<2;++t)for(unsigned g=0;g<groups;++g){
                    const size_t base=t*heads_count+g*width;
                    h[base]=(t?-1.0f:1.0f)*(g&1u?1.0078125f:1.0f);
                    h[base+block_width]=t?-1.0f:1.0f;
                }
            }else{
                for(size_t b=0;b<blocks;++b){
                    uint8_t *p=w+b*block_bytes;
                    uint32_t z=outa_random((uint32_t)b+sample*1709u);
                    uint16_t scale=(uint16_t)(0x1800u+(z&0x1fffu));
                    memcpy(p,&scale,2u);
                    if(!format){uint16_t minimum=(uint16_t)(0x1000u+((z>>8)&0x1fffu));memcpy(p+2,&minimum,2u);}
                    for(unsigned j=format?2u:4u;j<block_bytes;++j)p[j]=(uint8_t)outa_random(z+j*211u);
                }
                for(size_t i=0;i<2u*heads_count;++i){
                    uint32_t z=outa_random((uint32_t)i+sample*3907u);
                    uint32_t bits=0x3c000000u+(z&0x03ff0000u);
                    bits|=z&0x80000000u;memcpy(h+i,&bits,4u);
                }
            }
            uint8_t *saved_weight=malloc(weight.bytes),*saved_heads=malloc(heads.bytes);
            CHECK(saved_weight&&saved_heads);
            memcpy(saved_weight,w,weight.bytes);memcpy(saved_heads,h,heads.bytes);
            for(unsigned concurrent=0;concurrent<2;++concurrent)for(unsigned automatic=0;automatic<2;++automatic){
                myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(automatic);
                memset(myds_metal_buffer_data(scalar.view),0xa5,scalar.bytes);
                for(unsigned t=0;t<2;++t){
                    CHECK(myds_metal_begin_batch());
                    CHECK(myds_metal_attn_output_a_bf16(low_rows[t],weight.view,head_rows[t],groups,width,rank,type));
                    CHECK(myds_metal_commit_batch());
                }
                for(unsigned repeat=0;repeat<4;++repeat){
                    memset(myds_metal_buffer_data(paired.view),0xa5,paired.bytes);
                    CHECK(myds_metal_begin_batch());
                    CHECK(myds_metal_attn_output_a_bf16_batch(paired.view,weight.view,heads.view,groups,width,rank,2u,type));
                    CHECK(myds_metal_commit_batch());
                    if(!verify_exact(scalar.view,paired.view,2u*low_count)){
                        fprintf(stderr,"OutA precision failure: type=%s sample=%u concurrent=%u auto_barrier=%u repeat=%u shared2_diagnostic=%d\n",
                            format?"Q8_0":"Q4_K",sample,concurrent,automatic,repeat,diagnostic);
                        return 0;
                    }
                    if(!sample){
                        const float *p=myds_metal_buffer_data(paired.view);
                        for(unsigned t=0;t<2;++t)for(unsigned g=0;g<groups;++g)for(unsigned row=0;row<rank;++row)
                            CHECK(p[t*low_count+g*rank+row]==(t?-1.0f:1.0f)*(g&1u?1.015625f:1.0f));
                    }
                    CHECK(guarded_ok(&weight)&&guarded_ok(&heads)&&guarded_ok(&scalar)&&guarded_ok(&paired));
                    comparisons+=2u*low_count;
                }
                CHECK(!memcmp(saved_weight,w,weight.bytes)&&!memcmp(saved_heads,h,heads.bytes));
            }
            free(saved_weight);free(saved_heads);
        }
        for(unsigned t=0;t<2;++t){myds_metal_buffer_free(head_rows[t]);myds_metal_buffer_free(low_rows[t]);}
        guarded_free(&weight);guarded_free(&heads);guarded_free(&scalar);guarded_free(&paired);
    }
    myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
    printf("  [PASS] OutA full-shape precision: Q4_K/Q8_0, 4 modes, even/odd BF16 midpoint + 2 multiscale cases, 4 repeats, %zu byteexact comparisons (%s; synthetic fixture scope).\n",comparisons,diagnostic?"explicit shared2 diagnostic":"production scalar PSO default/explicit0");
    return 1;
}

static int test_batch2_shared_swiglu(void) {
    printf("[TEST] Batch-2 Shared Gate+Up SwiGLU (2304 x 5120 Q4_K)...\n");
    const uint32_t out_dim = 2304;
    const uint32_t in_dim = 5120;
    const size_t row_bytes = (in_dim / 256u) * 144u;
    const size_t w_bytes = (size_t)out_dim * row_bytes;

    myds_metal_buffer_t *gate = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *up = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *in_seq[2], *out_seq[2];
    myds_metal_buffer_t *in_b2 = myds_metal_buffer_alloc(2 * (size_t)in_dim * sizeof(float));
    myds_metal_buffer_t *out_b2 = myds_metal_buffer_alloc(2 * (size_t)out_dim * sizeof(float));
    CHECK(gate && up && in_b2 && out_b2);

    fill_q4_k_weights(gate);
    fill_q4_k_weights(up);

    for (uint32_t t = 0; t < 2; ++t) {
        in_seq[t] = myds_metal_buffer_alloc((size_t)in_dim * sizeof(float));
        out_seq[t] = myds_metal_buffer_alloc((size_t)out_dim * sizeof(float));
        CHECK(in_seq[t] && out_seq[t]);
        fill_act(in_seq[t], 31u + 41u * t);
        memcpy((float *)myds_metal_buffer_data(in_b2) + t * in_dim,
               myds_metal_buffer_data(in_seq[t]), in_dim * sizeof(float));
    }

    // 1. Sequential execution
    CHECK(myds_metal_begin_batch());
    for (uint32_t t = 0; t < 2; ++t) {
        CHECK(myds_v41_shared_gate_up_swiglu(out_seq[t], gate, up, in_seq[t], out_dim, in_dim, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 0.0f));
    }
    CHECK(myds_metal_commit_batch());

    // 2. Batch-2 execution
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_shared_gate_up_swiglu_batch2(out_b2, gate, up, in_b2, out_dim, in_dim, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 0.0f));
    CHECK(myds_metal_commit_batch());

    // 3. Bit-exact comparison
    for (uint32_t t = 0; t < 2; ++t) {
        myds_metal_buffer_t *out_sub = myds_metal_buffer_sub_view(out_b2, t * (size_t)out_dim * sizeof(float), out_dim * sizeof(float));
        CHECK(verify_exact(out_seq[t], out_sub, out_dim));
        myds_metal_buffer_free(out_sub);
        myds_metal_buffer_free(in_seq[t]);
        myds_metal_buffer_free(out_seq[t]);
    }
    myds_metal_buffer_free(in_b2);
    myds_metal_buffer_free(out_b2);
    myds_metal_buffer_free(gate);
    myds_metal_buffer_free(up);

    printf("  [PASS] Diagnostic shared2 GU fixture parity (clamp=0); current-checkpoint full-model parity remains unaccepted.\n");
    return 1;
}

static void iq2_fill(myds_metal_buffer_t *b, unsigned seed) {
    uint8_t *p = (uint8_t *)myds_metal_buffer_data(b);
    size_t bytes = myds_metal_buffer_size(b);
    for (size_t block = 0; block < bytes / 66; block++) {
        uint16_t d = (uint16_t)(0x1400u + (block % 4u) * 0x100u);
        memcpy(p + block * 66, &d, 2);
        for (size_t j = 0; j < 64; j++) p[block * 66 + 2 + j] = (uint8_t)(block * 37u + j * 29u + seed);
    }
}

static void q2_fill(myds_metal_buffer_t *b, unsigned seed) {
    uint8_t *p = (uint8_t *)myds_metal_buffer_data(b);
    size_t bytes = myds_metal_buffer_size(b);
    for (size_t block = 0; block < bytes / 84; block++) {
        for (size_t j = 0; j < 16; j++) p[block * 84 + j] = (uint8_t)(1u + (block * 13u + j * 17u + seed) % 255u);
        for (size_t j = 0; j < 64; j++) p[block * 84 + 16 + j] = (uint8_t)(block * 23u + j * 41u + seed);
        uint16_t d = (uint16_t)(0x1c00u + (block % 3u) * 0x80u), m = (uint16_t)(0x1000u + (block % 5u) * 0x80u);
        memcpy(p + block * 84 + 80, &d, 2);
        memcpy(p + block * 84 + 82, &m, 2);
    }
}

static int test_batch2_routed_moe(void) {
    printf("[TEST] Batch-2 Routed MoE (n_tok=2 vs n_tok=1x2, deduplicated IQ2 SwiGLU + Q2_K Down)...\n");
    const uint32_t dim = 5120;
    const uint32_t expert_dim = 2304;
    const uint32_t active = 6;
    const uint32_t experts = 14;

    const size_t gate_bytes = (size_t)experts * expert_dim * (dim / 256u) * 66u;
    const size_t down_bytes = (size_t)experts * dim * (expert_dim / 256u) * 84u;

    struct guarded gate, up, down;
    CHECK(guarded_create(&gate, gate_bytes));
    CHECK(guarded_create(&up, gate_bytes));
    CHECK(guarded_create(&down, down_bytes));

    iq2_fill(gate.view, 17);
    iq2_fill(up.view, 71);
    q2_fill(down.view, 29);

    struct guarded x_seq0, x_seq1, x_b2;
    CHECK(guarded_create(&x_seq0, dim * sizeof(float)));
    CHECK(guarded_create(&x_seq1, dim * sizeof(float)));
    CHECK(guarded_create(&x_b2, 2 * dim * sizeof(float)));
    fill_act(x_seq0.view, 43);
    fill_act(x_seq1.view, 97);
    memcpy((float *)myds_metal_buffer_data(x_b2.view), myds_metal_buffer_data(x_seq0.view), dim * sizeof(float));
    memcpy((float *)myds_metal_buffer_data(x_b2.view) + dim, myds_metal_buffer_data(x_seq1.view), dim * sizeof(float));

    /* Token 0: {11, 3, 0, 12, 5, 7}, Token 1: {3, 0, 12, 5, 7, 2} - 5 shared experts! */
    const int32_t ids0[6] = {11, 3, 0, 12, 5, 7};
    const int32_t ids1[6] = {3, 0, 12, 5, 7, 2};
    const float w0[6] = {0.3f, 0.25f, 0.2f, 0.15f, 0.07f, 0.03f};
    const float w1[6] = {0.28f, 0.24f, 0.18f, 0.16f, 0.08f, 0.06f};

    struct guarded sel_seq0, sel_seq1, sel_b2;
    struct guarded w_seq0, w_seq1, w_b2;
    CHECK(guarded_create(&sel_seq0, active * sizeof(int32_t)));
    CHECK(guarded_create(&sel_seq1, active * sizeof(int32_t)));
    CHECK(guarded_create(&sel_b2, 2 * active * sizeof(int32_t)));
    CHECK(guarded_create(&w_seq0, active * sizeof(float)));
    CHECK(guarded_create(&w_seq1, active * sizeof(float)));
    CHECK(guarded_create(&w_b2, 2 * active * sizeof(float)));

    memcpy(myds_metal_buffer_data(sel_seq0.view), ids0, active * sizeof(int32_t));
    memcpy(myds_metal_buffer_data(sel_seq1.view), ids1, active * sizeof(int32_t));
    memcpy(myds_metal_buffer_data(sel_b2.view), ids0, active * sizeof(int32_t));
    memcpy((int32_t *)myds_metal_buffer_data(sel_b2.view) + active, ids1, active * sizeof(int32_t));

    memcpy(myds_metal_buffer_data(w_seq0.view), w0, active * sizeof(float));
    memcpy(myds_metal_buffer_data(w_seq1.view), w1, active * sizeof(float));
    memcpy(myds_metal_buffer_data(w_b2.view), w0, active * sizeof(float));
    memcpy((float *)myds_metal_buffer_data(w_b2.view) + active, w1, active * sizeof(float));

    /* Scratch buffers */
    struct guarded sg_seq, su_seq, smid_seq, sdown_seq, out_seq0, out_seq1;
    CHECK(guarded_create(&sg_seq, active * expert_dim * sizeof(float)));
    CHECK(guarded_create(&su_seq, active * expert_dim * sizeof(float)));
    CHECK(guarded_create(&smid_seq, active * expert_dim * sizeof(float)));
    CHECK(guarded_create(&sdown_seq, active * dim * sizeof(float)));
    CHECK(guarded_create(&out_seq0, dim * sizeof(float)));
    CHECK(guarded_create(&out_seq1, dim * sizeof(float)));

    struct guarded sg_b2, su_b2, smid_b2, sdown_b2, out_b2;
    CHECK(guarded_create(&sg_b2, 2 * active * expert_dim * sizeof(float)));
    CHECK(guarded_create(&su_b2, 2 * active * expert_dim * sizeof(float)));
    CHECK(guarded_create(&smid_b2, 2 * active * expert_dim * sizeof(float)));
    CHECK(guarded_create(&sdown_b2, 2 * active * dim * sizeof(float)));
    CHECK(guarded_create(&out_b2, 2 * dim * sizeof(float)));

    /* 1. Sequential execution (n_tok = 1 for each token) */
    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_routed_moe_batch(out_seq0.view, x_seq0.view, gate.view, up.view, down.view,
        sel_seq0.view, w_seq0.view, sg_seq.view, su_seq.view, smid_seq.view, sdown_seq.view,
        dim, expert_dim, experts, active, 1, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));
    CHECK(myds_metal_commit_batch());

    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_routed_moe_batch(out_seq1.view, x_seq1.view, gate.view, up.view, down.view,
        sel_seq1.view, w_seq1.view, sg_seq.view, su_seq.view, smid_seq.view, sdown_seq.view,
        dim, expert_dim, experts, active, 1, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));
    CHECK(myds_metal_commit_batch());

    /* 2. Batch-2 execution (n_tok = 2, triggering deduplication kernels) */
    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_routed_moe_batch(out_b2.view, x_b2.view, gate.view, up.view, down.view,
        sel_b2.view, w_b2.view, sg_b2.view, su_b2.view, smid_b2.view, sdown_b2.view,
        dim, expert_dim, experts, active, 2, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));
    CHECK(myds_metal_commit_batch());

    /* 3. Verify 100% bit-exact parity between Sequential and Batch-2 */
    myds_metal_buffer_t *out_b2_tok0 = myds_metal_buffer_sub_view(out_b2.view, 0, dim * sizeof(float));
    myds_metal_buffer_t *out_b2_tok1 = myds_metal_buffer_sub_view(out_b2.view, dim * sizeof(float), dim * sizeof(float));
    CHECK(verify_exact(out_seq0.view, out_b2_tok0, dim));
    CHECK(verify_exact(out_seq1.view, out_b2_tok1, dim));
    myds_metal_buffer_free(out_b2_tok0);
    myds_metal_buffer_free(out_b2_tok1);

    /* 4. Benchmark Sequential vs Batch-2 */
    const int warmup = 5;
    const int iters = 50;
    for (int i = 0; i < warmup; i++) {
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_routed_moe_batch(out_b2.view, x_b2.view, gate.view, up.view, down.view,
            sel_b2.view, w_b2.view, sg_b2.view, su_b2.view, smid_b2.view, sdown_b2.view,
            dim, expert_dim, experts, active, 2, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));
        CHECK(myds_metal_commit_batch());
    }

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int i = 0; i < iters; i++) {
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_routed_moe_batch(out_b2.view, x_b2.view, gate.view, up.view, down.view,
            sel_b2.view, w_b2.view, sg_b2.view, su_b2.view, smid_b2.view, sdown_b2.view,
            dim, expert_dim, experts, active, 2, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));
        CHECK(myds_metal_commit_batch());
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double b2_us = ((ts1.tv_sec - ts0.tv_sec) * 1e6 + (ts1.tv_nsec - ts0.tv_nsec) / 1e3) / iters;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int i = 0; i < iters; i++) {
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_routed_moe_batch(out_seq0.view, x_seq0.view, gate.view, up.view, down.view,
            sel_seq0.view, w_seq0.view, sg_seq.view, su_seq.view, smid_seq.view, sdown_seq.view,
            dim, expert_dim, experts, active, 1, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));
        CHECK(myds_metal_routed_moe_batch(out_seq1.view, x_seq1.view, gate.view, up.view, down.view,
            sel_seq1.view, w_seq1.view, sg_seq.view, su_seq.view, smid_seq.view, sdown_seq.view,
            dim, expert_dim, experts, active, 1, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));
        CHECK(myds_metal_commit_batch());
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double seq_us = ((ts1.tv_sec - ts0.tv_sec) * 1e6 + (ts1.tv_nsec - ts0.tv_nsec) / 1e3) / iters;

    printf("  [BENCH] MoE Per-Layer: Sequential 2x = %.1f us (40L = %.2f ms), Batch-2 = %.1f us (40L = %.2f ms, %.1f%% of seq)\n",
           seq_us, seq_us * 40.0 / 1000.0, b2_us, b2_us * 40.0 / 1000.0, b2_us / seq_us * 100.0);

    guarded_free(&gate); guarded_free(&up); guarded_free(&down);
    guarded_free(&x_seq0); guarded_free(&x_seq1); guarded_free(&x_b2);
    guarded_free(&sel_seq0); guarded_free(&sel_seq1); guarded_free(&sel_b2);
    guarded_free(&w_seq0); guarded_free(&w_seq1); guarded_free(&w_b2);
    guarded_free(&sg_seq); guarded_free(&su_seq); guarded_free(&smid_seq); guarded_free(&sdown_seq);
    guarded_free(&out_seq0); guarded_free(&out_seq1);
    guarded_free(&sg_b2); guarded_free(&su_b2); guarded_free(&smid_b2); guarded_free(&sdown_b2);
    guarded_free(&out_b2);

    printf("  [PASS] Batch-2 Routed MoE 100%% bit-exact!\n");
    return 1;
}

static int test_batch2_hc_mix(void) {
    printf("[TEST] Batch-2 HC Mix (Attention & FFN)...\n");
    const uint32_t dim = 5120;
    const float eps = 1e-6f;
    const float hc_eps = 1e-4f;
    const size_t fn_bytes = 24u * 4u * dim * sizeof(uint16_t); // F16: 24 x 20480

    myds_metal_buffer_t *fn = myds_metal_buffer_alloc(fn_bytes);
    myds_metal_buffer_t *scale = myds_metal_buffer_alloc(3 * sizeof(float));
    myds_metal_buffer_t *base = myds_metal_buffer_alloc(24 * sizeof(float));
    CHECK(fn && scale && base);

    uint16_t *fn_p = (uint16_t *)myds_metal_buffer_data(fn);
    for (size_t i = 0; i < 24u * 4u * dim; i++) fn_p[i] = (uint16_t)(0x3400u + (i % 29));
    float *scale_p = (float *)myds_metal_buffer_data(scale);
    scale_p[0] = 0.5f; scale_p[1] = 0.25f; scale_p[2] = 0.125f;
    fill_act(base, 42);

    myds_metal_buffer_t *flat_norm_b2 = myds_metal_buffer_alloc(3 * 4 * dim * sizeof(float));
    myds_metal_buffer_t *mix_b2 = myds_metal_buffer_alloc(3 * 24 * sizeof(float));
    myds_metal_buffer_t *split_b2 = myds_metal_buffer_alloc(3 * 24 * sizeof(float));
    CHECK(flat_norm_b2 && mix_b2 && split_b2);

    myds_metal_buffer_t *residual[2];
    myds_metal_buffer_t *flat_norm_seq[2], *mix_seq[2], *split_seq[2];
    myds_metal_buffer_t *flat_norm_cand[2], *mix_cand[2], *split_cand[2];

    for (uint32_t t = 0; t < 2; ++t) {
        residual[t] = myds_metal_buffer_alloc(4 * dim * sizeof(float));
        flat_norm_seq[t] = myds_metal_buffer_alloc(4 * dim * sizeof(float));
        mix_seq[t] = myds_metal_buffer_alloc(24 * sizeof(float));
        split_seq[t] = myds_metal_buffer_alloc(24 * sizeof(float));
        CHECK(residual[t] && flat_norm_seq[t] && mix_seq[t] && split_seq[t]);
        fill_act(residual[t], 100u + 50u * t);

        flat_norm_cand[t] = myds_metal_buffer_sub_view(flat_norm_b2, t * 4 * dim * sizeof(float), 4 * dim * sizeof(float));
        mix_cand[t] = myds_metal_buffer_sub_view(mix_b2, t * 24 * sizeof(float), 24 * sizeof(float));
        split_cand[t] = myds_metal_buffer_sub_view(split_b2, t * 24 * sizeof(float), 24 * sizeof(float));
        CHECK(flat_norm_cand[t] && mix_cand[t] && split_cand[t]);
    }

    /* 1. Sequential reference */
    CHECK(myds_metal_begin_batch());
    for (uint32_t t = 0; t < 2; ++t) {
        CHECK(myds_metal_v41_hc_mix(split_seq[t], residual[t], fn, GGML_TYPE_F16, scale, base,
                                    flat_norm_seq[t], mix_seq[t], dim, eps, hc_eps));
    }
    CHECK(myds_metal_commit_batch());

    /* 2. Batched candidate */
    const myds_metal_buffer_t *res_const[2] = {residual[0], residual[1]};
    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_v41_hc_mix_batch2(split_b2, split_cand, res_const, fn, GGML_TYPE_F16, scale, base,
                                       flat_norm_cand, mix_b2, mix_cand, dim, eps, hc_eps));
    CHECK(myds_metal_commit_batch());

    /* 3. Bit-exact verification */
    for (uint32_t t = 0; t < 2; ++t) {
        CHECK(verify_exact(split_seq[t], split_cand[t], 24));
        myds_metal_buffer_free(split_cand[t]);
        myds_metal_buffer_free(mix_cand[t]);
        myds_metal_buffer_free(flat_norm_cand[t]);
        myds_metal_buffer_free(split_seq[t]);
        myds_metal_buffer_free(mix_seq[t]);
        myds_metal_buffer_free(flat_norm_seq[t]);
        myds_metal_buffer_free(residual[t]);
    }
    myds_metal_buffer_free(split_b2);
    myds_metal_buffer_free(mix_b2);
    myds_metal_buffer_free(flat_norm_b2);
    myds_metal_buffer_free(base);
    myds_metal_buffer_free(scale);
    myds_metal_buffer_free(fn);
    printf("  [PASS] Batch-2 HC Mix 100%% bit-exact!\n");
    return 1;
}

int main(void) {
    printf("\n=======================================================\n");
    printf("   Phase 3: Metal Per-Layer Batch-2 Parity Test Suite  \n");
    printf("=======================================================\n");

    /* Setter modes below must select the actual encoder; clear the ambient
     * concurrent override before the runtime caches it on its first CB. */
    unsetenv("MYDS_METAL_CONCURRENT");
    if (!myds_metal_init(NULL)) {
        fprintf(stderr, "Failed to initialize Metal\n");
        return 1;
    }

    // 1. Attention Q_A projection (1280 x 5120)
    if (!test_batch2_gemv_projections(1280, 5120)) return 1;

    // 2. Attention KV_A projection (512 x 5120)
    if (!test_batch2_gemv_projections(512, 5120)) return 1;

    // 3. Attention Q_B projection (32768 x 1280)
    if (!test_batch2_gemv_projections(32768, 1280)) return 1;

    // 4. Attention Output B projection (5120 x 8192)
    if (!test_batch2_gemv_projections(5120, 8192)) return 1;

    // 5. MoE Shared Expert Gate/Up (2304 x 5120)
    if (!test_batch2_gemv_projections(2304, 5120)) return 1;

    // 6. MoE Shared Expert Down (5120 x 2304)
    if (!test_batch2_gemv_projections(5120, 2304)) return 1;

    // 7. MoE Router Gate (384 x 5120)
    if (!test_batch2_gemv_projections(384, 5120)) return 1;

    // 8. MoE Fused Router (rows = 2)
    if (!test_batch2_moe_route()) return 1;
    if (!test_authoritative_projection(1280, 5120, GGML_TYPE_Q8_0, MYDS_V41_BATCH2_QKV)) return 1;
    if (!test_authoritative_projection(1280, 5120, GGML_TYPE_Q4_K, MYDS_V41_BATCH2_QKV)) return 1;
    if (!test_authoritative_projection(512, 5120, GGML_TYPE_Q4_K, MYDS_V41_BATCH2_QKV)) return 1;
    if (!test_authoritative_projection(32768, 1280, GGML_TYPE_Q4_K, MYDS_V41_BATCH2_QKV)) return 1;
    if (!test_authoritative_projection(5120, 8192, GGML_TYPE_Q8_0, MYDS_V41_BATCH2_TAIL)) return 1;
    if (!test_authoritative_projection(5120, 8192, GGML_TYPE_Q4_K, MYDS_V41_BATCH2_TAIL)) return 1;
    if (!test_authoritative_projection(5120, 2304, GGML_TYPE_Q8_0, MYDS_V41_BATCH2_SHARED)) return 1;
    if (!test_authoritative_projection(5120, 2304, GGML_TYPE_Q4_K, MYDS_V41_BATCH2_SHARED)) return 1;
    if (!test_authoritative_projection(384, 5120, GGML_TYPE_F16, MYDS_V41_BATCH2_ROUTER)) return 1;
    if (!test_authoritative_projection(384, 5120, GGML_TYPE_F32, MYDS_V41_BATCH2_ROUTER)) return 1;
    if (!test_authoritative_projection(1024, 5120, GGML_TYPE_Q8_0, MYDS_V41_BATCH2_LM_HEAD)) return 1;
    if (!test_authoritative_projection(129280, 5120, GGML_TYPE_Q8_0, MYDS_V41_BATCH2_LM_HEAD)) return 1;
    if (!test_authoritative_projection(1024, 5120, GGML_TYPE_Q4_K, MYDS_V41_BATCH2_LM_HEAD)) return 1;
    if (!test_authoritative_projection(129280, 5120, GGML_TYPE_Q4_K, MYDS_V41_BATCH2_LM_HEAD)) return 1;

    // 9. Batch-2 Attention Output A
    if (!test_batch2_attn_output_a()) return 1;
    if (!test_batch2_outa_precision()) return 1;

    // 10. Batch-2 HC Mix
    if (!test_batch2_hc_mix()) return 1;

    // 11. Batch-2 Shared Gate+Up SwiGLU
    if (!test_batch2_shared_swiglu()) return 1;

    // 12. Batch-2 Routed MoE (Deduplicated SwiGLU + Down)
    if (!test_batch2_routed_moe()) return 1;

    printf("\n=== ALL PHASE 3 BATCH-2 PARITY TESTS PASSED CLEANLY! ===\n\n");
    return 0;
}
