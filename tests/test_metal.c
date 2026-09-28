#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        return 0; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    float diff = fabsf((float)(a) - (float)(b)); \
    if (diff > (tol)) { \
        fprintf(stderr, "[FAIL] %s:%d: %s (%.6f) != %s (%.6f), diff=%.6f > tol=%.6f\n", \
                __FILE__, __LINE__, #a, (float)(a), #b, (float)(b), diff, (float)(tol)); \
        return 0; \
    } \
} while(0)

static int test_rmsnorm(void) {
    const uint32_t dim = 2048;
    const float eps = 1e-6f;

    myds_metal_buffer_t *x = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *w = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(dim * sizeof(float));
    ASSERT_TRUE(x && w && out);

    float *xv = (float *)myds_metal_buffer_data(x);
    float *wv = (float *)myds_metal_buffer_data(w);
    float *ref = (float *)malloc(dim * sizeof(float));

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        xv[i] = ((float)(i % 17) - 8.0f) * 0.125f;
        wv[i] = 1.0f + 0.01f * (float)(i % 13);
        sum_sq += xv[i] * xv[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    for (uint32_t i = 0; i < dim; i++) {
        ref[i] = xv[i] * scale * wv[i];
    }

    ASSERT_TRUE(myds_metal_rmsnorm(out, x, w, dim, eps));

    float *outv = (float *)myds_metal_buffer_data(out);
    for (uint32_t i = 0; i < dim; i++) {
        ASSERT_NEAR(outv[i], ref[i], 1e-4f);
    }

    free(ref);
    myds_metal_buffer_free(x);
    myds_metal_buffer_free(w);
    myds_metal_buffer_free(out);
    printf("  [PASS] RMSNorm test (%u dims)\n", dim);
    return 1;
}

static int test_swiglu(void) {
    const uint32_t dim = 1024;
    myds_metal_buffer_t *gate = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *up = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(dim * sizeof(float));
    ASSERT_TRUE(gate && up && out);

    float *gv = (float *)myds_metal_buffer_data(gate);
    float *uv = (float *)myds_metal_buffer_data(up);
    float *ref = (float *)malloc(dim * sizeof(float));

    for (uint32_t i = 0; i < dim; i++) {
        gv[i] = ((float)(i % 31) - 15.0f) * 0.2f;
        uv[i] = ((float)(i % 23) - 11.0f) * 0.3f;
        float silu = gv[i] / (1.0f + expf(-gv[i]));
        ref[i] = silu * uv[i];
    }

    ASSERT_TRUE(myds_metal_swiglu(out, gate, up, dim));

    float *outv = (float *)myds_metal_buffer_data(out);
    for (uint32_t i = 0; i < dim; i++) {
        ASSERT_NEAR(outv[i], ref[i], 1e-4f);
    }

    free(ref);
    myds_metal_buffer_free(gate);
    myds_metal_buffer_free(up);
    myds_metal_buffer_free(out);
    printf("  [PASS] SwiGLU activation test (%u dims)\n", dim);
    return 1;
}

static int test_rope(void) {
    const uint32_t n_heads = 16;
    const uint32_t rope_dim = 64;
    const uint32_t pos = 42;

    myds_metal_buffer_t *q = myds_metal_buffer_alloc(n_heads * rope_dim * sizeof(float));
    myds_metal_buffer_t *k = myds_metal_buffer_alloc(rope_dim * sizeof(float));
    ASSERT_TRUE(q && k);

    float *qv = (float *)myds_metal_buffer_data(q);
    float *kv = (float *)myds_metal_buffer_data(k);
    for (uint32_t i = 0; i < n_heads * rope_dim; i++) qv[i] = 1.0f;
    for (uint32_t i = 0; i < rope_dim; i++) kv[i] = 1.0f;

    ASSERT_TRUE(myds_metal_rope(q, k, n_heads, rope_dim, pos, 10000.0f, 1.0f));

    /* Verify magnitude is conserved after rotation: q0^2 + q1^2 == 1^2 + 1^2 == 2 */
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < rope_dim; i += 2) {
            float q0 = qv[h * rope_dim + i];
            float q1 = qv[h * rope_dim + i + 1];
            float mag = q0 * q0 + q1 * q1;
            ASSERT_NEAR(mag, 2.0f, 1e-4f);
        }
    }

    myds_metal_buffer_free(q);
    myds_metal_buffer_free(k);
    printf("  [PASS] Decoupled RoPE test (n_heads=%u, rope_dim=%u, pos=%u)\n", n_heads, rope_dim, pos);
    return 1;
}

static int test_moe_routing(void) {
    const uint32_t n_routed = 64;
    const uint32_t top_k = 6;
    const float scaling = 1.5f;

    myds_metal_buffer_t *logits = myds_metal_buffer_alloc(n_routed * sizeof(float));
    ASSERT_TRUE(logits);

    float *lv = (float *)myds_metal_buffer_data(logits);
    for (uint32_t i = 0; i < n_routed; i++) {
        lv[i] = -2.0f + 0.1f * (float)i;
    }

    int32_t expert_ids[6];
    float expert_weights[6];

    ASSERT_TRUE(myds_metal_moe_route(expert_ids, expert_weights, logits, n_routed, top_k, scaling));

    for (uint32_t k = 0; k < top_k; k++) {
        ASSERT_TRUE(expert_ids[k] == (int32_t)(63 - k));
    }

    float sum_w = 0.0f;
    for (uint32_t k = 0; k < top_k; k++) {
        sum_w += expert_weights[k];
    }
    ASSERT_NEAR(sum_w, scaling, 1e-4f);

    myds_metal_buffer_free(logits);
    printf("  [PASS] DeepSeek-MoE router test (%u experts, top-%u, scale=%.1f)\n", n_routed, top_k, scaling);
    return 1;
}

static int test_gemv_f32(void) {
    const uint32_t out_dim = 256;
    const uint32_t in_dim = 512;

    myds_metal_buffer_t *W = myds_metal_buffer_alloc(out_dim * in_dim * sizeof(float));
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(in_dim * sizeof(float));
    myds_metal_buffer_t *y = myds_metal_buffer_alloc(out_dim * sizeof(float));
    ASSERT_TRUE(W && x && y);

    float *wv = (float *)myds_metal_buffer_data(W);
    float *xv = (float *)myds_metal_buffer_data(x);
    for (uint32_t i = 0; i < in_dim; i++) xv[i] = 1.0f;
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t c = 0; c < in_dim; c++) {
            wv[r * in_dim + c] = 0.01f * (float)(r % 7);
        }
    }

    ASSERT_TRUE(myds_metal_gemv(y, W, x, out_dim, in_dim, GGML_TYPE_F32));

    float *yv = (float *)myds_metal_buffer_data(y);
    for (uint32_t r = 0; r < out_dim; r++) {
        float expected = 0.01f * (float)(r % 7) * (float)in_dim;
        ASSERT_NEAR(yv[r], expected, 1e-3f);
    }

    myds_metal_buffer_free(W);
    myds_metal_buffer_free(x);
    myds_metal_buffer_free(y);
    printf("  [PASS] GEMV F32 test (%ux%u)\n", out_dim, in_dim);
    return 1;
}

static int test_gemv_q8_0(void) {
    const uint32_t out_dim = 64;
    const uint32_t in_dim = 256; /* 8 blocks of 32 */

    typedef struct {
        uint16_t d; /* half scale */
        int8_t   qs[32];
    } block_q8_0_t;

    size_t q8_bytes = (size_t)out_dim * (in_dim / 32) * sizeof(block_q8_0_t);
    myds_metal_buffer_t *W = myds_metal_buffer_alloc(q8_bytes);
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(in_dim * sizeof(float));
    myds_metal_buffer_t *y = myds_metal_buffer_alloc(out_dim * sizeof(float));
    ASSERT_TRUE(W && x && y);

    block_q8_0_t *blocks = (block_q8_0_t *)myds_metal_buffer_data(W);
    float *xv = (float *)myds_metal_buffer_data(x);
    for (uint32_t i = 0; i < in_dim; i++) xv[i] = 1.0f;

    /* Fill Q8 blocks with scale = 0.5f and qs = 2 => value = 1.0f */
    /* Half float 0.5f in binary is 0x3800 */
    uint16_t half_half = 0x3800;
    uint32_t n_blocks = out_dim * (in_dim / 32);
    for (uint32_t b = 0; b < n_blocks; b++) {
        blocks[b].d = half_half;
        for (int i = 0; i < 32; i++) {
            blocks[b].qs[i] = 2;
        }
    }

    ASSERT_TRUE(myds_metal_gemv(y, W, x, out_dim, in_dim, GGML_TYPE_Q8_0));

    float *yv = (float *)myds_metal_buffer_data(y);
    for (uint32_t r = 0; r < out_dim; r++) {
        /* Dot product of row: in_dim elements of 1.0 * (0.5 * 2) = in_dim */
        ASSERT_NEAR(yv[r], (float)in_dim, 1.0f);
    }

    myds_metal_buffer_free(W);
    myds_metal_buffer_free(x);
    myds_metal_buffer_free(y);
    printf("  [PASS] GEMV Q8_0 quantization dot-product test (%ux%u)\n", out_dim, in_dim);
    return 1;
}

static int test_mul_mm_q8_0(void) {
    const uint32_t out_dim = 64;
    const uint32_t in_dim = 256;
    const uint32_t n_tok = 4;

    typedef struct {
        uint16_t d;
        int8_t   qs[32];
    } block_q8_0_t;

    size_t q8_bytes = (size_t)out_dim * (in_dim / 32) * sizeof(block_q8_0_t);
    myds_metal_buffer_t *W = myds_metal_buffer_alloc(q8_bytes);
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(n_tok * in_dim * sizeof(float));
    myds_metal_buffer_t *y = myds_metal_buffer_alloc(n_tok * out_dim * sizeof(float));
    ASSERT_TRUE(W && x && y);

    block_q8_0_t *blocks = (block_q8_0_t *)myds_metal_buffer_data(W);
    float *xv = (float *)myds_metal_buffer_data(x);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            xv[t * in_dim + i] = sinf((float)(t * in_dim + i + 1));
        }
    }

    uint16_t half_half = 0x3800; // 0.5f
    uint32_t n_blocks = out_dim * (in_dim / 32);
    for (uint32_t b = 0; b < n_blocks; b++) {
        blocks[b].d = half_half;
        for (int i = 0; i < 32; i++) {
            blocks[b].qs[i] = (int8_t)((b * 7 + i) % 15 - 7);
        }
    }

    /* Compute baseline per-token using myds_metal_gemv */
    myds_metal_buffer_t *y_gemv = myds_metal_buffer_alloc(n_tok * out_dim * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++) {
        myds_metal_buffer_t *xt = myds_metal_buffer_sub_view(x, t * in_dim * sizeof(float), in_dim * sizeof(float));
        myds_metal_buffer_t *yt = myds_metal_buffer_sub_view(y_gemv, t * out_dim * sizeof(float), out_dim * sizeof(float));
        ASSERT_TRUE(myds_metal_gemv(yt, W, xt, out_dim, in_dim, GGML_TYPE_Q8_0));
        myds_metal_buffer_free(xt);
        myds_metal_buffer_free(yt);
    }

    ASSERT_TRUE(myds_metal_mul_mm(y, W, x, out_dim, in_dim, n_tok, GGML_TYPE_Q8_0));

    float *yv = (float *)myds_metal_buffer_data(y);
    float *y_expected = (float *)myds_metal_buffer_data(y_gemv);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            ASSERT_NEAR(yv[t * out_dim + r], y_expected[t * out_dim + r], 1e-2f);
        }
    }

    myds_metal_buffer_free(y_gemv);
    myds_metal_buffer_free(x);
    myds_metal_buffer_free(y);

    /* Also test unaligned token count across tile boundaries: n_tok = 35 */
    const uint32_t n_tok_odd = 35;
    x = myds_metal_buffer_alloc(n_tok_odd * in_dim * sizeof(float));
    y = myds_metal_buffer_alloc(n_tok_odd * out_dim * sizeof(float));
    y_gemv = myds_metal_buffer_alloc(n_tok_odd * out_dim * sizeof(float));
    ASSERT_TRUE(x && y && y_gemv);

    xv = (float *)myds_metal_buffer_data(x);
    for (uint32_t t = 0; t < n_tok_odd; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            xv[t * in_dim + i] = sinf((float)(t * in_dim + i + 1));
        }
    }

    for (uint32_t t = 0; t < n_tok_odd; t++) {
        myds_metal_buffer_t *xt = myds_metal_buffer_sub_view(x, t * in_dim * sizeof(float), in_dim * sizeof(float));
        myds_metal_buffer_t *yt = myds_metal_buffer_sub_view(y_gemv, t * out_dim * sizeof(float), out_dim * sizeof(float));
        ASSERT_TRUE(myds_metal_gemv(yt, W, xt, out_dim, in_dim, GGML_TYPE_Q8_0));
        myds_metal_buffer_free(xt);
        myds_metal_buffer_free(yt);
    }

    ASSERT_TRUE(myds_metal_mul_mm(y, W, x, out_dim, in_dim, n_tok_odd, GGML_TYPE_Q8_0));

    yv = (float *)myds_metal_buffer_data(y);
    y_expected = (float *)myds_metal_buffer_data(y_gemv);
    for (uint32_t t = 0; t < n_tok_odd; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            ASSERT_NEAR(yv[t * out_dim + r], y_expected[t * out_dim + r], 2e-2f);
        }
    }
    myds_metal_buffer_free(y_gemv);

    myds_metal_buffer_free(W);
    myds_metal_buffer_free(x);
    myds_metal_buffer_free(y);
    printf("  [PASS] Batched Matrix-Matrix Q8_0 test (out_dim=%u, in_dim=%u, n_tok=%u and %u)\n", out_dim, in_dim, n_tok, n_tok_odd);
    return 1;
}

static int test_moe_sum(void) {
    const uint32_t n_activated = 4;
    const uint32_t dim = 512;

    myds_metal_buffer_t *expert_outs = myds_metal_buffer_alloc(n_activated * dim * sizeof(float));
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(dim * sizeof(float));
    ASSERT_TRUE(expert_outs && out);

    float *eo = (float *)myds_metal_buffer_data(expert_outs);
    for (uint32_t k = 0; k < n_activated; k++) {
        for (uint32_t i = 0; i < dim; i++) {
            eo[k * dim + i] = (float)(k + 1);
        }
    }

    float weights[4] = { 0.4f, 0.3f, 0.2f, 0.1f };
    /* Expected sum: 0.4*1 + 0.3*2 + 0.2*3 + 0.1*4 = 0.4 + 0.6 + 0.6 + 0.4 = 2.0 */

    ASSERT_TRUE(myds_metal_moe_sum(out, expert_outs, weights, n_activated, dim));

    float *ov = (float *)myds_metal_buffer_data(out);
    for (uint32_t i = 0; i < dim; i++) {
        ASSERT_NEAR(ov[i], 2.0f, 1e-4f);
    }

    myds_metal_buffer_free(expert_outs);
    myds_metal_buffer_free(out);
    printf("  [PASS] DeepSeek-MoE expert output accumulation test (top-%u, dim=%u)\n", n_activated, dim);
    return 1;
}

static int test_add(void) {
    const uint32_t count = 4096;
    myds_metal_buffer_t *a = myds_metal_buffer_alloc(count * sizeof(float));
    myds_metal_buffer_t *b = myds_metal_buffer_alloc(count * sizeof(float));
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(count * sizeof(float));
    ASSERT_TRUE(a && b && out);

    float *av = (float *)myds_metal_buffer_data(a);
    float *bv = (float *)myds_metal_buffer_data(b);
    for (uint32_t i = 0; i < count; i++) {
        av[i] = 1.5f * (float)i;
        bv[i] = 2.5f * (float)i;
    }

    ASSERT_TRUE(myds_metal_add(out, a, b, count));

    float *ov = (float *)myds_metal_buffer_data(out);
    for (uint32_t i = 0; i < count; i++) {
        ASSERT_NEAR(ov[i], 4.0f * (float)i, 1e-4f);
    }

    myds_metal_buffer_free(a);
    myds_metal_buffer_free(b);
    myds_metal_buffer_free(out);
    printf("  [PASS] Element-wise residual addition test (%u elements)\n", count);
    return 1;
}

static int test_softmax(void) {
    const uint32_t count = 128;
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(count * sizeof(float));
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(count * sizeof(float));
    ASSERT_TRUE(x && out);

    float *xv = (float *)myds_metal_buffer_data(x);
    for (uint32_t i = 0; i < count; i++) {
        xv[i] = (float)i * 0.1f;
    }

    ASSERT_TRUE(myds_metal_softmax(out, x, count));

    float *ov = (float *)myds_metal_buffer_data(out);
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        ASSERT_TRUE(ov[i] >= 0.0f);
        sum += ov[i];
        if (i > 0) {
            /* Monotonically increasing */
            ASSERT_TRUE(ov[i] >= ov[i - 1]);
        }
    }
    ASSERT_NEAR(sum, 1.0f, 1e-4f);

    myds_metal_buffer_free(x);
    myds_metal_buffer_free(out);
    printf("  [PASS] Softmax normalization and monotonicity test (%u elements)\n", count);
    return 1;
}

static int test_buffer_read_write(void) {
    size_t size = 1024;
    myds_metal_buffer_t *buf = myds_metal_buffer_alloc(size);
    ASSERT_TRUE(buf != NULL);
    ASSERT_TRUE(myds_metal_buffer_size(buf) == size);

    uint8_t payload[256];
    for (int i = 0; i < 256; i++) payload[i] = (uint8_t)(i ^ 0x5A);

    ASSERT_TRUE(myds_metal_buffer_write(buf, 128, payload, 256));

    uint8_t readback[256];
    ASSERT_TRUE(myds_metal_buffer_read(buf, 128, readback, 256));
    ASSERT_TRUE(memcmp(payload, readback, 256) == 0);

    /* Out of bounds should fail gracefully */
    ASSERT_TRUE(myds_metal_buffer_write(buf, size - 10, payload, 256) == 0);
    ASSERT_TRUE(myds_metal_buffer_read(buf, size - 10, readback, 256) == 0);

    myds_metal_buffer_free(buf);
    printf("  [PASS] Metal unified buffer allocation and read/write boundary test\n");
    return 1;
}

static int test_buffer_from_ptr_zerocopy(void) {
    size_t page_sz = (size_t)getpagesize();
    size_t alloc_bytes = page_sz * 4;
    void *raw_mem = NULL;
    int rc = posix_memalign(&raw_mem, page_sz, alloc_bytes);
    ASSERT_TRUE(rc == 0 && raw_mem != NULL);

    /* Fill memory with recognizable pattern */
    uint8_t *byte_ptr = (uint8_t *)raw_mem;
    for (size_t i = 0; i < alloc_bytes; i++) {
        byte_ptr[i] = (uint8_t)((i * 31) ^ 0xA5);
    }

    /* Wrap aligned buffer */
    myds_metal_buffer_t *buf_aligned = myds_metal_buffer_from_ptr(raw_mem, alloc_bytes);
    ASSERT_TRUE(buf_aligned != NULL);
    ASSERT_TRUE(myds_metal_buffer_is_view(buf_aligned) == true);
    ASSERT_TRUE(myds_metal_buffer_offset(buf_aligned) == 0);
    ASSERT_TRUE(myds_metal_buffer_data(buf_aligned) == raw_mem);

    /* Wrap unaligned slice inside the memory */
    size_t sub_offset = 128;
    size_t sub_bytes = 1024;
    void *unaligned_ptr = (void *)(byte_ptr + sub_offset);
    myds_metal_buffer_t *buf_unaligned = myds_metal_buffer_from_ptr(unaligned_ptr, sub_bytes);
    ASSERT_TRUE(buf_unaligned != NULL);
    ASSERT_TRUE(myds_metal_buffer_is_view(buf_unaligned) == true);
    ASSERT_TRUE(myds_metal_buffer_offset(buf_unaligned) == sub_offset);
    ASSERT_TRUE(myds_metal_buffer_data(buf_unaligned) == unaligned_ptr);
    ASSERT_TRUE(myds_metal_buffer_size(buf_unaligned) == sub_bytes);

    /* Verify data matches directly without copying */
    uint8_t *view_data = (uint8_t *)myds_metal_buffer_data(buf_unaligned);
    ASSERT_TRUE(memcmp(view_data, unaligned_ptr, sub_bytes) == 0);

    /* Modify source memory and ensure zero-copy view reflects change immediately */
    byte_ptr[sub_offset] ^= 0xFF;
    ASSERT_TRUE(view_data[0] == byte_ptr[sub_offset]);

    myds_metal_buffer_free(buf_aligned);
    myds_metal_buffer_free(buf_unaligned);
    free(raw_mem);

    printf("  [PASS] Metal zero-copy buffer wrapping (newBufferWithBytesNoCopy) and slice offsets\n");
    return 1;
}

static int test_gemv_unsupported_quant_failfast(void) {
    const uint32_t out_dim = 32;
    const uint32_t in_dim = 64;

    myds_metal_buffer_t *x = myds_metal_buffer_alloc(in_dim * sizeof(float));
    myds_metal_buffer_t *y = myds_metal_buffer_alloc(out_dim * sizeof(float));
    myds_metal_buffer_t *W = myds_metal_buffer_alloc(out_dim * in_dim);
    ASSERT_TRUE(x && y && W);

    /* GGML_TYPE_Q4_K (12), GGML_TYPE_Q6_K (14), GGML_TYPE_IQ4_NL (20) must fail fast and return 0 */
    int rc_q4 = myds_metal_gemv(y, W, x, out_dim, in_dim, GGML_TYPE_Q4_K);
    ASSERT_TRUE(rc_q4 == 0);

    int rc_q6 = myds_metal_gemv(y, W, x, out_dim, in_dim, GGML_TYPE_Q6_K);
    ASSERT_TRUE(rc_q6 == 0);

    int rc_iq4 = myds_metal_gemv(y, W, x, out_dim, in_dim, GGML_TYPE_IQ4_NL);
    ASSERT_TRUE(rc_iq4 == 0);

    myds_metal_buffer_free(x);
    myds_metal_buffer_free(y);
    myds_metal_buffer_free(W);

    printf("  [PASS] myds_metal_gemv fail-fast rejection for unsupported quantization types (Q4_K, Q6_K, IQ4_NL)\n");
    return 1;
}

static int test_batch_command_buffer(void) {
    const uint32_t dim = 256;
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *w = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *norm_out = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *W_gemv = myds_metal_buffer_alloc(dim * dim * sizeof(float));
    myds_metal_buffer_t *gemv_out = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *res_out = myds_metal_buffer_alloc(dim * sizeof(float));
    ASSERT_TRUE(x && w && norm_out && W_gemv && gemv_out && res_out);

    float *xv = (float *)myds_metal_buffer_data(x);
    float *wv = (float *)myds_metal_buffer_data(w);
    float *Wv = (float *)myds_metal_buffer_data(W_gemv);
    for (uint32_t i = 0; i < dim; i++) {
        xv[i] = 1.0f;
        wv[i] = 1.0f;
        for (uint32_t c = 0; c < dim; c++) {
            Wv[i * dim + c] = 0.005f;
        }
    }

    uint64_t c0 = myds_metal_get_commit_count();
    ASSERT_TRUE(myds_metal_begin_batch());

    /* Dispatch 3 chained operations: RMSNorm -> GEMV -> Add */
    ASSERT_TRUE(myds_metal_rmsnorm(norm_out, x, w, dim, 1e-5f));
    ASSERT_TRUE(myds_metal_gemv(gemv_out, W_gemv, norm_out, dim, dim, GGML_TYPE_F32));
    ASSERT_TRUE(myds_metal_add(res_out, norm_out, gemv_out, dim));

    /* Inside batch, commit count MUST not increase */
    ASSERT_TRUE(myds_metal_get_commit_count() == c0);

    /* Single commit for the entire batch */
    ASSERT_TRUE(myds_metal_commit_batch());
    ASSERT_TRUE(myds_metal_get_commit_count() == c0 + 1);

    /* Verify math:
     * x is all 1.0f, dim=256 -> mean(x^2)=1.0f -> norm_out = all 1.0f.
     * W_gemv is all 0.005f -> gemv_out = 256 * 0.005f = 1.28f.
     * res_out = 1.0f + 1.28f = 2.28f.
     */
    float *rv = (float *)myds_metal_buffer_data(res_out);
    for (uint32_t i = 0; i < dim; i++) {
        ASSERT_NEAR(rv[i], 2.28f, 1e-3f);
    }

    myds_metal_buffer_free(x);
    myds_metal_buffer_free(w);
    myds_metal_buffer_free(norm_out);
    myds_metal_buffer_free(W_gemv);
    myds_metal_buffer_free(gemv_out);
    myds_metal_buffer_free(res_out);

    printf("  [PASS] Batch command buffer: multi-operator chaining with single commit (T1-1)\n");
    return 1;
}

int main(void) {
    printf("====================================================\n");
    printf("   Running myds Metal GPU Kernel Verification\n");
    printf("====================================================\n");

    if (!myds_metal_init("metal")) {
        fprintf(stderr, "Failed to initialize myds_metal\n");
        return 1;
    }

    int passed = 0;
    passed += test_rmsnorm();
    passed += test_swiglu();
    passed += test_rope();
    passed += test_moe_routing();
    passed += test_gemv_f32();
    passed += test_gemv_q8_0();
    passed += test_mul_mm_q8_0();
    passed += test_moe_sum();
    passed += test_add();
    passed += test_softmax();
    passed += test_buffer_read_write();
    passed += test_buffer_from_ptr_zerocopy();
    passed += test_gemv_unsupported_quant_failfast();
    passed += test_batch_command_buffer();

    myds_metal_cleanup();

    printf("====================================================\n");
    if (passed == 14) {
        printf("  ALL 14 METAL OPERATOR & BATCH TESTS PASSED! (Apple Silicon GPU)\n");
        printf("====================================================\n");
        return 0;
    } else {
        printf("  TESTS FAILED (passed %d/14)\n", passed);
        return 1;
    }
}
