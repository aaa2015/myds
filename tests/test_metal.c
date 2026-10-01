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
    if (!isfinite((float)(a)) || !isfinite((float)(b)) || diff > (tol)) { \
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

static int test_swiglu_flat_shapes(void) {
    const uint32_t shapes[] = { 1, 31, 257, 1025, 32769 };
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        uint32_t dim = shapes[shape];
        size_t bytes = (size_t)dim * sizeof(float);
        myds_metal_buffer_t *gate = myds_metal_buffer_alloc(bytes);
        myds_metal_buffer_t *up = myds_metal_buffer_alloc(bytes);
        myds_metal_buffer_t *storage = myds_metal_buffer_alloc(bytes + 2 * sizeof(float));
        ASSERT_TRUE(gate && up && storage);
        myds_metal_buffer_t *out = myds_metal_buffer_sub_view(storage, sizeof(float), bytes);
        ASSERT_TRUE(out);
        float *gv = myds_metal_buffer_data(gate);
        float *uv = myds_metal_buffer_data(up);
        float *sv = myds_metal_buffer_data(storage);
        float *ov = myds_metal_buffer_data(out);
        float *ref = malloc(bytes);
        ASSERT_TRUE(ref);
        sv[0] = sv[dim + 1] = -9876.0f;
        for (uint32_t i = 0; i < dim; i++) {
            gv[i] = ((int)(i % 101) - 50) * 0.125f;
            uv[i] = ((int)(i % 37) - 18) * 0.0625f;
            ref[i] = gv[i] / (1.0f + expf(-gv[i])) * uv[i];
        }
        ASSERT_TRUE(!myds_metal_swiglu(out, gate, up, dim + 1));
        uint64_t commits = myds_metal_get_commit_count();
        ASSERT_TRUE(myds_metal_begin_batch());
        ASSERT_TRUE(myds_metal_swiglu(out, gate, up, dim));
        ASSERT_TRUE(myds_metal_add(out, out, up, dim));
        ASSERT_TRUE(myds_metal_commit_batch());
        ASSERT_TRUE(myds_metal_get_commit_count() == commits + 1);
        for (uint32_t i = 0; i < dim; i++) ASSERT_NEAR(ov[i], ref[i] + uv[i], 1e-5f);
        ASSERT_NEAR(sv[0], -9876.0f, 0.0f);
        ASSERT_NEAR(sv[dim + 1], -9876.0f, 0.0f);
        ASSERT_TRUE(myds_metal_swiglu(gate, gate, up, dim));
        for (uint32_t i = 0; i < dim; i++) ASSERT_NEAR(gv[i], ref[i], 1e-5f);
        free(ref);
        myds_metal_buffer_free(out);
        myds_metal_buffer_free(storage);
        myds_metal_buffer_free(gate);
        myds_metal_buffer_free(up);
    }
    printf("  [PASS] Flat SwiGLU tails, subviews, in-place output and batched dependency\n");
    return 1;
}

static int test_gemv_batch_exact(void) {
    const ggml_type types[] = { GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_F16 };
    typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0_t;
    for (size_t type_idx = 0; type_idx < 3; type_idx++) {
        ggml_type type = types[type_idx];
        for (int shape = 0; shape < 3; shape++) {
            uint32_t in_dim = shape == 0 ? 256 : (shape == 1 || type == GGML_TYPE_Q8_0 ? 64 : 35);
            uint32_t out_dim = shape == 0 ? 64 : 5;
            uint32_t n_tok = shape == 0 ? 7 : 3;
            size_t row_bytes = ggml_type_row_size(type, in_dim);
            size_t w_bytes = row_bytes * out_dim;
            size_t x_bytes = (size_t)n_tok * in_dim * sizeof(float);
            size_t y_bytes = (size_t)n_tok * out_dim * sizeof(float);
            myds_metal_buffer_t *wp = myds_metal_buffer_alloc(w_bytes + 16);
            myds_metal_buffer_t *xp = myds_metal_buffer_alloc(x_bytes + 16);
            myds_metal_buffer_t *yp = myds_metal_buffer_alloc(y_bytes + 32);
            myds_metal_buffer_t *ref = myds_metal_buffer_alloc(y_bytes);
            ASSERT_TRUE(wp && xp && yp && ref);
            myds_metal_buffer_t *w = myds_metal_buffer_sub_view(wp, 16, w_bytes);
            myds_metal_buffer_t *x = myds_metal_buffer_sub_view(xp, 16, x_bytes);
            myds_metal_buffer_t *y = myds_metal_buffer_sub_view(yp, 16, y_bytes);
            ASSERT_TRUE(w && x && y);
            memset(myds_metal_buffer_data(yp), 0xA5, y_bytes + 32);
            float *xv = myds_metal_buffer_data(x);
            for (size_t i = 0; i < (size_t)n_tok * in_dim; i++) xv[i] = sinf((float)i * 0.17f);
            if (type == GGML_TYPE_Q8_0) {
                block_q8_0_t *blocks = myds_metal_buffer_data(w);
                for (size_t b = 0; b < w_bytes / sizeof(*blocks); b++) {
                    blocks[b].d = 0x3000; /* 0.125 */
                    for (int i = 0; i < 32; i++) blocks[b].qs[i] = (int8_t)((b * 7 + i) % 15) - 7;
                }
            } else {
                for (size_t i = 0; i < (size_t)out_dim * in_dim; i++) {
                    float value = ((int)(i % 17) - 8) * 0.125f;
                    if (type == GGML_TYPE_F32) ((float *)myds_metal_buffer_data(w))[i] = value;
                    else ((_Float16 *)myds_metal_buffer_data(w))[i] = (_Float16)value;
                }
            }
            ASSERT_TRUE(myds_metal_begin_batch());
            for (uint32_t t = 0; t < n_tok; t++) {
                myds_metal_buffer_t *xt = myds_metal_buffer_sub_view(x, (size_t)t * in_dim * sizeof(float), in_dim * sizeof(float));
                myds_metal_buffer_t *yt = myds_metal_buffer_sub_view(ref, (size_t)t * out_dim * sizeof(float), out_dim * sizeof(float));
                ASSERT_TRUE(xt && yt && myds_metal_gemv(yt, w, xt, out_dim, in_dim, type));
                myds_metal_buffer_free(xt);
                myds_metal_buffer_free(yt);
            }
            ASSERT_TRUE(myds_metal_commit_batch());
            uint64_t commits = myds_metal_get_commit_count();
            ASSERT_TRUE(myds_metal_begin_batch());
            ASSERT_TRUE(myds_metal_gemv_batch(y, w, x, out_dim, in_dim, n_tok, type));
            ASSERT_TRUE(myds_metal_get_commit_count() == commits);
            ASSERT_TRUE(myds_metal_commit_batch());
            ASSERT_TRUE(myds_metal_get_commit_count() == commits + 1);
            ASSERT_TRUE(memcmp(myds_metal_buffer_data(y), myds_metal_buffer_data(ref), y_bytes) == 0);
            float *yv = myds_metal_buffer_data(y);
            for (uint32_t t = 0; t < n_tok; t++) {
                for (uint32_t r = 0; r < out_dim; r++) {
                    double expected = 0.0;
                    for (uint32_t c = 0; c < in_dim; c++) {
                        float weight;
                        if (type == GGML_TYPE_Q8_0) {
                            block_q8_0_t *blocks = myds_metal_buffer_data(w);
                            weight = 0.125f * blocks[(size_t)r * (in_dim / 32) + c / 32].qs[c % 32];
                        } else if (type == GGML_TYPE_F32) weight = ((float *)myds_metal_buffer_data(w))[(size_t)r * in_dim + c];
                        else weight = (float)((_Float16 *)myds_metal_buffer_data(w))[(size_t)r * in_dim + c];
                        expected += (double)weight * xv[(size_t)t * in_dim + c];
                    }
                    ASSERT_NEAR(yv[(size_t)t * out_dim + r], expected, 1e-4f);
                }
            }
            const uint8_t *guards = myds_metal_buffer_data(yp);
            for (size_t i = 0; i < 16; i++) {
                ASSERT_TRUE(guards[i] == 0xA5 && guards[16 + y_bytes + i] == 0xA5);
            }
            ASSERT_TRUE(!myds_metal_gemv_batch(y, w, x, out_dim, in_dim, n_tok + 1, type));
            ASSERT_TRUE(!myds_metal_gemv_batch(y, w, x, UINT32_MAX, in_dim, n_tok, type));
            if (type == GGML_TYPE_Q8_0) ASSERT_TRUE(!myds_metal_gemv_batch(y, w, x, out_dim, in_dim - 1, n_tok, type));
            myds_metal_buffer_free(w);
            myds_metal_buffer_free(x);
            myds_metal_buffer_free(y);
            myds_metal_buffer_free(wp);
            myds_metal_buffer_free(xp);
            myds_metal_buffer_free(yp);
            myds_metal_buffer_free(ref);
        }
    }
    printf("  [PASS] Batched GEMV is bit-exact to single-token Q8/F32/F16 (tails, subviews, bounds)\n");
    return 1;
}

static int test_v41_gemv_batch_exact(void) {
    const uint32_t configs[][3] = {
        {1280, 2048, 4},
        {512,  2048, 6},
        {8192, 1280, 3},
        {384,  2048, 5},
    };
    const ggml_type types[] = {GGML_TYPE_F16, GGML_TYPE_F32};
    for (size_t c = 0; c < sizeof(configs)/sizeof(configs[0]); c++) {
        uint32_t out_dim = configs[c][0];
        uint32_t in_dim = configs[c][1];
        uint32_t n_tok = configs[c][2];
        for (size_t t = 0; t < sizeof(types)/sizeof(types[0]); t++) {
            ggml_type type = types[t];
            size_t elem_bytes = type == GGML_TYPE_F32 ? 4 : 2;
            size_t w_bytes = (size_t)out_dim * in_dim * elem_bytes;
            size_t x_bytes = (size_t)n_tok * in_dim * sizeof(float);
            size_t y_bytes = (size_t)n_tok * out_dim * sizeof(float);
            myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
            myds_metal_buffer_t *x = myds_metal_buffer_alloc(x_bytes);
            myds_metal_buffer_t *y = myds_metal_buffer_alloc(y_bytes);
            myds_metal_buffer_t *ref = myds_metal_buffer_alloc(y_bytes);
            ASSERT_TRUE(w && x && y && ref);
            float *xv = myds_metal_buffer_data(x);
            for (size_t i = 0; i < (size_t)n_tok * in_dim; i++) xv[i] = sinf((float)i * 0.17f);
            for (size_t i = 0; i < (size_t)out_dim * in_dim; i++) {
                float val = ((int)(i % 17) - 8) * 0.125f;
                if (type == GGML_TYPE_F32) ((float *)myds_metal_buffer_data(w))[i] = val;
                else ((_Float16 *)myds_metal_buffer_data(w))[i] = (_Float16)val;
            }
            ASSERT_TRUE(myds_metal_begin_batch());
            for (uint32_t tok = 0; tok < n_tok; tok++) {
                myds_metal_buffer_t *xt = myds_metal_buffer_sub_view(x, (size_t)tok * in_dim * sizeof(float), in_dim * sizeof(float));
                myds_metal_buffer_t *yt = myds_metal_buffer_sub_view(ref, (size_t)tok * out_dim * sizeof(float), out_dim * sizeof(float));
                ASSERT_TRUE(xt && yt && myds_metal_v41_gemv(yt, w, xt, out_dim, in_dim, type));
                myds_metal_buffer_free(xt);
                myds_metal_buffer_free(yt);
            }
            ASSERT_TRUE(myds_metal_commit_batch());
            ASSERT_TRUE(myds_metal_begin_batch());
            ASSERT_TRUE(myds_metal_v41_gemv_batch(y, w, x, out_dim, in_dim, n_tok, type));
            ASSERT_TRUE(myds_metal_commit_batch());
            ASSERT_TRUE(memcmp(myds_metal_buffer_data(y), myds_metal_buffer_data(ref), y_bytes) == 0);
            myds_metal_buffer_free(w);
            myds_metal_buffer_free(x);
            myds_metal_buffer_free(y);
            myds_metal_buffer_free(ref);
        }
    }
    printf("  [PASS] V4.1 Batched GEMV is bit-exact to single-token V4.1 GEMV across dimensions\n");
    return 1;
}

static int test_event_validation(void) {
    uint64_t commits = myds_metal_get_commit_count();
    ASSERT_TRUE(myds_metal_begin_batch());
    ASSERT_TRUE(!myds_metal_commit_and_wait_event(0));
    ASSERT_TRUE(!myds_metal_wait_event(UINT64_MAX));
    ASSERT_TRUE(!myds_metal_commit_and_wait_event(UINT64_MAX));
    ASSERT_TRUE(myds_metal_get_commit_count() == commits);
    ASSERT_TRUE(myds_metal_commit_batch());
    ASSERT_TRUE(myds_metal_get_commit_count() == commits + 1);
    printf("  [PASS] Invalid event values preserve the unsubmitted batch\n");
    return 1;
}

static int test_event_tail_completion(int cleanup) {
    const uint32_t dim = 1u << 18;
    const uint32_t repetitions = 64;
    myds_metal_buffer_t *input = myds_metal_buffer_alloc((size_t)dim * sizeof(float));
    myds_metal_buffer_t *out = myds_metal_buffer_alloc((size_t)dim * sizeof(float));
    ASSERT_TRUE(input && out);
    float *iv = myds_metal_buffer_data(input);
    float *ov = myds_metal_buffer_data(out);
    for (uint32_t i = 0; i < dim; i++) { iv[i] = 0.125f; ov[i] = 0.0f; }
    ASSERT_TRUE(myds_metal_begin_batch());
    uint64_t event = 0;
    ASSERT_TRUE(myds_metal_signal_event(&event));
    for (uint32_t i = 0; i < repetitions; i++) ASSERT_TRUE(myds_metal_add(out, out, input, dim));
    ASSERT_TRUE(myds_metal_commit_and_wait_event(event));
    if (cleanup) myds_metal_cleanup();
    else ASSERT_TRUE(myds_metal_synchronize());
    for (uint32_t i = 0; i < dim; i++) ASSERT_NEAR(ov[i], repetitions * 0.125f, 0.0f);
    myds_metal_buffer_free(input);
    myds_metal_buffer_free(out);
    printf("  [PASS] %s waits for GPU work after the shared-event signal\n", cleanup ? "Cleanup" : "Synchronize");
    return 1;
}

/* Independent known-answer cases: BF16 expected words are literal IEEE
 * encodings; HC oracle uses double arithmetic and analytical projection inputs. */
static float v41_test_bf16(double x) {
    if (x == 0) return (float)x;
    int exponent;
    frexp(fabs(x), &exponent);
    double quantum = ldexp(1.0, exponent-8);
    return (float)(nearbyint(x/quantum)*quantum);
}

static int test_v41_bf16_norm(void) {
    const uint32_t input[] = {0x3f808000,0x3f818000,0xbf808000,0xbf818000,
        0x00008000,0x00018000,0x007fffff,0x7f7fffff,0x7f800000,
        0xff800000,0x7fc12345,0x80000000,0x3f800001};
    const uint32_t expected[] = {0x3f800000,0x3f820000,0xbf800000,0xbf820000,
        0x00000000,0x00020000,0x00800000,0x7f800000,0x7f800000,
        0xff800000,0x7fc10000,0x80000000,0x3f800000};
    myds_metal_buffer_t *storage=myds_metal_buffer_alloc(sizeof(input)+32);
    myds_metal_buffer_t *b=myds_metal_buffer_sub_view(storage,16,sizeof(input));
    ASSERT_TRUE(storage && b);
    memset(myds_metal_buffer_data(storage),0xa5,sizeof(input)+32);
    ASSERT_TRUE(myds_metal_buffer_write(b,0,input,sizeof(input)));
    ASSERT_TRUE(myds_metal_v41_bf16(b,sizeof(input)/4));
    ASSERT_TRUE(!memcmp(myds_metal_buffer_data(b),expected,sizeof(expected)));
    const unsigned char *guard=myds_metal_buffer_data(storage);
    for(unsigned i=0;i<16;i++) ASSERT_TRUE(guard[i]==0xa5 && guard[16+sizeof(input)+i]==0xa5);
    ASSERT_TRUE(!myds_metal_v41_bf16(b,sizeof(input)/4+1));
    myds_metal_buffer_free(b); myds_metal_buffer_free(storage);
    const uint32_t sizes[]={128,512,5120,20480};
    for(unsigned k=0;k<sizeof(sizes)/sizeof(sizes[0]);k++) {
        uint32_t n=sizes[k];
        myds_metal_buffer_t *x=myds_metal_buffer_alloc(n*4),*w=myds_metal_buffer_alloc(n*4),*o=myds_metal_buffer_alloc(n*4);
        ASSERT_TRUE(x&&w&&o);
        float *xp=myds_metal_buffer_data(x),*wp=myds_metal_buffer_data(w),*op=myds_metal_buffer_data(o);
        for(uint32_t i=0;i<n;i++){xp[i]=(i&1)?-2:2;uint32_t bits=(i&2)?0x3f818000:0x3f808000;memcpy(&wp[i],&bits,4);}
        ASSERT_TRUE(myds_metal_v41_rmsnorm(o,x,w,n,1e-20f,true));
        for(uint32_t i=0;i<n;i++) ASSERT_NEAR(op[i],((i&1)?-1:1)*((i&2)?1.015625f:1.0f),0);
        ASSERT_TRUE(myds_metal_v41_rmsnorm(o,x,NULL,n,1e-20f,false));
        for(uint32_t i=0;i<n;i++) ASSERT_NEAR(op[i],(i&1)?-1:1,0);
        ASSERT_TRUE(!myds_metal_v41_rmsnorm(o,x,w,n+128,1e-20f,true));
        ASSERT_TRUE(!myds_metal_v41_rmsnorm(o,x,w,64,1e-20f,true));
        myds_metal_buffer_free(x);myds_metal_buffer_free(w);myds_metal_buffer_free(o);
    }
    printf("  [PASS] V4.1 BF16 literal bits, ties, subnormals, tails and reference RMS reductions\n");
    return 1;
}

static void v41_test_split(double out[24],const float base[24],const float scale[3]) {
    for(unsigned i=0;i<4;i++) out[i]=.5*tanh(.5*((double)((int)i-12)/8*scale[0]+base[i]))+.5+1e-6;
    for(unsigned i=4;i<8;i++) out[i]=tanh(.5*((double)((int)i-12)/8*scale[1]+base[i]))+1;
    for(unsigned r=0;r<4;r++) {
        double peak=-INFINITY,sum=0;
        for(unsigned c=0;c<4;c++){unsigned i=8+r*4+c;out[i]=(double)((int)i-12)/8*scale[2]+base[i];if(out[i]>peak)peak=out[i];}
        for(unsigned c=0;c<4;c++){unsigned i=8+r*4+c;out[i]=exp(out[i]-peak);sum+=out[i];}
        for(unsigned c=0;c<4;c++){unsigned i=8+r*4+c;out[i]=out[i]/sum+1e-6;}
    }
    for(unsigned iter=0;iter<20;iter++) {
        if(iter)for(unsigned r=0;r<4;r++){double sum=1e-6;for(unsigned c=0;c<4;c++)sum+=out[8+r*4+c];for(unsigned c=0;c<4;c++)out[8+r*4+c]/=sum;}
        for(unsigned c=0;c<4;c++){double sum=1e-6;for(unsigned r=0;r<4;r++)sum+=out[8+r*4+c];for(unsigned r=0;r<4;r++)out[8+r*4+c]/=sum;}
    }
}

static int test_v41_hc(void) {
    enum {D=128};
    myds_metal_buffer_t *res=myds_metal_buffer_alloc(4*D*4),*fn=myds_metal_buffer_alloc(24*4*D*2),
        *scale=myds_metal_buffer_alloc(12),*base=myds_metal_buffer_alloc(96),*flat=myds_metal_buffer_alloc(4*D*4),
        *mix=myds_metal_buffer_alloc(96),*split=myds_metal_buffer_alloc(96),*block=myds_metal_buffer_alloc(D*4),
        *out=myds_metal_buffer_alloc(4*D*4),*pre=myds_metal_buffer_alloc(16);
    ASSERT_TRUE(res&&fn&&scale&&base&&flat&&mix&&split&&block&&out&&pre);
    float *r=myds_metal_buffer_data(res),*s=myds_metal_buffer_data(scale),*b=myds_metal_buffer_data(base);
    _Float16 *f=myds_metal_buffer_data(fn);
    for(unsigned i=0;i<4*D;i++)r[i]=2;
    memset(f,0,24*4*D*2);for(unsigned i=0;i<24;i++)f[i*4*D]=(_Float16)((float)((int)i-12)/8);
    s[0]=.75f;s[1]=.5f;s[2]=1.25f;
    for(unsigned i=0;i<24;i++)b[i]=((int)((i*7)%11)-5)*.25f;
    myds_metal_set_auto_barrier(0);
    ASSERT_TRUE(myds_metal_begin_batch());
    ASSERT_TRUE(myds_metal_v41_hc_mix(split,res,fn,GGML_TYPE_F16,scale,base,flat,mix,D,1e-20f,1e-6f));
    ASSERT_TRUE(myds_metal_commit_batch());
    const float *m=myds_metal_buffer_data(mix),*sp=myds_metal_buffer_data(split);
    double oracle[24];v41_test_split(oracle,b,s);
    for(unsigned i=0;i<24;i++){ASSERT_NEAR(m[i],(float)((int)i-12)/8,0);ASSERT_NEAR(sp[i],oracle[i],2e-6f);}
    for(unsigned h=0;h<4;h++)for(unsigned i=0;i<D;i++)r[h*D+i]=(float)(h+1)+(float)(i%7)/8;
    float *p=myds_metal_buffer_data(pre);p[0]=1;p[1]=p[2]=p[3]=0;
    ASSERT_TRUE(myds_metal_v41_hc_weighted_sum(block,res,pre,D));
    float *bl=myds_metal_buffer_data(block);for(unsigned i=0;i<D;i++)ASSERT_NEAR(bl[i],r[i],0);
    p[0]=.5f;p[1]=.25f;p[2]=p[3]=.125f;
    ASSERT_TRUE(myds_metal_v41_hc_weighted_sum(block,res,pre,D));
    for(unsigned i=0;i<D;i++)ASSERT_NEAR(bl[i],v41_test_bf16(1.875+(double)(i%7)/8),0);
    /* Non-symmetric combination matrix catches the source/destination order. */
    float *splitp=myds_metal_buffer_data(split);
    for(unsigned i=0;i<24;i++)splitp[i]=0;
    for(unsigned h=0;h<4;h++){splitp[4+h]=(float)(h+1)/4;splitp[8+((h+1)%4)*4+h]=1;}
    ASSERT_TRUE(myds_metal_v41_hc_expand(out,block,res,split,D));
    float *o=myds_metal_buffer_data(out);
    for(unsigned h=0;h<4;h++)for(unsigned i=0;i<D;i++)ASSERT_NEAR(o[h*D+i],v41_test_bf16((double)bl[i]*(h+1)/4+r[((h+1)%4)*D+i]),0);
    ASSERT_TRUE(myds_metal_v41_hc_expand(res,block,res,split,D));
    ASSERT_TRUE(!memcmp(r,o,4*D*4));
    ASSERT_TRUE(!myds_metal_v41_hc_weighted_sum(block,res,pre,D+1));
    myds_metal_set_auto_barrier(1);
    myds_metal_buffer_free(res);myds_metal_buffer_free(fn);myds_metal_buffer_free(scale);myds_metal_buffer_free(base);
    myds_metal_buffer_free(flat);myds_metal_buffer_free(mix);myds_metal_buffer_free(split);myds_metal_buffer_free(block);
    myds_metal_buffer_free(out);myds_metal_buffer_free(pre);
    printf("  [PASS] V4.1 HC F16 projection, independent Sinkhorn oracle, carried pre, expansion orientation/in-place\n");
    return 1;
}

static int test_v41_engram_dispatch_copy(void) {
    enum {D=128};
    myds_metal_buffer_t *r=myds_metal_buffer_alloc(4*D*4),*p=myds_metal_buffer_alloc(5*D*4),
        *q=myds_metal_buffer_alloc(4*D*4),*k=myds_metal_buffer_alloc(4*D*4),*copy=myds_metal_buffer_alloc(4*D*4);
    ASSERT_TRUE(r&&p&&q&&k&&copy);
    float *rv=myds_metal_buffer_data(r),*pv=myds_metal_buffer_data(p),*qv=myds_metal_buffer_data(q),*kv=myds_metal_buffer_data(k);
    for(unsigned i=0;i<4*D;i++){rv[i]=2;pv[i]=1;qv[i]=i/D<2?.25f:-.25f;kv[i]=1;}
    for(unsigned i=0;i<D;i++)pv[4*D+i]=(float)(i%7)/4;
    ASSERT_TRUE(myds_metal_begin_batch());
    ASSERT_TRUE(myds_metal_v41_engram_add(r,p,q,k,D,1e-20f));
    ASSERT_TRUE(myds_metal_buffer_copy(copy,0,r,0,4*D*4));
    uint64_t count=4*D;uint32_t groups[]={4,1,1},threads[]={32,1,1};const myds_metal_buffer_t *buffers[]={copy};
    ASSERT_TRUE(myds_metal_dispatch_kernel("kernel_dsv41_bf16_linear",&count,sizeof(count),buffers,1,groups,threads,0));
    ASSERT_TRUE(myds_metal_commit_batch());
    for(unsigned h=0;h<4;h++) {
        double dot=(h<2?.25:-.25)*sqrt(D), gate=1/(1+exp(-copysign(sqrt(fabs(dot)),dot)));
        for(unsigned i=0;i<D;i++)ASSERT_NEAR(rv[h*D+i],v41_test_bf16(2+gate*pv[4*D+i]),0);
    }
    ASSERT_TRUE(!memcmp(rv,myds_metal_buffer_data(copy),4*D*4));
    ASSERT_TRUE(!myds_metal_buffer_copy(copy,1,copy,0,16));
    ASSERT_TRUE(!myds_metal_buffer_copy(copy,0,r,0,4*D*4+1));
    ASSERT_TRUE(!myds_metal_dispatch_kernel("no_such_kernel",&count,sizeof(count),buffers,1,groups,threads,0));
    threads[0]=UINT32_MAX;
    ASSERT_TRUE(!myds_metal_dispatch_kernel("kernel_dsv41_bf16_linear",&count,sizeof(count),buffers,1,groups,threads,0));
    myds_metal_buffer_free(r);myds_metal_buffer_free(p);myds_metal_buffer_free(q);myds_metal_buffer_free(k);myds_metal_buffer_free(copy);
    printf("  [PASS] V4.1 Engram analytic gates and ordered native dispatch/copy bridge\n");
    return 1;
}

static int test_moe_batch_mm(void) {
    const uint32_t dim = 256;
    const uint32_t expert_dim = 256;
    const uint32_t n_routed_experts = 8;
    const uint32_t n_activated_experts = 6;
    const uint32_t n_tok = 32;

    uint64_t gate_bytes = (uint64_t)n_routed_experts * expert_dim * (dim / 256u) * 66u;
    uint64_t down_bytes = (uint64_t)n_routed_experts * dim * (expert_dim / 256u) * 84u;

    myds_metal_buffer_t *moe_out = myds_metal_buffer_alloc(n_tok * dim * sizeof(float));
    myds_metal_buffer_t *x_in = myds_metal_buffer_alloc(n_tok * dim * sizeof(float));
    myds_metal_buffer_t *gate_exps = myds_metal_buffer_alloc(gate_bytes);
    myds_metal_buffer_t *up_exps = myds_metal_buffer_alloc(gate_bytes);
    myds_metal_buffer_t *down_exps = myds_metal_buffer_alloc(down_bytes);
    myds_metal_buffer_t *selected_ids = myds_metal_buffer_alloc(n_tok * 6 * sizeof(int32_t));
    myds_metal_buffer_t *selected_weights = myds_metal_buffer_alloc(n_tok * 6 * sizeof(float));
    myds_metal_buffer_t *scratch_gate = myds_metal_buffer_alloc(n_tok * 6 * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_up = myds_metal_buffer_alloc(n_tok * 6 * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_mid = myds_metal_buffer_alloc(n_tok * 6 * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_down = myds_metal_buffer_alloc(n_tok * 6 * dim * sizeof(float));

    ASSERT_TRUE(moe_out && x_in && gate_exps && up_exps && down_exps &&
                selected_ids && selected_weights && scratch_gate && scratch_up &&
                scratch_mid && scratch_down);

    memset(myds_metal_buffer_data(gate_exps), 0, gate_bytes);
    memset(myds_metal_buffer_data(up_exps), 0, gate_bytes);
    memset(myds_metal_buffer_data(down_exps), 0, down_bytes);

    float *x = (float *)myds_metal_buffer_data(x_in);
    for (uint32_t i = 0; i < n_tok * dim; i++) x[i] = 1.0f;

    int32_t *ids = (int32_t *)myds_metal_buffer_data(selected_ids);
    float *w = (float *)myds_metal_buffer_data(selected_weights);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t k = 0; k < 6; k++) {
            ids[t * 6 + k] = (int32_t)(k % n_routed_experts);
            w[t * 6 + k] = 1.0f / 6.0f;
        }
    }

    /* Test 1: n_tok = 32 exercises the MM path */
    ASSERT_TRUE(myds_metal_routed_moe_batch(
        moe_out, x_in, gate_exps, up_exps, down_exps,
        selected_ids, selected_weights,
        scratch_gate, scratch_up, scratch_mid, scratch_down,
        dim, expert_dim, n_routed_experts, n_activated_experts, n_tok,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));

    float *out = (float *)myds_metal_buffer_data(moe_out);
    for (uint32_t i = 0; i < n_tok * dim; i++) {
        ASSERT_TRUE(isfinite(out[i]));
    }

    /* Test 2: n_tok = 1 exercises the decode MV path */
    ASSERT_TRUE(myds_metal_routed_moe_batch(
        moe_out, x_in, gate_exps, up_exps, down_exps,
        selected_ids, selected_weights,
        scratch_gate, scratch_up, scratch_mid, scratch_down,
        dim, expert_dim, n_routed_experts, n_activated_experts, 1,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f));

    for (uint32_t i = 0; i < dim; i++) {
        ASSERT_TRUE(isfinite(out[i]));
    }

    myds_metal_buffer_free(moe_out);
    myds_metal_buffer_free(x_in);
    myds_metal_buffer_free(gate_exps);
    myds_metal_buffer_free(up_exps);
    myds_metal_buffer_free(down_exps);
    myds_metal_buffer_free(selected_ids);
    myds_metal_buffer_free(selected_weights);
    myds_metal_buffer_free(scratch_gate);
    myds_metal_buffer_free(scratch_up);
    myds_metal_buffer_free(scratch_mid);
    myds_metal_buffer_free(scratch_down);

    printf("  [PASS] MoE batch MM path (n_tok=32) and decode MV path (n_tok=1) verification\n");
    return 1;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  视觉编码器可行性验证：BF16 → F16 + 复用现有 F16 GEMM
 *
 *  背景：DeepSeek-V4.1-Flash-Vision.gguf 的 266 个张量【全是 BF16】，
 *        而 myds 只有 F16/F32/Q8_0 的 GEMM。
 *
 *  待验假定：BF16→F16 对模型权重要【无损】，于是不必新写 BF16 内核。
 *    依据：BF16 尾数 7 位，F16 尾数 10 位 ⇒ 正常值域内 F16 精度【更高】；
 *          权重幅值 ~1e-2，远在 F16 范围 [6e-5, 65504] 内。
 *
 *  判据（不看"像不像对"，看两路是否一致）：
 *    参考值 = 用【原始 BF16 值】在 double 下累加  （= ds4 那边看到的东西）
 *    实测值 = GPU 上的 F16 GEMM
 *    两者之差 = 「BF16→F16 + 引擎累加次序」引入的全部误差
 *  用【视觉编码器的真实形状】跑，而不是随便造个矩阵。
 * ═══════════════════════════════════════════════════════════════════════ */

static uint16_t t_f32_to_bf16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;                 /* round to nearest even */
    return (uint16_t)(x >> 16);
}
static float t_bf16_to_f32(uint16_t b) {
    uint32_t x = ((uint32_t)b) << 16;
    float f; memcpy(&f, &x, 4); return f;
}
/* f32 → f16，正确支持【次正规数】与就近偶数舍入。
 *
 * ★ 第一版我写成「e < -14 直接返回 0」—— 那会把 F16 次正规区间
 *   [6e-8, 6.1e-5) 的权重【全部刷成零】。实测代价：aligner 的 47M 权重里
 *   约 5.8 万个被刷零，造成的误差(7.2e-3) 正好等于当时观测到的 6.1e-3。
 *   ⇒ 那次「失败」是我的测试代码的缺陷，不是 BF16→F16 这条路线的问题。
 *   教训：**测量工具自身要先被验证**，否则得到的是工具的行为、不是被测对象的。 */
static uint16_t t_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 31) & 1u;
    const int32_t  e    = (int32_t)((x >> 23) & 0xffu) - 127;
    const uint32_t m    = x & 0x7fffffu;               /* 23 位尾数 */
    if (((x >> 23) & 0xffu) == 0xffu)                  /* inf / nan */
        return (uint16_t)((sign << 15) | 0x7c00u | (m ? 0x200u : 0u));
    if (e > 15) return (uint16_t)((sign << 15) | 0x7c00u);      /* 上溢 */
    if (e >= -14) {                                             /* 正规数 */
        uint32_t bits = ((uint32_t)(e + 15) << 10) | (m >> 13);
        uint32_t rem  = m & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u))) bits++;
        return (uint16_t)((sign << 15) | bits);
    }
    if (e < -25) return (uint16_t)(sign << 15);                 /* 太小 ⇒ 0 */
    /* 次正规：value = sig * 2^(-23+e)，要表示成 m5 * 2^-24 ⇒ m5 = sig >> (-(e+1)) */
    const uint32_t sig   = m | 0x800000u;
    const int      shift = -(e + 1);                            /* 1..24 */
    uint32_t m5  = sig >> shift;
    uint32_t rem = sig & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (m5 & 1u))) m5++;
    if (m5 > 0x3ffu) m5 = 0x400u;                               /* 进位到最小正规数 */
    return (uint16_t)((sign << 15) | (m5 & 0x3ffu));
}
/* 确定性伪随机（不用 rand，保证可复现） */
static uint32_t t_rng_state;
static float t_rng_next(void) {
    t_rng_state = t_rng_state * 1664525u + 1013904223u;
    return ((float)((t_rng_state >> 8) & 0xffffu) / 32768.0f) - 1.0f;  /* [-1,1) */
}

static int test_vision_bf16_to_f16_gemm(void) {
    /* 全部取自 DeepSeek-V4.1-Flash-Vision.gguf 的真实张量形状 */
    static const struct { const char *name; uint32_t out_dim, in_dim, n_tok; } cs[] = {
        { "patch_embed [588,1024]",   1024,  588, 196 },   /* 14x14 网格 */
        { "qkv         [1024,3072]",  3072, 1024, 196 },
        { "attn.wo     [1024,1024]",  1024, 1024, 196 },
        { "mlp.w1      [1024,5632]",  5632, 1024, 196 },
        { "mlp.w2      [2816,1024]",  1024, 2816, 196 },
        { "aligner.w1  [9216,5120]",  5120, 9216,  23 },   /* downsample 后 */
        { "aligner.w2  [5120,5120]",  5120, 5120,  23 },
    };

    double worst_f16 = 0.0, worst_f32 = 0.0, worst_conv = 0.0;
    const char *worst_name16 = "", *worst_name32 = "";

    for (size_t ci = 0; ci < sizeof(cs)/sizeof(cs[0]); ci++) {
        const uint32_t O = cs[ci].out_dim, I = cs[ci].in_dim, T = cs[ci].n_tok;

        myds_metal_buffer_t *W = myds_metal_buffer_alloc((size_t)O * I * 2u);
        myds_metal_buffer_t *x = myds_metal_buffer_alloc((size_t)T * I * sizeof(float));
        myds_metal_buffer_t *y = myds_metal_buffer_alloc((size_t)T * O * sizeof(float));
        ASSERT_TRUE(W && x && y);

        /* 权重：先量化到 BF16（模拟真实文件），再转 F16 喂给 GPU。
         * 同时留一份 BF16 的 F32 还原值给 CPU 参考 —— 这才是"真值"。 */
        t_rng_state = 20261001u;
        uint16_t *wb = (uint16_t *)malloc((size_t)O * I * sizeof(uint16_t));
        uint16_t *w16 = (uint16_t *)malloc((size_t)O * I * sizeof(uint16_t));
        float    *wr = (float *)malloc((size_t)O * I * sizeof(float));
        ASSERT_TRUE(wb && w16 && wr);
        for (size_t i = 0; i < (size_t)O * I; i++) {
            float v = t_rng_next() * 0.05f;          /* 典型权重幅值 */
            wb[i]  = t_f32_to_bf16(v);               /* 文件里存的 */
            wr[i]  = t_bf16_to_f32(wb[i]);           /* 真值（BF16 能表示的） */
            w16[i] = t_f32_to_f16(wr[i]);            /* 喂 GPU 的 */
        }
        /* 转换自身引入多少误差？（BF16 值 → F16 值，与真值比） */
        double conv_max = 0.0; size_t subnormal = 0;
        for (size_t i = 0; i < (size_t)O * I; i++) {
            float back; uint16_t h = w16[i];
            uint32_t sg = (h >> 15) & 1u, ex = (h >> 10) & 0x1fu, fr = h & 0x3ffu;
            if (ex == 0) { back = (float)fr * 5.9604644775390625e-08f; if (sg) back = -back; subnormal++; }
            else { uint32_t bx = (sg << 31) | ((ex - 15u + 127u) << 23) | (fr << 13);
                   memcpy(&back, &bx, 4); }
            double d = fabs((double)back - (double)wr[i]);
            if (d > conv_max) conv_max = d;
        }
        float *xv = (float *)malloc((size_t)T * I * sizeof(float));
        ASSERT_TRUE(xv);
        for (size_t i = 0; i < (size_t)T * I; i++) xv[i] = t_rng_next();

        /* CPU 参考：double 累加，权重用【BF16 还原值】= 文件真值 */
        double *ref = (double *)malloc((size_t)T * O * sizeof(double));
        ASSERT_TRUE(ref);
        double sum_sq = 0.0;
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t o = 0; o < O; o++) {
                double acc = 0.0;
                for (uint32_t i = 0; i < I; i++)
                    acc += (double)wr[(size_t)o * I + i] * (double)xv[(size_t)t * I + i];
                ref[(size_t)t * O + o] = acc;
                sum_sq += acc * acc;
            }
        }
        const double rms = sqrt(sum_sq / ((double)T * O));

        /* ── 路径 A：BF16 → F16，用 F16 GEMM ── */
        memcpy(myds_metal_buffer_data(W), w16, (size_t)O * I * 2u);
        memcpy(myds_metal_buffer_data(x), xv, (size_t)T * I * sizeof(float));
        ASSERT_TRUE(myds_metal_mul_mm(y, W, x, O, I, T, GGML_TYPE_F16));
        float *yv = (float *)myds_metal_buffer_data(y);
        double max16 = 0.0;
        for (size_t i = 0; i < (size_t)T * O; i++) {
            double d = fabs(ref[i] - (double)yv[i]);
            if (d > max16) max16 = d;
        }

        /* ── 路径 B：BF16 → F32（无损！BF16 就是 F32 的高 16 位），用 F32 GEMM ── */
        myds_metal_buffer_t *W32 = myds_metal_buffer_alloc((size_t)O * I * sizeof(float));
        ASSERT_TRUE(W32);
        memcpy(myds_metal_buffer_data(W32), wr, (size_t)O * I * sizeof(float));
        ASSERT_TRUE(myds_metal_mul_mm(y, W32, x, O, I, T, GGML_TYPE_F32));
        yv = (float *)myds_metal_buffer_data(y);
        double max32 = 0.0;
        for (size_t i = 0; i < (size_t)T * O; i++) {
            double d = fabs(ref[i] - (double)yv[i]);
            if (d > max32) max32 = d;
        }

        const double rel16 = rms > 0 ? max16 / rms : 0.0;
        const double rel32 = rms > 0 ? max32 / rms : 0.0;
        printf("    %-14s in=%-5u tok=%-4u | 转换=%.1e | F16路=%.3e (相对 %.1e) | F32路=%.3e (相对 %.1e)\n",
               cs[ci].name, I, T, conv_max, max16, rel16, max32, rel32);
        if (rel16 > worst_f16) { worst_f16 = rel16; worst_name16 = cs[ci].name; }
        if (rel32 > worst_f32) { worst_f32 = rel32; worst_name32 = cs[ci].name; }
        if (conv_max > worst_conv) worst_conv = conv_max;

        free(ref); free(wb); free(w16); free(wr); free(xv);
        myds_metal_buffer_free(W32);
        myds_metal_buffer_free(W); myds_metal_buffer_free(x); myds_metal_buffer_free(y);
    }

    printf("\n    转换本身最大误差 (BF16→F16): %.2e\n", worst_conv);
    printf("    路径 A  F16 权重 + F16 GEMM 最差相对误差: %.2e (%s)\n", worst_f16, worst_name16);
    printf("    路径 B  F32 权重 + F32 GEMM 最差相对误差: %.2e (%s)\n", worst_f32, worst_name32);
    printf("\n    ※ F16 内核的累加器是 simdgroup_half8x8（F16 累加）⇒ 长点积精度差；\n");
    printf("      F32 内核是 simdgroup_float8x8（F32 累加）。BF16→F32 是【无损】的。\n");
    /* 转换必须无损：BF16→F16 的误差应当在 F16 次正规量级以下 */
    ASSERT_TRUE(worst_conv < 1e-6);
    /* F32 路径必须达到【F32 累加本身的精度】：长点积的模型是
     *   max|err| ≈ c · sqrt(I) · eps_f32 · 量级   （c ~ 4~6，取最大值若干标准差）
     * I=9216 时约 96 · 6e-8 = 6e-6（RMS），最大值到 3~4e-5 合理。
     * 阈值取 1e-4：比 F32 实测(3.5e-5) 宽 3 倍，比 F16 实测(4.1e-3) 严 40 倍
     * ⇒ 能真正区分两条路径，而不是"看着差不多"。 */
    ASSERT_TRUE(worst_f32 < 1e-4);
    printf("  [PASS] 视觉编码器 GEMM 可行性：F32 路径达 F32 累加精度（转换无损）\n");
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
    int total = 0;
#define RUN_TEST(test) do { total++; passed += (test); } while (0)
    RUN_TEST(test_v41_bf16_norm());
    RUN_TEST(test_v41_hc());
    RUN_TEST(test_v41_engram_dispatch_copy());
    RUN_TEST(test_moe_batch_mm());
    RUN_TEST(test_rmsnorm());
    RUN_TEST(test_swiglu());
    RUN_TEST(test_rope());
    RUN_TEST(test_moe_routing());
    RUN_TEST(test_gemv_f32());
    RUN_TEST(test_gemv_q8_0());
    RUN_TEST(test_mul_mm_q8_0());
    RUN_TEST(test_moe_sum());
    RUN_TEST(test_add());
    RUN_TEST(test_softmax());
    RUN_TEST(test_buffer_read_write());
    RUN_TEST(test_buffer_from_ptr_zerocopy());
    RUN_TEST(test_gemv_unsupported_quant_failfast());
    RUN_TEST(test_batch_command_buffer());
    RUN_TEST(test_swiglu_flat_shapes());
    RUN_TEST(test_gemv_batch_exact());
    RUN_TEST(test_v41_gemv_batch_exact());
    RUN_TEST(test_vision_bf16_to_f16_gemm());
    RUN_TEST(test_event_validation());
    RUN_TEST(test_event_tail_completion(0));
    RUN_TEST(test_event_tail_completion(1));
#undef RUN_TEST

    myds_metal_cleanup();

    printf("====================================================\n");
    if (passed == total) {
        printf("  ALL %d METAL OPERATOR & BATCH TESTS PASSED! (Apple Silicon GPU)\n", total);
        printf("====================================================\n");
        return 0;
    } else {
        printf("  TESTS FAILED (passed %d/%d)\n", passed, total);
        return 1;
    }
}
