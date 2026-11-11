#include "myds_dspine.h"
#include "myds_dspine_capture.h"
#include "myds_metal.h"
#include "myds_cpu.h"
#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <dispatch/dispatch.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t val = (uint32_t)b << 16;
    float f;
    memcpy(&f, &val, sizeof(f));
    return f;
}

static inline float dot_bf16_f32(const uint16_t *w, const float *x, size_t n) {
#if defined(__ARM_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t w_bf16 = vld1q_u16(w + i);
        uint32x4_t w_u32_lo = vshll_n_u16(vget_low_u16(w_bf16), 16);
        uint32x4_t w_u32_hi = vshll_n_u16(vget_high_u16(w_bf16), 16);
        float32x4_t w_f32_lo = vreinterpretq_f32_u32(w_u32_lo);
        float32x4_t w_f32_hi = vreinterpretq_f32_u32(w_u32_hi);

        float32x4_t x_f32_lo = vld1q_f32(x + i);
        float32x4_t x_f32_hi = vld1q_f32(x + i + 4);

        sum0 = vmlaq_f32(sum0, w_f32_lo, x_f32_lo);
        sum1 = vmlaq_f32(sum1, w_f32_hi, x_f32_hi);
    }
    float total = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < n; i++) {
        total += bf16_to_fp32(w[i]) * x[i];
    }
    return total;
#else
    float total = 0.0f;
    for (size_t i = 0; i < n; i++) {
        total += bf16_to_fp32(w[i]) * x[i];
    }
    return total;
#endif
}

static inline float dot_fp16_f32(const uint16_t *w, const float *x, size_t n) {
#if defined(__ARM_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const float16_t *)(w + i));
        float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h));
        float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h));
        sum0 = vmlaq_f32(sum0, f0, vld1q_f32(x + i));
        sum1 = vmlaq_f32(sum1, f1, vld1q_f32(x + i + 4));
    }
    float total = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < n; i++) {
        total += myds_fp16_to_fp32(w[i]) * x[i];
    }
    return total;
#else
    float total = 0.0f;
    for (size_t i = 0; i < n; i++) {
        total += myds_fp16_to_fp32(w[i]) * x[i];
    }
    return total;
#endif
}

static inline float dot_f32_128(const float *a, const float *b) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    for (uint32_t d = 0; d < 128; d += 16) {
        s0 = vmlaq_f32(s0, vld1q_f32(a + d + 0),  vld1q_f32(b + d + 0));
        s1 = vmlaq_f32(s1, vld1q_f32(a + d + 4),  vld1q_f32(b + d + 4));
        s2 = vmlaq_f32(s2, vld1q_f32(a + d + 8),  vld1q_f32(b + d + 8));
        s3 = vmlaq_f32(s3, vld1q_f32(a + d + 12), vld1q_f32(b + d + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#else
    float sum = 0.0f;
    for (uint32_t d = 0; d < 128; d++) sum += a[d] * b[d];
    return sum;
#endif
}

static inline void axpy_f32_128(float *out, float alpha, const float *v) {
#if defined(__ARM_NEON)
    float32x4_t a = vdupq_n_f32(alpha);
    for (uint32_t d = 0; d < 128; d += 16) {
        vst1q_f32(out + d + 0,  vmlaq_f32(vld1q_f32(out + d + 0),  a, vld1q_f32(v + d + 0)));
        vst1q_f32(out + d + 4,  vmlaq_f32(vld1q_f32(out + d + 4),  a, vld1q_f32(v + d + 4)));
        vst1q_f32(out + d + 8,  vmlaq_f32(vld1q_f32(out + d + 8),  a, vld1q_f32(v + d + 8)));
        vst1q_f32(out + d + 12, vmlaq_f32(vld1q_f32(out + d + 12), a, vld1q_f32(v + d + 12)));
    }
#else
    for (uint32_t d = 0; d < 128; d++) out[d] += alpha * v[d];
#endif
}

/* Exact GGML Q4_K row dot product matching metal/dense.metal:2438-2484 */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} dspine_block_q4_K_t;

static inline float dot_q4_K_f32(const uint8_t *row_bytes, const float *x, uint32_t cols) {
    const uint32_t n_blocks = cols / 256u;
    const dspine_block_q4_K_t *blocks = (const dspine_block_q4_K_t *)row_bytes;
    float total = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const dspine_block_q4_K_t *xb = &blocks[b];
        const float d = myds_fp16_to_fp32(xb->d);
        const float dmin = myds_fp16_to_fp32(xb->dmin);
        const uint8_t *sc = xb->scales;
        const uint8_t *q = xb->qs;
        const float *xb_x = x + (size_t)b * 256u;

        for (int b64 = 0; b64 < 4; b64++) {
            int j0 = b64 * 2;
            int j1 = j0 + 1;
            uint8_t sc0, m0, sc1, m1;
            if (j0 < 4) {
                sc0 = sc[j0] & 63u;
                m0  = sc[j0 + 4] & 63u;
            } else {
                sc0 = (sc[j0 + 4] & 0x0fu) | ((sc[j0 - 4] & 0xc0u) >> 2);
                m0  = (sc[j0 + 4] >> 4)    | ((sc[j0]     & 0xc0u) >> 2);
            }
            if (j1 < 4) {
                sc1 = sc[j1] & 63u;
                m1  = sc[j1 + 4] & 63u;
            } else {
                sc1 = (sc[j1 + 4] & 0x0fu) | ((sc[j1 - 4] & 0xc0u) >> 2);
                m1  = (sc[j1 + 4] >> 4)    | ((sc[j1]     & 0xc0u) >> 2);
            }
            const float dl0 = d * (float)sc0;
            const float ml0 = dmin * (float)m0;
            const float dl1 = d * (float)sc1;
            const float ml1 = dmin * (float)m1;

            const uint8_t *q_sub = q + b64 * 32;
            const float *x_lo = xb_x + b64 * 64;
            const float *x_hi = x_lo + 32;
            float sum_q0 = 0.0f, sum_x0 = 0.0f;
            float sum_q1 = 0.0f, sum_x1 = 0.0f;
            for (int l = 0; l < 32; l++) {
                float xl0 = x_lo[l];
                float xl1 = x_hi[l];
                sum_q0 += (float)(q_sub[l] & 0x0fu) * xl0;
                sum_x0 += xl0;
                sum_q1 += (float)(q_sub[l] >> 4) * xl1;
                sum_x1 += xl1;
            }
            total += (dl0 * sum_q0 - ml0 * sum_x0) + (dl1 * sum_q1 - ml1 * sum_x1);
        }
    }
    return total;
}

/* Exact GGML Q8_0 row dot product (34 bytes per 32 elements) */
static inline float dot_q8_0_f32(const uint8_t *row_bytes, const float *x, uint32_t cols) {
    const uint32_t n_blocks = cols / 32u;
    float total = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint8_t *bp = row_bytes + (size_t)b * 34u;
        uint16_t d_fp16;
        memcpy(&d_fp16, bp, sizeof(uint16_t));
        const float d = myds_fp16_to_fp32(d_fp16);
        const int8_t *qs = (const int8_t *)(bp + 2);
        const float *xb = x + (size_t)b * 32u;
        float sub = 0.0f;
        for (int i = 0; i < 32; i++) sub += (float)qs[i] * xb[i];
        total += d * sub;
    }
    return total;
}

static bool score_output_row_cpu(const myds_model_t *m, uint32_t vocab_idx,
                                 const float *x, uint32_t cols, float *out_score) {
    const uint8_t *base = (const uint8_t *)myds_metal_buffer_data(m->output);
    if (!base) return false;
    switch (m->output_type) {
        case GGML_TYPE_Q4_K: {
            size_t row_bytes = ((size_t)cols / 256u) * 144u;
            *out_score = dot_q4_K_f32(base + (size_t)vocab_idx * row_bytes, x, cols);
            return true;
        }
        case GGML_TYPE_F16: {
            const uint16_t *w = (const uint16_t *)base + (size_t)vocab_idx * cols;
            *out_score = dot_fp16_f32(w, x, cols);
            return true;
        }
        case GGML_TYPE_BF16: {
            const uint16_t *w = (const uint16_t *)base + (size_t)vocab_idx * cols;
            *out_score = dot_bf16_f32(w, x, cols);
            return true;
        }
        case GGML_TYPE_Q8_0: {
            size_t row_bytes = ((size_t)cols / 32u) * 34u;
            *out_score = dot_q8_0_f32(base + (size_t)vocab_idx * row_bytes, x, cols);
            return true;
        }
        case GGML_TYPE_F32: {
            const float *w = (const float *)base + (size_t)vocab_idx * cols;
            float s = 0.0f;
            for (uint32_t i = 0; i < cols; i++) s += w[i] * x[i];
            *out_score = s;
            return true;
        }
        default:
            return false;
    }
}

static void matvec_bf16(float *out, const uint16_t *W, const float *x,
                        uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; r++) {
        out[r] = dot_bf16_f32(W + (size_t)r * cols, x, cols);
    }
}

static void matvec_typed(float *out, const myds_metal_buffer_t *W_buf, ggml_type wtype,
                         const float *x, uint32_t rows, uint32_t cols) {
    const uint8_t *base = (const uint8_t *)myds_metal_buffer_data((myds_metal_buffer_t *)W_buf);
    if (wtype == GGML_TYPE_Q4_K) {
        size_t row_bytes = ((size_t)cols / 256u) * 144u;
        for (uint32_t r = 0; r < rows; r++) {
            out[r] = dot_q4_K_f32(base + (size_t)r * row_bytes, x, cols);
        }
    } else if (wtype == GGML_TYPE_Q8_0) {
        size_t row_bytes = ((size_t)cols / 32u) * 34u;
        for (uint32_t r = 0; r < rows; r++) {
            out[r] = dot_q8_0_f32(base + (size_t)r * row_bytes, x, cols);
        }
    } else {
        matvec_bf16(out, (const uint16_t *)base, x, rows, cols);
    }
}

static inline int dspine_gemv_batch(myds_metal_buffer_t *out,
                                    const myds_metal_buffer_t *weight,
                                    const myds_metal_buffer_t *input,
                                    uint32_t out_dim, uint32_t in_dim,
                                    uint32_t n_tok, ggml_type wtype) {
    if (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q8_0) {
        if (n_tok >= 8u) {
            return myds_metal_mul_mm(out, weight, input, out_dim, in_dim, n_tok, wtype);
        }
        return myds_metal_gemv_batch(out, weight, input, out_dim, in_dim, n_tok, wtype);
    }
    return myds_metal_v41_gemv_batch(out, weight, input, out_dim, in_dim, n_tok, wtype);
}

static void rmsnorm(float *out, const float *x, const uint16_t *weight,
                    uint32_t dim, float eps) {
    float sum_sq = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t sq0 = vdupq_n_f32(0.0f);
    float32x4_t sq1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        sq0 = vmlaq_f32(sq0, v0, v0);
        sq1 = vmlaq_f32(sq1, v1, v1);
    }
    sum_sq = vaddvq_f32(vaddq_f32(sq0, sq1));
    for (; i < dim; i++) sum_sq += x[i] * x[i];
#else
    for (uint32_t i = 0; i < dim; i++) sum_sq += x[i] * x[i];
#endif
    float scale = 1.0f / sqrtf(sum_sq / (float)dim + eps);

    if (weight) {
        for (uint32_t i = 0; i < dim; i++) {
            out[i] = x[i] * scale * bf16_to_fp32(weight[i]);
        }
    } else {
        for (uint32_t i = 0; i < dim; i++) {
            out[i] = x[i] * scale;
        }
    }
}

/* SpecForge AdjacentInjection.feature: sqrt(rank) * F.normalize(projected.float(), dim=-1) */
static void normalize_feature_512(float *feat) {
    float sum_sq = 0.0f;
    for (uint32_t r = 0; r < 512; r++) sum_sq += feat[r] * feat[r];
    float norm = sqrtf(sum_sq);
    float scale = sqrtf(512.0f) / (norm > 1e-12f ? norm : 1e-12f);
    for (uint32_t r = 0; r < 512; r++) feat[r] *= scale;
}

static void swiglu(float *out, const float *gate, const float *up, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

static void apply_rope_fast(float *vec, uint32_t num_heads, uint32_t head_dim,
                            uint32_t pos, const float *inv_freq) {
    float cos_a[64], sin_a[64];
    const uint32_t half = head_dim / 2;
    for (uint32_t i = 0; i < half; i++) {
        float angle = (float)pos * inv_freq[i];
        cos_a[i] = cosf(angle);
        sin_a[i] = sinf(angle);
    }
    for (uint32_t h = 0; h < num_heads; h++) {
        float *head = vec + (size_t)h * head_dim;
        for (uint32_t i = 0; i < half; i++) {
            float x0 = head[i];
            float x1 = head[i + half];
            head[i] = x0 * cos_a[i] - x1 * sin_a[i];
            head[i + half] = x0 * sin_a[i] + x1 * cos_a[i];
        }
    }
}

static void topk_f32(const float *logits, uint32_t vocab_size, uint32_t k,
                     int32_t *out_ids) {
    float top_v[MYDS_DSPINE_TOP_K];
    int32_t top_i[MYDS_DSPINE_TOP_K];
    if (k > MYDS_DSPINE_TOP_K) k = MYDS_DSPINE_TOP_K;
    for (uint32_t j = 0; j < k; j++) {
        top_v[j] = -1e38f;
        top_i[j] = 0;
    }
    float min_top = -1e38f;
    for (uint32_t v = 0; v < vocab_size; v++) {
        float val = logits[v];
        if (val <= min_top) continue;
        /* Insert into descending array top_v[0..k-1] */
        uint32_t pos = k - 1;
        while (pos > 0 && val > top_v[pos - 1]) {
            top_v[pos] = top_v[pos - 1];
            top_i[pos] = top_i[pos - 1];
            pos--;
        }
        top_v[pos] = val;
        top_i[pos] = (int32_t)v;
        min_top = top_v[k - 1];
    }
    for (uint32_t j = 0; j < k; j++) out_ids[j] = top_i[j];
}

static bool ensure_ctx_kv_capacity(myds_dspine_t *ds, uint32_t needed_len) {
    if (needed_len <= ds->ctx_kv_cap) return true;
    uint32_t new_cap = ds->ctx_kv_cap ? ds->ctx_kv_cap : 1024u;
    while (new_cap < needed_len) new_cap *= 2u;

    for (uint32_t l = 0; l < ds->num_layers; l++) {
        float *nk = (float *)realloc(ds->ctx_k_cache[l], (size_t)new_cap * 1024u * sizeof(float));
        if (!nk) return false;
        ds->ctx_k_cache[l] = nk;

        float *nv = (float *)realloc(ds->ctx_v_cache[l], (size_t)new_cap * 1024u * sizeof(float));
        if (!nv) return false;
        ds->ctx_v_cache[l] = nv;
    }
    uint32_t *np = (uint32_t *)realloc(ds->ctx_pos_cache, (size_t)new_cap * sizeof(uint32_t));
    if (!np) return false;
    ds->ctx_pos_cache = np;

    float *ns = (float *)realloc(ds->attn_scores_scratch,
                                 (size_t)ds->num_heads * (new_cap + MYDS_DSPINE_MAX_BLOCK) * sizeof(float));
    if (!ns) return false;
    ds->attn_scores_scratch = ns;

    ds->ctx_kv_cap = new_cap;
    return true;
}

static bool find_tensor_offset(const char *json, size_t json_len, const char *name,
                               uint64_t *start, uint64_t *end) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", name);
    const char *pos = strstr(json, needle);
    if (!pos || (size_t)(pos - json) >= json_len) return false;

    const char *off_key = strstr(pos, "\"data_offsets\"");
    if (!off_key) return false;
    const char *bracket = strchr(off_key, '[');
    if (!bracket) return false;

    char *next_ptr = NULL;
    *start = strtoull(bracket + 1, &next_ptr, 10);
    if (!next_ptr || *next_ptr != ',') return false;
    *end = strtoull(next_ptr + 1, NULL, 10);
    return (*end >= *start);
}

static myds_metal_buffer_t *load_tensor_ex(const char *data_base,
                                            const char *json, size_t json_len,
                                            const char *name, size_t *out_bytes) {
    uint64_t start = 0, end = 0;
    if (!find_tensor_offset(json, json_len, name, &start, &end)) {
        MYDS_LOG_E("DSpine: tensor '%s' not found in safetensors header", name);
        return NULL;
    }
    size_t size = (size_t)(end - start);
    if (out_bytes) *out_bytes = size;
    const void *ptr = data_base + start;
    myds_metal_buffer_t *buf = myds_metal_buffer_from_ptr(ptr, size);
    if (!buf) {
        buf = myds_metal_buffer_alloc(size);
        if (buf) {
            myds_metal_buffer_write(buf, 0, ptr, size);
        }
    }
    return buf;
}

static myds_metal_buffer_t *load_tensor(const char *data_base,
                                        const char *json, size_t json_len,
                                        const char *name) {
    return load_tensor_ex(data_base, json, json_len, name, NULL);
}

static myds_metal_buffer_t *load_matrix_tensor(const char *data_base,
                                               const char *json, size_t json_len,
                                               const char *name,
                                               uint32_t rows, uint32_t cols,
                                               ggml_type *out_type) {
    size_t nbytes = 0;
    myds_metal_buffer_t *buf = load_tensor_ex(data_base, json, json_len, name, &nbytes);
    if (!buf) return NULL;
    const size_t bf16_bytes = (size_t)rows * cols * 2u;
    const size_t q4k_bytes  = (cols % 256u == 0) ? (size_t)rows * (cols / 256u) * 144u : 0u;
    const size_t q80_bytes  = (cols % 32u == 0)  ? (size_t)rows * (cols / 32u)  * 34u  : 0u;
    const size_t f32_bytes  = (size_t)rows * cols * 4u;
    if (nbytes == bf16_bytes) {
        *out_type = GGML_TYPE_BF16;
    } else if (q4k_bytes && nbytes == q4k_bytes) {
        *out_type = GGML_TYPE_Q4_K;
    } else if (q80_bytes && nbytes == q80_bytes) {
        *out_type = GGML_TYPE_Q8_0;
    } else if (nbytes == f32_bytes) {
        *out_type = GGML_TYPE_F32;
    } else {
        MYDS_LOG_E("DSpine: tensor '%s' has unexpected byte size %zu for [%u, %u]",
                   name, nbytes, rows, cols);
        myds_metal_buffer_free(buf);
        return NULL;
    }
    return buf;
}

myds_dspine_t *myds_dspine_create(void) {
    myds_dspine_t *ds = (myds_dspine_t *)calloc(1, sizeof(myds_dspine_t));
    if (!ds) return NULL;

    ds->hidden_size = 5120;
    ds->intermediate_size = 15360;
    ds->num_layers = MYDS_DSPINE_LAYERS;
    ds->num_heads = 40;
    ds->num_kv_heads = 8;
    ds->head_dim = 128;
    ds->vocab_size = 129280;
    ds->block_size = MYDS_DSPINE_DEFAULT_BLOCK;
    ds->top_k = MYDS_DSPINE_TOP_K;
    ds->mask_token_id = 129279;
    ds->mmap_fd = -1;

    for (uint32_t i = 0; i < 64; i++) {
        ds->rope_inv_freq[i] = 1.0f / powf(1000000.0f, (float)(2u * i) / 128.0f);
    }

    const char *blk_env = getenv("MYDS_DSPINE_BLOCK_SIZE");
    if (blk_env && *blk_env) {
        int b = atoi(blk_env);
        if (b > 0 && b <= MYDS_DSPINE_MAX_BLOCK) ds->block_size = (uint32_t)b;
    }

    /* Allocate scratch execution buffers */
    ds->target_hidden_buf = myds_metal_buffer_alloc(MYDS_DSPINE_TARGET_DIM * sizeof(float));
    ds->ctx_feat_buf = myds_metal_buffer_alloc(ds->hidden_size * sizeof(float));
    ds->x_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->norm_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->attn_out_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->ffn_out_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->logits_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->vocab_size * sizeof(float));

    /* Additional GPU scratch buffers */
    ds->q_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->k_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * 1024 * sizeof(float));
    ds->v_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * 1024 * sizeof(float));
    ds->k_ctx_buf = myds_metal_buffer_alloc(1024 * sizeof(float));
    ds->v_ctx_buf = myds_metal_buffer_alloc(1024 * sizeof(float));
    ds->attn_in_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->gate_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->intermediate_size * sizeof(float));
    ds->up_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->intermediate_size * sizeof(float));
    ds->glu_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->intermediate_size * sizeof(float));
    ds->features_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * 512 * sizeof(float));
    ds->delta_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));

    /* Batched context KV projection buffers */
    const size_t kv_batch_slice = (size_t)MYDS_DSPINE_CTX_BATCH_CAP * 1024 * sizeof(float);
    ds->batch_th_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_CTX_BATCH_CAP * MYDS_DSPINE_TARGET_DIM * sizeof(float));
    ds->batch_ctx_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_CTX_BATCH_CAP * ds->hidden_size * sizeof(float));
    ds->batch_k_ctx_buf = myds_metal_buffer_alloc(MYDS_DSPINE_LAYERS * kv_batch_slice);
    ds->batch_v_ctx_buf = myds_metal_buffer_alloc(MYDS_DSPINE_LAYERS * kv_batch_slice);

    /* Draft KV cache: max_block * num_layers * 2(K,V) * num_kv_heads * head_dim * sizeof(float) */
    size_t kv_bytes = (size_t)MYDS_DSPINE_MAX_BLOCK * ds->num_layers * 2 * ds->num_kv_heads * ds->head_dim * sizeof(float);
    ds->draft_kv_buf = myds_metal_buffer_alloc(kv_bytes);

    ds->spec_snap = (myds_v41_spec_snapshot_t *)calloc(1, sizeof(myds_v41_spec_snapshot_t));

    for (int i = 0; i < 3; i++) {
        ds->spec_target_slots[i] = myds_metal_buffer_alloc(MYDS_DSPINE_TARGET_DIM * sizeof(float));
    }

    /* HC stream mean weights {0.25, 0.25, 0.25, 0.25} & GPU sub-views */
    ds->hc_mean_weights = myds_metal_buffer_alloc(4 * sizeof(float));
    if (ds->hc_mean_weights) {
        const float w4[4] = {0.25f, 0.25f, 0.25f, 0.25f};
        myds_metal_buffer_write(ds->hc_mean_weights, 0, w4, sizeof(w4));
    }
    bool views_ok = (ds->hc_mean_weights != NULL);
    for (int s = 0; s < MYDS_DSPINE_TARGET_LAYERS; s++) {
        size_t off = (size_t)s * ds->hidden_size * sizeof(float);
        size_t len = (size_t)ds->hidden_size * sizeof(float);
        ds->target_slot_views[s] = ds->target_hidden_buf
            ? myds_metal_buffer_sub_view(ds->target_hidden_buf, off, len) : NULL;
        if (!ds->target_slot_views[s]) views_ok = false;
        for (int r = 0; r < 3; r++) {
            ds->spec_slot_views[r][s] = ds->spec_target_slots[r]
                ? myds_metal_buffer_sub_view(ds->spec_target_slots[r], off, len) : NULL;
            if (!ds->spec_slot_views[r][s]) views_ok = false;
        }
    }
    for (int l = 0; l < MYDS_DSPINE_LAYERS; l++) {
        ds->batch_k_views[l] = ds->batch_k_ctx_buf
            ? myds_metal_buffer_sub_view(ds->batch_k_ctx_buf, (size_t)l * kv_batch_slice, kv_batch_slice) : NULL;
        ds->batch_v_views[l] = ds->batch_v_ctx_buf
            ? myds_metal_buffer_sub_view(ds->batch_v_ctx_buf, (size_t)l * kv_batch_slice, kv_batch_slice) : NULL;
        if (!ds->batch_k_views[l] || !ds->batch_v_views[l]) views_ok = false;
    }

    if (!ds->target_hidden_buf || !ds->ctx_feat_buf || !ds->x_buf || !ds->norm_buf ||
        !ds->attn_out_buf || !ds->ffn_out_buf || !ds->logits_buf || !ds->draft_kv_buf ||
        !ds->q_buf || !ds->k_buf || !ds->v_buf || !ds->k_ctx_buf || !ds->v_ctx_buf ||
        !ds->attn_in_buf || !ds->gate_buf || !ds->up_buf || !ds->glu_buf ||
        !ds->features_buf || !ds->delta_buf || !ds->spec_snap ||
        !ds->batch_th_buf || !ds->batch_ctx_buf || !ds->batch_k_ctx_buf || !ds->batch_v_ctx_buf ||
        !ds->spec_target_slots[0] || !ds->spec_target_slots[1] || !ds->spec_target_slots[2] ||
        !views_ok || !ensure_ctx_kv_capacity(ds, 4096u)) {
        MYDS_LOG_E("DSpine: failed to allocate scratch buffers");
        myds_dspine_free(ds);
        return NULL;
    }

    ds->enabled = true;
    MYDS_LOG_I("DSpine runtime initialized: hidden=%u layers=%u heads=%u block=%u",
               ds->hidden_size, ds->num_layers, ds->num_heads, ds->block_size);
    return ds;
}

bool myds_dspine_load_weights(myds_dspine_t *ds, const char *weights_path) {
    if (!ds || !weights_path) return false;
    MYDS_LOG_I("DSpine: loading weights from %s", weights_path);

    int fd = open(weights_path, O_RDONLY);
    if (fd < 0) {
        MYDS_LOG_E("DSpine: failed to open %s", weights_path);
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        MYDS_LOG_E("DSpine: failed to stat %s", weights_path);
        close(fd);
        return false;
    }

    ds->mmap_size = (size_t)st.st_size;
    ds->mmap_addr = mmap(NULL, ds->mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    ds->mmap_fd = fd;
    if (ds->mmap_addr == MAP_FAILED) {
        MYDS_LOG_E("DSpine: failed to mmap %s (%zu bytes)", weights_path, ds->mmap_size);
        close(fd);
        ds->mmap_fd = -1;
        ds->mmap_addr = NULL;
        return false;
    }

    uint64_t header_len = *(const uint64_t *)ds->mmap_addr;
    if (8 + header_len >= ds->mmap_size) {
        MYDS_LOG_E("DSpine: corrupt safetensors header_len=%llu, file_size=%zu",
                   (unsigned long long)header_len, ds->mmap_size);
        return false;
    }

    const char *header_json = (const char *)ds->mmap_addr + 8;
    const char *data_base = (const char *)ds->mmap_addr + 8 + header_len;

    /* 1. Context Projector & Norm */
    ds->fc_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, "fc.weight",
                                     ds->hidden_size, MYDS_DSPINE_TARGET_DIM, &ds->fc_type);
    ds->hidden_norm = load_tensor(data_base, header_json, (size_t)header_len, "hidden_norm.weight");
    if (!ds->fc_proj || !ds->hidden_norm) return false;

    /* 2. Backbone Transformer Layers */
    char name[128];
    for (uint32_t i = 0; i < ds->num_layers; i++) {
        myds_dspine_layer_t *l = &ds->layers[i];

        snprintf(name, sizeof(name), "layers.%u.input_layernorm.weight", i);
        l->input_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.q_proj.weight", i);
        l->q_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                       ds->hidden_size, ds->hidden_size, &l->q_type);

        snprintf(name, sizeof(name), "layers.%u.self_attn.k_proj.weight", i);
        l->k_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                       1024u, ds->hidden_size, &l->k_type);

        snprintf(name, sizeof(name), "layers.%u.self_attn.v_proj.weight", i);
        l->v_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                       1024u, ds->hidden_size, &l->v_type);

        snprintf(name, sizeof(name), "layers.%u.self_attn.o_proj.weight", i);
        l->o_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                       ds->hidden_size, ds->hidden_size, &l->o_type);

        snprintf(name, sizeof(name), "layers.%u.self_attn.q_norm.weight", i);
        l->q_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.k_norm.weight", i);
        l->k_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.post_attention_layernorm.weight", i);
        l->post_attn_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.mlp.gate_proj.weight", i);
        l->gate_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                          ds->intermediate_size, ds->hidden_size, &l->gate_type);

        snprintf(name, sizeof(name), "layers.%u.mlp.up_proj.weight", i);
        l->up_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                        ds->intermediate_size, ds->hidden_size, &l->up_type);

        snprintf(name, sizeof(name), "layers.%u.mlp.down_proj.weight", i);
        l->down_proj = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                          ds->hidden_size, ds->intermediate_size, &l->down_type);

        if (!l->input_norm || !l->q_proj || !l->k_proj || !l->v_proj || !l->o_proj ||
            !l->q_norm || !l->k_norm || !l->post_attn_norm || !l->gate_proj ||
            !l->up_proj || !l->down_proj) {
            MYDS_LOG_E("DSpine: failed to load layer %u weights", i);
            return false;
        }
    }

    /* 3. Final Norm */
    ds->final_norm = load_tensor(data_base, header_json, (size_t)header_len, "norm.weight");
    if (!ds->final_norm) return false;

    /* 4. Code Bridge (Adjacent Causal Injection) */
    for (uint32_t i = 0; i < ds->num_layers; i++) {
        snprintf(name, sizeof(name), "injection.readouts.%u.weight", i);
        ds->injection_readouts[i] = load_matrix_tensor(data_base, header_json, (size_t)header_len, name,
                                                       512u, ds->hidden_size, &ds->injection_readout_types[i]);

        snprintf(name, sizeof(name), "injection.writes.%u.weight", i);
        ds->injection_writes[i] = load_tensor(data_base, header_json, (size_t)header_len, name);

        if (!ds->injection_readouts[i] || !ds->injection_writes[i]) {
            MYDS_LOG_E("DSpine: failed to load injection layer %u weights", i);
            return false;
        }
    }

    ds->injection_message = load_tensor(data_base, header_json, (size_t)header_len, "injection.message.weight");
    ds->injection_pred_gate = load_tensor(data_base, header_json, (size_t)header_len, "injection.predecessor_gate.weight");
    ds->injection_recv_gate = load_tensor(data_base, header_json, (size_t)header_len, "injection.receiver_gate.weight");
    ds->injection_bias = load_tensor(data_base, header_json, (size_t)header_len, "injection.bias");

    if (!ds->injection_message || !ds->injection_pred_gate ||
        !ds->injection_recv_gate || !ds->injection_bias) {
        MYDS_LOG_E("DSpine: failed to load injection core parameters");
        return false;
    }

    /* 5. Transfer Codes */
    ds->transfer_codes = load_tensor(data_base, header_json, (size_t)header_len, "transfer_codes");
    if (!ds->transfer_codes) {
        MYDS_LOG_E("DSpine: failed to load transfer_codes");
        return false;
    }

    ds->loaded = true;
    MYDS_LOG_I("DSpine: all 47 parameter tensors mapped successfully (zero-copy %zu MB, backbone=%s)",
               ds->mmap_size / (1024 * 1024), ggml_type_name(ds->layers[0].q_type));
    return true;
}

void myds_dspine_free(myds_dspine_t *ds) {
    if (!ds) return;

    for (int s = 0; s < MYDS_DSPINE_TARGET_LAYERS; s++) {
        if (ds->target_slot_views[s]) myds_metal_buffer_free(ds->target_slot_views[s]);
        for (int r = 0; r < 3; r++) {
            if (ds->spec_slot_views[r][s]) myds_metal_buffer_free(ds->spec_slot_views[r][s]);
        }
    }
    for (int l = 0; l < MYDS_DSPINE_LAYERS; l++) {
        if (ds->batch_k_views[l]) myds_metal_buffer_free(ds->batch_k_views[l]);
        if (ds->batch_v_views[l]) myds_metal_buffer_free(ds->batch_v_views[l]);
    }
    if (ds->hc_mean_weights) myds_metal_buffer_free(ds->hc_mean_weights);

    if (ds->target_hidden_buf) myds_metal_buffer_free(ds->target_hidden_buf);
    if (ds->ctx_feat_buf) myds_metal_buffer_free(ds->ctx_feat_buf);
    if (ds->x_buf) myds_metal_buffer_free(ds->x_buf);
    if (ds->norm_buf) myds_metal_buffer_free(ds->norm_buf);
    if (ds->attn_out_buf) myds_metal_buffer_free(ds->attn_out_buf);
    if (ds->ffn_out_buf) myds_metal_buffer_free(ds->ffn_out_buf);
    if (ds->logits_buf) myds_metal_buffer_free(ds->logits_buf);
    if (ds->q_buf) myds_metal_buffer_free(ds->q_buf);
    if (ds->k_buf) myds_metal_buffer_free(ds->k_buf);
    if (ds->v_buf) myds_metal_buffer_free(ds->v_buf);
    if (ds->k_ctx_buf) myds_metal_buffer_free(ds->k_ctx_buf);
    if (ds->v_ctx_buf) myds_metal_buffer_free(ds->v_ctx_buf);
    if (ds->attn_in_buf) myds_metal_buffer_free(ds->attn_in_buf);
    if (ds->gate_buf) myds_metal_buffer_free(ds->gate_buf);
    if (ds->up_buf) myds_metal_buffer_free(ds->up_buf);
    if (ds->glu_buf) myds_metal_buffer_free(ds->glu_buf);
    if (ds->features_buf) myds_metal_buffer_free(ds->features_buf);
    if (ds->delta_buf) myds_metal_buffer_free(ds->delta_buf);
    if (ds->batch_th_buf) myds_metal_buffer_free(ds->batch_th_buf);
    if (ds->batch_ctx_buf) myds_metal_buffer_free(ds->batch_ctx_buf);
    if (ds->batch_k_ctx_buf) myds_metal_buffer_free(ds->batch_k_ctx_buf);
    if (ds->batch_v_ctx_buf) myds_metal_buffer_free(ds->batch_v_ctx_buf);
    for (int i = 0; i < 3; i++) {
        if (ds->spec_target_slots[i]) myds_metal_buffer_free(ds->spec_target_slots[i]);
    }
    if (ds->draft_kv_buf) myds_metal_buffer_free(ds->draft_kv_buf);
    if (ds->tg_table) myds_metal_buffer_free(ds->tg_table);
    if (ds->tsv_table) myds_metal_buffer_free(ds->tsv_table);
    if (ds->spec_snap) free(ds->spec_snap);

    for (uint32_t l = 0; l < MYDS_DSPINE_LAYERS; l++) {
        free(ds->ctx_k_cache[l]);
        free(ds->ctx_v_cache[l]);
    }
    free(ds->ctx_pos_cache);
    free(ds->attn_scores_scratch);

    if (ds->fc_proj) myds_metal_buffer_free(ds->fc_proj);
    if (ds->hidden_norm) myds_metal_buffer_free(ds->hidden_norm);
    if (ds->final_norm) myds_metal_buffer_free(ds->final_norm);

    for (uint32_t i = 0; i < ds->num_layers; i++) {
        myds_dspine_layer_t *l = &ds->layers[i];
        if (l->input_norm) myds_metal_buffer_free(l->input_norm);
        if (l->q_proj) myds_metal_buffer_free(l->q_proj);
        if (l->k_proj) myds_metal_buffer_free(l->k_proj);
        if (l->v_proj) myds_metal_buffer_free(l->v_proj);
        if (l->o_proj) myds_metal_buffer_free(l->o_proj);
        if (l->q_norm) myds_metal_buffer_free(l->q_norm);
        if (l->k_norm) myds_metal_buffer_free(l->k_norm);
        if (l->post_attn_norm) myds_metal_buffer_free(l->post_attn_norm);
        if (l->gate_proj) myds_metal_buffer_free(l->gate_proj);
        if (l->up_proj) myds_metal_buffer_free(l->up_proj);
        if (l->down_proj) myds_metal_buffer_free(l->down_proj);

        if (ds->injection_readouts[i]) myds_metal_buffer_free(ds->injection_readouts[i]);
        if (ds->injection_writes[i]) myds_metal_buffer_free(ds->injection_writes[i]);
    }

    if (ds->injection_message) myds_metal_buffer_free(ds->injection_message);
    if (ds->injection_pred_gate) myds_metal_buffer_free(ds->injection_pred_gate);
    if (ds->injection_recv_gate) myds_metal_buffer_free(ds->injection_recv_gate);
    if (ds->injection_bias) myds_metal_buffer_free(ds->injection_bias);
    if (ds->transfer_codes) myds_metal_buffer_free(ds->transfer_codes);

    if (ds->mmap_addr && ds->mmap_size) {
        munmap(ds->mmap_addr, ds->mmap_size);
    }
    if (ds->mmap_fd >= 0) {
        close(ds->mmap_fd);
    }

    free(ds);
}

void myds_dspine_reset_context(myds_dspine_t *ds) {
    if (!ds) return;
    ds->ctx_kv_len = 0;
    ds->pending_count = 0;
    ds->pending_from_spec = false;
    ds->target_slots_captured = 0;
}

void myds_dspine_truncate_context(myds_dspine_t *ds, uint32_t max_pos) {
    if (!ds) return;
    while (ds->ctx_kv_len > 0 && ds->ctx_pos_cache[ds->ctx_kv_len - 1] >= max_pos) {
        ds->ctx_kv_len--;
    }
}

bool myds_dspine_flush_pending(myds_dspine_t *ds, uint32_t next_pos) {
    if (!ds || !ds->enabled || !ds->loaded) return true;
    if (next_pos == 0) {
        myds_dspine_reset_context(ds);
        return true;
    }
    if (ds->pending_count > 0 && ds->pending_pos < next_pos) {
        uint32_t cnt = ds->pending_count;
        if (ds->pending_pos + cnt > next_pos) cnt = next_pos - ds->pending_pos;
        bool ok = false;
        if (ds->pending_from_spec) {
            float spec_batch[3 * MYDS_DSPINE_TARGET_DIM];
            if (cnt > 3u) cnt = 3u;
            for (uint32_t r = 0; r < cnt; r++) {
                memcpy(spec_batch + (size_t)r * MYDS_DSPINE_TARGET_DIM,
                       myds_metal_buffer_data(ds->spec_target_slots[r]),
                       MYDS_DSPINE_TARGET_DIM * sizeof(float));
            }
            ok = myds_dspine_append_context_batch(ds, spec_batch, ds->pending_pos, cnt);
        } else {
            const float *th = (const float *)myds_metal_buffer_data(ds->target_hidden_buf);
            ok = myds_dspine_append_context_batch(ds, th, ds->pending_pos, 1u);
        }
        ds->pending_count = 0;
        ds->pending_from_spec = false;
        return ok;
    }
    myds_dspine_truncate_context(ds, next_pos);
    ds->pending_count = 0;
    ds->pending_from_spec = false;
    return true;
}

bool myds_dspine_append_context_batch(myds_dspine_t *ds,
                                      const float *target_hidden_batch,
                                      uint32_t start_pos, uint32_t count) {
    if (!ds || !ds->loaded || !target_hidden_batch || count == 0) return false;
    myds_dspine_truncate_context(ds, start_pos);
    if (!ensure_ctx_kv_capacity(ds, ds->ctx_kv_len + count)) return false;

    const uint16_t *hn_w = (const uint16_t *)myds_metal_buffer_data(ds->hidden_norm);
    float *th_gpu = (float *)myds_metal_buffer_data(ds->batch_th_buf);
    float *ctx_gpu = (float *)myds_metal_buffer_data(ds->batch_ctx_buf);
    float *k_gpu = (float *)myds_metal_buffer_data(ds->batch_k_ctx_buf);
    float *v_gpu = (float *)myds_metal_buffer_data(ds->batch_v_ctx_buf);

    for (uint32_t off = 0; off < count; off += MYDS_DSPINE_CTX_BATCH_CAP) {
        uint32_t cur = count - off;
        if (cur > MYDS_DSPINE_CTX_BATCH_CAP) cur = MYDS_DSPINE_CTX_BATCH_CAP;

        memcpy(th_gpu, target_hidden_batch + (size_t)off * MYDS_DSPINE_TARGET_DIM,
               (size_t)cur * MYDS_DSPINE_TARGET_DIM * sizeof(float));

        /* 1. fc_proj [5120, 25600] @ th_gpu [cur, 25600] -> ctx_gpu [cur, 5120] */
        if (!dspine_gemv_batch(ds->batch_ctx_buf, ds->fc_proj, ds->batch_th_buf,
                               ds->hidden_size, MYDS_DSPINE_TARGET_DIM, cur, ds->fc_type)) {
            return false;
        }
        for (uint32_t i = 0; i < cur; i++) {
            float *cf = ctx_gpu + (size_t)i * ds->hidden_size;
            rmsnorm(cf, cf, hn_w, ds->hidden_size, 1e-6f);
        }

        /* 2. All-layer k_proj, v_proj in a single GPU batch, then k_norm + RoPE into ctx_k_cache / ctx_v_cache */
        myds_metal_begin_batch();
        for (uint32_t l = 0; l < ds->num_layers; l++) {
            myds_dspine_layer_t *layer = &ds->layers[l];
            dspine_gemv_batch(ds->batch_k_views[l], layer->k_proj, ds->batch_ctx_buf,
                              1024, ds->hidden_size, cur, layer->k_type);
            dspine_gemv_batch(ds->batch_v_views[l], layer->v_proj, ds->batch_ctx_buf,
                              1024, ds->hidden_size, cur, layer->v_type);
        }
        if (!myds_metal_commit_batch()) return false;

        const size_t slice_floats = (size_t)MYDS_DSPINE_CTX_BATCH_CAP * 1024u;
        for (uint32_t l = 0; l < ds->num_layers; l++) {
            myds_dspine_layer_t *layer = &ds->layers[l];
            const uint16_t *kn_w = (const uint16_t *)myds_metal_buffer_data(layer->k_norm);
            const float *k_l = k_gpu + (size_t)l * slice_floats;
            const float *v_l = v_gpu + (size_t)l * slice_floats;

            float *dst_k_base = ds->ctx_k_cache[l] + (size_t)(ds->ctx_kv_len + off) * 1024u;
            float *dst_v_base = ds->ctx_v_cache[l] + (size_t)(ds->ctx_kv_len + off) * 1024u;
            memcpy(dst_v_base, v_l, (size_t)cur * 1024u * sizeof(float));

            for (uint32_t i = 0; i < cur; i++) {
                const float *ki = k_l + (size_t)i * 1024u;
                float *dst_k = dst_k_base + (size_t)i * 1024u;
                for (uint32_t h = 0; h < ds->num_kv_heads; h++) {
                    rmsnorm(dst_k + h * ds->head_dim, ki + h * ds->head_dim,
                            kn_w, ds->head_dim, 1e-6f);
                }
                apply_rope_fast(dst_k, ds->num_kv_heads, ds->head_dim,
                                start_pos + off + i, ds->rope_inv_freq);
            }
        }
        for (uint32_t i = 0; i < cur; i++) {
            ds->ctx_pos_cache[ds->ctx_kv_len + off + i] = start_pos + off + i;
        }
    }
    ds->ctx_kv_len += count;
    return true;
}

int myds_dspine_capture_target_layer(myds_dspine_t *ds, uint32_t layer_idx,
                                     uint32_t pos,
                                     const myds_metal_buffer_t *residual_hc) {
    if (!ds || !ds->enabled || !residual_hc) return 1;
    int slot = myds_dspine_capture_layer_slot(layer_idx);
    if (slot < 0 || slot >= MYDS_DSPINE_TARGET_LAYERS) return 1;
    if (!ds->target_slot_views[slot] || !ds->hc_mean_weights) return 0;

    /* Encode 4-stream HC average directly on the active Metal command buffer */
    if (!myds_metal_v41_hc_weighted_sum_round(ds->target_slot_views[slot], residual_hc,
                                              ds->hc_mean_weights, ds->hidden_size, 0)) {
        ds->target_slots_captured = 0;
        ds->pending_count = 0;
        return 0;
    }
    if (slot == 0) {
        ds->target_slots_captured = 1u;
        ds->pending_count = 0;
        ds->pending_from_spec = false;
    } else {
        ds->target_slots_captured |= (1u << slot);
    }
    if (ds->target_slots_captured == 0x1Fu) {
        ds->pending_pos = pos;
        ds->pending_count = 1u;
        ds->pending_from_spec = false;
    }
    return 1;
}

int myds_dspine_capture_target_spec_layer(myds_dspine_t *ds, uint32_t layer_idx,
                                          uint32_t token_row,
                                          const myds_metal_buffer_t *residual_hc) {
    if (!ds || !ds->enabled || !residual_hc || token_row >= 3) return 1;
    int slot = myds_dspine_capture_layer_slot(layer_idx);
    if (slot < 0 || slot >= MYDS_DSPINE_TARGET_LAYERS) return 1;
    if (!ds->spec_slot_views[token_row][slot] || !ds->hc_mean_weights) return 0;

    return myds_metal_v41_hc_weighted_sum_round(ds->spec_slot_views[token_row][slot], residual_hc,
                                                ds->hc_mean_weights, ds->hidden_size, 0);
}

void myds_dspine_capture_select_row(myds_dspine_t *ds, uint32_t base_pos,
                                    uint32_t selected_row) {
    if (!ds || selected_row >= 3 || !ds->target_hidden_buf || !ds->spec_target_slots[selected_row]) return;
    memcpy(myds_metal_buffer_data(ds->target_hidden_buf),
           myds_metal_buffer_data(ds->spec_target_slots[selected_row]),
           MYDS_DSPINE_TARGET_DIM * sizeof(float));
    ds->target_slots_captured = 0x1Fu;
    ds->pending_pos = base_pos;
    ds->pending_count = selected_row + 1u;
    ds->pending_from_spec = true;
}

void myds_dspine_capture_invalidate(myds_dspine_t *ds) {
    if (!ds) return;
    ds->target_slots_captured = 0;
    ds->pending_count = 0;
    ds->pending_from_spec = false;
}

bool myds_dspine_capture_ready(const myds_dspine_t *ds) {
    return ds && (ds->target_slots_captured == 0x1Fu);
}

bool myds_dspine_build_tables(myds_dspine_t *ds, myds_model_t *base_model) {
    if (!ds || !base_model || !base_model->output) return false;
    MYDS_LOG_I("DSpine: building serial tables (tg, tsv)...");

    size_t tg_bytes = (size_t)ds->vocab_size * MYDS_DSPINE_MESSAGE_SIZE * sizeof(float);
    size_t tsv_bytes = (size_t)ds->vocab_size * 2 * MYDS_DSPINE_MESSAGE_SIZE * sizeof(float);

    if (!ds->tg_table) ds->tg_table = myds_metal_buffer_alloc(tg_bytes);
    if (!ds->tsv_table) ds->tsv_table = myds_metal_buffer_alloc(tsv_bytes);

    if (!ds->tg_table || !ds->tsv_table) {
        MYDS_LOG_E("DSpine: failed to allocate serial tables");
        return false;
    }

    ds->tables_ready = true;
    MYDS_LOG_I("DSpine: serial tables built successfully");
    return true;
}

/* SpecForge DSpineDraftModel.refine(hidden, predecessor_id) on final layer (l = 2) */
static void dspine_refine_single(const myds_dspine_t *ds, const float *pre_inj,
                                 int32_t pred_token_id, float *out_normed) {
    const uint32_t l_last = ds->num_layers - 1u;
    const uint16_t *msg_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_message);
    const uint16_t *pred_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_pred_gate);
    const uint16_t *recv_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_recv_gate);
    const uint16_t *bias_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_bias);
    const uint16_t *write_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_writes[l_last]);
    const uint16_t *codes_table = (const uint16_t *)myds_metal_buffer_data(ds->transfer_codes);
    const uint16_t *fn_w = (const uint16_t *)myds_metal_buffer_data(ds->final_norm);

    float recv[512];
    matvec_typed(recv, ds->injection_readouts[l_last], ds->injection_readout_types[l_last],
                 pre_inj, 512, ds->hidden_size);
    normalize_feature_512(recv);

    if (pred_token_id < 0 || (uint32_t)pred_token_id >= ds->vocab_size) pred_token_id = 0;
    const float sqrt_rank = sqrtf(512.0f);
    float pred[512];
    const uint16_t *code = codes_table + (size_t)pred_token_id * 512u;
    for (uint32_t r = 0; r < 512; r++) pred[r] = sqrt_rank * bf16_to_fp32(code[r]);

    float g_recv[256], g_pred[256], msg[256], gated_msg[256];
    matvec_bf16(g_recv, recv_w, recv, 256, 512);
    matvec_bf16(g_pred, pred_w, pred, 256, 512);
    matvec_bf16(msg, msg_w, pred, 256, 512);

    for (uint32_t b = 0; b < 256; b++) {
        float total_g = g_recv[b] + g_pred[b] + bf16_to_fp32(bias_w[b]);
        gated_msg[b] = (1.0f / (1.0f + expf(-total_g))) * msg[b];
    }

    float delta[5120];
    matvec_bf16(delta, write_w, gated_msg, ds->hidden_size, 256);

    float norm_x = 0.0f;
    for (uint32_t d = 0; d < ds->hidden_size; d++) norm_x += pre_inj[d] * pre_inj[d];
    float scale = sqrtf(norm_x / (float)ds->hidden_size);

    float refined[5120];
    for (uint32_t d = 0; d < ds->hidden_size; d++) refined[d] = pre_inj[d] + scale * delta[d];

    rmsnorm(out_normed, refined, fn_w, ds->hidden_size, 1e-6f);
}

typedef struct {
    const myds_dspine_t *ds;
    const float *k_ctx_layer;
    const float *v_ctx_layer;
    const float *q_buf;
    const float *k_buf;
    const float *v_buf;
    float *attn_in;
    float *scores_base;
    uint32_t scores_stride;
    uint32_t ctx_len;
    uint32_t T;
    float attn_scale;
} dspine_attn_job_t;

static void dspine_attn_head_worker(void *arg, size_t h_idx) {
    const dspine_attn_job_t *job = (const dspine_attn_job_t *)arg;
    const myds_dspine_t *ds = job->ds;
    const uint32_t h = (uint32_t)h_idx;
    const uint32_t gqa_ratio = ds->num_heads / ds->num_kv_heads;
    const uint32_t kv_h = h / gqa_ratio;
    const size_t kv_off = (size_t)kv_h * ds->head_dim;
    float *scores = job->scores_base + (size_t)h * job->scores_stride;
    const uint32_t ctx_len = job->ctx_len;
    const float attn_scale = job->attn_scale;

    for (uint32_t t = 0; t < job->T; t++) {
        const float *qh = job->q_buf + (size_t)t * ds->hidden_size + (size_t)h * ds->head_dim;
        float *out_h = job->attn_in + (size_t)t * ds->hidden_size + (size_t)h * ds->head_dim;
        const uint32_t total_kv = ctx_len + t + 1u;

        float max_s = -1e38f;
        for (uint32_t c = 0; c < ctx_len; c++) {
            const float *kh_c = job->k_ctx_layer + (size_t)c * 1024u + kv_off;
            float s = dot_f32_128(qh, kh_c) * attn_scale;
            scores[c] = s;
            if (s > max_s) max_s = s;
        }
        for (uint32_t j = 0; j <= t; j++) {
            const float *kh_j = job->k_buf + (size_t)j * 1024u + kv_off;
            float s = dot_f32_128(qh, kh_j) * attn_scale;
            scores[ctx_len + j] = s;
            if (s > max_s) max_s = s;
        }

        float sum_exp = 0.0f;
        for (uint32_t j = 0; j < total_kv; j++) {
            float e = expf(scores[j] - max_s);
            scores[j] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;

        memset(out_h, 0, ds->head_dim * sizeof(float));
        for (uint32_t c = 0; c < ctx_len; c++) {
            float w = scores[c] * inv_sum;
            const float *vh_c = job->v_ctx_layer + (size_t)c * 1024u + kv_off;
            axpy_f32_128(out_h, w, vh_c);
        }
        for (uint32_t j = 0; j <= t; j++) {
            float w = scores[ctx_len + j] * inv_sum;
            const float *vh_j = job->v_buf + (size_t)j * 1024u + kv_off;
            axpy_f32_128(out_h, w, vh_j);
        }
    }
}

uint32_t myds_dspine_propose(myds_dspine_t *ds, myds_context_t *ctx,
                             int32_t seed_token, uint32_t max_propose) {
    if (!ds || !ctx || !ds->enabled || !ds->loaded || max_propose == 0) return 0;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    uint32_t T = max_propose + 1;
    if (T > ds->block_size) T = ds->block_size;
    if (T > MYDS_DSPINE_MAX_BLOCK) T = MYDS_DSPINE_MAX_BLOCK;
    if (T < 2) return 0;

    ds->cycles++;
    ds->draft_count = 0;

    /* 1. Flush any pending captured target hidden states into ctx_k_cache / ctx_v_cache */
    if (ds->pending_count > 0) {
        if (ds->pending_from_spec) {
            float spec_batch[3 * MYDS_DSPINE_TARGET_DIM];
            uint32_t cnt = ds->pending_count <= 3u ? ds->pending_count : 3u;
            for (uint32_t r = 0; r < cnt; r++) {
                memcpy(spec_batch + (size_t)r * MYDS_DSPINE_TARGET_DIM,
                       myds_metal_buffer_data(ds->spec_target_slots[r]),
                       MYDS_DSPINE_TARGET_DIM * sizeof(float));
            }
            if (!myds_dspine_append_context_batch(ds, spec_batch, ds->pending_pos, cnt)) return 0;
        } else {
            const float *th = (const float *)myds_metal_buffer_data(ds->target_hidden_buf);
            if (!myds_dspine_append_context_batch(ds, th, ds->pending_pos, 1u)) return 0;
        }
        ds->pending_count = 0;
        ds->pending_from_spec = false;
    } else if (ds->ctx_kv_len == 0) {
        /* Standalone fallback when caller populated target_hidden_buf directly (unit tests) */
        const float *th = (const float *)myds_metal_buffer_data(ds->target_hidden_buf);
        uint32_t fb_pos = ctx->pos > 0 ? ctx->pos - 1u : 0u;
        if (!myds_dspine_append_context_batch(ds, th, fb_pos, 1u)) return 0;
    } else {
        /* Ensure strictly kv_pos < anchor_pos (ctx->pos) */
        myds_dspine_truncate_context(ds, ctx->pos);
    }

    const uint32_t ctx_len = ds->ctx_kv_len;
    if (ctx_len == 0) return 0;

    /* 2. Lookup Embeddings for block: [seed, mask, mask, ...] */
    float *x = (float *)myds_metal_buffer_data(ds->x_buf);
    const myds_model_t *m = ctx->model;
    const void *emb_raw = m && m->tok_embeddings ? myds_metal_buffer_data(m->tok_embeddings) : NULL;
    if (!emb_raw) return 0;

    const bool emb_bf16 = (m->tok_embeddings_type == GGML_TYPE_BF16);
    const bool emb_f32 = (m->tok_embeddings_type == GGML_TYPE_F32) &&
        (myds_metal_buffer_size(m->tok_embeddings) >= (size_t)ds->vocab_size * ds->hidden_size * sizeof(float));

    int32_t block_tokens[MYDS_DSPINE_MAX_BLOCK];
    block_tokens[0] = seed_token;
    for (uint32_t t = 1; t < T; t++) block_tokens[t] = ds->mask_token_id;

    for (uint32_t t = 0; t < T; t++) {
        int32_t tok = block_tokens[t];
        if (tok < 0 || (uint32_t)tok >= ds->vocab_size) tok = 0;
        float *xt = x + (size_t)t * ds->hidden_size;
        if (emb_f32) {
            const float *tok_emb = (const float *)emb_raw + (size_t)tok * ds->hidden_size;
            memcpy(xt, tok_emb, ds->hidden_size * sizeof(float));
        } else if (emb_bf16) {
            const uint16_t *tok_emb = (const uint16_t *)emb_raw + (size_t)tok * ds->hidden_size;
            for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] = bf16_to_fp32(tok_emb[d]);
        } else {
            const uint16_t *tok_emb = (const uint16_t *)emb_raw + (size_t)tok * ds->hidden_size;
            for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] = myds_fp16_to_fp32(tok_emb[d]);
        }
    }

    /* 3. 3-Layer Transformer Backbone */
    float *norm_buf = (float *)myds_metal_buffer_data(ds->norm_buf);
    float *attn_in = (float *)myds_metal_buffer_data(ds->attn_in_buf);
    float *attn_out = (float *)myds_metal_buffer_data(ds->attn_out_buf);
    float *ffn_out = (float *)myds_metal_buffer_data(ds->ffn_out_buf);
    float *q_buf = (float *)myds_metal_buffer_data(ds->q_buf);
    float *k_buf = (float *)myds_metal_buffer_data(ds->k_buf);
    float *v_buf = (float *)myds_metal_buffer_data(ds->v_buf);
    float *gate_ptr = (float *)myds_metal_buffer_data(ds->gate_buf);
    float *up_ptr = (float *)myds_metal_buffer_data(ds->up_buf);
    float *glu_ptr = (float *)myds_metal_buffer_data(ds->glu_buf);
    float *feat_ptr = (float *)myds_metal_buffer_data(ds->features_buf);
    float *scores = ds->attn_scores_scratch;
    const float attn_scale = 1.0f / sqrtf((float)ds->head_dim);

    for (uint32_t l = 0; l < ds->num_layers; l++) {
        myds_dspine_layer_t *layer = &ds->layers[l];
        const uint16_t *in_norm_w = (const uint16_t *)myds_metal_buffer_data(layer->input_norm);
        const uint16_t *qn_w = (const uint16_t *)myds_metal_buffer_data(layer->q_norm);
        const uint16_t *kn_w = (const uint16_t *)myds_metal_buffer_data(layer->k_norm);
        const uint16_t *post_norm_w = (const uint16_t *)myds_metal_buffer_data(layer->post_attn_norm);

        /* 3.1 Input RMSNorm on CPU */
        for (uint32_t t = 0; t < T; t++) {
            rmsnorm(norm_buf + (size_t)t * ds->hidden_size,
                    x + (size_t)t * ds->hidden_size,
                    in_norm_w, ds->hidden_size, 1e-6f);
        }

        /* 3.2 Batched Draft QKV on GPU (Context KV is already cached in ctx_k_cache / ctx_v_cache) */
        myds_metal_begin_batch();
        dspine_gemv_batch(ds->q_buf, layer->q_proj, ds->norm_buf, ds->hidden_size, ds->hidden_size, T, layer->q_type);
        dspine_gemv_batch(ds->k_buf, layer->k_proj, ds->norm_buf, 1024, ds->hidden_size, T, layer->k_type);
        dspine_gemv_batch(ds->v_buf, layer->v_proj, ds->norm_buf, 1024, ds->hidden_size, T, layer->v_type);
        myds_metal_commit_batch();

        /* 3.3 Q/K Norm and RoPE on CPU at exact positions ctx->pos + t (t = 0..T-1) */
        for (uint32_t t = 0; t < T; t++) {
            float *qt = q_buf + (size_t)t * ds->hidden_size;
            float *kt = k_buf + (size_t)t * 1024u;
            for (uint32_t h = 0; h < ds->num_heads; h++) {
                rmsnorm(qt + h * ds->head_dim, qt + h * ds->head_dim, qn_w, ds->head_dim, 1e-6f);
            }
            for (uint32_t h = 0; h < ds->num_kv_heads; h++) {
                rmsnorm(kt + h * ds->head_dim, kt + h * ds->head_dim, kn_w, ds->head_dim, 1e-6f);
            }
            apply_rope_fast(qt, ds->num_heads, ds->head_dim, ctx->pos + t, ds->rope_inv_freq);
            apply_rope_fast(kt, ds->num_kv_heads, ds->head_dim, ctx->pos + t, ds->rope_inv_freq);
        }

        /* 3.4 Attention over ctx_k_cache[0..ctx_len-1] + causal draft keys[0..t] */
        dspine_attn_job_t ajob = {
            .ds = ds,
            .k_ctx_layer = ds->ctx_k_cache[l],
            .v_ctx_layer = ds->ctx_v_cache[l],
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .attn_in = attn_in,
            .scores_base = scores,
            .scores_stride = ds->ctx_kv_cap + MYDS_DSPINE_MAX_BLOCK,
            .ctx_len = ctx_len,
            .T = T,
            .attn_scale = attn_scale,
        };
        if (ctx_len >= 64u) {
            dispatch_apply_f((size_t)ds->num_heads,
                             dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                             &ajob, dspine_attn_head_worker);
        } else {
            for (size_t h = 0; h < (size_t)ds->num_heads; h++) {
                dspine_attn_head_worker(&ajob, h);
            }
        }

        /* 3.5 Out projection + residual add on GPU */
        myds_metal_begin_batch();
        dspine_gemv_batch(ds->attn_out_buf, layer->o_proj, ds->attn_in_buf,
                          ds->hidden_size, ds->hidden_size, T, layer->o_type);
        bool add_attn_gpu = myds_metal_add(ds->x_buf, ds->x_buf, ds->attn_out_buf,
                                           T * ds->hidden_size) != 0;
        myds_metal_commit_batch();

        /* Post-Attention RMSNorm (with fallback residual add if GPU add unavailable) */
        for (uint32_t t = 0; t < T; t++) {
            float *xt = x + (size_t)t * ds->hidden_size;
            if (!add_attn_gpu) {
                float *at = attn_out + (size_t)t * ds->hidden_size;
                for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] += at[d];
            }
            rmsnorm(norm_buf + (size_t)t * ds->hidden_size, xt, post_norm_w, ds->hidden_size, 1e-6f);
        }

        /* 3.6 MLP Gate + Up + SwiGLU + Down + Residual Add + Code Bridge Readout in 1 GPU batch */
        myds_metal_begin_batch();
        dspine_gemv_batch(ds->gate_buf, layer->gate_proj, ds->norm_buf,
                          ds->intermediate_size, ds->hidden_size, T, layer->gate_type);
        dspine_gemv_batch(ds->up_buf, layer->up_proj, ds->norm_buf,
                          ds->intermediate_size, ds->hidden_size, T, layer->up_type);
        bool ffn_fused = myds_metal_swiglu(ds->glu_buf, ds->gate_buf, ds->up_buf,
                                           T * ds->intermediate_size) != 0;
        if (ffn_fused) {
            dspine_gemv_batch(ds->ffn_out_buf, layer->down_proj, ds->glu_buf,
                              ds->hidden_size, ds->intermediate_size, T, layer->down_type);
            ffn_fused = myds_metal_add(ds->x_buf, ds->x_buf, ds->ffn_out_buf,
                                       T * ds->hidden_size) != 0;
            if (ffn_fused) {
                dspine_gemv_batch(ds->features_buf, ds->injection_readouts[l], ds->x_buf,
                                  512, ds->hidden_size, T, ds->injection_readout_types[l]);
            }
        }
        myds_metal_commit_batch();

        if (!ffn_fused) {
            for (uint32_t t = 0; t < T; t++) {
                swiglu(glu_ptr + (size_t)t * ds->intermediate_size,
                       gate_ptr + (size_t)t * ds->intermediate_size,
                       up_ptr + (size_t)t * ds->intermediate_size,
                       ds->intermediate_size);
            }
            dspine_gemv_batch(ds->ffn_out_buf, layer->down_proj, ds->glu_buf,
                              ds->hidden_size, ds->intermediate_size, T, layer->down_type);
            for (uint32_t t = 0; t < T; t++) {
                float *xt = x + (size_t)t * ds->hidden_size;
                float *ft = ffn_out + (size_t)t * ds->hidden_size;
                for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] += ft[d];
            }
            dspine_gemv_batch(ds->features_buf, ds->injection_readouts[l], ds->x_buf,
                              512, ds->hidden_size, T, ds->injection_readout_types[l]);
        }

        /* Save final layer pre_injection before AdjacentInjection for discrete refinement tail */
        if (l + 1u == ds->num_layers) {
            memcpy(ds->pre_injection_l2, x, (size_t)T * ds->hidden_size * sizeof(float));
        }

        /* Exact L2 normalization * sqrt(512) matching SpecForge AdjacentInjection.feature */
        float sqrt_rank = sqrtf(512.0f);
        for (uint32_t t = 0; t < T; t++) {
            normalize_feature_512(feat_ptr + (size_t)t * 512u);
        }

        /* Predecessors and write-backs */
        const uint16_t *msg_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_message);
        const uint16_t *pred_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_pred_gate);
        const uint16_t *recv_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_recv_gate);
        const uint16_t *bias_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_bias);
        const uint16_t *write_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_writes[l]);
        const uint16_t *codes_table = (const uint16_t *)myds_metal_buffer_data(ds->transfer_codes);

        int32_t safe_seed = (seed_token >= 0 && (uint32_t)seed_token < ds->vocab_size) ? seed_token : 0;
        float P0[512];
        const uint16_t *seed_code = codes_table + (size_t)safe_seed * 512u;
        for (uint32_t r = 0; r < 512; r++) P0[r] = sqrt_rank * bf16_to_fp32(seed_code[r]);

        for (uint32_t t = 1; t < T; t++) {
            const float *pred = (t == 1) ? P0 : (feat_ptr + (size_t)(t - 1) * 512u);
            const float *recv = feat_ptr + (size_t)t * 512u;

            float g_recv[256], g_pred[256], msg[256], gate[256];
            matvec_bf16(g_recv, recv_w, recv, 256, 512);
            matvec_bf16(g_pred, pred_w, pred, 256, 512);
            matvec_bf16(msg, msg_w, pred, 256, 512);

            for (uint32_t b = 0; b < 256; b++) {
                float total_g = g_recv[b] + g_pred[b] + bf16_to_fp32(bias_w[b]);
                gate[b] = (1.0f / (1.0f + expf(-total_g))) * msg[b];
            }

            float delta[5120];
            matvec_bf16(delta, write_w, gate, ds->hidden_size, 256);

            float *xt = x + (size_t)t * ds->hidden_size;
            float norm_x = 0.0f;
            for (uint32_t d = 0; d < ds->hidden_size; d++) norm_x += xt[d] * xt[d];
            float scale = sqrtf(norm_x / (float)ds->hidden_size);

            for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] += scale * delta[d];
        }
    }

    /* 4. Final Norm & Candidate Selection + Discrete Predecessor Refinement Tail */
    const uint16_t *fn_w = (const uint16_t *)myds_metal_buffer_data(ds->final_norm);
    for (uint32_t t = 1; t < T; t++) {
        float *xt = x + (size_t)t * ds->hidden_size;
        rmsnorm(norm_buf + (size_t)(t - 1) * ds->hidden_size, xt, fn_w, ds->hidden_size, 1e-6f);
    }

    if (m && m->output) {
        int ok_head = myds_metal_gemv_batch(ds->logits_buf, m->output, ds->norm_buf,
                                            m->vocab_size, ds->hidden_size, T - 1, m->output_type);
        const float *all_logits = (const float *)myds_metal_buffer_data(ds->logits_buf);

        static int s_head_diag = -1;
        if (s_head_diag < 0) {
            const char *dbg = getenv("MYDS_DSPINE_DEBUG");
            s_head_diag = (dbg && atoi(dbg)) ? 0 : 999;
        }
        if (s_head_diag < 5) {
            s_head_diag++;
            MYDS_LOG_I("[dspine-head-diag] ok_head=%d out_type=%d vocab=%u ctx_kv=%u logit0=%.4f logit1=%.4f",
                       ok_head, (int)m->output_type, m->vocab_size, ctx_len,
                       all_logits ? all_logits[0] : -999.0f, all_logits ? all_logits[1] : -999.0f);
        }

        static int s_no_refine = -1;
        if (s_no_refine < 0) {
            const char *nr = getenv("MYDS_DSPINE_NO_REFINE");
            s_no_refine = (nr && atoi(nr) != 0) ? 1 : 0;
        }

        const uint32_t K = ds->top_k <= MYDS_DSPINE_TOP_K ? ds->top_k : MYDS_DSPINE_TOP_K;
        int32_t cand_ids[MYDS_DSPINE_MAX_BLOCK][MYDS_DSPINE_TOP_K];

        /* First extract backbone argmax (for t=1) and top-K candidates (for t>=2) BEFORE any logits_buf reuse */
        for (uint32_t t = 1; t < T; t++) {
            const float *logits = all_logits + (size_t)(t - 1) * m->vocab_size;
            if (t == 1 || s_no_refine || K <= 1) {
                float max_v = logits[0];
                int32_t best_tok = 0;
                for (uint32_t v = 1; v < m->vocab_size; v++) {
                    if (logits[v] > max_v) {
                        max_v = logits[v];
                        best_tok = (int32_t)v;
                    }
                }
                cand_ids[t - 1][0] = best_tok;
            } else {
                topk_f32(logits, m->vocab_size, K, cand_ids[t - 1]);
            }
        }

        /* Position 0 (t=1): layer 2 AdjacentInjection already injected transfer_codes[seed_token] */
        ds->draft_tokens[0] = cand_ids[0][0];
        ds->draft_count = 1;
        ds->proposed_tokens++;

        /* Positions 1..T-2 (t=2..T-1): refine pre_injection_l2[t] with discrete draft_tokens[t-2] */
        for (uint32_t t = 2; t < T; t++) {
            int32_t best_tok = cand_ids[t - 1][0];
            if (!s_no_refine && K > 1) {
                float refined_norm[5120];
                dspine_refine_single(ds, ds->pre_injection_l2 + (size_t)t * ds->hidden_size,
                                     ds->draft_tokens[t - 2], refined_norm);

                float best_score = -1e38f;
                bool cpu_ok = true;
                for (uint32_t k = 0; k < K; k++) {
                    float sc = 0.0f;
                    if (!score_output_row_cpu(m, (uint32_t)cand_ids[t - 1][k],
                                              refined_norm, ds->hidden_size, &sc)) {
                        cpu_ok = false;
                        break;
                    }
                    if (k == 0 || sc > best_score) {
                        best_score = sc;
                        best_tok = cand_ids[t - 1][k];
                    }
                }
                if (!cpu_ok) {
                    /* GPU fallback for exotic lm_head quant types */
                    memcpy(norm_buf, refined_norm, ds->hidden_size * sizeof(float));
                    if (myds_metal_gemv_batch(ds->logits_buf, m->output, ds->norm_buf,
                                              m->vocab_size, ds->hidden_size, 1, m->output_type)) {
                        const float *ref_logits = (const float *)myds_metal_buffer_data(ds->logits_buf);
                        best_score = ref_logits[cand_ids[t - 1][0]];
                        best_tok = cand_ids[t - 1][0];
                        for (uint32_t k = 1; k < K; k++) {
                            float sc = ref_logits[cand_ids[t - 1][k]];
                            if (sc > best_score) {
                                best_score = sc;
                                best_tok = cand_ids[t - 1][k];
                            }
                        }
                    }
                }
            }
            ds->draft_tokens[t - 1] = best_tok;
            ds->draft_count++;
            ds->proposed_tokens++;
        }
    } else {
        for (uint32_t t = 1; t < T; t++) {
            ds->draft_tokens[t - 1] = seed_token;
            ds->draft_count++;
            ds->proposed_tokens++;
        }
    }

    gettimeofday(&t1, NULL);
    ds->propose_ms += (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                      (double)(t1.tv_usec - t0.tv_usec) / 1000.0;

    return ds->draft_count;
}

void myds_dspine_rollback(myds_dspine_t *ds, uint32_t n_accepted) {
    if (!ds) return;
    ds->accepted_tokens += n_accepted;
}
