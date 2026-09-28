#include "myds.h"
#include "myds_metal.h"
#include "myds_gguf.h"
#include "myds_tokenizer.h"
#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <dispatch/dispatch.h>

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} engine_block_q8_0_t;

static inline float fp16_to_fp32(uint16_t h) {
#if defined(__arm64__) || defined(__aarch64__)
    return (float)(*(_Float16 *)&h);
#else
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    int32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) {
            float zero;
            memcpy(&zero, &sign, sizeof(float));
            return zero;
        }
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp += (127 - 15);
    }
    uint32_t f = sign | ((uint32_t)exp << 23) | (mant << 13);
    float val;
    memcpy(&val, &f, sizeof(float));
    return val;
#endif
}

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float dot_f32(const float *a, const float *b, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
#else
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
#endif
}

static inline void axpy_f32(float *y, const float *x, float a, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    const float32x4_t av = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     av, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), av, vld1q_f32(x + i + 4)));
    }
    for (; i < n; i++) y[i] += a * x[i];
#else
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
#endif
}

static inline void scale_f32(float *x, float a, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    const float32x4_t av = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(x + i,     vmulq_f32(vld1q_f32(x + i),     av));
        vst1q_f32(x + i + 4, vmulq_f32(vld1q_f32(x + i + 4), av));
    }
    for (; i < n; i++) x[i] *= a;
#else
    for (uint32_t i = 0; i < n; i++) x[i] *= a;
#endif
}


static void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (size_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];

        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

static void rope_tail_inplace(
        float    *x,
        uint32_t  n_head,
        uint32_t  head_dim,
        uint32_t  n_rot,
        uint32_t  pos,
        float     freq_base,
        float     freq_scale,
        bool      inverse) {
    if (n_rot == 0 || n_rot > head_dim) return;
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;

    for (uint32_t h = 0; h < n_head; h++) {
        float *tail = x + (size_t)h * head_dim + n_nope;
        float theta_extrap = (float)pos;

        for (uint32_t i = 0; i < n_rot; i += 2) {
            const float theta = freq_scale * theta_extrap;
            const float c = cosf(theta);
            const float s = sin_sign * sinf(theta);
            const float x0 = tail[i + 0];
            const float x1 = tail[i + 1];

            tail[i + 0] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;

            theta_extrap *= theta_scale;
        }
    }
}

typedef struct {
    float             *out_heads;
    const float       *sinks;
    const float       *q;
    const float       *kv_rows;
    uint32_t           n_kv;
    uint32_t           head_dim;
    float              kq_scale;
} attn_sinks_job_t;

static void layer_attention_sinks_head_worker(void *context, size_t h) {
    const attn_sinks_job_t *job = (const attn_sinks_job_t *)context;
    uint32_t head_dim = job->head_dim;
    uint32_t n_kv = job->n_kv;
    const float *qh = job->q + h * head_dim;

    float score_stack[4096];
    float *score = (n_kv <= 4096) ? score_stack : (float *)malloc((size_t)n_kv * sizeof(float));
    if (!score) return;

    float max_score = job->sinks ? job->sinks[h] : -INFINITY;
    for (uint32_t r = 0; r < n_kv; r++) {
        const float *kv = job->kv_rows + (size_t)r * head_dim;
        score[r] = dot_f32(qh, kv, head_dim) * job->kq_scale;
        if (score[r] > max_score) max_score = score[r];
    }

    float *oh = job->out_heads + h * head_dim;
    memset(oh, 0, (size_t)head_dim * sizeof(float));

    float denom = job->sinks ? expf(job->sinks[h] - max_score) : 0.0f;
    for (uint32_t r = 0; r < n_kv; r++) {
        const float weight = expf(score[r] - max_score);
        const float *kv = job->kv_rows + (size_t)r * head_dim;
        denom += weight;
        axpy_f32(oh, kv, weight, head_dim);
    }

    const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    scale_f32(oh, inv, head_dim);

    if (score != score_stack) free(score);
}

static void layer_attention_sinks(
        float             *out_heads,
        const float       *sinks,
        const float       *q,
        const float       *kv_rows,
        uint32_t           n_kv,
        uint32_t           n_heads,
        uint32_t           head_dim) {
    const float kq_scale = 1.0f / sqrtf((float)head_dim);
    attn_sinks_job_t job = {
        .out_heads = out_heads,
        .sinks = sinks,
        .q = q,
        .kv_rows = kv_rows,
        .n_kv = n_kv,
        .head_dim = head_dim,
        .kq_scale = kq_scale,
    };
    dispatch_queue_t q_concurrent = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    dispatch_apply_f((size_t)n_heads, q_concurrent, &job, layer_attention_sinks_head_worker);
}

#include "myds_internal.h"

static __attribute__((unused)) void matvec_grouped_rows(
        float                     *out,
        const myds_metal_buffer_t *W,
        const float               *x,
        uint32_t                   n_groups,
        uint64_t                   group_dim,
        uint64_t                   rank,
        ggml_type                  type) {
    if (!out || !W || !x || n_groups == 0 || group_dim == 0 || rank == 0) return;

    uint64_t total_rows = (uint64_t)n_groups * rank;

    if (type == GGML_TYPE_Q8_0) {
        const engine_block_q8_0_t *blocks = (const engine_block_q8_0_t *)myds_metal_buffer_data((myds_metal_buffer_t *)W);
        uint64_t blocks_per_row = group_dim / 32;

        for (uint64_t r = 0; r < total_rows; r++) {
            uint32_t group = (uint32_t)(r / rank);
            const float *xg = x + (size_t)group * group_dim;
            const engine_block_q8_0_t *row_blocks = blocks + r * blocks_per_row;

            float sum = 0.0f;
            for (uint64_t b = 0; b < blocks_per_row; b++) {
                float d = fp16_to_fp32(row_blocks[b].d);
                const int8_t *qs = row_blocks[b].qs;
                const float *xb = xg + b * 32;
                for (int k = 0; k < 32; k++) {
                    sum += d * (float)qs[k] * xb[k];
                }
            }
            out[r] = sum;
        }
    } else if (type == GGML_TYPE_F32) {
        const float *w_f32 = (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)W);
        for (uint64_t r = 0; r < total_rows; r++) {
            uint32_t group = (uint32_t)(r / rank);
            const float *xg = x + (size_t)group * group_dim;
            const float *row = w_f32 + r * group_dim;
            out[r] = dot_f32(row, xg, (uint32_t)group_dim);
        }
    } else if (type == GGML_TYPE_F16) {
        const uint16_t *w_f16 = (const uint16_t *)myds_metal_buffer_data((myds_metal_buffer_t *)W);
        for (uint64_t r = 0; r < total_rows; r++) {
            uint32_t group = (uint32_t)(r / rank);
            const float *xg = x + (size_t)group * group_dim;
            const uint16_t *row = w_f16 + r * group_dim;
            float sum = 0.0f;
            for (uint64_t i = 0; i < group_dim; i++) {
                sum += fp16_to_fp32(row[i]) * xg[i];
            }
            out[r] = sum;
        }
    }
}

/* Helper: verify supported quantization types for GEMV operations */
static bool is_supported_quant_type(ggml_type type) {
    return (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0);
}

/* Helper: verify supported quantization types for MoE gate & up expert projections */
static bool is_supported_moe_gate_up_quant(ggml_type type) {
    return (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F32 || type == GGML_TYPE_F16);
}

/* Helper: verify supported quantization types for MoE down expert projections */
static bool is_supported_moe_down_quant(ggml_type type) {
    return (type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F32 || type == GGML_TYPE_F16);
}

/* Helper: wrap GGUF tensor memory into Metal buffer using zero-copy mapping */
static myds_metal_buffer_t *create_metal_buffer_from_tensor(const gguf_tensor_t *t) {
    if (!t || !t->data) return NULL;
    myds_metal_buffer_t *buf = myds_metal_buffer_from_ptr(t->data, t->size_bytes);
    if (!buf) {
        /* Fallback if no-copy wrapper cannot be created */
        buf = myds_metal_buffer_alloc(t->size_bytes);
        if (buf) {
            myds_metal_buffer_write(buf, 0, t->data, t->size_bytes);
        }
    }
    return buf;
}

myds_model_t *myds_model_load(const char *model_path) {
    if (!myds_metal_init("metal")) {
        MYDS_LOG_E("Failed to initialize Metal backend (check Apple Silicon GPU & shaders)");
        return NULL;
    }

    gguf_file_t *gf = gguf_open(model_path);
    if (!gf) {
        MYDS_LOG_E("Failed to open GGUF file '%s'", model_path);
        return NULL;
    }

    MYDS_LOG_I("Opened GGUF model: %s (tensors: %llu, kv_pairs: %llu)",
               model_path, (unsigned long long)gf->tensor_count, (unsigned long long)gf->metadata_kv_count);

    myds_model_t *m = (myds_model_t *)calloc(1, sizeof(myds_model_t));
    m->gf = gf;
    m->tokenizer = myds_tokenizer_load(gf);

    const char *arch = gguf_get_str(gf, "general.architecture", "deepseek41");

    /* Validate architecture: exclusively supports DeepSeek V4 / V4.1 (V2/V3 removed) */
    if (!myds_is_supported_arch(arch)) {
        if (strcmp(arch, "deepseek2") == 0 || strcmp(arch, "deepseek3") == 0 ||
            strcmp(arch, "deepseek") == 0 || strcmp(arch, "deepseek-v2") == 0 ||
            strcmp(arch, "deepseek-v3") == 0) {
            MYDS_LOG_E("Architecture '%s' rejected: DeepSeek V2 and V3 support has been removed. myds exclusively supports DeepSeek V4 / V4.1.", arch);
        } else {
            MYDS_LOG_E("Architecture '%s' rejected: myds exclusively supports DeepSeek V4 / V4.1 (expected 'deepseek41' or 'deepseek4*').", arch);
        }
        if (m->tokenizer) myds_tokenizer_free(m->tokenizer);
        free(m);
        gguf_close(gf);
        return NULL;
    }

    strncpy(m->arch, arch, sizeof(m->arch) - 1);

    char key_buf[128];
    snprintf(key_buf, sizeof(key_buf), "%s.embedding_length", arch);
    m->dim = gguf_get_u32(gf, key_buf, 0);
    if (m->dim == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.hidden_size", arch);
        m->dim = gguf_get_u32(gf, key_buf, 2048);
    }

    snprintf(key_buf, sizeof(key_buf), "%s.block_count", arch);
    m->n_layers = gguf_get_u32(gf, key_buf, 0);
    if (m->n_layers == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.num_hidden_layers", arch);
        m->n_layers = gguf_get_u32(gf, key_buf, 28);
    }

    snprintf(key_buf, sizeof(key_buf), "%s.attention.head_count", arch);
    m->n_heads = gguf_get_u32(gf, key_buf, 0);
    if (m->n_heads == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.num_attention_heads", arch);
        m->n_heads = gguf_get_u32(gf, key_buf, 16);
    }

    snprintf(key_buf, sizeof(key_buf), "%s.attention.head_count_kv", arch);
    m->n_kv_heads = gguf_get_u32(gf, key_buf, 0);
    if (m->n_kv_heads == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.num_key_value_heads", arch);
        m->n_kv_heads = gguf_get_u32(gf, key_buf, m->n_heads);
    }

    if (m->n_heads == 0 || m->dim == 0) {
        MYDS_LOG_E("Invalid model dimensions: dim=%u, n_heads=%u", m->dim, m->n_heads);
        goto fail;
    }

    snprintf(key_buf, sizeof(key_buf), "%s.head_dim", arch);
    uint32_t explicit_head_dim = gguf_get_u32(gf, key_buf, 0);
    if (explicit_head_dim > 0) {
        m->head_dim = explicit_head_dim;
    } else {
        m->head_dim = m->dim / m->n_heads;
    }

    snprintf(key_buf, sizeof(key_buf), "%s.feed_forward_length", arch);
    m->hidden_dim = gguf_get_u32(gf, key_buf, m->dim * 4);

    snprintf(key_buf, sizeof(key_buf), "%s.attention.layer_norm_rms_epsilon", arch);
    m->norm_eps = gguf_get_f32(gf, key_buf, 1e-6f);

    snprintf(key_buf, sizeof(key_buf), "%s.rope.freq_base", arch);
    m->freq_base = gguf_get_f32(gf, key_buf, 10000.0f);

    snprintf(key_buf, sizeof(key_buf), "%s.rope.scale_linear", arch);
    m->freq_scale = gguf_get_f32(gf, key_buf, 1.0f);

    /* MoE parameters: DeepSeek V4.1 uses .n_routed_experts, .num_experts_per_tok, .n_shared_experts */
    snprintf(key_buf, sizeof(key_buf), "%s.n_routed_experts", arch);
    m->n_routed_experts = gguf_get_u32(gf, key_buf, 0);
    if (m->n_routed_experts == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.expert_count", arch);
        m->n_routed_experts = gguf_get_u32(gf, key_buf, 0);
    }

    snprintf(key_buf, sizeof(key_buf), "%s.num_experts_per_tok", arch);
    m->n_activated_experts = gguf_get_u32(gf, key_buf, 0);
    if (m->n_activated_experts == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.expert_used_count", arch);
        m->n_activated_experts = gguf_get_u32(gf, key_buf, 0);
    }

    snprintf(key_buf, sizeof(key_buf), "%s.n_shared_experts", arch);
    m->n_shared_experts = gguf_get_u32(gf, key_buf, 0);
    if (m->n_shared_experts == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.expert_shared_count", arch);
        m->n_shared_experts = gguf_get_u32(gf, key_buf, 0);
    }

    snprintf(key_buf, sizeof(key_buf), "%s.routed_scaling_factor", arch);
    m->routed_scaling_factor = gguf_get_f32(gf, key_buf, 0.0f);
    if (m->routed_scaling_factor == 0.0f) {
        snprintf(key_buf, sizeof(key_buf), "%s.expert_weights_scale", arch);
        m->routed_scaling_factor = gguf_get_f32(gf, key_buf, 1.0f);
    }

    snprintf(key_buf, sizeof(key_buf), "%s.moe_intermediate_size", arch);
    m->moe_expert_dim = gguf_get_u32(gf, key_buf, 2304);

    snprintf(key_buf, sizeof(key_buf), "%s.swiglu_limit", arch);
    m->swiglu_clamp_exp = gguf_get_f32(gf, key_buf, 10.0f);

    /* Detect routed experts presence from tensors if not set via KV */
    const gguf_tensor_t *t_routed_exps = gguf_find_tensor(gf, "blk.0.ffn_gate_exps.weight");
    if (t_routed_exps != NULL) {
        if (m->n_routed_experts == 0 && t_routed_exps->n_dims >= 3) {
            m->n_routed_experts = (uint32_t)t_routed_exps->dims[2];
        }
        if (m->n_activated_experts == 0) {
            m->n_activated_experts = 6;
        }
        if (m->moe_expert_dim == 0 && t_routed_exps->n_dims >= 2) {
            m->moe_expert_dim = (uint32_t)t_routed_exps->dims[1];
        }
    }

    if (m->n_routed_experts > 0) {
        MYDS_LOG_I("MoE enabled: %u routed experts (top-%u active, scale=%.2f, exp_dim=%u, clamp=%.1f), %u shared experts",
                   m->n_routed_experts, m->n_activated_experts, m->routed_scaling_factor,
                   m->moe_expert_dim, m->swiglu_clamp_exp, m->n_shared_experts);
    }

    m->vocab_size = m->tokenizer ? m->tokenizer->vocab_size : 32000;

    /* Check whether model uses MLA (Multi-Head Latent Attention) */
    const gguf_tensor_t *t_mla_sample = gguf_find_tensor(gf, "blk.0.attn_q_a.weight");
    if (!t_mla_sample) t_mla_sample = gguf_find_tensor(gf, "blk.0.attn_kv.weight");
    m->is_mla = (t_mla_sample != NULL);

    if (m->is_mla) {
        /* Default MLA parameters for DeepSeek V4 / V4.1 */
        m->q_lora_rank = 1280;
        m->kv_lora_rank = 512;
        m->qk_rope_head_dim = 64;
        m->n_out_groups = 8;
        m->out_group_rank = 1024;

        const gguf_tensor_t *t_q_norm = gguf_find_tensor(gf, "blk.0.attn_q_a_norm.weight");
        if (t_q_norm && t_q_norm->dims[0] > 0) m->q_lora_rank = (uint32_t)t_q_norm->dims[0];

        const gguf_tensor_t *t_kv_norm = gguf_find_tensor(gf, "blk.0.attn_kv_a_norm.weight");
        if (t_kv_norm && t_kv_norm->dims[0] > 0) m->kv_lora_rank = (uint32_t)t_kv_norm->dims[0];

        const gguf_tensor_t *t_sinks = gguf_find_tensor(gf, "blk.0.attn_sinks.weight");
        if (t_sinks && t_sinks->dims[0] > 0) m->n_heads = (uint32_t)t_sinks->dims[0];

        m->qk_rope_head_dim = gguf_get_u32(gf, "deepseek41.qk_rope_head_dim", 64);

        m->head_dim = m->kv_lora_rank;
        if (m->qk_rope_head_dim > m->head_dim) m->qk_rope_head_dim = m->head_dim;

        const gguf_tensor_t *t_out_a = gguf_find_tensor(gf, "blk.0.attn_output_a.weight");
        if (t_out_a && t_out_a->dims[0] > 0) {
            uint64_t gdim = t_out_a->dims[0];
            uint64_t total_r = t_out_a->dims[1];
            if (gdim <= (uint64_t)m->n_heads * m->head_dim) {
                m->n_out_groups = (uint32_t)((m->n_heads * m->head_dim) / gdim);
                if (m->n_out_groups == 0) m->n_out_groups = 1;
                m->out_group_rank = (uint32_t)(total_r / m->n_out_groups);
            }
        }
    }

    /* Load Embedding */
    const gguf_tensor_t *t_embd = gguf_find_tensor(gf, "token_embd.weight");
    if (!t_embd) {
        MYDS_LOG_E("Missing required embedding tensor 'token_embd.weight' (T0-3 gate)");
        goto fail;
    }
    if (t_embd->type != GGML_TYPE_F32 && t_embd->type != GGML_TYPE_F16) {
        MYDS_LOG_E("Embedding tensor 'token_embd.weight' has unsupported type %d (%s). Expected F32 or F16 (T0-1 gate).",
                   (int)t_embd->type, ggml_type_name(t_embd->type));
        goto fail;
    }
    m->tok_embeddings = create_metal_buffer_from_tensor(t_embd);
    m->tok_embeddings_type = t_embd->type;
    if (!m->tok_embeddings) {
        MYDS_LOG_E("Failed to create buffer for 'token_embd.weight'");
        goto fail;
    }
    if (t_embd->n_dims >= 2 && t_embd->dims[1] > 0) {
        m->vocab_size = (uint32_t)t_embd->dims[1];
    }

    /* Load Output Norm & Output (LM Head) */
    const gguf_tensor_t *t_out_norm = gguf_find_tensor(gf, "output_norm.weight");
    if (!t_out_norm) {
        MYDS_LOG_E("Missing required norm tensor 'output_norm.weight' (T0-3 gate)");
        goto fail;
    }
    m->output_norm = create_metal_buffer_from_tensor(t_out_norm);
    if (!m->output_norm) {
        MYDS_LOG_E("Failed to create buffer for 'output_norm.weight'");
        goto fail;
    }

    const gguf_tensor_t *t_out = gguf_find_tensor(gf, "output.weight");
    if (!t_out) {
        /* Weight tying with embedding */
        m->output = m->tok_embeddings;
        m->output_type = m->tok_embeddings_type;
    } else {
        if (!is_supported_quant_type(t_out->type)) {
            MYDS_LOG_E("Output projection tensor 'output.weight' has unsupported quant type %d (%s). Expected F32, F16, or Q8_0 (T0-1 gate).",
                       (int)t_out->type, ggml_type_name(t_out->type));
            goto fail;
        }
        m->output = create_metal_buffer_from_tensor(t_out);
        m->output_type = t_out->type;
        if (!m->output) {
            MYDS_LOG_E("Failed to create buffer for 'output.weight'");
            goto fail;
        }
        if (t_out->n_dims >= 2 && t_out->dims[1] > 0) {
            m->vocab_size = (uint32_t)t_out->dims[1];
        }
    }

    /* Load Layers */
    m->layers = (myds_layer_t *)calloc(m->n_layers, sizeof(myds_layer_t));
    if (!m->layers) {
        MYDS_LOG_E("Failed to allocate memory for %u layers", m->n_layers);
        goto fail;
    }

    for (uint32_t i = 0; i < m->n_layers; i++) {
        char name[128];

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        const gguf_tensor_t *tan = gguf_find_tensor(gf, name);
        if (!tan) {
            MYDS_LOG_E("Layer %u missing required attention norm tensor '%s' (T0-3 gate)", i, name);
            goto fail;
        }
        m->layers[i].attn_norm = create_metal_buffer_from_tensor(tan);

        if (m->is_mla) {
            snprintf(name, sizeof(name), "blk.%u.attn_q_a.weight", i);
            const gguf_tensor_t *tqa = gguf_find_tensor(gf, name);
            if (!tqa) {
                MYDS_LOG_E("Layer %u missing required MLA attention Q_A tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(tqa->type)) {
                MYDS_LOG_E("Layer %u attn_q_a tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)tqa->type, ggml_type_name(tqa->type));
                goto fail;
            }
            m->layers[i].attn_q_a = create_metal_buffer_from_tensor(tqa);
            m->layers[i].attn_q_a_type = tqa->type;

            snprintf(name, sizeof(name), "blk.%u.attn_q_a_norm.weight", i);
            const gguf_tensor_t *tqan = gguf_find_tensor(gf, name);
            if (!tqan) {
                MYDS_LOG_E("Layer %u missing required MLA attention Q_A norm tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            m->layers[i].attn_q_a_norm = create_metal_buffer_from_tensor(tqan);

            snprintf(name, sizeof(name), "blk.%u.attn_q_b.weight", i);
            const gguf_tensor_t *tqb = gguf_find_tensor(gf, name);
            if (!tqb) {
                MYDS_LOG_E("Layer %u missing required MLA attention Q_B tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(tqb->type)) {
                MYDS_LOG_E("Layer %u attn_q_b tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)tqb->type, ggml_type_name(tqb->type));
                goto fail;
            }
            m->layers[i].attn_q_b = create_metal_buffer_from_tensor(tqb);
            m->layers[i].attn_q_b_type = tqb->type;

            snprintf(name, sizeof(name), "blk.%u.attn_kv.weight", i);
            const gguf_tensor_t *tkv = gguf_find_tensor(gf, name);
            if (!tkv) {
                MYDS_LOG_E("Layer %u missing required MLA attention KV tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(tkv->type)) {
                MYDS_LOG_E("Layer %u attn_kv tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)tkv->type, ggml_type_name(tkv->type));
                goto fail;
            }
            m->layers[i].attn_kv = create_metal_buffer_from_tensor(tkv);
            m->layers[i].attn_kv_type = tkv->type;

            snprintf(name, sizeof(name), "blk.%u.attn_kv_a_norm.weight", i);
            const gguf_tensor_t *tkvan = gguf_find_tensor(gf, name);
            if (!tkvan) {
                MYDS_LOG_E("Layer %u missing required MLA attention KV_A norm tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            m->layers[i].attn_kv_a_norm = create_metal_buffer_from_tensor(tkvan);

            snprintf(name, sizeof(name), "blk.%u.attn_sinks.weight", i);
            const gguf_tensor_t *tsinks = gguf_find_tensor(gf, name);
            if (tsinks) {
                m->layers[i].attn_sinks = create_metal_buffer_from_tensor(tsinks);
            }

            snprintf(name, sizeof(name), "blk.%u.attn_output_a.weight", i);
            const gguf_tensor_t *toa = gguf_find_tensor(gf, name);
            if (!toa) {
                MYDS_LOG_E("Layer %u missing required MLA attention output_a tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(toa->type)) {
                MYDS_LOG_E("Layer %u attn_output_a tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)toa->type, ggml_type_name(toa->type));
                goto fail;
            }
            m->layers[i].attn_output_a = create_metal_buffer_from_tensor(toa);
            m->layers[i].attn_output_a_type = toa->type;

            snprintf(name, sizeof(name), "blk.%u.attn_output_b.weight", i);
            const gguf_tensor_t *tob = gguf_find_tensor(gf, name);
            if (!tob) {
                MYDS_LOG_E("Layer %u missing required MLA attention output_b tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(tob->type)) {
                MYDS_LOG_E("Layer %u attn_output_b tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)tob->type, ggml_type_name(tob->type));
                goto fail;
            }
            m->layers[i].attn_output_b = create_metal_buffer_from_tensor(tob);
            m->layers[i].attn_output_b_type = tob->type;
        } else {
            snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
            const gguf_tensor_t *tq = gguf_find_tensor(gf, name);
            if (!tq) {
                MYDS_LOG_E("Layer %u missing required attention Q tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(tq->type)) {
                MYDS_LOG_E("Layer %u attention Q tensor '%s' has unsupported quant type %d (%s). Expected F32, F16, or Q8_0 (T0-1 gate).",
                           i, name, (int)tq->type, ggml_type_name(tq->type));
                goto fail;
            }
            m->layers[i].attn_q = create_metal_buffer_from_tensor(tq);
            m->layers[i].attn_q_type = tq->type;

            snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
            const gguf_tensor_t *tk = gguf_find_tensor(gf, name);
            if (!tk) {
                MYDS_LOG_E("Layer %u missing required attention K tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(tk->type)) {
                MYDS_LOG_E("Layer %u attention K tensor '%s' has unsupported quant type %d (%s). Expected F32, F16, or Q8_0 (T0-1 gate).",
                           i, name, (int)tk->type, ggml_type_name(tk->type));
                goto fail;
            }
            m->layers[i].attn_k = create_metal_buffer_from_tensor(tk);
            m->layers[i].attn_k_type = tk->type;

            snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
            const gguf_tensor_t *tv = gguf_find_tensor(gf, name);
            if (!tv) {
                MYDS_LOG_E("Layer %u missing required attention V tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(tv->type)) {
                MYDS_LOG_E("Layer %u attention V tensor '%s' has unsupported quant type %d (%s). Expected F32, F16, or Q8_0 (T0-1 gate).",
                           i, name, (int)tv->type, ggml_type_name(tv->type));
                goto fail;
            }
            m->layers[i].attn_v = create_metal_buffer_from_tensor(tv);
            m->layers[i].attn_v_type = tv->type;

            snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
            const gguf_tensor_t *to = gguf_find_tensor(gf, name);
            if (!to) {
                MYDS_LOG_E("Layer %u missing required attention output tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_quant_type(to->type)) {
                MYDS_LOG_E("Layer %u attention output tensor '%s' has unsupported quant type %d (%s). Expected F32, F16, or Q8_0 (T0-1 gate).",
                           i, name, (int)to->type, ggml_type_name(to->type));
                goto fail;
            }
            m->layers[i].attn_o = create_metal_buffer_from_tensor(to);
            m->layers[i].attn_o_type = to->type;
        }

        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", i);
        const gguf_tensor_t *tfn = gguf_find_tensor(gf, name);
        if (!tfn) {
            MYDS_LOG_E("Layer %u missing required FFN norm tensor '%s' (T0-3 gate)", i, name);
            goto fail;
        }
        m->layers[i].ffn_norm = create_metal_buffer_from_tensor(tfn);

        /* FFN: Check for routed MoE experts, dense FFN, and shared experts */
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", i);
        const gguf_tensor_t *t_ge = gguf_find_tensor(gf, name);
        if (t_ge) {
            /* Routed MoE gate experts (IQ2_XXS / Q8_0) */
            if (!is_supported_moe_gate_up_quant(t_ge->type)) {
                MYDS_LOG_E("Layer %u ffn_gate_exps tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)t_ge->type, ggml_type_name(t_ge->type));
                goto fail;
            }
            m->layers[i].ffn_gate_exps = create_metal_buffer_from_tensor(t_ge);
            m->layers[i].ffn_gate_exps_type = t_ge->type;

            /* Routed MoE up experts (IQ2_XXS / Q8_0) */
            snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", i);
            const gguf_tensor_t *t_ue = gguf_find_tensor(gf, name);
            if (!t_ue) {
                MYDS_LOG_E("Layer %u missing required MoE ffn_up_exps tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_moe_gate_up_quant(t_ue->type)) {
                MYDS_LOG_E("Layer %u ffn_up_exps tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)t_ue->type, ggml_type_name(t_ue->type));
                goto fail;
            }
            m->layers[i].ffn_up_exps = create_metal_buffer_from_tensor(t_ue);
            m->layers[i].ffn_up_exps_type = t_ue->type;

            /* Routed MoE down experts (Q2_K / Q8_0) */
            snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", i);
            const gguf_tensor_t *t_de = gguf_find_tensor(gf, name);
            if (!t_de) {
                MYDS_LOG_E("Layer %u missing required MoE ffn_down_exps tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            if (!is_supported_moe_down_quant(t_de->type)) {
                MYDS_LOG_E("Layer %u ffn_down_exps tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)t_de->type, ggml_type_name(t_de->type));
                goto fail;
            }
            m->layers[i].ffn_down_exps = create_metal_buffer_from_tensor(t_de);
            m->layers[i].ffn_down_exps_type = t_de->type;

            /* Router gate projection (ffn_gate_inp) */
            snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", i);
            const gguf_tensor_t *t_rg = gguf_find_tensor(gf, name);
            if (!t_rg) {
                MYDS_LOG_E("Layer %u missing required MoE router gate tensor '%s' (T0-3 gate)", i, name);
                goto fail;
            }
            m->layers[i].router_gate = create_metal_buffer_from_tensor(t_rg);
            m->layers[i].router_gate_type = t_rg->type;

            /* Router bias (exp_probs_b.bias) */
            snprintf(name, sizeof(name), "blk.%u.exp_probs_b.bias", i);
            const gguf_tensor_t *t_rb = gguf_find_tensor(gf, name);
            if (t_rb) {
                m->layers[i].router_bias = create_metal_buffer_from_tensor(t_rb);
            }
        } else {
            /* Dense FFN */
            snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", i);
            const gguf_tensor_t *fg = gguf_find_tensor(gf, name);
            if (fg) {
                if (!is_supported_quant_type(fg->type)) {
                    MYDS_LOG_E("Layer %u ffn_gate tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                               i, name, (int)fg->type, ggml_type_name(fg->type));
                    goto fail;
                }
                m->layers[i].ffn_gate = create_metal_buffer_from_tensor(fg);
                m->layers[i].ffn_gate_type = fg->type;

                snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", i);
                const gguf_tensor_t *fu = gguf_find_tensor(gf, name);
                if (!fu || !is_supported_quant_type(fu->type)) {
                    MYDS_LOG_E("Layer %u missing or invalid ffn_up (T0-3 / T0-1 gate)", i);
                    goto fail;
                }
                m->layers[i].ffn_up = create_metal_buffer_from_tensor(fu);
                m->layers[i].ffn_up_type = fu->type;

                snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", i);
                const gguf_tensor_t *fd = gguf_find_tensor(gf, name);
                if (!fd || !is_supported_quant_type(fd->type)) {
                    MYDS_LOG_E("Layer %u missing or invalid ffn_down (T0-3 / T0-1 gate)", i);
                    goto fail;
                }
                m->layers[i].ffn_down = create_metal_buffer_from_tensor(fd);
                m->layers[i].ffn_down_type = fd->type;
            }
        }

        /* Shared expert FFN */
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", i);
        const gguf_tensor_t *t_sg = gguf_find_tensor(gf, name);
        if (t_sg) {
            if (!is_supported_quant_type(t_sg->type)) {
                MYDS_LOG_E("Layer %u shared gate tensor '%s' has unsupported quant type %d (%s) (T0-1 gate)",
                           i, name, (int)t_sg->type, ggml_type_name(t_sg->type));
                goto fail;
            }
            m->layers[i].shared_gate = create_metal_buffer_from_tensor(t_sg);
            m->layers[i].shared_gate_type = t_sg->type;

            snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", i);
            const gguf_tensor_t *t_su = gguf_find_tensor(gf, name);
            if (!t_su || !is_supported_quant_type(t_su->type)) {
                MYDS_LOG_E("Layer %u missing or invalid shared up tensor '%s' (T0-3 / T0-1 gate)", i, name);
                goto fail;
            }
            m->layers[i].shared_up = create_metal_buffer_from_tensor(t_su);
            m->layers[i].shared_up_type = t_su->type;

            snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", i);
            const gguf_tensor_t *t_sd = gguf_find_tensor(gf, name);
            if (!t_sd || !is_supported_quant_type(t_sd->type)) {
                MYDS_LOG_E("Layer %u missing or invalid shared down tensor '%s' (T0-3 / T0-1 gate)", i, name);
                goto fail;
            }
            m->layers[i].shared_down = create_metal_buffer_from_tensor(t_sd);
            m->layers[i].shared_down_type = t_sd->type;
        }

        bool has_dense_ffn = (m->layers[i].ffn_gate && m->layers[i].ffn_up && m->layers[i].ffn_down);
        bool has_routed_ffn = (m->layers[i].ffn_gate_exps && m->layers[i].ffn_up_exps && m->layers[i].ffn_down_exps && m->layers[i].router_gate);
        bool has_shared_ffn = (m->layers[i].shared_gate && m->layers[i].shared_up && m->layers[i].shared_down);
        if (!has_dense_ffn && !has_routed_ffn && !has_shared_ffn) {
            MYDS_LOG_E("Layer %u missing FFN tensors (neither complete dense FFN, routed MoE, nor shared expert found) (T0-3 gate)", i);
            goto fail;
        }
    }

    /* Pre-warm resident weights (embeddings, output, attention, shared experts) */
    myds_metal_warm_buffer(m->tok_embeddings);
    myds_metal_warm_buffer(m->output_norm);
    if (m->output && m->output != m->tok_embeddings) myds_metal_warm_buffer(m->output);

    for (uint32_t i = 0; i < m->n_layers; i++) {
        myds_metal_warm_buffer(m->layers[i].attn_norm);
        myds_metal_warm_buffer(m->layers[i].attn_q_a);
        myds_metal_warm_buffer(m->layers[i].attn_q_a_norm);
        myds_metal_warm_buffer(m->layers[i].attn_q_b);
        myds_metal_warm_buffer(m->layers[i].attn_kv);
        myds_metal_warm_buffer(m->layers[i].attn_kv_a_norm);
        myds_metal_warm_buffer(m->layers[i].attn_sinks);
        myds_metal_warm_buffer(m->layers[i].attn_output_a);
        myds_metal_warm_buffer(m->layers[i].attn_output_b);
        myds_metal_warm_buffer(m->layers[i].router_gate);
        myds_metal_warm_buffer(m->layers[i].router_bias);
        myds_metal_warm_buffer(m->layers[i].shared_gate);
        myds_metal_warm_buffer(m->layers[i].shared_up);
        myds_metal_warm_buffer(m->layers[i].shared_down);
        myds_metal_warm_buffer(m->layers[i].ffn_norm);

        /* Hint XNU kernel to asynchronously read-ahead MoE routed expert weights */
        if (m->layers[i].ffn_gate_exps) {
            void *p = myds_metal_buffer_data(m->layers[i].ffn_gate_exps);
            size_t sz = myds_metal_buffer_size(m->layers[i].ffn_gate_exps);
            if (p && sz > 0) posix_madvise(p, sz, POSIX_MADV_WILLNEED);
        }
        if (m->layers[i].ffn_up_exps) {
            void *p = myds_metal_buffer_data(m->layers[i].ffn_up_exps);
            size_t sz = myds_metal_buffer_size(m->layers[i].ffn_up_exps);
            if (p && sz > 0) posix_madvise(p, sz, POSIX_MADV_WILLNEED);
        }
        if (m->layers[i].ffn_down_exps) {
            void *p = myds_metal_buffer_data(m->layers[i].ffn_down_exps);
            size_t sz = myds_metal_buffer_size(m->layers[i].ffn_down_exps);
            if (p && sz > 0) posix_madvise(p, sz, POSIX_MADV_WILLNEED);
        }
    }

    return m;

fail:
    if (m) {
        myds_model_free(m);
    } else if (gf) {
        gguf_close(gf);
    }
    return NULL;
}

void myds_model_free(myds_model_t *m) {
    if (!m) return;
    if (m->tok_embeddings) myds_metal_buffer_free(m->tok_embeddings);
    if (m->output_norm)    myds_metal_buffer_free(m->output_norm);
    if (m->output && m->output != m->tok_embeddings) myds_metal_buffer_free(m->output);

    if (m->layers) {
        for (uint32_t i = 0; i < m->n_layers; i++) {
            if (m->layers[i].attn_norm)      myds_metal_buffer_free(m->layers[i].attn_norm);
            if (m->layers[i].attn_q)         myds_metal_buffer_free(m->layers[i].attn_q);
            if (m->layers[i].attn_k)         myds_metal_buffer_free(m->layers[i].attn_k);
            if (m->layers[i].attn_v)         myds_metal_buffer_free(m->layers[i].attn_v);
            if (m->layers[i].attn_o)         myds_metal_buffer_free(m->layers[i].attn_o);
            if (m->layers[i].attn_q_a)       myds_metal_buffer_free(m->layers[i].attn_q_a);
            if (m->layers[i].attn_q_a_norm)  myds_metal_buffer_free(m->layers[i].attn_q_a_norm);
            if (m->layers[i].attn_q_b)       myds_metal_buffer_free(m->layers[i].attn_q_b);
            if (m->layers[i].attn_kv)        myds_metal_buffer_free(m->layers[i].attn_kv);
            if (m->layers[i].attn_kv_a_norm) myds_metal_buffer_free(m->layers[i].attn_kv_a_norm);
            if (m->layers[i].attn_sinks)     myds_metal_buffer_free(m->layers[i].attn_sinks);
            if (m->layers[i].attn_output_a)  myds_metal_buffer_free(m->layers[i].attn_output_a);
            if (m->layers[i].attn_output_b)  myds_metal_buffer_free(m->layers[i].attn_output_b);
            if (m->layers[i].ffn_norm)       myds_metal_buffer_free(m->layers[i].ffn_norm);
            if (m->layers[i].ffn_gate)       myds_metal_buffer_free(m->layers[i].ffn_gate);
            if (m->layers[i].ffn_up)         myds_metal_buffer_free(m->layers[i].ffn_up);
            if (m->layers[i].ffn_down)       myds_metal_buffer_free(m->layers[i].ffn_down);
            if (m->layers[i].router_gate)    myds_metal_buffer_free(m->layers[i].router_gate);
            if (m->layers[i].router_bias)    myds_metal_buffer_free(m->layers[i].router_bias);
            if (m->layers[i].ffn_gate_exps)  myds_metal_buffer_free(m->layers[i].ffn_gate_exps);
            if (m->layers[i].ffn_up_exps)    myds_metal_buffer_free(m->layers[i].ffn_up_exps);
            if (m->layers[i].ffn_down_exps)  myds_metal_buffer_free(m->layers[i].ffn_down_exps);
            if (m->layers[i].shared_gate)    myds_metal_buffer_free(m->layers[i].shared_gate);
            if (m->layers[i].shared_up)      myds_metal_buffer_free(m->layers[i].shared_up);
            if (m->layers[i].shared_down)    myds_metal_buffer_free(m->layers[i].shared_down);
        }
        free(m->layers);
    }

    if (m->tokenizer) myds_tokenizer_free(m->tokenizer);
    if (m->gf)        gguf_close(m->gf);
    free(m);
}

void myds_model_print_info(const myds_model_t *m) {
    if (!m) return;
    printf("====================================================\n");
    printf(" DeepSeek V4 / V4.1 Model Information\n");
    printf("====================================================\n");
    printf(" Architecture : %s\n", m->arch);
    printf(" Layers       : %u\n", m->n_layers);
    printf(" Hidden Dim   : %u\n", m->dim);
    if (m->is_mla) {
        printf(" Attention    : MLA (%u heads, head_dim=%u, q_rank=%u, kv_rank=%u, rope_dim=%u)\n",
               m->n_heads, m->head_dim, m->q_lora_rank, m->kv_lora_rank, m->qk_rope_head_dim);
    } else {
        printf(" Attention    : %u heads (%u KV), head_dim=%u\n", m->n_heads, m->n_kv_heads, m->head_dim);
    }
    printf(" FFN Dim      : %u\n", m->hidden_dim);
    printf(" Vocab Size   : %u\n", m->vocab_size);
    if (m->n_routed_experts > 0) {
        printf(" MoE Routed   : %u experts (top-%u active), scale=%.2f\n",
               m->n_routed_experts, m->n_activated_experts, m->routed_scaling_factor);
        printf(" MoE Shared   : %u experts\n", m->n_shared_experts);
    } else {
        printf(" MoE          : Dense model\n");
    }
    if (m->gf) {
        mach_task_basic_info_data_t task_info_data;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        double resident_gb = 0.0;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&task_info_data, &count) == KERN_SUCCESS) {
            resident_gb = (double)task_info_data.resident_size / (1024.0 * 1024.0 * 1024.0);
        }
        double file_gb = (double)m->gf->mmap_size / (1024.0 * 1024.0 * 1024.0);
        printf(" Memory (mmap): %.2f GB mapped, %.2f GB resident (strategy: demand paged)\n", file_gb, resident_gb);
    }
    printf("====================================================\n");
}

/* Production Sampler: greedy, temperature, top-k, min-p, top-p, repetition penalty */
typedef struct {
    float prob;
    int32_t id;
} myds_cand_t;

static int compare_cand_desc(const void *a, const void *b) {
    const myds_cand_t *ca = (const myds_cand_t *)a;
    const myds_cand_t *cb = (const myds_cand_t *)b;
    if (ca->prob < cb->prob) return 1;
    if (ca->prob > cb->prob) return -1;
    return 0;
}

static void quickselect_cand_desc(myds_cand_t *arr, int l, int r, int k) {
    while (l < r) {
        int i = l, j = r;
        float pivot = arr[l + (r - l) / 2].prob;
        while (i <= j) {
            while (arr[i].prob > pivot) i++;
            while (arr[j].prob < pivot) j--;
            if (i <= j) {
                myds_cand_t tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
                i++;
                j--;
            }
        }
        if (k <= j) {
            r = j;
        } else if (k >= i) {
            l = i;
        } else {
            break;
        }
    }
}

myds_context_t *myds_context_create(myds_model_t *model, const myds_config_t *cfg) {
    if (!model) return NULL;

    myds_context_t *ctx = (myds_context_t *)calloc(1, sizeof(myds_context_t));
    ctx->model = model;
    if (cfg) {
        ctx->cfg = *cfg;
    } else {
        ctx->cfg.max_seq_len = 4096;
        ctx->cfg.max_new_tokens = 256;
        ctx->cfg.temperature = 0.7f;
        ctx->cfg.top_p = 0.9f;
        ctx->cfg.top_k = 40;
        ctx->cfg.seed = (int32_t)time(NULL);
        ctx->cfg.stream = true;
    }
    if (ctx->cfg.max_seq_len == 0) ctx->cfg.max_seq_len = 4096;

    srand(ctx->cfg.seed);

    uint32_t dim = model->dim;
    uint32_t hidden_dim = model->hidden_dim;
    uint32_t vocab_size = model->vocab_size;

    ctx->x            = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->x_norm       = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->attn_out     = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->ffn_norm_out = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->gate_out     = myds_metal_buffer_alloc(hidden_dim * sizeof(float));
    ctx->up_out       = myds_metal_buffer_alloc(hidden_dim * sizeof(float));
    ctx->swiglu_out   = myds_metal_buffer_alloc(hidden_dim * sizeof(float));
    ctx->ffn_out      = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->logits       = myds_metal_buffer_alloc(vocab_size * sizeof(float));

    if (model->is_mla) {
        ctx->qr         = myds_metal_buffer_alloc(model->q_lora_rank * sizeof(float));
        ctx->qr_norm    = myds_metal_buffer_alloc(model->q_lora_rank * sizeof(float));
        ctx->q          = myds_metal_buffer_alloc(model->n_heads * model->head_dim * sizeof(float));
        ctx->kv_act     = myds_metal_buffer_alloc(model->head_dim * sizeof(float));
        ctx->attn_heads = myds_metal_buffer_alloc(model->n_heads * model->head_dim * sizeof(float));
        ctx->attn_low   = myds_metal_buffer_alloc((size_t)model->n_out_groups * model->out_group_rank * sizeof(float));

        /* In MLA, latent KV cache stores only 1 latent vector of width head_dim per token */
        ctx->k_cache = (float **)calloc(model->n_layers, sizeof(float *));
        ctx->v_cache = NULL;
        size_t kv_layer_size = (size_t)ctx->cfg.max_seq_len * model->head_dim;
        for (uint32_t l = 0; l < model->n_layers; l++) {
            ctx->k_cache[l] = (float *)calloc(kv_layer_size, sizeof(float));
        }
    } else {
        ctx->q            = myds_metal_buffer_alloc(model->n_heads * model->head_dim * sizeof(float));
        ctx->k            = myds_metal_buffer_alloc(model->n_kv_heads * model->head_dim * sizeof(float));
        ctx->v            = myds_metal_buffer_alloc(model->n_kv_heads * model->head_dim * sizeof(float));
        ctx->attn_scores  = myds_metal_buffer_alloc(ctx->cfg.max_seq_len * sizeof(float));

        ctx->k_cache = (float **)calloc(model->n_layers, sizeof(float *));
        ctx->v_cache = (float **)calloc(model->n_layers, sizeof(float *));
        size_t kv_layer_size = (size_t)ctx->cfg.max_seq_len * model->n_kv_heads * model->head_dim;
        for (uint32_t l = 0; l < model->n_layers; l++) {
            ctx->k_cache[l] = (float *)calloc(kv_layer_size, sizeof(float));
            ctx->v_cache[l] = (float *)calloc(kv_layer_size, sizeof(float));
        }
    }

    if (model->n_routed_experts > 0) {
        uint32_t n_routed = model->n_routed_experts;
        uint32_t n_active = model->n_activated_experts;
        uint32_t exp_dim = model->moe_expert_dim > 0 ? model->moe_expert_dim : 2304;

        ctx->router_logits    = myds_metal_buffer_alloc(n_routed * sizeof(float));
        ctx->selected_ids     = myds_metal_buffer_alloc(n_active * sizeof(int32_t));
        ctx->selected_weights = myds_metal_buffer_alloc(n_active * sizeof(float));
        ctx->moe_gate         = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        ctx->moe_up           = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        ctx->moe_mid          = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        ctx->moe_down         = myds_metal_buffer_alloc(n_active * dim * sizeof(float));
        ctx->moe_out          = myds_metal_buffer_alloc(dim * sizeof(float));

        if (model->n_shared_experts > 0) {
            ctx->shared_gate_out   = myds_metal_buffer_alloc(exp_dim * sizeof(float));
            ctx->shared_up_out     = myds_metal_buffer_alloc(exp_dim * sizeof(float));
            ctx->shared_swiglu_out = myds_metal_buffer_alloc(exp_dim * sizeof(float));
            ctx->shared_out        = myds_metal_buffer_alloc(dim * sizeof(float));
        }
    }

    ctx->sample_probs = (float *)malloc(vocab_size * sizeof(float));
    ctx->sample_indices = (int32_t *)malloc(vocab_size * sizeof(int32_t));
    ctx->sample_cands = malloc(vocab_size * sizeof(myds_cand_t));
    ctx->rng_state = ((uint64_t)ctx->cfg.seed ^ 0x5DEECE66DULL) | 1ULL;

    ctx->pos = 0;
    return ctx;
}

void myds_context_free(myds_context_t *ctx) {
    if (!ctx) return;
    if (ctx->x)            myds_metal_buffer_free(ctx->x);
    if (ctx->x_norm)       myds_metal_buffer_free(ctx->x_norm);
    if (ctx->q)            myds_metal_buffer_free(ctx->q);
    if (ctx->k)            myds_metal_buffer_free(ctx->k);
    if (ctx->v)            myds_metal_buffer_free(ctx->v);
    if (ctx->qr)           myds_metal_buffer_free(ctx->qr);
    if (ctx->qr_norm)      myds_metal_buffer_free(ctx->qr_norm);
    if (ctx->kv_act)       myds_metal_buffer_free(ctx->kv_act);
    if (ctx->attn_heads)   myds_metal_buffer_free(ctx->attn_heads);
    if (ctx->attn_low)     myds_metal_buffer_free(ctx->attn_low);
    if (ctx->attn_scores)  myds_metal_buffer_free(ctx->attn_scores);
    if (ctx->attn_out)     myds_metal_buffer_free(ctx->attn_out);
    if (ctx->ffn_norm_out) myds_metal_buffer_free(ctx->ffn_norm_out);
    if (ctx->gate_out)     myds_metal_buffer_free(ctx->gate_out);
    if (ctx->up_out)       myds_metal_buffer_free(ctx->up_out);
    if (ctx->swiglu_out)   myds_metal_buffer_free(ctx->swiglu_out);
    if (ctx->ffn_out)      myds_metal_buffer_free(ctx->ffn_out);
    if (ctx->logits)       myds_metal_buffer_free(ctx->logits);

    if (ctx->router_logits)    myds_metal_buffer_free(ctx->router_logits);
    if (ctx->selected_ids)     myds_metal_buffer_free(ctx->selected_ids);
    if (ctx->selected_weights) myds_metal_buffer_free(ctx->selected_weights);
    if (ctx->moe_gate)         myds_metal_buffer_free(ctx->moe_gate);
    if (ctx->moe_up)           myds_metal_buffer_free(ctx->moe_up);
    if (ctx->moe_mid)          myds_metal_buffer_free(ctx->moe_mid);
    if (ctx->moe_down)         myds_metal_buffer_free(ctx->moe_down);
    if (ctx->moe_out)          myds_metal_buffer_free(ctx->moe_out);
    if (ctx->shared_gate_out)   myds_metal_buffer_free(ctx->shared_gate_out);
    if (ctx->shared_up_out)     myds_metal_buffer_free(ctx->shared_up_out);
    if (ctx->shared_swiglu_out) myds_metal_buffer_free(ctx->shared_swiglu_out);
    if (ctx->shared_out)        myds_metal_buffer_free(ctx->shared_out);

    if (ctx->k_cache) {
        for (uint32_t l = 0; l < ctx->model->n_layers; l++) free(ctx->k_cache[l]);
        free(ctx->k_cache);
    }
    if (ctx->v_cache) {
        for (uint32_t l = 0; l < ctx->model->n_layers; l++) free(ctx->v_cache[l]);
        free(ctx->v_cache);
    }
    if (ctx->sample_probs)   free(ctx->sample_probs);
    if (ctx->sample_indices) free(ctx->sample_indices);
    if (ctx->sample_cands)   free(ctx->sample_cands);
    free(ctx);
}

myds_config_t myds_context_get_config(const myds_context_t *ctx) {
    if (!ctx) {
        myds_config_t empty = {0};
        return empty;
    }
    return ctx->cfg;
}

void myds_context_set_config(myds_context_t *ctx, const myds_config_t *cfg) {
    if (!ctx || !cfg) return;
    ctx->cfg = *cfg;
}

int32_t myds_sample_next_token(
    const float         *logits,
    uint32_t             vocab_size,
    const myds_config_t *cfg,
    myds_context_t      *ctx,
    const int32_t       *prev_tokens,
    size_t               n_prev_tokens) {

    if (!logits || vocab_size == 0) return 0;

    float temp = cfg ? cfg->temperature : 0.0f;
    float top_p = cfg ? cfg->top_p : 0.9f;
    uint32_t top_k = cfg ? cfg->top_k : 0;
    float min_p = cfg ? cfg->min_p : 0.0f;
    float rep_pen = (cfg && cfg->repetition_penalty > 0.0f) ? cfg->repetition_penalty : 1.0f;

    /* 1. Greedy argmax check (if temp <= 0) */
    if (temp <= 0.0f) {
        int32_t best_id = 0;
        float max_l = logits[0];
        if (rep_pen > 1.0f && prev_tokens && n_prev_tokens > 0) {
            for (size_t p = 0; p < n_prev_tokens; p++) {
                if (prev_tokens[p] == 0) {
                    max_l = (max_l > 0.0f) ? (max_l / rep_pen) : (max_l * rep_pen);
                    break;
                }
            }
        }
        for (uint32_t i = 1; i < vocab_size; i++) {
            float l = logits[i];
            if (rep_pen > 1.0f && prev_tokens && n_prev_tokens > 0) {
                for (size_t p = 0; p < n_prev_tokens; p++) {
                    if (prev_tokens[p] == (int32_t)i) {
                        l = (l > 0.0f) ? (l / rep_pen) : (l * rep_pen);
                        break;
                    }
                }
            }
            if (l > max_l) {
                max_l = l;
                best_id = (int32_t)i;
            }
        }
        return best_id;
    }

    /* 2. Repetition penalty check & max logit search in single linear pass */
    float max_l = -INFINITY;
    bool has_rep_pen = (rep_pen > 1.0f && prev_tokens && n_prev_tokens > 0);
    for (uint32_t i = 0; i < vocab_size; i++) {
        float l = logits[i];
        if (has_rep_pen) {
            for (size_t p = 0; p < n_prev_tokens; p++) {
                if (prev_tokens[p] == (int32_t)i) {
                    l = (l > 0.0f) ? (l / rep_pen) : (l * rep_pen);
                    break;
                }
            }
        }
        if (l > max_l) max_l = l;
    }

    /* 3. Buffer acquisition and single-pass candidate collection (diff >= -20.0f) */
    float *probs = (ctx && ctx->sample_probs) ? ctx->sample_probs : (float *)malloc(vocab_size * sizeof(float));
    int32_t *indices = (ctx && ctx->sample_indices) ? ctx->sample_indices : (int32_t *)malloc(vocab_size * sizeof(int32_t));

    float inv_temp = 1.0f / temp;
    float sum_exp = 0.0f;
    uint32_t n_cands = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        float l = logits[i];
        if (has_rep_pen) {
            for (size_t p = 0; p < n_prev_tokens; p++) {
                if (prev_tokens[p] == (int32_t)i) {
                    l = (l > 0.0f) ? (l / rep_pen) : (l * rep_pen);
                    break;
                }
            }
        }
        float diff = (l - max_l) * inv_temp;
        if (diff >= -20.0f) {
            float e = expf(diff);
            probs[n_cands] = e;
            indices[n_cands] = (int32_t)i;
            sum_exp += e;
            n_cands++;
        }
    }

    if (n_cands == 0) {
        if (!ctx || !ctx->sample_probs) { free(probs); free(indices); }
        return 0;
    }

    /* 4. Normalize candidate probabilities */
    float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
    float max_prob = 0.0f;
    for (uint32_t i = 0; i < n_cands; i++) {
        probs[i] *= inv_sum;
        if (probs[i] > max_prob) max_prob = probs[i];
    }

    /* 5. Min-P filtering on candidate pool */
    if (min_p > 0.0f && min_p <= 1.0f) {
        float min_thresh = max_prob * min_p;
        uint32_t kept = 0;
        for (uint32_t i = 0; i < n_cands; i++) {
            if (probs[i] >= min_thresh) {
                probs[kept] = probs[i];
                indices[kept] = indices[i];
                kept++;
            }
        }
        n_cands = kept;
        if (n_cands == 0) {
            if (!ctx || !ctx->sample_probs) { free(probs); free(indices); }
            return 0;
        }
    }

    myds_cand_t stack_cands[1024];
    myds_cand_t *cands = (ctx && ctx->sample_cands) ? (myds_cand_t *)ctx->sample_cands :
                         (n_cands <= 1024) ? stack_cands : (myds_cand_t *)malloc(n_cands * sizeof(myds_cand_t));
    for (uint32_t i = 0; i < n_cands; i++) {
        cands[i].prob = probs[i];
        cands[i].id = indices[i];
    }

    /* Apply top_k truncation with O(N) quickselect before qsort */
    uint32_t target_k = top_k;
    if (target_k == 0 && top_p > 0.0f && top_p < 1.0f && n_cands > 1024) {
        target_k = 1024;
    }
    if (target_k > 0 && target_k < n_cands) {
        quickselect_cand_desc(cands, 0, (int)n_cands - 1, (int)target_k - 1);
        n_cands = target_k;
    }
    qsort(cands, n_cands, sizeof(myds_cand_t), compare_cand_desc);

    /* Apply top_p truncation */
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        uint32_t keep = 0;
        for (uint32_t i = 0; i < n_cands; i++) {
            cum += cands[i].prob;
            keep++;
            if (cum >= top_p) break;
        }
        n_cands = keep;
    }

    /* Re-normalize remaining candidates */
    float norm_sum = 0.0f;
    for (uint32_t i = 0; i < n_cands; i++) norm_sum += cands[i].prob;
    float inv_norm = 1.0f / (norm_sum > 0.0f ? norm_sum : 1.0f);
    for (uint32_t i = 0; i < n_cands; i++) cands[i].prob *= inv_norm;

    /* 7. Sample using per-context RNG or fallback */
    float r;
    if (ctx) {
        ctx->rng_state = ctx->rng_state * 6364136223846793005ULL + 1ULL;
        r = (float)((ctx->rng_state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    } else {
        r = (float)rand() / (float)RAND_MAX;
    }

    float cum = 0.0f;
    int32_t sampled_id = cands[0].id;
    for (uint32_t i = 0; i < n_cands; i++) {
        cum += cands[i].prob;
        if (cum >= r) {
            sampled_id = cands[i].id;
            break;
        }
    }

    if ((!ctx || !ctx->sample_cands) && cands != stack_cands) {
        free(cands);
    }
    if (!ctx || !ctx->sample_probs) {
        free(probs);
        free(indices);
    }
    return sampled_id;
}

static int execute_ffn_layer(
    myds_context_t *ctx,
    myds_layer_t   *layer,
    uint32_t        l,
    uint32_t        dim,
    uint32_t        hidden_dim) {
    myds_model_t *m = ctx->model;
    struct timeval tf0, tf1, tf2;
    memset(&tf0, 0, sizeof(tf0));
    memset(&tf1, 0, sizeof(tf1));
    memset(&tf2, 0, sizeof(tf2));
    bool do_ffn_prof = (l == 0 && getenv("MYDS_PROF") != NULL);
    if (do_ffn_prof) gettimeofday(&tf0, NULL);

    if (layer->ffn_gate_exps && layer->ffn_up_exps && layer->ffn_down_exps && layer->router_gate) {
        uint32_t n_routed = m->n_routed_experts;
        uint32_t n_active = m->n_activated_experts;
        uint32_t exp_dim = m->moe_expert_dim > 0 ? m->moe_expert_dim : 2304;

        /* Enqueue router_gate in Batch 2 (takes ffn_norm_out as input) */
        if (!myds_metal_gemv(ctx->router_logits, layer->router_gate, ctx->ffn_norm_out, n_routed, dim, layer->router_gate_type)) {
            MYDS_LOG_E("Forward failed: GEMV for router_gate at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }

        /* Signal router_event immediately after router_gate is queued */
        uint64_t router_event = 0;
        if (!myds_metal_signal_event(&router_event)) {
            MYDS_LOG_E("Forward failed: signal router_event at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }

        /* In the SAME command buffer, enqueue the entire shared expert: gate, up, swiglu, and down */
        uint64_t shexp_event = 0;
        if (layer->shared_gate && layer->shared_up && layer->shared_down && ctx->shared_out) {
            if (!myds_metal_gemv(ctx->shared_gate_out, layer->shared_gate, ctx->ffn_norm_out, exp_dim, dim, layer->shared_gate_type) ||
                !myds_metal_gemv(ctx->shared_up_out, layer->shared_up, ctx->ffn_norm_out, exp_dim, dim, layer->shared_up_type)) {
                MYDS_LOG_E("Forward failed: GEMV for shared_gate/up at layer %u failed", l);
                myds_metal_commit_batch();
                return 0;
            }
            myds_metal_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out, ctx->shared_up_out, exp_dim);
            if (!myds_metal_gemv(ctx->shared_out, layer->shared_down, ctx->shared_swiglu_out, dim, exp_dim, layer->shared_down_type)) {
                MYDS_LOG_E("Forward failed: GEMV for shared_down at layer %u failed", l);
                myds_metal_commit_batch();
                return 0;
            }
            myds_metal_signal_event(&shexp_event);
        }

        /* Commit Batch 2 and wait ONLY for router_event (~8 us GEMV math), eliminating full GPU drain bubble */
        if (!myds_metal_commit_and_wait_event(router_event)) {
            MYDS_LOG_E("Forward failed: commit and wait for router_event at layer %u failed", l);
            return 0;
        }
        if (do_ffn_prof) gettimeofday(&tf1, NULL);

        int32_t *sel_ids = (int32_t *)myds_metal_buffer_data(ctx->selected_ids);
        float *sel_weights = (float *)myds_metal_buffer_data(ctx->selected_weights);
        if (!myds_metal_moe_route_biased(
                sel_ids,
                sel_weights,
                ctx->router_logits,
                layer->router_bias,
                n_routed,
                n_active,
                m->routed_scaling_factor)) {
            MYDS_LOG_E("Forward failed: MoE biased routing failed at layer %u", l);
            return 0;
        }
        if (do_ffn_prof) gettimeofday(&tf2, NULL);

        /* Begin Batch 3: routed MoE + combine with shared expert */
        myds_metal_begin_batch();
        if (shexp_event > 0) {
            myds_metal_wait_event(shexp_event);
        }
        if (!myds_metal_routed_moe(
                ctx->moe_out,
                ctx->ffn_norm_out,
                layer->ffn_gate_exps,
                layer->ffn_up_exps,
                layer->ffn_down_exps,
                ctx->selected_ids,
                ctx->selected_weights,
                ctx->moe_gate,
                ctx->moe_up,
                ctx->moe_mid,
                ctx->moe_down,
                dim,
                exp_dim,
                n_routed,
                n_active,
                layer->ffn_gate_exps_type,
                layer->ffn_up_exps_type,
                layer->ffn_down_exps_type,
                m->swiglu_clamp_exp)) {
            MYDS_LOG_E("Forward failed: routed MoE execution failed at layer %u", l);
            myds_metal_commit_batch();
            return 0;
        }

        /* Combine shared expert output with routed expert output */
        if (layer->shared_gate && layer->shared_up && layer->shared_down && ctx->shared_out) {
            myds_metal_add(ctx->moe_out, ctx->moe_out, ctx->shared_out, dim);
        }

        /* Add MoE output to residual */
        myds_metal_add(ctx->x, ctx->x, ctx->moe_out, dim);
        /* Inter-layer batch fusion: do not commit here. Keep batch open so that
         * layer l FFN residual add and layer l+1 pre-attention RMSNorm + QKV projections
         * execute in the same Metal command buffer without CPU roundtrip. */
        if (do_ffn_prof) {
            myds_metal_commit_batch();
            struct timeval tf3;
            gettimeofday(&tf3, NULL);
            double ms_gate = (tf1.tv_sec - tf0.tv_sec)*1000.0 + (tf1.tv_usec - tf0.tv_usec)/1000.0;
            double ms_route = (tf2.tv_sec - tf1.tv_sec)*1000.0 + (tf2.tv_usec - tf1.tv_usec)/1000.0;
            double ms_moe_batch = (tf3.tv_sec - tf2.tv_sec)*1000.0 + (tf3.tv_usec - tf2.tv_usec)/1000.0;
            double ms_ffn_tot = (tf3.tv_sec - tf0.tv_sec)*1000.0 + (tf3.tv_usec - tf0.tv_usec)/1000.0;
            printf("[PROF FFN L0] router_gate=%.2f ms, moe_route=%.2f ms, moe_batch=%.2f ms | FFN_TOTAL=%.2f ms\n",
                   ms_gate, ms_route, ms_moe_batch, ms_ffn_tot);
            fflush(stdout);
        }
        return 1;
    } else if (layer->ffn_gate && layer->ffn_up && layer->ffn_down) {
        if (!myds_metal_gemv(ctx->gate_out, layer->ffn_gate, ctx->ffn_norm_out, hidden_dim, dim, layer->ffn_gate_type) ||
            !myds_metal_gemv(ctx->up_out, layer->ffn_up, ctx->ffn_norm_out, hidden_dim, dim, layer->ffn_up_type)) {
            MYDS_LOG_E("Forward failed: GEMV for ffn_gate/up at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }
        myds_metal_swiglu(ctx->swiglu_out, ctx->gate_out, ctx->up_out, hidden_dim);
        if (!myds_metal_gemv(ctx->ffn_out, layer->ffn_down, ctx->swiglu_out, dim, hidden_dim, layer->ffn_down_type)) {
            MYDS_LOG_E("Forward failed: GEMV for ffn_down at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }
        myds_metal_add(ctx->x, ctx->x, ctx->ffn_out, dim);
    } else if (layer->shared_gate && layer->shared_up && layer->shared_down && ctx->shared_out) {
        uint32_t exp_dim = m->moe_expert_dim > 0 ? m->moe_expert_dim : 2304;
        if (!myds_metal_gemv(ctx->shared_gate_out, layer->shared_gate, ctx->ffn_norm_out, exp_dim, dim, layer->shared_gate_type) ||
            !myds_metal_gemv(ctx->shared_up_out, layer->shared_up, ctx->ffn_norm_out, exp_dim, dim, layer->shared_up_type)) {
            MYDS_LOG_E("Forward failed: GEMV for shared_gate/up at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }
        myds_metal_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out, ctx->shared_up_out, exp_dim);
        if (!myds_metal_gemv(ctx->shared_out, layer->shared_down, ctx->shared_swiglu_out, dim, exp_dim, layer->shared_down_type)) {
            MYDS_LOG_E("Forward failed: GEMV for shared_down at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }
        myds_metal_add(ctx->x, ctx->x, ctx->shared_out, dim);
    }

    if (do_ffn_prof) {
        myds_metal_commit_batch();
    }

    if (do_ffn_prof && tf1.tv_sec > 0) {
        struct timeval tf3;
        gettimeofday(&tf3, NULL);
        double ms_gate = (tf1.tv_sec - tf0.tv_sec)*1000.0 + (tf1.tv_usec - tf0.tv_usec)/1000.0;
        double ms_route = (tf2.tv_sec - tf1.tv_sec)*1000.0 + (tf2.tv_usec - tf1.tv_usec)/1000.0;
        double ms_moe_shexp = (tf3.tv_sec - tf2.tv_sec)*1000.0 + (tf3.tv_usec - tf2.tv_usec)/1000.0;
        double ms_ffn_tot = (tf3.tv_sec - tf0.tv_sec)*1000.0 + (tf3.tv_usec - tf0.tv_usec)/1000.0;
        printf("[PROF FFN L0] router_gate=%.2f ms, moe_route=%.2f ms, moe_batch=%.2f ms | FFN_TOTAL=%.2f ms\n",
               ms_gate, ms_route, ms_moe_shexp, ms_ffn_tot);
        fflush(stdout);
    } else if (do_ffn_prof) {
        struct timeval tf3;
        gettimeofday(&tf3, NULL);
        double ms_ffn_tot = (tf3.tv_sec - tf0.tv_sec)*1000.0 + (tf3.tv_usec - tf0.tv_usec)/1000.0;
        printf("[PROF FFN L0] dense_ffn=%.2f ms | FFN_TOTAL=%.2f ms\n", ms_ffn_tot, ms_ffn_tot);
        fflush(stdout);
    }
    return 1;
}

static int execute_transformer_layer(myds_context_t *ctx, uint32_t l, uint32_t pos) {
    myds_model_t *m = ctx->model;
    uint32_t dim = m->dim;
    uint32_t n_heads = m->n_heads;
    uint32_t n_kv_heads = m->n_kv_heads;
    uint32_t head_dim = m->head_dim;
    myds_layer_t *layer = &m->layers[l];
    float *xv = (float *)myds_metal_buffer_data(ctx->x);

    if (m->is_mla) {
        /* =========================================================
         * Multi-Head Latent Attention (MLA)
         * ========================================================= */

        /* Batch 1: Pre-Attention RMSNorm + Query LoRA down-proj + Latent KV proj */
        struct timeval t0, t1, t2, t3, t4, t5, t6;
        bool do_prof = (l == 0 && getenv("MYDS_PROF") != NULL);
        if (do_prof) gettimeofday(&t0, NULL);

        myds_metal_begin_batch();
        if (layer->attn_norm) {
            myds_metal_rmsnorm(ctx->x_norm, ctx->x, layer->attn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(ctx->x_norm), xv, dim * sizeof(float));
        }

        if (!myds_metal_gemv(ctx->qr, layer->attn_q_a, ctx->x_norm, m->q_lora_rank, dim, layer->attn_q_a_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_q_a at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }

        if (!myds_metal_gemv(ctx->kv_act, layer->attn_kv, ctx->x_norm, m->head_dim, dim, layer->attn_kv_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_kv at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }

        /* GPU Query LoRA RMSNorm */
        if (!myds_metal_rmsnorm(ctx->qr_norm, ctx->qr, layer->attn_q_a_norm, m->q_lora_rank, m->norm_eps)) {
            MYDS_LOG_E("Forward failed: rmsnorm for qr_norm at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }

        /* Query up-projection: qr_norm (q_lora_rank) -> q (n_heads * head_dim) */
        if (!myds_metal_gemv(ctx->q, layer->attn_q_b, ctx->qr_norm, m->n_heads * m->head_dim, m->q_lora_rank, layer->attn_q_b_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_q_b at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }

        /* GPU Key/Value RMSNorm */
        if (!myds_metal_rmsnorm(ctx->kv_act, ctx->kv_act, layer->attn_kv_a_norm, m->head_dim, m->norm_eps)) {
            MYDS_LOG_E("Forward failed: rmsnorm for kv_act at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }
        myds_metal_commit_batch();
        if (do_prof) gettimeofday(&t1, NULL);

        float *qv = (float *)myds_metal_buffer_data(ctx->q);
        head_rms_norm_inplace(qv, m->n_heads, m->head_dim, m->norm_eps);

        float *kv_v = (float *)myds_metal_buffer_data(ctx->kv_act);
        if (do_prof) gettimeofday(&t2, NULL);

        /* Tail RoPE on Q (all heads) and KV (1 head) */
        rope_tail_inplace(qv, m->n_heads, m->head_dim, m->qk_rope_head_dim, pos, m->freq_base, m->freq_scale, false);
        rope_tail_inplace(kv_v, 1, m->head_dim, m->qk_rope_head_dim, pos, m->freq_base, m->freq_scale, false);

        /* Push to latent KV cache: 512 floats per token per layer */
        memcpy(ctx->k_cache[l] + (size_t)pos * m->head_dim, kv_v, m->head_dim * sizeof(float));

        /* Sink-aware Multi-Head Attention over latent KV cache */
        float *heads_v = (float *)myds_metal_buffer_data(ctx->attn_heads);
        const float *sinks_w = layer->attn_sinks ? (const float *)myds_metal_buffer_data(layer->attn_sinks) : NULL;
        layer_attention_sinks(heads_v, sinks_w, qv, ctx->k_cache[l], pos + 1, m->n_heads, m->head_dim);
        if (do_prof) gettimeofday(&t3, NULL);

        /* Inverse RoPE on attention output heads tail */
        rope_tail_inplace(heads_v, m->n_heads, m->head_dim, m->qk_rope_head_dim, pos, m->freq_base, m->freq_scale, true);

        /* Batch 2: Attention output down-projection + up-projection + residual add + Pre-FFN RMSNorm + Router GEMV */
        myds_metal_begin_batch();
        uint32_t group_dim = (m->n_heads / m->n_out_groups) * m->head_dim;
        if (!myds_metal_attn_output_a(ctx->attn_low, layer->attn_output_a, ctx->attn_heads,
                                      m->n_out_groups, group_dim, m->out_group_rank, layer->attn_output_a_type)) {
            MYDS_LOG_E("Forward failed: attn_output_a at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }
        if (do_prof) gettimeofday(&t4, NULL);

        if (!myds_metal_gemv(ctx->attn_out, layer->attn_output_b, ctx->attn_low, dim, m->n_out_groups * m->out_group_rank, layer->attn_output_b_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_output_b at layer %u failed", l);
            myds_metal_commit_batch();
            return 0;
        }
        myds_metal_add(ctx->x, ctx->x, ctx->attn_out, dim);

        /* Pre-FFN RMSNorm */
        if (layer->ffn_norm) {
            myds_metal_rmsnorm(ctx->ffn_norm_out, ctx->x, layer->ffn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(ctx->ffn_norm_out), xv, dim * sizeof(float));
        }
        if (do_prof) gettimeofday(&t5, NULL);

        if (!execute_ffn_layer(ctx, layer, l, dim, m->hidden_dim)) {
            return 0;
        }
        if (do_prof) {
            gettimeofday(&t6, NULL);
            double ms1 = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_usec - t0.tv_usec)/1000.0;
            double ms2 = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
            double ms3 = (t3.tv_sec - t2.tv_sec)*1000.0 + (t3.tv_usec - t2.tv_usec)/1000.0;
            double ms4 = (t4.tv_sec - t3.tv_sec)*1000.0 + (t4.tv_usec - t3.tv_usec)/1000.0;
            double ms5 = (t5.tv_sec - t4.tv_sec)*1000.0 + (t5.tv_usec - t4.tv_usec)/1000.0;
            double ms6 = (t6.tv_sec - t5.tv_sec)*1000.0 + (t6.tv_usec - t5.tv_usec)/1000.0;
            double ms_tot = (t6.tv_sec - t0.tv_sec)*1000.0 + (t6.tv_usec - t0.tv_usec)/1000.0;
            printf("[PROF L0] batch1_qkv=%.2f ms, q_b=%.2f ms, attn_sinks=%.2f ms, out_a=%.2f ms, batch2_out_b=%.2f ms, ffn=%.2f ms | TOTAL=%.2f ms\n",
                   ms1, ms2, ms3, ms4, ms5, ms6, ms_tot);
            fflush(stdout);
        }
    } else {
        /* =========================================================
         * Multi-Head Attention (MHA / GQA)
         * ========================================================= */

        /* Batch 1: Pre-Attention RMSNorm + Q, K, V projections */
        myds_metal_begin_batch();
        if (layer->attn_norm) {
            myds_metal_rmsnorm(ctx->x_norm, ctx->x, layer->attn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(ctx->x_norm), xv, dim * sizeof(float));
        }

        if (layer->attn_q) {
            if (!myds_metal_gemv(ctx->q, layer->attn_q, ctx->x_norm, n_heads * head_dim, dim, layer->attn_q_type)) {
                MYDS_LOG_E("Forward failed: GEMV for attn_q at layer %u failed", l);
                myds_metal_commit_batch();
                return 0;
            }
        }
        if (layer->attn_k) {
            if (!myds_metal_gemv(ctx->k, layer->attn_k, ctx->x_norm, n_kv_heads * head_dim, dim, layer->attn_k_type)) {
                MYDS_LOG_E("Forward failed: GEMV for attn_k at layer %u failed", l);
                myds_metal_commit_batch();
                return 0;
            }
        }
        if (layer->attn_v) {
            if (!myds_metal_gemv(ctx->v, layer->attn_v, ctx->x_norm, n_kv_heads * head_dim, dim, layer->attn_v_type)) {
                MYDS_LOG_E("Forward failed: GEMV for attn_v at layer %u failed", l);
                myds_metal_commit_batch();
                return 0;
            }
        }
        myds_metal_commit_batch();

        /* Decoupled RoPE on Q & K */
        myds_metal_rope(ctx->q, ctx->k, n_heads, head_dim, pos, m->freq_base, m->freq_scale);

        /* Write into KV Cache */
        float *cur_k = (float *)myds_metal_buffer_data(ctx->k);
        float *cur_v = (float *)myds_metal_buffer_data(ctx->v);
        size_t kv_stride = n_kv_heads * head_dim;
        memcpy(ctx->k_cache[l] + (size_t)pos * kv_stride, cur_k, kv_stride * sizeof(float));
        memcpy(ctx->v_cache[l] + (size_t)pos * kv_stride, cur_v, kv_stride * sizeof(float));

        /* Multi-Head Attention computation */
        float *qv = (float *)myds_metal_buffer_data(ctx->q);
        float *attn_out_v = (float *)myds_metal_buffer_data(ctx->attn_out);
        memset(attn_out_v, 0, dim * sizeof(float));

        float scale_attn = 1.0f / sqrtf((float)head_dim);
        uint32_t kv_mul = n_heads / n_kv_heads;

        for (uint32_t h = 0; h < n_heads; h++) {
            uint32_t kv_h = h / kv_mul;
            float *q_head = qv + h * head_dim;

            /* Attention scores */
            float *scores = (float *)myds_metal_buffer_data(ctx->attn_scores);
            float max_s = -INFINITY;
            for (uint32_t t = 0; t <= pos; t++) {
                const float *k_head = ctx->k_cache[l] + (size_t)t * kv_stride + kv_h * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += q_head[d] * k_head[d];
                }
                scores[t] = dot * scale_attn;
                if (scores[t] > max_s) max_s = scores[t];
            }

            /* Softmax */
            float sum_exp = 0.0f;
            for (uint32_t t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / sum_exp;
            for (uint32_t t = 0; t <= pos; t++) {
                scores[t] *= inv_sum;
            }

            /* Value weighted sum */
            float *out_head = attn_out_v + h * head_dim;
            for (uint32_t t = 0; t <= pos; t++) {
                const float *v_head = ctx->v_cache[l] + (size_t)t * kv_stride + kv_h * head_dim;
                float weight = scores[t];
                for (uint32_t d = 0; d < head_dim; d++) {
                    out_head[d] += weight * v_head[d];
                }
            }
        }

        /* Batch 2: Output projection + residual add + Pre-FFN RMSNorm + FFN */
        myds_metal_begin_batch();
        if (layer->attn_o) {
            if (!myds_metal_gemv(ctx->ffn_out, layer->attn_o, ctx->attn_out, dim, n_heads * head_dim, layer->attn_o_type)) {
                MYDS_LOG_E("Forward failed: GEMV for attn_o at layer %u failed", l);
                myds_metal_commit_batch();
                return 0;
            }
            /* Residual connection: x = x + attn_output */
            myds_metal_add(ctx->x, ctx->x, ctx->ffn_out, dim);
        }

        /* Pre-FFN RMSNorm */
        if (layer->ffn_norm) {
            myds_metal_rmsnorm(ctx->ffn_norm_out, ctx->x, layer->ffn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(ctx->ffn_norm_out), xv, dim * sizeof(float));
        }

        if (!execute_ffn_layer(ctx, layer, l, dim, m->hidden_dim)) {
            return 0;
        }
    }
    return 1;
}

static int32_t myds_forward_token_internal(myds_context_t *ctx, int32_t token_id, uint32_t pos, bool need_logits) {
    myds_model_t *m = ctx->model;
    uint32_t dim = m->dim;

    /* 1. Embedding lookup */
    float *xv = (float *)myds_metal_buffer_data(ctx->x);
    if (m->tok_embeddings) {
        if (m->tok_embeddings_type == GGML_TYPE_F32) {
            const float *emb = (const float *)myds_metal_buffer_data(m->tok_embeddings);
            memcpy(xv, emb + (size_t)token_id * dim, dim * sizeof(float));
        } else if (m->tok_embeddings_type == GGML_TYPE_F16) {
            const uint16_t *emb = (const uint16_t *)myds_metal_buffer_data(m->tok_embeddings);
            const uint16_t *row = emb + (size_t)token_id * dim;
            for (uint32_t i = 0; i < dim; i++) {
                uint32_t w = (uint32_t)row[i] << 16;
                memcpy(&xv[i], &w, sizeof(float));
            }
        } else {
            memset(xv, 0, dim * sizeof(float));
        }
    }

    /* 2. Transformer layers */
    for (uint32_t l = 0; l < m->n_layers; l++) {
        if (!execute_transformer_layer(ctx, l, pos)) {
            return -1;
        }
    }

    if (!need_logits) {
        myds_metal_commit_batch();
        return 0;
    }

    /* 3. Final RMSNorm & LM Head Logits Batch */
    struct timeval t_lm0, t_lm1, t_sp0, t_sp1;
    bool do_head_prof = (getenv("MYDS_PROF") != NULL);
    if (do_head_prof) gettimeofday(&t_lm0, NULL);

    myds_metal_begin_batch();
    if (m->output_norm) {
        myds_metal_rmsnorm(ctx->x_norm, ctx->x, m->output_norm, dim, m->norm_eps);
    } else {
        myds_metal_commit_batch();
        memcpy(myds_metal_buffer_data(ctx->x_norm), xv, dim * sizeof(float));
        myds_metal_begin_batch();
    }

    if (m->output) {
        if (!myds_metal_gemv(ctx->logits, m->output, ctx->x_norm, m->vocab_size, dim, m->output_type)) {
            MYDS_LOG_E("Forward failed: LM head GEMV failed");
            myds_metal_commit_batch();
            return -1;
        }
    }
    myds_metal_commit_batch();
    if (do_head_prof) {
        gettimeofday(&t_lm1, NULL);
        gettimeofday(&t_sp0, NULL);
    }

    /* 4. Sample next token */
    const float *logits_v = (const float *)myds_metal_buffer_data(ctx->logits);
    int32_t next_token = myds_sample_next_token(logits_v, m->vocab_size, &ctx->cfg, ctx, NULL, 0);
    if (do_head_prof) {
        gettimeofday(&t_sp1, NULL);
        double ms_lm = (t_lm1.tv_sec - t_lm0.tv_sec)*1000.0 + (t_lm1.tv_usec - t_lm0.tv_usec)/1000.0;
        double ms_sp = (t_sp1.tv_sec - t_sp0.tv_sec)*1000.0 + (t_sp1.tv_usec - t_sp0.tv_usec)/1000.0;
        printf("[PROF HEAD] lm_head=%.2f ms, sampler=%.2f ms\n", ms_lm, ms_sp);
        fflush(stdout);
    }

    return next_token;
}

int32_t myds_forward_token(myds_context_t *ctx, int32_t token_id, uint32_t pos) {
    return myds_forward_token_internal(ctx, token_id, pos, true);
}

#define MYDS_PREFILL_CHUNK_SIZE 512

typedef struct {
    /* Contiguous unified chunk buffers for batched GEMM */
    myds_metal_buffer_t *x_buf;
    myds_metal_buffer_t *x_norm_buf;
    myds_metal_buffer_t *qr_buf;
    myds_metal_buffer_t *qr_norm_buf;
    myds_metal_buffer_t *q_buf;
    myds_metal_buffer_t *kv_act_buf;
    myds_metal_buffer_t *attn_heads_buf;
    myds_metal_buffer_t *attn_low_buf;
    myds_metal_buffer_t *attn_out_buf;
    myds_metal_buffer_t *ffn_norm_out_buf;
    myds_metal_buffer_t *router_logits_buf;
    myds_metal_buffer_t *shared_gate_out_buf;
    myds_metal_buffer_t *shared_up_out_buf;
    myds_metal_buffer_t *shared_swiglu_out_buf;
    myds_metal_buffer_t *shared_out_buf;
    myds_metal_buffer_t *moe_out_buf;

    /* Per-token slice views for token workers */
    myds_metal_buffer_t *x[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *x_norm[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *qr[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *qr_norm[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *q[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *kv_act[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *attn_heads[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *attn_low[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *attn_out[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *ffn_norm_out[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *router_logits[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *shared_gate_out[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *shared_up_out[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *shared_swiglu_out[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *shared_out[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *moe_out[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *selected_ids[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *selected_weights[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *moe_gate[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *moe_up[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *moe_mid[MYDS_PREFILL_CHUNK_SIZE];
    myds_metal_buffer_t *moe_down[MYDS_PREFILL_CHUNK_SIZE];
} myds_prefill_chunk_t;

static myds_prefill_chunk_t *create_prefill_chunk(myds_model_t *model) {
    myds_prefill_chunk_t *chunk = (myds_prefill_chunk_t *)calloc(1, sizeof(myds_prefill_chunk_t));
    if (!chunk) return NULL;
    uint32_t dim = model->dim;
    uint32_t exp_dim = model->moe_expert_dim > 0 ? model->moe_expert_dim : 2304;
    uint32_t n_routed = model->n_routed_experts > 0 ? model->n_routed_experts : 384;
    uint32_t n_active = model->n_activated_experts > 0 ? model->n_activated_experts : 6;
    size_t C = MYDS_PREFILL_CHUNK_SIZE;

    /* 1. Allocate contiguous chunk tensors */
    chunk->x_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    chunk->x_norm_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    chunk->attn_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    chunk->ffn_norm_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    if (model->is_mla) {
        chunk->qr_buf = myds_metal_buffer_alloc(C * model->q_lora_rank * sizeof(float));
        chunk->qr_norm_buf = myds_metal_buffer_alloc(C * model->q_lora_rank * sizeof(float));
        chunk->q_buf = myds_metal_buffer_alloc(C * model->n_heads * model->head_dim * sizeof(float));
        chunk->kv_act_buf = myds_metal_buffer_alloc(C * model->head_dim * sizeof(float));
        chunk->attn_heads_buf = myds_metal_buffer_alloc(C * model->n_heads * model->head_dim * sizeof(float));
        chunk->attn_low_buf = myds_metal_buffer_alloc(C * model->n_out_groups * model->out_group_rank * sizeof(float));
    }
    chunk->router_logits_buf = myds_metal_buffer_alloc(C * n_routed * sizeof(float));
    chunk->shared_gate_out_buf = myds_metal_buffer_alloc(C * exp_dim * sizeof(float));
    chunk->shared_up_out_buf = myds_metal_buffer_alloc(C * exp_dim * sizeof(float));
    chunk->shared_swiglu_out_buf = myds_metal_buffer_alloc(C * exp_dim * sizeof(float));
    chunk->shared_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    chunk->moe_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));

    /* 2. Create subviews for per-token workers and allocate scratch buffers */
    for (size_t t = 0; t < C; t++) {
        chunk->x[t] = myds_metal_buffer_sub_view(chunk->x_buf, t * dim * sizeof(float), dim * sizeof(float));
        chunk->x_norm[t] = myds_metal_buffer_sub_view(chunk->x_norm_buf, t * dim * sizeof(float), dim * sizeof(float));
        chunk->attn_out[t] = myds_metal_buffer_sub_view(chunk->attn_out_buf, t * dim * sizeof(float), dim * sizeof(float));
        chunk->ffn_norm_out[t] = myds_metal_buffer_sub_view(chunk->ffn_norm_out_buf, t * dim * sizeof(float), dim * sizeof(float));
        if (model->is_mla) {
            chunk->qr[t] = myds_metal_buffer_sub_view(chunk->qr_buf, t * model->q_lora_rank * sizeof(float), model->q_lora_rank * sizeof(float));
            chunk->qr_norm[t] = myds_metal_buffer_sub_view(chunk->qr_norm_buf, t * model->q_lora_rank * sizeof(float), model->q_lora_rank * sizeof(float));
            chunk->q[t] = myds_metal_buffer_sub_view(chunk->q_buf, t * model->n_heads * model->head_dim * sizeof(float), model->n_heads * model->head_dim * sizeof(float));
            chunk->kv_act[t] = myds_metal_buffer_sub_view(chunk->kv_act_buf, t * model->head_dim * sizeof(float), model->head_dim * sizeof(float));
            chunk->attn_heads[t] = myds_metal_buffer_sub_view(chunk->attn_heads_buf, t * model->n_heads * model->head_dim * sizeof(float), model->n_heads * model->head_dim * sizeof(float));
            chunk->attn_low[t] = myds_metal_buffer_sub_view(chunk->attn_low_buf, t * model->n_out_groups * model->out_group_rank * sizeof(float), model->n_out_groups * model->out_group_rank * sizeof(float));
        }
        chunk->router_logits[t] = myds_metal_buffer_sub_view(chunk->router_logits_buf, t * n_routed * sizeof(float), n_routed * sizeof(float));
        chunk->shared_gate_out[t] = myds_metal_buffer_sub_view(chunk->shared_gate_out_buf, t * exp_dim * sizeof(float), exp_dim * sizeof(float));
        chunk->shared_up_out[t] = myds_metal_buffer_sub_view(chunk->shared_up_out_buf, t * exp_dim * sizeof(float), exp_dim * sizeof(float));
        chunk->shared_swiglu_out[t] = myds_metal_buffer_sub_view(chunk->shared_swiglu_out_buf, t * exp_dim * sizeof(float), exp_dim * sizeof(float));
        chunk->shared_out[t] = myds_metal_buffer_sub_view(chunk->shared_out_buf, t * dim * sizeof(float), dim * sizeof(float));
        chunk->moe_out[t] = myds_metal_buffer_sub_view(chunk->moe_out_buf, t * dim * sizeof(float), dim * sizeof(float));

        chunk->selected_ids[t] = myds_metal_buffer_alloc(n_active * sizeof(int32_t));
        chunk->selected_weights[t] = myds_metal_buffer_alloc(n_active * sizeof(float));
        chunk->moe_gate[t] = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        chunk->moe_up[t] = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        chunk->moe_mid[t] = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        chunk->moe_down[t] = myds_metal_buffer_alloc(dim * sizeof(float));
    }
    return chunk;
}

static void free_prefill_chunk(myds_prefill_chunk_t *chunk) {
    if (!chunk) return;
    for (int t = 0; t < MYDS_PREFILL_CHUNK_SIZE; t++) {
        if (chunk->x[t]) myds_metal_buffer_free(chunk->x[t]);
        if (chunk->x_norm[t]) myds_metal_buffer_free(chunk->x_norm[t]);
        if (chunk->attn_out[t]) myds_metal_buffer_free(chunk->attn_out[t]);
        if (chunk->ffn_norm_out[t]) myds_metal_buffer_free(chunk->ffn_norm_out[t]);
        if (chunk->qr[t]) myds_metal_buffer_free(chunk->qr[t]);
        if (chunk->qr_norm[t]) myds_metal_buffer_free(chunk->qr_norm[t]);
        if (chunk->q[t]) myds_metal_buffer_free(chunk->q[t]);
        if (chunk->kv_act[t]) myds_metal_buffer_free(chunk->kv_act[t]);
        if (chunk->attn_heads[t]) myds_metal_buffer_free(chunk->attn_heads[t]);
        if (chunk->attn_low[t]) myds_metal_buffer_free(chunk->attn_low[t]);
        if (chunk->router_logits[t]) myds_metal_buffer_free(chunk->router_logits[t]);
        if (chunk->shared_gate_out[t]) myds_metal_buffer_free(chunk->shared_gate_out[t]);
        if (chunk->shared_up_out[t]) myds_metal_buffer_free(chunk->shared_up_out[t]);
        if (chunk->shared_swiglu_out[t]) myds_metal_buffer_free(chunk->shared_swiglu_out[t]);
        if (chunk->shared_out[t]) myds_metal_buffer_free(chunk->shared_out[t]);
        if (chunk->moe_out[t]) myds_metal_buffer_free(chunk->moe_out[t]);
        if (chunk->selected_ids[t]) myds_metal_buffer_free(chunk->selected_ids[t]);
        if (chunk->selected_weights[t]) myds_metal_buffer_free(chunk->selected_weights[t]);
        if (chunk->moe_gate[t]) myds_metal_buffer_free(chunk->moe_gate[t]);
        if (chunk->moe_up[t]) myds_metal_buffer_free(chunk->moe_up[t]);
        if (chunk->moe_mid[t]) myds_metal_buffer_free(chunk->moe_mid[t]);
        if (chunk->moe_down[t]) myds_metal_buffer_free(chunk->moe_down[t]);
    }
    if (chunk->x_buf) myds_metal_buffer_free(chunk->x_buf);
    if (chunk->x_norm_buf) myds_metal_buffer_free(chunk->x_norm_buf);
    if (chunk->attn_out_buf) myds_metal_buffer_free(chunk->attn_out_buf);
    if (chunk->ffn_norm_out_buf) myds_metal_buffer_free(chunk->ffn_norm_out_buf);
    if (chunk->qr_buf) myds_metal_buffer_free(chunk->qr_buf);
    if (chunk->qr_norm_buf) myds_metal_buffer_free(chunk->qr_norm_buf);
    if (chunk->q_buf) myds_metal_buffer_free(chunk->q_buf);
    if (chunk->kv_act_buf) myds_metal_buffer_free(chunk->kv_act_buf);
    if (chunk->attn_heads_buf) myds_metal_buffer_free(chunk->attn_heads_buf);
    if (chunk->attn_low_buf) myds_metal_buffer_free(chunk->attn_low_buf);
    if (chunk->router_logits_buf) myds_metal_buffer_free(chunk->router_logits_buf);
    if (chunk->shared_gate_out_buf) myds_metal_buffer_free(chunk->shared_gate_out_buf);
    if (chunk->shared_up_out_buf) myds_metal_buffer_free(chunk->shared_up_out_buf);
    if (chunk->shared_swiglu_out_buf) myds_metal_buffer_free(chunk->shared_swiglu_out_buf);
    if (chunk->shared_out_buf) myds_metal_buffer_free(chunk->shared_out_buf);
    if (chunk->moe_out_buf) myds_metal_buffer_free(chunk->moe_out_buf);
    free(chunk);
}

typedef struct {
    myds_prefill_chunk_t *chunk;
    const float          *sinks_w;
    const float          *k_cache_layer;
    uint32_t              cur_pos;
    uint32_t              n_heads;
    uint32_t              head_dim;
    float                 kq_scale;
} chunk_attn_job_t;

typedef struct {
    myds_prefill_chunk_t *chunk;
    myds_context_t       *ctx;
    uint32_t              l;
    uint32_t              cur_pos;
    uint32_t              n_heads;
    uint32_t              head_dim;
    uint32_t              qk_rope_head_dim;
    float                 freq_base;
    float                 freq_scale;
    float                 norm_eps;
} chunk_prep_job_t;

static void chunk_prep_worker(void *context, size_t t) {
    const chunk_prep_job_t *job = (const chunk_prep_job_t *)context;
    uint32_t pos = job->cur_pos + (uint32_t)t;
    float *qv = (float *)myds_metal_buffer_data(job->chunk->q[t]);
    head_rms_norm_inplace(qv, job->n_heads, job->head_dim, job->norm_eps);

    float *kv_v = (float *)myds_metal_buffer_data(job->chunk->kv_act[t]);
    rope_tail_inplace(qv, job->n_heads, job->head_dim, job->qk_rope_head_dim, pos, job->freq_base, job->freq_scale, false);
    rope_tail_inplace(kv_v, 1, job->head_dim, job->qk_rope_head_dim, pos, job->freq_base, job->freq_scale, false);

    memcpy(job->ctx->k_cache[job->l] + (size_t)pos * job->head_dim, kv_v, job->head_dim * sizeof(float));
}

static void chunk_post_rope_worker(void *context, size_t t) {
    const chunk_prep_job_t *job = (const chunk_prep_job_t *)context;
    uint32_t pos = job->cur_pos + (uint32_t)t;
    float *heads_v = (float *)myds_metal_buffer_data(job->chunk->attn_heads[t]);
    rope_tail_inplace(heads_v, job->n_heads, job->head_dim, job->qk_rope_head_dim, pos, job->freq_base, job->freq_scale, true);
}

typedef struct {
    myds_prefill_chunk_t      *chunk;
    const myds_metal_buffer_t *router_bias;
    uint32_t                   n_routed;
    uint32_t                   n_active;
    float                      routed_scaling_factor;
} chunk_router_job_t;

static void chunk_router_worker(void *context, size_t t) {
    const chunk_router_job_t *job = (const chunk_router_job_t *)context;
    int32_t *sel_ids = (int32_t *)myds_metal_buffer_data(job->chunk->selected_ids[t]);
    float *sel_weights = (float *)myds_metal_buffer_data(job->chunk->selected_weights[t]);
    myds_metal_moe_route_biased(
        sel_ids,
        sel_weights,
        job->chunk->router_logits[t],
        job->router_bias,
        job->n_routed,
        job->n_active,
        job->routed_scaling_factor);
}

static void chunk_attention_token_worker(void *context, size_t t) {
    const chunk_attn_job_t *job = (const chunk_attn_job_t *)context;
    uint32_t n_heads = job->n_heads;
    uint32_t head_dim = job->head_dim;
    uint32_t pos = job->cur_pos + (uint32_t)t;
    uint32_t n_kv = pos + 1;

    const float *qv = (const float *)myds_metal_buffer_data(job->chunk->q[t]);
    float *heads_v = (float *)myds_metal_buffer_data(job->chunk->attn_heads[t]);

    float score_stack[4096];
    float *score = (n_kv <= 4096) ? score_stack : (float *)malloc((size_t)n_kv * sizeof(float));
    if (!score) return;

    for (uint32_t h = 0; h < n_heads; h++) {
        const float *qh = qv + (size_t)h * head_dim;
        float max_score = job->sinks_w ? job->sinks_w[h] : -INFINITY;
        for (uint32_t r = 0; r < n_kv; r++) {
            const float *kv = job->k_cache_layer + (size_t)r * head_dim;
            score[r] = dot_f32(qh, kv, head_dim) * job->kq_scale;
            if (score[r] > max_score) max_score = score[r];
        }

        float *oh = heads_v + (size_t)h * head_dim;
        memset(oh, 0, (size_t)head_dim * sizeof(float));

        float denom = job->sinks_w ? expf(job->sinks_w[h] - max_score) : 0.0f;
        for (uint32_t r = 0; r < n_kv; r++) {
            const float weight = expf(score[r] - max_score);
            const float *kv = job->k_cache_layer + (size_t)r * head_dim;
            denom += weight;
            axpy_f32(oh, kv, weight, head_dim);
        }

        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        scale_f32(oh, inv, head_dim);
    }

    if (score != score_stack) free(score);
}

static inline bool chunk_linear_proj(
    myds_metal_buffer_t **dst_vecs,
    myds_metal_buffer_t  *dst_buf,
    const myds_metal_buffer_t *W,
    myds_metal_buffer_t **src_vecs,
    myds_metal_buffer_t  *src_buf,
    uint32_t d_out,
    uint32_t d_in,
    uint32_t chunk_len,
    int type,
    bool force_gemv) {
    if (force_gemv) {
        for (uint32_t t = 0; t < chunk_len; t++) {
            if (!myds_metal_gemv(dst_vecs[t], W, src_vecs[t], d_out, d_in, type)) return false;
        }
        return true;
    }
    return myds_metal_mul_mm(dst_buf, W, src_buf, d_out, d_in, chunk_len, type);
}

static int execute_prefill_chunk_layer(
    myds_context_t       *ctx,
    myds_prefill_chunk_t *chunk,
    uint32_t              l,
    uint32_t              cur_pos,
    uint32_t              chunk_len) {
    myds_model_t *m = ctx->model;
    uint32_t dim = m->dim;
    uint32_t head_dim = m->head_dim;
    uint32_t n_heads = m->n_heads;
    uint32_t n_routed = m->n_routed_experts > 0 ? m->n_routed_experts : 384;
    uint32_t n_active = m->n_activated_experts > 0 ? m->n_activated_experts : 6;
    uint32_t exp_dim = m->moe_expert_dim > 0 ? m->moe_expert_dim : 2304;
    myds_layer_t *layer = &m->layers[l];

    const char *env_m = getenv("MYDS_PREFILL_MODE");
    int prefill_mode = env_m ? atoi(env_m) : 0;
    /* prefill_mode:
     * 0: all GEMV (100% bit-exact baseline)
     * -1: all mul_mm
     * >0: bitmask:
     *   bit 0 (1): attn_q_a + attn_kv
     *   bit 1 (2): attn_q_b
     *   bit 2 (4): attn_output_b
     *   bit 3 (8): router_gate
     *   bit 4 (16): shared_gate + shared_up
     *   bit 5 (32): shared_down
     */
    bool use_mm_qa      = (prefill_mode == -1) || (prefill_mode > 0 && (prefill_mode & 1));
    bool use_mm_qb      = (prefill_mode == -1) || (prefill_mode > 0 && (prefill_mode & 2));
    bool use_mm_outb    = (prefill_mode == -1) || (prefill_mode > 0 && (prefill_mode & 4));
    bool use_mm_router  = (prefill_mode == -1) || (prefill_mode > 0 && (prefill_mode & 8));
    bool use_mm_sh_up   = (prefill_mode == -1) || (prefill_mode > 0 && (prefill_mode & 16));
    bool use_mm_sh_down = (prefill_mode == -1) || (prefill_mode > 0 && (prefill_mode & 32));

    struct timeval tp0, tp1, tp2, tp3, tp4;
    bool do_prof = (l == 0 && cur_pos == 0 && getenv("MYDS_PROF") != NULL);
    if (do_prof) gettimeofday(&tp0, NULL);

    /* Batch 1: RMSNorm + Q_a + KV + QR_Norm + Q_b + KV_Norm across all chunk tokens (barrier optimized) */
    myds_metal_set_auto_barrier(0);
    myds_metal_begin_batch();

    /* 1.1 Pre-Attention RMSNorm */
    for (uint32_t t = 0; t < chunk_len; t++) {
        if (layer->attn_norm) {
            myds_metal_rmsnorm(chunk->x_norm[t], chunk->x[t], layer->attn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(chunk->x_norm[t]), myds_metal_buffer_data(chunk->x[t]), dim * sizeof(float));
        }
    }
    myds_metal_barrier();

    /* 1.2 Q_a and KV projections */
    if (!chunk_linear_proj(chunk->qr, chunk->qr_buf, layer->attn_q_a, chunk->x_norm, chunk->x_norm_buf, m->q_lora_rank, dim, chunk_len, layer->attn_q_a_type, !use_mm_qa) ||
        !chunk_linear_proj(chunk->kv_act, chunk->kv_act_buf, layer->attn_kv, chunk->x_norm, chunk->x_norm_buf, m->head_dim, dim, chunk_len, layer->attn_kv_type, !use_mm_qa)) {
        MYDS_LOG_E("Prefill chunk Batch 1 Q_a/KV failed at layer %u", l);
        myds_metal_set_auto_barrier(1);
        myds_metal_commit_batch();
        return 0;
    }
    myds_metal_barrier();

    /* 1.3 RMSNorm for QR and KV */
    for (uint32_t t = 0; t < chunk_len; t++) {
        if (!myds_metal_rmsnorm(chunk->qr_norm[t], chunk->qr[t], layer->attn_q_a_norm, m->q_lora_rank, m->norm_eps) ||
            !myds_metal_rmsnorm(chunk->kv_act[t], chunk->kv_act[t], layer->attn_kv_a_norm, m->head_dim, m->norm_eps)) {
            MYDS_LOG_E("Prefill chunk Batch 1 norm failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 1.4 Q_b projection */
    if (!chunk_linear_proj(chunk->q, chunk->q_buf, layer->attn_q_b, chunk->qr_norm, chunk->qr_norm_buf, m->n_heads * m->head_dim, m->q_lora_rank, chunk_len, layer->attn_q_b_type, !use_mm_qb)) {
        MYDS_LOG_E("Prefill chunk Batch 1 Q_b failed at layer %u", l);
        myds_metal_set_auto_barrier(1);
        myds_metal_commit_batch();
        return 0;
    }
    myds_metal_commit_batch();
    if (do_prof) gettimeofday(&tp1, NULL);

    /* Step 2: CPU Attention & RoPE across chunk tokens (parallelized across tokens with full core cache locality) */
    dispatch_queue_t q_concurrent = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    chunk_prep_job_t pjob = {
        .chunk = chunk,
        .ctx = ctx,
        .l = l,
        .cur_pos = cur_pos,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .qk_rope_head_dim = m->qk_rope_head_dim,
        .freq_base = m->freq_base,
        .freq_scale = m->freq_scale,
        .norm_eps = m->norm_eps,
    };
    dispatch_apply_f((size_t)chunk_len, q_concurrent, &pjob, chunk_prep_worker);

    const float *sinks_w = layer->attn_sinks ? (const float *)myds_metal_buffer_data(layer->attn_sinks) : NULL;
    chunk_attn_job_t cjob = {
        .chunk = chunk,
        .sinks_w = sinks_w,
        .k_cache_layer = ctx->k_cache[l],
        .cur_pos = cur_pos,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .kq_scale = 1.0f / sqrtf((float)head_dim),
    };
    dispatch_apply_f((size_t)chunk_len, q_concurrent, &cjob, chunk_attention_token_worker);

    dispatch_apply_f((size_t)chunk_len, q_concurrent, &pjob, chunk_post_rope_worker);
    if (do_prof) gettimeofday(&tp2, NULL);

    /* Batch 2: Attn out A + Attn out B + Residual add + Pre-FFN RMSNorm + Router Gate + Shared Gate/Up */
    myds_metal_begin_batch();
    uint32_t group_dim = (m->n_heads / m->n_out_groups) * m->head_dim;

    /* 2.1 Attn out A */
    for (uint32_t t = 0; t < chunk_len; t++) {
        if (!myds_metal_attn_output_a(chunk->attn_low[t], layer->attn_output_a, chunk->attn_heads[t],
                                      m->n_out_groups, group_dim, m->out_group_rank, layer->attn_output_a_type)) {
            MYDS_LOG_E("Prefill chunk Batch 2 out_a failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 2.2 Attn out B */
    if (!chunk_linear_proj(chunk->attn_out, chunk->attn_out_buf, layer->attn_output_b, chunk->attn_low, chunk->attn_low_buf, dim, m->n_out_groups * m->out_group_rank, chunk_len, layer->attn_output_b_type, !use_mm_outb)) {
        MYDS_LOG_E("Prefill chunk Batch 2 out_b failed at layer %u", l);
        myds_metal_set_auto_barrier(1);
        myds_metal_commit_batch();
        return 0;
    }
    myds_metal_barrier();

    /* 2.3 Residual add */
    myds_metal_add(chunk->x_buf, chunk->x_buf, chunk->attn_out_buf, chunk_len * dim);
    myds_metal_barrier();

    /* 2.4 FFN RMSNorm */
    for (uint32_t t = 0; t < chunk_len; t++) {
        if (layer->ffn_norm) {
            myds_metal_rmsnorm(chunk->ffn_norm_out[t], chunk->x[t], layer->ffn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(chunk->ffn_norm_out[t]), myds_metal_buffer_data(chunk->x[t]), dim * sizeof(float));
        }
    }
    myds_metal_barrier();

    /* 2.5 Router gate + Shared gate/up */
    if (!chunk_linear_proj(chunk->router_logits, chunk->router_logits_buf, layer->router_gate, chunk->ffn_norm_out, chunk->ffn_norm_out_buf, n_routed, dim, chunk_len, layer->router_gate_type, !use_mm_router)) {
        MYDS_LOG_E("Prefill chunk router_gate failed at layer %u", l);
        myds_metal_set_auto_barrier(1);
        myds_metal_commit_batch();
        return 0;
    }
    if (layer->shared_gate && layer->shared_up && chunk->shared_out_buf) {
        if (!chunk_linear_proj(chunk->shared_gate_out, chunk->shared_gate_out_buf, layer->shared_gate, chunk->ffn_norm_out, chunk->ffn_norm_out_buf, exp_dim, dim, chunk_len, layer->shared_gate_type, !use_mm_sh_up) ||
            !chunk_linear_proj(chunk->shared_up_out, chunk->shared_up_out_buf, layer->shared_up, chunk->ffn_norm_out, chunk->ffn_norm_out_buf, exp_dim, dim, chunk_len, layer->shared_up_type, !use_mm_sh_up)) {
            MYDS_LOG_E("Prefill chunk shared_gate/up failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 2.6 Shared swiglu */
    for (uint32_t t = 0; t < chunk_len; t++) {
        if (layer->shared_gate && layer->shared_up && chunk->shared_out[t]) {
            myds_metal_swiglu(chunk->shared_swiglu_out[t], chunk->shared_gate_out[t], chunk->shared_up_out[t], exp_dim);
        }
    }
    myds_metal_commit_batch();
    if (do_prof) gettimeofday(&tp3, NULL);

    /* Step 4: CPU Router Top-6 selection for all chunk tokens (parallel across 24 cores) */
    chunk_router_job_t rjob = {
        .chunk = chunk,
        .router_bias = layer->router_bias,
        .n_routed = n_routed,
        .n_active = n_active,
        .routed_scaling_factor = m->routed_scaling_factor,
    };
    dispatch_apply_f((size_t)chunk_len, q_concurrent, &rjob, chunk_router_worker);

    /* Batch 3: MoE execution + Shared down + Residual add for all chunk tokens */
    myds_metal_begin_batch();
    for (uint32_t t = 0; t < chunk_len; t++) {
        if (!myds_metal_routed_moe(
                chunk->moe_out[t],
                chunk->ffn_norm_out[t],
                layer->ffn_gate_exps,
                layer->ffn_up_exps,
                layer->ffn_down_exps,
                chunk->selected_ids[t],
                chunk->selected_weights[t],
                chunk->moe_gate[t],
                chunk->moe_up[t],
                chunk->moe_mid[t],
                chunk->moe_down[t],
                dim,
                exp_dim,
                n_routed,
                n_active,
                layer->ffn_gate_exps_type,
                layer->ffn_up_exps_type,
                layer->ffn_down_exps_type,
                m->swiglu_clamp_exp)) {
            MYDS_LOG_E("Prefill chunk routed MoE failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 3.2 Shared down */
    if (layer->shared_gate && layer->shared_up && layer->shared_down && chunk->shared_out_buf) {
        if (!chunk_linear_proj(chunk->shared_out, chunk->shared_out_buf, layer->shared_down, chunk->shared_swiglu_out, chunk->shared_swiglu_out_buf, dim, exp_dim, chunk_len, layer->shared_down_type, !use_mm_sh_down)) {
            MYDS_LOG_E("Prefill chunk shared_down failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 3.3 Final residual addition */
    if (layer->shared_gate && layer->shared_up && layer->shared_down && chunk->shared_out_buf) {
        myds_metal_add(chunk->moe_out_buf, chunk->moe_out_buf, chunk->shared_out_buf, chunk_len * dim);
    }
    myds_metal_add(chunk->x_buf, chunk->x_buf, chunk->moe_out_buf, chunk_len * dim);
    myds_metal_commit_batch();
    myds_metal_set_auto_barrier(1);

    if (do_prof) {
        gettimeofday(&tp4, NULL);
        double ms_b1 = (tp1.tv_sec - tp0.tv_sec)*1000.0 + (tp1.tv_usec - tp0.tv_usec)/1000.0;
        double ms_attn = (tp2.tv_sec - tp1.tv_sec)*1000.0 + (tp2.tv_usec - tp1.tv_usec)/1000.0;
        double ms_b2 = (tp3.tv_sec - tp2.tv_sec)*1000.0 + (tp3.tv_usec - tp2.tv_usec)/1000.0;
        double ms_b3 = (tp4.tv_sec - tp3.tv_sec)*1000.0 + (tp4.tv_usec - tp3.tv_usec)/1000.0;
        double ms_tot = (tp4.tv_sec - tp0.tv_sec)*1000.0 + (tp4.tv_usec - tp0.tv_usec)/1000.0;
        printf("[PREFILL CHUNK L0] batch1_qkv=%.2f ms, cpu_attn=%.2f ms, batch2_ffn_router=%.2f ms, batch3_moe=%.2f ms | CHUNK_TOTAL=%.2f ms (%.2f ms/tok)\n",
               ms_b1, ms_attn, ms_b2, ms_b3, ms_tot, ms_tot / chunk_len);
        fflush(stdout);
    }
    return 1;
}

static int myds_prefill_tokens(myds_context_t *ctx, const int32_t *tokens, int32_t n_tokens, volatile int *cancel_flag) {
    if (n_tokens <= 0) return 1;
    myds_model_t *m = ctx->model;
    uint32_t dim = m->dim;

    /* For very short prompts (<= 4 tokens), standard sequential forward has virtually zero overhead */
    if (n_tokens <= 4 || !m->is_mla) {
        for (int32_t i = 0; i < n_tokens; i++) {
            if (cancel_flag && *cancel_flag) return -2;
            if (myds_forward_token_internal(ctx, tokens[i], ctx->pos, false) < 0) return -1;
            ctx->pos++;
        }
        return 1;
    }

    myds_prefill_chunk_t *chunk = create_prefill_chunk(m);
    if (!chunk) {
        for (int32_t i = 0; i < n_tokens; i++) {
            if (cancel_flag && *cancel_flag) return -2;
            if (myds_forward_token_internal(ctx, tokens[i], ctx->pos, false) < 0) return -1;
            ctx->pos++;
        }
        return 1;
    }

    float *prompt_x = (float *)malloc((size_t)n_tokens * dim * sizeof(float));
    if (!prompt_x) {
        free_prefill_chunk(chunk);
        for (int32_t i = 0; i < n_tokens; i++) {
            if (cancel_flag && *cancel_flag) return -2;
            if (myds_forward_token_internal(ctx, tokens[i], ctx->pos, false) < 0) return -1;
            ctx->pos++;
        }
        return 1;
    }

    /* 1. Embedding lookup for all prefill tokens */
    for (int32_t i = 0; i < n_tokens; i++) {
        float *dest = prompt_x + (size_t)i * dim;
        int32_t tid = tokens[i];
        if (m->tok_embeddings) {
            if (m->tok_embeddings_type == GGML_TYPE_F32) {
                const float *emb = (const float *)myds_metal_buffer_data(m->tok_embeddings);
                memcpy(dest, emb + (size_t)tid * dim, dim * sizeof(float));
            } else if (m->tok_embeddings_type == GGML_TYPE_F16) {
                const uint16_t *emb = (const uint16_t *)myds_metal_buffer_data(m->tok_embeddings);
                const uint16_t *row = emb + (size_t)tid * dim;
                for (uint32_t j = 0; j < dim; j++) {
                    uint32_t w = (uint32_t)row[j] << 16;
                    memcpy(&dest[j], &w, sizeof(float));
                }
            } else {
                memset(dest, 0, dim * sizeof(float));
            }
        } else {
            memset(dest, 0, dim * sizeof(float));
        }
    }

    uint32_t base_pos = ctx->pos;

    /* 2. Layer-major prefill: each layer's weights are loaded ONCE across the entire prompt */
    for (uint32_t l = 0; l < m->n_layers; l++) {
        for (int32_t c_start = 0; c_start < n_tokens; c_start += MYDS_PREFILL_CHUNK_SIZE) {
            if (cancel_flag && *cancel_flag) {
                free(prompt_x);
                free_prefill_chunk(chunk);
                return -2;
            }

            uint32_t chunk_len = (uint32_t)(n_tokens - c_start);
            if (chunk_len > MYDS_PREFILL_CHUNK_SIZE) chunk_len = MYDS_PREFILL_CHUNK_SIZE;
            uint32_t chunk_pos = base_pos + (uint32_t)c_start;

            /* Load chunk activations from prompt_x */
            memcpy(myds_metal_buffer_data(chunk->x_buf), prompt_x + (size_t)c_start * dim, (size_t)chunk_len * dim * sizeof(float));

            /* Pass chunk through layer l */
            if (!execute_prefill_chunk_layer(ctx, chunk, l, chunk_pos, chunk_len)) {
                free(prompt_x);
                free_prefill_chunk(chunk);
                return -1;
            }

            /* Store chunk updated activations back to prompt_x */
            memcpy(prompt_x + (size_t)c_start * dim, myds_metal_buffer_data(chunk->x_buf), (size_t)chunk_len * dim * sizeof(float));
        }
    }

    ctx->pos = base_pos + (uint32_t)n_tokens;
    if (ctx && ctx->cfg.temperature > 0.0f) {
        for (int32_t i = 0; i < n_tokens; i++) {
            ctx->rng_state = ctx->rng_state * 6364136223846793005ULL + 1ULL;
        }
    }
    free(prompt_x);
    free_prefill_chunk(chunk);
    return 1;
}

int myds_generate_ex(myds_context_t *ctx,
                     const char     *prompt,
                     myds_stream_callback_t cb,
                     void           *user_data,
                     myds_stop_reason_t *out_reason,
                     uint32_t       *out_prompt_tokens,
                     volatile int   *cancel_flag) {
    if (!ctx || !prompt) {
        if (out_reason) *out_reason = MYDS_STOP_ERROR;
        return 0;
    }
    myds_model_t *m = ctx->model;

    /* 1. Tokenize prompt with dynamic allocation based on max_seq_len (P1-2) */
    size_t prompt_alloc_tokens = ctx->cfg.max_seq_len > 4096 ? ctx->cfg.max_seq_len : 4096;
    int32_t *prompt_tokens = (int32_t *)malloc(prompt_alloc_tokens * sizeof(int32_t));
    if (!prompt_tokens) {
        MYDS_LOG_E("Out of memory allocating prompt token buffer");
        if (out_reason) *out_reason = MYDS_STOP_ERROR;
        return 0;
    }

    int32_t n_prompt = myds_tokenizer_encode(m->tokenizer, prompt, prompt_tokens, (uint32_t)prompt_alloc_tokens, true);
    if (n_prompt <= 0) {
        free(prompt_tokens);
        if (out_reason) *out_reason = MYDS_STOP_ERROR;
        return 0;
    }

    if (out_prompt_tokens) {
        *out_prompt_tokens = (uint32_t)n_prompt;
    }

    if ((uint32_t)n_prompt >= ctx->cfg.max_seq_len) {
        MYDS_LOG_E("Prompt length (%d tokens) exceeds maximum sequence context (%u)", n_prompt, ctx->cfg.max_seq_len);
        free(prompt_tokens);
        if (out_reason) *out_reason = MYDS_STOP_CTX_FULL;
        return 0;
    }

    ctx->pos = 0;
    int32_t cur_token = prompt_tokens[0];

    /* 2. Prefill phase (bypassing LM head & sampling for tokens 0..n_prompt-2) */
    if (n_prompt > 1) {
        int res = myds_prefill_tokens(ctx, prompt_tokens, n_prompt - 1, cancel_flag);
        if (res == -2) {
            free(prompt_tokens);
            if (out_reason) *out_reason = MYDS_STOP_CANCELLED;
            return 1;
        } else if (res < 0) {
            MYDS_LOG_E("Prefill phase failed");
            free(prompt_tokens);
            if (out_reason) *out_reason = MYDS_STOP_ERROR;
            return 0;
        }
    }
    cur_token = prompt_tokens[n_prompt - 1];
    free(prompt_tokens);

    /* 3. Decode phase */
    myds_stop_reason_t stop_reason = MYDS_STOP_LENGTH;
    uint32_t max_gen = ctx->cfg.max_new_tokens;
    int success = 1;

    for (uint32_t step = 0; step < max_gen; step++) {
        if (cancel_flag && *cancel_flag) {
            stop_reason = MYDS_STOP_CANCELLED;
            break;
        }
        int32_t next_token = myds_forward_token(ctx, cur_token, ctx->pos);
        if (next_token < 0) {
            MYDS_LOG_E("Generation step %u failed", step);
            stop_reason = MYDS_STOP_ERROR;
            success = 0;
            break;
        }
        ctx->pos++;

        if (next_token == m->tokenizer->eos_id) {
            stop_reason = MYDS_STOP_EOS;
            break;
        }

        const char *piece = myds_tokenizer_decode_piece(m->tokenizer, next_token);
        if (cb) {
            cb(piece, next_token, user_data);
        }

        cur_token = next_token;
        if (ctx->pos >= ctx->cfg.max_seq_len) {
            stop_reason = MYDS_STOP_CTX_FULL;
            break;
        }
    }

    if (out_reason) *out_reason = stop_reason;
    return success;
}

int myds_generate(myds_context_t *ctx, const char *prompt, myds_stream_callback_t cb, void *user_data) {
    return myds_generate_ex(ctx, prompt, cb, user_data, NULL, NULL, NULL);
}
