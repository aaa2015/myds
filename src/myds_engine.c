#include "myds.h"
#include "myds_metal.h"
#include "myds_gguf.h"
#include "myds_tokenizer.h"
#include "myds_v41.h"
#include "myds_align.h"
#include "myds_log.h"
#include "myds_cpu.h"
#include "myds_dspark.h"
#include "myds_pld.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <unistd.h>

bool myds_trace_f32(const char *stage, uint32_t layer, uint32_t pos,
                      const myds_metal_buffer_t *buffer, size_t count);

/* Cache the process policy atomically: different library contexts can first
 * generate concurrently. Environment changes require a process restart. */
static bool prefer_deeper_disk_prefix(void) {
    static atomic_int policy = ATOMIC_VAR_INIT(-1);
    int value = atomic_load_explicit(&policy, memory_order_acquire);
    if (value < 0) {
        const char *e = getenv("MYDS_KV_DSK_PREFER_DEEPER");
        const int chosen = (!e || atoi(e) != 0) ? 1 : 0;
        int expected = -1;
        if (atomic_compare_exchange_strong_explicit(&policy, &expected, chosen,
                                                   memory_order_acq_rel, memory_order_acquire))
            value = chosen;
        else value = expected;
    }
    return value != 0;
}

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} engine_block_q8_0_t;

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

static inline bool is_myds_prof_enabled(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        cached = (getenv("MYDS_PROF") != NULL) ? 1 : 0;
    }
    return cached == 1;
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
                float d = myds_fp16_to_fp32(row_blocks[b].d);
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
                sum += myds_fp16_to_fp32(row[i]) * xg[i];
            }
            out[r] = sum;
        }
    }
}

/* Helper: verify supported quantization types for GEMV operations */
static bool is_supported_quant_type(ggml_type type) {
    return (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K);
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

static const char *unimplemented_graph_feature(const gguf_file_t *gf) {
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const char *name = gf->tensors[i].name;
        if (strstr(name, "hc_attn_") || strstr(name, "hc_ffn_") ||
            strstr(name, "hc_head")) return "HC multi-stream residual connections";
        if (strstr(name, "engram")) return "Engram";
        if (strstr(name, "attn_compress")) return "compressed KV";
        if (strstr(name, ".indexer.")) return "KV indexer";
    }
    /* These metadata keys distinguish the full V4.1 graph even if its
     * required feature tensors have been omitted from an invalid file. */
    if (gguf_find_key(gf, "deepseek41.hc_mult") ||
        gguf_find_key(gf, "deepseek41.kv_source_layer_ids") ||
        gguf_find_key(gf, "deepseek41.engram.layer_ids")) return "DeepSeek V4.1 graph";
    return NULL;
}

/* ── 正确性门禁的「已验收」白名单 ─────────────────────────────────
 *
 * 上面这些 V4.1 特性原先一路被当作「未实现」，加载时直接拒绝，
 * 目的正确：宁可不跑，也不要【静默跳过算子】而输出错误结果。
 *
 * 但截至 2026-09-29，它们已全部实现，并通过了独立的三方逐位对照：
 *
 *   证据  docs/verification/2026-09-29-batched-prefill-parity.md
 *         docs/verification/2026-09-29-decode-optimization-verification.md
 *         docs/CORRECTNESS_STATUS.md（第三轮 "原生全阶段补齐"）
 *   方法  以 ds4 的 scalar 路径为【外部参照】，myds scalar/batched 为【内部一致】；
 *         覆盖 embedding / attn_norm / q_rope / attn_out / layer_out /
 *         route_* / ffnblock / logits（全 129280 值）等每位置 84 个阶段张量；
 *         3 种提示词 × 多位点（含跨块 511/512、1023/1024 与跨滑窗 127/128）
 *         × 层 0/14/39。
 *   结果  全部 max|Δ| = +0.0（664/672 项，其余为 myds 独有的 engram_out）
 *
 * ⇒ 默认【放行】。同时保留两个环境开关，出事时不用重编译即可回退：
 *      MYDS_DENY_V41=1   强制按旧行为拒绝（紧急刹车）
 *      MYDS_ALLOW_V41=1  强制放行（原语义；用于尚未验收的实验图）
 *      diagnostic 模式    始终放行（仅诊断用）
 *
 * ⚠️ 往白名单里加新特性之前，必须先有同等的逐位验收证据。
 */
static bool graph_feature_verified(const char *feature) {
    if (!feature) return true;
    return strcmp(feature, "HC multi-stream residual connections") == 0 ||
           strcmp(feature, "Engram") == 0 ||
           strcmp(feature, "compressed KV") == 0 ||
           strcmp(feature, "KV indexer") == 0 ||
           strcmp(feature, "DeepSeek V4.1 graph") == 0;
}

bool myds_validate_fp8_block_size(const gguf_file_t *gf, const char *arch) {
    if (!gf || !arch) return true;
    const char *keys[] = {
        "deepseek41.weight_block_size",
        "quantization_config.weight_block_size",
        "deepseek4.weight_block_size",
        "general.weight_block_size"
    };
    bool is_v41 = (strcmp(arch, "deepseek41") == 0 || strcmp(arch, "deepseek-v4.1") == 0);
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        const gguf_kv_t *kv = gguf_find_key(gf, keys[i]);
        if (!kv) continue;
        uint32_t b0 = 0, b1 = 0;
        if (kv->type == GGUF_TYPE_ARRAY && kv->value.val_arr.count >= 2 &&
            (kv->value.val_arr.elem_type == GGUF_TYPE_UINT32 || kv->value.val_arr.elem_type == GGUF_TYPE_INT32)) {
            const uint32_t *arr = (const uint32_t *)kv->value.val_arr.data;
            if (arr) { b0 = arr[0]; b1 = arr[1]; }
        } else if (kv->type == GGUF_TYPE_UINT32) {
            b0 = b1 = kv->value.val_u32;
        } else if (kv->type == GGUF_TYPE_INT32) {
            b0 = b1 = (uint32_t)kv->value.val_i32;
        }
        if (b0 > 0 && b1 > 0) {
            if (is_v41 && (b0 != 32 || b1 != 32)) {
                MYDS_LOG_E("DeepSeek V4.1 FP8 tile block size mismatch for key '%s': expected [32, 32], got [%u, %u]. "
                           "(128x128 is legacy V4 block size; using it with V4.1 causes silent dequantization corruption).",
                           keys[i], b0, b1);
                return false;
            }
            MYDS_LOG_I("FP8 tile block size validated for '%s': [%u, %u]", keys[i], b0, b1);
        }
    }
    return true;
}

static myds_model_t *model_load_internal(const char *model_path, bool diagnostic) {

    gguf_file_t *gf = gguf_open(model_path);
    if (!gf) {
        MYDS_LOG_E("Failed to open GGUF file '%s'", model_path);
        return NULL;
    }

    const char *missing_feature = unimplemented_graph_feature(gf);
    const bool gate_verified   = graph_feature_verified(missing_feature);
    const bool gate_force_on   = getenv("MYDS_ALLOW_V41") != NULL;   /* 强制放行 */
    const bool gate_force_off  = getenv("MYDS_DENY_V41")  != NULL;   /* 紧急刹车 */
    bool allowed = diagnostic || gate_force_on || (gate_verified && !gate_force_off);
    if (missing_feature && !allowed) {
        MYDS_LOG_E("Cannot run this model correctly: native graph has not passed the correctness gate for %s. "
                   "Refusing inference rather than silently skipping model operations. "
                   "See docs/CORRECTNESS_STATUS.md.", missing_feature);
        gguf_close(gf);
        return NULL;
    }
    if (missing_feature && !gate_verified) {
        MYDS_LOG_W("DIAGNOSTIC ONLY: unvalidated graph (%s); logits require independent comparison",
                   missing_feature);
    } else if (missing_feature && !gate_force_off) {
        MYDS_LOG_I("V4.1 native graph enabled: %s (verified — see docs/CORRECTNESS_STATUS.md)",
                   missing_feature);
    }
    if (!myds_metal_init("metal")) {
        MYDS_LOG_E("Failed to initialize Metal backend (check Apple Silicon GPU & shaders)");
        gguf_close(gf);
        return NULL;
    }

    MYDS_LOG_I("Opened GGUF model: %s (tensors: %llu, kv_pairs: %llu)",
               model_path, (unsigned long long)gf->tensor_count, (unsigned long long)gf->metadata_kv_count);

    myds_model_t *m = (myds_model_t *)calloc(1, sizeof(myds_model_t));
    if (!m) { gguf_close(gf); return NULL; }
    static atomic_uint_fast64_t s_model_load_counter = 0;
    uint64_t seq = (uint64_t)atomic_fetch_add(&s_model_load_counter, 1) + 1;
    uint64_t t = (uint64_t)time(NULL);
    uint64_t p = (uint64_t)getpid();
    m->load_generation = ((p << 32) ^ t ^ (seq * 0x9e3779b97f4a7c15ull));
    if (!m->load_generation) m->load_generation = seq;

    m->gf = gf;
    if (model_path) strncpy(m->model_path, model_path, sizeof(m->model_path) - 1);
    m->tokenizer = myds_tokenizer_load(gf);
    if (!m->tokenizer) {
        MYDS_LOG_E("Unable to load the model tokenizer");
        gguf_close(gf);
        free(m);
        return NULL;
    }

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

    if (!myds_validate_fp8_block_size(gf, arch)) {
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
    if (strcmp(arch, "deepseek41") == 0 || strcmp(arch, "deepseek-v4.1") == 0) {
        snprintf(key_buf, sizeof(key_buf), "%s.rms_norm_eps", arch);
        m->norm_eps = gguf_get_f32(gf, key_buf, m->norm_eps);
    }

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
    if (t_embd->n_dims != 2 || t_embd->dims[0] != m->dim ||
        t_embd->dims[1] != m->tokenizer->vocab_size) {
        MYDS_LOG_E("Embedding shape must match hidden dimension %u and tokenizer vocabulary %u",
                   m->dim, m->tokenizer->vocab_size);
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
    if (t_out_norm->type != GGML_TYPE_F32 || t_out_norm->n_dims != 1 ||
        t_out_norm->dims[0] != m->dim) {
        MYDS_LOG_E("Output norm must be a one-dimensional F32 tensor of length %u", m->dim);
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
        if (t_out->n_dims != 2 || t_out->dims[0] != m->dim ||
            t_out->dims[1] != m->vocab_size) {
            MYDS_LOG_E("LM head shape must match embedding dimension %u and vocabulary %u",
                       m->dim, m->vocab_size);
            goto fail;
        }
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
            /* 诊断 MYDS_QKV_ADJACENCY=1：attn_q_a 与 attn_kv 在 GGUF 里是否物理相邻。
             * 若相邻，一个 buffer 就能覆盖两者 ⇒ 单次 GEMV 覆盖 1792 行，省一次固定开销。
             * 判定标准：kv 的 data 起点 == q_a 的 data 起点 + q_a 的 size。 */
            /* MYDS_QKV_MERGE=1：把 attn_q_a（1280 行）与 attn_kv（512 行）拼成连续 1792 行。
             * 两者输入相同（都是 x_norm）、in_dim 相同（5120），且互不依赖 ⇒
             * decode 时一次 GEMV 覆盖两者，省掉一次调用的固定开销（实测 ~33 µs/层）。
             * 代价：每层多 4.92 MB 常驻（40 层合计 197 MB）。 */
            if (getenv("MYDS_QKV_MERGE") && tqa->type == tkv->type) {
                const size_t merged_bytes = tqa->size_bytes + tkv->size_bytes;
                myds_metal_buffer_t *mb = myds_metal_buffer_alloc(merged_bytes);
                myds_metal_buffer_t *ob = myds_metal_buffer_alloc(
                    ((size_t)1280u + 512u) * sizeof(float));
                if (mb && ob &&
                    myds_metal_buffer_copy(mb, 0, m->layers[i].attn_q_a, 0, tqa->size_bytes) &&
                    myds_metal_buffer_copy(mb, tqa->size_bytes, m->layers[i].attn_kv, 0, tkv->size_bytes)) {
                    m->layers[i].attn_qkv_merged = mb;
                    m->layers[i].attn_qkv_out = ob;
                } else {
                    if (mb) myds_metal_buffer_free(mb);
                    if (ob) myds_metal_buffer_free(ob);
                    fprintf(stderr, "myds_engine: QKV merge buffers failed at layer %u\n", i);
                }
            }
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

    if (missing_feature && !strcmp(m->arch, "deepseek41")) {
        m->v41 = myds_v41_model_create(m, model_path);
        if (!m->v41) goto fail;
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

    /* 模型已全部映射：装配 residency set 并启动保活心跳。
     * 放在这里（而不是首个请求）是为了把那一次性逐页校验付在加载期。 */
    myds_metal_residency_activate();

    return m;

fail:
    if (m) {
        myds_model_free(m);
    } else if (gf) {
        gguf_close(gf);
    }
    return NULL;
}

myds_model_t *myds_model_load(const char *model_path) {
    return model_load_internal(model_path, false);
}

/* Internal diagnostic harness only. The CLI/server always use the validated
 * loader above, so a trace cannot accidentally enable unsupported production. */
myds_model_t *myds_model_load_diagnostic(const char *model_path) {
    return model_load_internal(model_path, true);
}

/* synchronize has no timeout: its waits finish even when a command reports an
 * error. Check a second drain after any failure before releasing borrowed mmap
 * memory. Cleanup of invalid work is never successful inference or recovery. */
static void drain_gpu_before_release(const char *owner) {
    if (myds_metal_synchronize()) return;
    MYDS_LOG_E("%s teardown: GPU synchronization failed; draining prior submissions before releasing resources", owner);
    if (!myds_metal_synchronize()) {
        MYDS_LOG_E("%s teardown: prior GPU work also failed after blocking wait; releasing resources without recovery", owner);
    }
}

void myds_model_free(myds_model_t *m) {
    if (!m) return;
    /* All weight views borrow the GGUF mapping released below. */
    drain_gpu_before_release("model");
    myds_v41_model_free(m->v41);
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

static void free_prefill_chunk(myds_prefill_chunk_t *chunk);

myds_context_t *myds_context_create(myds_model_t *model, const myds_config_t *cfg) {
    if (!model) return NULL;

    myds_context_t *ctx = (myds_context_t *)calloc(1, sizeof(myds_context_t));
    if (!ctx) return NULL;
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

    ctx->kv_capacity = ctx->cfg.max_seq_len;
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
        if (!model->v41) {
        ctx->k_cache = (float **)calloc(model->n_layers, sizeof(float *));
        ctx->v_cache = NULL;
        size_t kv_layer_size = (size_t)ctx->cfg.max_seq_len * model->head_dim;
        for (uint32_t l = 0; l < model->n_layers; l++) {
            ctx->k_cache[l] = (float *)calloc(kv_layer_size, sizeof(float));
        }
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
    if (model->v41) {
        ctx->v41 = myds_v41_context_create(ctx);
        if (!ctx->v41) { myds_context_free(ctx); return NULL; }
        /* 低位区 KV 的 SSD 落盘：由服务端显式打开（MYDS_KV_DSK=<目录>）。
         * 不设就完全不开 —— 与 MYDS_PREFIX_CACHE 同一个道理：库的调用方（含单测）
         * 依赖「每次 generate 都是从位置 0 全量 prefill」。 */
        {
            const char *dsk = getenv("MYDS_KV_DSK");
            /* 2026-10-22: 除空串外，"0" 与 "none" 也表示关闭。
             * 原判断是 if(dsk && *dsk)，于是 MYDS_KV_DSK=0 会被当成目录名 "0"
             * —— 文档却承诺"=0 即不落盘"。两处不一致，这里以文档为准修代码。 */
            const bool dsk_off = !dsk || !*dsk ||
                                 strcmp(dsk, "0") == 0 || strcmp(dsk, "none") == 0;
            if (!dsk_off) (void)myds_v41_low_kv_enable(ctx->v41, dsk);
        }
    }
    return ctx;
}

/* ── 视觉：图片嵌入注入 ──
 * 为什么需要它：图片块的所有位置共用同一个 token id（129264），
 * 但每个位置的向量必须不同（图片格=编码器输出；行尾=image_newline；首尾=start/end）。
 * 词表查表做不到这件事 ⇒ 单独给一段位置挂向量。
 * 与 ds4 一致：ds4 也是推同一个 id，靠注入的向量区分。
 * 这里拷一份数据（3~4MB 量级），避免调用方还要管生命周期 —— 少一个出错点。 */
void myds_context_add_image_embeddings(myds_context_t *ctx, uint32_t start_pos,
                                       uint32_t count, const float *data, uint32_t dim,
                                       uint32_t image_token_id) {
    if (!ctx || !data || count == 0 || dim == 0) return;
    if (ctx->img_emb_n > 0 && ctx->img_emb_dim != dim) {
        MYDS_LOG_E("image embeddings: 向量宽度不一致（已有 %u，本次 %u），拒绝添加",
                   ctx->img_emb_dim, dim);
        return;
    }
    if (ctx->img_emb_n == ctx->img_emb_cap) {
        const uint32_t cap = ctx->img_emb_cap ? ctx->img_emb_cap * 2u : 4u;
        float **nd = (float **)realloc(ctx->img_emb_data, cap * sizeof(*nd));
        if (!nd) { MYDS_LOG_E("image embeddings: 区间表扩容失败"); return; }
        ctx->img_emb_data = nd;
        uint32_t *ns = (uint32_t *)realloc(ctx->img_emb_start, cap * sizeof(*ns));
        if (!ns) { MYDS_LOG_E("image embeddings: 区间表扩容失败"); return; }
        ctx->img_emb_start = ns;
        uint32_t *nc = (uint32_t *)realloc(ctx->img_emb_count, cap * sizeof(*nc));
        if (!nc) { MYDS_LOG_E("image embeddings: 区间表扩容失败"); return; }
        ctx->img_emb_count = nc;
        ctx->img_emb_cap = cap;
    }
    const size_t bytes = (size_t)count * dim * sizeof(float);
    float *copy = (float *)malloc(bytes);
    if (!copy) {
        MYDS_LOG_E("image embeddings: 无法分配 %.1f MB", (double)bytes / 1048576.0);
        return;
    }
    memcpy(copy, data, bytes);
    ctx->img_emb_data[ctx->img_emb_n] = copy;
    ctx->img_emb_start[ctx->img_emb_n] = start_pos;
    ctx->img_emb_count[ctx->img_emb_n] = count;
    ctx->img_emb_n++;
    ctx->img_emb_dim = dim;
    ctx->img_emb_token_id = image_token_id;
    MYDS_LOG_I("image embeddings: 区间 %u = 位置 [%u, %u)，每维 %u",
               ctx->img_emb_n, start_pos, start_pos + count, dim);
}

void myds_context_clear_image_embeddings(myds_context_t *ctx) {
    if (!ctx) return;
    for (uint32_t i = 0; i < ctx->img_emb_n; i++) free(ctx->img_emb_data[i]);
    free(ctx->img_emb_data);      ctx->img_emb_data = NULL;
    free(ctx->img_emb_start);     ctx->img_emb_start = NULL;
    free(ctx->img_emb_count);     ctx->img_emb_count = NULL;
    ctx->img_emb_n = 0;
    ctx->img_emb_cap = 0;
    ctx->img_emb_dim = 0;
}

void myds_context_free(myds_context_t *ctx) {
    if (!ctx) return;
    /* DSpark owns an mmap: retaining its MTLBuffer wrappers does not preserve
     * that backing memory. Drain before any destructor can unmap or free it. */
    drain_gpu_before_release("context");
    free(ctx->decode_cycles);
    free(ctx->pcache_tokens); ctx->pcache_tokens=NULL; ctx->pcache_len=ctx->pcache_cap=0;
    if (ctx->dspark) {
        myds_dspark_free(ctx->dspark);
        ctx->dspark = NULL;
    }
    myds_context_clear_image_embeddings(ctx);
    myds_v41_context_free(ctx->v41);
    free_prefill_chunk(ctx->prefill_chunk);
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
    /* Changing sampling options cannot grow the already allocated KV buffers. */
    if (ctx->cfg.max_seq_len == 0 || ctx->cfg.max_seq_len > ctx->kv_capacity) {
        ctx->cfg.max_seq_len = ctx->kv_capacity;
    }
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
    bool has_rep_pen = (rep_pen > 1.0f && prev_tokens && n_prev_tokens > 0);

    /* 1. Greedy argmax check (if temp <= 0) */
    if (temp <= 0.0f) {
        int32_t best_id = 0;
        float max_l = logits[0];
        if (!has_rep_pen) {
            for (uint32_t i = 1; i < vocab_size; i++) {
                if (logits[i] > max_l) {
                    max_l = logits[i];
                    best_id = (int32_t)i;
                }
            }
            return best_id;
        }
        for (size_t p = 0; p < n_prev_tokens; p++) {
            if (prev_tokens[p] == 0) {
                max_l = (max_l > 0.0f) ? (max_l / rep_pen) : (max_l * rep_pen);
                break;
            }
        }
        for (uint32_t i = 1; i < vocab_size; i++) {
            float l = logits[i];
            for (size_t p = 0; p < n_prev_tokens; p++) {
                if (prev_tokens[p] == (int32_t)i) {
                    l = (l > 0.0f) ? (l / rep_pen) : (l * rep_pen);
                    break;
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
    if (!has_rep_pen) {
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (logits[i] > max_l) max_l = logits[i];
        }
    } else {
        for (uint32_t i = 0; i < vocab_size; i++) {
            float l = logits[i];
            for (size_t p = 0; p < n_prev_tokens; p++) {
                if (prev_tokens[p] == (int32_t)i) {
                    l = (l > 0.0f) ? (l / rep_pen) : (l * rep_pen);
                    break;
                }
            }
            if (l > max_l) max_l = l;
        }
    }

    /* 3. Buffer acquisition and single-pass candidate collection (diff >= -20.0f) */
    float *probs = (ctx && ctx->sample_probs) ? ctx->sample_probs : (float *)malloc(vocab_size * sizeof(float));
    int32_t *indices = (ctx && ctx->sample_indices) ? ctx->sample_indices : (int32_t *)malloc(vocab_size * sizeof(int32_t));

    float inv_temp = 1.0f / temp;
    float sum_exp = 0.0f;
    uint32_t n_cands = 0;
    if (!has_rep_pen) {
        for (uint32_t i = 0; i < vocab_size; i++) {
            float diff = (logits[i] - max_l) * inv_temp;
            if (diff >= -20.0f) {
                float e = expf(diff);
                probs[n_cands] = e;
                indices[n_cands] = (int32_t)i;
                sum_exp += e;
                n_cands++;
            }
        }
    } else {
        for (uint32_t i = 0; i < vocab_size; i++) {
            float l = logits[i];
            for (size_t p = 0; p < n_prev_tokens; p++) {
                if (prev_tokens[p] == (int32_t)i) {
                    l = (l > 0.0f) ? (l / rep_pen) : (l * rep_pen);
                    break;
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

    /* quickselect 取出前 target_k 个再 qsort。
     * top_k 是显式上界；top_p（nucleus）要的是【累计质量刚好达到 top_p 的那一段】，
     * 固定截断到 1024 只在这段装得下时才精确 —— 所以 1024 现在只是【起点】，
     * 质量不够就成倍放大重来。旧行为是直接砍到 1024：分布偏平（或温度偏高）时
     * 会在一个被截断的集合上重新归一化，采样分布是错的
     * （tests/test_engine.c 里有一条扁平分布的用例专门盯这个）。 */
    const uint32_t cand_total = n_cands;
    const bool want_nucleus = (top_k == 0 && top_p > 0.0f && top_p < 1.0f);
    uint32_t target_k = want_nucleus ? (cand_total < 1024u ? cand_total : 1024u) : top_k;
    for (;;) {
        /* quickselect 是就地分区、不要求有序，所以扩大前缀时对整个候选集重来一次即可。 */
        const uint32_t cut = (target_k > 0 && target_k < cand_total) ? target_k : cand_total;
        if (cut < cand_total) quickselect_cand_desc(cands, 0, (int)cand_total - 1, (int)cut - 1);
        qsort(cands, cut, sizeof(myds_cand_t), compare_cand_desc);

        /* Apply top_p truncation */
        float cum = 0.0f;
        uint32_t keep = cut;
        if (top_p > 0.0f && top_p < 1.0f) {
            keep = 0;
            for (uint32_t i = 0; i < cut; i++) {
                cum += cands[i].prob;
                keep++;
                if (cum >= top_p) break;
            }
        }
        /* 当前前缀的累计质量还不到 top_p ⇒ nucleus 比它更大，放大前缀重来。
         * （放大到整个候选集仍不够时保留全部 —— 候选池自身的质量就不足 top_p。） */
        if (want_nucleus && cum < top_p && cut < cand_total) {
            uint32_t next = cut * 8u;
            if (next <= cut || next > cand_total) next = cand_total;
            target_k = next;
            continue;
        }
        n_cands = keep;
        break;
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
    bool do_ffn_prof = (l == 0 && is_myds_prof_enabled());
    if (do_ffn_prof) gettimeofday(&tf0, NULL);

    if (layer->ffn_gate_exps && layer->ffn_up_exps && layer->ffn_down_exps && layer->router_gate) {
        uint32_t n_routed = m->n_routed_experts;
        uint32_t n_active = m->n_activated_experts;
        uint32_t exp_dim = m->moe_expert_dim > 0 ? m->moe_expert_dim : 2304;

        /* Enqueue router_gate in Batch 2 (takes ffn_norm_out as input) */
        if (!myds_metal_gemv(ctx->router_logits, layer->router_gate, ctx->ffn_norm_out, n_routed, dim, layer->router_gate_type)) {
            MYDS_LOG_E("Forward failed: GEMV for router_gate at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }

        /* Signal router_event immediately after router_gate is queued */
        uint64_t router_event = 0;
        if (!myds_metal_signal_event(&router_event)) {
            MYDS_LOG_E("Forward failed: signal router_event at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }

        /* In the SAME command buffer, enqueue the entire shared expert: gate, up, swiglu, and down */
        uint64_t shexp_event = 0;
        if (layer->shared_gate && layer->shared_up && layer->shared_down && ctx->shared_out) {
            if (!myds_metal_gemv(ctx->shared_gate_out, layer->shared_gate, ctx->ffn_norm_out, exp_dim, dim, layer->shared_gate_type) ||
                !myds_metal_gemv(ctx->shared_up_out, layer->shared_up, ctx->ffn_norm_out, exp_dim, dim, layer->shared_up_type)) {
                MYDS_LOG_E("Forward failed: GEMV for shared_gate/up at layer %u failed", l);
                (void)myds_metal_commit_batch();
                return 0;
            }
            myds_metal_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out, ctx->shared_up_out, exp_dim);
            if (!myds_metal_gemv(ctx->shared_out, layer->shared_down, ctx->shared_swiglu_out, dim, exp_dim, layer->shared_down_type)) {
                MYDS_LOG_E("Forward failed: GEMV for shared_down at layer %u failed", l);
                (void)myds_metal_commit_batch();
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
            (void)myds_metal_commit_batch();
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
            if (!myds_metal_commit_batch()) return 0;
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
            (void)myds_metal_commit_batch();
            return 0;
        }
        myds_metal_swiglu(ctx->swiglu_out, ctx->gate_out, ctx->up_out, hidden_dim);
        if (!myds_metal_gemv(ctx->ffn_out, layer->ffn_down, ctx->swiglu_out, dim, hidden_dim, layer->ffn_down_type)) {
            MYDS_LOG_E("Forward failed: GEMV for ffn_down at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }
        myds_metal_add(ctx->x, ctx->x, ctx->ffn_out, dim);
    } else if (layer->shared_gate && layer->shared_up && layer->shared_down && ctx->shared_out) {
        uint32_t exp_dim = m->moe_expert_dim > 0 ? m->moe_expert_dim : 2304;
        if (!myds_metal_gemv(ctx->shared_gate_out, layer->shared_gate, ctx->ffn_norm_out, exp_dim, dim, layer->shared_gate_type) ||
            !myds_metal_gemv(ctx->shared_up_out, layer->shared_up, ctx->ffn_norm_out, exp_dim, dim, layer->shared_up_type)) {
            MYDS_LOG_E("Forward failed: GEMV for shared_gate/up at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }
        myds_metal_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out, ctx->shared_up_out, exp_dim);
        if (!myds_metal_gemv(ctx->shared_out, layer->shared_down, ctx->shared_swiglu_out, dim, exp_dim, layer->shared_down_type)) {
            MYDS_LOG_E("Forward failed: GEMV for shared_down at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }
        myds_metal_add(ctx->x, ctx->x, ctx->shared_out, dim);
    }

    if (do_ffn_prof) {
        if (!myds_metal_commit_batch()) return 0;
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
        bool do_prof = (l == 0 && is_myds_prof_enabled());
        if (do_prof) gettimeofday(&t0, NULL);

        myds_metal_begin_batch();
        if (layer->attn_norm) {
            myds_metal_rmsnorm(ctx->x_norm, ctx->x, layer->attn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(ctx->x_norm), xv, dim * sizeof(float));
        }

        if (!myds_metal_gemv(ctx->qr, layer->attn_q_a, ctx->x_norm, m->q_lora_rank, dim, layer->attn_q_a_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_q_a at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }

        if (!myds_metal_gemv(ctx->kv_act, layer->attn_kv, ctx->x_norm, m->head_dim, dim, layer->attn_kv_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_kv at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }

        /* GPU Query LoRA RMSNorm */
        if (!myds_metal_rmsnorm(ctx->qr_norm, ctx->qr, layer->attn_q_a_norm, m->q_lora_rank, m->norm_eps)) {
            MYDS_LOG_E("Forward failed: rmsnorm for qr_norm at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }

        /* Query up-projection: qr_norm (q_lora_rank) -> q (n_heads * head_dim) */
        if (!myds_metal_gemv(ctx->q, layer->attn_q_b, ctx->qr_norm, m->n_heads * m->head_dim, m->q_lora_rank, layer->attn_q_b_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_q_b at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }

        /* GPU Key/Value RMSNorm */
        if (!myds_metal_rmsnorm(ctx->kv_act, ctx->kv_act, layer->attn_kv_a_norm, m->head_dim, m->norm_eps)) {
            MYDS_LOG_E("Forward failed: rmsnorm for kv_act at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }
        if (!myds_metal_commit_batch()) return 0;
        if (do_prof) gettimeofday(&t1, NULL);

        if (!myds_trace_f32("attn_norm", l, pos, ctx->x_norm, dim) ||
            !myds_trace_f32("q_a_raw", l, pos, ctx->qr, m->q_lora_rank) ||
            !myds_trace_f32("q_a_norm", l, pos, ctx->qr_norm, m->q_lora_rank) ||
            !myds_trace_f32("q_b", l, pos, ctx->q, m->n_heads * m->head_dim) ||
            !myds_trace_f32("kv_norm", l, pos, ctx->kv_act, m->head_dim)) return 0;

        float *qv = (float *)myds_metal_buffer_data(ctx->q);
        /* V4.1 Q_B feeds RoPE directly; V4's per-head RMSNorm is absent. */
        if (strcmp(m->arch, "deepseek41") != 0 && strcmp(m->arch, "deepseek-v4.1") != 0)
            head_rms_norm_inplace(qv, m->n_heads, m->head_dim, m->norm_eps);

        float *kv_v = (float *)myds_metal_buffer_data(ctx->kv_act);
        if (do_prof) gettimeofday(&t2, NULL);

        /* Tail RoPE on Q (all heads) and KV (1 head) */
        myds_rope_tail_inplace(qv, m->n_heads, m->head_dim, m->qk_rope_head_dim, pos, m->freq_base, m->freq_scale, false);
        myds_rope_tail_inplace(kv_v, 1, m->head_dim, m->qk_rope_head_dim, pos, m->freq_base, m->freq_scale, false);
        if (!myds_trace_f32("q_rope", l, pos, ctx->q, m->n_heads * m->head_dim) ||
            !myds_trace_f32("kv_rope", l, pos, ctx->kv_act, m->head_dim)) return 0;

        /* Push to latent KV cache: 512 floats per token per layer */
        memcpy(ctx->k_cache[l] + (size_t)pos * m->head_dim, kv_v, m->head_dim * sizeof(float));

        /* Sink-aware Multi-Head Attention over latent KV cache */
        float *heads_v = (float *)myds_metal_buffer_data(ctx->attn_heads);
        const float *sinks_w = layer->attn_sinks ? (const float *)myds_metal_buffer_data(layer->attn_sinks) : NULL;
        layer_attention_sinks(heads_v, sinks_w, qv, ctx->k_cache[l], pos + 1, m->n_heads, m->head_dim);
        if (do_prof) gettimeofday(&t3, NULL);

        /* Inverse RoPE on attention output heads tail */
        myds_rope_tail_inplace(heads_v, m->n_heads, m->head_dim, m->qk_rope_head_dim, pos, m->freq_base, m->freq_scale, true);

        /* Batch 2: Attention output down-projection + up-projection + residual add + Pre-FFN RMSNorm + Router GEMV */
        myds_metal_begin_batch();
        uint32_t group_dim = (m->n_heads / m->n_out_groups) * m->head_dim;
        if (!myds_metal_attn_output_a(ctx->attn_low, layer->attn_output_a, ctx->attn_heads,
                                      m->n_out_groups, group_dim, m->out_group_rank, layer->attn_output_a_type)) {
            MYDS_LOG_E("Forward failed: attn_output_a at layer %u failed", l);
            (void)myds_metal_commit_batch();
            return 0;
        }
        if (do_prof) gettimeofday(&t4, NULL);

        if (!myds_metal_gemv(ctx->attn_out, layer->attn_output_b, ctx->attn_low, dim, m->n_out_groups * m->out_group_rank, layer->attn_output_b_type)) {
            MYDS_LOG_E("Forward failed: GEMV for attn_output_b at layer %u failed", l);
            (void)myds_metal_commit_batch();
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
                (void)myds_metal_commit_batch();
                return 0;
            }
        }
        if (layer->attn_k) {
            if (!myds_metal_gemv(ctx->k, layer->attn_k, ctx->x_norm, n_kv_heads * head_dim, dim, layer->attn_k_type)) {
                MYDS_LOG_E("Forward failed: GEMV for attn_k at layer %u failed", l);
                (void)myds_metal_commit_batch();
                return 0;
            }
        }
        if (layer->attn_v) {
            if (!myds_metal_gemv(ctx->v, layer->attn_v, ctx->x_norm, n_kv_heads * head_dim, dim, layer->attn_v_type)) {
                MYDS_LOG_E("Forward failed: GEMV for attn_v at layer %u failed", l);
                (void)myds_metal_commit_batch();
                return 0;
            }
        }
        if (!myds_metal_commit_batch()) return 0;

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
                (void)myds_metal_commit_batch();
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

/* Keep prefill and decode on one typed embedding path. F16 rows must be
 * converted as IEEE binary16; shifting their bits into float's high half
 * instead interprets them as BF16 and corrupts every activation. */
static bool embedding_lookup_row(const myds_model_t *m, int32_t token_id, float *dest) {
    if (!m->tok_embeddings || token_id < 0 || (uint32_t)token_id >= m->vocab_size) {
        MYDS_LOG_E("Embedding lookup failed for token %d (vocab %u)", token_id, m->vocab_size);
        return false;
    }
    const void *data = myds_metal_buffer_data(m->tok_embeddings);
    if (m->tok_embeddings_type == GGML_TYPE_F32) {
        const float *row = (const float *)data + (size_t)token_id * m->dim;
        memcpy(dest, row, m->dim * sizeof(float));
    } else if (m->tok_embeddings_type == GGML_TYPE_F16) {
        const uint16_t *row = (const uint16_t *)data + (size_t)token_id * m->dim;
        for (uint32_t i = 0; i < m->dim; i++) dest[i] = myds_fp16_to_fp32(row[i]);
    } else {
        MYDS_LOG_E("Embedding lookup failed: unsupported type %d", (int)m->tok_embeddings_type);
        return false;
    }
    return true;
}

/* Decimal CSV/all filter shared by every native graph's trace stages. */
static bool trace_selected(const char *filter, uint32_t value) {
    if (!filter || !*filter) return value == 0;
    if (!strcmp(filter, "all")) return true;
    while (*filter) {
        if (*filter < '0' || *filter > '9') return false;
        char *end = NULL;
        unsigned long n = strtoul(filter, &end, 10);
        if (n > UINT32_MAX || (*end && *end != ',')) return false;
        if (n == value) return true;
        if (!*end) break;
        filter = end + 1;
    }
    return false;
}

/* Opt-in numerical diagnostics, independent of sampling/text decoding. Files
 * contain native little-endian F32 on Apple Silicon. Never enabled by default. */
static atomic_int g_trace_checked = 0;
static bool g_trace_active_cached = false;

bool myds_trace_active(void) {
    int checked = atomic_load_explicit(&g_trace_checked, memory_order_relaxed);
    if (__builtin_expect(!checked, 0)) {
        const char *prefix = getenv("MYDS_TRACE_PREFIX");
        if (!prefix || !*prefix) prefix = getenv("MYDS_BATCH_TRACE_PREFIX");
        g_trace_active_cached = (prefix && *prefix);
        atomic_store_explicit(&g_trace_checked, 1, memory_order_relaxed);
    }
    return g_trace_active_cached;
}

bool myds_trace_f32(const char *stage, uint32_t layer, uint32_t pos,
                      const myds_metal_buffer_t *buffer, size_t count) {
    if (!myds_trace_active()) return true;
    const char *prefix = getenv("MYDS_TRACE_PREFIX");
    if (!prefix || !*prefix) prefix = getenv("MYDS_BATCH_TRACE_PREFIX");
    if (!prefix || !*prefix) return true;
    const char *layer_env = getenv("MYDS_TRACE_LAYER");
    const char *pos_env = getenv("MYDS_TRACE_POS");
    if (!trace_selected(layer_env, layer) || !trace_selected(pos_env, pos)) return true;
    if (!buffer || count > myds_metal_buffer_size(buffer) / sizeof(float)) return false;
    if (!myds_metal_commit_batch()) return false;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, stage, layer, pos);
    if (n < 0 || (size_t)n >= sizeof(path)) return false;
    FILE *f = fopen(path, "wb");
    if (!f) {
        MYDS_LOG_E("Cannot create trace '%s'", path);
        return false;
    }
    size_t written = fwrite(myds_metal_buffer_data((myds_metal_buffer_t *)buffer),
                            sizeof(float), count, f);
    int closed = fclose(f);
    return written == count && closed == 0;
}

static int32_t myds_forward_token_internal(myds_context_t *ctx, int32_t token_id, uint32_t pos, bool need_logits) {
    if (ctx->v41) return myds_v41_forward(ctx, token_id, pos, need_logits);
    myds_model_t *m = ctx->model;
    uint32_t dim = m->dim;

    /* 1. Embedding lookup */
    float *xv = (float *)myds_metal_buffer_data(ctx->x);
    if (!embedding_lookup_row(m, token_id, xv)) return -1;
    if (!myds_trace_f32("embedding", 0, pos, ctx->x, dim)) return -1;

    /* 2. Transformer layers */
    for (uint32_t l = 0; l < m->n_layers; l++) {
        if (!execute_transformer_layer(ctx, l, pos)) {
            return -1;
        }
        if (!myds_trace_f32("layer_out", l, pos, ctx->x, dim)) return -1;
        if (m->is_mla && !myds_trace_f32("attn_out", l, pos, ctx->attn_out, dim)) return -1;
    }

    if (!need_logits) {
        return myds_metal_commit_batch() ? 0 : -1;
    }

    /* 3. Final RMSNorm & LM Head Logits Batch */
    struct timeval t_lm0, t_lm1, t_sp0, t_sp1;
    bool do_head_prof = is_myds_prof_enabled();
    if (do_head_prof) gettimeofday(&t_lm0, NULL);

    myds_metal_begin_batch();
    if (m->output_norm) {
        myds_metal_rmsnorm(ctx->x_norm, ctx->x, m->output_norm, dim, m->norm_eps);
    } else {
        if (!myds_metal_commit_batch()) return -1;
        memcpy(myds_metal_buffer_data(ctx->x_norm), xv, dim * sizeof(float));
        myds_metal_begin_batch();
    }

    if (m->output) {
        if (!myds_metal_gemv(ctx->logits, m->output, ctx->x_norm, m->vocab_size, dim, m->output_type)) {
            MYDS_LOG_E("Forward failed: LM head GEMV failed");
            (void)myds_metal_commit_batch();
            return -1;
        }
    }
    if (!myds_metal_commit_batch()) return -1;
    if (!myds_trace_f32("logits", 0, pos, ctx->logits, m->vocab_size)) return -1;
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
    if (ctx && !ctx->v41 && ctx->decode_metrics_active) {
        ctx->decode_stats.target_forward_calls++;
        ctx->decode_stats.target_rows++;
    }
    return myds_forward_token_internal(ctx, token_id, pos, true);
}

#define MYDS_PREFILL_CHUNK_SIZE 512

struct myds_prefill_chunk {
    uint32_t capacity;
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
};

static myds_prefill_chunk_t *create_prefill_chunk(myds_model_t *model, uint32_t capacity) {
    if (!model || capacity == 0 || capacity > MYDS_PREFILL_CHUNK_SIZE) return NULL;
    myds_prefill_chunk_t *chunk = (myds_prefill_chunk_t *)calloc(1, sizeof(myds_prefill_chunk_t));
    if (!chunk) return NULL;
    uint32_t dim = model->dim;
    uint32_t exp_dim = model->moe_expert_dim > 0 ? model->moe_expert_dim : 2304;
    uint32_t n_routed = model->n_routed_experts > 0 ? model->n_routed_experts : 384;
    uint32_t n_active = model->n_activated_experts > 0 ? model->n_activated_experts : 6;
    size_t C = capacity;
    chunk->capacity = capacity;

    /* 1. Allocate contiguous chunk tensors */
    chunk->x_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    if (!chunk->x_buf) goto fail;
    chunk->x_norm_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    if (!chunk->x_norm_buf) goto fail;
    chunk->attn_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    if (!chunk->attn_out_buf) goto fail;
    chunk->ffn_norm_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    if (!chunk->ffn_norm_out_buf) goto fail;
    if (model->is_mla) {
        chunk->qr_buf = myds_metal_buffer_alloc(C * model->q_lora_rank * sizeof(float));
        if (!chunk->qr_buf) goto fail;
        chunk->qr_norm_buf = myds_metal_buffer_alloc(C * model->q_lora_rank * sizeof(float));
        if (!chunk->qr_norm_buf) goto fail;
        chunk->q_buf = myds_metal_buffer_alloc(C * model->n_heads * model->head_dim * sizeof(float));
        if (!chunk->q_buf) goto fail;
        chunk->kv_act_buf = myds_metal_buffer_alloc(C * model->head_dim * sizeof(float));
        if (!chunk->kv_act_buf) goto fail;
        chunk->attn_heads_buf = myds_metal_buffer_alloc(C * model->n_heads * model->head_dim * sizeof(float));
        if (!chunk->attn_heads_buf) goto fail;
        chunk->attn_low_buf = myds_metal_buffer_alloc(C * model->n_out_groups * model->out_group_rank * sizeof(float));
        if (!chunk->attn_low_buf) goto fail;
    }
    chunk->router_logits_buf = myds_metal_buffer_alloc(C * n_routed * sizeof(float));
    if (!chunk->router_logits_buf) goto fail;
    chunk->shared_gate_out_buf = myds_metal_buffer_alloc(C * exp_dim * sizeof(float));
    if (!chunk->shared_gate_out_buf) goto fail;
    chunk->shared_up_out_buf = myds_metal_buffer_alloc(C * exp_dim * sizeof(float));
    if (!chunk->shared_up_out_buf) goto fail;
    chunk->shared_swiglu_out_buf = myds_metal_buffer_alloc(C * exp_dim * sizeof(float));
    if (!chunk->shared_swiglu_out_buf) goto fail;
    chunk->shared_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    if (!chunk->shared_out_buf) goto fail;
    chunk->moe_out_buf = myds_metal_buffer_alloc(C * dim * sizeof(float));
    if (!chunk->moe_out_buf) goto fail;

    /* 2. Create subviews for per-token workers and allocate scratch buffers */
    for (size_t t = 0; t < C; t++) {
        chunk->x[t] = myds_metal_buffer_sub_view(chunk->x_buf, t * dim * sizeof(float), dim * sizeof(float));
        if (!chunk->x[t]) goto fail;
        chunk->x_norm[t] = myds_metal_buffer_sub_view(chunk->x_norm_buf, t * dim * sizeof(float), dim * sizeof(float));
        if (!chunk->x_norm[t]) goto fail;
        chunk->attn_out[t] = myds_metal_buffer_sub_view(chunk->attn_out_buf, t * dim * sizeof(float), dim * sizeof(float));
        if (!chunk->attn_out[t]) goto fail;
        chunk->ffn_norm_out[t] = myds_metal_buffer_sub_view(chunk->ffn_norm_out_buf, t * dim * sizeof(float), dim * sizeof(float));
        if (!chunk->ffn_norm_out[t]) goto fail;
        if (model->is_mla) {
            chunk->qr[t] = myds_metal_buffer_sub_view(chunk->qr_buf, t * model->q_lora_rank * sizeof(float), model->q_lora_rank * sizeof(float));
            if (!chunk->qr[t]) goto fail;
            chunk->qr_norm[t] = myds_metal_buffer_sub_view(chunk->qr_norm_buf, t * model->q_lora_rank * sizeof(float), model->q_lora_rank * sizeof(float));
            if (!chunk->qr_norm[t]) goto fail;
            chunk->q[t] = myds_metal_buffer_sub_view(chunk->q_buf, t * model->n_heads * model->head_dim * sizeof(float), model->n_heads * model->head_dim * sizeof(float));
            if (!chunk->q[t]) goto fail;
            chunk->kv_act[t] = myds_metal_buffer_sub_view(chunk->kv_act_buf, t * model->head_dim * sizeof(float), model->head_dim * sizeof(float));
            if (!chunk->kv_act[t]) goto fail;
            chunk->attn_heads[t] = myds_metal_buffer_sub_view(chunk->attn_heads_buf, t * model->n_heads * model->head_dim * sizeof(float), model->n_heads * model->head_dim * sizeof(float));
            if (!chunk->attn_heads[t]) goto fail;
            chunk->attn_low[t] = myds_metal_buffer_sub_view(chunk->attn_low_buf, t * model->n_out_groups * model->out_group_rank * sizeof(float), model->n_out_groups * model->out_group_rank * sizeof(float));
            if (!chunk->attn_low[t]) goto fail;
        }
        chunk->router_logits[t] = myds_metal_buffer_sub_view(chunk->router_logits_buf, t * n_routed * sizeof(float), n_routed * sizeof(float));
        if (!chunk->router_logits[t]) goto fail;
        chunk->shared_gate_out[t] = myds_metal_buffer_sub_view(chunk->shared_gate_out_buf, t * exp_dim * sizeof(float), exp_dim * sizeof(float));
        if (!chunk->shared_gate_out[t]) goto fail;
        chunk->shared_up_out[t] = myds_metal_buffer_sub_view(chunk->shared_up_out_buf, t * exp_dim * sizeof(float), exp_dim * sizeof(float));
        if (!chunk->shared_up_out[t]) goto fail;
        chunk->shared_swiglu_out[t] = myds_metal_buffer_sub_view(chunk->shared_swiglu_out_buf, t * exp_dim * sizeof(float), exp_dim * sizeof(float));
        if (!chunk->shared_swiglu_out[t]) goto fail;
        chunk->shared_out[t] = myds_metal_buffer_sub_view(chunk->shared_out_buf, t * dim * sizeof(float), dim * sizeof(float));
        if (!chunk->shared_out[t]) goto fail;
        chunk->moe_out[t] = myds_metal_buffer_sub_view(chunk->moe_out_buf, t * dim * sizeof(float), dim * sizeof(float));
        if (!chunk->moe_out[t]) goto fail;

        chunk->selected_ids[t] = myds_metal_buffer_alloc(n_active * sizeof(int32_t));
        if (!chunk->selected_ids[t]) goto fail;
        chunk->selected_weights[t] = myds_metal_buffer_alloc(n_active * sizeof(float));
        if (!chunk->selected_weights[t]) goto fail;
        chunk->moe_gate[t] = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        if (!chunk->moe_gate[t]) goto fail;
        chunk->moe_up[t] = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        if (!chunk->moe_up[t]) goto fail;
        chunk->moe_mid[t] = myds_metal_buffer_alloc(n_active * exp_dim * sizeof(float));
        if (!chunk->moe_mid[t]) goto fail;
        chunk->moe_down[t] = myds_metal_buffer_alloc((size_t)n_active * dim * sizeof(float));
        if (!chunk->moe_down[t]) goto fail;
    }
    return chunk;

fail:
    free_prefill_chunk(chunk);
    return NULL;
}

static void free_prefill_chunk(myds_prefill_chunk_t *chunk) {
    if (!chunk) return;
    for (uint32_t t = 0; t < chunk->capacity; t++) {
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
    bool                  q_head_norm;
} chunk_prep_job_t;

static void chunk_prep_worker(void *context, size_t t) {
    const chunk_prep_job_t *job = (const chunk_prep_job_t *)context;
    uint32_t pos = job->cur_pos + (uint32_t)t;
    float *qv = (float *)myds_metal_buffer_data(job->chunk->q[t]);
    if (job->q_head_norm) head_rms_norm_inplace(qv, job->n_heads, job->head_dim, job->norm_eps);

    float *kv_v = (float *)myds_metal_buffer_data(job->chunk->kv_act[t]);
    myds_rope_tail_inplace(qv, job->n_heads, job->head_dim, job->qk_rope_head_dim, pos, job->freq_base, job->freq_scale, false);
    myds_rope_tail_inplace(kv_v, 1, job->head_dim, job->qk_rope_head_dim, pos, job->freq_base, job->freq_scale, false);

    memcpy(job->ctx->k_cache[job->l] + (size_t)pos * job->head_dim, kv_v, job->head_dim * sizeof(float));
}

static void chunk_post_rope_worker(void *context, size_t t) {
    const chunk_prep_job_t *job = (const chunk_prep_job_t *)context;
    uint32_t pos = job->cur_pos + (uint32_t)t;
    float *heads_v = (float *)myds_metal_buffer_data(job->chunk->attn_heads[t]);
    myds_rope_tail_inplace(heads_v, job->n_heads, job->head_dim, job->qk_rope_head_dim, pos, job->freq_base, job->freq_scale, true);
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
    (void)dst_vecs;
    (void)src_vecs;
    if (force_gemv) {
        return myds_metal_gemv_batch(dst_buf, W, src_buf, d_out, d_in, chunk_len, type);
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
    bool do_prof = (l == 0 && cur_pos == 0 && is_myds_prof_enabled());
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
        (void)myds_metal_commit_batch();
        return 0;
    }
    myds_metal_barrier();

    /* 1.3 RMSNorm for QR and KV */
    for (uint32_t t = 0; t < chunk_len; t++) {
        if (!myds_metal_rmsnorm(chunk->qr_norm[t], chunk->qr[t], layer->attn_q_a_norm, m->q_lora_rank, m->norm_eps) ||
            !myds_metal_rmsnorm(chunk->kv_act[t], chunk->kv_act[t], layer->attn_kv_a_norm, m->head_dim, m->norm_eps)) {
            MYDS_LOG_E("Prefill chunk Batch 1 norm failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            (void)myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 1.4 Q_b projection */
    if (!chunk_linear_proj(chunk->q, chunk->q_buf, layer->attn_q_b, chunk->qr_norm, chunk->qr_norm_buf, m->n_heads * m->head_dim, m->q_lora_rank, chunk_len, layer->attn_q_b_type, !use_mm_qb)) {
        MYDS_LOG_E("Prefill chunk Batch 1 Q_b failed at layer %u", l);
        myds_metal_set_auto_barrier(1);
        (void)myds_metal_commit_batch();
        return 0;
    }
    if (!myds_metal_commit_batch()) {
        myds_metal_set_auto_barrier(1);
        return 0;
    }
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
        .q_head_norm = strcmp(m->arch, "deepseek41") != 0 && strcmp(m->arch, "deepseek-v4.1") != 0,
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
            (void)myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 2.2 Attn out B */
    if (!chunk_linear_proj(chunk->attn_out, chunk->attn_out_buf, layer->attn_output_b, chunk->attn_low, chunk->attn_low_buf, dim, m->n_out_groups * m->out_group_rank, chunk_len, layer->attn_output_b_type, !use_mm_outb)) {
        MYDS_LOG_E("Prefill chunk Batch 2 out_b failed at layer %u", l);
        myds_metal_set_auto_barrier(1);
        (void)myds_metal_commit_batch();
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
        (void)myds_metal_commit_batch();
        return 0;
    }
    if (layer->shared_gate && layer->shared_up && chunk->shared_out_buf) {
        if (!chunk_linear_proj(chunk->shared_gate_out, chunk->shared_gate_out_buf, layer->shared_gate, chunk->ffn_norm_out, chunk->ffn_norm_out_buf, exp_dim, dim, chunk_len, layer->shared_gate_type, !use_mm_sh_up) ||
            !chunk_linear_proj(chunk->shared_up_out, chunk->shared_up_out_buf, layer->shared_up, chunk->ffn_norm_out, chunk->ffn_norm_out_buf, exp_dim, dim, chunk_len, layer->shared_up_type, !use_mm_sh_up)) {
            MYDS_LOG_E("Prefill chunk shared_gate/up failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            (void)myds_metal_commit_batch();
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
    if (!myds_metal_commit_batch()) {
        myds_metal_set_auto_barrier(1);
        return 0;
    }
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
            (void)myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 3.2 Shared down */
    if (layer->shared_gate && layer->shared_up && layer->shared_down && chunk->shared_out_buf) {
        if (!chunk_linear_proj(chunk->shared_out, chunk->shared_out_buf, layer->shared_down, chunk->shared_swiglu_out, chunk->shared_swiglu_out_buf, dim, exp_dim, chunk_len, layer->shared_down_type, !use_mm_sh_down)) {
            MYDS_LOG_E("Prefill chunk shared_down failed at layer %u", l);
            myds_metal_set_auto_barrier(1);
            (void)myds_metal_commit_batch();
            return 0;
        }
    }
    myds_metal_barrier();

    /* 3.3 Final residual addition */
    if (layer->shared_gate && layer->shared_up && layer->shared_down && chunk->shared_out_buf) {
        myds_metal_add(chunk->moe_out_buf, chunk->moe_out_buf, chunk->shared_out_buf, chunk_len * dim);
        myds_metal_barrier();
    }
    myds_metal_add(chunk->x_buf, chunk->x_buf, chunk->moe_out_buf, chunk_len * dim);
    if (!myds_metal_commit_batch()) {
        myds_metal_set_auto_barrier(1);
        return 0;
    }
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

/* The batched implementation requires routed MLA in every layer. Mixed/dense
 * models retain the general sequential path instead of dispatching absent weights. */
static bool model_supports_chunk_prefill(const myds_model_t *m) {
    if (!m->is_mla || m->n_routed_experts == 0 || m->n_activated_experts == 0) return false;
    for (uint32_t l = 0; l < m->n_layers; l++) {
        const myds_layer_t *layer = &m->layers[l];
        if (!layer->attn_q_a || !layer->attn_q_a_norm || !layer->attn_q_b ||
            !layer->attn_kv || !layer->attn_kv_a_norm || !layer->attn_output_a ||
            !layer->attn_output_b || !layer->router_gate || !layer->ffn_gate_exps ||
            !layer->ffn_up_exps || !layer->ffn_down_exps) return false;
    }
    return true;
}

static int myds_prefill_tokens(myds_context_t *ctx, const int32_t *tokens, int32_t n_tokens, volatile int *cancel_flag) {
    if (n_tokens <= 0) return 1;
    myds_model_t *m = ctx->model;
    uint32_t dim = m->dim;

    /* For V4.1, use native batched prefill unless tracing or explicitly disabled. */
    if (m->v41) {
        if (getenv("MYDS_DISABLE_V41_PREFILL") || getenv("MYDS_TRACE_PREFIX")) {
            for (int32_t i = 0; i < n_tokens; i++) {
                if (cancel_flag && *cancel_flag) return -2;
                if (myds_forward_token_internal(ctx, tokens[i], ctx->pos, false) < 0) return -1;
                ctx->pos++;
            }
            return 1;
        }
        return myds_v41_prefill_tokens(ctx, tokens, (uint32_t)n_tokens, cancel_flag);
    }

    /* For very short prompts (<= 4 tokens), standard sequential forward has virtually zero overhead */
    if (n_tokens <= 4 || getenv("MYDS_TRACE_PREFIX") || !model_supports_chunk_prefill(m)) {
        for (int32_t i = 0; i < n_tokens; i++) {
            if (cancel_flag && *cancel_flag) return -2;
            if (myds_forward_token_internal(ctx, tokens[i], ctx->pos, false) < 0) return -1;
            ctx->pos++;
        }
        return 1;
    }

    if (cancel_flag && *cancel_flag) return -2;
    uint32_t requested_capacity = (uint32_t)n_tokens;
    if (requested_capacity > MYDS_PREFILL_CHUNK_SIZE) requested_capacity = MYDS_PREFILL_CHUNK_SIZE;
    if (ctx->prefill_capacity < requested_capacity) {
        /* Allocate before swapping so an allocation failure preserves the prior workspace. */
        myds_prefill_chunk_t *larger = create_prefill_chunk(m, requested_capacity);
        if (larger) {
            free_prefill_chunk(ctx->prefill_chunk);
            ctx->prefill_chunk = larger;
            ctx->prefill_capacity = requested_capacity;
        }
    }
    myds_prefill_chunk_t *chunk = ctx->prefill_chunk;
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
        if (!embedding_lookup_row(m, tokens[i], dest)) {
            free(prompt_x);
            return -1;
        }
    }

    uint32_t base_pos = ctx->pos;

    /* 2. Layer-major prefill improves weight locality across prompt chunks.
     * GEMV still reads weights for each token; only GEMM tiles share those loads. */
    for (uint32_t l = 0; l < m->n_layers; l++) {
        for (int32_t c_start = 0; c_start < n_tokens; c_start += (int32_t)chunk->capacity) {
            if (cancel_flag && *cancel_flag) {
                free(prompt_x);
                return -2;
            }

            uint32_t chunk_len = (uint32_t)(n_tokens - c_start);
            if (chunk_len > chunk->capacity) chunk_len = chunk->capacity;
            uint32_t chunk_pos = base_pos + (uint32_t)c_start;

            /* Load chunk activations from prompt_x */
            memcpy(myds_metal_buffer_data(chunk->x_buf), prompt_x + (size_t)c_start * dim, (size_t)chunk_len * dim * sizeof(float));

            /* Pass chunk through layer l */
            if (!execute_prefill_chunk_layer(ctx, chunk, l, chunk_pos, chunk_len)) {
                free(prompt_x);
                return -1;
            }

            /* Store chunk updated activations back to prompt_x */
            memcpy(prompt_x + (size_t)c_start * dim, myds_metal_buffer_data(chunk->x_buf), (size_t)chunk_len * dim * sizeof(float));
        }
    }

    ctx->pos = base_pos + (uint32_t)n_tokens;
    /* Preserve the existing random stream for reproducible generation. */
    if (ctx->cfg.temperature > 0.0f) {
        for (int32_t i = 0; i < n_tokens; i++) {
            ctx->rng_state = ctx->rng_state * 6364136223846793005ULL + 1ULL;
        }
    }
    free(prompt_x);
    return 1;
}

static double decode_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static bool decode_env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && atoi(value) != 0;
}

static void decode_cycle_finish(myds_context_t *ctx) {
    if (!ctx->decode_cycle_open) return;
    const uint64_t outputs = ctx->decode_stats.output_tokens - ctx->decode_cycle_output_start;
    const double elapsed = ctx->decode_stats.timing_enabled
        ? decode_now_ms() - ctx->decode_cycle_start_ms : 0.0;
    ctx->decode_stats.cycle_count++;
    if (ctx->decode_stats.timing_enabled && ctx->decode_cycles &&
        ctx->decode_stats.recorded_cycles < ctx->decode_cycle_capacity) {
        myds_decode_cycle_t *cycle = &ctx->decode_cycles[ctx->decode_stats.recorded_cycles++];
        cycle->elapsed_ms = elapsed;
        cycle->output_tokens = (uint32_t)outputs;
    }
    if (ctx->decode_cost_scheduler && elapsed > 0.0 && outputs > 0) {
        if (!ctx->decode_cycle_speculative) {
            const double per_token = elapsed / (double)outputs;
            ctx->decode_target_ms = ctx->decode_target_ms > 0.0
                ? 0.8 * ctx->decode_target_ms + 0.2 * per_token : per_token;
        } else if (ctx->decode_target_ms > 0.0) {
            if (elapsed >= ctx->decode_target_ms * (double)outputs) {
                if (++ctx->decode_costly_cycles >= 3) {
                    ctx->decode_cost_backoff = 16;
                    ctx->decode_costly_cycles = 0;
                }
            } else {
                ctx->decode_costly_cycles = 0;
            }
        }
    }
    ctx->decode_cycle_open = false;
}

static void decode_cycle_begin(myds_context_t *ctx) {
    decode_cycle_finish(ctx);
    ctx->decode_cycle_open = true;
    ctx->decode_cycle_start_ms = ctx->decode_stats.timing_enabled ? decode_now_ms() : 0.0;
    ctx->decode_cycle_output_start = ctx->decode_stats.output_tokens;
    ctx->decode_cycle_speculative = false;
    ctx->decode_pld_candidate = false;
}

static bool decode_cost_skip(myds_context_t *ctx) {
    if (!ctx->decode_cost_scheduler) return false;
    /* Establish one uncontaminated target cycle before comparing costs. */
    if (ctx->decode_target_ms <= 0.0) return true;
    if (!ctx->decode_cost_backoff) return false;
    ctx->decode_cost_backoff--;
    ctx->decode_stats.backoff_steps++;
    return true;
}

static void decode_emit(myds_context_t *ctx, myds_stream_callback_t cb,
                        myds_stream_bytes_callback_t bytes_cb, const char *piece,
                        size_t length, int32_t token, void *user_data) {
    ctx->decode_stats.output_tokens++;
    if (bytes_cb) bytes_cb(piece, length, token, user_data);
    else if (cb) cb(piece, token, user_data);
}

static uint32_t decode_pld_propose(myds_context_t *ctx, const int32_t *history,
                                  uint32_t length, const myds_pld_config_t *cfg,
                                  int32_t *draft) {
    const double start = ctx->decode_stats.timing_enabled ? decode_now_ms() : 0.0;
    ctx->decode_stats.pld_queries++;
    ctx->decode_cycle_speculative = true;
    const uint32_t count = myds_pld_propose(history, length, cfg, draft);
    if (ctx->decode_stats.timing_enabled) ctx->decode_stats.pld_ms += decode_now_ms() - start;
    return count;
}

/* Publish only the target inputs whose authoritative outputs the caller has
 * kept. A streaming callback may cancel after either accepted draft; the
 * borrowed first/second checkpoints are still valid until the next forward. */
static bool decode_linear_keep(myds_context_t *ctx,
    const myds_v41_spec_snapshot_t *snap, uint32_t evaluated, uint32_t keep,
    uint32_t pos, const int32_t *inputs) {
    if (keep > evaluated) return false;
    if (keep < evaluated) {
        const bool restored = keep ?
            myds_v41_spec_snapshot_restore_partial(ctx, snap, keep) :
            myds_v41_spec_snapshot_restore(ctx, snap);
        if (!restored) { ctx->pcache_len = 0; return false; }
    }
    ctx->pos = pos + keep;
    if (ctx->pcache_tokens && pos <= ctx->pcache_cap &&
        keep <= ctx->pcache_cap - pos) {
        for (uint32_t i = 0; i < keep; ++i) ctx->pcache_tokens[pos + i] = inputs[i];
        ctx->pcache_len = pos + keep;
    } else {
        ctx->pcache_len = 0;
    }
    return true;
}

static void decode_dspark_linear_backoff(myds_dspark_t *ds, uint32_t accepted) {
    static int threshold = -1;
    if (threshold < 0) {
        const char *e = getenv("MYDS_DSPARK_BACKOFF_THRESH");
        threshold = e ? atoi(e) : 6;
    }
    if (accepted) {
        ds->consecutive_misses = ds->backoff_counter = 0;
    } else {
        ds->consecutive_misses++;
        if (threshold > 0) {
            if (ds->consecutive_misses >= (uint32_t)threshold + 2u) ds->backoff_counter = 8;
            else if (ds->consecutive_misses >= (uint32_t)threshold) ds->backoff_counter = 4;
        }
    }
}

static int myds_generate_internal(myds_context_t *ctx,
                     const char     *prompt,
                     myds_stream_callback_t cb,
                     myds_stream_bytes_callback_t bytes_cb,
                     void           *user_data,
                     myds_stop_reason_t *out_reason,
                     uint32_t       *out_prompt_tokens,
                     volatile int   *cancel_flag) {
    if (!ctx) {
        if (out_reason) *out_reason = MYDS_STOP_ERROR;
        return 0;
    }
    myds_model_t *m = ctx->model;
    memset(&ctx->decode_stats, 0, sizeof(ctx->decode_stats));
    myds_dspark_reset_proposal_state(ctx->dspark);
    ctx->decode_metrics_active = false;
    ctx->decode_cycle_open = false;
    ctx->decode_target_ms = 0.0;
    ctx->decode_costly_cycles = ctx->decode_cost_backoff = 0;
    ctx->decode_cost_scheduler = decode_env_enabled("MYDS_SPEC_COST_SCHEDULER");
    const bool record_decode_cycles = decode_env_enabled("MYDS_DECODE_METRICS");
    ctx->decode_stats.timing_enabled = record_decode_cycles || ctx->decode_cost_scheduler;
    const uint32_t cycle_capacity = ctx->cfg.max_new_tokens < ctx->cfg.max_seq_len
        ? ctx->cfg.max_new_tokens : ctx->cfg.max_seq_len;
    if (record_decode_cycles && cycle_capacity > ctx->decode_cycle_capacity) {
        myds_decode_cycle_t *cycles = realloc(ctx->decode_cycles,
            (size_t)cycle_capacity * sizeof(*cycles));
        if (cycles) {
            ctx->decode_cycles = cycles;
            ctx->decode_cycle_capacity = cycle_capacity;
        }
    }
    if (!record_decode_cycles) {
        free(ctx->decode_cycles);
        ctx->decode_cycles = NULL;
        ctx->decode_cycle_capacity = 0;
    }
    if (!prompt) {
        if (out_reason) *out_reason = MYDS_STOP_ERROR;
        return 0;
    }

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

    /* ★ previous_response_id：客户端只发了【新增内容】，前缀要由引擎补上。
     * 直接用 token 而不是「把上次的文本重新拼一遍再分词」——后者要过一遍 tokenizer 往返，
     * 边界处的合并可能产生不同的 token，从而悄悄错位。pcache_tokens 就是那条精确序列。 */
    if (ctx->continue_from_cache) {
        ctx->continue_from_cache = false;   /* 一次性：不能影响后续请求 */
        const uint32_t pl = ctx->pcache_len;
        if (pl && ctx->pcache_tokens) {
            if ((size_t)pl + (size_t)n_prompt > prompt_alloc_tokens) {
                size_t need = (size_t)pl + (size_t)n_prompt;
                int32_t *nt = (int32_t *)realloc(prompt_tokens, need * sizeof(int32_t));
                if (!nt) {
                    MYDS_LOG_E("continuation: out of memory (%u cached + %d new)", pl, n_prompt);
                    free(prompt_tokens);
                    if (out_reason) *out_reason = MYDS_STOP_ERROR;
                    return 0;
                }
                prompt_tokens = nt; prompt_alloc_tokens = need;
            }
            /* ★ 必须先丢掉 delta 开头的 BOS：分词器会给每段文本与位一个 BOS，
             * 但这段文本是要接在【已有序列】后面的 ⇒ 那个 BOS 会落在序列中间，是错的。
             * 实测证据：不丢时续跑比全量重放【多 1 个 token】（2058 vs 2057），
             * 而两条路径的其它 token 逐位一致。 */
            bool bos_dropped = false;
            if (n_prompt > 0 && m->tokenizer && prompt_tokens[0] == m->tokenizer->bos_id) {
                memmove(prompt_tokens, prompt_tokens + 1, (size_t)(n_prompt - 1) * sizeof(int32_t));
                --n_prompt;
                bos_dropped = true;
                MYDS_LOG_I("previous_response_id: dropped the leading BOS from the delta "
                           "(it would sit mid-sequence)");
            }
            memmove(prompt_tokens + pl, prompt_tokens, (size_t)n_prompt * sizeof(int32_t));
            memcpy(prompt_tokens, ctx->pcache_tokens, (size_t)pl * sizeof(int32_t));
            n_prompt += (int32_t)pl;
            /* ★ 图像向量的区间也必须跟着搬家。
             * 它们由服务端按【本次请求那段文本】的下标登记（含 BOS 于 0），而这里
             * ① 丢掉了 delta 的 BOS、② 把整段挪到缓存前缀之后 ⇒ 坐标系变了两次。
             * 不搬的后果非常具体、也非常隐蔽：新图的向量落回了【上一次那张图】的位置，
             * 于是每一轮都识别出第一张图（实测连续三轮不同图片，答案全是第一张）。
             * 正确换算：delta 下标 i ⇒ pl - (丢过BOS ? 1 : 0) + i。 */
            const int32_t head_shift = (int32_t)pl - (bos_dropped ? 1 : 0);
            if (head_shift != 0 && ctx->img_emb_n) {
                for (uint32_t i = 0; i < ctx->img_emb_n; i++) {
                    ctx->img_emb_start[i] = (uint32_t)((int64_t)ctx->img_emb_start[i] + head_shift);
                }
                MYDS_LOG_I("previous_response_id: 图像向量区间后移 %d 位（%u 个区间）",
                           head_shift, ctx->img_emb_n);
            }
            MYDS_LOG_I("previous_response_id: %u cached + %d new = %d tokens",
                       pl, n_prompt - (int32_t)pl, n_prompt);
        } else {
            /* 状态已被后续请求换掉或推理失败清空 ⇒ 前缀不在。
             * 坚决报错，绝不静默把后缀当作全量输入运行。 */
            MYDS_LOG_E("previous_response_id: continuation requested but prefix cache is empty (pcache_len=%u)", pl);
            free(prompt_tokens);
            if (out_reason) *out_reason = MYDS_STOP_ERROR;
            return 0;
        }
    }

    /* ★ 图像向量区间的自检与纠正。
     *
     * 为什么需要它：区间是【服务端】按请求文本算的下标，而序列在【引擎】里可能被
     * 续接逻辑改过（丢 BOS、前置缓存前缀）。两个坐标系一旦不一致，图片向量就会贴到
     * 别的 token 上 —— 不报错，只是认错图，属于最难查的一类。上面已按公式换算过，
     * 这里再独立验一遍：判据来自图片块自身的性质（每个位置都是同一个图片 token），
     * 这也让正确位置可恢复。判据与恢复算法是纯函数，见 myds_align.h（有单测）。
     * 找不到就拒绝本轮 —— 宁可报错，也不要把图片向量贴到错的 token 上。 */
    if (ctx->img_emb_n && n_prompt > 0) {
        uint32_t scan_from = 0;
        for (uint32_t i = 0; i < ctx->img_emb_n; i++) {
            const uint32_t cnt = ctx->img_emb_count[i];
            const uint32_t declared = ctx->img_emb_start[i];
            const int64_t at = myds_image_range_resolve(prompt_tokens, (uint32_t)n_prompt,
                                                        ctx->img_emb_token_id, declared,
                                                        cnt, scan_from);
            if (at < 0) {
                MYDS_LOG_E("image embeddings: 区间 %u（声明位置 [%u,%u)，长 %u）"
                           "在序列里找不到对应的图片 token，拒绝本轮",
                           i + 1, declared, declared + cnt, cnt);
                free(prompt_tokens);
                if (out_reason) *out_reason = MYDS_STOP_ERROR;
                return 0;
            }
            if ((uint32_t)at != declared) {
                MYDS_LOG_W("image embeddings: 区间 %u 声明位置 [%u,%u) 与序列不符，"
                           "按实际 token 位置纠正为 [%lld,%lld)",
                           i + 1, declared, declared + cnt,
                           (long long)at, (long long)at + (long long)cnt);
                ctx->img_emb_start[i] = (uint32_t)at;
            }
            scan_from = (uint32_t)at + cnt;
        }
    }

    /* 诊断（临时）：把最终 token 序列的首尾打出来，用于比较「续跑」与「全量重放」
     * 两条路径是否逐位一致。MYDS_DEBUG_TAIL=1 开启。 */
    if (getenv("MYDS_DEBUG_TAIL")) {
        char head[256] = "", tail[256] = "";
        size_t ho = 0, to = 0;
        uint32_t nh = n_prompt < 8 ? (uint32_t)n_prompt : 8u;
        uint32_t nt = n_prompt < 12 ? (uint32_t)n_prompt : 12u;
        for (uint32_t i = 0; i < nh; ++i)
            ho += (size_t)snprintf(head + ho, sizeof(head) - ho, "%s%d", i ? "," : "", prompt_tokens[i]);
        for (uint32_t i = n_prompt - nt; i < (uint32_t)n_prompt; ++i)
            to += (size_t)snprintf(tail + to, sizeof(tail) - to, "%s%d", i == n_prompt - nt ? "" : ",", prompt_tokens[i]);
        MYDS_LOG_I("TOKDIAG n=%d head=[%s] tail=[%s]", n_prompt, head, tail);
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

    /* Decide capture policy before either prefill or prefix restoration.
     * The speculative loop always consumes the last prompt token with a
     * complete target forward before its first neural proposal; that step
     * primes fresh slots even when the reused prefix has no slot snapshot. */
    const bool serial_probe = decode_env_enabled("MYDS_SPEC_SERIAL_PROBE");
    const uint32_t dspark_depth = serial_probe ? 1u : myds_dspark_decode_depth(ctx);
    const bool spec_requested = !ctx->v41 || myds_v41_batch2_spec_requested(ctx) ||
        (ctx->dspark && dspark_depth == 2u);
    const bool use_spec = spec_requested || serial_probe;
    myds_dspark_capture_begin_request(ctx->dspark, use_spec && ctx->cfg.max_new_tokens > 1u);

    /* ── Prefix cache（服务端调度）──
     * 上一个请求留下的状态覆盖 token 序列 pcache_tokens[0..pcache_len)，位置与之一一对应。
     * 如果这次 prompt 是它的【严格续写】（公共前缀恰好等于它的全部长度），就从 pcache_len
     * 继续跑，跳过公共部分的 prefill —— 多轮对话的第二轮起，公共部分就是整段历史。
     * 为什么必须严格相等：engram 的状态是逐 token 的滚动链，只有已消费的序列完全一致时
     * 才对齐（回滚做不到）；而 decode 产出的 EOS 从未被喂入，所以缓存序列天然就是
     * 「prompt + 生成文本」，客户端下一轮重发的历史与之逐 token 对齐。
     * MYDS_PREFIX_CACHE=0 关闭（用于对照）。 */
    uint32_t resume_from = 0;
    /* 【默认关闭】：这是服务端调度优化，不是库语义 —— 库的调用方（含单测）
     * 依赖「每次 generate = 从位置 0 全量 prefill」。服务端在启动环境里
     * 显式打开（MYDS_PREFIX_CACHE=1）。 */
    static int pcache_on = -1;
    if (pcache_on < 0) {
        const char *e = getenv("MYDS_PREFIX_CACHE");
        pcache_on = (e && atoi(e) != 0) ? 1 : 0;
    }
    {
        bool snap_hit = false;
        uint32_t L = 0;                    /* 与【上一条请求序列】的公共前缀长度 */
        const bool have_prev = pcache_on && ctx->pcache_tokens && ctx->pcache_len;
        if (have_prev)
            while (L < ctx->pcache_len && L < (uint32_t)n_prompt &&
                   ctx->pcache_tokens[L] == prompt_tokens[L]) L++;

        /* ★ 每一次都【必须说话】：一个优化不生效时，沉默是最贵的输出。 */
        if (!pcache_on) {
            MYDS_LOG_I("prefix cache: off (MYDS_PREFIX_CACHE=0)");
        } else if (!have_prev) {
            MYDS_LOG_I("prefix cache: nothing cached yet (this is the first request)");
        } else if (L == ctx->pcache_len && (uint32_t)n_prompt > ctx->pcache_len) {
            resume_from = L;               /* 严格续写：上一条的整段都是新提示词的前缀 */
            MYDS_LOG_I("prefix cache: prompt %d tok, cached %u, common prefix %u -> resume from cached prefix",
                       n_prompt, ctx->pcache_len, L);
        } else if (L) {
            /* 不是续写，但公共前缀可能覆盖到某个【快照点】。窗口环只有 128 槽、事后无法回溯，
             * 所以 prefill 经过边界时就把状态存了下来 ⇒ 从那里续跑。
             * 典型场景：客户端每次提问都当新对话，只带系统提示词与工具定义（兄弟请求）。 */
            /* The last prompt token must still be consumed by decode. A
             * snapshot at the prompt end would make prefill length negative. */
            const uint32_t snapshot_limit = L < (uint32_t)n_prompt ? L : (uint32_t)n_prompt - 1u;
            uint32_t snap = ctx->v41 ? myds_v41_snapshot_best(ctx->v41, snapshot_limit) : 0;
            if (snap) {
                resume_from = snap; snap_hit = true;
                MYDS_LOG_I("prefix cache: prompt %d tok, cached %u, common prefix %u -> resume from snapshot "
                           "at %u (reuse %u tok, still to prefill %u)",
                           n_prompt, ctx->pcache_len, L, snap, snap, (uint32_t)n_prompt - 1u - snap);
            } else {
                MYDS_LOG_I("prefix cache: skipped - common prefix %u of cached %u, but no snapshot within it "
                           "(the state cannot be rewound, so this restarts from 0)", L, ctx->pcache_len);
            }
        } else {
            MYDS_LOG_I("prefix cache: skipped - no common prefix with the cached %u tokens", ctx->pcache_len);
        }

        /* Check if prompt matches persistent anchor snapshot (e.g. System Prompt) */
        if (pcache_on && ctx->v41 && resume_from == 0 && ctx->anchor_len > 0 &&
            (uint32_t)n_prompt > ctx->anchor_len) {
            if (memcmp(prompt_tokens, ctx->anchor_tokens, (size_t)ctx->anchor_len * sizeof(int32_t)) == 0) {
                resume_from = ctx->anchor_len;
                snap_hit = true;
                MYDS_LOG_I("prefix cache: prompt %d tok -> resume from ANCHOR snapshot at %u "
                           "(reuse %u tok, still to prefill %u)",
                           n_prompt, ctx->anchor_len, ctx->anchor_len,
                           (uint32_t)n_prompt - 1u - ctx->anchor_len);
            }
        }

        /* Compare metadata even when RAM has a shallow match. Loading only
         * a strictly deeper SSD entry avoids unnecessary payload I/O at ties.
         * The transactional loader preserves RAM state on every failure, so
         * the already-selected snapshot or strict continuation is a fallback.
         * MYDS_KV_DSK_PREFER_DEEPER=0 keeps the previous selection policy. */
        bool disk_loaded = false;
        struct timespec disk_restore_started = {0};
        if (pcache_on && ctx->v41 && (prefer_deeper_disk_prefix() || resume_from == 0)) {
            const uint32_t candidate = myds_v41_low_kv_probe_deeper(ctx->v41, prompt_tokens,
                                                                    resume_from, (uint32_t)n_prompt - 1u);
            if (candidate > resume_from) {
                const uint32_t memory = resume_from;
                clock_gettime(CLOCK_MONOTONIC, &disk_restore_started);
                MYDS_LOG_I("prefix cache: deeper DISK candidate %u > RAM %u; validating payload",
                           candidate, memory);
                const uint32_t disk = myds_v41_low_kv_load_deeper(ctx->v41, prompt_tokens,
                                                                 memory, candidate);
                if (disk > memory) {
                    resume_from = disk; snap_hit = true;
                    disk_loaded = true;
                    MYDS_LOG_I("prefix cache: prompt %d tok -> resume from DISK snapshot at %u "
                               "(RAM %u, reuse %u tok, still to prefill %u)",
                               n_prompt, disk, memory, disk, (uint32_t)n_prompt - 1u - disk);
                } else {
                    MYDS_LOG_W("prefix cache: DISK candidate %u unavailable; retaining RAM %u",
                               candidate, memory);
                }
            }
        }

        /* 快照续跑必须把状态装回去（engram tail + 各层窗口环）；装不回去就退回全量。 */
        if (snap_hit && (!ctx->v41 || !myds_v41_snapshot_restore(ctx->v41, resume_from))) {
            MYDS_LOG_W("prefix cache: snapshot restore at %u failed -> full prefill", resume_from);
            resume_from = 0;
        }
        if (disk_loaded) {
            struct timespec ended;
            clock_gettime(CLOCK_MONOTONIC, &ended);
            const double ms = (double)(ended.tv_sec - disk_restore_started.tv_sec) * 1e3 +
                (double)(ended.tv_nsec - disk_restore_started.tv_nsec) / 1e6;
            MYDS_LOG_I("prefix cache: DISK load+apply %.3f ms, selected %u tokens", ms, resume_from);
        }
        /* 清空：本次 prefill 会重建属于【本次序列】的点。否则可能用到「已被后续请求改写过的
         * 压缩 KV」所对应的旧点，把续跑接在错误的状态上。 */
        if (ctx->v41) myds_v41_snapshot_clear(ctx->v41, resume_from);

        /* 失配诊断：一旦没能复用，唯一有用的问题是「在第几个 token 分叉、两边分别是什么」。 */
        if (have_prev && resume_from == 0 && L < ctx->pcache_len) {
            {
                char why[640];
                size_t o = 0;
                #define APPEND_TOK(TAG, ARR, LEN, IDX)                                       \
                    do {                                                                     \
                        if (o + 80u >= sizeof(why)) break;                                   \
                        const uint32_t _idx = (IDX);                                         \
                        if (_idx >= (uint32_t)(LEN)) {                                       \
                            int _w = snprintf(why + o, sizeof(why) - o, "%s[%u]=<end> ", TAG, _idx); \
                            if (_w > 0) o += (size_t)_w;                                     \
                            break;                                                          \
                        }                                                                   \
                        int32_t _id = (int32_t)(ARR)[_idx];                                   \
                        const char *_t = (m->tokenizer && _id >= 0 &&                        \
                                          (uint32_t)_id < m->tokenizer->vocab_size &&        \
                                          m->tokenizer->vocab[_id].decoded)                  \
                            ? m->tokenizer->vocab[_id].decoded : "?";                        \
                        int _w = snprintf(why + o, sizeof(why) - o, "%s[%u]=%d:", TAG, _idx, _id); \
                        if (_w > 0) o += (size_t)_w;                                         \
                        for (const char *_q = _t; *_q && o + 6 < sizeof(why); ++_q) {        \
                            unsigned char _c = (unsigned char)*_q;                           \
                            if (_c == '\n') { memcpy(why + o, "\\n", 2); o += 2; }          \
                            else if (_c == '\r') { memcpy(why + o, "\\r", 2); o += 2; }      \
                            else if (_c < 0x20 || _c == 0x7f) {                              \
                                int _w2 = snprintf(why + o, sizeof(why) - o, "\\x%02x", _c); \
                                if (_w2 > 0) o += (size_t)_w2;                                \
                            } else why[o++] = (char)_c;                                      \
                        }                                                                    \
                        if (o < sizeof(why) - 1u) why[o++] = ' ';                              \
                    } while (0)
                uint32_t lo = L > 3 ? L - 3 : 0;
                for (uint32_t k = lo; k <= L && k < ctx->pcache_len && o + 80 < sizeof(why); ++k)
                    APPEND_TOK("c", ctx->pcache_tokens, ctx->pcache_len, k);
                APPEND_TOK("c", ctx->pcache_tokens, ctx->pcache_len,
                           L + 1 < ctx->pcache_len ? L + 1 : ctx->pcache_len);
                for (uint32_t k = lo; k <= L && k < (uint32_t)n_prompt && o + 80 < sizeof(why); ++k)
                    APPEND_TOK("p", prompt_tokens, n_prompt, k);
                APPEND_TOK("p", prompt_tokens, n_prompt,
                           L + 1 < (uint32_t)n_prompt ? L + 1 : (uint32_t)n_prompt);
                #undef APPEND_TOK
                why[o < sizeof(why) ? o : sizeof(why) - 1] = 0;
                MYDS_LOG_I("prefix cache divergence at %u: %s", L, why);
            }
        }
    }
    /* 缓存序列的落点：容量按配置的上下文一次要够（续跑时只追加，所以不重排前缀）。 */
    if (ctx->pcache_cap < ctx->cfg.max_seq_len) {
        int32_t *nt = (int32_t *)realloc(ctx->pcache_tokens, (size_t)ctx->cfg.max_seq_len * sizeof(int32_t));
        if (nt) { ctx->pcache_tokens = nt; ctx->pcache_cap = ctx->cfg.max_seq_len; }
        else { ctx->pcache_len = 0; }
    }
    if (ctx->pcache_tokens && ctx->pcache_cap >= (uint32_t)n_prompt) {
        memcpy(ctx->pcache_tokens, prompt_tokens, (size_t)n_prompt * sizeof(int32_t));
        ctx->pcache_len = (uint32_t)n_prompt;
    } else {
        ctx->pcache_len = 0;
    }

    ctx->pos = resume_from;
    int32_t cur_token = prompt_tokens[0];

    /* 2. Prefill phase (bypassing LM head & sampling for tokens 0..n_prompt-2) */
    /* 统计口径：prefill_tokens 只算【真正跑过的】token，复用掉的不算 —— 否则「缓存命中的轮次」
     * 会因为只 prefill 少量 token 而显示成极慢，那是把固定开销算进了速率里。 */
    ctx->last_prompt_tokens = (uint32_t)n_prompt;
    ctx->last_cached_tokens = resume_from;
    ctx->last_prefill_tokens = ((uint32_t)n_prompt - 1 > resume_from)
                                   ? (uint32_t)(n_prompt - 1) - resume_from : 0;
    ctx->last_prefill_seconds = 0.0;
    {
        struct timespec _pf0, _pf1;
        clock_gettime(CLOCK_MONOTONIC, &_pf0);
        int res = 1;
        const char *anch_env = getenv("MYDS_ANCHOR_SNAPSHOT");
        bool anchor_enabled = (anch_env == NULL || atoi(anch_env) != 0);

        if (pcache_on && anchor_enabled && ctx->v41 && resume_from == 0) {
            uint32_t anchor_pos = 0;
            for (uint32_t i = 0; i < (uint32_t)n_prompt - 1u; i++) {
                if (prompt_tokens[i] == 128803) { /* <｜User｜> */
                    anchor_pos = i + 1u;
                    break;
                }
            }
            if (anchor_pos >= 4u && anchor_pos < (uint32_t)n_prompt - 1u && anchor_pos <= 1024u &&
                (ctx->anchor_len == 0 || memcmp(prompt_tokens, ctx->anchor_tokens, (size_t)anchor_pos * sizeof(int32_t)) != 0)) {
                /* 阶段 1：先 prefill 到系统提示词锚点（pos=anchor_pos） */
                int res0 = myds_prefill_tokens(ctx, prompt_tokens, (int32_t)anchor_pos, cancel_flag);
                if (res0 == 1) {
                    if (myds_v41_snapshot_store_slot(ctx->v41, MYDS_V41_ANCHOR_SLOT, anchor_pos)) {
                        memcpy(ctx->anchor_tokens, prompt_tokens, (size_t)anchor_pos * sizeof(int32_t));
                        ctx->anchor_len = anchor_pos;
                        MYDS_LOG_I("prefix cache: captured persistent anchor snapshot at pos %u (slot %u)",
                                   anchor_pos, MYDS_V41_ANCHOR_SLOT);
                    }
                    /* 阶段 2：prefill 剩余的 user 消息 token */
                    res = myds_prefill_tokens(ctx, prompt_tokens + anchor_pos,
                                              n_prompt - 1 - (int32_t)anchor_pos, cancel_flag);
                } else {
                    res = res0;
                }
            } else {
                res = myds_prefill_tokens(ctx, prompt_tokens, n_prompt - 1, cancel_flag);
            }
        } else {
            res = myds_prefill_tokens(ctx, prompt_tokens + resume_from,
                                      n_prompt - 1 - (int32_t)resume_from, cancel_flag);
        }
        clock_gettime(CLOCK_MONOTONIC, &_pf1);
        ctx->last_prefill_seconds = (double)(_pf1.tv_sec - _pf0.tv_sec) +
                                    (double)(_pf1.tv_nsec - _pf0.tv_nsec) / 1e9;
        /* 中途失败/取消时，状态可能只推进了一部分 ⇒ 整条缓存作废（下次全量重跑）。
         * 绝不能留下「pcache_len 说已缓存 n_prompt、而 GPU 状态只到一半」的组合：
         * 那会让下一个请求从错误的位置续跑（直接撞 base_pos != next_position 的检查）。 */
        if (res != 1) {
            MYDS_LOG_W("prefix cache invalidated: prefill returned %d", res);
            ctx->pcache_len = 0;
            myds_dspark_capture_end_request(ctx->dspark, false);
            if (res == -2) {
                free(prompt_tokens);
                if (out_reason) *out_reason = MYDS_STOP_CANCELLED;
                return 1;
            }
            MYDS_LOG_E("Prefill phase failed");
            free(prompt_tokens);
            if (out_reason) *out_reason = MYDS_STOP_ERROR;
            return 0;
        }
    }
    /* 记下 prefill 末尾的状态，供下一个只共享前缀的请求续跑（续跑点为 0 token 时，
     * prefill 里的保存点不会执行，所以必须在这里显式存一次）。 */
    if (ctx->v41) (void)myds_v41_snapshot_store_now(ctx->v41, (uint32_t)n_prompt - 1u);
    /* 低位区快照落盘（内容没变则不重写 ⇒ 稳态零写盘）。
     * 必须在这里：下一行 prompt_tokens 就释放了，而哈希与快照位置都要用到它。 */
    if (ctx->v41) (void)myds_v41_low_kv_save(ctx->v41, prompt_tokens, (uint32_t)n_prompt);
    uint32_t max_gen = ctx->cfg.max_new_tokens;
    const char *pld_env_str = getenv("MYDS_PLD");
    /* A sequential verifier cannot amortize target weight reads. Preserve it
     * only as an explicit measurement/reference arm, avoiding draft and
     * snapshot work in ordinary V4.1 decode. */
    const bool pld_enabled = use_spec && (pld_env_str == NULL || atoi(pld_env_str) != 0);

    int32_t *pld_history = NULL;
    uint32_t pld_len = 0;
    uint32_t pld_cap = 0;
    if (pld_enabled && prompt_tokens && n_prompt > 0) {
        pld_cap = (uint32_t)n_prompt + max_gen + 16u;
        pld_history = (int32_t *)malloc(pld_cap * sizeof(int32_t));
        if (pld_history) {
            memcpy(pld_history, prompt_tokens, n_prompt * sizeof(int32_t));
            pld_len = (uint32_t)n_prompt;
        }
    }

    cur_token = prompt_tokens[n_prompt - 1];
    free(prompt_tokens);

    /* 3. Decode phase */
    myds_stop_reason_t stop_reason = MYDS_STOP_LENGTH;
    int success = 1;

    ctx->decode_metrics_active = true;
    if (ctx->dspark && use_spec) {
        uint32_t step = 0;
        /* V4.1 capture belongs to the last input the target consumed. The
         * emitted cur_token is its successor and has not entered the target
         * yet. Keep the consumed input independently of prefix-cache storage. */
        int32_t last_consumed_token = -1;
        while (step < max_gen) {
            if (cancel_flag && *cancel_flag) {
                stop_reason = MYDS_STOP_CANCELLED;
                break;
            }
            if (ctx->pos >= ctx->cfg.max_seq_len) {
                stop_reason = MYDS_STOP_CTX_FULL;
                break;
            }

            decode_cycle_begin(ctx);
            if (step == 0) {
                if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                    ctx->pcache_tokens[ctx->pos] = cur_token;
                    if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
                }
                int32_t next_token = myds_forward_token(ctx, cur_token, ctx->pos);
                if (next_token < 0 || (uint32_t)next_token >= m->tokenizer->vocab_size) {
                    ctx->pcache_len = 0;
                    MYDS_LOG_E("DSpark initial forward step failed at pos %u", ctx->pos);
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                last_consumed_token = cur_token;
                ctx->pos++;
                step++;
                if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = next_token;

                if (next_token == m->tokenizer->eos_id) {
                    stop_reason = MYDS_STOP_EOS;
                    break;
                }

                size_t piece_len = 0;
                const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, next_token, &piece_len);
                if (!piece) {
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                decode_emit(ctx, cb, bytes_cb, piece, piece_len, next_token, user_data);
                cur_token = next_token;
                continue;
            }

            /* Linear depth 2 needs room for the seed and both draft inputs.
             * Short tails retain the depth-1 verifier and its bonus output. */
            const bool linear_room = dspark_depth == 2u && max_gen - step >= 3u &&
                ctx->cfg.max_seq_len - ctx->pos >= 3u;
            uint32_t draft_cap = ctx->v41 ? (linear_room ? 2u : 1u) :
                (ctx->dspark ? ctx->dspark->block_size : 4);
            if (draft_cap > max_gen - step) draft_cap = max_gen - step;

            int32_t cand_a = -1, cand_b = -1;
            bool cand_b_pld = false;
            uint32_t neural_count = 0;

            const bool cost_skip = decode_cost_skip(ctx) || (!serial_probe && max_gen - step < 2);
            if (ctx->dspark && !cost_skip) {
                const uint64_t before_cycles = ctx->dspark->cycles;
                const uint64_t before_proposed = ctx->dspark->proposed_tokens;
                const bool backed_off = ctx->dspark->backoff_counter > 0;
                const double draft_start = ctx->decode_stats.timing_enabled ? decode_now_ms() : 0.0;
                uint32_t n_dspark = ctx->v41 ?
                    myds_dspark_propose_next(ctx->dspark, ctx, last_consumed_token, cur_token, draft_cap) :
                    myds_dspark_propose(ctx->dspark, ctx, cur_token, draft_cap);
                neural_count = n_dspark;
                ctx->decode_stats.draft_cycles += ctx->dspark->cycles - before_cycles;
                ctx->decode_stats.draft_proposed += ctx->dspark->proposed_tokens - before_proposed;
                if (backed_off) ctx->decode_stats.backoff_steps++;
                else ctx->decode_cycle_speculative = true;
                if (ctx->decode_stats.timing_enabled)
                    ctx->decode_stats.draft_ms += decode_now_ms() - draft_start;
                if (n_dspark > 0) {
                    cand_a = ctx->dspark->draft_tree_tokens[0];
                    if (dspark_depth == 1u && ctx->dspark->draft_tree_count >= 2) {
                        cand_b = ctx->dspark->draft_tree_tokens[1];
                    }
                }
            }

            if (!cost_skip && pld_history && pld_len >= 2) {
                if (cand_a < 0) {
                    int32_t pld_draft[MYDS_PLD_DEFAULT_MAX_DRAFT];
                    myds_pld_config_t pld_cfg = myds_pld_default_config(m->tokenizer->eos_id);
                    uint32_t n_pld = decode_pld_propose(ctx, pld_history, pld_len, &pld_cfg, pld_draft);
                    if (n_pld > 0) {
                        cand_a = pld_draft[0];
                        ctx->decode_pld_candidate = true;
                    }
                } else if (dspark_depth == 1u && cand_b < 0) {
                    int32_t pld_draft[MYDS_PLD_DEFAULT_MAX_DRAFT];
                    myds_pld_config_t pld_cfg = myds_pld_default_config(m->tokenizer->eos_id);
                    uint32_t n_pld = decode_pld_propose(ctx, pld_history, pld_len, &pld_cfg, pld_draft);
                    if (n_pld > 0 && pld_draft[0] != cand_a) {
                        cand_b = pld_draft[0];
                        cand_b_pld = true;
                    }
                }
            }

            /* Tree A/B are alternatives at the same position. The sequential
             * tail uses only real continuation drafts, or the single PLD A. */
            uint32_t n_draft = cand_a < 0 ? 0 :
                (ctx->decode_pld_candidate ? 1 : neural_count);

            if (cand_a < 0) {
                if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                    ctx->pcache_tokens[ctx->pos] = cur_token;
                    if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
                }
                int32_t next_token = myds_forward_token(ctx, cur_token, ctx->pos);
                if (next_token < 0 || (uint32_t)next_token >= m->tokenizer->vocab_size) {
                    ctx->pcache_len = 0;
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                last_consumed_token = cur_token;
                ctx->pos++;
                step++;
                if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = next_token;
                if (next_token == m->tokenizer->eos_id) {
                    stop_reason = MYDS_STOP_EOS;
                    break;
                }
                size_t piece_len = 0;
                const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, next_token, &piece_len);
                if (!piece) {
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                decode_emit(ctx, cb, bytes_cb, piece, piece_len, next_token, user_data);
                cur_token = next_token;
                continue;
            }

            struct timeval tv_ver0, tv_ver1;
            gettimeofday(&tv_ver0, NULL);

            myds_v41_spec_snapshot_t *snap = ctx->dspark ? ctx->dspark->spec_snap : NULL;
            bool snap_saved = false;
            if (snap) snap->valid = false;

            uint32_t accepted_in_cycle = 0;

            if (linear_room && snap && neural_count >= 2u && !ctx->decode_pld_candidate) {
                if (cancel_flag && *cancel_flag) {
                    stop_reason = MYDS_STOP_CANCELLED;
                    break;
                }
                const uint32_t spec_pos = ctx->pos;
                const int32_t seed = cur_token;
                const int32_t inputs[3] = { seed, ctx->dspark->draft_tokens[0],
                    ctx->dspark->draft_tokens[1] };
                int32_t out_tokens[3] = { -1, -1, -1 };
                const double verify_start = decode_now_ms();
                int ret = myds_v41_forward_linear_spec_ex(ctx, inputs, 3u,
                    spec_pos, out_tokens, snap);
                if (ret < 0 && myds_v41_linear_spec_retry_ready(ctx, snap)) {
                    /* The target has committed a complete restoration to the
                     * entry prefix. Recover only the original seed, without
                     * publishing any result from the failed three-row graph. */
                    out_tokens[0] = myds_forward_token(ctx, seed, spec_pos);
                    if (out_tokens[0] >= 0 && (uint32_t)out_tokens[0] < m->tokenizer->vocab_size &&
                        myds_dspark_capture_ready(ctx->dspark)) {
                        ret = 1;
                        ctx->decode_stats.depth2_fallbacks++;
                        ctx->decode_stats.batch2_fallbacks++;
                    }
                }
                ctx->decode_stats.verify_ms += decode_now_ms() - verify_start;
                bool valid_output = ret >= 1 && ret <= 3;
                for (int i = 0; valid_output && i < ret; ++i)
                    valid_output = out_tokens[i] >= 0 &&
                        (uint32_t)out_tokens[i] < m->tokenizer->vocab_size;
                if (!valid_output) {
                    if (ret >= 0 && snap->valid) (void)myds_v41_spec_snapshot_restore(ctx, snap);
                    ctx->pcache_len = 0;
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }

                uint32_t keep = 0;
                for (uint32_t i = 0; i < (uint32_t)ret; ++i) {
                    if (cancel_flag && *cancel_flag) {
                        stop_reason = MYDS_STOP_CANCELLED;
                        break;
                    }
                    if (step >= max_gen) break;
                    if (out_tokens[i] == m->tokenizer->eos_id) {
                        keep = i + 1u;
                        step++;
                        cur_token = out_tokens[i];
                        if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = out_tokens[i];
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }
                    size_t piece_len = 0;
                    const char *piece = myds_tokenizer_decode_bytes(m->tokenizer,
                        out_tokens[i], &piece_len);
                    if (!piece) {
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }
                    keep = i + 1u;
                    step++;
                    cur_token = out_tokens[i];
                    if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = out_tokens[i];
                    decode_emit(ctx, cb, bytes_cb, piece, piece_len, out_tokens[i], user_data);
                }
                const int32_t kept_inputs[3] = { seed, out_tokens[0], out_tokens[1] };
                if (!decode_linear_keep(ctx, snap, (uint32_t)ret, keep, spec_pos, kept_inputs)) {
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                } else if (keep) {
                    last_consumed_token = kept_inputs[keep - 1u];
                    /* An accepted draft is an input which remains consumed;
                     * the final authoritative output is the pending seed. */
                    accepted_in_cycle = keep - 1u;
                    ctx->dspark->accepted_tokens += accepted_in_cycle;
                    ctx->decode_stats.draft_accepted += accepted_in_cycle;
                }
                gettimeofday(&tv_ver1, NULL);
                ctx->dspark->verify_ms += (double)(tv_ver1.tv_sec - tv_ver0.tv_sec) * 1000.0 +
                    (double)(tv_ver1.tv_usec - tv_ver0.tv_usec) / 1000.0;
                if (success && stop_reason != MYDS_STOP_CANCELLED)
                    decode_dspark_linear_backoff(ctx->dspark, accepted_in_cycle);
                if (stop_reason == MYDS_STOP_EOS || stop_reason == MYDS_STOP_CANCELLED ||
                    !success || step >= max_gen) break;
                continue;
            }

            if (ctx->v41 && snap && max_gen - step >= 2 &&
                ctx->pos < ctx->cfg.max_seq_len && ctx->cfg.max_seq_len - ctx->pos >= 2) {
                if (cancel_flag && *cancel_flag) {
                    stop_reason = MYDS_STOP_CANCELLED;
                    break;
                }

                if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                    ctx->pcache_tokens[ctx->pos] = cur_token;
                    if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
                }

                int32_t tokens_to_forward[3];
                uint32_t n_fwd = 2;
                tokens_to_forward[0] = cur_token;
                tokens_to_forward[1] = cand_a;

                if (cand_b >= 0 && cand_b != cand_a) {
                    tokens_to_forward[2] = cand_b;
                    n_fwd = 3;
                }

                int32_t out_tokens[3] = { 0, 0, 0 };

                uint32_t spec_pos = ctx->pos;
                const double verify_start = decode_now_ms();
                int ret = myds_v41_forward_batch_spec_ex(ctx, tokens_to_forward, n_fwd, spec_pos, out_tokens, snap);
                ctx->decode_stats.verify_ms += decode_now_ms() - verify_start;
                snap_saved = snap->valid;
                if (ret < 0 || out_tokens[0] < 0 || (uint32_t)out_tokens[0] >= m->tokenizer->vocab_size) {
                    if (snap_saved) myds_v41_spec_snapshot_restore(ctx, snap);
                    ctx->pcache_len = 0;
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                /* A hit consumes the matched A/B input as its second row;
                 * a miss keeps only the original cur_token target row. */
                last_consumed_token = ret == 2 ? out_tokens[0] : cur_token;

                if (out_tokens[0] == m->tokenizer->eos_id) {
                    ctx->pos = spec_pos + 1u;
                    stop_reason = MYDS_STOP_EOS;
                    break;
                }
                const bool checked_pld = ctx->decode_pld_candidate ||
                    (cand_b_pld && out_tokens[0] != cand_a);
                const bool accepted_from_pld = ctx->decode_pld_candidate ||
                    (cand_b_pld && out_tokens[0] == cand_b);
                if (checked_pld) {
                    ctx->decode_stats.pld_cycles++;
                    ctx->decode_stats.pld_proposed++;
                }
                bool matched = (ret == 2);
                static int s_spec_diag_count = 0;
                static int s_spec_diag_enabled = -1;
                if (s_spec_diag_enabled < 0) {
                    const char *env = getenv("MYDS_DSPARK_DEBUG");
                    s_spec_diag_enabled = env ? atoi(env) : 0;
                }
                if (ctx->dspark && !ctx->decode_pld_candidate && (s_spec_diag_enabled || s_spec_diag_count < 15)) {
                    s_spec_diag_count++;
                    size_t cand_len = 0, cand_b_len = 0, truth_len = 0;
                    const char *cand_str = myds_tokenizer_decode_bytes(m->tokenizer, cand_a, &cand_len);
                    const char *cand_b_str = (cand_b >= 0 && cand_b != cand_a) ? myds_tokenizer_decode_bytes(m->tokenizer, cand_b, &cand_b_len) : NULL;
                    const char *truth_str = myds_tokenizer_decode_bytes(m->tokenizer, out_tokens[0], &truth_len);
                    if (cand_b_str) {
                        MYDS_LOG_I("[dspark-spec] step=%u cand_a=%d ('%.*s') cand_b=%d ('%.*s') truth=%d ('%.*s') -> %s (%s)",
                                   step,
                                   cand_a, (int)cand_len, cand_str ? cand_str : "",
                                   cand_b, (int)cand_b_len, cand_b_str,
                                   out_tokens[0], (int)truth_len, truth_str ? truth_str : "",
                                   matched ? "ACCEPT" : "REJECT",
                                   matched ? (out_tokens[0] == cand_a ? "hit:a" : "hit:b") : "miss");
                    } else {
                        MYDS_LOG_I("[dspark-spec] step=%u cand=%d ('%.*s') truth=%d ('%.*s') -> %s",
                                   step, cand_a, (int)(cand_str ? cand_len : 0), cand_str ? cand_str : "",
                                   out_tokens[0], (int)(truth_str ? truth_len : 0), truth_str ? truth_str : "",
                                   matched ? "ACCEPT" : "REJECT");
                    }
                }
                if (matched) {
                    if (accepted_from_pld) ctx->decode_stats.pld_accepted++;
                    else {
                        if (ctx->dspark) ctx->dspark->accepted_tokens++;
                        ctx->decode_stats.draft_accepted++;
                    }
                    if (!accepted_from_pld) accepted_in_cycle++;

                    // 1. 采纳第 1 个 Token
                    size_t piece_len0 = 0;
                    const char *piece0 = myds_tokenizer_decode_bytes(m->tokenizer, out_tokens[0], &piece_len0);
                    if (!piece0) {
                        if (snap_saved) myds_v41_spec_snapshot_restore(ctx, snap);
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }
                    decode_emit(ctx, cb, bytes_cb, piece0, piece_len0, out_tokens[0], user_data);

                    step++;
                    if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = out_tokens[0];

                    if (out_tokens[0] == m->tokenizer->eos_id) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, snap, 1);
                        ctx->pos = spec_pos + 1u;
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }

                    if (step >= max_gen) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, snap, 1);
                        ctx->pos = spec_pos + 1u;
                        break;
                    }

                    if (cancel_flag && *cancel_flag) {
                        if (snap_saved && !myds_v41_spec_snapshot_restore_partial(ctx, snap, 1)) {
                            ctx->pcache_len = 0;
                            stop_reason = MYDS_STOP_ERROR;
                            success = 0;
                        } else {
                            stop_reason = MYDS_STOP_CANCELLED;
                        }
                        ctx->pos = spec_pos + 1u;
                        break;
                    }

                    // 2. 采纳第 2 个 Token (权威模型在命中分支上推理出的真实输出)
                    if (out_tokens[1] < 0 || (uint32_t)out_tokens[1] >= m->tokenizer->vocab_size) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, snap, 1);
                        ctx->pos = spec_pos + 1u;
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }

                    if (ctx->pcache_tokens && (spec_pos + 1u) < ctx->pcache_cap) {
                        ctx->pcache_tokens[spec_pos + 1u] = out_tokens[0];
                        if (spec_pos + 2u > ctx->pcache_len) ctx->pcache_len = spec_pos + 2u;
                    }

                    if (out_tokens[1] == m->tokenizer->eos_id) {
                        ctx->pos = spec_pos + 2u;
                        step++;
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }

                    size_t piece_len1 = 0;
                    const char *piece1 = myds_tokenizer_decode_bytes(m->tokenizer, out_tokens[1], &piece_len1);
                    if (!piece1) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, snap, 1);
                        ctx->pos = spec_pos + 1u;
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }
                    decode_emit(ctx, cb, bytes_cb, piece1, piece_len1, out_tokens[1], user_data);

                    ctx->pos = spec_pos + 2u;
                    step++;
                    cur_token = out_tokens[1];
                    if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = out_tokens[1];

                    if (out_tokens[1] == m->tokenizer->eos_id) {
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }
                } else {
                    // 推测树未命中！保留第 1 个真值
                    size_t piece_len = 0;
                    const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, out_tokens[0], &piece_len);
                    if (!piece) {
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }
                    decode_emit(ctx, cb, bytes_cb, piece, piece_len, out_tokens[0], user_data);

                    ctx->pos = spec_pos + 1u;
                    step++;
                    cur_token = out_tokens[0];
                    if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = out_tokens[0];

                    if (out_tokens[0] == m->tokenizer->eos_id) {
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }
                }

                gettimeofday(&tv_ver1, NULL);
                if (ctx->dspark) {
                    ctx->dspark->verify_ms += (double)(tv_ver1.tv_sec - tv_ver0.tv_sec) * 1000.0 +
                                              (double)(tv_ver1.tv_usec - tv_ver0.tv_usec) / 1000.0;

                    static int s_backoff_thresh = -1;
                    if (__builtin_expect(s_backoff_thresh < 0, 0)) {
                        const char *e = getenv("MYDS_DSPARK_BACKOFF_THRESH");
                        s_backoff_thresh = e ? atoi(e) : 6;
                    }
                    if (!ctx->decode_pld_candidate && accepted_in_cycle == 0) {
                        ctx->dspark->consecutive_misses++;
                        if (s_backoff_thresh > 0) {
                            if (ctx->dspark->consecutive_misses >= (uint32_t)s_backoff_thresh + 2) {
                                ctx->dspark->backoff_counter = 8;
                            } else if (ctx->dspark->consecutive_misses >= (uint32_t)s_backoff_thresh) {
                                ctx->dspark->backoff_counter = 4;
                            }
                        }
                    } else if (!ctx->decode_pld_candidate) {
                        ctx->dspark->consecutive_misses = 0;
                        ctx->dspark->backoff_counter = 0;
                    }
                }

                if (stop_reason == MYDS_STOP_EOS || stop_reason == MYDS_STOP_CANCELLED || !success || step >= max_gen) {
                    break;
                }
                continue;
            }

            if (ctx->decode_pld_candidate) ctx->decode_stats.pld_cycles++;
            for (uint32_t d = 0; d < n_draft; d++) {
                if (cancel_flag && *cancel_flag) {
                    stop_reason = MYDS_STOP_CANCELLED;
                    if (snap_saved && d == 0) {
                        myds_v41_spec_snapshot_restore(ctx, snap);
                    }
                    break;
                }
                if (ctx->pos >= ctx->cfg.max_seq_len) {
                    stop_reason = MYDS_STOP_CTX_FULL;
                    break;
                }

                if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                    ctx->pcache_tokens[ctx->pos] = cur_token;
                    if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
                }

                int32_t true_next = myds_forward_token(ctx, cur_token, ctx->pos);
                if (true_next < 0 || (uint32_t)true_next >= m->tokenizer->vocab_size) {
                    if (snap_saved) {
                        myds_v41_spec_snapshot_restore(ctx, snap);
                    }
                    ctx->pcache_len = 0;
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                last_consumed_token = cur_token;
                ctx->pos++;
                step++;

                if (pld_history && pld_len < pld_cap) pld_history[pld_len++] = true_next;
                const int32_t candidate = d == 0 ? cand_a : ctx->dspark->draft_tokens[d];
                const bool matched = true_next == candidate;
                if (ctx->decode_pld_candidate) ctx->decode_stats.pld_proposed++;
                if (matched) {
                    if (ctx->decode_pld_candidate) ctx->decode_stats.pld_accepted++;
                    else {
                        ctx->dspark->accepted_tokens++;
                        ctx->decode_stats.draft_accepted++;
                        accepted_in_cycle++;
                    }
                }

                if (true_next == m->tokenizer->eos_id) {
                    stop_reason = MYDS_STOP_EOS;
                    break;
                }

                size_t piece_len = 0;
                const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, true_next, &piece_len);
                if (!piece) {
                    if (snap_saved) {
                        myds_v41_spec_snapshot_restore(ctx, snap);
                    }
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                decode_emit(ctx, cb, bytes_cb, piece, piece_len, true_next, user_data);

                cur_token = true_next;

                if (!matched) {
                    break;
                }
                if (step >= max_gen) break;
            }

            gettimeofday(&tv_ver1, NULL);
            ctx->dspark->verify_ms += (double)(tv_ver1.tv_sec - tv_ver0.tv_sec) * 1000.0 +
                                      (double)(tv_ver1.tv_usec - tv_ver0.tv_usec) / 1000.0;

            static int s_backoff_thresh_old = -1;
            if (__builtin_expect(s_backoff_thresh_old < 0, 0)) {
                const char *e = getenv("MYDS_DSPARK_BACKOFF_THRESH");
                s_backoff_thresh_old = e ? atoi(e) : 6;
            }
            if (!ctx->decode_pld_candidate && accepted_in_cycle == 0) {
                ctx->dspark->consecutive_misses++;
                if (s_backoff_thresh_old > 0) {
                    if (ctx->dspark->consecutive_misses >= (uint32_t)s_backoff_thresh_old + 2) {
                        ctx->dspark->backoff_counter = 8;
                    } else if (ctx->dspark->consecutive_misses >= (uint32_t)s_backoff_thresh_old) {
                        ctx->dspark->backoff_counter = 4;
                    }
                }
            } else if (!ctx->decode_pld_candidate) {
                ctx->dspark->consecutive_misses = 0;
                ctx->dspark->backoff_counter = 0;
            }

            if (stop_reason == MYDS_STOP_EOS || stop_reason == MYDS_STOP_CANCELLED || !success) {
                break;
            }
        }
    } else if (pld_enabled && pld_history) {
        myds_v41_spec_snapshot_t *pld_snap = (myds_v41_spec_snapshot_t *)calloc(1, sizeof(myds_v41_spec_snapshot_t));
        uint32_t step = 0;
        uint64_t pld_cycles = 0;
        uint64_t pld_proposed = 0;
        uint64_t pld_accepted = 0;

        while (step < max_gen) {
            if (cancel_flag && *cancel_flag) {
                stop_reason = MYDS_STOP_CANCELLED;
                break;
            }
            if (ctx->pos >= ctx->cfg.max_seq_len) {
                stop_reason = MYDS_STOP_CTX_FULL;
                break;
            }

            decode_cycle_begin(ctx);
            int32_t draft_tokens[MYDS_PLD_DEFAULT_MAX_DRAFT];
            myds_pld_config_t pld_cfg = myds_pld_default_config(m->tokenizer->eos_id);
            if (pld_cfg.max_draft > max_gen - step) pld_cfg.max_draft = max_gen - step;
            uint32_t n_draft = 0;
            if (!decode_cost_skip(ctx) && (serial_probe || max_gen - step >= 2) && pld_len >= 2) {
                n_draft = decode_pld_propose(ctx, pld_history, pld_len, &pld_cfg, draft_tokens);
            }
            ctx->decode_pld_candidate = n_draft > 0;

            if (n_draft == 0) {
                if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                    ctx->pcache_tokens[ctx->pos] = cur_token;
                    if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
                }
                int32_t next_token = myds_forward_token(ctx, cur_token, ctx->pos);
                if (next_token < 0 || (uint32_t)next_token >= m->tokenizer->vocab_size) {
                    ctx->pcache_len = 0;
                    MYDS_LOG_E("Generation step %u failed", step);
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                ctx->pos++;
                step++;
                if (pld_len < pld_cap) pld_history[pld_len++] = next_token;

                if (next_token == m->tokenizer->eos_id) {
                    stop_reason = MYDS_STOP_EOS;
                    break;
                }

                size_t piece_len = 0;
                const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, next_token, &piece_len);
                if (!piece) {
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                decode_emit(ctx, cb, bytes_cb, piece, piece_len, next_token, user_data);

                cur_token = next_token;
                continue;
            }

            pld_cycles++;
            ctx->decode_stats.pld_cycles++;

            if (ctx->v41 && pld_snap && n_draft >= 1 && max_gen - step >= 2 &&
                ctx->pos < ctx->cfg.max_seq_len && ctx->cfg.max_seq_len - ctx->pos >= 2) {
                pld_proposed++;
                ctx->decode_stats.pld_proposed++;
                pld_snap->valid = false;
                bool snap_saved = false;

                if (cancel_flag && *cancel_flag) {
                    stop_reason = MYDS_STOP_CANCELLED;
                    break;
                }

                // 预记录当前 token 至 prefix cache
                if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                    ctx->pcache_tokens[ctx->pos] = cur_token;
                    if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
                }

                int32_t tokens_to_forward[2] = { cur_token, draft_tokens[0] };
                int32_t out_tokens[2] = { 0, 0 };

                uint32_t pld_pos = ctx->pos;
                const double verify_start = decode_now_ms();
                int ret = myds_v41_forward_batch_spec_ex(ctx, tokens_to_forward, 2, pld_pos, out_tokens, pld_snap);
                ctx->decode_stats.verify_ms += decode_now_ms() - verify_start;
                snap_saved = pld_snap->valid;
                if (ret < 0 || out_tokens[0] < 0 || (uint32_t)out_tokens[0] >= m->tokenizer->vocab_size) {
                    if (snap_saved) myds_v41_spec_snapshot_restore(ctx, pld_snap);
                    ctx->pcache_len = 0;
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }

                if (out_tokens[0] == m->tokenizer->eos_id) {
                    ctx->pos = pld_pos + 1u;
                    stop_reason = MYDS_STOP_EOS;
                    break;
                }
                bool matched = (ret == 2);
                if (matched) {
                    // 第 1 个草稿 Token 命中！
                    pld_accepted++;
                    ctx->decode_stats.pld_accepted++;

                    // 1. 采纳第 1 个 Token (draft_tokens[0] / out_tokens[0])
                    if (pld_len < pld_cap) pld_history[pld_len++] = out_tokens[0];

                    size_t piece_len0 = 0;
                    const char *piece0 = myds_tokenizer_decode_bytes(m->tokenizer, out_tokens[0], &piece_len0);
                    if (!piece0) {
                        if (snap_saved) myds_v41_spec_snapshot_restore(ctx, pld_snap);
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }
                    decode_emit(ctx, cb, bytes_cb, piece0, piece_len0, out_tokens[0], user_data);

                    step++;

                    if (out_tokens[0] == m->tokenizer->eos_id) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, pld_snap, 1);
                        ctx->pos = pld_pos + 1u;
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }

                    if (step >= max_gen) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, pld_snap, 1);
                        ctx->pos = pld_pos + 1u;
                        break;
                    }

                    if (cancel_flag && *cancel_flag) {
                        if (snap_saved && !myds_v41_spec_snapshot_restore_partial(ctx, pld_snap, 1)) {
                            ctx->pcache_len = 0;
                            stop_reason = MYDS_STOP_ERROR;
                            success = 0;
                        } else {
                            stop_reason = MYDS_STOP_CANCELLED;
                        }
                        ctx->pos = pld_pos + 1u;
                        break;
                    }

                    // 2. 采纳第 2 个 Token (out_tokens[1]，权威模型在真值 draft_tokens[0] 上推理出的真实输出)
                    if (out_tokens[1] < 0 || (uint32_t)out_tokens[1] >= m->tokenizer->vocab_size) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, pld_snap, 1);
                        ctx->pos = pld_pos + 1u;
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }

                    if (ctx->pcache_tokens && (pld_pos + 1u) < ctx->pcache_cap) {
                        ctx->pcache_tokens[pld_pos + 1u] = out_tokens[0];
                        if (pld_pos + 2u > ctx->pcache_len) ctx->pcache_len = pld_pos + 2u;
                    }
                    if (pld_len < pld_cap) pld_history[pld_len++] = out_tokens[1];

                    if (out_tokens[1] == m->tokenizer->eos_id) {
                        ctx->pos = pld_pos + 2u;
                        step++;
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }

                    size_t piece_len1 = 0;
                    const char *piece1 = myds_tokenizer_decode_bytes(m->tokenizer, out_tokens[1], &piece_len1);
                    if (!piece1) {
                        if (snap_saved) myds_v41_spec_snapshot_restore_partial(ctx, pld_snap, 1);
                        ctx->pos = pld_pos + 1u;
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }
                    decode_emit(ctx, cb, bytes_cb, piece1, piece_len1, out_tokens[1], user_data);

                    ctx->pos = pld_pos + 2u;
                    step++;
                    cur_token = out_tokens[1];

                    if (out_tokens[1] == m->tokenizer->eos_id) {
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }
                } else {
                    // 第 1 个草稿 Token 未命中！
                    // 回滚第 2 个槽位对 KV Cache 的污染修改，保留当前槽位的权威真值
                    /* The verifier already retains the authoritative first
                     * state on a miss; a second restore would clobber it. */
                    ctx->pos = pld_pos + 1u;

                    if (pld_len < pld_cap) pld_history[pld_len++] = out_tokens[0];

                    size_t piece_len = 0;
                    const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, out_tokens[0], &piece_len);
                    if (!piece) {
                        stop_reason = MYDS_STOP_ERROR;
                        success = 0;
                        break;
                    }
                    decode_emit(ctx, cb, bytes_cb, piece, piece_len, out_tokens[0], user_data);

                    step++;
                    cur_token = out_tokens[0];

                    if (out_tokens[0] == m->tokenizer->eos_id) {
                        stop_reason = MYDS_STOP_EOS;
                        break;
                    }
                }

                if (stop_reason == MYDS_STOP_EOS || stop_reason == MYDS_STOP_CANCELLED || !success || step >= max_gen) {
                    break;
                }
                continue;
            }

            for (uint32_t d = 0; d < n_draft; d++) {
                if (cancel_flag && *cancel_flag) {
                    stop_reason = MYDS_STOP_CANCELLED;
                    break;
                }
                if (ctx->pos >= ctx->cfg.max_seq_len) {
                    stop_reason = MYDS_STOP_CTX_FULL;
                    break;
                }
                if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                    ctx->pcache_tokens[ctx->pos] = cur_token;
                    if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
                }

                int32_t true_next = myds_forward_token(ctx, cur_token, ctx->pos);
                if (true_next < 0 || (uint32_t)true_next >= m->tokenizer->vocab_size) {
                    ctx->pcache_len = 0;
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                ctx->pos++;
                step++;
                if (pld_len < pld_cap) pld_history[pld_len++] = true_next;

                pld_proposed++;
                ctx->decode_stats.pld_proposed++;
                bool matched = (true_next == draft_tokens[d]);
                if (matched) {
                    pld_accepted++;
                    ctx->decode_stats.pld_accepted++;
                }

                if (true_next == m->tokenizer->eos_id) {
                    stop_reason = MYDS_STOP_EOS;
                    break;
                }

                size_t piece_len = 0;
                const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, true_next, &piece_len);
                if (!piece) {
                    stop_reason = MYDS_STOP_ERROR;
                    success = 0;
                    break;
                }
                decode_emit(ctx, cb, bytes_cb, piece, piece_len, true_next, user_data);

                cur_token = true_next;
                if (!matched) {
                    break;
                }
                if (step >= max_gen) break;
            }

            if (stop_reason == MYDS_STOP_EOS || stop_reason == MYDS_STOP_CANCELLED || !success) {
                break;
            }
        }

        if (pld_cycles > 0) {
            MYDS_LOG_I("PLD speculative: %llu cycles, %llu/%llu tokens accepted (%.1f%%)",
                       (unsigned long long)pld_cycles,
                       (unsigned long long)pld_accepted,
                       (unsigned long long)pld_proposed,
                       pld_proposed ? 100.0 * (double)pld_accepted / (double)pld_proposed : 0.0);
        }
        if (ctx->decode_stats.batch2_calls > 0) {
            MYDS_LOG_I("Batch2 verifier: %llu calls (%llu hits, %llu misses), total=%.1f ms (avg=%.2f ms/call) "
                       "[prep=%.2f ms, encode=%.2f ms, gpu=%.2f ms, arb=%.2f ms]",
                       (unsigned long long)ctx->decode_stats.batch2_calls,
                       (unsigned long long)ctx->decode_stats.batch2_hits,
                       (unsigned long long)ctx->decode_stats.batch2_misses,
                       ctx->decode_stats.verify_ms,
                       ctx->decode_stats.batch2_calls ? ctx->decode_stats.verify_ms / ctx->decode_stats.batch2_calls : 0.0,
                       ctx->decode_stats.batch2_calls ? ctx->decode_stats.batch2_prep_ms / ctx->decode_stats.batch2_calls : 0.0,
                       ctx->decode_stats.batch2_calls ? ctx->decode_stats.batch2_encode_ms / ctx->decode_stats.batch2_calls : 0.0,
                       ctx->decode_stats.batch2_calls ? ctx->decode_stats.batch2_gpu_ms / ctx->decode_stats.batch2_calls : 0.0,
                       ctx->decode_stats.batch2_calls ? ctx->decode_stats.batch2_arb_ms / ctx->decode_stats.batch2_calls : 0.0);
        }
        if (pld_snap) free(pld_snap);
    } else {
        for (uint32_t step = 0; step < max_gen; step++) {
            if (cancel_flag && *cancel_flag) {
                stop_reason = MYDS_STOP_CANCELLED;
                break;
            }
            decode_cycle_begin(ctx);
            /* 记录「正在被喂入的这个 token」（位置 = ctx->pos）⇒ 缓存序列与状态一一对应，
             * 而产出的 EOS 从未被喂入，所以不会被记进去。 */
            if (ctx->pcache_tokens && ctx->pos < ctx->pcache_cap) {
                ctx->pcache_tokens[ctx->pos] = cur_token;
                if (ctx->pos + 1u > ctx->pcache_len) ctx->pcache_len = ctx->pos + 1u;
            }
            int32_t next_token = myds_forward_token(ctx, cur_token, ctx->pos);
            if (next_token < 0 || (uint32_t)next_token >= m->tokenizer->vocab_size) {
                /* 这一步没跑完 ⇒ 刚记下的那个 token 并未真正被消费（外部评审指出的盲区）。
                 * 若留着它，下个请求会从一个状态从未到达的位置续跑 —— 直接撞位置检查。
                 * 与 prefill 失败同理：整条缓存作废。 */
                ctx->pcache_len = 0;
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

            size_t piece_len = 0;
            const char *piece = myds_tokenizer_decode_bytes(m->tokenizer, next_token, &piece_len);
            if (!piece) {
                stop_reason = MYDS_STOP_ERROR;
                success = 0;
                break;
            }
            decode_emit(ctx, cb, bytes_cb, piece, piece_len, next_token, user_data);

            cur_token = next_token;
            if (ctx->pos >= ctx->cfg.max_seq_len) {
                stop_reason = MYDS_STOP_CTX_FULL;
                break;
            }
        }
    }

    decode_cycle_finish(ctx);
    ctx->decode_metrics_active = false;
    myds_dspark_capture_end_request(ctx->dspark,
        success && stop_reason != MYDS_STOP_ERROR && stop_reason != MYDS_STOP_CANCELLED);
    if (stop_reason == MYDS_STOP_ERROR || !success) ctx->pcache_len = 0;
    /* Prefill reserves the whole prompt but leaves its last token for decode.
     * Cancellation before that step, or a zero-output request, must retain
     * only tokens that actually reached the model state. */
    else if (ctx->pcache_len > ctx->pos) ctx->pcache_len = ctx->pos;
    if (pld_history) free(pld_history);
    if (out_reason) *out_reason = stop_reason;
    return success;
}

/* 最近一次请求的 prefill 统计。故意做成「读 ctx 上的字段」而不是改 generate 的签名：
 * 生成 API 的签名被库、单测、服务端多处依赖，加出参是多余的破坏。 */
void myds_last_prefill_stats(const myds_context_t *ctx, myds_prefill_stats_t *out) {
    if (!out) return;
    if (!ctx) { memset(out, 0, sizeof(*out)); return; }
    out->prompt_tokens    = ctx->last_prompt_tokens;
    out->cached_tokens    = ctx->last_cached_tokens;
    out->prefill_tokens   = ctx->last_prefill_tokens;
    out->prefill_seconds  = ctx->last_prefill_seconds;
}

void myds_last_decode_stats(const myds_context_t *ctx, myds_decode_stats_t *out) {
    if (!out) return;
    if (ctx) *out = ctx->decode_stats;
    else memset(out, 0, sizeof(*out));
}

bool myds_last_decode_cycle(const myds_context_t *ctx, uint32_t index, myds_decode_cycle_t *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!ctx || !ctx->decode_cycles || index >= ctx->decode_stats.recorded_cycles) return false;
    *out = ctx->decode_cycles[index];
    return true;
}

/* 供服务端在收到 previous_response_id 时调用：下一次 generate 把 pcache 序列当作前缀。 */
void myds_set_continue_from_cache(myds_context_t *ctx, bool enable) {
    if (ctx) ctx->continue_from_cache = enable;
}

uint32_t myds_context_get_pcache_len(const myds_context_t *ctx) {
    return ctx ? ctx->pcache_len : 0;
}

int myds_generate_bytes_ex(myds_context_t *ctx, const char *prompt,
                          myds_stream_bytes_callback_t cb, void *user_data,
                          myds_stop_reason_t *out_reason, uint32_t *out_prompt_tokens,
                          volatile int *cancel_flag) {
    return myds_generate_internal(ctx, prompt, NULL, cb, user_data,
                                  out_reason, out_prompt_tokens, cancel_flag);
}

int myds_generate_ex(myds_context_t *ctx, const char *prompt,
                     myds_stream_callback_t cb, void *user_data,
                     myds_stop_reason_t *out_reason, uint32_t *out_prompt_tokens,
                     volatile int *cancel_flag) {
    return myds_generate_internal(ctx, prompt, cb, NULL, user_data,
                                  out_reason, out_prompt_tokens, cancel_flag);
}

int myds_generate(myds_context_t *ctx, const char *prompt, myds_stream_callback_t cb, void *user_data) {
    return myds_generate_ex(ctx, prompt, cb, user_data, NULL, NULL, NULL);
}
