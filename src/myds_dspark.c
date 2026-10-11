#include "myds_dspark.h"
#include "myds_cpu.h"
#include "myds_gguf.h"
#include "myds_metal.h"
#include "myds_v41.h"
#include "myds_v41_gpu.h"
#include "myds_v41_moe.h"
#include "myds_tokenizer.h"
#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Proposal contract fixtures replace only the neural stage boundary; the
 * production selector, Markov conditioning and publication still run. */
#ifndef DSPARK_EVAL_STAGES
#define DSPARK_EVAL_STAGES dspark_eval_requested_stages
#define DSPARK_DEFAULT_STAGE_EVAL 1
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} q8_0_block_t;

static inline double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static inline float dspark_fp16_to_fp32(uint16_t h) {
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    __fp16 val;
    memcpy(&val, &h, sizeof(val));
    return (float)val;
#else
    return myds_fp16_to_fp32(h);
#endif
}

static myds_metal_buffer_t *create_metal_buf(const gguf_tensor_t *t) {
    if (!t || !t->data || !t->size_bytes) return NULL;
    myds_metal_buffer_t *buf = myds_metal_buffer_from_ptr(t->data, t->size_bytes);
    if (!buf) {
        buf = myds_metal_buffer_alloc(t->size_bytes);
        if (buf) {
            myds_metal_buffer_write(buf, 0, t->data, t->size_bytes);
        }
    }
    return buf;
}

static int dspark_markov_only(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *env = getenv("MYDS_DSPARK_MARKOV_ONLY");
        mode = env && atoi(env) != 0;
    }
    return mode;
}

static void dspark_format_token_preview(const myds_tokenizer_t *tok, int32_t token_id, char *buf, size_t buf_size) {
    if (!buf || buf_size == 0) return;
    buf[0] = '\0';
    if (!tok || token_id < 0) {
        snprintf(buf, buf_size, "%d", token_id);
        return;
    }
    size_t piece_len = 0;
    const char *piece = myds_tokenizer_decode_bytes(tok, token_id, &piece_len);
    if (!piece || piece_len == 0) {
        snprintf(buf, buf_size, "%d", token_id);
        return;
    }
    char esc[48];
    size_t e = 0;
    for (size_t i = 0; i < piece_len && e + 2 < sizeof(esc); i++) {
        unsigned char c = (unsigned char)piece[i];
        if (c == '\n') { esc[e++] = '\\'; esc[e++] = 'n'; }
        else if (c == '\r') { esc[e++] = '\\'; esc[e++] = 'r'; }
        else if (c == '\t') { esc[e++] = '\\'; esc[e++] = 't'; }
        else if (c == ' ') { esc[e++] = '_'; }
        else if (c >= 32 && c < 127) { esc[e++] = (char)c; }
        else { esc[e++] = (char)c; }
    }
    esc[e] = '\0';
    snprintf(buf, buf_size, "%d ('%s')", token_id, esc);
}

static int capture_parse_flag(const char *name) {
    const char *env = getenv(name);
    if (!env || strcmp(env, "0") == 0) return 0;
    if (strcmp(env, "1") == 0) return 1;
    MYDS_LOG_E("%s must be exactly 0 or 1", name);
    return -1;
}

int myds_dspark_capture_parse_mode(void) {
    const int broad = capture_parse_flag("MYDS_DSPARK_CAPTURE_BYPASS");
    const int engine = capture_parse_flag("MYDS_ENGINE_CAPTURE_BYPASS");
    if (broad < 0 || engine < 0) return -1;
    if (broad && engine) {
        MYDS_LOG_E("MYDS_DSPARK_CAPTURE_BYPASS=1 and MYDS_ENGINE_CAPTURE_BYPASS=1 are mutually exclusive");
        return -1;
    }
    return engine ? 2 : broad;
}

bool myds_dspark_capture_configure(myds_dspark_t *ds) {
    if (!ds) return false;
    if (ds->capture_configured) return true;
    const int mode = myds_dspark_capture_parse_mode();
    if (mode < 0) return false;
    ds->capture_bypass_mode = mode != 0;
    ds->capture_engine_only = mode == 2;
    ds->capture_configured = true;
    ds->capture_disabled = false;
    ds->capture_valid_mask = 0;
    MYDS_LOG_I("DSpark target capture bypass: %d (request neural-consumer scope)", mode != 0);
    MYDS_LOG_I("Engine target capture bypass: %d (Engine call chain; rejects experimental tree when enabled)", mode == 2);
    return true;
}

int myds_dspark_capture_bypass_enabled(const myds_dspark_t *ds) {
    return ds && ds->capture_bypass_mode;
}
int myds_dspark_engine_capture_bypass_enabled(const myds_dspark_t *ds) {
    return ds && ds->capture_engine_only;
}
bool myds_dspark_capture_required(const myds_dspark_t *ds) {
    return ds && (!ds->capture_bypass_mode || !ds->capture_disabled);
}
bool myds_dspark_capture_ready(const myds_dspark_t *ds) {
    return ds && (!ds->capture_bypass_mode ||
        (!ds->capture_disabled && ds->capture_valid_mask == 7u && ds->capture_committed));
}
void myds_dspark_capture_counters(myds_dspark_t *ds, myds_dspark_capture_counters_t *out, bool reset) {
    if (out) *out = ds ? ds->capture_counters : (myds_dspark_capture_counters_t){0};
    if (reset && ds) memset(&ds->capture_counters, 0, sizeof(ds->capture_counters));
}
void myds_dspark_reset_proposal_state(myds_dspark_t *ds) {
    if (!ds) return;
    ds->consecutive_misses = 0;
    ds->backoff_counter = 0;
    ds->draft_count = 0;
    ds->draft_tree_count = 0;
    ds->is_high_confidence = false;
}
void myds_dspark_capture_invalidate(myds_dspark_t *ds) {
    if (ds && ds->capture_bypass_mode) {
        ds->capture_valid_mask = 0;
        ds->capture_committed = false;
    }
}
void myds_dspark_capture_publish(myds_dspark_t *ds) {
    if (ds && ds->capture_bypass_mode) {
        ds->capture_valid_mask = myds_dspark_capture_required(ds) ? 7u : 0u;
        ds->capture_committed = ds->capture_valid_mask == 7u;
    }
}
void myds_dspark_capture_begin_request(myds_dspark_t *ds, bool speculative_scope) {
    if (!ds || !ds->capture_bypass_mode) return;
    ds->capture_disabled = !speculative_scope || dspark_markov_only() || !ds->main_proj;
    myds_dspark_capture_invalidate(ds);
}
void myds_dspark_capture_end_request(myds_dspark_t *ds, bool success) {
    if (!ds || !ds->capture_bypass_mode) return;
    if (!success || ds->capture_disabled) myds_dspark_capture_invalidate(ds);
    ds->capture_disabled = false;
}

/* Public capture_layer callers need not know the target graph's publisher.
 * Complete their already encoded three-slot capture before allowing a read;
 * authoritative forward paths publish themselves and never take this branch. */
static bool capture_consumer_ready(myds_dspark_t *ds) {
    if (ds && ds->capture_bypass_mode && myds_dspark_capture_required(ds) &&
        ds->capture_valid_mask == 7u && !ds->capture_committed) {
        if (!myds_metal_commit_batch()) {
            ds->capture_counters.failures++;
            myds_dspark_capture_invalidate(ds);
            return false;
        }
        myds_dspark_capture_publish(ds);
    }
    return myds_dspark_capture_ready(ds);
}

myds_dspark_t *myds_dspark_load(const char *path, uint32_t vocab_size) {
    if (!path || !vocab_size) return NULL;
    /* Reject malformed explicit modes before opening/allocating any support
     * model resources. Publish the mode only once on the completed context. */
    const int capture_mode = myds_dspark_capture_parse_mode();
    if (capture_mode < 0) return NULL;

    gguf_file_t *gguf = gguf_open(path);
    if (!gguf) {
        MYDS_LOG_E("dspark: failed to open GGUF file: %s", path);
        return NULL;
    }

    const char *arch = gguf_get_str(gguf, "general.architecture", NULL);
    if (!arch || (strcmp(arch, "deepseek4-dspark") != 0 && strcmp(arch, "deepseek41-dflash-draft") != 0)) {
        MYDS_LOG_E("dspark: architecture mismatch (got %s, expected deepseek4-dspark or deepseek41-dflash-draft)", arch ? arch : "null");
        gguf_close(gguf);
        return NULL;
    }

    myds_dspark_t *ds = calloc(1, sizeof(*ds));
    if (!ds) {
        gguf_close(gguf);
        return NULL;
    }
    ds->gf = gguf;
    ds->capture_bypass_mode = capture_mode != 0;
    ds->capture_engine_only = capture_mode == 2;
    ds->capture_configured = true;
    MYDS_LOG_I("DSpark target capture bypass: %d (request neural-consumer scope)", capture_mode != 0);
    MYDS_LOG_I("Engine target capture bypass: %d (Engine call chain; rejects experimental tree when enabled)", capture_mode == 2);

    uint32_t stages = gguf_get_u32(gguf, "dspark.stage_count", 0);
    if (!stages) stages = gguf_get_u32(gguf, "deepseek41-dflash-draft.block_count", 3);
    ds->stage_count = stages;

    uint32_t bsize = gguf_get_u32(gguf, "dspark.block_size", 0);
    if (!bsize) bsize = gguf_get_u32(gguf, "deepseek41-dflash-draft.dflash.block_size", MYDS_DSPARK_DEFAULT_BLOCK_SIZE);
    ds->block_size = bsize;

    uint32_t mrank = gguf_get_u32(gguf, "dspark.markov_rank", 0);
    if (!mrank) mrank = gguf_get_u32(gguf, "deepseek41-dflash-draft.dflash.dspark.markov_rank", MYDS_DSPARK_MARKOV_RANK);
    ds->markov_rank = mrank;

    uint32_t ntoken = gguf_get_u32(gguf, "dspark.noise_token_id", 0);
    if (!ntoken) ntoken = gguf_get_u32(gguf, "deepseek41-dflash-draft.dflash.mask_token_id", 128799);
    ds->noise_token_id = ntoken;

    uint32_t nexp = gguf_get_u32(gguf, "dspark.n_routed_experts", 0);
    if (!nexp) nexp = gguf_get_u32(gguf, "deepseek41-dflash-draft.expert_count", 128);
    ds->n_routed_experts = nexp;

    uint32_t kexp = gguf_get_u32(gguf, "dspark.num_experts_per_tok", 0);
    if (!kexp) kexp = gguf_get_u32(gguf, "deepseek41-dflash-draft.expert_used_count", 3);
    ds->num_experts_per_tok = kexp;
    ds->current_seed_token = -1;
    ds->rms_eps = gguf_get_f32(gguf, "deepseek41-dflash-draft.attention.layer_norm_rms_epsilon", 1e-6f);
    ds->hc_eps = gguf_get_f32(gguf, "deepseek41-dflash-draft.hyper_connection.epsilon", 1e-5f);
    ds->route_scale = gguf_get_f32(gguf, "deepseek41-dflash-draft.expert_weights_scale", 1.5f);
    ds->swiglu_clamp = gguf_get_f32(gguf, "deepseek41-dflash-draft.swiglu_clamp_exp", 10.0f);

    /* The workspace and kernels implement this checkpoint geometry. Reject
     * metadata that would resize CPU state or overrun fixed router views. */
    if (ds->stage_count != 3u || ds->block_size != MYDS_DSPARK_DEFAULT_BLOCK_SIZE ||
        ds->markov_rank != MYDS_DSPARK_MARKOV_RANK || ds->n_routed_experts != 128u ||
        ds->num_experts_per_tok != 3u || ds->noise_token_id >= vocab_size ||
        !isfinite(ds->rms_eps) || ds->rms_eps < 0.0f ||
        !isfinite(ds->hc_eps) || ds->hc_eps <= 0.0f ||
        !isfinite(ds->route_scale) || ds->route_scale <= 0.0f ||
        !isfinite(ds->swiglu_clamp) || ds->swiglu_clamp < 0.0f) {
        MYDS_LOG_E("dspark: unsupported checkpoint geometry or numerical metadata");
        gguf_close(gguf);
        free(ds);
        return NULL;
    }

    // Official checkpoint capture defaults are zero-based blocks {36, 37, 38}.
    /* capture_layer_ids are zero-based block indices. target_layer_ids in
     * the official DFlash GGUF describe hidden-state boundaries (one-based).
     * The target graph passes its zero-based post-FFN block index here. */
    static const char *const target_layer_keys[] = {
        "deepseek41-dflash-draft.dflash.capture_layer_ids",
        "dflash.capture_layer_ids",
        "dspark.capture_layer_ids",
        "deepseek41-dflash-draft.dflash.target_layer_ids",
        "dflash.target_layer_ids",
        "dspark.target_layer_ids",
        "deepseek4.dspark.target_layer_ids",
        "deepseek4.dspark_target_layer_ids",
        "dspark.target_layers",
    };
    const gguf_kv_t *kv_cap = NULL;
    bool hidden_boundary_ids = false;
    for (size_t k = 0; k < sizeof(target_layer_keys) / sizeof(target_layer_keys[0]); k++) {
        kv_cap = gguf_find_key(gguf, target_layer_keys[k]);
        if (kv_cap && kv_cap->type == GGUF_TYPE_ARRAY && kv_cap->value.val_arr.count >= MYDS_DSPARK_TARGET_LAYERS &&
            (kv_cap->value.val_arr.elem_type == GGUF_TYPE_UINT32 || kv_cap->value.val_arr.elem_type == GGUF_TYPE_INT32)) {
            hidden_boundary_ids = (k == 3u || k == 4u);
            break;
        }
        kv_cap = NULL;
    }

    if (kv_cap) {
        const uint32_t *arr = (const uint32_t *)kv_cap->value.val_arr.data;
        for (int i = 0; i < MYDS_DSPARK_TARGET_LAYERS; i++) {
            if (hidden_boundary_ids && arr[i] == 0u) {
                MYDS_LOG_E("dspark: invalid one-based target hidden-state boundary");
                gguf_close(gguf);
                free(ds);
                return NULL;
            }
            ds->target_layers[i] = arr[i] - (hidden_boundary_ids ? 1u : 0u);
        }
    } else {
        ds->target_layers[0] = 36;
        ds->target_layers[1] = 37;
        ds->target_layers[2] = 38;
    }

    const char *env_layers = getenv("MYDS_DSPARK_TARGET_LAYERS");
    if (env_layers && *env_layers) {
        unsigned l0 = 0, l1 = 0, l2 = 0;
        if (sscanf(env_layers, "%u,%u,%u", &l0, &l1, &l2) == 3) {
            ds->target_layers[0] = l0;
            ds->target_layers[1] = l1;
            ds->target_layers[2] = l2;
            MYDS_LOG_I("dspark: target_layers overridden by env to {%u, %u, %u}", l0, l1, l2);
        }
    }
    MYDS_LOG_I("dspark: capture_layers = {%u, %u, %u}",
               ds->target_layers[0], ds->target_layers[1], ds->target_layers[2]);

    // Stage 0 entry projection
    const gguf_tensor_t *t_main_proj = gguf_find_tensor(gguf, "mtp.0.main_proj.weight");
    if (!t_main_proj) t_main_proj = gguf_find_tensor(gguf, "dflash.fc.weight");
    const gguf_tensor_t *t_main_norm = gguf_find_tensor(gguf, "mtp.0.main_norm.weight");
    if (!t_main_norm) t_main_norm = gguf_find_tensor(gguf, "dflash.hidden_norm.weight");
    if (t_main_proj) {
        ds->main_proj = create_metal_buf(t_main_proj);
        ds->main_proj_type = t_main_proj->type;
    }
    if (t_main_norm) {
        ds->main_norm = create_metal_buf(t_main_norm);
    }

    // Load stages
    char name[128];
    for (uint32_t s = 0; s < ds->stage_count; s++) {
        myds_dspark_stage_t *st = &ds->stages[s];

        #define BIND_T(field, tname, ftype) do { \
            snprintf(name, sizeof(name), "mtp.%u." tname, s); \
            const gguf_tensor_t *t = gguf_find_tensor(gguf, name); \
            if (!t) { \
                snprintf(name, sizeof(name), "blk.%u." tname, s); \
                t = gguf_find_tensor(gguf, name); \
            } \
            if (t) { \
                st->field = create_metal_buf(t); \
                st->ftype = t->type; \
            } \
        } while (0)

        #define BIND_BUF(field, tname) do { \
            snprintf(name, sizeof(name), "mtp.%u." tname, s); \
            const gguf_tensor_t *t = gguf_find_tensor(gguf, name); \
            if (!t) { \
                snprintf(name, sizeof(name), "blk.%u." tname, s); \
                t = gguf_find_tensor(gguf, name); \
            } \
            if (t) st->field = create_metal_buf(t); \
        } while (0)

        BIND_BUF(attn_norm, "attn_norm.weight");
        BIND_T(attn_q_a, "attn_q_a.weight", attn_q_a_type);
        BIND_BUF(attn_q_a_norm, "attn_q_a_norm.weight");
        BIND_T(attn_q_b, "attn_q_b.weight", attn_q_b_type);
        BIND_T(attn_kv, "attn_kv.weight", attn_kv_type);
        BIND_BUF(attn_kv_a_norm, "attn_kv_a_norm.weight");
        BIND_BUF(attn_sinks, "attn_sinks.weight");
        BIND_T(attn_output_a, "attn_output_a.weight", attn_output_a_type);
        BIND_T(attn_output_b, "attn_output_b.weight", attn_output_b_type);

        BIND_BUF(hc_attn_base, "hc_attn_base.weight");
        BIND_T(hc_attn_fn, "hc_attn_fn.weight", hc_attn_fn_type);
        BIND_BUF(hc_attn_scale, "hc_attn_scale.weight");

        BIND_BUF(ffn_norm, "ffn_norm.weight");
        BIND_T(router_gate, "ffn_gate_inp.weight", router_gate_type);
        BIND_BUF(router_bias, "exp_probs_b.bias");

        BIND_T(shared_gate, "ffn_gate_shexp.weight", shared_gate_type);
        BIND_T(shared_up, "ffn_up_shexp.weight", shared_up_type);
        BIND_T(shared_down, "ffn_down_shexp.weight", shared_down_type);

        BIND_T(ffn_gate_exps, "ffn_gate_exps.weight", ffn_gate_exps_type);
        BIND_T(ffn_up_exps, "ffn_up_exps.weight", ffn_up_exps_type);
        BIND_T(ffn_down_exps, "ffn_down_exps.weight", ffn_down_exps_type);

        BIND_BUF(hc_ffn_base, "hc_ffn_base.weight");
        BIND_T(hc_ffn_fn, "hc_ffn_fn.weight", hc_ffn_fn_type);
        BIND_BUF(hc_ffn_scale, "hc_ffn_scale.weight");

        #undef BIND_T
        #undef BIND_BUF
    }

    // Final stage heads (on stage ds->stage_count - 1, typically mtp.2 or blk.2)
    uint32_t final_s = ds->stage_count - 1u;
    snprintf(name, sizeof(name), "mtp.%u.norm.weight", final_s);
    const gguf_tensor_t *t_fnorm = gguf_find_tensor(gguf, name);
    if (!t_fnorm) {
        snprintf(name, sizeof(name), "blk.%u.norm.weight", final_s);
        t_fnorm = gguf_find_tensor(gguf, name);
    }
    if (!t_fnorm) {
        t_fnorm = gguf_find_tensor(gguf, "output_norm.weight");
    }
    if (t_fnorm) ds->final_norm = create_metal_buf(t_fnorm);

    snprintf(name, sizeof(name), "mtp.%u.markov_head.markov_w1.weight", final_s);
    const gguf_tensor_t *t_mw1 = gguf_find_tensor(gguf, name);
    if (!t_mw1) {
        snprintf(name, sizeof(name), "mtp.%u.markov_head.embed.weight", final_s);
        t_mw1 = gguf_find_tensor(gguf, name);
    }
    if (!t_mw1) {
        t_mw1 = gguf_find_tensor(gguf, "dflash.dspark.markov.w1");
    }
    if (t_mw1) {
        ds->markov_w1 = create_metal_buf(t_mw1);
        ds->markov_w1_type = t_mw1->type;
    }

    snprintf(name, sizeof(name), "mtp.%u.markov_head.markov_w2.weight", final_s);
    const gguf_tensor_t *t_mw2 = gguf_find_tensor(gguf, name);
    if (!t_mw2) {
        snprintf(name, sizeof(name), "mtp.%u.markov_head.head.weight", final_s);
        t_mw2 = gguf_find_tensor(gguf, name);
    }
    if (!t_mw2) {
        t_mw2 = gguf_find_tensor(gguf, "dflash.dspark.markov.w2");
    }
    if (t_mw2) {
        ds->markov_w2 = create_metal_buf(t_mw2);
        ds->markov_w2_type = t_mw2->type;
    }

    snprintf(name, sizeof(name), "mtp.%u.confidence_head.proj.weight", final_s);
    const gguf_tensor_t *t_cproj = gguf_find_tensor(gguf, name);
    if (!t_cproj) {
        t_cproj = gguf_find_tensor(gguf, "dflash.dspark.confidence.weight");
    }
    if (t_cproj) {
        ds->confidence_proj = create_metal_buf(t_cproj);
        ds->confidence_proj_type = t_cproj->type;
        MYDS_LOG_I("confidence_proj: n_dims=%u, dims=[%llu, %llu], type=%d (%s)",
                   t_cproj->n_dims,
                   (unsigned long long)t_cproj->dims[0],
                   (unsigned long long)(t_cproj->n_dims > 1 ? t_cproj->dims[1] : 0),
                   (int)t_cproj->type, ggml_type_name(t_cproj->type));
    }

    snprintf(name, sizeof(name), "mtp.%u.confidence_head.proj.bias", final_s);
    const gguf_tensor_t *t_cbias = gguf_find_tensor(gguf, name);
    if (!t_cbias) {
        t_cbias = gguf_find_tensor(gguf, "dflash.dspark.confidence.bias");
    }
    if (t_cbias && t_cbias->data) {
        if (t_cbias->type == GGML_TYPE_F32) {
            ds->confidence_bias = *(const float *)t_cbias->data;
        } else if (t_cbias->type == GGML_TYPE_F16) {
            ds->confidence_bias = dspark_fp16_to_fp32(*(const uint16_t *)t_cbias->data);
        }
        MYDS_LOG_I("confidence_bias: %.6f (type=%d)", ds->confidence_bias, (int)t_cbias->type);
    } else {
        ds->confidence_bias = 0.0f;
    }

    // Allocate inference runtime scratch buffers
    ds->target_hidden_buf = myds_metal_buffer_alloc(MYDS_DSPARK_TARGET_LAYERS * MYDS_DSPARK_DIM * sizeof(float));
    for (int i = 0; i < MYDS_DSPARK_TARGET_LAYERS; i++) {
        ds->target_slots[i] = myds_metal_buffer_sub_view(ds->target_hidden_buf,
                                                         (size_t)i * MYDS_DSPARK_DIM * sizeof(float),
                                                         MYDS_DSPARK_DIM * sizeof(float));
    }

    ds->hc_mean_weights = myds_metal_buffer_alloc(4 * sizeof(float));
    const float mean_w[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    myds_metal_buffer_write(ds->hc_mean_weights, 0, mean_w, sizeof(mean_w));

    ds->stage_x_buf = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
    ds->noise_hc = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
    ds->noise_hc_initialized = false;
    ds->draft_logits_buf = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * vocab_size * sizeof(float));
    ds->markov_state_host = calloc(ds->markov_rank, sizeof(float));
    ds->markov_bias_host = calloc(vocab_size, sizeof(float));
    ds->spec_snap = calloc(1, sizeof(myds_v41_spec_snapshot_t));

    // Allocate 3-Stage Transformer Workspace (Multi-token block: 6 rows, 5 draft slots)
    myds_dspark_workspace_t *ws = calloc(1, sizeof(*ws));
    if (ws) {
        ws->stage_input_hc = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * 4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->stage_draft_hc = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->after_attn_hc = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->stage_flat_hc_norm = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * 4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->draft_flat_hc_norm = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->hc_mix = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * 24 * sizeof(float));
        ws->hc_split = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * 24 * sizeof(float));
        ws->draft_ffn_hc_mix = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 24 * sizeof(float));
        ws->draft_ffn_hc_split = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 24 * sizeof(float));

        ws->attn_cur = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * MYDS_DSPARK_DIM * sizeof(float));
        ws->attn_norm = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * MYDS_DSPARK_DIM * sizeof(float));
        ws->kv = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * 512 * sizeof(float));
        ws->kv_norm = myds_metal_buffer_alloc(MYDS_DSPARK_TOTAL_ROWS * 512 * sizeof(float));
        ws->qr = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 1280 * sizeof(float));
        ws->qr_norm = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 1280 * sizeof(float));
        ws->q = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 64 * 512 * sizeof(float));

        for (int s = 0; s < MYDS_DSPARK_MAX_STAGES; s++) {
            ws->stage_kv_cache[s] = myds_metal_buffer_alloc(128 * 512 * sizeof(float));
            if (ws->stage_kv_cache[s]) {
                memset(myds_metal_buffer_data(ws->stage_kv_cache[s]), 0, 128 * 512 * sizeof(float));
            }
        }

        ws->attn_heads = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 64 * 512 * sizeof(float));
        ws->attn_low = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 8192 * sizeof(float));
        ws->attn_out = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));

        ws->ffn_cur = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));
        ws->ffn_norm = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));
        ws->shared_gate = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->shared_up = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->shared_swiglu = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->shared_out = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));

        ws->router_logits = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 128 * sizeof(float));
        ws->selected_ids = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 6 * sizeof(int32_t));
        ws->selected_weights = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 6 * sizeof(float));
        ws->moe_gate = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->moe_up = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->moe_mid = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->moe_down = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * 6 * MYDS_DSPARK_DIM * sizeof(float));
        ws->moe_out = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));
        ws->ffn_total = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));

        ws->final_embd = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));
        ws->final_norm_out = myds_metal_buffer_alloc(MYDS_DSPARK_DEFAULT_BLOCK_SIZE * MYDS_DSPARK_DIM * sizeof(float));

        // Subviews for 6 rows
        for (int r = 0; r < MYDS_DSPARK_TOTAL_ROWS; r++) {
            ws->v_stage_input_hc[r] = myds_metal_buffer_sub_view(ws->stage_input_hc, (size_t)r * 4 * MYDS_DSPARK_DIM * sizeof(float), 4 * MYDS_DSPARK_DIM * sizeof(float));
            ws->v_stage_flat_hc_norm[r] = myds_metal_buffer_sub_view(ws->stage_flat_hc_norm, (size_t)r * 4 * MYDS_DSPARK_DIM * sizeof(float), 4 * MYDS_DSPARK_DIM * sizeof(float));
            ws->v_hc_mix[r] = myds_metal_buffer_sub_view(ws->hc_mix, (size_t)r * 24 * sizeof(float), 24 * sizeof(float));
            ws->v_hc_split[r] = myds_metal_buffer_sub_view(ws->hc_split, (size_t)r * 24 * sizeof(float), 24 * sizeof(float));
            ws->v_attn_cur[r] = myds_metal_buffer_sub_view(ws->attn_cur, (size_t)r * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_attn_norm[r] = myds_metal_buffer_sub_view(ws->attn_norm, (size_t)r * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_kv[r] = myds_metal_buffer_sub_view(ws->kv, (size_t)r * 512 * sizeof(float), 512 * sizeof(float));
            ws->v_kv_norm[r] = myds_metal_buffer_sub_view(ws->kv_norm, (size_t)r * 512 * sizeof(float), 512 * sizeof(float));
        }

        // Aliases for Row 0 (Target row) and legacy compatibility
        ws->stage_target_hc = ws->v_stage_input_hc[0];
        ws->target_flat_hc_norm = ws->v_stage_flat_hc_norm[0];
        ws->target_hc_mix = ws->v_hc_mix[0];
        ws->target_hc_split = ws->v_hc_split[0];
        ws->target_attn_cur = ws->v_attn_cur[0];
        ws->target_attn_norm = ws->v_attn_norm[0];
        ws->target_kv = ws->v_kv[0];
        ws->target_kv_norm = ws->v_kv_norm[0];

        ws->draft_attn_cur = ws->v_attn_cur[1];
        ws->draft_attn_norm = ws->v_attn_norm[1];
        ws->draft_kv = ws->v_kv[1];
        ws->draft_kv_norm = ws->v_kv_norm[1];
        ws->draft_hc_mix = ws->v_hc_mix[1];
        ws->draft_hc_split = ws->v_hc_split[1];

        ws->stage_hc = ws->stage_draft_hc;
        ws->flat_hc_norm = ws->stage_flat_hc_norm;

        // Subviews for 5 draft rows
        for (int d = 0; d < MYDS_DSPARK_DEFAULT_BLOCK_SIZE; d++) {
            ws->v_stage_draft_hc[d] = myds_metal_buffer_sub_view(ws->stage_draft_hc, (size_t)d * 4 * MYDS_DSPARK_DIM * sizeof(float), 4 * MYDS_DSPARK_DIM * sizeof(float));
            ws->v_after_attn_hc[d] = myds_metal_buffer_sub_view(ws->after_attn_hc, (size_t)d * 4 * MYDS_DSPARK_DIM * sizeof(float), 4 * MYDS_DSPARK_DIM * sizeof(float));
            ws->v_draft_flat_hc_norm[d] = myds_metal_buffer_sub_view(ws->draft_flat_hc_norm, (size_t)d * 4 * MYDS_DSPARK_DIM * sizeof(float), 4 * MYDS_DSPARK_DIM * sizeof(float));
            ws->v_draft_ffn_hc_mix[d] = myds_metal_buffer_sub_view(ws->draft_ffn_hc_mix, (size_t)d * 24 * sizeof(float), 24 * sizeof(float));
            ws->v_draft_ffn_hc_split[d] = myds_metal_buffer_sub_view(ws->draft_ffn_hc_split, (size_t)d * 24 * sizeof(float), 24 * sizeof(float));
            ws->v_qr[d] = myds_metal_buffer_sub_view(ws->qr, (size_t)d * 1280 * sizeof(float), 1280 * sizeof(float));
            ws->v_qr_norm[d] = myds_metal_buffer_sub_view(ws->qr_norm, (size_t)d * 1280 * sizeof(float), 1280 * sizeof(float));
            ws->v_q[d] = myds_metal_buffer_sub_view(ws->q, (size_t)d * 64 * 512 * sizeof(float), 64 * 512 * sizeof(float));
            ws->v_attn_heads[d] = myds_metal_buffer_sub_view(ws->attn_heads, (size_t)d * 64 * 512 * sizeof(float), 64 * 512 * sizeof(float));
            ws->v_attn_low[d] = myds_metal_buffer_sub_view(ws->attn_low, (size_t)d * 8192 * sizeof(float), 8192 * sizeof(float));
            ws->v_attn_out[d] = myds_metal_buffer_sub_view(ws->attn_out, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_ffn_cur[d] = myds_metal_buffer_sub_view(ws->ffn_cur, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_ffn_norm[d] = myds_metal_buffer_sub_view(ws->ffn_norm, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_shared_gate[d] = myds_metal_buffer_sub_view(ws->shared_gate, (size_t)d * MYDS_DSPARK_EXP_DIM * sizeof(float), MYDS_DSPARK_EXP_DIM * sizeof(float));
            ws->v_shared_up[d] = myds_metal_buffer_sub_view(ws->shared_up, (size_t)d * MYDS_DSPARK_EXP_DIM * sizeof(float), MYDS_DSPARK_EXP_DIM * sizeof(float));
            ws->v_shared_swiglu[d] = myds_metal_buffer_sub_view(ws->shared_swiglu, (size_t)d * MYDS_DSPARK_EXP_DIM * sizeof(float), MYDS_DSPARK_EXP_DIM * sizeof(float));
            ws->v_shared_out[d] = myds_metal_buffer_sub_view(ws->shared_out, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_router_logits[d] = myds_metal_buffer_sub_view(ws->router_logits, (size_t)d * 128 * sizeof(float), 128 * sizeof(float));
            ws->v_selected_ids[d] = myds_metal_buffer_sub_view(ws->selected_ids, (size_t)d * 6 * sizeof(int32_t), 6 * sizeof(int32_t));
            ws->v_selected_weights[d] = myds_metal_buffer_sub_view(ws->selected_weights, (size_t)d * 6 * sizeof(float), 6 * sizeof(float));
            ws->v_moe_gate[d] = myds_metal_buffer_sub_view(ws->moe_gate, (size_t)d * 6 * MYDS_DSPARK_EXP_DIM * sizeof(float), 6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
            ws->v_moe_up[d] = myds_metal_buffer_sub_view(ws->moe_up, (size_t)d * 6 * MYDS_DSPARK_EXP_DIM * sizeof(float), 6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
            ws->v_moe_mid[d] = myds_metal_buffer_sub_view(ws->moe_mid, (size_t)d * 6 * MYDS_DSPARK_EXP_DIM * sizeof(float), 6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
            ws->v_moe_down[d] = myds_metal_buffer_sub_view(ws->moe_down, (size_t)d * 6 * MYDS_DSPARK_DIM * sizeof(float), 6 * MYDS_DSPARK_DIM * sizeof(float));
            ws->v_moe_out[d] = myds_metal_buffer_sub_view(ws->moe_out, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_ffn_total[d] = myds_metal_buffer_sub_view(ws->ffn_total, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_final_embd[d] = myds_metal_buffer_sub_view(ws->final_embd, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_final_norm_out[d] = myds_metal_buffer_sub_view(ws->final_norm_out, (size_t)d * MYDS_DSPARK_DIM * sizeof(float), MYDS_DSPARK_DIM * sizeof(float));
            ws->v_logits[d] = myds_metal_buffer_sub_view(ds->draft_logits_buf, (size_t)d * vocab_size * sizeof(float), vocab_size * sizeof(float));
        }

        ws->gpu_scratch = myds_v41_gpu_scratch_create(128);
        ws->conf_features_host = calloc(5376, sizeof(float));
        ds->ws = ws;
    }

    MYDS_LOG_I("dspark: loaded successfully from %s (stages=%u, block=%u, markov_rank=%u, routed_exp=%u/%u)",
               path, ds->stage_count, ds->block_size, ds->markov_rank,
               ds->num_experts_per_tok, ds->n_routed_experts);
    return ds;
}

static void dspark_workspace_free(myds_dspark_workspace_t *ws) {
    if (!ws) return;

    // Free 6-row subviews
    for (int r = 0; r < MYDS_DSPARK_TOTAL_ROWS; r++) {
        if (ws->v_stage_input_hc[r]) myds_metal_buffer_free(ws->v_stage_input_hc[r]);
        if (ws->v_stage_flat_hc_norm[r]) myds_metal_buffer_free(ws->v_stage_flat_hc_norm[r]);
        if (ws->v_hc_mix[r]) myds_metal_buffer_free(ws->v_hc_mix[r]);
        if (ws->v_hc_split[r]) myds_metal_buffer_free(ws->v_hc_split[r]);
        if (ws->v_attn_cur[r]) myds_metal_buffer_free(ws->v_attn_cur[r]);
        if (ws->v_attn_norm[r]) myds_metal_buffer_free(ws->v_attn_norm[r]);
        if (ws->v_kv[r]) myds_metal_buffer_free(ws->v_kv[r]);
        if (ws->v_kv_norm[r]) myds_metal_buffer_free(ws->v_kv_norm[r]);
    }

    // Free 5-draft-slot subviews
    for (int d = 0; d < MYDS_DSPARK_DEFAULT_BLOCK_SIZE; d++) {
        if (ws->v_stage_draft_hc[d]) myds_metal_buffer_free(ws->v_stage_draft_hc[d]);
        if (ws->v_after_attn_hc[d]) myds_metal_buffer_free(ws->v_after_attn_hc[d]);
        if (ws->v_draft_flat_hc_norm[d]) myds_metal_buffer_free(ws->v_draft_flat_hc_norm[d]);
        if (ws->v_draft_ffn_hc_mix[d]) myds_metal_buffer_free(ws->v_draft_ffn_hc_mix[d]);
        if (ws->v_draft_ffn_hc_split[d]) myds_metal_buffer_free(ws->v_draft_ffn_hc_split[d]);
        if (ws->v_qr[d]) myds_metal_buffer_free(ws->v_qr[d]);
        if (ws->v_qr_norm[d]) myds_metal_buffer_free(ws->v_qr_norm[d]);
        if (ws->v_q[d]) myds_metal_buffer_free(ws->v_q[d]);
        if (ws->v_attn_heads[d]) myds_metal_buffer_free(ws->v_attn_heads[d]);
        if (ws->v_attn_low[d]) myds_metal_buffer_free(ws->v_attn_low[d]);
        if (ws->v_attn_out[d]) myds_metal_buffer_free(ws->v_attn_out[d]);
        if (ws->v_ffn_cur[d]) myds_metal_buffer_free(ws->v_ffn_cur[d]);
        if (ws->v_ffn_norm[d]) myds_metal_buffer_free(ws->v_ffn_norm[d]);
        if (ws->v_shared_gate[d]) myds_metal_buffer_free(ws->v_shared_gate[d]);
        if (ws->v_shared_up[d]) myds_metal_buffer_free(ws->v_shared_up[d]);
        if (ws->v_shared_swiglu[d]) myds_metal_buffer_free(ws->v_shared_swiglu[d]);
        if (ws->v_shared_out[d]) myds_metal_buffer_free(ws->v_shared_out[d]);
        if (ws->v_router_logits[d]) myds_metal_buffer_free(ws->v_router_logits[d]);
        if (ws->v_selected_ids[d]) myds_metal_buffer_free(ws->v_selected_ids[d]);
        if (ws->v_selected_weights[d]) myds_metal_buffer_free(ws->v_selected_weights[d]);
        if (ws->v_moe_gate[d]) myds_metal_buffer_free(ws->v_moe_gate[d]);
        if (ws->v_moe_up[d]) myds_metal_buffer_free(ws->v_moe_up[d]);
        if (ws->v_moe_mid[d]) myds_metal_buffer_free(ws->v_moe_mid[d]);
        if (ws->v_moe_down[d]) myds_metal_buffer_free(ws->v_moe_down[d]);
        if (ws->v_moe_out[d]) myds_metal_buffer_free(ws->v_moe_out[d]);
        if (ws->v_ffn_total[d]) myds_metal_buffer_free(ws->v_ffn_total[d]);
        if (ws->v_final_embd[d]) myds_metal_buffer_free(ws->v_final_embd[d]);
        if (ws->v_final_norm_out[d]) myds_metal_buffer_free(ws->v_final_norm_out[d]);
        if (ws->v_logits[d]) myds_metal_buffer_free(ws->v_logits[d]);
    }

    // Free base buffers
    if (ws->stage_input_hc) myds_metal_buffer_free(ws->stage_input_hc);
    if (ws->stage_draft_hc) myds_metal_buffer_free(ws->stage_draft_hc);
    if (ws->after_attn_hc) myds_metal_buffer_free(ws->after_attn_hc);
    if (ws->stage_flat_hc_norm) myds_metal_buffer_free(ws->stage_flat_hc_norm);
    if (ws->draft_flat_hc_norm) myds_metal_buffer_free(ws->draft_flat_hc_norm);
    if (ws->hc_mix) myds_metal_buffer_free(ws->hc_mix);
    if (ws->hc_split) myds_metal_buffer_free(ws->hc_split);
    if (ws->draft_ffn_hc_mix) myds_metal_buffer_free(ws->draft_ffn_hc_mix);
    if (ws->draft_ffn_hc_split) myds_metal_buffer_free(ws->draft_ffn_hc_split);

    if (ws->attn_cur) myds_metal_buffer_free(ws->attn_cur);
    if (ws->attn_norm) myds_metal_buffer_free(ws->attn_norm);
    if (ws->kv) myds_metal_buffer_free(ws->kv);
    if (ws->kv_norm) myds_metal_buffer_free(ws->kv_norm);
    if (ws->qr) myds_metal_buffer_free(ws->qr);
    if (ws->qr_norm) myds_metal_buffer_free(ws->qr_norm);
    if (ws->q) myds_metal_buffer_free(ws->q);

    for (int s = 0; s < MYDS_DSPARK_MAX_STAGES; s++) {
        if (ws->stage_kv_cache[s]) myds_metal_buffer_free(ws->stage_kv_cache[s]);
    }

    if (ws->attn_heads) myds_metal_buffer_free(ws->attn_heads);
    if (ws->attn_low) myds_metal_buffer_free(ws->attn_low);
    if (ws->attn_out) myds_metal_buffer_free(ws->attn_out);

    if (ws->ffn_cur) myds_metal_buffer_free(ws->ffn_cur);
    if (ws->ffn_norm) myds_metal_buffer_free(ws->ffn_norm);
    if (ws->shared_gate) myds_metal_buffer_free(ws->shared_gate);
    if (ws->shared_up) myds_metal_buffer_free(ws->shared_up);
    if (ws->shared_swiglu) myds_metal_buffer_free(ws->shared_swiglu);
    if (ws->shared_out) myds_metal_buffer_free(ws->shared_out);

    if (ws->router_logits) myds_metal_buffer_free(ws->router_logits);
    if (ws->selected_ids) myds_metal_buffer_free(ws->selected_ids);
    if (ws->selected_weights) myds_metal_buffer_free(ws->selected_weights);
    if (ws->moe_gate) myds_metal_buffer_free(ws->moe_gate);
    if (ws->moe_up) myds_metal_buffer_free(ws->moe_up);
    if (ws->moe_mid) myds_metal_buffer_free(ws->moe_mid);
    if (ws->moe_down) myds_metal_buffer_free(ws->moe_down);
    if (ws->moe_out) myds_metal_buffer_free(ws->moe_out);
    if (ws->ffn_total) myds_metal_buffer_free(ws->ffn_total);

    if (ws->final_embd) myds_metal_buffer_free(ws->final_embd);
    if (ws->final_norm_out) myds_metal_buffer_free(ws->final_norm_out);

    if (ws->gpu_scratch) myds_v41_gpu_scratch_free(ws->gpu_scratch);
    if (ws->conf_features_host) free(ws->conf_features_host);
    free(ws);
}

void myds_dspark_free(myds_dspark_t *ds) {
    if (!ds) return;
    if (ds->ws) dspark_workspace_free(ds->ws);
    if (ds->main_proj) myds_metal_buffer_free(ds->main_proj);
    if (ds->main_norm) myds_metal_buffer_free(ds->main_norm);

    for (uint32_t s = 0; s < ds->stage_count; s++) {
        myds_dspark_stage_t *st = &ds->stages[s];
        if (st->attn_norm) myds_metal_buffer_free(st->attn_norm);
        if (st->attn_q_a) myds_metal_buffer_free(st->attn_q_a);
        if (st->attn_q_a_norm) myds_metal_buffer_free(st->attn_q_a_norm);
        if (st->attn_q_b) myds_metal_buffer_free(st->attn_q_b);
        if (st->attn_kv) myds_metal_buffer_free(st->attn_kv);
        if (st->attn_kv_a_norm) myds_metal_buffer_free(st->attn_kv_a_norm);
        if (st->attn_sinks) myds_metal_buffer_free(st->attn_sinks);
        if (st->attn_output_a) myds_metal_buffer_free(st->attn_output_a);
        if (st->attn_output_b) myds_metal_buffer_free(st->attn_output_b);

        if (st->hc_attn_base) myds_metal_buffer_free(st->hc_attn_base);
        if (st->hc_attn_fn) myds_metal_buffer_free(st->hc_attn_fn);
        if (st->hc_attn_scale) myds_metal_buffer_free(st->hc_attn_scale);

        if (st->ffn_norm) myds_metal_buffer_free(st->ffn_norm);
        if (st->router_gate) myds_metal_buffer_free(st->router_gate);
        if (st->router_bias) myds_metal_buffer_free(st->router_bias);

        if (st->shared_gate) myds_metal_buffer_free(st->shared_gate);
        if (st->shared_up) myds_metal_buffer_free(st->shared_up);
        if (st->shared_down) myds_metal_buffer_free(st->shared_down);

        if (st->ffn_gate_exps) myds_metal_buffer_free(st->ffn_gate_exps);
        if (st->ffn_up_exps) myds_metal_buffer_free(st->ffn_up_exps);
        if (st->ffn_down_exps) myds_metal_buffer_free(st->ffn_down_exps);

        if (st->hc_ffn_base) myds_metal_buffer_free(st->hc_ffn_base);
        if (st->hc_ffn_fn) myds_metal_buffer_free(st->hc_ffn_fn);
        if (st->hc_ffn_scale) myds_metal_buffer_free(st->hc_ffn_scale);
    }

    if (ds->final_norm) myds_metal_buffer_free(ds->final_norm);
    if (ds->markov_w1) myds_metal_buffer_free(ds->markov_w1);
    if (ds->markov_w2) myds_metal_buffer_free(ds->markov_w2);
    if (ds->confidence_proj) myds_metal_buffer_free(ds->confidence_proj);

    for (int i = 0; i < MYDS_DSPARK_TARGET_LAYERS; i++) {
        if (ds->target_slots[i]) myds_metal_buffer_free(ds->target_slots[i]);
    }
    if (ds->hc_mean_weights) myds_metal_buffer_free(ds->hc_mean_weights);
    if (ds->target_hidden_buf) myds_metal_buffer_free(ds->target_hidden_buf);
    if (ds->stage_x_buf) myds_metal_buffer_free(ds->stage_x_buf);
    if (ds->noise_hc) myds_metal_buffer_free(ds->noise_hc);
    if (ds->draft_logits_buf) myds_metal_buffer_free(ds->draft_logits_buf);

    if (ds->markov_state_host) free(ds->markov_state_host);
    if (ds->markov_bias_host) free(ds->markov_bias_host);
    if (ds->spec_snap) free(ds->spec_snap);

    if (ds->gf) gguf_close(ds->gf);
    free(ds);
}

void myds_dspark_capture_layer(myds_dspark_t *ds, uint32_t layer_idx, const myds_metal_buffer_t *residual) {
    (void)myds_dspark_capture_layer_to(ds, layer_idx, residual, NULL);
}

int myds_dspark_capture_layer_to(myds_dspark_t *ds, uint32_t layer_idx,
        const myds_metal_buffer_t *residual, myds_metal_buffer_t *output) {
    if (!ds) return 1;
    int slot = myds_dspark_target_slot(ds, layer_idx);
    if (slot < 0) return 1;
    ds->capture_counters.requests++;
    if (!myds_dspark_capture_required(ds)) {
        ds->capture_counters.bypassed++;
        myds_dspark_capture_invalidate(ds);
        return 1;
    }
    const bool target_output = output == NULL;
    if (target_output) output = ds->target_slots[slot];
    const bool actual_slot = output == ds->target_slots[slot];
    if (actual_slot && ds->capture_bypass_mode) {
        /* Any overwrite of a published generation starts a fresh capture.
         * Direct callers cannot combine a new slot1/2 with old slot0 state. */
        if (slot == 0 || ds->capture_committed) myds_dspark_capture_invalidate(ds);
        ds->capture_committed = false;
    }
    const int ok = residual && (!target_output || ds->target_hidden_buf) && ds->hc_mean_weights && output &&
        myds_metal_v41_hc_weighted_sum_round(output, residual, ds->hc_mean_weights, MYDS_DSPARK_DIM, 0);
    if (ok) {
        ds->capture_counters.encoded++;
        if (actual_slot && ds->capture_bypass_mode) ds->capture_valid_mask |= 1u << slot;
    } else {
        ds->capture_counters.failures++;
        myds_dspark_capture_invalidate(ds);
    }
    return ok || !ds->capture_bypass_mode;
}

static inline float dot_f16_row256(const uint16_t *row, const float *state) {
    float total = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (int k = 0; k < 256; k += 16) {
        float16x8_t h0 = vld1q_f16((const float16_t *)(row + k));
        float16x8_t h1 = vld1q_f16((const float16_t *)(row + k + 8));

        float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h0));
        float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h0));
        float32x4_t f2 = vcvt_f32_f16(vget_low_f16(h1));
        float32x4_t f3 = vcvt_f32_f16(vget_high_f16(h1));

        acc0 = vfmaq_f32(acc0, f0, vld1q_f32(state + k));
        acc1 = vfmaq_f32(acc1, f1, vld1q_f32(state + k + 4));
        acc2 = vfmaq_f32(acc2, f2, vld1q_f32(state + k + 8));
        acc3 = vfmaq_f32(acc3, f3, vld1q_f32(state + k + 12));
    }
    total = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#else
    for (int k = 0; k < 256; k++) {
        total += dspark_fp16_to_fp32(row[k]) * state[k];
    }
#endif
    return total;
}

static inline float dot_f16_row(const uint16_t *row, const float *state, uint32_t n) {
    float total = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const float16_t *)(row + i));
        float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h));
        float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h));
        acc0 = vfmaq_f32(acc0, f0, vld1q_f32(state + i));
        acc1 = vfmaq_f32(acc1, f1, vld1q_f32(state + i + 4));
    }
    total = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) {
        total += dspark_fp16_to_fp32(row[i]) * state[i];
    }
#else
    for (uint32_t i = 0; i < n; i++) {
        total += dspark_fp16_to_fp32(row[i]) * state[i];
    }
#endif
    return total;
}

static bool dspark_lookup_token_embedding(const myds_model_t *m, int32_t token_id, float *dest) {
    if (!m || !m->tok_embeddings || token_id < 0 || (uint32_t)token_id >= m->vocab_size) {
        return false;
    }
    const void *data = myds_metal_buffer_data(m->tok_embeddings);
    if (!data) return false;
    if (m->tok_embeddings_type == GGML_TYPE_F32) {
        const float *row = (const float *)data + (size_t)token_id * m->dim;
        memcpy(dest, row, m->dim * sizeof(float));
        return true;
    } else if (m->tok_embeddings_type == GGML_TYPE_F16) {
        const uint16_t *row = (const uint16_t *)data + (size_t)token_id * m->dim;
#if defined(__ARM_NEON)
        uint32_t i = 0;
        for (; i + 8 <= m->dim; i += 8) {
            float16x8_t h = vld1q_f16((const float16_t *)(row + i));
            vst1q_f32(dest + i, vcvt_f32_f16(vget_low_f16(h)));
            vst1q_f32(dest + i + 4, vcvt_f32_f16(vget_high_f16(h)));
        }
        for (; i < m->dim; i++) {
            dest[i] = dspark_fp16_to_fp32(row[i]);
        }
#else
        for (uint32_t i = 0; i < m->dim; i++) dest[i] = dspark_fp16_to_fp32(row[i]);
#endif
        return true;
    }
    return false;
}

static inline float dot_q8_row(const q8_0_block_t *row, const float *state, uint32_t n_blocks) {
    float total = 0.0f;
#if defined(__ARM_NEON)
    for (uint32_t b = 0; b < n_blocks; b++) {
        float d = dspark_fp16_to_fp32(row[b].d);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        const int8_t *qs = row[b].qs;
        const float *s = state + b * 32;

        for (int k = 0; k < 32; k += 8) {
            int8x8_t q8 = vld1_s8(qs + k);
            int16x8_t q16 = vmovl_s8(q8);
            float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            acc0 = vfmaq_f32(acc0, qf0, vld1q_f32(s + k));
            acc1 = vfmaq_f32(acc1, qf1, vld1q_f32(s + k + 4));
        }
        total += vaddvq_f32(vaddq_f32(acc0, acc1)) * d;
    }
#else
    for (uint32_t b = 0; b < n_blocks; b++) {
        float d = dspark_fp16_to_fp32(row[b].d);
        const float *s = state + b * 32;
        float bsum = 0.0f;
        for (int k = 0; k < 32; k++) bsum += (float)row[b].qs[k] * s[k];
        total += d * bsum;
    }
#endif
    return total;
}

static inline float dot_q8_row256(const q8_0_block_t *row, const float *state) {
    return dot_q8_row(row, state, 8);
}

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#define MARKOV_NUM_SLICES 16
#else
#define MARKOV_NUM_SLICES 1
#endif

static inline void dspark_update_top2(int32_t tok, float score,
                                      int32_t *t1, float *s1,
                                      int32_t *t2, float *s2) {
    if (score > *s1) {
        *s2 = *s1;
        *t2 = *t1;
        *s1 = score;
        *t1 = tok;
    } else if (score > *s2) {
        *s2 = score;
        *t2 = tok;
    }
}

static int dspark_eval_stages_range(myds_dspark_t *ds, myds_context_t *ctx,
    float min_conf, float *out_confidence, uint32_t eval_slots,
    uint32_t logit_first, uint32_t logit_count) {
    if (out_confidence) *out_confidence = NAN;
    if (!ds || !ctx || !ctx->model) return 0;
    if (!capture_consumer_ready(ds)) {
        ds->capture_counters.stale_rejections++;
        return 0;
    }
    if (!ds->ws || !ds->stage_count || ds->stage_count > MYDS_DSPARK_MAX_STAGES ||
        ds->current_seed_position > 1048576u - MYDS_DSPARK_DEFAULT_BLOCK_SIZE ||
        ds->current_output_slot >= MYDS_DSPARK_DEFAULT_BLOCK_SIZE) return 0;
    if (!eval_slots || eval_slots > MYDS_DSPARK_DEFAULT_BLOCK_SIZE ||
        !logit_count || logit_first >= eval_slots || logit_count > eval_slots - logit_first ||
        ds->current_output_slot < logit_first ||
        ds->current_output_slot >= logit_first + logit_count) return 0;
    if (!ds->main_proj || !ds->target_hidden_buf || !ds->draft_logits_buf) return 0;

    myds_dspark_workspace_t *ws = ds->ws;
    if (!ws->gpu_scratch || !ds->main_norm || !ds->final_norm ||
        !isfinite(ds->rms_eps) || ds->rms_eps < 0.0f ||
        !isfinite(ds->hc_eps) || ds->hc_eps <= 0.0f) return 0;
    for (uint32_t i = 0; i < ds->stage_count; i++) {
        const myds_dspark_stage_t *st = &ds->stages[i];
        if (!st->attn_norm || !st->attn_q_a || !st->attn_q_a_norm || !st->attn_q_b ||
            !st->attn_kv || !st->attn_kv_a_norm || !st->attn_sinks ||
            !st->attn_output_a || !st->attn_output_b ||
            !st->hc_attn_fn || !st->hc_attn_scale || !st->hc_attn_base ||
            !st->ffn_norm || !st->router_gate || !st->router_bias ||
            !st->shared_gate || !st->shared_up || !st->shared_down ||
            !st->ffn_gate_exps || !st->ffn_up_exps || !st->ffn_down_exps ||
            !st->hc_ffn_fn || !st->hc_ffn_scale || !st->hc_ffn_base) return 0;
    }
    const uint32_t eval_rows = eval_slots + 1u;
    const uint32_t logit_end = logit_first + logit_count;
    /* This small graph has ordinary data dependencies throughout. Keep its
     * encoder barriers active even when the target verifier uses manual fences. */
    const int saved_barrier = myds_metal_get_auto_barrier();
    myds_metal_set_auto_barrier(1);
#define DSPARK_ENCODE(call) do { if (!(call)) goto encode_failure; } while (0)
    const uint32_t in_dim = MYDS_DSPARK_TARGET_LAYERS * MYDS_DSPARK_DIM;
    const size_t stream_bytes = (size_t)MYDS_DSPARK_DIM * sizeof(float);

    // Cache noise token embedding in ds->noise_hc on first use
    if (!ds->noise_hc_initialized && ds->noise_hc && ctx->model) {
        float *noise_ptr = (float *)myds_metal_buffer_data(ds->noise_hc);
        if (noise_ptr && dspark_lookup_token_embedding(ctx->model, (int32_t)ds->noise_token_id, noise_ptr)) {
            for (size_t c = 1; c < 4; c++) {
                memcpy(noise_ptr + c * MYDS_DSPARK_DIM, noise_ptr, stream_bytes);
            }
            ds->noise_hc_initialized = true;
        }
    }

    // 1. Initialize Row 1 (Seed Token Row) from seed token embedding (replicated 4x across 4 HC streams)
    float *seed_ptr = (float *)myds_metal_buffer_data(ws->v_stage_input_hc[1]);
    bool seeded_draft_hc = false;
    if (seed_ptr && ds->current_seed_token >= 0 &&
        dspark_lookup_token_embedding(ctx->model, ds->current_seed_token, seed_ptr)) {
        for (size_t c = 1; c < 4; c++) {
            memcpy(seed_ptr + c * MYDS_DSPARK_DIM, seed_ptr, stream_bytes);
        }
        seeded_draft_hc = true;
    }

    if (!seeded_draft_hc || !ds->noise_hc_initialized) goto encode_failure;

    // Begin batched Metal command buffer for stages cascade
    DSPARK_ENCODE(myds_metal_begin_batch());

    // 2. Stage 0 Entry Projection: target_hidden -> main_x [15360] -> [5120]
    if (!myds_metal_gemv(ds->stage_x_buf, ds->main_proj, ds->target_hidden_buf,
                         MYDS_DSPARK_DIM, in_dim, ds->main_proj_type)) {
        goto encode_failure;
    }
    if (ds->main_norm) {
        if (!myds_metal_v41_rmsnorm(ds->stage_x_buf, ds->stage_x_buf, ds->main_norm,
                                    MYDS_DSPARK_DIM, ds->rms_eps, false)) {
            goto encode_failure;
        }
    }

    // 2b. Initialize Row 0 (Target Row) from main_x (stage_x_buf replicated 4x across 4 HC streams)
    // Row 0 remains CONSTANT across all stages
    for (size_t c = 0; c < 4; c++) {
        DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_stage_input_hc[0], c * stream_bytes, ds->stage_x_buf, 0, stream_bytes));
    }

    // Initialize only the mask rows needed by the causal draft prefix.
    for (uint32_t r = 2; r < eval_rows; r++) {
        DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_stage_input_hc[r], 0, ds->noise_hc, 0, 4 * stream_bytes));
    }

    myds_v41_gpu_scratch_t *gpu_s = ws->gpu_scratch;

    // 3. Multi-token Block Transformer Stages Cascade (0, 1, 2)
    for (uint32_t s = 0; s < ds->stage_count; s++) {
        myds_dspark_stage_t *st = &ds->stages[s];

        // --- Step A: Pre-attn HC projection for target + draft prefix ---
        for (uint32_t r = 0; r < eval_rows; r++) {
            DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_stage_flat_hc_norm[r], ws->v_stage_input_hc[r], NULL, 4 * MYDS_DSPARK_DIM, ds->rms_eps, false));
            if (st->hc_attn_fn) {
                DSPARK_ENCODE(myds_metal_gemv(ws->v_hc_mix[r], st->hc_attn_fn, ws->v_stage_flat_hc_norm[r], 24, 4 * MYDS_DSPARK_DIM, st->hc_attn_fn_type));
                DSPARK_ENCODE(myds_metal_v41_hc_split_batch(ws->v_hc_split[r], ws->v_hc_mix[r], st->hc_attn_scale, st->hc_attn_base, 1, ds->hc_eps));
                DSPARK_ENCODE(myds_metal_v41_hc_weighted_sum_round(ws->v_attn_cur[r], ws->v_stage_input_hc[r], ws->v_hc_split[r], MYDS_DSPARK_DIM, 0));
            } else {
                DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_attn_cur[r], 0, ws->v_stage_input_hc[r], 0, stream_bytes));
            }

            if (st->attn_norm) {
                DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_attn_norm[r], ws->v_attn_cur[r], st->attn_norm, MYDS_DSPARK_DIM, ds->rms_eps, false));
            } else {
                DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_attn_norm[r], 0, ws->v_attn_cur[r], 0, stream_bytes));
            }
        }

        // --- Step B: Q path for every draft slot in the causal prefix ---
        for (uint32_t d = 0; d < eval_slots; d++) {
            if (st->attn_q_a && st->attn_q_b) {
                DSPARK_ENCODE(myds_metal_gemv(ws->v_qr[d], st->attn_q_a, ws->v_attn_norm[1 + d], 1280, MYDS_DSPARK_DIM, st->attn_q_a_type));
                if (st->attn_q_a_norm) {
                    DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_qr_norm[d], ws->v_qr[d], st->attn_q_a_norm, 1280, ds->rms_eps, false));
                }
                DSPARK_ENCODE(myds_metal_gemv(ws->v_q[d], st->attn_q_b, ws->v_qr_norm[d] ? ws->v_qr_norm[d] : ws->v_qr[d], 64 * 512, 1280, st->attn_q_b_type));

                // Per-head RMSNorm on Q (64 heads x 512 head_dim)
                DSPARK_ENCODE(myds_metal_v41_rmsnorm_batch(ws->v_q[d], ws->v_q[d], NULL, 512, 64, ds->rms_eps, false));

                // RoPE on Q (Query d predicts for position ds->current_seed_position + d)
                DSPARK_ENCODE(myds_dspark_gpu_rope(ws->v_q[d], 512, 64, ds->current_seed_position + d, false));
            }
        }

        // --- Step C: KV path for target + draft prefix, preserving row offsets ---
        for (uint32_t r = 0; r < eval_rows; r++) {
            if (st->attn_kv) {
                DSPARK_ENCODE(myds_metal_gemv(ws->v_kv[r], st->attn_kv, ws->v_attn_norm[r], 512, MYDS_DSPARK_DIM, st->attn_kv_type));
                if (st->attn_kv_a_norm) {
                    DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_kv_norm[r], ws->v_kv[r], st->attn_kv_a_norm, 512, ds->rms_eps, false));
                }
                uint32_t kv_pos = (r <= 1) ? ds->current_seed_position : (ds->current_seed_position + (uint32_t)(r - 1));
                DSPARK_ENCODE(myds_dspark_gpu_rope(ws->v_kv_norm[r] ? ws->v_kv_norm[r] : ws->v_kv[r], 512, 1, kv_pos, false));
                DSPARK_ENCODE(myds_dspark_gpu_quant_kv(ws->v_kv_norm[r] ? ws->v_kv_norm[r] : ws->v_kv[r], 1));
                DSPARK_ENCODE(myds_metal_buffer_copy(ws->stage_kv_cache[s], (size_t)r * 512 * sizeof(float),
                                             ws->v_kv_norm[r] ? ws->v_kv_norm[r] : ws->v_kv[r], 0, 512 * sizeof(float)));
            }
        }

        // --- Step D: Each query d sees exactly keys 0 .. 1 + d (n_raw = 2 + d) ---
        for (uint32_t d = 0; d < eval_slots; d++) {
            if (gpu_s) {
                DSPARK_ENCODE(myds_v41_gpu_attention_direct(ws->v_attn_heads[d], st->attn_sinks, ws->v_q[d],
                                                    ws->stage_kv_cache[s], 2 + d, 0,
                                                    NULL, NULL, 0, 0, gpu_s));
            }

            // F32 inverse RoPE on attention heads: slot d is at position ds->current_seed_position + d
            DSPARK_ENCODE(myds_dspark_gpu_rope(ws->v_attn_heads[d], 512, 64, ds->current_seed_position + d, true));

            // Attention Output Down-Up Projection
            if (st->attn_output_a && st->attn_output_b) {
                DSPARK_ENCODE(myds_metal_attn_output_a(ws->v_attn_low[d], st->attn_output_a, ws->v_attn_heads[d],
                                               8, 4096, 1024, st->attn_output_a_type));
                DSPARK_ENCODE(myds_metal_gemv(ws->v_attn_out[d], st->attn_output_b, ws->v_attn_low[d],
                                      MYDS_DSPARK_DIM, 8192, st->attn_output_b_type));
            }

            // HC Expand after Attention on Draft Row (round_bf16 = 0 for full F32 precision matching DSpark)
            if (st->hc_attn_fn) {
                DSPARK_ENCODE(myds_metal_v41_hc_expand_round(ws->v_after_attn_hc[d], ws->v_attn_out[d],
                                                     ws->v_stage_input_hc[1 + d], ws->v_hc_split[1 + d],
                                                     MYDS_DSPARK_DIM, 0));
            } else {
                for (size_t c = 0; c < 4; c++) {
                    DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_after_attn_hc[d], c * stream_bytes, ws->v_attn_out[d], 0, stream_bytes));
                }
            }
        }

        // --- Step E: FFN / MoE on every draft slot in the causal prefix ---
        for (uint32_t d = 0; d < eval_slots; d++) {
            // HC RMSNorm over flat 4 streams (20480) of after_attn_hc
            DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_draft_flat_hc_norm[d], ws->v_after_attn_hc[d], NULL, 4 * MYDS_DSPARK_DIM, ds->rms_eps, false));

            // Project to 24 mix values
            if (st->hc_ffn_fn) {
                DSPARK_ENCODE(myds_metal_gemv(ws->v_draft_ffn_hc_mix[d], st->hc_ffn_fn, ws->v_draft_flat_hc_norm[d], 24, 4 * MYDS_DSPARK_DIM, st->hc_ffn_fn_type));
                DSPARK_ENCODE(myds_metal_v41_hc_split_batch(ws->v_draft_ffn_hc_split[d], ws->v_draft_ffn_hc_mix[d], st->hc_ffn_scale, st->hc_ffn_base, 1, ds->hc_eps));
                DSPARK_ENCODE(myds_metal_v41_hc_weighted_sum_round(ws->v_ffn_cur[d], ws->v_after_attn_hc[d], ws->v_draft_ffn_hc_split[d], MYDS_DSPARK_DIM, 0));
            } else {
                DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_ffn_cur[d], 0, ws->v_after_attn_hc[d], 0, stream_bytes));
            }

            // FFN pre-norm
            if (st->ffn_norm) {
                DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_ffn_norm[d], ws->v_ffn_cur[d], st->ffn_norm, MYDS_DSPARK_DIM, ds->rms_eps, false));
            } else {
                DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_ffn_norm[d], 0, ws->v_ffn_cur[d], 0, stream_bytes));
            }

            // 1. Shared Expert (SwiGLU)
            if (st->shared_gate && st->shared_up && st->shared_down) {
                DSPARK_ENCODE(myds_metal_gemv(ws->v_shared_gate[d], st->shared_gate, ws->v_ffn_norm[d], MYDS_DSPARK_EXP_DIM, MYDS_DSPARK_DIM, st->shared_gate_type));
                DSPARK_ENCODE(myds_metal_gemv(ws->v_shared_up[d], st->shared_up, ws->v_ffn_norm[d], MYDS_DSPARK_EXP_DIM, MYDS_DSPARK_DIM, st->shared_up_type));
                DSPARK_ENCODE(myds_dspark_shared_swiglu(ws->v_shared_swiglu[d], ws->v_shared_gate[d], ws->v_shared_up[d], MYDS_DSPARK_EXP_DIM, ds->swiglu_clamp));
                DSPARK_ENCODE(myds_metal_gemv(ws->v_shared_out[d], st->shared_down, ws->v_shared_swiglu[d], MYDS_DSPARK_DIM, MYDS_DSPARK_EXP_DIM, st->shared_down_type));
            }

            // 2. Routed MoE (128 experts, top 3)
            if (st->router_gate && st->ffn_gate_exps && st->ffn_up_exps && st->ffn_down_exps) {
                DSPARK_ENCODE(myds_metal_gemv(ws->v_router_logits[d], st->router_gate, ws->v_ffn_norm[d], ds->n_routed_experts, MYDS_DSPARK_DIM, st->router_gate_type));

                const float route_scale = ds->route_scale; // DSpark GGUF expert_weights_scale, independent of target

                myds_metal_barrier_buffer(ws->v_router_logits[d]);
                DSPARK_ENCODE(myds_dspark_route_fused(ws->v_selected_ids[d], ws->v_selected_weights[d],
                    ws->v_router_logits[d], st->router_bias, ds->num_experts_per_tok, route_scale, 1));

                const float clamp_exp = ds->swiglu_clamp;
                DSPARK_ENCODE(myds_dspark_moe(ws->v_moe_out[d], ws->v_ffn_norm[d],
                                            st->ffn_gate_exps, st->ffn_up_exps, st->ffn_down_exps,
                                            ws->v_selected_ids[d], ws->v_selected_weights[d],
                                            ws->v_moe_gate[d], ws->v_moe_up[d], ws->v_moe_mid[d], ws->v_moe_down[d],
                                            MYDS_DSPARK_DIM, MYDS_DSPARK_EXP_DIM,
                                            ds->n_routed_experts, ds->num_experts_per_tok,
                                            st->ffn_gate_exps_type, st->ffn_up_exps_type, st->ffn_down_exps_type,
                                            clamp_exp));

                DSPARK_ENCODE(myds_metal_add(ws->v_ffn_total[d], ws->v_shared_out[d], ws->v_moe_out[d], MYDS_DSPARK_DIM));
            } else if (st->shared_gate && st->shared_down) {
                DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_ffn_total[d], 0, ws->v_shared_out[d], 0, stream_bytes));
            }

            // HC Expand after FFN -> stage_draft_hc for next stage (round_bf16 = 0 for full F32 precision)
            if (st->hc_ffn_fn) {
                DSPARK_ENCODE(myds_metal_v41_hc_expand_round(ws->v_stage_draft_hc[d], ws->v_ffn_total[d], ws->v_after_attn_hc[d], ws->v_draft_ffn_hc_split[d], MYDS_DSPARK_DIM, 0));
            } else {
                for (size_t c = 0; c < 4; c++) {
                    DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_stage_draft_hc[d], c * stream_bytes, ws->v_ffn_total[d], 0, stream_bytes));
                }
            }
        }

        // --- Step F: Propagate ws->v_stage_draft_hc[d] to next stage input Row 1 + d ---
        if (s + 1 < ds->stage_count) {
            for (uint32_t d = 0; d < eval_slots; d++) {
                DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_stage_input_hc[1 + d], 0, ws->v_stage_draft_hc[d], 0, 4 * stream_bytes));
            }
        }
    }

    // 4. Only requested outputs need final collapse/norm and the LM head.
    myds_dspark_stage_t *final_st = &ds->stages[ds->stage_count - 1u];
    for (uint32_t d = logit_first; d < logit_end; d++) {
        if (final_st->hc_ffn_fn) {
            DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_draft_flat_hc_norm[d], ws->v_stage_draft_hc[d], NULL, 4 * MYDS_DSPARK_DIM, ds->rms_eps, false));
            DSPARK_ENCODE(myds_metal_gemv(ws->v_draft_ffn_hc_mix[d], final_st->hc_ffn_fn, ws->v_draft_flat_hc_norm[d], 24, 4 * MYDS_DSPARK_DIM, final_st->hc_ffn_fn_type));
            DSPARK_ENCODE(myds_metal_v41_hc_split_batch(ws->v_draft_ffn_hc_split[d], ws->v_draft_ffn_hc_mix[d], final_st->hc_ffn_scale, final_st->hc_ffn_base, 1, ds->hc_eps));
            DSPARK_ENCODE(myds_metal_v41_hc_weighted_sum_round(ws->v_final_embd[d], ws->v_stage_draft_hc[d], ws->v_draft_ffn_hc_split[d], MYDS_DSPARK_DIM, 0));
        } else {
            DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_final_embd[d], 0, ws->v_stage_draft_hc[d], 0, stream_bytes));
        }

        if (ds->final_norm) {
            DSPARK_ENCODE(myds_metal_v41_rmsnorm(ws->v_final_norm_out[d], ws->v_final_embd[d], ds->final_norm, MYDS_DSPARK_DIM, ds->rms_eps, false));
        } else {
            DSPARK_ENCODE(myds_metal_buffer_copy(ws->v_final_norm_out[d], 0, ws->v_final_embd[d], 0, stream_bytes));
        }
    }

    const bool has_conf_proj = (ds->confidence_proj &&
                                (ds->confidence_proj_type == GGML_TYPE_F16 ||
                                 ds->confidence_proj_type == GGML_TYPE_Q8_0));
    const bool can_early_exit = (has_conf_proj && min_conf > 0.0f);

    if (can_early_exit) {
        // Commit final stages batch to complete GPU work for hidden states so CPU can test confidence
        DSPARK_ENCODE(myds_metal_commit_batch());

        float conf_prob = 1.0f;
        const float *final_embd_data = (const float *)myds_metal_buffer_data(ws->v_final_embd[ds->current_output_slot]);
        if (final_embd_data) {
            float conf_score = 0.0f;
            if (ds->confidence_proj_type == GGML_TYPE_F16) {
                const uint16_t *w_conf = (const uint16_t *)myds_metal_buffer_data(ds->confidence_proj);
                conf_score = dot_f16_row(w_conf, final_embd_data, MYDS_DSPARK_DIM);
                if (ds->markov_state_host) {
                    conf_score += dot_f16_row(w_conf + MYDS_DSPARK_DIM, ds->markov_state_host, ds->markov_rank);
                }
            } else if (ds->confidence_proj_type == GGML_TYPE_Q8_0) {
                const q8_0_block_t *w_conf = (const q8_0_block_t *)myds_metal_buffer_data(ds->confidence_proj);
                conf_score = dot_q8_row(w_conf, final_embd_data, MYDS_DSPARK_DIM / 32);
                if (ds->markov_state_host) {
                    conf_score += dot_q8_row(w_conf + MYDS_DSPARK_DIM / 32, ds->markov_state_host, ds->markov_rank / 32);
                }
            }
            conf_score += ds->confidence_bias;
            conf_prob = 1.0f / (1.0f + expf(-conf_score));
        }
        if (out_confidence) *out_confidence = conf_prob;

        // Early exit: skip the requested LM head GEMVs on low confidence.
        if (conf_prob < min_conf) {
            myds_metal_set_auto_barrier(saved_barrier);
            return 0;
        }

        // Project only the requested slots, at their original buffer offsets.
        DSPARK_ENCODE(myds_metal_begin_batch());
        for (uint32_t d = logit_first; d < logit_end; d++) {
            DSPARK_ENCODE(myds_metal_gemv(ws->v_logits[d], ctx->model->output, ws->v_final_norm_out[d],
                                  ctx->model->vocab_size, MYDS_DSPARK_DIM, ctx->model->output_type));
        }
        DSPARK_ENCODE(myds_metal_commit_batch());
    } else {
        // Enqueue the requested LM head GEMVs with the stage command buffer.
        for (uint32_t d = logit_first; d < logit_end; d++) {
            DSPARK_ENCODE(myds_metal_gemv(ws->v_logits[d], ctx->model->output, ws->v_final_norm_out[d],
                                  ctx->model->vocab_size, MYDS_DSPARK_DIM, ctx->model->output_type));
        }
        DSPARK_ENCODE(myds_metal_commit_batch());

        if (out_confidence) {
            float conf_prob = 1.0f;
            if (has_conf_proj) {
                const float *final_embd_data = (const float *)myds_metal_buffer_data(ws->v_final_embd[ds->current_output_slot]);
                if (final_embd_data) {
                    float conf_score = 0.0f;
                    if (ds->confidence_proj_type == GGML_TYPE_F16) {
                        const uint16_t *w_conf = (const uint16_t *)myds_metal_buffer_data(ds->confidence_proj);
                        conf_score = dot_f16_row(w_conf, final_embd_data, MYDS_DSPARK_DIM);
                        if (ds->markov_state_host) {
                            conf_score += dot_f16_row(w_conf + MYDS_DSPARK_DIM, ds->markov_state_host, ds->markov_rank);
                        }
                    } else if (ds->confidence_proj_type == GGML_TYPE_Q8_0) {
                        const q8_0_block_t *w_conf = (const q8_0_block_t *)myds_metal_buffer_data(ds->confidence_proj);
                        conf_score = dot_q8_row(w_conf, final_embd_data, MYDS_DSPARK_DIM / 32);
                        if (ds->markov_state_host) {
                            conf_score += dot_q8_row(w_conf + MYDS_DSPARK_DIM / 32, ds->markov_state_host, ds->markov_rank / 32);
                        }
                    }
                    conf_score += ds->confidence_bias;
                    conf_prob = 1.0f / (1.0f + expf(-conf_score));
                }
            }
            *out_confidence = conf_prob;
        }
    }

    myds_metal_set_auto_barrier(saved_barrier);
    return 1;
encode_failure:
    /* Drain partially encoded work; none of its logits become a proposal. */
    (void)myds_metal_commit_batch();
    myds_metal_set_auto_barrier(saved_barrier);
    if (out_confidence) *out_confidence = NAN;
    MYDS_LOG_E("DSpark stage encoding/commit failed; proposal discarded");
    return 0;
#undef DSPARK_ENCODE
}

int myds_dspark_eval_stages(myds_dspark_t *ds, myds_context_t *ctx,
    float min_conf, float *out_confidence) {
    return dspark_eval_stages_range(ds, ctx, min_conf, out_confidence,
        MYDS_DSPARK_DEFAULT_BLOCK_SIZE, 0u, MYDS_DSPARK_DEFAULT_BLOCK_SIZE);
}

int myds_dspark_eval_stages_prefix(myds_dspark_t *ds, myds_context_t *ctx,
    float min_conf, float *out_confidence, uint32_t first_slot, uint32_t count) {
    if (first_slot >= MYDS_DSPARK_DEFAULT_BLOCK_SIZE || !count ||
        count > MYDS_DSPARK_DEFAULT_BLOCK_SIZE - first_slot) {
        if (out_confidence) *out_confidence = NAN;
        return 0;
    }
    return dspark_eval_stages_range(ds, ctx, min_conf, out_confidence,
        first_slot + count, first_slot, count);
}

#if defined(DSPARK_DEFAULT_STAGE_EVAL)
static int dspark_eval_requested_stages(myds_dspark_t *ds, myds_context_t *ctx,
    float min_conf, float *out_confidence) {
    /* Read on every proposal so diagnostic runs can compare both graphs on
     * the same process and model without changing their initial state. */
    const char *prefix = getenv("MYDS_DSPARK_EVAL_PREFIX");
    if (prefix && strcmp(prefix, "0") == 0)
        return myds_dspark_eval_stages(ds, ctx, min_conf, out_confidence);
    return myds_dspark_eval_stages_prefix(ds, ctx, min_conf, out_confidence,
        ds ? ds->current_output_slot : 0u, ds ? ds->current_output_count : 0u);
}
#endif

static void dspark_reject_proposal(myds_dspark_t *ds, myds_context_t *ctx, bool confidence) {
    ds->draft_count = ds->draft_tree_count = 0;
    ds->is_high_confidence = false;
    ds->consecutive_misses++;
    if (ds->consecutive_misses >= 4) ds->backoff_counter = 8;
    else if (ds->consecutive_misses >= 3) ds->backoff_counter = 4;
    if (ctx->decode_metrics_active) {
        ctx->decode_stats.draft_rejects++;
        if (confidence) ctx->decode_stats.confidence_rejects++;
        else ctx->decode_stats.margin_rejects++;
    }
}

static uint32_t dspark_propose_at(myds_dspark_t *ds, myds_context_t *ctx,
    int32_t last_token, uint32_t position, int32_t markov_seed, uint32_t first_slot, uint32_t draft_cap) {
    if (!ds || !ctx || !ctx->model) return 0;
    ds->draft_count = ds->draft_tree_count = 0;
    ds->is_high_confidence = false;
    if (ds->backoff_counter > 0) {
        ds->backoff_counter--;
        return 0;
    }

    double t0 = now_ms();
    ds->cycles++;
    ds->current_seed_token = last_token;
    ds->current_seed_position = position;
    ds->current_output_slot = first_slot;

    uint32_t K = draft_cap > 0 && draft_cap < ds->block_size ? draft_cap : ds->block_size;
    if (K > MYDS_DSPARK_DEFAULT_BLOCK_SIZE - first_slot) K = MYDS_DSPARK_DEFAULT_BLOCK_SIZE - first_slot;
    ds->current_output_count = K;

    const int markov_only = dspark_markov_only();
    if (!markov_only && ds->main_proj && !capture_consumer_ready(ds)) {
        ds->capture_counters.stale_rejections++;
        return 0;
    }

    static float s_cached_markov_scale = -999.0f;
    static int s_markov_scale_logged = 0;
    float markov_scale = 1.0f;
    if (__builtin_expect(s_cached_markov_scale < -500.0f, 0)) {
        const char *no_m = getenv("DS4_DSPARK_NO_MARKOV");
        if (!no_m) no_m = getenv("MYDS_DSPARK_NO_MARKOV");
        const bool markov_disabled = (no_m && no_m[0] && strcmp(no_m, "0") != 0);

        const char *scale_env = getenv("MYDS_DSPARK_MARKOV_SCALE");
        if (scale_env && scale_env[0]) {
            markov_scale = (float)atof(scale_env);
        } else if (markov_disabled) {
            markov_scale = 0.0f;
        } else {
            markov_scale = 1.0f;
        }
        s_cached_markov_scale = markov_scale;
        if (!s_markov_scale_logged) {
            s_markov_scale_logged = 1;
            MYDS_LOG_I("DSpark Markov bias additive scale: %.2f (no_markov=%d)", markov_scale, (int)markov_disabled);
        }
    } else {
        markov_scale = s_cached_markov_scale;
    }

    static float min_conf = -1.0f;
    if (__builtin_expect(min_conf < -0.5f, 0)) {
        const char *env = getenv("MYDS_DSPARK_MIN_CONF");
        min_conf = env ? (float)atof(env) : 0.0f; // Default 0.0f so authoritative verifier always runs
        MYDS_LOG_I("DSpark confidence head threshold: %.2f", min_conf);
    }

    int32_t prev = markov_seed;
    uint32_t n_proposed = 0;
    const uint32_t vocab_size = ctx->model->vocab_size;

    // Markov state update: Load w1[prev] into markov_state_host unconditionally so confidence head
    // [ws->final_embd; markov_state_host] is always correctly conditioned on previous token.
    const bool is_w1_f16 = (ds->markov_w1_type == GGML_TYPE_F16);
    const bool is_w1_q8  = (ds->markov_w1_type == GGML_TYPE_Q8_0);
    const void *w1_base  = ds->markov_w1 ? myds_metal_buffer_data(ds->markov_w1) : NULL;

    if (w1_base && ds->markov_state_host && (is_w1_f16 || is_w1_q8) && prev >= 0 && (uint32_t)prev < vocab_size) {
        if (is_w1_f16) {
            const uint16_t *w1_row = ((const uint16_t *)w1_base) + (size_t)prev * 256;
#if defined(__ARM_NEON)
            for (int k = 0; k < 256; k += 8) {
                float16x8_t h = vld1q_f16((const float16_t *)(w1_row + k));
                vst1q_f32(ds->markov_state_host + k, vcvt_f32_f16(vget_low_f16(h)));
                vst1q_f32(ds->markov_state_host + k + 4, vcvt_f32_f16(vget_high_f16(h)));
            }
#else
            for (int k = 0; k < 256; k++) {
                ds->markov_state_host[k] = dspark_fp16_to_fp32(w1_row[k]);
            }
#endif
        } else if (is_w1_q8) {
            const q8_0_block_t *w1_row = ((const q8_0_block_t *)w1_base) + (size_t)prev * 8;
            for (int b = 0; b < 8; b++) {
                float d_scale = dspark_fp16_to_fp32(w1_row[b].d);
                for (int k = 0; k < 32; k++) {
                    ds->markov_state_host[b * 32 + k] = d_scale * (float)w1_row[b].qs[k];
                }
            }
        }
    }

    const bool is_f16 = (ds->markov_w1_type == GGML_TYPE_F16 && ds->markov_w2_type == GGML_TYPE_F16);
    const bool is_q8  = (ds->markov_w1_type == GGML_TYPE_Q8_0 && ds->markov_w2_type == GGML_TYPE_Q8_0);
    const bool has_markov = (markov_scale > 0.0f &&
                             ds->markov_w1 && ds->markov_w2 && (is_f16 || is_q8) &&
                             ds->markov_state_host && ds->markov_bias_host);
    const void *w2_base = has_markov ? myds_metal_buffer_data(ds->markov_w2) : NULL;

    float confidence = 1.0f;
    const float *base_logits = NULL;

    if (!markov_only) {
        if (!ds->main_proj || !ds->target_hidden_buf) return 0;
        if (!DSPARK_EVAL_STAGES(ds, ctx, min_conf, &confidence)) {
            if (isfinite(confidence) && min_conf > 0.0f && confidence < min_conf)
                dspark_reject_proposal(ds, ctx, true);
            ds->propose_ms += now_ms() - t0;
            return 0;
        }
        base_logits = (const float *)myds_metal_buffer_data(ds->draft_logits_buf);
    }

    static uint64_t s_log_cycle = 0;
    if ((s_log_cycle++ % 50) == 0 || s_log_cycle <= 5) {
        MYDS_LOG_I("[DSpark Propose #%llu] conf=%.4f (min=%.2f, passed=%s), base_logits=%s",
                   (unsigned long long)ds->cycles, confidence, min_conf,
                   (!markov_only && min_conf > 0.0f && confidence < min_conf) ? "NO" : "YES",
                   base_logits ? "valid" : "none");
    }

    // The stages have already run; repeated rejections must also back off.
    if (!markov_only && min_conf > 0.0f && confidence < min_conf) {
        dspark_reject_proposal(ds, ctx, true);
        ds->propose_ms += (now_ms() - t0);
        return 0;
    }
    ds->is_high_confidence = false;

    if (!has_markov && !base_logits) return 0;

    const bool has_conf_proj = (ds->confidence_proj &&
                                (ds->confidence_proj_type == GGML_TYPE_F16 ||
                                 ds->confidence_proj_type == GGML_TYPE_Q8_0));
    myds_dspark_workspace_t *ws = ds->ws;

    for (uint32_t d = 0; d < K; d++) {
        const uint32_t slot = first_slot + d;
        const float *slot_logits = base_logits ? (base_logits + (size_t)slot * vocab_size) : NULL;

        // Confidence always conditions on this slot's preceding proposal,
        // including when additive Markov bias is disabled.
        if (d > 0 && w1_base && ds->markov_state_host && prev >= 0 && (uint32_t)prev < vocab_size) {
            if (is_w1_f16) {
                const uint16_t *w1_row = ((const uint16_t *)w1_base) + (size_t)prev * 256;
#if defined(__ARM_NEON)
                for (int k = 0; k < 256; k += 8) {
                    float16x8_t h = vld1q_f16((const float16_t *)(w1_row + k));
                    vst1q_f32(ds->markov_state_host + k, vcvt_f32_f16(vget_low_f16(h)));
                    vst1q_f32(ds->markov_state_host + k + 4, vcvt_f32_f16(vget_high_f16(h)));
                }
#else
                for (int k = 0; k < 256; k++) {
                    ds->markov_state_host[k] = dspark_fp16_to_fp32(w1_row[k]);
                }
#endif
            } else if (is_w1_q8) {
                const q8_0_block_t *w1_row = ((const q8_0_block_t *)w1_base) + (size_t)prev * 8;
                for (int b = 0; b < 8; b++) {
                    float d_scale = dspark_fp16_to_fp32(w1_row[b].d);
                    for (int k = 0; k < 32; k++) {
                        ds->markov_state_host[b * 32 + k] = d_scale * (float)w1_row[b].qs[k];
                    }
                }
            }
        }

        if (d > 0 && min_conf > 0.0f && has_conf_proj && ws) {
            float slot_conf = 1.0f;
            const float *final_embd_d = (const float *)myds_metal_buffer_data(ws->v_final_embd[slot]);
            if (final_embd_d) {
                float conf_score = 0.0f;
                if (ds->confidence_proj_type == GGML_TYPE_F16) {
                    const uint16_t *w_conf = (const uint16_t *)myds_metal_buffer_data(ds->confidence_proj);
                    conf_score = dot_f16_row(w_conf, final_embd_d, MYDS_DSPARK_DIM);
                    if (ds->markov_state_host) {
                        conf_score += dot_f16_row(w_conf + MYDS_DSPARK_DIM, ds->markov_state_host, ds->markov_rank);
                    }
                } else if (ds->confidence_proj_type == GGML_TYPE_Q8_0) {
                    const q8_0_block_t *w_conf = (const q8_0_block_t *)myds_metal_buffer_data(ds->confidence_proj);
                    conf_score = dot_q8_row(w_conf, final_embd_d, MYDS_DSPARK_DIM / 32);
                    if (ds->markov_state_host) {
                        conf_score += dot_q8_row(w_conf + MYDS_DSPARK_DIM / 32, ds->markov_state_host, ds->markov_rank / 32);
                    }
                }
                conf_score += ds->confidence_bias;
                slot_conf = 1.0f / (1.0f + expf(-conf_score));
            }
            if (slot_conf < min_conf) {
                break;
            }
        }

        int32_t best_token = 0;
        int32_t best_token2 = 0;
        float best_score = -INFINITY;
        float best_score2 = -INFINITY;

        if (has_markov && w1_base && w2_base && prev >= 0 && (uint32_t)prev < vocab_size) {

#if defined(__APPLE__)
            int32_t slice_tokens[MARKOV_NUM_SLICES];
            float slice_scores[MARKOV_NUM_SLICES];
            int32_t slice_tokens2[MARKOV_NUM_SLICES];
            float slice_scores2[MARKOV_NUM_SLICES];
            int32_t *tokens_out = slice_tokens;
            float *scores_out = slice_scores;
            int32_t *tokens2_out = slice_tokens2;
            float *scores2_out = slice_scores2;
            uint32_t chunk = (vocab_size + MARKOV_NUM_SLICES - 1) / MARKOV_NUM_SLICES;

            dispatch_apply(MARKOV_NUM_SLICES, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t s) {
                uint32_t r0 = (uint32_t)s * chunk;
                uint32_t r1 = r0 + chunk;
                if (r1 > vocab_size) r1 = vocab_size;

                int32_t local_t1 = 0, local_t2 = 0;
                float local_s1 = -INFINITY, local_s2 = -INFINITY;

                if (is_f16) {
                    const uint16_t *w2_base_f16 = (const uint16_t *)w2_base;
                    for (uint32_t r = r0; r < r1; r++) {
                        const uint16_t *w2_row = w2_base_f16 + (size_t)r * 256;
                        float bias = dot_f16_row256(w2_row, ds->markov_state_host);
                        float score = slot_logits ? (slot_logits[r] + markov_scale * bias) : bias;
                        dspark_update_top2((int32_t)r, score, &local_t1, &local_s1, &local_t2, &local_s2);
                    }
                } else if (is_q8) {
                    const q8_0_block_t *w2_base_q8 = (const q8_0_block_t *)w2_base;
                    for (uint32_t r = r0; r < r1; r++) {
                        const q8_0_block_t *w2_row = w2_base_q8 + (size_t)r * 8;
                        float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                        float score = slot_logits ? (slot_logits[r] + markov_scale * bias) : bias;
                        dspark_update_top2((int32_t)r, score, &local_t1, &local_s1, &local_t2, &local_s2);
                    }
                }
                tokens_out[s] = local_t1;
                scores_out[s] = local_s1;
                tokens2_out[s] = local_t2;
                scores2_out[s] = local_s2;
            });

            for (int s = 0; s < MARKOV_NUM_SLICES; s++) {
                dspark_update_top2(slice_tokens[s], slice_scores[s], &best_token, &best_score, &best_token2, &best_score2);
                dspark_update_top2(slice_tokens2[s], slice_scores2[s], &best_token, &best_score, &best_token2, &best_score2);
            }
#else
            if (is_f16) {
                const uint16_t *w2_base_f16 = (const uint16_t *)w2_base;
                for (uint32_t r = 0; r < vocab_size; r++) {
                    const uint16_t *w2_row = w2_base_f16 + (size_t)r * 256;
                    float bias = dot_f16_row256(w2_row, ds->markov_state_host);
                    float score = slot_logits ? (slot_logits[r] + markov_scale * bias) : bias;
                    dspark_update_top2((int32_t)r, score, &best_token, &best_score, &best_token2, &best_score2);
                }
            } else if (is_q8) {
                const q8_0_block_t *w2_base_q8 = (const q8_0_block_t *)w2_base;
                for (uint32_t r = 0; r < vocab_size; r++) {
                    const q8_0_block_t *w2_row = w2_base_q8 + (size_t)r * 8;
                    float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                    float score = slot_logits ? (slot_logits[r] + markov_scale * bias) : bias;
                    dspark_update_top2((int32_t)r, score, &best_token, &best_score, &best_token2, &best_score2);
                }
            }
#endif
        } else if (slot_logits) {
            // Simple argmax / top2 fallback
            for (uint32_t r = 0; r < vocab_size; r++) {
                dspark_update_top2((int32_t)r, slot_logits[r], &best_token, &best_score, &best_token2, &best_score2);
            }
        }

        if (!isfinite(best_score) || !isfinite(best_score2)) {
            ds->draft_count = ds->draft_tree_count = 0;
            ds->propose_ms += now_ms() - t0;
            return 0;
        }

        if (d == 0) {
            ds->draft_tree_tokens[0] = best_token;
            ds->draft_tree_scores[0] = best_score;
            ds->draft_tree_tokens[1] = best_token2;
            ds->draft_tree_scores[1] = best_score2;
            ds->draft_tree_count = 2;

            static float min_margin = -1.0f;
            if (__builtin_expect(min_margin < -0.5f, 0)) {
                const char *env = getenv("MYDS_DSPARK_MIN_MARGIN");
                min_margin = env ? (float)atof(env) : 0.0f; // Default 0.0f so authoritative verifier always runs
                MYDS_LOG_I("DSpark logit margin gating active: threshold=%.2f", min_margin);
            }

            const float margin = best_score - best_score2;
            if (min_margin > 0.0f && margin < min_margin) {
                dspark_reject_proposal(ds, ctx, false);
                ds->propose_ms += (now_ms() - t0);
                return 0;
            }

            static float high_conf_margin = -1.0f;
            if (__builtin_expect(high_conf_margin < -0.5f, 0)) {
                const char *env = getenv("MYDS_DSPARK_HIGH_CONF_MARGIN");
                high_conf_margin = env ? (float)atof(env) : 0.0f;
                MYDS_LOG_I("DSpark high-confidence speculative margin active: threshold=%.2f", high_conf_margin);
            }

            static float high_conf_thresh = -1.0f;
            if (__builtin_expect(high_conf_thresh < -0.5f, 0)) {
                const char *env = getenv("MYDS_DSPARK_HIGH_CONF_THRESH");
                high_conf_thresh = env ? (float)atof(env) : 0.0f;
                MYDS_LOG_I("DSpark high-confidence speculative threshold active: threshold=%.2f", high_conf_thresh);
            }

            const bool conf_ok = isnan(confidence) ? (high_conf_thresh <= 0.0f) : (confidence >= high_conf_thresh);
            const bool margin_ok = margin >= high_conf_margin;
            if (conf_ok || margin_ok) {
                ds->is_high_confidence = true;
            }

            static int s_prop_diag_count = 0;
            static int s_prop_diag_enabled = -1;
            if (s_prop_diag_enabled < 0) {
                const char *env = getenv("MYDS_DSPARK_DEBUG");
                s_prop_diag_enabled = env ? atoi(env) : 0;
            }
            if (s_prop_diag_enabled || s_prop_diag_count < 30) {
                s_prop_diag_count++;
                int32_t neural_cand = -1;
                float neural_score = -INFINITY;
                if (slot_logits) {
                    for (uint32_t r = 0; r < vocab_size; r++) {
                        if (slot_logits[r] > neural_score) {
                            neural_score = slot_logits[r];
                            neural_cand = (int32_t)r;
                        }
                    }
                }
                char str_seed[96], str_cand_a[96], str_cand_b[96], str_neural[96];
                const myds_tokenizer_t *tok = (ctx && ctx->model) ? ctx->model->tokenizer : NULL;
                dspark_format_token_preview(tok, last_token, str_seed, sizeof(str_seed));
                dspark_format_token_preview(tok, best_token, str_cand_a, sizeof(str_cand_a));
                dspark_format_token_preview(tok, best_token2, str_cand_b, sizeof(str_cand_b));
                dspark_format_token_preview(tok, neural_cand, str_neural, sizeof(str_neural));
                MYDS_LOG_I("[dspark-prop] cycle=%llu seed=%s cand_a=%s (s=%.2f) cand_b=%s (s=%.2f) neural=%s (s=%.2f) margin=%.2f conf=%.3f high_conf=%d",
                           (unsigned long long)ds->cycles, str_seed,
                           str_cand_a, best_score, str_cand_b, best_score2,
                           str_neural, neural_score, margin, confidence, (int)ds->is_high_confidence);
            }
        }

        ds->draft_tokens[n_proposed++] = best_token;
        prev = best_token;
    }

    static int s_seq_diag_count = 0;
    static int s_seq_diag_enabled = -1;
    if (s_seq_diag_enabled < 0) {
        const char *env = getenv("MYDS_DSPARK_DEBUG");
        s_seq_diag_enabled = env ? atoi(env) : 0;
    }
    if ((s_seq_diag_enabled || s_seq_diag_count < 30) && n_proposed > 0) {
        s_seq_diag_count++;
        char seq_buf[256] = {0};
        size_t off = 0;
        const myds_tokenizer_t *tok = (ctx && ctx->model) ? ctx->model->tokenizer : NULL;
        for (uint32_t d = 0; d < n_proposed && off < sizeof(seq_buf) - 24; d++) {
            char tok_str[48];
            dspark_format_token_preview(tok, ds->draft_tokens[d], tok_str, sizeof(tok_str));
            off += snprintf(seq_buf + off, sizeof(seq_buf) - off, "%s%s", d > 0 ? " " : "", tok_str);
        }
        MYDS_LOG_I("[dspark-seq] cycle=%llu proposed=%u tokens=[%s]",
                   (unsigned long long)ds->cycles, n_proposed, seq_buf);
    }

    ds->draft_count = n_proposed;
    ds->proposed_tokens += (ctx && ctx->v41 ? 1 : n_proposed);
    ds->propose_ms += (now_ms() - t0);

    return n_proposed;
}

uint32_t myds_dspark_propose(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token, uint32_t draft_cap) {
    return dspark_propose_at(ds, ctx, last_token, ctx ? ctx->pos : 0, last_token, 0, draft_cap);
}

uint32_t myds_dspark_propose_next(myds_dspark_t *ds, myds_context_t *ctx,
    int32_t consumed_token, int32_t emitted_seed, uint32_t draft_cap) {
    if (!ds || !ctx || !ctx->v41 || !ctx->pos) {
        if (ds) { ds->draft_count = ds->draft_tree_count = 0; ds->is_high_confidence = false; }
        return 0;
    }
    if (dspark_markov_only()) return myds_dspark_propose(ds, ctx, emitted_seed, draft_cap);
    return dspark_propose_at(ds, ctx, consumed_token, ctx->pos - 1u, emitted_seed, 1u, draft_cap);
}

uint32_t myds_dspark_propose_tree(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token) {
    if (!ds || !ctx) return 0;
    return myds_dspark_propose(ds, ctx, last_token, 1);
}
