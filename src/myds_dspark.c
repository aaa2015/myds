#include "myds_dspark.h"
#include "myds_cpu.h"
#include "myds_gguf.h"
#include "myds_metal.h"
#include "myds_v41.h"
#include "myds_v41_gpu.h"
#include "myds_v41_moe.h"
#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

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
    if (ds->stage_count > MYDS_DSPARK_MAX_STAGES) ds->stage_count = MYDS_DSPARK_MAX_STAGES;

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

    // Target capture layer ids for DeepSeek-V4.1: load from GGUF metadata if available, defaulting to {37, 38, 39}
    static const char *const target_layer_keys[] = {
        "deepseek41-dflash-draft.dflash.target_layer_ids",
        "dspark.target_layer_ids",
        "deepseek4.dspark.target_layer_ids",
        "deepseek4.dspark_target_layer_ids",
        "dflash.target_layer_ids",
        "dspark.target_layers",
        "deepseek41-dflash-draft.dflash.capture_layer_ids",
        "dflash.capture_layer_ids",
        "dspark.capture_layer_ids",
    };
    const gguf_kv_t *kv_cap = NULL;
    for (size_t k = 0; k < sizeof(target_layer_keys) / sizeof(target_layer_keys[0]); k++) {
        kv_cap = gguf_find_key(gguf, target_layer_keys[k]);
        if (kv_cap && kv_cap->type == GGUF_TYPE_ARRAY && kv_cap->value.val_arr.count >= MYDS_DSPARK_TARGET_LAYERS &&
            (kv_cap->value.val_arr.elem_type == GGUF_TYPE_UINT32 || kv_cap->value.val_arr.elem_type == GGUF_TYPE_INT32)) {
            break;
        }
        kv_cap = NULL;
    }

    if (kv_cap) {
        const uint32_t *arr = (const uint32_t *)kv_cap->value.val_arr.data;
        for (int i = 0; i < MYDS_DSPARK_TARGET_LAYERS; i++) {
            ds->target_layers[i] = arr[i];
        }
    } else {
        ds->target_layers[0] = 37;
        ds->target_layers[1] = 38;
        ds->target_layers[2] = 39;
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
    ds->draft_logits_buf = myds_metal_buffer_alloc(vocab_size * sizeof(float));
    ds->markov_state_host = calloc(ds->markov_rank, sizeof(float));
    ds->markov_bias_host = calloc(vocab_size, sizeof(float));
    ds->spec_snap = calloc(1, sizeof(myds_v41_spec_snapshot_t));

    // Allocate 3-Stage Transformer Workspace (Multi-row: Target Row and Draft Row)
    myds_dspark_workspace_t *ws = calloc(1, sizeof(*ws));
    if (ws) {
        ws->stage_target_hc = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->stage_draft_hc = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->stage_hc = ws->stage_draft_hc;
        ws->after_attn_hc = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->target_flat_hc_norm = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->draft_flat_hc_norm = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->flat_hc_norm = ws->draft_flat_hc_norm;
        ws->target_hc_mix = myds_metal_buffer_alloc(24 * sizeof(float));
        ws->draft_hc_mix = myds_metal_buffer_alloc(24 * sizeof(float));
        ws->hc_mix = ws->draft_hc_mix;
        ws->target_hc_split = myds_metal_buffer_alloc(24 * sizeof(float));
        ws->draft_hc_split = myds_metal_buffer_alloc(24 * sizeof(float));
        ws->hc_split = ws->draft_hc_split;

        ws->target_attn_cur = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->draft_attn_cur = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->attn_cur = ws->draft_attn_cur;
        ws->target_attn_norm = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->draft_attn_norm = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->attn_norm = ws->draft_attn_norm;
        ws->target_kv = myds_metal_buffer_alloc(512 * sizeof(float));
        ws->target_kv_norm = myds_metal_buffer_alloc(512 * sizeof(float));
        ws->draft_kv = myds_metal_buffer_alloc(512 * sizeof(float));
        ws->draft_kv_norm = myds_metal_buffer_alloc(512 * sizeof(float));
        ws->kv = ws->draft_kv;
        ws->kv_norm = ws->draft_kv_norm;
        ws->qr = myds_metal_buffer_alloc(1280 * sizeof(float));
        ws->qr_norm = myds_metal_buffer_alloc(1280 * sizeof(float));
        ws->q = myds_metal_buffer_alloc(64 * 512 * sizeof(float));

        for (int s = 0; s < MYDS_DSPARK_MAX_STAGES; s++) {
            ws->stage_kv_cache[s] = myds_metal_buffer_alloc(128 * 512 * sizeof(float));
            if (ws->stage_kv_cache[s]) {
                memset(myds_metal_buffer_data(ws->stage_kv_cache[s]), 0, 128 * 512 * sizeof(float));
            }
        }

        ws->attn_heads = myds_metal_buffer_alloc(64 * 512 * sizeof(float));
        ws->attn_low = myds_metal_buffer_alloc(8192 * sizeof(float));
        ws->attn_out = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));

        ws->ffn_cur = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->ffn_norm = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->shared_gate = myds_metal_buffer_alloc(MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->shared_up = myds_metal_buffer_alloc(MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->shared_swiglu = myds_metal_buffer_alloc(MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->shared_out = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));

        ws->router_logits = myds_metal_buffer_alloc(128 * sizeof(float));
        ws->selected_ids = myds_metal_buffer_alloc(6 * sizeof(int32_t));
        ws->selected_weights = myds_metal_buffer_alloc(6 * sizeof(float));
        ws->moe_gate = myds_metal_buffer_alloc(6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->moe_up = myds_metal_buffer_alloc(6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->moe_mid = myds_metal_buffer_alloc(6 * MYDS_DSPARK_EXP_DIM * sizeof(float));
        ws->moe_down = myds_metal_buffer_alloc(6 * MYDS_DSPARK_DIM * sizeof(float));
        ws->moe_out = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->ffn_total = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));

        ws->final_embd = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->final_norm_out = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));

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
    if (ws->stage_target_hc) myds_metal_buffer_free(ws->stage_target_hc);
    if (ws->stage_draft_hc) myds_metal_buffer_free(ws->stage_draft_hc);
    if (ws->after_attn_hc) myds_metal_buffer_free(ws->after_attn_hc);
    if (ws->target_flat_hc_norm) myds_metal_buffer_free(ws->target_flat_hc_norm);
    if (ws->draft_flat_hc_norm) myds_metal_buffer_free(ws->draft_flat_hc_norm);
    if (ws->target_hc_mix) myds_metal_buffer_free(ws->target_hc_mix);
    if (ws->draft_hc_mix) myds_metal_buffer_free(ws->draft_hc_mix);
    if (ws->target_hc_split) myds_metal_buffer_free(ws->target_hc_split);
    if (ws->draft_hc_split) myds_metal_buffer_free(ws->draft_hc_split);

    if (ws->target_attn_cur) myds_metal_buffer_free(ws->target_attn_cur);
    if (ws->draft_attn_cur) myds_metal_buffer_free(ws->draft_attn_cur);
    if (ws->target_attn_norm) myds_metal_buffer_free(ws->target_attn_norm);
    if (ws->draft_attn_norm) myds_metal_buffer_free(ws->draft_attn_norm);
    if (ws->target_kv) myds_metal_buffer_free(ws->target_kv);
    if (ws->target_kv_norm) myds_metal_buffer_free(ws->target_kv_norm);
    if (ws->draft_kv) myds_metal_buffer_free(ws->draft_kv);
    if (ws->draft_kv_norm) myds_metal_buffer_free(ws->draft_kv_norm);
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

int myds_dspark_eval_stages(myds_dspark_t *ds, myds_context_t *ctx, float min_conf, float *out_confidence) {
    if (!ds || !ctx || !ctx->model) return 0;
    if (!capture_consumer_ready(ds)) {
        ds->capture_counters.stale_rejections++;
        return 0;
    }
    if (!ds->ws) return 0;
    if (!ds->main_proj || !ds->target_hidden_buf || !ds->draft_logits_buf) return 0;

    myds_dspark_workspace_t *ws = ds->ws;
    const uint32_t in_dim = MYDS_DSPARK_TARGET_LAYERS * MYDS_DSPARK_DIM;
    const size_t stream_bytes = (size_t)MYDS_DSPARK_DIM * sizeof(float);

    // 1. Initialize Row 1 (Draft Row) from seed token embedding (replicated 4x across 4 HC streams)
    float *draft_hc_ptr = (float *)myds_metal_buffer_data(ws->stage_draft_hc);
    bool seeded_draft_hc = false;
    if (draft_hc_ptr && ds->current_seed_token >= 0 &&
        dspark_lookup_token_embedding(ctx->model, ds->current_seed_token, draft_hc_ptr)) {
        for (size_t c = 1; c < 4; c++) {
            memcpy(draft_hc_ptr + c * MYDS_DSPARK_DIM, draft_hc_ptr, stream_bytes);
        }
        seeded_draft_hc = true;
    }

    // Begin batched Metal command buffer for stages cascade
    (void)myds_metal_begin_batch();

    // 2. Stage 0 Entry Projection: target_hidden -> main_x [15360] -> [5120]
    if (!myds_metal_gemv(ds->stage_x_buf, ds->main_proj, ds->target_hidden_buf,
                         MYDS_DSPARK_DIM, in_dim, ds->main_proj_type)) {
        (void)myds_metal_commit_batch();
        return 0;
    }
    if (ds->main_norm) {
        if (!myds_metal_v41_rmsnorm(ds->stage_x_buf, ds->stage_x_buf, ds->main_norm,
                                    MYDS_DSPARK_DIM, 1e-6f, true)) {
            (void)myds_metal_commit_batch();
            return 0;
        }
    }

    // 2b. Initialize Row 0 (Target Row) from main_x (stage_x_buf replicated 4x across 4 HC streams)
    // Row 0 remains CONSTANT across all 3 stages!
    for (size_t c = 0; c < 4; c++) {
        (void)myds_metal_buffer_copy(ws->stage_target_hc, c * stream_bytes, ds->stage_x_buf, 0, stream_bytes);
    }

    // Fallback: If seed token embedding lookup was not possible, initialize Draft Row from main_x
    if (!seeded_draft_hc) {
        for (size_t c = 0; c < 4; c++) {
            (void)myds_metal_buffer_copy(ws->stage_draft_hc, c * stream_bytes, ds->stage_x_buf, 0, stream_bytes);
        }
    }

    // Precompute Target Row flat HC norm (constant across stages since stage_target_hc is unchanged)
    (void)myds_metal_v41_rmsnorm(ws->target_flat_hc_norm, ws->stage_target_hc, NULL, 4 * MYDS_DSPARK_DIM, 1e-6f, false);

    myds_v41_gpu_scratch_t *gpu_s = ws->gpu_scratch;

    // 3. Transformer Stages Cascade (0, 1, 2)
    for (uint32_t s = 0; s < ds->stage_count; s++) {
        myds_dspark_stage_t *st = &ds->stages[s];

        // --- Step A: Target Row (Row 0) Projection for Stage s ---
        // Mix & split for Target Row
        if (st->hc_attn_fn) {
            (void)myds_metal_gemv(ws->target_hc_mix, st->hc_attn_fn, ws->target_flat_hc_norm, 24, 4 * MYDS_DSPARK_DIM, st->hc_attn_fn_type);
            (void)myds_metal_v41_hc_split_batch(ws->target_hc_split, ws->target_hc_mix, st->hc_attn_scale, st->hc_attn_base, 1, 1e-5f);
            (void)myds_metal_v41_hc_weighted_sum_round(ws->target_attn_cur, ws->stage_target_hc, ws->target_hc_split, MYDS_DSPARK_DIM, 0);
        } else {
            (void)myds_metal_buffer_copy(ws->target_attn_cur, 0, ws->stage_target_hc, 0, stream_bytes);
        }

        // Target pre-norm
        if (st->attn_norm) {
            (void)myds_metal_v41_rmsnorm(ws->target_attn_norm, ws->target_attn_cur, st->attn_norm, MYDS_DSPARK_DIM, 1e-6f, true);
        } else {
            (void)myds_metal_buffer_copy(ws->target_attn_norm, 0, ws->target_attn_cur, 0, stream_bytes);
        }

        // Target KV projection -> Slot 0 of stage_kv_cache[s]
        if (st->attn_kv) {
            (void)myds_metal_gemv(ws->target_kv, st->attn_kv, ws->target_attn_norm, 512, MYDS_DSPARK_DIM, st->attn_kv_type);
            if (st->attn_kv_a_norm) {
                (void)myds_metal_v41_rmsnorm(ws->target_kv_norm, ws->target_kv, st->attn_kv_a_norm, 512, 1e-6f, true);
            }
            (void)myds_v41_gpu_rope(ws->target_kv_norm ? ws->target_kv_norm : ws->target_kv, 512, 1, ctx->pos, false, false);
            (void)myds_metal_buffer_copy(ws->stage_kv_cache[s], 0,
                                         ws->target_kv_norm ? ws->target_kv_norm : ws->target_kv, 0, 512 * sizeof(float));
        }

        // --- Step B: Draft Row (Row 1) Processing for Stage s ---
        // HC RMSNorm over flat 4 streams (20480) of Draft Row
        (void)myds_metal_v41_rmsnorm(ws->draft_flat_hc_norm, ws->stage_draft_hc, NULL, 4 * MYDS_DSPARK_DIM, 1e-6f, false);

        // Mix & split for Draft Row
        if (st->hc_attn_fn) {
            (void)myds_metal_gemv(ws->draft_hc_mix, st->hc_attn_fn, ws->draft_flat_hc_norm, 24, 4 * MYDS_DSPARK_DIM, st->hc_attn_fn_type);
            (void)myds_metal_v41_hc_split_batch(ws->draft_hc_split, ws->draft_hc_mix, st->hc_attn_scale, st->hc_attn_base, 1, 1e-5f);
            (void)myds_metal_v41_hc_weighted_sum_round(ws->draft_attn_cur, ws->stage_draft_hc, ws->draft_hc_split, MYDS_DSPARK_DIM, 0);
        } else {
            (void)myds_metal_buffer_copy(ws->draft_attn_cur, 0, ws->stage_draft_hc, 0, stream_bytes);
        }

        // Draft pre-norm
        if (st->attn_norm) {
            (void)myds_metal_v41_rmsnorm(ws->draft_attn_norm, ws->draft_attn_cur, st->attn_norm, MYDS_DSPARK_DIM, 1e-6f, true);
        } else {
            (void)myds_metal_buffer_copy(ws->draft_attn_norm, 0, ws->draft_attn_cur, 0, stream_bytes);
        }

        // Draft Q projection from draft_attn_norm: [5120] -> [1280] -> [32768]
        if (st->attn_q_a && st->attn_q_b) {
            (void)myds_metal_gemv(ws->qr, st->attn_q_a, ws->draft_attn_norm, 1280, MYDS_DSPARK_DIM, st->attn_q_a_type);
            if (st->attn_q_a_norm) {
                (void)myds_metal_v41_rmsnorm(ws->qr_norm, ws->qr, st->attn_q_a_norm, 1280, 1e-6f, true);
            }
            (void)myds_metal_gemv(ws->q, st->attn_q_b, ws->qr_norm ? ws->qr_norm : ws->qr, 64 * 512, 1280, st->attn_q_b_type);

            // Per-head RMSNorm on Q (64 heads x 512 head_dim) - critical fix for attention magnitude
            (void)myds_metal_v41_rmsnorm_batch(ws->q, ws->q, NULL, 512, 64, 1e-6f, false);

            // RoPE on Q
            (void)myds_v41_gpu_rope(ws->q, 512, 64, ctx->pos, false, false);
        }

        // Draft KV projection from draft_attn_norm: [5120] -> [512] -> Slot 1 of stage_kv_cache[s]
        if (st->attn_kv) {
            (void)myds_metal_gemv(ws->draft_kv, st->attn_kv, ws->draft_attn_norm, 512, MYDS_DSPARK_DIM, st->attn_kv_type);
            if (st->attn_kv_a_norm) {
                (void)myds_metal_v41_rmsnorm(ws->draft_kv_norm, ws->draft_kv, st->attn_kv_a_norm, 512, 1e-6f, true);
            }
            (void)myds_v41_gpu_rope(ws->draft_kv_norm ? ws->draft_kv_norm : ws->draft_kv, 512, 1, ctx->pos, false, false);
            (void)myds_metal_buffer_copy(ws->stage_kv_cache[s], 512 * sizeof(float),
                                         ws->draft_kv_norm ? ws->draft_kv_norm : ws->draft_kv, 0, 512 * sizeof(float));
        }

        // --- Step C: Direct Attention attending to BOTH keys: Slot 0 (target) and Slot 1 (draft), n_raw = 2 ---
        if (gpu_s) {
            (void)myds_v41_gpu_attention_direct(ws->attn_heads, st->attn_sinks, ws->q,
                                                ws->stage_kv_cache[s], 2, 0,
                                                NULL, NULL, 0, 0, gpu_s);
        }

        // RoPE round on attention heads
        (void)myds_v41_gpu_rope_round(ws->attn_heads, 512, 64, ctx->pos, false, true);

        // Attention Output Down-Up Projection
        if (st->attn_output_a && st->attn_output_b) {
            (void)myds_metal_attn_output_a(ws->attn_low, st->attn_output_a, ws->attn_heads,
                                           8, 4096, 1024, st->attn_output_a_type);
            (void)myds_metal_gemv(ws->attn_out, st->attn_output_b, ws->attn_low,
                                  MYDS_DSPARK_DIM, 8192, st->attn_output_b_type);
        }

        // HC Expand after Attention on Draft Row (round_bf16 = 0 for full F32 precision matching DSpark)
        if (st->hc_attn_fn) {
            (void)myds_metal_v41_hc_expand_round(ws->after_attn_hc, ws->attn_out, ws->stage_draft_hc, ws->draft_hc_split, MYDS_DSPARK_DIM, 0);
        } else {
            for (size_t c = 0; c < 4; c++) {
                (void)myds_metal_buffer_copy(ws->after_attn_hc, c * stream_bytes, ws->attn_out, 0, stream_bytes);
            }
        }

        // --- Step D: FFN / MoE Block on Draft Row ---
        // HC RMSNorm over flat 4 streams (20480) of after_attn_hc
        (void)myds_metal_v41_rmsnorm(ws->draft_flat_hc_norm, ws->after_attn_hc, NULL, 4 * MYDS_DSPARK_DIM, 1e-6f, false);

        // Project to 24 mix values
        if (st->hc_ffn_fn) {
            (void)myds_metal_gemv(ws->draft_hc_mix, st->hc_ffn_fn, ws->draft_flat_hc_norm, 24, 4 * MYDS_DSPARK_DIM, st->hc_ffn_fn_type);
            (void)myds_metal_v41_hc_split_batch(ws->draft_hc_split, ws->draft_hc_mix, st->hc_ffn_scale, st->hc_ffn_base, 1, 1e-5f);
            (void)myds_metal_v41_hc_weighted_sum_round(ws->ffn_cur, ws->after_attn_hc, ws->draft_hc_split, MYDS_DSPARK_DIM, 0);
        } else {
            (void)myds_metal_buffer_copy(ws->ffn_cur, 0, ws->after_attn_hc, 0, stream_bytes);
        }

        // FFN pre-norm
        if (st->ffn_norm) {
            (void)myds_metal_v41_rmsnorm(ws->ffn_norm, ws->ffn_cur, st->ffn_norm, MYDS_DSPARK_DIM, 1e-6f, true);
        } else {
            (void)myds_metal_buffer_copy(ws->ffn_norm, 0, ws->ffn_cur, 0, stream_bytes);
        }

        // 1. Shared Expert (SwiGLU)
        if (st->shared_gate && st->shared_up && st->shared_down) {
            (void)myds_metal_gemv(ws->shared_gate, st->shared_gate, ws->ffn_norm, MYDS_DSPARK_EXP_DIM, MYDS_DSPARK_DIM, st->shared_gate_type);
            (void)myds_metal_gemv(ws->shared_up, st->shared_up, ws->ffn_norm, MYDS_DSPARK_EXP_DIM, MYDS_DSPARK_DIM, st->shared_up_type);
            (void)myds_metal_swiglu(ws->shared_swiglu, ws->shared_gate, ws->shared_up, MYDS_DSPARK_EXP_DIM);
            (void)myds_metal_gemv(ws->shared_out, st->shared_down, ws->shared_swiglu, MYDS_DSPARK_DIM, MYDS_DSPARK_EXP_DIM, st->shared_down_type);
        }

        // 2. Routed MoE (128 experts, top 3)
        if (st->router_gate && st->ffn_gate_exps && st->ffn_up_exps && st->ffn_down_exps) {
            (void)myds_metal_gemv(ws->router_logits, st->router_gate, ws->ffn_norm, ds->n_routed_experts, MYDS_DSPARK_DIM, st->router_gate_type);

            float route_scale = (ctx && ctx->model && ctx->model->routed_scaling_factor > 0.0f) ?
                                ctx->model->routed_scaling_factor : 1.5f;

            myds_metal_barrier_buffer(ws->router_logits);
            if (!myds_dspark_route_fused(ws->selected_ids, ws->selected_weights,
                                         ws->router_logits, st->router_bias,
                                         ds->num_experts_per_tok, route_scale, 1)) {
                // Fallback to CPU route if GPU dispatch fails
                (void)myds_metal_commit_batch();
                int32_t *sel_ids = (int32_t *)myds_metal_buffer_data(ws->selected_ids);
                float *sel_weights = (float *)myds_metal_buffer_data(ws->selected_weights);
                (void)myds_metal_moe_route_biased(sel_ids, sel_weights, ws->router_logits, st->router_bias,
                                                  ds->n_routed_experts, ds->num_experts_per_tok, route_scale);
                (void)myds_metal_begin_batch();
            }

            float clamp_exp = (ctx && ctx->model && ctx->model->swiglu_clamp_exp > 0.0f) ?
                              ctx->model->swiglu_clamp_exp : 10.0f;
            (void)myds_metal_routed_moe(ws->moe_out, ws->ffn_norm,
                                        st->ffn_gate_exps, st->ffn_up_exps, st->ffn_down_exps,
                                        ws->selected_ids, ws->selected_weights,
                                        ws->moe_gate, ws->moe_up, ws->moe_mid, ws->moe_down,
                                        MYDS_DSPARK_DIM, MYDS_DSPARK_EXP_DIM,
                                        ds->n_routed_experts, ds->num_experts_per_tok,
                                        st->ffn_gate_exps_type, st->ffn_up_exps_type, st->ffn_down_exps_type,
                                        clamp_exp);

            (void)myds_metal_add(ws->ffn_total, ws->shared_out, ws->moe_out, MYDS_DSPARK_DIM);
        } else if (st->shared_gate && st->shared_down) {
            (void)myds_metal_buffer_copy(ws->ffn_total, 0, ws->shared_out, 0, stream_bytes);
        }

        // HC Expand after FFN -> stage_draft_hc for next stage (round_bf16 = 0 for full F32 precision)
        if (st->hc_ffn_fn) {
            (void)myds_metal_v41_hc_expand_round(ws->stage_draft_hc, ws->ffn_total, ws->after_attn_hc, ws->draft_hc_split, MYDS_DSPARK_DIM, 0);
        } else {
            for (size_t c = 0; c < 4; c++) {
                (void)myds_metal_buffer_copy(ws->stage_draft_hc, c * stream_bytes, ws->ffn_total, 0, stream_bytes);
            }
        }
    }

    // 4. Final Head on stage_draft_hc (mtp.2.norm + Base Model LM Head)
    myds_dspark_stage_t *final_st = &ds->stages[ds->stage_count - 1u];
    if (final_st->hc_ffn_fn) {
        (void)myds_metal_v41_rmsnorm(ws->draft_flat_hc_norm, ws->stage_draft_hc, NULL, 4 * MYDS_DSPARK_DIM, 1e-6f, false);
        (void)myds_metal_gemv(ws->draft_hc_mix, final_st->hc_ffn_fn, ws->draft_flat_hc_norm, 24, 4 * MYDS_DSPARK_DIM, final_st->hc_ffn_fn_type);
        (void)myds_metal_v41_hc_split_batch(ws->draft_hc_split, ws->draft_hc_mix, final_st->hc_ffn_scale, final_st->hc_ffn_base, 1, 1e-5f);
        (void)myds_metal_v41_hc_weighted_sum_round(ws->final_embd, ws->stage_draft_hc, ws->draft_hc_split, MYDS_DSPARK_DIM, 0);
    } else {
        (void)myds_metal_buffer_copy(ws->final_embd, 0, ws->stage_draft_hc, 0, stream_bytes);
    }

    if (ds->final_norm) {
        (void)myds_metal_v41_rmsnorm(ws->final_norm_out, ws->final_embd, ds->final_norm, MYDS_DSPARK_DIM, 1e-6f, true);
    } else {
        (void)myds_metal_buffer_copy(ws->final_norm_out, 0, ws->final_embd, 0, stream_bytes);
    }

    const bool has_conf_proj = (ds->confidence_proj &&
                                (ds->confidence_proj_type == GGML_TYPE_F16 ||
                                 ds->confidence_proj_type == GGML_TYPE_Q8_0));
    const bool can_early_exit = (has_conf_proj && min_conf > 0.0f);

    if (can_early_exit) {
        // Commit final stages batch to complete GPU work for hidden states so CPU can test confidence
        (void)myds_metal_commit_batch();

        float conf_prob = 1.0f;
        const float *final_embd_data = (const float *)myds_metal_buffer_data(ws->final_embd);
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

        // Zero-overhead early exit: If confidence is below threshold, skip the 129,280x5120 LM Head GEMV!
        if (conf_prob < min_conf) {
            return 0;
        }

        // Project to LM Head logits only when confidence test passed
        (void)myds_metal_begin_batch();
        (void)myds_metal_gemv(ds->draft_logits_buf, ctx->model->output, ws->final_norm_out,
                              ctx->model->vocab_size, MYDS_DSPARK_DIM, ctx->model->output_type);
        (void)myds_metal_commit_batch();
    } else {
        // Direct zero-stall pipelining: LM Head GEMV enqueued in the SAME command buffer as stages
        (void)myds_metal_gemv(ds->draft_logits_buf, ctx->model->output, ws->final_norm_out,
                              ctx->model->vocab_size, MYDS_DSPARK_DIM, ctx->model->output_type);
        (void)myds_metal_commit_batch();

        if (out_confidence) {
            float conf_prob = 1.0f;
            if (has_conf_proj) {
                const float *final_embd_data = (const float *)myds_metal_buffer_data(ws->final_embd);
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

    return 1;
}

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

uint32_t myds_dspark_propose(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token, uint32_t draft_cap) {
    if (!ds || !ctx || !ctx->model) return 0;
    if (ds->backoff_counter > 0) {
        ds->backoff_counter--;
        return 0;
    }

    double t0 = now_ms();
    ds->cycles++;
    ds->current_seed_token = last_token;

    uint32_t K = draft_cap > 0 && draft_cap < ds->block_size ? draft_cap : ds->block_size;
    if (K > MYDS_DSPARK_DEFAULT_BLOCK_SIZE) K = MYDS_DSPARK_DEFAULT_BLOCK_SIZE;

    const int markov_only = dspark_markov_only();
    if (!markov_only && ds->main_proj && !capture_consumer_ready(ds)) {
        ds->capture_counters.stale_rejections++;
        return 0;
    }

    static float s_cached_markov_scale = -1.0f;
    static int s_markov_scale_logged = 0;
    float markov_scale = 1.0f;
    const char *no_m = getenv("DS4_DSPARK_NO_MARKOV");
    if (!no_m) no_m = getenv("MYDS_DSPARK_NO_MARKOV");
    const bool markov_disabled = (no_m && no_m[0] && strcmp(no_m, "0") != 0);

    const char *scale_env = getenv("MYDS_DSPARK_MARKOV_SCALE");
    if (scale_env && scale_env[0]) {
        markov_scale = (float)atof(scale_env);
    } else if (markov_disabled) {
        markov_scale = 0.0f;
    } else if (s_cached_markov_scale >= 0.0f) {
        markov_scale = s_cached_markov_scale;
    }
    s_cached_markov_scale = markov_scale;

    if (!s_markov_scale_logged) {
        s_markov_scale_logged = 1;
        MYDS_LOG_I("DSpark Markov bias additive scale: %.2f (no_markov=%d)", markov_scale, (int)markov_disabled);
    }

    static float min_conf = -1.0f;
    if (__builtin_expect(min_conf < -0.5f, 0)) {
        const char *env = getenv("MYDS_DSPARK_MIN_CONF");
        min_conf = env ? (float)atof(env) : 0.0f; // Default 0.0f so authoritative verifier always runs
        MYDS_LOG_I("DSpark confidence head threshold: %.2f", min_conf);
    }

    int32_t prev = last_token;
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
    const bool has_markov = (!markov_disabled && markov_scale > 0.0f &&
                             ds->markov_w1 && ds->markov_w2 && (is_f16 || is_q8) &&
                             ds->markov_state_host && ds->markov_bias_host);
    const void *w2_base = has_markov ? myds_metal_buffer_data(ds->markov_w2) : NULL;

    float confidence = 1.0f;
    const float *base_logits = NULL;

    if (!markov_only && ds->main_proj && ds->target_hidden_buf) {
        if (myds_dspark_eval_stages(ds, ctx, min_conf, &confidence)) {
            base_logits = (const float *)myds_metal_buffer_data(ds->draft_logits_buf);
        }
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

    for (uint32_t d = 0; d < K; d++) {
        int32_t best_token = 0;
        int32_t best_token2 = 0;
        float best_score = -INFINITY;
        float best_score2 = -INFINITY;

        if (has_markov && w1_base && w2_base && prev >= 0 && (uint32_t)prev < vocab_size) {
            // Dequantize / load row `prev` if d > 0 (d == 0 already done)
            if (d > 0) {
                if (is_f16) {
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
                } else if (is_q8) {
                    const q8_0_block_t *w1_row = ((const q8_0_block_t *)w1_base) + (size_t)prev * 8;
                    for (int b = 0; b < 8; b++) {
                        float d_scale = dspark_fp16_to_fp32(w1_row[b].d);
                        for (int k = 0; k < 32; k++) {
                            ds->markov_state_host[b * 32 + k] = d_scale * (float)w1_row[b].qs[k];
                        }
                    }
                }
            }

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
                        float score = base_logits ? (base_logits[r] + markov_scale * bias) : bias;
                        dspark_update_top2((int32_t)r, score, &local_t1, &local_s1, &local_t2, &local_s2);
                    }
                } else if (is_q8) {
                    const q8_0_block_t *w2_base_q8 = (const q8_0_block_t *)w2_base;
                    for (uint32_t r = r0; r < r1; r++) {
                        const q8_0_block_t *w2_row = w2_base_q8 + (size_t)r * 8;
                        float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                        float score = base_logits ? (base_logits[r] + markov_scale * bias) : bias;
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
                    float score = base_logits ? (base_logits[r] + markov_scale * bias) : bias;
                    dspark_update_top2((int32_t)r, score, &best_token, &best_score, &best_token2, &best_score2);
                }
            } else if (is_q8) {
                const q8_0_block_t *w2_base_q8 = (const q8_0_block_t *)w2_base;
                for (uint32_t r = 0; r < vocab_size; r++) {
                    const q8_0_block_t *w2_row = w2_base_q8 + (size_t)r * 8;
                    float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                    float score = base_logits ? (base_logits[r] + markov_scale * bias) : bias;
                    dspark_update_top2((int32_t)r, score, &best_token, &best_score, &best_token2, &best_score2);
                }
            }
#endif
        } else if (base_logits) {
            // Simple argmax / top2 fallback
            for (uint32_t r = 0; r < vocab_size; r++) {
                dspark_update_top2((int32_t)r, base_logits[r], &best_token, &best_score, &best_token2, &best_score2);
            }
        }

        if (d == 0) {
            ds->draft_tree_tokens[0] = best_token;
            ds->draft_tree_scores[0] = best_score;
            ds->draft_tree_tokens[1] = best_token2;
            ds->draft_tree_scores[1] = best_score2;
            ds->draft_tree_count = 2;

            static int s_prop_diag_count = 0;
            static int s_prop_diag_enabled = -1;
            if (s_prop_diag_enabled < 0) {
                const char *env = getenv("MYDS_DSPARK_DEBUG");
                s_prop_diag_enabled = env ? atoi(env) : 0;
            }
            if (s_prop_diag_enabled || s_prop_diag_count < 15) {
                s_prop_diag_count++;
                int32_t neural_cand = -1;
                float neural_score = -INFINITY;
                if (base_logits) {
                    for (uint32_t r = 0; r < vocab_size; r++) {
                        if (base_logits[r] > neural_score) {
                            neural_score = base_logits[r];
                            neural_cand = (int32_t)r;
                        }
                    }
                }
                MYDS_LOG_I("[dspark-prop] cycle=%llu seed=%d cand_a=%d (s=%.2f) cand_b=%d (s=%.2f) neural=%d (s=%.2f) conf=%.3f",
                           (unsigned long long)ds->cycles, last_token,
                           best_token, best_score, best_token2, best_score2,
                           neural_cand, neural_score, confidence);
            }

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
                high_conf_margin = env ? (float)atof(env) : 1.50f;
                MYDS_LOG_I("DSpark high-confidence speculative margin active: threshold=%.2f", high_conf_margin);
            }

            static float high_conf_thresh = -1.0f;
            if (__builtin_expect(high_conf_thresh < -0.5f, 0)) {
                const char *env = getenv("MYDS_DSPARK_HIGH_CONF_THRESH");
                high_conf_thresh = env ? (float)atof(env) : 0.70f;
                MYDS_LOG_I("DSpark high-confidence speculative threshold active: threshold=%.2f", high_conf_thresh);
            }

            if (confidence >= high_conf_thresh || margin >= high_conf_margin) {
                ds->is_high_confidence = true;
            }
        }

        ds->draft_tokens[n_proposed++] = best_token;
        prev = best_token;
    }

    ds->draft_count = n_proposed;
    ds->proposed_tokens += n_proposed;
    ds->propose_ms += (now_ms() - t0);

    return n_proposed;
}

uint32_t myds_dspark_propose_tree(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token) {
    if (!ds || !ctx) return 0;
    return myds_dspark_propose(ds, ctx, last_token, 1);
}
