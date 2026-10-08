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

myds_dspark_t *myds_dspark_load(const char *path, uint32_t vocab_size) {
    if (!path || !vocab_size) return NULL;

    gguf_file_t *gguf = gguf_open(path);
    if (!gguf) {
        MYDS_LOG_E("dspark: failed to open GGUF file: %s", path);
        return NULL;
    }

    const char *arch = gguf_get_str(gguf, "general.architecture", NULL);
    if (!arch || strcmp(arch, "deepseek4-dspark") != 0) {
        MYDS_LOG_E("dspark: architecture mismatch (got %s, expected deepseek4-dspark)", arch ? arch : "null");
        gguf_close(gguf);
        return NULL;
    }

    myds_dspark_t *ds = calloc(1, sizeof(*ds));
    if (!ds) {
        gguf_close(gguf);
        return NULL;
    }
    ds->gf = gguf;

    ds->stage_count = gguf_get_u32(gguf, "dspark.stage_count", 3);
    if (ds->stage_count > MYDS_DSPARK_MAX_STAGES) ds->stage_count = MYDS_DSPARK_MAX_STAGES;

    ds->block_size = gguf_get_u32(gguf, "dspark.block_size", MYDS_DSPARK_DEFAULT_BLOCK_SIZE);
    ds->markov_rank = gguf_get_u32(gguf, "dspark.markov_rank", MYDS_DSPARK_MARKOV_RANK);
    ds->noise_token_id = gguf_get_u32(gguf, "dspark.noise_token_id", 128799);
    ds->n_routed_experts = gguf_get_u32(gguf, "dspark.n_routed_experts", 128);
    ds->num_experts_per_tok = gguf_get_u32(gguf, "dspark.num_experts_per_tok", 3);

    // Default target layer ids for DeepSeek-V4.1: {37, 38, 39}
    ds->target_layers[0] = 37;
    ds->target_layers[1] = 38;
    ds->target_layers[2] = 39;

    // Stage 0 entry projection
    const gguf_tensor_t *t_main_proj = gguf_find_tensor(gguf, "mtp.0.main_proj.weight");
    const gguf_tensor_t *t_main_norm = gguf_find_tensor(gguf, "mtp.0.main_norm.weight");
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
            if (t) { \
                st->field = create_metal_buf(t); \
                st->ftype = t->type; \
            } \
        } while (0)

        #define BIND_BUF(field, tname) do { \
            snprintf(name, sizeof(name), "mtp.%u." tname, s); \
            const gguf_tensor_t *t = gguf_find_tensor(gguf, name); \
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

    // Final stage heads (on stage ds->stage_count - 1, typically mtp.2)
    uint32_t final_s = ds->stage_count - 1u;
    snprintf(name, sizeof(name), "mtp.%u.norm.weight", final_s);
    const gguf_tensor_t *t_fnorm = gguf_find_tensor(gguf, name);
    if (t_fnorm) ds->final_norm = create_metal_buf(t_fnorm);

    snprintf(name, sizeof(name), "mtp.%u.markov_head.markov_w1.weight", final_s);
    const gguf_tensor_t *t_mw1 = gguf_find_tensor(gguf, name);
    if (!t_mw1) {
        snprintf(name, sizeof(name), "mtp.%u.markov_head.embed.weight", final_s);
        t_mw1 = gguf_find_tensor(gguf, name);
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
    if (t_mw2) {
        ds->markov_w2 = create_metal_buf(t_mw2);
        ds->markov_w2_type = t_mw2->type;
    }

    snprintf(name, sizeof(name), "mtp.%u.confidence_head.proj.weight", final_s);
    const gguf_tensor_t *t_cproj = gguf_find_tensor(gguf, name);
    if (t_cproj) {
        ds->confidence_proj = create_metal_buf(t_cproj);
        ds->confidence_proj_type = t_cproj->type;
        MYDS_LOG_I("confidence_proj: n_dims=%u, dims=[%llu, %llu], type=%d (%s)",
                   t_cproj->n_dims,
                   (unsigned long long)t_cproj->dims[0],
                   (unsigned long long)(t_cproj->n_dims > 1 ? t_cproj->dims[1] : 0),
                   (int)t_cproj->type, ggml_type_name(t_cproj->type));
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

    // Allocate 3-Stage Transformer Workspace
    myds_dspark_workspace_t *ws = calloc(1, sizeof(*ws));
    if (ws) {
        ws->stage_hc = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->after_attn_hc = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->flat_hc_norm = myds_metal_buffer_alloc(4 * MYDS_DSPARK_DIM * sizeof(float));
        ws->hc_mix = myds_metal_buffer_alloc(24 * sizeof(float));
        ws->hc_split = myds_metal_buffer_alloc(24 * sizeof(float));

        ws->attn_cur = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->attn_norm = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        ws->qr = myds_metal_buffer_alloc(1280 * sizeof(float));
        ws->qr_norm = myds_metal_buffer_alloc(1280 * sizeof(float));
        ws->q = myds_metal_buffer_alloc(64 * 512 * sizeof(float));
        ws->kv = myds_metal_buffer_alloc(512 * sizeof(float));
        ws->kv_norm = myds_metal_buffer_alloc(512 * sizeof(float));

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
    if (ws->stage_hc) myds_metal_buffer_free(ws->stage_hc);
    if (ws->after_attn_hc) myds_metal_buffer_free(ws->after_attn_hc);
    if (ws->flat_hc_norm) myds_metal_buffer_free(ws->flat_hc_norm);
    if (ws->hc_mix) myds_metal_buffer_free(ws->hc_mix);
    if (ws->hc_split) myds_metal_buffer_free(ws->hc_split);

    if (ws->attn_cur) myds_metal_buffer_free(ws->attn_cur);
    if (ws->attn_norm) myds_metal_buffer_free(ws->attn_norm);
    if (ws->qr) myds_metal_buffer_free(ws->qr);
    if (ws->qr_norm) myds_metal_buffer_free(ws->qr_norm);
    if (ws->q) myds_metal_buffer_free(ws->q);
    if (ws->kv) myds_metal_buffer_free(ws->kv);
    if (ws->kv_norm) myds_metal_buffer_free(ws->kv_norm);

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
    if (!ds || !residual || !ds->target_hidden_buf || !ds->hc_mean_weights) return;
    int slot = -1;
    for (int i = 0; i < MYDS_DSPARK_TARGET_LAYERS; i++) {
        if (ds->target_layers[i] == layer_idx) {
            slot = i;
            break;
        }
    }
    if (slot < 0 || !ds->target_slots[slot]) return;
    (void)myds_metal_v41_hc_weighted_sum(ds->target_slots[slot], residual, ds->hc_mean_weights, MYDS_DSPARK_DIM);
}

static inline float dot_q8_row(const q8_0_block_t *row, const float *state, uint32_t n_blocks) {
    float total = 0.0f;
#if defined(__ARM_NEON)
    for (uint32_t b = 0; b < n_blocks; b++) {
        float d = myds_fp16_to_fp32(row[b].d);
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
        float d = myds_fp16_to_fp32(row[b].d);
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

int myds_dspark_eval_stages(myds_dspark_t *ds, myds_context_t *ctx, float *out_confidence) {
    if (!ds || !ctx || !ctx->model || !ds->ws) return 0;
    if (!ds->main_proj || !ds->target_hidden_buf || !ds->draft_logits_buf) return 0;

    myds_dspark_workspace_t *ws = ds->ws;
    const uint32_t in_dim = MYDS_DSPARK_TARGET_LAYERS * MYDS_DSPARK_DIM;

    // 1. Stage 0 Entry Projection: [15360] -> [5120]
    if (!myds_metal_gemv(ds->stage_x_buf, ds->main_proj, ds->target_hidden_buf,
                         MYDS_DSPARK_DIM, in_dim, ds->main_proj_type)) {
        return 0;
    }
    if (ds->main_norm) {
        if (!myds_metal_v41_rmsnorm(ds->stage_x_buf, ds->stage_x_buf, ds->main_norm,
                                    MYDS_DSPARK_DIM, 1e-6f, true)) {
            return 0;
        }
    }

    // Initialize 4 HC streams from stage_x_buf (replicate 5120 floats 4 times)
    const size_t stream_bytes = (size_t)MYDS_DSPARK_DIM * sizeof(float);
    for (size_t c = 0; c < 4; c++) {
        (void)myds_metal_buffer_copy(ws->stage_hc, c * stream_bytes, ds->stage_x_buf, 0, stream_bytes);
    }

    myds_v41_gpu_scratch_t *gpu_s = ws->gpu_scratch;

    // 2. Transformer Stages Cascade (0, 1, 2)
    for (uint32_t s = 0; s < ds->stage_count; s++) {
        myds_dspark_stage_t *st = &ds->stages[s];

        // --- Attention Block ---
        // HC RMSNorm over flat 4 streams (20480)
        (void)myds_metal_v41_rmsnorm(ws->flat_hc_norm, ws->stage_hc, NULL, 4 * MYDS_DSPARK_DIM, 1e-6f, false);

        // Project to 24 mix values
        if (st->hc_attn_fn) {
            (void)myds_metal_gemv(ws->hc_mix, st->hc_attn_fn, ws->flat_hc_norm, 24, 4 * MYDS_DSPARK_DIM, st->hc_attn_fn_type);
            (void)myds_metal_v41_hc_split_batch(ws->hc_split, ws->hc_mix, st->hc_attn_scale, st->hc_attn_base, 1, 1e-5f);
            (void)myds_metal_v41_hc_weighted_sum(ws->attn_cur, ws->stage_hc, ws->hc_split, MYDS_DSPARK_DIM);
        } else {
            (void)myds_metal_buffer_copy(ws->attn_cur, 0, ws->stage_hc, 0, stream_bytes);
        }

        // Attention pre-norm
        if (st->attn_norm) {
            (void)myds_metal_v41_rmsnorm(ws->attn_norm, ws->attn_cur, st->attn_norm, MYDS_DSPARK_DIM, 1e-6f, true);
        } else {
            (void)myds_metal_buffer_copy(ws->attn_norm, 0, ws->attn_cur, 0, stream_bytes);
        }

        // Q projection: [5120] -> [1280] -> [32768]
        if (st->attn_q_a && st->attn_q_b) {
            (void)myds_metal_gemv(ws->qr, st->attn_q_a, ws->attn_norm, 1280, MYDS_DSPARK_DIM, st->attn_q_a_type);
            if (st->attn_q_a_norm) {
                (void)myds_metal_v41_rmsnorm(ws->qr_norm, ws->qr, st->attn_q_a_norm, 1280, 1e-6f, true);
            }
            (void)myds_metal_gemv(ws->q, st->attn_q_b, ws->qr_norm ? ws->qr_norm : ws->qr, 64 * 512, 1280, st->attn_q_b_type);
        }

        // KV projection: [5120] -> [512]
        if (st->attn_kv) {
            (void)myds_metal_gemv(ws->kv, st->attn_kv, ws->attn_norm, 512, MYDS_DSPARK_DIM, st->attn_kv_type);
            if (st->attn_kv_a_norm) {
                (void)myds_metal_v41_rmsnorm(ws->kv_norm, ws->kv, st->attn_kv_a_norm, 512, 1e-6f, true);
            }
        }

        // RoPE on Q & KV
        (void)myds_v41_gpu_rope(ws->q, 512, 64, ctx->pos, false, false);
        (void)myds_v41_gpu_rope(ws->kv_norm ? ws->kv_norm : ws->kv, 512, 1, ctx->pos, false, false);

        // Store into Stage KV ring buffer
        uint32_t ring_slot = (ctx->pos % 128);
        (void)myds_metal_buffer_copy(ws->stage_kv_cache[s], (size_t)ring_slot * 512 * sizeof(float),
                                    ws->kv_norm ? ws->kv_norm : ws->kv, 0, 512 * sizeof(float));

        // Direct Attention with sinks
        uint32_t nraw = (ctx->pos + 1u < 128) ? (ctx->pos + 1u) : 128;
        uint32_t raw_start = (ctx->pos + 1u - nraw) % 128;
        if (gpu_s) {
            (void)myds_v41_gpu_attention_direct(ws->attn_heads, st->attn_sinks, ws->q,
                                                ws->stage_kv_cache[s], nraw, raw_start,
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

        // HC Expand after Attention
        if (st->hc_attn_fn) {
            (void)myds_metal_v41_hc_expand(ws->after_attn_hc, ws->attn_out, ws->stage_hc, ws->hc_split, MYDS_DSPARK_DIM);
        } else {
            for (size_t c = 0; c < 4; c++) {
                (void)myds_metal_buffer_copy(ws->after_attn_hc, c * stream_bytes, ws->attn_out, 0, stream_bytes);
            }
        }

        // --- FFN / MoE Block ---
        // HC RMSNorm over flat 4 streams (20480)
        (void)myds_metal_v41_rmsnorm(ws->flat_hc_norm, ws->after_attn_hc, NULL, 4 * MYDS_DSPARK_DIM, 1e-6f, false);

        // Project to 24 mix values
        if (st->hc_ffn_fn) {
            (void)myds_metal_gemv(ws->hc_mix, st->hc_ffn_fn, ws->flat_hc_norm, 24, 4 * MYDS_DSPARK_DIM, st->hc_ffn_fn_type);
            (void)myds_metal_v41_hc_split_batch(ws->hc_split, ws->hc_mix, st->hc_ffn_scale, st->hc_ffn_base, 1, 1e-5f);
            (void)myds_metal_v41_hc_weighted_sum(ws->ffn_cur, ws->after_attn_hc, ws->hc_split, MYDS_DSPARK_DIM);
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
            (void)myds_metal_commit_batch();

            int32_t *sel_ids = (int32_t *)myds_metal_buffer_data(ws->selected_ids);
            float *sel_weights = (float *)myds_metal_buffer_data(ws->selected_weights);
            (void)myds_metal_moe_route_biased(sel_ids, sel_weights, ws->router_logits, st->router_bias,
                                              ds->n_routed_experts, ds->num_experts_per_tok, 2.5f);

            (void)myds_metal_routed_moe(ws->moe_out, ws->ffn_norm,
                                        st->ffn_gate_exps, st->ffn_up_exps, st->ffn_down_exps,
                                        ws->selected_ids, ws->selected_weights,
                                        ws->moe_gate, ws->moe_up, ws->moe_mid, ws->moe_down,
                                        MYDS_DSPARK_DIM, MYDS_DSPARK_EXP_DIM,
                                        ds->n_routed_experts, ds->num_experts_per_tok,
                                        st->ffn_gate_exps_type, st->ffn_up_exps_type, st->ffn_down_exps_type,
                                        0.0f);

            (void)myds_metal_add(ws->ffn_total, ws->shared_out, ws->moe_out, MYDS_DSPARK_DIM);
        } else if (st->shared_gate && st->shared_down) {
            (void)myds_metal_buffer_copy(ws->ffn_total, 0, ws->shared_out, 0, stream_bytes);
        }

        // HC Expand after FFN -> stage_hc for next stage
        if (st->hc_ffn_fn) {
            (void)myds_metal_v41_hc_expand(ws->stage_hc, ws->ffn_total, ws->after_attn_hc, ws->hc_split, MYDS_DSPARK_DIM);
        } else {
            for (size_t c = 0; c < 4; c++) {
                (void)myds_metal_buffer_copy(ws->stage_hc, c * stream_bytes, ws->ffn_total, 0, stream_bytes);
            }
        }
    }

    // 3. Final Head (mtp.2.norm + Base Model LM Head)
    myds_dspark_stage_t *final_st = &ds->stages[ds->stage_count - 1u];
    if (final_st->hc_ffn_fn) {
        (void)myds_metal_v41_hc_weighted_sum(ws->final_embd, ws->stage_hc, ws->hc_split, MYDS_DSPARK_DIM);
    } else {
        (void)myds_metal_buffer_copy(ws->final_embd, 0, ws->stage_hc, 0, stream_bytes);
    }

    if (ds->final_norm) {
        (void)myds_metal_v41_rmsnorm(ws->final_norm_out, ws->final_embd, ds->final_norm, MYDS_DSPARK_DIM, 1e-6f, true);
    } else {
        (void)myds_metal_buffer_copy(ws->final_norm_out, 0, ws->final_embd, 0, stream_bytes);
    }

    // Project to LM Head logits
    (void)myds_metal_gemv(ds->draft_logits_buf, ctx->model->output, ws->final_norm_out,
                          ctx->model->vocab_size, MYDS_DSPARK_DIM, ctx->model->output_type);
    (void)myds_metal_commit_batch();

    // 4. Confidence Head Gating:
    // confidence_proj is [5376, 1] (Q8_0)
    // Features: [5120 from final_embd, 256 from markov_state_host]
    float conf_prob = 1.0f;
    if (ds->confidence_proj && ds->confidence_proj_type == GGML_TYPE_Q8_0) {
        const q8_0_block_t *w_conf = (const q8_0_block_t *)myds_metal_buffer_data(ds->confidence_proj);
        const float *final_embd_data = (const float *)myds_metal_buffer_data(ws->final_embd);
        if (w_conf && final_embd_data) {
            float conf_score = dot_q8_row(w_conf, final_embd_data, 160);
            if (ds->markov_state_host) {
                conf_score += dot_q8_row(w_conf + 160, ds->markov_state_host, 8);
            }
            conf_prob = 1.0f / (1.0f + expf(-conf_score));
        }
    }
    if (out_confidence) *out_confidence = conf_prob;

    return 1;
}

uint32_t myds_dspark_propose(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token, uint32_t draft_cap) {
    if (!ds || !ctx || !ctx->model) return 0;
    if (ds->backoff_counter > 0) {
        ds->backoff_counter--;
        return 0;
    }

    double t0 = now_ms();
    ds->cycles++;

    uint32_t K = draft_cap > 0 && draft_cap < ds->block_size ? draft_cap : ds->block_size;
    if (K > MYDS_DSPARK_DEFAULT_BLOCK_SIZE) K = MYDS_DSPARK_DEFAULT_BLOCK_SIZE;

    static int markov_only = -1;
    if (__builtin_expect(markov_only < 0, 0)) {
        const char *env = getenv("MYDS_DSPARK_MARKOV_ONLY");
        markov_only = (env && atoi(env) != 0) ? 1 : 0;
    }

    static float markov_scale = -1.0f;
    if (__builtin_expect(markov_scale < 0.0f, 0)) {
        const char *env = getenv("MYDS_DSPARK_MARKOV_SCALE");
        markov_scale = env ? (float)atof(env) : 0.20f;
        MYDS_LOG_I("DSpark Markov bias Bayesian scale: %.2f", markov_scale);
    }

    float confidence = 1.0f;
    const float *base_logits = NULL;

    if (!markov_only && ds->main_proj && ds->target_hidden_buf) {
        if (myds_dspark_eval_stages(ds, ctx, &confidence)) {
            base_logits = (const float *)myds_metal_buffer_data(ds->draft_logits_buf);
        }
    }

    static float min_conf = -1.0f;
    if (__builtin_expect(min_conf < -0.5f, 0)) {
        const char *env = getenv("MYDS_DSPARK_MIN_CONF");
        min_conf = env ? (float)atof(env) : 0.80f; // 0.80 threshold
        MYDS_LOG_I("DSpark confidence head threshold: %.2f", min_conf);
    }

    // Confidence Gating: if below threshold, return 0 (0 overhead fallback to 29.8 tok/s)
    if (!markov_only && min_conf > 0.0f && confidence < min_conf) {
        ds->draft_count = 0;
        ds->draft_tree_count = 0;
        ds->is_high_confidence = false;
        ds->propose_ms += (now_ms() - t0);
        return 0;
    }
    ds->is_high_confidence = (confidence >= 0.80f);
    int32_t prev = last_token;
    uint32_t n_proposed = 0;
    const uint32_t vocab_size = ctx->model->vocab_size;

    const bool has_markov = (ds->markov_w1 && ds->markov_w2 &&
                             ds->markov_w1_type == GGML_TYPE_Q8_0 &&
                             ds->markov_w2_type == GGML_TYPE_Q8_0 &&
                             ds->markov_state_host && ds->markov_bias_host);

    if (!has_markov && !base_logits) return 0;

    const q8_0_block_t *w1_base = has_markov ? (const q8_0_block_t *)myds_metal_buffer_data(ds->markov_w1) : NULL;
    const q8_0_block_t *w2_base = has_markov ? (const q8_0_block_t *)myds_metal_buffer_data(ds->markov_w2) : NULL;

    for (uint32_t d = 0; d < K; d++) {
        int32_t best_token = 0;
        int32_t best_token2 = 0;
        float best_score = -INFINITY;
        float best_score2 = -INFINITY;

        if (has_markov && w1_base && w2_base && prev >= 0 && (uint32_t)prev < vocab_size) {
            // Dequantize 8 blocks from markov_w1 for row `prev`
            const q8_0_block_t *w1_row = w1_base + (size_t)prev * 8;
            for (int b = 0; b < 8; b++) {
                float d_scale = myds_fp16_to_fp32(w1_row[b].d);
                for (int k = 0; k < 32; k++) {
                    ds->markov_state_host[b * 32 + k] = d_scale * (float)w1_row[b].qs[k];
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

                for (uint32_t r = r0; r < r1; r++) {
                    const q8_0_block_t *w2_row = w2_base + (size_t)r * 8;
                    float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                    float score = base_logits ? (base_logits[r] + markov_scale * bias) : bias;
                    dspark_update_top2((int32_t)r, score, &local_t1, &local_s1, &local_t2, &local_s2);
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
            for (uint32_t r = 0; r < vocab_size; r++) {
                const q8_0_block_t *w2_row = w2_base + (size_t)r * 8;
                float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                float score = base_logits ? (base_logits[r] + markov_scale * bias) : bias;
                dspark_update_top2((int32_t)r, score, &best_token, &best_score, &best_token2, &best_score2);
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

            static float min_margin = -1.0f;
            if (__builtin_expect(min_margin < -0.5f, 0)) {
                const char *env = getenv("MYDS_DSPARK_MIN_MARGIN");
                min_margin = env ? (float)atof(env) : 0.30f;
                MYDS_LOG_I("DSpark logit margin gating active: threshold=%.2f", min_margin);
            }

            if (min_margin > 0.0f && (best_score - best_score2) < min_margin) {
                // Top-1 vs Top-2 margin too narrow: skip speculative proposal
                ds->draft_count = 0;
                ds->draft_tree_count = 0;
                ds->propose_ms += (now_ms() - t0);
                return 0;
            }

            if ((best_score - best_score2) >= 2.00f) {
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

