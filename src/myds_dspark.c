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
        BIND_BUF(hc_attn_fn, "hc_attn_fn.weight");
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
        BIND_BUF(hc_ffn_fn, "hc_ffn_fn.weight");
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

    MYDS_LOG_I("dspark: loaded successfully from %s (stages=%u, block=%u, markov_rank=%u, routed_exp=%u/%u)",
               path, ds->stage_count, ds->block_size, ds->markov_rank,
               ds->num_experts_per_tok, ds->n_routed_experts);
    return ds;
}

void myds_dspark_free(myds_dspark_t *ds) {
    if (!ds) return;
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

static inline float dot_q8_row256(const q8_0_block_t *row, const float *state) {
    float total = 0.0f;
#if defined(__ARM_NEON)
    for (int b = 0; b < 8; b++) {
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
    for (int b = 0; b < 8; b++) {
        float d = myds_fp16_to_fp32(row[b].d);
        const float *s = state + b * 32;
        float bsum = 0.0f;
        for (int k = 0; k < 32; k++) bsum += (float)row[b].qs[k] * s[k];
        total += d * bsum;
    }
#endif
    return total;
}

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#define MARKOV_NUM_SLICES 16
#else
#define MARKOV_NUM_SLICES 1
#endif

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

    const float *base_logits = NULL;
    if (!markov_only && ds->main_proj && ds->target_hidden_buf && ds->stage_x_buf && ds->draft_logits_buf) {
        // 1. Stage0 Entry Projection: [15360] -> [5120]
        const uint32_t in_dim = MYDS_DSPARK_TARGET_LAYERS * MYDS_DSPARK_DIM;
        if (myds_metal_gemv(ds->stage_x_buf, ds->main_proj, ds->target_hidden_buf,
                            MYDS_DSPARK_DIM, in_dim, ds->main_proj_type)) {
            if (ds->main_norm) {
                (void)myds_metal_v41_rmsnorm(ds->stage_x_buf, ds->stage_x_buf, ds->main_norm,
                                            MYDS_DSPARK_DIM, 1e-6f, true);
            }
            if (ds->final_norm) {
                (void)myds_metal_v41_rmsnorm(ds->stage_x_buf, ds->stage_x_buf, ds->final_norm,
                                            MYDS_DSPARK_DIM, 1e-6f, true);
            }

            // 2. Compute Base Logits using Main Model LM Head
            if (myds_metal_gemv(ds->draft_logits_buf, ctx->model->output, ds->stage_x_buf,
                                ctx->model->vocab_size, MYDS_DSPARK_DIM, ctx->model->output_type)) {
                // Synchronize to read logits
                (void)myds_metal_commit_batch();
                base_logits = (const float *)myds_metal_buffer_data(ds->draft_logits_buf);
            }
        }
    }

    // 3. Draft generation with Markov Head
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
            int32_t *tokens_out = slice_tokens;
            float *scores_out = slice_scores;
            uint32_t chunk = (vocab_size + MARKOV_NUM_SLICES - 1) / MARKOV_NUM_SLICES;

            dispatch_apply(MARKOV_NUM_SLICES, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t s) {
                uint32_t r0 = (uint32_t)s * chunk;
                uint32_t r1 = r0 + chunk;
                if (r1 > vocab_size) r1 = vocab_size;

                int32_t local_best_token = 0;
                float local_best_score = -INFINITY;

                for (uint32_t r = r0; r < r1; r++) {
                    const q8_0_block_t *w2_row = w2_base + (size_t)r * 8;
                    float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                    float score = base_logits ? (base_logits[r] + bias) : bias;
                    if (score > local_best_score) {
                        local_best_score = score;
                        local_best_token = (int32_t)r;
                    }
                }
                tokens_out[s] = local_best_token;
                scores_out[s] = local_best_score;
            });

            best_token = slice_tokens[0];
            float best_score = slice_scores[0];
            for (int s = 1; s < MARKOV_NUM_SLICES; s++) {
                if (slice_scores[s] > best_score) {
                    best_score = slice_scores[s];
                    best_token = slice_tokens[s];
                }
            }
#else
            float best_score = -INFINITY;
            for (uint32_t r = 0; r < vocab_size; r++) {
                const q8_0_block_t *w2_row = w2_base + (size_t)r * 8;
                float bias = dot_q8_row256(w2_row, ds->markov_state_host);
                float score = base_logits ? (base_logits[r] + bias) : bias;
                if (score > best_score) {
                    best_score = score;
                    best_token = (int32_t)r;
                }
            }
#endif
        } else if (base_logits) {
            // Simple argmax fallback
            float best_score = -INFINITY;
            for (uint32_t r = 0; r < vocab_size; r++) {
                if (base_logits[r] > best_score) {
                    best_score = base_logits[r];
                    best_token = (int32_t)r;
                }
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

