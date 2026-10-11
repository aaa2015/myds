#ifndef MYDS_DSPARK_H
#define MYDS_DSPARK_H

#include "myds_internal.h"
#include "myds_gguf.h"
#include <stdint.h>
#include <stdbool.h>

#define MYDS_DSPARK_MAX_STAGES 4
#define MYDS_DSPARK_TARGET_LAYERS 3
#define MYDS_DSPARK_DEFAULT_BLOCK_SIZE 5
#define MYDS_DSPARK_TOTAL_ROWS 6
#define MYDS_DSPARK_MARKOV_RANK 256
#define MYDS_DSPARK_DIM 5120
#define MYDS_DSPARK_EXP_DIM 2304

typedef struct {
    // Attention weights
    myds_metal_buffer_t *attn_norm;
    myds_metal_buffer_t *attn_q_a;
    myds_metal_buffer_t *attn_q_a_norm;
    myds_metal_buffer_t *attn_q_b;
    myds_metal_buffer_t *attn_kv;
    myds_metal_buffer_t *attn_kv_a_norm;
    myds_metal_buffer_t *attn_sinks;
    myds_metal_buffer_t *attn_output_a;
    myds_metal_buffer_t *attn_output_b;
    ggml_type attn_q_a_type, attn_q_b_type, attn_kv_type, attn_output_a_type, attn_output_b_type;

    // HC Attention weights
    myds_metal_buffer_t *hc_attn_base;
    myds_metal_buffer_t *hc_attn_fn;
    myds_metal_buffer_t *hc_attn_scale;
    ggml_type hc_attn_fn_type;

    // FFN / MoE weights
    myds_metal_buffer_t *ffn_norm;
    myds_metal_buffer_t *router_gate; // ffn_gate_inp
    myds_metal_buffer_t *router_bias; // exp_probs_b.bias
    ggml_type router_gate_type;

    // Shared expert
    myds_metal_buffer_t *shared_gate; // ffn_gate_shexp
    myds_metal_buffer_t *shared_up;   // ffn_up_shexp
    myds_metal_buffer_t *shared_down; // ffn_down_shexp
    ggml_type shared_gate_type, shared_up_type, shared_down_type;

    // Routed experts (128 experts, top 3)
    myds_metal_buffer_t *ffn_gate_exps;
    myds_metal_buffer_t *ffn_up_exps;
    myds_metal_buffer_t *ffn_down_exps;
    ggml_type ffn_gate_exps_type, ffn_up_exps_type, ffn_down_exps_type;

    // HC FFN weights
    myds_metal_buffer_t *hc_ffn_base;
    myds_metal_buffer_t *hc_ffn_fn;
    myds_metal_buffer_t *hc_ffn_scale;
    ggml_type hc_ffn_fn_type;
} myds_dspark_stage_t;

struct myds_v41_gpu_scratch;

typedef struct {
    // Multi-row Stage residual streams (Row 0: Target, Rows 1..5: Draft)
    myds_metal_buffer_t *stage_target_hc;       // 4 * 5120 * sizeof(float) (Row 0, constant across stages)
    myds_metal_buffer_t *stage_draft_hc;        // 5 * 4 * 5120 * sizeof(float) (Rows 1..5, updated across stages)
    myds_metal_buffer_t *stage_hc;              // alias to stage_draft_hc
    myds_metal_buffer_t *stage_input_hc;        // 6 * 4 * 5120 * sizeof(float) (all 6 rows concatenated)
    myds_metal_buffer_t *after_attn_hc;         // 5 * 4 * 5120 * sizeof(float) (5 draft rows after attn)
    myds_metal_buffer_t *stage_flat_hc_norm;    // 6 * 4 * 5120 * sizeof(float) (6 rows flat normalized)
    myds_metal_buffer_t *target_flat_hc_norm;   // subview: row 0
    myds_metal_buffer_t *draft_flat_hc_norm;    // 5 * 4 * 5120 * sizeof(float) (5 draft rows for FFN)
    myds_metal_buffer_t *flat_hc_norm;          // alias to stage_flat_hc_norm
    myds_metal_buffer_t *hc_mix;                // 6 * 24 * sizeof(float)
    myds_metal_buffer_t *target_hc_mix;         // subview: row 0
    myds_metal_buffer_t *draft_hc_mix;          // subview: rows 1..5
    myds_metal_buffer_t *hc_split;              // 6 * 24 * sizeof(float)
    myds_metal_buffer_t *target_hc_split;       // subview: row 0
    myds_metal_buffer_t *draft_hc_split;        // subview: rows 1..5
    myds_metal_buffer_t *draft_ffn_hc_mix;      // 5 * 24 * sizeof(float) (5 draft rows for FFN)
    myds_metal_buffer_t *draft_ffn_hc_split;    // 5 * 24 * sizeof(float) (5 draft rows for FFN)

    // Attention intermediate buffers
    myds_metal_buffer_t *attn_cur;              // 6 * 5120 * sizeof(float)
    myds_metal_buffer_t *target_attn_cur;       // subview: row 0
    myds_metal_buffer_t *draft_attn_cur;        // subview: rows 1..5
    myds_metal_buffer_t *attn_norm;             // 6 * 5120 * sizeof(float)
    myds_metal_buffer_t *target_attn_norm;      // subview: row 0
    myds_metal_buffer_t *draft_attn_norm;       // subview: rows 1..5
    myds_metal_buffer_t *kv;                    // 6 * 512 * sizeof(float)
    myds_metal_buffer_t *kv_norm;               // 6 * 512 * sizeof(float)
    myds_metal_buffer_t *target_kv;             // subview: row 0
    myds_metal_buffer_t *target_kv_norm;        // subview: row 0
    myds_metal_buffer_t *draft_kv;              // subview: rows 1..5
    myds_metal_buffer_t *draft_kv_norm;         // subview: rows 1..5
    myds_metal_buffer_t *qr;                    // 5 * 1280 * sizeof(float)
    myds_metal_buffer_t *qr_norm;               // 5 * 1280 * sizeof(float)
    myds_metal_buffer_t *q;                     // 5 * 64 * 512 * sizeof(float)
    myds_metal_buffer_t *stage_kv_cache[MYDS_DSPARK_MAX_STAGES]; // 128 * 512 * sizeof(float)
    myds_metal_buffer_t *attn_heads;            // 5 * 64 * 512 * sizeof(float)
    myds_metal_buffer_t *attn_low;              // 5 * 8192 * sizeof(float)
    myds_metal_buffer_t *attn_out;              // 5 * 5120 * sizeof(float)

    // FFN / MoE intermediate buffers (5 draft rows)
    myds_metal_buffer_t *ffn_cur;               // 5 * 5120 * sizeof(float)
    myds_metal_buffer_t *ffn_norm;              // 5 * 5120 * sizeof(float)
    myds_metal_buffer_t *shared_gate;           // 5 * 2304 * sizeof(float)
    myds_metal_buffer_t *shared_up;             // 5 * 2304 * sizeof(float)
    myds_metal_buffer_t *shared_swiglu;         // 5 * 2304 * sizeof(float)
    myds_metal_buffer_t *shared_out;            // 5 * 5120 * sizeof(float)

    myds_metal_buffer_t *router_logits;         // 5 * 128 * sizeof(float)
    myds_metal_buffer_t *selected_ids;          // 5 * 6 * sizeof(int32_t)
    myds_metal_buffer_t *selected_weights;      // 5 * 6 * sizeof(float)
    myds_metal_buffer_t *moe_gate;              // 5 * 6 * 2304 * sizeof(float)
    myds_metal_buffer_t *moe_up;                // 5 * 6 * 2304 * sizeof(float)
    myds_metal_buffer_t *moe_mid;               // 5 * 6 * 2304 * sizeof(float)
    myds_metal_buffer_t *moe_down;              // 5 * 6 * 5120 * sizeof(float)
    myds_metal_buffer_t *moe_out;               // 5 * 5120 * sizeof(float)
    myds_metal_buffer_t *ffn_total;             // 5 * 5120 * sizeof(float)

    // Final stage output (5 draft rows)
    myds_metal_buffer_t *final_embd;            // 5 * 5120 * sizeof(float)
    myds_metal_buffer_t *final_norm_out;        // 5 * 5120 * sizeof(float)

    // Pre-allocated subviews for 6 rows (row 0 = target, rows 1..5 = draft)
    myds_metal_buffer_t *v_stage_input_hc[MYDS_DSPARK_TOTAL_ROWS];
    myds_metal_buffer_t *v_stage_flat_hc_norm[MYDS_DSPARK_TOTAL_ROWS];
    myds_metal_buffer_t *v_hc_mix[MYDS_DSPARK_TOTAL_ROWS];
    myds_metal_buffer_t *v_hc_split[MYDS_DSPARK_TOTAL_ROWS];
    myds_metal_buffer_t *v_attn_cur[MYDS_DSPARK_TOTAL_ROWS];
    myds_metal_buffer_t *v_attn_norm[MYDS_DSPARK_TOTAL_ROWS];
    myds_metal_buffer_t *v_kv[MYDS_DSPARK_TOTAL_ROWS];
    myds_metal_buffer_t *v_kv_norm[MYDS_DSPARK_TOTAL_ROWS];

    // Pre-allocated subviews for 5 draft rows
    myds_metal_buffer_t *v_stage_draft_hc[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_after_attn_hc[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_draft_flat_hc_norm[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_draft_ffn_hc_mix[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_draft_ffn_hc_split[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_qr[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_qr_norm[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_q[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_attn_heads[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_attn_low[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_attn_out[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_ffn_cur[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_ffn_norm[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_shared_gate[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_shared_up[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_shared_swiglu[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_shared_out[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_router_logits[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_selected_ids[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_selected_weights[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_moe_gate[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_moe_up[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_moe_mid[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_moe_down[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_moe_out[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_ffn_total[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_final_embd[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_final_norm_out[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    myds_metal_buffer_t *v_logits[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];

    // Scratch for GPU attention
    struct myds_v41_gpu_scratch *gpu_scratch;

    float *conf_features_host;                 // 5376 * sizeof(float)
} myds_dspark_workspace_t;

typedef struct {
    uint64_t requests, encoded, bypassed, failures, stale_rejections;
} myds_dspark_capture_counters_t;

struct myds_dspark {
    gguf_file_t *gf;
    uint32_t stage_count;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layers[MYDS_DSPARK_TARGET_LAYERS]; // {37, 38, 39}
    uint32_t n_routed_experts;    // 128
    uint32_t num_experts_per_tok; // 3
    myds_dspark_workspace_t *ws;

    // Stage 0 entry projection
    myds_metal_buffer_t *main_proj; // [15360, 5120]
    myds_metal_buffer_t *main_norm; // [5120]
    ggml_type main_proj_type;

    // Transformer Stages (0, 1, 2)
    myds_dspark_stage_t stages[MYDS_DSPARK_MAX_STAGES];

    // Final stage head
    myds_metal_buffer_t *final_norm;       // mtp.2.norm.weight
    myds_metal_buffer_t *markov_w1;        // mtp.2.markov_head.markov_w1.weight (embed)
    myds_metal_buffer_t *markov_w2;        // mtp.2.markov_head.markov_w2.weight (head)
    myds_metal_buffer_t *confidence_proj;  // mtp.2.confidence_head.proj.weight
    float confidence_bias;
    ggml_type markov_w1_type, markov_w2_type, confidence_proj_type;

    // Context / Scratch Buffers for inference
    myds_metal_buffer_t *target_hidden_buf; // 3 * 5120 * sizeof(float)
    myds_metal_buffer_t *target_slots[MYDS_DSPARK_TARGET_LAYERS]; // Views into target_hidden_buf
    myds_metal_buffer_t *hc_mean_weights;   // 4 * sizeof(float) {0.25f, 0.25f, 0.25f, 0.25f}
    /* Immutable construction mode. Request policy is set before prefill;
     * unmanaged/direct scalar and Batch2 forwards remain capture-active.
     * Engine mode excludes the experimental tree API, not those direct APIs. */
    bool capture_configured, capture_bypass_mode, capture_disabled, capture_committed;
    bool capture_engine_only;
    uint32_t capture_valid_mask;
    myds_dspark_capture_counters_t capture_counters;
    myds_metal_buffer_t *stage_x_buf;       // 5120 * sizeof(float)
    myds_metal_buffer_t *noise_hc;          // 4 * 5120 * sizeof(float)
    bool noise_hc_initialized;
    myds_metal_buffer_t *draft_logits_buf;  // 5 * vocab_size * sizeof(float)
    float *markov_state_host;               // markov_rank * sizeof(float)
    float *markov_bias_host;                // vocab_size * sizeof(float)

    // Draft proposal storage
    int32_t current_seed_token;
    int32_t draft_tokens[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    uint32_t draft_count;

    // Draft tree proposals (Top-2 candidates for Tree-based Speculative Decoding)
    int32_t draft_tree_tokens[2];
    float   draft_tree_scores[2];
    uint32_t draft_tree_count;

    // Speculative statistics
    uint64_t cycles;
    uint64_t proposed_tokens;
    uint64_t accepted_tokens;
    double propose_ms;
    double verify_ms;

    // Adaptive Speculative Scheduler
    uint32_t consecutive_misses;
    uint32_t backoff_counter;
    bool     is_high_confidence;

    // Speculative snapshot buffer (heap-allocated, zero stack footprint)
    struct myds_v41_spec_snapshot *spec_snap;
};

static inline int myds_dspark_target_slot(const myds_dspark_t *ds, uint32_t layer_idx) {
    if (!ds) return -1;
    for (int s = 0; s < MYDS_DSPARK_TARGET_LAYERS; s++) {
        if (ds->target_layers[s] == layer_idx) return s;
    }
    return -1;
}

/* Initialize and load DSpark support model from GGUF */
myds_dspark_t *myds_dspark_load(const char *path, uint32_t vocab_size);

/* Free DSpark support model resources */
void myds_dspark_free(myds_dspark_t *ds);

/* Capture target layer hidden state (layer 37, 38, or 39) from residual HC streams */
void myds_dspark_capture_layer(myds_dspark_t *ds, uint32_t layer_idx, const myds_metal_buffer_t *residual);

/* Shared early parser: strict absent/0/1 flags, with explicit old/new 1
 * mutually exclusive. Returns -1 on error, 0 reference, 1 old broad bypass,
 * or 2 Engine-call-chain bypass (experimental tree API excluded). */
int myds_dspark_capture_parse_mode(void);
/* Repeating configure never rereads the environment on a live context. */
bool myds_dspark_capture_configure(myds_dspark_t *ds);
/* Effective mechanism getter: true for either bypass mode. */
int myds_dspark_capture_bypass_enabled(const myds_dspark_t *ds);
int myds_dspark_engine_capture_bypass_enabled(const myds_dspark_t *ds);
bool myds_dspark_capture_required(const myds_dspark_t *ds);
bool myds_dspark_capture_ready(const myds_dspark_t *ds);
void myds_dspark_capture_counters(myds_dspark_t *ds, myds_dspark_capture_counters_t *out, bool reset);
/* Speculative scope includes serial probing. This setter also excludes the
 * actual Markov-only mode and missing neural entry projection. */
void myds_dspark_capture_begin_request(myds_dspark_t *ds, bool speculative_scope);
void myds_dspark_capture_end_request(myds_dspark_t *ds, bool success);
void myds_dspark_capture_invalidate(myds_dspark_t *ds);
/* Only call after a successful target/restore commit which has written all
 * three slots. Whole rollback instead invalidates; a target retry recaptures. */
void myds_dspark_capture_publish(myds_dspark_t *ds);
/* NULL output selects the real target slot; a non-NULL output is a verifier
 * workspace slot. Candidate failures propagate; mode0 preserves the legacy
 * unchecked capture behavior. A bypass succeeds without reading/writing GPU. */
int myds_dspark_capture_layer_to(myds_dspark_t *ds, uint32_t layer_idx,
    const myds_metal_buffer_t *residual, myds_metal_buffer_t *output);

/* Propose K draft tokens given last token and captured target hidden states */
uint32_t myds_dspark_propose(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token, uint32_t draft_cap);

/* Evaluate 3-Stage Transformer draft model to compute draft logits and confidence score */
int myds_dspark_eval_stages(myds_dspark_t *ds, myds_context_t *ctx, float min_conf, float *out_confidence);

/* Propose Top-2 tree branches for tree-based speculative decoding */
uint32_t myds_dspark_propose_tree(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token);

#endif /* MYDS_DSPARK_H */
