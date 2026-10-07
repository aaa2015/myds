#ifndef MYDS_DSPARK_H
#define MYDS_DSPARK_H

#include "myds_internal.h"
#include <stdint.h>
#include <stdbool.h>

#define MYDS_DSPARK_MAX_STAGES 4
#define MYDS_DSPARK_TARGET_LAYERS 3
#define MYDS_DSPARK_DEFAULT_BLOCK_SIZE 5
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
} myds_dspark_stage_t;

struct myds_dspark {
    uint32_t stage_count;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layers[MYDS_DSPARK_TARGET_LAYERS]; // {37, 38, 39}
    uint32_t n_routed_experts;    // 128
    uint32_t num_experts_per_tok; // 3

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
    ggml_type markov_w1_type, markov_w2_type, confidence_proj_type;

    // Context / Scratch Buffers for inference
    myds_metal_buffer_t *target_hidden_buf; // 3 * 5120 * sizeof(float)
    myds_metal_buffer_t *target_slots[MYDS_DSPARK_TARGET_LAYERS]; // Views into target_hidden_buf
    myds_metal_buffer_t *hc_mean_weights;   // 4 * sizeof(float) {0.25f, 0.25f, 0.25f, 0.25f}
    myds_metal_buffer_t *stage_x_buf;       // 5120 * sizeof(float)
    myds_metal_buffer_t *draft_logits_buf;  // vocab_size * sizeof(float)
    float *markov_state_host;               // markov_rank * sizeof(float)
    float *markov_bias_host;                // vocab_size * sizeof(float)

    // Draft proposal storage
    int32_t draft_tokens[MYDS_DSPARK_DEFAULT_BLOCK_SIZE];
    uint32_t draft_count;

    // Speculative statistics
    uint64_t cycles;
    uint64_t proposed_tokens;
    uint64_t accepted_tokens;
    double propose_ms;
    double verify_ms;
};

/* Initialize and load DSpark support model from GGUF */
myds_dspark_t *myds_dspark_load(const char *path, uint32_t vocab_size);

/* Free DSpark support model resources */
void myds_dspark_free(myds_dspark_t *ds);

/* Capture target layer hidden state (layer 37, 38, or 39) from residual HC streams */
void myds_dspark_capture_layer(myds_dspark_t *ds, uint32_t layer_idx, const myds_metal_buffer_t *residual);

/* Propose K draft tokens given last token and captured target hidden states */
uint32_t myds_dspark_propose(myds_dspark_t *ds, myds_context_t *ctx, int32_t last_token, uint32_t draft_cap);

#endif /* MYDS_DSPARK_H */
