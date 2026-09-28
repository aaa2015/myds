#ifndef MYDS_INTERNAL_H
#define MYDS_INTERNAL_H

#include "myds.h"
#include "myds_gguf.h"
#include "myds_tokenizer.h"
#include "myds_metal.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    myds_metal_buffer_t *attn_norm;
    myds_metal_buffer_t *attn_q;
    myds_metal_buffer_t *attn_k;
    myds_metal_buffer_t *attn_v;
    myds_metal_buffer_t *attn_o;

    /* Multi-Head Latent Attention (MLA) */
    myds_metal_buffer_t *attn_q_a;
    myds_metal_buffer_t *attn_q_a_norm;
    myds_metal_buffer_t *attn_q_b;
    myds_metal_buffer_t *attn_kv;
    myds_metal_buffer_t *attn_kv_a_norm;
    myds_metal_buffer_t *attn_sinks;
    myds_metal_buffer_t *attn_output_a;
    myds_metal_buffer_t *attn_output_b;

    myds_metal_buffer_t *ffn_norm;
    myds_metal_buffer_t *ffn_gate;
    myds_metal_buffer_t *ffn_up;
    myds_metal_buffer_t *ffn_down;

    /* DeepSeek MoE router */
    myds_metal_buffer_t *router_gate;
    myds_metal_buffer_t *shared_gate;
    myds_metal_buffer_t *shared_up;
    myds_metal_buffer_t *shared_down;

    /* Weight types */
    ggml_type attn_q_type, attn_k_type, attn_v_type, attn_o_type;
    ggml_type attn_q_a_type, attn_q_b_type, attn_kv_type;
    ggml_type attn_output_a_type, attn_output_b_type;
    ggml_type ffn_gate_type, ffn_up_type, ffn_down_type;
} myds_layer_t;

struct myds_model {
    gguf_file_t         *gf;
    myds_tokenizer_t    *tokenizer;

    char                 arch[64];
    uint32_t             dim;
    uint32_t             n_layers;
    uint32_t             n_heads;
    uint32_t             n_kv_heads;
    uint32_t             head_dim;
    uint32_t             vocab_size;
    uint32_t             hidden_dim;

    /* DeepSeek MLA parameters */
    bool                 is_mla;
    uint32_t             q_lora_rank;        /* default: 1280 */
    uint32_t             kv_lora_rank;       /* default: 512 */
    uint32_t             qk_rope_head_dim;   /* default: 64 */
    uint32_t             n_out_groups;       /* default: 8 */
    uint32_t             out_group_rank;     /* default: 1024 */

    /* DeepSeek MoE parameters */
    uint32_t             n_routed_experts;
    uint32_t             n_activated_experts;
    uint32_t             n_shared_experts;
    float                routed_scaling_factor;

    float                norm_eps;
    float                freq_base;
    float                freq_scale;

    myds_metal_buffer_t *tok_embeddings;
    ggml_type            tok_embeddings_type;

    myds_layer_t        *layers;

    myds_metal_buffer_t *output_norm;
    myds_metal_buffer_t *output;
    ggml_type            output_type;
};

struct myds_context {
    myds_model_t        *model;
    myds_config_t        cfg;
    uint32_t             pos;

    /* Metal activation buffers */
    myds_metal_buffer_t *x;
    myds_metal_buffer_t *x_norm;
    myds_metal_buffer_t *q;
    myds_metal_buffer_t *k;
    myds_metal_buffer_t *v;
    myds_metal_buffer_t *attn_scores;
    myds_metal_buffer_t *attn_out;
    myds_metal_buffer_t *ffn_norm_out;
    myds_metal_buffer_t *gate_out;
    myds_metal_buffer_t *up_out;
    myds_metal_buffer_t *swiglu_out;
    myds_metal_buffer_t *ffn_out;
    myds_metal_buffer_t *logits;

    /* MLA scratch activation buffers */
    myds_metal_buffer_t *qr;         /* [q_lora_rank] */
    myds_metal_buffer_t *qr_norm;    /* [q_lora_rank] */
    myds_metal_buffer_t *kv_act;     /* [head_dim] */
    myds_metal_buffer_t *attn_heads; /* [n_heads * head_dim] */
    myds_metal_buffer_t *attn_low;   /* [n_out_groups * out_group_rank] */

    /* DeepSeek KV Cache */
    float              **k_cache; /* [n_layers][max_seq_len * stride] */
    float              **v_cache; /* [n_layers][max_seq_len * stride] (MHA only) */

    /* Sampler persistent scratch buffers (zero malloc churn per token) */
    float               *sample_probs;
    int32_t             *sample_indices;
    uint64_t             rng_state;
};

/* Unified production sampler: greedy, temperature, top-k, min-p, top-p, repetition penalty */
int32_t myds_sample_next_token(
    const float         *logits,
    uint32_t             vocab_size,
    const myds_config_t *cfg,
    myds_context_t      *ctx,
    const int32_t       *prev_tokens,
    size_t               n_prev_tokens);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_INTERNAL_H */
