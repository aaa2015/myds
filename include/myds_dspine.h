#ifndef MYDS_DSPINE_H
#define MYDS_DSPINE_H

#include "myds_internal.h"
#include "myds_v41.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYDS_DSPINE_LAYERS 3
#define MYDS_DSPINE_MAX_BLOCK 16
#define MYDS_DSPINE_DEFAULT_BLOCK 8
#define MYDS_DSPINE_TARGET_LAYERS 5
#define MYDS_DSPINE_TARGET_DIM (MYDS_DSPINE_TARGET_LAYERS * 5120) /* 25600 */
#define MYDS_DSPINE_TRANSFER_RANK 512
#define MYDS_DSPINE_MESSAGE_SIZE 256
#define MYDS_DSPINE_TOP_K 16
#define MYDS_DSPINE_CTX_BATCH_CAP 64

/* Layer weights for the dense DSpine draft Transformer */
typedef struct {
    myds_metal_buffer_t *input_norm;
    myds_metal_buffer_t *q_proj;
    myds_metal_buffer_t *k_proj;
    myds_metal_buffer_t *v_proj;
    myds_metal_buffer_t *o_proj;
    myds_metal_buffer_t *q_norm;
    myds_metal_buffer_t *k_norm;

    myds_metal_buffer_t *post_attn_norm;
    myds_metal_buffer_t *gate_proj;
    myds_metal_buffer_t *up_proj;
    myds_metal_buffer_t *down_proj;
} myds_dspine_layer_t;

/* Complete DSpine runtime state */
typedef struct myds_dspine_s {
    uint32_t hidden_size;        /* 5120 */
    uint32_t intermediate_size;  /* 15360 */
    uint32_t num_layers;         /* 3 */
    uint32_t num_heads;          /* 40 */
    uint32_t num_kv_heads;       /* 8 */
    uint32_t head_dim;           /* 128 */
    uint32_t vocab_size;         /* 129280 */
    uint32_t block_size;         /* default 8 or 16 */
    uint32_t top_k;              /* 16 */
    int32_t  mask_token_id;      /* 129279 */

    /* Mmap handle for zero-copy safetensors weights */
    int      mmap_fd;
    void    *mmap_addr;
    size_t   mmap_size;

    /* Model Weights */
    myds_metal_buffer_t *fc_proj;       /* [5120, 25600] */
    myds_metal_buffer_t *hidden_norm;   /* [5120] */
    myds_dspine_layer_t  layers[MYDS_DSPINE_LAYERS];
    myds_metal_buffer_t *final_norm;    /* [5120] */

    /* Code Bridge (Adjacent Causal Injection) */
    myds_metal_buffer_t *injection_readouts[MYDS_DSPINE_LAYERS]; /* [512, 5120] */
    myds_metal_buffer_t *injection_message;                      /* [256, 512] */
    myds_metal_buffer_t *injection_writes[MYDS_DSPINE_LAYERS];   /* [5120, 256] */
    myds_metal_buffer_t *injection_pred_gate;                    /* [256, 512] */
    myds_metal_buffer_t *injection_recv_gate;                    /* [256, 512] */
    myds_metal_buffer_t *injection_bias;                         /* [256] */

    /* DSpine Transfer Space (Refinement Tail) */
    myds_metal_buffer_t *transfer_codes; /* [129280, 512] */

    /* Precomputed Serial Tables for Refinement Tail */
    myds_metal_buffer_t *tg_table;       /* [129280, 256] */
    myds_metal_buffer_t *tsv_table;      /* [129280, 512] */
    bool                 tables_ready;

    /* GPU Capture Views & Weights (dispatched on active Metal CB) */
    myds_metal_buffer_t *hc_mean_weights; /* [4] = {0.25, 0.25, 0.25, 0.25} */
    myds_metal_buffer_t *target_slot_views[MYDS_DSPINE_TARGET_LAYERS];
    myds_metal_buffer_t *spec_slot_views[3][MYDS_DSPINE_TARGET_LAYERS];

    /* Execution & Scratch Buffers */
    myds_metal_buffer_t *target_hidden_buf; /* [25600 * sizeof(float)] */
    myds_metal_buffer_t *ctx_feat_buf;      /* [5120 * sizeof(float)] */
    myds_metal_buffer_t *x_buf;             /* [max_block * 5120 * sizeof(float)] */
    myds_metal_buffer_t *norm_buf;          /* [max_block * 5120 * sizeof(float)] */
    myds_metal_buffer_t *attn_out_buf;      /* [max_block * 5120 * sizeof(float)] */
    myds_metal_buffer_t *ffn_out_buf;       /* [max_block * 5120 * sizeof(float)] */
    myds_metal_buffer_t *logits_buf;        /* [max_block * vocab_size * sizeof(float)] */
    myds_metal_buffer_t *draft_kv_buf;      /* [max_block * num_layers * 2 * num_kv_heads * head_dim * sizeof(float)] */

    /* Additional GPU scratch buffers */
    myds_metal_buffer_t *q_buf;        /* [max_block * 5120 * sizeof(float)] */
    myds_metal_buffer_t *k_buf;        /* [max_block * 1024 * sizeof(float)] */
    myds_metal_buffer_t *v_buf;        /* [max_block * 1024 * sizeof(float)] */
    myds_metal_buffer_t *k_ctx_buf;    /* [1024 * sizeof(float)] */
    myds_metal_buffer_t *v_ctx_buf;    /* [1024 * sizeof(float)] */
    myds_metal_buffer_t *attn_in_buf;  /* [max_block * 5120 * sizeof(float)] */
    myds_metal_buffer_t *gate_buf;     /* [max_block * intermediate_size * sizeof(float)] */
    myds_metal_buffer_t *up_buf;       /* [max_block * intermediate_size * sizeof(float)] */
    myds_metal_buffer_t *glu_buf;      /* [max_block * intermediate_size * sizeof(float)] */
    myds_metal_buffer_t *features_buf; /* [max_block * 512 * sizeof(float)] */
    myds_metal_buffer_t *delta_buf;    /* [max_block * 5120 * sizeof(float)] */

    /* Batched context KV projection buffers */
    myds_metal_buffer_t *batch_th_buf;    /* [CTX_BATCH_CAP * 25600 * sizeof(float)] */
    myds_metal_buffer_t *batch_ctx_buf;   /* [CTX_BATCH_CAP * 5120 * sizeof(float)] */
    myds_metal_buffer_t *batch_k_ctx_buf; /* [CTX_BATCH_CAP * 1024 * sizeof(float)] */
    myds_metal_buffer_t *batch_v_ctx_buf; /* [CTX_BATCH_CAP * 1024 * sizeof(float)] */

    /* Full Context KV Cache across all prior context tokens [0 .. anchor_pos - 1] */
    float   *ctx_k_cache[MYDS_DSPINE_LAYERS]; /* [cap * 1024] */
    float   *ctx_v_cache[MYDS_DSPINE_LAYERS]; /* [cap * 1024] */
    uint32_t *ctx_pos_cache;                  /* [cap] */
    float   *attn_scores_scratch;             /* [cap + max_block] */
    uint32_t ctx_kv_len;
    uint32_t ctx_kv_cap;

    /* Pending captured target slot(s) to flush into ctx_k_cache before propose */
    uint32_t pending_pos;
    uint32_t pending_count;
    bool     pending_from_spec;

    /* Precomputed RoPE inverse frequencies for head_dim=128 (theta=1e6) */
    float    rope_inv_freq[64];

    /* Layer 2 pre-injection states for discrete predecessor refinement tail */
    float    pre_injection_l2[MYDS_DSPINE_MAX_BLOCK * 5120];

    /* Output candidates */
    int32_t  draft_tokens[MYDS_DSPINE_MAX_BLOCK];
    uint32_t draft_count;

    /* Snapshot for verification rollback */
    myds_v41_spec_snapshot_t *spec_snap;
    myds_metal_buffer_t *spec_target_slots[3]; /* [25600 * sizeof(float)] */

    /* Performance counters */
    uint64_t proposed_tokens;
    uint64_t accepted_tokens;
    uint64_t cycles;
    double   propose_ms;
    double   verify_ms;

    bool     loaded;
    bool     enabled;
    uint32_t target_slots_captured; /* bitmask of captured target slots */
} myds_dspine_t;

/* Lifecycle */
myds_dspine_t *myds_dspine_create(void);
bool           myds_dspine_load_weights(myds_dspine_t *ds, const char *weights_path);
void           myds_dspine_free(myds_dspine_t *ds);

/* Context KV cache management */
void           myds_dspine_reset_context(myds_dspine_t *ds);
void           myds_dspine_truncate_context(myds_dspine_t *ds, uint32_t max_pos);
bool           myds_dspine_flush_pending(myds_dspine_t *ds, uint32_t next_pos);
bool           myds_dspine_append_context_batch(myds_dspine_t *ds,
                                                const float *target_hidden_batch,
                                                uint32_t start_pos, uint32_t count);

/* Capture hook during target model decode / prefill */
int            myds_dspine_capture_target_layer(myds_dspine_t *ds, uint32_t layer_idx,
                                                uint32_t pos,
                                                const myds_metal_buffer_t *residual_hc);
int            myds_dspine_capture_target_spec_layer(myds_dspine_t *ds, uint32_t layer_idx,
                                                     uint32_t token_row,
                                                     const myds_metal_buffer_t *residual_hc);
void           myds_dspine_capture_select_row(myds_dspine_t *ds, uint32_t base_pos,
                                              uint32_t selected_row);
void           myds_dspine_capture_invalidate(myds_dspine_t *ds);
bool           myds_dspine_capture_ready(const myds_dspine_t *ds);

/* Build Refinement Tail serial tables (tg, tsv) against base model lm_head */
bool           myds_dspine_build_tables(myds_dspine_t *ds, myds_model_t *base_model);

/* Propose candidate tokens for the current step */
uint32_t       myds_dspine_propose(myds_dspine_t *ds, myds_context_t *ctx,
                                   int32_t seed_token, uint32_t max_propose);

/* Rollback / accept count update */
void           myds_dspine_rollback(myds_dspine_t *ds, uint32_t n_accepted);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_DSPINE_H */
