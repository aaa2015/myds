#ifndef MYDS_V41_DECODE_H
#define MYDS_V41_DECODE_H

#include "myds.h"
#include "myds_metal.h"
#include "myds_v41_gpu.h"
#include <stdbool.h>

/* Single-token output-B/shared-expert projection only. The caller retains its
 * original barrier placement. Q8 may combine GEMV and the original BF16 store;
 * other formats, trace and MYDS_DECODE_Q8_FUSE_BF16=0 retain separate dispatches.
 * Output views at offsets 4/8/12 modulo 16 retain the original separate path
 * and its BF16 API's alignment rejection. They are not supported BF16 outputs.
 * A failed fused call is propagated and never retried with another kernel.
 * The environment switch is read once per process, shared by both call sites.
 * Existing MYDS_QKV_NR0 selects NR4 (default) or NR2 in the Metal implementation.
 * Missing fused PSOs are handled by that implementation's original fallback.
 */
int myds_v41_decode_gemv_bf16(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t rows, uint32_t width, ggml_type type, bool trace_on);
/* Query the same process-once fuse policy used by the scalar wrapper. */
bool myds_v41_decode_q8_fuse_bf16_enabled(void);

/* Experimental, local QKV/Indexer encoder scope. MYDS_QKV_CONCURRENT=1
 * explicitly opts in; the default and trace path keep the caller's mode.
 * End must run on success and failure, before draining/freeing GPU work.
 * It closes the dependency region and restores both caller settings. */
typedef struct {
    int auto_barrier, concurrent;
    bool active;
} myds_v41_decode_qkv_scope_t;
void myds_v41_decode_qkv_scope_begin(myds_v41_decode_qkv_scope_t *scope, bool trace_on);
void myds_v41_decode_qkv_scope_end(myds_v41_decode_qkv_scope_t *scope);

/* Default-off ordinary-decode attention HC stage overlap (MYDS_ATTN_HC_OVERLAP)
 * and compressor/indexer decode cohort overlap (MYDS_INDEXER_DECODE_OVERLAP).
 * Both accept only absent/0/1 (default 0) and fail closed on invalid env,
 * incompatible pipeline settings, aliasing scratch, or stage failures. */
int myds_v41_attn_hc_overlap_enabled(void);
int myds_v41_attn_hc_overlap_parse_mode(void);
void myds_v41_attn_hc_overlap_reset_mode(void);
void myds_v41_attn_hc_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset);

int myds_v41_indexer_decode_overlap_enabled(void);
int myds_v41_indexer_decode_overlap_parse_mode(void);
void myds_v41_indexer_decode_overlap_reset_mode(void);
void myds_v41_indexer_decode_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset);

typedef struct {
    uint32_t layer, pos, capacity;
    float hc_eps;
    myds_metal_buffer_t *residual;
    const myds_metal_buffer_t *hc_in_pre;
    myds_metal_buffer_t *attn_split;
    const myds_metal_buffer_t *attn_fn;
    const myds_metal_buffer_t *attn_scale;
    const myds_metal_buffer_t *attn_base;
    myds_metal_buffer_t *flat_norm;
    myds_metal_buffer_t *mix;
    myds_metal_buffer_t *window;
    const myds_metal_buffer_t *compressor_kv;
    const myds_metal_buffer_t *compressor_gate;
    const myds_metal_buffer_t *compressor_norm;
    const myds_metal_buffer_t *index_k_weight;
    const myds_metal_buffer_t *index_k_norm;
    const myds_metal_buffer_t *index_q_weight;
    const myds_metal_buffer_t *index_proj_weight;
    myds_metal_buffer_t *pool_kv;
    myds_metal_buffer_t *pool_score;
    myds_metal_buffer_t *latent;
    myds_metal_buffer_t *previous_kv;
    myds_metal_buffer_t *previous_score;
    myds_metal_buffer_t *compressed_kv;
    myds_metal_buffer_t *index_cache;
    myds_metal_buffer_t *index_k_buf;
    myds_metal_buffer_t *index_q_buf;
    myds_metal_buffer_t *index_weights;
    myds_metal_buffer_t *index_scores;
    myds_metal_buffer_t *selected;
    myds_metal_buffer_t *block_scores;
    myds_metal_buffer_t *block_selected;
    myds_metal_buffer_t *block_mask;
    myds_v41_gpu_scratch_t *gpu_scratch;
} myds_v41_decode_attn_args_t;

int myds_v41_decode_attn_indexer_scoped(myds_context_t *ctx,
    const myds_v41_decode_attn_args_t *args, bool ordinary_decode_allowed);

#endif
