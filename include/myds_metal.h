#ifndef MYDS_METAL_H
#define MYDS_METAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "myds_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Metal buffer handle (unified memory on Apple Silicon) */
typedef struct myds_metal_buffer myds_metal_buffer_t;

/* Initialize Metal device, command queue, and compile DeepSeek shaders */
int  myds_metal_init(const char *shader_dir);
void myds_metal_cleanup(void);

/* Independent F16 GEMV BF16-store experiment. The parser performs no Metal
 * operation: absent/"0" -> 0, "1" -> 1, all other values -> -1. The mode is
 * published only by successful Metal initialization, never reread at launch,
 * and reset by cleanup. Explicit1 requires all NSG1..8 candidate PSOs. */
int myds_metal_f16_gemv_bf16_fuse_parse_mode(void);
int myds_metal_f16_gemv_bf16_fuse_enabled(void);
typedef struct { uint64_t requests, encoded, failures; }
    myds_metal_f16_gemv_bf16_fuse_counters_t;
/* Per-field relaxed snapshots; reset belongs to an idle/exclusive boundary.
 * Requests/failures count explicit API calls. Encoded counts dispatches, not
 * completed GPU work; outside-batch completion failure may count both encoded
 * and failure, while an enclosing batch failure is reported by its caller. */
void myds_metal_f16_gemv_bf16_fuse_counters(
    myds_metal_f16_gemv_bf16_fuse_counters_t *out, bool reset);
/* Explicit candidate: F16 weights, one F32 input row, NR2 and the unchanged
 * input-width NSG/reduction, with BF16 values stored as F32. Requires mode1;
 * no CPU/reference fallback. Width must be a positive multiple of four.
 * Weight/input/output offsets and shared pointers require 8/16/16 alignment.
 * Odd output widths require one padded weight row because the unchanged NR2
 * dot loop reads two rows; output has only out_dim words. Output may not
 * overlap either input range, including distinct no-copy wrappers or views.
 * With auto barriers off, the caller must join previous input/output users
 * before launch and join OUT before consuming or reusing any buffers. This
 * helper adds no defensive join; auto1 retains the common dispatch barrier.
 * Production opts in only from an explicit ordinary scalar projection scope;
 * this diagnostic API itself does not infer Engine/decode purpose. */
int myds_metal_v41_gemv_f16_bf16(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim);

/* Independent indexer forward RoPE + mode2 E2M1 candidate. Strict parsing is
 * CPU-only: absent/"0" -> 0, "1" -> 1, otherwise -1. Successful initialization
 * publishes an immutable mode; explicit1 requires the separate SIMD32 PSO.
 * Cleanup unpublishes the mode and releases the PSO. */
int myds_metal_indexer_rope_quant_fuse_parse_mode(void);
int myds_metal_indexer_rope_quant_fuse_enabled(void);
typedef struct { uint64_t requests, encoded, failures; }
    myds_metal_indexer_rope_quant_fuse_counters_t;
/* Relaxed per-field host encoding snapshots, reset only when idle/exclusive.
 * Encoded is not command completion. A standalone completion failure counts
 * both encoded and failure; an enclosing batch failure belongs to its caller. */
void myds_metal_indexer_rope_quant_fuse_counters(
    myds_metal_indexer_rope_quant_fuse_counters_t *out, bool reset);
/* Internal explicit candidate bridge. One contiguous row of heads1 or heads32,
 * each 128 F32 values; forward compressed frequencies are supplied by the V4.1
 * wrapper. In-place, no CPU/reference fallback. Checks device, full range and
 * actual offset/pointer 4-byte alignment before any writes/dispatch. Auto0
 * requires caller input/WAR joins and an OUT join before consume/reuse; this
 * bridge adds no defensive join. Auto1 retains the common dispatch barrier. */
int myds_metal_indexer_rope_quant_dispatch(myds_metal_buffer_t *x,
    uint32_t heads, uint32_t pos, const float frequencies[32]);

/* Internal head-slot scorer bridge. Policy is fixed by Metal initialization,
 * reset by cleanup, and has no public setter. An explicit1 init requires the
 * independent PSO (SIMD32, 128 threads, 640 dynamic TG bytes). Shape/ranges,
 * device identity, alignment and writable/read-only overlap are checked before
 * encoding. Actual dispatch and unavailable status are returned separately so
 * the V4.1 host counters do not confuse encoding with command completion. */
int myds_metal_indexer_head_slots_enabled(void);
int myds_metal_indexer_head_slots_dispatch(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t n_heads,bool cache_bf16,
    bool *encoded,bool *unavailable);

/* Initialization-time LM-head experiment. MYDS_LM_HEAD_Q8_NR0 accepts only
 * absent/2/4; absent and 2 retain the original path. NR4 applies solely to
 * Q8_0, one token, out_dim>65536, and keeps NSG8/full F32 logits. Explicit NR4
 * requires its PSO/device geometry during init; dispatch errors never fall
 * back to CPU or NR2. Cleanup resets the mode to 2. No public setter exists.
 * Counters describe NR4 host requests/encoding/failures, not GPU time or
 * physical traffic. An encoded outside-batch command may also fail completion;
 * a later enclosing-batch failure is reported by the caller's graph gate. */
typedef struct { uint64_t requests, encoded, failures; } myds_metal_lm_head_q8_counters_t;
int myds_metal_lm_head_q8_nr0(void);
void myds_metal_lm_head_q8_counters(myds_metal_lm_head_q8_counters_t *out, bool reset);

/* Buffer allocation and memory mapping */
myds_metal_buffer_t *myds_metal_buffer_alloc(size_t bytes);
myds_metal_buffer_t *myds_metal_buffer_from_ptr(const void *ptr, size_t bytes);
myds_metal_buffer_t *myds_metal_buffer_sub_view(const myds_metal_buffer_t *parent, size_t offset, size_t bytes);
myds_metal_buffer_t *myds_metal_buffer_create_view(void);
int                  myds_metal_buffer_update_sub_view(myds_metal_buffer_t *view, const myds_metal_buffer_t *parent, size_t offset, size_t bytes);
void                 myds_metal_buffer_free(myds_metal_buffer_t *buf);
void                *myds_metal_buffer_data(myds_metal_buffer_t *buf);
size_t               myds_metal_buffer_size(const myds_metal_buffer_t *buf);
size_t               myds_metal_buffer_offset(const myds_metal_buffer_t *buf);
bool                 myds_metal_buffer_is_view(const myds_metal_buffer_t *buf);
int                  myds_metal_buffer_write(myds_metal_buffer_t *buf, size_t offset, const void *src, size_t bytes);
int                  myds_metal_buffer_read(const myds_metal_buffer_t *buf, size_t offset, void *dst, size_t bytes);
/* GPU-ordered copy; overlapping source/destination ranges are rejected. */
/* 诊断用：GPU dispatch 计数（参谋长 2026-09-30） */
#include <stdio.h>
uint64_t myds_metal_dispatch_count(void);
int myds_metal_dispatch_hist_dump(FILE *out);
void myds_metal_dispatch_count_reset(void);

int myds_metal_buffer_copy(myds_metal_buffer_t *dst, size_t dst_offset,
    const myds_metal_buffer_t *src, size_t src_offset, size_t bytes);
/* Internal native-graph bridge. args occupies slot0, buffers slots1..N.
 * The caller validates the selected kernel's tensor shapes/strides. The bridge
 * validates dispatch limits and encodes an unconditional dependency barrier. */
int myds_metal_dispatch_kernel(const char *name, const void *args, size_t args_size,
    const myds_metal_buffer_t *const *buffers, size_t n_buffers,
    const uint32_t groups[3], const uint32_t threads[3], size_t threadgroup_memory);
typedef struct { uint32_t index; bool is_bool; int32_t value; } myds_metal_constant_t;
uint32_t myds_metal_kernel_max_threads(const char *name);
/* Independent initialization-time partition-NaN shuffle experiment. Only
 * absent/0/1 are accepted. Preflight preserves the reference partition size
 * and validates both sort and merge before the caller writes any scratch. */
int myds_metal_indexer_topk_nan_shuffle_enabled(void);
int myds_metal_indexer_topk_nan_shuffle_preflight(bool causal,uint32_t reference_nth);
int myds_metal_indexer_topk_threshold_enabled(void);
/* Internal candidate preflight: initializes/checks all independent PSOs and
 * the init-owned GPU counter buffer without changing policy or waiting. */
int myds_metal_indexer_topk_threshold_prepare(void);
myds_metal_buffer_t *myds_metal_indexer_topk_threshold_stats_buffer(void);
int myds_metal_indexer_topk_threshold_stats(uint32_t words[9],bool reset);
int myds_metal_buffer_native_valid(const myds_metal_buffer_t *buffer);
int myds_metal_dispatch_kernel_constants(const char *name, const void *args, size_t args_size,
    const myds_metal_buffer_t *const *buffers, size_t n_buffers,
    const uint32_t groups[3], const uint32_t threads[3], size_t threadgroup_memory,
    const myds_metal_constant_t *constants, size_t n_constants);

/* =========================================================================
 * Core DeepSeek GPU Operators
 * ========================================================================= */

/* RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight */
int myds_metal_rmsnorm(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *weight,
    uint32_t                   dim,
    float                      eps);

/* V4.1 numerical/HC primitives. All activation buffers contain float32 words;
 * BF16 results are stored as exactly representable float32 values. These
 * helpers enforce their own GPU dependencies even with auto barriers disabled.
 * RMSNorm accepts NULL weight for an unweighted norm. Width must be a multiple
 * of 128 (a complete SIMD group of float4 values); its reduction matches ds4.
 */
int myds_metal_v41_bf16(myds_metal_buffer_t *x, uint32_t count);
int myds_metal_v41_rmsnorm(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *x, const myds_metal_buffer_t *weight,
    uint32_t dim, float eps, bool round_bf16);

/* One token, four stream-major residual rows [4, dim]. The mixer projection
 * fn is [24, 4*dim], scale is F32[3], base is F32[24]. Scratch flat_norm is
 * F32[4*dim], mix is F32[24]. split receives [pre4, post4, comb4x4] using
 * 20 Sinkhorn iterations. HC normalization/projection remain F32.
 * V4.1 attention consumes the PREVIOUS FFN pre4 (initially {1,0,0,0});
 * FFN consumes the current attention pre4. The current split is used for
 * post/comb expansion, and FFN pre4 is carried into the next layer/head.
 */
int myds_metal_v41_hc_mix(myds_metal_buffer_t *split,
    const myds_metal_buffer_t *residual, const myds_metal_buffer_t *fn,
    ggml_type fn_type, const myds_metal_buffer_t *scale,
    const myds_metal_buffer_t *base, myds_metal_buffer_t *flat_norm,
    myds_metal_buffer_t *mix, uint32_t dim, float eps, float hc_eps);
int myds_metal_v41_hc_weighted_sum(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *residual, const myds_metal_buffer_t *weights,
    uint32_t dim);
/* Fuses HC-4 weighted reduction and RMSNorm + weight multiply into a single
 * dispatch. If sum_out is non-NULL, writes the BF16-rounded intermediate sum to sum_out.
 * norm_out receives the final normalized and weighted activations. */
int myds_metal_v41_hc_sum_rmsnorm(
    myds_metal_buffer_t *norm_out,
    myds_metal_buffer_t *sum_out,
    const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *norm_weight,
    uint32_t dim,
    float eps,
    bool round_bf16);
/* out may equal residual: the specialized four-stream kernel loads all four
 * source stream values before writing. Other input/output overlap is invalid. */
int myds_metal_v41_hc_expand(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *block, const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *split, uint32_t dim);
int myds_metal_v41_moe_add_hc_expand(myds_metal_buffer_t *out,
    myds_metal_buffer_t *ffn_out, const myds_metal_buffer_t *moe_out,
    const myds_metal_buffer_t *shared_out, const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *split, uint32_t dim);
/* projected is F32[5,dim]; q/k weights and residual are F32[4,dim]. */
int myds_metal_v41_engram_add(myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *projected, const myds_metal_buffer_t *q_weight,
    const myds_metal_buffer_t *k_weight, uint32_t dim, float eps);
int myds_metal_v41_engram_add_batch(myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *projected, const myds_metal_buffer_t *q_weight,
    const myds_metal_buffer_t *k_weight, uint32_t dim, uint32_t rows, float eps);

/* Decoupled RoPE (Rotary Position Embedding) for DeepSeek Q and K */
int myds_metal_rope(
    myds_metal_buffer_t *q_rope,
    myds_metal_buffer_t *k_rope,
    uint32_t             n_heads,
    uint32_t             rope_dim,
    uint32_t             pos,
    float                freq_base,
    float                freq_scale);

/* Quantized Matrix-Vector Multiply: y = W * x */
int myds_metal_gemv(
    myds_metal_buffer_t       *y,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *x,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    ggml_type                  weight_type);

/* V4.1 F16 projection dispatch. The reference graph uses the four-wide
 * matvec kernel with its input-width-dependent eight-simdgroup reduction;
 * this is deliberately separate from the legacy GEMV defaults. */
int myds_metal_v41_gemv(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim, ggml_type weight_type);

/* Decode QKV projection: same Q8_0 GEMV reduction with a BF16 final store.
 * Other types / unavailable fused PSOs use the original projection + BF16.
 * The capability helper disables fused window stores during copy injection. */
int myds_metal_v41_gemv_bf16(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight,const myds_metal_buffer_t *input,
    uint32_t out_dim,uint32_t in_dim,ggml_type weight_type);
/* Experimental two-input Q8_0 projection, used only by Batch2 output-B.
 * Each row retains the scalar GEMV K walk, NSG/NR and BF16 store. Outputs
 * must not overlap each other or any input/weight range; inputs may overlap.
 * There is no scalar/CPU fallback after this helper is requested.
 * mode: absent/"0" -> 0, "1" -> 1, any other value -> -1 (with diagnostic).
 * Encoded counts dispatches, including those whose later commit may fail;
 * requests/failures count calls to this helper, not scalar bypasses. */
int myds_metal_batch2_q8_shared_mode(void);
int myds_metal_batch2_q4_shared_mode(void);
int myds_metal_batch2_dense_shared_mode(void);
int myds_metal_v41_gemv_bf16_shared2(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim);
int myds_metal_v41_gemv_bf16_shared2_type(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim, ggml_type type);
int myds_metal_v41_gemv_shared2_dense(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim, ggml_type type);
int myds_metal_v41_gemv_shared2_q8_f32(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim);
typedef struct {
    uint64_t requests, encoded, failures;
} myds_metal_batch2_q8_shared_counters_t;
/* Relaxed per-field snapshots/exchanges; reset belongs to an idle/exclusive
 * measurement boundary, not a concurrent multi-request accounting window. */
void myds_metal_batch2_q8_shared_counters(
    myds_metal_batch2_q8_shared_counters_t *out, bool reset);
int myds_metal_v41_qkv_fusion_available(void);

int myds_metal_v41_gemv_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim, uint32_t n_tok, ggml_type weight_type);

/* Multiple independent GEMVs in one dispatch. Uses the same arithmetic and
 * reduction as gemv, with contiguous input/output rows in token order. */
int myds_metal_gemv_batch(
    myds_metal_buffer_t       *y,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *x,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    uint32_t                   n_tok,
    ggml_type                  weight_type);

/* Batched Matrix-Matrix Multiplication (GEMM): dst [n_tok, out_dim] = src1 [n_tok, in_dim] * W^T [out_dim, in_dim] */
int myds_metal_mul_mm(
    myds_metal_buffer_t       *dst,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *src1,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    uint32_t                   n_tok,
    ggml_type                  weight_type);
/* 分组矩阵乘：一次派发完成所有组（见 myds_metal.m 的说明）。 */
int myds_metal_mul_mm_grouped(
    myds_metal_buffer_t       *dst,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *src1,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    uint32_t                   n_groups,
    uint32_t                   n_tok,
    ggml_type                  weight_type);

/* SwiGLU activation: out = (x0 / (1 + exp(-x0))) * x1 */
int myds_metal_swiglu(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *gate,
    const myds_metal_buffer_t *up,
    uint32_t                   dim);

/* DeepSeek-MoE Router: selects top-k experts using sigmoid/softplus gating & scaling */
int myds_metal_moe_route(
    int32_t                   *out_expert_ids,
    float                     *out_expert_weights,
    const myds_metal_buffer_t *router_logits,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    float                      routed_scaling_factor);

/* DeepSeek V4.1 Biased Router: softplus + router_bias addition + top-k + normalized router weights */
int myds_metal_moe_route_biased(
    int32_t                   *out_expert_ids,
    float                     *out_expert_weights,
    const myds_metal_buffer_t *router_logits,
    const myds_metal_buffer_t *router_bias,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    float                      routed_scaling_factor);

/* DeepSeek Routed MoE execution on GPU: gate/up (IQ2_XXS) -> SwiGLU -> down (Q2_K) -> sum6 */
/* MYDS_MOE_Q2_SUM_CACHE is an init-time absent/0/1 setting (default 0).
 * cleanup + init is required to change it. Existing routed single/batch APIs
 * always retain the reference; only the explicitly scoped API may use it. */
int myds_metal_moe_q2_sum_cache_enabled(void);
typedef struct {
    uint64_t requests, encoded, failures;
} myds_metal_moe_q2_sum_cache_counters_t;
/* Eligible scoped attempts and successful whole IQ2->prep->Q2 encodings.
 * Later batch command completion remains the caller's responsibility. */
void myds_metal_moe_q2_sum_cache_counters(
    myds_metal_moe_q2_sum_cache_counters_t *out, bool reset);
/* MYDS_MOE_Q2_SUM_CACHE_FUSED is a separate init-time absent/0/1 setting
 * (default 0), mutually exclusive with MYDS_MOE_Q2_SUM_CACHE=1. Fused mode
 * requires MYDS_MOE_IQ2_CONSTANT_LUT=0/absent and IQ2 NSG2/4. cleanup + init
 * is required to change modes; malformed, conflicting or unavailable explicit
 * configurations fail initialization without fallback. The scoped API below
 * uses the same legal shape and caller dependency contract, with IQ2 producing
 * F32 MID and the private F32 cache in one dispatch, then cached Q2 down.
 * Existing unscoped single/batch APIs always remain the reference. */
int myds_metal_moe_q2_sum_cache_fused_enabled(void);
typedef struct {
    uint64_t requests, encoded, failures;
} myds_metal_moe_q2_sum_cache_fused_counters_t;
/* Eligible fused attempts and successful whole two-dispatch host encodings.
 * These are not GPU completion counts. The old independent-cache counters
 * stay zero when only fused mode is enabled. Both counter sets survive cleanup
 * until explicitly reset. */
void myds_metal_moe_q2_sum_cache_fused_counters(
    myds_metal_moe_q2_sum_cache_fused_counters_t *out, bool reset);
/* Scope must come from the public, nontrace scalar logits graph, including
 * legacy serial verifier callers. Scalar prefill and private Batch2/tree/rescue
 * origins are excluded. Explicit mode1 requests
 * require DIM5120, width2304, active6, IQ2/IQ2/Q2 and legal buffer ranges.
 * An eligible one-token MM override is rejected instead of changing its tree.
 * Caller preserves upstream X/ID/route-weight RAW dependencies as for the
 * reference API. Candidate keeps mid RAW and adds cache RAW. As in the
 * reference, a batch caller with auto barriers disabled must join OUT before
 * consuming it or issuing another call that overwrites MID/private cache.
 * That join completes the cached down readers before scratch reuse. Automatic
 * barriers retain the reference's final buffer-scope barrier; standalone calls
 * wait for command completion. Default reference is unchanged.
 * Invalid eligible requests fail without reference or CPU fallback. */
int myds_metal_routed_moe_scoped(
    myds_metal_buffer_t       *moe_out,
    const myds_metal_buffer_t *x_in,
    const myds_metal_buffer_t *gate_exps,
    const myds_metal_buffer_t *up_exps,
    const myds_metal_buffer_t *down_exps,
    const myds_metal_buffer_t *selected_ids,
    const myds_metal_buffer_t *selected_weights,
    myds_metal_buffer_t       *scratch_gate,
    myds_metal_buffer_t       *scratch_up,
    myds_metal_buffer_t       *scratch_mid,
    myds_metal_buffer_t       *scratch_down,
    uint32_t                   dim,
    uint32_t                   expert_dim,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    ggml_type                  gate_type,
    ggml_type                  up_type,
    ggml_type                  down_type,
    float                      clamp_val,
    bool                       ordinary_decode_allowed);

int myds_metal_routed_moe(
    myds_metal_buffer_t       *moe_out,
    const myds_metal_buffer_t *x_in,
    const myds_metal_buffer_t *gate_exps,
    const myds_metal_buffer_t *up_exps,
    const myds_metal_buffer_t *down_exps,
    const myds_metal_buffer_t *selected_ids,
    const myds_metal_buffer_t *selected_weights,
    myds_metal_buffer_t       *scratch_gate,
    myds_metal_buffer_t       *scratch_up,
    myds_metal_buffer_t       *scratch_mid,
    myds_metal_buffer_t       *scratch_down,
    uint32_t                   dim,
    uint32_t                   expert_dim,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    ggml_type                  gate_type,
    ggml_type                  up_type,
    ggml_type                  down_type,
    float                      clamp_val);

int myds_metal_routed_moe_batch(
    myds_metal_buffer_t       *moe_out,
    const myds_metal_buffer_t *x_in,
    const myds_metal_buffer_t *gate_exps,
    const myds_metal_buffer_t *up_exps,
    const myds_metal_buffer_t *down_exps,
    const myds_metal_buffer_t *selected_ids,
    const myds_metal_buffer_t *selected_weights,
    myds_metal_buffer_t       *scratch_gate,
    myds_metal_buffer_t       *scratch_up,
    myds_metal_buffer_t       *scratch_mid,
    myds_metal_buffer_t       *scratch_down,
    uint32_t                   dim,
    uint32_t                   expert_dim,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    uint32_t                   n_tok,
    ggml_type                  gate_type,
    ggml_type                  up_type,
    ggml_type                  down_type,
    float                      clamp_val);

/* Grouped attention output down-projection on GPU: heads -> attn_low */
int myds_metal_attn_output_a(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    ggml_type                  weight_type);

int myds_metal_attn_output_a_bf16(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    ggml_type                  weight_type);

int myds_metal_attn_output_a_bf16_batch(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    uint32_t                   n_tok,
    ggml_type                  weight_type);

int myds_metal_attn_output_a_batch(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    uint32_t                   n_tok,
    ggml_type                  weight_type);

/* DeepSeek-MoE Expert Output Accumulation: y = sum(w_i * expert_out_i) */
int myds_metal_moe_sum(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *expert_outs,
    const float               *weights,
    uint32_t                   n_activated,
    uint32_t                   dim);

/* Element-wise residual addition: out = a + b */
int myds_metal_add(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b,
    uint32_t                   count);

/* Element-wise residual addition with fused BF16 round epilogue: out = bf16(a + b) */
int myds_metal_v41_add_bf16(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b,
    uint32_t                   count);


/* Softmax over row */
int myds_metal_softmax(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *x,
    uint32_t                   count);

/* Batch command buffer management */
int      myds_metal_begin_batch(void);
int      myds_metal_commit_batch(void);
int      myds_metal_flush_batch(void);
uint64_t myds_metal_get_commit_count(void);
void     myds_metal_barrier(void);
void     myds_metal_barrier_buffers(const myds_metal_buffer_t *const *buffers, size_t count);
void     myds_metal_barrier_buffer(const myds_metal_buffer_t *buf);
void     myds_metal_set_auto_barrier(int enable);
int      myds_metal_get_auto_barrier(void);
void     myds_metal_set_concurrent(int enable);
int      myds_metal_get_concurrent(void);
bool     myds_trace_active(void);

/* Diagnostic fault injection for testing (0 = disabled) */
void     myds_metal_inject_copy_failure(uint32_t fail_at_call_index);
void     myds_metal_inject_commit_failure(uint32_t fail_at_call_index);
uint32_t myds_metal_get_copy_call_count(void);
uint32_t myds_metal_get_commit_call_count(void);

/* Shared Event management for overlapping router gate with shared expert.
 * commit_and_wait_event waits only through the signal point; subsequent work
 * remains in flight until commit_batch, synchronize, or cleanup waits for it. */
int      myds_metal_signal_event(uint64_t *out_val);
int      myds_metal_wait_event(uint64_t val);
int      myds_metal_commit_and_wait_event(uint64_t val);

/* Submit an active batch and wait for all submissions, including work after
 * shared-event signals. Returns 0 if any command buffer failed. */
int myds_metal_synchronize(void);

/* Pre-warm memory pages in unified RAM */
void myds_metal_warm_memory(const void *addr, size_t size);
void myds_metal_warm_buffer(const myds_metal_buffer_t *buf);

/* Residency keep-alive (llama.cpp PR #17766 的移植).
 *
 * 在 macOS 上，驻留集的 wired 状态会在进程空闲后退化，空闲后的第一条命令缓冲
 * 要为每个已绑定字节付一次逐页重校验。activate() 在【模型映射完成后】调用一次，
 * 把这次一次性开销付在加载期并启动 5 ms 心跳；keep_alive() 在每次真实提交时重置
 * 倒计时；stop() 用于收尾。MYDS_NO_RESIDENCY_HB=1 可整体关闭。
 * 依据与实测：docs/verification/2026-09-29-per-request-latency-root-cause.md §14 */
void myds_metal_residency_activate(void);
void myds_metal_residency_keep_alive(void);
void myds_metal_residency_stop(void);

/* Batched V4.1 HC and RMSNorm primitives across multiple tokens */
int myds_metal_v41_rmsnorm_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *x, const myds_metal_buffer_t *weight,
    uint32_t dim, uint32_t n_tok, float eps, bool round_bf16);

int myds_metal_v41_hc_weighted_sum_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *residual, const myds_metal_buffer_t *weights,
    uint32_t dim, uint32_t n_tok, uint64_t weight_stride_bytes);

int myds_metal_v41_hc_expand_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *block, const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *split, uint32_t dim, uint32_t n_tok);

int myds_metal_v41_hc_split_batch(myds_metal_buffer_t *split,
    const myds_metal_buffer_t *mix, const myds_metal_buffer_t *scale,
    const myds_metal_buffer_t *base, uint32_t n_tokens, float hc_eps);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_METAL_H */
