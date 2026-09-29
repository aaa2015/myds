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
/* out may equal residual: the specialized four-stream kernel loads all four
 * source stream values before writing. Other input/output overlap is invalid. */
int myds_metal_v41_hc_expand(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *block, const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *split, uint32_t dim);
/* projected is F32[5,dim]; q/k weights and residual are F32[4,dim]. */
int myds_metal_v41_engram_add(myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *projected, const myds_metal_buffer_t *q_weight,
    const myds_metal_buffer_t *k_weight, uint32_t dim, float eps);

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

/* Softmax over row */
int myds_metal_softmax(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *x,
    uint32_t                   count);

/* Batch command buffer management */
int      myds_metal_begin_batch(void);
int      myds_metal_commit_batch(void);
uint64_t myds_metal_get_commit_count(void);
void     myds_metal_barrier(void);
void     myds_metal_set_auto_barrier(int enable);

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
