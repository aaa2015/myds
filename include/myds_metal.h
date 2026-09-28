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
void                 myds_metal_buffer_free(myds_metal_buffer_t *buf);
void                *myds_metal_buffer_data(myds_metal_buffer_t *buf);
size_t               myds_metal_buffer_size(const myds_metal_buffer_t *buf);
int                  myds_metal_buffer_write(myds_metal_buffer_t *buf, size_t offset, const void *src, size_t bytes);
int                  myds_metal_buffer_read(const myds_metal_buffer_t *buf, size_t offset, void *dst, size_t bytes);

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

/* SwiGLU activation: out = (x0 / (1 + exp(-x0))) * x1 */
int myds_metal_swiglu(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *gate,
    const myds_metal_buffer_t *up,
    uint32_t                   dim);

/* DeepSeek-MoE Router: selects top-k experts using sigmoid gating & scaling */
int myds_metal_moe_route(
    int32_t                   *out_expert_ids,
    float                     *out_expert_weights,
    const myds_metal_buffer_t *router_logits,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    float                      routed_scaling_factor);

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

/* Synchronize GPU command queue */
int myds_metal_synchronize(void);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_METAL_H */
