#ifndef MYDS_V41_MOE_H
#define MYDS_V41_MOE_H
#include "myds_internal.h"
/* Native V4.1 text router. Scratch probabilities/scores are [n_experts] F32;
 * selected IDs are I32. The current target uses 384 experts and top-6. */
int myds_v41_route(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probabilities, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t n_experts, uint32_t top_k, float scale);
/* 批量版：一次处理 rows 个 token 的路由（logits/probs/scores 为 rows*n_experts 连续，
 * selected/weights 为 rows*top_k 连续，denominator 为 rows 个）。数值与逐 token 版一致。 */
int myds_v41_route_batch(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probabilities, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t n_experts, uint32_t top_k, float scale, uint32_t rows);
int myds_v41_shared_swiglu(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *gate, const myds_metal_buffer_t *up,
    uint32_t width, float clamp);
/* Shared Expert Gate + Up + SwiGLU 三合一融合算子
 * 消除 2 次 GPU Dispatch 与中间 buffer 显存往返。
 * 当权重非 Q8_0 或未启用时返回 0，调用方平滑 fallback 到原有分步路径。 */
int myds_v41_shared_gate_up_swiglu(myds_metal_buffer_t *out,
                                  const myds_metal_buffer_t *gate,
                                  const myds_metal_buffer_t *up,
                                  const myds_metal_buffer_t *input,
                                  uint32_t out_dim, uint32_t in_dim,
                                  ggml_type gate_type, ggml_type up_type,
                                  float clamp);
/* Input ctx->ffn_norm_out; output ctx->ffn_out. Retains an active GPU batch. */
int myds_v41_moe(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos);
/* S10 算子融合版：MoE Output + Shared + HC Expand 融合为单次派发 */
int myds_v41_moe_fused(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
                       myds_metal_buffer_t *residual,
                       const myds_metal_buffer_t *after_attn,
                       const myds_metal_buffer_t *ffn_split);
/* S11 跨阶段全并发流水线：HC Mix 与 Shared Expert / Router GEMV / MoE 深度重叠 */
int myds_v41_moe_pipeline(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
                          myds_metal_buffer_t *residual,
                          const myds_metal_buffer_t *after_attn,
                          myds_metal_buffer_t *ffn_split,
                          const myds_metal_buffer_t *ffn_fn,
                          const myds_metal_buffer_t *ffn_scale,
                          const myds_metal_buffer_t *ffn_base,
                          myds_metal_buffer_t *flat_norm,
                          myds_metal_buffer_t *mix,
                          float hc_eps);
int myds_v41_moe_eval(myds_context_t *ctx, uint32_t layer, uint32_t pos,

                      const myds_metal_buffer_t *ffn_norm,
                      const myds_metal_buffer_t *router_logits,
                      const myds_metal_buffer_t *shared_out);
#endif
