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
/* MYDS_ROUTE_SIMD_SHUFFLE accepts absent/0/1, default 0, at each route entry.
 * Mode1 selects an independent exact all-in-one PSO and never falls back.
 * Requests count validated eligible calls; encoded counts successful host
 * dispatch calls, not GPU completion. bench_phase reports decode_* deltas
 * from the first callback snapshot, matching its N-1 decode intervals. */
int myds_v41_route_simd_shuffle_mode(void);
typedef struct { uint64_t requests, encoded, failures; }
    myds_v41_route_simd_shuffle_counters_t;
void myds_v41_route_simd_shuffle_counters(
    myds_v41_route_simd_shuffle_counters_t *out, bool reset);
/* MYDS_MOE_HC_OVERLAP: absent/0 -> 0, 1 -> 1, otherwise -1. Parse at
 * context creation, then use the cached getter on the hot path. A later
 * context creation reparses; reset_mode is for module host fixtures (the
 * Metal-only runtime does not depend on this C module). Explicit ordinary
 * nontrace calls outside fixed101 active batches fail before dispatch.
 * Nonordinary/trace origins preserve their original helper. Encoded
 * counts a successful original HC split-stage dispatch, not GPU completion;
 * a later layer failure also increments failures and terminates the layer. */
int myds_v41_moe_hc_overlap_enabled(void);
int myds_v41_moe_hc_overlap_parse_mode(void);
void myds_v41_moe_hc_overlap_reset_mode(void);
void myds_v41_moe_hc_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset);
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
/* Production two-row execution retains the authoritative single-row PSO,
 * reduction and BF16 boundaries for each input. The caller joins inputs and
 * outputs at the same boundaries as the single-row entry. */
int myds_v41_shared_gate_up_swiglu_pair(myds_metal_buffer_t *const out[2],
                                      const myds_metal_buffer_t *gate,
                                      const myds_metal_buffer_t *up,
                                      myds_metal_buffer_t *const input[2],
                                      uint32_t out_dim, uint32_t in_dim,
                                      ggml_type gate_type, ggml_type up_type,
                                      float clamp);
/* Three rows with one quantized weight load and independent scalar-order
 * accumulators/BF16 boundaries. Unsupported inputs fail without fallback. */
int myds_v41_shared_gate_up_swiglu_shared3(myds_metal_buffer_t *const out[3],
    const myds_metal_buffer_t *gate, const myds_metal_buffer_t *up,
    myds_metal_buffer_t *const input[3], uint32_t out_dim, uint32_t in_dim,
    ggml_type gate_type, ggml_type up_type, float clamp);
typedef struct { uint64_t requests, encoded, failures; }
    myds_v41_shared_gu_shared3_counters_t;
void myds_v41_shared_gu_shared3_counters(
    myds_v41_shared_gu_shared3_counters_t *out, bool reset);
/* Diagnostic candidate only. This shared-weight kernel has not passed the
 * current checkpoint's full-model parity gate and is not selected by Batch2. */
int myds_v41_shared_gate_up_swiglu_batch2(myds_metal_buffer_t *out,
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
/* Explicit scalar-logits origin; older wrappers never opt into Q2 sum cache. */
int myds_v41_moe_pipeline_scoped(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
                          myds_metal_buffer_t *residual,
                          const myds_metal_buffer_t *after_attn,
                          myds_metal_buffer_t *ffn_split,
                          const myds_metal_buffer_t *ffn_fn,
                          const myds_metal_buffer_t *ffn_scale,
                          const myds_metal_buffer_t *ffn_base,
                          myds_metal_buffer_t *flat_norm,
                          myds_metal_buffer_t *mix,
                          float hc_eps, bool ordinary_decode_allowed);
int myds_v41_moe_eval(myds_context_t *ctx, uint32_t layer, uint32_t pos,

                      const myds_metal_buffer_t *ffn_norm,
                      const myds_metal_buffer_t *router_logits,
                      const myds_metal_buffer_t *shared_out);

/* DSpark 专属融合路由内核：128 专家、Top-K (通常为 3) 全在位双调排序与归一化
 * 消除 CPU-GPU 交互与批次中断 */
int myds_dspark_route_fused(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
                            const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
                            uint32_t top_k, float scale, uint32_t rows);

/* DSpark Q2_K gate/up/down with three dynamically selected experts. F32
 * SwiGLU clamps gate only above limit and up to [-limit, limit], applies route
 * weights before down projection and sums exactly three expert outputs.
 * dim/expert_dim must be multiples of 256. scratch_mid holds 3*expert_dim
 * floats; other scratch arguments preserve the routed_moe call signature and
 * are unused. Unsupported quantization or top-k fails before GPU dispatch. */
int myds_dspark_moe(myds_metal_buffer_t *moe_out,
    const myds_metal_buffer_t *x_in,
    const myds_metal_buffer_t *gate_exps, const myds_metal_buffer_t *up_exps,
    const myds_metal_buffer_t *down_exps,
    const myds_metal_buffer_t *selected_ids, const myds_metal_buffer_t *selected_weights,
    myds_metal_buffer_t *scratch_gate, myds_metal_buffer_t *scratch_up,
    myds_metal_buffer_t *scratch_mid, myds_metal_buffer_t *scratch_down,
    uint32_t dim, uint32_t expert_dim, uint32_t n_routed_experts,
    uint32_t n_activated_experts, ggml_type gate_type, ggml_type up_type,
    ggml_type down_type, float clamp_val);
/* DSpark shared experts use the same clamp but preserve F32 output. */
int myds_dspark_shared_swiglu(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *gate, const myds_metal_buffer_t *up,
    uint32_t width, float clamp_val);

#define MYDS_V41_MOE_HAS_FFN_HC_OVERLAP 1
int myds_v41_ffn_hc_overlap_parse_mode(void);
int myds_v41_ffn_hc_overlap_enabled(void);
void myds_v41_ffn_hc_overlap_reset_mode(void);
void myds_v41_ffn_hc_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset);
int myds_v41_moe_ffn_hc_overlap_scoped(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
    myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *after_attn,
    const myds_metal_buffer_t *attn_split,
    myds_metal_buffer_t *ffn_split,
    const myds_metal_buffer_t *ffn_fn,
    const myds_metal_buffer_t *ffn_scale,
    const myds_metal_buffer_t *ffn_base,
    myds_metal_buffer_t *flat_norm,
    myds_metal_buffer_t *mix,
    float hc_eps, bool ordinary_decode_allowed);

#endif
