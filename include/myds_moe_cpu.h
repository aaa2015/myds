/* CPU 侧的 routed MoE（prefill 用）
 *
 * 目的：prefill 的 MoE 是 GEMM，GPU 与 CPU 可并行。把一段 token 交给 CPU，
 * GPU 只算剩下的 token，两边写 moe_out 的不同行。
 *
 * 布局假设（与 runtime 完全一致）：
 *   gate/up: 专家池 [n_experts][exp_dim][dim/256] × 66 B  (IQ2_XXS)
 *   down   : 专家池 [n_experts][dim][exp_dim/256] × 84 B (Q2_K, scales 在前)
 */
#ifndef MYDS_MOE_CPU_H
#define MYDS_MOE_CPU_H

#include <stdint.h>

/* 计算 [0, n_tok) 这些 token 的 routed MoE，结果写 out（每行 dim 个 float）。
 * 权重指针取自 gguf_tensor_t->data（mmap，无需拷贝）。
 * 每个 token 的结果与批内其它 token 无关。返回 1 成功。 */
int myds_cpu_moe_tokens(const void *gate_w,
                        const void *up_w,
                        const void *down_w,
                        uint32_t dim,
                        uint32_t exp_dim,
                        uint32_t n_experts,
                        const float *x,
                        const int32_t *ids,
                        const float *wts,
                        float *out,
                        uint32_t n_tok,
                        uint32_t n_act,
                        float clamp_val,
                        uint32_t nthreads);

int myds_cpu_moe_debug_gemv(const void *w, uint32_t rows, uint32_t nb, int fmt,
                            const float *x, float *out);

#endif /* MYDS_MOE_CPU_H */
