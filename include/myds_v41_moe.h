#ifndef MYDS_V41_MOE_H
#define MYDS_V41_MOE_H
#include "myds_internal.h"
/* Native V4.1 text router. Scratch probabilities/scores are [n_experts] F32;
 * selected IDs are I32. The current target uses 384 experts and top-6. */
int myds_v41_route(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probabilities, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t n_experts, uint32_t top_k, float scale);
int myds_v41_shared_swiglu(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *gate, const myds_metal_buffer_t *up,
    uint32_t width, float clamp);
/* Input ctx->ffn_norm_out; output ctx->ffn_out. Retains an active GPU batch. */
int myds_v41_moe(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos);
int myds_v41_moe_eval(myds_context_t *ctx, uint32_t layer, uint32_t pos,
                      const myds_metal_buffer_t *ffn_norm,
                      const myds_metal_buffer_t *router_logits,
                      const myds_metal_buffer_t *shared_out);
#endif
