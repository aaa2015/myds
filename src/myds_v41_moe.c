#include "myds_v41_moe.h"
#include "myds_log.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* 必须与 metal/myds_v41_moe.metal 的 myds_v41_route_args 逐字段一致 */
struct route_args { uint32_t count, top_k, op; float scale; uint32_t rows, row, row4; };
struct sort_args {
    int32_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne0, ne1, ne2, ne3, top_k;
    uint32_t causal_start, causal_ratio;
};
struct glu_args {
    int32_t ne00; uint64_t nb01;
    int32_t ne10; uint64_t nb11;
    int32_t ne0; uint64_t nb1;
    int32_t i00, i10; float alpha, limit; int32_t round_bf16;
};

static int sized(const myds_metal_buffer_t *buf, size_t count) {
    return buf && count <= myds_metal_buffer_size(buf) / sizeof(float);
}
static int dispatch(const char *kernel, const void *args, size_t size,
    const myds_metal_buffer_t *const *buffers, size_t count,
    uint32_t groups, uint32_t threads, size_t scratch) {
    const uint32_t grid[3] = {groups, 1, 1}, block[3] = {threads, 1, 1};
    return myds_metal_dispatch_kernel(kernel, args, size, buffers, count, grid, block, scratch);
}

int myds_v41_route_batch(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probs, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t experts, uint32_t top_k, float scale, uint32_t rows) {
    if (experts != 384 || top_k != 6 || !isfinite(scale) || scale <= 0 || rows == 0 ||
        !sized(selected, (size_t)top_k * rows) || !sized(weights, (size_t)top_k * rows) ||
        !sized(probs, (size_t)experts * rows) || !sized(scores, (size_t)experts * rows) ||
        !sized(logits, (size_t)experts * rows) || !sized(bias, experts)) return 0;
    const uint64_t total = (uint64_t)experts * (uint64_t)rows;
    struct route_args args = {(uint32_t)total, top_k, 0, scale, rows, experts, experts / 4u};
    const uint32_t tgroups = (uint32_t)((total / 4u + 31u) / 32u);

    const myds_metal_buffer_t *softplus[] = {logits, bias, probs};
    args.op = 0;
    if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), softplus, 3,
                  tgroups, 32, 0)) {
        MYDS_LOG_E("V4.1 route transform dispatch failed");
        return 0;
    }
    const myds_metal_buffer_t *sqrt_buffers[] = {probs, bias, probs};
    args.op = 1;
    if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), sqrt_buffers, 3,
                  tgroups, 32, 0)) return 0;
    const myds_metal_buffer_t *add[] = {probs, bias, scores};
    args.op = 2;
    if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), add, 3,
                  tgroups, 32, 0)) {
        MYDS_LOG_E("V4.1 route bias dispatch failed");
        return 0;
    }

    /* One row per threadgroup: the kernel already indexes rows by tgpig[0] % ne01. */
    const uint64_t row_bytes = (uint64_t)experts * sizeof(float);
    struct sort_args sort = {.ne00=(int32_t)experts,.ne01=(int32_t)rows,.ne02=1,.ne03=1,
        .nb00=4,.nb01=row_bytes,.nb02=row_bytes,.nb03=row_bytes,.ne0=(int32_t)top_k,
        .ne1=1,.ne2=1,.ne3=1,.top_k=(int32_t)top_k};
    const myds_metal_buffer_t *sort_buffers[] = {scores, selected};
    if (!dispatch("kernel_argsort_f32_i32_desc", &sort, sizeof(sort), sort_buffers, 2,
                  rows, 512, 512 * (sizeof(int32_t) + sizeof(float)))) {
        MYDS_LOG_E("V4.1 route top-k dispatch failed");
        return 0;
    }
    const myds_metal_buffer_t *sum[] = {probs, selected, weights, scores};
    const myds_metal_buffer_t *divide[] = {weights, scores};
    const myds_metal_buffer_t *multiply[] = {weights};
    return dispatch("kernel_myds_v41_route_sum", &args, sizeof(args), sum, 4, rows, top_k, 128) &&
        dispatch("kernel_myds_v41_route_divide", &args, sizeof(args), divide, 2, rows, top_k, 0) &&
        dispatch("kernel_myds_v41_route_scale", &args, sizeof(args), multiply, 1, rows, top_k, 0);
}

int myds_v41_route(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probs, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t experts, uint32_t top_k, float scale) {
    return myds_v41_route_batch(selected, weights, probs, scores, logits, bias,
                                experts, top_k, scale, 1);
}

int myds_v41_shared_swiglu(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *gate, const myds_metal_buffer_t *up,
    uint32_t width, float clamp) {
    if (!width || width > INT32_MAX || !isfinite(clamp) || clamp < 0 ||
        !sized(out,width) || !sized(gate,width) || !sized(up,width)) return 0;
    struct glu_args args = {.ne00=(int32_t)width,.nb01=(uint64_t)width*4,
        .ne10=(int32_t)width,.nb11=(uint64_t)width*4,.ne0=(int32_t)width,
        .nb1=(uint64_t)width*4,.alpha=1,.limit=clamp,.round_bf16=1};
    const myds_metal_buffer_t *buffers[] = {gate,up,out};
    return dispatch("kernel_swiglu_flat_f32", &args, sizeof(args), buffers, 3,
                    (width+255)/256,256,0);
}

int myds_v41_moe_eval(myds_context_t *ctx, uint32_t layer, uint32_t pos,
                      const myds_metal_buffer_t *ffn_norm,
                      const myds_metal_buffer_t *router_logits,
                      const myds_metal_buffer_t *shared_out) {
    if (!ctx || !ctx->model || layer >= ctx->model->n_layers) return 0;
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim=m->dim, width=m->moe_expert_dim, k=m->n_activated_experts;
    if (m->n_routed_experts!=384 || k!=6 || m->n_shared_experts!=1 ||
        !dim || !width || dim%256 || width%256 ||
        l->ffn_gate_exps_type!=GGML_TYPE_IQ2_XXS ||
        l->ffn_up_exps_type!=GGML_TYPE_IQ2_XXS ||
        l->ffn_down_exps_type!=GGML_TYPE_Q2_K) return 0;

    if (!myds_trace_f32("route_logits",layer,pos,router_logits,m->n_routed_experts) ||
        !myds_v41_route(ctx->selected_ids,ctx->selected_weights,ctx->moe_gate,ctx->moe_up,
                        router_logits,l->router_bias,m->n_routed_experts,k,m->routed_scaling_factor) ||
        !myds_trace_f32("route_probs",layer,pos,ctx->moe_gate,m->n_routed_experts) ||
        !myds_trace_f32("route_weights",layer,pos,ctx->selected_weights,k)) return 0;
    if (myds_trace_active()) {
        const myds_metal_buffer_t *id_buffers[] = {ctx->selected_ids,ctx->moe_up};
        if (!dispatch("kernel_myds_v41_ids_f32",&k,sizeof(k),id_buffers,2,1,32,0) ||
            !myds_trace_f32("route_ids",layer,pos,ctx->moe_up,k)) return 0;
    }
    if (!myds_trace_f32("shared_out",layer,pos,shared_out,dim)) return 0;
    myds_metal_barrier();
    if (!myds_metal_routed_moe(ctx->moe_out,ffn_norm,l->ffn_gate_exps,
        l->ffn_up_exps,l->ffn_down_exps,ctx->selected_ids,ctx->selected_weights,
        ctx->moe_gate,ctx->moe_up,ctx->moe_mid,ctx->moe_down,dim,width,
        m->n_routed_experts,k,l->ffn_gate_exps_type,l->ffn_up_exps_type,
        l->ffn_down_exps_type,m->swiglu_clamp_exp)) return 0;
    myds_metal_barrier();
    if (!myds_trace_f32("routed_out",layer,pos,ctx->moe_out,dim) ||
        !myds_metal_add(ctx->ffn_out,ctx->moe_out,shared_out,dim)) return 0;
    myds_metal_barrier();
    return myds_metal_v41_bf16(ctx->ffn_out,dim) &&
        myds_trace_f32("ffnblock",layer,pos,ctx->ffn_out,dim);
}

int myds_v41_moe(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos) {
    (void)token; // V4.1 text router is score-based, without token-hash selection.
    if (!ctx || !ctx->model || layer >= ctx->model->n_layers) return 0;
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim=m->dim, width=m->moe_expert_dim;

    if (!myds_metal_v41_gemv(ctx->router_logits,l->router_gate,ctx->ffn_norm_out,
                             m->n_routed_experts,dim,l->router_gate_type)) return 0;
    if (!myds_metal_gemv(ctx->shared_gate_out,l->shared_gate,ctx->ffn_norm_out,
                         width,dim,l->shared_gate_type) ||
        !myds_metal_v41_bf16(ctx->shared_gate_out,width) ||
        !myds_metal_gemv(ctx->shared_up_out,l->shared_up,ctx->ffn_norm_out,
                         width,dim,l->shared_up_type) ||
        !myds_metal_v41_bf16(ctx->shared_up_out,width) ||
        !myds_v41_shared_swiglu(ctx->shared_swiglu_out,ctx->shared_gate_out,
                               ctx->shared_up_out,width,m->swiglu_clamp_exp) ||
        !myds_metal_gemv(ctx->shared_out,l->shared_down,ctx->shared_swiglu_out,
                         dim,width,l->shared_down_type) ||
        !myds_metal_v41_bf16(ctx->shared_out,dim)) return 0;

    return myds_v41_moe_eval(ctx, layer, pos, ctx->ffn_norm_out, ctx->router_logits, ctx->shared_out);
}
