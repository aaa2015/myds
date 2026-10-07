#include "myds_v41_moe.h"
#include "myds_v41_decode.h"
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

static inline int get_route_fuse_level(void) {
    const char *v = getenv("MYDS_MOE_ROUTE_FUSE");
    if (!v) return 2; /* 默认启用全部融合 (all-in-one 单 kernel 派发) */
    if (strcmp(v, "0") == 0) return 0; /* 禁用融合 (7 个小 kernel) */
    if (strcmp(v, "1") == 0) return 1; /* 3 个 kernel (prepare + argsort + finish) */
    return atoi(v); /* >= 2: 全融合 */
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

    const int fuse_level = get_route_fuse_level();
    if (fuse_level >= 2) {
        const myds_metal_buffer_t *fused_bufs[] = {logits, bias, probs, scores, selected, weights};
        /* 共享内存: 512 ints + 512 floats + 384 floats + 32 floats = 5760 字节 */
        const size_t shmem_bytes = 512 * sizeof(int32_t) + 512 * sizeof(float) +
                                   experts * sizeof(float) + 32 * sizeof(float);
        if (dispatch("kernel_myds_v41_route_fused", &args, sizeof(args), fused_bufs, 6,
                     rows, 512, shmem_bytes)) {
            const myds_metal_buffer_t *out_bufs[] = {selected, weights};
            myds_metal_barrier_buffers(out_bufs, 2);
            return 1;
        }
        /* 若全融合派发失败，平滑退回到分阶段路径 */
    }

    const uint32_t tgroups = (uint32_t)((total / 4u + 31u) / 32u);

    int fused_prep_ok = 0;
    if (fuse_level >= 1) {
        const myds_metal_buffer_t *prep[] = {logits, bias, probs, scores};
        fused_prep_ok = dispatch("kernel_myds_v41_route_prepare", &args, sizeof(args), prep, 4,
                                 tgroups, 32, 0);
    }

    if (!fused_prep_ok) {
        const myds_metal_buffer_t *softplus[] = {logits, bias, probs};
        args.op = 0;
        if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), softplus, 3,
                      tgroups, 32, 0)) {
            MYDS_LOG_E("V4.1 route transform dispatch failed");
            return 0;
        }
        myds_metal_barrier_buffer(probs);
        const myds_metal_buffer_t *sqrt_buffers[] = {probs, bias, probs};
        args.op = 1;
        if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), sqrt_buffers, 3,
                      tgroups, 32, 0)) return 0;
        myds_metal_barrier_buffer(probs);
        const myds_metal_buffer_t *add[] = {probs, bias, scores};
        args.op = 2;
        if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), add, 3,
                      tgroups, 32, 0)) {
            MYDS_LOG_E("V4.1 route bias dispatch failed");
            return 0;
        }
    }

    myds_metal_barrier_buffer(scores);

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

    const myds_metal_buffer_t *finish_in[] = {probs, selected};
    myds_metal_barrier_buffers(finish_in, 2);

    if (fuse_level >= 1) {
        const myds_metal_buffer_t *finish[] = {probs, selected, weights, scores};
        if (dispatch("kernel_myds_v41_route_finish", &args, sizeof(args), finish, 4, rows, top_k, 128)) {
            const myds_metal_buffer_t *out_bufs[] = {selected, weights};
            myds_metal_barrier_buffers(out_bufs, 2);
            return 1;
        }
    }

    const myds_metal_buffer_t *sum[] = {probs, selected, weights, scores};
    const myds_metal_buffer_t *divide[] = {weights, scores};
    const myds_metal_buffer_t *multiply[] = {weights};
    int ok = dispatch("kernel_myds_v41_route_sum", &args, sizeof(args), sum, 4, rows, top_k, 128);
    const myds_metal_buffer_t *div_in[] = {weights, scores};
    myds_metal_barrier_buffers(div_in, 2);
    ok = ok && dispatch("kernel_myds_v41_route_divide", &args, sizeof(args), divide, 2, rows, top_k, 0);
    myds_metal_barrier_buffer(weights);
    ok = ok && dispatch("kernel_myds_v41_route_scale", &args, sizeof(args), multiply, 1, rows, top_k, 0);
    const myds_metal_buffer_t *out_bufs[] = {selected, weights};
    myds_metal_barrier_buffers(out_bufs, 2);
    return ok;
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
        !myds_metal_v41_add_bf16(ctx->ffn_out,ctx->moe_out,shared_out,dim)) return 0;
    return myds_trace_f32("ffnblock",layer,pos,ctx->ffn_out,dim);
}

static inline bool is_moe_concurrent_enabled(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        const char *v = getenv("MYDS_MOE_CONCURRENT");
        cached = (v && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

int myds_v41_moe(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos) {
    (void)token; // V4.1 text router is score-based, without token-hash selection.
    if (!ctx || !ctx->model || layer >= ctx->model->n_layers) return 0;
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim = m->dim, width = m->moe_expert_dim, k = m->n_activated_experts;
    const bool trace_on = myds_trace_active();

    if (m->n_routed_experts != 384 || k != 6 || m->n_shared_experts != 1 ||
        !dim || !width || dim % 256 || width % 256 ||
        l->ffn_gate_exps_type != GGML_TYPE_IQ2_XXS ||
        l->ffn_up_exps_type != GGML_TYPE_IQ2_XXS ||
        l->ffn_down_exps_type != GGML_TYPE_Q2_K) return 0;

    /* 状态与跟踪门禁：若 trace 开启或显式关闭并发，退回原有严格串行路径 */
    if (trace_on || !is_moe_concurrent_enabled()) {
        /* 1. 优先派发 router_logits GEMV 与路由打分排序 */
        if (!myds_metal_v41_gemv(ctx->router_logits, l->router_gate, ctx->ffn_norm_out,
                                 m->n_routed_experts, dim, l->router_gate_type)) return 0;
        if (trace_on && !myds_trace_f32("route_logits", layer, pos, ctx->router_logits, m->n_routed_experts)) return 0;

        if (!myds_v41_route(ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up,
                            ctx->router_logits, l->router_bias, m->n_routed_experts, k, m->routed_scaling_factor)) return 0;
        if (trace_on) {
            if (!myds_trace_f32("route_probs", layer, pos, ctx->moe_gate, m->n_routed_experts) ||
                !myds_trace_f32("route_weights", layer, pos, ctx->selected_weights, k)) return 0;
            const myds_metal_buffer_t *id_buffers[] = {ctx->selected_ids, ctx->moe_up};
            if (!dispatch("kernel_myds_v41_ids_f32", &k, sizeof(k), id_buffers, 2, 1, 32, 0) ||
                !myds_trace_f32("route_ids", layer, pos, ctx->moe_up, k)) return 0;
        }

        /* 2. 紧接着派发 shared expert (gate, up, swiglu, down) */
        if (!myds_v41_decode_gemv_bf16(ctx->shared_gate_out, l->shared_gate, ctx->ffn_norm_out,
                             width, dim, l->shared_gate_type, trace_on) ||
            !myds_v41_decode_gemv_bf16(ctx->shared_up_out, l->shared_up, ctx->ffn_norm_out,
                             width, dim, l->shared_up_type, trace_on) ||
            !myds_v41_shared_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out,
                                   ctx->shared_up_out, width, m->swiglu_clamp_exp) ||
            !myds_v41_decode_gemv_bf16(ctx->shared_out, l->shared_down, ctx->shared_swiglu_out,
                             dim, width, l->shared_down_type, trace_on)) return 0;
        if (trace_on && !myds_trace_f32("shared_out", layer, pos, ctx->shared_out, dim)) return 0;

        /* 3. 此时路由选择与 shared expert 计算流已就绪，启动 routed MoE */
        myds_metal_barrier();
        if (!myds_metal_routed_moe(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
            l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
            ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
            m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
            l->ffn_down_exps_type, m->swiglu_clamp_exp)) return 0;
        myds_metal_barrier();
        if (trace_on && !myds_trace_f32("routed_out", layer, pos, ctx->moe_out, dim)) return 0;

        /* 4. 融合 shared_out 与 routed_moe */
        if (!myds_metal_v41_add_bf16(ctx->ffn_out, ctx->moe_out, ctx->shared_out, dim)) return 0;
        return !trace_on || myds_trace_f32("ffnblock", layer, pos, ctx->ffn_out, dim);
    }

    /* === S1 并发流水调度分支 === */
    const int prev_auto_barrier = myds_metal_get_auto_barrier();
    const int prev_concurrent = myds_metal_get_concurrent();
    myds_metal_set_auto_barrier(0);
    myds_metal_set_concurrent(1);

    /* 1. 并发派发 Router GEMV 与 Shared Expert Gate/Up GEMVs */
    if (!myds_metal_v41_gemv(ctx->router_logits, l->router_gate, ctx->ffn_norm_out,
                             m->n_routed_experts, dim, l->router_gate_type) ||
        !myds_v41_decode_gemv_bf16(ctx->shared_gate_out, l->shared_gate, ctx->ffn_norm_out,
                                  width, dim, l->shared_gate_type, false) ||
        !myds_v41_decode_gemv_bf16(ctx->shared_up_out, l->shared_up, ctx->ffn_norm_out,
                                  width, dim, l->shared_up_type, false)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 2. 仅同步 router_logits，派发路由打分与 Top-K 选择流水线 */
    myds_metal_barrier_buffer(ctx->router_logits);
    if (!myds_v41_route(ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up,
                        ctx->router_logits, l->router_bias, m->n_routed_experts, k, m->routed_scaling_factor)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 3. 仅同步 shared_gate_out 与 shared_up_out，派发 Shared SwiGLU */
    const myds_metal_buffer_t *shared_gu_deps[] = { ctx->shared_gate_out, ctx->shared_up_out };
    myds_metal_barrier_buffers(shared_gu_deps, 2);
    if (!myds_v41_shared_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out,
                               ctx->shared_up_out, width, m->swiglu_clamp_exp)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 4. 仅同步 shared_swiglu_out，派发 Shared Down GEMV */
    myds_metal_barrier_buffer(ctx->shared_swiglu_out);
    if (!myds_v41_decode_gemv_bf16(ctx->shared_out, l->shared_down, ctx->shared_swiglu_out,
                                  dim, width, l->shared_down_type, false)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 5. 仅同步路由输出与 scratch 释放，启动 Routed MoE（与 Shared Down GEMV 在 GPU 上并发） */
    const myds_metal_buffer_t *moe_deps[] = {
        ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up
    };
    myds_metal_barrier_buffers(moe_deps, 4);
    if (!myds_metal_routed_moe(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
        l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
        m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
        l->ffn_down_exps_type, m->swiglu_clamp_exp)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 6. 等待 Routed MoE (moe_out) 与 Shared Expert (shared_out) 两路汇聚相加 */
    const myds_metal_buffer_t *final_deps[] = { ctx->moe_out, ctx->shared_out };
    myds_metal_barrier_buffers(final_deps, 2);
    if (!myds_metal_v41_add_bf16(ctx->ffn_out, ctx->moe_out, ctx->shared_out, dim)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }
    myds_metal_barrier_buffer(ctx->ffn_out);
    myds_metal_set_auto_barrier(prev_auto_barrier);
    myds_metal_set_concurrent(prev_concurrent);
    return 1;
}
