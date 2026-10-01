/*
 * myds_visfwd.c — 视觉编码器前向。流程与舍入位置见 metal/visenc.metal 的注释。
 */
#include "myds_visfwd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VIS_PATCH_AREA (14u * 14u)
#define VIS_PATCH_IN   (3u * VIS_PATCH_AREA)      /* 588 */
#define VIS_WIDTH      1024u
#define VIS_QKV        3072u
#define VIS_FFN        2816u
#define VIS_FFN2       5632u                       /* 2 * ffn，SwiGLU 拼接 */
#define VIS_ALIGN_IN   9216u                       /* 3*3*1024 */

static void fwd_error(char *error, size_t cap, const char *msg) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", msg);
}

/* 出错的阶段名。派发失败时报「哪一步」，比只报「失败了」有用得多 ——
 * ★ 这里第一条失败是「内核清单没登记 visenc.metal」，表象是内核找不到，
 *   若没有阶段名很容易误判成内核写错。 */
static const char *g_stage = "?";
#define STAGE(name) do { g_stage = (name); } while (0)

/* ── 中间量转储（对拍用）── */
static const char *g_dump_prefix = NULL;

void myds_vis_fwd_set_dump(const char *prefix) { g_dump_prefix = prefix; }
int myds_vis_fwd_dump_enabled(void) { return g_dump_prefix && *g_dump_prefix; }

/* ★ 转储必须在【提交之后】读，否则读到的是陈旧数据（缓冲区还没被 GPU 写过）。
 *   第一版把 dump 夹在批处理中间，结果所有中间量都是 0 —— 我一度以为「矩阵乘没跑」，
 *   其实是没有同步就取数。这跟 AGENTS 里那条「CPU 直接读 GPU 正在写的 buffer = 未同步」
 *   是同一个坑，我换了个场景又踩了一次。 */
static void dump_stage(const char *name, const myds_metal_buffer_t *buf, size_t count) {
    if (!g_dump_prefix || !*g_dump_prefix) return;
    /* 先收口当前批处理，读到的才是真值；转储只在开发期开启，这点开销无所谓 */
    if (!myds_metal_commit_batch()) return;
    if (!myds_metal_begin_batch()) return;
    char path[512];
    snprintf(path, sizeof(path), "%s.%s.f32", g_dump_prefix, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    float *tmp = malloc(count * sizeof(float));
    if (!tmp) { fclose(f); return; }
    if (myds_metal_buffer_read(buf, 0, tmp, count * sizeof(float))) {
        fwrite(tmp, sizeof(float), count, f);
    }
    free(tmp);
    fclose(f);
}

/* 一维行式内核的派发：threads=(min(W,256),1,1)，groups=(ceil(W/tx),rows,1) */
static int dispatch_rows(const char *kernel, uint32_t width, uint32_t rows,
                         myds_metal_buffer_t *const *bufs, size_t n,
                         float eps, int with_eps) {
    struct { uint32_t width, rows; float eps; } args;
    args.width = width;
    args.rows = rows;
    args.eps = with_eps ? eps : 0.0f;
    const uint32_t tx = width < 256u ? width : 256u;
    const uint32_t groups[3] = { (width + tx - 1u) / tx, rows, 1u };
    const uint32_t threads[3] = { tx, 1u, 1u };
    return myds_metal_dispatch_kernel(kernel, &args, sizeof(args),
                                      (const myds_metal_buffer_t *const *)bufs, n,
                                      groups, threads, 0);
}

/* 按行批处理的 RMSNorm（含 BF16 舍入）。★ 不能用引擎的单行 rmsnorm：
 *   那个只处理第 0 行 —— 见 metal/visenc.metal 的说明。 */
static int dispatch_rmsnorm(const char *k, myds_metal_buffer_t *out,
                            myds_metal_buffer_t *x, myds_metal_buffer_t *weight,
                            uint32_t width, uint32_t rows, float eps) {
    struct { uint32_t width, rows; float eps; } args = { width, rows, eps };
    const myds_metal_buffer_t *b[3];
    myds_metal_buffer_t *bs[3] = { x, weight, out };
    b[0] = bs[0]; b[1] = bs[1]; b[2] = bs[2];
    const uint32_t groups[3] = { rows, 1u, 1u };
    const uint32_t threads[3] = { 256u, 1u, 1u };
    return myds_metal_dispatch_kernel(k, &args, sizeof(args),
                                      (const myds_metal_buffer_t *const *)bs, 3,
                                      groups, threads, 8u * sizeof(float));
}

int myds_vis_encode(const myds_vis_weights *w, const myds_vis_patches *patches,
                    float *out, uint32_t *out_rows, char *error, size_t error_cap) {
    if (!w || !patches || !patches->patches || !out) {
        fwd_error(error, error_cap, "invalid vision encode input");
        return 0;
    }
    const uint32_t gh = patches->grid_height, gw = patches->grid_width;
    const uint32_t rows = gh * gw;
    const uint32_t P = w->spec.projection;
    const float eps = w->spec.rms_eps;
    if (rows == 0 || P == 0 || gh == 0 || gw == 0) {
        fwd_error(error, error_cap, "invalid vision grid");
        return 0;
    }
    if (w->spec.patch != 14u || w->spec.block_count == 0) {
        fwd_error(error, error_cap, "vision weights are not usable");
        return 0;
    }

    /* 下采样后每个 3x3 块出一行；行数 = ceil(gh/3) * ceil(gw/3) */
    const uint32_t out_w = (gw + 2u) / 3u;
    const uint32_t aligned_rows = ((gh + 2u) / 3u) * out_w;

    myds_metal_buffer_t *patch = NULL, *qa = NULL, *qb = NULL;
    myds_metal_buffer_t *qkv = NULL, *q = NULL, *k = NULL, *v = NULL, *attn = NULL;
    myds_metal_buffer_t *w1 = NULL, *mid = NULL;
    myds_metal_buffer_t *ain = NULL, *aa = NULL, *ab = NULL;
    int ok = 0;

#define ALLOC(name, bytes)                                                    \
    do {                                                                      \
        name = myds_metal_buffer_alloc(bytes);                                \
        if (!name) { fwd_error(error, error_cap, "vision scratch alloc failed"); goto done; } \
    } while (0)
    ALLOC(patch, (size_t)rows * VIS_PATCH_IN * sizeof(float));
    ALLOC(qa,    (size_t)rows * VIS_WIDTH * sizeof(float));
    ALLOC(qb,    (size_t)rows * VIS_WIDTH * sizeof(float));
    ALLOC(qkv,   (size_t)rows * VIS_QKV * sizeof(float));
    ALLOC(q,     (size_t)rows * VIS_WIDTH * sizeof(float));
    ALLOC(k,     (size_t)rows * VIS_WIDTH * sizeof(float));
    ALLOC(v,     (size_t)rows * VIS_WIDTH * sizeof(float));
    ALLOC(attn,  (size_t)rows * VIS_WIDTH * sizeof(float));
    ALLOC(w1,    (size_t)rows * VIS_FFN2 * sizeof(float));
    ALLOC(mid,   (size_t)rows * VIS_FFN * sizeof(float));
    ALLOC(ain,   (size_t)aligned_rows * VIS_ALIGN_IN * sizeof(float));
    ALLOC(aa,    (size_t)aligned_rows * P * sizeof(float));
    ALLOC(ab,    (size_t)aligned_rows * P * sizeof(float));
#undef ALLOC

    if (!myds_metal_buffer_write(patch, 0, patches->patches,
                                 (size_t)rows * VIS_PATCH_IN * sizeof(float))) {
        fwd_error(error, error_cap, "cannot upload vision patches");
        goto done;
    }
    if (!myds_metal_begin_batch()) {
        fwd_error(error, error_cap, "cannot begin vision batch");
        goto done;
    }

    /* ── patch 嵌入：输入先舍 BF16（V4.1），矩阵乘，加偏置，再舍 ── */
    STAGE("round(patch)");
    ok = dispatch_rows("kernel_visenc_round", VIS_PATCH_IN, rows, &patch, 1, 0.0f, 0);
    if (ok) STAGE("patch_embed gemm");
    if (ok) ok = myds_metal_mul_mm(qa, w->patch_weight, patch,
                                   VIS_WIDTH, VIS_PATCH_IN, rows, GGML_TYPE_F32);
    if (ok) {
        myds_metal_buffer_t *b[2] = { qa, w->patch_bias };
        STAGE("patch bias");
        ok = dispatch_rows("kernel_visenc_bias", VIS_WIDTH, rows, b, 2, 0.0f, 0);
    }
    if (ok) ok = dispatch_rows("kernel_visenc_round", VIS_WIDTH, rows, &qa, 1, 0.0f, 0);
    if (ok) dump_stage("patch_embed", qa, (size_t)rows * VIS_WIDTH);

    myds_metal_buffer_t *cur = qa, *tmp = qb;

    for (uint32_t il = 0; ok && il < w->spec.block_count; il++) {
        const myds_vis_layer_weights *L = &w->layer[il];

        /* --- 注意力分支 --- */
        STAGE("rmsnorm norm1");
        ok = dispatch_rmsnorm("kernel_visenc_rmsnorm", tmp, cur, L->norm1,
                              VIS_WIDTH, rows, eps);
        if (ok) STAGE("qkv gemm");
        if (ok && il == 0u) dump_stage("b0_norm1", tmp, (size_t)rows * VIS_WIDTH);
        if (ok) ok = myds_metal_mul_mm(qkv, L->qkv_weight, tmp,
                                       VIS_QKV, VIS_WIDTH, rows, GGML_TYPE_F32);
        if (ok) {
            struct { uint32_t rows, grid_width; } qa_args = { rows, gw };
            myds_metal_buffer_t *bufs[5] = { qkv, L->qkv_bias, q, k, v };
            const uint32_t groups[3] = { rows, 16u, 1u };
            const uint32_t threads[3] = { 32u, 1u, 1u };
            STAGE("qkv_rope");
            ok = myds_metal_dispatch_kernel("kernel_visenc_qkv_rope", &qa_args,
                                            sizeof(qa_args),
                                            (const myds_metal_buffer_t *const *)bufs, 5,
                                            groups, threads, 0);
        }
        if (ok) {
            struct { uint32_t rows; float scale; } at_args = { rows, 0.125f };
            myds_metal_buffer_t *bufs[4] = { q, k, v, attn };
            const uint32_t groups[3] = { rows, 16u, 1u };
            const uint32_t threads[3] = { 32u, 1u, 1u };
            STAGE("attention");
            ok = myds_metal_dispatch_kernel("kernel_visenc_attention", &at_args,
                                            sizeof(at_args),
                                            (const myds_metal_buffer_t *const *)bufs, 4,
                                            groups, threads, 0);
        }
        if (ok) ok = dispatch_rows("kernel_visenc_round", VIS_WIDTH, rows, &attn, 1, 0.0f, 0);
        if (ok && il == 0u) dump_stage("b0_attn", attn, (size_t)rows * VIS_WIDTH);
        if (ok) STAGE("attn wo gemm");
        if (ok) ok = myds_metal_mul_mm(tmp, L->attn_proj_weight, attn,
                                       VIS_WIDTH, VIS_WIDTH, rows, GGML_TYPE_F32);
        if (ok) {
            /* ★ 两次舍入（V4.1 特有） */
            myds_metal_buffer_t *b[3] = { tmp, L->attn_proj_bias, cur };
            STAGE("bias_residual41");
            ok = dispatch_rows("kernel_visenc_bias_residual41", VIS_WIDTH, rows,
                               b, 3, 0.0f, 0);
        }
        { myds_metal_buffer_t *sw = cur; cur = tmp; tmp = sw; }
        if (ok && il == 0u) dump_stage("block0_attn_residual", cur, (size_t)rows * VIS_WIDTH);

        /* --- FFN 分支 --- */
        if (ok) STAGE("rmsnorm norm2");
        if (ok) ok = dispatch_rmsnorm("kernel_visenc_rmsnorm", tmp, cur, L->norm2,
                                      VIS_WIDTH, rows, eps);
        if (ok) STAGE("mlp w1 gemm");
        if (ok) ok = myds_metal_mul_mm(w1, L->mlp_w1, tmp,
                                       VIS_FFN2, VIS_WIDTH, rows, GGML_TYPE_F32);
        if (ok) ok = dispatch_rows("kernel_visenc_round", VIS_FFN2, rows, &w1, 1, 0.0f, 0);
        /* swiglu_split 读 w1、写 mid ⇒ 两个缓冲同时在场，单独派发 */
        if (ok) {
            myds_metal_buffer_t *b[2] = { w1, mid };
            const uint32_t tx = VIS_FFN < 256u ? VIS_FFN : 256u;
            const uint32_t groups[3] = { (VIS_FFN + tx - 1u) / tx, rows, 1u };
            const uint32_t threads[3] = { tx, 1u, 1u };
            struct { uint32_t width, rows; } args = { VIS_FFN, rows };
            STAGE("swiglu_split41");
            ok = myds_metal_dispatch_kernel("kernel_visenc_swiglu_split41", &args,
                                            sizeof(args),
                                            (const myds_metal_buffer_t *const *)b, 2,
                                            groups, threads, 0);
        }
        if (ok && il == 0u) dump_stage("b0_wo", tmp, (size_t)rows * VIS_WIDTH);
        if (ok && il == 0u) dump_stage("b0_mlp_mid", mid, (size_t)rows * VIS_FFN);
        if (ok) STAGE("mlp w2 gemm");
        if (ok) ok = myds_metal_mul_mm(tmp, L->mlp_w2, mid,
                                       VIS_WIDTH, VIS_FFN, rows, GGML_TYPE_F32);
        if (ok) ok = dispatch_rows("kernel_visenc_round", VIS_WIDTH, rows, &tmp, 1, 0.0f, 0);
        if (ok) {
            myds_metal_buffer_t *b[2] = { tmp, cur };
            STAGE("add_residual");
            ok = dispatch_rows("kernel_visenc_add_residual", VIS_WIDTH, rows, b, 2, 0.0f, 0);
        }
        { myds_metal_buffer_t *sw = cur; cur = tmp; tmp = sw; }
        if (ok && g_dump_prefix && il <= 6u) {
            char stage[24];
            snprintf(stage, sizeof(stage), "block%u", il);
            dump_stage(stage, cur, (size_t)rows * VIS_WIDTH);
        }
    }

    /* ── 收尾：最终 norm → 下采样重排 → aligner 两层 ── */
    if (ok) STAGE("rmsnorm post_norm");
    if (ok) ok = dispatch_rmsnorm("kernel_visenc_rmsnorm", tmp, cur, w->post_norm,
                                  VIS_WIDTH, rows, eps);
    if (ok) dump_stage("vision_norm", tmp, (size_t)rows * VIS_WIDTH);

    if (ok) {
        struct { uint32_t grid_height, grid_width, output_rows; } aargs = { gh, gw, aligned_rows };
        myds_metal_buffer_t *b[2] = { tmp, ain };
        const uint32_t tx = VIS_ALIGN_IN < 256u ? VIS_ALIGN_IN : 256u;
        const uint32_t groups[3] = { (VIS_ALIGN_IN + tx - 1u) / tx, aligned_rows, 1u };
        const uint32_t threads[3] = { tx, 1u, 1u };
        STAGE("aligner_reorder");
        ok = myds_metal_dispatch_kernel("kernel_visenc_aligner_reorder", &aargs,
                                        sizeof(aargs),
                                        (const myds_metal_buffer_t *const *)b, 2,
                                        groups, threads, 0);
    }
    if (ok) STAGE("aligner w1 gemm");
    if (ok) ok = myds_metal_mul_mm(aa, w->aligner_w1, ain,
                                   P, VIS_ALIGN_IN, aligned_rows, GGML_TYPE_F32);
    if (ok) {
        myds_metal_buffer_t *b[3] = { aa, w->aligner_w1_bias, ab };
        STAGE("gelu_bias");
        ok = dispatch_rows("kernel_visenc_gelu_bias", P, aligned_rows, b, 3, 0.0f, 0);
    }
    if (ok) STAGE("aligner w2 gemm");
    if (ok) ok = myds_metal_mul_mm(aa, w->aligner_w2, ab,
                                   P, P, aligned_rows, GGML_TYPE_F32);
    if (ok) {
        myds_metal_buffer_t *b[2] = { aa, w->aligner_w2_bias };
        ok = dispatch_rows("kernel_visenc_bias", P, aligned_rows, b, 2, 0.0f, 0);
    }
    if (ok) ok = dispatch_rows("kernel_visenc_round", P, aligned_rows, &aa, 1, 0.0f, 0);
    if (ok) dump_stage("aligned", aa, (size_t)aligned_rows * P);

    if (ok) STAGE("commit");
    if (ok) ok = myds_metal_commit_batch();
    if (ok) ok = myds_metal_buffer_read(aa, 0, out,
                                        (size_t)aligned_rows * P * sizeof(float));

    if (!ok) {
        if (error && error_cap)
            snprintf(error, error_cap, "vision encoder step failed at: %s", g_stage);
    }

done:
    myds_metal_buffer_free(patch);
    myds_metal_buffer_free(qa);
    myds_metal_buffer_free(qb);
    myds_metal_buffer_free(qkv);
    myds_metal_buffer_free(q);
    myds_metal_buffer_free(k);
    myds_metal_buffer_free(v);
    myds_metal_buffer_free(attn);
    myds_metal_buffer_free(w1);
    myds_metal_buffer_free(mid);
    myds_metal_buffer_free(ain);
    myds_metal_buffer_free(aa);
    myds_metal_buffer_free(ab);
    if (out_rows) *out_rows = aligned_rows;
    return ok;
}
