/*
 * myds_vjob.c — 视觉作业（见 include/myds_vjob.h）
 */
#include "myds_vjob.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_internal.h"      /* 取 model->tokenizer */
#include "myds_log.h"
#include "myds_metal.h"
#include "myds_tokenizer.h"
#include "myds_visfwd.h"
#include "myds_vispre.h"
#include "myds_visw.h"

/* 视觉状态。一个进程一套（与引擎上下文同寿命）。 */
static myds_vis_weights *g_w = NULL;
static char             *g_placeholder = NULL;
static uint32_t          g_image_token = 0;

static void vjob_error(char *error, size_t cap, const char *msg) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", msg);
}

int myds_vjob_init(myds_context_t *ctx, const char *vision_gguf) {
    if (g_w) return 1;
    if (!ctx || !vision_gguf || !*vision_gguf) return 0;

    char err[512] = {0};
    if (!myds_vis_weights_load(vision_gguf, &g_w, err, sizeof(err))) {
        MYDS_LOG_E("视觉编码器加载失败: %s", err);
        return 0;
    }
    g_image_token = g_w->spec.image_token_id;
    MYDS_LOG_I("视觉编码器已就绪: %u 层 · projection %u · image token %u · F32 %.2f GiB",
               g_w->spec.block_count, g_w->spec.projection, g_image_token,
               (double)g_w->spec.f32_bytes / 1073741824.0);

    /* ── 取占位 token 的文本，并【验证往返】──
     * 服务端把提示词当字符串交给引擎，所以占位符必须满足：
     * 它的文本编码回来恰好是 1 个 image token。不满足就宁可不用视觉，
     * 也不要让 tokenizer 把图片块编成别的长度 —— 那会让位置对不上、静默出错。 */
    const myds_tokenizer_t *tok = ctx->model ? ctx->model->tokenizer : NULL;
    if (!tok) {
        MYDS_LOG_E("视觉: 拿不到 tokenizer，拒绝启用");
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    const char *piece = myds_tokenizer_decode_piece(tok, (int32_t)g_image_token);
    if (!piece) {
        MYDS_LOG_E("视觉: 词表里没有 token %u 的文本", g_image_token);
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    int32_t back[4] = {0};
    int32_t n = myds_tokenizer_encode(tok, piece, back, 4, false);
    if (n != 1 || back[0] != (int32_t)g_image_token) {
        MYDS_LOG_E("视觉: 占位符往返不一致（%d 个 token），拒绝启用", n);
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    g_placeholder = strdup(piece);
    if (!g_placeholder) {
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    MYDS_LOG_I("视觉占位符: %s（往返一致，1 个 token）", g_placeholder);
    return 1;
}

bool myds_vjob_ready(void) { return g_w != NULL && g_placeholder != NULL; }
const char *myds_vjob_placeholder(void) { return g_placeholder; }
uint32_t myds_vjob_image_token_id(void) { return g_image_token; }

/* 把 F32 缓冲里的哨兵向量读出来（image_start / image_newline / image_end）。
 * 这些是模型学出来的向量，对每个位置都一样，用来标记块的首、行尾、末。 */
static int read_sentinel(const myds_metal_buffer_t *buf, float *out, uint32_t dim) {
    return buf && myds_metal_buffer_read(buf, 0, out, (size_t)dim * sizeof(float));
}

int myds_vjob_plan(const uint8_t *encoded, size_t len, myds_vjob_plan_t *out,
                   char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!myds_vjob_ready()) {
        vjob_error(error, error_cap, "视觉编码器未启用");
        return 0;
    }
    const uint32_t dim = g_w->spec.projection;

    myds_vis_image image;
    if (!myds_vis_image_decode_memory(&image, encoded, len, error, error_cap)) return 0;
    myds_vis_patches pat;
    if (!myds_vis_preprocess(&pat, &image, error, error_cap)) {
        myds_vis_image_free(&image);
        return 0;
    }
    /* ★ 布局用的是【下采样后的 llm 网格】，不是 patch 网格（V4.1 的规矩） */
    myds_vis_layout lay;
    if (!myds_vis_layout_build(&lay, pat.llm_grid_height, pat.llm_grid_width,
                               error, error_cap)) {
        myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }

    float *emb = (float *)malloc((size_t)pat.patch_count * dim * sizeof(float));
    if (!emb) {
        vjob_error(error, error_cap, "图片嵌入分配失败");
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }
    uint32_t rows = 0;
    if (!myds_vis_encode(g_w, &pat, emb, &rows, error, error_cap) || rows != pat.patch_count) {
        if (error && !error[0]) vjob_error(error, error_cap, "视觉编码失败");
        free(emb);
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }

    /* ── 按布局拼出逐位置的向量 ──
     * IMAGE 位置取编码器输出（按布局给的排列序）；三处哨兵取模型学出来的向量。 */
    const uint32_t n = lay.token_count;
    float *block = (float *)malloc((size_t)n * dim * sizeof(float));
    float *sent_start = (float *)malloc((size_t)dim * sizeof(float));
    float *sent_nl    = (float *)malloc((size_t)dim * sizeof(float));
    float *sent_end   = (float *)malloc((size_t)dim * sizeof(float));
    int ok = block && sent_start && sent_nl && sent_end
             && read_sentinel(g_w->image_start, sent_start, dim)
             && read_sentinel(g_w->image_newline, sent_nl, dim)
             && read_sentinel(g_w->image_end, sent_end, dim);
    if (!ok) {
        vjob_error(error, error_cap, "哨兵向量读取失败");
    } else {
        uint32_t img_pos = 0;
        for (uint32_t i = 0; i < n; i++) {
            float *dst = block + (size_t)i * dim;
            switch (lay.types[i]) {
                case MYDS_VIS_TOKEN_IMAGE: {
                    const uint32_t src = lay.perm[img_pos++];
                    memcpy(dst, emb + (size_t)src * dim, (size_t)dim * sizeof(float));
                    break;
                }
                case MYDS_VIS_TOKEN_START:   memcpy(dst, sent_start, (size_t)dim * sizeof(float)); break;
                case MYDS_VIS_TOKEN_NEWLINE: memcpy(dst, sent_nl,    (size_t)dim * sizeof(float)); break;
                case MYDS_VIS_TOKEN_END:     memcpy(dst, sent_end,   (size_t)dim * sizeof(float)); break;
                default:                     memset(dst, 0, (size_t)dim * sizeof(float)); break;
            }
        }
        if (img_pos != lay.image_count) {
            vjob_error(error, error_cap, "内部错误：布局的图片数与编码输出不符");
            ok = 0;
        }
    }
    free(sent_start); free(sent_nl); free(sent_end); free(emb);
    if (!ok) {
        free(block);
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }

    /* 占位文本 = 占位符重复 n 次（连续，构成一块） */
    const size_t plen = strlen(g_placeholder);
    char *text = (char *)malloc((size_t)n * plen + 1);
    if (!text) {
        vjob_error(error, error_cap, "占位文本分配失败");
        free(block);
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }
    for (uint32_t i = 0; i < n; i++) memcpy(text + (size_t)i * plen, g_placeholder, plen);
    text[(size_t)n * plen] = 0;

    out->token_count = n;
    out->dim = dim;
    out->block = block;
    out->text = text;
    out->grid_height = pat.grid_height;
    out->grid_width = pat.grid_width;
    out->ok = true;

    myds_vis_layout_free(&lay);
    myds_vis_patches_free(&pat);
    myds_vis_image_free(&image);
    return 1;
}

void myds_vjob_plan_free(myds_vjob_plan_t *plan) {
    if (!plan) return;
    free(plan->block);
    free(plan->text);
    memset(plan, 0, sizeof(*plan));
}

int64_t myds_vjob_token_offset(myds_context_t *ctx, const char *prompt) {
    if (!ctx || !prompt || !g_placeholder) return -1;
    const char *at = strstr(prompt, g_placeholder);
    if (!at) return -1;
    const myds_tokenizer_t *tok = ctx->model ? ctx->model->tokenizer : NULL;
    if (!tok) return -1;
    /* 编码「占位块之前的那段」，token 数就是图片块的起始下标。
     * ★ add_bos 必须与引擎内部分词时一致（myds_generate_bytes_ex 传 true），
     *   否则会整体差 1 个位置 —— 那会静默地把图片向量贴到错的 token 上。 */
    const size_t prefix_len = (size_t)(at - prompt);
    char *prefix = (char *)malloc(prefix_len + 1);
    if (!prefix) return -1;
    memcpy(prefix, prompt, prefix_len);
    prefix[prefix_len] = 0;
    int32_t count = myds_tokenizer_encode(tok, prefix, NULL, 0, true);
    free(prefix);
    return count < 0 ? -1 : (int64_t)count;
}
