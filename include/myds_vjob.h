/*
 * myds_vjob.h — 视觉作业：把一张图片变成「提示词里的占位符 + 逐位置的向量」
 *
 * 服务端拿到图片后需要两样东西：
 *   ① 一段文本 —— 插进提示词里，让 tokenizer 把图片块编成 N 个占位 token
 *   ② N 个向量 —— 每个占位 token 位置该用什么向量（词表查不出来）
 * 本模块就产出这两样，服务端不需要知道布局/哨兵/编码器的细节。
 *
 * 用法（服务端）：
 *   if (myds_vjob_init(vision_gguf_path) != 1) → 降级为纯文本
 *   plan = myds_vjob_plan(encoded_bytes, len, ...);   // 编一次图
 *   把 plan->text 插进提示词
 *   prompt 拼好后：token_offset = myds_vjob_token_offset(ctx, prompt, ...)
 *   myds_context_set_image_embeddings(ctx, token_offset, plan->token_count, plan->block, plan->dim);
 */
#ifndef MYDS_VJOB_H
#define MYDS_VJOB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "myds.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 一次图片作业的产物 */
typedef struct {
    uint32_t token_count;   /* 占位 token 个数（= 布局长度） */
    uint32_t dim;           /* 每个向量的宽度（= 语言模型嵌入宽度） */
    float   *block;         /* [token_count * dim] 逐位置向量 */
    char    *text;          /* 占位文本（占位符重复 token_count 次） */
    uint32_t grid_height;   /* 供日志 */
    uint32_t grid_width;
    bool     ok;
} myds_vjob_plan_t;

/* 一次请求的图片槽。服务端把它传进提示词构建函数，
 * 构建过程中遇到 input_image 就填进来（目前只支持一张）。 */
typedef struct {
    myds_vjob_plan_t plan;
    bool             have;
} myds_vjob_sink_t;

/* 加载视觉编码器权重。成功返回 1，失败返回 0（服务端据此降级为纯文本）。 */
int  myds_vjob_init(myds_context_t *ctx, const char *vision_gguf);
bool myds_vjob_ready(void);
/* 占位 token 的文本。V4.1 是 "<｜deepseek_image｜>"。
 * ★ 由【主模型的词表】决定，不是写死的常量 —— 这里在 init 时从词表取，
 *   并验证「该文本编码回来恰好是 1 个 image token」，不满足就拒绝启用。 */
const char *myds_vjob_placeholder(void);
uint32_t    myds_vjob_image_token_id(void);

/* 编一张图。encoded 是原始 JPEG/PNG 字节。 */
int  myds_vjob_plan(const uint8_t *encoded, size_t len, myds_vjob_plan_t *out,
                    char *error, size_t error_cap);
void myds_vjob_plan_free(myds_vjob_plan_t *plan);

/* 在【最终提示词】里找图片块，返回它的起始 token 下标；找不到返回 -1。
 * 做法：定位占位文本的首次出现位置，再把该位置之前的文本编码一次数 token ——
 * 与引擎内部分词的方式（add_bos=true）保持一致。 */
int64_t myds_vjob_token_offset(myds_context_t *ctx, const char *prompt);

#ifdef __cplusplus
}
#endif
#endif /* MYDS_VJOB_H */
