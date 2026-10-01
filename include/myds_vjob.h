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
 *   prompt 拼好后，对每个块求偏移并注入：
 *     off_i = myds_vjob_token_offset_at(ctx, prompt, 第 i 个块的占位文本起点);
 *     myds_context_add_image_embeddings(ctx, off_i, plans[i].token_count,
 *                                      plans[i].block, plans[i].dim);
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

/* 一次请求的图片槽【列表】。服务端把它传进提示词构建函数，
 * 构建过程中每遇到一个 input_image 就追加一个 plan。
 * ★ 是列表而不是单个：网关每轮重发完整历史，历史里可能有多张图。 */
typedef struct myds_vjob_sink_s {
    myds_vjob_plan_t *plans;
    uint32_t          count;
    uint32_t          cap;
    bool              failed;     /* 某个图片块处理失败 ⇒ 整个请求应当被拒 */
    char              why[192];   /* 失败原因（要回给调用方，不能只写日志） */
} myds_vjob_sink_t;

/* 往 sink 里追加一张图（服务端内部用；失败时置 failed 并返回 false）。 */
int  myds_vjob_sink_add(myds_vjob_sink_t *sink, const uint8_t *encoded, size_t len,
                        char *error, size_t error_cap);
void myds_vjob_sink_free(myds_vjob_sink_t *sink);

/* 加载视觉编码器权重。成功返回 1，失败返回 0（服务端据此降级为纯文本）。 */
int  myds_vjob_init(myds_context_t *ctx, const char *vision_gguf);
bool myds_vjob_ready(void);
void myds_vjob_set_gpu_lock(void (*lock_fn)(void), void (*unlock_fn)(void));
/* 占位 token 的文本。V4.1 是 "<｜deepseek_image｜>"。
 * ★ 由【主模型的词表】决定，不是写死的常量 —— 这里在 init 时从词表取，
 *   并验证「该文本编码回来恰好是 1 个 image token」，不满足就拒绝启用。 */
const char *myds_vjob_placeholder(void);
uint32_t    myds_vjob_image_token_id(void);

/* 编一张图。encoded 是原始 JPEG/PNG 字节。 */
int  myds_vjob_plan(const uint8_t *encoded, size_t len, myds_vjob_plan_t *out,
                    char *error, size_t error_cap);
void myds_vjob_plan_free(myds_vjob_plan_t *plan);

/* 在【最终提示词】里找某个图片块的起始 token 下标；找不到返回 -1。
 * block_at 指向提示词里该块占位文本的起点。
 * 做法：把「该点之前的文本」编码一次数 token —— 与引擎内部分词的方式
 * （add_bos=true）保持一致（差 1 会整体错位）。 */
int64_t myds_vjob_token_offset_at(myds_context_t *ctx, const char *prompt_base,
                                  const char *block_at);

/* 编码缓存：同一张图在历史里反复出现（网关每轮重发）⇒ 按内容哈希复用上一次的结果。
 * 命中时省掉整次编码（实测 ~6s/图）。 */
uint32_t myds_vjob_cache_hits(void);
uint32_t myds_vjob_cache_misses(void);

#ifdef __cplusplus
}
#endif
#endif /* MYDS_VJOB_H */
