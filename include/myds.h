#ifndef MYDS_H
#define MYDS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture-family recognition only, NOT proof of graph completeness.
 * myds_model_load additionally rejects required operations not implemented by
 * this native engine (HC, Engram, compressed KV and indexer). */
static inline bool myds_is_supported_arch(const char *arch) {
    if (!arch) return false;
    /* Explicitly reject legacy DeepSeek V2 and V3 models */
    if (strcmp(arch, "deepseek2") == 0 || strcmp(arch, "deepseek3") == 0 ||
        strcmp(arch, "deepseek") == 0 || strcmp(arch, "deepseek-v2") == 0 ||
        strcmp(arch, "deepseek-v3") == 0) {
        return false;
    }
    /* Exclusively accept DeepSeek V4 / V4.1 architecture variants */
    if (strncmp(arch, "deepseek4", 9) == 0 ||
        strcmp(arch, "deepseek-v4") == 0 ||
        strcmp(arch, "deepseek-v4.1") == 0) {
        return true;
    }
    return false;
}

typedef struct myds_model myds_model_t;
typedef struct myds_context myds_context_t;

typedef struct {
    uint32_t max_seq_len;       /* Maximum context window (default: 4096) */
    uint32_t max_new_tokens;    /* Maximum generated tokens (default: 256) */
    float    temperature;       /* Sampling temperature (0.0 = greedy argmax) */
    float    top_p;             /* Nucleus top-p sampling (default: 0.9) */
    uint32_t top_k;             /* Top-k sampling (0 = disabled, default: 40) */
    float    min_p;             /* Min-p sampling (0.0 = disabled, default: 0.0) */
    float    repetition_penalty; /* Repetition penalty (1.0 = disabled, default: 1.0) */
    /* Presence penalty（2026-10-24）：token 在近期上下文出现过就【减去】该值，
     * 与出现的次数无关（按次数加权的是 frequency penalty，本实现不提供）。
     * 0.0 = 关闭；正值抑制重复，负值鼓励重复。OpenAI 的取值范围是 [-2, 2]。 */
    float    presence_penalty;   /* Presence penalty (0.0 = disabled, default: 0.0) */
    int32_t  seed;              /* Random seed */
    /* ⚠️ 2026-10-24 审计：本字段全库【只写不读】—— main.c 初始化、engine 复位，
     * 没有任何读取点。流式与否实际由调用方传不传回调（myds_generate_bytes_ex）
     * 决定，与服务端请求体的 "stream" 也无关（那是 myds_server.c 自己解析的）。
     * 保留仅为 ABI 兼容；新代码不要依赖它。 */
    bool     stream;            /* Deprecated: set but never read; see note above. */
    bool     thinking_mode;     /* Enable thinking / reasoning mode (default: false) */
    /* 2026-10-25：结构化输出的 bitmask 【不放在这里】。
     *
     * 原设计在 myds_config_t 里放了 const uint32_t *logit_bitmask，与
     * myds_context_t.logit_bitmask 形成双重所有权。配合 server 的
     *     myds_config_t prev_cfg = myds_context_get_config(ctx);
     *     myds_config_t req_cfg  = prev_cfg;
     * 就会把同一个指针复制进 ctx->cfg —— clear 释放 ctx->logit_bitmask 之后，
     * ctx->cfg.logit_bitmask 立刻悬空，下一次请求 use-after-free。
     *
     * 改为单一所有权：只有 ctx 拥有 bitmask，通过
     * myds_context_set_logit_bitmask / clear_logit_bitmask 管理。 */
} myds_config_t;

typedef void (*myds_stream_callback_t)(const char *piece, int32_t token_id, void *user_data);
/* Token fragments are bytes: they may contain NUL or an incomplete UTF-8
 * character. Text transports must assemble UTF-8 across callbacks. */
typedef void (*myds_stream_bytes_callback_t)(const char *piece, size_t length,
                                            int32_t token_id, void *user_data);

/* Model lifecycle */
myds_model_t   *myds_model_load(const char *model_path);
void            myds_model_free(myds_model_t *model);
void            myds_model_print_info(const myds_model_t *model);

/* Inference context lifecycle */
myds_context_t *myds_context_create(myds_model_t *model, const myds_config_t *cfg);
void            myds_context_free(myds_context_t *ctx);
myds_config_t   myds_context_get_config(const myds_context_t *ctx);
void            myds_context_set_config(myds_context_t *ctx, const myds_config_t *cfg);
void            myds_context_set_logit_bitmask(myds_context_t *ctx, const uint32_t *bitmask);
void            myds_context_clear_logit_bitmask(myds_context_t *ctx);
struct myds_steer;
void            myds_context_set_steer(myds_context_t *ctx, struct myds_steer *steer);
struct myds_steer *myds_context_get_steer(const myds_context_t *ctx);

/* 最近一次请求的 prefill 统计（供服务端日志/工具使用）。
 * 「prefill 速度」这个词有歧义：这里 prefill_tokens 是【真正被 prefilled 的 token 数】，
 * 前缀缓存复用的部分计入 cached_tokens 而不算进 prefill —— 否则「复用越多越慢」会被误读。 */
typedef struct {
    uint32_t prompt_tokens;    /* 本次请求的提示词总长 */
    uint32_t cached_tokens;    /* 前缀缓存复用掉的 token 数（0 = 全量 prefill） */
    uint32_t prefill_tokens;   /* 实际跑了 prefill 的 token 数 */
    double   prefill_seconds;  /* prefill 阶段墙钟耗时 */
    /* ★ 这份统计【是不是本请求产生的】。
     * 2026-10-25：这四个字段存在 context 上，而 context 跨请求复用。
     * 如果某个请求在分词阶段就退出（客户端断连 / 参数非法），从没写过这些字段，
     * 那么调用方读到的是【上一个请求】的数 —— 日志里就会同时出现
     * 「prompt: 0 tok」和「prefill: FAILED after 15815 tok」这种自相矛盾的行，
     * 而那 15815 是上一个请求的。
     * 0 = 本请求没有产出过 prefill 统计（尚未开始或从未开始）。 */
    int      valid;
} myds_prefill_stats_t;

/* 读取最近一次生成（myds_generate*）的 prefill 统计；ctx 为 NULL 时清零返回。 */
void            myds_last_prefill_stats(const myds_context_t *ctx, myds_prefill_stats_t *out);
/* 显式作废上一次的 prefill 统计。
 *
 * 为什么服务端必须调它：调用方可能【根本不进入】myds_generate*。例如 SSE 首写
 * 失败时，服务端写的是 `!cancel && myds_generate(...)`，短路后 generate 不执行，
 * 引擎内部的入口重置也不会跑 ⇒ 统计仍指着上一个请求。那时打日志就会把上次的
 * token 数当成本次报出去（2026-10-25 定位的那条
 * “prompt: 0 tok, prefill: FAILED after 15815 tok”）。
 * 调用时机：拿到推理名额之后、任何可能短路调用 generate 的分支之前。 */
void            myds_context_reset_prefill_stats(myds_context_t *ctx);

/* Per-request decode counters. Calls include attempted single-row and batch-2
 * target invocations; rows include attempted work and rejected future rows.
 * output_tokens counts non-EOS token fragments delivered by generation.
 * Timings use a monotonic host clock, not GPU kernel counters. They are only
 * collected with MYDS_DECODE_METRICS=1 or the opt-in cost scheduler. */
typedef struct {
    uint64_t target_forward_calls, target_rows;
    uint64_t batch2_calls, batch2_hits, batch2_misses, batch2_failures, batch2_fallbacks;
    /* Continuous [seed,draft1,draft2] attempts are also included in batch2
     * compatibility totals. hits1/hits2 consume two/three target rows. */
    uint64_t depth2_calls, depth2_hits1, depth2_hits2, depth2_misses, depth2_failures, depth2_fallbacks;
    uint64_t draft_cycles, draft_proposed, draft_accepted, draft_rejects;
    uint64_t confidence_rejects, margin_rejects, backoff_steps;
    uint64_t pld_queries, pld_cycles, pld_proposed, pld_accepted;
    uint64_t output_tokens, cycle_count;
    double draft_ms, verify_ms, rollback_ms, snapshot_ms, pld_ms;
    double batch2_prep_ms, batch2_encode_ms, batch2_gpu_ms, batch2_arb_ms;
    bool timing_enabled;
    uint32_t recorded_cycles;
} myds_decode_stats_t;

typedef struct {
    double elapsed_ms;       /* Full scheduler cycle, including draft and callbacks. */
    uint32_t output_tokens;
} myds_decode_cycle_t;

void            myds_last_decode_stats(const myds_context_t *ctx, myds_decode_stats_t *out);
/* Cycle records are opt-in (MYDS_DECODE_METRICS=1). False means no such record,
 * including a timing allocation failure; aggregate counters remain available. */
bool            myds_last_decode_cycle(const myds_context_t *ctx, uint32_t index,
                                      myds_decode_cycle_t *out);

/* Generation termination reasons */
typedef enum {
    MYDS_STOP_EOS = 0,
    MYDS_STOP_LENGTH,
    MYDS_STOP_CTX_FULL,
    MYDS_STOP_ERROR,
    MYDS_STOP_CANCELLED
} myds_stop_reason_t;

/* Generation */
/* previous_response_id：置位后，下一次 generate 把引擎缓存的序列当作 prompt 前缀，
 * 调用方传入的 prompt 只作为【新增后缀】。一次性；引擎消费后自动清除。 */
void            myds_set_continue_from_cache(myds_context_t *ctx, bool enable);
uint32_t        myds_context_get_pcache_len(const myds_context_t *ctx);

int             myds_generate_bytes_ex(myds_context_t *ctx,
                                       const char *prompt,
                                       myds_stream_bytes_callback_t cb,
                                       void *user_data,
                                       myds_stop_reason_t *out_reason,
                                       uint32_t *out_prompt_tokens,
                                       volatile int *cancel_flag);

/* Legacy C-string callback: use myds_generate_bytes_ex for lossless output. */
int             myds_generate_ex(myds_context_t *ctx,
                                 const char     *prompt,
                                 myds_stream_callback_t cb,
                                 void           *user_data,
                                 myds_stop_reason_t *out_reason,
                                 uint32_t       *out_prompt_tokens,
                                 volatile int   *cancel_flag);

int             myds_generate(myds_context_t *ctx,
                              const char     *prompt,
                              myds_stream_callback_t cb,
                              void           *user_data);

/* Single token forward step */
int32_t         myds_forward_token(myds_context_t *ctx, int32_t token_id, uint32_t pos);

/* ── 视觉：图片嵌入注入 ──
 * 把提示词里位置 [start_pos, start_pos + count) 的嵌入替换为 data 里的向量
 * （count 个、每个 dim 长）。用于图片块：那些位置共用同一个 image token id，
 * 但每格的向量不同（编码器输出 / 行尾哨兵 / 首尾哨兵）。
 *
 * ★ 可以调用【多次】来添加多个区间 —— 一次请求的历史里可能有多张图，
 *   每个图片块是一个独立区间。（第一版只支持一个区间，导致「发过图之后
 *   的任何对话」都被拒：网关每轮重发完整历史，历史里不止一张图。）
 * 添加时【拷贝】数据，调用方不必管生命周期；clear / 下一次生成结束时统一释放。 */
void            myds_context_add_image_embeddings(myds_context_t *ctx, uint32_t start_pos,
                                                  uint32_t count, const float *data, uint32_t dim,
                                                  uint32_t image_token_id);
void            myds_context_clear_image_embeddings(myds_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_H */
