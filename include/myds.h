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
    int32_t  seed;              /* Random seed */
    bool     stream;            /* Stream output tokens to callback */
    bool     thinking_mode;     /* Enable thinking / reasoning mode (default: false) */
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

/* 最近一次请求的 prefill 统计（供服务端日志/工具使用）。
 * 「prefill 速度」这个词有歧义：这里 prefill_tokens 是【真正被 prefilled 的 token 数】，
 * 前缀缓存复用的部分计入 cached_tokens 而不算进 prefill —— 否则「复用越多越慢」会被误读。 */
typedef struct {
    uint32_t prompt_tokens;    /* 本次请求的提示词总长 */
    uint32_t cached_tokens;    /* 前缀缓存复用掉的 token 数（0 = 全量 prefill） */
    uint32_t prefill_tokens;   /* 实际跑了 prefill 的 token 数 */
    double   prefill_seconds;  /* prefill 阶段墙钟耗时 */
} myds_prefill_stats_t;

/* 读取最近一次生成（myds_generate*）的 prefill 统计；ctx 为 NULL 时清零返回。 */
void            myds_last_prefill_stats(const myds_context_t *ctx, myds_prefill_stats_t *out);

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
