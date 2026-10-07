#ifndef MYDS_V41_H
#define MYDS_V41_H

#include "myds.h"
#include "myds_engram.h"
#include <stdbool.h>

typedef struct myds_v41_model myds_v41_model_t;
typedef struct myds_v41_context myds_v41_context_t;

/* Native, text-only V4.1 graph. Kept behind the diagnostic loader until its
 * independently captured stage tensors and logits satisfy the release gate. */
myds_v41_model_t *myds_v41_model_create(myds_model_t *model, const char *path);
void myds_v41_model_free(myds_v41_model_t *model);
myds_v41_context_t *myds_v41_context_create(myds_context_t *context);
void myds_v41_context_free(myds_v41_context_t *context);

/* ── 前缀快照（供服务端调度复用状态）──
 * 窗口环只有 128 槽、事后无法回溯，因此 prefill 经过边界时就把状态存下来。
 * 新请求的公共前缀只要覆盖到某个快照点，就能从那里续跑 —— 覆盖
 * 「客户端每次都当新对话、只带系统提示词」这一类（严格续写之外的）用法。 */
uint32_t myds_v41_snapshot_best(const myds_v41_context_t *context, uint32_t common);
bool     myds_v41_snapshot_restore(myds_v41_context_t *context, uint32_t position);
/* 丢弃位置 > keep_up_to 的快照（本次请求会从那里开始写，更早的点仍然有效）。 */
void     myds_v41_snapshot_clear(myds_v41_context_t *context, uint32_t keep_up_to);
/* prefill 结束后记下当前状态（末尾槽），供下一个共享前缀的请求复用。 */
bool     myds_v41_snapshot_store_now(myds_v41_context_t *context, uint32_t position);

/* ── 低位区 KV 的 SSD 落盘 ──
 * 内存快照只活在【进程内、且只在与上一条序列共享前缀时】可用。跨进程（重启）或
 * 跨会话（客户端每次都当新对话）就没有了，于是那几秒～十几秒的常量前缀 prefill
 * 每次都要重付。这里把【低位区】那批快照落到磁盘，让它们能跨进程复用。
 *
 * 只做低位区（位置 1024..11264，对应系统提示词 + 工具定义那一段），不做环形槽：
 * 文件于是有界，淘汰退化成「一个位置一条，新的覆盖旧的」。
 *
 * 为什么还要存压缩 KV：注意力除了窗口环还要读 compressed/index_cache（跨位置累积的
 * 压缩 KV）。内存快照能省掉它们，是因为「与上一条序列共享前缀」已经保证了那批行仍然
 * 对应当前前缀；跨进程恢复没有这个前提，所以必须一并存。
 *
 * `dir` 为空或 MYDS_KV_DSK=0 则不启用（库的调用方依赖「每次 generate 都是全量」，
 * 这个优化由服务端显式打开）。 */
bool     myds_v41_low_kv_enable(myds_v41_context_t *context, const char *dir);
bool     myds_v41_low_kv_enabled(const myds_v41_context_t *context);
/* 把当前【低位槽】里还需要的快照写盘（内容没变则不重写）。
 * MYDS_KV_DSK_ASYNC=1 时，位置 < n_tokens 的条目交给后台线程写（与 decode 并行）；
 * 下一次 snapshot_clear/restore/store_now、low_kv_load/save、prefill、context_free 会先汇合。 */
bool     myds_v41_low_kv_save(myds_v41_context_t *context, const int32_t *tokens,
                              uint32_t n_tokens);
/* 显式等待后台写盘结束（未启用/无在途写盘时立即返回）。 */
void     myds_v41_low_kv_wait(myds_v41_context_t *context);
/* Metadata-only deepest-prefix query. Joins any pending save, but does not
 * read payloads, change hit counts, or change live/snapshot state. A probe
 * hit remains tentative until load validates its complete payload. */
uint32_t myds_v41_low_kv_probe(myds_v41_context_t *context, const int32_t *tokens,
                               uint32_t max_position);
/* Probe only positions strictly above an already selected RAM prefix. This
 * avoids scanning ways/hashing prefixes that cannot improve the RAM hit. */
uint32_t myds_v41_low_kv_probe_deeper(myds_v41_context_t *context, const int32_t *tokens,
                                     uint32_t memory_position, uint32_t max_position);
/* 在 (0, max_position] 内找【最深】的、前缀哈希匹配的条目，装回内存快照槽。
 * 完整窗口及 packed KV/index 均在私有 RAM 中通过 CRC/布局校验后才发布。
 * 返回 0 时保留原 live state 与内存快照，可安全继续用浅内存前缀。
 * 返回命中的位置，随后由 myds_v41_snapshot_restore 应用。
 * 临时内存最多约 44.4 MiB，成功/失败后立即释放。 */
uint32_t myds_v41_low_kv_load(myds_v41_context_t *context, const int32_t *tokens,
                              uint32_t max_position);
/* Like load, but never publishes a position <= memory_position, including
 * when the earlier metadata probe has become unavailable. */
uint32_t myds_v41_low_kv_load_deeper(myds_v41_context_t *context, const int32_t *tokens,
                                     uint32_t memory_position, uint32_t max_position);
/* 落盘条目的诊断信息（命中数 / 有效数），供日志与测试读取。 */
void     myds_v41_low_kv_stats(const myds_v41_context_t *context,
                               uint32_t *live_entries, uint32_t *slots);
int32_t myds_v41_forward(myds_context_t *context, int32_t token,
                        uint32_t position, bool need_logits);
int myds_v41_prefill_tokens(myds_context_t *context, const int32_t *tokens,
                            uint32_t n_tokens, volatile int *cancel_flag);

/* ── 投机解码轻量状态快照与回退 ── */
#define MYDS_V41_SPEC_MAX_AHEAD 16
#define MYDS_V41_LAYERS_COUNT   40

typedef struct {
    uint32_t pos;
    uint32_t next_position;
    myds_engram_history_t engram_history;
    uint32_t engram_position;
    float previous_kv[4][512];
    float previous_score[4][512];
    uint32_t window_start_pos;
    uint32_t window_count;
    float window_saved[MYDS_V41_LAYERS_COUNT][MYDS_V41_SPEC_MAX_AHEAD][512];
    bool valid;
} myds_v41_spec_snapshot_t;

bool myds_v41_spec_snapshot_save(myds_context_t *context, myds_v41_spec_snapshot_t *snap, uint32_t count);
bool myds_v41_spec_snapshot_restore(myds_context_t *context, const myds_v41_spec_snapshot_t *snap);

#endif

