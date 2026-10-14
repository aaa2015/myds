#ifndef MYDS_V41_H
#define MYDS_V41_H

#include "myds.h"
#include "myds_engram.h"
#include "myds_v41_decode.h"
#include <stdbool.h>

typedef struct myds_v41_model myds_v41_model_t;
typedef struct myds_v41_context myds_v41_context_t;

/* Native, text-only V4.1 graph. Kept behind the diagnostic loader until its
 * independently captured stage tensors and logits satisfy the release gate. */
myds_v41_model_t *myds_v41_model_create(myds_model_t *model, const char *path);
void myds_v41_model_free(myds_v41_model_t *model);
myds_v41_context_t *myds_v41_context_create(myds_context_t *context);
void myds_v41_context_free(myds_v41_context_t *context);

/* MYDS_ENGRAM_PROJECT_ROUND_SKIP accepts only absent/0/1 (default 0),
 * captured when this V4.1 context is created. Changing getenv afterwards
 * cannot change the context policy. Only ordinary scalar decode with logits
 * and without tracing can omit the redundant producer BF16 round; prefill,
 * Batch2/tree and their scalar verification calls retain the reference. */
int myds_v41_engram_project_round_skip_enabled(const myds_v41_context_t *context);
typedef struct {
    uint64_t requests, encoded, failures;
} myds_v41_engram_project_round_counters_t;
/* Global host encoding counters. Requests count eligible skip attempts;
 * encoded counts an entire projection + consumer pair successfully encoded.
 * Later command-buffer completion failures belong to the caller's graph gate.
 * Resetting counters does not change any context's immutable policy. */
void myds_v41_engram_project_round_counters(
    myds_v41_engram_project_round_counters_t *out, bool reset);
/* Model-free diagnostic entry into the same production pair helper, with the
 * real DIM=5120, Engram input width=6144 and F16 projection. It retains the
 * producer RAW and consumer completion join used by ordinary auto0 decode;
 * auto1 retains the existing full barriers. The caller owns buffers and
 * command-buffer completion. Explicit scope arguments exercise reference-only
 * prefill/verification/trace cases without changing the stored policy. This
 * diagnostic rejects overlapping writable/read ranges and invalid alignment
 * before writing in every mode (projected/input: 16 bytes, weights: 8 bytes,
 * residual/q/k: 4 bytes); only eligible skip failures affect the counters. */
int myds_v41_engram_project_round_probe(const myds_v41_context_t *context,
    myds_metal_buffer_t *residual, myds_metal_buffer_t *projected,
    const myds_metal_buffer_t *projection, const myds_metal_buffer_t *input,
    const myds_metal_buffer_t *q_weight, const myds_metal_buffer_t *k_weight,
    float eps, bool ordinary_decode, bool need_logits, bool trace_on);

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
#define MYDS_V41_ANCHOR_SLOT 43u
bool     myds_v41_snapshot_store_slot(myds_v41_context_t *context, uint32_t slot, uint32_t position);

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

typedef struct myds_v41_spec_snapshot {
    uint32_t pos;
    uint32_t next_position;
    uint64_t rng_state;
    myds_engram_history_t engram_history;
    uint32_t engram_position;
    float previous_kv[4][512];
    float previous_score[4][512];
    uint32_t window_start_pos;
    uint32_t window_count;
    float window_saved[MYDS_V41_LAYERS_COUNT][MYDS_V41_SPEC_MAX_AHEAD][512];
    /* Borrowed views into the verifier's reusable workspace. A successful
     * K=2 call publishes the exact state after its first authoritative token;
     * consume immediately, before the next batch reuses the workspace. */
    myds_metal_buffer_t *first_previous_kv[4], *first_previous_score[4];
    myds_metal_buffer_t *first_residual, *first_pre, *first_logits;
    myds_metal_buffer_t *first_dspark_slots[3];
    myds_engram_history_t first_engram_history;
    uint64_t first_rng_state;
    bool first_state_valid;
    /* Continuous depth2 only: exact state after consuming seed and draft1.
     * These views have the same next-forward lifetime as the first checkpoint;
     * a first-row miss/EOS leaves second_state_valid false. */
    myds_metal_buffer_t *second_previous_kv[4], *second_previous_score[4];
    myds_metal_buffer_t *second_residual, *second_pre, *second_logits;
    myds_metal_buffer_t *second_dspark_slots[3];
    myds_engram_history_t second_engram_history;
    uint64_t second_rng_state;
    bool second_state_valid;
    bool linear_retry_valid;
    bool valid;
} myds_v41_spec_snapshot_t;

bool myds_v41_spec_snapshot_save(myds_context_t *context, myds_v41_spec_snapshot_t *snap, uint32_t count);
bool myds_v41_spec_snapshot_restore(myds_context_t *context, const myds_v41_spec_snapshot_t *snap);
bool myds_v41_spec_snapshot_restore_partial(myds_context_t *context, const myds_v41_spec_snapshot_t *snap, uint32_t keep_count);
/* Capture the actual, completed first target step for the opt-in sequential
 * comparison path; also usable by diagnostic state fixtures. */
bool myds_v41_spec_snapshot_capture_first(myds_context_t *context, myds_v41_spec_snapshot_t *snap);
/* Capture a completed authoritative second step, for diagnostics or a legal
 * scalar retry. Requires an existing first checkpoint and a count>=3 snapshot.
 * Capturing it does not reuse the first checkpoint's backing buffers. */
bool myds_v41_spec_snapshot_capture_second(myds_context_t *context, myds_v41_spec_snapshot_t *snap);

/* Independent greedy, text-only continuous verifier. Exactly three inputs
 * {seed,draft1,draft2} occupy position,position+1,position+2; out has capacity3.
 * A single three-row transformer graph computes the candidates, followed by
 * lazy authoritative heads. Returns1/2/3 consumed rows (including seed),
 * stopping on the first mismatch or EOS. The last returned output is a bonus
 * token and has not itself been consumed. No force/env flag enables sampling.
 * Failure returns-1 with no output publication and attempts original-prefix
 * rollback; an invalid context must never be retried. No scalar fallback is
 * hidden in this API. Borrowed checkpoints support immediate partial1/2. */
int myds_v41_forward_linear_spec_ex(myds_context_t *context, const int32_t *tokens,
    uint32_t n_tokens, uint32_t position, int32_t *out_tokens,
    myds_v41_spec_snapshot_t *snapshot);
/* True only after this API failed and completed its original-prefix rollback.
 * The retry must begin with the original seed at snapshot->pos. */
bool myds_v41_linear_spec_retry_ready(const myds_context_t *context,
    const myds_v41_spec_snapshot_t *snapshot);

/* ── 投机微批前向通道 ── */
/* Explicit opt-in, greedy text only. Requested excludes the per-draft
 * confidence gate so callers can skip drafter work when batching is off. */
bool myds_v41_batch2_spec_requested(const myds_context_t *context);
bool myds_v41_batch2_spec_enabled(const myds_context_t *context, uint32_t n_tokens);
int myds_v41_forward_batch_spec(myds_context_t *context, const int32_t *tokens,
                                uint32_t n_tokens, uint32_t position,
                                int32_t *out_tokens);
/* Optional reusable caller snapshot is captured once by the verifier. On
 * success it permits retaining only the first output at length/EOS/cancel;
 * errors restore it before any sequential retry. No memset is required. */
int myds_v41_forward_batch_spec_ex(myds_context_t *context, const int32_t *tokens,
                                   uint32_t n_tokens, uint32_t position,
                                   int32_t *out_tokens, myds_v41_spec_snapshot_t *snapshot);
/* Diagnostic fixture: projected Q/KV inputs pass through the production
 * batch-2 raw-window attention sequence. Real Metal kernels, no model weights.
 * The caller owns the buffers and a completed batch; failure restores window.
 * This proves the window/attention component, not full-model logits parity. */
int myds_v41_batch2_window_probe(myds_metal_buffer_t *window,
    const myds_metal_buffer_t *sinks, myds_metal_buffer_t *q[2],
    myds_metal_buffer_t *kv[2], myds_metal_buffer_t *heads[2], uint32_t position,
    bool fail_after_first);
/* Diagnostic entry into the production Batch-2 projection boundary. Each
 * row retains the authoritative decode helper's reduction and BF16 store.
 * Q4 shared providers failed current-checkpoint parity: mode0 and mode1 both
 * use scalar PSOs in this entry, including the LM head. Direct Metal shared2
 * APIs retain their diagnostic candidates. Q8/dense selection is unchanged. */
typedef enum {
    MYDS_V41_BATCH2_QKV, MYDS_V41_BATCH2_TAIL,
    MYDS_V41_BATCH2_SHARED, MYDS_V41_BATCH2_ROUTER,
    MYDS_V41_BATCH2_LM_HEAD
} myds_v41_batch2_projection_kind_t;
int myds_v41_batch2_projection_probe(myds_metal_buffer_t *out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *input[2],
    uint32_t rows, uint32_t width, ggml_type type,
    myds_v41_batch2_projection_kind_t kind);

/* Continuous depth2 provider selector: unset/0 retains the validated graph,
 * literal 1 uses the independent shared3 kernels. Other values fail closed.
 * Does not select a provider for ordinary decode, Batch2 or the sibling tree. */
int myds_v41_depth2_shared3_mode(void);
typedef struct { uint64_t requests, encoded, failures; }
    myds_v41_depth2_routed_shared3_counters_t;
void myds_v41_depth2_routed_shared3_counters(
    myds_v41_depth2_routed_shared3_counters_t *out, bool reset);
/* Exactly three F32 inputs and top-six routes in contiguous row order.
 * Reuses each union expert's quantized weights, while preserving each row's
 * original slot and block reduction order. No scalar fallback on failure. */
int myds_v41_routed_moe_shared3(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *x, const myds_metal_buffer_t *gate,
    const myds_metal_buffer_t *up, const myds_metal_buffer_t *down,
    const myds_metal_buffer_t *ids, const myds_metal_buffer_t *weights,
    myds_metal_buffer_t *mid, uint32_t dim, uint32_t width, uint32_t experts,
    ggml_type gate_type, ggml_type up_type, ggml_type down_type, float clamp);

/* ── 推测树（Tree-based Speculative Decoding）级联前向与3-Way裁决器 ── */
/* Experimental: tokens={root,A,B}; A/B occupy the same position+1.
 * Returns one/two authoritative outputs. Production dispatcher uses this only
 * with MYDS_TREE_SPEC=1 and native speculation enabled; default remains off.
 * Borrowed first-state checkpoint in _ex remains valid until the next forward. */
int myds_v41_forward_tree_spec(myds_context_t *context, const int32_t *tokens,
                               uint32_t position, int32_t *out_tokens);

#endif
