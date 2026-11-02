#ifndef MYDS_TRACE_H
#define MYDS_TRACE_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 请求追踪：最近若干条请求的分阶段耗时，供 /trace 页面读取。
 *
 * 为什么单独一个模块：这是**观测**，不参与推理。放进 myds_server.c 会让那个文件
 * 继续变大，也会让「记录一条」与「处理一个请求」的失败模式混在一起。
 *
 * 记账原则（与 request_phases 日志一致）：**未测量的阶段存 -1，不是 0**。
 * 打印一个 0.0 与「测了，真的没花时间」无法区分，而页面上的 0 会被当成事实读走。
 * 这里把这条规则放进数据结构本身，而不是靠每个使用点记得。 */

#define MYDS_TRACE_LAYERS_MAX 64
#define MYDS_TRACE_PATH_MAX   48
#define MYDS_TRACE_CLIENT_MAX 48
#define MYDS_TRACE_FINISH_MAX 40
#define MYDS_TRACE_RESPID_MAX 48

/* 一个阶段的耗时。负值 = 该请求没有这个阶段（早退、取消、引擎未调用）。 */
typedef struct {
    double queue_s;         /* 排队等推理名额 */
    double render_s;        /* 解析请求 + 渲染提示词 */
    double setup_s;         /* 渲染之后、进引擎之前 */
    double engine_s;        /* generate() 全程 */
    double prefill_s;       /* 引擎自报 */
    double decode_s;        /* 引擎自报 */
    double unattributed_s;  /* engine - prefill - decode；NaN = 算不出 */
} myds_trace_phases_t;

/* 细粒度分解（2026-11-03）。全部来自引擎的常开计数器。
 *
 * ★ 关键口径：gpu 与 wait **会重叠**。异步提交时 GPU 可以与 CPU 并行，
 *   所以 gpu + wait 不必等于阶段墙钟。页面必须分开显示，不得相加。
 *   host = 阶段墙钟 - wait（提交/调度/采样，全在 CPU）。
 *
 * 负值一律 = 未测量。绝不填 0。 */
typedef struct {
    double prefill_gpu_s;   /* prefill 期间 GPU 真正在算的时间 */
    double prefill_wait_s;  /* prefill 期间阻塞在 waitUntilCompleted 的墙钟 */
    double decode_gpu_s;
    double decode_wait_s;
    double tokenize_s;      /* 分词（在引擎内部，不属于服务端 render） */
    double kvdsk_s;         /* 磁盘 KV 快照加载 + 应用 */
    double emit_s;          /* 回调里把 token 交出去的累计（写 socket / 拼缓冲） */
    uint32_t prefill_cbs;
    uint32_t decode_cbs;
    /* 按阶段标签的 GPU 忙时（下标即 myds_ph_tag_t，见 myds_metal.h）。
     * 只用于页面的树状明细；为 0 的桶不输出。 */
    double   prefill_tag_gpu[52];
    uint64_t prefill_tag_cbs[52];   /* ★ 必须与引擎侧同宽：uint32 会与 uint64 错位 */
    double   decode_tag_gpu[52];
    uint64_t decode_tag_cbs[52];
} myds_trace_detail_t;

typedef struct {
    uint64_t seq;                            /* 单调递增，1 起 */
    int64_t  epoch_ms;                       /* 请求到达的墙上时间 */
    char     path[MYDS_TRACE_PATH_MAX];
    char     client[MYDS_TRACE_CLIENT_MAX];
    char     finish[MYDS_TRACE_FINISH_MAX];  /* finish_reason 字符串 */
    /* 客户端拿到的 response id。把它记进追踪的理由：用户报「那次请求慢」时，
     * 手上有的是响应里的 id，不是这里的序号；两者能对上才有用。 */
    char     resp_id[MYDS_TRACE_RESPID_MAX];
    myds_trace_phases_t phases;
    myds_trace_detail_t detail;
    uint32_t prompt_tokens;
    uint32_t cached_tokens;
    uint32_t out_tokens;
    int      stream;

    /* 吞吐与延迟。同样遵守「未测量 = 负值 = null」：
     * 取消的请求没有 decode 速度，早退的请求没有 TTFT，
     * 把它们印成 0 t/s 会被读成「慢到零」，而真相是「没发生」。 */
    /* 前缀缓存命中率 = cached / prompt。
     * 分母用 prompt_tokens 而不是「prompt - 1」：最后一个 token 留给 decode
     * 是引擎的内部约定，对外讲命中率时按整段 prompt 算才符合直觉。
     * 负值 = 未测量（请求早退/被拒，没有 prompt 统计）。 */
    double   cache_hit_rate;
    double   ttft_s;        /* 首字时间：请求到达 → 第一个 token 吐出（含排队） */
    double   prefill_tps;   /* 引擎自报的 prefill 速率 */
    double   decode_tps;    /* 生成阶段速率 */
    /* 逐模型层耗时。**默认全 0 且 layer_count == 0** —— 引擎只在
     * MYDS_PROFILE_PREFILL=1 时才有逐层计时，而那个开关会把 prefill chunk
     * 退化成串行，不能为了页面好看默认打开。
     * layer_count == 0 时页面必须显示「未测量」，不能画成一条平线。 */
    uint32_t layer_count;
    float    layer_ms[MYDS_TRACE_LAYERS_MAX];
    /* 1 = 请求已到达但还没结束。页面据此显示「进行中」并用墙上时间算已耗时，
     * 否则一次 40 秒的 prefill 在追踪里完全不可见 —— 只能等它结束才出现，
     * 而这正是最需要看的时候。 */
    int      in_flight;
} myds_trace_rec_t;

/* 初始化环形缓冲。capacity 上限 MYDS_TRACE_CAP_MAX；0 或过大都夹取。 */
void myds_trace_init(uint32_t capacity);

/* 存一条。seq 与 epoch_ms 由本函数填写，调用方不用管。
 * 缓冲未初始化时自动按默认容量初始化，所以不必保证 init 先于 record。 */
void myds_trace_record(myds_trace_rec_t *rec);

/* 请求到达即登记，返回它的 seq；结束时用 myds_trace_complete 回填。
 *
 * 为什么要两段式：单段式（只在结束时写）会让长请求在整个执行期间不可见。
 * 排查「现在卡在哪」时恰恰最需要看正在进行的那条，而不是等它结束。
 *
 * begin 只填 path / client / resp_id / epoch_ms，并把 in_flight 置 1；
 * complete 按 seq 找回同一条，覆盖其余字段并清 in_flight。找不到（已被环形缓冲
 * 挤掉）时静默丢弃 —— 那条记录本来就已经不在了，补一条新的反而会打乱顺序。 */
uint64_t myds_trace_begin(const char *path, const char *client, const char *resp_id);
void     myds_trace_complete(uint64_t seq, const myds_trace_rec_t *rec);

/* 已记录的条数（不超过容量）与容量。 */
uint32_t myds_trace_count(void);
uint32_t myds_trace_capacity(void);

/* 拷贝第 index 条，index 0 = 最旧。越界返回 false。 */
bool myds_trace_get(uint32_t index, myds_trace_rec_t *out);

/* 按 seq 查。返回 false = 已被环形缓冲覆盖（页面要能说清「这条没了」）。 */
bool myds_trace_get_by_seq(uint64_t seq, myds_trace_rec_t *out);

/* 清空（测试用）。 */
void myds_trace_reset(void);

/* ── JSON 渲染：纯格式化，无全局状态，便于单测 ──
 *
 * 列表：{"count":N,"capacity":M,"records":[{...}, ...]}，最新的在前。
 * 单条：{"record":{...}}
 * 任一缓冲不足返回 false（调用方据此回 500，而不是发出截断的 JSON ——
 * 截断的 JSON 客户端解析失败，反而更难查）。 */
bool myds_trace_list_json(char *out, size_t cap);
bool myds_trace_detail_json(uint64_t seq, char *out, size_t cap);

/* 页面本体（HTML）的模板。占位符尚未替换，不要直接发给客户端。 */
const char *myds_trace_page_html(void);

/* 用真实 token 展开模板，返回新分配的字符串（调用方 free）。
 *
 * 必须用这个而不是直接发模板：页面里的 fetch 是独立请求，不会继承地址栏的
 * query，所以 token 要由服务端注入到脚本里。token 含非 URL 安全字符时注入空串，
 * 页面仍可打开但 API 会 404 —— 显式失败，好过拼出坏掉的 JS。 */
char *myds_trace_page_html_for_token(const char *token, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_TRACE_H */
