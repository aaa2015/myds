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

typedef struct {
    uint64_t seq;                            /* 单调递增，1 起 */
    int64_t  epoch_ms;                       /* 请求到达的墙上时间 */
    char     path[MYDS_TRACE_PATH_MAX];
    char     client[MYDS_TRACE_CLIENT_MAX];
    char     finish[MYDS_TRACE_FINISH_MAX];  /* finish_reason 字符串 */
    myds_trace_phases_t phases;
    uint32_t prompt_tokens;
    uint32_t cached_tokens;
    uint32_t out_tokens;
    int      stream;
    /* 逐模型层耗时。**默认全 0 且 layer_count == 0** —— 引擎只在
     * MYDS_PROFILE_PREFILL=1 时才有逐层计时，而那个开关会把 prefill chunk
     * 退化成串行，不能为了页面好看默认打开。
     * layer_count == 0 时页面必须显示「未测量」，不能画成一条平线。 */
    uint32_t layer_count;
    float    layer_ms[MYDS_TRACE_LAYERS_MAX];
} myds_trace_rec_t;

/* 初始化环形缓冲。capacity 上限 MYDS_TRACE_CAP_MAX；0 或过大都夹取。 */
void myds_trace_init(uint32_t capacity);

/* 存一条。seq 与 epoch_ms 由本函数填写，调用方不用管。
 * 缓冲未初始化时自动按默认容量初始化，所以不必保证 init 先于 record。 */
void myds_trace_record(myds_trace_rec_t *rec);

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

/* 页面本体（HTML）。返回静态字符串，不需要释放。 */
const char *myds_trace_page_html(void);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_TRACE_H */
