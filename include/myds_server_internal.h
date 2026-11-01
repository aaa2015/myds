#ifndef MYDS_SERVER_INTERNAL_H
#define MYDS_SERVER_INTERNAL_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

void json_escape(const char *src, char *dst, size_t dst_size);
/* Binary-safe JSON text encoding. Invalid or final incomplete UTF-8 becomes
 * U+FFFD; embedded NUL becomes \u0000. Insufficient space returns false. */
bool json_escape_bytes(const char *src, size_t length, char *dst, size_t dst_size, size_t *out_length);

typedef struct {
    unsigned char pending[4];
    size_t pending_length;
} myds_utf8_state_t;

typedef struct myds_sse_writer myds_sse_writer_t;

typedef struct {
    int client_fd;
    bool responses_api;
    uint32_t token_count;
    double t_first;
    double t_last;
    volatile int cancel_flag;
    bool failed;
    myds_utf8_state_t utf8;
    /* 累积原始文本：DSML 工具块要跨 token 才能识别，识别到就停止输出正文 */
    char *acc;
    size_t acc_len, acc_cap;
    size_t emitted;
    bool in_tool_block;
    bool thinking;
    int think_state;    /* 0: NONE, 1: REASONING, 2: CONTENT */
    int msg_output_index; /* 0 if !thinking, 1 if thinking */
    /* Request-owned transport; only the producer mutates this stream context. */
    myds_sse_writer_t *writer;
} myds_sse_stream_ctx_t;

typedef struct {
    int client_fd;
    volatile int cancel_flag;
    bool failed;
    char *buf;
    size_t cap;
    size_t len;
    uint32_t token_count;
    double t_first;
    double t_last;
} myds_non_stream_ctx_t;

bool myds_server_send_all(int fd, const char *data, size_t length);
/* 对端是否已关闭（非阻塞 peek）。SSE/非流式回调与 prefill 探针共用。 */
bool client_disconnected(int fd);
/* 引擎 prefill 期间回调它；配合 g_prefill_probe_fd 使用。 */
bool server_client_alive_probe(void *user_data);
extern int g_prefill_probe_fd;
/* 单调时钟秒数（SSE 排空截止与请求分阶段计时共用）。 */
double myds_server_now_s(void);
/* 组成格式化串（alloc 所有权归调用方）。 */
char *format_alloc(const char *format, ...);
/* 把一段正文/收尾事件写进 SSE 流。handle_client 收尾时要用。 */
bool sse_emit_content(myds_sse_stream_ctx_t *sctx, const char *piece, size_t length, bool final);
bool sse_emit_responses_close_message(myds_sse_stream_ctx_t *sctx,
                                      const char *text, size_t text_len);
/* 一次性生成完整 HTTP 响应（含头部）。 */
bool send_response(int fd, int code, const char *status, const char *content_type, const char *body);
uint32_t myds_server_default_max_tokens(uint32_t configured_max_tokens, bool thinking,
                                        const char *reasoning_effort);
void myds_server_sse_callback(const char *piece, size_t length, int32_t token_id, void *user_data);
bool myds_server_sse_finish(myds_sse_stream_ctx_t *stream);
/* Start a bounded FIFO transport. Enqueue never waits for the socket or space;
 * exceeding the byte budget cancels the stream. Join drains with one overall
 * deadline, or discards pending bytes when aborting. Join precedes fd close. */
bool myds_server_sse_writer_start(myds_sse_stream_ctx_t *stream, size_t byte_limit,
                                  unsigned drain_timeout_ms);
bool myds_server_sse_send(myds_sse_stream_ctx_t *stream, const char *data, size_t length);
bool myds_server_sse_writer_join(myds_sse_stream_ctx_t *stream, bool aborting);
unsigned myds_server_sse_drain_timeout_ms(void);
/* Deterministic regression hook: 1 = next thread creation, 2 = next packet
 * allocation. A point is consumed once; zero disables injection. */
void myds_server_sse_writer_inject_failure(unsigned point);
void myds_server_non_stream_callback(const char *piece, size_t length, int32_t token_id, void *user_data);

void json_unescape_slice(const char *src, size_t len, char *dst, size_t dst_size);
int extract_http_header_int(const char *headers, const char *header_name);
int extract_http_header_str(const char *headers, const char *header_name,
                            char *out, size_t out_size);
int json_extract_string_field(const char *json, const char *key, char *out, size_t out_size);
/* 顶层字段存在性（深度 1），避免误判 input 内容里的同名字样。 */
bool json_has_top_field(const char *body, const char *key);
int json_extract_bool_field(const char *json, const char *key, int default_val);
int json_extract_int_field(const char *json, const char *key, int default_val);
float json_extract_float_field(const char *json, const char *key, float default_val);
const char *find_matching_brace(const char *start);
void extract_message_text(const char *obj_buf, char *out, size_t out_size);
/* False rejects malformed/unsupported JSON or insufficient capacity; output is empty. */
bool parse_chat_prompt(const char *body, char *out_prompt, size_t max_prompt_len);
/* 同上，但把【失败原因】回传给调用方（服务端用它写日志与精确报错）。
 * 两者原本共用一个笼统文案，导致「带 tools 被拒」与「提示词太大」在线上无法区分。 */
/* previous_response_id：只渲染新增部分（无 BOS / instructions / tools），前缀由引擎缓存提供。 */
bool parse_chat_prompt_suffix_ex(const char *body, char *out_prompt, size_t max_prompt_len,
                                 const char **out_reason);

/* 响应 id 注册表（供单测）。语义：只有【最新登记】的那一个是 live。
 * tools_sig = 该轮所用工具集的指纹（0 = 无 tools）；续跑时用于拒绝「中途换工具集」。 */
void resp_id_new(char *out, size_t cap);
void resp_id_register(const char *id, unsigned long long tools_sig);
void resp_id_register_text(const char *id, unsigned long long tools_sig, const char *accumulated);
const char *resp_id_saved_text(const char *id);
/* 在锁内拷贝，调用方负责 free(*out)。返回字节数，0 = 没有该 id。 */
size_t resp_id_saved_text_copy(const char *id, char **out);
unsigned long long resp_id_live_tools_sig(void);
unsigned long long resp_id_tools_sig(const char *id);
/* 工具集指纹（供单测与续跑一致性校验）。0 = 无 tools。 */
unsigned long long server_tools_signature(const char *body, size_t scratch_cap);

/* response_format 分类（供单测与请求校验共用）。
 * 0 = 未指定 / 1 = json_object / 2 = json_schema。
 * 本引擎没有约束解码器，所以 handler 对 2 直接 400，对 1 只告警 ——
 * 判定逻辑抽出来单测，免得「静默接受一个不会生效的语法约定」再回归。 */
int server_response_format_kind(const char *body);
#define SERVER_RF_NONE        0
#define SERVER_RF_JSON_OBJECT 1
#define SERVER_RF_JSON_SCHEMA 2
bool resp_id_is_live(const char *id);
bool resp_id_seen(const char *id);
void resp_id_invalidate_all(void);

/* 服务运行标志：信号处理器写它，多个地方读它。
 * 2026-11-01 拆分时暴露出来 —— 队列的排队等待循环必须能看见服务正在关闭。 */
extern volatile sig_atomic_t g_running;

/* 推理名额队列（myds_infer_queue.c）。
 * 返回 0 = 拿到名额；-1 = 队列已满；-2 = 排队超时。调用方拿到名额后
 * 还要自己取 g_infer_mutex —— 队列只决定「轮到谁」，不决定「谁在跑」。 */
int  server_acquire_infer_slot(int timeout_ms, int max_depth);
void server_release_infer_slot(void);
uint32_t server_infer_queue_depth(void);

void server_vjob_gpu_lock(void);
void server_vjob_gpu_unlock(void);
int server_vjob_gpu_last_error(void);

/* 带视觉槽的版本：img 非空时，input_image 内容块会被处理并填进 sink。 */
struct myds_vjob_sink_s;
bool parse_chat_prompt_vis(const char *body, char *out_prompt, size_t max_prompt_len,
                           const char **out_reason, struct myds_vjob_sink_s *img);
bool parse_chat_prompt_suffix_vis(const char *body, char *out_prompt, size_t max_prompt_len,
                                  const char **out_reason, struct myds_vjob_sink_s *img);

bool parse_chat_prompt_ex(const char *body, char *out_prompt, size_t max_prompt_len,
                          const char **out_reason);
bool extract_request_thinking_mode(const char *body, bool default_val);
int extract_request_reasoning_budget(const char *body, int default_budget);
void split_reasoning_content(const char *text, size_t n, char **content_out, char **reasoning_out);

/* ── 工具调用（官方 DeepSeek-V4.1 DSML 格式，见 docs/DEEPSEEK_V41_PROMPT_FORMAT.md）──
 * 服务端把工具定义渲染进提示词，并把模型输出的 DSML 块解析成结构化调用。 */
typedef struct { char *id; char *name; char *arguments; } myds_tool_call_t;

/* 解析完成文本：*content_len 为工具块之前的正文字节数；
 * calls_out 与 count_out 为解析结果（调用方用 myds_free_tool_calls 释放）。
 * 无工具块 ⇒ count=0 且返回 true；格式不合 ⇒ 返回 false（调用方按纯文本处理）。 */
/* DSML 工具块的状态。调用方据此决定「当正文发 / 当工具调用执行 / 隔离」。 */
typedef enum {
    MYDS_DSML_NONE = 0,    /* 没有 <|DSML| calls> */
    MYDS_DSML_COMPLETE,    /* 有开有闭，内部结构有效 */
    MYDS_DSML_TRUNCATED,   /* 有开无闭：被 max_tokens 截断或模型没写完 */
    MYDS_DSML_MALFORMED    /* 有开有闭，但内部结构不合语法 */
} myds_dsml_state_t;

/* DSML 标签前缀（全角竖线）。合并在头里，免得两个编译单元各写一遍字面量。 */
#define MYDS_DSML_S "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"

/* 拼 JSON 的缓冲原语。导出给 SSE 发射与 Responses 输出构造使用；
 * 唯一用途就是「拼工具调用 JSON」，所以与解析放在同一个编译单元。 */
typedef struct { char *data; size_t len, cap; } dsml_buf_t;
bool  dsml_put(dsml_buf_t *b, const char *p, size_t n);
bool  dsml_putf(dsml_buf_t *b, const char *fmt, ...);
bool  dsml_put_escaped(dsml_buf_t *b, const char *p, size_t n);
char *dsml_dup(const char *p, size_t n);
/* 找到 lit 并返回【其之后】的位置；未找到返回 NULL。 */
const char *dsml_find(const char *pos, const char *end, const char *lit);
#define DSML_PUTS(b, lit) dsml_put((b), (lit), strlen(lit))

/* json_escape_alloc 转义但不加引号，供 dsml_put_escaped 使用。 */
char *json_escape_alloc(const char *src, size_t length);
/* UTF-8 跨 token 分片转义（SSE 每次只拿到一个 token 的字节）。 */
bool escape_utf8_chunk(myds_utf8_state_t *state, const char *src, size_t length,
                       bool final, char *dst, size_t dst_size, size_t *out_length);

/* 服务级默认思考模式：--thinking 决定，请求可覆盖。 */
extern bool g_server_default_thinking;
int  extract_reasoning_object_effort(const char *body, char *out, size_t out_size);
const char *cors_origin_header(void);
/* OpenAI 的 tool_calls 数组（arguments 是 JSON 字符串）。 */
char *openai_tool_calls_json(const myds_tool_call_t *calls, size_t count);

myds_dsml_state_t myds_dsml_classify(const char *text, size_t len, size_t *content_len);
/* 正文里出现了「另一套方言」的工具标记（如 <function_calls>）但要按纯文本发出去。
 * 只在日志里用它提醒，不改变任何行为。 */
bool dsml_looks_like_foreign_dialect(const char *text, size_t len);

bool myds_parse_completion_tool_calls(const char *text, size_t len, size_t *content_len,
                                      myds_tool_call_t **calls_out, size_t *count_out);
void myds_free_tool_calls(myds_tool_call_t *calls, size_t count);

/* 工具调用事件的发射（内部实现，暴露出来是为了让测试能直接校验产出的 JSON 合法性） */
bool sse_emit_chat_tool_calls(myds_sse_stream_ctx_t *sctx, const myds_tool_call_t *calls, size_t count);
bool sse_emit_responses_function_calls(myds_sse_stream_ctx_t *sctx,
                                       const myds_tool_call_t *calls, size_t count);

/* Responses 流式的【收尾】事件（暴露给测试：要能直接校验聚合文本确实写进去了）。
 * 只发增量不发聚合，是 OpenAI 兼容实现里常见坑 —— 读汇总的客户端会拿到空字符串。 */
bool sse_emit_responses_finish(myds_sse_stream_ctx_t *sctx, const char *model_alias,
                               const char *resp_id, unsigned prompt_tokens, unsigned output_tokens,
                               const char *text, size_t text_len,
                               const char *output_items_json, int emit_text_events);
/* 文本响应的 output 数组（一条 message 项），内部自己转义。 */
char *responses_output_message_json(const char *text, size_t text_len);
/* Responses API 的 output 数组（包含思考、正文、工具调用） */
char *responses_output_items_json(const char *content, size_t content_len,
                                  const myds_tool_call_t *calls, size_t count);
char *responses_output_items_json_ex(bool thinking, const char *reasoning, size_t reasoning_len,
                                     const char *content, size_t content_len,
                                     const myds_tool_call_t *calls, size_t count);
/* 工具调用响应的 output 数组（给 response.completed 用）。 */
char *responses_function_calls_output_json(const myds_tool_call_t *calls, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_SERVER_INTERNAL_H */
