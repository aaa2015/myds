#ifndef MYDS_SERVER_INTERNAL_H
#define MYDS_SERVER_INTERNAL_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

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
uint32_t myds_server_default_max_tokens(uint32_t configured_max_tokens);
void myds_server_sse_callback(const char *piece, size_t length, int32_t token_id, void *user_data);
bool myds_server_sse_finish(myds_sse_stream_ctx_t *stream);
void myds_server_non_stream_callback(const char *piece, size_t length, int32_t token_id, void *user_data);

void json_unescape_slice(const char *src, size_t len, char *dst, size_t dst_size);
int extract_http_header_int(const char *headers, const char *header_name);
int json_extract_string_field(const char *json, const char *key, char *out, size_t out_size);
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

/* 响应 id 注册表（供单测）。语义：只有【最新登记】的那一个是 live。 */
void resp_id_new(char *out, size_t cap);
void resp_id_register(const char *id);
bool resp_id_is_live(const char *id);
bool resp_id_seen(const char *id);

bool parse_chat_prompt_ex(const char *body, char *out_prompt, size_t max_prompt_len,
                          const char **out_reason);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_SERVER_INTERNAL_H */

/* ── 工具调用（官方 DeepSeek-V4.1 DSML 格式，见 docs/DEEPSEEK_V41_PROMPT_FORMAT.md）──
 * 服务端把工具定义渲染进提示词，并把模型输出的 DSML 块解析成结构化调用。 */
typedef struct { char *id; char *name; char *arguments; } myds_tool_call_t;

/* 解析完成文本：*content_len 为工具块之前的正文字节数；
 * calls_out 与 count_out 为解析结果（调用方用 myds_free_tool_calls 释放）。
 * 无工具块 ⇒ count=0 且返回 true；格式不合 ⇒ 返回 false（调用方按纯文本处理）。 */
bool myds_parse_completion_tool_calls(const char *text, size_t len, size_t *content_len,
                                      myds_tool_call_t **calls_out, size_t *count_out);
void myds_free_tool_calls(myds_tool_call_t *calls, size_t count);

/* 工具调用事件的发射（内部实现，暴露出来是为了让测试能直接校验产出的 JSON 合法性） */
bool sse_emit_chat_tool_calls(myds_sse_stream_ctx_t *sctx, const myds_tool_call_t *calls, size_t count);
bool sse_emit_responses_function_calls(myds_sse_stream_ctx_t *sctx,
                                       const myds_tool_call_t *calls, size_t count);
