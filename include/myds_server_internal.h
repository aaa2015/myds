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

#ifdef __cplusplus
}
#endif

#endif /* MYDS_SERVER_INTERNAL_H */
