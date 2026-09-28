#ifndef MYDS_SERVER_INTERNAL_H
#define MYDS_SERVER_INTERNAL_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void json_escape(const char *src, char *dst, size_t dst_size);
void json_unescape_slice(const char *src, size_t len, char *dst, size_t dst_size);
int extract_http_header_int(const char *headers, const char *header_name);
int json_extract_string_field(const char *json, const char *key, char *out, size_t out_size);
int json_extract_bool_field(const char *json, const char *key, int default_val);
int json_extract_int_field(const char *json, const char *key, int default_val);
float json_extract_float_field(const char *json, const char *key, float default_val);
const char *find_matching_brace(const char *start);
void extract_message_text(const char *obj_buf, char *out, size_t out_size);
void parse_chat_prompt(const char *body, char *out_prompt, size_t max_prompt_len);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_SERVER_INTERNAL_H */
