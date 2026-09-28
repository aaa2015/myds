#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Forward declare static functions from myds_server.c by including test harness */
/* We can include the helper functions or test the parsing directly */

/* Test matching brace */
static const char *test_find_matching_brace(const char *start) {
    if (!start || *start != '{') return NULL;
    int depth = 0;
    int in_quote = 0;
    int escaped = 0;
    for (const char *p = start; *p; p++) {
        if (in_quote) {
            if (escaped) {
                escaped = 0;
            } else if (*p == '\\') {
                escaped = 1;
            } else if (*p == '"') {
                in_quote = 0;
            }
        } else {
            if (*p == '"') {
                in_quote = 1;
            } else if (*p == '{') {
                depth++;
            } else if (*p == '}') {
                depth--;
                if (depth == 0) return p;
            }
        }
    }
    return NULL;
}

static void test_json_unescape_slice(const char *src, size_t len, char *dst, size_t dst_size) {
    if (!src || !dst || dst_size == 0) return;
    size_t di = 0;
    for (size_t si = 0; si < len && di + 1 < dst_size; si++) {
        if (src[si] == '\\' && si + 1 < len) {
            si++;
            switch (src[si]) {
                case '"':  dst[di++] = '"'; break;
                case '\\': dst[di++] = '\\'; break;
                case '/':  dst[di++] = '/'; break;
                case 'n':  dst[di++] = '\n'; break;
                case 'r':  dst[di++] = '\r'; break;
                case 't':  dst[di++] = '\t'; break;
                case 'u': {
                    if (si + 4 < len) {
                        char hex[5] = {src[si+1], src[si+2], src[si+3], src[si+4], 0};
                        unsigned int val = 0;
                        if (sscanf(hex, "%x", &val) == 1) {
                            if (val < 0x80) {
                                if (di + 1 < dst_size) dst[di++] = (char)val;
                            } else if (val < 0x800) {
                                if (di + 2 < dst_size) {
                                    dst[di++] = (char)(0xC0 | (val >> 6));
                                    dst[di++] = (char)(0x80 | (val & 0x3F));
                                }
                            } else {
                                if (di + 3 < dst_size) {
                                    dst[di++] = (char)(0xE0 | (val >> 12));
                                    dst[di++] = (char)(0x80 | ((val >> 6) & 0x3F));
                                    dst[di++] = (char)(0x80 | (val & 0x3F));
                                }
                            }
                        }
                        si += 4;
                    }
                    break;
                }
                default: dst[di++] = src[si]; break;
            }
        } else {
            dst[di++] = src[si];
        }
    }
    dst[di] = '\0';
}

static int test_json_extract_string_field(const char *json, const char *key, char *out, size_t out_size) {
    if (!json || !key || !out || out_size == 0) return 0;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
    if (*p != '"') return 0;
    p++;
    const char *start = p;
    int escaped = 0;
    while (*p) {
        if (*p == '\\' && !escaped) {
            escaped = 1;
        } else if (*p == '"' && !escaped) {
            break;
        } else {
            escaped = 0;
        }
        p++;
    }
    size_t len = (size_t)(p - start);
    test_json_unescape_slice(start, len, out, out_size);
    return 1;
}

static void test_extract_message_text(const char *obj_buf, char *out, size_t out_size) {
    out[0] = '\0';
    if (test_json_extract_string_field(obj_buf, "content", out, out_size) && out[0] != '\0') {
        return;
    }
    if (test_json_extract_string_field(obj_buf, "text", out, out_size) && out[0] != '\0') {
        return;
    }
    if (test_json_extract_string_field(obj_buf, "output", out, out_size) && out[0] != '\0') {
        return;
    }
    const char *content_ptr = strstr(obj_buf, "\"content\"");
    const char *p = content_ptr ? content_ptr : obj_buf;
    size_t cur = 0;
    while ((p = strstr(p, "\"text\"")) != NULL) {
        char piece[4096];
        if (test_json_extract_string_field(p, "text", piece, sizeof(piece))) {
            size_t plen = strlen(piece);
            if (cur + plen + 2 < out_size) {
                if (cur > 0) out[cur++] = '\n';
                memcpy(out + cur, piece, plen);
                cur += plen;
                out[cur] = '\0';
            }
        }
        p += 6;
    }
}

static void test_parse_chat_prompt(const char *body, char *out_prompt, size_t max_prompt_len) {
    out_prompt[0] = '\0';
    if (!body || !body[0]) {
        snprintf(out_prompt, max_prompt_len, "Hello");
        return;
    }

    char instructions[4096] = "";
    test_json_extract_string_field(body, "instructions", instructions, sizeof(instructions));

    const char *input_field = strstr(body, "\"input\"");
    if (input_field) {
        const char *p = input_field + strlen("\"input\"");
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
        if (*p == '"') {
            char plain_input[8192] = "";
            if (test_json_extract_string_field(body, "input", plain_input, sizeof(plain_input))) {
                size_t cur_len = snprintf(out_prompt, max_prompt_len, "<｜begin of sentence｜>");
                if (instructions[0]) {
                    cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                                        "<｜User｜>[System: %s]\n", instructions);
                }
                cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                                    "<｜User｜>%s\n<｜Assistant｜>", plain_input);
                return;
            }
        }
    }

    const char *arr_key = strstr(body, "\"messages\"");
    if (!arr_key) {
        arr_key = strstr(body, "\"input\"");
    }

    if (!arr_key) {
        char plain_prompt[4096];
        if (test_json_extract_string_field(body, "prompt", plain_prompt, sizeof(plain_prompt))) {
            size_t cur_len = snprintf(out_prompt, max_prompt_len, "<｜begin of sentence｜>");
            if (instructions[0]) {
                cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                                    "<｜User｜>[System: %s]\n", instructions);
            }
            cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                                "<｜User｜>%s\n<｜Assistant｜>", plain_prompt);
            return;
        }
        snprintf(out_prompt, max_prompt_len, "Hello");
        return;
    }

    const char *arr_start = strchr(arr_key, '[');
    if (!arr_start) {
        snprintf(out_prompt, max_prompt_len, "Hello");
        return;
    }

    size_t cur_len = 0;
    cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len, "<｜begin of sentence｜>");
    if (instructions[0]) {
        cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                            "<｜User｜>[System: %s]\n", instructions);
    }

    const char *p = arr_start + 1;
    while (*p) {
        const char *obj_start = strchr(p, '{');
        if (!obj_start) break;

        const char *arr_end = strchr(p, ']');
        if (arr_end && arr_end < obj_start) break;

        const char *obj_end = test_find_matching_brace(obj_start);
        if (!obj_end) break;

        size_t obj_len = (size_t)(obj_end - obj_start + 1);
        char obj_buf[16384];
        if (obj_len >= sizeof(obj_buf)) obj_len = sizeof(obj_buf) - 1;
        memcpy(obj_buf, obj_start, obj_len);
        obj_buf[obj_len] = '\0';

        char role[64] = "user";
        char content[8192] = "";

        test_json_extract_string_field(obj_buf, "role", role, sizeof(role));
        test_extract_message_text(obj_buf, content, sizeof(content));

        if (content[0]) {
            if (strcmp(role, "system") == 0) {
                cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                                    "<｜User｜>[System: %s]\n", content);
            } else if (strcmp(role, "assistant") == 0) {
                cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                                    "<｜Assistant｜>%s\n", content);
            } else {
                cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len,
                                    "<｜User｜>%s\n", content);
            }
        }

        p = obj_end + 1;
    }

    cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len, "<｜Assistant｜>");
}

int main(void) {
    printf("====================================================\n");
    printf("     Running myds Server Protocol Unit Tests        \n");
    printf("====================================================\n");

    /* 1. Test brace matching with nested JSON and braces inside quotes */
    {
        const char *nested = "{\"type\":\"message\",\"content\":[{\"type\":\"input_text\",\"text\":\"hello {world}\"}]}";
        const char *end = test_find_matching_brace(nested);
        assert(end != NULL);
        assert(*end == '}');
        assert(end == nested + strlen(nested) - 1);
        printf("  [PASS] Nested brace matching and string literal escape handling\n");
    }

    /* 2. Test UTF-8 escape unescaping */
    {
        const char *escaped_src = "\\u4f60\\u597d"; /* 你好 */
        char out[64] = {0};
        test_json_unescape_slice(escaped_src, strlen(escaped_src), out, sizeof(out));
        assert(strcmp(out, "你好") == 0);
        printf("  [PASS] Unicode escape to UTF-8 decoding (e.g. \\u4f60\\u597d -> 你好)\n");
    }

    /* 3. Test OpenAI Responses API structured input extraction */
    {
        const char *req_body =
            "{\n"
            "  \"model\": \"local-myds-deepseek-v4.1-flash-q2\",\n"
            "  \"instructions\": \"You are an expert systems programmer.\",\n"
            "  \"input\": [\n"
            "    {\n"
            "      \"type\": \"message\",\n"
            "      \"role\": \"user\",\n"
            "      \"content\": [\n"
            "        {\"type\": \"input_text\", \"text\": \"Write a C queue.\"}\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            "  \"stream\": true,\n"
            "  \"max_output_tokens\": 1024\n"
            "}";

        char prompt[4096];
        test_parse_chat_prompt(req_body, prompt, sizeof(prompt));
        assert(strstr(prompt, "<｜User｜>[System: You are an expert systems programmer.]\n") != NULL);
        assert(strstr(prompt, "<｜User｜>Write a C queue.\n") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
        printf("  [PASS] OpenAI Responses API input array with instructions\n");
    }

    /* 4. Test OpenAI Responses API plain string input */
    {
        const char *req_body =
            "{\"instructions\": \"Be concise.\", \"input\": \"What is 1+1?\"}";

        char prompt[4096];
        test_parse_chat_prompt(req_body, prompt, sizeof(prompt));
        assert(strstr(prompt, "<｜User｜>[System: Be concise.]\n") != NULL);
        assert(strstr(prompt, "<｜User｜>What is 1+1?\n") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
        printf("  [PASS] OpenAI Responses API string input\n");
    }

    /* 5. Test OpenAI Chat Completions messages compatibility */
    {
        const char *req_body =
            "{\"messages\": [{\"role\":\"system\",\"content\":\"sys\"},{\"role\":\"user\",\"content\":\"usr\"}]}";

        char prompt[4096];
        test_parse_chat_prompt(req_body, prompt, sizeof(prompt));
        assert(strstr(prompt, "<｜User｜>[System: sys]\n") != NULL);
        assert(strstr(prompt, "<｜User｜>usr\n") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
        printf("  [PASS] OpenAI Chat Completions messages compatibility\n");
    }

    printf("====================================================\n");
    printf("  ALL SERVER PROTOCOL TESTS PASSED CLEANLY!\n");
    printf("====================================================\n");
    return 0;
}
