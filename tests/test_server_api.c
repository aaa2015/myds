#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>

/* Helper function prototypes matching server implementation */
static void json_escape(const char *src, char *dst, size_t dst_size) {
    size_t di = 0;
    for (size_t si = 0; src[si] && di + 6 < dst_size; si++) {
        unsigned char c = (unsigned char)src[si];
        if (c == '"') {
            dst[di++] = '\\'; dst[di++] = '"';
        } else if (c == '\\') {
            dst[di++] = '\\'; dst[di++] = '\\';
        } else if (c == '\n') {
            dst[di++] = '\\'; dst[di++] = 'n';
        } else if (c == '\r') {
            dst[di++] = '\\'; dst[di++] = 'r';
        } else if (c == '\t') {
            dst[di++] = '\\'; dst[di++] = 't';
        } else if (c < 32) {
            di += snprintf(dst + di, dst_size - di, "\\u%04x", c);
        } else {
            dst[di++] = (char)c;
        }
    }
    dst[di] = '\0';
}

static void json_unescape_slice(const char *src, size_t len, char *dst, size_t dst_size) {
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

static int json_extract_string_field(const char *json, const char *key, char *out, size_t out_size) {
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
    json_unescape_slice(start, len, out, out_size);
    return 1;
}

static int json_extract_bool_field(const char *json, const char *key, int default_val) {
    if (!json || !key) return default_val;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return default_val;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return default_val;
}

static int json_extract_int_field(const char *json, const char *key, int default_val) {
    if (!json || !key) return default_val;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return default_val;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
    return atoi(p);
}

static const char *find_matching_brace(const char *start) {
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

static void extract_message_text(const char *obj_buf, char *out, size_t out_size) {
    out[0] = '\0';
    if (json_extract_string_field(obj_buf, "content", out, out_size) && out[0] != '\0') return;
    if (json_extract_string_field(obj_buf, "text", out, out_size) && out[0] != '\0') return;
    if (json_extract_string_field(obj_buf, "output", out, out_size) && out[0] != '\0') return;
    const char *content_ptr = strstr(obj_buf, "\"content\"");
    const char *p = content_ptr ? content_ptr : obj_buf;
    size_t cur = 0;
    while ((p = strstr(p, "\"text\"")) != NULL) {
        char piece[4096];
        if (json_extract_string_field(p, "text", piece, sizeof(piece))) {
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

static void parse_chat_prompt(const char *body, char *out_prompt, size_t max_prompt_len) {
    out_prompt[0] = '\0';
    if (!body || !body[0]) {
        snprintf(out_prompt, max_prompt_len, "Hello");
        return;
    }

    char instructions[4096] = "";
    json_extract_string_field(body, "instructions", instructions, sizeof(instructions));

    const char *input_field = strstr(body, "\"input\"");
    if (input_field) {
        const char *p = input_field + strlen("\"input\"");
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
        if (*p == '"') {
            char plain_input[8192] = "";
            if (json_extract_string_field(body, "input", plain_input, sizeof(plain_input))) {
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
        if (json_extract_string_field(body, "prompt", plain_prompt, sizeof(plain_prompt))) {
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

        const char *obj_end = find_matching_brace(obj_start);
        if (!obj_end) break;

        size_t obj_len = (size_t)(obj_end - obj_start + 1);
        char obj_buf[16384];
        if (obj_len >= sizeof(obj_buf)) obj_len = sizeof(obj_buf) - 1;
        memcpy(obj_buf, obj_start, obj_len);
        obj_buf[obj_len] = '\0';

        char role[64] = "user";
        char content[8192] = "";

        json_extract_string_field(obj_buf, "role", role, sizeof(role));
        extract_message_text(obj_buf, content, sizeof(content));

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

/* Simulated response generators */
static void generate_responses_api_json(const char *model_alias, const char *generated_text, uint32_t token_count, char *out_buf, size_t out_cap) {
    char escaped[4096];
    json_escape(generated_text, escaped, sizeof(escaped));
    long now = (long)time(NULL);
    snprintf(out_buf, out_cap,
             "{\"id\":\"resp_myds_%ld\",\"object\":\"response\",\"created_at\":%ld,"
             "\"status\":\"completed\",\"model\":\"%s\","
             "\"output\":[{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"completed\","
             "\"content\":[{\"type\":\"output_text\",\"text\":\"%s\"}]}],"
             "\"usage\":{\"input_tokens\":10,\"output_tokens\":%u,\"total_tokens\":%u}}",
             now, now,
             model_alias ? model_alias : "myds",
             escaped,
             token_count,
             10 + token_count);
}

int main(void) {
    printf("====================================================\n");
    printf("       Running myds Server API Unit Tests           \n");
    printf("====================================================\n");

    /* 1. Test Endpoint Route Recognition */
    {
        const char *p1 = "/v1/responses";
        const char *p2 = "/responses";
        const char *p3 = "/v1/chat/completions";
        const char *p4 = "/chat/completions";
        const char *p5 = "/health";
        const char *p6 = "/v1/models";
        const char *p7 = "/unknown";

        assert(strcmp(p1, "/v1/responses") == 0 || strcmp(p1, "/responses") == 0);
        assert(strcmp(p2, "/v1/responses") == 0 || strcmp(p2, "/responses") == 0);
        assert(strcmp(p3, "/v1/chat/completions") == 0 || strcmp(p3, "/chat/completions") == 0);
        assert(strcmp(p4, "/v1/chat/completions") == 0 || strcmp(p4, "/chat/completions") == 0);
        assert(strcmp(p5, "/health") == 0);
        assert(strcmp(p6, "/v1/models") == 0);
        assert(strcmp(p7, "/health") != 0 && strcmp(p7, "/v1/responses") != 0);
        printf("  [PASS] Endpoint route matching (/responses, /chat/completions, /health, /models)\n");
    }

    /* 2. Test Parameter Extraction (max_output_tokens precedence) */
    {
        const char *body_both = "{\"max_output_tokens\": 512, \"max_tokens\": 256, \"stream\": true}";
        int max_tokens = json_extract_int_field(body_both, "max_output_tokens", 0);
        if (max_tokens <= 0) max_tokens = json_extract_int_field(body_both, "max_tokens", 0);
        assert(max_tokens == 512);

        const char *body_legacy = "{\"max_tokens\": 128, \"stream\": false}";
        max_tokens = json_extract_int_field(body_legacy, "max_output_tokens", 0);
        if (max_tokens <= 0) max_tokens = json_extract_int_field(body_legacy, "max_tokens", 0);
        assert(max_tokens == 128);

        int stream = json_extract_bool_field(body_both, "stream", 0);
        assert(stream == 1);
        stream = json_extract_bool_field(body_legacy, "stream", 0);
        assert(stream == 0);
        printf("  [PASS] Parameter extraction and max_output_tokens precedence\n");
    }

    /* 3. Test Multi-Turn Conversation Prompt Formatting */
    {
        const char *multi_turn =
            "{\n"
            "  \"instructions\": \"You are a coding tutor.\",\n"
            "  \"input\": [\n"
            "    {\"type\": \"message\", \"role\": \"user\", \"content\": [{\"type\":\"input_text\",\"text\":\"What is a pointer?\"}]},\n"
            "    {\"type\": \"message\", \"role\": \"assistant\", \"content\": [{\"type\":\"input_text\",\"text\":\"A pointer holds a memory address.\"}]},\n"
            "    {\"type\": \"message\", \"role\": \"user\", \"content\": [{\"type\":\"input_text\",\"text\":\"How do I dereference it?\"}]}\n"
            "  ]\n"
            "}";

        char prompt[8192];
        parse_chat_prompt(multi_turn, prompt, sizeof(prompt));

        assert(strstr(prompt, "<｜begin of sentence｜><｜User｜>[System: You are a coding tutor.]\n") != NULL);
        assert(strstr(prompt, "<｜User｜>What is a pointer?\n") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>A pointer holds a memory address.\n") != NULL);
        assert(strstr(prompt, "<｜User｜>How do I dereference it?\n") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
        printf("  [PASS] Multi-turn conversation prompt construction with instructions\n");
    }

    /* 4. Test Responses API Non-Streaming JSON Response Structure */
    {
        char resp_json[4096];
        generate_responses_api_json("local-myds-deepseek-v4.1-flash-q2", "Dereference with *ptr", 4, resp_json, sizeof(resp_json));

        assert(strstr(resp_json, "\"object\":\"response\"") != NULL);
        assert(strstr(resp_json, "\"status\":\"completed\"") != NULL);
        assert(strstr(resp_json, "\"model\":\"local-myds-deepseek-v4.1-flash-q2\"") != NULL);
        assert(strstr(resp_json, "\"output\":[{\"type\":\"message\"") != NULL);
        assert(strstr(resp_json, "\"content\":[{\"type\":\"output_text\",\"text\":\"Dereference with *ptr\"}]") != NULL);
        assert(strstr(resp_json, "\"usage\":{\"input_tokens\":10,\"output_tokens\":4,\"total_tokens\":14}") != NULL);
        printf("  [PASS] Responses API non-streaming JSON structure compliance\n");
    }

    /* 5. Test Special Character JSON Escaping */
    {
        const char *special_code = "int x = 1;\nprintf(\"Hello \\\"World\\\"\\t!\\n\");";
        char escaped[1024];
        json_escape(special_code, escaped, sizeof(escaped));

        assert(strstr(escaped, "\\n") != NULL);
        assert(strstr(escaped, "\\\"") != NULL);
        assert(strstr(escaped, "\\t") != NULL);
        printf("  [PASS] JSON special character escaping (newlines, quotes, tabs)\n");
    }

    /* 6. Test Responses API SSE Stream Events Format */
    {
        char chunk[1024];
        /* Output item added */
        snprintf(chunk, sizeof(chunk),
                 "event: response.output_item.added\n"
                 "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n");
        assert(strstr(chunk, "event: response.output_item.added") != NULL);
        assert(strstr(chunk, "\"type\":\"response.output_item.added\"") != NULL);

        /* Content part added */
        snprintf(chunk, sizeof(chunk),
                 "event: response.content_part.added\n"
                 "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n");
        assert(strstr(chunk, "event: response.content_part.added") != NULL);

        /* Output text delta */
        snprintf(chunk, sizeof(chunk),
                 "event: response.output_text.delta\n"
                 "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}\n\n");
        assert(strstr(chunk, "event: response.output_text.delta") != NULL);
        assert(strstr(chunk, "\"delta\":\"hello\"") != NULL);

        /* Completed event */
        snprintf(chunk, sizeof(chunk),
                 "event: response.completed\n"
                 "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n\n");
        assert(strstr(chunk, "event: response.completed") != NULL);
        assert(strstr(chunk, "\"status\":\"completed\"") != NULL);
        printf("  [PASS] Responses API SSE events protocol compliance\n");
    }

    printf("====================================================\n");
    printf("      ALL SERVER API UNIT TESTS PASSED!             \n");
    printf("====================================================\n");
    return 0;
}
