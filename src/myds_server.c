#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static volatile sig_atomic_t g_running = 1;
static int g_server_fd = -1;
static pthread_mutex_t g_infer_mutex = PTHREAD_MUTEX_INITIALIZER;

static void handle_signal(int sig) {
    (void)sig;
    g_running = 0;
    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }
}

/* Helper to JSON-escape a string */
void json_escape(const char *src, char *dst, size_t dst_size) {
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

/* Helper to unescape JSON string slice into normal text */
void json_unescape_slice(const char *src, size_t len, char *dst, size_t dst_size) {
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

/* Standard HTTP header integer parser (e.g. Content-Length: 1234) */
int extract_http_header_int(const char *headers, const char *header_name) {
    if (!headers || !header_name) return 0;
    const char *p = headers;
    size_t name_len = strlen(header_name);
    while (*p) {
        if (strncasecmp(p, header_name, name_len) == 0 && p[name_len] == ':') {
            p += name_len + 1;
            while (*p == ' ' || *p == '\t') p++;
            return atoi(p);
        }
        p = strchr(p, '\n');
        if (!p) break;
        p++;
    }
    return 0;
}

/* Very simple JSON string value finder: searches for "key"\s*:\s*"value" */
int json_extract_string_field(const char *json, const char *key, char *out, size_t out_size) {
    if (!json || !key || !out || out_size == 0) return 0;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
    if (*p != '"') return 0;
    p++; /* skip opening quote */
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

/* Search for boolean field: "key"\s*:\s*(true|false) */
int json_extract_bool_field(const char *json, const char *key, int default_val) {
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

/* Search for number field */
int json_extract_int_field(const char *json, const char *key, int default_val) {
    if (!json || !key) return default_val;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return default_val;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
    return atoi(p);
}

float json_extract_float_field(const char *json, const char *key, float default_val) {
    if (!json || !key) return default_val;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return default_val;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
    return (float)atof(p);
}

/* Find matching closing brace '}' considering nested braces and string literals */
const char *find_matching_brace(const char *start) {
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

/* Helper to extract text from a message JSON object string.
 * Handles:
 *  - "content": "text"
 *  - "content": [{"type":"input_text","text":"text"}]
 *  - "text": "text"
 *  - "output": "text" (for function_call_output items)
 */
void extract_message_text(const char *obj_buf, char *out, size_t out_size) {
    out[0] = '\0';
    if (json_extract_string_field(obj_buf, "content", out, out_size) && out[0] != '\0') {
        return;
    }
    if (json_extract_string_field(obj_buf, "text", out, out_size) && out[0] != '\0') {
        return;
    }
    if (json_extract_string_field(obj_buf, "output", out, out_size) && out[0] != '\0') {
        return;
    }
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

/* Parse chat completion messages or Responses API input and format into DeepSeek prompt */
void parse_chat_prompt(const char *body, char *out_prompt, size_t max_prompt_len) {
    out_prompt[0] = '\0';
    if (!body || !body[0]) {
        snprintf(out_prompt, max_prompt_len, "Hello");
        return;
    }

    char instructions[4096] = "";
    json_extract_string_field(body, "instructions", instructions, sizeof(instructions));

    /* Check if "input" is a direct string: e.g. "input": "hello" */
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

    /* Check for array in "messages" or "input" */
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
        /* Find start of next message object '{' */
        const char *obj_start = strchr(p, '{');
        if (!obj_start) break;

        /* If there's a ']' before '{', we reached end of array */
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

    /* Final assistant prefix */
    cur_len += snprintf(out_prompt + cur_len, max_prompt_len - cur_len, "<｜Assistant｜>");
}

/* Chat completion streaming context */
typedef struct {
    int client_fd;
    const char *model_alias;
    uint32_t token_count;
} sse_stream_ctx_t;

static void sse_token_callback(const char *piece, int32_t token_id, void *user_data) {
    (void)token_id;
    sse_stream_ctx_t *sctx = (sse_stream_ctx_t *)user_data;
    sctx->token_count++;

    char escaped[2048];
    json_escape(piece, escaped, sizeof(escaped));

    char chunk[2560];
    int len = snprintf(chunk, sizeof(chunk),
                       "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                       escaped);
    send(sctx->client_fd, chunk, len, 0);
}

/* Responses API streaming context */
typedef struct {
    int client_fd;
    const char *model_alias;
    uint32_t token_count;
} responses_sse_stream_ctx_t;

static void responses_sse_token_callback(const char *piece, int32_t token_id, void *user_data) {
    (void)token_id;
    responses_sse_stream_ctx_t *sctx = (responses_sse_stream_ctx_t *)user_data;
    sctx->token_count++;

    char escaped[2048];
    json_escape(piece, escaped, sizeof(escaped));

    char chunk[3072];
    int len = snprintf(chunk, sizeof(chunk),
                       "event: response.output_text.delta\n"
                       "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"%s\"}\n\n",
                       escaped);
    send(sctx->client_fd, chunk, len, 0);
}

/* Non-streaming buffer context */
typedef struct {
    char *buf;
    size_t cap;
    size_t len;
    uint32_t token_count;
} non_stream_ctx_t;

static void non_stream_callback(const char *piece, int32_t token_id, void *user_data) {
    (void)token_id;
    non_stream_ctx_t *ns = (non_stream_ctx_t *)user_data;
    ns->token_count++;
    size_t plen = strlen(piece);
    if (ns->len + plen + 1 > ns->cap) {
        ns->cap = (ns->cap + plen + 4096) * 2;
        ns->buf = realloc(ns->buf, ns->cap);
    }
    memcpy(ns->buf + ns->len, piece, plen);
    ns->len += plen;
    ns->buf[ns->len] = '\0';
}

static void send_response(int fd, int code, const char *status, const char *content_type, const char *body) {
    size_t body_len = body ? strlen(body) : 0;
    char header[512];
    int hlen = snprintf(header, sizeof(header),
                        "HTTP/1.1 %d %s\r\n"
                        "Content-Type: %s\r\n"
                        "Content-Length: %zu\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: *\r\n"
                        "Connection: close\r\n\r\n",
                        code, status, content_type, body_len);
    send(fd, header, hlen, 0);
    if (body_len > 0) {
        send(fd, body, body_len, 0);
    }
}

static void handle_client(int client_fd, myds_context_t *ctx, const char *model_alias, const char *client_ip, int client_port) {
    /* Protect against slowloris and stalled connections */
    struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    size_t req_cap = 65536;
    char *req_buf = malloc(req_cap);
    if (!req_buf) {
        MYDS_LOG_E("[%s:%d] Failed to allocate %zu bytes for request buffer", client_ip, client_port, req_cap);
        close(client_fd);
        return;
    }
    ssize_t total_read = 0;

    /* Read HTTP headers */
    while (total_read < (ssize_t)req_cap - 1) {
        ssize_t n = recv(client_fd, req_buf + total_read, req_cap - 1 - total_read, 0);
        if (n <= 0) break;
        total_read += n;
        req_buf[total_read] = '\0';
        if (strstr(req_buf, "\r\n\r\n")) break;
    }

    if (total_read <= 0) {
        free(req_buf);
        close(client_fd);
        return;
    }

    char method[16] = {0};
    char path[256] = {0};
    sscanf(req_buf, "%15s %255s", method, path);

    /* CORS preflight */
    if (strcmp(method, "OPTIONS") == 0) {
        MYDS_LOG_D("[%s:%d] OPTIONS %s (CORS preflight)", client_ip, client_port, path);
        const char *cors_res =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: *\r\n"
            "Connection: close\r\n\r\n";
        send(client_fd, cors_res, strlen(cors_res), 0);
        free(req_buf);
        close(client_fd);
        return;
    }

    /* GET /health or GET / */
    if (strcmp(path, "/health") == 0 || strcmp(path, "/") == 0) {
        MYDS_LOG_D("[%s:%d] GET %s -> 200 OK", client_ip, client_port, path);
        send_response(client_fd, 200, "OK", "application/json", "{\"status\":\"ok\"}");
        free(req_buf);
        close(client_fd);
        return;
    }

    /* GET /v1/models */
    if (strcmp(path, "/v1/models") == 0) {
        MYDS_LOG_I("[%s:%d] GET /v1/models -> 200 OK (alias: %s)", client_ip, client_port, model_alias ? model_alias : "myds");
        char resp[512];
        snprintf(resp, sizeof(resp),
                 "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"myds\"}]}",
                 model_alias ? model_alias : "myds");
        send_response(client_fd, 200, "OK", "application/json", resp);
        free(req_buf);
        close(client_fd);
        return;
    }

    /* POST /v1/chat/completions or /v1/responses */
    int is_chat_endpoint = (strcmp(path, "/v1/chat/completions") == 0 || strcmp(path, "/chat/completions") == 0);
    int is_responses_endpoint = (strcmp(path, "/v1/responses") == 0 || strcmp(path, "/responses") == 0);

    if (is_chat_endpoint || is_responses_endpoint) {
        struct timeval t_start, t_end;
        gettimeofday(&t_start, NULL);

        /* Find body start and read complete body */
        char *body = strstr(req_buf, "\r\n\r\n");
        if (body) {
            body += 4;
            int content_len = extract_http_header_int(req_buf, "Content-Length");
            size_t header_len = (size_t)(body - req_buf);
            size_t needed = header_len + (content_len > 0 ? (size_t)content_len : 0) + 1;
            if (needed > req_cap) {
                req_cap = needed + 4096;
                char *new_buf = realloc(req_buf, req_cap);
                if (new_buf) {
                    req_buf = new_buf;
                    body = req_buf + header_len;
                }
            }
            ssize_t body_have = total_read - (ssize_t)header_len;
            while (body_have < content_len && total_read < (ssize_t)req_cap - 1) {
                ssize_t n = recv(client_fd, req_buf + total_read, req_cap - 1 - total_read, 0);
                if (n <= 0) break;
                total_read += n;
                req_buf[total_read] = '\0';
                body_have += n;
            }
        } else {
            body = "";
        }

        int stream = json_extract_bool_field(body, "stream", 0);
        int max_tokens = json_extract_int_field(body, "max_output_tokens", 0);
        if (max_tokens <= 0) {
            max_tokens = json_extract_int_field(body, "max_tokens", 0);
        }
        float temp = json_extract_float_field(body, "temperature", -1.0f);
        float top_p = json_extract_float_field(body, "top_p", -1.0f);

        size_t prompt_cap = (total_read * 2 > 65536) ? (total_read * 2 + 8192) : 65536;
        char *prompt = malloc(prompt_cap);
        if (!prompt) {
            MYDS_LOG_E("[%s:%d] Out of memory allocating prompt buffer (%zu bytes)", client_ip, client_port, prompt_cap);
            send_response(client_fd, 500, "Internal Server Error", "application/json", "{\"error\":\"Out of memory\"}");
            free(req_buf);
            close(client_fd);
            return;
        }
        parse_chat_prompt(body, prompt, prompt_cap);

        MYDS_LOG_I("[%s:%d] %s %s (protocol=%s, stream=%s, max_tokens=%d, temp=%.2f, top_p=%.2f, prompt_chars=%zu)",
                   client_ip, client_port, method, path,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   stream ? "true" : "false",
                   max_tokens, temp, top_p,
                   strlen(prompt));

        pthread_mutex_lock(&g_infer_mutex);

        /* Apply request configuration overrides if provided */
        myds_config_t prev_cfg = myds_context_get_config(ctx);
        myds_config_t req_cfg = prev_cfg;
        if (max_tokens > 0) req_cfg.max_new_tokens = (uint32_t)max_tokens;
        if (temp >= 0.0f) req_cfg.temperature = temp;
        if (top_p >= 0.0f) req_cfg.top_p = top_p;
        myds_context_set_config(ctx, &req_cfg);

        uint32_t final_tokens = 0;

        if (is_responses_endpoint) {
            /* OpenAI Responses API format */
            if (stream) {
                const char *sse_header =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Connection: close\r\n\r\n";
                send(client_fd, sse_header, strlen(sse_header), 0);

                const char *ev_item_added =
                    "event: response.output_item.added\n"
                    "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n";
                send(client_fd, ev_item_added, strlen(ev_item_added), 0);

                const char *ev_part_added =
                    "event: response.content_part.added\n"
                    "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n";
                send(client_fd, ev_part_added, strlen(ev_part_added), 0);

                responses_sse_stream_ctx_t sctx = {
                    .client_fd = client_fd,
                    .model_alias = model_alias ? model_alias : "myds",
                    .token_count = 0,
                };

                myds_generate(ctx, prompt, responses_sse_token_callback, &sctx);
                final_tokens = sctx.token_count;

                const char *ev_part_done =
                    "event: response.content_part.done\n"
                    "data: {\"type\":\"response.content_part.done\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n";
                send(client_fd, ev_part_done, strlen(ev_part_done), 0);

                const char *ev_item_done =
                    "event: response.output_item.done\n"
                    "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]}}\n\n";
                send(client_fd, ev_item_done, strlen(ev_item_done), 0);

                char ev_completed[1024];
                int clen = snprintf(ev_completed, sizeof(ev_completed),
                                    "event: response.completed\n"
                                    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_myds\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"%s\",\"usage\":{\"input_tokens\":10,\"output_tokens\":%u,\"total_tokens\":%u}}}\n\n",
                                    model_alias ? model_alias : "myds",
                                    sctx.token_count,
                                    10 + sctx.token_count);
                send(client_fd, ev_completed, clen, 0);
            } else {
                non_stream_ctx_t ns = {
                    .cap = 4096,
                    .len = 0,
                    .token_count = 0,
                };
                ns.buf = malloc(ns.cap);
                ns.buf[0] = '\0';

                myds_generate(ctx, prompt, non_stream_callback, &ns);
                final_tokens = ns.token_count;

                char *escaped = malloc(ns.len * 2 + 1024);
                json_escape(ns.buf, escaped, ns.len * 2 + 1024);

                size_t resp_cap = strlen(escaped) + 2048;
                char *resp_json = malloc(resp_cap);
                long now = (long)time(NULL);
                snprintf(resp_json, resp_cap,
                         "{\"id\":\"resp_myds_%ld\",\"object\":\"response\",\"created_at\":%ld,"
                         "\"status\":\"completed\",\"model\":\"%s\","
                         "\"output\":[{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"completed\","
                         "\"content\":[{\"type\":\"output_text\",\"text\":\"%s\"}]}],"
                         "\"usage\":{\"input_tokens\":10,\"output_tokens\":%u,\"total_tokens\":%u}}",
                         now, now,
                         model_alias ? model_alias : "myds",
                         escaped,
                         ns.token_count,
                         10 + ns.token_count);

                send_response(client_fd, 200, "OK", "application/json", resp_json);

                free(resp_json);
                free(escaped);
                free(ns.buf);
            }
        } else {
            /* Chat Completions API format */
            if (stream) {
                const char *sse_header =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Connection: close\r\n\r\n";
                send(client_fd, sse_header, strlen(sse_header), 0);

                char initial_chunk[256];
                snprintf(initial_chunk, sizeof(initial_chunk),
                         "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n");
                send(client_fd, initial_chunk, strlen(initial_chunk), 0);

                sse_stream_ctx_t sctx = {
                    .client_fd = client_fd,
                    .model_alias = model_alias ? model_alias : "myds",
                    .token_count = 0,
                };

                myds_generate(ctx, prompt, sse_token_callback, &sctx);
                final_tokens = sctx.token_count;

                const char *stop_chunk =
                    "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                    "data: [DONE]\n\n";
                send(client_fd, stop_chunk, strlen(stop_chunk), 0);
            } else {
                non_stream_ctx_t ns = {
                    .cap = 4096,
                    .len = 0,
                    .token_count = 0,
                };
                ns.buf = malloc(ns.cap);
                ns.buf[0] = '\0';

                myds_generate(ctx, prompt, non_stream_callback, &ns);
                final_tokens = ns.token_count;

                char *escaped = malloc(ns.len * 2 + 1024);
                json_escape(ns.buf, escaped, ns.len * 2 + 1024);

                size_t resp_cap = strlen(escaped) + 1024;
                char *resp_json = malloc(resp_cap);
                snprintf(resp_json, resp_cap,
                         "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                         "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
                         "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                         time(NULL),
                         model_alias ? model_alias : "myds",
                         escaped,
                         ns.token_count,
                         10 + ns.token_count);

                send_response(client_fd, 200, "OK", "application/json", resp_json);

                free(resp_json);
                free(escaped);
                free(ns.buf);
            }
        }

        gettimeofday(&t_end, NULL);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_usec - t_start.tv_usec) / 1e6;
        double speed = (elapsed > 0.0) ? (final_tokens / elapsed) : 0.0;
        MYDS_LOG_I("[%s:%d] Completed %s %s 200 OK - %u tokens in %.2fs (%.2f tokens/s)",
                   client_ip, client_port,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   path, final_tokens, elapsed, speed);

        myds_context_set_config(ctx, &prev_cfg);
        pthread_mutex_unlock(&g_infer_mutex);

        free(prompt);
        free(req_buf);
        close(client_fd);
        return;
    }

    /* Not found */
    MYDS_LOG_W("[%s:%d] %s %s -> 404 Not Found", client_ip, client_port, method, path);
    send_response(client_fd, 404, "Not Found", "application/json", "{\"error\":\"Not found\"}");
    free(req_buf);
    close(client_fd);
}

int myds_server_start(myds_context_t *ctx, const char *host, int port, const char *model_alias) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        MYDS_LOG_E("Failed to create server socket: %s", strerror(errno));
        return 1;
    }

    int opt = 1;
    setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (!host || strcmp(host, "0.0.0.0") == 0 || strlen(host) == 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, host, &addr.sin_addr);
    }

    if (bind(g_server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        MYDS_LOG_E("Failed to bind socket to %s:%d: %s", host ? host : "0.0.0.0", port, strerror(errno));
        close(g_server_fd);
        return 1;
    }

    if (listen(g_server_fd, 64) < 0) {
        MYDS_LOG_E("Failed to listen on socket: %s", strerror(errno));
        close(g_server_fd);
        return 1;
    }

    /* Print exactly matching start_fleet.sh grep regex:
     * grep -qiE "listening on http|main: server is listening|running on http|diffusion-gemma.*ready|listening on: [0-9]+|started!"
     */
    printf("myds: listening on http://%s:%d (alias: %s)\n",
           (host && strlen(host) > 0) ? host : "0.0.0.0",
           port,
           model_alias ? model_alias : "myds");
    fflush(stdout);

    MYDS_LOG_I("HTTP server listening on http://%s:%d (model_alias: %s)",
               (host && strlen(host) > 0) ? host : "0.0.0.0",
               port,
               model_alias ? model_alias : "myds");
    MYDS_LOG_I("Protocols active: OpenAI Responses API (/v1/responses) & Chat Completions (/v1/chat/completions)");

    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(g_server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (!g_running) break;
            if (errno == EINTR) continue;
            MYDS_LOG_E("Socket accept error: %s", strerror(errno));
            break;
        }

        char client_ip[INET_ADDRSTRLEN] = "127.0.0.1";
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        int client_port = ntohs(client_addr.sin_port);

        handle_client(client_fd, ctx, model_alias, client_ip, client_port);
    }

    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }

    printf("\nmyds: server stopped gracefully\n");
    fflush(stdout);
    MYDS_LOG_I("Server stopped gracefully");
    return 0;
}
