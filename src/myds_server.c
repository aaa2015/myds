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
#include <stdarg.h>
#include <limits.h>
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

static bool append_json_bytes(char *dst, size_t cap, size_t *used, const char *bytes, size_t length) {
    if (*used >= cap || length >= cap - *used) return false;
    memcpy(dst + *used, bytes, length);
    *used += length;
    return true;
}

/* The pending bytes are a valid UTF-8 prefix. Keep it across token boundaries,
 * because a byte-BPE token can end in the middle of a Unicode character. */
static bool escape_utf8_chunk(myds_utf8_state_t *state, const char *src, size_t length,
                              bool final, char *dst, size_t cap, size_t *out_length) {
    if (!dst || cap == 0 || (!src && length)) return false;
    myds_utf8_state_t next = *state;
    size_t used = 0;
    const char replacement[] = "\xef\xbf\xbd";
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < length;) {
        unsigned char c = (unsigned char)src[i];
        if (next.pending_length != 0) {
            unsigned char lead = next.pending[0];
            bool continuation = (c >= 0x80 && c <= 0xbf);
            if (next.pending_length == 1) {
                if (lead == 0xe0 && c < 0xa0) continuation = false;
                if (lead == 0xed && c >= 0xa0) continuation = false;
                if (lead == 0xf0 && c < 0x90) continuation = false;
                if (lead == 0xf4 && c >= 0x90) continuation = false;
            }
            if (!continuation) {
                if (!append_json_bytes(dst, cap, &used, replacement, 3)) goto fail;
                next.pending_length = 0;
                continue; /* Reconsider this byte as a new character. */
            }
            next.pending[next.pending_length++] = c;
            i++;
            size_t expected = lead < 0xe0 ? 2 : (lead < 0xf0 ? 3 : 4);
            if (next.pending_length == expected) {
                if (!append_json_bytes(dst, cap, &used, (const char *)next.pending, expected)) goto fail;
                next.pending_length = 0;
            }
            continue;
        }
        i++;
        if (c >= 0xc2 && c <= 0xf4) {
            next.pending[0] = c;
            next.pending_length = 1;
        } else if (c >= 0x80) {
            if (!append_json_bytes(dst, cap, &used, replacement, 3)) goto fail;
        } else if (c == '"' || c == '\\') {
            char escaped[] = {'\\', (char)c};
            if (!append_json_bytes(dst, cap, &used, escaped, 2)) goto fail;
        } else if (c == '\n' || c == '\r' || c == '\t') {
            char escaped[] = {'\\', c == '\n' ? 'n' : (c == '\r' ? 'r' : 't')};
            if (!append_json_bytes(dst, cap, &used, escaped, 2)) goto fail;
        } else if (c < 32) {
            char escaped[] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
            if (!append_json_bytes(dst, cap, &used, escaped, 6)) goto fail;
        } else {
            if (!append_json_bytes(dst, cap, &used, (const char *)&c, 1)) goto fail;
        }
    }
    if (final && next.pending_length) {
        if (!append_json_bytes(dst, cap, &used, replacement, 3)) goto fail;
        next.pending_length = 0;
    }
    dst[used] = '\0';
    *state = next;
    if (out_length) *out_length = used;
    return true;
fail:
    dst[0] = '\0';
    if (out_length) *out_length = 0;
    return false;
}

bool json_escape_bytes(const char *src, size_t length, char *dst, size_t dst_size, size_t *out_length) {
    myds_utf8_state_t state = {0};
    return escape_utf8_chunk(&state, src, length, true, dst, dst_size, out_length);
}

void json_escape(const char *src, char *dst, size_t dst_size) {
    (void)json_escape_bytes(src, src ? strlen(src) : 0, dst, dst_size, NULL);
}

static char *json_escape_alloc(const char *src, size_t length) {
    if (length > (SIZE_MAX - 1) / 6) return NULL;
    size_t cap = length * 6 + 1;
    char *escaped = malloc(cap);
    if (escaped && !json_escape_bytes(src, length, escaped, cap, NULL)) {
        free(escaped);
        return NULL;
    }
    return escaped;
}

static char *format_alloc(const char *format, ...) {
    va_list args, copy;
    va_start(args, format);
    va_copy(copy, args);
    int length = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    char *result = length < 0 ? NULL : malloc((size_t)length + 1);
    if (result) vsnprintf(result, (size_t)length + 1, format, args);
    va_end(args);
    return result;
}

bool myds_server_send_all(int fd, const char *data, size_t length) {
    while (length) {
        size_t part = length > (size_t)SSIZE_MAX ? (size_t)SSIZE_MAX : length;
        ssize_t sent = send(fd, data, part, 0);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return false;
        data += (size_t)sent;
        length -= (size_t)sent;
    }
    return true;
}

uint32_t myds_server_default_max_tokens(uint32_t configured_max_tokens) {
    /* Bound omitted API limits even when the CLI server configuration permits
     * very long requests. Explicit request limits and CLI generation retain
     * their existing behavior; the engine enforces the context capacity. */
    return configured_max_tokens > 4096 ? 4096 : configured_max_tokens;
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

/* Text requests use the released DeepSeek V4.1 non-thinking chat format.
 * Match ds4's render_deepseek41_chat_ordered: role markers carry the boundaries,
 * assistant history ends with EOS, and only adjacent user turns gain "\n\n".
 * Keep JSON parsing structural so nested fields cannot impersonate top-level
 * input and reject malformed/oversized data instead of truncating a prompt. */
typedef enum { PJ_OBJECT, PJ_ARRAY, PJ_STRING, PJ_NULL, PJ_SCALAR } prompt_json_kind_t;
typedef struct {
    prompt_json_kind_t kind; char *text; size_t next;
    size_t off, len;   /* 原始 JSON 的字节区间（相对 base）；用于原样抽出工具 schema */
} prompt_json_node_t;
typedef struct {
    const char *cursor;
    const char *base;  /* 请求体起点，配合 off/len 取原始切片 */
    prompt_json_node_t *nodes;
    size_t count, capacity;
} prompt_json_t;
/* overflow 用来把【缓冲不够】与【形状不支持】区分开：两者原本都只是 return false，
 * 于是一个「工具调用被拒」和一个「提示词太大」会给出完全相同的报错。 */
typedef struct { char *data; size_t capacity, length; bool overflow; } prompt_buffer_t;

static void prompt_json_ws(prompt_json_t *j) {
    while (*j->cursor == ' ' || *j->cursor == '\t' || *j->cursor == '\r' || *j->cursor == '\n') ++j->cursor;
}
static int prompt_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static bool prompt_hex4(const char **p, const char *end, uint32_t *cp) {
    if (end - *p < 4) return false;
    *cp = 0;
    for (int i = 0; i < 4; ++i) {
        int digit = prompt_hex(*(*p)++);
        if (digit < 0) return false;
        *cp = (*cp << 4) | (uint32_t)digit;
    }
    return true;
}
static char *prompt_json_string(prompt_json_t *j) {
    if (*j->cursor != '"') return NULL;
    const char *start = ++j->cursor, *end = start;
    while (*end && *end != '"') {
        if (*end == '\\') { ++end; if (!*end) return NULL; }
        ++end;
    }
    if (*end != '"') return NULL;
    char *decoded = malloc((size_t)(end - start) + 1);
    if (!decoded) return NULL;
    size_t used = 0;
    for (const char *p = start; p < end;) {
        unsigned char c = (unsigned char)*p++;
        if (c < 0x20) goto invalid;
        if (c == '\\') {
            if (p >= end) goto invalid;
            c = (unsigned char)*p++;
            if (c == 'u') {
                uint32_t cp;
                if (!prompt_hex4(&p, end, &cp)) goto invalid;
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    uint32_t low;
                    if (end - p < 6 || p[0] != '\\' || p[1] != 'u') goto invalid;
                    p += 2;
                    if (!prompt_hex4(&p, end, &low) || low < 0xdc00 || low > 0xdfff) goto invalid;
                    cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                } else if (cp >= 0xdc00 && cp <= 0xdfff) goto invalid;
                /* The engine prompt API is a C string. Reject U+0000 explicitly
                 * until that API accepts a byte count, rather than losing tail text. */
                if (!cp) goto invalid;
                if (cp < 0x80) decoded[used++] = (char)cp;
                else if (cp < 0x800) {
                    decoded[used++] = (char)(0xc0 | (cp >> 6));
                    decoded[used++] = (char)(0x80 | (cp & 63));
                } else if (cp < 0x10000) {
                    decoded[used++] = (char)(0xe0 | (cp >> 12));
                    decoded[used++] = (char)(0x80 | ((cp >> 6) & 63));
                    decoded[used++] = (char)(0x80 | (cp & 63));
                } else {
                    decoded[used++] = (char)(0xf0 | (cp >> 18));
                    decoded[used++] = (char)(0x80 | ((cp >> 12) & 63));
                    decoded[used++] = (char)(0x80 | ((cp >> 6) & 63));
                    decoded[used++] = (char)(0x80 | (cp & 63));
                }
                continue;
            }
            switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = '\b'; break; case 'f': c = '\f'; break;
                case 'n': c = '\n'; break; case 'r': c = '\r'; break; case 't': c = '\t'; break;
                default: goto invalid;
            }
        } else if (c >= 0x80) {
            size_t continuation = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : 0;
            if (!continuation || (size_t)(end - p) < continuation) goto invalid;
            uint32_t cp = c & (continuation == 1 ? 31 : continuation == 2 ? 15 : 7);
            for (size_t i = 0; i < continuation; ++i) {
                if (((unsigned char)p[i] & 0xc0) != 0x80) goto invalid;
                cp = (cp << 6) | ((unsigned char)p[i] & 63);
            }
            if ((continuation == 1 && cp < 0x80) || (continuation == 2 && cp < 0x800) ||
                (continuation == 3 && cp < 0x10000) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) goto invalid;
            decoded[used++] = (char)c;
            memcpy(decoded + used, p, continuation); used += continuation; p += continuation;
            continue;
        }
        decoded[used++] = (char)c;
    }
    decoded[used] = 0; j->cursor = end + 1; return decoded;
invalid:
    free(decoded); return NULL;
}
static bool prompt_json_value(prompt_json_t *j, unsigned depth) {
    if (depth > 64) return false;
    prompt_json_ws(j);
    if (j->count == j->capacity) {
        size_t capacity = j->capacity ? j->capacity * 2 : 32;
        if (capacity > SIZE_MAX / sizeof(*j->nodes)) return false;
        void *nodes = realloc(j->nodes, capacity * sizeof(*j->nodes));
        if (!nodes) return false;
        j->nodes = nodes; j->capacity = capacity;
    }
    size_t index = j->count++;
    j->nodes[index] = (prompt_json_node_t){.kind = PJ_SCALAR};
    if (j->base) j->nodes[index].off = (size_t)(j->cursor - j->base);
    char first = *j->cursor;
    if (first == '"') {
        j->nodes[index].kind = PJ_STRING;
        if (!(j->nodes[index].text = prompt_json_string(j))) return false;
    } else if (first == '{' || first == '[') {
        j->nodes[index].kind = first == '{' ? PJ_OBJECT : PJ_ARRAY;
        char close = first == '{' ? '}' : ']';
        ++j->cursor; prompt_json_ws(j);
        if (*j->cursor != close) for (;;) {
            if (first == '{') {
                prompt_json_ws(j);
                if (*j->cursor != '"' || !prompt_json_value(j, depth + 1)) return false;
                prompt_json_ws(j);
                if (*j->cursor++ != ':') return false;
            }
            if (!prompt_json_value(j, depth + 1)) return false;
            prompt_json_ws(j);
            if (*j->cursor != ',') break;
            ++j->cursor;
        }
        if (*j->cursor != close) return false;
        ++j->cursor;
    } else if (!strncmp(j->cursor, "null", 4)) {
        j->nodes[index].kind = PJ_NULL; j->cursor += 4;
    } else if (!strncmp(j->cursor, "true", 4)) j->cursor += 4;
    else if (!strncmp(j->cursor, "false", 5)) j->cursor += 5;
    else {
        const char *p = j->cursor;
        if (*p == '-') ++p;
        if (*p == '0') ++p;
        else { if (*p < '1' || *p > '9') return false; do { ++p; } while (*p >= '0' && *p <= '9'); }
        if (*p == '.') { ++p; if (*p < '0' || *p > '9') return false; do { ++p; } while (*p >= '0' && *p <= '9'); }
        if (*p == 'e' || *p == 'E') {
            ++p; if (*p == '+' || *p == '-') ++p;
            if (*p < '0' || *p > '9') return false;
            do { ++p; } while (*p >= '0' && *p <= '9');
        }
        j->cursor = p;
    }
    j->nodes[index].next = j->count;
    if (j->base) j->nodes[index].len = (size_t)(j->cursor - j->base) - j->nodes[index].off;
    return true;
}
/* A duplicate relevant field is ambiguous and fails the request. */
static const prompt_json_node_t *prompt_json_field(const prompt_json_t *j, const prompt_json_node_t *object,
                                                  const char *key, bool *valid) {
    if (!object || object->kind != PJ_OBJECT) { *valid = false; return NULL; }
    const prompt_json_node_t *found = NULL;
    size_t begin = (size_t)(object - j->nodes) + 1;
    for (size_t i = begin; i < object->next;) {
        const prompt_json_node_t *name = &j->nodes[i], *value = &j->nodes[name->next];
        if (!strcmp(name->text, key)) { if (found) *valid = false; found = value; }
        i = value->next;
    }
    return found;
}
/* This renderer handles text, not tool definitions or assistant invocations.
 * Fail explicitly rather than accepting a request with omitted semantics. */
static bool prompt_absent_or_empty(const prompt_json_t *j, const prompt_json_node_t *node) {
    return !node || node->kind == PJ_NULL ||
           (node->kind == PJ_ARRAY && (size_t)(node - j->nodes) + 1 == node->next);
}
/* 按长度追加：工具 schema 是原始 JSON 切片，不能当 C 字符串用。 */
static bool prompt_append_n(prompt_buffer_t *b, const char *text, size_t length) {
    if (b->length >= b->capacity || length >= b->capacity - b->length) { b->overflow = true; return false; }
    memcpy(b->data + b->length, text, length); b->length += length; b->data[b->length] = 0;
    return true;
}
static bool prompt_append(prompt_buffer_t *b, const char *text) {
    size_t length = strlen(text);
    if (b->length >= b->capacity || length >= b->capacity - b->length) { b->overflow = true; return false; }
    memcpy(b->data + b->length, text, length + 1); b->length += length; return true;
}
static bool prompt_content(const prompt_json_t *j, const prompt_json_node_t *content, prompt_buffer_t *out) {
    if (!content) return false;
    if (content->kind == PJ_NULL) return true;
    if (content->kind == PJ_STRING) return prompt_append(out, content->text);
    if (content->kind != PJ_ARRAY) return false;
    for (size_t i = (size_t)(content - j->nodes) + 1; i < content->next; i = j->nodes[i].next) {
        bool valid = true;
        const prompt_json_node_t *type = prompt_json_field(j, &j->nodes[i], "type", &valid);
        const prompt_json_node_t *text = prompt_json_field(j, &j->nodes[i], "text", &valid);
        if (!valid || !type || type->kind != PJ_STRING || !text ||
            (strcmp(type->text, "text") && strcmp(type->text, "input_text") && strcmp(type->text, "output_text")) ||
            (text->kind != PJ_NULL && (text->kind != PJ_STRING || !prompt_append(out, text->text)))) return false;
    }
    return true;
}
static bool prompt_tool_text(prompt_buffer_t *out, const char *text) {
    if (!prompt_append(out, "<tool_result>")) return false;
    for (const char *p = text; *p; ++p) {
        const char *q = p;
        if (*q == '&') { ++q; while (!strncmp(q, "amp;", 4)) q += 4; }
        bool escape = !strncmp(p, "</tool_result>", 14) || (*p == '&' && !strncmp(q, "lt;/tool_result>", 16));
        char byte[2] = {*p, 0};
        if (!prompt_append(out, escape ? (*p == '<' ? "&lt;" : "&amp;") : byte)) return false;
    }
    return prompt_append(out, "</tool_result>");
}
typedef struct { bool pending, user_open; size_t count; } prompt_chat_state_t;
static bool prompt_render_message(prompt_buffer_t *out, prompt_chat_state_t *state, const char *role,
                                  const char *text, const char *suffix) {
    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        if (!prompt_append(out, "<｜System｜>") || !prompt_append(out, text)) return false;
        (void)suffix;
        state->pending = state->count > 0; state->user_open = false;
    } else if (!strcmp(role, "user") || !strcmp(role, "tool") || !strcmp(role, "function")) {
        (void)suffix;
        if (!prompt_append(out, state->user_open ? "\n\n" : "<｜User｜>")) return false;
        if (!strcmp(role, "user") ? !prompt_append(out, text) : !prompt_tool_text(out, text)) return false;
        state->pending = state->user_open = true;
    } else if (!strcmp(role, "assistant")) {
        if (state->pending && !prompt_append(out, "<｜Assistant｜></think>")) return false;
        if (!prompt_append(out, text)) return false;
        if (suffix && !prompt_append(out, suffix)) return false;   /* 工具调用在 EOS 之前 */
        if (!prompt_append(out, "<｜end▁of▁sentence｜>")) return false;
        state->pending = state->user_open = false;
    } else return false;
    ++state->count; return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 * 工具调用（官方 DeepSeek-V4.1 格式）—— 见 docs/DEEPSEEK_V41_PROMPT_FORMAT.md
 * 该模型 GGUF 无 chat_template、词表无工具控制符 ⇒ 格式全由这里渲染进提示词。
 * DSML 标签名带【前导空格】（V4.1 相对 V4 的改动）：" calls" / " invoke" / " parameter"
 * ══════════════════════════════════════════════════════════════════════════ */
/* 注意：\x9c 后紧跟 D（十六进制位）会被并入转义 ⇒ 必须拆开字面量 */
#define PROMPT_DSML              "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"   /* ｜DSML｜（全角竖线） */
#define PROMPT_TOOL_CALLS_OPEN   "<" PROMPT_DSML " calls>"
#define PROMPT_TOOL_CALLS_CLOSE  "</" PROMPT_DSML " calls>"
#define PROMPT_TOOL_INVOKE_OPEN  "<" PROMPT_DSML " invoke name=\""
#define PROMPT_TOOL_INVOKE_CLOSE "</" PROMPT_DSML " invoke>"
#define PROMPT_TOOL_PARAM_OPEN   "<" PROMPT_DSML " parameter name=\""
#define PROMPT_TOOL_PARAM_CLOSE  "</" PROMPT_DSML " parameter>"

/* 工具名/参数名会被写进 DSML 属性 ⇒ 只允许安全字符，避免格式注入。 */
static bool prompt_name_ok(const char *name) {
    if (!name || !*name || strlen(name) > 128) return false;
    for (const char *p = name; *p; ++p) {
        char c = *p;
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  c == '_' || c == '-' || c == '.' || c == ':' || c == '/' || c == ' ';
        if (!ok) return false;
    }
    return true;
}
/* 取节点的原始 JSON 切片（含引号的字符串、或数字/对象等字面量）。 */
static const char *prompt_node_raw(const prompt_json_t *j, const prompt_json_node_t *n, size_t *len) {
    if (!n || !j->base || !n->len) return NULL;
    if (len) *len = n->len;
    return j->base + n->off;
}

/* 一行工具 schema：{"name":…[,"description":…][,"parameters":…]}
 * 用【原始切片】拼装 ⇒ 参数 schema 逐字节原样，不会因重新序列化而失真。 */
static bool prompt_append_tool_schema(prompt_buffer_t *out, const prompt_json_t *j,
                                      const prompt_json_node_t *tool) {
    if (!tool || tool->kind != PJ_OBJECT) return false;
    bool valid = true;
    const prompt_json_node_t *fn = prompt_json_field(j, tool, "function", &valid);
    const prompt_json_node_t *base = (fn && fn->kind == PJ_OBJECT) ? fn : tool;
    const prompt_json_node_t *name = prompt_json_field(j, base, "name", &valid);
    const prompt_json_node_t *desc = prompt_json_field(j, base, "description", &valid);
    const prompt_json_node_t *params = prompt_json_field(j, base, "parameters", &valid);
    if (!valid || !name || name->kind != PJ_STRING || !prompt_name_ok(name->text)) return false;
    size_t nlen = 0, dlen = 0, plen = 0;
    const char *nraw = prompt_node_raw(j, name, &nlen);
    const char *draw = prompt_node_raw(j, desc, &dlen);
    const char *praw = prompt_node_raw(j, params, &plen);
    if (!nraw) return false;
    if (!prompt_append(out, "{\"name\":")) return false;
    if (!prompt_append_n(out, nraw, nlen)) return false;
    if (draw && desc->kind == PJ_STRING) {
        if (!prompt_append(out, ",\"description\":")) return false;
        if (!prompt_append_n(out, draw, dlen)) return false;
    }
    if (praw && params->kind == PJ_OBJECT) {
        if (!prompt_append(out, ",\"parameters\":")) return false;
        if (!prompt_append_n(out, praw, plen)) return false;
    }
    return prompt_append(out, "}");
}

/* 官方 TOOLS_TEMPLATE：追加到 system 消息之后（调用方负责先补 "\n\n"）。 */
static bool prompt_render_tools(prompt_buffer_t *out, const prompt_json_t *j,
                                const prompt_json_node_t *tools) {
    if (!tools) return false;
    if (!prompt_append(out,
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \""
        PROMPT_TOOL_CALLS_OPEN "\" block like the following:\n\n"
        PROMPT_TOOL_CALLS_OPEN "\n"
        PROMPT_TOOL_INVOKE_OPEN "$TOOL_NAME\">\n"
        PROMPT_TOOL_PARAM_OPEN "$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE" PROMPT_TOOL_PARAM_CLOSE "\n"
        "...\n"
        PROMPT_TOOL_INVOKE_CLOSE "\n"
        PROMPT_TOOL_INVOKE_OPEN "$TOOL_NAME2\">\n"
        "...\n"
        PROMPT_TOOL_INVOKE_CLOSE "\n"
        PROMPT_TOOL_CALLS_CLOSE "\n\n"
        "String parameters should be specified as is and set `string=\"true\"`. For all other types "
        "(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside "
        "<think>...</think> BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n")) return false;
    size_t count = 0;
    if (tools->kind == PJ_ARRAY) {
        for (size_t i = (size_t)(tools - j->nodes) + 1; i < tools->next; i = j->nodes[i].next) {
            if (count++ && !prompt_append(out, "\n")) return false;
            if (!prompt_append_tool_schema(out, j, &j->nodes[i])) return false;
        }
    } else if (tools->kind == PJ_OBJECT) {
        count = 1;
        if (!prompt_append_tool_schema(out, j, tools)) return false;
    } else return false;
    if (!count) return false;
    return prompt_append(out, "\n\nYou MUST strictly follow the above defined tool name and parameter "
                              "schemas to invoke tool calls.");
}

/* assistant 历史里的工具调用 ⇒ DSML 块（返回值以 "\n\n" 开头，便于直接接到 content 之后）。 */
static bool prompt_render_tool_call_block(prompt_buffer_t *out, const prompt_json_t *j,
                                          const prompt_json_node_t *calls) {
    if (!calls) return false;
    /* 兼容三种形状：数组（OpenAI tool_calls）/ 单对象（function_call）/ 对象 */
    const prompt_json_node_t *array = calls;
    const prompt_json_node_t *single = NULL;
    if (calls->kind == PJ_OBJECT) {
        bool valid = true;
        const prompt_json_node_t *inner = prompt_json_field(j, calls, "tool_calls", &valid);
        if (inner && inner->kind == PJ_ARRAY) array = inner;
        else if (!prompt_json_field(j, calls, "name", &valid)) return false;
        else { array = NULL; single = calls; }
    } else if (calls->kind != PJ_ARRAY) return false;

    if (!prompt_append(out, "\n\n" PROMPT_TOOL_CALLS_OPEN "\n")) return false;
    size_t emitted = 0;
    size_t i = array ? (size_t)(array - j->nodes) + 1 : 0;
    for (;;) {
        const prompt_json_node_t *entry;
        if (array) {
            if (i >= array->next) break;
            entry = &j->nodes[i];
            i = j->nodes[i].next;
        } else { entry = single; i = 0; if (emitted) break; }
        if (!entry || entry->kind != PJ_OBJECT) return false;
        bool valid = true;
        const prompt_json_node_t *fn = prompt_json_field(j, entry, "function", &valid);
        const prompt_json_node_t *base = (fn && fn->kind == PJ_OBJECT) ? fn : entry;
        const prompt_json_node_t *name = prompt_json_field(j, base, "name", &valid);
        const prompt_json_node_t *args = prompt_json_field(j, base, "arguments", &valid);
        if (!args) args = prompt_json_field(j, base, "parameters", &valid);
        if (!valid || !name || name->kind != PJ_STRING || !prompt_name_ok(name->text)) return false;
        if (emitted++ && !prompt_append(out, "\n")) return false;
        if (!prompt_append(out, PROMPT_TOOL_INVOKE_OPEN)) return false;
        if (!prompt_append(out, name->text)) return false;
        if (!prompt_append(out, "\">")) return false;

        /* arguments 是 JSON 字符串（OpenAI）或对象；统一解析成对象再逐参数渲染。 */
        const char *arg_text = NULL;
        char *arg_copy = NULL;
        if (!args || args->kind == PJ_NULL) {
            /* 无参数 */
        } else if (args->kind == PJ_STRING) {
            arg_text = args->text;
        } else if (args->kind == PJ_OBJECT) {
            size_t alen = 0;
            const char *araw = prompt_node_raw(j, args, &alen);
            if (!araw) return false;
            arg_copy = malloc(alen + 1);
            if (!arg_copy) return false;
            memcpy(arg_copy, araw, alen); arg_copy[alen] = 0; arg_text = arg_copy;
        } else return false;

        if (arg_text && *arg_text) {
            prompt_json_t aj = {.cursor = arg_text, .base = arg_text};
            if (!prompt_json_value(&aj, 0) || aj.nodes[0].kind != PJ_OBJECT) {
                for (size_t k = 0; k < aj.count; ++k) free(aj.nodes[k].text);
                free(aj.nodes); free(arg_copy);
                /* 参数不是对象 ⇒ 按官方容错：包成 {"arguments": <原值>} */
                if (!prompt_append(out, "\n" PROMPT_TOOL_PARAM_OPEN "arguments\" string=\"false\">")) {
                    return false;
                }
                size_t rl = 0; const char *rraw = prompt_node_raw(j, args, &rl);
                if (rraw && !prompt_append_n(out, rraw, rl)) return false;
                if (!prompt_append(out, PROMPT_TOOL_PARAM_CLOSE)) return false;
            } else {
                if (!prompt_append(out, "\n")) { free(arg_copy); return false; }
                size_t pn = 0;
                for (size_t k = 1; k < aj.nodes[0].next;) {
                    const prompt_json_node_t *key = &aj.nodes[k];
                    const prompt_json_node_t *val = &aj.nodes[key->next];
                    if (pn++ && !prompt_append(out, "\n")) break;
                    if (!prompt_append(out, PROMPT_TOOL_PARAM_OPEN) ||
                        !prompt_append(out, key->text ? key->text : "") ||
                        !prompt_append(out, "\" string=\"")) break;
                    if (val->kind == PJ_STRING) {
                        if (!prompt_append(out, "true\">") || !prompt_append(out, val->text ? val->text : "")) break;
                    } else {
                        size_t vl = 0; const char *vraw = prompt_node_raw(&aj, val, &vl);
                        if (!prompt_append(out, "false\">") || !vraw || !prompt_append_n(out, vraw, vl)) break;
                    }
                    if (!prompt_append(out, PROMPT_TOOL_PARAM_CLOSE)) break;
                    k = val->next;
                }
                for (size_t k = 0; k < aj.count; ++k) free(aj.nodes[k].text);
                free(aj.nodes);
            }
        }
        free(arg_copy);
        if (!prompt_append(out, "\n" PROMPT_TOOL_INVOKE_CLOSE)) return false;
    }
    if (!emitted) return false;
    return prompt_append(out, "\n" PROMPT_TOOL_CALLS_CLOSE);
}

/* 失败原因（供日志与报错使用）。调用方每次进入前不必清空 —— 每个失败路径都会写它。 */
bool parse_chat_prompt_ex(const char *body, char *out_prompt, size_t max_prompt_len,
                          const char **out_reason) {
    const char **why = out_reason;
    const char *local_why = "unsupported prompt shape";
    if (!why) why = &local_why;
    *why = "unsupported prompt shape";
    if (!out_prompt || !max_prompt_len) { *why = "server misconfigured (no prompt buffer)"; return false; }
    out_prompt[0] = 0;
    if (!body) { *why = "no request body"; return false; }
    prompt_json_t json = {.cursor = body, .base = body};
    prompt_buffer_t out = {out_prompt, max_prompt_len, 0, false};
    prompt_chat_state_t state = {0};
    char *content_buffer = NULL; char *suffix = NULL; bool ok = false, valid = true;
    if (!prompt_json_value(&json, 0)) { *why = "body is not valid JSON"; goto done; }
    prompt_json_ws(&json);
    if (*json.cursor || json.nodes[0].kind != PJ_OBJECT) { *why = "body is not a JSON object"; goto done; }
    const prompt_json_node_t *root = &json.nodes[0];
    const prompt_json_node_t *instructions = prompt_json_field(&json, root, "instructions", &valid);
    const prompt_json_node_t *messages = prompt_json_field(&json, root, "messages", &valid);
    const prompt_json_node_t *input = prompt_json_field(&json, root, "input", &valid);
    const prompt_json_node_t *prompt = prompt_json_field(&json, root, "prompt", &valid);
    const prompt_json_node_t *tools = prompt_json_field(&json, root, "tools", &valid);
    const prompt_json_node_t *functions = prompt_json_field(&json, root, "functions", &valid);
    /* 工具定义：接受 tools（chat 嵌套 {"type","function":{…}} 或 Responses 扁平 {name,description,parameters}），
     * 也接受旧式 functions。渲染成官方 TOOLS_TEMPLATE 追加到 system 消息之后。 */
    bool has_tools = !prompt_absent_or_empty(&json, tools) || !prompt_absent_or_empty(&json, functions);
    const prompt_json_node_t *tool_defs = !prompt_absent_or_empty(&json, tools) ? tools : functions;
    if (!valid) { *why = "malformed JSON in messages/input/prompt"; goto done; }
    if (!messages && !input && !prompt) { *why = "no messages, input or prompt field"; goto done; }
    if ((messages && input) || (prompt && (messages || input))) {
        *why = "messages, input and prompt are mutually exclusive";
        goto done;
    }
    const prompt_json_node_t *items = messages ? messages : input ? input : prompt;
    /* 工具块要插在 system 消息之后；先看 items 里有没有 system 角色。 */
    bool system_in_items = false;
    if (has_tools && items->kind == PJ_ARRAY) {
        for (size_t i = (size_t)(items - json.nodes) + 1; i < items->next; i = json.nodes[i].next) {
            bool v2 = true;
            const prompt_json_node_t *r = prompt_json_field(&json, &json.nodes[i], "role", &v2);
            if (r && r->kind == PJ_STRING && !strcmp(r->text, "system")) { system_in_items = true; break; }
        }
    }
    bool tools_done = false;
    if (!prompt_append(&out, "<｜begin▁of▁sentence｜>")) goto done;
    if (instructions && instructions->kind != PJ_NULL) {
        if (instructions->kind != PJ_STRING ||
            !prompt_render_message(&out, &state, "system", instructions->text, NULL)) goto done;
        if (has_tools) {
            if (!prompt_append(&out, "\n\n") || !prompt_render_tools(&out, &json, tool_defs)) {
                *why = "invalid or oversized tool definitions (tools/functions)";
                goto done;
            }
            tools_done = true;
        }
    } else if (has_tools && !system_in_items) {
        /* 没有 system 消息：自建一条只装工具定义 */
        if (!prompt_render_message(&out, &state, "system", "", NULL) ||
            !prompt_append(&out, "\n\n") || !prompt_render_tools(&out, &json, tool_defs)) {
            *why = "invalid or oversized tool definitions (tools/functions)";
            goto done;
        }
        tools_done = true;
    }
    if (items->kind == PJ_STRING && !messages) {
        if (!prompt_render_message(&out, &state, "user", items->text, NULL)) goto done;
    } else if (items->kind == PJ_ARRAY && !prompt) {
        if ((size_t)(items - json.nodes) + 1 == items->next) goto done;
        content_buffer = malloc(max_prompt_len);
        if (!content_buffer) goto done;
        for (size_t i = (size_t)(items - json.nodes) + 1; i < items->next; i = json.nodes[i].next) {
            const prompt_json_node_t *item = &json.nodes[i];
            const prompt_json_node_t *role = prompt_json_field(&json, item, "role", &valid);
            const prompt_json_node_t *type = prompt_json_field(&json, item, "type", &valid);
            const prompt_json_node_t *content = prompt_json_field(&json, item, "content", &valid);
            const prompt_json_node_t *calls = prompt_json_field(&json, item, "tool_calls", &valid);
            const prompt_json_node_t *call = prompt_json_field(&json, item, "function_call", &valid);
            /* assistant 的工具调用 ⇒ DSML 块（接在 content 之后、EOS 之前） */
            const prompt_json_node_t *tc = !prompt_absent_or_empty(&json, calls) ? calls
                                         : (!prompt_absent_or_empty(&json, call) ? call : NULL);
            const char *role_text = role && role->kind == PJ_STRING ? role->text : "user";
            if (!valid || (role && role->kind != PJ_STRING) || (type && type->kind != PJ_STRING) || (messages && !role)) goto done;
            if (type && !strcmp(type->text, "function_call_output")) {
                content = prompt_json_field(&json, item, "output", &valid); role_text = "tool";
            } else if (type && strcmp(type->text, "message")) goto done;
            if (!content) content = prompt_json_field(&json, item, "text", &valid);
            prompt_buffer_t decoded = {content_buffer, max_prompt_len, 0, false}; content_buffer[0] = 0;
            /* 工具调用只能挂在 assistant 上；其余角色出现即视为请求有误 */
            if (tc && strcmp(role_text, "assistant")) {
                *why = "tool_calls/function_call is only supported on assistant messages";
                goto done;
            }
            if (tc) {
                suffix = malloc(max_prompt_len);
                if (!suffix) goto done;
                prompt_buffer_t sb = {suffix, max_prompt_len, 0, false};
                if (!prompt_render_tool_call_block(&sb, &json, tc)) {
                    *why = "malformed tool_calls/function_call in the message history";
                    goto done;
                }
            }
            if (!valid || !prompt_content(&json, content, &decoded) ||
                !prompt_render_message(&out, &state, role_text, decoded.data, suffix)) goto done;
            free(suffix); suffix = NULL;
            if (!strcmp(role_text, "system") && has_tools && !tools_done) {
                if (!prompt_append(&out, "\n\n") || !prompt_render_tools(&out, &json, tool_defs)) {
                    *why = "invalid or oversized tool definitions (tools/functions)";
                    goto done;
                }
                tools_done = true;
            }
        }
    } else goto done;
    if (state.pending && !prompt_append(&out, "<｜Assistant｜></think>")) goto done;
    ok = true;
done:
    if (!ok && out.overflow) *why = "the prompt does not fit the prompt buffer (request too large)";
    free(suffix);
    free(content_buffer);
    for (size_t i = 0; i < json.count; ++i) free(json.nodes[i].text);
    free(json.nodes);
    if (!ok) out_prompt[0] = 0;
    return ok;
}

/* 兼容包装：测试与其它调用方沿用旧的三参签名（它们不关心失败原因）。 */
bool parse_chat_prompt(const char *body, char *out_prompt, size_t max_prompt_len) {
    return parse_chat_prompt_ex(body, out_prompt, max_prompt_len, NULL);
}

/* ══════════════════════════════════════════════════════════════════════════
 * 解析模型输出里的 DSML 工具调用（官方 DeepSeek-V4.1 格式）
 * 官方的 parse_message_from_completion_text 是【严格】实现（格式稍不合就抛错）；
 * 这里做【宽容】解析：任何不合格式之处一律退回「整段当纯文本」，
 * 绝不因为模型一处手滑就把回答吞掉。
 * ══════════════════════════════════════════════════════════════════════════ */
#define MYDS_DSML_S "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"   /* ｜DSML｜（全角竖线） */

typedef struct { char *data; size_t len, cap; } dsml_buf_t;
static bool dsml_put(dsml_buf_t *b, const char *p, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 64;
        while (cap < b->len + n + 1) {
            if (cap > SIZE_MAX / 2) return false;
            cap *= 2;
        }
        char *grown = realloc(b->data, cap);
        if (!grown) return false;
        b->data = grown; b->cap = cap;
    }
    if (n) memcpy(b->data + b->len, p, n);
    b->len += n; b->data[b->len] = 0;
    return true;
}
static char *dsml_dup(const char *p, size_t n) {
    char *s = malloc(n + 1);
    if (!s) return NULL;
    memcpy(s, p, n); s[n] = 0;
    return s;
}
static const char *dsml_find(const char *pos, const char *end, const char *lit) {
    size_t n = strlen(lit);
    if (!n || pos > end || (size_t)(end - pos) < n) return NULL;
    for (const char *q = pos; q + n <= end; ++q) if (!memcmp(q, lit, n)) return q + n;
    return NULL;
}
static bool dsml_match(const char *p, const char *end, const char *lit) {
    size_t n = strlen(lit);
    return (size_t)(end - p) >= n && !memcmp(p, lit, n);
}
static const char *dsml_skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    return p;
}
/* 读 key="value" 属性；返回闭合引号之后的位置。 */
static const char *dsml_attr(const char *p, const char *end, const char *key,
                             const char **value, size_t *value_len) {
    if (!p) return NULL;
    if (!(p = dsml_find(p, end, key))) return NULL;
    if (p >= end || *p != '"') return NULL;
    const char *v = ++p;
    while (p < end && *p != '"') ++p;
    if (p >= end) return NULL;
    *value = v; *value_len = (size_t)(p - v);
    return p + 1;
}

/* 成功：content_len 收工具块之前的正文字节数；calls_out 与 count_out 收解析出的调用（调用方负责释放）。
 * 没有工具块时 count=0 且返回 true；格式不合格式时返回 false（调用方按纯文本处理）。 */
bool myds_parse_completion_tool_calls(const char *text, size_t len, size_t *content_len,
                                      myds_tool_call_t **calls_out, size_t *count_out) {
    static const char *open_tag = "<" MYDS_DSML_S " calls>";
    static const char *close_tag = "</" MYDS_DSML_S " calls>";
    static const char *invoke_open = "<" MYDS_DSML_S " invoke";
    static const char *invoke_close = "</" MYDS_DSML_S " invoke>";
    static const char *param_open = "<" MYDS_DSML_S " parameter";
    static const char *param_close = "</" MYDS_DSML_S " parameter>";
    *calls_out = NULL; *count_out = 0; *content_len = len;
    const char *end = text + len;
    const char *body = dsml_find(text, end, open_tag);
    if (!body) return true;
    const char *content_end = body - strlen(open_tag);
    while (content_end > text && (content_end[-1] == '\n' || content_end[-1] == '\r' ||
                                  content_end[-1] == ' ' || content_end[-1] == '\t')) --content_end;
    *content_len = (size_t)(content_end - text);

    size_t cap = 4, count = 0;
    myds_tool_call_t *calls = calloc(cap, sizeof(*calls));
    if (!calls) return false;
    const char *p = body;
    bool ok = true;
    while (ok) {
        p = dsml_skip_ws(p, end);
        if (dsml_match(p, end, close_tag)) { p += strlen(close_tag); break; }
        if (!dsml_match(p, end, invoke_open)) { ok = false; break; }
        p += strlen(invoke_open);
        const char *name = NULL; size_t name_len = 0;
        if (!(p = dsml_attr(p, end, "name=", &name, &name_len))) { ok = false; break; }
        p = dsml_skip_ws(p, end);
        if (p >= end || *p != '>') { ok = false; break; }
        ++p;

        dsml_buf_t args = {0};
        if (!dsml_put(&args, "{", 1)) { ok = false; break; }
        size_t n_params = 0;
        while (ok) {
            p = dsml_skip_ws(p, end);
            if (dsml_match(p, end, invoke_close)) { p += strlen(invoke_close); break; }
            if (!dsml_match(p, end, param_open)) { ok = false; break; }
            p += strlen(param_open);
            const char *pk = NULL; size_t pk_len = 0;
            const char *sv = NULL; size_t sv_len = 0;
            if (!(p = dsml_attr(p, end, "name=", &pk, &pk_len))) { ok = false; break; }
            if (!(p = dsml_attr(p, end, "string=", &sv, &sv_len))) { ok = false; break; }
            p = dsml_skip_ws(p, end);
            if (p >= end || *p != '>') { ok = false; break; }
            ++p;
            const char *vend = dsml_find(p, end, param_close);
            if (!vend) { ok = false; break; }
            size_t val_len = (size_t)(vend - strlen(param_close) - p);
            while (val_len && (p[val_len - 1] == '\n' || p[val_len - 1] == '\r')) --val_len;
            bool is_str = (sv_len == 4 && !memcmp(sv, "true", 4));
            if (n_params++ && !dsml_put(&args, ", ", 2)) { ok = false; break; }
            if (!dsml_put(&args, "\"", 1) || !dsml_put(&args, pk, pk_len) || !dsml_put(&args, "\": ", 3)) {
                ok = false; break;
            }
            if (is_str) {
                /* json_escape_alloc 只做转义、不加引号 ⇒ 引号要自己补 */
                char *esc = json_escape_alloc(p, val_len);
                if (!esc) { ok = false; break; }
                bool put = dsml_put(&args, "\"", 1) && dsml_put(&args, esc, strlen(esc)) &&
                           dsml_put(&args, "\"", 1);
                free(esc);
                if (!put) { ok = false; break; }
            } else if (!dsml_put(&args, p, val_len)) { ok = false; break; }
            p = vend;
        }
        if (ok && !dsml_put(&args, "}", 1)) ok = false;
        if (!ok) { free(args.data); break; }

        if (count == cap) {
            size_t grown_cap = cap * 2;
            myds_tool_call_t *grown = realloc(calls, grown_cap * sizeof(*calls));
            if (!grown) { free(args.data); ok = false; break; }
            calls = grown; cap = grown_cap;
        }
        char idbuf[32];
        snprintf(idbuf, sizeof(idbuf), "call_%zu", count + 1);
        myds_tool_call_t *tc = &calls[count];
        memset(tc, 0, sizeof(*tc));
        tc->id = dsml_dup(idbuf, strlen(idbuf));
        tc->name = dsml_dup(name, name_len);
        tc->arguments = args.data;
        if (!tc->id || !tc->name || !tc->arguments) { ok = false; break; }
        ++count;
    }
    if (!ok) {
        for (size_t k = 0; k < count; ++k) { free(calls[k].id); free(calls[k].name); free(calls[k].arguments); }
        free(calls);
        return false;
    }
    if (!count) { free(calls); *count_out = 0; return true; }
    *calls_out = calls; *count_out = count;
    return true;
}

#define DSML_PUTS(b, lit) dsml_put((b), (lit), strlen(lit))
static bool dsml_put_escaped(dsml_buf_t *b, const char *p, size_t n) {
    char *e = json_escape_alloc(p, n);
    if (!e) return false;
    bool ok = dsml_put(b, e, strlen(e));
    free(e);
    return ok;
}

/* OpenAI 的 tool_calls 数组：arguments 是【JSON 字符串】。 */
static char *openai_tool_calls_json(const myds_tool_call_t *calls, size_t count) {
    dsml_buf_t b = {0};
    if (!DSML_PUTS(&b, "[")) return NULL;
    for (size_t i = 0; i < count; ++i) {
        if ((i && !DSML_PUTS(&b, ",")) ||
            !DSML_PUTS(&b, "{\"id\":\"") ||
            !dsml_put_escaped(&b, calls[i].id, strlen(calls[i].id)) ||
            !DSML_PUTS(&b, "\",\"type\":\"function\",\"function\":{\"name\":\"") ||
            !dsml_put_escaped(&b, calls[i].name, strlen(calls[i].name)) ||
            !DSML_PUTS(&b, "\",\"arguments\":\"") ||
            !dsml_put_escaped(&b, calls[i].arguments, strlen(calls[i].arguments)) ||
            !DSML_PUTS(&b, "\"}}")) { free(b.data); return NULL; }
    }
    if (!DSML_PUTS(&b, "]")) { free(b.data); return NULL; }
    return b.data;
}

/* Responses API 的 output 数组：正文（若有）作 message 项，工具调用作 function_call 项。 */
static char *responses_output_items_json(const char *content, size_t content_len,
                                        const myds_tool_call_t *calls, size_t count) {
    dsml_buf_t b = {0};
    if (!DSML_PUTS(&b, "[")) return NULL;
    bool first = true;
    if (content_len) {
        if (!DSML_PUTS(&b, "{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\","
                           "\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"") ||
            !dsml_put_escaped(&b, content, content_len) ||
            !DSML_PUTS(&b, "\"}]}")) { free(b.data); return NULL; }
        first = false;
    }
    for (size_t i = 0; i < count; ++i) {
        char ids[96];
        snprintf(ids, sizeof(ids), "fc_%zu", i + 1);
        if ((!first && !DSML_PUTS(&b, ",")) ||
            !DSML_PUTS(&b, "{\"type\":\"function_call\",\"id\":\"") ||
            !DSML_PUTS(&b, ids) ||
            !DSML_PUTS(&b, "\",\"call_id\":\"") ||
            !dsml_put_escaped(&b, calls[i].id, strlen(calls[i].id)) ||
            !DSML_PUTS(&b, "\",\"name\":\"") ||
            !dsml_put_escaped(&b, calls[i].name, strlen(calls[i].name)) ||
            !DSML_PUTS(&b, "\",\"arguments\":\"") ||
            !dsml_put_escaped(&b, calls[i].arguments, strlen(calls[i].arguments)) ||
            !DSML_PUTS(&b, "\",\"status\":\"completed\"}")) { free(b.data); return NULL; }
        first = false;
    }
    if (!DSML_PUTS(&b, "]")) { free(b.data); return NULL; }
    return b.data;
}

void myds_free_tool_calls(myds_tool_call_t *calls, size_t count) {
    for (size_t i = 0; i < count; ++i) { free(calls[i].id); free(calls[i].name); free(calls[i].arguments); }
    free(calls);
}

static inline double myds_server_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static bool client_disconnected(int fd) {
    if (fd < 0) return false;
    char byte;
    ssize_t r;
    do { r = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT); } while (r < 0 && errno == EINTR);
    return r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
}

static bool sse_emit_content(myds_sse_stream_ctx_t *sctx, const char *piece, size_t length, bool final) {
    if (sctx->cancel_flag) return false;
    if (length > (SIZE_MAX - 1) / 6 - sctx->utf8.pending_length) goto failed;
    size_t capacity = (length + sctx->utf8.pending_length) * 6 + 1;
    char *escaped = malloc(capacity);
    if (!escaped) goto failed;
    size_t escaped_length = 0;
    bool encoded = escape_utf8_chunk(&sctx->utf8, piece, length, final,
                                     escaped, capacity, &escaped_length);
    if (!encoded) {
        free(escaped);
        goto failed;
    }
    if (escaped_length == 0) { /* Only an incomplete UTF-8 prefix so far. */
        free(escaped);
        return true;
    }
    const char *format = sctx->responses_api
        ? "event: response.output_text.delta\n"
          "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"%s\"}\n\n"
        : "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n";
    char *chunk = format_alloc(format, escaped);
    free(escaped);
    if (!chunk) goto failed;
    bool sent = myds_server_send_all(sctx->client_fd, chunk, strlen(chunk));
    free(chunk);
    if (!sent) sctx->cancel_flag = 1;
    return sent;
failed:
    sctx->failed = true;
    sctx->cancel_flag = 1;
    return false;
}

/* 累积原始文本，并在【确认进入 DSML 工具块】之前照常输出正文 delta。
 * 工具块一旦出现就停止输出正文 —— 收尾阶段改为发结构化工具调用。
 * 尾部若可能是开标签的一部分，先扣住不发，避免把半截标签当正文发给客户端。 */
static bool sse_emit_bytes(myds_sse_stream_ctx_t *sctx, const char *piece, size_t length, bool final) {
    static const char *open_tag = "<" MYDS_DSML_S " calls>";
    const size_t tag_len = strlen(open_tag);
    if (length) {
        /* 先挡住长度溢出：acc_len + length + 1 自身会回绕，
         * 回绕后判断为「不用扩容」⇒ 直接 memcpy 到空指针（既有测试正是用 SIZE_MAX 打这里）。 */
        if (length > SIZE_MAX - 1 - sctx->acc_len) {
            sctx->failed = true; sctx->cancel_flag = 1;
            return false;
        }
        if (sctx->acc_len + length + 1 > sctx->acc_cap) {
            size_t cap = sctx->acc_cap ? sctx->acc_cap : 4096;
            while (cap < sctx->acc_len + length + 1) {
                if (cap > SIZE_MAX / 2) { sctx->failed = true; sctx->cancel_flag = 1; return false; }
                cap *= 2;
            }
            char *grown = realloc(sctx->acc, cap);
            if (!grown) { sctx->failed = true; sctx->cancel_flag = 1; return false; }
            sctx->acc = grown; sctx->acc_cap = cap;
        }
        memcpy(sctx->acc + sctx->acc_len, piece, length);
        sctx->acc_len += length;
        sctx->acc[sctx->acc_len] = 0;
    }
    if (sctx->in_tool_block) return true;              /* 工具块中 ⇒ 不再输出正文 */
    const char *base = sctx->acc + sctx->emitted;
    size_t avail = sctx->acc_len - sctx->emitted;
    const char *tag = avail ? dsml_find(base, sctx->acc + sctx->acc_len, open_tag) : NULL;
    if (tag) {
        size_t upto = (size_t)(tag - tag_len - base);
        while (upto && (base[upto - 1] == '\n' || base[upto - 1] == '\r')) --upto;
        bool ok = upto ? sse_emit_content(sctx, base, upto, false) : true;
        sctx->emitted += upto;
        sctx->in_tool_block = true;
        return ok;
    }
    size_t maxk = tag_len - 1 < avail ? tag_len - 1 : avail;
    size_t keep = 0;
    for (size_t k = maxk; k > 0; --k) {
        if (!memcmp(base + avail - k, open_tag, k)) { keep = k; break; }
    }
    /* 尾部空白也先扣住：工具块通常紧跟两个换行之后，立刻发出会让客户端先看到空行。
     * 注意必须【无条件】做 —— 若只因「标签前缀匹配成功(keep>0)」就跳过，
     * 标签前那串换行就会被当成正文发出去（实测泄露 "\n\n"）。 */
    while (keep < avail && (base[avail - 1 - keep] == '\n' || base[avail - 1 - keep] == '\r')) ++keep;
    size_t upto = avail - keep;
    bool ok = upto ? sse_emit_content(sctx, base, upto, false) : true;
    sctx->emitted += upto;
    if (!ok) return false;
    if (final) {
        if (keep) {                                    /* 半截标签始终没补全 ⇒ 当正文发出去 */
            ok = sse_emit_content(sctx, base + upto, keep, false);
            sctx->emitted += keep;
        }
        ok = ok && sse_emit_content(sctx, NULL, 0, true);
    }
    return ok;
}

/* chat/completions：一次性给出全部工具调用的 delta（OpenAI 允许多个 index）。 */
static char *chat_tool_calls_delta_json(const myds_tool_call_t *calls, size_t count) {
    dsml_buf_t b = {0};
    if (!DSML_PUTS(&b, "[")) return NULL;
    for (size_t i = 0; i < count; ++i) {
        char idx[32];
        snprintf(idx, sizeof(idx), "%zu", i);
        if ((i && !DSML_PUTS(&b, ",")) ||
            !DSML_PUTS(&b, "{\"index\":") || !DSML_PUTS(&b, idx) ||
            !DSML_PUTS(&b, ",\"id\":\"") || !dsml_put_escaped(&b, calls[i].id, strlen(calls[i].id)) ||
            !DSML_PUTS(&b, "\",\"type\":\"function\",\"function\":{\"name\":\"") ||
            !dsml_put_escaped(&b, calls[i].name, strlen(calls[i].name)) ||
            !DSML_PUTS(&b, "\",\"arguments\":\"") ||
            !dsml_put_escaped(&b, calls[i].arguments, strlen(calls[i].arguments)) ||
            !DSML_PUTS(&b, "\"}}")) { free(b.data); return NULL; }
    }
    if (!DSML_PUTS(&b, "]")) { free(b.data); return NULL; }
    return b.data;
}

/* chat/completions：工具调用 delta + finish_reason=tool_calls + [DONE] */
static bool sse_emit_chat_tool_calls(myds_sse_stream_ctx_t *sctx, const myds_tool_call_t *calls, size_t count) {
    char *tc = chat_tool_calls_delta_json(calls, count);
    if (!tc) return false;
    char *chunk = format_alloc(
        "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":%s},\"finish_reason\":null}]}\n\n", tc);
    if (!chunk) { free(tc); return false; }
    bool sent = myds_server_send_all(sctx->client_fd, chunk, strlen(chunk));
    free(chunk); free(tc);
    if (!sent) return false;
    const char *finish_chunk =
        "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
        "\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n";
    if (!myds_server_send_all(sctx->client_fd, finish_chunk, strlen(finish_chunk))) return false;
    const char *done = "data: [DONE]\n\n";
    return myds_server_send_all(sctx->client_fd, done, strlen(done));
}

/* Responses API：function_call 项的 added / arguments.delta / done */
static bool sse_emit_responses_function_calls(myds_sse_stream_ctx_t *sctx,
                                              const myds_tool_call_t *calls, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        char ids[96];
        snprintf(ids, sizeof(ids), "fc_%zu", i + 1);
        char *added = format_alloc(
            "event: response.output_item.added\n"
            "data: {\"type\":\"response.output_item.added\",\"output_index\":%zu,\"item\":"
            "{\"type\":\"function_call\",\"id\":\"%s\",\"call_id\":\"%s\",\"name\":\"%s\",\"arguments\":\"\","
            "\"status\":\"in_progress\"}}\n\n", i, ids, calls[i].id, calls[i].name);
        if (!added) return false;
        bool sent = myds_server_send_all(sctx->client_fd, added, strlen(added));
        free(added);
        if (!sent) return false;
        char *delta = format_alloc(
            "event: response.function_call_arguments.delta\n"
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"%s\","
            "\"output_index\":%zu,\"delta\":\"%s\"}\n\n", ids, i, calls[i].arguments);
        if (!delta) return false;
        sent = myds_server_send_all(sctx->client_fd, delta, strlen(delta));
        free(delta);
        if (!sent) return false;
        char *done = format_alloc(
            "event: response.output_item.done\n"
            "data: {\"type\":\"response.output_item.done\",\"output_index\":%zu,\"item\":"
            "{\"type\":\"function_call\",\"id\":\"%s\",\"call_id\":\"%s\",\"name\":\"%s\",\"arguments\":\"%s\","
            "\"status\":\"completed\"}}\n\n", i, ids, calls[i].id, calls[i].name, calls[i].arguments);
        if (!done) return false;
        sent = myds_server_send_all(sctx->client_fd, done, strlen(done));
        free(done);
        if (!sent) return false;
    }
    return true;
}

void myds_server_sse_callback(const char *piece, size_t length, int32_t token_id, void *user_data) {
    (void)token_id;
    myds_sse_stream_ctx_t *sctx = user_data;
    double now = myds_server_now_s();
    if (sctx->token_count == 0) sctx->t_first = now;
    sctx->t_last = now;
    sctx->token_count++;
    if (client_disconnected(sctx->client_fd)) {
        sctx->cancel_flag = 1;
        return;
    }
    (void)sse_emit_bytes(sctx, piece, length, false);
}

bool myds_server_sse_finish(myds_sse_stream_ctx_t *sctx) {
    return sse_emit_bytes(sctx, NULL, 0, true);
}

void myds_server_non_stream_callback(const char *piece, size_t length, int32_t token_id, void *user_data) {
    (void)token_id;
    myds_non_stream_ctx_t *ns = user_data;
    double now = myds_server_now_s();
    if (ns->token_count == 0) ns->t_first = now;
    ns->t_last = now;
    ns->token_count++;
    if (ns->cancel_flag) return;
    if (client_disconnected(ns->client_fd)) {
        ns->cancel_flag = 1;
        return;
    }
    if (ns->len == SIZE_MAX || length > SIZE_MAX - ns->len - 1) goto failed;
    size_t needed = ns->len + length + 1;
    if (needed > ns->cap) {
        size_t capacity = ns->cap ? ns->cap : 4096;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2) { capacity = needed; break; }
            capacity *= 2;
        }
        char *larger = realloc(ns->buf, capacity);
        if (!larger) goto failed;
        ns->buf = larger;
        ns->cap = capacity;
    }
    if (length) memcpy(ns->buf + ns->len, piece, length);
    ns->len += length;
    ns->buf[ns->len] = '\0';
    return;
failed:
    ns->failed = true;
    ns->cancel_flag = 1;
}

static bool send_response(int fd, int code, const char *status, const char *content_type, const char *body) {
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
    if (hlen < 0 || (size_t)hlen >= sizeof(header)) return false;
    if (!myds_server_send_all(fd, header, (size_t)hlen)) return false;
    return body_len == 0 || myds_server_send_all(fd, body, body_len);
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
        (void)myds_server_send_all(client_fd, cors_res, strlen(cors_res));
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
        double t_start = myds_server_now_s();
        double t_first = 0.0;
        double t_last = 0.0;

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
                } else {
                    send_response(client_fd, 500, "Internal Server Error", "application/json",
                                  "{\"error\":{\"message\":\"Request allocation failed\"}}");
                    free(req_buf);
                    close(client_fd);
                    return;
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
            /* The JSON/parser API accepts C strings: raw NUL must fail rather
             * than hiding a suffix. Never parse a partial Content-Length body. */
            if (content_len <= 0 || body_have < content_len ||
                memchr(body, '\0', (size_t)content_len)) {
                send_response(client_fd, 400, "Bad Request", "application/json",
                              "{\"error\":{\"message\":\"Incomplete or invalid JSON body\"}}");
                free(req_buf);
                close(client_fd);
                return;
            }
            body[content_len] = '\0';
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
        int top_k = json_extract_int_field(body, "top_k", -1);
        int include_usage = (strstr(body, "\"include_usage\":true") != NULL || strstr(body, "\"include_usage\": true") != NULL);

        size_t prompt_cap = (total_read * 4 > 65536) ? (total_read * 4 + 8192) : 65536;
        char *prompt = malloc(prompt_cap);
        if (!prompt) {
            MYDS_LOG_E("[%s:%d] Out of memory allocating prompt buffer (%zu bytes)", client_ip, client_port, prompt_cap);
            send_response(client_fd, 500, "Internal Server Error", "application/json", "{\"error\":\"Out of memory\"}");
            free(req_buf);
            close(client_fd);
            return;
        }
        const char *parse_reason = NULL;
        if (!parse_chat_prompt_ex(body, prompt, prompt_cap, &parse_reason)) {
            /* 必须落日志：此前这条路径【完全不记录】，导致线上只能看到一个笼统的 400。 */
            MYDS_LOG_W("[%s:%d] ChatCompletions rejected (400): %s [body=%zd bytes, prompt_cap=%zu]",
                       client_ip, client_port, parse_reason, total_read, prompt_cap);
            char err[512];
            snprintf(err, sizeof(err),
                     "{\"error\":{\"message\":\"Invalid or unsupported text prompt: %s\",\"type\":\"invalid_request_error\"}}",
                     parse_reason ? parse_reason : "unknown");
            send_response(client_fd, 400, "Bad Request", "application/json", err);
            free(prompt);
            free(req_buf);
            close(client_fd);
            return;
        }

        MYDS_LOG_I("[%s:%d] %s %s (protocol=%s, stream=%s, max_tokens=%d, temp=%.2f, top_p=%.2f, top_k=%d, prompt_chars=%zu)",
                   client_ip, client_port, method, path,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   stream ? "true" : "false",
                   max_tokens, temp, top_p, top_k,
                   strlen(prompt));

        pthread_mutex_lock(&g_infer_mutex);

        /* Apply request configuration overrides if provided */
        myds_config_t prev_cfg = myds_context_get_config(ctx);
        myds_config_t req_cfg = prev_cfg;
        req_cfg.max_new_tokens = myds_server_default_max_tokens(prev_cfg.max_new_tokens);
        if (max_tokens > 0) req_cfg.max_new_tokens = (uint32_t)max_tokens;
        if (temp >= 0.0f) req_cfg.temperature = temp;
        if (top_p >= 0.0f) req_cfg.top_p = top_p;
        if (top_k >= 0) req_cfg.top_k = (uint32_t)top_k;
        myds_context_set_config(ctx, &req_cfg);

        myds_stop_reason_t stop_reason = MYDS_STOP_EOS;
        uint32_t prompt_tokens = 0;
        uint32_t final_tokens = 0;

        if (stream) {
            const char *sse_header =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
            const char *initial = is_responses_endpoint
                ? "event: response.output_item.added\n"
                  "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n"
                  "event: response.content_part.added\n"
                  "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n"
                : "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n";
            myds_sse_stream_ctx_t sctx = {
                .client_fd = client_fd,
                .responses_api = is_responses_endpoint,
            };
            if (!myds_server_send_all(client_fd, sse_header, strlen(sse_header)) ||
                !myds_server_send_all(client_fd, initial, strlen(initial))) {
                sctx.cancel_flag = 1;
            }
            int generated = myds_generate_bytes_ex(ctx, prompt, myds_server_sse_callback, &sctx,
                                                   &stop_reason, &prompt_tokens, &sctx.cancel_flag);
            if (generated && !sctx.cancel_flag) (void)myds_server_sse_finish(&sctx);
            final_tokens = sctx.token_count;
            t_first = sctx.t_first;
            t_last = sctx.t_last;
            if (!generated || sctx.failed) {
                stop_reason = MYDS_STOP_ERROR;
                const char *error_event = is_responses_endpoint
                    ? "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\n"
                    : "data: {\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n";
                (void)myds_server_send_all(client_fd, error_event, strlen(error_event));
            } else if (!sctx.cancel_flag) {
                /* 模型可能用 DSML 块发起工具调用 ⇒ 从累积文本里解析（与协议无关） */
                size_t scontent_len = sctx.acc_len;
                myds_tool_call_t *stool = NULL;
                size_t stool_count = 0;
                if (!sctx.acc || !myds_parse_completion_tool_calls(sctx.acc, sctx.acc_len,
                                                                 &scontent_len, &stool, &stool_count)) {
                    scontent_len = sctx.acc_len; stool = NULL; stool_count = 0;
                }
                if (!stool_count && sctx.acc && sctx.emitted < sctx.acc_len) {
                    /* 曾疑似进入工具块但格式不合 ⇒ 把没发出的正文补出去，绝不吞内容 */
                    (void)sse_emit_content(&sctx, sctx.acc + sctx.emitted,
                                           sctx.acc_len - sctx.emitted, false);
                    sctx.emitted = sctx.acc_len;
                }
                if (stool_count) {
                    bool tsent = is_responses_endpoint
                        ? sse_emit_responses_function_calls(&sctx, stool, stool_count)
                        : sse_emit_chat_tool_calls(&sctx, stool, stool_count);
                    if (!tsent) stop_reason = MYDS_STOP_CANCELLED;
                    myds_free_tool_calls(stool, stool_count);
                    stool = NULL;
                }
                if (!stool_count && is_responses_endpoint) {
                    char *alias_json = json_escape_alloc(model_alias ? model_alias : "myds",
                                                         strlen(model_alias ? model_alias : "myds"));
                    char *completed = alias_json ? format_alloc(
                        "event: response.content_part.done\n"
                        "data: {\"type\":\"response.content_part.done\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n"
                        "event: response.output_item.done\n"
                        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]}}\n\n"
                        "event: response.completed\n"
                        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_myds\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"%s\",\"usage\":{\"input_tokens\":%u,\"output_tokens\":%u,\"total_tokens\":%u}}}\n\n",
                        alias_json, prompt_tokens, final_tokens, prompt_tokens + final_tokens) : NULL;
                    free(alias_json);
                    if (completed) {
                        if (!myds_server_send_all(client_fd, completed, strlen(completed))) stop_reason = MYDS_STOP_CANCELLED;
                        free(completed);
                    } else {
                        stop_reason = MYDS_STOP_ERROR;
                        const char *error_event = "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Response allocation failed\",\"type\":\"server_error\"}}\n\n";
                        (void)myds_server_send_all(client_fd, error_event, strlen(error_event));
                    }
                } else if (!stool_count) {
                    const char *finish = (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL) ? "length" : "stop";
                    char chunk[512];
                    int len = snprintf(chunk, sizeof(chunk),
                        "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n", finish);
                    bool sent = len > 0 && (size_t)len < sizeof(chunk) && myds_server_send_all(client_fd, chunk, (size_t)len);
                    if (sent && include_usage) {
                        len = snprintf(chunk, sizeof(chunk),
                            "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}\n\n",
                            prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                        sent = len > 0 && (size_t)len < sizeof(chunk) && myds_server_send_all(client_fd, chunk, (size_t)len);
                    }
                    const char *done = "data: [DONE]\n\n";
                    if (!sent || !myds_server_send_all(client_fd, done, strlen(done))) stop_reason = MYDS_STOP_CANCELLED;
                }
            } else {
                stop_reason = MYDS_STOP_CANCELLED;
            }
            free(sctx.acc); sctx.acc = NULL;   /* 每请求一份累积缓冲，必须释放 */
        } else {
            myds_non_stream_ctx_t ns = {.client_fd = client_fd};
            int generated = myds_generate_bytes_ex(ctx, prompt, myds_server_non_stream_callback, &ns,
                                                   &stop_reason, &prompt_tokens, &ns.cancel_flag);
            final_tokens = ns.token_count;
            t_first = ns.t_first;
            t_last = ns.t_last;
            if (!generated || ns.failed) {
                stop_reason = MYDS_STOP_ERROR;
                send_response(client_fd, 500, "Internal Server Error", "application/json",
                              "{\"error\":{\"message\":\"Inference or response allocation failed\",\"type\":\"server_error\"}}");
            } else if (!ns.cancel_flag) {
                char *escaped = json_escape_alloc(ns.buf, ns.len);
                char *alias_json = json_escape_alloc(model_alias ? model_alias : "myds",
                                                     strlen(model_alias ? model_alias : "myds"));
                char *response = NULL;
                const char *finish = (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL) ? "length" : "stop";
                long now = (long)time(NULL);

                /* 模型可能在正文之后用 DSML 块发起工具调用 ⇒ 拆出来，按各自协议输出。 */
                size_t tool_content_len = ns.len;
                myds_tool_call_t *tool_calls = NULL;
                size_t tool_count = 0;
                if (!myds_parse_completion_tool_calls(ns.buf, ns.len, &tool_content_len, &tool_calls, &tool_count)) {
                    tool_content_len = ns.len; tool_calls = NULL; tool_count = 0;   /* 格式不合 ⇒ 整段当正文 */
                }
                char *body_escaped = json_escape_alloc(ns.buf, tool_content_len);
                char *calls_json = tool_count ? openai_tool_calls_json(tool_calls, tool_count) : NULL;
                char *items_json = responses_output_items_json(ns.buf, tool_content_len, tool_calls, tool_count);
                char *content_field = body_escaped ? format_alloc("\"%s\"", body_escaped) : NULL;
                if (tool_count) finish = "tool_calls";

                if (body_escaped && alias_json) {
                    if (is_responses_endpoint) {
                        response = items_json ? format_alloc(
                            "{\"id\":\"resp_myds_%ld\",\"object\":\"response\",\"created_at\":%ld,"
                            "\"status\":\"completed\",\"model\":\"%s\","
                            "\"output\":%s,"
                            "\"usage\":{\"input_tokens\":%u,\"output_tokens\":%u,\"total_tokens\":%u}}",
                            now, now, alias_json, items_json,
                            prompt_tokens, final_tokens, prompt_tokens + final_tokens) : NULL;
                    } else if (tool_count && calls_json && content_field) {
                        response = format_alloc(
                            "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                            "\"content\":%s,\"tool_calls\":%s},"
                            "\"finish_reason\":\"%s\"}],"
                            "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                            now, alias_json, content_field, calls_json, finish,
                            prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                    } else {
                        response = format_alloc(
                            "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                            "\"finish_reason\":\"%s\"}],"
                            "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                            now, alias_json, body_escaped, finish, prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                    }
                }
                free(content_field);
                myds_free_tool_calls(tool_calls, tool_count);
                free(items_json);
                free(calls_json);
                free(body_escaped);
                free(escaped);
                free(alias_json);
                if (response) {
                    if (!send_response(client_fd, 200, "OK", "application/json", response)) stop_reason = MYDS_STOP_CANCELLED;
                    free(response);
                } else {
                    stop_reason = MYDS_STOP_ERROR;
                    send_response(client_fd, 500, "Internal Server Error", "application/json",
                                  "{\"error\":{\"message\":\"Response allocation failed\",\"type\":\"server_error\"}}");
                }
            }
            free(ns.buf);
        }

        double t_end = myds_server_now_s();
        double wall_s = t_end - t_start;
        double prefill_s = (final_tokens > 0 && t_first > t_start) ? (t_first - t_start) : wall_s;
        double ttft_s = prefill_s;
        double decode_s = (final_tokens > 1 && t_last > t_first) ? (t_last - t_first) : 0.0;
        double decode_tps = (decode_s > 0.0 && final_tokens > 1) ? ((double)(final_tokens - 1) / decode_s) : ((wall_s > 0.0) ? ((double)final_tokens / wall_s) : 0.0);
        const char *final_finish_str = (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL) ? "length" : (stop_reason == MYDS_STOP_CANCELLED ? "cancelled" : (stop_reason == MYDS_STOP_ERROR ? "error" : "stop"));

        MYDS_LOG_I("[%s:%d] Completed %s %s - prompt: %u tok, ttft: %.3fs, decode: %u tok in %.3fs = %.2f t/s, wall: %.3fs, finish_reason: %s",
                   client_ip, client_port,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   path, prompt_tokens, ttft_s, final_tokens, decode_s, decode_tps, wall_s, final_finish_str);

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

typedef struct {
    int client_fd;
    myds_context_t *ctx;
    char model_alias[64];
    char client_ip[INET_ADDRSTRLEN];
    int client_port;
} server_client_job_t;

static void *server_client_thread(void *arg) {
    server_client_job_t *job = (server_client_job_t *)arg;
    handle_client(job->client_fd, job->ctx, job->model_alias, job->client_ip, job->client_port);
    free(job);
    return NULL;
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

        server_client_job_t *job = (server_client_job_t *)malloc(sizeof(server_client_job_t));
        if (!job) {
            handle_client(client_fd, ctx, model_alias, client_ip, client_port);
            continue;
        }
        job->client_fd = client_fd;
        job->ctx = ctx;
        strncpy(job->model_alias, model_alias ? model_alias : "myds", sizeof(job->model_alias) - 1);
        job->model_alias[sizeof(job->model_alias) - 1] = '\0';
        strncpy(job->client_ip, client_ip, sizeof(job->client_ip) - 1);
        job->client_ip[sizeof(job->client_ip) - 1] = '\0';
        job->client_port = client_port;

        pthread_t tid;
        if (pthread_create(&tid, NULL, server_client_thread, job) == 0) {
            pthread_detach(tid);
        } else {
            handle_client(client_fd, ctx, model_alias, client_ip, client_port);
            free(job);
        }
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
