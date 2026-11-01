/* myds_proto_openai.c — 请求解析与提示词渲染（OpenAI / Responses 协议面）
 *
 * 2026-11-01 从 myds_server.c 纯机械拆分而来：函数体与注释未改。
 *
 * 负责两端：把客户端的 JSON 请求变成引擎能吃的东西（提示词、采样参数、
 * 工具集指纹），以及把正文/工具调用拼成 OpenAI 与 Responses 两种响应的 JSON。
 *
 * 不负责：socket、SSE 线程、排队。那些在 http_io / infer_queue 里。
 * 分界依据是「有没有 fd」：本文件的函数只收字符串和缓冲区，不碰连接。
 */
#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_vjob.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>

static bool append_json_bytes(char *dst, size_t cap, size_t *used, const char *bytes, size_t length) {
    if (*used >= cap || length >= cap - *used) return false;
    memcpy(dst + *used, bytes, length);
    *used += length;
    return true;
}

/* The pending bytes are a valid UTF-8 prefix. Keep it across token boundaries,
 * because a byte-BPE token can end in the middle of a Unicode character. */
bool escape_utf8_chunk(myds_utf8_state_t *state, const char *src, size_t length,
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

char *json_escape_alloc(const char *src, size_t length) {
    if (length > (SIZE_MAX - 1) / 6) return NULL;
    size_t cap = length * 6 + 1;
    char *escaped = malloc(cap);
    if (escaped && !json_escape_bytes(src, length, escaped, cap, NULL)) {
        free(escaped);
        return NULL;
    }
    return escaped;
}

uint32_t myds_server_default_max_tokens(uint32_t configured_max_tokens, bool thinking,
                                        const char *reasoning_effort) {
    uint32_t want = 8192u;                                  /* DEFAULT_NON_THINKING_MAX_TOKENS */
    if (thinking) {
        want = 65536u;                                      /* DEFAULT_THINKING_MAX_TOKENS */
        /* 官方映射表：max 与 ultra 都落到 max 档（xhigh 落 high，不在此列）。 */
        if (reasoning_effort &&
            (strcmp(reasoning_effort, "max") == 0 || strcmp(reasoning_effort, "ultra") == 0))
            want = 131072u;                                 /* DEFAULT_MAX_EFFORT_MAX_TOKENS */
    }
    if (configured_max_tokens && want > configured_max_tokens) want = configured_max_tokens;
    return want ? want : 1u;
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
            /* 2026-10-25: 原本用 atoi，超 INT_MAX 时行为未定义（实测 "2147483648"
             * 得到 INT_MIN，"10000000000" 得到 1410065408）。
             * 对 Content-Length 而言这不是“随便一个数字”——它决定分配多少内存。
             * 改成 strtol + 显式范围检查：超出 int 能表达的范围就返回 -1，
             * 由调用方当成“非法头”处理（而不是误读成一个看似合理的小值）。 */
            errno = 0;
            char *end = NULL;
            const long v = strtol(p, &end, 10);
            if (end == p || errno == ERANGE || v > (long)INT_MAX || v < (long)INT_MIN) {
                return -1;
            }
            return (int)v;
        }
        p = strchr(p, '\n');
        if (!p) break;
        p++;
    }
    return 0;
}

/* Copy one header value into `out`. Returns true when the header was present.
 *
 * Trimmed of trailing CR/LF and surrounding spaces; truncated silently if `cap` is
 * too small, which for a caller comparing a secret means "will not match" rather
 * than "matches a prefix". */
int extract_http_header_str(const char *headers, const char *header_name,
                            char *out, size_t cap) {
    if (!headers || !header_name || !out || cap == 0) return 0;
    out[0] = 0;
    const size_t name_len = strlen(header_name);
    const char *p = headers;
    while (*p) {
        if (strncasecmp(p, header_name, name_len) == 0 && p[name_len] == ':') {
            p += name_len + 1;
            while (*p == ' ' || *p == '\t') p++;
            size_t n = 0;
            while (*p && *p != '\r' && *p != '\n' && n + 1 < cap) out[n++] = *p++;
            while (n && (out[n - 1] == ' ' || out[n - 1] == '\t')) n--;
            out[n] = 0;
            return 1;
        }
        p = strchr(p, '\n');
        if (!p) break;
        p++;
    }
    return 0;
}

/* Very simple JSON string value finder: searches for "key"\s*:\s*"value" */
/* 判断 body 的【顶层】是否有某个 key。
 * 只认深度 1：input 里的消息内容也可能含 "tools" 之类的字样，不能被误判。
 * 用途：看请求是否重复携带了已在续接序列里的 instructions/tools。 */
bool json_has_top_field(const char *body, const char *key) {
    if (!body || !key) return false;
    const size_t klen = strlen(key);
    int depth = 0;
    const char *p = body;
    while (*p) {
        if (*p == '{' || *p == '[') { ++depth; ++p; continue; }
        if (*p == '}' || *p == ']') { --depth; ++p; continue; }
        if (*p == '"') {
            const char *start = p + 1;
            const char *q = start;
            while (*q) {
                if (*q == '\\') { q += 2; continue; }
                if (*q == '"') break;
                ++q;
            }
            if (!*q) break;
            const char *after = q + 1;
            while (*after == ' ' || *after == '\t' || *after == '\r' || *after == '\n') ++after;
            if (depth == 1 && *after == ':' && (size_t)(q - start) == klen &&
                !strncmp(start, key, klen)) return true;
            p = q + 1;
            continue;
        }
        ++p;
    }
    return false;
}

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

/* 取顶层键的值区间（不深入嵌套）。返回指向值首字符的指针，*len 为区间长。
 *
 * 为什么不直接 strstr：2026-10-25 修的就是那个误判 —— 一个无关的【嵌套】对象
 * 只要带同名键，就会被当成请求参数：
 *   {"metadata":{"response_format":{"type":"json_schema"}}}  ⇒ 整请求被 400
 * 注意【不是】提示词里的文字引起的：JSON 要求引号转义，字符串内部的同名文字
 * 在原始字节里是 \"response_format\"，strstr("response_format") 根本不会命中
 * （已用旧实现实测验证）。真正会命中的只有真实键名，而真实键名可能在任意深度。
 * 这里与 json_has_top_field 同一套扫描规则：跟踪括号深度、跳过字符串字面量，
 * 只在 depth==1 处匹配键且后面紧跟 ':'。
 *
 * 值区间：以 { 或 [ 开头时按括号配对（同样跳字符串），否则到本层 , 或 } 为止。
 * 值不存在/格式异常时返回 NULL。 */
static const char *json_top_field_value(const char *body, const char *key, size_t *len) {
    if (len) *len = 0;
    if (!body || !key) return NULL;
    const size_t klen = strlen(key);
    int depth = 0;
    const char *p = body;
    while (*p) {
        if (*p == '{' || *p == '[') { ++depth; ++p; continue; }
        if (*p == '}' || *p == ']') { --depth; ++p; continue; }
        if (*p != '"') { ++p; continue; }
        const char *start = p + 1;
        const char *q = start;
        while (*q) {
            if (*q == '\\') { q += 2; continue; }
            if (*q == '"') break;
            ++q;
        }
        if (!*q) break;
        const char *after = q + 1;
        while (*after == ' ' || *after == '\t' || *after == '\r' || *after == '\n') ++after;
        if (depth == 1 && *after == ':' && (size_t)(q - start) == klen &&
            !strncmp(start, key, klen)) {
            const char *v = after + 1;
            while (*v == ' ' || *v == '\t' || *v == '\r' || *v == '\n') ++v;
            const char *end = v;
            if (*v == '{' || *v == '[') {
                int d = 0;
                while (*end) {
                    if (*end == '\\') { end += 2; continue; }
                    if (*end == '"') {
                        ++end;
                        while (*end) {
                            if (*end == '\\') { end += 2; continue; }
                            if (*end == '"') { ++end; break; }
                            ++end;
                        }
                        continue;
                    }
                    if (*end == '{' || *end == '[') ++d;
                    else if (*end == '}' || *end == ']') { --d; if (d == 0) { ++end; break; } }
                    ++end;
                }
            } else {
                while (*end && *end != ',' && *end != '}') ++end;
            }
            if (len) *len = (size_t)(end - v);
            return v;
        }
        p = q + 1;
    }
    return NULL;
}

/* response_format 分类。
 * ★ 只看【顶层】 response_format 的值，且在其值区间内找类型名。
 *   不以 raw body 全文 strstr —— 那会让任意深度的同名嵌套键（例如
 *   metadata.response_format）被当成请求参数，直接 400 掉一个无关请求。
 * json_schema 先判：它的对象里嵌着同名键，也可能嵌 json_object 字样。 */
int server_response_format_kind(const char *body) {
    if (!body) return SERVER_RF_NONE;
    size_t vlen = 0;
    const char *v = json_top_field_value(body, "response_format", &vlen);
    if (!v || !vlen) return SERVER_RF_NONE;
    static const char kSchema[] = "\"json_schema\"";
    static const char kObject[] = "\"json_object\"";
    for (size_t i = 0; i + sizeof(kSchema) - 1u <= vlen; i++) {
        if (memcmp(v + i, kSchema, sizeof(kSchema) - 1u) == 0) return SERVER_RF_JSON_SCHEMA;
    }
    for (size_t i = 0; i + sizeof(kObject) - 1u <= vlen; i++) {
        if (memcmp(v + i, kObject, sizeof(kObject) - 1u) == 0) return SERVER_RF_JSON_OBJECT;
    }
    return SERVER_RF_NONE;
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
/* ── base64 解码（客户端把图片塞在 data URL 里）──
 * 只接受标准字母表 + '=' 填充；遇到空白跳过（有些客户端会换行）。
 * 返回 malloc 出来的字节（调用方 free），长度为 0 或 NULL 表示失败。 */
static uint8_t *server_base64_decode(const char *text, size_t len, size_t *out_len) {
    static int8_t rev[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; i++) rev[i] = -1;
        const char *alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) rev[(unsigned char)alpha[i]] = (int8_t)i;
        init = true;
    }
    uint8_t *out = (uint8_t *)malloc(len / 4u * 3u + 4u);
    if (!out) return NULL;
    size_t used = 0;
    uint32_t acc = 0;
    int nbits = 0;
    for (size_t i = 0; i < len; i++) {
        const unsigned char c = (unsigned char)text[i];
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int8_t v = rev[c];
        if (v < 0) { free(out); return NULL; }
        acc = (acc << 6) | (uint32_t)v;
        nbits += 6;
        if (nbits >= 8) {
            nbits -= 8;
            out[used++] = (uint8_t)((acc >> nbits) & 0xffu);
        }
    }
    if (used == 0) { free(out); return NULL; }
    *out_len = used;
    return out;
}

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
/* Append a run of ordinary bytes with the same overflow behavior as appending
 * each byte separately. Unlike prompt_append_n, a run which does not fit must
 * leave the largest possible prefix in the output before reporting overflow. */
static bool prompt_append_run(prompt_buffer_t *b, const char *text, size_t length) {
    if (b->length >= b->capacity) { b->overflow = true; return false; }
    const size_t available = b->capacity - b->length - 1u;
    if (length > available) {
        if (available && !prompt_append_n(b, text, available)) return false;
        b->overflow = true;
        return false;
    }
    return prompt_append_n(b, text, length);
}
/* 处理一个 input_image 内容块：取出 data URL、解码、编一次图，
 * 然后把它对应的【占位文本】追加进提示词。向量本身留在 sink 里，
 * 等提示词全部拼好后由调用方按位置注入。 */
/* 每条早退都报原因：静默失败会让「图片没生效」变成一个很难查的现象
 * ——第一版就是只回一句「不支持的内容块」，看不出到底哪一步不认。 */
/* ★ 宏参数名不能取 why：预处理器按 token 替换，会把 img->why 里的 why 一并替掉，
 *   变成 img->"…"。这种错编译器只报 "expected identifier"，很难一眼看出。 */
#define IMG_REJECT(msg_) do { \
        MYDS_LOG_E("input_image 被拒: %s", (msg_)); \
        if (img) snprintf(img->why, sizeof(img->why), "%.190s", (msg_)); \
        return false; } while (0)

static bool prompt_content_image(const prompt_json_t *j, const prompt_json_node_t *node,
                                 prompt_buffer_t *out, myds_vjob_sink_t *img) {
    if (!img) IMG_REJECT("服务端未启用视觉（启动时没给 --vision？）");
    bool valid = true;
    const prompt_json_node_t *url = prompt_json_field(j, node, "image_url", &valid);
    if (!valid) IMG_REJECT("image_url 字段重复或结构异常");
    if (url && url->kind == PJ_OBJECT) {    /* Chat Completions 风格 {"url": "..."} */
        url = prompt_json_field(j, url, "url", &valid);
        if (!valid) IMG_REJECT("image_url.url 字段重复或结构异常");
    }
    if (!url) IMG_REJECT("没有 image_url 字段");
    if (url->kind != PJ_STRING) IMG_REJECT("image_url 不是字符串（只支持 data: 内联 URL）");
    const char *spec = url->text;
    MYDS_LOG_I("input_image: image_url 前 40 字节 = %.40s", spec);
    if (strncmp(spec, "data:", 5) != 0) IMG_REJECT("不是 data: 内联 URL");
    const char *comma = strchr(spec, ',');
    if (!comma) IMG_REJECT("data URL 里没有逗号分隔");
    if (!strstr(spec, "base64")) IMG_REJECT("data URL 不是 base64 编码");
    size_t nbytes = 0;
    uint8_t *bytes = server_base64_decode(comma + 1, strlen(comma + 1), &nbytes);
    if (!bytes) IMG_REJECT("base64 解码失败（含非法字符或为空）");
    MYDS_LOG_I("input_image: base64 解出 %zu 字节", nbytes);

    char err[256] = {0};
    /* ★ 追加而不是覆盖：网关每轮重发完整历史，历史里可能有多张图。
     *   第一版只支持一张，结果「发过图之后的任何对话」都会被拒。 */
    if (!myds_vjob_sink_add(img, bytes, nbytes, err, sizeof(err))) {
        MYDS_LOG_E("图片处理失败: %s", err[0] ? err : "未知原因");
        free(bytes);
        return false;
    }
    free(bytes);
    const myds_vjob_plan_t *plan = &img->plans[img->count - 1];
    MYDS_LOG_I("收到图片 #%u: 网格 %ux%u → %u 个占位 token（累计 %u 张）",
               img->count, plan->grid_width, plan->grid_height, plan->token_count, img->count);
    return prompt_append(out, plan->text);
}

static bool prompt_tool_text(prompt_buffer_t *out, const char *text) {
    if (!prompt_append(out, "<tool_result>")) return false;
    const char *run = text, *p = text;
    for (; *p; ++p) {
        /* Stop as soon as the next byte cannot fit: the old byte-at-a-time
         * loop did not scan the rest of an oversized value on failure. */
        if ((size_t)(p - run) >= out->capacity - out->length - 1u) {
            if (!prompt_append_run(out, run, (size_t)(p - run))) return false;
            out->overflow = true;
            return false;
        }
        if (*p != '<' && *p != '&') continue;
        const char *q = p;
        if (*q == '&') { ++q; while (!strncmp(q, "amp;", 4)) q += 4; }
        bool escape = !strncmp(p, "</tool_result>", 14) || (*p == '&' && !strncmp(q, "lt;/tool_result>", 16));
        if (escape) {
            if ((p != run && !prompt_append_run(out, run, (size_t)(p - run))) ||
                !prompt_append(out, *p == '<' ? "&lt;" : "&amp;")) return false;
            run = p + 1;
        }
    }
    return (p == run || prompt_append_run(out, run, (size_t)(p - run))) &&
           prompt_append(out, "</tool_result>");
}

static bool prompt_content(const prompt_json_t *j, const prompt_json_node_t *content,
                           prompt_buffer_t *out, myds_vjob_sink_t *img) {
    if (!content) return false;
    if (content->kind == PJ_NULL) return true;
    if (content->kind == PJ_STRING) return prompt_append(out, content->text);
    if (content->kind != PJ_ARRAY) return false;
    size_t count = 0;
    for (size_t i = (size_t)(content - j->nodes) + 1; i < content->next; i = j->nodes[i].next) {
        bool valid = true;
        const prompt_json_node_t *type = prompt_json_field(j, &j->nodes[i], "type", &valid);
        const prompt_json_node_t *text = prompt_json_field(j, &j->nodes[i], "text", &valid);
        if (!valid || !type || type->kind != PJ_STRING) return false;
        if (!strcmp(type->text, "input_image")) {
            if (count++ && !prompt_append(out, "\n\n")) return false;
            if (!prompt_content_image(j, &j->nodes[i], out, img)) return false;
            continue;
        }
        if (!strcmp(type->text, "tool_result")) {
            const prompt_json_node_t *t_content = prompt_json_field(j, &j->nodes[i], "content", &valid);
            if (!t_content) t_content = text;
            if (count++ && !prompt_append(out, "\n\n")) return false;
            if (t_content && t_content->kind == PJ_STRING) {
                if (!prompt_tool_text(out, t_content->text)) return false;
            } else if (t_content && t_content->kind == PJ_ARRAY) {
                char *tmp = malloc(out->capacity);
                if (!tmp) return false;
                prompt_buffer_t tb = {tmp, out->capacity, 0, false};
                if (!prompt_content(j, t_content, &tb, img)) { free(tmp); return false; }
                bool ok = prompt_tool_text(out, tb.data);
                free(tmp);
                if (!ok) return false;
            }
            continue;
        }
        if ((strcmp(type->text, "text") && strcmp(type->text, "input_text") && strcmp(type->text, "output_text")) ||
            !text ||
            (text->kind != PJ_NULL && text->kind != PJ_STRING)) return false;
        if (text->kind == PJ_STRING) {
            if (count++ && !prompt_append(out, "\n\n")) return false;
            if (!prompt_append(out, text->text)) return false;
        }
    }
    return true;
}
typedef struct { bool pending, user_open, system_open; size_t count; } prompt_chat_state_t;
static bool prompt_render_message(prompt_buffer_t *out, prompt_chat_state_t *state, const char *role,
                                  const char *text, const char *suffix, const char *reasoning) {
    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        if (!state->system_open) {
            if (!prompt_append(out, "<｜System｜>")) return false;
        }
        state->system_open = false;
        if (!prompt_append(out, text)) return false;
        (void)suffix; (void)reasoning;
        state->pending = state->count > 0; state->user_open = false;
    } else if (!strcmp(role, "user") || !strcmp(role, "tool") || !strcmp(role, "function")) {
        (void)suffix; (void)reasoning;
        state->system_open = false;
        if (!prompt_append(out, state->user_open ? "\n\n" : "<｜User｜>")) return false;
        if (!strcmp(role, "user") ? !prompt_append(out, text) : !prompt_tool_text(out, text)) return false;
        state->pending = state->user_open = true;
    } else if (!strcmp(role, "assistant")) {
        if (state->pending) {
            if (reasoning && reasoning[0]) {
                if (!prompt_append(out, "<｜Assistant｜><think>") ||
                    !prompt_append(out, reasoning) ||
                    !prompt_append(out, "</think>")) return false;
            } else if (text && strncmp(text, "<think>", 7) == 0) {
                if (!prompt_append(out, "<｜Assistant｜>")) return false;
            } else {
                if (!prompt_append(out, "<｜Assistant｜></think>")) return false;
            }
        }
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

static bool prompt_dump_json_string(prompt_buffer_t *out, const char *s) {
    if (!prompt_append(out, "\"")) return false;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        switch (*p) {
            case '"': if (!prompt_append(out, "\\\"")) return false; break;
            case '\\': if (!prompt_append(out, "\\\\")) return false; break;
            case '\b': if (!prompt_append(out, "\\b")) return false; break;
            case '\f': if (!prompt_append(out, "\\f")) return false; break;
            case '\n': if (!prompt_append(out, "\\n")) return false; break;
            case '\r': if (!prompt_append(out, "\\r")) return false; break;
            case '\t': if (!prompt_append(out, "\\t")) return false; break;
            default:
                if (*p < 0x20) {
                    char esc[8];
                    snprintf(esc, sizeof(esc), "\\u%04x", *p);
                    if (!prompt_append(out, esc)) return false;
                } else {
                    char b[2] = {(char)*p, 0};
                    if (!prompt_append(out, b)) return false;
                }
                break;
        }
    }
    return prompt_append(out, "\"");
}

static bool prompt_dump_json_node(prompt_buffer_t *out, const prompt_json_t *j,
                                  const prompt_json_node_t *node) {
    if (!node) return false;
    switch (node->kind) {
        case PJ_NULL:
            return prompt_append(out, "null");
        case PJ_STRING:
            return prompt_dump_json_string(out, node->text ? node->text : "");
        case PJ_SCALAR:
            if (!j->base || !node->len) return false;
            return prompt_append_n(out, j->base + node->off, node->len);
        case PJ_ARRAY: {
            if (!prompt_append(out, "[")) return false;
            size_t count = 0;
            for (size_t i = (size_t)(node - j->nodes) + 1; i < node->next; i = j->nodes[i].next) {
                if (count++ && !prompt_append(out, ", ")) return false;
                if (!prompt_dump_json_node(out, j, &j->nodes[i])) return false;
            }
            return prompt_append(out, "]");
        }
        case PJ_OBJECT: {
            if (!prompt_append(out, "{")) return false;
            size_t count = 0;
            for (size_t i = (size_t)(node - j->nodes) + 1; i < node->next; ) {
                const prompt_json_node_t *key = &j->nodes[i];
                const prompt_json_node_t *val = &j->nodes[key->next];
                if (count++ && !prompt_append(out, ", ")) return false;
                if (!prompt_dump_json_string(out, key->text ? key->text : "")) return false;
                if (!prompt_append(out, ": ")) return false;
                if (!prompt_dump_json_node(out, j, val)) return false;
                i = val->next;
            }
            return prompt_append(out, "}");
        }
        default:
            return false;
    }
}

/* 一行工具 schema：{"name": ...} 采用与官方 to_json(json.dumps ensure_ascii=False) 完全一致的格式 */
static bool prompt_append_tool_schema(prompt_buffer_t *out, const prompt_json_t *j,
                                      const prompt_json_node_t *tool) {
    if (!tool || tool->kind != PJ_OBJECT) return false;
    bool valid = true;
    const prompt_json_node_t *fn = prompt_json_field(j, tool, "function", &valid);
    const prompt_json_node_t *base = (fn && fn->kind == PJ_OBJECT) ? fn : tool;
    if (!valid || base->kind != PJ_OBJECT) return false;

    /* 提取 namespace（支持字符串或带 name/description 的对象） */
    bool v_ns = true;
    const prompt_json_node_t *ns_node = prompt_json_field(j, tool, "namespace", &v_ns);
    if (!ns_node) ns_node = prompt_json_field(j, base, "namespace", &v_ns);
    const char *ns_name = NULL;
    const char *ns_desc = NULL;
    if (ns_node) {
        if (ns_node->kind == PJ_STRING) {
            ns_name = ns_node->text;
        } else if (ns_node->kind == PJ_OBJECT) {
            bool v_name = true, v_desc = true;
            const prompt_json_node_t *n_val = prompt_json_field(j, ns_node, "name", &v_name);
            const prompt_json_node_t *d_val = prompt_json_field(j, ns_node, "description", &v_desc);
            if (n_val && n_val->kind == PJ_STRING) ns_name = n_val->text;
            if (d_val && d_val->kind == PJ_STRING) ns_desc = d_val->text;
        }
    }

    bool v_name = true;
    const prompt_json_node_t *name_node = prompt_json_field(j, base, "name", &v_name);
    if (!v_name || !name_node || name_node->kind != PJ_STRING || !prompt_name_ok(name_node->text))
        return false;

    char qualified_name[256];
    const char *colon_colon = strstr(name_node->text, "::");
    if (colon_colon) {
        size_t prefix_len = (size_t)(colon_colon - name_node->text);
        if (ns_name && (strncmp(ns_name, name_node->text, prefix_len) != 0 || ns_name[prefix_len] != 0)) {
            return false; /* conflicting namespace */
        }
        snprintf(qualified_name, sizeof(qualified_name), "%s", name_node->text);
    } else if (ns_name && *ns_name) {
        snprintf(qualified_name, sizeof(qualified_name), "%s::%s", ns_name, name_node->text);
    } else {
        snprintf(qualified_name, sizeof(qualified_name), "%s", name_node->text);
    }

    if (!prompt_append(out, "{")) return false;
    size_t count = 0;
    for (size_t i = (size_t)(base - j->nodes) + 1; i < base->next; ) {
        const prompt_json_node_t *key = &j->nodes[i];
        const prompt_json_node_t *val = &j->nodes[key->next];
        i = val->next;
        if (!key->text) continue;
        if (!strcmp(key->text, "namespace")) continue;

        if (count++ && !prompt_append(out, ", ")) return false;
        if (!prompt_dump_json_string(out, key->text)) return false;
        if (!prompt_append(out, ": ")) return false;

        if (!strcmp(key->text, "name")) {
            if (!prompt_dump_json_string(out, qualified_name)) return false;
        } else if (!strcmp(key->text, "description") && ns_desc && *ns_desc) {
            char merged_desc[2048];
            snprintf(merged_desc, sizeof(merged_desc), "%s\n%s", ns_desc, val->text ? val->text : "");
            if (!prompt_dump_json_string(out, merged_desc)) return false;
        } else {
            if (!prompt_dump_json_node(out, j, val)) return false;
        }
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
                              "schemas to invoke tool calls.\n");
}

/* assistant 历史里的工具调用 ⇒ DSML 块（返回值以 "\n\n" 开头，便于直接接到 content 之后）。 */
/* 渲染一个 <｜DSML｜ invoke>。entry 可以是 {"function":{name,arguments}}（OpenAI 嵌套）
 * 或扁平的 {name,arguments}（Responses）。 */
static bool prompt_emit_invoke(prompt_buffer_t *out, const prompt_json_t *j,
                               const prompt_json_node_t *entry) {
    if (!entry || entry->kind != PJ_OBJECT) return false;
    bool valid = true;
    const prompt_json_node_t *fn = prompt_json_field(j, entry, "function", &valid);
    const prompt_json_node_t *base = (fn && fn->kind == PJ_OBJECT) ? fn : entry;
    const prompt_json_node_t *name = prompt_json_field(j, base, "name", &valid);
    const prompt_json_node_t *args = prompt_json_field(j, base, "arguments", &valid);
    if (!args) args = prompt_json_field(j, base, "parameters", &valid);
    if (!valid || !name || name->kind != PJ_STRING || !prompt_name_ok(name->text)) return false;
    if (!prompt_append(out, PROMPT_TOOL_INVOKE_OPEN) ||
        !prompt_append(out, name->text) ||
        !prompt_append(out, "\">")) return false;

    /* arguments 是 JSON 字符串（OpenAI）或对象（Responses）；统一解析成对象再逐参数渲染。 */
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
            bool okp = prompt_append(out, "\n");
            size_t pn = 0;
            for (size_t k = 1; okp && k < aj.nodes[0].next;) {
                const prompt_json_node_t *key = &aj.nodes[k];
                const prompt_json_node_t *val = &aj.nodes[key->next];
                if (pn++ && !prompt_append(out, "\n")) { okp = false; break; }
                if (!prompt_append(out, PROMPT_TOOL_PARAM_OPEN) ||
                    !prompt_append(out, key->text ? key->text : "") ||
                    !prompt_append(out, "\" string=\"")) { okp = false; break; }
                if (val->kind == PJ_STRING) {
                    if (!prompt_append(out, "true\">") ||
                        !prompt_append(out, val->text ? val->text : "")) { okp = false; break; }
                } else {
                    size_t vl = 0; const char *vraw = prompt_node_raw(&aj, val, &vl);
                    if (!prompt_append(out, "false\">") || !vraw ||
                        !prompt_append_n(out, vraw, vl)) { okp = false; break; }
                }
                if (!prompt_append(out, PROMPT_TOOL_PARAM_CLOSE)) { okp = false; break; }
                k = val->next;
            }
            for (size_t k = 0; k < aj.count; ++k) free(aj.nodes[k].text);
            free(aj.nodes);
            if (!okp) { free(arg_copy); return false; }
        }
    }
    free(arg_copy);
    return prompt_append(out, "\n" PROMPT_TOOL_INVOKE_CLOSE);
}

/* 一个 DSML 块：兼容三种形状（数组 tool_calls / 单对象 function_call / 含 tool_calls 的对象）。 */
static bool prompt_render_tool_call_block(prompt_buffer_t *out, const prompt_json_t *j,
                                          const prompt_json_node_t *calls) {
    if (!calls) return false;
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
        if (emitted++ && !prompt_append(out, "\n")) return false;
        if (!prompt_emit_invoke(out, j, entry)) return false;
    }
    if (!emitted) return false;
    return prompt_append(out, "\n" PROMPT_TOOL_CALLS_CLOSE);
}

/* Responses 形状：正文是 message 项、调用是紧随其后的 function_call 项。模型把两者生成在
 * 【同一个 assistant 轮】里（官方模板 "{reasoning}{content}{tool_calls}" + 单个 EOS），
 * 所以这里把连续的一串 function_call 项合并成【一个】DSML 块，挂在同一轮末尾。
 * 不合并的话会多出 EOS 与 <｜Assistant｜> ⇒ 下一轮前缀缓存对不上，每轮都全量重跑 prefill。 */
static bool prompt_render_tool_call_items(prompt_buffer_t *out, const prompt_json_t *j,
                                          size_t begin, size_t last) {
    size_t stop = j->nodes[last].next;
    if (!prompt_append(out, "\n\n" PROMPT_TOOL_CALLS_OPEN "\n")) return false;
    size_t emitted = 0;
    for (size_t k = begin; k < stop; k = j->nodes[k].next) {
        if (emitted++ && !prompt_append(out, "\n")) return false;
        if (!prompt_emit_invoke(out, j, &j->nodes[k])) return false;
    }
    if (!emitted) return false;
    return prompt_append(out, "\n" PROMPT_TOOL_CALLS_CLOSE);
}

/* 失败原因（供日志与报错使用）。调用方每次进入前不必清空 —— 每个失败路径都会写它。 */
bool g_server_default_thinking = false;

/* ── CORS（2026-10-25）────────────────────────────────────────
 *
 * 默认仍是 `*`：这个服务原本就是给本机/内网工具直接调的，收紧默认值会
 * 默默弄坏现有的浏览器页面。但“无鉴权 + 任意源”这个组合必须能被关掉，
 * 否则把服务放到反向代理后面时，浏览器端可以任意页面对它发请求。
 *
 * MYDS_CORS_ORIGIN 的取值：
 *   未设 / `*`        → 保持旧行为
 *   `none`            → 不发送任何 CORS 头（纯后端/同源部署）
 *   具体 origin       → 只回这一个（如 https://example.com）
 *
 * 注意：这里按需求“回显”而不是做复杂校验——多 origin 名单用逗号会引入
 * 解析与回显错误的风险，而 CORS 本身不是安全边界（它只约束浏览器读取响应）。
 * 真正的安全手段是鉴权与本机监听。 */
const char *cors_origin_header(void) {
    static const char *cached = NULL;
    static bool resolved = false;
    if (resolved) return cached;
    resolved = true;
    cached = "*";
    const char *e = getenv("MYDS_CORS_ORIGIN");
    if (!e || !*e) return cached;
    if (strcmp(e, "none") == 0) { cached = NULL; return cached; }
    cached = e;
    MYDS_LOG_I("CORS allow-origin = %s", cached);
    return cached;
}

/* 从请求体的 `reasoning` **对象**里读 effort。
 *
 * 2026-10-24：mycode（以及 OpenAI Responses 生态的多数客户端）用
 *     "reasoning": {"effort": "high", "summary": "detailed"}
 * 表达思考强度，而此前 myds 只认顶层的 "reasoning_effort" 字符串 ⇒
 * 这类请求被完全忽略、thinking 回落到启动默认值。实测判别：
 *     带 reasoning:{effort:"none"} → 无 reasoning 输出（对）
 *     带 reasoning:{effort:"high"} → 无 reasoning 输出（错！应当有）
 *     带 reasoning_effort:"high"   → 有 reasoning 输出（对）
 * 这里把对象形式补上。返回 1 = 读到了 effort（或 enabled 折算成的 effort）。 */
int extract_reasoning_object_effort(const char *body, char *out, size_t out_size) {
    if (!body || !out || out_size == 0) return 0;
    /* "reasoning" 也可能作为【值】出现（input 数组里的 {"type":"reasoning"}）——
     * 只有后面紧跟 '{' 的才是我们关心的对象。 */
    for (const char *p = strstr(body, "\"reasoning\""); p; p = strstr(p + 1, "\"reasoning\"")) {
        const char *q = p + 11;   /* strlen("\"reasoning\"") */
        while (*q && (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n' || *q == ':')) q++;
        if (*q != '{') continue;
        const char *end = find_matching_brace(q);
        if (!end) continue;
        const size_t n2 = (size_t)(end - q + 1);
        if (n2 >= 512u) continue;         /* 异常大的对象不处理 */
        char buf[512];
        memcpy(buf, q, n2);
        buf[n2] = 0;
        if (json_extract_string_field(buf, "effort", out, out_size)) return 1;
        const int en = json_extract_bool_field(buf, "enabled", -1);
        if (en != -1) {
            snprintf(out, out_size, "%s", en ? "high" : "none");
            return 1;
        }
    }
    return 0;
}

bool extract_request_thinking_mode(const char *body, bool default_val) {
    if (!body) return default_val;
    int b = json_extract_bool_field(body, "enable_thinking", -1);
    if (b != -1) return b != 0;

    const char *p = strstr(body, "\"thinking\"");
    if (p) {
        p += 10;
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
        if (strncmp(p, "true", 4) == 0) return true;
        if (strncmp(p, "false", 5) == 0) return false;
        if (*p == '{') {
            const char *end = find_matching_brace(p);
            if (end) {
                if (strstr(p, "\"enabled\"")) return true;
                if (strstr(p, "\"disabled\"")) return false;
            }
        }
    }

    /* mycode / OpenAI Responses 用对象形式表达思考强度。
     * 值域按 DeepSeek 官方 thinking_mode 文档的映射表（2026-10 版）：
     *   minimal→low  low→low  medium→high  high→high  xhigh→high  max→max  ultra→max
     *   none 表示关闭（这是 Responses API 里唯一的关闭表达）。
     * 未识别的值不在这里下结论，交给后面的字段与默认值。 */
    char obj_effort[32] = "";
    if (extract_reasoning_object_effort(body, obj_effort, sizeof(obj_effort))) {
        if (!strcmp(obj_effort, "none") || !strcmp(obj_effort, "off") ||
            !strcmp(obj_effort, "disabled")) return false;
        if (!strcmp(obj_effort, "minimal") || !strcmp(obj_effort, "low") ||
            !strcmp(obj_effort, "medium") || !strcmp(obj_effort, "high") ||
            !strcmp(obj_effort, "xhigh") || !strcmp(obj_effort, "max") ||
            !strcmp(obj_effort, "ultra")) return true;
    }

    char effort[32] = "";
    if (json_extract_string_field(body, "reasoning_effort", effort, sizeof(effort))) {
        if (!strcmp(effort, "none") || !strcmp(effort, "off")) return false;
        if (!strcmp(effort, "minimal") || !strcmp(effort, "low") || !strcmp(effort, "medium") ||
            !strcmp(effort, "high") || !strcmp(effort, "xhigh") || !strcmp(effort, "max") ||
            !strcmp(effort, "ultra")) return true;
    }

    const char *ctk = strstr(body, "\"chat_template_kwargs\"");
    if (ctk) {
        ctk += 22;
        while (*ctk && (*ctk == ' ' || *ctk == '\t' || *ctk == '\r' || *ctk == '\n' || *ctk == ':')) ctk++;
        if (*ctk == '{') {
            int tb = json_extract_bool_field(ctk, "thinking", -1);
            if (tb != -1) return tb != 0;
        }
    }

    return default_val;
}

int extract_request_reasoning_budget(const char *body, int default_budget) {
    int budget = default_budget > 0 ? default_budget : 75;
    const char *env_eff = getenv("MYDS_DEFAULT_REASONING_EFFORT");
    if (env_eff) {
        if (!strcmp(env_eff, "low")) budget = 50;
        else if (!strcmp(env_eff, "medium")) budget = 62;
        else if (!strcmp(env_eff, "high")) budget = 75;
        else if (!strcmp(env_eff, "max")) budget = 100;
        else {
            int val = atoi(env_eff);
            if (val >= 1 && val <= 100) budget = val;
        }
    }
    if (!body) return budget;

    /* 强度映射。值域按 DeepSeek 官方 thinking_mode 文档的映射表：
     *   minimal→low  low→low  medium→high  high→high  xhigh→high  max→max  ultra→max
     * myds 在 low/high 之间保留了 medium=62 这一档更细的粒度（官方把 medium 归到 high），
     * 既覆盖官方全部取值，也不改变既有行为。 */
    char effort[32] = "";
    bool have_effort = json_extract_string_field(body, "reasoning_effort", effort, sizeof(effort)) != 0;
    if (!have_effort) have_effort = extract_reasoning_object_effort(body, effort, sizeof(effort)) != 0;
    if (have_effort) {
        if (!strcmp(effort, "minimal") || !strcmp(effort, "low")) return 50;
        if (!strcmp(effort, "medium")) return 62;
        if (!strcmp(effort, "high") || !strcmp(effort, "xhigh")) return 75;
        if (!strcmp(effort, "max") || !strcmp(effort, "ultra")) return 100;
    }
    const char *p = strstr(body, "\"reasoning_effort\"");
    if (p) {
        p += 18;
    } else {
        p = strstr(body, "\"reasoning_budget\"");
        if (p) p += 18;
    }
    if (p) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':')) p++;
        if (*p >= '0' && *p <= '9') {
            int val = atoi(p);
            if (val >= 1 && val <= 100) return val;
        }
    }
    return budget;
}

void split_reasoning_content(const char *text, size_t n, char **content_out, char **reasoning_out) {
    if (!content_out || !reasoning_out) return;
    *content_out = NULL;
    *reasoning_out = NULL;
    if (!text || n == 0) {
        *content_out = calloc(1, 1);
        return;
    }

    const char *think_end = strstr(text, "</think>");
    if (think_end && (size_t)(think_end - text) <= n) {
        const char *t_start = text;
        size_t r_len = (size_t)(think_end - t_start);
        if (strncmp(t_start, "<think>", 7) == 0) {
            t_start += 7;
            if (r_len >= 7) r_len -= 7;
            else r_len = 0;
        }
        *reasoning_out = malloc(r_len + 1);
        if (*reasoning_out) {
            memcpy(*reasoning_out, t_start, r_len);
            (*reasoning_out)[r_len] = '\0';
        }

        const char *c_start = think_end + 8;
        while (c_start < text + n && (*c_start == '\r' || *c_start == '\n')) c_start++;
        size_t c_offset = (size_t)(c_start - text);
        if (c_offset <= n) {
            size_t c_len = n - c_offset;
            *content_out = malloc(c_len + 1);
            if (*content_out) {
                memcpy(*content_out, c_start, c_len);
                (*content_out)[c_len] = '\0';
            }
        } else {
            *content_out = calloc(1, 1);
        }
    } else if (strncmp(text, "<think>", 7) == 0) {
        size_t r_len = n > 7 ? n - 7 : 0;
        *reasoning_out = malloc(r_len + 1);
        if (*reasoning_out) {
            memcpy(*reasoning_out, text + 7, r_len);
            (*reasoning_out)[r_len] = '\0';
        }
        *content_out = calloc(1, 1);
    } else {
        *reasoning_out = NULL;
        *content_out = malloc(n + 1);
        if (*content_out) {
            memcpy(*content_out, text, n);
            (*content_out)[n] = '\0';
        }
    }
}

/* 工具集指纹：把 tools/functions 段用【渲染器本身】渲染成规范文本后做 FNV-1a。
 *
 * 为什么不能直接哈希原始 JSON 字节：同一个工具集只要空白、键序、数字写法有一点差异
 * 就会算出不同指纹，把一个正常续跑误判成「工具变了」。走 prompt_render_tools 则两边
 * 都是同一套规范化结果，字节一致当且仅当语义一致。
 *
 * 返回 0 表示【无 tools】（或解析失败）；带 tools 时返回值强制非 0，以区分两者。 */
unsigned long long server_tools_signature(const char *body, size_t scratch_cap) {
    if (!body || !scratch_cap) return 0ull;
    prompt_json_t json = {.cursor = body, .base = body};
    unsigned long long sig = 0ull;
    char *buf = NULL;
    if (!prompt_json_value(&json, 0)) goto done;
    if (json.nodes[0].kind != PJ_OBJECT) goto done;
    {
        bool valid = true;
        const prompt_json_node_t *root = &json.nodes[0];
        const prompt_json_node_t *t = prompt_json_field(&json, root, "tools", &valid);
        const prompt_json_node_t *f = prompt_json_field(&json, root, "functions", &valid);
        if (!valid) goto done;
        const prompt_json_node_t *defs = !prompt_absent_or_empty(&json, t) ? t : f;
        if (prompt_absent_or_empty(&json, defs)) goto done;
        buf = malloc(scratch_cap);
        if (!buf) goto done;
        {
            prompt_buffer_t pb = {buf, scratch_cap, 0, false};
            if (!prompt_render_tools(&pb, &json, defs)) goto freebuf;
            unsigned long long h = 1469598103934665603ull;
            for (size_t i = 0; i < pb.length; ++i) {
                h ^= (unsigned long long)(unsigned char)buf[i];
                h *= 1099511628211ull;
            }
            sig = h ? h : 1ull;      /* 0 保留给「无 tools」 */
        }
    }
freebuf:
    free(buf);
done:
    for (size_t i = 0; i < json.count; ++i) free(json.nodes[i].text);
    free(json.nodes);
    return sig;
}

static bool parse_chat_prompt_impl(const char *body, char *out_prompt, size_t max_prompt_len,
                                   const char **out_reason, bool suffix_only,
                                   myds_vjob_sink_t *img) {
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
    const bool req_thinking = extract_request_thinking_mode(body, g_server_default_thinking);
    const int reasoning_budget = extract_request_reasoning_budget(body, 75);
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
    /* 历史里有没有 system 消息 —— ★ 与 has_tools 无关：
     * 之前只在有 tools 时才算，于是「无 tools 但历史带 system」时判定漏掉，
     * 系统提示词照样进两遍。这里无条件扫（只扫 items 的角色字段，代价可忽略）。 */
    bool system_in_items = false;
    const prompt_json_node_t *first_system = NULL;
    if (items->kind == PJ_ARRAY) {
        for (size_t i = (size_t)(items - json.nodes) + 1; i < items->next; i = json.nodes[i].next) {
            bool v2 = true;
            const prompt_json_node_t *r = prompt_json_field(&json, &json.nodes[i], "role", &v2);
            if (r && r->kind == PJ_STRING && !strcmp(r->text, "system")) {
                system_in_items = true;
                if (!first_system) first_system = &json.nodes[i];
            }
        }
    }
    bool tools_done = false;
    /* suffix 模式（previous_response_id）：服务端会把【上一次响应的完整序列】接在前面，
     * 所以这里只渲染【本次新增的消息】，并且不能带 BOS、instructions 与 tools
     * —— 它们已经在被续接的那段序列里了，重复渲染会把前缀整体错位。 */
    if (suffix_only) {
        /* ★ 必须先补上【上一个 assistant 回合的 EOS】。
         * 被续接的那段序列停在生成文本之后 —— 生成时 prompt 末尾是
         * "<｜Assistant｜></think>"，生成的 token 被喂进去，但 EOS【从未被喂】
         * （EOS 只在渲染【历史中的】assistant 消息时追加）。
         * 而全量重放那条路径里，assistant 消息会渲染成 "…" + EOS。
         * 不补这一个 token，续跑的序列就与全量重放【差一个 EOS】——
         * 实测 prompt 会少 1（6427 vs 6428），属静默错位。 */
        if (!prompt_append(&out, "<｜end▁of▁sentence｜>")) goto done;
        tools_done = true;
        goto render_items;
    }
    if (!prompt_append(&out, "<｜begin▁of▁sentence｜>")) { *why = "the prompt buffer is too small"; goto done; }
    if (req_thinking) {
        char effort_hdr[160];
        snprintf(effort_hdr, sizeof(effort_hdr),
                 "<｜System｜>Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n",
                 reasoning_budget);
        if (!prompt_append(&out, effort_hdr)) { *why = "the prompt buffer is too small"; goto done; }
        state.system_open = true;
    }
    /* ── 系统提示词去重（2026-10-01 线上实证）──
     * 客户端可能【同时】发 instructions 字段、又在 input 历史里带一条 system 消息
     * （OpenClaw 就是这样）。旧逻辑只要看到 instructions 就渲染 ⇒ 系统提示词进两遍：
     *   实测 渲染出的提示词里系统提示词出现 2 次，prompt_chars 从 21073 涨到 42428。
     * 代价是 token 与延迟翻倍，而且模型会看到两遍同样的指令。
     * 规则：历史里已有 system 消息 ⇒ 以【历史】为准，跳过 instructions。
     *   · 历史是这段对话的既成事实，丢历史消息比丢一个重复字段更危险；
     *   · 这样渲染结果也更稳定（利于前缀缓存）。
     * tools 不受影响：它仍会渲染在历史那条 system 消息之后（见下方 items 循环）。 */
    if (instructions && instructions->kind != PJ_NULL && system_in_items) {
        /* 两边文本不同 ⇒ 我们正丢掉 instructions 字段里的一份内容。不静默做这件事。 */
        bool v3 = true;
        const prompt_json_node_t *sys_content =
            first_system ? prompt_json_field(&json, first_system, "content", &v3) : NULL;
        const char *hist_text = (sys_content && sys_content->kind == PJ_STRING) ? sys_content->text : NULL;
        if (!hist_text || strcmp(hist_text, instructions->text)) {
            MYDS_LOG_W("instructions 与历史里的 system 消息内容不同 —— 以【历史】为准，"
                       "本次忽略 instructions 字段（避免系统提示词进两遍）");
        }
    }
    if (instructions && instructions->kind != PJ_NULL && !system_in_items) {
        if (instructions->kind != PJ_STRING ||
            !prompt_render_message(&out, &state, "system", instructions->text, NULL, NULL)) goto done;
        if (has_tools) {
            if (!prompt_append(&out, "\n\n") || !prompt_render_tools(&out, &json, tool_defs)) {
                *why = "invalid or oversized tool definitions (tools/functions)";
                goto done;
            }
            tools_done = true;
        }
    } else if (has_tools && !system_in_items) {
        /* 没有 system 消息：自建一条只装工具定义 */
        if (!prompt_render_message(&out, &state, "system", "", NULL, NULL) ||
            !prompt_append(&out, "\n\n") || !prompt_render_tools(&out, &json, tool_defs)) {
            *why = "invalid or oversized tool definitions (tools/functions)";
            goto done;
        }
        tools_done = true;
    }
render_items:
    if (items->kind == PJ_STRING && !messages) {
        if (!prompt_render_message(&out, &state, "user", items->text, NULL, NULL)) goto done;
    } else if (items->kind == PJ_ARRAY && !prompt) {
        if ((size_t)(items - json.nodes) + 1 == items->next) { *why = "the message list is empty"; goto done; }
        content_buffer = malloc(max_prompt_len);
        if (!content_buffer) goto done;
        char last_call_ids[64][128];
        size_t last_call_id_count = 0;
        for (size_t i = (size_t)(items - json.nodes) + 1, i_next = 0; i < items->next; i = i_next) {
            i_next = json.nodes[i].next;   /* 默认推进一条；合并 function_call 时会跳过若干条 */
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
            if (!valid || (role && role->kind != PJ_STRING) || (type && type->kind != PJ_STRING)) {
                *why = "a message has a malformed role/type field";
                goto done;
            }
            /* ★ {type:"reasoning"} —— 模型内部项，不是对话内容。
             * OpenClaw 对 reasoning:true 的模型会把上一轮的 reasoning 项【连同历史一起回灌】
             * （Responses 契约如此），而它在这里曾经被当成未知 type 直接拒掉 ⇒
             * 任何【带历史的请求】都 400，整条链路不可用（实测 2026-10-01 与 10-02 两次踩中，
             * 而当时的小探针因为不带历史而全绿 —— 所以【必须用真实轮次验收】）。
             * 正确处置：安全跳过。它不贡献 prompt token，也不改变上下文。 */
            if (type && !strcmp(type->text, "reasoning")) continue;

            /* ── 工具结果排序（按官方 sort_tool_results_by_call_order 规则）──
             * 连续的 tool / function_call_output 消息按上一轮 assistant 的 tool_calls 顺序重排。 */
            bool is_tool = (role && role->kind == PJ_STRING && (!strcmp(role->text, "tool") || !strcmp(role->text, "function"))) ||
                           (type && type->kind == PJ_STRING && !strcmp(type->text, "function_call_output"));
            if (is_tool) {
                size_t tool_indices[64];
                int tool_ranks[64];
                size_t tool_count = 0;
                size_t scan = i;
                while (scan < items->next && tool_count < 64) {
                    bool v_chk = true;
                    const prompt_json_node_t *cur = &json.nodes[scan];
                    const prompt_json_node_t *r = prompt_json_field(&json, cur, "role", &v_chk);
                    const prompt_json_node_t *t = prompt_json_field(&json, cur, "type", &v_chk);
                    bool cur_is_tool = (r && r->kind == PJ_STRING && (!strcmp(r->text, "tool") || !strcmp(r->text, "function"))) ||
                                       (t && t->kind == PJ_STRING && !strcmp(t->text, "function_call_output"));
                    if (!cur_is_tool) break;
                    tool_indices[tool_count] = scan;

                    const prompt_json_node_t *tcid = prompt_json_field(&json, cur, "tool_call_id", &v_chk);
                    if (!tcid) tcid = prompt_json_field(&json, cur, "call_id", &v_chk);
                    if (!tcid) tcid = prompt_json_field(&json, cur, "id", &v_chk);

                    int rank = 1000 + (int)tool_count;
                    if (tcid && tcid->kind == PJ_STRING && tcid->text) {
                        for (size_t c = 0; c < last_call_id_count; ++c) {
                            if (!strcmp(last_call_ids[c], tcid->text)) {
                                rank = (int)c;
                                break;
                            }
                        }
                    }
                    tool_ranks[tool_count] = rank;
                    tool_count++;
                    scan = json.nodes[scan].next;
                }
                i_next = scan;

                /* 稳定插入排序 */
                for (size_t a = 1; a < tool_count; ++a) {
                    size_t key_idx = tool_indices[a];
                    int key_rank = tool_ranks[a];
                    size_t b = a;
                    while (b > 0 && tool_ranks[b - 1] > key_rank) {
                        tool_indices[b] = tool_indices[b - 1];
                        tool_ranks[b] = tool_ranks[b - 1];
                        b--;
                    }
                    tool_indices[b] = key_idx;
                    tool_ranks[b] = key_rank;
                }

                for (size_t a = 0; a < tool_count; ++a) {
                    const prompt_json_node_t *cur_item = &json.nodes[tool_indices[a]];
                    bool v_item = true;
                    const prompt_json_node_t *cur_type = prompt_json_field(&json, cur_item, "type", &v_item);
                    const prompt_json_node_t *cur_content = NULL;
                    if (cur_type && !strcmp(cur_type->text, "function_call_output")) {
                        cur_content = prompt_json_field(&json, cur_item, "output", &v_item);
                    } else {
                        cur_content = prompt_json_field(&json, cur_item, "content", &v_item);
                    }
                    if (!cur_content) cur_content = prompt_json_field(&json, cur_item, "text", &v_item);

                    prompt_buffer_t decoded = {content_buffer, max_prompt_len, 0, false};
                    content_buffer[0] = 0;
                    if (cur_content && !prompt_content(&json, cur_content, &decoded, img)) {
                        *why = "unsupported content block in tool result";
                        goto done;
                    }
                    if (!prompt_render_message(&out, &state, "tool", decoded.data, NULL, NULL)) {
                        *why = "a message could not be rendered (or the prompt buffer is full)";
                        goto done;
                    }
                }
                continue;
            }

            if (messages && !role) {
                *why = "a message has a malformed role/type field";
                goto done;
            }
            bool content_optional = false;
            if (type && !strcmp(type->text, "function_call_output")) {
                content = prompt_json_field(&json, item, "output", &valid); role_text = "tool";
            } else if (type && !strcmp(type->text, "function_call")) {
                /* Responses API 的标准多轮形状：把上一轮模型的 function_call 原样发回来。
                 * 它是一条【assistant 项】，name/arguments 就在这一层 ⇒ 直接当工具调用渲染。 */
                role_text = "assistant";
                tc = item;
                content_optional = true;
            } else if (type && strcmp(type->text, "message")) {
                *why = "a message has an unsupported \"type\" value";
                goto done;
            }
            if (!content) content = prompt_json_field(&json, item, "text", &valid);
            prompt_buffer_t decoded = {content_buffer, max_prompt_len, 0, false}; content_buffer[0] = 0;
            /* 工具调用只能挂在 assistant 上；其余角色出现即视为请求有误 */
            if (tc && strcmp(role_text, "assistant")) {
                *why = "tool_calls/function_call is only supported on assistant messages";
                goto done;
            }
            /* Responses 形状：调用是【紧随正文的 function_call 项】。模型把它生成在同一个
             * assistant 轮里 ⇒ 必须合并渲染，否则多出 EOS/<｜Assistant｜> 会让下一轮前缀失配。 */
            size_t run_begin = 0, run_last = 0;
            if (!strcmp(role_text, "assistant")) {
                size_t k = i_next;
                while (k < items->next) {
                    bool v4 = true;
                    const prompt_json_node_t *t2 = prompt_json_field(&json, &json.nodes[k], "type", &v4);
                    if (!v4 || !t2 || t2->kind != PJ_STRING || strcmp(t2->text, "function_call")) break;
                    if (!run_begin) run_begin = k;
                    run_last = k;
                    k = json.nodes[k].next;
                }
                if (run_begin) i_next = json.nodes[run_last].next;
            }
            if (tc || run_begin) {
                last_call_id_count = 0;
                if (tc) {
                    const prompt_json_node_t *array = tc;
                    const prompt_json_node_t *single = NULL;
                    if (tc->kind == PJ_OBJECT) {
                        bool v_tc = true;
                        const prompt_json_node_t *inner = prompt_json_field(&json, tc, "tool_calls", &v_tc);
                        if (inner && inner->kind == PJ_ARRAY) array = inner;
                        else if (prompt_json_field(&json, tc, "name", &v_tc)) single = tc;
                    }
                    if (array && array->kind == PJ_ARRAY) {
                        for (size_t k = (size_t)(array - json.nodes) + 1; k < array->next; k = json.nodes[k].next) {
                            bool v_tc = true;
                            const prompt_json_node_t *entry = &json.nodes[k];
                            const prompt_json_node_t *id_node = prompt_json_field(&json, entry, "id", &v_tc);
                            if (!id_node) {
                                const prompt_json_node_t *fn = prompt_json_field(&json, entry, "function", &v_tc);
                                if (fn && fn->kind == PJ_OBJECT) id_node = prompt_json_field(&json, fn, "id", &v_tc);
                            }
                            if (id_node && id_node->kind == PJ_STRING && id_node->text && last_call_id_count < 64) {
                                strncpy(last_call_ids[last_call_id_count], id_node->text, 127);
                                last_call_ids[last_call_id_count][127] = 0;
                                last_call_id_count++;
                            }
                        }
                    } else if (single) {
                        bool v_tc = true;
                        const prompt_json_node_t *id_node = prompt_json_field(&json, single, "id", &v_tc);
                        if (id_node && id_node->kind == PJ_STRING && id_node->text && last_call_id_count < 64) {
                            strncpy(last_call_ids[last_call_id_count], id_node->text, 127);
                            last_call_ids[last_call_id_count][127] = 0;
                            last_call_id_count++;
                        }
                    }
                }
                if (run_begin) {
                    size_t stop = json.nodes[run_last].next;
                    for (size_t k = run_begin; k < stop; k = json.nodes[k].next) {
                        bool v_tc = true;
                        const prompt_json_node_t *id_node = prompt_json_field(&json, &json.nodes[k], "call_id", &v_tc);
                        if (!id_node) id_node = prompt_json_field(&json, &json.nodes[k], "id", &v_tc);
                        if (id_node && id_node->kind == PJ_STRING && id_node->text && last_call_id_count < 64) {
                            strncpy(last_call_ids[last_call_id_count], id_node->text, 127);
                            last_call_ids[last_call_id_count][127] = 0;
                            last_call_id_count++;
                        }
                    }
                }
                suffix = malloc(max_prompt_len);
                if (!suffix) { *why = "out of memory rendering tool calls"; goto done; }
                prompt_buffer_t sb = {suffix, max_prompt_len, 0, false};
                if (tc && !prompt_render_tool_call_block(&sb, &json, tc)) {
                    *why = "malformed tool_calls/function_call in the message history";
                    goto done;
                }
                if (run_begin && !prompt_render_tool_call_items(&sb, &json, run_begin, run_last)) {
                    *why = "malformed function_call item in the message history";
                    goto done;
                }
            } else if (!strcmp(role_text, "assistant")) {
                last_call_id_count = 0;
            }
            if (!valid) { *why = "malformed content in the message history"; goto done; }
            /* function_call 项没有 content（name/arguments 在这一层）⇒ 正文按空处理 */
            if (content_optional && !content) {
                decoded.data[0] = 0;
            } else if (!prompt_content(&json, content, &decoded, img)) {
                *why = "unsupported content block in the message history";
                goto done;
            }
            const char *reasoning_text = NULL;
            if (!strcmp(role_text, "assistant")) {
                const prompt_json_node_t *rnode = prompt_json_field(&json, item, "reasoning_content", &valid);
                if (!rnode) rnode = prompt_json_field(&json, item, "reasoning", &valid);
                if (!rnode) rnode = prompt_json_field(&json, item, "thought", &valid);
                if (rnode && rnode->kind == PJ_STRING) reasoning_text = rnode->text;
            }
            if (!prompt_render_message(&out, &state, role_text, decoded.data, suffix, reasoning_text)) {
                *why = "a message could not be rendered (or the prompt buffer is full)";
                goto done;
            }
            free(suffix); suffix = NULL;
            if (!strcmp(role_text, "system") && has_tools && !tools_done) {
                if (!prompt_append(&out, "\n\n") || !prompt_render_tools(&out, &json, tool_defs)) {
                    *why = "invalid or oversized tool definitions (tools/functions)";
                    goto done;
                }
                tools_done = true;
            }
        }
    } else { *why = "messages/input must be a string or an array"; goto done; }
    if (state.pending) {
        if (req_thinking) {
            if (!prompt_append(&out, "<｜Assistant｜><think>")) goto done;
        } else {
            if (!prompt_append(&out, "<｜Assistant｜></think>")) goto done;
        }
    }
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
bool parse_chat_prompt_ex(const char *body, char *out_prompt, size_t max_prompt_len,
                          const char **out_reason) {
    return parse_chat_prompt_impl(body, out_prompt, max_prompt_len, out_reason, false, NULL);
}

/* 带视觉槽的版本：服务端在启用视觉时走这两个。 */
bool parse_chat_prompt_vis(const char *body, char *out_prompt, size_t max_prompt_len,
                           const char **out_reason, myds_vjob_sink_t *img) {
    return parse_chat_prompt_impl(body, out_prompt, max_prompt_len, out_reason, false, img);
}

bool parse_chat_prompt_suffix_vis(const char *body, char *out_prompt, size_t max_prompt_len,
                                  const char **out_reason, myds_vjob_sink_t *img) {
    return parse_chat_prompt_impl(body, out_prompt, max_prompt_len, out_reason, true, img);
}

/* previous_response_id：只渲染新增部分（无 BOS / instructions / tools），
 * 前缀由引擎的缓存序列提供。 */
bool parse_chat_prompt_suffix_ex(const char *body, char *out_prompt, size_t max_prompt_len,
                                 const char **out_reason) {
    return parse_chat_prompt_impl(body, out_prompt, max_prompt_len, out_reason, true, NULL);
}

bool parse_chat_prompt(const char *body, char *out_prompt, size_t max_prompt_len) {
    return parse_chat_prompt_ex(body, out_prompt, max_prompt_len, NULL);
}

/* OpenAI 的 tool_calls 数组：arguments 是【JSON 字符串】。 */
char *openai_tool_calls_json(const myds_tool_call_t *calls, size_t count) {
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

/* Responses API 的 output 数组：思考项（若有）、正文项、工具调用项。 */
char *responses_output_items_json_ex(bool thinking, const char *reasoning, size_t reasoning_len,
                                     const char *content, size_t content_len,
                                     const myds_tool_call_t *calls, size_t count) {
    dsml_buf_t b = {0};
    if (!DSML_PUTS(&b, "[")) return NULL;
    bool first = true;
    if (thinking || reasoning_len > 0) {
        if (!DSML_PUTS(&b, "{\"type\":\"reasoning\",\"id\":\"rs_myds_0\",\"status\":\"completed\","
                           "\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"") ||
            !dsml_put_escaped(&b, reasoning ? reasoning : "", reasoning ? reasoning_len : 0) ||
            !DSML_PUTS(&b, "\"}]}")) { free(b.data); return NULL; }
        first = false;
    }
    if (content_len || (!thinking && count == 0)) {
        if ((!first && !DSML_PUTS(&b, ",")) ||
            !DSML_PUTS(&b, "{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\","
                           "\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"") ||
            !dsml_put_escaped(&b, content ? content : "", content ? content_len : 0) ||
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

char *responses_output_items_json(const char *content, size_t content_len,
                                  const myds_tool_call_t *calls, size_t count) {
    return responses_output_items_json_ex(false, NULL, 0, content, content_len, calls, count);
}
