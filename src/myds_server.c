#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_internal.h"
#include "myds_dspark.h"
#include "myds_vjob.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include <stdarg.h>
#include <limits.h>
#include <pthread.h>
#include <poll.h>
#include <stdatomic.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

static volatile sig_atomic_t g_running = 1;
static int g_server_fd = -1;
static pthread_mutex_t g_infer_mutex = PTHREAD_MUTEX_INITIALIZER;
void server_vjob_gpu_lock(void);
void server_vjob_gpu_unlock(void);
static pthread_once_t  g_vjob_lock_once = PTHREAD_ONCE_INIT;
static void init_vjob_gpu_lock(void) {
    myds_vjob_set_gpu_lock(server_vjob_gpu_lock, server_vjob_gpu_unlock);
}

/* ── 推理请求 FIFO 队列与背压保护 ──
 * 单槽推理引擎同一时刻只能运行一个推理请求。
 * 为避免多并发请求无序竞争互斥锁或长期阻塞占用连接与线程池，实现显式 FIFO 队列与背压：
 * - 超过最大排队深度 (MYDS_MAX_QUEUE_DEPTH, 默认 32) 时立即返回 HTTP 503 (Server Busy)。
 * - 排队等待超时 (MYDS_QUEUE_TIMEOUT_MS, 默认 60000ms) 时返回 HTTP 503 (Queue Timeout)。
 * - 退出时按先来先到顺序依次唤醒，保证无饥饿。 */
typedef struct infer_waiter {
    pthread_cond_t cond;
    bool granted;
    bool cancelled;
    struct infer_waiter *next;
} infer_waiter_t;

static pthread_mutex_t g_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static infer_waiter_t *g_queue_head = NULL;
static infer_waiter_t *g_queue_tail = NULL;
static uint32_t g_queue_count = 0;
static bool g_infer_busy = false;

static int server_acquire_infer_slot(int timeout_ms, int max_depth) {
    pthread_mutex_lock(&g_queue_lock);
    if (!g_infer_busy && g_queue_count == 0) {
        g_infer_busy = true;
        pthread_mutex_unlock(&g_queue_lock);
        return 0;
    }
    if (max_depth > 0 && (int)g_queue_count >= max_depth) {
        pthread_mutex_unlock(&g_queue_lock);
        return -1; /* queue full */
    }

    infer_waiter_t w;
    pthread_cond_init(&w.cond, NULL);
    w.granted = false;
    w.cancelled = false;
    w.next = NULL;

    if (g_queue_tail) {
        g_queue_tail->next = &w;
    } else {
        g_queue_head = &w;
    }
    g_queue_tail = &w;
    g_queue_count++;

    struct timespec ts;
    if (timeout_ms > 0) {
        struct timeval now;
        gettimeofday(&now, NULL);
        ts.tv_sec = now.tv_sec + timeout_ms / 1000;
        ts.tv_nsec = (now.tv_usec + (timeout_ms % 1000) * 1000) * 1000;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }
    }

    int rc = 0;
    while (!w.granted && !w.cancelled && g_running) {
        if (timeout_ms > 0) {
            rc = pthread_cond_timedwait(&w.cond, &g_queue_lock, &ts);
            if (rc == ETIMEDOUT && !w.granted) {
                w.cancelled = true;
                break;
            }
        } else {
            pthread_cond_wait(&w.cond, &g_queue_lock);
        }
    }

    if (w.cancelled || (!w.granted && !g_running)) {
        infer_waiter_t **curr = &g_queue_head;
        while (*curr) {
            if (*curr == &w) {
                *curr = w.next;
                if (g_queue_tail == &w) {
                    infer_waiter_t *prev = NULL, *p = g_queue_head;
                    while (p) { prev = p; p = p->next; }
                    g_queue_tail = prev;
                }
                g_queue_count--;
                break;
            }
            curr = &(*curr)->next;
        }
        pthread_cond_destroy(&w.cond);
        pthread_mutex_unlock(&g_queue_lock);
        return -2; /* timed out or cancelled */
    }

    pthread_cond_destroy(&w.cond);
    pthread_mutex_unlock(&g_queue_lock);
    return 0;
}

static void server_release_infer_slot(void) {
    pthread_mutex_lock(&g_queue_lock);
    infer_waiter_t *w = g_queue_head;
    while (w && w->cancelled) {
        g_queue_head = w->next;
        if (!g_queue_head) g_queue_tail = NULL;
        g_queue_count--;
        w = g_queue_head;
    }
    if (w) {
        g_queue_head = w->next;
        if (!g_queue_head) g_queue_tail = NULL;
        g_queue_count--;
        w->granted = true;
        pthread_cond_signal(&w->cond);
    } else {
        g_infer_busy = false;
    }
    pthread_mutex_unlock(&g_queue_lock);
}

/* ── 视觉 GPU 锁：统一获取推理名额，排队覆盖视觉编码等待 ── */
static __thread int t_infer_slot_depth = 0;
static __thread int t_vjob_slot_error = 0;

void server_vjob_gpu_lock(void) {
    if (t_infer_slot_depth > 0) {
        t_infer_slot_depth++;
        return;
    }
    int q_timeout_ms = 60000;
    const char *to_env = getenv("MYDS_QUEUE_TIMEOUT_MS");
    if (to_env && to_env[0]) {
        int v = atoi(to_env);
        if (v >= 0) q_timeout_ms = v;
    }
    int max_q_depth = 32;
    const char *qd_env = getenv("MYDS_MAX_QUEUE_DEPTH");
    if (qd_env && qd_env[0]) {
        int v = atoi(qd_env);
        if (v > 0) max_q_depth = v;
    }

    int rc = server_acquire_infer_slot(q_timeout_ms, max_q_depth);
    if (rc != 0) {
        t_vjob_slot_error = rc;
        return;
    }
    pthread_mutex_lock(&g_infer_mutex);
    t_infer_slot_depth = 1;
    t_vjob_slot_error = 0;
}

void server_vjob_gpu_unlock(void) {
    if (t_infer_slot_depth <= 0) return;
    t_infer_slot_depth--;
    if (t_infer_slot_depth == 0) {
        pthread_mutex_unlock(&g_infer_mutex);
        server_release_infer_slot();
    }
}

int server_vjob_gpu_last_error(void) {
    return t_vjob_slot_error;
}

/* ── previous_response_id：响应 id 注册表 ──
 * Responses API 允许客户端只发【新增内容】+ 上一次响应的 id，让服务端接着那次的状态跑。
 *
 * myds 是单槽：只有【最近一次】响应的状态还活着 —— 引擎的 pcache 序列正好停在它的末尾，
 * 连快照都不用恢复。所以这里只认「最近一次」，更早的 id 一律明确报错。
 * 这与 ds4 的取舍一致（它直接拒绝该字段），但比【静默忽略】安全得多：静默忽略会让客户端
 * 以为服务端持有历史，实际只渲染了新增的那几条 ⇒ 模型拿不到上下文却照常作答，输出静默错误。 */
#define RESP_ID_SLOTS 4u
static char            g_resp_ids[RESP_ID_SLOTS][64];
static unsigned        g_resp_id_ring = 0;         /* 下一个写入位置 */
static unsigned long   g_resp_id_seq = 0;
static pthread_mutex_t g_resp_id_mutex = PTHREAD_MUTEX_INITIALIZER;

void resp_id_new(char *out, size_t cap) {
    unsigned long n;
    pthread_mutex_lock(&g_resp_id_mutex);
    n = ++g_resp_id_seq;
    pthread_mutex_unlock(&g_resp_id_mutex);
    snprintf(out, cap, "resp_myds_%lu_%04x", n, (unsigned)(rand() & 0xffffu));
}
/* 一次响应完成后登记：它成为唯一「活」的 id。 */
void resp_id_register(const char *id) {
    if (!id || !id[0]) return;
    pthread_mutex_lock(&g_resp_id_mutex);
    snprintf(g_resp_ids[g_resp_id_ring], sizeof(g_resp_ids[0]), "%s", id);
    g_resp_id_ring = (g_resp_id_ring + 1u) % RESP_ID_SLOTS;
    pthread_mutex_unlock(&g_resp_id_mutex);
}
/* 最新的那个 = 唯一可续跑的 */
bool resp_id_is_live(const char *id) {
    if (!id || !id[0]) return false;
    bool live;
    pthread_mutex_lock(&g_resp_id_mutex);
    unsigned prev = (g_resp_id_ring + RESP_ID_SLOTS - 1u) % RESP_ID_SLOTS;
    live = (g_resp_ids[prev][0] != '\0' && strcmp(g_resp_ids[prev], id) == 0);
    pthread_mutex_unlock(&g_resp_id_mutex);
    return live;
}
/* 是否见过（区分「过期」与「未知」，便于客户端判断） */
bool resp_id_seen(const char *id) {
    if (!id || !id[0]) return false;
    bool seen = false;
    pthread_mutex_lock(&g_resp_id_mutex);
    for (unsigned i = 0; i < RESP_ID_SLOTS; ++i)
        if (g_resp_ids[i][0] && strcmp(g_resp_ids[i], id) == 0) { seen = true; break; }
    pthread_mutex_unlock(&g_resp_id_mutex);
    return seen;
}

/* 当推理失败、被取消或上下文被清空时，旧响应 ID 必须立即失效，防止后续请求产生脱节续写 */
void resp_id_invalidate_all(void) {
    pthread_mutex_lock(&g_resp_id_mutex);
    for (unsigned i = 0; i < RESP_ID_SLOTS; ++i) {
        g_resp_ids[i][0] = '\0';
    }
    g_resp_id_ring = 0;
    pthread_mutex_unlock(&g_resp_id_mutex);
}

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
    for (const char *p = text; *p; ++p) {
        const char *q = p;
        if (*q == '&') { ++q; while (!strncmp(q, "amp;", 4)) q += 4; }
        bool escape = !strncmp(p, "</tool_result>", 14) || (*p == '&' && !strncmp(q, "lt;/tool_result>", 16));
        char byte[2] = {*p, 0};
        if (!prompt_append(out, escape ? (*p == '<' ? "&lt;" : "&amp;") : byte)) return false;
    }
    return prompt_append(out, "</tool_result>");
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
static bool g_server_default_thinking = false;

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

    char effort[32] = "";
    if (json_extract_string_field(body, "reasoning_effort", effort, sizeof(effort))) {
        if (!strcmp(effort, "none") || !strcmp(effort, "off")) return false;
        if (!strcmp(effort, "low") || !strcmp(effort, "medium") || !strcmp(effort, "high") || !strcmp(effort, "max")) return true;
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

    char effort[32] = "";
    if (json_extract_string_field(body, "reasoning_effort", effort, sizeof(effort))) {
        if (!strcmp(effort, "low")) return 50;
        if (!strcmp(effort, "medium")) return 62;
        if (!strcmp(effort, "high")) return 75;
        if (!strcmp(effort, "max")) return 100;
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
/* 小整数格式化（index 之类）——别用 %s 拼裸数字 */
static bool dsml_putf(dsml_buf_t *b, const char *fmt, ...) {
    char tmp[64];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    return n > 0 && (size_t)n < sizeof(tmp) && dsml_put(b, tmp, (size_t)n);
}
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

void myds_free_tool_calls(myds_tool_call_t *calls, size_t count) {
    for (size_t i = 0; i < count; ++i) { free(calls[i].id); free(calls[i].name); free(calls[i].arguments); }
    free(calls);
}

static inline double myds_server_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct sse_packet {
    struct sse_packet *next;
    size_t length;
    char data[];
} sse_packet_t;

struct myds_sse_writer {
    int fd, original_fd_flags;
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t ready;
    sse_packet_t *head, *tail;
    size_t pending_bytes, byte_limit;
    bool closing, aborted, failed;
    double drain_deadline;
    unsigned drain_timeout_ms;
};

static atomic_uint sse_writer_failure_point = 0;

void myds_server_sse_writer_inject_failure(unsigned point) {
    atomic_store(&sse_writer_failure_point, point);
}

static bool sse_writer_injected_failure(unsigned point) {
    return atomic_compare_exchange_strong(&sse_writer_failure_point, &point, 0);
}

unsigned myds_server_sse_drain_timeout_ms(void) {
    const char *value = getenv("MYDS_SSE_DRAIN_TIMEOUT_MS");
    if (value && *value) {
        char *end;
        unsigned long timeout = strtoul(value, &end, 10);
        if (!*end && timeout >= 1 && timeout <= 30000) return (unsigned)timeout;
    }
    return 1000;
}

static bool sse_writer_stopped(myds_sse_writer_t *w) {
    pthread_mutex_lock(&w->lock);
    bool stopped = w->aborted;
    if (w->closing && myds_server_now_s() >= w->drain_deadline) {
        w->failed = w->aborted = true;
        stopped = true;
    }
    pthread_mutex_unlock(&w->lock);
    return stopped;
}

static int sse_writer_poll_wait_ms(myds_sse_writer_t *w) {
    int wait_ms = 20;
    pthread_mutex_lock(&w->lock);
    if (w->closing) {
        const double remaining_ms = (w->drain_deadline - myds_server_now_s()) * 1000.0;
        if (remaining_ms <= 0.0) wait_ms = 0;
        else if (remaining_ms < wait_ms) wait_ms = (int)remaining_ms + 1;
    }
    pthread_mutex_unlock(&w->lock);
    return wait_ms;
}

static bool sse_writer_send_packet(myds_sse_writer_t *w, const sse_packet_t *packet) {
    size_t sent = 0;
    while (sent < packet->length) {
        if (sse_writer_stopped(w)) return false;
        int flags = MSG_DONTWAIT;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        ssize_t n = send(w->fd, packet->data + sent, packet->length - sent, flags);
        if (n > 0) { sent += (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd writable = {.fd = w->fd, .events = POLLOUT};
            const int rc = poll(&writable, 1, sse_writer_poll_wait_ms(w));
            if (rc < 0 && errno == EINTR) continue;
            if (rc < 0 || (rc > 0 && (writable.revents & (POLLERR | POLLHUP | POLLNVAL))))
                return false;
            continue;
        }
        return false;
    }
    return true;
}

static void *sse_writer_thread(void *opaque) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    myds_sse_writer_t *w = opaque;
    for (;;) {
        pthread_mutex_lock(&w->lock);
        while (!w->head && !w->closing && !w->aborted) pthread_cond_wait(&w->ready, &w->lock);
        if (w->aborted || (!w->head && w->closing)) {
            pthread_mutex_unlock(&w->lock);
            break;
        }
        sse_packet_t *packet = w->head;
        w->head = packet->next;
        if (!w->head) w->tail = NULL;
        pthread_mutex_unlock(&w->lock);
        /* The in-flight packet remains charged to the budget until fully sent. */
        const bool ok = sse_writer_send_packet(w, packet);
        pthread_mutex_lock(&w->lock);
        w->pending_bytes -= sizeof(*packet) + packet->length;
        if (!ok) w->failed = w->aborted = true;
        pthread_mutex_unlock(&w->lock);
        free(packet);
        if (!ok) { shutdown(w->fd, SHUT_RDWR); break; }
    }
    return NULL;
}

bool myds_server_sse_writer_start(myds_sse_stream_ctx_t *stream, size_t byte_limit,
                                  unsigned drain_timeout_ms) {
    if (!stream || stream->writer || stream->client_fd < 0 ||
        byte_limit <= sizeof(sse_packet_t) || !drain_timeout_ms || drain_timeout_ms > 30000) {
        errno = EINVAL; return false;
    }
    myds_sse_writer_t *w = calloc(1, sizeof(*w));
    if (!w) return false;
    w->fd = stream->client_fd;
    w->byte_limit = byte_limit;
    w->drain_timeout_ms = drain_timeout_ms;
    /* Set descriptor nonblocking explicitly: per-send MSG_DONTWAIT alone
     * does not bound large writes reliably on all supported socket types. */
    w->original_fd_flags = fcntl(w->fd, F_GETFL);
    if (w->original_fd_flags < 0 || fcntl(w->fd, F_SETFL, w->original_fd_flags | O_NONBLOCK) < 0) {
        free(w); return false;
    }
#ifdef SO_NOSIGPIPE
    int no_sigpipe = 1;
    if (setsockopt(w->fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe)) != 0) {
        const int saved_errno = errno;
        (void)fcntl(w->fd, F_SETFL, w->original_fd_flags);
        free(w); errno = saved_errno; return false;
    }
#endif
    int error = pthread_mutex_init(&w->lock, NULL);
    if (error) {
        (void)fcntl(w->fd, F_SETFL, w->original_fd_flags);
        free(w); errno = error; return false;
    }
    error = pthread_cond_init(&w->ready, NULL);
    if (error) {
        pthread_mutex_destroy(&w->lock);
        (void)fcntl(w->fd, F_SETFL, w->original_fd_flags);
        free(w); errno = error; return false;
    }
    error = sse_writer_injected_failure(1) ? EAGAIN : pthread_create(&w->thread, NULL, sse_writer_thread, w);
    if (error) {
        pthread_cond_destroy(&w->ready); pthread_mutex_destroy(&w->lock);
        (void)fcntl(w->fd, F_SETFL, w->original_fd_flags);
        free(w); errno = error; return false;
    }
    stream->writer = w;
    return true;
}

bool myds_server_sse_send(myds_sse_stream_ctx_t *stream, const char *data, size_t length) {
    if (!stream || (length && !data)) { errno = EINVAL; return false; }
    if (!stream->writer) return myds_server_send_all(stream->client_fd, data, length);
    myds_sse_writer_t *w = stream->writer;
    pthread_mutex_lock(&w->lock);
    bool ok = !w->aborted && !w->closing;
    if (ok && length) {
        if (length > SIZE_MAX - sizeof(sse_packet_t) ||
            sizeof(sse_packet_t) + length > w->byte_limit - w->pending_bytes) {
            w->aborted = true;
            ok = false;
        } else {
            sse_packet_t *packet = NULL;
            if (sse_writer_injected_failure(2)) errno = ENOMEM;
            else packet = malloc(sizeof(*packet) + length);
            if (!packet) { w->failed = w->aborted = true; stream->failed = true; ok = false; }
            else {
                packet->next = NULL; packet->length = length;
                memcpy(packet->data, data, length);
                if (w->tail) w->tail->next = packet; else w->head = packet;
                w->tail = packet;
                w->pending_bytes += sizeof(*packet) + length;
                pthread_cond_signal(&w->ready);
            }
        }
    }
    if (!ok) {
        stream->cancel_flag = 1;
        pthread_cond_signal(&w->ready);
    }
    pthread_mutex_unlock(&w->lock);
    if (!ok) shutdown(w->fd, SHUT_RDWR);
    return ok;
}

bool myds_server_sse_writer_join(myds_sse_stream_ctx_t *stream, bool aborting) {
    if (!stream || !stream->writer) return stream && !stream->cancel_flag;
    myds_sse_writer_t *w = stream->writer;
    pthread_mutex_lock(&w->lock);
    w->closing = true;
    w->drain_deadline = myds_server_now_s() + (double)w->drain_timeout_ms / 1000.0;
    if (aborting) w->aborted = true;
    pthread_cond_signal(&w->ready);
    pthread_mutex_unlock(&w->lock);
    if (aborting) shutdown(w->fd, SHUT_RDWR);
    pthread_join(w->thread, NULL);
    const bool ok = !w->failed && !w->aborted;
    (void)fcntl(w->fd, F_SETFL, w->original_fd_flags);
    while (w->head) { sse_packet_t *next = w->head->next; free(w->head); w->head = next; }
    pthread_cond_destroy(&w->ready); pthread_mutex_destroy(&w->lock);
    free(w); stream->writer = NULL;
    if (!ok) stream->cancel_flag = 1;
    return ok;
}

static bool sse_transport_failed(myds_sse_stream_ctx_t *stream) {
    if (!stream->writer) return false;
    pthread_mutex_lock(&stream->writer->lock);
    const bool failed = stream->writer->aborted;
    pthread_mutex_unlock(&stream->writer->lock);
    if (failed) stream->cancel_flag = 1;
    return failed;
}

static bool client_disconnected(int fd) {
    if (fd < 0) return false;
    char byte;
    ssize_t r;
    do { r = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT); } while (r < 0 && errno == EINTR);
    return r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
}

static bool sse_emit_reasoning(myds_sse_stream_ctx_t *sctx, const char *piece, size_t length, bool final) {
    if (sctx->cancel_flag || sse_transport_failed(sctx)) return false;
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
    if (escaped_length == 0) {
        free(escaped);
        return true;
    }
    char *chunk = NULL;
    if (sctx->responses_api) {
        chunk = format_alloc(
            "event: response.reasoning_text.delta\n"
            "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"%s\"}\n\n",
            escaped);
    } else {
        chunk = format_alloc(
            "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"%s\"},\"finish_reason\":null}]}\n\n",
            escaped);
    }
    free(escaped);
    if (!chunk) goto failed;
    bool sent = myds_server_sse_send(sctx, chunk, strlen(chunk));
    free(chunk);
    if (!sent) sctx->cancel_flag = 1;
    return sent;
failed:
    sctx->failed = true;
    sctx->cancel_flag = 1;
    return false;
}

static bool sse_emit_responses_close_reasoning(myds_sse_stream_ctx_t *sctx,
                                                const char *text, size_t text_len) {
    if (!sctx || sctx->cancel_flag || sctx->failed) return false;
    char *text_json = json_escape_alloc(text ? text : "", text ? text_len : 0);
    if (!text_json) return false;
    char *ev = format_alloc(
        "event: response.reasoning_text.done\n"
        "data: {\"type\":\"response.reasoning_text.done\",\"output_index\":0,\"content_index\":0,\"text\":\"%s\"}\n\n"
        "event: response.content_part.done\n"
        "data: {\"type\":\"response.content_part.done\",\"output_index\":0,\"content_index\":0,"
        "\"part\":{\"type\":\"reasoning_text\",\"text\":\"%s\"}}\n\n"
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,"
        "\"item\":{\"type\":\"reasoning\",\"id\":\"rs_myds_0\",\"status\":\"completed\",\"summary\":[],"
        "\"content\":[{\"type\":\"reasoning_text\",\"text\":\"%s\"}]}}\n\n",
        text_json, text_json, text_json);
    free(text_json);
    if (!ev) return false;
    const bool ok = myds_server_sse_send(sctx, ev, strlen(ev));
    free(ev);
    return ok;
}

static bool sse_emit_responses_open_message(myds_sse_stream_ctx_t *sctx) {
    if (!sctx || sctx->cancel_flag || sctx->failed) return false;
    const char *ev =
        "event: response.output_item.added\n"
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n"
        "event: response.content_part.added\n"
        "data: {\"type\":\"response.content_part.added\",\"output_index\":1,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n";
    return myds_server_sse_send(sctx, ev, strlen(ev));
}

static bool sse_emit_content(myds_sse_stream_ctx_t *sctx, const char *piece, size_t length, bool final) {
    if (sctx->cancel_flag || sse_transport_failed(sctx)) return false;
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
          "data: {\"type\":\"response.output_text.delta\",\"output_index\":%d,\"content_index\":0,\"delta\":\"%s\"}\n\n"
        : "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n";
    char *chunk = sctx->responses_api ? format_alloc(format, sctx->msg_output_index, escaped) : format_alloc(format, escaped);
    free(escaped);
    if (!chunk) goto failed;
    bool sent = myds_server_sse_send(sctx, chunk, strlen(chunk));
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

    /* ── 阶段 1: 思考流式输出 (thinking mode) ── */
    if (sctx->thinking && sctx->think_state == 1) {
        static const char *think_close = "</think>";
        const size_t close_len = 8;
        const char *base = sctx->acc + sctx->emitted;
        size_t avail = sctx->acc_len - sctx->emitted;
        const char *tag = avail ? strstr(base, think_close) : NULL;
        if (tag) {
            size_t upto = (size_t)(tag - base);
            bool ok = upto ? sse_emit_reasoning(sctx, base, upto, false) : true;
            ok = ok && sse_emit_reasoning(sctx, NULL, 0, true); /* flush utf8 */
            sctx->emitted += upto + close_len;
            while (sctx->emitted < sctx->acc_len && (sctx->acc[sctx->emitted] == '\r' || sctx->acc[sctx->emitted] == '\n')) sctx->emitted++;
            if (sctx->responses_api) {
                size_t total_thought_len = (size_t)(tag - sctx->acc);
                const char *thought_start = sctx->acc;
                if (strncmp(thought_start, "<think>", 7) == 0) {
                    thought_start += 7;
                    if (total_thought_len >= 7) total_thought_len -= 7;
                    else total_thought_len = 0;
                }
                ok = ok && sse_emit_responses_close_reasoning(sctx, thought_start, total_thought_len);
                ok = ok && sse_emit_responses_open_message(sctx);
            }
            sctx->think_state = 2; /* 进入正文阶段 */
            if (!ok) return false;
        } else {
            size_t maxk = close_len - 1 < avail ? close_len - 1 : avail;
            size_t keep = 0;
            for (size_t k = maxk; k > 0; --k) {
                if (!memcmp(base + avail - k, think_close, k)) { keep = k; break; }
            }
            size_t upto = avail - keep;
            bool ok = upto ? sse_emit_reasoning(sctx, base, upto, false) : true;
            sctx->emitted += upto;
            if (!ok) return false;
            if (final) {
                if (keep) {
                    ok = sse_emit_reasoning(sctx, base + upto, keep, false);
                    sctx->emitted += keep;
                }
                ok = ok && sse_emit_reasoning(sctx, NULL, 0, true);
                if (sctx->responses_api) {
                    const char *thought_start = sctx->acc;
                    size_t total_thought_len = sctx->acc_len;
                    if (strncmp(thought_start, "<think>", 7) == 0) {
                        thought_start += 7;
                        if (total_thought_len >= 7) total_thought_len -= 7;
                        else total_thought_len = 0;
                    }
                    ok = ok && sse_emit_responses_close_reasoning(sctx, thought_start, total_thought_len);
                }
            }
            return ok;
        }
    }

    /* ── 阶段 2: 正文与工具调用流式输出 (content & DSML tool block) ── */
    static const char *open_tag = "<" MYDS_DSML_S " calls>";
    const size_t tag_len = strlen(open_tag);
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

/* ── Responses：把流里那条【正文 message 项】（output_index 0）按契约收尾 ──
 *
 * 为什么必须有：本端点一开流就【无条件】发过
 *   response.output_item.added(index 0, item=message, status=in_progress)
 * 而工具轮次过去直接从 index 0 开始发 function_call ⇒ **同一个 output_index 被复用**。
 * 客户端据此判定非法流（OpenClaw 的报错原文：“Responses stream reused active output index 0”）⇒
 * 整轮失败。实测 2026-10-01 与 10-02 两次因此整条链路不可用。
 *
 * 修法：发 function_call（index 从 1 起，见 sse_emit_responses_function_calls）【之前】，
 * 先把 index 0 这条 message 项按契约关掉：output_text.done → content_part.done → output_item.done。
 * 文本取【工具块之前的正文】（调用方传 scontent_len），与本流已发的 delta 一致。 */
static bool sse_emit_responses_close_message(myds_sse_stream_ctx_t *sctx,
                                             const char *text, size_t text_len) {
    if (!sctx || sctx->cancel_flag || sctx->failed) return false;
    char *text_json = json_escape_alloc(text ? text : "", text ? text_len : 0);
    if (!text_json) return false;
    const int out_idx = sctx->msg_output_index;
    char *ev = format_alloc(
        "event: response.output_text.done\n"
        "data: {\"type\":\"response.output_text.done\",\"output_index\":%d,\"content_index\":0,\"text\":\"%s\"}\n\n"
        "event: response.content_part.done\n"
        "data: {\"type\":\"response.content_part.done\",\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":\"%s\"}}\n\n"
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"output_index\":%d,"
        "\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"completed\","
        "\"content\":[{\"type\":\"output_text\",\"text\":\"%s\"}]}}\n\n",
        out_idx, text_json, out_idx, text_json, out_idx, text_json);
    free(text_json);
    if (!ev) return false;
    const bool ok = myds_server_sse_send(sctx, ev, strlen(ev));
    free(ev);
    return ok;
}

/* chat/completions：工具调用 delta + finish_reason=tool_calls + [DONE] */
bool sse_emit_chat_tool_calls(myds_sse_stream_ctx_t *sctx, const myds_tool_call_t *calls, size_t count) {
    char *tc = chat_tool_calls_delta_json(calls, count);
    if (!tc) return false;
    char *chunk = format_alloc(
        "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":%s},\"finish_reason\":null}]}\n\n", tc);
    if (!chunk) { free(tc); return false; }
    bool sent = myds_server_sse_send(sctx, chunk, strlen(chunk));
    free(chunk); free(tc);
    if (!sent) return false;
    const char *finish_chunk =
        "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
        "\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n";
    if (!myds_server_sse_send(sctx, finish_chunk, strlen(finish_chunk))) return false;
    const char *done = "data: [DONE]\n\n";
    return myds_server_sse_send(sctx, done, strlen(done));
}

bool sse_emit_responses_function_calls(myds_sse_stream_ctx_t *sctx,
                                       const myds_tool_call_t *calls, size_t count) {
    const size_t base_idx = sctx ? (size_t)sctx->msg_output_index + 1 : 1;
    for (size_t i = 0; i < count; ++i) {
        char ids[96];
        snprintf(ids, sizeof(ids), "fc_%zu", i + 1);
        dsml_buf_t ev = {0};
        bool ok = true;
        /* added */
        ok = ok && DSML_PUTS(&ev, "event: response.output_item.added\n"
            "data: {\"type\":\"response.output_item.added\",\"output_index\":")
            && dsml_putf(&ev, "%zu", base_idx + i)
            && DSML_PUTS(&ev, ",\"item\":{\"type\":\"function_call\",\"id\":\"")
            && DSML_PUTS(&ev, ids)
            && DSML_PUTS(&ev, "\",\"call_id\":\"")
            && dsml_put_escaped(&ev, calls[i].id, strlen(calls[i].id))
            && DSML_PUTS(&ev, "\",\"name\":\"")
            && dsml_put_escaped(&ev, calls[i].name, strlen(calls[i].name))
            && DSML_PUTS(&ev, "\",\"arguments\":\"\",\"status\":\"in_progress\"}}\n\n");
        /* arguments delta */
        ok = ok && DSML_PUTS(&ev, "event: response.function_call_arguments.delta\n"
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"")
            && DSML_PUTS(&ev, ids)
            && DSML_PUTS(&ev, "\",\"output_index\":")
            && dsml_putf(&ev, "%zu", base_idx + i)
            && DSML_PUTS(&ev, ",\"delta\":\"")
            && dsml_put_escaped(&ev, calls[i].arguments, strlen(calls[i].arguments))
            && DSML_PUTS(&ev, "\"}\n\n");
        /* done */
        ok = ok && DSML_PUTS(&ev, "event: response.output_item.done\n"
            "data: {\"type\":\"response.output_item.done\",\"output_index\":")
            && dsml_putf(&ev, "%zu", base_idx + i)
            && DSML_PUTS(&ev, ",\"item\":{\"type\":\"function_call\",\"id\":\"")
            && DSML_PUTS(&ev, ids)
            && DSML_PUTS(&ev, "\",\"call_id\":\"")
            && dsml_put_escaped(&ev, calls[i].id, strlen(calls[i].id))
            && DSML_PUTS(&ev, "\",\"name\":\"")
            && dsml_put_escaped(&ev, calls[i].name, strlen(calls[i].name))
            && DSML_PUTS(&ev, "\",\"arguments\":\"")
            && dsml_put_escaped(&ev, calls[i].arguments, strlen(calls[i].arguments))
            && DSML_PUTS(&ev, "\",\"status\":\"completed\"}}\n\n");
        if (ok) ok = myds_server_sse_send(sctx, ev.data, ev.len);
        free(ev.data);
        if (!ok) return false;
    }
    return true;
}

bool sse_emit_responses_finish(myds_sse_stream_ctx_t *sctx,
                               const char *model_alias,
                               const char *resp_id,
                               unsigned prompt_tokens,
                               unsigned output_tokens,
                               const char *text, size_t text_len,
                               const char *output_items_json,
                               int emit_text_events) {
    if (!sctx || sctx->cancel_flag || sctx->failed) return false;
    char *alias_json = json_escape_alloc(model_alias ? model_alias : "myds",
                                        strlen(model_alias ? model_alias : "myds"));
    char *text_json = json_escape_alloc(text ? text : "", text ? text_len : 0);
    if (!alias_json || !text_json) {
        free(alias_json);
        free(text_json);
        return false;
    }
    const int out_idx = sctx ? sctx->msg_output_index : 0;
    const char *items = output_items_json ? output_items_json : "[]";
    char *completed = emit_text_events
        ? format_alloc(
            "event: response.output_text.done\n"
            "data: {\"type\":\"response.output_text.done\",\"output_index\":%d,\"content_index\":0,\"text\":\"%s\"}\n\n"
            "event: response.content_part.done\n"
            "data: {\"type\":\"response.content_part.done\",\"output_index\":%d,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"%s\"}}\n\n"
            "event: response.output_item.done\n"
            "data: {\"type\":\"response.output_item.done\",\"output_index\":%d,\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"%s\"}]}}\n\n"
            "event: response.completed\n"
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"%s\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"%s\",\"output\":%s,\"usage\":{\"input_tokens\":%u,\"output_tokens\":%u,\"total_tokens\":%u}}}\n\n",
            out_idx, text_json, out_idx, text_json, out_idx, text_json, resp_id, alias_json, items,
            prompt_tokens, output_tokens, prompt_tokens + output_tokens)
        : format_alloc(
            "event: response.completed\n"
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"%s\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"%s\",\"output\":%s,\"usage\":{\"input_tokens\":%u,\"output_tokens\":%u,\"total_tokens\":%u}}}\n\n",
            resp_id, alias_json, items,
            prompt_tokens, output_tokens, prompt_tokens + output_tokens);
    free(alias_json);
    free(text_json);
    if (!completed) return false;
    const bool ok = myds_server_sse_send(sctx, completed, strlen(completed));
    free(completed);
    return ok;
}

char *responses_output_message_json_ex(bool thinking, const char *reasoning, size_t reasoning_len,
                                       const char *text, size_t text_len) {
    return responses_output_items_json_ex(thinking, reasoning, reasoning_len, text, text_len, NULL, 0);
}

/* 文本响应的 output 数组：一条 message 项，文本与增量聚合后的完全一致。 */
char *responses_output_message_json(const char *text, size_t text_len) {
    return responses_output_message_json_ex(false, NULL, 0, text, text_len);
}

/* 工具调用响应的 output 数组（给 response.completed 用）。 */
char *responses_function_calls_output_json(const myds_tool_call_t *calls, size_t count) {
    dsml_buf_t out = {0};
    bool ok = DSML_PUTS(&out, "[");
    for (size_t i = 0; ok && i < count; ++i) {
        char ids[32];
        snprintf(ids, sizeof(ids), "fc_%zu", i + 1);
        ok = (i == 0 || DSML_PUTS(&out, ",")) &&
             DSML_PUTS(&out, "{\"type\":\"function_call\",\"id\":\"") &&
             DSML_PUTS(&out, ids) &&
             DSML_PUTS(&out, "\",\"call_id\":\"") &&
             dsml_put_escaped(&out, calls[i].id, strlen(calls[i].id)) &&
             DSML_PUTS(&out, "\",\"name\":\"") &&
             dsml_put_escaped(&out, calls[i].name, strlen(calls[i].name)) &&
             DSML_PUTS(&out, "\",\"arguments\":\"") &&
             dsml_put_escaped(&out, calls[i].arguments, strlen(calls[i].arguments)) &&
             DSML_PUTS(&out, "\",\"status\":\"completed\"}");
    }
    ok = ok && DSML_PUTS(&out, "]");
    if (!ok) {
        free(out.data);
        return NULL;
    }
    return out.data;
}

void myds_server_sse_callback(const char *piece, size_t length, int32_t token_id, void *user_data) {
    (void)token_id;
    myds_sse_stream_ctx_t *sctx = user_data;
    double now = myds_server_now_s();
    if (sctx->token_count == 0) sctx->t_first = now;
    sctx->t_last = now;
    sctx->token_count++;
    if (sse_transport_failed(sctx) || client_disconnected(sctx->client_fd)) {
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

/* ── 把请求里的所有图片块注入引擎 ──
 * 图片块在提示词里的位置：第 i 个块 = 第 i 次出现占位符的地方。
 * 逐块求「该点之前的 token 数」得到起始位置，然后添加区间。
 * 任一块找不到 ⇒ 返回 false（调用方拒绝整个请求）——
 * 宁可报错，也不要把图片向量贴到错的 token 上（那是静默出错）。
 *
 * ★ 为什么收敛成一个函数：之前流式/非流式各写了一份，复制体迟早会分叉。 */
static bool inject_image_embeddings(myds_context_t *ctx, const char *prompt,
                                    myds_vjob_sink_t *sink,
                                    const char *client_ip, int client_port) {
    if (!sink || sink->count == 0) return true;
    if (sink->failed) {
        MYDS_LOG_E("[%s:%d] 本次请求有图片块处理失败，拒绝", client_ip, client_port);
        return false;
    }
    const char *ph = myds_vjob_placeholder();
    const size_t phlen = ph ? strlen(ph) : 0;
    if (!ph || phlen == 0) {
        MYDS_LOG_E("[%s:%d] 视觉占位符缺失，拒绝", client_ip, client_port);
        return false;
    }
    const char *cursor = prompt;
    for (uint32_t i = 0; i < sink->count; i++) {
        const myds_vjob_plan_t *plan = &sink->plans[i];
        const char *at = strstr(cursor, ph);
        if (!at) {
            MYDS_LOG_E("[%s:%d] 第 %u 个图片块的占位文本在提示词里找不到，拒绝",
                       client_ip, client_port, i + 1);
            return false;
        }
        const int64_t off = myds_vjob_token_offset_at(ctx, prompt, at);
        if (off < 0) {
            MYDS_LOG_E("[%s:%d] 第 %u 个图片块的 token 偏移量算不出来，拒绝",
                       client_ip, client_port, i + 1);
            return false;
        }
        myds_context_add_image_embeddings(ctx, (uint32_t)off, plan->token_count,
                                         plan->block, plan->dim,
                                         myds_vjob_image_token_id());
        /* 跳过本块的整段占位文本，保证第 i+1 个块不会匹配到同一个位置
         * （相邻图片块的首个占位符字符完全相同） */
        cursor = at + (size_t)plan->token_count * phlen;
    }
    return true;
}

static void handle_client(int client_fd, myds_context_t *ctx, const char *model_alias, const char *client_ip, int client_port) {
    pthread_once(&g_vjob_lock_once, init_vjob_gpu_lock);
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
        /* 视觉槽：本次请求若带图，构建提示词时填进来，提示词拼好后按位置注入向量。
         * ★ 必须在 parse 之前声明 —— 它是构建过程的输出。 */
        myds_vjob_sink_t img_sink;
        memset(&img_sink, 0, sizeof(img_sink));
        char *prompt = malloc(prompt_cap);
        if (!prompt) {
            MYDS_LOG_E("[%s:%d] Out of memory allocating prompt buffer (%zu bytes)", client_ip, client_port, prompt_cap);
            send_response(client_fd, 500, "Internal Server Error", "application/json", "{\"error\":\"Out of memory\"}");
            free(req_buf);
            close(client_fd);
            return;
        }
        /* previous_response_id：客户端只发新增内容，前缀由引擎的缓存序列提供。
         * 非空且不是「最近一次」⇒ 明确报错，绝不静默截断（见注册表处注释）。 */
        char prev_id[64] = "";
        (void)json_extract_string_field(body, "previous_response_id", prev_id, sizeof(prev_id));
        const bool is_continuation = prev_id[0] != 0;
        char resp_id[64];
        resp_id_new(resp_id, sizeof(resp_id));
        /* 排队与背压保护：在解析与任何 GPU 工作（包括视觉处理）前统一获取推理名额 */
        int q_timeout_ms = 60000;
        const char *to_env = getenv("MYDS_QUEUE_TIMEOUT_MS");
        if (to_env && to_env[0]) {
            int v = atoi(to_env);
            if (v >= 0) q_timeout_ms = v;
        }
        int max_q_depth = 32;
        const char *qd_env = getenv("MYDS_MAX_QUEUE_DEPTH");
        if (qd_env && qd_env[0]) {
            int v = atoi(qd_env);
            if (v > 0) max_q_depth = v;
        }

        int q_res = server_acquire_infer_slot(q_timeout_ms, max_q_depth);
        if (q_res != 0) {
            MYDS_LOG_W("[%s:%d] server reject due to %s (depth=%u, timeout=%dms)",
                       client_ip, client_port, q_res == -1 ? "queue full" : "queue wait timeout",
                       g_queue_count, q_timeout_ms);
            const char *err_json = (q_res == -1)
                ? "{\"error\":{\"message\":\"Server is at capacity (queue full); please retry later\",\"type\":\"server_error\",\"code\":\"server_busy\"}}"
                : "{\"error\":{\"message\":\"Request timed out waiting in inference queue; please retry later\",\"type\":\"server_error\",\"code\":\"queue_timeout\"}}";
            send_response(client_fd, 503, "Service Unavailable", "application/json", err_json);
            myds_vjob_sink_free(&img_sink);
            free(prompt); free(req_buf); close(client_fd); return;
        }

        pthread_mutex_lock(&g_infer_mutex);
        t_infer_slot_depth = 1;
        myds_config_t prev_cfg = myds_context_get_config(ctx);
        uint64_t dspark_prop_before = ctx->dspark ? ctx->dspark->proposed_tokens : 0;
        uint64_t dspark_acc_before  = ctx->dspark ? ctx->dspark->accepted_tokens : 0;
        uint64_t dspark_cyc_before  = ctx->dspark ? ctx->dspark->cycles : 0;

        if (is_continuation && (!resp_id_is_live(prev_id) || myds_context_get_pcache_len(ctx) == 0)) {
            t_infer_slot_depth = 0;
            pthread_mutex_unlock(&g_infer_mutex);
            server_release_infer_slot();
            MYDS_LOG_W("[%s:%d] previous_response_id rejected (%s): %s",
                       client_ip, client_port,
                       resp_id_seen(prev_id) ? "expired" : "unknown", prev_id);
            char err[512];
            snprintf(err, sizeof(err),
                     "{\"error\":{\"message\":\"previous_response_id '%s' is not the most recent response "
                     "(this server keeps one live state); replay the full input instead\","
                     "\"type\":\"invalid_request_error\"}}", prev_id);
            send_response(client_fd, 400, "Bad Request", "application/json", err);
            myds_vjob_sink_free(&img_sink);
            free(prompt); free(req_buf); close(client_fd); return;
        }

        const char *parse_reason = NULL;
        const bool parsed_ok = is_continuation
            ? parse_chat_prompt_suffix_vis(body, prompt, prompt_cap, &parse_reason, &img_sink)
            : parse_chat_prompt_vis(body, prompt, prompt_cap, &parse_reason, &img_sink);
        if (!parsed_ok) {
            MYDS_LOG_W("[%s:%d] rejected (400): %s%s%s [body=%zd bytes, prompt_cap=%zu]",
                       client_ip, client_port, parse_reason,
                       img_sink.why[0] ? " — " : "", img_sink.why, total_read, prompt_cap);
            char err[512];
            snprintf(err, sizeof(err),
                     "{\"error\":{\"message\":\"Invalid or unsupported text prompt: %s%s%s\",\"type\":\"invalid_request_error\"}}",
                     parse_reason ? parse_reason : "unknown",
                     img_sink.why[0] ? " — " : "", img_sink.why);
            send_response(client_fd, 400, "Bad Request", "application/json", err);
            t_infer_slot_depth = 0;
            pthread_mutex_unlock(&g_infer_mutex);
            server_release_infer_slot();
            myds_vjob_sink_free(&img_sink);
            free(prompt);
            free(req_buf);
            close(client_fd);
            return;
        }

        const bool req_has_instructions = json_has_top_field(body, "instructions");
        const bool req_has_tools = json_has_top_field(body, "tools");
        MYDS_LOG_I("[%s:%d] %s %s (protocol=%s, stream=%s, max_tokens=%d, temp=%.2f, top_p=%.2f, top_k=%d, "
                   "prompt_chars=%zu, body=%zd, has_instructions=%d, has_tools=%d)",
                   client_ip, client_port, method, path,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   stream ? "true" : "false",
                   max_tokens, temp, top_p, top_k,
                   strlen(prompt), total_read, req_has_instructions ? 1 : 0, req_has_tools ? 1 : 0);

        if (is_continuation) myds_set_continue_from_cache(ctx, true);

        /* Apply request configuration overrides if provided */
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

        /* 视觉注入：提示词已拼好，把所有图片块的逐位置向量交给引擎。
         * ★ 必须在 generate 之前；在发送 200 响应头之前完成，避免注入失败时向客户端连发 200 + 400 */
        if (!inject_image_embeddings(ctx, prompt, &img_sink, client_ip, client_port)) {
            resp_id_invalidate_all();
            send_response(client_fd, 400, "Bad Request", "application/json",
                          "{\"error\":{\"message\":\"image block could not be aligned with the prompt\"}}");
            goto cleanup_infer;
        }

        bool transport_ok = true;
        const bool req_thinking = extract_request_thinking_mode(body, g_server_default_thinking);
        if (stream) {
            const char *sse_header =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
            const char *initial = NULL;
            if (is_responses_endpoint) {
                if (req_thinking) {
                    initial =
                        "event: response.output_item.added\n"
                        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_myds_0\",\"status\":\"in_progress\",\"summary\":[],\"content\":[]}}\n\n"
                        "event: response.content_part.added\n"
                        "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"reasoning_text\",\"text\":\"\"}}\n\n";
                } else {
                    initial =
                        "event: response.output_item.added\n"
                        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n"
                        "event: response.content_part.added\n"
                        "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n";
                }
            } else {
                if (req_thinking) {
                    initial = "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"\"},\"finish_reason\":null}]}\n\n";
                } else {
                    initial = "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n";
                }
            }
            myds_sse_stream_ctx_t sctx = {
                .client_fd = client_fd,
                .responses_api = is_responses_endpoint,
                .thinking = req_thinking,
                .think_state = req_thinking ? 1 : 0,
                .msg_output_index = req_thinking ? 1 : 0,
            };
            if (!myds_server_sse_writer_start(&sctx, 1024u * 1024u, myds_server_sse_drain_timeout_ms()) ||
                !myds_server_sse_send(&sctx, sse_header, strlen(sse_header)) ||
                !myds_server_sse_send(&sctx, initial, strlen(initial))) {
                sctx.cancel_flag = 1;
                if (!sctx.writer) sctx.failed = true;
            }
            int generated = !sctx.cancel_flag && myds_generate_bytes_ex(ctx, prompt, myds_server_sse_callback, &sctx,
                                                                       &stop_reason, &prompt_tokens, &sctx.cancel_flag);
            myds_context_clear_image_embeddings(ctx); myds_vjob_sink_free(&img_sink);
            if (generated && !sctx.cancel_flag) (void)myds_server_sse_finish(&sctx);
            if (!generated || sctx.failed || sctx.cancel_flag) {
                resp_id_invalidate_all();
            }
            final_tokens = sctx.token_count;
            t_first = sctx.t_first;
            t_last = sctx.t_last;
            if (!generated || sctx.failed) {
                resp_id_invalidate_all();
                stop_reason = MYDS_STOP_ERROR;
                const char *error_event = is_responses_endpoint
                    ? "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\n"
                    : "data: {\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n";
                if (sctx.writer) (void)myds_server_sse_send(&sctx, error_event, strlen(error_event));
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
                    bool replayed = sse_emit_content(&sctx, sctx.acc + sctx.emitted,
                                                     sctx.acc_len - sctx.emitted, false);
                    sctx.emitted = sctx.acc_len;
                    if (!replayed || sctx.failed || sctx.cancel_flag) {
                        stop_reason = sctx.failed ? MYDS_STOP_ERROR : MYDS_STOP_CANCELLED;
                    }
                }
                /* 拆分 reasoning 与 content */
                const char *reasoning_ptr = NULL;
                size_t reasoning_len = 0;
                const char *content_ptr = sctx.acc ? sctx.acc : "";
                size_t content_len = scontent_len;

                if (sctx.thinking && sctx.acc) {
                    const char *t_end = strstr(sctx.acc, "</think>");
                    if (t_end && (size_t)(t_end - sctx.acc) <= scontent_len) {
                        const char *t_start = sctx.acc;
                        size_t r_len = (size_t)(t_end - t_start);
                        if (strncmp(t_start, "<think>", 7) == 0) {
                            t_start += 7;
                            if (r_len >= 7) r_len -= 7;
                            else r_len = 0;
                        }
                        reasoning_ptr = t_start;
                        reasoning_len = r_len;

                        const char *c_start = t_end + 8;
                        while (c_start < sctx.acc + scontent_len && (*c_start == '\r' || *c_start == '\n')) c_start++;
                        content_ptr = c_start;
                        if ((size_t)(c_start - sctx.acc) <= scontent_len) {
                            content_len = scontent_len - (size_t)(c_start - sctx.acc);
                        } else {
                            content_len = 0;
                        }
                    } else {
                        const char *t_start = sctx.acc;
                        size_t r_len = scontent_len;
                        if (strncmp(t_start, "<think>", 7) == 0) {
                            t_start += 7;
                            if (r_len >= 7) r_len -= 7;
                            else r_len = 0;
                        }
                        reasoning_ptr = t_start;
                        reasoning_len = r_len;
                        content_ptr = "";
                        content_len = 0;
                    }
                }

                if (stool_count && !sctx.failed && !sctx.cancel_flag) {
                    bool tsent;
                    if (is_responses_endpoint) {
                        tsent = sse_emit_responses_close_message(&sctx, content_ptr, content_len) &&
                                sse_emit_responses_function_calls(&sctx, stool, stool_count);
                    } else {
                        tsent = sse_emit_chat_tool_calls(&sctx, stool, stool_count);
                    }
                    if (!tsent) stop_reason = MYDS_STOP_CANCELLED;
                    if (tsent && is_responses_endpoint) {
                        char *items = responses_output_items_json_ex(
                            sctx.thinking, reasoning_ptr, reasoning_len, content_ptr, content_len, stool, stool_count);
                        const bool finished = items && sse_emit_responses_finish(
                            &sctx, model_alias, resp_id, (unsigned)prompt_tokens, (unsigned)final_tokens,
                            "", 0, items, 0);
                        free(items);
                        if (!finished) stop_reason = MYDS_STOP_ERROR;
                    }
                    myds_free_tool_calls(stool, stool_count);
                    stool = NULL;
                } else if (stool) {
                    myds_free_tool_calls(stool, stool_count);
                    stool = NULL;
                }
                if (!stool_count && !sctx.failed && !sctx.cancel_flag && is_responses_endpoint) {
                    char *items = responses_output_items_json_ex(
                        sctx.thinking, reasoning_ptr, reasoning_len, content_ptr, content_len, NULL, 0);
                    int emit_text_events = (sctx.think_state != 1) ? 1 : 0;
                    const bool finished = items && sse_emit_responses_finish(
                        &sctx, model_alias, resp_id, (unsigned)prompt_tokens, (unsigned)final_tokens,
                        content_ptr, content_len, items, emit_text_events);
                    free(items);
                    if (!finished) {
                        stop_reason = MYDS_STOP_ERROR;
                        const char *error_event = "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Response allocation failed\",\"type\":\"server_error\"}}\n\n";
                        (void)myds_server_sse_send(&sctx, error_event, strlen(error_event));
                    }
                } else if (!stool_count && !sctx.failed && !sctx.cancel_flag) {
                    const char *finish = (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL) ? "length" : "stop";
                    char chunk[512];
                    int len = snprintf(chunk, sizeof(chunk),
                        "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n", finish);
                    bool sent = len > 0 && (size_t)len < sizeof(chunk) && myds_server_sse_send(&sctx, chunk, (size_t)len);
                    if (sent && include_usage) {
                        len = snprintf(chunk, sizeof(chunk),
                            "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}\n\n",
                            prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                        sent = len > 0 && (size_t)len < sizeof(chunk) && myds_server_sse_send(&sctx, chunk, (size_t)len);
                    }
                    const char *done = "data: [DONE]\n\n";
                    if (!sent || !myds_server_sse_send(&sctx, done, strlen(done))) stop_reason = MYDS_STOP_CANCELLED;
                } else if (sctx.failed) {
                    stop_reason = MYDS_STOP_ERROR;
                    const char *error_event = is_responses_endpoint
                        ? "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\n"
                        : "data: {\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n";
                    (void)myds_server_sse_send(&sctx, error_event, strlen(error_event));
                } else if (sctx.cancel_flag) {
                    stop_reason = MYDS_STOP_CANCELLED;
                }
            } else {
                stop_reason = MYDS_STOP_CANCELLED;
            }
            /* Socket I/O runs in the writer. Drain has one overall deadline
             * (default 1s, MYDS_SSE_DRAIN_TIMEOUT_MS accepts 1..30000), even
             * if a slow peer keeps making partial progress.
             * The inference slot remains held during this bounded tail wait. */
            const bool drained = myds_server_sse_writer_join(&sctx, sctx.cancel_flag && !sctx.failed);
            if (!drained && stop_reason != MYDS_STOP_ERROR) stop_reason = MYDS_STOP_CANCELLED;
            if (!drained || sctx.failed || sctx.cancel_flag) transport_ok = false;
            free(sctx.acc); sctx.acc = NULL;   /* 每请求一份累积缓冲，必须释放 */
        } else {
            myds_non_stream_ctx_t ns = {.client_fd = client_fd};
            int generated = myds_generate_bytes_ex(ctx, prompt, myds_server_non_stream_callback, &ns,
                                                   &stop_reason, &prompt_tokens, &ns.cancel_flag);
            myds_context_clear_image_embeddings(ctx); myds_vjob_sink_free(&img_sink);
            if (!generated || ns.failed || ns.cancel_flag) {
                resp_id_invalidate_all();
            }
            final_tokens = ns.token_count;
            t_first = ns.t_first;
            t_last = ns.t_last;
            if (!generated || ns.failed) {
                resp_id_invalidate_all();
                stop_reason = MYDS_STOP_ERROR;
                transport_ok = false;
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

                /* 拆分 reasoning 与 content */
                const char *reasoning_ptr = NULL;
                size_t reasoning_len = 0;
                const char *content_ptr = ns.buf ? ns.buf : "";
                size_t content_len = tool_content_len;

                if (req_thinking && ns.buf) {
                    const char *t_end = strstr(ns.buf, "</think>");
                    if (t_end && (size_t)(t_end - ns.buf) <= tool_content_len) {
                        const char *t_start = ns.buf;
                        size_t r_len = (size_t)(t_end - t_start);
                        if (strncmp(t_start, "<think>", 7) == 0) {
                            t_start += 7;
                            if (r_len >= 7) r_len -= 7;
                            else r_len = 0;
                        }
                        reasoning_ptr = t_start;
                        reasoning_len = r_len;

                        const char *c_start = t_end + 8;
                        while (c_start < ns.buf + tool_content_len && (*c_start == '\r' || *c_start == '\n')) c_start++;
                        content_ptr = c_start;
                        if ((size_t)(c_start - ns.buf) <= tool_content_len) {
                            content_len = tool_content_len - (size_t)(c_start - ns.buf);
                        } else {
                            content_len = 0;
                        }
                    } else {
                        const char *t_start = ns.buf;
                        size_t r_len = tool_content_len;
                        if (strncmp(t_start, "<think>", 7) == 0) {
                            t_start += 7;
                            if (r_len >= 7) r_len -= 7;
                            else r_len = 0;
                        }
                        reasoning_ptr = t_start;
                        reasoning_len = r_len;
                        content_ptr = "";
                        content_len = 0;
                    }
                }

                char *body_escaped = json_escape_alloc(content_ptr, content_len);
                char *reasoning_escaped = reasoning_ptr ? json_escape_alloc(reasoning_ptr, reasoning_len) : NULL;
                char *calls_json = tool_count ? openai_tool_calls_json(tool_calls, tool_count) : NULL;
                char *items_json = responses_output_items_json_ex(
                    req_thinking, reasoning_ptr, reasoning_len, content_ptr, content_len, tool_calls, tool_count);
                char *content_field = body_escaped ? format_alloc("\"%s\"", body_escaped) : NULL;
                if (tool_count) finish = "tool_calls";

                if (body_escaped && alias_json) {
                    if (is_responses_endpoint) {
                        response = items_json ? format_alloc(
                            "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,"
                            "\"status\":\"completed\",\"model\":\"%s\",\"output\":%s,"
                            "\"usage\":{\"input_tokens\":%u,\"output_tokens\":%u,\"total_tokens\":%u}}",
                            resp_id, now, alias_json, items_json,
                            prompt_tokens, final_tokens, prompt_tokens + final_tokens) : NULL;
                    } else if (tool_count && calls_json && content_field) {
                        if (reasoning_escaped) {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                                "\"content\":%s,\"reasoning_content\":\"%s\",\"tool_calls\":%s},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                                now, alias_json, content_field, reasoning_escaped, calls_json, finish,
                                prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                        } else {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                                "\"content\":%s,\"tool_calls\":%s},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                                now, alias_json, content_field, calls_json, finish,
                                prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                        }
                    } else {
                        if (reasoning_escaped) {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\",\"reasoning_content\":\"%s\"},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                                now, alias_json, body_escaped, reasoning_escaped, finish, prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                        } else {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                                now, alias_json, body_escaped, finish, prompt_tokens, final_tokens, prompt_tokens + final_tokens);
                        }
                    }
                }
                free(reasoning_escaped);
                free(content_field);
                myds_free_tool_calls(tool_calls, tool_count);
                free(items_json);
                free(calls_json);
                free(body_escaped);
                free(escaped);
                free(alias_json);
                if (response) {
                    if (!send_response(client_fd, 200, "OK", "application/json", response)) {
                        stop_reason = MYDS_STOP_CANCELLED;
                        transport_ok = false;
                    }
                    free(response);
                } else {
                    stop_reason = MYDS_STOP_ERROR;
                    transport_ok = false;
                    send_response(client_fd, 500, "Internal Server Error", "application/json",
                                  "{\"error\":{\"message\":\"Response allocation failed\",\"type\":\"server_error\"}}");
                }
            } else {
                transport_ok = false;
            }
            if (ns.failed || ns.cancel_flag) transport_ok = false;
            free(ns.buf);
        }

        if (stop_reason != MYDS_STOP_ERROR && stop_reason != MYDS_STOP_CANCELLED && transport_ok) {
            resp_id_register(resp_id);
        } else {
            resp_id_invalidate_all();
        }

        double t_end = myds_server_now_s();
        double wall_s = t_end - t_start;
        double prefill_s = (final_tokens > 0 && t_first > t_start) ? (t_first - t_start) : wall_s;
        double ttft_s = prefill_s;
        double decode_s = (final_tokens > 1 && t_last > t_first) ? (t_last - t_first) : 0.0;
        double decode_tps = (decode_s > 0.0 && final_tokens > 1) ? ((double)(final_tokens - 1) / decode_s) : ((wall_s > 0.0) ? ((double)final_tokens / wall_s) : 0.0);
        const char *final_finish_str = (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL) ? "length" : (stop_reason == MYDS_STOP_CANCELLED ? "cancelled" : (stop_reason == MYDS_STOP_ERROR ? "error" : "stop"));

        /* prefill 速率直接取自引擎的 prefill 阶段计时，不用 ttft 反推：
         * ttft 混了排队/分词/首个解码步，反推会低估速率（实测差数倍），
         * 而「复用命中的轮次」更会把固定开销摊到极少量 token 上 ⇒ 看起来极慢。 */
        myds_prefill_stats_t pf;
        myds_last_prefill_stats(ctx, &pf);
        char prefill_str[96];
        if (pf.prefill_tokens == 0) {
            snprintf(prefill_str, sizeof(prefill_str), "none (all %u cached)", pf.cached_tokens);
        } else {
            double pf_tps = (pf.prefill_seconds > 0.0)
                                ? (double)pf.prefill_tokens / pf.prefill_seconds : 0.0;
            snprintf(prefill_str, sizeof(prefill_str), "%u tok in %.3fs = %.1f t/s (cached %u)",
                     pf.prefill_tokens, pf.prefill_seconds, pf_tps, pf.cached_tokens);
        }
        MYDS_LOG_I("[%s:%d] Completed %s %s - prompt: %u tok, prefill: %s, ttft: %.3fs, "
                   "decode: %u tok in %.3fs = %.2f t/s, wall: %.3fs, finish_reason: %s",
                   client_ip, client_port,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   path, prompt_tokens, prefill_str, ttft_s,
                   final_tokens, decode_s, decode_tps, wall_s, final_finish_str);
        if (ctx->dspark && ctx->dspark->proposed_tokens > dspark_prop_before) {
            uint64_t req_prop = ctx->dspark->proposed_tokens - dspark_prop_before;
            uint64_t req_acc  = ctx->dspark->accepted_tokens - dspark_acc_before;
            uint64_t req_cyc  = ctx->dspark->cycles - dspark_cyc_before;
            MYDS_LOG_I("[%s:%d] DSpark Speculative: proposed=%llu, accepted=%llu (%.1f%% accept rate, %.2f tok/cycle, cumulative: %llu/%llu = %.1f%%)",
                       client_ip, client_port,
                       (unsigned long long)req_prop,
                       (unsigned long long)req_acc,
                       req_prop > 0 ? (100.0 * (double)req_acc / (double)req_prop) : 0.0,
                       (double)req_acc / (double)(req_cyc > 0 ? req_cyc : 1),
                       (unsigned long long)ctx->dspark->accepted_tokens,
                       (unsigned long long)ctx->dspark->proposed_tokens,
                       ctx->dspark->proposed_tokens > 0 ? (100.0 * (double)ctx->dspark->accepted_tokens / (double)ctx->dspark->proposed_tokens) : 0.0);
        }

cleanup_infer:
        myds_context_clear_image_embeddings(ctx);
        myds_vjob_sink_free(&img_sink);
        myds_set_continue_from_cache(ctx, false);
        myds_context_set_config(ctx, &prev_cfg);
        t_infer_slot_depth = 0;
        pthread_mutex_unlock(&g_infer_mutex);
        server_release_infer_slot();

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
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    server_client_job_t *job = (server_client_job_t *)arg;
    handle_client(job->client_fd, job->ctx, job->model_alias, job->client_ip, job->client_port);
    free(job);
    return NULL;
}

int myds_server_start(myds_context_t *ctx, const char *host, int port, const char *model_alias) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
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

    if (ctx) {
        g_server_default_thinking = ctx->cfg.thinking_mode;
    }

    if (listen(g_server_fd, 64) < 0) {
        MYDS_LOG_E("Failed to listen on socket: %s", strerror(errno));
        close(g_server_fd);
        return 1;
    }

    MYDS_LOG_I("HTTP server listening on http://%s:%d (alias: %s)",
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
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 4 * 1024 * 1024);
        if (pthread_create(&tid, &attr, server_client_thread, job) == 0) {
            pthread_detach(tid);
        } else {
            handle_client(client_fd, ctx, model_alias, client_ip, client_port);
            free(job);
        }
        pthread_attr_destroy(&attr);
    }

    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }

    MYDS_LOG_I("Server stopped gracefully");
    return 0;
}
