/* myds_http_io.c — HTTP 传输：socket 写入、SSE 有界队列与发射
 *
 * 2026-11-01 从 myds_server.c 纯机械拆分而来：函数体与注释未改。
 *
 * 判据是「碰不碰 fd」。本文件里每个函数要么写 socket，要么管理给它们送字节的
 * 线程与队列；凡是只处理字符串的（提示词渲染、响应 JSON 组装）都在
 * myds_proto_openai.c。
 *
 * 这一层最容易出的一类错误是「写端已经没了，生产端还在造」—— 所以
 * client_disconnected / sse_transport_failed / 有界队列的取消路径都集中在这里。
 */
#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <poll.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/types.h>

double myds_server_now_s(void) {
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


char *format_alloc(const char *format, ...) {
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
    if (w->closing && !w->aborted && myds_server_now_s() >= w->drain_deadline) {
        w->failed = w->aborted = true;
        stopped = true;
        /* 排空超时是**真**异常：客户端还连着（否则会是 EPIPE/ECONNRESET 路径），
         * 但字节在规定时间内发不完 —— 要么对端读得极慢，要么我们自己写不动。
         * 原实现静默置位，上层只看到「取消了」，查不到原因。 */
        MYDS_LOG_W("sse: drain deadline hit with %zu bytes still pending "
                   "(byte_limit %zu, timeout %ums) - dropping the stream",
                   w->pending_bytes, w->byte_limit, w->drain_timeout_ms);
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
        const int saved_errno = errno;
        const bool ok = sse_writer_send_packet(w, packet);
        const int send_errno = ok ? 0 : errno;
        (void)saved_errno;
        pthread_mutex_lock(&w->lock);
        w->pending_bytes -= sizeof(*packet) + packet->length;
        if (!ok) w->failed = w->aborted = true;
        pthread_mutex_unlock(&w->lock);
        /* 客户端主动断开是**常态**（agent 每次取消都会），打 WARN 会刷爆日志，
         * 也让真正的故障淹没。只有不属于「对端走了」的失败才值得一条 WARN。 */
        if (!ok && send_errno != EPIPE && send_errno != ECONNRESET &&
            send_errno != ENOTCONN && send_errno != 0) {
            MYDS_LOG_W("sse: send failed with %zu bytes in flight (errno %d: %s) - "
                       "this is not a clean client disconnect",
                       packet->length, send_errno, strerror(send_errno));
        }
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
            if (!packet) {
                w->failed = w->aborted = true; stream->failed = true; ok = false;
                MYDS_LOG_E("sse: cannot allocate a %zu-byte packet (byte_limit %zu, "
                           "pending %zu) - streaming aborted",
                           length, w->byte_limit, w->pending_bytes);
            }
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

bool client_disconnected(int fd) {
    if (fd < 0) return false;
    char byte;
    ssize_t r;
    do { r = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT); } while (r < 0 && errno == EINTR);
    return r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
}

/* 引擎在 prefill 的 macroblock 边界回调它。
 * 单个静态 fd：推理由 g_infer_mutex 串行化，同一时刻只有一个请求在生成。 */
int g_prefill_probe_fd = -1;
bool server_client_alive_probe(void *user_data) {
    (void)user_data;
    return !client_disconnected(g_prefill_probe_fd);
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

bool sse_emit_content(myds_sse_stream_ctx_t *sctx, const char *piece, size_t length, bool final) {
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
bool sse_emit_responses_close_message(myds_sse_stream_ctx_t *sctx,
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

bool send_response(int fd, int code, const char *status, const char *content_type, const char *body) {
    size_t body_len = body ? strlen(body) : 0;
    char header[1024];
    /* 2026-10-25: CORS 可关可限，见 cors_origin_header()。 */
    char cors[600] = "";
    const char *origin = cors_origin_header();
    if (origin) {
        snprintf(cors, sizeof(cors),
                 "Access-Control-Allow-Origin: %s\r\n"
                 "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                 "Access-Control-Allow-Headers: *\r\n",
                 origin);
    }
    int hlen = snprintf(header, sizeof(header),
                        "HTTP/1.1 %d %s\r\n"
                        "Content-Type: %s\r\n"
                        "Content-Length: %zu\r\n"
                        "%s"
                        "Connection: close\r\n\r\n",
                        code, status, content_type, body_len, cors);
    if (hlen < 0 || (size_t)hlen >= sizeof(header)) return false;
    if (!myds_server_send_all(fd, header, (size_t)hlen)) return false;
    return body_len == 0 || myds_server_send_all(fd, body, body_len);
}
