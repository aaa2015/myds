/* myds_infer_queue.c — 推理请求 FIFO 队列与背压
 *
 * 2026-11-01 从 myds_server.c 纯机械拆分而来：函数体与注释未改。
 *
 * 单槽推理引擎同一时刻只能运行一个推理请求，队列负责排队与背压（503 语义）。
 * 与本次拆分的关系：这块只依赖 pthread 与 MYDS_LOG_*，与 HTTP 传输、协议渲染
 * 无关；把它单独放能让「为什么这个请求被拒」与「请求怎么解析」分开读。
 *
 * 注意边界：真正的推理互斥量 g_infer_mutex 不在这里 —— 它由 handle_client
 * 跨越整个请求持有，队列只决定「谁拿到名额」，不决定「谁在跑」。
 */
#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_log.h"
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

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

int server_acquire_infer_slot(int timeout_ms, int max_depth) {
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

void server_release_infer_slot(void) {
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

/* 日志要打「拒绝时的队列深度」。g_queue_count 是块内静态变量，
 * 拆出来后必须有个读口 —— 直接 extern 一个静态计数器更容易被误写。 */
uint32_t server_infer_queue_depth(void) {
    pthread_mutex_lock(&g_queue_lock);
    const uint32_t n = g_queue_count;
    pthread_mutex_unlock(&g_queue_lock);
    return n;
}
