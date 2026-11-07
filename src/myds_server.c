#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_internal.h"
#include "myds_dspark.h"
#include "myds_vjob.h"
#include "myds_steer.h"
#include "myds_log.h"
#include "myds_trace.h"
#include "myds_debugstate.h"
#include "myds_sysstat.h"
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
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
/* Implemented in myds_debugstate.c, which is linked only into full builds.
 * Declared weak so that including myds_server.c for other reasons does not pull
 * the knob inventory, the GPU counters and the Metal library path into that
 * target. A NULL pointer here means "not linked", and the endpoint reports 404. */
/* /debug/state is registered by the full binary rather than linked implicitly.
 *
 * A weak symbol was tried first and does not survive this toolchain's link rules
 * for a strong declaration elsewhere; more importantly, either approach would tie
 * myds_server.c to the knob inventory, the GPU counters and the Metal library
 * path, which breaks every lightweight test that includes this file. That already
 * happened once today with a different function. A provider pointer keeps
 * myds_server.c's dependency set unchanged and makes "not linked" an explicit,
 * observable state instead of a linker accident. */
static myds_debug_state_provider_t g_debug_state_provider = NULL;
static myds_debug_predicate_t g_debug_enabled = NULL;
static myds_debug_loopback_t g_debug_is_loopback = NULL;
static myds_debug_token_t g_debug_token_ok = NULL;

void myds_server_set_debug_provider(myds_debug_state_provider_t state,
                                    myds_debug_predicate_t enabled,
                                    myds_debug_loopback_t is_loopback,
                                    myds_debug_token_t token_ok) {
    g_debug_state_provider = state;
    g_debug_enabled = enabled;
    g_debug_is_loopback = is_loopback;
    g_debug_token_ok = token_ok;
}

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

volatile sig_atomic_t g_running = 1;
/* 2026-10-25: 在服务里的客户端线程数。用来给连接数封顶——
 * 每条连接分配 4 MB 栈（pthread_attr_setstacksize），无上限时
 * 几千条空闲连接即可把内存吃光。默认 256，可用 MYDS_MAX_CONN 调整。 */
static _Atomic unsigned g_client_threads;
static unsigned g_emfile_streak;   /* 连续 EMFILE 次数，用于日志降噪 */
/* 2026-11-07：关机时不再接受新请求，并等已有请求线程退完再让 main 释放 context。
 *
 * 为什么必须有：客户端线程是 pthread_detach 的，accept 循环退出后
 * myds_server_start 立刻返回，而 main 紧接着调 myds_context_free(ctx)。
 * 此时还在跑的请求线程会用已经释放的 ctx / chunk_workspace / prompt_engram_buf。
 *
 * 实测两种崩溃都由此而来（栈相同，只是最后几帧不同）：
 *   既有：objc_release → myds_metal_buffer_free → v41_chunk_workspace_free
 *   本仓 P0：myds_engram_decode_row → prepare_batch_step → v41_engram_async_main
 * 判据：拿 P0 之前的二进制 7185c173 做同样操作，崩在同一位置
 * ⇒ 不是某次改动引入的，是「关机不等在跑请求」这个结构缺陷。
 *
 * ★ 曾经有一个 `g_draining` 原子标志，只写不读（accept 循环退出后 fd 已关，
 *   不会再进新连接，所以没有读者）。已删除：一个不被读取的标志会让人以为
 *   「已经有 draining 语义」，而实际行为只由 g_client_threads 归零决定。
 *   若将来要让超长 prefill 在 SIGTERM 时更快主动中断，需要的是在 prefill 的
 *   分块循环里检查一个新的谓词（并明确「提前中断 = 请求被截断」这个语义），
 *   而不是复用一个空标志。
 */

static unsigned max_client_threads(void) {
    static unsigned cached;
    if (cached) return cached;
    cached = 256;
    const char *e = getenv("MYDS_MAX_CONN");
    if (e && *e) {
        const long v = atol(e);
        if (v >= 1 && v <= 4096) cached = (unsigned)v;
        else MYDS_LOG_W("MYDS_MAX_CONN=%s 非法（需 1..4096），按 256 处理", e);
    }
    return cached;
}
static int g_server_fd = -1;
static pthread_mutex_t g_infer_mutex = PTHREAD_MUTEX_INITIALIZER;
void server_vjob_gpu_lock(void);
void server_vjob_gpu_unlock(void);
static pthread_once_t  g_vjob_lock_once = PTHREAD_ONCE_INIT;
static void init_vjob_gpu_lock(void) {
    myds_vjob_set_gpu_lock(server_vjob_gpu_lock, server_vjob_gpu_unlock);
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

static void handle_signal(int sig) {
    (void)sig;
    g_running = 0;
    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }
}



/* MYDS_REQUEST_PHASES=1 prints one JSON line per finished request with the
 * server-side stage split (queue / render / setup / engine). Parsed once: this
 * sits on the request path and a getenv per request is avoidable noise.
 *
 * Deliberately only clock_gettime around existing boundaries. Ledger §B.3: the
 * instrumentation must not perturb what it measures, so nothing here inserts a
 * command buffer or forces a commit inside the engine. */

/* OpenAI usage 的扩展段：缓存明细 + llama.cpp 风格的 timings。
 *
 * 为什么要加：客户端（myagent 等）直接读服务器给的值，字段名沿用
 * llama-server 的约定 —— prompt_n / prompt_ms / prompt_per_second /
 * predicted_n / predicted_ms / predicted_per_second，外加 cache_n。
 * 我们原来只发 prompt_tokens / completion_tokens / total_tokens，
 * 于是同一块界面在参考实现上有数、在我们这里只能走兜底（显示 0.0）。
 *
 * 口径与 llama.cpp 一致：
 *   prompt_n            本次真正跑过的 prompt token（不含命中缓存的部分）
 *   prompt_per_second   prompt_n / prompt_ms
 *   predicted_n         本次 decode 的步数
 *   predicted_per_second predicted_n / predicted_ms
 *   cache_n             命中前缀缓存的 token 数
 *
 * 取不到的值输出 null，不是 0 —— 0 会被读成「测了，是零」。
 * 同时发 OpenAI 标准的 usage.prompt_tokens_details.cached_tokens，
 * 让不认 timings 的客户端也能看到缓存命中。 */

/* 对外的 completion_tokens：内容 token 数 + （生成了 EOS ? 1 : 0）。
 * 与 DeepSeek / llama.cpp 一致 —— 它们的 completion_tokens / predicted_n 都含 EOS。
 * 内部 final_tokens 仍是纯内容数，供 decode 速率与追踪使用。 */
static inline uint32_t reported_completion_tokens(uint32_t content, bool eos) {
    return content + (eos ? 1u : 0u);
}

/* 一次算清、四处复用的计时结构。
 *
 * 为什么抽成结构体：同一组数字要出现在三个位置（usage 里的扁平字段、
 * usage.timings、响应顶层的 timings），而客户端读的位置各不相同 ——
 * mycode 读顶层 `timings` 或 `response.timings`，myagent 的 Flutter 层读
 * `usage.prompt_tps`。三处各算一遍必然会漂。
 *
 * 口径（与 DeepSeek / llama.cpp 一致）：
 *   prompt_n  本次真正跑过 prefill 的 token 数（不含命中缓存的）
 *   cached    prompt_tokens - prompt_n —— 「没有花 prefill 算力」的那部分
 *   predicted_n decode 步数，生成 EOS 时含求 EOS 的那一步
 */
typedef struct {
    uint32_t prompt_n;
    uint32_t dec_n;
    uint32_t cached;
    double   prompt_ms, prompt_ps;
    double   dec_ms, dec_ps;
} myds_timings_t;

static void server_timings_compute(myds_context_t *ctx, uint32_t out_tokens,
                                   double t_first, double t_end, bool eos_generated,
                                   myds_timings_t *t) {
    memset(t, 0, sizeof(*t));
    myds_prefill_stats_t pf;
    myds_last_prefill_stats(ctx, &pf);

    const bool have_prompt = pf.valid && pf.prompt_tokens > 0;
    const uint32_t prompt_tokens = have_prompt ? pf.prompt_tokens : 0u;
    t->prompt_n = have_prompt ? pf.prefill_tokens : 0u;

    /* cached 的定义是「没有花 prefill 算力的 prompt token」，即 prompt - prefill_n。
     *
     * 不用 resume_from：最后一个 prompt token 本来就不参与 prefill（它是 decode 的种子），
     * 于是 prompt_tokens - resume_from 会恒等于 prefill_n + 1。客户端按
     * (prompt_tokens - cached_tokens) / ttft 反推 prefill 速率时，那个 +1 会让
     * 「全部命中」误显示成「1 token 用了 138ms = 7.2 t/s」。
     * 按「没重算」定义，全命中时残留为 0，客户端不会再编出一个速率。 */
    t->cached = have_prompt ? (prompt_tokens - t->prompt_n) : 0u;

    t->prompt_ms = (have_prompt && t->prompt_n > 0 && pf.prefill_seconds > 0.0)
        ? pf.prefill_seconds * 1000.0 : -1.0;
    t->prompt_ps = (t->prompt_n > 0 && pf.prefill_seconds > 0.0)
        ? (double)t->prompt_n / pf.prefill_seconds : -1.0;

    t->dec_ms = -1.0;
    if (out_tokens > 0 && t_first > 0.0 && t_end > t_first) {
        t->dec_ms = (t_end - t_first) * 1000.0;
        t->dec_n = eos_generated ? out_tokens : (out_tokens > 0 ? out_tokens - 1u : 0u);
    }
    t->dec_ps = (t->dec_n > 0 && t->dec_ms > 0.0)
        ? (double)t->dec_n / (t->dec_ms / 1000.0) : -1.0;
}

/* 只产出 timings 对象本身（不含前导逗号），供「放在哪」由调用方决定。 */
static void server_timings_json(const myds_timings_t *t, char *out, size_t cap) {
    char pms[24] = "null", pps[24] = "null", dms[24] = "null", dps[24] = "null";
    char ptm[24] = "null", dtm[24] = "null";
    if (t->prompt_ms >= 0.0) snprintf(pms, sizeof(pms), "%.3f", t->prompt_ms);
    if (t->prompt_ps >= 0.0) snprintf(pps, sizeof(pps), "%.2f", t->prompt_ps);
    if (t->dec_ms >= 0.0) snprintf(dms, sizeof(dms), "%.3f", t->dec_ms);
    if (t->dec_ps >= 0.0) snprintf(dps, sizeof(dps), "%.2f", t->dec_ps);
    if (t->prompt_ms >= 0.0 && t->prompt_n > 0)
        snprintf(ptm, sizeof(ptm), "%.4f", t->prompt_ms / t->prompt_n);
    if (t->dec_ms >= 0.0 && t->dec_n > 0)
        snprintf(dtm, sizeof(dtm), "%.4f", t->dec_ms / t->dec_n);

    snprintf(out, cap,
             "{\"prompt_n\":%u,\"prompt_ms\":%s,\"prompt_per_token_ms\":%s,"
             "\"prompt_per_second\":%s,\"predicted_n\":%u,\"predicted_ms\":%s,"
             "\"predicted_per_token_ms\":%s,\"predicted_per_second\":%s,\"cache_n\":%u}",
             t->prompt_n, pms, ptm, pps, t->dec_n, dms, dtm, dps, t->cached);
}

/* usage 里的扩展段。timings 在这份里是嵌套的（usage.timings），
 * 同时把速率扁平化（prompt_tps 等）—— 不同客户端读不同位置。 */
static void server_usage_extra_json(const myds_timings_t *t, char *out, size_t cap) {
    char pps[24] = "null", dps[24] = "null", chr_buf[24] = "null";
    if (t->prompt_ps >= 0.0) snprintf(pps, sizeof(pps), "%.2f", t->prompt_ps);
    if (t->dec_ps >= 0.0) snprintf(dps, sizeof(dps), "%.2f", t->dec_ps);
    if (t->cached + t->prompt_n > 0)
        snprintf(chr_buf, sizeof(chr_buf), "%.2f",
                 100.0 * (double)t->cached / (double)(t->cached + t->prompt_n));

    char tim[384];
    server_timings_json(t, tim, sizeof(tim));

    snprintf(out, cap,
             ",\"prompt_tps\":%s,\"completion_tps\":%s,"
             "\"server_prompt_tps\":%s,\"server_predicted_tps\":%s"
             ",\"prompt_tokens_details\":{\"cached_tokens\":%u}"
             ",\"cache_hit_tokens\":%u,\"cache_hit_rate\":%s"
             ",\"timings\":%s",
             pps, dps, pps, dps, t->cached, t->cached, chr_buf, tim);
}

static bool myds_server_request_phases_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("MYDS_REQUEST_PHASES");
        cached = (value && *value && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

/* Build the request_phases JSON body. Pure formatting so it can be unit-tested
 * without a model: a printf whose argument types drift from the format string
 * compiles fine and produces unparsable output, which is worse than no logging.
 *
 * Every unavailable stage is emitted as null rather than 0.0 -- a printed 0.0 is
 * indistinguishable from "measured, took no time", and that is how a fabricated
 * number reaches a ledger. */
typedef struct {
    double queue_s;
    double render_s;      /* < 0 => not measured */
    double setup_s;       /* < 0 => not measured */
    double engine_s;      /* < 0 => not measured */
    double prefill_s;     /* < 0 => not measured */
    double decode_s;      /* < 0 => not measured */
    double unattributed_s;
    uint32_t prompt_tokens;
    uint32_t cached_tokens;
    uint32_t out_tokens;
    int stream;
} myds_request_phases_t;

static void myds_request_phases_json(const myds_request_phases_t *p, char *out, size_t cap) {
    char render[32] = "null", setup[32] = "null", engine[32] = "null";
    char prefill[32] = "null", decode[32] = "null", unattr[32] = "null";
    if (p->render_s >= 0.0) snprintf(render, sizeof(render), "%.6f", p->render_s);
    if (p->setup_s >= 0.0) snprintf(setup, sizeof(setup), "%.6f", p->setup_s);
    if (p->engine_s >= 0.0) snprintf(engine, sizeof(engine), "%.6f", p->engine_s);
    if (p->prefill_s >= 0.0) snprintf(prefill, sizeof(prefill), "%.6f", p->prefill_s);
    if (p->decode_s >= 0.0) snprintf(decode, sizeof(decode), "%.6f", p->decode_s);
    /* Negative is meaningful here (t_last precedes the engine's return), so it is
     * printed as-is; only NaN means "could not be computed" and becomes null. */
    if (p->unattributed_s == p->unattributed_s)
        snprintf(unattr, sizeof(unattr), "%.6f", p->unattributed_s);
    snprintf(out, cap,
             "{\"queue\":%.6f,\"render\":%s,\"setup\":%s,\"engine\":%s,"
             "\"prefill\":%s,\"decode\":%s,\"engine_unattributed\":%s,"
             "\"prompt_tok\":%u,\"cached_tok\":%u,\"out_tok\":%u,\"stream\":%s}",
             p->queue_s, render, setup, engine, prefill, decode, unattr,
             p->prompt_tokens, p->cached_tokens, p->out_tokens,
             p->stream ? "true" : "false");
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

static void handle_client(int client_fd, myds_context_t *ctx, const char *model_alias, const char *client_ip, int client_port);

bool myds_server_format_models_json(const char *model_alias, bool vision_enabled,
                                    char *out, size_t out_cap) {
    if (!out || out_cap == 0) return false;
    out[0] = '\0';
    const char *raw_id = (model_alias && model_alias[0]) ? model_alias : "myds";
    char escaped_id[512];
    if (!json_escape_bytes(raw_id, strlen(raw_id), escaped_id, sizeof(escaped_id), NULL)) {
        return false;
    }
    const char *in_mods = vision_enabled ? "\"text\",\"image\"" : "\"text\"";
    int n = snprintf(out, out_cap,
                     "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"myds\","
                     "\"architecture\":{\"input_modalities\":[%s],\"output_modalities\":[\"text\"]}}]}",
                     escaped_id, in_mods);
    if (n < 0 || (size_t)n >= out_cap) {
        out[0] = '\0';
        return false;
    }
    return true;
}

static void handle_client(int client_fd, myds_context_t *ctx, const char *model_alias, const char *client_ip, int client_port) {
    pthread_once(&g_vjob_lock_once, init_vjob_gpu_lock);
    /* ── 接收截止时间（2026-10-25）─────────────────────────
     * SO_RCVTIMEO 是【每次 recv】的超时。攻击者每 29 秒发 1 个字节，
     * 就能把这个连接永远挂在半开状态——每次 recv 都“没超时”，
     * 而总时长无上限。配合连接数上限，几百条 drip 就能把名额占满。
     *
     * 这里加一个【整体】接收 deadline：从进入 handle_client 开始计时，
     * 超过 MYDS_RECV_DEADLINE_S（默认 120s）就放弃这条连接。
     * 120s 足够传 8 MB 体（即使慢速网络也够），但不足以被无限拖延。
     */
    const char *dl_env = getenv("MYDS_RECV_DEADLINE_S");
    double recv_deadline_s = 120.0;
    if (dl_env && *dl_env) {
        const double v = atof(dl_env);
        if (v >= 5.0 && v <= 3600.0) recv_deadline_s = v;
        else MYDS_LOG_W("MYDS_RECV_DEADLINE_S=%s 非法（5..3600），按 120 处理", dl_env);
    }
    const double recv_start = myds_server_now_s();
#define RECV_DEADLINE_HIT() (myds_server_now_s() - recv_start > recv_deadline_s)

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
        if (RECV_DEADLINE_HIT()) {
            MYDS_LOG_W("[%s:%d] 接收超时（整体 deadline %.0fs），已读 %zd 字节 ⇒ 408",
                       client_ip, client_port, recv_deadline_s, total_read);
            send_response(client_fd, 408, "Request Timeout", "application/json",
                          "{\"error\":{\"message\":\"Request body not received in time\"}}");
            free(req_buf);
            close(client_fd);
            return;
        }
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
        char cors_res_buf[700];
        const char *origin_pre = cors_origin_header();
        if (origin_pre) {
            snprintf(cors_res_buf, sizeof(cors_res_buf),
                     "HTTP/1.1 204 No Content\r\n"
                     "Access-Control-Allow-Origin: %s\r\n"
                     "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                     "Access-Control-Allow-Headers: *\r\n"
                     "Connection: close\r\n\r\n", origin_pre);
        } else {
            /* MYDS_CORS_ORIGIN=none：不发布任何 CORS 头，OPTIONS 仍回 204 */
            snprintf(cors_res_buf, sizeof(cors_res_buf),
                     "HTTP/1.1 204 No Content\r\n"
                     "Connection: close\r\n\r\n");
        }
        (void)myds_server_send_all(client_fd, cors_res_buf, strlen(cors_res_buf));
        free(req_buf);
        close(client_fd);
        return;
    }

    /* GET /debug/state.
     *
     * myds_debug_state_json is linked only into full builds (weak declaration
     * below), so a test that includes this file for other reasons still links.
     * Closed means 404, not 403: an unauthenticated server should not confirm
     * that a hidden endpoint exists.
     *
     * The counter accessor inside is called with reset=false. With reset=true it
     * atomically zeroes the counters, which would destroy the arm separation that
     * every A/B in the ledger relies on. A read must never subtract evidence. */
    if (strcmp(path, "/debug/state") == 0) {
        /* Two independent controls, because either alone is insufficient:
         *   loopback -- cheap, but a co-located reverse proxy satisfies it;
         *   token    -- survives proxies, and fails closed when unset. */
        char presented[256] = "";
        extract_http_header_str(req_buf, "X-Debug-Token", presented, sizeof presented);
        /* 来源门槛：回环，或 MYDS_DEBUG_ALLOW_CIDR 列出的网段（默认只认回环）。
         * token 仍然必需 —— 白名单只决定「从哪来」，不决定「是不是自己人」。 */
        const bool loopback = myds_debug_ip_allowed(client_ip);
        const bool token_ok = g_debug_token_ok != NULL && g_debug_token_ok(presented);
        const bool open = g_debug_state_provider != NULL && g_debug_enabled != NULL &&
                          g_debug_enabled() && loopback && token_ok;
        if (!open) {
            MYDS_LOG_W("[%s:%d] GET /debug/state -> 404 (registered %s, switch %s, "
                       "loopback %s, token %s)",
                       client_ip, client_port,
                       g_debug_state_provider ? "yes" : "no",
                       (g_debug_enabled && g_debug_enabled()) ? "on" : "off",
                       loopback ? "yes" : "no",
                       presented[0] ? (token_ok ? "ok" : "mismatch") : "absent");
            send_response(client_fd, 404, "Not Found", "application/json",
                          "{\"error\":\"not found\"}");
            free(req_buf);
            close(client_fd);
            return;
        }
        char state[2048];
        g_debug_state_provider(state, sizeof(state), ctx);
        MYDS_LOG_I("[%s:%d] GET /debug/state -> 200 OK", client_ip, client_port);
        send_response(client_fd, 200, "OK", "application/json", state);
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

    /* GET /trace* — 请求追踪页面与数据。
     *
     * 与 /debug/state 同一套门槛（回环 + token + 开关），理由相同：这是未鉴权服务，
     * 暴露「谁在什么时候发了什么请求、各自花了多久」属于运行数据，不能默认对全网开放。
     * 但页面本身要能一眼打开，所以门槛与 debug 一致而不是更严。 */
    if (strncmp(path, "/trace", 6) == 0) {
        /* path 来自 sscanf("%255s")，**包含 query string**（`/trace?token=x` 是一个
         * 完整 token）。所以路由前必须先切掉 `?` 之后的部分，否则 strcmp 全部落空 ——
         * 第一版就是这样：带 ?token= 的页面直接 404，而 header 方式却是 200，
         * 看起来像是「query 没被接受」，实际是路径没匹配上。
         * 原 path 保留给 token 提取用。 */
        char route[256];
        snprintf(route, sizeof(route), "%s", path);
        char *qmark = strchr(route, '?');
        if (qmark) *qmark = '\0';

        char presented[256] = "";
        extract_http_header_str(req_buf, "X-Debug-Token", presented, sizeof presented);
        /* 浏览器打不开一个需要自定义 header 的页面 —— 没有插件就带不上 X-Debug-Token。
         * 所以允许 ?token=。取舍：token 会进浏览器历史与本机日志，对回环/内网可接受；
         * 一旦要对外暴露这个页面，就必须改成 form POST 或会话 cookie。
         * header 优先，query 只在 header 为空时使用，避免两者冲突时判断不清。 */
        if (!presented[0]) {
            const char *q = strstr(path, "token=");
            if (q) {
                size_t n = strcspn(q + 6, "&");
                if (n >= sizeof(presented)) n = sizeof(presented) - 1;
                memcpy(presented, q + 6, n);
                presented[n] = '\0';
            }
        }
        /* 来源门槛：回环，或 MYDS_DEBUG_ALLOW_CIDR 列出的网段（默认只认回环）。
         * token 仍然必需 —— 白名单只决定「从哪来」，不决定「是不是自己人」。 */
        const bool loopback = myds_debug_ip_allowed(client_ip);
        const bool token_ok = g_debug_token_ok != NULL && g_debug_token_ok(presented);
        if (!(loopback && token_ok && g_debug_enabled && g_debug_enabled())) {
            MYDS_LOG_W("[%s:%d] GET %s -> 404 (loopback %s, token %s)",
                       client_ip, client_port, path,
                       loopback ? "yes" : "no",
                       presented[0] ? (token_ok ? "ok" : "mismatch") : "absent");
            send_response(client_fd, 404, "Not Found", "application/json",
                          "{\"error\":\"not found\"}");
            free(req_buf);
            close(client_fd);
            return;
        }
        if (strcmp(route, "/trace") == 0) {
            MYDS_LOG_I("[%s:%d] GET /trace -> 200 OK (page)", client_ip, client_port);
            /* token 注入进页面脚本。原因见 myds_trace_page_html_for_token：
             * 页面里的 fetch 不继承地址栏 query，少了注入就是「页面能开、
             * 表格永远空、每 3 秒一次 404」。 */
            char *page = myds_trace_page_html_for_token(presented, NULL);
            send_response(client_fd, 200, "OK", "text/html; charset=utf-8",
                          page ? page : myds_trace_page_html());
            free(page);
            free(req_buf);
            close(client_fd);
            return;
        }
        if (strcmp(route, "/trace/api/requests") == 0) {
            /* 列表可能很长：按容量上限（512 条）算，每条 ~300 B ⇒ 160 KB。
             * 堆上分配，失败就 500 —— 发一段截断的 JSON 比不发更难查。 */
            /* 每条最坏约 1546 B（26 个 prefill 桶 + 23 个 decode 桶全非零时实测）。
             * 原先按 512 B/条估：128 条实际要 197,821 B，而旧公式只给 135,168 B
             * ⇒ list_json 返回 false ⇒ 整个接口 500 ⇒ 页面一行都不显示，
             * 还把原因写成「token 可能已失效」。也就是说，按文档把
             * PHASE_SPLIT_MOE/ATTN 打开、记录攒满之后，页面会毫无征兆地全空。
             * 按 2048 B/条留余量。 */
            const size_t cap = myds_trace_list_json_max_bytes(myds_trace_capacity());
            char *json = (char *)malloc(cap);
            if (!json || !myds_trace_list_json(json, cap)) {
                free(json);
                send_response(client_fd, 500, "Internal Server Error", "application/json",
                              "{\"error\":\"trace buffer too small or unavailable\"}");
            } else {
                send_response(client_fd, 200, "OK", "application/json", json);
                free(json);
            }
            free(req_buf);
            close(client_fd);
            return;
        }
        /* 注意 "/trace/api/requests" 也以 "/trace/api/request" 开头，
         * 靠上面的分支先返回才没出错。要求前缀之后就是结尾，去掉这个隐含顺序依赖。 */
        if (strcmp(route, "/trace/api/system") == 0) {
            char sys[512];
            if (!myds_sysstat_json(sys, sizeof(sys))) {
                send_response(client_fd, 500, "Internal Server Error", "application/json",
                              "{\"error\":\"system stats unavailable\"}");
            } else {
                send_response(client_fd, 200, "OK", "application/json", sys);
            }
            free(req_buf);
            close(client_fd);
            return;
        }
        if (strncmp(route, "/trace/api/request", 18) == 0 && route[18] == '\0') {
            /* ?seq=N。没有 seq 或解析失败一律 400 —— 静默给个默认值会让客户端
             * 以为自己在看请求 1，而实际看的是别的。 */
            /* seq 必须从【原始 path】取，不能从 route —— route 在路由前已经被
             * 切掉了 '?' 之后的部分，那里正是 query。第一版写成 route，
             * 结果详情接口永远回 "missing or invalid seq"。 */
            const char *q = strstr(path, "seq=");
            char *end = NULL;
            const unsigned long long seq = q ? strtoull(q + 4, &end, 10) : 0ull;
            if (!q || end == q + 4) {
                send_response(client_fd, 400, "Bad Request", "application/json",
                              "{\"error\":\"missing or invalid seq\"}");
                free(req_buf);
                close(client_fd);
                return;
            }
            const size_t cap = 4096u;
            char *json = (char *)malloc(cap);
            if (!json || !myds_trace_detail_json(seq, json, cap)) {
                free(json);
                send_response(client_fd, 404, "Not Found", "application/json",
                              "{\"error\":\"no such request (evicted or never seen)\"}");
            } else {
                send_response(client_fd, 200, "OK", "application/json", json);
                free(json);
            }
            free(req_buf);
            close(client_fd);
            return;
        }
        send_response(client_fd, 404, "Not Found", "application/json",
                      "{\"error\":\"not found\"}");
        free(req_buf);
        close(client_fd);
        return;
    }

    /* GET /v1/models */
    if (strcmp(path, "/v1/models") == 0) {
        const bool vis = myds_vjob_ready();
        MYDS_LOG_I("[%s:%d] GET /v1/models -> 200 OK (alias: %s, vision: %d)",
                   client_ip, client_port, model_alias ? model_alias : "myds", vis ? 1 : 0);
        char resp[1024];
        if (!myds_server_format_models_json(model_alias, vis, resp, sizeof(resp))) {
            send_response(client_fd, 500, "Internal Server Error", "application/json",
                          "{\"error\":\"failed to format model list\"}");
        } else {
            send_response(client_fd, 200, "OK", "application/json", resp);
        }
        free(req_buf);
        close(client_fd);
        return;
    }

    /* POST /v1/chat/completions or /v1/responses */
    int is_chat_endpoint = (strcmp(path, "/v1/chat/completions") == 0 || strcmp(path, "/chat/completions") == 0);
    int is_responses_endpoint = (strcmp(path, "/v1/responses") == 0 || strcmp(path, "/responses") == 0);

    if (is_chat_endpoint || is_responses_endpoint) {
        double t_start = myds_server_now_s();
        /* 请求到达时刻（解析前）。与 t_start 分开保留：t_start 会在拿到推理名额后
         * 重置成「开始推理」的时刻，而 wall 必须仍是【端到端】——外部监控和压测
         * 工具拿它跟客户端实测比。两者混用会让并发排队看起来像引擎变慢。 */
        const double t_arrive = t_start;
        double t_first = 0.0;
        double t_last = 0.0;
        /* Stage boundaries for MYDS_REQUEST_PHASES. All zero when the switch is
         * off, so the emission code can detect "not measured" instead of
         * reporting a plausible-looking 0.0 s. */
        const bool phases = myds_server_request_phases_enabled();
        double t_render_begin = 0.0, t_render_end = 0.0, t_generate_begin = 0.0;

        /* Find body start and read complete body */
        char *body = strstr(req_buf, "\r\n\r\n");
        if (body) {
            body += 4;
            const int content_len = extract_http_header_int(req_buf, "Content-Length");
            if (content_len < 0) {
                /* 2026-10-25: 负值（或溢出的非数字）以前会被当成 0，
                 * 继续走到“Incomplete body”分支——不影响安全，但报错会误导。 */
                send_response(client_fd, 400, "Bad Request", "application/json",
                              "{\"error\":{\"message\":\"Invalid Content-Length\"}}");
                free(req_buf);
                close(client_fd);
                return;
            }
            /* 2026-10-25: 请求体上限。
             *
             * 原实现直接按 Content-Length 算出 needed 并 realloc，
             * 于是任意客户端发一条 `Content-Length: 10000000000` 就能要求
             * 分配 10 GB —— 一个连接即可把实例打挂。公开仓库会迎来各种扫描与本机压测。
             *
             * 上限取 max_seq_len 的字节上界的 4 倍（默认 ctx 327680 token、
             * UTF-8 最多 4 字节/token ⇒ 1.3 MB，宽松 4 倍已足够容纳图片 base64 的
             * 一部分）再加上固定的图片编码余量；同时硬上限 256 MB 防止
             * ctx 被配得极大时把内存打穿。可用 MYDS_MAX_BODY_MB 调整。
             */
            size_t body_limit = 8u * 1024u * 1024u;
            {
                const char *e = getenv("MYDS_MAX_BODY_MB");
                if (e && *e) {
                    const long v = atol(e);
                    if (v >= 1 && v <= 1024) body_limit = (size_t)v * 1024u * 1024u;
                    else MYDS_LOG_W("MYDS_MAX_BODY_MB=%s 非法（需 1..1024），按 8 MB 处理", e);
                }
            }
            if ((size_t)content_len > body_limit) {
                MYDS_LOG_W("[%s:%d] 请求体 %d 字节超过上限 %zu ⇒ 413",
                           client_ip, client_port, content_len, body_limit);
                send_response(client_fd, 413, "Payload Too Large", "application/json",
                              "{\"error\":{\"message\":\"Request body too large\"}}");
                free(req_buf);
                close(client_fd);
                return;
            }
            const size_t header_len = (size_t)(body - req_buf);
            const size_t needed = header_len + (size_t)content_len + 1;
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
                if (RECV_DEADLINE_HIT()) {
                    MYDS_LOG_W("[%s:%d] 接收 body 超时（整体 deadline %.0fs）⇒ 408",
                               client_ip, client_port, recv_deadline_s);
                    send_response(client_fd, 408, "Request Timeout", "application/json",
                                  "{\"error\":{\"message\":\"Request body not received in time\"}}");
                    free(req_buf);
                    close(client_fd);
                    return;
                }
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
        /* 2026-10-24：min_p / repetition_penalty 在 myds_config_t 里【一直存在且真的参与
         * 采样】（engine.c 的 min_p 阈值与 rep_pen 乘子），但此前没有任何入口能给它赋值：
         * main.c 没有对应 CLI，请求体也不解析 ⇒ 恒为默认 0.0 / 1.0（等于永久禁用）。
         * 这里补上 per-request 入口。默认 -1 表示「本次未指定」，不覆盖启动配置。 */
        float min_p = json_extract_float_field(body, "min_p", -1.0f);
        float rep_pen = json_extract_float_field(body, "repetition_penalty", -1.0f);
        float pres_pen = json_extract_float_field(body, "presence_penalty", -1.0f);
        char req_steer_name[64] = {0};
        json_extract_string_field(body, "steer", req_steer_name, sizeof(req_steer_name));
        if (req_steer_name[0] == '\0') {
            json_extract_string_field(body, "steer_mode", req_steer_name, sizeof(req_steer_name));
        }
        float req_steer_alpha = json_extract_float_field(body, "steer_alpha", -1.0f);
        int req_steer_enable = json_extract_int_field(body, "steer_enable", -1);
        if (req_steer_enable < 0) {
            if (strstr(body, "\"steer_enable\":true") || strstr(body, "\"steer_enable\": true")) req_steer_enable = 1;
            else if (strstr(body, "\"steer_enable\":false") || strstr(body, "\"steer_enable\": false")) req_steer_enable = 0;
        }
        int include_usage = (strstr(body, "\"include_usage\":true") != NULL || strstr(body, "\"include_usage\": true") != NULL);
        /* Structured output response_format：本引擎【没有】约束解码器。
         * 2026-10-25：此前这里只把 json_object/json_schema 记成一个布尔并打印日志，
         * 客户端以为输出受语法约束，实际拿到的是自由文本 —— 这是静默说谎。
         * 现在：json_schema 自带语法合约，直接 400 拒绝；json_object 保留接受
         * （不少客户端只是「希望」JSON 而自带重试），但显式告警。 */
        const int rf_kind = server_response_format_kind(body);
        if (rf_kind == SERVER_RF_JSON_SCHEMA) {
            MYDS_LOG_W("[%s:%d] 拒绝 response_format=json_schema：无约束解码，语法会被忽略",
                       client_ip, client_port);
            send_response(client_fd, 400, "Bad Request", "application/json",
                          "{\"error\":{\"message\":\"response_format.json_schema is not supported: "
                          "this engine has no constrained decoding, so the grammar would be silently ignored\"}}");
            free(req_buf);
            close(client_fd);
            return;
        }
        const bool req_json_mode = (rf_kind == SERVER_RF_JSON_OBJECT);
        if (req_json_mode) {
            MYDS_LOG_W("[%s:%d] response_format=json_object 未强制：无约束解码，输出不保证是合法 JSON",
                       client_ip, client_port);
        }

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
        /* 请求一到达就登记一条 in_flight 记录，结束时再回填。
         * 只写结尾的话，一次 40 秒的 prefill 在页面里完全不可见 ——
         * 而「现在卡在哪」恰恰只能看正在进行的那条。 */
        /* ★ 只看 header 不够：这个头会清 ctx->pcache_len（下一个请求丢全部前缀缓存），
         * 且本次请求不进 trace。服务默认监听 0.0.0.0，外网一个 20 token 的请求
         * 就能让别人的 40 s prefill 重跑一遍，而且页面上看不到这一次 —— 正好是
         * 有人来查「我的请求为什么变慢」时会看的地方。与 /trace 同一套来源判定。 */
        const bool warmup_hdr = myds_server_is_warmup_request(req_buf);
        const bool is_warmup_probe = warmup_hdr && myds_debug_ip_allowed(client_ip);
        if (warmup_hdr && !is_warmup_probe)
            MYDS_LOG_W("[%s:%d] X-Myds-Warmup 被忽略：来源不在白名单（该头会清前缀缓存）",
                       client_ip, client_port);
        char trace_client[MYDS_TRACE_CLIENT_MAX];
        snprintf(trace_client, sizeof(trace_client), "%s:%d", client_ip, client_port);
        const uint64_t trace_seq = is_warmup_probe ? 0 : myds_trace_begin(path, trace_client, resp_id);
        /* 早退时用它收尾：清掉 in_flight，阶段全部保持「未测量」。
         * 不做这一步的话，被拒绝的请求会永远挂在页面上显示「进行中」。 */
        myds_trace_rec_t trec_fail;
        memset(&trec_fail, 0, sizeof(trec_fail));
        snprintf(trec_fail.path, sizeof(trec_fail.path), "%s", path);
        snprintf(trec_fail.client, sizeof(trec_fail.client), "%s", trace_client);
        snprintf(trec_fail.resp_id, sizeof(trec_fail.resp_id), "%s", resp_id);
        snprintf(trec_fail.finish, sizeof(trec_fail.finish), "rejected");
        /* 一个函数搞定 phases / 速率 / detail —— 原先手写这两行，漏了 detail，
         * 于是被拒请求的 prefill_gpu 是 0.000000，页面显示成「<0.1us」。 */
        myds_trace_rec_mark_unmeasured(&trec_fail);
        trec_fail.stream = stream ? 1 : 0;
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

        int q_res = server_acquire_infer_slot_for(q_timeout_ms, max_q_depth, prev_id);
        if (q_res != 0) {
            MYDS_LOG_W("[%s:%d] server reject due to %s (depth=%u, timeout=%dms)",
                       client_ip, client_port, q_res == -1 ? "queue full" : "queue wait timeout",
                       server_infer_queue_depth(), q_timeout_ms);
            const char *err_json = (q_res == -1)
                ? "{\"error\":{\"message\":\"Server is at capacity (queue full); please retry later\",\"type\":\"server_error\",\"code\":\"server_busy\"}}"
                : "{\"error\":{\"message\":\"Request timed out waiting in inference queue; please retry later\",\"type\":\"server_error\",\"code\":\"queue_timeout\"}}";
            send_response(client_fd, 503, "Service Unavailable", "application/json", err_json);
            myds_vjob_sink_free(&img_sink);
            myds_trace_complete(trace_seq, &trec_fail);
            free(prompt); free(req_buf); close(client_fd); return;
        }

        pthread_mutex_lock(&g_infer_mutex);
        t_infer_slot_depth = 1;
        /* ★ 排队等待必须单独计量。
         * t_start 在请求开头（解析前）就记了，而这里可能已经等了很久；
         * 旧实现把 queue 时间算进 wall/prefill，于是“并发时变慢”会被误认为
         * 引擎退化（实测 2026-10-25: 一个 34.7s 的“失败请求”里 prefill 数取自上一个
         * 请求，wall 全是等锁）。现在从拿到锁开始重新计时。 */
        const double t_slot = myds_server_now_s();
        const double queue_s = (t_slot > t_start) ? (t_slot - t_start) : 0.0;
        t_start = t_slot;        /* ★ 先作废上一次的 prefill 统计。
         * 下面可能因 cancel_flag 短路而根本不调 generate，那时引擎内的重置也不会跑；
         * 不作废就会把【上一个请求】的 token 数当成这个请求的打进日志。 */
        myds_context_reset_prefill_stats(ctx);
        /* 同一理由，分阶段归因也必须作废：短路路径不跑 generate，
         * 引擎内的重置不会执行，不在此清就会把上一请求的 gpu/wait 报出去。 */
        myds_context_reset_phase_detail(ctx);
        myds_config_t prev_cfg = myds_context_get_config(ctx);
        uint64_t dspark_prop_before = ctx->dspark ? ctx->dspark->proposed_tokens : 0;
        uint64_t dspark_acc_before  = ctx->dspark ? ctx->dspark->accepted_tokens : 0;
        uint64_t dspark_cyc_before  = ctx->dspark ? ctx->dspark->cycles : 0;
        double   dspark_prop_ms_before = ctx->dspark ? ctx->dspark->propose_ms : 0.0;
        double   dspark_ver_ms_before  = ctx->dspark ? ctx->dspark->verify_ms : 0.0;

        /* ★ 2026-11-01：出队即验。
         *
         * 排队可能等几十秒。客户端在这段时间里 abort / 超时重连是常态（agent 的
         * AbortSignal 尤其频繁）。原实现拿到锁之后不做任何存活确认就直接分词 +
         * prefill —— 一个已经没人等的请求照样吃掉几秒到几分钟的 GPU。
         * 这里在花任何算力之前先问一次；探针注册给引擎，prefill 期间继续复查。 */
        if (client_disconnected(client_fd)) {
            MYDS_LOG_I("[%s:%d] client gone while queued (waited %.1fs) -> dropped before any GPU work",
                       client_ip, client_port, queue_s);
            t_infer_slot_depth = 0;
            pthread_mutex_unlock(&g_infer_mutex);
            server_release_infer_slot();
            myds_vjob_sink_free(&img_sink);
            myds_trace_complete(trace_seq, &trec_fail);
            free(prompt); free(req_buf); close(client_fd); return;
        }
        /* 探针状态放在静态区：推理由 g_infer_mutex 串行化，同一时刻只有一个请求在跑，
         * 所以一个槽就够；放栈上则会在函数返回后悬空（探针可能被引擎缓存）。 */
        g_prefill_probe_fd = client_fd;
        myds_context_set_liveness_probe(ctx, server_client_alive_probe, NULL);

        /* 2026-10-24：重启后 pcache 为空，但若该 id 的【完整累积提示词】已持久化，
         * 就不必杀请求 —— 用「持久化前缀 + 本次 suffix」重建出客户端原本会发送的完整输入，
         * 退回普通 prefill。正确性优先；若 kvdisk 里还有对应前缀的 KV，普通路径会自己命中。 */
        /* ★ 2026-11-01：单活 → 多活。
         *
         * 原实现只承认【最近一次】响应，更早的一律 400。多会话 agent 交错调用时
         * 这是必然失败而不是偶发：A 提问 → B 提问 → A 续跑，A 手里的 id 在这一刻
         * 就已经 expired，哪怕它三秒前才拿到。
         *
         * 现在两条路：
         *   快路径   id 就是最近一次、且 pcache 还活着 ⇒ 零拷贝续跑（原行为不变）
         *   恢复路径 id 仍在表里（但不一定最新）⇒ 用持久化的累积文本 + 本次 suffix
         *            重建客户端原本会发送的完整输入，走普通全量 prefill。
         *            正确性优先，速度次之；前缀缓存与锚点仍会自动命中能命中的部分。
         * 只有【从未见过】的 id 才 400 —— 那种情况下服务端确实没有任何依据。 */
        bool resp_recover = false;
        const bool live_fast = is_continuation && resp_id_is_live(prev_id) &&
                               myds_context_get_pcache_len(ctx) > 0;
        if (is_continuation && !live_fast) {
            char *saved = NULL;
            const size_t saved_len = resp_id_saved_text_copy(prev_id, &saved);
            if (saved_len > 0) {
                resp_recover = true;
                MYDS_LOG_I("[%s:%d] previous_response_id '%s' %s; reconstructing full prompt "
                           "from persisted prefix (%zu chars) -> full prefill",
                           client_ip, client_port, prev_id,
                           resp_id_is_live(prev_id) ? "state is cold (pcache empty)"
                                                    : "is not the most recent response",
                           saved_len);
            }
            free(saved);
        }
        if (is_continuation && !resp_recover && !live_fast) {
            t_infer_slot_depth = 0;
            pthread_mutex_unlock(&g_infer_mutex);
            server_release_infer_slot();
            MYDS_LOG_W("[%s:%d] previous_response_id rejected (unknown): %s",
                       client_ip, client_port, prev_id);
            char err[512];
            snprintf(err, sizeof(err),
                     "{\"error\":{\"message\":\"previous_response_id '%s' is unknown to this server "
                     "(no stored prefix for it, and it is not the current one); replay the full input "
                     "instead\",\"type\":\"invalid_request_error\","
                     "\"code\":\"unknown_previous_response_id\"}}", prev_id);
            send_response(client_fd, 400, "Bad Request", "application/json", err);
            myds_vjob_sink_free(&img_sink);
            myds_trace_complete(trace_seq, &trec_fail);
            free(prompt); free(req_buf); close(client_fd); return;
        }

        /* ★ 续跑时工具集必须一致。
         *
         * suffix 模式只渲染新增消息，tools 段【不重渲】（它已在前缀里）。旧实现遇到
         * body 里带 tools 就静默 goto 跳过 —— 若客户端在续跑时改了工具集，模型仍拿着
         * 旧工具列表在跑，却看不到任何报错。这里改为显式校验：
         *   · 未带 tools ⇒ 沿用前缀里的（正常情况）
         *   · 带了且与前缀指纹相同 ⇒ 放行（客户端重复声明未变，无副作用）
         *   · 带了但与指纹不同 ⇒ 400 明确报错，让调用方开启新对话
         * 仍然不渲染，只做一致性判定。 */
        const unsigned long long tools_sig = server_tools_signature(body, prompt_cap);
        /* ★ 2026-11-01：比对基准必须是【祖先 id】的工具指纹，不是「最近一次」的。
         * 多活之后 prev_id 很可能不是最新那条，拿最新的指纹比会误报/漏报。 */
        if (is_continuation && tools_sig != 0ull && resp_id_tools_sig(prev_id) != tools_sig) {
            t_infer_slot_depth = 0;
            pthread_mutex_unlock(&g_infer_mutex);
            server_release_infer_slot();
            MYDS_LOG_W("[%s:%d] previous_response_id rejected (tools changed): %s",
                       client_ip, client_port, prev_id);
            send_response(client_fd, 400, "Bad Request", "application/json",
                          "{\"error\":{\"message\":\"tools/functions differ from the previous_response_id "
                          "conversation: a continuation cannot change the tool set, because the prefix "
                          "already holds the old schemas. Start a new conversation without "
                          "previous_response_id to change tools\",\"type\":\"invalid_request_error\","
                          "\"code\":\"tools_changed_on_continuation\"}}");
            myds_vjob_sink_free(&img_sink);
            myds_trace_complete(trace_seq, &trec_fail);
            free(prompt); free(req_buf); close(client_fd); return;
        }

        const char *parse_reason = NULL;
        /* 时间戳无条件采集：/trace 页面读它们。开关只控制「打不打日志行」。 */
        t_render_begin = myds_server_now_s();
        const bool parsed_ok = is_continuation
            ? parse_chat_prompt_suffix_vis(body, prompt, prompt_cap, &parse_reason, &img_sink)
            : parse_chat_prompt_vis(body, prompt, prompt_cap, &parse_reason, &img_sink);
        t_render_end = myds_server_now_s();
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


        /* 重建：把持久化前缀拼到本次新增部分之前。解析 suffix 时已跳过 BOS/instructions/tools，
         * 所以直接前缀拼接即可复现客户端原本会发送的完整输入。 */
        if (resp_recover) {
            char *saved = NULL;
            (void)resp_id_saved_text_copy(prev_id, &saved);
            if (saved && saved[0]) {
                const size_t plen = strlen(saved), slen = strlen(prompt), need = plen + slen + 1u;
                char *full = (char *)malloc(need);
                if (!full) {
                    t_infer_slot_depth = 0;
                    pthread_mutex_unlock(&g_infer_mutex);
                    server_release_infer_slot();
                    send_response(client_fd, 500, "Internal Server Error", "application/json",
                                  "{\"error\":{\"message\":\"Out of memory rebuilding continuation\",\"type\":\"server_error\"}}");
                    myds_vjob_sink_free(&img_sink);
                    myds_trace_complete(trace_seq, &trec_fail);
                    free(prompt); free(req_buf); close(client_fd); return;
                }
                memcpy(full, saved, plen);
                memcpy(full + plen, prompt, slen + 1u);
                free(prompt);
                prompt = full;
                prompt_cap = need;
            }
            free(saved);
        }
        const bool req_has_instructions = json_has_top_field(body, "instructions");
        const bool req_has_tools = json_has_top_field(body, "tools");
        /* 2026-10-24：把 min_p / rep_pen / pres_pen 也打出来。
         * 此前只打 temp/top_p/top_k，另外三个即使客户端发了也无从确认 ——
         * 「客户端以为设了、服务端没收到/没应用」正是最难查的一类。 */
        MYDS_LOG_I("[%s:%d] %s %s (protocol=%s, stream=%s, max_tokens=%d, temp=%.2f, top_p=%.2f, "
                   "top_k=%d, min_p=%.3f, rep_pen=%.3f, pres_pen=%.3f, "
                   "prompt_chars=%zu, body=%zd, has_instructions=%d, has_tools=%d)",
                   client_ip, client_port, method, path,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   stream ? "true" : "false",
                   max_tokens, temp, top_p, top_k,
                   (double)min_p, (double)rep_pen, (double)pres_pen,
                   strlen(prompt), total_read, req_has_instructions ? 1 : 0, req_has_tools ? 1 : 0);

        /* 恢复模式下 prefix cache 是空的（重启所致），必须走【普通全量 prefill】：
         * 不能置 continue_from_cache，否则引擎会因 pcache_len==0 直接报错。 */
        if (is_continuation && !resp_recover) myds_set_continue_from_cache(ctx, true);

        /* Apply request configuration overrides if provided */
        myds_config_t req_cfg = prev_cfg;
        /* 本次请求是否思考模式：后面「temperature 在思考模式下无效」要用到，
         * 所以在覆盖块之前先算出来（只算一次，避免两处判定漂移）。 */
        const bool req_thinking_now =
            extract_request_thinking_mode(body, g_server_default_thinking);
        {
            char eff[32] = "";
            if (!json_extract_string_field(body, "reasoning_effort", eff, sizeof(eff)))
                (void)extract_reasoning_object_effort(body, eff, sizeof(eff));
            req_cfg.max_new_tokens = myds_server_default_max_tokens(prev_cfg.max_new_tokens,
                req_thinking_now, eff);
        }
        if (max_tokens > 0) req_cfg.max_new_tokens = (uint32_t)max_tokens;

        /* ★ DeepSeek thinking_mode 文档（2026-10）明确：
         *   「思考模式不支持 temperature、presence_penalty、frequency_penalty 参数。
         *     为了兼容已有软件，设置参数不会报错，但也不会生效。」
         * 之前的实现无条件应用 temperature，于是思考模式下温度和官方行为不一致：
         * 客户端以为设了 0.3，实际采样并不受它约束。
         * 这里对齐官方：思考模式下忽略 temperature，并留一条 WARN 让人看得见——
         * 静默忽略正是最难发现的那类问题。
         * top_p / top_k 不在禁用名单内（文档还专门说明 top_p 在思考模式下的
         * 模型相关行为），继续生效。 */
        /* ★ 官方规范：「思考模式不支持 temperature …… 设置参数不会报错，但也不会生效。」
         * 语义是「不受它约束」，因此这里【固定成 1.0（标准分布、不缩放）】，
         * 而不是「保留上一轮的旧值」——后者会让启动参数里的 0.7 意外主导思考采样。
         * 仅当请求确实传了非 1.0 的值时才告警，避免官方推荐值（1.0）也刷日志。 */
        if (req_thinking_now) {
            if (temp >= 0.0f && temp != 1.0f) {
                MYDS_LOG_W("[%s:%d] thinking 模式不支持 temperature（官方规范），"
                           "请求中的 temp=%.2f 不生效，按 1.0 采样", client_ip, client_port, temp);
            }
            req_cfg.temperature = 1.0f;
        } else if (temp >= 0.0f) {
            req_cfg.temperature = temp;
        }
        if (top_p >= 0.0f) req_cfg.top_p = top_p;
        if (top_k >= 0) req_cfg.top_k = (uint32_t)top_k;
        /* min_p 语义：0.0 = 关闭；有效范围 (0,1]。超范围的值拒绝而不静默夹取，
         * 否则客户端以为设了 5%，实际按别的值采样。 */
        if (min_p >= 0.0f) {
            if (min_p <= 1.0f) {
                req_cfg.min_p = min_p;
            } else {
                MYDS_LOG_W("[%s:%d] min_p=%.3f 超出 [0,1]，忽略该字段", client_ip, client_port, min_p);
            }
        }
        /* presence_penalty 语义：0.0 = 关闭；OpenAI 取值范围 [-2, 2]。
         * 正值抑制已出现的 token，负值鼓励。 */
        if (pres_pen != -1.0f) {
            if (pres_pen >= -2.0f && pres_pen <= 2.0f) {
                req_cfg.presence_penalty = pres_pen;
            } else {
                MYDS_LOG_W("[%s:%d] presence_penalty=%.3f 超出 [-2,2]，忽略该字段",
                           client_ip, client_port, pres_pen);
            }
        }
        /* repetition_penalty 语义：1.0 = 关闭；常用范围 (1.0, 2.0]。<=0 视为非法。 */
        if (rep_pen >= 0.0f) {
            if (rep_pen == 0.0f || (rep_pen >= 1.0f && rep_pen <= 4.0f)) {
                req_cfg.repetition_penalty = rep_pen;
            } else {
                MYDS_LOG_W("[%s:%d] repetition_penalty=%.3f 非法（需 0 或 [1,4]），忽略该字段",
                           client_ip, client_port, rep_pen);
            }
        }
        myds_context_set_config(ctx, &req_cfg);
        /* 2026-10-24：打印【生效值】而不是请求里的原始值。
         * 两者不同正是「配了但没生效」的来源：原始值可能是 -1（未提供），
         * 也可能被范围校验拒绝，而行首那条日志只反映字面量。
         * 这一行紧跟在 set_config 之后，内容就是采样器实际会读到的 cfg。 */
        MYDS_LOG_I("[%s:%d] effective sampling: temp=%.3f, top_p=%.3f, top_k=%u, "
                   "min_p=%.3f, rep_pen=%.3f, pres_pen=%.3f, thinking=%d%s",
                   client_ip, client_port,
                   (double)req_cfg.temperature, (double)req_cfg.top_p, req_cfg.top_k,
                   (double)req_cfg.min_p, (double)req_cfg.repetition_penalty,
                   (double)req_cfg.presence_penalty, req_thinking_now ? 1 : 0,
                   req_json_mode ? ", response_format=json" : "");

        myds_stop_reason_t stop_reason = MYDS_STOP_EOS;
        uint32_t prompt_tokens = 0;
        uint32_t final_tokens = 0;

        myds_steer_t *prev_steer = ctx ? ctx->steer : NULL;
        float prev_steer_alpha = -1.0f;
        int prev_steer_enabled = -1;
        if (ctx) {
            if (req_steer_name[0] != '\0') {
                if (strcasecmp(req_steer_name, "none") == 0 ||
                    strcasecmp(req_steer_name, "off") == 0 ||
                    strcasecmp(req_steer_name, "false") == 0) {
                    req_steer_enable = 0;
                } else {
                    myds_steer_t *found = myds_context_find_steer(ctx, req_steer_name);
                    if (found) {
                        ctx->steer = found;
                        if (req_steer_enable < 0) req_steer_enable = 1;
                    } else {
                        MYDS_LOG_W("[%s:%d] steer profile '%s' not found", client_ip, client_port, req_steer_name);
                    }
                }
            }

            if (ctx->steer) {
                prev_steer_alpha = ctx->steer->alpha_effective;
                prev_steer_enabled = ctx->steer->enabled ? 1 : 0;
                if (req_steer_alpha >= 0.0f) {
                    ctx->steer->alpha_effective = req_steer_alpha;
                }
                if (req_steer_enable >= 0) {
                    ctx->steer->enabled = (req_steer_enable != 0);
                }
                if (req_steer_alpha >= 0.0f || req_steer_enable >= 0 || req_steer_name[0] != '\0') {
                    MYDS_LOG_I("[%s:%d] dynamic steer override: profile=%s, alpha=%.3f, enabled=%d",
                               client_ip, client_port, ctx->steer->name, ctx->steer->alpha_effective, ctx->steer->enabled ? 1 : 0);
                }
            }
        }

        /* 视觉注入：提示词已拼好，把所有图片块的逐位置向量交给引擎。
         * ★ 必须在 generate 之前；在发送 200 响应头之前完成，避免注入失败时向客户端连发 200 + 400 */
        if (!inject_image_embeddings(ctx, prompt, &img_sink, client_ip, client_port)) {
            resp_id_invalidate_all();
            send_response(client_fd, 400, "Bad Request", "application/json",
                          "{\"error\":{\"message\":\"image block could not be aligned with the prompt\"}}");
            goto cleanup_infer;
        }

        bool transport_ok = true;
        /* 回调写出累计。两条分支各写各的，出一分支后仍要能读 —— 所以放在外层。
         * 初始 -1 = 未测量；只有真的生成过 token 的分支才改写。 */
        double emit_measured_s = -1.0;
        const bool req_thinking = extract_request_thinking_mode(body, g_server_default_thinking);
        /* Everything from here to the engine call is setup: config application,
         * image-embedding injection, response-header construction. E1 needs this
         * split because the short-prompt intercept may not be prefill at all.
         * The setup interval is t_render_end -> t_generate_begin, so no separate
         * timestamp is needed here. */
        if (stream) {
            /* 2026-10-25: SSE 也要尊重 CORS 设置。origin 为空时整行省略。 */
            char sse_header_buf[768];
            const char *origin_sse = cors_origin_header();
            if (origin_sse) {
                snprintf(sse_header_buf, sizeof(sse_header_buf),
                         "HTTP/1.1 200 OK\r\n"
                         "Content-Type: text/event-stream\r\n"
                         "Cache-Control: no-cache\r\n"
                         "Access-Control-Allow-Origin: %s\r\n"
                         "Connection: close\r\n\r\n", origin_sse);
            } else {
                snprintf(sse_header_buf, sizeof(sse_header_buf),
                         "HTTP/1.1 200 OK\r\n"
                         "Content-Type: text/event-stream\r\n"
                         "Cache-Control: no-cache\r\n"
                         "Connection: close\r\n\r\n");
            }
            const char *sse_header = sse_header_buf;
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
            t_generate_begin = myds_server_now_s();
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
            emit_measured_s = sctx.emit_s;
            /* ★ 客户端断连不是服务端错误。
             * 2026-10-25：旧实现只看 !generated || sctx.failed，于是「对端在首字节之前
             * 就断开」被记成 finish_reason=error。而 generated 的短路求值
             * （!sctx.cancel_flag && myds_generate...）意味着 cancel 时 generate 压根没跑，
             * prompt_tokens 仍为 0。两个现象叠在一起就产生了那条矛盾的日志。
             * cancel_flag 单独成案：标 cancelled，不标 error。 */
            if (!generated || sctx.failed) {
                resp_id_invalidate_all();
                stop_reason = sctx.cancel_flag ? MYDS_STOP_CANCELLED : MYDS_STOP_ERROR;
                if (stop_reason == MYDS_STOP_ERROR) {
                    const char *error_event = is_responses_endpoint
                        ? "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\n"
                        : "data: {\"error\":{\"message\":\"Inference or response encoding failed\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n";
                    if (sctx.writer) (void)myds_server_sse_send(&sctx, error_event, strlen(error_event));
                }
            } else if (!sctx.cancel_flag) {
                /* 模型可能用 DSML 块发起工具调用 ⇒ 从累积文本里解析（与协议无关） */
                size_t scontent_len = sctx.acc_len;
                myds_tool_call_t *stool = NULL;
                size_t stool_count = 0;
                /* 有工具块但没能解析出来 ⇒ 输出是不完整的，finish_reason 必须反映这一点。 */
                bool dsml_block_dropped = false;
                /* ★ 2026-11-01：区分「没有工具块」和「工具块坏了」。
                 * 原实现两者都走同一个 fallback（整段当正文），于是在被 max_tokens
                 * 截断时把半截 `<|DSML| calls><|DSML| invoke ...` 原样当正文发给客户端。
                 * 现在坏块【隔离】：正文只到开标签之前，块本身丢弃并记日志。 */
                const myds_dsml_state_t dsml_state =
                    sctx.acc ? myds_dsml_classify(sctx.acc, sctx.acc_len, &scontent_len)
                             : MYDS_DSML_NONE;
                if (dsml_state == MYDS_DSML_NONE ||
                    (dsml_state != MYDS_DSML_NONE && sctx.acc &&
                     myds_parse_completion_tool_calls(sctx.acc, sctx.acc_len,
                                                      &scontent_len, &stool, &stool_count))) {
                    /* 正常：要么没有工具块，要么解析成功（stool 可能为空） */
                } else {
                    stool = NULL; stool_count = 0;
                    dsml_block_dropped = true;
                    MYDS_LOG_W("[%s:%d] %s DSML tool block discarded (%zu chars withheld from content)",
                               client_ip, client_port,
                               dsml_state == MYDS_DSML_TRUNCATED ? "truncated" : "malformed",
                               sctx.acc_len - scontent_len);
                }
                if (!stool_count && dsml_state == MYDS_DSML_NONE && sctx.acc &&
                    dsml_looks_like_foreign_dialect(sctx.acc, sctx.acc_len)) {
                    MYDS_LOG_W("[%s:%d] tool-call markup found but this request's dialect is not "
                               "|DSML|calls; it will be sent as plain content. The model emitted "
                               "<function_calls>/<invoke> instead of the requested |DSML| form.",
                               client_ip, client_port);
                }
                if (!stool_count && sctx.acc && sctx.emitted < scontent_len) {
                    /* 只补发【正文】部分（emitted 之后、工具块之前）。绝不补发工具块本身。 */
                    bool replayed = sse_emit_content(&sctx, sctx.acc + sctx.emitted,
                                                     scontent_len - sctx.emitted, false);
                    sctx.emitted = scontent_len;
                    if (!replayed || sctx.failed || sctx.cancel_flag) {
                        stop_reason = sctx.failed ? MYDS_STOP_ERROR : MYDS_STOP_CANCELLED;
                    }
                }
                /* 流式分支也要算一份 timings：response.completed 里要带它，
                 * mycode 从 response.timings 读。此时生成已结束，t_end 取当前时刻。 */
                myds_timings_t tim;
                server_timings_compute(ctx, final_tokens, t_first, myds_server_now_s(),
                                       (stop_reason == MYDS_STOP_EOS), &tim);
                char top_timings[384];
                server_timings_json(&tim, top_timings, sizeof(top_timings));
                /* Responses 的 usage 明细。用 input_tokens_details 是 OpenAI
                 * Responses 的标准位置；mycode 与官方 SDK 都认它。 */
                char usage_det[96];
                snprintf(usage_det, sizeof(usage_det),
                         "\"input_tokens_details\":{\"cached_tokens\":%u},"
                         "\"prompt_tokens_details\":{\"cached_tokens\":%u}",
                         tim.cached, tim.cached);

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
                            &sctx, model_alias, resp_id, (unsigned)prompt_tokens,
                            (unsigned)reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS),
                            "", 0, items, 0, top_timings, usage_det);
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
                        &sctx, model_alias, resp_id, (unsigned)prompt_tokens,
                            (unsigned)reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS),
                        content_ptr, content_len, items, emit_text_events, top_timings,
                        usage_det);
                    free(items);
                    if (!finished) {
                        stop_reason = MYDS_STOP_ERROR;
                        const char *error_event = "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"Response allocation failed\",\"type\":\"server_error\"}}\n\n";
                        (void)myds_server_sse_send(&sctx, error_event, strlen(error_event));
                    }
                } else if (!stool_count && !sctx.failed && !sctx.cancel_flag) {
                    /* LOOP_GUARD 归入 "length"：输出确实被截断了，客户端必须能分辨。
                     * 映射成 "stop" 会让半截答案看起来像正常收尾（原实现的问题）。 */
                    /* 丢弃了残缺工具块 ⇒ 输出同样是不完整的，归入 "length"。
                     * 归 "stop" 会让客户端以为这是一次完整收尾，正是本次要修的那类谎报。 */
                    const char *finish = (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL ||
                                          stop_reason == MYDS_STOP_LOOP_GUARD ||
                                          dsml_block_dropped) ? "length" : "stop";
                    char chunk[512];
                    int len = snprintf(chunk, sizeof(chunk),
                        "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n", finish);
                    bool sent = len > 0 && (size_t)len < sizeof(chunk) && myds_server_sse_send(&sctx, chunk, (size_t)len);
                    if (sent && include_usage) {
                        /* 流式分支的这个 usage 块在 finish 事件之后才发，
                         * 此时 t_last 已定；t_end 取当前时刻（生成已经结束）。 */
                        /* 768 不是随手写的：这段 JSON 约 400 字符（扁平字段 + timings 九个键）。
                         * 原来 384，加了字段后被 snprintf 截断，末尾少一个括号，整个响应无法解析。
                         * 截断的 JSON 比拒绝更糟 —— 客户端只说“解析失败”，看不出是我们截的。 */
                        char usage_extra[768];
                        myds_timings_t tim;
                        server_timings_compute(ctx, final_tokens, t_first,
                                               myds_server_now_s(),
                                               (stop_reason == MYDS_STOP_EOS), &tim);
                        server_usage_extra_json(&tim, usage_extra, sizeof(usage_extra));
                        len = snprintf(chunk, sizeof(chunk),
                            "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u%s}\n\n",
                            prompt_tokens, reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), prompt_tokens + reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), usage_extra);
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
            t_generate_begin = myds_server_now_s();
            int generated = myds_generate_bytes_ex(ctx, prompt, myds_server_non_stream_callback, &ns,
                                                   &stop_reason, &prompt_tokens, &ns.cancel_flag);
            myds_context_clear_image_embeddings(ctx); myds_vjob_sink_free(&img_sink);
            if (!generated || ns.failed || ns.cancel_flag) {
                resp_id_invalidate_all();
            }
            final_tokens = ns.token_count;
            t_first = ns.t_first;
            t_last = ns.t_last;
            emit_measured_s = ns.emit_s;
            if (!generated || ns.failed) {
                resp_id_invalidate_all();
                /* ★ 与流式分支保持一致：客户端断连不是服务端错误。
                 * 2026-10-25：此前这里无条件写 MYDS_STOP_ERROR，于是非流式请求
                 * 因客户端超时断开时会被记成 finish_reason=error，并向已关闭的
                 * socket 再发一次 500。cancel_flag 单独成案。 */
                stop_reason = ns.cancel_flag ? MYDS_STOP_CANCELLED : MYDS_STOP_ERROR;
                transport_ok = false;
                if (stop_reason == MYDS_STOP_ERROR) {
                    send_response(client_fd, 500, "Internal Server Error", "application/json",
                                  "{\"error\":{\"message\":\"Inference or response allocation failed\",\"type\":\"server_error\"}}");
                }
            } else if (!ns.cancel_flag) {
                char *escaped = json_escape_alloc(ns.buf, ns.len);
                char *alias_json = json_escape_alloc(model_alias ? model_alias : "myds",
                                                     strlen(model_alias ? model_alias : "myds"));
                char *response = NULL;
                /* usage 扩展段（缓存明细 + timings）。必须在构造 response 之前算好，
                 * 因为 t_end 在这里就是「生成刚返回」的时刻。 */
                /* 768 不是随手写的：这段 JSON 约 400 字符（扁平字段 + timings 九个键）。
                 * 原来 384，加了字段后被 snprintf 截断，末尾少一个括号，整个响应无法解析。
                 * 截断的 JSON 比拒绝更糟 —— 客户端只说“解析失败”，看不出是我们截的。 */
                char usage_extra[768];
                myds_timings_t tim;
                server_timings_compute(ctx, final_tokens, t_first, myds_server_now_s(),
                                       (stop_reason == MYDS_STOP_EOS), &tim);
                server_usage_extra_json(&tim, usage_extra, sizeof(usage_extra));
                /* 顶层 timings —— mycode 的 ChatCompletionResponse.timings 读这里，
                 * 不是 usage 里面。两份都给：不同客户端读不同位置。 */
                char top_timings[384];
                server_timings_json(&tim, top_timings, sizeof(top_timings));
                /* LOOP_GUARD 归入 "length"：输出确实被截断了，客户端必须能分辨。
                 * 映射成 "stop" 会让半截答案看起来像正常收尾（原实现的问题）。 */
                const char *finish = (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL ||
                                      stop_reason == MYDS_STOP_LOOP_GUARD) ? "length" : "stop";
                long now = (long)time(NULL);

                /* 模型可能在正文之后用 DSML 块发起工具调用 ⇒ 拆出来，按各自协议输出。
                 *
                 * ★ 2026-11-01：原实现在解析失败时「整段当正文」，于是被 max_tokens 截断的
                 * 半截 `<|DSML| calls><|DSML| invoke ...` 会原样出现在 content 里。
                 * 现在按四态分类（见 myds_dsml_classify）：坏块【隔离】——正文只到开标签之前，
                 * 块本身丢弃并告警，finish_reason 归入 "length"（不谎报完整）。 */
                size_t tool_content_len = ns.len;
                myds_tool_call_t *tool_calls = NULL;
                size_t tool_count = 0;
                bool ns_dsml_dropped = false;
                const myds_dsml_state_t ns_dsml = myds_dsml_classify(ns.buf, ns.len, &tool_content_len);
                if (ns_dsml != MYDS_DSML_NONE &&
                    !myds_parse_completion_tool_calls(ns.buf, ns.len, &tool_content_len,
                                                      &tool_calls, &tool_count)) {
                    tool_calls = NULL; tool_count = 0;
                    ns_dsml_dropped = true;
                    MYDS_LOG_W("[%s:%d] %s DSML tool block discarded (%zu chars withheld from content)",
                               client_ip, client_port,
                               ns_dsml == MYDS_DSML_TRUNCATED ? "truncated" : "malformed",
                               ns.len - tool_content_len);
                }

                if (!tool_count && ns_dsml == MYDS_DSML_NONE &&
                    dsml_looks_like_foreign_dialect(ns.buf, ns.len)) {
                    MYDS_LOG_W("[%s:%d] tool-call markup found but this request's dialect is not "
                               "|DSML|calls; it will be sent as plain content. The model emitted "
                               "<function_calls>/<invoke> instead of the requested |DSML| form.",
                               client_ip, client_port);
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
                if (ns_dsml_dropped) finish = "length";
                if (tool_count) finish = "tool_calls";

                if (body_escaped && alias_json) {
                    if (is_responses_endpoint) {
                        response = items_json ? format_alloc(
                            "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,"
                            "\"status\":\"completed\",\"model\":\"%s\",\"output\":%s,"
                            "\"usage\":{\"input_tokens\":%u,\"output_tokens\":%u,\"total_tokens\":%u%s},\"timings\":%s}",
                            resp_id, now, alias_json, items_json,
                            prompt_tokens, reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS),
                            prompt_tokens + reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS),
                            usage_extra, top_timings) : NULL;
                    } else if (tool_count && calls_json && content_field) {
                        if (reasoning_escaped) {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                                "\"content\":%s,\"reasoning_content\":\"%s\",\"tool_calls\":%s},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u%s},\"timings\":%s}",
                                now, alias_json, content_field, reasoning_escaped, calls_json, finish,
                                prompt_tokens, reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), prompt_tokens + reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), usage_extra, top_timings);
                        } else {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                                "\"content\":%s,\"tool_calls\":%s},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u%s},\"timings\":%s}",
                                now, alias_json, content_field, calls_json, finish,
                                prompt_tokens, reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), prompt_tokens + reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), usage_extra, top_timings);
                        }
                    } else {
                        if (reasoning_escaped) {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\",\"reasoning_content\":\"%s\"},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u%s},\"timings\":%s}",
                                now, alias_json, body_escaped, reasoning_escaped, finish, prompt_tokens, reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), prompt_tokens + reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), usage_extra, top_timings);
                        } else {
                            response = format_alloc(
                                "{\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion\",\"created\":%ld,"
                                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                                "\"finish_reason\":\"%s\"}],"
                                "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u%s},\"timings\":%s}",
                                now, alias_json, body_escaped, finish, prompt_tokens, reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), prompt_tokens + reported_completion_tokens(final_tokens, stop_reason == MYDS_STOP_EOS), usage_extra, top_timings);
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

        if (!is_warmup_probe && stop_reason != MYDS_STOP_ERROR && stop_reason != MYDS_STOP_CANCELLED && transport_ok) {
            /* 续跑自身不重渲 tools ⇒ 它不改变「当前活响应」的工具集，沿用上一轮指纹。 */
            /* 累积提示词：完整模式直接存本次 prompt；suffix 模式存「上一轮前缀 + 本次新增」。
             * 这是重启后能重建输入的唯一依据 —— kvdisk 只存 KV 字节，不存 token 归属。 */
            {
                char *prev_text = NULL;
                if (is_continuation) (void)resp_id_saved_text_copy(prev_id, &prev_text);
                char *accum = NULL;
                if (prev_text && prev_text[0]) {
                    const size_t a = strlen(prev_text), b = strlen(prompt);
                    accum = (char *)malloc(a + b + 1u);
                    if (accum) { memcpy(accum, prev_text, a); memcpy(accum + a, prompt, b + 1u); }
                } else {
                    const size_t b = strlen(prompt);
                    accum = (char *)malloc(b + 1u);
                    if (accum) memcpy(accum, prompt, b + 1u);
                }
                free(prev_text);
                const unsigned long long reg_sig = is_continuation
                    ? (resp_id_tools_sig(prev_id) ? resp_id_tools_sig(prev_id) : resp_id_live_tools_sig())
                    : tools_sig;
                resp_id_register_text(resp_id, reg_sig, accum);
                server_infer_queue_set_active_resp_ids(resp_id, is_continuation ? prev_id : NULL);
                free(accum);
            }
        } else {
            resp_id_invalidate_all();
            server_infer_queue_set_active_resp_id(NULL);
        }

        double t_end = myds_server_now_s();
        /* EOS 是否真的生成了。远程 DeepSeek 与 llama.cpp 都把 EOS 计入
         * completion_tokens / predicted_n；我们原来不计，于是同样的内容
         * 两边 token 数永远差 1（"y" 是 出:2 对 出:1），界面上看着像我们少生成。 */
        const bool eos_generated = (stop_reason == MYDS_STOP_EOS);
        /* 三种时间分开报，不再混成一个 wall：
         *   wall  = 端到端（客户端看到的），含排队；外部监控/压测拿它与客户端实测比。
         *   infer = 拿到推理名额之后的时间，即引擎实际消耗。
         *   queue = 等推理名额的时间。
         * 2026-10-25 之前 wall 包含排队，导致并发时把等锁当成引擎变慢；
         * 中间一版又把 wall 改成不含排队，反而让「wall 与客户端不符」。
         * 两个都有问题，所以三个数各报一个。 */
        double infer_s = t_end - t_start;
        double wall_s = t_end - t_arrive;
        double prefill_s = (final_tokens > 0 && t_first > t_start) ? (t_first - t_start) : infer_s;
        double ttft_s = prefill_s;
        /* decode 窗口与计入的步数。
         *
         * out >= 2：(N-1) 个 token 落在 [第一个 token, 最后一个 token] 之间 ——
         *           那 N-1 次前向正好产出这 N-1 个 token，口径干净。
         * out == 1：内容只有 1 个 token，但引擎之后还跑了一次前向去求 EOS。
         *           那一步是真实的 decode 工作，用 [第一个 token, 生成返回] 度量它。
         *           远程 DeepSeek 把 EOS 计入 completion_tokens（内容只有 "y"
         *           却报 出:2），于是它也能算出 49.1 t/s —— 那个数就是这一步。
         *           不计这一步的话，out=1 永远是 n/a，而用户看到的是「远程有、我们没有」。
         *
         * 两者都记「每秒多少步前向」，不是「每秒多少内容 token」——
         * 这一点在页面的 tooltip 里写明，避免和 completion token 计数混淆。 */
        /* 统一的 decode 记账，日志 / 追踪 / usage 三处共用这一份。
         *
         * 窗口 = [第一个 token 吐出, generate 返回]。
         * 这个窗口里的前向次数：
         *   生成了 EOS ⇒ 恰好 final_tokens 次（含最后那一次求 EOS）
         *   被 length 截断 ⇒ final_tokens - 1 次（首个 token 来自 prefill 末位，免费）
         *
         * 为什么把 EOS 那次也算进来：远程 DeepSeek 把 EOS 计入 completion_tokens
         * （内容只有 "y" 也报「出:2」），LLaMA 系同样把它算进 predicted_n。
         * 不算的话，同一个内容两边 token 数永远差 1，界面上看起来就像我们少生成了东西。
         */
        double decode_s = 0.0;
        double decode_steps = 0.0;
        if (final_tokens > 0 && t_first > 0.0 && t_end > t_first) {
            decode_s = t_end - t_first;
            decode_steps = eos_generated ? (double)final_tokens
                                         : (double)(final_tokens - 1);
        }
        /* 没有 decode 阶段时**不给速率**（-1），不要拿 infer_s 凑一个。
         *
         * 原来的 fallback 是 final_tokens / infer_s。对 out=1 的请求，
         * 那等于「1 个 token 除以整个请求耗时」，会打出
         *   decode: 1 tok in 0.0s = 0.6 t/s
         * 看起来像「decode 慢到 0.6 t/s」，而真相是这次根本没有 decode 阶段
         * —— 那一个 token 的首字延迟已经算在 ttft 里了。
         * 编一个数比留空更难查：读者会去追一个不存在的问题。 */
        const bool have_decode_phase = decode_s > 0.0 && decode_steps > 0.0;
        double decode_tps = have_decode_phase ? (decode_steps / decode_s) : -1.0;
        const char *final_finish_str =
            (stop_reason == MYDS_STOP_LENGTH || stop_reason == MYDS_STOP_CTX_FULL ||
             stop_reason == MYDS_STOP_LOOP_GUARD) ? "length"
            : (stop_reason == MYDS_STOP_CANCELLED ? "cancelled"
            : (stop_reason == MYDS_STOP_ERROR ? "error" : "stop"));

        /* prefill 速率直接取自引擎的 prefill 阶段计时，不用 ttft 反推：
         * ttft 混了排队/分词/首个解码步，反推会低估速率（实测差数倍），
         * 而「复用命中的轮次」更会把固定开销摊到极少量 token 上 ⇒ 看起来极慢。 */
        myds_prefill_stats_t pf;
        myds_last_prefill_stats(ctx, &pf);
        char prefill_str[96];
        if (!pf.valid) {
            /* 本请求从未产出 prefill 统计 ⇒ 分词/排队阶段就结束了。
             * 旧实现会借用 ctx 上【上一个请求】的 token 数，打出
             * 「prompt: 0 tok, prefill: FAILED after 15815 tok」这种自相矛盾的行。 */
            snprintf(prefill_str, sizeof(prefill_str), "not started");
        } else if (pf.prefill_tokens == 0) {
            snprintf(prefill_str, sizeof(prefill_str), "none (all %u cached)", pf.cached_tokens);
        } else if (stop_reason == MYDS_STOP_CANCELLED) {
            /* 取消时引擎的 prefill_tokens 仍是【计划量】，不是实际跑过的量。
             * 原实现把它按速率打出来，于是「客户端早断开、prefill 在第一个 chunk 就中止」
             * 会显示成「19995 tok in 5.3s = 3762 t/s」—— 看起来像跑完了。
             * 取消时不给速率，只报计划量与事实。 */
            snprintf(prefill_str, sizeof(prefill_str), "CANCELLED (planned %u tok, cached %u)",
                     pf.prefill_tokens, pf.cached_tokens);
        } else if (stop_reason == MYDS_STOP_ERROR) {
            /* prefill 失败时不能按速率打。原来会打出「399952 tok in 0.000s = 33329333333.3 t/s」
             * 这种荒谬的成功行（实测），在排查事故时极具误导性 ——
             * 那条 500K 请求把机器打 panic 的日志里就混着这种行。只报失败与已处理量。 */
            snprintf(prefill_str, sizeof(prefill_str), "FAILED after %u tok (cached %u)",
                     pf.prefill_tokens, pf.cached_tokens);
        } else {
            double pf_tps = (pf.prefill_seconds > 0.0)
                                ? (double)pf.prefill_tokens / pf.prefill_seconds : 0.0;
            snprintf(prefill_str, sizeof(prefill_str), "%u tok in %.1fs = %.1f t/s (cached %u)",
                     pf.prefill_tokens, pf.prefill_seconds, pf_tps, pf.cached_tokens);
        }
        /* decode 段的措辞。没有阶段时不打数字，直接说明原因 ——
         * 「1 tok in 0.0s = 0.6 t/s」这种行会让人去查一个不存在的问题。 */
        char decode_str[128];
        if (have_decode_phase && final_tokens > 1)
            snprintf(decode_str, sizeof(decode_str), "%u tok in %.1fs = %.1f t/s",
                     final_tokens, decode_s, decode_tps);
        else if (have_decode_phase)
            /* out=1：一个内容 token + 一次求 EOS 的前向。数字是「秒/次前向」。 */
            snprintf(decode_str, sizeof(decode_str),
                     "1 tok, eos-step %.0fms = %.1f t/s", decode_s * 1000.0, decode_tps);
        else
            snprintf(decode_str, sizeof(decode_str), "n/a (%u tok total)", final_tokens);

        char steer_summary[64] = "";
        if (ctx && ctx->steer && ctx->steer->enabled && ctx->steer->alpha_effective > 0.0f) {
            snprintf(steer_summary, sizeof(steer_summary), ", steer: %s(a=%.2f)",
                     ctx->steer->name, ctx->steer->alpha_effective);
        } else if (ctx && ctx->steer && !ctx->steer->enabled) {
            snprintf(steer_summary, sizeof(steer_summary), ", steer: disabled");
        }

        MYDS_LOG_I("[%s:%d] Completed %s %s - prompt: %u tok, prefill: %s, ttft: %.1fs, "
                   "decode: %s, wall: %.1fs (infer %.1fs, queue %.1fs), finish_reason: %s%s",
                   client_ip, client_port,
                   is_responses_endpoint ? "Responses" : "ChatCompletions",
                   path, prompt_tokens, prefill_str, ttft_s,
                   decode_str, wall_s, infer_s, queue_s, final_finish_str, steer_summary);
        /* 构造阶段分解并**无条件**存进环形缓冲（/trace 页面读它），
         * 日志行仍只在 MYDS_REQUEST_PHASES=1 时打。
         *
         * 为什么记录不受开关控制：页面默认就该有数据，否则点开是空的，
         * 使用者还得先知道有个环境变量才能看到东西 —— 那等于没有页面。
         * 开销是复制十几个已经算好的 double，不在请求路径上新增任何测量。 */
        {
            /* One machine-readable line, emitted only under MYDS_REQUEST_PHASES=1.
             *
             * Each field is null when that stage did not exist for this request
             * (early return, cancellation, engine never called), because a printed
             * 0.0 is indistinguishable from "measured, took no time" -- and that is
             * how a fabricated number ends up in a ledger.
             *
             * `engine` is the total time inside generate(); `prefill` and `decode`
             * come from the engine's own counters. engine - prefill - decode is what
             * the engine does not attribute: tokenize, workspace setup, prefix
             * lookup, snapshot restore, first-sample latency. */
            const bool have_render = t_render_end > 0.0 && t_render_begin > 0.0;
            /* t_last is the time of the last decoded token, which the engine
             * reports slightly before generate() actually returns. A request that
             * decoded nothing leaves it at 0, so "engine" needs both. */
            const bool have_generate = t_generate_begin > 0.0 && t_last > 0.0 &&
                                       t_last >= t_generate_begin;
            const bool have_decode = final_tokens > 1 && t_last > t_first && t_first > 0.0;
            const double render_s = have_render ? (t_render_end - t_render_begin) : 0.0;
            const double setup_s = (have_generate && have_render) ? (t_generate_begin - t_render_end) : 0.0;
            const double engine_s = have_generate ? (t_last - t_generate_begin) : 0.0;
            const double decode_phase_s = have_decode ? (t_last - t_first) : 0.0;
            const double unattributed_s = (have_generate && pf.valid)
                ? engine_s - pf.prefill_seconds - decode_phase_s : 0.0;
            myds_request_phases_t phases_record = {
                .queue_s = queue_s,
                .render_s = have_render ? render_s : -1.0,
                .setup_s = (have_generate && have_render) ? setup_s : -1.0,
                .engine_s = have_generate ? engine_s : -1.0,
                .prefill_s = pf.valid ? pf.prefill_seconds : -1.0,
                .decode_s = have_decode ? decode_phase_s : -1.0,
                .unattributed_s = (have_generate && pf.valid) ? unattributed_s : (0.0 / 0.0),
                .prompt_tokens = prompt_tokens,
                .cached_tokens = pf.valid ? pf.cached_tokens : 0u,
                .out_tokens = final_tokens,
                .stream = stream ? 1 : 0,
            };
            myds_trace_rec_t trec;
            memset(&trec, 0, sizeof(trec));
            /* detail 的默认值也必须是「未测量」：pd.valid == 0 时不能留 0，
             * 否则页面把 GPU 忙画成 <0.1us。phases 下面会逐个赋值，不受影响。 */
            myds_trace_rec_mark_unmeasured(&trec);
            snprintf(trec.path, sizeof(trec.path), "%s", path);
            snprintf(trec.client, sizeof(trec.client), "%s:%d", client_ip, client_port);
            snprintf(trec.finish, sizeof(trec.finish), "%s", final_finish_str);
            snprintf(trec.resp_id, sizeof(trec.resp_id), "%s", resp_id);
            trec.phases.queue_s = phases_record.queue_s;
            trec.phases.render_s = phases_record.render_s;
            trec.phases.setup_s = phases_record.setup_s;
            trec.phases.engine_s = phases_record.engine_s;
            trec.phases.prefill_s = phases_record.prefill_s;
            trec.phases.decode_s = phases_record.decode_s;
            trec.phases.unattributed_s = phases_record.unattributed_s;
            trec.prompt_tokens = phases_record.prompt_tokens;
            trec.cached_tokens = phases_record.cached_tokens;
            trec.out_tokens = phases_record.out_tokens;
            trec.stream = phases_record.stream;
            /* 细粒度分解。未量测的保持 -1（trace 初始值）⇒ JSON null ⇒ 页面「—」。
             * 绝不因为“看起来应该差不多”就填个 0 或推算值。 */
            {
                myds_phase_detail_t pd;
                myds_last_phase_detail(ctx, &pd);
                if (pd.valid) {
                    trec.detail.prefill_gpu_s = pd.prefill_gpu_s;
                    trec.detail.prefill_wait_s = pd.prefill_wait_s;
                    trec.detail.prefill_cbs = (uint32_t)pd.prefill_cbs;
                    trec.detail.decode_gpu_s = pd.decode_gpu_s;
                    trec.detail.decode_wait_s = pd.decode_wait_s;
                    trec.detail.decode_cbs = (uint32_t)pd.decode_cbs;
                    trec.detail.tokenize_s = pd.tokenize_s;
                    trec.detail.kvdsk_s = pd.kvdsk_s;
                    memcpy(trec.detail.prefill_tag_gpu, pd.prefill_tag_gpu,
                           sizeof(trec.detail.prefill_tag_gpu));
                    memcpy(trec.detail.prefill_tag_cbs, pd.prefill_tag_cbs,
                           sizeof(trec.detail.prefill_tag_cbs));
                    memcpy(trec.detail.decode_tag_gpu, pd.decode_tag_gpu,
                           sizeof(trec.detail.decode_tag_gpu));
                    memcpy(trec.detail.decode_tag_cbs, pd.decode_tag_cbs,
                           sizeof(trec.detail.decode_tag_cbs));
                }
                /* emit 是服务端量的，不走引擎。只有确实生成了 token 才有意义。 */
                if (emit_measured_s >= 0.0) trec.detail.emit_s = emit_measured_s;
            }
            /* 三个速度指标。口径写清楚，因为页面上并排显示、很容易读串：
             *   ttft        请求到达 → 第一个 token（含排队）
             *   prefill_tps 引擎自报，只用真正跑过的 token 除以耗时（不含排队/分词）
             *   decode_tps  生成阶段；取消/失败时是负值 ⇒ null，不是 0
             * 原实现用 ttft 反推 prefill 速率，实测会让复用命中的轮次看起来极慢
             * —— 固定开销摊在极少量 token 上。这里直接取引擎的计时。 */
            /* 前缀缓存命中率。用 pf.cached_tokens 而不是 ctx 上的旧值 ——
             * pf 是本请求的统计，ctx 上的值在早退时会是上一次请求的残值。
             * 没有 prompt 统计 ⇒ 负值 ⇒ 页面显示「—」，不是 0%。 */
            /* 与 usage 同一口径：cached = prompt - 实际 prefill 的 token。
             * 用 resume_from 会多算 1（最后一个 prompt token 是 decode 的种子，
             * 本来就不参与 prefill），页面与 API 会差 0.02 个百分点。 */
            trec.cached_tokens = pf.valid ? (pf.prompt_tokens - pf.prefill_tokens) : 0u;
            trec.cache_hit_rate = (pf.valid && pf.prompt_tokens > 0)
                ? 100.0 * (double)trec.cached_tokens / (double)pf.prompt_tokens : -1.0;
            trec.ttft_s = (final_tokens > 0 && t_first > t_arrive) ? (t_first - t_arrive) : -1.0;
            /* 必须要求 prefill_tokens > 0：整个 prompt 都命中缓存时 tokens 是 0，
             * 0/seconds 会算出 0.0 t/s —— 页面显示「prefill 0 t/s」，
             * 读起来像「慢到零」，而真相是「这次没跑 prefill」。
             * 没有 token 就没有速率，输出 null。 */
            trec.prefill_tps = (pf.valid && pf.prefill_tokens > 0 && pf.prefill_seconds > 0.0)
                ? (double)pf.prefill_tokens / pf.prefill_seconds : -1.0;
            /* 与日志用同一个判定：有 decode 窗口就给数（含 out=1 的 EOS 步）。 */
            trec.decode_tps = (decode_s > 0.0 && decode_steps > 0.0) ? decode_tps : -1.0;
            /* layer_count 保持 0：引擎不默认给逐层计时，见 myds_trace.h 的说明。 */
            myds_trace_complete(trace_seq, &trec);

            char phases_json[512];
            myds_request_phases_json(&phases_record, phases_json, sizeof(phases_json));
            /* t_last precedes the engine's return by the last callback, so
             * engine_unattributed can come out slightly negative on a short
             * request. It bounds "neither prefill nor decode", it is not an exact
             * stage cost; a large negative means the split itself is wrong. */
            if (phases)
                MYDS_LOG_I("[%s:%d] request_phases %s", client_ip, client_port, phases_json);
            if (is_warmup_probe) {
                ctx->pcache_len = 0;
                MYDS_LOG_I("[%s:%d] Warmup probe finished (X-Myds-Warmup: 1): prompt=%u tok, prefill=%.3fs, trace skipped & pcache reset",
                           client_ip, client_port, prompt_tokens,
                           pf.valid ? pf.prefill_seconds : prefill_s);
            }
        }
        if (ctx->dspark && ctx->dspark->proposed_tokens > dspark_prop_before) {
            uint64_t req_prop = ctx->dspark->proposed_tokens - dspark_prop_before;
            uint64_t req_acc  = ctx->dspark->accepted_tokens - dspark_acc_before;
            uint64_t req_cyc  = ctx->dspark->cycles - dspark_cyc_before;
            double req_prop_ms = ctx->dspark->propose_ms - dspark_prop_ms_before;
            double req_ver_ms  = ctx->dspark->verify_ms - dspark_ver_ms_before;
            MYDS_LOG_I("[%s:%d] DSpark Speculative: proposed=%llu, accepted=%llu (%.1f%% accept rate, %.2f tok/cycle, cumulative: %llu/%llu = %.1f%%) | propose=%.1fms (%.2fms/cyc), verify=%.1fms (%.2fms/cyc)",
                       client_ip, client_port,
                       (unsigned long long)req_prop,
                       (unsigned long long)req_acc,
                       req_prop > 0 ? (100.0 * (double)req_acc / (double)req_prop) : 0.0,
                       (double)req_acc / (double)(req_cyc > 0 ? req_cyc : 1),
                       (unsigned long long)ctx->dspark->accepted_tokens,
                       (unsigned long long)ctx->dspark->proposed_tokens,
                       ctx->dspark->proposed_tokens > 0 ? (100.0 * (double)ctx->dspark->accepted_tokens / (double)ctx->dspark->proposed_tokens) : 0.0,
                       req_prop_ms, req_cyc > 0 ? req_prop_ms / (double)req_cyc : 0.0,
                       req_ver_ms, req_cyc > 0 ? req_ver_ms / (double)req_cyc : 0.0);
        }
        if (ctx->decode_stats.batch2_calls > 0) {
            const double calls = (double)ctx->decode_stats.batch2_calls;
            /* verify_ms covers all request verification, including scalar B
             * rescue/fallback. GPU phase is CPU commit wait, not GPU duration. */
            MYDS_LOG_I("[%s:%d] Spec Verify Stats: calls=%llu (hits=%llu, misses=%llu, failures=%llu, fallbacks=%llu) | request_verify_total=%.2fms, mean_prep=%.2fms, mean_encode=%.2fms, mean_commit_wait=%.2fms, mean_arb_publish=%.2fms",
                       client_ip, client_port,
                       (unsigned long long)ctx->decode_stats.batch2_calls,
                       (unsigned long long)ctx->decode_stats.batch2_hits,
                       (unsigned long long)ctx->decode_stats.batch2_misses,
                       (unsigned long long)ctx->decode_stats.batch2_failures,
                       (unsigned long long)ctx->decode_stats.batch2_fallbacks,
                       ctx->decode_stats.verify_ms,
                       ctx->decode_stats.batch2_prep_ms / calls,
                       ctx->decode_stats.batch2_encode_ms / calls,
                       ctx->decode_stats.batch2_gpu_ms / calls,
                       ctx->decode_stats.batch2_arb_ms / calls);

        }

        if (ctx->decode_stats.depth2_calls > 0) {
            MYDS_LOG_I("[%s:%d] Depth2 Verify: calls=%llu, hits1=%llu, hits2=%llu, misses=%llu, failures=%llu, fallbacks=%llu",
                       client_ip, client_port,
                       (unsigned long long)ctx->decode_stats.depth2_calls,
                       (unsigned long long)ctx->decode_stats.depth2_hits1,
                       (unsigned long long)ctx->decode_stats.depth2_hits2,
                       (unsigned long long)ctx->decode_stats.depth2_misses,
                       (unsigned long long)ctx->decode_stats.depth2_failures,
                       (unsigned long long)ctx->decode_stats.depth2_fallbacks);
        }

cleanup_infer:
        myds_context_clear_image_embeddings(ctx);
        /* 2026-10-25：清理本请求可能设置的结构化输出 bitmask。
         *
         * ★ 必须清：bitmask 存在 ctx 上（ctx 跨请求复用），不清就会把上一个
         * 请求的 token 白名单带到下一个请求 —— 那是功能错误，也是越权风险。
         *
         * ★ 为什么内联而不是调 myds_context_clear_logit_bitmask()：
         * 那个函数在 myds_engine.c，而 tests/test_http_bytes.c 只编译 server.c
         * （它 #include 了本文件），链接期会 undefined symbol ⇒ make test 失败。
         * server.c 已 include myds_internal.h，ctx 定义可见，直接内联即可。
         * 逻辑与 engine 版完全一致（含容量归零），改一处必须同步改另一处。 */
        /* ★ 必须检查 ctx：handle_client 的 ctx 参数允许为 NULL
         * （tests/test_http_bytes.c 就传 NULL 模拟无 model 场景）。
         * 我第一版内联时漏了这个判断，直接把 ctx->logit_bitmask 解引用，
         * 测试立刻 SIGSEGV（退出码 139）。engine 版函数开头有 if (!ctx) return;
         * 内联时【必须把等价的保护一起搬过来】。 */
        if (ctx) {
            if (ctx->logit_bitmask) {
                free(ctx->logit_bitmask);
                ctx->logit_bitmask = NULL;
            }
            ctx->logit_bitmask_words = 0;
            if (ctx->steer) {
                if (prev_steer_alpha >= 0.0f) ctx->steer->alpha_effective = prev_steer_alpha;
                if (prev_steer_enabled >= 0) ctx->steer->enabled = (prev_steer_enabled != 0);
            }
            ctx->steer = prev_steer;
        }
        myds_vjob_sink_free(&img_sink);
        myds_set_continue_from_cache(ctx, false);
        myds_context_set_config(ctx, &prev_cfg);
        /* 摘掉探针：fd 马上就要 close，留着它下次可能命中一个复用了同号的
         * 新连接 —— 那时探针会把「新连接还活着」当成「这个请求还活着」。 */
        myds_context_set_liveness_probe(ctx, NULL, NULL);
        g_prefill_probe_fd = -1;
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
    /* 2026-10-25: 每个连接固定 4 MB 栈，无上限时几千条连接就能把内存吃光。
     * 计数放在连接结束时递减，与 accept 处的递增成对。 */
    atomic_fetch_sub_explicit(&g_client_threads, 1, memory_order_relaxed);
    return NULL;
}

int myds_server_start(myds_context_t *ctx, const char *host, int port, const char *model_alias) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* 请求追踪缓冲。默认 128 条（约 190 KB；每条最坏 1.5 KB —— 阶段标签全开时
     * 26 个 prefill 桶 + 23 个 decode 桶都有值）；MYDS_TRACE_RECORDS 可调，上限 512。
     *
     * 在 start 时显式初始化，而不是等第一条 record 惰性分配：
     * 否则 /trace 在还没有请求时报告 capacity=0，页面显示成「容量未知」——
     * 那不是「没有数据」，是「还没准备」，两者该分开。 */
    {
        uint32_t cap = 128u;
        const char *env = getenv("MYDS_TRACE_RECORDS");
        if (env && *env) {
            const long v = atol(env);
            if (v > 0 && v <= 512) cap = (uint32_t)v;
            else MYDS_LOG_W("MYDS_TRACE_RECORDS=%s 超出 (0,512]，用默认 128", env);
        }
        myds_trace_init(cap);
        MYDS_LOG_I("request trace: %u records in the ring buffer (page: GET /trace)",
                   cap);
    }

    /* LoopGuard 默认【关闭】，需要显式 MYDS_LOOP_GUARD=1 才启用。
     *
     * 2026-11-01 验收改判：原实现默认开启，但假阳性率从未测量过。
     * 把真实文本喂给同一份检测器（周期 4、连续 4 次即触发）实测：
     *   src/myds_engine.c   74,700 tok ⇒ 23 次触发
     *   docs 账本           397,908 tok ⇒ 172 次触发
     * 触发窗口全是 `|---|---|---|`、`────────────` 这类表格分隔行和注释框线，
     * 不是死循环。默认开启等于用「截断正常回答」去防一个未观测到的风险。
     * 等官方套件上有假阳性读数之后，再决定默认值。 */

    /* 2026-10-25: 把 fd 上限提升到合理值。
     * 默认 ulimit -n 常为 256，而每条连接占一个 fd ⇒ 265 条连接即 EMFILE。
     * 这里尝试抬到 4096（不超过硬上限），失败不致命——但至少让
     * “默认部署”有能力服务上百并发。 */
    {
        struct rlimit rl;
        if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
            const rlim_t want = 4096;
            if (rl.rlim_cur < want) {
                rlim_t target = (rl.rlim_max == RLIM_INFINITY || rl.rlim_max > want)
                                ? want : rl.rlim_max;
                rl.rlim_cur = target;
                if (setrlimit(RLIMIT_NOFILE, &rl) == 0) {
                    MYDS_LOG_I("fd 上限已提升到 %llu", (unsigned long long)target);
                } else {
                    MYDS_LOG_W("fd 上限提升失败(%s)，保留原值 %llu",
                               strerror(errno), (unsigned long long)rl.rlim_cur);
                }
            }
        }
    }

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
        /* 2026-10-25: 必须检查返回值。inet_pton 失败时返回 0 且【不修改】目标，
         * 而 addr 刚被 memset 清零 ⇒ sin_addr 保持 0 = INADDR_ANY = 0.0.0.0。
         * 于是 `--host localhost`（不是点分十进制）会默默监听全部接口，
         * 而日志依旧打印 localhost —— 这是静默提权，比默认值问题严重得多。
         * 这里只接受 IPv4 点分十进制；主机名请用 --host 0.0.0.0 显式表达意图。 */
        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            MYDS_LOG_E("拒绝启动：无法解析 --host '%s'。只接受 IPv4 点分十进制地址"
                       "（如 127.0.0.1 / 0.0.0.0）；主机名请直接写目标地址。", host);
            close(g_server_fd);
            g_server_fd = -1;
            return 1;
        }
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
            /* 2026-10-25: EMFILE/ENFILE 是【暂时的资源耗尽】，不是致命错误。
             * 原实现一律 break ⇒ 任何客户端只要开满 fd，服务器就自己退出了。
             * 实测：ulimit -n 为 256 时，开 264 条空闲连接即可让服务死掉。
             * 现在改成：回一条 503、短暂退避后继续 accept。
             * 退避是必需的——否则 fd 仍然耗尽时 accept 会立刻再失败，形成忙循环。 */
            if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS || errno == ENOMEM) {
                if (++g_emfile_streak == 1u || (g_emfile_streak % 100u) == 0u) {
                    MYDS_LOG_W("Socket accept 资源不足(%s)，继续提供服务（第 %u 次）",
                               strerror(errno), g_emfile_streak);
                }
                struct timespec backoff = {0, 50 * 1000 * 1000};   /* 50 ms */
                nanosleep(&backoff, NULL);
                continue;
            }
            MYDS_LOG_E("Socket accept error: %s", strerror(errno));
            break;
        }
        g_emfile_streak = 0;

        char client_ip[INET_ADDRSTRLEN] = "127.0.0.1";
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        int client_port = ntohs(client_addr.sin_port);

        /* 2026-10-25: 连接数封顶。超出时直接回 503 并关闭，
         * 而不是默默开新线程（每条 4 MB 栈，无上限时内存可被吃光）。
         * ★ 用原子比较交换而不是先读后写，否则并发 accept 会让计数超过上限。 */
        {
            const unsigned cap = max_client_threads();
            unsigned cur = atomic_load_explicit(&g_client_threads, memory_order_relaxed);
            int admitted = 0;
            while (cur < cap) {
                if (atomic_compare_exchange_weak_explicit(&g_client_threads, &cur, cur + 1u,
                                                          memory_order_relaxed,
                                                          memory_order_relaxed)) {
                    admitted = 1;
                    break;
                }
            }
            if (!admitted) {
                MYDS_LOG_W("[%s:%d] 并发连接已达上限 %u ⇒ 503", client_ip, client_port, cap);
                send_response(client_fd, 503, "Service Unavailable", "application/json",
                              "{\"error\":{\"message\":\"Too many concurrent connections\"}}");
                close(client_fd);
                continue;
            }
        }

        server_client_job_t *job = (server_client_job_t *)malloc(sizeof(server_client_job_t));
        if (!job) {
            /* 同步执行完成后也必须退还名额 */
            handle_client(client_fd, ctx, model_alias, client_ip, client_port);
            atomic_fetch_sub_explicit(&g_client_threads, 1, memory_order_relaxed);
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
            /* 线程没建成功，名额同样要退 */
            atomic_fetch_sub_explicit(&g_client_threads, 1, memory_order_relaxed);
        }
        pthread_attr_destroy(&attr);
    }

    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }

    /* ★ 等在场请求退完，再让 main 释放 context。
     *
     * 默认等 120 s：32K prefill 要 ~50 s，262K 冷 prefill 要 ~12 min ——
     * 后者等不完，所以给上限而不是无限等。超时就【不释放】context：
     * 宁可泄漏、让进程退出时由内核回收，也不能在别的线程还在写的时候 free。
     * 这就是原实现崩掉的那个场景。 */
    {
        const char *env = getenv("MYDS_SHUTDOWN_DRAIN_S");
        long wait_s = 120;
        if (env && *env) { long v = strtol(env, NULL, 10); if (v >= 0 && v <= 3600) wait_s = v; }
        const unsigned n0 = atomic_load_explicit(&g_client_threads, memory_order_acquire);
        if (n0) {
            MYDS_LOG_I("Shutdown: waiting for %u in-flight request(s) (up to %ld s; "
                       "MYDS_SHUTDOWN_DRAIN_S overrides)", n0, wait_s);
            const int64_t deadline_ms = (int64_t)wait_s * 1000;
            int64_t waited = 0;
            while (atomic_load_explicit(&g_client_threads, memory_order_acquire) != 0u) {
                if (waited >= deadline_ms) break;
                struct timespec ts = {0, 5 * 1000 * 1000};   /* 5 ms */
                nanosleep(&ts, NULL);
                waited += 5;
            }
            const unsigned n1 = atomic_load_explicit(&g_client_threads, memory_order_acquire);
            if (n1) {
                /* 超时：context 绝不能释放（还有线程在用）。进程马上退出，
                 * 内核会回收全部内存；泄漏一份比 SIGSEGV 好。 */
                MYDS_LOG_E("Shutdown: %u request(s) still running after %ld s ⇒ "
                           "skipping context teardown (process exit reclaims memory)",
                           n1, wait_s);
                return 2;   /* 特殊返回值：调用方不得 free context */
            }
            MYDS_LOG_I("Shutdown: all request threads drained");
        }
    }

    MYDS_LOG_I("Server stopped gracefully");
    return 0;
}
