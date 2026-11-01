/* myds_resp_store.c — previous_response_id 注册表与持久化
 *
 * 2026-11-01 从 myds_server.c 纯机械拆分而来：函数体、全局变量、
 * 注释一个字未改，只把 static 链接性调整到能跨编译单元使用。
 *
 * 这块单独成文件的原因：它是「多轮会话状态」，与 HTTP 传输、协议渲染、
 * 推理调度都无关；混在 5000 行的 myds_server.c 里时，改它要重新理解整个服务器。
 */
#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>

/* ── previous_response_id：响应 id 注册表 ──
 * Responses API 允许客户端只发【新增内容】+ 上一次响应的 id，让服务端接着那次的状态跑。
 *
 * 2026-11-01：单活 → 多活。
 *
 * 原设计只有【最近一次】响应的状态活着（pcache 恰好停在它末尾，连快照都不用恢复），
 * 更早的 id 一律报错。这个取舍在单会话下没问题，但多会话 agent 交错时是【必然】失败：
 * A 提问 → B 提问 → A 续跑，A 手里的 id 在这一刻已经不是最新。
 *
 * 现在保留快路径（最新 + pcache 在 ⇒ 零拷贝），并补一条恢复路径
 * （id 仍在表里 ⇒ 用持久化累积文本重建完整输入，全量 prefill）。
 * 只有真正未知的 id 才拒绝 —— 静默忽略永远不做，那会让模型拿不到上下文却照常作答。 */
/* 2026-11-01：从 4 扩到可配置（默认 16）。
 *
 * 为什么：多会话 agent（orchestrator + N 个 worker + 子代理）交错调用 /v1/responses，
 * 4 个槽在三个会话交错一次之后就绕回来了。而每个槽只是一份累积文本，
 * 16 个槽的成本是几十 KB，不是内存问题。 */
#define RESP_ID_SLOTS_DEFAULT 16u
#define RESP_ID_SLOTS_MAX 64u
static unsigned g_resp_id_slots = RESP_ID_SLOTS_DEFAULT;
static void resp_id_slots_init(void) {
    static bool done = false;
    if (done) return;
    done = true;
    const char *e = getenv("MYDS_RESP_SLOTS");
    if (e && *e) {
        const long v = atol(e);
        if (v >= 1 && v <= (long)RESP_ID_SLOTS_MAX) g_resp_id_slots = (unsigned)v;
        else MYDS_LOG_W("MYDS_RESP_SLOTS=%s 超出 [1,%u]，用默认 %u", e, RESP_ID_SLOTS_MAX, RESP_ID_SLOTS_DEFAULT);
    }
}
#define RESP_ID_SLOTS (g_resp_id_slots)
static char            g_resp_ids[RESP_ID_SLOTS_MAX][64];
/* 每个响应登记时所用的【工具集指纹】。
 *
 * 为什么必须存：suffix 模式（previous_response_id）只渲染新增消息，
 * tools 段【不重渲】（它已在前缀里）。若客户端在续跑时改动了 tools，
 * 旧实现会直接 goto 跳过 —— 既不渲染也不告警，模型看不到新工具，
 * 调用方只看到模型「假装没有这个工具」，无法定位。
 * 这里改为：登记响应时记下工具指纹；续跑时若带 tools 就比对，
 * 相同放行（等价于「重复声明未变」），不同则 400 明确报错。 */
static unsigned long long g_resp_tools_sig[RESP_ID_SLOTS_MAX];
static unsigned        g_resp_id_ring = 0;         /* 下一个写入位置 */
static unsigned long   g_resp_id_seq = 0;
static pthread_mutex_t g_resp_id_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ── 2026-10-24：响应状态的持久化 ──
 *
 * 🔒 隐私声明（2026-10-25 补充）
 *
 * 为了跨重启续跑，本机制会把【完整的累积提示词文本】以【明文】写入
 *   $HOME/.cache/myds/responses.idx
 * 文件权限为 0600（仅属主可读），但【未加密】。这意味着：
 *   · 你向本服务发过的全部对话内容都会落盘，可能包含代码、密钥、个人数据；
 *   · 同机的其他进程只要能读到该用户的文件，就能拿到这些内容；
 *   · 备份/快照工具会把它们一并带走。
 *
 * 如果这不适合你的部署：
 *   · 把 HOME 指到一个临时目录（缓存随之隔离）；
 *   · 或定期删除该文件（代价是重启后无法续跑 previous_response_id）；
 *   · 上游客户端也可以不用 previous_response_id，改为每轮重发完整历史。
 *
 * 为什么必须存文本而不是只存 token：kvdisk 只存 KV 字节，不存“这些字节
 * 对应哪些 token”，没有文本就无法在重启后重建输入。
 *
 * 问题：上面的注册表是静态数组，进程重启即清空。而客户端一旦改用
 * previous_response_id（suffix 模式），就不再重发历史 —— 服务端一重启，
 * 它手上的 id 变成 unknown，客户端却没有可 replay 的完整输入，会话直接断掉。
 *
 * kvdisk 只存 KV 字节，不存「这些字节对应哪些 token」。所以必须在这里补上
 * 【完整的累积提示词文本】：
 *
 *   turn 1（完整模式）：accumulated = 本次渲染出的完整 prompt
 *   turn N（suffix 模式）：accumulated = 上一轮 accumulated + 本次新增部分
 *
 * 重启后即使 pcache 空了，也能用 accumulated + 本次 suffix 重建出与客户端
 * 原本会发送的【逐字节相同】的输入，退回普通 prefill（正确性优先，速度次之）。
 * 若 kvdisk 里恰好还有对应前缀的 KV，普通路径自己会命中。
 *
 * 若磁盘上没有可用状态，仍按原逻辑返回 400，绝不静默丢上下文。
 */
#define RESP_STATE_MAGIC "MYDSRSP1"
#define RESP_STATE_VERSION 1u
static char   *g_resp_text[RESP_ID_SLOTS_MAX];  /* 累积提示词（堆） */
static bool    g_resp_state_loaded = false;
static char    g_resp_state_path[1200];

static void resp_state_path(char *out, size_t cap) {
    const char *env = getenv("MYDS_RESP_STATE");
    if (env && *env) { snprintf(out, cap, "%s", env); return; }
    const char *home = getenv("HOME");
    snprintf(out, cap, "%s/.cache/myds/responses.idx", home && *home ? home : ".");
}

static void resp_state_load_once(void) {
    if (g_resp_state_loaded) return;
    g_resp_state_loaded = true;
    resp_id_slots_init();
    resp_state_path(g_resp_state_path, sizeof(g_resp_state_path));
    FILE *f = fopen(g_resp_state_path, "rb");
    if (!f) return;
    char magic[8];
    uint32_t ver = 0, n = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, RESP_STATE_MAGIC, 8) != 0) { fclose(f); return; }
    if (fread(&ver, sizeof(ver), 1, f) != 1 || ver != RESP_STATE_VERSION) { fclose(f); return; }
    if (fread(&n, sizeof(n), 1, f) != 1 || n > RESP_ID_SLOTS_MAX) { fclose(f); return; }
    pthread_mutex_lock(&g_resp_id_mutex);
    for (uint32_t i = 0; i < n; ++i) {
        unsigned long long sig = 0;
        uint32_t len = 0;
        char id[64] = {0};
        if (fread(&sig, sizeof(sig), 1, f) != 1) break;
        if (fread(&len, sizeof(len), 1, f) != 1) break;
        if (fread(id, 1, sizeof(id), f) != sizeof(id)) break;
        if (len > (64u << 20)) break;                 /* 单条上限 64 MiB，防坏文件 */
        char *text = (char *)malloc((size_t)len + 1u);
        if (!text) break;
        if (len && fread(text, 1, len, f) != len) { free(text); break; }
        text[len] = '\0';
        snprintf(g_resp_ids[i], sizeof(g_resp_ids[i]), "%s", id);
        g_resp_tools_sig[i] = sig;
        free(g_resp_text[i]);
        g_resp_text[i] = text;
    }
    g_resp_id_ring = (unsigned)(n % RESP_ID_SLOTS);
    pthread_mutex_unlock(&g_resp_id_mutex);
    fclose(f);
    MYDS_LOG_I("response state: loaded %u slot(s) from %s", n, g_resp_state_path);
}

static void resp_state_save_locked(void) {
    resp_state_path(g_resp_state_path, sizeof(g_resp_state_path));
    char tmp[1300];
    snprintf(tmp, sizeof(tmp), "%s.tmp", g_resp_state_path);
    char dir[1200];
    snprintf(dir, sizeof(dir), "%s", g_resp_state_path);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = '\0'; mkdir(dir, 0700); }
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    /* 2026-10-25: 显式收窄到 0600。目录已经 0700，但【文件】之前没设权限，
     * 受 umask 影响实测落成 0644 —— 同机其他用户能读到完整的对话文本。
     * 这里在写入【前】就 chmod，而不是写完再改，避免中间窗口可读。 */
    (void)chmod(tmp, 0600);
    uint32_t ver = RESP_STATE_VERSION, n = RESP_ID_SLOTS;
    bool ok = fwrite(RESP_STATE_MAGIC, 1, 8, f) == 8 &&
              fwrite(&ver, sizeof(ver), 1, f) == 1 &&
              fwrite(&n, sizeof(n), 1, f) == 1;
    for (uint32_t i = 0; ok && i < RESP_ID_SLOTS; ++i) {
        const char *t = g_resp_text[i] ? g_resp_text[i] : "";
        uint32_t len = (uint32_t)strlen(t);
        ok = fwrite(&g_resp_tools_sig[i], sizeof(g_resp_tools_sig[i]), 1, f) == 1 &&
             fwrite(&len, sizeof(len), 1, f) == 1 &&
             fwrite(g_resp_ids[i], 1, 64, f) == 64 &&
             (len == 0 || fwrite(t, 1, len, f) == len);
    }
    fflush(f);
    (void)chmod(tmp, 0600);   /* 冗余设置：确保 rename 后仍是 0600 */
    fclose(f);
    if (ok) (void)rename(tmp, g_resp_state_path);
    else    (void)remove(tmp);
}

/* 取某个 id 对应的累积提示词（无则 NULL）。返回内部指针，调用方不得释放。
 *
 * ⚠ 这个接口有竞态：解锁后返回内部指针，另一个线程登记新响应时可能 free 掉它。
 * 新代码请用 resp_id_saved_text_copy()。保留它是为了不动既有调用点的语义。 */
const char *resp_id_saved_text(const char *id) {
    if (!id || !id[0]) return NULL;
    resp_state_load_once();
    const char *out = NULL;
    pthread_mutex_lock(&g_resp_id_mutex);
    for (unsigned i = 0; i < RESP_ID_SLOTS; ++i)
        if (g_resp_ids[i][0] && strcmp(g_resp_ids[i], id) == 0) { out = g_resp_text[i]; break; }
    pthread_mutex_unlock(&g_resp_id_mutex);
    return out;
}

/* 在锁内拷贝，调用方拿到的是自己的内存。返回拷贝的字节数，0 = 没有该 id。 */
size_t resp_id_saved_text_copy(const char *id, char **out) {
    if (out) *out = NULL;
    if (!id || !id[0] || !out) return 0;
    resp_state_load_once();
    size_t n = 0;
    pthread_mutex_lock(&g_resp_id_mutex);
    for (unsigned i = 0; i < RESP_ID_SLOTS; ++i) {
        if (!g_resp_ids[i][0] || strcmp(g_resp_ids[i], id) != 0) continue;
        const char *t = g_resp_text[i];
        if (t && *t) {
            const size_t len = strlen(t);
            char *copy = (char *)malloc(len + 1u);
            if (copy) { memcpy(copy, t, len + 1u); *out = copy; n = len; }
        }
        break;
    }
    pthread_mutex_unlock(&g_resp_id_mutex);
    return n;
}


void resp_id_new(char *out, size_t cap) {
    unsigned long n;
    pthread_mutex_lock(&g_resp_id_mutex);
    n = ++g_resp_id_seq;
    pthread_mutex_unlock(&g_resp_id_mutex);
    snprintf(out, cap, "resp_myds_%lu_%04x", n, (unsigned)(rand() & 0xffffu));
}
/* 一次响应完成后登记：它成为唯一「活」的 id。sig = 本次请求所用工具集指纹（0 表示无 tools）。 */
void resp_id_register(const char *id, unsigned long long tools_sig) {
    resp_id_register_text(id, tools_sig, NULL);
}

/* accumulated = 本轮结束后的【完整累积提示词】。NULL 表示本次不更新文本（保留旧的）。 */
void resp_id_register_text(const char *id, unsigned long long tools_sig, const char *accumulated) {
    if (!id || !id[0]) return;
    resp_state_load_once();
    char *copy = NULL;
    if (accumulated) {
        size_t n = strlen(accumulated);
        copy = (char *)malloc(n + 1u);
        if (copy) memcpy(copy, accumulated, n + 1u);
    }
    pthread_mutex_lock(&g_resp_id_mutex);
    unsigned slot = g_resp_id_ring;
    snprintf(g_resp_ids[slot], sizeof(g_resp_ids[0]), "%s", id);
    g_resp_tools_sig[slot] = tools_sig;
    if (copy) { free(g_resp_text[slot]); g_resp_text[slot] = copy; }
    g_resp_id_ring = (g_resp_id_ring + 1u) % RESP_ID_SLOTS;
    resp_state_save_locked();
    pthread_mutex_unlock(&g_resp_id_mutex);
}
/* 当前「活」响应的工具指纹；无活响应时返回 0。 */
unsigned long long resp_id_live_tools_sig(void) {
    unsigned long long sig;
    pthread_mutex_lock(&g_resp_id_mutex);
    unsigned prev = (g_resp_id_ring + RESP_ID_SLOTS - 1u) % RESP_ID_SLOTS;
    sig = (g_resp_ids[prev][0] != '\0') ? g_resp_tools_sig[prev] : 0ull;
    pthread_mutex_unlock(&g_resp_id_mutex);
    return sig;
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
/* 某个 id 登记时的工具集指纹。0 = 没有该 id 或它当时没带 tools。
 * 多活续跑必须用【祖先 id】的指纹：累积文本来自那次响应，
 * 用「最近一次」的指纹会把 A 的续跑按 B 的工具集校验。 */
unsigned long long resp_id_tools_sig(const char *id) {
    if (!id || !id[0]) return 0ull;
    unsigned long long sig = 0ull;
    pthread_mutex_lock(&g_resp_id_mutex);
    for (unsigned i = 0; i < RESP_ID_SLOTS; ++i)
        if (g_resp_ids[i][0] && strcmp(g_resp_ids[i], id) == 0) { sig = g_resp_tools_sig[i]; break; }
    pthread_mutex_unlock(&g_resp_id_mutex);
    return sig;
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
        free(g_resp_text[i]); g_resp_text[i] = NULL;
        g_resp_tools_sig[i] = 0ull;
    }
    g_resp_id_ring = 0;
    resp_state_save_locked();
    pthread_mutex_unlock(&g_resp_id_mutex);
}
