/* myds_debug_acl.c — /debug/state 与 /trace 的来源网段白名单
 *
 * 2026-11-01 新增。原因：这两个端点原本只允许回环来源。
 *
 * 为什么需要白名单而不是只放开回环：
 *   /trace 是给人看的页面。要求它只能从服务器本机的浏览器打开，等于要求操作者
 *   先开一条 SSH 隧道 —— 而隧道绑不了局域网（macOS 上 -L 0.0.0.0: 报
 *   "Can't assign requested address"），于是从另一台机器永远打不开。
 *
 * 设计取舍：
 *   · **默认关闭**：不设 MYDS_DEBUG_ALLOW_CIDR 时，行为与之前完全一致（只认回环）。
 *     这条很重要 —— 默认放开内网会把「谁在什么时候发了什么请求」暴露给整个网段。
 *   · **只做加法**：白名单只会让更多来源通过，不会让原本通过的失败。
 *   · **不能替代 token**：两者是 AND 关系。白名单放行来源，token 仍要匹配。
 *   · **fail-closed on parse error**：写错的 CIDR 整条忽略并告警，绝不退化成
 *     "0.0.0.0/0"。一个拼错的网段不该变成全网开放。
 *
 * 只支持 IPv4：服务器的监听 socket 是 AF_INET，peer 地址由
 * inet_ntop(AF_INET, ...) 产生，IPv6 来源在这里不会出现（与 is_loopback 同一前提）。
 */
#include "myds_debugstate.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MYDS_ACL_MAX 16

typedef struct {
    uint32_t network;   /* 已按 mask 归一化 */
    uint32_t mask;
} myds_acl_entry_t;

static myds_acl_entry_t g_acl[MYDS_ACL_MAX];
static int g_acl_count = -1;   /* -1 = 尚未解析 */

/* Loopback check over the *IPv4 dotted-quad the server actually produces*.
 *
 * Two corrections over the first version:
 *
 * 1. It accepted "::1" and "::ffff:127.*". Those can never arrive: the listener is
 *    AF_INET, the peer address is read with inet_ntop(AF_INET, ...) into a
 *    char[16]. The branches were unreachable, and a unit test asserted they were
 *    true -- testing a path production cannot take, which reads as IPv6 coverage
 *    that does not exist. Anything non-IPv4 is now rejected explicitly, and the
 *    test asserts that rejection instead.
 *
 * 2. 127.0.0.0/8 is checked numerically rather than by string compare, so
 *    127.0.0.2 (also loopback) is accepted and "127.0.0.1.evil" is not.
 *
 * This is still only a network-layer control. A reverse proxy running on the same
 * host connects from 127.0.0.1 and would pass it -- see the token gate below.
 */
bool myds_debug_is_loopback(const char *ip) {
    if (!ip) return false;
    unsigned a = 0, b = 0, c = 0, d = 0;
    char tail = 0;
    if (sscanf(ip, "%u.%u.%u.%u%c", &a, &b, &c, &d, &tail) != 4) return false;
    if (a > 255u || b > 255u || c > 255u || d > 255u) return false;
    return a == 127u;
}

static bool parse_ipv4(const char *s, uint32_t *out) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    char tail = 0;
    if (!s || sscanf(s, "%u.%u.%u.%u%c", &a, &b, &c, &d, &tail) != 4) return false;
    if (a > 255u || b > 255u || c > 255u || d > 255u) return false;
    *out = (a << 24) | (b << 16) | (c << 8) | d;
    return true;
}

/* 解析一次，缓存。环境变量在进程生命周期内不变（配置文件也是启动时读的）。 */
static void acl_load_once(void) {
    if (g_acl_count >= 0) return;
    g_acl_count = 0;
    const char *env = getenv("MYDS_DEBUG_ALLOW_CIDR");
    if (!env || !*env) return;

    char buf[512];
    snprintf(buf, sizeof(buf), "%s", env);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ' || *tok == '\t') ++tok;
        char *endp = tok + strlen(tok);
        while (endp > tok && (endp[-1] == ' ' || endp[-1] == '\t')) *--endp = '\0';
        if (!*tok) continue;

        char addr[64];
        int prefix = 32;
        const char *slash = strchr(tok, '/');
        if (slash) {
            const size_t n = (size_t)(slash - tok);
            if (n >= sizeof(addr)) { MYDS_LOG_W("debug acl: entry too long, ignored"); continue; }
            memcpy(addr, tok, n);
            addr[n] = '\0';
            char *pe = NULL;
            const long v = strtol(slash + 1, &pe, 10);
            /* 前缀长度必须是纯数字且在 [0,32]：写错就整条丢掉，不猜。 */
            if (!pe || *pe != '\0' || v < 0 || v > 32) {
                MYDS_LOG_W("debug acl: bad prefix length in \"%s\", ignored", tok);
                continue;
            }
            prefix = (int)v;
        } else {
            snprintf(addr, sizeof(addr), "%s", tok);
        }

        uint32_t ip = 0;
        if (!parse_ipv4(addr, &ip)) {
            MYDS_LOG_W("debug acl: not an IPv4 address in \"%s\", ignored", tok);
            continue;
        }
        if (g_acl_count >= MYDS_ACL_MAX) {
            MYDS_LOG_W("debug acl: more than %d entries, rest ignored", MYDS_ACL_MAX);
            break;
        }
        /* 前缀 0 会把整个 IPv4 空间放进来 —— 合法但危险，明确告警。 */
        if (prefix == 0) {
            MYDS_LOG_W("debug acl: \"%s\" allows the entire IPv4 space; "
                       "the token is now the only control", tok);
        }
        const uint32_t mask = prefix == 0 ? 0u : (0xffffffffu << (32 - prefix));
        g_acl[g_acl_count].mask = mask;
        g_acl[g_acl_count].network = ip & mask;
        g_acl_count++;
    }
    if (g_acl_count > 0) {
        MYDS_LOG_I("debug acl: %d source range(s) allowed beyond loopback "
                   "(MYDS_DEBUG_ALLOW_CIDR); the token gate still applies",
                   g_acl_count);
    }
}

bool myds_debug_ip_allowed(const char *ip) {
    if (!ip || !*ip) return false;
    if (myds_debug_is_loopback(ip)) return true;
    acl_load_once();
    if (g_acl_count <= 0) return false;
    uint32_t v = 0;
    if (!parse_ipv4(ip, &v)) return false;   /* 非 IPv4 来源：没有白名单条目能匹配 */
    for (int i = 0; i < g_acl_count; ++i)
        if ((v & g_acl[i].mask) == g_acl[i].network) return true;
    return false;
}

/* 测试用：丢掉缓存，让下一次调用重新读环境变量。 */
void myds_debug_acl_reset_for_test(void) {
    g_acl_count = -1;
    memset(g_acl, 0, sizeof(g_acl));
}
