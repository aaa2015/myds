/* tests/test_debug_acl.c — 调试端点的来源白名单
 *
 * 这个文件守的是一件很具体的事：**默认不放宽**。
 *
 * /trace 与 /debug/state 会暴露「谁在什么时候发了什么请求」。加白名单是为了让
 * 内网能打开页面，但如果实现写错（比如空变量当成 0.0.0.0/0、或写坏的条目被忽略后
 * 反而全部放行），结果就是把运行数据交给整个网段。所以每条都断言两件事：
 * 该放行的放行，以及【不该放行的仍然拒绝】。
 */
#include "myds_debugstate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    printf("[FAIL] %s:%d  %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

static void set_cidr(const char *v) {
    if (v) setenv("MYDS_DEBUG_ALLOW_CIDR", v, 1);
    else unsetenv("MYDS_DEBUG_ALLOW_CIDR");
    myds_debug_acl_reset_for_test();
}

static void test_default_is_loopback_only(void) {
    set_cidr(NULL);
    CHECK(myds_debug_ip_allowed("127.0.0.1"));
    CHECK(myds_debug_ip_allowed("127.0.0.2"));      /* 127/8 整段都是回环 */
    CHECK(!myds_debug_ip_allowed("192.168.88.4"));  /* ← 默认必须拒绝 */
    CHECK(!myds_debug_ip_allowed("10.0.0.1"));
    CHECK(!myds_debug_ip_allowed(NULL));
    CHECK(!myds_debug_ip_allowed(""));
}

static void test_cidr_allows_lan(void) {
    set_cidr("192.168.88.0/24");
    CHECK(myds_debug_ip_allowed("127.0.0.1"));      /* 回环仍然通过（加法） */
    CHECK(myds_debug_ip_allowed("192.168.88.4"));
    CHECK(myds_debug_ip_allowed("192.168.88.192"));
    CHECK(!myds_debug_ip_allowed("192.168.89.4"));  /* 邻段必须拒绝 */
    CHECK(!myds_debug_ip_allowed("192.169.88.4"));
}

static void test_exact_host(void) {
    set_cidr("10.1.2.3");
    CHECK(myds_debug_ip_allowed("10.1.2.3"));
    CHECK(!myds_debug_ip_allowed("10.1.2.4"));      /* /32 只放行一个地址 */
}

static void test_multiple_entries(void) {
    set_cidr("192.168.88.0/24, 10.0.0.0/8 ,172.16.0.0/12");
    CHECK(myds_debug_ip_allowed("192.168.88.9"));
    CHECK(myds_debug_ip_allowed("10.9.9.9"));
    CHECK(myds_debug_ip_allowed("172.16.5.5"));
    CHECK(!myds_debug_ip_allowed("172.32.0.1"));
}

/* 拼错的条目【整条丢掉】，而不是退化成全网放行。 */
static void test_malformed_entries_do_not_widen(void) {
    set_cidr("192.168.88.0/24,garbage");
    CHECK(myds_debug_ip_allowed("192.168.88.4"));
    CHECK(!myds_debug_ip_allowed("8.8.8.8"));       /* garbage 没变成 0/0 */

    set_cidr("999.1.1.1/24");
    CHECK(!myds_debug_ip_allowed("8.8.8.8"));       /* 非法地址 ⇒ 零条目 */
    CHECK(myds_debug_ip_allowed("127.0.0.1"));      /* 回环不受影响 */

    set_cidr("10.0.0.0/99");                        /* 前缀越界 */
    CHECK(!myds_debug_ip_allowed("10.0.0.1"));

    set_cidr("10.0.0.0/abc");
    CHECK(!myds_debug_ip_allowed("10.0.0.1"));

    set_cidr("");                                   /* 空 ⇒ 等同未设 */
    CHECK(!myds_debug_ip_allowed("10.0.0.1"));
}

/* 前缀 0 是合法但危险的：允许，然后靠 token 兜底。这里断言它【确实允许】，
 * 因为有人显式写 0.0.0.0/0 时应得到它要求的东西，而不是被静默改成拒绝。 */
static void test_prefix_zero_is_honoured(void) {
    set_cidr("0.0.0.0/0");
    CHECK(myds_debug_ip_allowed("1.2.3.4"));
    CHECK(myds_debug_ip_allowed("255.255.255.255"));
}

/* 非 IPv4 输入不能意外匹配任何条目 */
static void test_non_ipv4_rejected(void) {
    set_cidr("192.168.88.0/24");
    CHECK(!myds_debug_ip_allowed("::1"));
    CHECK(!myds_debug_ip_allowed("localhost"));
    CHECK(!myds_debug_ip_allowed("192.168.88"));
    CHECK(!myds_debug_ip_allowed("192.168.88.4.5"));
    CHECK(!myds_debug_ip_allowed("192.168.88.4evil"));
}

int main(void) {
    test_default_is_loopback_only();
    test_cidr_allows_lan();
    test_exact_host();
    test_multiple_entries();
    test_malformed_entries_do_not_widen();
    test_prefix_zero_is_honoured();
    test_non_ipv4_rejected();
    set_cidr(NULL);

    if (failures) { printf("\n%d acl test(s) FAILED\n", failures); return 1; }
    printf("PASS: debug ACL defaults to loopback and only widens when told to\n");
    return 0;
}
