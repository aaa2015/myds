/*
 * test_kvdsk.c — 低位区 KV 落盘的存储层单测（不需要模型）
 *
 * 为什么这些必须单测：这一层的错误【都不会报错】，只会表现为
 *   · 「本该命中却总是不命中」（quiet 退化，最难发现）
 *   · 或者更糟：「命中了但字节是坏的」⇒ 接在错的状态上继续生成
 * 所以判据要逐条钉死：提交协议、校验和、身份失效、越界拒绝。
 *
 * 编译运行：cc -O1 -Iinclude tests/test_kvdsk.c src/myds_kvdsk.c src/myds_log.c -o /tmp/t && /tmp/t
 */
#include "myds_kvdsk.h"
#include "myds_lowkv_policy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, ...)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            failures++;                                    \
            printf("  ✗ %s:%d  ", __FILE__, __LINE__);     \
            printf(__VA_ARGS__);                           \
            printf("\n");                                  \
        }                                                  \
    } while (0)

static char g_path[256];

static void fresh_path(const char *tag, size_t sz) {
    snprintf(g_path, sizeof(g_path), "/tmp/myds_kvdsk_test/%s_%d.bin", tag, (int)getpid());
    unlink(g_path);
    (void)sz;
}

#define SLOTS 3u
#define STRIDE 4096u

static void fill(unsigned char *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)((i * 31u + seed * 7u) & 0xFFu);
}

static void test_hash_is_stable_and_discriminating(void) {
    const int32_t a[4] = {11, 22, 33, 44};
    const int32_t b[4] = {11, 22, 33, 45};
    const uint64_t ha1 = myds_kvdsk_hash_tokens(a, 4);
    const uint64_t ha2 = myds_kvdsk_hash_tokens(a, 4);
    const uint64_t hb = myds_kvdsk_hash_tokens(b, 4);
    CHECK(ha1 == ha2, "同一条序列的哈希必须稳定（跨进程复用的唯一安全性来源）");
    CHECK(ha1 != hb, "不同序列必须得到不同哈希");
    /* 前缀性质：长度不同就是不同哈希（1024 与 1025 个 token 不能混） */
    CHECK(myds_kvdsk_hash_tokens(a, 3) != myds_kvdsk_hash_tokens(a, 4),
          "不同长度必须是不同哈希");
}

static void test_identity_discriminates(void) {
    const uint64_t v1[3] = {40, 128, 512};
    const uint64_t v2[3] = {40, 128, 513};
    CHECK(myds_kvdsk_identity("v41", v1, 3) == myds_kvdsk_identity("v41", v1, 3), "同参数同身份");
    CHECK(myds_kvdsk_identity("v41", v1, 3) != myds_kvdsk_identity("v41", v2, 3), "布局变了必须换身份");
    CHECK(myds_kvdsk_identity("v41", v1, 3) != myds_kvdsk_identity("v42", v1, 3), "tag 变了必须换身份");
}

static void test_roundtrip(void) {
    fresh_path("rt", 0);
    myds_kvdsk_t *k = myds_kvdsk_open(g_path, 0xABCDEFull, SLOTS, STRIDE, 0);
    CHECK(k != NULL, "open 失败: %s", myds_kvdsk_open_error());
    if (!k) return;

    unsigned char w[1000], r[1000];
    fill(w, sizeof(w), 3);
    CHECK(myds_kvdsk_write_start(k, 1), "write_start");
    CHECK(myds_kvdsk_write_chunk(k, 0, w, sizeof(w)), "write_chunk");
    CHECK(myds_kvdsk_write_finish(k, 1, 4608u, 0x1122334455667788ull, sizeof(w)), "write_finish");

    uint32_t pos = 0; uint64_t h = 0, used = 0;
    CHECK(myds_kvdsk_slot_meta(k, 1, &pos, &h, &used), "slot_meta");
    CHECK(pos == 4608u, "position 应为 4608，实际 %u", pos);
    CHECK(h == 0x1122334455667788ull, "prefix_hash 不符");
    CHECK(used == sizeof(w), "used_bytes 应为 %zu，实际 %llu", sizeof(w), (unsigned long long)used);

    uint64_t got_used = 0;
    CHECK(myds_kvdsk_read_start(k, 1, &got_used), "read_start");
    memset(r, 0, sizeof(r));
    CHECK(myds_kvdsk_read_chunk(k, 0, r, sizeof(w)), "read_chunk");
    CHECK(myds_kvdsk_read_finish(k), "read_finish");
    CHECK(memcmp(w, r, sizeof(w)) == 0, "读回的数据必须逐字节相同");
    myds_kvdsk_close(k);
}

static void test_empty_slot_is_a_clean_miss(void) {
    fresh_path("empty", 0);
    myds_kvdsk_t *k = myds_kvdsk_open(g_path, 1u, SLOTS, STRIDE, 0);
    if (!k) { CHECK(0, "open 失败"); return; }
    uint64_t used = 0;
    CHECK(!myds_kvdsk_read_start(k, 0, &used), "空槽必须读不到（返回 false，而不是读到垃圾）");
    uint32_t pos = 999;
    CHECK(!myds_kvdsk_slot_meta(k, 0, &pos, NULL, NULL), "空槽没有元信息");
    CHECK(pos == 999, "失败的 slot_meta 不该改写出参");
    myds_kvdsk_close(k);
}

/* 崩溃安全：write_start 之后中途放弃（模拟掉电）⇒ 该槽必须是「空」，不是「半份数据」。 */
static void test_abandoned_write_leaves_slot_empty(void) {
    fresh_path("torn", 0);
    myds_kvdsk_t *k = myds_kvdsk_open(g_path, 2u, SLOTS, STRIDE, 0);
    if (!k) { CHECK(0, "open 失败"); return; }
    unsigned char w[500];
    fill(w, sizeof(w), 9);
    CHECK(myds_kvdsk_write_start(k, 2), "write_start");
    CHECK(myds_kvdsk_write_chunk(k, 0, w, sizeof(w)), "write_chunk");
    /* 这里【不】调 write_finish —— 相当于写到一半掉电 */
    myds_kvdsk_close(k);

    myds_kvdsk_t *k2 = myds_kvdsk_open(g_path, 2u, SLOTS, STRIDE, 0);
    if (!k2) { CHECK(0, "重开失败"); return; }
    uint64_t used = 0;
    CHECK(!myds_kvdsk_read_start(k2, 2, &used),
          "未提交的半份数据必须读不到（宁可 miss，也不能接在坏状态上）");
    myds_kvdsk_close(k2);
}

/* 坏数据：把 payload 改一个字节 ⇒ 读必须失败，并且该槽被作废（第二次读也读不到）。 */
static void test_corrupt_payload_is_rejected_and_invalidated(void) {
    fresh_path("corrupt", 0);
    myds_kvdsk_t *k = myds_kvdsk_open(g_path, 3u, SLOTS, STRIDE, 0);
    if (!k) { CHECK(0, "open 失败"); return; }
    unsigned char w[800], r[800];
    fill(w, sizeof(w), 5);
    CHECK(myds_kvdsk_write_start(k, 0), "write_start");
    CHECK(myds_kvdsk_write_chunk(k, 0, w, sizeof(w)), "write_chunk");
    CHECK(myds_kvdsk_write_finish(k, 0, 2048u, 0xDEADBEEFull, sizeof(w)), "write_finish");
    myds_kvdsk_close(k);

    /* 越过存储层直接破坏 payload（模拟介质损坏 / payload 与条目落盘不一致） */
    FILE *f = fopen(g_path, "r+b");
    CHECK(f != NULL, "重开文件失败");
    if (f) {
        /* payload 起点 = 头(4096) + 条目表(slots*64) 再按 4096 对齐 —— 与实现的
         * 布局一致。用真实偏移才能真的打到 payload 上（我第一次写错了偏移，
         * 打在了条目表与 payload 之间的填充区，于是「损坏」根本没发生）。 */
        const long payload_base = ((4096 + (long)SLOTS * 64 + 4095) / 4096) * 4096;
        fseek(f, payload_base + 200, SEEK_SET);
        unsigned char c = 0x5A; fwrite(&c, 1, 1, f);
        fclose(f);
    }

    myds_kvdsk_t *k2 = myds_kvdsk_open(g_path, 3u, SLOTS, STRIDE, 0);
    if (!k2) { CHECK(0, "重开失败"); return; }
    uint64_t used = 0;
    CHECK(myds_kvdsk_read_start(k2, 0, &used), "read_start 应该能起来（读的时候才知道坏没坏）");
    memset(r, 0, sizeof(r));
    CHECK(myds_kvdsk_read_chunk(k2, 0, r, sizeof(w)), "read_chunk");
    CHECK(!myds_kvdsk_read_finish(k2), "校验和不符时必须返回 false");
    uint64_t used2 = 0;
    CHECK(!myds_kvdsk_read_start(k2, 0, &used2), "坏条目必须已被作废（否则每次命中都白读）");
    myds_kvdsk_close(k2);
}

/* 身份变了 ⇒ 整份作废（模型或布局改了，旧字节没有意义）。 */
static void test_identity_change_invalidates_file(void) {
    fresh_path("ident", 0);
    myds_kvdsk_t *k = myds_kvdsk_open(g_path, 111u, SLOTS, STRIDE, 0);
    if (!k) { CHECK(0, "open 失败"); return; }
    unsigned char w[300];
    fill(w, sizeof(w), 1);
    CHECK(myds_kvdsk_write_start(k, 0) && myds_kvdsk_write_chunk(k, 0, w, sizeof(w)) &&
          myds_kvdsk_write_finish(k, 0, 1024u, 7u, sizeof(w)), "写入");
    CHECK(myds_kvdsk_live_entries(k) == 1u, "应有 1 条有效");
    myds_kvdsk_close(k);

    myds_kvdsk_t *k2 = myds_kvdsk_open(g_path, 222u, SLOTS, STRIDE, 0);
    if (!k2) { CHECK(0, "重开失败"); return; }
    CHECK(myds_kvdsk_live_entries(k2) == 0u, "身份不符必须整份作废");
    myds_kvdsk_close(k2);
}

/* 越界与参数拒绝 —— 这些如果静默接受，会变成堆溢出或读到别人的数据。 */
static void test_bounds_are_enforced(void) {
    fresh_path("bounds", 0);
    myds_kvdsk_t *k = myds_kvdsk_open(g_path, 4u, SLOTS, STRIDE, 0);
    if (!k) { CHECK(0, "open 失败"); return; }
    unsigned char w[STRIDE];
    fill(w, sizeof(w), 2);
    CHECK(myds_kvdsk_write_start(k, 0), "write_start");
    CHECK(!myds_kvdsk_write_chunk(k, STRIDE - 10u, w, 100u),
          "越过槽容量的写必须被拒（不能静默写进下一个槽）");
    CHECK(!myds_kvdsk_write_finish(k, 0, 1024u, 1u, STRIDE + 1u), "used > stride 必须被拒");
    /* 被拒 = 事务终止（契约），所以重新开一次再写合法的 */
    CHECK(!myds_kvdsk_write_finish(k, 0, 1024u, 1u, 16u), "被拒后事务已终止，不能接着提交");
    CHECK(myds_kvdsk_write_start(k, 0), "重开写事务");
    CHECK(myds_kvdsk_write_chunk(k, 0, w, 16u), "写 16 字节");
    CHECK(myds_kvdsk_write_finish(k, 0, 1024u, 1u, 16u), "合法写入应成功");
    uint64_t used = 0;
    CHECK(!myds_kvdsk_read_start(k, 99, &used), "越界 slot 必须被拒");
    /* 两个事务不能同时开着 */
    CHECK(myds_kvdsk_write_start(k, 1), "write_start 槽1");
    CHECK(!myds_kvdsk_read_start(k, 0, &used), "写事务未结束前不得开读事务");
    myds_kvdsk_close(k);
}

/* 预算不足时打开必须失败并说清原因（而不是偷偷截断槽位）。 */
static void test_budget_is_reported_not_silently_trimmed(void) {
    fresh_path("budget", 0);
    myds_kvdsk_t *k = myds_kvdsk_open(g_path, 5u, 64u, 1u << 20, 1u << 20);
    CHECK(k == NULL, "预算不足必须打开失败");
    CHECK(myds_kvdsk_open_error()[0] != 0, "失败原因不能是空字符串");
}


/* ── 选位策略（纯函数 myds_lowkv_policy.h）──
 * 这四条就是「用 system + tools 作为去重逻辑」的全部行为：
 *   ① 同前缀 ⇒ 认出自己，不重写
 *   ② 不同前缀 + 有空位 ⇒ 各占一个 way【共存】（改之前它会顶掉人家）
 *   ③ way=1 时共存不了 ⇒ 只能顶掉（保留旧行为，作为对照）
 *   ④ 全占满 ⇒ 淘汰命中最少的：「命中过」比「刚写进来的」更值钱
 */
static myds_lowkv_way_t W(bool occ, uint32_t pos, uint64_t hash, uint32_t hits) {
    myds_lowkv_way_t w; w.occupied = occ; w.position = pos; w.hash = hash; w.hits = hits;
    return w;
}

static void test_policy_recognizes_the_same_prefix(void) {
    const myds_lowkv_way_t ways[3] = { W(true, 4608, 0xAAAA, 0), W(true, 4608, 0xBBBB, 7), W(false,0,0,0) };
    bool already = false;
    const uint32_t w = myds_lowkv_pick_way(ways, 3, 4608, 0xBBBB, &already);
    CHECK(already, "同位置同哈希必须被认出来（already_here=true）");
    CHECK(w == 1u, "并且指向它自己那一格（实际 %u）", w);
}

static void test_policy_gives_a_different_prefix_its_own_way(void) {
    /* ★ 核心：第二个前缀必须去用【空位】，而不是把第一个顶掉。 */
    const myds_lowkv_way_t ways[3] = { W(true, 4608, 0xAAAA, 0), W(false,0,0,0), W(false,0,0,0) };
    bool already = false;
    const uint32_t w = myds_lowkv_pick_way(ways, 3, 4608, 0xCCCC, &already);
    CHECK(!already, "不同前缀不该被当成已有条目");
    CHECK(w == 1u, "应当去用第一个空位（way 1），而不是顶掉 way 0（实际 %u）", w);
}

static void test_policy_with_one_way_must_displace(void) {
    /* ways=1 时没有共存空间 —— 这就是改之前的行为，留作对照。 */
    const myds_lowkv_way_t ways[1] = { W(true, 4608, 0xAAAA, 5) };
    bool already = false;
    const uint32_t w = myds_lowkv_pick_way(ways, 1, 4608, 0xCCCC, &already);
    CHECK(w == 0u && !already, "只有一个 way 时只能覆盖它");
}

static void test_policy_evicts_the_least_hit_when_full(void) {
    const myds_lowkv_way_t ways[3] = { W(true, 4608, 1, 9), W(true, 4608, 2, 0), W(true, 4608, 3, 4) };
    bool already = false;
    const uint32_t w = myds_lowkv_pick_way(ways, 3, 4608, 0x99, &already);
    CHECK(w == 1u, "全占满时淘汰命中最少的（way 1，hits=0），实际 %u", w);
    /* 并列时取下标小的 —— 行为要确定、可复现 */
    const myds_lowkv_way_t tie[3] = { W(true, 4608, 1, 2), W(true, 4608, 2, 2), W(true, 4608, 3, 2) };
    CHECK(myds_lowkv_pick_way(tie, 3, 4608, 0x99, NULL) == 0u, "并列时必须确定地取 way 0");
}

static void test_policy_handles_no_way(void) {
    bool already = false;
    CHECK(myds_lowkv_pick_way(NULL, 0, 4608, 1, &already) == MYDS_LOWKV_NO_WAY,
          "没有 way 时必须明确返回 NO_WAY（不能瞎返回 0）");
}

int main(void) {
    printf("myds 低位区 KV 落盘（存储层）\n");
    test_hash_is_stable_and_discriminating();
    test_identity_discriminates();
    test_roundtrip();
    test_empty_slot_is_a_clean_miss();
    test_abandoned_write_leaves_slot_empty();
    test_corrupt_payload_is_rejected_and_invalidated();
    test_identity_change_invalidates_file();
    test_bounds_are_enforced();
    test_budget_is_reported_not_silently_trimmed();
    test_policy_recognizes_the_same_prefix();
    test_policy_gives_a_different_prefix_its_own_way();
    test_policy_with_one_way_must_displace();
    test_policy_evicts_the_least_hit_when_full();
    test_policy_handles_no_way();
    if (failures == 0) { printf("  ✓ 13 组断言全部通过\n"); return 0; }
    printf("  ✗ %d 处断言失败\n", failures);
    return 1;
}
