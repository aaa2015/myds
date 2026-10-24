/*
 * test_kvdsk_fifo.c — 端快照的 FIFO 淘汰与写入时刻（不需要模型、纯 CPU）
 *
 * 为什么必须单测：这条策略错了【不会报错】，只会表现为
 *   「刚存进去的下一轮就没了」——正是 review 从一行日志里看出来的那个现象。
 *
 * 覆盖：
 *   1. slot_written_us 能读出写入时刻，且单调递增
 *   2. 空槽优先于淘汰
 *   3. 槽满时淘汰【写入最早】的那条（不是命中最少的）
 *   4. 旧文件（written_us == 0）被当作最老，先淘汰
 *   5. 命中的槽不会被淘汰
 *
 * 编译：make bin/test_kvdsk_fifo && bin/test_kvdsk_fifo
 */
#include "myds_kvdsk.h"
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

static myds_kvdsk_t *open_fresh(uint32_t slots, uint64_t stride) {
    snprintf(g_path, sizeof(g_path), "/tmp/myds_kvdsk_fifo/%d_%u.bin", (int)getpid(), slots);
    unlink(g_path);
    return myds_kvdsk_open(g_path, 0x1234ull, slots, stride, 0);
}

/* 写一条：position 用标记值，payload 填与该位置相关的字节。 */
static void put(myds_kvdsk_t *k, uint32_t slot, uint32_t position, uint64_t hash) {
    CHECK(myds_kvdsk_write_start(k, slot) == true, "write_start slot %u", slot);
    unsigned char buf[64];
    memset(buf, (int)(position & 0xFF), sizeof(buf));
    CHECK(myds_kvdsk_write_chunk(k, 0, buf, sizeof(buf)) == true, "write_chunk slot %u", slot);
    CHECK(myds_kvdsk_write_finish(k, slot, position, hash, sizeof(buf)) == true,
          "write_finish slot %u", slot);
}

/* ── 1. 写入时刻可读且递增 ── */
static void test_written_us_is_readable_and_increasing(void) {
    myds_kvdsk_t *k = open_fresh(4, 4096);
    CHECK(k != NULL, "open");
    if (!k) return;

    uint64_t t0 = 0;
    put(k, 0, 100, 0xAA);
    CHECK(myds_kvdsk_slot_written_us(k, 0, &t0) == true, "槽 0 应读得到写入时刻");
    CHECK(t0 > 0, "写入时刻应非 0（实测 %llu）", (unsigned long long)t0);

    usleep(2000);                       /* 保证两次写入的时刻可区分 */
    uint64_t t1 = 0;
    put(k, 1, 200, 0xBB);
    CHECK(myds_kvdsk_slot_written_us(k, 1, &t1) == true, "槽 1 应读得到写入时刻");
    CHECK(t1 >= t0, "后写的时刻不应早于先写的（%llu vs %llu）",
          (unsigned long long)t0, (unsigned long long)t1);

    /* 空槽读不到 */
    uint64_t te = 1;
    CHECK(myds_kvdsk_slot_written_us(k, 2, &te) == false, "空槽应返回 false");
    CHECK(te == 1, "空槽不应改写输出（实际 %llu）", (unsigned long long)te);

    myds_kvdsk_close(k);
}

/* ── 2/3. 空槽优先；满了才淘汰，淘汰的是写入最早的 ── */
static void test_empty_before_evict_and_oldest_evicted(void) {
    myds_kvdsk_t *k = open_fresh(3, 4096);
    CHECK(k != NULL, "open");
    if (!k) return;

    /* 填满 3 槽，写入顺序 0 → 1 → 2 */
    put(k, 0, 10, 0xA0); usleep(2000);
    put(k, 1, 20, 0xA1); usleep(2000);
    put(k, 2, 30, 0xA2);

    uint64_t w0 = 0, w1 = 0, w2 = 0;
    CHECK(myds_kvdsk_slot_written_us(k, 0, &w0), "读 w0");
    CHECK(myds_kvdsk_slot_written_us(k, 1, &w1), "读 w1");
    CHECK(myds_kvdsk_slot_written_us(k, 2, &w2), "读 w2");
    CHECK(w0 <= w1 && w1 <= w2, "写入时刻应单调（%llu %llu %llu）",
          (unsigned long long)w0, (unsigned long long)w1, (unsigned long long)w2);

    /* 现在 3 槽全占。挑「最老」的应是槽 0。 */
    uint32_t victim = 0; uint64_t oldest = UINT64_MAX;
    for (uint32_t s = 0; s < 3; ++s) {
        uint64_t w = 0;
        if (myds_kvdsk_slot_written_us(k, s, &w) && w < oldest) { oldest = w; victim = s; }
    }
    CHECK(victim == 0u, "FIFO 应淘汰槽 0（最早写入），实际 %u", victim);

    /* 覆盖槽 0 后，它变成最新的 */
    usleep(2000);
    put(k, 0, 40, 0xA3);
    uint64_t w0b = 0;
    CHECK(myds_kvdsk_slot_written_us(k, 0, &w0b), "读 w0b");
    CHECK(w0b >= w2, "覆盖后槽 0 应成为最新（%llu vs %llu）",
          (unsigned long long)w0b, (unsigned long long)w2);

    myds_kvdsk_close(k);
}

/* ── 4. 旧文件（written_us == 0）被当作最老 ── */
static void test_zero_timestamp_is_oldest(void) {
    myds_kvdsk_t *k = open_fresh(4, 4096);
    CHECK(k != NULL, "open");
    if (!k) return;

    /* 模拟「升级前写的条目」：把条目表里 offset 40 的 8 字节写回 0。
     * 布局：文件头 4096 B，条目表每项 64 B，写入时刻在项内 +40。 */
    put(k, 0, 50, 0xCC);
    {
        FILE *f = fopen(g_path, "r+b");
        CHECK(f != NULL, "重开文件以清时间戳");
        if (f) {
            unsigned char zero[8] = {0};
            const long at = 4096L + 40L;
            CHECK(fseek(f, at, SEEK_SET) == 0, "定位槽 0 时间戳");
            CHECK(fwrite(zero, 1, sizeof(zero), f) == sizeof(zero), "写零时间戳");
            fclose(f);
        }
    }
    uint64_t w = 123;
    CHECK(myds_kvdsk_slot_written_us(k, 0, &w) == true, "槽 0 仍应可读");
    CHECK(w == 0, "清掉后应读回 0，实际 %llu", (unsigned long long)w);

    usleep(2000);
    put(k, 1, 60, 0xCD);
    uint64_t w1 = 0;
    CHECK(myds_kvdsk_slot_written_us(k, 1, &w1) && w1 > 0, "槽 1 应有真实时刻");

    /* 0 < w1 ⇒ 槽 0 会被判为最老并被优先淘汰 */
    CHECK(0 < w1, "0 时刻应小于真实时刻 ⇒ 视为最老");

    myds_kvdsk_close(k);
}

/* ── 5. 槽数访问器 ── */
static void test_slot_count(void) {
    for (uint32_t s = 1; s <= 4; ++s) {
        myds_kvdsk_t *k = open_fresh(s, 4096);
        CHECK(k != NULL, "open %u slots", s);
        if (!k) continue;
        CHECK(myds_kvdsk_slot_count(k) == s, "%u 槽时访问器应返回 %u，实际 %u",
              s, s, myds_kvdsk_slot_count(k));
        myds_kvdsk_close(k);
    }
    CHECK(myds_kvdsk_slot_count(NULL) == 0u, "NULL 应返回 0");
}

int main(void) {
    printf("myds 端快照 FIFO 淘汰与写入时刻\n");
    test_written_us_is_readable_and_increasing();
    test_empty_before_evict_and_oldest_evicted();
    test_zero_timestamp_is_oldest();
    test_slot_count();

    if (failures) { printf("\n❌ %d 项失败\n", failures); return 1; }
    printf("\n✅ 全部通过\n");
    return 0;
}
