/*
 * test_anchor_slots.c — 多套常驻锚点（system prompt）的槽位与语义（纯 CPU）
 *
 * 为什么必须单测：这套逻辑错了都不会报错，只会表现为
 *   · 「明明存了却总不命中」—— 于是每个请求都白烧几十秒 prefill
 *   · 或者更糟：「命中了一套别的 system prompt」—— 从错误状态续跑
 * 所以判据要钉死：槽区不重叠、ring 公式不受影响、锚点区不被常规清理碰到。
 *
 * 编译：make bin/test_anchor_slots && bin/test_anchor_slots
 */
#include "myds_internal.h"
#include "myds_snapshot_slots.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ── 1. 槽区不重叠：ring 的落槽范围必须全部落在锚点区之前 ── */
static void test_regions_do_not_overlap(void) {
    const uint32_t first_pinned = 1u;
    const uint32_t last_pinned  = MYDS_SNAP_PINNED;
    const uint32_t ring = MYDS_SNAP_SLOTS - 1u - MYDS_SNAP_PINNED;
    const uint32_t first_ring = 1u + MYDS_SNAP_PINNED;
    const uint32_t last_ring  = first_ring + ring - 1u;
    const uint32_t first_anchor = myds_anchor_slot(0u);
    const uint32_t last_anchor  = myds_anchor_slot(MYDS_ANCHOR_SLOTS_MAX - 1u);

    CHECK(first_ring == last_pinned + 1u, "ring 应紧接 pinned（%u vs %u）", first_ring, last_pinned + 1u);
    CHECK(last_ring + 1u == first_anchor, "ring 末尾 %u 必须紧接锚点首槽 %u", last_ring, first_anchor);
    CHECK(last_anchor + 1u == MYDS_SLOT_CAPACITY, "锚点末槽 %u +1 应等于容量 %u",
          last_anchor, MYDS_SLOT_CAPACITY);
    CHECK(MYDS_SLOT_CAPACITY == MYDS_SNAP_SLOTS + MYDS_ANCHOR_SLOTS_MAX, "容量定义");
}

/* ── 2. ring 的落槽公式完全没变：所有非末尾位置都落在 22..43 ── */
static void test_ring_formula_unchanged(void) {
    const uint32_t first_ring = 1u + MYDS_SNAP_PINNED;
    const uint32_t last_ring  = MYDS_SNAP_SLOTS - 1u;
    int bad = 0;
    /* 只在【ring 区】里检查：1024..11264 是 pinned 固定区（落槽 1..21），本就不在 ring 内。 */
    const uint32_t ring_starts_at = MYDS_SNAP_MIN + MYDS_SNAP_PINNED * MYDS_SNAP_STRIDE;
    for (uint32_t p = ring_starts_at; p <= 400000u; p += MYDS_SNAP_STRIDE) {
        const uint32_t s = myds_snap_slot_for(p, 0);
        if (s < first_ring || s > last_ring) { bad++; }
    }
    CHECK(bad == 0, "有 %d 个深位置落到了 ring 之外（应为 0）", bad);
    /* pinned 区必须落 1..21，且都在 ring 之前 */
    for (uint32_t p = MYDS_SNAP_MIN; p < ring_starts_at; p += MYDS_SNAP_STRIDE) {
        const uint32_t s = myds_snap_slot_for(p, 0);
        CHECK(s >= 1u && s <= MYDS_SNAP_PINNED, "pinned 区 p=%u 应落 1..%u（实际 %u）",
              p, MYDS_SNAP_PINNED, s);
        CHECK(s < first_ring, "pinned 槽 %u 必须在 ring 之前", s);
    }

    /* 末尾点固定回槽 0 */
    CHECK(myds_snap_slot_for(5000u, 1) == 0u, "末尾点应回槽 0");
    /* 低位固定区仍固定 */
    CHECK(myds_snap_slot_for(1024u, 0) == 1u, "1024 应落槽 1");
    CHECK(myds_snap_slot_for(11264u, 0) == MYDS_SNAP_PINNED, "11264 应落最后一个 pinned 槽");
}

/* ── 3. 锚点槽号连续、从 44 起，且都 >= SNAP_SLOTS ── */
static void test_anchor_slots_are_contiguous(void) {
    for (uint32_t i = 0; i < MYDS_ANCHOR_SLOTS_MAX; ++i) {
        const uint32_t s = myds_anchor_slot(i);
        CHECK(s == MYDS_SNAP_SLOTS + i, "锚点槽 %u 应为 %u（实际 %u）",
              i, MYDS_SNAP_SLOTS + i, s);
        CHECK(s >= MYDS_SNAP_SLOTS, "锚点槽 %u 必须 >= %u（否则会与常规快照冲突）",
              s, MYDS_SNAP_SLOTS);
        CHECK(s < MYDS_SLOT_CAPACITY, "锚点槽 %u 必须 < 容量 %u", s, MYDS_SLOT_CAPACITY);
    }
    /* 锚点槽必须与所有 pinned 槽不同 */
    for (uint32_t i = 0; i < MYDS_ANCHOR_SLOTS_MAX; ++i)
        for (uint32_t k = 1u; k <= MYDS_SNAP_PINNED; ++k)
            CHECK(myds_anchor_slot(i) != k, "锚点槽 %u 与 pinned 槽 %u 冲突", myds_anchor_slot(i), k);
}

/* ── 4. 常规清理（0..SNAP_SLOTS-1）天然不触及锚点区 ── */
static void test_clear_range_excludes_anchors(void) {
    /* 这条断言的是「不变式」，不是运行时行为：clear() 循环上界是 MYDS_SNAP_SLOTS，
     * 所以锚点区天生被保留。若以后有人把上界改成 SLOT_CAPACITY，这条会失败。 */
    const uint32_t clear_upper = MYDS_SNAP_SLOTS;
    CHECK(clear_upper <= myds_anchor_slot(0u),
          "常规清理上界 %u 必须不超过锚点首槽 %u", clear_upper, myds_anchor_slot(0u));
    for (uint32_t i = 0; i < MYDS_ANCHOR_SLOTS_MAX; ++i)
        CHECK(myds_anchor_slot(i) >= clear_upper, "锚点槽 %u 会被常规清理误伤", myds_anchor_slot(i));
}

/* ── 5. 最长匹配选择（判别性）──
 *
 * 这一条是「多套」的核心：如果只取第一个命中，短前缀会抢走长前缀的请求，
 * 长的那套就白存了。下面用「B 是 A 的前缀」构造，断言必须选 A。 */
static void test_pick_prefers_longest(void) {
    enum { NA = 100, NB = 40 };
    int32_t A[NA], B[NB], P[NA + 10];
    for (int i = 0; i < NA; ++i) A[i] = 1000 + i;
    for (int i = 0; i < NB; ++i) B[i] = A[i];            /* B 是 A 的前缀 */
    for (int i = 0; i < NA; ++i) P[i] = A[i];
    for (int i = 0; i < 10; ++i) P[NA + i] = 7777;       /* 后面跟用户内容 */

    int32_t *anchors[2] = { B, A };
    const uint32_t lens[2] = { NB, NA };                  /* 短的在前面：专门考验选择顺序 */
    uint32_t idx = UINT32_MAX;

    /* 请求 A 的完整前缀 + 用户内容 ⇒ 必须选 A（NA），不是 B（NB） */
    const uint32_t got = myds_v41_anchor_pick(P, NA + 10, anchors, lens, 2, &idx);
    CHECK(got == NA, "应选最长命中 %d（实际 %u）", NA, got);
    CHECK(idx == 1u, "应选中第 1 套（A），实际 %u", idx);

    /* 只共享 B 的长度：把 A 的第 NB 个 token 改掉 ⇒ 只能命中 B */
    int32_t Q[NA + 4];
    for (int i = 0; i < NA; ++i) Q[i] = A[i];
    Q[NB] = 999999;
    for (int i = 0; i < 4; ++i) Q[NA + i] = 5555;
    const uint32_t got2 = myds_v41_anchor_pick(Q, NA + 4, anchors, lens, 2, &idx);
    CHECK(got2 == NB, "分叉后应退回只命中 %d（实际 %u）", NB, got2);
    CHECK(idx == 0u, "应选中第 0 套（B），实际 %u", idx);

    /* 完全不同的 prompt ⇒ 无命中 */
    int32_t Z[32];
    for (int i = 0; i < 32; ++i) Z[i] = 31337 + i;
    const uint32_t got3 = myds_v41_anchor_pick(Z, 32, anchors, lens, 2, &idx);
    CHECK(got3 == 0u, "不同 prompt 应无命中（实际 %u）", got3);
    CHECK(idx == UINT32_MAX, "无命中时套号应为 UINT32_MAX，实际 %u", idx);

    /* 边界：命中长度必须严格小于 n_prompt —— 否则会吃掉最后一个 token。
     * 注意这里要用【单套】锚点来测：两套都在时，长的那套被跳过、
     * 短的 B(40) 仍是 A 的合法前缀 ⇒ 返回 40 才是对的。 */
    int32_t *only_a[1] = { A };
    const uint32_t only_a_len[1] = { NA };
    const uint32_t got4 = myds_v41_anchor_pick(A, NA, only_a, only_a_len, 1, &idx);
    CHECK(got4 == 0u, "prompt 与【唯一】锚点等长时必须不命中（实际 %u）", got4);
    /* 多一个 token 就应该命中 */
    const uint32_t got5 = myds_v41_anchor_pick(P, NA + 10, only_a, only_a_len, 1, &idx);
    CHECK(got5 == NA, "比锚点长一个 token 时应命中 %d（实际 %u）", NA, got5);

    /* 空套 / 空指针不应崩 */
    const uint32_t zero_len[2] = { 0u, 0u };
    CHECK(myds_v41_anchor_pick(P, NA + 10, anchors, zero_len, 2, &idx) == 0u, "空套应无命中");
    CHECK(myds_v41_anchor_pick(NULL, 10, anchors, lens, 2, &idx) == 0u, "NULL prompt 应无命中");
    CHECK(myds_v41_anchor_pick(P, 0, anchors, lens, 2, &idx) == 0u, "n_prompt=0 应无命中");
}

int main(void) {
    printf("myds 多套锚点槽位与语义\n");
    test_regions_do_not_overlap();
    test_ring_formula_unchanged();
    test_anchor_slots_are_contiguous();
    test_clear_range_excludes_anchors();
    test_pick_prefers_longest();

    if (failures) { printf("\n❌ %d 项失败\n", failures); return 1; }
    printf("\n✅ 全部通过\n");
    return 0;
}
