/*
 * myds_lowkv_policy.h — 低位区落盘的【选位策略】（纯函数，便于单测）
 *
 * 为什么单独抽出来：这段逻辑住在 v41 里，而 v41 需要模型与 Metal 才能跑 ——
 * 那就意味着「第二个前缀会不会把第一个顶掉」这种问题只能上机试，出错也只会表现为
 * 「命中率莫名变低」（静默）。抽成纯函数之后可以拿合成状态直接验。
 *
 * ── 背景：为什么需要 way ──
 * 固定槽是按【位置】分的（1024、1536、…、11264，共 MYDS_SNAP_PINNED 个），而它们不是「21 个前缀」，
 * 是【同一个前缀的 21 个深度层】。所以「一个位置一条」等于「一个前缀独占全部槽」：
 * 第二个前缀进来会把第一个整个顶掉，两者交替出现时互相挤没（每个请求都全量重算）。
 *
 * 一组 way 装的是【同一个位置】，区别只在【前缀哈希】。于是：
 *   同一组里哈希相同 ⇒ 同一条前缀 ⇒ 正是它，原样用
 *   哈希不同       ⇒ 另一条前缀 ⇒ 另找一个 way，别动人家的
 *
 * ── 优先级（顺序即优先级）──
 *   ① 同位置同哈希 ⇒ 已有，不必重写
 *   ② 空位         ⇒ 直接用     ★ 这一条让第二个前缀与第一个【共存】而不是顶掉它
 *   ③ 全占满       ⇒ 淘汰命中数最少的
 *        「命中过」比「刚写进来的」更值得保留 —— 它证明这份快照真的在被用。
 *        （ds4 的 KV_CACHE_MIN_EFFECTIVE_HITS 是同一个判断。）
 */
#ifndef MYDS_LOWKV_POLICY_H
#define MYDS_LOWKV_POLICY_H

#include <stdbool.h>
#include <stdint.h>

#define MYDS_LOWKV_NO_WAY 0xFFFFFFFFu

typedef struct {
    bool     occupied;
    uint32_t position;
    uint64_t hash;
    uint32_t hits;
} myds_lowkv_way_t;

/* 挑一个 way 来写。返回 way 下标；`*already_here` = true 表示这条前缀已在盘上
 * （调用方应当跳过，不做任何写入）。没有可用 way 时返回 MYDS_LOWKV_NO_WAY
 * （只有 n_ways == 0 才会发生）。 */
static inline uint32_t myds_lowkv_pick_way(const myds_lowkv_way_t *ways, uint32_t n_ways,
                                           uint32_t position, uint64_t hash,
                                           bool *already_here) {
    if (already_here) *already_here = false;
    if (!ways || n_ways == 0) return MYDS_LOWKV_NO_WAY;

    /* ① 同位置同哈希 —— 就是它本身 */
    for (uint32_t w = 0; w < n_ways; w++) {
        if (ways[w].occupied && ways[w].position == position && ways[w].hash == hash) {
            if (already_here) *already_here = true;
            return w;
        }
    }
    /* ② 空位 —— 优先用空位，这样第二条前缀不会去动第一条 */
    for (uint32_t w = 0; w < n_ways; w++)
        if (!ways[w].occupied) return w;

    /* ③ 全占满 —— 淘汰命中最少的（并列时取下标小的，行为确定、可复现） */
    uint32_t best = 0, best_hits = ways[0].hits;
    for (uint32_t w = 1; w < n_ways; w++) {
        if (ways[w].hits < best_hits) { best_hits = ways[w].hits; best = w; }
    }
    return best;
}

#endif /* MYDS_LOWKV_POLICY_H */
