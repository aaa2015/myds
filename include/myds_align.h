/*
 * myds_align.h — 图片块区间与最终序列的对齐（纯函数，便于单测）
 *
 * 为什么单独放这里：图片向量的区间是【服务端】按请求里那段文本的下标登记的，
 * 而序列在【引擎】里可能被续接逻辑改过（丢掉 delta 的 BOS、再整体挪到缓存前缀之后）。
 * 两个坐标系一旦不一致，图片向量就会贴到别的 token 上 —— 不报错，只是认错图，
 * 属于最难查的一类。实测就是这样：连续三轮发不同的图，每轮都识别出第一张图。
 *
 * 判据来自图片块自身的性质：它占用的每个位置都是【同一个】图片 token
 * （V4.1 是 129264）。所以
 *   · 「区间内的 token 不全是它」 ⇒ 必然错位；
 *   · 反过来，这也让正确位置是可恢复的 —— 找长度【恰好】等于块长的连续图片 token。
 *
 * 纯函数、无副作用、不依赖引擎状态 ⇒ 可以直接喂合成序列做单测。
 */
#ifndef MYDS_ALIGN_H
#define MYDS_ALIGN_H

#include <stdint.h>
#include <stdbool.h>

/* 区间内每个位置是否都正好是图片 token。 */
static inline bool myds_image_range_aligned(const int32_t *tokens, uint32_t n_tokens,
                                            uint32_t img_token_id, uint32_t start,
                                            uint32_t count) {
    if (!tokens || img_token_id == 0 || count == 0) return false;
    if ((uint64_t)start + count > (uint64_t)n_tokens) return false;
    for (uint32_t k = 0; k < count; k++) {
        if ((uint32_t)tokens[start + k] != img_token_id) return false;
    }
    return true;
}

/* 在 tokens[scan_from..n_tokens) 里找一段长度【恰好】为 count 的连续图片 token，
 * 返回其起点；找不到返回 -1。
 *
 * 「恰好」很重要：相邻图片块之间没有其它 token 隔开时，一段更长的连续图片 token
 * 会被误当成某个块 —— 所以【两侧】边界都必须卡住（除非左边界正好是搜索起点）。 */
static inline int64_t myds_image_range_locate(const int32_t *tokens, uint32_t n_tokens,
                                              uint32_t img_token_id, uint32_t count,
                                              uint32_t scan_from) {
    if (!tokens || img_token_id == 0 || count == 0) return -1;
    for (uint32_t pos = scan_from; (uint64_t)pos + count <= (uint64_t)n_tokens; pos++) {
        if ((uint32_t)tokens[pos] != img_token_id) continue;
        bool all = true;
        for (uint32_t k = 1; k < count; k++) {
            if ((uint32_t)tokens[pos + k] != img_token_id) { all = false; break; }
        }
        if (!all) continue;
        /* ★ 两个边界都要卡住「恰好 count 个」，只卡右边不够 ——
         * 一段更长的连续图片 token（比如 12 个）里，仍然能找到若干满足
         * 「右边不是图片 token」的 3~6 长片段，会把块定位到错的位置。
         * （这条是单测发现的：它先写错了，测试当场抓住。）
         * 左边界的例外：允许正好从 scan_from 开始 —— 上一块的末尾就在 scan_from-1，
         * 那里必然是图片 token，不能因此把紧邻的块判掉。 */
        if (pos != scan_from && (uint32_t)tokens[pos - 1] == img_token_id) continue;
        if ((uint64_t)pos + count < (uint64_t)n_tokens &&
            (uint32_t)tokens[pos + count] == img_token_id) {
            continue;   /* 落在更长的块里，不是本块 */
        }
        return (int64_t)pos;
    }
    return -1;
}

/* 声明的区间若已对齐则原样返回；否则按实际 token 位置恢复；找不到返回 -1。 */
static inline int64_t myds_image_range_resolve(const int32_t *tokens, uint32_t n_tokens,
                                               uint32_t img_token_id, uint32_t declared_start,
                                               uint32_t count, uint32_t scan_from) {
    if (myds_image_range_aligned(tokens, n_tokens, img_token_id, declared_start, count)) {
        return (int64_t)declared_start;
    }
    return myds_image_range_locate(tokens, n_tokens, img_token_id, count, scan_from);
}

#endif /* MYDS_ALIGN_H */
