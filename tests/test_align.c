/*
 * test_align.c — 图片块区间对齐的单元测试（纯函数，不需要模型）
 *
 * 背景：连续发多张不同的图，模型每轮都识别出【第一张】。根因是续接时序列被
 * 前置了缓存前缀、又丢掉了 delta 的 BOS，而图片向量的区间还是按 delta 的下标登记，
 * 于是向量整体贴早了，正好落到上一次那张图的位置上。
 *
 * 这个测试守住两件事：
 *   ① 声明的位置对得上时，原样采信（不能被"纠正"歪）；
 *   ② 对不上时，能按图片 token 的实际位置恢复（这正是让 bug 现形的那条路）。
 * 用的是合成序列 —— 不需要加载模型，也就没有 155GB 常驻实例的风险。
 *
 * 编译运行：make test-align   或   cc -Iinclude tests/test_align.c -o /tmp/t && /tmp/t
 */
#include "myds_align.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, ...)                                            \
    do {                                                            \
        if (!(cond)) {                                              \
            failures++;                                             \
            printf("  ✗ %s:%d  ", __FILE__, __LINE__);              \
            printf(__VA_ARGS__);                                    \
            printf("\n");                                           \
        }                                                           \
    } while (0)

#define IMG 129264
#define OTHER 7

static void test_aligned_range_is_kept(void) {
    /* 序列： [前缀若干] + 6 个图片 token + [后续]
     * 声明位置正好落在图片块上 ⇒ 必须原样采信。 */
    int32_t t[20];
    for (int i = 0; i < 20; i++) t[i] = OTHER;
    for (int i = 5; i < 11; i++) t[i] = IMG;

    CHECK(myds_image_range_aligned(t, 20, IMG, 5, 6), "区间应判定为已对齐");
    CHECK(myds_image_range_resolve(t, 20, IMG, 5, 6, 0) == 5, "已对齐的区间不该被改动");
}

static void test_misaligned_declaration_is_corrected(void) {
    /* 复刻线上那次：序列被前置了 pl 个 token，图片块实际在 pl + 5。
     * 声明位置还停在 5（delta 坐标系）⇒ 必须被纠正到真实位置。 */
    const uint32_t pl = 19;
    int32_t t[64];
    for (int i = 0; i < 64; i++) t[i] = OTHER;
    for (uint32_t i = 0; i < pl; i++) t[i] = (int32_t)(1000 + i);   /* 缓存前缀 */
    for (uint32_t i = pl + 5; i < pl + 11; i++) t[i] = IMG;

    CHECK(!myds_image_range_aligned(t, 64, IMG, 5, 6), "错位的区间不该被判为对齐");
    const int64_t at = myds_image_range_resolve(t, 64, IMG, 5, 6, 0);
    CHECK(at == (int64_t)pl + 5, "应纠正到 %u，实际 %lld", pl + 5, (long long)at);
}

static void test_two_blocks_resolve_in_order(void) {
    /* 历史里两张图：第一张在缓存前缀里，第二张在增量里。
     * 第二张的声明位置同样错位 ⇒ 从第一张之后开始找，必须找到第二个块。 */
    const uint32_t pl = 19;
    int32_t t[96];
    for (int i = 0; i < 96; i++) t[i] = OTHER;
    for (uint32_t i = 3; i < 9; i++) t[i] = IMG;            /* 块一（缓存前缀内） */
    for (uint32_t i = pl + 20; i < pl + 26; i++) t[i] = IMG; /* 块二 */

    int64_t a = myds_image_range_resolve(t, 96, IMG, 3, 6, 0);
    CHECK(a == 3, "第一个块应在 3，实际 %lld", (long long)a);
    int64_t b = myds_image_range_resolve(t, 96, IMG, 20, 6, (uint32_t)(a + 6));
    CHECK(b == (int64_t)pl + 20, "第二个块应纠正到 %u，实际 %lld", pl + 20, (long long)b);
}

static void test_longer_run_is_not_mistaken_for_a_block(void) {
    /* 一段 12 个连续图片 token，块长声明为 6 ⇒ 不能把它当成某个 6 长的块。 */
    int32_t t[40];
    for (int i = 0; i < 40; i++) t[i] = OTHER;
    for (int i = 4; i < 16; i++) t[i] = IMG;
    CHECK(myds_image_range_locate(t, 40, IMG, 6, 0) == -1,
          "长度不符的连续图片 token 不该被当作块");
    /* 但块长就是 12 时应能定位 */
    CHECK(myds_image_range_locate(t, 40, IMG, 12, 0) == 4, "块长为 12 时应定位到 4");
}

static void test_missing_block_is_reported(void) {
    int32_t t[32];
    for (int i = 0; i < 32; i++) t[i] = OTHER;
    CHECK(myds_image_range_resolve(t, 32, IMG, 8, 6, 0) == -1,
          "序列里没有图片 token 时必须返回 -1（调用方据此拒绝本轮）");
}

static void test_out_of_bounds_declaration_is_rejected(void) {
    int32_t t[10];
    for (int i = 0; i < 10; i++) t[i] = IMG;
    CHECK(!myds_image_range_aligned(t, 10, IMG, 8, 6), "越界区间不该被判为对齐");
    CHECK(myds_image_range_aligned(t, 10, IMG, 4, 6), "恰好贴住末尾应算对齐");
}

int main(void) {
    printf("myds 图片块区间对齐（纯函数）\n");
    test_aligned_range_is_kept();
    test_misaligned_declaration_is_corrected();
    test_two_blocks_resolve_in_order();
    test_longer_run_is_not_mistaken_for_a_block();
    test_missing_block_is_reported();
    test_out_of_bounds_declaration_is_rejected();
    if (failures == 0) {
        printf("  ✓ 6 组断言全部通过\n");
        return 0;
    }
    printf("  ✗ %d 处断言失败\n", failures);
    return 1;
}
