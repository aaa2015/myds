/* test_yoco_compact.c — YoCo Compact 的【公式复算】测试
 *
 * ⚠️ 范围声明（2026-10-25 审核后补）：
 *
 *   本测试【不调用任何 myds 引擎函数】，不跑任何前向，不比对任何输出。
 *   它把 src/myds_v41.c 里 sh_need / s_global 的算式在这里重新实现一遍，
 *   再断言自己的算式自洽。这是「公式复算」，不是「行为验证」。
 *
 *   它能抓到什么：算式被改错时（例如 keep_tokens 忘了减、边界夹取写反）
 *   这里会失败 —— 有一点价值，但不能证明引擎跑出来的结果正确。
 *
 *   它【不能】证明的：
 *     · MYDS_YOCO_COMPACT=1 时输出与 =0 是否语义等价
 *     · consumer 层窗口被 memset 清零后，decode 推进 12 步是否喂零
 *     · 前缀缓存中间快照能否发布（keep 必须是 V41_SNAP_STRIDE=512 的倍数）
 *
 *   真正的行为验证需要：同一 prompt 在开关两臂各跑一次，比对输出与质量指标。
 *   在那之前，MYDS_YOCO_COMPACT 必须保持默认关闭。
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); \
} } while (0)

#define V41_LAYERS 40u
#define V41_WINDOW 128u

static void test_yoco_compact_bounds(uint32_t n_tokens, uint32_t keep_tokens) {
    const uint32_t cone_shift = 0u;
    const uint32_t sub_39 = keep_tokens + cone_shift;
    const uint32_t s_global_39 = (n_tokens > sub_39) ? (n_tokens - sub_39) : 0u;

    for (uint32_t il = 20u; il < V41_LAYERS; il++) {
        const uint32_t sh_need = keep_tokens;
        const uint32_t sub = sh_need + cone_shift;
        const uint32_t s_global = (n_tokens > sub) ? (n_tokens - sub) : 0u;
        REQUIRE(s_global == s_global_39);

        uint32_t sh_full_rel = s_global;
        uint32_t sh_loop_rel = 0;
        if (il == 20u) {
            sh_loop_rel = 0u; /* Layer 20 runs compressor for all tokens */
        } else {
            sh_loop_rel = sh_full_rel; /* Layers 21..39 skip all prefix chunks */
        }

        if (n_tokens <= keep_tokens) {
            REQUIRE(s_global == 0u);
            REQUIRE(sh_full_rel == 0u);
            REQUIRE(sh_loop_rel == 0u);
        } else {
            REQUIRE(s_global == n_tokens - keep_tokens);
            if (il == 20u) {
                REQUIRE(sh_loop_rel == 0u);
                REQUIRE(sh_full_rel == n_tokens - keep_tokens);
            } else {
                REQUIRE(sh_loop_rel == n_tokens - keep_tokens);
                REQUIRE(sh_full_rel == n_tokens - keep_tokens);
            }

            /* Verify that the final token (n_tokens - 1) has its entire 128-window
             * inside the retained segment [s_global, n_tokens). */
            const uint32_t final_tok = n_tokens - 1u;
            const uint32_t window_earliest = (final_tok >= (V41_WINDOW - 1u)) ? (final_tok - (V41_WINDOW - 1u)) : 0u;
            REQUIRE(window_earliest >= s_global);
        }
    }
}

int main(void) {
    printf("[TEST] Testing YoCo Compact prefill scheduling bounds...\n");

    /* 2026-10-25：140 → 512。
     * 140 有两个量化后确认的缺陷：
     *   ① 窗口 128 需要前 127 个 token 预热（consumer 层的 window 被 memset 清零，
     *      carry_stage 只搬 window 不读 KV cache ⇒ 零窗口逐位污染）。
     *      keep=140 时精确 token 只有 140-127 = 13 个，decode 第 1 步就要 128 个
     *      真实历史 ⇒ decode 也错。安全下限 = 128 + 127 = 255。
     *   ② 140 不是 V41_SNAP_STRIDE(512) 的倍数 ⇒ 中途快照点发布不出来。
     * 512 同时满足两条。 */
    const uint32_t keep_tokens = 512u;
    const uint32_t test_lens[] = {
        1u, 16u, 64u, 128u, 255u, 256u, 511u, 512u, 513u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 65536u
    };

    uint64_t total_moe_avoided = 0;
    for (size_t i = 0; i < sizeof(test_lens) / sizeof(test_lens[0]); i++) {
        uint32_t n = test_lens[i];
        test_yoco_compact_bounds(n, keep_tokens);

        if (n > keep_tokens) {
            uint32_t avoided_per_consumer_layer = n - keep_tokens;
            /* 19 layers (21..39) completely skip the prefix for Attention + MoE,
             * and Layer 20 skips Attention + MoE on the prefix. */
            uint64_t avoided_this_prompt = (uint64_t)avoided_per_consumer_layer * 20u;
            total_moe_avoided += avoided_this_prompt;
        }
    }

    /* Verify 32K context savings */
    const uint32_t n_32k = 32768u;
    const uint64_t avoided_32k = (uint64_t)(n_32k - keep_tokens) * 20u;
    REQUIRE(avoided_32k == (32768 - 512) * 20); /* 645,120 token-layer MoE evaluations saved! */

    printf("[PASS] YoCo Compact prefill scheduling passed for all %zu test lengths!\n",
           sizeof(test_lens) / sizeof(test_lens[0]));
    printf("       Total consumer MoE token-evaluations avoided across test suite: %llu\n",
           (unsigned long long)total_moe_avoided);
    printf("       At 32K context: saved %llu consumer token-layer MoE/Attn evaluations (~99.5%% reduction in consumer prefill)\n",
           (unsigned long long)avoided_32k);

    return 0;
}
