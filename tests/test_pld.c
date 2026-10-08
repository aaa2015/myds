/*
 * tests/test_pld.c — Prompt-Lookup Decoding (PLD) 单元测试
 */
#include "myds_pld.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

static void test_pld_null_and_bounds(void) {
    int32_t draft[8];
    myds_pld_config_t cfg = myds_pld_default_config(-1);

    /* 空指针与 0 长度 */
    assert(myds_pld_propose(NULL, 10, &cfg, draft) == 0);
    int32_t tokens[] = {1, 2, 3};
    assert(myds_pld_propose(tokens, 0, &cfg, draft) == 0);
    assert(myds_pld_propose(tokens, 3, NULL, draft) == 0);
    assert(myds_pld_propose(tokens, 3, &cfg, NULL) == 0);

    /* 历史长度不足一个 ngram */
    cfg.min_ngram = 2;
    assert(myds_pld_propose(tokens, 2, &cfg, draft) == 0);
    assert(myds_pld_propose(tokens, 3, &cfg, draft) == 0); /* 3 < 2*2 */

    puts("  [PASS] PLD null and boundary conditions");
}

static void test_pld_exact_match(void) {
    /*
     * 场景：Prompt 中有函数调用 "def process_data(data, config):"
     * 生成文本时出现 "process_data("，预测后面应该接 "data, config):"
     * 映射为 Token:
     *   [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 20, 30, 40]
     *   末尾 3-gram 是 (20, 30, 40)
     *   在前面的 index 1 处匹配到了 (20, 30, 40)，其后面是 (50, 60, 70, 80)
     */
    int32_t tokens[] = {
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        20, 30, 40
    };
    uint32_t pos = sizeof(tokens) / sizeof(tokens[0]);

    myds_pld_config_t cfg = myds_pld_default_config(-1);
    cfg.max_draft = 4;
    cfg.max_ngram = 3;
    cfg.min_ngram = 2;

    int32_t draft[8] = {0};
    uint32_t n = myds_pld_propose(tokens, pos, &cfg, draft);

    assert(n == 4);
    assert(draft[0] == 50);
    assert(draft[1] == 60);
    assert(draft[2] == 70);
    assert(draft[3] == 80);

    puts("  [PASS] PLD exact 3-gram extraction");
}

static void test_pld_adaptive_fallback(void) {
    /*
     * 场景：3-gram 未命中，但 2-gram 命中了
     *   Prompt: [100, 200, 300, 400, 500]
     *   Current: [999, 200, 300]
     *   末尾 3-gram 是 (999, 200, 300)，无匹配；
     *   退化到 2-gram (200, 300)，在 index 1 命中，其后续为 (400, 500)
     */
    int32_t tokens[] = {
        100, 200, 300, 400, 500,
        999, 200, 300
    };
    uint32_t pos = sizeof(tokens) / sizeof(tokens[0]);

    myds_pld_config_t cfg = myds_pld_default_config(-1);
    cfg.max_draft = 3;
    cfg.max_ngram = 3;
    cfg.min_ngram = 2;

    int32_t draft[8] = {0};
    uint32_t n = myds_pld_propose(tokens, pos, &cfg, draft);

    assert(n == 3);
    assert(draft[0] == 400);
    assert(draft[1] == 500);
    assert(draft[2] == 999);

    puts("  [PASS] PLD adaptive fallback from 3-gram to 2-gram");
}

static void test_pld_recency_preference(void) {
    /*
     * 场景：多处匹配时，优先选最近的匹配点
     * 第一次出现：[1, 2, 3] -> 后续 [10, 20]
     * 第二次出现：[1, 2, 3] -> 后续 [88, 99]
     * 当前末尾：  [1, 2, 3]
     * 预期选 [88, 99]，而不是较远的 [10, 20]
     */
    int32_t tokens[] = {
        1, 2, 3, 10, 20, 500,
        1, 2, 3, 88, 99, 600,
        1, 2, 3
    };
    uint32_t pos = sizeof(tokens) / sizeof(tokens[0]);

    myds_pld_config_t cfg = myds_pld_default_config(-1);
    cfg.max_draft = 2;
    cfg.max_ngram = 3;
    cfg.min_ngram = 3;

    int32_t draft[8] = {0};
    uint32_t n = myds_pld_propose(tokens, pos, &cfg, draft);

    assert(n == 2);
    assert(draft[0] == 88);
    assert(draft[1] == 99);

    puts("  [PASS] PLD recency preference (closest to tail)");
}

static void test_pld_eos_truncation(void) {
    /*
     * 场景：后续草稿中包含 EOS token (如 2)，应在 EOS 处截断
     */
    int32_t tokens[] = {
        10, 20, 30, 40, 2 /* EOS */, 50, 60,
        100, 200, 20, 30
    };
    uint32_t pos = sizeof(tokens) / sizeof(tokens[0]);

    myds_pld_config_t cfg = myds_pld_default_config(2 /* eos */);
    cfg.max_draft = 5;
    cfg.max_ngram = 2;
    cfg.min_ngram = 2;

    int32_t draft[8] = {0};
    uint32_t n = myds_pld_propose(tokens, pos, &cfg, draft);

    /* 匹配 (20, 30)，后续原本为 [40, 2, 50]，截断后应为 [40, 2] */
    assert(n == 2);
    assert(draft[0] == 40);
    assert(draft[1] == 2);

    puts("  [PASS] PLD EOS truncation");
}

/* 模拟模型在给定历史序列下的下一个 token 生成（确定性 oracle） */
static int32_t mock_oracle_forward(const int32_t *tokens, uint32_t pos) {
    if (pos == 0) return 100;
    int32_t prev = tokens[pos - 1];
    /* 如果前一个是 42，后面跟 43；如果是 43，跟 44；如果是 44，跟 45；如果是 45，跟 999 */
    if (prev == 42) return 43;
    if (prev == 43) return 44;
    if (prev == 44) return 45;
    if (prev == 45) return 999;
    /* 默认模运算转移 */
    return (prev * 31 + 7) % 1000;
}

static void test_pld_verification_parity(void) {
    /* 场景：prompt 中有 [42, 43, 44, 45, 999] 的模式 */
    const int32_t prompt[] = {
        10, 20, 30, 42, 43, 44, 45, 999, 50, 60, 42
    };
    const uint32_t n_prompt = sizeof(prompt) / sizeof(prompt[0]);
    const uint32_t max_new = 30;

    /* 1. 标准自回归生成 */
    int32_t seq_baseline[128];
    memcpy(seq_baseline, prompt, sizeof(prompt));
    uint32_t len_baseline = n_prompt;
    for (uint32_t i = 0; i < max_new; i++) {
        int32_t next = mock_oracle_forward(seq_baseline, len_baseline);
        seq_baseline[len_baseline++] = next;
    }

    /* 2. PLD 投机验证生成 */
    int32_t seq_pld[128];
    memcpy(seq_pld, prompt, sizeof(prompt));
    uint32_t len_pld = n_prompt;
    uint32_t pld_step = 0;
    uint32_t pld_accepted = 0;
    uint32_t pld_cycles = 0;

    myds_pld_config_t cfg = myds_pld_default_config(9999);
    cfg.max_draft = 3;

    while (pld_step < max_new) {
        int32_t draft[8] = {0};
        uint32_t n_draft = myds_pld_propose(seq_pld, len_pld, &cfg, draft);

        if (n_draft == 0) {
            int32_t next = mock_oracle_forward(seq_pld, len_pld);
            seq_pld[len_pld++] = next;
            pld_step++;
            continue;
        }

        pld_cycles++;
        for (uint32_t d = 0; d < n_draft && pld_step < max_new; d++) {
            int32_t true_next = mock_oracle_forward(seq_pld, len_pld);
            seq_pld[len_pld++] = true_next;
            pld_step++;

            if (true_next == draft[d]) {
                pld_accepted++;
            } else {
                break; /* 猜测失败，终止本轮投机 */
            }
        }
    }

    /* 3. 严格逐位断言（Bit-Exact Parity） */
    assert(len_pld == len_baseline);
    assert(memcmp(seq_pld, seq_baseline, len_baseline * sizeof(int32_t)) == 0);
    assert(pld_accepted > 0 && "PLD should have accepted matching tokens");
    assert(pld_cycles > 0);

    printf("  [PASS] PLD verification parity (memcmp == 0, accepted %u tokens across %u cycles)\n",
           pld_accepted, pld_cycles);
}

int main(void) {
    printf("====================================================\n");
    printf("       Running myds Prompt-Lookup Decoding Tests    \n");
    printf("====================================================\n");

    test_pld_null_and_bounds();
    test_pld_exact_match();
    test_pld_adaptive_fallback();
    test_pld_recency_preference();
    test_pld_eos_truncation();
    test_pld_verification_parity();

    printf("====================================================\n");
    printf("    ALL PLD TESTS PASSED CLEANLY & SUCCESSFULLY!    \n");
    printf("====================================================\n");
    return 0;
}
