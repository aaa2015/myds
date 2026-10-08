/*
 * myds_pld.h — Prompt-Lookup Decoding (PLD) 草稿生成器
 *
 * 核心原理：
 * 在大模型代码生成、结构化输出、问答续写等任务中，生成内容往往高度复用提示词（Prompt）
 * 或前文中已出现的变量、术语、结构。PLD 利用当前序列末尾的 N-gram 在历史上下文中反向
 * 检索相同的模式，提取其后续的连续 Token 作为候选 Draft。
 *
 * 物理特性：
 * 1. 纯 CPU 纳秒级完成（反向线性检索/前缀滑动匹配），零显存占用，零模型开销。
 * 2. 匹配命中率随 Prompt 复杂度提升，在代码和 JSON 等场景下接受率显著。
 * 3. 未命中时返回 0，无缝退回基线自回归解码，零精度损失与零负面开销。
 */
#ifndef MYDS_PLD_H
#define MYDS_PLD_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define MYDS_PLD_DEFAULT_MAX_DRAFT 4
#define MYDS_PLD_DEFAULT_MIN_NGRAM 2
#define MYDS_PLD_DEFAULT_MAX_NGRAM 3

typedef struct {
    uint32_t max_draft;      /* 最大候选 draft 长度（通常 3~5） */
    uint32_t min_ngram;      /* 最小匹配窗口大小（通常 2） */
    uint32_t max_ngram;      /* 最大匹配窗口大小（通常 3） */
    int32_t  eos_token_id;   /* 遇到 EOS 时提前截断（若不关注传 -1） */
} myds_pld_config_t;

/* 初始化默认配置 */
static inline myds_pld_config_t myds_pld_default_config(int32_t eos_id) {
    myds_pld_config_t cfg;
    cfg.max_draft = MYDS_PLD_DEFAULT_MAX_DRAFT;
    cfg.min_ngram = MYDS_PLD_DEFAULT_MIN_NGRAM;
    cfg.max_ngram = MYDS_PLD_DEFAULT_MAX_NGRAM;
    cfg.eos_token_id = eos_id;
    return cfg;
}

/*
 * 在历史序列 tokens[0 .. pos-1] 中查找匹配末尾的 N-gram，并提取后续 token 作为草稿。
 *
 * 参数：
 *   tokens:    完整 token 序列缓冲区（包含 prompt 和已生成的 token）
 *   pos:       当前已生成的总 token 数量（即当前待预测位置）
 *   cfg:       PLD 配置参数
 *   out_draft: 接收提取到的 draft token 数组（至少需要 cfg.max_draft 空间）
 *
 * 返回：
 *   实际提取到的 draft token 数量（0 表示未匹配到任何可靠草稿）
 */
static inline uint32_t myds_pld_propose(const int32_t *tokens,
                                        uint32_t pos,
                                        const myds_pld_config_t *cfg,
                                        int32_t *out_draft) {
    if (!tokens || !cfg || !out_draft || pos == 0 || cfg->max_draft == 0) return 0;

    const uint32_t min_ng = cfg->min_ngram ? cfg->min_ngram : 1;
    const uint32_t max_ng = cfg->max_ngram >= min_ng ? cfg->max_ngram : min_ng;

    /* 从最大的 N-gram 窗口开始向下尝试匹配 */
    for (uint32_t ng = max_ng; ng >= min_ng; --ng) {
        if (pos < ng * 2u) {
            /* 历史长度至少要容纳一个待匹配窗口和一个先前的目标窗口 */
            continue;
        }

        const int32_t *tail_ngram = &tokens[pos - ng];

        /* 反向扫描历史：从 (pos - ng - 1) 往前找，优先选取距离当前语义最近的匹配 */
        for (int64_t i = (int64_t)(pos - ng - 1); i >= 0; --i) {
            bool matched = true;
            for (uint32_t k = 0; k < ng; ++k) {
                if (tokens[i + k] != tail_ngram[k]) {
                    matched = false;
                    break;
                }
            }

            if (matched) {
                /* 找到匹配！从 i + ng 处提取后续作为草稿 */
                uint32_t draft_start = (uint32_t)(i + ng);
                uint32_t available = (pos > draft_start) ? (pos - draft_start) : 0;
                uint32_t draft_len = cfg->max_draft;
                if (draft_len > available) draft_len = available;

                /* 检查 EOS 并填充草稿 */
                uint32_t actual = 0;
                for (uint32_t d = 0; d < draft_len; ++d) {
                    int32_t tok = tokens[draft_start + d];
                    out_draft[actual++] = tok;
                    if (cfg->eos_token_id >= 0 && tok == cfg->eos_token_id) {
                        /* 包含 EOS 后停止 */
                        break;
                    }
                }
                if (actual > 0) {
                    return actual;
                }
            }
        }
    }

    return 0;
}

#endif /* MYDS_PLD_H */
