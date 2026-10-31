/*
 * myds_loop_guard.h — 滑动窗口周期性死循环检测器（借鉴 Project Maya LoopGuard）
 *
 * 核心原理：
 * 在低比特量化（如 Q2_K / IQ2）与长思维链生成中，模型可能在贪婪采样或极端分支下陷入
 * 周期性死循环（如反复输出相同的几句话、代码段或标点），持续消耗算力并占满上下文。
 *
 * LoopGuard 在生成流上以 O(P) 极低开销维护最近 Token 环形历史：
 * - 针对候选周期 P ∈ [MIN_PERIOD, MAX_PERIOD] 进行回溯比较；
 * - 短周期（P < 16）需连续重复 4 次，长周期（P >= 16）连续重复 3 次；
 * - 命中后立即通知引擎安全截断（MYDS_STOP_EOS），避免失控。
 */
#ifndef MYDS_LOOP_GUARD_H
#define MYDS_LOOP_GUARD_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define MYDS_LOOP_GUARD_CAPACITY 1024
#define MYDS_LOOP_GUARD_MIN_PERIOD 4
#define MYDS_LOOP_GUARD_MAX_PERIOD 256

typedef struct {
    int32_t  history[MYDS_LOOP_GUARD_CAPACITY];
    uint32_t head;
    uint32_t count;
    uint32_t detected_period;
    uint32_t detected_reps;
    bool     enabled;
} myds_loop_guard_t;

static inline void myds_loop_guard_init(myds_loop_guard_t *lg, bool enabled) {
    if (!lg) return;
    memset(lg->history, 0, sizeof(lg->history));
    lg->head = 0;
    lg->count = 0;
    lg->detected_period = 0;
    lg->detected_reps = 0;
    lg->enabled = enabled;
}

/* 读取距离最近 token 往前第 k 个 token（k=0 为最新生成的 token） */
static inline int32_t myds_loop_guard_at(const myds_loop_guard_t *lg, uint32_t k) {
    int32_t idx = ((int32_t)lg->head - 1 - (int32_t)k) % (int32_t)MYDS_LOOP_GUARD_CAPACITY;
    if (idx < 0) idx += (int32_t)MYDS_LOOP_GUARD_CAPACITY;
    return lg->history[idx];
}

/*
 * 将新 token 输入 LoopGuard。
 * 若检测到周期性死循环重复，返回 true；此时可通过 lg->detected_period 和 lg->detected_reps 查看详情。
 */
static inline bool myds_loop_guard_feed(myds_loop_guard_t *lg, int32_t token) {
    if (!lg || !lg->enabled) return false;

    lg->history[lg->head] = token;
    lg->head = (lg->head + 1) % MYDS_LOOP_GUARD_CAPACITY;
    if (lg->count < MYDS_LOOP_GUARD_CAPACITY) lg->count++;

    if (lg->count < MYDS_LOOP_GUARD_MIN_PERIOD * 3) return false;

    uint32_t max_p = lg->count / 3;
    if (max_p > MYDS_LOOP_GUARD_MAX_PERIOD) max_p = MYDS_LOOP_GUARD_MAX_PERIOD;

    /* 从短周期到长周期逐步检查 */
    for (uint32_t p = MYDS_LOOP_GUARD_MIN_PERIOD; p <= max_p; ++p) {
        uint32_t min_reps = (p < 16) ? 4 : 3;
        if (lg->count < p * min_reps) continue;

        bool is_periodic = true;
        for (uint32_t rep = 1; rep < min_reps; ++rep) {
            for (uint32_t i = 0; i < p; ++i) {
                int32_t t0 = myds_loop_guard_at(lg, i);
                int32_t t1 = myds_loop_guard_at(lg, rep * p + i);
                if (t0 != t1) {
                    is_periodic = false;
                    break;
                }
            }
            if (!is_periodic) break;
        }

        if (is_periodic) {
            lg->detected_period = p;
            lg->detected_reps = min_reps;
            return true;
        }
    }

    return false;
}

#endif /* MYDS_LOOP_GUARD_H */

