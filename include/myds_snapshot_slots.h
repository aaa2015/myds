#ifndef MYDS_SNAPSHOT_SLOTS_H
#define MYDS_SNAPSHOT_SLOTS_H

/* 前缀快照的【槽位分配】——抽出来是为了能单独测：这段逻辑错了不会报错，
 * 只会让本该命中的请求退化成全量重算（或者更糟：接到错的点上）。
 *
 * 背景：一个长对话里，真正被反复复用的部分是【序列开头那一段常量前缀】——
 * 系统提示词 + 工具定义 + 头几轮。每个新会话、每条短提示词都从它分叉。
 * 而快照本来就按固定间隔（stride）存，位置越深越值钱、越浅越常用。
 *
 * 原来的「按位置取模」分配会让远处的点把近处的槽顶掉：
 *   8 个槽、7 个环形 ⇒ 位置 2048（槽 5）会被 5632 / 9216 顶掉，
 *   于是对话一长过 ~9k token，常量前缀就没有快照可用了
 *   —— 实测日志里出现过 "common prefix 1067 of cached 5169, but no snapshot within it"，
 *   那就是这个退化的样子（有公共前缀，但快照已被挤掉 ⇒ 全量重算）。
 *
 * 现在把【低位区】固定下来（不参与环形替换），环形槽只服务更深的点：
 *   槽 0           本次序列末尾（保持原语义）
 *   槽 1..PINNED   位置 1024, 1536, ... 固定对应，覆盖常量前缀所在区间
 *   其余槽         更深的点按取模复用
 *
 * 代价只有内存：每个槽 = 层数 × 窗口 × head_dim × 4B ≈ 10.5 MB（本模型），
 * 24 槽约 240 MB。相对「新会话每次全量 prefill 几秒」是划算的。 */

#include <stdint.h>

#define MYDS_SNAP_SLOTS    24u   /* 含槽 0 与固定低位区 */
#define MYDS_SNAP_STRIDE   512u  /* 每隔多少 token 存一份 */
#define MYDS_SNAP_MIN      1024u /* 低于此长度不值得存（太浅的前缀复用价值低） */
#define MYDS_SNAP_PINNED   11u   /* 固定槽数量：覆盖位置 1024 .. 1024+(PINNED-1)*STRIDE */

/* 位置 p 的快照该放哪个槽。`is_last` 表示这是本次序列的末尾点。
 * 调用方需保证 p >= MYDS_SNAP_MIN（更浅的点本就不存）。 */
static inline uint32_t myds_snap_slot_for(uint32_t p, int is_last) {
    if (is_last) return 0u;
    const uint32_t first = MYDS_SNAP_MIN / MYDS_SNAP_STRIDE;      /* = 2 */
    const uint32_t idx = p / MYDS_SNAP_STRIDE;
    if (idx < first) return 0u;                                   /* 见上：调用方已拦，这里兜底 */
    if (idx < first + MYDS_SNAP_PINNED)
        return 1u + (idx - first);                                /* 低位区：固定，不参与替换 */
    const uint32_t ring = MYDS_SNAP_SLOTS - 1u - MYDS_SNAP_PINNED;
    if (ring == 0u) return 1u + MYDS_SNAP_PINNED;
    return 1u + MYDS_SNAP_PINNED + ((idx - first - MYDS_SNAP_PINNED) % ring);
}

/* 该槽是否是「固定低位槽」（供日志/诊断区分：这些点不会被后续请求挤掉）。 */
static inline int myds_snap_slot_is_pinned(uint32_t slot) {
    return slot >= 1u && slot <= MYDS_SNAP_PINNED;
}

#endif /* MYDS_SNAPSHOT_SLOTS_H */
