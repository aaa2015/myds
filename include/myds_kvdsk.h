/*
 * myds_kvdsk.h — 低位区 KV 的【SSD 落盘】（定长槽 + 变长 payload 的条目存储）
 *
 * 目标：把「序列开头那段常量前缀」的 KV 状态存到磁盘，让【新会话】与【重启之后】
 * 都能直接恢复，不必重算。实测这段前缀（系统提示词 + 工具定义）约 4800 token，
 * 全量 prefill 要 17 秒 —— 而每个新会话都要重付一次。
 *
 * 为什么只做【低位区】：快照本来就按固定间隔存，而真正被反复复用的是序列开头那一段
 * （见 myds_snapshot_slots.h）。位置越深越少被命中，越浅越常用。所以只把「固定低位槽」
 * （MYDS_SNAP_PINNED 个，覆盖位置 1024..6144）落盘，环形槽不进磁盘。
 * 好处是文件有界、且淘汰策略退化成「一个位置一条，新的覆盖旧的」。
 *
 * ── 本模块的职责边界 ──
 * 它【不知道】KV 的布局，只管：
 *   · 条目读写（position / prefix_hash / payload 长度 / CRC32）
 *   · 身份戳（identity）：不一致就整份作废 —— 模型或布局变了，旧字节没有意义
 *   · 崩溃安全：写一条前先把该槽置空，payload 写完才落「位置」这一步提交。
 *     中途掉电 ⇒ 该槽是「空」而不是「坏数据」，最坏只是少一次命中，绝不会接错状态。
 * payload 的内部布局由调用方（myds_v41.c）定义，它负责把布局参数并进 identity。
 *
 * ── 为什么 payload 是变长的 ──
 * 一条快照里最大的一块是「压缩 KV」，它的行数随位置线性增长（每个 owner 每 ratio 个
 * token 一行）。位置 1024 与位置 6144 的实际数据量差了 6 倍，所以槽的容量按【最深
 * 位置】定，但每条只写、只读它自己那一段（payload_used），不写不读填充区。
 */
#ifndef MYDS_KVDSK_H
#define MYDS_KVDSK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct myds_kvdsk myds_kvdsk_t;

/*
 * 打开（不存在则创建）。identity / 版本 / 槽数 / 槽容量任何一个不符，都重建为空表 ——
 * 那不是「损坏」，是运行环境变了，旧内容不该被复用。
 *
 * `slots`         条目数（= 低位位置组数 × 每组 way 数）。
 *                 调用方按「一个位置 = 一组 way」使用，见 myds_v41.c 的 lowkv_pick_entry。
 * `slot_stride`   每个槽在文件里的字节容量（按最深位置算）
 * `budget_bytes`  文件总字节上限；0 = 不限。超出则在打开时报错（由调用方下调槽数）。
 * 失败返回 NULL；原因用 myds_kvdsk_open_error() 取。
 */
myds_kvdsk_t *myds_kvdsk_open(const char *path, uint64_t identity,
                              uint32_t slots, uint64_t slot_stride,
                              uint64_t budget_bytes);
void          myds_kvdsk_close(myds_kvdsk_t *k);
const char   *myds_kvdsk_open_error(void);

/* ── 条目元信息（不读 payload）── */
bool myds_kvdsk_slot_meta(const myds_kvdsk_t *k, uint32_t slot,
                          uint32_t *position, uint64_t *prefix_hash, uint64_t *used_bytes);
/* 命中计数 +1（只是证据：用来回答「这段前缀到底有没有被复用」）。 */
bool myds_kvdsk_slot_touch(myds_kvdsk_t *k, uint32_t slot);
/* 读命中数。★ 淘汰要用的判据：命中过 = 这份快照真的在被用，
 * 比「刚写进来的」更值得保留（ds4 的 KV_CACHE_MIN_EFFECTIVE_HITS 是同一个理由）。 */
bool myds_kvdsk_slot_hits(const myds_kvdsk_t *k, uint32_t slot, uint32_t *hits);
/* 置空某槽（写之前必须先调；发现坏数据时也用它作废）。 */
bool myds_kvdsk_slot_clear(myds_kvdsk_t *k, uint32_t slot);
/* 当前非空条目数（诊断用）。 */
uint32_t myds_kvdsk_live_entries(const myds_kvdsk_t *k);

/* ── 流式读 ── */
/* 成功时 *used_bytes 给出本条实际字节数（≤ slot_stride）。 */
bool myds_kvdsk_read_start(myds_kvdsk_t *k, uint32_t slot, uint64_t *used_bytes);
bool myds_kvdsk_read_chunk(myds_kvdsk_t *k, uint64_t offset, void *dst, uint64_t bytes);
/* 校验 CRC。不符 ⇒ 把该槽置空并返回 false（调用方必须丢弃本次恢复）。 */
bool myds_kvdsk_read_finish(myds_kvdsk_t *k);
/* 显式终止读事务（遇到语义校验失败等外部错误时调用，避免读事务状态泄漏）。 */
void myds_kvdsk_read_abort(myds_kvdsk_t *k);

/* ── 流式写 ── */
bool myds_kvdsk_write_start(myds_kvdsk_t *k, uint32_t slot);
bool myds_kvdsk_write_chunk(myds_kvdsk_t *k, uint64_t offset, const void *src, uint64_t bytes);
/* 落条目。position > 0 且 used_bytes ≤ slot_stride 才有效；到此为止失败 ⇒ 该槽保持空。 */
bool myds_kvdsk_write_finish(myds_kvdsk_t *k, uint32_t slot, uint32_t position,
                             uint64_t prefix_hash, uint64_t used_bytes);

/* 前缀哈希（FNV-1a 64，按 token 逐字节）。对【同一条 token 序列】必须稳定 ——
 * 跨进程复用唯一的安全性来源就是它：字节能不能用，取决于它是不是同一段前缀。
 * 单独放出来是为了能单测（不需要模型）。 */
uint64_t myds_kvdsk_hash_tokens(const int32_t *tokens, uint32_t count);

/* 身份戳：把「决定字节含义的所有参数」并成一个 u64，便于比对。 */
uint64_t myds_kvdsk_identity(const char *tag, const uint64_t *values, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_KVDSK_H */
