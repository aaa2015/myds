/*
 * myds_kvdsk.h — 低位区 KV 的【SSD 落盘】（定长槽 + 变长 payload 的条目存储）
 *
 * 目标：把可复用的前缀状态存到磁盘，供新会话与重启后的恢复使用。
 *
 * 为什么只做【低位区】：快照本来就按固定间隔存，而真正被反复复用的是序列开头那一段
 * （见 myds_snapshot_slots.h）。位置越深越少被命中，越浅越常用。所以只把「固定低位槽」
 * （MYDS_SNAP_PINNED 个，覆盖位置 1024..1024+(MYDS_SNAP_PINNED-1)*MYDS_SNAP_STRIDE；现为
 * 1024..11264）落盘，环形槽不进磁盘。
 * 文件有界；每个位置组的多个 way 可保存不同前缀，由调用方选择淘汰条目。
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
 * 存储层不假设 payload 是完整快照还是共享压缩 KV 段；两者可分别使用独立实例。
 * 槽容量由调用方给出，每条只写、只读实际 payload_used，不处理填充区。
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
 * `budget_bytes`  文件总字节上限；0 = 不限（默认，见 myds_v41_low_kv_enable）。超出则
 *                 【打开失败并报错】，不会自动下调槽数。注意：没有调用方会在失败后下调槽数
 *                 重试，所以【设了】预算就必须按 slots × slot_stride 配足。
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
/* Inspect a committed payload range without starting a read transaction or
 * updating checksum/hits. This is only for descriptor comparison; restore
 * must still validate the complete payload with read_start/chunk/finish. */
bool myds_kvdsk_peek_committed(myds_kvdsk_t *k, uint32_t slot, uint64_t offset,
                               void *dst, uint64_t bytes);
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
/* Finish a complete checksum read used only for save-time validation; unlike
 * read_finish this does not count a restore hit or write hit metadata. */
bool myds_kvdsk_read_verify(myds_kvdsk_t *k);
/* 显式终止读事务（遇到语义校验失败等外部错误时调用，避免读事务状态泄漏）。 */
void myds_kvdsk_read_abort(myds_kvdsk_t *k);

/* ── 流式写 ── */
bool myds_kvdsk_write_start(myds_kvdsk_t *k, uint32_t slot);
bool myds_kvdsk_write_chunk(myds_kvdsk_t *k, uint64_t offset, const void *src, uint64_t bytes);
/* 落条目。position > 0 且 used_bytes ≤ slot_stride 才有效；到此为止失败 ⇒ 该槽保持空。 */
bool myds_kvdsk_write_finish(myds_kvdsk_t *k, uint32_t slot, uint32_t position,
                             uint64_t prefix_hash, uint64_t used_bytes);
/* Cancel an uncommitted write after a caller-side error. The slot remains a
 * clean miss and the store can accept a new transaction without reopening. */
void myds_kvdsk_write_abort(myds_kvdsk_t *k);

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
