/*
 * myds_kvdsk.c — 低位区 KV 落盘的存储层实现（见 myds_kvdsk.h 的设计说明）
 *
 * 文件布局（小端；本仓库只跑 Apple Silicon，不做字节序转换，但哈希与 CRC 都按
 * 显式字节序处理，保证「同一段 token 序列在两次运行里得到同一个哈希」）：
 *
 *   偏移 0        头（固定 4096 字节）
 *                   u8[8] magic / u32 version / u32 slots / u64 identity
 *                   u64 slot_stride / u64 file_bytes
 *   偏移 4096     条目表：slots × 64 字节
 *                   u32 position（0 = 空；非 0 才是【提交点】）
 *                   u32 hits
 *                   u64 prefix_hash
 *                   u64 used_bytes
 *                   u32 crc（覆盖 payload[0..used_bytes)）
 *                   u32 state（0 空 / 1 写入中 / 2 有效）
 *   偏移 payload_base（4096 对齐）  payload：slots × slot_stride
 *                   第 s 槽 = payload_base + s*slot_stride
 *
 * 提交协议（崩溃安全的关键）：先把 position 写成 0 ⇒ 再写 payload ⇒ 再写其余字段 ⇒
 * 最后单独写 4 字节 position。任何一步掉电，该槽都停在「空」，而不是「半份数据」。
 * 没有 fsync：丢一份快照的代价只是一次 miss（回退全量 prefill），不值得为它付同步写。
 * 万一 payload 与条目落盘不一致，读时的 CRC 会把这条作废 —— 两道都要。
 */
#include "myds_kvdsk.h"
#include <time.h>
#include "myds_log.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define KVDSK_VERSION      1u
#define KVDSK_HEADER_BYTES 4096u
#define KVDSK_ENTRY_BYTES  64u
#define KVDSK_MAGIC        "MYKVLO01"

#define KVDSK_STATE_EMPTY  0u
#define KVDSK_STATE_WRITING 1u
#define KVDSK_STATE_VALID  2u

struct myds_kvdsk {
    int      fd;
    uint32_t slots;
    uint64_t identity;
    uint64_t slot_stride;
    uint64_t payload_base;
    uint64_t file_bytes;
    /* 当前读事务 */
    int      reading;
    uint32_t read_slot;
    uint64_t read_crc;
    uint64_t read_expected_crc;
    uint64_t read_used;
    /* 当前写事务 */
    int      writing;
    uint32_t write_slot;
    uint64_t write_crc;
};

static char g_open_error[512];

const char *myds_kvdsk_open_error(void) { return g_open_error; }

static void set_error(const char *what, const char *path) {
    snprintf(g_open_error, sizeof(g_open_error), "%s%s%s%s",
             what ? what : "?", path ? " [" : "", path ? path : "", path ? "]" : "");
}

/* ── 校验和 ──
 * 为什么不用逐字节的 CRC32：第一次落盘要写 11 个槽、约 550 MB，逐字节查表会花掉
 * ~1.5 秒 CPU，而那笔开销要算在【首个请求】的 ttft 上。这里按 64 位字混合，
 * 吞吐约 2~3 GB/s（同样 550 MB 约 0.2 秒）。
 * 强度上够用：它只需要发现「撕裂写」和「介质损坏」，而「这份字节是不是给这条前缀的」
 * 由 prefix_hash 独立保证 —— 两者职责不同，不该用一个 32 位 CRC 同时顶。 */
static uint64_t cks_begin(void) { return 1469598103934665603ull; }

static uint64_t cks_update(uint64_t h, const void *buf, size_t len) {
    const unsigned char *p = (const unsigned char *)buf;
    while (len >= 8) {
        uint64_t w = 0;
        for (int b = 0; b < 8; b++) w |= (uint64_t)p[b] << (8 * b);   /* 显式小端 */
        h ^= w;
        h *= 1099511628211ull;
        p += 8; len -= 8;
    }
    while (len--) { h ^= (uint64_t)*p++; h *= 1099511628211ull; }
    return h;
}

/* ── 定长字段读写（避免直接 memcpy 结构体：结构体有填充，布局不该进文件）── */
static void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFFu); p[1] = (unsigned char)((v >> 8) & 0xFFu);
    p[2] = (unsigned char)((v >> 16) & 0xFFu); p[3] = (unsigned char)((v >> 24) & 0xFFu);
}
static uint32_t get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void put_u64(unsigned char *p, uint64_t v) {
    put_u32(p, (uint32_t)(v & 0xFFFFFFFFu));
    put_u32(p + 4, (uint32_t)((v >> 32) & 0xFFFFFFFFu));
}
static uint64_t get_u64(const unsigned char *p) {
    return (uint64_t)get_u32(p) | ((uint64_t)get_u32(p + 4) << 32);
}

static int read_at(int fd, void *dst, size_t bytes, uint64_t offset) {
    unsigned char *p = (unsigned char *)dst;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0) { if (errno == EINTR) continue; return 0; }
        if (n == 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static int write_at(int fd, const void *src, size_t bytes, uint64_t offset) {
    const unsigned char *p = (const unsigned char *)src;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pwrite(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0) { if (errno == EINTR) continue; return 0; }
        if (n == 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

/* 条目表的位置只由 slot 决定（头是定长），不需要 context。 */
static uint64_t entry_offset(uint32_t slot) {
    return (uint64_t)KVDSK_HEADER_BYTES + (uint64_t)slot * KVDSK_ENTRY_BYTES;
}
static uint64_t payload_offset(const myds_kvdsk_t *k, uint32_t slot) {
    return k->payload_base + (uint64_t)slot * k->slot_stride;
}
static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1u) / a * a; }

/* 目录不存在就建出来（含中间层）。失败不算致命，后面 open 会报真正的原因。 */
static void ensure_parent_dir(const char *path) {
    if (!path) return;
    char buf[1024];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof(buf)) return;
    memcpy(buf, path, n + 1);
    char *slash = strrchr(buf, '/');
    if (!slash || slash == buf) return;
    *slash = 0;
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        (void)mkdir(buf, 0755);
        *p = '/';
    }
    (void)mkdir(buf, 0755);
}

/* ── 条目读写 ── */
static int entry_read(const myds_kvdsk_t *k, uint32_t slot, unsigned char out[KVDSK_ENTRY_BYTES]) {
    return read_at(k->fd, out, KVDSK_ENTRY_BYTES, entry_offset(slot));
}

/* 只写「提交点」之外的部分（position 留在最后单独写）。 */
static int entry_write_body(const myds_kvdsk_t *k, uint32_t slot, uint32_t hits,
                            uint64_t prefix_hash, uint64_t used_bytes,
                            uint64_t crc, uint32_t state) {
    unsigned char e[KVDSK_ENTRY_BYTES];
    memset(e, 0, sizeof(e));
    put_u32(e + 0, 0u);                 /* position：body 阶段一律 0 */
    put_u32(e + 4, hits);
    put_u64(e + 8, prefix_hash);
    put_u64(e + 16, used_bytes);
    put_u32(e + 24, state);
    put_u64(e + 32, crc);          /* u64：校验和随 payload 规模一起长大 */
    /* 2026-10-24：写入时刻（u64 微秒，offset 40，原空闲区）。
     * 仅用于 FIFO 淘汰：槽满时丢最老的一条，而不是「命中数最少」的那条。
     * 用的是 wall clock —— 它不带单调保证，但只用于相对排序，够用。 */
    {
        struct timespec ts;
        uint64_t now_us = 0;
        if (clock_gettime(CLOCK_REALTIME, &ts) == 0)
            now_us = (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
        put_u64(e + 40, now_us);
    }
    return write_at(k->fd, e, KVDSK_ENTRY_BYTES, entry_offset(slot));
}

static int entry_write_position(const myds_kvdsk_t *k, uint32_t slot, uint32_t position) {
    unsigned char p[4];
    put_u32(p, position);
    return write_at(k->fd, p, 4, entry_offset(slot));
}

bool myds_kvdsk_slot_clear(myds_kvdsk_t *k, uint32_t slot) {
    if (!k || slot >= k->slots) return false;
    if (!entry_write_body(k, slot, 0u, 0u, 0u, 0u, KVDSK_STATE_EMPTY)) return false;
    return entry_write_position(k, slot, 0u) ? true : false;
}

bool myds_kvdsk_slot_meta(const myds_kvdsk_t *k, uint32_t slot,
                          uint32_t *position, uint64_t *prefix_hash, uint64_t *used_bytes) {
    if (!k || slot >= k->slots) return false;
    unsigned char e[KVDSK_ENTRY_BYTES];
    if (!entry_read(k, slot, e)) return false;
    const uint32_t pos = get_u32(e + 0);
    const uint32_t state = get_u32(e + 24);
    if (pos == 0u || state != KVDSK_STATE_VALID) return false;
    const uint64_t used = get_u64(e + 16);
    if (used == 0u || used > k->slot_stride) return false;
    if (position) *position = pos;
    if (prefix_hash) *prefix_hash = get_u64(e + 8);
    if (used_bytes) *used_bytes = used;
    return true;
}

bool myds_kvdsk_peek_committed(myds_kvdsk_t *k, uint32_t slot, uint64_t offset,
                               void *dst, uint64_t bytes) {
    uint64_t used = 0;
    if (!k || k->writing || (!dst && bytes) || bytes > SIZE_MAX ||
        !myds_kvdsk_slot_meta(k, slot, NULL, NULL, &used) ||
        offset > used || bytes > used - offset) return false;
    if (bytes == 0) return true;
    return read_at(k->fd, dst, (size_t)bytes, payload_offset(k, slot) + offset) != 0;
}

bool myds_kvdsk_slot_touch(myds_kvdsk_t *k, uint32_t slot) {
    if (!k || slot >= k->slots) return false;
    unsigned char e[KVDSK_ENTRY_BYTES];
    if (!entry_read(k, slot, e)) return false;
    if (get_u32(e + 0) == 0u) return false;
    unsigned char hits[4];
    put_u32(hits, get_u32(e + 4) + 1u);
    return write_at(k->fd, hits, 4, entry_offset(slot) + 4u) ? true : false;
}

/* 2026-10-24：读取该槽的写入时刻（微秒）。旧文件里是 0 ⇒ 视为最老。 */
bool myds_kvdsk_slot_written_us(const myds_kvdsk_t *k, uint32_t slot, uint64_t *written_us) {
    if (!k || slot >= k->slots || !written_us) return false;
    unsigned char e[KVDSK_ENTRY_BYTES];
    if (!entry_read(k, slot, e)) return false;
    if (get_u32(e + 0) == 0u) return false;
    *written_us = get_u64(e + 40);
    return true;
}

bool myds_kvdsk_slot_hits(const myds_kvdsk_t *k, uint32_t slot, uint32_t *hits) {
    if (!k || slot >= k->slots) return false;
    unsigned char e[KVDSK_ENTRY_BYTES];
    if (!entry_read(k, slot, e)) return false;
    if (get_u32(e + 0) == 0u) return false;
    if (hits) *hits = get_u32(e + 4);
    return true;
}

uint32_t myds_kvdsk_live_entries(const myds_kvdsk_t *k) {
    if (!k) return 0;
    uint32_t n = 0;
    for (uint32_t s = 0; s < k->slots; s++) {
        unsigned char e[KVDSK_ENTRY_BYTES];
        if (!entry_read(k, s, e)) break;
        if (get_u32(e + 0) != 0u && get_u32(e + 24) == KVDSK_STATE_VALID) n++;
    }
    return n;
}

/* ── 读事务 ── */
bool myds_kvdsk_read_start(myds_kvdsk_t *k, uint32_t slot, uint64_t *used_bytes) {
    if (!k || slot >= k->slots || k->reading || k->writing) return false;
    unsigned char e[KVDSK_ENTRY_BYTES];
    if (!entry_read(k, slot, e)) return false;
    const uint32_t pos = get_u32(e + 0);
    if (pos == 0u || get_u32(e + 24) != KVDSK_STATE_VALID) return false;
    const uint64_t used = get_u64(e + 16);
    if (used == 0u || used > k->slot_stride) return false;
    k->reading = 1;
    k->read_slot = slot;
    k->read_used = used;
    k->read_expected_crc = get_u64(e + 32);
    k->read_crc = cks_begin();
    if (used_bytes) *used_bytes = used;
    return true;
}

/* 契约：事务内任何失败都【终止该事务】。调用方看到 false 就知道「本次读作废」，
 * 不必再去想事务还开着没有。 */
bool myds_kvdsk_read_chunk(myds_kvdsk_t *k, uint64_t offset, void *dst, uint64_t bytes) {
    if (!k || !k->reading || !dst) return false;
    if (offset + bytes > k->read_used) { k->reading = 0; return false; }
    if (!read_at(k->fd, dst, (size_t)bytes, payload_offset(k, k->read_slot) + offset)) {
        k->reading = 0;
        return false;
    }
    k->read_crc = cks_update(k->read_crc, dst, (size_t)bytes);
    return true;
}

static bool read_finish(myds_kvdsk_t *k, bool touch) {
    if (!k || !k->reading) return false;
    const uint32_t slot = k->read_slot;
    const uint64_t got = k->read_crc;
    k->reading = 0;
    if (got != k->read_expected_crc) {
        MYDS_LOG_W("kvdisk: slot %u 的 payload 校验和不符（读到 %016llx，应为 %016llx）"
                   "—— 作废该条，本次读取失败。",
                   slot, (unsigned long long)got, (unsigned long long)k->read_expected_crc);
        (void)myds_kvdsk_slot_clear(k, slot);
        return false;
    }
    if (touch) (void)myds_kvdsk_slot_touch(k, slot);
    return true;
}

bool myds_kvdsk_read_finish(myds_kvdsk_t *k) { return read_finish(k, true); }
bool myds_kvdsk_read_verify(myds_kvdsk_t *k) { return read_finish(k, false); }

void myds_kvdsk_read_abort(myds_kvdsk_t *k) {
    if (!k || !k->reading) return;
    k->reading = 0;
}

/* ── 写事务 ── */
bool myds_kvdsk_write_start(myds_kvdsk_t *k, uint32_t slot) {
    if (!k || slot >= k->slots || k->reading || k->writing) return false;
    /* 先置空：这一步之后中途掉电，该槽是「空」而不是「半份数据」。 */
    if (!myds_kvdsk_slot_clear(k, slot)) return false;
    k->writing = 1;
    k->write_slot = slot;
    k->write_crc = cks_begin();
    return true;
}

/* 契约同上：失败即终止写事务（该槽保持之前的「空」状态 —— write_start 已经清过）。 */
bool myds_kvdsk_write_chunk(myds_kvdsk_t *k, uint64_t offset, const void *src, uint64_t bytes) {
    if (!k || !k->writing || (!src && bytes)) return false;
    if (offset + bytes > k->slot_stride) { k->writing = 0; return false; }
    if (!write_at(k->fd, src, (size_t)bytes, payload_offset(k, k->write_slot) + offset)) {
        k->writing = 0;
        return false;
    }
    k->write_crc = cks_update(k->write_crc, src, (size_t)bytes);
    return true;
}

void myds_kvdsk_write_abort(myds_kvdsk_t *k) {
    if (k) k->writing = 0;
}

/* 收尾。参数不合法或落盘失败 ⇒ 返回 false，且该槽保持「空」（write_start 已清）。 */
bool myds_kvdsk_write_finish(myds_kvdsk_t *k, uint32_t slot, uint32_t position,
                             uint64_t prefix_hash, uint64_t used_bytes) {
    if (!k || !k->writing || slot != k->write_slot) return false;
    k->writing = 0;
    if (position == 0u || used_bytes == 0u || used_bytes > k->slot_stride) return false;
    const uint64_t crc = k->write_crc;
    if (!entry_write_body(k, slot, 0u, prefix_hash, used_bytes, crc, KVDSK_STATE_VALID)) return false;
    return entry_write_position(k, slot, position) ? true : false;
}

/* ── 哈希与身份 ── */
uint64_t myds_kvdsk_hash_tokens(const int32_t *tokens, uint32_t count) {
    uint64_t h = 1469598103934665603ull;      /* FNV-1a 64 offset basis */
    if (!tokens) return h;
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t v = (uint32_t)tokens[i];
        for (int b = 0; b < 4; b++) {          /* 显式小端：与编译目标无关 */
            h ^= (uint64_t)((v >> (8 * b)) & 0xFFu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

uint64_t myds_kvdsk_identity(const char *tag, const uint64_t *values, uint32_t count) {
    uint64_t h = 1469598103934665603ull;
    if (tag) {
        for (const unsigned char *p = (const unsigned char *)tag; *p; p++) {
            h ^= (uint64_t)*p;
            h *= 1099511628211ull;
        }
    }
    for (uint32_t i = 0; values && i < count; i++) {
        const uint64_t v = values[i];
        for (int b = 0; b < 8; b++) {
            h ^= (v >> (8 * b)) & 0xFFu;
            h *= 1099511628211ull;
        }
    }
    return h;
}

/* ── 打开 ── */
myds_kvdsk_t *myds_kvdsk_open(const char *path, uint64_t identity,
                              uint32_t slots, uint64_t slot_stride,
                              uint64_t budget_bytes) {
    g_open_error[0] = 0;
    if (!path || slots == 0 || slot_stride == 0) {
        set_error("参数非法（path/slots/slot_stride）", path);
        return NULL;
    }
    const uint64_t payload_base = align_up((uint64_t)KVDSK_HEADER_BYTES + (uint64_t)slots * KVDSK_ENTRY_BYTES, 4096u);
    const uint64_t file_bytes = payload_base + slot_stride * (uint64_t)slots;
    if (budget_bytes && file_bytes > budget_bytes) {
        set_error("超出磁盘预算（下调槽数或提高预算）", path);
        return NULL;
    }

    ensure_parent_dir(path);
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        snprintf(g_open_error, sizeof(g_open_error), "open 失败: %s [%s]", strerror(errno), path);
        return NULL;
    }

    myds_kvdsk_t *k = (myds_kvdsk_t *)calloc(1, sizeof(*k));
    if (!k) { close(fd); set_error("内存不足", path); return NULL; }
    k->fd = fd;
    k->slots = slots;
    k->identity = identity;
    k->slot_stride = slot_stride;
    k->payload_base = payload_base;
    k->file_bytes = file_bytes;

    /* 头是否可复用：magic / 版本 / 槽数 / 槽容量 / 身份 全部一致才算。
     * 任何一项不符都重建 —— 旧字节在新环境里没有意义，宁可全丢。 */
    unsigned char hdr[KVDSK_HEADER_BYTES];
    int reuse = 0;
    if (read_at(fd, hdr, sizeof(hdr), 0)) {
        if (memcmp(hdr, KVDSK_MAGIC, 8) == 0 &&
            get_u32(hdr + 8) == KVDSK_VERSION &&
            get_u32(hdr + 12) == slots &&
            get_u64(hdr + 16) == identity &&
            get_u64(hdr + 24) == slot_stride) {
            reuse = 1;
        }
    }

    if (!reuse) {
        unsigned char fresh[KVDSK_HEADER_BYTES];
        memset(fresh, 0, sizeof(fresh));
        memcpy(fresh, KVDSK_MAGIC, 8);
        put_u32(fresh + 8, KVDSK_VERSION);
        put_u32(fresh + 12, slots);
        put_u64(fresh + 16, identity);
        put_u64(fresh + 24, slot_stride);
        put_u64(fresh + 32, file_bytes);
        if (!write_at(fd, fresh, sizeof(fresh), 0) || ftruncate(fd, (off_t)file_bytes) != 0) {
            snprintf(g_open_error, sizeof(g_open_error), "初始化失败: %s [%s]", strerror(errno), path);
            close(fd); free(k); return NULL;
        }
        /* 条目表清零 ⇒ 全部为空。 */
        unsigned char zero[KVDSK_ENTRY_BYTES * 64];
        memset(zero, 0, sizeof(zero));
        for (uint32_t s = 0; s < slots; s += 64) {
            const uint32_t n = (slots - s < 64u) ? (slots - s) : 64u;
            if (!write_at(fd, zero, (size_t)n * KVDSK_ENTRY_BYTES,
                          (uint64_t)KVDSK_HEADER_BYTES + (uint64_t)s * KVDSK_ENTRY_BYTES)) {
                snprintf(g_open_error, sizeof(g_open_error), "条目表初始化失败: %s [%s]", strerror(errno), path);
                close(fd); free(k); return NULL;
            }
        }
        MYDS_LOG_I("kvdisk: %s 重建（identity %016llx，%u 槽 × %.1f MiB，共 %.0f MiB）",
                   path, (unsigned long long)identity, slots,
                   (double)slot_stride / 1048576.0, (double)file_bytes / 1048576.0);
    } else {
        MYDS_LOG_I("kvdisk: %s 复用（identity %016llx，%u 槽，%u 条有效）",
                   path, (unsigned long long)identity, slots, myds_kvdsk_live_entries(k));
    }
    return k;
}

void myds_kvdsk_close(myds_kvdsk_t *k) {
    if (!k) return;
    if (k->fd >= 0) close(k->fd);
    free(k);
}
