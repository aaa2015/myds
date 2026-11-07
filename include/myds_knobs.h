/* myds_knobs.h — what the process was told, versus what it read.
 *
 * The recurring failure in this project is not compute, it is a config value
 * that nobody reads: `MYDS_TOPK_THRESHOLD=1` sat in the production config for
 * days while no code path consumed it. `--print-config` exists to make that
 * class of mistake fail loudly instead of silently.
 *
 * The inventory is generated from source by tools/knob_inventory.py and covers
 * both direct getenv("MYDS_*") sites and keys passed to env helpers (and .m
 * files), so it cannot be reproduced by a plain grep.
 */
#ifndef MYDS_KNOBS_H
#define MYDS_KNOBS_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const char *name;      /* "MYDS_..." */
    const char *file;      /* first source location that reads it */
    unsigned line;
    int direct;            /* 1 = getenv("MYDS_*") at the call site */
} myds_knob_t;

/* Size of the generated inventory and read-only access to it. */
size_t myds_knob_count(void);
const myds_knob_t *myds_knob_at(size_t index);

/* NULL when no code path in this build reads that exact key. */
const myds_knob_t *myds_knob_find(const char *name);

typedef struct {
    char name[96];
    char value[256];
    const myds_knob_t *knob;   /* NULL => nothing reads this variable */
} myds_knob_seen_t;

/* Collect MYDS_* entries from environ, sorted by name.
 * Returns the entry count, or -1 when `out` cannot hold them all. */
int myds_knobs_from_environ(myds_knob_seen_t *out, size_t cap);

/* Number of MYDS_* entries in environ. */
int myds_knobs_environ_count(void);

/* Print the report. mode_mask selects sections:
 *   1 = set here and readable, with the source location that reads it
 *   2 = set here and unreadable (the failure this tool exists for)
 *   4 = known but unset, listed only when verbose
 * Returns the number of unreadable keys so main() can exit non-zero. */
int myds_knobs_report(unsigned mode_mask, bool verbose);

/* Effective (post-parse) values recorded by modules at init time. Modules that
 * never call this simply do not appear, which is still better than presenting
 * the raw string as though it were the value actually in force. */
void myds_knobs_record_effective(const char *name, const char *resolved);
/* 本 build 认识的 knob 名字集合的 FNV-1a 64 指纹。
 *
 * 二进制与 myds.conf 现在是成对的：旧二进制不认新键会拒绝启动（退出码 2），
 * 所以回滚时只换二进制会把服务打死。这个指纹让「配对」可机器判定：
 * 部署脚本用 tools/knob_fingerprint.py 从源码算出期望值，与 `myds --knob-fingerprint`
 * 的输出比对；不相等就说明 conf 里有这个二进制不认识的键。
 *
 * 只取决于源码（knobs_generated.h），与编译器/SDK 无关。 */
uint64_t myds_knobs_fingerprint(void);

int myds_knobs_effective_count(void);
const char *myds_knobs_effective(const char *name);

#endif /* MYDS_KNOBS_H */
