/* myds_knobs.c — see include/myds_knobs.h */
#include "myds_knobs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_knobs_generated.h"

size_t myds_knob_count(void) { return MYDS_KNOB_COUNT; }

const myds_knob_t *myds_knob_at(size_t index) {
    return index < MYDS_KNOB_COUNT ? &myds_knobs_generated[index] : NULL;
}

/* The generated table is in source order. Index it once into name order so
 * lookups are binary searches instead of linear scans. */
static const myds_knob_t *g_sorted[MYDS_KNOB_COUNT];
static int g_sorted_ready = 0;

static int by_name(const void *a, const void *b) {
    const myds_knob_t *x = *(const myds_knob_t *const *)a;
    const myds_knob_t *y = *(const myds_knob_t *const *)b;
    return strcmp(x->name, y->name);
}

static void build_sorted(void) {
    if (g_sorted_ready) return;
    for (size_t i = 0; i < MYDS_KNOB_COUNT; ++i) g_sorted[i] = &myds_knobs_generated[i];
    qsort(g_sorted, MYDS_KNOB_COUNT, sizeof(g_sorted[0]), by_name);
    g_sorted_ready = 1;
}

const myds_knob_t *myds_knob_find(const char *name) {
    if (!name) return NULL;
    build_sorted();
    size_t lo = 0, hi = MYDS_KNOB_COUNT;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int order = strcmp(g_sorted[mid]->name, name);
        if (order == 0) return g_sorted[mid];
        if (order < 0) lo = mid + 1; else hi = mid;
    }
    return NULL;
}

extern char **environ;

static int is_myds(const char *entry) { return entry && strncmp(entry, "MYDS_", 5) == 0; }

int myds_knobs_environ_count(void) {
    int count = 0;
    for (char **e = environ; e && *e; ++e) if (is_myds(*e)) ++count;
    return count;
}

static int seen_compare(const void *a, const void *b) {
    const myds_knob_seen_t *x = a, *y = b;
    return strcmp(x->name, y->name);
}

int myds_knobs_from_environ(myds_knob_seen_t *out, size_t cap) {
    if (!out) return -1;
    const int total = myds_knobs_environ_count();
    if ((size_t)total > cap) return -1;
    int n = 0;
    for (char **e = environ; e && *e; ++e) {
        if (!is_myds(*e)) continue;
        const char *eq = strchr(*e, '=');
        const size_t name_len = eq ? (size_t)(eq - *e) : strlen(*e);
        if (name_len == 0 || name_len >= sizeof(out[n].name)) continue;
        memcpy(out[n].name, *e, name_len);
        out[n].name[name_len] = 0;
        const char *value = eq ? eq + 1 : "";
        size_t vlen = strlen(value);
        if (vlen >= sizeof(out[n].value)) vlen = sizeof(out[n].value) - 1;
        memcpy(out[n].value, value, vlen);
        out[n].value[vlen] = 0;
        out[n].knob = myds_knob_find(out[n].name);
        ++n;
    }
    qsort(out, (size_t)n, sizeof(out[0]), seen_compare);
    return n;
}

/* ---------------------------------------------------------------- effective */

typedef struct { char name[96]; char value[256]; } effective_t;
static effective_t g_effective[64];
static int g_effective_n = 0;

void myds_knobs_record_effective(const char *name, const char *resolved) {
    if (!name || !resolved || g_effective_n >= (int)(sizeof(g_effective) / sizeof(g_effective[0]))) return;
    for (int i = 0; i < g_effective_n; ++i) {
        if (!strcmp(g_effective[i].name, name)) {
            snprintf(g_effective[i].value, sizeof(g_effective[i].value), "%s", resolved);
            return;
        }
    }
    snprintf(g_effective[g_effective_n].name, sizeof(g_effective[0].name), "%s", name);
    snprintf(g_effective[g_effective_n].value, sizeof(g_effective[0].value), "%s", resolved);
    ++g_effective_n;
}

int myds_knobs_effective_count(void) { return g_effective_n; }

const char *myds_knobs_effective(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < g_effective_n; ++i) if (!strcmp(g_effective[i].name, name)) return g_effective[i].value;
    return NULL;
}

/* ------------------------------------------------------------------- report */

int myds_knobs_report(unsigned mode_mask, bool verbose) {
    static myds_knob_seen_t seen[1024];
    const int n = myds_knobs_from_environ(seen, sizeof(seen) / sizeof(seen[0]));
    if (n < 0) { fprintf(stderr, "myds_knobs: too many MYDS_* variables in environ\n"); return -1; }

    int unread = 0, readable = 0;
    for (int i = 0; i < n; ++i) { if (seen[i].knob) ++readable; else ++unread; }

    printf("knob inventory : %u keys known to this build "
           "(%u direct getenv, %u through env helpers)\n",
           (unsigned)MYDS_KNOB_COUNT,
           (unsigned)MYDS_KNOB_DIRECT_COUNT,
           (unsigned)MYDS_KNOB_WRAPPED_COUNT);
    printf("environment    : %d MYDS_* variables set\n", n);

    if (mode_mask & 2u) {
        printf("\nUNREAD (%d) -- set here, no code path reads them:\n", unread);
        if (!unread) printf("  none\n");
        for (int i = 0; i < n; ++i) {
            if (seen[i].knob) continue;
            printf("  %-40s = %s\n", seen[i].name, seen[i].value);
        }
    }

    if (mode_mask & 1u) {
        printf("\nEFFECTIVE (%d) -- set and read by this build:\n", readable);
        if (!readable) printf("  none\n");
        for (int i = 0; i < n; ++i) {
            if (!seen[i].knob) continue;
            const char *resolved = myds_knobs_effective(seen[i].name);
            printf("  %-40s = %-18s  %s:%u%s%s\n",
                   seen[i].name, seen[i].value,
                   seen[i].knob->file, seen[i].knob->line,
                   resolved ? "  -> " : "",
                   resolved ? resolved : "");
        }
    }

    if ((mode_mask & 4u) && verbose) {
        printf("\nKNOWN BUT UNSET (default applies):\n");
        size_t shown = 0;
        for (size_t i = 0; i < MYDS_KNOB_COUNT; ++i) {
            int present = 0;
            for (int j = 0; j < n; ++j)
                if (!strcmp(seen[j].name, myds_knobs_generated[i].name)) { present = 1; break; }
            if (!present) { printf("  %s\n", myds_knobs_generated[i].name); ++shown; }
        }
        if (!shown) printf("  none\n");
    }

    if (unread) {
        printf("\nVERDICT: FAIL -- %d key(s) set here are read by nothing.\n"
               "  Either the name is wrong (rename/typo) or the switch is dead.\n"
               "  Cross-check with: python3 tools/knob_inventory.py --check\n", unread);
        return unread;
    }
    printf("\nVERDICT: PASS -- every MYDS_* variable set here has a reading site.\n");
    return 0;
}

/* ── knob 集合指纹（2026-11-07，§156）──────────────────────────────
 *
 * 为什么需要：二进制与 myds.conf 现在是【成对】的。旧二进制不认识
 * MYDS_INDEXER_ROWS4_MIN，配新 conf 会拒绝启动（退出码 2），
 * 而回滚时如果只换二进制不换 conf，服务就起不来 —— 2026-11-07 实测踩到两轮。
 *
 * 指纹 = 本 build 认识的【全部 knob 名字】按字节序排序后做 FNV-1a 64。
 * 它只取决于 include/myds_knobs_generated.h，也就是只取决于源码，
 * 与编译器/SDK 无关 —— 所以两台机器上同一源码必然得到同一个指纹。
 *
 * conf 侧的期望值由 tools/knob_fingerprint.py 从源码算出；部署脚本比对两者。
 * 不相等就意味着「这份 conf 里有这个二进制不认识的键」，回滚前必须先对账。
 */
uint64_t myds_knobs_fingerprint(void) {
    /* 名字先收集排序：生成表的顺序取决于源码扫描顺序，不能直接拿来算哈希。 */
    static const char *names[1024];
    size_t n = 0;
    for (size_t i = 0; i < MYDS_KNOB_COUNT && n < sizeof(names)/sizeof(names[0]); ++i)
        names[n++] = myds_knobs_generated[i].name;
    /* 插入排序：260 个元素，且只在启动/CLI 用一次。 */
    for (size_t i = 1; i < n; ++i) {
        const char *key = names[i];
        size_t j = i;
        while (j > 0 && strcmp(names[j - 1], key) > 0) { names[j] = names[j - 1]; --j; }
        names[j] = key;
    }
    uint64_t h = 1469598103934665603ULL;          /* FNV-1a offset basis */
    for (size_t i = 0; i < n; ++i) {
        for (const char *p = names[i]; *p; ++p) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
        h ^= 0x1Fu; h *= 1099511628211ULL;         /* 分隔符，避免前缀混淆 */
    }
    h ^= (uint64_t)n; h *= 1099511628211ULL;       /* 计入个数 */
    return h;
}
