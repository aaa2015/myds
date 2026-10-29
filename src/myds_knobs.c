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
