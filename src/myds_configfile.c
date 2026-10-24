/* myds_configfile.c — 见 myds_configfile.h 的设计说明 */
#include "myds_configfile.h"
#include "myds_log.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CF_MAX_LINE 4096

/* 去掉行尾空白。返回新的长度。 */
static size_t rstrip(char *s, size_t n) {
    while (n > 0) {
        const char c = s[n - 1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') { s[--n] = '\0'; }
        else break;
    }
    return n;
}

/* 找行尾注释：要求 `#` 前面是空白，避免把值里的 `#` 当注释。
 * 只处理值部分，调用方已定位到 '='。 */
static void strip_trailing_comment(char *v) {
    for (char *p = v; *p; ++p) {
        if (*p == '#' && p > v && (p[-1] == ' ' || p[-1] == '\t')) { *p = '\0'; break; }
    }
    /* 再去掉注释前的尾随空白 */
    size_t n = strlen(v);
    while (n > 0 && (v[n - 1] == ' ' || v[n - 1] == '\t')) v[--n] = '\0';
}

static int is_comment_or_blank(const char *line) {
    const char *p = line;
    while (*p == ' ' || *p == '\t') ++p;
    return *p == '\0' || *p == '#';
}

int myds_configfile_load(const char *path, bool override_env) {
    /* override_env 自 2026-10-24 起被忽略：配置文件永远优先。
     * 保留参数只为不改动头文件签名。 */
    (void)override_env;
    if (!path || !*path) return -1;
    FILE *f = fopen(path, "r");
    if (!f) {
        MYDS_LOG_W("config: 打不开 %s（%s）—— 继续用环境变量与代码默认值",
                   path, strerror(errno));
        return -1;
    }

    char line[CF_MAX_LINE];
    int set = 0, skipped_env = 0, skipped_bad = 0;

    while (fgets(line, sizeof(line), f)) {
        if (is_comment_or_blank(line)) continue;

        size_t n = rstrip(line, strlen(line));
        if (n == 0) continue;

        char *eq = strchr(line, '=');
        if (!eq) { skipped_bad++; continue; }
        *eq = '\0';

        /* 去 key 两端空白 */
        char *key = line;
        while (*key == ' ' || *key == '\t') ++key;
        char *kend = key + strlen(key);
        while (kend > key && (kend[-1] == ' ' || kend[-1] == '\t')) *--kend = '\0';

        char *val = eq + 1;
        while (*val == ' ' || *val == '\t') ++val;
        strip_trailing_comment(val);

        /* 只接受 MYDS_ 前缀：这个文件不该能改 PATH / HOME 之类 */
        if (strncmp(key, "MYDS_", 5) != 0) { skipped_env++; continue; }
        if (!*key) { skipped_bad++; continue; }

        /* 2026-10-24：不再读取环境变量、也不再为「覆盖」报警。
         *
         * 配置文件是 MYDS_* 的唯一来源：读到什么就写什么，无条件 setenv。
         * 进程启动时的环境变量里即使有同名项，也会被这里直接覆盖，且不产生任何日志 ——
         * 因为按现在的设计那不算「冲突」，只是配置生效。
         *
         * （setenv 仍是必需的：引擎各处通过 getenv 取值，配置文件只是把值灌进去。） */
        if (setenv(key, val, 1) != 0) { skipped_bad++; continue; }
        set++;
    }
    fclose(f);

    MYDS_LOG_I("config: 已加载 %s —— 生效 %d 条", path, set);
    if (skipped_env)
        MYDS_LOG_W("config: 跳过 %d 条非 MYDS_ 前缀的键（只允许 MYDS_*）", skipped_env);
    if (skipped_bad)
        MYDS_LOG_W("config: 跳过 %d 行格式错误（需要 KEY=VALUE）", skipped_bad);
    return set;
}

/* 按字典序输出所有 MYDS_* 环境变量，便于 diff 与人工比对 */
extern char **environ;

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

int myds_configfile_dump_env(const char *path) {
    if (!path || !*path) return -1;

    size_t cap = 256, n = 0;
    const char **items = (const char **)malloc(cap * sizeof(*items));
    if (!items) return -1;

    for (char **e = environ; e && *e; ++e) {
        if (strncmp(*e, "MYDS_", 5) != 0) continue;
        if (n == cap) {
            cap *= 2;
            const char **ni = (const char **)realloc(items, cap * sizeof(*items));
            if (!ni) { free(items); return -1; }
            items = ni;
        }
        items[n++] = *e;
    }
    qsort(items, n, sizeof(*items), cmp_str);

    FILE *f = fopen(path, "w");
    if (!f) { free(items); return -1; }
    fprintf(f, "# myds 生效配置（由 `myds --dump-config %s` 从运行中的进程导出）\n", path);
    fprintf(f, "# 生成时间由调用方记录。格式 KEY=VALUE，# 开头为注释。\n");
    fprintf(f, "# 优先级：命令行 > 环境变量 > 本文件 > 代码默认值。\n\n");
    for (size_t i = 0; i < n; i++) fprintf(f, "%s\n", items[i]);
    fclose(f);
    free(items);

    MYDS_LOG_I("config: 已导出 %zu 条 MYDS_* 到 %s", n, path);
    return (int)n;
}

int myds_configfile_count(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[CF_MAX_LINE];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        if (is_comment_or_blank(line)) continue;
        if (strchr(line, '=')) n++;
    }
    fclose(f);
    return n;
}

