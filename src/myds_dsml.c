/* myds_dsml.c — DSML 工具调用解析（官方 DeepSeek-V4.1 格式）
 *
 * 2026-11-01 从 myds_server.c 纯机械拆分而来：函数体与注释未改。
 *
 * 这里只负责【把模型输出里的工具块解析成结构体】，不负责把它重新渲染成
 * OpenAI / Responses 的 JSON —— 那是 proto_openai 的事。分界依据：
 * 解析要知道模型怎么写，渲染要知道客户端要什么，两者会朝不同方向演进。
 *
 * 构建 JSON 的缓冲原语（dsml_buf_t / dsml_put*）仍留在这里并被导出：
 * 它们唯一的使用者就是「拼工具调用 JSON」，放在解析旁边才不会分叉。
 */
#include "myds_server.h"
#include "myds_server_internal.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>

/* ══════════════════════════════════════════════════════════════════════════
 * 解析模型输出里的 DSML 工具调用（官方 DeepSeek-V4.1 格式）
 * 官方的 parse_message_from_completion_text 是【严格】实现（格式稍不合就抛错）；
 * 这里做【宽容】解析：任何不合格式之处一律退回「整段当纯文本」，
 * 绝不因为模型一处手滑就把回答吞掉。
 * ══════════════════════════════════════════════════════════════════════════ */
#define MYDS_DSML_S "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"   /* ｜DSML｜（全角竖线） */

bool dsml_put(dsml_buf_t *b, const char *p, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 64;
        while (cap < b->len + n + 1) {
            if (cap > SIZE_MAX / 2) return false;
            cap *= 2;
        }
        char *grown = realloc(b->data, cap);
        if (!grown) return false;
        b->data = grown; b->cap = cap;
    }
    if (n) memcpy(b->data + b->len, p, n);
    b->len += n; b->data[b->len] = 0;
    return true;
}
char *dsml_dup(const char *p, size_t n) {
    char *s = malloc(n + 1);
    if (!s) return NULL;
    memcpy(s, p, n); s[n] = 0;
    return s;
}
const char *dsml_find(const char *pos, const char *end, const char *lit) {
    size_t n = strlen(lit);
    if (!n || pos > end || (size_t)(end - pos) < n) return NULL;
    for (const char *q = pos; q + n <= end; ++q) if (!memcmp(q, lit, n)) return q + n;
    return NULL;
}
static bool dsml_match(const char *p, const char *end, const char *lit) {
    size_t n = strlen(lit);
    return (size_t)(end - p) >= n && !memcmp(p, lit, n);
}
static const char *dsml_skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    return p;
}
/* 读 key="value" 属性；返回闭合引号之后的位置。 */
static const char *dsml_attr(const char *p, const char *end, const char *key,
                             const char **value, size_t *value_len) {
    if (!p) return NULL;
    if (!(p = dsml_find(p, end, key))) return NULL;
    if (p >= end || *p != '"') return NULL;
    const char *v = ++p;
    while (p < end && *p != '"') ++p;
    if (p >= end) return NULL;
    *value = v; *value_len = (size_t)(p - v);
    return p + 1;
}

/* 成功：content_len 收工具块之前的正文字节数；calls_out 与 count_out 收解析出的调用（调用方负责释放）。
 * 没有工具块时 count=0 且返回 true；格式不合格式时返回 false（调用方按纯文本处理）。 */
/* ── 两套工具调用方言（2026-11-01 真机实测）────────────────────────
 *
 * 模板要求 `<｜DSML｜calls>`，但实测部分措辞下模型稳定输出另一套同构写法
 * `<function_calls>`（不带全角竖线）。两套都是同一结构，只是标签名不同。
 *
 * 安全规则（关键）：**只接受"结尾块"** —— 工具块必须是输出的最后一段
 * （后面只允许空白）。这样正文里引用了这种标记的讲解/代码片段不会被误吞。
 * DSML 那套沿用原有行为（它本来就是特殊 token，正文里几乎不可能出现）。
 */
typedef struct {
    const char *open_tag, *close_tag, *invoke_open, *invoke_close;
    const char *param_open, *param_close;
    bool        trailing_only;
} dsml_dialect_t;

static const char *dsml_dialect_pick(const char *text, const char *end, dsml_dialect_t *out) {
    static const char *const S = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
    static dsml_dialect_t dsml, plain;
    static bool init = false;
    if (!init) {
        init = true;
        /* 用静态缓冲拼标签，避免每次请求都 malloc。 */
        static char b[8][64];
        const size_t sl = strlen(S);
        snprintf(b[0], 64, "<%.*s calls>",   (int)sl, S);
        snprintf(b[1], 64, "</%.*s calls>",  (int)sl, S);
        snprintf(b[2], 64, "<%.*s invoke",   (int)sl, S);
        snprintf(b[3], 64, "</%.*s invoke>", (int)sl, S);
        snprintf(b[4], 64, "<%.*s parameter",(int)sl, S);
        snprintf(b[5], 64, "</%.*s parameter>", (int)sl, S);
        dsml = (dsml_dialect_t){ b[0], b[1], b[2], b[3], b[4], b[5], false };
        static char c[6][64];
        snprintf(c[0], 64, "<function_calls>");
        snprintf(c[1], 64, "</function_calls>");
        snprintf(c[2], 64, "<invoke");
        snprintf(c[3], 64, "</invoke>");
        snprintf(c[4], 64, "<parameter");
        snprintf(c[5], 64, "</parameter>");
        plain = (dsml_dialect_t){ c[0], c[1], c[2], c[3], c[4], c[5], true };
    }
    /* 返回的是【开标签之后】的位置（与 dsml_find 的契约一致），不是标签本身。 */
    const char *dbody = dsml_find(text, end, dsml.open_tag);
    if (dbody) { if (out) *out = dsml; return dbody; }
    const char *hit = dsml_find(text, end, plain.open_tag);
    if (!hit) return NULL;
    /* trailing_only：开标签之后必须能一路走到闭合标签，且闭合之后只剩空白。 */
    const char *close = dsml_find(hit, end, plain.close_tag);
    if (!close) return NULL;                     /* 截断 ⇒ 不算（交给 TRUNCATED 逻辑） */
    for (const char *q = close; q < end; ++q)
        if (*q != ' ' && *q != '\n' && *q != '\r' && *q != '\t') return NULL;
    if (out) *out = plain;
    return hit;   /* 开标签之后的位置 */
}

/* 分类：这一段文本里的 DSML 工具块到底处于什么状态。
 *
 * 之所以要单独出一个分类器，而不是让调用方从「parse 返回 false」去猜：
 *   · 没有工具块     ⇒ 正常正文，什么都不用做
 *   · 有开有闭、结构有效 ⇒ 正常工具调用
 *   · 有开无闭       ⇒ 被 max_tokens 截断 / 模型自己没写完
 *   · 有开有闭、结构坏 ⇒ 模型吐出了不合语法的块
 * 后两种的正确处置是【隔离】：既不能当工具调用执行，也不能把原始 XML 当正文发出去。
 * 原实现两种都当成「格式不合 ⇒ 整段当正文」，于是在截断时把半截
 * `<|DSML| calls><|DSML| invoke ...` 原样发给了客户端。 */
myds_dsml_state_t myds_dsml_classify(const char *text, size_t len, size_t *content_len) {
    if (content_len) *content_len = len;
    if (!text || !len) return MYDS_DSML_NONE;
    const char *end = text + len;
    dsml_dialect_t d;
    const char *body = dsml_dialect_pick(text, end, &d);
    const char *open_tag = d.open_tag;
    const char *close_tag = d.close_tag;
    if (!body) {
        /* 有开标签但方言判定不成立（如 plain 方言被截断）⇒ 也要报 TRUNCATED，
         * 否则截断的 <function_calls> 会整段当正文发出去。 */
        static const char *plain_open = "<function_calls>";
        const char *raw = dsml_find(text, end, plain_open);
        if (raw && !dsml_find(raw, end, "</function_calls>")) {
            const char *ce = raw - strlen(plain_open);
            while (ce > text && (ce[-1] == '\n' || ce[-1] == '\r' || ce[-1] == ' ' || ce[-1] == '\t')) --ce;
            if (content_len) *content_len = (size_t)(ce - text);
            return MYDS_DSML_TRUNCATED;
        }
        return MYDS_DSML_NONE;
    }

    /* 正文只到开标签之前（顺带削掉尾部空白，工具块通常跟在两个换行后面）。 */
    const char *ce = body - strlen(open_tag);
    while (ce > text && (ce[-1] == '\n' || ce[-1] == '\r' || ce[-1] == ' ' || ce[-1] == '\t')) --ce;
    if (content_len) *content_len = (size_t)(ce - text);

    if (!dsml_find(body, end, close_tag)) return MYDS_DSML_TRUNCATED;

    /* 有闭标签 ⇒ 交给真正的解析器判断内部结构是否有效。
     * 这里不返回 calls（调用方本来就要再解析一次），只做状态判定。 */
    size_t cl = len;
    myds_tool_call_t *calls = NULL;
    size_t n = 0;
    const bool ok = myds_parse_completion_tool_calls(text, len, &cl, &calls, &n);
    for (size_t k = 0; k < n; ++k) { free(calls[k].id); free(calls[k].name); free(calls[k].arguments); }
    free(calls);
    return ok ? MYDS_DSML_COMPLETE : MYDS_DSML_MALFORMED;
}

/* 另一套方言探测（2026-11-01 实测发现）。
 *
 * 提示词模板要求模型输出 `<｜DSML｜calls>`，但实测这台机器上的 1017 权重
 * 在 tools 已渲染（日志 has_tools=1）的情况下仍然吐 `<function_calls>` ——
 * 即不带 `｜DSML｜` 前缀的同构写法。解析器认不出来，于是整块当正文发出去。
 *
 * 这属于【另一件事】（方言不匹配），不在本次「截断隔离」范围内，所以这里只做观测：
 * 一旦正文里出现这类标记就告警，让它不再静默。不去猜着兼容 —— 多认一种方言
 * 要连带定义转义/嵌套/歧义规则，得单独设计和验证。 */
bool dsml_looks_like_foreign_dialect(const char *text, size_t len) {
    if (!text || !len) return false;
    const char *end = text + len;
    return dsml_find(text, end, "<function_calls>") != NULL ||
           dsml_find(text, end, "<invoke name=") != NULL;
}

bool myds_parse_completion_tool_calls(const char *text, size_t len, size_t *content_len,
                                      myds_tool_call_t **calls_out, size_t *count_out) {
    *calls_out = NULL; *count_out = 0; *content_len = len;
    const char *end = text + len;
    dsml_dialect_t d;
    if (!dsml_dialect_pick(text, end, &d)) return true;   /* 没有工具块 */
    const char *open_tag = d.open_tag;
    const char *close_tag = d.close_tag;
    const char *invoke_open = d.invoke_open;
    const char *invoke_close = d.invoke_close;
    const char *param_open = d.param_open;
    const char *param_close = d.param_close;
    const char *body = dsml_find(text, end, open_tag);
    if (!body) return true;
    const char *content_end = body - strlen(open_tag);
    while (content_end > text && (content_end[-1] == '\n' || content_end[-1] == '\r' ||
                                  content_end[-1] == ' ' || content_end[-1] == '\t')) --content_end;
    *content_len = (size_t)(content_end - text);

    size_t cap = 4, count = 0;
    myds_tool_call_t *calls = calloc(cap, sizeof(*calls));
    if (!calls) return false;
    const char *p = body;
    bool ok = true;
    /* 失败时必须能说出「在哪一步坏的」。原实现只返回 false，调用方只能报
     * 「格式不合」——模型换了措辞、或被 max_tokens 切在半个属性上，都长这样，
     * 而它们的处理方式完全不同。offset 是相对工具块起点的字节位置。 */
    const char *fail_at = NULL;
    const char *fail_why = NULL;
    while (ok) {
        p = dsml_skip_ws(p, end);
        if (dsml_match(p, end, close_tag)) { p += strlen(close_tag); break; }
        if (!dsml_match(p, end, invoke_open)) {
            ok = false; fail_at = p; fail_why = "expected <invoke> or </calls>"; break;
        }
        p += strlen(invoke_open);
        const char *name = NULL; size_t name_len = 0;
        const char *at_name = p;
        if (!(p = dsml_attr(p, end, "name=", &name, &name_len))) {
            ok = false; fail_at = at_name; fail_why = "<invoke> without name=\""; break;
        }
        p = dsml_skip_ws(p, end);
        if (p >= end || *p != '>') {
            ok = false; fail_at = p; fail_why = "<invoke> name attribute not followed by '>'"; break;
        }
        ++p;

        dsml_buf_t args = {0};
        if (!dsml_put(&args, "{", 1)) { ok = false; break; }
        size_t n_params = 0;
        while (ok) {
            p = dsml_skip_ws(p, end);
            if (dsml_match(p, end, invoke_close)) { p += strlen(invoke_close); break; }
            if (!dsml_match(p, end, param_open)) {
                ok = false; fail_at = p; fail_why = "expected <parameter> or </invoke>"; break;
            }
            p += strlen(param_open);
            const char *pk = NULL; size_t pk_len = 0;
            const char *sv = NULL; size_t sv_len = 0;
            const char *at_pname = p;
            if (!(p = dsml_attr(p, end, "name=", &pk, &pk_len))) {
                ok = false; fail_at = at_pname; fail_why = "<parameter> without name=\""; break;
            }
            /* string= 在 <function_calls> 方言里常常不写（实测模型只写 name=）。
             * DSML 方言里它是必需的，缺失就说明结构不对。 */
            {
                const char *after_name = p;
                if (!(p = dsml_attr(p, end, "string=", &sv, &sv_len))) {
                    if (!d.trailing_only) {
                        ok = false; fail_at = after_name;
                        fail_why = "<parameter> without string=\" (required by the |DSML| dialect)";
                        break;
                    }
                    p = after_name;
                }
            }
            p = dsml_skip_ws(p, end);
            if (p >= end || *p != '>') {
                ok = false; fail_at = p;
                fail_why = "<parameter> attributes not followed by '>'"; break;
            }
            ++p;
            const char *vend = dsml_find(p, end, param_close);
            if (!vend) {
                ok = false; fail_at = p; fail_why = "<parameter> never closed"; break;
            }
            size_t val_len = (size_t)(vend - strlen(param_close) - p);
            while (val_len && (p[val_len - 1] == '\n' || p[val_len - 1] == '\r')) --val_len;
            /* 有 string= 就照它；没有（<function_calls> 方言常见）就按值的形状推断：
             * 能当成 JSON 标量/容器的原样放，否则当字符串加引号。 */
            bool is_str;
            if (sv_len) {
                is_str = (sv_len == 4 && !memcmp(sv, "true", 4));
            } else {
                const char *v = p;
                size_t n = val_len;
                while (n && (*v == ' ' || *v == '\n' || *v == '\r' || *v == '\t')) { ++v; --n; }
                while (n && (v[n-1] == ' ' || v[n-1] == '\n' || v[n-1] == '\r' || v[n-1] == '\t')) --n;
                bool scalarish = false;
                if (n && (*v == '{' || *v == '[')) scalarish = true;
                else if (n == 4 && (!memcmp(v,"true",4) || !memcmp(v,"null",4))) scalarish = true;
                else if (n == 5 && !memcmp(v,"false",5)) scalarish = true;
                else if (n) {
                    /* 纯数字（含负号/小数点/指数）才算 */
                    size_t k = 0;
                    if (v[0] == '-' || v[0] == '+') k = 1;
                    size_t digits = 0;
                    for (; k < n; ++k) {
                        if (v[k] >= '0' && v[k] <= '9') { digits++; continue; }
                        if (v[k] == '.' || v[k] == 'e' || v[k] == 'E' || v[k] == '+' || v[k] == '-') continue;
                        break;
                    }
                    scalarish = (k == n && digits > 0);
                }
                is_str = !scalarish;
            }
            if (n_params++ && !dsml_put(&args, ", ", 2)) { ok = false; break; }
            if (!dsml_put(&args, "\"", 1) || !dsml_put(&args, pk, pk_len) || !dsml_put(&args, "\": ", 3)) {
                ok = false; break;
            }
            if (is_str) {
                /* json_escape_alloc 只做转义、不加引号 ⇒ 引号要自己补 */
                char *esc = json_escape_alloc(p, val_len);
                if (!esc) { ok = false; break; }
                bool put = dsml_put(&args, "\"", 1) && dsml_put(&args, esc, strlen(esc)) &&
                           dsml_put(&args, "\"", 1);
                free(esc);
                if (!put) { ok = false; break; }
            } else if (!dsml_put(&args, p, val_len)) { ok = false; break; }
            p = vend;
        }
        if (ok && !dsml_put(&args, "}", 1)) ok = false;
        if (!ok) { free(args.data); break; }

        if (count == cap) {
            size_t grown_cap = cap * 2;
            myds_tool_call_t *grown = realloc(calls, grown_cap * sizeof(*calls));
            if (!grown) { free(args.data); ok = false; break; }
            calls = grown; cap = grown_cap;
        }
        char idbuf[32];
        snprintf(idbuf, sizeof(idbuf), "call_%zu", count + 1);
        myds_tool_call_t *tc = &calls[count];
        memset(tc, 0, sizeof(*tc));
        tc->id = dsml_dup(idbuf, strlen(idbuf));
        tc->name = dsml_dup(name, name_len);
        tc->arguments = args.data;
        if (!tc->id || !tc->name || !tc->arguments) { ok = false; break; }
        ++count;
    }
    if (!ok) {
        /* 只在失败时说话（正常请求不打）。给出「第几步、为什么」，让调用方不必
         * 再自己猜 —— 截断与语法错误在这一层是截然不同的两件事。 */
        MYDS_LOG_W("dsml: tool block rejected at offset %ld (%zu bytes into the block): %s",
                   (long)(fail_at ? fail_at - text : -1),
                   (size_t)(fail_at ? fail_at - body : 0),
                   fail_why ? fail_why : "unspecified");
        for (size_t k = 0; k < count; ++k) { free(calls[k].id); free(calls[k].name); free(calls[k].arguments); }
        free(calls);
        return false;
    }
    if (!count) { free(calls); *count_out = 0; return true; }
    *calls_out = calls; *count_out = count;
    return true;
}


/* 小整数格式化（index 之类）——别用 %s 拼裸数字 */
bool dsml_putf(dsml_buf_t *b, const char *fmt, ...) {
    char tmp[64];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    return n > 0 && (size_t)n < sizeof(tmp) && dsml_put(b, tmp, (size_t)n);
}
bool dsml_put_escaped(dsml_buf_t *b, const char *p, size_t n) {
    char *e = json_escape_alloc(p, n);
    if (!e) return false;
    bool ok = dsml_put(b, e, strlen(e));
    free(e);
    return ok;
}

void myds_free_tool_calls(myds_tool_call_t *calls, size_t count) {
    for (size_t i = 0; i < count; ++i) { free(calls[i].id); free(calls[i].name); free(calls[i].arguments); }
    free(calls);
}
