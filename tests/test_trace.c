/* tests/test_trace.c — 请求追踪缓冲与 JSON 渲染
 *
 * 不需要模型：这一层是纯数据与格式化。
 *
 * 测的每一条都对应一个会被误读的失败模式：
 *   - 未测量的阶段必须是 null，不能是 0（0 会被读成「测了，没花时间」）
 *   - 环形缓冲绕回后必须仍然按 seq 找得到最近的，且最旧的确实被挤掉
 *   - 列表必须最新在前（页面默认看刚发生的）
 *   - 客户端可控字符串必须转义（一个引号能让整个 JSON 失效）
 *   - 缓冲不足必须返回 false，不能发出截断的 JSON
 */
#include "myds_trace.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("[FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

static void fill(myds_trace_rec_t *r, const char *path, double engine_s) {
    memset(r, 0, sizeof(*r));
    snprintf(r->path, sizeof(r->path), "%s", path);
    snprintf(r->client, sizeof(r->client), "127.0.0.1:1");
    snprintf(r->finish, sizeof(r->finish), "stop");
    r->phases.queue_s = 0.001;
    r->phases.render_s = 0.002;
    r->phases.setup_s = 0.003;
    r->phases.engine_s = engine_s;
    r->phases.prefill_s = engine_s * 0.8;
    r->phases.decode_s = engine_s * 0.1;
    r->phases.unattributed_s = engine_s * 0.1;
    r->prompt_tokens = 100;
    r->cached_tokens = 0;
    r->out_tokens = 20;
    r->stream = 1;
}

static void test_empty(void) {
    myds_trace_reset();
    myds_trace_init(8);
    CHECK(myds_trace_count() == 0);
    char buf[512];
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    CHECK(strstr(buf, "\"count\":0") != NULL);
    CHECK(strstr(buf, "\"records\":[]") != NULL);
}

static void test_unmeasured_is_null_not_zero(void) {
    myds_trace_reset();
    myds_trace_init(8);
    myds_trace_rec_t r;
    fill(&r, "/v1/chat/completions", 1.5);
    /* 三个阶段明确「没测」 */
    r.phases.render_s = -1.0;
    r.phases.prefill_s = -1.0;
    r.phases.unattributed_s = NAN;
    myds_trace_record(&r);

    char buf[2048];
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    /* 三个都必须 null —— 这是整个模块最重要的一条不变量。 */
    CHECK(strstr(buf, "\"render\":null") != NULL);
    CHECK(strstr(buf, "\"prefill\":null") != NULL);
    CHECK(strstr(buf, "\"unattributed\":null") != NULL);
    /* 测过的必须是数字 */
    CHECK(strstr(buf, "\"engine\":1.500000") != NULL);
    CHECK(strstr(buf, "\"queue\":0.001000") != NULL);
}

static void test_ring_wraps_and_evicts(void) {
    myds_trace_reset();
    myds_trace_init(4);
    myds_trace_rec_t r;
    for (int i = 0; i < 10; ++i) {
        fill(&r, "/r", (double)i);
        myds_trace_record(&r);
    }
    CHECK(myds_trace_count() == 4);            /* 不增长 */
    CHECK(myds_trace_capacity() == 4);

    myds_trace_rec_t got;
    /* 最旧的一条是第 7 次写入（seq 7），第 1..6 已被挤掉。 */
    CHECK(myds_trace_get(0, &got));
    CHECK(got.seq == 7);
    CHECK(myds_trace_get(3, &got));
    CHECK(got.seq == 10);
    CHECK(!myds_trace_get(4, &got));           /* 越界 */

    /* 被挤掉的 seq 必须查不到，且要能与「从来没存在过」区分开 */
    CHECK(!myds_trace_get_by_seq(1, &got));
    CHECK(myds_trace_get_by_seq(10, &got));
}

static void test_list_newest_first(void) {
    myds_trace_reset();
    myds_trace_init(8);
    myds_trace_rec_t r;
    for (int i = 0; i < 3; ++i) { fill(&r, "/r", 1.0); myds_trace_record(&r); }

    char buf[4096];
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    const char *a = strstr(buf, "\"seq\":3");
    const char *b = strstr(buf, "\"seq\":2");
    const char *c = strstr(buf, "\"seq\":1");
    CHECK(a && b && c);
    CHECK(a < b && b < c);                     /* 3, 2, 1 的顺序 */
}

static void test_path_escaped(void) {
    myds_trace_reset();
    myds_trace_init(8);
    myds_trace_rec_t r;
    fill(&r, "/v1/x\"y\\z", 1.0);
    myds_trace_record(&r);
    char buf[2048];
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    /* 原始引号不能出现在输出里，否则 JSON 直接失效 */
    CHECK(strstr(buf, "x\\\"y\\\\z") != NULL);
    CHECK(strstr(buf, "x\"y") == NULL);
}

static void test_layers_null_when_unmeasured(void) {
    myds_trace_reset();
    myds_trace_init(8);
    myds_trace_rec_t r;
    fill(&r, "/r", 1.0);
    r.layer_count = 0;
    myds_trace_record(&r);
    char buf[2048];
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    /* 空数组会被画成一条零线；null 才能让页面说「未测量」 */
    CHECK(strstr(buf, "\"layers\":null") != NULL);

    /* 有数据时是数组 */
    myds_trace_reset();
    myds_trace_init(8);
    fill(&r, "/r", 1.0);
    r.layer_count = 3;
    r.layer_ms[0] = 1.5f; r.layer_ms[1] = 0.5f; r.layer_ms[2] = 2.25f;
    myds_trace_record(&r);
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    CHECK(strstr(buf, "\"layers\":[1.5000,0.5000,2.2500]") != NULL);
}

static void test_buffer_too_small_is_an_error(void) {
    myds_trace_reset();
    myds_trace_init(64);
    myds_trace_rec_t r;
    for (int i = 0; i < 64; ++i) { fill(&r, "/v1/chat/completions", 1.0); myds_trace_record(&r); }
    /* 64 条记录塞进 200 字节必然放不下：必须报错，不能返回截断的 JSON。 */
    char small[200];
    CHECK(!myds_trace_list_json(small, sizeof(small)));
}

static void test_detail_by_seq(void) {
    myds_trace_reset();
    myds_trace_init(8);
    myds_trace_rec_t r;
    fill(&r, "/v1/responses", 2.0);
    myds_trace_record(&r);
    char buf[2048];
    CHECK(!myds_trace_detail_json(999, buf, sizeof(buf)));   /* 没有这条 */
    CHECK(myds_trace_detail_json(1, buf, sizeof(buf)));
    CHECK(strstr(buf, "\"record\":") != NULL);
    CHECK(strstr(buf, "/v1/responses") != NULL);
}

static void test_capacity_clamped(void) {
    myds_trace_reset();
    myds_trace_init(0);                        /* 0 ⇒ 默认容量 */
    CHECK(myds_trace_capacity() > 0);
    myds_trace_reset();
    myds_trace_init(100000);                   /* 过大 ⇒ 夹到上限 */
    CHECK(myds_trace_capacity() == 512);
}

/* 页面里「耗时」列的口径必须是端到端（含 engine）。
 *
 * 这条守的不是 C 代码，而是那段内联 JS 的算法：它曾经漏了 engine，于是
 * 514 ms 的请求在列表里显示成 0.12 ms —— 数值看着合理，量级错 4000 倍，
 * 而这类错在页面上根本看不出来。这里把同一公式在 C 里算一遍对账。 */
static void test_wall_includes_engine(void) {
    myds_trace_reset();
    myds_trace_init(8);
    myds_trace_rec_t r;
    fill(&r, "/v1/chat/completions", 0.514276);
    r.phases.queue_s = 0.000034;
    r.phases.render_s = 0.000013;
    r.phases.setup_s = 0.000075;
    myds_trace_record(&r);

    myds_trace_rec_t got;
    CHECK(myds_trace_get(0, &got));
    /* 与页面 JS 完全相同的公式 */
    const double wall = got.phases.queue_s + got.phases.render_s +
                        got.phases.setup_s + got.phases.engine_s;
    /* 漏掉 engine 会得到 0.000122，差 4000 倍，任何正常容差都能分开。 */
    CHECK(wall > 0.514 && wall < 0.515);

    /* 早退请求：engine 未测量 ⇒ 端到端不可知，页面显示 “—” 而不是 0。 */
    myds_trace_reset();
    myds_trace_init(8);
    fill(&r, "/v1/chat/completions", -1.0);
    r.phases.render_s = -1.0;
    r.phases.setup_s = -1.0;
    myds_trace_record(&r);
    CHECK(myds_trace_get(0, &got));
    CHECK(got.phases.engine_s < 0.0);
}

/* 页面里的 JS 必须是【能被解析】的。这一条比它听起来重要：
 *
 * 页面是 C 字符串拼出来的，一个漏掉的闭合引号会让整段 JS 变成字符串内容，
 * 结果是「页面能打开、表格空白」—— 服务端没有任何错误，单测也全绿。
 * 我写这段时在同一地方错了两次（'</td>+ 少了收尾引号）。
 *
 * 这里不追求完整校验：检查引号成对、括号成对、每行都以引号收尾，
 * 足以抓住「拼接时漏引号」这一类。真正的语法检查交给构建外的一次 node --check。 */
static int count_char(const char *s, char c) {
    int n = 0;
    for (; *s; ++s) if (*s == c) n++;
    return n;
}

static void test_page_js_is_well_formed(void) {
    const char *html = myds_trace_page_html();
    const char *beg = strstr(html, "<script>");
    const char *end = strstr(html, "</script>");
    CHECK(beg && end && end > beg);
    if (!beg || !end) return;

    const size_t len = (size_t)(end - (beg + 8));
    char *js = (char *)malloc(len + 1);
    CHECK(js != NULL);
    if (!js) return;
    memcpy(js, beg + 8, len);
    js[len] = '\0';

    /* 引号必须成对（含模板字符串的反引号）。偶数是最低门槛，
     * 但恰好能抓住「漏一个收尾引号」—— 那正是这里的真实缺陷。 */
    CHECK(count_char(js, '\'') % 2 == 0);
    CHECK(count_char(js, '\"') % 2 == 0);
    CHECK(count_char(js, '`') % 2 == 0);
    /* 花括号也必须配平，否则后面的事件绑定会被解析进错误的块 */
    CHECK(count_char(js, '{') == count_char(js, '}'));
    CHECK(count_char(js, '(') == count_char(js, ')'));

    /* 列表渲染必须用模板字符串：'...'+x+'...' 的写法每行都要自己补引号，
     * 这个文件在这里错过两次。断言它用的是模板形式，防止有人改回去。 */
    CHECK(strstr(js, "return `<tr data-seq=") != NULL);

    free(js);
}

/* token 必须由服务端注入到页面脚本里。
 *
 * 背景（真实故障）：页面是从 /trace?token=X 打开的，但页面里的
 * fetch('/trace/api/requests') 是**另一个请求**，不会继承地址栏的 query。
 * 结果：页面 200 打得开、表格永远空、控制台每 3 秒一次 404 ——
 * 用户看到的就是「请求了但没效果」。
 *
 * 这两条断言把那个状态钉死：模板必须留占位符（否则没法注入），
 * 展开后必须不含占位符且带着 token。 */
/* 页面脚本只能引用模板里真实存在的元素。
 *
 * 这条来自一次真实故障：删掉标题栏里的 count/容量显示时，一并删掉了 id="hdr"
 * 的 span，而脚本里还有三处 getElementById('hdr') —— 请求出错时页面不会显示
 * 任何错误，因为写错误信息的那行自己抛了 TypeError。看起来就是「一直空白」。
 *
 * 检查方式：脚本里出现的 id 必须在 HTML 里作为 id="..." 存在。 */
/* resp_id 必须进 JSON，且能被转义安全地带出去。
 *
 * 用户报「那次请求慢」时手里有的是响应里的 id，不是追踪里的序号 ——
 * 两者对得上，这个页面才有排查价值。 */
/* begin / complete 两段式：请求到达即可见，结束时回填。
 *
 * 为什么要单测这条：单段式写入会让长请求在整个执行期间不可见 ——
 * 用户看到的正是「只有等数据全返回才出现这一条」。所以断言三件事：
 * begin 之后立刻能查到、in_flight 为真、各阶段是「未测量」而不是 0。 */
static void test_in_flight_visible_immediately(void) {
    myds_trace_reset();
    myds_trace_init(8);

    const uint64_t seq = myds_trace_begin("/v1/chat/completions", "1.2.3.4:5", "resp_x");
    CHECK(seq > 0);
    CHECK(myds_trace_count() == 1);          /* 还没结束就能查到 */

    myds_trace_rec_t got;
    CHECK(myds_trace_get_by_seq(seq, &got));
    CHECK(got.in_flight == 1);
    CHECK(strcmp(got.resp_id, "resp_x") == 0);
    /* 阶段必须是 -1（未测量）。若是 0，页面会显示「瞬时完成」。 */
    CHECK(got.phases.engine_s < 0.0);
    CHECK(got.phases.prefill_s < 0.0);
    CHECK(got.ttft_s < 0.0);

    char json[2048];
    CHECK(myds_trace_list_json(json, sizeof(json)));
    CHECK(strstr(json, "\"in_flight\":true") != NULL);

    /* 结束回填：同一 seq，in_flight 清零，阶段有值 */
    myds_trace_rec_t done;
    memset(&done, 0, sizeof(done));
    snprintf(done.path, sizeof(done.path), "/v1/chat/completions");
    snprintf(done.resp_id, sizeof(done.resp_id), "resp_x");
    snprintf(done.finish, sizeof(done.finish), "stop");
    done.phases.engine_s = 1.25;
    done.phases.queue_s = 0.01;
    done.ttft_s = 0.30;
    done.prefill_tps = 100.0;
    done.decode_tps = 40.0;
    myds_trace_complete(seq, &done);

    CHECK(myds_trace_count() == 1);          /* 没有变成两条 */
    CHECK(myds_trace_get_by_seq(seq, &got));
    CHECK(got.in_flight == 0);
    CHECK(got.phases.engine_s > 1.0);
    CHECK(strcmp(got.finish, "stop") == 0);

    /* complete 到已被挤掉的 seq：静默丢弃，不新增记录 */
    myds_trace_reset();
    myds_trace_init(2);
    (void)myds_trace_begin("/a", "c", "r1");
    (void)myds_trace_begin("/b", "c", "r2");
    (void)myds_trace_begin("/c", "c", "r3");   /* r1 被挤掉 */
    myds_trace_complete(1, &done);
    CHECK(myds_trace_count() == 2);
}

static void test_resp_id_recorded(void) {
    myds_trace_reset();
    myds_trace_init(8);
    myds_trace_rec_t r;
    fill(&r, "/v1/chat/completions", 1.0);
    snprintf(r.resp_id, sizeof(r.resp_id), "resp_myds_7_abc123");
    myds_trace_record(&r);

    char buf[2048];
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    CHECK(strstr(buf, "\"resp_id\":\"resp_myds_7_abc123\"") != NULL);

    /* 空 id 也要输出成空串而不是缺失：页面用 rec.resp_id||'—' 兜底，
     * 字段缺失会让那一列在不同行之间抖动。 */
    myds_trace_reset();
    myds_trace_init(8);
    fill(&r, "/v1/chat/completions", 1.0);
    myds_trace_record(&r);
    CHECK(myds_trace_list_json(buf, sizeof(buf)));
    CHECK(strstr(buf, "\"resp_id\":\"\"") != NULL);
}

static void test_page_script_refs_exist(void) {
    const char *html = myds_trace_page_html();
    const char *sb = strstr(html, "<script>");
    CHECK(sb != NULL);
    if (!sb) return;

    /* 收集 getElementById('x') 里的 x */
    const char *p = sb;
    const char *needle = "getElementById('";
    const size_t nl = strlen(needle);
    int checked = 0;
    while ((p = strstr(p, needle)) != NULL) {
        p += nl;
        const char *end = strchr(p, '\'');
        if (!end) break;
        char id[64];
        const size_t n = (size_t)(end - p);
        if (n == 0 || n >= sizeof(id)) continue;
        memcpy(id, p, n);
        id[n] = '\0';
        /* 有少数元素是脚本运行时才建出来的（画布 lc 在选中请求后才有）。
         * 把它们列成白名单，比放宽断言好：放宽了，真正的漏 id 也会被放过。 */
        static const char *const runtime_created[] = { "lc" };
        bool skip = false;
        for (size_t k = 0; k < sizeof(runtime_created)/sizeof(runtime_created[0]); ++k)
            if (!strcmp(id, runtime_created[k])) { skip = true; break; }
        if (skip) continue;

        /* 只在 <script> 之前找：脚本自己也会拼出 id="x" 的模板字符串
         * （sysbox 就是这么重建自己的），拿整份文档搜会把脚本里的字符串
         * 当成真实元素，断言就永远为真 —— 这一版的第一稿正是如此。 */
        char attr[80];
        snprintf(attr, sizeof(attr), "id=\"%s\"", id);
        const size_t markup_len = (size_t)(sb - html);
        bool found = false;
        for (size_t i = 0; i + strlen(attr) <= markup_len; ++i) {
            if (!memcmp(html + i, attr, strlen(attr))) { found = true; break; }
        }
        CHECK(found);
        checked++;
    }
    CHECK(checked >= 3);   /* rows / sysbox / detail 至少三个 */
}

static void test_token_is_injected_into_page(void) {
    const char *tpl = myds_trace_page_html();
    CHECK(strstr(tpl, "__MYDS_TRACE_TOKEN__") != NULL);
    CHECK(strstr(tpl, "__MYDS_TRACE_PARAM__") != NULL);

    size_t len = 0;
    char *page = myds_trace_page_html_for_token("abc123", &len);
    CHECK(page != NULL);
    if (page) {
        CHECK(len == strlen(page));
        CHECK(strstr(page, "__MYDS_TRACE_TOKEN__") == NULL);
        CHECK(strstr(page, "__MYDS_TRACE_PARAM__") == NULL);
        CHECK(strstr(page, "abc123") != NULL);
        /* 参数名由服务端拼出，页面里必须真的出现它，否则 API 还是不带凭据 */
        CHECK(strstr(page, "tok" "en=") != NULL);
        free(page);
    }

    /* 含非 URL 安全字符的 token：不注入，页面仍生成（API 会 404，显式失败） */
    char *unsafe = myds_trace_page_html_for_token("a\"b", NULL);
    CHECK(unsafe != NULL);
    if (unsafe) {
        CHECK(strstr(unsafe, "__MYDS_TRACE_TOKEN__") == NULL);
        CHECK(strstr(unsafe, "a\"b") == NULL);   /* 没被拼进 JS 字面量 */
        free(unsafe);
    }

    /* NULL token 也不能崩 */
    char *none = myds_trace_page_html_for_token(NULL, NULL);
    CHECK(none != NULL);
    free(none);
}

static void test_page_is_html(void) {
    const char *html = myds_trace_page_html();
    CHECK(html != NULL);
    CHECK(strstr(html, "<!DOCTYPE html>") != NULL);
    /* 页面必须自带数据源与「未测量」的说明，否则打开是一片空白。 */
    CHECK(strstr(html, "/trace/api/requests") != NULL);
    CHECK(strstr(html, "MYDS_PROFILE_PREFILL") != NULL);
}

int main(void) {
    test_empty();
    test_unmeasured_is_null_not_zero();
    test_ring_wraps_and_evicts();
    test_list_newest_first();
    test_path_escaped();
    test_layers_null_when_unmeasured();
    test_buffer_too_small_is_an_error();
    test_detail_by_seq();
    test_capacity_clamped();
    test_wall_includes_engine();
    test_page_js_is_well_formed();
    test_in_flight_visible_immediately();
    test_resp_id_recorded();
    test_page_script_refs_exist();
    test_token_is_injected_into_page();
    test_page_is_html();
    myds_trace_reset();

    if (failures) {
        printf("\n%d trace test(s) FAILED\n", failures);
        return 1;
    }
    printf("PASS: trace ring buffer, JSON shape, null-vs-zero, escaping, eviction\n");
    return 0;
}
