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
    test_page_is_html();
    myds_trace_reset();

    if (failures) {
        printf("\n%d trace test(s) FAILED\n", failures);
        return 1;
    }
    printf("PASS: trace ring buffer, JSON shape, null-vs-zero, escaping, eviction\n");
    return 0;
}
