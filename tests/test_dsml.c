/* tests/test_dsml.c — DSML 解析的失败诊断
 *
 * 重点是「失败时说得清为什么」。原实现只返回 false，调用方只能报「格式不合」，
 * 而「被 max_tokens 切在半个属性上」与「模型换了写法」是完全不同的两件事，
 * 处理方式也不同（前者该报 length，后者该隔离）。
 *
 * 不需要模型：直接喂字符串。
 *
 * ★ 标记必须用 MYDS_DSML_S 拼，不能手写 `<|DSML|` —— 真实标记是全角竖线
 *   U+FF5C（`\xef\xbd\x9c`），半角 `|` 是另一个字节序列，解析器根本不认。
 *   第一版就是这么错的，全部用例一起失败。
 */
#include "myds_server_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("[FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

/* myds_dsml.c 的 JSON 缓冲原语依赖 json_escape_alloc（定义在 proto_openai.c）。
 * 这里给一个等价实现，避免为测解析器把整个协议层拖进来。 */
char *json_escape_alloc(const char *src, size_t length) {
    if (!src) return NULL;
    size_t cap = length * 6 + 1, w = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    for (size_t i = 0; i < length; ++i) {
        unsigned char c = (unsigned char)src[i];
        if (c == '"' || c == '\\') { out[w++] = '\\'; out[w++] = (char)c; }
        else if (c < 0x20) { w += (size_t)snprintf(out + w, cap - w, "\\u%04x", c); }
        else out[w++] = (char)c;
    }
    out[w] = '\0';
    return out;
}

static void free_calls(myds_tool_call_t *calls, size_t n) {
    for (size_t i = 0; i < n; ++i) { free(calls[i].id); free(calls[i].name); free(calls[i].arguments); }
    free(calls);
}

/* 合法结构，用真正的全角标记拼出来。 */
#define O   "<" MYDS_DSML_S " calls>"
#define C   "</" MYDS_DSML_S " calls>"
#define IO  "<" MYDS_DSML_S " invoke name=\""
#define IC  "</" MYDS_DSML_S " invoke>"
#define PO  "<" MYDS_DSML_S " parameter name=\""
#define PC  "</" MYDS_DSML_S " parameter>"

static void test_good_block_parses(void) {
    const char *good =
        O IO "get_weather\">"
        PO "city\" string=\"true\">Beijing" PC
        IC C;
    size_t clen = 0, count = 0;
    myds_tool_call_t *calls = NULL;
    CHECK(myds_parse_completion_tool_calls(good, strlen(good), &clen, &calls, &count));
    CHECK(count == 1);
    if (count == 1) {
        CHECK(strcmp(calls[0].name, "get_weather") == 0);
        CHECK(strstr(calls[0].arguments, "Beijing") != NULL);
    }
    CHECK(myds_dsml_classify(good, strlen(good), &clen) == MYDS_DSML_COMPLETE);
    free_calls(calls, count);
}

static void test_truncated_mid_attribute(void) {
    /* 被 max_tokens 砍在半个属性上 —— 最常见的真实形态。 */
    const char *bad = O IO "f\">" PO "a";
    size_t clen = 0, count = 0;
    myds_tool_call_t *calls = NULL;
    CHECK(!myds_parse_completion_tool_calls(bad, strlen(bad), &clen, &calls, &count));
    CHECK(count == 0);
    CHECK(calls == NULL);
    CHECK(myds_dsml_classify(bad, strlen(bad), &clen) == MYDS_DSML_TRUNCATED);
}

static void test_parameter_without_string_attr(void) {
    /* |DSML| 方言要求 string=；缺了就是结构错（<function_calls> 方言才允许省略）。 */
    const char *bad = O IO "f\">" PO "a\">v" PC IC C;
    size_t clen = 0, count = 0;
    myds_tool_call_t *calls = NULL;
    CHECK(!myds_parse_completion_tool_calls(bad, strlen(bad), &clen, &calls, &count));
    CHECK(myds_dsml_classify(bad, strlen(bad), &clen) == MYDS_DSML_MALFORMED);
}

static void test_invoke_without_name(void) {
    const char *bad = O "<" MYDS_DSML_S " invoke>" IC C;
    size_t clen = 0, count = 0;
    myds_tool_call_t *calls = NULL;
    CHECK(!myds_parse_completion_tool_calls(bad, strlen(bad), &clen, &calls, &count));
}

static void test_content_boundary_before_bad_block(void) {
    /* 正文必须保留：坏块隔离，但不能把正文一起吞掉。 */
    const char *bad = "hello there\n\n" O IO "f\">" PO "a";
    size_t clen = 0, count = 0;
    myds_tool_call_t *calls = NULL;
    CHECK(!myds_parse_completion_tool_calls(bad, strlen(bad), &clen, &calls, &count));
    CHECK(clen == strlen("hello there"));
}

static void test_no_block_is_not_an_error(void) {
    const char *plain = "just an answer, no tools here";
    size_t clen = 0, count = 0;
    myds_tool_call_t *calls = NULL;
    CHECK(myds_parse_completion_tool_calls(plain, strlen(plain), &clen, &calls, &count));
    CHECK(count == 0);
    CHECK(clen == strlen(plain));
    CHECK(myds_dsml_classify(plain, strlen(plain), &clen) == MYDS_DSML_NONE);
}

static void test_function_calls_dialect(void) {
    /* 另一套方言：只在【结尾】才算工具块，且参数可以没有 string= */
    const char *ok =
        "sure\n\n<function_calls>\n<invoke name=\"f\">\n"
        "<parameter name=\"n\">42</parameter>\n</invoke>\n</function_calls>";
    size_t clen = 0, count = 0;
    myds_tool_call_t *calls = NULL;
    CHECK(myds_parse_completion_tool_calls(ok, strlen(ok), &clen, &calls, &count));
    CHECK(count == 1);
    CHECK(clen == strlen("sure"));
    if (count == 1) {
        /* 无 string= 时按形状推断：42 是数字，不加引号 */
        CHECK(strstr(calls[0].arguments, ": 42") != NULL);
    }
    free_calls(calls, count);

    /* 同样的标记出现在正文中间 ⇒ 不认，整段保留（这是安全边界） */
    const char *mid =
        "the markup looks like <function_calls><invoke name=\"f\"></invoke>"
        "</function_calls> but it is only an example.";
    CHECK(myds_dsml_classify(mid, strlen(mid), &clen) == MYDS_DSML_NONE);
    CHECK(clen == strlen(mid));

    /* 截断的 <function_calls> 也要报 TRUNCATED，否则会被当正文发出去 */
    const char *cut =
        "answer\n\n<function_calls>\n<invoke name=\"f\">\n<parameter name=\"a\"";
    CHECK(myds_dsml_classify(cut, strlen(cut), &clen) == MYDS_DSML_TRUNCATED);
    CHECK(clen == strlen("answer"));
}

int main(void) {
    test_good_block_parses();
    test_truncated_mid_attribute();
    test_parameter_without_string_attr();
    test_invoke_without_name();
    test_content_boundary_before_bad_block();
    test_no_block_is_not_an_error();
    test_function_calls_dialect();

    if (failures) {
        printf("\n%d dsml test(s) FAILED\n", failures);
        return 1;
    }
    printf("PASS: DSML parse failures keep their reason and their content boundary\n");
    return 0;
}
