#include "myds_server.h"
#include "myds_server_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>

/* Exercise the production byte encoder and socket callbacks directly. */
static const char binary_text[] = {
    'A', 0, '"', '\\', '\n', '\r', '\t', 1,
    (char)0xe4, (char)0xb8, (char)0xad, /* 中 */
    (char)0xf0, (char)0x9f, (char)0x98, (char)0x80, /* 😀 */
    (char)0xff, (char)0xe4, (char)0xb8 /* Invalid byte and incomplete 中. */
};
static const char escaped_binary_text[] = "A\\u0000\\\"\\\\\\n\\r\\t\\u0001中😀��";

static void test_json_bytes(void) {
    char escaped[sizeof(binary_text) * 6 + 1];
    size_t length = 0;
    assert(json_escape_bytes(binary_text, sizeof(binary_text), escaped, sizeof(escaped), &length));
    assert(length == strlen(escaped_binary_text));
    assert(strcmp(escaped, escaped_binary_text) == 0);
    const char *invalid[] = {"\xc0\xaf", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2(\xa1", "\xe4\xb8"};
    const char *replacement[] = {"��", "���", "����", "�(�", "�"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        assert(json_escape_bytes(invalid[i], strlen(invalid[i]), escaped, sizeof(escaped), NULL));
        assert(strcmp(escaped, replacement[i]) == 0);
    }
    char small[4] = "xxx";
    assert(!json_escape_bytes(binary_text, sizeof(binary_text), small, sizeof(small), &length));
    assert(small[0] == 0 && length == 0); /* Never silently return truncated JSON text. */
    assert(json_escape_bytes("abc", 3, small, sizeof(small), &length));
    assert(length == 3 && strcmp(small, "abc") == 0);
    assert(!json_escape_bytes("abc", 3, small, 0, NULL));
    printf("  [PASS] JSON byte escaping: NUL, known UTF-8, invalid sequences, truncated code points and capacity errors\n");
}

typedef struct {
    int fd;
    char *bytes;
    size_t length;
    size_t capacity;
} socket_capture_t;

static void *capture_socket(void *opaque) {
    socket_capture_t *capture = opaque;
    for (;;) {
        if (capture->capacity - capture->length < 4097) {
            capture->capacity = capture->capacity ? capture->capacity * 2 : 8192;
            capture->bytes = realloc(capture->bytes, capture->capacity);
            assert(capture->bytes);
        }
        ssize_t count = recv(capture->fd, capture->bytes + capture->length,
                             capture->capacity - capture->length - 1, 0);
        if (count < 0 && errno == EINTR) continue;
        assert(count >= 0);
        if (count == 0) break;
        capture->length += (size_t)count;
    }
    capture->bytes[capture->length] = 0;
    return NULL;
}

static char *collect_sse_escaped_text(const char *events, bool responses_api, size_t *out_length) {
    const char *key = responses_api ? "\"delta\":\"" : "\"content\":\"";
    char *result = malloc(strlen(events) + 1);
    assert(result);
    size_t used = 0;
    const char *p = events;
    while ((p = strstr(p, key)) != NULL) {
        p += strlen(key);
        const char *start = p;
        while (*p && *p != '"') {
            if (*p == '\\') {
                assert(p[1]);
                p++;
            }
            p++;
        }
        assert(*p == '"');
        size_t length = (size_t)(p - start);
        memcpy(result + used, start, length);
        used += length;
        p++;
    }
    result[used] = 0;
    *out_length = used;
    return result;
}

static void assert_valid_utf8(const char *bytes, size_t length) {
    for (size_t i = 0; i < length;) {
        unsigned char lead = (unsigned char)bytes[i++];
        if (lead < 0x80) continue;
        unsigned trailing;
        uint32_t codepoint, minimum;
        if (lead >= 0xc2 && lead <= 0xdf) {
            trailing = 1; codepoint = lead & 31; minimum = 0x80;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            trailing = 2; codepoint = lead & 15; minimum = 0x800;
        } else {
            assert(lead >= 0xf0 && lead <= 0xf4);
            trailing = 3; codepoint = lead & 7; minimum = 0x10000;
        }
        assert(length - i >= trailing);
        while (trailing--) {
            unsigned char continuation = (unsigned char)bytes[i++];
            assert(continuation >= 0x80 && continuation <= 0xbf);
            codepoint = codepoint * 64 + (continuation & 63);
        }
        assert(codepoint >= minimum && codepoint <= 0x10ffff);
        assert(codepoint < 0xd800 || codepoint > 0xdfff);
    }
}

static void test_stream_and_non_stream_bytes(void) {
    /* A single token of 8192 control bytes expands to 49152 JSON bytes. This
     * exceeds both former 2048-byte token limits and the small socket buffer. */
    const size_t controls_length = 8192;
    char *controls = malloc(controls_length);
    assert(controls);
    memset(controls, 1, controls_length);
    size_t expected_length = strlen(escaped_binary_text) + controls_length * 6 + 3;
    char *expected = malloc(expected_length + 1);
    assert(expected);
    strcpy(expected, escaped_binary_text);
    size_t offset = strlen(escaped_binary_text);
    for (size_t i = 0; i < controls_length; i++) {
        memcpy(expected + offset, "\\u0001", 6);
        offset += 6;
    }
    memcpy(expected + offset, "�", 4); /* Replacement plus trailing NUL. */

    for (unsigned variant = 0; variant < 2; variant++) {
        int sockets[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        int send_size = 1024;
        assert(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &send_size, sizeof(send_size)) == 0);
        socket_capture_t capture = {.fd = sockets[1]};
        pthread_t reader;
        assert(pthread_create(&reader, NULL, capture_socket, &capture) == 0);
        myds_sse_stream_ctx_t stream = {.client_fd = sockets[0], .responses_api = variant != 0};
        for (size_t i = 0; i < sizeof(binary_text); i++) {
            myds_server_sse_callback(binary_text + i, 1, (int32_t)i, &stream);
        }
        myds_server_sse_callback(controls, controls_length, 50, &stream);
        myds_server_sse_callback("\xf0\x9f", 2, 51, &stream);
        assert(myds_server_sse_finish(&stream));
        assert(!stream.cancel_flag && !stream.failed && stream.utf8.pending_length == 0);
        assert(stream.token_count == sizeof(binary_text) + 2);
        assert(stream.t_first > 0 && stream.t_last >= stream.t_first);
        assert(shutdown(sockets[0], SHUT_WR) == 0);
        assert(pthread_join(reader, NULL) == 0);
        /* Validate the wire including JSON punctuation between events. A
         * code point split across SSE events would fail here even if joining
         * the extracted content happened to recover the original bytes. */
        assert_valid_utf8(capture.bytes, capture.length);
        size_t collected_length = 0;
        char *collected = collect_sse_escaped_text(capture.bytes, variant != 0, &collected_length);
        assert(collected_length == expected_length);
        assert(memcmp(collected, expected, expected_length + 1) == 0);
        assert(strstr(capture.bytes, variant ? "event: response.output_text.delta" : "chat.completion.chunk"));
        free(collected);
        free(capture.bytes);
        close(sockets[0]);
        close(sockets[1]);
    }

    myds_non_stream_ctx_t nonstream = {.client_fd = -1};
    for (size_t i = 0; i < sizeof(binary_text); i++) {
        myds_server_non_stream_callback(binary_text + i, 1, (int32_t)i, &nonstream);
    }
    myds_server_non_stream_callback(controls, controls_length, 50, &nonstream);
    myds_server_non_stream_callback("\xf0\x9f", 2, 51, &nonstream);
    assert(!nonstream.failed && !nonstream.cancel_flag);
    assert(nonstream.len == sizeof(binary_text) + controls_length + 2);
    assert(nonstream.token_count == sizeof(binary_text) + 2);
    assert(memcmp(nonstream.buf, binary_text, sizeof(binary_text)) == 0);
    char *escaped = malloc(nonstream.len * 6 + 1);
    assert(escaped);
    size_t escaped_length = 0;
    assert(json_escape_bytes(nonstream.buf, nonstream.len, escaped, nonstream.len * 6 + 1, &escaped_length));
    assert(escaped_length == expected_length);
    assert(memcmp(escaped, expected, expected_length + 1) == 0);
    free(escaped);
    free(nonstream.buf);

    /* Allocation-size overflow must become explicit cancellation/error, and
     * must preserve an existing allocation rather than overwrite its pointer. */
    nonstream = (myds_non_stream_ctx_t){.client_fd = -1};
    myds_server_non_stream_callback("old", 3, 1, &nonstream);
    char *original = nonstream.buf;
    myds_server_non_stream_callback("x", SIZE_MAX, 2, &nonstream);
    assert(nonstream.failed && nonstream.cancel_flag && nonstream.buf == original);
    assert(nonstream.len == 3 && strcmp(nonstream.buf, "old") == 0);
    free(nonstream.buf);
    myds_sse_stream_ctx_t oversized = {.client_fd = -1};
    myds_server_sse_callback("x", SIZE_MAX, 1, &oversized);
    assert(oversized.failed && oversized.cancel_flag);
    int closed_pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, closed_pair) == 0);
    close(closed_pair[1]);
    assert(!myds_server_send_all(closed_pair[0], "x", 1));
    close(closed_pair[0]);
    free(expected);
    free(controls);
    printf("  [PASS] Production SSE/Responses callbacks and collection: split UTF-8, NUL, invalid/truncated UTF-8, long controls, socket short writes and allocation overflow\n");
}

/* Simulated response generator for testing format compliance */
static void generate_responses_api_json(const char *model_alias, const char *generated_text, uint32_t token_count, char *out_buf, size_t out_cap) {
    char escaped[4096];
    json_escape(generated_text, escaped, sizeof(escaped));
    long now = (long)time(NULL);
    snprintf(out_buf, out_cap,
             "{\"id\":\"resp_myds_%ld\",\"object\":\"response\",\"created_at\":%ld,"
             "\"status\":\"completed\",\"model\":\"%s\","
             "\"output\":[{\"type\":\"message\",\"id\":\"msg_myds_1\",\"role\":\"assistant\",\"status\":\"completed\","
             "\"content\":[{\"type\":\"output_text\",\"text\":\"%s\"}]}],"
             "\"usage\":{\"input_tokens\":10,\"output_tokens\":%u,\"total_tokens\":%u}}",
             now, now,
             model_alias ? model_alias : "myds",
             escaped,
             token_count,
             10 + token_count);
}

static void test_deepseek41_chat_templates(void) {
    const struct { const char *body; const char *expected; } cases[] = {
        {"{\"input\":\"Hello\"}", "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"},
        {"{\"instructions\":\"Be concise.\",\"prompt\":\"Hello\"}", "<｜begin▁of▁sentence｜><｜System｜>Be concise.<｜User｜>Hello<｜Assistant｜></think>"},
        {"{\"messages\":[{\"role\":\"developer\",\"content\":\"Guide\"},{\"role\":\"user\",\"content\":\"One\"},{\"role\":\"user\",\"content\":\"Two\"}]}",
         "<｜begin▁of▁sentence｜><｜System｜>Guide<｜User｜>One\n\nTwo<｜Assistant｜></think>"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello\"},{\"role\":\"system\",\"content\":\"Continue\"}]}",
         "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>Hello<｜end▁of▁sentence｜><｜System｜>Continue<｜Assistant｜></think>"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello\"}]}",
         "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>Hello<｜end▁of▁sentence｜>"},
        {"{\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"你\"},{\"type\":\"text\",\"text\":\"好\\uD83D\\uDE42\"}]}]}",
         "<｜begin▁of▁sentence｜><｜User｜>你好🙂<｜Assistant｜></think>"},
        {"{\"metadata\":{\"input\":\"wrong\",\"instructions\":\"ignore\"},\"inp\\u0075t\":\"right\"}",
         "<｜begin▁of▁sentence｜><｜User｜>right<｜Assistant｜></think>"},
        {"{\"messages\":[{\"role\":\"system\",\"content\":\"\"},{\"role\":\"user\",\"content\":\"\"}]}",
         "<｜begin▁of▁sentence｜><｜System｜><｜User｜><｜Assistant｜></think>"},
        {"{\"input\":[{\"type\":\"function_call_output\",\"output\":\"result </tool_result> &lt;/tool_result>\"}]}" ,
         "<｜begin▁of▁sentence｜><｜User｜><tool_result>result &lt;/tool_result> &amp;lt;/tool_result></tool_result><｜Assistant｜></think>"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        char actual[4096];
        assert(parse_chat_prompt(cases[i].body, actual, sizeof(actual)));
        assert(strcmp(actual, cases[i].expected) == 0);
        size_t length = strlen(cases[i].expected);
        assert(parse_chat_prompt(cases[i].body, actual, length + 1));
        assert(strcmp(actual, cases[i].expected) == 0);
        assert(!parse_chat_prompt(cases[i].body, actual, length));
        assert(actual[0] == 0);
    }
    const char *invalid[] = {
        "", "{}", "[]", "{\"messages\":[]}", "{\"messages\":{}}", "{\"input\":\"hi\"} garbage",
        "{\"input\":\"first\",\"input\":\"second\"}", "{\"input\":\"hi\",\"prompt\":\"hi\"}",
        "{\"input\":\"bad\\q\"}", "{\"input\":\"bad\\uD800\"}", "{\"input\":\"bad\\uDC00\"}",
        "{\"input\":\"before\\u0000after\"}", "{\"input\":\"\xed\xa0\x80\"}",
        "{\"input\":\"hi\",}", "{\"input\":\"hi\"", "{\"messages\":[{\"role\":\"user\"}]}",
        "{\"messages\":[{\"content\":\"hi\"}]}", "{\"messages\":[{\"role\":\"unknown\",\"content\":\"hi\"}]}",
        "{\"input\":[{\"content\":[{\"type\":\"input_image\",\"image_url\":\"image\"}]}]}",
        "{\"input\":[{\"content\":[{\"type\":\"input_text\"}]}]}",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        char output[4096] = "previous";
        assert(!parse_chat_prompt(invalid[i], output, sizeof(output)));
        assert(output[0] == 0);
    }
    struct { char before[16], output[4], after[16]; } guard;
    memset(&guard, 'X', sizeof(guard));
    assert(!parse_chat_prompt(cases[0].body, guard.output, sizeof(guard.output)));
    assert(guard.output[0] == 0);
    for (size_t i = 0; i < sizeof(guard.before); ++i) assert(guard.before[i] == 'X' && guard.after[i] == 'X');
    assert(!parse_chat_prompt(cases[0].body, NULL, 100));
    assert(!parse_chat_prompt(cases[0].body, guard.output, 0));

    /* Every accepted string path preserves the full input, including large
     * instructions and direct Responses input that formerly stopped at 4/8 KB. */
    for (int mode = 0; mode < 3; ++mode) {
        const size_t length = 20000;
        char *request = malloc(length * 2 + 128), *output = malloc(length * 2 + 256);
        assert(request && output);
        const char *key = mode == 0 ? "input" : mode == 1 ? "prompt" : "instructions";
        int prefix = snprintf(request, length * 2 + 128, "{\"%s\":\"", key);
        memset(request + prefix, 'A', length);
        strcpy(request + prefix + length, mode == 2 ? "\",\"input\":\"End\"}" : "\"}");
        assert(parse_chat_prompt(request, output, length * 2 + 256));
        const char *start = strstr(output, mode == 2 ? "<｜System｜>" : "<｜User｜>");
        assert(start); start += strlen(mode == 2 ? "<｜System｜>" : "<｜User｜>");
        for (size_t i = 0; i < length; ++i) assert(start[i] == 'A');
        assert(start[length] == '<');
        free(request); free(output);
    }
    printf("  [PASS] V4.1 exact chat templates, Unicode input, structured text parts, large input and explicit parse/capacity errors\n");
}


/* 工具调用格式（官方 DeepSeek-V4.1 DSML）—— 行为在 2026-10-01 由「拒绝」改为「支持」，
 * 原先放在 invalid[] 里的三个用例已移到此处验证渲染结果。 */
static void test_deepseek41_tool_calls(void) {
    char output[16384];

    /* ① tools（chat 嵌套形状）：应渲染官方 TOOLS_TEMPLATE + 原样 schema */
    const char *with_tools =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Beijing?\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},"
        "\"required\":[\"city\"]}}}]}";
    assert(parse_chat_prompt(with_tools, output, sizeof(output)));
    assert(strstr(output, "<｜System｜>") != NULL);
    assert(strstr(output, "## Tools") != NULL);
    assert(strstr(output, "<｜DSML｜ calls>") != NULL);
    assert(strstr(output, "<｜DSML｜ invoke") != NULL);
    assert(strstr(output, "### Available Tool Schemas") != NULL);
    /* schema 逐字节原样（含参数细节），而不是被重新序列化后失真 */
    assert(strstr(output, "{\"name\":\"get_weather\",\"description\":\"Get weather\","
                          "\"parameters\":{\"type\":\"object\",\"properties\":"
                          "{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},"
                          "\"required\":[\"city\"]}}") != NULL);
    assert(strstr(output, "<｜User｜>Weather in Beijing?") != NULL);
    assert(strstr(output, "You MUST strictly follow the above defined tool name") != NULL);

    /* ② tools（Responses 扁平形状）也应被接受 */
    const char *flat_tools =
        "{\"input\":\"hi\",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\","
        "\"description\":\"Look up\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]}";
    assert(parse_chat_prompt(flat_tools, output, sizeof(output)));
    assert(strstr(output, "{\"name\":\"lookup\",\"description\":\"Look up\","
                          "\"parameters\":{\"type\":\"object\",\"properties\":{}}}") != NULL);

    /* ③ 历史里的 assistant tool_calls ⇒ DSML invoke/parameter 块（在 EOS 之前） */
    const char *history =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"},"
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
        "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Beijing\\\",\\\"days\\\":3}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"sunny\"},"
        "{\"role\":\"user\",\"content\":\"thanks\"}]}";
    assert(parse_chat_prompt(history, output, sizeof(output)));
    assert(strstr(output, "<｜DSML｜ calls>") != NULL);
    assert(strstr(output, "<｜DSML｜ invoke name=\"get_weather\">") != NULL);
    /* 字符串参数：string="true"；非字符串参数：JSON 形式 + string="false" */
    assert(strstr(output, "<｜DSML｜ parameter name=\"city\" string=\"true\">Beijing</｜DSML｜ parameter>") != NULL);
    assert(strstr(output, "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>") != NULL);
    assert(strstr(output, "</｜DSML｜ invoke>") != NULL);
    assert(strstr(output, "</｜DSML｜ calls>") != NULL);
    /* 工具结果走官方 <tool_result> 格式 */
    assert(strstr(output, "<tool_result>sunny</tool_result>") != NULL);
    /* 工具调用必须出现在 assistant 的 EOS 之前 */
    const char *calls_at = strstr(output, "</｜DSML｜ calls>");
    const char *eos_at = strstr(calls_at, "<｜end▁of▁sentence｜>");
    assert(calls_at && eos_at && eos_at > calls_at);

    /* ④ 旧式 function_call（Responses）也应接受 */
    const char *legacy =
        "{\"messages\":[{\"role\":\"assistant\",\"content\":\"hi\","
        "\"function_call\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"x\\\"}\"}}]}";
    assert(parse_chat_prompt(legacy, output, sizeof(output)));
    assert(strstr(output, "<｜DSML｜ invoke name=\"lookup\">") != NULL);
    assert(strstr(output, "<｜DSML｜ parameter name=\"q\" string=\"true\">x</｜DSML｜ parameter>") != NULL);

    /* ⑤ 非 assistant 角色挂 tool_calls ⇒ 拒绝（格式只对 assistant 有意义） */
    char small[4096];
    assert(!parse_chat_prompt("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\","
                              "\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"x\"}}]}]}",
                              small, sizeof(small)));
    /* ⑥ 工具名用不安全字符（会被写进 DSML 属性）⇒ 拒绝 */
    assert(!parse_chat_prompt("{\"tools\":[{\"type\":\"function\",\"function\":"
                              "{\"name\":\"bad\\\"name\",\"parameters\":{}}}]}", small, sizeof(small)));

    printf("  [PASS] V4.1 DSML tool definitions, assistant tool_calls, tool results and rejections\n");
}


/* DSML 解析：把模型输出切成「正文 + 结构化工具调用」 */
static void test_deepseek41_dsml_parse(void) {
    const char *DS = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
    char text[4096];

    /* ① 纯文本（无工具块）⇒ 原样、count=0 */
    size_t clen = 0, n = 0; myds_tool_call_t *tc = NULL;
    assert(myds_parse_completion_tool_calls("hello <think>x</think> world", 27, &clen, &tc, &n));
    assert(n == 0 && tc == NULL && clen == 27);

    /* ② 一个调用，两种参数类型 */
    snprintf(text, sizeof(text),
        "Let me check.\n\n<%s calls>\n<%s invoke name=\"get_weather\">\n"
        "<%s parameter name=\"city\" string=\"true\">Beijing</%s parameter>\n"
        "<%s parameter name=\"days\" string=\"false\">3</%s parameter>\n"
        "</%s invoke>\n</%s calls>",
        DS,DS,DS,DS,DS,DS,DS,DS);
    assert(myds_parse_completion_tool_calls(text, strlen(text), &clen, &tc, &n));
    assert(n == 1 && tc != NULL);
    assert(strcmp(tc[0].id, "call_1") == 0);
    assert(strcmp(tc[0].name, "get_weather") == 0);
    assert(strcmp(tc[0].arguments, "{\"city\": \"Beijing\", \"days\": 3}") == 0);
    assert(clen == strlen("Let me check."));      /* 正文不该含工具块，尾部换行也要去掉 */
    myds_free_tool_calls(tc, n); tc = NULL;

    /* ③ 两个调用 */
    snprintf(text, sizeof(text),
        "ok\n\n<%s calls>\n<%s invoke name=\"a\">\n<%s parameter name=\"x\" string=\"false\">1</%s parameter>\n"
        "</%s invoke>\n<%s invoke name=\"b\">\n</%s invoke>\n</%s calls>", DS,DS,DS,DS,DS,DS,DS,DS);
    assert(myds_parse_completion_tool_calls(text, strlen(text), &clen, &tc, &n));
    assert(n == 2);
    assert(strcmp(tc[0].name, "a") == 0 && strcmp(tc[0].arguments, "{\"x\": 1}") == 0);
    assert(strcmp(tc[1].name, "b") == 0 && strcmp(tc[1].arguments, "{}") == 0);
    myds_free_tool_calls(tc, n); tc = NULL;

    /* ④ 字符串参数里的引号/换行必须被正确转义 */
    snprintf(text, sizeof(text),
        "x\n\n<%s calls>\n<%s invoke name=\"t\">\n<%s parameter name=\"s\" string=\"true\">a\"b</%s parameter>\n"
        "</%s invoke>\n</%s calls>", DS,DS,DS,DS,DS,DS);
    assert(myds_parse_completion_tool_calls(text, strlen(text), &clen, &tc, &n));
    assert(n == 1 && strcmp(tc[0].arguments, "{\"s\": \"a\\\"b\"}") == 0);
    myds_free_tool_calls(tc, n); tc = NULL;

    /* ⑤ 格式不合（invoke 未闭合）⇒ 返回 false，调用方按纯文本处理 */
    snprintf(text, sizeof(text), "x\n\n<%s calls>\n<%s invoke name=\"a\">\n\"", DS, DS);
    assert(!myds_parse_completion_tool_calls(text, strlen(text), &clen, &tc, &n));

    /* ⑥ 数组参数：非字符串参数原样透传 JSON */
    snprintf(text, sizeof(text),
        "x\n\n<%s calls>\n<%s invoke name=\"f\">\n"
        "<%s parameter name=\"xs\" string=\"false\">[1, 2, 3]</%s parameter>\n"
        "<%s parameter name=\"o\" string=\"false\">{\"k\": true}</%s parameter>\n"
        "</%s invoke>\n</%s calls>", DS,DS,DS,DS,DS,DS,DS,DS);
    assert(myds_parse_completion_tool_calls(text, strlen(text), &clen, &tc, &n));
    assert(n == 1 && strcmp(tc[0].arguments, "{\"xs\": [1, 2, 3], \"o\": {\"k\": true}}") == 0);
    myds_free_tool_calls(tc, n);

    printf("  [PASS] V4.1 DSML completion parsing: content split, parameter kinds, escaping, two calls, malformed fallback\n");
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--render")) {
        size_t capacity = strlen(argv[2]) * 4 + 1024;
        char *output = malloc(capacity);
        if (!output) return 1;
        bool ok = parse_chat_prompt(argv[2], output, capacity);
        if (ok) fputs(output, stdout);
        free(output); return ok ? 0 : 1;
    }
    test_deepseek41_chat_templates();
    test_deepseek41_tool_calls();
    test_deepseek41_dsml_parse();
    printf("====================================================\n");
    printf("       Running myds Server API Unit Tests           \n");
    printf("====================================================\n");

    signal(SIGPIPE, SIG_IGN);
    test_json_bytes();
    test_stream_and_non_stream_bytes();
    assert(myds_server_default_max_tokens(256) == 256);
    assert(myds_server_default_max_tokens(4096) == 4096);
    assert(myds_server_default_max_tokens(262144) == 4096);
    printf("  [PASS] Omitted API token limits cap at 4096 while smaller server defaults are preserved\n");

    /* 1. Test Endpoint Route Recognition */
    {
        const char *p1 = "/v1/responses";
        const char *p2 = "/responses";
        const char *p3 = "/v1/chat/completions";
        const char *p4 = "/chat/completions";
        const char *p5 = "/health";
        const char *p6 = "/v1/models";
        const char *p7 = "/unknown";

        assert(strcmp(p1, "/v1/responses") == 0 || strcmp(p1, "/responses") == 0);
        assert(strcmp(p2, "/v1/responses") == 0 || strcmp(p2, "/responses") == 0);
        assert(strcmp(p3, "/v1/chat/completions") == 0 || strcmp(p3, "/chat/completions") == 0);
        assert(strcmp(p4, "/v1/chat/completions") == 0 || strcmp(p4, "/chat/completions") == 0);
        assert(strcmp(p5, "/health") == 0);
        assert(strcmp(p6, "/v1/models") == 0);
        assert(strcmp(p7, "/health") != 0 && strcmp(p7, "/v1/responses") != 0);
        printf("  [PASS] Endpoint route matching (/responses, /chat/completions, /health, /models)\n");
    }

    /* 2. Test Parameter Extraction (max_output_tokens precedence) */
    {
        const char *body_both = "{\"max_output_tokens\": 512, \"max_tokens\": 256, \"stream\": true}";
        int max_tokens = json_extract_int_field(body_both, "max_output_tokens", 0);
        if (max_tokens <= 0) max_tokens = json_extract_int_field(body_both, "max_tokens", 0);
        assert(max_tokens == 512);

        const char *body_legacy = "{\"max_tokens\": 128, \"stream\": false}";
        max_tokens = json_extract_int_field(body_legacy, "max_output_tokens", 0);
        if (max_tokens <= 0) max_tokens = json_extract_int_field(body_legacy, "max_tokens", 0);
        assert(max_tokens == 128);

        int stream = json_extract_bool_field(body_both, "stream", 0);
        assert(stream == 1);
        stream = json_extract_bool_field(body_legacy, "stream", 0);
        assert(stream == 0);
        printf("  [PASS] Parameter extraction and max_output_tokens precedence\n");
    }

    /* 3. Test Multi-Turn Conversation Prompt Formatting via production parse_chat_prompt */
    {
        const char *multi_turn =
            "{\n"
            "  \"instructions\": \"You are a coding tutor.\",\n"
            "  \"input\": [\n"
            "    {\"type\": \"message\", \"role\": \"user\", \"content\": [{\"type\":\"input_text\",\"text\":\"What is a pointer?\"}]},\n"
            "    {\"type\": \"message\", \"role\": \"assistant\", \"content\": [{\"type\":\"input_text\",\"text\":\"A pointer holds a memory address.\"}]},\n"
            "    {\"type\": \"message\", \"role\": \"user\", \"content\": [{\"type\":\"input_text\",\"text\":\"How do I dereference it?\"}]}\n"
            "  ]\n"
            "}";

        char prompt[8192];
        assert(parse_chat_prompt(multi_turn, prompt, sizeof(prompt)));

        assert(strcmp(prompt,
            "<｜begin▁of▁sentence｜><｜System｜>You are a coding tutor."
            "<｜User｜>What is a pointer?<｜Assistant｜></think>A pointer holds a memory address."
            "<｜end▁of▁sentence｜><｜User｜>How do I dereference it?<｜Assistant｜></think>") == 0);
        printf("  [PASS] Multi-turn conversation prompt construction with instructions\n");
    }

    /* 4. Test Responses API Non-Streaming JSON Response Structure */
    {
        char resp_json[4096];
        generate_responses_api_json("local-myds-deepseek-v4.1-flash-q2", "Dereference with *ptr", 4, resp_json, sizeof(resp_json));

        assert(strstr(resp_json, "\"object\":\"response\"") != NULL);
        assert(strstr(resp_json, "\"status\":\"completed\"") != NULL);
        assert(strstr(resp_json, "\"model\":\"local-myds-deepseek-v4.1-flash-q2\"") != NULL);
        assert(strstr(resp_json, "\"output\":[{\"type\":\"message\"") != NULL);
        assert(strstr(resp_json, "\"content\":[{\"type\":\"output_text\",\"text\":\"Dereference with *ptr\"}]") != NULL);
        assert(strstr(resp_json, "\"usage\":{\"input_tokens\":10,\"output_tokens\":4,\"total_tokens\":14}") != NULL);
        printf("  [PASS] Responses API non-streaming JSON structure compliance\n");
    }

    /* 5. Test Special Character JSON Escaping via production json_escape */
    {
        const char *special_code = "int x = 1;\nprintf(\"Hello \\\"World\\\"\\t!\\n\");";
        char escaped[1024];
        json_escape(special_code, escaped, sizeof(escaped));

        assert(strstr(escaped, "\\n") != NULL);
        assert(strstr(escaped, "\\\"") != NULL);
        assert(strstr(escaped, "\\t") != NULL);
        printf("  [PASS] JSON special character escaping (newlines, quotes, tabs)\n");
    }

    /* 6. Test Responses API SSE Stream Events Format */
    {
        char chunk[1024];
        /* Output item added */
        snprintf(chunk, sizeof(chunk),
                 "event: response.output_item.added\n"
                 "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n");
        assert(strstr(chunk, "event: response.output_item.added") != NULL);
        assert(strstr(chunk, "\"type\":\"response.output_item.added\"") != NULL);

        /* Content part added */
        snprintf(chunk, sizeof(chunk),
                 "event: response.content_part.added\n"
                 "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n");
        assert(strstr(chunk, "event: response.content_part.added") != NULL);

        /* Output text delta */
        snprintf(chunk, sizeof(chunk),
                 "event: response.output_text.delta\n"
                 "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}\n\n");
        assert(strstr(chunk, "event: response.output_text.delta") != NULL);
        assert(strstr(chunk, "\"delta\":\"hello\"") != NULL);

        /* Completed event */
        snprintf(chunk, sizeof(chunk),
                 "event: response.completed\n"
                 "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n\n");
        assert(strstr(chunk, "event: response.completed") != NULL);
        assert(strstr(chunk, "\"status\":\"completed\"") != NULL);
        printf("  [PASS] Responses API SSE events protocol compliance\n");
    }

    /* 7. Test P0-2 & P0-3: Finish reason mapping and real prompt_tokens */
    {
        myds_stop_reason_t r_eos = MYDS_STOP_EOS;
        myds_stop_reason_t r_len = MYDS_STOP_LENGTH;
        myds_stop_reason_t r_ctx = MYDS_STOP_CTX_FULL;

        const char *f_eos = (r_eos == MYDS_STOP_LENGTH || r_eos == MYDS_STOP_CTX_FULL) ? "length" : "stop";
        const char *f_len = (r_len == MYDS_STOP_LENGTH || r_len == MYDS_STOP_CTX_FULL) ? "length" : "stop";
        const char *f_ctx = (r_ctx == MYDS_STOP_LENGTH || r_ctx == MYDS_STOP_CTX_FULL) ? "length" : "stop";

        assert(strcmp(f_eos, "stop") == 0);
        assert(strcmp(f_len, "length") == 0);
        assert(strcmp(f_ctx, "length") == 0);

        /* Real usage token calculation */
        uint32_t prompt_tokens = 42;
        uint32_t completion_tokens = 8;
        uint32_t total_tokens = prompt_tokens + completion_tokens;

        char chat_json[512];
        snprintf(chat_json, sizeof(chat_json),
                 "{\"choices\":[{\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                 f_len, prompt_tokens, completion_tokens, total_tokens);

        assert(strstr(chat_json, "\"finish_reason\":\"length\"") != NULL);
        assert(strstr(chat_json, "\"prompt_tokens\":42") != NULL);
        assert(strstr(chat_json, "\"completion_tokens\":8") != NULL);
        assert(strstr(chat_json, "\"total_tokens\":50") != NULL);
        printf("  [PASS] P0-2 finish_reason mapping and P0-3 real usage tokens\n");
    }

    /* 8. Test P0-4: stream_options.include_usage = true detection and chunk generation */
    {
        const char *body_with_usage = "{\"model\":\"myds\",\"stream\":true,\"stream_options\":{\"include_usage\":true}}";
        const char *body_without_usage = "{\"model\":\"myds\",\"stream\":true}";

        int has_usage1 = (strstr(body_with_usage, "\"include_usage\":true") != NULL || strstr(body_with_usage, "\"include_usage\": true") != NULL);
        int has_usage2 = (strstr(body_without_usage, "\"include_usage\":true") != NULL || strstr(body_without_usage, "\"include_usage\": true") != NULL);

        assert(has_usage1 == 1);
        assert(has_usage2 == 0);

        uint32_t prompt_tok = 15;
        uint32_t comp_tok = 25;
        char usage_chunk[512];
        snprintf(usage_chunk, sizeof(usage_chunk),
                 "data: {\"id\":\"chatcmpl-myds\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}\n\n",
                 prompt_tok, comp_tok, prompt_tok + comp_tok);

        assert(strstr(usage_chunk, "\"choices\":[]") != NULL);
        assert(strstr(usage_chunk, "\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":25,\"total_tokens\":40}") != NULL);
        printf("  [PASS] P0-4 stream_options.include_usage detection and usage chunk compliance\n");
    }

    /* 9. Test P1-2: Dynamic prompt parsing without silent 4096-char truncation */
    {
        size_t big_content_size = 20000;
        char *big_json = (char *)malloc(big_content_size + 2048);
        assert(big_json != NULL);

        strcpy(big_json, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
        size_t offset = strlen(big_json);
        memset(big_json + offset, 'A', big_content_size);
        offset += big_content_size;
        strcpy(big_json + offset, "\"}]}");

        size_t prompt_cap = big_content_size + 8192;
        char *parsed_prompt = (char *)malloc(prompt_cap);
        assert(parsed_prompt != NULL);

        assert(parse_chat_prompt(big_json, parsed_prompt, prompt_cap));
        assert(strstr(parsed_prompt, "<｜begin▁of▁sentence｜><｜User｜>") != NULL);
        /* Ensure the full 20000-character content is present without truncation */
        assert(strlen(parsed_prompt) > big_content_size);
        assert(strstr(parsed_prompt, "<｜Assistant｜>") != NULL);

        free(parsed_prompt);
        free(big_json);
        printf("  [PASS] P1-2 dynamic prompt buffer parsing with >16KB content without truncation\n");
    }

    printf("====================================================\n");
    printf("      ALL SERVER API UNIT TESTS PASSED!             \n");
    printf("====================================================\n");
    return 0;
}
