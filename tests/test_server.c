#include "myds_server.h"
#include "myds_server_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    printf("====================================================\n");
    printf("     Running myds Server Protocol Unit Tests        \n");
    printf("====================================================\n");

    /* 1. Test brace matching with nested JSON and braces inside quotes */
    {
        const char *nested = "{\"type\":\"message\",\"content\":[{\"type\":\"input_text\",\"text\":\"hello {world}\"}]}";
        const char *end = find_matching_brace(nested);
        assert(end != NULL);
        assert(*end == '}');
        assert(end == nested + strlen(nested) - 1);
        printf("  [PASS] Nested brace matching and string literal escape handling\n");
    }

    /* 2. Test UTF-8 escape unescaping */
    {
        const char *escaped_src = "\\u4f60\\u597d"; /* 你好 */
        char out[64] = {0};
        json_unescape_slice(escaped_src, strlen(escaped_src), out, sizeof(out));
        assert(strcmp(out, "你好") == 0);
        printf("  [PASS] Unicode escape to UTF-8 decoding (e.g. \\u4f60\\u597d -> 你好)\n");
    }

    /* 3. Test OpenAI Responses API structured input extraction */
    {
        const char *req_body =
            "{\n"
            "  \"model\": \"local-myds-deepseek-v4.1-flash-q2\",\n"
            "  \"instructions\": \"You are an expert systems programmer.\",\n"
            "  \"input\": [\n"
            "    {\n"
            "      \"type\": \"message\",\n"
            "      \"role\": \"user\",\n"
            "      \"content\": [\n"
            "        {\"type\": \"input_text\", \"text\": \"Write a C queue.\"}\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            "  \"stream\": true,\n"
            "  \"max_output_tokens\": 1024\n"
            "}";

        char prompt[4096];
        assert(parse_chat_prompt(req_body, prompt, sizeof(prompt)));
        assert(strstr(prompt, "<｜System｜>You are an expert systems programmer.") != NULL);
        assert(strstr(prompt, "<｜User｜>Write a C queue.") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
        printf("  [PASS] OpenAI Responses API input array with instructions\n");
    }

    /* 4. Test OpenAI Responses API plain string input */
    {
        const char *req_body =
            "{\"instructions\": \"Be concise.\", \"input\": \"What is 1+1?\"}";

        char prompt[4096];
        assert(parse_chat_prompt(req_body, prompt, sizeof(prompt)));
        assert(strstr(prompt, "<｜System｜>Be concise.") != NULL);
        assert(strstr(prompt, "<｜User｜>What is 1+1?") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
        printf("  [PASS] OpenAI Responses API string input\n");
    }

    /* 5. Test OpenAI Chat Completions messages compatibility */
    {
        const char *req_body =
            "{\"messages\": [{\"role\":\"system\",\"content\":\"sys\"},{\"role\":\"user\",\"content\":\"usr\"}]}";

        char prompt[4096];
        assert(parse_chat_prompt(req_body, prompt, sizeof(prompt)));
        assert(strstr(prompt, "<｜System｜>sys") != NULL);
        assert(strstr(prompt, "<｜User｜>usr") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
        printf("  [PASS] OpenAI Chat Completions messages compatibility\n");
    }

    /* 6. Test HTTP Content-Length header parsing (T3-3 fix verification) */
    {
        const char *headers =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: localhost:8181\r\n"
            "User-Agent: curl/8.7.1\r\n"
            "Accept: */*\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 4218\r\n"
            "\r\n";

        int cl = extract_http_header_int(headers, "Content-Length");
        assert(cl == 4218);

        /* Case insensitivity */
        int cl_lower = extract_http_header_int(headers, "content-length");
        assert(cl_lower == 4218);

        /* Missing header returns 0 */
        int cl_missing = extract_http_header_int(headers, "X-Non-Existent");
        assert(cl_missing == 0);

        printf("  [PASS] Standard HTTP Content-Length header parsing (T3-3)\n");
    }

    /* 7. Test JSON scalar field extractions */
    {
        const char *json = "{\"temp\": 0.75, \"top_p\": 0.95, \"max_tokens\": 128, \"stream\": true}";
        assert(json_extract_int_field(json, "max_tokens", 0) == 128);
        assert(json_extract_bool_field(json, "stream", 0) == 1);
        float temp = json_extract_float_field(json, "temp", 0.0f);
        assert(temp >= 0.74f && temp <= 0.76f);
        printf("  [PASS] JSON scalar field extractions\n");
    }

    /* 8. min_p / repetition_penalty 的请求级解析（2026-10-24 新增入口）
     *
     * 这两个字段在 myds_config_t 里一直存在且真的参与采样，但此前没有任何
     * 入口能赋值（main.c 无 CLI、请求体不解析）⇒ 恒为 0.0 / 1.0 = 永久禁用。
     * 下面钉死「缺省探测值」与「显式值能读出来」两件事。 */
    {
        const char *no_field = "{\"temperature\": 0.7}";
        assert(json_extract_float_field(no_field, "min_p", -1.0f) < 0.0f);
        assert(json_extract_float_field(no_field, "repetition_penalty", -1.0f) < 0.0f);

        const char *with_field = "{\"min_p\": 0.05, \"repetition_penalty\": 1.15}";
        float mip = json_extract_float_field(with_field, "min_p", -1.0f);
        float rep = json_extract_float_field(with_field, "repetition_penalty", -1.0f);
        assert(mip > 0.04f && mip < 0.06f);
        assert(rep > 1.14f && rep < 1.16f);

        /* 0.0 必须是「显式关闭」而不是「没提供」——两者语义不同：
         * 没提供 = 沿用启动配置；提供 0.0 = 本次关掉。 */
        const char *zero = "{\"min_p\": 0.0}";
        assert(json_extract_float_field(zero, "min_p", -1.0f) == 0.0f);
        printf("  [PASS] min_p / repetition_penalty request-level parsing\n");
    }

    /* 8b. presence_penalty 的请求级解析（2026-10-24 新增）
     * 与 min_p / repetition_penalty 同一套「缺省探测」语义：
     * 没提供 ⇒ 沿用启动配置；显式 0.0 ⇒ 本次关闭。 */
    {
        const char *absent = "{\"temperature\": 0.7}";
        assert(json_extract_float_field(absent, "presence_penalty", -1.0f) < 0.0f);

        const char *pos = "{\"presence_penalty\": 0.8}";
        float v = json_extract_float_field(pos, "presence_penalty", -1.0f);
        assert(v > 0.79f && v < 0.81f);

        /* 负值合法（OpenAI 允许 [-2,2]，负值鼓励重复） */
        const char *neg = "{\"presence_penalty\": -1.5}";
        v = json_extract_float_field(neg, "presence_penalty", -1.0f);
        assert(v > -1.51f && v < -1.49f);

        /* 显式 0.0 必须与「未提供」区分开 */
        const char *zero = "{\"presence_penalty\": 0.0}";
        assert(json_extract_float_field(zero, "presence_penalty", -1.0f) == 0.0f);
        printf("  [PASS] presence_penalty request-level parsing\n");
    }

    /* 9. 思考模式：reasoning 对象形式（2026-10-24）
     *
     * mycode / OpenAI Responses 生态用 "reasoning": {"effort": "high"} 表达强度。
     * 此前 myds 只认顶层 reasoning_effort 字符串 ⇒ 对象形式被静默忽略、
     * thinking 回落到启动默认值。实测判别：effort=high 却没有任何 reasoning 输出。
     * 下面把四种形态钉死。 */
    {
        /* 对象形式：high/max/low/medium ⇒ 开 */
        assert(extract_request_thinking_mode("{\"reasoning\":{\"effort\":\"high\"}}", false) == true);
        assert(extract_request_thinking_mode("{\"reasoning\":{\"effort\":\"max\"}}", false) == true);
        assert(extract_request_thinking_mode("{\"reasoning\": {\"effort\" : \"low\"}}", false) == true);
        /* 对象形式：none/off ⇒ 关（即使默认是开） */
        assert(extract_request_thinking_mode("{\"reasoning\":{\"effort\":\"none\"}}", true) == false);
        /* enabled 折算 */
        assert(extract_request_thinking_mode("{\"reasoning\":{\"enabled\":true}}", false) == true);
        assert(extract_request_thinking_mode("{\"reasoning\":{\"enabled\":false}}", true) == false);
        /* 没提任何思考字段 ⇒ 用默认值 */
        assert(extract_request_thinking_mode("{\"input\":\"hi\"}", false) == false);
        assert(extract_request_thinking_mode("{\"input\":\"hi\"}", true) == true);
        /* ★ "reasoning" 作为【值】出现（input 里的 reasoning 项）不得误判 */
        assert(extract_request_thinking_mode(
            "{\"input\":[{\"type\":\"reasoning\",\"content\":[]}],\"reasoning\":{\"effort\":\"high\"}}",
            false) == true);
        assert(extract_request_thinking_mode(
            "{\"input\":[{\"type\":\"reasoning\"}]}", false) == false);
        /* 顶层字符串形式仍然有效（不能被对象解析抢走） */
        assert(extract_request_thinking_mode("{\"reasoning_effort\":\"high\"}", false) == true);
        assert(extract_request_thinking_mode("{\"reasoning_effort\":\"none\"}", true) == false);

        /* 官方映射表要求覆盖的全部 effort 取值（DeepSeek thinking_mode 文档）：
         *   minimal low medium high xhigh max ultra  ⇒ 开
         *   none/off/disabled                        ⇒ 关
         * 少一个就会被静默当成「没提」，回落到默认值。 */
        const char *on_vals[] = {"minimal","low","medium","high","xhigh","max","ultra"};
        for (unsigned k = 0; k < sizeof(on_vals)/sizeof(on_vals[0]); ++k) {
            char b[128];
            snprintf(b, sizeof(b), "{\"reasoning\":{\"effort\":\"%s\"}}", on_vals[k]);
            if (extract_request_thinking_mode(b, false) != true)
                printf("  ✗ reasoning.effort=%s 应判为开\n", on_vals[k]);
            assert(extract_request_thinking_mode(b, false) == true);
            char b2[128];
            snprintf(b2, sizeof(b2), "{\"reasoning_effort\":\"%s\"}", on_vals[k]);
            assert(extract_request_thinking_mode(b2, false) == true);
        }
        printf("  [PASS] thinking mode: reasoning object form\n");
    }

    /* 10. 思考强度 → budget 映射（官方 thinking_mode 文档的映射表）
     *   minimal→low(50)  low→low(50)  medium→62  high→high(75)
     *   xhigh→high(75)   max→max(100)  ultra→max(100)
     * 少映射一个值，思考强度就会被静默降级成默认档。 */
    {
        struct { const char *v; int want; } cases[] = {
            {"minimal", 50}, {"low",  50}, {"medium", 62}, {"high", 75},
            {"xhigh",   75}, {"max", 100}, {"ultra", 100},
        };
        for (unsigned k = 0; k < sizeof(cases)/sizeof(cases[0]); ++k) {
            char b[96], b2[96];
            snprintf(b, sizeof(b), "{\"reasoning\":{\"effort\":\"%s\"}}", cases[k].v);
            const int got = extract_request_reasoning_budget(b, 75);
            if (got != cases[k].want) printf("  ✗ reasoning.effort=%s → %d（应 %d）\n", cases[k].v, got, cases[k].want);
            assert(got == cases[k].want);
            snprintf(b2, sizeof(b2), "{\"reasoning_effort\":\"%s\"}", cases[k].v);
            assert(extract_request_reasoning_budget(b2, 75) == cases[k].want);
        }
        /* 整数形式仍然有效（1-100） */
        assert(extract_request_reasoning_budget("{\"reasoning_effort\":42}", 75) == 42);
        printf("  [PASS] thinking effort → budget mapping\n");
    }

    printf("====================================================\n");
    printf("  ALL SERVER PROTOCOL TESTS PASSED CLEANLY!\n");
    printf("====================================================\n");
    return 0;
}
