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

    printf("====================================================\n");
    printf("  ALL SERVER PROTOCOL TESTS PASSED CLEANLY!\n");
    printf("====================================================\n");
    return 0;
}
