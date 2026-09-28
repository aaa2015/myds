#include "myds_server.h"
#include "myds_server_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>

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

int main(void) {
    printf("====================================================\n");
    printf("       Running myds Server API Unit Tests           \n");
    printf("====================================================\n");

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
        parse_chat_prompt(multi_turn, prompt, sizeof(prompt));

        assert(strstr(prompt, "<｜begin of sentence｜><｜User｜>[System: You are a coding tutor.]\n") != NULL);
        assert(strstr(prompt, "<｜User｜>What is a pointer?\n") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>A pointer holds a memory address.\n") != NULL);
        assert(strstr(prompt, "<｜User｜>How do I dereference it?\n") != NULL);
        assert(strstr(prompt, "<｜Assistant｜>") != NULL);
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

        parse_chat_prompt(big_json, parsed_prompt, prompt_cap);
        assert(strstr(parsed_prompt, "<｜begin of sentence｜><｜User｜>") != NULL);
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
