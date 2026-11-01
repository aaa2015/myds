/* HTTP transport regression: runs the production handler with a deterministic
 * byte generator. This checks transport/serialization, not model correctness.
 * Compile and run through tests/test_http_bytes.py; no model/GPU is required. */
#include "../src/myds_server.c"
#include <assert.h>
static myds_config_t config = {.max_new_tokens = 262144, .max_seq_len = 8192};
static int force_failure;
static uint32_t expected_limit;
static unsigned generation_calls;
myds_config_t myds_context_get_config(const myds_context_t *ctx) { (void)ctx; return config; }
void myds_context_set_config(myds_context_t *ctx, const myds_config_t *cfg) { (void)ctx; config = *cfg; }
void myds_log(myds_log_level_t level, const char *fmt, ...) { (void)level; (void)fmt; }
void myds_context_add_image_embeddings(myds_context_t *ctx, uint32_t start_pos,
                                       uint32_t count, const float *data, uint32_t dim,
                                       uint32_t image_token_id) {
    (void)ctx; (void)start_pos; (void)count; (void)data; (void)dim; (void)image_token_id;
}
void myds_context_clear_image_embeddings(myds_context_t *ctx) { (void)ctx; }
void myds_last_prefill_stats(const myds_context_t *ctx, myds_prefill_stats_t *out) {
    (void)ctx;
    if (out) memset(out, 0, sizeof(*out));
}
void myds_context_reset_prefill_stats(myds_context_t *ctx) { (void)ctx; }
void myds_set_continue_from_cache(myds_context_t *ctx, bool enable) { (void)ctx; (void)enable; }
uint32_t myds_context_get_pcache_len(const myds_context_t *ctx) { (void)ctx; return 100; }
myds_steer_t *myds_context_find_steer(const myds_context_t *ctx, const char *name) { (void)ctx; (void)name; return NULL; }
void myds_context_set_liveness_probe(myds_context_t *ctx, myds_liveness_probe_t probe, void *user_data) { (void)ctx; (void)probe; (void)user_data; }
void myds_vjob_set_gpu_lock(void (*lock_fn)(void), void (*unlock_fn)(void)) { (void)lock_fn; (void)unlock_fn; }
const char *myds_vjob_placeholder(void) { return "<|image_pad|>"; }
uint32_t myds_vjob_image_token_id(void) { return 0; }
int myds_vjob_sink_add(myds_vjob_sink_t *sink, const uint8_t *encoded, size_t len,
                       char *error, size_t error_cap) {
    (void)sink; (void)encoded; (void)len; (void)error; (void)error_cap;
    return 1;
}
void myds_vjob_sink_free(myds_vjob_sink_t *sink) {
    if (sink) memset(sink, 0, sizeof(*sink));
}
int64_t myds_vjob_token_offset_at(myds_context_t *ctx, const char *prompt_base,
                                  const char *needle) {
    (void)ctx; (void)prompt_base; (void)needle;
    return 0;
}
int myds_generate_bytes_ex(myds_context_t *ctx, const char *prompt, myds_stream_bytes_callback_t cb, void *data, myds_stop_reason_t *reason, uint32_t *n_prompt, volatile int *cancel) {
    (void)ctx; (void)prompt;
    generation_calls++;
    assert(config.max_new_tokens == expected_limit);
    *n_prompt = 3;
    if (force_failure) { *reason = MYDS_STOP_ERROR; return 0; }
    const unsigned char raw[] = {'A', 0, 0xe4, 0xb8, 0xad, 0xf0, 0x9f, 0x98, 0x80, 0xff, 0xe4, 0xb8};
    for (size_t i = 0; i < sizeof(raw); i++) cb((const char *)raw + i, 1, (int32_t)i, data);
    char controls[8192];
    memset(controls, 1, sizeof(controls));
    cb(controls, sizeof(controls), 20, data);
    cb("\xf0\x9f", 2, 21, data);
    assert(!*cancel);
    *reason = MYDS_STOP_LENGTH;
    return 1;
}
static void *handle(void *arg) { handle_client(*(int *)arg, NULL, "test-\"alias", "local", 1234); return NULL; }
/* ── DSML 工具块隔离（2026-11-01）──────────────────────────────────
 * 只测分类器与「正文边界」这一层：它决定客户端会不会收到原始 XML。
 * 端到端（真模型 + 截断）见 tests/dsml_truncation_live.py。 */
static void dsml_isolation_cases(void) {
    const char *OPEN = "<" MYDS_DSML_S " calls>";
    const char *CLOSE = "</" MYDS_DSML_S " calls>";
    char buf[2048];
    size_t clen;

    /* 1) 没有工具块 ⇒ 全是正文 */
    snprintf(buf, sizeof(buf), "plain answer with no tools");
    clen = 0;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_NONE);
    assert(clen == strlen(buf));

    /* 2) 完整块 ⇒ COMPLETE，正文只到开标签之前（尾部空白削掉） */
    snprintf(buf, sizeof(buf),
             "here you go\n\n%s\n<" MYDS_DSML_S " invoke name=\"f\"><" MYDS_DSML_S
             " parameter name=\"a\" string=\"true\">v</" MYDS_DSML_S " parameter></"
             MYDS_DSML_S " invoke>\n%s\n", OPEN, CLOSE);
    clen = 0;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_COMPLETE);
    assert(clen == strlen("here you go"));

    /* 3) 被截断（有开无闭）⇒ TRUNCATED，正文同样只到开标签之前。
     *    这正是 max_tokens 打断工具调用时出现的形状。 */
    snprintf(buf, sizeof(buf), "answer\n\n%s\n<" MYDS_DSML_S " invoke name=\"f\"><"
             MYDS_DSML_S " parameter name=\"a", OPEN);
    clen = 0;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_TRUNCATED);
    assert(clen == strlen("answer"));

    /* 4) 有开有闭但内部语法坏 ⇒ MALFORMED，正文依然只到开标签之前 */
    snprintf(buf, sizeof(buf), "answer\n%s garbage %s", OPEN, CLOSE);
    clen = 0;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_MALFORMED);
    assert(clen == strlen("answer"));

    /* 5) 开标签在一开头、正文为空 ⇒ content_len 必须是 0（不能退化成整段） */
    snprintf(buf, sizeof(buf), "%s junk", OPEN);
    clen = 999;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_TRUNCATED);
    assert(clen == 0);

    /* 6) 空输入 */
    clen = 999;
    assert(myds_dsml_classify("", 0, &clen) == MYDS_DSML_NONE);
    assert(clen == 0);

    /* 7) 另一套方言：<function_calls> 在【结尾】⇒ 认定为工具块，正文只到开标签之前 */
    snprintf(buf, sizeof(buf),
             "answer\n\n<function_calls>\n<invoke name=\"f\">\n"
             "<parameter name=\"a\">v</parameter>\n</invoke>\n</function_calls>\n");
    clen = 0;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_COMPLETE);
    assert(clen == strlen("answer"));

    /* 8) 同一方言出现在【正文中间】⇒ 不认，整段保持正文。
     *    这条是安全边界：讲解/代码示例里引用这种标记不能被吞掉。 */
    snprintf(buf, sizeof(buf),
             "The markup looks like <function_calls>\n<invoke name=\"f\">\n</invoke>\n"
             "</function_calls>\nbut remember it is only an example.");
    clen = 0;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_NONE);
    assert(clen == strlen(buf));

    /* 9) <function_calls> 被截断（有开无闭）⇒ TRUNCATED */
    snprintf(buf, sizeof(buf),
             "answer\n\n<function_calls>\n<invoke name=\"f\">\n<parameter name=\"a\"");
    clen = 0;
    assert(myds_dsml_classify(buf, strlen(buf), &clen) == MYDS_DSML_TRUNCATED);
    assert(clen == strlen("answer"));

    puts("PASS DSML classification isolates truncated and malformed tool blocks from content");
    puts("PASS <function_calls> dialect accepted only as a trailing block");
}

int main(int argc, char **argv) {
    assert(argc == 2); /* Existing directory for the eight raw HTTP responses. */
    signal(SIGPIPE, SIG_IGN);
    for (unsigned response = 0; response < 2; response++) {
        for (unsigned stream = 0; stream < 2; stream++) {
            for (unsigned fail = 0; fail < 2; fail++) {
                int sockets[2];
                assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
                force_failure = fail;
                expected_limit = fail ? 7000 : 8192;
                char body[256], request[512];
                snprintf(body, sizeof(body), "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":%s%s}", stream ? "true" : "false", fail ? ",\"max_tokens\":7000" : "");
                int n = snprintf(request, sizeof(request), "POST /v1/%s HTTP/1.1\r\nContent-Length: %zu\r\n\r\n%s", response ? "responses" : "chat/completions", strlen(body), body);
                assert(myds_server_send_all(sockets[1], request, (size_t)n));
                pthread_t worker;
                assert(pthread_create(&worker, NULL, handle, &sockets[0]) == 0);
                char path[PATH_MAX];
                int path_length = snprintf(path, sizeof(path), "%s/myds-http-%u-%u-%u.txt",
                                           argv[1], response, stream, fail);
                assert(path_length > 0 && (size_t)path_length < sizeof(path));
                FILE *out = fopen(path, "wb");
                assert(out);
                char received[8192];
                ssize_t count;
                while ((count = recv(sockets[1], received, sizeof(received), 0)) > 0) assert(fwrite(received, 1, (size_t)count, out) == (size_t)count);
                assert(count == 0);
                fclose(out);
                close(sockets[1]);
                assert(pthread_join(worker, NULL) == 0);
                assert(config.max_new_tokens == 262144);
            }
        }
    }
    for (unsigned malformed = 0; malformed < 2; malformed++) {
        int sockets[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        const char nul_body[] = "{\"input\":\"hello\"}\0ignored suffix";
        const char *body = malformed == 0 ? nul_body : "{\"input\":\"hello\"}";
        size_t length = malformed == 0 ? sizeof(nul_body) - 1 : strlen(body);
        char header[256];
        int n = snprintf(header, sizeof(header),
            "POST /v1/responses HTTP/1.1\r\nContent-Length: %zu\r\n\r\n",
            length + (malformed == 1 ? 3 : 0));
        assert(myds_server_send_all(sockets[1], header, (size_t)n));
        assert(myds_server_send_all(sockets[1], body, length));
        assert(shutdown(sockets[1], SHUT_WR) == 0);
        unsigned before = generation_calls;
        handle_client(sockets[0], NULL, "test", "local", 1234);
        char response[1024] = {0};
        assert(recv(sockets[1], response, sizeof(response) - 1, 0) > 0);
        assert(strncmp(response, "HTTP/1.1 400 Bad Request", 24) == 0);
        assert(generation_calls == before);
        close(sockets[1]);
    }
    puts("PASS raw NUL and incomplete Content-Length rejected before generation");
    dsml_isolation_cases();
    return 0;
}
