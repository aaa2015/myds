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
int main(int argc, char **argv) {
    assert(argc == 2); /* Existing directory for the eight raw HTTP responses. */
    signal(SIGPIPE, SIG_IGN);
    for (unsigned response = 0; response < 2; response++) {
        for (unsigned stream = 0; stream < 2; stream++) {
            for (unsigned fail = 0; fail < 2; fail++) {
                int sockets[2];
                assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
                force_failure = fail;
                expected_limit = fail ? 7000 : 4096;
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
    return 0;
}
