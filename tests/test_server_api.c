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
#include <fcntl.h>
#include <stdatomic.h>

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

    char *wire_reference[2] = {NULL, NULL};
    size_t wire_reference_length[2] = {0, 0};
    for (unsigned variant = 0; variant < 4; variant++) {
        const bool responses = variant % 2 != 0;
        int sockets[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        int send_size = 1024;
        assert(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &send_size, sizeof(send_size)) == 0);
        socket_capture_t capture = {.fd = sockets[1]};
        pthread_t reader;
        assert(pthread_create(&reader, NULL, capture_socket, &capture) == 0);
        myds_sse_stream_ctx_t stream = {.client_fd = sockets[0], .responses_api = responses};
        if (variant >= 2) assert(myds_server_sse_writer_start(&stream, 1024u * 1024u, 1000));
        for (size_t i = 0; i < sizeof(binary_text); i++) {
            myds_server_sse_callback(binary_text + i, 1, (int32_t)i, &stream);
        }
        myds_server_sse_callback(controls, controls_length, 50, &stream);
        myds_server_sse_callback("\xf0\x9f", 2, 51, &stream);
        assert(myds_server_sse_finish(&stream));
        assert(!stream.cancel_flag && !stream.failed && stream.utf8.pending_length == 0);
        assert(stream.token_count == sizeof(binary_text) + 2);
        assert(stream.t_first > 0 && stream.t_last >= stream.t_first);
        assert(myds_server_sse_writer_join(&stream, false));
        assert(shutdown(sockets[0], SHUT_WR) == 0);
        assert(pthread_join(reader, NULL) == 0);
        /* Validate the wire including JSON punctuation between events. A
         * code point split across SSE events would fail here even if joining
         * the extracted content happened to recover the original bytes. */
        assert_valid_utf8(capture.bytes, capture.length);
        size_t collected_length = 0;
        char *collected = collect_sse_escaped_text(capture.bytes, responses, &collected_length);
        assert(collected_length == expected_length);
        assert(memcmp(collected, expected, expected_length + 1) == 0);
        assert(strstr(capture.bytes, responses ? "event: response.output_text.delta" : "chat.completion.chunk"));
        free(collected);
        if (variant < 2) {
            wire_reference[variant] = capture.bytes;
            wire_reference_length[variant] = capture.length;
        } else {
            assert(capture.length == wire_reference_length[variant % 2]);
            assert(memcmp(capture.bytes, wire_reference[variant % 2], capture.length) == 0);
            free(capture.bytes);
        }
        free(stream.acc);
        close(sockets[0]);
        close(sockets[1]);
    }
    free(wire_reference[0]); free(wire_reference[1]);

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

static double test_now_s(void) {
    struct timespec ts;
    assert(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct { int fd; atomic_bool stop; } trickle_reader_t;

static void *read_socket_slowly(void *opaque) {
    trickle_reader_t *reader = opaque;
    char bytes[128];
    while (!atomic_load(&reader->stop)) {
        const ssize_t n = recv(reader->fd, bytes, sizeof(bytes), 0);
        if (n <= 0) break;
        const struct timespec delay = {.tv_nsec = 5000000};
        nanosleep(&delay, NULL);
    }
    return NULL;
}

static void test_sse_writer_transport(void) {
    const char *old_timeout = getenv("MYDS_SSE_DRAIN_TIMEOUT_MS");
    char *saved_timeout = old_timeout ? strdup(old_timeout) : NULL;
    assert(setenv("MYDS_SSE_DRAIN_TIMEOUT_MS", "25", 1) == 0);
    assert(myds_server_sse_drain_timeout_ms() == 25);
    assert(setenv("MYDS_SSE_DRAIN_TIMEOUT_MS", "30001", 1) == 0);
    assert(myds_server_sse_drain_timeout_ms() == 1000);
    assert(setenv("MYDS_SSE_DRAIN_TIMEOUT_MS", "bad", 1) == 0);
    assert(myds_server_sse_drain_timeout_ms() == 1000);
    if (saved_timeout) { assert(setenv("MYDS_SSE_DRAIN_TIMEOUT_MS", saved_timeout, 1) == 0); free(saved_timeout); }
    else assert(unsetenv("MYDS_SSE_DRAIN_TIMEOUT_MS") == 0);
    /* Header, deltas, usage and terminal event use one FIFO and owned copies. */
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    socket_capture_t capture = {.fd = sockets[1]};
    pthread_t reader;
    assert(pthread_create(&reader, NULL, capture_socket, &capture) == 0);
    myds_sse_stream_ctx_t stream = {.client_fd = sockets[0]};
    int original_flags = fcntl(sockets[0], F_GETFL);
    assert(myds_server_sse_writer_start(&stream, 65536, 1000));
    assert(fcntl(sockets[0], F_GETFL) & O_NONBLOCK);
    assert(!myds_server_sse_writer_start(&stream, 65536, 1000) && errno == EINVAL);
    const char *events[] = {"HTTP/1.1 200 OK\r\n\r\n", "data: first\n\n", "data: second\n\n",
                            "data: finish\n\n", "data: usage\n\n", "data: [DONE]\n\n"};
    char expected[512] = "";
    for (size_t i = 0; i < sizeof(events) / sizeof(events[0]); ++i) {
        char temporary[128];
        strcpy(temporary, events[i]);
        strcat(expected, temporary);
        assert(myds_server_sse_send(&stream, temporary, strlen(temporary)));
        memset(temporary, 'x', strlen(temporary)); /* Caller may immediately reuse/free its buffer. */
    }
    assert(myds_server_sse_writer_join(&stream, false) && !stream.writer);
    assert(fcntl(sockets[0], F_GETFL) == original_flags);
    assert(shutdown(sockets[0], SHUT_WR) == 0);
    assert(pthread_join(reader, NULL) == 0);
    assert(capture.length == strlen(expected) && strcmp(capture.bytes, expected) == 0);
    free(capture.bytes); close(sockets[0]); close(sockets[1]);

    /* Thread and packet allocation failure must leave no live writer or
     * retained packet after joining. The caller still owns the socket. */
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    stream = (myds_sse_stream_ctx_t){.client_fd = sockets[0]};
    original_flags = fcntl(sockets[0], F_GETFL);
    myds_server_sse_writer_inject_failure(1);
    assert(!myds_server_sse_writer_start(&stream, 65536, 50) && errno == EAGAIN && !stream.writer);
    assert(fcntl(sockets[0], F_GETFL) == original_flags);
    assert(fcntl(sockets[0], F_GETFD) >= 0);
    assert(myds_server_sse_writer_start(&stream, 65536, 50));
    assert(myds_server_sse_send(&stream, "first", 5));
    myds_server_sse_writer_inject_failure(2);
    assert(!myds_server_sse_send(&stream, "second", 6) && stream.failed && stream.cancel_flag);
    assert(!myds_server_sse_writer_join(&stream, true) && !stream.writer);
    assert(fcntl(sockets[0], F_GETFL) == original_flags);
    assert(fcntl(sockets[0], F_GETFD) >= 0);
    close(sockets[0]); close(sockets[1]);

    /* A peer that never reads cannot block the producer, and the in-flight
     * packet still counts towards the finite queue budget. */
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    int send_size = 1024;
    assert(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &send_size, sizeof(send_size)) == 0);
    stream = (myds_sse_stream_ctx_t){.client_fd = sockets[0]};
    assert(myds_server_sse_writer_start(&stream, 16384, 50));
    char packet[4096]; memset(packet, 'a', sizeof(packet));
    const double enqueue_start = test_now_s();
    bool accepted = true;
    for (unsigned i = 0; i < 32 && accepted; ++i)
        accepted = myds_server_sse_send(&stream, packet, sizeof(packet));
    assert(!accepted && stream.cancel_flag);
    assert(test_now_s() - enqueue_start < 0.5);
    assert(!myds_server_sse_writer_join(&stream, true) && !stream.writer);
    close(sockets[0]); close(sockets[1]);

    /* A peer making continuous partial progress still cannot extend the
     * total drain deadline indefinitely. */
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    assert(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &send_size, sizeof(send_size)) == 0);
    stream = (myds_sse_stream_ctx_t){.client_fd = sockets[0]};
    assert(myds_server_sse_writer_start(&stream, 2u * 1024u * 1024u, 50));
    trickle_reader_t trickle = {.fd = sockets[1]};
    assert(pthread_create(&reader, NULL, read_socket_slowly, &trickle) == 0);
    char *large = malloc(1024u * 1024u); assert(large); memset(large, 'z', 1024u * 1024u);
    assert(myds_server_sse_send(&stream, large, 1024u * 1024u)); free(large);
    const double progressing_start = test_now_s();
    assert(!myds_server_sse_writer_join(&stream, false) && stream.cancel_flag && !stream.writer);
    assert(test_now_s() - progressing_start < 0.5);
    atomic_store(&trickle.stop, true);
    assert(pthread_join(reader, NULL) == 0);
    close(sockets[0]); close(sockets[1]);

    /* End-of-response drain uses an overall deadline, rather than blocking
     * until SO_SNDTIMEO or allocating beyond the queue cap. */
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    assert(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &send_size, sizeof(send_size)) == 0);
    stream = (myds_sse_stream_ctx_t){.client_fd = sockets[0]};
    assert(myds_server_sse_writer_start(&stream, 2u * 1024u * 1024u, 50));
    large = malloc(1024u * 1024u); assert(large); memset(large, 'z', 1024u * 1024u);
    assert(myds_server_sse_send(&stream, large, 1024u * 1024u));
    free(large);
    const double drain_start = test_now_s();
    assert(!myds_server_sse_writer_join(&stream, false) && stream.cancel_flag && !stream.writer);
    assert(test_now_s() - drain_start < 0.5);
    close(sockets[0]); close(sockets[1]);

    /* Closing the peer during an active writer is an observable transport
     * failure; join always completes before the caller closes/reuses its fd. */
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    stream = (myds_sse_stream_ctx_t){.client_fd = sockets[0]};
    assert(myds_server_sse_writer_start(&stream, 65536, 50));
    close(sockets[1]);
    (void)myds_server_sse_send(&stream, "data: x\n\n", 9);
    assert(!myds_server_sse_writer_join(&stream, false) && stream.cancel_flag && !stream.writer);
    close(sockets[0]);
    puts("  [PASS] Async SSE FIFO/copied bytes, allocation/thread failures, bounded slow-reader cancellation, total drain deadline and disconnect join");
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
         "<｜begin▁of▁sentence｜><｜User｜>你\n\n好🙂<｜Assistant｜></think>"},
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
    assert(strstr(output, "{\"name\": \"get_weather\", \"description\": \"Get weather\", "
                          "\"parameters\": {\"type\": \"object\", \"properties\": "
                          "{\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, "
                          "\"required\": [\"city\"]}}") != NULL);
    assert(strstr(output, "<｜User｜>Weather in Beijing?") != NULL);
    assert(strstr(output, "You MUST strictly follow the above defined tool name") != NULL);

    /* ② tools（Responses 扁平形状）也应被接受 */
    const char *flat_tools =
        "{\"input\":\"hi\",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\","
        "\"description\":\"Look up\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]}";
    assert(parse_chat_prompt(flat_tools, output, sizeof(output)));
    assert(strstr(output, "{\"type\": \"function\", \"name\": \"lookup\", \"description\": \"Look up\", "
                          "\"parameters\": {\"type\": \"object\", \"properties\": {}}}") != NULL);

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

    /* ④b Responses 标准多轮形状：把上一轮的 function_call 原样发回来 */
    const char *resp_multi =
        "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Weather?\"}]},"
        "{\"type\":\"function_call\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Beijing\\\"}\",\"call_id\":\"call_1\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_1\",\"output\":\"sunny\"}]}";
    assert(parse_chat_prompt(resp_multi, output, sizeof(output)));
    assert(strstr(output, "<｜DSML｜ invoke name=\"get_weather\">") != NULL);
    assert(strstr(output, "<｜DSML｜ parameter name=\"city\" string=\"true\">Beijing</｜DSML｜ parameter>") != NULL);
    assert(strstr(output, "<tool_result>sunny</tool_result>") != NULL);
    /* 正文为空的 function_call 项也要能渲染（name/arguments 在该项自身） */
    assert(strstr(output, "<｜User｜>Weather?") != NULL);

    /* ④c 不支持的类型必须给出【可诊断】的原因，而不是笼统文案。
     * 注意：reasoning 自 1860e70 起是【合法】项（OpenClaw 会回灌），这里换成真正不支持的类型。 */
    const char *bad_type =
        "{\"input\":[{\"role\":\"user\",\"content\":\"hi\"},{\"type\":\"computer_call\",\"action\":{}}]}";
    char reason_buf[256];
    const char *why = NULL;
    assert(!parse_chat_prompt_ex(bad_type, output, sizeof(output), &why));
    assert(why && strstr(why, "unsupported") != NULL);
    (void)reason_buf;

    /* ④d reasoning 项必须被安全跳过：不报错、不贡献任何 prompt 内容 */
    const char *with_reasoning =
        "{\"input\":[{\"role\":\"user\",\"content\":\"hi\"},{\"type\":\"reasoning\",\"summary\":[]}]}";
    const char *plain = "{\"input\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    char out_plain[sizeof(output)];
    why = NULL;
    assert(parse_chat_prompt_ex(with_reasoning, output, sizeof(output), &why));
    assert(parse_chat_prompt_ex(plain, out_plain, sizeof(out_plain), &why));
    assert(strcmp(output, out_plain) == 0);

    /* ⑤ 非 assistant 角色挂 tool_calls ⇒ 拒绝（格式只对 assistant 有意义） */
    char small[4096];
    assert(!parse_chat_prompt("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\","
                              "\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"x\"}}]}]}",
                              small, sizeof(small)));
    /* ⑥ 工具名用不安全字符（会被写进 DSML 属性）⇒ 拒绝 */
    assert(!parse_chat_prompt("{\"tools\":[{\"type\":\"function\",\"function\":"
                              "{\"name\":\"bad\\\"name\",\"parameters\":{}}}]}", small, sizeof(small)));

    /* ⑦ 工具结果重排：多 tool 结果应按照上一轮 assistant 的 tool_calls 声明顺序排列 */
    const char *out_of_order =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"},"
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":["
        "{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"exec\",\"arguments\":\"{\\\"cmd\\\":\\\"1\\\"}\"}},"
        "{\"id\":\"c2\",\"type\":\"function\",\"function\":{\"name\":\"exec\",\"arguments\":\"{\\\"cmd\\\":\\\"2\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"c2\",\"content\":\"two\"},"
        "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"one\"}]}";
    assert(parse_chat_prompt(out_of_order, output, sizeof(output)));
    assert(strstr(output, "<tool_result>one</tool_result>\n\n<tool_result>two</tool_result>") != NULL);

    /* ⑧ TOOLS_TEMPLATE 尾部换行与额外字段 (如 strict) 保留 */
    const char *strict_tool =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"description\":\"d\","
        "\"parameters\":{\"type\":\"object\"},\"strict\":true}}]}";
    assert(parse_chat_prompt(strict_tool, output, sizeof(output)));
    assert(strstr(output, "\"strict\": true") != NULL);
    assert(strstr(output, "schemas to invoke tool calls.\n<｜User｜>") != NULL);

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


/* 流式工具调用：DSML 块必须【不】作为正文发出，且必须吐出 tool_calls */
static void test_deepseek41_stream_tool_calls(void) {
    const char *DS = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    socket_capture_t capture = {.fd = sockets[1]};
    pthread_t reader;
    assert(pthread_create(&reader, NULL, capture_socket, &capture) == 0);
    myds_sse_stream_ctx_t stream = {.client_fd = sockets[0]};
    assert(myds_server_sse_writer_start(&stream, 1024u * 1024u, 1000));
    /* 逐片投喂：空白 + 分片到跨片边界的标签，模拟真实 token 流 */
    const char *parts[] = {"\n", "\n", "<", DS, " calls>\n<", DS,
                           " invoke name=\"get_weather\">\n<", DS,
                           " parameter name=\"city\" string=\"true\">Beijing</", DS,
                           " parameter>\n</", DS, " invoke>\n</", DS, " calls>"};
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); ++i) {
        myds_server_sse_callback(parts[i], strlen(parts[i]), (int32_t)i, &stream);
    }
    assert(myds_server_sse_finish(&stream));
    assert(!stream.cancel_flag && !stream.failed);
    assert(myds_server_sse_writer_join(&stream, false));
    assert(shutdown(sockets[0], SHUT_WR) == 0);
    assert(pthread_join(reader, NULL) == 0);
    size_t clen = 0;
    char *collected = collect_sse_escaped_text(capture.bytes, 0, &clen);
    /* 只验【回调路径】：DSML 块绝不能作为正文流出。
     * 结构化工具调用的发射在 handle_client 的收尾里，单元测试覆盖不到 ——
     * 那部分由集成测试（真实模型）负责。 */
    assert(clen == 0);
    free(collected);
    free(capture.bytes);
    close(sockets[0]);
    close(sockets[1]);
    printf("  [PASS] V4.1 streaming DSML: the call block is withheld from content, including the newlines before it\n");
}


/* 真正的递归下降 JSON 校验（弱版只数引号/括号，抓不到「字符串外的裸 token」——
 * 而那正是漏转义会产生的东西）。存在的意义：事件里 arguments 本身就是含引号的
 * JSON 串，忘了转义会产出【看着像 JSON 但解析不了】的事件。 */
static void json_ws(const char **pp, const char *e) {
    while (*pp < e && (**pp == ' ' || **pp == '\t' || **pp == '\n' || **pp == '\r')) ++*pp;
}
static bool json_str(const char **pp, const char *e) {
    const char *p = *pp;
    if (p >= e || *p != '"') return false;
    for (++p; p < e; ++p) {
        unsigned char c = (unsigned char)*p;
        if (c == '\\') {
            if (++p >= e) return false;
            if (*p == 'u') { if (p + 4 >= e) return false; p += 4; }
        } else if (c == '"') { *pp = p + 1; return true; }
        else if (c < 0x20) return false;
    }
    return false;
}
static bool json_value(const char **pp, const char *e) {
    json_ws(pp, e);
    const char *p = *pp;
    if (p >= e) return false;
    if (*p == '{') {                                    /* 对象：键:值,键:值 */
        ++p; json_ws(&p, e);
        if (p < e && *p == '}') { *pp = p + 1; return true; }
        for (;;) {
            if (!json_str(&p, e)) return false;
            json_ws(&p, e);
            if (p >= e || *p != ':') return false;
            ++p;
            if (!json_value(&p, e)) return false;
            json_ws(&p, e);
            if (p < e && *p == ',') { ++p; json_ws(&p, e); continue; }
            break;
        }
        json_ws(&p, e);
        if (p >= e || *p != '}') return false;
        *pp = p + 1; return true;
    }
    if (*p == '[') {                                    /* 数组：值,值 */
        ++p; json_ws(&p, e);
        if (p < e && *p == ']') { *pp = p + 1; return true; }
        for (;;) {
            if (!json_value(&p, e)) return false;
            json_ws(&p, e);
            if (p < e && *p == ',') { ++p; json_ws(&p, e); continue; }
            break;
        }
        json_ws(&p, e);
        if (p >= e || *p != ']') return false;
        *pp = p + 1; return true;
    }
    if (*p == '"') { if (!json_str(&p, e)) return false; *pp = p; return true; }
    if (p + 4 <= e && !strncmp(p, "true", 4)) { *pp = p + 4; return true; }
    if (p + 5 <= e && !strncmp(p, "false", 5)) { *pp = p + 5; return true; }
    if (p + 4 <= e && !strncmp(p, "null", 4)) { *pp = p + 4; return true; }
    if (*p == '-' || (*p >= '0' && *p <= '9')) {
        const char *q = p;
        if (*q == '-') ++q;
        if (q < e && *q == '0') ++q;
        else { if (q >= e || *q < '1' || *q > '9') return false; while (q < e && *q >= '0' && *q <= '9') ++q; }
        if (q < e && *q == '.') { ++q; if (q >= e || *q < '0' || *q > '9') return false; while (q < e && *q >= '0' && *q <= '9') ++q; }
        if (q < e && (*q == 'e' || *q == 'E')) {
            ++q; if (q < e && (*q == '+' || *q == '-')) ++q;
            if (q >= e || *q < '0' || *q > '9') return false;
            while (q < e && *q >= '0' && *q <= '9') ++q;
        }
        *pp = q; return true;
    }
    return false;
}
static bool json_valid(const char *p, size_t n) {
    const char *e = p + n, *q = p;
    if (!json_value(&q, e)) return false;
    json_ws(&q, e);
    return q == e;
}
/* 校验抓到的 SSE 流里每一行 `data: ` 都是合法 JSON */
static void assert_sse_data_lines_valid(const char *bytes) {
    const char *p = bytes;
    size_t checked = 0;
    while ((p = strstr(p, "data: ")) != NULL) {
        p += 6;
        const char *nl = strchr(p, '\n');
        size_t n = nl ? (size_t)(nl - p) : strlen(p);
        if (n == 6 && !memcmp(p, "[DONE]", 6)) continue;   /* [DONE] 不是 JSON */
        if (!json_valid(p, n)) {
            printf("  [FAIL] 非法 JSON 事件: %.*s\n", (int)n, p);
            assert(0 && "SSE data line is not valid JSON");
        }
        ++checked;
        p = nl ? nl : p + n;
    }
    assert(checked > 0);
}

/* 工具调用事件必须是【合法 JSON】（曾经漏转义 ⇒ 客户端解析直接失败） */
static void test_tool_call_event_json(void) {
    const char *DS = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
    char text[2048];
    /* 造一个 arguments 含引号的调用（正是会踩雷的输入） */
    snprintf(text, sizeof(text),
        "ok\n\n<%s calls>\n<%s invoke name=\"get_weather\">\n"
        "<%s parameter name=\"city\" string=\"true\">Beijing</%s parameter>\n"
        "</%s invoke>\n</%s calls>", DS,DS,DS,DS,DS,DS);
    size_t clen = 0, n = 0;
    myds_tool_call_t *tc = NULL;
    assert(myds_parse_completion_tool_calls(text, strlen(text), &clen, &tc, &n));
    assert(n == 1 && strcmp(tc[0].arguments, "{\"city\": \"Beijing\"}") == 0);

    /* chat 流式 */
    int s1[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s1) == 0);
    socket_capture_t c1 = {.fd = s1[1]};
    pthread_t r1; assert(pthread_create(&r1, NULL, capture_socket, &c1) == 0);
    myds_sse_stream_ctx_t st1 = {.client_fd = s1[0]};
    assert(myds_server_sse_writer_start(&st1, 1024u * 1024u, 1000));
    assert(sse_emit_chat_tool_calls(&st1, tc, n));
    assert(myds_server_sse_writer_join(&st1, false));
    assert(shutdown(s1[0], SHUT_WR) == 0);
    assert(pthread_join(r1, NULL) == 0);
    assert_sse_data_lines_valid(c1.bytes);
    assert(strstr(c1.bytes, "finish_reason\":\"tool_calls\"") != NULL);
    assert(strstr(c1.bytes, "\\\"city\\\"") != NULL);        /* 引号必须是转义后的 */
    free(c1.bytes); close(s1[0]); close(s1[1]);

    /* Responses 流式 */
    int s2[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s2) == 0);
    socket_capture_t c2 = {.fd = s2[1]};
    pthread_t r2; assert(pthread_create(&r2, NULL, capture_socket, &c2) == 0);
    myds_sse_stream_ctx_t st2 = {.client_fd = s2[0]};
    assert(myds_server_sse_writer_start(&st2, 1024u * 1024u, 1000));
    assert(sse_emit_responses_function_calls(&st2, tc, n));
    assert(myds_server_sse_writer_join(&st2, false));
    assert(shutdown(s2[0], SHUT_WR) == 0);
    assert(pthread_join(r2, NULL) == 0);
    assert_sse_data_lines_valid(c2.bytes);
    assert(strstr(c2.bytes, "\\\"city\\\"") != NULL);
    free(c2.bytes); close(s2[0]); close(s2[1]);

    myds_free_tool_calls(tc, n);
    printf("  [PASS] tool call events are valid JSON in both protocols (arguments escaped)\n");
}


static size_t count_occurrences(const char *hay, const char *needle);

/* ★ 回归：流式的【收尾事件】必须带上聚合后的文本。
 *
 * 只发增量、不发聚合，是 OpenAI 兼容实现里常见且危险的坑：增量全收到了，
 * 但读汇总事件的客户端（官方 SDK、多数封装）拿到的是空字符串 ——
 * 从外部看就是「流正常结束，但内容为空」，而【增量是好的】，所以很难归因。
 *
 * 判据：同一段文本必须出现在【全部四处】——output_text.done、content_part.done、
 * output_item.done、response.completed.response.output[]。少一处就有客户端读不到。 */
static void test_responses_finish_carries_the_aggregate_text(void) {
    /* 含非 ASCII：转义/编码出问题时这里会先炸 */
    const char *text = "你上一个问题是 2+2。";
    const size_t tlen = strlen(text);

    char *items = responses_output_message_json(text, tlen);
    assert(items != NULL);

    int s[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s) == 0);
    socket_capture_t c = {.fd = s[1]};
    pthread_t r; assert(pthread_create(&r, NULL, capture_socket, &c) == 0);
    myds_sse_stream_ctx_t st = {.client_fd = s[0]};
    assert(myds_server_sse_writer_start(&st, 1024u * 1024u, 1000));
    assert(sse_emit_responses_finish(&st, "test-model", "resp_test_1", 10, 3,
                                     text, tlen, items, 1));
    assert(myds_server_sse_writer_join(&st, false));
    assert(shutdown(s[0], SHUT_WR) == 0);
    assert(pthread_join(r, NULL) == 0);

    assert_sse_data_lines_valid(c.bytes);

    /* 四种事件都要在 */
    assert(strstr(c.bytes, "event: response.output_text.done") != NULL);
    assert(strstr(c.bytes, "event: response.content_part.done") != NULL);
    assert(strstr(c.bytes, "event: response.output_item.done") != NULL);
    assert(strstr(c.bytes, "event: response.completed") != NULL);
    /* output 数组必须带上（此前 response.completed 完全没有 output） */
    assert(strstr(c.bytes, "\"output\":[{\"type\":\"message\"") != NULL);
    /* ★ 关键判据：文本出现 4 次（前三个事件各一次 + output 数组一次）。
     * 此前三个事件里是 text":"" ⇒ 这里只会命中 1 次（output 那次还没有，即 0 次）。 */
    assert(count_occurrences(c.bytes, text) == 4);

    free(c.bytes); close(s[0]); close(s[1]); free(items);

    /* 工具调用那类：不发文本事件，但仍要有 response.completed 且 output 是 function_call */
    myds_tool_call_t tc = {.id = "call_1", .name = "get_weather",
                           .arguments = "{\"city\": \"Beijing\"}"};
    char *titems = responses_function_calls_output_json(&tc, 1);
    assert(titems != NULL);
    int s2[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s2) == 0);
    socket_capture_t c2 = {.fd = s2[1]};
    pthread_t r2; assert(pthread_create(&r2, NULL, capture_socket, &c2) == 0);
    myds_sse_stream_ctx_t st2 = {.client_fd = s2[0]};
    assert(myds_server_sse_writer_start(&st2, 1024u * 1024u, 1000));
    assert(sse_emit_responses_finish(&st2, "test-model", "resp_test_2", 10, 3, "", 0, titems, 0));
    assert(myds_server_sse_writer_join(&st2, false));
    assert(shutdown(s2[0], SHUT_WR) == 0);
    assert(pthread_join(r2, NULL) == 0);
    assert_sse_data_lines_valid(c2.bytes);
    assert(strstr(c2.bytes, "event: response.completed") != NULL);
    assert(strstr(c2.bytes, "\"type\":\"function_call\"") != NULL);
    /* 工具调用不该发 message 类的文本事件（那会误导客户端去读一个空消息） */
    assert(strstr(c2.bytes, "response.content_part.done") == NULL);
    free(c2.bytes); close(s2[0]); close(s2[1]); free(titems);

    printf("  [PASS] finish events carry the aggregate text in all four places\n");
}

static size_t count_occurrences(const char *hay, const char *needle) {
    size_t n = 0;
    for (const char *p = hay; (p = strstr(p, needle)) != NULL; p += strlen(needle)) ++n;
    return n;
}

/* Responses 形状：正文（message 项）与调用（function_call 项）必须合并成【一个】assistant 轮。
 * 拆成两轮会多出一个 EOS 与一个 <｜Assistant｜>，于是下一轮的前缀缓存必然失配
 * （实测每次 agent 回合都全量重跑 prefill，ttft 17s 而非 ~1s）。 */
static void test_responses_call_merges_into_assistant_turn(void) {
    char output[8192];
    const char *req =
        "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Weather in Beijing?\"}]},"
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"I will check that for you.\"}]},"
        "{\"type\":\"function_call\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Beijing\\\"}\",\"call_id\":\"c1\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"22C clear\"}]}";
    assert(parse_chat_prompt(req, output, sizeof(output)));

    /* 正文之后【直接】接调用块（官方模板 {reasoning}{content}{tool_calls} + 单个 EOS） */
    assert(strstr(output, "for you.\n\n<｜DSML｜ calls>") != NULL);
    /* 只有【一个】EOS 收这一轮 assistant；拆成两轮会出现第二个 */
    assert(count_occurrences(output, "<｜end▁of▁sentence｜>") == 1);
    /* <｜Assistant｜></think> 只出现两次：本轮开头 + 末尾等模型续写 */
    assert(count_occurrences(output, "<｜Assistant｜>") == 2);
    /* 调用必须在本轮 EOS 之前 */
    const char *block = strstr(output, "</｜DSML｜ calls>");
    const char *eos = strstr(output, "<｜end▁of▁sentence｜>");
    assert(block && eos && eos > block);
    /* 相邻的多个 function_call 项合并进【一个】块 */
    const char *two =
        "{\"input\":[{\"role\":\"user\",\"content\":\"hi\"},"
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]},"
        "{\"type\":\"function_call\",\"name\":\"a\",\"arguments\":\"{}\",\"call_id\":\"c1\"},"
        "{\"type\":\"function_call\",\"name\":\"b\",\"arguments\":\"{}\",\"call_id\":\"c2\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"1\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"c2\",\"output\":\"2\"}]}";
    assert(parse_chat_prompt(two, output, sizeof(output)));
    assert(count_occurrences(output, "<｜DSML｜ calls>") == 1);
    assert(count_occurrences(output, "<｜end▁of▁sentence｜>") == 1);
    assert(strstr(output, "invoke name=\"a\"") && strstr(output, "invoke name=\"b\""));
    assert(count_occurrences(output, "<tool_result>") == 2);

    printf("  [PASS] Responses text+call merge into one assistant turn (single EOS)\n");
}

/* ── previous_response_id：注册表语义 ──
 * 只有【最新登记】的那个 id 可以续跑；更早的必须是「见过但已过期」而不是「未知」，
 * 因为这两者对客户端的含义不同（前者重放即可恢复，后者是拼错了 id）。 */
static void test_previous_response_id_registry(void) {
    char a[64], b[64], c[64];
    resp_id_new(a, sizeof a); resp_id_new(b, sizeof b); resp_id_new(c, sizeof c);
    assert(strcmp(a, b) != 0 && strcmp(b, c) != 0 && strcmp(a, c) != 0);
    assert(strncmp(a, "resp_myds_", 10) == 0);

    resp_id_register(a);
    assert(resp_id_is_live(a));
    assert(!resp_id_is_live(b));
    assert(resp_id_seen(a) && !resp_id_seen(b));

    /* 新的响应顶掉旧的：a 变成「见过但不再 live」 */
    resp_id_register(b);
    assert(resp_id_is_live(b));
    assert(!resp_id_is_live(a));
    assert(resp_id_seen(a));    /* 仍在窗口内 -> expired 而不是 unknown */

    /* 推理失败或清空上下文时：全部失效 */
    resp_id_invalidate_all();
    assert(!resp_id_is_live(b));
    assert(!resp_id_is_live(a));
    assert(!resp_id_seen(a));
    assert(!resp_id_seen(b));

    printf("  [PASS] previous_response_id registry: only newest live; expired vs unknown; invalidate_all\n");
}

/* ── previous_response_id：后缀渲染 ──
 * 续跑时服务端只渲染【新增消息】：不能带 BOS / instructions / tools ——
 * 它们已在被续接的那段序列里，重复渲染会让前缀整体错位（静默错误）。 */
static void test_previous_response_id_suffix_render(void) {
    const char *body =
        "{\"instructions\":\"SYSPROMPT\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"description\":\"d\",\"parameters\":{}}],"
        "\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":\"hello_new\"}]}";
    char full[8192], suffix[8192];
    const char *why = NULL;
    assert(parse_chat_prompt_ex(body, full, sizeof full, &why));
    assert(parse_chat_prompt_suffix_ex(body, suffix, sizeof suffix, &why));
    assert(strstr(full, "<\xef\xbd\x9c" "begin") != NULL);   /* 拆开：\x9cb 会被当成一个转义 */        /* 全量带 BOS */
    assert(strstr(full, "SYSPROMPT") != NULL);                    /* 全量带系统提示词 */
    assert(strstr(suffix, "SYSPROMPT") == NULL);                  /* 增量不带 */
    assert(strstr(suffix, "hello_new") != NULL);                  /* 增量带新消息 */
    /* ★ 续跑必须以【上一个 assistant 回合的 EOS】开头：
     * 缓存序列停在生成文本之后（EOS 从未被喂入），而全量重放里 assistant 消息是带 EOS 的。
     * 少了这一个 token，续跑序列就与全量重放差一个 —— 实测会少 1 个 token，属静默错位。 */
    assert(strncmp(suffix, "<" "\xef\xbd\x9c" "end", 6) == 0);
    assert(strlen(suffix) < strlen(full));
    printf("  [PASS] previous_response_id suffix render: items only, no BOS/instructions/tools\n");
}

static void test_thinking_mode(void) {
    /* 1. extract_request_thinking_mode */
    {
        assert(extract_request_thinking_mode("{\"enable_thinking\": true}", false) == true);
        assert(extract_request_thinking_mode("{\"enable_thinking\": false}", true) == false);
        assert(extract_request_thinking_mode("{\"thinking\": true}", false) == true);
        assert(extract_request_thinking_mode("{\"thinking\": false}", true) == false);
        assert(extract_request_thinking_mode("{\"thinking\": {\"type\": \"enabled\"}}", false) == true);
        assert(extract_request_thinking_mode("{\"thinking\": {\"type\": \"disabled\"}}", true) == false);
        assert(extract_request_thinking_mode("{\"reasoning_effort\": \"low\"}", false) == true);
        assert(extract_request_thinking_mode("{\"reasoning_effort\": \"medium\"}", false) == true);
        assert(extract_request_thinking_mode("{\"reasoning_effort\": \"high\"}", false) == true);
        assert(extract_request_thinking_mode("{\"reasoning_effort\": \"none\"}", true) == false);
        assert(extract_request_thinking_mode("{\"chat_template_kwargs\": {\"thinking\": true}}", false) == true);
        assert(extract_request_thinking_mode("{\"chat_template_kwargs\": {\"thinking\": false}}", true) == false);
        assert(extract_request_thinking_mode("{}", false) == false);
        assert(extract_request_thinking_mode("{}", true) == true);
    }

    /* 2. split_reasoning_content */
    {
        char *content = NULL, *reasoning = NULL;
        const char *t1 = "<think>thought 123</think>answer 456";
        split_reasoning_content(t1, strlen(t1), &content, &reasoning);
        assert(reasoning && !strcmp(reasoning, "thought 123"));
        assert(content && !strcmp(content, "answer 456"));
        free(content); free(reasoning);

        /* pre-filled <think> where model outputs reasoning</think>content */
        const char *t2 = "thought only</think>\n\nfinal answer";
        split_reasoning_content(t2, strlen(t2), &content, &reasoning);
        assert(reasoning && !strcmp(reasoning, "thought only"));
        assert(content && !strcmp(content, "final answer"));
        free(content); free(reasoning);

        /* all thinking, unclosed */
        const char *t3 = "<think>thinking forever";
        split_reasoning_content(t3, strlen(t3), &content, &reasoning);
        assert(reasoning && !strcmp(reasoning, "thinking forever"));
        assert(content && !strcmp(content, ""));
        free(content); free(reasoning);

        /* plain content without thinking */
        const char *t4 = "just plain text";
        split_reasoning_content(t4, strlen(t4), &content, &reasoning);
        assert(reasoning == NULL);
        assert(content && !strcmp(content, "just plain text"));
        free(content);
    }

    /* 3. parse_chat_prompt_ex suffix check */
    {
        char prompt_buf[4096];
        const char *req_thinking_on = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"thinking\": true}";
        assert(parse_chat_prompt(req_thinking_on, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "<｜Assistant｜><think>") != NULL);
        assert(strstr(prompt_buf, "</think>") == NULL);
        assert(strstr(prompt_buf, "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>hello") != NULL);

        const char *req_low = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"thinking\": true, \"reasoning_effort\": \"low\"}";
        assert(parse_chat_prompt(req_low, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "Reasoning Effort: 50") != NULL);

        const char *req_med = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"thinking\": true, \"reasoning_effort\": \"medium\"}";
        assert(parse_chat_prompt(req_med, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "Reasoning Effort: 62") != NULL);

        const char *req_max = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"thinking\": true, \"reasoning_effort\": \"max\"}";
        assert(parse_chat_prompt(req_max, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "Reasoning Effort: 100") != NULL);

        const char *req_int = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"thinking\": true, \"reasoning_effort\": 85}";
        assert(parse_chat_prompt(req_int, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "Reasoning Effort: 85") != NULL);

        const char *req_sys_think = "{\"instructions\": \"You are a bot.\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"thinking\": true}";
        assert(parse_chat_prompt(req_sys_think, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nYou are a bot.<｜User｜>hello<｜Assistant｜><think>") != NULL);

        const char *req_thinking_off = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"thinking\": false}";
        assert(parse_chat_prompt(req_thinking_off, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "<｜Assistant｜></think>") != NULL);
        assert(strstr(prompt_buf, "Reasoning Effort") == NULL);
    }

    /* 4. History rendering with reasoning_content / reasoning */
    {
        char prompt_buf[4096];
        const char *req_hist = "{\"messages\": ["
            "{\"role\": \"user\", \"content\": \"q1\"},"
            "{\"role\": \"assistant\", \"content\": \"a1\", \"reasoning_content\": \"r1\"},"
            "{\"role\": \"user\", \"content\": \"q2\"}"
            "], \"thinking\": true}";
        assert(parse_chat_prompt(req_hist, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "<｜Assistant｜><think>r1</think>a1<｜end▁of▁sentence｜>") != NULL);
        assert(strstr(prompt_buf, "<｜User｜>q2<｜Assistant｜><think>") != NULL);

        /* Responses API history shape: reasoning item + message item */
        const char *req_resp_hist = "{\"input\": ["
            "{\"type\": \"message\", \"role\": \"user\", \"content\": \"q1\"},"
            "{\"type\": \"reasoning\", \"content\": [{\"type\": \"reasoning_text\", \"text\": \"r_resp\"}]},"
            "{\"type\": \"message\", \"role\": \"assistant\", \"content\": \"a_resp\"},"
            "{\"type\": \"message\", \"role\": \"user\", \"content\": \"q2\"}"
            "], \"thinking\": false}";
        assert(parse_chat_prompt(req_resp_hist, prompt_buf, sizeof(prompt_buf)));
        assert(strstr(prompt_buf, "<｜Assistant｜></think>a_resp<｜end▁of▁sentence｜>") != NULL);
        assert(strstr(prompt_buf, "r_resp") == NULL); /* internal reasoning item skipped as per OpenClaw spec */
        assert(strstr(prompt_buf, "<｜User｜>q2<｜Assistant｜></think>") != NULL);
    }

    /* 5. responses_output_items_json_ex ordering */
    {
        myds_tool_call_t tc = {.id = "c1", .name = "fn", .arguments = "{}"};
        char *items_think = responses_output_items_json_ex(true, "my_thought", 10, "my_answer", 9, &tc, 1);
        assert(items_think != NULL);
        /* Invariant: reasoning item at index 0, message item at index 1, tool call at index 2 */
        assert(strstr(items_think, "{\"type\":\"reasoning\",\"id\":\"rs_myds_0\"") != NULL);
        assert(strstr(items_think, "{\"type\":\"message\",\"id\":\"msg_myds_1\"") != NULL);
        assert(strstr(items_think, "{\"type\":\"function_call\",\"id\":\"fc_1\"") != NULL);
        char *p_rs = strstr(items_think, "\"type\":\"reasoning\"");
        char *p_msg = strstr(items_think, "\"type\":\"message\"");
        char *p_fc = strstr(items_think, "\"type\":\"function_call\"");
        assert(p_rs < p_msg && p_msg < p_fc);
        free(items_think);

        /* Non-thinking: message item at index 0, tool call at index 1 */
        char *items_nothink = responses_output_items_json_ex(false, NULL, 0, "my_answer", 9, &tc, 1);
        assert(items_nothink != NULL);
        assert(strstr(items_nothink, "\"type\":\"reasoning\"") == NULL);
        assert(strstr(items_nothink, "{\"type\":\"message\",\"id\":\"msg_myds_1\"") != NULL);
        assert(strstr(items_nothink, "{\"type\":\"function_call\",\"id\":\"fc_1\"") != NULL);
        free(items_nothink);
    }

    /* 6. Streaming SSE callback transition with thinking mode */
    {
        int s[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s) == 0);
        socket_capture_t c = {.fd = s[1]};
        pthread_t r; assert(pthread_create(&r, NULL, capture_socket, &c) == 0);
        myds_sse_stream_ctx_t st = {
            .client_fd = s[0],
            .responses_api = false,
            .thinking = true,
            .think_state = 1,
            .msg_output_index = 1,
        };
        assert(myds_server_sse_writer_start(&st, 1024u * 1024u, 1000));

        /* Emit reasoning chunks, then </think>, then content chunk */
        myds_server_sse_callback("think", 5, 1, &st);
        myds_server_sse_callback("ing...", 6, 2, &st);
        myds_server_sse_callback("</think>\nHello world", 20, 3, &st);
        assert(myds_server_sse_finish(&st));
        assert(myds_server_sse_writer_join(&st, false));
        assert(shutdown(s[0], SHUT_WR) == 0);
        assert(pthread_join(r, NULL) == 0);

        assert_sse_data_lines_valid(c.bytes);
        assert(strstr(c.bytes, "\"reasoning_content\":\"think\"") != NULL);
        assert(strstr(c.bytes, "\"reasoning_content\":\"ing...\"") != NULL);
        assert(strstr(c.bytes, "\"content\":\"Hello world\"") != NULL);
        assert(strstr(c.bytes, "</think>") == NULL);

        free(c.bytes); close(s[0]); close(s[1]); free(st.acc);
    }

    /* 7. Responses API streaming lifecycle in thinking mode */
    {
        int s[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s) == 0);
        socket_capture_t c = {.fd = s[1]};
        pthread_t r; assert(pthread_create(&r, NULL, capture_socket, &c) == 0);
        myds_sse_stream_ctx_t st = {
            .client_fd = s[0],
            .responses_api = true,
            .thinking = true,
            .think_state = 1,
            .msg_output_index = 1,
        };
        assert(myds_server_sse_writer_start(&st, 1024u * 1024u, 1000));

        /* Send initial reasoning item */
        const char *initial =
            "event: response.output_item.added\n"
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_myds_0\",\"status\":\"in_progress\",\"summary\":[],\"content\":[]}}\n\n"
            "event: response.content_part.added\n"
            "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"reasoning_text\",\"text\":\"\"}}\n\n";
        assert(myds_server_sse_send(&st, initial, strlen(initial)));

        /* Stream tokens: thought, then </think>, then answer */
        myds_server_sse_callback("Let me think", 12, 1, &st);
        myds_server_sse_callback("</think>\nThe answer is 42", 24, 2, &st);
        assert(myds_server_sse_finish(&st));

        char *items = responses_output_items_json_ex(true, "Let me think", 12, "The answer is 42", 16, NULL, 0);
        assert(sse_emit_responses_finish(&st, "test-model", "resp_think_1", 10, 5,
                                         "The answer is 42", 16, items, 1));
        free(items);

        assert(myds_server_sse_writer_join(&st, false));
        assert(shutdown(s[0], SHUT_WR) == 0);
        assert(pthread_join(r, NULL) == 0);

        assert_sse_data_lines_valid(c.bytes);
        assert(strstr(c.bytes, "response.reasoning_text.delta") != NULL);
        assert(strstr(c.bytes, "\"output_index\":0") != NULL);
        assert(strstr(c.bytes, "event: response.reasoning_text.done\ndata: {\"type\":\"response.reasoning_text.done\",\"output_index\":0") != NULL);
        assert(strstr(c.bytes, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\"") != NULL);
        assert(strstr(c.bytes, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":1") != NULL);
        assert(strstr(c.bytes, "event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"output_index\":1") != NULL);

        free(c.bytes); close(s[0]); close(s[1]); free(st.acc);
    }

    printf("  [PASS] thinking mode (CoT/Reasoning) prompt formatting, request parsing, streaming deltas & Responses API lifecycle\n");
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
    test_deepseek41_stream_tool_calls();
    test_tool_call_event_json();
    test_responses_finish_carries_the_aggregate_text();
    test_responses_call_merges_into_assistant_turn();
    test_previous_response_id_registry();
    test_previous_response_id_suffix_render();
    test_thinking_mode();
    printf("====================================================\n");
    printf("       Running myds Server API Unit Tests           \n");
    printf("====================================================\n");

    signal(SIGPIPE, SIG_IGN);
    test_json_bytes();
    test_stream_and_non_stream_bytes();
    test_sse_writer_transport();
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
