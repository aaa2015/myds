/* Per-kernel GPU timing harness for steady-state decode.
 *
 *   bin/bench_kprof model.gguf request.json [warm_tokens=8] [kprof_tokens=24] [max_seq=4096]
 *
 * Loads the model once, generates warm_tokens + kprof_tokens + 1 tokens, and
 * records per-dispatch GPU timestamps for exactly kprof_tokens decode steps
 * (armed from the warm_tokens-th output callback, disarmed kprof_tokens later).
 * stdout carries only JSONL (kprof_kernel / kprof_summary / kprof_run); logs go
 * to stderr. Feed the JSONL to tools/kprof_report.py.
 *
 * Armed kernels run in their own encoders (serialized). Durations are isolated
 * per-kernel GPU times; the armed run's wall time is meaningless and is not
 * reported as speed. Use bench_phase for throughput.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_server_internal.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint32_t count, warm, window;
    int armed, done;
} kp_state_t;

static void on_token(const char *piece, size_t length, int32_t id, void *opaque) {
    (void)piece; (void)length; (void)id;
    kp_state_t *s = opaque;
    ++s->count;
    if (!s->armed && !s->done && s->count == s->warm) {
        myds_metal_kprof_arm(1);
        s->armed = 1;
    } else if (s->armed && s->count == s->warm + s->window) {
        myds_metal_kprof_arm(0);
        s->armed = 0;
        s->done = 1;
    }
}

static uint32_t arg_u32(const char *s, uint32_t lo, uint32_t hi, const char *name) {
    char *end = NULL; errno = 0;
    const unsigned long v = strtoul(s, &end, 10);
    if (errno || !s[0] || !end || *end || v < lo || v > hi) {
        fprintf(stderr, "Invalid %s (%u..%u): %s\n", name, lo, hi, s);
        exit(2);
    }
    return (uint32_t)v;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *b = malloc((size_t)n + 1u);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f);
    b[n] = 0;
    return b;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "Usage: %s model.gguf request.json [warm_tokens=8] [kprof_tokens=24] [max_seq=4096]\n", argv[0]);
        return 2;
    }
    const uint32_t warm = argc > 3 ? arg_u32(argv[3], 1, 4096, "warm_tokens") : 8;
    const uint32_t window = argc > 4 ? arg_u32(argv[4], 1, 64, "kprof_tokens") : 24;
    const uint32_t max_seq = argc > 5 ? arg_u32(argv[5], 256, UINT32_MAX, "max_seq") : 4096;
    /* 尊重外部传入的 MYDS_KPROF：为 0 时按普通 decode 跑，用于隔离
     * 「失败是 kprof 引起的」还是「这个 bench 本身就不支持该配置」。 */
    if (!getenv("MYDS_KPROF")) setenv("MYDS_KPROF", "1", 1);

    char *body = read_file(argv[2]);
    if (!body) return 2;
    const size_t cap = strlen(body) * 4u + 65536u;
    char *prompt = malloc(cap);
    if (!prompt || !parse_chat_prompt(body, prompt, cap)) {
        fputs("parse_chat_prompt failed\n", stderr);
        free(prompt); free(body);
        return 2;
    }
    free(body);

    /* stdout is reserved for JSONL; native logging goes to stderr. */
    fflush(stdout);
    const int json_fd = dup(STDOUT_FILENO);
    FILE *json = json_fd < 0 ? NULL : fdopen(json_fd, "w");
    if (!json || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) { perror("redirect stdout"); return 2; }

    myds_model_t *model = myds_model_load(argv[1]);
    if (!model) { fputs("Model load failed\n", stderr); return 1; }
    const myds_config_t cfg = {.max_seq_len = max_seq, .max_new_tokens = warm + window + 1u,
        .temperature = 0, .top_p = 1, .top_k = 0, .min_p = 0, .repetition_penalty = 1,
        .seed = 42, .stream = false};
    myds_context_t *ctx = myds_context_create(model, &cfg);
    if (!ctx) { fputs("Context creation failed\n", stderr); myds_model_free(model); return 1; }
    if (!myds_metal_kprof_enabled()) {
        fputs("GPU kernel timing unavailable on this device (see message above)\n", stderr);
        myds_context_free(ctx); myds_model_free(model);
        return 2;
    }

    kp_state_t st = {.warm = warm, .window = window};
    myds_stop_reason_t reason = MYDS_STOP_ERROR;
    uint32_t prompt_tokens = 0;
    const int ok = myds_generate_bytes_ex(ctx, prompt, on_token, &st, &reason, &prompt_tokens, NULL);
    if (st.armed) myds_metal_kprof_arm(0);   /* generation ended inside the window */

    int status = 0;
    double floor_us = 0;
    if (!ok || !st.done) {
        fprintf(stderr, "generation ended before the window closed (ok=%d outputs=%u needed=%u)\n",
                ok, st.count, warm + window);
        status = 1;
    } else {
        if (!myds_metal_kprof_calibrate(&floor_us))
            fputs("warning: encoder floor calibration failed; floor_us=0 (raw bracket times)\n", stderr);
        fprintf(json, "{\"kind\":\"kprof_run\",\"model\":\"%s\",\"request\":\"%s\",\"prompt_tokens\":%u,"
                "\"warm_tokens\":%u,\"kprof_tokens\":%u,\"max_seq\":%u,\"outputs\":%u,\"floor_us\":%.3f}\n",
                argv[1], argv[2], prompt_tokens, warm, window, max_seq, st.count, floor_us);
        const int rc = myds_metal_kprof_report(json, "decode", window);
        if (rc != 1) {
            fprintf(stderr, rc == 2 ? "kprof window truncated: sample buffers full before %u tokens finished; results invalid\n"
                                    : "kprof report failed: no valid samples\n", window);
            status = 1;
        }
    }
    fflush(json);
    fclose(json);
    myds_context_free(ctx);
    myds_model_free(model);
    free(prompt);
    return status;
}
