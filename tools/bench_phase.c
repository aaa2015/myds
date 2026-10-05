/* Isolated measurement harness: one model/context, one warmup, then repeats.
 * API logging and phase records go to stderr; stdout contains only JSONL.
 * The HTTP prompt renderer is reused, while transport/queue work is excluded. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_server_internal.h"
#include "myds_v41_gpu.h"
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/qos.h>
#endif

enum { MAX_OUTPUT_TOKENS = 64 };
typedef struct {
    uint32_t count;
    bool overflow;
    int32_t ids[MAX_OUTPUT_TOKENS];
    double times[MAX_OUTPUT_TOKENS], first, last;
    uint64_t byte_count, bytes_hash, ids_hash;
} capture_t;

static double now_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) { perror("clock_gettime"); exit(2); }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t hash_bytes(uint64_t hash, const void *bytes, size_t length) {
    const unsigned char *p = bytes;
    for (size_t i = 0; i < length; ++i) { hash ^= p[i]; hash *= UINT64_C(1099511628211); }
    return hash;
}

static void capture(const char *piece, size_t length, int32_t id, void *opaque) {
    capture_t *c = opaque;
    const double t = now_s();
    if (c->count == 0) c->first = t;
    c->last = t;
    if (c->count < MAX_OUTPUT_TOKENS) {
        c->ids[c->count] = id;
        c->times[c->count] = t;
    } else c->overflow = true;
    ++c->count;
    c->byte_count += length;
    c->bytes_hash = hash_bytes(c->bytes_hash, piece, length);
    unsigned char encoded[4];
    for (unsigned i = 0; i < 4; ++i) encoded[i] = (unsigned char)((uint32_t)id >> (8u * i));
    c->ids_hash = hash_bytes(c->ids_hash, encoded, sizeof(encoded));
}

static uint32_t positive(const char *s, uint32_t maximum, const char *name) {
    char *end = NULL; errno = 0;
    const unsigned long value = strtoul(s, &end, 10);
    if (errno || !s[0] || !end || *end || value == 0 || value > maximum) {
        fprintf(stderr, "Invalid %s (1..%u): %s\n", name, maximum, s); exit(2);
    }
    return (uint32_t)value;
}

static char *read_request(const char *path, size_t *length) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const long n = ftell(f);
    if (n < 0 || (uint64_t)n > SIZE_MAX - 1u || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *body = malloc((size_t)n + 1u);
    if (!body) { fclose(f); return NULL; }
    if (fread(body, 1, (size_t)n, f) != (size_t)n || memchr(body, 0, (size_t)n)) {
        fprintf(stderr, "Request read failed or contains NUL: %s\n", path);
        free(body); fclose(f); return NULL;
    }
    fclose(f); body[n] = 0; *length = (size_t)n; return body;
}

static void json_string(FILE *out, const char *value) {
    if (!value) { fputs("null", out); return; }
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        if (*p == '"' || *p == '\\') { fputc('\\', out); fputc(*p, out); }
        else if (*p < 0x20u) fprintf(out, "\\u%04x", (unsigned)*p);
        else fputc(*p, out);
    }
    fputc('"', out);
}

static const char *stop_name(myds_stop_reason_t reason) {
    switch (reason) {
        case MYDS_STOP_EOS: return "eos";
        case MYDS_STOP_LENGTH: return "length";
        case MYDS_STOP_CTX_FULL: return "ctx_full";
        case MYDS_STOP_ERROR: return "error";
        case MYDS_STOP_CANCELLED: return "cancelled";
    }
    return "unknown";
}

static void metric(FILE *out, const char *key, bool valid, double value) {
    fprintf(out, ",\"%s\":", key);
    if (valid) fprintf(out, "%.9f", value); else fputs("null", out);
}

/* Optional storage evidence, outside every request's timed interval. These
 * are global block-driver counters, not per-process or physical NAND bytes. */
static bool storage_counters(FILE *json, uint32_t run, const char *phase) {
    if (!getenv("MYDS_BENCH_STORAGE_COUNTERS")) return true;
    FILE *pipe = popen("/usr/sbin/ioreg -r -c IOBlockStorageDriver -l -d 1", "r");
    if (!pipe) return false;
    char text[65536];
    size_t n = fread(text, 1, sizeof(text) - 1u, pipe);
    bool ok = !ferror(pipe) && n < sizeof(text) - 1u;
    if (pclose(pipe) != 0) ok = false;
    if (!ok) return false;
    text[n] = 0;
    fprintf(json, "{\"kind\":\"storage_counters\",\"run\":%u,\"phase\":", run);
    json_string(json, phase);
    fputs(",\"ioreg\":", json); json_string(json, text); fputs("}\n", json); fflush(json);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "Usage: %s model.gguf request.json [repeats=3 (1..4)] [max_new_tokens=64 (32|64)] [max_seq_len=262144]\n", argv[0]);
        return 2;
    }
    const uint32_t repeats = argc > 3 ? positive(argv[3], 4, "repeats") : 3;
    const uint32_t max_new = argc > 4 ? positive(argv[4], MAX_OUTPUT_TOKENS, "max_new_tokens") : 64;
    const uint32_t max_seq = argc > 5 ? positive(argv[5], UINT32_MAX, "max_seq_len") : 262144;
    if (max_new != 32 && max_new != 64) { fputs("max_new_tokens must be 32 or 64\n", stderr); return 2; }
    const char *prefix = getenv("MYDS_PREFIX_CACHE");
    if (!prefix || strcmp(prefix, "0")) { fputs("Set MYDS_PREFIX_CACHE=0 for full-prefill measurements\n", stderr); return 2; }
    const char *forbidden[] = {"MYDS_PROFILE_PREFILL", "MYDS_PROFILE_SPLIT", "MYDS_PROFILE_SPLIT2",
        "MYDS_PROFILE_SPLIT3", "MYDS_PROFILE_MOE", "MYDS_PROFILE_STEP1", "MYDS_MOE_STEP_PROF",
        "MYDS_TRACE_PREFIX", "MYDS_BATCH_TRACE_PREFIX", "MYDS_DISABLE_V41_PREFILL",
        "MYDS_SKIP_Q_A", "MYDS_SKIP_KV", "MYDS_SKIP_NORM", "MYDS_SKIP_KV_ROPE",
        "MYDS_SKIP_KV_QUANT", "MYDS_SKIP_Q_B", "MYDS_SKIP_Q_ROPE"};
    for (size_t i = 0; i < sizeof(forbidden)/sizeof(forbidden[0]); ++i) {
        if (getenv(forbidden[i])) { fprintf(stderr, "Unset %s: it changes the measured execution path\n", forbidden[i]); return 2; }
    }
    size_t body_bytes = 0;
    char *body = read_request(argv[2], &body_bytes);
    if (!body || body_bytes > (SIZE_MAX - 8192u) / 4u) { free(body); return 2; }
    size_t prompt_cap = body_bytes * 4u + 8192u;
    if (prompt_cap < 65536u) prompt_cap = 65536u;
    char *prompt = malloc(prompt_cap);
    if (!prompt || !parse_chat_prompt(body, prompt, prompt_cap)) {
        fputs("parse_chat_prompt failed; no model was loaded\n", stderr);
        free(prompt); free(body); return 2;
    }
    const uint64_t seed_hash = UINT64_C(14695981039346656037);
    const uint64_t request_hash = hash_bytes(seed_hash, body, body_bytes);
    const size_t prompt_bytes = strlen(prompt);
    const uint64_t prompt_hash = hash_bytes(seed_hash, prompt, prompt_bytes);
    free(body);

    /* Retain a separate JSON output handle, then keep native printf/logging off it. */
    fflush(stdout);
    const int json_fd = dup(STDOUT_FILENO);
    FILE *json = json_fd < 0 ? NULL : fdopen(json_fd, "w");
    if (!json || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        perror("redirect stdout"); if (json) fclose(json); else if (json_fd >= 0) close(json_fd);
        free(prompt); return 2;
    }
    int qos_result = 0;
#if defined(__APPLE__)
    qos_result = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    fprintf(json, "{\"kind\":\"metadata\",\"model\":"); json_string(json, argv[1]);
    fputs(",\"request_file\":", json); json_string(json, argv[2]);
    fprintf(json, ",\"repeats\":%u,\"warmup_runs\":1,\"max_new_tokens\":%u,\"max_seq_len\":%u,"
        "\"temperature\":0,\"top_p\":1,\"top_k\":0,\"min_p\":0,\"repetition_penalty\":1,\"seed\":42,"
        "\"qos\":\"user_interactive\",\"qos_result\":%d,\"prompt_renderer\":\"parse_chat_prompt\","
        "\"request_bytes\":%zu,\"prompt_bytes\":%zu,\"request_fnv1a64\":\"%016" PRIx64 "\","
        "\"prompt_fnv1a64\":\"%016" PRIx64 "\",\"clock\":\"CLOCK_MONOTONIC\","
        "\"transport\":\"direct_byte_callback\",\"decode_basis\":\"first_to_last_callback/(tokens-1)\",\"environment\":{",
        repeats, max_new, max_seq, qos_result, body_bytes, prompt_bytes, request_hash, prompt_hash);
    const char *env_names[] = {"MYDS_PHASE_PROFILE", "MYDS_PHASE_PROFILE_FILE", "MYDS_PREFIX_CACHE", "MYDS_KV_DSK", "MYDS_KV_DSK_ASYNC",
        "MYDS_PREFILL_CHUNK_SIZE", "MYDS_PREFILL_MACROBLOCK_SIZE", "MYDS_MOE_CHUNKS", "MYDS_MOE_MM_MIN_TOKENS",
        "MYDS_DISABLE_MOE_MM", "MYDS_DECODE_SPLIT", "MYDS_ATTN_GATHER", "MYDS_SUFFIX_SHRINK", "MYDS_SUFFIX_SHRINK_SHIFT",
        "MYDS_RESIDENCY_KEEP_ALIVE_S", "MYDS_RESIDENCY_HB_MS", "MYDS_NO_RESIDENCY_HB", "MYDS_NO_PREFIX_SNAPSHOT",
        "MYDS_CHUNK_KEY_MAX", "MYDS_ATTN_KEYS_CLAMP", "MYDS_INDEXER_BATCH_FLOATS", "MYDS_INDEXER_TILED",
        "MYDS_BLOCK_BUDGET", "MYDS_DISABLE_MUL_MM", "MYDS_MM_MIN_M", "MYDS_DISABLE_MM_ENGRAM",
        "MYDS_DISABLE_MM_ATTN_A", "MYDS_COMPRESSOR_PER_TOKEN", "MYDS_INDEXER_PER_TOKEN",
        /* 2026-10-05 path selectors: without them two configurations of one
         * binary produce identical metadata (see head-first-long-validation). */
        "MYDS_ATTN_VEC_HEAD_FIRST", "MYDS_ATTN_GATHER_MIN_KEYS", "MYDS_PREFILL_ATTN",
        "MYDS_QKV_FUSE_BF16", "MYDS_QKV_FUSE_POST", "MYDS_QKV_NR0", "MYDS_MOE_IQ2_TAIL_CULL",
        "MYDS_MOE_ROUTE_FUSE",
        "MYDS_ATTN_OUT_A_FUSE_BF16",
        "MYDS_ATTN_DIRECT_PACK",
        "MYDS_ENGRAM_IO_THREADS", "MYDS_ENGRAM_NOCACHE", "MYDS_BENCH_STORAGE_COUNTERS",
        "MYDS_GEMV_BATCH_SIZE"};
    for (size_t i = 0; i < sizeof(env_names)/sizeof(env_names[0]); ++i) {
        if (i) fputc(',', json); json_string(json, env_names[i]); fputc(':', json); json_string(json, getenv(env_names[i]));
    }
    fputs("}}\n", json); fflush(json);

    const double load_start = now_s();
    myds_model_t *model = myds_model_load(argv[1]);
    const double load_s = now_s() - load_start;
    if (!model) { fputs("Model load failed\n", stderr); free(prompt); fclose(json); return 1; }
    const myds_config_t cfg = {.max_seq_len = max_seq, .max_new_tokens = max_new,
        .temperature = 0, .top_p = 1, .top_k = 0, .min_p = 0, .repetition_penalty = 1, .seed = 42, .stream = false};
    const double context_start = now_s();
    myds_context_t *ctx = myds_context_create(model, &cfg);
    const double context_s = now_s() - context_start;
    if (!ctx) { fputs("Context creation failed\n", stderr); myds_model_free(model); free(prompt); fclose(json); return 1; }
    fprintf(json, "{\"kind\":\"load\",\"load_s\":%.9f,\"context_create_s\":%.9f,\"model_arch\":", load_s, context_s);
    json_string(json, model->arch);
    fprintf(json, ",\"layers\":%u,\"dim\":%u,\"heads\":%u,\"head_dim\":%u,\"vocab_size\":%u,"
        "\"routed_experts\":%u,\"activated_experts\":%u,\"expert_width\":%u,\"shared_experts\":%u,\"native_v41\":%s}\n",
        model->n_layers, model->dim, model->n_heads, model->head_dim, model->vocab_size,
        model->n_routed_experts, model->n_activated_experts, model->moe_expert_dim, model->n_shared_experts,
        model->v41 ? "true" : "false");
    fflush(json);
    int status = 0;
    uint64_t expected_ids = 0, expected_bytes = 0;
    uint32_t expected_count = 0;
#ifdef MYDS_V41_GPU_HAS_ATTENTION_COUNTERS
    myds_v41_gpu_attention_counters(NULL, true);   /* exclude load/context creation */
#endif
    for (uint32_t run = 0; run <= repeats; ++run) {
        capture_t c = {.bytes_hash = seed_hash, .ids_hash = seed_hash};
        myds_stop_reason_t reason = MYDS_STOP_ERROR; uint32_t prompt_tokens = 0;
        if (!storage_counters(json, run, "before")) { status = 1; break; }
        const double start = now_s();
        const int ok = myds_generate_bytes_ex(ctx, prompt, capture, &c, &reason, &prompt_tokens, NULL);
        const double wall_s = now_s() - start;
        if (!storage_counters(json, run, "after")) { status = 1; break; }
        myds_prefill_stats_t pf; myds_last_prefill_stats(ctx, &pf);
        /* Resolve available profiler records only after generation's existing waits.
         * This export is deliberately outside the measured request wall interval. */
        /* No profiler in this uninstrumented baseline. */
#ifdef MYDS_V41_GPU_HAS_ATTENTION_COUNTERS
        /* Host counters only, read after generation returned (outside wall_s). */
        myds_v41_gpu_attention_counters_t ac;
        myds_v41_gpu_attention_counters(&ac, true);
#endif
        const bool generation_ok = ok && (reason == MYDS_STOP_EOS || reason == MYDS_STOP_LENGTH);
        const bool full_prefill = pf.cached_tokens == 0 && prompt_tokens > 0 &&
            pf.prefill_tokens == prompt_tokens - 1u && pf.prompt_tokens == prompt_tokens;
        const bool decode_valid = c.count >= 2 && !c.overflow && c.last > c.first;
        const bool count_valid = c.count <= max_new && (reason != MYDS_STOP_LENGTH || c.count == max_new);
        if (run == 0) { expected_ids = c.ids_hash; expected_bytes = c.bytes_hash; expected_count = c.count; }
        const bool output_equal = c.ids_hash == expected_ids && c.bytes_hash == expected_bytes && c.count == expected_count;
        const bool valid = generation_ok && full_prefill && decode_valid && count_valid && output_equal;
        fprintf(json, "{\"kind\":\"sample\",\"run\":%u,\"warmup\":%s,\"generation_ok\":%s,\"sample_valid\":%s,"
            "\"stop_reason\":\"%s\",\"stop_reason_code\":%d,\"prompt_tokens\":%u,\"cached_tokens\":%u,"
            "\"prefill_tokens\":%u,\"prefill_s\":%.9f,\"output_tokens\":%u,\"decode_intervals\":%u,\"wall_s\":%.9f,"
            "\"output_bytes\":%" PRIu64 ",\"output_bytes_fnv1a64\":\"%016" PRIx64 "\",\"token_ids_fnv1a64\":\"%016" PRIx64 "\","
            "\"output_equal_to_warmup\":%s,\"token_ids\":[",
            run, run == 0 ? "true" : "false", generation_ok ? "true" : "false", valid ? "true" : "false",
            stop_name(reason), (int)reason, prompt_tokens, pf.cached_tokens, pf.prefill_tokens, pf.prefill_seconds,
            c.count, c.count > 0 ? c.count - 1u : 0u, wall_s, c.byte_count, c.bytes_hash, c.ids_hash, output_equal ? "true" : "false");
        const uint32_t retained = c.count < MAX_OUTPUT_TOKENS ? c.count : MAX_OUTPUT_TOKENS;
        for (uint32_t i = 0; i < retained; ++i) fprintf(json, "%s%d", i ? "," : "", c.ids[i]);
        fputc(']', json);
        metric(json, "ttft_s", c.count > 0, c.first - start);
        metric(json, "first_callback_s", c.count > 0, c.first - start);
        metric(json, "last_callback_s", c.count > 0, c.last - start);
        metric(json, "prefill_ms_per_token", full_prefill && pf.prefill_tokens > 0, 1000.0 * pf.prefill_seconds / (pf.prefill_tokens ? pf.prefill_tokens : 1));
        metric(json, "decode_s", decode_valid, c.last - c.first);
        metric(json, "decode_ms_per_token", decode_valid, 1000.0 * (c.last - c.first) / (c.count > 1 ? c.count - 1u : 1u));
#ifdef MYDS_V41_GPU_HAS_ATTENTION_COUNTERS
        fprintf(json, ",\"attention_paths\":{\"calls\":%" PRIu64 ",\"gather_calls\":%" PRIu64
            ",\"slab_calls\":%" PRIu64 ",\"head_first_attempts\":%" PRIu64 ",\"head_first_encoded\":%" PRIu64
            ",\"head_first_failures\":%" PRIu64 ",\"head_first_cached_fallbacks\":%" PRIu64
            ",\"original_vec_encoded\":%" PRIu64 ",\"matrix_encoded\":%" PRIu64 ",\"matrix_fallbacks\":%" PRIu64 "}",
            ac.calls, ac.gather_calls, ac.slab_calls, ac.head_first_attempts, ac.head_first_encoded,
            ac.head_first_failures, ac.head_first_cached_fallbacks, ac.original_vec_encoded,
            ac.matrix_encoded, ac.matrix_fallbacks);
        if (ac.head_first_failures || ac.head_first_cached_fallbacks || ac.matrix_fallbacks)
            fprintf(stderr, "run %u: attention fallback: head_first_failures=%" PRIu64
                " cached_fallbacks=%" PRIu64 " matrix_fallbacks=%" PRIu64 "\n", run,
                ac.head_first_failures, ac.head_first_cached_fallbacks, ac.matrix_fallbacks);
#endif
        fputs(",\"callback_intervals_ms\":[", json);
        for (uint32_t i = 1; i < retained; ++i) fprintf(json, "%s%.9f", i > 1 ? "," : "", 1000.0 * (c.times[i] - c.times[i-1]));
        fputs("]}\n", json); fflush(json);
        if (!valid) { fprintf(stderr, "Invalid measurement run %u: ok=%d stop=%s full_prefill=%d count=%u output_equal=%d\n",
            run, ok, stop_name(reason), full_prefill, c.count, output_equal); status = 1; break; }
    }
    myds_context_free(ctx); myds_model_free(model); free(prompt); fclose(json);
    return status;
}
