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
#include "myds_v41.h"
#include "myds_dspark.h"
#include <math.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
extern char **environ;
#if defined(__APPLE__)
#include <sys/qos.h>
#endif

enum { MAX_OUTPUT_TOKENS = 512 };
typedef struct {
    uint32_t count;
    bool overflow;
    int32_t ids[MAX_OUTPUT_TOKENS];
    double times[MAX_OUTPUT_TOKENS], first, last;
    uint64_t byte_count, bytes_hash, ids_hash;
    myds_dspark_t *dspark;
    myds_dspark_capture_counters_t first_dspark_capture;
    myds_v41_gpu_topk_prefix_counters_t first_topk_prefix;
    myds_v41_gpu_topk_nan_fuse_counters_t first_nan_fuse;
    myds_v41_gpu_topk_nan_shuffle_counters_t first_nan_shuffle;
    myds_v41_gpu_indexer_head_slots_counters_t first_indexer_head_slots;
    myds_v41_gpu_topk_threshold_counters_t first_threshold_host;
    myds_v41_gpu_topk_threshold_stats_t first_threshold_gpu;
    bool first_threshold_stats_completed;
    myds_metal_batch2_q8_shared_counters_t first_q8_shared;
    myds_metal_lm_head_q8_counters_t first_lm_head_q8;
    myds_v41_engram_project_round_counters_t first_engram_round;
    myds_metal_moe_q2_sum_cache_counters_t first_moe_q2_sum_cache;
    myds_metal_moe_q2_sum_cache_fused_counters_t first_moe_q2_sum_cache_fused;
    myds_metal_f16_gemv_bf16_fuse_counters_t first_f16_bf16_fuse;
    myds_metal_indexer_rope_quant_fuse_counters_t first_indexer_rope_quant;
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
    if (c->count == 0) {
        c->first = t;
        myds_dspark_capture_counters(c->dspark,&c->first_dspark_capture,false);
        myds_v41_gpu_topk_prefix_counters(&c->first_topk_prefix, false);
        myds_v41_gpu_topk_nan_fuse_counters(&c->first_nan_fuse, false);
        myds_v41_gpu_topk_nan_shuffle_counters(&c->first_nan_shuffle, false);
        myds_v41_gpu_indexer_head_slots_counters(&c->first_indexer_head_slots, false);
        myds_v41_gpu_topk_threshold_counters(&c->first_threshold_host, false);
        c->first_threshold_stats_completed = myds_v41_gpu_topk_threshold_stats(&c->first_threshold_gpu, false) != 0;
        myds_metal_batch2_q8_shared_counters(&c->first_q8_shared, false);
        myds_metal_lm_head_q8_counters(&c->first_lm_head_q8, false);
        myds_v41_engram_project_round_counters(&c->first_engram_round, false);
        myds_metal_moe_q2_sum_cache_counters(&c->first_moe_q2_sum_cache, false);
        myds_metal_moe_q2_sum_cache_fused_counters(&c->first_moe_q2_sum_cache_fused, false);
        myds_metal_f16_gemv_bf16_fuse_counters(&c->first_f16_bf16_fuse, false);
        myds_metal_indexer_rope_quant_fuse_counters(&c->first_indexer_rope_quant, false);
    }
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

/* Diagnostic opt-in, read after generation's waits and outside the timed
 * interval. Keep raw floats for investigating a rejected hash gate; never
 * overwrite an earlier run and do not weaken the native validity check. */
static bool export_logits(uint32_t run, const float *logits, uint32_t count) {
    const char *dir = getenv("MYDS_BENCH_LOGITS_DIR");
    if (!dir) return true;
    if (!*dir || !logits) return false;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/run-%u.f32", dir, run);
    if (n < 0 || (size_t)n >= sizeof(path)) return false;
    FILE *f = fopen(path, "wbx");
    if (!f) { perror(path); return false; }
    bool ok = fwrite(logits, sizeof(*logits), count, f) == count;
    if (fclose(f) != 0) ok = false;
    return ok;
}

static void decode_metrics(FILE *json, myds_context_t *ctx) {
    myds_decode_stats_t ds = {0};
    myds_last_decode_stats(ctx, &ds);
    fputs(",\"decode_telemetry\":{", json);
#define COUNTER(name) fprintf(json, "\"" #name "\":%" PRIu64 ",", ds.name)
    COUNTER(target_forward_calls); COUNTER(target_rows);
    COUNTER(batch2_calls); COUNTER(batch2_hits); COUNTER(batch2_misses);
    COUNTER(batch2_failures); COUNTER(batch2_fallbacks);
    COUNTER(draft_cycles); COUNTER(draft_proposed); COUNTER(draft_accepted); COUNTER(draft_rejects);
    COUNTER(confidence_rejects); COUNTER(margin_rejects); COUNTER(backoff_steps);
    COUNTER(pld_queries); COUNTER(pld_cycles); COUNTER(pld_proposed); COUNTER(pld_accepted);
    COUNTER(output_tokens); COUNTER(cycle_count);
#undef COUNTER
    fprintf(json, "\"timing_enabled\":%s,\"recorded_cycles\":%u", ds.timing_enabled ? "true" : "false", ds.recorded_cycles);
    metric(json, "draft_ms", ds.timing_enabled, ds.draft_ms);
    metric(json, "verify_ms", ds.timing_enabled, ds.verify_ms);
    metric(json, "rollback_ms", ds.timing_enabled, ds.rollback_ms);
    metric(json, "snapshot_ms", ds.timing_enabled, ds.snapshot_ms);
    metric(json, "pld_ms", ds.timing_enabled, ds.pld_ms);
    fputs(",\"cycle_ms\":[", json);
    for (uint32_t i = 0; i < ds.recorded_cycles; ++i) {
        myds_decode_cycle_t cycle = {0};
        if (!myds_last_decode_cycle(ctx, i, &cycle)) { fputs(i ? ",null" : "null", json); continue; }
        fprintf(json, "%s%.9f", i ? "," : "", cycle.elapsed_ms);
    }
    fputs("],\"cycle_output_tokens\":[", json);
    for (uint32_t i = 0; i < ds.recorded_cycles; ++i) {
        myds_decode_cycle_t cycle = {0};
        if (!myds_last_decode_cycle(ctx, i, &cycle)) { fputs(i ? ",null" : "null", json); continue; }
        fprintf(json, "%s%u", i ? "," : "", cycle.output_tokens);
    }
    fputs("]}", json);
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
    if (argc < 3 || argc > 7) {
        fprintf(stderr, "Usage: %s model.gguf request.json [repeats=3 (1..5)] [max_new_tokens=64 (32|64|256|512)] [max_seq_len=262144] [dspark-support.gguf]\n", argv[0]);
        return 2;
    }
    const uint32_t repeats = argc > 3 ? positive(argv[3], 5, "repeats") : 3;
    const uint32_t max_new = argc > 4 ? positive(argv[4], MAX_OUTPUT_TOKENS, "max_new_tokens") : 64;
    const uint32_t max_seq = argc > 5 ? positive(argv[5], UINT32_MAX, "max_seq_len") : 262144;
    if (max_new != 32 && max_new != 64 && max_new != 256 && max_new != 512) {
        fputs("max_new_tokens must be 32, 64, 256 or 512\n", stderr); return 2;
    }
    const char *dspark_path = argc > 6 ? argv[6] : NULL;
    if (myds_metal_f16_gemv_bf16_fuse_parse_mode() < 0) return 2;
    if (myds_metal_indexer_rope_quant_fuse_parse_mode() < 0) return 2;
    const int capture_mode = myds_dspark_capture_parse_mode();
    if (capture_mode < 0) return 2;
    if (capture_mode != 0 && !dspark_path) {
        fputs("Capture bypass mode requires DSpark support; refusing target-only fallback\n", stderr);
        return 2;
    }
    const char *prefix = getenv("MYDS_PREFIX_CACHE");
    if (!prefix || strcmp(prefix, "0")) { fputs("Set MYDS_PREFIX_CACHE=0 for full-prefill measurements\n", stderr); return 2; }
    const char *forbidden[] = {"MYDS_PROFILE_PREFILL", "MYDS_PROFILE_SPLIT", "MYDS_PROFILE_SPLIT2",
        "MYDS_PROFILE_SPLIT3", "MYDS_PROFILE_MOE", "MYDS_PROFILE_STEP1", "MYDS_MOE_STEP_PROF",
        "MYDS_PROF", "MYDS_DISPATCH_HIST", "MYDS_DEBUG_DISPATCH",
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
    fputs(",\"dspark_model\":", json); json_string(json, dspark_path);
    fprintf(json, ",\"protocol_version\":2,\"repeats\":%u,\"warmup_runs\":1,\"max_new_tokens\":%u,\"max_seq_len\":%u,"
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
        "MYDS_INDEXER_COMPACT_SCORER", "MYDS_INDEXER_TOPK_PREFIX_MERGE", "MYDS_INDEXER_TOPK_NAN_FUSE", "MYDS_INDEXER_HEAD_SLOTS",
        "MYDS_BLOCK_BUDGET", "MYDS_DISABLE_MUL_MM", "MYDS_MM_MIN_M", "MYDS_DISABLE_MM_ENGRAM",
        "MYDS_DISABLE_MM_ATTN_A", "MYDS_COMPRESSOR_PER_TOKEN", "MYDS_INDEXER_PER_TOKEN",
        /* 2026-10-05 path selectors: without them two configurations of one
         * binary produce identical metadata (see head-first-long-validation). */
        "MYDS_ATTN_VEC_HEAD_FIRST", "MYDS_ATTN_GATHER_MIN_KEYS", "MYDS_PREFILL_ATTN",
        "MYDS_QKV_FUSE_BF16", "MYDS_QKV_FUSE_POST", "MYDS_QKV_NR0", "MYDS_MOE_IQ2_TAIL_CULL",
        "MYDS_MOE_IQ2_CONSTANT_LUT", "MYDS_ATTN_PACKED_NSG", "MYDS_ATTN_PACKED_REDUCE",
        "MYDS_MOE_ROUTE_FUSE",
        "MYDS_ATTN_OUT_A_FUSE_BF16",
        "MYDS_ATTN_DIRECT_PACK",
        "MYDS_ENGRAM_IO_THREADS", "MYDS_ENGRAM_NOCACHE", "MYDS_BENCH_STORAGE_COUNTERS",
        "MYDS_GEMV_BATCH_SIZE", "MYDS_PLD", "MYDS_BATCH2_SPEC", "MYDS_BATCH2_Q8_SHARED", "MYDS_TREE_SPEC",
        "MYDS_DSPARK_MARKOV_ONLY", "MYDS_DSPARK_MARKOV_SCALE", "MYDS_DSPARK_MIN_CONF",
        "MYDS_DSPARK_MIN_MARGIN", "MYDS_DSPARK_HIGH_CONF_MARGIN", "MYDS_DSPARK_COST_GATE",
        "MYDS_ANCHOR_SNAPSHOT", "MYDS_KV_DSK_PREFER_DEEPER", "MYDS_KV_DSK_BUDGET_MB",
        "MYDS_KV_DSK_WAYS", "MYDS_MODEL_DIGEST", "MYDS_FORCE_FULL_DIGEST",
        "MYDS_METAL_CONCURRENT", "MYDS_MOE_CONCURRENT", "MYDS_MOE_PIPELINE",
        "MYDS_MOE_FUSE_EXPAND", "MYDS_MOE_Q2_SUM_CACHE", "MYDS_MOE_Q2_SUM_CACHE_FUSED", "MYDS_MOE_NSG", "MYDS_MOE_DOWN_NSG", "MYDS_SHARED_FUSE", "MYDS_SHARED_NR0",
        "MYDS_ATTN_INDEXER_PIPELINE", "MYDS_ATTN_OUTPUT_PIPELINE", "MYDS_QKV_PIPELINE",
        "MYDS_FUSE_HC_SUM_RMSNORM", "MYDS_DECODE_Q8_FUSE_BF16", "MYDS_F16_GEMV_BF16_FUSE", "MYDS_INDEXER_ROPE_QUANT_FUSE", "MYDS_SUFFIX_SHRINK_MIN_TOKENS",
        "MYDS_DECODE_METRICS", "MYDS_SPEC_SERIAL_PROBE", "MYDS_SPEC_COST_SCHEDULER",
        "MYDS_DSPARK_CAPTURE_BYPASS", "MYDS_ENGINE_CAPTURE_BYPASS"};
    for (size_t i = 0; i < sizeof(env_names)/sizeof(env_names[0]); ++i) {
        if (i) fputc(',', json); json_string(json, env_names[i]); fputc(':', json); json_string(json, getenv(env_names[i]));
    }
    /* Also bind future/experimental MYDS_ selectors, without silently omitting
     * a requested environment setting when this list has not yet been updated. */
    for (char **entry = environ; entry && *entry; ++entry) {
        if (strncmp(*entry, "MYDS_", 5) != 0) continue;
        const char *equal = strchr(*entry, '=');
        if (!equal) continue;
        const size_t length = (size_t)(equal - *entry);
        bool known = false;
        for (size_t i = 0; i < sizeof(env_names)/sizeof(env_names[0]); ++i) {
            if (strlen(env_names[i]) == length && memcmp(env_names[i], *entry, length) == 0) { known = true; break; }
        }
        if (known) continue;
        char *name = malloc(length + 1u);
        if (!name) { free(prompt); fclose(json); return 1; }
        memcpy(name, *entry, length); name[length] = 0;
        fputc(',', json); json_string(json, name); fputc(':', json); json_string(json, equal + 1);
        free(name);
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
    double dspark_load_s = 0;
    if (dspark_path) {
        const double dspark_start = now_s();
        ctx->dspark = myds_dspark_load(dspark_path, model->vocab_size);
        dspark_load_s = now_s() - dspark_start;
        if (!ctx->dspark) {
            fputs("DSpark support model load failed; refusing target-only fallback\n", stderr);
            myds_context_free(ctx); myds_model_free(model); free(prompt); fclose(json); return 1;
        }
    }
    fprintf(json, "{\"kind\":\"load\",\"load_s\":%.9f,\"context_create_s\":%.9f,\"model_arch\":", load_s, context_s);
    json_string(json, model->arch);
    fprintf(json, ",\"layers\":%u,\"dim\":%u,\"heads\":%u,\"head_dim\":%u,\"vocab_size\":%u,"
        "\"routed_experts\":%u,\"activated_experts\":%u,\"expert_width\":%u,\"shared_experts\":%u,\"native_v41\":%s,"
        "\"dspark_loaded\":%s,\"dspark_load_s\":%.9f,\"dspark_block_size\":%u,\"dspark_stages\":%u,"
        "\"lm_head_weight_type\":%d,\"moe_q2_sum_cache_mode\":%d,\"moe_q2_sum_cache_fused_mode\":%d,\"indexer_head_slots_mode\":%d,\"topk_threshold_mode\":%d,\"topk_nan_shuffle_mode\":%d,\"dspark_capture_bypass_mode\":%d,\"engine_capture_bypass_mode\":%d,\"f16_gemv_bf16_fuse_mode\":%d,\"indexer_rope_quant_fuse_mode\":%d}\n",
        model->n_layers, model->dim, model->n_heads, model->head_dim, model->vocab_size,
        model->n_routed_experts, model->n_activated_experts, model->moe_expert_dim, model->n_shared_experts,
        model->v41 ? "true" : "false", ctx->dspark ? "true" : "false", dspark_load_s,
        ctx->dspark ? ctx->dspark->block_size : 0, ctx->dspark ? ctx->dspark->stage_count : 0,
        (int)model->output_type, myds_metal_moe_q2_sum_cache_enabled(),
        myds_metal_moe_q2_sum_cache_fused_enabled(), myds_v41_gpu_indexer_head_slots_enabled(), myds_v41_gpu_topk_threshold_enabled(), myds_v41_gpu_topk_nan_shuffle_enabled(),myds_dspark_capture_bypass_enabled(ctx->dspark),
        myds_dspark_engine_capture_bypass_enabled(ctx->dspark), myds_metal_f16_gemv_bf16_fuse_enabled(), myds_metal_indexer_rope_quant_fuse_enabled());
    fflush(json);
    int status = 0;
    uint64_t expected_ids = 0, expected_bytes = 0;
    uint64_t expected_logits = 0;
    uint32_t expected_count = 0;
#ifdef MYDS_V41_GPU_HAS_ATTENTION_COUNTERS
    myds_v41_gpu_attention_counters(NULL, true);   /* exclude load/context creation */
#endif
    for (uint32_t run = 0; run <= repeats; ++run) {
        capture_t c = {.bytes_hash = seed_hash, .ids_hash = seed_hash,.dspark=ctx->dspark};
        myds_dspark_capture_counters(ctx->dspark,NULL,true);
        myds_stop_reason_t reason = MYDS_STOP_ERROR; uint32_t prompt_tokens = 0;
        /* Request-local host path counters; exclude previous requests and load. */
        myds_v41_gpu_indexer_compact_counters(NULL, true);
        myds_v41_gpu_topk_prefix_counters(NULL, true);
        myds_v41_gpu_topk_nan_fuse_counters(NULL, true);
        myds_v41_gpu_topk_nan_shuffle_counters(NULL, true);
        myds_v41_gpu_indexer_head_slots_counters(NULL, true);
        myds_v41_gpu_topk_threshold_counters(NULL, true);
        if (!myds_v41_gpu_topk_threshold_stats(NULL, true)) {
            fputs("TopK threshold stats reset attempted before GPU completion\n", stderr); status = 1; break;
        }
        myds_metal_batch2_q8_shared_counters(NULL, true);
        myds_metal_lm_head_q8_counters(NULL, true);
        myds_v41_engram_project_round_counters(NULL, true);
        myds_metal_moe_q2_sum_cache_counters(NULL, true);
        myds_metal_moe_q2_sum_cache_fused_counters(NULL, true);
        myds_metal_f16_gemv_bf16_fuse_counters(NULL, true);
        myds_metal_indexer_rope_quant_fuse_counters(NULL, true);
        if (!storage_counters(json, run, "before")) { status = 1; break; }
        const double start = now_s();
        const int ok = myds_generate_bytes_ex(ctx, prompt, capture, &c, &reason, &prompt_tokens, NULL);
        const double wall_s = now_s() - start;
        myds_dspark_capture_counters_t dspark_capture={0};
        myds_dspark_capture_counters(ctx->dspark,&dspark_capture,false);
        myds_v41_gpu_indexer_compact_counters_t compact;
        myds_v41_gpu_indexer_compact_counters(&compact, false);
        myds_v41_gpu_topk_prefix_counters_t topk_prefix;
        myds_v41_gpu_topk_prefix_counters(&topk_prefix, false);
        myds_v41_gpu_topk_nan_fuse_counters_t nan_fuse;
        myds_v41_gpu_topk_nan_fuse_counters(&nan_fuse, false);
        myds_v41_gpu_topk_nan_shuffle_counters_t nan_shuffle;
        myds_v41_gpu_topk_nan_shuffle_counters(&nan_shuffle, false);
        myds_v41_gpu_indexer_head_slots_counters_t indexer_head_slots;
        myds_v41_gpu_indexer_head_slots_counters(&indexer_head_slots, false);
        myds_v41_gpu_topk_threshold_counters_t threshold_host = {0};
        myds_v41_gpu_topk_threshold_stats_t threshold_gpu = {0};
        myds_v41_gpu_topk_threshold_counters(&threshold_host, false);
        const bool threshold_stats_completed = myds_v41_gpu_topk_threshold_stats(&threshold_gpu, false) != 0;
        myds_metal_batch2_q8_shared_counters_t q8_shared;
        myds_metal_batch2_q8_shared_counters(&q8_shared, false);
        myds_metal_lm_head_q8_counters_t lm_head_q8;
        myds_metal_lm_head_q8_counters(&lm_head_q8, false);
        myds_v41_engram_project_round_counters_t engram_round;
        myds_v41_engram_project_round_counters(&engram_round, false);
        myds_metal_moe_q2_sum_cache_counters_t moe_q2_sum_cache;
        myds_metal_moe_q2_sum_cache_counters(&moe_q2_sum_cache, false);
        myds_metal_moe_q2_sum_cache_fused_counters_t moe_q2_sum_cache_fused;
        myds_metal_moe_q2_sum_cache_fused_counters(&moe_q2_sum_cache_fused, false);
        myds_metal_f16_gemv_bf16_fuse_counters_t f16_bf16_fuse;
        myds_metal_f16_gemv_bf16_fuse_counters(&f16_bf16_fuse, false);
        myds_metal_indexer_rope_quant_fuse_counters_t indexer_rope_quant;
        myds_metal_indexer_rope_quant_fuse_counters(&indexer_rope_quant, false);
        if (!storage_counters(json, run, "after")) { status = 1; break; }
        myds_prefill_stats_t pf; myds_last_prefill_stats(ctx, &pf);
        /* Final logits are read only after the existing generation waits, outside
         * wall_s. Equality here is a regression gate, not independent model proof. */
        const float *logits = (const float *)myds_metal_buffer_data(ctx->logits);
        bool logits_finite = logits != NULL;
        uint64_t logits_hash = seed_hash;
        if (logits) {
            for (uint32_t i = 0; i < model->vocab_size; ++i) {
                if (!isfinite(logits[i])) logits_finite = false;
                logits_hash = hash_bytes(logits_hash, logits + i, sizeof(*logits));
            }
        }
        if (!export_logits(run, logits, model->vocab_size)) {
            fputs("Final logits diagnostic export failed\n", stderr); status = 1; break;
        }
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
        if (run == 0) {
            expected_ids = c.ids_hash; expected_bytes = c.bytes_hash; expected_count = c.count;
            expected_logits = logits_hash;
        }
        const bool output_equal = c.ids_hash == expected_ids && c.bytes_hash == expected_bytes && c.count == expected_count;
        const bool valid = generation_ok && full_prefill && decode_valid && count_valid && output_equal &&
                           logits_finite && logits_hash == expected_logits;
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
        fprintf(json, ",\"final_logits\":{\"elements\":%u,\"finite\":%s,\"fnv1a64_bytes\":\"%016" PRIx64 "\"}",
                model->vocab_size, logits_finite ? "true" : "false", logits_hash);
        decode_metrics(json, ctx);
        fprintf(json, ",\"gpu_indexer_compact\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"unavailable\":%" PRIu64 ",\"dense_rows\":%" PRIu64 ",\"scored_row_slots\":%" PRIu64 "}",
            compact.requests, compact.encoded, compact.unavailable, compact.dense_rows, compact.scored_row_slots);
        fprintf(json, ",\"gpu_topk_prefix\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"unavailable\":%" PRIu64 ",\"classified_rows\":%" PRIu64 ",\"merge_dispatches\":%" PRIu64
            ",\"dense_write_slots\":%" PRIu64 ",\"prefix_write_slots\":%" PRIu64
            ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64 ",\"decode_unavailable\":%" PRIu64 "}",
            topk_prefix.requests, topk_prefix.encoded, topk_prefix.unavailable, topk_prefix.classified_rows,
            topk_prefix.merge_dispatches, topk_prefix.dense_write_slots, topk_prefix.prefix_write_slots,
            topk_prefix.requests-c.first_topk_prefix.requests,
            topk_prefix.encoded-c.first_topk_prefix.encoded,
            topk_prefix.unavailable-c.first_topk_prefix.unavailable);
        fprintf(json, ",\"gpu_topk_nan_fuse\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"unavailable\":%" PRIu64 ",\"classified_partitions\":%" PRIu64
            ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64 ",\"decode_unavailable\":%" PRIu64 "}",
            nan_fuse.requests, nan_fuse.encoded, nan_fuse.unavailable, nan_fuse.classified_partitions,
            nan_fuse.requests-c.first_nan_fuse.requests, nan_fuse.encoded-c.first_nan_fuse.encoded,
            nan_fuse.unavailable-c.first_nan_fuse.unavailable);
        fprintf(json, ",\"gpu_topk_nan_shuffle\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}",nan_shuffle.requests,nan_shuffle.encoded,nan_shuffle.failures,
            nan_shuffle.requests-c.first_nan_shuffle.requests,nan_shuffle.encoded-c.first_nan_shuffle.encoded,
            nan_shuffle.failures-c.first_nan_shuffle.failures);
        fputs(",\"gpu_dspark_capture\":{",json);
#define CAPTURE_JSON(field) \
        fprintf(json,"\"" #field "\":%" PRIu64 ",\"decode_" #field "\":%" PRIu64, \
            dspark_capture.field,dspark_capture.field-c.first_dspark_capture.field)
        CAPTURE_JSON(requests);fputc(',',json);
        CAPTURE_JSON(encoded);fputc(',',json);
        CAPTURE_JSON(bypassed);fputc(',',json);
        CAPTURE_JSON(failures);fputc(',',json);
        CAPTURE_JSON(stale_rejections);
#undef CAPTURE_JSON
        fputc('}',json);
        fprintf(json, ",\"gpu_indexer_head_slots\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"unavailable\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64
            ",\"decode_encoded\":%" PRIu64 ",\"decode_unavailable\":%" PRIu64 ",\"decode_failures\":%" PRIu64 "}",
            indexer_head_slots.requests, indexer_head_slots.encoded, indexer_head_slots.unavailable, indexer_head_slots.failures,
            indexer_head_slots.requests-c.first_indexer_head_slots.requests,
            indexer_head_slots.encoded-c.first_indexer_head_slots.encoded,
            indexer_head_slots.unavailable-c.first_indexer_head_slots.unavailable,
            indexer_head_slots.failures-c.first_indexer_head_slots.failures);
        /* GPU decisions are observed only after generation's existing completion
         * synchronization; no per-TopK wait/readback is introduced here. */
        fprintf(json, ",\"gpu_topk_threshold\":{\"stats_completed\":%s",
            threshold_stats_completed && c.first_threshold_stats_completed ? "true" : "false");
#define THRESHOLD_JSON(field, current, first) \
        fprintf(json, ",\"" #field "\":%" PRIu64 ",\"decode_" #field "\":%" PRIu64, \
            (current).field, (current).field - (first).field)
        THRESHOLD_JSON(requests, threshold_host, c.first_threshold_host);
        THRESHOLD_JSON(encoded, threshold_host, c.first_threshold_host);
        THRESHOLD_JSON(errors, threshold_host, c.first_threshold_host);
        THRESHOLD_JSON(validated, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(adopted, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(fallback, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(nan, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(inf, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(insufficient, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(capacity, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(certificate, threshold_gpu, c.first_threshold_gpu);
        THRESHOLD_JSON(tie, threshold_gpu, c.first_threshold_gpu);
#undef THRESHOLD_JSON
        fputc('}', json);
        fprintf(json, ",\"gpu_batch2_q8_shared\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}", q8_shared.requests, q8_shared.encoded, q8_shared.failures,
            q8_shared.requests-c.first_q8_shared.requests, q8_shared.encoded-c.first_q8_shared.encoded,
            q8_shared.failures-c.first_q8_shared.failures);
        fprintf(json, ",\"gpu_lm_head_q8\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}", lm_head_q8.requests, lm_head_q8.encoded, lm_head_q8.failures,
            lm_head_q8.requests-c.first_lm_head_q8.requests, lm_head_q8.encoded-c.first_lm_head_q8.encoded,
            lm_head_q8.failures-c.first_lm_head_q8.failures);
        fprintf(json, ",\"gpu_engram_project_round\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}", engram_round.requests, engram_round.encoded, engram_round.failures,
            engram_round.requests-c.first_engram_round.requests, engram_round.encoded-c.first_engram_round.encoded,
            engram_round.failures-c.first_engram_round.failures);
        fprintf(json, ",\"gpu_moe_q2_sum_cache\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}", moe_q2_sum_cache.requests, moe_q2_sum_cache.encoded, moe_q2_sum_cache.failures,
            moe_q2_sum_cache.requests-c.first_moe_q2_sum_cache.requests, moe_q2_sum_cache.encoded-c.first_moe_q2_sum_cache.encoded,
            moe_q2_sum_cache.failures-c.first_moe_q2_sum_cache.failures);
        fprintf(json, ",\"gpu_moe_q2_sum_cache_fused\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}", moe_q2_sum_cache_fused.requests, moe_q2_sum_cache_fused.encoded, moe_q2_sum_cache_fused.failures,
            moe_q2_sum_cache_fused.requests-c.first_moe_q2_sum_cache_fused.requests, moe_q2_sum_cache_fused.encoded-c.first_moe_q2_sum_cache_fused.encoded,
            moe_q2_sum_cache_fused.failures-c.first_moe_q2_sum_cache_fused.failures);
        fprintf(json, ",\"gpu_f16_gemv_bf16_fuse\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}", f16_bf16_fuse.requests, f16_bf16_fuse.encoded, f16_bf16_fuse.failures,
            f16_bf16_fuse.requests-c.first_f16_bf16_fuse.requests, f16_bf16_fuse.encoded-c.first_f16_bf16_fuse.encoded,
            f16_bf16_fuse.failures-c.first_f16_bf16_fuse.failures);
        fprintf(json, ",\"gpu_indexer_rope_quant_fuse\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64
            ",\"failures\":%" PRIu64 ",\"decode_requests\":%" PRIu64 ",\"decode_encoded\":%" PRIu64
            ",\"decode_failures\":%" PRIu64 "}", indexer_rope_quant.requests, indexer_rope_quant.encoded, indexer_rope_quant.failures,
            indexer_rope_quant.requests-c.first_indexer_rope_quant.requests, indexer_rope_quant.encoded-c.first_indexer_rope_quant.encoded,
            indexer_rope_quant.failures-c.first_indexer_rope_quant.failures);
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
