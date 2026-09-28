/* Deterministic, small MLA + routed MoE integration benchmark.
 * This exercises production GGUF loading, tokenization, prefill and decoding.
 * Synthetic weights measure scheduling/allocator overhead, not model quality or
 * the bandwidth-limited throughput of a real DeepSeek model.
 * Link this same source against saved baseline objects for a fair A/B check.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds.h"
#include "myds_gguf.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { DIM = 256, HEADS = 4, EXPERTS = 8, ACTIVE = 6, VOCAB = 256 };
typedef struct {
    char name[96];
    ggml_type type;
    uint32_t ndims;
    uint64_t dims[3], offset;
    size_t bytes;
} tensor_t;

static void checked_write(const void *p, size_t n, FILE *f) {
    if (fwrite(p, 1, n, f) != n) { perror("write synthetic model"); exit(1); }
}
static void u32(FILE *f, uint32_t n) { checked_write(&n, sizeof(n), f); }
static void u64(FILE *f, uint64_t n) { checked_write(&n, sizeof(n), f); }
static void str(FILE *f, const char *s) { u64(f, strlen(s)); checked_write(s, strlen(s), f); }
static void kv_u32(FILE *f, const char *key, uint32_t n) {
    str(f, key); u32(f, GGUF_TYPE_UINT32); u32(f, n);
}
static uint32_t random_u32(uint32_t *state) {
    *state ^= *state << 13; *state ^= *state >> 17; *state ^= *state << 5;
    return *state;
}
static void half_store(unsigned char *dst, float value) {
    _Float16 h = (_Float16)value;
    memcpy(dst, &h, sizeof(h));
}
static void add_tensor(tensor_t *ts, size_t *count, const char *name,
                       ggml_type type, uint64_t d0, uint64_t d1, uint64_t d2) {
    tensor_t *t = &ts[(*count)++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->type = type; t->ndims = d2 ? 3 : d1 ? 2 : 1;
    t->dims[0] = d0; t->dims[1] = d1; t->dims[2] = d2;
    t->bytes = ggml_type_row_size(type, d0) * (d1 ? d1 : 1) * (d2 ? d2 : 1);
}
static void fill_tensor(const tensor_t *t, unsigned char *data, uint32_t seed) {
    uint32_t state = seed;
    if (t->type == GGML_TYPE_F32) {
        float *p = (float *)data;
        for (size_t i = 0; i < t->bytes / sizeof(float); i++) {
            float r = ((int32_t)(random_u32(&state) % 2001) - 1000) / 1000.0f;
            p[i] = strstr(t->name, "norm.weight") ? 1.0f : r * 0.125f;
            if (strstr(t->name, "attn_sinks")) p[i] = -0.5f + r * 0.1f;
        }
    } else if (t->type == GGML_TYPE_Q8_0) {
        for (size_t offset = 0; offset < t->bytes; offset += 34) {
            half_store(data + offset, 0.002f);
            for (size_t j = 2; j < 34; j++)
                data[offset + j] = (unsigned char)(int8_t)((int)(random_u32(&state) % 33) - 16);
        }
    } else if (t->type == GGML_TYPE_IQ2_XXS) {
        for (size_t offset = 0; offset < t->bytes; offset += 66) {
            half_store(data + offset, 0.001f);
            for (size_t j = 2; j < 66; j++) data[offset + j] = (unsigned char)random_u32(&state);
        }
    } else if (t->type == GGML_TYPE_Q2_K) {
        for (size_t offset = 0; offset < t->bytes; offset += 84) {
            for (size_t j = 0; j < 80; j++) data[offset + j] = (unsigned char)random_u32(&state);
            half_store(data + offset + 80, 0.001f);
            half_store(data + offset + 82, 0.001f);
        }
    }
}
static int write_model(const char *path, uint32_t layers) {
    tensor_t *ts = calloc(3 + (size_t)layers * 20, sizeof(*ts));
    if (!ts) return 0;
    size_t count = 0;
    add_tensor(ts, &count, "token_embd.weight", GGML_TYPE_F32, DIM, VOCAB, 0);
    add_tensor(ts, &count, "output_norm.weight", GGML_TYPE_F32, DIM, 0, 0);
    add_tensor(ts, &count, "output.weight", GGML_TYPE_Q8_0, DIM, VOCAB, 0);
    for (uint32_t l = 0; l < layers; l++) {
        char name[96];
#define ADD(suffix, type, d0, d1, d2) do { \
    snprintf(name, sizeof(name), "blk.%u.%s", l, suffix); \
    add_tensor(ts, &count, name, type, d0, d1, d2); \
} while (0)
        ADD("attn_norm.weight", GGML_TYPE_F32, DIM, 0, 0);
        ADD("attn_q_a.weight", GGML_TYPE_Q8_0, DIM, DIM, 0);
        ADD("attn_q_a_norm.weight", GGML_TYPE_F32, DIM, 0, 0);
        ADD("attn_q_b.weight", GGML_TYPE_Q8_0, DIM, HEADS * DIM, 0);
        ADD("attn_kv.weight", GGML_TYPE_Q8_0, DIM, DIM, 0);
        ADD("attn_kv_a_norm.weight", GGML_TYPE_F32, DIM, 0, 0);
        ADD("attn_sinks.weight", GGML_TYPE_F32, HEADS, 0, 0);
        ADD("attn_output_a.weight", GGML_TYPE_Q8_0, 2 * DIM, DIM, 0);
        ADD("attn_output_b.weight", GGML_TYPE_Q8_0, DIM, DIM, 0);
        ADD("ffn_norm.weight", GGML_TYPE_F32, DIM, 0, 0);
        ADD("ffn_gate_inp.weight", GGML_TYPE_Q8_0, DIM, EXPERTS, 0);
        ADD("exp_probs_b.bias", GGML_TYPE_F32, EXPERTS, 0, 0);
        ADD("ffn_gate_exps.weight", GGML_TYPE_IQ2_XXS, DIM, DIM, EXPERTS);
        ADD("ffn_up_exps.weight", GGML_TYPE_IQ2_XXS, DIM, DIM, EXPERTS);
        ADD("ffn_down_exps.weight", GGML_TYPE_Q2_K, DIM, DIM, EXPERTS);
        ADD("ffn_gate_shexp.weight", GGML_TYPE_Q8_0, DIM, DIM, 0);
        ADD("ffn_up_shexp.weight", GGML_TYPE_Q8_0, DIM, DIM, 0);
        ADD("ffn_down_shexp.weight", GGML_TYPE_Q8_0, DIM, DIM, 0);
#undef ADD
    }
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); free(ts); return 0; }
    u32(f, GGUF_MAGIC); u32(f, 3); u64(f, count); u64(f, 13);
    str(f, "general.architecture"); u32(f, GGUF_TYPE_STRING); str(f, "deepseek41");
    kv_u32(f, "general.alignment", 32);
    kv_u32(f, "deepseek41.embedding_length", DIM);
    kv_u32(f, "deepseek41.block_count", layers);
    kv_u32(f, "deepseek41.attention.head_count", HEADS);
    kv_u32(f, "deepseek41.attention.head_count_kv", 1);
    kv_u32(f, "deepseek41.feed_forward_length", DIM);
    kv_u32(f, "deepseek41.qk_rope_head_dim", 64);
    kv_u32(f, "deepseek41.n_routed_experts", EXPERTS);
    kv_u32(f, "deepseek41.num_experts_per_tok", ACTIVE);
    kv_u32(f, "deepseek41.n_shared_experts", 1);
    kv_u32(f, "deepseek41.moe_intermediate_size", DIM);
    kv_u32(f, "deepseek41.head_dim", DIM);
    uint64_t offset = 0;
    for (size_t i = 0; i < count; i++) {
        tensor_t *t = &ts[i]; t->offset = offset;
        str(f, t->name); u32(f, t->ndims);
        for (uint32_t d = 0; d < t->ndims; d++) u64(f, t->dims[d]);
        u32(f, t->type); u64(f, offset);
        offset += (t->bytes + 31) & ~(size_t)31;
    }
    while (ftell(f) % 32) fputc(0, f);
    for (size_t i = 0; i < count; i++) {
        size_t padded = (ts[i].bytes + 31) & ~(size_t)31;
        unsigned char *data = calloc(1, padded);
        if (!data) { fclose(f); free(ts); return 0; }
        fill_tensor(&ts[i], data, 0x13579bdu + (uint32_t)i * 0x1234567u);
        checked_write(data, padded, f); free(data);
    }
    int ok = fclose(f) == 0;
    free(ts); return ok;
}
static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
typedef struct {
    double first, last;
    uint32_t count, capacity;
    int32_t *tokens;
    uint64_t hash;
} capture_t;
static void capture(const char *piece, int32_t token, void *userdata) {
    (void)piece;
    capture_t *c = userdata;
    double now = seconds();
    if (!c->count) c->first = now;
    c->last = now;
    if (c->count < c->capacity) c->tokens[c->count] = token;
    c->count++;
    for (uint32_t b = 0; b < 4; b++) {
        c->hash ^= ((uint32_t)token >> (8 * b)) & 255u;
        c->hash *= UINT64_C(1099511628211);
    }
}
static uint32_t parse_positive(const char *s, uint32_t limit) {
    char *end = NULL; errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno || !*s || *end || v == 0 || v > limit) {
        fprintf(stderr, "Invalid positive integer: %s (maximum %u)\n", s, limit); exit(2);
    }
    return (uint32_t)v;
}
int main(int argc, char **argv) {
    uint32_t prompt_tokens = 33, decode_tokens = 32, repeats = 3, layers = 2;
    const char *output_model = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("Usage: bench_synthetic [--prompt-tokens N] [--decode-tokens N] [--repeats N] [--layers N]\n"
                 "       bench_synthetic --write-model PATH [--layers N]\n"
                 "Default: 33 prompt tokens, 32 output tokens, 3 measured runs after one warmup, 2 layers.\n"
                 "Same context is reused. TTFT includes tokenization, prefill and first decode step.\n"
                 "decode_tps counts output tokens 2..N over their wall time. Synthetic weights only.");
            return 0;
        }
        if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", argv[i]); return 2; }
        const char *key = argv[i++], *value = argv[i];
        if (!strcmp(key, "--prompt-tokens")) prompt_tokens = parse_positive(value, 16384);
        else if (!strcmp(key, "--decode-tokens")) decode_tokens = parse_positive(value, 4096);
        else if (!strcmp(key, "--repeats")) repeats = parse_positive(value, 100);
        else if (!strcmp(key, "--layers")) layers = parse_positive(value, 32);
        else if (!strcmp(key, "--write-model")) output_model = value;
        else { fprintf(stderr, "Unknown argument: %s\n", key); return 2; }
    }
    if (output_model) return write_model(output_model, layers) ? 0 : 1;
    char path[] = "/tmp/myds-synthetic-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); return 1; }
    close(fd);
    if (!write_model(path, layers)) { unlink(path); return 1; }
    myds_model_t *model = myds_model_load(path);
    unlink(path);
    if (!model) return 1;
    myds_config_t cfg = {
        .max_seq_len = prompt_tokens + decode_tokens + 1,
        .max_new_tokens = decode_tokens, .temperature = 0.0f, .top_p = 1.0f,
        .repetition_penalty = 1.0f, .seed = 42, .stream = false,
    };
    myds_context_t *ctx = myds_context_create(model, &cfg);
    char *prompt = calloc(prompt_tokens, 1);
    int32_t *tokens = calloc(decode_tokens, sizeof(*tokens));
    if (!ctx || !prompt || !tokens) {
        free(prompt); free(tokens); myds_context_free(ctx); myds_model_free(model); return 1;
    }
    const char alphabet[] = "Synthetic inference benchmark 0123456789 abcdefghijklmnopqrstuvwxyz.";
    for (uint32_t i = 0; i + 1 < prompt_tokens; i++) prompt[i] = alphabet[i % (sizeof(alphabet) - 1)];
    uint64_t expected_hash = 0;
    int status = 0;
    for (uint32_t run = 0; run <= repeats; run++) {
        capture_t c = {.capacity = decode_tokens, .tokens = tokens, .hash = UINT64_C(14695981039346656037)};
        myds_stop_reason_t reason = MYDS_STOP_ERROR;
        uint32_t actual_prompt = 0;
        double start = seconds();
        int ok = myds_generate_ex(ctx, prompt, capture, &c, &reason, &actual_prompt, NULL);
        double elapsed = seconds() - start;
        if (!ok || reason != MYDS_STOP_LENGTH || actual_prompt != prompt_tokens || c.count != decode_tokens) {
            fprintf(stderr, "Incomplete synthetic run: ok=%d reason=%d prompt=%u/%u output=%u/%u\n",
                    ok, reason, actual_prompt, prompt_tokens, c.count, decode_tokens);
            status = 1; break;
        }
        if (run == 0) expected_hash = c.hash;
        if (c.hash != expected_hash) {
            fprintf(stderr, "Repeated greedy generation diverged\n"); status = 1; break;
        }
        double decode_s = c.last - c.first;
        printf("BENCH_JSON: {\"synthetic\":true,\"run\":%u,\"warmup\":%s,\"layers\":%u,"
               "\"prompt_tokens\":%u,\"output_tokens\":%u,\"ttft_s\":%.9f,\"decode_s\":%.9f,"
               "\"decode_tps\":%.6f,\"wall_s\":%.9f,\"token_fnv1a64\":\"%016" PRIx64 "\"}\n",
               run, run == 0 ? "true" : "false", layers, actual_prompt, c.count,
               c.first - start, decode_s, decode_s > 0 ? (c.count - 1) / decode_s : 0.0,
               elapsed, c.hash);
        if (run == 0) {
            printf("TOKEN_IDS:");
            for (uint32_t i = 0; i < c.count; i++) printf(" %d", tokens[i]);
            putchar('\n');
        }
    }
    free(prompt); free(tokens); myds_context_free(ctx); myds_model_free(model);
    return status;
}
