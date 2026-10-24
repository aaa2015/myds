#define _POSIX_C_SOURCE 200809L
#include "myds.h"
#include "myds_gguf.h"
#include "myds_internal.h"
#include "myds_dspark.h"
#include "myds_snapshot_slots.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/wait.h>

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }
static void write_str(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    write_u64(f, len);
    fwrite(s, 1, len, f);
}

typedef struct {
    const char *name;
    ggml_type   type;
    uint64_t    dim0;
    uint64_t    dim1;
} synth_tensor_desc_t;

static const char *create_gate_test_gguf(
    const char *path,
    const char *arch,
    uint32_t n_layers,
    uint32_t n_routed_experts,
    const synth_tensor_desc_t *tensors,
    size_t n_tensors) {

    FILE *f = fopen(path, "wb");
    if (!f) return NULL;

    uint32_t magic = GGUF_MAGIC;
    uint32_t version = 3;
    uint64_t n_kvs = 10;

    fwrite(&magic, 4, 1, f);
    write_u32(f, version);
    write_u64(f, n_tensors);
    write_u64(f, n_kvs);

    write_str(f, "general.architecture");
    write_u32(f, GGUF_TYPE_STRING);
    write_str(f, arch);

    char key_buf[128];
    snprintf(key_buf, sizeof(key_buf), "%s.embedding_length", arch);
    write_str(f, key_buf);
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 64);

    snprintf(key_buf, sizeof(key_buf), "%s.feed_forward_length", arch);
    write_str(f, key_buf);
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 64);

    snprintf(key_buf, sizeof(key_buf), "%s.block_count", arch);
    write_str(f, key_buf);
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, n_layers);

    snprintf(key_buf, sizeof(key_buf), "%s.attention.head_count", arch);
    write_str(f, key_buf);
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 2);

    snprintf(key_buf, sizeof(key_buf), "%s.attention.head_count_kv", arch);
    write_str(f, key_buf);
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 2);

    snprintf(key_buf, sizeof(key_buf), "%s.expert_count", arch);
    write_str(f, key_buf);
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, n_routed_experts);

    write_str(f, "general.alignment");
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 32);

    snprintf(key_buf, sizeof(key_buf), "%s.rms_norm_eps", arch);
    write_str(f, key_buf);
    write_u32(f, GGUF_TYPE_FLOAT32);
    float reference_epsilon = 1e-20f;
    fwrite(&reference_epsilon, sizeof(reference_epsilon), 1, f);

    /* Match the fixture's 16 embedding/output rows explicitly. Falling back
     * to a synthetic 256-token vocabulary would hide a loader contract bug. */
    write_str(f, "tokenizer.ggml.tokens");
    write_u32(f, GGUF_TYPE_ARRAY);
    write_u32(f, GGUF_TYPE_STRING);
    write_u64(f, 16);
    for (unsigned i = 0; i < 16; i++) {
        char token[] = {(char)('a' + i), 0};
        write_str(f, token);
    }

    uint64_t cur_offset = 0;
    for (size_t i = 0; i < n_tensors; i++) {
        write_str(f, tensors[i].name);
        uint32_t n_dims = (tensors[i].dim1 > 0) ? 2 : 1;
        write_u32(f, n_dims);
        write_u64(f, tensors[i].dim0);
        if (n_dims == 2) write_u64(f, tensors[i].dim1);
        write_u32(f, (uint32_t)tensors[i].type);
        write_u64(f, cur_offset);

        size_t n_el = tensors[i].dim0 * (tensors[i].dim1 > 0 ? tensors[i].dim1 : 1);
        size_t sz = ggml_type_row_size(tensors[i].type, n_el);
        sz = (sz + 31) / 32 * 32;
        cur_offset += sz;
    }

    long pos = ftell(f);
    long pad = (32 - (pos % 32)) % 32;
    for (long i = 0; i < pad; i++) fputc(0, f);

    for (size_t i = 0; i < n_tensors; i++) {
        size_t n_el = tensors[i].dim0 * (tensors[i].dim1 > 0 ? tensors[i].dim1 : 1);
        size_t sz = ggml_type_row_size(tensors[i].type, n_el);
        size_t sz_padded = (sz + 31) / 32 * 32;
        uint8_t *dummy = (uint8_t *)calloc(1, sz_padded);
        if (tensors[i].type == GGML_TYPE_F32) {
            float *fvals = (float *)dummy;
            for (size_t k = 0; k < n_el; k++) {
                fvals[k] = 0.05f * (float)((k % 7) + 1);
            }
        }
        fwrite(dummy, 1, sz_padded, f);
        free(dummy);
    }

    fclose(f);
    return path;
}

/* Test RoPE frequency formula */
static float calculate_rope_freq(uint32_t dim_idx, uint32_t total_dim, float freq_base) {
    float exponent = (float)(2 * dim_idx) / (float)total_dim;
    return 1.0f / powf(freq_base, exponent);
}

/* Small nonzero MLA model with a real routed-expert path and shared FFN.
 * Construct weights directly so lifecycle tests are independent of model files. */
static myds_metal_buffer_t *test_f32_buffer(size_t n, bool norm, uint32_t salt) {
    myds_metal_buffer_t *buf = myds_metal_buffer_alloc(n * sizeof(float));
    assert(buf != NULL);
    float *v = (float *)myds_metal_buffer_data(buf);
    for (size_t i = 0; i < n; i++) {
        v[i] = norm ? 1.0f : (float)((int)((i * 17 + salt) % 31) - 15) * 0.002f;
    }
    return buf;
}

static myds_metal_buffer_t *test_zero_experts(ggml_type type, size_t n) {
    size_t bytes = ggml_type_row_size(type, n);
    myds_metal_buffer_t *buf = myds_metal_buffer_alloc(bytes);
    assert(buf != NULL);
    memset(myds_metal_buffer_data(buf), 0, bytes);
    return buf;
}

static myds_model_t *test_prefill_model(void) {
    myds_model_t *m = (myds_model_t *)calloc(1, sizeof(*m));
    assert(m != NULL);
    m->dim = m->hidden_dim = m->moe_expert_dim = 256;
    m->vocab_size = 256;
    m->n_layers = 1;
    m->n_heads = m->n_out_groups = 2;
    m->n_kv_heads = 1;
    m->head_dim = m->kv_lora_rank = m->q_lora_rank = 32;
    m->out_group_rank = 16;
    m->qk_rope_head_dim = 16;
    m->n_routed_experts = 8;
    m->n_activated_experts = 6;
    m->n_shared_experts = 1;
    m->norm_eps = 1e-6f;
    m->freq_base = 10000.0f;
    m->freq_scale = m->routed_scaling_factor = 1.0f;
    m->swiglu_clamp_exp = 10.0f;
    m->is_mla = true;
    gguf_file_t empty = {0};
    m->tokenizer = myds_tokenizer_load(&empty);
    assert(m->tokenizer != NULL);
    m->tok_embeddings_type = GGML_TYPE_F32;
    m->tok_embeddings = test_f32_buffer((size_t)m->vocab_size * m->dim, false, 5);
    m->output_norm = test_f32_buffer(m->dim, true, 0);
    m->output = test_f32_buffer((size_t)m->vocab_size * m->dim, false, 3);
    m->output_type = GGML_TYPE_F32;
    m->layers = (myds_layer_t *)calloc(1, sizeof(*m->layers));
    assert(m->layers != NULL);
    myds_layer_t *l = m->layers;
    l->attn_norm = test_f32_buffer(m->dim, true, 0);
    l->attn_q_a = test_f32_buffer((size_t)m->dim * m->q_lora_rank, false, 7);
    l->attn_q_a_norm = test_f32_buffer(m->q_lora_rank, true, 0);
    l->attn_q_b = test_f32_buffer((size_t)m->q_lora_rank * m->n_heads * m->head_dim, false, 11);
    l->attn_kv = test_f32_buffer((size_t)m->dim * m->head_dim, false, 13);
    l->attn_kv_a_norm = test_f32_buffer(m->head_dim, true, 0);
    l->attn_sinks = test_f32_buffer(m->n_heads, false, 17);
    l->attn_output_a = test_f32_buffer((size_t)m->head_dim * m->n_out_groups * m->out_group_rank, false, 19);
    l->attn_output_b = test_f32_buffer((size_t)m->n_out_groups * m->out_group_rank * m->dim, false, 23);
    l->ffn_norm = test_f32_buffer(m->dim, true, 0);
    l->router_gate = test_f32_buffer((size_t)m->dim * m->n_routed_experts, false, 29);
    l->ffn_gate_exps_type = l->ffn_up_exps_type = GGML_TYPE_IQ2_XXS;
    l->ffn_down_exps_type = GGML_TYPE_Q2_K;
    size_t n_expert_weights = (size_t)m->n_routed_experts * m->dim * m->moe_expert_dim;
    l->ffn_gate_exps = test_zero_experts(GGML_TYPE_IQ2_XXS, n_expert_weights);
    l->ffn_up_exps = test_zero_experts(GGML_TYPE_IQ2_XXS, n_expert_weights);
    l->ffn_down_exps = test_zero_experts(GGML_TYPE_Q2_K, n_expert_weights);
    l->shared_gate = test_f32_buffer((size_t)m->dim * m->moe_expert_dim, false, 31);
    l->shared_up = test_f32_buffer((size_t)m->dim * m->moe_expert_dim, false, 37);
    l->shared_down = test_f32_buffer((size_t)m->dim * m->moe_expert_dim, false, 41);
    return m;
}

static void assert_prefill_matches_sequential(myds_context_t *batched, myds_context_t *reference, const char *prompt) {
    int32_t tokens[1024];
    int32_t n = myds_tokenizer_encode(reference->model->tokenizer, prompt, tokens, 1024, true);
    assert(n > 0);
    for (int32_t i = 0; i < n; i++) {
        assert(myds_forward_token(reference, tokens[i], (uint32_t)i) >= 0);
    }
    uint32_t n_prompt = 0;
    myds_stop_reason_t reason = MYDS_STOP_ERROR;
    assert(myds_generate_ex(batched, prompt, NULL, NULL, &reason, &n_prompt, NULL));
    assert(reason == MYDS_STOP_LENGTH || reason == MYDS_STOP_EOS);
    assert(n_prompt == (uint32_t)n && batched->pos == (uint32_t)n);
    const float *a = (const float *)myds_metal_buffer_data(batched->logits);
    const float *b = (const float *)myds_metal_buffer_data(reference->logits);
    for (uint32_t i = 0; i < batched->model->vocab_size; i++) {
        assert(isfinite(a[i]) && isfinite(b[i]));
        assert(fabsf(a[i] - b[i]) < 1e-5f);
    }
    for (uint32_t l = 0; l < batched->model->n_layers; l++) {
        for (size_t i = 0; i < (size_t)n * batched->model->head_dim; i++) {
            assert(fabsf(batched->k_cache[l][i] - reference->k_cache[l][i]) < 1e-5f);
        }
    }
}

/* ★ 统计泄露回归（2026-10-25）。
 *
 * 场景：先跑一次【成功】的请求，把 last_* 四个字段写成真实值；然后一个在分词
 * 阶段就失败的请求（这里用空 prompt，与「客户端断连/参数非法」同一条退出路径）。
 * 旧实现不区分「这是不是本请求写的」，于是服务端会把上一个请求的 token 数当成
 * 这个请求的报出去 —— 日志里出现「prompt: 0 tok, prefill: FAILED after 15815 tok」。
 *
 * 断言两件事：失败请求读到的统计必须是全零且 valid=0；成功请求必须 valid=1。
 */
static void test_prefill_stats_do_not_leak(void) {
    myds_model_t *model = test_prefill_model();
    strcpy(model->arch, "deepseek4");
    myds_config_t cfg = {.max_seq_len = 1024, .max_new_tokens = 1, .temperature = 0,
                         .top_p = 1, .seed = 42};
    myds_context_t *ctx = myds_context_create(model, &cfg);
    assert(ctx);

    myds_stop_reason_t reason = MYDS_STOP_ERROR;
    uint32_t prompt_count = 0;
    assert(myds_generate_bytes_ex(ctx, "hello world", NULL, NULL, &reason, &prompt_count, NULL));
    myds_prefill_stats_t good;
    myds_last_prefill_stats(ctx, &good);
    assert(good.valid == 1);
    assert(good.prompt_tokens > 0);

    /* 超长 prompt：超过 max_seq_len，在 prefill 之前就返回 MYDS_STOP_CTX_FULL。
     * （空 prompt 不行：分词后仍剩 BOS，n_prompt=1 > 0，会真的跑完 prefill。） */
    static char huge[8192];
    huge[0] = 0;
    for (size_t i = 0; i < 4000u; i += 4u) memcpy(huge + i, "abc ", 4u);
    huge[4000] = 0;
    reason = MYDS_STOP_ERROR;
    prompt_count = 0xdeadbeefu;
    (void)myds_generate_bytes_ex(ctx, huge, NULL, NULL, &reason, &prompt_count, NULL);
    assert(reason == MYDS_STOP_CTX_FULL);
    myds_prefill_stats_t bad;
    myds_last_prefill_stats(ctx, &bad);
    assert(bad.valid == 0);
    assert(bad.prompt_tokens == 0 && bad.cached_tokens == 0 &&
           bad.prefill_tokens == 0 && bad.prefill_seconds == 0.0);

    /* ★ 关键用例：服务端在 cancel 短路时【根本不会调 generate】。
     * 这时只能靠显式作废，否则日志仍会借用上一个请求的数。 */
    assert(myds_generate_bytes_ex(ctx, "hello world", NULL, NULL, &reason, &prompt_count, NULL));
    myds_prefill_stats_t before;
    myds_last_prefill_stats(ctx, &before);
    assert(before.valid == 1 && before.prompt_tokens > 0);
    myds_context_reset_prefill_stats(ctx);
    myds_prefill_stats_t after_reset;
    myds_last_prefill_stats(ctx, &after_reset);
    assert(after_reset.valid == 0 && after_reset.prompt_tokens == 0 &&
           after_reset.prefill_tokens == 0 && after_reset.cached_tokens == 0);
    /* idempotent + NULL-safe */
    myds_context_reset_prefill_stats(ctx);
    myds_context_reset_prefill_stats(NULL);

    /* 再跑一次成功请求：必须重新变成有效值，不能被上面的清零卡死。 */
    reason = MYDS_STOP_ERROR;
    assert(myds_generate_bytes_ex(ctx, "hello world", NULL, NULL, &reason, &prompt_count, NULL));
    myds_prefill_stats_t again;
    myds_last_prefill_stats(ctx, &again);
    assert(again.valid == 1 && again.prompt_tokens > 0);

    /* ctx 为 NULL 时也必须安全清零。 */
    myds_prefill_stats_t none;
    memset(&none, 0xff, sizeof(none));
    myds_last_prefill_stats(NULL, &none);
    assert(none.valid == 0 && none.prompt_tokens == 0);

    myds_context_free(ctx);
    myds_model_free(model); myds_metal_cleanup();
    puts("  [PASS] prefill stats: failure path reports zeros with valid=0, never the previous request");
}

typedef struct { char divergence[4096]; } prefix_log_capture_t;

static void capture_prefix_log(myds_log_level_t level, const char *message, void *opaque) {
    (void)level;
    prefix_log_capture_t *capture = opaque;
    if (strstr(message, "prefix cache divergence"))
        snprintf(capture->divergence, sizeof(capture->divergence), "%s", message);
}

static uint32_t prefix_prompt_tokens(myds_model_t *model, const char *prompt, int32_t *tokens) {
    const int32_t n = myds_tokenizer_encode(model->tokenizer, prompt, tokens, 1024, true);
    assert(n > 1 && n < 1024);
    return (uint32_t)n;
}

static void assert_prefix_generate(myds_context_t *ctx, const char *prompt, uint32_t cached) {
    int32_t tokens[1024];
    const uint32_t n = prefix_prompt_tokens(ctx->model, prompt, tokens);
    myds_stop_reason_t reason = MYDS_STOP_ERROR;
    uint32_t prompt_count = 0;
    assert(myds_generate_bytes_ex(ctx, prompt, NULL, NULL, &reason, &prompt_count, NULL));
    assert(reason == MYDS_STOP_LENGTH || reason == MYDS_STOP_EOS);
    myds_prefill_stats_t stats;
    myds_last_prefill_stats(ctx, &stats);
    assert(stats.valid == 1);
    assert(prompt_count == n && stats.prompt_tokens == n && stats.cached_tokens == cached);
    assert(cached <= n - 1u && stats.prefill_tokens == n - 1u - cached);
    assert(ctx->pos == n && ctx->pcache_len == n);
    assert(memcmp(ctx->pcache_tokens, tokens, n * sizeof(*tokens)) == 0);
    /* A legitimate success path must also match a fresh full-prefill context. */
    myds_context_t *cold = myds_context_create(ctx->model, &ctx->cfg);
    assert(cold);
    assert(myds_generate_bytes_ex(cold, prompt, NULL, NULL, &reason, NULL, NULL));
    const float *a = myds_metal_buffer_data(ctx->logits), *b = myds_metal_buffer_data(cold->logits);
    for (uint32_t i = 0; i < ctx->model->vocab_size; ++i)
        assert(isfinite(a[i]) && isfinite(b[i]) && fabsf(a[i] - b[i]) < 1e-5f);
    myds_context_free(cold);
}

static void test_prefix_generate_boundaries(void) {
    assert(setenv("MYDS_PREFIX_CACHE", "1", 1) == 0);
    assert(unsetenv("MYDS_NO_PREFIX_SNAPSHOT") == 0);
    assert(myds_metal_init("metal"));
    myds_model_t *model = test_prefill_model();
    strcpy(model->arch, "deepseek4");
    myds_config_t cfg = {.max_seq_len = 1024, .max_new_tokens = 1, .temperature = 0,
                         .top_p = 1, .seed = 42};
    myds_context_t *ctx = myds_context_create(model, &cfg);
    assert(ctx);
    int32_t tokens[1024];
    const uint32_t short_n = prefix_prompt_tokens(model, "abc", tokens);

    /* Exercise real snapshot selection/restore through generate, then cancel
     * before token execution. This small snapshot fixture intentionally has
     * no V4.1 model weights. The success cases below run the real tiny model. */
    ctx->v41 = myds_v41_context_create_snapshot_fixture();
    assert(ctx->v41 && myds_v41_snapshot_get_snap_window(ctx->v41, 0));
    assert(myds_v41_snapshot_get_snap_window(ctx->v41, 1));
    ctx->pcache_tokens = malloc(1024 * sizeof(*ctx->pcache_tokens));
    assert(ctx->pcache_tokens); ctx->pcache_cap = 1024;
    for (unsigned variant = 0; variant < 4; ++variant) {
        memcpy(ctx->pcache_tokens, tokens, short_n * sizeof(*tokens));
        ctx->pcache_len = variant == 0 ? short_n : short_n + 4u;
        for (uint32_t i = short_n; i < ctx->pcache_len; ++i)
            ctx->pcache_tokens[i] = variant == 2 ? INT32_MAX : tokens[i % short_n];
        myds_v41_snapshot_fixture_publish(ctx->v41, 0, short_n);
        myds_v41_snapshot_fixture_publish(ctx->v41, 1, variant == 3 ? 0 : short_n - 1u);
        volatile int cancel = 1;
        myds_stop_reason_t reason = MYDS_STOP_ERROR;
        assert(myds_generate_bytes_ex(ctx, "abc", NULL, NULL, &reason, NULL, &cancel));
        assert(reason == MYDS_STOP_CANCELLED);
        myds_prefill_stats_t stats; myds_last_prefill_stats(ctx, &stats);
        assert(stats.valid == 1);
        const uint32_t wanted = variant == 3 ? 0 : short_n - 1u;
        assert(stats.cached_tokens == wanted && stats.prefill_tokens == short_n - 1u - wanted);
        assert(stats.prompt_tokens == short_n && stats.cached_tokens < short_n);
        /* The last prompt token has not reached decode. Cancellation may
         * retain only the prefix actually restored/consumed, not that token. */
        assert(ctx->pcache_len == ctx->pos);
        if (ctx->pcache_len)
            assert(myds_v41_snapshot_get_next_position(ctx->v41) == ctx->pos);
    }
    myds_v41_context_free(ctx->v41); ctx->v41 = NULL;
    ctx->pcache_len = 0;
    prefix_log_capture_t capture = {0};
    myds_log_set_level(MYDS_LOG_INFO);
    myds_log_set_callback(capture_prefix_log, &capture);
    assert_prefix_generate(ctx, "abcdefghij", 0);
    assert_prefix_generate(ctx, "abc", 0); /* Strictly shorter prefix. */
    char end_marker[64]; snprintf(end_marker, sizeof(end_marker), "p[%u]=<end>", short_n);
    assert(strstr(capture.divergence, end_marker));
    assert_prefix_generate(ctx, "abc", 0); /* Equal-length prompt. */
    assert_prefix_generate(ctx, "abcde", short_n); /* Legitimate continuation. */
    for (unsigned poison = 0; poison < 2; ++poison) {
        assert_prefix_generate(ctx, "abcdefghij", ctx->pcache_len);
        for (uint32_t i = short_n; i < ctx->pcache_len; ++i)
            ctx->pcache_tokens[i] = poison ? INT32_MAX : ctx->pcache_tokens[i % short_n];
        memset(&capture, 0, sizeof(capture));
        assert_prefix_generate(ctx, "abc", 0);
        assert(strstr(capture.divergence, end_marker));
    }
    /* Long diagnostic token text must truncate inside the fixed why buffer,
     * including its final token/end-marker append calls. */
    assert_prefix_generate(ctx, "abcdefghij", ctx->pcache_len);
    char *saved_decoded = model->tokenizer->vocab[(unsigned char)'a'].decoded;
    char *long_decoded = malloc(1401); assert(long_decoded);
    memset(long_decoded, '\n', 1400); long_decoded[1400] = 0;
    model->tokenizer->vocab[(unsigned char)'a'].decoded = long_decoded;
    memset(&capture, 0, sizeof(capture));
    assert_prefix_generate(ctx, "abc", 0);
    assert(strstr(capture.divergence, "prefix cache divergence at") && strlen(capture.divergence) < 700);
    model->tokenizer->vocab[(unsigned char)'a'].decoded = saved_decoded;
    free(long_decoded);
    myds_log_set_callback(NULL, NULL);
    myds_context_free(ctx); myds_model_free(model); myds_metal_cleanup();
    puts("  [PASS] Production generate prefix bounds: short/equal/snapshot-end, matching/poisoned tails and successful continuation with cold-logit parity");
}

static void test_empty_decode_continuation(void) {
    assert(setenv("MYDS_PREFIX_CACHE", "1", 1) == 0);
    assert(setenv("MYDS_PLD", "0", 1) == 0);
    assert(myds_metal_init("metal"));
    myds_model_t *model = test_prefill_model();
    strcpy(model->arch, "deepseek4");
    model->tokenizer->eos_id = -1;
    myds_config_t cfg = {.max_seq_len = 64, .max_new_tokens = 0,
                         .temperature = 0, .top_p = 1, .seed = 42};
    myds_context_t *ctx = myds_context_create(model, &cfg);
    assert(ctx && !ctx->v41);
    int32_t initial_tokens[1024];
    const uint32_t initial_n = prefix_prompt_tokens(model, "abcdefghij", initial_tokens);
    myds_stop_reason_t reason = MYDS_STOP_ERROR;
    assert(myds_generate_bytes_ex(ctx, "abcdefghij", NULL, NULL, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_LENGTH && ctx->pos == initial_n - 1u);
    myds_decode_stats_t decode; myds_last_decode_stats(ctx, &decode);
    assert(decode.output_tokens == 0 && decode.target_forward_calls == 0);
    fprintf(stderr, "EMPTY_DECODE_CACHE consumed=%u cached=%u expected=%u\n",
            ctx->pos, ctx->pcache_len, initial_n - 1u);
    assert(ctx->pcache_len == initial_n - 1u);
    assert(!memcmp(ctx->pcache_tokens, initial_tokens, ctx->pcache_len * sizeof(*initial_tokens)));

    /* A longer full request on the same context must replay the pending last
     * prompt token. Compare its actual cache and logits with a cold request. */
    cfg.max_new_tokens = 1;
    myds_context_set_config(ctx, &cfg);
    assert_prefix_generate(ctx, "abcdefghijkl", initial_n - 1u);
    myds_context_t *cold = myds_context_create(model, &cfg);
    assert(cold);
    assert(myds_generate_bytes_ex(cold, "abcdefghijkl", NULL, NULL, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_LENGTH && cold->pos == ctx->pos && cold->pcache_len == ctx->pcache_len);
    assert(!memcmp(cold->pcache_tokens, ctx->pcache_tokens, ctx->pcache_len * sizeof(*ctx->pcache_tokens)));
    const size_t cache_elements = (size_t)ctx->pos * model->head_dim;
    for (uint32_t l = 0; l < model->n_layers; ++l) {
        bool nonzero = false;
        for (size_t i = 0; i < cache_elements; ++i) {
            const float a = ctx->k_cache[l][i], b = cold->k_cache[l][i];
            assert(isfinite(a) && isfinite(b) && fabsf(a - b) < 1e-5f);
            if (a != 0.0f) nonzero = true;
        }
        assert(nonzero);
    }
    const float *a = myds_metal_buffer_data(ctx->logits), *b = myds_metal_buffer_data(cold->logits);
    for (uint32_t i = 0; i < model->vocab_size; ++i)
        assert(isfinite(a[i]) && isfinite(b[i]) && fabsf(a[i] - b[i]) < 1e-5f);
    myds_context_free(cold); myds_context_free(ctx); myds_model_free(model); myds_metal_cleanup();
    puts("  [PASS] Zero-output real tiny-model request retains only consumed prefix; same-context continuation KV/state/logits match cold replay");
}

typedef struct {
    uint32_t count;
    int32_t tokens[64];
    volatile int *cancel;
    myds_context_t *slow_spec;
} decode_test_capture_t;

static void capture_decode_test(const char *bytes, size_t length, int32_t token, void *opaque) {
    (void)bytes; (void)length;
    decode_test_capture_t *capture = opaque;
    assert(capture->count < 64);
    capture->tokens[capture->count++] = token;
    /* Controlled callback cost exercises the actual cycle-based scheduler,
     * rather than assigning its counters by hand. */
    if (capture->slow_spec && capture->slow_spec->decode_cycle_speculative) usleep(10000);
    if (capture->cancel) *capture->cancel = 1;
}

static void test_decode_request_metrics(void) {
    assert(setenv("MYDS_PREFIX_CACHE", "0", 1) == 0);
    assert(setenv("MYDS_PLD", "0", 1) == 0);
    assert(setenv("MYDS_DECODE_METRICS", "1", 1) == 0);
    assert(unsetenv("MYDS_SPEC_COST_SCHEDULER") == 0);
    assert(myds_metal_init("metal"));
    myds_model_t *model = test_prefill_model();
    strcpy(model->arch, "deepseek4");
    model->tokenizer->eos_id = -1;
    myds_config_t cfg = {.max_seq_len = 64, .max_new_tokens = 8, .temperature = 0,
                         .top_p = 1, .repetition_penalty = 1, .seed = 42};
    myds_context_t *ctx = myds_context_create(model, &cfg);
    assert(ctx);
    decode_test_capture_t timed = {0}, untimed = {0};
    myds_stop_reason_t reason;
    assert(myds_generate_bytes_ex(ctx, "abcd", capture_decode_test, &timed, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_LENGTH && timed.count == 8);
    myds_decode_stats_t stats;
    myds_last_decode_stats(ctx, &stats);
    assert(stats.timing_enabled && stats.output_tokens == timed.count);
    assert(stats.target_forward_calls == timed.count && stats.target_rows == timed.count);
    assert(stats.cycle_count == timed.count && stats.recorded_cycles == timed.count);
    assert(stats.batch2_calls == 0 && stats.draft_cycles == 0 && stats.pld_queries == 0);
    for (uint32_t i = 0; i < stats.recorded_cycles; ++i) {
        myds_decode_cycle_t cycle;
        assert(myds_last_decode_cycle(ctx, i, &cycle));
        assert(isfinite(cycle.elapsed_ms) && cycle.elapsed_ms > 0.0 && cycle.output_tokens == 1);
    }
    myds_decode_cycle_t missing;
    assert(!myds_last_decode_cycle(ctx, stats.recorded_cycles, &missing));
    assert(missing.elapsed_ms == 0 && missing.output_tokens == 0);
    assert(setenv("MYDS_DECODE_METRICS", "0", 1) == 0);
    assert(myds_generate_bytes_ex(ctx, "abcd", capture_decode_test, &untimed, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_LENGTH && untimed.count == timed.count);
    assert(memcmp(timed.tokens, untimed.tokens, timed.count * sizeof(int32_t)) == 0);
    myds_last_decode_stats(ctx, &stats);
    assert(!stats.timing_enabled && stats.output_tokens == 8 && stats.target_forward_calls == 8);
    assert(stats.recorded_cycles == 0 && !myds_last_decode_cycle(ctx, 0, &missing));
    /* Real cancellation after one callback retains exactly one completed step. */
    volatile int cancel = 0;
    decode_test_capture_t cancelled = {.cancel = &cancel};
    assert(myds_generate_bytes_ex(ctx, "abcd", capture_decode_test, &cancelled, &reason, NULL, &cancel));
    assert(reason == MYDS_STOP_CANCELLED && cancelled.count == 1);
    myds_last_decode_stats(ctx, &stats);
    assert(stats.output_tokens == 1 && stats.target_forward_calls == 1 && stats.cycle_count == 1);
    /* EOS consumes a target step but is never reported as an output fragment. */
    model->tokenizer->eos_id = timed.tokens[0];
    decode_test_capture_t eos = {0};
    assert(myds_generate_bytes_ex(ctx, "abcd", capture_decode_test, &eos, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_EOS && eos.count == 0);
    myds_last_decode_stats(ctx, &stats);
    assert(stats.output_tokens == 0 && stats.target_forward_calls == 1 && stats.cycle_count == 1);
    model->tokenizer->eos_id = -1;
    memset(myds_metal_buffer_data(model->output), 0,
           (size_t)model->vocab_size * model->dim * sizeof(float));
    cfg.max_new_tokens = 24;
    myds_context_set_config(ctx, &cfg);
    assert(setenv("MYDS_PLD", "1", 1) == 0);
    assert(setenv("MYDS_SPEC_COST_SCHEDULER", "1", 1) == 0);
    decode_test_capture_t budgeted = {.slow_spec = ctx};
    assert(myds_generate_bytes_ex(ctx, "abcd", capture_decode_test, &budgeted, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_LENGTH && budgeted.count == 24);
    myds_last_decode_stats(ctx, &stats);
    assert(stats.timing_enabled && stats.recorded_cycles == 0);
    assert(stats.target_forward_calls == budgeted.count && stats.target_rows == budgeted.count);
    assert(stats.pld_queries > 0 && stats.pld_accepted > 0 && stats.pld_accepted <= stats.pld_proposed);
    assert(stats.backoff_steps >= 16 && stats.output_tokens == budgeted.count);
    for (uint32_t i = 0; i < budgeted.count; ++i) assert(budgeted.tokens[i] == 0);
    assert(!myds_last_decode_cycle(ctx, 0, &missing));
    /* A drafter without neural weights exercises PLD A fallback after request
     * initialization clears stale scheduler state and candidate availability. */
    assert(setenv("MYDS_SPEC_COST_SCHEDULER", "0", 1) == 0);
    ctx->dspark = calloc(1, sizeof(*ctx->dspark));
    assert(ctx->dspark);
    ctx->dspark->block_size = 1;
    ctx->dspark->backoff_counter = 100;
    ctx->dspark->draft_tokens[0] = 0;
    decode_test_capture_t pld_fallback = {0};
    assert(myds_generate_bytes_ex(ctx, "abcd", capture_decode_test, &pld_fallback, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_LENGTH && pld_fallback.count == 24);
    myds_last_decode_stats(ctx, &stats);
    assert(stats.draft_proposed == 0 && stats.draft_accepted == 0);
    assert(stats.pld_accepted > 0 && stats.pld_accepted <= stats.pld_proposed);
    assert(ctx->dspark->accepted_tokens == 0 && ctx->dspark->consecutive_misses == 0);
    assert(ctx->dspark->backoff_counter == 0);
    for (uint32_t i = 0; i < pld_fallback.count; ++i) assert(pld_fallback.tokens[i] == 0);
    assert(!myds_generate_bytes_ex(ctx, NULL, capture_decode_test, &budgeted, &reason, NULL, NULL));
    assert(reason == MYDS_STOP_ERROR);
    myds_last_decode_stats(ctx, &stats);
    assert(stats.output_tokens == 0 && stats.target_forward_calls == 0 && stats.cycle_count == 0);
    myds_last_decode_stats(NULL, &stats);
    assert(stats.target_forward_calls == 0 && stats.output_tokens == 0);
    assert(!myds_last_decode_cycle(NULL, 0, &missing));
    myds_context_free(ctx); myds_model_free(model); myds_metal_cleanup();
    puts("  [PASS] Real tiny-model decode counters, EOS/reset/cancel/token parity and cycle-cost backoff");
}

static void test_prefill_workspace_reuse(const char *arch) {
    myds_model_t *m = test_prefill_model();
    snprintf(m->arch, sizeof(m->arch), "%s", arch);
    myds_config_t cfg = {.max_seq_len = 1024, .max_new_tokens = 1, .temperature = 0.0f,
                         .top_p = 1.0f, .seed = 42};
    myds_context_t *ctx = myds_context_create(m, &cfg);
    myds_context_t *reference = myds_context_create(m, &cfg);
    assert(ctx && reference && ctx->prefill_chunk == NULL && ctx->prefill_capacity == 0);
    assert_prefill_matches_sequential(ctx, reference, "abcde");
    assert(ctx->prefill_capacity == 5);
    myds_prefill_chunk_t *first = ctx->prefill_chunk;
    assert_prefill_matches_sequential(ctx, reference, "abcde");
    assert(ctx->prefill_chunk == first && ctx->prefill_capacity == 5);
    assert_prefill_matches_sequential(ctx, reference, "abcdefghijklmnopq");
    assert(ctx->prefill_capacity == 17 && ctx->prefill_chunk != first);
    myds_prefill_chunk_t *larger = ctx->prefill_chunk;
    assert_prefill_matches_sequential(ctx, reference, "abcde");
    assert(ctx->prefill_chunk == larger && ctx->prefill_capacity == 17);

    volatile int cancel = 1;
    myds_stop_reason_t reason = MYDS_STOP_ERROR;
    assert(myds_generate_ex(ctx, "abcdefghijklmnopqrstuv", NULL, NULL, &reason, NULL, &cancel));
    assert(reason == MYDS_STOP_CANCELLED && ctx->pos == 0);
    assert(ctx->prefill_chunk == larger && ctx->prefill_capacity == 17);
    assert_prefill_matches_sequential(ctx, reference, "abcde");

    char long_prompt[514];
    memset(long_prompt, 'a', sizeof(long_prompt) - 1);
    long_prompt[sizeof(long_prompt) - 1] = '\0';
    assert_prefill_matches_sequential(ctx, reference, long_prompt);
    assert(ctx->prefill_capacity == 512);

    cfg.max_seq_len = 16;
    myds_context_set_config(ctx, &cfg);
    assert(myds_context_get_config(ctx).max_seq_len == 16);
    cfg.max_seq_len = 2048;
    myds_context_set_config(ctx, &cfg);
    assert(myds_context_get_config(ctx).max_seq_len == 1024);
    cfg.max_seq_len = 0;
    myds_context_set_config(ctx, &cfg);
    assert(myds_context_get_config(ctx).max_seq_len == 1024);
    myds_context_free(reference);
    myds_context_free(ctx);
    myds_model_free(m);
    printf("  [PASS] Prefill short/long/short workspace reuse, 512-token boundary, output equivalence, cancellation and KV capacity\n");
}

static void test_f16_embedding(void) {
    /* Literal binary16 inputs and exact mathematical values, deliberately
     * independent of the production converter. Normal, subnormal and signed
     * values catch BF16 interpretation even if prefill/decode share a bug. */
    static const uint16_t half[] = {
        0x0000, 0x8000, 0x3c00, 0xbc00, 0x3800, 0xb800, 0x4000, 0xc000,
        0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3555, 0xb555
    };
    static const float values[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
        0x1p-24f, -0x1p-24f, 1023.0f * 0x1p-24f, -1023.0f * 0x1p-24f,
        0x1p-14f, -0x1p-14f, 1365.0f / 4096.0f, -1365.0f / 4096.0f
    };
    myds_model_t *f16 = test_prefill_model();
    myds_model_t *f32 = test_prefill_model();
    size_t elements = (size_t)f16->vocab_size * f16->dim;
    myds_metal_buffer_free(f16->tok_embeddings);
    f16->tok_embeddings = myds_metal_buffer_alloc(elements * sizeof(uint16_t));
    f16->tok_embeddings_type = GGML_TYPE_F16;
    assert(f16->tok_embeddings != NULL);
    uint16_t *half_rows = myds_metal_buffer_data(f16->tok_embeddings);
    float *float_rows = myds_metal_buffer_data(f32->tok_embeddings);
    for (uint32_t t = 0; t < f16->vocab_size; t++) {
        for (uint32_t i = 0; i < f16->dim; i++) {
            size_t index = (size_t)t * f16->dim + i;
            unsigned pattern = (t * 7 + i) % 16;
            half_rows[index] = half[pattern];
            float_rows[index] = values[pattern];
        }
    }
    myds_config_t cfg = {.max_seq_len = 64, .max_new_tokens = 1, .temperature = 0.0f,
                         .top_p = 1.0f, .seed = 42};
    myds_context_t *decode = myds_context_create(f16, &cfg);
    myds_context_t *batched = myds_context_create(f16, &cfg);
    myds_context_t *reference = myds_context_create(f32, &cfg);
    assert(decode && batched && reference);

    /* No transformer layers: x exposes the embedding values directly. */
    f16->n_layers = 0;
    assert(myds_forward_token(decode, 37, 0) >= 0);
    assert(memcmp(myds_metal_buffer_data(decode->x), float_rows + 37 * f16->dim,
                  f16->dim * sizeof(float)) == 0);
    assert(myds_forward_token(decode, -1, 0) == -1);
    assert(myds_forward_token(decode, (int32_t)f16->vocab_size, 0) == -1);
    f16->n_layers = 1;

    /* Five prefill tokens plus BOS force the chunk path; two tokens exercise
     * sequential prefill. Both compare complete logits and KV rows against
     * a separate F32 model populated with the values above. */
    assert_prefill_matches_sequential(batched, reference, "abcde");
    assert(batched->prefill_chunk != NULL);
    assert_prefill_matches_sequential(decode, reference, "ab");
    assert(decode->prefill_chunk == NULL);

    myds_context_free(reference);
    myds_context_free(batched);
    myds_context_free(decode);
    myds_model_free(f32);
    myds_model_free(f16);
    printf("  [PASS] F16 embedding known values, bounds, decode and chunk prefill against independent F32 weights\n");
}

static void test_model_vocabulary_shapes(void) {
    const char *path = "/tmp/test_myds_vocabulary_shapes.gguf";
    const synth_tensor_desc_t valid[] = {
        {"token_embd.weight", GGML_TYPE_F32, 64, 16},
        {"output_norm.weight", GGML_TYPE_F32, 64, 0},
        {"output.weight", GGML_TYPE_F32, 64, 16},
        {"blk.0.attn_norm.weight", GGML_TYPE_F32, 64, 0},
        {"blk.0.attn_q.weight", GGML_TYPE_F32, 64, 64},
        {"blk.0.attn_k.weight", GGML_TYPE_F32, 64, 64},
        {"blk.0.attn_v.weight", GGML_TYPE_F32, 64, 64},
        {"blk.0.attn_output.weight", GGML_TYPE_F32, 64, 64},
        {"blk.0.ffn_norm.weight", GGML_TYPE_F32, 64, 0},
        {"blk.0.ffn_gate.weight", GGML_TYPE_F32, 64, 64},
        {"blk.0.ffn_up.weight", GGML_TYPE_F32, 64, 64},
        {"blk.0.ffn_down.weight", GGML_TYPE_F32, 64, 64},
    };
    const size_t tensor_count = sizeof(valid) / sizeof(valid[0]);
    assert(create_gate_test_gguf(path, "deepseek41", 1, 0, valid, tensor_count));
    myds_model_t *m = myds_model_load(path);
    assert(m && m->vocab_size == 16 && m->tokenizer->vocab_size == 16);
    myds_model_free(m);
    const struct { unsigned tensor; uint64_t dim0, dim1; ggml_type type; } invalid[] = {
        {0, 64, 15, GGML_TYPE_F32}, /* Embedding vocabulary mismatch. */
        {0, 32, 16, GGML_TYPE_F32}, /* Embedding width mismatch. */
        {0, 64, 0, GGML_TYPE_F32},  /* Embedding is not a matrix. */
        {2, 64, 15, GGML_TYPE_F32}, /* Output vocabulary mismatch. */
        {2, 32, 16, GGML_TYPE_F32}, /* Output width mismatch. */
        {2, 64, 0, GGML_TYPE_F32},  /* Output is not a matrix. */
        {1, 32, 0, GGML_TYPE_F32},  /* Norm width mismatch. */
        {1, 64, 1, GGML_TYPE_F32},  /* Norm is not a vector. */
        {1, 64, 0, GGML_TYPE_F16},  /* Norm kernel expects F32. */
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        synth_tensor_desc_t tensors[sizeof(valid) / sizeof(valid[0])];
        memcpy(tensors, valid, sizeof(tensors));
        tensors[invalid[i].tensor].dim0 = invalid[i].dim0;
        tensors[invalid[i].tensor].dim1 = invalid[i].dim1;
        tensors[invalid[i].tensor].type = invalid[i].type;
        assert(create_gate_test_gguf(path, "deepseek41", 1, 0, tensors, tensor_count));
        assert(myds_model_load(path) == NULL);
    }
    unlink(path);
    printf("  [PASS] Tokenizer/embedding/output vocabulary, dimensions and norm type/shape loader contracts\n");
}

static void count_byte_callback(const char *bytes, size_t length, int32_t token_id, void *opaque) {
    (void)bytes; (void)length; (void)token_id;
    (*(unsigned *)opaque)++;
}

static void test_invalid_generated_token(void) {
    /* Deliberately corrupt an otherwise valid in-memory model after loading:
     * sampling 200 is possible, but tokenizer vocabulary has only 128 rows.
     * decode_bytes() returns an empty piece for invalid IDs, so generation
     * must check the ID rather than relying on a NULL decoder return. */
    myds_model_t *m = test_prefill_model();
    m->n_layers = 0;
    m->tokenizer->vocab_size = 128;
    float *embeddings = myds_metal_buffer_data(m->tok_embeddings);
    for (size_t i = 0; i < (size_t)m->dim * m->vocab_size; i++) embeddings[i] = 1.0f;
    float *output = myds_metal_buffer_data(m->output);
    memset(output, 0, (size_t)m->dim * m->vocab_size * sizeof(float));
    for (uint32_t i = 0; i < m->dim; i++) output[200 * m->dim + i] = 1.0f;
    myds_config_t cfg = {.max_seq_len = 64, .max_new_tokens = 1, .temperature = 0.0f,
                         .top_p = 1.0f, .seed = 42};
    myds_context_t *ctx = myds_context_create(m, &cfg);
    assert(ctx);
    assert(myds_forward_token(ctx, 97, 0) == 200);
    unsigned callback_count = 0;
    myds_stop_reason_t reason = MYDS_STOP_LENGTH;
    assert(!myds_generate_bytes_ex(ctx, "a", count_byte_callback, &callback_count,
                                   &reason, NULL, NULL));
    assert(reason == MYDS_STOP_ERROR && callback_count == 0);
    myds_context_free(ctx);
    m->tokenizer->vocab_size = 256;
    m->n_layers = 1;
    myds_model_free(m);
    printf("  [PASS] Invalid sampled token returns generation error without an empty success callback\n");
}


/* ── prefix cache 的诊断输出测试 ──
 * 动机：这个缓存不生效时原本一行日志都不打，于是「为什么这轮没省下来」只能读代码猜。
 * 这里断言【五种结局每一种都说话】—— 尤其「跳过」的两种：它们是当前设计里最贵的 miss，
 * 也正是实际踩到的那一种（客户端把每次提问当成全新对话、只带系统提示词）。 */

typedef enum {
    PCACHE_OFF = 0,      /* 功能被关闭 */
    PCACHE_EMPTY,        /* 还没缓存任何东西（首个请求） */
    PCACHE_TOO_SHORT,    /* 新提示词不长于缓存序列 ⇒ 不可能是续写 */
    PCACHE_PARTIAL,      /* 有公共前缀但不足整段 */
    PCACHE_RESUME        /* 命中：整段严格续写 */
} pcache_verdict_t;

/* 判定逻辑（与 myds_engine.c 内联的那份同构），抽出来以便在无模型的情况下测试 */
static pcache_verdict_t pcache_decide(int on, uint32_t cached_len, uint32_t n_prompt,
                                      const int32_t *cached, const int32_t *prompt,
                                      uint32_t *out_common) {
    if (out_common) *out_common = 0;
    if (!on) return PCACHE_OFF;
    if (!cached_len) return PCACHE_EMPTY;
    if (n_prompt <= cached_len) return PCACHE_TOO_SHORT;
    uint32_t L = 0;
    while (L < cached_len && cached[L] == prompt[L]) L++;
    if (out_common) *out_common = L;
    return (L == cached_len) ? PCACHE_RESUME : PCACHE_PARTIAL;
}

/* 对应的日志行（不返回空串 —— 沉默就是这次要根治的东西） */
static void pcache_line(pcache_verdict_t v, uint32_t cached_len, uint32_t n_prompt,
                        uint32_t common, char *buf, size_t cap) {
    switch (v) {
    case PCACHE_OFF:
        snprintf(buf, cap, "prefix cache: off (MYDS_PREFIX_CACHE=0)");
        break;
    case PCACHE_EMPTY:
        snprintf(buf, cap, "prefix cache: nothing cached yet (this is the first request)");
        break;
    case PCACHE_TOO_SHORT:
        snprintf(buf, cap, "prefix cache: skipped - prompt %u tok is not longer than the cached %u",
                 n_prompt, cached_len);
        break;
    case PCACHE_PARTIAL:
        snprintf(buf, cap, "prefix cache: skipped - common prefix %u of cached %u tokens", common,
                 cached_len);
        break;
    default:
        snprintf(buf, cap, "prefix cache: prompt %u tok, cached %u, common prefix %u -> resume",
                 n_prompt, cached_len, common);
        break;
    }
}

static void test_prefix_cache_reports_every_outcome(void) {
    int32_t cached[16];
    for (int i = 0; i < 16; ++i) cached[i] = 100 + i;

    int32_t longer[20];                       /* 严格续写 */
    for (int i = 0; i < 16; ++i) longer[i] = 100 + i;
    longer[16] = 999; longer[17] = 998; longer[18] = 997; longer[19] = 996;

    int32_t shortp[8];                        /* 只带系统提示词的新对话 */
    for (int i = 0; i < 8; ++i) shortp[i] = 100 + i;

    int32_t diverge[20];                      /* 第 5 个 token 起分叉 */
    for (int i = 0; i < 20; ++i) diverge[i] = 100 + i;
    diverge[5] = 7;

    char buf[256];
    uint32_t common = 0;

    assert(pcache_decide(0, 16, 20, cached, longer, &common) == PCACHE_OFF);
    pcache_line(PCACHE_OFF, 16, 20, 0, buf, sizeof buf);
    assert(strstr(buf, "prefix cache:") == buf && strstr(buf, "off") != NULL);

    assert(pcache_decide(1, 0, 20, cached, longer, &common) == PCACHE_EMPTY);
    pcache_line(PCACHE_EMPTY, 0, 20, 0, buf, sizeof buf);
    assert(strstr(buf, "first request") != NULL);

    /* ★ 最贵的那种 miss：新提示词比缓存还短 */
    assert(pcache_decide(1, 16, 8, cached, shortp, &common) == PCACHE_TOO_SHORT);
    pcache_line(PCACHE_TOO_SHORT, 16, 8, 0, buf, sizeof buf);
    assert(strstr(buf, "not longer than") != NULL);

    assert(pcache_decide(1, 16, 20, cached, diverge, &common) == PCACHE_PARTIAL && common == 5);
    pcache_line(PCACHE_PARTIAL, 16, 20, common, buf, sizeof buf);
    assert(strstr(buf, "common prefix 5") != NULL);

    assert(pcache_decide(1, 16, 20, cached, longer, &common) == PCACHE_RESUME && common == 16);
    pcache_line(PCACHE_RESUME, 16, 20, common, buf, sizeof buf);
    assert(strstr(buf, "resume") != NULL);

    /* 每一种结局都必须产出【非空且带前缀】的一行 */
    for (int i = PCACHE_OFF; i <= PCACHE_RESUME; ++i) {
        pcache_line((pcache_verdict_t)i, 16, 20, 5, buf, sizeof buf);
        assert(buf[0] != 0 && strstr(buf, "prefix cache:") == buf);
    }
    printf("  [PASS] prefix cache reports every outcome, including the two skipped cases\n");
}

/* ── 前缀快照的槽位分配 ──
 * 判据是【常量前缀必须永远有槽可用】：
 * 一个会话走得很深之后，位置 1024/2048/... 的槽不能被远处的点顶掉，
 * 否则新会话（短提示词）就只能全量重算 —— 实测出现过这种退化，且没有任何报错。 */
static void test_snapshot_slots_keep_the_constant_prefix(void) {
    /* ① 槽 0 专供序列末尾 */
    assert(myds_snap_slot_for(999999u, 1) == 0u);
    assert(myds_snap_slot_for(1024u, 1) == 0u);

    /* ② 低位区：位置与槽【固定一一对应】，与后续走了多深无关 */
    const uint32_t first = MYDS_SNAP_MIN / MYDS_SNAP_STRIDE;
    (void)first;
    for (uint32_t k = 0; k < MYDS_SNAP_PINNED; ++k) {
        const uint32_t p = MYDS_SNAP_MIN + k * MYDS_SNAP_STRIDE;
        const uint32_t slot = myds_snap_slot_for(p, 0);
        assert(slot == 1u + k);
        assert(myds_snap_slot_is_pinned(slot));
    }

    /* ③ 【关键性质】一个很深的点不会映射到任何低位槽 ——
     *    这正是原来「按位置取模」退化的根源（位置 9216 会顶掉 2048）。 */
    const uint32_t deepest_pinned = MYDS_SNAP_MIN + (MYDS_SNAP_PINNED - 1u) * MYDS_SNAP_STRIDE;
    for (uint32_t p = deepest_pinned + MYDS_SNAP_STRIDE; p <= 200000u; p += MYDS_SNAP_STRIDE) {
        assert(!myds_snap_slot_is_pinned(myds_snap_slot_for(p, 0)));
    }

    /* ④ 环形区：只在环内复用，且始终落在合法槽上 */
    for (uint32_t p = 0; p <= 400000u; p += MYDS_SNAP_STRIDE) {
        const uint32_t s = myds_snap_slot_for(p, 0);
        assert(s < MYDS_SNAP_SLOTS);
    }
    /* 环形区确实会复用（说明它是环，不是无限增长） */
    const uint32_t r1 = myds_snap_slot_for(deepest_pinned + MYDS_SNAP_STRIDE, 0);
    const uint32_t r2 = myds_snap_slot_for(
        deepest_pinned + MYDS_SNAP_STRIDE * (MYDS_SNAP_SLOTS - 1u - MYDS_SNAP_PINNED + 1u), 0);
    assert(r1 == r2);

    printf("  [PASS] snapshot slots keep the constant prefix pinned (%u slots, %u pinned up to pos %u)\n",
           (unsigned)MYDS_SNAP_SLOTS, (unsigned)MYDS_SNAP_PINNED, (unsigned)deepest_pinned);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--decode-metrics") == 0) {
        test_decode_request_metrics();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prefix-cache") == 0) {
        test_prefix_generate_boundaries();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--empty-decode") == 0) {
        test_empty_decode_continuation();
        return 0;
    }
    /* Prefix enablement is cached process-wide by production generate. Run
     * its enabled-path regression in a fresh process before other GPU work. */
    assert(argc == 1);
    const pid_t child = fork();
    assert(child >= 0);
    if (!child) { execl(argv[0], argv[0], "--prefix-cache", (char *)NULL); _exit(127); }
    int child_status;
    assert(waitpid(child, &child_status, 0) == child);
    assert(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);
    const pid_t empty_child = fork();
    assert(empty_child >= 0);
    if (!empty_child) { execl(argv[0], argv[0], "--empty-decode", (char *)NULL); _exit(127); }
    assert(waitpid(empty_child, &child_status, 0) == empty_child);
    assert(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);
    const pid_t metrics_child = fork();
    assert(metrics_child >= 0);
    if (!metrics_child) { execl(argv[0], argv[0], "--decode-metrics", (char *)NULL); _exit(127); }
    assert(waitpid(metrics_child, &child_status, 0) == metrics_child);
    assert(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);
    printf("====================================================\n");
    printf("       Running myds Engine Sampling Unit Tests      \n");
    printf("====================================================\n");

    test_prefix_cache_reports_every_outcome();
    test_snapshot_slots_keep_the_constant_prefix();
    test_prefill_stats_do_not_leak();

    /* 1. Production Greedy Argmax Sampling */
    {
        float logits[8] = {-2.5f, 0.1f, 10.5f, 3.2f, -1.0f, 8.9f, 2.0f, -0.5f};
        myds_config_t cfg = {.temperature = 0.0f, .top_p = 1.0f, .repetition_penalty = 1.0f};
        int32_t picked = myds_sample_next_token(logits, 8, &cfg, NULL, NULL, 0);
        assert(picked == 2); /* 10.5f is maximum at index 2 */
        printf("  [PASS] Production greedy argmax token selection (index 2 picked)\n");

        /* Structured output bitmask filter: disallow token 2, allow tokens 1, 3, 5.
         *
         * 2026-10-25：bitmask 改为 ctx 独占所有权（myds_config_t 里的字段已删，
         * 那会造成 cfg/ctx 双重所有权 → clear 后悬空 → 下一次请求 use-after-free）。
         * 因此这里必须给真实 ctx，不能再传 NULL。 */
        myds_context_t cx;
        memset(&cx, 0, sizeof cx);
        uint32_t bm[1] = {(1u << 1) | (1u << 3) | (1u << 5)};
        cx.logit_bitmask = bm;
        cx.logit_bitmask_words = 1;

        myds_config_t cfg_bm = {.temperature = 0.0f};
        int32_t picked_bm = myds_sample_next_token(logits, 8, &cfg_bm, &cx, NULL, 0);
        assert(picked_bm == 5); /* Token 5 (8.9f) is highest among allowed {1, 3, 5} */

        /* temperature 路径同样必须遵守 bitmask */
        cfg_bm.temperature = 0.8f;
        cfg_bm.seed = 42;
        int32_t picked_temp = myds_sample_next_token(logits, 8, &cfg_bm, &cx, NULL, 0);
        assert(picked_temp == 1 || picked_temp == 3 || picked_temp == 5);

        /* ★ 全禁时的边界：不得崩溃，返回值必须落在词表内 */
        uint32_t allzero[1] = {0u};
        cx.logit_bitmask = allzero;
        const int32_t none_ok = myds_sample_next_token(logits, 8, &cfg_bm, &cx, NULL, 0);
        assert(none_ok >= 0 && none_ok < 8);

        cx.logit_bitmask = NULL;
        cx.logit_bitmask_words = 0;
        printf("  [PASS] Structured output logit bitmask filter (disallowed token 2, selected %d)\n", picked_bm);
    }

    /* 2. Production Repetition Penalty */
    {
        float logits[8] = {-2.5f, 0.1f, 10.5f, 3.2f, -1.0f, 8.9f, 2.0f, -0.5f};
        /* Penalize token 2 with rep_pen = 2.0 -> 10.5 / 2.0 = 5.25, so token 5 (8.9f) becomes argmax */
        int32_t prev[1] = {2};
        myds_config_t cfg = {.temperature = 0.0f, .top_p = 1.0f, .repetition_penalty = 2.0f};
        int32_t picked = myds_sample_next_token(logits, 8, &cfg, NULL, prev, 1);
        (void)picked;

        /* 2026-10-24：presence_penalty 语义
         *   token 在近期上下文出现过 ⇒ 该 token 的 logit 减 pres_pen，与出现次数无关。
         * 构造：token 2 的 logit 比 token 5 高 0.5；pres_pen=1.0 应把 token 2 压到
         * token 5 之下 ⇒ 选中 5。pres_pen=0 时应选中 2。 */
        {
            float l2[8] = {0};
            l2[2] = 2.0f; l2[5] = 1.5f;
            myds_config_t c0 = cfg;
            c0.temperature = 0.0f;          /* greedy，结果确定 */
            c0.repetition_penalty = 1.0f;
            c0.presence_penalty = 0.0f;
            int32_t prev2[1] = {2};
            assert(myds_sample_next_token(l2, 8, &c0, NULL, prev2, 1) == 2);

            c0.presence_penalty = 1.0f;      /* 2.0 - 1.0 = 1.0 < 1.5 */
            assert(myds_sample_next_token(l2, 8, &c0, NULL, prev2, 1) == 5);

            c0.presence_penalty = 0.2f;      /* 2.0 - 0.2 = 1.8 > 1.5 */
            assert(myds_sample_next_token(l2, 8, &c0, NULL, prev2, 1) == 2);

            /* 负值鼓励重复：2.0 + 1.0 = 3.0 */
            c0.presence_penalty = -1.0f;
            assert(myds_sample_next_token(l2, 8, &c0, NULL, prev2, 1) == 2);

            /* 与出现次数无关：token 2 出现 3 次，仍只减一次 */
            int32_t prev3[3] = {2, 2, 2};
            c0.presence_penalty = 1.0f;
            assert(myds_sample_next_token(l2, 8, &c0, NULL, prev3, 3) == 5);
        }

        /* ★ 生产路径：两处调用点都传 (NULL, 0)，历史来自 ctx 的环形缓冲。
         * 上面几条走的是显式 prev_tokens，不覆盖这条分支 —— 而它才是线上真正跑的。 */
        {
            myds_context_t c;
            memset(&c, 0, sizeof c);
            c.cfg = cfg;
            c.cfg.temperature = 0.0f;
            c.cfg.repetition_penalty = 1.0f;
            c.cfg.presence_penalty = 0.0f;

            float l3[8] = {0};
            l3[2] = 2.0f; l3[5] = 1.5f;

            /* 缓冲为空 ⇒ 无惩罚，选 2 */
            assert(myds_sample_next_token(l3, 8, &c.cfg, &c, NULL, 0) == 2);

            /* 记录 token 2 之后，pres_pen=1.0 应把 2.0 压到 1.0 ⇒ 选 5 */
            myds_note_recent_token(&c, 2);
            assert(c.recent_count == 1u);
            c.cfg.presence_penalty = 1.0f;
            assert(myds_sample_next_token(l3, 8, &c.cfg, &c, NULL, 0) == 5);

            /* 环形缓冲：连写 300 个（容量 256）不应越界，计数封顶在容量 */
            for (int i = 0; i < 300; ++i) myds_note_recent_token(&c, 7);
            const uint32_t cap = (uint32_t)(sizeof(c.recent_tokens)/sizeof(c.recent_tokens[0]));
            assert(c.recent_count == cap);
            assert(c.recent_next < cap);

            /* 负 token（无效）不得写入 */
            const uint32_t before = c.recent_count;
            myds_note_recent_token(&c, -1);
            assert(c.recent_count == before);
        }
        assert(picked == 5);
        printf("  [PASS] Production repetition penalty: token 2 penalized -> token 5 selected\n");
    }

    /* 3. Production Top-K Truncation */
    {
        float logits[6] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        /* With top_k = 2, only indices 4 and 5 are eligible */
        myds_config_t cfg = {.temperature = 1.0f, .top_k = 2, .top_p = 1.0f, .repetition_penalty = 1.0f, .seed = 123};
        for (int iter = 0; iter < 20; iter++) {
            int32_t picked = myds_sample_next_token(logits, 6, &cfg, NULL, NULL, 0);
            assert(picked == 4 || picked == 5);
        }
        printf("  [PASS] Production top-k truncation constraint (only top 2 candidate tokens sampled)\n");
    }

    /* 3b. Production Top-P (nucleus) must span the whole mass prefix.
     *     构造一个「尾部也占大量质量」的分布，让真 nucleus 远大于任何固定截断：
     *       token 0..1023   logit 0       （合计 8.2% 质量，且前 1024 名【无歧义】）
     *       token 1024..    logit -0.5    （合计 91.8% 质量 ⇒ 尾部才是大头）
     *       top_p = 0.95 ⇒ 真 nucleus 约 1.9 万个（覆盖到尾部）
     *     旧实现先砍到 1024 个候选再归一化 ⇒ 只可能采到 token < 1024，尾部永远抽不到；
     *     新实现应让约 91% 的样本落在 1024 号之后（期望约 183/200）。 */
    {
        const uint32_t V = 20000;
        float *logits = (float *)malloc(V * sizeof(float));
        assert(logits != NULL);
        for (uint32_t i = 0; i < V; i++) logits[i] = (i < 1024u) ? 0.0f : -0.5f;
        myds_config_t cfg = {.temperature = 1.0f, .top_k = 0, .top_p = 0.95f,
                             .repetition_penalty = 1.0f, .seed = 7};
        uint32_t beyond = 0;
        for (int iter = 0; iter < 200; iter++) {
            int32_t picked = myds_sample_next_token(logits, V, &cfg, NULL, NULL, 0);
            assert(picked >= 0 && (uint32_t)picked < V);
            if (picked >= 1024) beyond++;
        }
        /* 期望约 183/200。旧实现恒为 0，所以 >128 这个门槛宽松但决定性。 */
        assert(beyond > 128);
        free(logits);
        printf("  [PASS] Production top-p nucleus spans the mass prefix (%u/200 samples beyond the top 1024)\n", beyond);
    }

    /* 4. Production Min-P Filtering */
    {
        /* Logits: token 0 dominant, token 1 tiny */
        float logits[3] = {10.0f, 0.0f, -5.0f};
        myds_config_t cfg = {.temperature = 1.0f, .min_p = 0.1f, .top_p = 1.0f, .seed = 42};
        int32_t picked = myds_sample_next_token(logits, 3, &cfg, NULL, NULL, 0);
        assert(picked == 0);
        printf("  [PASS] Production min-p probability threshold filtering\n");
    }

    /* 5. Decoupled RoPE Frequency Calculation */
    {
        float f0 = calculate_rope_freq(0, 64, 10000.0f);
        assert(fabsf(f0 - 1.0f) < 1e-5f); /* 1.0 / (10000 ^ 0) == 1.0 */

        float f_last = calculate_rope_freq(32, 64, 10000.0f);
        assert(fabsf(f_last - 0.0001f) < 1e-6f); /* 1.0 / (10000 ^ 1) == 0.0001 */
        printf("  [PASS] RoPE decoupled angular frequency calculation\n");
    }

    /* A family label does not establish support for the model's graph. */
    {
        const char *path = "/tmp/test_myds_unsupported_hc.gguf";
        synth_tensor_desc_t required[] = {
            {"blk.0.hc_attn_fn.weight", GGML_TYPE_F16, 64, 24}
        };
        assert(create_gate_test_gguf(path, "deepseek41", 1, 0, required, 1));
        assert(myds_model_load(path) == NULL);
        unlink(path);
        printf("  [PASS] Required unimplemented HC graph is rejected before inference\n");
    }

    /* 6. DeepSeek V4/V4.1 family recognition & V2/V3 Rejection */
    {
        /* DeepSeek V4 and V4.1 accepted architectures */
        assert(myds_is_supported_arch("deepseek41") == true);
        assert(myds_is_supported_arch("deepseek4") == true);
        assert(myds_is_supported_arch("deepseek4-dspark") == true);
        assert(myds_is_supported_arch("deepseek-v4") == true);
        assert(myds_is_supported_arch("deepseek-v4.1") == true);

        /* DeepSeek V2, V3, and legacy architectures explicitly rejected */
        assert(myds_is_supported_arch("deepseek2") == false);
        assert(myds_is_supported_arch("deepseek3") == false);
        assert(myds_is_supported_arch("deepseek") == false);
        assert(myds_is_supported_arch("deepseek-v2") == false);
        assert(myds_is_supported_arch("deepseek-v3") == false);

        /* Non-DeepSeek architectures rejected */
        assert(myds_is_supported_arch("llama") == false);
        assert(myds_is_supported_arch("qwen2") == false);
        assert(myds_is_supported_arch("") == false);
        assert(myds_is_supported_arch(NULL) == false);

        printf("  [PASS] DeepSeek V4/V4.1 architecture enforcement & V2/V3 rejection\n");
    }

    /* 6b. DeepSeek V4.1 FP8 tile block size defense (32x32 required, 128x128 rejected) */
    {
        gguf_file_t gf;
        memset(&gf, 0, sizeof(gf));
        assert(myds_validate_fp8_block_size(&gf, "deepseek41") == true);

        gguf_kv_t kv;
        memset(&kv, 0, sizeof(kv));
        gf.kvs = &kv;
        gf.metadata_kv_count = 1;

        uint32_t b32[2] = {32, 32};
        uint32_t b128[2] = {128, 128};

        char key_wbs[] = "deepseek41.weight_block_size";
        kv.key = key_wbs;
        kv.type = GGUF_TYPE_ARRAY;
        kv.value.val_arr.elem_type = GGUF_TYPE_UINT32;
        kv.value.val_arr.count = 2;
        kv.value.val_arr.data = b32;
        assert(myds_validate_fp8_block_size(&gf, "deepseek41") == true);

        kv.value.val_arr.data = b128;
        assert(myds_validate_fp8_block_size(&gf, "deepseek41") == false);

        char key_qc[] = "quantization_config.weight_block_size";
        kv.key = key_qc;
        assert(myds_validate_fp8_block_size(&gf, "deepseek-v4.1") == false);
        kv.value.val_arr.data = b32;
        assert(myds_validate_fp8_block_size(&gf, "deepseek-v4.1") == true);

        /* Scalar test */
        kv.type = GGUF_TYPE_UINT32;
        kv.value.val_u32 = 128;
        assert(myds_validate_fp8_block_size(&gf, "deepseek41") == false);
        kv.value.val_u32 = 32;
        assert(myds_validate_fp8_block_size(&gf, "deepseek41") == true);

        printf("  [PASS] DeepSeek V4.1 FP8 tile block size 32x32 defense validation (128x128 rejected)\n");
    }

    /* 7. Gate T0-3: Multi-Head Latent Attention (MLA) without MHA fail-fast rejection */
    {
        const char *path = "/tmp/test_myds_gate_mla.gguf";
        synth_tensor_desc_t tensors[] = {
            { "token_embd.weight", GGML_TYPE_F32, 64, 16 },
            { "output_norm.weight", GGML_TYPE_F32, 64, 0 },
            { "blk.0.attn_q_a.weight", GGML_TYPE_F32, 64, 16 }
        };
        create_gate_test_gguf(path, "deepseek41", 1, 0, tensors, sizeof(tensors)/sizeof(tensors[0]));

        myds_model_t *m = myds_model_load(path);
        assert(m == NULL); /* Must be rejected by T0-3 MLA gate */
        unlink(path);
        printf("  [PASS] T0-3 gate: MLA model without MHA tensors rejected fail-fast\n");
    }

    /* 8. Gate T0-4: MoE with routed experts fail-fast rejection */
    {
        const char *path = "/tmp/test_myds_gate_moe.gguf";
        synth_tensor_desc_t tensors[] = {
            { "token_embd.weight", GGML_TYPE_F32, 64, 16 },
            { "output_norm.weight", GGML_TYPE_F32, 64, 0 },
            { "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q8_0, 64, 64 }
        };
        create_gate_test_gguf(path, "deepseek41", 1, 64, tensors, sizeof(tensors)/sizeof(tensors[0]));

        myds_model_t *m = myds_model_load(path);
        assert(m == NULL); /* Must be rejected by T0-4 MoE gate */
        unlink(path);
        printf("  [PASS] T0-4 gate: MoE model with routed experts rejected fail-fast\n");
    }

    /* 9. Gate T0-1: Unsupported quantization type fail-fast rejection */
    {
        const char *path = "/tmp/test_myds_gate_quant.gguf";
        synth_tensor_desc_t tensors[] = {
            { "token_embd.weight", GGML_TYPE_F32, 64, 16 },
            { "output_norm.weight", GGML_TYPE_F32, 64, 0 },
            { "output.weight", GGML_TYPE_F32, 64, 16 },
            { "blk.0.attn_norm.weight", GGML_TYPE_F32, 64, 0 },
            { "blk.0.attn_q.weight", GGML_TYPE_Q4_K, 64, 64 } /* Q4_K is unsupported for GEMV */
        };
        create_gate_test_gguf(path, "deepseek41", 1, 0, tensors, sizeof(tensors)/sizeof(tensors[0]));

        myds_model_t *m = myds_model_load(path);
        assert(m == NULL); /* Must be rejected by T0-1 unsupported quant gate */
        unlink(path);
        printf("  [PASS] T0-1 gate: Unsupported quantization (Q4_K) rejected fail-fast\n");
    }

    /* 10. Gate T0-3: Missing required attention tensors fail-fast rejection */
    {
        const char *path = "/tmp/test_myds_gate_missing.gguf";
        synth_tensor_desc_t tensors[] = {
            { "token_embd.weight", GGML_TYPE_F32, 64, 16 },
            { "output_norm.weight", GGML_TYPE_F32, 64, 0 }
            /* blk.0 attention tensors completely missing */
        };
        create_gate_test_gguf(path, "deepseek41", 1, 0, tensors, sizeof(tensors)/sizeof(tensors[0]));

        myds_model_t *m = myds_model_load(path);
        assert(m == NULL); /* Must be rejected by T0-3 missing tensor gate */
        unlink(path);
        printf("  [PASS] T0-3 gate: Missing required attention/FFN tensors rejected fail-fast\n");
    }

    /* 11. MLA End-to-End: Full Multi-Head Latent Attention forward execution & latent KV cache verification */
    {
        const char *path = "/tmp/test_myds_mla_forward.gguf";
        synth_tensor_desc_t tensors[] = {
            { "token_embd.weight",          GGML_TYPE_F32, 64, 16 },
            { "output_norm.weight",         GGML_TYPE_F32, 64, 0 },
            { "output.weight",              GGML_TYPE_F32, 64, 16 },
            { "blk.0.attn_norm.weight",      GGML_TYPE_F32, 64, 0 },
            { "blk.0.attn_q_a.weight",       GGML_TYPE_F32, 64, 16 },
            { "blk.0.attn_q_a_norm.weight",  GGML_TYPE_F32, 16, 0 },
            { "blk.0.attn_q_b.weight",       GGML_TYPE_F32, 16, 32 }, /* 2 heads * head_dim 16 = 32 */
            { "blk.0.attn_kv.weight",        GGML_TYPE_F32, 64, 16 }, /* head_dim 16 */
            { "blk.0.attn_kv_a_norm.weight", GGML_TYPE_F32, 16, 0 },
            { "blk.0.attn_sinks.weight",     GGML_TYPE_F32, 2, 0 },   /* 2 heads */
            { "blk.0.attn_output_a.weight",  GGML_TYPE_F32, 16, 16 }, /* 2 groups of rank 8, group_dim 16 */
            { "blk.0.attn_output_b.weight",  GGML_TYPE_F32, 16, 64 }, /* 16 -> 64 */
            { "blk.0.ffn_norm.weight",       GGML_TYPE_F32, 64, 0 },
            { "blk.0.ffn_gate.weight",       GGML_TYPE_F32, 64, 64 },
            { "blk.0.ffn_up.weight",         GGML_TYPE_F32, 64, 64 },
            { "blk.0.ffn_down.weight",       GGML_TYPE_F32, 64, 64 },
        };
        create_gate_test_gguf(path, "deepseek41", 1, 0, tensors, sizeof(tensors)/sizeof(tensors[0]));

        myds_model_t *m = myds_model_load(path);
        assert(m != NULL);
        assert(m->is_mla == true);
        assert(m->norm_eps == 1e-20f);
        assert(m->dim == 64);
        assert(m->head_dim == 16);
        assert(m->q_lora_rank == 16);
        assert(m->kv_lora_rank == 16);
        assert(m->n_heads == 2);

        myds_config_t cfg = {
            .max_seq_len = 32,
            .max_new_tokens = 4,
            .temperature = 0.0f, /* greedy */
            .top_p = 1.0f,
            .seed = 42,
            .stream = false
        };
        myds_context_t *ctx = myds_context_create(m, &cfg);
        assert(ctx != NULL);
        assert(ctx->k_cache != NULL);
        assert(ctx->v_cache == NULL); /* In MLA, v_cache is not needed */

        /* Forward token at pos 0 */
        int32_t next_t0 = myds_forward_token(ctx, 1, 0);
        assert(next_t0 >= 0 && next_t0 < 16);

        /* Verify latent KV cache is populated at pos 0 */
        bool kv0_nonzero = false;
        for (uint32_t i = 0; i < m->head_dim; i++) {
            if (ctx->k_cache[0][i] != 0.0f) kv0_nonzero = true;
        }
        assert(kv0_nonzero);

        /* Forward token at pos 1 (cross-attention over pos 0) */
        int32_t next_t1 = myds_forward_token(ctx, next_t0, 1);
        assert(next_t1 >= 0 && next_t1 < 16);

        /* Verify latent KV cache is populated at pos 1 */
        bool kv1_nonzero = false;
        for (uint32_t i = 0; i < m->head_dim; i++) {
            if (ctx->k_cache[0][m->head_dim + i] != 0.0f) kv1_nonzero = true;
        }
        assert(kv1_nonzero);

        /* Dense MLA prompts longer than four prefill tokens must use the
         * sequential path, since chunk FFN currently requires routed MoE. */
        cfg.max_new_tokens = 1;
        myds_context_set_config(ctx, &cfg);
        myds_stop_reason_t reason = MYDS_STOP_ERROR;
        assert(myds_generate_ex(ctx, "defgh", NULL, NULL, &reason, NULL, NULL));
        assert(reason == MYDS_STOP_LENGTH || reason == MYDS_STOP_EOS);
        assert(ctx->pos == 6 && ctx->prefill_chunk == NULL);

        myds_context_free(ctx);
        myds_model_free(m);
        unlink(path);
        printf("  [PASS] MLA forward pass & latent KV cache cross-attention verified\n");
    }

    test_prefill_workspace_reuse("deepseek4");
    /* Exercise the scalar/chunk Q-no-head-norm branch as a scheduling
     * regression as well; this synthetic model is not a V4.1 quality oracle. */
    test_prefill_workspace_reuse("deepseek41");
    test_f16_embedding();
    test_model_vocabulary_shapes();
    test_invalid_generated_token();

    printf("====================================================\n");
    printf("   ALL ENGINE UNIT & GATE TESTS PASSED!             \n");
    printf("====================================================\n");
    return 0;
}
