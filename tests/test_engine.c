#include "myds.h"
#include "myds_gguf.h"
#include "myds_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

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
    uint64_t n_kvs = 8;

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

static void test_prefill_workspace_reuse(void) {
    myds_model_t *m = test_prefill_model();
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

int main(void) {
    printf("====================================================\n");
    printf("       Running myds Engine Sampling Unit Tests      \n");
    printf("====================================================\n");

    /* 1. Production Greedy Argmax Sampling */
    {
        float logits[8] = {-2.5f, 0.1f, 10.5f, 3.2f, -1.0f, 8.9f, 2.0f, -0.5f};
        myds_config_t cfg = {.temperature = 0.0f, .top_p = 1.0f, .repetition_penalty = 1.0f};
        int32_t picked = myds_sample_next_token(logits, 8, &cfg, NULL, NULL, 0);
        assert(picked == 2); /* 10.5f is maximum at index 2 */
        printf("  [PASS] Production greedy argmax token selection (index 2 picked)\n");
    }

    /* 2. Production Repetition Penalty */
    {
        float logits[8] = {-2.5f, 0.1f, 10.5f, 3.2f, -1.0f, 8.9f, 2.0f, -0.5f};
        /* Penalize token 2 with rep_pen = 2.0 -> 10.5 / 2.0 = 5.25, so token 5 (8.9f) becomes argmax */
        int32_t prev[1] = {2};
        myds_config_t cfg = {.temperature = 0.0f, .top_p = 1.0f, .repetition_penalty = 2.0f};
        int32_t picked = myds_sample_next_token(logits, 8, &cfg, NULL, prev, 1);
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

    /* 6. DeepSeek V4/V4.1 Architecture Enforcement & V2/V3 Rejection */
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
        assert(myds_generate_ex(ctx, "\1\2\3\4\5", NULL, NULL, &reason, NULL, NULL));
        assert(reason == MYDS_STOP_LENGTH || reason == MYDS_STOP_EOS);
        assert(ctx->pos == 6 && ctx->prefill_chunk == NULL);

        myds_context_free(ctx);
        myds_model_free(m);
        unlink(path);
        printf("  [PASS] MLA forward pass & latent KV cache cross-attention verified\n");
    }

    test_prefill_workspace_reuse();

    printf("====================================================\n");
    printf("   ALL ENGINE UNIT & GATE TESTS PASSED!             \n");
    printf("====================================================\n");
    return 0;
}
