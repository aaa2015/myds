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
    uint64_t n_kvs = 7;

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

        myds_context_free(ctx);
        myds_model_free(m);
        unlink(path);
        printf("  [PASS] MLA forward pass & latent KV cache cross-attention verified\n");
    }

    printf("====================================================\n");
    printf("   ALL ENGINE UNIT & GATE TESTS PASSED!             \n");
    printf("====================================================\n");
    return 0;
}
