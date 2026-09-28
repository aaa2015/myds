#include "myds.h"
#include "myds_metal.h"
#include "myds_gguf.h"
#include "myds_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    myds_metal_buffer_t *attn_norm;
    myds_metal_buffer_t *attn_q;
    myds_metal_buffer_t *attn_k;
    myds_metal_buffer_t *attn_v;
    myds_metal_buffer_t *attn_o;

    myds_metal_buffer_t *ffn_norm;
    myds_metal_buffer_t *ffn_gate;
    myds_metal_buffer_t *ffn_up;
    myds_metal_buffer_t *ffn_down;

    /* DeepSeek MoE router */
    myds_metal_buffer_t *router_gate;
    myds_metal_buffer_t *shared_gate;
    myds_metal_buffer_t *shared_up;
    myds_metal_buffer_t *shared_down;

    /* Weight types */
    ggml_type attn_q_type, attn_k_type, attn_v_type, attn_o_type;
    ggml_type ffn_gate_type, ffn_up_type, ffn_down_type;
} myds_layer_t;

struct myds_model {
    gguf_file_t         *gf;
    myds_tokenizer_t    *tokenizer;

    char                 arch[64];
    uint32_t             dim;
    uint32_t             n_layers;
    uint32_t             n_heads;
    uint32_t             n_kv_heads;
    uint32_t             head_dim;
    uint32_t             vocab_size;
    uint32_t             hidden_dim;

    /* DeepSeek MoE parameters */
    uint32_t             n_routed_experts;
    uint32_t             n_activated_experts;
    uint32_t             n_shared_experts;
    float                routed_scaling_factor;

    float                norm_eps;
    float                freq_base;
    float                freq_scale;

    myds_metal_buffer_t *tok_embeddings;
    ggml_type            tok_embeddings_type;

    myds_layer_t        *layers;

    myds_metal_buffer_t *output_norm;
    myds_metal_buffer_t *output;
    ggml_type            output_type;
};

struct myds_context {
    myds_model_t        *model;
    myds_config_t        cfg;
    uint32_t             pos;

    /* Metal activation buffers */
    myds_metal_buffer_t *x;
    myds_metal_buffer_t *x_norm;
    myds_metal_buffer_t *q;
    myds_metal_buffer_t *k;
    myds_metal_buffer_t *v;
    myds_metal_buffer_t *attn_scores;
    myds_metal_buffer_t *attn_out;
    myds_metal_buffer_t *ffn_norm_out;
    myds_metal_buffer_t *gate_out;
    myds_metal_buffer_t *up_out;
    myds_metal_buffer_t *swiglu_out;
    myds_metal_buffer_t *ffn_out;
    myds_metal_buffer_t *logits;

    /* DeepSeek KV Cache */
    float              **k_cache; /* [n_layers][max_seq_len * n_kv_heads * head_dim] */
    float              **v_cache; /* [n_layers][max_seq_len * n_kv_heads * head_dim] */
};

/* Helper: wrap GGUF tensor memory into Metal buffer */
static myds_metal_buffer_t *create_metal_buffer_from_tensor(const gguf_tensor_t *t) {
    if (!t || !t->data) return NULL;
    myds_metal_buffer_t *buf = myds_metal_buffer_alloc(t->size_bytes);
    if (!buf) return NULL;
    myds_metal_buffer_write(buf, 0, t->data, t->size_bytes);
    return buf;
}

myds_model_t *myds_model_load(const char *model_path) {
    if (!myds_metal_init("metal")) {
        fprintf(stderr, "myds_model_load: failed to initialize Metal backend\n");
        return NULL;
    }

    gguf_file_t *gf = gguf_open(model_path);
    if (!gf) {
        fprintf(stderr, "myds_model_load: failed to open GGUF file '%s'\n", model_path);
        return NULL;
    }

    myds_model_t *m = (myds_model_t *)calloc(1, sizeof(myds_model_t));
    m->gf = gf;
    m->tokenizer = myds_tokenizer_load(gf);

    const char *arch = gguf_get_str(gf, "general.architecture", "deepseek2");
    strncpy(m->arch, arch, sizeof(m->arch) - 1);

    char key_buf[128];
    snprintf(key_buf, sizeof(key_buf), "%s.embedding_length", arch);
    m->dim = gguf_get_u32(gf, key_buf, 2048);

    snprintf(key_buf, sizeof(key_buf), "%s.block_count", arch);
    m->n_layers = gguf_get_u32(gf, key_buf, 28);

    snprintf(key_buf, sizeof(key_buf), "%s.attention.head_count", arch);
    m->n_heads = gguf_get_u32(gf, key_buf, 16);

    snprintf(key_buf, sizeof(key_buf), "%s.attention.head_count_kv", arch);
    m->n_kv_heads = gguf_get_u32(gf, key_buf, m->n_heads);

    m->head_dim = m->dim / m->n_heads;

    snprintf(key_buf, sizeof(key_buf), "%s.feed_forward_length", arch);
    m->hidden_dim = gguf_get_u32(gf, key_buf, m->dim * 4);

    snprintf(key_buf, sizeof(key_buf), "%s.attention.layer_norm_rms_epsilon", arch);
    m->norm_eps = gguf_get_f32(gf, key_buf, 1e-6f);

    snprintf(key_buf, sizeof(key_buf), "%s.rope.freq_base", arch);
    m->freq_base = gguf_get_f32(gf, key_buf, 10000.0f);

    snprintf(key_buf, sizeof(key_buf), "%s.rope.scale_linear", arch);
    m->freq_scale = gguf_get_f32(gf, key_buf, 1.0f);

    /* MoE parameters */
    snprintf(key_buf, sizeof(key_buf), "%s.expert_count", arch);
    m->n_routed_experts = gguf_get_u32(gf, key_buf, 0);

    snprintf(key_buf, sizeof(key_buf), "%s.expert_used_count", arch);
    m->n_activated_experts = gguf_get_u32(gf, key_buf, 0);

    snprintf(key_buf, sizeof(key_buf), "%s.expert_shared_count", arch);
    m->n_shared_experts = gguf_get_u32(gf, key_buf, 0);

    snprintf(key_buf, sizeof(key_buf), "%s.expert_weights_scale", arch);
    m->routed_scaling_factor = gguf_get_f32(gf, key_buf, 1.0f);

    m->vocab_size = m->tokenizer ? m->tokenizer->vocab_size : 32000;

    /* Load Embedding */
    const gguf_tensor_t *t_embd = gguf_find_tensor(gf, "token_embd.weight");
    if (t_embd) {
        m->tok_embeddings = create_metal_buffer_from_tensor(t_embd);
        m->tok_embeddings_type = t_embd->type;
    }

    /* Load Output Norm & Output (LM Head) */
    const gguf_tensor_t *t_out_norm = gguf_find_tensor(gf, "output_norm.weight");
    if (t_out_norm) m->output_norm = create_metal_buffer_from_tensor(t_out_norm);

    const gguf_tensor_t *t_out = gguf_find_tensor(gf, "output.weight");
    if (!t_out) t_out = t_embd; /* weight tying */
    if (t_out) {
        m->output = create_metal_buffer_from_tensor(t_out);
        m->output_type = t_out->type;
    }

    /* Load Layers */
    m->layers = (myds_layer_t *)calloc(m->n_layers, sizeof(myds_layer_t));
    for (uint32_t i = 0; i < m->n_layers; i++) {
        char name[128];

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        m->layers[i].attn_norm = create_metal_buffer_from_tensor(gguf_find_tensor(gf, name));

        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
        const gguf_tensor_t *tq = gguf_find_tensor(gf, name);
        if (tq) {
            m->layers[i].attn_q = create_metal_buffer_from_tensor(tq);
            m->layers[i].attn_q_type = tq->type;
        }

        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
        const gguf_tensor_t *tk = gguf_find_tensor(gf, name);
        if (tk) {
            m->layers[i].attn_k = create_metal_buffer_from_tensor(tk);
            m->layers[i].attn_k_type = tk->type;
        }

        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
        const gguf_tensor_t *tv = gguf_find_tensor(gf, name);
        if (tv) {
            m->layers[i].attn_v = create_metal_buffer_from_tensor(tv);
            m->layers[i].attn_v_type = tv->type;
        }

        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
        const gguf_tensor_t *to = gguf_find_tensor(gf, name);
        if (to) {
            m->layers[i].attn_o = create_metal_buffer_from_tensor(to);
            m->layers[i].attn_o_type = to->type;
        }

        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", i);
        m->layers[i].ffn_norm = create_metal_buffer_from_tensor(gguf_find_tensor(gf, name));

        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", i);
        const gguf_tensor_t *fg = gguf_find_tensor(gf, name);
        if (fg) {
            m->layers[i].ffn_gate = create_metal_buffer_from_tensor(fg);
            m->layers[i].ffn_gate_type = fg->type;
        }

        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", i);
        const gguf_tensor_t *fu = gguf_find_tensor(gf, name);
        if (fu) {
            m->layers[i].ffn_up = create_metal_buffer_from_tensor(fu);
            m->layers[i].ffn_up_type = fu->type;
        }

        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", i);
        const gguf_tensor_t *fd = gguf_find_tensor(gf, name);
        if (fd) {
            m->layers[i].ffn_down = create_metal_buffer_from_tensor(fd);
            m->layers[i].ffn_down_type = fd->type;
        }

        /* MoE router & shared experts */
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", i);
        m->layers[i].router_gate = create_metal_buffer_from_tensor(gguf_find_tensor(gf, name));

        snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", i);
        m->layers[i].shared_gate = create_metal_buffer_from_tensor(gguf_find_tensor(gf, name));

        snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", i);
        m->layers[i].shared_up = create_metal_buffer_from_tensor(gguf_find_tensor(gf, name));

        snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", i);
        m->layers[i].shared_down = create_metal_buffer_from_tensor(gguf_find_tensor(gf, name));
    }

    return m;
}

void myds_model_free(myds_model_t *m) {
    if (!m) return;
    if (m->tok_embeddings) myds_metal_buffer_free(m->tok_embeddings);
    if (m->output_norm)    myds_metal_buffer_free(m->output_norm);
    if (m->output && m->output != m->tok_embeddings) myds_metal_buffer_free(m->output);

    if (m->layers) {
        for (uint32_t i = 0; i < m->n_layers; i++) {
            if (m->layers[i].attn_norm)   myds_metal_buffer_free(m->layers[i].attn_norm);
            if (m->layers[i].attn_q)      myds_metal_buffer_free(m->layers[i].attn_q);
            if (m->layers[i].attn_k)      myds_metal_buffer_free(m->layers[i].attn_k);
            if (m->layers[i].attn_v)      myds_metal_buffer_free(m->layers[i].attn_v);
            if (m->layers[i].attn_o)      myds_metal_buffer_free(m->layers[i].attn_o);
            if (m->layers[i].ffn_norm)    myds_metal_buffer_free(m->layers[i].ffn_norm);
            if (m->layers[i].ffn_gate)    myds_metal_buffer_free(m->layers[i].ffn_gate);
            if (m->layers[i].ffn_up)      myds_metal_buffer_free(m->layers[i].ffn_up);
            if (m->layers[i].ffn_down)    myds_metal_buffer_free(m->layers[i].ffn_down);
            if (m->layers[i].router_gate) myds_metal_buffer_free(m->layers[i].router_gate);
            if (m->layers[i].shared_gate) myds_metal_buffer_free(m->layers[i].shared_gate);
            if (m->layers[i].shared_up)   myds_metal_buffer_free(m->layers[i].shared_up);
            if (m->layers[i].shared_down) myds_metal_buffer_free(m->layers[i].shared_down);
        }
        free(m->layers);
    }

    if (m->tokenizer) myds_tokenizer_free(m->tokenizer);
    if (m->gf)        gguf_close(m->gf);
    free(m);
}

void myds_model_print_info(const myds_model_t *m) {
    if (!m) return;
    printf("====================================================\n");
    printf(" DeepSeek Model Information\n");
    printf("====================================================\n");
    printf(" Architecture : %s\n", m->arch);
    printf(" Layers       : %u\n", m->n_layers);
    printf(" Hidden Dim   : %u\n", m->dim);
    printf(" Attention    : %u heads (%u KV), head_dim=%u\n", m->n_heads, m->n_kv_heads, m->head_dim);
    printf(" FFN Dim      : %u\n", m->hidden_dim);
    printf(" Vocab Size   : %u\n", m->vocab_size);
    if (m->n_routed_experts > 0) {
        printf(" MoE Routed   : %u experts (top-%u active), scale=%.2f\n",
               m->n_routed_experts, m->n_activated_experts, m->routed_scaling_factor);
        printf(" MoE Shared   : %u experts\n", m->n_shared_experts);
    } else {
        printf(" MoE          : Dense model\n");
    }
    printf("====================================================\n");
}

myds_context_t *myds_context_create(myds_model_t *model, const myds_config_t *cfg) {
    if (!model) return NULL;

    myds_context_t *ctx = (myds_context_t *)calloc(1, sizeof(myds_context_t));
    ctx->model = model;
    if (cfg) {
        ctx->cfg = *cfg;
    } else {
        ctx->cfg.max_seq_len = 4096;
        ctx->cfg.max_new_tokens = 256;
        ctx->cfg.temperature = 0.7f;
        ctx->cfg.top_p = 0.9f;
        ctx->cfg.seed = (int32_t)time(NULL);
        ctx->cfg.stream = true;
    }
    if (ctx->cfg.max_seq_len == 0) ctx->cfg.max_seq_len = 4096;

    srand(ctx->cfg.seed);

    uint32_t dim = model->dim;
    uint32_t hidden_dim = model->hidden_dim;
    uint32_t vocab_size = model->vocab_size;

    ctx->x            = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->x_norm       = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->q            = myds_metal_buffer_alloc(model->n_heads * model->head_dim * sizeof(float));
    ctx->k            = myds_metal_buffer_alloc(model->n_kv_heads * model->head_dim * sizeof(float));
    ctx->v            = myds_metal_buffer_alloc(model->n_kv_heads * model->head_dim * sizeof(float));
    ctx->attn_scores  = myds_metal_buffer_alloc(ctx->cfg.max_seq_len * sizeof(float));
    ctx->attn_out     = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->ffn_norm_out = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->gate_out     = myds_metal_buffer_alloc(hidden_dim * sizeof(float));
    ctx->up_out       = myds_metal_buffer_alloc(hidden_dim * sizeof(float));
    ctx->swiglu_out   = myds_metal_buffer_alloc(hidden_dim * sizeof(float));
    ctx->ffn_out      = myds_metal_buffer_alloc(dim * sizeof(float));
    ctx->logits       = myds_metal_buffer_alloc(vocab_size * sizeof(float));

    /* Allocate KV cache */
    ctx->k_cache = (float **)calloc(model->n_layers, sizeof(float *));
    ctx->v_cache = (float **)calloc(model->n_layers, sizeof(float *));
    size_t kv_layer_size = (size_t)ctx->cfg.max_seq_len * model->n_kv_heads * model->head_dim;
    for (uint32_t l = 0; l < model->n_layers; l++) {
        ctx->k_cache[l] = (float *)calloc(kv_layer_size, sizeof(float));
        ctx->v_cache[l] = (float *)calloc(kv_layer_size, sizeof(float));
    }

    ctx->pos = 0;
    return ctx;
}

void myds_context_free(myds_context_t *ctx) {
    if (!ctx) return;
    if (ctx->x)            myds_metal_buffer_free(ctx->x);
    if (ctx->x_norm)       myds_metal_buffer_free(ctx->x_norm);
    if (ctx->q)            myds_metal_buffer_free(ctx->q);
    if (ctx->k)            myds_metal_buffer_free(ctx->k);
    if (ctx->v)            myds_metal_buffer_free(ctx->v);
    if (ctx->attn_scores)  myds_metal_buffer_free(ctx->attn_scores);
    if (ctx->attn_out)     myds_metal_buffer_free(ctx->attn_out);
    if (ctx->ffn_norm_out) myds_metal_buffer_free(ctx->ffn_norm_out);
    if (ctx->gate_out)     myds_metal_buffer_free(ctx->gate_out);
    if (ctx->up_out)       myds_metal_buffer_free(ctx->up_out);
    if (ctx->swiglu_out)   myds_metal_buffer_free(ctx->swiglu_out);
    if (ctx->ffn_out)      myds_metal_buffer_free(ctx->ffn_out);
    if (ctx->logits)       myds_metal_buffer_free(ctx->logits);

    if (ctx->k_cache) {
        for (uint32_t l = 0; l < ctx->model->n_layers; l++) free(ctx->k_cache[l]);
        free(ctx->k_cache);
    }
    if (ctx->v_cache) {
        for (uint32_t l = 0; l < ctx->model->n_layers; l++) free(ctx->v_cache[l]);
        free(ctx->v_cache);
    }
    free(ctx);
}

myds_config_t myds_context_get_config(const myds_context_t *ctx) {
    if (!ctx) {
        myds_config_t empty = {0};
        return empty;
    }
    return ctx->cfg;
}

void myds_context_set_config(myds_context_t *ctx, const myds_config_t *cfg) {
    if (!ctx || !cfg) return;
    ctx->cfg = *cfg;
}

/* Sampler */
static int32_t sample_next_token(const float *logits, uint32_t vocab_size, float temp, float top_p) {
    if (temp <= 0.0f) {
        /* Greedy argmax */
        int32_t best_id = 0;
        float max_l = logits[0];
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best_id = (int32_t)i;
            }
        }
        return best_id;
    }

    /* Softmax with temperature */
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    float max_l = -INFINITY;
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_l) / temp);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] *= inv_sum;
    }

    /* Nucleus (top-p) sampling */
    float r = ((float)rand() / (float)RAND_MAX) * top_p;
    float cum = 0.0f;
    int32_t sampled_id = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        cum += probs[i];
        if (cum >= r) {
            sampled_id = (int32_t)i;
            break;
        }
    }
    free(probs);
    return sampled_id;
}

int32_t myds_forward_token(myds_context_t *ctx, int32_t token_id, uint32_t pos) {
    myds_model_t *m = ctx->model;
    uint32_t dim = m->dim;
    uint32_t n_heads = m->n_heads;
    uint32_t n_kv_heads = m->n_kv_heads;
    uint32_t head_dim = m->head_dim;

    /* 1. Embedding lookup */
    float *xv = (float *)myds_metal_buffer_data(ctx->x);
    if (m->tok_embeddings) {
        if (m->tok_embeddings_type == GGML_TYPE_F32) {
            const float *emb = (const float *)myds_metal_buffer_data(m->tok_embeddings);
            memcpy(xv, emb + (size_t)token_id * dim, dim * sizeof(float));
        } else if (m->tok_embeddings_type == GGML_TYPE_F16) {
            const uint16_t *emb = (const uint16_t *)myds_metal_buffer_data(m->tok_embeddings);
            const uint16_t *row = emb + (size_t)token_id * dim;
            for (uint32_t i = 0; i < dim; i++) {
                uint32_t w = (uint32_t)row[i] << 16;
                memcpy(&xv[i], &w, sizeof(float));
            }
        } else {
            memset(xv, 0, dim * sizeof(float));
        }
    }

    /* 2. Transformer layers */
    for (uint32_t l = 0; l < m->n_layers; l++) {
        myds_layer_t *layer = &m->layers[l];

        /* Pre-Attention RMSNorm */
        if (layer->attn_norm) {
            myds_metal_rmsnorm(ctx->x_norm, ctx->x, layer->attn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(ctx->x_norm), xv, dim * sizeof(float));
        }

        /* Q, K, V projections */
        if (layer->attn_q) {
            myds_metal_gemv(ctx->q, layer->attn_q, ctx->x_norm, n_heads * head_dim, dim, layer->attn_q_type);
        }
        if (layer->attn_k) {
            myds_metal_gemv(ctx->k, layer->attn_k, ctx->x_norm, n_kv_heads * head_dim, dim, layer->attn_k_type);
        }
        if (layer->attn_v) {
            myds_metal_gemv(ctx->v, layer->attn_v, ctx->x_norm, n_kv_heads * head_dim, dim, layer->attn_v_type);
        }

        /* Decoupled RoPE on Q & K */
        myds_metal_rope(ctx->q, ctx->k, n_heads, head_dim, pos, m->freq_base, m->freq_scale);

        /* Write into KV Cache */
        float *cur_k = (float *)myds_metal_buffer_data(ctx->k);
        float *cur_v = (float *)myds_metal_buffer_data(ctx->v);
        size_t kv_stride = n_kv_heads * head_dim;
        memcpy(ctx->k_cache[l] + (size_t)pos * kv_stride, cur_k, kv_stride * sizeof(float));
        memcpy(ctx->v_cache[l] + (size_t)pos * kv_stride, cur_v, kv_stride * sizeof(float));

        /* Multi-Head Attention computation */
        float *qv = (float *)myds_metal_buffer_data(ctx->q);
        float *attn_out_v = (float *)myds_metal_buffer_data(ctx->attn_out);
        memset(attn_out_v, 0, dim * sizeof(float));

        float scale_attn = 1.0f / sqrtf((float)head_dim);
        uint32_t kv_mul = n_heads / n_kv_heads;

        for (uint32_t h = 0; h < n_heads; h++) {
            uint32_t kv_h = h / kv_mul;
            float *q_head = qv + h * head_dim;

            /* Attention scores */
            float *scores = (float *)myds_metal_buffer_data(ctx->attn_scores);
            float max_s = -INFINITY;
            for (uint32_t t = 0; t <= pos; t++) {
                const float *k_head = ctx->k_cache[l] + (size_t)t * kv_stride + kv_h * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += q_head[d] * k_head[d];
                }
                scores[t] = dot * scale_attn;
                if (scores[t] > max_s) max_s = scores[t];
            }

            /* Softmax */
            float sum_exp = 0.0f;
            for (uint32_t t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / sum_exp;
            for (uint32_t t = 0; t <= pos; t++) {
                scores[t] *= inv_sum;
            }

            /* Value weighted sum */
            float *out_head = attn_out_v + h * head_dim;
            for (uint32_t t = 0; t <= pos; t++) {
                const float *v_head = ctx->v_cache[l] + (size_t)t * kv_stride + kv_h * head_dim;
                float weight = scores[t];
                for (uint32_t d = 0; d < head_dim; d++) {
                    out_head[d] += weight * v_head[d];
                }
            }
        }

        /* Output projection */
        if (layer->attn_o) {
            myds_metal_buffer_t *tmp_out = myds_metal_buffer_alloc(dim * sizeof(float));
            myds_metal_gemv(tmp_out, layer->attn_o, ctx->attn_out, dim, n_heads * head_dim, layer->attn_o_type);
            /* Residual connection: x = x + attn_output */
            myds_metal_add(ctx->x, ctx->x, tmp_out, dim);
            myds_metal_buffer_free(tmp_out);
        }

        /* Pre-FFN RMSNorm */
        if (layer->ffn_norm) {
            myds_metal_rmsnorm(ctx->ffn_norm_out, ctx->x, layer->ffn_norm, dim, m->norm_eps);
        } else {
            memcpy(myds_metal_buffer_data(ctx->ffn_norm_out), xv, dim * sizeof(float));
        }

        /* FFN: MoE or Dense */
        if (m->n_routed_experts > 0 && layer->router_gate) {
            /* DeepSeek MoE routing */
            myds_metal_buffer_t *router_logits = myds_metal_buffer_alloc(m->n_routed_experts * sizeof(float));
            myds_metal_gemv(router_logits, layer->router_gate, ctx->ffn_norm_out, m->n_routed_experts, dim, GGML_TYPE_F32);

            int32_t exp_ids[32];
            float exp_weights[32];
            uint32_t k_used = (m->n_activated_experts < 32) ? m->n_activated_experts : 32;

            myds_metal_moe_route(exp_ids, exp_weights, router_logits, m->n_routed_experts, k_used, m->routed_scaling_factor);
            myds_metal_buffer_free(router_logits);

            /* Shared expert forward if present */
            if (layer->shared_gate && layer->shared_up && layer->shared_down) {
                myds_metal_gemv(ctx->gate_out, layer->shared_gate, ctx->ffn_norm_out, m->hidden_dim, dim, GGML_TYPE_Q8_0);
                myds_metal_gemv(ctx->up_out, layer->shared_up, ctx->ffn_norm_out, m->hidden_dim, dim, GGML_TYPE_Q8_0);
                myds_metal_swiglu(ctx->swiglu_out, ctx->gate_out, ctx->up_out, m->hidden_dim);
                myds_metal_gemv(ctx->ffn_out, layer->shared_down, ctx->swiglu_out, dim, m->hidden_dim, GGML_TYPE_Q8_0);
                myds_metal_add(ctx->x, ctx->x, ctx->ffn_out, dim);
            }
        } else if (layer->ffn_gate && layer->ffn_up && layer->ffn_down) {
            /* Dense SwiGLU FFN */
            myds_metal_gemv(ctx->gate_out, layer->ffn_gate, ctx->ffn_norm_out, m->hidden_dim, dim, layer->ffn_gate_type);
            myds_metal_gemv(ctx->up_out, layer->ffn_up, ctx->ffn_norm_out, m->hidden_dim, dim, layer->ffn_up_type);
            myds_metal_swiglu(ctx->swiglu_out, ctx->gate_out, ctx->up_out, m->hidden_dim);
            myds_metal_gemv(ctx->ffn_out, layer->ffn_down, ctx->swiglu_out, dim, m->hidden_dim, layer->ffn_down_type);

            /* Residual connection: x = x + ffn_output */
            myds_metal_add(ctx->x, ctx->x, ctx->ffn_out, dim);
        }
    }

    /* 3. Final RMSNorm */
    if (m->output_norm) {
        myds_metal_rmsnorm(ctx->x_norm, ctx->x, m->output_norm, dim, m->norm_eps);
    } else {
        memcpy(myds_metal_buffer_data(ctx->x_norm), xv, dim * sizeof(float));
    }

    /* 4. LM Head Logits */
    if (m->output) {
        myds_metal_gemv(ctx->logits, m->output, ctx->x_norm, m->vocab_size, dim, m->output_type);
    }

    /* 5. Sample next token */
    const float *logits_v = (const float *)myds_metal_buffer_data(ctx->logits);
    int32_t next_token = sample_next_token(logits_v, m->vocab_size, ctx->cfg.temperature, ctx->cfg.top_p);

    return next_token;
}

int myds_generate(myds_context_t *ctx, const char *prompt, myds_stream_callback_t cb, void *user_data) {
    if (!ctx || !prompt) return 0;
    myds_model_t *m = ctx->model;

    /* 1. Tokenize prompt */
    int32_t prompt_tokens[4096];
    int32_t n_prompt = myds_tokenizer_encode(m->tokenizer, prompt, prompt_tokens, 4096, true);
    if (n_prompt <= 0) return 0;

    ctx->pos = 0;
    int32_t cur_token = prompt_tokens[0];

    /* 2. Prefill phase */
    for (int32_t i = 0; i < n_prompt - 1; i++) {
        myds_forward_token(ctx, prompt_tokens[i], ctx->pos);
        ctx->pos++;
    }
    cur_token = prompt_tokens[n_prompt - 1];

    /* 3. Decode phase */
    uint32_t max_gen = ctx->cfg.max_new_tokens;
    for (uint32_t step = 0; step < max_gen; step++) {
        int32_t next_token = myds_forward_token(ctx, cur_token, ctx->pos);
        ctx->pos++;

        if (next_token == m->tokenizer->eos_id) {
            break;
        }

        const char *piece = myds_tokenizer_decode_piece(m->tokenizer, next_token);
        if (cb) {
            cb(piece, next_token, user_data);
        }

        cur_token = next_token;
        if (ctx->pos >= ctx->cfg.max_seq_len) break;
    }

    return 1;
}
