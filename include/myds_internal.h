#ifndef MYDS_INTERNAL_H
#define MYDS_INTERNAL_H

#include "myds.h"
#include "myds_gguf.h"
#include "myds_tokenizer.h"
#include "myds_metal.h"
#include "myds_v41.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    myds_metal_buffer_t *attn_norm;
    myds_metal_buffer_t *attn_q;
    myds_metal_buffer_t *attn_k;
    myds_metal_buffer_t *attn_v;
    myds_metal_buffer_t *attn_o;

    /* Multi-Head Latent Attention (MLA) */
    myds_metal_buffer_t *attn_q_a;
    myds_metal_buffer_t *attn_q_a_norm;
    myds_metal_buffer_t *attn_q_b;
    myds_metal_buffer_t *attn_kv;
    myds_metal_buffer_t *attn_kv_a_norm;
    myds_metal_buffer_t *attn_sinks;
    myds_metal_buffer_t *attn_output_a;
    myds_metal_buffer_t *attn_output_b;

    myds_metal_buffer_t *ffn_norm;
    myds_metal_buffer_t *ffn_gate;
    myds_metal_buffer_t *ffn_up;
    myds_metal_buffer_t *ffn_down;

    /* DeepSeek MoE router & experts */
    myds_metal_buffer_t *router_gate;
    myds_metal_buffer_t *router_bias;
    myds_metal_buffer_t *ffn_gate_exps;
    myds_metal_buffer_t *ffn_up_exps;
    myds_metal_buffer_t *ffn_down_exps;
    myds_metal_buffer_t *shared_gate;
    myds_metal_buffer_t *shared_up;
    myds_metal_buffer_t *shared_down;

    /* Weight types */
    ggml_type attn_q_type, attn_k_type, attn_v_type, attn_o_type;
    ggml_type attn_q_a_type, attn_q_b_type, attn_kv_type;
    ggml_type attn_output_a_type, attn_output_b_type;
    ggml_type ffn_gate_type, ffn_up_type, ffn_down_type;
    ggml_type router_gate_type;
    ggml_type ffn_gate_exps_type, ffn_up_exps_type, ffn_down_exps_type;
    ggml_type shared_gate_type, shared_up_type, shared_down_type;
} myds_layer_t;

struct myds_model {
    gguf_file_t         *gf;
    myds_tokenizer_t    *tokenizer;
    myds_v41_model_t    *v41;
    char                 model_path[1024];

    char                 arch[64];
    uint32_t             dim;
    uint32_t             n_layers;
    uint32_t             n_heads;
    uint32_t             n_kv_heads;
    uint32_t             head_dim;
    uint32_t             vocab_size;
    uint32_t             hidden_dim;

    /* DeepSeek MLA parameters */
    bool                 is_mla;
    uint32_t             q_lora_rank;        /* default: 1280 */
    uint32_t             kv_lora_rank;       /* default: 512 */
    uint32_t             qk_rope_head_dim;   /* default: 64 */
    uint32_t             n_out_groups;       /* default: 8 */
    uint32_t             out_group_rank;     /* default: 1024 */

    /* DeepSeek MoE parameters */
    uint32_t             n_routed_experts;
    uint32_t             n_activated_experts;
    uint32_t             n_shared_experts;
    uint32_t             moe_expert_dim;     /* default: 2304 */
    float                routed_scaling_factor;
    float                swiglu_clamp_exp;   /* default: 10.0f */

    float                norm_eps;
    float                freq_base;
    float                freq_scale;

    myds_metal_buffer_t *tok_embeddings;
    ggml_type            tok_embeddings_type;

    myds_layer_t        *layers;

    myds_metal_buffer_t *output_norm;
    myds_metal_buffer_t *output;
    ggml_type            output_type;

    /* Model load instance generation (for cache isolation when manifest digest is absent) */
    uint64_t             load_generation;
};

typedef struct myds_prefill_chunk myds_prefill_chunk_t;

struct myds_context {
    myds_model_t        *model;
    myds_v41_context_t  *v41;
    myds_config_t        cfg;
    uint32_t             pos;
    uint32_t             kv_capacity; /* Allocated KV rows; cfg may request a smaller limit. */
    myds_prefill_chunk_t *prefill_chunk; /* Lazily allocated, reused between requests. */
    uint32_t             prefill_capacity;

    /* Prefix cache（服务端调度用）：记录状态里当前的【完整 token 序列】
     * （prompt + 已生成，EOS 不在其中，因为它从未被喂入）。
     * 下一个请求的 prompt 若是它的严格续写，就可以跳过公共部分的 prefill。
     * 要求严格相等是刻意的：engram 的状态是逐 token 的滚动链，只有已消费的
     * token 序列完全一致时才对齐，回滚做不到。 */
    int32_t             *pcache_tokens;
    uint32_t             pcache_len;
    uint32_t             pcache_cap;
    /* previous_response_id：本请求传入的 prompt 只是【新增后缀】，前缀由 pcache_tokens 提供。
     * 一次性标志（引擎消费后自清），由服务端在收到该字段时设置。 */
    bool                 continue_from_cache;

    /* 最近一次请求的 prefill 统计（只写不读，供 myds_last_prefill_stats 暴露给服务端日志） */
    uint32_t             last_prompt_tokens;
    uint32_t             last_cached_tokens;
    uint32_t             last_prefill_tokens;
    double               last_prefill_seconds;

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

    /* MLA scratch activation buffers */
    myds_metal_buffer_t *qr;         /* [q_lora_rank] */
    myds_metal_buffer_t *qr_norm;    /* [q_lora_rank] */
    myds_metal_buffer_t *kv_act;     /* [head_dim] */
    myds_metal_buffer_t *attn_heads; /* [n_heads * head_dim] */
    myds_metal_buffer_t *attn_low;   /* [n_out_groups * out_group_rank] */

    /* MoE scratch activation buffers */
    myds_metal_buffer_t *router_logits;    /* [n_routed_experts] */
    myds_metal_buffer_t *selected_ids;     /* [n_activated_experts] int32 */
    myds_metal_buffer_t *selected_weights; /* [n_activated_experts] float */
    myds_metal_buffer_t *moe_gate;         /* [n_activated_experts * expert_dim] */
    myds_metal_buffer_t *moe_up;           /* [n_activated_experts * expert_dim] */
    myds_metal_buffer_t *moe_mid;          /* [n_activated_experts * expert_dim] */
    myds_metal_buffer_t *moe_down;         /* [n_activated_experts * dim] */
    myds_metal_buffer_t *moe_out;          /* [dim] */
    myds_metal_buffer_t *shared_gate_out;  /* [expert_dim] */
    myds_metal_buffer_t *shared_up_out;    /* [expert_dim] */
    myds_metal_buffer_t *shared_swiglu_out;/* [expert_dim] */
    myds_metal_buffer_t *shared_out;       /* [dim] */

    /* DeepSeek KV Cache */
    float              **k_cache; /* [n_layers][max_seq_len * stride] */
    float              **v_cache; /* [n_layers][max_seq_len * stride] (MHA only) */

    /* Sampler persistent scratch buffers (zero malloc churn per token) */
    float               *sample_probs;
    int32_t             *sample_indices;
    void                *sample_cands;
    uint64_t             rng_state;

    /* ── 图片嵌入注入（视觉）──
     * 序列位置 [img_emb_start, img_emb_start + img_emb_count) 的嵌入不用词表查表，
     * 而是取下面这份数据。理由是【同一 token id 在不同位置要有不同向量】：
     * V4.1 的图片块所有位置共用 image token id（129264），
     * 逐位置的向量才区分（图片格 = 编码器输出；行尾 = image_newline 哨兵；首尾 = start/end）。
     * 这与 ds4 的做法一致（ds4 也是推同一个 id、靠注入的向量区分）。
     * 指针由 context 持有（添加时拷贝一份），clear / context_free 时释放。
     * ★ 是【数组】而不是单个区间：网关每轮重发完整历史，历史里会有多张图
     *   ⇒ 一次请求天然可能带 N 个图片块。（我第一版写死单区间，
     *     结果第一张图之后的所有对话全被拒 —— 见 docs/VISION.md 的排障节。） */
    float              **img_emb_data;   /* 每个区间一份 [count * dim] */
    uint32_t            *img_emb_start;
    uint32_t            *img_emb_count;
    uint32_t             img_emb_n;      /* 区间个数 */
    uint32_t             img_emb_cap;
    uint32_t             img_emb_dim;    /* 各区间共同的向量宽度 */
    /* 图片块占用的那个 token id（V4.1 是 129264）。引擎用它【自检】区间是否对齐：
     * 区间内每个位置都必须是这个 id，否则向量就贴错了地方。
     * 由调用方在 add 时告知 —— 引擎不该去依赖视觉模块。 */
    uint32_t             img_emb_token_id;
};

/* Unified production sampler: greedy, temperature, top-k, min-p, top-p, repetition penalty */
/* Allows incomplete graphs solely to diagnose tensor/logit mismatches.
 * Never use for user-facing generation or service readiness. */
myds_model_t *myds_model_load_diagnostic(const char *model_path);
bool myds_trace_active(void);
bool myds_trace_f32(const char *stage, uint32_t layer, uint32_t pos,
                    const myds_metal_buffer_t *buffer, size_t count);

int32_t myds_sample_next_token(
    const float         *logits,
    uint32_t             vocab_size,
    const myds_config_t *cfg,
    myds_context_t      *ctx,
    const int32_t       *prev_tokens,
    size_t               n_prev_tokens);

uint64_t myds_v41_model_identity(const myds_model_t *m);
myds_v41_context_t *myds_v41_context_create_for_model(const myds_model_t *model);
myds_v41_context_t *myds_v41_context_create_snapshot_fixture(void);
uint32_t myds_v41_snapshot_get_pos(const myds_v41_context_t *g, uint32_t slot);
uint32_t myds_v41_snapshot_get_staged_pos(const myds_v41_context_t *g, uint32_t slot);
myds_metal_buffer_t *myds_v41_snapshot_get_window(myds_v41_context_t *g, uint32_t layer);
myds_metal_buffer_t *myds_v41_snapshot_get_previous_kv(myds_v41_context_t *g, uint32_t o);
myds_metal_buffer_t *myds_v41_snapshot_get_previous_score(myds_v41_context_t *g, uint32_t o);
void myds_v41_snapshot_set_engram_history(myds_v41_context_t *g, uint32_t t0, uint32_t t1, uint32_t t2);
void myds_v41_snapshot_get_engram_history(const myds_v41_context_t *g, uint32_t out[3]);
void myds_v41_snapshot_set_positions(myds_v41_context_t *g, uint32_t engram_pos, uint32_t next_pos);
bool myds_v41_snapshot_get_valid(const myds_v41_context_t *g);
/* 低位区落盘的单测夹具：给快照夹具补上压缩 KV 缓冲并直接打开磁盘存储（不需要模型）。 */
bool myds_v41_low_kv_fixture_enable(myds_v41_context_t *g, const char *path, uint64_t identity,
                                    uint32_t ways, bool async_on, uint32_t max_pos);
void myds_v41_low_kv_segment_stats(myds_v41_context_t *g, uint32_t *live,
                                   uint64_t *writes, uint64_t *bytes);
void myds_v41_low_kv_write_bytes(myds_v41_context_t *g, uint64_t *snapshots, uint64_t *segments);
struct myds_kvdsk;
struct myds_kvdsk *myds_v41_low_kv_fixture_segments(myds_v41_context_t *g);
struct myds_kvdsk *myds_v41_low_kv_fixture_ends(myds_v41_context_t *g);
void myds_v41_low_kv_fixture_fail_segment(myds_v41_context_t *g, uint32_t end);
void myds_v41_snapshot_fixture_publish(myds_v41_context_t *g, uint32_t slot, uint32_t p);
myds_metal_buffer_t *myds_v41_snapshot_get_snap_window(myds_v41_context_t *g, uint32_t slot);
myds_metal_buffer_t *myds_v41_snapshot_get_snap_prev_kv(myds_v41_context_t *g, uint32_t slot, uint32_t o);
myds_metal_buffer_t *myds_v41_snapshot_get_snap_prev_score(myds_v41_context_t *g, uint32_t slot, uint32_t o);
uint64_t myds_v41_snapshot_allocated_bytes(const myds_v41_context_t *g);
myds_metal_buffer_t *myds_v41_snapshot_get_compressed(myds_v41_context_t *g, uint32_t o);
myds_metal_buffer_t *myds_v41_snapshot_get_index_cache(myds_v41_context_t *g, uint32_t o);
uint32_t myds_v41_resolve_macroblock_size(void);
bool myds_validate_fp8_block_size(const gguf_file_t *gf, const char *arch);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_INTERNAL_H */
