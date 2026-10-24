#ifndef MYDS_INTERNAL_H
#define MYDS_INTERNAL_H

#include "myds.h"
#include "myds_gguf.h"
#include "myds_tokenizer.h"
#include "myds_metal.h"
#include "myds_v41.h"
#include "myds_snapshot_slots.h"   /* MYDS_ANCHOR_SLOTS_MAX / myds_anchor_slot */
#include <stdbool.h>
#include <string.h>

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
    /* MYDS_QKV_MERGE=1 时：attn_q_a（1280 行）与 attn_kv（512 行）拼成连续 1792 行，
     * 使 decode 的一次 GEMV 覆盖两者，省掉一次调用的固定开销（实测 ~33 µs/层）。 */
    myds_metal_buffer_t *attn_qkv_merged;
    myds_metal_buffer_t *attn_qkv_out;   /* 1792 行输出，再拆到 qr / kv_act */
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
typedef struct myds_dspark myds_dspark_t;

struct myds_context {
    myds_model_t        *model;
    myds_v41_context_t  *v41;
    myds_dspark_t       *dspark;
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
    /* 常量前缀锚点快照（系统提示词等）：跨非连续请求持久复用 */
    /* 2026-10-24：原为固定 anchor_tokens[1024]。§90 把锚点上限改成可配置
     * （MYDS_ANCHOR_MAX_TOKENS，默认 65536）却没同步这里 ⇒ 任何 >1024 token 的
     * 系统提示词都会把 216 KB 写进 4 KB 数组，砸穿紧随其后的 anchor_len 与
     * img_emb_data/start/count/n（实测 SIGSEGV 在 prefill 的图像注入路径）。
     * 改为按上限动态分配；所有读写都必须先比对 anchor_cap。 */
    /* 2026-10-24：锚点从「单套」改成「多套」——按 MYDS_ANCHOR_SLOTS（默认 16）。
     * 第 i 套占锚点槽 myds_anchor_slot(i)，token 前缀在 anchor_tokens[i]。
     * 判定时遍历全部套、取【最长命中】；捕获时优先填空槽，满了替换最短的一套
     * （最短的前缀复用价值最低）。 */
    int32_t             *anchor_tokens[MYDS_ANCHOR_SLOTS_MAX];  /* 每套容量 anchor_cap */
    uint32_t             anchor_len[MYDS_ANCHOR_SLOTS_MAX];     /* 0 = 空套 */
    uint32_t             anchor_n;       /* 已占用套数（仅用于日志/统计） */
    uint32_t             anchor_slots;   /* 运行时生效的套数上限（MYDS_ANCHOR_SLOTS） */
    uint32_t             anchor_cap;     /* 每套的 token 容量（<= MYDS_ANCHOR_MAX_TOKENS） */
    /* previous_response_id：本请求传入的 prompt 只是【新增后缀】，前缀由 pcache_tokens 提供。
     * 一次性标志（引擎消费后自清），由服务端在收到该字段时设置。 */
    bool                 continue_from_cache;

    /* 最近一次请求的 prefill 统计（只写不读，供 myds_last_prefill_stats 暴露给服务端日志）
     * ★ valid 必须在【每次请求入口】清零：它不是「上一次的值好不好」，而是
     *   「这份值是不是本请求写的」。忘了清就会让服务端把上一个请求的 token 数
     *   当成这个请求的报出去（2026-10-25 修的就是这个）。 */
    uint32_t             last_prompt_tokens;
    uint32_t             last_cached_tokens;
    uint32_t             last_prefill_tokens;
    double               last_prefill_seconds;
    int                  last_prefill_valid;

    myds_decode_stats_t   decode_stats;
    bool                 decode_metrics_active;
    myds_decode_cycle_t  *decode_cycles;
    uint32_t             decode_cycle_capacity;
    bool                 decode_cycle_open;
    double               decode_cycle_start_ms;
    uint64_t             decode_cycle_output_start;
    bool                 decode_cycle_speculative;
    bool                 decode_pld_candidate; /* Candidate A came from prompt lookup. */
    bool                 decode_cost_scheduler;
    double               decode_target_ms;
    uint32_t             decode_costly_cycles;
    uint32_t             decode_cost_backoff;

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

    /* 2026-10-24：近期 token 环形缓冲，供采样器的 presence / repetition 惩罚使用。
     *
     * ★ 为什么必须自己维护：采样函数 myds_sample_next_token 的 prev_tokens 参数
     *   在两处生产调用点都是 (NULL, 0)，于是 has_rep_pen 恒为 false ——
     *   repetition_penalty 配了也从未生效。这里在每次 forward 入口记录，
     *   覆盖 prompt 与生成的全部 token，顺序天然正确，调用方无需改动。
     * 只保留最近 MYDS_PENALTY_WINDOW 个：惩罚作用于「近期上下文」，
     * 而非全部历史（全量会让长上下文下每个 token 的采样退化成 O(n)。 */
    /* 每请求只打一次采样器内部状态（证明惩罚确实被采样器看到）。 */
    uint32_t             sampler_trace_once;
    int32_t              recent_tokens[256];
    uint32_t             recent_count;   /* 0..256，已填充个数 */
    uint32_t             recent_next;    /* 下一个写入下标 */
    void                *sample_cands;
    uint64_t             rng_state;
    /* Structured output token bitmask (borrowed from DeepInfer) */
    /* 结构化输出 bitmask：ctx 独占所有权。
     * logit_bitmask_words 记录已分配的 uint32 数量 —— 没有它就无法在
     * vocab 变化时判断现有缓冲够不够，会导致 memcpy 堆溢出。 */
    uint32_t            *logit_bitmask;
    uint32_t             logit_bitmask_words;

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

/* 记录一个进入模型的 token（prompt 与生成同等对待）。 */
void myds_note_recent_token(myds_context_t *ctx, int32_t token_id);

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
uint32_t myds_v41_snapshot_get_next_position(const myds_v41_context_t *g);
uint32_t myds_v41_snapshot_get_engram_position(const myds_v41_context_t *g);
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
uint32_t myds_v41_resolve_macroblock_size(uint32_t n_tokens);

/* 2026-10-24：kvdisk 两个 store 的容量参数（对外只读，供单测按真实值算预算）。
 * 都是「每次进程解析一次 env 后缓存」的值。 */
uint32_t myds_v41_prompt_cache_slots(void);
uint32_t myds_v41_prompt_cache_max_tokens(void);
uint32_t myds_v41_anchor_slots(void);
uint32_t myds_v41_anchor_max_tokens(void);
uint64_t myds_v41_end_stride_for(uint32_t max_tokens);   /* 每槽字节数（含 header/tail） */

/* 2026-10-24：多套锚点的「最长前缀命中」选择器。
 *
 * 放在头文件里的 static inline，而不是 .c 里的外部函数：它是纯逻辑、无状态，
 * 这样 engine 与单测都能直接用，单测不必为它链接整个 myds_v41.o。
 *
 * 为什么要有这条逻辑：若只取第一个命中，短前缀会抢走长前缀的请求，
 * 长的那套就白存了 —— 多套的意义就是各前缀各得其所。
 *
 * 返回命中的 token 数（0 = 无命中），*out_index 给套号（UINT32_MAX = 无）。
 * 命中的长度必须【严格小于】n_prompt：前缀缓存不能吃掉最后一个 token，
 * 否则剩余 prefill 长度为负。 */
static inline uint32_t myds_v41_anchor_pick(const int32_t *prompt, uint32_t n_prompt,
                                            int32_t *const *anchors, const uint32_t *lens,
                                            uint32_t count, uint32_t *out_index) {
    if (out_index) *out_index = UINT32_MAX;
    if (!prompt || !anchors || !lens || !n_prompt || !count) return 0;
    uint32_t best_len = 0, best_i = UINT32_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t L = lens[i];
        if (!L || !anchors[i]) continue;
        if (L >= n_prompt) continue;            /* 留至少一个 token 给 decode */
        if (L <= best_len) continue;            /* 已有更长命中 */
        if (memcmp(prompt, anchors[i], (size_t)L * sizeof(int32_t)) == 0) {
            best_len = L; best_i = i;
        }
    }
    if (best_i != UINT32_MAX && out_index) *out_index = best_i;
    return best_len;
}
bool myds_validate_fp8_block_size(const gguf_file_t *gf, const char *arch);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_INTERNAL_H */
