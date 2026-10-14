#ifndef MYDS_V41_GPU_H
#define MYDS_V41_GPU_H
#include "myds_metal.h"
/* Decode primitives for the fixed DeepSeek V4.1 graph. Float buffers hold
 * dequantized BF16/FP8/FP4 values; ids buffers hold int32. All operations are
 * GPU ordered within the caller's command batch. */
typedef struct myds_v41_gpu_scratch myds_v41_gpu_scratch_t;
/* 批量 indexer（prefill Step 4）的临时缓冲预算，单位 float。分数表是
 * “本组 token 数 × 压缩键数”的行主序矩形，top-k 的中间排序结果与块选择缓冲
 * 都按同一个预算分配：调用方保证组内 token 数 × 该组最大压缩键数 ≤ 这个值。
 * 1 MB/缓冲 ⇒ 长 prompt 下每组约 40~60 个 token（vs 逐 token 每次内核 ~9-14 次
 * dispatch），短 prompt 下整 chunk 一组。 */
#define MYDS_V41_INDEXER_BATCH_FLOATS 262144u
/* 单个 chunk 的 token 上界。这是【单一真源】：host 侧钳 chunk 尺寸、chunk attention
 * 的合法性检查都用它。以前 host 钳到 4096、attention 只收 2048 —— 配置合法但运行失败。 */
#define MYDS_V41_CHUNK_MAX 2048u
/* MYDS_ATTN_GATHER_MIN_KEYS defaults to zero. A valid threshold in [0,key_max]
 * selects slab below the threshold and gather at/above it when gather is enabled
 * and a gathered buffer is supplied. Invalid thresholds reject scratch creation.
 * The explicit strategy is captured per scratch; no measured threshold is forced. */
myds_v41_gpu_scratch_t *myds_v41_gpu_scratch_create(uint32_t max_comp);
/* Default-off ordinary direct/RF1 packing candidate. The process configuration
 * accepts only absent/0/1 and is fixed after the first getter call. Explicit1
 * requires RF1, packed register1/physical8/reduce1 and DIRECT_PACK absent/1. Public direct, staged and
 * trace paths retain the reference packing shaders. Missing selected PSO or
 * invalid input fails before scratch writes; no retry/fallback is permitted.
 * Counters cover the selected pack dispatch; later attention/RF failures are
 * reported by the enclosing forward and its existing counters. */
int myds_v41_gpu_attention_direct_pack_fuse_enabled(void);
const char *myds_v41_gpu_attention_direct_pack_fuse_selected_kernel(void);
void myds_v41_gpu_attention_direct_pack_fuse_counters(
    myds_metal_decode_candidate_counters_t *out,bool reset);
/* Pack the upper 16 bits of already BF16-rounded F32 values. Offsets are in
 * destination uint16 words / source floats; no numerical conversion or rounding. */
int myds_v41_gpu_pack_bf16(myds_metal_buffer_t *dst,size_t dst_offset_words,
    const myds_metal_buffer_t *src,size_t src_offset_floats,uint32_t count);
/* chunk attention 的运行时几何量（由【配置的上下文】推出，见 scratch_create）。
 * 旧版把键上界写死成 16384，导致「配置 262144、实际 16384 就整个 prefill 失败」
 * （而且注释谎称会回退逐 token，代码里其实是 goto fail）。
 *   chunk_max_c()   = 单个 chunk 的 token 上界
 *   chunk_key_max() = 一次 attention 能覆盖的压缩键总数上界（超了必须提前拒绝） */
uint32_t myds_v41_gpu_chunk_max_c(const myds_v41_gpu_scratch_t *scratch);
uint32_t myds_v41_gpu_chunk_key_max(const myds_v41_gpu_scratch_t *scratch);
/* Actual shared-memory bytes occupied by the chunk KV slab and mask. Initially
 * compact; a compressed slab fallback may grow it to the configured key cap. */
size_t myds_v41_gpu_chunk_workspace_bytes(const myds_v41_gpu_scratch_t *scratch);
void myds_v41_gpu_scratch_free(myds_v41_gpu_scratch_t *scratch);
/* Diagnostic fixtures only: 0 disables, 1 requests a nonexistent matrix PSO,
 * 2 requests unsupported shared memory. Set only between completed GPU batches. */
void myds_v41_gpu_inject_attention_matrix_failure(uint32_t kind);
/* Gather prefill vec is head-first by default; MYDS_ATTN_VEC_HEAD_FIRST=0 selects
 * the original grid. Missing optional PSOs automatically cache a fallback.
 * Diagnostic status: 0 disabled/null, 1 enabled, 2 cached unavailable; before
 * the first gathered vec dispatch, enabled does not yet imply PSO availability. */
uint32_t myds_v41_gpu_attention_vec_head_first_status(const myds_v41_gpu_scratch_t *scratch);
/* Diagnostic fixture only: force an absent head-first PSO. Change only between
 * completed GPU batches. Both default and explicit enablement remain best effort. */
void myds_v41_gpu_inject_attention_vec_head_first_failure(bool enabled);
/* Process-wide prefill attention path counters (host-side, relaxed atomics).
 * Each attention_chunk call that passes validation counts once as gather or
 * slab; the vec/matrix fields count encoded kernels. head_first_failures is
 * the call that made a scratch fall back; cached_fallbacks are gathered calls
 * that used the original grid because of an earlier failure. The macro lets
 * tools compile against headers that predate this API. */
#define MYDS_V41_GPU_HAS_ATTENTION_COUNTERS 1
typedef struct {
    uint64_t calls, gather_calls, slab_calls;
    uint64_t head_first_attempts, head_first_encoded, head_first_failures, head_first_cached_fallbacks;
    uint64_t original_vec_encoded, matrix_encoded, matrix_fallbacks;
} myds_v41_gpu_attention_counters_t;
/* Snapshot into *out (may be NULL); reset=true atomically zeroes each field. */
void myds_v41_gpu_attention_counters(myds_v41_gpu_attention_counters_t *out, bool reset);
int myds_v41_gpu_rope(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool compressed,bool inverse);
int myds_v41_gpu_rope_round(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool compressed,bool inverse);
/* DSpark's F32 activation contract is separate from target-model BF16 QAT.
 * Adjacent-pair RoPE touches only the last 64 columns, base 10000, no YaRN.
 * KV QAT uses 64-column E4M3FN blocks on columns [0,448), leaving [448,512)
 * unchanged. These in-place helpers join prior readers/writers in auto0. */
int myds_dspark_gpu_rope(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool inverse);
int myds_dspark_gpu_quant_kv(myds_metal_buffer_t *x,uint32_t rows);
int myds_v41_gpu_rope_batch(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t start_pos,uint32_t rows,bool compressed,bool inverse);
int myds_v41_gpu_rope_batch_round(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t start_pos,uint32_t rows,bool compressed,bool inverse);
/* 同上，但行间的位置步长可指定（stride=1 即 rope_batch）。压缩器批量路径用它：
 * ratio==2 的提交行位置是 first_pos + 2*row。内核算的是 theta=(start+row*stride)*freq。 */
int myds_v41_gpu_rope_batch_stride(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t start_pos,uint32_t rows,uint32_t stride,bool compressed,bool inverse);
/* Fixed decode Q[64,512], KV[512], F32 window[128,512]. Fuses forward
 * Q/K RoPE, mode-1 KV quantization and ring store with all BF16 boundaries.
 * Missing fused kernel / copy injection / overlapping storage use originals. */
int myds_v41_gpu_qkv_finish(myds_metal_buffer_t *q,myds_metal_buffer_t *kv,
    myds_metal_buffer_t *window,uint32_t pos,bool compressed);
/* 0 BF16, 1 FP8 E8M0 scales, 2 FP4 E8M0 scales, 3 FP4 E4M3 scales. */
int myds_v41_gpu_quant(myds_metal_buffer_t *x,uint32_t width,uint32_t rows,uint32_t mode);
/* Explicit opt-in diagnostic API: one contiguous indexer row, heads1 or
 * heads32, fixed width128 / forward compressed RoPE / mode2 E2M1. Requires
 * immutable MYDS_INDEXER_ROPE_QUANT_FUSE=1. The generic RoPE/quant APIs above
 * always retain their reference dispatches; Engine scope is supplied by the
 * ordinary scalar call sites, not inferred here from token count or env. */
int myds_v41_gpu_indexer_rope_quant(myds_metal_buffer_t *x,uint32_t heads,uint32_t pos);
/* Even positions cache kv/scores; odd positions write the pooled output. */
int myds_v41_gpu_pool2(myds_metal_buffer_t *out,const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores,myds_metal_buffer_t *previous_kv,
    myds_metal_buffer_t *previous_scores,uint32_t width,uint32_t pos);
/* 整 chunk 版：out 的第 p 行 = 第 p 对（偶数位 kv/scores 与紧随的奇数位）合并的结果，
 * 每对的算术与 pool2 逐 token 版完全相同（逐元素，无跨线程归约），故逐位一致。
 * 要求 chunk 从偶数位开始且长度为偶数（调用方保证）。 */
/* 续跑（prefix cache）时用：把上一段结束位置之前的 127 个 KV 从【每层环形窗】
 * 按位置顺序搬进 slab 的 carry 段。窗是按位置旋转的（槽位 = 位置 %% 128），
 * 所以这里是两段拷贝。内容与 chunk 路径自己留下的 carry 逐位相同（同一个张量）。 */
int myds_v41_gpu_carry_stage(myds_v41_gpu_scratch_t *s,const myds_metal_buffer_t *window,
    uint32_t base_pos);
int myds_v41_gpu_pool2_pairs(myds_metal_buffer_t *out,const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores,uint32_t width,uint32_t pairs);
int myds_v41_gpu_indexscores(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    uint32_t n_comp,uint32_t n_heads);
int myds_v41_gpu_indexscores_bf16(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    uint32_t n_comp,uint32_t n_heads);
int myds_v41_gpu_indexscores_masked(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *mask,uint32_t n_comp,uint32_t n_heads);
int myds_v41_gpu_indexscores_masked_bf16(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *mask,uint32_t n_comp,uint32_t n_heads);
/* Initialization-time MYDS_INDEXER_HEAD_SLOTS accepts only absent/0/1. The
 * explicit candidate supports one scalar query, 32 heads and head_dim=128;
 * F32/BF16 keys and an optional original block mask retain their old meaning.
 * It never retries the reference scorer or CPU after failure. Original scorer
 * APIs above are always reference, irrespective of this mode. Auto0 callers
 * supply upstream reader dependencies, score RAW before consumption, and WAR
 * before reusing scores; the candidate adds no default dependency barriers. */
int myds_v41_gpu_indexer_head_slots_enabled(void);
typedef struct { uint64_t requests, encoded, unavailable, failures; }
    myds_v41_gpu_indexer_head_slots_counters_t;
/* Host counts: unavailable is a pre-encode subset of failures. Encoded records
 * an actual dispatch, even if standalone completion subsequently fails; an
 * enclosing batch's later completion failure remains the caller's contract. */
void myds_v41_gpu_indexer_head_slots_counters(
    myds_v41_gpu_indexer_head_slots_counters_t *out,bool reset);
int myds_v41_gpu_indexscores_head_slots(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t n_heads,bool cache_bf16);
/* Optional 32-head decode scorer: expand the unique selected block IDs (8
 * original rows/block), write the original dense score locations, and leave
 * every other row at -INFINITY. The caller must pass all allowed block IDs
 * exactly once, as produced by the reference block top-k. Dense top-k stays
 * unchanged, including its original partition/network-dependent tie order.
 * Returns 1 encoded, 0 optional PSOs unavailable (no writes), -1 validation or
 * dispatch failure (caller must fail rather than retry after partial writes). */
int myds_v41_gpu_indexscores_selected_blocks_bf16(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache,const myds_metal_buffer_t *mask,
    const myds_metal_buffer_t *block_ids,uint32_t n_comp,uint32_t n_selected_blocks);
typedef struct {
    uint64_t requests, encoded, unavailable, dense_rows, scored_row_slots;
} myds_v41_gpu_indexer_compact_counters_t;
void myds_v41_gpu_indexer_compact_counters(myds_v41_gpu_indexer_compact_counters_t *out, bool reset);
/* Diagnostic fixture only: emulate absent optional PSOs before any dispatch. */
void myds_v41_gpu_inject_indexer_compact_unavailable(bool enabled);
/* 整 chunk 批量版（n_tokens 行 × n_comp 列，行主序，行步长 n_comp）。只支持
 * n_heads == 32；每行的因果宽度 = (causal_start + row + 1) / causal_ratio，
 * 与逐 token 的 count 相同；行数不够的格子写 -INFINITY。 */
int myds_v41_gpu_indexscores_chunk(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    uint32_t n_comp,uint32_t n_heads,uint32_t n_tokens,
    uint32_t causal_start,uint32_t causal_ratio);
int myds_v41_gpu_indexscores_chunk_bf16(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    uint32_t n_comp,uint32_t n_heads,uint32_t n_tokens,
    uint32_t causal_start,uint32_t causal_ratio);
/* 批量 indexer 的缓冲（分数表、top-k 中间结果、块分数、块选择结果）。
 * 调用方只用它们，不拥有它们（随 scratch 释放）。 */
myds_metal_buffer_t *myds_v41_gpu_indexer_scores(const myds_v41_gpu_scratch_t *scratch);
myds_metal_buffer_t *myds_v41_gpu_indexer_blocks(const myds_v41_gpu_scratch_t *scratch);
myds_metal_buffer_t *myds_v41_gpu_indexer_block_selected(const myds_v41_gpu_scratch_t *scratch);
uint32_t myds_v41_gpu_indexer_batch_floats(const myds_v41_gpu_scratch_t *scratch);
int myds_v41_gpu_candidates(myds_metal_buffer_t *blocks,const myds_metal_buffer_t *scores,
    uint32_t n_comp,uint32_t pos,uint32_t ratio);
int myds_v41_gpu_filter(myds_metal_buffer_t *scores,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t pos,uint32_t ratio);
/* 按行批量版：rows 行、行首位置 start、每行因果宽度 (start+row+1)/ratio；
 * mask_stride 是 block_mask/分数缓冲的行步长（float 为单位）。 */
int myds_v41_gpu_candidates_rows(myds_metal_buffer_t *blocks,const myds_metal_buffer_t *scores,
    uint32_t n_comp,uint32_t rows,uint32_t start,uint32_t ratio,uint32_t mask_stride);
int myds_v41_gpu_filter_rows(myds_metal_buffer_t *scores,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t rows,uint32_t start,uint32_t ratio,uint32_t mask_stride);
int myds_v41_gpu_topk(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,myds_v41_gpu_scratch_t *scratch);
/* Immutable MYDS_INDEXER_TOPK_NAN_SHUFFLE=absent/0/1. Only the original
 * prefix eligibility with NAN_FUSE=1 opts into independent scalar/causal PSOs.
 * Reference nth, partition/merge order, IEEE flag planes and joins are retained.
 * Explicit eligible failures reject without dense/CPU fallback; counters are
 * host whole-chain encoding decisions, not command-buffer completion. */
int myds_v41_gpu_topk_nan_shuffle_enabled(void);
typedef struct { uint64_t requests, encoded, failures; } myds_v41_gpu_topk_nan_shuffle_counters_t;
void myds_v41_gpu_topk_nan_shuffle_counters(myds_v41_gpu_topk_nan_shuffle_counters_t *out,bool reset);
/* Immutable init policy MYDS_INDEXER_TOPK_THRESHOLD=absent/0/1. Only the
 * authoritative ordinary scalar indexer uses the purpose API; generic top-k,
 * block selection, prefill and private speculative graphs remain reference. */
int myds_v41_gpu_topk_threshold_enabled(void);
typedef struct { uint64_t requests, encoded, errors; } myds_v41_gpu_topk_threshold_counters_t;
void myds_v41_gpu_topk_threshold_counters(myds_v41_gpu_topk_threshold_counters_t *out,bool reset);
typedef struct {
    uint64_t validated, adopted, fallback, nan, inf, insufficient, capacity, certificate, tie;
} myds_v41_gpu_topk_threshold_stats_t;
/* Actual GPU decisions, uint32 accumulators promoted to uint64. Read/reset only
 * after the caller's existing completed synchronization; pending work returns
 * zero without waiting or resetting. No buffer (default0) returns success/0s.
 * Reset per request before encoding; accumulation must remain below 2^32. */
int myds_v41_gpu_topk_threshold_stats(myds_v41_gpu_topk_threshold_stats_t *out,bool reset);
/* Completed-batch diagnostic readback only. No allocation, dispatch or wait;
 * pending GPU work/missing workspace rejects. Pairs are meaningful only when
 * final sorting ran (adopted or tie/certificate after a successful compact),
 * and may be stale for NaN/Inf/insufficient/capacity early fallback. */
int myds_v41_gpu_topk_threshold_debug(const myds_v41_gpu_scratch_t *scratch,
    uint32_t control[12],uint32_t pairs[2048]);
int myds_v41_gpu_topk_threshold(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,myds_v41_gpu_scratch_t *scratch);
/* Explicit diagnostic, independent of policy. Scalar Top512 with 512..262144
 * scores; all PSO/buffer preflight failures reject before writes. GPU numerical
 * guard failure executes the preencoded original reference, never a CPU wait. */
int myds_v41_gpu_topk_threshold_candidate(myds_metal_buffer_t *ids,
    const myds_metal_buffer_t *scores,uint32_t n_cols,myds_v41_gpu_scratch_t *scratch);
void myds_v41_gpu_inject_topk_threshold_unavailable(bool enabled);
/* 按行批量 + 逐行因果宽度（与 kernel_argsort_f32_i32_desc_causal 配套）：
 * ids 是 n_rows × top 行主序，scores 是 n_rows × n_cols 行主序；
 * 中间结果用 scratch 里的 chunk_tmp_a/b（大小为 MYDS_V41_INDEXER_BATCH_FLOATS），
 * 要求 n_rows × (npr==1 ? top : n_cols) ≤ 该预算。 */
int myds_v41_gpu_topk_causal(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,uint32_t n_rows,uint32_t causal_start,uint32_t causal_ratio,
    myds_v41_gpu_scratch_t *scratch);
/* Opt-in MYDS_INDEXER_TOPK_PREFIX_MERGE=1 preserves original partitions, row
 * IDs and merge ties. Multi-stage top<=512 calls classify visible NaNs on GPU;
 * NaN rows keep dense merging. Missing optional PSOs/allocation failures fall
 * back before sort writes. Counters describe host encoding, not GPU traffic or
 * timing. Write slots are planned dense/finite-row prefix destinations: NaN
 * rows dynamically write the dense layout and are not synchronously read back. */
typedef struct {
    uint64_t requests, encoded, unavailable, classified_rows, merge_dispatches;
    uint64_t dense_write_slots, prefix_write_slots;
} myds_v41_gpu_topk_prefix_counters_t;
void myds_v41_gpu_topk_prefix_counters(myds_v41_gpu_topk_prefix_counters_t *out,bool reset);
/* Diagnostic preflight-unavailable injection; leaves reference sorting usable. */
void myds_v41_gpu_inject_topk_prefix_unavailable(bool enabled);
/* MYDS_INDEXER_TOPK_NAN_FUSE accepts exactly 0/1 (absent=0). With an eligible
 * prefix call, 1 replaces the separate NaN scan with an independent sort PSO
 * that writes one IEEE-bit flag per original partition, including empty causal
 * partitions. Merge checks the whole row before using the original body.
 * Missing candidate PSOs or flag allocation fall back to dense before writes;
 * a dispatch failure after encoding starts returns 0. Original partition size,
 * comparison/tie order and the prefix-counter ABI are unchanged. */
#define MYDS_V41_GPU_HAS_TOPK_NAN_FUSE_COUNTERS 1
typedef struct {
    uint64_t requests, encoded, unavailable, classified_partitions;
} myds_v41_gpu_topk_nan_fuse_counters_t;
/* Host encoding counters only; no GPU NaN/dense-fallback count is read back.
 * classified_partitions counts flag slots written by encoded sorts (rows*npr),
 * including empty causal partitions; encoded means the complete call encoded. */
void myds_v41_gpu_topk_nan_fuse_counters(myds_v41_gpu_topk_nan_fuse_counters_t *out,bool reset);
enum { MYDS_TOPK_NAN_FUSE_UNAVAILABLE_SORT=1u, MYDS_TOPK_NAN_FUSE_UNAVAILABLE_MERGE=2u };
/* Diagnostic fixture only, between completed batches: independently request a
 * nonexistent candidate sort and/or candidate merge PSO during preflight. */
void myds_v41_gpu_inject_topk_nan_fuse_unavailable(uint32_t mask);
int myds_v41_gpu_topkmask(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top);
/* 按行批量版：mask 是 n_rows × mask_stride（float，行主序），n_cols 是每行要清/写的列数。 */
int myds_v41_gpu_topkmask_rows(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top,uint32_t rows,uint32_t mask_stride);
/* ids must originate from topk with the same source_rows (trusted GPU indices). */
int myds_v41_gpu_gather(myds_metal_buffer_t *out,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *ids,uint32_t source_rows,uint32_t selected_rows);
int myds_v41_gpu_gather_bf16(myds_metal_buffer_t *out,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *ids,uint32_t source_rows,uint32_t selected_rows);
/* Fixed64 heads, width512, raw ring capacity128, selected_rows<=512.
 * Reference F16 staging + 32-way split-K FlashAttention, no output BF16/RoPE:
 * caller must round then inverse-RoPE the heads at the prescribed boundary. */
/* Explicit ordinary single-query candidate. MYDS_ATTN_PACKED_REGISTER accepts
 * absent/literal 0 (reference) or literal 1 (register kernel). Returns -1 for
 * invalid configuration: candidate requires physical NSG8 and packed reduce
 * absent/literal 1. Configuration is checked at scratch creation and every
 * public single-attention call before staging; change env only between graphs.
 * Explicit candidate PSO/dispatch failure returns 0 without reference fallback. */
#define MYDS_V41_GPU_HAS_ATTN_PACKED_REGISTER 1
int myds_v41_gpu_attention_packed_register_enabled(void);
typedef struct {
    uint64_t requests, encoded, failures, direct_encoded, staged_encoded;
} myds_v41_gpu_attention_packed_register_counters_t;
/* Process-wide relaxed host counters. Request counts explicit candidate calls
 * including validation failures; encoded counts an actual kernel dispatch even
 * if standalone completion subsequently fails. Decode deltas are obtained by
 * caller snapshots, rather than inferring Engine purpose from this low API. */
void myds_v41_gpu_attention_packed_register_counters(
    myds_v41_gpu_attention_packed_register_counters_t *out,bool reset);
/* Ordinary single-query reducer -> inverse rope_round fusion. Default off.
 * Strict MYDS_ATTN_ROPE_FUSE=1 requires packed-register1/physical8/reduce1;
 * malformed flags/configurations fail initialization. Explicit candidate
 * failure never retries the original. Generic APIs below stay intact.
 * The scoped APIs always initialize *rope_consumed=false; true means the
 * fused inverse RoPE/BF16 operation was successfully encoded. Outside ordinary
 * nontrace scope they use original attention and leave rope_consumed=false. Retain
 * the heads -> OutputA RAW join and skip only its separate rope_round dispatch.
 * Requests include predispatch failures; encoded counts the actual fused PSO,
 * even if standalone command completion subsequently fails. */
#define MYDS_V41_GPU_HAS_ATTN_ROPE_FUSE 1
int myds_v41_attention_rope_fuse_enabled(void);
void myds_v41_attention_rope_fuse_counters(myds_metal_decode_candidate_counters_t *out,bool reset);
int myds_v41_gpu_attention_rope_fuse(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,
    myds_v41_gpu_scratch_t *scratch,uint32_t pos,bool compressed,
    bool ordinary_allowed,bool trace_on,bool *rope_consumed);
int myds_v41_gpu_attention_direct_rope_fuse(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *selected_ids,
    uint32_t source_rows,uint32_t selected_rows,myds_v41_gpu_scratch_t *scratch,
    uint32_t pos,bool compressed,bool ordinary_allowed,bool trace_on,bool *rope_consumed);
int myds_v41_gpu_attention(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,
    myds_v41_gpu_scratch_t *scratch);
/* Fused direct attention: single-dispatch vectorized circular window copy and
 * direct BF16->F16 gather, eliminating intermediate selected_kv staging buffer
 * and reducing decode attention dispatches. */
int myds_v41_gpu_attention_direct(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *selected_ids,
    uint32_t source_rows,uint32_t selected_rows,myds_v41_gpu_scratch_t *scratch);
/* 整个 chunk 一次算完的 attention（chunked prefill 用）。 */
int myds_v41_gpu_attention_chunk(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *chunk_kv,
    const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *ids,
    myds_metal_buffer_t *gathered,
    uint32_t c_len,uint32_t n_comp,uint32_t top_k,
    uint32_t base_pos,uint32_t c_start,uint32_t ratio,myds_v41_gpu_scratch_t *s);
/* These BF16 variants accept packed uint16 cache/comp_kv values; every other
 * buffer and the arithmetic/reduction order matches the corresponding F32 API. */
int myds_v41_gpu_attention_chunk_bf16(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *chunk_kv,
    const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *ids,
    myds_metal_buffer_t *gathered,
    uint32_t c_len,uint32_t n_comp,uint32_t top_k,
    uint32_t base_pos,uint32_t c_start,uint32_t ratio,myds_v41_gpu_scratch_t *s);
#endif
