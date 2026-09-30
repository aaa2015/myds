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
myds_v41_gpu_scratch_t *myds_v41_gpu_scratch_create(uint32_t max_comp);
void myds_v41_gpu_scratch_free(myds_v41_gpu_scratch_t *scratch);
int myds_v41_gpu_rope(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool compressed,bool inverse);
int myds_v41_gpu_rope_batch(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t start_pos,uint32_t rows,bool compressed,bool inverse);
/* 0 BF16, 1 FP8 E8M0 scales, 2 FP4 E8M0 scales, 3 FP4 E4M3 scales. */
int myds_v41_gpu_quant(myds_metal_buffer_t *x,uint32_t width,uint32_t rows,uint32_t mode);
/* Even positions cache kv/scores; odd positions write the pooled output. */
int myds_v41_gpu_pool2(myds_metal_buffer_t *out,const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores,myds_metal_buffer_t *previous_kv,
    myds_metal_buffer_t *previous_scores,uint32_t width,uint32_t pos);
int myds_v41_gpu_indexscores(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    uint32_t n_comp,uint32_t n_heads);
/* 整 chunk 批量版（n_tokens 行 × n_comp 列，行主序，行步长 n_comp）。只支持
 * n_heads == 32；每行的因果宽度 = (causal_start + row + 1) / causal_ratio，
 * 与逐 token 的 count 相同；行数不够的格子写 -INFINITY。 */
int myds_v41_gpu_indexscores_chunk(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    uint32_t n_comp,uint32_t n_heads,uint32_t n_tokens,
    uint32_t causal_start,uint32_t causal_ratio);
/* 批量 indexer 的缓冲（分数表、top-k 中间结果、块分数、块选择结果）。
 * 调用方只用它们，不拥有它们（随 scratch 释放）。 */
myds_metal_buffer_t *myds_v41_gpu_indexer_scores(const myds_v41_gpu_scratch_t *scratch);
myds_metal_buffer_t *myds_v41_gpu_indexer_blocks(const myds_v41_gpu_scratch_t *scratch);
myds_metal_buffer_t *myds_v41_gpu_indexer_block_selected(const myds_v41_gpu_scratch_t *scratch);
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
/* 按行批量 + 逐行因果宽度（与 kernel_argsort_f32_i32_desc_causal 配套）：
 * ids 是 n_rows × top 行主序，scores 是 n_rows × n_cols 行主序；
 * 中间结果用 scratch 里的 chunk_tmp_a/b（大小为 MYDS_V41_INDEXER_BATCH_FLOATS），
 * 要求 n_rows × (npr==1 ? top : n_cols) ≤ 该预算。 */
int myds_v41_gpu_topk_causal(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,uint32_t n_rows,uint32_t causal_start,uint32_t causal_ratio,
    myds_v41_gpu_scratch_t *scratch);
int myds_v41_gpu_topkmask(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top);
/* 按行批量版：mask 是 n_rows × mask_stride（float，行主序），n_cols 是每行要清/写的列数。 */
int myds_v41_gpu_topkmask_rows(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top,uint32_t rows,uint32_t mask_stride);
/* ids must originate from topk with the same source_rows (trusted GPU indices). */
int myds_v41_gpu_gather(myds_metal_buffer_t *out,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *ids,uint32_t source_rows,uint32_t selected_rows);
/* Fixed64 heads, width512, raw ring capacity128, selected_rows<=512.
 * Reference F16 staging + 32-way split-K FlashAttention, no output BF16/RoPE:
 * caller must round then inverse-RoPE the heads at the prescribed boundary. */
int myds_v41_gpu_attention(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,
    myds_v41_gpu_scratch_t *scratch);
/* 整个 chunk 一次算完的 attention（chunked prefill 用）。 */
int myds_v41_gpu_attention_chunk(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *chunk_kv,
    const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *ids,
    uint32_t c_len,uint32_t n_comp,uint32_t top_k,
    uint32_t base_pos,uint32_t c_start,uint32_t ratio,myds_v41_gpu_scratch_t *s);
#endif
