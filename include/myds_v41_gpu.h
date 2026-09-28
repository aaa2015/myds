#ifndef MYDS_V41_GPU_H
#define MYDS_V41_GPU_H
#include "myds_metal.h"
/* Decode primitives for the fixed DeepSeek V4.1 graph. Float buffers hold
 * dequantized BF16/FP8/FP4 values; ids buffers hold int32. All operations are
 * GPU ordered within the caller's command batch. */
typedef struct myds_v41_gpu_scratch myds_v41_gpu_scratch_t;
myds_v41_gpu_scratch_t *myds_v41_gpu_scratch_create(uint32_t max_comp);
void myds_v41_gpu_scratch_free(myds_v41_gpu_scratch_t *scratch);
int myds_v41_gpu_rope(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool compressed,bool inverse);
/* 0 BF16, 1 FP8 E8M0 scales, 2 FP4 E8M0 scales, 3 FP4 E4M3 scales. */
int myds_v41_gpu_quant(myds_metal_buffer_t *x,uint32_t width,uint32_t rows,uint32_t mode);
/* Even positions cache kv/scores; odd positions write the pooled output. */
int myds_v41_gpu_pool2(myds_metal_buffer_t *out,const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores,myds_metal_buffer_t *previous_kv,
    myds_metal_buffer_t *previous_scores,uint32_t width,uint32_t pos);
int myds_v41_gpu_indexscores(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    uint32_t n_comp,uint32_t n_heads);
int myds_v41_gpu_candidates(myds_metal_buffer_t *blocks,const myds_metal_buffer_t *scores,
    uint32_t n_comp,uint32_t pos,uint32_t ratio);
int myds_v41_gpu_filter(myds_metal_buffer_t *scores,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t pos,uint32_t ratio);
int myds_v41_gpu_topk(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,myds_v41_gpu_scratch_t *scratch);
int myds_v41_gpu_topkmask(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top);
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
#endif
