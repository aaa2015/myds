/*
 * myds_visw.h — 视觉编码器的权重加载（BF16 GGUF → F32 Metal 缓冲）
 *
 * 设计要点：
 *  · 分两段。probe() 只读元数据、【零分配】，把命名/维度/类型全查一遍，
 *    并算出需要多少内存；load() 才真分配。这样「加载器对不对」可以在
 *    小内存机器上全量验证，把大额分配留给内存够的目标机。
 *  · 权重一律转成 F32。理由见 tests/test_metal.c 的可行性结论：
 *    F16 路径的累加器是 simdgroup_half8x8，长点积会漂到 4e-3；
 *    F32 路径是 simdgroup_float8x8，落到 3.5e-5。
 *    而 BF16→F32 是【精确无损】的（BF16 就是 F32 的高 16 位，只差左移）。
 *    代价是内存翻倍（970MB → 1.94GB）—— 用 probe() 先算清楚再决定。
 *  · 校验照抄 ds4 的口径（arch / revision / checkpoint_variant / 张量数 / 各维），
 *    因为这些正是「拿错文件」时最早暴露的信号。
 */
#ifndef MYDS_VISW_H
#define MYDS_VISW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "myds_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MYDS_VIS_MAX_BLOCKS 64u

/* 只读元数据结论 */
typedef struct {
    uint32_t block_count;
    uint32_t dim;               /* embedding_length */
    uint32_t ffn;               /* feed_forward_length */
    uint32_t heads;             /* attention.head_count */
    uint32_t projection;        /* projection_length（语言模型维度） */
    uint32_t patch;             /* patch_size */
    uint32_t downsample;        /* downsample_ratio */
    uint32_t max_tokens;        /* image.max_tokens */
    uint32_t min_pixels;        /* image.min_pixels */
    uint32_t image_token_id;    /* image.token_id */
    float    rms_eps;
    uint32_t tensor_count;      /* 文件里的张量数 */
    uint64_t f32_bytes;         /* 全部权重转 F32 需要多少字节 */
    uint32_t param_count_m;     /* 参数量（百万），便于读 */
} myds_vis_spec;

/* 每个 block 的 8 个张量 */
typedef struct {
    myds_metal_buffer_t *norm1;
    myds_metal_buffer_t *qkv_weight;
    myds_metal_buffer_t *qkv_bias;
    myds_metal_buffer_t *attn_proj_weight;
    myds_metal_buffer_t *attn_proj_bias;
    myds_metal_buffer_t *norm2;
    myds_metal_buffer_t *mlp_w1;
    myds_metal_buffer_t *mlp_w2;
} myds_vis_layer_weights;

typedef struct myds_vis_weights {
    myds_vis_spec spec;
    myds_metal_buffer_t *patch_weight;      /* [588, dim] */
    myds_metal_buffer_t *patch_bias;        /* [dim] */
    myds_metal_buffer_t *post_norm;         /* vision.norm.weight [dim] */
    myds_metal_buffer_t *aligner_w1;        /* [9216, projection] */
    myds_metal_buffer_t *aligner_w1_bias;   /* [projection] */
    myds_metal_buffer_t *aligner_w2;        /* [projection, projection] */
    myds_metal_buffer_t *aligner_w2_bias;   /* [projection] */
    myds_metal_buffer_t *image_start;       /* [projection] */
    myds_metal_buffer_t *image_newline;     /* [projection] */
    myds_metal_buffer_t *image_end;         /* [projection] */
    myds_vis_layer_weights layer[MYDS_VIS_MAX_BLOCKS];
} myds_vis_weights;

/* 只读元数据 + 完整校验，不做任何分配。成功返回 1。 */
/* 同源校验：expected_revision 必须等于视觉 GGUF 的 general.source.revision。
 * V4.1 有多个已发布 checkpoint（df42c109 / 1bae0e10 …），写死某一个哈希会造成双向
 * 错误：更新 checkpoint 的正确文件被拒，而错贴旧标签的文件反而放行。 */
int myds_vis_weights_probe(const char *path, const char *expected_revision,
                           myds_vis_spec *out,
                           char *error, size_t error_cap);

/* 全量加载（BF16 → F32 → Metal 缓冲）。成功返回 1，*out 需用 free() 释放。 */
int myds_vis_weights_load(const char *path, const char *expected_revision,
                          myds_vis_weights **out,
                          char *error, size_t error_cap);
void myds_vis_weights_free(myds_vis_weights *w);

#ifdef __cplusplus
}
#endif
#endif /* MYDS_VISW_H */
