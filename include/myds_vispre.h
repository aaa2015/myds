/*
 * myds_vispre.h — 视觉编码器的图片解码与预处理
 *
 * 目标：把一张 JPEG/PNG 变成 DeepSeek V4.1 视觉编码器要的 patch 张量，
 *       并把 patch 排成语言模型能吞的 token 序列骨架。
 *
 * ★ 与 ds4 的关系：本文件是 ds4 侧（ds4_image.c / ds4_image.h）
 *   DeepSeek-V4 vision 通路的【忠实移植】，算法逐行对应，不改算术次序，
 *   以便与 ds4 的输出【逐字节】对拍（tests/test_vispre.c 就是干这个的）。
 *   移植而不是链接 ds4，是因为两边引擎布局不同；但对拍保证行为一致。
 *
 * 处理链（与 ds4 一致）：
 *   解码(JPEG/PNG + EXIF 方向) → 计划尺寸 → min_pixels 放大 → 对齐到 14
 *   → token 预算收敛 → 居中贴到画布(底色 127) → 双三次缩放
 *   → 归一化 (v/127.5 - 1) → 按 14×14 切 patch，通道优先
 */
#ifndef MYDS_VISPRE_H
#define MYDS_VISPRE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 与 ds4 相同的上限，避免解码炸弹 */
#define MYDS_VIS_MAX_ENCODED_BYTES (64u * 1024u * 1024u)
#define MYDS_VIS_MAX_DIMENSION     16384u
#define MYDS_VIS_MAX_PIXELS        (64u * 1024u * 1024u)

/* patch 边长与下采样倍率（与 GGUF 元数据 deepseek4-vision.* 一致） */
#define MYDS_VIS_PATCH       14u
#define MYDS_VIS_DOWNSAMPLE   3u

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t *rgb;                   /* width*height*3 */
} myds_vis_image;

typedef struct {
    uint32_t content_width;         /* 内容真实尺寸（缩放后，未贴边） */
    uint32_t content_height;
    uint32_t padded_width;          /* 画布尺寸，14 的倍数 */
    uint32_t padded_height;
    uint32_t grid_width;            /* padded / 14 */
    uint32_t grid_height;
    uint32_t llm_grid_width;        /* 语言模型侧网格（/3 之后的） */
    uint32_t llm_grid_height;
    uint32_t patch_count;           /* grid_height * grid_width */
    float *patches;                 /* patch_count * 3 * 14 * 14，通道优先 */
} myds_vis_patches;

/* 图片 token 的类型；数值与 ds4 的 ds4_deepseek4_image_token_type 一致 */
typedef enum {
    MYDS_VIS_TOKEN_START   = 0,
    MYDS_VIS_TOKEN_PAD     = 1,
    MYDS_VIS_TOKEN_IMAGE   = 2,
    MYDS_VIS_TOKEN_NEWLINE = 3,
    MYDS_VIS_TOKEN_END     = 4,
} myds_vis_token_type;

typedef struct {
    uint32_t token_count;           /* 整段图片区占的 token 数 */
    uint32_t image_count;           /* 其中真正的图片 token 数（= grid_h*grid_w） */
    uint8_t *types;                 /* token_count 个 myds_vis_token_type */
    uint32_t *perm;                 /* image_count 个：编码器输出序 → 行内位置 */
} myds_vis_layout;

/* ── 解码 ── */
int  myds_vis_image_decode_memory(myds_vis_image *out, const uint8_t *encoded,
                                  size_t encoded_len, char *error, size_t error_cap);
int  myds_vis_image_decode_file(myds_vis_image *out, const char *path,
                                char *error, size_t error_cap);
void myds_vis_image_free(myds_vis_image *image);

/* ── 预处理（DeepSeek V4.1）── */
int  myds_vis_preprocess(myds_vis_patches *out, const myds_vis_image *image,
                         char *error, size_t error_cap);
void myds_vis_patches_free(myds_vis_patches *patches);

/* ── 图片 token 序列骨架（V4.1 版）──
 * ★ 传的是【下采样后的 llm 网格】(grid+2)/3，不是 patch 网格。
 * ★ V4.1 的排布很朴素：[START] + 每行(width×IMAGE + NEWLINE) + [END]，
 *   且 perm 是恒等映射 —— 没有 v4.0(vision-exp) 那套 compress_pad / 成对重排。
 *   第一版我误移植了 v4.0 的复杂布局，被对拍门禁当场抓住。 */
int  myds_vis_layout_build(myds_vis_layout *out, uint32_t llm_grid_height,
                           uint32_t llm_grid_width,
                           char *error, size_t error_cap);
void myds_vis_layout_free(myds_vis_layout *layout);

/* 自检用：把 patches 的校验和算出来（对拍时比对，避免打印几十 MB） */
uint64_t myds_vis_patches_hash(const myds_vis_patches *patches);

#ifdef __cplusplus
}
#endif
#endif /* MYDS_VISPRE_H */
