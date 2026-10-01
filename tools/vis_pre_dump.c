/*
 * vis_pre_dump.c — 图片预处理/布局的【对拍转储】工具
 *
 * 目的：把「解码 → 预处理 → 图片 token 布局」的结果写成规范化的二进制文件，
 *       以便与参照实现（ds4）的输出做【逐字节】比较。
 *
 * 为什么用两个可执行文件而不是一个测试程序直接比对：
 *   ds4_image.c 与 myds_vispre.c 都定义了 JPEG_IMPLEMENTATION / PNG_IMPLEMENTATION，
 *   链到同一个二进制会符号冲突。既然算法要与 ds4 完全一致，最干净的判据是
 *   「两边的输出文件必须一模一样」—— 这比在测试里调两套 API 更强，也更好查错。
 *
 * 用法: vis_pre_dump <image.jpg|png> <out.bin>
 * 编译:
 *   本仓实现：cc ... -o bin/vis_pre_dump
 *   参照实现：cc -DUSE_DS4_REF -I<ds4> ... ds4_image.c vis_pre_dump.c -o bin/vis_pre_dump_ref
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_DS4_REF
#include "ds4_image.h"
typedef ds4_image                     vis_image_t;
typedef ds4_deepseek4_image_patches   vis_patches_t;
typedef ds4_deepseek4_image_layout    vis_layout_t;
#define vis_decode_file(o, p, e, c)      ds4_image_decode_file((o), (p), (e), (c))
#define vis_image_free(i)                ds4_image_free(i)
#define vis_preprocess41(o, i, e, c)     ds4_image_preprocess_deepseek41((o), (i), (e), (c))
#define vis_patches_free(p)              ds4_deepseek4_image_patches_free(p)
#define vis_layout_build(o, h, w, e, c) ds4_deepseek41_image_layout_build((o), (h), (w), (e), (c))
#define vis_layout_free(l)               ds4_deepseek4_image_layout_free(l)
#else
#include "myds_vispre.h"
typedef myds_vis_image     vis_image_t;
typedef myds_vis_patches   vis_patches_t;
typedef myds_vis_layout    vis_layout_t;
#define vis_decode_file(o, p, e, c)      myds_vis_image_decode_file((o), (p), (e), (c))
#define vis_image_free(i)                myds_vis_image_free(i)
#define vis_preprocess41(o, i, e, c)     myds_vis_preprocess((o), (i), (e), (c))
#define vis_patches_free(p)              myds_vis_patches_free(p)
#define vis_layout_build(o, h, w, e, c) myds_vis_layout_build((o), (h), (w), (e), (c))
#define vis_layout_free(l)               myds_vis_layout_free(l)
#endif

static int usage(void) {
    fprintf(stderr, "usage: vis_pre_dump <image> <out.bin>\n");
    return 2;
}

int main(int argc, char **argv) {
    if (argc != 3) return usage();

    char err[512] = {0};
    vis_image_t img;
    if (!vis_decode_file(&img, argv[1], err, sizeof(err))) {
        /* 两边都必须同样地失败；打印可比较的失败信号 */
        printf("DECODE_FAIL\n");
        fprintf(stderr, "%s\n", err);
        return 0;
    }

    vis_patches_t pat;
    if (!vis_preprocess41(&pat, &img, err, sizeof(err))) {
        printf("PREPROCESS_FAIL\n");
        fprintf(stderr, "%s\n", err);
        vis_image_free(&img);
        return 0;
    }

    FILE *f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }

    /* —— 规范化二进制格式（两种实现必须写出完全相同的字节）—— */
    const char magic[8] = {'V','I','S','P','R','E','1','\0'};
    fwrite(magic, 1, 8, f);
    fwrite(&img.width, 4, 1, f);
    fwrite(&img.height, 4, 1, f);
    fwrite(img.rgb, 1, (size_t)img.width * img.height * 3u, f);

    fwrite(&pat.content_width, 4, 1, f);
    fwrite(&pat.content_height, 4, 1, f);
    fwrite(&pat.padded_width, 4, 1, f);
    fwrite(&pat.padded_height, 4, 1, f);
    fwrite(&pat.grid_width, 4, 1, f);
    fwrite(&pat.grid_height, 4, 1, f);
    fwrite(&pat.llm_grid_width, 4, 1, f);
    fwrite(&pat.llm_grid_height, 4, 1, f);
    fwrite(&pat.patch_count, 4, 1, f);
    fwrite(pat.patches, sizeof(float),
           (size_t)pat.patch_count * 3u * 14u * 14u, f);

    /* 布局：V4.1 用【下采样后的 llm 网格】 */
    {
        vis_layout_t lay;
        if (!vis_layout_build(&lay, pat.llm_grid_height, pat.llm_grid_width,
                              err, sizeof(err))) {
            printf("LAYOUT_FAIL\n");
        } else {
            fwrite(&lay.token_count, 4, 1, f);
            fwrite(&lay.image_count, 4, 1, f);
            fwrite(lay.types, 1, lay.token_count, f);
            fwrite(lay.perm, 4, lay.image_count, f);
            vis_layout_free(&lay);
        }
    }

    fclose(f);
    printf("OK w=%u h=%u grid=%ux%u patches=%u\n",
           img.width, img.height, pat.grid_width, pat.grid_height, pat.patch_count);

    vis_patches_free(&pat);
    vis_image_free(&img);
    return 0;
}
