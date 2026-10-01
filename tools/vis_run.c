/*
 * vis_run.c — 跑一遍视觉编码器并转储结果
 *
 * 用法: vis_run <vision.gguf> <image> <out.f32> [dump_prefix] [--metal-dir DIR]
 *
 * 做的事：加载权重 → 解码+预处理图片 → 编码 → 写出 [out_rows, projection] 的 F32。
 * 给了 dump_prefix 就同时把各阶段中间量写成 <prefix>.<stage>.f32，便于逐层对拍。
 *
 * 输出末尾打印自检摘要（有限性、范数、行数），便于一眼看出「是不是在算东西」。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_metal.h"
#include "myds_visfwd.h"
#include "myds_vispre.h"
#include "myds_visw.h"

int main(int argc, char **argv) {
    /* ★ 不缓冲：Metal 的断言是 abort()，块缓冲的 stdout 会被丢掉 ——
     *   第一版就是这样把「失败阶段」这条最有用的信息弄丢了。 */
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 4) {
        fprintf(stderr, "usage: %s <vision.gguf> <image> <out.f32> [dump_prefix] [--metal-dir DIR]\n",
                argv[0]);
        return 2;
    }
    const char *gguf = argv[1], *img = argv[2], *outpath = argv[3];
    const char *dump = argc > 4 && argv[4][0] != '-' ? argv[4] : NULL;

    const char *metal_dir = getenv("MYDS_METAL_DIR");
    for (int i = 4; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--metal-dir")) metal_dir = argv[i + 1];
    }
    if (!metal_dir || !*metal_dir) metal_dir = "metal";

    char err[512] = {0};
    if (!myds_metal_init(metal_dir)) {
        fprintf(stderr, "myds_metal_init(%s) failed\n", metal_dir);
        return 1;
    }

    myds_vis_weights *w = NULL;
    if (!myds_vis_weights_load(gguf, &w, err, sizeof(err))) {
        fprintf(stderr, "加载权重失败: %s\n", err); return 1;
    }
    printf("权重: %u 层 · dim %u · projection %u · F32 %.2f GiB\n",
           w->spec.block_count, w->spec.dim, w->spec.projection,
           (double)w->spec.f32_bytes / 1073741824.0);

    myds_vis_image image;
    if (!myds_vis_image_decode_file(&image, img, err, sizeof(err))) {
        fprintf(stderr, "解码失败: %s\n", err); return 1;
    }
    myds_vis_patches pat;
    if (!myds_vis_preprocess(&pat, &image, err, sizeof(err))) {
        fprintf(stderr, "预处理失败: %s\n", err); return 1;
    }
    printf("图片: %ux%u → 画布 %ux%u · patch 网格 %ux%u (%u 行) · llm 网格 %ux%u\n",
           image.width, image.height, pat.padded_width, pat.padded_height,
           pat.grid_width, pat.grid_height, pat.patch_count,
           pat.llm_grid_width, pat.llm_grid_height);

    /* 图片 token 布局（顺带验证一下） */
    myds_vis_layout lay;
    if (!myds_vis_layout_build(&lay, pat.llm_grid_height, pat.llm_grid_width,
                               err, sizeof(err))) {
        fprintf(stderr, "布局失败: %s\n", err); return 1;
    }
    printf("图片区 token 数: %u（其中图片 %u）\n", lay.token_count, lay.image_count);

    const uint32_t P = w->spec.projection;
    const size_t cap = (size_t)4096 * P;
    float *out = malloc(cap * sizeof(float));
    if (!out) { fprintf(stderr, "输出缓冲分配失败\n"); return 1; }

    if (dump) myds_vis_fwd_set_dump(dump);
    uint32_t out_rows = 0;
    if (!myds_vis_encode(w, &pat, out, &out_rows, err, sizeof(err))) {
        fprintf(stderr, "编码失败: %s\n", err); return 1;
    }
    printf("编码完成: 输出 %u 行 × %u 维\n", out_rows, P);

    /* 自检：有限性 + 范数 + 逐行均值（一行全同 ⇒ 大概是没接上） */
    size_t n = (size_t)out_rows * P;
    size_t bad = 0; double sum = 0.0, sumsq = 0.0; float mn = 1e30f, mx = -1e30f;
    for (size_t i = 0; i < n; i++) {
        float x = out[i];
        if (!isfinite(x)) { bad++; continue; }
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        sum += x; sumsq += (double)x * x;
    }
    const double mean = sum / (double)n;
    const double var = sumsq / (double)n - mean * mean;
    printf("自检: 非有限值 %zu · min %.6g · max %.6g · 均值 %.6g · 标准差 %.6g\n",
           bad, (double)mn, (double)mx, mean, sqrt(var > 0 ? var : 0));

    /* 行间是否真的不同（否则说明注入/计算没生效） */
    double rowdiff = 0.0;
    if (out_rows > 1) {
        for (uint32_t d = 0; d < P; d++)
            rowdiff += fabs((double)out[d] - (double)out[P + d]);
        rowdiff /= P;
    }
    printf("自检: 相邻行平均绝对差 %.6g\n", rowdiff);

    FILE *f = fopen(outpath, "wb");
    if (!f) { fprintf(stderr, "打不开 %s\n", outpath); return 1; }
    fwrite(&out_rows, 4, 1, f);
    fwrite(&P, 4, 1, f);
    fwrite(out, sizeof(float), n, f);
    fclose(f);
    printf("已写出: %s (%zu 个 float)\n", outpath, n);

    free(out);
    myds_vis_layout_free(&lay);
    myds_vis_patches_free(&pat);
    myds_vis_image_free(&image);
    myds_vis_weights_free(w);
    myds_metal_cleanup();
    return bad ? 1 : 0;
}
