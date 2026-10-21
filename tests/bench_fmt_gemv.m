/* GPU: 同一形状、不同量化格式的 GEMV 吞吐对比
 *
 * 回答：Q2/IQ2 升到 Q4，性能会不会提升？
 *
 * 要点：
 *   - 字节数：IQ2_XXS 2.0625 bit, Q2_K 2.625, Q4_K 4.5
 *   - 如果 IQ2 是 ALU 受限（查表开销大），Q4 可能反而快
 *   - 如果都是带宽受限，Q4 必然慢（字节多）
 *
 * 判据：实测有效带宽 (bytes / time)
 *   - IQ2 带宽 << Q4 带宽 ⇒ IQ2 被 ALU 限制 ⇒ 升 Q4 可能更快
 *   - 两者带宽接近 ⇒ 都带宽受限 ⇒ 升 Q4 按字节比例变慢
 */
#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1e6 + (double)ts.tv_nsec*1e-3;
}

/* 每种格式的块大小与每块字节 */
static void fmt_info(int t, size_t *block, size_t *bytes, const char **name, double *bpp) {
    if (t == GGML_TYPE_IQ2_XXS) { *block=256; *bytes=66;  *name="IQ2_XXS"; *bpp=2.0625; }
    else if (t == GGML_TYPE_Q2_K) { *block=256; *bytes=84; *name="Q2_K";   *bpp=2.625;  }
    else if (t == GGML_TYPE_Q4_K) { *block=256; *bytes=144;*name="Q4_K";   *bpp=4.5;    }
    else if (t == GGML_TYPE_Q6_K) { *block=256; *bytes=210;*name="Q6_K";   *bpp=6.5625; }
    else if (t == GGML_TYPE_Q8_0) { *block=32;  *bytes=34; *name="Q8_0";   *bpp=8.5;    }
    else { *block=256; *bytes=144; *name="?"; *bpp=4.5; }
}

static void fill_blocks(void *ptr, size_t nblocks, size_t bytes_per_block, int type) {
    uint8_t *p = (uint8_t *)ptr;
    for (size_t b = 0; b < nblocks; b++) {
        uint8_t *blk = p + b*bytes_per_block;
        for (size_t i = 0; i < bytes_per_block; i++)
            blk[i] = (uint8_t)((b*37 + i*101) & 0xff);
        /* d / dmin 设成合理 fp16（约 1.0 和 0.5），避免 inf/nan 拖慢 */
        if (type == GGML_TYPE_IQ2_XXS) {
            uint16_t d = 0x2c00;      /* ~0.0625 */
            memcpy(blk, &d, 2);
        } else if (type == GGML_TYPE_Q2_K) {
            uint16_t d = 0x2c00, dm = 0x2800;   /* Q2_K: d, dmin, scales[16], qs[64] */
            memcpy(blk, &d, 2); memcpy(blk+2, &dm, 2);
        } else if (type == GGML_TYPE_Q4_K) {
            uint16_t d = 0x2c00, dm = 0x2800;
            memcpy(blk, &d, 2); memcpy(blk+2, &dm, 2);
        } else if (type == GGML_TYPE_Q8_0) {
            uint16_t d = 0x2c00;
            memcpy(blk, &d, 2);
        }
    }
}

int main(int argc, char **argv) {
    uint32_t out_dim = argc > 1 ? (uint32_t)atoi(argv[1]) : 2304;
    uint32_t in_dim  = argc > 2 ? (uint32_t)atoi(argv[2]) : 5120;
    int reps = argc > 3 ? atoi(argv[3]) : 200;

    int types[] = { GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    const int NT = sizeof(types)/sizeof(types[0]);

    printf("  GPU GEMV 格式对比: out=%u in=%u reps=%d\n", out_dim, in_dim, reps);
    printf("  %-9s %10s %10s %12s %12s\n", "格式", "字节(MB)", "耗时(us)", "有效GB/s", "相对IQ2");

    double base_t = 0;
    for (int k = 0; k < NT; k++) {
        int t = types[k];
        size_t block, bpb; const char *name; double bpp;
        fmt_info(t, &block, &bpb, &name, &bpp);
        size_t nblocks_row = in_dim / block;
        size_t row_bytes = nblocks_row * bpb;
        size_t w_bytes = row_bytes * out_dim;
        size_t total_blocks = w_bytes / bpb;

        myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
        myds_metal_buffer_t *in = myds_metal_buffer_alloc(in_dim*sizeof(float));
        myds_metal_buffer_t *out = myds_metal_buffer_alloc(out_dim*sizeof(float));
        if (!w || !in || !out) { printf("  alloc fail %s\n", name); continue; }

        fill_blocks(myds_metal_buffer_data(w), total_blocks, bpb, t);
        float *ip = (float*)myds_metal_buffer_data(in);
        for (uint32_t i = 0; i < in_dim; i++)
            ip[i] = (float)(((int)(i*43+97)%201)-100)/100.0f;

        /* 预热 */
        for (int i = 0; i < 20; i++) {
            myds_metal_begin_batch();
            myds_metal_v41_gemv_bf16(out, w, in, out_dim, in_dim, t);
            myds_metal_commit_batch();
        }
        myds_metal_synchronize();

        double best = 1e18;
        for (int r = 0; r < 5; r++) {
            double t0 = now_us();
            for (int i = 0; i < reps; i++) {
                myds_metal_begin_batch();
                myds_metal_v41_gemv_bf16(out, w, in, out_dim, in_dim, t);
                myds_metal_commit_batch();
            }
            myds_metal_synchronize();
            double dt = (now_us()-t0)/reps;
            if (dt < best) best = dt;
        }
        double gbs = (double)w_bytes / (best*1e-6) / 1e9;
        if (k == 0) base_t = best;
        printf("  %-9s %10.2f %10.2f %12.1f %11.2fx\n",
               name, w_bytes/1e6, best, gbs, base_t/best);

        myds_metal_buffer_free(w);
        myds_metal_buffer_free(in);
        myds_metal_buffer_free(out);
    }

    printf("\n  ═══ 判据 ═══\n");
    printf("    若 IQ2 的 GB/s 明显低于 Q4 ⇒ IQ2 被 ALU（查表）限制 ⇒ 升 Q4 可能更快\n");
    printf("    若两者 GB/s 接近     ⇒ 都带宽受限   ⇒ 升 Q4 按字节比例变慢\n");
    return 0;
}
