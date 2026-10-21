/* CPU 版 IQ2_XXS GEMV 原型 —— 测 16 线程下的真实吞吐。
 *
 * 目的：判断「CPU 做部分专家」是否有意义。
 *   判据：CPU 在 16 线程下能跑到多少 GB/s，以及每层耗时。
 *
 * 解码语义严格对齐 metal/moe.metal 的 dequantize_iq2_xxs：
 *   struct block_iq2_xxs { half d; ushort qs[32]; };   // 66 B / 256 元素
 *   dl = d * (0.5 + (aux32_s >> 28)) * 0.25
 *   grid = iq2xxs_grid[aux8[2*il]], signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127]
 * 这里用简化的 grid/signs（不查表，用确定性伪随机替代），
 * 因为测的是【内存吞吐 + 解码算术量】，不是数值正确性。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

#define QK_K 256
#define NBLK (QK_K / 8)          /* 32 ushort per block */

typedef struct { uint16_t d; uint16_t qs[NBLK]; } block_iq2_xxs;   /* 66 B */

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* 一次 GEMV：(rows × cols) 的 IQ2_XXS 权重 × 一个 float 激活向量 → rows 个输出 */
typedef struct {
    const block_iq2_xxs *w;
    const float *x;
    float *out;
    int rows;
    int cols;          /* 必须是 256 的倍数 */
    int row_start, row_end;
    double t0, t1;
} work_t;

static void *gemv_worker(void *p) {
    work_t *w = (work_t *)p;
    const int nblk_per_row = w->cols / QK_K;
    w->t0 = now();
    for (int r = w->row_start; r < w->row_end; r++) {
        const block_iq2_xxs *row = w->w + (size_t)r * nblk_per_row;
        float acc = 0.0f;
        for (int b = 0; b < nblk_per_row; b++) {
            const block_iq2_xxs *blk = row + b;
            /* 解码 4 个 32-元素子块（对齐 metal 的 il 0..3 语义） */
            float d = (float)(blk->d & 0x7FFF) * 1e-4f;   /* 近似 half→float */
            for (int il = 0; il < 4; il++) {
                const int ib32 = il / 2;
                const uint16_t *q2 = blk->qs + 4 * ib32;
                uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
                uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
                const uint8_t *aux8 = (const uint8_t *)&aux32_g;
                float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
                /* 模拟两次 8 元素乘加（用 grid 索引做伪解码，保持算术量） */
                float s0 = 0.0f, s1 = 0.0f;
                const float *xs = w->x + b * QK_K + il * 64;
                for (int i = 0; i < 8; i++) {
                    s0 += xs[i]     * (float)(aux8[2 * il] + i);
                    s1 += xs[i + 8] * (float)(aux8[2 * il + 1] + i);
                }
                acc += dl * (s0 + s1);
            }
        }
        w->out[r] = acc;
    }
    w->t1 = now();
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 16;
    int rows = 2304;        /* expert_dim */
    int cols = 5120;        /* dim */
    int repeat = argc > 2 ? atoi(argv[2]) : 5;

    size_t nblk = (size_t)rows * (cols / QK_K);
    size_t wbytes = nblk * sizeof(block_iq2_xxs);
    printf("  CPU GEMV 原型：%d × %d，IQ2_XXS %.2f MB，%d 线程\n",
           rows, cols, wbytes / 1e6, nthreads);

    block_iq2_xxs *w = aligned_alloc(64, wbytes);
    float *x = aligned_alloc(64, (size_t)cols * 4);
    float *out = aligned_alloc(64, (size_t)rows * 4);
    if (!w || !x || !out) { printf("  alloc failed\n"); return 1; }
    /* 填数据：让解码走真实路径（非零） */
    for (size_t i = 0; i < nblk; i++) {
        w[i].d = (uint16_t)(0x3C00u + (i % 64));       /* 近似 1.0 附近 */
        for (int k = 0; k < NBLK; k++) w[i].qs[k] = (uint16_t)((i * 7919u + k * 104729u) & 0xFFFF);
    }
    for (int i = 0; i < cols; i++) x[i] = 0.001f * (float)(i % 97);

    /* 预热 */
    {
        pthread_t th[64]; work_t wk[64];
        for (int t = 0; t < nthreads; t++) {
            wk[t] = (work_t){ w, x, out, rows, cols,
                              rows * t / nthreads, rows * (t + 1) / nthreads, 0, 0 };
            pthread_create(&th[t], NULL, gemv_worker, &wk[t]);
        }
        for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
    }

    double best = 1e9;
    for (int r = 0; r < repeat; r++) {
        pthread_t th[64]; work_t wk[64];
        double t0 = now();
        for (int t = 0; t < nthreads; t++) {
            wk[t] = (work_t){ w, x, out, rows, cols,
                              rows * t / nthreads, rows * (t + 1) / nthreads, 0, 0 };
            pthread_create(&th[t], NULL, gemv_worker, &wk[t]);
        }
        for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }

    double gbs = wbytes / best / 1e9;
    printf("  ── 结果 ──\n");
    printf("    单次 GEMV: %.3f ms\n", best * 1000);
    printf("    带宽      : %.1f GB/s\n", gbs);
    printf("\n  ═══ 外推到 MoE ═══\n");
    /* 单专家 gate+up = 2 × 2304×5120 IQ2_XXS */
    double per_expert_mb = 2.0 * wbytes / 1e6;
    printf("    单专家 gate+up = %.2f MB，CPU 需 %.1f ms\n",
           per_expert_mb, per_expert_mb / (gbs * 1e3) * 1e3);
    /* 分工设想：CPU 做 2 个专家，GPU 做 4 个 */
    printf("    若 CPU 做 2/6 专家（gate+up）: %.1f ms/层\n",
           2 * per_expert_mb / (gbs * 1e3) * 1e3);
    printf("    GPU 现在整层 gate+up      : 0.132 ms\n");
    printf("    ⇒ CPU 做 2 专家%s\n",
           2 * per_expert_mb / (gbs * 1e3) * 1e3 < 0.132 * 2 / 6 * 2 ? " 来得及" : " **太慢**");
    free(w); free(x); free(out);
    return 0;
}
