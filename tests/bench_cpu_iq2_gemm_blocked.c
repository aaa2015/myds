/* 分块融合：IQ2_XXS 反量化 + GEMM，控制内存占用
 *
 * 现状问题：整专家反量化需要 47.19 MB float 缓冲（3.04 MB 压缩 → 15.5x）
 *   384 专家同时处理会爆内存。
 *
 * 方案：按行块（row block）处理：
 *   for each row_block of R rows:
 *       dequant R rows -> 临时缓冲 (R × 5120 × 4 B)
 *       GEMM (R × 5120) × (5120 × batch)
 *
 * 验证：R 取不同值时，总时间是否仍接近 2.19 ms。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <pthread.h>

#include "iq2_tables.h"

#define QK_K 256
typedef struct { uint16_t d; uint16_t qs[32]; } block_iq2_xxs;

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static inline float h2f(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}
static inline void dequant_block(const block_iq2_xxs *b, float *out) {
    const float d = h2f(b->d);
    for (int il = 0; il < 16; il++) {
        const int ib32 = il / 2, ilp = il % 2;
        const uint16_t *q2 = b->qs + 4 * ib32;
        const uint32_t g32 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t s32 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const uint8_t *a8 = (const uint8_t *)&g32;
        const float dl = d * (0.5f + (float)(s32 >> 28)) * 0.25f;
        const uint64_t g0 = IQ2_GRID[a8[2*ilp+0]];
        const uint64_t g1 = IQ2_GRID[a8[2*ilp+1]];
        const uint8_t s0 = IQ2_KSIGNS[(s32 >> (14*ilp)) & 127];
        const uint8_t s1 = IQ2_KSIGNS[(s32 >> (14*ilp+7)) & 127];
        const uint8_t *p0 = (const uint8_t *)&g0, *p1 = (const uint8_t *)&g1;
        float *o = out + il * 16;
        for (int i = 0; i < 8; i++) {
            o[i]     = dl * (float)p0[i] * ((s0 & IQ2_KMASK[i]) ? -1.0f : 1.0f);
            o[i + 8] = dl * (float)p1[i] * ((s1 & IQ2_KMASK[i]) ? -1.0f : 1.0f);
        }
    }
}
static void dequant_row(const block_iq2_xxs *row, int nb, float *out) {
    for (int b = 0; b < nb; b++) dequant_block(row + b, out + b * QK_K);
}

/* 分段 worker：每个线程处理若干行块 */
typedef struct {
    const block_iq2_xxs *w;
    float *wf;
    int nb, cols;
    int r0, r1;
} dq_t;
static void *dq_worker(void *p) {
    dq_t *d = (dq_t *)p;
    for (int r = d->r0; r < d->r1; r++)
        dequant_row(d->w + (size_t)r*d->nb, d->nb, d->wf + (size_t)r*d->cols);
    return NULL;
}

int main(int argc, char **argv) {
    int rows = 2304, cols = 5120;
    int batch   = argc > 1 ? atoi(argv[1]) : 256;
    int rblk    = argc > 2 ? atoi(argv[2]) : 2304;   /* 每次处理多少行 */
    int nthr    = argc > 3 ? atoi(argv[3]) : 16;
    int rep     = argc > 4 ? atoi(argv[4]) : 3;

    int nb = cols / QK_K;
    size_t nblk = (size_t)rows * nb;
    size_t wbytes = nblk * sizeof(block_iq2_xxs);
    size_t tmpbytes = (size_t)rblk * cols * 4;

    printf("  IQ2_XXS 分块融合: %dx%d batch=%d 块=%d行 线程=%d\n",
           rows, cols, batch, rblk, nthr);
    printf("    全专家 float 需 %.1f MB；分块只需 %.1f MB（%.0f%%）\n",
           (double)rows*cols*4/1e6, tmpbytes/1e6,
           100.0*tmpbytes/((double)rows*cols*4));

    block_iq2_xxs *w = aligned_alloc(64, wbytes);
    float *tmp = aligned_alloc(64, tmpbytes);
    float *B = aligned_alloc(64, (size_t)cols*batch*4);
    float *C = aligned_alloc(64, (size_t)rows*batch*4);
    if (!w || !tmp || !B || !C) { printf("  alloc fail\n"); return 1; }
    for (size_t i = 0; i < nblk; i++) {
        w[i].d = 0x2119;
        for (int k = 0; k < 32; k++) w[i].qs[k] = (uint16_t)((i*7919 + k*104729) & 0xFFFF);
    }
    for (size_t i = 0; i < (size_t)cols*batch; i++) B[i] = 0.001f*((i*37)%100);

    /* 正确性：分块 vs 全量 */
    {
        float *full = aligned_alloc(64, (size_t)rows*cols*4);
        for (int r = 0; r < rows; r++) dequant_row(w + (size_t)r*nb, nb, full + (size_t)r*cols);
        int bad = 0;
        for (int r0 = 0; r0 < rows; r0 += rblk) {
            int r1 = (r0 + rblk > rows) ? rows : r0 + rblk;
            for (int r = r0; r < r1; r++)
                dequant_row(w + (size_t)r*nb, nb, tmp + (size_t)(r-r0)*cols);
            for (int r = r0; r < r1; r++)
                if (memcmp(full + (size_t)r*cols, tmp + (size_t)(r-r0)*cols, cols*4) != 0) bad++;
        }
        printf("    正确性：分块 vs 全量，%d 行不同 %s\n", bad, bad==0?"✅":"❌");
        free(full);
        if (bad) return 1;
    }

    double best = 1e9;
    for (int rp = 0; rp < rep; rp++) {
        double t0 = now();
        for (int r0 = 0; r0 < rows; r0 += rblk) {
            int r1 = (r0 + rblk > rows) ? rows : r0 + rblk;
            int n = r1 - r0;
            /* 并行反量化 n 行 */
            pthread_t th[32]; dq_t dq[32];
            for (int t = 0; t < nthr; t++) {
                dq[t] = (dq_t){ w + (size_t)r0*nb, tmp, nb, cols,
                                n*t/nthr, n*(t+1)/nthr };
                pthread_create(&th[t], NULL, dq_worker, &dq[t]);
            }
            for (int t = 0; t < nthr; t++) pthread_join(th[t], NULL);
            /* GEMM 本块 */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, batch, cols,
                        1.0f, tmp, cols, B, batch, 0.0f, C + (size_t)r0*batch, batch);
        }
        double dt = now()-t0;
        if (dt < best) best = dt;
    }
    double gflops = 2.0*rows*cols*batch/best/1e9;
    printf("\n  ── 结果 ──\n");
    printf("    总耗时 : %.2f ms\n", best*1000);
    printf("    算力   : %.0f GFLOPS\n", gflops);
    double gpu_ms = 0.58;
    printf("    vs GPU : %.1fx\n", best*1000/gpu_ms);
    free(w); free(tmp); free(B); free(C);
    return 0;
}
