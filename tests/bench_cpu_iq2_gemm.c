/* CPU: IQ2_XXS 反量化 + GEMM 的完整链路测速
 *
 * 目标：验证「CPU 做 prefill MoE」是否可能。
 *   - 反量化：IQ2_XXS → float（用真实查表，与 GPU 逐位一致）
 *   - GEMM：反量化结果 × 激活矩阵（走 Accelerate/AMX）
 *
 * 关键数字：反量化 + GEMM 的总时间，对比 GPU 单专家时间。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
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

/* 反量化一个块到 256 float（与 metal 逐位一致：16 个 il，每个 16 元素） */
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

/* 反量化整行（cols 个元素） */
static void dequant_row(const block_iq2_xxs *row, int nb, float *out) {
    for (int b = 0; b < nb; b++) dequant_block(row + b, out + b * QK_K);
}

typedef struct { const block_iq2_xxs *w; float *wf; int nb, r0, r1, cols; } dq_t;
static void *dq_worker(void *p) {
    dq_t *d = (dq_t *)p;
    for (int r = d->r0; r < d->r1; r++)
        dequant_row(d->w + (size_t)r*d->nb, d->nb, d->wf + (size_t)r*d->cols);
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = argc > 3 ? atoi(argv[3]) : 1;
    int rows = 2304;       /* expert_dim */
    int cols = 5120;       /* dim */
    int batch = argc > 1 ? atoi(argv[1]) : 256;   /* token 数 */
    int rep = argc > 2 ? atoi(argv[2]) : 3;

    int nb = cols / QK_K;
    size_t nblk = (size_t)rows * nb;
    size_t wbytes = nblk * sizeof(block_iq2_xxs);
    size_t fbytes = (size_t)rows * cols * 4;
    printf("  IQ2_XXS 反量化+GEMM: %d×%d, batch=%d\n", rows, cols, batch);
    printf("    压缩权重 %.2f MB → float %.2f MB (%.1fx)\n",
           wbytes/1e6, fbytes/1e6, (double)fbytes/wbytes);

    block_iq2_xxs *w = aligned_alloc(64, wbytes);
    float *wf = aligned_alloc(64, fbytes);
    float *B = aligned_alloc(64, (size_t)cols * batch * 4);
    float *C = aligned_alloc(64, (size_t)rows * batch * 4);
    if (!w || !wf || !B || !C) { printf("  alloc fail\n"); return 1; }
    for (size_t i = 0; i < nblk; i++) {
        w[i].d = 0x2119;
        for (int k = 0; k < 32; k++) w[i].qs[k] = (uint16_t)((i*7919 + k*104729) & 0xFFFF);
    }
    for (size_t i = 0; i < (size_t)cols*batch; i++) B[i] = 0.001f*((i*37)%100);

    printf("\n  ── 阶段 1：反量化（%d 线程）──\n", nthreads);
    double dq_ms = 1e9;
    for (int rp = 0; rp < 3; rp++) {
        pthread_t th[32]; dq_t dq[32];
        double t0 = now();
        for (int t = 0; t < nthreads; t++) {
            dq[t] = (dq_t){ w, wf, nb, rows*t/nthreads, rows*(t+1)/nthreads, cols };
            pthread_create(&th[t], NULL, dq_worker, &dq[t]);
        }
        for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
        double dt = (now()-t0)*1000;
        if (dt < dq_ms) dq_ms = dt;
    }
    printf("    %d 线程反量化 : %.2f ms  (%.1f GB/s 压缩流)\n",
           nthreads, dq_ms, wbytes/1e9/(dq_ms/1000));

    printf("\n  ── 阶段 2：GEMM（float，走 AMX）──\n");
    double best = 1e9;
    for (int r = 0; r < rep; r++) {
        double t1 = now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, batch, cols,
                    1.0f, wf, cols, B, batch, 0.0f, C, batch);
        double dt = now()-t1;
        if (dt < best) best = dt;
    }
    double gemm_ms = best*1000;
    double gflops = 2.0*rows*cols*batch/best/1e9;
    printf("    GEMM         : %.2f ms  (%.0f GFLOPS)\n", gemm_ms, gflops);

    printf("\n  ═══ 合计 ═══\n");
    double total = dq_ms + gemm_ms;
    printf("    反量化 + GEMM: %.2f ms（反量化占 %.0f%%）\n", total, 100*dq_ms/total);

    /* GPU 对照：单专家 prefill 的 MoE（IQ2 gate+up+down） */
    double gpu_ms = 0.58;    /* 从 §21 kprof 推算的单层单专家量级 */
    printf("\n  ═══ 与 GPU 对比 ═══\n");
    printf("    GPU 单专家（IQ2, 含 gate+up）: ~%.2f ms\n", gpu_ms);
    printf("    CPU 反量化+GEMM              : %.2f ms（%.1fx）\n", total, total/gpu_ms);
    printf("    ⇒ %s\n", total < gpu_ms*4 ? "CPU 慢但可并行，有讨论价值" : "**CPU 太慢，不可行**");

    free(w); free(wf); free(B); free(C);
    return 0;
}
