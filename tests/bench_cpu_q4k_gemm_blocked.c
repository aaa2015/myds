/* Q4_K 反量化 + GEMM（CPU），与 IQ2_XXS 版本对照
 *
 * Q4_K 块：144 B / 256 元素（4.5 bit）
 * 反量化：dl*(q&mask) - ml，无 grid/sign 查表 ⇒ 应远快于 IQ2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <pthread.h>

#define QK_K 256
typedef struct {
    uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128];
} block_q4_K;   /* 144 B */

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static inline float h2f(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}
static inline void get_scale_min_k4_just2(int j, int k, const uint8_t *q,
                                          uint8_t *sc0, uint8_t *sc1) {
    if (j < 4) { *sc0 = q[j+k] & 63; *sc1 = q[j+4+k] & 63; }
    else { *sc0 = (q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2);
           *sc1 = (q[j+4+k] >> 4)    | ((q[j+k]   & 0xc0) >> 2); }
}

/* 反量化一个 256 元素块 */
static void dequant_block(const block_q4_K *b, float *out) {
    for (int il = 0; il < 16; il++) {
        const uint8_t *q = b->qs + (il/4)*32 + 16*(il&1);
        int is = (il/4)*2, ilp = il & 3;
        uint8_t sc0, sc1;
        get_scale_min_k4_just2(is, ilp/2, b->scales, &sc0, &sc1);
        float d = (ilp < 2) ? h2f(b->d) : h2f(b->d) * (1.0f/16.0f);
        float mn = h2f(b->dmin);
        float dl = d * (float)sc0, ml = mn * (float)sc1;
        uint8_t mask = (ilp < 2) ? 0x0F : 0xF0;
        /* NEON: 16 个 u8 → f32，fma(q, dl, -ml) */
        uint8x16_t qv = vld1q_u8(q);
        uint8x16_t mv = vandq_u8(qv, vdupq_n_u8(mask));
        uint16x8_t lo = vmovl_u8(vget_low_u8(mv));
        uint16x8_t hi = vmovl_u8(vget_high_u8(mv));
        const float32x4_t dlv = vdupq_n_f32(dl), nml = vdupq_n_f32(-ml);
        float32x4_t g0 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),  dlv);
        float32x4_t g1 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), dlv);
        float32x4_t g2 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),  dlv);
        float32x4_t g3 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), dlv);
        vst1q_f32(out + il*16 + 0,  g0);
        vst1q_f32(out + il*16 + 4,  g1);
        vst1q_f32(out + il*16 + 8,  g2);
        vst1q_f32(out + il*16 + 12, g3);
    }
}
static void dequant_row(const block_q4_K *row, int nb, float *out) {
    for (int b = 0; b < nb; b++) dequant_block(row + b, out + b*QK_K);
}

typedef struct { const block_q4_K *w; float *wf; int nb, cols, r0, r1; } dq_t;
static void *dq_worker(void *p) {
    dq_t *d = (dq_t *)p;
    for (int r = d->r0; r < d->r1; r++)
        dequant_row(d->w + (size_t)r*d->nb, d->nb, d->wf + (size_t)r*d->cols);
    return NULL;
}

int main(int argc, char **argv) {
    int rows = 2304, cols = 5120;
    int batch = argc > 1 ? atoi(argv[1]) : 256;
    int rblk  = argc > 2 ? atoi(argv[2]) : 512;
    int nthr  = argc > 3 ? atoi(argv[3]) : 16;
    int rep   = argc > 4 ? atoi(argv[4]) : 3;

    int nb = cols / QK_K;
    size_t nblk = (size_t)rows * nb;
    size_t wbytes = nblk * sizeof(block_q4_K);
    size_t tmpb = (size_t)rblk * cols * 4;

    printf("  Q4_K 分块融合: %dx%d batch=%d 块=%d行 线程=%d\n", rows, cols, batch, rblk, nthr);
    printf("    压缩权重 %.2f MB；反量化后 %.1f MB (%.1fx)；分块缓冲 %.1f MB\n",
           wbytes/1e6, (double)rows*cols*4/1e6, (double)rows*cols*4/wbytes, tmpb/1e6);

    block_q4_K *w = aligned_alloc(64, wbytes);
    float *tmp = aligned_alloc(64, tmpb);
    float *B = aligned_alloc(64, (size_t)cols*batch*4);
    float *C = aligned_alloc(64, (size_t)rows*batch*4);
    if (!w||!tmp||!B||!C) { printf("  alloc fail\n"); return 1; }
    for (size_t i = 0; i < nblk; i++) {
        w[i].d = 0x2119; w[i].dmin = 0x2119;
        for (int k=0;k<12;k++) w[i].scales[k] = (uint8_t)((i*31+k*17)%64);
        for (int k=0;k<128;k++) w[i].qs[k] = (uint8_t)((i*7919+k*104729)&0xFF);
    }
    for (size_t i = 0; i < (size_t)cols*batch; i++) B[i] = 0.001f*((i*37)%100);

    double best = 1e9;
    for (int rp = 0; rp < rep; rp++) {
        double t0 = now();
        for (int r0 = 0; r0 < rows; r0 += rblk) {
            int r1 = (r0+rblk > rows) ? rows : r0+rblk;
            int n = r1 - r0;
            pthread_t th[32]; dq_t dq[32];
            for (int t = 0; t < nthr; t++) {
                dq[t] = (dq_t){ w + (size_t)r0*nb, tmp, nb, cols, n*t/nthr, n*(t+1)/nthr };
                pthread_create(&th[t], NULL, dq_worker, &dq[t]);
            }
            for (int t = 0; t < nthr; t++) pthread_join(th[t], NULL);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, batch, cols,
                        1.0f, tmp, cols, B, batch, 0.0f, C + (size_t)r0*batch, batch);
        }
        double dt = now()-t0;
        if (dt < best) best = dt;
    }
    double gflops = 2.0*rows*cols*batch/best/1e9;
    printf("\n  ── 结果 ──\n    总耗时 : %.2f ms\n    算力   : %.0f GFLOPS\n", best*1000, gflops);
    printf("\n  ═══ 对照 ═══\n");
    printf("    IQ2_XXS（分块 512 行）: 3.74 ms\n");
    printf("    Q4_K  （分块 %d 行）: %.2f ms %s\n", rblk, best*1000,
           best*1000 < 3.74 ? "← 更快" : "← 更慢");
    printf("    GPU 单专家           : ~0.58 ms\n");
    free(w); free(tmp); free(B); free(C);
    return 0;
}
