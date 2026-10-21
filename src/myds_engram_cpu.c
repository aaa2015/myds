/* CPU 卸载：engram_kv 投影（F16 GEMV，M=25600, N=6144, 314.6 MB）
 *
 * 实测（tests/bench_cpu_f16_gemv.c，M2 Ultra）：
 *   16 线程 219.8 GB/s ⇒ 单次 1.43 ms
 *   GPU 同形状 508 GB/s ⇒ 0.62 ms
 *   CPU 满载时 GPU decode 掉 −0.02%（互不影响）
 *
 * 数据流：engram_rows（CPU 已在 token 起始写入）→ 本 GEMV → engram_kv
 * ⇒ CPU 可在 token 一开始就算，与 GPU 并行，窗口 = 整个 token（23 ms）
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <pthread.h>
#include <arm_neon.h>

#define ENGRAM_OUT_DIM   (5u * 5120u)   /* 25600 */
#define ENGRAM_IN_DIM    6144u

typedef struct {
    const __fp16 *w;      /* [out][in] 行主序 */
    const float *x;       /* [in] */
    float *y;             /* [out] */
    uint32_t r0, r1;
} engram_cpu_work_t;

static void *engram_cpu_worker(void *p) {
    engram_cpu_work_t *w = (engram_cpu_work_t *)p;
    const uint32_t n = ENGRAM_IN_DIM;
    for (uint32_t r = w->r0; r < w->r1; r++) {
        const __fp16 *row = w->w + (size_t)r * n;
        float32x4_t v0 = vdupq_n_f32(0), v1 = vdupq_n_f32(0);
        float32x4_t v2 = vdupq_n_f32(0), v3 = vdupq_n_f32(0);
        uint32_t c = 0;
        for (; c + 16 <= n; c += 16) {
            float16x4_t h0 = vld1_f16(row + c);
            float16x4_t h1 = vld1_f16(row + c + 4);
            float16x4_t h2 = vld1_f16(row + c + 8);
            float16x4_t h3 = vld1_f16(row + c + 12);
            v0 = vfmaq_f32(v0, vcvt_f32_f16(h0), vld1q_f32(w->x + c));
            v1 = vfmaq_f32(v1, vcvt_f32_f16(h1), vld1q_f32(w->x + c + 4));
            v2 = vfmaq_f32(v2, vcvt_f32_f16(h2), vld1q_f32(w->x + c + 8));
            v3 = vfmaq_f32(v3, vcvt_f32_f16(h3), vld1q_f32(w->x + c + 12));
        }
        for (; c < n; c++) {
            v0 = vfmaq_f32(v0, vdupq_n_f32((float)row[c]), vdupq_n_f32(w->x[c]));
        }
        w->y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
    }
    return NULL;
}

/* 入口：weights 是 F16 行主序 [25600][6144]，x 是 [6144] float，y 是 [25600] float。
 * 返回 1 成功。注意：y 的布局必须与 GPU kernel 一致（逐元素相同）。
 * 由于是纯 FP32 累加、顺序与 GPU 不同，本函数【不保证 0-ULP】——
 * 需要调用方用 double-single 或接受语义等价。 */
int myds_cpu_engram_project(const void *weights_f16, const float *x, float *y,
                            uint32_t nthreads) {
    if (!weights_f16 || !x || !y) return 0;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    pthread_t th[16];
    engram_cpu_work_t wk[16];
    for (uint32_t t = 0; t < nthreads; t++) {
        wk[t].w = (const __fp16 *)weights_f16;
        wk[t].x = x;
        wk[t].y = y;
        wk[t].r0 = ENGRAM_OUT_DIM * t / nthreads;
        wk[t].r1 = ENGRAM_OUT_DIM * (t + 1) / nthreads;
        if (pthread_create(&th[t], NULL, engram_cpu_worker, &wk[t]) != 0) {
            /* 退化：本线程自己算剩余 */
            for (uint32_t k = t; k < nthreads; k++) engram_cpu_worker(&wk[k]);
            for (uint32_t k = 0; k < t; k++) pthread_join(th[k], NULL);
            return 1;
        }
    }
    for (uint32_t t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
    return 1;
}
