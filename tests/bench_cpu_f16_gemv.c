/* CPU F16 GEMV —— 无需解码，只需 fp16→fp32 转换 + 乘加。
 * 目标形状：engram_kv（6144 × 25600 F16 = 314.6 MB） */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>
#include <math.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
    const __fp16 *w;     /* rows × cols */
    const float *x;      /* cols */
    float *out;
    int rows, cols;
    int r0, r1;
} work_t;

static void *worker(void *p) {
    work_t *w = (work_t *)p;
    for (int r = w->r0; r < w->r1; r++) {
        const __fp16 *row = w->w + (size_t)r * w->cols;
        float32x4_t v0 = vdupq_n_f32(0), v1 = vdupq_n_f32(0);
        float32x4_t v2 = vdupq_n_f32(0), v3 = vdupq_n_f32(0);
        int c = 0;
        for (; c + 16 <= w->cols; c += 16) {
            /* 一次处理 16 个：2×float16x8 → 4×float32x4 */
            /* float16x8 → 两个 float32x4 */
            float16x4_t h0 = vld1_f16(row + c);
            float16x4_t h1 = vld1_f16(row + c + 4);
            float16x4_t h2 = vld1_f16(row + c + 8);
            float16x4_t h3 = vld1_f16(row + c + 12);
            v0 = vfmaq_f32(v0, vcvt_f32_f16(h0), vld1q_f32(w->x + c));
            v1 = vfmaq_f32(v1, vcvt_f32_f16(h1), vld1q_f32(w->x + c + 4));
            v2 = vfmaq_f32(v2, vcvt_f32_f16(h2), vld1q_f32(w->x + c + 8));
            v3 = vfmaq_f32(v3, vcvt_f32_f16(h3), vld1q_f32(w->x + c + 12));
        }
        for (; c < w->cols; c++) v0 = vfmaq_f32(v0, vdupq_n_f32((float)row[c]), vdupq_n_f32(w->x[c]));
        float32x4_t s = vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3));
        w->out[r] = vaddvq_f32(s);
    }
    return NULL;
}

/* 标量参考，验证正确性 */
static float ref_row(const __fp16 *row, int cols, const float *x) {
    float a = 0.0f;
    for (int c = 0; c < cols; c++) a += (float)row[c] * x[c];
    return a;
}

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 16;
    int rows = 6144, cols = 25600;
    int rep = argc > 2 ? atoi(argv[2]) : 5;
    size_t wb = (size_t)rows * cols * 2;
    printf("  CPU F16 GEMV：%d × %d，权重 %.1f MB，%d 线程\n", rows, cols, wb / 1e6, nt);

    __fp16 *w = aligned_alloc(64, wb);
    float *x = aligned_alloc(64, (size_t)cols * 4);
    float *out = aligned_alloc(64, (size_t)rows * 4);
    if (!w || !x || !out) { printf("  alloc failed\n"); return 1; }
    for (size_t i = 0; i < (size_t)rows * cols; i++)
        w[i] = (__fp16)(0.001f * (float)((i * 7919u) % 2001 - 1000));
    for (int i = 0; i < cols; i++) x[i] = 0.001f * (float)((i * 37) % 101 - 50);

    /* 正确性 */
    {
        work_t wk = { w, x, out, rows, cols, 0, 4 };
        worker(&wk);
        double mx = 0;
        for (int r = 0; r < 4; r++) {
            float ref = ref_row(w + (size_t)r * cols, cols, x);
            double d = fabs((double)(out[r] - ref));
            double rel = d / (fabs((double)ref) > 1e-3 ? fabs((double)ref) : 1e-3);
            if (rel > mx) mx = rel;
        }
        printf("  正确性：最大相对差 %.3e %s\n", mx, mx < 1e-4 ? "✅" : "❌");
        if (mx >= 1e-4) return 1;
    }

    double best = 1e9;
    for (int r = 0; r < rep; r++) {
        pthread_t th[64]; work_t wk[64];
        double t0 = now();
        for (int t = 0; t < nt; t++) {
            wk[t] = (work_t){ w, x, out, rows, cols, rows * t / nt, rows * (t + 1) / nt };
            pthread_create(&th[t], NULL, worker, &wk[t]);
        }
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    double gbs = wb / best / 1e9;
    printf("\n  ── 结果 ──\n");
    printf("    单次 GEMV : %.3f ms\n", best * 1000);
    printf("    带宽      : %.1f GB/s\n", gbs);
    printf("\n  ═══ 对比 ═══\n");
    printf("    GPU 同形状（engram_kv）: 508 GB/s（1.239 ms/token，2 层）\n");
    printf("    CPU 本次               : %.1f GB/s\n", gbs);
    printf("    单层耗时：GPU %.2f ms，CPU %.2f ms\n",
           (cols * (double)rows * 2 / 1e9) / 0.508, (cols * (double)rows * 2 / 1e9) / gbs);
    free(w); free(x); free(out);
    return 0;
}
