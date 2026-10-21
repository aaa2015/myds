/* CPU Q8_0 GEMV —— Q8_0 解码极简：d * (int8)q
 * struct block_q8_0 { half d; int8 qs[32]; };  = 34 B / 32 元素
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;   /* 34 B */

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static inline float h2f(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

typedef struct {
    const block_q8_0 *w; const float *x; float *out;
    uint32_t rows, cols, r0, r1;
} work_t;

static void *worker(void *p) {
    work_t *w = (work_t *)p;
    const uint32_t nb = w->cols / 32;
    for (uint32_t r = w->r0; r < w->r1; r++) {
        const block_q8_0 *row = w->w + (size_t)r * nb;
        float32x4_t v0 = vdupq_n_f32(0), v1 = vdupq_n_f32(0);
        for (uint32_t b = 0; b < nb; b++) {
            const float d = h2f(row[b].d);
            const int8_t *q = row[b].qs;
            const float *xs = w->x + b * 32;
            /* int8 → f32，一次 16 个（4 组 ×4） */
            int8x16_t q0 = vld1q_s8(q);
            int8x16_t q1 = vld1q_s8(q + 16);
            int16x8_t l0 = vmovl_s8(vget_low_s8(q0));
            int16x8_t h0 = vmovl_s8(vget_high_s8(q0));
            int16x8_t l1 = vmovl_s8(vget_low_s8(q1));
            int16x8_t h1 = vmovl_s8(vget_high_s8(q1));
            const float32x4_t dv = vdupq_n_f32(d);
            v0 = vfmaq_f32(v0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l0))),  dv), vld1q_f32(xs + 0));
            v1 = vfmaq_f32(v1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l0))), dv), vld1q_f32(xs + 4));
            v0 = vfmaq_f32(v0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h0))),  dv), vld1q_f32(xs + 8));
            v1 = vfmaq_f32(v1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h0))), dv), vld1q_f32(xs + 12));
            v0 = vfmaq_f32(v0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l1))),  dv), vld1q_f32(xs + 16));
            v1 = vfmaq_f32(v1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l1))), dv), vld1q_f32(xs + 20));
            v0 = vfmaq_f32(v0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h1))),  dv), vld1q_f32(xs + 24));
            v1 = vfmaq_f32(v1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h1))), dv), vld1q_f32(xs + 28));
        }
        w->out[r] = vaddvq_f32(vaddq_f32(v0, v1));
    }
    return NULL;
}

static float ref_row(const block_q8_0 *row, uint32_t nb, const float *x) {
    float a = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        float d = h2f(row[b].d);
        for (int i = 0; i < 32; i++) a += d * (float)row[b].qs[i] * x[b * 32 + i];
    }
    return a;
}

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 16;
    uint32_t rows = 2304, cols = 5120;
    int rep = argc > 2 ? atoi(argv[2]) : 20;
    size_t nb = (size_t)rows * (cols / 32);
    size_t wb = nb * sizeof(block_q8_0);
    printf("  CPU Q8_0 GEMV：%u × %u，权重 %.2f MB，%d 线程\n", rows, cols, wb/1e6, nt);

    block_q8_0 *w = aligned_alloc(64, wb);
    float *x = aligned_alloc(64, (size_t)cols * 4);
    float *out = aligned_alloc(64, (size_t)rows * 4);
    if (!w || !x || !out) return 1;
    for (size_t i = 0; i < nb; i++) {
        w[i].d = 0x2119;
        for (int k = 0; k < 32; k++) w[i].qs[k] = (int8_t)(((i * 7919 + k * 104729) % 255) - 127);
    }
    for (uint32_t i = 0; i < cols; i++) x[i] = 0.001f * (float)((i * 37) % 101 - 50);

    {
        work_t wk = { w, x, out, rows, cols, 0, 4 };
        worker(&wk);
        double mx = 0;
        for (uint32_t r = 0; r < 4; r++) {
            float ref = ref_row(w + (size_t)r * (cols/32), cols/32, x);
            double d = fabs((double)(out[r] - ref));
            double rel = d / (fabs((double)ref) > 1e-3 ? fabs((double)ref) : 1e-3);
            if (rel > mx) mx = rel;
        }
        printf("  正确性：最大相对差 %.3e %s\n", mx, mx < 1e-4 ? "✅" : "❌");
        if (mx >= 1e-4) return 1;
    }

    double best = 1e9;
    for (int r = 0; r < rep; r++) {
        pthread_t th[16]; work_t wk[16];
        double t0 = now();
        for (int t = 0; t < nt; t++) {
            wk[t] = (work_t){ w, x, out, rows, cols,
                              (uint32_t)(rows*t/nt), (uint32_t)(rows*(t+1)/nt) };
            pthread_create(&th[t], NULL, worker, &wk[t]);
        }
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    double gbs = wb / best / 1e9;
    printf("\n  ── 结果 ──\n    单次 GEMV : %.3f ms\n    带宽      : %.1f GB/s\n", best*1000, gbs);
    double ratio = 8.5 / 2.06;
    printf("\n  ═══ 对照（等效 IQ2 吞吐）═══\n");
    printf("    Q8_0    : %.1f GB/s / %.2fx = **%.1f**\n", gbs, ratio, gbs/ratio);
    printf("    F16     : 219.8 GB/s / 7.77x = 28.3\n");
    printf("    Q4_K    : 12.9 GB/s / 2.18x = 5.9\n");
    printf("    IQ2_XXS : 0.9 GB/s / 1.00x = 0.9\n");
    free(w); free(x); free(out);
    return 0;
}
