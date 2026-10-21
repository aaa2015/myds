/* CPU Q4_K GEMV —— 测真实反量化下的吞吐。
 * 目标：判断「把部分层从 IQ2_XXS 升到 Q4_K，交给 CPU」是否值得。
 *
 * Q4_K 块结构（对齐 metal/moe.metal）：
 *   struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
 *   = 144 bytes / 256 元素
 *   每 il（0..15）产 16 个元素：dl*(q[i]&mask) - ml
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

#define QK_K 256

typedef struct {
    uint16_t d;          /* fp16 */
    uint16_t dmin;       /* fp16 */
    uint8_t  scales[12];
    uint8_t  qs[128];
} block_q4_K;            /* 144 B */

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static inline float h2f(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

static inline void get_scale_min_k4_just2(int j, int k, const uint8_t *q,
                                          uint8_t *sc0, uint8_t *sc1) {
    if (j < 4) {
        *sc0 = q[j + 0 + k] & 63;
        *sc1 = q[j + 4 + k] & 63;
    } else {
        *sc0 = (q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2);
        *sc1 = (q[j + 4 + k] >> 4)    | ((q[j - 0 + k] & 0xc0) >> 2);
    }
}

/* 解码一个 il（16 个元素）并与激活做 FMA —— 融合，不留缓冲 */
static inline float32x4_t q4k_il_fma(const block_q4_K *b, int il,
                                     const float *x, float32x4_t vacc) {
    const uint8_t *q = b->qs;
    int is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    int ilp = il & 3;
    uint8_t sc0, sc1;
    get_scale_min_k4_just2(is, ilp / 2, b->scales, &sc0, &sc1);
    const float d = (ilp < 2) ? h2f(b->d) : h2f(b->d) * (1.0f / 16.0f);
    const float mn = h2f(b->dmin);
    const float dl = d * (float)sc0;
    const float ml = mn * (float)sc1;
    const uint8_t mask = (ilp < 2) ? 0x0F : 0xF0;

    /* 16 个值：dl*(q[i]&mask) - ml，用 NEON 批量 */
    /* u8 → f32（4 组），一次算完，无冗余 */
    uint8x16_t qv = vld1q_u8(q);
    uint8x16_t mv = vandq_u8(qv, vdupq_n_u8(mask));
    uint16x8_t lo16 = vmovl_u8(vget_low_u8(mv));
    uint16x8_t hi16 = vmovl_u8(vget_high_u8(mv));
    const float32x4_t dlv = vdupq_n_f32(dl), mlv = vdupq_n_f32(ml);
    /* dl*q - ml = fma(q, dl, -ml) */
    const float32x4_t nml = vnegq_f32(mlv);
    float32x4_t g0 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))),  dlv);
    float32x4_t g1 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))), dlv);
    float32x4_t g2 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))),  dlv);
    float32x4_t g3 = vfmaq_f32(nml, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))), dlv);

    const float *xs = x + il * 16;
    vacc = vfmaq_f32(vacc, g0, vld1q_f32(xs + 0));
    vacc = vfmaq_f32(vacc, g1, vld1q_f32(xs + 4));
    vacc = vfmaq_f32(vacc, g2, vld1q_f32(xs + 8));
    vacc = vfmaq_f32(vacc, g3, vld1q_f32(xs + 12));
    return vacc;
}

typedef struct {
    const block_q4_K *w;
    const float *x;
    float *out;
    uint32_t rows, cols;
    uint32_t r0, r1;
} work_t;

static void *worker(void *p) {
    work_t *w = (work_t *)p;
    const uint32_t nb = w->cols / QK_K;
    for (uint32_t r = w->r0; r < w->r1; r++) {
        const block_q4_K *row = w->w + (size_t)r * nb;
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (uint32_t b = 0; b < nb; b++) {
            const float *xs = w->x + b * QK_K;
            for (int il = 0; il < 16; il++)
                vacc = q4k_il_fma(row + b, il, xs, vacc);
        }
        w->out[r] = vaddvq_f32(vacc);
    }
    return NULL;
}

/* 标量参考 */
static float ref_row(const block_q4_K *row, uint32_t nb, const float *x) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const block_q4_K *blk = row + b;
        for (int il = 0; il < 16; il++) {
            const uint8_t *q = blk->qs + (il / 4) * 32 + 16 * (il & 1);
            int is = (il / 4) * 2, ilp = il & 3;
            uint8_t sc0, sc1;
            get_scale_min_k4_just2(is, ilp / 2, blk->scales, &sc0, &sc1);
            float d = (ilp < 2) ? h2f(blk->d) : h2f(blk->d) * (1.0f / 16.0f);
            float mn = h2f(blk->dmin);
            float dl = d * sc0, ml = mn * sc1;
            uint8_t mask = (ilp < 2) ? 0x0F : 0xF0;
            const float *xs = x + b * QK_K + il * 16;
            for (int i = 0; i < 16; i++)
                acc += (dl * (float)(q[i] & mask) - ml) * xs[i];
        }
    }
    return acc;
}

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 16;
    uint32_t rows = 2304, cols = 5120;   /* MoE gate 单专家形状 */
    int rep = argc > 2 ? atoi(argv[2]) : 20;

    size_t nb = (size_t)rows * (cols / QK_K);
    size_t wb = nb * sizeof(block_q4_K);
    printf("  CPU Q4_K GEMV：%u × %u，权重 %.2f MB，%d 线程\n", rows, cols, wb / 1e6, nt);

    block_q4_K *w = aligned_alloc(64, wb);
    float *x = aligned_alloc(64, (size_t)cols * 4);
    float *out = aligned_alloc(64, (size_t)rows * 4);
    if (!w || !x || !out) { printf("  alloc failed\n"); return 1; }
    for (size_t i = 0; i < nb; i++) {
        w[i].d = 0x2119;      /* ≈0.0195 */
        w[i].dmin = 0x2119;
        for (int k = 0; k < 12; k++) w[i].scales[k] = (uint8_t)((i * 31 + k * 17) % 64);
        for (int k = 0; k < 128; k++) w[i].qs[k] = (uint8_t)((i * 7919 + k * 104729) & 0xFF);
    }
    for (uint32_t i = 0; i < cols; i++) x[i] = 0.001f * (float)((i * 37) % 101 - 50);

    /* 正确性 */
    {
        work_t wk = { w, x, out, rows, cols, 0, 4 };
        worker(&wk);
        double mx = 0;
        for (uint32_t r = 0; r < 4; r++) {
            float ref = ref_row(w + (size_t)r * (cols / QK_K), cols / QK_K, x);
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
            wk[t] = (work_t){ w, x, out, rows, cols, (uint32_t)(rows * t / nt), (uint32_t)(rows * (t + 1) / nt) };
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
    printf("\n  ═══ 与 IQ2 对比（等效吞吐，按压缩后字节折算）═══\n");
    double ratio = 4.5 / 2.06;
    printf("    Q4_K 字节倍率 %.2fx ⇒ 等效 IQ2 吞吐 %.1f GB/s\n", ratio, gbs / ratio);
    printf("    IQ2_XXS 实测 0.9 GB/s ⇒ **%.0f 倍**\n", (gbs / ratio) / 0.9);
    free(w); free(x); free(out);
    return 0;
}
