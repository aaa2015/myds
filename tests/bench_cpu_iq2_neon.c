/* 真实 IQ2_XXS NEON 解码 + GEMV，测 CPU 端在真实解码量下的吞吐。
 *
 * 解码语义严格对齐 metal/moe.metal 的 dequantize_iq2_xxs：
 *   struct block_iq2_xxs { half d; ushort qs[32]; };   // 66 B / 256 元素
 *   每块分成 4 个 64-元素子块（ib32 = il/2, il = il%2）
 *   aux32_g = q2[0] | q2[1]<<16 ; aux32_s = q2[2] | q2[3]<<16
 *   dl = d * (0.5 + (aux32_s>>28)) * 0.25
 *   grid0 = IQ2_GRID[aux8[2*il+0]], signs0 = IQ2_KSIGNS[(aux32_s>>(14*il))&127]
 *   grid1 = IQ2_GRID[aux8[2*il+1]], signs1 = IQ2_KSIGNS[(aux32_s>>(14*il+7))&127]
 *   元素 j: dl * grid[j] * (signs & (1<<j) ? -1 : +1)
 *
 * 关键：这里做【真实查表 + 真实符号展开】，不用假的伪解码。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>
#include <math.h>

#include "iq2_tables.h"

#define QK_K 256

typedef struct { uint16_t d; uint16_t qs[32]; } block_iq2_xxs;   /* 66 B */

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* half → float（IQ2 的 d 是 fp16） */
static inline float h2f(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

/* 融合版：解码一个 il 的 16 个元素后立即与激活做 FMA，不写中间缓冲。
 * 一个 256 元素的块 = 16 个 il。 */
static inline float32x4_t decode_il_fma_d(float d, const block_iq2_xxs *b, int il,
                                          const float *x, float32x4_t vacc) {
    const int ib32 = il / 2, ilp = il % 2;
    const uint16_t *q2 = b->qs + 4 * ib32;
    const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    const uint8_t *aux8 = (const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;

    const uint64_t g0 = IQ2_GRID[aux8[2 * ilp + 0]];
    const uint64_t g1 = IQ2_GRID[aux8[2 * ilp + 1]];
    const uint8_t s0 = IQ2_KSIGNS[(aux32_s >> (14 * ilp)) & 127];
    const uint8_t s1 = IQ2_KSIGNS[(aux32_s >> (14 * ilp + 7)) & 127];

    /* 两个 8 元素组 */
    /* 标量符号展开（已验证正确）；NEON 只用于最后的 FMA 累加。
     * 注：曾尝试用 vtst_u8 + vbslq_f32 做全向量化符号展开，但 8 位掩码
     * 与 32 位 float 选择器宽度不匹配，导致符号错——已回退。 */
    const uint8_t *p0 = (const uint8_t *)&g0;
    const uint8_t *p1 = (const uint8_t *)&g1;
    float v[16];
    for (int i = 0; i < 8; i++) {
        v[i]     = dl * (float)p0[i] * ((s0 & IQ2_KMASK[i]) ? -1.0f : 1.0f);
        v[i + 8] = dl * (float)p1[i] * ((s1 & IQ2_KMASK[i]) ? -1.0f : 1.0f);
    }
    const float *xs = x + il * 16;
    vacc = vfmaq_f32(vacc, vld1q_f32(v + 0),  vld1q_f32(xs + 0));
    vacc = vfmaq_f32(vacc, vld1q_f32(v + 4),  vld1q_f32(xs + 4));
    vacc = vfmaq_f32(vacc, vld1q_f32(v + 8),  vld1q_f32(xs + 8));
    vacc = vfmaq_f32(vacc, vld1q_f32(v + 12), vld1q_f32(xs + 12));
    return vacc;
}

/* 一次 GEMV：rows × cols 的 IQ2_XXS 权重 × float 激活 → rows 个输出 */
typedef struct {
    const block_iq2_xxs *w;
    const float *x;
    float *out;
    int rows, cols;
    int row_start, row_end;
} work_t;

static void *gemv_worker(void *p) {
    work_t *w = (work_t *)p;
    const int nb = w->cols / QK_K;
    for (int r = w->row_start; r < w->row_end; r++) {
        const block_iq2_xxs *row = w->w + (size_t)r * nb;
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const float *xs = w->x + b * QK_K;
            const float dd = h2f(row[b].d);
            for (int il = 0; il < 16; il++)
                vacc = decode_il_fma_d(dd, row + b, il, xs, vacc);
        }
        w->out[r] = vaddvq_f32(vacc);
    }
    return NULL;
}

/* 参考实现：逐元素标量，用于验证 NEON 版正确性 */
static float ref_gemv_row(const block_iq2_xxs *row, int nb, const float *x) {
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const block_iq2_xxs *blk = row + b;
        const float dd = h2f(blk->d);
        for (int il = 0; il < 16; il++) {
            const int ib32 = il / 2, ilp = il % 2;
            const uint16_t *q2 = blk->qs + 4 * ib32;
            const uint32_t g32 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t s32 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            const uint8_t *a8 = (const uint8_t *)&g32;
            const float dl = dd * (0.5f + (float)(s32 >> 28)) * 0.25f;
            const uint64_t g0v = IQ2_GRID[a8[2*ilp+0]], g1v = IQ2_GRID[a8[2*ilp+1]];
            const uint8_t sg0 = IQ2_KSIGNS[(s32 >> (14*ilp)) & 127];
            const uint8_t sg1 = IQ2_KSIGNS[(s32 >> (14*ilp+7)) & 127];
            const uint8_t *p0 = (const uint8_t *)&g0v, *p1 = (const uint8_t *)&g1v;
            const float *xs = x + b * QK_K + il * 16;
            for (int i = 0; i < 8; i++) {
                acc += dl * (float)p0[i] * ((sg0 & IQ2_KMASK[i]) ? -1.0f : 1.0f) * xs[i];
                acc += dl * (float)p1[i] * ((sg1 & IQ2_KMASK[i]) ? -1.0f : 1.0f) * xs[i+8];
            }
        }
    }
    return acc;
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 16;
    int rows = 2304, cols = 5120;
    int repeat = argc > 2 ? atoi(argv[2]) : 20;
    int verify = argc > 3 ? atoi(argv[3]) : 1;

    size_t nb = (size_t)rows * (cols / QK_K);
    size_t wbytes = nb * sizeof(block_iq2_xxs);
    printf("  IQ2_XXS NEON GEMV：%d × %d，权重 %.2f MB，%d 线程，%d 次\n",
           rows, cols, wbytes / 1e6, nthreads, repeat);

    block_iq2_xxs *w = aligned_alloc(64, wbytes);
    float *x = aligned_alloc(64, (size_t)cols * 4);
    float *out = aligned_alloc(64, (size_t)rows * 4);
    float *out_ref = aligned_alloc(64, (size_t)rows * 4);
    if (!w || !x || !out || !out_ref) { printf("  alloc failed\n"); return 1; }

    /* 填真实形状的数据（d 用 fp16 1.0 附近，qs 全范围） */
    for (size_t i = 0; i < nb; i++) {
        w[i].d = 0x2119;   /* fp16 ≈0.0195，接近真实 IQ2 的 d */
        for (int k = 0; k < 32; k++)
            w[i].qs[k] = (uint16_t)((i * 7919u + k * 104729u) & 0xFFFF);
    }
    for (int i = 0; i < cols; i++) x[i] = 0.001f * (float)((i * 37) % 101 - 50);

    /* ---- 正确性：单线程 vs 参考实现 ---- */
    if (verify) {
        int ncheck = 8;
        work_t wk = { w, x, out, rows, cols, 0, ncheck };
        gemv_worker(&wk);
        double maxrel = 0, maxabs = 0;
        for (int r = 0; r < ncheck; r++) {
            float ref = ref_gemv_row(w + (size_t)r * (cols / QK_K), cols / QK_K, x);
            double d = fabs((double)(out[r] - ref));
            double mag = fabs((double)ref);
            if (d > maxabs) maxabs = d;
            /* 用相对 ref 量级 + 绝对量双重判据：避免抵消导致的虚假相对差 */
            double rel = d / (mag > 1e-3 ? mag : 1e-3);
            if (rel > maxrel) maxrel = rel;
            if (r < 3) printf("    row%d NEON=%.6g ref=%.6g diff=%.3e\n", r, out[r], ref, d);
        }
        printf("  正确性：最大绝对差 %.3e，最大相对差 %.3e %s\n",
               maxabs, maxrel, maxrel < 1e-5 ? "✅" : "❌");
        if (maxrel >= 1e-5) { printf("  拒绝继续计时\n"); return 1; }
    }

    /* ---- 计时 ---- */
    double best = 1e9;
    for (int r = 0; r < repeat; r++) {
        pthread_t th[64]; work_t wk[64];
        double t0 = now();
        for (int t = 0; t < nthreads; t++) {
            wk[t] = (work_t){ w, x, out, rows, cols,
                              rows * t / nthreads, rows * (t + 1) / nthreads };
            pthread_create(&th[t], NULL, gemv_worker, &wk[t]);
        }
        for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }

    double gbs = wbytes / best / 1e9;
    printf("\n  ── 结果 ──\n");
    printf("    单次 GEMV : %.3f ms\n", best * 1000);
    printf("    带宽      : %.1f GB/s\n", gbs);
    printf("\n  ═══ 与 GPU 并行时的推算 ═══\n");
    printf("    GPU（MoE 实际）: 277 GB/s\n");
    printf("    CPU（本次）    : %.1f GB/s\n", gbs);
    printf("    合计           : %.1f GB/s（+%.1f%%）\n", 277.0 + gbs, 100 * gbs / 277.0);
    double per_layer = 59.72;   /* MB/token */
    double us_now = per_layer / 1000 / 277.0 * 1e6;
    double us_new = per_layer / 1000 / (277.0 + gbs) * 1e6;
    printf("    每层 %.2f MB: %.1f µs → %.1f µs（省 %.1f µs）\n",
           per_layer, us_now, us_new, us_now - us_new);
    double save_ms = (us_now - us_new) * 29 / 1000;
    printf("    29 层省 %.2f ms/token ⇒ decode %.2f → %.2f ms ⇒ **+%.1f%%**\n",
           save_ms, 23.35, 23.35 - save_ms, 100 * (23.35 / (23.35 - save_ms) - 1));

    free(w); free(x); free(out); free(out_ref);
    return 0;
}
