/* 验证：把 MoE down 的累加改用 double 后，
 *   (a) 输出是否与专家累加顺序无关（顺序无关性 = 可安全拆分的前提）
 *   (b) 与当前 FP32 顺序累加的差异有多大（准备规格变更）
 *   (c) 单次调用耗时与寄存器影响
 *
 * 做法：在 CPU 上用完全相同的 IQ2/Q2_K 解码语义模拟两种累加方式，
 * 不需要改 kernel —— 先用数值实验确认「double 累加是否真的顺序无关」。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NEI0 6            /* 每 token 的专家数 */
#define NB 20             /* dim=5120 / QK_K=256 */
#define NROW 64           /* 输出行数（取样，足够统计） */

/* 模拟一个 Q2_K 块的“已解码贡献值” */
static double block_contrib(unsigned seed, int slot, int blk, int row) {
    /* 用确定性伪随机模拟真实权重/激活的乘积和，量级接近真实（正负抵消） */
    uint64_t x = (uint64_t)seed * 1000003ull + (uint64_t)slot * 7919ull
               + (uint64_t)blk * 104729ull + (uint64_t)row * 1299709ull;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
    double u = (double)(x & 0xFFFFFF) / (double)0xFFFFFF;   /* [0,1) */
    /* 真实 MoE 的块贡献有正有负，量级跨 2 个数量级 */
    return (u - 0.5) * (1.0 + (double)((x >> 40) & 0xFF) / 32.0);
}

/* 顺序 0..5 的 FP32 累加（当前生产语义） */
static float sum_f32_fwd(unsigned seed, int row) {
    float sumf = 0.0f;
    for (int slot = 0; slot < NEI0; slot++)
        for (int blk = 0; blk < NB; blk++)
            sumf += (float)block_contrib(seed, slot, blk, row);
    return sumf;
}

/* 任意顺序的 FP32 累加 */
static float sum_f32_order(unsigned seed, int row, const int *order) {
    float sumf = 0.0f;
    for (int i = 0; i < NEI0; i++) {
        int slot = order[i];
        for (int blk = 0; blk < NB; blk++)
            sumf += (float)block_contrib(seed, slot, blk, row);
    }
    return sumf;
}

/* 任意顺序的 double 累加，最后一次性舍入到 float */
static float sum_f64_order(unsigned seed, int row, const int *order) {
    double sum = 0.0;
    for (int i = 0; i < NEI0; i++) {
        int slot = order[i];
        for (int blk = 0; blk < NB; blk++)
            sum += block_contrib(seed, slot, blk, row);
    }
    return (float)sum;
}

/* 分段相加（模拟 CPU 做前半 / GPU 做后半，各自 FP32 求和再合并） */
static float sum_f32_split(unsigned seed, int row, int nsplit) {
    float parts[8];
    int half = NEI0 / nsplit;
    for (int s = 0; s < nsplit; s++) {
        float p = 0.0f;
        for (int slot = s * half; slot < (s + 1) * half; slot++)
            for (int blk = 0; blk < NB; blk++)
                p += (float)block_contrib(seed, slot, blk, row);
        parts[s] = p;
    }
    float acc = 0.0f;
    for (int s = 0; s < nsplit; s++) acc += parts[s];
    return acc;
}

/* 分段相加但每段用 double */
static float sum_f64_split(unsigned seed, int row, int nsplit) {
    double parts[8];
    int half = NEI0 / nsplit;
    for (int s = 0; s < nsplit; s++) {
        double p = 0.0;
        for (int slot = s * half; slot < (s + 1) * half; slot++)
            for (int blk = 0; blk < NB; blk++)
                p += block_contrib(seed, slot, blk, row);
        parts[s] = p;
    }
    double acc = 0.0;
    for (int s = 0; s < nsplit; s++) acc += parts[s];
    return (float)acc;
}

static int ulp_diff(float a, float b) {
    if (a == b) return 0;
    int32_t ia, ib;
    memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if (ia < 0) ia = (int32_t)0x80000000u - ia;
    if (ib < 0) ib = (int32_t)0x80000000u - ib;
    return abs((int)(ia - ib));
}

int main(void) {
    const int orders[4][NEI0] = {
        {0,1,2,3,4,5},     /* 原顺序 */
        {5,4,3,2,1,0},     /* 完全反转 */
        {0,2,4,1,3,5},     /* 交错 */
        {3,0,5,1,4,2},     /* 随机 */
    };
    const char *oname[4] = {"0..5", "5..0", "交错", "随机"};

    printf("  ═══ A. FP32 顺序累加：是否顺序无关？ ═══\n");
    int f32_diff_seeds = 0, f32_maxulp = 0;
    for (unsigned seed = 1; seed <= 20; seed++) {
        int maxulp = 0;
        for (int row = 0; row < NROW; row++) {
            float base = sum_f32_fwd(seed, row);
            for (int o = 1; o < 4; o++) {
                int u = ulp_diff(base, sum_f32_order(seed, row, orders[o]));
                if (u > maxulp) maxulp = u;
            }
        }
        if (maxulp > 0) f32_diff_seeds++;
        if (maxulp > f32_maxulp) f32_maxulp = maxulp;
    }
    printf("    %d/20 个种子出现差异，最大 %d ULP\n", f32_diff_seeds, f32_maxulp);
    printf("    ⇒ %s\n\n", f32_diff_seeds ? "❌ FP32 累加顺序敏感（需高精度）" : "✅ 顺序无关");

    printf("  ═══ B. double 累加：是否顺序无关？ ═══\n");
    int f64_diff_seeds = 0, f64_maxulp = 0;
    for (unsigned seed = 1; seed <= 20; seed++) {
        int maxulp = 0;
        for (int row = 0; row < NROW; row++) {
            float base = sum_f64_order(seed, row, orders[0]);
            for (int o = 1; o < 4; o++) {
                int u = ulp_diff(base, sum_f64_order(seed, row, orders[o]));
                if (u > maxulp) maxulp = u;
            }
        }
        if (maxulp > 0) f64_diff_seeds++;
        if (maxulp > f64_maxulp) f64_maxulp = maxulp;
    }
    printf("    %d/20 个种子出现差异，最大 %d ULP\n", f64_diff_seeds, f64_maxulp);
    printf("    ⇒ %s\n\n", f64_diff_seeds ? "❌ double 也不够" : "✅ 顺序无关（拆分的数学前提成立）");

    printf("  ═══ C. 分段相加（模拟 CPU/GPU 分工）═══\n");
    printf("    %-22s %10s %10s\n", "方式", "vs 原顺序", "vs double");
    for (int ns = 2; ns <= 3; ns++) {
        int max1 = 0, max2 = 0;
        for (unsigned seed = 1; seed <= 20; seed++)
            for (int row = 0; row < NROW; row++) {
                float base32 = sum_f32_fwd(seed, row);
                float base64 = sum_f64_order(seed, row, orders[0]);
                int u1 = ulp_diff(base32, sum_f32_split(seed, row, ns));
                int u2 = ulp_diff(base64, sum_f64_split(seed, row, ns));
                if (u1 > max1) max1 = u1;
                if (u2 > max2) max2 = u2;
            }
        char lbl[64];
        snprintf(lbl, sizeof lbl, "FP32 分 %d 段", ns);
        printf("    %-22s %8d ULP %8s\n", lbl, max1, "—");
        snprintf(lbl, sizeof lbl, "double 分 %d 段", ns);
        printf("    %-22s %8s %8d ULP\n", lbl, "—", max2);
    }

    printf("\n  ═══ D. 代价：double 累加多几个寄存器？ ═══\n");
    printf("    float sumf[nr0]   : nr0 个 FP32 寄存器\n");
    printf("    double sumf[nr0]  : 2*nr0 个 FP32 寄存器（64-bit 占两格）\n");
    printf("    nr0=2 时多 2 个；kernel 现用 64 threads/tg，寄存器本就紧（§22）\n");
    return 0;
}
