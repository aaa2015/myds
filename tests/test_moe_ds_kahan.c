/* 验证 Metal 可行的两种高精度求和（无 double）：
 *   B. double-single（TwoSum 补偿）
 *   C. Kahan 补偿求和
 * 判据：打乱专家累加顺序后是否逐位相同。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NEI0 6
#define NB 20
#define NROW 64

static double block_contrib(unsigned seed, int slot, int blk, int row) {
    uint64_t x = (uint64_t)seed * 1000003ull + (uint64_t)slot * 7919ull
               + (uint64_t)blk * 104729ull + (uint64_t)row * 1299709ull;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
    double u = (double)(x & 0xFFFFFF) / (double)0xFFFFFF;
    return (u - 0.5) * (1.0 + (double)((x >> 40) & 0xFF) / 32.0);
}

/* ---- double-single：hi + lo，用 TwoSum ---- */
typedef struct { float hi, lo; } ds;

static inline void two_sum(float a, float b, float *s, float *e) {
    float t = a + b;
    float bb = t - a;
    *e = (a - (t - bb)) + (b - bb);
    *s = t;
}
static inline void ds_add(ds *acc, float v) {
    float s, e;
    two_sum(acc->hi, v, &s, &e);
    acc->lo += e;
    /* 再规范化一次 */
    float s2, e2;
    two_sum(s, acc->lo, &s2, &e2);
    acc->hi = s2;
    acc->lo = e2;
}

/* ---- Kahan ---- */
static inline void kahan_add(float *sum, float *c, float v) {
    float y = v - *c;
    float t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

static float sum_ds(unsigned seed, int row, const int *order) {
    ds acc = {0.0f, 0.0f};
    for (int i = 0; i < NEI0; i++) {
        int slot = order[i];
        for (int blk = 0; blk < NB; blk++)
            ds_add(&acc, (float)block_contrib(seed, slot, blk, row));
    }
    return acc.hi + acc.lo;
}

static float sum_kahan(unsigned seed, int row, const int *order) {
    float sum = 0.0f, c = 0.0f;
    for (int i = 0; i < NEI0; i++) {
        int slot = order[i];
        for (int blk = 0; blk < NB; blk++)
            kahan_add(&sum, &c, (float)block_contrib(seed, slot, blk, row));
    }
    return sum;
}

static float sum_f32(unsigned seed, int row, const int *order) {
    float s = 0.0f;
    for (int i = 0; i < NEI0; i++) {
        int slot = order[i];
        for (int blk = 0; blk < NB; blk++)
            s += (float)block_contrib(seed, slot, blk, row);
    }
    return s;
}

/* 分段 double-single（模拟 CPU/GPU 分工后合并）*/
static float sum_ds_split(unsigned seed, int row, int nsplit, const int *order) {
    ds parts[8];
    int half = NEI0 / nsplit;
    for (int s = 0; s < nsplit; s++) {
        parts[s].hi = parts[s].lo = 0.0f;
        for (int i = s * half; i < (s + 1) * half; i++) {
            int slot = order[i];
            for (int blk = 0; blk < NB; blk++)
                ds_add(&parts[s], (float)block_contrib(seed, slot, blk, row));
        }
    }
    ds acc = {0.0f, 0.0f};
    for (int s = 0; s < nsplit; s++) {
        ds_add(&acc, parts[s].hi);
        ds_add(&acc, parts[s].lo);
    }
    return acc.hi + acc.lo;
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
        {0,1,2,3,4,5}, {5,4,3,2,1,0}, {0,2,4,1,3,5}, {3,0,5,1,4,2}
    };

    printf("  ═══ B. double-single（TwoSum 补偿）═══\n");
    {
        int bad = 0, mx = 0;
        for (unsigned seed = 1; seed <= 20; seed++)
            for (int row = 0; row < NROW; row++) {
                float base = sum_ds(seed, row, orders[0]);
                for (int o = 1; o < 4; o++) {
                    int u = ulp_diff(base, sum_ds(seed, row, orders[o]));
                    if (u) bad++;
                    if (u > mx) mx = u;
                }
            }
        printf("    顺序差异次数 %d，最大 %d ULP ⇒ %s\n", bad, mx, bad == 0 ? "✅ 顺序无关" : "❌");
    }

    printf("\n  ═══ C. Kahan 补偿求和 ═══\n");
    {
        int bad = 0, mx = 0;
        for (unsigned seed = 1; seed <= 20; seed++)
            for (int row = 0; row < NROW; row++) {
                float base = sum_kahan(seed, row, orders[0]);
                for (int o = 1; o < 4; o++) {
                    int u = ulp_diff(base, sum_kahan(seed, row, orders[o]));
                    if (u) bad++;
                    if (u > mx) mx = u;
                }
            }
        printf("    顺序差异次数 %d，最大 %d ULP ⇒ %s\n", bad, mx, bad == 0 ? "✅ 顺序无关" : "❌");
    }

    printf("\n  ═══ D. 分段（模拟 CPU/GPU 分工）═══\n");
    for (int ns = 2; ns <= 3; ns++) {
        int bad = 0, mx = 0;
        for (unsigned seed = 1; seed <= 20; seed++)
            for (int row = 0; row < NROW; row++) {
                float base = sum_ds(seed, row, orders[0]);
                for (int o = 0; o < 4; o++) {
                    int u = ulp_diff(base, sum_ds_split(seed, row, ns, orders[o]));
                    if (u) bad++;
                    if (u > mx) mx = u;
                }
            }
        printf("    double-single 分 %d 段：差异 %d 次，最大 %d ULP ⇒ %s\n",
               ns, bad, mx, bad == 0 ? "✅" : "❌");
    }

    printf("\n  ═══ E. 与当前 FP32 的差距（规格变更幅度）═══\n");
    {
        double maxrel = 0;
        for (unsigned seed = 1; seed <= 20; seed++)
            for (int row = 0; row < NROW; row++) {
                float a = sum_f32(seed, row, orders[0]);
                float b = sum_ds(seed, row, orders[0]);
                if (a != 0.0f) {
                    double rel = fabs((double)(b - a) / (double)a);
                    if (rel > maxrel) maxrel = rel;
                }
            }
        printf("    最大相对差 %.3e（≈ %.0f ULP 量级）\n", maxrel, maxrel / 6e-8);
    }
    return 0;
}
