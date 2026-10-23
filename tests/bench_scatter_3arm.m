/* 三臂散读基准 v2：钉死 MoE 归因（2026-10-23）
 *
 * v1 的教训：用 6 次 kernel_pureread 调用模拟 6 个散专家，
 *   测出来的是【调用次数】差异（每次调用含 ~0.21 ms 固定开销）：
 *   A 6 次 → 1.509 ms；B/C 各 36 次 → 7.9/8.3 ms，「5.2 倍」全是调用数。
 *
 * v2 改用 kernel_pureread_scatter6：**一次 dispatch 内**由 gid 分配到 6 个区域。
 *   A 连续：n_regions=1，读 region_bytes*6（一整段）
 *   B 散读：n_regions=6，offs = 6 个远距离专家起点
 *   C 同址：n_regions=6，offs 全相同（测 SLC 命中）
 * 三臂都是 1 次 dispatch ⇒ 固定开销相同，差异只能来自地址模式。
 *
 * 用法: bench_scatter_3arm [单块MB=32] [重复R=6]
 */
#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

#define NEXP 384
#define EXP_BYTES (3041u * 1024u)

int main(int argc, char **argv) {
    const size_t per_mb = argc > 1 ? (size_t)atoi(argv[1]) : 32;
    const int R = argc > 2 ? atoi(argv[2]) : 6;
    const size_t region = per_mb * 1024u * 1024u;
    const size_t pool = (size_t)NEXP * EXP_BYTES;

    if (!myds_metal_init(NULL)) { printf("metal init 失败\n"); return 1; }
    printf("  池 %.2f GB | 每块 %.0f MB × 6 = %.0f MB/次 | 每臂 %d 次（总 %.0f MB）\n\n",
           pool/1e9, (double)region/1e6, (double)region*6/1e6, R, (double)region*6*R/1e6);

    myds_metal_buffer_t *w = myds_metal_buffer_alloc(pool);
    myds_metal_buffer_t *d = myds_metal_buffer_alloc(4096 * 4);
    if (!w || !d) { printf("alloc 失败\n"); return 1; }
    { uint8_t *b = (uint8_t *)myds_metal_buffer_data(w);
      for (size_t i = 0; i < pool; i += 4096) b[i] = (uint8_t)((i * 37u) >> 12); }

    const uint32_t exps[6] = { 5, 47, 111, 173, 280, 370 };
    uint64_t offs_scat[6], offs_same[6];
    for (int i = 0; i < 6; i++) {
        offs_scat[i] = (uint64_t)exps[i] * EXP_BYTES;
        offs_same[i] = (uint64_t)exps[0] * EXP_BYTES;
    }

    struct { const char *name; const uint64_t *offs; uint32_t nreg; } arms[] = {
        { "A 连续  (1 段 192MB)",   offs_same, 1 },
        { "B 散读  (6 个散专家)",   offs_scat, 6 },
        { "C 同址  (6 次同 offset)", offs_same, 6 },
    };

    printf("  %-28s %10s %12s\n", "臂", "耗时(ms)", "有效GB/s");
    double bw[3] = {0,0,0};
    for (int a = 0; a < 3; a++) {
        double best = 1e18;
        for (int rep = 0; rep < R + 2; rep++) {
            double t0 = now_us();
            for (int k = 0; k < R; k++)
                if (!myds_metal_pureread_scatter6(d, w, arms[a].offs, region, arms[a].nreg)) return 2;
            myds_metal_synchronize();          /* 每臂一次 */
            double dt = now_us() - t0;
            if (rep >= 2 && dt < best) best = dt;
        }
        const double bytes = (double)region * 6.0 * R;
        bw[a] = bytes / (best * 1e-6) / 1e9;
        printf("  %-28s %10.3f %12.1f\n", arms[a].name, best / 1000.0, bw[a]);
    }
    printf("\n  ═══ 判读 ═══\n");
    printf("  A/B = %.2f 倍  ⇒ 这是「散读」的纯代价（三臂 dispatch 数相同）\n", bw[0]/bw[1]);
    printf("  A/C = %.2f 倍  ⇒ C 是同一地址重复读，差值反映 SLC 命中收益\n", bw[0]/bw[2]);
    myds_metal_buffer_free(w); myds_metal_buffer_free(d);
    return 0;
}
