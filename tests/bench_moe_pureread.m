/* MoE 带宽上限判定：同样 6 个专家、同样字节，但**不做解码**
 *
 * 目的：区分
 *   A) 纯读就能到 700 GB/s  ⇒ 瓶颈是 IQ2 查表 / Q2_K 解码的指令吞吐
 *   B) 纯读也只有 ~285 GB/s ⇒ 瓶颈是访存模式
 */
#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

int main(int argc, char **argv) {
    const size_t pool = (size_t)384 * 2304 * 20 * 66;   /* 1.17 GB，与 gate 池一致 */
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    if (!myds_metal_init(NULL)) { printf("metal init 失败\n"); return 1; }

    printf("  池 %.2f GB，%d 次\n", pool / 1e9, reps);
    myds_metal_buffer_t *w = myds_metal_buffer_alloc(pool);
    myds_metal_buffer_t *d = myds_metal_buffer_alloc(4096 * 4);
    if (!w || !d) { printf("alloc 失败\n"); return 1; }
    { uint8_t *b = (uint8_t *)myds_metal_buffer_data(w);
      for (size_t i = 0; i < pool; i++) b[i] = (uint8_t)((i * 37u) & 0xFF); }

    /* 每种形态：(读多少字节, 步长) */
    struct { const char *name; size_t bytes; size_t stride; } cases[] = {
        { "6 专家   连续 (18 MB)",   6 * 3040 * 1024,  16 },
        { "6 专家   散开 (stride 64)", 6 * 3040 * 1024, 64 },
        { "64 专家  连续 (194 MB)", 64 * 3040 * 1024,  16 },
        { "384 专家 全池 (1.17 GB)", 384 * 3040 * 1024, 16 },
    };

    printf("\n  %-30s %10s %12s %10s\n", "形态", "耗时(ms)", "有效GB/s", "线程数");
    for (unsigned c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const size_t bytes = cases[c].bytes;
        const size_t stride = cases[c].stride;
        const size_t nthreads = bytes / stride;
        double best = 1e18;
        for (int r = 0; r < reps + 3; r++) {
            double t0 = now_us();
            if (!myds_metal_pureread(d, w, bytes, 0, stride, (uint32_t)nthreads)) {
                printf("  %-30s  调用失败\n", cases[c].name);
                break;
            }
            myds_metal_synchronize();
            double dt = now_us() - t0;
            if (r >= 3 && dt < best) best = dt;
        }
        if (best < 1e17)
            printf("  %-30s %10.3f %12.1f %10zu\n",
                   cases[c].name, best / 1000.0,
                   (double)bytes / (best * 1e-6) / 1e9, nthreads);
    }

    printf("\n  ═══ 判读 ═══\n");
    printf("    MoE 实测（带 IQ2/Q2_K 解码）: 285 GB/s\n");
    printf("    纯读接近 700 ⇒ 瓶颈是解码指令吞吐\n");
    printf("    纯读也 ~285  ⇒ 瓶颈是访存模式\n");
    myds_metal_buffer_free(w);
    myds_metal_buffer_free(d);
    return 0;
}
