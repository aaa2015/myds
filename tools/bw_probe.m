/* 吞吐 vs 片段尺寸探测（诊断工具）
 *
 * 用途：回答「某段代码的有效带宽为什么低于峰值」。
 * 结论与用法见性能账本（原 docs/verification/2026-09-29-gpu-bottleneck-ablation.md §9.1，
 * 该文件未随仓库发布）。
 *
 * 手动编译（Makefile 未收录，避免改动公共构建目标）:
 *   make bin/myds && cc -O3 -Wall -Wextra -Iinclude -fobjc-arc -c tools/bw_probe.m -o build/bw_probe.o
 *   cc build/bw_probe.o build/myds_metal.o build/myds_gguf.o build/myds_log.o \\
 *      -framework Foundation -framework Metal -lm -lpthread -o bin/bw_probe
 * 运行（必须在仓库根，着色器按 cwd 的 metal/ 读取）:
 *   ./bin/bw_probe metal 50 250 268435456
 *
 * 原始说明：
 *
 * 修了 v1 的 bug（target 传 GB 数被当字节 ⇒ reps 恒为 2，测量被批边界开销污染）。
 *
 * v2 方法：**两点回归**。
 *   对每个尺寸测 reps_lo 与 reps_hi 两次，用差分分离：
 *     t(reps) = c0 + c1 * reps
 *       c1 = (t_hi - t_lo) / (reps_hi - reps_lo)   ← 每次 dispatch 的【边际成本】
 *       c0 = t_lo - c1 * reps_lo                   ← 批边界固定开销
 *   有效带宽 = per_call_bytes / c1                 ← 只看边际成本，不含批边界
 *
 * 避免 L2 污染：分配一个池（<= pool_cap），把池切成 n_sub 个不重叠的子视图轮转使用
 *   ⇒ 同一份数据不会被反复读入缓存（池远大于 L2）。
 *
 * ⚠️ 内存安全：池上限 pool_cap（默认 256 MB）。生产此刻驻留 151 GB / 192 GB。
 *    逐尺寸「分配 → 用 → 释放」。
 */
#import <Foundation/Foundation.h>
#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

#define Q8_0_BYTES_PER_ELEM 1.0625
/* 池必须显著大于 SLC（M2 Ultra 约 96 MB），否则同一份权重会被缓存反复命中，
 * 测出的带宽【偏高】—— 这是本工具第一版的实际缺陷。默认 1.5 GiB。 */
#define POOL_CAP_DEFAULT (1536u << 20)

typedef struct { uint32_t out_dim, in_dim; const char *label; } probe_t;

static double measure(const myds_metal_buffer_t *y, const myds_metal_buffer_t *x,
                      myds_metal_buffer_t **views, uint64_t n_sub,
                      uint32_t out_dim, uint32_t in_dim, uint64_t reps) {
    const double t0 = now_s();
    if (!myds_metal_begin_batch()) return -1.0;
    for (uint64_t i = 0; i < reps; i++) {
        const myds_metal_buffer_t *W = views[i % n_sub];
        if (!myds_metal_gemv((myds_metal_buffer_t *)y, W, x, out_dim, in_dim, GGML_TYPE_Q8_0)) {
            myds_metal_commit_batch();
            return -1.0;
        }
    }
    myds_metal_commit_batch();
    return now_s() - t0;
}

static int run_probe(const probe_t *p, uint64_t reps_lo, uint64_t reps_hi, size_t pool_cap) {
    const uint64_t w_bytes = (uint64_t)((double)p->out_dim * p->in_dim * Q8_0_BYTES_PER_ELEM);
    const uint64_t x_bytes = (uint64_t)p->in_dim * sizeof(float);
    const uint64_t y_bytes = (uint64_t)p->out_dim * sizeof(float);

    uint64_t n_sub = pool_cap / w_bytes;
    if (n_sub < 1) n_sub = 1;
    const size_t pool_bytes = (size_t)(n_sub * w_bytes);

    myds_metal_buffer_t *pool = myds_metal_buffer_alloc(pool_bytes);
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(x_bytes);
    myds_metal_buffer_t *y = myds_metal_buffer_alloc(y_bytes);
    if (!pool || !x || !y) {
        printf("  %-20s ★分配失败 (池 %.0f MB)\n", p->label, pool_bytes / 1048576.0);
        if (pool) myds_metal_buffer_free(pool);
        if (x) myds_metal_buffer_free(x);
        if (y) myds_metal_buffer_free(y);
        return 0;
    }
    myds_metal_buffer_t **views = calloc(n_sub, sizeof(*views));
    for (uint64_t i = 0; i < n_sub; i++)
        views[i] = myds_metal_buffer_sub_view(pool, (size_t)(i * w_bytes), (size_t)w_bytes);
    if (!views[0]) { printf("  %-20s sub_view 失败\n", p->label); return 0; }

    memset(myds_metal_buffer_data(pool), 0, pool_bytes);
    memset(myds_metal_buffer_data(x), 0, x_bytes);
    memset(myds_metal_buffer_data(y), 0, y_bytes);

    /* 预热 */
    myds_metal_begin_batch();
    for (int i = 0; i < 8; i++) myds_metal_gemv(y, views[i % n_sub], x, p->out_dim, p->in_dim, GGML_TYPE_Q8_0);
    myds_metal_commit_batch();

    const double t_lo = measure(y, x, views, n_sub, p->out_dim, p->in_dim, reps_lo);
    const double t_hi = measure(y, x, views, n_sub, p->out_dim, p->in_dim, reps_hi);
    if (t_lo < 0 || t_hi < 0) { printf("  %-20s gemv 失败\n", p->label); return 0; }

    const double c1 = (t_hi - t_lo) / (double)(reps_hi - reps_lo);   /* s / dispatch */
    const double c0 = t_lo - c1 * (double)reps_lo;                    /* s，批边界 */
    const double gbps = (double)w_bytes / c1 / 1e9;

    printf("  %-20s 片段=%7.2f MB  池=%3llu份/%.0f MB%s  c1=%8.2f µs/次  带宽=%6.1f GB/s (%4.1f%%)  c0=%7.1f µs\n",
           p->label, w_bytes / 1048576.0, (unsigned long long)n_sub, pool_bytes / 1048576.0,
           (n_sub == 1 ? "  ★仅1份⇒读数偏高,忽略" : (pool_bytes > pool_cap ? "  (超上限!)" : "")),
           c1 * 1e6, gbps, 100.0 * gbps / 773.0, c0 * 1e6);

    for (uint64_t i = 0; i < n_sub; i++) myds_metal_buffer_free(views[i]);
    free(views);
    myds_metal_buffer_free(pool);
    myds_metal_buffer_free(x);
    myds_metal_buffer_free(y);
    return 1;
}

int main(int argc, char **argv) {
    const char *shader = argc > 1 ? argv[1] : "metal";
    const uint64_t lo = argc > 2 ? strtoull(argv[2], NULL, 10) : 16;
    const uint64_t hi = argc > 3 ? strtoull(argv[3], NULL, 10) : 116;
    const size_t cap  = argc > 4 ? (size_t)strtoull(argv[4], NULL, 10) : POOL_CAP_DEFAULT;

    if (!myds_metal_init(shader)) { fprintf(stderr, "bw_probe: init 失败\n"); return 1; }
    printf("=== v2: 两点回归分离 批边界开销(c0) 与 每次 dispatch 边际成本(c1) ===\n");
    printf("reps: %llu → %llu   池上限: %.0f MB\n\n",
           (unsigned long long)lo, (unsigned long long)hi, cap / 1048576.0);

    printf("[A] in_dim=5120 —— 模型真实主干/输出头尺寸\n");
    probe_t a[] = {
        {     512, 5120, "attn_kv 512"      },
        {    1280, 5120, "attn_q_a 1280"    },
        {    5120, 5120, "attn_output_b"    },
        {    8192, 5120, "attn_output_a"    },
        {   32768, 5120, "attn_q_b 32768"   },
        {  129280, 5120, "output 129280"    },
    };
    for (unsigned i = 0; i < sizeof(a)/sizeof(a[0]); i++) run_probe(&a[i], lo, hi, cap);

    printf("\n[B] MoE 专家片段\n");
    probe_t b[] = {
        {   20480, 5120, "expert gate/up"   },
        {   20480, 1280, "expert 窄in1280"  },
    };
    for (unsigned i = 0; i < sizeof(b)/sizeof(b[0]); i++) run_probe(&b[i], lo, hi, cap);

    printf("\n[C] 碎片化代价（同 in_dim，递减 out_dim）\n");
    probe_t c[] = {
        {   32768, 5120, "大 32768"         },
        {    8192, 5120, "中 8192"          },
        {    1280, 5120, "碎 1280"          },
        {     512, 5120, "极碎 512"         },
    };
    for (unsigned i = 0; i < sizeof(c)/sizeof(c[0]); i++) run_probe(&c[i], lo, hi, cap);

    printf("\n=== 对照基准 ===\n");
    printf("  层循环 : 277 GB/s (9.71 GB / 35.08 ms / 760 dispatch ≈ 46 µs per dispatch)\n");
    printf("  输出头 : 573 GB/s (0.66 GB /  1.15 ms)\n");
    printf("  纯读上限: 773 GB/s\n");
    return 0;
}
