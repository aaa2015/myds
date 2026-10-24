/* indexer key 方向距离窗口（MYDS_INDEXER_KEY_WINDOW）的对照测试。
 *
 * 为什么需要这个测试：窗口是本仓库唯一一条【非 0-ULP】的近似路径，而它此前
 * 一条断言都没有 —— 全靠读代码审。0-ULP 门禁只覆盖 window=0。
 *
 * 做法：同一个 fixture 跑两次内核，用两条独立实现互校。
 *   参考：MYDS_INDEXER_CHUNK_EXACT=0 → kernel_glm_indexer_score_chunk_direct，
 *         该内核根本不读 args.window ⇒ 永远是完整因果窗口。
 *   候选：MYDS_INDEXER_CHUNK_EXACT=1 → kernel_myds_indexer_score_chunk_exact，
 *         带窗口下界。
 * 断言（逐 cell）：
 *   vis = (start + tok + 1) / ratio
 *   lo  = window && vis > window ? vis - window : 0
 *   row ∈ [lo, vis)  ⇒ 与参考臂【数值一致】（容差内，见下）
 *   row ∉ [lo, vis)  ⇒ 必须是 -INFINITY（逐位）
 *
 * ★ 为什么不用逐位相同：这两个内核的归约顺序本就不同 —— direct 走
 *   scratch[tid] + 树形归约，exact 走 float4 dot + simd_sum。实测在
 *   tok=0,row=0 就差 1 ULP（bd0e5310 vs bd0e530f），且这个差异在本次改动
 *   【之前就存在】（用原始 metallib 复验过）。dsv4_misc.metal 里「100% 逐位
 *   全同（bit-exact）」那句关于这对内核的话不成立，已在此备注，不要在
 *   kernel 注释里继续引用。窗口的语义约束用「窗口外的值必须恰好是 -INF」
 *   来守 —— 那条是逐位的。
 *
 * ★ 这条断言正是旧缺陷的探针：早退曾用 max_vis（块内最晚 token）算下界，
 *   于是块内靠前 token 仍可见的行被整块填 -INF。选 window=8 时的
 *   n_tokens=24/n_comp=48 配置会命中那个 tile 边界（见 run() 里的注释）。
 *
 * 整个进程只解析一次 window（pthread_once），所以不同窗口值靠 Makefile
 * 分别起进程跑：无 env、=8、=20。
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); return 0; } } while (0)

#define HEADS 32u
#define DIM   128u

static float sample(uint32_t i, uint32_t seed) {
    uint32_t x = (i + 1u) * 747796405u + seed * 2891336453u;
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737u;
    x = (x >> 22) ^ x;
    return ((int32_t)(x % 8191u) - 4095) / 4096.0f;
}

static uint32_t window_from_env(void) {
    const char *e = getenv("MYDS_INDEXER_KEY_WINDOW");
    if (!e || !*e) return 0u;
    const long v = atol(e);
    return v > 0 ? (uint32_t)v : 0u;
}

struct fixture {
    uint32_t n_comp, n_tokens, start, ratio;
    myds_metal_buffer_t *q, *w, *cache, *scores;
};

static void cleanup(struct fixture *f) {
    myds_metal_buffer_t *b[] = { f->q, f->w, f->cache, f->scores };
    for (size_t i = 0; i < sizeof(b) / sizeof(*b); i++) myds_metal_buffer_free(b[i]);
    f->q = f->w = f->cache = f->scores = NULL;
}

static int init(struct fixture *f, uint32_t n_comp, uint32_t n_tokens,
                uint32_t start, uint32_t ratio) {
    memset(f, 0, sizeof(*f));
    f->n_comp = n_comp;
    f->n_tokens = n_tokens;
    f->start = start;
    f->ratio = ratio;
    f->q = myds_metal_buffer_alloc((size_t)n_tokens * HEADS * DIM * sizeof(float));
    f->w = myds_metal_buffer_alloc((size_t)n_tokens * HEADS * sizeof(float));
    f->cache = myds_metal_buffer_alloc((size_t)n_comp * DIM * sizeof(float));
    f->scores = myds_metal_buffer_alloc((size_t)n_comp * n_tokens * sizeof(float));
    CHECK(f->q && f->w && f->cache && f->scores);
    myds_metal_buffer_t *bs[] = { f->q, f->w, f->cache };
    for (int k = 0; k < 3; k++) {
        float *p = myds_metal_buffer_data(bs[k]);
        const size_t n = myds_metal_buffer_size(bs[k]) / sizeof(float);
        for (size_t j = 0; j < n; j++) p[j] = sample((uint32_t)j, (uint32_t)k + 1u);
    }
    return 1;
}

static int run_scores(struct fixture *f) {
    CHECK(myds_metal_begin_batch());
    CHECK(myds_v41_gpu_indexscores_chunk(f->scores, f->q, f->w, f->cache,
                                         f->n_comp, HEADS, f->n_tokens,
                                         f->start, f->ratio));
    CHECK(myds_metal_commit_batch());
    return 1;
}

static int run(uint32_t n_comp, uint32_t n_tokens, uint32_t start, uint32_t ratio) {
    struct fixture f;
    CHECK(init(&f, n_comp, n_tokens, start, ratio));
    const uint32_t window = window_from_env();
    const size_t cells = (size_t)n_comp * n_tokens;
    float *ref = malloc(cells * sizeof(float));
    float *got = malloc(cells * sizeof(float));
    if (!ref || !got) { free(ref); free(got); cleanup(&f); return 0; }

    /* 参考臂：direct 内核，不读 window。 */
    setenv("MYDS_INDEXER_CHUNK_EXACT", "0", 1);
    CHECK(run_scores(&f));
    memcpy(ref, myds_metal_buffer_data(f.scores), cells * sizeof(float));

    /* 候选臂：exact 内核，带窗口下界。 */
    setenv("MYDS_INDEXER_CHUNK_EXACT", "1", 1);
    CHECK(run_scores(&f));
    memcpy(got, myds_metal_buffer_data(f.scores), cells * sizeof(float));

    uint64_t in_window = 0, out_window = 0, killed_by_window = 0, ref_killed = 0;
    for (uint32_t tok = 0; tok < n_tokens; tok++) {
        const uint32_t vis = (start + tok + 1u) / ratio;
        const uint32_t lo = (window && vis > window) ? (vis - window) : 0u;
        for (uint32_t row = 0; row < n_comp; row++) {
            const size_t i = (size_t)tok * n_comp + row;
            const bool ref_inf = (ref[i] == -INFINITY);
            if (row >= vis) {
                if (!ref_inf) {
                    fprintf(stderr,
                            "FAIL reference arm disagrees with causality: tok=%u row=%u vis=%u ref=%g\n",
                            tok, row, vis, (double)ref[i]);
                    free(ref); free(got); cleanup(&f); return 0;
                }
                ref_killed++;
            }
            if (row >= vis) continue;
            if (row >= lo) {
                /* 窗口内：候选臂必须与参考臂数值一致，且不是 -INF。 */
                if (ref_inf) continue;   /* 参考臂自己的因果判定，上面已单独校验 */
                const float d = fabsf(got[i] - ref[i]);
                const float m = fmaxf(1.0f, fmaxf(fabsf(ref[i]), fabsf(got[i])));
                if (!(d <= 1e-6f * m)) {
                    fprintf(stderr,
                            "FAIL window=%u tok=%u row=%u vis=%u lo=%u: ref=%.9g got=%.9g diff=%.3g\n",
                            window, tok, row, vis, lo, (double)ref[i], (double)got[i], (double)d);
                    free(ref); free(got); cleanup(&f); return 0;
                }
                in_window++;
            } else {
                if (got[i] != -INFINITY) {
                    fprintf(stderr,
                            "FAIL window=%u tok=%u row=%u vis=%u lo=%u: expected -INF, got %.9g\n",
                            window, tok, row, vis, lo, (double)got[i]);
                    free(ref); free(got); cleanup(&f); return 0;
                }
                out_window++;
                killed_by_window++;
            }
        }
    }

    /* 防「窗口被静默忽略」：window>0 且确有 token 超过它时，必须真的杀掉过行。
     * 否则这个测试对窗口分支是空转的 —— 空转的绿灯比红灯更危险。 */
    if (window && n_tokens > window) {
        if (!killed_by_window) {
            fprintf(stderr, "FAIL window=%u was ignored: no row was excluded by the window\n", window);
            free(ref); free(got); cleanup(&f); return 0;
        }
    }
    if (!window && out_window) {
        fprintf(stderr, "FAIL window=0 excluded %llu rows below the causal bound\n",
                (unsigned long long)out_window);
        free(ref); free(got); cleanup(&f); return 0;
    }

    printf("PASS n_comp=%u n_tokens=%u start=%u ratio=%u window=%u "
           "in_window=%llu window_excluded=%llu causal_excluded=%llu invariant-ok\n",
           n_comp, n_tokens, start, ratio, window, (unsigned long long)in_window,
           (unsigned long long)killed_by_window, (unsigned long long)ref_killed);
    free(ref);
    free(got);
    cleanup(&f);
    return 1;
}

int main(void) {
    /* 第一组：ratio=1、24 token × 48 键。一个 token tile 覆盖 8 个 token，
     * 可见上限在 tile 内跨 8 行 ⇒ window=8 时 tile_min_vis 与 max_vis-window
     * 必然落在同一个 16 行块的两侧，正是旧早退缺陷（误杀靠前 token 的行）的
     * 触发形状。 */
    if (!run(48, 24, 0, 1)) return 1;
    /* 第二组：非零起始 + ratio=4，覆盖「原始位置 → 压缩键」的除法路径。
     * start=200 是刻意的：ratio=4 下 24 个 token 只把 vis 推进 6 行，起始位置
     * 不够大时 lo 恒为 0，窗口分支就是空转（第一版就踩了：window=8/20 一条行
     * 都没排除，防「窗口被静默忽略」的断言直接把它拦住）。 */
    if (!run(64, 24, 200, 4)) return 1;
    return 0;
}
