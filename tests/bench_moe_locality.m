/* MoE 局部性判定：字节数完全相同，只改专家在池中的分布
 *
 * 假设 A（§13 的根因）：专家散落在 3.82 GB 池中 → DRAM 页/TLB 局部性差 → 带宽低
 * 假设 B：Q2_K/IQ2 反量化算力受限 → 带宽低
 *
 * 判别方法：
 *   同一批 token、同样 6 个专家、同样读 59.6 MB，
 *   只把专家 id 从 {0..5}（连续，同页）换成 {0,64,128,192,256,320}（散开）。
 *   - 散开明显更慢 ⇒ A 成立
 *   - 两者一样      ⇒ A 不成立，瓶颈在解码算力
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

static void fill_iq2(void *ptr, size_t nblocks) {
    uint8_t *p = (uint8_t *)ptr;
    for (size_t b = 0; b < nblocks; b++) {
        uint8_t *blk = p + b * 66;
        for (int i = 0; i < 66; i++) blk[i] = (uint8_t)((b * 37 + i * 101) & 0xFF);
        uint16_t d = 0x2c00;
        memcpy(blk, &d, 2);
    }
}

int main(int argc, char **argv) {
    const uint32_t dim = 5120, exp_dim = 2304, n_exp = 384, n_act = 6;
    const uint32_t n_tok = argc > 1 ? (uint32_t)atoi(argv[1]) : 1;
    const int reps = argc > 2 ? atoi(argv[2]) : 60;
    const int scoped = argc > 3 ? atoi(argv[3]) : 1;   /* 1=生产融合路径 */

    if (!myds_metal_init(NULL)) { printf("metal init 失败\n"); return 1; }

    const size_t g_bytes = (size_t)n_exp * exp_dim * (dim / 256) * 66;
    const size_t d_bytes = (size_t)n_exp * dim * (exp_dim / 256) * 84;
    printf("  MoE 局部性测试: tok=%u 专家池 %.2f GB\n", n_tok, (2.0 * g_bytes + d_bytes) / 1e9);

    myds_metal_buffer_t *gate = myds_metal_buffer_alloc(g_bytes);
    myds_metal_buffer_t *up   = myds_metal_buffer_alloc(g_bytes);
    myds_metal_buffer_t *down = myds_metal_buffer_alloc(d_bytes);
    myds_metal_buffer_t *x    = myds_metal_buffer_alloc((size_t)dim * n_tok * 4);
    myds_metal_buffer_t *out  = myds_metal_buffer_alloc((size_t)dim * n_tok * 4);
    myds_metal_buffer_t *ids  = myds_metal_buffer_alloc((size_t)n_act * n_tok * 4);
    myds_metal_buffer_t *wts  = myds_metal_buffer_alloc((size_t)n_act * n_tok * 4);
    myds_metal_buffer_t *sg   = myds_metal_buffer_alloc((size_t)n_act * n_tok * exp_dim * 4);
    myds_metal_buffer_t *su   = myds_metal_buffer_alloc((size_t)n_act * n_tok * exp_dim * 4);
    myds_metal_buffer_t *sm   = myds_metal_buffer_alloc((size_t)n_act * n_tok * exp_dim * 4);
    myds_metal_buffer_t *sd   = myds_metal_buffer_alloc((size_t)n_act * n_tok * dim * 4);
    if (!gate || !up || !down || !x || !out || !ids || !wts || !sg || !su || !sm || !sd) {
        printf("buffer 分配失败\n"); return 1;
    }
    fill_iq2(myds_metal_buffer_data(gate), g_bytes / 66);
    fill_iq2(myds_metal_buffer_data(up),   g_bytes / 66);
    fill_iq2(myds_metal_buffer_data(down), d_bytes / 84);

    float *xp = (float *)myds_metal_buffer_data(x);
    for (size_t i = 0; i < (size_t)dim * n_tok; i++) xp[i] = (float)(((int)(i * 43) % 201) - 100) / 500.0f;
    float *wp = (float *)myds_metal_buffer_data(wts);

    struct { const char *name; int32_t ids[6]; } sets[5] = {
        { "连续 {0..5}          ", { 0, 1, 2, 3, 4, 5 } },
        { "散开 {0,64,..,320}    ", { 0, 64, 128, 192, 256, 320 } },
        { "半散 {0,1,2,192,193,194}", { 0, 1, 2, 192, 193, 194 } },
        { "12 专家 (0,32,..352) ", { 0, 32, 64, 96, 128, 160 } },   /* 见下：额外两轮 */
        { "全池 384 专家        ", { 0, 1, 2, 3, 4, 5 } },
    };

    const double touched = (double)(2.0 * g_bytes + d_bytes) * n_act / n_exp;
    printf("\n  %-24s %10s %10s %12s\n", "专家分布", "耗时(ms)", "有效GB/s", "相对连续");

    for (int s = 0; s < 3; s++) {
        int32_t *ip = (int32_t *)myds_metal_buffer_data(ids);
        for (uint32_t t = 0; t < n_tok; t++)
            for (uint32_t k = 0; k < n_act; k++) {
                ip[t * n_act + k] = sets[s].ids[k];
                wp[t * n_act + k] = 1.0f / n_act;
            }
        int ok = scoped
            ? myds_metal_routed_moe_scoped(out, x, gate, up, down, ids, wts,
                                           sg, su, sm, sd, dim, exp_dim, n_exp, n_act,
                                           GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K,
                                           10.0f, n_tok == 1)
            : myds_metal_routed_moe_batch(out, x, gate, up, down, ids, wts,
                                          sg, su, sm, sd, dim, exp_dim, n_exp, n_act, n_tok,
                                          GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f);
        myds_metal_synchronize();
        if (!ok) { printf("  %-24s  不被支持\n", sets[s].name); continue; }
        for (int i = 0; i < 4; i++)
            scoped ? myds_metal_routed_moe_scoped(out, x, gate, up, down, ids, wts, sg, su, sm, sd,
                                                  dim, exp_dim, n_exp, n_act,
                                                  GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K,
                                                  10.0f, n_tok == 1)
                   : myds_metal_routed_moe_batch(out, x, gate, up, down, ids, wts, sg, su, sm, sd,
                                                 dim, exp_dim, n_exp, n_act, n_tok,
                                                 GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f);
        myds_metal_synchronize();

        double best = 1e18;
        for (int r = 0; r < 3; r++) {
            double t0 = now_us();
            for (int i = 0; i < reps; i++)
                scoped ? (void)myds_metal_routed_moe_scoped(out, x, gate, up, down, ids, wts, sg, su, sm, sd,
                                                            dim, exp_dim, n_exp, n_act,
                                                            GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K,
                                                            10.0f, n_tok == 1)
                       : (void)myds_metal_routed_moe_batch(out, x, gate, up, down, ids, wts, sg, su, sm, sd,
                                                           dim, exp_dim, n_exp, n_act, n_tok,
                                                           GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f);
            myds_metal_synchronize();
            double dt = (now_us() - t0) / reps / 1000.0;
            if (dt < best) best = dt;
        }
        static double base = 0;
        if (s == 0) base = best;
        printf("  %-24s %10.3f %10.1f %11.2fx\n", sets[s].name, best, touched / (best * 1e-3) / 1e9, best / base);
    }

    /* 真实路由形态：每个 token 用不同的 6 个专家 ⇒ 不同专家数随 token 增长 */
    if (n_tok >= 2) {
        int32_t *ip = (int32_t *)myds_metal_buffer_data(ids);
        int used[384] = {0};
        int distinct = 0;
        for (uint32_t t = 0; t < n_tok; t++)
            for (uint32_t k = 0; k < n_act; k++) {
                int e = (int)((t * n_act + k) % n_exp);
                ip[t * n_act + k] = e;
                wp[t * n_act + k] = 1.0f / n_act;
                if (!used[e]) { used[e] = 1; distinct++; }
            }
        const double bytes = (double)(2.0 * g_bytes + d_bytes) * distinct / n_exp;
        int okd = myds_metal_routed_moe_scoped(out, x, gate, up, down, ids, wts, sg, su, sm, sd,
                                              dim, exp_dim, n_exp, n_act,
                                              GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f, n_tok == 1);
        myds_metal_synchronize();
        if (!okd) { printf("  每 token 不同专家: 调用返回 0（该组合被拒），跳过\n"); }
        double best = 1e18;
        for (int r = 0; r < 3; r++) {
            double t0 = now_us();
            for (int i = 0; i < reps; i++)
                myds_metal_routed_moe_scoped(out, x, gate, up, down, ids, wts, sg, su, sm, sd,
                                             dim, exp_dim, n_exp, n_act,
                                             GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 10.0f, n_tok == 1);
            myds_metal_synchronize();
            double dt = (now_us() - t0) / reps / 1000.0;
            if (dt < best) best = dt;
        }
        if (okd)
            printf("  %-24s %10.3f %10.1f  (读 %.1f MB)\n",
                   "每 token 不同专家", best, bytes / (best * 1e-3) / 1e9, bytes / 1e6);
    }

    printf("\n  读固定 %.1f MB；只有专家 id 不同 ⇒ 差异只能来自地址分布\n", touched / 1e6);
    return 0;
}
