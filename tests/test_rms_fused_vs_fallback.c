/* 诊断：hc_sum_rmsnorm_scale_batch 融合路径 vs 降级路径的端到端对照。
 * 现有 test_v41_gpu 只比「融合① vs 融合②」，从未与降级路径比。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "myds_metal.h"

static void fill(myds_metal_buffer_t *b, unsigned seed) {
    float *p = myds_metal_buffer_data(b);
    for (size_t i = 0; i < myds_metal_buffer_size(b) / 4; i++)
        p[i] = (float)((int)((i + seed) % 53) - 26) / 64.0f;
}
static void dump(const char *tag, myds_metal_buffer_t *b, size_t n) {
    const float *p = myds_metal_buffer_data(b);
    printf("  %-22s", tag);
    for (size_t i = 0; i < n && i < 6; i++) printf(" %+.6f", p[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    const uint32_t dim = 5120u, flat_dim = 4u * dim, mix_rows = 24u;
    const uint32_t n_tok = (argc > 1) ? (uint32_t)atoi(argv[1]) : 3u;

    myds_metal_buffer_t *res    = myds_metal_buffer_alloc((size_t)n_tok * flat_dim * 4u);
    myds_metal_buffer_t *pre    = myds_metal_buffer_alloc((size_t)n_tok * 24u * 4u);
    myds_metal_buffer_t *nw     = myds_metal_buffer_alloc((size_t)dim * 4u);
    myds_metal_buffer_t *w_f32  = myds_metal_buffer_alloc((size_t)mix_rows * flat_dim * 4u);

    myds_metal_buffer_t *xn_fused = myds_metal_buffer_alloc((size_t)n_tok * dim * 4u);
    myds_metal_buffer_t *scales   = myds_metal_buffer_alloc((size_t)n_tok * 4u);
    myds_metal_buffer_t *mix_fused= myds_metal_buffer_alloc((size_t)n_tok * mix_rows * 4u);

    myds_metal_buffer_t *flat_sum = myds_metal_buffer_alloc((size_t)n_tok * flat_dim * 4u);
    myds_metal_buffer_t *flat_nrm = myds_metal_buffer_alloc((size_t)n_tok * flat_dim * 4u);
    myds_metal_buffer_t *xn_fb    = myds_metal_buffer_alloc((size_t)n_tok * dim * 4u);
    myds_metal_buffer_t *mix_fb   = myds_metal_buffer_alloc((size_t)n_tok * mix_rows * 4u);

    fill(res, 3u); fill(pre, 7u); fill(nw, 13u); fill(w_f32, 17u);

    /* ── 融合路径 ── */
    if (!myds_metal_begin_batch()) { printf("begin fail\n"); return 1; }
    if (!myds_metal_v41_hc_sum_rmsnorm_scale_batch(xn_fused, NULL, scales, res, pre, nw,
                                                   dim, n_tok, 96u, 1e-5f, true)) {
        printf("FUSED scale_batch REJECTED (return 0)\n");
        return 2;
    }
    myds_metal_barrier();
    if (!myds_metal_v41_gemv_batch_scaled(mix_fused, w_f32, res, scales, mix_rows, flat_dim, n_tok, GGML_TYPE_F32)) {
        printf("gemv_scaled fail\n"); return 3;
    }
    myds_metal_commit_batch();

    /* ── 降级路径 ── */
    if (!myds_metal_begin_batch()) { printf("begin fail\n"); return 1; }
    /* 正确的降级参照：对 res 直接做 rmsnorm（见 myds_v41.c:6974） */
    if (!myds_metal_v41_rmsnorm_batch(flat_nrm, res, NULL, flat_dim, n_tok, 1e-5f, false)) {
        printf("rmsnorm fail\n"); return 5;
    }
    myds_metal_barrier();
    if (!myds_metal_v41_gemv_batch(mix_fb, w_f32, flat_nrm, mix_rows, flat_dim, n_tok, GGML_TYPE_F32)) {
        printf("gemv fail\n"); return 6;
    }
    myds_metal_commit_batch();

    /* ── 顺带：融合① 的 x_norm（用作对照）── */
    myds_metal_begin_batch();
    myds_metal_v41_hc_sum_rmsnorm_batch(xn_fb, NULL, res, pre, nw, dim, n_tok, 96u, 1e-5f, true);
    myds_metal_commit_batch();

    printf("\n=== n_tok=%u ===\n", n_tok);
    dump("scales(fused)", scales, n_tok);
    dump("x_norm fused①", xn_fb, dim);
    dump("x_norm fused②", xn_fused, dim);
    dump("mix  fused", mix_fused, mix_rows);
    dump("mix  fallback", mix_fb, mix_rows);

    /* x_norm：融合① vs 融合② */
    {
        const float *a = myds_metal_buffer_data(xn_fb), *b = myds_metal_buffer_data(xn_fused);
        size_t n = (size_t)n_tok * dim, d = 0; double mx = 0;
        for (size_t i = 0; i < n; i++) {
            if (memcmp(&a[i], &b[i], 4)) d++;
            double e = fabs((double)a[i] - b[i]); if (e > mx) mx = e;
        }
        printf("\n  x_norm  融合①vs融合② : %zu/%zu bytes differ, max|Δ|=%.3e\n", d, n, mx);
    }
    /* mix：融合 vs 降级 */
    {
        const float *a = myds_metal_buffer_data(mix_fused), *b = myds_metal_buffer_data(mix_fb);
        size_t n = (size_t)n_tok * mix_rows, d = 0; double mx = 0;
        for (size_t i = 0; i < n; i++) {
            if (memcmp(&a[i], &b[i], 4)) d++;
            double e = fabs((double)a[i] - b[i]); if (e > mx) mx = e;
        }
        printf("  mix     融合 vs 降级   : %zu/%zu differ, max|Δ|=%.3e  %s\n",
               d, n, mx, d ? "<-- MISMATCH" : "OK");
    }
    /* 最终 logits 域等价：mix 逐元素最大相对误差 */
    return 0;
}
