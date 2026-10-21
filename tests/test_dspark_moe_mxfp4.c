/* MXFP4 routed MoE 单测：GPU 路径 vs CPU 参考，逐元素对比
 *
 * 为什么要这个单测：之前直接在整模型上试（每轮 2-3 分钟，只看到最终接受率），
 * 无法区分「MoE 算错」还是「drafter 本身不行」。这个单测 2 秒出结果。
 *
 * 形状用 drafter 真实值：dim=5120, expert_dim=2304, n_experts=128, top-3
 */
#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const float LUT[16] = {
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
};
static inline float e8m0(uint8_t e) {
    uint32_t bits = (e == 0) ? 0x00400000u : ((uint32_t)e << 23);
    float f; memcpy(&f, &bits, 4); return f;
}
/* 解一整行（cols 个元素，cols 必须是 32 的倍数） */
static void dequant_row(const uint8_t *row, uint32_t cols, float *out) {
    const uint32_t nb = cols / 32u;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * 17u;
        const float d = e8m0(blk[0]);
        for (int i = 0; i < 16; i++) {
            const uint8_t byte = blk[1 + i];
            out[b * 32u + 2u * i]     = LUT[byte & 0x0F] * d;
            out[b * 32u + 2u * i + 1] = LUT[(byte >> 4) & 0x0F] * d;
        }
    }
}

static uint32_t rng = 12345u;
static float rnd(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)((rng >> 8) & 0xFFFF) / 65536.0f - 0.5f;
}
static void fill_mxfp4(void *p, size_t bytes) {
    uint8_t *b = (uint8_t *)p;
    for (size_t i = 0; i < bytes; i++) b[i] = (uint8_t)((i * 37u + 11u) & 0xFF);
    /* e 设为 2^-3，避免全 0 或溢出 */
    for (size_t blk = 0; blk + 17 <= bytes; blk += 17) b[blk] = (uint8_t)(127 - 3);
}

int main(void) {
    const uint32_t dim = 5120, exp_dim = 2304, n_exp = 128, n_act = 3;
    const size_t gate_row = (size_t)(dim / 32u) * 17u;       /* 2720 B */
    const size_t gate_exp = gate_row * exp_dim;
    const size_t down_row = (size_t)(exp_dim / 32u) * 17u;   /* 1224 B */
    const size_t down_exp = down_row * dim;
    const float clamp_val = 10.0f;

    printf("  MXFP4 MoE 单测: dim=%u exp=%u experts=%u top=%u\n", dim, exp_dim, n_exp, n_act);
    printf("    gate 行 %zu B / 专家 %.2f MB\n", gate_row, gate_exp / 1e6);
    printf("    down 行 %zu B / 专家 %.2f MB\n\n", down_row, down_exp / 1e6);

    if (!myds_metal_init(NULL)) { printf("  metal init 失败\n"); return 1; }

    myds_metal_buffer_t *gate = myds_metal_buffer_alloc(gate_exp * n_exp);
    myds_metal_buffer_t *up   = myds_metal_buffer_alloc(gate_exp * n_exp);
    myds_metal_buffer_t *down = myds_metal_buffer_alloc(down_exp * n_exp);
    myds_metal_buffer_t *x    = myds_metal_buffer_alloc((size_t)dim * 4);
    myds_metal_buffer_t *out  = myds_metal_buffer_alloc((size_t)dim * 4);
    myds_metal_buffer_t *ids  = myds_metal_buffer_alloc((size_t)n_act * 4);
    myds_metal_buffer_t *wts  = myds_metal_buffer_alloc((size_t)n_act * 4);
    myds_metal_buffer_t *sg   = myds_metal_buffer_alloc((size_t)exp_dim * n_act * 4);
    myds_metal_buffer_t *su   = myds_metal_buffer_alloc((size_t)exp_dim * n_act * 4);
    myds_metal_buffer_t *sm   = myds_metal_buffer_alloc((size_t)exp_dim * n_act * 4);
    myds_metal_buffer_t *sd   = myds_metal_buffer_alloc((size_t)dim * n_act * 4);
    if (!gate || !up || !down || !x || !out || !ids || !wts || !sg || !su || !sm || !sd) {
        printf("  buffer 分配失败\n"); return 1;
    }

    fill_mxfp4(myds_metal_buffer_data(gate), gate_exp * n_exp);
    fill_mxfp4(myds_metal_buffer_data(up),   gate_exp * n_exp);
    fill_mxfp4(myds_metal_buffer_data(down), down_exp * n_exp);

    float *xp = (float *)myds_metal_buffer_data(x);
    for (uint32_t i = 0; i < dim; i++) xp[i] = rnd() * 0.1f;
    int32_t *ip = (int32_t *)myds_metal_buffer_data(ids);
    float *wp = (float *)myds_metal_buffer_data(wts);
    ip[0] = 5; ip[1] = 42; ip[2] = 100;
    wp[0] = 0.5f; wp[1] = 0.3f; wp[2] = 0.2f;

    int ok = myds_metal_dspark_moe_mxfp4(out, x, gate, up, down, ids, wts,
                                         sg, su, sm, sd, dim, exp_dim, n_exp, n_act, clamp_val);
    myds_metal_synchronize();
    if (!ok) { printf("  ❌ MoE 调用返回 0\n"); return 1; }
    const float *got = (const float *)myds_metal_buffer_data(out);
    /* 诊断：看中间量有没有写进去 */
    {
        const float *sg_v = (const float *)myds_metal_buffer_data(sg);
        const float *su_v = (const float *)myds_metal_buffer_data(su);
        const float *sm_v = (const float *)myds_metal_buffer_data(sm);
        const float *sd_v = (const float *)myds_metal_buffer_data(sd);
        double r1=0,r2=0,r3=0,r4=0;
        for (uint32_t i=0;i<exp_dim*3;i++){ r1+=sg_v[i]*sg_v[i]; r2+=su_v[i]*su_v[i]; r3+=sm_v[i]*sm_v[i]; }
        for (uint32_t i=0;i<dim*3;i++) r4+=sd_v[i]*sd_v[i];
        printf("    [中间量] scratch_gate RMS=%.4e  scratch_up RMS=%.4e\n",
               sqrt(r1/(exp_dim*3)), sqrt(r2/(exp_dim*3)));
        printf("    [中间量] scratch_mid  RMS=%.4e  scratch_down RMS=%.4e\n",
               sqrt(r3/(exp_dim*3)), sqrt(r4/(dim*3)));
    }

    /* ── CPU 参考 ── */
    float *g   = malloc((size_t)exp_dim * 4);
    float *u   = malloc((size_t)exp_dim * 4);
    float *mid = malloc((size_t)exp_dim * 4);
    float *acc = calloc(dim, 4);
    float *tmp = malloc((size_t)(dim > exp_dim ? dim : exp_dim) * 4);
    if (!g || !u || !mid || !acc || !tmp) { printf("  malloc 失败\n"); return 1; }

    const uint8_t *gate_b = (const uint8_t *)myds_metal_buffer_data(gate);
    const uint8_t *up_b   = (const uint8_t *)myds_metal_buffer_data(up);
    const uint8_t *down_b = (const uint8_t *)myds_metal_buffer_data(down);

    for (uint32_t k = 0; k < n_act; k++) {
        const int32_t e = ip[k];
        const uint8_t *gb = gate_b + (size_t)e * gate_exp;
        const uint8_t *ub = up_b   + (size_t)e * gate_exp;
        const uint8_t *db = down_b + (size_t)e * down_exp;
        for (uint32_t r = 0; r < exp_dim; r++) {
            dequant_row(gb + (size_t)r * gate_row, dim, tmp);
            double s = 0;
            for (uint32_t i = 0; i < dim; i++) s += (double)tmp[i] * xp[i];
            g[r] = (float)s;
            dequant_row(ub + (size_t)r * gate_row, dim, tmp);
            s = 0;
            for (uint32_t i = 0; i < dim; i++) s += (double)tmp[i] * xp[i];
            u[r] = (float)s;
        }
        for (uint32_t r = 0; r < exp_dim; r++) {
            float gv = g[r], uv = u[r];
            if (clamp_val > 1e-6f) {
                if (gv > clamp_val) gv = clamp_val;
                if (uv > clamp_val) uv = clamp_val;
                if (uv < -clamp_val) uv = -clamp_val;
            }
            mid[r] = (gv / (1.0f + expf(-gv))) * uv * wp[k];
        }
        for (uint32_t r = 0; r < dim; r++) {
            dequant_row(db + (size_t)r * down_row, exp_dim, tmp);
            double s = 0;
            for (uint32_t i = 0; i < exp_dim; i++) s += (double)tmp[i] * mid[i];
            acc[r] += (float)s;
        }
    }

    /* ── 逐级对比：重算第 0 个专家的 gate/up/mid，与 GPU 的 scratch 比 ── */
    {
        const int32_t e0 = ip[0];
        const uint8_t *gb0 = gate_b + (size_t)e0 * gate_exp;
        const uint8_t *ub0 = up_b   + (size_t)e0 * gate_exp;
        const uint8_t *db0 = down_b + (size_t)e0 * down_exp;
        float *cg = malloc((size_t)exp_dim * 4);
        float *cu = malloc((size_t)exp_dim * 4);
        float *cm = malloc((size_t)exp_dim * 4);
        float *cd = malloc((size_t)dim * 4);
        float *t2 = malloc((size_t)(dim > exp_dim ? dim : exp_dim) * 4);
        for (uint32_t r = 0; r < exp_dim; r++) {
            dequant_row(gb0 + (size_t)r * gate_row, dim, t2);
            double sd = 0; for (uint32_t i = 0; i < dim; i++) sd += (double)t2[i] * xp[i];
            cg[r] = (float)sd;
            dequant_row(ub0 + (size_t)r * gate_row, dim, t2);
            sd = 0; for (uint32_t i = 0; i < dim; i++) sd += (double)t2[i] * xp[i];
            cu[r] = (float)sd;
        }
        for (uint32_t r = 0; r < exp_dim; r++) {
            float gv = cg[r], uv = cu[r];
            if (gv > clamp_val) gv = clamp_val;
            if (uv > clamp_val) uv = clamp_val;
            if (uv < -clamp_val) uv = -clamp_val;
            cm[r] = (gv / (1.0f + expf(-gv))) * uv;   /* 不带 route weight */
        }
        for (uint32_t r = 0; r < dim; r++) {
            dequant_row(db0 + (size_t)r * down_row, exp_dim, t2);
            double sd = 0; for (uint32_t i = 0; i < exp_dim; i++) sd += (double)t2[i] * cm[i];
            cd[r] = (float)sd;
        }
        const float *gg = (const float *)myds_metal_buffer_data(sg);
        const float *gu = (const float *)myds_metal_buffer_data(su);
        const float *gm = (const float *)myds_metal_buffer_data(sm);
        const float *gd = (const float *)myds_metal_buffer_data(sd);
        double sg2=0,se2=0,sdg=0, sgm2=0,scm2=0,su2=0,scu2=0;
        for (uint32_t r = 0; r < exp_dim; r++) {
            sg2 += gg[r]*gg[r];
            su2 += gu[r]*gu[r]; scu2 += cu[r]*cu[r];
            sgm2 += gm[r]*gm[r]; scm2 += cm[r]*cm[r];
        }
        for (uint32_t r = 0; r < dim; r++) { sdg += gd[r]*gd[r]; }
        printf("    [逐级] gate : CPU %.4e  GPU %.4e\n", sqrt(su2>0?su2/exp_dim:0)*0+sqrt(sg2/exp_dim), sqrt(su2/exp_dim));
        printf("    [逐级] up   : CPU %.4e  GPU %.4e\n", sqrt(scu2/exp_dim), sqrt(su2/exp_dim));
        printf("    [逐级] mid  : CPU %.4e  GPU %.4e\n", sqrt(scm2/exp_dim), sqrt(sgm2/exp_dim));
        printf("    [逐级] down : CPU %.4e  GPU %.4e\n", 0.0, sqrt(sdg/dim));
        printf("    [逐级] gate[0..2]: CPU %.4e %.4e %.4e | GPU %.4e %.4e %.4e\n",
               cg[0], cg[1], cg[2], gg[0], gg[1], gg[2]);
        printf("    [逐级] mid [0..2]: CPU %.4e %.4e %.4e | GPU %.4e %.4e %.4e\n",
               cm[0], cm[1], cm[2], gm[0], gm[1], gm[2]);
        free(cg); free(cu); free(cm); free(cd); free(t2);
    }

    /* ── 对比 ── */
    double max_abs = 0, ref_sq = 0, diff_sq = 0, got_sq = 0;
    uint32_t worst = 0;
    for (uint32_t i = 0; i < dim; i++) {
        const double a = acc[i], b = got[i];
        const double d = fabs(a - b);
        if (d > max_abs) { max_abs = d; worst = i; }
        ref_sq += a * a; got_sq += b * b; diff_sq += d * d;
    }
    const double ref_rms = sqrt(ref_sq / dim), got_rms = sqrt(got_sq / dim);
    const double diff_rms = sqrt(diff_sq / dim);
    const double rel = diff_rms / (ref_rms > 1e-30 ? ref_rms : 1);

    printf("  ── 结果 ──\n");
    printf("    CPU 参考 RMS : %.6e\n", ref_rms);
    printf("    GPU  输出 RMS: %.6e\n", got_rms);
    printf("    差异 RMS     : %.6e  (相对 %.4f)\n", diff_rms, rel);
    printf("    最大绝对差   : %.6e  @ idx %u\n", max_abs, worst);
    printf("    CPU[0..3]    : %.6e %.6e %.6e %.6e\n", acc[0], acc[1], acc[2], acc[3]);
    printf("    GPU[0..3]    : %.6e %.6e %.6e %.6e\n", got[0], got[1], got[2], got[3]);
    printf("\n    %s\n", rel < 1e-3 ? "✅ 一致（MXFP4 MoE 正确）"
                                  : "❌ 不一致（MXFP4 MoE 有 bug）");
    free(g); free(u); free(mid); free(acc); free(tmp);
    return rel < 1e-3 ? 0 : 1;
}
