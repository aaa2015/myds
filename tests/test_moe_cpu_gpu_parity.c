/* CPU MoE 与 GPU MoE 的数值一致性测试
 *
 * 用真实模型权重（blk.<L>.ffn_gate_exps / up_exps / down_exps），
 * 同一批 token、同一组专家，两边各算一遍，比较输出。
 *
 * 判定：
 *   - 相对误差 ~1e-6  ⇒ CPU 解码（IQ2_XXS / Q2_K 布局与查表）正确
 *   - 相对误差 >1e-2   ⇒ 解码走错布局
 */
#include "myds_metal.h"
#include "myds_gguf.h"
#include "myds_moe_cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static float half_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -1; uint32_t m = mant;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s model.gguf layer [n_tok=8] [nthreads=16]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const int layer = atoi(argv[2]);
    const uint32_t n_tok = argc > 3 ? (uint32_t)atoi(argv[3]) : 8u;
    const uint32_t nthr  = argc > 4 ? (uint32_t)atoi(argv[4]) : 16u;
    const int single     = argc > 5 ? atoi(argv[5]) : 0;

    gguf_file_t *gf = gguf_open(path);
    if (!gf) { fprintf(stderr, "打开模型失败\n"); return 1; }

    char k1[128], k2[128], k3[128];
    snprintf(k1, sizeof k1, "blk.%d.ffn_gate_exps.weight", layer);
    snprintf(k2, sizeof k2, "blk.%d.ffn_up_exps.weight",   layer);
    snprintf(k3, sizeof k3, "blk.%d.ffn_down_exps.weight", layer);
    const gguf_tensor_t *tg = gguf_find_tensor(gf, k1);
    const gguf_tensor_t *tu = gguf_find_tensor(gf, k2);
    const gguf_tensor_t *td = gguf_find_tensor(gf, k3);
    if (!tg || !tu || !td) { fprintf(stderr, "找不到层 %d 的专家张量\n", layer); return 1; }

    const uint32_t dim     = (uint32_t)tg->dims[0];      /* 5120 输入 */
    const uint32_t exp_dim = (uint32_t)tg->dims[1];      /* 2304 */
    const uint32_t n_exp   = (uint32_t)tg->dims[2];      /* 384 */
    const uint32_t n_act   = 6u;
    printf("  层 %d: dim=%u exp_dim=%u experts=%u tok=%u thr=%u\n",
           layer, dim, exp_dim, n_exp, n_tok, nthr);
    printf("  gate type=%d up type=%d down type=%d (期望 16/16/10)\n",
           (int)tg->type, (int)tu->type, (int)td->type);

    if (!myds_metal_init(NULL)) { fprintf(stderr, "metal init 失败\n"); return 1; }

    const size_t x_bytes  = (size_t)n_tok * dim * sizeof(float);
    const size_t o_bytes  = (size_t)n_tok * dim * sizeof(float);
    const size_t id_bytes = (size_t)n_tok * n_act * sizeof(int32_t);
    const size_t w_bytes  = (size_t)n_tok * n_act * sizeof(float);
    const size_t mid_scr  = (size_t)n_tok * n_act * exp_dim * sizeof(float);
    const size_t dn_scr   = (size_t)n_tok * n_act * dim * sizeof(float);

    myds_metal_buffer_t *xb = myds_metal_buffer_alloc(x_bytes);
    myds_metal_buffer_t *ob = myds_metal_buffer_alloc(o_bytes);
    myds_metal_buffer_t *idb = myds_metal_buffer_alloc(id_bytes);
    myds_metal_buffer_t *wb = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *sg = myds_metal_buffer_alloc(mid_scr);
    myds_metal_buffer_t *su = myds_metal_buffer_alloc(mid_scr);
    myds_metal_buffer_t *sm = myds_metal_buffer_alloc(mid_scr);
    myds_metal_buffer_t *sd = myds_metal_buffer_alloc(dn_scr);
    /* 专家权重必须放进 Metal buffer（GPU 侧只能读 MTLBuffer） */
    printf("  拷贝权重到 Metal buffer: %.2f GB\n",
           (tg->size_bytes + tu->size_bytes + td->size_bytes) / 1e9);
    myds_metal_buffer_t *gb = myds_metal_buffer_alloc(tg->size_bytes);
    myds_metal_buffer_t *ub = myds_metal_buffer_alloc(tu->size_bytes);
    myds_metal_buffer_t *db = myds_metal_buffer_alloc(td->size_bytes);
    if (!xb || !ob || !idb || !wb || !sg || !su || !sm || !sd || !gb || !ub || !db) {
        fprintf(stderr, "buffer 分配失败\n"); return 1;
    }
    memcpy(myds_metal_buffer_data(gb), tg->data, tg->size_bytes);
    memcpy(myds_metal_buffer_data(ub), tu->data, tu->size_bytes);
    memcpy(myds_metal_buffer_data(db), td->data, td->size_bytes);

    float *xp = (float *)myds_metal_buffer_data(xb);
    int32_t *ip = (int32_t *)myds_metal_buffer_data(idb);
    float *wp = (float *)myds_metal_buffer_data(wb);
    srand(12345);
    for (size_t i = 0; i < (size_t)n_tok * dim; i++)
        xp[i] = (float)((rand() % 2001) - 1000) / 4000.0f;
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t k = 0; k < n_act; k++) {
            ip[t * n_act + k] = single ? 0 : (int32_t)((t * 37u + k * 101u + 11u) % n_exp);
            wp[t * n_act + k] = single ? (k == 0 ? 1.0f : 0.0f) : 0.35f / (float)(k + 1);
        }

    const float clamp_val = 10.0f;
    if (!myds_metal_routed_moe_batch(ob, xb, gb, ub, db, idb, wb,
                                     sg, su, sm, sd, dim, exp_dim, n_exp, n_act, n_tok,
                                     tg->type, tu->type, td->type, clamp_val)) {
        fprintf(stderr, "GPU routed_moe_batch 失败\n"); return 1;
    }
    myds_metal_synchronize();

    float *ref = (float *)malloc(o_bytes);
    memcpy(ref, myds_metal_buffer_data(ob), o_bytes);

    float *cpu_out = (float *)malloc(o_bytes);
    double t0 = now_s();
    if (!myds_cpu_moe_tokens(tg->data, tu->data, td->data, dim, exp_dim, n_exp,
                             xp, ip, wp, cpu_out, n_tok, n_act, clamp_val, nthr)) {
        fprintf(stderr, "CPU MoE 失败\n"); return 1;
    }
    double cpu_s = now_s() - t0;

    /* GPU 计时 */
    double gbest = 1e9;
    for (int r = 0; r < 3; r++) {
        double g0 = now_s();
        myds_metal_routed_moe_batch(ob, xb, gb, ub, db, idb, wb,
                                    sg, su, sm, sd, dim, exp_dim, n_exp, n_act, n_tok,
                                    tg->type, tu->type, td->type, clamp_val);
        myds_metal_synchronize();
        double dt = now_s() - g0;
        if (dt < gbest) gbest = dt;
    }

    /* 诊断：比较 token0/expert0 的 gate 中间量（隔离子矩阵解码问题） */
    {
        const int32_t e0 = ip[0];
        const size_t g_row = (size_t)(dim / 256u) * 66u;
        float *cg = (float *)malloc((size_t)exp_dim * sizeof(float));
        if (myds_cpu_moe_debug_gemv((const uint8_t *)tg->data + (size_t)e0 * exp_dim * g_row,
                                    exp_dim, dim / 256u, 0, xp, cg)) {
            const float *gg = (const float *)myds_metal_buffer_data(sg);
            double s1 = 0, s2 = 0, sd = 0;
            for (uint32_t r = 0; r < exp_dim; r++) {
                double a = gg[r], b = cg[r];
                s1 += a * a; s2 += b * b; sd += (a - b) * (a - b);
            }
            printf("\n  ── 诊断 gate[tok0,exp%lld] ──\n", (long long)e0);
            printf("    GPU RMS=%.6e  CPU RMS=%.6e  差异 RMS=%.6e (相对 %.4f)\n",
                   sqrt(s1 / exp_dim), sqrt(s2 / exp_dim), sqrt(sd / exp_dim),
                   sqrt(sd / (s1 > 0 ? s1 : 1)));
            printf("    GPU 前4: %.6e %.6e %.6e %.6e\n", gg[0], gg[1], gg[2], gg[3]);
            printf("    CPU 前4: %.6e %.6e %.6e %.6e\n", cg[0], cg[1], cg[2], cg[3]);
        }
        free(cg);
    }

    /* 隔离诊断：MM 路径把 swiglu 结果写进 scratch_mid（f16，行 = tok*n_act+k）。
     * 用 GPU 的 mid 单独验 Q2_K 解码；用 CPU 的 mid 验 IQ2 解码。 */
    if (single) {
        const int32_t e0 = 0;
        const size_t g_row = (size_t)(dim / 256u) * 66u;
        const size_t d_row = (size_t)(exp_dim / 256u) * 84u;
        float *cg = (float *)malloc((size_t)exp_dim * sizeof(float));
        float *cu = (float *)malloc((size_t)exp_dim * sizeof(float));
        float *cm = (float *)malloc((size_t)exp_dim * sizeof(float));
        float *gm = (float *)malloc((size_t)exp_dim * sizeof(float));
        float *cd = (float *)malloc((size_t)dim * sizeof(float));
        const uint16_t *hmid = (const uint16_t *)myds_metal_buffer_data(sm);
        for (uint32_t r = 0; r < exp_dim; r++) {
            gm[r] = half_to_f32(hmid[r]);
        }
        myds_cpu_moe_debug_gemv((const uint8_t *)tg->data + (size_t)e0 * exp_dim * g_row,
                                exp_dim, dim / 256u, 0, xp, cg);
        myds_cpu_moe_debug_gemv((const uint8_t *)tu->data + (size_t)e0 * exp_dim * g_row,
                                exp_dim, dim / 256u, 0, xp, cu);
        for (uint32_t r = 0; r < exp_dim; r++) {
            float gv = cg[r], uv = cu[r];
            if (clamp_val > 1.0e-6f) { if (gv > clamp_val) gv = clamp_val;
                if (uv > clamp_val) uv = clamp_val; if (uv < -clamp_val) uv = -clamp_val; }
            cm[r] = (gv / (1.0f + expf(-gv))) * uv;   /* w=1 */
        }
        double sgm = 0, scm = 0, sdm = 0;
        for (uint32_t r = 0; r < exp_dim; r++) {
            sgm += gm[r] * gm[r]; scm += cm[r] * cm[r];
            sdm += (gm[r] - cm[r]) * (gm[r] - cm[r]);
        }
        printf("\n  ── 隔离 1: swiglu 中间量（mid, 2304 维）──\n");
        printf("    GPU mid RMS=%.6e  CPU mid RMS=%.6e  差=%.6e (相对 %.4f)\n",
               sqrt(sgm / exp_dim), sqrt(scm / exp_dim), sqrt(sdm / exp_dim),
               sqrt(sdm / (sgm > 0 ? sgm : 1)));
        myds_cpu_moe_debug_gemv((const uint8_t *)td->data + (size_t)e0 * dim * d_row,
                                dim, exp_dim / 256u, 1, gm, cd);
        const float *gout = (const float *)myds_metal_buffer_data(ob);
        double sg2 = 0, sc2 = 0, sd2 = 0;
        for (uint32_t r = 0; r < dim; r++) {
            sg2 += gout[r] * gout[r]; sc2 += cd[r] * cd[r];
            sd2 += (gout[r] - cd[r]) * (gout[r] - cd[r]);
        }
        printf("\n  ── 隔离 2: 用 GPU 的 mid 单验 Q2_K down ──\n");
        printf("    GPU out RMS=%.6e  CPU down RMS=%.6e  差=%.6e (相对 %.4f)\n",
               sqrt(sg2 / dim), sqrt(sc2 / dim), sqrt(sd2 / dim),
               sqrt(sd2 / (sg2 > 0 ? sg2 : 1)));
        free(cg); free(cu); free(cm); free(gm); free(cd);
    }

    double max_abs = 0, max_rel = 0, ref_rms = 0, diff_rms = 0;
    for (size_t i = 0; i < (size_t)n_tok * dim; i++) {
        double a = ref[i], b = cpu_out[i];
        double d = fabs(a - b);
        double denom = fabs(a) > 1e-3 ? fabs(a) : 1e-3;
        if (d > max_abs) max_abs = d;
        if (d / denom > max_rel) max_rel = d / denom;
        ref_rms += a * a;
        diff_rms += d * d;
    }
    ref_rms = sqrt(ref_rms / ((double)n_tok * dim));
    diff_rms = sqrt(diff_rms / ((double)n_tok * dim));

    printf("\n  ── 数值 ──\n");
    printf("    GPU 输出 RMS   : %.6e\n", ref_rms);
    printf("    最大绝对差     : %.6e\n", max_abs);
    printf("    最大相对差     : %.6e\n", max_rel);
    printf("    相对 RMS 误差  : %.6e\n", diff_rms / (ref_rms > 1e-12 ? ref_rms : 1));
    printf("    %s\n", (diff_rms / (ref_rms > 1e-12 ? ref_rms : 1) < 1e-3)
           ? "✅ 解码一致" : "❌ 解码不一致（布局或查表错）");

    {
        double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
        const size_t n = (size_t)n_tok * dim;
        for (size_t i = 0; i < n; i++) {
            double a = ref[i], b = cpu_out[i];
            sa += a; sb += b; sab += a * b; saa += a * a; sbb += b * b;
        }
        const double num = n * sab - sa * sb;
        const double den = sqrt((n * saa - sa * sa) * (n * sbb - sb * sb));
        printf("    相关系数       : %.6f  (1=同向，0=无关)\n", den > 1e-30 ? num / den : 0.0);
        printf("    CPU/GPU 幅度比 : %.4f\n", sqrt(sbb / (saa > 1e-30 ? saa : 1)));
    }

    printf("\n  ── 耗时 ──\n");
    printf("    GPU : %7.3f ms  (%.2f ms/token)\n", gbest * 1000, gbest * 1000 / n_tok);
    printf("    CPU : %7.3f ms  (%.2f ms/token, %u 线程)\n", cpu_s * 1000,
           cpu_s * 1000 / n_tok, nthr);
    printf("    CPU/GPU 速度比 : %.2fx\n", cpu_s / gbest);

    free(ref); free(cpu_out);
    return 0;
}
