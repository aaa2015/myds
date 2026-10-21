/* MoE kernel 级 A/B：IQ2_XXS vs Q4_K
 *
 * 目的：回答「Q2 升 Q4，性能会不会提升」——不是模拟，是真实 kernel。
 *
 * 同形状、同专家数、同 offload 结构，只换量化格式：
 *   A: gate/up = IQ2_XXS, down = Q2_K   （生产现状）
 *   B: gate/up = Q4_K,   down = Q4_K   （全部升 Q4）
 *
 * 判据（有效带宽 = 权重字节 / 耗时）：
 *   - B 带宽 >> A 带宽 ⇒ A 被 ALU（IQ2 查表）拖住 ⇒ 升 Q4 更快
 *   - 两者带宽接近     ⇒ 都带宽受限 ⇒ 升 Q4 按字节比例变慢
 */
#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1e6 + (double)ts.tv_nsec*1e-3;
}

typedef struct { size_t block; size_t bytes; const char *name; } fmt_t;
static fmt_t fmt_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ2_XXS: return (fmt_t){256, 66,  "IQ2_XXS"};
        case GGML_TYPE_Q2_K:    return (fmt_t){256, 84,  "Q2_K"};
        case GGML_TYPE_Q4_K:    return (fmt_t){256, 144, "Q4_K"};
        case GGML_TYPE_Q8_0:    return (fmt_t){32,  34,  "Q8_0"};
        default:                return (fmt_t){256, 144, "?"};
    }
}

/* 填可解码的合成权重：d/dt 用合理 fp16，量化位随机 */
static void fill_weight(void *ptr, size_t nbytes, size_t bytes_per_block, ggml_type t) {
    uint8_t *p = (uint8_t*)ptr;
    for (size_t i = 0; i < nbytes; i++) p[i] = (uint8_t)((i*37u + (i>>7)*101u) & 0xFF);
    size_t nb = nbytes / bytes_per_block;
    for (size_t b = 0; b < nb; b++) {
        uint8_t *blk = p + b*bytes_per_block;
        uint16_t d = 0x2c00;                 /* ~0.0625，避免 inf/nan */
        memcpy(blk, &d, 2);
        if (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q4_K) {
            uint16_t dm = 0x2800;            /* ~0.03125 */
            memcpy(blk+2, &dm, 2);
        }
    }
}

int main(int argc, char **argv) {
    uint32_t dim      = argc > 1 ? (uint32_t)atoi(argv[1]) : 5120;
    uint32_t exp_dim  = argc > 2 ? (uint32_t)atoi(argv[2]) : 2304;
    uint32_t n_exp    = argc > 3 ? (uint32_t)atoi(argv[3]) : 384;
    uint32_t n_act    = argc > 4 ? (uint32_t)atoi(argv[4]) : 6;
    int reps          = argc > 5 ? atoi(argv[5]) : 40;
    int n_tok         = argc > 6 ? atoi(argv[6]) : 1;

    if (!myds_metal_init(NULL)) { printf("  metal init 失败\n"); return 1; }

    printf("  MoE kernel A/B: dim=%u exp_dim=%u experts=%u act=%u tok=%d\n",
           dim, exp_dim, n_exp, n_act, n_tok);

    struct { ggml_type gate, up, down; const char *label; } cfg[2] = {
        { GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, "A 生产(IQ2+Q2_K)" },
        { GGML_TYPE_Q4_K,    GGML_TYPE_Q4_K,    GGML_TYPE_Q4_K, "B 全升Q4_K     " },
    };

    printf("\n  %-18s %10s %10s %12s %10s\n", "配置", "权重(GB)", "耗时(ms)", "有效GB/s", "相对A");
    double t_a = 0, b_a = 0;

    for (int c = 0; c < 2; c++) {
        fmt_t fg = fmt_of(cfg[c].gate), fu = fmt_of(cfg[c].up), fd = fmt_of(cfg[c].down);
        size_t g_bytes = (size_t)n_exp * exp_dim * (dim / fg.block) * fg.bytes;
        size_t u_bytes = g_bytes;
        size_t d_bytes = (size_t)n_exp * dim * (exp_dim / fd.block) * fd.bytes;
        size_t tot = g_bytes + u_bytes + d_bytes;

        myds_metal_buffer_t *gate = myds_metal_buffer_alloc(g_bytes);
        myds_metal_buffer_t *up   = myds_metal_buffer_alloc(u_bytes);
        myds_metal_buffer_t *down = myds_metal_buffer_alloc(d_bytes);
        myds_metal_buffer_t *x    = myds_metal_buffer_alloc((size_t)dim*n_tok*sizeof(float));
        myds_metal_buffer_t *out  = myds_metal_buffer_alloc((size_t)dim*n_tok*sizeof(float));
        myds_metal_buffer_t *ids  = myds_metal_buffer_alloc((size_t)n_act*n_tok*sizeof(int32_t));
        myds_metal_buffer_t *wts  = myds_metal_buffer_alloc((size_t)n_act*n_tok*sizeof(float));
        myds_metal_buffer_t *sg   = myds_metal_buffer_alloc((size_t)n_act*n_tok*exp_dim*sizeof(float));
        myds_metal_buffer_t *su   = myds_metal_buffer_alloc((size_t)n_act*n_tok*exp_dim*sizeof(float));
        myds_metal_buffer_t *sm   = myds_metal_buffer_alloc((size_t)n_act*n_tok*exp_dim*sizeof(float));
        myds_metal_buffer_t *sd   = myds_metal_buffer_alloc((size_t)n_act*n_tok*dim*sizeof(float));
        if (!gate||!up||!down||!x||!out||!ids||!wts||!sg||!su||!sm||!sd) {
            printf("  %s alloc 失败\n", cfg[c].label); continue;
        }

        fill_weight(myds_metal_buffer_data(gate), g_bytes, fg.bytes, cfg[c].gate);
        fill_weight(myds_metal_buffer_data(up),   u_bytes, fu.bytes, cfg[c].up);
        fill_weight(myds_metal_buffer_data(down), d_bytes, fd.bytes, cfg[c].down);

        float *xp = (float*)myds_metal_buffer_data(x);
        for (size_t i = 0; i < (size_t)dim*n_tok; i++) xp[i] = (float)(((int)(i*43)%201)-100)/500.0f;
        int32_t *ip = (int32_t*)myds_metal_buffer_data(ids);
        float *wp = (float*)myds_metal_buffer_data(wts);
        for (int t = 0; t < n_tok; t++)
            for (uint32_t k = 0; k < n_act; k++) {
                ip[t*n_act+k] = (int32_t)((t*7 + k*53) % n_exp);
                wp[t*n_act+k] = 1.0f/n_act;
            }

        int ok = myds_metal_routed_moe_batch(out, x, gate, up, down, ids, wts,
                                            sg, su, sm, sd,
                                            dim, exp_dim, n_exp, n_act, (uint32_t)n_tok,
                                            cfg[c].gate, cfg[c].up, cfg[c].down, 0.0f);
        myds_metal_synchronize();
        if (!ok) { printf("  %s: routed_moe_batch 返回 0 — 该格式组合不被支持\n", cfg[c].label); continue; }

        for (int i = 0; i < 5; i++)
            myds_metal_routed_moe_batch(out, x, gate, up, down, ids, wts,
                                        sg, su, sm, sd, dim, exp_dim, n_exp, n_act,
                                        (uint32_t)n_tok, cfg[c].gate, cfg[c].up, cfg[c].down, 0.0f);
        myds_metal_synchronize();

        double best = 1e18;
        for (int r = 0; r < 3; r++) {
            double t0 = now_us();
            for (int i = 0; i < reps; i++)
                myds_metal_routed_moe_batch(out, x, gate, up, down, ids, wts,
                                            sg, su, sm, sd, dim, exp_dim, n_exp, n_act,
                                            (uint32_t)n_tok, cfg[c].gate, cfg[c].up, cfg[c].down, 0.0f);
            myds_metal_synchronize();
            double dt = (now_us()-t0)/reps;
            if (dt < best) best = dt;
        }
        /* 每 token 只读选中专家；这里记整池字节会高估，故按 n_act/n_exp 折算 */
        double touched = (double)tot * n_act / n_exp * n_tok;
        double gbs = touched / (best*1e-6) / 1e9;
        if (c == 0) { t_a = best; b_a = touched; }
        printf("  %-18s %10.2f %10.2f %12.1f %9.2fx\n",
               cfg[c].label, tot/1e9, best/1000.0, gbs, t_a > 0 ? best/t_a : 1.0);

        myds_metal_buffer_free(gate); myds_metal_buffer_free(up); myds_metal_buffer_free(down);
        myds_metal_buffer_free(x); myds_metal_buffer_free(out);
        myds_metal_buffer_free(ids); myds_metal_buffer_free(wts);
        myds_metal_buffer_free(sg); myds_metal_buffer_free(su);
        myds_metal_buffer_free(sm); myds_metal_buffer_free(sd);
    }
    printf("\n  注：耗时含 MoE 全流程（route 结果给定，仅 gate/up/swiglu/down/sum）\n");
    return 0;
}
