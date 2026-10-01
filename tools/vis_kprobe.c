/*
 * vis_kprobe.c — 视觉内核的逐个独立探针
 *
 * 为什么先写这个：整条前向一次跑不通时，失败可能是「某个内核写错」，
 * 也可能是「批处理/派发接错」。把每个内核单独喂小数据跑一遍，
 * 就能把两类问题分开 —— 比在前向里逐段加日志快得多。
 *
 * 用法: vis_kprobe [--metal-dir DIR]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_metal.h"
#include <stdint.h>

static int fails = 0;

static void report(const char *name, int ok, int detailed_ok, const char *detail) {
    printf("  %s %-28s %s\n", (ok && detailed_ok) ? "✓" : "✗", name, detail);
    if (!ok || !detailed_ok) fails++;
}

/* 行式派发（与 myds_visfwd.c 里同一套形状） */
static int rows_dispatch(const char *k, uint32_t width, uint32_t rows,
                         myds_metal_buffer_t *const *b, size_t n) {
    struct { uint32_t width, rows; } args = { width, rows };
    const uint32_t tx = width < 256u ? width : 256u;
    const uint32_t groups[3] = { (width + tx - 1u) / tx, rows, 1u };
    const uint32_t threads[3] = { tx, 1u, 1u };
    return myds_metal_dispatch_kernel(k, &args, sizeof(args),
                                      (const myds_metal_buffer_t *const *)b, n,
                                      groups, threads, 0);
}

/* 对照：完全走引擎自带的算子 + 同一个批处理模式。
 * 它若也崩 ⇒ 问题在「批处理/链接」而非我的派发。 */
static int test_control(void) {
    const uint32_t n = 64;
    myds_metal_buffer_t *a = myds_metal_buffer_alloc(n * 4);
    myds_metal_buffer_t *b = myds_metal_buffer_alloc(n * 4);
    myds_metal_buffer_t *o = myds_metal_buffer_alloc(n * 4);
    if (!a || !b || !o) return 0;
    float v[64];
    for (uint32_t i = 0; i < n; i++) v[i] = 1.0f;
    myds_metal_buffer_write(a, 0, v, n * 4);
    myds_metal_buffer_write(b, 0, v, n * 4);
    fprintf(stderr, "    [step] control: begin...\n");
    int ok = myds_metal_begin_batch();
    fprintf(stderr, "    [step] control: begin=%d\n", ok);
    if (ok) ok = myds_metal_add(o, a, b, n);
    fprintf(stderr, "    [step] control: add=%d\n", ok);
    if (ok) ok = myds_metal_commit_batch();
    fprintf(stderr, "    [step] control: commit=%d\n", ok);
    float out[64] = {0};
    int ro = ok && myds_metal_buffer_read(o, 0, out, n * 4);
    int good = ro && fabsf(out[0] - 2.0f) < 1e-6f;
    report("对照 myds_metal_add（引擎自带）", ok && ro, good, good ? "1+1=2 正确" : "值不对");
    myds_metal_buffer_free(a); myds_metal_buffer_free(b); myds_metal_buffer_free(o);
    return good;
}

static int test_round(void) {
    const uint32_t W = 8, H = 4;
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(W * H * 4);
    if (!x) return 0;
    float v[32];
    for (int i = 0; i < 32; i++) v[i] = 0.1f * (i + 1);
    myds_metal_buffer_write(x, 0, v, sizeof(v));
    fprintf(stderr, "    [step] alloc+write 完成\n");
    {
        const char *names[] = { "kernel_visenc_round", "kernel_visenc_bias",
            "kernel_visenc_bias_residual41", "kernel_visenc_add_residual",
            "kernel_visenc_qkv_rope", "kernel_visenc_attention",
            "kernel_visenc_swiglu_split41", "kernel_visenc_aligner_reorder",
            "kernel_visenc_gelu_bias", "kernel_mul_mm_f32_bco", "kernel_mul_mm_f32" };
        for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++)
            fprintf(stderr, "    [probe] %-32s maxThreads=%u\n", names[i],
                    myds_metal_kernel_max_threads(names[i]));
    }
    int ok = myds_metal_begin_batch();
    fprintf(stderr, "    [step] begin_batch=%d\n", ok);
    if (ok) ok = rows_dispatch("kernel_visenc_round", W, H, &x, 1);
    fprintf(stderr, "    [step] dispatch=%d\n", ok);
    if (ok) ok = myds_metal_commit_batch();
    fprintf(stderr, "    [step] commit=%d\n", ok);
    float out[32] = {0};
    int ro = ok && myds_metal_buffer_read(x, 0, out, sizeof(out));
    /* BF16 舍入后：低 16 位必须为 0 */
    int exact = 1;
    for (int i = 0; i < 32; i++) {
        uint32_t bits; memcpy(&bits, &out[i], 4);
        if (bits & 0xffffu) exact = 0;
    }
    report("kernel_visenc_round", ok && ro, exact, exact ? "低16位全0" : "存在非零低位");
    myds_metal_buffer_free(x);
    return ok && ro && exact;
}

static int test_bias_and_residual(void) {
    const uint32_t W = 8, H = 2;
    myds_metal_buffer_t *x = myds_metal_buffer_alloc(W * H * 4);
    myds_metal_buffer_t *b = myds_metal_buffer_alloc(W * 4);
    myds_metal_buffer_t *r = myds_metal_buffer_alloc(W * H * 4);
    if (!x || !b || !r) return 0;
    float xv[16], bv[8], rv[16];
    for (int i = 0; i < 16; i++) { xv[i] = (float)i; rv[i] = 1000.0f; }
    for (int i = 0; i < 8; i++) bv[i] = 0.5f;
    myds_metal_buffer_write(x, 0, xv, sizeof(xv));
    myds_metal_buffer_write(b, 0, bv, sizeof(bv));
    myds_metal_buffer_write(r, 0, rv, sizeof(rv));
    int ok = myds_metal_begin_batch();
    if (ok) { myds_metal_buffer_t *bs[2] = { x, b }; ok = rows_dispatch("kernel_visenc_bias", W, H, bs, 2); }
    if (ok) ok = myds_metal_commit_batch();
    float out[16] = {0};
    int ro = ok && myds_metal_buffer_read(x, 0, out, sizeof(out));
    int good = ro && fabsf(out[0] - 0.5f) < 1e-6f && fabsf(out[9] - 9.5f) < 1e-6f;
    report("kernel_visenc_bias", ok && ro, good, good ? "+0.5 正确" : "值不对");

    int ok2 = myds_metal_begin_batch();
    if (ok2) {
        myds_metal_buffer_t *bs[3] = { x, b, r };
        struct { uint32_t width, rows; } args = { W, H };
        const uint32_t tx = W, groups[3] = { 1, H, 1 }, threads[3] = { tx, 1, 1 };
        ok2 = myds_metal_dispatch_kernel("kernel_visenc_bias_residual41", &args, sizeof(args),
                                         (const myds_metal_buffer_t *const *)bs, 3,
                                         groups, threads, 0);
    }
    if (ok2) ok2 = myds_metal_commit_batch();
    float out2[16] = {0};
    int ro2 = ok2 && myds_metal_buffer_read(x, 0, out2, sizeof(out2));
    /* x = round(round(x+b) + 1000) ⇒ 至少 1000 量级 */
    int good2 = ro2 && fabsf(out2[0] - 1000.5f) < 1.0f;
    report("kernel_visenc_bias_residual41", ok2 && ro2, good2, good2 ? "~1000.5" : "值不对");
    myds_metal_buffer_free(x); myds_metal_buffer_free(b); myds_metal_buffer_free(r);
    return good && good2;
}

static int test_attention(void) {
    /* 行长 = 64（1 个头）时不好喂；这里用最小可跑形状：rows=4, 16 头 */
    const uint32_t rows = 4;
    const size_t sz = (size_t)rows * 1024 * 4;
    myds_metal_buffer_t *q = myds_metal_buffer_alloc(sz);
    myds_metal_buffer_t *k = myds_metal_buffer_alloc(sz);
    myds_metal_buffer_t *v = myds_metal_buffer_alloc(sz);
    myds_metal_buffer_t *o = myds_metal_buffer_alloc(sz);
    if (!q || !k || !v || !o) return 0;
    float *buf = malloc(sz);
    if (!buf) return 0;
    for (size_t i = 0; i < sz / 4; i++) buf[i] = 0.01f;
    myds_metal_buffer_write(q, 0, buf, sz);
    myds_metal_buffer_write(k, 0, buf, sz);
    for (size_t i = 0; i < sz / 4; i++) buf[i] = 1.0f;
    myds_metal_buffer_write(v, 0, buf, sz);
    free(buf);

    int ok = myds_metal_begin_batch();
    if (ok) {
        struct { uint32_t rows; float scale; } args = { rows, 0.125f };
        myds_metal_buffer_t *bs[4] = { q, k, v, o };
        const uint32_t groups[3] = { rows, 16u, 1u }, threads[3] = { 32u, 1u, 1u };
        ok = myds_metal_dispatch_kernel("kernel_visenc_attention", &args, sizeof(args),
                                        (const myds_metal_buffer_t *const *)bs, 4,
                                        groups, threads, 0);
    }
    if (ok) ok = myds_metal_commit_batch();
    float out[8] = {0};
    int ro = ok && myds_metal_buffer_read(o, 0, out, sizeof(out));
    /* v 全是 1 ⇒ 注意力输出应当恒为 1（加权平均） */
    int good = ro && fabsf(out[0] - 1.0f) < 1e-4f && fabsf(out[5] - 1.0f) < 1e-4f;
    report("kernel_visenc_attention", ok && ro, good, good ? "全 1 正确（v 恒 1）" : "值不对");
    myds_metal_buffer_free(q); myds_metal_buffer_free(k);
    myds_metal_buffer_free(v); myds_metal_buffer_free(o);
    return good;
}

static int test_swiglu_reorder_gelu(void) {
    /* swiglu_split41: gate_up [2, 4] → out [2, 4]... 需要 width 与 rows 匹配真实语义 */
    const uint32_t W = 4, H = 2;
    myds_metal_buffer_t *gu = myds_metal_buffer_alloc(W * H * 2 * 4);
    myds_metal_buffer_t *o = myds_metal_buffer_alloc(W * H * 4);
    if (!gu || !o) return 0;
    float gv[16];
    for (int i = 0; i < 16; i++) gv[i] = 1.0f;      /* gate=1, up=1 */
    myds_metal_buffer_write(gu, 0, gv, sizeof(gv));
    int ok = myds_metal_begin_batch();
    if (ok) {
        struct { uint32_t width, rows; } args = { W, H };
        myds_metal_buffer_t *bs[2] = { gu, o };
        const uint32_t tx = W, groups[3] = { 1, H, 1 }, threads[3] = { tx, 1, 1 };
        ok = myds_metal_dispatch_kernel("kernel_visenc_swiglu_split41", &args, sizeof(args),
                                        (const myds_metal_buffer_t *const *)bs, 2,
                                        groups, threads, 0);
    }
    if (ok) ok = myds_metal_commit_batch();
    float out[8] = {0};
    int ro = ok && myds_metal_buffer_read(o, 0, out, sizeof(out));
    /* ★ V4.1 会【舍两次】：先舍 silu，再舍 silu*up。
     *   silu(1)=0.731058579 → 舍 BF16 → 0.73046875 → ×1 → 再舍 → 0.73046875。
     *   我第一版拿未舍入的 0.7310586 当期望值 ⇒ 报「值不对」。
     *   再次印证：拿到「不对」的结果，先检查尺子。 */
    int good = ro && fabsf(out[0] - 0.73046875f) < 1e-7f;
    report("kernel_visenc_swiglu_split41", ok && ro, good,
           good ? "silu(1) 舍两次 = 0.73046875" : "值不对");
    myds_metal_buffer_free(gu); myds_metal_buffer_free(o);
    return good;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *dir = "metal";
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], "--metal-dir")) dir = argv[i + 1];

    printf("====================================================\n");
    printf("   视觉内核探针（逐内核独立验证）\n");
    printf("====================================================\n");
    if (!myds_metal_init(dir)) { printf("  ✗ metal 初始化失败 (%s)\n", dir); return 1; }

    test_control();
    test_round();
    test_bias_and_residual();
    test_attention();
    test_swiglu_reorder_gelu();

    printf("----------------------------------------------------\n");
    printf("  %s（失败 %d 项）\n", fails ? "有失败" : "全部通过", fails);
    myds_metal_cleanup();
    return fails ? 1 : 0;
}
