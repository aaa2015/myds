/* MXFP4 → Q2_K 重量化：GGUF 重写
 *
 * 只改 9 个张量（3 层 × gate/up/down expert），其余逐字节原样复制。
 * 输出一个新的 GGUF v3 文件。
 *
 * 用法: requant in.gguf out.gguf
 */
#include "myds_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <arm_neon.h>

#define QK_K 256
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } block_q2_K;
typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;
_Static_assert(sizeof(block_q2_K) == 84, "Q2_K = 84 B");
_Static_assert(sizeof(block_mxfp4) == 17, "MXFP4 = 17 B");

static const float LUT[16] = {
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
};
static inline float e8m0(uint8_t e) {
    uint32_t bits = (e == 0) ? 0x00400000u : ((uint32_t)e << 23);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline uint16_t f2h(float f) {
    float16_t h = (float16_t)f; uint16_t r; memcpy(&r, &h, 2); return r;
}
static inline float h2f(uint16_t h) {
    _Float16 f; memcpy(&f, &h, 2); return (float)f;
}

static void mxfp4_to_f32(const uint8_t *src, uint32_t n, float *out) {
    const uint32_t nb = n / 32u;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = src + (size_t)b * 17u;
        const float d = e8m0(blk[0]);
        for (int i = 0; i < 16; i++) {
            const uint8_t byte = blk[1 + i];
            out[b * 32u + 2u * i]     = LUT[byte & 0x0F] * d;
            out[b * 32u + 2u * i + 1] = LUT[(byte >> 4) & 0x0F] * d;
        }
    }
}

/* 每块：16 组 × 16 元素。值 = d*(sc&0xF)*coef*(q&mask 右移) - dmin*(sc>>4) */
static void f32_to_q2k(const float *x, block_q2_K *y, uint32_t n) {
    const uint32_t nb = n / QK_K;
    for (uint32_t b = 0; b < nb; b++) {
        const float *xb = x + (size_t)b * QK_K;
        block_q2_K *yb = y + b;
        float gmin[16], gmax[16];
        for (int g = 0; g < 16; g++) {
            const float *xg = xb + g * 16;
            float mn = xg[0], mx = xg[0];
            for (int i = 1; i < 16; i++) {
                if (xg[i] < mn) mn = xg[i];
                if (xg[i] > mx) mx = xg[i];
            }
            gmin[g] = mn; gmax[g] = mx;
        }
        float max_sc = 0.0f, max_mn = 0.0f;
        for (int g = 0; g < 16; g++) {
            const float sc = (gmax[g] - gmin[g]) / 3.0f;
            if (sc > max_sc) max_sc = sc;
            if (-gmin[g] > max_mn) max_mn = -gmin[g];
        }
        if (max_sc <= 0.0f) max_sc = 1e-8f;
        if (max_mn <= 0.0f) max_mn = 1e-8f;
        /* 朴素做法（d = max_sc/15）误差偏大。这里对 d 与 dmin 各试几档缩放，
         * 取块内平方误差最小的组合 —— 等价于 ggml 的 scale 搜索，但只做
         * 粗网格，够把真实权重的相对误差从 ~33% 压到 ~20%。 */
        float best_d = max_sc / 15.0f, best_dm = max_mn / 15.0f;
        double best_err = 1e300;
        static const float fsc[5] = {1.00f, 0.85f, 0.70f, 0.55f, 0.40f};
        for (int ai = 0; ai < 5; ai++) {
            for (int bi = 0; bi < 5; bi++) {
                const float dd = max_sc / 15.0f * fsc[ai];
                const float ddm = max_mn / 15.0f * fsc[bi];
                if (dd <= 0.0f || ddm <= 0.0f) continue;
                double err = 0.0;
                for (int g = 0; g < 16; g++) {
                    const float sc = (gmax[g] - gmin[g]) / 3.0f;
                    int si = (int)lrintf(sc / dd);
                    int mi = (int)lrintf(-gmin[g] / ddm);
                    if (si < 0) si = 0; if (si > 15) si = 15;
                    if (mi < 0) mi = 0; if (mi > 15) mi = 15;
                    const float dl = dd * (float)si, ml = ddm * (float)mi;
                    const float *xg = xb + g * 16;
                    for (int i = 0; i < 16; i++) {
                        int q = (dl > 1e-30f) ? (int)lrintf((xg[i] + ml) / dl) : 0;
                        if (q < 0) q = 0; if (q > 3) q = 3;
                        const double v = (double)(dl * (float)q - ml);
                        err += (v - (double)xg[i]) * (v - (double)xg[i]);
                    }
                }
                if (err < best_err) { best_err = err; best_d = dd; best_dm = ddm; }
            }
        }
        const float d = best_d, dm = best_dm;
        yb->d = f2h(d); yb->dmin = f2h(dm);
        memset(yb->qs, 0, 64);
        for (int g = 0; g < 16; g++) {
            const float sc = (gmax[g] - gmin[g]) / 3.0f;
            int s = (int)lrintf(sc / d), m = (int)lrintf(-gmin[g] / dm);
            if (s < 0) s = 0; if (s > 15) s = 15;
            if (m < 0) m = 0; if (m > 15) m = 15;
            yb->scales[g] = (uint8_t)((m << 4) | s);
            /* 注意：metal 的 coef（1, 1/4, 1/16, 1/64）与 mask（3, 12, 48, 192）
             * 精确抵消 —— coef*(q&mask) = qv（0..3），与 il2 无关。
             * 所以量化值就是裸 2bit 量程，按 il2 放到位即可。 */
            const float dl = d * (float)s, ml = dm * (float)m;
            const float *xg = xb + g * 16;
            const int qoff = 32 * (g / 8) + 16 * (g & 1);
            const int shift = ((g / 2) % 4) * 2;
            for (int i = 0; i < 16; i++) {
                int q = (dl > 1e-30f) ? (int)lrintf((xg[i] + ml) / dl) : 0;
                if (q < 0) q = 0; if (q > 3) q = 3;
                yb->qs[qoff + i] |= (uint8_t)(q << shift);
            }
        }
    }
}

static int is_expert_tensor(const char *n) {
    return strstr(n, "ffn_gate_exps.weight") || strstr(n, "ffn_up_exps.weight") ||
           strstr(n, "ffn_down_exps.weight");
}

/* 自检：f32 → Q2_K → f32 的往返误差（不碰 GGUF） */
static int selftest(void) {
    const uint32_t n = QK_K * 4;
    float *x = malloc(n * 4), *y = malloc(n * 4);
    block_q2_K *q = malloc((n / QK_K) * sizeof(block_q2_K));
    uint32_t rng = 999u;
    /* 高斯分布（Box-Muller），更接近真实权重 */
    for (uint32_t i = 0; i + 1 < n; i += 2) {
        double u1, u2;
        rng = rng * 1664525u + 1013904223u; u1 = ((rng >> 8) & 0xFFFF) / 65536.0;
        rng = rng * 1664525u + 1013904223u; u2 = ((rng >> 8) & 0xFFFF) / 65536.0;
        if (u1 < 1e-9) u1 = 1e-9;
        const double r = sqrt(-2.0 * log(u1)), th = 2.0 * M_PI * u2;
        x[i] = (float)(r * cos(th) * 0.05);
        x[i + 1] = (float)(r * sin(th) * 0.05);
    }
    f32_to_q2k(x, q, n);
    /* 用 metal 的语义反解 */
    for (uint32_t b = 0; b < n / QK_K; b++) {
        const block_q2_K *yb = &q[b];
        const float d = h2f(yb->d), dm = h2f(yb->dmin);
        for (int g = 0; g < 16; g++) {
            const int qoff = 32 * (g / 8) + 16 * (g & 1);
            const int shift = ((g / 2) % 4) * 2;
            const uint8_t sc = yb->scales[g];
            const int g_il2 = (g / 2) % 4;
            const float dl = d * (float)(sc & 0x0F), ml = dm * (float)(sc >> 4);
            for (int i = 0; i < 16; i++) {
                int qv = (yb->qs[qoff + i] >> shift) & 3;
                y[b * QK_K + g * 16 + i] = dl * (float)qv - ml;
            }
        }
    }
    double se = 0, sr = 0, mx = 0;
    for (uint32_t i = 0; i < n; i++) {
        double d = (double)y[i] - x[i];
        se += d * d; sr += (double)x[i] * x[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    const double rms_x = sqrt(sr / n), rms_e = sqrt(se / n);
    printf("  Q2_K roundtrip: 原 RMS %.5f  误差 RMS %.5f  相对 %.2f%%  最大差 %.5f\n",
           rms_x, rms_e, 100 * rms_e / rms_x, mx);
    printf("  %s\n", rms_e / rms_x < 0.35 ? "✅ 误差在 Q2_K 合理范围（<35%，2bit 量化）" : "❌ 量化器有 bug");
    free(x); free(y); free(q);
    return rms_e / rms_x < 0.35 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc < 3) { fprintf(stderr, "用法: %s in.gguf out.gguf  |  %s --selftest\n", argv[0], argv[0]); return 2; }
    gguf_file_t *gf = gguf_open(argv[1]);
    if (!gf) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }

    FILE *in = fopen(argv[1], "rb");
    if (!in) return 1;
    /* 读原始头，用于逐字节复制 KV 段 */
    fseek(in, 0, SEEK_SET);
    uint8_t magic[4]; fread(magic, 1, 4, in);
    uint32_t ver; fread(&ver, 4, 1, in);
    uint64_t nten, nkv;
    fread(&nten, 8, 1, in); fread(&nkv, 8, 1, in);
    long kv_start = ftell(in);
    long data_start = (long)gf->tensor_data_offset;

    /* 统计：哪些张量要转换 */
    int n_conv = 0;
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const gguf_tensor_t *t = &gf->tensors[i];
        if (is_expert_tensor(t->name) && t->type == GGML_TYPE_MXFP4) n_conv++;
    }
    fprintf(stderr, "  张量 %llu 个，其中待转换 %d 个\n",
            (unsigned long long)gf->tensor_count, n_conv);
    if (!n_conv) { fprintf(stderr, "  没有需要转换的张量，退出\n"); return 1; }

    /* 新偏移：转换后的张量变大（84 vs 原始 MXFP4 的 136/256≈0.53B/元素 → 84/256=0.328?
     * 实际 MXFP4 每 256 元素 136 B，Q2_K 每 256 元素 84 B ⇒ 变小。重新算偏移。 */
    uint64_t align = gf->alignment ? gf->alignment : 32;
    uint64_t off = 0;
    uint64_t *new_off = calloc(gf->tensor_count, sizeof(uint64_t));
    uint32_t *new_type = calloc(gf->tensor_count, sizeof(uint32_t));
    uint64_t *new_size = calloc(gf->tensor_count, sizeof(uint64_t));
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const gguf_tensor_t *t = &gf->tensors[i];
        uint64_t elems = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) elems *= t->dims[d];
        if (is_expert_tensor(t->name) && t->type == GGML_TYPE_MXFP4) {
            new_type[i] = GGML_TYPE_Q2_K;
            new_size[i] = (elems / QK_K) * sizeof(block_q2_K);
        } else {
            new_type[i] = t->type;
            new_size[i] = t->size_bytes;
        }
        off = (off + align - 1) / align * align;
        new_off[i] = off;
        off += new_size[i];
    }
    uint64_t total_data = off;
    fprintf(stderr, "  新数据区 %.2f GB（原 %.2f GB）\n",
            total_data / 1e9, (double)(gf->mmap_size - data_start) / 1e9);

    FILE *out = fopen(argv[2], "wb");
    if (!out) { fprintf(stderr, "写不了 %s\n", argv[2]); return 1; }

    /* 1) 头：magic + ver + counts + KV 段（逐字节复制） */
    fwrite(magic, 1, 4, out);
    fwrite(&ver, 4, 1, out);
    fwrite(&nten, 8, 1, out);
    fwrite(&nkv, 8, 1, out);
    {
        long kv_bytes = 0;
        /* KV 段结束 = 第一个张量信息开始；用 data_start 反推不可靠，
         * 改为重新扫描：从 kv_start 读到 tensors 数组之前。
         * myds 的 gguf 已解析，这里简单地按「头的原样长度」复制。 */
        /* 找 tensor infos 起点：从文件头重新解析一遍 */
        fseek(in, (long)(4+4+8+8), SEEK_SET);
        /* 跳过 KV */
        for (uint64_t k = 0; k < nkv; k++) {
            uint64_t sl; fread(&sl, 8, 1, in); fseek(in, (long)sl, SEEK_CUR);
            uint32_t t; fread(&t, 4, 1, in);
            long sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            if (t == 8) { uint64_t l; fread(&l, 8, 1, in); fseek(in, (long)l, SEEK_CUR); }
            else if (t == 9) {
                uint32_t at; uint64_t n;
                fread(&at, 4, 1, in); fread(&n, 8, 1, in);
                for (uint64_t e = 0; e < n; e++) {
                    if (at == 8) { uint64_t l; fread(&l, 8, 1, in); fseek(in, (long)l, SEEK_CUR); }
                    else fseek(in, sz[at], SEEK_CUR);
                }
            } else fseek(in, sz[t], SEEK_CUR);
        }
        long kv_end = ftell(in);
        kv_bytes = kv_end - kv_start;
        fseek(in, kv_start, SEEK_SET);
        uint8_t *buf = malloc(kv_bytes);
        if (!buf || fread(buf, 1, kv_bytes, in) != (size_t)kv_bytes) {
            fprintf(stderr, "读 KV 段失败\n"); return 1;
        }
        fwrite(buf, 1, kv_bytes, out);
        free(buf);
    }

    /* 2) tensor infos：名字/维度原样，type 与 offset 更新 */
    fseek(in, (long)(4+4+8+8), SEEK_SET);
    /* 重新定位到 tensor infos 起点 */
    {
        for (uint64_t k = 0; k < nkv; k++) {
            uint64_t sl; fread(&sl, 8, 1, in); fseek(in, (long)sl, SEEK_CUR);
            uint32_t t; fread(&t, 4, 1, in);
            long sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            if (t == 8) { uint64_t l; fread(&l, 8, 1, in); fseek(in, (long)l, SEEK_CUR); }
            else if (t == 9) {
                uint32_t at; uint64_t n;
                fread(&at, 4, 1, in); fread(&n, 8, 1, in);
                for (uint64_t e = 0; e < n; e++) {
                    if (at == 8) { uint64_t l; fread(&l, 8, 1, in); fseek(in, (long)l, SEEK_CUR); }
                    else fseek(in, sz[at], SEEK_CUR);
                }
            } else fseek(in, sz[t], SEEK_CUR);
        }
    }
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const gguf_tensor_t *t = &gf->tensors[i];
        uint64_t sl = strlen(t->name);
        fwrite(&sl, 8, 1, out);
        fwrite(t->name, 1, sl, out);
        fwrite(&t->n_dims, 4, 1, out);
        for (uint32_t d = 0; d < t->n_dims; d++) fwrite(&t->dims[d], 8, 1, out);
        fwrite(&new_type[i], 4, 1, out);
        fwrite(&new_off[i], 8, 1, out);
    }

    /* 3) 对齐到 data_start */
    {
        long pos = ftell(out);
        long target = data_start;
        while (pos < target) { fputc(0, out); pos++; }
    }

    /* 4) 数据区：非转换张量逐字节复制，转换张量重编码 */
    uint32_t *order = malloc(sizeof(uint32_t) * (size_t)gf->tensor_count);
    for (uint64_t i = 0; i < gf->tensor_count; i++) order[i] = (uint32_t)i;
    /* 按新偏移排序，顺序写出 */
    for (uint64_t a = 0; a < gf->tensor_count; a++)
        for (uint64_t b = a + 1; b < gf->tensor_count; b++)
            if (new_off[order[b]] < new_off[order[a]]) {
                uint32_t tmp = order[a]; order[a] = order[b]; order[b] = tmp;
            }

    uint8_t *rd = NULL; size_t rd_cap = 0;
    float *f32b = NULL; size_t f32_cap = 0;
    long written = 0;
    for (uint64_t oi = 0; oi < gf->tensor_count; oi++) {
        uint64_t i = order[oi];
        const gguf_tensor_t *t = &gf->tensors[i];
        /* 填充到目标偏移 */
        long pos = ftell(out);
        long target = (long)(data_start + new_off[i]);
        while (pos < target) { fputc(0, out); pos++; written++; }

        if (new_type[i] == t->type) {
            /* 原样复制 */
            if (t->size_bytes > rd_cap) { rd_cap = t->size_bytes; rd = realloc(rd, rd_cap); }
            fseek(in, (long)(data_start + t->offset), SEEK_SET);
            if (fread(rd, 1, t->size_bytes, in) != t->size_bytes) {
                fprintf(stderr, "读 %s 失败\n", t->name); return 1;
            }
            fwrite(rd, 1, t->size_bytes, out);
            written += (long)t->size_bytes;
        } else {
            uint64_t elems = 1;
            for (uint32_t d = 0; d < t->n_dims; d++) elems *= t->dims[d];
            if (elems > rd_cap) { rd_cap = elems; rd = realloc(rd, rd_cap); }
            if (elems * sizeof(float) > f32_cap) { f32_cap = elems * sizeof(float); f32b = realloc(f32b, f32_cap); }
            fseek(in, (long)(data_start + t->offset), SEEK_SET);
            if (fread(rd, 1, t->size_bytes, in) != t->size_bytes) {
                fprintf(stderr, "读 %s 失败\n", t->name); return 1;
            }
            mxfp4_to_f32(rd, (uint32_t)elems, f32b);
            uint64_t nblk = elems / QK_K;
            block_q2_K *qb = malloc(nblk * sizeof(block_q2_K));
            if (!qb) { fprintf(stderr, "malloc 失败\n"); return 1; }
            f32_to_q2k(f32b, qb, (uint32_t)elems);
            /* 报告真实 roundtrip 误差（用 metal 的语义反解） */
            {
                double se = 0, sr = 0;
                for (uint64_t b = 0; b < nblk; b++) {
                    const block_q2_K *yb = &qb[b];
                    const float d = h2f(yb->d), dm = h2f(yb->dmin);
                    for (int g = 0; g < 16; g++) {
                        const int qoff = 32 * (g / 8) + 16 * (g & 1);
                        const int shift = ((g / 2) % 4) * 2;
                        const uint8_t sc = yb->scales[g];
                        const float dl = d * (float)(sc & 0x0F), ml = dm * (float)(sc >> 4);
                        for (int i = 0; i < 16; i++) {
                            const int qv = (yb->qs[qoff + i] >> shift) & 3;
                            const double v = (double)(dl * (float)qv - ml);
                            const double o = (double)f32b[b * QK_K + g * 16 + i];
                            se += (v - o) * (v - o); sr += o * o;
                        }
                    }
                }
                const double rx = sqrt(sr / (double)elems), re = sqrt(se / (double)elems);
                fprintf(stderr, "  转换 %s: %.2f → %.2f MB  原RMS %.5f 误差RMS %.5f 相对 %.1f%%\n",
                        t->name, t->size_bytes / 1e6, (double)(nblk * sizeof(block_q2_K)) / 1e6,
                        rx, re, 100.0 * re / (rx > 1e-30 ? rx : 1));
            }
            fwrite(qb, sizeof(block_q2_K), nblk, out);
            written += (long)(nblk * sizeof(block_q2_K));
            free(qb);
        }
    }
    fclose(in); fclose(out);
    fprintf(stderr, "  完成，输出 %.2f GB\n", (double)written / 1e9);
    free(rd); free(f32b); free(new_off); free(new_type); free(new_size); free(order);
    gguf_close(gf);
    return 0;
}
