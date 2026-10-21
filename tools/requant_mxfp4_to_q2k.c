/* GGUF 选择性重量化：只把指定的 expert 张量从 MXFP4 转成 Q2_K
 *
 * 用途：1017 drafter 的 routed experts 是 MXFP4，而 myds_dspark_moe 只吃 Q2_K。
 *       把 3 层 × 3 个张量（gate/up/down）重编码成 Q2_K，其余张量逐字节原样复制。
 *
 * 为什么只动 expert：
 *   其余张量（fc / attn / norm / markov / conf_proj）格式已是 myds 支持的，
 *   重编码只会引入误差。
 *
 * Q2_K 布局（与 metal/moe.metal 的 block_q2_K 一致）：
 *   uchar scales[16]; uchar qs[64]; half d; half dmin;   = 84 B / 256 元素
 *   每 16 个元素一组，scales[i] 低 4 位是 d 的乘数、高 4 位是 dmin 的乘数，
 *   量化值 2 bit 打包在 qs 里（每 8 个元素一组、4 组交错）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <arm_neon.h>

#define QK_K 256
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } block_q2_K;
typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;

static const float MXFP4_LUT[16] = {
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
};
static inline float e8m0_to_f32(uint8_t e) {
    uint32_t bits = (e == 0) ? 0x00400000u : ((uint32_t)e << 23);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    float out;
    __asm__("fcvt %s0, %h1" : "=w"(out) : "w"(h));
    return out;
}
static inline uint16_t f32_to_f16(float f) {
    uint16_t out;
    __asm__("fcvt %h0, %s1" : "=w"(out) : "w"(f));
    return out;
}

/* MXFP4 → float（32 个元素 / 块） */
static void mxfp4_dequant(const uint8_t *src, float *dst, size_t n_blocks) {
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = src + b * 17;
        const float d = e8m0_to_f32(blk[0]);
        for (int i = 0; i < 16; i++) {
            const uint8_t byte = blk[1 + i];
            dst[b * 32 + 2 * i]     = MXFP4_LUT[byte & 0x0F] * d;
            dst[b * 32 + 2 * i + 1] = MXFP4_LUT[(byte >> 4) & 0x0F] * d;
        }
    }
}

/* float → Q2_K（256 元素 / 块），按 ggml 的 quantize_row_q2_K 做法：
 * 每 16 元素一个 scale（4bit 量程 + 4bit min），2bit 量化值 */
static void quantize_row_q2_K(const float *x, block_q2_K *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t b = 0; b < nb; b++) {
        const float *xb = x + b * QK_K;
        block_q2_K *yb = y + b;
        float max_scale = 0.0f, max_min = 0.0f;
        float scales[16], mins[16];
        /* 逐 16 元素求 min/scale */
        for (int g = 0; g < 16; g++) {
            const float *xg = xb + g * 16;
            float mn = xg[0], mx = xg[0];
            for (int i = 1; i < 16; i++) {
                if (xg[i] < mn) mn = xg[i];
                if (xg[i] > mx) mx = xg[i];
            }
            /* Q2_K 的 min 取负值部分，scale = (max - min)/3 */
            float sc = (mx - mn) / 3.0f;
            float mi = mn;
            if (sc > max_scale) max_scale = sc;
            if (-mi > max_min) max_min = -mi;
            scales[g] = sc;
            mins[g] = mi;
        }
        if (max_scale == 0.0f) max_scale = 1e-8f;
        if (max_min == 0.0f) max_min = 1e-8f;
        const float d = max_scale / 15.0f;    /* 4bit 量程 */
        const float dm = max_min / 15.0f;
        yb->d = f32_to_f16(d);
        yb->dmin = f32_to_f16(dm);
        /* 打包 scales：低 4 位 scale/d，高 4 位 min/dm */
        for (int g = 0; g < 16; g++) {
            int s = (int)lrintf(scales[g] / d);
            int m = (int)lrintf(-mins[g] / dm);
            if (s > 15) s = 15; if (s < 0) s = 0;
            if (m > 15) m = 15; if (m < 0) m = 0;
            yb->scales[g] = (uint8_t)((m << 4) | s);
        }
        /* 2bit 量化：每 32 元素一段，4 段交错进 qs */
        memset(yb->qs, 0, 64);
        for (int g = 0; g < 16; g++) {
            const float *xg = xb + g * 16;
            const float dl = d * (float)(yb->scales[g] & 0x0F);
            const float ml = dm * (float)(yb->scales[g] >> 4);
            /* 4 个 32-bit 字覆盖 16 个 2bit 值 */
            uint32_t pack[4] = {0, 0, 0, 0};
            for (int i = 0; i < 16; i++) {
                float v = xg[i];
                int q;
                if (dl > 1e-30f) q = (int)lrintf((v + ml) / dl);
                else q = 0;
                if (q < 0) q = 0; if (q > 3) q = 3;
                const int word = i / 4;
                const int shift = (i % 4) * 2;
                pack[word] |= ((uint32_t)q) << shift;
            }
            /* q2_K 的 qs 布局：g/8 段，g%8 组；每 16 元素占 4 字节 */
            const int seg = g / 8;         /* 0 或 1 */
            const int lane = g % 8;        /* 0..7 */
            for (int w = 0; w < 4; w++)
                yb->qs[seg * 32 + w * 8 + lane] = (uint8_t)(pack[w] & 0xFF);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "用法: %s in.gguf out.gguf\n"
            "  只重编码名为 ffn_{gate,up,down}_exps.weight 且类型为 MXFP4 的张量 → Q2_K\n", argv[0]);
        return 2;
    }
    fprintf(stderr, "本工具尚未接 GGUF 重写层；先用 --dry-run 验证量化器\n");
    return 0;
}
