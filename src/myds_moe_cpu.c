/* CPU 侧 routed MoE —— prefill 专用
 *
 * 为什么是 prefill：decode 是 GEMV（每权重只用一次），CPU 反量化成本吃掉全部收益；
 * prefill 是 GEMM，可以按 token 切分给 CPU 并行。
 *
 * 切分方式：按 token 切，不是按专家切。
 *   - 每个 token 的 MoE 输出与批内其它 token 无关
 *   - 两边写 moe_out 的不同行，无需归约
 *   - GPU kernel 用 n_tok 参数天然支持只算前 n_gpu 个 token
 *
 * 数值：CPU 的累加顺序与 GPU simd_sum 不同，故仅 CPU 计算的 token 输出与 GPU 不同；
 * GPU 计算的 token 逐位不变。
 */
#include "myds_moe_cpu.h"
#include "myds_iq2_tables.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <arm_neon.h>

#define IQ2_BLOCK   256
#define IQ2_BYTES    66
#define Q2K_BLOCK   256
#define Q2K_BYTES    84

static inline float h2f(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

/* 一行 IQ2_XXS（nb 个 256 元素块）与 x 的内积 */
static float iq2_dot_row(const uint8_t *row, uint32_t nb, const float *x) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * IQ2_BYTES;
        const float d = h2f(*(const uint16_t *)blk);
        const uint8_t *qs = blk + 2;
        const float *xb = x + (size_t)b * IQ2_BLOCK;
        for (int il = 0; il < 16; il++) {
            const int ib32 = il >> 1, ilp = il & 1;
            const uint8_t *q2 = qs + 4 * ib32;
            const uint32_t g32 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t s32 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            const uint8_t *a8 = (const uint8_t *)&g32;
            const float dl = d * (0.5f + (float)(s32 >> 28)) * 0.25f;
            const uint64_t g0 = IQ2_GRID[a8[2 * ilp + 0]];
            const uint64_t g1 = IQ2_GRID[a8[2 * ilp + 1]];
            const uint8_t s0 = IQ2_KSIGNS[(s32 >> (14 * ilp)) & 127];
            const uint8_t s1 = IQ2_KSIGNS[(s32 >> (14 * ilp + 7)) & 127];
            float v[16];
            for (int i = 0; i < 8; i++) {
                v[i]     = dl * (float)((g0 >> (8 * i)) & 0xFF) * ((s0 & IQ2_KMASK[i]) ? -1.0f : 1.0f);
                v[i + 8] = dl * (float)((g1 >> (8 * i)) & 0xFF) * ((s1 & IQ2_KMASK[i]) ? -1.0f : 1.0f);
            }
            const float *xs = xb + il * 16;
            acc = vfmaq_f32(acc, vld1q_f32(v + 0),  vld1q_f32(xs + 0));
            acc = vfmaq_f32(acc, vld1q_f32(v + 4),  vld1q_f32(xs + 4));
            acc = vfmaq_f32(acc, vld1q_f32(v + 8),  vld1q_f32(xs + 8));
            acc = vfmaq_f32(acc, vld1q_f32(v + 12), vld1q_f32(xs + 12));
        }
    }
    return vaddvq_f32(acc);
}

/* 一行 Q2_K（nb 个 256 元素块）与 x 的内积。
 * 块布局按 runtime：scales[16] | qs[64] | half d | half dmin（共 84 B）。 */
static float q2k_dot_row(const uint8_t *row, uint32_t nb, const float *x) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q2K_BYTES;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16;
        const float d = h2f(*(const uint16_t *)(blk + 80));
        const float dmin = h2f(*(const uint16_t *)(blk + 82));
        const float *xb = x + (size_t)b * Q2K_BLOCK;
        for (int il = 0; il < 16; il++) {
            const uint8_t *q = qs + 32 * (il / 8) + 16 * (il & 1);
            const int il2 = (il / 2) % 4;
            const float coef = il2 > 1 ? (il2 > 2 ? 1.0f / 64.0f : 1.0f / 16.0f)
                                       : (il2 > 0 ? 0.25f : 1.0f);
            const uint8_t mask = il2 > 1 ? (il2 > 2 ? 192 : 48) : (il2 > 0 ? 12 : 3);
            const uint8_t sc = scales[il];
            const float dl = d * (float)(sc & 0x0F) * coef;
            const float ml = dmin * (float)(sc >> 4);
            float v[16];
            for (int i = 0; i < 16; i++) v[i] = dl * (float)(q[i] & mask) - ml;
            const float *xs = xb + il * 16;
            acc = vfmaq_f32(acc, vld1q_f32(v + 0),  vld1q_f32(xs + 0));
            acc = vfmaq_f32(acc, vld1q_f32(v + 4),  vld1q_f32(xs + 4));
            acc = vfmaq_f32(acc, vld1q_f32(v + 8),  vld1q_f32(xs + 8));
            acc = vfmaq_f32(acc, vld1q_f32(v + 12), vld1q_f32(xs + 12));
        }
    }
    return vaddvq_f32(acc);
}

typedef struct {
    const uint8_t *gate;
    const uint8_t *up;
    const uint8_t *down;
    uint32_t dim, exp_dim, n_experts, n_act, nb_in, nb_mid;
    const float *x;
    const int32_t *ids;
    const float *wts;
    float *out;
    float clamp_val;
    uint32_t tok0, tok1;
} cpu_moe_arg_t;

static void *cpu_moe_worker(void *p) {
    const cpu_moe_arg_t *a = (const cpu_moe_arg_t *)p;
    const size_t row_gate = (size_t)a->nb_in * IQ2_BYTES;
    const size_t row_down = (size_t)a->nb_mid * Q2K_BYTES;
    float *gate = aligned_alloc(64, (size_t)a->exp_dim * sizeof(float));
    float *up   = aligned_alloc(64, (size_t)a->exp_dim * sizeof(float));
    float *mid  = aligned_alloc(64, (size_t)a->exp_dim * sizeof(float));
    float *acc  = aligned_alloc(64, (size_t)a->dim * sizeof(float));
    if (!gate || !up || !mid || !acc) { free(gate); free(up); free(mid); free(acc); return NULL; }

    const uint8_t *gate_base = a->gate;
    const uint8_t *up_base = a->up;
    const uint8_t *down_base = a->down;

    for (uint32_t t = a->tok0; t < a->tok1; t++) {
        const float *x = a->x + (size_t)t * a->dim;
        memset(acc, 0, (size_t)a->dim * sizeof(float));
        for (uint32_t k = 0; k < a->n_act; k++) {
            const int32_t e = a->ids[(size_t)t * a->n_act + k];
            if (e < 0 || (uint32_t)e >= a->n_experts) continue;
            const float w = a->wts[(size_t)t * a->n_act + k];
            const uint8_t *g = gate_base + (size_t)e * a->exp_dim * row_gate;
            const uint8_t *u = up_base + (size_t)e * a->exp_dim * row_gate;
            const uint8_t *dn = down_base + (size_t)e * a->dim * row_down;

            for (uint32_t r = 0; r < a->exp_dim; r++) {
                gate[r] = iq2_dot_row(g + (size_t)r * row_gate, a->nb_in, x);
                up[r]   = iq2_dot_row(u + (size_t)r * row_gate, a->nb_in, x);
            }
            /* swiglu（与 metal ds4_glm_swiglu 一致）：mid = silu(clamp(g)) * clamp(u) * w */
            const float c = a->clamp_val;
            for (uint32_t r = 0; r < a->exp_dim; r++) {
                float gv = gate[r], uv = up[r];
                if (c > 1.0e-6f) {
                    if (gv > c) gv = c;
                    if (uv > c) uv = c;
                    if (uv < -c) uv = -c;
                }
                const float silu = gv / (1.0f + expf(-gv));
                mid[r] = silu * uv * w;
            }
            for (uint32_t r = 0; r < a->dim; r++) {
                acc[r] += q2k_dot_row(dn + (size_t)r * row_down, a->nb_mid, mid);
            }
        }
        memcpy(a->out + (size_t)t * a->dim, acc, (size_t)a->dim * sizeof(float));
    }
    free(gate); free(up); free(mid); free(acc);
    return NULL;
}

/* 调试用：对单专家的一行块做 GEMV（out[r] = w 第 r 行 · x）。
 * fmt: 0 = IQ2_XXS(66B)，1 = Q2_K(84B)。仅测试调用。 */
int myds_cpu_moe_debug_gemv(const void *w, uint32_t rows, uint32_t nb, int fmt,
                            const float *x, float *out) {
    if (!w || !x || !out || !nb) return 0;
    const uint8_t *p = (const uint8_t *)w;
    const size_t rb = (size_t)nb * (fmt ? Q2K_BYTES : IQ2_BYTES);
    for (uint32_t r = 0; r < rows; r++)
        out[r] = fmt ? q2k_dot_row(p + (size_t)r * rb, nb, x)
                     : iq2_dot_row(p + (size_t)r * rb, nb, x);
    return 1;
}

int myds_cpu_moe_tokens(const void *gate_w,
                        const void *up_w,
                        const void *down_w,
                        uint32_t dim,
                        uint32_t exp_dim,
                        uint32_t n_experts,
                        const float *x,
                        const int32_t *ids,
                        const float *wts,
                        float *out,
                        uint32_t n_tok,
                        uint32_t n_act,
                        float clamp_val,
                        uint32_t nthreads) {
    if (!gate_w || !up_w || !down_w || !x || !ids || !wts || !out) return 0;
    if (!dim || !exp_dim || !n_experts || !n_tok || !n_act) return 0;
    if (dim % 256u || exp_dim % 256u) return 0;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > n_tok) nthreads = n_tok;

    pthread_t th[32];
    cpu_moe_arg_t arg[32];
    uint32_t started = 0;
    for (uint32_t i = 0; i < nthreads; i++) {
        arg[i] = (cpu_moe_arg_t){
            .gate = (const uint8_t *)gate_w,
            .up = (const uint8_t *)up_w,
            .down = (const uint8_t *)down_w,
            .dim = dim,
            .exp_dim = exp_dim,
            .n_experts = n_experts,
            .n_act = n_act,
            .nb_in = dim / 256u,
            .nb_mid = exp_dim / 256u,
            .x = x,
            .ids = ids,
            .wts = wts,
            .out = out,
            .clamp_val = clamp_val,
            .tok0 = (uint32_t)((uint64_t)n_tok * i / nthreads),
            .tok1 = (uint32_t)((uint64_t)n_tok * (i + 1) / nthreads),
        };
        if (pthread_create(&th[i], NULL, cpu_moe_worker, &arg[i]) != 0) break;
        started++;
    }
    if (!started) return 0;
    /* 线程不足时由调用线程补齐剩余区间 */
    if (started < nthreads) {
        cpu_moe_arg_t tail = arg[started];
        tail.tok0 = (uint32_t)((uint64_t)n_tok * started / nthreads);
        tail.tok1 = n_tok;
        cpu_moe_worker(&tail);
    }
    for (uint32_t i = 0; i < started; i++) pthread_join(th[i], NULL);
    return 1;
}
