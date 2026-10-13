// 反量化代价对照 kernel（2026-10-13 参谋长）。仅诊断，不进生产路径。
//
// 目的：判断「用四元码本替换 IQ2_XXS 的 256 项网格表 + 128 项符号表」值不值得
//       —— 多花的字节（若有）能否换回更便宜的反量化。
//
// 公平性设计：两个变体处理【同样多的权重】(5120/行)，且【字节数几乎相同】
//   ref     IQ2_XXS      : 20 块 × 66 B                = 1320 B
//   codebook ROCMFP2 式  : 640 B 码(2bit) + 160×4B 尺度 = 1280 B
//   ⇒ 两者都不触 DRAM（寄存器/常量表/L1 常驻），差异只来自【反量化指令】
//
// 反量化路径对比：
//   ref      : svalues[] 网格表(256×8B, threadgroup) + ssigns[] 符号表(128B) + 掩码分支
//   vec      : 同上但符号改算术 (popcount)，去掉符号表
//   q4code   : 每 16 权重 2bit 码 + 1 个 f32 尺度；码本 {-1,0,1,2}；无表、无符号位、无掩码
//
// 直接复用 moe.metal 里的 ds4_metal_iq2xxs_grid / ds4_metal_ksigns_iq2xs
// （探针 metallib 由 PREAMBLE + moe.metal + 本文件 拼源编译）。

#include <metal_stdlib>
using namespace metal;

#define DQ_ROW_W   5120
#define DQ_IQ2_BLK 20          // 20 × 256 = 5120 权重
#define DQ_CB_GRP  320         // 320 × 16  = 5120 权重

struct dq_args { int32_t rows, reps; };

// ── 基准：原 IQ2_XXS 反量化（网格表 + 符号表 + 掩码）──
kernel void bench_dq_ref_iq2(
        constant dq_args &a,
        device const char * wbuf,
        device const float * x,
        device float * out,
        threadgroup char * shmem [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong *svalues = (threadgroup ulong *)(shmem);
    threadgroup uchar *ssigns  = (threadgroup uchar *)(shmem + 256 * 8);
    {
        const int lt = 32 * sgitg + tiisg;
        if (lt < 64) {
            for (int i = 0; i < 4; ++i) svalues[lt * 4 + i] = ds4_metal_iq2xxs_grid[lt * 4 + i];
            for (int i = 0; i < 2; ++i) ssigns[lt * 2 + i] = ds4_metal_ksigns_iq2xs[lt * 2 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int row = (int)tgpig;
    device const uchar *base = (device const uchar *)wbuf + (size_t)row * 1320;
    float acc = 0.f;
    for (int k = 0; k < a.reps; ++k) {
        for (int ib = 0; ib < DQ_IQ2_BLK; ++ib) {
            device const uchar *blk = base + ib * 66;
            const float d = (float)(*(device const half *)blk);
            for (int sub = 0; sub < 4; ++sub) {
                device const uchar *q = blk + 2 + sub * 16;
                const uint32_t a32 = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
                float s = 0.f;
                for (int l = 0; l < 4; ++l) {
                    const threadgroup uchar *g = (const threadgroup uchar *)(svalues + q[l]);
                    const uchar sg = ssigns[(a32 >> (7 * l)) & 127u];
                    for (int j = 0; j < 8; ++j) {
                        const float v = x[(ib * 4 + sub) * 64 + l * 8 + j];
                        s += v * (float)g[j] * ((sg & ds4_metal_kmask_iq2xs[j]) ? -1.f : 1.f);
                    }
                }
                acc += d * s;
            }
        }
    }
    if (tiisg == 0) out[row] = acc;
}

// ── 变体 A：符号改算术（去掉 128 项符号表；等价于已落地的 popcount 路径）──
kernel void bench_dq_vec_iq2(
        constant dq_args &a,
        device const char * wbuf,
        device const float * x,
        device float * out,
        threadgroup char * shmem [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong *svalues = (threadgroup ulong *)(shmem);
    {
        const int lt = 32 * sgitg + tiisg;
        if (lt < 64) for (int i = 0; i < 4; ++i) svalues[lt * 4 + i] = ds4_metal_iq2xxs_grid[lt * 4 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int row = (int)tgpig;
    device const uchar *base = (device const uchar *)wbuf + (size_t)row * 1320;
    float acc = 0.f;
    for (int k = 0; k < a.reps; ++k) {
        for (int ib = 0; ib < DQ_IQ2_BLK; ++ib) {
            device const uchar *blk = base + ib * 66;
            const float d = (float)(*(device const half *)blk);
            for (int sub = 0; sub < 4; ++sub) {
                device const uchar *q = blk + 2 + sub * 16;
                const uint32_t a32 = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
                float s = 0.f;
                for (int l = 0; l < 4; ++l) {
                    const threadgroup uchar *g = (const threadgroup uchar *)(svalues + q[l]);
                    const uint32_t s7 = (a32 >> (7 * l)) & 127u;
                    const uchar sg = (uchar)(s7 | ((popcount(s7) & 1u) << 7));
                    for (int j = 0; j < 8; ++j) {
                        const float v = x[(ib * 4 + sub) * 64 + l * 8 + j];
                        s += v * (float)g[j] * ((sg & ds4_metal_kmask_iq2xs[j]) ? -1.f : 1.f);
                    }
                }
                acc += d * s;
            }
        }
    }
    if (tiisg == 0) out[row] = acc;
}

// 四元码本值表 {-1,0,1,2}（ROCMFP2 的码本）
constant float dq_cb[4] = {-1.f, 0.f, 1.f, 2.f};

// ── 变体 B(old)：四元码本，朴素的 4 路分支链解码（不公平，仅留作对照）──
kernel void bench_dq_q4code(
        constant dq_args &a,
        device const char * wbuf,
        device const float * x,
        device float * out,
        threadgroup char * shmem [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int row = (int)tgpig;
    device const uchar *codes  = (device const uchar *)wbuf + (size_t)row * 1280;
    device const float *scales = (device const float *)(codes + 640);
    float acc = 0.f;
    for (int k = 0; k < a.reps; ++k) {
        for (int g = 0; g < DQ_CB_GRP; ++g) {
            const device uchar *c4 = codes + g * 2;          // 16 个 2bit 码 = 4 B... 见下
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uchar c = (c4[j >> 2] >> ((j & 3) * 2)) & 3;
                const float v = (c == 0) ? -1.f : (c == 1 ? 0.f : (c == 2 ? 1.f : 2.f));
                s += v * x[g * 16 + j];
            }
            acc += scales[g] * s;
        }
    }
    if (tiisg == 0) out[row] = acc;
}

// ── 变体 C：四元码本 + 4 项常量表查值（ROCMFP2 的忠实实现）──
kernel void bench_dq_cb_table(
        constant dq_args &a,
        device const char * wbuf,
        device const float * x,
        device float * out,
        threadgroup char * shmem [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int row = (int)tgpig;
    device const uchar *codes  = (device const uchar *)wbuf + (size_t)row * 1280;
    device const float *scales = (device const float *)(codes + 640);
    float acc = 0.f;
    for (int k = 0; k < a.reps; ++k) {
        for (int g = 0; g < DQ_CB_GRP; ++g) {
            const device uchar *c4 = codes + g * 2;
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uchar c = (c4[j >> 2] >> ((j & 3) * 2)) & 3;
                s += dq_cb[c] * x[g * 16 + j];
            }
            acc += scales[g] * s;
        }
    }
    if (tiisg == 0) out[row] = acc;
}

