// IQ2_XXS fixed-shape scalar V4.1 screening and optional production providers.
// Frozen production helper: metal/moe.metal ds4_iq2_pair_swiglu_popcnt_impl
// SHA256 (exact declaration + body, before identifier-only renames): 8344379eda8be1389fb0722ff8c4e7e1eb7a0667afeeda4208bed5c7bea27a75
// Standalone: xcrun metal -mmacosx-version-min=15.0 -O3 -ffast-math -c ...
// Mainlib: place after dense.metal/moe.metal; PREAMBLE defines FC_MUL_MV.
// Host must require scalar5120/2304/384/Top6/NR4, NSG2/4, full TP/base0,
// original F32 MID/write_clamped0 strides, and valid full buffer ranges.
// Unsupported providers must fail before encoding; there is no GPU fallback.
#include <metal_stdlib>
using namespace metal;
#ifdef FC_MUL_MV
#define MYDS_IQ2_V41_NSG FC_mul_mv_nsg
#else
constant short MYDS_IQ2_V41_NSG [[function_constant(600)]];
#endif
#define MYDS_IQ2_V41_QK 256
#define MYDS_IQ2_V41_NR 4

struct myds_iq2_v41_block {
    half d;
    ushort qs[MYDS_IQ2_V41_QK/8];
};

struct myds_iq2_v41_args {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
    /* Tensor-parallel expert ownership: when tp_world > 1, routed-MoE
     * kernels skip experts outside this rank's contiguous id range so this
     * rank never touches their weights. Zero-initialized (tp_world 0) means
     * no split; only the routed id kernels read these. */
    int32_t  tp_rank;
    int32_t  tp_world;
    /* When set, the sum6 kernels add this per-row vector (the shared-expert
     * partial) into their accumulation, folding the TP local-add. */
    int32_t  tp_addend;
    /* First expert id present at the bound blob base (TP mappings bind only
     * the owned range; kernels rebase before pointer math). */
    int32_t  tp_expert_base;
};

struct myds_iq2_v41_act {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

static inline bool myds_iq2_v41_owns_expert(int expert, int n_total,
                                      int tp_rank, int tp_world) {
    if (tp_world <= 1) return true;
    const int first = tp_rank * (n_total / tp_world);
    const int last = tp_rank + 1 == tp_world ? n_total : first + n_total / tp_world;
    return expert >= first && expert < last;
}

static constant uchar myds_iq2_v41_kmask[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar myds_iq2_v41_signs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong myds_iq2_v41_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static inline void myds_iq2_v41_frozen_impl(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = MYDS_IQ2_V41_NSG;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    tgpig.z = 0;
    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
    if (!myds_iq2_v41_owns_expert(i02, args.ne02, args.tp_rank, args.tp_world)) return;
    const int i02b = i02 - args.tp_expert_base;
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;
    const int nb = args.ne00 / MYDS_IQ2_V41_QK;
    const int first_row = (tgpig.x * NSG + sgitg) * MYDS_IQ2_V41_NR;
    const int nb32 = nb * (MYDS_IQ2_V41_QK / 32);
    device const myds_iq2_v41_block *xg =
        (device const myds_iq2_v41_block *)(src0_gate + (int64_t)i02b * args.nb02 + (uint64_t)first_row * args.nb01);
    device const myds_iq2_v41_block *xu =
        (device const myds_iq2_v41_block *)(src0_up + (int64_t)i02b * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);
    float yl[32];
    float sumg[MYDS_IQ2_V41_NR] = {0.f};
    float sumu[MYDS_IQ2_V41_NR] = {0.f};
    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        const int linear_tid = 32 * sgitg + tiisg;
        if (linear_tid < 64) {
            int nval = 4;
            int pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) svalues[pos + i] = myds_iq2_v41_grid[pos + i];
            nval = 2;
            pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) ssigns[pos + i] = myds_iq2_v41_signs[pos + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        device const float4 *y4_f4 = (device const float4 *)y4;
        thread float4 *yl_f4 = (thread float4 *)yl;
        #pragma unroll
        for (short i = 0; i < 8; ++i) yl_f4[i] = y4_f4[i];
        const int ibl = ib32 / (MYDS_IQ2_V41_QK / 32);
        const int ib  = ib32 % (MYDS_IQ2_V41_QK / 32);
        device const myds_iq2_v41_block *xgr = xg + ibl;
        device const myds_iq2_v41_block *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;
        for (short row = 0; row < MYDS_IQ2_V41_NR; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));
            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint s7g = (aux32g >> 7 * l) & 127u;
                const uint s7u = (aux32u >> 7 * l) & 127u;
                const uint8_t signg = (uint8_t)(s7g | ((popcount(s7g) & 1u) << 7));
                const uint8_t signu = (uint8_t)(s7u | ((popcount(s7u) & 1u) << 7));
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;
            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }
        y4 += 32 * 32;
    }
    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    float4 reduced_gate;
    float4 reduced_up;
    for (int row = 0; row < MYDS_IQ2_V41_NR; ++row) {
        reduced_gate[row] = simd_sum(sumg[row]);
        reduced_up[row] = simd_sum(sumu[row]);
    }
    if (tiisg < MYDS_IQ2_V41_NR && first_row + tiisg < args.ne0) {
        const uint out_row = first_row + tiisg;
        const float gate = reduced_gate[tiisg] * 0.25f;
        const float up = reduced_up[tiisg] * 0.25f;
        float g = gate;
        float u = up;
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
        }
        if (act.write_clamped != 0) {
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
        }
        const float silu = g / (1.0f + exp(-g));
        dst_mid_f32[out_row] = silu * u * route_weight;
    }
    (void)tiitg;
}

kernel void probe_iq2_v41_frozen_f32(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    myds_iq2_v41_frozen_impl(args, act, src0_gate, src0_up, src1,
        dst_gate, dst_up, dst_mid, ids, weights, shmem, tgpig, tiitg, tiisg, sgitg);
}

// shape64: address/control-only substitution; original FP loop and expressions.
static inline void myds_iq2_v41_shape64_impl(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = MYDS_IQ2_V41_NSG;
    if (args.tp_world > 1 || args.tp_expert_base != 0) return;
    const int iid1 = 0;
    const int idx  = (int)tgpig.z;
    tgpig.z = 0;
    const int32_t i02 = ((device const int32_t *) ids)[idx];
    if ((uint)i02 >= 384u) return;
    const int i02b = i02;
    const int64_t i11 = 0;
    const int64_t i12 = iid1;
    const int nb = 20;
    const int first_row = (tgpig.x * NSG + sgitg) * MYDS_IQ2_V41_NR;
    const int nb32 = nb * (MYDS_IQ2_V41_QK / 32);
    device const myds_iq2_v41_block *xg =
        (device const myds_iq2_v41_block *)(src0_gate + (int64_t)i02b * 3041280u + (uint64_t)first_row * 1320u);
    device const myds_iq2_v41_block *xu =
        (device const myds_iq2_v41_block *)(src0_up + (int64_t)i02b * 3041280u + (uint64_t)first_row * 1320u);
    device const float *y =
        (device const float *)(src1 + i11 * 20480u + i12 * 20480u);
    float yl[32];
    float sumg[MYDS_IQ2_V41_NR] = {0.f};
    float sumu[MYDS_IQ2_V41_NR] = {0.f};
    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        const int linear_tid = 32 * sgitg + tiisg;
        if (linear_tid < 64) {
            int nval = 4;
            int pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) svalues[pos + i] = myds_iq2_v41_grid[pos + i];
            nval = 2;
            pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) ssigns[pos + i] = myds_iq2_v41_signs[pos + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        device const float4 *y4_f4 = (device const float4 *)y4;
        thread float4 *yl_f4 = (thread float4 *)yl;
        #pragma unroll
        for (short i = 0; i < 8; ++i) yl_f4[i] = y4_f4[i];
        const int ibl = ib32 / (MYDS_IQ2_V41_QK / 32);
        const int ib  = ib32 % (MYDS_IQ2_V41_QK / 32);
        device const myds_iq2_v41_block *xgr = xg + ibl;
        device const myds_iq2_v41_block *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;
        for (short row = 0; row < MYDS_IQ2_V41_NR; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));
            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint s7g = (aux32g >> 7 * l) & 127u;
                const uint s7u = (aux32u >> 7 * l) & 127u;
                const uint8_t signg = (uint8_t)(s7g | ((popcount(s7g) & 1u) << 7));
                const uint8_t signu = (uint8_t)(s7u | ((popcount(s7u) & 1u) << 7));
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;
            dhg += 1320u / 2;
            dhu += 1320u / 2;
            qg  += 1320u / 2;
            qu  += 1320u / 2;
        }
        y4 += 32 * 32;
    }
    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * 2304 * 6 + (uint64_t)i11 * 2304;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * 2304 * 6 + (uint64_t)i11 * 2304;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)6 + (uint64_t)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + pair_row * 9216u);
    device const float *route_w =
        (device const float *)(weights + pair_row * 4u);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    float4 reduced_gate;
    float4 reduced_up;
    for (int row = 0; row < MYDS_IQ2_V41_NR; ++row) {
        reduced_gate[row] = simd_sum(sumg[row]);
        reduced_up[row] = simd_sum(sumu[row]);
    }
    if (tiisg < MYDS_IQ2_V41_NR) {
        const uint out_row = first_row + tiisg;
        const float gate = reduced_gate[tiisg] * 0.25f;
        const float up = reduced_up[tiisg] * 0.25f;
        float g = gate;
        float u = up;
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
        }
        if (false) {
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
        }
        const float silu = g / (1.0f + exp(-g));
        dst_mid_f32[out_row] = silu * u * route_weight;
    }
    (void)tiitg;
}

kernel void kernel_myds_iq2_v41_shape64_f32(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    myds_iq2_v41_shape64_impl(args, act, src0_gate, src0_up, src1,
        dst_gate, dst_up, dst_mid, ids, weights, shmem, tgpig, tiitg, tiisg, sgitg);
}

// addr32: address/control-only substitution; original FP loop and expressions.
static inline void myds_iq2_v41_addr32_impl(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = MYDS_IQ2_V41_NSG;
    if (args.tp_world > 1 || args.tp_expert_base != 0) return;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    tgpig.z = 0;
    const int32_t i02 = ((device const int32_t *) (ids + iid1 * (uint)args.nbi1))[idx];
    if ((uint)i02 >= 384u) return;
    const int i02b = i02 - args.tp_expert_base;
    const uint i11 = idx % args.ne11;
    const uint i12 = iid1;
    const int nb = args.ne00 / MYDS_IQ2_V41_QK;
    const int first_row = (tgpig.x * NSG + sgitg) * MYDS_IQ2_V41_NR;
    const int nb32 = nb * (MYDS_IQ2_V41_QK / 32);
    device const myds_iq2_v41_block *xg =
        (device const myds_iq2_v41_block *)(src0_gate + (uint)i02b * (uint)args.nb02 + (uint)first_row * (uint)args.nb01);
    device const myds_iq2_v41_block *xu =
        (device const myds_iq2_v41_block *)(src0_up + (uint)i02b * (uint)args.nb02 + (uint)first_row * (uint)args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * (uint)args.nb11 + i12 * (uint)args.nb12);
    float yl[32];
    float sumg[MYDS_IQ2_V41_NR] = {0.f};
    float sumu[MYDS_IQ2_V41_NR] = {0.f};
    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        const int linear_tid = 32 * sgitg + tiisg;
        if (linear_tid < 64) {
            int nval = 4;
            int pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) svalues[pos + i] = myds_iq2_v41_grid[pos + i];
            nval = 2;
            pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) ssigns[pos + i] = myds_iq2_v41_signs[pos + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        device const float4 *y4_f4 = (device const float4 *)y4;
        thread float4 *yl_f4 = (thread float4 *)yl;
        #pragma unroll
        for (short i = 0; i < 8; ++i) yl_f4[i] = y4_f4[i];
        const int ibl = ib32 / (MYDS_IQ2_V41_QK / 32);
        const int ib  = ib32 % (MYDS_IQ2_V41_QK / 32);
        device const myds_iq2_v41_block *xgr = xg + ibl;
        device const myds_iq2_v41_block *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;
        for (short row = 0; row < MYDS_IQ2_V41_NR; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));
            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint s7g = (aux32g >> 7 * l) & 127u;
                const uint s7u = (aux32u >> 7 * l) & 127u;
                const uint8_t signg = (uint8_t)(s7g | ((popcount(s7g) & 1u) << 7));
                const uint8_t signu = (uint8_t)(s7u | ((popcount(s7u) & 1u) << 7));
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;
            dhg += (uint)args.nb01 / 2;
            dhu += (uint)args.nb01 / 2;
            qg  += (uint)args.nb01 / 2;
            qu  += (uint)args.nb01 / 2;
        }
        y4 += 32 * 32;
    }
    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint)i12 * args.ne0 * args.ne1 + (uint)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint)i12 * args.ne0 * args.ne1 + (uint)i11 * args.ne0;
    const uint pair_row = (uint)i12 * (uint)args.nei0 + (uint)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + pair_row * (uint)act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + pair_row * (uint)act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    float4 reduced_gate;
    float4 reduced_up;
    for (int row = 0; row < MYDS_IQ2_V41_NR; ++row) {
        reduced_gate[row] = simd_sum(sumg[row]);
        reduced_up[row] = simd_sum(sumu[row]);
    }
    if (tiisg < MYDS_IQ2_V41_NR && first_row + tiisg < args.ne0) {
        const uint out_row = first_row + tiisg;
        const float gate = reduced_gate[tiisg] * 0.25f;
        const float up = reduced_up[tiisg] * 0.25f;
        float g = gate;
        float u = up;
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
        }
        if (act.write_clamped != 0) {
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
        }
        const float silu = g / (1.0f + exp(-g));
        dst_mid_f32[out_row] = silu * u * route_weight;
    }
    (void)tiitg;
}

kernel void kernel_myds_iq2_v41_addr32_f32(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    myds_iq2_v41_addr32_impl(args, act, src0_gate, src0_up, src1,
        dst_gate, dst_up, dst_mid, ids, weights, shmem, tgpig, tiitg, tiisg, sgitg);
}

// shape32: address/control-only substitution; original FP loop and expressions.
static inline void myds_iq2_v41_shape32_impl(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = MYDS_IQ2_V41_NSG;
    if (args.tp_world > 1 || args.tp_expert_base != 0) return;
    const int iid1 = 0;
    const int idx  = (int)tgpig.z;
    tgpig.z = 0;
    const int32_t i02 = ((device const int32_t *) ids)[idx];
    if ((uint)i02 >= 384u) return;
    const int i02b = i02;
    const uint i11 = 0;
    const uint i12 = iid1;
    const int nb = 20;
    const int first_row = (tgpig.x * NSG + sgitg) * MYDS_IQ2_V41_NR;
    const int nb32 = nb * (MYDS_IQ2_V41_QK / 32);
    device const myds_iq2_v41_block *xg =
        (device const myds_iq2_v41_block *)(src0_gate + (uint)i02b * 3041280u + (uint)first_row * 1320u);
    device const myds_iq2_v41_block *xu =
        (device const myds_iq2_v41_block *)(src0_up + (uint)i02b * 3041280u + (uint)first_row * 1320u);
    device const float *y =
        (device const float *)(src1 + i11 * 20480u + i12 * 20480u);
    float yl[32];
    float sumg[MYDS_IQ2_V41_NR] = {0.f};
    float sumu[MYDS_IQ2_V41_NR] = {0.f};
    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        const int linear_tid = 32 * sgitg + tiisg;
        if (linear_tid < 64) {
            int nval = 4;
            int pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) svalues[pos + i] = myds_iq2_v41_grid[pos + i];
            nval = 2;
            pos = linear_tid * nval;
            for (int i = 0; i < nval; ++i) ssigns[pos + i] = myds_iq2_v41_signs[pos + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        device const float4 *y4_f4 = (device const float4 *)y4;
        thread float4 *yl_f4 = (thread float4 *)yl;
        #pragma unroll
        for (short i = 0; i < 8; ++i) yl_f4[i] = y4_f4[i];
        const int ibl = ib32 / (MYDS_IQ2_V41_QK / 32);
        const int ib  = ib32 % (MYDS_IQ2_V41_QK / 32);
        device const myds_iq2_v41_block *xgr = xg + ibl;
        device const myds_iq2_v41_block *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;
        for (short row = 0; row < MYDS_IQ2_V41_NR; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));
            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint s7g = (aux32g >> 7 * l) & 127u;
                const uint s7u = (aux32u >> 7 * l) & 127u;
                const uint8_t signg = (uint8_t)(s7g | ((popcount(s7g) & 1u) << 7));
                const uint8_t signu = (uint8_t)(s7u | ((popcount(s7u) & 1u) << 7));
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & myds_iq2_v41_kmask[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;
            dhg += 1320u / 2;
            dhu += 1320u / 2;
            qg  += 1320u / 2;
            qu  += 1320u / 2;
        }
        y4 += 32 * 32;
    }
    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint)i12 * 2304 * 6 + (uint)i11 * 2304;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint)i12 * 2304 * 6 + (uint)i11 * 2304;
    const uint pair_row = (uint)i12 * (uint)6 + (uint)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + pair_row * 9216u);
    device const float *route_w =
        (device const float *)(weights + pair_row * 4u);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    float4 reduced_gate;
    float4 reduced_up;
    for (int row = 0; row < MYDS_IQ2_V41_NR; ++row) {
        reduced_gate[row] = simd_sum(sumg[row]);
        reduced_up[row] = simd_sum(sumu[row]);
    }
    if (tiisg < MYDS_IQ2_V41_NR) {
        const uint out_row = first_row + tiisg;
        const float gate = reduced_gate[tiisg] * 0.25f;
        const float up = reduced_up[tiisg] * 0.25f;
        float g = gate;
        float u = up;
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
        }
        if (false) {
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
        }
        const float silu = g / (1.0f + exp(-g));
        dst_mid_f32[out_row] = silu * u * route_weight;
    }
    (void)tiitg;
}

kernel void kernel_myds_iq2_v41_shape32_f32(
        constant myds_iq2_v41_args & args,
        constant myds_iq2_v41_act & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    myds_iq2_v41_shape32_impl(args, act, src0_gate, src0_up, src1,
        dst_gate, dst_up, dst_mid, ids, weights, shmem, tgpig, tiitg, tiisg, sgitg);
}

#undef MYDS_IQ2_V41_QK
#undef MYDS_IQ2_V41_NR
#ifdef FC_MUL_MV
#undef MYDS_IQ2_V41_NSG
#endif
