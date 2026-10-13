// Standalone Q2_K down probes. Excluded from build_metallib.py SHADER_FILES.
// Compile with the same production flags: -O3 -ffast-math.
// Frozen reference comes from metal/moe.metal:7299-7398 at HEAD 04dee3f.
// Only its exported name differs; block/expert/lane arithmetic is unchanged.
// Original kernel text SHA256: bac66b725d05728eeaaa6f4a2840df817a28c965a883083ee822330f50641e85
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define N_R0_Q2_K 2
constant short FC_mul_mv_nsg [[function_constant(600)]];

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

struct ds4_metal_args_mul_mv_id {
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

static inline bool ds4_tp_owns_expert(int expert, int n_total,
                                      int tp_rank, int tp_world) {
    if (tp_world <= 1) return true;
    const int first = tp_rank * (n_total / tp_world);
    const int last = tp_rank + 1 == tp_world ? n_total : first + n_total / tp_world;
    return expert >= first && expert < last;
}

kernel void probe_q2_down_ref_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (int64_t)(expert - args.tp_expert_base)*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

// Row-only specialization. Host dispatch must use NR1.
kernel void probe_q2_down_nr1_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = 1;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (int64_t)(expert - args.tp_expert_base)*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

// Row-only specialization. Host dispatch must use NR4.
kernel void probe_q2_down_nr4_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = 4;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (int64_t)(expert - args.tp_expert_base)*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void probe_q2_down_yvec_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (int64_t)(expert - args.tp_expert_base)*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            // Eight aligned float4 loads instead of 32 scalar load expressions.
            thread float4 * yl4 = (thread float4 *)yl;
            #pragma unroll
            for (short c = 0; c < 4; ++c) {
                device const float4 * yc = (device const float4 *)(y4 + 32*c);
                yl4[2*c+0] = yc[0];
                yl4[2*c+1] = yc[1];
            }
            // Preserve all four original eight-add chains in their original order.
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                sumy[0] += yl[i+ 0];
                sumy[1] += yl[i+ 8];
                sumy[2] += yl[i+16];
                sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void probe_q2_down_qvec_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (!ds4_tp_owns_expert(expert, args.ne02, args.tp_rank, args.tp_world)) continue;
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (int64_t)(expert - args.tp_expert_base)*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    // 84B blocks and 756B rows are not uniformly 8B aligned.
                    // packed_ushort4 preserves the original 2B alignment contract.
                    const ushort4 q = ushort4(*(device const packed_ushort4 *)qs);
                    #pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        const uint4 qv = uint4(q[i/2]);
                        const float4 even_y = {yl[i+0], yl[i+8], yl[i+16], yl[i+24]};
                        const float4 odd_y = {yl[i+1], yl[i+9], yl[i+17], yl[i+25]};
                        acc1 += even_y * float4(qv & uint4(0x0003,0x000c,0x0030,0x00c0));
                        acc2 += odd_y * float4(qv & uint4(0x0300,0x0c00,0x3000,0xc000));
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

// Fixed 5120/2304/384/6 scalar geometry. Dedicated experiment only.
// row=756B, expert=3870720B, MID=9216B/slot, nine K blocks, NR2.
// Preserves original expert/K walk and all floating expressions.
kernel void probe_q2_down_v41_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // Scalar V4.1 only. TP ranks with split expert mappings are ineligible.
    // Valid route IDs 0..383 and full rows fit all offsets in uint32_t:
    // the entire expert tensor is 1486356480B, below UINT32_MAX.
    if (args.tp_world > 1) return;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = 2;
    const int nb = 9;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * 24u);
    device const char *token_src1 = src1 + (uint64_t)token * 55296u;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (uint)expert*3870720u + (uint)first_row*756u);
        device const float * y = (device const float *)(token_src1 + expert_slot*9216u);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += 756u/2;
                sc += 756u;
                dh += 756u/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * 20480u);
    for (int row = 0; row < nr0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}
