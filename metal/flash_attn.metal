#define FC_FLASH_ATTN_EXT_PAD 100
#define FC_FLASH_ATTN_EXT_BLK 200
#define FC_FLASH_ATTN_EXT 300
#define FC_FLASH_ATTN_EXT_VEC 400
#define FC_FLASH_ATTN_EXT_VEC_REDUCE 500
#define OP_FLASH_ATTN_EXT_NQPSG 8
#define OP_FLASH_ATTN_EXT_NCPSG 64
#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#ifndef PAD2
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#endif

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg);

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg);

struct ds4_metal_args_flash_attn_ext_pad {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext_blk {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
    /* 每 query 的 K/V 基址步长（字节）。非 0 时 `k`/`v` 再偏移 iq1*nb_kvq，用于把
     * 「本 query 可见的键」从稀疏前缀 gather 成紧凑 slab（见
     * docs/B_MAX_SPARSE_ATTENTION_SPEC.md）。为 0 时与旧行为完全一致。
     * ⚠ 必须与宿主侧 struct flash_vec_args 的最后一项逐字对齐。 */
    uint64_t nb_kvq;
};
/* 与宿主 C 侧 struct flash_vec_args 手工镜像 —— 在这里同样钉死尺寸/偏移，
 * 两侧任一边改动而另一边没跟，编译期就炸，不会跑出垃圾 K 地址。 */
static_assert(sizeof(struct ds4_metal_args_flash_attn_ext_vec) == 200,
              "manifest 与宿主 flash_vec_args 尺寸不一致");
static_assert(__builtin_offsetof(struct ds4_metal_args_flash_attn_ext_vec, nb_kvq) == 192,
              "nb_kvq 偏移与宿主不一致");

struct ds4_metal_args_flash_attn_ext_vec_reduce {
    int32_t nrows;
};

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];
constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// DS4 FlashAttention padding: pads the final partial K/V/mask cache block so the
// vector FlashAttention kernel can read full 32-row chunks.
kernel void kernel_flash_attn_ext_pad(
        constant ds4_metal_args_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// DS4 FlashAttention mask scan: marks blocks so the non-vector kernel can skip
// blocks that are entirely masked or entirely zero.
kernel void kernel_flash_attn_ext_blk(
        constant ds4_metal_args_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;
        const int32_t q0 = i1*Q;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            if (q0 + j < args.ne31) {
                device const half * mask_src =
                    (device const half *) (mask + (q0 + j)*args.nb31 + i2*args.nb32 + i3*args.nb33) +
                    i0*C + tiisg;

                FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                    mmin = min(mmin, mask_src[ii*NW]);
                    mmax = max(mmax, mask_src[ii*NW]);
                }
            }
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];

// DS4 non-vector FlashAttention. The only exported instance uses the model's
// 512-wide F16 K/V rows; keeping the template body generic preserves the same
// arithmetic for dense and compressed-attention prefill.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q,
    short C,
    short NSG,
    bool HEAD_BATCH = false>
void kernel_flash_attn_ext_impl(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV;

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            if (is_same<kd4x4_t, k4x4_t>::value) {
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[HEAD_BATCH ? iq1 + j : iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        const uint64_t rid = HEAD_BATCH ? (uint64_t)iq2*args.ne1 + iq1 + j :
            (uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1;
        device float4 * dst4 = (device float4 *) dst + rid*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

// Batched FlashAttention for prompt/prefill rows. It computes QK, applies mask,
// sinks, ALiBi/softcap options when enabled, and multiplies by V without
// materializing the full attention matrix.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q  = OP_FLASH_ATTN_EXT_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_NCPSG,
    bool HEAD_BATCH = false>
kernel void kernel_flash_attn_ext(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4, HEAD_BATCH>(FWD_ARGS); break;
        case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8, HEAD_BATCH>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

#define FA_NONVEC_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8

typedef decltype(kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>) flash_attn_ext_dk512_t;
typedef decltype(kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>) flash_attn_ext_dk256_t;

// Host-visible prefill FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]
kernel flash_attn_ext_dk512_t kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>;

// Reference smem-O eight-head variant retained for A/B 0-ULP and timing checks.
template [[host_name("kernel_myds_flash_attn_heads_smem_f16_dk512_dv512")]]
kernel flash_attn_ext_dk512_t kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512, 8, 64, true>;

// Host-visible prefill FlashAttention variant for GLM's 256-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]
kernel flash_attn_ext_dk256_t kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>;

#undef FA_NONVEC_TYPES

// Register-resident output accumulator (Reg-O) FlashAttention for F16 K/V heads.
// Eliminates the Q*PV float32 `so` bank (16 KiB at Q=8) and the unused 4 KiB
// `sk/sv` dequant bank, keeping `simdgroup_float8x8 lo[QB][NO_COL]` (32 floats
// per lane) live across K/V blocks and broadcasting per-row `ms` via `s_ms[Q]`.
//
// On Apple Silicon GPUs, an 8x8 `simdgroup_float8x8` fragment holds two
// adjacent column elements of the SAME row per SIMD lane:
//   row  = ((tiisg >> 2) & 4) | ((tiisg >> 1) & 3)   in [0, 7]
//   col0 = ((tiisg >> 1) & 4) | ((tiisg & 1) << 1)   in {0, 2, 4, 6}
//   col1 = col0 + 1                                  in {1, 3, 5, 7}
template<
    short DK,
    short DV,
    short Q,
    short C,
    short NSG,
    bool HEAD_BATCH = true,
    bool MASKLESS = false,
    bool DIRECT_SLAB = false>
void kernel_myds_flash_attn_heads_reg_o_impl(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    static_assert(DK == 512 && DV == 512, "Reg-O heads expects DK=512, DV=512");
    static_assert(Q % 8 == 0 && Q % NSG == 0, "Q must be multiple of 8 and NSG");
    static_assert((C / 8) % NSG == 0 && (C / 8) % 2 == 0, "C/8 must be divisible by NSG and 2");

    constexpr short DK4    = DK/4;
    constexpr short DK8    = DK/8;
    constexpr short PV     = PAD2(DV, 64);
    constexpr short PV8    = PV/8;
    static_assert(PV8 % (2*NSG) == 0, "PV8 must be divisible by 2*NSG");

    constexpr short NW     = N_SIMDWIDTH;
    constexpr short QB     = Q/8;
    constexpr short NQ     = Q/NSG;
    constexpr short NO_COL = PV8/NSG;
    static_assert(QB * NO_COL == 16, "exactly 16 simdgroup_float8x8 (32 floats/lane)");

    constexpr short SH = MASKLESS ? C : 2*C;
    constexpr short TS = 2*SH;

    // Shared memory layout (no `so`, no `sk/sv`):
    //   [0 .. Q*DK)                : sq      (Q*DK halves = 8 KiB @ Q=8, 16 KiB @ Q=16)
    //   [Q*DK .. Q*(DK+TS))        : ss+sm2  (Q*256 halves = 4 KiB @ Q=8) or ss only (2 KiB @ MASKLESS)
    //   [Q*(DK+TS) .. +2*Q)        : s_ms    (Q floats = 32 B @ Q=8, 64 B @ Q=16)
    // Total dynamic smem: 12,320 B (Q=8, NSG=4), 10,272 B (Q=8, NSG=4, MASKLESS -> 3 TGs/core), or 24,640 B (Q=16).
    threadgroup half   * sq   = shmem_f16;
    threadgroup half4  * sq4  = (threadgroup half4 *) shmem_f16;
    threadgroup float  * ss   = (threadgroup float *) (shmem_f16 + Q*DK);
    threadgroup float2 * ss2  = (threadgroup float2 *) (shmem_f16 + Q*DK);
    threadgroup half2  * sm2  = MASKLESS ? nullptr : (threadgroup half2 *) (shmem_f16 + Q*DK + 2*C);
    threadgroup float  * s_ms = (threadgroup float *) (shmem_f16 + Q*(DK + TS));

    const short lo_row = ((tiisg >> 2) & 4) | ((tiisg >> 1) & 3);
    const short lo_col = ((tiisg >> 1) & 4) | ((tiisg & 1) << 1);

    device const half2 * pm2[NQ];

    if (!MASKLESS) {
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
        }

        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    device const char * slab_k = DIRECT_SLAB ? (pad + (uint64_t)iq2*args.nb11) : nullptr;
    device const char * slab_v = DIRECT_SLAB ? (pad + (uint64_t)iq2*args.nb21) : nullptr;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (half4) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    simdgroup_float8x8 lo[QB][NO_COL];
    FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
        FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
            lo[qb][ii] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (!MASKLESS && FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = MASKLESS ? 2 : 1;

            if (!MASKLESS && FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne01 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            {
                device const char * k_src = (DIRECT_SLAB && ic < 128)
                    ? (slab_k + (uint64_t)ic*args.nb11)
                    : (k + (uint64_t)(DIRECT_SLAB ? (ic - 128) : ic)*args.nb11);
                device      const half  * pk = (device const half *) k_src + sgitg*(8*NS10);
                threadgroup       float * ps = ss + sgitg*8;

                constexpr short NC_QK = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC_QK; ++cc) {
                    simdgroup_float8x8 mqk[QB];
                    FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
                        mqk[qb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    }

                    #pragma unroll 16
                    for (short i = 0; i < DK8/2; ++i) {
                        simdgroup_barrier(mem_flags::mem_none);

                        simdgroup_half8x8 mk[2];
                        simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                        simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                        FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
                            simdgroup_half8x8 mq[2];
                            threadgroup const half * pq = sq + qb*8*DK;
                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk[qb], mq[0], mk[0], mqk[qb]);
                            simdgroup_multiply_accumulate(mqk[qb], mq[1], mk[1], mqk[qb]);
                        }
                    }

                    FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
                        simdgroup_store(mqk[qb], ps + qb*8*SH, SH, 0, false);
                    }

                    pk += 8*(NSG*NS10);
                    ps += 8*NSG;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                if (!MASKLESS && blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += float2(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += float2(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (tiisg == 0) {
                    s_ms[j] = ms;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                #pragma clang fp contract(off)
                FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
                    const float ms_r = s_ms[qb*8 + lo_row];
                    FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                        lo[qb][ii].thread_elements()[0] *= ms_r;
                        lo[qb][ii].thread_elements()[1] *= ms_r;
                    }
                }
            }

            {
                device const char * v_src = (DIRECT_SLAB && ic < 128)
                    ? (slab_v + (uint64_t)ic*args.nb21)
                    : (v + (uint64_t)(DIRECT_SLAB ? (ic - 128) : ic)*args.nb21);
                device const half * pv = (device const half *) v_src + 8*sgitg;

                constexpr short NC_PV = (C/8)/2;

                FOR_UNROLL (short cc = 0; cc < NC_PV; ++cc) {
                    simdgroup_float8x8 vs[QB][2];

                    FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
                        simdgroup_load(vs[qb][0], ss + qb*8*SH + 16*cc + 0, SH, 0, false);
                        simdgroup_load(vs[qb][1], ss + qb*8*SH + 16*cc + 8, SH, 0, false);
                    }

                    FOR_UNROLL (short ii = 0; ii < NO_COL/2; ++ii) {
                        simdgroup_half8x8 mv[4];

                        simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                        simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                        simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                        simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                        FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
                            simdgroup_multiply_accumulate(lo[qb][2*ii + 0], vs[qb][0], mv[0], lo[qb][2*ii + 0]);
                            simdgroup_multiply_accumulate(lo[qb][2*ii + 1], vs[qb][0], mv[1], lo[qb][2*ii + 1]);
                            simdgroup_multiply_accumulate(lo[qb][2*ii + 0], vs[qb][1], mv[2], lo[qb][2*ii + 0]);
                            simdgroup_multiply_accumulate(lo[qb][2*ii + 1], vs[qb][1], mv[3], lo[qb][2*ii + 1]);
                        }
                    }

                    pv += 2*8*NS20;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = (tiisg == 0 && (!HEAD_BATCH || iq1 + j < args.ne01))
                    ? ((device const float *) sinks)[HEAD_BATCH ? iq1 + j : iq2]
                    : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                if (tiisg == 0) {
                    s_ms[j] = ms;
                }
            }
        }
    }

    threadgroup float * s_scale = ss;
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        const float scale = S[jj] == 0.0f ? 0.0f : 1.0f/S[jj];
        if (tiisg == 0) {
            s_scale[j] = scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (FC_flash_attn_ext_has_sinks) {
        #pragma clang fp contract(off)
        FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
            const float ms_sink = s_ms[qb*8 + lo_row];
            FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                lo[qb][ii].thread_elements()[0] *= ms_sink;
                lo[qb][ii].thread_elements()[1] *= ms_sink;
            }
        }
    }

    FOR_UNROLL (short qb = 0; qb < QB; ++qb) {
        const short j = qb*8 + lo_row;
        if (iq1 + j < args.ne01) {
            #pragma clang fp contract(off)
            const float scale = s_scale[j];
            const uint64_t rid = HEAD_BATCH ? (uint64_t)iq2*args.ne1 + iq1 + j :
                (uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1;
            device float2 * dst2 = (device float2 *) ((device float *) dst + rid*DV);
            FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                const short c2 = (8*sgitg + ii*8*NSG + lo_col) / 2;
                dst2[c2] = float2(lo[qb][ii].thread_elements()[0] * scale,
                                  lo[qb][ii].thread_elements()[1] * scale);
            }
        }
    }

#undef NS10
#undef NS20
}

template<
    short DK,
    short DV,
    short Q,
    short C,
    short NSG,
    bool HEAD_BATCH = true,
    bool MASKLESS = false,
    bool DIRECT_SLAB = false>
kernel void kernel_myds_flash_attn_heads_reg_o(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    if (FC_flash_attn_ext_nsg == NSG) {
        kernel_myds_flash_attn_heads_reg_o_impl<DK, DV, Q, C, NSG, HEAD_BATCH, MASKLESS, DIRECT_SLAB>(
            args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg);
    }
}

typedef decltype(kernel_myds_flash_attn_heads_reg_o<512, 512, 8, 64, 4, true>) flash_attn_heads_reg_o_t;
typedef decltype(kernel_myds_flash_attn_heads_reg_o<512, 512, 8, 64, 4, true, true, true>) flash_attn_heads_reg_o_fast_t;

// Step 1 (Q=8, C=64, NSG=4 Reg-O): 12,320 B smem (2 TGs = 256 threads/core).
template [[host_name("kernel_myds_flash_attn_heads_f16_dk512_dv512")]]
kernel flash_attn_heads_reg_o_t kernel_myds_flash_attn_heads_reg_o<512, 512, 8, 64, 4, true>;

// [P-NEW-3] Maskless + Direct Window Slab Reg-O (Q=8, C=64, NSG=4):
// SH=C=64 => 10,272 B smem (3 TGs = 384 threads/core, +50% occupancy).
template [[host_name("kernel_myds_flash_attn_heads_maskless_slab_f16_dk512_dv512")]]
kernel flash_attn_heads_reg_o_fast_t kernel_myds_flash_attn_heads_reg_o<512, 512, 8, 64, 4, true, true, true>;

// Step 2 (Q=16, C=64, NSG=8 Reg-O): 24,640 B smem (1 TG = 256 threads/core).
template [[host_name("kernel_myds_flash_attn_heads_q16_f16_dk512_dv512")]]
kernel flash_attn_heads_reg_o_t kernel_myds_flash_attn_heads_reg_o<512, 512, 16, 64, 8, true>;

// [CAND-A2] Direct-Indirect Compressed KV inside q8 Maskless-Slab Reg-O
// (Q=8, C=64, NSG=4, DK=512, DV=512, 10,272 B smem -> 3 TGs/core on Apple8).
//
// Eliminates gather_comp_512 (0 bytes written to kv_gathered, 0 gather dispatch)
// by reading compressed KV rows directly from comp_kv via ids[iq2 * top_k + j]
// for ic >= 128, while keeping ic < 128 direct-slab reads identical to
// heads_maskless_slab. Explicitly scalarized lo0..lo15 eliminates the 288 B
// Apple8 stack alloca spill of the array-indexed Reg-O path (92 VGPRs, 0 B spill,
// Apple8 maxThr = 576, 3 TGs/core).
//
// Strict 0-ULP (memcmp == 0) vs gather_comp_512 + heads_maskless_slab:
//   - STORAGE == 2u (NVFP4, 288 B/row, 32B aligned): Option A direct-to-register
//     uint4 QK + uint2 PV with exact 16-entry E2M1FN x E4M3FN -> half bijection.
//   - STORAGE == 1u (BF16, 1024 B/row, 2B aligned): Option B1 32x32 coalesced QK
//     stage reusing ss (0 extra smem) + Option A 1-key PV.
struct ds4_metal_args_flash_attn_direct_comp {
    ds4_metal_args_flash_attn_ext ext;
    uint32_t n_comp;
    uint32_t top_k;
};
static_assert(sizeof(ds4_metal_args_flash_attn_direct_comp) == 200,
              "ds4_metal_args_flash_attn_direct_comp must stay 200 bytes");
static_assert(__builtin_offsetof(ds4_metal_args_flash_attn_direct_comp, n_comp) == 192,
              "n_comp must live at offset 192");
static_assert(__builtin_offsetof(ds4_metal_args_flash_attn_direct_comp, top_k) == 196,
              "top_k must live at offset 196");

constant static constexpr half k_myds_fa_e2m1_f16[16] = {
     0.0h,  0.5h,  1.0h,  1.5h,  2.0h,  3.0h,  4.0h,  6.0h,
    -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
};

static inline half myds_fa_e4m3fn_to_f16(uint sc) {
    const uint s = sc & 0x7fu;
    const half norm = as_type<half>(ushort(0x2000u + (s << 7u)));
    const half sub  = half(s) * half(0.001953125h);
    return select(norm, sub, s < 8u);
}

static inline half2 myds_fa_nvfp4_qk_f16x2(uint c0, uint c1, half2 sc) {
    #pragma clang fp contract(off)
    return half2(k_myds_fa_e2m1_f16[c0], k_myds_fa_e2m1_f16[c1]) * sc;
}

static inline half2 myds_fa_nvfp4_pv_f16x2(uint byte8, half sc) {
    #pragma clang fp contract(off)
    return half2(k_myds_fa_e2m1_f16[byte8 & 0xfu],
                 k_myds_fa_e2m1_f16[(byte8 >> 4u) & 0xfu]) * sc;
}

static inline half2 myds_fa_bf16x2_to_f16x2(ushort u0, ushort u1) {
    return half2(as_type<float2>(uint2(u0, u1) << 16u));
}

template<uint STORAGE>
void kernel_myds_flash_attn_heads_maskless_slab_direct_comp_impl(
        constant ds4_metal_args_flash_attn_direct_comp & dargs,
        device const char * q,
        device const char * comp_kv,
        device const int  * ids,
        device const char * sinks,
        device const char * slab,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    constant ds4_metal_args_flash_attn_ext & args = dargs.ext;
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0] * 8;

#define DC_NS10 (FC_flash_attn_ext_ns10)
#define DC_NS20 (FC_flash_attn_ext_ns20)

    constexpr short DK  = 512;
    constexpr short DV  = 512;
    constexpr short Q   = 8;
    constexpr short C   = 64;
    constexpr short NSG = 4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short DK4 = DK / 4;
    constexpr short DK8 = DK / 8;
    constexpr short NQ  = Q / NSG;
    constexpr short SH  = C;
    constexpr short TS  = 2 * SH;

    // Shared memory layout (10,272 B total, identical to heads_maskless_slab -> 3 TGs/core):
    //   [0 .. 4096)        : sq   (8 * 512 halves = 8,192 B)
    //   [4096 .. 5120)     : ss   (8 * 64 floats  = 2,048 B; also reused as 32x32 sk in BF16 QK^T)
    //   [5120 .. 5136)     : s_ms (8 floats       =    32 B)
    threadgroup half   * sq   = shmem_f16;
    threadgroup half4  * sq4  = (threadgroup half4 *) shmem_f16;
    threadgroup float  * ss   = (threadgroup float *) (shmem_f16 + Q * DK);
    threadgroup float2 * ss2  = (threadgroup float2 *) (shmem_f16 + Q * DK);
    threadgroup float  * s_ms = (threadgroup float *) (shmem_f16 + Q * (DK + TS));

    const short lo_row = ((tiisg >> 2) & 4) | ((tiisg >> 1) & 3);
    const short lo_col = ((tiisg >> 1) & 4) | ((tiisg & 1) << 1);

    q += iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03;
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *) ((device const char *) q + j * args.nb01);
        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j * DK4 + i] = (half4) q4[i];
            } else {
                sq4[j * DK4 + i] = 0;
            }
        }
    }

    // Explicitly scalarized 16 x simdgroup_float8x8 output accumulators (32 floats/lane,
    // 0 stack alloca / 0 B spill on Apple8 & Apple9). Partitioned as 16*sgitg + 64*jj + {0,8}
    // so each (jj, sgitg) pair aligns 1:1 with a single 16-element NVFP4 scale block.
    simdgroup_float8x8 lo0  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo1  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo2  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo3  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo4  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo5  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo6  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo7  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo8  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo9  = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo10 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo11 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo12 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo13 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo14 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 lo15 = make_filled_simdgroup_matrix<float, 8>(0.0f);

#define DC_SCALE1(x, s) do { (x).thread_elements()[0] *= (s); (x).thread_elements()[1] *= (s); } while (0)
#define DC_SCALE_ALL(s) do { \
    DC_SCALE1(lo0,  s); DC_SCALE1(lo1,  s); DC_SCALE1(lo2,  s); DC_SCALE1(lo3,  s); \
    DC_SCALE1(lo4,  s); DC_SCALE1(lo5,  s); DC_SCALE1(lo6,  s); DC_SCALE1(lo7,  s); \
    DC_SCALE1(lo8,  s); DC_SCALE1(lo9,  s); DC_SCALE1(lo10, s); DC_SCALE1(lo11, s); \
    DC_SCALE1(lo12, s); DC_SCALE1(lo13, s); DC_SCALE1(lo14, s); DC_SCALE1(lo15, s); \
} while (0)

#define DC_SOFTMAX_AND_SCALE() do { \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    { \
        const short j0 = sgitg; \
        const float m = M0; \
        float2 s2 = ss2[j0 * SH / 2 + tiisg] * args.scale; \
        if (FC_flash_attn_ext_has_scap) { s2 = args.logit_softcap * precise::tanh(s2); } \
        M0 = simd_max(max(M0, max(s2[0], s2[1]))); \
        const float  ms  = exp(m  - M0); \
        const float2 vs2 = exp(s2 - M0); \
        S0 = S0 * ms + simd_sum(vs2[0] + vs2[1]); \
        ss2[j0 * SH / 2 + tiisg] = vs2; \
        if (tiisg == 0) s_ms[j0] = ms; \
    } \
    { \
        const short j1 = NSG + sgitg; \
        const float m = M1; \
        float2 s2 = ss2[j1 * SH / 2 + tiisg] * args.scale; \
        if (FC_flash_attn_ext_has_scap) { s2 = args.logit_softcap * precise::tanh(s2); } \
        M1 = simd_max(max(M1, max(s2[0], s2[1]))); \
        const float  ms  = exp(m  - M1); \
        const float2 vs2 = exp(s2 - M1); \
        S1 = S1 * ms + simd_sum(vs2[0] + vs2[1]); \
        ss2[j1 * SH / 2 + tiisg] = vs2; \
        if (tiisg == 0) s_ms[j1] = ms; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    { \
        _Pragma("clang fp contract(off)") \
        const float ms_r = s_ms[lo_row]; \
        DC_SCALE_ALL(ms_r); \
    } \
} while (0)

#define DC_PV_SLAB_PAIR(jj, a, b) do { \
    simdgroup_half8x8 mv0, mv1; \
    simdgroup_load(mv0, pv + (jj) * 64 + 0, DC_NS20, 0, false); \
    simdgroup_load(mv1, pv + (jj) * 64 + 8, DC_NS20, 0, false); \
    simdgroup_multiply_accumulate(a, vs, mv0, a); \
    simdgroup_multiply_accumulate(b, vs, mv1, b); \
} while (0)

#define DC_PV_NVFP4_PAIR(jj, a, b) do { \
    const uint blk = (jj) * 4u + sgitg; \
    const half sc = vis ? myds_fa_e4m3fn_to_f16(uint(row288[256u + blk])) : 0.0h; \
    const uint2 w = vis ? ((device const uint2 *)row288)[blk] : uint2(0u, 0u); \
    const half2 v0 = myds_fa_nvfp4_pv_f16x2(w.x >> col_sh, sc); \
    const half2 v1 = myds_fa_nvfp4_pv_f16x2(w.y >> col_sh, sc); \
    simdgroup_half8x8 mv0, mv1; \
    mv0.thread_elements()[0] = v0.x; mv0.thread_elements()[1] = v0.y; \
    mv1.thread_elements()[0] = v1.x; mv1.thread_elements()[1] = v1.y; \
    simdgroup_multiply_accumulate(a, vs, mv0, a); \
    simdgroup_multiply_accumulate(b, vs, mv1, b); \
} while (0)

#define DC_PV_BF16_PAIR(jj, a, b) do { \
    const uint base = (jj) * 64u + sgitg * 16u + uint(lo_col); \
    const half2 v0 = vis ? myds_fa_bf16x2_to_f16x2(row_bf16[base + 0u], row_bf16[base + 1u]) : half2(0.0h); \
    const half2 v1 = vis ? myds_fa_bf16x2_to_f16x2(row_bf16[base + 8u], row_bf16[base + 9u]) : half2(0.0h); \
    simdgroup_half8x8 mv0, mv1; \
    mv0.thread_elements()[0] = v0.x; mv0.thread_elements()[1] = v0.y; \
    mv1.thread_elements()[0] = v1.x; mv1.thread_elements()[1] = v1.y; \
    simdgroup_multiply_accumulate(a, vs, mv0, a); \
    simdgroup_multiply_accumulate(b, vs, mv1, b); \
} while (0)

#define DC_STORE_PAIR(jj, a, b) do { \
    const short c0 = ((jj) * 64 + 16 * sgitg + 0 + lo_col) / 2; \
    const short c1 = ((jj) * 64 + 16 * sgitg + 8 + lo_col) / 2; \
    dst2[c0] = float2((a).thread_elements()[0] * scale, (a).thread_elements()[1] * scale); \
    dst2[c1] = float2((b).thread_elements()[0] * scale, (b).thread_elements()[1] * scale); \
} while (0)

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S0 = 0.0f, S1 = 0.0f;
    float M0 = -FLT_MAX / 2, M1 = -FLT_MAX / 2;

    // Phase 1: Sliding-window slab (ic = 0, 64 -> 128 keys)
    {
        device const char * slab_k = slab + (uint64_t)iq2 * args.nb11;
        device const char * slab_v = slab + (uint64_t)iq2 * args.nb21;
        for (short ic0 = 0; ic0 < 2; ++ic0) {
            const int ic = ic0 * C;
            if (ic >= args.ne11) break;
            {
                device const half * pk = (device const half *)(slab_k + (uint64_t)ic * args.nb11) + sgitg * (8 * DC_NS10);
                threadgroup float * ps = ss + sgitg * 8;
                FOR_UNROLL (short cc = 0; cc < 2; ++cc) {
                    simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    for (short i = 0; i < DK8 / 2; ++i) {
                        simdgroup_half8x8 mk0, mk1, mq0, mq1;
                        simdgroup_load(mk0, pk + 0 + 16 * i, DC_NS10, 0, true);
                        simdgroup_load(mk1, pk + 8 + 16 * i, DC_NS10, 0, true);
                        simdgroup_load(mq0, sq + 0 + 16 * i, DK);
                        simdgroup_load(mq1, sq + 8 + 16 * i, DK);
                        simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                        simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                    }
                    simdgroup_store(mqk, ps, SH, 0, false);
                    pk += 8 * (NSG * DC_NS10);
                    ps += 8 * NSG;
                }
            }
            DC_SOFTMAX_AND_SCALE();
            {
                device const half * pv = (device const half *)(slab_v + (uint64_t)ic * args.nb21) + 16 * sgitg;
                for (short cc = 0; cc < 8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                    DC_PV_SLAB_PAIR(0, lo0,  lo1);  DC_PV_SLAB_PAIR(1, lo2,  lo3);
                    DC_PV_SLAB_PAIR(2, lo4,  lo5);  DC_PV_SLAB_PAIR(3, lo6,  lo7);
                    DC_PV_SLAB_PAIR(4, lo8,  lo9);  DC_PV_SLAB_PAIR(5, lo10, lo11);
                    DC_PV_SLAB_PAIR(6, lo12, lo13); DC_PV_SLAB_PAIR(7, lo14, lo15);
                    pv += 8 * DC_NS20;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Phase 2: Direct-Indirect Compressed KV (ic = 128 .. ne11, typically 8 tiles x 64 = 512 keys)
    {
        device const int * row_ids = ids + (uint64_t)iq2 * dargs.top_k;
        const uint row_sh = (uint)lo_row * 4u;
        const uint col_sh = (uint)lo_col * 4u;
        const int n_comp_tiles = (args.ne11 > 128) ? ((args.ne11 - 128) / C) : 0;
        for (int ic0 = 0; ic0 < n_comp_tiles; ++ic0) {
            const uint comp_base = (uint)ic0 * 64u;
            if (STORAGE == 2u) {
                // NVFP4 (288 B/row): Option A Direct-to-Register QK^T (128-bit uint4 broadcast loads)
                threadgroup float * ps = ss + sgitg * 8;
                FOR_UNROLL (short cc = 0; cc < 2; ++cc) {
                    const uint k_off = comp_base + (uint)cc * 32u + (uint)sgitg * 8u + (uint)lo_col;
                    const int id0 = row_ids[k_off + 0u];
                    const int id1 = row_ids[k_off + 1u];
                    const bool v0 = (id0 >= 0) && ((uint)id0 < dargs.n_comp);
                    const bool v1 = (id1 >= 0) && ((uint)id1 < dargs.n_comp);
                    device const uchar * r0 = (device const uchar *)(comp_kv + (uint64_t)(v0 ? id0 : 0) * 288u);
                    device const uchar * r1 = (device const uchar *)(comp_kv + (uint64_t)(v1 ? id1 : 0) * 288u);
                    simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    for (short i2 = 0; i2 < 16; ++i2) {
                        const ushort sc0p = v0 ? ((device const ushort *)(r0 + 256u))[i2] : 0u;
                        const ushort sc1p = v1 ? ((device const ushort *)(r1 + 256u))[i2] : 0u;
                        const uint4 w0 = v0 ? ((device const uint4 *)r0)[i2] : uint4(0u);
                        const uint4 w1 = v1 ? ((device const uint4 *)r1)[i2] : uint4(0u);
                        {
                            const half2 sc = half2(v0 ? myds_fa_e4m3fn_to_f16(uint(sc0p) & 0xffu) : 0.0h,
                                                   v1 ? myds_fa_e4m3fn_to_f16(uint(sc1p) & 0xffu) : 0.0h);
                            const half2 k0 = myds_fa_nvfp4_qk_f16x2((w0.x >> row_sh) & 0xfu, (w1.x >> row_sh) & 0xfu, sc);
                            const half2 k1 = myds_fa_nvfp4_qk_f16x2((w0.y >> row_sh) & 0xfu, (w1.y >> row_sh) & 0xfu, sc);
                            simdgroup_half8x8 mk0, mk1, mq0, mq1;
                            mk0.thread_elements()[0] = k0.x; mk0.thread_elements()[1] = k0.y;
                            mk1.thread_elements()[0] = k1.x; mk1.thread_elements()[1] = k1.y;
                            simdgroup_load(mq0, sq + 32 * i2 + 0, DK);
                            simdgroup_load(mq1, sq + 32 * i2 + 8, DK);
                            simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                            simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                        }
                        {
                            const half2 sc = half2(v0 ? myds_fa_e4m3fn_to_f16(uint(sc0p) >> 8u) : 0.0h,
                                                   v1 ? myds_fa_e4m3fn_to_f16(uint(sc1p) >> 8u) : 0.0h);
                            const half2 k0 = myds_fa_nvfp4_qk_f16x2((w0.z >> row_sh) & 0xfu, (w1.z >> row_sh) & 0xfu, sc);
                            const half2 k1 = myds_fa_nvfp4_qk_f16x2((w0.w >> row_sh) & 0xfu, (w1.w >> row_sh) & 0xfu, sc);
                            simdgroup_half8x8 mk0, mk1, mq0, mq1;
                            mk0.thread_elements()[0] = k0.x; mk0.thread_elements()[1] = k0.y;
                            mk1.thread_elements()[0] = k1.x; mk1.thread_elements()[1] = k1.y;
                            simdgroup_load(mq0, sq + 32 * i2 + 16, DK);
                            simdgroup_load(mq1, sq + 32 * i2 + 24, DK);
                            simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                            simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                        }
                    }
                    simdgroup_store(mqk, ps, SH, 0, false);
                    ps += 8 * NSG;
                }
            } else {
                // BF16 (1024 B/row): Option B1 32x32 coalesced QK^T stage reusing ss (2,048 B, 0 extra smem)
                const ushort tiitg = sgitg * 32u + tiisg;
                const uint k_local = tiitg >> 2;
                const uint sub4 = tiitg & 3u;
                threadgroup half * sk = (threadgroup half *)ss;
                threadgroup half4 * dst_sk = (threadgroup half4 *)(sk + k_local * 32u + sub4 * 8u);
                simdgroup_float8x8 mqk0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_float8x8 mqk1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
                FOR_UNROLL (short cc = 0; cc < 2; ++cc) {
                    const int kid = row_ids[comp_base + (uint)cc * 32u + k_local];
                    const bool kvis = (kid >= 0) && ((uint)kid < dargs.n_comp);
                    device const ushort * krow = (device const ushort *)(comp_kv + (uint64_t)(kvis ? kid : 0) * 1024u);
                    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    for (short dblk = 0; dblk < 16; ++dblk) {
                        const uint d0 = (uint)dblk * 32u;
                        if (kvis) {
                            device const ushort * s = krow + d0 + sub4 * 8u;
                            dst_sk[0] = half4(as_type<float4>(uint4(s[0], s[1], s[2], s[3]) << 16u));
                            dst_sk[1] = half4(as_type<float4>(uint4(s[4], s[5], s[6], s[7]) << 16u));
                        } else {
                            dst_sk[0] = half4(0.0h);
                            dst_sk[1] = half4(0.0h);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        threadgroup const half * pk_s = sk + sgitg * (8 * 32);
                        FOR_UNROLL (short kstep = 0; kstep < 4; ++kstep) {
                            simdgroup_half8x8 mk, mq;
                            simdgroup_load(mk, pk_s + kstep * 8, 32, 0, true);
                            simdgroup_load(mq, sq + d0 + kstep * 8, DK);
                            simdgroup_multiply_accumulate(acc, mq, mk, acc);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    if (cc == 0) mqk0 = acc; else mqk1 = acc;
                }
                simdgroup_store(mqk0, ss + sgitg * 8, SH, 0, false);
                simdgroup_store(mqk1, ss + sgitg * 8 + 32, SH, 0, false);
            }

            DC_SOFTMAX_AND_SCALE();

            // P x V: Option A 1-key direct-to-register (16*sgitg + 64*jj layout)
            for (short cc = 0; cc < 8; ++cc) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                const int idv = row_ids[comp_base + (uint)cc * 8u + (uint)lo_row];
                const bool vis = (idv >= 0) && ((uint)idv < dargs.n_comp);
                if (STORAGE == 2u) {
                    device const uchar * row288 = (device const uchar *)(comp_kv + (uint64_t)(vis ? idv : 0) * 288u);
                    DC_PV_NVFP4_PAIR(0, lo0,  lo1);  DC_PV_NVFP4_PAIR(1, lo2,  lo3);
                    DC_PV_NVFP4_PAIR(2, lo4,  lo5);  DC_PV_NVFP4_PAIR(3, lo6,  lo7);
                    DC_PV_NVFP4_PAIR(4, lo8,  lo9);  DC_PV_NVFP4_PAIR(5, lo10, lo11);
                    DC_PV_NVFP4_PAIR(6, lo12, lo13); DC_PV_NVFP4_PAIR(7, lo14, lo15);
                } else {
                    device const ushort * row_bf16 = (device const ushort *)(comp_kv + (uint64_t)(vis ? idv : 0) * 1024u);
                    DC_PV_BF16_PAIR(0, lo0,  lo1);  DC_PV_BF16_PAIR(1, lo2,  lo3);
                    DC_PV_BF16_PAIR(2, lo4,  lo5);  DC_PV_BF16_PAIR(3, lo6,  lo7);
                    DC_PV_BF16_PAIR(4, lo8,  lo9);  DC_PV_BF16_PAIR(5, lo10, lo11);
                    DC_PV_BF16_PAIR(6, lo12, lo13); DC_PV_BF16_PAIR(7, lo14, lo15);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (FC_flash_attn_ext_has_sinks) {
        {
            const short j0 = sgitg;
            const float m = M0;
            const float s = (tiisg == 0 && iq1 + j0 < args.ne01)
                ? ((device const float *) sinks)[iq1 + j0] : -FLT_MAX / 2;
            M0 = simd_max(max(M0, s));
            const float ms = exp(m - M0);
            const float vs = exp(s - M0);
            S0 = S0 * ms + simd_sum(vs);
            if (tiisg == 0) s_ms[j0] = ms;
        }
        {
            const short j1 = NSG + sgitg;
            const float m = M1;
            const float s = (tiisg == 0 && iq1 + j1 < args.ne01)
                ? ((device const float *) sinks)[iq1 + j1] : -FLT_MAX / 2;
            M1 = simd_max(max(M1, s));
            const float ms = exp(m - M1);
            const float vs = exp(s - M1);
            S1 = S1 * ms + simd_sum(vs);
            if (tiisg == 0) s_ms[j1] = ms;
        }
    }

    threadgroup float * s_scale = ss;
    if (tiisg == 0) {
        s_scale[sgitg]       = S0 == 0.0f ? 0.0f : 1.0f / S0;
        s_scale[NSG + sgitg] = S1 == 0.0f ? 0.0f : 1.0f / S1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (FC_flash_attn_ext_has_sinks) {
        #pragma clang fp contract(off)
        const float ms_sink = s_ms[lo_row];
        DC_SCALE_ALL(ms_sink);
    }

    if (iq1 + lo_row < args.ne01) {
        #pragma clang fp contract(off)
        const float scale = s_scale[lo_row];
        const uint64_t rid = (uint64_t)iq2 * args.ne1 + iq1 + lo_row;
        device float2 * dst2 = (device float2 *) ((device float *) dst + rid * DV);
        DC_STORE_PAIR(0, lo0,  lo1);  DC_STORE_PAIR(1, lo2,  lo3);
        DC_STORE_PAIR(2, lo4,  lo5);  DC_STORE_PAIR(3, lo6,  lo7);
        DC_STORE_PAIR(4, lo8,  lo9);  DC_STORE_PAIR(5, lo10, lo11);
        DC_STORE_PAIR(6, lo12, lo13); DC_STORE_PAIR(7, lo14, lo15);
    }

#undef DC_NS10
#undef DC_NS20
#undef DC_SCALE1
#undef DC_SCALE_ALL
#undef DC_SOFTMAX_AND_SCALE
#undef DC_PV_SLAB_PAIR
#undef DC_PV_NVFP4_PAIR
#undef DC_PV_BF16_PAIR
#undef DC_STORE_PAIR
}

template<uint STORAGE>
kernel void kernel_myds_flash_attn_heads_maskless_slab_direct_comp(
        constant ds4_metal_args_flash_attn_direct_comp & args,
        device const char * q,
        device const char * comp_kv,
        device const int  * ids,
        device const char * sinks,
        device const char * slab,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {
    if (FC_flash_attn_ext_nsg == 4) {
        kernel_myds_flash_attn_heads_maskless_slab_direct_comp_impl<STORAGE>(
            args, q, comp_kv, ids, sinks, slab, dst, shmem_f16, tgpig, tiisg, sgitg);
    }
}

typedef decltype(kernel_myds_flash_attn_heads_maskless_slab_direct_comp<2u>) flash_attn_heads_direct_comp_t;
template [[host_name("kernel_myds_flash_attn_heads_maskless_slab_direct_bf16_dk512_dv512")]]
kernel flash_attn_heads_direct_comp_t kernel_myds_flash_attn_heads_maskless_slab_direct_comp<1u>;
template [[host_name("kernel_myds_flash_attn_heads_maskless_slab_direct_nvfp4_dk512_dv512")]]
kernel flash_attn_heads_direct_comp_t kernel_myds_flash_attn_heads_maskless_slab_direct_comp<2u>;

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];
constant bool FC_flash_attn_ext_vec_shared_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 5)]];
constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];

// Decode FlashAttention for one query row. DS4 uses this in generation to scan
// raw and compressed KV cache chunks, optionally splitting long contexts across
// workgroups and writing partial softmax state for a later reduction.
template<
    typename q4_t,
    typename k4_t,
    typename v4_t,
    typename qk_t,
    typename s_t,
    typename s4_t,
    typename o4_t,
    typename kd4_t,
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,
    short DV,
    short NE = 4,
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG,
    bool HEAD_FIRST = false>
kernel void kernel_flash_attn_ext_vec(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)
#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = HEAD_FIRST ? tgpig[0] : tgpig[1];
    const ushort iq1 = HEAD_FIRST ? tgpig[1] : tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
        /* B 方案：紧凑 slab 下每个 query 的键从自己的 [0,640) 段取。
         * nb_kvq==0 时整句为恒等变换，与旧行为逐位一致。 */
        if (args.nb_kvq != 0) {
            k += (ulong)iq1 * args.nb_kvq;
            v += (ulong)iq1 * args.nb_kvq;
        }
    }

    device const float4 * q4 = (device const float4 *) ((device const char *) q);

    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                sq4[i] = (q4_t) q4[i];
            } else {
                sq4[i] = (q4_t) 0.0f;
            }
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;

        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v = k;
                    mask = k + args.nb11*C*args.ne_12_2*args.ne_12_3 +
                        args.nb21*C*args.ne_12_2*args.ne_12_3;
                } else {
                    v = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                    mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;
                }

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    if (ic + tiisg >= args.ne11) {
                        sm[tiisg] = -MAXHALF;
                    }
                } else {
                    pm = (device const half *) (mask) +
                        iq1*C +
                        (iq2%args.ne32)*(C*args.ne31) +
                        (iq3%args.ne33)*(C*args.ne31*args.ne32);
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            if (simd_max(sm[tiisg]) <= -MAXHALF) {
                continue;
            }

            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        if (NE <= 1) {
                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        }
                        if (NE <= 2) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
                        }
                        if (NE <= 4) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
                        }
                        if (NE <= 8) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
                        }
                        if (NE <= 16) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
                        }

                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                   !FC_flash_attn_ext_vec_has_scap &&
                   !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
                } else {
                    mqk[tx] *= args.scale;

                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = args.logit_softcap*precise::tanh(mqk[tx]);
                    }

                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE*tx + ty];
                    }

                    ss[NE*tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }

                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);
                    }

                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);
                    }

                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);
                    }

                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S*ms + simd_sum(vs);

            if ((DV4/NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[ii*NL] *= ms;
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;
        }

        if (NWG > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}


#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

// Host-visible decode FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,  1, dequantize_f16_t4, half4,  1, dequantize_f16_t4, 512, 512, 1>;


// The same per-head arithmetic, with adjacent threadgroups visiting adjacent heads.
template [[host_name("kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512")]]
kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 512, 512, 1, 1, 32, true>;

#undef FA_TYPES
#undef FA_TYPES_F32

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

// Reduces split-K decode FlashAttention partials. It combines each workgroup's
// output vector and softmax (sum,max) pair into the final attention result.
/* Shared and deliberately noinline so the split-K reduction is compiled once and
 * every caller gets identical codegen. The RoPE-fused sibling in dsv4_rope.metal
 * calls this same body, which is what keeps the fusion bit-exact. */
static __attribute__((noinline)) void ds4_flash_attn_vec_reduce_row(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig,
        ushort tiisg,
        ushort sgitg,
        short  NWG_,
        short  DV_) {
    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV_*NWG_;

    float S = ss[rid*(2*NWG_) + 2*iwg + 0];
    float M = ss[rid*(2*NWG_) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV_/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG_;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG_) {
        const float4 v = simd_sum(htmp4[i*NWG_ + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }
}

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    ds4_flash_attn_vec_reduce_row(args, htmp, dst, tgpig, tiisg, sgitg,
                                  (short)FC_flash_attn_ext_vec_reduce_NWG,
                                  (short)FC_flash_attn_ext_vec_reduce_DV);
}

// M5 decode specialization: time-slice all 32 split-K workgroups through eight
// physical simdgroups, then reduce through the same 32-lane topology without a
// device partial buffer. The host gate fixes the exact F16 512-wide geometry.
static inline void ds4_flash_attn_vec_packed8_reduce_f16_512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  char * shmem,
        uint head,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW = 32;
    constexpr short C = 32;
    constexpr short NSG = 8;
    constexpr short NWG = 32;
    constexpr short DK4 = 128;
    constexpr short DV4 = 128;
    constexpr short SH = 128;

    /* 24,448 dynamic bytes: shared Q, eight score/mask banks, all 32
     * split-local weights and stats, sink scales, and a padded 32x33 F32
     * float4 partial plane. */
    threadgroup half4 *q_shared = (threadgroup half4 *)shmem;
    threadgroup half *score_banks =
        (threadgroup half *)(q_shared + DK4);
    threadgroup volatile float *weights =
        (threadgroup volatile float *)(score_banks + NSG * SH);
    threadgroup volatile float *stats = weights + NWG * C;
    threadgroup volatile float *sink_scale = stats + 2 * NWG;
    threadgroup volatile float4 *partial_plane =
        (threadgroup volatile float4 *)(sink_scale + NWG);

    const short lane = (short)tiisg;
    threadgroup half *bank = score_banks + (short)sgitg * SH;
    threadgroup float *ss = (threadgroup float *)bank;
    threadgroup half *sm = bank + 2 * C;

    device const float4 *q4 =
        (device const float4 *)(q + (uint64_t)head * args.nb02);
    if (sgitg == 0) {
        for (short i = lane; i < DK4; i += NW) {
            q_shared[i] = (half4)q4[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Eight physical simdgroups time-slice the exact 32 legacy split-K
     * workgroups. The official <=1024-key gate gives each virtual split at
     * most one 32-row block, so its value partial can be formed later from
     * these materialized weights without changing online-softmax order. */
    for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
        float S = 0.0f;
        float M = -FLT_MAX / 2;
        float out_scale = 1.0f;
        const int ic_original = (int)iwg * C;

        weights[(uint)iwg * C + (uint)lane] = 0.0f;
        ss[lane] = 0.0f;
        sm[lane] = (half)0.0h;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ic_original < args.ne11) {
            device const char *k_block = k;
            device const char *v_block = v;
            device const half *pm = (device const half *)mask;
            int ic = ic_original;

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k_block = pad;
                const uint64_t k_pad_bytes =
                    args.nb11 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                const uint64_t v_pad_bytes =
                    args.nb21 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v_block = k_block;
                    pm = (device const half *)(k_block +
                                                k_pad_bytes + v_pad_bytes);
                } else {
                    v_block = k_block + k_pad_bytes;
                    pm = (device const half *)(v_block + v_pad_bytes);
                }
                ic = 0;
            }

            sm[lane] = pm[ic + lane];
            if (simd_max(sm[lane]) > -MAXHALF) {
                device const half4 *pk4 =
                    (device const half4 *)(k_block +
                                           (uint64_t)ic * args.nb11);
                threadgroup const half4 *pq4 = q_shared;
                pk4 += lane;
                pq4 += lane;

                float lane_mqk = 0.0f;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    float mqk = 0.0f;
                    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                        mqk += dot((float4)pk4[cc * DK4 + ii * NW],
                                   (float4)pq4[ii * NW]);
                    }
                    mqk = simd_sum(mqk);
                    if (lane == cc) {
                        lane_mqk = mqk;
                    }
                }

                ss[lane] = fma(lane_mqk, args.scale,
                               (float)sm[lane]);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                const float old_m = M;
                const float score = ss[lane];
                M = simd_max(max(M, score));
                const float ms = exp(old_m - M);
                const float vs = exp(score - M);
                S = S * ms + simd_sum(vs);
                ss[lane] = vs;
                simdgroup_barrier(mem_flags::mem_threadgroup);

                weights[(uint)iwg * C + (uint)lane] = ss[lane];
            }

            if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
                const float old_m = M;
                const float sink = lane == 0
                    ? ((device const float *)sinks)[head]
                    : -FLT_MAX / 2;
                M = simd_max(max(M, sink));
                const float ms = exp(old_m - M);
                const float vs = exp(sink - M);
                S = S * ms + simd_sum(vs);
                out_scale = ms;
            }
        } else if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
            const float old_m = M;
            const float sink = lane == 0
                ? ((device const float *)sinks)[head]
                : -FLT_MAX / 2;
            M = simd_max(max(M, sink));
            const float ms = exp(old_m - M);
            const float vs = exp(sink - M);
            S = S * ms + simd_sum(vs);
            out_scale = ms;
        }

        if (lane == 0) {
            stats[2 * (uint)iwg + 0] = S;
            stats[2 * (uint)iwg + 1] = M;
            sink_scale[(uint)iwg] = out_scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Recreate the legacy 32-lane reducer exactly: lane is the virtual split
     * index, including neutral idle splits in their original tree positions. */
    const short split = lane;
    float reduce_S = stats[2 * (uint)split + 0];
    float reduce_M = stats[2 * (uint)split + 1];
    const float reduce_max = simd_max(reduce_M);
    const float reduce_ms = exp(reduce_M - reduce_max);
    reduce_S = simd_sum(reduce_S * reduce_ms);
    const float reduce_inv =
        reduce_S == 0.0f ? 0.0f : 1.0f / reduce_S;

    device float4 *dst4 =
        (device float4 *)(dst +
                          (uint64_t)head * 512u * sizeof(float));

    /* Form one 32-float4 output quadrant at a time. During production each
     * physical simdgroup time-slices four virtual splits while SIMD lanes are
     * contiguous output columns, exactly matching the legacy V loads and
     * cc-major accumulation. A padded 33-column plane avoids a 32-way TG-bank
     * conflict when the reducer transposes lanes back to virtual splits. */
    for (short quadrant = 0; quadrant < 4; ++quadrant) {
        for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
            float4 lo = float4(0.0f);
            const int ic_original = (int)iwg * C;
            if (ic_original < args.ne11) {
                device const char *v_block = v;
                int ic = ic_original;
                if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                    device const char *k_block = pad;
                    const uint64_t k_pad_bytes =
                        args.nb11 * (uint64_t)C *
                        (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                    if (FC_flash_attn_ext_vec_shared_kvpad) {
                        v_block = k_block;
                    } else {
                        v_block = k_block + k_pad_bytes;
                    }
                    ic = 0;
                }

                device const half4 *pv4 =
                    (device const half4 *)(v_block +
                                           (uint64_t)ic * args.nb21);
                threadgroup volatile float *split_weights =
                    weights + (uint)iwg * C;
                const short oc = quadrant * NW + lane;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    lo += float4(pv4[cc * DV4 + oc]) *
                          float4(split_weights[cc]);
                }

                float4 acc = float4(0.0f);
                acc += lo;
                if (iwg == 0) {
                    acc *= sink_scale[0];
                }
                lo = acc;
            }
            partial_plane[(uint)iwg * 33u + (uint)lane] = lo;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* lane is now the legacy split index and each physical simdgroup
         * reduces four output columns through the identical simd_sum tree. */
        for (short out_lane = (short)sgitg; out_lane < NW; out_lane += NSG) {
            const float4 materialized =
                (float4)partial_plane[(uint)lane * 33u + (uint)out_lane];
            const float4 reduced = simd_sum(materialized * reduce_ms);
            if (lane == 0) {
                dst4[quadrant * NW + out_lane] = reduced * reduce_inv;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Host-visible fused packed-reduce decode FlashAttention kernel for 512-wide F16 K/V rows.
// Computes all 32 split-K tiles in threadgroup memory and reduces directly to destination heads buffer,
// eliminating device partial buffer traffic and the separate reduce kernel dispatch.
kernel void kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        threadgroup  char * shmem                     [[threadgroup(0)]],
        uint head                                     [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    if (!FC_flash_attn_ext_vec_has_mask ||
        !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias ||
         FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 ||
         FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 ||
         FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
         args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
         args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
         args.ne11 <= 0 || args.ne11 > 1024 || head >= (uint)args.ne02 ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    ds4_flash_attn_vec_packed8_reduce_f16_512(
        args, q, k, v, mask, sinks, pad, dst, shmem,
        head, tiisg, sgitg);
}

/* Opt-in physical-layout candidates. The original packed8 helper and entry
 * point above remain untouched. Only score-bank count and virtual-split
 * scheduling change; every split and the 32-lane reducer keep their arithmetic.
 * Host dynamic allocation is 22,400 + 256*NSG bytes (4:23,424;16:26,496). */
template <short NSG>
static inline void ds4_flash_attn_vec_packed_candidate_reduce_f16_512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  char * shmem,
        uint head,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW = 32;
    constexpr short C = 32;
    static_assert(NSG == 4 || NSG == 16, "packed candidate physical NSG must be 4 or 16");
    constexpr short NWG = 32;
    constexpr short DK4 = 128;
    constexpr short DV4 = 128;
    constexpr short SH = 128;

    /* 22,400 + 256*NSG dynamic bytes: shared Q, physical score/mask banks, all 32
     * split-local weights and stats, sink scales, and a padded 32x33 F32
     * float4 partial plane. */
    threadgroup half4 *q_shared = (threadgroup half4 *)shmem;
    threadgroup half *score_banks =
        (threadgroup half *)(q_shared + DK4);
    threadgroup volatile float *weights =
        (threadgroup volatile float *)(score_banks + NSG * SH);
    threadgroup volatile float *stats = weights + NWG * C;
    threadgroup volatile float *sink_scale = stats + 2 * NWG;
    threadgroup volatile float4 *partial_plane =
        (threadgroup volatile float4 *)(sink_scale + NWG);

    const short lane = (short)tiisg;
    threadgroup half *bank = score_banks + (short)sgitg * SH;
    threadgroup float *ss = (threadgroup float *)bank;
    threadgroup half *sm = bank + 2 * C;

    device const float4 *q4 =
        (device const float4 *)(q + (uint64_t)head * args.nb02);
    if (sgitg == 0) {
        for (short i = lane; i < DK4; i += NW) {
            q_shared[i] = (half4)q4[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Physical simdgroups time-slice the exact 32 legacy split-K
     * workgroups. The official <=1024-key gate gives each virtual split at
     * most one 32-row block, so its value partial can be formed later from
     * these materialized weights without changing online-softmax order. */
    for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
        float S = 0.0f;
        float M = -FLT_MAX / 2;
        float out_scale = 1.0f;
        const int ic_original = (int)iwg * C;

        weights[(uint)iwg * C + (uint)lane] = 0.0f;
        ss[lane] = 0.0f;
        sm[lane] = (half)0.0h;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ic_original < args.ne11) {
            device const char *k_block = k;
            device const char *v_block = v;
            device const half *pm = (device const half *)mask;
            int ic = ic_original;

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k_block = pad;
                const uint64_t k_pad_bytes =
                    args.nb11 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                const uint64_t v_pad_bytes =
                    args.nb21 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v_block = k_block;
                    pm = (device const half *)(k_block +
                                                k_pad_bytes + v_pad_bytes);
                } else {
                    v_block = k_block + k_pad_bytes;
                    pm = (device const half *)(v_block + v_pad_bytes);
                }
                ic = 0;
            }

            sm[lane] = pm[ic + lane];
            if (simd_max(sm[lane]) > -MAXHALF) {
                device const half4 *pk4 =
                    (device const half4 *)(k_block +
                                           (uint64_t)ic * args.nb11);
                threadgroup const half4 *pq4 = q_shared;
                pk4 += lane;
                pq4 += lane;

                float lane_mqk = 0.0f;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    float mqk = 0.0f;
                    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                        mqk += dot((float4)pk4[cc * DK4 + ii * NW],
                                   (float4)pq4[ii * NW]);
                    }
                    mqk = simd_sum(mqk);
                    if (lane == cc) {
                        lane_mqk = mqk;
                    }
                }

                ss[lane] = fma(lane_mqk, args.scale,
                               (float)sm[lane]);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                const float old_m = M;
                const float score = ss[lane];
                M = simd_max(max(M, score));
                const float ms = exp(old_m - M);
                const float vs = exp(score - M);
                S = S * ms + simd_sum(vs);
                ss[lane] = vs;
                simdgroup_barrier(mem_flags::mem_threadgroup);

                weights[(uint)iwg * C + (uint)lane] = ss[lane];
            }

            if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
                const float old_m = M;
                const float sink = lane == 0
                    ? ((device const float *)sinks)[head]
                    : -FLT_MAX / 2;
                M = simd_max(max(M, sink));
                const float ms = exp(old_m - M);
                const float vs = exp(sink - M);
                S = S * ms + simd_sum(vs);
                out_scale = ms;
            }
        } else if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
            const float old_m = M;
            const float sink = lane == 0
                ? ((device const float *)sinks)[head]
                : -FLT_MAX / 2;
            M = simd_max(max(M, sink));
            const float ms = exp(old_m - M);
            const float vs = exp(sink - M);
            S = S * ms + simd_sum(vs);
            out_scale = ms;
        }

        if (lane == 0) {
            stats[2 * (uint)iwg + 0] = S;
            stats[2 * (uint)iwg + 1] = M;
            sink_scale[(uint)iwg] = out_scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Recreate the legacy 32-lane reducer exactly: lane is the virtual split
     * index, including neutral idle splits in their original tree positions. */
    const short split = lane;
    float reduce_S = stats[2 * (uint)split + 0];
    float reduce_M = stats[2 * (uint)split + 1];
    const float reduce_max = simd_max(reduce_M);
    const float reduce_ms = exp(reduce_M - reduce_max);
    reduce_S = simd_sum(reduce_S * reduce_ms);
    const float reduce_inv =
        reduce_S == 0.0f ? 0.0f : 1.0f / reduce_S;

    device float4 *dst4 =
        (device float4 *)(dst +
                          (uint64_t)head * 512u * sizeof(float));

    /* Form one 32-float4 output quadrant at a time. During production each
     * physical simdgroup time-slices 32/NSG virtual splits while SIMD lanes are
     * contiguous output columns, exactly matching the legacy V loads and
     * cc-major accumulation. A padded 33-column plane avoids a 32-way TG-bank
     * conflict when the reducer transposes lanes back to virtual splits. */
    for (short quadrant = 0; quadrant < 4; ++quadrant) {
        for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
            float4 lo = float4(0.0f);
            const int ic_original = (int)iwg * C;
            if (ic_original < args.ne11) {
                device const char *v_block = v;
                int ic = ic_original;
                if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                    device const char *k_block = pad;
                    const uint64_t k_pad_bytes =
                        args.nb11 * (uint64_t)C *
                        (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                    if (FC_flash_attn_ext_vec_shared_kvpad) {
                        v_block = k_block;
                    } else {
                        v_block = k_block + k_pad_bytes;
                    }
                    ic = 0;
                }

                device const half4 *pv4 =
                    (device const half4 *)(v_block +
                                           (uint64_t)ic * args.nb21);
                threadgroup volatile float *split_weights =
                    weights + (uint)iwg * C;
                const short oc = quadrant * NW + lane;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    lo += float4(pv4[cc * DV4 + oc]) *
                          float4(split_weights[cc]);
                }

                float4 acc = float4(0.0f);
                acc += lo;
                if (iwg == 0) {
                    acc *= sink_scale[0];
                }
                lo = acc;
            }
            partial_plane[(uint)iwg * 33u + (uint)lane] = lo;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* lane is now the legacy split index and each physical simdgroup
         * reduces 32/NSG output columns through the identical simd_sum tree. */
        for (short out_lane = (short)sgitg; out_lane < NW; out_lane += NSG) {
            const float4 materialized =
                (float4)partial_plane[(uint)lane * 33u + (uint)out_lane];
            const float4 reduced = simd_sum(materialized * reduce_ms);
            if (lane == 0) {
                dst4[quadrant * NW + out_lane] = reduced * reduce_inv;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_myds_flash_attn_vec_packed4_reduce_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        threadgroup  char * shmem                     [[threadgroup(0)]],
        uint head                                     [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    if (!FC_flash_attn_ext_vec_has_mask ||
        !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias ||
         FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 ||
         FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 ||
         FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
         args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
         args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
         args.ne11 <= 0 || args.ne11 > 1024 || head >= (uint)args.ne02 ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    ds4_flash_attn_vec_packed_candidate_reduce_f16_512<4>(
        args, q, k, v, mask, sinks, pad, dst, shmem,
        head, tiisg, sgitg);
}

kernel void kernel_myds_flash_attn_vec_packed16_reduce_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        threadgroup  char * shmem                     [[threadgroup(0)]],
        uint head                                     [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    if (!FC_flash_attn_ext_vec_has_mask ||
        !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias ||
         FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 ||
         FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 ||
         FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
         args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
         args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
         args.ne11 <= 0 || args.ne11 > 1024 || head >= (uint)args.ne02 ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    ds4_flash_attn_vec_packed_candidate_reduce_f16_512<16>(
        args, q, k, v, mask, sinks, pad, dst, shmem,
        head, tiisg, sgitg);
}

/* Explicit register candidate: exactly the independently measured probe arithmetic.
 * Lane-local mask/score/value no longer use score banks or SIMD barriers.
 * Original helpers/entries above remain byte-for-byte intact. Four TG barrier
 * sites, all 32 logical split partials, and their final reduction are retained.
 * Host must use 256 threads and exactly 22,400 dynamic shared bytes. */
static inline void ds4_flash_attn_vec_packed_register_reduce_f16_512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  char * shmem,
        uint head,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW = 32;
    constexpr short C = 32;
    constexpr short NSG = 8;
    constexpr short NWG = 32;
    constexpr short DK4 = 128;
    constexpr short DV4 = 128;
    constexpr short SH = 128;

    /* 22,400 dynamic bytes: shared Q, lane-local score/mask registers, all 32
     * split-local weights and stats, sink scales, and a padded 32x33 F32
     * float4 partial plane. */
    threadgroup half4 *q_shared = (threadgroup half4 *)shmem;
    threadgroup volatile float *weights =
        (threadgroup volatile float *)(q_shared + DK4);
    threadgroup volatile float *stats = weights + NWG * C;
    threadgroup volatile float *sink_scale = stats + 2 * NWG;
    threadgroup volatile float4 *partial_plane =
        (threadgroup volatile float4 *)(sink_scale + NWG);

    const short lane = (short)tiisg;

    device const float4 *q4 =
        (device const float4 *)(q + (uint64_t)head * args.nb02);
    if (sgitg == 0) {
        for (short i = lane; i < DK4; i += NW) {
            q_shared[i] = (half4)q4[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Eight physical simdgroups time-slice the exact 32 legacy split-K
     * workgroups. The official <=1024-key gate gives each virtual split at
     * most one 32-row block, so its value partial can be formed later from
     * these materialized weights without changing online-softmax order. */
    for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
        float S = 0.0f;
        float M = -FLT_MAX / 2;
        float out_scale = 1.0f;
        const int ic_original = (int)iwg * C;

        weights[(uint)iwg * C + (uint)lane] = 0.0f;

        if (ic_original < args.ne11) {
            device const char *k_block = k;
            device const char *v_block = v;
            device const half *pm = (device const half *)mask;
            int ic = ic_original;

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k_block = pad;
                const uint64_t k_pad_bytes =
                    args.nb11 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                const uint64_t v_pad_bytes =
                    args.nb21 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v_block = k_block;
                    pm = (device const half *)(k_block +
                                                k_pad_bytes + v_pad_bytes);
                } else {
                    v_block = k_block + k_pad_bytes;
                    pm = (device const half *)(v_block + v_pad_bytes);
                }
                ic = 0;
            }

            const half lane_mask = pm[ic + lane];
            if (simd_max(lane_mask) > -MAXHALF) {
                device const half4 *pk4 =
                    (device const half4 *)(k_block +
                                           (uint64_t)ic * args.nb11);
                threadgroup const half4 *pq4 = q_shared;
                pk4 += lane;
                pq4 += lane;

                float lane_mqk = 0.0f;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    float mqk = 0.0f;
                    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                        mqk += dot((float4)pk4[cc * DK4 + ii * NW],
                                   (float4)pq4[ii * NW]);
                    }
                    mqk = simd_sum(mqk);
                    if (lane == cc) {
                        lane_mqk = mqk;
                    }
                }

                const float lane_score = fma(lane_mqk, args.scale,
                                             (float)lane_mask);

                const float old_m = M;
                const float score = lane_score;
                M = simd_max(max(M, score));
                const float ms = exp(old_m - M);
                const float vs = exp(score - M);
                S = S * ms + simd_sum(vs);

                weights[(uint)iwg * C + (uint)lane] = vs;
            }

            if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
                const float old_m = M;
                const float sink = lane == 0
                    ? ((device const float *)sinks)[head]
                    : -FLT_MAX / 2;
                M = simd_max(max(M, sink));
                const float ms = exp(old_m - M);
                const float vs = exp(sink - M);
                S = S * ms + simd_sum(vs);
                out_scale = ms;
            }
        } else if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
            const float old_m = M;
            const float sink = lane == 0
                ? ((device const float *)sinks)[head]
                : -FLT_MAX / 2;
            M = simd_max(max(M, sink));
            const float ms = exp(old_m - M);
            const float vs = exp(sink - M);
            S = S * ms + simd_sum(vs);
            out_scale = ms;
        }

        if (lane == 0) {
            stats[2 * (uint)iwg + 0] = S;
            stats[2 * (uint)iwg + 1] = M;
            sink_scale[(uint)iwg] = out_scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Recreate the legacy 32-lane reducer exactly: lane is the virtual split
     * index, including neutral idle splits in their original tree positions. */
    const short split = lane;
    float reduce_S = stats[2 * (uint)split + 0];
    float reduce_M = stats[2 * (uint)split + 1];
    const float reduce_max = simd_max(reduce_M);
    const float reduce_ms = exp(reduce_M - reduce_max);
    reduce_S = simd_sum(reduce_S * reduce_ms);
    const float reduce_inv =
        reduce_S == 0.0f ? 0.0f : 1.0f / reduce_S;

    device float4 *dst4 =
        (device float4 *)(dst +
                          (uint64_t)head * 512u * sizeof(float));

    /* Form one 32-float4 output quadrant at a time. During production each
     * physical simdgroup time-slices four virtual splits while SIMD lanes are
     * contiguous output columns, exactly matching the legacy V loads and
     * cc-major accumulation. A padded 33-column plane avoids a 32-way TG-bank
     * conflict when the reducer transposes lanes back to virtual splits. */
    for (short quadrant = 0; quadrant < 4; ++quadrant) {
        for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
            float4 lo = float4(0.0f);
            const int ic_original = (int)iwg * C;
            if (ic_original < args.ne11) {
                device const char *v_block = v;
                int ic = ic_original;
                if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                    device const char *k_block = pad;
                    const uint64_t k_pad_bytes =
                        args.nb11 * (uint64_t)C *
                        (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                    if (FC_flash_attn_ext_vec_shared_kvpad) {
                        v_block = k_block;
                    } else {
                        v_block = k_block + k_pad_bytes;
                    }
                    ic = 0;
                }

                device const half4 *pv4 =
                    (device const half4 *)(v_block +
                                           (uint64_t)ic * args.nb21);
                threadgroup volatile float *split_weights =
                    weights + (uint)iwg * C;
                const short oc = quadrant * NW + lane;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    lo += float4(pv4[cc * DV4 + oc]) *
                          float4(split_weights[cc]);
                }

                float4 acc = float4(0.0f);
                acc += lo;
                if (iwg == 0) {
                    acc *= sink_scale[0];
                }
                lo = acc;
            }
            partial_plane[(uint)iwg * 33u + (uint)lane] = lo;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* lane is now the legacy split index and each physical simdgroup
         * reduces four output columns through the identical simd_sum tree. */
        for (short out_lane = (short)sgitg; out_lane < NW; out_lane += NSG) {
            const float4 materialized =
                (float4)partial_plane[(uint)lane * 33u + (uint)out_lane];
            const float4 reduced = simd_sum(materialized * reduce_ms);
            if (lane == 0) {
                dst4[quadrant * NW + out_lane] = reduced * reduce_inv;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_myds_flash_attn_vec_packed8_register_reduce_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        threadgroup  char * shmem                     [[threadgroup(0)]],
        uint head                                     [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    if (!FC_flash_attn_ext_vec_has_mask ||
        !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias ||
         FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 ||
         FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 ||
         FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
         args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
         args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
         args.ne11 <= 0 || args.ne11 > 1024 || head >= (uint)args.ne02 ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    ds4_flash_attn_vec_packed_register_reduce_f16_512(
        args, q, k, v, mask, sinks, pad, dst, shmem,
        head, tiisg, sgitg);
}

/* Ordinary single-query opt-in: the complete original register reducer above
 * remains unchanged. This clone changes only its final float4 store, after
 * the original 32-split simd_sum and multiply. Same 22,400 dynamic bytes.
 * Preserve kernel_dsv41_rope_round's two BF16 boundaries and runtime inverse
 * sign selection. Exact parity remains a private component acceptance gate. */
struct myds_attention_rope_fuse_args {
    ds4_metal_args_flash_attn_ext_vec attention;
    uint pos, inverse;
    float frequencies[32];
};
static_assert(sizeof(myds_attention_rope_fuse_args)==336, "attention+rope ABI");
static inline float myds_attention_rope_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}
static inline float4 myds_attention_rope_store(float4 value, uint oc,
        constant myds_attention_rope_fuse_args & rope) {
    float4 result = float4(myds_attention_rope_bf16(value.x),
        myds_attention_rope_bf16(value.y), myds_attention_rope_bf16(value.z),
        myds_attention_rope_bf16(value.w));
    if (oc >= 112u) {
        const uint pair = (oc-112u)*2u;
        const float theta0 = float(rope.pos)*rope.frequencies[pair];
        const float c0 = precise::cos(theta0);
        const float s0 = rope.inverse ? -precise::sin(theta0) : precise::sin(theta0);
        const float theta1 = float(rope.pos)*rope.frequencies[pair+1u];
        const float c1 = precise::cos(theta1);
        const float s1 = rope.inverse ? -precise::sin(theta1) : precise::sin(theta1);
        const float re0=result.x, im0=result.y, re1=result.z, im1=result.w;
        result.x = myds_attention_rope_bf16(re0*c0-im0*s0);
        /* Pin the original scalar AIR's imaginary operand order (im*c,re*s).
         * This contraction direction is a parity candidate, not M2 ISA proof. */
        result.y = myds_attention_rope_bf16(precise::fma(im0,c0,re0*s0));
        result.z = myds_attention_rope_bf16(re1*c1-im1*s1);
        result.w = myds_attention_rope_bf16(precise::fma(im1,c1,re1*s1));
    }
    return result;
}
static inline void myds_attention_register_rope_fuse_impl(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        constant myds_attention_rope_fuse_args & rope,
        threadgroup  char * shmem,
        uint head,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW = 32;
    constexpr short C = 32;
    constexpr short NSG = 14;
    constexpr short NWG = 32;
    constexpr short DK4 = 128;
    constexpr short DV4 = 128;
    constexpr short SH = 128;

    /* 22,400 dynamic bytes: shared Q, lane-local score/mask registers, all 32
     * split-local weights and stats, sink scales, and a padded 32x33 F32
     * float4 partial plane. */
    threadgroup half4 *q_shared = (threadgroup half4 *)shmem;
    threadgroup volatile float *weights =
        (threadgroup volatile float *)(q_shared + DK4);
    threadgroup volatile float *stats = weights + NWG * C;
    threadgroup volatile float *sink_scale = stats + 2 * NWG;
    threadgroup volatile float4 *partial_plane =
        (threadgroup volatile float4 *)(sink_scale + NWG);

    const short lane = (short)tiisg;

    device const float4 *q4 =
        (device const float4 *)(q + (uint64_t)head * args.nb02);
    if (sgitg == 0) {
        for (short i = lane; i < DK4; i += NW) {
            q_shared[i] = (half4)q4[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Eight physical simdgroups time-slice the exact 32 legacy split-K
     * workgroups. The official <=1024-key gate gives each virtual split at
     * most one 32-row block, so its value partial can be formed later from
     * these materialized weights without changing online-softmax order. */
    for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
        float S = 0.0f;
        float M = -FLT_MAX / 2;
        float out_scale = 1.0f;
        const int ic_original = (int)iwg * C;

        weights[(uint)iwg * C + (uint)lane] = 0.0f;

        if (ic_original < args.ne11) {
            device const char *k_block = k;
            device const char *v_block = v;
            device const half *pm = (device const half *)mask;
            int ic = ic_original;

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k_block = pad;
                const uint64_t k_pad_bytes =
                    args.nb11 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                const uint64_t v_pad_bytes =
                    args.nb21 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v_block = k_block;
                    pm = (device const half *)(k_block +
                                                k_pad_bytes + v_pad_bytes);
                } else {
                    v_block = k_block + k_pad_bytes;
                    pm = (device const half *)(v_block + v_pad_bytes);
                }
                ic = 0;
            }

            const half lane_mask = pm[ic + lane];
            if (simd_max(lane_mask) > -MAXHALF) {
                device const half4 *pk4 =
                    (device const half4 *)(k_block +
                                           (uint64_t)ic * args.nb11);
                threadgroup const half4 *pq4 = q_shared;
                pk4 += lane;
                pq4 += lane;

                float lane_mqk = 0.0f;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    float mqk = 0.0f;
                    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                        mqk += dot((float4)pk4[cc * DK4 + ii * NW],
                                   (float4)pq4[ii * NW]);
                    }
                    mqk = simd_sum(mqk);
                    if (lane == cc) {
                        lane_mqk = mqk;
                    }
                }

                const float lane_score = fma(lane_mqk, args.scale,
                                             (float)lane_mask);

                const float old_m = M;
                const float score = lane_score;
                M = simd_max(max(M, score));
                const float ms = exp(old_m - M);
                const float vs = exp(score - M);
                S = S * ms + simd_sum(vs);

                weights[(uint)iwg * C + (uint)lane] = vs;
            }

            if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
                const float old_m = M;
                const float sink = lane == 0
                    ? ((device const float *)sinks)[head]
                    : -FLT_MAX / 2;
                M = simd_max(max(M, sink));
                const float ms = exp(old_m - M);
                const float vs = exp(sink - M);
                S = S * ms + simd_sum(vs);
                out_scale = ms;
            }
        } else if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
            const float old_m = M;
            const float sink = lane == 0
                ? ((device const float *)sinks)[head]
                : -FLT_MAX / 2;
            M = simd_max(max(M, sink));
            const float ms = exp(old_m - M);
            const float vs = exp(sink - M);
            S = S * ms + simd_sum(vs);
            out_scale = ms;
        }

        if (lane == 0) {
            stats[2 * (uint)iwg + 0] = S;
            stats[2 * (uint)iwg + 1] = M;
            sink_scale[(uint)iwg] = out_scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Recreate the legacy 32-lane reducer exactly: lane is the virtual split
     * index, including neutral idle splits in their original tree positions. */
    const short split = lane;
    float reduce_S = stats[2 * (uint)split + 0];
    float reduce_M = stats[2 * (uint)split + 1];
    const float reduce_max = simd_max(reduce_M);
    const float reduce_ms = exp(reduce_M - reduce_max);
    reduce_S = simd_sum(reduce_S * reduce_ms);
    const float reduce_inv =
        reduce_S == 0.0f ? 0.0f : 1.0f / reduce_S;

    device float4 *dst4 =
        (device float4 *)(dst +
                          (uint64_t)head * 512u * sizeof(float));

    /* Form one 32-float4 output quadrant at a time. During production each
     * physical simdgroup time-slices four virtual splits while SIMD lanes are
     * contiguous output columns, exactly matching the legacy V loads and
     * cc-major accumulation. A padded 33-column plane avoids a 32-way TG-bank
     * conflict when the reducer transposes lanes back to virtual splits. */
    for (short quadrant = 0; quadrant < 4; ++quadrant) {
        for (short iwg = (short)sgitg; iwg < NWG; iwg += NSG) {
            float4 lo = float4(0.0f);
            const int ic_original = (int)iwg * C;
            if (ic_original < args.ne11) {
                device const char *v_block = v;
                int ic = ic_original;
                if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                    device const char *k_block = pad;
                    const uint64_t k_pad_bytes =
                        args.nb11 * (uint64_t)C *
                        (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                    if (FC_flash_attn_ext_vec_shared_kvpad) {
                        v_block = k_block;
                    } else {
                        v_block = k_block + k_pad_bytes;
                    }
                    ic = 0;
                }

                device const half4 *pv4 =
                    (device const half4 *)(v_block +
                                           (uint64_t)ic * args.nb21);
                threadgroup volatile float *split_weights =
                    weights + (uint)iwg * C;
                const short oc = quadrant * NW + lane;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    lo += float4(pv4[cc * DV4 + oc]) *
                          float4(split_weights[cc]);
                }

                float4 acc = float4(0.0f);
                acc += lo;
                if (iwg == 0) {
                    acc *= sink_scale[0];
                }
                lo = acc;
            }
            partial_plane[(uint)iwg * 33u + (uint)lane] = lo;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* lane is now the legacy split index and each physical simdgroup
         * reduces four output columns through the identical simd_sum tree. */
        for (short out_lane = (short)sgitg; out_lane < NW; out_lane += NSG) {
            const float4 materialized =
                (float4)partial_plane[(uint)lane * 33u + (uint)out_lane];
            const float4 reduced = simd_sum(materialized * reduce_ms);
            if (lane == 0) {
                const uint oc = (uint)quadrant * (uint)NW + (uint)out_lane;
                const float4 value = reduced * reduce_inv;
                dst4[oc] = myds_attention_rope_store(value, oc, rope);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_myds_attention_register_rope_fuse_f16_512(
        constant myds_attention_rope_fuse_args & fused [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        threadgroup  char * shmem                     [[threadgroup(0)]],
        uint head                                     [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    constant ds4_metal_args_flash_attn_ext_vec & args = fused.attention;
    if (fused.pos >= 1048576u || !FC_flash_attn_ext_vec_has_mask ||
        !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias ||
         FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 ||
         FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 ||
         FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
         args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
         args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
         args.ne11 <= 0 || args.ne11 > 1024 || head >= (uint)args.ne02 ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    myds_attention_register_rope_fuse_impl(
        args, q, k, v, mask, sinks, pad, dst, fused, shmem,
        head, tiisg, sgitg);
}

/* ==========================================================================
 * D1: Flash-Decoding Multi-Threadgroup Attention (2-Pass Split-K)
 * Pass 1: Grid (64, 4, 1) = 256 threadgroups across 76 M2 Ultra GPU cores.
 *         Each TG has 8 simdgroups (256 threads). Each simdgroup computes
 *         exactly 1 of the 32 virtual splits in parallel (zero time-slicing).
 * Pass 2: Grid (64, 1, 1), 4 simdgroups (128 threads) reduce across 32 splits
 *         with exact 0-ULP arithmetic parity and fuse RoPE.
 * ========================================================================== */

kernel void kernel_myds_attention_flash_dec_split_f16_512(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * partial                   [[buffer(7)]],
        threadgroup  char * shmem                     [[threadgroup(0)]],
        uint2 tgpig                                   [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    if (!FC_flash_attn_ext_vec_has_mask ||
        !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias ||
         FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 ||
         FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 ||
         FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
         args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
         args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
         args.ne11 <= 0 || args.ne11 > 1024 || tgpig.x >= (uint)args.ne02 ||
         tgpig.y >= 4u ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    constexpr short NW = 32;
    constexpr short C = 32;
    constexpr short DK4 = 128;
    constexpr short DV4 = 128;
    constexpr short NWG = 32;

    const uint head = tgpig.x;
    const uint split_group = tgpig.y;
    const uint iwg = split_group * 8u + (uint)sgitg;
    const short lane = (short)tiisg;

    threadgroup half4 *q_shared = (threadgroup half4 *)shmem;
    threadgroup volatile float *weights = (threadgroup volatile float *)(q_shared + DK4);

    device const float4 *q4 =
        (device const float4 *)(q + (uint64_t)head * args.nb02);
    if (sgitg == 0) {
        for (short i = lane; i < DK4; i += NW) {
            q_shared[i] = (half4)q4[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;
    float out_scale = 1.0f;
    const int ic_original = (int)iwg * C;

    weights[(uint)sgitg * (uint)C + (uint)lane] = 0.0f;

    if (ic_original < args.ne11) {
        device const char *k_block = k;
        device const char *v_block = v;
        device const half *pm = (device const half *)mask;
        int ic = ic_original;

        if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
            k_block = pad;
            const uint64_t k_pad_bytes =
                args.nb11 * (uint64_t)C *
                (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
            const uint64_t v_pad_bytes =
                args.nb21 * (uint64_t)C *
                (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
            if (FC_flash_attn_ext_vec_shared_kvpad) {
                v_block = k_block;
                pm = (device const half *)(k_block +
                                            k_pad_bytes + v_pad_bytes);
            } else {
                v_block = k_block + k_pad_bytes;
                pm = (device const half *)(v_block + v_pad_bytes);
            }
            ic = 0;
        }

        const half lane_mask = pm[ic + lane];
        if (simd_max(lane_mask) > -MAXHALF) {
            device const half4 *pk4 =
                (device const half4 *)(k_block +
                                       (uint64_t)ic * args.nb11);
            threadgroup const half4 *pq4 = q_shared;
            pk4 += lane;
            pq4 += lane;

            float lane_mqk = 0.0f;
            FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                float mqk = 0.0f;
                FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                    mqk += dot((float4)pk4[cc * DK4 + ii * NW],
                               (float4)pq4[ii * NW]);
                }
                mqk = simd_sum(mqk);
                if (lane == cc) {
                    lane_mqk = mqk;
                }
            }

            const float lane_score = fma(lane_mqk, args.scale,
                                         (float)lane_mask);

            const float old_m = M;
            const float score = lane_score;
            M = simd_max(max(M, score));
            const float ms = exp(old_m - M);
            const float vs = exp(score - M);
            S = S * ms + simd_sum(vs);

            weights[(uint)sgitg * (uint)C + (uint)lane] = vs;
        }

        if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
            const float old_m = M;
            const float sink = lane == 0
                ? ((device const float *)sinks)[head]
                : -FLT_MAX / 2;
            M = simd_max(max(M, sink));
            const float ms = exp(old_m - M);
            const float vs = exp(sink - M);
            S = S * ms + simd_sum(vs);
            out_scale = ms;
        }
    } else if (FC_flash_attn_ext_vec_has_sinks && iwg == 0) {
        const float old_m = M;
        const float sink = lane == 0
            ? ((device const float *)sinks)[head]
            : -FLT_MAX / 2;
        M = simd_max(max(M, sink));
        const float ms = exp(old_m - M);
        const float vs = exp(sink - M);
        S = S * ms + simd_sum(vs);
        out_scale = ms;
    }

    if (lane == 0) {
        device float *stats = (device float *)partial + (uint64_t)64u * (uint64_t)(512u * (uint32_t)NWG);
        stats[(uint64_t)head * (uint64_t)(2 * NWG) + 2u * (uint64_t)iwg + 0u] = S;
        stats[(uint64_t)head * (uint64_t)(2 * NWG) + 2u * (uint64_t)iwg + 1u] = M;
    }

    device float4 *htmp4 = (device float4 *)partial + (uint64_t)head * (uint64_t)(DV4 * NWG);
    threadgroup volatile float *split_weights = weights + (uint)sgitg * (uint)C;

    for (short quadrant = 0; quadrant < 4; ++quadrant) {
        const short oc = quadrant * NW + lane;
        float4 lo = float4(0.0f);
        if (ic_original < args.ne11) {
            device const char *v_block = v;
            int ic = ic_original;
            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                device const char *k_block = pad;
                const uint64_t k_pad_bytes =
                    args.nb11 * (uint64_t)C *
                    (uint64_t)args.ne_12_2 * (uint64_t)args.ne_12_3;
                if (FC_flash_attn_ext_vec_shared_kvpad) {
                    v_block = k_block;
                } else {
                    v_block = k_block + k_pad_bytes;
                }
                ic = 0;
            }

            device const half4 *pv4 =
                (device const half4 *)(v_block + (uint64_t)ic * args.nb21);
            FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                lo += float4(pv4[cc * DV4 + oc]) * float4(split_weights[cc]);
            }
            if (iwg == 0) {
                lo *= out_scale;
            }
        }
        htmp4[(uint)oc * (uint)NWG + (uint)iwg] = lo;
    }
}

kernel void kernel_myds_attention_flash_dec_reduce_rope_f16_512(
        constant myds_attention_rope_fuse_args & fused [[buffer(0)]],
        device const char * partial                   [[buffer(1)]],
        device       char * dst                       [[buffer(2)]],
        uint head                                     [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = 32;
    constexpr short NWG = 32;
    constexpr short DV4 = 128;

    if (fused.pos >= 1048576u || head >= 64u) {
        return;
    }

    const uint64_t rid = (uint64_t)head;
    const short iwg = (short)tiisg;

    device const float * ss = (device const float *)partial + (uint64_t)64u * (uint64_t)(512u * (uint32_t)NWG);

    float S = ss[rid * (uint64_t)(2 * NWG) + 2u * (uint64_t)iwg + 0u];
    float M = ss[rid * (uint64_t)(2 * NWG) + 2u * (uint64_t)iwg + 1u];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S * ms);
    const float reduce_inv = S == 0.0f ? 0.0f : 1.0f / S;

    device const float4 * htmp4 = (device const float4 *)partial + rid * (uint64_t)(DV4 * NWG);
    device       float4 * dst4  = (device       float4 *)(dst + rid * 512u * sizeof(float));

    const short quadrant = (short)sgitg;
    for (short out_lane = 0; out_lane < NW; ++out_lane) {
        const short oc = quadrant * NW + out_lane;
        const float4 materialized = htmp4[(uint)oc * (uint)NWG + (uint)iwg];
        const float4 reduced = simd_sum(materialized * ms);

        if (iwg == 0) {
            const float4 value = reduced * reduce_inv;
            dst4[oc] = myds_attention_rope_store(value, (uint)oc, fused);
        }
    }
}

kernel void kernel_myds_attention_flash_dec_reduce_f16_512(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * partial                   [[buffer(1)]],
        device       char * dst                       [[buffer(2)]],
        uint head                                     [[threadgroup_position_in_grid]],
        ushort tiisg                                  [[thread_index_in_simdgroup]],
        ushort sgitg                                  [[simdgroup_index_in_threadgroup]]) {
    (void)args;
    constexpr short NW = 32;
    constexpr short NWG = 32;
    constexpr short DV4 = 128;

    if (head >= 64u) {
        return;
    }

    const uint64_t rid = (uint64_t)head;
    const short iwg = (short)tiisg;

    device const float * ss = (device const float *)partial + (uint64_t)64u * (uint64_t)(512u * (uint32_t)NWG);

    float S = ss[rid * (uint64_t)(2 * NWG) + 2u * (uint64_t)iwg + 0u];
    float M = ss[rid * (uint64_t)(2 * NWG) + 2u * (uint64_t)iwg + 1u];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S * ms);
    const float reduce_inv = S == 0.0f ? 0.0f : 1.0f / S;

    device const float4 * htmp4 = (device const float4 *)partial + rid * (uint64_t)(DV4 * NWG);
    device       float4 * dst4  = (device       float4 *)(dst + rid * 512u * sizeof(float));

    const short quadrant = (short)sgitg;
    for (short out_lane = 0; out_lane < NW; ++out_lane) {
        const short oc = quadrant * NW + out_lane;
        const float4 materialized = htmp4[(uint)oc * (uint)NWG + (uint)iwg];
        const float4 reduced = simd_sum(materialized * ms);

        if (iwg == 0) {
            dst4[oc] = reduced * reduce_inv;
        }
    }
}

/* ==========================================================================
 * [D-A1] Single-Pass MQA-64 MMA Decode Attention with Fused RoPE
 *
 * Tiles 64 query heads x 1 shared KV head as Q=8 heads x C=64 keys, NSG=4
 * (128 threads/TG, 10,272 B dynamic smem -> 3 TGs/core on 32 KiB TG memory).
 * Grid = (8 head_tiles, D_SPLIT, 1) = 16 / 32 / 64 TGs.
 * Each TG owns DV_SLICE = 512 / D_SPLIT output columns (NO_COL = 8 / 4 / 2
 * simdgroup_float8x8 accumulators per simdgroup -> tiny register footprint,
 * zero VGPR spill, zero 2nd pass, zero partial buffer, zero pad kernel).
 * ========================================================================== */

template<short D_SPLIT, bool FP16_P, bool FUSE_ROPE>
static inline void myds_attention_decode_mma_rope_fuse_impl(
        constant myds_attention_rope_fuse_args & fused,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    constant ds4_metal_args_flash_attn_ext_vec & args = fused.attention;
    if (fused.pos >= 1048576u ||
        args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
        args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
        args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
        args.ne11 <= 0 || args.ne11 > 1024 ||
        tgpig.x >= 8u || tgpig.y >= (uint)D_SPLIT ||
        args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    constexpr short NW        = 32;
    constexpr short DK        = 512;
    constexpr short DV        = 512;
    constexpr short DK4       = DK / 4;                  // 128
    constexpr short DK8       = DK / 8;                  // 64
    constexpr short Q         = 8;
    constexpr short C         = 64;
    constexpr short NSG       = 4;
    constexpr short NQ        = Q / NSG;                 // 2
    constexpr short SH        = C;                       // 64 floats/row = 256 B
    constexpr short SP_STRIDE = 2 * SH;                  // 128 halfs/row = 256 B
    constexpr short DV_SLICE  = DV / D_SPLIT;            // 256 / 128 / 64
    constexpr short NO_COL    = DV_SLICE / (8 * NSG);    // 8 / 4 / 2
    constexpr float NEG_LARGE = -FLT_MAX / 2.0f;

    const short head_base = (short)(tgpig.x * (uint)Q);
    const short dv_base   = (short)(tgpig.y * (uint)DV_SLICE);

    // 10,272 B total dynamic smem (16-byte aligned):
    //   sq:   8 * 512 * 2 B = 8,192 B
    //   ss:   8 *  64 * 4 B = 2,048 B  (sp overlays first 128 B of each 256 B row)
    //   s_ms: 8 *       4 B =    32 B
    threadgroup half   * sq   = shmem_f16;
    threadgroup half4  * sq4  = (threadgroup half4 *) shmem_f16;
    threadgroup float  * ss   = (threadgroup float *) (shmem_f16 + Q * DK);
    threadgroup float2 * ss2  = (threadgroup float2 *) ss;
    threadgroup half   * sp   = (threadgroup half *) ss;
    threadgroup float  * s_ms = ss + Q * SH;

    const short lo_row = ((tiisg >> 2) & 4) | ((tiisg >> 1) & 3);
    const short lo_col = ((tiisg >> 1) & 4) | ((tiisg & 1) << 1);

    // Load 8 query heads [head_base .. head_base + 7] from F32 -> F16 into sq.
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 =
            (device const float4 *)(q + (uint64_t)(head_base + j) * args.nb02);
        for (short i = (short)tiisg; i < DK4; i += NW) {
            sq4[j * DK4 + i] = (half4)q4[i];
        }
    }

    simdgroup_float8x8 lo[NO_COL];
    FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
        lo[ii] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { 0.0f, 0.0f };
    float M[NQ] = { NEG_LARGE, NEG_LARGE };

    device const half * pm = (device const half *)mask;
    const short c0 = (short)(2 * tiisg);
    const short c1 = c0 + 1;
    const int full_end = args.ne11 & ~((int)C - 1);

    // ---- Hot Loop: Full 64-key tiles (zero tail branches, zero register bloat) ----
    for (int ic = 0; ic < full_end; ic += C) {
        const half m0 = pm[ic + c0];
        const half m1 = pm[ic + c1];
        if (simd_max(max(m0, m1)) <= -MAXHALF) {
            continue;
        }

        // Step 1: Q[8, 512] * K[64, 512]^T -> ss[8, 64]
        // Contiguous 16-row K pairs per SG (sgitg*16, sgitg*16+8) share sq loads
        // across both 8-key blocks. Keep unroll bounded to 2 with mem_none fences
        // so Apple8 (M2 Ultra, static VGPR file, no Dynamic Caching) never hoists
        // loads across 16 iterations and spills to DRAM (~56-68 VGPRs, 0 B spill).
        const short k0_off = (short)(sgitg * 16);
        const short k1_off = k0_off + 8;
        device const half * pk0 = (device const half *)(k + (uint64_t)(ic + k0_off) * args.nb11);
        device const half * pk1 = pk0 + (8 * DK);

        simdgroup_float8x8 mqk0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 mqk1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

        #pragma unroll 2
        for (short i = 0; i < DK8 / 2; ++i) {
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_half8x8 mq0, mq1, mka0, mka1, mkb0, mkb1;
            simdgroup_load(mq0,  sq  + 16 * i + 0, DK);
            simdgroup_load(mq1,  sq  + 16 * i + 8, DK);
            simdgroup_load(mka0, pk0 + 16 * i + 0, DK, 0, true);
            simdgroup_load(mkb0, pk1 + 16 * i + 0, DK, 0, true);
            simdgroup_load(mka1, pk0 + 16 * i + 8, DK, 0, true);
            simdgroup_load(mkb1, pk1 + 16 * i + 8, DK, 0, true);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mqk0, mq0, mka0, mqk0);
            simdgroup_multiply_accumulate(mqk1, mq0, mkb0, mqk1);
            simdgroup_multiply_accumulate(mqk0, mq1, mka1, mqk0);
            simdgroup_multiply_accumulate(mqk1, mq1, mkb1, mqk1);
        }

        simdgroup_store(mqk0, ss + k0_off, SH, 0, false);
        simdgroup_store(mqk1, ss + k1_off, SH, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Online Softmax (2 rows/SG, -ffast-math safe finite guards)
        const bool v0 = (m0 > -MAXHALF);
        const bool v1 = (m1 > -MAXHALF);

        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const float old_m = M[jj];

            float2 s2 = ss2[j * (SH / 2) + tiisg];
            s2.x = v0 ? fma(s2.x, args.scale, float(m0)) : NEG_LARGE;
            s2.y = v1 ? fma(s2.y, args.scale, float(m1)) : NEG_LARGE;

            const float new_m = simd_max(max(old_m, max(s2.x, s2.y)));
            const float ms = (old_m > -1e37f) ? exp(old_m - new_m) : 0.0f;
            float2 vs2;
            vs2.x = v0 ? exp(s2.x - new_m) : 0.0f;
            vs2.y = v1 ? exp(s2.y - new_m) : 0.0f;

            M[jj] = new_m;
            S[jj] = S[jj] * ms + simd_sum(vs2.x + vs2.y);

            if (FP16_P) {
                ((threadgroup half2 *)(ss + j * SH))[tiisg] = (half2)vs2;
            } else {
                ss2[j * (SH / 2) + tiisg] = vs2;
            }

            if (tiisg == 0) {
                s_ms[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Rescale running O accumulator
        if (ic > 0) {
            #pragma clang fp contract(off)
            const float ms_r = s_ms[lo_row];
            FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                lo[ii].thread_elements()[0] *= ms_r;
                lo[ii].thread_elements()[1] *= ms_r;
            }
        }

        // Step 4: P[8, 64] * V[64, DV_SLICE] -> lo[NO_COL]
        device const half * pv =
            (device const half *)(v + (uint64_t)ic * args.nb21) + dv_base + 8 * sgitg;

        constexpr short NC_PV = (C / 8) / 2;
        #pragma clang loop unroll(disable)
        for (short cc = 0; cc < NC_PV; ++cc) {
            if (FP16_P) {
                simdgroup_half8x8 vp0, vp1;
                simdgroup_load(vp0, sp + 16 * cc + 0, SP_STRIDE, 0, false);
                simdgroup_load(vp1, sp + 16 * cc + 8, SP_STRIDE, 0, false);

                FOR_UNROLL (short ii = 0; ii < NO_COL / 2; ++ii) {
                    simdgroup_half8x8 mv0, mv1, mv2, mv3;
                    simdgroup_load(mv0, pv + 0 * 8 * NSG + 16 * ii * NSG + 0 * 8 * DV, DV, 0, false);
                    simdgroup_load(mv1, pv + 1 * 8 * NSG + 16 * ii * NSG + 0 * 8 * DV, DV, 0, false);
                    simdgroup_load(mv2, pv + 0 * 8 * NSG + 16 * ii * NSG + 1 * 8 * DV, DV, 0, false);
                    simdgroup_load(mv3, pv + 1 * 8 * NSG + 16 * ii * NSG + 1 * 8 * DV, DV, 0, false);

                    simdgroup_multiply_accumulate(lo[2 * ii + 0], vp0, mv0, lo[2 * ii + 0]);
                    simdgroup_multiply_accumulate(lo[2 * ii + 1], vp0, mv1, lo[2 * ii + 1]);
                    simdgroup_multiply_accumulate(lo[2 * ii + 0], vp1, mv2, lo[2 * ii + 0]);
                    simdgroup_multiply_accumulate(lo[2 * ii + 1], vp1, mv3, lo[2 * ii + 1]);
                }
            } else {
                simdgroup_float8x8 vp0, vp1;
                simdgroup_load(vp0, ss + 16 * cc + 0, SH, 0, false);
                simdgroup_load(vp1, ss + 16 * cc + 8, SH, 0, false);

                FOR_UNROLL (short ii = 0; ii < NO_COL / 2; ++ii) {
                    simdgroup_half8x8 mv0, mv1, mv2, mv3;
                    simdgroup_load(mv0, pv + 0 * 8 * NSG + 16 * ii * NSG + 0 * 8 * DV, DV, 0, false);
                    simdgroup_load(mv1, pv + 1 * 8 * NSG + 16 * ii * NSG + 0 * 8 * DV, DV, 0, false);
                    simdgroup_load(mv2, pv + 0 * 8 * NSG + 16 * ii * NSG + 1 * 8 * DV, DV, 0, false);
                    simdgroup_load(mv3, pv + 1 * 8 * NSG + 16 * ii * NSG + 1 * 8 * DV, DV, 0, false);

                    simdgroup_multiply_accumulate(lo[2 * ii + 0], vp0, mv0, lo[2 * ii + 0]);
                    simdgroup_multiply_accumulate(lo[2 * ii + 1], vp0, mv1, lo[2 * ii + 1]);
                    simdgroup_multiply_accumulate(lo[2 * ii + 0], vp1, mv2, lo[2 * ii + 0]);
                    simdgroup_multiply_accumulate(lo[2 * ii + 1], vp1, mv3, lo[2 * ii + 1]);
                }
            }
            pv += 2 * 8 * DV;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Optional Single Tail Tile (1 <= rem <= 63, zero OOB reads) ----
    if (full_end < args.ne11) {
        const int ic = full_end;
        const short rem = (short)(args.ne11 - ic);

        const half m0 = (c0 < rem) ? pm[ic + c0] : -MAXHALF;
        const half m1 = (c1 < rem) ? pm[ic + c1] : -MAXHALF;
        if (simd_max(max(m0, m1)) > -MAXHALF) {
            // Interleaved 8-row MMA blocks keep all 4 SGs busy for rem >= 32.
            const short tail_start = rem & ~7;
            #pragma clang loop unroll(disable)
            for (short cc = 0; cc < 2; ++cc) {
                const short k_off = cc * 32 + sgitg * 8;
                if (k_off + 8 <= rem) {
                    device const half * pk =
                        (device const half *)(k + (uint64_t)(ic + k_off) * args.nb11);
                    simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    #pragma clang loop unroll(disable)
                    for (short i = 0; i < DK8 / 2; ++i) {
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_half8x8 mk0, mk1, mq0, mq1;
                        simdgroup_load(mk0, pk + 16 * i + 0, DK, 0, true);
                        simdgroup_load(mk1, pk + 16 * i + 8, DK, 0, true);
                        simdgroup_load(mq0, sq + 16 * i + 0, DK);
                        simdgroup_load(mq1, sq + 16 * i + 8, DK);
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                        simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                    }
                    simdgroup_store(mqk, ss + k_off, SH, 0, false);
                }
            }
            // Remaining 1..7 partial rows: all 4 SGs share the 8 query heads (2 heads/SG,
            // single streaming K load per row -> 4x parallel + 4x fewer K loads, low regs).
            if (tail_start < rem) {
                const short qh0 = (short)sgitg;
                const short qh1 = qh0 + NSG;
                for (short r = tail_start; r < rem; ++r) {
                    device const half4 * pk4 =
                        (device const half4 *)(k + (uint64_t)(ic + r) * args.nb11);
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    for (short d = (short)tiisg; d < DK4; d += NW) {
                        const float4 kv = (float4)pk4[d];
                        acc0 += dot(kv, (float4)sq4[qh0 * DK4 + d]);
                        acc1 += dot(kv, (float4)sq4[qh1 * DK4 + d]);
                    }
                    acc0 = simd_sum(acc0);
                    acc1 = simd_sum(acc1);
                    if (tiisg == 0) {
                        ss[qh0 * SH + r] = acc0;
                        ss[qh1 * SH + r] = acc1;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const bool v0 = (c0 < rem) && (m0 > -MAXHALF);
            const bool v1 = (c1 < rem) && (m1 > -MAXHALF);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj * NSG + sgitg;
                const float old_m = M[jj];

                float2 s2 = ss2[j * (SH / 2) + tiisg];
                s2.x = v0 ? fma(s2.x, args.scale, float(m0)) : NEG_LARGE;
                s2.y = v1 ? fma(s2.y, args.scale, float(m1)) : NEG_LARGE;

                const float new_m = simd_max(max(old_m, max(s2.x, s2.y)));
                const float ms = (old_m > -1e37f) ? exp(old_m - new_m) : 0.0f;
                float2 vs2;
                vs2.x = v0 ? exp(s2.x - new_m) : 0.0f;
                vs2.y = v1 ? exp(s2.y - new_m) : 0.0f;

                M[jj] = new_m;
                S[jj] = S[jj] * ms + simd_sum(vs2.x + vs2.y);

                if (FP16_P) {
                    ((threadgroup half2 *)(ss + j * SH))[tiisg] = (half2)vs2;
                } else {
                    ss2[j * (SH / 2) + tiisg] = vs2;
                }

                if (tiisg == 0) {
                    s_ms[j] = ms;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (ic > 0) {
                #pragma clang fp contract(off)
                const float ms_r = s_ms[lo_row];
                FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                    lo[ii].thread_elements()[0] *= ms_r;
                    lo[ii].thread_elements()[1] *= ms_r;
                }
            }

            device const half * pv =
                (device const half *)(v + (uint64_t)ic * args.nb21) + dv_base + 8 * sgitg;
            const short full8 = rem >> 3;
            #pragma clang loop unroll(disable)
            for (short c8 = 0; c8 < full8; ++c8) {
                if (FP16_P) {
                    simdgroup_half8x8 vp;
                    simdgroup_load(vp, sp + 8 * c8, SP_STRIDE, 0, false);
                    FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                        simdgroup_half8x8 mv;
                        simdgroup_load(mv, pv + ii * 8 * NSG, DV, 0, false);
                        simdgroup_multiply_accumulate(lo[ii], vp, mv, lo[ii]);
                    }
                } else {
                    simdgroup_float8x8 vp;
                    simdgroup_load(vp, ss + 8 * c8, SH, 0, false);
                    FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                        simdgroup_half8x8 mv;
                        simdgroup_load(mv, pv + ii * 8 * NSG, DV, 0, false);
                        simdgroup_multiply_accumulate(lo[ii], vp, mv, lo[ii]);
                    }
                }
                pv += 8 * DV;
            }
            for (short r = full8 * 8; r < rem; ++r) {
                const float p_val = FP16_P ? float(sp[lo_row * SP_STRIDE + r]) : ss[lo_row * SH + r];
                device const half * pv_row =
                    (device const half *)(v + (uint64_t)(ic + r) * args.nb21);
                FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                    const short col = dv_base + 8 * sgitg + ii * 8 * NSG + lo_col;
                    const half2 v2 = *(device const half2 *)(pv_row + col);
                    lo[ii].thread_elements()[0] = fma(p_val, float(v2.x), lo[ii].thread_elements()[0]);
                    lo[ii].thread_elements()[1] = fma(p_val, float(v2.y), lo[ii].thread_elements()[1]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ---- Step 5: Attention Sinks + Final Normalization Scale ----
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        const float old_m = M[jj];
        const float sink = ((device const float *)sinks)[head_base + j];
        const bool old_valid = (old_m > -1e37f);
        const bool sink_valid = (sink > -1e37f);
        const float new_m = max(old_m, sink);
        const float ms = old_valid ? exp(old_m - new_m) : 0.0f;
        const float vs = sink_valid ? exp(sink - new_m) : 0.0f;
        const float s_total = S[jj] * ms + vs;
        const float scale = (s_total == 0.0f) ? 0.0f : (ms / s_total);
        if (tiisg == 0) {
            s_ms[j] = scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 6: Epilogue (Scale + BF16 Round + Fused RoPE on cols 448..511) ----
    {
        #pragma clang fp contract(off)
        const short head = head_base + lo_row;
        const float scale = s_ms[lo_row];
        device float2 * dst2 =
            (device float2 *)((device float *)dst + (uint64_t)head * (uint64_t)DV);
        const bool is_rope_slice = (tgpig.y == (uint)(D_SPLIT - 1));
        FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
            const short col = dv_base + 8 * sgitg + ii * 8 * NSG + lo_col;
            const short c2 = col >> 1;
            float re = lo[ii].thread_elements()[0] * scale;
            float im = lo[ii].thread_elements()[1] * scale;
            if (FUSE_ROPE) {
                re = myds_attention_rope_bf16(re);
                im = myds_attention_rope_bf16(im);
                if (is_rope_slice && c2 >= 224) {
                    const uint pair = (uint)(c2 - 224);
                    const float theta = float(fused.pos) * fused.frequencies[pair];
                    const float c = precise::cos(theta);
                    const float s = fused.inverse ? -precise::sin(theta) : precise::sin(theta);
                    const float r_re = myds_attention_rope_bf16(re * c - im * s);
                    const float r_im = myds_attention_rope_bf16(precise::fma(im, c, re * s));
                    re = r_re;
                    im = r_im;
                }
            }
            dst2[c2] = float2(re, im);
        }
    }
}

template<short D_SPLIT, bool FP16_P>
kernel void kernel_myds_attention_decode_mma_rope_fuse_tmpl(
        constant myds_attention_rope_fuse_args & fused [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device       char * dst                       [[buffer(6)]],
        threadgroup  half * shmem_f16                 [[threadgroup(0)]],
        uint3   tgpig                                 [[threadgroup_position_in_grid]],
        ushort  tiisg                                 [[thread_index_in_simdgroup]],
        ushort  sgitg                                 [[simdgroup_index_in_threadgroup]]) {
    myds_attention_decode_mma_rope_fuse_impl<D_SPLIT, FP16_P, true>(
        fused, q, k, v, mask, sinks, dst, shmem_f16, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_myds_attention_decode_mma_rope_fuse_tmpl<2, false>) attn_decode_mma_t;

// FP32-P (simdgroup_float8x8 x half8x8 -> float8x8) instantiations:
template [[host_name("kernel_myds_attention_decode_mma_rope_fuse_f16_512_d2")]]
kernel attn_decode_mma_t kernel_myds_attention_decode_mma_rope_fuse_tmpl<2, false>;
template [[host_name("kernel_myds_attention_decode_mma_rope_fuse_f16_512_d4")]]
kernel attn_decode_mma_t kernel_myds_attention_decode_mma_rope_fuse_tmpl<4, false>;
template [[host_name("kernel_myds_attention_decode_mma_rope_fuse_f16_512_d8")]]
kernel attn_decode_mma_t kernel_myds_attention_decode_mma_rope_fuse_tmpl<8, false>;

// FP16-P (simdgroup_half8x8 x half8x8 -> float8x8) instantiations:
template [[host_name("kernel_myds_attention_decode_mma_rope_fuse_f16_512_d2_p16")]]
kernel attn_decode_mma_t kernel_myds_attention_decode_mma_rope_fuse_tmpl<2, true>;
template [[host_name("kernel_myds_attention_decode_mma_rope_fuse_f16_512_d4_p16")]]
kernel attn_decode_mma_t kernel_myds_attention_decode_mma_rope_fuse_tmpl<4, true>;
template [[host_name("kernel_myds_attention_decode_mma_rope_fuse_f16_512_d8_p16")]]
kernel attn_decode_mma_t kernel_myds_attention_decode_mma_rope_fuse_tmpl<8, true>;

// Canonical default entry point (D_SPLIT=4, FP32-P).
kernel void kernel_myds_attention_decode_mma_rope_fuse_f16_512(
        constant myds_attention_rope_fuse_args & fused [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device       char * dst                       [[buffer(6)]],
        threadgroup  half * shmem_f16                 [[threadgroup(0)]],
        uint3   tgpig                                 [[threadgroup_position_in_grid]],
        ushort  tiisg                                 [[thread_index_in_simdgroup]],
        ushort  sgitg                                 [[simdgroup_index_in_threadgroup]]) {
    myds_attention_decode_mma_rope_fuse_impl<4, false, true>(
        fused, q, k, v, mask, sinks, dst, shmem_f16, tgpig, tiisg, sgitg);
}

/* ==========================================================================
 * [D-A1 v2] Single-Pass In-TG Split-K MQA-64 MMA Decode Attention + Fused RoPE
 *
 * Eliminates serial K-tile execution on long decode (keys > 128, e.g. 640k):
 * - Grid = (8 head_tiles, D_SPLIT=8, 1) = 64 TGs across 76 M2 Ultra cores.
 * - KSG independent simdgroups/TG (KSG in {4, 8, 10, 16}) each own strided
 *   16-key K tiles (ic = sgitg*16; ic < keys; ic += KSG*16) with ZERO
 *   threadgroup_barriers inside the K-loop.
 * - 100% in-register 4-lane simd_shuffle_xor(^1, ^8) online softmax directly
 *   on simdgroup_float8x8 mqk0/mqk1, reusing mqk0/mqk1 in-place as P operands
 *   for P*V (zero smem store/load for scores/weights during K-loop).
 * - Post-loop: overwrites sq (8,192 B) within the same 10,272 B dynamic smem
 *   budget (3 TGs/core) to merge KSG partial (O, M, S) + sinks + BF16 + RoPE
 *   (dims 448..511 on tgpig.y == D_SPLIT-1) in-TG with ZERO 2nd pass and
 *   ZERO global partial buffer.
 * - Verified on Apple8 (applegpu_g14d, macOS 15/26/27): <= 64 VGPRs, 0 B spill
 *   (maxTotalThreadsPerThreadgroup = 832).
 * ========================================================================== */

template<short D_SPLIT, short KSG, short COLS_PER_PASS, bool FUSE_ROPE>
static inline void myds_attention_decode_mma_splitk_impl(
        constant myds_attention_rope_fuse_args & fused,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    constant ds4_metal_args_flash_attn_ext_vec & args = fused.attention;
    if (fused.pos >= 1048576u ||
        args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
        args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
        args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
        args.ne11 <= 0 || args.ne11 > 1024 ||
        tgpig.x >= 8u || tgpig.y >= (uint)D_SPLIT ||
        args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024) {
        return;
    }

    constexpr short NW        = 32;
    constexpr short DK        = 512;
    constexpr short DV        = 512;
    constexpr short DK4       = DK / 4;                  // 128
    constexpr short DK8       = DK / 8;                  // 64
    constexpr short Q         = 8;
    constexpr short DV_SLICE  = DV / D_SPLIT;            // 64 for D_SPLIT=8
    constexpr short NO_COL    = DV_SLICE / 8;            // 8 for D_SPLIT=8
    constexpr float NEG_LARGE = -FLT_MAX / 2.0f;

    const short head_base = (short)(tgpig.x * (uint)Q);
    const short dv_base   = (short)(tgpig.y * (uint)DV_SLICE);
    const short tid_tg    = (short)(sgitg * NW + tiisg);
    const short n_thr     = (short)(KSG * NW);

    // Phase 1 smem: sq[8][512] halfs = 8,192 B (read-only during K-loop).
    threadgroup half  * sq  = shmem_f16;
    threadgroup half4 * sq4 = (threadgroup half4 *) shmem_f16;

    // Cooperative Q load across all KSG*32 threads (8*128 = 1024 float4s).
    for (short idx = tid_tg; idx < Q * DK4; idx += n_thr) {
        const short j = idx >> 7;
        const short i = idx & 127;
        device const float4 * q4 =
            (device const float4 *)(q + (uint64_t)(head_base + j) * args.nb02);
        sq4[idx] = (half4)q4[i];
    }

    simdgroup_float8x8 lo[NO_COL];
    FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
        lo[ii] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apple Silicon canonical simdgroup_float8x8 fragment layout:
    // Each row r in 0..7 spans 4 lanes differing in bit 0 (^1) and bit 3 (^8).
    const short lo_row  = ((tiisg >> 2) & 4) | ((tiisg >> 1) & 3);
    const short lo_col  = ((tiisg >> 1) & 4) | ((tiisg & 1) << 1); // 0, 2, 4, 6

    float M_row = NEG_LARGE;
    float S_row = 0.0f;

    device const half * pm = (device const half *)mask;
    const int ne11 = args.ne11;
    const int n_full16 = ne11 >> 4;
    const int full_end = n_full16 << 4;
    const int stride_k = (int)KSG * 16;

    // ---- Phase 1A: Independent Per-SG Full 16-Key MMA Loop (ZERO barriers!) ----
    for (int ic = (int)sgitg * 16; ic < full_end; ic += stride_k) {
        const half m0 = pm[ic + lo_col + 0];
        const half m1 = pm[ic + lo_col + 1];
        const half m2 = pm[ic + 8 + lo_col + 0];
        const half m3 = pm[ic + 8 + lo_col + 1];
        if (simd_max(max(max(m0, m1), max(m2, m3))) <= -MAXHALF) {
            continue;
        }

        device const half * pk0 = (device const half *)(k + (uint64_t)ic * args.nb11);
        device const half * pk1 = pk0 + (8 * DK);

        simdgroup_float8x8 mqk0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 mqk1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

        // Global K loads issued ahead of smem Q loads; #pragma unroll 2 +
        // mem_none fences keep Apple8 VGPRs <= 64 with 0 B stack spill.
        #pragma unroll 2
        for (short i = 0; i < DK8 / 2; ++i) {
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_half8x8 mq0, mq1, mka0, mka1, mkb0, mkb1;
            simdgroup_load(mka0, pk0 + 16 * i + 0, DK, 0, true);
            simdgroup_load(mka1, pk0 + 16 * i + 8, DK, 0, true);
            simdgroup_load(mkb0, pk1 + 16 * i + 0, DK, 0, true);
            simdgroup_load(mkb1, pk1 + 16 * i + 8, DK, 0, true);
            simdgroup_load(mq0,  sq  + 16 * i + 0, DK);
            simdgroup_load(mq1,  sq  + 16 * i + 8, DK);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mqk0, mq0, mka0, mqk0);
            simdgroup_multiply_accumulate(mqk1, mq0, mkb0, mqk1);
            simdgroup_multiply_accumulate(mqk0, mq1, mka1, mqk0);
            simdgroup_multiply_accumulate(mqk1, mq1, mkb1, mqk1);
        }

        // In-register 4-lane online softmax per lo_row (-ffast-math safe)
        const bool v0 = (m0 > -MAXHALF);
        const bool v1 = (m1 > -MAXHALF);
        const bool v2 = (m2 > -MAXHALF);
        const bool v3 = (m3 > -MAXHALF);

        const float s0 = v0 ? fma(mqk0.thread_elements()[0], args.scale, float(m0)) : NEG_LARGE;
        const float s1 = v1 ? fma(mqk0.thread_elements()[1], args.scale, float(m1)) : NEG_LARGE;
        const float s2 = v2 ? fma(mqk1.thread_elements()[0], args.scale, float(m2)) : NEG_LARGE;
        const float s3 = v3 ? fma(mqk1.thread_elements()[1], args.scale, float(m3)) : NEG_LARGE;

        float tile_m = max(max(s0, s1), max(s2, s3));
        tile_m = max(tile_m, simd_shuffle_xor(tile_m, 1));
        tile_m = max(tile_m, simd_shuffle_xor(tile_m, 8));

        const float old_m = M_row;
        const float new_m = max(old_m, tile_m);
        const float ms = (old_m > -1e37f) ? exp(old_m - new_m) : 0.0f;

        const float p0 = v0 ? exp(s0 - new_m) : 0.0f;
        const float p1 = v1 ? exp(s1 - new_m) : 0.0f;
        const float p2 = v2 ? exp(s2 - new_m) : 0.0f;
        const float p3 = v3 ? exp(s3 - new_m) : 0.0f;

        float tile_s = (p0 + p1) + (p2 + p3);
        tile_s += simd_shuffle_xor(tile_s, 1);
        tile_s += simd_shuffle_xor(tile_s, 8);

        M_row = new_m;
        S_row = S_row * ms + tile_s;

        if (old_m > -1e37f) {
            #pragma clang fp contract(off)
            FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                lo[ii].thread_elements()[0] *= ms;
                lo[ii].thread_elements()[1] *= ms;
            }
        }

        // Reuse mqk0/mqk1 in-place as P[8,16] operands for P*V MMA.
        mqk0.thread_elements()[0] = p0;
        mqk0.thread_elements()[1] = p1;
        mqk1.thread_elements()[0] = p2;
        mqk1.thread_elements()[1] = p3;

        device const half * pv0 = (device const half *)(v + (uint64_t)ic * args.nb21) + dv_base;
        device const half * pv1 = pv0 + (8 * DV);
        FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
            simdgroup_half8x8 mv0, mv1;
            simdgroup_load(mv0, pv0 + ii * 8, DV, 0, false);
            simdgroup_load(mv1, pv1 + ii * 8, DV, 0, false);
            simdgroup_multiply_accumulate(lo[ii], mqk0, mv0, lo[ii]);
            simdgroup_multiply_accumulate(lo[ii], mqk1, mv1, lo[ii]);
        }
    }

    // ---- Phase 1B/1C: Balanced Distributed Tail (1..15 keys, zero OOB) ----
    // Assign the optional 8-key MMA block to the first least-loaded SG
    // (n_full16 % KSG), then distribute the remaining 1..7 scalar rows
    // round-robin across subsequent SGs (1 row/SG for KSG >= 8) using
    // 32-lane coalesced K reads (4 half4 loads/row shared across 8 heads).
    int tail_base = full_end;
    int tail_rem  = ne11 - full_end;
    short sg_cur  = (short)((uint)n_full16 % (uint)KSG);
    if (tail_rem >= 8) {
        if ((short)sgitg == sg_cur) {
            const int ic = tail_base;
            const half m0 = pm[ic + lo_col + 0];
            const half m1 = pm[ic + lo_col + 1];
            if (simd_max(max(m0, m1)) > -MAXHALF) {
                device const half * pk0 = (device const half *)(k + (uint64_t)ic * args.nb11);
                simdgroup_float8x8 mqk0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
                #pragma unroll 2
                for (short i = 0; i < DK8 / 2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_half8x8 mq0, mq1, mka0, mka1;
                    simdgroup_load(mka0, pk0 + 16 * i + 0, DK, 0, true);
                    simdgroup_load(mka1, pk0 + 16 * i + 8, DK, 0, true);
                    simdgroup_load(mq0,  sq  + 16 * i + 0, DK);
                    simdgroup_load(mq1,  sq  + 16 * i + 8, DK);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk0, mq0, mka0, mqk0);
                    simdgroup_multiply_accumulate(mqk0, mq1, mka1, mqk0);
                }
                const bool v0 = (m0 > -MAXHALF);
                const bool v1 = (m1 > -MAXHALF);
                const float s0 = v0 ? fma(mqk0.thread_elements()[0], args.scale, float(m0)) : NEG_LARGE;
                const float s1 = v1 ? fma(mqk0.thread_elements()[1], args.scale, float(m1)) : NEG_LARGE;

                float tile_m = max(s0, s1);
                tile_m = max(tile_m, simd_shuffle_xor(tile_m, 1));
                tile_m = max(tile_m, simd_shuffle_xor(tile_m, 8));

                const float old_m = M_row;
                const float new_m = max(old_m, tile_m);
                const float ms = (old_m > -1e37f) ? exp(old_m - new_m) : 0.0f;
                const float p0 = v0 ? exp(s0 - new_m) : 0.0f;
                const float p1 = v1 ? exp(s1 - new_m) : 0.0f;

                float tile_s = p0 + p1;
                tile_s += simd_shuffle_xor(tile_s, 1);
                tile_s += simd_shuffle_xor(tile_s, 8);

                M_row = new_m;
                S_row = S_row * ms + tile_s;

                if (old_m > -1e37f) {
                    #pragma clang fp contract(off)
                    FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                        lo[ii].thread_elements()[0] *= ms;
                        lo[ii].thread_elements()[1] *= ms;
                    }
                }

                mqk0.thread_elements()[0] = p0;
                mqk0.thread_elements()[1] = p1;

                device const half * pv0 = (device const half *)(v + (uint64_t)ic * args.nb21) + dv_base;
                FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                    simdgroup_half8x8 mv0;
                    simdgroup_load(mv0, pv0 + ii * 8, DV, 0, false);
                    simdgroup_multiply_accumulate(lo[ii], mqk0, mv0, lo[ii]);
                }
            }
        }
        tail_base += 8;
        tail_rem  -= 8;
        sg_cur = (sg_cur + 1 < KSG) ? (sg_cur + 1) : 0;
    }
    if (tail_rem > 0) {
        short r_off = (short)sgitg - sg_cur;
        if (r_off < 0) {
            r_off += KSG;
        }
        for (short r = r_off; r < (short)tail_rem; r += KSG) {
            const int key_idx = tail_base + r;
            const half mr = pm[key_idx];
            if (mr <= -MAXHALF) {
                continue;
            }
            device const half4 * pk4 =
                (device const half4 *)(k + (uint64_t)key_idx * args.nb11);
            const half4 kv0 = pk4[tiisg];
            const half4 kv1 = pk4[32 + tiisg];
            const half4 kv2 = pk4[64 + tiisg];
            const half4 kv3 = pk4[96 + tiisg];

            float my_acc = 0.0f;
            #pragma clang loop unroll(disable)
            for (short h = 0; h < Q; ++h) {
                threadgroup const half4 * qh = sq4 + h * DK4;
                float acc = dot((float4)kv0, (float4)qh[tiisg]);
                acc += dot((float4)kv1, (float4)qh[32 + tiisg]);
                acc += dot((float4)kv2, (float4)qh[64 + tiisg]);
                acc += dot((float4)kv3, (float4)qh[96 + tiisg]);
                acc = simd_sum(acc);
                if (lo_row == h) {
                    my_acc = acc;
                }
            }

            const float score = fma(my_acc, args.scale, float(mr));
            const float old_m = M_row;
            const float new_m = max(old_m, score);
            const float ms = (old_m > -1e37f) ? exp(old_m - new_m) : 0.0f;
            const float pr = exp(score - new_m);
            M_row = new_m;
            S_row = S_row * ms + pr;

            device const half * pv_row =
                (device const half *)(v + (uint64_t)key_idx * args.nb21);
            {
                #pragma clang fp contract(off)
                FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
                    const short col = dv_base + ii * 8 + lo_col;
                    const half2 v2 = *(device const half2 *)(pv_row + col);
                    lo[ii].thread_elements()[0] = fma(pr, float(v2.x), lo[ii].thread_elements()[0] * ms);
                    lo[ii].thread_elements()[1] = fma(pr, float(v2.y), lo[ii].thread_elements()[1] * ms);
                }
            }
        }
    }

    // ---- Phase 2: Single-Pass In-TG Merge across KSG Simdgroups + Fused RoPE ----
    // Wait until all SGs finish reading sq, then reuse shmem_f16 (<= 10,272 B).
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * s_M = (threadgroup float *)shmem_f16;
    threadgroup float * s_S = s_M + KSG * Q;

    if (lo_col == 0) {
        s_M[sgitg * Q + lo_row] = M_row;
        s_S[sgitg * Q + lo_row] = S_row;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sink = ((device const float *)sinks)[head_base + lo_row];
    float g_m = sink;
    FOR_UNROLL (short s = 0; s < KSG; ++s) {
        g_m = max(g_m, s_M[s * Q + lo_row]);
    }
    float g_s = (sink > -1e37f) ? exp(sink - g_m) : 0.0f;
    FOR_UNROLL (short s = 0; s < KSG; ++s) {
        const float ms_i = s_M[s * Q + lo_row];
        if (ms_i > -1e37f) {
            g_s += s_S[s * Q + lo_row] * exp(ms_i - g_m);
        }
    }
    const float w_sg = (M_row > -1e37f && g_s > 0.0f) ? (exp(M_row - g_m) / g_s) : 0.0f;

    // Wait for all threads to finish reading s_M/s_S before s_O reuses offset 0.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        #pragma clang fp contract(off)
        FOR_UNROLL (short ii = 0; ii < NO_COL; ++ii) {
            lo[ii].thread_elements()[0] *= w_sg;
            lo[ii].thread_elements()[1] *= w_sg;
        }
    }

    // s_O overwrites shmem_f16 from offset 0:
    //   KSG=4,  COLS_PER_PASS=64 ->  8,192 B (1 pass)
    //   KSG=8,  COLS_PER_PASS=32 ->  8,192 B (2 passes)
    //   KSG=10, COLS_PER_PASS=32 -> 10,240 B (2 passes, <= 10,272 B)
    //   KSG=16, COLS_PER_PASS=16 ->  8,192 B (4 passes)
    threadgroup float * s_O = (threadgroup float *)shmem_f16;
    const bool is_rope_slice = (tgpig.y == (uint)(D_SPLIT - 1));
    constexpr short TILES_PER_PASS = COLS_PER_PASS / 8;
    constexpr short N_PASS         = NO_COL / TILES_PER_PASS;
    constexpr short PAIRS_PER_ROW  = COLS_PER_PASS / 2;
    constexpr short PAIRS_PER_PASS = Q * PAIRS_PER_ROW;

    // Unrolled so lo[...] indices stay compile-time constants (zero stack alloc).
    FOR_UNROLL (short pass = 0; pass < N_PASS; ++pass) {
        const short base_o = (short)sgitg * (Q * COLS_PER_PASS) + lo_row * COLS_PER_PASS + lo_col;
        FOR_UNROLL (short t = 0; t < TILES_PER_PASS; ++t) {
            s_O[base_o + t * 8 + 0] = lo[pass * TILES_PER_PASS + t].thread_elements()[0];
            s_O[base_o + t * 8 + 1] = lo[pass * TILES_PER_PASS + t].thread_elements()[1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short idx_pair = tid_tg; idx_pair < PAIRS_PER_PASS; idx_pair += n_thr) {
            const short r = idx_pair / PAIRS_PER_ROW;
            const short p = (idx_pair % PAIRS_PER_ROW) * 2;
            float re = 0.0f;
            float im = 0.0f;
            FOR_UNROLL (short s = 0; s < KSG; ++s) {
                const short idx = s * (Q * COLS_PER_PASS) + r * COLS_PER_PASS + p;
                re += s_O[idx + 0];
                im += s_O[idx + 1];
            }
            const short col = dv_base + pass * COLS_PER_PASS + p;
            const short c2 = col >> 1;
            if (FUSE_ROPE) {
                #pragma clang fp contract(off)
                re = myds_attention_rope_bf16(re);
                im = myds_attention_rope_bf16(im);
                if (is_rope_slice && c2 >= 224) {
                    const uint pair = (uint)(c2 - 224);
                    const float theta = float(fused.pos) * fused.frequencies[pair];
                    const float c = precise::cos(theta);
                    const float s = fused.inverse ? -precise::sin(theta) : precise::sin(theta);
                    const float r_re = myds_attention_rope_bf16(re * c - im * s);
                    const float r_im = myds_attention_rope_bf16(precise::fma(im, c, re * s));
                    re = r_re;
                    im = r_im;
                }
            }
            device float2 * dst2 =
                (device float2 *)((device float *)dst + (uint64_t)(head_base + r) * (uint64_t)DV);
            dst2[c2] = float2(re, im);
        }

        if (pass + 1 < N_PASS) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template<short D_SPLIT, short KSG, short COLS_PER_PASS>
kernel void kernel_myds_attention_decode_mma_splitk_tmpl(
        constant myds_attention_rope_fuse_args & fused [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device       char * dst                       [[buffer(6)]],
        threadgroup  half * shmem_f16                 [[threadgroup(0)]],
        uint3   tgpig                                 [[threadgroup_position_in_grid]],
        ushort  tiisg                                 [[thread_index_in_simdgroup]],
        ushort  sgitg                                 [[simdgroup_index_in_threadgroup]]) {
    myds_attention_decode_mma_splitk_impl<D_SPLIT, KSG, COLS_PER_PASS, true>(
        fused, q, k, v, mask, sinks, dst, shmem_f16, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_myds_attention_decode_mma_splitk_tmpl<8, 4, 64>) attn_decode_mma_sk_t;

// Single-pass in-TG Split-K instantiations (all < 64 chars for C-array PSO cache):
template [[host_name("kernel_myds_attention_decode_mma_sk4_f16_512_d8")]]
kernel attn_decode_mma_sk_t kernel_myds_attention_decode_mma_splitk_tmpl<8, 4, 64>;
template [[host_name("kernel_myds_attention_decode_mma_sk8_f16_512_d8")]]
kernel attn_decode_mma_sk_t kernel_myds_attention_decode_mma_splitk_tmpl<8, 8, 32>;
template [[host_name("kernel_myds_attention_decode_mma_sk10_f16_512_d8")]]
kernel attn_decode_mma_sk_t kernel_myds_attention_decode_mma_splitk_tmpl<8, 10, 32>;
template [[host_name("kernel_myds_attention_decode_mma_sk16_f16_512_d8")]]
kernel attn_decode_mma_sk_t kernel_myds_attention_decode_mma_splitk_tmpl<8, 16, 16>;


