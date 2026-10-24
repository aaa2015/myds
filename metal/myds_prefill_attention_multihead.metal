// Default-off gathered-prefill experiment, R2. Assemble after flash_attn.metal.
//
// Each physical SIMD owns one head and executes the original NE=1, NSG=1,
// NWG=1 head-first vector arithmetic. Only K loads are shared: a 16-key
// half tile is read by all heads before the next half tile replaces it.
// A 32-key online-softmax block remains intact; splitting the block into two
// softmax updates or changing the PV key order would violate the 0-ULP goal.
// PV retains the native device loads and one uninterrupted 32-key loop.
// R1 shared both K and V and failed full-chunk 0-ULP. Its executed source is
// retained in window1-executed-source.tar.gz. The uninstrumented direct-V
// ablation passed one real query; R2 still requires the complete shape gate.
//
// ABI: the original 200-byte args and buffers 1..7 (Q/K/V/mask/sinks/pad/out).
// FC400..405 and FC420..423 retain the original values, including logical
// NSG=1 and NWG=1.  The physical SIMD count is the template parameter HEADS.
// Host contract: 64 heads, 640 gathered F16 keys per query, shared mask,
// 512-wide Q/K/V, no bias/softcap/pad, and one independent gathered KV slab
// per query (nb_kvq=655360).  Dispatch (64/HEADS, rows, 1) groups with
// (32,HEADS,1) threads.  Dynamic scratch is HEADS*3328+16384 bytes:
// HEADS=2 -> 23040 bytes; HEADS=4 -> 29696 bytes.
// Host must validate resource ranges/offsets/aliasing before encoding.

template<ushort HEADS>
inline void myds_prefill_stage_half16(
        device const char * source,
        threadgroup half4 * tile,
        ushort tiisg,
        ushort sgitg) {
    const uint tid = uint(sgitg)*32u + tiisg;
    device const half4 * source4 = (device const half4 *) source;
    for (uint i = tid; i < 16u*128u; i += 32u*HEADS) {
        tile[i] = source4[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<ushort HEADS>
kernel void kernel_myds_prefill_attention_multihead(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q [[buffer(1)]],
        device const char * k [[buffer(2)]],
        device const char * v [[buffer(3)]],
        device const char * mask [[buffer(4)]],
        device const char * sinks [[buffer(5)]],
        device const char * pad [[buffer(6)]],
        device char * dst [[buffer(7)]],
        threadgroup half * shmem_f16 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    static_assert(HEADS == 2 || HEADS == 4, "supported head count");

    // Every condition is threadgroup uniform.  These narrow guards also make
    // the all-masked skip below uniform across the participating heads.
    if (!FC_flash_attn_ext_vec_has_mask || !FC_flash_attn_ext_vec_has_sinks ||
        FC_flash_attn_ext_vec_has_bias || FC_flash_attn_ext_vec_has_scap ||
        FC_flash_attn_ext_vec_has_kvpad || FC_flash_attn_ext_vec_shared_kvpad ||
        FC_flash_attn_ext_vec_ns10 != 512 || FC_flash_attn_ext_vec_ns20 != 512 ||
        FC_flash_attn_ext_vec_nsg != 1 || FC_flash_attn_ext_vec_nwg != 1 ||
        args.ne02 != 64 || args.ne03 != 1 || args.ne_12_2 != 1 ||
        args.ne_12_3 != 1 || args.ne11 != 640 || args.ne32 != 1 ||
        args.ne33 != 1 || args.nb11 != 1024 || args.nb21 != 1024 ||
        args.nb_kvq != 655360 || tgpig.x >= 64u/HEADS ||
        tgpig.y >= uint(args.ne01) || tgpig.z != 0u) {
        return;
    }

    constexpr short DK4 = 128;
    constexpr short DV4 = 128;
    constexpr short PK = 512;
    constexpr short PK4 = 128;
    constexpr short NW = 32;
    constexpr short NL = 32;
    constexpr short C = 32;
    constexpr short SH = 4*C;
    constexpr uint PRIVATE_HALF = 3328u/sizeof(half);

    const ushort iq3 = 0;
    const ushort iq2 = tgpig.x*HEADS + sgitg;
    const ushort iq1 = tgpig.y;

    // Preserve the native single-head layout inside each private region.
    // In particular, the native float4 score-bank zeroing also overlaps the
    // beginning of its already-zero output bank; it remains head-private.
    threadgroup half * head_mem = shmem_f16 + sgitg*PRIVATE_HALF;
    threadgroup half4 * sq4 = (threadgroup half4 *) (head_mem + 0*PK);
    threadgroup float * ss = (threadgroup float *) (head_mem + PK);
    threadgroup float4 * ss4 = (threadgroup float4 *) (head_mem + PK);
    threadgroup half * sm = head_mem + 2*C + PK;
    threadgroup float4 * so4 = (threadgroup float4 *) (head_mem + PK + SH);
    threadgroup half4 * tile = (threadgroup half4 *)
        (shmem_f16 + HEADS*PRIVATE_HALF);
    so4 += tiisg;

    q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;
    k += (ulong)iq1 * args.nb_kvq;
    v += (ulong)iq1 * args.nb_kvq;

    device const float4 * q4 = (device const float4 *) q;
    for (short i = tiisg; i < PK4; i += NW) {
        sq4[i] = (half4) q4[i];
    }
    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (float4) 0.0f;
    }
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (float4) 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX/2;
    const short tx = tiisg;
    device const half * pm = (device const half *)
        (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 +
         (iq3%args.ne33)*args.nb33);

    for (int ic0 = 0; ; ++ic0) {
        const int ic = ic0*C;
        if (ic >= args.ne11) {
            break;
        }
        sm[tiisg] = pm[ic + tiisg];
        // ne32=ne33=1 and all SIMD groups share iq1, so all groups inspect
        // the exact same 32 half mask values and take this branch together.
        if (simd_max(sm[tiisg]) <= -MAXHALF) {
            continue;
        }

        {
            threadgroup const half4 * pq4 = sq4 + tx;
            float mqk[C] = { [0 ... C - 1] = 0.0f };

            FOR_UNROLL (short pass = 0; pass < 2; ++pass) {
                myds_prefill_stage_half16<HEADS>(
                    k + (ic + pass*16)*args.nb11, tile, tiisg, sgitg);
                threadgroup const half4 * pk4 = tile + tx;
                FOR_UNROLL (short cc = 0; cc < 16; ++cc) {
                    FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                        mqk[pass*16 + cc] += dot(
                            (float4) pk4[cc*128 + ii*NL],
                            (float4) pq4[ii*NL]);
                    }
                    mqk[pass*16 + cc] = simd_sum(mqk[pass*16 + cc]);
                }
                // No head may overwrite a tile another head still reads.
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            ss[tx] = fma(mqk[tx], args.scale, (float) sm[tx]);
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
            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                so4[ii*NL] *= ms;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        {
            float4 lo[DV4/NL];
            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                lo[ii] = 0.0f;
            }
            const auto sst = ss;
            device const half4 * pv4 = (device const half4 *)(v + ic*args.nb21);
            pv4 += tx;
            FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] += float4(float4(pv4[cc*128 + ii*NL])*float4(sst[cc]));
                }
            }
            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                so4[ii*NL] += lo[ii];
            }
        }
    }

    // Native sink and normalization expressions, once for each head.
    {
        const float m = M;
        const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;
        M = simd_max(max(M, s));
        const float ms = exp(m - M);
        const float vs = exp(s - M);
        S = S*ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
            so4[ii*NL] *= ms;
        }
    }
    if (tiisg == 0) {
        ss[0] = S;
        ss[1] = M;
    }
    so4 -= tiisg;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;
    device float4 * dst4 = (device float4 *) dst;
    const float invS = ss[0] == 0.0f ? 0.0f : 1.0f/ss[0];
    for (short i = tiisg; i < DV4; i += NW) {
        dst4[rid*DV4 + i] = (float4) so4[i]*invS;
    }
}

typedef decltype(kernel_myds_prefill_attention_multihead<2>) myds_prefill_attention_multihead_t;
template [[host_name("kernel_myds_prefill_attention_head2_f16_dk512_dv512")]]
kernel myds_prefill_attention_multihead_t kernel_myds_prefill_attention_multihead<2>;
template [[host_name("kernel_myds_prefill_attention_head4_f16_dk512_dv512")]]
kernel myds_prefill_attention_multihead_t kernel_myds_prefill_attention_multihead<4>;
