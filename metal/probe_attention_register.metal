/* Private probe. Not included in official SHADER_FILES.
 * Original arithmetic and logical splits remain fixed; debug adds full partial/stat exports. */
template <bool CAPTURE>
static inline void probe_attn_tg_impl(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        device       char * partial_dump,
        device       char * stats_dump,
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
            if (CAPTURE) {
                device float * dump = (device float *)stats_dump;
                const uint at = ((uint)head * 32u + (uint)iwg) * 3u;
                dump[at+0] = S; dump[at+1] = M; dump[at+2] = out_scale;
            }
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
            if (CAPTURE) {
                device float4 * dump = (device float4 *)partial_dump;
                dump[((uint)head * 32u + (uint)iwg) * 128u + (uint)quadrant * 32u + (uint)lane] = lo;
            }
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

template <bool CAPTURE>
static inline void probe_attn_register_impl(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        device       char * partial_dump,
        device       char * stats_dump,
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
            if (CAPTURE) {
                device float * dump = (device float *)stats_dump;
                const uint at = ((uint)head * 32u + (uint)iwg) * 3u;
                dump[at+0] = S; dump[at+1] = M; dump[at+2] = out_scale;
            }
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
            if (CAPTURE) {
                device float4 * dump = (device float4 *)partial_dump;
                dump[((uint)head * 32u + (uint)iwg) * 128u + (uint)quadrant * 32u + (uint)lane] = lo;
            }
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

kernel void probe_attention_tg(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        device       char * partial_dump              [[buffer(8)]],
        device       char * stats_dump                [[buffer(9)]],
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

    probe_attn_tg_impl<false>(
        args, q, k, v, mask, sinks, pad, dst, partial_dump, stats_dump, shmem,
        head, tiisg, sgitg);
}


kernel void probe_attention_tg_debug(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        device       char * partial_dump              [[buffer(8)]],
        device       char * stats_dump                [[buffer(9)]],
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

    probe_attn_tg_impl<true>(
        args, q, k, v, mask, sinks, pad, dst, partial_dump, stats_dump, shmem,
        head, tiisg, sgitg);
}


kernel void probe_attention_register(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        device       char * partial_dump              [[buffer(8)]],
        device       char * stats_dump                [[buffer(9)]],
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

    probe_attn_register_impl<false>(
        args, q, k, v, mask, sinks, pad, dst, partial_dump, stats_dump, shmem,
        head, tiisg, sgitg);
}


kernel void probe_attention_register_debug(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        device       char * partial_dump              [[buffer(8)]],
        device       char * stats_dump                [[buffer(9)]],
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

    probe_attn_register_impl<true>(
        args, q, k, v, mask, sinks, pad, dst, partial_dump, stats_dump, shmem,
        head, tiisg, sgitg);
}

