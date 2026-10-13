// Independent Dense Q4_K activation-sum probe. Append AFTER production shaders.
// Not part of build_metallib.py SHADER_FILES; no production symbol is changed.
// For every 256-value block cache[it] stores the exact four eight-term sequential
// sums used by kernel_mul_mv_q4_K_f32_impl. The consumer keeps its original K
// partition, yl/yh values, nibble/scale math, accumulation and SIMD reduction.
// Producer costs and its RAW barrier must be included in pair timing.
#ifndef QK_K
#define QK_K 256
#endif
struct probe_q4_sumy_args { uint width; uint pad; };
kernel void probe_q4_prepare_sumy(
        constant probe_q4_sumy_args &args [[buffer(0)]],
        device const float *input [[buffer(1)]],
        device float4 *cache [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint nb = args.width / 256u;
    if (gid >= nb * 8u) return;
    const uint ib = gid / 8u;
    const uint it = gid % 8u;
    const uint iq = it / 4u;
    const uint ir = it % 4u;
    device const float *y4 = input + ib * 256u + 64u * iq + 8u * ir;
    float4 sumy = float4(0.0f);
    for (short i = 0; i < 8; ++i) {
        sumy[0] += y4[i +   0];
        sumy[1] += y4[i +  32];
        sumy[2] += y4[i + 128];
        sumy[3] += y4[i + 160];
    }
    cache[gid] = sumy;
}

template<int nr0, typename args_t = constant ds4_metal_args_mul_mv &, bool ROUND_BF16 = false>
void probe_mul_mv_q4_K_sumy_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        device const float4 *sum_cache,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K *x = (device const block_q4_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};

    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        const float4 sumy = sum_cache[ib * 8 + it];

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0];
            yl[i + 8] = y4[i +  32];
            yh[i + 0] = y4[i + 128];
            yh[i + 8] = y4[i + 160];
        }

        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            if (first_row + row >= args.ne0) break;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = ROUND_BF16 ? dsv4_mv_bf16_store(sum_all) : sum_all;
        }
    }

    (void)shmem;
}

kernel void probe_q4_sumy_f32(
        constant ds4_metal_args_mul_mv &args [[buffer(0)]],
        device const char *weight [[buffer(1)]],
        device const char *input [[buffer(2)]],
        device char *output [[buffer(3)]],
        device const float4 *sum_cache [[buffer(4)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    probe_mul_mv_q4_K_sumy_impl<2, constant ds4_metal_args_mul_mv &, false>(
        args, weight, input, output, sum_cache, shmem, tgpig, tiisg, sgitg);
}

kernel void probe_q4_sumy_bf16(
        constant ds4_metal_args_mul_mv &args [[buffer(0)]],
        device const char *weight [[buffer(1)]],
        device const char *input [[buffer(2)]],
        device char *output [[buffer(3)]],
        device const float4 *sum_cache [[buffer(4)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    probe_mul_mv_q4_K_sumy_impl<2, constant ds4_metal_args_mul_mv &, true>(
        args, weight, input, output, sum_cache, shmem, tgpig, tiisg, sgitg);
}
