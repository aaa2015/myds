// Default-off, append-only decode-weight representation probe.
// Native vector4 K traversal and helper_mv_reduce_and_write are retained.
// Host admission: one token, NSG8, NR1 raw / NR2 fused-BF16, full contiguous
// qualified tensors only; original F16/F32 weights remain available for prefill.
// E4M3 scales are finite positive powers of two selected by the exact CPU audit.

struct myds_narrow_expand_args {
    uint32_t cols;
    uint32_t rows;
    uint64_t count;
};

// The result is a normal F32 or signed zero for every qualified F16 weight.
// Construct bits before dot: fast-math cannot move a floating scale into x/dot.
static inline uint myds_narrow_e4m3_f32_bits(uchar code, int scale_exp) {
    const uint sign = uint(code & 0x80u) << 24u;
    const uint mag = uint(code & 0x7fu);
    if (mag == 0u) return sign;
    if (mag == 0x7fu) return sign | 0x7fc00000u;
    const uint e = mag >> 3u;
    const uint m = mag & 7u;
    if (e != 0u) return sign | (uint(scale_exp + int(e) - 7) << 23u) | (m << 20u);
    const int shift = m >= 4u ? 2 : m >= 2u ? 1 : 0;
    return sign | (uint(scale_exp + shift - 9) << 23u) |
           ((m - (1u << uint(shift))) << uint(23 - shift));
}

static inline float4 myds_narrow_e4m3_dec4(uint packed, int scale_exp) {
    return as_type<float4>(uint4(
        myds_narrow_e4m3_f32_bits(uchar(packed), scale_exp),
        myds_narrow_e4m3_f32_bits(uchar(packed >> 8u), scale_exp),
        myds_narrow_e4m3_f32_bits(uchar(packed >> 16u), scale_exp),
        myds_narrow_e4m3_f32_bits(uchar(packed >> 24u), scale_exp)));
}

static inline float4 myds_narrow_bf16_dec4(uint2 packed) {
    return as_type<float4>(uint4(packed.x << 16u, packed.x & 0xffff0000u,
                                packed.y << 16u, packed.y & 0xffff0000u));
}

// Integer F32->F16 bit packing for exact F16-valued audit inputs. This avoids
// letting diagnostic half arithmetic flush a source half subnormal to zero.
static inline ushort myds_narrow_exact_f16_bits(uint bits) {
    const uint sign = (bits >> 16u) & 0x8000u;
    const uint e = (bits >> 23u) & 255u;
    const uint m = bits & 0x7fffffu;
    if ((bits & 0x7fffffffu) == 0u) return ushort(sign);
    if (e >= 113u && e <= 142u && (m & 0x1fffu) == 0u)
        return ushort(sign | ((e - 112u) << 10u) | (m >> 13u));
    if (e >= 103u && e <= 112u) {
        const uint shift = 126u - e, significand = m | 0x800000u;
        if ((significand & ((1u << shift) - 1u)) == 0u)
            return ushort(sign | (significand >> shift));
    }
    return ushort(sign | 0x7e00u); // Invalid admission cannot masquerade as valid F16.
}

kernel void kernel_myds_narrow_expand_router_f32(
        constant myds_narrow_expand_args &args [[buffer(0)]],
        device const ushort *codes [[buffer(1)]],
        device const float *scales [[buffer(2)]],
        device uint *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    (void)scales;
    if (ulong(i) < args.count) out[i] = uint(codes[i]) << 16u;
}

kernel void kernel_myds_narrow_expand_e4m3_f32(
        constant myds_narrow_expand_args &args [[buffer(0)]],
        device const uchar *codes [[buffer(1)]],
        device const float *scales [[buffer(2)]],
        device uint *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (ulong(i) < args.count) {
        const ulong row = ulong(i) / args.cols, col = ulong(i) % args.cols;
        const uint scale_bits = as_type<uint>(scales[(row / 32u) * (args.cols / 32u) + col / 32u]);
        out[i] = myds_narrow_e4m3_f32_bits(codes[i], int((scale_bits >> 23u) & 255u));
    }
}

kernel void kernel_myds_narrow_expand_e4m3_f16(
        constant myds_narrow_expand_args &args [[buffer(0)]],
        device const uchar *codes [[buffer(1)]],
        device const float *scales [[buffer(2)]],
        device ushort *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (ulong(i) < args.count) {
        const ulong row = ulong(i) / args.cols, col = ulong(i) % args.cols;
        const uint scale_bits = as_type<uint>(scales[(row / 32u) * (args.cols / 32u) + col / 32u]);
        const uint bits = myds_narrow_e4m3_f32_bits(codes[i], int((scale_bits >> 23u) & 255u));
        out[i] = myds_narrow_exact_f16_bits(bits);
    }
}

template<short NR0, bool E4M3, bool ROUND_BF16>
static inline void myds_narrow_decode_mv(
        constant ds4_metal_args_mul_mv &args,
        device const uchar *codes, device const char *src1,
        device char *dst, device const float *scales, threadgroup char *shmem,
        uint3 tgpig, ushort tiisg, ushort sgitg) {
    constexpr short NSG = 8;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF / 4;
    const int nb = args.ne00 / NB;
    const int r0 = int(tgpig.x) * NR0;
    device const float4 *y4 = (device const float4 *)src1;
    float sumf[NR0] = {0.f};
    const short ix = tiisg / (NW / NF);
    const short il = tiisg % (NW / NF);
    const int ib0 = sgitg * NF + ix;
    float4 yl4[NF4];
    device const float4 *yb4 = y4 + (ib0 * NB + il * NF) / 4;
    for (int ib = ib0; ib < nb; ib += NSG * NF) {
        for (short i = 0; i < NF4; ++i) yl4[i] = yb4[i];
        // NR2 starts on an even row, so both rows share one 32-row scale block.
        int scale_exp = 0;
        if (E4M3) scale_exp = int((as_type<uint>(scales[(ulong(r0) / 32u) * uint(nb) + uint(ib)]) >> 23u) & 255u);
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            const ulong offset = ulong(r0 + row) * args.nb01 + ulong(ib * NB + il * NF) * args.nb00;
            device const uint4 *xb = (device const uint4 *)(codes + offset);
            float4 w4[NF4];
            if (E4M3) {
                const uint4 p = xb[0];
                FOR_UNROLL (short i = 0; i < NF4; ++i) w4[i] = myds_narrow_e4m3_dec4(p[i], scale_exp);
            } else {
                const uint4 a = xb[0], b = xb[1];
                const uint2 p[NF4] = {a.xy, a.zw, b.xy, b.zw};
                FOR_UNROLL (short i = 0; i < NF4; ++i) w4[i] = myds_narrow_bf16_dec4(p[i]);
            }
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(w4[i]), float4(yl4[i]));
            }
            sumf[row] += sumq;
        }
        yb4 += NSG * NF * NW / 4;
    }
    // Full scoped widths are multiples of32; native tail loop has zero iterations.
    helper_mv_reduce_and_write<NR0, ROUND_BF16>((device float *)dst, sumf,
                                               r0, args.ne01, tiisg, sgitg, shmem);
}

#define MYDS_NARROW_MV(NAME, NR, E4, ROUND) \
kernel void NAME(constant ds4_metal_args_mul_mv &args [[buffer(0)]], \
        device const uchar *codes [[buffer(1)]], device const char *src1 [[buffer(2)]], \
        device char *dst [[buffer(3)]], device const float *scales [[buffer(4)]], \
        threadgroup char *shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    myds_narrow_decode_mv<NR,E4,ROUND>(args,codes,src1,dst,scales,shmem,tgpig,tiisg,sgitg); \
}
MYDS_NARROW_MV(kernel_myds_narrow_router_bf16_nr1, 1, false, false)
MYDS_NARROW_MV(kernel_myds_narrow_e4m3_f32_nr1, 1, true, false)
MYDS_NARROW_MV(kernel_myds_narrow_e4m3_f32_nr2, 2, true, false)
MYDS_NARROW_MV(kernel_myds_narrow_e4m3_bf16_nr2, 2, true, true)
#undef MYDS_NARROW_MV
