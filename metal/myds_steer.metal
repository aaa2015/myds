// myds_steer.metal — GLP Activation Steering Kernels for DeepSeek V4.1
// Implements projective control vector ablation: h <- h - alpha * (h . d) * d
// Supports both single-vector projection (e.g. attention output) and
// hyper-connections residual stream projection (4-copy mean reduction).

struct myds_steer_args {
    uint32_t dim;
    float    alpha;
    uint32_t n_tokens;
};

// 1. Single 1D vector steering: h <- h - alpha * (h . d) * d
// Used for Spec v3 dual hook (at attention o_proj output).
kernel void kernel_myds_steer_vec(
    constant myds_steer_args &args        [[buffer(0)]],
    device   float           *h           [[buffer(1)]],
    device   const float     *dir         [[buffer(2)]],
    threadgroup float        *tg_cache    [[threadgroup(0)]],
    ushort3  tpitg                        [[thread_position_in_threadgroup]],
    ushort   sgitg                        [[simdgroup_index_in_threadgroup]],
    ushort   tiisg                        [[thread_index_in_simdgroup]],
    ushort3  ntg                          [[threads_per_threadgroup]],
    uint3    tg_pos                       [[threadgroup_position_in_grid]])
{
    const uint tok_idx = tg_pos.y;
    const uint dim = args.dim;
    const uint n_vec4 = dim / 4u;
    const float alpha = args.alpha;

    device float4 *h4 = (device float4 *)(h + (uint64_t)tok_idx * dim);
    device const float4 *d4 = (device const float4 *)dir;

    float local_sum = 0.0f;
    for (uint i = tpitg.x; i < n_vec4; i += ntg.x) {
        local_sum += dot(h4[i], d4[i]);
    }

    local_sum = simd_sum(local_sum);
    if (tiisg == 0) {
        tg_cache[sgitg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const ushort n_sg = ntg.x / 32u;
        float s = (tiisg < n_sg) ? tg_cache[tiisg] : 0.0f;
        s = simd_sum(s);
        if (tiisg == 0) {
            tg_cache[0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float coeff = alpha * tg_cache[0];
    for (uint i = tpitg.x; i < n_vec4; i += ntg.x) {
        h4[i] -= coeff * d4[i];
    }
}

// 2. 4-copy Hyper-Connections residual stream steering:
// Computes dot product on the mean of the 4 HC copies:
//   h_mean = 0.25 * (r0 + r1 + r2 + r3)
//   coeff = alpha * (h_mean . d)
// Subtracts (coeff * d) from each of the 4 stream copies r0..r3.
kernel void kernel_myds_steer_hc_residual(
    constant myds_steer_args &args        [[buffer(0)]],
    device   float           *residual    [[buffer(1)]],
    device   const float     *dir         [[buffer(2)]],
    threadgroup float        *tg_cache    [[threadgroup(0)]],
    ushort3  tpitg                        [[thread_position_in_threadgroup]],
    ushort   sgitg                        [[simdgroup_index_in_threadgroup]],
    ushort   tiisg                        [[thread_index_in_simdgroup]],
    ushort3  ntg                          [[threads_per_threadgroup]],
    uint3    tg_pos                       [[threadgroup_position_in_grid]])
{
    const uint tok_idx = tg_pos.y;
    const uint dim = args.dim;
    const uint n_vec4 = dim / 4u;
    const float alpha = args.alpha;

    device float4 *r0 = (device float4 *)(residual + (uint64_t)tok_idx * 4u * dim);
    device float4 *r1 = (device float4 *)(residual + (uint64_t)tok_idx * 4u * dim + dim);
    device float4 *r2 = (device float4 *)(residual + (uint64_t)tok_idx * 4u * dim + 2u * dim);
    device float4 *r3 = (device float4 *)(residual + (uint64_t)tok_idx * 4u * dim + 3u * dim);
    device const float4 *d4 = (device const float4 *)dir;

    float local_sum = 0.0f;
    for (uint i = tpitg.x; i < n_vec4; i += ntg.x) {
        float4 mean_h = 0.25f * (r0[i] + r1[i] + r2[i] + r3[i]);
        local_sum += dot(mean_h, d4[i]);
    }

    local_sum = simd_sum(local_sum);
    if (tiisg == 0) {
        tg_cache[sgitg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const ushort n_sg = ntg.x / 32u;
        float s = (tiisg < n_sg) ? tg_cache[tiisg] : 0.0f;
        s = simd_sum(s);
        if (tiisg == 0) {
            tg_cache[0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float coeff = alpha * tg_cache[0];
    for (uint i = tpitg.x; i < n_vec4; i += ntg.x) {
        float4 sub = coeff * d4[i];
        r0[i] -= sub;
        r1[i] -= sub;
        r2[i] -= sub;
        r3[i] -= sub;
    }
}
