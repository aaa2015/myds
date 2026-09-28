/* Native V4.1 routing preserves the pinned ds4 generic 384-expert graph's
 * arithmetic boundaries. No 256-expert V4 routing constants are reused. */
struct myds_v41_route_args { uint count; uint top_k; uint op; float scale; };

kernel void kernel_myds_v41_route_transform(
    constant myds_v41_route_args &args,
    device const float4 *input, device const float4 *bias, device float4 *output,
    uint i [[thread_position_in_grid]]) {
    if (i >= args.count / 4u) return;
    const float4 x = input[i];
    if (args.op == 0u) {
        // Exact softplus expression from the pinned generic unary kernel.
        const float4 ex = exp(x);
        const float4 em = min(ex, (float4)0.03125f);
        const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        output[i] = select(select(log(1 + ex), poly, ex < (float4)0.03125f), x, x > 20);
    } else if (args.op == 1u) {
        output[i] = sqrt(x);
    } else {
        output[i] = x + bias[i];
    }
}

// Dispatch exactly top_k (six) active threads, matching sum_rows(width=6).
// Separate dispatches enforce F32 stores before divide and final scale.
kernel void kernel_myds_v41_route_sum(
    constant myds_v41_route_args &args,
    device const float *probs, device const int *selected,
    device float *weights, device float *denominator,
    threadgroup float *scratch [[threadgroup(0)]],
    ushort tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (simdgroup == 0) scratch[lane] = 0.0f;
    const float p = probs[selected[tid]];
    weights[tid] = p;
    float sumf = 0.0f;
    sumf += p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[simdgroup] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = scratch[lane];
    sumf = simd_sum(sumf);
    if (tid == 0) denominator[0] = clamp(sumf, 6.103515625e-5f, INFINITY);
}

kernel void kernel_myds_v41_route_divide(
    constant myds_v41_route_args &args,
    device float *weights, device const float *denominator,
    uint i [[thread_position_in_grid]]) {
    if (i < args.top_k) weights[i] = weights[i] / denominator[0];
}

kernel void kernel_myds_v41_route_scale(
    constant myds_v41_route_args &args, device float *weights,
    uint i [[thread_position_in_grid]]) {
    if (i < args.top_k) weights[i] = weights[i] * args.scale;
}


kernel void kernel_myds_v41_ids_f32(constant uint &count,
    device const int *ids, device float *out, uint i [[thread_position_in_grid]]) {
    if (i < count) out[i] = float(ids[i]);
}
