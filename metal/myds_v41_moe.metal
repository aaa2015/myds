/* Native V4.1 routing preserves the pinned ds4 generic 384-expert graph's
 * arithmetic boundaries. No 256-expert V4 routing constants are reused. */
/* count = rows*row (总元素数); row = 每行专家数; row4 = row/4; rows = token 数。rows==1 时与旧单行语义完全一致。 */
struct myds_v41_route_args { uint count; uint top_k; uint op; float scale; uint rows; uint row; uint row4; };

kernel void kernel_myds_v41_route_transform(
    constant myds_v41_route_args &args,
    device const float4 *input, device const float4 *bias, device float4 *output,
    uint i [[thread_position_in_grid]]) {
    if (i >= args.count / 4u) return;
    const float4 x = input[i];
    const uint b4 = i % args.row4;   /* bias 只有一行，按行内位置取 */
    if (args.op == 0u) {
        // Exact softplus expression from the pinned generic unary kernel.
        const float4 ex = exp(x);
        const float4 em = min(ex, (float4)0.03125f);
        const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        output[i] = select(select(log(1 + ex), poly, ex < (float4)0.03125f), x, x > 20);
    } else if (args.op == 1u) {
        output[i] = sqrt(x);
    } else {
        output[i] = x + bias[b4];
    }
}

kernel void kernel_myds_v41_route_prepare(
    constant myds_v41_route_args &args,
    device const float4 *logits, device const float4 *bias,
    device float4 *probs, device float4 *scores,
    uint i [[thread_position_in_grid]]) {
    if (i >= args.count / 4u) return;
    const float4 x = logits[i];
    const uint b4 = i % args.row4;   /* bias 只有一行，按行内位置取 */
    const float4 ex = exp(x);
    const float4 em = min(ex, (float4)0.03125f);
    const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
    const float4 sp = select(select(log(1 + ex), poly, ex < (float4)0.03125f), x, x > 20);
    const float4 pr = sqrt(sp);
    probs[i] = pr;
    scores[i] = pr + bias[b4];
}

// Dispatch exactly top_k (six) active threads, matching sum_rows(width=6).
// Separate dispatches enforce F32 stores before divide and final scale.
kernel void kernel_myds_v41_route_sum(
    constant myds_v41_route_args &args,
    device const float *probs, device const int *selected,
    device float *weights, device float *denominator,
    threadgroup float *scratch [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint t = tgid;                     /* 每个 threadgroup 负责一个 token */
    if (t >= args.rows) return;
    if (simdgroup == 0) scratch[lane] = 0.0f;
    const float p = probs[t * args.row + selected[t * args.top_k + tid]];
    weights[t * args.top_k + tid] = p;
    float sumf = 0.0f;
    sumf += p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[simdgroup] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = scratch[lane];
    sumf = simd_sum(sumf);
    if (tid == 0) denominator[t] = clamp(sumf, 6.103515625e-5f, INFINITY);
}

kernel void kernel_myds_v41_route_divide(
    constant myds_v41_route_args &args,
    device float *weights, device const float *denominator,
    uint i [[thread_position_in_grid]]) {
    if (i < args.top_k * args.rows) weights[i] = weights[i] / denominator[i / args.top_k];
}

kernel void kernel_myds_v41_route_scale(
    constant myds_v41_route_args &args, device float *weights,
    uint i [[thread_position_in_grid]]) {
    if (i < args.top_k * args.rows) weights[i] = weights[i] * args.scale;
}

kernel void kernel_myds_v41_route_finish(
    constant myds_v41_route_args &args,
    device const float *probs, device const int *selected,
    device float *weights, device float *denominator,
    threadgroup float *scratch [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint t = tgid;                     /* 每个 threadgroup 负责一个 token */
    if (t >= args.rows) return;
    if (simdgroup == 0) scratch[lane] = 0.0f;
    const float p = (tid < args.top_k) ? probs[t * args.row + selected[t * args.top_k + tid]] : 0.0f;
    float sumf = 0.0f;
    sumf += p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[simdgroup] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = scratch[lane];
    sumf = simd_sum(sumf);
    const float denom = clamp(sumf, 6.103515625e-5f, INFINITY);
    if (tid == 0 && denominator) denominator[t] = denom;
    if (tid < args.top_k) {
        weights[t * args.top_k + tid] = p / denom;
        threadgroup_barrier(mem_flags::mem_device);
        weights[t * args.top_k + tid] = weights[t * args.top_k + tid] * args.scale;
    }
}


kernel void kernel_myds_v41_ids_f32(constant uint &count,
    device const int *ids, device float *out, uint i [[thread_position_in_grid]]) {
    if (i < count) out[i] = float(ids[i]);
}
