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

/* S5 优化：单 Threadgroup 全融合 Route 内核（Prepare + Argsort + Finish 一体化）
 * 将 3 次 GPU Kernel 派发与 2 次全局内存屏障融合成单次派发，消除中间显存往返。
 * 共享内存布局：512 ints (索引) + 512 floats (排序分数) + 384 floats (概率) + 32 floats (归约)，共 5760 字节。 */
kernel void kernel_myds_v41_route_fused(
    constant myds_v41_route_args &args,
    device const float4 *logits,
    device const float4 *bias,
    device float4 *probs,
    device float4 *scores,
    device int *selected,
    device float *weights,
    threadgroup char *shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint t = tgid;
    if (t >= args.rows) return;

    threadgroup int *sh_indices = (threadgroup int *)shmem;
    threadgroup float *sh_scores = (threadgroup float *)(sh_indices + 512);
    threadgroup float *sh_probs = (threadgroup float *)(sh_scores + 512);
    threadgroup float *scratch = (threadgroup float *)(sh_probs + 384);

    // 1. 阶段一：计算 softplus -> sqrt -> + bias
    // 384 个专家对应 96 个 float4
    if (tid < 96u) {
        const uint i = t * 96u + tid;
        const float4 x = logits[i];
        const float4 b = bias[tid];
        const float4 ex = exp(x);
        const float4 em = min(ex, (float4)0.03125f);
        const float4 poly = em * (1.0f - em * (0.5f - em * (1.0f / 3.0f - 0.25f * em)));
        const float4 sp = select(select(log(1 + ex), poly, ex < (float4)0.03125f), x, x > 20);
        const float4 pr = sqrt(sp);
        const float4 sc = pr + b;
        if (probs) probs[i] = pr;
        if (scores) scores[i] = sc;
        ((threadgroup float4 *)sh_probs)[tid] = pr;
        ((threadgroup float4 *)sh_scores)[tid] = sc;
    }
    // 512 线程中 >= 384 的位置填入 -INFINITY
    if (tid >= 384u) {
        sh_scores[tid] = -INFINITY;
    }
    sh_indices[tid] = (int)tid;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. 阶段二：双调排序（Bitonic Sort Descending，降序）
    // 512 个元素双调网络，与 kernel_argsort_f32_i32 完全一致的比较拓扑与平手规则
    for (int k = 2; k <= 512; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = (int)tid ^ j;
            if (ixj > (int)tid) {
                if (((int)tid & k) == 0) {
                    if (sh_indices[tid] >= 384 ||
                       (sh_indices[ixj] < 384 && sh_scores[sh_indices[tid]] < sh_scores[sh_indices[ixj]])) {
                        int tmp = sh_indices[tid];
                        sh_indices[tid] = sh_indices[ixj];
                        sh_indices[ixj] = tmp;
                    }
                } else {
                    if (sh_indices[ixj] >= 384 ||
                       (sh_indices[tid] < 384 && sh_scores[sh_indices[tid]] > sh_scores[sh_indices[ixj]])) {
                        int tmp = sh_indices[tid];
                        sh_indices[tid] = sh_indices[ixj];
                        sh_indices[ixj] = tmp;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 3. 阶段三：提取 Top-K 并归一化写出
    const uint top_k = args.top_k;
    if (sg == 0) scratch[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float p = 0.0f;
    if (tid < top_k) {
        const int best_id = sh_indices[tid];
        selected[t * top_k + tid] = best_id;
        p = sh_probs[best_id];
    }

    float sumf = p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = scratch[lane];
    sumf = simd_sum(sumf);
    const float denom = clamp(sumf, 6.103515625e-5f, INFINITY);

    if (tid < top_k) {
        weights[t * top_k + tid] = p / denom;
        threadgroup_barrier(mem_flags::mem_device);
        weights[t * top_k + tid] = weights[t * top_k + tid] * args.scale;
    }
}
