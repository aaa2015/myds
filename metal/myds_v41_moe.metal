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

/* S6 优化：Shared Expert Gate + Up + SwiGLU 三合一融合算子
 * 消除 78 次 GPU Dispatch，消除 shared_gate_out 与 shared_up_out 显存读写往返。
 * 输入激活 x 在寄存器中同时用于 Gate 与 Up 的 Q8_0 点积。
 * 输出直接经过 dsv4_mv_bf16_store -> clamp -> silu * up -> dsv4_mv_bf16_store 写出。
 */
struct myds_v41_shared_swiglu_args {
    uint in_dim;
    uint out_dim;
    uint row_bytes;
    float clamp;
};

template<short NR0>
void kernel_myds_v41_shared_gate_up_swiglu_impl(
        constant myds_v41_shared_swiglu_args &args,
        device const char *src_gate,
        device const char *src_up,
        device const char *src_in,
        device char *dst_out,
        threadgroup char *shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NSG = 4;
    constexpr short NW = N_SIMDWIDTH; // 32
    constexpr short NQ = 8;

    const int nb = (int)args.in_dim / QK8_0;
    const int r0 = tgpig.x * NR0;

    device const float *y = (device const float *)src_in;

    device const block_q8_0 *ax_gate[NR0];
    device const block_q8_0 *ax_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (uint64_t)(r0 + row) * args.row_bytes;
        const bool active = (r0 + row < (int)args.out_dim);
        ax_gate[row] = (device const block_q8_0 *)(src_gate + (active ? offset0 : 0));
        ax_up[row]   = (device const block_q8_0 *)(src_up   + (active ? offset0 : 0));
    }

    float sumf_gate[NR0] = { 0.f };
    float sumf_up[NR0]   = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);

    const int ib0 = sgitg * NQ + ix;

    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; ++row) {
            device const int8_t *qs_g = ax_gate[row][ib].qs + il * NQ;
            device const int8_t *qs_u = ax_up[row][ib].qs   + il * NQ;

            float sumq_g = 0.f;
            float sumq_u = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq_g += qs_g[i] * yl[i];
                sumq_u += qs_u[i] * yl[i];
            }

            sumf_gate[row] += sumq_g * ax_gate[row][ib].d;
            sumf_up[row]   += sumq_u * ax_up[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    // 线程组内双规约 (Gate 和 Up 各占 NR0 * NW 个 float)
    threadgroup float *shmem_gate[NR0];
    threadgroup float *shmem_up[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_gate[row] = (threadgroup float *)shmem + NW * row;
        shmem_up[row]   = (threadgroup float *)shmem + NW * (NR0 + row);

        if (sgitg == 0) {
            shmem_gate[row][tiisg] = 0.0f;
            shmem_up[row][tiisg]   = 0.0f;
        }

        sumf_gate[row] = simd_sum(sumf_gate[row]);
        sumf_up[row]   = simd_sum(sumf_up[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_gate[row][sgitg] = sumf_gate[row];
            shmem_up[row][sgitg]   = sumf_up[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *dst_f32 = (device float *)dst_out;

    for (short row = 0; row < NR0 && r0 + row < (int)args.out_dim; ++row) {
        float tot_g = simd_sum(shmem_gate[row][tiisg]);
        float tot_u = simd_sum(shmem_up[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            float g = dsv4_mv_bf16_store(tot_g);
            float u = dsv4_mv_bf16_store(tot_u);

            if (args.clamp > 1.0e-6f) {
                g = min(g, args.clamp);
                u = clamp(u, -args.clamp, args.clamp);
            }

            const float silu = g / (1.0f + exp(-g));
            const float glu_v = silu * u;
            dst_f32[r0 + row] = dsv4_mv_bf16_store(glu_v);
        }
    }
}

kernel void kernel_myds_v41_shared_gate_up_swiglu(
    constant myds_v41_shared_swiglu_args &args [[buffer(0)]],
    device const char *src_gate [[buffer(1)]],
    device const char *src_up [[buffer(2)]],
    device const char *src_in [[buffer(3)]],
    device char *dst_out [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_myds_v41_shared_gate_up_swiglu_impl<2>(
        args, src_gate, src_up, src_in, dst_out, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_myds_v41_shared_gate_up_swiglu_r4(
    constant myds_v41_shared_swiglu_args &args [[buffer(0)]],
    device const char *src_gate [[buffer(1)]],
    device const char *src_up [[buffer(2)]],
    device const char *src_in [[buffer(3)]],
    device char *dst_out [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_myds_v41_shared_gate_up_swiglu_impl<4>(
        args, src_gate, src_up, src_in, dst_out, shmem, tgpig, tiisg, sgitg);
}
