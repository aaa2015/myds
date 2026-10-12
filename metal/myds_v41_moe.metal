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
    }
    // Preserve the F32 divide/store/load/scale boundary for every threadgroup.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < args.top_k) {
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
    }
    // All 512 threads must reach the barrier, including the 506 inactive lanes.
    // Keep divide -> device F32 store -> load -> scale in the original order.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < top_k) {
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

template<short NR0>
void kernel_myds_v41_shared_gate_up_swiglu_q4_impl(
        constant myds_v41_shared_swiglu_args &args,
        device const char *src_gate,
        device const char *src_up,
        device const float *src_in,
        device float *dst_out,
        threadgroup char *shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NSG = 4;
    constexpr int QK_K = 256;
    const int nb = (int)args.in_dim / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    if (first_row >= (int)args.out_dim) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    device const float *y = src_in;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    device const block_q4_K *ax_gate[NR0];
    device const block_q4_K *ax_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (uint64_t)(first_row + row) * args.row_bytes;
        const bool active = (first_row + row < (int)args.out_dim);
        ax_gate[row] = (device const block_q4_K *)(src_gate + (active ? offset0 : 0));
        ax_up[row]   = (device const block_q4_K *)(src_up   + (active ? offset0 : 0));
    }

    float yl[16];
    float yh[16];
    float sumf_gate[NR0] = { 0.f };
    float sumf_up[NR0]   = { 0.f };

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        FOR_UNROLL (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (first_row + row >= (int)args.out_dim) break;

            // Gate dot product
            device const uint16_t *sc_g = (device const uint16_t *)ax_gate[row][ib].scales + iq;
            device const uint16_t *q1_g = (device const uint16_t *)ax_gate[row][ib].qs + 16 * iq + 4 * ir;
            device const half *dh_g = &ax_gate[row][ib].d;

            sc16[0] = sc_g[0] & kmask1;
            sc16[1] = sc_g[2] & kmask1;
            sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
            sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);

            device const uint16_t *q2_g = q1_g + 32;

            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (q1_g[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (q1_g[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (q1_g[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (q1_g[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (q2_g[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (q2_g[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (q2_g[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (q2_g[i] & 0xF000);
            }

            sumf_gate[row] += dh_g[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                         (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                         (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                         (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                              dh_g[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            // Up dot product
            device const uint16_t *sc_u = (device const uint16_t *)ax_up[row][ib].scales + iq;
            device const uint16_t *q1_u = (device const uint16_t *)ax_up[row][ib].qs + 16 * iq + 4 * ir;
            device const half *dh_u = &ax_up[row][ib].d;

            sc16[0] = sc_u[0] & kmask1;
            sc16[1] = sc_u[2] & kmask1;
            sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
            sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);

            device const uint16_t *q2_u = q1_u + 32;

            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (q1_u[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (q1_u[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (q1_u[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (q1_u[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (q2_u[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (q2_u[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (q2_u[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (q2_u[i] & 0xF000);
            }

            sumf_up[row] += dh_u[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                       (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                       (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                       (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                            dh_u[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0 && first_row + row < (int)args.out_dim; ++row) {
        float tot_g = simd_sum(sumf_gate[row]);
        float tot_u = simd_sum(sumf_up[row]);

        if (tiisg == 0) {
            float g = dsv4_mv_bf16_store(tot_g);
            float u = dsv4_mv_bf16_store(tot_u);

            if (args.clamp > 1.0e-6f) {
                g = min(g, args.clamp);
                u = clamp(u, -args.clamp, args.clamp);
            }

            const float silu = g / (1.0f + exp(-g));
            const float glu_v = silu * u;
            dst_out[first_row + row] = dsv4_mv_bf16_store(glu_v);
        }
    }
    (void)shmem;
}

kernel void kernel_myds_v41_shared_gate_up_swiglu_q4(
    constant myds_v41_shared_swiglu_args &args [[buffer(0)]],
    device const char *src_gate [[buffer(1)]],
    device const char *src_up [[buffer(2)]],
    device const float *src_in [[buffer(3)]],
    device float *dst_out [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_myds_v41_shared_gate_up_swiglu_q4_impl<2>(
        args, src_gate, src_up, src_in, dst_out, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_myds_v41_shared_gate_up_swiglu_q4_r4(
    constant myds_v41_shared_swiglu_args &args [[buffer(0)]],
    device const char *src_gate [[buffer(1)]],
    device const char *src_up [[buffer(2)]],
    device const float *src_in [[buffer(3)]],
    device float *dst_out [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_myds_v41_shared_gate_up_swiglu_q4_impl<4>(
        args, src_gate, src_up, src_in, dst_out, shmem, tgpig, tiisg, sgitg);
}

template<short NR0>
void kernel_myds_v41_shared_gate_up_swiglu_q4_shared2_impl(
        constant myds_v41_shared_swiglu_args &args,
        device const char *src_gate,
        device const char *src_up,
        device const float *src_in,
        device float *dst_out,
        threadgroup char *shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NSG = 4;
    constexpr int QK_K = 256;
    const int nb = (int)args.in_dim / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    if (first_row >= (int)args.out_dim) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    device const float *y0 = src_in;
    device const float *y0_4 = y0 + ix * QK_K + 64 * iq + 8 * ir;
    device const float *y1 = src_in + args.in_dim;
    device const float *y1_4 = y1 + ix * QK_K + 64 * iq + 8 * ir;

    device const block_q4_K *ax_gate[NR0];
    device const block_q4_K *ax_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (uint64_t)(first_row + row) * args.row_bytes;
        const bool active = (first_row + row < (int)args.out_dim);
        ax_gate[row] = (device const block_q4_K *)(src_gate + (active ? offset0 : 0));
        ax_up[row]   = (device const block_q4_K *)(src_up   + (active ? offset0 : 0));
    }

    float yl0[16], yh0[16];
    float yl1[16], yh1[16];
    float sumf_gate0[NR0] = { 0.f };
    float sumf_up0[NR0]   = { 0.f };
    float sumf_gate1[NR0] = { 0.f };
    float sumf_up1[NR0]   = { 0.f };

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy0 = {0.f, 0.f, 0.f, 0.f};
        float4 sumy1 = {0.f, 0.f, 0.f, 0.f};

        FOR_UNROLL (short i = 0; i < 8; ++i) {
            yl0[i + 0] = y0_4[i +   0]; sumy0[0] += yl0[i + 0];
            yl0[i + 8] = y0_4[i +  32]; sumy0[1] += yl0[i + 8];
            yh0[i + 0] = y0_4[i + 128]; sumy0[2] += yh0[i + 0];
            yh0[i + 8] = y0_4[i + 160]; sumy0[3] += yh0[i + 8];

            yl1[i + 0] = y1_4[i +   0]; sumy1[0] += yl1[i + 0];
            yl1[i + 8] = y1_4[i +  32]; sumy1[1] += yl1[i + 8];
            yh1[i + 0] = y1_4[i + 128]; sumy1[2] += yh1[i + 0];
            yh1[i + 8] = y1_4[i + 160]; sumy1[3] += yh1[i + 8];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (first_row + row >= (int)args.out_dim) break;

            // Gate dot product
            device const uint16_t *sc_g = (device const uint16_t *)ax_gate[row][ib].scales + iq;
            device const uint16_t *q1_g = (device const uint16_t *)ax_gate[row][ib].qs + 16 * iq + 4 * ir;
            device const half *dh_g = &ax_gate[row][ib].d;

            sc16[0] = sc_g[0] & kmask1;
            sc16[1] = sc_g[2] & kmask1;
            sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
            sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);

            device const uint16_t *q2_g = q1_g + 32;

            float4 acc1g0 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g0 = {0.f, 0.f, 0.f, 0.f};
            float4 acc1g1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g1 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const uint16_t q1 = q1_g[i];
                const uint16_t q2 = q2_g[i];
                const float m0 = (float)(q1 & 0x000F);
                const float m1 = (float)(q1 & 0x0F00);
                const float m2 = (float)(q1 & 0x00F0);
                const float m3 = (float)(q1 & 0xF000);
                const float m4 = (float)(q2 & 0x000F);
                const float m5 = (float)(q2 & 0x0F00);
                const float m6 = (float)(q2 & 0x00F0);
                const float m7 = (float)(q2 & 0xF000);

                acc1g0[0] += yl0[2 * i + 0] * m0;
                acc1g0[1] += yl0[2 * i + 1] * m1;
                acc1g0[2] += yl0[2 * i + 8] * m2;
                acc1g0[3] += yl0[2 * i + 9] * m3;
                acc2g0[0] += yh0[2 * i + 0] * m4;
                acc2g0[1] += yh0[2 * i + 1] * m5;
                acc2g0[2] += yh0[2 * i + 8] * m6;
                acc2g0[3] += yh0[2 * i + 9] * m7;

                acc1g1[0] += yl1[2 * i + 0] * m0;
                acc1g1[1] += yl1[2 * i + 1] * m1;
                acc1g1[2] += yl1[2 * i + 8] * m2;
                acc1g1[3] += yl1[2 * i + 9] * m3;
                acc2g1[0] += yh1[2 * i + 0] * m4;
                acc2g1[1] += yh1[2 * i + 1] * m5;
                acc2g1[2] += yh1[2 * i + 8] * m6;
                acc2g1[3] += yh1[2 * i + 9] * m7;
            }

            sumf_gate0[row] += dh_g[0] * ((acc1g0[0] + 1.f / 256.f * acc1g0[1]) * sc8[0] +
                                          (acc1g0[2] + 1.f / 256.f * acc1g0[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2g0[0] + 1.f / 256.f * acc2g0[1]) * sc8[4] +
                                          (acc2g0[2] + 1.f / 256.f * acc2g0[3]) * sc8[5] * 1.f / 16.f) -
                               dh_g[1] * (sumy0[0] * sc8[2] + sumy0[1] * sc8[3] + sumy0[2] * sc8[6] + sumy0[3] * sc8[7]);

            sumf_gate1[row] += dh_g[0] * ((acc1g1[0] + 1.f / 256.f * acc1g1[1]) * sc8[0] +
                                          (acc1g1[2] + 1.f / 256.f * acc1g1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2g1[0] + 1.f / 256.f * acc2g1[1]) * sc8[4] +
                                          (acc2g1[2] + 1.f / 256.f * acc2g1[3]) * sc8[5] * 1.f / 16.f) -
                               dh_g[1] * (sumy1[0] * sc8[2] + sumy1[1] * sc8[3] + sumy1[2] * sc8[6] + sumy1[3] * sc8[7]);

            // Up dot product
            device const uint16_t *sc_u = (device const uint16_t *)ax_up[row][ib].scales + iq;
            device const uint16_t *q1_u = (device const uint16_t *)ax_up[row][ib].qs + 16 * iq + 4 * ir;
            device const half *dh_u = &ax_up[row][ib].d;

            sc16[0] = sc_u[0] & kmask1;
            sc16[1] = sc_u[2] & kmask1;
            sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
            sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);

            device const uint16_t *q2_u = q1_u + 32;

            float4 acc1u0 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u0 = {0.f, 0.f, 0.f, 0.f};
            float4 acc1u1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u1 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const uint16_t q1 = q1_u[i];
                const uint16_t q2 = q2_u[i];
                const float m0 = (float)(q1 & 0x000F);
                const float m1 = (float)(q1 & 0x0F00);
                const float m2 = (float)(q1 & 0x00F0);
                const float m3 = (float)(q1 & 0xF000);
                const float m4 = (float)(q2 & 0x000F);
                const float m5 = (float)(q2 & 0x0F00);
                const float m6 = (float)(q2 & 0x00F0);
                const float m7 = (float)(q2 & 0xF000);

                acc1u0[0] += yl0[2 * i + 0] * m0;
                acc1u0[1] += yl0[2 * i + 1] * m1;
                acc1u0[2] += yl0[2 * i + 8] * m2;
                acc1u0[3] += yl0[2 * i + 9] * m3;
                acc2u0[0] += yh0[2 * i + 0] * m4;
                acc2u0[1] += yh0[2 * i + 1] * m5;
                acc2u0[2] += yh0[2 * i + 8] * m6;
                acc2u0[3] += yh0[2 * i + 9] * m7;

                acc1u1[0] += yl1[2 * i + 0] * m0;
                acc1u1[1] += yl1[2 * i + 1] * m1;
                acc1u1[2] += yl1[2 * i + 8] * m2;
                acc1u1[3] += yl1[2 * i + 9] * m3;
                acc2u1[0] += yh1[2 * i + 0] * m4;
                acc2u1[1] += yh1[2 * i + 1] * m5;
                acc2u1[2] += yh1[2 * i + 8] * m6;
                acc2u1[3] += yh1[2 * i + 9] * m7;
            }

            sumf_up0[row] += dh_u[0] * ((acc1u0[0] + 1.f / 256.f * acc1u0[1]) * sc8[0] +
                                        (acc1u0[2] + 1.f / 256.f * acc1u0[3]) * sc8[1] * 1.f / 16.f +
                                        (acc2u0[0] + 1.f / 256.f * acc2u0[1]) * sc8[4] +
                                        (acc2u0[2] + 1.f / 256.f * acc2u0[3]) * sc8[5] * 1.f / 16.f) -
                             dh_u[1] * (sumy0[0] * sc8[2] + sumy0[1] * sc8[3] + sumy0[2] * sc8[6] + sumy0[3] * sc8[7]);

            sumf_up1[row] += dh_u[0] * ((acc1u1[0] + 1.f / 256.f * acc1u1[1]) * sc8[0] +
                                        (acc1u1[2] + 1.f / 256.f * acc1u1[3]) * sc8[1] * 1.f / 16.f +
                                        (acc2u1[0] + 1.f / 256.f * acc2u1[1]) * sc8[4] +
                                        (acc2u1[2] + 1.f / 256.f * acc2u1[3]) * sc8[5] * 1.f / 16.f) -
                             dh_u[1] * (sumy1[0] * sc8[2] + sumy1[1] * sc8[3] + sumy1[2] * sc8[6] + sumy1[3] * sc8[7]);
        }

        y0_4 += 4 * QK_K;
        y1_4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0 && first_row + row < (int)args.out_dim; ++row) {
        float tot_g0 = simd_sum(sumf_gate0[row]);
        float tot_u0 = simd_sum(sumf_up0[row]);
        float tot_g1 = simd_sum(sumf_gate1[row]);
        float tot_u1 = simd_sum(sumf_up1[row]);

        if (tiisg == 0) {
            float g0 = dsv4_mv_bf16_store(tot_g0);
            float u0 = dsv4_mv_bf16_store(tot_u0);
            float g1 = dsv4_mv_bf16_store(tot_g1);
            float u1 = dsv4_mv_bf16_store(tot_u1);

            if (args.clamp > 1.0e-6f) {
                g0 = min(g0, args.clamp);
                u0 = clamp(u0, -args.clamp, args.clamp);
                g1 = min(g1, args.clamp);
                u1 = clamp(u1, -args.clamp, args.clamp);
            }

            const float silu0 = g0 / (1.0f + exp(-g0));
            const float glu0 = silu0 * u0;
            dst_out[first_row + row] = dsv4_mv_bf16_store(glu0);

            const float silu1 = g1 / (1.0f + exp(-g1));
            const float glu1 = silu1 * u1;
            dst_out[args.out_dim + first_row + row] = dsv4_mv_bf16_store(glu1);
        }
    }
    (void)shmem;
}

kernel void kernel_myds_v41_shared_gate_up_swiglu_q4_shared2(
    constant myds_v41_shared_swiglu_args &args [[buffer(0)]],
    device const char *src_gate [[buffer(1)]],
    device const char *src_up [[buffer(2)]],
    device const float *src_in [[buffer(3)]],
    device float *dst_out [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_myds_v41_shared_gate_up_swiglu_q4_shared2_impl<2>(
        args, src_gate, src_up, src_in, dst_out, shmem, tgpig, tiisg, sgitg);
}

/* Distinct three-input candidate. Keep the scalar Q4 lane/block assignment,
 * ib += 4, dot expression and BF16 boundaries. Quantized Gate/Up operands are
 * loaded outside the token loop, so the same values serve all three inputs.
 * Neither this source order nor a component PASS substitutes for the required
 * real-model checkpoint/partial2 byte-parity gate. */
/* Give each token one scalar-shaped block-dot compilation scope. Weight
 * words and scales remain thread-local values loaded once outside the token
 * loop. The explicit noinline boundary preserves the local operation shape;
 * no global fast-math or scalar PSO contract changes. */
__attribute__((noinline))
float myds_v41_gu_shared_q4_scalar_block(float previous, thread const float *yl, thread const float *yh,
    float4 sumy, thread const ushort *q1, thread const ushort *q2,
    thread const uchar *sc8, half d, half dm) {
    float4 acc1={0.f,0.f,0.f,0.f},acc2={0.f,0.f,0.f,0.f};
    FOR_UNROLL(short i=0;i<4;++i) {
        acc1[0] += yl[2*i] * (q1[i]&0x000F);
        acc1[1] += yl[2*i+1] * (q1[i]&0x0F00);
        acc1[2] += yl[2*i+8] * (q1[i]&0x00F0);
        acc1[3] += yl[2*i+9] * (q1[i]&0xF000);
        acc2[0] += yh[2*i] * (q2[i]&0x000F);
        acc2[1] += yh[2*i+1] * (q2[i]&0x0F00);
        acc2[2] += yh[2*i+8] * (q2[i]&0x00F0);
        acc2[3] += yh[2*i+9] * (q2[i]&0xF000);
    }
    previous += d*((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                   (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                   (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                   (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                dm*(sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
    return previous;
}
kernel void kernel_myds_v41_shared_gate_up_swiglu_q4_shared3(
    constant myds_v41_shared_swiglu_args &args [[buffer(0)]],
    device const char *src_gate [[buffer(1)]],
    device const char *src_up [[buffer(2)]],
    device const float *src_in0 [[buffer(3)]],
    device const float *src_in1 [[buffer(4)]],
    device const float *src_in2 [[buffer(5)]],
    device float *dst_out0 [[buffer(6)]],
    device float *dst_out1 [[buffer(7)]],
    device float *dst_out2 [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 4;
    constexpr short NR0 = 2;
    const int nb = (int)args.in_dim / 256;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    if (first_row >= (int)args.out_dim) return;
    constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const short ix = tiisg / 8, it = tiisg % 8;
    const short iq = it / 4, ir = it % 4;
    device const float *y4[3] = {
        src_in0 + ix * 256 + 64 * iq + 8 * ir,
        src_in1 + ix * 256 + 64 * iq + 8 * ir,
        src_in2 + ix * 256 + 64 * iq + 8 * ir
    };
    device float *dst[3] = {dst_out0, dst_out1, dst_out2};
    device const block_q4_K *ax_gate[NR0], *ax_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset = (uint64_t)(first_row + row) * args.row_bytes;
        const bool active = first_row + row < (int)args.out_dim;
        ax_gate[row] = (device const block_q4_K *)(src_gate + (active ? offset : 0));
        ax_up[row] = (device const block_q4_K *)(src_up + (active ? offset : 0));
    }
    float sumf_gate[3][NR0] = {{0.f}}, sumf_up[3][NR0] = {{0.f}};
    for (int ib = ix; ib < nb; ib += 4) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (first_row + row >= (int)args.out_dim) break;
            device const uint16_t *sc_g = (device const uint16_t *)ax_gate[row][ib].scales + iq;
            device const uint16_t *q1_g = (device const uint16_t *)ax_gate[row][ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *q2_g = q1_g + 32;
            device const uint16_t *sc_u = (device const uint16_t *)ax_up[row][ib].scales + iq;
            device const uint16_t *q1_u = (device const uint16_t *)ax_up[row][ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *q2_u = q1_u + 32;
            uint16_t sc16g[4], sc16u[4], qg1[4], qg2[4], qu1[4], qu2[4];
            sc16g[0] = sc_g[0] & kmask1;
            sc16g[1] = sc_g[2] & kmask1;
            sc16g[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
            sc16g[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);
            sc16u[0] = sc_u[0] & kmask1;
            sc16u[1] = sc_u[2] & kmask1;
            sc16u[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
            sc16u[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);
            thread const uint8_t *sc8g = (thread const uint8_t *)sc16g;
            thread const uint8_t *sc8u = (thread const uint8_t *)sc16u;
            const half dg = ax_gate[row][ib].d, dmg = ax_gate[row][ib].dmin;
            const half du = ax_up[row][ib].d, dmu = ax_up[row][ib].dmin;
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                qg1[i] = q1_g[i]; qg2[i] = q2_g[i];
                qu1[i] = q1_u[i]; qu2[i] = q2_u[i];
            }
            for (short t = 0; t < 3; ++t) {
                float yl[16], yh[16]; float4 sumy = {0.f,0.f,0.f,0.f};
                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    yl[i] = y4[t][i]; sumy[0] += yl[i];
                    yl[i+8] = y4[t][i+32]; sumy[1] += yl[i+8];
                    yh[i] = y4[t][i+128]; sumy[2] += yh[i];
                    yh[i+8] = y4[t][i+160]; sumy[3] += yh[i+8];
                }
                sumf_gate[t][row]=myds_v41_gu_shared_q4_scalar_block(sumf_gate[t][row],yl,yh,sumy,qg1,qg2,sc8g,dg,dmg);
                sumf_up[t][row]=myds_v41_gu_shared_q4_scalar_block(sumf_up[t][row],yl,yh,sumy,qu1,qu2,sc8u,du,dmu);
            }
        }
        for (short t = 0; t < 3; ++t) y4[t] += 4 * 256;
    }
    for (short row = 0; row < NR0 && first_row + row < (int)args.out_dim; ++row) {
        for (short t = 0; t < 3; ++t) {
            float tot_g = simd_sum(sumf_gate[t][row]), tot_u = simd_sum(sumf_up[t][row]);
            if (tiisg == 0) {
                float g = dsv4_mv_bf16_store(tot_g), u = dsv4_mv_bf16_store(tot_u);
                if (args.clamp > 1.0e-6f) {
                    g = min(g, args.clamp); u = clamp(u, -args.clamp, args.clamp);
                }
                const float silu = g / (1.0f + exp(-g));
                const float glu_v = silu * u;
                dst[t][first_row + row] = dsv4_mv_bf16_store(glu_v);
            }
        }
    }
}

template<short NR0>
void kernel_myds_v41_shared_gate_up_swiglu_q8_shared3_impl(
    constant myds_v41_shared_swiglu_args &args,
    device const char *src_gate, device const char *src_up,
    device const float *src_in0, device const float *src_in1, device const float *src_in2,
    device float *dst_out0, device float *dst_out1, device float *dst_out2,
    threadgroup char *shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    constexpr short NSG = 4, NW = N_SIMDWIDTH, NQ = 8;
    const int nb = (int)args.in_dim / QK8_0, r0 = tgpig.x * NR0;
    device const block_q8_0 *ax_gate[NR0], *ax_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset = (uint64_t)(r0 + row) * args.row_bytes;
        const bool active = r0 + row < (int)args.out_dim;
        ax_gate[row] = (device const block_q8_0 *)(src_gate + (active ? offset : 0));
        ax_up[row] = (device const block_q8_0 *)(src_up + (active ? offset : 0));
    }
    float sumf_gate[3][NR0] = {{0.f}}, sumf_up[3][NR0] = {{0.f}};
    const short ix = tiisg / (NW / NQ), il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    device const float *yb[3] = {src_in0 + ib0 * QK8_0 + il * NQ,
                                src_in1 + ib0 * QK8_0 + il * NQ,
                                src_in2 + ib0 * QK8_0 + il * NQ};
    device float *dst[3] = {dst_out0, dst_out1, dst_out2};
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        float yl[3][NQ];
        for (short t = 0; t < 3; ++t)
            for (short i = 0; i < NQ; ++i) yl[t][i] = yb[t][i];
        for (short row = 0; row < NR0; ++row) {
            int8_t qg[NQ], qu[NQ];
            const half dg = ax_gate[row][ib].d, du = ax_up[row][ib].d;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                qg[i] = ax_gate[row][ib].qs[il * NQ + i];
                qu[i] = ax_up[row][ib].qs[il * NQ + i];
            }
            for (short t = 0; t < 3; ++t) {
                float sumq_g = 0.f, sumq_u = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq_g += qg[i] * yl[t][i];
                    sumq_u += qu[i] * yl[t][i];
                }
                sumf_gate[t][row] += sumq_g * dg;
                sumf_up[t][row] += sumq_u * du;
            }
        }
        for (short t = 0; t < 3; ++t) yb[t] += NSG * NQ * QK8_0;
    }
    threadgroup float *sg[3][NR0], *su[3][NR0];
    for (short t = 0; t < 3; ++t) for (short row = 0; row < NR0; ++row) {
        sg[t][row] = (threadgroup float *)shmem + NW * (t * 2 * NR0 + row);
        su[t][row] = (threadgroup float *)shmem + NW * (t * 2 * NR0 + NR0 + row);
        if (sgitg == 0) {sg[t][row][tiisg] = 0.f; su[t][row][tiisg] = 0.f;}
        sumf_gate[t][row] = simd_sum(sumf_gate[t][row]);
        sumf_up[t][row] = simd_sum(sumf_up[t][row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short t = 0; t < 3; ++t) for (short row = 0; row < NR0; ++row)
        if (tiisg == 0) {sg[t][row][sgitg] = sumf_gate[t][row]; su[t][row][sgitg] = sumf_up[t][row];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && r0 + row < (int)args.out_dim; ++row) {
        for (short t = 0; t < 3; ++t) {
            float tot_g = simd_sum(sg[t][row][tiisg]), tot_u = simd_sum(su[t][row][tiisg]);
            if (tiisg == 0 && sgitg == 0) {
                float g = dsv4_mv_bf16_store(tot_g), u = dsv4_mv_bf16_store(tot_u);
                if (args.clamp > 1.0e-6f) {g = min(g, args.clamp); u = clamp(u, -args.clamp, args.clamp);}
                const float silu = g / (1.0f + exp(-g));
                const float glu_v = silu * u;
                dst[t][r0 + row] = dsv4_mv_bf16_store(glu_v);
            }
        }
    }
}

#define MYDS_GU_SHARED3_Q8_KERNEL(NAME, NR0) \
kernel void NAME( \
    constant myds_v41_shared_swiglu_args &args [[buffer(0)]], \
    device const char *src_gate [[buffer(1)]], device const char *src_up [[buffer(2)]], \
    device const float *src_in0 [[buffer(3)]], device const float *src_in1 [[buffer(4)]], \
    device const float *src_in2 [[buffer(5)]], device float *dst_out0 [[buffer(6)]], \
    device float *dst_out1 [[buffer(7)]], device float *dst_out2 [[buffer(8)]], \
    threadgroup char *shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]], \
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    kernel_myds_v41_shared_gate_up_swiglu_q8_shared3_impl<NR0>( \
        args, src_gate, src_up, src_in0, src_in1, src_in2, dst_out0, dst_out1, dst_out2, \
        shmem, tgpig, tiisg, sgitg); \
}
MYDS_GU_SHARED3_Q8_KERNEL(kernel_myds_v41_shared_gate_up_swiglu_q8_shared3_r2, 2)
MYDS_GU_SHARED3_Q8_KERNEL(kernel_myds_v41_shared_gate_up_swiglu_q8_shared3_r4, 4)
#undef MYDS_GU_SHARED3_Q8_KERNEL

/* ============================================================================
 * DSpark 专属融合路由内核：128 专家、Top-3 单 Threadgroup 全在位双调排序与归一化
 * 消除 CPU-GPU 往返同步与命令缓冲区打断（Zero-Stall GPU Pipelining）
 * ============================================================================ */
kernel void kernel_myds_dspark_route_fused_128(
    constant myds_v41_route_args &args [[buffer(0)]],
    device const float4 *logits [[buffer(1)]],
    device const float4 *bias [[buffer(2)]],
    device int *selected [[buffer(3)]],
    device float *weights [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint t = tgid;
    if (t >= args.rows) return;

    threadgroup int *sh_indices = (threadgroup int *)shmem;
    threadgroup float *sh_scores = (threadgroup float *)(sh_indices + 128);
    threadgroup float *sh_probs = (threadgroup float *)(sh_scores + 128);
    threadgroup float *scratch = (threadgroup float *)(sh_probs + 128);

    // 1. 阶段一：计算 softplus -> sqrt -> + bias
    // 128 个专家对应 32 个 float4。由 tid < 32 的线程向量化读取与计算
    if (tid < 32u) {
        const uint i = t * 32u + tid;
        const float4 x = logits[i];
        const float4 b = (args.op != 0) ? bias[tid] : (float4)0.0f;
        const float4 ex = exp(x);
        const float4 em = min(ex, (float4)0.03125f);
        const float4 poly = em * (1.0f - em * (0.5f - em * (1.0f / 3.0f - 0.25f * em)));
        const float4 sp = select(select(log(1.0f + ex), poly, ex < (float4)0.03125f), x, x > 20.0f);
        const float4 pr = sqrt(sp);
        const float4 sc = pr + b;
        ((threadgroup float4 *)sh_probs)[tid] = pr;
        ((threadgroup float4 *)sh_scores)[tid] = sc;
    }
    sh_indices[tid] = (int)tid;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. 阶段二：双调排序（Bitonic Sort Descending，降序，精确 128 元素）
    for (int k = 2; k <= 128; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = (int)tid ^ j;
            if (ixj > (int)tid) {
                if (((int)tid & k) == 0) {
                    if (sh_scores[sh_indices[tid]] < sh_scores[sh_indices[ixj]] ||
                       (sh_scores[sh_indices[tid]] == sh_scores[sh_indices[ixj]] && sh_indices[tid] > sh_indices[ixj])) {
                        int tmp = sh_indices[tid];
                        sh_indices[tid] = sh_indices[ixj];
                        sh_indices[ixj] = tmp;
                    }
                } else {
                    if (sh_scores[sh_indices[tid]] > sh_scores[sh_indices[ixj]] ||
                       (sh_scores[sh_indices[tid]] == sh_scores[sh_indices[ixj]] && sh_indices[tid] < sh_indices[ixj])) {
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
        weights[t * top_k + tid] = (p / denom) * args.scale;
    }
}
