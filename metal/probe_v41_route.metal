// Independent production-topology route probes. Not in SHADER_FILES.
// Compile standalone with -O3 -ffast-math, then compare against the actual
// production metallib PSO before treating any candidate as equivalent.
// Frozen route kernel from metal/myds_v41_moe.metal:125-253 at HEAD04dee3f.
// Original kernel text SHA256: 2e0818694abc7dea45ea7b3aee0869fb69daea9e0bd89b4ae1fb7823413a53c0
#include <metal_stdlib>
using namespace metal;

struct myds_v41_route_args { uint count; uint top_k; uint op; float scale; uint rows; uint row; uint row4; };

kernel void probe_v41_route_ref(
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

kernel void probe_v41_route_shuffle(
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

    // Same 512-element bitonic network and same strict comparisons. Only
    // exchanges with j<32 move through SIMD registers. We intentionally retain
    // the original padding behavior, including swaps of two invalid indices.
    int current_id = sh_indices[tid];
    float current_score = sh_scores[current_id];
    for (int k = 2; k <= 512; k *= 2) {
        if (k >= 64) {
            // Publish the preceding register stages once per cross-SIMD run.
            sh_indices[tid] = current_id;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int j = k/2; j >= 32; j /= 2) {
                const int ixj = (int)tid ^ j;
                if (ixj > (int)tid) {
                    if (((int)tid & k) == 0) {
                        if (sh_indices[tid] >= 384 ||
                           (sh_indices[ixj] < 384 && sh_scores[sh_indices[tid]] < sh_scores[sh_indices[ixj]])) {
                            const int tmp = sh_indices[tid];
                            sh_indices[tid] = sh_indices[ixj];
                            sh_indices[ixj] = tmp;
                        }
                    } else {
                        if (sh_indices[ixj] >= 384 ||
                           (sh_indices[tid] < 384 && sh_scores[sh_indices[tid]] > sh_scores[sh_indices[ixj]])) {
                            const int tmp = sh_indices[tid];
                            sh_indices[tid] = sh_indices[ixj];
                            sh_indices[ixj] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            current_id = sh_indices[tid];
            current_score = sh_scores[current_id];
        }
        for (int j = min(k/2,16); j > 0; j /= 2) {
            const int peer_id = simd_shuffle_xor(current_id, (ushort)j);
            const float peer_score = simd_shuffle_xor(current_score, (ushort)j);
            const bool is_low = (((int)tid & j) == 0);
            const int low_id = is_low ? current_id : peer_id;
            const int high_id = is_low ? peer_id : current_id;
            const float low_score = is_low ? current_score : peer_score;
            const float high_score = is_low ? peer_score : current_score;
            const bool descending = (((int)tid & k) == 0);
            const bool swap = descending ?
                (low_id >= 384 || (high_id < 384 && low_score < high_score)) :
                (high_id >= 384 || (low_id < 384 && low_score > high_score));
            if (swap) { current_id = peer_id; current_score = peer_score; }
        }
    }
    // The existing stage-three barrier below publishes these final indices.
    sh_indices[tid] = current_id;

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
