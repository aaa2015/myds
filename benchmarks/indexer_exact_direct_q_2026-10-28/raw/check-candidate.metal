kernel void probe_indexer_exact_direct_q(
        constant ds4_metal_args_glm_indexer_score_chunk & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        uint2 tgpid [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row_base = tgpid.x * 16u;
    const uint tok_base = tgpid.y * 8u;
    if (row_base >= args.n_rows || tok_base >= args.n_tokens ||
        args.n_head != 32u || args.head_dim != 128u) {
        return;
    }

    const uint last_tok = min(tok_base + 8u, args.n_tokens);
    const uint tile_tokens = last_tok - tok_base;
    const uint max_vis = (args.causal_start + (last_tok - 1u) + 1u) / args.causal_ratio;

    /* 2026-10-25：距离窗口下界。
     * ★ 下界必须按【块内最早的那个 token】算，不能用 max_vis（块内最晚 token）。
     *   下界随可见上限单调递增，所以最早的 token 给出块内最小下界：只有
     *   row < tile_min_vis 才能保证对块内每个 token 都越界。用 max_vis 算会
     *   把 token 0..6 仍需要的行整块填成 -INF（每块最多误丢 max_vis - vis_0 行，
     *   ratio=1 时是 7 行）。
     * window=0 ⇒ tile_min_vis=0 ⇒ 下界条件永不成立 ⇒ 与改动前逐位等价。 */
    const uint vis_0 = (args.causal_start + tok_base + 1u) / args.causal_ratio;
    const uint tile_min_vis = (args.window && vis_0 > args.window)
                              ? (vis_0 - args.window) : 0u;

    /* 整块早退：超出因果上限（未来）或整块低于距离窗口（太早）都填 -INF 返回 */
    if (row_base >= max_vis || (tile_min_vis && row_base + 16u <= tile_min_vis)) {
        const uint cell_t = (uint)tid >> 4;
        const uint cell_c = (uint)tid & 15u;
        const uint cell_tok = tok_base + cell_t;
        const uint cell_row = row_base + cell_c;
        if (cell_tok < args.n_tokens && cell_row < args.n_rows) {
            scores[(uint64_t)cell_tok * args.n_rows + cell_row] = -INFINITY;
        }
        return;
    }

    float4 k_reg[4];
    const uint sg_row0 = row_base + (uint)sg * 4u;
    FOR_UNROLL (uint r = 0; r < 4u; r++) {
        const uint row = sg_row0 + r;
        if (row < args.n_rows && row < max_vis) {
            const uint64_t base_idx = (uint64_t)row * 128u + (uint)lane * 4u;
            k_reg[r] = float4(
                glm_cache_load_f32_or_f16(indexer_key_cache, base_idx + 0u, args.cache_f16),
                glm_cache_load_f32_or_f16(indexer_key_cache, base_idx + 1u, args.cache_f16),
                glm_cache_load_f32_or_f16(indexer_key_cache, base_idx + 2u, args.cache_f16),
                glm_cache_load_f32_or_f16(indexer_key_cache, base_idx + 3u, args.cache_f16));
        } else {
            k_reg[r] = float4(0.0f);
        }
    }

    float acc = 0.0f;
    for (uint head = 0; head < 32u; head++) {

        for (uint t = 0; t < tile_tokens; t++) {
            const uint vis_t = (args.causal_start + tok_base + t + 1u) / args.causal_ratio;
            if (sg_row0 >= vis_t) continue;
            /* 逐 token 窗口下界：整块早退只挡住「对全块都越界」的行，边界那一块
             * 里的旧行仍要靠这里逐行剔掉，否则窗口语义只到 16 行 tile 粒度。 */
            const uint lo_t = (args.window && vis_t > args.window)
                              ? (vis_t - args.window) : 0u;
            const uint tok = tok_base + t;
            device const float4 *q4_ptr = (device const float4 *)(q +
                ((uint64_t)tok * 32u + head) * 128u * sizeof(float));
            const float4 q4 = q4_ptr[(uint)lane];
            const float w = weights[(uint64_t)tok * 32u + head];
            FOR_UNROLL (uint r = 0; r < 4u; r++) {
                if (sg_row0 + r < lo_t) continue;
                float s = dot(q4, k_reg[r]);
                s = simd_sum(s);
                const float contrib = max(s * args.scale, 0.0f) * w;
                if ((uint)lane == (t * 4u + r)) {
                    acc += contrib;
                }
            }
        }
    }

    const uint out_t = (uint)lane >> 2;
    const uint out_r = (uint)lane & 3u;
    const uint out_tok = tok_base + out_t;
    const uint out_row = sg_row0 + out_r;
    if (out_tok < args.n_tokens && out_row < args.n_rows) {
        const uint vis = (args.causal_start + out_tok + 1u) / args.causal_ratio;
        const uint lo = (args.window && vis > args.window) ? (vis - args.window) : 0u;
        scores[(uint64_t)out_tok * args.n_rows + out_row] =
            (out_row < vis && out_row >= lo) ? acc : -INFINITY;
    }
}