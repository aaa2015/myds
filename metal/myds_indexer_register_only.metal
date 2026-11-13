// Independent BF16 prefill scorer with 4 KB cooperative Q threadgroup reuse.
// Same DSV4 72-byte ABI, block/head arithmetic order and score layout as the
// native tiled scorer. Resident K fragments stay in registers (zero K/dot smem),
// while 4 simdgroups cooperatively stage one head of Q (8x128 floats = 4 KB)
// with XOR bank-conflict-free swizzling to cut device Q traffic by 75%.
static inline short2 myds_indexer_register_only_coord(ushort lane) {
    const short qid = lane / 4;
    return short2((qid & 2) * 2 + (lane % 2) * 2,
                  (qid & 4) + ((lane / 2) % 4));
}

kernel void kernel_myds_indexer_scores_register_only_bf16(
        constant ds4_metal_args_dsv4_indexer_scores_fused &args [[buffer(0)]],
        device const char *q                                  [[buffer(1)]],
        device const char *weights                            [[buffer(2)]],
        device const char *index_comp                         [[buffer(3)]],
        device char *scores                                   [[buffer(4)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    // Uniform rejection before any write. Buffer spans/alignment/no-alias,
    // BF16 format, key-window=0 and 128-thread geometry are host preconditions.
    if (args.n_head != 32u || args.head_dim != 128u ||
        (args.ratio != 1u && args.ratio != 2u) ||
        args.n_comp <= 512u || args.n_comp > 1048576u ||
        args.n_tokens < 8u || args.n_tokens > 2048u ||
        args.pos0 > 1048576u - args.n_tokens || args.scale != 1.0f / 64.0f ||
        sg >= 4u) return;

    const uint token_base = tgpig.y * 8u;
    const uint tg_comp_base = tgpig.x * 32u;
    if (token_base >= args.n_tokens || tg_comp_base >= args.n_comp) return;
    const uint comp_base = tg_comp_base + uint(sg) * 8u;
    const short2 xy = myds_indexer_register_only_coord(lane);
    const uint token = token_base + uint(xy.y);
    const uint comp0 = comp_base + uint(xy.x);
    const uint comp1 = comp0 + 1u;
    const bool live = token < args.n_tokens;
    const uint last_token = min(token_base + 8u, args.n_tokens);
    const uint max_visible = min((args.pos0 + last_token) / args.ratio, args.n_comp);

    // TG-uniform causal invisibility check: zero barriers for invisible tiles.
    if (tg_comp_base >= max_visible) {
        if (live) {
            device float *dst = (device float *)(scores + (ulong)token * args.score_token_stride);
            if (comp0 < args.n_comp) dst[comp0] = -INFINITY;
            if (comp1 < args.n_comp) dst[comp1] = -INFINITY;
        }
        return;
    }

    // 4 KB static threadgroup buffer (8 tokens x 64 float2 = 4096 B).
    // XOR swizzle c2 ^ (t * 4u) eliminates 8-way bank conflicts across xy.y.
    threadgroup float2 smem_q2[512];
    const uint tid = uint(sg) * 32u + uint(lane);
    const uint lt = tid >> 6u;
    const uint lc2 = tid & 63u;
    const uint rd_t = uint(xy.y);
    const uint rd_j = uint(xy.x) >> 1u;
    const bool sg_active = comp_base < max_visible;

    simdgroup_float8x8 resident_k[16];
    if (sg_active) {
        // Fragment element (dim,key) = K[comp_base+key][db*8+dim].
        // The two elements per lane are adjacent columns in one matrix row.
        #pragma clang loop unroll(full)
        for (uint db = 0; db < 16u; db++) {
            const uint d = db * 8u + uint(xy.y);
            float k0 = 0.0f, k1 = 0.0f;
            if (comp0 < args.n_comp) {
                device const ushort *row = (device const ushort *)(index_comp +
                    (ulong)comp0 * args.index_row_stride);
                k0 = as_type<float>(uint(row[d]) << 16u);
            }
            if (comp1 < args.n_comp) {
                device const ushort *row = (device const ushort *)(index_comp +
                    (ulong)comp1 * args.index_row_stride);
                k1 = as_type<float>(uint(row[d]) << 16u);
            }
            resident_k[db].thread_elements()[0] = k0;
            resident_k[db].thread_elements()[1] = k1;
        }
    }

    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint head = 0; head < 32u; head++) {
        #pragma clang loop unroll(full)
        for (uint u = 0; u < 4u; u++) {
            const uint t = u * 2u + lt;
            const uint tok = token_base + t;
            float2 v = float2(0.0f);
            if (tok < args.n_tokens) {
                device const packed_float2 *qh2 = (device const packed_float2 *)(q +
                    (ulong)tok * args.q_token_stride + (ulong)head * args.q_head_stride);
                v = float2(qh2[lc2]);
            }
            smem_q2[t * 64u + (lc2 ^ (t * 4u))] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint db = 0; db < 16u; db++) {
                simdgroup_float8x8 mq;
                const float2 qp = smem_q2[rd_t * 64u + ((db * 4u + rd_j) ^ (rd_t * 4u))];
                mq.thread_elements()[0] = qp.x;
                mq.thread_elements()[1] = qp.y;
                simdgroup_multiply_accumulate(mdot, mq, resident_k[db], mdot);
            }
            // Preserve the LOCAL DSV4 expression; upstream GLM places scale
            // before ReLU instead. Do not reassociate or replace with an fma.
            if (live && comp0 < args.n_comp) {
                #pragma clang fp contract(off) reassociate(off)
                device const float *w = (device const float *)(weights +
                    (ulong)token * args.weights_token_stride);
                const float s = mdot.thread_elements()[0];
                const float weighted = max(s, 0.0f) * w[head];
                const float scaled = weighted * args.scale;
                acc0 = scaled + acc0;
            }
            if (live && comp1 < args.n_comp) {
                #pragma clang fp contract(off) reassociate(off)
                device const float *w = (device const float *)(weights +
                    (ulong)token * args.weights_token_stride);
                const float s = mdot.thread_elements()[1];
                const float weighted = max(s, 0.0f) * w[head];
                const float scaled = weighted * args.scale;
                acc1 = scaled + acc1;
            }
        }
        if (head + 1u < 32u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (sg_active) {
        if (live) {
            const uint visible = min((args.pos0 + token + 1u) / args.ratio, args.n_comp);
            device float *dst = (device float *)(scores + (ulong)token * args.score_token_stride);
            if (comp0 < args.n_comp) dst[comp0] = comp0 < visible ? acc0 : -INFINITY;
            if (comp1 < args.n_comp) dst[comp1] = comp1 < visible ? acc1 : -INFINITY;
        }
    } else if (live) {
        // Whole invisible SG tile is still part of the dense score contract.
        device float *dst = (device float *)(scores + (ulong)token * args.score_token_stride);
        if (comp0 < args.n_comp) dst[comp0] = -INFINITY;
        if (comp1 < args.n_comp) dst[comp1] = -INFINITY;
    }
}
