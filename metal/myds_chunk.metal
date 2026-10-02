/* Chunk prefill attention helpers.
 *
 * The chunked prefill stages the whole chunk's KV, the carry of the previous 127
 * KV vectors and the compressed table prefix into one f16 slab, and lets each
 * query row mask what it may not see.  That is what collapses the per-token
 * dispatch storm into a handful of chunk-wide dispatches, and it is the shape
 * the shared flash attention kernel is already built for: one mask row per query
 * row, selected by threadgroup position. */

struct myds_mask_chunk_args {
    uint base;       /* first compressed column within a mask row */
    uint stride;     /* uint16 elements per mask row */
    uint n_comp;     /* compressed columns belonging to this chunk */
    uint top_k;      /* ids per query row */
    uint rows;       /* query rows, i.e. tokens in the chunk */
    uint c_start;    /* chunk start within the current batch */
    uint base_pos;   /* absolute position of the batch start */
    uint ratio;      /* compression ratio of this layer */
};

/* One thread per (row, compressed column): everything not selected is invisible. */
kernel void kernel_myds_mask_chunk_fill(
        constant myds_mask_chunk_args &args,
        device ushort *mask,
        uint2 gid [[thread_position_in_grid]]) {
    /* gid.x walks the whole compressed region, not just this chunk's live columns,
     * so the picks a previous chunk opened are cleared as well. */
    const uint width = args.stride - args.base;
    if (gid.x >= width || gid.y >= args.rows) return;
    mask[(ulong)gid.y * args.stride + args.base + gid.x] = 0xfc00u;
}

/* One thread per (row, selected slot): the selected compressed entries become visible.
 * A row's live count is (position + 1) / ratio, so the tail of the id row is ignored. */
kernel void kernel_myds_mask_chunk_scatter(
        constant myds_mask_chunk_args &args,
        device ushort *mask,
        device const int *ids,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.top_k || gid.y >= args.rows) return;
    const uint row = gid.y;
    const uint count = (args.base_pos + args.c_start + row + 1u) / args.ratio;
    if (gid.x >= count) return;
    const int idx = ids[(ulong)row * args.top_k + gid.x];
    if (idx < 0 || (uint)idx >= args.n_comp) return;
    mask[(ulong)row * args.stride + args.base + (uint)idx] = 0u;
}

/* ── B 方案：把「本 query 可见的键」gather 成紧凑段，让 flash 内核只算该算的 ──
 *
 * 原路径把整个压缩前缀 stage 进 slab，再让每个 query 用掩码挡掉不可见的键；
 * 因为跳块粒度是 32 键、而选中键散布，实测约有 4.7× 的算力花在「算了再掩掉」上
 * （见 docs/B_MAX_SPARSE_ATTENTION_SPEC.md）。
 *
 * 紧凑段布局：每个 query 一行 640 个键 = [0,128) 滑窗 + [128,640) 它的 top-512。
 * 滑窗无需真正收集 —— 在 staged slab 里它就是行 [q, q+127]，连续。 */

struct myds_attn_gather_args {
    uint rows;         /* query 行数（本 chunk 的 token 数） */
    uint top_k;        /* 每 query 的压缩 picks 数（512） */
    uint n_comp;       /* 压缩表行数 */
    uint valid_from;   /* slab 行号 >= 此值才是真实位置（= CARRY - carry） */
    uint mask_stride;  /* 掩码每行 u16 元素数（沿用 kv slab 的列宽） */
    uint c_start;      /* 本 chunk 在批内的起点 */
    uint base_pos;     /* 批的绝对起点 */
    uint ratio;        /* 本层压缩比 */
};

/* 每个线程组 = (query q, 滑窗第 d 行)；128 线程 × half4 正好覆盖一行的 512 个 half。
 * 注：不可见的行必须写 0（不能用未初始化内存）—— 掩码把它们置 -inf，
 * 但 0 * NaN 仍是 NaN，会把 softmax 搅坏。 */
kernel void kernel_myds_attn_gather_window(
        constant myds_attn_gather_args &args,
        device half         *dst,       /* rows * 640 * 512 halfs，紧凑段 */
        device ushort       *mask,      /* rows * mask_stride u16 */
        device const half   *slab,      /* staged f16 slab：行 = carry + rows 个 KV 向量 */
        ushort2 tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]]) {
    if (tgpig.y >= args.rows || tgpig.x > 127u) return;
    const uint q = tgpig.y, d = tgpig.x;
    const uint src = q + d;                       /* q<=rows-1, d<=127 ⇒ src < 127+rows */
    const bool vis = (src >= args.valid_from);
    device half4 *o = (device half4 *)(dst + ((ulong)q * 640u + d) * 512u);
    if (vis) {
        device const half4 *s = (device const half4 *)(slab + (ulong)src * 512u);
        o[tiitg] = s[tiitg];
    } else {
        o[tiitg] = half4(0.0h);
    }
    if (tiitg == 0u) mask[(ulong)q * args.mask_stride + d] = vis ? 0u : 0xfc00u;
}

/* 每个线程组 = (query q, 第 j 个 pick)，同样 128 线程 × half4。 */
kernel void kernel_myds_attn_gather_comp(
        constant myds_attn_gather_args &args,
        device half          *dst,
        device ushort        *mask,
        device const float   *comp,     /* n_comp * 512 f32 */
        device const int     *ids,      /* rows * top_k */
        ushort2 tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]]) {
    if (tgpig.y >= args.rows || tgpig.x >= args.top_k) return;
    const uint q = tgpig.y, j = tgpig.x;
    /* 因果：第 q 行只有前 count 个 pick 是活的（与 kernel_myds_mask_chunk_scatter 同式）。 */
    const uint count = (args.base_pos + args.c_start + q + 1u) / args.ratio;
    int idx = -1;
    if (j < count) idx = ids[(ulong)q * args.top_k + j];
    const bool vis = (idx >= 0) && ((uint)idx < args.n_comp);
    device half4 *o = (device half4 *)(dst + ((ulong)q * 640u + 128u + j) * 512u);
    if (vis) {
        device const float4 *s = (device const float4 *)(comp + (ulong)idx * 512u);
        const float4 x = s[tiitg];
        o[tiitg] = half4((half)x.x, (half)x.y, (half)x.z, (half)x.w);
    } else {
        o[tiitg] = half4(0.0h);
    }
    if (tiitg == 0u) mask[(ulong)q * args.mask_stride + 128u + j] = vis ? 0u : 0xfc00u;
}
