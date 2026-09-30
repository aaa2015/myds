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
