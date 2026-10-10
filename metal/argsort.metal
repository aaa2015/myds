struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    uint32_t causal_start;
    uint32_t causal_ratio;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
    uint32_t causal_start;
    uint32_t causal_ratio;
    uint32_t block_width;
    uint32_t block_top_k;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order, bool causal = false, bool shuffle = false>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(args.ne00,
        int((args.causal_start + uint(i01) + 1u) / args.causal_ratio)) : args.ne00;
    if (i00 >= width) return;
    const int work_width = causal ?
        ((width - 1) / ntg.x) * args.top_k + min((width - 1) % ntg.x + 1, args.top_k) : args.ne0;

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < width) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_f32_i32_desc_causal")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_f32_i32_desc_causal_shuffle")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order, bool causal>
static inline void ds4_argsort_merge_body(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        bool prefix, int prefix_top,
        uint3 tgpig, ushort3 tpitg, ushort3 ntg) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);
    const uint width = causal ? min(uint(args.ne00),
        (args.causal_start + uint(i01) + 1u) / args.causal_ratio) : 0u;
    // Empty causal rows leave ids untouched, like the initial sort. In
    // particular, do not underflow width-1 into an out-of-bounds run length.
    if (causal && width == 0u) return;
    const int work_width = causal ? int(((width - 1u) / args.block_width) * args.block_top_k +
        min((width - 1u) % args.block_width + 1u, args.block_top_k)) : args.ne0;

    // A sorted run contributes at most final top entries. Keep the original
    // sparse run offsets/row stride; holes after a prefix are never read by
    // later prefix stages. NaN rows use exactly the dense lengths and chunks.
    const int read_limit = prefix ? min(args.len, prefix_top) : args.len;
    const int len0 = MIN(read_limit, MAX(0, work_width - start));
    const int len1 = MIN(read_limit, MAX(0, work_width - (start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int limit = prefix ? min(args.top_k, prefix_top) : args.top_k;
    const int k1 = MIN(MIN(k0 + chunk, total), limit);

    if (k0 >= limit) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

template<ds4_sort_order order, bool causal = false, bool prefix = false>
kernel void kernel_argsort_merge_f32_i32(
        constant ds4_metal_args_argsort_merge & args,
        device const char * src0, device const int32_t * tmp, device int32_t * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    ds4_argsort_merge_body<order, causal>(args, src0, tmp, dst, prefix, 512,
                                         tgpig, tpitg, ntg);
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal_prefix")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

struct myds_argsort_nan_args { uint cols, rows, start, ratio; };
// One group per row. Test IEEE bits, not isnan(): Metal's fast-math mode must
// not erase the unordered-input guard. Future causal columns are not consumed
// by the sort/merge and therefore do not force a dense row.
kernel void kernel_myds_argsort_visible_nan_rows(
        constant myds_argsort_nan_args & args,
        device const uint * score_bits, device uint * nan_rows,
        uint row[[threadgroup_position_in_grid]],
        ushort tid[[thread_index_in_threadgroup]],
        ushort lane[[thread_index_in_simdgroup]],
        ushort simd[[simdgroup_index_in_threadgroup]]) {
    const uint width=args.ratio?min(args.cols,(args.start+row+1u)/args.ratio):args.cols;
    uint found=0u;
    for(uint col=tid;col<width;col+=256u)
        found|=(score_bits[(ulong)row*args.cols+col]&0x7fffffffu)>0x7f800000u;
    threadgroup uint group_found[8];
    const uint any=simd_sum(found);
    if(lane==0)group_found[simd]=any;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        uint flag=0u;
        for(uint i=0;i<8u;i++)flag|=group_found[i];
        nan_rows[row]=flag;
    }
}

struct myds_argsort_prefix_merge_args {
    ds4_metal_args_argsort_merge merge;
    uint keep;
};
template<bool causal>
kernel void kernel_myds_argsort_merge_prefix_guarded(
        constant myds_argsort_prefix_merge_args & args,
        device const char * src0, device const int32_t * tmp, device int32_t * dst,
        device const uint * nan_rows,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint row=tgpig.x%uint(args.merge.ne01);
    const bool prefix=nan_rows[row]==0u;
    ds4_argsort_merge_body<DS4_SORT_ORDER_DESC, causal>(args.merge,src0,tmp,dst,
        prefix,int(args.keep),tgpig,tpitg,ntg);
}
typedef void (myds_argsort_prefix_merge_t)(
        constant myds_argsort_prefix_merge_args & args,
        device const char * src0, device const int32_t * tmp, device int32_t * dst,
        device const uint * nan_rows,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]);
template [[host_name("kernel_myds_argsort_merge_f32_i32_desc_prefix_guarded")]]
kernel myds_argsort_prefix_merge_t kernel_myds_argsort_merge_prefix_guarded<false>;
template [[host_name("kernel_myds_argsort_merge_f32_i32_desc_causal_prefix_guarded")]]
kernel myds_argsort_prefix_merge_t kernel_myds_argsort_merge_prefix_guarded<true>;

// Independent opt-in clone: original sort exports/body above stay unchanged.
// Flag classification shares the existing score staging and its TG barrier.
template<ds4_sort_order order, bool causal = false, bool shuffle = false>
kernel void kernel_myds_argsort_partition_nan(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        device        uint * partition_nan,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(args.ne00,
        int((args.causal_start + uint(i01) + 1u) / args.causal_ratio)) : args.ne00;
    const uint parts=(uint(args.ne00)+uint(ntg.x)-1u)/uint(ntg.x);
    const ulong flag_slot=(ulong(i01)+ulong(args.ne01)*i02+
        ulong(args.ne01)*args.ne02*i03)*parts+uint(ib);
    // Every dispatched partition owns and overwrites one flag, including
    // partitions with no visible causal values. Never inherit a prior row.
    if (i00 >= width) {
        if (col == 0) partition_nan[flag_slot]=0u;
        return;
    }
    const int work_width = causal ?
        ((width - 1) / ntg.x) * args.top_k + min((width - 1) % ntg.x + 1, args.top_k) : args.ne0;

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    uint found=0u;
    if (i00 + col < width) {
        // Use the source IEEE bits: fast-math must not erase a NaN test or
        // quiet signaling NaNs before classification. One source load supplies
        // both the bit guard and float staging, without changing any IEEE bits
        // or the original comparison network.
        const uint bits=((device const uint *)src0_row)[i00 + col];
        shmem_f32[col] = as_type<float>(bits);
        found=(bits&0x7fffffffu)>0x7f800000u;
    }
    threadgroup uint group_found[32]; // host preserves nth and requires <=1024
    const uint any=simd_sum(found);
    if ((col&31) == 0) group_found[uint(col)/32u]=any;

    // The original staging barrier also publishes SIMD classification partials.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (col == 0) {
        uint flag=0u;
        for (uint i=0u;i<(uint(ntg.x)+31u)/32u;i++) flag|=group_found[i];
        partition_nan[flag_slot]=flag!=0u;
    }

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
}

typedef void (myds_argsort_partition_nan_t)(
        constant ds4_metal_args_argsort & args,
        device const char * src0, device int32_t * dst, device uint * partition_nan,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]);
template [[host_name("kernel_myds_argsort_f32_i32_desc_partition_nan")]]
kernel myds_argsort_partition_nan_t kernel_myds_argsort_partition_nan<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_myds_argsort_f32_i32_desc_causal_partition_nan")]]
kernel myds_argsort_partition_nan_t kernel_myds_argsort_partition_nan<DS4_SORT_ORDER_DESC, true>;

struct myds_argsort_partition_prefix_merge_args {
    ds4_metal_args_argsort_merge merge;
    uint keep, parts;
};
template<bool causal>
kernel void kernel_myds_argsort_merge_partition_prefix_guarded(
        constant myds_argsort_partition_prefix_merge_args & args,
        device const char * src0, device const int32_t * tmp, device int32_t * dst,
        device const uint * partition_nan,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]],
        ushort lane[[thread_index_in_simdgroup]]) {
    const uint row=tgpig.x%uint(args.merge.ne01);
    uint found=0u;
    // Each SIMD independently ORs the entire row. A NaN in any original run
    // forces dense merging from the FIRST merge, even in other merge groups.
    // Non-power-of-two top-k also produces groups such as 48/62/511 threads.
    // Each SIMD, including a partial LAST SIMD, needs its own actual width as
    // the stride so every participating thread receives the same row decision.
    const uint simd_start=(uint(tpitg.x)/32u)*32u;
    const uint stride=min(uint(ntg.x)-simd_start,32u);
    for(uint part=uint(lane);part<args.parts;part+=stride)
        found|=partition_nan[(ulong)row*args.parts+part];
    const bool prefix=simd_sum(found)==0u;
    ds4_argsort_merge_body<DS4_SORT_ORDER_DESC, causal>(args.merge,src0,tmp,dst,
        prefix,int(args.keep),tgpig,tpitg,ntg);
}
typedef void (myds_argsort_partition_prefix_merge_t)(
        constant myds_argsort_partition_prefix_merge_args & args,
        device const char * src0, device const int32_t * tmp, device int32_t * dst,
        device const uint * partition_nan,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]],
        ushort lane[[thread_index_in_simdgroup]]);
template [[host_name("kernel_myds_argsort_merge_f32_i32_desc_partition_prefix_guarded")]]
kernel myds_argsort_partition_prefix_merge_t kernel_myds_argsort_merge_partition_prefix_guarded<false>;
template [[host_name("kernel_myds_argsort_merge_f32_i32_desc_causal_partition_prefix_guarded")]]
kernel myds_argsort_partition_prefix_merge_t kernel_myds_argsort_merge_partition_prefix_guarded<true>;

// Independent guarded reference. The original sort and merge bodies above stay intact.
template<ds4_sort_order order, bool causal = false, bool shuffle = false>
kernel void kernel_myds_argsort_threshold_guarded(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        device const uint * threshold_control,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (threshold_control[0] != 0u) return; // uniform GPU decision
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(args.ne00,
        int((args.causal_start + uint(i01) + 1u) / args.causal_ratio)) : args.ne00;
    if (i00 >= width) return;
    const int work_width = causal ?
        ((width - 1) / ntg.x) * args.top_k + min((width - 1) % ntg.x + 1, args.top_k) : args.ne0;

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < width) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
}


typedef void (myds_argsort_threshold_t)(constant ds4_metal_args_argsort & args,
    device const char *src0, device int32_t *dst, device const uint *control,
    threadgroup int32_t *shmem_i32, uint3 tgpig, ushort3 tpitg, ushort3 ntg);
template [[host_name("kernel_myds_argsort_f32_i32_desc_threshold_guarded")]]
kernel myds_argsort_threshold_t kernel_myds_argsort_threshold_guarded<DS4_SORT_ORDER_DESC>;
struct myds_argsort_threshold_merge_args { ds4_metal_args_argsort_merge merge; uint keep, prefix; };
kernel void kernel_myds_argsort_merge_f32_i32_desc_threshold_guarded(
    constant myds_argsort_threshold_merge_args &args [[buffer(0)]],
    device const char *src0 [[buffer(1)]],device const int32_t *tmp [[buffer(2)]],
    device int32_t *dst [[buffer(3)]],device const uint *control [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]]) {
    if(control[0]!=0u)return; // uniform, no scratch reads on adopted rows
    const bool prefix=args.prefix!=0u&&(control[5]&1u)==0u;
    ds4_argsort_merge_body<DS4_SORT_ORDER_DESC,false>(args.merge,src0,tmp,dst,
        prefix,int(args.keep),tgpig,tpitg,ntg);
}

// Independent opt-in symbols. The original sorter bodies/symbols remain intact;
// only the existing compile-time SIMD-local shuffle branch is instantiated.
template [[host_name("kernel_myds_argsort_f32_i32_desc_partition_nan_shuffle")]]
kernel myds_argsort_partition_nan_t kernel_myds_argsort_partition_nan<DS4_SORT_ORDER_DESC, false, true>;
template [[host_name("kernel_myds_argsort_f32_i32_desc_causal_partition_nan_shuffle")]]
kernel myds_argsort_partition_nan_t kernel_myds_argsort_partition_nan<DS4_SORT_ORDER_DESC, true, true>;
