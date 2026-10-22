// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};


static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

/* DSpark checkpoints use F32 transformer activations. Keep these primitives
 * separate from the target graph's BF16-rounded RoPE and 32-wide FP8 QAT. */
kernel void kernel_myds_dspark_rope_f32(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = (ulong)group.x * args.width + args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = re * c - im * s;
    x[i + 1u] = re * s + im * c;
}

kernel void kernel_myds_dspark_kv_fp8_f32(
        constant uint &rows,
        device float *x,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if(group.x>=7u||group.y>=rows)return;
    const ulong i=(ulong)group.y*512u+group.x*64u+tid;
    const float value=x[i];
    scratch[tid]=abs(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=32u;stride>0u;stride>>=1u) {
        if(tid<stride)scratch[tid]=max(scratch[tid],scratch[tid+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale=dsv4_pow2_ceil_scale(max(scratch[0],1.0e-4f),448.0f);
    x[i]=dsv4_e4m3fn_dequant(clamp(value/scale,-448.0f,448.0f))*scale;
}

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_rope_round(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const ulong head_base = ((ulong)group.y * (ulong)args.heads + (ulong)group.x) * (ulong)args.width;
    const uint unrotated = args.width >= 64u ? args.width - 64u : 0u;
    for (uint k = 0; k < unrotated; k += 32u) {
        if (k + lane < unrotated) {
            const ulong idx = head_base + (ulong)(k + lane);
            x[idx] = dsv41_bf16(x[idx]);
        }
    }
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = head_base + (ulong)unrotated + 2u * (ulong)lane;
    const float re = dsv41_bf16(x[i]), im = dsv41_bf16(x[i + 1u]);
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

/* Decode-only fused finish. Each Q head retains the original 32-pair RoPE
 * mapping. The other 16 groups process one 32-value KV quantization block.
 * KV tail pairs are read BEFORE any writes; paired elements always share a
 * SIMD group, so no cross-group read/write dependency is introduced. */
struct dsv41_qkv_finish_args { uint pos, pad; float frequencies[32]; };
kernel void kernel_dsv41_qkv_finish(
        constant dsv41_qkv_finish_args &args,
        device float *q, device float *kv, device float *window,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if(group<64u) {
        const float theta=float(args.pos)*args.frequencies[lane];
        const float c=precise::cos(theta),s=precise::sin(theta);
        const uint i=group*512u+448u+2u*lane;
        const float re=q[i],im=q[i+1u];
        q[i]=dsv41_bf16(re*c-im*s);
        q[i+1u]=dsv41_bf16(re*s+im*c);
        return;
    }
    const uint column=(group-64u)*32u+lane;
    float value=kv[column];
    if(column>=448u) {
        const uint pair=(column-448u)/2u;
        const uint i=448u+pair*2u;
        const float theta=float(args.pos)*args.frequencies[pair];
        const float c=precise::cos(theta),s=precise::sin(theta);
        const float re=kv[i],im=kv[i+1u];
        value=dsv41_bf16((column&1u)?re*s+im*c:re*c-im*s);
    }
    /* Match rope's BF16 store, quant's BF16 input, E4M3 with power-of-two
     * scale, then quant's final BF16 store. Non-tail values only enter quant. */
    value=dsv41_bf16(value);
    const float amax=simd_max(abs(value));
    const float scale=dsv41_pow2_ceil(max(amax,1.0e-4f)*(1.0f/448.0f));
    const float result=dsv41_bf16(copysign(dsv4_e4m3fn_dequant(abs(value)/scale),value)*scale);
    kv[column]=result;
    window[(args.pos%128u)*512u+column]=result;
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct dsv41_pack_bf16_args {
    ulong dst_offset, src_offset;
    uint count, pad;
};
kernel void kernel_dsv41_pack_bf16(
        constant dsv41_pack_bf16_args &args,
        device const float *src, device ushort *dst,
        uint gid [[thread_position_in_grid]]) {
    if(gid<args.count)
        dst[args.dst_offset+gid]=ushort(as_type<uint>(src[args.src_offset+gid])>>16u);
}
kernel void kernel_dsv41_stage_bf16_f16(
        constant uint &count,
        device const ushort *src, device half *dst,
        uint gid [[thread_position_in_grid]]) {
    if(gid<count)dst[gid]=half(as_type<float>(uint(src[gid])<<16u));
}
kernel void kernel_dsv41_get_rows_bf16(
        constant uint2 &args,
        device const ushort *src, device const int *ids, device float *dst,
        uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=512u||gid.y>=args.y)return;
    const int row=ids[gid.y];
    const float value=row>=0&&uint(row)<args.x?
        as_type<float>(uint(src[(ulong)row*512u+gid.x])<<16u):0.0f;
    dst[(ulong)gid.y*512u+gid.x]=value;
}

kernel void kernel_dsv41_copy_window_f32_to_f16(
        constant uint2 &args,
        device const float *window,
        device half *dst,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 128u || gid.y >= args.x) return;
    const uint src_row = (args.y + gid.y) % 128u;
    device const float4 *w4 = (device const float4 *)(window + (ulong)src_row * 512u);
    device half4 *dst4 = (device half4 *)(dst + (ulong)gid.y * 512u);
    dst4[gid.x] = half4(w4[gid.x]);
}

struct dsv41_get_rows_bf16_f16_args {
    uint source_rows;
    uint selected_rows;
    uint dst_row_offset;
    uint pad;
};

kernel void kernel_dsv41_get_rows_bf16_to_f16(
        constant dsv41_get_rows_bf16_f16_args &args,
        device const ushort *src,
        device const int *ids,
        device half *dst,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 128u || gid.y >= args.selected_rows) return;
    const int row = ids[gid.y];
    const ulong dst_row = (ulong)(args.dst_row_offset + gid.y);
    device half4 *dst4 = (device half4 *)(dst + dst_row * 512u);
    if (row >= 0 && uint(row) < args.source_rows) {
        device const ushort4 *src4 = (device const ushort4 *)(src + (ulong)row * 512u);
        const ushort4 u = src4[gid.x];
        const float4 f = float4(
            as_type<float>(uint(u.x) << 16u),
            as_type<float>(uint(u.y) << 16u),
            as_type<float>(uint(u.z) << 16u),
            as_type<float>(uint(u.w) << 16u)
        );
        dst4[gid.x] = half4(f);
    } else {
        dst4[gid.x] = half4(0.0h);
    }
}

// Direct single-query packing only. Each TG owns one complete 512-value row;
// the two uniform branches reproduce the original half4 conversions verbatim.
// No attention/RoPE/reduction arithmetic or padding layout is changed here.
struct myds_dsv41_direct_pack_args {
    uint n_raw, raw_start, source_rows, selected_rows;
};
static_assert(sizeof(myds_dsv41_direct_pack_args)==16,"direct pack host ABI");
kernel void kernel_myds_dsv41_direct_pack_fuse(
        constant myds_dsv41_direct_pack_args &args,
        device const float *window,
        device const ushort *src,
        device const int *ids,
        device half *dst,
        uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=128u||gid.y>=args.n_raw+args.selected_rows)return;
    if(gid.y<args.n_raw) {
        const uint src_row=(args.raw_start+gid.y)%128u;
        device const float4 *w4=(device const float4 *)(window+(ulong)src_row*512u);
        device half4 *dst4=(device half4 *)(dst+(ulong)gid.y*512u);
        dst4[gid.x]=half4(w4[gid.x]);
    } else {
        const uint selected=gid.y-args.n_raw;
        const int row=ids[selected];
        device half4 *dst4=(device half4 *)(dst+(ulong)gid.y*512u);
        if(row>=0&&uint(row)<args.source_rows) {
            device const ushort4 *src4=(device const ushort4 *)(src+(ulong)row*512u);
            const ushort4 u=src4[gid.x];
            const float4 f=float4(
                as_type<float>(uint(u.x)<<16u),
                as_type<float>(uint(u.y)<<16u),
                as_type<float>(uint(u.z)<<16u),
                as_type<float>(uint(u.w)<<16u));
            dst4[gid.x]=half4(f);
        } else dst4[gid.x]=half4(0.0h);
    }
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
    /* block_mask 的【行步长】（float 为单位）。逐 token 时每行的可见块数就是
     * (width+7)/8，可以当下标算；批量后每行的可见块数不同，而 prompt_block_mask
     * 的行步长是全局的，所以必须显式传。逐 token 调用传 (width+7)/8 ⇒ 行为不变。 */
    uint mask_stride;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * args.mask_stride + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

/* ── 位打包的 prompt 块掩码（2026-10-22）───────────────────────────
 * 上面 kernel_dsv41_candidate_filter 用 float 存每个「8 块组」的可见性，
 * 而它只判断 == 0.0f。prompt_block_mask 的尺寸是 cap x (blocks) 个 float，
 * 于是 500K 上下文要吃 25 GB —— 只为存一堆布尔。
 *
 * 这里改成 1 bit / 块，32 倍压缩。语义完全等价：位为 1 ⇔ float 为 0。
 * 原来的 float 版本保留（decode 侧的 dense mask 仍在用），
 * 两条路径由 host 侧的 MYDS_PROMPT_MASK_BITS 选择，便于 0-ULP 对照。
 *
 * 写入用 atomic_fetch_or：同一 word 的 32 个位可能被不同线程并发置位。 */
struct ds4_metal_args_dsv41_maskbits {
    uint width;        /* 掩码条目数 = 块数 (n_comp+7)/8 */
    uint rows;         /* token 行数 */
    uint stride;       /* 每行 word 数 */
    uint start, ratio;
    ulong nb_top0, nb_top1;   /* topk 的 行/列 字节步长 */
    uint top;                 /* 每行 top_k 个数 */
};

/* 清零：整块 words。grid 覆盖 rows*stride。 */
kernel void kernel_dsv41_maskbits_clear(
        constant ds4_metal_args_dsv41_maskbits &args,
        device uint *words,
        uint gid [[thread_position_in_grid]]) {
    const ulong n = (ulong)args.rows * args.stride;
    if ((ulong)gid >= n) return;
    words[gid] = 0u;
}

/* 置位：把 topk 选中的组打开。grid 覆盖 rows*top。 */
kernel void kernel_dsv41_maskbits_scatter(
        constant ds4_metal_args_dsv41_maskbits &args,
        device const char *topk,
        device uint *words,
        uint gid [[thread_position_in_grid]]) {
    const ulong n = (ulong)args.rows * args.top;
    if ((ulong)gid >= n) return;
    const uint ik = (uint)((ulong)gid % args.top);
    const uint it = (uint)((ulong)gid / args.top);
    const int32_t idx = *((device const int32_t *)(topk + (ulong)ik * args.nb_top0 + (ulong)it * args.nb_top1));
    /* idx 就是【块号】：float 版把每个块的可见性存成一个 float，
     * 读取侧用 x/8（x = 压缩键号）取块号。这里一个块一个 bit。 */
    if (idx < 0 || (uint)idx >= args.width) return;
    const uint blk = (uint)idx;
    atomic_fetch_or_explicit(&((device atomic_uint *)words)[(ulong)it * args.stride + (blk >> 5)],
                             1u << (blk & 31u), memory_order_relaxed);
}

/* 读取：位为 1 才保留分数，否则 -INFINITY。与 float 版逐位等价。 */
kernel void kernel_dsv41_candidate_filter_bits(
        constant ds4_metal_args_dsv41_maskbits &args,
        device const float *scores,
        device float *out,
        device const uint *words,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    const uint blk = index.x >> 3;                 /* 8 个压缩键共用 1 个块位 */
    const bool allowed = blk < args.width &&
        (words[(ulong)index.y * args.stride + (blk >> 5)] & (1u << (blk & 31u))) != 0u;
    out[offset] = (index.x < visible && allowed) ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

/* Independent indexer candidate, one SIMD32 per 128-value head. Keep the
 * reference RoPE pair->lane mapping/trig and both BF16 outputs in registers.
 * All tail inputs are read before any stores. Each subsequent quant block
 * has exactly the reference quantizer's column->lane mapping and simd_max.
 * The two tail blocks shuffle within this same complete SIMD32 only. */
struct dsv41_indexer_rope_quant_args {
    uint heads, pos;
    float frequencies[32];
};
kernel void kernel_dsv41_indexer_rope_quant(
        constant dsv41_indexer_rope_quant_args &args,
        device float *x,
        uint head [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const ulong base = (ulong)head * 128u;
    const float theta = float(args.pos) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = precise::sin(theta);
    const ulong i = base + 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    const float rotated_re = dsv41_bf16(re * c - im * s);
    const float rotated_im = dsv41_bf16(re * s + im * c);
    for (uint block = 0; block < 4u; ++block) {
        float source;
        if (block < 2u) {
            source = x[base + block * 32u + lane];
        } else {
            const uint pair = (block - 2u) * 16u + lane / 2u;
            const float r_re = simd_shuffle(rotated_re, pair);
            const float r_im = simd_shuffle(rotated_im, pair);
            source = (lane & 1u) ? r_im : r_re;
        }
        /* Preserve quant's BF16 input even for already-rounded RoPE values. */
        const float value = dsv41_bf16(source);
        const float amax = simd_max(abs(value));
        const float scale = dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        const float result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
        x[base + block * 32u + lane] = dsv41_bf16(result);
    }
}
