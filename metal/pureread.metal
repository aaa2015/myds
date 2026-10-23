/* 诊断：纯读带宽（不做任何解码）。每线程读 16 字节，累加到线程私有 sink。 */
#include <metal_stdlib>
using namespace metal;

struct pureread_args {
    uint64_t n_bytes;     /* 本次要读的总字节 */
    uint64_t base_off;    /* 起点偏移 */
    uint64_t stride;      /* 逐线程步长（字节） */
};

kernel void kernel_pureread_f32(
        constant pureread_args & args [[buffer(0)]],
        device const uchar *src [[buffer(1)]],
        device float *dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid == 0xFFFFFFFFu) return;
    const uint64_t per = 16u;
    uint64_t off = args.base_off + (uint64_t)gid * args.stride;
    if (off + per > args.base_off + args.n_bytes) return;
    /* 真读：一次 16 字节的向量化 load */
    device const uint4 *p = (device const uint4 *)(src + off);
    uint4 v = *p;
    uint acc = v.x ^ v.y ^ v.z ^ v.w;
    if (acc == 0xDEADBEEFu) dst[gid & 1023u] = 1.0f;
}

/* ── 2026-10-23：一次 dispatch 读 6 个散块（钉死 MoE 归因用）─────────────
 * 动机：用 6 次 kernel_pureread_f32 调用去模拟「6 个散专家」，测出来的
 *       是【调用次数】的差异（每次调用含固定开销），不是散读。
 *       要单独测「散读」，必须在【一个 dispatch 内】覆盖 6 个相距很远的区域。
 *
 * 线程到地址的映射：
 *   per_region_threads = region_bytes / 16
 *   r = gid / per_region_threads      （第几个区域，0..5）
 *   i = gid % per_region_threads      （区域内的线程号）
 *   off = offs[r] + i*16
 *
 * offsets 未填的区域（offs[r]==0 且 r>0）会被跳过，用来做 A 臂（连续）。
 */
struct pureread_scatter6_args {
    uint64_t offs[6];
    uint64_t region_bytes;      /* 每个区域的字节数（6 个区域相同） */
    uint32_t n_regions;         /* 1 = 连续（全读 offs[0] 起 region_bytes*6） */
    uint32_t pad;
};

kernel void kernel_pureread_scatter6(
        constant pureread_scatter6_args & args [[buffer(0)]],
        device const uchar *src [[buffer(1)]],
        device float *dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid == 0xFFFFFFFFu) return;

    uint64_t off;
    if (args.n_regions <= 1u) {
        /* A 臂：单块连续，region_bytes*6 当作一整段 */
        off = args.offs[0] + (uint64_t)gid * 16u;
    } else {
        const uint64_t npr = args.region_bytes / 16u;
        if (npr == 0u) return;
        const uint32_t r = (uint32_t)(gid / npr);
        const uint64_t i = (uint64_t)gid % npr;
        if (r >= args.n_regions || r >= 6u) return;
        off = args.offs[r] + i * 16u;
    }

    device const uint4 *p = (device const uint4 *)(src + off);
    uint4 v = *p;
    uint acc = v.x ^ v.y ^ v.z ^ v.w;
    if (acc == 0xDEADBEEFu) dst[gid & 1023u] = 1.0f;
}

/* ── 2026-10-23 P2：把「读 → 解码 → 累加」逐级加上去，定位 53% 的来源 ──────
 * 四臂共用同一套地址映射（6 个散区域、一次 dispatch），只差【计算量】：
 *   mode 0  只读，XOR 折起来                    ← §68 的 B 臂，522 GB/s
 *   mode 1  读 + 62 次 FMA，单累加器（依赖链）   ← 模拟每 16B = 62 个权重
 *   mode 2  读 + 62 次 FMA，4 累加器（可并行）
 *   mode 3  读 + 真实 IQ2 式解码：查表 + 符号 + 掩码 + FMA（依赖链）
 *
 * 每 16 B 对应 62 个权重（IQ2_XXS 2.0625 bpw ⇒ 0.2578 B/权重）。
 *
 * 判读：
 *   mode1 ≈ mode0            ⇒ FMA 不要钱，53% 来自解码指令
 *   mode1 << mode2           ⇒ 依赖链是主因（可改）
 *   mode3 ≈ mode0            ⇒ 解码指令也不要钱，53% 来自别处
 *   mode3 ≈ mode1 << mode0   ⇒ 算术/指令是主因
 */
struct pureread_fma6_args {
    uint64_t offs[6];
    uint64_t region_bytes;
    uint32_t n_regions;
    uint32_t mode;
};

#define PR_W_PER_U4 62

kernel void kernel_pureread_fma6(
        constant pureread_fma6_args & args [[buffer(0)]],
        device const uchar *src [[buffer(1)]],
        device float *dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_threadgroup]],
        ushort tgid  [[threadgroup_position_in_grid]]) {
    if (gid == 0xFFFFFFFFu) return;

    (void)tiisg; (void)tgid;

    uint64_t off;
    if (args.n_regions <= 1u) {
        off = args.offs[0] + (uint64_t)gid * 16u;
    } else {
        const uint64_t npr = args.region_bytes / 16u;
        if (npr == 0u) return;
        const uint32_t r = (uint32_t)(gid / npr);
        const uint64_t i = (uint64_t)gid % npr;
        if (r >= args.n_regions || r >= 6u) return;
        off = args.offs[r] + i * 16u;
    }

    device const uint4 *p = (device const uint4 *)(src + off);
    uint4 v = *p;

    float acc;
    if (args.mode == 0u) {
        uint x = v.x ^ v.y ^ v.z ^ v.w;
        if (x == 0xDEADBEEFu) dst[gid & 1023u] = 1.0f;
        return;
    } else if (args.mode == 1u) {
        /* 单累加器：每次 FMA 都依赖上一次 ⇒ 62 × 延迟 */
        float wf = (float)(v.y & 0xFFFFu) * 1e-6f;
        float c  = (float)(v.x & 0xFFFFu) * 1e-6f;
        acc = (float)(v.z & 0xFFFFu) * 1e-6f;
        #pragma unroll
        for (int i = 0; i < PR_W_PER_U4; ++i)
            acc = fma(acc, wf, c * (float)(i & 7) * 1e-7f);
    } else if (args.mode == 2u) {
        /* 4 个独立累加器：延迟被掩盖 */
        float wf = (float)(v.y & 0xFFFFu) * 1e-6f;
        float c  = (float)(v.x & 0xFFFFu) * 1e-6f;
        float a0 = (float)(v.z & 0xFFFFu) * 1e-6f;
        float a1 = a0 + 1e-6f, a2 = a0 + 2e-6f, a3 = a0 + 3e-6f;
        #pragma unroll
        for (int i = 0; i < 15; ++i) {
            const float k = c * (float)(i & 7) * 1e-7f;
            a0 = fma(a0, wf, k); a1 = fma(a1, wf, k);
            a2 = fma(a2, wf, k); a3 = fma(a3, wf, k);
        }
        a0 = fma(a0, wf, c); a1 = fma(a1, wf, c);
        acc = a0 + a1 + a2 + a3;
    } else if (args.mode == 3u) {
        /* mode 3：合成近似（含 64 位乘法冒充 LDS，偏重） */
        thread const uint8_t *b8 = (thread const uint8_t *)&v;
        float s = (float)(v.w & 0xFFu) * 1e-6f;
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            /* 网格表：用位运算冒充 8 项查表（真实路径是 threadgroup LDS） */
            const uint64_t g = ((uint64_t)b8[l] * 0x9E3779B97F4A7C15ull) ^ (uint64_t)l;
            const uint32_t s7 = (uint32_t)(g >> (7 * (l & 3))) & 127u;
            const int sign = (popcount(s7) & 1u) ? -1 : 1;   /* 真实路径的符号位 */
            const float base = (float)(g & 0xFFu) * 1e-6f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                /* 掩码 + 取反 + FMA，与 myds_iq2_v41_shape64_impl 内层同形 */
                const float gv = (sign < 0 && (s7 & (1u << (j & 7)))) ? -base : base;
                s = fma(s, gv, (float)((g >> (j & 7)) & 1u) * 1e-7f);
            }
        }
        acc = s;
    } else {
        /* mode 4：忠实形态 —— threadgroup 查表（真 LDS）+ 符号 popcount + 掩码取反 + FMA
         * 与 myds_iq2_v41_shape64_impl 的 8 元素内层一一对应，只是去掉指针算术。 */
        threadgroup uint64_t tbl[256];
        tbl[tiisg] = ((uint64_t)tiisg * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)tgid << 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        thread const uint8_t *b8 = (thread const uint8_t *)&v;
        float s = (float)(v.w & 0xFFu) * 1e-6f;
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const uint aux8 = b8[l];
            const threadgroup uint8_t *grid = (threadgroup const uint8_t *)(tbl + aux8);
            const uint s7 = aux8 & 127u;
            const uint8_t sign = (uint8_t)(s7 | ((popcount(s7) & 1u) << 7));
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float w = (float)grid[j] * 1e-6f;
                s = fma(s, w, (sign & (1u << (j & 7))) ? -1e-7f : 1e-7f);
            }
        }
        acc = s;
    }

    if (acc == 0xDEADBEEFp0f) dst[gid & 1023u] = acc;
}
