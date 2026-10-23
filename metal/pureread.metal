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
