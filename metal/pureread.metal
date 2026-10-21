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
