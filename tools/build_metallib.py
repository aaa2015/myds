#!/usr/bin/env python3
"""
build_metallib.py - Precompile all Metal shader sources into bin/myds.metallib.
Eliminates runtime pipeline compilation latency (~500ms) on server launch / first inference.
"""

import os
import sys
import subprocess
import argparse

SHADER_FILES = [
    "flash_attn.metal",
    "dense.metal",
    "moe.metal",
    "myds_iq2_v41.metal",
    "myds_q4_v41.metal",
    "dsv4_hc.metal",
    "unary.metal",
    "dsv4_kv.metal",
    "dsv41.metal",
    "myds_v41_moe.metal",
    "dsv4_rope.metal",
    "dsv4_misc.metal",
    "argsort.metal",
    "topk_threshold.metal",
    "cpy.metal",
    "concat.metal",
    "get_rows.metal",
    "sum_rows.metal",
    "softmax.metal",
    "repeat.metal",
    "glu.metal",
    "norm.metal",
    "bin.metal",
    "set_rows.metal",
    "myds_chunk.metal",
    "visenc.metal",
]

PREAMBLE = """#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#ifndef QK_K
#define QK_K 256
#endif
#define N_SIMDWIDTH 32
#ifndef N_R0_Q8_0
#define N_R0_Q8_0 2
#endif
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
"""

COPY_KERNELS = """
struct ds4_compute_copy_args {
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t count;
};
kernel void kernel_compute_copy_u8(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant ds4_compute_copy_args &args [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < args.count) {
        (dst + args.dst_offset)[gid] = (src + args.src_offset)[gid];
    }
}
kernel void kernel_compute_copy_u32(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant ds4_compute_copy_args &args [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < args.count) {
        device const uint *s = (device const uint *)(src + args.src_offset);
        device uint *d = (device uint *)(dst + args.dst_offset);
        d[gid] = s[gid];
    }
}
kernel void kernel_compute_copy_u32x4(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant ds4_compute_copy_args &args [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < args.count) {
        device const uint4 *s = (device const uint4 *)(src + args.src_offset);
        device uint4 *d = (device uint4 *)(dst + args.dst_offset);
        d[gid] = s[gid];
    }
}
"""

def main():
    parser = argparse.ArgumentParser(description="Compile Metal shaders into .metallib")
    parser.add_argument("--bin-dir", default=None, help="Directory to write myds.metallib")
    parser.add_argument("--build-dir", default=None, help="Directory to write intermediate .air and .metal files")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metal_dir = os.path.join(repo_root, "metal")
    build_dir = os.path.abspath(args.build_dir) if args.build_dir else os.path.join(repo_root, "build")
    bin_dir = os.path.abspath(args.bin_dir) if args.bin_dir else os.path.join(repo_root, "bin")

    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    combined_metal = os.path.join(build_dir, "myds_all.metal")
    air_file = os.path.join(build_dir, "myds_all.air")
    metallib_file = os.path.join(bin_dir, "myds.metallib")

    print("[metallib] Assembling full Metal shader source...")
    with open(combined_metal, "w", encoding="utf-8") as out:
        out.write(PREAMBLE + "\n")
        for fn in SHADER_FILES:
            src_path = os.path.join(metal_dir, fn)
            if not os.path.exists(src_path):
                print(f"Error: shader file not found: {src_path}", file=sys.stderr)
                sys.exit(1)
            with open(src_path, "r", encoding="utf-8") as sf:
                out.write(f'\n#line 1 "{fn}"\n' + sf.read())
        out.write(COPY_KERNELS + "\n")

    print(f"[metallib] Compiling AIR with xcrun metal -> {air_file}...")
    compile_cmd = [
        "xcrun", "-sdk", "macosx", "metal",
        "-mmacosx-version-min=15.0",
        "-O3", "-ffast-math",
        "-DDS4_METAL_HC_STABLE=1",
        "-DDS4_METAL_NORM_RSQRT_DISABLE=1",
        "-c", combined_metal,
        "-o", air_file
    ]
    res = subprocess.run(compile_cmd)
    if res.returncode != 0:
        print("[metallib] Compilation failed!", file=sys.stderr)
        sys.exit(res.returncode)

    print(f"[metallib] Linking library with xcrun metallib -> {metallib_file}...")
    link_cmd = [
        "xcrun", "-sdk", "macosx", "metallib",
        air_file,
        "-o", metallib_file
    ]
    res = subprocess.run(link_cmd)
    if res.returncode != 0:
        print("[metallib] Linking failed!", file=sys.stderr)
        sys.exit(res.returncode)

    size = os.path.getsize(metallib_file)
    print(f"[metallib] Successfully built {metallib_file} ({size:,} bytes).")

if __name__ == "__main__":
    main()
