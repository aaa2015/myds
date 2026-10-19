#!/usr/bin/env python3
"""gguf_kv.py — 打印 GGUF 的字符串型元数据（用于核对 imatrix / revision / 名称）。

用法: gguf_kv.py <model.gguf> [--filter substr]

只按需读取，不会把 362 GB 文件读进内存。
"""
import mmap
import os
import struct
import sys

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_kv(path):
    fsize = os.path.getsize(path)
    f = open(path, "rb")
    # 头部元数据通常 < 8 MB；mmap 整个文件但不触碰数据区页
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    off = 4
    ver = struct.unpack_from("<I", mm, off)[0]
    off += 4
    nt = struct.unpack_from("<Q", mm, off)[0]
    off += 8
    nk = struct.unpack_from("<Q", mm, off)[0]
    off += 8

    def rs():
        nonlocal off
        n = struct.unpack_from("<Q", mm, off)[0]
        off += 8
        if off + n > fsize:
            raise ValueError("truncated")
        s = mm[off:off + n].decode("utf-8", "replace")
        off += n
        return s

    def skip(t):
        nonlocal off
        if t == 8:
            rs()
        elif t == 9:
            et = struct.unpack_from("<I", mm, off)[0]
            off += 4
            n = struct.unpack_from("<Q", mm, off)[0]
            off += 8
            if et == 8:
                for _ in range(n):
                    rs()
            else:
                off += SZ[et] * n
        else:
            off += SZ[t]

    kv = {}
    for _ in range(nk):
        k = rs()
        t = struct.unpack_from("<I", mm, off)[0]
        off += 4
        if t == 8:
            kv[k] = rs()
        else:
            skip(t)
    mm.close()
    f.close()
    return ver, nt, nk, kv


def main():
    path = sys.argv[1]
    filt = None
    if "--filter" in sys.argv:
        filt = sys.argv[sys.argv.index("--filter") + 1]
    ver, nt, nk, kv = read_kv(path)
    print(f"  {os.path.basename(path)}")
    print(f"    {os.path.getsize(path)} B  v{ver}  tensors={nt}  kv={nk}")
    for k in sorted(kv):
        if filt and filt not in k:
            continue
        v = kv[k]
        if isinstance(v, str) and len(v) > 90:
            v = v[:87] + "..."
        print(f"    {k:<34} = {v}")


if __name__ == "__main__":
    main()
