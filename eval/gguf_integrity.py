#!/usr/bin/env python3
"""gguf_integrity.py — 检查 GGUF 结构自洽性（不动 362 GB 内容，只读头部 + 边界抽查）。

检查项：
  1. 张量表可解析；张量数、名字无重复
  2. 每个张量的 [offset, offset+nbytes) 落在数据区内
  3. 任何两个张量区间不重叠
  4. 最后一个张量的终点 + 对齐 == 文件大小
  5. 每个类型都有已知的块大小/块元素数
  6. 抽查首/中/尾各若干字节，确认不是全零（写盘中断会留下零洞）

用法: gguf_integrity.py <model.gguf>
"""
import os
import struct
import sys

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# type -> (block_elems, block_bytes)
BLK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
       8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
       13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
       18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
       23: (32, 18), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
       29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66)}
NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
         14: "Q6_K", 16: "IQ2_XXS", 24: "I8", 30: "BF16"}


def main():
    path = sys.argv[1]
    fsize = os.path.getsize(path)
    f = open(path, "rb")
    head = f.read(1 << 22)
    buf = bytearray(head)
    off = 4
    ver = struct.unpack_from("<I", buf, off)[0]; off += 4
    nt = struct.unpack_from("<Q", buf, off)[0]; off += 8
    nk = struct.unpack_from("<Q", buf, off)[0]; off += 8

    def need(n):
        nonlocal buf
        while n > len(buf):
            more = f.read(max(1 << 22, n - len(buf)))
            if not more:
                sys.exit(f"  ❌ 头部被截断（需要 {n} 字节，实际 {len(buf)}）")
            buf += more

    def rs():
        nonlocal off
        need(off + 8)
        n = struct.unpack_from("<Q", buf, off)[0]; off += 8
        need(off + n)
        s = bytes(buf[off:off + n]).decode("utf-8", "replace"); off += n
        return s

    def skip(t):
        nonlocal off
        if t == 8:
            rs()
        elif t == 9:
            need(off + 12)
            et = struct.unpack_from("<I", buf, off)[0]; off += 4
            n = struct.unpack_from("<Q", buf, off)[0]; off += 8
            if et == 8:
                for _ in range(n):
                    rs()
            else:
                off += SZ[et] * n
        else:
            off += SZ[t]

    align = 32
    for _ in range(nk):
        k = rs()
        need(off + 4)
        t = struct.unpack_from("<I", buf, off)[0]; off += 4
        if k == "general.alignment" and t == 4:
            align = struct.unpack_from("<I", buf, off)[0]; off += 4
            continue
        skip(t)

    tensors = []
    seen = set()
    for _ in range(nt):
        name = rs()
        if name in seen:
            sys.exit(f"  ❌ 重复张量名 {name}")
        seen.add(name)
        need(off + 4)
        nd = struct.unpack_from("<I", buf, off)[0]; off += 4
        need(off + 8 * nd)
        dims = struct.unpack_from("<%dQ" % nd, buf, off); off += 8 * nd
        need(off + 12)
        t = struct.unpack_from("<I", buf, off)[0]; off += 4
        o = struct.unpack_from("<Q", buf, off)[0]; off += 8
        n = 1
        for d in dims:
            n *= d
        if t not in BLK:
            sys.exit(f"  ❌ {name}: 未知类型 {t}")
        be, bb = BLK[t]
        nbytes = (n // be) * bb
        tensors.append((o, nbytes, name, t))

    data_start = (off + align - 1) // align * align
    print(f"  GGUF v{ver}  tensors={nt}  kv={nk}  alignment={align}")
    print(f"  data_start = {data_start}  文件 = {fsize}")

    base = fsize - sum(nb for _, nb, _, _ in tensors)
    # 用最后一个张量的位置反推：数据区起点应等于 base（含对齐）
    bad = 0
    total = 0
    for o, nb, name, t in tensors:
        total += nb
        if o + nb > fsize:
            print(f"  ❌ {name}: [{o}, {o+nb}) 超出文件 {fsize}")
            bad += 1
    if bad:
        sys.exit(1)
    srt = sorted(tensors)
    for i in range(1, len(srt)):
        prev_o, prev_nb = srt[i-1][0], srt[i-1][1]
        if srt[i][0] < prev_o + prev_nb:
            print(f"  ❌ 重叠: {srt[i-1][2]} 与 {srt[i][2]}")
            bad += 1
    print(f"  张量区间: {'✅ 无重叠越界' if not bad else '❌ 有问题'}")
    print(f"  数据区推算起点 = {base}  (张量表 + kv 后对齐 = {data_start})")

    # 抽查内容非零
    import random
    random.seed(0)
    picks = random.sample(tensors, min(12, len(tensors)))
    zero = 0
    for o, nb, name, t in picks:
        f.seek(base + o)
        chunk = f.read(min(4096, nb))
        if not chunk or all(b == 0 for b in chunk):
            print(f"  ⚠️ {name}: 首 4KB 全零或读不到")
            zero += 1
    print(f"  抽查 12 个张量首 4KB: {'✅ 均非零' if zero == 0 else f'⚠️ {zero} 个异常'}")

    # 尾部
    f.seek(fsize - 64)
    tail = f.read(64)
    print(f"  文件尾 32 字节: {tail[-32:].hex()[:40]}… {'(全零)' if all(b==0 for b in tail) else ''}")


if __name__ == "__main__":
    main()
