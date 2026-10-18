#!/usr/bin/env python3
"""imatrix_gguf_to_dat.py — 把 llama.cpp 新版 imatrix.gguf 转成我们用的 legacy .dat。

两种格式的差别
--------------
gguf（新版，llama.cpp #28696 之后）:
  每个被统计的张量有两组条目：
    <name>.in_sum2   float32 [ncols] 或 [ncols, n_expert]
    <name>.counts    float32 [1]     或 [1, n_expert]
  表示 sum(x_j^2) 与样本数。重要性需自行归一。

.dat（legacy，我们的量化器读这个）:
  int32 n_entries
  for each: int32 name_len, name, int32 ncall, int32 nvalues, float32[nvalues]
  对 routed experts 是【按专家拼接】的单个数组，长度 n_expert * ncols。

转换规则：imp[expert][col] = in_sum2[col][expert] / max(counts[expert], 1)
  · 除以 counts 得到「平均激活能量」，让样本多的专家与样本少的可比
  · counts 为 0 的专家整段写 0，量化器会把它当成「无数据」并回退
"""
import struct
import sys

import numpy as np

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf(path):
    buf = bytearray(open(path, "rb").read())
    off = 4
    off += 4
    nt = struct.unpack_from("<Q", buf, off)[0]; off += 8
    nk = struct.unpack_from("<Q", buf, off)[0]; off += 8

    def rs():
        nonlocal off
        n = struct.unpack_from("<Q", buf, off)[0]; off += 8
        s = bytes(buf[off:off + n]).decode("utf-8", "replace"); off += n
        return s

    def skip(t):
        nonlocal off
        if t == 8:
            rs()
        elif t == 9:
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
        t = struct.unpack_from("<I", buf, off)[0]; off += 4
        if k == "general.alignment" and t == 4:
            align = struct.unpack_from("<I", buf, off)[0]; off += 4
            continue
        skip(t)

    info = {}
    for _ in range(nt):
        name = rs()
        nd = struct.unpack_from("<I", buf, off)[0]; off += 4
        dims = struct.unpack_from("<%dQ" % nd, buf, off); off += 8 * nd
        t = struct.unpack_from("<I", buf, off)[0]; off += 4
        o = struct.unpack_from("<Q", buf, off)[0]; off += 8
        info[name] = (dims, t, o)
    data_start = (off + align - 1) // align * align
    return buf, info, data_start


def main():
    src, dst = sys.argv[1], sys.argv[2]
    buf, info, data_start = read_gguf(src)
    print(f"  {len(info)} tensors, data_start={data_start}")

    # 按基础张量名聚合
    tensors = {}
    for name in info:
        for suffix in (".in_sum2", ".counts"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                tensors.setdefault(base, {})[suffix[1:]] = name
                break

    entries = []   # (name, np.array float32)
    skipped = []
    for base in sorted(tensors):
        d = tensors[base]
        if "in_sum2" not in d or "counts" not in d:
            skipped.append((base, "missing in_sum2/counts"))
            continue
        dims_s, t_s, o_s = info[d["in_sum2"]]
        dims_c, t_c, o_c = info[d["counts"]]
        if t_s != 0 or t_c != 0:
            skipped.append((base, f"unexpected dtype {t_s}/{t_c}"))
            continue
        n_s = 1
        for x in dims_s:
            n_s *= x
        n_c = 1
        for x in dims_c:
            n_c *= x
        sum2 = np.frombuffer(buf, dtype="<f4", count=n_s, offset=data_start + o_s).astype(np.float64)
        cnt = np.frombuffer(buf, dtype="<f4", count=n_c, offset=data_start + o_c).astype(np.float64)

        if len(dims_s) == 2:
            ncols, nexp = int(dims_s[0]), int(dims_s[1])
            if n_s != ncols * nexp:
                skipped.append((base, "shape mismatch"))
                continue
            # GGUF dims 是【最快变化维度在前】：dims=(ncols, nexp) 表示
            # 扁平数组按 expert-major 排布，每个专家 ncols 个连续值。
            m = sum2.reshape(nexp, ncols)          # [expert][col]
            c = cnt.reshape(-1)
            if c.size != nexp:
                skipped.append((base, f"counts {c.size} vs experts {nexp}"))
                continue
            safe = np.where(c > 0, c, 1.0)
            norm = m / safe[:, None]               # 每专家平均激活能量
            norm[c <= 0, :] = 0.0                  # 无样本 ⇒ 0（量化器回退）
            flat = norm.reshape(-1).astype("<f4")  # 专家优先拼接，与 .dat 约定一致
        else:
            c = float(cnt.reshape(-1)[0]) if n_c else 0.0
            if c <= 0:
                continue
            flat = (sum2 / c).astype("<f4")
        entries.append((base, flat))

    print(f"  转换 {len(entries)} 条，跳过 {len(skipped)} 条")
    for b, why in skipped[:6]:
        print(f"    跳过 {b}: {why}")

    with open(dst, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, vals in entries:
            nb = name.encode()
            f.write(struct.pack("<i", len(nb)))
            f.write(nb)
            f.write(struct.pack("<i", 1))          # ncall
            f.write(struct.pack("<i", vals.size))
            f.write(vals.tobytes())
    print(f"  写出 {dst}")


if __name__ == "__main__":
    main()
