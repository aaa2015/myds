#!/usr/bin/env python3
"""imatrix_cross.py — 两份 imatrix 谁更好：交叉评估。

设计
----
两臂都从同一源权重量化同一张量，只是重要性不同：
  A = 用 imatrix_A 量化
  B = 用 imatrix_B 量化
然后用【对方的】重要性作为评估权重：
  err(A | w_B)  vs  err(B | w_B)     ← B 的参照系
  err(A | w_A)  vs  err(B | w_A)     ← A 的参照系

真正的「更好」必须在【对方的参照系】里也赢。如果各自只在自己的参照系里赢，
说明两份 imatrix 只是在强调不同的列，没有绝对优劣。

额外提供一个中性参照：全列均匀权重（不偏向任何一方）。
"""
import os
from pathlib import Path
import struct
import sys

import numpy as np

DS4 = os.environ.get("DS4_DIR", str(Path(__file__).resolve().parent.parent.parent / "ds4"))
sys.path.insert(0, f"{DS4}/gguf-tools")
from glm53_quantize import SourceDB                              # noqa: E402
from deepseek41_quantize import NativeQuantizer, validate_scales  # noqa: E402
import re                                                         # noqa: E402

MYDS = os.environ.get("MYDS_DIR", str(Path(__file__).resolve().parent.parent))
q = NativeQuantizer(f"{DS4}/gguf-tools/libds4quants.dylib")
QTYPE_IQ2_XXS = 16


def load_tables():
    src = open(f"{MYDS}/metal/moe.metal").read()
    m = re.search(r"ds4_metal_iq2xxs_grid\[256\]\s*=\s*\{(.*?)\};", src, re.S)
    grid = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{16})", m.group(1))]
    m2 = re.search(r"ds4_metal_ksigns_iq2xs\[128\]\s*=\s*\{(.*?)\};", src, re.S)
    signs = np.array([int(x) for x in re.findall(r"\b(\d+)\b", m2.group(1))], np.uint8)
    return (np.array([[(g >> (8 * i)) & 0xFF for i in range(8)] for g in grid], np.float32),
            signs)


GRID, KSIGNS = load_tables()
KMASK = (1 << np.arange(8)).astype(np.uint8)


def dequant(raw, ncols):
    per = (ncols // 256) * 66
    nr = len(raw) // per
    b = np.frombuffer(raw, np.uint8, count=nr * per).reshape(nr, -1, 66)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
    qs = b[:, :, 2:66].copy().view(np.uint16)
    out = np.empty((nr, ncols // 256, 256), np.float32)
    for il in range(16):
        ib, il2 = il // 2, il % 2
        q2 = qs[:, :, 4 * ib:4 * ib + 4].astype(np.uint32)
        ag = q2[:, :, 0] | (q2[:, :, 1] << 16)
        as_ = q2[:, :, 2] | (q2[:, :, 3] << 16)
        a8 = np.stack([(ag >> (8 * i)) & 255 for i in range(4)], axis=-1).astype(np.uint8)
        dl = d[:, :, 0] * (0.5 + (as_ >> 28).astype(np.float32)) * 0.25
        for half in range(2):
            idx = a8[:, :, 2 * il2 + half].astype(np.int64)
            sb = KSIGNS[((as_ >> np.uint32(14 * il2 + 7 * half)) & 127).astype(np.int64)]
            sg = np.where((sb[:, :, None] & KMASK[None, None, :]) != 0, -1.0, 1.0)
            s = il * 16 + half * 8
            out[:, :, s:s + 8] = (dl[:, :, None] * GRID[idx] * sg).astype(np.float32)
    return out.reshape(nr, ncols)


def load(p):
    d = {}
    with open(p, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        for _ in range(n):
            nl = struct.unpack("<i", f.read(4))[0]
            name = f.read(nl).decode()
            f.read(4)
            vc = struct.unpack("<i", f.read(4))[0]
            d[name] = np.frombuffer(f.read(vc * 4), dtype="<f4").copy()
    return d


def wmse(w, x, imp):
    return float((((w - x) ** 2) * imp[None, :]).sum() / imp.sum())


def main():
    datA, datB, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nameA, nameB = sys.argv[4] if len(sys.argv) > 4 else "A", "B"
    n_exp = int(sys.argv[6]) if len(sys.argv) > 6 else 6

    db = SourceDB(os.environ.get("MYDS_MODEL_DIR", str(Path.home() / "hf-1017")), index_validator=lambda _: None,
                  scale_validator=validate_scales)
    A = load(datA)
    B = load(datB)
    key = f"blk.{layer}.ffn_gate_exps.weight"
    ia, ib = A[key], B[key]
    ncols = ia.size // 384
    ra, rb = ia.reshape(384, ncols).astype(np.float32), ib.reshape(384, ncols).astype(np.float32)
    live = [e for e in range(384) if ra[e].sum() > 0 and rb[e].sum() > 0]
    picks = live[:: max(1, len(live) // n_exp)][:n_exp]
    print(f"  {key}: {len(live)} 专家有双方样本，取 {len(picks)} 个")

    rows = []
    for e in picks:
        w = np.ascontiguousarray(q.to_f32(db, f"layers.{layer}.ffn.experts.{e}.w1.weight"),
                                 dtype=np.float32)
        if w.shape[1] != ncols:
            w = w.T
        wa = dequant(q.encode(w, QTYPE_IQ2_XXS, ra[e]), ncols)
        wb = dequant(q.encode(w, QTYPE_IQ2_XXS, rb[e]), ncols)
        uni = np.ones(ncols, np.float32)

        r = dict(e=e,
                 a_on_a=wmse(w, wa, ra[e]), b_on_a=wmse(w, wb, ra[e]),
                 a_on_b=wmse(w, wa, rb[e]), b_on_b=wmse(w, wb, rb[e]),
                 a_on_u=wmse(w, wa, uni), b_on_u=wmse(w, wb, uni))
        rows.append(r)

    def med(k):
        import statistics as st
        return st.median(r[k] for r in rows)

    def wins(k_a, k_b):
        return sum(1 for r in rows if r[k_a] < r[k_b])

    print()
    print(f"  {'评估参照':<14} {'A':>13} {'B':>13} {'B/A':>9}  谁赢")
    for ref, ka, kb in ((f"{nameA} 的 imp", "a_on_a", "b_on_a"),
                        (f"{nameB} 的 imp", "a_on_b", "b_on_b"),
                        ("均匀", "a_on_u", "b_on_u")):
        a, b = med(ka), med(kb)
        w = wins(ka, kb)
        print(f"  {ref:<14} {a:13.6e} {b:13.6e} {100*(b/a-1):+8.2f}%  A 胜 {w}/{len(rows)}")


if __name__ == "__main__":
    main()
