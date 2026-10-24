#!/usr/bin/env python3
"""hadamard_probe.py — 判别「块状 Hadamard 旋转能不能让 IQ2_XXS 更准」。

背景
----
Prism ML 的 Ternary-Bonsai 用三步做到 1.72 bpw：
  三元权重 {−1,0,+1} + 每组 128 共享 FP16 scale + 【块状 Hadamard 旋转】
旋转把离群值摊平，是三元能成立的前提。旋转本身与量化类型无关，
所以可以先只试旋转，权重格式仍用 IQ2_XXS。

为什么不用导激活
----------------
旋转会把输入列混在一起，imatrix 的「逐列能量」在旋转后不再是逐列的。
但 H 正交，所以可以在原坐标里量等价误差：

    真实误差 = ||(W − Ŵ'Hᵀ)·diag(√imp)||_F
    （Ŵ' 是旋转后量化的权重）

不需要 x 向量，只需要 imp 这个逐列能量（imatrix 已提供）。

自证
----
· 不量化时 Ŵ' = W'，误差必须为 0（旋转可逆）——检验 H 的实现。
· 提供 --calibrate 时用 Q8_0 交叉验证解码器（Q8_0 应有 ~0.5% 相对误差）。

用法
----
  hadamard_probe.py <hf_dir> <imatrix.dat> <layer> [n_experts] [blocks...]
"""
import os
from pathlib import Path
import ctypes
import struct
import sys

import numpy as np

DS4 = os.environ.get("DS4_DIR", str(Path(__file__).resolve().parent.parent.parent / "ds4"))
sys.path.insert(0, f"{DS4}/gguf-tools")
from glm53_quantize import SourceDB          # noqa: E402
from deepseek41_quantize import NativeQuantizer, validate_scales  # noqa: E402

QTYPE_IQ2_XXS = 16

q = NativeQuantizer(f"{DS4}/gguf-tools/libds4quants.dylib")
# 解码器：IQ2_XXS 的 Metal 反量化在这里用 numpy 复刻（布局见 metal/moe.metal）
import re  # noqa: E402

MYDS = os.environ.get("MYDS_DIR", str(Path(__file__).resolve().parent.parent))


def load_tables():
    src = open(f"{MYDS}/metal/moe.metal").read()
    m = re.search(r"ds4_metal_iq2xxs_grid\[256\]\s*=\s*\{(.*?)\};", src, re.S)
    grid = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{16})", m.group(1))]
    m2 = re.search(r"ds4_metal_ksigns_iq2xs\[128\]\s*=\s*\{(.*?)\};", src, re.S)
    signs = np.array([int(x) for x in re.findall(r"\b(\d+)\b", m2.group(1))], np.uint8)
    gb = np.array([[(g >> (8 * i)) & 0xFF for i in range(8)] for g in grid], np.float32)
    return gb, signs


GRID, KSIGNS = load_tables()
KMASK = (1 << np.arange(8)).astype(np.uint8)


def dequant_iq2_xxs(raw, ncols):
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


def fwht(x, block):
    """沿最后一维按 block 分段做归一化 Walsh-Hadamard。x: (rows, n), n % block == 0。"""
    rows, n = x.shape
    nb = n // block
    a = x.reshape(rows, nb, block).astype(np.float32).copy()
    h = 1
    while h < block:
        a = a.reshape(rows, nb, block // (2 * h), 2, h)
        lo = a[:, :, :, 0, :].copy()
        hi = a[:, :, :, 1, :].copy()
        a[:, :, :, 0, :] = lo + hi
        a[:, :, :, 1, :] = lo - hi
        a = a.reshape(rows, nb, block)
        h *= 2
    a /= np.float32(np.sqrt(block))
    return a.reshape(rows, n)


def load_imatrix(path):
    entries = {}
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        for _ in range(n):
            nl = struct.unpack("<i", f.read(4))[0]
            name = f.read(nl).decode()
            f.read(4)
            vc = struct.unpack("<i", f.read(4))[0]
            entries[name] = np.frombuffer(f.read(vc * 4), dtype="<f4").copy()
    return entries


def main():
    hf, dat, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_exp = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    blocks = [int(x) for x in sys.argv[5:]] or [256, 1024]

    db = SourceDB(hf, index_validator=lambda _: None, scale_validator=validate_scales)
    imx = load_imatrix(dat)
    key = f"blk.{layer}.ffn_gate_exps.weight"
    imp_all = imx[key]
    ncols = imp_all.size // 384
    per = imp_all.reshape(384, ncols)
    live = [e for e in range(384) if per[e].sum() > 0]
    print(f"  {key}: 384 x {ncols}, 有样本 {len(live)}")
    picks = live[:: max(1, len(live) // n_exp)][:n_exp]

    # 自证：H 可逆
    probe = np.random.RandomState(0).randn(4, ncols).astype(np.float32)
    for b in blocks:
        rt = fwht(fwht(probe, b), b)
        print(f"  H roundtrip block={b}: max|Δ|={np.abs(rt - probe).max():.3e}")

    rows = []
    for e in picks:
        name = f"layers.{layer}.ffn.experts.{e}.w1.weight"
        w = np.ascontiguousarray(q.to_f32(db, name), dtype=np.float32)
        if w.shape[1] != ncols:
            sys.exit(f"{name} shape {w.shape} vs ncols {ncols}")
        imp = per[e].astype(np.float32)
        sq = np.sqrt(imp).astype(np.float32)

        # 臂 A：现状（不旋转）
        rawA = q.encode(w, QTYPE_IQ2_XXS, None)
        wA = dequant_iq2_xxs(rawA, ncols)
        errA = float((((w - wA) ** 2) * imp[None, :]).sum() / imp.sum())

        rec = dict(expert=e, errA=errA, blocks={})
        for b in blocks:
            if ncols % b:
                continue
            wr = fwht(w, b)                      # W' = W H
            rawB = q.encode(wr, QTYPE_IQ2_XXS, None)
            wrq = dequant_iq2_xxs(rawB, ncols)
            weff = fwht(wrq, b)                  # Ŵ' Hᵀ（H 对称且正交）
            errB = float((((w - weff) ** 2) * imp[None, :]).sum() / imp.sum())
            # 臂 D：旋转 + 激活重要性（旋转基里 diag(Σ') 每块为常数）
            imp_rot = np.repeat(imp.reshape(-1, b).mean(axis=1), b).astype(np.float32)
            rawD = q.encode(wr, QTYPE_IQ2_XXS, imp_rot)
            wD = fwht(dequant_iq2_xxs(rawD, ncols), b)
            errD = float((((w - wD) ** 2) * imp[None, :]).sum() / imp.sum())
            rec["blocks"][b] = errB
            rec.setdefault("rot_imx", {})[b] = errD
        rows.append(rec)
        line = f"    e{e:<3} A={errA:.6e}"
        for b in sorted(rec["blocks"]):
            v = rec["blocks"][b]
            line += f"  H{b}={100*(v/errA-1):+6.2f}%  H{b}+imx={100*(rec['rot_imx'][b]/errA-1):+6.2f}%"
        print(line)

    import statistics as st
    print()
    print("  ── 中位数 ──")
    base = st.median(r["errA"] for r in rows)
    print(f"    现状 A = {base:.6e}")
    for b in blocks:
        rs = [r for r in rows if b in r["blocks"]]
        if not rs:
            continue
        med = st.median(r["blocks"][b] for r in rs)
        medD = st.median(r["rot_imx"][b] for r in rs)
        wins = sum(1 for r in rs if r["blocks"][b] < r["errA"])
        winsD = sum(1 for r in rs if r["rot_imx"][b] < r["errA"])
        print(f"    Hadamard-{b:<4}      = {med:.6e}  ({100*(med/base-1):+6.2f}%)  赢 {wins}/{len(rs)}")
        print(f"    Hadamard-{b:<4}+imatrix = {medD:.6e}  ({100*(medD/base-1):+6.2f}%)  赢 {winsD}/{len(rs)}")


if __name__ == "__main__":
    main()
