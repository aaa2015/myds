#!/usr/bin/env python3
"""imatrix_probe.py — 判别「补 imatrix 对 IQ2_XXS 专家量化有没有用」。

原理
----
生产模型没有 --imatrix。但量化器对 IQ2_XXS 有内建回退：

    if qtype == IQ2_XXS and imatrix is None:
        imatrix = sum(w**2, axis=0)

也就是说它并非「均匀权重」，而是按【权重列能量】分配比特。
补 imatrix 等于把重要性从「权重能量」换成「激活能量」。

本工具对同一批源专家权重做两次量化（A=权重能量，B=激活能量），
反量化后比较重构误差。IQ2_XXS 的反量化在这里用 numpy 复刻，
码表直接从 metal/moe.metal 里解析，避免手抄。

自证
----
解码器正确性用「解码后再量化应得到逐字节相同的块」检验：
量化器的 C 实现在这条路径上与解码器无共享代码，
若某个块的 bytes 能还原，说明解码布局（d / qs / grid / signs）对齐。

用法
----
  imatrix_probe.py <hf_dir> <model.gguf> <imatrix.dat> <layer> [n_experts]
"""
import os
from pathlib import Path
import ctypes
import json
import re
import struct
import sys

import numpy as np

DS4 = os.environ.get("DS4_DIR", str(Path(__file__).resolve().parent.parent.parent / "ds4"))
MYDS = os.environ.get("MYDS_DIR", str(Path(__file__).resolve().parent.parent))

QTYPE_IQ2_XXS = 16
BLOCK = 256
BLOCK_BYTES = 66


# ── 码表：从 moe.metal 解析，不手抄 ────────────────────────────────
def load_tables():
    src = open(f"{MYDS}/metal/moe.metal").read()
    m = re.search(r"ds4_metal_iq2xxs_grid\[256\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        sys.exit("cannot find iq2xxs_grid")
    grid = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{16})", m.group(1))]
    if len(grid) != 256:
        sys.exit(f"grid has {len(grid)} entries, expected 256")
    m2 = re.search(r"ds4_metal_ksigns_iq2xs\[128\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m2:
        sys.exit("cannot find ksigns_iq2xs")
    signs = [int(x) for x in re.findall(r"\b(\d+)\b", m2.group(1))]
    if len(signs) != 128:
        sys.exit(f"ksigns has {len(signs)} entries, expected 128")
    # grid → (256, 8) uint8，小端字节序（C 里按 uint8_t* 下标访问）
    grid_bytes = np.array(
        [[(g >> (8 * i)) & 0xFF for i in range(8)] for g in grid], dtype=np.float32
    )
    return grid_bytes, np.array(signs, dtype=np.uint8)


GRID, KSIGNS = load_tables()
KMASK = (1 << np.arange(8)).astype(np.uint8)


def dequant_iq2_xxs(raw, ncols):
    """raw: 每行 (ncols/256)*66 字节。返回 (nrows, ncols) float32。"""
    per_row = (ncols // BLOCK) * BLOCK_BYTES
    nrows = len(raw) // per_row
    b = np.frombuffer(raw, dtype=np.uint8, count=nrows * per_row).reshape(nrows, -1)
    nb = ncols // BLOCK
    b = b.reshape(nrows, nb, BLOCK_BYTES)

    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)  # (r, nb, 1)
    qs = b[:, :, 2:66].copy().view(np.uint16)                   # (r, nb, 32)

    out = np.empty((nrows, nb, BLOCK), dtype=np.float32)
    for il in range(16):
        ib32, il2 = il // 2, il % 2
        q2 = qs[:, :, 4 * ib32:4 * ib32 + 4].astype(np.uint32)  # (r, nb, 4)
        aux_g = q2[:, :, 0] | (q2[:, :, 1] << 16)
        aux_s = q2[:, :, 2] | (q2[:, :, 3] << 16)
        aux8 = np.stack([(aux_g >> (8 * i)) & 0xFF for i in range(4)], axis=-1).astype(np.uint8)
        dl = d[:, :, 0] * (0.5 + (aux_s >> 28).astype(np.float32)) * 0.25

        for half in range(2):
            idx = aux8[:, :, 2 * il2 + half].astype(np.int64)
            sbits = KSIGNS[((aux_s >> np.uint32(14 * il2 + 7 * half)) & 127).astype(np.int64)]
            sign = np.where((sbits[:, :, None] & KMASK[None, None, :]) != 0, -1.0, 1.0)
            out[:, :, il * 16 + half * 8: il * 16 + half * 8 + 8] = (
                dl[:, :, None] * GRID[idx] * sign
            ).astype(np.float32)
    return out.reshape(nrows, ncols)


# ── 源权重 ────────────────────────────────────────────────────────
sys.path.insert(0, f"{DS4}/gguf-tools")
from glm53_quantize import SourceDB  # noqa: E402
from deepseek41_quantize import NativeQuantizer, validate_scales  # noqa: E402

q = NativeQuantizer(f"{DS4}/gguf-tools/libds4quants.dylib")
db = SourceDB(sys.argv[1], index_validator=lambda _: None,
              scale_validator=validate_scales)


def load_imatrix(path):
    entries = {}
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        for _ in range(n):
            nl = struct.unpack("<i", f.read(4))[0]
            name = f.read(nl).decode()
            f.read(4)  # call count
            vc = struct.unpack("<i", f.read(4))[0]
            entries[name] = np.frombuffer(f.read(vc * 4), dtype="<f4").copy()
    return entries


def scale_name(name):
    return name.removesuffix(".weight") + ".scale"


def to_f32_any(db, name):
    """NativeQuantizer 已处理 I8(4-bit 打包)+E8M0 与 F8_E4M3。"""
    return q.to_f32(db, name)


def encode(values, imatrix):
    return q.encode(values, QTYPE_IQ2_XXS, imatrix)


def main():
    hf, gguf, dat, layer = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    n_exp = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    imx = load_imatrix(dat)
    gate_key = f"blk.{layer}.ffn_gate_exps.weight"
    if gate_key not in imx:
        sys.exit(f"no imatrix entry {gate_key}")
    imp_all = imx[gate_key]
    ncols_imx = imp_all.size // 384
    print(f"  imatrix {gate_key}: {imp_all.size} values = 384 x {ncols_imx}")

    # 找出有样本的专家（向量非全零）
    per = imp_all.reshape(384, ncols_imx)
    live = [e for e in range(384) if per[e].sum() > 0]
    print(f"  有样本的专家: {len(live)}/384")
    picks = live[:: max(1, len(live) // n_exp)][:n_exp]

    rows = []
    for e in picks:
        name = f"layers.{layer}.ffn.experts.{e}.w1.weight"
        w = to_f32_any(db, name)
        # 定向：最后一维 = 输入列（= imatrix 宽度）
        if w.shape[-1] != ncols_imx:
            if w.shape[0] == ncols_imx:
                w = w.T
            else:
                sys.exit(f"{name} shape {w.shape} cannot match {ncols_imx}")
        w = np.ascontiguousarray(w, dtype=np.float32)
        nrows = w.size // ncols_imx
        w2 = w.reshape(nrows, ncols_imx)

        imp_act = per[e].astype(np.float32)
        # 必须与库内回退公式逐位一致：np.square(float32).sum(axis=0, dtype=float32)
        imp_w = np.square(w2, dtype=np.float32).sum(axis=0, dtype=np.float32)

        raw_a = encode(w2, None)            # 现状：权重能量
        raw_b = encode(w2, imp_act)         # 候选：激活能量

        da = dequant_iq2_xxs(raw_a, ncols_imx)
        dbq = dequant_iq2_xxs(raw_b, ncols_imx)

        # 自证：解码后再量化应逐字节还原
        # 显式传回原重要性：encode(None) 会从【解码后】权重重算，不可比
        idem_a = encode(da, imp_w) == raw_a
        idem_b = encode(dbq, imp_act) == raw_b

        def wmse(ref, got, imp):
            diff = (ref - got).astype(np.float64) ** 2
            return float((diff * imp[None, :]).sum() / imp.sum())

        rows.append(dict(
            expert=e, shape=tuple(w2.shape),
            idem_a=idem_a, idem_b=idem_b,
            mse_a=float(((w2 - da) ** 2).mean()),
            mse_b=float(((w2 - dbq) ** 2).mean()),
            act_a=wmse(w2, da, imp_act), act_b=wmse(w2, dbq, imp_act),
            w_a=wmse(w2, da, imp_w), w_b=wmse(w2, dbq, imp_w),
            bytes_same=raw_a == raw_b,
        ))
        r = rows[-1]
        print(f"    e{e:<3} idem={r['idem_a']}/{r['idem_b']} "
              f"act_err A={r['act_a']:.6e} B={r['act_b']:.6e} "
              f"({100*(r['act_b']/r['act_a']-1):+.2f}%)  "
              f"w_err A={r['w_a']:.6e} B={r['w_b']:.6e} "
              f"({100*(r['w_b']/r['w_a']-1):+.2f}%)")

    import statistics as st
    def med(k):
        return st.median(r[k] for r in rows)
    print()
    print("  ── 汇总（中位数）──")
    print(f"    自证 idempotent: A={sum(r['idem_a'] for r in rows)}/{len(rows)}  "
          f"B={sum(r['idem_b'] for r in rows)}/{len(rows)}")
    print(f"    激活加权误差: A={med('act_a'):.6e}  B={med('act_b'):.6e}  "
          f"→ {100*(med('act_b')/med('act_a')-1):+.2f}%")
    print(f"    权重加权误差: A={med('w_a'):.6e}  B={med('w_b'):.6e}  "
          f"→ {100*(med('w_b')/med('w_a')-1):+.2f}%")
    print(f"    朴素 MSE    : A={med('mse_a'):.6e}  B={med('mse_b'):.6e}  "
          f"→ {100*(med('mse_b')/med('mse_a')-1):+.2f}%")
    print(f"    量化字节完全相同: {sum(r['bytes_same'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
