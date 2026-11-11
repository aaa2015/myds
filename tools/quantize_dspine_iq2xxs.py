#!/usr/bin/env python3
"""Quantize a BF16/F32 DSpine .safetensors checkpoint to native GGML IQ2_XXS.

Heavy 2D projection matrices (cols % 256 == 0 and cols >= 5120):
  - fc.weight                       [5120, 25600]
  - layers.{0,1,2}.{q,k,v,o}_proj   [5120|1024, 5120]
  - layers.{0,1,2}.{gate,up}_proj   [15360, 5120]
  - layers.{0,1,2}.down_proj        [5120, 15360]
  - injection.readouts.{0,1,2}      [512, 5120]
are quantized into GGML block_iq2_xxs (66 bytes per 256 elements = 2.0625 bpw)
using libds4quants with real activation imatrix calibration (E[x^2]) from
captured DeepSeek-V4.1 shards when --shards-dir is supplied.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import glob
import json
import math
import os
from pathlib import Path
import struct
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


DS4Q_TYPE_IQ2_XXS = 16
IQ2_MIN_COLS = 5120


def find_quants_lib(explicit: Optional[str]) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_p = os.environ.get("MYDS_QUANTS_LIB")
    if env_p:
        candidates.append(env_p)
    home = Path.home()
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend([
        str(home / "Projects/ds4/gguf-tools/libds4quants.dylib"),
        str(repo_root.parent / "ds4/gguf-tools/libds4quants.dylib"),
    ])
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find libds4quants.dylib in {candidates}")


class DS4Quants:
    def __init__(self, lib_path: str) -> None:
        self.lib = ctypes.CDLL(lib_path)
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_init.restype = None

        self.lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ds4q_row_size.restype = ctypes.c_size_t

        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init(DS4Q_TYPE_IQ2_XXS)

    def quantize_iq2_xxs(
        self,
        w_f32: np.ndarray,
        imatrix_f32: np.ndarray,
        num_workers: int = 16,
    ) -> np.ndarray:
        assert w_f32.ndim == 2 and w_f32.dtype == np.float32
        rows, cols = w_f32.shape
        assert cols % 256 == 0
        w_c = np.ascontiguousarray(w_f32, dtype=np.float32)
        im_c = np.ascontiguousarray(imatrix_f32, dtype=np.float32)
        assert im_c.shape == (cols,)

        row_bytes = (cols // 256) * 66
        assert self.lib.ds4q_row_size(DS4Q_TYPE_IQ2_XXS, cols) == row_bytes
        out = np.empty((rows, row_bytes), dtype=np.uint8)

        src_ptr = w_c.ctypes.data
        dst_ptr = out.ctypes.data
        im_ptr = im_c.ctypes.data

        chunk_rows = max(16, (rows + num_workers * 4 - 1) // (num_workers * 4))
        tasks = []
        for r0 in range(0, rows, chunk_rows):
            nr = min(chunk_rows, rows - r0)
            tasks.append((r0, nr))

        def _worker(task: tuple[int, int]) -> None:
            r0, nr = task
            start_elem = r0 * cols
            self.lib.ds4q_quantize_chunk(
                DS4Q_TYPE_IQ2_XXS,
                ctypes.c_void_p(src_ptr),
                ctypes.c_void_p(dst_ptr),
                ctypes.c_int64(start_elem),
                ctypes.c_int64(nr),
                ctypes.c_int64(cols),
                ctypes.c_void_p(im_ptr),
            )

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(pool.map(_worker, tasks))

        return out


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w.float()


@torch.no_grad()
def calibrate_imatrix_from_shards(
    ckpt_path: str,
    shards_dir: str,
    max_shards: int = 32,
) -> Dict[str, np.ndarray]:
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "shard-*.safetensors")))[:max_shards]
    if not shard_files:
        print(f"[imatrix] No shards found in {shards_dir}; falling back to weight-energy proxy.")
        return {}

    device = torch.device("cpu")
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    print(f"[imatrix] Calibrating real activation E[x^2] across {len(shard_files)} shards on {device}...")

    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if k != "transfer_codes":
                tensors[k] = sf.get_tensor(k).to(device=device, dtype=torch.float32)

    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    def accum(key: str, act_2d: torch.Tensor) -> None:
        sq = (act_2d.float() ** 2).sum(dim=0).cpu()
        if key not in sums:
            sums[key] = sq
            counts[key] = int(act_2d.shape[0])
        else:
            sums[key] += sq
            counts[key] += int(act_2d.shape[0])

    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128.0))

    def apply_rope(t_4d: torch.Tensor, pos_1d: torch.Tensor) -> torch.Tensor:
        # t_4d: [N, H, 128]
        freqs = torch.outer(pos_1d.float(), inv_freq)  # [N, 64]
        cos = torch.cos(freqs).unsqueeze(1)
        sin = torch.sin(freqs).unsqueeze(1)
        x1 = t_4d[..., :64]
        x2 = t_4d[..., 64:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    total_tokens = 0
    for sp in shard_files:
        with safe_open(sp, framework="pt", device="cpu") as sf:
            hs_raw = sf.get_tensor("hidden_states")      # [S, 25600]
            last_h = sf.get_tensor("target_last_hidden_states")  # [S, 5120]
        seq_len = hs_raw.shape[0]
        if seq_len <= 2:
            continue
        # Exclude pos=0 BOS sink outlier (~2.8e5) so imatrix reflects actual decode positions
        hs = hs_raw[1:].to(device=device, dtype=torch.float32)
        x = last_h[1:].to(device=device, dtype=torch.float32)
        n_tok = hs.shape[0]
        total_tokens += n_tok
        pos = torch.arange(1, seq_len, device=device)

        accum("fc.weight", hs)
        ctx_h = rmsnorm(hs @ tensors["fc.weight"].T, tensors["hidden_norm.weight"])

        for l in range(3):
            pfx = f"layers.{l}"
            nx = rmsnorm(x, tensors[f"{pfx}.input_layernorm.weight"])
            # Combine context KV input distribution and draft QKV input distribution
            kv_in = 0.5 * (nx + ctx_h)
            accum(f"{pfx}.self_attn.q_proj.weight", nx)
            accum(f"{pfx}.self_attn.k_proj.weight", kv_in)
            accum(f"{pfx}.self_attn.v_proj.weight", kv_in)

            q = (nx @ tensors[f"{pfx}.self_attn.q_proj.weight"].T).view(n_tok, 40, 128)
            k = (ctx_h @ tensors[f"{pfx}.self_attn.k_proj.weight"].T).view(n_tok, 8, 128)
            v = (ctx_h @ tensors[f"{pfx}.self_attn.v_proj.weight"].T).view(n_tok, 8, 128)
            q = apply_rope(rmsnorm(q, tensors[f"{pfx}.self_attn.q_norm.weight"]), pos)
            k = apply_rope(rmsnorm(k, tensors[f"{pfx}.self_attn.k_norm.weight"]), pos)
            k_rep = k.repeat_interleave(5, dim=1)
            v_rep = v.repeat_interleave(5, dim=1)

            # Causal self-attention (chunked if long)
            q_t = q.transpose(0, 1)      # [40, N, 128]
            k_t = k_rep.transpose(0, 1)  # [40, N, 128]
            v_t = v_rep.transpose(0, 1)  # [40, N, 128]
            attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
            attn_flat = attn_out.transpose(0, 1).contiguous().view(n_tok, 5120)
            accum(f"{pfx}.self_attn.o_proj.weight", attn_flat)

            x = x + attn_flat @ tensors[f"{pfx}.self_attn.o_proj.weight"].T
            post_nx = rmsnorm(x, tensors[f"{pfx}.post_attention_layernorm.weight"])
            accum(f"{pfx}.mlp.gate_proj.weight", post_nx)
            accum(f"{pfx}.mlp.up_proj.weight", post_nx)

            gate = post_nx @ tensors[f"{pfx}.mlp.gate_proj.weight"].T
            up = post_nx @ tensors[f"{pfx}.mlp.up_proj.weight"].T
            glu = F.silu(gate) * up
            accum(f"{pfx}.mlp.down_proj.weight", glu)

            x = x + glu @ tensors[f"{pfx}.mlp.down_proj.weight"].T
            accum(f"injection.readouts.{l}.weight", x)

    imatrix: Dict[str, np.ndarray] = {}
    for k, sq in sums.items():
        mean_sq = (sq / max(counts[k], 1)).numpy().astype(np.float32)
        # Floor tiny channels at 1e-6 * mean so dead channels never produce NaN/zero weights
        floor = max(float(mean_sq.mean()) * 1e-6, 1e-12)
        mean_sq = np.maximum(mean_sq, floor)
        imatrix[k] = mean_sq

    print(f"[imatrix] Calibrated {len(imatrix)} matrices over {total_tokens} tokens.")
    return imatrix


def should_quantize_iq2(name: str, shape: tuple[int, ...]) -> bool:
    if name == "transfer_codes":
        return False
    if len(shape) != 2:
        return False
    rows, cols = shape
    return cols >= IQ2_MIN_COLS and (cols % 256 == 0) and rows >= 64


def write_aligned_safetensors(
    out_path: str,
    header_entries: Dict[str, dict],
    payload_chunks: list[bytes],
    alignment: int = 256,
) -> None:
    header_obj = dict(header_entries)
    header_obj["__metadata__"] = {"format": "pt", "quantization": "myds_iq2_xxs"}
    raw_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    rem = (8 + len(raw_json)) % alignment
    pad = (alignment - rem) % alignment
    padded_json = raw_json + (b" " * pad)

    with open(out_path, "wb") as fp:
        fp.write(struct.pack("<Q", len(padded_json)))
        fp.write(padded_json)
        for chunk in payload_chunks:
            fp.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize DSpine BF16 safetensors to IQ2_XXS")
    parser.add_argument("input_path", help="Input BF16/F32 .safetensors file")
    parser.add_argument("output_path", help="Output IQ2_XXS .safetensors file")
    parser.add_argument("--shards-dir", default="", help="Optional captured shards dir for real imatrix")
    parser.add_argument("--max-shards", type=int, default=32)
    parser.add_argument("--quants-lib", default="", help="Path to libds4quants.dylib")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    t0 = time.time()
    lib_path = find_quants_lib(args.quants_lib or None)
    print(f"Using libds4quants: {lib_path}")
    qlib = DS4Quants(lib_path)

    imatrix_map: Dict[str, np.ndarray] = {}
    if args.shards_dir and os.path.isdir(args.shards_dir):
        imatrix_map = calibrate_imatrix_from_shards(args.input_path, args.shards_dir, args.max_shards)

    with open(args.input_path, "rb") as fp:
        in_hdr_len = struct.unpack("<Q", fp.read(8))[0]
        in_hdr = json.loads(fp.read(in_hdr_len).decode("utf-8"))

    tensor_names = [k for k in in_hdr.keys() if k != "__metadata__"]

    header_entries: Dict[str, dict] = {}
    payload_chunks: list[bytes] = []
    cur_offset = 0
    alignment = 256

    n_quantized = 0
    in_total_bytes = os.path.getsize(args.input_path)

    with safe_open(args.input_path, framework="pt", device="cpu") as sf:
        for name in tensor_names:
            meta = in_hdr[name]
            shape = tuple(meta["shape"])
            orig_dtype = meta["dtype"]

            if should_quantize_iq2(name, shape):
                t = sf.get_tensor(name).to(dtype=torch.float32).numpy()
                rows, cols = shape
                im = imatrix_map.get(name)
                im_src = "real"
                if im is None:
                    # Column-energy fallback if no shards provided
                    col_e = np.mean(t * t, axis=0).astype(np.float32)
                    im = np.maximum(col_e, float(np.mean(col_e)) * 1e-4 + 1e-12)
                    im_src = "col_var"

                q_u8 = qlib.quantize_iq2_xxs(t, im, num_workers=args.workers)
                raw = q_u8.tobytes()
                out_dtype = "U8"
                out_shape = list(q_u8.shape)
                n_quantized += 1
                print(
                    f"  [IQ2_XXS] {name:42s} {str(list(shape)):16s} -> {str(out_shape):16s} "
                    f"({len(raw)/1048576:6.2f} MiB, imatrix={im_src})"
                )
            else:
                with open(args.input_path, "rb") as raw_fp:
                    s_off, e_off = meta["data_offsets"]
                    raw_fp.seek(8 + in_hdr_len + s_off)
                    raw = raw_fp.read(e_off - s_off)
                out_dtype = orig_dtype
                out_shape = list(shape)
                print(
                    f"  [{out_dtype:7s}] {name:42s} {str(out_shape):16s} (kept, {len(raw)/1048576:6.2f} MiB)"
                )

            start = cur_offset
            end = start + len(raw)
            header_entries[name] = {
                "dtype": out_dtype,
                "shape": out_shape,
                "data_offsets": [start, end],
            }
            rem = len(raw) % alignment
            pad = (alignment - rem) % alignment
            payload_chunks.append(raw + (b"\x00" * pad) if pad else raw)
            cur_offset = end + pad

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    write_aligned_safetensors(args.output_path, header_entries, payload_chunks, alignment=alignment)
    out_total_bytes = os.path.getsize(args.output_path)
    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.2f}s: quantized {n_quantized}/{len(tensor_names)} tensors | "
        f"{in_total_bytes/1048576:.1f} MiB -> {out_total_bytes/1048576:.1f} MiB "
        f"({in_total_bytes/max(out_total_bytes,1):.2f}x compression)"
    )


if __name__ == "__main__":
    main()
