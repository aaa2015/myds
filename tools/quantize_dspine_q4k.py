#!/usr/bin/env python3
"""Quantize a BF16/F32 DSpine .safetensors checkpoint to native GGML Q4_K.

Heavy 2D projection matrices with cols % 256 == 0 and cols >= 5120:
  - fc.weight                       [5120, 25600]
  - layers.{0,1,2}.{q,k,v,o}_proj   [5120|1024, 5120]
  - layers.{0,1,2}.{gate,up}_proj   [15360, 5120]
  - layers.{0,1,2}.down_proj        [5120, 15360]
  - injection.readouts.{0,1,2}      [512, 5120]
are quantized into GGML block_q4_K (144 bytes per 256 elements) stored as U8
tensors of shape [rows, (cols // 256) * 144] in .safetensors.

Small sensitive tensors (RMSNorms, Code Bridge gates/message/writes/bias, and
transfer_codes [129280, 512]) are preserved verbatim in BF16.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from typing import Dict, Tuple

import numpy as np
import torch
from safetensors import safe_open


Q4K_MIN_COLS = 5120


def quantize_q4_k_chunk(w_f32: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Quantize [R, C] float32 tensor (C % 256 == 0) into [R, (C // 256) * 144] uint8."""
    rows, cols = w_f32.shape
    assert cols % 256 == 0, f"cols={cols} must be divisible by 256"
    n_sb = cols // 256

    # Shape: [R, n_sb, 8, 32]
    x = w_f32.reshape(rows, n_sb, 8, 32)

    # Initial non-negative min/scale: x ≈ s * q - m, q in [0..15], s >= 0, m >= 0
    x_min = x.amin(dim=-1)
    x_max = x.amax(dim=-1)
    m = (-x_min).clamp_min(0.0)
    s = ((x_max + m) / 15.0).clamp_min(0.0)

    # Two rounds of weighted least-squares refinement (matching ggml make_qkx2_quants spirit)
    sigma2 = (x * x).mean(dim=-1, keepdim=True) * 0.1 + 1e-12
    weights = sigma2 + x * x  # [R, n_sb, 8, 32]
    sw = weights.sum(dim=-1)
    sx = (weights * x).sum(dim=-1)

    for _ in range(2):
        inv_s = torch.where(s > 0.0, 1.0 / s.clamp_min(1e-30), torch.zeros_like(s))
        q = torch.round((x + m.unsqueeze(-1)) * inv_s.unsqueeze(-1)).clamp_(0.0, 15.0)
        sq = (weights * q).sum(dim=-1)
        sqq = (weights * q * q).sum(dim=-1)
        sqx = (weights * q * x).sum(dim=-1)
        det = sw * sqq - sq * sq

        valid = det > 1e-20
        s_ls = torch.where(valid, (sw * sqx - sq * sx) / det.clamp_min(1e-20), s)
        m_ls = torch.where(valid, (sq * sqx - sqq * sx) / det.clamp_min(1e-20), m)

        # If m_ls < 0, clamp m=0 and solve 1-parameter s = sqx / sqq
        neg_m = m_ls < 0.0
        s_nom = torch.where(sqq > 1e-20, sqx / sqq.clamp_min(1e-20), s)
        s = torch.where(neg_m, s_nom, s_ls).clamp_min(0.0)
        m = torch.where(neg_m, torch.zeros_like(m_ls), m_ls).clamp_min(0.0)

    # Super-block FP16 d and dmin
    max_s = s.amax(dim=-1)  # [R, n_sb]
    max_m = m.amax(dim=-1)  # [R, n_sb]
    d_fp16 = (max_s / 63.0).to(torch.float16)
    dmin_fp16 = (max_m / 63.0).to(torch.float16)

    d_f32 = d_fp16.to(torch.float32)
    dmin_f32 = dmin_fp16.to(torch.float32)

    inv_d = torch.where(d_f32 > 0.0, 1.0 / d_f32.clamp_min(1e-30), torch.zeros_like(d_f32))
    inv_dmin = torch.where(dmin_f32 > 0.0, 1.0 / dmin_f32.clamp_min(1e-30), torch.zeros_like(dmin_f32))

    sc = torch.round(s * inv_d.unsqueeze(-1)).clamp_(0.0, 63.0).to(torch.uint8)    # [R, n_sb, 8]
    im = torch.round(m * inv_dmin.unsqueeze(-1)).clamp_(0.0, 63.0).to(torch.uint8)  # [R, n_sb, 8]

    # Final quantized nibbles using exact effective FP16-derived scales
    dl = d_f32.unsqueeze(-1) * sc.to(torch.float32)     # [R, n_sb, 8]
    ml = dmin_f32.unsqueeze(-1) * im.to(torch.float32)  # [R, n_sb, 8]
    inv_dl = torch.where(dl > 0.0, 1.0 / dl.clamp_min(1e-30), torch.zeros_like(dl))
    q_u8 = torch.round((x + ml.unsqueeze(-1)) * inv_dl.unsqueeze(-1)).clamp_(0.0, 15.0).to(torch.uint8)

    # Compute dequantized error metrics
    x_hat = dl.unsqueeze(-1) * q_u8.to(torch.float32) - ml.unsqueeze(-1)
    diff_norm2 = float(((x_hat - x) ** 2).sum().item())
    orig_norm2 = float((x ** 2).sum().item())

    # Pack into [R, n_sb, 144] uint8
    out = torch.empty((rows, n_sb, 144), dtype=torch.uint8, device=w_f32.device)
    out[:, :, 0:2] = d_fp16.contiguous().unsqueeze(-1).view(torch.uint8)
    out[:, :, 2:4] = dmin_fp16.contiguous().unsqueeze(-1).view(torch.uint8)

    # Pack 12-byte scales:
    # k = 0..3:
    #   scales[k]     = (sc[k] & 0x3f) | ((sc[k+4] >> 4) << 6)
    #   scales[k + 4] = (im[k] & 0x3f) | ((im[k+4] >> 4) << 6)
    #   scales[k + 8] = (sc[k+4] & 0x0f) | ((im[k+4] & 0x0f) << 4)
    sc0_3 = sc[:, :, 0:4]
    sc4_7 = sc[:, :, 4:8]
    im0_3 = im[:, :, 0:4]
    im4_7 = im[:, :, 4:8]

    out[:, :, 4:8] = (sc0_3 & 0x3F) | ((sc4_7 >> 4) << 6)
    out[:, :, 8:12] = (im0_3 & 0x3F) | ((im4_7 >> 4) << 6)
    out[:, :, 12:16] = (sc4_7 & 0x0F) | ((im4_7 & 0x0F) << 4)

    # Pack 128-byte qs:
    # b64 = 0..3: low nibble = q[2*b64], high nibble = q[2*b64 + 1]
    q_even = q_u8[:, :, 0::2, :]  # [R, n_sb, 4, 32]
    q_odd = q_u8[:, :, 1::2, :]   # [R, n_sb, 4, 32]
    qs = (q_even & 0x0F) | ((q_odd & 0x0F) << 4)
    out[:, :, 16:144] = qs.reshape(rows, n_sb, 128)

    return out.reshape(rows, n_sb * 144).cpu(), diff_norm2, orig_norm2


def should_quantize_q4k(name: str, shape: Tuple[int, ...]) -> bool:
    if name == "transfer_codes":
        return False
    if len(shape) != 2:
        return False
    rows, cols = shape
    return cols >= Q4K_MIN_COLS and (cols % 256 == 0) and rows >= 64


def write_aligned_safetensors(
    out_path: str,
    header_entries: Dict[str, dict],
    payload_chunks: list[bytes],
    alignment: int = 256,
) -> None:
    header_obj = dict(header_entries)
    header_obj["__metadata__"] = {"format": "pt", "quantization": "myds_q4_k"}
    raw_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    # Ensure (8 + padded_json_len) % alignment == 0 so every tensor starts 256-byte aligned
    rem = (8 + len(raw_json)) % alignment
    pad = (alignment - rem) % alignment
    padded_json = raw_json + (b" " * pad)

    with open(out_path, "wb") as fp:
        fp.write(struct.pack("<Q", len(padded_json)))
        fp.write(padded_json)
        for chunk in payload_chunks:
            fp.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize DSpine BF16 safetensors to Q4_K")
    parser.add_argument("input_path", help="Input BF16/F32 .safetensors file")
    parser.add_argument("output_path", help="Output Q4_K .safetensors file")
    parser.add_argument("--row-batch", type=int, default=1024, help="Rows per quantization chunk")
    parser.add_argument("--device", default="auto", help="Torch device (auto/mps/cpu)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = time.time()
    header_entries: Dict[str, dict] = {}
    payload_chunks: list[bytes] = []
    offset = 0
    in_bytes_total = 0
    out_bytes_total = 0
    q4k_count = 0

    print(f"[quantize_dspine_q4k] input={args.input_path} -> output={args.output_path} (device={device})")

    with safe_open(args.input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for name in keys:
            t = f.get_tensor(name)
            shape = tuple(t.shape)
            in_nbytes = t.numel() * t.element_size()
            in_bytes_total += in_nbytes

            if should_quantize_q4k(name, shape):
                rows, cols = shape
                q_parts = []
                diff_sum = 0.0
                orig_sum = 0.0
                for r0 in range(0, rows, args.row_batch):
                    r1 = min(rows, r0 + args.row_batch)
                    sub = t[r0:r1].to(device=device, dtype=torch.float32)
                    q_sub, dn2, on2 = quantize_q4_k_chunk(sub)
                    q_parts.append(q_sub)
                    diff_sum += dn2
                    orig_sum += on2
                q_tensor = torch.cat(q_parts, dim=0).contiguous()
                raw_bytes = q_tensor.numpy().tobytes()
                rel_err = (diff_sum / max(orig_sum, 1e-30)) ** 0.5
                dtype_str = "U8"
                out_shape = list(q_tensor.shape)
                q4k_count += 1
                print(
                    f"  [Q4_K] {name:34s} {str(list(shape)):16s} "
                    f"{in_nbytes / (1024*1024):6.2f} MiB -> {len(raw_bytes) / (1024*1024):6.2f} MiB "
                    f"(rel_rmse={rel_err:.4%})"
                )
            else:
                t_bf16 = t.to(torch.bfloat16).contiguous()
                raw_bytes = t_bf16.view(torch.uint16).numpy().tobytes()
                dtype_str = "BF16"
                out_shape = list(t_bf16.shape)
                # Pad individual small tensor tail with zeros only in file offset if needed?
                # No: data_offsets must match exact tensor byte size! All BF16 tensors here
                # have >=128 elements = >=256 bytes (multiple of 256!).

            start = offset
            end = offset + len(raw_bytes)
            header_entries[name] = {
                "dtype": dtype_str,
                "shape": out_shape,
                "data_offsets": [start, end],
            }
            payload_chunks.append(raw_bytes)
            offset = end
            out_bytes_total += len(raw_bytes)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    write_aligned_safetensors(args.output_path, header_entries, payload_chunks, alignment=256)
    dt = time.time() - t0
    file_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(
        f"[quantize_dspine_q4k] Done in {dt:.2f}s: quantized {q4k_count} matrices, "
        f"{in_bytes_total / (1024*1024):.1f} MiB -> {file_mb:.1f} MiB "
        f"({in_bytes_total / max(out_bytes_total, 1):.2f}x compression)"
    )


if __name__ == "__main__":
    main()
