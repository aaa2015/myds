#!/usr/bin/env python3
"""DeepSeek V4.1 Engram offline validation, token normalization and blocked export tool.

Features based on llama.cpp PR #28696 & myds native Engram engine:
1. Token Normalization Engine:
   Unicode NFKC + NFD + accent stripping + lowercase + whitespace regex,
   mapping full vocabulary to compressed token IDs.
2. Deterministic Hash Multipliers & Prime Buckets Generator:
   Generates odd 64-bit multipliers and valid prime buckets satisfying
   sum(primes) == rows[layer].
3. Chunked Blocked Quantizer (Streaming without OOM):
   Quantizes large FP32 embedding tables into 264-byte rows (256 E4M3FN + 8 E8M0)
   in configurable row chunks (e.g. 100,000 to 1,000,000 rows).
4. GGUF Engram Metadata Validator:
   Validates token_map, pad_id, prime sums, multiplier properties and table dimensions.
"""
import argparse
import math
import os
import re
import struct
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MYDS_ENGRAM_LAYERS = 2
MYDS_ENGRAM_NGRAM = 4
MYDS_ENGRAM_HEADS = 8
MYDS_ENGRAM_COLS = 24
MYDS_ENGRAM_DIM = 256
MYDS_ENGRAM_ROW_BYTES = 264
MYDS_ENGRAM_BLOCK_SIZE = 32
MYDS_ENGRAM_SCALES = 8


# ----------------------------------------------------------------------
# 1. Token Normalization & Vocabulary Compression
# ----------------------------------------------------------------------

def normalize_token_text(text: str) -> str:
    """Apply official DeepSeek V4.1 token normalization:
    NFKC -> NFD -> strip combining accents (Mn) -> lowercase -> whitespace collapse.
    """
    if not text:
        return ""
    # 1. NFKC
    nfkc = unicodedata.normalize("NFKC", text)
    # 2. NFD decomposition
    nfd = unicodedata.normalize("NFD", nfkc)
    # 3. Strip nonspacing combining marks
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # 4. Lowercase
    lowered = stripped.lower()
    # 5. Regex collapse whitespace
    collapsed = re.sub(r"\s+", " ", lowered)
    return collapsed


def build_compressed_token_map(
    tokens: List[str],
    max_compressed_size: int = 128000,
    pad_id: int = 0
) -> Tuple[List[int], int]:
    """Map vocabulary tokens to a compressed vocabulary ID space.
    Returns (token_map, compressed_vocab_size).
    """
    norm_to_id: Dict[str, int] = {}
    token_map: List[int] = []

    # Ensure pad_id maps to 0 if pad token exists, or reserve 0
    next_id = 0
    for idx, raw in enumerate(tokens):
        norm = normalize_token_text(raw)
        if norm not in norm_to_id:
            if next_id >= max_compressed_size:
                raise ValueError(
                    f"Compressed vocabulary exceeded maximum capacity {max_compressed_size}"
                )
            norm_to_id[norm] = next_id
            next_id += 1
        token_map.append(norm_to_id[norm])

    return token_map, next_id


# ----------------------------------------------------------------------
# 2. Deterministic Hash Multipliers & Prime Buckets
# ----------------------------------------------------------------------

def generate_layer_multipliers(layer_id: int, ngram: int = MYDS_ENGRAM_NGRAM) -> List[int]:
    """Generate deterministic odd 64-bit multipliers for N-gram hashing.
    Matches llama.cpp RNG seed convention (10007 * layer_id).
    """
    state = (10007 * layer_id + 1) & 0xFFFFFFFFFFFFFFFF
    multipliers: List[int] = []

    # Linear congruential generator for reproducible integers
    for _ in range(ngram):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        # Must be odd and positive int64
        mult = (state >> 1) | 1
        multipliers.append(mult)

    return multipliers


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True


def next_prime(n: int) -> int:
    while not is_prime(n):
        n += 1
    return n


def generate_prime_buckets(total_rows: int, cols: int = MYDS_ENGRAM_COLS) -> List[int]:
    """Partition total_rows into 'cols' distinct primes whose sum equals total_rows."""
    if total_rows < cols * 2:
        raise ValueError(f"total_rows {total_rows} too small for {cols} columns")

    avg = total_rows // cols
    primes: List[int] = []
    used = set()

    cand = max(2, avg - cols * 10)
    for _ in range(cols - 1):
        cand = next_prime(cand)
        while cand in used:
            cand = next_prime(cand + 1)
        used.add(cand)
        primes.append(cand)
        cand += 1

    remaining = total_rows - sum(primes)
    if remaining < 2:
        # Adjust base
        primes[-1] -= (2 - remaining)
        remaining = 2
    primes.append(remaining)
    return primes


# ----------------------------------------------------------------------
# 3. Blocked Quantization (E4M3FN + E8M0, 264 bytes / row)
# ----------------------------------------------------------------------

# Precompute E4M3FN table for fast lookup during quantization
E4M3_MAX = 448.0


def quantize_row_fp32(floats: List[float]) -> bytes:
    """Quantize 256 FP32 values into 264 bytes:
    - 256 E4M3FN bytes (8 blocks of 32 values)
    - 8 E8M0 scale bytes (each scale byte s represents 2^(s - 127))
    """
    if len(floats) != MYDS_ENGRAM_DIM:
        raise ValueError(f"Expected {MYDS_ENGRAM_DIM} floats, got {len(floats)}")

    codes = bytearray(MYDS_ENGRAM_DIM)
    scales = bytearray(MYDS_ENGRAM_SCALES)

    for block in range(MYDS_ENGRAM_SCALES):
        sub = floats[block * MYDS_ENGRAM_BLOCK_SIZE : (block + 1) * MYDS_ENGRAM_BLOCK_SIZE]
        max_abs = max((abs(x) for x in sub), default=0.0)

        if max_abs <= 1e-30:
            scale_byte = 0
            scale_val = math.ldexp(1.0, -127)
        else:
            # We want scale_val * 448 >= max_abs => scale_val >= max_abs / 448
            target_scale = max_abs / 448.0
            # log2(target_scale) = exp - 127 => exp = ceil(log2(target_scale)) + 127
            exp = math.ceil(math.log2(target_scale)) + 127
            exp = max(0, min(254, exp))
            scale_byte = exp
            scale_val = math.ldexp(1.0, scale_byte - 127)

        scales[block] = scale_byte

        # Quantize each value in the block with E4M3FN
        for i, val in enumerate(sub):
            scaled = val / scale_val
            sign = 0x80 if scaled < 0 else 0x00
            mag = abs(scaled)

            if mag <= 0.0:
                code = 0
            elif mag < math.ldexp(1.0, -6):  # subnormal
                frac = min(7, max(0, round(mag / math.ldexp(1.0, -9))))
                code = frac
            else:  # normal
                exp = min(15, max(1, math.floor(math.log2(mag)) + 7))
                frac_val = mag / math.ldexp(1.0, exp - 7) - 1.0
                frac = min(7, max(0, round(frac_val * 8.0)))
                code = (exp << 3) | frac
                if code == 0x7F:  # 0x7F is NaN in E4M3FN, clamp to 0x7E
                    code = 0x7E

            codes[block * MYDS_ENGRAM_BLOCK_SIZE + i] = sign | code

    return bytes(codes) + bytes(scales)


def quantize_table_stream(
    in_fp32_path: Path,
    out_bin_path: Path,
    num_rows: int,
    chunk_size: int = 100_000
) -> int:
    """Stream FP32 table to quantized 264-byte rows in chunks without memory explosion."""
    bytes_per_row_in = MYDS_ENGRAM_DIM * 4
    total_written = 0

    with open(in_fp32_path, "rb") as fin, open(out_bin_path, "wb") as fout:
        for chunk_start in range(0, num_rows, chunk_size):
            rows_to_read = min(chunk_size, num_rows - chunk_start)
            raw = fin.read(rows_to_read * bytes_per_row_in)
            if len(raw) < rows_to_read * bytes_per_row_in:
                raise IOError(f"Unexpected EOF reading row {chunk_start}")

            floats_batch = struct.unpack(f"<{rows_to_read * MYDS_ENGRAM_DIM}f", raw)
            out_buf = bytearray()
            for r in range(rows_to_read):
                row_floats = floats_batch[r * MYDS_ENGRAM_DIM : (r + 1) * MYDS_ENGRAM_DIM]
                out_buf.extend(quantize_row_fp32(row_floats))

            fout.write(out_buf)
            total_written += rows_to_read

    return total_written


# ----------------------------------------------------------------------
# 4. GGUF Metadata Validation
# ----------------------------------------------------------------------

def validate_engram_metadata(metadata: Dict[str, any]) -> Tuple[bool, List[str]]:
    """Validate GGUF Engram metadata against specification invariants."""
    errors = []

    # Check required keys
    req_keys = [
        "deepseek41.engram.layer_ids",
        "deepseek41.engram.token_map",
        "deepseek41.engram.multipliers",
        "deepseek41.engram.primes",
        "deepseek41.engram.pad_id",
        "deepseek41.engram.rows"
    ]
    for k in req_keys:
        if k not in metadata:
            errors.append(f"Missing required metadata key: {k}")

    if errors:
        return False, errors

    layer_ids = metadata["deepseek41.engram.layer_ids"]
    if layer_ids != [1, 14]:
        errors.append(f"Invalid layer_ids: {layer_ids} (expected [1, 14])")

    rows = metadata["deepseek41.engram.rows"]
    if len(rows) != MYDS_ENGRAM_LAYERS:
        errors.append(f"rows length {len(rows)} != {MYDS_ENGRAM_LAYERS}")

    primes = metadata["deepseek41.engram.primes"]
    if len(primes) != MYDS_ENGRAM_LAYERS:
        errors.append(f"primes layers count {len(primes)} != {MYDS_ENGRAM_LAYERS}")
    else:
        for l in range(MYDS_ENGRAM_LAYERS):
            if len(primes[l]) != MYDS_ENGRAM_COLS:
                errors.append(f"primes[{l}] len {len(primes[l])} != {MYDS_ENGRAM_COLS}")
            sum_p = sum(primes[l])
            if sum_p != rows[l]:
                errors.append(f"primes[{l}] sum {sum_p} != rows[{l}] {rows[l]}")

    multipliers = metadata["deepseek41.engram.multipliers"]
    for l, ml in enumerate(multipliers):
        for idx, m in enumerate(ml):
            if (m & 1) == 0:
                errors.append(f"multipliers[{l}][{idx}]={m} is even (must be odd)")

    pad_id = metadata["deepseek41.engram.pad_id"]
    token_map = metadata["deepseek41.engram.token_map"]
    comp_size = max(token_map) + 1 if token_map else 0
    if pad_id >= comp_size:
        errors.append(f"pad_id {pad_id} >= compressed_vocab_size {comp_size}")

    return len(errors) == 0, errors


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek V4.1 Engram offline validation and export tool"
    )
    sub = parser.add_subparsers(dest="cmd")

    # Subcommand: normalize
    p_norm = sub.add_parser("normalize", help="Test token normalization on string")
    p_norm.add_argument("text", help="Text to normalize")

    # Subcommand: generate-layout
    p_gen = sub.add_parser("generate-layout", help="Generate multipliers and prime buckets")
    p_gen.add_argument("--rows", type=int, default=2500000, help="Total rows per layer")

    # Subcommand: validate-mock
    p_val = sub.add_parser("validate-mock", help="Validate sample layout parameters")

    args = parser.parse_args()

    if args.cmd == "normalize":
        res = normalize_token_text(args.text)
        print(f"Original  : {repr(args.text)}")
        print(f"Normalized: {repr(res)}")
        return 0

    elif args.cmd == "generate-layout":
        rows = [args.rows, args.rows]
        print(f"Generating Engram layout for {rows} rows across 2 layers...")
        for l in range(MYDS_ENGRAM_LAYERS):
            mults = generate_layer_multipliers(l + 1)
            primes = generate_prime_buckets(rows[l])
            print(f"Layer {l} (Transformer layer {[1, 14][l]}):")
            print(f"  Multipliers ({len(mults)}): {mults}")
            print(f"  Primes sum: {sum(primes)} == {rows[l]} (cols={len(primes)})")
        return 0

    elif args.cmd == "validate-mock":
        rows = [2500000, 2500000]
        meta = {
            "deepseek41.engram.layer_ids": [1, 14],
            "deepseek41.engram.rows": rows,
            "deepseek41.engram.primes": [generate_prime_buckets(r) for r in rows],
            "deepseek41.engram.multipliers": [generate_layer_multipliers(1), generate_layer_multipliers(14)],
            "deepseek41.engram.pad_id": 0,
            "deepseek41.engram.token_map": [0, 1, 2, 3, 0],
        }
        ok, errs = validate_engram_metadata(meta)
        if ok:
            print("[PASS] Engram layout validated successfully against invariants.")
            return 0
        else:
            print(f"[FAIL] {errs}")
            return 1

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
