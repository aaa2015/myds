#!/usr/bin/env python3
"""Convert DeepSeek V4.1 Flash Dense layers from Q8_0 to Q4_K.

This tool slims the resident Dense weights (MLA Attention projections, Shared Experts,
and LM Head) from Q8_0 (34B/32W = 8.5 b/w) down to Q4_K (144B/256W = 4.5 b/w).
All other tensors (routed MoE experts, norms, embeddings, Engram tables) are preserved
bit-for-bit with zero modification.

The original model file is strictly preserved without modification.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import os
import shutil
import struct
import sys
import time
from typing import List, Tuple

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 16384

# GGUF Metadata Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Quantization Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q4_K = 12
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_I8 = 24
GGML_TYPE_BF16 = 30

TYPE_LAYOUT = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_Q8_0: (32, 34),
    GGML_TYPE_Q2_K: (256, 84),
    GGML_TYPE_Q4_K: (256, 144),
    GGML_TYPE_IQ2_XXS: (256, 66),
    GGML_TYPE_I8: (1, 1),
    GGML_TYPE_BF16: (1, 2),
}

STREAM_BUFFER_SIZE = 32 * 1024 * 1024  # 32 MiB streaming buffer


def align(value: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_nbytes(ttype: int, dims: tuple[int, ...]) -> int:
    if ttype not in TYPE_LAYOUT:
        raise ValueError(f"Unsupported GGML type {ttype} for byte calculation")
    block, block_bytes = TYPE_LAYOUT[ttype]
    w = dims[0]
    if w % block != 0:
        raise ValueError(f"Dimension 0 ({w}) must be divisible by block size ({block})")
    nrows = 1
    for d in dims[1:]:
        nrows *= d
    return (w // block) * block_bytes * nrows


class TensorInfo:
    def __init__(self, idx: int, name: str, ndims: int, dims: tuple[int, ...],
                 old_type: int, old_offset: int, is_target: bool):
        self.idx = idx
        self.name = name
        self.ndims = ndims
        self.dims = dims
        self.old_type = old_type
        self.old_offset = old_offset
        self.is_target = is_target
        self.new_type = GGML_TYPE_Q4_K if is_target else old_type
        self.old_size = tensor_nbytes(self.old_type, self.dims)
        self.new_size = tensor_nbytes(self.new_type, self.dims)
        self.new_offset = 0


def is_dense_target(name: str, ttype: int) -> bool:
    if ttype != GGML_TYPE_Q8_0:
        return False
    if name == "output.weight":
        return True
    if name.startswith("blk."):
        if name.endswith((
            "_shexp.weight",
            ".attn_q_a.weight",
            ".attn_q_b.weight",
            ".attn_kv.weight",
            ".attn_output_a.weight",
            ".attn_output_b.weight",
        )):
            return True
    return False


def load_quants_library(custom_path: str | None = None) -> ctypes.CDLL:
    paths = []
    if custom_path:
        paths.append(custom_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = "dylib" if sys.platform == "darwin" else "so"
    paths.extend([
        os.path.join(base_dir, f"../ds4/gguf-tools/libds4quants.{suffix}"),
        os.path.join(base_dir, f"../../ds4/gguf-tools/libds4quants.{suffix}"),
        os.path.join(base_dir, f"libds4quants.{suffix}"),
        f"~/Projects/ds4/gguf-tools/libds4quants.{suffix}",
    ])
    for p in paths:
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            lib.ds4q_convert_q8_0_to_q4_k.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
            ]
            lib.ds4q_convert_q8_0_to_q4_k.restype = ctypes.c_size_t
            return lib
    raise FileNotFoundError(f"Could not locate libds4quants.{suffix}. Tried: {paths}")


def read_str(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def skip_val(f, vtype: int):
    scalar_sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype == GGUF_TYPE_STRING:
        length = struct.unpack("<Q", f.read(8))[0]
        f.seek(length, os.SEEK_CUR)
    elif vtype == GGUF_TYPE_ARRAY:
        etype, count = struct.unpack("<IQ", f.read(12))
        if etype == GGUF_TYPE_STRING:
            for _ in range(count):
                length = struct.unpack("<Q", f.read(8))[0]
                f.seek(length, os.SEEK_CUR)
        else:
            f.seek(scalar_sz[etype] * count, os.SEEK_CUR)
    else:
        f.seek(scalar_sz[vtype], os.SEEK_CUR)


def quantize_rows_chunk(lib: ctypes.CDLL, src_bytes: bytes, nrows: int, ncols: int) -> bytes:
    dst_size = (ncols // 256) * 144 * nrows
    dst_buf = bytearray(dst_size)
    src_ptr = ctypes.cast(src_bytes, ctypes.c_void_p).value
    dst_ptr = (ctypes.c_char * dst_size).from_buffer(dst_buf)
    written = lib.ds4q_convert_q8_0_to_q4_k(src_ptr, dst_ptr, nrows, ncols)
    if written != dst_size:
        raise RuntimeError(f"Quantizer wrote {written} bytes, expected {dst_size}")
    return bytes(dst_buf)


def parallel_convert_q8_to_q4(lib: ctypes.CDLL, src_data: bytes, ncols: int, nrows: int,
                              num_threads: int, pool: concurrent.futures.ThreadPoolExecutor) -> bytes:
    if nrows == 1 or num_threads == 1:
        return quantize_rows_chunk(lib, src_data, nrows, ncols)

    q8_row_size = (ncols // 32) * 34
    q4_row_size = (ncols // 256) * 144
    dst_total_size = q4_row_size * nrows
    dst_buf = bytearray(dst_total_size)

    # Chunk rows across threads
    chunk_rows = max(1, (nrows + num_threads - 1) // num_threads)
    futures = []

    def task(start_r: int, count_r: int):
        src_offset = start_r * q8_row_size
        src_slice = src_data[src_offset:src_offset + count_r * q8_row_size]
        converted = quantize_rows_chunk(lib, src_slice, count_r, ncols)
        dst_offset = start_r * q4_row_size
        dst_buf[dst_offset:dst_offset + len(converted)] = converted

    for r_start in range(0, nrows, chunk_rows):
        r_count = min(chunk_rows, nrows - r_start)
        futures.append(pool.submit(task, r_start, r_count))

    for f in futures:
        f.result()

    return bytes(dst_buf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="Input GGUF path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output GGUF path")
    parser.add_argument("--quants-lib", dest="quants_lib", default=None, help="Path to libds4quants")
    parser.add_argument("--threads", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and summary without modifying disk")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted partial run")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    in_path = os.path.abspath(args.in_path)
    out_path = os.path.abspath(args.out_path)

    if not os.path.isfile(in_path):
        sys.exit(f"Error: input file {in_path} does not exist")
    if in_path == out_path:
        sys.exit("Error: --in and --out must be different files to preserve the original baseline!")

    if os.path.exists(out_path) and not args.force:
        sys.exit(f"Error: output file {out_path} already exists. Use --force or specify a different path.")

    # 1. Parse header from input file
    with open(in_path, "rb") as fin:
        magic = fin.read(4)
        if magic != GGUF_MAGIC:
            sys.exit(f"Error: {in_path} is not a valid GGUF file (magic={magic})")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fin.read(20))
        if version not in (2, 3):
            sys.exit(f"Error: unsupported GGUF version {version}")

        # Read all metadata records verbatim to preserve
        kv_records = []
        alignment = GGUF_DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            kv_start = fin.tell()
            k = read_str(fin)
            vt = struct.unpack("<I", fin.read(4))[0]
            if k == "general.alignment":
                if vt == GGUF_TYPE_UINT32:
                    alignment = struct.unpack("<I", fin.read(4))[0]
                elif vt == GGUF_TYPE_UINT64:
                    alignment = struct.unpack("<Q", fin.read(8))[0]
            else:
                skip_val(fin, vt)
            kv_end = fin.tell()
            fin.seek(kv_start)
            kv_records.append(fin.read(kv_end - kv_start))

        # Read tensor descriptors
        tensors: List[TensorInfo] = []
        target_count = 0
        target_q8_bytes = 0
        target_q4_bytes = 0

        for idx in range(n_tensors):
            name = read_str(fin)
            ndims = struct.unpack("<I", fin.read(4))[0]
            dims = struct.unpack(f"<{ndims}Q", fin.read(8 * ndims))
            ttype, offset = struct.unpack("<IQ", fin.read(12))

            is_tgt = is_dense_target(name, ttype)
            t_info = TensorInfo(idx, name, ndims, dims, ttype, offset, is_tgt)
            tensors.append(t_info)

            if is_tgt:
                target_count += 1
                target_q8_bytes += t_info.old_size
                target_q4_bytes += t_info.new_size

        in_header_end = fin.tell()
        in_data_start = align(in_header_end, alignment)

    # 2. Recalculate tensor offsets
    curr_offset = 0
    for t in tensors:
        t.new_offset = curr_offset
        curr_offset = align(curr_offset + t.new_size, alignment)

    out_data_bytes = curr_offset
    total_out_size = in_data_start + out_data_bytes
    in_file_size = os.path.getsize(in_path)
    savings_bytes = target_q8_bytes - target_q4_bytes

    print("==================================================================")
    print("      DeepSeek V4.1 Flash Dense Slimming Plan (Q8_0 -> Q4_K)      ")
    print("==================================================================")
    print(f"Input file:           {in_path}")
    print(f"Output file:          {out_path}")
    print(f"GGUF Version:         {version}, Alignment: {alignment} bytes")
    print(f"Total Tensors:        {n_tensors}, Metadata KV pairs: {n_kv}")
    print(f"Target Dense Tensors: {target_count} (MLA Q/K/V/Out, Shared FFN, LM Head)")
    print(f"Preserved Tensors:    {n_tensors - target_count} (MoE Routed Experts, Norms, Engram, Embed)")
    print("------------------------------------------------------------------")
    print(f"Original Dense Q8_0:  {target_q8_bytes / (1 << 30):.3f} GiB ({target_q8_bytes / 1e9:.3f} GB)")
    print(f"Slimmed Dense Q4_K:   {target_q4_bytes / (1 << 30):.3f} GiB ({target_q4_bytes / 1e9:.3f} GB)")
    print(f"Memory Saved:         {savings_bytes / (1 << 30):.3f} GiB ({savings_bytes / 1e9:.3f} GB) (~{savings_bytes / target_q8_bytes * 100:.1f}% reduction)")
    print(f"Total File Size:      {in_file_size / (1 << 30):.2f} GiB -> {total_out_size / (1 << 30):.2f} GiB")
    print("==================================================================")

    if args.dry_run:
        print("[DRY-RUN] Verification complete. No files modified.")
        return

    # Check disk space
    out_dir = os.path.dirname(out_path) or "."
    free_space = shutil.disk_usage(out_dir).free
    needed_space = total_out_size + (10 << 30)  # 10 GiB safety reserve
    if free_space < needed_space:
        sys.exit(f"Error: Insufficient disk space in {out_dir}. Available: {free_space / (1 << 30):.2f} GiB, Required: {needed_space / (1 << 30):.2f} GiB")

    # Load quantizer library
    quants_lib = load_quants_library(args.quants_lib)

    # 3. Construct new header
    # GGUF Magic + version + n_tensors + n_kv
    out_header = GGUF_MAGIC + struct.pack("<IQQ", version, n_tensors, n_kv)
    # KV records
    out_header += b"".join(kv_records)
    # Tensor descriptors
    for t in tensors:
        name_bytes = t.name.encode("utf-8")
        out_header += struct.pack("<Q", len(name_bytes)) + name_bytes
        out_header += struct.pack("<I", t.ndims)
        out_header += struct.pack(f"<{t.ndims}Q", *t.dims)
        out_header += struct.pack("<IQ", t.new_type, t.new_offset)

    if len(out_header) != in_header_end:
        raise RuntimeError(f"Header length mismatch: generated {len(out_header)} != input {in_header_end}")

    # Pad header to data_start
    out_header += bytes(in_data_start - len(out_header))

    partial_path = out_path + ".partial"
    journal_path = out_path + ".journal.json"
    start_idx = 0

    if args.resume and os.path.exists(partial_path) and os.path.exists(journal_path):
        with open(journal_path, "r") as jf:
            journal_data = json.load(jf)
            start_idx = journal_data.get("completed_tensor_idx", 0)
        print(f"[RESUME] Resuming from tensor index {start_idx}/{n_tensors}...")
    else:
        with open(partial_path, "wb") as fout:
            fout.write(out_header)
            fout.flush()
            os.fsync(fout.fileno())

    # 4. Stream and convert tensors
    start_time = time.monotonic()
    last_print = start_time
    total_bytes_written = 0

    with open(in_path, "rb") as fin, open(partial_path, "r+b") as fout, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:

        if start_idx > 0:
            expected_pos = in_data_start + tensors[start_idx].new_offset
            fout.seek(expected_pos)
        else:
            fout.seek(in_data_start)

        for i in range(start_idx, n_tensors):
            t = tensors[i]
            expected_fout_pos = in_data_start + t.new_offset
            if fout.tell() != expected_fout_pos:
                raise RuntimeError(f"Output seek error for {t.name}: at {fout.tell()}, expected {expected_fout_pos}")

            fin.seek(in_data_start + t.old_offset)

            if t.is_target:
                # Dense Q8_0 -> Q4_K conversion
                q8_data = fin.read(t.old_size)
                if len(q8_data) != t.old_size:
                    raise RuntimeError(f"Short read for tensor {t.name}")

                ncols = t.dims[0]
                nrows = 1
                for d in t.dims[1:]:
                    nrows *= d

                q4_data = parallel_convert_q8_to_q4(quants_lib, q8_data, ncols, nrows, args.threads, pool)
                fout.write(q4_data)
                total_bytes_written += len(q4_data)
            else:
                # Stream copy verbatim in chunks
                rem = t.old_size
                while rem > 0:
                    chunk = min(rem, STREAM_BUFFER_SIZE)
                    buf = fin.read(chunk)
                    if len(buf) != chunk:
                        raise RuntimeError(f"Short read while copying {t.name}")
                    fout.write(buf)
                    rem -= chunk
                    total_bytes_written += chunk

            # Write alignment padding
            pad_len = align(t.new_size, alignment) - t.new_size
            if pad_len > 0:
                fout.write(bytes(pad_len))
                total_bytes_written += pad_len

            # Periodic log and journal update
            now = time.monotonic()
            if now - last_print >= 2.0 or i == n_tensors - 1 or t.is_target:
                pct = (i + 1) / n_tensors * 100
                speed = total_bytes_written / (now - start_time) / (1 << 20)
                elapsed = now - start_time
                est_total = elapsed / ((i + 1) / n_tensors)
                eta = max(0, est_total - elapsed)
                tag = "[Q4_K]" if t.is_target else "[COPY]"
                print(f"[{i+1:4d}/{n_tensors}] {pct:5.1f}% {tag} {t.name:<40s} | {speed:6.1f} MB/s | ETA: {eta:.0f}s", flush=True)
                last_print = now

                # Write journal
                with open(journal_path, "w") as jf:
                    json.dump({"completed_tensor_idx": i + 1, "tensor_name": t.name}, jf)

        fout.flush()
        os.fsync(fout.fileno())

    # Atomically rename to destination
    os.rename(partial_path, out_path)
    if os.path.exists(journal_path):
        os.unlink(journal_path)

    total_time = time.monotonic() - start_time
    avg_speed = total_out_size / total_time / (1 << 20)
    print("==================================================================")
    print("                 SLIMMING COMPLETED SUCCESSFULLY!                 ")
    print("==================================================================")
    print(f"Source file:   {in_path} (PRESERVED)")
    print(f"Output file:   {out_path} ({os.path.getsize(out_path) / (1 << 30):.2f} GiB)")
    print(f"Saved memory:  {savings_bytes / (1 << 30):.3f} GiB ({savings_bytes / 1e9:.3f} GB)")
    print(f"Total time:    {total_time:.1f}s (Average throughput: {avg_speed:.1f} MB/s)")
    print("==================================================================")


if __name__ == "__main__":
    main()
