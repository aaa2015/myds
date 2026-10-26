#!/usr/bin/env python3
"""Read-only CPU qualification of the upstream V4.1 narrow weight formats.

  python3 tools/audit_v41_narrow_weights.py --model /abs/model.gguf --output /abs/audit.json

Only Python's standard library and a C11 compiler with _Float16 are required.
The companion C worker is compiled into a fresh retained artifact directory; no Metal
or inference code is linked. The audit reads GGUF headers and the 50 target
payloads, never the full Engram embedding tables. SHA identities deliberately
cover that scope, not the entire model. Exit 0: all 50 tensors qualify; exit 1:
valid audit with rejected tensors; exit 2: malformed input/tool/identity error.

E4M3 failure streams use explicit placeholders plus a per-block validity mask;
their hashes are diagnostic only. No converted model is emitted or modified.
"""
from __future__ import annotations

import argparse
import bisect
import ctypes
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time

INDEX_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
ENGRAM_LAYERS = (1, 14)
UPSTREAM = {
    "7d9903b2ed6bd74c14aed06a89a591a14d613e9e": "c64a447efd6b41f815ba48be37db742da43cd0dfa1a37509c275ee427ef9db6a",
    "55b1e75e64ec66b458e5cd5cf0944bec20590ae8": "17883c1499b92b18bf02f47d9a886a77a1b647acb74357dd769185613aef37fb",
}
TYPE_LAYOUT = {0:(1,4), 1:(1,2), 2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24),
               8:(32,34), 9:(32,40), 10:(256,84), 11:(256,110), 12:(256,144),
               13:(256,176), 14:(256,210), 15:(256,292), 16:(256,66), 17:(256,74),
               18:(256,98), 19:(256,50), 20:(32,18), 21:(256,110), 22:(256,82),
               23:(256,136), 24:(1,1), 25:(1,2), 26:(1,4), 27:(1,8), 28:(1,8),
               29:(256,56), 30:(1,2), 39:(32,17)}
TYPE_NAMES = {0:"F32", 1:"F16", 30:"BF16"}
VALUE_FORMATS = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"B", 10:"Q", 11:"q", 12:"d"}
ARRAY_KEYS = {"deepseek41.index_source_layer_ids", "deepseek41.engram.layer_ids", "deepseek41.kv_source_layer_ids"}
MAX_HEADER = 512 * 1024 * 1024
STAT_FIELDS = ("elements blocks failed_blocks values_without_accepted_block unencodable_at_all_scales "
               "roundtrip_mismatches shader_f32_mismatches positive_zero negative_zero subnormal normal infinity nan "
               "failed_scale_trials invalid_scales witness_scale_values_tested under_witness_scale_mismatch_count "
               "blocks_without_finite_witness_scale first_kind first_index first_source_bits first_reconstructed_bits "
               "first_e0 first_scale_bits").split()
FIRST_KINDS = {1:"F32_low_16_bits_nonzero", 2:"nonfinite_F16_block_rejected_before_scale_search",
               3:"value_unencodable_at_all_three_reference_scales", 4:"F16_roundtrip_bits_differ",
               5:"shader_reconstruction_F32_bits_differ",7:"block_has_no_common_scale_despite_individual_fits"}


class AuditError(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(path):
    p = Path(path).resolve(strict=True)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}


def file_stat(st):
    return {k: getattr(st, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


class LogicalFile:
    """Read-only pread over one file or an explicitly validated byte split."""
    def __init__(self, paths):
        self.parts, self.starts, self.size = [], [], 0
        seen = set()
        try:
            for path in paths:
                p = Path(path).absolute()
                fd = os.open(p, os.O_RDONLY)
                s = os.fstat(fd)
                if not stat.S_ISREG(s.st_mode) or (s.st_dev, s.st_ino) in seen:
                    os.close(fd)
                    raise AuditError("non-regular or aliased model parts")
                seen.add((s.st_dev, s.st_ino))
                self.starts.append(self.size)
                self.parts.append({"path": str(p), "realpath": str(p.resolve()), "fd": fd, "before": file_stat(s)})
                self.size += s.st_size
        except BaseException:
            self.close()
            raise

    def close(self):
        for p in self.parts:
            os.close(p["fd"])
        self.parts = []

    def read(self, offset, size):
        if offset < 0 or size < 0 or offset > self.size or size > self.size-offset:
            raise AuditError(f"read outside logical file: {offset}+{size}/{self.size}")
        chunks = []
        while size:
            i = bisect.bisect_right(self.starts, offset)-1
            p = self.parts[i]
            local = offset-self.starts[i]
            n = min(size, p["before"]["st_size"]-local)
            if n <= 0:
                raise AuditError("empty part inside logical span")
            data = os.pread(p["fd"], n, local)
            if len(data) != n:
                raise AuditError("short model read")
            chunks.append(data)
            offset += n
            size -= n
        return b"".join(chunks)

    def digest(self, offset, size):
        h = hashlib.sha256()
        for i in range(0, size, 1024*1024):
            h.update(self.read(offset+i, min(1024*1024, size-i)))
        return h.hexdigest()

    def finish_identity(self):
        out = []
        for p in self.parts:
            after_fd, after_path = file_stat(os.fstat(p["fd"])), file_stat(os.stat(p["path"]))
            unchanged = p["before"] == after_fd == after_path
            out.append({k:v for k,v in p.items() if k != "fd"} | {"after_fd": after_fd, "after_path": after_path, "unchanged": unchanged})
        return out


def byte_split_paths(path):
    p = Path(path).absolute()
    match = re.fullmatch(r"(.+)\.part([0-9]{2,})", str(p))
    base = Path(match[1]) if match else p
    manifest = Path(str(base)+".parts")
    if not match and p.exists():
        return [p], None
    if not manifest.is_file():
        if match:
            raise AuditError("byte-split part requires its .parts manifest")
        return [p], None
    raw = manifest.read_bytes()
    values = {}
    for line in raw.decode("ascii").splitlines():
        pair = line.split()
        if not pair:
            continue
        if len(pair) != 2 or pair[0] not in {"myds-split", "parts", "chunk", "total"} or pair[0] in values or not pair[1].isdigit():
            raise AuditError("invalid/duplicate byte-split manifest field")
        values[pair[0]] = int(pair[1])
    if set(values) != {"myds-split", "parts", "chunk", "total"} or values["myds-split"] != 1:
        raise AuditError("incomplete byte-split manifest")
    n, chunk, total = (values[k] for k in ("parts", "chunk", "total"))
    if not (2 <= n <= 65535 and chunk > 0 and chunk % os.sysconf("SC_PAGE_SIZE") == 0 and (n-1)*chunk < total <= n*chunk):
        raise AuditError("invalid byte-split dimensions/alignment")
    paths = [Path(f"{base}.part{i:02d}") for i in range(1,n+1)]
    if match and (int(match[2]) not in range(1,n+1) or p not in paths):
        raise AuditError("byte-split input part name is not canonical")
    for i, part in enumerate(paths):
        if part.stat().st_size != (chunk if i<n-1 else total-(n-1)*chunk):
            raise AuditError("byte-split missing/truncated/extra-sized part")
    return paths, {"path":str(manifest), "sha256":sha(raw), "bytes":len(raw), "fields":values}


class Reader:
    def __init__(self, f):
        self.f, self.pos, self.cache_start, self.cache = f, 0, 0, b""

    def read(self, n):
        if n < 0 or self.pos+n > min(self.f.size, MAX_HEADER):
            raise AuditError("truncated or excessively large GGUF header")
        if self.pos < self.cache_start or self.pos+n > self.cache_start+len(self.cache):
            self.cache_start = self.pos
            self.cache = self.f.read(self.pos, min(max(65536,n),self.f.size-self.pos))
        start = self.pos-self.cache_start
        self.pos += n
        return self.cache[start:start+n]

    def unpack(self, fmt):
        return struct.unpack("<"+fmt, self.read(struct.calcsize("<"+fmt)))[0]

    def string(self, keep=True):
        n = self.unpack("Q")
        if n > 64*1024*1024:
            raise AuditError("GGUF string exceeds audit limit")
        raw = self.read(n)
        if keep:
            value = raw.decode("utf-8", "strict")
            if "\0" in value:
                raise AuditError("NUL in GGUF identifier/string")
            return value
        return None

    def value(self, t, keep=False):
        if t == 8:
            return self.string(keep)
        if t == 9:
            et, n = self.unpack("I"), self.unpack("Q")
            if n > 100_000_000 or et not in VALUE_FORMATS and et != 8:
                raise AuditError("invalid/unsupported metadata array")
            if keep and n > 4096:
                raise AuditError("retained metadata array exceeds audit limit")
            if et == 8:
                for _ in range(n):
                    self.string(False)
                return {"element_type":et,"count":n}
            fmt = VALUE_FORMATS[et]
            if keep:
                return [self.unpack(fmt) for _ in range(n)]
            size = n*struct.calcsize("<"+fmt)
            if size > min(self.f.size,MAX_HEADER)-self.pos:
                raise AuditError("truncated metadata array")
            self.pos += size
            return {"element_type":et,"count":n}
        if t not in VALUE_FORMATS:
            raise AuditError(f"unknown GGUF metadata type {t}")
        v = self.unpack(VALUE_FORMATS[t])
        if t == 7 and v not in (0,1):
            raise AuditError("invalid metadata BOOL")
        if isinstance(v,float) and not math.isfinite(v):
            return {"nonfinite_bits":struct.pack("<"+VALUE_FORMATS[t],v).hex()}
        return v


def tensor_size(t):
    if t["type"] not in TYPE_LAYOUT:
        raise AuditError(f"unsupported tensor layout {t['type']} ({t['name']})")
    block, size = TYPE_LAYOUT[t["type"]]
    dims = t["dims"]
    if dims[0] % block:
        raise AuditError("quantized tensor row is not block aligned")
    return math.prod(dims)*size//block


def parse_gguf(f, part_index=0):
    r = Reader(f)
    if r.read(4) != b"GGUF":
        raise AuditError("not little-endian GGUF")
    version, nt, nk = r.unpack("I"), r.unpack("Q"), r.unpack("Q")
    if version not in (2,3) or nt > 1_000_000 or nk > 1_000_000:
        raise AuditError("unsupported GGUF version/count")
    metadata, metadata_types, identities, array_types = {}, {}, {}, {}
    for _ in range(nk):
        key, t = r.string(), r.unpack("I")
        if key in metadata:
            raise AuditError("duplicate metadata key")
        metadata_types[key] = t
        keep = t != 9 or key in ARRAY_KEYS
        start = r.pos
        metadata[key] = r.value(t, keep)
        identities[key] = {"type":t,"bytes":r.pos-start,"sha256":f.digest(start,r.pos-start)}
        if t == 9:
            array_types[key] = struct.unpack("<I",f.read(start,4))[0]
    alignment = metadata.get("general.alignment",32)
    if not isinstance(alignment,int) or alignment < 1 or alignment > 65536 or alignment & (alignment-1):
        raise AuditError("invalid GGUF alignment")
    tensors, names = [], set()
    for _ in range(nt):
        name, nd = r.string(), r.unpack("I")
        if name in names or not 1 <= nd <= 8:
            raise AuditError("duplicate tensor or invalid rank")
        names.add(name)
        dims = [r.unpack("Q") for _ in range(nd)]
        if any(d == 0 or d > 2**63-1 for d in dims):
            raise AuditError("invalid tensor dimensions")
        t = {"name":name,"dims":dims,"type":r.unpack("I"),"offset":r.unpack("Q"),"part_index":part_index}
        t["bytes"] = tensor_size(t)
        if t["offset"] % alignment:
            raise AuditError("unaligned tensor offset")
        tensors.append(t)
    data_offset = (r.pos+alignment-1)//alignment*alignment
    end = data_offset
    for t in sorted(tensors,key=lambda t:t["offset"]):
        t["file_offset"] = data_offset+t["offset"]
        if t["file_offset"] < end or t["bytes"] > f.size-t["file_offset"]:
            raise AuditError("overlapping/truncated tensor span")
        end = t["file_offset"]+t["bytes"]
    if data_offset > f.size:
        raise AuditError("truncated GGUF alignment padding")
    public = {k:v for k,v in metadata.items() if k.startswith(("general.","deepseek4", "split."))}
    return {"version":version,"tensor_count":nt,"metadata_count":nk,"alignment":alignment,
            "data_offset":data_offset,"header_sha256":f.digest(0,data_offset),
            "metadata":public,"metadata_types":{k:metadata_types[k] for k in public},
            "metadata_array_types":{k:v for k,v in array_types.items() if k in public},
            "metadata_value_identities":identities,"tensors":tensors,"reader":f}


def open_model(path):
    path = Path(path).absolute()
    match = re.fullmatch(r"(.+)-([0-9]+)-of-([0-9]+)\.gguf",str(path))
    readers, headers, manifest = [], [], None
    try:
        if match and int(match[3]) > 1:
            width, n = len(match[2]), int(match[3])
            if width != len(match[3]) or not 1 <= int(match[2]) <= n <= 65535:
                raise AuditError("invalid tensor-split name")
            for i in range(1,n+1):
                f = LogicalFile([f"{match[1]}-{i:0{width}}-of-{n:0{width}}.gguf"])
                readers.append(f)
                h = parse_gguf(f,i-1)
                m = h["metadata"]
                if m.get("split.no") != i-1 or m.get("split.count") != n:
                    raise AuditError("tensor-split filename/metadata disagreement")
                headers.append(h)
            common = {k:v for k,v in headers[0]["metadata_value_identities"].items() if not k.startswith("split.")}
            for h in headers[1:]:
                if h["version"] != headers[0]["version"] or h["alignment"] != headers[0]["alignment"]:
                    raise AuditError("tensor-split version/alignment mismatch")
                present = {k:v for k,v in h["metadata_value_identities"].items() if not k.startswith("split.")}
                if any(k not in common or common[k] != v for k,v in present.items()):
                    raise AuditError("tensor-split identity metadata mismatch")
            total = sum(h["tensor_count"] for h in headers)
            if any(h["metadata"].get("split.tensors.count") != total for h in headers):
                raise AuditError("tensor-split total tensor count mismatch")
            kind = "tensor_split"
        else:
            paths, manifest = byte_split_paths(path)
            f = LogicalFile(paths)
            readers.append(f)
            headers.append(parse_gguf(f))
            if headers[0]["metadata"].get("split.count",1) != 1:
                raise AuditError("split GGUF requires canonical split filename")
            kind = "byte_split" if manifest else "single"
        tensors = {}
        inodes = set()
        for f in readers:
            for p in f.parts:
                inode = (p["before"]["st_dev"],p["before"]["st_ino"])
                if inode in inodes:
                    raise AuditError("physical file reused across model parts")
                inodes.add(inode)
        for h in headers:
            for t in h["tensors"]:
                if t["name"] in tensors:
                    raise AuditError("duplicate tensor across model parts")
                tensors[t["name"]] = t | {"reader":h["reader"]}
        return {"kind":kind,"readers":readers,"headers":headers,"tensors":tensors,"byte_split_manifest":manifest}
    except BaseException:
        for f in readers:
            f.close()
        raise


def targets():
    return ([("router",f"blk.{i}.ffn_gate_inp.weight",[5120,384],0) for i in range(40)] +
            [("index_qb",f"blk.{i}.indexer.attn_q_b.weight",[1280,4096],1) for i in INDEX_LAYERS] +
            [("engram",f"blk.{i}.engram_kv.weight",[6144,25600],1) for i in ENGRAM_LAYERS])


def validate_scope(model):
    header = model["headers"][0]
    m = header["metadata"]
    arch = m.get("general.architecture","")
    if arch != "deepseek41" or header["metadata_types"].get("general.architecture") != 8:
        raise AuditError("target architecture must be exactly deepseek41 (native V4.1 model-create contract)")

    def dimension(primary, fallback, required):
        # Match myds_engine.c's actual key order, without accepting contradictory
        # aliases, absent dimensions, implicit defaults, or uint64->uint32 wrap.
        keys = [arch+"."+primary,arch+"."+fallback]
        present = []
        for key in keys:
            if key in m:
                value, t = m[key], header["metadata_types"].get(key)
                if t not in (4,5,10) or type(value) is not int:
                    raise AuditError(f"dimension metadata {key} must use an integer type accepted by gguf_get_u32")
                present.append({"key":key,"value":value,"gguf_type":t,
                    "metadata_value_sha256":header["metadata_value_identities"][key]["sha256"]})
        if not present:
            raise AuditError("missing dimension metadata: "+" or ".join(keys))
        if len(present)==2 and present[0]["value"] != present[1]["value"]:
            raise AuditError("conflicting dimension metadata: "+repr(present))
        if present[0]["value"] != required:
            raise AuditError(f"dimension {primary}/{fallback} must equal {required}, got {present[0]['value']}")
        return {"value":required,"selected_key":present[0]["key"],"selected_gguf_type":present[0]["gguf_type"],
                "present_aliases":present,"alias_policy":"primary then fallback; simultaneous aliases must agree exactly"}

    binding = {"architecture":{"key":"general.architecture","value":arch,"gguf_type":8},
               "layers":dimension("block_count","num_hidden_layers",40),
               "width":dimension("embedding_length","hidden_size",5120)}
    if m.get("deepseek41.index_source_layer_ids") != list(INDEX_LAYERS) or m.get("deepseek41.engram.layer_ids") != list(ENGRAM_LAYERS):
        raise AuditError("actual index/Engram source layers differ from the audited scope")
    for key in ("deepseek41.index_source_layer_ids","deepseek41.engram.layer_ids"):
        if model["headers"][0]["metadata_array_types"].get(key) != 4:
            raise AuditError("source layer metadata must use UINT32 arrays")
    expected = {t[1] for t in targets()}
    relevant = re.compile(r"^blk\.[0-9]+\.(ffn_gate_inp|indexer\.attn_q_b|engram_kv)\.weight$")
    extra = sorted(n for n in model["tensors"] if relevant.fullmatch(n) and n not in expected)
    if extra:
        raise AuditError("unexpected target-family tensors: "+repr(extra))
    return binding


class Stats(ctypes.Structure):
    _fields_ = [(k,ctypes.c_uint64) for k in STAT_FIELDS]


def encoder_body(text):
    start = text.index("bool ds41_e4m3_code(")
    start = text.index("{",start)
    depth = 0
    for i in range(start,len(text)):
        depth += (text[i] == "{")-(text[i] == "}")
        if depth == 0:
            return text[start:i+1]
    raise AuditError("truncated reference encoder")


def build_worker(tmpdir, cc, patch_dir):
    original_source = Path(__file__).with_name("v41_narrow_weights_worker.c")
    directory = Path(tmpdir)
    source = directory/"v41_narrow_weights_worker.c"
    refs = []
    for commit, expected in UPSTREAM.items():
        p = patch_dir/(commit+".patch")
        d = descriptor(p)
        if d["sha256"] != expected:
            raise AuditError("upstream patch identity mismatch: "+str(p))
        refs.append(d | {"commit":commit})
    patch = Path(refs[0]["path"]).read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    if encoder_body(original_source.read_text()) != encoder_body(added):
        raise AuditError("worker E4M3 encoder differs from pinned upstream body")
    source.write_bytes(original_source.read_bytes())
    (directory/"audit_v41_narrow_weights.py").write_bytes(Path(__file__).read_bytes())
    reference_dir = directory/"upstream-commits"
    reference_dir.mkdir(exist_ok=True)
    for d in refs:
        (reference_dir/Path(d["path"]).name).write_bytes(Path(d["path"]).read_bytes())
    compiler = shutil.which(cc)
    if not compiler:
        raise AuditError("C compiler not found: "+cc)
    so = Path(tmpdir)/("worker.dylib" if sys.platform == "darwin" else "worker.so")
    cmd = [compiler,"-std=c11","-O2","-fno-fast-math","-ffp-contract=off","-fPIC",
           "-dynamiclib" if sys.platform == "darwin" else "-shared",str(source),"-o",str(so),"-lm"]
    # Do not inherit unrelated host credentials or compiler-injection variables.
    env = {"PATH":os.environ.get("PATH",os.defpath),"LC_ALL":"C"}
    for k in ("DEVELOPER_DIR","SDKROOT","MACOSX_DEPLOYMENT_TARGET","TMPDIR","SYSTEMROOT"):
        if k in os.environ:
            env[k] = os.environ[k]
    command_path = directory/"compile-command.json"
    command_path.write_text(json.dumps({"argv":cmd,"environment":env,"cwd":os.getcwd()},indent=2,sort_keys=True)+"\n")
    built = subprocess.run(cmd,capture_output=True,text=True,check=False,env=env)
    (directory/"compile-stdout.txt").write_text(built.stdout)
    (directory/"compile-stderr.txt").write_text(built.stderr)
    if built.returncode:
        raise AuditError("worker CPU build failed: "+built.stderr)
    lib = ctypes.CDLL(str(so))
    lib.v41_audit_abi.restype = ctypes.c_uint32
    if lib.v41_audit_abi() != ctypes.sizeof(Stats):
        raise AuditError("worker stats ABI mismatch")
    ptr = ctypes.c_void_p
    lib.v41_audit_router.argtypes = [ptr,ctypes.c_uint64,ptr,ptr,ctypes.POINTER(Stats)]
    lib.v41_audit_e4m3.argtypes = [ptr,ctypes.c_uint64,ptr,ptr,ptr,ptr,ctypes.POINTER(Stats)]
    for fn in (lib.v41_audit_router,lib.v41_audit_e4m3):
        fn.restype = ctypes.c_int
    version = subprocess.run([compiler,"--version"],capture_output=True,text=True,check=False,env=env)
    (directory/"compiler-version.txt").write_text(version.stdout+version.stderr)
    return lib, {"source":descriptor(source),"original_source":descriptor(original_source),
                 "binary":descriptor(so),"argv":cmd,"compile_command":descriptor(command_path),
                 "compile_environment":env,"compile_stdout":descriptor(directory/"compile-stdout.txt"),
                 "compile_stderr":descriptor(directory/"compile-stderr.txt"),
                 "compiler_version":version.stdout,"stdout":built.stdout,"stderr":built.stderr,
                 "upstream_patches":refs,"encoder_body_sha256":sha(encoder_body(added).encode())}


def first_record(s, base, cols, width):
    if not s.first_kind:
        return None
    i = base+s.first_index
    def number(bits):
        x = struct.unpack("<f" if width==4 else "<e",int(bits).to_bytes(width,"little"))[0]
        return x if math.isfinite(x) else "NaN" if math.isnan(x) else "+Infinity" if x>0 else "-Infinity"
    reconstructed = None if s.first_kind in (2,3,7) else s.first_reconstructed_bits
    return {"kind":FIRST_KINDS[s.first_kind],"index":i,"row":i//cols,"column":i%cols,
            "source_bits":f"0x{s.first_source_bits:0{width*2}x}",
            "source_value":number(s.first_source_bits),
            "reconstructed_bits":None if reconstructed is None else f"0x{reconstructed:0{width*2}x}",
            "reconstructed_value":None if reconstructed is None else number(reconstructed),
            "e0":ctypes.c_int64(s.first_e0).value,"scale_f32_bits":f"0x{s.first_scale_bits:08x}",
            "note":"witness fails under e0; each value fits some trial, but no single scale fits the block" if s.first_kind == 7 else None}


def audit_tensor(t, family, lib):
    cols, rows = t["dims"]
    projection = family != "router"
    width = 2 if projection else 4
    source_h, restored_h, codes_h, scales_h, valid_h = (hashlib.sha256() for _ in range(5))
    totals = {k:0 for k in STAT_FIELDS if not k.startswith("first_")}
    first, first_nonfinite, scales_count, scale_min, scale_max = None, None, 0, None, None
    stride_rows = 32 if projection else min(rows,32)
    for row in range(0,rows,stride_rows):
        nr = min(stride_rows,rows-row)
        raw = t["reader"].read(t["file_offset"]+row*cols*width,nr*cols*width)
        source_h.update(raw)
        src = ctypes.create_string_buffer(raw,len(raw))
        n = nr*cols
        code = ctypes.create_string_buffer(n if projection else 2*n)
        restored = ctypes.create_string_buffer(width*n)
        s = Stats()
        if projection:
            if nr != 32:
                raise AuditError("projection stripe is not 32-row aligned")
            scales, mask = ctypes.create_string_buffer(cols//32*4), ctypes.create_string_buffer(cols//32)
            rc = lib.v41_audit_e4m3(src,cols,code,scales,restored,mask,ctypes.byref(s))
            scales_h.update(scales.raw)
            valid_h.update(mask.raw)
            for (scale,), valid in zip(struct.iter_unpack("<f",scales.raw),mask.raw):
                if valid:
                    if not math.isfinite(scale) or scale <= 0 or math.frexp(scale)[0] != 0.5:
                        raise AuditError("worker returned invalid non-power-of-two scale")
                    scales_count += 1
                    scale_min = scale if scale_min is None else min(scale_min,scale)
                    scale_max = scale if scale_max is None else max(scale_max,scale)
        else:
            rc = lib.v41_audit_router(src,n,code,restored,ctypes.byref(s))
        if rc:
            raise AuditError("worker rejected a validated input span")
        codes_h.update(code.raw)
        restored_h.update(restored.raw)
        for k in totals:
            totals[k] += getattr(s,k)
        if first_nonfinite is None and (s.infinity or s.nan):
            exp_mask = 0x7f800000 if width == 4 else 0x7c00
            for i,(bits,) in enumerate(struct.iter_unpack("<I" if width==4 else "<H",raw)):
                if bits & exp_mask == exp_mask:
                    index = row*cols+i
                    first_nonfinite = {"index":index,"row":index//cols,"column":index%cols,
                                       "source_bits":f"0x{bits:0{width*2}x}"}
                    break
        f = first_record(s,row*cols,cols,width)
        if f and (first is None or f["index"] < first["index"]):
            first = f
    bits_reversible = not any(totals[k] for k in ("failed_blocks","roundtrip_mismatches","shader_f32_mismatches","invalid_scales"))
    finite = totals["infinity"] == totals["nan"] == 0
    accepted = bits_reversible and finite
    return {"qualification_passed":accepted,"bits_reversible":bits_reversible,"finite_weights":finite,
            "qualification_rejection_reasons":([] if finite else ["nonfinite_weights"]) + ([] if bits_reversible else ["not_bits_reversible"]),
            "counts":totals,"first_failure":first,"first_nonfinite":first_nonfinite,
            "source_sha256":source_h.hexdigest(),"reconstructed_stream_sha256":restored_h.hexdigest(),
            "codes_stream_sha256":codes_h.hexdigest(),"scales_stream_sha256":scales_h.hexdigest() if projection else None,
            "block_valid_mask_sha256":valid_h.hexdigest() if projection else None,
            "streams_form_valid_complete_representation":bits_reversible,
            "invalid_block_placeholders":{"code":0,"scale_f32_bits":"0x7fc00000","reconstructed_f16_bits":"0x7e00","mask":0} if projection else None,
            "valid_scales":{"count":scales_count,"min":scale_min,"max":scale_max} if projection else None,
            "source_bytes_scanned":t["bytes"],"code_payload_bytes":rows*cols*(1 if projection else 2),
            "scale_payload_bytes":rows//32*(cols//32)*4 if projection else 0}


def refuse_output_alias(output, model, extra_inputs=()):
    out = Path(output).absolute()
    inputs = list(extra_inputs)
    for f in model["readers"]:
        inputs += [p["path"] for p in f.parts]
    if model["byte_split_manifest"]:
        inputs.append(model["byte_split_manifest"]["path"])
    for p in inputs:
        if out.resolve() == Path(p).resolve() or out.exists() and Path(p).exists() and os.path.samefile(out,p):
            raise AuditError("output aliases an input file")


def candidate_input_paths(path):
    """Protect output aliases even when a malformed model fails before open_model returns."""
    p = Path(path).absolute()
    result = [p]
    byte = re.fullmatch(r"(.+)\.part[0-9]{2,}",str(p))
    base = Path(byte[1]) if byte else p
    result.append(Path(str(base)+".parts"))
    result.extend(base.parent.glob(base.name+".part[0-9]*"))
    split = re.fullmatch(r"(.+)-([0-9]+)-of-([0-9]+)\.gguf",str(p))
    if split:
        prefix = Path(split[1])
        result.extend(prefix.parent.glob(prefix.name+"-*-of-"+split[3]+".gguf"))
    return result


def write_report(path, report):
    p = Path(path).absolute()
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=p.parent,prefix=".narrow-audit-",suffix=".tmp",delete=False) as f:
        tmp = f.name
        try:
            json.dump(report,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            os.unlink(tmp)
            raise
    try:
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run(args):
    report = {"schema_version":1,"kind":"v41_narrow_weights_CPU_bit_reversibility_audit",
              "GPU_work_submitted":False,"model_modified":False,"full_model_sha256":None,
              "identity_scope":"all GGUF headers, all target payload bytes, per-file stat before/after; not non-target payload contents",
              "arithmetic_equivalence_or_performance_proven":False,"argv":sys.argv,"tensors":[],"families":{},"errors":[],
              "host":{"platform":platform.platform(),"machine":platform.machine(),"python":sys.version,"libc":platform.libc_ver()},
              "expected_targets":[{"family":f,"name":n,"dims":d,"source_type":TYPE_NAMES[t]} for f,n,d,t in targets()],
              "policies":{"router":"upstream low-16-bit test and raw BF16 re-expansion; finite-weight qualification is a separate stricter gate",
                          "e4m3":"original encoder and e0,e0+1,e0-1 search; reject nonfinite blocks before undefined scale calculation",
                          "under_witness_scale_mismatch_count":"number of values rejected by the exact encoder or differing in F32 shader reconstruction under the e0 witness scale; not approximate quantization",
                          "failed_blocks":"complete count; values_without_accepted_block is not the individually unencodable-value count"}}
    model, safe_output = None, False
    started = time.monotonic()
    try:
        if sys.byteorder != "little":
            raise AuditError("worker ABI currently requires a little-endian CPU")
        patch_dir = Path(args.upstream_patches).absolute()
        protected = [__file__,str(Path(__file__).with_name("v41_narrow_weights_worker.c"))]
        protected += [str(patch_dir/(c+".patch")) for c in UPSTREAM]
        if shutil.which(args.cc):
            protected.append(shutil.which(args.cc))
        refuse_output_alias(args.output,{"readers":[],"byte_split_manifest":None},protected+candidate_input_paths(args.model))
        safe_output = True
        model = open_model(args.model)
        safe_output = False
        refuse_output_alias(args.output,model,protected)
        safe_output = True
        report["tool"] = descriptor(__file__)
        report["model"] = {"input":str(Path(args.model).absolute()),"kind":model["kind"],
                           "byte_split_manifest":model["byte_split_manifest"],
                           "headers":[{k:v for k,v in h.items() if k not in ("reader","tensors")} for h in model["headers"]]}
        report["scope_binding"] = validate_scope(model)
        artifact = Path(args.artifact_dir if args.artifact_dir else str(Path(args.output).absolute())+".artifacts")
        if not artifact.is_absolute():
            raise AuditError("--artifact-dir must be an absolute NEW directory")
        if Path(args.output).resolve().is_relative_to(artifact.resolve()):
            safe_output = False
            raise AuditError("report output must be outside the retained artifact directory")
        artifact.mkdir(mode=0o700)
        report["artifact_directory"] = str(artifact)
        lib, provenance = build_worker(artifact,args.cc,patch_dir)
        report["worker"] = provenance
        for family,name,dims,ttype in targets():
            rec = {"family":family,"name":name,"expected_dims":dims,"expected_type":TYPE_NAMES[ttype]}
            t = model["tensors"].get(name)
            if t is None:
                rec.update(qualification_passed=False, gate_error="missing_target_tensor")
            else:
                rec.update({k:v for k,v in t.items() if k != "reader"})
                rec["actual_type_name"] = TYPE_NAMES.get(t["type"],f"GGML_TYPE_{t['type']}")
                if t["dims"] != dims:
                    rec.update(qualification_passed=False,gate_error="target_shape_mismatch_payload_not_scanned")
                elif t["type"] != ttype:
                    rec.update(qualification_passed=False,gate_error="upstream_converter_requires_different_source_type",
                               source_sha256=t["reader"].digest(t["file_offset"],t["bytes"]),source_bytes_scanned=t["bytes"])
                else:
                    rec.update(audit_tensor(t,family,lib))
            report["tensors"].append(rec)
            print(f"{family} {name}: {'PASS' if rec['qualification_passed'] else 'REJECT'}",file=sys.stderr,flush=True)
        for family in ("router","index_qb","engram"):
            records = [r for r in report["tensors"] if r["family"] == family]
            count_keys = {k for r in records for k in r.get("counts",{})}
            rejected = [r for r in records if not r["qualification_passed"]]
            first = rejected[0] if rejected else None
            identities = [{"name":r["name"],"type":r.get("type"),"dims":r.get("dims"),
                           "source_sha256":r.get("source_sha256")} for r in records]
            report["families"][family] = {"tensor_count":len(records),"qualified_tensor_count":sum(r["qualification_passed"] for r in records),
                "all_tensors_qualify":all(r["qualification_passed"] for r in records),
                "source_bytes_scanned":sum(r.get("source_bytes_scanned",0) for r in records),
                "potential_code_payload_bytes":sum(r.get("code_payload_bytes",0) for r in records),
                "potential_scale_payload_bytes":sum(r.get("scale_payload_bytes",0) for r in records),
                "qualified_code_payload_bytes":sum(r.get("code_payload_bytes",0) for r in records if r["qualification_passed"]),
                "qualified_scale_payload_bytes":sum(r.get("scale_payload_bytes",0) for r in records if r["qualification_passed"]),
                "counts":{k:sum(r.get("counts",{}).get(k,0) for r in records) for k in sorted(count_keys)},
                "first_rejection":None if first is None else {"name":first["name"],"gate_error":first.get("gate_error"),
                    "qualification_rejection_reasons":first.get("qualification_rejection_reasons"),"first_failure":first.get("first_failure"),
                    "first_nonfinite":first.get("first_nonfinite")},
                "source_tensor_manifest_sha256":sha(json.dumps(identities,sort_keys=True,separators=(",",":")).encode()),
                "source_tensor_manifest_hash_definition":"SHA256 of canonical JSON array of name/type/dims/source_sha256, in expected target order; not raw concatenated model bytes",
                "gate_failures":[{"name":r["name"],"reason":r["gate_error"]} for r in records if "gate_error" in r],
                "upstream_fallback_scope":"all index-QB layers retain F16 if any index layer fails" if family == "index_qb" else "all routers retain F32 after any router failure" if family == "router" else "each Engram table qualifies independently"}
        report["audit_completed"] = True
    except (AuditError,OSError,UnicodeError,struct.error,OverflowError,subprocess.SubprocessError) as e:
        report["errors"].append({"type":type(e).__name__,"message":str(e)})
        report["audit_completed"] = False
    finally:
        if model:
            try:
                report["model_file_identity"] = [p for f in model["readers"] for p in f.finish_identity()]
                if not all(p["unchanged"] for p in report["model_file_identity"]):
                    report["errors"].append({"type":"IdentityError","message":"model file identity/stat changed during audit"})
                # Re-read all metadata headers: stat alone is not a content oracle.
                for h in model["headers"]:
                    if h["reader"].digest(0,h["data_offset"]) != h["header_sha256"]:
                        report["errors"].append({"type":"IdentityError","message":"GGUF header changed during audit"})
                if model["byte_split_manifest"]:
                    d = model["byte_split_manifest"]
                    if sha(Path(d["path"]).read_bytes()) != d["sha256"]:
                        report["errors"].append({"type":"IdentityError","message":"byte-split manifest changed during audit"})
            except (OSError,AuditError) as e:
                report["errors"].append({"type":"IdentityError","message":str(e)})
            for f in model["readers"]:
                f.close()
        if report.get("artifact_directory"):
            d = Path(report["artifact_directory"])
            inventory = [{"relative_path":str(p.relative_to(d)),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())}
                         for p in sorted(d.rglob("*")) if p.is_file() and p.name != "artifact-manifest.json"]
            manifest = d/"artifact-manifest.json"
            manifest.write_text(json.dumps({"schema_version":1,"kind":"CPU_narrow_audit_compiler_artifacts",
                "GPU_work_submitted":False,"files":inventory},indent=2,sort_keys=True)+"\n")
            report["artifact_manifest"] = descriptor(manifest)
    report["elapsed_seconds"] = time.monotonic()-started
    report["qualification_passed"] = bool(report.get("audit_completed") and not report["errors"] and len(report["tensors"])==50 and all(r["qualification_passed"] for r in report["tensors"]))
    report["exit_code"] = 2 if report["errors"] else 0 if report["qualification_passed"] else 1
    if safe_output:
        try:
            write_report(args.output,report)
        except OSError as e:
            report["errors"].append({"type":"ReportWriteError","message":str(e)})
            report["exit_code"] = 2
            print(json.dumps(report,sort_keys=True,allow_nan=False),file=sys.stderr)
    else:
        print(json.dumps(report,sort_keys=True,allow_nan=False),file=sys.stderr)
    return report["exit_code"]


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",required=True,help="single GGUF, canonical tensor-split part, or byte-split base/.partNN")
    p.add_argument("--output",required=True,help="JSON audit report, separate from every model/source input")
    p.add_argument("--cc",default=os.environ.get("CC","cc"),help="C compiler executable, not a shell command")
    p.add_argument("--artifact-dir",help="absolute NEW directory retaining actual compiler inputs/binary/logs; default OUTPUT.artifacts")
    p.add_argument("--upstream-patches",default=str(Path(__file__).resolve().parents[1]/"docs/benchmarks/2026-10-26-upstream-plan-review/upstream-commits"))
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
