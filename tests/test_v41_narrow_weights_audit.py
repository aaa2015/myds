#!/usr/bin/env python3
"""CPU fixtures for the read-only audit; no model, Metal, or GPU required."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("narrow_audit",ROOT/"tools/audit_v41_narrow_weights.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
PATCHES = ROOT/"docs/benchmarks/2026-10-26-upstream-plan-review/upstream-commits"
if not PATCHES.exists():
    PATCHES = ROOT.parent/"mydsdoc/benchmarks/2026-10-26-upstream-plan-review/upstream-commits"


def ggstr(s):
    raw = s.encode()
    return struct.pack("<Q",len(raw))+raw


def metadata():
    return {"general.architecture":(8,"deepseek41"),"deepseek41.block_count":(4,40),
            "deepseek41.embedding_length":(4,5120),
            "deepseek41.index_source_layer_ids":(9,(4,list(A.INDEX_LAYERS))),
            "deepseek41.engram.layer_ids":(9,(4,list(A.ENGRAM_LAYERS)))}


def native_header_metadata():
    """Actual target's names/types, independently specified from loader code."""
    return {"general.architecture":(8,"deepseek41"),"deepseek41.num_hidden_layers":(4,40),
            "deepseek41.hidden_size":(4,5120),
            "deepseek41.index_source_layer_ids":(9,(4,[2,8,14,20,24,28,32,36])),
            "deepseek41.engram.layer_ids":(9,(4,[1,14]))}


def make_gguf(path, tensors=(), kv=None, alignment=32):
    kv = dict(metadata() if kv is None else kv)
    kv["general.alignment"] = (4,alignment)
    head = bytearray(b"GGUF"+struct.pack("<IQQ",3,len(tensors),len(kv)))
    for key,(t,v) in kv.items():
        head += ggstr(key)+struct.pack("<I",t)
        if t==8:
            head += ggstr(v)
        elif t==9:
            et,values = v
            head += struct.pack("<IQ",et,len(values))+struct.pack("<"+A.VALUE_FORMATS[et]*len(values),*values)
        else:
            head += struct.pack("<"+A.VALUE_FORMATS[t],v)
    payload = bytearray()
    for name,dims,ttype,raw in tensors:
        payload += b"\0"*((-len(payload))%alignment)
        head += ggstr(name)+struct.pack("<I",len(dims))+struct.pack("<"+"Q"*len(dims),*dims)+struct.pack("<IQ",ttype,len(payload))
        payload += raw
    head += b"\0"*((-len(head))%alignment)
    Path(path).write_bytes(head+payload)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_tmp = tempfile.TemporaryDirectory(prefix="narrow-audit-tests-")
        cls.lib, cls.provenance = A.build_worker(cls.build_tmp.name,"cc",PATCHES)
        # Independently compile the ORIGINAL block-row selector from the pinned
        # patch. Only __fp16 storage spelling is changed to portable _Float16;
        # encoder and selector bodies are copied exactly, with original early exits.
        patch = (PATCHES/(next(iter(A.UPSTREAM))+".patch")).read_text()
        added = "\n".join(s[1:] for s in patch.splitlines() if s.startswith("+") and not s.startswith("+++"))
        enc = "static bool ds41_e4m3_code(float q, uint8_t *code) "+A.encoder_body(added)
        begin = added.index("typedef struct {\n    const __fp16 *src;")
        finish = added.index("\n#endif",begin)
        job = added[begin:finish].replace("__fp16","_Float16")
        ref = "#include <stdbool.h>\n#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n"+enc+"\n"+job+"\n"+"""
int reference_block(const uint16_t *src, uint8_t *codes, float *scale) {
    ds41_e4m3_job j={(const _Float16 *)src,32,codes,scale,false};
    ds41_e4m3_block_row(&j,0); return !j.failed;
}
"""
        source = Path(cls.build_tmp.name)/"original_reference.c"
        source.write_text(ref)
        binary = Path(cls.build_tmp.name)/"original_reference.so"
        cmd = ["cc","-std=c11","-O2","-fno-fast-math","-ffp-contract=off","-fPIC",
               "-dynamiclib" if sys.platform=="darwin" else "-shared",str(source),"-o",str(binary),"-lm"]
        subprocess.run(cmd,check=True,capture_output=True)
        cls.reference = ctypes.CDLL(str(binary))
        cls.reference.reference_block.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        cls.reference.reference_block.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.build_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def audit_raw(self,raw,family,cols=32):
        p = self.dir/"tensor.raw"
        p.write_bytes(raw)
        f = A.LogicalFile([p])
        self.addCleanup(f.close)
        width = 4 if family=="router" else 2
        t = {"dims":[cols,len(raw)//width//cols],"reader":f,"file_offset":0,"bytes":len(raw)}
        return A.audit_tensor(t,family,self.lib)

    def projection_worker(self,raw,cols=32):
        n = len(raw)//2
        src, codes = ctypes.create_string_buffer(raw,len(raw)), ctypes.create_string_buffer(n)
        scales, restored = ctypes.create_string_buffer(cols//32*4),ctypes.create_string_buffer(len(raw))
        valid, stats = ctypes.create_string_buffer(cols//32), A.Stats()
        rc = self.lib.v41_audit_e4m3(src,cols,codes,scales,restored,valid,ctypes.byref(stats))
        self.assertEqual(rc,0)
        return src,codes,scales,restored,valid,stats

    def test_router_raw_bits_signed_zero_subnormal_and_nan(self):
        bits = [0,0x80000000,0x00010000,1,0x3f800000,0x3f800001,0x7f800000,0xff800000,0x7fc00000,0x7f800001]
        # Worker has no row constraint; fixture pads to one diagnostic row.
        bits += [0]*(32-len(bits))
        r = self.audit_raw(struct.pack("<32I",*bits),"router")
        self.assertFalse(r["qualification_passed"])
        self.assertFalse(r["bits_reversible"])
        self.assertFalse(r["finite_weights"])
        self.assertEqual(r["counts"]["roundtrip_mismatches"],3)
        self.assertEqual(r["counts"]["negative_zero"],1)
        self.assertEqual(r["counts"]["subnormal"],2)
        self.assertEqual(r["counts"]["infinity"],2)
        self.assertEqual(r["counts"]["nan"],2)
        self.assertEqual(r["first_failure"]["index"],3)
        self.assertEqual(r["source_sha256"],hashlib.sha256(struct.pack("<32I",*bits)).hexdigest())

    def test_router_nonfinite_can_be_raw_reversible_but_never_finite_qualified(self):
        r = self.audit_raw(struct.pack("<32I",*([0x7fc00000]+[0]*31)),"router")
        self.assertTrue(r["bits_reversible"])
        self.assertTrue(r["streams_form_valid_complete_representation"])
        self.assertFalse(r["qualification_passed"])
        self.assertFalse(r["finite_weights"])
        self.assertEqual(r["first_nonfinite"]["index"],0)

    def test_router_finite_exact_complete_sha(self):
        raw = struct.pack("<32I",*([0,0x80000000,0x00010000,0x3f800000]*8))
        r = self.audit_raw(raw,"router")
        self.assertTrue(r["qualification_passed"])
        self.assertEqual(r["source_sha256"],r["reconstructed_stream_sha256"])
        self.assertEqual(r["code_payload_bytes"],64)

    def test_projection_signed_zero_subnormal_and_large_finite(self):
        for h in (0,0x8000,1,0x8001,0x3c00,0x7b00):
            with self.subTest(bits=hex(h)):
                raw = struct.pack("<H",h)*1024
                r = self.audit_raw(raw,"index_qb")
                self.assertTrue(r["qualification_passed"])
                self.assertEqual(r["source_sha256"],r["reconstructed_stream_sha256"])
                self.assertEqual(r["counts"]["elements"],1024)
                self.assertEqual(r["counts"]["shader_f32_mismatches"],0)
                self.assertEqual(r["valid_scales"]["count"],1)
        zero = self.audit_raw(b"\0\0"*1024,"engram")
        self.assertEqual(zero["valid_scales"]["min"],1.0)

    def test_original_reference_selector_and_scale_order(self):
        rng = random.Random(0x741)
        blocks = [[0]*1024,[0x8000]*1024,[1]*1024,[0x3c00]*1024,[0x3c01]*1024,[0x7bff]*1024]
        # A mix of arbitrary F16 finite blocks and exact E4M3 block payloads.
        for _ in range(24):
            blocks.append([rng.randrange(0x7c00) | (rng.randrange(2)<<15) for _ in range(1024)])
        for exp in (-24,-16,-8,0,7):
            hs = []
            for _ in range(1024):
                c = rng.randrange(0x7f) | (rng.randrange(2)<<7)
                h = ((c&0x80)<<8) | ((c&0x7f)<<7)
                v = struct.unpack("<e",struct.pack("<H",h))[0]*2.0**(exp+8)
                hs.append(struct.unpack("<H",struct.pack("<e",v))[0])
            blocks.append(hs)
        for block in blocks:
            raw = struct.pack("<1024H",*block)
            src,codes,scales,restored,valid,s = self.projection_worker(raw)
            ref_codes, ref_scale = ctypes.create_string_buffer(1024),ctypes.create_string_buffer(4)
            ok = self.reference.reference_block(src,ref_codes,ref_scale)
            self.assertEqual(bool(valid.raw[0]),bool(ok))
            self.assertEqual(s.failed_blocks,0 if ok else 1)
            if ok:
                self.assertEqual(codes.raw,ref_codes.raw)
                self.assertEqual(scales.raw,ref_scale.raw)
                self.assertEqual(restored.raw,raw)

    def test_failure_does_not_stop_remaining_blocks_or_stripes(self):
        hs = [0x3c00]*(64*64)
        hs[4] = 0x3c01
        hs[32*64+40] = 0x3c01
        r = self.audit_raw(struct.pack("<4096H",*hs),"engram",64)
        self.assertFalse(r["qualification_passed"])
        self.assertEqual(r["counts"]["blocks"],4)
        self.assertEqual(r["counts"]["failed_blocks"],2)
        self.assertEqual(r["counts"]["values_without_accepted_block"],2048)
        self.assertEqual(r["counts"]["unencodable_at_all_scales"],2)
        self.assertEqual(r["counts"]["under_witness_scale_mismatch_count"],2)
        self.assertEqual(r["valid_scales"]["count"],2)
        self.assertEqual(r["source_bytes_scanned"],8192)
        self.assertFalse(r["streams_form_valid_complete_representation"])
        self.assertEqual(r["first_failure"]["index"],4)

    def test_individual_values_can_fit_but_no_single_block_scale_exists(self):
        # 448 needs e>=0; 2^-10 needs e<=-1. Neither value is intrinsically
        # unencodable, yet the common 32x32 block cannot be reconstructed.
        hs = [struct.unpack("<H",struct.pack("<e",448.0))[0]]*1024
        hs[17] = struct.unpack("<H",struct.pack("<e",2.0**-10))[0]
        r = self.audit_raw(struct.pack("<1024H",*hs),"engram")
        self.assertFalse(r["qualification_passed"])
        self.assertEqual(r["counts"]["failed_blocks"],1)
        self.assertEqual(r["counts"]["values_without_accepted_block"],1024)
        self.assertEqual(r["counts"]["unencodable_at_all_scales"],0)
        self.assertEqual(r["counts"]["under_witness_scale_mismatch_count"],1)
        self.assertEqual(r["first_failure"]["index"],17)
        self.assertEqual(r["first_failure"]["kind"],"block_has_no_common_scale_despite_individual_fits")

    def test_nonfinite_projection_rejects_before_undefined_scale_conversion(self):
        for h in (0x7c00,0xfc00,0x7e00,0x7c01,0xfe01):
            with self.subTest(bits=hex(h)):
                hs = [0x3c00]*1024
                hs[17] = h
                r = self.audit_raw(struct.pack("<1024H",*hs),"index_qb")
                self.assertFalse(r["qualification_passed"])
                self.assertEqual(r["counts"]["failed_blocks"],1)
                self.assertEqual(r["counts"]["blocks_without_finite_witness_scale"],1)
                self.assertEqual(r["counts"]["failed_scale_trials"],0)
                self.assertEqual(r["first_failure"]["index"],17)

    def test_worker_span_rejection(self):
        b = ctypes.create_string_buffer(4096)
        s = A.Stats()
        self.assertEqual(self.lib.v41_audit_e4m3(b,31,b,b,b,b,ctypes.byref(s)),-1)
        self.assertEqual(self.lib.v41_audit_e4m3(None,32,b,b,b,b,ctypes.byref(s)),-1)

    def test_gguf_parse_target_type_and_source_spans(self):
        p = self.dir/"x.gguf"
        make_gguf(p,[("x",[32,32],1,b"\0"*2048)])
        model = A.open_model(p)
        try:
            A.validate_scope(model)
            t = model["tensors"]["x"]
            self.assertEqual(t["bytes"],2048)
            self.assertEqual(t["reader"].read(t["file_offset"],4),b"\0"*4)
            self.assertTrue(model["readers"][0].finish_identity()[0]["unchanged"])
        finally:
            for f in model["readers"]: f.close()

    def test_truncated_tensor_and_duplicate_names_rejected(self):
        p = self.dir/"bad.gguf"
        make_gguf(p,[("x",[32,32],1,b"\0"*2048)])
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(A.AuditError): A.open_model(p)
        make_gguf(p,[("x",[32],1,b"\0"*64),("x",[32],1,b"\0"*64)])
        with self.assertRaises(A.AuditError): A.open_model(p)

    def test_target_shape_scope_and_missing_tensor_cannot_pass(self):
        p, out = self.dir/"empty.gguf",self.dir/"audit.json"
        make_gguf(p)
        before = hashlib.sha256(p.read_bytes()).hexdigest()
        proc = subprocess.run([sys.executable,str(ROOT/"tools/audit_v41_narrow_weights.py"),"--model",str(p),"--output",str(out)],capture_output=True,text=True)
        self.assertEqual(proc.returncode,1,proc.stderr)
        r = json.loads(out.read_text())
        self.assertEqual(len(r["expected_targets"]),50)
        self.assertEqual(len(r["tensors"]),50)
        self.assertEqual(r["families"]["router"]["tensor_count"],40)
        self.assertFalse(r["qualification_passed"])
        self.assertTrue(r["audit_completed"])
        self.assertFalse(r["GPU_work_submitted"])
        self.assertEqual(before,hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertTrue(Path(r["worker"]["binary"]["path"]).is_file())
        manifest = json.loads(Path(r["artifact_manifest"]["path"]).read_text())
        for item in manifest["files"]:
            artifact = Path(r["artifact_directory"])/item["relative_path"]
            self.assertEqual(item["sha256"],hashlib.sha256(artifact.read_bytes()).hexdigest())
        self.assertIn("compile-command.json",[f["relative_path"] for f in manifest["files"]])

    def test_wrong_scope_and_unexpected_family_tensor_rejected(self):
        p = self.dir/"wrong.gguf"
        kv = metadata(); kv["deepseek41.engram.layer_ids"] = (9,(4,[1,15]))
        make_gguf(p,kv=kv)
        model = A.open_model(p)
        try:
            with self.assertRaises(A.AuditError): A.validate_scope(model)
        finally:
            for f in model["readers"]: f.close()
        make_gguf(p,[("blk.3.indexer.attn_q_b.weight",[32],1,b"\0"*64)])
        model = A.open_model(p)
        try:
            with self.assertRaises(A.AuditError): A.validate_scope(model)
        finally:
            for f in model["readers"]: f.close()

    def test_actual_native_header_names_bind_and_reach_tensor_gates(self):
        p, out = self.dir/"native-header.gguf", self.dir/"native-header.json"
        make_gguf(p,kv=native_header_metadata())
        proc = subprocess.run([sys.executable,str(ROOT/"tools/audit_v41_narrow_weights.py"),"--model",str(p),"--output",str(out)],capture_output=True,text=True)
        # No payloads in this fixture: metadata passes, then all missing tensor
        # gates reject. This must be exit 1, unlike the v1 metadata exit 2.
        self.assertEqual(proc.returncode,1,proc.stderr)
        report = json.loads(out.read_text())
        self.assertTrue(report["audit_completed"])
        self.assertFalse(report["qualification_passed"])
        binding = report["scope_binding"]
        self.assertEqual(binding["layers"]["selected_key"],"deepseek41.num_hidden_layers")
        self.assertEqual(binding["width"]["selected_key"],"deepseek41.hidden_size")
        self.assertEqual(binding["layers"]["selected_gguf_type"],4)
        self.assertEqual(binding["width"]["selected_gguf_type"],4)
        self.assertEqual(binding["layers"]["present_aliases"][0]["metadata_value_sha256"],hashlib.sha256(struct.pack("<I",40)).hexdigest())
        self.assertEqual(len(report["tensors"]),50)

    def test_dimension_aliases_agree_and_primary_keys_are_recorded(self):
        p = self.dir/"aliases.gguf"
        kv = metadata() | native_header_metadata()
        make_gguf(p,kv=kv)
        model = A.open_model(p)
        try:
            binding = A.validate_scope(model)
            self.assertEqual(binding["layers"]["selected_key"],"deepseek41.block_count")
            self.assertEqual(binding["width"]["selected_key"],"deepseek41.embedding_length")
            self.assertEqual(len(binding["layers"]["present_aliases"]),2)
            self.assertEqual(len(binding["width"]["present_aliases"]),2)
        finally:
            for f in model["readers"]: f.close()

    def test_dimension_alias_conflicts_and_wrong_values_types_rejected(self):
        p = self.dir/"bad-aliases.gguf"
        variants = [
            (metadata() | {"deepseek41.num_hidden_layers":(4,41)},"conflicting"),
            (metadata() | {"deepseek41.hidden_size":(4,4096)},"conflicting"),
            (native_header_metadata() | {"deepseek41.num_hidden_layers":(4,39)},"must equal 40"),
            (native_header_metadata() | {"deepseek41.hidden_size":(4,4096)},"must equal 5120"),
            (native_header_metadata() | {"deepseek41.num_hidden_layers":(6,40.0)},"integer type"),
            (native_header_metadata() | {"deepseek41.num_hidden_layers":(10,2**32+40)},"must equal 40"),
        ]
        for kv,reason in variants:
            with self.subTest(reason=reason,kv=kv):
                make_gguf(p,kv=kv)
                model = A.open_model(p)
                try:
                    with self.assertRaisesRegex(A.AuditError,reason): A.validate_scope(model)
                finally:
                    for f in model["readers"]: f.close()

    def test_architecture_is_exact_deepseek41_not_prefix(self):
        p = self.dir/"bad-architecture.gguf"
        for architecture in ("deepseek4","deepseek41-other"):
            make_gguf(p,kv=native_header_metadata() | {"general.architecture":(8,architecture)})
            model = A.open_model(p)
            try:
                with self.assertRaisesRegex(A.AuditError,"exactly deepseek41"): A.validate_scope(model)
            finally:
                for f in model["readers"]: f.close()
    def test_tensor_split_identity_missing_and_duplicate(self):
        paths = [self.dir/f"x-{i:05d}-of-00002.gguf" for i in (1,2)]
        for i,p in enumerate(paths):
            kv = metadata() | {"split.no":(2,i),"split.count":(2,2),"split.tensors.count":(10,2)}
            make_gguf(p,[(f"x{i}",[32],1,b"\0"*64)],kv)
        model = A.open_model(paths[1])
        for f in model["readers"]: f.close()
        raw = paths[1].read_bytes()
        kv = metadata() | {"split.no":(2,1),"split.count":(2,2),"split.tensors.count":(10,2)}
        kv["deepseek41.block_count"] = (4,41)
        make_gguf(paths[1],[("x1",[32],1,b"\0"*64)],kv)
        with self.assertRaises(A.AuditError): A.open_model(paths[0])
        paths[1].write_bytes(raw)
        paths[1].unlink()
        with self.assertRaises(OSError): A.open_model(paths[0])

    def test_byte_split_cross_part_read_and_manifest_gates(self):
        p = self.dir/"raw.gguf"
        chunk = os.sysconf("SC_PAGE_SIZE")
        payload = bytes(range(256))*(2*chunk//256)
        make_gguf(p,[("x",[32,len(payload)//64],1,payload)])
        raw, base = p.read_bytes(), self.dir/"split"
        n = (len(raw)+chunk-1)//chunk
        self.assertGreater(n,1)
        for i in range(n):
            Path(f"{base}.part{i+1:02d}").write_bytes(raw[i*chunk:(i+1)*chunk])
        manifest = Path(str(base)+".parts")
        manifest.write_text(f"myds-split 1\nparts {n}\nchunk {chunk}\ntotal {len(raw)}\n")
        model = A.open_model(str(base)+".part01")
        try:
            t = model["tensors"]["x"]
            self.assertEqual(t["reader"].read(t["file_offset"],t["bytes"]),payload)
        finally:
            for f in model["readers"]: f.close()
        manifest.write_text(manifest.read_text()+"parts 2\n")
        with self.assertRaises(A.AuditError): A.open_model(base)

    def test_output_input_hardlink_symlink_alias_refused(self):
        p = self.dir/"model.gguf"
        make_gguf(p)
        model = A.open_model(p)
        try:
            hard, sym = self.dir/"hard.json",self.dir/"sym.json"
            os.link(p,hard); sym.symlink_to(p)
            for out in (p,hard,sym):
                with self.assertRaises(A.AuditError): A.refuse_output_alias(out,model)
        finally:
            for f in model["readers"]: f.close()

    def test_changed_or_replaced_file_is_identity_failure(self):
        p = self.dir/"model.gguf"; p.write_bytes(b"x"*64)
        f = A.LogicalFile([p])
        try:
            replacement = self.dir/"new"; replacement.write_bytes(b"x"*64); os.replace(replacement,p)
            self.assertFalse(f.finish_identity()[0]["unchanged"])
        finally:
            f.close()

    def test_upstream_patch_tamper_rejected(self):
        for commit in A.UPSTREAM:
            (self.dir/(commit+".patch")).write_bytes((PATCHES/(commit+".patch")).read_bytes()+b"\n")
        with self.assertRaises(A.AuditError): A.build_worker(self.dir,"cc",self.dir)

    def test_cli_wrong_actual_type_and_shape_recorded(self):
        p, out = self.dir/"types.gguf",self.dir/"report.json"
        raw = b"\0"*(5120*384*2)
        make_gguf(p,[("blk.0.ffn_gate_inp.weight",[5120,384],30,raw),
                     ("blk.2.indexer.attn_q_b.weight",[32,32],1,b"\0"*2048)])
        proc = subprocess.run([sys.executable,str(ROOT/"tools/audit_v41_narrow_weights.py"),"--model",str(p),"--output",str(out)],capture_output=True,text=True)
        self.assertEqual(proc.returncode,1,proc.stderr)
        r = json.loads(out.read_text())
        router = r["tensors"][0]
        index = next(t for t in r["tensors"] if t["family"]=="index_qb")
        self.assertEqual(router["actual_type_name"],"BF16")
        self.assertEqual(router["source_sha256"],hashlib.sha256(raw).hexdigest())
        self.assertIn("source_type",router["gate_error"])
        self.assertIn("shape_mismatch",index["gate_error"])
        self.assertNotIn("source_sha256",index)

    def test_cli_malformed_input_preserves_error_json(self):
        p, out = self.dir/"bad.gguf", self.dir/"failure.json"
        p.write_bytes(b"GGUF\x03")
        proc = subprocess.run([sys.executable,str(ROOT/"tools/audit_v41_narrow_weights.py"),"--model",str(p),"--output",str(out)],capture_output=True,text=True)
        self.assertEqual(proc.returncode,2,proc.stderr)
        r = json.loads(out.read_text())
        self.assertFalse(r["audit_completed"])
        self.assertTrue(r["errors"])
        self.assertEqual(p.read_bytes(),b"GGUF\x03")

    def test_retained_artifact_directory_cannot_be_reused_or_contain_report(self):
        p, out, artifact = self.dir/"empty.gguf", self.dir/"report.json", self.dir/"artifacts"
        make_gguf(p)
        artifact.mkdir(); marker=artifact/"keep"; marker.write_bytes(b"unchanged")
        command = [sys.executable,str(ROOT/"tools/audit_v41_narrow_weights.py"),"--model",str(p),"--output",str(out),"--artifact-dir",str(artifact)]
        proc = subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(proc.returncode,2,proc.stderr)
        self.assertEqual(marker.read_bytes(),b"unchanged")
        self.assertEqual(len(list(artifact.iterdir())),1)
        inside = self.dir/"new-artifacts"/"worker.dylib"
        command[command.index(str(out))] = str(inside)
        command[command.index(str(artifact))] = str(inside.parent)
        proc = subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(proc.returncode,2,proc.stderr)
        self.assertFalse(inside.parent.exists())


if __name__ == "__main__":
    unittest.main()
