#!/usr/bin/env python3
"""CPU-only source/ABI/coverage/failure fixtures; no Metal device or GPU work."""
import contextlib
from copy import deepcopy
import ctypes
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import build_indexer_register_only_probe as builder

NATIVE = (ROOT/"metal/dsv4_misc.metal").read_text()
CANDIDATE = (ROOT/"tools/indexer_register_only.metal.inc").read_text()


def problem(**changes):
    args=dict(n_comp=543,n_tokens=9,n_head=32,head_dim=128,pos0=0,ratio=2,
              q_token_stride=16384,q_head_stride=512,weights_token_stride=128,
              index_row_stride=256,score_token_stride=543*4,scale=1/64)
    args.update(changes)
    return args


def fake_source(root):
    (root/"metal").mkdir(parents=True);(root/"tools").mkdir()
    (root/"tools/build_metallib.py").write_text('SHADER_FILES=["first.metal","dsv4_misc.metal"]\nPREAMBLE="//PREAMBLE"\nCOPY_KERNELS="//COPY"\nif __name__=="__main__":\n raise AssertionError("production main must not execute")\n')
    (root/"metal/first.metal").write_bytes(b"// first shader exactly, with no final LF")
    args=builder.extract_block(NATIVE,"struct "+builder.ARG_TYPE)+";\n"
    template=builder.extract_block(NATIVE,"template<bool BF16, bool WARP_ONLY = false>\nkernel void kernel_dsv4_indexer_scores_tiled_storage")
    (root/"metal/dsv4_misc.metal").write_text(args+template+"\n")


class RegisterOnlyBuilderTests(unittest.TestCase):
    def test_actual_ABI_offsets_and_size_independent_ctypes(self):
        class Args(ctypes.Structure):
            _fields_=[("n_comp",ctypes.c_uint32),("n_tokens",ctypes.c_uint32),("n_head",ctypes.c_uint32),
                ("head_dim",ctypes.c_uint32),("pos0",ctypes.c_uint32),("ratio",ctypes.c_uint32),
                ("q_token_stride",ctypes.c_uint64),("q_head_stride",ctypes.c_uint64),("weights_token_stride",ctypes.c_uint64),
                ("index_row_stride",ctypes.c_uint64),("score_token_stride",ctypes.c_uint64),("scale",ctypes.c_float)]
        proof=builder.validate_sources(NATIVE,CANDIDATE)
        self.assertEqual(ctypes.sizeof(Args),72);self.assertEqual(ctypes.alignment(Args),8)
        self.assertEqual({n:getattr(Args,n).offset for n,_ in Args._fields_},{r["name"]:r["offset"]for r in proof["abi"]["fields"]})
        args=problem();packed=bytes(Args(**args));self.assertEqual(packed,struct.pack("<6I5Qf4x",*(args[n]for n,_ in Args._fields_)))
        # Reinterpreting these same 72B as old GLM was the previous plan's trap.
        old=struct.unpack("<8I4Qf4x",packed)
        self.assertEqual(old[6],16384);self.assertEqual(old[8],512)

    def test_coordinate_and_Q_K_fragment_address_bijections(self):
        cells={(y,x+p)for lane in range(32)for x,y in [builder.coord(lane)]for p in (0,1)}
        self.assertEqual(cells,{(r,c)for r in range(8)for c in range(8)})
        q={(y,db*8+x+p)for db in range(16)for lane in range(32)for x,y in [builder.coord(lane)]for p in (0,1)}
        k={(x+p,db*8+y)for db in range(16)for lane in range(32)for x,y in [builder.coord(lane)]for p in (0,1)}
        expected={(r,d)for r in range(8)for d in range(128)}
        self.assertEqual(q,expected);self.assertEqual(k,expected)
        self.assertEqual(len(q),1024);self.assertEqual(len(k),1024)
        for lane in (-1,32,True):
            with self.assertRaises(ValueError):builder.coord(lane)

    def test_complete_dense_store_coverage_invisible_tiles_and_both_tails(self):
        # Independently enumerate every lane's two logical output cells. This
        # checks coverage/masks, not numerical MMA results or backend layout.
        for tokens,keys,pos in ((9,543,0),(15,1025,1024),(8,16383,32760)):
            with self.subTest(tokens=tokens,keys=keys):
                stores={};whole_invisible=0
                for gy in range((tokens+7)//8):
                    maximum=min((pos+min(gy*8+8,tokens))//2,keys)
                    for gx in range((keys+31)//32):
                        for sg in range(4):
                            base=gx*32+sg*8
                            for lane in range(32):
                                x,y=builder.coord(lane);token=gy*8+y
                                for key in (base+x,base+x+1):
                                    if token>=tokens or key>=keys:continue
                                    self.assertNotIn((token,key),stores)
                                    visible=min((pos+token+1)//2,keys)
                                    stores[token,key]=key<visible
                                    if base>=maximum:whole_invisible+=1;self.assertFalse(stores[token,key])
                self.assertEqual(len(stores),tokens*keys)
                if pos==0:self.assertGreater(whole_invisible,tokens*keys//2)

    def test_scope_boundaries_geometry_and_required_padded_strides(self):
        args=problem(q_head_stride=520,q_token_stride=31*520+512+12,weights_token_stride=140,index_row_stride=258,score_token_stride=543*4+20)
        result=builder.validate_problem(args)
        self.assertEqual(result["grid"],[17,2,1]);self.assertEqual(result["threads"],[32,4,1]);self.assertEqual(result["dynamic_threadgroup_bytes"],0)
        spans=result["required_buffer_bytes"]
        self.assertEqual(spans["q"],8*args["q_token_stride"]+31*520+512)
        self.assertEqual(spans["cache"],542*258+256)
        builder.validate_problem(args,spans)
        for key in spans:
            bad=dict(spans);bad[key]-=1
            with self.subTest(key=key),self.assertRaises(ValueError):builder.validate_problem(args,bad)
        builder.validate_problem(problem(n_comp=16384,n_tokens=8,pos0=32760,score_token_stride=65536))
        builder.validate_problem(problem(n_comp=513,n_tokens=2048,score_token_stride=2052))

    def test_invalid_scope_or_stride_and_uint64_span_overflow_rejected(self):
        changes=[dict(n_head=64),dict(head_dim=64),dict(ratio=1),dict(n_comp=512),dict(n_comp=16385),dict(n_tokens=7),dict(n_tokens=2049),
                 dict(pos0=32761,n_tokens=8),dict(scale=1/32),dict(n_comp=True),dict(index_row_stride=257),dict(q_head_stride=508),
                 dict(q_head_stride=516,q_token_stride=16384),dict(weights_token_stride=124),dict(score_token_stride=542*4),dict(q_token_stride=(1<<64)-4)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):builder.validate_problem(problem(**change))

    def test_missing_invisible_fill_old_math_and_TG_reintroduction_rejected(self):
        for old,new in [("} else if (live) {","} else if (false) {"),
                        ("max(s, 0.0f) * (w[head] * args.scale)","max(s * args.scale, 0.0f) * w[head]"),
                        ("uint(row[d]) << 16u","uint(row[d])"),
                        ("float acc0 = 0.0f, acc1 = 0.0f;","threadgroup_barrier(mem_flags::mem_threadgroup); float acc0 = 0.0f, acc1 = 0.0f;")]:
            with self.subTest(old=old),self.assertRaises(ValueError):builder.validate_sources(NATIVE,CANDIDATE.replace(old,new))
        bad=NATIVE.replace("uint64_t index_row_stride;","uint64_t wrong_row_stride;",1)
        with self.assertRaises(ValueError):builder.validate_sources(bad,CANDIDATE)

    def test_complete_native_production_assembly_and_append_inverse_exact(self):
        build,raw=builder.load_production_builder(ROOT)
        native,records,payloads=builder.assemble_native(ROOT,build)
        expected=(build.PREAMBLE+"\n"+"".join('\n#line 1 "'+name+'"\n'+(ROOT/"metal"/name).read_text()for name in build.SHADER_FILES)+build.COPY_KERNELS+"\n").encode()
        self.assertEqual(native,expected);self.assertEqual(len(records),29)
        combined,proof=builder.compose(native,CANDIDATE)
        self.assertEqual(combined[:proof["native_bytes"]],native)
        self.assertEqual(combined[proof["candidate_source_begin"]:],CANDIDATE.encode())
        self.assertEqual(proof["native_sha256"],builder.sha(native))
        self.assertEqual([r["path"]for r in records],["metal/"+name for name in build.SHADER_FILES])
        for r in records:self.assertEqual(native[r["source_begin"]:r["source_end"]],payloads[r["path"]])
        with self.assertRaises(ValueError):builder.compose(native+builder.SYMBOL.encode(),CANDIDATE)

    def test_duplicate_unsafe_shader_and_CRLF_refused_without_normalization(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);fake_source(root);build,_=builder.load_production_builder(root)
            for names in (["first.metal","first.metal"],["../escape.metal"],["/absolute.metal"]):
                with self.subTest(names=names),self.assertRaises(ValueError):builder.assemble_native(root,SimpleNamespace(SHADER_FILES=names,PREAMBLE="",COPY_KERNELS=""))
            (root/"metal/first.metal").write_bytes(b"//CRLF\r\n")
            with self.assertRaises(ValueError):builder.assemble_native(root,build)

    def test_emit_only_snapshots_external_fragment_no_compiler_or_overwrite(self):
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);root=base/"production";fake_source(root);out=base/"emit"
            argv=["builder","--source-root",str(root),"--candidate-source",str(ROOT/"tools/indexer_register_only.metal.inc"),"--out-dir",str(out),"--emit-only"]
            with patch.object(sys,"argv",argv),patch.object(builder.subprocess,"run",side_effect=AssertionError("emit-only invoked compiler")),contextlib.redirect_stdout(io.StringIO()):builder.main()
            identity=json.loads((out/"source-identity.json").read_text());manifest=json.loads((out/"build-manifest.json").read_text())
            self.assertTrue(manifest["compilation_skipped"]);self.assertFalse(manifest["compile_only_pass"]);self.assertFalse(manifest["gpu_work_submitted"])
            self.assertEqual(manifest["selected_symbol"],builder.SYMBOL);self.assertEqual(len(identity["commands"]),4)
            self.assertEqual((out/"source/tools/indexer_register_only.metal.inc").read_text(),CANDIDATE)
            for name,item in identity["source_inventory"].items():self.assertEqual(builder.sha((out/"source"/name).read_bytes()),item["sha256"])
            original=(out/"all.metal").read_bytes()
            with patch.object(sys,"argv",argv),self.assertRaises(ValueError):builder.main()
            self.assertEqual((out/"all.metal").read_bytes(),original)

    def test_production_compile_flags_defaults_and_explicit_inputs_frozen(self):
        a=SimpleNamespace(metal_minver=None,metal_std=None)
        options,proof=builder.compile_options(a,{})
        self.assertEqual(options,["-mmacosx-version-min=15.0","-O3","-ffast-math","-DDS4_METAL_HC_STABLE=1","-DDS4_METAL_NORM_RSQRT_DISABLE=1"])
        options,proof=builder.compile_options(a,{"MYDS_METAL_MINVER":"26.0","MYDS_METAL_STD":"metal4.0","MYDS_METALLIB_DEFINES":"-DMYDS_PROBE=1"})
        self.assertIn("-std=metal4.0",options);self.assertEqual(proof["extra_flags"],["-DMYDS_PROBE=1"])
        with self.assertRaises(ValueError):builder.compile_options(SimpleNamespace(metal_minver="15.0;bad",metal_std=None),{})


if __name__=="__main__":unittest.main()
