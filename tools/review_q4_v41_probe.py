#!/usr/bin/env python3
"""CPU-only full-payload and A-B-A audit for bench_q4_v41. No Metal/HTTP."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

SHAPES={"head":(129280,5120,"f32"),"qb":(32768,1280,"bf16"),"ob":(5120,8192,"bf16")}
ARGS=struct.Struct("<iii4xQQQQiii4xQQQQiiihh")
assert ARGS.size==112
def native_sample_finite(value):
    # @(isfinite(us) && us > 0) boxes a C int on the measured ObjC ABI.
    # Accept exactly JSON true or integer 1, not truthy strings/floats/2.
    return value is True or (type(value) is int and value==1)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while chunk:=f.read(8<<20):h.update(chunk)
    return h.hexdigest()
def audit(raw,stdout):
    m=json.loads((raw/"artifact-manifest.json").read_text())
    assert m["status"]=="complete" and m["all_output_gates_passed"] and m["input_guards_passed"]
    rows,width,epilogue=SHAPES[m["shape"]]
    assert (m["rows"],m["width"],m["epilogue"],m["nsg"],m["nr0"],m["dynamic_tg_bytes"])==(rows,width,epilogue,4,2,256)
    rotations,seeds,samples,reps=(m[k] for k in ("rotations","seeds","samples","reps"))
    assert 1<=rotations<=16 and 1<=seeds<=32 and 3<=samples<=31 and 1<=reps<=64
    assert m["cold_dram_proven"] is False and m["fullmodel_speedup_proven"] is False
    rowbytes=width//256*144;wbytes=rows*rowbytes;obytes=rows*4;ibytes=width*4
    a=ARGS.unpack(bytes.fromhex(m["production_args_hex"].replace("<","").replace(">","").replace(" ","").replace("\n","")))
    assert a==(width,rows,1,144,rowbytes,wbytes,wbytes,width,1,1,4,ibytes,ibytes,ibytes,rows,1,2,1,1)
    expected_names=[f"kernel_mul_mv_q4_K_dense_f32"+("_bf16" if epilogue=="bf16" else "")]*2+[f"kernel_myds_q4_v41_{m['shape']}_addr_{epilogue}",f"kernel_myds_q4_v41_{m['shape']}_nb_{epilogue}"]
    assert len(m["psos"])==4
    for i,p in enumerate(m["psos"]):
        assert p["kernel"]==expected_names[i] and p["index"]==i and p["production_library"]==(i==0)
        assert p["NSG600"]==4 and p["max_threads"]>=128
    inventory={}
    total=0
    for x in m["files"]:
        name=x["path"];assert Path(name).name==name and name not in inventory
        p=raw/name;assert p.is_file() and p.stat().st_size==x["bytes"] and digest(p)==x["sha256"]
        inventory[name]=x;total+=x["bytes"]
    assert set(p.name for p in raw.iterdir())==set(inventory)|{"artifact-manifest.json"}
    def read(name):assert name in inventory;return (raw/name).read_bytes()
    def output(name):
        b=read(name);assert len(b)==obytes+128 and b[:64]==b[-64:]==bytes([0xa5])*64
        payload=b[64:-64]
        assert all(math.isfinite(v[0]) for v in struct.iter_unpack("<f",payload))
        assert all(u[0]!=0xa5a5a5a5 for u in struct.iter_unpack("<I",payload))
        if epilogue=="bf16":assert all((u[0]&0xffff)==0 for u in struct.iter_unpack("<I",payload))
        return payload
    assert len(m["weights"])==rotations
    for slot,w in enumerate(m["weights"]):
        assert w["slot"]==slot and w["bytes"]==wbytes and w["sha256"]==inventory[w["raw_file"]]["sha256"]
        assert inventory[w["raw_file"]]["bytes"]==wbytes
        assert read(f"weight-slot{slot}-guards.bin")==bytes([0xa5])*128
        b=read(f"final-slot{slot}-input.guarded")
        assert len(b)==ibytes+128 and b[:64]==b[-64:]==bytes([0xa5])*64
        assert b[64:-64]==read(f"seed{seeds-1}-slot{slot}-input.f32")
    expected_parity=set();expected_timing=set();expected_summaries=set()
    for seed in range(seeds):
        for slot in range(rotations):
            x=read(f"seed{seed}-slot{slot}-input.f32");assert len(x)==ibytes and all(math.isfinite(v[0]) for v in struct.iter_unpack("<f",x))
            baseline=output(f"seed{seed}-slot{slot}-k0-output.guarded")
            for k in range(4):
                assert output(f"seed{seed}-slot{slot}-k{k}-output.guarded")==baseline
                expected_parity.add((seed,slot,k))
    records=[json.loads(s) for s in stdout.read_text().splitlines() if s.startswith("{")]
    identity=[r for r in records if r.get("kind")=="identity"]
    assert len(identity)==1 and identity[0]["psos"]==m["psos"] and identity[0]["weights"]==m["weights"]
    seen=set()
    for r in records:
        if r.get("kind")=="parity":
            key=(r["seed"],r["slot"],r["pso"]);assert key in expected_parity and key not in seen;seen.add(key)
            assert r["finite"] and r["guards"] and r["full_output_same"] and r["poison_free"] is True and r["production_golden"]==(key[2]==0)
            assert r["sha256"]==hashlib.sha256(output(f"seed{key[0]}-slot{key[1]}-k{key[2]}-output.guarded")).hexdigest()
    assert seen==expected_parity
    timings={}
    for candidate in (2,3):
        for scheme in ("hot","rotating"):
            expected_summaries.add((candidate,scheme))
            active=1 if scheme=="hot" else rotations
            for arm in ("A1","B","A2"):
                for sample in range(samples+1):
                    key=(candidate,scheme,arm,sample);expected_timing.add(key)
                    for slot in range(active):
                        assert output(f"timing-c{candidate}-{scheme}-{arm}-s{sample}-slot{slot}-output.guarded")==output(f"seed{seeds-1}-slot{slot}-k0-output.guarded")
    for r in records:
        if r.get("kind")!="timing":continue
        key=(r["candidate"],r["scheme"],r["arm"],r["sample"]);assert key in expected_timing and key not in timings
        assert r["warmup"]==(r["sample"]==0)
        assert all(r[k] is True for k in ("full_output_same","guards","finite","poison_free"))
        assert native_sample_finite(r["sample_finite"])
        assert r["pso"]==(r["candidate"] if r["arm"]=="B" else 0) and r["production_control"]==(r["arm"]!="B")
        active=1 if r["scheme"]=="hot" else rotations
        assert r["dispatches"]==active*reps
        start,end=r["gpu_start_s"],r["gpu_end_s"];assert math.isfinite(start) and math.isfinite(end) and end>start
        us=(end-start)*1e6/r["dispatches"];assert math.isclose(us,r["gpu_us"],rel_tol=1e-12,abs_tol=1e-9)
        timings[key]=us
    assert set(timings)==expected_timing
    summaries=[];seen_summary=set()
    for r in records:
        if r.get("kind")!="summary":continue
        key=(r["candidate"],r["scheme"]);assert key in expected_summaries and key not in seen_summary;seen_summary.add(key)
        med=[statistics.median(timings[(*key,arm,s)] for s in range(1,samples+1)) for arm in ("A1","B","A2")]
        saved=(1-med[1]/((med[0]+med[2])/2))*100;drift=(med[2]/med[0]-1)*100
        for k,v in zip(("a1_us","b_us","a2_us","time_saved_pct","control_drift_pct"),(*med,saved,drift)):
            assert math.isclose(r[k],v,rel_tol=1e-12,abs_tol=1e-9)
        assert r["fullmodel_speedup_proven"] is False
        summaries.append({"candidate":key[0],"scheme":key[1],"median_gpu_us":med,"time_saved_pct":saved,"control_drift_pct":drift,
                          "faster_than_both_controls":med[1]<min(med[0],med[2])})
    assert seen_summary==expected_summaries
    complete=[r for r in records if r.get("kind")=="complete"];assert len(complete)==1 and complete[0]["pass"] and complete[0]["raw_files"]==len(inventory)
    return {"pass":True,"shape":m["shape"],"source":"synthetic" if m["synthetic_weights"] else "real_GGUF",
            "all_manifest_sha256_recomputed":True,"full_production_clone_candidate_bytes_equal":True,
            "output_and_input_guards_recomputed":True,"timing_raw_timestamp_medians_recomputed":True,
            "raw_files":len(inventory),"raw_bytes":total,"parity_records":len(seen),"timing_records":len(timings),
            "summaries":summaries,"fullmodel_speedup_proven":False,"cold_dram_proven":False,
            "reviewer_sha256":digest(Path(__file__)),"manifest_sha256":digest(raw/"artifact-manifest.json"),"stdout_sha256":digest(stdout)}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--stdout",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists(),"output already exists"
    result=audit(a.raw,a.stdout)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"pass":True,"report":str(a.out),"shape":result["shape"],"raw_files":result["raw_files"]}))
if __name__=="__main__":main()

