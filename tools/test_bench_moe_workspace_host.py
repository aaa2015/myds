#!/usr/bin/env python3
"""Synthetic sparse GGUF parser controls. Child is always --host-only.

No model loader, GPU initialization or inference is requested. The sparse file
only supplies nine correctly sized tensor descriptors; host phase deliberately
does not read/hash its unmaterialized weight intervals. This is never real-model
or performance evidence. A separately CPU-built harness is required.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def string(s):
    b=s.encode();return struct.pack('<Q',len(b))+b


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bench',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();bench=a.bench.resolve(strict=True);controls=[]
    with tempfile.TemporaryDirectory(prefix='moe-workspace-CPU-only-') as temporary:
        root=Path(temporary).resolve();model=root/'synthetic-never-inferred.gguf';capture=root/'capture.json'
        tensors=[];offset=0
        for layer in (0,19,39):
            for kind in ('gate','up','down'):
                down=kind=='down';dims=(2304,5120,384) if down else (5120,2304,384)
                size=(dims[0]//256)*(84 if down else 66)*dims[1]*dims[2]
                tensors.append((f'blk.{layer}.ffn_{kind}_exps.weight',dims,10 if down else 16,offset));offset+=size
        metadata=[('general.architecture',8,string('deepseek4')),('deepseek4.swiglu_limit',6,struct.pack('<f',10.))]
        header=struct.pack('<IIQQ',0x46554747,3,len(tensors),len(metadata))
        for key,t,value in metadata:header+=string(key)+struct.pack('<I',t)+value
        for name,dims,t,off in tensors:header+=string(name)+struct.pack('<I',3)+struct.pack('<3Q',*dims)+struct.pack('<IQ',t,off)
        data_offset=(len(header)+31)//32*32
        with model.open('wb') as f:f.write(header);f.write(b'\0'*(data_offset-len(header)));f.truncate(data_offset+offset)
        def payload(name,data):
            p=root/name;p.write_bytes(data);return dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        cases=[]
        for index,layer in enumerate((0,19,39)):
            rows=33;ids=struct.pack('<6i',*range(6))*rows;weights=struct.pack('<6f',*([1/6]*6))*rows
            cases.append(dict(case=index,layer=layer,absolute_start=0,rows=rows,capacity=2048,moe_batch_capacity=16384,
                clamp=10.,actual_mm=True,all_finite=True,trace_on=False,observer_batch_restored=True,
                input_f32=payload(f'input-{index}.bin',b'\0'*(rows*20480)),ids_i32=payload(f'ids-{index}.bin',ids),
                routeweights_f32=payload(f'weights-{index}.bin',weights),native_output_f32=payload(f'output-{index}.bin',b'\0'*(rows*20480))))
        with model.open('rb') as f:prefix=hashlib.sha256(f.read(1048576)).hexdigest()
        original=dict(schema_version=1,capture_kind='myds_prefill_moe_workspace',model_bytes=model.stat().st_size,
            model_header_bytes=1048576,model_header_sha256=prefix,
            source=dict(kind='actual_native_prefill_routed_moe_batch',actual_tokens=16401,trace_on=False,performance_from_capture=False),cases=cases)
        def run(name,document,accept):
            capture.write_text(json.dumps(document));out=root/('result-'+name)
            proc=subprocess.run([str(bench),'--model',str(model),'--capture-json',str(capture),'--out-dir',str(out),'--host-only'],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
            controls.append(dict(case=name,expected='accept' if accept else 'reject',returncode=proc.returncode,
                passed=(proc.returncode==0)==accept,stderr=proc.stderr[-1000:]))
            if accept and proc.returncode==0:
                manifest=json.loads((out/'artifact-manifest.json').read_text())
                assert manifest['phase']=='host' and manifest['mode']=='not_initialized'
                assert manifest['full_f32_bit_exact'] is False
                assert all(not t['complete_interval_sha_checked'] and t['sha256'] is None for t in manifest['model_backed_tensors'])
        run('complete_CPU_metadata_payload_positive',original,True)
        def mutate(name,fn):
            v=copy.deepcopy(original);fn(v);run(name,v,False)
        mutate('wrong_capture_kind',lambda v:v.update(capture_kind='decode'))
        mutate('wrong_model_extent',lambda v:v.update(model_bytes=100))
        mutate('wrong_header_sha',lambda v:v.update(model_header_sha256='0'*64))
        mutate('wrong_source_kind',lambda v:v['source'].update(kind='random'))
        mutate('trace_capture',lambda v:v['source'].update(trace_on=True))
        mutate('wrong_actual_token_count',lambda v:v['source'].update(actual_tokens=16))
        mutate('missing_actual_layer',lambda v:v.update(cases=v['cases'][:-1]))
        mutate('input_descriptor_extent',lambda v:v['cases'][0]['input_f32'].update(bytes=4))
        mutate('input_descriptor_sha',lambda v:v['cases'][0]['input_f32'].update(sha256='0'*64))
        mutate('wrong_clamp',lambda v:v['cases'][0].update(clamp=11.))
        mutate('rows_exceed_workspace',lambda v:v['cases'][0].update(rows=16385))
        mutate('workspace_capacity_mismatch',lambda v:v['cases'][0].update(capacity=2049))
        p=root/'ids-0.bin';saved=p.read_bytes();p.write_bytes(struct.pack('<i',384)+saved[4:]);v=copy.deepcopy(original)
        v['cases'][0]['ids_i32']['sha256']=sha(p);run('actual_ID_out_of_range',v,False);p.write_bytes(saved)
        p.write_bytes(struct.pack('<2i',0,0)+saved[8:]);v=copy.deepcopy(original);v['cases'][0]['ids_i32']['sha256']=sha(p)
        run('actual_ID_duplicate',v,False);p.write_bytes(saved)
        p=root/'input-0.bin';saved=p.read_bytes();p.write_bytes(struct.pack('<I',0x7fc00000)+saved[4:]);v=copy.deepcopy(original)
        v['cases'][0]['input_f32']['sha256']=sha(p);run('actual_input_NaN',v,False);p.write_bytes(saved)
    result=dict(schema_version=1,gpu_work_submitted=False,model_inference_requested=False,
        child_modes=['--host-only'],fixture='Synthetic sparse tensor-descriptor/parser fixture; no weight bytes or real predictions',
        bench_sha256=sha(bench),all_controls_pass=all(c['passed'] for c in controls),controls=controls)
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(all_controls_pass=result['all_controls_pass'],controls=len(controls))))
    return 0 if result['all_controls_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
