#!/usr/bin/env python3
"""Mock only the full-model runner's child, then exercise raw CPU gate checks.

No run/preflight/stop/restore/monitor method is called; no subprocess, Metal,
network, model, or service operation is performed. Fake counters and outputs
are parser controls, never inference evidence. --out creates a new report.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner',type=Path,default=Path('tools/run_output_a_fullmodel_window.py'))
    p.add_argument('--out',type=Path);a=p.parse_args();runner=a.runner.resolve();sys.path.insert(0,str(runner.parent))
    source=runner.read_bytes();spec=importlib.util.spec_from_file_location('fullmodel_runner_cpu_review',runner)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source_sha=hashlib.sha256(source).hexdigest();controls=[]
    def fnv(data):
        if data.count(0)==len(data):return format((14695981039346656037*pow(1099511628211,len(data),1<<64))% (1<<64),'016x')
        return m.fnv(data)
    with tempfile.TemporaryDirectory(prefix='outa-fullmodel-runner-cpu-') as temporary:
        root=Path(temporary);stage=root/'candidate';gold=root/'gold';out=root/'results';out.mkdir()
        for s in (stage,gold):
            (s/'bin').mkdir(parents=True);(s/'bin/myds.metallib').write_bytes(b'fake_CPU_library')
            (s/'bin/bench_output_a_fullmodel').write_bytes(b'fake_CPU_binary')
            (s/'request-short.json').write_text('{}');(s/'request-business.json').write_text('{}')
        args=SimpleNamespace(stage=stage,gold_stage=gold,out=out,model=root/'fake.gguf')
        w=m.FullmodelWindow(args);w.environment=dict(MYDS_METAL_CONCURRENT='1',MYDS_PREFIX_CACHE='1',MYDS_PLD='1')
        w.plan=dict(workloads=dict(short=dict(expected_prompt_tokens=10),business=dict(expected_prompt_tokens=800)))
        mutation='valid'
        def fake_child(label,argv,env,timeout):
            setting=int(env[m.FLAG]);repeats=int(argv[3]);logits=Path(env['MYDS_BENCH_LOGITS_DIR']);state=Path(env['MYDS_BENCH_STATE_DIR'])
            metadata=dict(kind='metadata',environment={k:v for k,v in env.items() if k.startswith('MYDS_')},protocol_version=3,repeats=repeats,
                warmup_runs=1,max_new_tokens=256,max_seq_len=327680,temperature=0,top_p=1,seed=42,
                top_k=0,min_p=0,repetition_penalty=1,clock='CLOCK_MONOTONIC',transport='direct_byte_callback',
                decode_basis='first_to_last_callback/(tokens-1)',prompt_renderer='parse_chat_prompt')
            if mutation=='extra_selector':metadata['environment']['MYDS_SKIP_Q_A']='1'
            if mutation=='wrong_protocol_version':metadata['protocol_version']=2
            if mutation=='wrong_fixed_config':metadata['max_seq_len']=4096
            load=dict(kind='load',actual_metallib_path=str(w.a.stage/'bin/myds.metallib'),output_a_fixed_wrapper_mode=setting,
                output_a_fixed_wrapper_selected_symbol=m.FIXED if setting else None,output_a_fixed_wrapper_api_available='-gold' not in label,
                fresh_context_per_sample=True,cache_configuration_preserved=True)
            if mutation=='wrong_API_build':load['output_a_fixed_wrapper_api_available']=not load['output_a_fixed_wrapper_api_available']
            if mutation=='wrong_mode':load['output_a_fixed_wrapper_mode']=1-setting
            if mutation=='wrong_selected_symbol':load['output_a_fixed_wrapper_selected_symbol']='wrong_symbol'
            if mutation=='wrong_cache_preservation':load['cache_configuration_preserved']=False
            records=[metadata,load];ids=list(range(1,257));text=b'fake_CPU_text';raw_logits=b'\0'*(129280*4)
            for run in range(repeats+1):
                directory=state/f'run-{run}';directory.mkdir();(logits/f'run-{run}.f32').write_bytes(raw_logits)
                r=dict(kind='sample',run=run,warmup=run==0,sample_valid=True,generation_ok=True,output_tokens=256,prompt_tokens=10,cached_tokens=0,stop_reason='length',
                    prefill_tokens=9,final_logits=dict(finite=True,elements=129280,fnv1a64_bytes=fnv(raw_logits)),token_ids=ids,
                    output_bytes=len(text),output_bytes_fnv1a64=fnv(text),decode_s=2.55,prefill_s=.19,wall_s=3.,first_callback_s=.2,
                    last_callback_s=2.75,decode_ms_per_token=10.,decode_intervals=255,fresh_context=True,callback_intervals_ms=[10.]*255,
                    decode_telemetry=dict(target_forward_calls=256,target_rows=256,output_tokens=256,batch2_calls=0))
                r['gpu_output_a_fixed_wrapper']=dict(requests=10240 if setting else 0,encoded=10240 if setting else 0,failures=0,fallbacks=0,ineligible=0,
                    decode_requests=10200 if setting else 0,decode_encoded=10200 if setting else 0,decode_failures=0,decode_fallbacks=0,decode_ineligible=0,
                    nr0=2 if setting else 0,nsg=4 if setting else 0,fc600=4 if setting else 0)
                for key in ('gpu_attention_packed_register','gpu_attention_rope_fuse'):
                    r[key]=dict(requests=10240,encoded=10240,failures=0,decode_requests=10200,decode_encoded=10200,decode_failures=0)
                blobs={};next_pos=265;scalars=(265,42,327680,256,42,327680,265,0,0,1,265,265,0,10,0)
                blobs['context-scalars.u64le']=struct.pack('<15Q',*scalars);blobs['engram-tail.u32le']=struct.pack('<3I',1,2,3)
                blobs['snapshot-positions.u32le']=b'\0'*352;blobs['prefix-tokens.i32le']=b'\0'*(265*4);blobs['anchor-tokens.i32le']=b''
                blobs['tokens.i32le']=struct.pack('<256i',*ids);blobs['output.bin']=text
                for layer in range(40):blobs[f'layer-{layer:02d}-window-active.f32']=b'\0'*(128*512*4)
                for owner in range(4):
                    if owner<3:
                        for kind in ('kv','score'):blobs[f'owner-{owner}-previous-{kind}.f32']=b'\0'*2048
                    count=next_pos//(1 if owner==3 else 2)
                    blobs[f'owner-{owner}-compressed-active.bf16']=b'\0'*(count*1024)
                    blobs[f'owner-{owner}-index-active.bf16']=b'\0'*(count*256)
                sm=dict(fresh_context=True,layers=40,window_width=512,ctx_pos=265,next_position=265,window_rows=128,
                    byte_order='little',window_order='chronological_active_rows',compressed_row_bytes=1024,index_row_bytes=256,
                    rng_state='000000000000002a',engram_position=265,engram_tail=[1,2,3])
                if mutation=='wrong_stop_length':r['stop_reason']='eos'
                if mutation=='wrong_warmup':r['warmup']=not r['warmup']
                if mutation=='short_output':r['output_tokens']=255
                if mutation=='short_snapshots':blobs['snapshot-positions.u32le']=b'\0'*8
                if mutation=='missing_output':blobs.pop('output.bin')
                if mutation=='wrong_token_bytes':blobs['tokens.i32le']=b'\0'*1024
                if mutation=='wrong_rng_metadata':sm['rng_state']='ffffffffffffffff'
                if mutation=='partial_adoption' and setting:
                    r['gpu_output_a_fixed_wrapper'].update(requests=40,encoded=40,decode_requests=40,decode_encoded=40)
                if mutation=='callback_arithmetic':r['last_callback_s']=3.75
                if mutation=='callback_array':r['callback_intervals_ms']=[20.]*255
                files=[]
                for name,data in blobs.items():
                    (directory/name).write_bytes(data);files.append(dict(path=name,bytes=len(data),fnv1a64_bytes=fnv(data)))
                if mutation=='state_checksum':files[0]['fnv1a64_bytes']='ffffffffffffffff'
                manifest=dict(schema_version=1,run=run,scope='fresh_context_full_prefill_final_state',metadata=sm,files=files,file_count=len(files),complete=True)
                (directory/'manifest.json').write_text(json.dumps(manifest));records.append(r)
            (out/(label+'.stdout')).write_text(''.join(json.dumps(r)+'\n' for r in records))
        w.run_command=fake_child
        positive=[]
        for name,setting in [('gold',0),('off',0),('on',1)]:
            positive.append(w.arm('correctness','short',name,setting,1))
        for r in positive:m.compare_arms(positive[0],r)
        controls.append(dict(case='full_fake_gold_off_on_raw_positive',passed=True,expected='accept'))
        for case in ('missing_output','short_snapshots','state_checksum','wrong_rng_metadata','wrong_token_bytes',
                     'partial_adoption','extra_selector','callback_arithmetic','callback_array',
                     'wrong_protocol_version','wrong_fixed_config','wrong_API_build','wrong_mode','wrong_selected_symbol',
                     'wrong_cache_preservation','wrong_stop_length','wrong_warmup','short_output'):
            mutation=case;accepted=True;error=None
            try:w.arm('cpu-negative','short',case,1,1)
            except (RuntimeError,ValueError,KeyError,FileNotFoundError,struct.error) as exc:accepted=False;error=str(exc)
            controls.append(dict(case=case,passed=not accepted,expected='reject',accepted=accepted,error=error))
        # Deliberately keep the SHA witness identical while changing actual
        # retained text bytes, demonstrating the final comparison reads bytes.
        target=Path(positive[1]['payload_roots'][0])/'output.bin';saved=target.read_bytes();target.write_bytes(b'F'+saved[1:])
        rejected=False
        try:m.compare_arms(positive[0],positive[1])
        except (RuntimeError,ValueError):rejected=True
        finally:target.write_bytes(saved)
        controls.append(dict(case='same_hash_witness_different_original_text_bytes',expected='reject',passed=rejected))
    result=dict(schema_version=1,runner_path=str(runner),runner_sha256=source_sha,gpu_work_submitted=False,
                source_unchanged_during_controls=runner.read_bytes()==source,
                fixture_kind='synthetic_CPU_child_output_parser_controls_not_model_evidence',
                all_controls_pass=all(c['passed'] for c in controls),controls=controls)
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.out:
        with a.out.open('x') as f:f.write(text)
    print(text,end='');return 0 if result['all_controls_pass'] else 1


if __name__=='__main__':sys.exit(main())
