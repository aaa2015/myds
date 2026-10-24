#!/usr/bin/env python3
"""CPU parser controls for the independent reader; never runs inference."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reader',type=Path,default=Path(__file__).with_name('review_output_a_fullmodel_artifacts.py'))
    parser.add_argument('--protocol-copy-cpu-control',action='store_true')
    parser.add_argument('--library-alias-cpu-control',action='store_true');args=parser.parse_args()
    reader=args.reader.resolve()
    spec=importlib.util.spec_from_file_location('independent_outa_cpu_reader',reader)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    checks=[];hashes=r.ByteHashes()
    def checked(name,function,accept):
        error=None
        try:function();accepted=True
        except (ValueError,KeyError,FileNotFoundError,struct.error) as exc:accepted=False;error=str(exc)
        checks.append(dict(case=name,expected='accept' if accept else 'reject',accepted=accepted,passed=accepted==accept,error=error))
    try:
        with tempfile.TemporaryDirectory(prefix='outa-fullmodel-reader-cpu-') as tmp:
            root=Path(tmp).resolve();state=root/'state';state.mkdir();logits=root/'logits.f32';logits.write_bytes(b'\0'*(129280*4))
            ids=list(range(256));text=b'CPU only original text bytes';next_pos=2
            scalars=[2,42,327680,256,42,327680,2,0,0,1,2,2,0,10,0]
            blobs={'context-scalars.u64le':struct.pack('<15Q',*scalars),'engram-tail.u32le':struct.pack('<3I',1,2,3),
                   'snapshot-positions.u32le':b'\0'*352,'prefix-tokens.i32le':b'\0'*8,'anchor-tokens.i32le':b'',
                   'tokens.i32le':struct.pack('<256i',*ids),'output.bin':text}
            for layer in range(40):blobs[f'layer-{layer:02d}-window-active.f32']=b'\0'*4096
            for owner in range(4):
                if owner<3:
                    for kind in ('kv','score'):blobs[f'owner-{owner}-previous-{kind}.f32']=b'\0'*2048
                count=next_pos//(1 if owner==3 else 2)
                blobs[f'owner-{owner}-compressed-active.bf16']=b'\0'*(count*1024)
                blobs[f'owner-{owner}-index-active.bf16']=b'\0'*(count*256)
            for name,data in blobs.items():(state/name).write_bytes(data)
            metadata=dict(fresh_context=True,byte_order='little',layers=40,window_width=512,window_order='chronological_active_rows',
                compressed_row_bytes=1024,index_row_bytes=256,scalar_order=r.SCALAR_ORDER,ctx_pos=2,next_position=2,window_rows=2,
                rng_state='000000000000002a',engram_position=2,engram_tail=[1,2,3])
            manifest=dict(schema_version=1,scope='fresh_context_full_prefill_final_state',run=0,complete=True,metadata=metadata,
                files=[dict(path=n,bytes=len(b),fnv1a64_bytes=hashes.bytes(b)) for n,b in blobs.items()],file_count=len(blobs),
                limitations=['Historical snapshot Engram tails are outside export.','No allocator padding is compared.'])
            def save():
                (state/'manifest.json').write_text(json.dumps(manifest))
            save()
            sample=dict(run=0,warmup=True,generation_ok=True,sample_valid=True,fresh_context=True,output_tokens=256,
                prompt_tokens=10,cached_tokens=0,stop_reason='length',prefill_tokens=9,decode_s=2.55,prefill_s=.19,wall_s=3.,
                first_callback_s=.2,last_callback_s=2.75,decode_ms_per_token=10.,decode_intervals=255,callback_intervals_ms=[10.]*255,
                final_logits=dict(elements=129280,finite=True,fnv1a64_bytes=hashes.bytes(logits.read_bytes())),
                token_ids=ids,output_bytes=len(text),output_bytes_fnv1a64=hashes.bytes(text),token_ids_fnv1a64=hashes.bytes(blobs['tokens.i32le']),
                gpu_output_a_fixed_wrapper=dict(requests=40,encoded=40,decode_requests=40,decode_encoded=40,failures=0,fallbacks=0,ineligible=0,
                    decode_failures=0,decode_fallbacks=0,decode_ineligible=0,nr0=2,nsg=4,fc600=4))
            for key in ('gpu_attention_rope_fuse','gpu_attention_packed_register'):
                sample[key]=dict(requests=40,encoded=40,failures=0,decode_requests=40,decode_encoded=40,decode_failures=0)
            def verify():return r.verify_sample(sample,state,logits,1,10,hashes)
            checked('full_61_file_original_bytes_FNV_positive',verify,True)
            for name,mutate in (
                ('per_item_FNV_tamper',lambda:manifest['files'][0].update(fnv1a64_bytes='ffffffffffffffff')),
                ('duplicate_manifest_item',lambda:manifest['files'].append(copy.deepcopy(manifest['files'][0]))),
                ('missing_output_manifest_item',lambda:manifest['files'].__setitem__(slice(None),[i for i in manifest['files'] if i['path']!='output.bin'])),
                ('RNG_metadata_tamper',lambda:manifest['metadata'].update(rng_state='ffffffffffffffff')),
                ('Engram_tail_metadata_tamper',lambda:manifest['metadata'].update(engram_tail=[4,5,6])),
                ('scalar_layout_tamper',lambda:manifest['metadata'].update(scalar_order=['wrong'])),
                ('directory_traversal',lambda:manifest['files'][0].update(path='../outside'))):
                old=copy.deepcopy(manifest);mutate();save();checked(name,verify,False);manifest=old;save()
            for name,mutate in (
                ('callback_array_sum_disagrees',lambda:sample.update(callback_intervals_ms=[20.]*255)),
                ('callback_array_short',lambda:sample.update(callback_intervals_ms=[10.]*254)),
                ('callback_negative',lambda:sample.update(callback_intervals_ms=[-10.]+[10.]*254)),
                ('callback_arithmetic',lambda:sample.update(last_callback_s=3.75)),
                ('partial_adoption',lambda:sample['gpu_output_a_fixed_wrapper'].update(requests=20,encoded=20)),
                ('wrong_geometry',lambda:sample['gpu_output_a_fixed_wrapper'].update(nsg=2)),
                ('RF_fallback',lambda:sample['gpu_attention_rope_fuse'].update(encoded=39)),
                ('text_native_checksum_tamper',lambda:sample.update(output_bytes_fnv1a64='ffffffffffffffff')),
                ('native_token_ids_tamper',lambda:sample.update(token_ids=[1]*256))):
                old=copy.deepcopy(sample);mutate();checked(name,verify,False);sample=old
            original=(state/'snapshot-positions.u32le').read_bytes()
            (state/'snapshot-positions.u32le').write_bytes(struct.pack('<II',1,1)+b'\0'*344)
            checked('published_and_staged_same_slot',verify,False);(state/'snapshot-positions.u32le').write_bytes(original)
            (state/'snapshot-positions.u32le').write_bytes(b'\0'*8);checked('short_44_slots',verify,False)
            (state/'snapshot-positions.u32le').write_bytes(original)
            original=logits.read_bytes();logits.write_bytes(struct.pack('<I',0x7fc00000)+original[4:])
            checked('nonfinite_raw_logits',verify,False);logits.write_bytes(original)
            golden=verify();copydir=root/'copy';shutil.copytree(state,copydir);other=copy.deepcopy(golden);other['state_path']=str(copydir)
            checked('physical_original_bytes_comparison_positive',lambda:r.need(not r.compare_sample(golden,other),'Mismatch'),True)
            target=copydir/'output.bin';target.write_bytes(b'X'+text[1:])
            checked('original_text_changed_with_identical_hash_witness',lambda:r.need(not r.compare_sample(golden,other),'Mismatch'),False)
            target.write_bytes(text)
            target=copydir/'layer-00-window-active.f32';data=target.read_bytes();target.write_bytes(struct.pack('<I',1)+data[4:])
            checked('state_one_bit_changed_with_identical_hash_witness',lambda:r.need(not r.compare_sample(golden,other),'Mismatch'),False)
            checked('actual_environment_exact_positive',lambda:r.actual_environment(dict(MYDS_PLD='1'),dict(MYDS_PLD='1')),True)
            checked('known_unset_environment_selector_positive',lambda:r.actual_environment(dict(MYDS_PLD='1',MYDS_TRACE=None),dict(MYDS_PLD='1')),True)
            checked('extra_nonnull_actual_environment_selector',lambda:r.actual_environment(dict(MYDS_PLD='1',MYDS_SKIP_Q_A='1'),dict(MYDS_PLD='1')),False)
            def group(ds,ps=(1.,1.,1.)):
                return [dict(samples=[dict(decode_s=999,prefill_s=999)]+[dict(decode_s=d,prefill_s=p)]*3) for d,p in zip(ds,ps)]
            checked('ABA_passing_gate',lambda:r.need(r.performance_group('short',group((10.,9.,10.)))['gate_pass'],'Gate rejected'),True)
            for name,g in [('ABA_decode_regression',group((10.,11.,10.))),('ABA_control_drift',group((10.,9.,10.3))),
                           ('ABA_prefill_regression',group((10.,9.,10.),(1.,1.03,1.))),('ABA_prefill_control_drift',group((10.,9.,10.),(1.,1.,1.03)))]:
                checked(name,lambda g=g:r.need(r.performance_group('short',g)['gate_pass'],'Gate rejected'),False)
            # A complete fake window tests orchestration contracts as well as
            # individual parsers. These identity constants are temporarily
            # replaced only inside this imported CPU test module; no real
            # production hash is fabricated or written to an actual report.
            window=root/'fake-window';window.mkdir();candidate=root/'candidate';native=root/'gold';model=root/'fake.gguf'
            model.write_bytes(b'CPU_fixture_never_loaded_as_a_model')
            plan=json.loads((reader.parent.parent/'docs/benchmarks/2026-10-23-next-directions/output-a-fullmodel-plan.json').read_text())
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            required=['Makefile','tools/bench_output_a_fullmodel.c','src/myds_metal.m','src/myds_v41_gpu.c',
                      'bin/bench_output_a_fullmodel','bin/myds.metallib','bin/myds']
            for stage in (candidate,native):
                for n in required:
                    p=stage/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('fake_CPU_'+n).encode())
            for n in ('tools/run_output_a_fullmodel_window.py','tools/run_iq2_signed4_window.py','metal/probe_output_a_fixed.metal'):
                p=candidate/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('fake_CPU_'+n).encode())
            for workload in ('short','business'):
                p=candidate/f'request-{workload}.json';p.write_text('{}')
                plan['workloads'][workload]['sha256']=r.sha(p);plan['workloads'][workload]['bytes']=2
            planpath=root/'fullmodel-plan.json';write(planpath,plan);write(window/'protocol.json',plan)
            leaf1,leaf2,componentplan=[root/n for n in ('leaf1.json','leaf2.json','component-plan.json')]
            for p in (leaf1,leaf2,componentplan):write(p,dict(synthetic_CPU=True))
            approved=dict(component_and_frozen_two_window_replication_gate_pass=True,every_window_every_layer_every_control_pass=True,
                decision='eligible_for_separate_fullmodel_gate',all_12_groups=[dict(window=w,layer=l,control_kind=c,performance_gate_pass=True,
                    control_drift_percent=0,conservative_time_saved_percent=10) for w in (1,2) for l in (0,19,39) for c in ('production','probe')],
                reports=[dict(path=p.name,sha256=r.sha(p)) for p in (leaf1,leaf2)],plan_path=componentplan.name,plan_sha256=r.sha(componentplan))
            approval=root/'approval.json';write(approval,approved)
            old_binary,old_library=r.PRODUCTION_BINARY,r.PRODUCTION_LIBRARY
            r.PRODUCTION_BINARY=r.sha(native/'bin/myds');r.PRODUCTION_LIBRARY=r.sha(native/'bin/myds.metallib')
            environment=dict(MYDS_METAL_CONCURRENT='1',MYDS_PREFIX_CACHE='1',MYDS_PLD='1')
            before=dict(argv_raw_hex='00',myds_env_raw_hex='00',cwd='/fake/CPU',log='/fake/CPU.log',nice=0,
                binary=dict(path='/fake/CPU/myds'),metallib=dict(path='/fake/CPU/myds.metallib'),config=dict(path='/fake/CPU/myds.conf'),
                binary_sha256=r.PRODUCTION_BINARY,metallib_sha256=r.PRODUCTION_LIBRARY,
                config_sha256=r.hashlib.sha256(b'fake CPU config').hexdigest(),effective_myds_env=environment,health=dict(status='ok'),pid=11)
            write(window/'before.json',before);restored=dict(before,pid=22);write(window/'restored.json',restored)
            write(window/'effective-env.json',environment);(window/'original-myds.conf').write_bytes(b'fake CPU config')
            (window/'production-artifacts').mkdir()
            for n in ('myds','myds.metallib'):shutil.copyfile(native/'bin'/n,window/'production-artifacts'/n)
            steps=[];performance=[]
            for phase in ('correctness','performance'):
                for workload in ('short','business'):
                    decoded=[]
                    names=('gold','off','on') if phase=='correctness' else ('A1','B','A2')
                    for name in names:
                        label=f'{phase}-{workload}-{name}';d=window/label;d.mkdir();(d/'logits').mkdir();(d/'state').mkdir()
                        setting=int(name in ('on','B'));stage=native if name=='gold' else candidate;repeats=1 if phase=='correctness' else 3
                        env=dict(environment,**{r.FLAG:str(setting),'MYDS_BENCH_LOGITS_DIR':str(d/'logits'),'MYDS_BENCH_STATE_DIR':str(d/'state')})
                        write(d/'environment.json',env)
                        meta=dict(kind='metadata',environment=env,protocol_version=3,repeats=repeats,warmup_runs=1,max_new_tokens=256,
                            max_seq_len=327680,temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1,seed=42,clock='CLOCK_MONOTONIC',
                            transport='direct_byte_callback',decode_basis='first_to_last_callback/(tokens-1)',prompt_renderer='parse_chat_prompt',
                            model=str(model),request_file=str(candidate/f'request-{workload}.json'),request_bytes=2,request_fnv1a64=hashes.bytes(b'{}'),
                            prompt_bytes=5,prompt_fnv1a64=hashes.bytes(b'fake!'),dspark_model=None)
                        loaded=dict(kind='load',actual_metallib_path=str(stage/'bin/myds.metallib'),fresh_context_per_sample=True,
                            cache_configuration_preserved=True,output_a_fixed_wrapper_mode=setting,output_a_fixed_wrapper_api_available=name!='gold',
                            output_a_fixed_wrapper_selected_symbol=r.FIXED if setting else None,attention_rope_fuse_mode=1)
                        rows=[meta,loaded];ss=[]
                        for run in range(repeats+1):
                            s=copy.deepcopy(sample);s.update(run=run,warmup=run==0)
                            if workload=='business':s.update(prompt_tokens=800,prefill_tokens=799)
                            if not setting:s['gpu_output_a_fixed_wrapper']={k:0 for k in s['gpu_output_a_fixed_wrapper']}
                            if phase=='performance' and name=='B':s.update(decode_s=2.295,last_callback_s=2.495,decode_ms_per_token=9.,callback_intervals_ms=[9.]*255)
                            sd=d/'state'/f'run-{run}';shutil.copytree(state,sd);sm=copy.deepcopy(manifest);sm['run']=run;write(sd/'manifest.json',sm)
                            shutil.copyfile(logits,d/'logits'/f'run-{run}.f32');s['kind']='sample';rows.append(s);ss.append(s)
                        (window/(label+'.stdout')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
                        steps.append(dict(name=label,returncode=0,argv=[str(stage/'bin/bench_output_a_fullmodel'),str(model),
                            str(candidate/f'request-{workload}.json'),str(repeats),'256','327680']))
                        decoded.append(dict(samples=ss))
                    if phase=='performance':performance.append(r.performance_group(workload,decoded))
            write(window/'steps.json',steps)
            summary=dict(status='fullmodel_passed_review_required',correctness_pass=True,performance_gate_pass=True,performance=performance,
                production_adoption=False,fullmodel_run=True,model=dict(bytes=model.stat().st_size),model_header_sha256=r.sha(model,1048576),
                restoration=dict(restored=True,pid=22,exact_argv_myds_artifacts_config=True))
            write(window/'summary.json',summary);(window/'events.jsonl').write_text(json.dumps(dict(kind='window_complete',status=summary['status']))+'\n')
            bindings={str(p):r.sha(p) for stage in (candidate,native) for p in stage.rglob('*') if p.is_file()}
            bindings.update({str(p):r.sha(p) for p in (planpath,approval,leaf1,leaf2,componentplan)})
            write(window/'stage-inputs.json',bindings);sources=root/'sources.tar.gz';bins=root/'binaries.tar.gz'
            for archive,binary in ((sources,False),(bins,True)):
                with tarfile.open(archive,'w:gz') as tar:
                    for stage in (candidate,native):
                        for p in stage.rglob('*'):
                            if p.is_file() and (p.relative_to(stage).parts[0]=='bin')==binary:tar.add(p,arcname=stage.name+'/'+str(p.relative_to(stage)))
            try:
                checked('complete_12_arm_36_sample_archive_and_prerequisite_window_positive',
                    lambda:r.need(r.review(window,model,[sources],[bins])['fullmodel_performance_gate_pass'],'Full fake window rejected'),True)
                mutated=copy.deepcopy(summary);mutated['performance'][0]['conservative_decode_time_saved_percent']=99;write(window/'summary.json',mutated)
                checked('whole_window_published_summary_fabricated',lambda:r.review(window,model,[sources],[bins]),False);write(window/'summary.json',summary)
                if args.protocol_copy_cpu_control:
                    (window/'protocol.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
                    checked('protocol_JSON_reserialized_but_frozen_original_plan_unchanged',
                        lambda:r.need(r.review(window,model,[sources],[bins])['fullmodel_performance_gate_pass'],'Reserialized protocol rejected'),True)
                    changed=copy.deepcopy(plan);changed['max_new_tokens']=255;write(window/'protocol.json',changed)
                    checked('protocol_JSON_semantically_changed',lambda:r.review(window,model,[sources],[bins]),False)
                    write(window/'protocol.json',plan)
                if args.library_alias_cpu_control:
                    stdout=window/'correctness-short-gold.stdout';original=stdout.read_text();data=[json.loads(x) for x in original.splitlines()]
                    alias=root/'gold-path-alias';alias.symlink_to(native,target_is_directory=True)
                    data[1]['actual_metallib_path']=str(alias/'bin/myds.metallib')
                    stdout.write_text(''.join(json.dumps(x)+'\n' for x in data))
                    checked('actual_library_alias_resolves_to_same_frozen_file',
                        lambda:r.need(r.review(window,model,[sources],[bins])['fullmodel_performance_gate_pass'],'Same file alias rejected'),True)
                    data[1]['actual_metallib_path']=str(candidate/'bin/myds.metallib')
                    stdout.write_text(''.join(json.dumps(x)+'\n' for x in data))
                    checked('different_actual_library_path_even_if_same_bytes',lambda:r.review(window,model,[sources],[bins]),False)
                    stdout.write_text(original)
            finally:r.PRODUCTION_BINARY,r.PRODUCTION_LIBRARY=old_binary,old_library
    finally:hashes.close()
    result=dict(schema_version=1,reader_sha256=r.sha(reader),gpu_work_submitted=False,model_executed=False,service_operations=False,
        scope='Synthetic CPU parser/byte-comparison controls, never inference evidence',all_controls_pass=all(c['passed'] for c in checks),controls=checks)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(all_controls_pass=result['all_controls_pass'],controls=len(checks),failed=[c['case'] for c in checks if not c['passed']])))
    return 0 if result['all_controls_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
