#!/usr/bin/env python3
"""CPU-only positive and tamper controls for MoE workspace fullmodel raw reader."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    path=Path(__file__).with_name('review_moe_workspace_fullmodel_artifacts.py')
    spec=importlib.util.spec_from_file_location('moe_fullmodel_cpu_reader',path)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    results=[];hashes=r.ByteHashes()
    def checked(name,call,expected=True):
        error=None
        try:call();accepted=True
        except (ValueError,KeyError,FileNotFoundError,struct.error) as exc:accepted=False;error=str(exc)
        results.append(dict(case=name,expected_accept=expected,accepted=accepted,passed=accepted==expected,error=error))
    try:
        with tempfile.TemporaryDirectory(prefix='moe-fullmodel-reader-cpu-') as tmp:
            root=Path(tmp);state=root/'state';state.mkdir();logits=root/'logits.f32'
            logits.write_bytes(b'\0'*(129280*4));text=b'CPU fixture never inferred';ids=list(range(32));next_pos=2
            scalars=[2,42,327680,32,42,327680,2,0,0,1,2,2,0,10,0]
            positions=struct.pack('<IIII',1,0,0,2)+b'\0'*(352-16)
            blobs={'context-scalars.u64le':struct.pack('<15Q',*scalars),'engram-tail.u32le':struct.pack('<3I',1,2,3),
                'snapshot-positions.u32le':positions,'prefix-tokens.i32le':b'\0'*8,'anchor-tokens.i32le':b'',
                'tokens.i32le':struct.pack('<32i',*ids),'output.bin':text,'snapshot-00-engram-tail.u32le':struct.pack('<3I',4,5,6)}
            for layer in range(40):
                blobs[f'layer-{layer:02d}-window-active.f32']=b'\0'*4096
                blobs[f'snapshot-00-layer-{layer:02d}-window-active.f32']=b'\0'*2048
            for owner in range(4):
                if owner<3:
                    for kind in ('kv','score'):
                        blobs[f'owner-{owner}-previous-{kind}.f32']=b'\0'*2048
                        blobs[f'snapshot-00-owner-{owner}-previous-{kind}.f32']=b'\0'*2048
                n=next_pos//(1 if owner==3 else 2)
                blobs[f'owner-{owner}-compressed-active.bf16']=b'\0'*(n*1024)
                blobs[f'owner-{owner}-index-active.bf16']=b'\0'*(n*256)
            for name,data in blobs.items():(state/name).write_bytes(data)
            meta=dict(fresh_context=True,byte_order='little',layers=40,window_width=512,window_order='chronological_active_rows',
                compressed_row_bytes=1024,index_row_bytes=256,scalar_order=r.SCALAR_ORDER,ctx_pos=2,next_position=2,window_rows=2,
                rng_state='000000000000002a',engram_position=2,engram_tail=[1,2,3])
            manifest=dict(schema_version=1,scope='fresh_context_full_prefill_final_state',run=0,complete=True,metadata=meta,
                files=[dict(path=n,bytes=len(b),fnv1a64_bytes=hashes.bytes(b)) for n,b in blobs.items()],file_count=len(blobs),
                limitations=['Historical published snapshot Engram tails use an added pure read-only getter.','No allocator padding is compared.'])
            def save():(state/'manifest.json').write_text(json.dumps(manifest))
            save();B=8192;mid=B*6*2304*2
            ws=dict(mode=1,capacity=2048,moe_batch=B,generation=1,legacy_gate_bytes=0,legacy_up_bytes=0,legacy_mid_bytes=0,
                scratch_owner_bytes=mid,workspace_owner_bytes=mid+100,requests=40,mm_encoded=39,mv_encoded=1,failures=0,
                arena=dict(mm_capacity=B,f32_capacity=0,mm_mid_bytes=mid,f32_gate_bytes=0,f32_up_bytes=0,f32_mid_bytes=0,
                    retired_bytes=0,allocated_bytes=mid,peak_allocated_bytes=mid,requests=40,mm_encoded=39,mv_encoded=1,fallback_preencode=1,failures=0))
            sample=dict(run=0,warmup=True,generation_ok=True,sample_valid=True,fresh_context=True,output_tokens=32,
                prompt_tokens=6917,cached_tokens=0,stop_reason='length',prefill_tokens=6916,decode_s=.31,prefill_s=.19,wall_s=1.,
                first_callback_s=.2,last_callback_s=.51,decode_ms_per_token=10.,decode_intervals=31,callback_intervals_ms=[10.]*31,
                final_logits=dict(elements=129280,finite=True,fnv1a64_bytes=hashes.bytes(logits.read_bytes())),
                token_ids=ids,output_bytes=len(text),output_bytes_fnv1a64=hashes.bytes(text),token_ids_fnv1a64=hashes.bytes(blobs['tokens.i32le']),moe_workspace=ws)
            for key in ('gpu_attention_packed_register','gpu_attention_rope_fuse'):
                sample[key]=dict(requests=40,encoded=40,failures=0,decode_requests=39,decode_encoded=39,decode_failures=0)
            def verify():return r.verify_sample(sample,state,logits,1,6917,hashes)
            checked('complete32tokens_live_published_historicalEngram_positive',verify)
            for name,mutation in [
                ('historical_Engram_FNV_tamper',lambda:next(i for i in manifest['files'] if i['path']=='snapshot-00-engram-tail.u32le').update(fnv1a64_bytes='ffffffffffffffff')),
                ('missing_historical_Engram_manifest',lambda:manifest['files'].__setitem__(slice(None),[i for i in manifest['files'] if i['path']!='snapshot-00-engram-tail.u32le'])),
                ('duplicate_manifest',lambda:manifest['files'].append(copy.deepcopy(manifest['files'][0]))),
                ('RNG_metadata',lambda:manifest['metadata'].update(rng_state='ffffffffffffffff')),
                ('escaping_payload',lambda:manifest['files'][0].update(path='../outside'))]:
                old=copy.deepcopy(manifest);mutation();save();checked(name,verify,False);manifest=old;save()
            for name,mutation in [
                ('wrong_tokens256',lambda:sample.update(output_tokens=256)),
                ('short_callback_array',lambda:sample.update(callback_intervals_ms=[10.]*30)),
                ('wrong_actual_provider_count',lambda:sample['moe_workspace'].update(mm_encoded=38)),
                ('legacy_pool_retained',lambda:sample['moe_workspace'].update(legacy_gate_bytes=16)),
                ('wrong_B_capacity',lambda:sample['moe_workspace'].update(moe_batch=2048)),
                ('logical_bytes_tamper',lambda:sample['moe_workspace']['arena'].update(allocated_bytes=mid+1))]:
                old=copy.deepcopy(sample);mutation();checked(name,verify,False);sample=old
            golden=verify();other=copy.deepcopy(golden)
            checked('physical_raw_byte_comparison_positive',lambda:r.need(not r.compare_sample(golden,other),'difference'))
            # The sample peaks are CPU fixture numbers, not observations of an actual model.
            raw=[];reports=[];clock=1.
            for run in range(2):
                for phase,label in [(1,'context_create'),(2,'generation'),(3,'raw_export')]:
                    for k in range(2):
                        raw.append(dict(kind='memory_sample',monotonic_s=clock,run=run,phase=phase,
                            value=dict(task_valid=True,device_valid=True,rss_bytes=100+k,phys_footprint_bytes=200+k,device_allocated_bytes=300+k)));clock+=.01
                    reports.append(dict(kind='memory_phase',run=run,phase=phase,label=label,interval_ms=10,observed_samples=2,
                        observed_peak_rss_bytes=101,observed_peak_phys_footprint_bytes=201,observed_peak_device_allocated_bytes=301,sampling_ok=True))
            trace=root/'samples.jsonl'
            def save_trace():trace.write_text(''.join(json.dumps(x)+'\n' for x in raw))
            save_trace()
            checked('raw10ms_memory_phases_peak_positive',lambda:r.memory_review(reports,trace,1))
            old=copy.deepcopy(reports);reports[1]['observed_samples']=1
            checked('memory_missing_raw_count',lambda:r.memory_review(reports,trace,1),False);reports=old
            old=copy.deepcopy(reports);reports[1]['observed_peak_phys_footprint_bytes']=202
            checked('memory_peak_tamper',lambda:r.memory_review(reports,trace,1),False);reports=old
            old=copy.deepcopy(raw);raw[0]['value']['device_valid']=False;save_trace()
            checked('memory_invalid_device_measurement',lambda:r.memory_review(reports,trace,1),False);raw=old;save_trace()
            def group(pf=(10.,10.,10.),dec=(1.,1.,1.),phys=(2000000000,1600000000,2000000000),device=(1000000000,700000000,1000000000)):
                return [dict(samples=[dict(prefill_s=999,decode_s=999)]+[dict(prefill_s=p,decode_s=d)]*3,
                    memory=dict(generation=[{}]+[dict(observed_peak_phys_footprint_bytes=h,observed_peak_device_allocated_bytes=e,observed_peak_rss_bytes=h)]*3)) for p,d,h,e in zip(pf,dec,phys,device)]
            checked('ABA_memory_gain_nonregression_positive',lambda:r.need(r.performance_group('pf7k',group())['gate_pass'],'rejected'))
            for name,g in [('ABA_time_regression',group(pf=(10,10.2,10))),('ABA_control_drift',group(dec=(1,1,1.03))),
                ('ABA_only_logical_savings_no_actual',group(phys=(2000000000,2000000000,2000000000),device=(1000000000,1000000000,1000000000))),
                ('ABA_physical_control_drift',group(phys=(2000000000,1600000000,2300000000)))]:
                checked(name,lambda g=g:r.need(r.performance_group('pf7k',g)['gate_pass'],'rejected'),False)
            checked('all91_gold_source_hashes_embedded',lambda:r.need(len(r.GOLD_SOURCE_HASHES)==91,'wrong core inventory'))
    finally:hashes.close()
    result=dict(schema_version=1,scope='CPU fixture parser and false-accept controls only',gpu_or_model_executed=False,
        reader_sha256=r.sha(path),controls=results,all_pass=all(x['passed'] for x in results))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(all_pass=result['all_pass'],controls=len(results))))
    return 0 if result['all_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
