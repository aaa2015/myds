#!/usr/bin/env python3
"""Bounded CPU controls for reuse raw readers and mocked window lifecycle only."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode = True


def module(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True); parser.add_argument('--reader',default='review_moe_workspace_reuse_artifacts',choices=['review_moe_workspace_reuse_artifacts','review_moe_workspace_reuse_artifacts_v2'])
    args = parser.parse_args(); r = module(args.reader); c = module('run_moe_workspace_reuse_window')
    checks = []
    def test(name, function, accepted=True):
        error = None
        try: function(); actual = True
        except Exception as exc: actual = False; error = repr(exc)
        checks.append(dict(case=name,expected_accept=accepted,accepted=actual,passed=actual==accepted,error=error))
    def demand(value,message='CPU control failed'):
        if not value: raise AssertionError(message)
    with tempfile.TemporaryDirectory(prefix='moews-reuse-CPU-only-') as tmp:
        base = Path(tmp); data = []
        for run in range(3):
            for phase,label in ((1,'request_prepare'),(2,'generation'),(3,'raw_export')):
                value = dict(task_valid=True,device_valid=True,phys_footprint_bytes=1000+run*10,
                             rss_bytes=2000,device_allocated_bytes=3000+run*10)
                for sample in range(2):
                    data.append(dict(kind='memory_sample',run=run,phase=phase,monotonic_s=run+phase*.1+sample*.01,value=copy.deepcopy(value)))
        reports = [dict(kind='memory_phase',run=run,phase=phase,label=label,interval_ms=10,sampling_ok=True,
                       observed_samples=2,observed_peak_phys_footprint_bytes=1000+run*10,observed_peak_rss_bytes=2000,
                       observed_peak_device_allocated_bytes=3000+run*10,current=copy.deepcopy(data[run*6+(phase-1)*2]['value']))
                   for run in range(3) for phase,label in ((1,'request_prepare'),(2,'generation'),(3,'raw_export'))]
        trace = base/'memory.jsonl'
        def write_trace(values): trace.write_text(''.join(json.dumps(v)+'\n' for v in values))
        write_trace(data)
        test('all9_memory_phases_raw_peak_recalculation',lambda:demand(len(r.memory(reports,trace)['phases'])==9))
        for name, change in (
                ('missing_memory_sample',lambda d:d.pop(0)),
                ('nonmonotonic_memory_time',lambda d:d[1].update(monotonic_s=0)),
                ('invalid_task_observation',lambda d:d[0]['value'].update(task_valid=False)),
                ('negative_device_observation',lambda d:d[0]['value'].update(device_allocated_bytes=-1))):
            changed = copy.deepcopy(data); change(changed); write_trace(changed)
            test(name,lambda:r.memory(reports,trace),False)
        write_trace(data)
        changed = copy.deepcopy(reports); changed[1]['observed_peak_device_allocated_bytes'] += 1
        test('published_peak_cannot_override_raw',lambda:r.memory(changed,trace),False)
        def arms(candidate=1000,field='device_allocated_bytes',baseline=(1000,1000)):
            result = []
            for name,value in zip(r.ARMS,(baseline[0],candidate,baseline[1])):
                peaks = dict(phys_footprint_bytes=1000,rss_bytes=1000,device_allocated_bytes=1000); peaks[field] = value
                result.append(dict(name=name,memory=dict(phases=[dict(run=1,phase=2,peaks=peaks)])))
            return result
        test('no_observed_excess_is_unknown_never_pass',lambda:demand(r.peak_decision(arms())['decision']=='unknown_missing_shadow_peak_coverage' and not r.peak_decision(arms())['reuse_growth_gate_pass']))
        for field in ('device_allocated_bytes','phys_footprint_bytes'):
            test('actual_'+field+'_above128MiB_reject',lambda field=field:demand(r.peak_decision(arms(1000+134217729,field))['decision']=='reject_observed_growth_peak_regression'))
        test('exact128MiB_boundary_remains_unknown',lambda:demand(r.peak_decision(arms(1000+134217728))['decision']=='unknown_missing_shadow_peak_coverage'))
        test('RSS_is_reported_but_not_substituted_for_frozen_gate',lambda:demand(not r.peak_decision(arms(1000+134217729,'rss_bytes'))['observed_excess_blocks_adoption']))
        test('lower_A1_A2_conservative_control',lambda:demand(r.peak_decision(arms(1000+134217729,baseline=(1000,2000)))['observed_excess_blocks_adoption']))
        hashes = r.ByteHashes()
        try:
            directory = base/'state'; directory.mkdir()
            # Tiny complete CPU state, including a retained anchor beyond next.
            scalars = [3,1234,2,9,3,3,1,327680,8,42,0]
            positions = [(0,0)]*44; positions[0]=(2,0); positions[43]=(9,0)
            expected = {'context-scalars.u64le':88,'snapshot-positions.u32le':352,'engram-tail.u32le':12,
                        'prefix-tokens.i32le':8,'anchor-tokens.i32le':36,'tokens.i32le':32,'output.bin':7,'final-logits.f32':129280*4}
            for layer in range(40): expected[f'layer-{layer:02d}-window-active.f32']=3*2048
            for owner in range(4):
                if owner<3:
                    for kind in ('kv','score'):expected[f'owner-{owner}-previous-{kind}.f32']=2048
                n=3//(1 if owner==3 else 2)
                expected[f'owner-{owner}-compressed-active.bf16']=n*1024
                expected[f'owner-{owner}-index-active.bf16']=n*256
            for slot,position in ((0,2),(43,9)):
                expected[f'snapshot-{slot:02d}-engram-position-tail.u32le']=16
                for layer in range(40):expected[f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32']=position*2048
                for owner in range(3):
                    for kind in ('kv','score'):expected[f'snapshot-{slot:02d}-owner-{owner}-previous-{kind}.f32']=2048
            for name,size in expected.items():(directory/name).write_bytes(bytes(size))
            (directory/'context-scalars.u64le').write_bytes(struct.pack('<11Q',*scalars))
            (directory/'snapshot-positions.u32le').write_bytes(b''.join(struct.pack('<II',*p) for p in positions))
            (directory/'tokens.i32le').write_bytes(struct.pack('<8i',*range(8)));(directory/'output.bin').write_bytes(b'output\n')
            for slot,position in ((0,2),(43,9)):(directory/f'snapshot-{slot:02d}-engram-position-tail.u32le').write_bytes(struct.pack('<4I',position,0,0,0))
            manifest = dict(schema_version=1,phase='warm',scope='same_context_moe_workspace_reuse',complete=True,complete_live_state=True,
                scalar_order=r.SCALARS,ctx_pos=3,pcache_len=2,anchor_len=9,next_position=3,engram_position=3,
                rng_state=format(1234,'016x'),anchor_slot=43,requested_prefill_upper_bound=3,limitations=['CPU fixture only','inactive padding excluded'])
            sample = dict(prompt_tokens=3,output_bytes=7,tokens_fnv1a64_bytes=hashes.file(directory/'tokens.i32le')['fnv1a64_bytes'],
                          output_fnv1a64_bytes=hashes.file(directory/'output.bin')['fnv1a64_bytes'])
            def save_manifest(value):
                value=copy.deepcopy(value);value['files']=[dict(path=name,bytes=p.stat().st_size,fnv1a64_bytes=hashes.file(p)['fnv1a64_bytes']) for name in expected for p in [directory/name]]
                value['file_count']=len(value['files']);(directory/'manifest.json').write_text(json.dumps(value))
            save_manifest(manifest)
            test('complete_tiny_state_all_FNV_and_historical_anchor',lambda:demand(len(r.state(directory,'warm',sample,hashes)['published_snapshots'])==2))
            for name, mutate in (
                ('wrong_scope',lambda m:m.update(scope='fresh_context_full_prefill_final_state')),
                ('partial_state_cannot_pass',lambda m:m.update(complete_live_state=False)),
                ('manifest_rng_mismatch',lambda m:m.update(rng_state='0'*16)),
                ('manifest_next_mismatch',lambda m:m.update(next_position=4))):
                changed=copy.deepcopy(manifest);mutate(changed);save_manifest(changed);test(name,lambda:r.state(directory,'warm',sample,hashes),False)
            save_manifest(manifest)
            for name,filename,content in (
                ('updated_FNV_cannot_hide_changed_tokens','tokens.i32le',struct.pack('<8i',99,*range(1,8))),
                ('updated_FNV_cannot_hide_changed_text','output.bin',b'changed'),
                ('poison_logits_rejected','final-logits.f32',b'\xa5'*4+bytes(129280*4-4)),
                ('historical_Engram_position_rejected','snapshot-43-engram-position-tail.u32le',struct.pack('<4I',8,0,0,0))):
                p=directory/filename;original=p.read_bytes();p.write_bytes(content);save_manifest(manifest)
                test(name,lambda:r.state(directory,'warm',sample,hashes),False);p.write_bytes(original)
            save_manifest(manifest); p=directory/'manifest.json'; original=p.read_bytes(); changed=json.loads(original);changed['files'].append(changed['files'][0]);p.write_text(json.dumps(changed))
            test('duplicate_or_missing_raw_state_descriptor_rejected',lambda:r.state(directory,'warm',sample,hashes),False);p.write_bytes(original)
            other=base/'other.f32';other.write_bytes(struct.pack('<I',1));(base/'native.f32').write_bytes(bytes(4))
            test('complete_memcmp_reports_one_F32_bit_difference',lambda:demand(r.file_difference(base/'native.f32',other,True)['f32_bit_mismatches']==1))
        finally: hashes.close()

        # Mock lifecycle: every native invocation is a Python fixture writer.
        # No process, service or actual native benchmark is executed.
        stage=base/'stage';(stage/'bin').mkdir(parents=True);(stage/'tools').mkdir()
        required=('tools/bench_moe_workspace_reuse.c','tools/moe_workspace_reuse.mk','tools/run_moe_workspace_reuse_window.py',
                  'tools/review_moe_workspace_reuse_artifacts.py','bin/bench_moe_workspace_reuse','bin/myds.metallib')
        for name in required:(stage/name).write_text('CPU fake never executed')
        inputs=base/'inputs';inputs.mkdir();repo=Path(__file__).resolve().parent.parent
        for name in ('pf7k.json','pf16k.json'):(inputs/name).write_bytes((repo/'docs/benchmarks/2026-10-24-moe-workspace/requests'/name).read_bytes())
        plan=base/'reuse.json';plan.write_bytes((repo/'docs/benchmarks/2026-10-24-moe-workspace/reuse-growth-plan.json').read_bytes())
        component=base/'component.json';component.write_text('{"evidence_valid":true,"component_gate_pass":true}')
        recovery=base/'recovery.json';recovery.write_text('{"evidence_valid":true,"recovery_gate_pass":true}')
        environment={f'MYDS_CPU_FIXTURE_{i}':str(i) for i in range(48)}
        environment.update(MYDS_METAL_CONCURRENT='1',MYDS_PREFIX_CACHE='1',MYDS_PLD='1')
        count=[0]
        def exercise(mode='success',prepare=False):
            count[0]+=1;out=base/f'window-{count[0]}';out.mkdir();calls=[];operations=[]
            args2=SimpleNamespace(stage=stage,gold_stage=base/'gold',model=base/'fake.gguf',out=out,plan=base/'unused-mainplan',inputs=inputs,
                component_review=component,recovery_review=recovery,reuse_plan=plan,port=8187,prepare=prepare,run=not prepare)
            def preflight(self):
                self.before=dict(pid=123456789,effective_myds_env=dict(environment));self.summary['model']={'CPU_mock':True}
                self.environment=dict(environment);self.plan={'safety':{}}
                self.input_hashes={str(p):c.sha(p) for p in stage.rglob('*') if p.is_file()}
                self.input_hashes[str(Path(c.__file__).resolve())]=c.sha(Path(c.__file__))
            def stop(self):self.stopped=True;operations.append('stop_mock')
            def restore(self):
                operations.append('restore_mock')
                if mode=='restore-failure':raise ValueError('CPU injected restoration failure')
                return dict(restored=True,pid=222,exact_argv_myds_artifacts_config=True)
            def monitor(self):self.done.wait(10)
            def unchanged(self):operations.append('unchanged_mock')
            def native(self,name,argv,env=None,timeout=None):
                calls.append(name)
                if mode=='native-failure':raise ValueError('CPU injected native failure')
                a=list(map(str,argv));d=Path(a[a.index('--out-dir')+1]);d.mkdir()
                meta=dict(kind='metadata',workspace_mode=int(env[c.FLAG]),testing_api_used=False,same_context=True,
                    manual_cache_or_rng_reset=False,extra_gpu_waits_inserted=0,performance_measured=False,
                    environment={k:v for k,v in env.items() if k.startswith('MYDS_')},actual_metallib_path=str(stage/'bin/myds.metallib'))
                records=[meta]+[dict(kind='sample',run=i,step=label,generation_ok=True,raw_state_exported=True,output_tokens=8,
                    stop_reason='length',same_context=True) for i,label in enumerate(c.STEPS)]
                records.append(dict(kind='complete',three_generations_completed=True,actual_C2048_B8192_to16384_grow=True,
                    actual_B16384_retained_after_short=True,performance_measured=False,sampling_coverage_proven=False,adoption_gate_pass=False))
                if mode=='incomplete':records.pop(1)
                if mode=='wrong-actual-env':meta['environment'][c.FLAG]='9'
                (d/'summary.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in records));(d/'original.raw').write_bytes(b'CPU raw retained')
                self.steps.append(dict(name=name,argv=a,returncode=0));c.write_json(out/'steps.json',self.steps)
                if mode=='source-mutation':(stage/'tools/moe_workspace_reuse.mk').write_text('CPU changed frozen input')
            worker=c.ReuseWindow(args2);old=(stage/'tools/moe_workspace_reuse.mk').read_bytes()
            try:
                with patch.object(c.workspace.WorkspaceWindow,'preflight',preflight),patch.object(c.ReuseWindow,'stop',stop), \
                     patch.object(c.ReuseWindow,'restore',restore),patch.object(c.ReuseWindow,'monitor',monitor), \
                     patch.object(c.ReuseWindow,'unchanged',unchanged),patch.object(c.ReuseWindow,'run_command',native), \
                     patch.object(c.workspace.fcntl,'flock',lambda *a:None), \
                     patch.object(c.workspace.safety,'command',lambda *a:'_myds_metal_moe_workspace_test_control' if mode=='testing-binary' else '_normal_api_symbol'):
                    code=worker.run()
            finally:(stage/'tools/moe_workspace_reuse.mk').write_bytes(old)
            return code,worker,calls,operations,out
        def success():
            code,w,calls,ops,out=exercise();demand(code==0 and calls==list(c.ORDER) and ops==['stop_mock','restore_mock'])
            demand(w.summary['reuse_generations_completed']==9 and w.summary['reuse_growth_gate_pass'] is False and w.summary['performance_measured'] is False)
            demand(len(list((out/'arms').rglob('original.raw')))==3)
        test('mock_three_samecontext_arms_restore_raw_and_no_adoption_claim',success)
        def prepare():
            code,w,calls,ops,out=exercise(prepare=True);demand(code==0 and not calls and ops==['unchanged_mock'] and w.summary['status']=='prepared')
        test('prepare_never_stops_or_executes_native',prepare)
        for mode in ('native-failure','restore-failure','incomplete','wrong-actual-env','source-mutation','testing-binary'):
            def negative(mode=mode):
                code,w,calls,ops,out=exercise(mode);demand(code==1 and w.summary['reuse_growth_gate_pass'] is False)
                demand(ops==(['unchanged_mock'] if mode=='testing-binary' else ['stop_mock','restore_mock']))
            test('mock_'+mode+'_fails_closed_and_restores',negative)
        def changed_plan():
            original=plan.read_bytes();plan.write_bytes(original+b' ')
            try:
                code,w,calls,ops,out=exercise();demand(code==1 and not calls and ops==['unchanged_mock'])
            finally:plan.write_bytes(original)
        test('changed_frozen_plan_rejected_before_service_mutation',changed_plan)
        def failed_prerequisite():
            original=recovery.read_bytes();recovery.write_text('{"evidence_valid":true,"recovery_gate_pass":false}')
            try:
                code,w,calls,ops,out=exercise();demand(code==1 and not calls and ops==['unchanged_mock'])
            finally:recovery.write_bytes(original)
        test('failed_recovery_prerequisite_rejected_before_service_mutation',failed_prerequisite)

    result=dict(schema_version=1,scope='Bounded synthetic CPU state/memory and mocked lifecycle controls only',
                model_or_gpu_executed=False,service_operations_performed=False,controls=checks,
                all_controls_pass=all(v['passed'] for v in checks),
                reader_sha256=r.sha(Path(r.__file__)),controller_sha256=r.sha(Path(c.__file__)))
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(all_controls_pass=result['all_controls_pass'],controls=len(checks),failed=[v['case'] for v in checks if not v['passed']])))
    return 0 if result['all_controls_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
