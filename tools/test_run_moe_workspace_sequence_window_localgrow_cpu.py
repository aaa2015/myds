#!/usr/bin/env python3
"""CPU mock-only orchestration controls. No native/model/GPU/service operations."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import run_moe_workspace_sequence_window_localgrow as r


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); controls = []
    def check(name, fn):
        try: fn(); controls.append(dict(case=name, passed=True))
        except Exception as exc: controls.append(dict(case=name, passed=False, error=repr(exc)))
    def require(x, message):
        if not x: raise AssertionError(message)
    repo = Path(__file__).resolve().parent.parent
    plan_path = repo/'docs/benchmarks/2026-10-24-moe-localgrow/experiment-plan.json'
    real_plan = json.loads(plan_path.read_text())
    with tempfile.TemporaryDirectory(prefix='myds-localgrow-sequence-cpu-') as temp:
        root = Path(temp); stage = root/'stage'; gold = root/'gold'; inputs = root/'inputs'; inputs.mkdir()
        for directory in (stage, gold):
            for name in (*r.REQUIRED, *r.CORE, 'bin/myds', 'bin/myds.metallib'):
                path = directory/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('CPU fixture, never executed: '+name)
        for name in ('pf7k', 'pf16k'):
            (inputs/(name+'.json')).write_bytes((plan_path.parent/'requests'/(name+'.json')).read_bytes())
        model = root/'fake.gguf'; model.write_bytes(b'CPU fixture, never loaded'.ljust(1048576,b'\0'))
        badlib = root/'wrong.metallib'; badlib.write_text('CPU fixture wrong library')
        env = {f'MYDS_CPU_FIXTURE_{n}': str(n) for n in range(48)}
        env.update(MYDS_METAL_CONCURRENT='1', MYDS_PREFIX_CACHE='1', MYDS_PLD='1')
        model_record = dict(path=str(model), bytes=model.stat().st_size)
        component = root/'component.json'; recovery = root/'recovery.json'; prior = root/'probe.json'
        serial = [0]
        def exercise(mode='probe', kind='success', prepare=False):
            serial[0] += 1; out = root/f'out-{serial[0]}'; out.mkdir()
            r.write_json(component, dict(evidence_valid=kind!='bad-component', component_gate_pass=True, implementation_version=3))
            r.write_json(recovery, dict(evidence_valid=True, recovery_gate_pass=kind!='bad-recovery', recovery_plan_sha256=r.workspace.PLAN_SHA))
            identity = {label: {name:r.sha(s/name) for name in (*r.CORE, r.NATIVE, 'bin/myds.metallib')}
                        for label,s in (('candidate',stage),('gold',gold))}
            prerequisites = {key:dict(path=str(p),sha256=r.sha(p)) for p,key in
                             ((component,'component_gate_pass'),(recovery,'recovery_gate_pass'))}
            proof = dict(evidence_valid=True, growth_gate_pass=kind!='bad-probe', mode='probe',
                plan_sha256=r.workspace.PLAN_SHA, native_identity=identity, prerequisites=prerequisites,
                production_environment=env, model=model_record, model_header_sha256=r.sha(model,1048576))
            if kind=='probe-identity-mismatch': proof['native_identity']=dict(bad='different')
            if kind=='probe-env-mismatch': proof['production_environment']={}
            r.write_json(prior,proof)
            a=SimpleNamespace(stage=stage,gold_stage=gold,model=model,out=out,plan=plan_path,inputs=inputs,
                component_review=component,recovery_review=recovery,probe_review=prior if mode=='full' else None,
                mode=mode,port=8187,prepare=prepare,run=not prepare)
            calls=[]; ops=[]; comparisons=[]
            def preflight(w):
                w.before=dict(pid=123456789,effective_myds_env=copy.deepcopy(env)); w.environment=dict(env,PATH='/usr/bin:/bin')
                w.summary.update(model=model_record, model_header_sha256=r.sha(model,1048576), plan_sha256=r.workspace.PLAN_SHA)
                w.plan=copy.deepcopy(real_plan)
                files=[p for s in (stage,gold,inputs) for p in s.rglob('*') if p.is_file()]
                files += [plan_path,component,recovery,Path(r.__file__).resolve(),Path(r.review.__file__).resolve()]
                w.input_hashes={str(p):r.sha(p) for p in files}
                if kind=='missing-source': w.input_hashes.pop(str(stage/r.REQUIRED[2]))
            def nm(argv):
                require(argv[:2]==['/usr/bin/nm','-gU'],'Only CPU nm allowed')
                symbols='myds_v41_prefill_moe_workspace_layout myds_v41_prefill_moe_workspace_binding myds_v41_moe_workspace_growth_observer_set'
                return symbols+(' myds_metal_moe_workspace_test_control' if kind=='testing-symbol' else '')
            def stop(w): w.stopped=True; ops.append('stop_mock')
            def restore(w):
                ops.append('restore_mock')
                if kind=='restore-failure': raise RuntimeError('CPU injected restore failure')
                return dict(restored=True,pid=987654321)
            def monitor(w): w.done.wait(10)
            def command(w,name,argv,env=None,timeout=None):
                require(not prepare,'Prepare cannot invoke native')
                argv=list(map(str,argv)); calls.append(dict(name=name,argv=argv,env=copy.deepcopy(env),cwd=str(w.a.stage)))
                if kind=='native-exception': raise RuntimeError('CPU injected native failure')
                directory=Path(argv[argv.index('--out-dir')+1]); require(not directory.exists(),'Native raw must be new')
                directory.mkdir(); (directory/'summary.jsonl').write_text('{"CPU_mock_native":true}\n')
                (directory/'original-raw.bin').write_bytes(b'CPU fixture original raw preserved')
                if kind=='input-mutation': (inputs/'pf7k.json').write_bytes(b'CPU injected changed request')
                if kind=='monitor-fault': w.fault.set()
            def read_arm(directory,phase,setting,requests,expected_env,*,gold=False,expected_model=None):
                call=calls[-1]; av=call['argv']; require(av[av.index('--phase')+1]==phase,'Reader phase matches actual argv')
                require(av[av.index('--repeats')+1]==str({'probe':0,'correctness':1,'performance':3}[phase]),'Exact native repeats')
                require(expected_model==model and requests=={'setup':inputs/'pf7k.json','grow':inputs/'pf16k.json','retain':inputs/'pf7k.json'},'Reader binds model and original requests')
                require({k:v for k,v in call['env'].items() if k.startswith('MYDS_')}==expected_env==dict(env,**{r.FLAG:str(setting)}),'Every effective51 setting and explicit flag preserved')
                native_stage=Path(av[0]).parent.parent;require(call['cwd']==str(native_stage),'Library-priority working directory matches exact native stage')
                if kind=='reader-exception': raise ValueError('CPU injected raw-state reader failure')
                return dict(directory=str(directory),phase=phase,setting=setting,gold=gold,complete=True,
                    metadata=dict(actual_metallib_path=str(badlib if kind=='bad-library' else native_stage/'bin/myds.metallib')))
            def compare(reference,candidate):
                comparisons.append((reference['phase'],candidate['phase'],candidate['setting']))
                if kind=='correctness-mismatch' and candidate['phase']=='correctness' and candidate['setting']==1: raise ValueError('CPU injected last-byte state mismatch')
                if kind=='performance-mismatch' and candidate['phase']=='performance': raise ValueError('CPU injected performance raw mismatch')
                return dict(full_original_bytes_equal=True)
            def growth(*arms):
                require([x['setting'] for x in arms]==[0,1,0] and all(x['phase']=='probe' for x in arms),'One strict probe ABA')
                return dict(growth_gate_pass=kind!='negative-gate')
            def performance(*arms):
                require([x['setting'] for x in arms]==[0,1,0] and all(x['phase']=='performance' for x in arms),'One strict performance ABA')
                return dict(performance_gate_pass=kind!='negative-gate')
            original=(inputs/'pf7k.json').read_bytes()
            actual_open=open
            def mock_open(path,*a,**kw):
                return actual_open(root/'mock-window.lock' if str(path)=='/tmp/myds-codex-decode-window.lock' else path,*a,**kw)
            w=r.SequenceWindow(a)
            try:
                with patch.object(r.workspace.WorkspaceWindow,'preflight',preflight),patch.object(r.workspace,'open',mock_open,create=True),\
                     patch.object(r.workspace.fcntl,'flock',lambda *a:None),patch.object(r.workspace.signal,'signal',lambda *a:None),\
                     patch.object(r.SequenceWindow,'stop',stop),patch.object(r.SequenceWindow,'restore',restore),\
                     patch.object(r.SequenceWindow,'monitor',monitor),patch.object(r.SequenceWindow,'drain',lambda w:None),\
                     patch.object(r.SequenceWindow,'unchanged',lambda w:ops.append('unchanged_mock')),\
                     patch.object(r.SequenceWindow,'run_command',command),patch.object(r.workspace.safety,'command',nm),\
                     patch.object(r.review,'read_arm',read_arm),patch.object(r.review,'compare_arms',compare),\
                     patch.object(r.review,'growth_gate',growth),patch.object(r.review,'performance_gate',performance): code=w.run()
            finally:(inputs/'pf7k.json').write_bytes(original)
            last=json.loads((out/'events.jsonl').read_text().splitlines()[-1]); summary=json.loads((out/'summary.json').read_text())
            require(last['status']==summary['status'],'Terminal event and summary agree')
            require(summary['production_adoption'] is False,'No controller adopts/deploys')
            for path in (out/'arms').glob('*/original-raw.bin') if (out/'arms').exists() else []: require(path.read_bytes()==b'CPU fixture original raw preserved','Original native raw preserved')
            return code,w,calls,ops,comparisons,summary
        def positive(mode):
            code,w,calls,ops,comp,s=exercise(mode)
            require(code==0 and ops==['stop_mock','restore_mock'],'Exactly one mocked stop/restore')
            expected=[('probe',n,v,0)for n,v in r.ORDER] if mode=='probe' else [('correctness','gold',0,1),('correctness','off',0,1),('correctness','on',1,1)]+[('performance',n,v,3)for n,v in r.ORDER]
            require([c['name']for c in calls]==[p+'-'+n for p,n,_,_ in expected],'All arms ordered and complete')
            require(len(calls)==(3 if mode=='probe' else 6) and len(comp)==(3 if mode=='probe' else 6),'No duplicated cohorts, complete raw comparisons')
            require(s['evidence_valid'] and s['growth_gate_pass' if mode=='probe' else 'performance_gate_pass'],'Explicit reader gate passes')
            if mode=='full': require(s['correctness_generations_completed']==18 and s['performance_generations_completed']==36 and s['total_generations_completed']==54,'Frozen generation counts')
        check('probe3ABA9generations_fullraw_growth_gate',lambda:positive('probe'))
        check('full_gold_off_on18_then_singleABA36_fullraw',lambda:positive('full'))
        def prepared():
            code,w,calls,ops,comp,s=exercise(prepare=True);require(code==0 and not calls and ops==['unchanged_mock'] and s['status']=='prepared','Prepare CPU only no stop/native')
        check('prepare_no_stop_native_GPU',prepared)
        def negative(mode):
            code,w,calls,ops,comp,s=exercise(mode,'negative-gate');require(code==0 and s['evidence_valid'] and not s['growth_gate_pass'if mode=='probe'else'performance_gate_pass'] and 'rejected' in s['status'] and ops[-1]=='restore_mock','Valid negative verdict restores and rejects')
        check('negative_probe_valid_reject_not_adopt',lambda:negative('probe'));check('negative_full_valid_reject_not_adopt',lambda:negative('full'))
        for kind in ('bad-component','bad-recovery','testing-symbol','missing-source','bad-probe','probe-identity-mismatch','probe-env-mismatch'):
            def fail(kind=kind):
                code,w,calls,ops,comp,s=exercise('full',kind);require(code==1 and not calls and 'stop_mock'not in ops and s['status']=='failed'and not s['evidence_valid'],'Invalid prerequisite fails before stop')
            check(kind+'_preflight_failclosed',fail)
        for kind in ('native-exception','reader-exception','bad-library','correctness-mismatch','performance-mismatch','input-mutation','monitor-fault','restore-failure'):
            def fail(kind=kind):
                code,w,calls,ops,comp,s=exercise('full',kind);require(code==1 and s['status']=='failed'and not s['evidence_valid']and not s['performance_gate_pass'] and 'restore_mock'in ops,'Failure must restore and invalidate')
                if kind=='correctness-mismatch':require(len(calls)==3 and not any(x['name'].startswith('performance')for x in calls),'No performance after correctness mismatch')
            check(kind+'_restore_and_failclosed',fail)
        def cli_guard():
            argv=[]
            for n in ('stage','gold-stage','model','out','plan','inputs','component-review','recovery-review'):argv += ['--'+n,'/tmp/CPU-fixture']
            try:r.arguments([*argv,'--mode','full','--prepare'])
            except Exception:return
            raise AssertionError('Missing full prior probe falsely accepted')
        check('full_CLI_requires_prior_probe',cli_guard)
    report=dict(schema_version=1,scope='Mock-only controller lifecycle/ordering/arguments/schema controls; real safety capture/stop/restore/nm/reader/native/model/GPU were replaced, not executed',all_pass=all(x['passed']for x in controls),controls=controls,controller_sha256=r.sha(Path(r.__file__)),gpu_or_model_executed=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);r.write_json(args.out,report);print(json.dumps(report,indent=2));return 0 if report['all_pass'] else 1
if __name__=='__main__':raise SystemExit(main())
