#!/usr/bin/env python3
"""Mock-only lifecycle/orchestration controls; no native binary/service/GPU calls."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error("New versioned CPU report path required")
    tool=Path(__file__).with_name('run_moe_workspace_recovery_window_promptclear.py')
    spec=importlib.util.spec_from_file_location('moews_recovery_controller',tool)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    repo=tool.parent.parent;real_plan=repo/'docs/benchmarks/2026-10-24-moe-phys-cause/experiment-plan.json'
    results=[]
    def checked(name,fn):
        try:fn();passed=True;error=None
        except Exception as exc:passed=False;error=repr(exc)
        results.append(dict(case=name,passed=passed,error=error))
    def require(value,message):
        if not value:raise AssertionError(message)
    with tempfile.TemporaryDirectory(prefix='myds-moews-recovery-controller-cpu-') as temp:
        base=Path(temp);stage=base/'stage';(stage/'bin').mkdir(parents=True);(stage/'tools').mkdir()
        for name in ('test_moe_workspace_cancel_promptclear','test_moe_workspace_recovery_promptclear'):
            (stage/'bin'/name).write_text('CPU dummy: never executed')
            for suffix in ('.c','.py'):(stage/'tools'/(name+suffix)).write_text('CPU dummy: never executed')
        for name in (*r.workspace.CORE,'bin/myds.metallib','tools/moe_prompt_snapshot_promptclear.inc',
                     'tools/moe_workspace_layout_promptclear.inc','tools/review_moe_workspace_fullmodel_promptclear.py',
                     'tools/moe_workspace_recovery_promptclear.mk'):
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU dummy: never executed '+name)
        model=base/'fake.gguf';model.write_bytes(b'CPU fixture, never loaded'.ljust(1048576,b'\0'))
        model_record=r.workspace.safety.metadata(model)

        inputs=base/'inputs';inputs.mkdir()
        for name in ('pf7k.json','pf16k.json'):
            (inputs/name).write_bytes((real_plan.parent/'requests'/name).read_bytes())
        recovery_plan=base/'recovery-plan.json';recovery_plan.write_bytes(real_plan.read_bytes())
        component=base/'component-review.json'
        mainplan=base/'main-plan.json';mainplan.write_bytes(real_plan.read_bytes())
        environment={f'MYDS_CPU_FIXTURE_{i}':str(i) for i in range(48)}
        environment.update(MYDS_METAL_CONCURRENT='1',MYDS_PREFIX_CACHE='1',MYDS_PLD='1')
        counter=[0]
        def exercise(kind='success',prepare=False):
            counter[0]+=1;out=base/f'window-{counter[0]}';out.mkdir()
            long_report=base/f'component-independent-long-{counter[0]}.json';long_report.write_text('x'if kind=='component-long-bytes-bool'else '{\"CPU_fixture\":true}\n')
            value=dict(evidence_valid=True,component_gate_pass=kind!='bad-component',implementation_version=4,plan_sha256=r.workspace.PLAN_SHA,
                candidate_core_sha256={n:r.sha(stage/n)for n in r.workspace.CORE},metallib_sha256=r.sha(stage/'bin/myds.metallib'),
                model=copy.deepcopy(model_record),model_header_sha256=r.sha(model,1048576),production_environment=copy.deepcopy(environment),
                long_report=dict(path=str(long_report),sha256=r.sha(long_report),bytes=long_report.stat().st_size))
            if kind=='old-component':value['implementation_version']=2
            if kind=='component-core-mismatch':value['candidate_core_sha256'][r.workspace.CORE[0]]='0'*64
            if kind=='component-lib-mismatch':value['metallib_sha256']='0'*64
            if kind=='component-model-mismatch':value['model']['bytes']+=1
            if kind=='component-header-mismatch':value['model_header_sha256']='0'*64
            if kind=='component-env-mismatch':value['production_environment']['MYDS_PLD']='0'
            if kind=='component-plan-mismatch':value['plan_sha256']='0'*64
            if kind=='component-long-reference-missing':value.pop('long_report')
            if kind=='component-long-missing':value['long_report']['path']=str(base/'missing-long-report.json')
            if kind=='component-long-sha-mismatch':value['long_report']['sha256']='0'*64
            if kind=='component-long-bytes-mismatch':value['long_report']['bytes']+=1
            if kind=='component-long-bytes-negative':value['long_report']['bytes']=-1
            if kind=='component-long-bytes-bool':value['long_report']['bytes']=True
            if kind=='component-long-unknown-key':value['long_report']['unexpected']='CPU fixture forbidden field'
            r.write_json(component,value)
            a=SimpleNamespace(stage=stage,gold_stage=base/'gold',model=model,out=out,
                plan=mainplan,inputs=inputs,component_review=component,recovery_plan=recovery_plan,port=8187,
                prepare=prepare,run=not prepare)
            calls=[];operations=[]
            def preflight(self):
                self.before={'pid':123456789,'effective_myds_env':dict(environment)}
                self.summary.update(model=copy.deepcopy(model_record),model_header_sha256=r.sha(model,1048576),plan_sha256=r.workspace.PLAN_SHA)
                self.config_raw=b'CPU mock configuration, no actual service accessed'
                self.environment=dict(environment,PATH='/usr/bin:/bin')
                files=[*stage.rglob('*'),recovery_plan,component,mainplan,*inputs.iterdir(),tool]
                self.input_hashes={str(p):r.sha(p) for p in files if p.is_file()}
                self.plan=json.loads(real_plan.read_text())
            def stop(self):self.stopped=True;operations.append('stop_mock')
            def restore(self):
                operations.append('restore_mock')
                if kind=='restore-failure':raise RuntimeError('CPU injected restoration failure')
                return dict(restored=True,pid=987654321,exact_argv_myds_artifacts_config=True)
            def monitor(self):self.done.wait(10)
            def unchanged(self):operations.append('unchanged_mock')
            def run_command(self,name,argv,env=None,timeout=None):
                argv=list(map(str,argv));calls.append(dict(name=name,argv=argv,env=copy.deepcopy(env)))
                self.steps.append(dict(name=name,argv=argv,returncode=0,seconds=0))
                r.write_json(self.a.out/'steps.json',self.steps)
                if '-reader' in name:
                    if kind=='reader-exception':raise RuntimeError('CPU injected reader exception')
                    get=lambda option:Path(argv[argv.index(option)+1])
                    off=json.loads((get('--off')/'summary.json').read_text())
                    on=json.loads((get('--on')/'summary.json').read_text())
                    require(off['workspace_mode']==0 and on['workspace_mode']==1,'Reader must receive off/on exact order')
                    report=dict(evidence_valid=kind!='reader-negative',correctness_pass=kind!='reader-negative',performance_measured=False,
                        scenario=off['scenario'],temperature=off['temperature'],recovery_tokens=8)
                    r.write_json(get('--out'),report)
                else:
                    if kind=='native-exception':raise RuntimeError('CPU injected native failure')
                    get=lambda option:argv[argv.index(option)+1]
                    directory=Path(get('--out-dir'));require(not directory.exists(),'Native raw dir must not be precreated');directory.mkdir()
                    summary=dict(status='complete',fullmodel_recovery_within_arm_completed=True,scenario=get('--scenario'),
                        workspace_mode=int(env[r.FLAG]),temperature=float(get('--temperature')),recovery_tokens=int(get('--recover-tokens')),
                        same_context=True,rng_manually_reset=False,extra_gpu_waits_inserted=0)
                    (directory/'original-native-raw.bin').write_bytes(b'CPU mock original raw never cleaned')
                    r.write_json(directory/'summary.json',summary)
                    if kind=='input-mutation':
                        # This temporary input is restored after the final freeze check.
                        (inputs/'pf7k.json').write_bytes((inputs/'pf7k.json').read_bytes()+b' ')
            worker=r.RecoveryWindow(a)
            actual_open=open
            def mock_open(path,*args,**kwargs):
                return actual_open(base/'mock-window.lock'if str(path)=='/tmp/myds-codex-decode-window.lock'else path,*args,**kwargs)
            original=(inputs/'pf7k.json').read_bytes()
            try:
                with patch.object(r.workspace.WorkspaceWindow,'preflight',preflight),patch.object(r.workspace,'open',mock_open,create=True), \
                     patch.object(r.RecoveryWindow,'stop',stop),patch.object(r.RecoveryWindow,'restore',restore), \
                     patch.object(r.RecoveryWindow,'monitor',monitor),patch.object(r.RecoveryWindow,'unchanged',unchanged), \
                     patch.object(r.RecoveryWindow,'run_command',run_command),patch.object(r.workspace.fcntl,'flock',lambda *x:None):
                    code=worker.run()
            finally:(inputs/'pf7k.json').write_bytes(original)
            return code,worker,calls,operations,out
        def success():
            code,w,calls,ops,out=exercise()
            require(code==0 and ops==['stop_mock','restore_mock'],'Exactly one mock stop/restore')
            require(len(calls)==42 and len(w.arm_records)==28 and len(w.pair_reports)==14,'28 native plus14 pair readers')
            prerequisite=json.loads(component.read_text());long=prerequisite['long_report'];require(w.input_hashes.get(long['path'])==long['sha256'],'Actual helper freezes independent component long report')
            expected=[]
            for scenario in r.SCENARIOS:
                for temperature in r.TEMPERATURES:
                    p=r.pair_name(scenario,temperature);expected.extend([p+'-off',p+'-on',p+'-reader'])
            require([c['name'] for c in calls]==expected,'Frozen whole matrix order')
            for c in calls:
                actual={k:v for k,v in c['env'].items() if k.startswith('MYDS_')}
                setting='1' if c['name'].endswith('-on') else '0'
                require(actual==dict(environment,**{r.FLAG:setting,r.CLEAR_FLAG:setting}),'Exact51env plus explicitflag')
                if not c['name'].endswith('-reader'):
                    cmd=c['argv'];require(cmd[0]==str(stage/'bin'/('test_moe_workspace_cancel_promptclear' if cmd[cmd.index('--scenario')+1] in ('deterministic','async') else 'test_moe_workspace_recovery_promptclear')),'Actual binary by scope')
                    require(cmd[cmd.index('--recover-tokens')+1]=='8','Frozen recover count')
                    if '--checkpoint' in cmd:require(cmd[cmd.index('--checkpoint')+1]=='1' and cmd[cmd.index('--delay-ms')+1]=='0','Frozen cancellation checkpoint/delay')
            report=json.loads((out/'recovery-review.json').read_text());summary=json.loads((out/'summary.json').read_text())
            events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
            require(report['evidence_valid'] is report['recovery_gate_pass'] is True,'All14reports plus restoration allow review')
            require(report['implementation_version']==4 and report['plan_sha256']==r.workspace.PLAN_SHA==report['recovery_plan_sha256']and report['component_review_sha256']==r.sha(component),'Successful recovery top exact impl/plan/component binding')
            require(summary['fullmodel_run'] is True and summary['performance_measured'] is False,'Explicit fullmodel correctness scope')
            require(events[-1]['kind']=='recovery_decision_complete' and events[-1]['status']==summary['status'],'Final terminal event matches summary')
            require(len(list((out/'arms').rglob('original-native-raw.bin')))==28,'Original raw files retained')
        checked('complete28native_14reader_exactenv_restore_raw_terminal_event_actual3field_component_long',success)
        def prepare():
            code,w,calls,ops,out=exercise(prepare=True)
            require(code==0 and not calls and ops==['unchanged_mock'] and w.summary['status']=='prepared','Prepare has no stop/GPU/native/reader')
            require(not (out/'recovery-review.json').exists(),'Prepare cannot authorize recovery')
        checked('prepare_no_stop_native_reader_or_positive_report',prepare)
        for kind in ('native-exception','reader-negative','reader-exception','restore-failure','input-mutation'):
            def negative(kind=kind):
                code,w,calls,ops,out=exercise(kind)
                report=json.loads((out/'recovery-review.json').read_text())
                require(code==1 and report['evidence_valid'] is report['recovery_gate_pass'] is False,'Partial/failure cannot claim gate')
                require(ops==['stop_mock','restore_mock'],'Failure path still restores')
            checked(kind+'_fails_closed_and_restores',negative)
        for kind in ('bad-component','old-component','component-core-mismatch','component-lib-mismatch','component-model-mismatch',
                     'component-header-mismatch','component-env-mismatch','component-plan-mismatch',
                     'component-long-reference-missing','component-long-missing','component-long-sha-mismatch',
                     'component-long-bytes-mismatch','component-long-bytes-negative','component-long-bytes-bool','component-long-unknown-key'):
            def negative_prereq(kind=kind):
                code,w,calls,ops,out=exercise(kind);require(code==1 and not calls and ops==['unchanged_mock'],'Reject exact prerequisite before stop')
            checked(kind+'_rejected_before_stop',negative_prereq)

        def plan_tamper():
            original=recovery_plan.read_bytes();recovery_plan.write_bytes(original+b' ')
            try:
                code,w,calls,ops,out=exercise();require(code==1 and not calls and ops==['unchanged_mock'],'Reject exactplanSHA before stop')
            finally:recovery_plan.write_bytes(original)
        checked('mutated_recovery_plan_before_stop',plan_tamper)
        checked('frozen_matrix28_exact_pairs',lambda:require(len(r.frozen_matrix(json.loads(real_plan.read_text())))==28,'matrix'))
    report=dict(schema_version=3,fixture_contract='impl4-plan-core4-lib-model-env-and-long-two-exact-shapes-v3',test_sha256=r.sha(Path(__file__)),base_helper_sha256=r.sha(Path(r.workspace.__file__)),scope='CPU mocked orchestration/lifecycle only; dummy native/reader outputs are not GPU or raw-state evidence',
        model_or_gpu_executed=False,service_operations_performed=False,runner_sha256=r.sha(tool),controls=results,all_pass=all(x['passed'] for x in results))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(all_pass=report['all_pass'],controls=len(results))))
    return 0 if report['all_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
