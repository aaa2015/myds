#!/usr/bin/env python3
"""CPU fixtures/mocks for continuation identity and suffix orchestration only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('continuation',Path(__file__).with_name('run_moe_workspace_recovery_continue_promptclear.py'))
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
t=importlib.util.spec_from_file_location('actual_archive_CPU',Path(__file__).with_name('archive_moe_workspace_stage.py'))
ar=importlib.util.module_from_spec(t);t.loader.exec_module(ar)


def run():
    controls=[]
    def test(name,fn,reject=False):
        try:fn();ok=not reject
        except (RuntimeError,ValueError,KeyError,TypeError,OSError):ok=reject
        controls.append(dict(name=name,passed=ok));r.need(ok,'CPU control '+name)
    with tempfile.TemporaryDirectory(prefix='continuation-CPU-')as temp:
        base=Path(temp);counter=0
        plan=r.archive.load(Path(__file__).parent.parent/'docs/benchmarks/2026-10-24-moe-phys-cause/experiment-plan.json')
        matrix=r.original.frozen_matrix(plan)
        def setup():
            nonlocal counter
            counter+=1;home=base/str(counter);home.mkdir();oldstage=home/'oldstage';stage=home/'newstage'
            for st in (oldstage,stage):
                st.mkdir()
                for rel in ('Makefile',*r.IDENTITY_PATHS,*r.NATIVE_SOURCES,r.V2_PAIR_READER,'tools/review_moe_workspace_fullmodel_promptclear.py'):
                    p=st/rel;p.parent.mkdir(exist_ok=True,parents=True);p.write_text('CPU fixture only; never executable '+rel+'\n')
            env={f'MYDS_CPU_FIXTURE_{i}':'1'for i in range(50)};env['MYDS_METAL_CONCURRENT']='1'
            before={k:'CPU-fixture-'+k for k in r.archive.IDENTITY_KEYS};before.update(pid=31510,effective_myds_env=env,health={'status':'ok'})
            model=dict(path='CPU-never-loaded',bytes=123,mtime_ns=42,device=1,inode=2);summary=dict(model=model,model_header_sha256='CPU-header')
            component=home/'component.json';component.write_text('{"CPU_fixture_only":true}\n')
            oldroot=home/'original-failed';oldroot.mkdir();(oldroot/'arms').mkdir()
            restore=dict(restored=True,pid=31510,exact_argv_myds_artifacts_config=True)
            def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
            def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=r.sha(p))
            write(oldroot/'before.json',before);write(oldroot/'restored.json',before)
            write(oldroot/'summary.json',dict(status='failed',recovery_gate_pass=False,restoration=restore))
            records=[];steps=[];rawfiles=[]
            for item in r.matrix_records(matrix[:10]):
                directory=oldroot/'arms'/item['label'];directory.mkdir();p=directory/'summary.json'
                native=dict(status='complete',fullmodel_recovery_within_arm_completed=True,scenario=item['scenario'],workspace_mode=item['setting'],temperature=float(item['temperature']),recovery_tokens=8,same_context=True,rng_manually_reset=False,extra_gpu_waits_inserted=0)
                write(p,native);records.append(dict(item,raw_directory=str(directory),native_summary_sha256=r.sha(p)))
                binary='test_moe_workspace_cancel_promptclear'if item['scenario']in ('deterministic','async')else'test_moe_workspace_recovery_promptclear'
                steps.append(dict(name=item['label'],returncode=0,argv=[str(oldstage/'bin'/binary)]))
                rawfiles.append(dict(source_path=str(p),sha256=r.sha(p),bytes=p.stat().st_size))
            write(oldroot/'steps.json',steps);write(oldroot/'recovery-arms.json',records)
            inputs={str(p):r.sha(p)for p in oldstage.rglob('*')if p.is_file()};write(oldroot/'stage-inputs.json',inputs)
            executed=home/'executed';ar.create(oldroot,executed)
            raw=home/'raw';raw.mkdir();(raw/'raw.tar.gz').write_bytes(b'CPU fixture raw archive never used as native evidence')
            write(raw/'raw-files.json',dict(logical_files=10,files=rawfiles,virtual_kv_files=0))
            reports=home/'pair-reports';reports.mkdir();pairs=[]
            for i in range(0,10,2):
                a=matrix[i];pair=r.original.pair_name(a['scenario'],a['temperature']);p=reports/(pair+'.json')
                write(p,dict(evidence_valid=True,correctness_pass=True,performance_measured=False,scenario=a['scenario'],temperature=float(a['temperature']),recovery_tokens=8))
                pairs.append(dict(pair=pair,scenario=a['scenario'],temperature=a['temperature'],report=ref(p)))
            source_refs={}
            for key,rel in (('cancel','tools/test_moe_workspace_cancel_promptclear.py'),('failure_v1','tools/test_moe_workspace_recovery_promptclear.py'),('failure_v2',r.V2_PAIR_READER)):
                p=stage/rel
                if not p.exists():p.write_text('CPU fake '+key)
                source_refs[key]=ref(p)
            value=dict(schema_version=1,implementation_version=4,plan_sha256=r.workspace.PLAN_SHA,recovery_plan_sha256=r.workspace.PLAN_SHA,evidence_valid=True,prefix_gate_pass=True,recovery_gate_pass=False,completed_native=10,valid_complete_pairs=5,remaining_native=18,remaining_pairs=9,next_pair='allocation-t0p7',native_identity={'candidate':{p:r.sha(stage/p)for p in r.IDENTITY_PATHS}},production_environment=env,model=model,model_header_sha256='CPU-header',component_review_sha256=r.sha(component),original_failed_window=str(oldroot),original_summary=ref(oldroot/'summary.json'),original_before=ref(oldroot/'before.json'),original_restored=ref(oldroot/'restored.json'),restoration=restore,executed_archive=dict(directory=str(executed),source=ref(executed/'source.tar.gz'),artifacts=ref(executed/'artifacts.tar.gz'),bindings=ref(executed/'archive-bindings.json'),stage_inputs=ref(executed/'stage-inputs.json')),raw_archive=dict(directory=str(raw),archive=ref(raw/'raw.tar.gz'),manifest=ref(raw/'raw-files.json')),ordered_records=records,ordered_pair_reviews=pairs,pair_reader_sources=source_refs)
            replay=home/'independent-prefix.json';write(replay,value);value['independent_replay_report']=ref(replay)
            prefix=home/'prefix.json';write(prefix,value);out=home/'new-out';out.mkdir()
            args=SimpleNamespace(stage=stage,gold_stage=home/'gold',out=out,model=home/'CPU-model',component_review=component,
                prefix_manifest=prefix,prefix_manifest_sha256=r.sha(prefix),prepare=False,run=True,inputs=home/'inputs')
            return SimpleNamespace(home=home,oldroot=oldroot,stage=stage,args=args,before=before,summary=summary,value=value,write=write,ref=ref,matrix=matrix)
        def validate(f,value=None):return r.validate_prefix(value or f.value,f.args,f.before,f.summary,f.matrix,{})
        f=setup();test('real_archive_CPU_metadata_prefix_positive',lambda:validate(f))
        for field,new in (('completed_native',8),('valid_complete_pairs',4),('remaining_native',20),('next_pair','allocation-t0'),('recovery_gate_pass',True),('implementation_version',3)):
            f=setup();q=copy.deepcopy(f.value);q[field]=new;test('reject_prefix_'+field,lambda f=f,q=q:validate(f,q),True)
        for field,new in (('model',{}),('model_header_sha256','wrong'),('production_environment',{}),('component_review_sha256','wrong')):
            f=setup();q=copy.deepcopy(f.value);q[field]=new;test('reject_identity_'+field,lambda f=f,q=q:validate(f,q),True)
        f=setup();q=copy.deepcopy(f.value);q['ordered_records'][1]=q['ordered_records'][0];test('reject_duplicate_prefix_arm',lambda:validate(f,q),True)
        f=setup();q=copy.deepcopy(f.value);q['ordered_records'][0]['temperature']='0.7';test('reject_prefix_wrong_order',lambda:validate(f,q),True)
        f=setup();f.before['pid']=31511;test('reject_new_external_production_window',lambda:validate(f),True)
        f=setup();(f.stage/'src/myds_v41.c').write_text('differentcore');test('reject_new_core_changed',lambda:validate(f),True)
        f=setup();(f.stage/'bin/test_moe_workspace_recovery_promptclear').write_text('differentbin');test('reject_native_binary_changed',lambda:validate(f),True)
        f=setup();(f.stage/'tools/test_moe_workspace_recovery_promptclear.c').write_text('differentsource');test('reject_native_C_changed_with_same_fake_bin',lambda:validate(f),True)
        f=setup();(f.stage/r.V2_PAIR_READER).write_text('differentreader');test('reject_pair_reader_changed',lambda:validate(f),True)
        f=setup();p=Path(f.value['raw_archive']['archive']['path']);p.write_bytes(p.read_bytes()+b'x');test('reject_raw_archive_bytes_changed',lambda:validate(f),True)
        f=setup();p=Path(f.value['executed_archive']['source']['path']);p.write_bytes(p.read_bytes()+b'x');test('reject_old_source_archive_changed',lambda:validate(f),True)
        f=setup();q=copy.deepcopy(f.value);q['original_summary']['bytes']=True;test('reject_bool_descriptor_extent',lambda:validate(f,q),True)
        f=setup();q=r.archive.load(Path(f.value['independent_replay_report']['path']));q['model']={};p=Path(f.value['independent_replay_report']['path']);f.write(p,q);f.value['independent_replay_report']=f.ref(p)
        test('reject_other_prefix_independent_report_even_rehashed',lambda:validate(f),True)
        # Preflight parent and subprocess methods are mocked; all identity/file checks above remain real CPU work.
        f=setup();worker=object.__new__(r.RecoveryContinuation);worker.a=f.args;worker.before=f.before;worker.summary=copy.deepcopy(f.summary);worker.matrix=matrix
        worker.environment=copy.deepcopy(f.before['effective_myds_env']);worker.arm_records=[];worker.pair_reports=[]
        worker.requests={name:f.home/(name+'.json')for name in ('warm','failure','recover')}
        worker.input_hashes={str(p):r.sha(p)for p in f.stage.rglob('*')if p.is_file()};worker.input_hashes[str(Path(r.__file__).resolve())]=r.sha(Path(r.__file__))
        with patch.object(r.original.RecoveryWindow,'preflight',lambda self:None):
            test('continuation_preflight_freezes_original_evidence',worker.preflight)
            good_sha=worker.a.prefix_manifest_sha256;worker.a.prefix_manifest_sha256='0'*64
            test('reject_explicit_prefix_manifest_SHA_mismatch',worker.preflight,True)
            worker.a.prefix_manifest_sha256=good_sha
        test('only18_remaining_native_rows',lambda:r.need(len(worker.remaining_matrix)==18 and worker.remaining_matrix[0]==dict(scenario='allocation',temperature='0.7',setting=0),'suffix'))
        calls=[]
        def command(label,argv,env,timeout):
            argv=list(map(str,argv));calls.append((label,argv,copy.deepcopy(env)))
            get=lambda key:argv[argv.index(key)+1]
            if label.endswith('-reader'):
                off=r.archive.load(Path(get('--off'))/'summary.json');on=r.archive.load(Path(get('--on'))/'summary.json')
                r.need(off['workspace_mode']==0 and on['workspace_mode']==1 and Path(argv[1]).name=='test_moe_workspace_recovery_promptclear_v2.py','actual v2 off/onorder')
                f.write(Path(get('--out')),dict(evidence_valid=True,correctness_pass=True,performance_measured=False,scenario=off['scenario'],temperature=off['temperature'],recovery_tokens=8))
            else:
                d=Path(get('--out-dir'));r.need(not d.exists(),'freshnative context');d.mkdir()
                f.write(d/'summary.json',dict(status='complete',fullmodel_recovery_within_arm_completed=True,scenario=get('--scenario'),workspace_mode=int(env[r.FLAG]),temperature=float(get('--temperature')),recovery_tokens=8,same_context=True,rng_manually_reset=False,extra_gpu_waits_inserted=0))
        worker.run_command=command;test('mock_run_exact_remaining9pairs18natives',worker.component)
        native=[v for v in calls if not v[0].endswith('-reader')]
        test('no_cancel_or_prefix_native_rerun',lambda:r.need(len(native)==18 and [x[0]for x in native]==[x['label']for x in r.matrix_records(matrix[10:])],'native suffix'))
        test('all9_pair_calls_v2',lambda:r.need(len(calls)==27 and all(Path(x[1][1]).name=='test_moe_workspace_recovery_promptclear_v2.py'for x in calls if x[0].endswith('-reader')),'v2'))
        test('all51env_plus_exact_off_on_flags',lambda:r.need(all(x[2]==dict(f.before['effective_myds_env'],**{r.FLAG:str(i%2),r.CLEAR_FLAG:str(i%2)})for i,x in enumerate(native)),'env'))
        negative=copy.copy(worker);negative.a=copy.copy(worker.a);negative.a.out=f.home/'negative-pair';negative.a.out.mkdir();negative.arm_records=[];negative.pair_reports=[]
        started=len(calls)
        def negative_command(label,argv,env,timeout):
            command(label,argv,env,timeout)
            if label.endswith('-reader'):
                argv=list(map(str,argv));out=Path(argv[argv.index('--out')+1]);q=r.archive.load(out);q['evidence_valid']=False;f.write(out,q)
        negative.run_command=negative_command
        test('reject_negative_first_suffix_pair',negative.component,True)
        test('negative_pair_stops_after_only_two_new_natives',lambda:r.need(len(calls)-started==3 and len(negative.arm_records)==2 and not negative.pair_reports,'no later executions'))
        def lifecycle(name,error=False,restored=True,arms=18):
            o=object.__new__(r.RecoveryContinuation);out=f.home/name;out.mkdir();o.a=SimpleNamespace(out=out,prepare=False)
            o.summary=dict(status='component_complete_review_required',continuation_pairs_pass=True,restoration=dict(restored=restored));o.before=f.before;o.arm_records=[{}]*arms;o.pair_reports=[{}]*9
            if error:o.summary['error']='CPU injected failure'
            o.event=lambda *a,**k:None
            with patch.object(r.workspace.WorkspaceWindow,'run',lambda self:0):code=o.run()
            return code,r.archive.load(out/'recovery-continuation-review.json')
        test('suffix_pass_never_claims_full28_gate',lambda:r.need((lambda v:v[0]==0 and v[1]['continuation_gate_pass']is True and v[1]['recovery_gate_pass']is False and v[1]['requires_combined_independent_28_native_14_pair_review']is True)(lifecycle('life-positive')),'aggregate required'))
        test('reject_suffix_runtime_error',lambda:r.need(lifecycle('life-error',error=True)[0]==1,'error'))
        test('reject_suffix_restore_failure',lambda:r.need(lifecycle('life-restore',restored=False)[0]==1,'restore'))
        test('reject_suffix_missing_native',lambda:r.need(lifecycle('life-short',arms=17)[0]==1,'partial'))
    return dict(schema_version=1,scope='continuation_CPU_archive_fixtures_and_mock_orchestration_only',controller_sha256=r.sha(Path(r.__file__)),workspace_helper_sha256=r.sha(Path(r.workspace.__file__)),archive_helper_sha256=r.sha(Path(r.archive.__file__)),gpu_run=False,model_run=False,service_operations=False,subprocesses_run=False,controls=controls,passed=all(v['passed']for v in controls))


if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
