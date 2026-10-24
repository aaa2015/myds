#!/usr/bin/env python3
"""Continue only the unexecuted suffix of the frozen28-native recovery matrix.

The failed first window and its ten original native executions stay unchanged.
This window supplies18 additional executions; only a separate combined raw
review may certify the complete28/14 gate. No performance cohort is measured.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
sys.dont_write_bytecode=True
import run_moe_workspace_recovery_window_promptclear as original
import review_moe_workspace_fullmodel_promptclear as archive

workspace=original.workspace
need,sha,write_json=workspace.need,workspace.sha,workspace.write_json
FLAG,CLEAR_FLAG=original.FLAG,original.CLEAR_FLAG
PREFIX_NATIVES=10;PREFIX_PAIRS=5
IDENTITY_PATHS=(*workspace.CORE,'bin/myds.metallib','bin/test_moe_workspace_cancel_promptclear','bin/test_moe_workspace_recovery_promptclear')
NATIVE_SOURCES=('tools/test_moe_workspace_cancel_promptclear.c','tools/test_moe_workspace_recovery_promptclear.c',
    'tools/moe_prompt_snapshot_promptclear.inc','tools/moe_workspace_layout_promptclear.inc',
    'tools/bench_moe_workspace_fullmodel_promptclear.c','tools/bench_moe_workspace_fullmodel.c',
    'tools/moe_workspace_sequence_state_localgrow.inc','tools/moe_workspace_memory.inc')
V2_PAIR_READER='tools/test_moe_workspace_recovery_promptclear_v2.py'


def reference(value,frozen):
    need(type(value)is dict and set(value)=={'path','sha256','bytes'},'Exact original path/SHA/extent descriptor required')
    p=Path(value['path'])
    need(p.is_absolute()and p.is_file()and type(value['bytes'])is int and value['bytes']>=0 and
        p.stat().st_size==value['bytes']and sha(p)==value['sha256'],'Original referenced bytes/SHA differ: '+str(p))
    frozen[str(p)]=value['sha256'];return p


def matrix_records(matrix):
    return [dict(label=original.pair_name(x['scenario'],x['temperature'])+('-on'if x['setting']else'-off'),**x)for x in matrix]


def validate_prefix(value,args,before,summary,matrix,frozen):
    need(value.get('schema_version')==1 and value.get('implementation_version')==4 and
        value.get('plan_sha256')==value.get('recovery_plan_sha256')==workspace.PLAN_SHA and
        value.get('evidence_valid')is True and value.get('prefix_gate_pass')is True and
        value.get('recovery_gate_pass')is False and value.get('completed_native')==PREFIX_NATIVES and
        value.get('valid_complete_pairs')==PREFIX_PAIRS and value.get('remaining_native')==18 and value.get('remaining_pairs')==9 and
        value.get('next_pair')=='allocation-t0p7','Only the exact independently replayed10-native prefix may continue')
    need(value['production_environment']==before['effective_myds_env']and len(value['production_environment'])==51 and
        value['model']==summary['model']and value['model_header_sha256']==summary['model_header_sha256']and
        value['component_review_sha256']==sha(args.component_review),'Prefix actual model/environment/component differs')
    identity=value['native_identity']
    need(set(identity)=={'candidate'}and set(identity['candidate'])==set(IDENTITY_PATHS)and
        identity['candidate']=={p:sha(args.stage/p)for p in IDENTITY_PATHS},'Same four core sources/library/native binaries required')
    oldroot=Path(value['original_failed_window']);need(oldroot.is_absolute()and oldroot.is_dir()and oldroot.resolve()!=args.out.resolve(),'Original failed window must be retained separately')
    oldsummary=reference(value['original_summary'],frozen)
    need(oldsummary.resolve()==(oldroot/'summary.json').resolve(),'Original failed summary path differs')
    for key,name in (('original_before','before.json'),('original_restored','restored.json')):
        need(reference(value[key],frozen).resolve()==(oldroot/name).resolve(),'Original identity descriptor path differs')
    prior=archive.load(oldsummary);restored=archive.load(oldroot/'restored.json')
    need(prior['status']=='failed'and prior['recovery_gate_pass']is False and
        prior['restoration']==value['restoration']and prior['restoration']['restored']is True and
        prior['restoration']['exact_argv_myds_artifacts_config']is True and
        prior['restoration']['pid']==restored['pid']==before['pid']==31510 and
        all(restored[k]==before[k]for k in archive.IDENTITY_KEYS),'Exact original failure/finally31510 restored identity required')
    for name in ('before.json','restored.json','stage-inputs.json','steps.json','recovery-arms.json'):
        path=oldroot/name;need(path.is_file(),'Required original prefix metadata missing');frozen[str(path)]=sha(path)
    executed=value['executed_archive'];source=reference(executed['source'],frozen);artifacts=reference(executed['artifacts'],frozen)
    bindings=reference(executed['bindings'],frozen);inputmap=reference(executed['stage_inputs'],frozen)
    need(all(p.resolve().parent==Path(executed['directory']).resolve()for p in (source,artifacts,bindings,inputmap))and
        inputmap.resolve()==(Path(executed['directory'])/'stage-inputs.json').resolve()and
        archive.load(inputmap)==archive.load(oldroot/'stage-inputs.json'),'Original complete executed input map')
    checked=archive.executed_archives(oldroot,source,artifacts,bindings);index=checked['index']
    raw=value['raw_archive'];raw_tar=reference(raw['archive'],frozen);raw_json=reference(raw['manifest'],frozen)
    need(raw_tar.resolve().parent==raw_json.resolve().parent==Path(raw['directory']).resolve(),'Original raw archive directory binding')
    raw_manifest=archive.load(raw_json)
    need(raw_manifest['logical_files']==len(raw_manifest['files'])and raw_manifest.get('virtual_kv_files')==0,
        'Complete physical original raw archive required')
    raw_files={str(Path(v['source_path']).resolve()):v for v in raw_manifest['files']}
    need(len(raw_files)==len(raw_manifest['files']),'Unique original raw source paths required')
    records=value['ordered_records'];expected=matrix_records(matrix[:PREFIX_NATIVES])
    need(len(records)==10 and len({v['label']for v in records})==10 and all(type(v['setting'])is int for v in records)and
        all({k:v[k]for k in ('label','scenario','temperature','setting')}==e for v,e in zip(records,expected))and
        archive.load(oldroot/'recovery-arms.json')==records,'Exact original ordered10-native records; duplicates/skips forbidden')
    steps=archive.load(oldroot/'steps.json');native_steps=[s for s in steps if s['name']in {x['label']for x in expected}]
    need([s['name']for s in native_steps]==[x['label']for x in expected]and all(s['returncode']==0 for s in native_steps),
        'All retained native executions must actually have completed before the original reader failure')
    oldstage=Path(native_steps[0]['argv'][0]).resolve().parent.parent
    canonical={str(Path(p).resolve()):v for p,v in index.items()}
    for path in (*IDENTITY_PATHS,*NATIVE_SOURCES):
        need(canonical[str(oldstage/path)]['sha256']==sha(args.stage/path),'Native source or binary changed across continuation: '+path)
    for record,step in zip(records,native_steps):
        directory=Path(record['raw_directory']);path=directory/'summary.json';native=archive.load(path)
        need(directory.resolve()==(oldroot/'arms'/record['label']).resolve()and sha(path)==record['native_summary_sha256']and
            raw_files[str(path.resolve())]['sha256']==record['native_summary_sha256']and
            Path(step['argv'][0]).resolve().parent.parent==oldstage and
            native['status']=='complete'and native['fullmodel_recovery_within_arm_completed']is True and
            native['scenario']==record['scenario']and native['workspace_mode']==record['setting']and
            math.isclose(native['temperature'],float(record['temperature']),abs_tol=1e-6)and native['recovery_tokens']==8,
            'Original native raw/summary/archive/actualargv binding differs')
    pair_specs=[dict(pair=original.pair_name(matrix[i]['scenario'],matrix[i]['temperature']),
        scenario=matrix[i]['scenario'],temperature=matrix[i]['temperature'])for i in range(0,10,2)]
    reviews=value['ordered_pair_reviews'];need(len(reviews)==5 and
        all({k:v[k]for k in ('pair','scenario','temperature')}==e for v,e in zip(reviews,pair_specs)),
        'Exactly five complete ordered original-raw pair reviews')
    for item in reviews:
        report=archive.load(reference(item['report'],frozen))
        need(report['evidence_valid']is True and report['correctness_pass']is True and report['performance_measured']is False and
            report['scenario']==item['scenario']and math.isclose(report['temperature'],float(item['temperature']),abs_tol=1e-6)and
            report['recovery_tokens']==8,'Every retained pair must pass independent complete original-raw review')
    replay=archive.load(reference(value['independent_replay_report'],frozen))
    need(replay['evidence_valid']is True and replay['prefix_gate_pass']is True and replay['completed_native']==10 and
        replay['valid_complete_pairs']==5,'Complete independent original prefix replay report required')
    for key in ('native_identity','production_environment','model','model_header_sha256','component_review_sha256',
        'original_failed_window','original_summary','original_before','original_restored','restoration',
        'executed_archive','raw_archive','ordered_records','ordered_pair_reviews','pair_reader_sources'):
        need(replay[key]==value[key],'Independent replay belongs to a different original prefix: '+key)
    sources=value['pair_reader_sources']
    for key in ('cancel','failure_v1','failure_v2'):reference(sources[key],frozen)
    need(sources['failure_v2']['sha256']==sha(args.stage/V2_PAIR_READER),'Actual new pair reader must equal prefix v2 oracle')
    return dict(prefix_native=10,prefix_pairs=5,next_pair='allocation-t0p7',original_failed_window=str(oldroot),
        original_restored_pid=31510,executed_original_inputs_verified=len(index),native_identity=identity,
        executed_archive_verification={k:v for k,v in checked.items()if k not in ('small','index')},
        independent_replay_report=value['independent_replay_report'])


class RecoveryContinuation(original.RecoveryWindow):
    def preflight(self):
        super().preflight()
        need(str(Path(__file__).resolve())in self.input_hashes,'Execute frozen continuation controller')
        for name in (V2_PAIR_READER,'tools/review_moe_workspace_fullmodel_promptclear.py'):
            need(str(self.a.stage/name)in self.input_hashes,'Required actual CPU reader must be frozen')
        need(sha(self.a.prefix_manifest)==self.a.prefix_manifest_sha256,'Explicit immutable prefix manifest SHA differs')
        prefix=archive.load(self.a.prefix_manifest)
        self.prefix_review=validate_prefix(prefix,self.a,self.before,self.summary,self.matrix,self.input_hashes)
        self.input_hashes[str(self.a.prefix_manifest)]=self.a.prefix_manifest_sha256
        self.prefix=prefix;self.remaining_matrix=self.matrix[10:]
        need(len(self.remaining_matrix)==18 and self.remaining_matrix[0]==dict(scenario='allocation',temperature='0.7',setting=0),
            'Only the unexecuted frozen suffix may run')
        self.summary.update(mode='recovery_continuation',continuation_from_native=10,continuation_native_required=18,
            continuation_pairs_required=9,prefix_manifest=dict(path=str(self.a.prefix_manifest),sha256=self.a.prefix_manifest_sha256),
            prefix_review=self.prefix_review,native_identity=prefix['native_identity'],recovery_gate_pass=False)
        write_json(self.a.out/'stage-inputs.json',self.input_hashes)
        write_json(self.a.out/'continuation-matrix.json',self.remaining_matrix)
        write_json(self.a.out/'prefix-review.json',self.prefix_review)

    def reader_command(self,scenario,off,on,output):
        need(scenario not in ('deterministic','async'),'Already completed cancel pairs must never rerun')
        return ['/usr/bin/python3',self.a.stage/V2_PAIR_READER,'--off',off,'--on',on,
            '--warm-request',self.requests['warm'],'--failure-request',self.requests['failure'],
            '--recover-request',self.requests['recover'],'--out',output]

    def component(self):
        self.summary.update(recovery_run=True,fullmodel_run=True,performance_measured=False)
        for name in ('arms','arm-environments','reports'):(self.a.out/name).mkdir()
        for index in range(0,18,2):
            a,b=self.remaining_matrix[index:index+2]
            need(a['scenario']==b['scenario']and a['temperature']==b['temperature']and[a['setting'],b['setting']]==[0,1],
                'Exact off/on suffix pair order')
            scenario,temperature=a['scenario'],a['temperature'];pair=original.pair_name(scenario,temperature);directories=[]
            for setting in (0,1):
                label=pair+('-on'if setting else'-off');directory=self.a.out/'arms'/label
                need(not directory.exists(),'Fresh native process/context and original new raw directory required')
                env=dict(self.environment,**{FLAG:str(setting),CLEAR_FLAG:str(setting)})
                actual={k:v for k,v in env.items()if k.startswith('MYDS_')}
                need(actual==dict(self.before['effective_myds_env'],**{FLAG:str(setting),CLEAR_FLAG:str(setting)}),'Complete production environment')
                write_json(self.a.out/'arm-environments'/(label+'.json'),actual)
                self.run_command(label,self.native_command(scenario,temperature,directory),env=env,timeout=1800)
                path=directory/'summary.json';record=archive.load(path)
                need(record.get('status')=='complete'and record.get('fullmodel_recovery_within_arm_completed')is True and
                    record.get('scenario')==scenario and record.get('workspace_mode')==setting and
                    type(record.get('temperature'))in (int,float)and math.isclose(record['temperature'],float(temperature),abs_tol=1e-6)and
                    record.get('recovery_tokens')==8 and record.get('same_context')is True and
                    record.get('rng_manually_reset')is False and record.get('extra_gpu_waits_inserted')==0,'Actual exact recovery native scope')
                directories.append(directory);self.arm_records.append(dict(label=label,scenario=scenario,temperature=temperature,
                    setting=setting,raw_directory=str(directory),native_summary_sha256=sha(path)))
                write_json(self.a.out/'recovery-arms.json',self.arm_records)
            output=self.a.out/'reports'/(pair+'.json')
            self.run_command(pair+'-reader',self.reader_command(scenario,*directories,output),
                env=dict(self.environment,**{FLAG:'0',CLEAR_FLAG:'0'}),timeout=1200)
            report=archive.load(output)
            need(report.get('evidence_valid')is True and report.get('correctness_pass')is True and report.get('performance_measured')is False and
                report.get('scenario')==scenario and type(report.get('temperature'))in (int,float)and
                math.isclose(report['temperature'],float(temperature),abs_tol=1e-6)and report.get('recovery_tokens')==8,'Independent v2 full-raw suffix pair gate')
            self.pair_reports.append(dict(pair=pair,scenario=scenario,temperature=temperature,path=str(output),sha256=sha(output),evidence_valid=True,correctness_pass=True))
            write_json(self.a.out/'recovery-pairs.json',self.pair_reports)
        need(len(self.arm_records)==18 and len(self.pair_reports)==9,'All18 suffix natives and9pairs required')
        self.summary.update(continuation_pairs_pass=True,recovery_executions_completed=18,recovery_pair_reports_completed=9,
            total_matrix_coverage_with_prefix=28,combined_independent_review_required=True)

    def run(self):
        code=workspace.WorkspaceWindow.run(self)
        if self.a.prepare:return code
        passed=(code==0 and'error'not in self.summary and self.summary.get('status')=='component_complete_review_required'and
            self.summary.get('continuation_pairs_pass')is True and self.summary.get('restoration',{}).get('restored')is True and
            len(getattr(self,'arm_records',[]))==18 and len(getattr(self,'pair_reports',[]))==9)
        result=dict(schema_version=1,implementation_version=4,mode='recovery_continuation',evidence_valid=passed,
            continuation_gate_pass=passed,recovery_gate_pass=False,production_adoption=False,performance_measured=False,
            recovery_plan_sha256=self.summary.get('recovery_plan_sha256'),component_review_sha256=self.summary.get('component_review_sha256'),
            prefix_manifest=self.summary.get('prefix_manifest'),prefix_review=getattr(self,'prefix_review',None),
            native_identity=self.summary.get('native_identity'),production_environment=self.before['effective_myds_env']if self.before else None,
            model=self.summary.get('model'),model_header_sha256=self.summary.get('model_header_sha256'),
            original_pid=self.before['pid']if self.before else None,restoration=self.summary.get('restoration'),
            arm_executions=getattr(self,'arm_records',[]),pair_reports=getattr(self,'pair_reports',[]),
            requires_combined_independent_28_native_14_pair_review=True,
            decision='ready_for_combined_independent_raw_review'if passed else'reject_failed_or_incomplete_suffix')
        write_json(self.a.out/'recovery-continuation-review.json',result)
        self.summary.update(status='continuation_passed_review_required'if passed else'failed',continuation_gate_pass=passed,recovery_gate_pass=False)
        write_json(self.a.out/'summary.json',self.summary)
        self.event('recovery_continuation_decision_complete',status=self.summary['status'],continuation_gate_pass=passed,recovery_gate_pass=False)
        return 0 if passed else 1


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','gold-stage','model','out','plan','inputs','component-review','recovery-plan','prefix-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prefix-manifest-sha256',required=True);p.add_argument('--port',type=int,default=8187)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    return p.parse_args(argv)


def main():
    a=arguments();need(sys.platform=='darwin','macOS required')
    for name in ('stage','gold_stage','model','out','plan','inputs','component_review','recovery_plan','prefix_manifest'):
        path=getattr(a,name);need(path.is_absolute(),'Absolute paths required');setattr(a,name,path.resolve())
    need(all(Path('/tmp').resolve()in stage.parents for stage in (a.stage,a.gold_stage)),'Separate tmp stages required')
    a.out.mkdir(parents=True,exist_ok=False);os.chmod(a.out,0o700);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with(a.out/'runner.log').open('ab',buffering=0)as log,open(os.devnull,'rb')as null:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    return RecoveryContinuation(a).run()


if __name__=='__main__':raise SystemExit(main())
