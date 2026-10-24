#!/usr/bin/env python3
"""Independently seal the exact10-native/5-pair prefix of failed recovery1.

The failed original window remains failed. This CPU-only report is a continuation
prerequisite, never a complete28-native recovery gate or performance result.
"""
import argparse,hashlib,importlib.util,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sibling(name):return module('prefix_'+name,Path(__file__).with_name(name+'.py'))
base=sibling('review_moe_workspace_recovery_promptclear_artifacts')
revised=sibling('test_moe_workspace_recovery_promptclear_v2')
seq=base.seq;need=seq.need;load=seq.load;sha=seq.sha
PAIRS=[(s,t)for s in base.SCENARIOS for t in base.TEMPERATURES][:5]

def descriptor(p):
    p=Path(p).resolve(strict=True);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def write_new(p,v):
    with Path(p).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def prefix_coverage(arms):
    expected=[(s,t,f)for s,t in PAIRS for f in (0,1)]
    need(type(arms)is list and len(arms)==10 and [(a['scenario'],a['temperature'],a['setting'])for a in arms]==expected,'Exact ordered first10 native executions, no repeat/missing/replacement')
    need([a['label']for a in arms]==[base.pair_name(s,t)+('-on'if f else'-off')for s,t,f in expected],'Exact first10 labels')
    return True

def recorded_prefix_identity(root):
    before,after,summary=[load(root/n)for n in ('before.json','restored.json','summary.json')]
    need(summary['status']=='failed'and summary['recovery_gate_pass']is False and summary['error']=="RuntimeError('allocation-t0-reader failed with return code 1')",'Original reader-failed window preserved')
    need(summary['recovery_executions_required']==28 and summary['recovery_pair_reports_required']==14 and summary['performance_measured']is False and summary['production_adoption']is False,'Prefix is not the complete recovery matrix')
    need(summary['plan_sha256']==summary['recovery_plan_sha256']==seq.PLAN_SHA,'Frozen actual execution plan')
    need(all(before[k]==after[k]for k in seq.IDENTITY_KEYS)and before['health']==after['health']=={'status':'ok'},'Original exact service identity restored')
    need(summary['restoration']==dict(restored=True,pid=after['pid'],exact_argv_myds_artifacts_config=True),'Original finally restoration binding')
    need(before['binary_sha256']==seq.PRODUCTION_BINARY and before['metallib_sha256']==seq.PRODUCTION_LIBRARY and
         sha(root/'production-artifacts/myds')==seq.PRODUCTION_BINARY and sha(root/'production-artifacts/myds.metallib')==seq.PRODUCTION_LIBRARY and
         sha(root/'original-myds.conf')==before['config_sha256'] and load(root/'effective-env.json')==before['effective_myds_env'],'Original production/config/env bytes')
    env=before['effective_myds_env'];need(len(env)==51 and env.get(seq.FLAG,'0')==env.get(seq.CLEAR_FLAG,'0')=='0','Production51 settings and two defaults off')
    return before,after,summary

def review(a):
    root=a.root.resolve(strict=True);executed=a.executed.resolve(strict=True);raw=a.raw_archive.resolve(strict=True)
    before,after,summary=recorded_prefix_identity(root);arms=load(root/'recovery-arms.json');prefix_coverage(arms)
    archive=seq.executed_archives(root,executed/'source.tar.gz',executed/'artifacts.tar.gz',executed/'archive-bindings.json');idx,small=archive['index'],archive['small']
    plan=load(root/'protocol.json');base.frozen_plan_gate(plan,idx,small)
    need(load(root/'recovery-protocol.json')==plan and load(root/'recovery-matrix.json')==base.matrix(),'Exact frozen original28 matrix remains unedited')
    native_steps=[x for x in load(root/'steps.json')if not x['name'].endswith('-reader')]
    need(len(native_steps)==10 and [x['name']for x in native_steps]==[x['label']for x in arms]and all(x['returncode']==0 for x in native_steps),'Original10 native commands all completed')
    candidate=Path(native_steps[0]['argv'][0]).parent.parent
    identity={'candidate':{k:idx[str(candidate/k)]['sha256']for k in (*base.CORE,'bin/myds.metallib',*(str(Path('bin')/n)for n in base.NATIVES))}}
    for native in base.NATIVES:
        need(sha(candidate/'bin'/native)==identity['candidate']['bin/'+native],'Original native still matches archived executable')
    source_roots={Path(p).parent.parent for p in idx if p.endswith('/src/myds_v41.c')}
    gold=[p for p in source_roots if idx.get(str(p/'bin/myds'),{}).get('sha256')==seq.PRODUCTION_BINARY and all(idx.get(str(p/k),{}).get('sha256')==v for k,v in seq.GOLD_SOURCE_HASHES.items())]
    need(len(gold)==1,'Exact production gold91 source/build inputs')
    comp=[p for p in small if idx[p]['sha256']==summary['component_review_sha256']];need(len(comp)==1,'Unique original independent component prerequisite')
    component=seq.parse(small[comp[0]].decode());need(component['implementation_version']==4 and component['plan_sha256']==seq.PLAN_SHA and component['evidence_valid']is True and component['component_gate_pass']is True,'Exact current component gate')
    seq.long_reference_gate(component['long_report'],idx,'prefix component')
    seq.component_identity_gate(component,identity,before['effective_myds_env'],summary['model'],summary['model_header_sha256'])
    helper=module('prefix_lossless_archive_helper',a.archive_helper.resolve(strict=True));manifest=load(raw/'raw-files.json')
    raw_verification=helper.verify_archive(raw/'raw.tar.gz',manifest)
    need(raw_verification['verified']is True and len(manifest['windows'])==1 and Path(manifest['windows'][0]['source_root'])==root,'Lossless archive binds the exact failed original window')
    records=manifest['files'];need({r['source_path']for r in records}=={str(p)for p in root.rglob('*')if p.is_file()},'All original raw and failed-reader files preserved')
    for record in records:need(sha(Path(record['source_path']))==record['sha256'],'Every current original raw payload still matches sealed archive')
    original_pairs=load(root/'recovery-pairs.json');need(len(original_pairs)==4 and [(p['scenario'],p['temperature'])for p in original_pairs]==PAIRS[:4],'Original controller completed only4 pairs')
    failed_original=load(root/'reports/allocation-t0.json');need(failed_original['evidence_valid']is False and failed_original['correctness_pass']is False and failed_original['error']=='Successful warm and recovery prompt extent','Original false-rejection report retained exactly')
    replay=[];pair_records=[];requests={}
    for key,stem in (('warm','pf7k'),('cancel','pf16k'),('failure','pf16k'),('recover','pf7k')):
        matches=[p for p in small if Path(p).name==stem+'.json'and idx[p]['sha256']==plan['fullmodel']['requests'][stem]['sha256']]
        need(len(matches)==1,'Unique frozen original request '+key);requests[key]=Path(matches[0]);need(sha(requests[key])==idx[matches[0]]['sha256'],'Original request bytes still exact')
    for s,t in PAIRS:
        pair=base.pair_name(s,t);ds=[root/'arms'/(pair+'-'+setting)for setting in ('off','on')]
        for f,d in enumerate(ds):
            rec=arms[2*PAIRS.index((s,t))+f];need(rec['raw_directory']==str(d)and rec['native_summary_sha256']==sha(d/'summary.json'),'Original native directory/summary identity')
            base.two_flag_environment(load(root/'arm-environments'/(pair+('-on'if f else'-off')+'.json')),before['effective_myds_env'],f)
        value=base.cancel.review(*ds,requests['warm'],requests['cancel'],requests['recover'])if s in ('deterministic','async')else revised.review(*ds,{k:requests[k]for k in ('warm','failure','recover')})
        need(value['evidence_valid']is True and value['correctness_pass']is True,'Complete independent prefix raw replay')
        report=a.out/(pair+'.json');write_new(report,value);pair_records.append(dict(pair=pair,scenario=s,temperature=t,report=descriptor(report)))
        replay.append(dict(pair=pair,replay=value))
    facts={setting:{phase:{k:v for k,v in load(root/'arms'/('allocation-t0-'+setting)/('prompt-'+phase+'.json')).items()if k in ('cap','mask_bytes','total_bytes','cw_id','owners','views')}for phase in ('before-failure','after-failure','recovery')}for setting in ('off','on')}
    critical=dict(schema_version=1,implementation_version=4,plan_sha256=seq.PLAN_SHA,recovery_plan_sha256=seq.PLAN_SHA,
        evidence_valid=True,prefix_gate_pass=True,recovery_gate_pass=False,performance_measured=False,production_adoption=False,
        completed_native=10,valid_complete_pairs=5,next_pair='allocation-t0p7',remaining_native=18,remaining_pairs=9,
        original_failed_window=str(root),original_summary=descriptor(root/'summary.json'),restoration=summary['restoration'],
        original_before=descriptor(root/'before.json'),original_restored=descriptor(root/'restored.json'),
        ordered_records=arms,ordered_pair_reviews=pair_records,native_identity=identity,
        production_environment=before['effective_myds_env'],model=summary['model'],model_header_sha256=summary['model_header_sha256'],
        component_review_sha256=summary['component_review_sha256'],
        executed_archive=dict(directory=str(executed),bindings=descriptor(executed/'archive-bindings.json'),source=descriptor(executed/'source.tar.gz'),artifacts=descriptor(executed/'artifacts.tar.gz'),stage_inputs=descriptor(executed/'stage-inputs.json')),
        raw_archive=dict(directory=str(raw),archive=descriptor(raw/'raw.tar.gz'),manifest=descriptor(raw/'raw-files.json')),
        pair_reader_sources={'cancel':descriptor(Path(base.cancel.__file__)),'failure_v1':descriptor(Path(base.recovery.__file__)),'failure_v2':descriptor(Path(revised.__file__))})
    long=dict(critical,decision='prefix_only_reader_false_rejection_corrected_without_rerunning_native',
        source_archive_verification={k:v for k,v in archive.items()if k not in ('small','index')},all91_exact_production_gold_source_inputs=True,
        raw_archive_verification=raw_verification,allocation_prompt_facts=facts,
        full_bytes=base.byte_totals(replay),pair_replays=replay,
        source_order={'compact':'localgrow failure returns before prompt replacement; old7168 owners/views retained',
                      'legacy':'factory failure leaves cwNULL; prompt replacement publishes15360 before final !cw rejection; short recovery never shrinks'},
        limits=['Original window remains failed; original controller completed10 native and4 pair reports. Independent v2 replay validates5 pairs only. Complete28 recovery gate is false until exact remaining18 and all14 aggregate raw replays pass.',
                'No GPU/model/service action performed by this reader. NULL fault injection is not actual system-memory-pressure OOM. Defined raw scope and partial-state exclusions remain those of original readers.'])
    long_path=a.out/'prefix-independent-review.json';write_new(long_path,long)
    top=dict(critical,independent_replay_report=descriptor(long_path));write_new(a.out/'prefix-manifest.json',top);return top

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','executed','raw-archive','archive-helper','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=False,exist_ok=False)
    try:result=review(a)
    except Exception as e:
        write_new(a.out/'failure.json',dict(evidence_valid=False,prefix_gate_pass=False,recovery_gate_pass=False,error=repr(e)));raise
    print(json.dumps({k:result[k]for k in ('evidence_valid','prefix_gate_pass','completed_native','valid_complete_pairs','next_pair')},sort_keys=True))
if __name__=='__main__':main()
