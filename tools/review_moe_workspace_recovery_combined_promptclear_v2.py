#!/usr/bin/env python3
"""Independent two-window10+18 native recovery replay; no model/GPU execution.

The first window remains failed at its original reader. A validated prefix and
the exact unexecuted suffix must cover the frozen matrix once, without repeats.
Every complete original pair is re-read and compared here before the combined
gate becomes true. Original single-window reports are never rewritten.
"""
import argparse,copy,hashlib,importlib.util,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True

def sibling(name):
    s=importlib.util.spec_from_file_location('combined_'+name,Path(__file__).with_name(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prefix=sibling('review_moe_workspace_recovery_prefix_promptclear')
base=prefix.base;seq=base.seq;need=seq.need;load=seq.load;sha=seq.sha
revised=prefix.revised
IDENTITY=(*base.CORE,'bin/myds.metallib',*(str(Path('bin')/n)for n in base.NATIVES))
SOURCE_DEPS=('tools/test_moe_workspace_cancel_promptclear.c','tools/test_moe_workspace_recovery_promptclear.c',
 'tools/moe_prompt_snapshot_promptclear.inc','tools/moe_workspace_layout_promptclear.inc',
 'tools/bench_moe_workspace_fullmodel_promptclear.c','tools/bench_moe_workspace_fullmodel.c',
 'tools/moe_workspace_sequence_state_localgrow.inc','tools/moe_workspace_memory.inc')
MATRIX=base.matrix()

def descriptor(p):return prefix.descriptor(p)
def write_new(p,v):return prefix.write_new(p,v)
def check_reference(d):
    need(type(d)is dict and set(d)=={'path','sha256','bytes'}and type(d['bytes'])is int and d['bytes']>=0,'Exact reference path/SHA/extent')
    p=Path(d['path']);need(p.is_absolute()and p.stat().st_size==d['bytes']and sha(p)==d['sha256'],'Referenced original complete bytes differ');return p

def complete_coverage(old,new):
    prefix.prefix_coverage(old)
    need(type(new)is list and len(new)==18,'Exactly18 new native executions')
    all_records=old+new
    need(all(type(x['setting'])is int and x['setting']in (0,1)and type(x['scenario'])is str and type(x['temperature'])is str for x in all_records),'Strict actual matrix field types')
    need([(x['scenario'],x['temperature'],x['setting'])for x in all_records]==[(x['scenario'],x['temperature'],x['setting'])for x in MATRIX],
         'Two-window ordered28 matrix has no repeats, omissions or cherry-picked replacements')
    need([x['label']for x in all_records]==[base.pair_name(x['scenario'],x['temperature'])+('-on'if x['setting']else'-off')for x in MATRIX]and len({x['label']for x in all_records})==28,'Exact unique28 labels')
    return all_records

def source_windows_gate(windows):
    need(type(windows)is list and len(windows)==2 and len({w['window']for w in windows})==2 and
         all(type(w['window'])is str and Path(w['window']).is_absolute()for w in windows),'Two distinct absolute original source windows')
    need([(w['native_invocations'],w['pair_readers_replayed'])for w in windows]==[(10,5),(18,9)]and
         [w['status']for w in windows]==['failed','continuation_passed_review_required'],'Preserve failed-prefix and successful-suffix coverage separately')
    for w in windows:
        need(type(w['input_map_sha256'])is str and len(w['input_map_sha256'])==64 and all(c in '0123456789abcdef'for c in w['input_map_sha256'])and
             w['restoration']['restored']is True and w['restoration']['exact_argv_myds_artifacts_config']is True,'Each source window restored and binds exact frozen input-map')
        for name in ('source_archive','artifacts_archive','archive_bindings'):
            d=w[name];need(type(d)is dict and set(d)=={'path','sha256','bytes'}and type(d['path'])is str and Path(d['path']).is_absolute()and
                type(d['bytes'])is int and d['bytes']>=0 and type(d['sha256'])is str and len(d['sha256'])==64,'Bounded source-window archive reference')
    return True

def source_identity(window,executed):
    archive=seq.executed_archives(window,executed/'source.tar.gz',executed/'artifacts.tar.gz',executed/'archive-bindings.json')
    idx,small=archive['index'],archive['small'];plan=load(window/'protocol.json');base.frozen_plan_gate(plan,idx,small)
    steps=load(window/'steps.json');native=next(x for x in steps if '-reader'not in x['name'])
    candidate=Path(native['argv'][0]).parent.parent
    identity={k:idx[str(candidate/k)]['sha256']for k in IDENTITY}
    roots={Path(p).parent.parent for p in idx if p.endswith('/src/myds_v41.c')}
    gold=[p for p in roots if idx.get(str(p/'bin/myds'),{}).get('sha256')==seq.PRODUCTION_BINARY and all(idx.get(str(p/k),{}).get('sha256')==v for k,v in seq.GOLD_SOURCE_HASHES.items())]
    need(len(gold)==1,'All91 exact-production gold inputs in each executed source archive')
    return archive,candidate,identity

def restored_identity(root,*,continuation):
    before,after,s=[load(root/n)for n in ('before.json','restored.json','summary.json')]
    need(all(before[k]==after[k]for k in seq.IDENTITY_KEYS)and before['health']==after['health']=={'status':'ok'},'Exact recorded per-window restoration')
    need(s['restoration']==dict(restored=True,pid=after['pid'],exact_argv_myds_artifacts_config=True)and
         s['plan_sha256']==s['recovery_plan_sha256']==seq.PLAN_SHA and s['performance_measured']is False and s['production_adoption']is False,'Restoration/plan/nonperformance scope')
    need(before['binary_sha256']==seq.PRODUCTION_BINARY and before['metallib_sha256']==seq.PRODUCTION_LIBRARY and
         sha(root/'production-artifacts/myds')==seq.PRODUCTION_BINARY and sha(root/'production-artifacts/myds.metallib')==seq.PRODUCTION_LIBRARY and
         sha(root/'original-myds.conf')==before['config_sha256']and load(root/'effective-env.json')==before['effective_myds_env'],'Actual production artifacts/config/environment')
    if continuation:
        need(s['mode']=='recovery_continuation'and s['status']=='continuation_passed_review_required'and s['continuation_gate_pass']is True and
             s['recovery_gate_pass']is False and s['recovery_executions_completed']==18 and s['recovery_pair_reports_completed']==9 and s['total_matrix_coverage_with_prefix']==28,
             'Suffix completes only18/9 and cannot self-certify total recovery')
    else:prefix.recorded_prefix_identity(root)
    return before,after,s

def commands(candidate,root,requests,model,v2):
    result=[]
    for s in base.SCENARIOS:
        for t in base.TEMPERATURES:
            pair=base.pair_name(s,t);directories=[]
            for f in (0,1):
                d=root/'arms'/(pair+('-on'if f else'-off'));directories.append(d)
                native=base.NATIVES[0]if s in ('deterministic','async')else base.NATIVES[1]
                argv=[candidate/'bin'/native,'--run','--model',model,'--out-dir',d,'--scenario',s,'--temperature',t,'--recover-tokens','8']
                if s in ('deterministic','async'):argv+=['--warm-request',requests['warm'],'--cancel-request',requests['cancel'],'--recover-request',requests['recover'],'--checkpoint','1','--delay-ms','0']
                else:argv+=['--warm-request',requests['warm'],'--failure-request',requests['failure'],'--recover-request',requests['recover']]
                result.append((pair+('-on'if f else'-off'),list(map(str,argv))))
            name='test_moe_workspace_cancel_promptclear.py'if s in ('deterministic','async')else('test_moe_workspace_recovery_promptclear_v2.py'if v2 else'test_moe_workspace_recovery_promptclear.py')
            argv=['/usr/bin/python3',candidate/'tools'/name,'--off',directories[0],'--on',directories[1],'--recover-request',requests['recover'],'--out',root/'reports'/(pair+'.json'),'--warm-request',requests['warm']]
            argv+=['--cancel-request',requests['cancel']]if s in ('deterministic','async')else['--failure-request',requests['failure']]
            # Continuation reader preserves its own thin-controller option order.
            if v2 and s not in ('deterministic','async'):
                argv=['/usr/bin/python3',candidate/'tools'/name,'--off',directories[0],'--on',directories[1],'--warm-request',requests['warm'],'--failure-request',requests['failure'],'--recover-request',requests['recover'],'--out',root/'reports'/(pair+'.json')]
            result.append((pair+'-reader',list(map(str,argv))))
    return result

def invocation_gate(root,candidate,requests,model,*,continuation):
    expected=commands(candidate,root,requests,model,continuation)[15:]if continuation else commands(candidate,root,requests,model,False)[:15]
    steps=load(root/'steps.json');need([(x['name'],x['argv'])for x in steps]==expected,'Exact original native/reader command sequence and source-stage paths')
    for i,step in enumerate(steps):
        expected_exit=1 if not continuation and i==14 else 0
        need(step['returncode']==expected_exit and type(step['seconds'])in (int,float)and math.isfinite(step['seconds'])and step['seconds']>=0,'Original exit/duration record preserved')
    events=seq.rows(root/'events.jsonl');need([(x['name'],x['argv'])for x in events if x['kind']=='step_begin']==expected,'Actual command-begin events match original records')
    need(not any(x['kind']in ('monitor_failure','restoration_failure')for x in events),'No competition/restoration failure')
    need(events[-1]['kind']==('recovery_continuation_decision_complete'if continuation else'recovery_decision_complete'),'Exact original decision terminal event')
    return dict(native_invocations=18 if continuation else 10,pair_reader_invocations=9 if continuation else 5)

def review(a):
    manifest_path=a.prefix_manifest.resolve(strict=True);pre=load(manifest_path)
    need(sha(manifest_path)==a.prefix_manifest_sha256 and pre['evidence_valid']is True and pre['prefix_gate_pass']is True and pre['recovery_gate_pass']is False and
         pre['completed_native']==10 and pre['valid_complete_pairs']==5 and pre['plan_sha256']==pre['recovery_plan_sha256']==seq.PLAN_SHA,'Exact independently sealed original prefix')
    check_reference(pre['independent_replay_report']);oldroot=Path(pre['original_failed_window']);newroot=a.root.resolve(strict=True)
    need(oldroot!=newroot and oldroot.is_dir(),'Two distinct retained original windows')
    oldexecuted=Path(pre['executed_archive']['directory']);newexecuted=a.executed.resolve(strict=True)
    oldbefore,oldafter,oldsummary=restored_identity(oldroot,continuation=False)
    before,after,summary=restored_identity(newroot,continuation=True)
    need(before['pid']==oldafter['pid']==31510 and all(before[k]==oldafter[k]for k in seq.IDENTITY_KEYS),'Exact restoration identity connects both windows without intervening replacement')
    need(before['effective_myds_env']==pre['production_environment'] and summary['model']==oldsummary['model']==pre['model']and summary['model_header_sha256']==oldsummary['model_header_sha256']==pre['model_header_sha256']and
         summary['component_review_sha256']==oldsummary['component_review_sha256']==pre['component_review_sha256'],'Same original51/model/header/component across both windows')
    env=before['effective_myds_env'];need(len(env)==51 and env.get(seq.FLAG,'0')==env.get(seq.CLEAR_FLAG,'0')=='0','Original51 settings and candidate defaultoff')
    checked=[source_identity(oldroot,oldexecuted),source_identity(newroot,newexecuted)]
    need(checked[0][2]==checked[1][2]==pre['native_identity']['candidate'],'Exact native/core4/library bytes across both executed archives')
    for path in SOURCE_DEPS:need(checked[0][0]['index'][str(checked[0][1]/path)]['sha256']==checked[1][0]['index'][str(checked[1][1]/path)]['sha256'],'Actual native C/include dependency unchanged across continuation: '+path)
    for archive,candidate,_ in checked:
        for name,mod in [('test_moe_workspace_cancel_promptclear.py',base.cancel),('review_moe_workspace_fullmodel_promptclear.py',seq)]:
            need(archive['index'][str(candidate/'tools'/name)]['sha256']==sha(Path(mod.__file__)),'Exact frozen unchanged low-level replay source '+name)
    need(checked[1][0]['index'][str(checked[1][1]/'tools/test_moe_workspace_recovery_promptclear_v2.py')]['sha256']==sha(Path(revised.__file__))==pre['pair_reader_sources']['failure_v2']['sha256'],'Frozen strict v2 reader binds prefix and suffix')
    continuation_top=load(newroot/'recovery-continuation-review.json')
    need(continuation_top['evidence_valid']is True and continuation_top['continuation_gate_pass']is True and continuation_top['recovery_gate_pass']is False and
         continuation_top['requires_combined_independent_28_native_14_pair_review']is True and continuation_top['prefix_manifest']==summary['prefix_manifest']==dict(path=str(manifest_path),sha256=a.prefix_manifest_sha256),
         'Original suffix top cannot claim the combined gate and binds exact prior prefix')
    oldarms=load(oldroot/'recovery-arms.json');newarms=load(newroot/'recovery-arms.json');arms=complete_coverage(oldarms,newarms)
    need(oldarms==pre['ordered_records']and newarms==continuation_top['arm_executions'],'Exact retained arm records and suffix top')
    pair_reports=load(newroot/'recovery-pairs.json');need(len(pair_reports)==9 and pair_reports==continuation_top['pair_reports']and
         [(r['scenario'],r['temperature'])for r in pair_reports]==[(s,t)for s in base.SCENARIOS for t in base.TEMPERATURES][5:],'Exact9 previously unexecuted suffix pairs')
    requests=[]
    for archive,_,_ in checked:
        q={};plan=load((oldroot,newroot)[len(requests)]/'protocol.json')
        for name,stem in (('warm','pf7k'),('failure','pf16k'),('cancel','pf16k'),('recover','pf7k')):
            paths=[p for p in archive['small']if Path(p).name==stem+'.json'and archive['index'][p]['sha256']==plan['fullmodel']['requests'][stem]['sha256']]
            need(len(paths)==1 and sha(Path(paths[0]))==archive['index'][paths[0]]['sha256'],'Exact retained original request bytes');q[name]=Path(paths[0])
        requests.append(q)
    invocation=[invocation_gate(oldroot,checked[0][1],requests[0],summary['model']['path'],continuation=False),invocation_gate(newroot,checked[1][1],requests[1],summary['model']['path'],continuation=True)]
    # The lossless archives preserve original failures/stdout/metadata as well as
    # every raw file; verify current originals against each sealed logical SHA.
    raw_reports=[]
    for root,rawdir in ((oldroot,Path(pre['raw_archive']['directory'])),(newroot,a.raw_archive.resolve(strict=True))):
        helper=prefix.module('combined_lossless_'+str(len(raw_reports)),a.archive_helper.resolve(strict=True));m=load(rawdir/'raw-files.json');rv=helper.verify_archive(rawdir/'raw.tar.gz',m)
        need(rv['verified']is True and len(m['windows'])==1 and Path(m['windows'][0]['source_root'])==root,'Lossless archive binds exact original window')
        need({r['source_path']for r in m['files']}=={str(p)for p in root.rglob('*')if p.is_file()},'Complete original file-set retained')
        for record in m['files']:need(sha(Path(record['source_path']))==record['sha256'],'All current original bytes still equal sealed archive')
        raw_reports.append(dict(directory=str(rawdir),archive=descriptor(rawdir/'raw.tar.gz'),manifest=descriptor(rawdir/'raw-files.json'),verification=rv))
    results=[];records=[]
    for i,(s,t) in enumerate((s,t)for s in base.SCENARIOS for t in base.TEMPERATURES):
        root=oldroot if i<5 else newroot;req=requests[0 if i<5 else 1];pair=base.pair_name(s,t);ds=[root/'arms'/(pair+'-'+arm)for arm in ('off','on')]
        for f,d in enumerate(ds):
            rec=arms[2*i+f];need(rec['raw_directory']==str(d)and rec['native_summary_sha256']==sha(d/'summary.json'),'Original raw/native-summary identity')
            base.two_flag_environment(load(root/'arm-environments'/(pair+('-on'if f else'-off')+'.json')),env,f)
        result=base.cancel.review(*ds,req['warm'],req['cancel'],req['recover'])if s in ('deterministic','async')else revised.review(*ds,{k:req[k]for k in ('warm','failure','recover')})
        need(result['evidence_valid']is True and result['correctness_pass']is True,'Complete independent14pair raw replay')
        report=a.out.parent/('combined-'+pair+'.json');write_new(report,result)
        records.append(dict(pair=pair,scenario=s,temperature=t,path=str(report),sha256=sha(report),evidence_valid=True,correctness_pass=True,source_window=str(root)))
        results.append(dict(pair=pair,replay=result))
    need(len(records)==14 and len({x['pair']for x in records})==14,'Full14 pair oracle coverage')
    if a.model:
        st=a.model.stat();meta=dict(path=str(a.model),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns)
        need(meta==summary['model'],'Actual original model stat identity')
        with a.model.open('rb')as f:need(hashlib.sha256(f.read(1048576)).hexdigest()==summary['model_header_sha256'],'Actual original model first1MiB identity')
    source_windows=[]
    for i,(root,ex,s,inv)in enumerate(zip((oldroot,newroot),(oldexecuted,newexecuted),(oldsummary,summary),invocation)):
        source_windows.append(dict(window=str(root),status=s['status'],native_invocations=inv['native_invocations'],pair_readers_replayed=5 if i==0 else 9,input_map_sha256=sha(root/'stage-inputs.json'),
            source_archive=descriptor(ex/'source.tar.gz'),artifacts_archive=descriptor(ex/'artifacts.tar.gz'),archive_bindings=descriptor(ex/'archive-bindings.json'),restoration=s['restoration']))
    source_windows_gate(source_windows)
    return dict(schema_version=1,evidence_valid=True,recovery_gate_pass=True,correctness_pass=True,implementation_version=4,
        plan_sha256=seq.PLAN_SHA,recovery_plan_sha256=seq.PLAN_SHA,component_review_sha256=summary['component_review_sha256'],
        performance_measured=False,production_adoption=False,decision='pass_independent_two_window10plus18_exact28native14pair_recovery',
        native_invocations=28,pair_readers_replayed=14,source_windows=source_windows,arm_executions=arms,pair_reports=records,
        original_failed_top=descriptor(oldroot/'recovery-review.json'),original_top=descriptor(newroot/'recovery-continuation-review.json'),
        prefix_manifest=descriptor(manifest_path),production_environment=env,model=summary['model'],model_header_sha256=summary['model_header_sha256'],
        native_identity=pre['native_identity'],candidate_identity=pre['native_identity']['candidate'],before_pid=oldbefore['pid'],restored_pid=after['pid'],restoration=summary['restoration'],
        recorded_restoration_exact=True,live_production_sampled=False,all91_exact_production_gold_source_inputs=True,
        full_bytes=base.byte_totals(results),raw_archives=raw_reports,pair_replays=results,
        source_and_artifact_archives=[{k:v for k,v in q[0].items()if k not in ('small','index')}for q in checked],
        logical_frozen_inputs_verified=[len(q[0]['index'])for q in checked],
        limitations=['Original window remains failed at its unchanged v1 reader; original10 native outputs are retained, corrected strict v2 independently replays the failed fifth pair. Only18 previously unexecuted natives are added in a separate window. No native is selected, repeated, replaced or hidden.',
                    'NULL injection is not system-memory-pressure OOM. Partial interrupted live capacity, unpublished snapshots, allocator padding and persistent disk-cache contents are excluded exactly as in original raw readers.',
                    'This gate proves defined recovery state only, with no performance or production adoption claim. Current live restoration requires a separate read-only audit.'])

def small_summary(result,long_path):
    # Snapshot slots are integer-keyed in the verified in-memory oracle. JSON
    # represents object keys as strings. Validate the exact JSON data model,
    # rejecting duplicate keys/nonfinite values via the same strict parser;
    # the actual written long bytes remain independently bound by SHA/extent.
    normalized=seq.parse(json.dumps(result,sort_keys=True,allow_nan=False))
    need(load(long_path)==normalized and result.get('evidence_valid')is True and result.get('recovery_gate_pass')is True and result.get('correctness_pass')is True and
         result.get('implementation_version')==4 and result.get('plan_sha256')==result.get('recovery_plan_sha256')==seq.PLAN_SHA and
         result.get('performance_measured')is False and result.get('production_adoption')is False and
         result.get('native_invocations')==28 and result.get('pair_readers_replayed')==14 and result['full_bytes']['pair_readers_replayed']==14 and
         [(x['native_invocations'],x['pair_readers_replayed'])for x in result['source_windows']]==[(10,5),(18,9)],'Only complete independently replayed combined recovery may produce small gate')
    source_windows_gate(result['source_windows'])
    complete_coverage(result['arm_executions'][:10],result['arm_executions'][10:])
    need(len(result['pair_reports'])==14 and [(v['scenario'],v['temperature'])for v in result['pair_reports']]==[(s,t)for s in base.SCENARIOS for t in base.TEMPERATURES]and
         all(v.get('evidence_valid')is True and v.get('correctness_pass')is True for v in result['pair_reports']),'Small top retains complete ordered14 independent positive pair decisions')
    top={k:copy.deepcopy(v)for k,v in result.items()if k not in ('pair_replays','raw_archives','source_and_artifact_archives','full_bytes','logical_frozen_inputs_verified')}
    top.update(independent_fullraw_replay_completed=True,long_report=descriptor(long_path),independent_review=dict(full_bytes=result['full_bytes'],source_windows=result['source_windows'],source_and_artifact_archives=result['source_and_artifact_archives'],logical_frozen_inputs_verified=result['logical_frozen_inputs_verified'],all91_exact_production_gold_source_inputs=True,recorded_restoration_exact=True,live_production_sampled=False,limitations=result['limitations']))
    need(len(json.dumps(top,allow_nan=False).encode())<=100*1024,'Combined independent small prerequisite bounded100KiB');return top

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','executed','raw-archive','prefix-manifest','archive-helper','out','summary-out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--prefix-manifest-sha256',required=True);p.add_argument('--model',type=Path);a=p.parse_args()
    need(a.out.is_absolute()and a.summary_out.is_absolute()and a.out.parent.is_dir(),'New absolute output files in existing independent report directory')
    try:value=review(a)
    except Exception as e:
        write_new(a.out,dict(schema_version=1,evidence_valid=False,recovery_gate_pass=False,correctness_pass=False,implementation_version=4,plan_sha256=seq.PLAN_SHA,performance_measured=False,production_adoption=False,error=repr(e)));raise
    write_new(a.out,value);write_new(a.summary_out,small_summary(value,a.out));print(json.dumps(dict(evidence_valid=True,recovery_gate_pass=True,native_invocations=28,pair_readers_replayed=14),sort_keys=True))
if __name__=='__main__':main()
