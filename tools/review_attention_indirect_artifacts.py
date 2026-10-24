#!/usr/bin/env python3
"""Independent CPU review of single-head indirect-layout raw evidence.

Supports a preserved numerical correctness rejection and a complete correctness
phase followed by either layout-performance acceptance or rejection. Outputs are
compared as actual uint32 words; formal timings are recomputed from CB clocks.
Exit 0 means coherent evidence, not production acceptance. Source/artifact
archives and optional --model strengthen the provenance independently. No GPU,
service request or model execution occurs. --self-test uses small CPU fixtures.
"""
import argparse
import copy
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import tarfile
import tempfile

sys.dont_write_bytecode=True
import review_attention_multihead_artifacts as common
import build_attention_indirect_probe as builder

LAYERS=(2,8,19)
NATIVE='kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512'
INDIRECT='kernel_myds_prefill_attention_indirect_f16_dk512_dv512'
SYMBOLS=(NATIVE,NATIVE,INDIRECT)
OUT_BYTES=2048*131072
require,digest,load=common.require,common.digest,common.load


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def recorded_path(value):
    """Normalize macOS's recorded /tmp alias without requiring live inputs.

    The capture CLI can retain /tmp while stage freezing records realpath
    /private/tmp.  Offline review must compare those names without resolving
    them against a different machine's filesystem.
    """
    path=Path(value)
    if path.is_relative_to('/tmp'):
        path=Path('/private/tmp')/path.relative_to('/tmp')
    return str(path)


def finite_written(path):
    finite,written=True,True
    try:
        import numpy as np
    except ImportError:
        np=None
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            require(len(block)%4==0,'Unaligned F32 payload')
            if np is not None:
                words=np.frombuffer(block,dtype='<u4')
                finite=finite and not bool(np.any((words&0x7f800000)==0x7f800000))
                written=written and not bool(np.any(words==0xa5a5a5a5))
            else:
                for (word,) in struct.iter_unpack('<I',block):
                    finite=finite and (word&0x7f800000)!=0x7f800000
                    written=written and word!=0xa5a5a5a5
    return finite,written


def compare_words(first,second):
    require(first.stat().st_size==second.stat().st_size,'Output comparison length differs')
    count,first_bad,word_offset=0,None,0
    try:
        import numpy as np
    except ImportError:
        np=None
    with first.open('rb') as a,second.open('rb') as b:
        while True:
            x,y=a.read(4*1024*1024),b.read(4*1024*1024)
            if not x:break
            if x!=y:
                if np is not None:
                    bad=np.flatnonzero(np.frombuffer(x,dtype='<u4')!=np.frombuffer(y,dtype='<u4'))
                    count+=len(bad)
                    if first_bad is None:first_bad=word_offset+int(bad[0])
                else:
                    for index,(u,v) in enumerate(zip(struct.iter_unpack('<I',x),struct.iter_unpack('<I',y))):
                        if u!=v:
                            count+=1
                            if first_bad is None:first_bad=word_offset+index
            word_offset+=len(x)//4
    return count,first_bad


def checked_files(directory,manifest):
    files={};cache={}
    for entry in manifest['files']:
        name=entry['path'];require(name not in files,'Duplicate logical artifact path')
        path=common.path_inside(directory,name);st=path.stat()
        key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=digest(path)
        require(st.st_size==entry['bytes'] and cache[key]==entry['sha256'],'Raw SHA/bytes mismatch: '+name)
        if entry.get('storage_alias_of'):
            alias=common.path_inside(directory,entry['storage_alias_of']).stat()
            require((st.st_dev,st.st_ino)==(alias.st_dev,alias.st_ino),'Declared hardlink differs')
        files[name]=(path,entry)
    return files,len(cache)


def padded_mask_sha(path,stride):
    value=hashlib.sha256();blob=path.read_bytes()
    padding=b'\x00\xfc'*((stride-1280)//2)
    for row in range(2048):
        value.update(blob[row*1280:(row+1)*1280]);value.update(padding)
    return value.hexdigest()


def phase(directory,stdout,expected_phase):
    manifest=load(directory/'artifact-manifest.json');rows=records(stdout)
    require(manifest['probe']=='prefill_attention_indirect_real_capture' and manifest['phase']==expected_phase and
        manifest['scope']=='layout_only_cpu_dictionary_replay' and manifest['mode']=='concurrent' and
        manifest['providers']==3 and manifest['total_cases']==3 and manifest['samples']==9 and
        manifest['repetitions_per_cb']==1 and manifest['production_builder_verified'] is False and
        manifest['fullmodel'] is False,'Wrong phase/scope/production contract')
    files,unique=checked_files(directory,manifest)
    require('capture.json' in files and digest(files['capture.json'][0])==manifest['capture_json_sha256'],'Capture binding')
    capture=load(files['capture.json'][0])
    require(capture['capture_kind']=='myds_prefill_attention_multihead' and
            [c['layer'] for c in capture['cases']]==list(LAYERS),'Original real capture layer set')
    require(capture['source']['teacher_forced_tokens_used']==4096 and
            capture['source']['performance_from_capture'] is False and capture['source']['expanded_working_set_required'] is True,
            'Input capture provenance scope')
    expected_inputs={};case_reports=[];external={recorded_path(manifest['capture_json_path']):manifest['capture_json_sha256']}
    kinds={'args200':'args200','q_f32':'q.f32','kv_unique_f16':'kv-unique.f16','kv_row_ids_u32':'kv-rows.u32',
           'mask_f16':'mask-compact.f16','sinks_f32':'sinks.f32'}
    for case_index,case in enumerate(capture['cases']):
        require(all(case[k]==v for k,v in dict(layer=LAYERS[case_index],base_pos=0,c_start=2048,rows=2048,
            keys=640,ratio=2,n_comp=2048,top_k=512).items()),'Original full second-chunk geometry')
        paths={}
        for key,suffix in kinds.items():
            name=f'capture-case-{case_index:03d}-{suffix}';require(name in files,'Missing captured payload')
            path,entry=files[name];desc=case[key]
            require((entry['bytes'],entry['sha256'])==(desc['bytes'],desc['sha256']),'Capture payload descriptor differs')
            paths[key]=path;external[recorded_path(desc['path'])]=desc['sha256']
        stride=common.validate_args(paths['args200'].read_bytes())
        common.finite_written(paths['q_f32'],OUT_BYTES,written=False)
        common.finite_written(paths['sinks_f32'],256,written=False)
        coverage=common.mask_coverage(paths['mask_f16'])
        require(coverage['minimum_valid_picks']==512 and coverage['minimum_valid_window']==128 and
                coverage['full_512_pick_queries']==2048,'Complete 640-active-key real shape')
        expanded=common.expanded_kv_sha(paths['kv_unique_f16'],paths['kv_row_ids_u32'])
        ci=manifest['cases'][case_index]
        require(expanded==case['kv_expanded_sha256']==ci['kv_expanded_sha256'] and
                ci['unique_kv_rows']==paths['kv_unique_f16'].stat().st_size//1024 and ci['mask_coverage']==coverage,
                'Full expanded KV SHA, bounds or shape differs')
        expected_inputs[case_index]={
          'q_f32':(OUT_BYTES,case['q_f32']['sha256']),
          'expanded_kv_f16':(2048*655360,expanded),
          'kv_unique_f16':(case['kv_unique_f16']['bytes'],case['kv_unique_f16']['sha256']),
          'kv_row_ids_u32':(2048*640*4,case['kv_row_ids_u32']['sha256']),
          'mask_original_stride_f16':(2048*stride,padded_mask_sha(paths['mask_f16'],stride)),
          'sinks_f32':(256,case['sinks_f32']['sha256']),
          'unused_pad':(16,hashlib.sha256(b'\xa5'*16).hexdigest())}
        case_reports.append(dict(case=case_index,layer=LAYERS[case_index],rows=2048,keys=640,
            dictionary_rows=ci['unique_kv_rows'],source_bytes=case['kv_unique_f16']['bytes'],
            row_ids_bytes=2048*640*4,expanded_bytes=2048*655360,expanded_sha256=expanded,all_row_ids_in_bounds=True))
    bindings=manifest.get('bindings',[])
    require(len(bindings)==3,'Three selected pipeline bindings required')
    for provider,binding in enumerate(bindings):
        require(binding['provider']==provider and binding['symbol']==SYMBOLS[provider] and
            binding['library']==('production' if provider==0 else 'probe') and binding['threads']==[32,1,1] and
            binding['groups_x']==64 and binding['args_bytes']==200 and binding['dynamic_threadgroup_bytes']==3328 and
            binding['thread_execution_width']==32 and binding['max_threads']>=32 and
            binding['logical_nsg']==binding['logical_nwg']==1 and binding['ordered_constants']==common.FC,
            'Selected symbol/FC/geometry differs')
        slots=['args200','q_f32','kv_unique_f16' if provider==2 else 'expanded_kv_f16']
        slots+=[slots[-1],'mask_original_stride_f16','sinks_f32','unused_pad','full_output_f32']
        if provider==2:slots+=['kv_row_ids_u32']
        require(binding['buffer_slots']==slots,'Resource binding layout differs')
    integrity=manifest['input_integrity']
    require([r['command_ordinal'] for r in integrity]==list(range(1,len(integrity)+1)),
            'Input integrity command ordinals are incomplete or duplicated')
    commands=[r for r in rows if r.get('kind')==('parity' if expected_phase=='correctness' else 'timing')]
    require([(r['case'],r['provider']) for r in integrity]==[(r['case'],r['provider']) for r in commands],
            'Input integrity does not match each actual command provider')
    for record in integrity:
        expected=expected_inputs[record['case']]
        require(record['provider'] in range(3) and record['args200_unchanged'] is True and
            record['all_seven_gpu_inputs_sha_and_guards_unchanged'] is True,'Recorded command input integrity failed')
        buffers=record['buffers'];require(len(buffers)==7 and {b['name'] for b in buffers}==set(expected),'Seven input buffers required')
        for b in buffers:
            require((b['bytes'],b['expected_sha256'])==expected[b['name']] and b['actual_sha256']==b['expected_sha256'] and
                    b['guards_intact'] is True and b['unchanged'] is True,'Per-command input SHA differs from real source')
    for case_index in {r['case'] for r in integrity}:
        for kind in ('kv-unique','kv-row-ids'):
            name=f'input-case-{case_index:03d}-{kind}.guards'
            require(name in files and files[name][0].read_bytes()==b'\xa5'*128,
                    'Archived dictionary/row-ID guard bytes differ')
    return dict(manifest=manifest,rows=rows,files=files,unique=unique,capture=capture,cases=case_reports,external=external)


def correctness_report(data):
    m,rows,files=data['manifest'],data['rows'],data['files']
    require(not any(r.get('kind') in ('timing','summary') for r in rows),'Timing before all-layer correctness completed')
    parity=[r for r in rows if r.get('kind')=='parity']
    require(parity==m['parity'] and 0<len(parity)<=9,'Preserved correctness records required')
    expected=[(c,p) for c in range(3) for p in range(3)]
    require([(r['case'],r['provider']) for r in parity]==expected[:len(parity)],'Sequential correctness case/provider prefix')
    differences=[];finite_cache={}
    for ordinal,r in enumerate(parity,1):
        case,provider=r['case'],r['provider'];name=f'parity-case-{case:03d}-provider-{provider}-output.f32'
        require(name in files and name.replace('.f32','.guards') in files,'Preserved full output/guards missing')
        path,entry=files[name];oracle=files[f'parity-case-{case:03d}-provider-0-output.f32'][0]
        require(path.stat().st_size==OUT_BYTES and r['layer']==LAYERS[case] and r['rows']==2048 and
            r['keys']==640 and r['output_bytes']==OUT_BYTES and r['phase']=='correctness' and
            r['mode']=='concurrent' and r['completed_cb'] is True and r['command_ordinal']==ordinal,'Correctness command contract')
        bad,first=compare_words(oracle,path)
        st=path.stat();key=(st.st_dev,st.st_ino)
        if key not in finite_cache:finite_cache[key]=finite_written(path)
        finite,written=finite_cache[key];guard=files[name.replace('.f32','.guards')][0].read_bytes()==b'\xa5'*128
        require((bad,first)==(r['mismatches'],r['first_mismatch']) and r['output_sha256']==entry['sha256'] and
                (finite,written,guard)==(r['finite'],r['written'],r['guards_intact']), 'Raw output word/finite/guard calculation differs')
        ok=bad==0 and finite and written and guard
        require(r['full_f32_bit_exact']==ok and r['all_gpu_inputs_unchanged'] is True,'Raw correctness flag differs')
        if not ok:differences.append(dict(case=case,layer=LAYERS[case],provider=provider,mismatches=bad,first_mismatch=first,
                                         finite=finite,written=written,guards_intact=guard))
    require(len(m['input_integrity'])==len(parity),'Per-CB input integrity coverage differs')
    passed=len(parity)==9 and not differences
    if passed:
        require(m['status']=='complete' and m['full_f32_bit_exact'] is True and m['attention_output_checked'] is True and
                m['timing_command_buffers']==0,'Completed correctness phase required')
    else:
        require(m['status']=='failed' and differences and not any(r.get('kind')=='complete' for r in rows),
                'Numerical rejection must be preserved without a false completion')
    return dict(correctness_pass=passed,logical_outputs_verified=len(parity),logical_f32_verified=len(parity)*OUT_BYTES//4,
                raw_differences=differences)


def timing_report(rows):
    times=[r for r in rows if r.get('kind')=='timing'];published=[r for r in rows if r.get('kind')=='summary']
    require(len(times)==180 and len(published)==6,'180 timing CBs and six summaries required')
    expected=[(c,k,s,a) for c in range(3) for k in ('production','probe') for s in range(10) for a in ('A1','B','A2')]
    require([(r['case'],r['control_kind'],r['sample'],r['arm']) for r in times]==expected,'Strict interleaved A1/B/A2 sequence')
    groups=defaultdict(list);last_end=0
    for r in times:
        c,k,a=r['case'],r['control_kind'],r['arm'];provider=2 if a=='B' else (0 if k=='production' else 1)
        require(r['provider']==provider and r['symbol']==SYMBOLS[provider] and r['mode']=='concurrent' and
            r['layer']==LAYERS[c] and r['rows']==2048 and r['keys']==640 and r['commits']==r['dispatches']==1 and
            r['completed_cb'] is True and r['full_f32_bit_exact'] is True and r['all_gpu_inputs_unchanged'] is True and
            r['warmup']==(r['sample']==0),'Timing selected provider/shape/completion mismatch')
        start,end=r['gpu_start_s'],r['gpu_end_s'];value=(end-start)*1e6
        require(math.isfinite(start) and math.isfinite(end) and start>=last_end and end>start>0 and
                common.close(value,r['gpu_us_per_chunk']),'Invalid CB clock interval')
        last_end=end
        if r['sample']:groups[c,k,a].append(value)
    reports=[]
    for c in range(3):
        for k in ('production','probe'):
            a1,b,a2=[statistics.median(groups[c,k,a]) for a in ('A1','B','A2')]
            values=dict(a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=(a2/a1-1)*100,
                conservative_time_saved_percent=(1-b/min(a1,a2))*100,time_saved_percent=(1-b/((a1+a2)/2))*100)
            matches=[r for r in published if (r['case'],r['control_kind'])==(c,k)];require(len(matches)==1,'Unique published summary')
            pub=matches[0]
            require(pub['layer']==LAYERS[c] and pub['control_symbol']==NATIVE and pub['candidate_symbol']==INDIRECT and
                all(common.close(pub[key],value) for key,value in values.items()),'Published timing median arithmetic differs')
            reports.append(dict(case=c,layer=LAYERS[c],control_kind=k,**values,
                performance_gate_pass=abs(values['control_drift_percent'])<=2 and values['conservative_time_saved_percent']>2))
    return reports


def archive_bytes(archive,name):
    matches=[m for m in archive.getmembers() if m.name==name or m.name.endswith('/'+name)]
    require(len(matches)==1,'Missing/ambiguous source archive member: '+name)
    return archive.extractfile(matches[0]).read()


def provenance(root,data,source_archive,artifacts_archive):
    m,capture=data['manifest'],data['capture'];frozen=load(root/'stage-inputs.json')
    roots=[Path(p).parent for p in frozen if Path(p).name=='Makefile'];require(len(roots)==1,'Unique frozen stage root')
    source_root=roots[0];bins={};external=data['external'];source_count=0
    require((source_archive is None)==(artifacts_archive is None),'Supply both executed source/artifact archives')
    for path,expected in frozen.items():
        p=Path(path)
        if p.is_relative_to(source_root):
            if p.parent==source_root/'bin':bins[p.name]=expected
        elif p.name=='protocol.json':require(digest(root/'protocol.json')==expected,'Frozen protocol changed')
        else:require(recorded_path(p) in external and external[recorded_path(p)]==expected,'Unbound external frozen payload: '+str(p))
    require(m['binary_sha256']==bins.get('bench_attention_indirect') and m['probe_metallib_sha256']==bins.get('myds.metallib'),
            'Executed replay binary/library differs from frozen stage')
    result=dict(executed_stage_binary_bindings_verified=True,external_capture_payloads_bound=len(external))
    if source_archive:
        with tarfile.open(source_archive,'r:*') as archive:
            for path,expected in frozen.items():
                p=Path(path)
                if not p.is_relative_to(source_root) or p.parent==source_root/'bin':continue
                relative=str(p.relative_to(source_root));body=archive_bytes(archive,relative)
                require(hashlib.sha256(body).hexdigest()==expected,'Frozen source SHA differs: '+relative);source_count+=1
            flash=archive_bytes(archive,'metal/flash_attn.metal')
            probe=archive_bytes(archive,'metal/myds_prefill_attention_indirect.metal')
            require(hashlib.sha256(flash).hexdigest()==builder.FLASH_SHA256,'Native source is not SHA-pinned')
            generated,proof=builder.generate(flash.decode())
            require(probe==generated.encode() and proof['reverse_edits_restore_native_byte_for_byte'],
                    'Probe does not preserve source-template arithmetic')
        with tarfile.open(artifacts_archive,'r:*') as archive:
            for name,expected in bins.items():
                require(hashlib.sha256(archive_bytes(archive,'bin/'+name)).hexdigest()==expected,'Executed artifact SHA differs: '+name)
        result.update(source_members_verified=source_count,executed_artifact_members_verified=len(bins),
            source_archive_sha256=digest(source_archive),artifacts_archive_sha256=digest(artifacts_archive),
            native_template_reverse_edit_proof=True,probe_shader_sha256=hashlib.sha256(probe).hexdigest())
    else:result['executed_archives_supplied']=False
    return result


def review(root,source_archive=None,artifacts_archive=None,model=None):
    root=Path(root).resolve(strict=True);summary=load(root/'summary.json');protocol=load(root/'protocol.json')
    require(protocol['scope']=='single_head_indirect_f16_layout_only' and protocol['production_layout_preparation_tested'] is False and
            protocol['mode']=='concurrent' and protocol['layers']==list(LAYERS) and protocol['samples']==9 and protocol['reps']==1 and
            protocol['candidate_symbol']==INDIRECT and protocol['control_symbol']==NATIVE and protocol['providers']==3 and
            protocol['rows']==2048 and protocol['heads']==64 and protocol['keys']==640 and
            protocol['maximum_absolute_control_drift_percent']==2 and protocol['minimum_conservative_time_saved_percent_strict']==2,
            'Frozen indirect protocol differs')
    before,restored=load(root/'before.json'),load(root/'restored.json')
    keys=('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary','binary_sha256','metallib','metallib_sha256',
          'config','config_sha256','effective_myds_env')
    require(all(before[k]==restored[k] for k in keys),'Recorded restoration identity differs')
    require(digest(root/'original-myds.conf')==before['config_sha256'] and
            load(root/'effective-env.json')==before['effective_myds_env'],'Recorded config/effective environment differs')
    require(before['health']==restored['health']=={'status':'ok'},'Recorded restored health differs')
    events=records(root/'events.jsonl')
    require(events[-1]['kind']=='window_complete' and not any(r['kind'] in ('monitor_failure','restoration_failure','window_failure')
            for r in events),'Recorded window/monitor/restoration failure')
    data=phase(root/'correctness',root/'correctness.stdout','correctness');m=data['manifest'];c=correctness_report(data)
    require(m['production_metallib_sha256']==before['metallib_sha256']==digest(root/'production-artifacts/myds.metallib'),
            'Frozen production library differs from original service')
    capture=data['capture'];require(capture['source']['model_metadata']==summary['model'] and
        capture['source']['effective_environment_sha256']==digest(root/'effective-env.json'),'Real input model/environment source binding differs')
    result=dict(schema_version=1,reviewer='independent_indirect_layout_raw_review',gpu_work_submitted=False,evidence_valid=True,
        **c,cases=data['cases'],correctness_raw_logical_files_verified=len(data['files']),
        correctness_raw_unique_inodes_hashed=data['unique'],correctness_manifest_sha256=digest(root/'correctness/artifact-manifest.json'),
        correctness_stdout_sha256=digest(root/'correctness.stdout'),production_layout_preparation_tested=False,
        fullmodel_run=False,production_adoption=False,recorded_restoration_identity_matches=True,
        original_pid=before['pid'],restored_pid=restored['pid'],reviewer_sha256=digest(Path(__file__)))
    result.update(provenance(root,data,source_archive,artifacts_archive))
    if not c['correctness_pass']:
        require(not (root/'performance.stdout').exists() and summary['correctness_pass'] is False and
            summary['performance_measured'] is False and summary['status']=='component_rejected_correctness',
            'Correctness rejection incorrectly entered performance')
        result.update(decision='reject_component_correctness',performance_measured=False,layout_performance_gate_pass=False,
                      timing_command_buffers_verified=0)
    elif (root/'performance.stdout').exists():
        perf=phase(root/'performance',root/'performance.stdout','performance');pm=perf['manifest']
        require(pm['status']=='complete' and pm['timing_command_buffers']==180 and len(pm['input_integrity'])==180,
                'Performance phase did not complete 180 input-verified commands')
        require(pm['correctness_manifest_sha256']==digest(root/'correctness/artifact-manifest.json') and
            digest(root/'performance/correctness-manifest.json')==pm['correctness_manifest_sha256'],
            'Performance did not use the exact independently checked all-layer correctness gate')
        for key in ('binary_sha256','capture_json_sha256','probe_metallib_sha256','production_metallib_sha256','model','cases','total_cases'):
            require(pm[key]==m[key],'Performance identities changed after correctness')
        performance=timing_report(perf['rows']);passed=all(r['performance_gate_pass'] for r in performance if r['control_kind']=='production')
        require(summary['correctness_pass'] is True and summary['layout_performance_gate_pass']==passed and
            summary['performance_measured'] is True and
            summary['status']==('layout_passed_requires_production_preparation_gate' if passed else 'component_rejected_performance'),
            'Runner decision differs from independently recalculated performance')
        require(summary['timing_command_buffers']==180 and summary['formal_command_buffers']==162 and
            summary['warmup_command_buffers']==18 and summary['logical_outputs']==9 and
            summary['logical_f32']==9*OUT_BYTES//4 and
            summary['correctness_manifest_sha256']==digest(root/'correctness/artifact-manifest.json') and
            summary['performance_manifest_sha256']==digest(root/'performance/artifact-manifest.json'),
            'Runner counts or manifest bindings differ')
        published=summary['performance']
        require(len(published)==6,'Six runner timing summaries required')
        for calculated in performance:
            matches=[r for r in published if (r['case'],r['control_kind'])==
                     (calculated['case'],calculated['control_kind'])]
            require(len(matches)==1,'Unique runner summary per layer/control required')
            row=matches[0]
            require(row['layer']==calculated['layer'] and row['performance_gate_pass']==calculated['performance_gate_pass'] and
                all(common.close(row[key],calculated[key]) for key in
                    ('a1_us','b_us','a2_us','control_drift_percent','conservative_time_saved_percent')),
                'Runner summary differs from independently recomputed CB clocks')
        result.update(performance_measured=True,layout_performance_gate_pass=passed,timing_command_buffers_verified=180,
            formal_command_buffers=162,warmup_command_buffers=18,recomputed_performance=performance,
            performance_raw_logical_files_verified=len(perf['files']),performance_manifest_sha256=digest(root/'performance/artifact-manifest.json'),
            performance_stdout_sha256=digest(root/'performance.stdout'),
            decision='layout_passed_requires_production_preparation_gate' if passed else 'reject_layout_performance')
    else:
        require(summary['correctness_pass'] is True and not summary.get('performance_measured',False),
                'Correctness-only result claimed performance')
        result.update(decision='correctness_passed_performance_not_measured',performance_measured=False,
                      layout_performance_gate_pass=False,timing_command_buffers_verified=0)
    steps=load(root/'steps.json');expected_steps=['host-validation','correctness']
    if result['performance_measured']:expected_steps+=['performance']
    require([r['name'] for r in steps]==expected_steps and steps[0]['returncode']==0,
            'Unexpected/missing execution phase')
    require((steps[1]['returncode']==0)==c['correctness_pass'] and
            all(r['returncode']==0 for r in steps[2:]),'Execution exit codes disagree with raw correctness/performance')
    if model:
        import review_qkv_dual_artifacts as gguf
        names={f'blk.{layer}.attn_sinks.weight' for layer in LAYERS};descriptors,offset=gguf.gguf_descriptors(model,names)
        require(Path(model).stat().st_size==m['model']['bytes'] and offset==m['model']['tensor_data_offset'] and
            gguf.sha(model,length=m['model']['header_bytes'])==m['model']['header_sha256'],'Actual model size/prefix/metadata differ')
        for ci in m['cases']:
            desc=descriptors[ci['sink_tensor']]
            require(desc['type']==0 and desc['dimensions']==[64] and offset+desc['offset']==ci['sink_file_offset'] and
                    gguf.sha(model,ci['sink_file_offset'],256)==ci['sink_sha256'],'Actual model sink interval differs')
        result['live_GGUF_metadata_and_three_sink_intervals_verified']=True
    result['limitations']=['CPU-created dictionary construction cost is excluded; a layout win cannot enter full-model acceptance directly.',
        'Every preserved correctness output is compared in full. Successful performance CB tensors are compared in the frozen harness but not individually archived; raw clock/input-hash records are verified here.',
        'CPU output poison and input hashing are outside GPU timestamps and can affect cache conditions; no cold-DRAM or production graph speed claim.',
        'Offline before/restored records are compared; current live-process inspection is a separate audit.']
    require(summary['fullmodel_run'] is False and summary['production_adoption'] is False,'Unexpected production/fullmodel adoption claim')
    return result


def self_test():
    with tempfile.TemporaryDirectory(prefix='indirect-reader-cpu-') as d:
        a,b=Path(d)/'a',Path(d)/'b';a.write_bytes(struct.pack('<4I',0,0x3f800000,0x80000000,1));b.write_bytes(a.read_bytes())
        require(compare_words(a,b)==(0,None),'Exact raw comparison fixture')
        b.write_bytes(struct.pack('<4I',0,0x3f800001,0x80000000,1));require(compare_words(a,b)==(1,1),'One-ULP negative control')
        b.write_bytes(struct.pack('<2I',0x7fc00000,0xa5a5a5a5));require(finite_written(b)==(False,False),'Finite/poison negative controls')
    rows=[];ordinal=0
    for case in range(3):
        for kind in ('production','probe'):
            values=defaultdict(list)
            for sample in range(10):
                for arm in ('A1','B','A2'):
                    provider=2 if arm=='B' else (0 if kind=='production' else 1)
                    target=(100 if arm=='B' else (200 if arm=='A1' else 202))+sample
                    start=100.0+ordinal*.01;end=start+target*1e-6;ordinal+=1
                    duration=(end-start)*1e6
                    row=dict(kind='timing',case=case,control_kind=kind,sample=sample,arm=arm,
                        provider=provider,symbol=SYMBOLS[provider],mode='concurrent',layer=LAYERS[case],
                        rows=2048,keys=640,commits=1,dispatches=1,completed_cb=True,full_f32_bit_exact=True,
                        all_gpu_inputs_unchanged=True,warmup=sample==0,gpu_start_s=start,gpu_end_s=end,gpu_us_per_chunk=duration)
                    rows.append(row)
                    if sample:values[arm].append(duration)
            a1,b,a2=[statistics.median(values[a]) for a in ('A1','B','A2')]
            rows.append(dict(kind='summary',case=case,control_kind=kind,layer=LAYERS[case],control_symbol=NATIVE,
                candidate_symbol=INDIRECT,a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=(a2/a1-1)*100,
                conservative_time_saved_percent=(1-b/min(a1,a2))*100,time_saved_percent=(1-b/((a1+a2)/2))*100))
    result=timing_report(rows)
    require(len(result)==6 and all(r['performance_gate_pass'] for r in result) and
        all(math.isclose(r['conservative_time_saved_percent'],(1-105/205)*100,abs_tol=1e-5) for r in result),
        'Synthetic 180-CB median/gate positive fixture failed')
    rejected=[]
    def reject(name,fixture):
        try:timing_report(fixture)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Accepted negative timing fixture: '+name)
    bad=copy.deepcopy(rows);bad.append(bad[0]);reject('duplicate_CB',bad)
    bad=copy.deepcopy(rows);bad[0],bad[1]=bad[1],bad[0];reject('wrong_interleaving',bad)
    bad=copy.deepcopy(rows);next(r for r in bad if r['kind']=='summary')['b_us']+=1;reject('published_median_tamper',bad)
    bad=copy.deepcopy(rows);bad[0]['provider']=2;reject('wrong_selected_provider',bad)
    bad=copy.deepcopy(rows);bad[0]['gpu_end_s']=bad[0]['gpu_start_s'];reject('nonpositive_timestamp',bad)
    require(recorded_path('/tmp/a')==recorded_path('/private/tmp/a') and
            recorded_path('/tmp-other/a')=='/tmp-other/a','macOS recorded path alias fixture')
    return dict(self_test_passed=True,gpu_work_submitted=False,
                scope='Small CPU word fixtures and synthetic 180-CB protocol; not full experiment/GPU evidence',
                controls=['exact words','one-ULP mismatch','NaN rejection','poison rejection'],
                synthetic_timing_CBs=180,rejected_timing_negative_controls=rejected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--source-archive',type=Path)
    parser.add_argument('--artifacts-archive',type=Path);parser.add_argument('--model',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();require(not args.out.exists(),'Refuse existing review output overwrite')
    require(args.self_test or args.root is not None,'--root required')
    result=self_test() if args.self_test else review(args.root,args.source_archive,args.artifacts_archive,args.model)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','recomputed_performance','limitations')},indent=2))


if __name__=='__main__':main()
