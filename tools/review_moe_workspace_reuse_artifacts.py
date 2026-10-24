#!/usr/bin/env python3
"""Independent CPU review of successful same-context MoE workspace growth.

No controller analysis, model loader, inference, Metal API or service action is
called. Original tokens, text, logits, live caches and all published snapshots
are compared byte for byte. A missed allocation shadow is never a passing peak.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tarfile
sys.dont_write_bytecode = True
from review_moe_workspace_fullmodel_artifacts import ByteHashes, file_difference

PLAN_SHA = 'f4473d7b634f5b63502e53f3ac38ad8a0bde8edcd5b98e344cce842d889fd40e'
FLAG = 'MYDS_PREFILL_MOE_WORKSPACE'
LAYOUT = ('warm', 'grow', 'retain')
ARMS = ('A1', 'B', 'A2')
SCALARS = ['ctx_pos','rng','pcache_len','anchor_len','next','engram_position',
           'v41_valid','max_seq_len','max_new_tokens','seed','continue_from_cache']
METRICS = ('phys_footprint_bytes','rss_bytes','device_allocated_bytes')
IDENTITY_KEYS = ('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary','metallib',
                 'binary_sha256','metallib_sha256','config','config_sha256','effective_myds_env')


def need(value, message):
    if not value:
        raise ValueError(message)


def parse(text):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'Duplicate JSON key: '+key); result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+v)))


def load(path): return parse(path.read_text())
def rows(path): return [parse(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path, count=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while count is None or count:
            block = stream.read(min(count,1048576) if count is not None else 1048576)
            if not block: break
            digest.update(block)
            if count is not None: count -= len(block)
    need(count in (None,0), 'Short original hashed interval')
    return digest.hexdigest()


def bounded(directory, name):
    need(type(name) is str and name not in ('','.','..') and Path(name).name == name,
         'Flat original raw path required')
    target = directory/name
    need(target.is_file() and not target.is_symlink() and target.resolve().parent == directory.resolve(),
         'Original state payload missing or escaped')
    return target


def workspace(value, setting, B):
    need(type(value) is dict and value['mode'] == setting and value['capacity'] == 2048 and value['moe_batch'] == B,
         'Actual retained/grown workspace capacity differs')
    need(value['failures'] == 0 and value['requests'] == value['mm_encoded']+value['mv_encoded'] and
         value['requests'] > 0 and value['mm_encoded'] > 0, 'Actual complete MoE wrapper counters differ')
    legacy = ('legacy_gate_bytes','legacy_up_bytes','legacy_mid_bytes')
    if setting == 0:
        need(all(value[k] == B*6*2304*4 for k in legacy) and value['scratch_owner_bytes'] == 3*B*6*2304*4,
             'Original F32 owners differ')
    else:
        a = value['arena']; F = a['f32_capacity']
        need(all(value[k] == 0 for k in legacy) and a['mm_capacity'] == B and a['mm_mid_bytes'] == B*6*2304*2 and
             type(F) is int and 0 <= F <= B and all(a[k] == F*6*2304*4 for k in
                 ('f32_gate_bytes','f32_up_bytes','f32_mid_bytes')), 'Compact real owner extents differ')
        need(all(a[k] == value[k] for k in ('requests','mm_encoded','mv_encoded','failures')),
             'Arena and complete wrapper provider counters differ')
        total = sum(a[k] for k in ('mm_mid_bytes','f32_gate_bytes','f32_up_bytes','f32_mid_bytes','retired_bytes'))
        need(a['allocated_bytes'] == value['scratch_owner_bytes'] == total and
             a['peak_allocated_bytes'] >= total and value['workspace_owner_bytes'] >= total,
             'Actual active/retired owner byte accounting differs')


def state(directory, label, sample, hashes):
    manifest = load(directory/'manifest.json')
    need(manifest['schema_version'] == 1 and manifest['phase'] == label and
         manifest['scope'] == 'same_context_moe_workspace_reuse' and manifest['complete'] is True and
         manifest['complete_live_state'] is True and manifest['scalar_order'] == SCALARS,
         'Complete same-context successful state export required')
    scalars = struct.unpack('<11Q', bounded(directory,'context-scalars.u64le').read_bytes())
    positions = list(struct.iter_unpack('<II', bounded(directory,'snapshot-positions.u32le').read_bytes()))
    need(len(positions) == 44 and scalars[7:10] == (327680,8,42) and
         scalars[6] == 1 and scalars[10] in (0,1) and 0 <= scalars[2] <= 327680 and 0 <= scalars[3] <= 1024,
         'Original complete context configuration differs')
    next_pos = scalars[4]
    need(0 < next_pos <= 327680 and scalars[0] == next_pos == scalars[5] and
         sample['prompt_tokens']-1 <= next_pos <= sample['prompt_tokens']+8,
         'Complete live position/Engram extent differs')
    for name, index in (('ctx_pos',0),('pcache_len',2),('anchor_len',3),('next_position',4),('engram_position',5)):
        need(manifest[name] == scalars[index], 'Manifest/raw context metadata differs: '+name)
    need(manifest['rng_state'] == format(scalars[1],'016x') and manifest['anchor_slot'] == 43 and
         manifest['requested_prefill_upper_bound'] == next_pos and len(manifest['limitations']) >= 2,
         'Raw RNG/published-state boundary differs')
    expected = {'context-scalars.u64le':88,'engram-tail.u32le':12,'snapshot-positions.u32le':352,
                'prefix-tokens.i32le':scalars[2]*4,'anchor-tokens.i32le':scalars[3]*4,
                'tokens.i32le':32,'output.bin':sample['output_bytes'],'final-logits.f32':129280*4}
    for layer in range(40): expected[f'layer-{layer:02d}-window-active.f32'] = min(next_pos,128)*2048
    for owner in range(4):
        if owner < 3:
            for kind in ('kv','score'): expected[f'owner-{owner}-previous-{kind}.f32'] = 2048
        count = next_pos//(1 if owner == 3 else 2)
        expected[f'owner-{owner}-compressed-active.bf16'] = count*1024
        expected[f'owner-{owner}-index-active.bf16'] = count*256
    published = []
    for slot, (position, staged) in enumerate(positions):
        need(not(position and staged) and staged <= next_pos and
             ((slot == 43 and (not position or position == scalars[3])) or (slot != 43 and position <= next_pos)),
             'Published/staged snapshot or persistent anchor identity differs')
        if not position: continue
        published.append(dict(slot=slot,position=position))
        name = f'snapshot-{slot:02d}-engram-position-tail.u32le'
        expected[name] = 16
        need(struct.unpack('<4I',bounded(directory,name).read_bytes())[0] == position,
             'Historical Engram position differs from its published snapshot')
        for layer in range(40): expected[f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32'] = min(position,128)*2048
        for owner in range(3):
            for kind in ('kv','score'): expected[f'snapshot-{slot:02d}-owner-{owner}-previous-{kind}.f32'] = 2048
    names = [r['path'] for r in manifest['files']]
    need(len(names) == len(set(names)) == manifest['file_count'] and set(names) == set(expected) and
         {p.name for p in directory.iterdir()} == set(expected)|{'manifest.json'}, 'Exact complete raw state file set differs')
    files = {}
    for item in manifest['files']:
        name = item['path']; value = hashes.file(bounded(directory,name))
        need(value['bytes'] == item['bytes'] == expected[name] and value['fnv1a64_bytes'] == item['fnv1a64_bytes'],
             'Complete original state bytes/FNV differ: '+name)
        files[name] = value
    need(files['tokens.i32le']['fnv1a64_bytes'] == sample['tokens_fnv1a64_bytes'] and
         files['output.bin']['fnv1a64_bytes'] == sample['output_fnv1a64_bytes'], 'Actual callback tokens/text bytes differ')
    logits = array.array('I'); logits.frombytes(bounded(directory,'final-logits.f32').read_bytes())
    if sys.byteorder != 'little': logits.byteswap()
    need(all((v & 0x7f800000) != 0x7f800000 and v != 0xa5a5a5a5 for v in logits), 'Complete vocabulary logits nonfinite/poison')
    return dict(path=str(directory),manifest_sha256=sha(directory/'manifest.json'),files=files,
                published_snapshots=published,rng_state=manifest['rng_state'])


def memory(data, trace):
    samples = rows(trace)
    need(samples and all(r['kind'] == 'memory_sample' and type(r['monotonic_s']) in (int,float) and
         math.isfinite(r['monotonic_s']) and r['run'] in range(3) and r['phase'] in range(4) and
         r['value']['task_valid'] is True and r['value']['device_valid'] is True and
         all(type(r['value'][k]) is int and r['value'][k] >= 0 for k in METRICS) for r in samples),
         'Complete valid original10ms memory trace required')
    need(all(a['monotonic_s'] <= b['monotonic_s'] for a,b in zip(samples,samples[1:])), 'Nonmonotonic raw memory trace')
    reports = [r for r in data if r['kind'] == 'memory_phase']
    need([(r['run'],r['phase'],r['label']) for r in reports] ==
         [(run,phase,label) for run in range(3) for phase,label in ((1,'request_prepare'),(2,'generation'),(3,'raw_export'))],
         'Exact all-stage memory phase order differs')
    out = []
    for report in reports:
        group = [r for r in samples if r['run'] == report['run'] and r['phase'] == report['phase']]
        need(report['interval_ms'] == 10 and report['sampling_ok'] is True and len(group) == report['observed_samples'] > 0,
             'Raw observed phase coverage differs')
        current = report['current']
        need(current['task_valid'] is True and current['device_valid'] is True and
             all(type(current[k]) is int and current[k]>=0 for k in METRICS), 'Invalid actual phase-end memory checkpoint')
        peaks = {key:max(r['value'][key] for r in group) for key in METRICS}
        for suffix,key in (('phys_footprint','phys_footprint_bytes'),('rss','rss_bytes'),('device_allocated','device_allocated_bytes')):
            need(report['observed_peak_'+suffix+'_bytes'] == peaks[key], 'Published peak differs from every original sample')
        gaps = [b['monotonic_s']-a['monotonic_s'] for a,b in zip(group,group[1:])]
        need(report['phase'] != 2 or len(group)>1, 'Generation needs multiple observed points')
        out.append(dict(run=report['run'],step=LAYOUT[report['run']],phase=report['phase'],label=report['label'],
            observed_samples=len(group),maximum_observed_gap_ms=max(gaps)*1000 if gaps else 0,peaks=peaks,
            current=report['current']))
    return dict(trace_sha256=sha(trace),original_samples=len(samples),phases=out,
                continuous_peak_coverage_proven=False,allocation_shadow_observed=False)


def arm(root, name, expected_env, requests, model, hashes):
    directory = root/'arms'/name; data = rows(directory/'summary.jsonl')
    need(all(r['kind'] in ('metadata','sample','memory_phase','complete') for r in data), 'Unexpected native record type')
    meta = [r for r in data if r['kind'] == 'metadata']; complete = [r for r in data if r['kind'] == 'complete']
    samples = [r for r in data if r['kind'] == 'sample']; setting = int(name == 'B')
    need(len(meta) == len(complete) == 1 and data[-1] == complete[0], 'Exactly one complete native sequence required')
    meta, complete = meta[0], complete[0]
    need(meta['schema_version'] == 1 and meta['probe'] == 'same_context_moe_workspace_success_reuse' and
         meta['workspace_mode'] == setting and meta['same_context'] is True and meta['fresh_context_per_sample'] is False and
         meta['manual_cache_or_rng_reset'] is False and meta['extra_gpu_waits_inserted'] == 0 and
         meta['testing_api_used'] is False and meta['factory_lifetime_observer'] is False and
         meta['sampling_coverage_requires_independent_decision'] is True and meta['sampling_interval_ms'] == 10 and
         meta['performance_measured'] is False and meta['wall_time_scope'] == 'diagnostic_only' and
         (meta['max_seq_len'],meta['max_new_tokens'],meta['temperature'],meta['seed']) == (327680,8,0,42),
         'Exact normal-API same-context scope differs')
    need(Path(meta['model']).resolve() == model.resolve() and meta['environment'] == expected_env and
         load(root/'arm-environments'/(name+'.json')) == expected_env, 'Actual model/complete environment differs')
    need([(r['run'],r['step']) for r in samples] == list(enumerate(LAYOUT)), 'Three original successful stages required')
    need([r['step'] for r in meta['requests']] == list(LAYOUT), 'Actual request metadata order differs')
    for record,label,path in zip(meta['requests'],LAYOUT,requests):
        value = hashes.file(path)
        need(Path(record['path']).resolve() == path.resolve() and value['bytes'] == record['bytes'] and
             value['fnv1a64_bytes'] == record['fnv1a64_bytes'], 'Original actual rendered request binding differs')
    reviewed = []; prior = None
    for run, sample in enumerate(samples):
        prompt = (6917,15187,6917)[run]; B = (8192,16384,16384)[run]
        need(sample['same_context'] is True and sample['fresh_context'] == (run == 0) and sample['generation_ok'] is True and
             sample['raw_state_exported'] is True and sample['prompt_tokens'] == prompt and sample['output_tokens'] == 8 and
             sample['stop_reason'] == 'length' and type(sample['cached_tokens']) is int and
             0 <= sample['cached_tokens'] <= prompt-1 and sample['prefill_tokens'] == prompt-1-sample['cached_tokens'],
             'Actual successful same-context output/cache extent differs')
        for key in ('wall_s_diagnostic_only','prefill_s_diagnostic_only'):
            need(type(sample[key]) in (int,float) and math.isfinite(sample[key]) and sample[key]>=0, 'Invalid diagnostic duration')
        if run == 0: need(sample['before_workspace'] is None, 'Warm stage must start without a chunk workspace')
        else: need(sample['before_workspace'] == prior, 'Prior workspace was not retained on the same context')
        workspace(sample['after_workspace'], setting, B); prior = sample['after_workspace']
        reviewed.append(dict(step=LAYOUT[run],run=run,prompt_tokens=prompt,cached_tokens=sample['cached_tokens'],
            prefill_tokens=sample['prefill_tokens'],after_workspace=prior,
            state=state(directory/'state'/LAYOUT[run],LAYOUT[run],sample,hashes)))
    need(complete['three_generations_completed'] is True and complete['actual_C2048_B8192_to16384_grow'] is True and
         complete['actual_B16384_retained_after_short'] is True and complete['performance_measured'] is False and
         complete['sampling_coverage_proven'] is False and complete['adoption_gate_pass'] is False and
         complete['requires_independent_peak_coverage_and_raw_review'] is True, 'Native result must retain unproven-coverage boundary')
    return dict(name=name,setting=setting,metadata=meta,summary_sha256=sha(directory/'summary.jsonl'),samples=reviewed,
                memory=memory(data,directory/'memory/samples.jsonl'))


def peak_decision(arms):
    peaks = {a['name']:next(r['peaks'] for r in a['memory']['phases'] if r['run']==1 and r['phase']==2) for a in arms}
    comparisons = {}
    for key in METRICS:
        baseline = min(peaks['A1'][key],peaks['A2'][key]); candidate = peaks['B'][key]
        comparisons[key] = dict(A1=peaks['A1'][key],B=candidate,A2=peaks['A2'][key],
            conservative_control=baseline,B_excess_bytes=candidate-baseline)
    exceeds = any(comparisons[k]['B_excess_bytes']>134217728 for k in ('phys_footprint_bytes','device_allocated_bytes'))
    # No factory-lifetime observer exists in this frozen native implementation.
    # Seeing a sampled peak can prove a regression; missing a transient cannot
    # establish the upper bound needed for a passing adoption constraint.
    return dict(observed_threshold_bytes=134217728,comparisons=comparisons,observed_excess_blocks_adoption=exceeds,
        growth_shadow_peak_coverage_proven=False,reuse_growth_gate_pass=False,
        decision='reject_observed_growth_peak_regression' if exceeds else 'unknown_missing_shadow_peak_coverage',
        limitation='No simultaneous old+new allocation checkpoint; requested-byte arithmetic and10ms non-observation cannot prove a peak upper bound.')


def archives(root, sources, artifacts, stage):
    bindings = load(root/'stage-inputs.json'); roots = {Path(n).parent for n in bindings if Path(n).name == 'Makefile'}
    need(stage in roots, 'Actual executed reuse stage not frozen')
    index = {}; reports = []
    for kind, paths in (('source',sources),('artifacts',artifacts)):
        for path in paths:
            count = 0
            with tarfile.open(path,'r:*') as archive:
                for member in archive:
                    if member.isdir(): continue
                    need(not member.issym() and not member.isdev() and '..' not in Path(member.name).parts,
                         'Unsafe frozen archive member')
                    stream = archive.extractfile(member); need(stream is not None,'Missing archive bytes')
                    digest = hashlib.sha256(); size = 0
                    for block in iter(lambda:stream.read(1048576),b''): digest.update(block); size += len(block)
                    need(member.islnk() or size == member.size, 'Archived original extent differs')
                    index.setdefault((kind,digest.hexdigest()),[]).append(member.name); count += 1
            reports.append(dict(kind=kind,path=str(path),sha256=sha(path),members_verified=count))
    for name,digest in bindings.items():
        path = Path(name); matches = [r for r in roots if r in path.parents]
        if matches:
            need(len(matches)==1,'Ambiguous stage root'); relative = str(path.relative_to(matches[0]))
            kind = 'artifacts' if relative.startswith('bin/') else 'source'
            need(any(n==relative or n.endswith('/'+relative) for n in index.get((kind,digest),[])),
                 'Frozen source/bin missing from actual archive: '+name)
        else: need(path.is_file() and sha(path)==digest,'Frozen external evidence changed: '+name)
    required = ('tools/bench_moe_workspace_reuse.c','tools/moe_workspace_reuse.mk',
                'tools/run_moe_workspace_reuse_window.py','tools/review_moe_workspace_reuse_artifacts.py',
                'tools/bench_moe_workspace_fullmodel.c','tools/moe_workspace_memory.inc',
                'src/myds_v41.c','src/myds_metal.m','bin/bench_moe_workspace_reuse','bin/myds.metallib')
    need(all(str(stage/n) in bindings for n in required), 'Critical normal-API executed inputs not frozen')
    symbols = load(root/'normal-api-symbol-check.json')
    need(symbols['forbidden_symbols'] == ['myds_metal_moe_workspace_test_control','myds_v41_moe_workspace_probe_create',
                                        'myds_v41_moe_workspace_cancel_control'] and
         symbols['binary_sha256'] == bindings[str(stage/'bin/bench_moe_workspace_reuse')] and
         symbols['forbidden_symbols_present'] is False and not any(n in symbols['nm_stdout'] for n in symbols['forbidden_symbols']),
         'Frozen native binary normal-API symbol check differs')
    return reports


def review(root, model, sources, artifacts):
    plan = load(root/'reuse-protocol.json'); frozen = load(root/'stage-inputs.json')
    plan_paths = [Path(p) for p,h in frozen.items() if h==PLAN_SHA]
    need(len(plan_paths)==1 and sha(plan_paths[0])==PLAN_SHA and load(plan_paths[0])==plan,
         'Frozen pre-data reuse plan binding differs')
    need(plan['order']==list(ARMS) and plan['total_generations']==9 and
         plan['adoption_constraint']['reject_if_B_grow_observed_peak_device_or_phys_exceeds_conservative_A1_A2_by_bytes']==134217728 and
         plan['adoption_constraint']['missing_peak_coverage_is_not_a_pass'] is True,'Frozen peak scope/threshold differs')
    before, restored, summary = [load(root/n) for n in ('before.json','restored.json','summary.json')]
    need(all(before[k]==restored[k] for k in IDENTITY_KEYS) and before['health']==restored['health']=={'status':'ok'} and
         summary['restoration']['restored'] is True and summary['restoration']['pid']==restored['pid'] and
         summary['restoration']['exact_argv_myds_artifacts_config'] is True and summary['production_adoption'] is False and
         summary['fullmodel_run'] is True and summary['performance_measured'] is False and
         summary['status']=='reuse_complete_independent_review_required','Exact restoration/diagnostic-only boundary differs')
    need(before['binary_sha256']=='29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b' and
         before['metallib_sha256']=='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81' and
         sha(root/'production-artifacts/myds')==before['binary_sha256'] and
         sha(root/'production-artifacts/myds.metallib')==before['metallib_sha256'] and
         sha(root/'original-myds.conf')==before['config_sha256'],'Original production recovery artifacts differ')
    environment = before['effective_myds_env']
    need(len(environment)==51 and environment['MYDS_METAL_CONCURRENT']=='1' and environment['MYDS_PREFIX_CACHE']=='1' and
         environment['MYDS_PLD']=='1' and load(root/'effective-env.json')==environment,'Actual production51 cache settings differ')
    need(model.stat().st_size==summary['model']['bytes'] and sha(model,1048576)==summary['model_header_sha256'],
         'Retained actual model prefix/extent differs')
    steps = load(root/'steps.json')
    need([r['name'] for r in steps]==list(ARMS) and all(r['returncode']==0 for r in steps), 'Exact single A1/B/A2 execution required')
    events = rows(root/'events.jsonl')
    need(events[-1]['kind']=='reuse_decision_complete' and events[-1]['status']==summary['status'] and
         not any(r['kind'] in ('monitor_failure','restoration_failure') for r in events),'Competition/restoration event failure')
    stages = {Path(s['argv'][0]).parent.parent for s in steps}; need(len(stages)==1,'One same-build native reuse binary required')
    stage = next(iter(stages)); need(frozen[str(stage/'bin/myds.metallib')]==before['metallib_sha256'],'Arithmetic library changed')
    for step in steps:
        argv = step['argv']; need(len(argv)==12 and Path(argv[0]).name=='bench_moe_workspace_reuse' and
             argv[1:4]==['--run','--model',str(model)] and argv[4]=='--warm-request' and argv[6]=='--grow-request' and
             argv[8]=='--retain-request' and argv[10]=='--out-dir' and
             Path(argv[11]).resolve()==(root/'arms'/step['name']).resolve(),'Actual native invocation differs')
    requests = [Path(steps[0]['argv'][i]) for i in (5,7,9)]
    for path, field in zip(requests,('pf7k_request_sha256','pf16k_request_sha256','pf7k_request_sha256')):
        need(sha(path)==plan[field],'Original frozen request SHA differs')
    need(all([s['argv'][i] for i in (5,7,9)]==list(map(str,requests)) for s in steps),'Arms did not use identical raw requests')
    prerequisites = []
    for path,digest in frozen.items():
        p = Path(path)
        if not p.is_file() or p.suffix != '.json': continue
        value = load(p)
        if type(value) is not dict: continue
        for key in ('component_gate_pass','recovery_gate_pass'):
            if key in value and value.get('evidence_valid') is True:
                need(value[key] is True and sha(p)==digest, 'Frozen independent prerequisite changed or failed')
                prerequisites.append(dict(key=key,path=path,sha256=digest))
    need({p['key'] for p in prerequisites}=={'component_gate_pass','recovery_gate_pass'},
         'Both independent component and recovery gates must be frozen')
    hashes = ByteHashes()
    try: arms = [arm(root,name,dict(environment,**{FLAG:str(int(name=='B'))}),requests,model,hashes) for name in ARMS]
    finally: hashes.close()
    comparisons = []
    for run,label in enumerate(LAYOUT):
        gold = arms[0]['samples'][run]['state']
        for other in arms[1:]:
            candidate = other['samples'][run]['state']; need(set(gold['files'])==set(candidate['files']),'State coverage differs between arms')
            diffs = {name:file_difference(Path(gold['path'])/name,Path(candidate['path'])/name,name.endswith('.f32')) for name in gold['files']}
            need(all(v['equal'] for v in diffs.values()),'Complete original state/tokens/text/logits differ: '+label+'/'+other['name'])
            comparisons.append(dict(step=label,left='A1',right=other['name'],files=len(diffs),
                compared_bytes=sum(v['bytes'] for v in gold['files'].values()),full_original_bytes_equal=True))
    recorded_arms = load(root/'reuse-arms.json')
    need([a['name'] for a in recorded_arms]==list(ARMS) and all(
         record['setting']==value['setting'] and record['native_summary_sha256']==value['summary_sha256'] and
         Path(record['raw_directory']).resolve()==(root/'arms'/value['name']).resolve() and record['sampling_coverage_proven'] is False
         for record,value in zip(recorded_arms,arms)), 'Complete actual arm records/SHA differ')
    for value in arms:
        need(Path(value['metadata']['actual_metallib_path']).resolve()==(stage/'bin/myds.metallib').resolve(),
             'Actual native library binding differs')
    archive_report = archives(root,sources,artifacts,stage)
    decision = peak_decision(arms)
    return dict(schema_version=1,scope='independent_same_context_successful_MoE_growth_review',evidence_valid=True,
        correctness_pass=True,reuse_growth_gate_pass=decision['reuse_growth_gate_pass'],performance_measured=False,
        production_adoption=False,decision=decision['decision'],peak_review=decision,arms=arms,
        complete_original_byte_comparisons=comparisons,reader_sha256=sha(Path(__file__)),
        reuse_plan_sha256=PLAN_SHA,source_artifact_archives=archive_report,frozen_prerequisites=prerequisites,original_pid=before['pid'],restored_pid=restored['pid'],
        limits=['Exactly one same-context sequence per arm; this is a memory/correctness diagnostic, not new performance samples.',
                '10ms maxima are sampled observations; no upper peak bound is inferred from an unobserved allocation shadow.',
                'Requested owner bytes are checked as logical identities, not used as physical or device memory measurements.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','model','out'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-archive',type=Path,action='append',required=True)
    parser.add_argument('--artifacts-archive',type=Path,action='append',required=True)
    args = parser.parse_args()
    try: result = review(args.root.resolve(),args.model.resolve(),args.source_archive,args.artifacts_archive)
    except Exception as exc:
        result = dict(schema_version=1,evidence_valid=False,correctness_pass=False,reuse_growth_gate_pass=False,
                      performance_measured=False,production_adoption=False,error=repr(exc),reader_sha256=sha(Path(__file__)))
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,sort_keys=True); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('evidence_valid','correctness_pass','reuse_growth_gate_pass','decision','error') if k in result}))
    return 0 if result['evidence_valid'] else 1


if __name__=='__main__': raise SystemExit(main())
