#!/usr/bin/env python3
"""Separate fullmodel gate after approved real-component and recovery evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import sys
sys.dont_write_bytecode = True
import run_moe_workspace_window as workspace
need,sha,write_json=workspace.need,workspace.sha,workspace.write_json
FLAG=workspace.FLAG
SCALAR_ORDER=['ctx_pos','rng_state','max_seq_len','max_new_tokens','seed_u32','kv_capacity',
              'pcache_len','anchor_len','continue_from_cache','v41_valid','next_position',
              'engram_position','snapshot_allocated_bytes','prefill_capacity','low_kv_enabled']
_FNV_CACHE={}
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fnv(data):
    key=(len(data),hashlib.sha256(data).hexdigest())
    if key in _FNV_CACHE:return _FNV_CACHE[key]
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    result=format(value, '016x');_FNV_CACHE[key]=result;return result


def file_fnv(path,digest):
    # Actual SHA is already computed for every payload. Cache by content so
    # deterministic repeated samples do not repeat Python FNV over large state.
    key=(path.stat().st_size,digest)
    if key not in _FNV_CACHE:
        value=14695981039346656037
        with path.open('rb') as stream:
            while True:
                block=stream.read(1048576)
                if not block:break
                for byte in block:value=((value^byte)*1099511628211)&((1<<64)-1)
        _FNV_CACHE[key]=format(value,'016x')
    return _FNV_CACHE[key]


def workspace_contract(ws,setting,prompt_tokens,gold):
    B=2048*min(8,(prompt_tokens-1+2047)//2048)
    need(type(ws) is dict and ws['mode']==setting and ws['capacity']==2048 and ws['moe_batch']==B,
         'Actual macroblock-sized workspace/provider differs')
    legacy=('legacy_gate_bytes','legacy_up_bytes','legacy_mid_bytes')
    if not setting:
        need(all(ws[k]==B*6*2304*4 for k in legacy) and ws['scratch_owner_bytes']==3*B*6*2304*4,
             'Original complete three B-sized F32 scratch owners required')
    if gold:
        need(not setting and ws.get('provider_counters_available') is False,'Independent original gold diagnostic API differs')
        return
    need(ws['failures']==0 and ws['requests']==ws['mm_encoded']+ws['mv_encoded'] and ws['mm_encoded']>0,
         'Actual complete MoE host provider counters required')
    if setting:
        a=ws['arena'];F=a['f32_capacity']
        need(all(ws[k]==0 for k in legacy) and a['mm_capacity']==B and a['mm_mid_bytes']==B*6*2304*2 and
             type(F) is int and 0<=F<=B and all(a[k]==F*6*2304*4 for k in ('f32_gate_bytes','f32_up_bytes','f32_mid_bytes')),
             'Actual compact F16/F32 arena capacities differ')
        need(a['failures']==0 and a['requests']==a['mm_encoded']+a['mv_encoded'] and
             all(a[k]==ws[k] for k in ('requests','mm_encoded','mv_encoded')),
             'Arena and workspace actual provider counters differ')
        total=sum(a[k] for k in ('mm_mid_bytes','f32_gate_bytes','f32_up_bytes','f32_mid_bytes','retired_bytes'))
        need(a['allocated_bytes']==ws['scratch_owner_bytes']==total and a['peak_allocated_bytes']>=total and
             ws['workspace_owner_bytes']>=total,'Actual logical arena owner byte accounting differs')


def same_file(left, right):
    need(left.stat().st_size == right.stat().st_size, 'Original payload length differs')
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(1048576), b.read(1048576)
            need(x == y, 'Original raw payload bytes differ: ' + str(right))
            if not x: return


def compare_arms(gold, other):
    need(gold['witness'] == other['witness'], 'Numerical/state witness differs from gold')
    g = Path(gold['payload_roots'][0]); logits = Path(gold['logit_paths'][0])
    for root, lp in zip(other['payload_roots'], other['logit_paths']):
        same_file(logits, Path(lp))
        for name in gold['witness']['state_files']: same_file(g / name, Path(root) / name)


def memory_review(records,trace_path,repeats):
    raw=rows(trace_path);need(raw,'Actual10ms memory trace required')
    need(all(r['kind']=='memory_sample' and r['value']['task_valid'] is True and r['value']['device_valid'] is True and
             type(r['monotonic_s']) in (int,float) and math.isfinite(r['monotonic_s']) and
             r['run'] in range(repeats+1) and r['phase'] in (0,1,2,3) and
             all(type(r['value'][k]) is int and r['value'][k]>=0 for k in ('phys_footprint_bytes','rss_bytes','device_allocated_bytes'))
             for r in raw),
         'Every raw memory observation must be valid')
    need(all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(raw,raw[1:])),'Raw memory time must be monotonic')
    phases=[r for r in records if r.get('kind')=='memory_phase']
    result=[]
    for run in range(repeats+1):
        selected=[r for r in phases if r['run']==run and r['phase']==2]
        need(len(selected)==1 and selected[0]['label']=='generation','Exactly one generation memory phase per sample')
        rec=selected[0];observed=[r['value'] for r in raw if r['run']==run and r['phase']==2]
        need(rec['sampling_ok'] and rec['interval_ms']==10 and len(observed)==rec['observed_samples']>1,'Complete raw phase observations required')
        for key,source in [('phys_footprint','phys_footprint_bytes'),('rss','rss_bytes'),('device_allocated','device_allocated_bytes')]:
            need(rec['observed_peak_'+key+'_bytes']==max(v[source] for v in observed),'Raw memory peak arithmetic differs')
        result.append(rec)
    return result


class FullmodelWorkspaceWindow(workspace.WorkspaceWindow):
    def preflight(self):
        super().preflight()
        for path,key in [(self.a.component_review,'component_gate_pass'),(self.a.recovery_review,'recovery_gate_pass')]:
            record=json.loads(path.read_text());need(record.get(key) is True,'Independent prerequisite failed: '+key)
            need(record.get('evidence_valid') is True,'Independent evidence is incomplete')
            self.input_hashes[str(path)]=sha(path)
        write_json(self.a.out/'stage-inputs.json',self.input_hashes)
        self.oracles={}
    def arm(self, phase, workload, name, setting, repeats):
        stage = self.a.gold_stage if name == 'gold' else self.a.stage
        label = '-'.join((phase, workload, name))
        directory = self.a.out / label
        directory.mkdir()
        logits, state = directory / 'logits', directory / 'state'
        logits.mkdir(); state.mkdir(); memory = directory / 'memory'; memory.mkdir()
        env = dict(self.environment)
        env.update({FLAG: str(setting), 'MYDS_BENCH_LOGITS_DIR': str(logits), 'MYDS_BENCH_STATE_DIR': str(state), 'MYDS_BENCH_MEMORY_DIR': str(memory)})
        write_json(directory / 'environment.json', {k: v for k, v in env.items() if k.startswith('MYDS_')})
        argv = [stage / 'bin/bench_moe_workspace_fullmodel', self.a.model,
                self.a.inputs / (workload + '.json'), str(repeats), '32', '327680']
        # The working directory controls the first metallib search path.
        saved = self.a.stage
        self.a.stage = stage
        try:
            self.run_command(label, argv, env=env, timeout=1800)
        finally:
            self.a.stage = saved
        data = rows(self.a.out / (label + '.stdout'))
        samples = [r for r in data if r.get('kind') == 'sample']
        need(len(samples) == repeats + 1 and [r['run'] for r in samples] == list(range(repeats + 1)),
             'Exact warmup/formal sample sequence required')
        meta = [r for r in data if r.get('kind') == 'metadata']
        load = [r for r in data if r.get('kind') == 'load']
        need(len(meta) == len(load) == 1, 'One actual metadata and load record required')
        need(meta[0]['protocol_version'] == 4 and meta[0]['repeats'] == repeats and
             meta[0]['warmup_runs'] == 1 and meta[0]['max_new_tokens'] == 32 and
             meta[0]['max_seq_len'] == 327680 and meta[0]['temperature'] == 0 and
             meta[0]['top_p'] == 1 and meta[0]['top_k'] == meta[0]['min_p'] == 0 and
             meta[0]['repetition_penalty'] == 1 and meta[0]['seed'] == 42 and
             load[0]['fresh_context_per_sample'] is True and load[0]['cache_configuration_preserved'] is True,
             'Actual fullmodel sample configuration differs')
        expected_env={k:v for k,v in env.items() if k.startswith('MYDS_')}
        actual_env={k:v for k,v in meta[0]['environment'].items() if v is not None}
        need(actual_env==expected_env,'Actual complete MYDS environment differs, including an extra selector')
        need(Path(load[0]['actual_metallib_path']).resolve() == (stage / 'bin/myds.metallib').resolve(),
             'Actual metallib loader differs')
        need(load[0]['moe_workspace_mode'] == setting, 'Actual workspace mode differs')
        witnesses, payload_roots, logit_paths = [], [], []
        for r in samples:
            need(r['sample_valid'] is True and r['generation_ok'] is True and r['output_tokens'] == 32 and
                 r['warmup'] == (r['run'] == 0) and r['stop_reason'] == 'length' and
                 r['prompt_tokens'] == self.plan['fullmodel']['requests'][workload]['expected_prompt_tokens'] and
                 r['cached_tokens'] == 0 and r['final_logits']['finite'] is True,
                 'Complete fresh-context real-workload sample required')
            for key in ('decode_s', 'prefill_s', 'wall_s', 'first_callback_s', 'last_callback_s', 'decode_ms_per_token'):
                need(type(r[key]) in (int, float) and math.isfinite(r[key]) and r[key] > 0,
                     'Positive finite timing required: ' + key)
            need(r['decode_intervals'] == 31 and r['prefill_tokens'] == r['prompt_tokens'] - 1 and
                 math.isclose(r['decode_s'], r['last_callback_s'] - r['first_callback_s'], abs_tol=3e-9) and
                 math.isclose(r['decode_ms_per_token'], r['decode_s'] * 1000 / 31, abs_tol=1e-8),
                 'Actual raw wall/callback/prefill arithmetic differs')
            intervals=r['callback_intervals_ms']
            need(len(intervals)==31 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in intervals) and
                 math.isclose(sum(intervals),r['decode_s']*1000,abs_tol=1e-5) and r['last_callback_s']<=r['wall_s']+1e-8,
                 'Complete callback interval array differs from the recorded decode span')
            raw = (logits / ('run-' + str(r['run']) + '.f32')).read_bytes()
            need(len(raw) == 129280 * 4 and r['final_logits']['elements']==129280 and
                 all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)),
                 'Full finite raw vocabulary logits required')
            need(fnv(raw) == r['final_logits']['fnv1a64_bytes'], 'Raw logits differ from native record')
            ws = r['moe_workspace']
            workspace_contract(ws,setting,r['prompt_tokens'],name=='gold')
            for key in ('gpu_attention_packed_register', 'gpu_attention_rope_fuse'):
                provider = r[key]
                need(provider['requests'] == provider['encoded'] > 0 and provider['failures'] == 0 and
                     provider['decode_requests'] == provider['decode_encoded'] > 0 and
                     provider['decode_failures'] == 0, 'Actual production provider differs: ' + key)
            payload = state / ('run-' + str(r['run']))
            payload_roots.append(str(payload)); logit_paths.append(str(logits / ('run-' + str(r['run']) + '.f32')))
            manifest = json.loads((payload / 'manifest.json').read_text())
            files = {}
            for item in manifest['files']:
                p = (payload / item['path']).resolve(strict=True)
                need(payload.resolve() in p.parents and p.is_file() and p.stat().st_size == item['bytes'],
                     'Complete bounded raw state payload required')
                need(item['path'] not in files, 'Duplicate raw state payload')
                digest=sha(p)
                need(file_fnv(p,digest)==item['fnv1a64_bytes'],'Raw state item FNV differs: '+item['path'])
                files[item['path']] = dict(bytes=item['bytes'], sha256=digest)
            need(files and manifest.get('complete') is True, 'Complete state manifest required')
            need(manifest['schema_version'] == 1 and manifest['run'] == r['run'] and
                 manifest['scope'] == 'fresh_context_full_prefill_final_state' and
                 manifest['file_count'] == len(files), 'State manifest protocol differs')
            sm = manifest['metadata']
            scalars = struct.unpack('<15Q', (payload / 'context-scalars.u64le').read_bytes())
            need(sm['fresh_context'] is True and r['fresh_context'] is True and
                 sm['layers'] == 40 and sm['window_width'] == 512 and
                 sm['byte_order']=='little' and sm['window_order']=='chronological_active_rows' and
                 sm['compressed_row_bytes']==1024 and sm['index_row_bytes']==256 and sm['scalar_order']==SCALAR_ORDER and
                 sm['ctx_pos'] == scalars[0] and sm['next_position'] == scalars[10] and
                 scalars[2:5] == (327680, 32, 42) and scalars[9]==1 and
                 all(scalars[i] in (0,1) for i in (8,14)), 'Raw context/config metadata differs')
            tail = list(struct.unpack('<3I', (payload / 'engram-tail.u32le').read_bytes()))
            need(sm['rng_state'] == format(scalars[1], '016x') and sm['engram_position'] == scalars[11] and
                 sm['engram_tail'] == tail, 'Raw Engram/RNG metadata differs')
            expected = {'context-scalars.u64le': 120, 'engram-tail.u32le': 12,
                        'prefix-tokens.i32le': scalars[6] * 4, 'anchor-tokens.i32le': scalars[7] * 4,
                        'tokens.i32le': 32 * 4, 'output.bin': r['output_bytes']}
            next_pos = sm['next_position']
            need(0 < next_pos <= 327680 and sm['window_rows'] == min(next_pos, 128), 'Window state extent differs')
            for layer in range(40): expected[f'layer-{layer:02d}-window-active.f32'] = min(next_pos, 128) * 512 * 4
            for owner in range(4):
                if owner < 3:
                    for kind in ('kv', 'score'): expected[f'owner-{owner}-previous-{kind}.f32'] = 2048
                count = next_pos // (1 if owner == 3 else 2)
                expected[f'owner-{owner}-compressed-active.bf16'] = count * 1024
                expected[f'owner-{owner}-index-active.bf16'] = count * 256
            positions = (payload / 'snapshot-positions.u32le').read_bytes()
            need(len(positions) == 44 * 8, 'Exact 44 snapshot positions required')
            expected['snapshot-positions.u32le'] = len(positions)
            for slot, (position, staged) in enumerate(struct.iter_unpack('<II', positions)):
                need(staged <= next_pos and position <= next_pos and not (staged and position),
                     'Invalid or simultaneously published/staged snapshot metadata')
                if position:
                    expected[f'snapshot-{slot:02d}-engram-tail.u32le'] = 12
                    for layer in range(40): expected[f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32'] = min(position, 128) * 2048
                    for owner in range(3):
                        for kind in ('kv', 'score'): expected[f'snapshot-{slot:02d}-owner-{owner}-previous-{kind}.f32'] = 2048
            need(set(files) == set(expected) and all(files[k]['bytes'] == size for k, size in expected.items()),
                 'Complete active raw state file coverage/extent differs')
            need((payload / 'tokens.i32le').read_bytes() == struct.pack('<32i', *r['token_ids']) and
                 fnv((payload / 'output.bin').read_bytes()) == r['output_bytes_fnv1a64'],
                 'Complete original token/text bytes differ from native record')
            witnesses.append(dict(token_ids=r['token_ids'], output_bytes=r['output_bytes'],
                                  output_hash=r['output_bytes_fnv1a64'], logits_sha256=sha(logits / ('run-' + str(r['run']) + '.f32')),
                                  state_metadata=manifest['metadata'], state_files=files))
        need(all(w == witnesses[0] for w in witnesses), 'Same arm did not reproduce exact tokens/logits/state')
        measured = memory_review(data, memory / 'samples.jsonl', repeats)
        result = dict(name=label, stage=str(stage), samples=samples, memory=measured, witness=witnesses[0],
                      payload_roots=payload_roots, logit_paths=logit_paths,
                      binary_sha256=sha(stage / 'bin/bench_moe_workspace_fullmodel'),
                      metallib_sha256=sha(stage / 'bin/myds.metallib'))
        compare_arms(result, result)
        write_json(directory / 'review.json', result)
        return result

    def component(self):
        self.summary["fullmodel_run"]=True
        correctness=[]
        for workload in ('pf7k','pf16k'):
            group=[self.arm('correctness',workload,name,setting,1) for name,setting in [('gold',0),('off',0),('on',1)]]
            for arm in group:compare_arms(group[0],arm)
            self.oracles[workload]=group[0]
            correctness.append(dict(workload=workload,passed=True,arms=[x['name'] for x in group]))
        write_json(self.a.out/'correctness-gate.json',correctness)
        self.summary.update(fullmodel_run=True,correctness_pass=True,correctness=correctness)
        reports=[]
        for workload in ('pf7k','pf16k'):
            group=[self.arm('performance',workload,name,setting,3) for name,setting in [('A1',0),('B',1),('A2',0)]]
            for arm in group:compare_arms(self.oracles[workload],arm)
            pf=[statistics.median(r['prefill_s'] for r in a['samples'][1:]) for a in group]
            dec=[statistics.median(r['decode_s'] for r in a['samples'][1:]) for a in group]
            phys=[statistics.median(r['observed_peak_phys_footprint_bytes'] for r in a['memory'][1:]) for a in group]
            device=[statistics.median(r['observed_peak_device_allocated_bytes'] for r in a['memory'][1:]) for a in group]
            rss=[statistics.median(r['observed_peak_rss_bytes'] for r in a['memory'][1:]) for a in group]
            pfdrift=(pf[2]/pf[0]-1)*100;ddrift=(dec[2]/dec[0]-1)*100
            pfsave=(1-pf[1]/min(pf[0],pf[2]))*100;dsave=(1-dec[1]/min(dec[0],dec[2]))*100
            physsave=min(phys[0],phys[2])-phys[1];devsave=min(device[0],device[2])-device[1]
            timepass=abs(pfdrift)<=2 and abs(ddrift)<=2 and pfsave>=-1 and dsave>=-1
            mempass=abs(phys[2]-phys[0])<=268435456 and physsave>=-134217728 and devsave>0
            gain=physsave>=268435456 or (pfsave>2 and physsave>=0)
            reports.append(dict(workload=workload,prefill_seconds=dict(zip(('A1','B','A2'),pf)),decode_seconds=dict(zip(('A1','B','A2'),dec)),
                observed_peak_phys_footprint_bytes=dict(zip(('A1','B','A2'),phys)),observed_peak_device_allocated_bytes=dict(zip(('A1','B','A2'),device)),
                observed_peak_rss_bytes=dict(zip(('A1','B','A2'),rss)),prefill_control_drift_percent=pfdrift,decode_control_drift_percent=ddrift,
                conservative_prefill_time_saved_percent=pfsave,conservative_decode_time_saved_percent=dsave,
                conservative_peak_phys_reduction_bytes=physsave,conservative_peak_device_reduction_bytes=devsave,
                latency_gate_pass=timepass,memory_gate_pass=mempass,measured_gain_pass=gain,gate_pass=timepass and mempass and gain))
        self.summary.update(performance=reports,performance_gate_pass=all(r['gate_pass'] for r in reports))


def finalize_window(worker,code):
    if worker.summary['status']=='component_complete_review_required':
        worker.summary['status']='fullmodel_passed_review_required' if worker.summary['performance_gate_pass'] else 'fullmodel_rejected_gain_or_controls'
        write_json(worker.a.out/'summary.json',worker.summary)
        worker.event('fullmodel_decision_complete',status=worker.summary['status'])
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','gold-stage','model','out','plan','inputs','component-review','recovery-review'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=8187);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');a=p.parse_args()
    need(sys.platform=='darwin','macOS required')
    for k in ('stage','gold_stage','model','out','plan','inputs','component_review','recovery_review'):
        v=getattr(a,k);need(v.is_absolute(),'Absolute paths required');setattr(a,k,v.resolve())
    for stage in (a.stage,a.gold_stage):need(Path('/tmp').resolve() in stage.parents,'Separate tmp stages required')
    a.out.mkdir(parents=True,exist_ok=False);os.chmod(a.out,0o700);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with (a.out/'runner.log').open('ab',buffering=0) as log,open(os.devnull,'rb') as null:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    worker=FullmodelWorkspaceWindow(a);code=worker.run()
    return finalize_window(worker,code)

if __name__=='__main__':sys.exit(main())
