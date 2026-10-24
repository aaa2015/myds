#!/usr/bin/env python3
"""CPU-only independent same-context MoE cancellation/recovery evidence gate.

Deterministic cancellation compares defined state at the same observed checkpoint.
Asynchronous cancellation validates each partial state separately and compares the
complete successful recovery. Neither mode compares uninitialized or partial live
capacity, inserts GPU waits, measures performance, or supplies a fresh-context oracle.
"""
import argparse
import array
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import importlib.util
spec=importlib.util.spec_from_file_location("localgrow_sequence_reader",Path(__file__).with_name("review_moe_workspace_fullmodel_promptclear.py"))
layout_reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout_reader)

sys.dont_write_bytecode = True
OFFSET = 14695981039346656037
HASH_C = r'''
#include <stdint.h>
#include <stddef.h>
uint64_t moe_cancel_fnv(const unsigned char *p,size_t n,uint64_t h) {
    for(size_t i=0;i<n;i++) { h^=p[i]; h*=UINT64_C(1099511628211); }
    return h;
}
'''
SCALARS = ['ctx_pos', 'rng', 'pcache_len', 'anchor_len', 'next', 'engram_position',
           'v41_valid', 'max_seq_len', 'max_new_tokens', 'seed', 'continue_from_cache']


class Rejected(ValueError):
    pass


def need(value, message):
    if not value:
        raise Rejected(message)


class Hashes:
    """A separately compiled CPU hash loop avoids hours of Python FNV iteration."""
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='moe-cancel-reader-')
        path = Path(self.temp.name) / ('hash.dylib' if sys.platform == 'darwin' else 'hash.so')
        command = [os.environ.get('CC', 'cc'), '-O2', '-shared', '-fPIC', '-x', 'c', '-', '-o', str(path)]
        subprocess.run(command, input=HASH_C.encode(), check=True, capture_output=True)
        self.lib = ctypes.CDLL(str(path)); self.fn = self.lib.moe_cancel_fnv
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
        self.fn.restype = ctypes.c_uint64
        need(self.block(b'hello', OFFSET) == 0xa430d84680aabd0b, 'CPU FNV helper known answer')
        self.compiler = command[0]
        return self

    def __exit__(self, *unused):
        self.temp.cleanup()

    def block(self, data, h):
        return self.fn(ctypes.c_char_p(data), len(data), h)

    def file(self, path):
        sha = hashlib.sha256(); fnv = OFFSET; size = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                sha.update(block); fnv = self.block(block, fnv); size += len(block)
        return {'bytes': size, 'sha256': sha.hexdigest(), 'fnv1a64_bytes': f'{fnv:016x}'}


def summary_gate(value):
    need(value.get('schema_version') == 1 and value.get('probe') == 'same_context_moe_workspace_cancel_promptclear', 'Native cancellation schema')
    need(value.get('status') == 'complete' and value.get('fullmodel_recovery_within_arm_completed') is True, 'Native arm completed')
    need(value.get('implementation_version')==4 and value.get('prompt_view_clear_mode')==value.get('workspace_mode'),'Actual v4 two-flag selection')
    need(value.get('scenario') in ('deterministic', 'async') and value.get('workspace_mode') in (0, 1), 'Scenario/mode')
    need(value.get('seed') == 42 and value.get('recovery_tokens') in (8, 32) and
         (value.get('temperature') == 0 or abs(value.get('temperature', -10) - .7) < 1e-6), 'Frozen RNG/sampling/count')
    need(value.get('same_context') is True and value.get('rng_manually_reset') is False and
         value.get('extra_gpu_waits_inserted') == 0 and value.get('performance_measured') is False, 'Same-context no-wait scope')
    need(value.get('cancel_api_ok') is True and value.get('cancel_gate_pass') is True and
         value.get('cancel_state_exported') is True and value.get('cancel_stop_reason') == 'cancelled' and
         value.get('cancel_output_tokens') == 0 and value.get('cancel_prompt_tokens', 0) > 1, 'Actual prefill cancellation before first callback')
    need(value.get('target_checkpoint', 0) > 0 and value.get('observed_progress', -1) >= value['target_checkpoint'], 'Actual checkpoint observed')
    need(value.get('thread_timed_out') is False, 'Timeout must not create a false cancellation pass')
    if value['scenario'] == 'async':
        need(value.get('async_thread_set_flag') is True and
             value['target_checkpoint'] <= value.get('thread_flag_progress', -1) <= value['observed_progress'], 'Async thread set the real flag at measured progress')
    else:
        need(value.get('async_thread_set_flag') is False and value.get('thread_flag_progress') == 0, 'Deterministic hook scope')
    need(value.get('recovery_api_ok') is True and value.get('recovery_generation_ok') is True and
         value.get('recovery_state_exported') is True and value.get('recovery_stop_reason') == 'length' and
         value.get('recovery_output_tokens') == value['recovery_tokens'] and value.get('recovery_prompt_tokens', 0) > 1, 'Same-context successful fixed-length recovery')
    need(value.get('snapshot_slots') == 44 and isinstance(value.get('vocab_size'), int) and value['vocab_size'] > 0, 'Real state shape metadata')
    need(value.get('same_context_warm_before_cancel') is True and value.get('actual_B8192_to16384_growth_before_cancel') is True and value.get('warm_generation_ok') is True and value.get('warm_state_exported') is True and value.get('warm_prompt_tokens')==6917 and value.get('warm_output_tokens')==value['recovery_tokens'] and value.get('cancel_prompt_tokens')==15187 and value.get('recovery_prompt_tokens')==6917,'Actual 7k warm then 16k grow cancellation')
    for key in ('warm_workspace','cancel_workspace', 'recovery_workspace'):
        layout_reader.workspace(value[key],value['workspace_mode'],8192 if key=='warm_workspace' else 16384)
        w = value.get(key)
        need(isinstance(w, dict) and w.get('mode') == value['workspace_mode'] and
             w.get('requests', 0) > 0 and w.get('mm_encoded', 0) > 0 and w.get('failures') == 0, 'Actual workspace path and MM adoption: ' + key)
        if value['workspace_mode']:
            need(w.get('legacy_gate_bytes') == w.get('legacy_up_bytes') == w.get('legacy_mid_bytes') == 0 and
                 w.get('arena', {}).get('mm_encoded', 0) > 0 and w['arena'].get('failures') == 0, 'Compact actual arena adoption')


def state_metadata(manifest, scalars, positions, phase, summary):
    need(manifest.get('schema_version') == 1 and manifest.get('phase') == phase and
         manifest.get('scope') == 'same_context_prefill_recovery' and manifest.get('complete') is True, 'Complete state export: ' + phase)
    complete = phase in ('before-cancel', 'before-failure', 'recovery')
    need(manifest.get('complete_live_state') is complete and manifest.get('scalar_order') == SCALARS and
         len(manifest.get('limitations', [])) >= 2, 'Explicit partial/complete state boundary')
    need(len(scalars) == 11 and len(positions) == 88, 'All scalar/snapshot metadata')
    for key, index in (('ctx_pos', 0), ('pcache_len', 2), ('anchor_len', 3), ('next_position', 4), ('engram_position', 5)):
        need(manifest.get(key) == scalars[index], 'Manifest/raw scalar disagreement: ' + key)
    need(manifest.get('rng_state') == f'{scalars[1]:016x}' and scalars[6] in (0, 1) and
         scalars[7] == 327680 and scalars[8] == summary['recovery_tokens'] and scalars[9] == 42 and scalars[10] in (0, 1), 'Raw context/RNG/config witness')
    need(0 <= scalars[2] <= scalars[7] and 0 <= scalars[3] <= 1024 and 0 <= scalars[4] <= scalars[7], 'Published live metadata bounds')
    upper = manifest.get('requested_prefill_upper_bound'); anchor = manifest.get('anchor_slot')
    need(isinstance(upper, int) and 0 <= upper <= scalars[7] and anchor == 43, 'Actual request/anchor metadata bounds')
    if phase in ('after-cancel', 'after-failure'):
        key = 'cancel_prompt_tokens' if phase == 'after-cancel' else 'failure_prompt_tokens'
        need(scalars[2] == 0 and upper == summary[key] - 1, 'Interrupted prefix invalidation and actual request bound')
        # next/ctx/Engram and per-layer data need not be equal in interrupted prefill.
    else:
        need(scalars[0] == scalars[4] == scalars[5] and upper == scalars[4] and
             summary.get('warm_prompt_tokens',summary.get('recovery_prompt_tokens'))-1 <= scalars[4] <= summary.get('warm_prompt_tokens',summary.get('recovery_prompt_tokens'))+summary['recovery_tokens'] and scalars[2] <= scalars[4], 'Complete successful live state position')
    published = {}
    for slot, (p, staged) in enumerate(zip(positions[::2], positions[1::2])):
        need(not (p and staged) and staged <= upper, 'Publication/staging exclusivity or staged bound')
        need((slot == anchor and (not p or p == scalars[3])) or (slot != anchor and p <= upper), 'Published snapshot bound/retained anchor identity')
        if p:
            published[slot] = p
    return published


def expected_files(phase, scalars, published, summary, output_bytes):
    e = {'context-scalars.u64le': 88, 'engram-tail.u32le': 12, 'snapshot-positions.u32le': 352,
         'prefix-tokens.i32le': scalars[2]*4, 'anchor-tokens.i32le': scalars[3]*4}
    e['tokens.i32le'] = (summary['recovery_tokens'] if phase in ('before-cancel', 'before-failure', 'recovery') else 0)*4
    e['output.bin'] = output_bytes
    if phase in ('before-cancel', 'before-failure', 'recovery'):
        for layer in range(40):
            e[f'layer-{layer:02}-window-active.f32'] = min(scalars[4], 128)*2048
        for owner in range(4):
            if owner < 3:
                e[f'owner-{owner}-previous-kv.f32'] = e[f'owner-{owner}-previous-score.f32'] = 2048
            rows = scalars[4] // (1 if owner == 3 else 2)
            e[f'owner-{owner}-compressed-active.bf16'] = rows*1024
            e[f'owner-{owner}-index-active.bf16'] = rows*256
        e['final-logits.f32'] = summary['vocab_size']*4
    for slot, position in published.items():
        e[f'snapshot-{slot:02}-engram-position-tail.u32le'] = 16
        for layer in range(40):
            e[f'snapshot-{slot:02}-layer-{layer:02}-window-active.f32'] = min(position, 128)*2048
        for owner in range(3):
            e[f'snapshot-{slot:02}-owner-{owner}-previous-kv.f32'] = 2048
            e[f'snapshot-{slot:02}-owner-{owner}-previous-score.f32'] = 2048
    return e


def read_state(root, phase, summary, hashes):
    directory = root / phase; manifest_path = directory / 'manifest.json'
    need(directory.is_dir() and not directory.is_symlink() and not manifest_path.is_symlink(), 'Original state directory/manifest')
    manifest = layout_reader.load(manifest_path); descriptors = manifest.get('files', [])
    files = {}; witnessed = {}
    for descriptor in descriptors:
        name = descriptor.get('path')
        need(isinstance(name, str) and Path(name).name == name and name not in files, 'Unique flat safe state path')
        path = directory / name
        need(path.is_file() and not path.is_symlink(), 'Original regular state artifact')
        actual = hashes.file(path)
        need(actual['bytes'] == descriptor.get('bytes') and actual['fnv1a64_bytes'] == descriptor.get('fnv1a64_bytes'), 'Original raw extent/FNV: ' + name)
        files[name] = path; witnessed[name] = actual
    need(manifest.get('file_count') == len(files) == len(descriptors), 'Exact state file count')
    need('context-scalars.u64le' in files and 'snapshot-positions.u32le' in files, 'State metadata raw present')
    raw = files['context-scalars.u64le'].read_bytes(); positions_raw = files['snapshot-positions.u32le'].read_bytes()
    need(len(raw) == 88 and len(positions_raw) == 352, 'Metadata raw extents')
    scalars = struct.unpack('<11Q', raw); positions = struct.unpack('<88I', positions_raw)
    published = state_metadata(manifest, scalars, positions, phase, summary)
    expected = expected_files(phase, scalars, published, summary, witnessed.get('output.bin', {}).get('bytes', 0))
    need(set(files) == set(expected) and {p.name for p in directory.iterdir()}==set(expected)|{'manifest.json'}, 'Exact complete defined state file set: ' + phase)
    need(all(witnessed[name]['bytes'] == size for name, size in expected.items()), 'Exact active state extents')
    if phase in ('after-cancel', 'after-failure'):
        need(witnessed['output.bin']['bytes'] == 0, 'No callback bytes before cancellation')
    for slot, p in published.items():
        tail = files[f'snapshot-{slot:02}-engram-position-tail.u32le'].read_bytes()
        need(struct.unpack('<4I', tail)[0] == p, 'Historical snapshot Engram position is published position')
    if phase in ('before-cancel', 'before-failure', 'recovery'):
        logits = array.array('I'); logits.frombytes(files['final-logits.f32'].read_bytes())
        if sys.byteorder != 'little':
            logits.byteswap()
        need(all((word & 0x7f800000) != 0x7f800000 for word in logits), 'All original recovery logits finite')
    return {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'manifest': manifest, 'files': files, 'witnessed': witnessed, 'published': published}


def compare_states(left, right, phase):
    need(set(left['files']) == set(right['files']), 'Off/on defined state file set: ' + phase)
    compared = 0
    for name, a in left['files'].items():
        b = right['files'][name]
        with a.open('rb') as x, b.open('rb') as y:
            while True:
                p, q = x.read(1 << 20), y.read(1 << 20)
                need(p == q, f'Off/on full original raw bytes differ: {phase}/{name}')
                if not p:
                    break
                compared += len(p)
    return compared



def read_prompt_snapshot(path,setting):
    v=layout_reader.load(path)
    need(v.get('schema_version')==1 and v.get('scope')=='process_local_prompt_group_readonly_identity' and v.get('complete')is True and v.get('implementation_version')==4 and v.get('compact')==setting and v.get('clear_mode')==setting and v.get('quiescent')==1 and v.get('owner_order')==['RES','PRE','SELECTED','MASK','ENGRAM0','ENGRAM1'],'Complete v4 current prompt identity')
    for key in ('owners','views'):
        need(len(v[key])==6,'Six actual prompt identities')
        for x in v[key]:need(set(x)=={'wrapper_id','resource_id','offset','bytes','is_view'} and all(type(x[k])is int and x[k]>=0 for k in x) and x['is_view']in (0,1),'Actual bounded prompt identity')
    if v['cap']:
        need(all(x['wrapper_id']>0 and x['resource_id']>0 and x['bytes']>0 and not x['is_view'] for x in v['owners']) and len({x['resource_id']for x in v['owners']})==6 and sum(x['bytes']for x in v['owners'])==v['total_bytes'],'Six complete distinct prompt owners')
        need(v['cap']in (7168,15360) and v['mask_bytes']in (112,240),'Frozen actual prompt extent')
    else:need(not v['mask_bytes']and not v['total_bytes']and all(not x['wrapper_id']and not x['resource_id']and not x['bytes']for x in v['owners']),'Released prompt group must be completely absent')
    return dict(path=str(path),sha256=layout_reader.sha(path),value=v)

def prompt_snapshot_evidence(root,phases,setting):return [read_prompt_snapshot(root/('prompt-'+p+'.json'),setting)for p in phases]

def review(off, on, warm_request, cancel_request, recover_request):
    with Hashes() as hashes:
        summaries = [layout_reader.load(r/'summary.json') for r in (off,on)]
        for s in summaries:
            summary_gate(s)
        a, b = summaries
        need(a['workspace_mode'] == 0 and b['workspace_mode'] == 1, 'Independent off/on arms required')
        for key in ('scenario', 'seed', 'temperature', 'recovery_tokens', 'target_checkpoint','warm_prompt_tokens', 'cancel_prompt_tokens',
                    'recovery_prompt_tokens', 'vocab_size', 'snapshot_slots', 'request_inputs'):
            need(a.get(key) == b.get(key), 'Frozen off/on metadata: ' + key)
        requests = {}
        for name, path in (('warm',warm_request),('cancel', cancel_request), ('recover', recover_request)):
            actual = hashes.file(path); requests[name] = actual
            for s in summaries:
                need(all(s['request_inputs'][name].get(k) == actual[k] for k in ('bytes', 'fnv1a64_bytes')), 'Frozen original request identity: ' + name)
        arms = [{phase: read_state(root, phase, summary, hashes) for phase in ('before-cancel', 'after-cancel', 'recovery')}
                for root, summary in zip((off, on), summaries)]
        compared = {'before-cancel': compare_states(arms[0]['before-cancel'], arms[1]['before-cancel'], 'before-cancel')}
        layout_evidence=review_cancel_layouts(off,on,summaries)
        prompt_evidence=[]
        for root,sm in zip((off,on),summaries):
            ps=prompt_snapshot_evidence(root,('before-cancel','after-cancel','recovery'),sm['workspace_mode']);w,c,r=[x['value']for x in ps]
            need(w['cap']==7168 and c['cap']==r['cap']==15360 and c['owners']==r['owners'],'Cancelled grown prompt owners retained into short recovery')
            prompt_evidence.append(ps)
        deterministic = a['scenario'] == 'deterministic'
        if deterministic:
            need(a['observed_progress'] == b['observed_progress'], 'Conditional checkpoint oracle requires identical actual cancellation progress')
            compared['after-cancel'] = compare_states(arms[0]['after-cancel'], arms[1]['after-cancel'], 'after-cancel')
        compared['recovery'] = compare_states(arms[0]['recovery'], arms[1]['recovery'], 'recovery')
        return {'schema_version': 1, 'evidence_valid': True, 'correctness_pass': True,
                'same_context_recovery_raw_bit_exact': True,'prompt_identity_evidence':prompt_evidence,'implementation_version':4,'localgrow_layout_evidence':layout_evidence,'same_context_warm_before_cancel':True, 'scenario': a['scenario'],
                'temperature': a['temperature'], 'seed': 42, 'recovery_tokens': a['recovery_tokens'],
                'observed_progress': [s['observed_progress'] for s in summaries],
                'thread_flag_progress': [s['thread_flag_progress'] for s in summaries],
                'cancellation_progress_skew': b['observed_progress']-a['observed_progress'],
                'partial_states_compared': deterministic, 'raw_bytes_compared': compared,
                'request_inputs': requests, 'performance_measured': False,
                'decision': 'pass_same_context_recovery_within_exported_state_scope',
                'arms': [{phase: {k: value[k] for k in ('manifest_sha256', 'witnessed', 'published')}
                          for phase, value in arm.items()} for arm in arms],
                'cpu_hash_helper': {'source_sha256': hashlib.sha256(HASH_C.encode()).hexdigest(), 'compiler': hashes.compiler},
                'limitations': ['Cancelled partially updated live rings/cache/carry and unpublished capacity are omitted; published snapshots are checked at their own positions.',
                                'Async cancellation validates individual partial state legality; unequal transient progress is not forced to have equal raw data.',
                                'No persistent disk-cache payload, inactive capacity, allocator padding, performance, or production memory-pressure OOM is proven.',
                                'Checkpoint hooks and observed progress are correctness diagnostics; no extra GPU wait is inserted.']}


def review_cancel_layouts(off,on,summaries):
    checked=[];hashes=layout_reader.ByteHashes()
    try:
        for root,s in zip((off,on),summaries):
            v=[layout_reader.read_layout(root/('layout-'+phase+'.json'),hashes) for phase in ('before-cancel','after-cancel','recovery')]
            need(all(q['value']['present'] and q['value']['compact']==s['workspace_mode']for q in v),'Actual compact selector throughout same-context cancel')
            layout_reader.layout_transition(v[0],v[1],1,s['workspace_mode']);layout_reader.layout_transition(v[1],v[2],2,s['workspace_mode'])
            if s['workspace_mode']:
                layout_reader.counter_transition(s['warm_workspace'],s['cancel_workspace'],v[0],v[1],1)
                layout_reader.counter_transition(s['cancel_workspace'],s['recovery_workspace'],v[1],v[2],1)
            checked.append([dict(path=q['path'],sha256=q['sha256'],value={k:x for k,x in q['value'].items()if k!='bindings'},actual_bindings_fully_compared=True)for q in v])
    finally:hashes.close()
    return checked


def test_workspace(mode,B):
    w=dict(mode=mode,capacity=2048,moe_batch=B,requests=40,mm_encoded=39,mv_encoded=1,failures=0,workspace_owner_bytes=10*1024**3)
    legacy=B*6*2304*4 if not mode else 0
    w.update(legacy_gate_bytes=legacy,legacy_up_bytes=legacy,legacy_mid_bytes=legacy,scratch_owner_bytes=3*legacy)
    if mode:
        n=B*6*2304*2;w['scratch_owner_bytes']=n;w['arena']=dict(f32_capacity=0,mm_capacity=B,mm_mid_bytes=n,f32_gate_bytes=0,f32_up_bytes=0,f32_mid_bytes=0,requests=40,mm_encoded=39,mv_encoded=1,failures=0,retired_bytes=0,allocated_bytes=n,peak_allocated_bytes=n)
    return w


def cancel_fixture_summary():
    return dict(schema_version=1,implementation_version=4,prompt_view_clear_mode=0,probe='same_context_moe_workspace_cancel_promptclear',status='complete',fullmodel_recovery_within_arm_completed=True,scenario='deterministic',workspace_mode=0,seed=42,recovery_tokens=8,temperature=.7,same_context=True,rng_manually_reset=False,extra_gpu_waits_inserted=0,performance_measured=False,cancel_api_ok=True,cancel_gate_pass=True,cancel_state_exported=True,cancel_stop_reason='cancelled',cancel_output_tokens=0,cancel_prompt_tokens=15187,target_checkpoint=1,observed_progress=1,async_thread_set_flag=False,thread_timed_out=False,thread_flag_progress=0,recovery_api_ok=True,recovery_generation_ok=True,recovery_state_exported=True,recovery_stop_reason='length',recovery_output_tokens=8,recovery_prompt_tokens=6917,snapshot_slots=44,vocab_size=129280,same_context_warm_before_cancel=True,actual_B8192_to16384_growth_before_cancel=True,warm_generation_ok=True,warm_state_exported=True,warm_prompt_tokens=6917,warm_output_tokens=8,warm_workspace=test_workspace(0,8192),cancel_workspace=test_workspace(0,16384),recovery_workspace=test_workspace(0,16384))


def self_test():
    controls=[];s=cancel_fixture_summary()
    def test(name,fn,reject=False):
        try:fn();passed=not reject
        except (Rejected,ValueError,KeyError,TypeError):passed=reject
        controls.append(dict(name=name,passed=passed));need(passed,'CPU control '+name)
    test('warmed_deterministic_metadata',lambda:summary_gate(s))
    x=copy.deepcopy(s);x.update(scenario='async',async_thread_set_flag=True,thread_flag_progress=1,observed_progress=2);test('warmed_async_observed_skew',lambda:summary_gate(x))
    for key,bad in [('same_context_warm_before_cancel',False),('actual_B8192_to16384_growth_before_cancel',False),('warm_prompt_tokens',15187),('warm_generation_ok',False),('rng_manually_reset',True),('extra_gpu_waits_inserted',1),('cancel_output_tokens',1),('thread_timed_out',True),('recovery_prompt_tokens',15187),('recovery_output_tokens',7)]:
        x=copy.deepcopy(s);x[key]=bad;test('reject_'+key,lambda x=x:summary_gate(x),True)
    x=copy.deepcopy(s);x['warm_workspace']['moe_batch']=16384;test('reject_already_large_initial_pool',lambda:summary_gate(x),True)
    with tempfile.TemporaryDirectory(prefix='moe-localgrow-cancel-cpu-')as directory,Hashes()as hashes:
        p=Path(directory);(p/'a').write_bytes(b'raw\x00state');(p/'b').write_bytes(b'raw\x00state');states=[dict(files={'x':p/name})for name in ('a','b')]
        test('original_raw_bytes',lambda:compare_states(*states,'fixture'));(p/'b').write_bytes(b'raw\x01state');test('reject_raw_changed',lambda:compare_states(*states,'fixture'),True)
    return dict(schema_version=1,scope='cpu_promptclear_warm_cancel_reader_controls',gpu_run=False,controls=controls,passed=all(v['passed']for v in controls))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--off', type=Path); p.add_argument('--on', type=Path)
    p.add_argument('--warm-request', type=Path);p.add_argument('--cancel-request', type=Path); p.add_argument('--recover-request', type=Path)
    p.add_argument('--out', type=Path); p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    try:
        if args.self_test:
            report = self_test()
        else:
            need(all((args.off, args.on, args.warm_request, args.cancel_request, args.recover_request)), 'Off/on roots and original requests required')
            report = review(args.off.resolve(), args.on.resolve(), args.warm_request.resolve(), args.cancel_request.resolve(), args.recover_request.resolve())
    except (Rejected, OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        report = {'evidence_valid': False, 'correctness_pass': False, 'performance_measured': False,
                  'decision': 'reject_incomplete_or_inconsistent_recovery_evidence', 'error': str(error)}
    if args.out:
        with args.out.open('x') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report, indent=2)); return 0 if report.get('passed', report.get('correctness_pass', False)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
