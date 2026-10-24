#!/usr/bin/env python3
"""Isolated, default-off Output_A full-model gate with exact production restore.

Builds are deliberately external. --prepare only freezes inputs and reads the
service identity. --run requires independently approved steady components,
executes gold/off/on correctness before same-build A1/B/A2 performance, then
restores the original service in finally. No deployment is performed.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import sys
import threading
import shutil
sys.dont_write_bytecode = True
import run_iq2_signed4_window as safety

need, sha, write_json = safety.need, safety.sha, safety.write_json
FLAG = 'MYDS_ATTN_OUT_A_FIXED_WRAPPER'
NATIVE = 'kernel_dsv4_attn_out_low_q4_K_bf16'
FIXED = 'kernel_myds_outa_q4_v41_fixed_bf16'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(value, '016x')


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


def source_inputs(stage):
    result = [stage / 'Makefile']
    for directory in ('src', 'include', 'metal', 'tools', 'third_party'):
        result += [p for p in (stage / directory).rglob('*') if p.is_file() and
                   (p.suffix in ('.c', '.h', '.m', '.metal', '.py', '.inc', '.mk') or p.name == 'LICENSE')]
    result += [stage / 'bin/bench_output_a_fullmodel', stage / 'bin/myds.metallib', stage / 'bin/myds']
    return result


class FullmodelWindow(safety.Window):
    def preflight(self):
        self.before, self.config_raw = safety.capture(self.a.port)
        self.allowed.add(self.before['pid'])
        need(os.getpriority(os.PRIO_PROCESS, os.getpid()) <= self.before['nice'], 'Nice restoration unavailable')
        need(Path(safety.option(self.before['argv'], '-m')).resolve(strict=True) == self.a.model,
             'Model differs from actual production')
        need(self.before['binary_sha256'] == '29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b' and
             self.before['metallib_sha256'] == 'c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81',
             'Frozen production identity differs')
        for stage in (self.a.stage, self.a.gold_stage):
            need(str(stage) not in self.before['binary']['path'], 'Experimental stage overlaps production')
        need(sha(self.a.gold_stage / 'bin/myds') == self.before['binary_sha256'] and
             sha(self.a.gold_stage / 'bin/myds.metallib') == self.before['metallib_sha256'],
             'Gold stage must reproduce exact production artifacts')
        self.environment = {k: v for k, v in os.environ.items() if not k.startswith('MYDS_')}
        self.environment.update(self.before['effective_myds_env'])
        self.environment.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',
                                PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')
        need(self.environment.get('MYDS_PREFIX_CACHE') == '1' and self.environment.get('MYDS_PLD') == '1',
             'Actual production cache/PLD selectors required')
        need(self.environment.get('MYDS_METAL_CONCURRENT') == '1', 'Production concurrent mode required')
        for key in self.before['effective_myds_env']:
            need(not key.startswith(('MYDS_PROFILE_', 'MYDS_TRACE_', 'MYDS_SKIP_')), 'Diagnostic production environment')
        self.summary.update(model=safety.metadata(self.a.model), model_header_sha256=sha(self.a.model, 1048576),
                            component='ordinary_decode_output_a_fixed_wrapper_fullmodel',
                            runner_nice=os.getpriority(os.PRIO_PROCESS, os.getpid()))
        write_json(self.a.out / 'before.json', self.before)
        (self.a.out / 'original-myds.conf').write_bytes(self.config_raw)
        backup = self.a.out / 'production-artifacts'; backup.mkdir()
        for key in ('binary', 'metallib'):
            target = backup / ('myds' if key == 'binary' else 'myds.metallib')
            shutil.copy2(self.before[key]['path'], target)
            need(sha(target) == self.before[key + '_sha256'], 'Production backup hash differs')
        write_json(self.a.out / 'effective-env.json', self.before['effective_myds_env'])
        self.plan = json.loads(self.a.plan.read_text())
        need(self.plan['performance']['formal_samples'] == 3 and self.plan['max_seq_len'] == 327680 and
             self.plan['max_new_tokens'] == 256 and self.plan['performance']['repeat_limit'] == 1,
             'Frozen fullmodel plan differs')
        approval = json.loads(self.a.component_review.read_text())
        need(approval.get('component_and_frozen_two_window_replication_gate_pass') is True and
             approval.get('every_window_every_layer_every_control_pass') is True and
             approval.get('decision') == 'eligible_for_separate_fullmodel_gate',
             'Independent two-window component approval required')
        inputs = [*source_inputs(self.a.stage), *source_inputs(self.a.gold_stage),
                  self.a.gold_stage / 'bin/myds', self.a.plan, self.a.component_review]
        groups = approval['all_12_groups']
        need(len(groups) == 12 and {(g['window'], g['layer'], g['control_kind']) for g in groups} ==
             {(w, layer, kind) for w in (1, 2) for layer in (0, 19, 39) for kind in ('production', 'probe')},
             'Exact two-window six-group component coverage required')
        need(all(g['performance_gate_pass'] is True and abs(g['control_drift_percent']) <= 2 and
                 g['conservative_time_saved_percent'] > 2 for g in groups), 'Component group gate differs')
        for item in [*approval['reports'], dict(path=approval['plan_path'], sha256=approval['plan_sha256'])]:
            p = (self.a.component_review.parent / item['path']).resolve(strict=True)
            need(self.a.component_review.parent.resolve() in p.parents and sha(p) == item['sha256'],
                 'Component approval leaf report/plan hash differs')
            inputs.append(p)
        for workload in ('short', 'business'):
            p = self.a.stage / ('request-' + workload + '.json')
            need(sha(p) == self.plan['workloads'][workload]['sha256'], 'Frozen request bytes differ')
            inputs.append(p)
        self.input_hashes = {str(p): sha(p) for p in sorted(set(inputs))}
        need(str(Path(__file__).resolve()) in self.input_hashes and
             str(Path(safety.__file__).resolve()) in self.input_hashes, 'Execute frozen runner and safety helper')
        write_json(self.a.out / 'stage-inputs.json', self.input_hashes)
        write_json(self.a.out / 'protocol.json', self.plan)
        self.event('preflight_processes', relevant=self.scan())

    def arm(self, phase, workload, name, setting, repeats):
        stage = self.a.gold_stage if name == 'gold' else self.a.stage
        label = '-'.join((phase, workload, name))
        directory = self.a.out / label
        directory.mkdir()
        logits, state = directory / 'logits', directory / 'state'
        logits.mkdir(); state.mkdir()
        env = dict(self.environment)
        env.update({FLAG: str(setting), 'MYDS_BENCH_LOGITS_DIR': str(logits), 'MYDS_BENCH_STATE_DIR': str(state)})
        write_json(directory / 'environment.json', {k: v for k, v in env.items() if k.startswith('MYDS_')})
        argv = [stage / 'bin/bench_output_a_fullmodel', self.a.model,
                self.a.stage / ('request-' + workload + '.json'), str(repeats), '256', '327680']
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
        need(meta[0]['protocol_version'] == 3 and meta[0]['repeats'] == repeats and
             meta[0]['warmup_runs'] == 1 and meta[0]['max_new_tokens'] == 256 and
             meta[0]['max_seq_len'] == 327680 and meta[0]['temperature'] == 0 and
             meta[0]['top_p'] == 1 and meta[0]['seed'] == 42 and
             load[0]['fresh_context_per_sample'] is True and load[0]['cache_configuration_preserved'] is True,
             'Actual fullmodel sample configuration differs')
        for k, v in env.items():
            if k.startswith('MYDS_'):
                need(meta[0]['environment'].get(k) == v, 'Actual harness environment differs: ' + k)
        need(Path(load[0]['actual_metallib_path']).resolve() == (stage / 'bin/myds.metallib').resolve(),
             'Actual metallib loader differs')
        need(load[0]['output_a_fixed_wrapper_mode'] == setting and
             load[0]['output_a_fixed_wrapper_selected_symbol'] == (FIXED if setting else None),
             'Actual candidate mode/selected symbol differs')
        need(load[0]['output_a_fixed_wrapper_api_available'] == (name != 'gold'), 'Actual candidate API build differs')
        witnesses, payload_roots, logit_paths = [], [], []
        for r in samples:
            need(r['sample_valid'] is True and r['generation_ok'] is True and r['output_tokens'] == 256 and
                 r['warmup'] == (r['run'] == 0) and r['stop_reason'] == 'length' and
                 r['prompt_tokens'] == self.plan['workloads'][workload]['expected_prompt_tokens'] and
                 r['cached_tokens'] == 0 and r['final_logits']['finite'] is True,
                 'Complete fresh-context real-workload sample required')
            for key in ('decode_s', 'prefill_s', 'wall_s', 'first_callback_s', 'last_callback_s', 'decode_ms_per_token'):
                need(type(r[key]) in (int, float) and math.isfinite(r[key]) and r[key] > 0,
                     'Positive finite timing required: ' + key)
            need(r['decode_intervals'] == 255 and r['prefill_tokens'] == r['prompt_tokens'] - 1 and
                 math.isclose(r['decode_s'], r['last_callback_s'] - r['first_callback_s'], abs_tol=3e-9) and
                 math.isclose(r['decode_ms_per_token'], r['decode_s'] * 1000 / 255, abs_tol=1e-8),
                 'Actual raw wall/callback/prefill arithmetic differs')
            raw = (logits / ('run-' + str(r['run']) + '.f32')).read_bytes()
            need(len(raw) == 129280 * 4 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)),
                 'Full finite raw vocabulary logits required')
            need(fnv(raw) == r['final_logits']['fnv1a64_bytes'], 'Raw logits differ from native record')
            c = r['gpu_output_a_fixed_wrapper']
            need(c['encoded'] == c['requests'] and c['failures'] == c['fallbacks'] == c['ineligible'] == 0,
                 'Candidate requests/encodes/failure contract differs')
            need((c['encoded'] > 0) == bool(setting), 'Candidate selection was not actually exercised')
            need((c['nr0'], c['nsg'], c['fc600']) == ((2, 4, 4) if setting else (0, 0, 0)),
                 'Actual candidate geometry differs')
            need(c['decode_requests'] == c['decode_encoded'] and
                 c['decode_fallbacks'] == c['decode_ineligible'] == c['decode_failures'] == 0 and
                 ((c['decode_encoded'] > 0) == bool(setting)), 'Actual ordinary decode path differs')
            for key in ('gpu_attention_packed_register', 'gpu_attention_rope_fuse'):
                provider = r[key]
                need(provider['requests'] == provider['encoded'] > 0 and provider['failures'] == 0 and
                     provider['decode_requests'] == provider['decode_encoded'] > 0 and
                     provider['decode_failures'] == 0, 'Actual production provider differs: ' + key)
            if setting:
                rf = r['gpu_attention_rope_fuse']
                need(c['encoded'] == rf['encoded'] and c['decode_encoded'] == rf['decode_encoded'],
                     'Candidate did not cover every eligible ordinary RF layer')
            payload = state / ('run-' + str(r['run']))
            payload_roots.append(str(payload)); logit_paths.append(str(logits / ('run-' + str(r['run']) + '.f32')))
            manifest = json.loads((payload / 'manifest.json').read_text())
            files = {}
            for item in manifest['files']:
                p = (payload / item['path']).resolve(strict=True)
                need(payload.resolve() in p.parents and p.is_file() and p.stat().st_size == item['bytes'],
                     'Complete bounded raw state payload required')
                need(item['path'] not in files, 'Duplicate raw state payload')
                files[item['path']] = dict(bytes=item['bytes'], sha256=sha(p))
            need(files and manifest.get('complete') is True, 'Complete state manifest required')
            need(manifest['schema_version'] == 1 and manifest['run'] == r['run'] and
                 manifest['scope'] == 'fresh_context_full_prefill_final_state' and
                 manifest['file_count'] == len(files), 'State manifest protocol differs')
            sm = manifest['metadata']
            scalars = struct.unpack('<15Q', (payload / 'context-scalars.u64le').read_bytes())
            need(sm['fresh_context'] is True and r['fresh_context'] is True and
                 sm['layers'] == 40 and sm['window_width'] == 512 and
                 sm['ctx_pos'] == scalars[0] and sm['next_position'] == scalars[10] and
                 scalars[2:5] == (327680, 256, 42), 'Raw context/config metadata differs')
            tail = list(struct.unpack('<3I', (payload / 'engram-tail.u32le').read_bytes()))
            need(sm['rng_state'] == format(scalars[1], '016x') and sm['engram_position'] == scalars[11] and
                 sm['engram_tail'] == tail, 'Raw Engram/RNG metadata differs')
            expected = {'context-scalars.u64le': 120, 'engram-tail.u32le': 12,
                        'prefix-tokens.i32le': scalars[6] * 4, 'anchor-tokens.i32le': scalars[7] * 4,
                        'tokens.i32le': 256 * 4, 'output.bin': r['output_bytes']}
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
                    for layer in range(40): expected[f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32'] = min(position, 128) * 2048
                    for owner in range(3):
                        for kind in ('kv', 'score'): expected[f'snapshot-{slot:02d}-owner-{owner}-previous-{kind}.f32'] = 2048
            need(set(files) == set(expected) and all(files[k]['bytes'] == size for k, size in expected.items()),
                 'Complete active raw state file coverage/extent differs')
            need((payload / 'tokens.i32le').read_bytes() == struct.pack('<256i', *r['token_ids']) and
                 fnv((payload / 'output.bin').read_bytes()) == r['output_bytes_fnv1a64'],
                 'Complete original token/text bytes differ from native record')
            witnesses.append(dict(token_ids=r['token_ids'], output_bytes=r['output_bytes'],
                                  output_hash=r['output_bytes_fnv1a64'], logits_sha256=sha(logits / ('run-' + str(r['run']) + '.f32')),
                                  state_metadata=manifest['metadata'], state_files=files))
        need(all(w == witnesses[0] for w in witnesses), 'Same arm did not reproduce exact tokens/logits/state')
        result = dict(name=label, stage=str(stage), samples=samples, witness=witnesses[0],
                      payload_roots=payload_roots, logit_paths=logit_paths,
                      binary_sha256=sha(stage / 'bin/bench_output_a_fullmodel'),
                      metallib_sha256=sha(stage / 'bin/myds.metallib'))
        compare_arms(result, result)
        write_json(directory / 'review.json', result)
        return result

    def experiment(self):
        correctness, self.oracles = [], {}
        for workload in ('short', 'business'):
            group = [self.arm('correctness', workload, name, setting, 1)
                     for name, setting in (('gold', 0), ('off', 0), ('on', 1))]
            for arm in group: compare_arms(group[0], arm)
            self.oracles[workload] = group[0]
            correctness.append(dict(workload=workload, passed=True, arms=[r['name'] for r in group]))
        self.summary.update(fullmodel_run=True, correctness_pass=True, correctness=correctness)
        write_json(self.a.out / 'correctness-gate.json', correctness)
        reports = []
        for workload in ('short', 'business'):
            group = [self.arm('performance', workload, name, setting, 3)
                     for name, setting in (('A1', 0), ('B', 1), ('A2', 0))]
            for arm in group: compare_arms(self.oracles[workload], arm)
            d = [statistics.median(r['decode_s'] for r in x['samples'][1:]) for x in group]
            p = [statistics.median(r['prefill_s'] for r in x['samples'][1:]) for x in group]
            drift = (d[2] / d[0] - 1) * 100
            saved = (1 - d[1] / min(d[0], d[2])) * 100
            pf_regression = (p[1] / min(p[0], p[2]) - 1) * 100
            pf_drift = (p[2] / p[0] - 1) * 100
            reports.append(dict(workload=workload, decode_seconds=dict(zip(('A1', 'B', 'A2'), d)),
                                prefill_seconds=dict(zip(('A1', 'B', 'A2'), p)),
                                control_drift_percent=drift, conservative_decode_time_saved_percent=saved,
                                prefill_control_drift_percent=pf_drift,
                                conservative_prefill_regression_percent=pf_regression,
                                gate_pass=abs(drift) <= 2 and abs(pf_drift) <= 2 and saved > 2 and pf_regression <= 2))
        self.summary.update(performance=reports, performance_gate_pass=all(r['gate_pass'] for r in reports))

    def run(self):
        status, lock = 'failed', None
        def interrupted(number, frame):
            if self.recovering: self.event('deferred_signal_during_restoration', signal=number)
            else: raise KeyboardInterrupt('Fullmodel interrupted ' + str(number))
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, interrupted)
        try:
            lock = open('/tmp/myds-codex-decode-window.lock', 'a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.preflight()
            if self.a.prepare:
                status = 'prepared'
                self.summary.update(service_operations=False, gpu_work_submitted=False)
            else:
                self.thread = threading.Thread(target=self.monitor, daemon=True); self.thread.start()
                self.stop(); self.summary['fullmodel_run'] = True
                self.experiment()
                status = 'fullmodel_passed_review_required' if self.summary['performance_gate_pass'] else 'fullmodel_rejected_performance'
                for path, digest in self.input_hashes.items():
                    need(sha(path) == digest, 'Frozen input changed: ' + path)
        except BaseException as exc:
            self.summary['error'] = repr(exc); self.event('window_failure', error=repr(exc)); status = 'failed'
        finally:
            self.recovering = True
            try:
                self.drain()
                if self.stopped: self.summary['restoration'] = self.restore()
                else:
                    if self.before and 'model' in self.summary: self.unchanged()
                    self.summary['restoration'] = dict(restored=True, service_not_stopped=True)
            except BaseException as exc:
                status = 'failed'; self.summary['restoration'] = dict(restored=False, error=repr(exc))
                self.event('restoration_failure', error=repr(exc))
            self.done.set()
            if self.thread: self.thread.join(timeout=15)
            if self.fault.is_set() or (self.thread and self.thread.is_alive()): status = 'failed'
            self.summary.update(status=status, production_adoption=False)
            write_json(self.a.out / 'summary.json', self.summary); self.event('window_complete', status=status)
            if lock: lock.close()
        return 1 if status == 'failed' else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stage', 'gold-stage', 'model', 'out', 'plan', 'component-review'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, default=8187)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    a = p.parse_args(); need(sys.platform == 'darwin', 'macOS required')
    for name in ('stage', 'gold_stage', 'model', 'out', 'plan', 'component_review'):
        value = getattr(a, name); need(value.is_absolute(), 'Absolute paths required')
        setattr(a, name, value.resolve())
    for stage in (a.stage, a.gold_stage): need(Path('/tmp').resolve() in stage.parents, 'Isolated /tmp stages required')
    a.out.mkdir(parents=True, exist_ok=False); os.chmod(a.out, 0o700)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try: os.setsid()
    except PermissionError: pass
    with (a.out / 'runner.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2); os.dup2(null.fileno(), 0)
    return FullmodelWindow(a).run()


if __name__ == '__main__': sys.exit(main())
