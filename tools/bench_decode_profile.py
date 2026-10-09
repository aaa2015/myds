#!/usr/bin/env python3
"""Run an isolated diagnostic build on an idle Mac; never manage services.

This is separate from the uninstrumented performance runner. CB/host costs
overlap and must not be added. Optional Metal System Trace starts during run1
steady decode, not model loading or prefill. All subprocesses are owned and
terminated on failure; a surrounding exclusive controller restores production.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bench_reproduce', ROOT / 'tools/bench_reproduce.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def summarize(raw, native):
    rows = records(raw)
    native_samples = [r for r in records(native) if r.get('kind') == 'sample']
    samples = {r['run']: r for r in native_samples}
    if not samples or len(samples) != len(native_samples):
        raise ValueError('missing or duplicate native sample runs')
    if any(r.get('run') not in samples for r in rows):
        raise ValueError('diagnostic row has no native sample run')
    reports = []
    for run, sample in sorted(samples.items()):
        rr = [r for r in rows if r.get('run') == run]
        ss = [r for r in rr if r['kind'] == 'decode_diag_summary']
        if len(ss) != 1 or any(ss[0][k] for k in ('overflow', 'pending', 'invalid_gpu')):
            raise ValueError(f'incomplete GPU diagnostic run {run}: {ss}')
        all_cb = [r for r in rr if r['kind'] == 'decode_diag_cb']
        if len(all_cb) != ss[0]['submitted_cbs']:
            raise ValueError('CB count differs from summary')
        if len({r['cb'] for r in all_cb}) != len(all_cb):
            raise ValueError('duplicate command buffer record')
        for r in all_cb:
            if r.get('gpu_valid') is not True or r.get('status') != 4 or \
                    not all(isinstance(r[k], (int, float)) and math.isfinite(r[k])
                            for k in ('gpu_start_s', 'gpu_end_s', 'gpu_span_ms',
                                      'encode_wall_ms', 'commit_cpu_ms', 'wait_cpu_ms')) or \
                    r['gpu_start_s'] <= 0 or r['gpu_end_s'] <= r['gpu_start_s'] or \
                    any(r[k] < 0 for k in ('gpu_span_ms', 'encode_wall_ms', 'commit_cpu_ms', 'wait_cpu_ms')):
                raise ValueError('invalid command buffer timestamps/status')
            if not math.isclose(r['gpu_span_ms'], 1000*(r['gpu_end_s']-r['gpu_start_s']), abs_tol=0.00001):
                raise ValueError('GPU span differs from timestamps')
        # Match the native N-1 callback denominator, excluding prompt-final
        # forward and any scalar prefill fallback, irrespective of labels.
        first = sample['prompt_tokens'] - 1
        last = first + sample['decode_intervals']
        cbs = [r for r in all_cb if r['phase'] == 'decode' and first < r['position'] <= last]
        by_pos = defaultdict(list)
        for r in cbs:
            by_pos[r['position']].append(r)
        if len(by_pos) != sample['decode_intervals']:
            raise ValueError(f'decode position coverage differs: {len(by_pos)} vs {sample["decode_intervals"]}')
        token_rows = []
        for pos, cb in sorted(by_pos.items()):
            ordered = sorted(cb, key=lambda r: r['gpu_start_s'])
            union = 0.0
            end = None
            for r in ordered:
                begin, finish = r['gpu_start_s'], r['gpu_end_s']
                union += max(0, finish - max(begin, end if end is not None else begin))
                end = max(finish, end if end is not None else finish)
            span = max(r['gpu_end_s'] for r in ordered) - ordered[0]['gpu_start_s']
            token_rows.append({'position': pos, 'cbs': len(cb),
                               'gpu_cb_sum_ms': sum(r['gpu_span_ms'] for r in cb),
                               'gpu_union_ms': union*1000, 'gpu_envelope_ms': span*1000,
                               'gpu_gap_ms': max(0, span-union)*1000,
                               'encode_wall_ms': sum(r['encode_wall_ms'] for r in cb),
                               'commit_cpu_ms': sum(r['commit_cpu_ms'] for r in cb),
                               'wait_cpu_ms': sum(r['wait_cpu_ms'] for r in cb)})
        hosts = defaultdict(float)
        for r in rr:
            if r['kind'] == 'decode_diag_host' and r['phase'] == 'decode' and first < r['position'] <= last:
                hosts[r['stage']] += r['cpu_ms'] / sample['decode_intervals']
        means = {k: statistics.mean(t[k] for t in token_rows) for k in token_rows[0] if k != 'position'}
        reports.append({'run': run, 'warmup': sample['warmup'], 'decode_intervals': sample['decode_intervals'],
                        'mean_per_interval': means, 'host_ms_per_interval': dict(hosts),
                        'token_rows': token_rows, 'summary': ss[0]})
    return {'scope': 'CB elapsed/union/critical envelope and host time; costs overlap; dispatch counts are not GPU time or traffic',
            'reports': reports}


def stop(process):
    if not process:
        return
    # An exited leader can still leave children in its owned process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.05)
    raise RuntimeError('owned process group was not drained: '+str(process.pid))


def run(args):
    manifest = ROOT / 'decode-diag-manifest.json'
    if not manifest.is_file():
        raise ValueError('only a prepared isolated diagnostic tree may run this wrapper')
    frozen = json.loads(manifest.read_text())
    for name, digest in frozen['output_sha256'].items():
        if bench.sha(ROOT / name) != digest:
            raise ValueError('diagnostic source changed: '+name)
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents or out in ROOT.parents or out.exists():
        raise ValueError('output must be a new directory separate from diagnostic source')
    environment = {k: v for k, v in os.environ.items() if not k.startswith('MYDS_')}
    environment.update(bench.DEFAULT_ENV)
    environment.update(bench.ARM_ENV['target'])
    for expression in args.env:
        key, sep, value = expression.partition('=')
        if not sep or not key.startswith('MYDS_'):
            raise ValueError('invalid env override: '+expression)
        environment[key] = value
    bench.require_uninstrumented_environment(environment)
    for key in ('MYDS_PREFIX_CACHE', 'MYDS_KV_DSK', 'MYDS_PLD', 'MYDS_BATCH2_SPEC', 'MYDS_SPEC_SERIAL_PROBE'):
        if environment.get(key) != '0':
            raise ValueError('diagnostic requires target/cache-off: '+key)
    if args.mode != 'off':
        environment['MYDS_PROFILE_DECODE'] = 'cb'
        environment['MYDS_PROFILE_DECODE_FILE'] = str(out / 'decode-diag.jsonl')
    if args.mode == 'trace':
        environment['MYDS_PROFILE_DECODE_TRACE'] = '1'
    bench.check_idle(8181)
    out.mkdir(parents=True)
    native = ROOT / 'bin/bench_phase'
    argv = [str(native), str(args.model.resolve()), str(args.request.resolve()),
            str(args.repeats), str(args.max_new_tokens), str(args.context)]
    state = {'scope': 'diagnostic only, not a performance candidate acceptance run',
             'mode': args.mode, 'argv': argv, 'source': bench.source_state(),
             'binary_sha256': bench.sha(native), 'metallib_sha256': bench.sha(ROOT / 'bin/myds.metallib'),
             'manifest_sha256': bench.sha(manifest), 'request': bench.file_identity(args.request.resolve()),
             'request_sha256': bench.sha(args.request.resolve()), 'model': bench.file_identity(args.model.resolve()),
             'myds_environment': {k: v for k, v in environment.items() if k.startswith('MYDS_')},
             'before': bench.host_state(), 'status': 'running'}
    write(out / 'run-state.json', state)
    process = trace = reader = None
    thread_errors = []
    trace_info = {}
    trace_log = None
    started = time.monotonic()
    try:
        with (out / 'run-metrics.jsonl').open('wb') as stdout, (out / 'run.stderr.log').open('wb') as stderr:
            process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stdout, stderr=subprocess.PIPE,
                                       start_new_session=True)

            def consume():
                nonlocal trace, trace_log
                try:
                    for line in process.stderr:
                        stderr.write(line); stderr.flush()
                        if args.mode == 'trace' and line.startswith(b'DECODE_DIAG_TRACE_READY ') and trace is None:
                            trace_env = dict(environment, DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer')
                            trace_argv = ['/usr/bin/xcrun', 'xctrace', 'record', '--template', 'Metal System Trace',
                                          '--attach', str(process.pid), '--time-limit', '5s', '--no-prompt',
                                          '--output', str(out / 'steady-decode.trace')]
                            trace_log = (out / 'xctrace.log').open('wb')
                            trace_info.update(argv=trace_argv, marker=line.decode().strip(), target_pid=process.pid)
                            trace = subprocess.Popen(trace_argv, env=trace_env, stdout=trace_log,
                                                     stderr=subprocess.STDOUT, start_new_session=True)
                except Exception as error:
                    thread_errors.append(repr(error))

            reader = threading.Thread(target=consume, daemon=True)
            reader.start()
            rc = process.wait(timeout=args.timeout)
            reader.join(timeout=15)
            if reader.is_alive() or thread_errors:
                raise RuntimeError('stderr/trace reader failed: '+str(thread_errors))
            if rc != 0:
                raise RuntimeError('native harness exit '+str(rc))
        samples = [r for r in records(out / 'run-metrics.jsonl') if r.get('kind') == 'sample']
        if len(samples) != args.repeats+1 or not all(r['sample_valid'] for r in samples):
            raise ValueError('missing/invalid native samples')
        state['summary'] = {'formal_ms_per_token': [1000*r['decode_s']/r['decode_intervals'] for r in samples if not r['warmup']]}
        if args.mode != 'off':
            write(out / 'decode-diag-summary.json', summarize(out / 'decode-diag.jsonl', out / 'run-metrics.jsonl'))
        if args.mode == 'trace':
            if trace is None:
                raise RuntimeError('steady-decode trace marker was never reached')
            trace_rc = trace.wait(timeout=90)
            trace_info['returncode'] = trace_rc
            if trace_rc:
                raise RuntimeError('xctrace record failed; CB diagnostics remain available')
            export = ['/usr/bin/xcrun', 'xctrace', 'export', '--input', str(out / 'steady-decode.trace'), '--toc']
            with (out / 'trace-toc.xml').open('wb') as toc:
                subprocess.run(export, env=dict(environment, DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer'),
                               stdout=toc, stderr=subprocess.PIPE, check=True, timeout=60)
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        cleanup_errors = []
        try:
            stop(process)
        except BaseException as error:
            cleanup_errors.append('native: '+repr(error))
        if reader:
            reader.join(timeout=15)
            if reader.is_alive():
                cleanup_errors.append('stderr reader remains alive')
        try:
            stop(trace)
        except BaseException as error:
            cleanup_errors.append('trace: '+repr(error))
        if trace_log:
            trace_log.close()
        state.update(elapsed_s=time.monotonic()-started, trace=trace_info, cleanup_errors=cleanup_errors)
        try:
            state['after'] = bench.host_state()
        except Exception as error:
            state['after_error'] = repr(error)
        if cleanup_errors:
            state['status'] = 'failed'
        state['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': bench.sha(p)}
                          for p in out.iterdir() if p.is_file() and p.name != 'run-state.json'}
        write(out / 'run-state.json', state)
        if cleanup_errors:
            raise RuntimeError('diagnostic cleanup failed: '+str(cleanup_errors))
    print(json.dumps({'status': state['status'], 'out': str(out), 'summary': state['summary']}))


def main():
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            return  # allow the first interruption's finally cleanup to finish
        interrupted = True
        raise RuntimeError('interrupted by signal '+str(signum))

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('off', 'cb', 'trace'), default='cb')
    p.add_argument('--repeats', type=int, choices=(3,4), default=3)
    p.add_argument('--max-new-tokens', type=int, choices=(256,512), default=256)
    p.add_argument('--context', type=int, default=262144)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--env', action='append', default=[])
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
