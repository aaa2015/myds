#!/usr/bin/env python3
"""Execute frozen Q, independently review it, then P and the HTTP bridge.

Run under run_register_only_service_window.py. Every result root is new;
no sample replacement or performance retries are implemented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


class StageSignals:
    """Record interruption without losing a just-created Popen child."""
    def __init__(self):
        self.received = []

    def handle(self, signum, frame):
        self.received.append(signum)

    def check(self):
        if self.received:
            raise InterruptedError('integration stages interrupted by signal ' +
                                   str(self.received[0]))


def run_owned_stage(name, argv, stdout, stderr, event, signals, cleanup_timeout=75):
    """Let the exact child run its own finally; never use subprocess.run kill."""
    child = None
    cleanup_started = None
    try:
        signals.check()
        # Signal handlers only record. A signal in Popen's return/assignment
        # boundary cannot discard the only ownership handle for the child.
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout,
                                 stderr=stderr, start_new_session=True)
        event('stage_child_started', stage=name, pid=child.pid)
        while True:
            if signals.received and cleanup_started is None and child.poll() is None:
                cleanup_started = time.monotonic()
                child.send_signal(signal.SIGINT)
                event('stage_interrupt_forwarded', stage=name, pid=child.pid,
                      signals=list(signals.received), cleanup_timeout_s=cleanup_timeout)
            try:
                code = child.wait(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                if cleanup_started is not None and time.monotonic() - cleanup_started >= cleanup_timeout:
                    raise RuntimeError(f'owned stage PID {child.pid} did not finish cleanup; no SIGKILL sent')
        event('stage_exit', stage=name, pid=child.pid, exit_code=code)
        signals.check()
        return code
    except BaseException:
        # Covers event/file failures and other exceptions as well as signals.
        # Avoid a second interrupt while the child's own cleanup is running.
        if child is not None and child.poll() is None:
            if cleanup_started is None:
                cleanup_started = time.monotonic()
                child.send_signal(signal.SIGINT)
            remaining = max(0, cleanup_timeout - (time.monotonic() - cleanup_started))
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f'owned stage PID {child.pid} remains; no SIGKILL or automatic next stage') from exc
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--http-tool', type=Path, required=True)
    p.add_argument('--inputs-root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    build = root / 'native-build-v2'
    reader = build / 'validation/review_register_only_integration_fullmodel.py'
    preparer = build / 'validation/prepare_register_only_integration_native.py'
    wrapper = build / 'validation/frozen-driver/register-only-fullmodel-window-wrapper-v2.py'
    progress = root / 'integration-progress.jsonl'
    assert not progress.exists(), 'preserve prior attempts; new root required'
    signals = StageSignals()
    previous_handlers = {s: signal.signal(s, signals.handle)
                         for s in (signal.SIGINT, signal.SIGTERM)}

    def event(kind, **kw):
        item = dict(kind=kind, epoch=time.time(), **kw)
        with progress.open('a') as f:
            f.write(json.dumps(item) + '\n')
        print(json.dumps(item), flush=True)

    def run(name, argv):
        signals.check()
        event('stage_start', stage=name, argv=argv)
        with (root / (name + '.stdout')).open('xb') as out, (root / (name + '.stderr')).open('xb') as err:
            code = run_owned_stage(name, argv, out, err, event, signals)
        if code:
            raise RuntimeError(name + ' failed; preserve raw and stop without retry')

    def review(stage, initialize=False):
        directory = root / (stage + '-freeze-v1')
        output = directory / ('launch-preflight.json' if initialize else 'review.json')
        argv = [sys.executable, '-B', str(reader), '--protocol', str(directory / 'protocol.json'),
                '--window-root', str(directory / (stage + '-window')),
                '--private-build-manifest', str(build / 'private-build-manifest.json'),
                '--target-build-manifest', str(build / 'target-build-manifest.json'), '--out', str(output)]
        if initialize:
            argv.append('--initialize-only')
        run(stage + ('-initialize' if initialize else '-review'), argv)
        report = json.loads(output.read_text())
        assert report.get('evidence_error') is None, report.get('evidence_error')
        if not initialize:
            assert report['review_completed'] is True and report['numerical_gates_passed'] is True
            if stage == 'Q':
                assert report['qualification_passed'] is True and report['two_workload_qualification_completed'] is True
            else:
                assert report['native_prefill_gate_passed'] is True, 'new-build native performance gate failed'
        return report

    try:
        event('begin', tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        review('Q', True)
        run('Q-collect', [sys.executable, '-B', str(wrapper), '--protocol', str(root / 'Q-freeze-v1/protocol.json')])
        review('Q')
        run('P-freeze', [sys.executable, '-B', str(preparer), 'P',
             '--target-build-manifest', str(build / 'target-build-manifest.json'),
             '--qualification-protocol', str(root / 'Q-freeze-v1/protocol.json'),
             '--qualification-review', str(root / 'Q-freeze-v1/review.json'),
             '--out', str(root / 'P-freeze-v1')])
        review('P', True)
        run('P-collect', [sys.executable, '-B', str(wrapper), '--protocol', str(root / 'P-freeze-v1/protocol.json')])
        review('P')
        qp = json.loads((root / 'Q-freeze-v1/protocol.json').read_text())
        run('HTTP-bridge', [sys.executable, '-B', str(a.http_tool),
             '--stage-root', str(root / 'HTTP-bridge-v1'), '--gold-exe', str(root / 'gold-service/myds'),
             '--candidate-exe', str(root / 'candidate-service/myds'), '--config', str(root / 'native.conf'),
             '--model', qp['model']['path'], '--inputs-root', str(a.inputs_root), '--port', '8198'])
        event('complete', native_numerical_and_performance_passed=True, HTTP_tool_exit=0, deployment_performed=False)
        return 0
    except BaseException as exc:
        event('stopped', error=type(exc).__name__ + ': ' + str(exc),
              received_signals=list(signals.received), deployment_performed=False)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    sys.exit(main())
