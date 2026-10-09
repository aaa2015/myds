#!/usr/bin/env python3
"""CPU-only window ownership/recovery tests; never access a live service/GPU."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'docs/benchmarks/2026-10-09-decode/scripts/run_macstudio_window.py'
spec = importlib.util.spec_from_file_location('macstudio_window_test', SCRIPT)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class FixtureWindow(controller.Window):
    """Production inspection/kill/start are CPU fixtures; step children are real."""
    def __init__(self, args):
        super().__init__(args)
        self.trace = []
        self.active = True

    def preflight(self):
        self.frozen_inputs = self.freeze_inputs()
        self.snapshot = {'pid': 900001, 'argv': ['/fixture/original/myds', '--port', '8181'],
                         'environment': {'MYDS_DECODE_SPLIT': '20', 'MYDS_PLD': '1'},
                         'hashes': {'fixture': 'original'},
                         'restore_argv': ['/bin/zsh', str(self.fleet / 'start_fleet.sh')]}
        self.trace.append('preflight')
        return self.snapshot

    def stop_production(self):
        self.verify_frozen_inputs()
        self.stop_requested = True
        self.active = False
        self.trace.append('stop')

    def listener(self):
        return 900001 if self.active else None

    def processes(self):
        return []

    def no_other_inference(self, permitted_pid=None):
        pass

    def hashes(self):
        return self.snapshot['hashes']

    def restore(self):
        self.recovering = True
        self.trace.append('restore')
        self.active = True
        return {'restored': True, 'state': {'environment': self.snapshot['environment']}}


class WindowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='myds-window-cpu-', dir='/tmp')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source'
        (self.source / 'tools').mkdir(parents=True)
        (self.source / 'bin').mkdir()
        # A CPU-only executable fixture, explicitly not the real model runner.
        (self.source / 'tools/bench_reproduce.py').write_text("print('CPU plan fixture; no inference')\n")
        for name in ('bench_phase', 'myds.metallib'):
            (self.source / 'bin' / name).write_text('CPU identity fixture only\n')
        self.request = self.root / 'request.json'
        self.request.write_text('{}\n')
        self.model = self.root / 'model.gguf'
        self.model.write_text('Not a GGUF; identity fixture only\n')
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        subprocess.run(['git', '-C', str(self.source), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.source), '-c', 'user.name=CPU Fixture',
                        '-c', 'user.email=cpu@example.invalid', 'commit', '-qm', 'CPU fixture'], check=True)
        self.args = types.SimpleNamespace(source=self.source, out=self.root / 'window', model=self.model,
            support=None, plan=self.root / 'plan.json', gate=[], gate_timeout=3, step_timeout=3,
            restore_timeout=.01, restore_attempts=2, production=self.root / 'production',
            fleet=self.root / 'fleet', port=8181, execute=True)
        self.args.plan.write_text(json.dumps([{'name': 'target-a1', 'argv': [sys.executable,
            str(self.source / 'tools/bench_reproduce.py'), 'run', '--model', str(self.model),
            '--request', str(self.request), '--out', str(self.args.out / 'target-a1')]}]))
        self.args.gate = [json.dumps([sys.executable, '-c', "print('real harmless CPU gate')"])]

    def invoke(self, window):
        with mock.patch.object(controller, 'host_state', return_value={'swap': 'CPU fixture', 'memory': 'CPU fixture'}), \
                contextlib.redirect_stdout(io.StringIO()):
            return window.run()

    def result(self):
        return json.loads((self.args.out / 'window-result.json').read_text())

    def test_dry_run_never_stops_or_launches(self):
        self.args.execute = False
        window = FixtureWindow(self.args)
        with mock.patch.object(window, 'run_step', side_effect=AssertionError('dry run executed a gate/measurement')):
            self.assertEqual(self.invoke(window), 0)
        self.assertEqual(window.trace, ['preflight'])
        self.assertTrue(window.active)
        self.assertEqual(self.result()['status'], 'dry_run')
        self.assertFalse((self.args.out / 'gate-1.stdout.log').exists())

    def test_real_cpu_gate_and_plan_order_restore_original(self):
        window = FixtureWindow(self.args)
        self.assertEqual(self.invoke(window), 0)
        events = [json.loads(line) for line in (self.args.out / 'window.jsonl').read_text().splitlines()]
        self.assertEqual([event['name'] for event in events if event['kind'] == 'step_start'], ['gate-1', 'target-a1'])
        self.assertIn('real harmless CPU gate', (self.args.out / 'gate-1.stdout.log').read_text())
        self.assertIn('CPU plan fixture', (self.args.out / 'target-a1.stdout.log').read_text())
        self.assertEqual(window.trace, ['preflight', 'stop', 'restore'])
        self.assertTrue(self.result()['restoration']['restored'])

    def test_failed_gate_fails_fast_but_restores(self):
        self.args.gate = [json.dumps({'argv': [sys.executable, '-c', 'import sys; sys.exit(7)'],
                                    'timeout': 2, 'outputlogname': 'fullmodel-gate'})]
        window = FixtureWindow(self.args)
        self.assertEqual(self.invoke(window), 1)
        self.assertTrue(self.result()['restoration']['restored'])
        self.assertIn('exited 7', self.result()['error'])
        self.assertTrue((self.args.out / 'fullmodel-gate.stderr.log').is_file())
        self.assertFalse((self.args.out / 'target-a1.stdout.log').exists())

    def test_timeout_reaps_real_cpu_child_and_restores(self):
        self.args.gate = [json.dumps({'argv': [sys.executable, '-c', 'import time; time.sleep(20)'], 'timeout': .05})]
        window = FixtureWindow(self.args)
        self.assertEqual(self.invoke(window), 1)
        self.assertIsNone(window.child)
        self.assertTrue(self.result()['restoration']['restored'])
        self.assertIn('TimeoutExpired', self.result()['error'])

    def test_actual_sigterm_enters_finally_and_second_signal_is_deferred(self):
        self.args.gate = [json.dumps([sys.executable, '-c', 'import time; time.sleep(20)'])]
        window = FixtureWindow(self.args)
        original = signal.getsignal(signal.SIGTERM)
        real_restore = window.restore
        def restore():
            window.handle_signal(signal.SIGINT, None)
            return real_restore()
        window.restore = restore
        timer = threading.Timer(.1, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            self.assertEqual(self.invoke(window), 1)
        finally:
            timer.cancel()
            timer.join()
        self.assertTrue(self.result()['restoration']['restored'])
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)
        self.assertIn('signal 15', self.result()['error'])

    def test_cleanup_failure_prevents_overlapping_restore_launch(self):
        self.args.gate = [json.dumps([sys.executable, '-c', 'import sys; sys.exit(7)'])]
        window = FixtureWindow(self.args)
        window.drain_measurements = mock.Mock(side_effect=RuntimeError('owned child remains alive'))
        self.assertEqual(self.invoke(window), 1)
        self.assertNotIn('restore', window.trace)
        self.assertFalse(self.result()['restoration']['restored'])

    def test_request_and_plan_are_frozen_before_stop(self):
        for path in (self.request, self.args.plan):
            with self.subTest(path=path.name):
                self.args.out = self.root / ('window-' + path.name)
                # Keep planned output inside each new window.
                plan = json.loads(self.args.plan.read_text())
                argv = plan[0]['argv']; argv[argv.index('--out') + 1] = str(self.args.out / 'target-a1')
                self.args.plan.write_text(json.dumps(plan))
                window = FixtureWindow(self.args)
                original = window.preflight
                def preflight():
                    result = original()
                    path.write_text(path.read_text() + '\n')
                    return result
                window.preflight = preflight
                self.assertEqual(self.invoke(window), 1)
                self.assertEqual(window.trace, ['preflight'])
                self.assertTrue(window.active)

    def test_restore_uses_frozen_split20_and_clears_measurement_environment(self):
        window = FixtureWindow(self.args)
        window.out.mkdir()
        window.preflight(); window.active = False
        window.verify_restored = mock.Mock(return_value={'pid': 900002})
        launcher = mock.Mock(pid=900002)
        with mock.patch.dict(os.environ, {'MYDS_DECODE_METRICS': '1', 'MYDS_BATCH2_SPEC': '1',
                                         'MYDS_DECODE_SPLIT': '4', 'DS4_DSPARK': '1'}), \
                mock.patch.object(controller.subprocess, 'Popen', return_value=launcher) as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            result = controller.Window.restore(window)
        self.assertTrue(result['restored'])
        kwargs = launch.call_args.kwargs
        self.assertEqual(launch.call_args.args[0], window.snapshot['restore_argv'])
        self.assertEqual(kwargs['env']['MYDS_DECODE_SPLIT'], '20')
        self.assertEqual(kwargs['env']['MYDS_PLD'], '1')
        self.assertNotIn('MYDS_DECODE_METRICS', kwargs['env'])
        self.assertNotIn('MYDS_BATCH2_SPEC', kwargs['env'])
        self.assertNotIn('DS4_DSPARK', kwargs['env'])
        self.assertTrue(kwargs['start_new_session'])

    def test_extra_frozen_file_content_change_stops_measurement_and_restores(self):
        # Outside Git, with unchanged length and mtime: only the requested full
        # content digest can detect this mutation after a real harmless gate.
        token_file = self.root / 'state-input-tokens.txt'
        gate_file = self.root / 'external-gate-binary'
        token_file.write_text('1111\n')
        gate_file.write_text('CPU binary identity fixture\n')
        self.args.freeze_file = [token_file, gate_file]
        program = ('import os,pathlib; p=pathlib.Path(__import__("sys").argv[1]); '
                   'st=p.stat(); p.write_text("2222\\n"); '
                   'os.utime(p, ns=(st.st_atime_ns,st.st_mtime_ns))')
        self.args.gate = [json.dumps([sys.executable, '-c', program, str(token_file)])]
        original = token_file.stat()
        window = FixtureWindow(self.args)
        self.assertEqual(self.invoke(window), 1)
        self.assertEqual(window.trace, ['preflight', 'stop', 'restore'])
        self.assertTrue(self.result()['restoration']['restored'])
        self.assertIn('identity changed during window', self.result()['error'])
        self.assertFalse((self.args.out / 'target-a1.stdout.log').exists())
        self.assertEqual(token_file.stat().st_size, original.st_size)
        self.assertEqual(token_file.stat().st_mtime_ns, original.st_mtime_ns)
        frozen = window.frozen_inputs['extra_files_sha256']
        self.assertEqual(set(frozen), {str(token_file.resolve()), str(gate_file.resolve())})
        self.assertNotEqual(frozen[str(token_file.resolve())], controller.sha(token_file))
        self.assertEqual(frozen[str(gate_file.resolve())], controller.sha(gate_file))

    def test_missing_extra_frozen_file_refuses_stop_and_legacy_structure_is_unchanged(self):
        window = FixtureWindow(self.args)
        self.assertNotIn('extra_files_sha256', window.freeze_inputs())
        self.args.freeze_file = []
        self.assertNotIn('extra_files_sha256', window.freeze_inputs())
        self.args.freeze_file = [self.root / 'missing-gate-binary']
        self.assertEqual(self.invoke(window), 1)
        self.assertEqual(window.trace, [])
        self.assertTrue(window.active)
        self.assertIn('Missing extra frozen input:', self.result()['error'])
        self.assertEqual(self.result()['restoration']['mode'], 'production_not_stopped')
        self.assertFalse((self.args.out / 'gate-1.stdout.log').exists())

    def test_restore_timeout_drains_before_retry(self):
        window = FixtureWindow(self.args)
        window.out.mkdir(); window.preflight(); window.active = False
        trace = []
        def launch(*args, **kwargs):
            trace.append('launch')
            return mock.Mock(pid=900002)
        def verify():
            if trace.count('launch') == 1:
                raise RuntimeError('first loader not ready')
            return {'pid': 900003}
        window.verify_restored = verify
        window.drain_launcher = lambda: trace.append('drain')
        with mock.patch.object(controller.subprocess, 'Popen', side_effect=launch), \
                mock.patch.object(controller.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            result = controller.Window.restore(window)
        self.assertTrue(result['restored'])
        self.assertEqual(trace, ['launch', 'drain', 'launch'])

    def test_recovery_cleanup_failure_aborts_retries(self):
        window = FixtureWindow(self.args)
        window.out.mkdir(); window.preflight(); window.active = False
        window.verify_restored = mock.Mock(side_effect=RuntimeError('not ready'))
        window.drain_launcher = mock.Mock(side_effect=RuntimeError('late owned loader still alive'))
        with mock.patch.object(controller.subprocess, 'Popen', return_value=mock.Mock(pid=900002)) as launch, \
                mock.patch.object(controller.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            result = controller.Window.restore(window)
        self.assertFalse(result['restored'])
        self.assertEqual(launch.call_count, 1)

    def test_exited_leader_does_not_skip_surviving_process_group(self):
        window = FixtureWindow(self.args)
        process = mock.Mock(pid=900002)
        alive = {'value': True}
        events = []
        def killpg(pid, sig):
            events.append(sig)
            if sig == 0 and not alive['value']:
                raise ProcessLookupError
            if sig == signal.SIGKILL:
                alive['value'] = False
        with mock.patch.object(controller.os, 'killpg', side_effect=killpg), \
                mock.patch.object(controller.time, 'monotonic', side_effect=[0., 6., 6.]), \
                mock.patch.object(controller.time, 'sleep'):
            window.stop_group(process)
        self.assertIn(signal.SIGTERM, events)
        self.assertIn(signal.SIGKILL, events)
        process.wait.assert_called_once_with(timeout=5)

    def test_runtime_snapshot_retains_only_myds_variables(self):
        with mock.patch.object(controller, 'command', return_value='myds MYDS_DECODE_SPLIT=20 TOKEN=private MYDS_PLD=1 PASSWORD=private'):
            self.assertEqual(controller.runtime_environment(900001), {'MYDS_DECODE_SPLIT': '20', 'MYDS_PLD': '1'})


if __name__ == '__main__':
    unittest.main()
