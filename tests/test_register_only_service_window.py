#!/usr/bin/env python3
"""CPU-only window contracts. Every process/query/signal/network call is mocked."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('register_service_window', ROOT / 'tools/run_register_only_service_window.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Child:
    def __init__(self, pid, callback=None, restore=False):
        self.pid = pid; self.callback = callback; self.restore = restore
        self.alive = True; self.returncode = None; self.signals = []
    def poll(self):
        return None if self.alive else self.returncode
    def wait(self, timeout=None):
        if self.callback:
            callback, self.callback = self.callback, None
            result = callback()
        else:
            result = self.returncode if self.returncode is not None else 0
        self.alive = False; self.returncode = result
        return result
    def send_signal(self, sig):
        self.signals.append(sig); self.alive = False; self.returncode = 130


class WindowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='register-service-CPU-')
        self.addCleanup(temp.cleanup)
        base = Path(temp.name).resolve()
        self.cwd = base / 'original source'; (self.cwd / 'bin').mkdir(parents=True)
        self.exe = self.cwd / 'bin/myds'; self.exe.write_bytes(b'original CPU fixture binary')
        self.lib = self.exe.parent / 'myds.metallib'; self.lib.write_bytes(b'CPU fixture library')
        self.conf = self.cwd / 'config with spaces.conf'; self.conf.write_text('MYDS_TEST=with spaces\n')
        self.args = types.SimpleNamespace(root=base / 'window', cwd=self.cwd, pid=900001,
            port=8187, protect=[], command=['CPU-fixture-runner', '--argument', 'with spaces'],
            stop_timeout=.002, drain_timeout=.005, restore_timeout=.01)
        self.captured = dict(executable=str(self.exe),
            argv=['./bin/myds', '--config', str(self.conf), '--port', '8187', '--model', 'model with spaces.gguf'],
            environment={'MYDS_TEST': 'one two=three', 'PATH': '/fixture path/bin',
                         'PRIVATE_SECRET': 'secret never printed', 'EMPTY': ''})
        self.active = True; self.unknown = []; self.launched = []; self.callback = None
        self.stop_exits = True
        self.window = m.Window(self.args)

    def proc_list(self):
        if self.active:
            return [(900001, 'myds')]
        if self.unknown:
            return self.unknown
        if self.launched and self.launched[-1][1].restore:
            return [(900003, 'myds')]
        return []

    def kill(self, pid, sig):
        self.assertEqual(pid, 900001)
        self.assertIn(sig, (0, signal.SIGTERM))
        if sig == signal.SIGTERM:
            if self.stop_exits:
                self.active = False
        elif not self.active:
            raise ProcessLookupError()

    def query(self, *args):
        if args[0] == 'ps' and args[-1] == 'nice=':
            return '0'
        if args[0] == 'ps' and args[-1] == 'lstart=':
            return 'frozen start'
        if args[0] == 'lsof' and '-sTCP:LISTEN' in args:
            return 'p900003\nf1'
        # stdio capture: an empty answer means "no descriptor resolves to a file",
        # which selects the fallback branch. The branch that reuses the original
        # stdio targets is covered by tools/test_run_service_window_nice_cpu.py
        # against real files.
        if args[0] == 'lsof' and '-d' in args and '0,1,2' in args:
            return ''
        raise AssertionError('unexpected real query: ' + repr(args))

    def popen(self, command, **kwargs):
        restore = command == self.captured['argv']
        child = Child(900003 if restore else 900002, None if restore else self.callback, restore)
        self.launched.append((kwargs, child))
        return child

    def run_window(self, popen=None, writer=None, capture=None):
        # available_gib/process_rss_gib are pinned so the released-memory gate
        # cannot block here: a resident size of 0 makes the target equal the
        # pre-stop reading, satisfied on the first sample. The gate's own logic is
        # exercised with injected readings in
        # tools/test_run_service_window_nice_cpu.py.
        output = io.StringIO()
        response = io.StringIO('{"status":"ok"}'); response.status = 200
        with mock.patch.object(m, 'process_arguments', side_effect=capture or (lambda pid: copy.deepcopy(self.captured))), \
             mock.patch.object(m, 'process_cwd', return_value=self.cwd), \
             mock.patch.object(m, 'processes', side_effect=self.proc_list), \
             mock.patch.object(m, 'query', side_effect=self.query), \
             mock.patch.object(m.os, 'kill', side_effect=self.kill) as kill, \
             mock.patch.object(m.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1, stdout='')), \
             mock.patch.object(m.subprocess, 'Popen', side_effect=popen or self.popen), \
             mock.patch.object(m, 'available_gib', return_value=200.0), \
             mock.patch.object(m, 'process_rss_gib', return_value=0.0), \
             mock.patch.object(m.urllib.request, 'urlopen', return_value=response), \
             mock.patch.object(m.time, 'sleep'), \
             mock.patch.object(m.signal, 'signal'), \
             mock.patch.object(m.signal, 'getsignal', return_value=signal.SIG_DFL), \
             mock.patch.object(m, 'write', side_effect=writer or m.write), \
             contextlib.redirect_stdout(output):
            code = self.window.run()
        self.output = output.getvalue(); self.kill_calls = kill.call_args_list
        return code

    def test_success_exact_full_environment_config_backup_and_private_modes(self):
        self.assertEqual(self.run_window(), 0)
        self.assertTrue(self.window.state['restored'])
        kwargs, child = self.launched[-1]
        self.assertEqual(kwargs['env'], self.captured['environment'])
        self.assertEqual(kwargs['executable'], str(self.exe))
        lease = self.args.root / 'lease.json'
        self.assertEqual(lease.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.args.root.stat().st_mode & 0o777, 0o700)
        identities = json.loads(lease.read_text())['identities']
        self.assertIn(str(self.conf), identities)
        for item in identities.values():
            if item['exists']:
                self.assertEqual(Path(item['backup']).stat().st_mode & 0o777, 0o600)
                self.assertEqual(m.sha(item['backup']), item['sha256'])
        self.assertNotIn('PRIVATE_SECRET', (self.args.root / 'summary.json').read_text())
        self.assertNotIn('secret never printed', self.output)
        self.assertEqual([c.args for c in self.kill_calls if c.args[1]], [(900001, signal.SIGTERM)])

    def test_nonzero_experiment_still_restores_and_preserves_returncode(self):
        self.callback = lambda: 7
        self.assertEqual(self.run_window(), 7)
        self.assertTrue(self.window.state['restored'])

    def test_experiment_spawn_failure_restores(self):
        def launch(command, **kwargs):
            if command == self.args.command:
                raise OSError('CPU injected spawn failure')
            return self.popen(command, **kwargs)
        self.assertEqual(self.run_window(popen=launch), 1)
        self.assertTrue(self.window.state['restored'])

    def test_failure_immediately_after_stop_enters_restore(self):
        real_wait = self.window.wait_original_exit
        self.window.wait_original_exit = mock.Mock(side_effect=[OSError('post-stop wait failure'), None])
        self.assertEqual(self.run_window(), 1)
        self.assertTrue(self.window.state['restored'])
        self.assertEqual(self.window.wait_original_exit.call_count, 2)

    def test_post_stop_lease_io_failure_cannot_skip_restore(self):
        writer = m.write; injected = []
        def fail_once(path, data):
            if self.window.state['stop_signal_sent'] and not injected:
                injected.append(True); raise OSError('CPU injected write failure')
            return writer(path, data)
        self.assertEqual(self.run_window(writer=fail_once), 0)
        self.assertTrue(self.window.state['restored'])
        self.assertEqual(self.window.state['lease_write_errors'], ['OSError'])

    def test_canonical_drift_blocks_restore_and_keeps_original_backup(self):
        def mutate():
            self.exe.write_bytes(b'external replacement'); return 0
        self.callback = mutate
        self.assertEqual(self.run_window(), 1)
        self.assertTrue(self.window.state['restore_blocked'])
        self.assertEqual(len(self.launched), 1)
        saved = self.window.state['identities'][str(self.exe)]['backup']
        self.assertEqual(Path(saved).read_bytes(), b'original CPU fixture binary')
        self.assertEqual(self.exe.read_bytes(), b'external replacement')

    def test_unknown_child_never_signalled_and_blocks_overlapping_restore(self):
        def leave_unknown():
            self.unknown = [(900099, 'bench_unknown')]; return 3
        self.callback = leave_unknown
        self.assertEqual(self.run_window(), 1)
        self.assertTrue(self.window.state['restore_blocked'])
        self.assertEqual(len(self.launched), 1)
        self.assertTrue(all(call.args[0] == 900001 for call in self.kill_calls))

    def test_signal_during_experiment_drains_only_owned_parent(self):
        self.callback = lambda: self.window.handle_signal(signal.SIGTERM, None)
        self.assertEqual(self.run_window(), 1)
        self.assertTrue(self.window.state['restored'])
        self.assertEqual(self.launched[0][1].signals, [signal.SIGINT])
        self.window.handle_signal(signal.SIGTERM, None)
        self.assertEqual(self.window.state['deferred_recovery_signals'], [signal.SIGTERM])

    def test_signals_before_popen_returns_preserve_child_ownership_and_drain_before_restore(self):
        def launch(command, **kwargs):
            if command == self.args.command:
                child = self.popen(command, **kwargs)
                self.assertIsNone(self.window.experiment)
                self.window.handle_signal(signal.SIGINT, None)
                self.window.handle_signal(signal.SIGTERM, None)
                self.assertEqual(child.signals, [])
                return child
            experiment = self.launched[0][1]
            self.assertIs(self.window.experiment, experiment)
            self.assertEqual(experiment.signals, [signal.SIGINT])
            self.assertFalse(experiment.alive)
            return self.popen(command, **kwargs)
        self.assertEqual(self.run_window(popen=launch), 1)
        self.assertTrue(self.window.state['restored'])
        self.assertFalse(self.window.spawning_experiment)
        self.assertEqual(self.window.state['experiment_pid'], 900002)
        self.assertEqual(self.window.state['deferred_spawn_signals'], [signal.SIGINT, signal.SIGTERM])
        self.assertEqual([c.args for c in self.kill_calls if c.args[1]], [(900001, signal.SIGTERM)])

    def test_original_still_running_no_duplicate_and_no_sigkill(self):
        self.stop_exits = False
        self.assertEqual(self.run_window(), 1)
        self.assertTrue(self.window.state['restore_blocked'])
        self.assertFalse(self.launched)
        self.assertEqual([c.args for c in self.kill_calls if c.args[1]], [(900001, signal.SIGTERM)])

    def test_existing_window_not_overwritten(self):
        self.args.root.mkdir(); lease = self.args.root / 'lease.json'; lease.write_text('original lease')
        self.assertEqual(self.run_window(), 1)
        self.assertEqual(lease.read_text(), 'original lease')
        self.assertFalse(self.kill_calls)

    def test_identity_change_immediately_before_stop_sends_no_signal(self):
        calls = []
        def capture(pid):
            calls.append(pid); result = copy.deepcopy(self.captured)
            if len(calls) == 3:
                result['environment']['NEW_VARIABLE'] = 'different process identity'
            return result
        self.assertEqual(self.run_window(capture=capture), 1)
        self.assertFalse(self.kill_calls)
        self.assertFalse(self.launched)

    def test_higher_priority_library_creation_blocks_changed_load(self):
        def create_library():
            (self.cwd / 'metal').mkdir()
            (self.cwd / 'metal/myds.metallib').write_bytes(b'external higher priority library')
            return 0
        self.callback = create_library
        self.assertEqual(self.run_window(), 1)
        self.assertTrue(self.window.state['restore_blocked'])
        self.assertEqual(len(self.launched), 1)

    def test_owned_experiment_that_will_not_exit_blocks_restore(self):
        def launch(command, **kwargs):
            child = self.popen(command, **kwargs)
            child.wait = mock.Mock(side_effect=[RuntimeError('CPU interrupt'),
                subprocess.TimeoutExpired(command, self.args.stop_timeout)])
            child.send_signal = mock.Mock()
            return child
        self.assertEqual(self.run_window(popen=launch), 1)
        self.assertTrue(self.window.state['restore_blocked'])
        self.assertEqual(len(self.launched), 1)
        self.launched[0][1].send_signal.assert_called_once_with(signal.SIGINT)


class ParserTests(unittest.TestCase):
    def test_exact_spaces_equals_empty_arguments_and_environment(self):
        argv = ['./bin/myds', '--config', '/a path/myds.conf', '', '--port', '8187']
        env = {'MYDS_X': 'a b=c', 'EMPTY': '', 'TOKEN': 'hidden value'}
        raw = struct.pack('=i', len(argv)) + b'/a path/bin/myds\0\0\0'
        raw += b'\0'.join(os.fsencode(v) for v in argv) + b'\0'
        raw += b'\0'.join(os.fsencode(k + '=' + v) for k, v in env.items()) + b'\0\0'
        self.assertEqual(m.parse_procargs2(raw), dict(executable='/a path/bin/myds', argv=argv, environment=env))

    def test_duplicate_environment_refused(self):
        raw = struct.pack('=i', 1) + b'/bin/myds\0\0myds\0X=1\0X=2\0\0'
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            m.parse_procargs2(raw)

    def test_relative_kernel_execname_is_resolved_against_authoritative_path(self):
        raw = struct.pack('=i', 3) + b'./bin/myds\0\0./bin/myds\0--port\08187\0PATH=/a b/bin\0\0'
        captured = m.parse_procargs2(raw)
        self.assertEqual(captured['executable'], './bin/myds')
        with tempfile.TemporaryDirectory(prefix='register-path-CPU-') as directory:
            cwd = Path(directory).resolve()
            normalized = m.normalize_process_capture(captured, cwd / 'bin/myds', cwd)
            self.assertEqual(normalized['executable'], str(cwd / 'bin/myds'))
            self.assertEqual(normalized['argv'], ['./bin/myds', '--port', '8187'])
            self.assertEqual(normalized['environment'], {'PATH': '/a b/bin'})
            self.assertEqual(captured['executable'], './bin/myds')

    def test_normalized_restore_identity_allows_absolute_execname_with_original_argv(self):
        with tempfile.TemporaryDirectory(prefix='register-restore-path-CPU-') as directory:
            cwd = Path(directory).resolve(); executable = cwd / 'bin/myds'
            original = dict(executable='./bin/myds', argv=['./bin/myds', 'path with spaces'],
                            environment={'MYDS_X': 'two words'})
            restored = dict(original, executable=str(executable))
            self.assertEqual(m.normalize_process_capture(original, executable, cwd),
                             m.normalize_process_capture(restored, executable, cwd))
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                m.normalize_process_capture(original, cwd / 'another/myds', cwd)


if __name__ == '__main__':
    unittest.main()
