#!/usr/bin/env python3
"""CPU-only regression checks for the benchmark reproduction driver.

The accepted samples come from the committed exclusive Mac Studio measurements.
Runner tests replace Popen, host inspection, and exclusivity checks with explicit
CPU fixtures. Tiny temporary model/metallib files are identity fixtures only;
these tests never load a GGUF, create a Metal device, contact SSH, or run a model.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/benchmarks/2026-10-05-optimization'
WINDOW = EVIDENCE / 'raw/exclusive-20261005T022233Z'
HOST = {'platform': 'CPU test fixture', 'cpu': 'Apple M2 Ultra',
        'memory_bytes': 206158430208, 'os': 'CPU fixture macOS 26.5.2',
        'swap': 'CPU fixture: unchanged', 'vm_stat': 'CPU fixture: unchanged'}


def fnv1a64(payload):
    value = 14695981039346656037
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f'{value:016x}'


def read_samples(variant='base', workload='normal'):
    path = WINDOW / f'{variant}-{workload}-metrics.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_samples(path, records):
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))


class CompletedCPUHarness:
    """A Popen-shaped writer; it never executes its argv or opens a model."""
    pid = 987654321

    def __init__(self, argv, *, records, cwd, env, stdout, stderr, start_new_session):
        self.argv, self.cwd, self.env = argv, cwd, env
        self.returncode = 0
        self.start_new_session = start_new_session
        stdout.write(''.join(json.dumps(record) + '\n' for record in records))
        stderr.write('CPU harness fixture; no inference was executed.\n')

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


class BenchReproduceTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('myds_bench_reproduce_test',
                                                    ROOT / 'tools/bench_reproduce.py')
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)
        self.temporary = tempfile.TemporaryDirectory(prefix='myds-benchmark-cpu-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.binary = self.directory / 'bin/bench_phase'
        self.binary.parent.mkdir()
        self.binary.write_text('CPU identity fixture; never execute this file.\n')
        self.binary.chmod(0o755)
        (self.binary.parent / 'myds.metallib').write_bytes(b'CPU identity fixture only')
        self.model = self.directory / 'identity-only.gguf'
        self.model.write_bytes(b'Not a GGUF; CPU identity fixture only')
        self.request = self.directory / 'normal.json'
        self.request.write_bytes((EVIDENCE / 'requests/normal.json').read_bytes())

    def bound_records(self, variant='base', workload='normal', request=None, environment=None):
        records = copy.deepcopy(read_samples(variant, workload))
        request = request or self.request
        payload = request.read_bytes()
        metadata = records[0]
        metadata.update(model=str(self.model.resolve()), request_file=str(request.resolve()),
                        request_bytes=len(payload), request_fnv1a64=fnv1a64(payload))
        if environment is not None:
            for name, value in environment.items():
                metadata['environment'][name] = value
        return records

    def state_for(self, records, request=None):
        metadata = records[0]
        request = request or self.request
        payload = request.read_bytes()
        environment = {key: value for key, value in metadata['environment'].items()
                       if value is not None}
        return {'status': 'validated', 'returncode': 0,
                'argv': [str(self.binary.resolve()), str(self.model.resolve()),
                         str(request.resolve()), '3', str(metadata['max_new_tokens']),
                         str(metadata['max_seq_len'])],
                'cwd': str(self.binary.parent.parent), 'environment': environment,
                'binary_sha256': hashlib.sha256(self.binary.read_bytes()).hexdigest(),
                'metallib_sha256': hashlib.sha256((self.binary.parent / 'myds.metallib').read_bytes()).hexdigest(),
                'request_sha256': hashlib.sha256(payload).hexdigest(),
                'request_bytes': len(payload), 'request_fnv1a64': fnv1a64(payload),
                'model': {'path': str(self.model.resolve()), 'bytes': self.model.stat().st_size,
                          'mtime_ns': self.model.stat().st_mtime_ns,
                          'identity_note': 'CPU fixture; no model execution'},
                'host_before': copy.deepcopy(HOST), 'host_after': copy.deepcopy(HOST)}

    def comparison_fixture(self, workload='normal'):
        request = self.directory / f'compare-{workload}.json'
        request.write_bytes((EVIDENCE / f'requests/{workload}.json').read_bytes())
        groups = []
        for name, variant in (('base', 'base'), ('candidate', 'final')):
            directory = self.directory / f'{workload}-{name}'
            directory.mkdir()
            records = self.bound_records(variant, workload, request)
            write_samples(directory / 'run-metrics.jsonl', records)
            (directory / 'run-state.json').write_text(json.dumps(self.state_for(records, request)))
            groups.append(directory)
        return types.SimpleNamespace(base=groups[0], candidate=groups[1],
                                     out=self.directory / f'{workload}-comparison.json')

    def run_args(self):
        return types.SimpleNamespace(binary=self.binary, model=self.model, request=self.request,
                                     env=[], server_port=8181, out=self.directory / 'run',
                                     max_new_tokens=64, context=262144, timeout=3)

    def invoke_cpu_runner(self, args, records, host_side_effect=None, captured_stdout=None):
        """All process/host operations are mocked; only files and Python audit run."""
        stdout = captured_stdout if captured_stdout is not None else io.StringIO()
        constructor = mock.Mock(side_effect=lambda argv, **kwargs:
                                CompletedCPUHarness(argv, records=records, **kwargs))
        host_kwargs = ({'side_effect': host_side_effect} if host_side_effect is not None
                       else {'return_value': copy.deepcopy(HOST)})
        patches = (mock.patch.object(self.driver.platform, 'system', return_value='Darwin'),
                   mock.patch.object(self.driver, 'check_idle'),
                   mock.patch.object(self.driver, 'host_state', **host_kwargs),
                   mock.patch.object(self.driver.subprocess, 'Popen', constructor))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(stdout))
            result = self.driver.run(args)
        return result, constructor, stdout.getvalue()

    def test_historical_all_six_groups_pass_raw_audit(self):
        audit = self.driver.load_audit()
        for workload in ('short', 'normal', 'long'):
            for variant in ('base', 'final'):
                with self.subTest(workload=workload, variant=variant):
                    records = read_samples(variant, workload)
                    explicit_env = {key: value for key, value in records[0]['environment'].items()
                                    if value is not None}
                    group, samples = audit.validate_group(WINDOW, f'{variant}-{workload}',
                                                          {'environment': explicit_env})
                    self.assertEqual(len(samples), 4)
                    self.assertEqual(group['formal_runs'], 3)
                    self.assertTrue(group['warm_formal_raw_tokens_and_byte_hash_equal'])

    def test_compare_actual_historical_timings_and_outputs(self):
        expected = {'short': (9.3863755, 2.9336855),
                    'normal': (9.1380277577, 3.3009242441),
                    'long': (1.033723, 3.16549)}
        for workload in ('short', 'normal', 'long'):
            with self.subTest(workload=workload):
                args = self.comparison_fixture(workload)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.driver.compare(args), 0)
                summary = json.loads(args.out.read_text())
                self.assertTrue(summary['all_warm_and_formal_outputs_match'])
                records = read_samples('base', workload)
                self.assertEqual(summary['base']['token_ids_fnv1a64'], records[2]['token_ids_fnv1a64'])
                for phase, percent in zip(('prefill', 'decode'), expected[workload]):
                    self.assertAlmostEqual(summary['comparison'][phase]['time_saved_percent'], percent,
                                           delta=0.001)
                # The historical harness predates per-sample attention counters.
                self.assertEqual(summary['attention_paths'], {'base': None, 'candidate': None})

    def test_attention_paths_flags_head_first_fallback(self):
        clean = {'calls': 600, 'gather_calls': 570, 'slab_calls': 30, 'head_first_attempts': 570,
                 'head_first_encoded': 570, 'head_first_failures': 0, 'head_first_cached_fallbacks': 0,
                 'original_vec_encoded': 30, 'matrix_encoded': 0, 'matrix_fallbacks': 0}
        sticky = dict(clean, head_first_attempts=0, head_first_encoded=0,
                      head_first_cached_fallbacks=570, original_vec_encoded=600)
        samples = [{'run': 0, 'warmup': True, 'attention_paths': clean},
                   {'run': 1, 'warmup': False, 'attention_paths': clean}]
        result = self.driver.attention_paths(samples)
        self.assertFalse(result['any_fallback'])
        self.assertEqual([r['run'] for r in result['runs']], [0, 1])
        samples.append({'run': 2, 'warmup': False, 'attention_paths': sticky})
        result = self.driver.attention_paths(samples)
        self.assertTrue(result['any_fallback'])
        self.assertEqual(result['runs'][2]['head_first_cached_fallbacks'], 570)
        self.assertIsNone(self.driver.attention_paths([{'run': 0, 'warmup': True}]))

    def test_compare_rejects_non_greedy_metadata(self):
        args = self.comparison_fixture()
        for directory in (args.base, args.candidate):
            path = directory / 'run-metrics.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]['temperature'] = 0.7
            write_samples(path, records)
        with self.assertRaises(ValueError):
            self.driver.compare(args)
        self.assertFalse(args.out.exists())

    def test_compare_rejects_raw_token_hash_mismatch(self):
        args = self.comparison_fixture()
        path = args.candidate / 'run-metrics.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[2]['token_ids'][0] += 1
        write_samples(path, records)
        with self.assertRaisesRegex(ValueError, 'raw token/hash mismatch'):
            self.driver.compare(args)
        self.assertFalse(args.out.exists())

    def test_compare_rejects_incomplete_last_line(self):
        args = self.comparison_fixture()
        path = args.candidate / 'run-metrics.jsonl'
        path.write_text(path.read_text().rstrip('\n'))
        with self.assertRaisesRegex(ValueError, 'Incomplete final line'):
            self.driver.compare(args)
        self.assertFalse(args.out.exists())

    def test_compare_rejects_state_argv_metadata_mismatch(self):
        args = self.comparison_fixture()
        path = args.candidate / 'run-state.json'
        state = json.loads(path.read_text())
        state['argv'][5] = '123'
        path.write_text(json.dumps(state))
        with self.assertRaises(ValueError):
            self.driver.compare(args)
        self.assertFalse(args.out.exists())

    def test_runner_bound_cpu_fixture_and_separate_logs(self):
        args = self.run_args()
        records = self.bound_records(environment=self.driver.DEFAULT_ENV)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        with mock.patch.dict(self.driver.os.environ, {'MYDS_PROFILE_PREFILL': '1'}):
            result, constructor, stdout = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        self.assertEqual(constructor.call_count, 1)
        self.assertNotIn('MYDS_PROFILE_PREFILL', constructor.call_args.kwargs['env'])
        self.assertTrue(constructor.call_args.kwargs['start_new_session'])
        state = json.loads((args.out / 'run-state.json').read_text())
        self.assertEqual(state['status'], 'validated')
        self.assertEqual(state['returncode'], 0)
        self.assertEqual(state['host_after'], HOST)
        self.assertIn('"audit_pass": true', stdout)
        self.assertEqual(signal.getsignal(signal.SIGTERM), old_sigterm)
        self.assertIn('no inference was executed', (args.out / 'run.log').read_text())
        self.assertNotIn('CPU harness fixture', (args.out / 'run-metrics.jsonl').read_text())

    def test_runner_rejects_input_metadata_mismatches(self):
        mutations = {'model': lambda records: records[0].update(model='/wrong-model.gguf'),
                     'request_file': lambda records: records[0].update(request_file='/wrong-request.json'),
                     'request_bytes': lambda records: records[0].update(request_bytes=1),
                     'request_hash': lambda records: records[0].update(request_fnv1a64='0000000000000000'),
                     'context': lambda records: records[0].update(max_seq_len=123)}
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                args = self.run_args()
                args.out = self.directory / ('mismatch-' + name)
                records = self.bound_records(environment=self.driver.DEFAULT_ENV)
                mutate(records)
                with self.assertRaises(ValueError):
                    self.invoke_cpu_runner(args, records)
                state = json.loads((args.out / 'run-state.json').read_text())
                self.assertEqual(state['status'], 'failed')
                self.assertIn('error', state)

    def test_runner_host_after_failure_is_persisted_and_not_success(self):
        args = self.run_args()
        records = self.bound_records(environment=self.driver.DEFAULT_ENV)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(RuntimeError):
                self.invoke_cpu_runner(args, records,
                                       [copy.deepcopy(HOST), RuntimeError('CPU fixture host-after failure')],
                                       captured_stdout=stdout)
        state = json.loads((args.out / 'run-state.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['host_after_error'], 'CPU fixture host-after failure')
        self.assertEqual(signal.getsignal(signal.SIGTERM), old_sigterm)
        self.assertNotIn('"audit_pass": true', stdout.getvalue())

    def test_runner_timeout_stops_cpu_stub_and_preserves_failed_record(self):
        args = self.run_args()
        records = self.bound_records(environment=self.driver.DEFAULT_ENV)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        stdout = io.StringIO()

        class TimeoutCPUHarness(CompletedCPUHarness):
            def __init__(self, argv, **kwargs):
                super().__init__(argv, **kwargs)
                self.returncode = None
                self.wait_calls = 0

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired(self.argv, timeout)
                self.returncode = -signal.SIGINT
                return self.returncode

        constructor = mock.Mock(side_effect=lambda argv, **kwargs:
                                TimeoutCPUHarness(argv, records=records, **kwargs))
        with mock.patch.object(self.driver.platform, 'system', return_value='Darwin'), \
                mock.patch.object(self.driver, 'check_idle'), \
                mock.patch.object(self.driver, 'host_state', return_value=copy.deepcopy(HOST)), \
                mock.patch.object(self.driver.subprocess, 'Popen', constructor), \
                mock.patch.object(self.driver.os, 'killpg') as send_signal, \
                contextlib.redirect_stdout(stdout):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.driver.run(args)
        self.assertEqual(constructor.call_count, 1)
        send_signal.assert_called_once_with(CompletedCPUHarness.pid, signal.SIGINT)
        state = json.loads((args.out / 'run-state.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertIn('timed out', state['error'])
        self.assertEqual(state['host_after'], HOST)
        self.assertNotIn('"audit_pass": true', stdout.getvalue())
        self.assertEqual(signal.getsignal(signal.SIGTERM), old_sigterm)

    def test_runner_rejects_output_cap_argv_metadata_mismatch(self):
        args = self.run_args()
        args.max_new_tokens = 32
        records = self.bound_records(environment=self.driver.DEFAULT_ENV)
        # All historical sample counts remain valid under metadata's original 64
        # cap, so this failure must come from binding metadata to the requested 32.
        with self.assertRaisesRegex(ValueError, 'binding differs: max_new_tokens'):
            self.invoke_cpu_runner(args, records)
        self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

    def test_cli_rejects_nonfinite_timeout_and_out_of_range_context(self):
        invalid = [('--timeout', 'nan'), ('--timeout', 'inf'), ('--timeout', '-inf'),
                   ('--context', '0'), ('--context', '4294967296')]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                argv = ['bench_reproduce.py', 'run', '--model', str(self.model),
                        '--request', str(self.request), '--out', str(self.directory / 'invalid'),
                        key + '=' + value]
                with mock.patch.object(sys, 'argv', argv), mock.patch.object(self.driver, 'run') as runner:
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(self.driver.main(), 1)
                    runner.assert_not_called()

    def test_stop_child_reaps_process_even_if_signal_races_exit(self):
        child = mock.Mock(pid=987654321)
        child.poll.return_value = None
        child.wait.return_value = 0
        with mock.patch.object(self.driver.os, 'killpg', side_effect=ProcessLookupError):
            self.driver.stop_child(child)
        child.wait.assert_called()

    def test_stop_child_escalates_if_cpu_stub_ignores_interrupt(self):
        child = mock.Mock(pid=987654321)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('CPU stub', 5), -signal.SIGKILL]
        with mock.patch.object(self.driver.os, 'killpg') as send_signal:
            self.driver.stop_child(child)
        self.assertEqual(send_signal.call_args_list,
                         [mock.call(child.pid, signal.SIGINT), mock.call(child.pid, signal.SIGKILL)])
        self.assertEqual(child.wait.call_count, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
