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

    def v2_records(self, count=64, environment=None, support=None):
        """Synthetic telemetry around saved timings; no model/performance claim."""
        records = self.bound_records(environment=environment)
        records[0].update(protocol_version=2, max_new_tokens=count,
                          dspark_model=str(support.resolve()) if support else None)
        loaded = next(record for record in records if record['kind'] == 'load')
        loaded.update(dspark_loaded=support is not None, dspark_load_s=0.01 if support else 0)
        for sample in records:
            if sample['kind'] != 'sample':
                continue
            original = sample['callback_intervals_ms']
            sample['token_ids'] = [sample['token_ids'][index % len(sample['token_ids'])] for index in range(count)]
            sample['token_ids_fnv1a64'] = self.driver.load_audit().token_hash(sample['token_ids'])
            sample['callback_intervals_ms'] = [original[index % len(original)] for index in range(count - 1)]
            sample.update(output_tokens=count, decode_intervals=count - 1,
                          decode_s=sum(sample['callback_intervals_ms']) / 1000)
            sample['decode_ms_per_token'] = 1000 * sample['decode_s'] / (count - 1)
            sample['last_callback_s'] = sample['first_callback_s'] + sample['decode_s']
            sample['wall_s'] = sample['last_callback_s'] + .001
            sample['final_logits'] = {'finite': True, 'elements': loaded['vocab_size'],
                                      'fnv1a64_bytes': '1234567890abcdef'}
            telemetry = {key: 0 for key in self.driver.DECODE_COUNTERS}
            telemetry.update(target_forward_calls=count, target_rows=count, output_tokens=count,
                             cycle_count=count, recorded_cycles=count, timing_enabled=True,
                             cycle_ms=[1.] * count, cycle_output_tokens=[1] * count)
            telemetry.update({key: 0. for key in self.driver.DECODE_TIMERS})
            sample['decode_telemetry'] = telemetry
            sample['gpu_indexer_compact'] = {key: 0 for key in self.driver.INDEXER_COMPACT_COUNTERS}
            sample['gpu_topk_prefix'] = {key: 0 for key in self.driver.TOPK_PREFIX_COUNTERS}
        return records

    def invoke_cpu_runner(self, args, records, host_side_effect=None, captured_stdout=None):
        """All process/host operations are mocked; only files and Python audit run."""
        stdout = captured_stdout if captured_stdout is not None else io.StringIO()
        constructor = mock.Mock(side_effect=lambda argv, **kwargs:
                                CompletedCPUHarness(argv, records=records, **kwargs))
        host_kwargs = ({'side_effect': host_side_effect} if host_side_effect is not None
                       else {'return_value': copy.deepcopy(HOST)})
        patches = (mock.patch.object(self.driver.platform, 'system', return_value='Darwin'),
                   mock.patch.object(self.driver, 'check_idle'),
                   mock.patch.object(self.driver, 'source_state', return_value={'head': 'CPU fixture', 'dirty': False}),
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

    def test_v2_all_output_caps_keep_raw_tokens_and_periods(self):
        for count in (32, 64, 256, 512):
            with self.subTest(count=count):
                args = self.run_args()
                args.max_new_tokens = count
                args.out = self.directory / f'cap-{count}'
                records = self.v2_records(count, self.driver.DEFAULT_ENV)
                result, _, _ = self.invoke_cpu_runner(args, records)
                self.assertEqual(result, 0)
                summary = json.loads((args.out / 'run-summary.json').read_text())
                self.assertEqual(summary['output_tokens'], count)
                self.assertEqual(summary['callback_latency_ms']['count'], 3 * (count - 1))
                self.assertEqual(summary['decode_telemetry']['cycle_latency_ms']['count'], 3 * count)

    def test_v2_batched_callbacks_allow_zero_gaps_and_reject_negative_gaps(self):
        records = self.v2_records(environment=self.driver.DEFAULT_ENV)
        for sample in records:
            if sample['kind'] != 'sample':
                continue
            # First token, 31 two-token batches, then the final token. Each
            # batch has one real target interval and a same-tick emit gap.
            sample['callback_intervals_ms'] = [value for _ in range(31) for value in (5., 0.)] + [5.]
            sample['decode_s'] = sum(sample['callback_intervals_ms']) / 1000
            sample['decode_ms_per_token'] = 1000 * sample['decode_s'] / sample['decode_intervals']
            sample['last_callback_s'] = sample['first_callback_s'] + sample['decode_s']
            sample['wall_s'] = sample['last_callback_s'] + .001
            sample['decode_telemetry'].update(
                target_forward_calls=33, target_rows=64, batch2_calls=31, batch2_hits=31,
                pld_proposed=31, pld_accepted=31, cycle_count=33, recorded_cycles=33,
                cycle_ms=[5.] * 33, cycle_output_tokens=[1] + [2] * 31 + [1])
        args = self.run_args()
        result, _, _ = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        summary = json.loads((args.out / 'run-summary.json').read_text())
        self.assertEqual(summary['callback_latency_ms']['count'], 3 * 63)
        self.assertEqual(summary['callback_latency_ms']['min'], 0.)
        self.assertGreater(summary['stats']['decode_ms_per_token']['median'], 0.)
        self.assertGreater(summary['decode_telemetry']['cycle_latency_ms']['min'], 0.)

        negative = copy.deepcopy(records)
        negative[2]['callback_intervals_ms'][1] = -.001
        args.out = self.directory / 'negative-callback-gap'
        with self.assertRaisesRegex(ValueError, r'Invalid sign: run.run0.callback_interval'):
            self.invoke_cpu_runner(args, negative)
        self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

        zero_total = copy.deepcopy(records)
        zero_total[2].update(callback_intervals_ms=[0.] * 63, decode_s=0.,
                             decode_ms_per_token=0., last_callback_s=zero_total[2]['first_callback_s'])
        args.out = self.directory / 'zero-total-decode'
        with self.assertRaisesRegex(ValueError, r'Invalid sign: run.run0.decode_s'):
            self.invoke_cpu_runner(args, zero_total)

        # Preserve the original schema 1 audit rather than changing saved
        # single-token timing evidence to use the new batch emit semantics.
        legacy = copy.deepcopy(records)
        legacy[0]['protocol_version'] = 1
        args.out = self.directory / 'legacy-zero-callback-gap'
        with self.assertRaisesRegex(ValueError, r'Invalid sign: run.run0.callback_interval'):
            self.invoke_cpu_runner(args, legacy)

    def test_named_batch2_arm_binds_support_and_records_execution(self):
        args = self.run_args()
        args.arm = 'batch2'
        args.dspark_model = self.directory / 'support-identity-only.gguf'
        args.dspark_model.write_bytes(b'CPU fixture, never loaded')
        args.dspark_sha256 = 'a' * 64
        environment = dict(self.driver.DEFAULT_ENV, MYDS_DECODE_SPLIT='4', MYDS_DECODE_METRICS='1',
                           MYDS_SPEC_COST_SCHEDULER='0', **self.driver.ARM_ENV['batch2'])
        records = self.v2_records(environment=environment, support=args.dspark_model)
        for sample in records:
            if sample['kind'] == 'sample':
                sample['decode_telemetry'].update(batch2_calls=5, batch2_hits=2, batch2_misses=3)
        result, constructor, _ = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        self.assertEqual(constructor.call_args.args[0][-1], str(args.dspark_model.resolve()))
        state = json.loads((args.out / 'run-state.json').read_text())
        self.assertEqual(state['dspark_model']['artifact_sha256'], 'a' * 64)
        self.assertIn('did not rehash', state['dspark_model']['identity_note'])
        summary = json.loads((args.out / 'run-summary.json').read_text())
        self.assertTrue(summary['decode_telemetry']['batch2_executed_in_formal_runs'])

    def test_named_arm_rejects_missing_support_or_conflicting_flags(self):
        for arm, env in (('batch2', []), ('target', ['MYDS_PLD=1'])):
            with self.subTest(arm=arm):
                args = self.run_args()
                args.arm, args.env = arm, env
                with mock.patch.object(self.driver.platform, 'system', return_value='Darwin'), \
                        mock.patch.object(self.driver.subprocess, 'Popen') as execute:
                    with self.assertRaises(ValueError):
                        self.driver.run(args)
                    execute.assert_not_called()
                self.assertFalse(args.out.exists())

    def test_named_pld_arm_really_enables_serial_probe(self):
        args = self.run_args()
        args.arm = 'pld'
        environment = dict(self.driver.DEFAULT_ENV, MYDS_DECODE_SPLIT='4', MYDS_DECODE_METRICS='1',
                           MYDS_SPEC_COST_SCHEDULER='0', **self.driver.ARM_ENV['pld'])
        records = self.v2_records(environment=environment)
        result, constructor, _ = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        self.assertEqual(constructor.call_args.kwargs['env']['MYDS_SPEC_SERIAL_PROBE'], '1')

    def test_named_pld_batch2_arm_needs_no_support_and_records_batch_execution(self):
        args = self.run_args()
        args.arm = 'pld-batch2'
        environment = dict(self.driver.DEFAULT_ENV, MYDS_DECODE_SPLIT='4', MYDS_DECODE_METRICS='1',
                           MYDS_SPEC_COST_SCHEDULER='0', **self.driver.ARM_ENV['pld-batch2'])
        records = self.v2_records(environment=environment)
        for sample in records:
            if sample['kind'] == 'sample':
                sample['decode_telemetry'].update(pld_queries=10, pld_cycles=4, pld_proposed=4,
                                                  pld_accepted=2, batch2_calls=4, batch2_hits=2, batch2_misses=2)
        result, constructor, _ = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        self.assertEqual(len(constructor.call_args.args[0]), 6)
        self.assertEqual(constructor.call_args.kwargs['env']['MYDS_BATCH2_SPEC'], '1')
        self.assertEqual(constructor.call_args.kwargs['env']['MYDS_SPEC_SERIAL_PROBE'], '0')
        summary = json.loads((args.out / 'run-summary.json').read_text())
        self.assertTrue(summary['decode_telemetry']['batch2_executed_in_formal_runs'])
        state = json.loads((args.out / 'run-state.json').read_text())
        self.assertIsNone(state['dspark_model'])

    def test_v2_rejects_missing_logits_telemetry_or_support_fallback(self):
        mutations = {
            'nonfinite': lambda sample: sample['final_logits'].update(finite=False),
            'changed_logits': lambda sample: sample['final_logits'].update(fnv1a64_bytes='0' * 16),
            'missing_telemetry': lambda sample: sample.pop('decode_telemetry'),
            'bad_count': lambda sample: sample['decode_telemetry'].update(output_tokens=999),
            'bad_cycle': lambda sample: sample['decode_telemetry']['cycle_ms'].__setitem__(0, float('nan')),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                args = self.run_args()
                args.out = self.directory / name
                records = self.v2_records(environment=self.driver.DEFAULT_ENV)
                mutate(records[3])
                with self.assertRaises(ValueError):
                    self.invoke_cpu_runner(args, records)
                self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

        args = self.run_args()
        args.dspark_model = self.directory / 'support.gguf'
        args.dspark_model.write_bytes(b'CPU support identity only')
        args.out = self.directory / 'fallback'
        records = self.v2_records(environment=self.driver.DEFAULT_ENV, support=args.dspark_model)
        records[1]['dspark_loaded'] = False
        with self.assertRaisesRegex(ValueError, 'was not loaded'):
            self.invoke_cpu_runner(args, records)

    def test_v2_disabled_timing_is_null_without_false_zero_measurement(self):
        args = self.run_args()
        records = self.v2_records(environment=self.driver.DEFAULT_ENV)
        for sample in records:
            if sample['kind'] == 'sample':
                sample['decode_telemetry'].update(timing_enabled=False, recorded_cycles=0,
                                                  cycle_ms=[], cycle_output_tokens=[])
                sample['decode_telemetry'].update({key: None for key in self.driver.DECODE_TIMERS})
        result, _, _ = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        summary = json.loads((args.out / 'run-summary.json').read_text())
        self.assertIsNone(summary['decode_telemetry']['cycle_latency_ms'])

    def test_cost_scheduler_without_cycle_recording_reports_null_not_truncation(self):
        args = self.run_args()
        args.env = ['MYDS_DECODE_METRICS=0', 'MYDS_SPEC_COST_SCHEDULER=1']
        environment = dict(self.driver.DEFAULT_ENV, MYDS_DECODE_METRICS='0', MYDS_SPEC_COST_SCHEDULER='1')
        records = self.v2_records(environment=environment)
        for sample in records:
            if sample['kind'] == 'sample':
                sample['decode_telemetry'].update(recorded_cycles=0, cycle_ms=[], cycle_output_tokens=[])
        result, _, _ = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        summary = json.loads((args.out / 'run-summary.json').read_text())['decode_telemetry']
        self.assertIsNone(summary['cycle_latency_ms'])
        self.assertIsNone(summary['mean_tokens_per_recorded_cycle'])
        self.assertFalse(summary['cycle_recording_requested'])
        self.assertFalse(summary['cycles_truncated'])

    def test_requested_cycle_recording_with_zero_records_marks_missing_data(self):
        args = self.run_args()
        args.env = ['MYDS_DECODE_METRICS=1']
        environment = dict(self.driver.DEFAULT_ENV, MYDS_DECODE_METRICS='1')
        records = self.v2_records(environment=environment)
        for sample in records:
            if sample['kind'] == 'sample':
                sample['decode_telemetry'].update(recorded_cycles=0, cycle_ms=[], cycle_output_tokens=[])
        result, _, _ = self.invoke_cpu_runner(args, records)
        self.assertEqual(result, 0)
        summary = json.loads((args.out / 'run-summary.json').read_text())['decode_telemetry']
        self.assertIsNone(summary['cycle_latency_ms'])
        self.assertTrue(summary['cycle_recording_requested'])
        self.assertTrue(summary['cycles_truncated'])

    def test_compact_indexer_summary_requires_formal_encoding_not_warmup_or_env(self):
        # Counter records are synthetic CPU schema fixtures, never GPU execution.
        encoded = {'requests': 4, 'encoded': 4, 'unavailable': 0,
                   'dense_rows': 65536, 'scored_row_slots': 8192}
        unavailable = {'requests': 4, 'encoded': 0, 'unavailable': 4,
                       'dense_rows': 0, 'scored_row_slots': 0}
        zero = {key: 0 for key in self.driver.INDEXER_COMPACT_COUNTERS}
        for name, formal in (('encoded', encoded), ('warmup-only', zero), ('unavailable', unavailable)):
            with self.subTest(name=name):
                args = self.run_args()
                args.out = self.directory / ('compact-' + name)
                args.env = ['MYDS_INDEXER_COMPACT_SCORER=1']
                environment = dict(self.driver.DEFAULT_ENV, MYDS_INDEXER_COMPACT_SCORER='1')
                records = self.v2_records(environment=environment)
                for sample in records:
                    if sample['kind'] == 'sample':
                        sample['gpu_indexer_compact'] = dict(encoded if sample['warmup'] else formal)
                result, _, _ = self.invoke_cpu_runner(args, records)
                self.assertEqual(result, 0)
                summary = json.loads((args.out / 'run-summary.json').read_text())['gpu_indexer_compact']
                self.assertEqual(summary['encoded_in_formal_runs'], name == 'encoded')
                self.assertEqual(summary['any_unavailable_in_formal_runs'], name == 'unavailable')
                self.assertEqual(summary['formal_totals'], {key: 3 * value for key, value in formal.items()})
                self.assertEqual(summary['runs'][0]['encoded'], encoded['encoded'])

    def test_compact_indexer_audit_rejects_invalid_raw_counters(self):
        for name, invalid in (('negative', {'requests': -1}), ('bool', {'encoded': True}),
                              ('results', {'requests': 1, 'encoded': 1, 'unavailable': 1})):
            with self.subTest(name=name):
                args = self.run_args()
                args.out = self.directory / ('bad-compact-' + name)
                records = self.v2_records(environment=self.driver.DEFAULT_ENV)
                records[3]['gpu_indexer_compact'].update(invalid)
                with self.assertRaisesRegex(ValueError, 'compact indexer|Compact indexer'):
                    self.invoke_cpu_runner(args, records)
                self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

    def test_required_new_candidate_paths_reject_missing_and_prefill_only_evidence(self):
        # Native-counter schema fixtures exercise the real runner's rejection
        # and retained failure logs; no model or Metal work is executed here.
        for name, (flag, failure, keys) in self.driver.CANDIDATE_PATHS.items():
            enabled_value = self.driver.CANDIDATE_ENABLED_VALUES.get(name, '1')
            encoded = dict.fromkeys(keys, 0)
            encoded.update(requests=4, encoded=4, decode_requests=2, decode_encoded=2)
            if 'classified_partitions' in encoded:
                encoded['classified_partitions'] = 64
            for mode in ('missing', 'prefill-only', 'one-formal-only', 'failure', 'encoded'):
                with self.subTest(path=name, mode=mode):
                    args = self.run_args()
                    args.out = self.directory / (name + '-' + mode)
                    args.env = [flag + '=' + enabled_value]
                    options = {'gpu_topk_nan_fuse': 'require_topk_nan_fuse',
                               'gpu_batch2_q8_shared': 'require_batch2_q8_shared',
                               'gpu_lm_head_q8': 'require_lm_head_q8'}
                    setattr(args, options[name], True)
                    env = dict(self.driver.DEFAULT_ENV, **{flag: enabled_value})
                    records = self.v2_records(environment=env)
                    if name == 'gpu_batch2_q8_shared':
                        for sample in records:
                            if sample['kind'] == 'sample':
                                sample['decode_telemetry'].update(batch2_calls=5, batch2_hits=2, batch2_misses=3)
                    for sample in records:
                        if sample['kind'] != 'sample' or mode == 'missing':
                            continue
                        counters = dict(encoded)
                        if mode == 'prefill-only':
                            counters.update(decode_requests=0, decode_encoded=0)
                        elif mode == 'one-formal-only' and not sample['warmup'] and sample['run'] != 1:
                            counters = dict.fromkeys(keys, 0)
                        elif mode == 'failure':
                            counters.update(encoded=0, decode_encoded=0)
                            counters[failure], counters['decode_' + failure] = 4, 2
                        sample[name] = counters
                    if mode == 'encoded':
                        result, _, _ = self.invoke_cpu_runner(args, records)
                        self.assertEqual(result, 0)
                        summary = json.loads((args.out / 'run-summary.json').read_text())[name]
                        self.assertTrue(summary['encoded_in_formal_decode'])
                    else:
                        with self.assertRaisesRegex(ValueError, 'candidate path'):
                            self.invoke_cpu_runner(args, records)
                        state = json.loads((args.out / 'run-state.json').read_text())
                        self.assertEqual(state['status'], 'failed')
                        self.assertTrue((args.out / 'run-metrics.jsonl').is_file())

    def test_shared_helper_success_cannot_mask_failed_batch2_graph(self):
        for field, value in (('batch2_calls', 0), ('batch2_hits', 0),
                             ('batch2_failures', 1), ('batch2_fallbacks', 1)):
            args = self.run_args(); args.out = self.directory / ('graph-' + field)
            args.env = ['MYDS_BATCH2_Q8_SHARED=1']; args.require_batch2_q8_shared = True
            records = self.v2_records(environment=dict(self.driver.DEFAULT_ENV, MYDS_BATCH2_Q8_SHARED='1'))
            for sample in records:
                if sample['kind'] == 'sample':
                    sample['decode_telemetry'].update(batch2_calls=5, batch2_hits=2, batch2_misses=3)
                    sample['gpu_batch2_q8_shared'] = dict(requests=4, encoded=4, failures=0,
                                                        decode_requests=2, decode_encoded=2, decode_failures=0)
            bad = next(s for s in records if s['kind'] == 'sample' and s['run'] == 2)
            bad['decode_telemetry'][field] = value
            if field == 'batch2_calls':
                bad['decode_telemetry'].update(batch2_hits=0, batch2_misses=0)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Required Batch2 graph'):
                self.invoke_cpu_runner(args, records)
            self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

    def test_new_candidate_counters_reject_disabled_and_invalid_records(self):
        for name, (flag, _, keys) in self.driver.CANDIDATE_PATHS.items():
            for mode in ('negative', 'bool', 'decode-exceeds-total', 'disabled'):
                with self.subTest(path=name, mode=mode):
                    args = self.run_args()
                    args.out = self.directory / (name + '-bad-' + mode)
                    value = ('2' if name == 'gpu_lm_head_q8' else '0') if mode == 'disabled' \
                        else self.driver.CANDIDATE_ENABLED_VALUES.get(name, '1')
                    args.env = [flag + '=' + value]
                    records = self.v2_records(environment=dict(self.driver.DEFAULT_ENV, **{flag: value}))
                    for sample in records:
                        if sample['kind'] == 'sample':
                            sample[name] = dict.fromkeys(keys, 0)
                    counters = records[3][name]
                    if mode == 'negative': counters['requests'] = -1
                    elif mode == 'bool': counters['encoded'] = True
                    elif mode == 'decode-exceeds-total': counters.update(decode_requests=1, decode_encoded=1)
                    else: counters.update(requests=1, encoded=1)
                    with self.assertRaisesRegex(ValueError, '[Cc]andidate'):
                        self.invoke_cpu_runner(args, records)
                    self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

    def test_lm_head_row_cohort_rejects_invalid_values_even_without_counters(self):
        for value in ('0', '1', '3', '04', '4 ', '-2', 'garbage'):
            with self.subTest(value=value):
                records = self.v2_records(environment=dict(self.driver.DEFAULT_ENV, MYDS_LM_HEAD_Q8_NR0=value))
                directory = self.directory / ('invalid-lm-head-' + value.strip())
                directory.mkdir()
                write_samples(directory / 'run-metrics.jsonl', records)
                state = self.state_for(records)
                with self.assertRaisesRegex(ValueError, 'Invalid LM head row cohort'):
                    self.driver.validate_run(directory, state, self.driver.load_audit())

    def test_topk_prefix_summary_requires_formal_encoding(self):
        encoded = dict(requests=4, encoded=4, unavailable=0, classified_rows=4,
                       merge_dispatches=16, dense_write_slots=100000, prefix_write_slots=20000,
                       decode_requests=2, decode_encoded=2, decode_unavailable=0)
        unavailable = dict(requests=4, encoded=0, unavailable=4, classified_rows=0,
                           merge_dispatches=0, dense_write_slots=0, prefix_write_slots=0,
                           decode_requests=2, decode_encoded=0, decode_unavailable=2)
        zero = {key: 0 for key in self.driver.TOPK_PREFIX_COUNTERS}
        for name, formal in (('encoded', encoded), ('warmup-only', zero), ('unavailable', unavailable)):
            with self.subTest(name=name):
                args = self.run_args()
                args.out = self.directory / ('topk-' + name)
                args.env = ['MYDS_INDEXER_TOPK_PREFIX_MERGE=1']
                environment = dict(self.driver.DEFAULT_ENV, MYDS_INDEXER_TOPK_PREFIX_MERGE='1')
                records = self.v2_records(environment=environment)
                for sample in records:
                    if sample['kind'] == 'sample':
                        sample['gpu_topk_prefix'] = dict(encoded if sample['warmup'] else formal)
                result, _, _ = self.invoke_cpu_runner(args, records)
                self.assertEqual(result, 0)
                summary = json.loads((args.out / 'run-summary.json').read_text())['gpu_topk_prefix']
                self.assertEqual(summary['encoded_in_formal_runs'], name == 'encoded')
                self.assertEqual(summary['any_unavailable_in_formal_runs'], name == 'unavailable')
                self.assertEqual(summary['formal_totals'], {key: 3 * value for key, value in formal.items()})
                self.assertIn('not actual GPU traffic', summary['scope'])

    def test_required_topk_prefix_rejects_missing_fallback_and_prefill_only_paths(self):
        encoded = {key: 0 for key in self.driver.TOPK_PREFIX_COUNTERS}
        encoded.update(requests=4, encoded=4, classified_rows=4, merge_dispatches=16,
                       dense_write_slots=100000, prefix_write_slots=20000,
                       decode_requests=2, decode_encoded=2)
        unavailable = dict(encoded, encoded=0, unavailable=4, decode_encoded=0, decode_unavailable=2)
        prefill_only = dict(encoded, decode_requests=0, decode_encoded=0)
        for name in ('unavailable', 'one-formal-only', 'prefill-only', 'missing', 'encoded'):
            with self.subTest(name=name):
                args = self.run_args()
                args.out = self.directory / ('required-topk-' + name)
                args.require_topk_prefix = True
                records = self.v2_records(environment=self.driver.DEFAULT_ENV)
                for sample in records:
                    if sample['kind'] != 'sample':
                        continue
                    counters = unavailable if name == 'unavailable' else prefill_only if name == 'prefill-only' else encoded
                    if name == 'one-formal-only' and sample['run'] >= 2:
                        counters = unavailable
                    if name == 'missing':
                        del sample['gpu_topk_prefix']
                    else:
                        sample['gpu_topk_prefix'] = dict(counters)
                if name == 'encoded':
                    result, _, _ = self.invoke_cpu_runner(args, records)
                    self.assertEqual(result, 0)
                else:
                    with self.assertRaisesRegex(ValueError, 'Required topk prefix path'):
                        self.invoke_cpu_runner(args, records)
                    self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

    def test_same_build_comparison_rejects_changed_candidate_artifacts_and_source(self):
        args = self.comparison_fixture()
        args.same_build = True
        for directory in (args.base, args.candidate):
            path = directory / 'run-state.json'
            state = json.loads(path.read_text())
            state['source'] = {'head': 'same CPU fixture', 'dirty': False}
            path.write_text(json.dumps(state))
        path = args.candidate / 'run-state.json'
        original = json.loads(path.read_text())
        for key in ('binary_sha256', 'metallib_sha256', 'source'):
            changed = dict(original)
            changed[key] = {'head': 'different CPU fixture'} if key == 'source' else '0' * 64
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Same-build comparison differs: ' + key):
                self.driver.compare(args)
            self.assertFalse(args.out.exists())
        path.write_text(json.dumps(original))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.driver.compare(args), 0)

    def test_topk_prefix_audit_rejects_invalid_counters(self):
        for name, invalid in (('negative', {'requests': -1}), ('bool', {'encoded': True}),
                              ('results', {'requests': 1, 'encoded': 0, 'unavailable': 0}),
                              ('writes', {'dense_write_slots': 3, 'prefix_write_slots': 4})):
            with self.subTest(name=name):
                args = self.run_args()
                args.out = self.directory / ('bad-topk-' + name)
                records = self.v2_records(environment=self.driver.DEFAULT_ENV)
                records[3]['gpu_topk_prefix'].update(invalid)
                with self.assertRaisesRegex(ValueError, 'topk prefix|Topk prefix'):
                    self.invoke_cpu_runner(args, records)
                self.assertEqual(json.loads((args.out / 'run-state.json').read_text())['status'], 'failed')

    def test_a_b_a_control_reports_drift_and_rejects_changed_control(self):
        args = self.comparison_fixture()
        args.control = self.directory / 'control'
        args.control.mkdir()
        records = [json.loads(line) for line in (args.base / 'run-metrics.jsonl').read_text().splitlines()]
        write_samples(args.control / 'run-metrics.jsonl', records)
        state = json.loads((args.base / 'run-state.json').read_text())
        (args.control / 'run-state.json').write_text(json.dumps(state))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.driver.compare(args), 0)
        summary = json.loads(args.out.read_text())
        self.assertEqual(summary['control_drift_percent'], {'prefill': 0., 'decode': 0.})
        args.out.unlink()
        state['environment']['MYDS_PLD'] = '1'
        (args.control / 'run-state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'control identity differs'):
            self.driver.compare(args)
        self.assertFalse(args.out.exists())

    def test_a_b_a_control_rejects_valid_but_changed_stop_or_prefill_counts(self):
        args = self.comparison_fixture()
        args.control = self.directory / 'control'
        args.control.mkdir()
        records = self.v2_records(environment=self.driver.DEFAULT_ENV)
        state = self.state_for(records)
        for directory in (args.base, args.candidate, args.control):
            write_samples(directory / 'run-metrics.jsonl', records)
            (directory / 'run-state.json').write_text(json.dumps(state))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.driver.compare(args), 0)
        args.out.unlink()
        for changed, expected_field in (('stop', 'stop_reason'), ('counts', 'prompt_tokens')):
            with self.subTest(changed=changed):
                control = copy.deepcopy(records)
                for sample in control:
                    if sample['kind'] != 'sample':
                        continue
                    if changed == 'stop':
                        # An EOS at the cap is a valid sample with identical IDs,
                        # byte digest and logits, but a different stop contract.
                        sample['stop_reason'] = 'eos'
                    else:
                        # Keep the individual sample valid and its output equal;
                        # only the prompt/prefill count signature differs from A1.
                        sample['prompt_tokens'] += 1
                        sample['prefill_tokens'] += 1
                        sample['prefill_ms_per_token'] = 1000 * sample['prefill_s'] / sample['prefill_tokens']
                write_samples(args.control / 'run-metrics.jsonl', control)
                self.driver.validate_run(args.control, state, self.driver.load_audit())
                with self.assertRaisesRegex(ValueError,
                                            'Final control output/count differs in run 0: ' + expected_field):
                    self.driver.compare(args)
                self.assertFalse(args.out.exists())

    def test_compare_rejects_mixed_metrics_even_with_matching_raw_outputs(self):
        args = self.comparison_fixture()
        for directory, enabled in ((args.base, '1'), (args.candidate, '0')):
            path = directory / 'run-metrics.jsonl'
            environment = dict(self.driver.DEFAULT_ENV, MYDS_DECODE_METRICS=enabled)
            records = self.v2_records(environment=environment)
            if enabled == '0':
                for sample in records:
                    if sample['kind'] == 'sample':
                        sample['decode_telemetry'].update(timing_enabled=False, recorded_cycles=0,
                                                          cycle_ms=[], cycle_output_tokens=[])
                        sample['decode_telemetry'].update({key: None for key in self.driver.DECODE_TIMERS})
            write_samples(path, records)
            (directory / 'run-state.json').write_text(json.dumps(self.state_for(records)))
        with self.assertRaisesRegex(ValueError, 'Measurement setting differs: MYDS_DECODE_METRICS'):
            self.driver.compare(args)
        self.assertFalse(args.out.exists())

        # Both disabled arms remain comparable and expose no fictitious cycle data.
        base_records = [json.loads(line) for line in (args.candidate / 'run-metrics.jsonl').read_text().splitlines()]
        write_samples(args.base / 'run-metrics.jsonl', base_records)
        (args.base / 'run-state.json').write_text(json.dumps(self.state_for(base_records)))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.driver.compare(args), 0)
        summary = json.loads(args.out.read_text())
        self.assertIsNone(summary['base']['decode_telemetry']['cycle_latency_ms'])
        self.assertIsNone(summary['candidate']['decode_telemetry']['cycle_latency_ms'])

    def test_runner_rejects_presence_based_diagnostics_before_process_creation(self):
        for name in ('MYDS_PROF', 'MYDS_DISPATCH_HIST', 'MYDS_DEBUG_DISPATCH'):
            for value in ('0', '1', ''):
                with self.subTest(name=name, value=value):
                    args = self.run_args()
                    args.env = [name + '=' + value]
                    with mock.patch.object(self.driver.platform, 'system', return_value='Darwin'), \
                            mock.patch.object(self.driver.subprocess, 'Popen') as execute:
                        with self.assertRaisesRegex(ValueError, 'incompatible with this run: ' + name):
                            self.driver.run(args)
                        execute.assert_not_called()
                    self.assertFalse(args.out.exists())

    def test_saved_raw_metadata_rejects_unbound_presence_based_diagnostics(self):
        # Audit the actual native metadata even if a run-state omitted the setting.
        args = self.comparison_fixture()
        path = args.candidate / 'run-metrics.jsonl'
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for name in ('MYDS_PROF', 'MYDS_DISPATCH_HIST', 'MYDS_DEBUG_DISPATCH'):
            with self.subTest(name=name):
                records = copy.deepcopy(original)
                records[0]['environment'][name] = '0'
                write_samples(path, records)
                with self.assertRaisesRegex(ValueError, 'incompatible with this run: ' + name):
                    self.driver.compare(args)
                self.assertFalse(args.out.exists())

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
                self.assertEqual(summary['gpu_indexer_compact'], {'base': None, 'candidate': None})

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

    def test_compare_binds_support_identity_and_named_arm(self):
        support = self.directory / 'support.gguf'
        support.write_bytes(b'CPU identity fixture; never loaded')
        args = self.comparison_fixture()
        for directory in (args.base, args.candidate):
            records = self.v2_records(support=support)
            state = self.state_for(records)
            state['argv'].append(str(support.resolve()))
            state['dspark_model'] = self.driver.file_identity(support.resolve(), 'a' * 64)
            state['arm'] = 'custom'
            write_samples(directory / 'run-metrics.jsonl', records)
            (directory / 'run-state.json').write_text(json.dumps(state))
        original = json.loads((args.candidate / 'run-state.json').read_text())
        for field, message in (('dspark_model', 'File identity differs: dspark_model'),
                               ('arm', 'Named measurement arm differs')):
            state = copy.deepcopy(original)
            if field == 'dspark_model':
                state[field]['artifact_sha256'] = 'b' * 64
            else:
                state[field] = 'target'
            (args.candidate / 'run-state.json').write_text(json.dumps(state))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.driver.compare(args)
            self.assertFalse(args.out.exists())
        (args.candidate / 'run-state.json').write_text(json.dumps(original))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.driver.compare(args), 0)

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
                mock.patch.object(self.driver, 'source_state', return_value={'head': 'CPU fixture', 'dirty': False}), \
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
