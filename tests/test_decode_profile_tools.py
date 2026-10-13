#!/usr/bin/env python3
"""CPU-only diagnostic contracts: no Metal, model, service, trace or dispatch."""
import contextlib
import copy
import hashlib
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


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


profile = module('decode_profile_cpu', 'bench_decode_profile.py')
prepare = module('prepare_decode_profile_cpu', 'prepare_decode_profile.py')


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='myds-diag-summary-cpu-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw, self.native = self.root / 'diag.jsonl', self.root / 'native.jsonl'
        self.samples = [{'kind': 'sample', 'run': 1, 'warmup': False,
                         'prompt_tokens': 10, 'decode_intervals': 2}]
        self.rows = []
        cases = [(9, 'decode', 100., 101.), (10, 'decode', 2., 7.),
                 (10, 'decode', 3., 4.), (11, 'decode', 10., 11.),
                 (11, 'decode', 12., 13.), (10, 'prefill', 999., 1000.),
                 (11, 'prefill_scalar', 888., 889.), (12, 'decode', 777., 778.)]
        for i, (pos, phase, begin, end) in enumerate(cases):
            self.rows.append({'kind': 'decode_diag_cb', 'run': 1, 'cb': i,
                'position': pos, 'phase': phase, 'gpu_valid': True, 'status': 4,
                'gpu_start_s': begin, 'gpu_end_s': end, 'gpu_span_ms': 1000 * (end - begin),
                'encode_wall_ms': .01, 'commit_cpu_ms': .002, 'wait_cpu_ms': .02,
                'dispatches': 4, 'wait_calls': 0, 'first_layer': 0, 'last_layer': 1})
        self.rows.extend([{'kind': 'decode_diag_host', 'run': 1, 'position': pos,
            'phase': phase, 'stage': 'sampler', 'cpu_ms': value, 'calls': 1, 'bytes': 0}
            for pos, phase, value in [(9, 'decode', 999.), (10, 'decode', .3),
                                     (11, 'decode', .5), (10, 'prefill', 999.)]])
        self.rows.append({'kind': 'decode_diag_summary', 'run': 1,
            'created_cbs': 8, 'submitted_cbs': 8, 'overflow': 0, 'pending': 0, 'invalid_gpu': 0})

    def summarize(self):
        self.raw.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        self.native.write_text(''.join(json.dumps(r) + '\n' for r in self.samples))
        return profile.summarize(self.raw, self.native)

    def test_nested_overlap_and_gap_have_distinct_sum_union_envelope(self):
        report = self.summarize()['reports'][0]
        first, second = report['token_rows']
        self.assertEqual((first['position'], second['position']), (10, 11))
        self.assertEqual(first['gpu_cb_sum_ms'], 6000)
        self.assertEqual(first['gpu_union_ms'], 5000)
        self.assertEqual(first['gpu_envelope_ms'], 5000)
        self.assertEqual(second['gpu_union_ms'], 2000)
        self.assertEqual(second['gpu_envelope_ms'], 3000)
        self.assertEqual(second['gpu_gap_ms'], 1000)

    def test_prompt_final_prefill_scalar_and_out_of_window_are_excluded(self):
        report = self.summarize()['reports'][0]
        self.assertEqual(report['decode_intervals'], 2)
        self.assertEqual(report['mean_per_interval']['gpu_envelope_ms'], 4000)
        self.assertAlmostEqual(report['host_ms_per_interval']['sampler'], .4)

    def test_missing_cb_counter_or_entire_position_is_rejected(self):
        saved = copy.deepcopy(self.rows)
        self.rows.pop(3)
        with self.assertRaises(ValueError):
            self.summarize()
        self.rows = [r for r in saved if not (r['kind'] == 'decode_diag_cb' and r['position'] == 11)]
        self.rows[-1]['submitted_cbs'] = 6
        with self.assertRaises(ValueError):
            self.summarize()

    def test_missing_summary_and_pending_gpu_rejected(self):
        for field in (None, 'pending', 'overflow', 'invalid_gpu'):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.rows)
                if field is None:
                    self.rows.pop()
                else:
                    self.rows[-1][field] = 1
                with self.assertRaises(ValueError):
                    self.summarize()
                self.rows = saved

    def test_duplicate_cb_identity_rejected(self):
        self.rows[2]['cb'] = self.rows[1]['cb']
        with self.assertRaises(ValueError):
            self.summarize()

    def test_malformed_gpu_rows_rejected_even_if_summary_claims_valid(self):
        base = copy.deepcopy(self.rows)
        for changed in ({'gpu_valid': False}, {'status': 5}, {'gpu_start_s': 0.},
                        {'gpu_start_s': -2.}, {'gpu_end_s': 1.},
                        {'gpu_start_s': float('nan')}, {'gpu_end_s': float('inf')},
                        {'gpu_span_ms': -1.}, {'encode_wall_ms': -1.}):
            with self.subTest(changed=changed):
                self.rows = copy.deepcopy(base)
                self.rows[1].update(changed)
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_duplicate_native_run_does_not_silently_replace_sample(self):
        self.samples.append(dict(self.samples[0]))
        with self.assertRaises(ValueError):
            self.summarize()

    def test_unknown_run_and_inconsistent_span_fail_closed(self):
        original = copy.deepcopy(self.rows)
        ghost = dict(self.rows[0], run=99)
        self.rows.append(ghost)
        with self.assertRaises(ValueError):
            self.summarize()
        self.rows = original
        self.rows[1]['gpu_span_ms'] = 123.
        with self.assertRaises(ValueError):
            self.summarize()


class PrepareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='myds-diag-prepare-cpu-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.tree = Path(temporary.name) / 'tree'
        for name in prepare.FILES:
            path = self.tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())

    def test_late_anchor_failure_leaves_all_original_bytes_untouched(self):
        path = self.tree / 'tools/bench_phase.c'
        path.write_text(path.read_text().replace('        const double wall_s = now_s() - start;',
                                                '        const double wall_s = 0;'))
        before = {name: (self.tree / name).read_bytes() for name in prepare.FILES}
        with self.assertRaises(ValueError):
            prepare.prepare(self.tree)
        self.assertEqual(before, {name: (self.tree / name).read_bytes() for name in prepare.FILES})
        self.assertFalse((self.tree / 'decode-diag-manifest.json').exists())
        self.assertFalse((self.tree / 'src/myds_decode_profile.inc').exists())

    def test_success_preserves_operation_counts_and_binds_full_patch(self):
        original = (self.tree / 'src/myds_metal.m').read_text()
        counts = {key: original.count(anchor) for key, anchor in (
            ('created', '[g_queue commandBuffer]'), ('commit', '[cb commit];'),
            ('wait', '[cb waitUntilCompleted];'))}
        unretained = original.count('[g_queue commandBufferWithUnretainedReferences]')
        counts['created'] += unretained
        manifest = prepare.prepare(self.tree)
        self.assertEqual(manifest['operation_sites'], counts)
        self.assertEqual(manifest['unretained_creation_sites'], unretained)
        metal = (self.tree / 'src/myds_metal.m').read_text()
        self.assertEqual(metal.count('rp_created([g_queue commandBufferWithUnretainedReferences])'), unretained)
        forward = (self.tree / 'src/myds_v41.c').read_text()
        begin = forward.index('static int32_t v41_forward_impl(')
        end = forward.index('\nint32_t myds_v41_forward(', begin)
        self.assertIn('myds_remaining_context(need_logits', forward[begin:end])
        self.assertNotIn('myds_remaining_context(', forward[end:])
        for key in ('added_gpu_dispatches', 'added_gpu_barriers', 'added_encoder_boundaries',
                    'added_gpu_commits', 'added_gpu_waits'):
            self.assertEqual(manifest[key], 0)
        for name, digest in manifest['output_sha256'].items():
            self.assertEqual(hashlib.sha256((self.tree / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((self.tree / 'decode-diag.patch').read_bytes()).hexdigest(),
                         manifest['patch_sha256'])
        after = {name: (self.tree / name).read_bytes() for name in prepare.FILES}
        with self.assertRaises(ValueError):
            prepare.prepare(self.tree)
        self.assertEqual(after, {name: (self.tree / name).read_bytes() for name in prepare.FILES})

    def test_whitespace_only_declaration_drift_is_accepted(self):
        path = self.tree / 'src/myds_metal.m'
        original = path.read_text()
        import re
        changed, count = re.subn(r'^static[ \t]+int[ \t]+g_auto_barrier[ \t]*=[ \t]*1[ \t]*;[ \t]*$',
                                'static\tint g_auto_barrier=1;', original, flags=re.MULTILINE)
        self.assertEqual(count, 1)
        path.write_text(changed)
        prepare.prepare(self.tree)
        self.assertIn('static\tint g_auto_barrier=1;\n#include "myds_decode_profile.inc"', path.read_text())

    def test_changed_or_duplicate_declaration_rejects_before_mutation(self):
        path = self.tree / 'src/myds_metal.m'
        original = path.read_text()
        import re
        for fault in ('changed-initializer', 'duplicate'):
            with self.subTest(fault=fault):
                if fault == 'changed-initializer':
                    changed = re.sub(r'(g_auto_barrier[ \t]*=[ \t]*)1;', r'\g<1>0;', original, count=1)
                else:
                    changed = original + '\nstatic int g_auto_barrier = 1;\n'
                path.write_text(changed)
                before = {name: (self.tree / name).read_bytes() for name in prepare.FILES}
                with self.assertRaisesRegex(ValueError, 'g_auto_barrier declaration'):
                    prepare.prepare(self.tree)
                self.assertEqual(before, {name: (self.tree / name).read_bytes() for name in prepare.FILES})
                self.assertFalse((self.tree / 'decode-diag-manifest.json').exists())


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='myds-diag-owner-cpu-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tree = self.root / 'diagnostic'
        (self.tree / 'bin').mkdir(parents=True)
        for name in ('bench_phase', 'myds.metallib'):
            (self.tree / 'bin' / name).write_text('CPU identity fixture only\n')
        (self.tree / 'decode-diag-manifest.json').write_text('{"output_sha256":{}}\n')
        self.model, self.request = self.root / 'model.gguf', self.root / 'request.json'
        self.model.write_text('Not a GGUF\n'); self.request.write_text('{}\n')
        self.args = types.SimpleNamespace(model=self.model, request=self.request, out=self.root / 'output',
            mode='trace', repeats=3, max_new_tokens=256, context=262144, timeout=.01, env=[])
        self.trace_started = threading.Event()
        self.children = []
        self.live_groups = set()

    def popen(self, argv, **kwargs):
        child = mock.Mock(pid=900010 + len(self.children), stderr=io.BytesIO())
        child.poll.return_value = None
        if not self.children:
            child.stderr = io.BytesIO(b'DECODE_DIAG_TRACE_READY run=1 position=42 monotonic=100\n')
            def wait(timeout=None):
                if not getattr(child, '_waited', False):
                    child._waited = True
                    if self.args.mode == 'trace':
                        self.trace_started.wait(1)
                    raise subprocess.TimeoutExpired(argv, timeout)
                return -15
            child.wait.side_effect = wait
        else:
            child.wait.return_value = -15
            self.trace_started.set()
        # Avoid Mock's generated truthy attributes in wait's first-call check.
        child._waited = False
        self.children.append(child)
        self.live_groups.add(child.pid)
        return child

    def kill_group(self, pid, sig):
        if sig == 0:
            if pid not in self.live_groups:
                raise ProcessLookupError(pid)
        else:
            self.live_groups.discard(pid)

    def patches(self):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(profile, 'ROOT', self.tree))
        stack.enter_context(mock.patch.object(profile.bench, 'check_idle'))
        stack.enter_context(mock.patch.object(profile.bench, 'source_state', return_value={'scope': 'CPU fixture'}))
        stack.enter_context(mock.patch.object(profile.bench, 'host_state', return_value={'scope': 'CPU fixture'}))
        self.launch = stack.enter_context(mock.patch.object(profile.subprocess, 'Popen', side_effect=self.popen))
        self.kill = stack.enter_context(mock.patch.object(profile.os, 'killpg', side_effect=self.kill_group))
        return stack

    def test_timeout_finally_cleans_native_and_independent_trace_groups(self):
        with self.patches(), self.assertRaises(subprocess.TimeoutExpired):
            profile.run(self.args)
        self.assertEqual(len(self.children), 2)
        self.assertEqual({call.args for call in self.kill.call_args_list if call.args[1] != 0},
                         {(900010, signal.SIGTERM), (900011, signal.SIGTERM)})
        state = json.loads((self.args.out / 'run-state.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['trace']['target_pid'], 900010)
        self.assertEqual(self.launch.call_args_list[1].kwargs['env']['DEVELOPER_DIR'],
                         '/Applications/Xcode.app/Contents/Developer')

    def test_sigterm_main_enters_finally_and_second_signal_is_deferred(self):
        self.args.mode = 'off'
        handlers = {}
        def signal_handler(sig, handler):
            handlers[sig] = handler
        def popen(argv, **kwargs):
            child = self.popen(argv, **kwargs)
            called = False
            def wait(timeout=None):
                nonlocal called
                if not called:
                    called = True
                    handlers[signal.SIGTERM](signal.SIGTERM, None)
                return -15
            child.wait.side_effect = wait
            return child
        argv = ['bench_decode_profile.py', 'run', '--model', str(self.model),
                '--request', str(self.request), '--out', str(self.args.out), '--mode', 'off']
        with self.patches(), mock.patch.object(profile.subprocess, 'Popen', side_effect=popen), \
                mock.patch.object(profile.signal, 'signal', side_effect=signal_handler), \
                mock.patch.object(sys, 'argv', argv), self.assertRaisesRegex(RuntimeError, 'signal 15'):
            def kill(pid, sig):
                if sig:
                    handlers[signal.SIGINT](signal.SIGINT, None)
                self.kill_group(pid, sig)
            self.kill.side_effect = kill
            profile.main()
        self.assertIn(mock.call(900010, signal.SIGTERM), self.kill.call_args_list)
        self.assertEqual(json.loads((self.args.out / 'run-state.json').read_text())['status'], 'failed')

    def test_first_owner_cleanup_failure_does_not_skip_second_owner(self):
        calls = []
        def stop(child):
            if child is None:
                return
            calls.append(child.pid)
            if len(calls) == 1:
                raise RuntimeError('injected native cleanup error')
        with self.patches(), mock.patch.object(profile, 'stop', side_effect=stop), self.assertRaises(Exception):
            profile.run(self.args)
        self.assertEqual(calls, [900010, 900011])
        state = json.loads((self.args.out / 'run-state.json').read_text())
        self.assertEqual(state['status'], 'failed')

    def test_exited_leader_still_drains_its_owned_process_group(self):
        child = mock.Mock(pid=900050)
        child.poll.return_value = 0
        self.live_groups.add(child.pid)
        with mock.patch.object(profile.os, 'killpg', side_effect=self.kill_group) as kill:
            profile.stop(child)
        self.assertIn(mock.call(child.pid, signal.SIGTERM), kill.call_args_list)
        child.wait.assert_not_called()
        self.assertNotIn(child.pid, self.live_groups)


if __name__ == '__main__':
    unittest.main(verbosity=2)
