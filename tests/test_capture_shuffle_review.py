#!/usr/bin/env python3
"""CPU reader regressions with complete synthetic bytes; no inference evidence.

The identity files are not executable models or binaries. Tests never create a
Metal device, contact a server, or invoke a subprocess. In particular, a passing
fixture establishes reader rejection behavior, not GPU or production acceptance.
"""
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'docs/benchmarks/2026-10-10-decode-capture-shuffle/scripts'
sys.path.insert(0, str(SCRIPTS))
import review_state as STATE
import review_fullmodel as FULL


def write_json(path, value):
    path.write_text(json.dumps(value) + '\n')


def write_rows(path, values):
    path.write_text(''.join(json.dumps(v) + '\n' for v in values))


def comparisons(labels, slots, hidden, pre):
    sizes = {**{label: STATE.VOCAB for label in labels}, 'dspark_target_slot': 5120,
             'first_hidden': 20480, 'first_pre': 4}
    counts = {**{label: 4 for label in labels}, 'dspark_target_slot': slots,
              'first_hidden': hidden, 'first_pre': pre}
    return [dict(kind='comparison', label=k, elements=sizes[k], pass_=True,
                 **{'pass': True}, mismatches=0, max_abs=0)
            for k, count in counts.items() for _ in range(count)]


def base_rows(positions):
    return [dict(kind='metadata', positions=list(positions), temperature=0, seed=42),
            dict(kind='complete', native_v41=True, contexts=1, positions_completed=4,
                 exact_logits_and_state=True)]


def batch_rows(positions):
    result = comparisons(STATE.BATCH_LABELS, 156, 20, 20)
    for pos in positions:
        result.extend([
            dict(kind='batch2_result', pos=pos, ret=2, expected_ret=2,
                 out=[0, 1], expected_out=[0, 1], first_argmax=0, second_argmax=1,
                 first_logits_available=True, second_logits_available=True,
                 snapshot_valid=True, first_state_valid=True,
                 caller_auto_barrier_restored=True),
            dict(kind='position_complete', pos=pos, next0=0, next1=1,
                 batch2_calls=5, hits=1, misses=3, failures=1, fallbacks=1,
                 **{'pass': True})])
    return result


def export_base(directory, positions):
    directory.mkdir(parents=True)
    values = struct.pack('<f', .5) * STATE.VOCAB
    for pos in positions:
        for arm in ('reference', 'candidate'):
            for step in ('first', 'second'):
                (directory / f'{arm}_{step}_pos{pos}.f32').write_bytes(values)
        (directory / f'inputs_pos{pos}.txt').write_text('0\n' * (pos + 2))


def capture_fixture(window):
    d = window / 'capture-state'
    export_base(d, STATE.POSITIONS)
    rows = base_rows(STATE.POSITIONS) + comparisons(STATE.CAPTURE_LABELS, 60, 16, 16)
    rows.append(dict(kind='capture_metadata', configured_candidate_mode=1,
                     support_model_loaded=False, real_drafter=False))
    for pos in STATE.POSITIONS:
        rows.append(dict(kind='capture_protocol', pos=pos, reference_mode=0, candidate_mode=1,
                         bypass_requests=6, bypass_encoded=0, slots_untouched=True,
                         reenabled_slots_byte_exact=True, partial_first_ready=True,
                         whole_rollback_stale_rejected=True, prewrite_error_recovered=True,
                         requests=21, encoded=15, bypassed=6, failures=0,
                         stale_rejections=2, **{'pass': True}))
        for arm in ('reference', 'enabled', 'disabled'):
            (d / f'capture_{arm}_slots_pos{pos}.f32').write_bytes(
                struct.pack('<f', 123.125 if arm == 'disabled' else .5) * 15360)
    rows.append(dict(kind='capture_complete', raw_f32_files=108, replay_files=8,
                     positions_completed=4, graph_positions_completed=4, candidate_mode=1,
                     real_drafter=False, exact_target_state=True))
    write_rows(window / 'capture-state.stdout.log', rows)
    graph = d / 'capture-graphs'
    export_base(graph, STATE.POSITIONS)
    gr = batch_rows(STATE.POSITIONS)
    for label, n in (('capture_tree_same_graph_state', STATE.VOCAB),
                     ('capture_tree_same_graph_hidden', 20480),
                     ('capture_tree_same_graph_pre', 4),
                     ('capture_tree_failure_scalar_retry', STATE.VOCAB)):
        gr.extend(dict(kind='comparison', label=label, elements=n, mismatches=0,
                       max_abs=0, **{'pass': True})
                  for _ in range(4 if label.endswith('retry') else 12))
    gr.extend(comparisons((), 48, 0, 0))
    gr.extend(dict(kind='comparison', label='capture_tree_same_graph_logits',
                   elements=STATE.VOCAB, mismatches=0, max_abs=0, **{'pass': True})
              for _ in range(20))
    for label, n in (('capture_tree_failure_scalar_retry_hidden', 20480),
                     ('capture_tree_failure_scalar_retry_pre', 4)):
        gr.extend(dict(kind='comparison', label=label, elements=n, mismatches=0,
                       max_abs=0, **{'pass': True}) for _ in range(4))
    gr.append(dict(kind='capture_tree_metadata', capture_state_layout=[15360, 20480, 4],
                   reference_mode=0, candidate_mode=1, scalar_tree_equivalence_claimed=False,
                   raw_f32_files=64, state_f32_files=24, computed_logits_f32_files=40,
                   ctx_logits_used=False))
    for pos in STATE.POSITIONS:
        for branch in ('A', 'B', 'miss'):
            heads = [0, 1] if branch == 'A' else [0, 2] if branch == 'B' else [0]
            for arm in ('reference', 'candidate'):
                (graph / f'tree_{branch}_{arm}_capture_state_pos{pos}.f32').write_bytes(
                    struct.pack('<f', .5) * 35844)
                for head in heads:
                    values = bytearray(struct.pack('<f', .5) * STATE.VOCAB)
                    best = 0 if head == 0 else 1
                    values[best*4:best*4+4] = struct.pack('<f', .75)
                    (graph / f'tree_{branch}_{arm}_logits{head}_pos{pos}.f32').write_bytes(values)
            gr.append(dict(kind='capture_tree_branch', pos=pos, branch=branch,
                           same_graph_capture_state_byte_exact=True,
                           scalar_tree_equivalence_claimed=False,
                           reference_ret=1 if branch == 'miss' else 2,
                           candidate_ret=1 if branch == 'miss' else 2,
                           tree_next=0, scalar_next=0, selected_head_index=heads[-1],
                           computed_head_indices=heads, head_argmax=[0, 1] if len(heads)==2 else [0],
                           reference_out=[0, 1] if len(heads)==2 else [0, -1],
                           candidate_out=[0, 1] if len(heads)==2 else [0, -1],
                           full_computed_logits_byte_exact=True, current_state_capture=True,
                           ctx_logits_used=False, ready=True))
        gr.append(dict(kind='capture_tree_complete', pos=pos, branches_completed=3,
                       encoded_graph_failure_invalidated=True, scalar_retry_byte_exact=True))
    gr.append(dict(kind='capture_graph_complete', positions_completed=4, candidate_mode=1,
                   requests=120, encoded=120, bypassed=0, failures=0, ready=True,
                   exact_target_state=True))
    write_rows(d / 'capture-graphs.stdout.log', gr)


def shuffle_fixture(window):
    for arm, mode in (('state-a1', 0), ('state-b', 1)):
        export_base(window / arm, STATE.LONG_POSITIONS)
        rows = base_rows(STATE.LONG_POSITIONS) + batch_rows(STATE.LONG_POSITIONS)
        rows[0].update(configured_shuffle_mode=mode, max_seq_len=16386,
                       long_indexer_state=True, batch2_opt_in=True)
        rows.append(dict(kind='long_indexer_complete', enabled=bool(mode), actual_mode=mode,
                         positions_completed=4, failures=0, exact_target_state=True,
                         prefix_method='public_chunk_prefill', requests=8 if mode else 0,
                         encoded=8 if mode else 0))
        write_rows(window / (arm + '.stdout.log'), rows)


class StateReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='myds-state-reader-cpu-')
        cls.base = Path(cls.temp.name).resolve()
        capture_fixture(cls.base)
        shuffle_fixture(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def edit_rows(self, relative, mutate):
        path = self.base / relative
        original = path.read_bytes()
        values = [json.loads(x) for x in path.read_text().splitlines()]
        mutate(values)
        write_rows(path, values)
        self.addCleanup(path.write_bytes, original)

    def edit_bytes(self, relative, data):
        path = self.base / relative
        original = path.read_bytes()
        path.write_bytes(data(original))
        self.addCleanup(path.write_bytes, original)

    def test_complete_synthetic_capture_and_long_state_sets(self):
        capture = STATE.audit_capture(self.base, {})
        self.assertEqual((capture['full_f32_files'], capture['replay_files']), (108, 8))
        self.assertFalse(capture['real_support_engine_acceptance'])
        self.assertFalse(capture['tree_scalar_equivalence'])
        long = STATE.audit_shuffle(self.base, {})
        self.assertEqual((long['full_f32_files'], long['replay_files']), (32, 8))

    def test_missing_native_comparison_cannot_be_replaced_by_completion(self):
        self.edit_rows('capture-state.stdout.log', lambda rs: rs.pop(next(
            i for i, r in enumerate(rs) if r.get('kind') == 'comparison')))
        with self.assertRaisesRegex(ValueError, 'label collection'):
            STATE.audit_capture(self.base, {})

    def test_comparison_claiming_wrong_complete_size_rejected(self):
        self.edit_rows('capture-state.stdout.log', lambda rs: next(
            r for r in rs if r.get('kind') == 'comparison').update(elements=1))
        with self.assertRaisesRegex(ValueError, 'size differs'):
            STATE.audit_capture(self.base, {})

    def test_missing_full_raw_rejected(self):
        path = self.base / 'capture-state/candidate_second_pos255.f32'
        data = path.read_bytes(); path.unlink(); self.addCleanup(path.write_bytes, data)
        with self.assertRaisesRegex(ValueError, 'Missing/linked'):
            STATE.audit_capture(self.base, {})

    def test_tree_pre_tail_mismatch_detected_beyond_slots_and_hidden(self):
        self.edit_bytes('capture-state/capture-graphs/tree_B_candidate_capture_state_pos127.f32',
                        lambda data: data[:-4] + struct.pack('<f', .75))
        with self.assertRaisesRegex(ValueError, 'Tree same-graph'):
            STATE.audit_capture(self.base, {})

    def test_actual_tree_head_tail_mismatch_rejected_even_when_state_equal(self):
        self.edit_bytes('capture-state/capture-graphs/tree_B_candidate_logits2_pos127.f32',
                        lambda data: data[:-4] + struct.pack('<f', .6))
        with self.assertRaisesRegex(ValueError, '[Ll]ogits'):
            STATE.audit_capture(self.base, {})

    def test_missing_actual_tree_head_rejected(self):
        path = self.base / 'capture-state/capture-graphs/tree_A_candidate_logits1_pos255.f32'
        data=path.read_bytes();path.unlink();self.addCleanup(path.write_bytes,data)
        with self.assertRaisesRegex(ValueError, 'Missing/linked'):
            STATE.audit_capture(self.base, {})

    def test_actual_tree_head_argmax_metadata_misreport_rejected(self):
        self.edit_rows('capture-state/capture-graphs.stdout.log', lambda rs: next(
            r for r in rs if r.get('kind') == 'capture_tree_branch' and r.get('branch') == 'B').update(
                head_argmax=[0, 2], reference_out=[0, 2], candidate_out=[0, 2]))
        with self.assertRaisesRegex(ValueError, 'actual head/output binding'):
            STATE.audit_capture(self.base, {})

    def test_nonfinite_full_logits_rejected(self):
        self.edit_bytes('capture-state/reference_first_pos128.f32',
                        lambda data: data[:-4] + struct.pack('<f', float('nan')))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            STATE.audit_capture(self.base, {})

    def test_disabled_capture_write_rejected(self):
        self.edit_bytes('capture-state/capture_disabled_slots_pos126.f32',
                        lambda data: struct.pack('<f', .5) + data[4:])
        with self.assertRaisesRegex(ValueError, 'Disabled capture wrote'):
            STATE.audit_capture(self.base, {})

    def test_configured_mode_must_be_actual_candidate_one(self):
        self.edit_rows('capture-state.stdout.log', lambda rs: next(
            r for r in rs if r.get('kind') == 'capture_metadata').update(configured_candidate_mode=0))
        with self.assertRaisesRegex(ValueError, 'fixture boundary'):
            STATE.audit_capture(self.base, {})

    def test_batch_result_argmax_must_match_authoritative_ids(self):
        self.edit_rows('capture-state/capture-graphs.stdout.log', lambda rs: next(
            r for r in rs if r.get('kind') == 'batch2_result').update(first_argmax=2))
        with self.assertRaisesRegex(ValueError, 'Batch2 result'):
            STATE.audit_capture(self.base, {})

    def test_rollback_failure_protocol_cannot_be_omitted(self):
        self.edit_rows('capture-state/capture-graphs.stdout.log', lambda rs: next(
            r for r in rs if r.get('kind') == 'position_complete').update(failures=0))
        with self.assertRaisesRegex(ValueError, 'rollback/failure'):
            STATE.audit_capture(self.base, {})

    def test_long_state_actual_mode_and_reference_counters(self):
        for arm, updates in (('state-b', {'actual_mode': 0}),
                             ('state-a1', {'requests': 8, 'encoded': 8})):
            path = self.base / (arm + '.stdout.log'); original = path.read_bytes()
            values = [json.loads(x) for x in path.read_text().splitlines()]
            next(r for r in values if r['kind'] == 'long_indexer_complete').update(updates)
            write_rows(path, values)
            try:
                with self.assertRaises(ValueError): STATE.audit_shuffle(self.base, {})
            finally:
                path.write_bytes(original)

    def test_cross_mode_mismatch_rejected_even_when_each_arm_exact(self):
        for arm in ('reference', 'candidate'):
            self.edit_bytes(f'state-b/{arm}_first_pos8192.f32',
                            lambda data: data[:-4] + struct.pack('<f', .75))
        with self.assertRaisesRegex(ValueError, 'Cross-mode complete'):
            STATE.audit_shuffle(self.base, {})


class FrozenAndFullmodelTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='myds-freeze-reader-cpu-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        (self.source / 'tools').mkdir(parents=True)
        (self.source / 'bin').mkdir()
        for name in ('bench_reproduce.py', 'bench_phase_audit.py'):
            shutil.copyfile(ROOT / 'tools' / name, self.source / 'tools' / name)
        (self.source / 'bin/bench_phase').write_text('CPU identity fixture; never execute\n')
        for i in range(232): (self.source / f'input-{i}').write_text(str(i))
        files = [p for p in self.source.rglob('*') if p.is_file()]
        self.snapshot = dict(source_head='CPU identity only', frozen_inputs=dict(
            source_head='CPU identity only', extra_files_sha256={str(p): STATE.sha(p) for p in files}))
        write_json(self.root / 'production-snapshot.json', self.snapshot)
        write_json(self.root / 'window-result.json', dict(status='completed', error=None,
                    errors=[], restoration={'restored': True}))

    def test_complete_235_freeze_and_all_bin_inventory_are_required(self):
        STATE.frozen_binding(self.snapshot, {}, self.source)
        (self.source / 'bin/unrecorded').write_text('unexpected artifact')
        with self.assertRaisesRegex(ValueError, 'native artifacts changed'):
            STATE.frozen_binding(self.snapshot, {}, self.source)

    def test_current_reader_mutation_rejected_before_loading_python(self):
        path = self.source / 'tools/bench_reproduce.py'
        path.write_text('raise RuntimeError("MUST NOT EXECUTE")\n')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            FULL.audit(self.root, self.source, 'capture')

    def test_missing_raw_run_does_not_pass_with_only_closed_window(self):
        with self.assertRaises(FileNotFoundError):
            FULL.audit(self.root, self.source, 'capture')

    def test_restoration_required_before_fullmodel_reader_load(self):
        write_json(self.root / 'window-result.json', dict(status='completed', error=None,
                    errors=[], restoration={'restored': False}))
        with self.assertRaisesRegex(ValueError, 'not completed/restored'):
            FULL.audit(self.root, self.source, 'capture')


if __name__ == '__main__':
    unittest.main()
