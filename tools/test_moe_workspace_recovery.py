#!/usr/bin/env python3
"""Independent CPU gate: successful real context -> injected prefill failure -> recovery.

This imports only raw-byte/hash/state verification from the cancellation reader.
No GPU or service operations. Failed-engine workspace retention/release is recorded,
not forced to be identical between original and transactional factory contracts.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('moe_cancel_reader', Path(__file__).with_name('test_moe_workspace_cancel.py'))
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
need = common.need


def metadata_gate(s):
    need(s.get('schema_version') == 1 and s.get('probe') == 'same_context_moe_workspace_failure_recovery' and
         s.get('status') == 'complete' and s.get('fullmodel_recovery_within_arm_completed') is True, 'Complete real-context failure/recovery schema')
    need(s.get('scenario') in ('allocation', 'shared-pso') and s.get('workspace_mode') in (0, 1), 'Actual fault scope')
    need(s.get('same_context') is True and s.get('rng_manually_reset') is False and
         s.get('extra_gpu_waits_inserted') == 0 and s.get('performance_measured') is False and s.get('real_oom') is False, 'Simulated fault, same context, no reset/wait')
    need(s.get('seed') == 42 and s.get('recovery_tokens') in (8, 32) and
         (s.get('temperature') == 0 or abs(s.get('temperature', -100) - .7) < 1e-6), 'Frozen sampling')
    need(s.get('vocab_size', 0) > 0 and s.get('snapshot_slots') == 44, 'Real state shapes')
    need(s.get('warm_generation_ok') is True and s.get('warm_state_exported') is True and
         s.get('warm_output_tokens') == s['recovery_tokens'] and s.get('old_workspace_present_before_failure') is True and
         s.get('old_capacity', 0) > 0 and s.get('old_moe_batch', 0) >= s['old_capacity'], 'Actual successful old context/workspace')
    need(s.get('failure_api_ok') is False and s.get('failure_stop_reason') == 'error' and
         s.get('failure_output_tokens') == 0 and s.get('failure_gate_pass') is True and s.get('failure_state_exported') is True and
         s.get('failure_prompt_tokens', 0) > s.get('warm_prompt_tokens', 0) > 1, 'Actual longer prefill failure before first callback')
    need(s.get('fault_cleared_while_idle') is True and s.get('failed_moe_progress') == 0,
         'Existing failure path is idle, fault cleared, no successful failed-request MoE encode')
    if s['scenario'] == 'allocation':
        trace = s.get('allocation_trace', [])
        need(s.get('fault_ordinal') == 1 and s.get('mm_pso_mask') == 0 and s.get('injected_nulls') == 1 and
             len(trace) >= 1 and sum(t.get('success') is False for t in trace) == 1 and
             trace[0].get('ordinal') == 1 and trace[0].get('success') is False and trace[0].get('kind') == 1 and
             trace[0].get('bytes', 0) > 0 and trace[0].get('site') in ('x_buf', 'mm_mid'), 'Actual original/compact whole-workspace grow NULL')
    else:
        need(s.get('fault_ordinal') == 0 and s.get('mm_pso_mask') == 8 and s.get('selected_provider') == 0 and
             s.get('injected_nulls') == 0, 'Shared sum PSO is unavailable: NONE, not MV fallback')
    need(s.get('recovery_api_ok') is True and s.get('recovery_stop_reason') == 'length' and
         s.get('recovery_generation_ok') is True and s.get('recovery_state_exported') is True and
         s.get('recovery_output_tokens') == s['recovery_tokens'] and s.get('recovery_prompt_tokens', 0) > 1, 'Same original context successfully recovered')
    for key in ('warm_workspace', 'recovery_workspace'):
        w = s.get(key)
        need(isinstance(w, dict) and w.get('mode') == s['workspace_mode'] and w.get('requests', 0) > 0 and
             w.get('mm_encoded', 0) > 0 and w.get('failures') == 0, 'Actual normal-path adoption: ' + key)
        if s['workspace_mode']:
            need(w.get('legacy_gate_bytes') == w.get('legacy_up_bytes') == w.get('legacy_mid_bytes') == 0 and
                 w.get('arena', {}).get('mm_encoded', 0) > 0, 'Compact arena actual adoption')


def review(off, on, request_paths):
    with common.Hashes() as hashes:
        summaries = [json.loads((root/'summary.json').read_text()) for root in (off, on)]
        for s in summaries:
            metadata_gate(s)
        a, b = summaries
        need([s['workspace_mode'] for s in summaries] == [0, 1], 'Independent original then compact arms')
        for key in ('scenario', 'seed', 'temperature', 'recovery_tokens', 'warm_prompt_tokens', 'failure_prompt_tokens',
                    'recovery_prompt_tokens', 'vocab_size', 'snapshot_slots', 'request_inputs'):
            need(a.get(key) == b.get(key), 'Frozen original/compact request/config: ' + key)
        requests = {}
        for name, path in request_paths.items():
            actual = hashes.file(path); requests[name] = actual
            need(all(all(s['request_inputs'][name].get(k) == actual[k] for k in ('bytes', 'fnv1a64_bytes')) for s in summaries), 'Original request identity: ' + name)
        phases = ('before-failure', 'after-failure', 'recovery')
        arms = [{phase: common.read_state(root, phase, s, hashes) for phase in phases}
                for root, s in zip((off, on), summaries)]
        compared = {phase: common.compare_states(arms[0][phase], arms[1][phase], phase)
                    for phase in ('before-failure', 'recovery')}
        # Failure timing/cleanup differs by design. Individually validate defined
        # raw failed state; never demand identical partial live scratch or owner.
        return {'schema_version': 1, 'evidence_valid': True, 'correctness_pass': True,
                'decision': 'pass_same_context_failure_recovery_within_exported_state_scope',
                'scenario': a['scenario'], 'temperature': a['temperature'], 'seed': 42,
                'recovery_tokens': a['recovery_tokens'], 'same_context_recovery_raw_bit_exact': True,
                'performance_measured': False, 'real_oom': False, 'request_inputs': requests,
                'raw_bytes_compared': compared, 'transient_failure_raw_forced_equal': False,
                'workspace_contracts': [{k: s[k] for k in ('workspace_present_after_failure', 'old_capacity',
                                                         'old_moe_batch', 'old_generation', 'after_capacity',
                                                         'after_moe_batch', 'after_generation')} for s in summaries],
                'arms': [{phase: {k: state[k] for k in ('manifest_sha256', 'witnessed', 'published')}
                          for phase, state in arm.items()} for arm in arms],
                'limitations': ['Injected NULL simulates the allocator failure path; no real process/system memory-pressure OOM is applied.',
                                'Shared PSO unavailability is a diagnostic requirement mask, not destruction of live PSOs.',
                                'Partially updated live cache/carry and unpublished capacity after failure are excluded. Published snapshots/CPU state are validated individually.',
                                'Native and transactional factories may release/retain the old workspace differently on failure; successful recovery is compared byte for byte.',
                                'No extra GPU wait, RNG reset, fresh-context recovery substitute, persistent disk-cache payload or performance proof.']}


def fixture_summary():
    w = {'mode': 0, 'requests': 3, 'mm_encoded': 3, 'failures': 0}
    return dict(schema_version=1, probe='same_context_moe_workspace_failure_recovery', status='complete',
                fullmodel_recovery_within_arm_completed=True, scenario='allocation', workspace_mode=0,
                seed=42, temperature=.7, recovery_tokens=8, same_context=True, rng_manually_reset=False,
                extra_gpu_waits_inserted=0, performance_measured=False, real_oom=False, vocab_size=128,
                snapshot_slots=44, warm_generation_ok=True, warm_state_exported=True, warm_output_tokens=8,
                old_workspace_present_before_failure=True, old_capacity=32, old_moe_batch=32, old_generation=1,
                warm_prompt_tokens=33, failure_prompt_tokens=65, failure_api_ok=False, failure_stop_reason='error',
                failure_output_tokens=0, failure_gate_pass=True, failure_state_exported=True,
                fault_cleared_while_idle=True, failed_moe_progress=0, fault_ordinal=1, mm_pso_mask=0,
                injected_nulls=1, selected_provider=1,
                allocation_trace=[dict(ordinal=1, success=False, kind=1, bytes=655360, site='x_buf')],
                recovery_api_ok=True, recovery_stop_reason='length', recovery_generation_ok=True,
                recovery_state_exported=True, recovery_output_tokens=8, recovery_prompt_tokens=33,
                warm_workspace=w, recovery_workspace=w)


def self_test():
    controls = []; s = fixture_summary()
    def test(name, fn, reject=False):
        try:
            fn(); passed = not reject
        except (common.Rejected, OSError, KeyError, TypeError):
            passed = reject
        controls.append(dict(name=name, passed=passed)); need(passed, 'CPU control ' + name)
    test('actual_grow_failure_metadata', lambda: metadata_gate(s))
    ps = copy.deepcopy(s); ps.update(scenario='shared-pso', mm_pso_mask=8, fault_ordinal=0, selected_provider=0, injected_nulls=0, allocation_trace=[])
    test('shared_sum_failclosed_metadata', lambda: metadata_gate(ps))
    for key, bad in [('rng_manually_reset', True), ('same_context', False), ('extra_gpu_waits_inserted', 1),
                     ('real_oom', True), ('warm_generation_ok', False), ('old_workspace_present_before_failure', False),
                     ('failure_api_ok', True), ('failure_output_tokens', 1), ('failure_prompt_tokens', 33),
                     ('fault_cleared_while_idle', False), ('failed_moe_progress', 1), ('injected_nulls', 0),
                     ('recovery_output_tokens', 7), ('recovery_state_exported', False)]:
        x = copy.deepcopy(s); x[key] = bad
        test('reject_' + key, lambda x=x: metadata_gate(x), True)
    x = copy.deepcopy(ps); x['selected_provider'] = 2
    test('reject_shared_sum_fake_MV', lambda: metadata_gate(x), True)
    with tempfile.TemporaryDirectory(prefix='moe-recovery-reader-') as temp, common.Hashes() as hashes:
        root = Path(temp)
        for phase in ('before-failure', 'after-failure', 'recovery'):
            common.fixture_state(root/phase, phase, s, hashes)
            test('original_raw_extent_fnv_' + phase, lambda phase=phase: common.read_state(root, phase, s, hashes))
        target = root/'after-failure'/'context-scalars.u64le'; target.write_bytes(b'x'+target.read_bytes()[1:])
        test('reject_corrupt_failure_raw', lambda: common.read_state(root, 'after-failure', s, hashes), True)
    return {'schema_version': 1, 'scope': 'cpu_real_context_failure_reader_controls', 'gpu_run': False,
            'controls': controls, 'passed': all(x['passed'] for x in controls)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('off', 'on', 'warm-request', 'failure-request', 'recover-request', 'out'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--self-test', action='store_true'); args = p.parse_args()
    try:
        if args.self_test:
            report = self_test()
        else:
            need(all((args.off, args.on, args.warm_request, args.failure_request, args.recover_request)), 'Original/compact roots plus three original requests')
            report = review(args.off.resolve(), args.on.resolve(), {k: v.resolve() for k, v in
                            [('warm', args.warm_request), ('failure', args.failure_request), ('recover', args.recover_request)]})
    except (common.Rejected, OSError, ValueError, KeyError, TypeError) as error:
        report = dict(evidence_valid=False, correctness_pass=False, performance_measured=False,
                      decision='reject_incomplete_or_inconsistent_failure_recovery', error=str(error))
    if args.out:
        with args.out.open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report, indent=2)); return 0 if report.get('passed', report.get('correctness_pass', False)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
