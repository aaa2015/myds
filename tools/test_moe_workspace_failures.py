#!/usr/bin/env python3
"""Independent CPU reader for bench_moe_workspace_failures original artifacts.

No GPU launch. --self-test checks false acceptance using bounded CPU fixtures;
--root accepts only complete real functional evidence, never timing/cancel proof.
"""
import argparse
import array
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True


class Rejected(ValueError):
    pass


def need(value, message):
    if not value:
        raise Rejected(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def finite_words(block):
    words = array.array('I'); words.frombytes(block)
    if sys.byteorder != 'little':
        words.byteswap()
    return all((word & 0x7f800000) != 0x7f800000 for word in words)


def same_output(left, right, expected_bytes, finite=True):
    need(left.stat().st_size == right.stat().st_size == expected_bytes,
         'Original output extent differs')
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(1 << 20), b.read(1 << 20)
            need(x == y, 'Original output bytes differ')
            if not x:
                break
            need(not finite or (len(x) % 4 == 0 and finite_words(x)), 'Nonfinite full F32 output')


def guard(path):
    need(path.read_bytes() == bytes([0xa5]) * 512, 'Original guard bytes differ')


def pool_identity(stats):
    core = ['capacity', 'moe_batch', 'compact', 'generation', 'owner_bytes', 'owners', 'views',
            'legacy_gate_bytes', 'legacy_up_bytes', 'legacy_mid_bytes']
    scratch = ['mm_capacity', 'f32_capacity', 'generation', 'mm_mid_bytes', 'f32_gate_bytes',
               'f32_up_bytes', 'f32_mid_bytes', 'retired_bytes', 'retired_owners', 'allocated_bytes']
    return ([stats[key] for key in core], [stats['scratch'][key] for key in scratch])


def metadata_gate(manifest):
    need(manifest.get('schema_version') == 1 and manifest.get('probe') == 'real_prefill_moe_workspace_failures',
         'Diagnostic schema differs')
    need(manifest.get('status') == 'complete' and manifest.get('phase') == 'functional_failures',
         'Complete real functional diagnostics required')
    need(manifest.get('full_f32_bit_exact') is True and manifest.get('real_oom') is False and
         manifest.get('timing_measured') is False and manifest.get('fullmodel_cancel_recovery_proven') is False,
         'Functional scope/correctness declarations differ')
    results = manifest.get('results', [])
    need(len(results) == 9, 'Exact nine diagnostic groups required')
    masked = [row for row in results if row.get('kind') == 'mm_pso_mask']
    need({(row['mask'], row['rows']) for row in masked} ==
         {(1, 33), (2, 33), (4, 33), (1, 16384)} and len(masked) == 4,
         'All three MM-only requirements and full B16384 fallback required')
    for row in masked:
        need(row.get('same_mask_oracle') is True and row.get('timing_measured') is False,
             'Matching forced-provider oracle required')
        arms = row.get('arms', [])
        need(len(arms) == 2 and [a['provider'] for a in arms] == ['native', 'compact'], 'Native then compact arms')
        need(all(a.get('full_f32_bit_exact') is True and a.get('finite_guards_written') is True for a in arms),
             'Full mask output parity and guards required')
        need(all(a['workspace']['mm_encoded'] == 0 and a['workspace']['mv_encoded'] > 0 for a in arms),
             'Both authoritative and candidate actual providers must use matching-mask MV')
        s = arms[1]['workspace']['scratch']
        need(s['last_provider'] == 2 and s['mm_encoded'] == 0 and s['mv_encoded'] > 0,
             'Actual forced whole-batch MV provider required')
    shared = [row for row in results if row.get('kind') == 'shared_pso_failclosed']
    need(len(shared) == 1, 'Shared sum6 unavailability coverage required')
    row = shared[0]
    need(row['mask'] == 8 and row['rows'] == 33 and row['shared_dependency'] == 'sum6' and
         row['provider_when_unavailable'] == 'NONE' and row.get('timing_measured') is False,
         'A missing shared requirement must not claim MV fallback')
    need(len(row['arms']) == 2 and [a['provider'] for a in row['arms']] == ['native', 'compact'],
         'Both unavailable shared-provider arms required')
    need(all(a.get('returned_failure') is True and a['dispatches_before'] == a['dispatches_after'] and
             a.get('failed_output_unchanged') is True and a.get('same_handle_recovery') is True and
             a.get('full_f32_bit_exact') is True and a.get('finite_guards_written') is True for a in row['arms']),
         'Shared-provider failure must preserve output then recover on same handle')
    lazy = [row for row in results if row.get('kind') == 'fallback_allocation_failure']
    need(len(lazy) == 3 and {row['ordinal'] for row in lazy} == {1, 2, 3}, 'Three lazy pool allocation failures required')
    for row in lazy:
        need(row['mask'] == 1 and row['rows'] == 33 and row['site'] ==
             {1: 'fallback_gate', 2: 'fallback_up', 3: 'fallback_mid'}[row['ordinal']], 'Exact fallback allocation site')
        need(row.get('simulated_null') is True and row.get('real_oom') is False and row['failure_hits'] == 1,
             'Exactly one simulated NULL required')
        failed = [a for a in row['allocation_trace'] if not a['success']]
        need(len(failed) == 1 and failed[0]['ordinal'] == row['ordinal'] and failed[0]['site'] == row['site'],
             'Actual allocation trace must record requested failure')
        need(row['dispatches_before'] == row['dispatches_after'] and row.get('failed_output_unchanged') is True,
             'No partial failed batch dispatch or output write')
        need(pool_identity(row['before']) == pool_identity(row['after_failure']), 'Failed pool transaction changed ownership')
        need(row.get('full_f32_bit_exact') is True and row.get('finite_guards_written') is True and
             row.get('owner_accounting_restored') is True, 'Same-handle retry and release gate')
        need(row['retry']['scratch']['mm_encoded'] == 0 and row['retry']['scratch']['mv_encoded'] > 0,
             'Retry must use masked MV')
    reuse = [row for row in results if row.get('kind') == 'long_short']
    need(len(reuse) == 1, 'One explicit long then short group')
    row = reuse[0]
    need(row['long_rows'] == 16384 and row['short_rows'] == 31 and row.get('same_handle') is True and
         row.get('full_f32_bit_exact') is True and row.get('finite_guards_written') is True and
         row.get('timing_measured') is False, 'Same-workspace long then short exact output')
    need(row['long_workspace']['generation'] == row['short_workspace']['generation'], 'Workspace silently replaced')
    return results


def review(root):
    root = root.resolve(strict=True); path = root / 'artifact-manifest.json'
    manifest = json.loads(path.read_text()); results = metadata_gate(manifest)
    artifacts = manifest.get('files', []); seen = set()
    for item in artifacts:
        name = item['path']; relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts and name not in seen, 'Unique safe logical artifact path')
        seen.add(name); source = (root / relative).resolve(strict=True)
        need(root in source.parents and source.stat().st_size == item['bytes'] and digest(source) == item['sha256'],
             'Original artifact extent/SHA differs')
    comparisons = []
    def failed_output(stem, rows):
        need(stem + '.f32' in seen and stem + '.guards' in seen, 'Failed-attempt raw preservation')
        guard(root / (stem + '.guards'))
        failed = root / (stem + '.f32'); need(failed.stat().st_size == rows * 20480, 'Failed output extent')
        with failed.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                need(block == bytes([0xa5]) * len(block), 'Failed attempt wrote output')
    def compare(stem_a, stem_b, rows):
        for stem in (stem_a, stem_b):
            need(stem + '.f32' in seen and stem + '.guards' in seen, 'Original output and guards absent')
            guard(root / (stem + '.guards'))
        same_output(root / (stem_a + '.f32'), root / (stem_b + '.f32'), rows * 20480)
        comparisons.append(dict(left=stem_a, right=stem_b, rows=rows, full_f32_bit_exact=True))
    for row in results:
        if row['kind'] == 'mm_pso_mask':
            compare(row['arms'][0]['output_stem'], row['arms'][1]['output_stem'], row['rows'])
        elif row['kind'] == 'fallback_allocation_failure':
            compare(row['oracle_stem'], row['retry_output_stem'], row['rows'])
            failed_output(row['failed_output_stem'], row['rows'])
        elif row['kind'] == 'shared_pso_failclosed':
            for arm in row['arms']:
                failed_output(arm['failed_output_stem'], row['rows'])
            compare(row['arms'][0]['recovery_output_stem'], row['arms'][1]['recovery_output_stem'], row['rows'])
        else:
            compare(row['native_short_stem'], row['compact_short_stem'], row['short_rows'])
    return dict(schema_version=1, scope='independent_component_failure_artifact_review',
                manifest_sha256=digest(path), original_artifacts=len(artifacts), groups=len(results),
                original_output_comparisons=comparisons, evidence_valid=True,
                functional_failure_gate_pass=True, real_oom_tested=False,
                performance_measured=False, fullmodel_cancel_recovery_proven=False,
                decision='component_failure_artifacts_passed_fullmodel_recovery_still_required',
                limitations=['Raw original outputs/guards and all declared file SHA values are independently checked.',
                             'Native input/weight immutability and selected provider metadata come from the separately frozen native harness.',
                             'No asynchronous cancellation or full-model recovery is proved here.'])


def self_test():
    # Metadata false-accept controls avoid allocating a full 16384-row fixture.
    stats = dict(capacity=32, moe_batch=64, compact=True, generation=1, owner_bytes=100,
                 requests=1, mm_encoded=0, mv_encoded=1, failures=0,
                 owners=3, views=6, legacy_gate_bytes=0, legacy_up_bytes=0, legacy_mid_bytes=0,
                 scratch=dict(mm_capacity=64, f32_capacity=0, generation=1, mm_mid_bytes=100,
                              f32_gate_bytes=0, f32_up_bytes=0, f32_mid_bytes=0, retired_bytes=0,
                              retired_owners=0, allocated_bytes=100, last_provider=2, mm_encoded=0, mv_encoded=1))
    base = dict(schema_version=1, probe='real_prefill_moe_workspace_failures', status='complete',
                phase='functional_failures', full_f32_bit_exact=True, real_oom=False,
                timing_measured=False, fullmodel_cancel_recovery_proven=False, results=[])
    for mask, rows in [(1,33),(2,33),(4,33),(1,16384)]:
        base['results'].append(dict(kind='mm_pso_mask', mask=mask, rows=rows, same_mask_oracle=True,
                                   timing_measured=False, arms=[dict(provider=p, full_f32_bit_exact=True,
                                   finite_guards_written=True, workspace=copy.deepcopy(stats)) for p in ('native','compact')]))
    base['results'].append(dict(kind='shared_pso_failclosed', mask=8, rows=33,
          shared_dependency='sum6', provider_when_unavailable='NONE', timing_measured=False,
          arms=[dict(provider=p, returned_failure=True, dispatches_before=5, dispatches_after=5,
                     failed_output_unchanged=True, same_handle_recovery=True,
                     full_f32_bit_exact=True, finite_guards_written=True) for p in ('native','compact')]))
    for ordinal, site in [(1,'fallback_gate'),(2,'fallback_up'),(3,'fallback_mid')]:
        base['results'].append(dict(kind='fallback_allocation_failure', mask=1, rows=33, ordinal=ordinal,
             site=site, simulated_null=True, real_oom=False, failure_hits=1,
             allocation_trace=[dict(success=0, ordinal=ordinal, site=site)], dispatches_before=5,
             dispatches_after=5, failed_output_unchanged=True, before=copy.deepcopy(stats),
             after_failure=copy.deepcopy(stats), retry=copy.deepcopy(stats), full_f32_bit_exact=True,
             finite_guards_written=True, owner_accounting_restored=True))
    base['results'].append(dict(kind='long_short', long_rows=16384, short_rows=31, same_handle=True,
             full_f32_bit_exact=True, finite_guards_written=True, timing_measured=False,
             long_workspace=copy.deepcopy(stats), short_workspace=copy.deepcopy(stats)))
    metadata_gate(base); controls = ['positive_complete_metadata']
    mutations = [lambda j:j.update(status='failed'), lambda j:j.update(phase='host_only'),
        lambda j:j.update(full_f32_bit_exact=False), lambda j:j.update(real_oom=True),
        lambda j:j['results'].pop(0), lambda j:j['results'][0].update(mask=2),
        lambda j:j['results'][0]['arms'][1]['workspace']['scratch'].update(mm_encoded=1),
        lambda j:j['results'][5].update(failure_hits=0), lambda j:j['results'][5].update(site='mm_mid'),
        lambda j:j['results'][5].update(dispatches_after=6),
        lambda j:j['results'][5]['after_failure'].update(generation=2),
        lambda j:j['results'][-1]['short_workspace'].update(generation=2),
        lambda j:j['results'][4].update(provider_when_unavailable='MV_F32'),
        lambda j:j['results'][4]['arms'][0].update(dispatches_after=6),
        lambda j:j['results'][4]['arms'][1].update(same_handle_recovery=False)]
    for index, mutate in enumerate(mutations):
        bad = copy.deepcopy(base); mutate(bad)
        try:metadata_gate(bad)
        except Rejected:controls.append('reject_metadata_mutation_' + str(index));continue
        raise AssertionError('false acceptance ' + str(index))
    with tempfile.TemporaryDirectory(prefix='myds-moe-ws-reader-cpu-') as temp:
        a,b=Path(temp)/'a',Path(temp)/'b'; a.write_bytes(bytes(16));b.write_bytes(bytes(16));same_output(a,b,16)
        b.write_bytes(bytes([1])+bytes(15))
        try:same_output(a,b,16)
        except Rejected:controls.append('reject_full_output_byte_change')
        else:raise AssertionError('changed output accepted')
        a.write_bytes(bytes.fromhex('0000807f'));b.write_bytes(a.read_bytes())
        try:same_output(a,b,4)
        except Rejected:controls.append('reject_equal_nonfinite_output')
        else:raise AssertionError('equal infinity accepted')
        a.write_bytes(bytes([0xa5])*512);guard(a);a.write_bytes(bytes([0xa5])*511+bytes(1))
        try:guard(a)
        except Rejected:controls.append('reject_guard_change')
        else:raise AssertionError('guard corruption accepted')
    return dict(schema_version=1, scope='cpu_reader_fixture_controls_only', controls=controls,
                all_pass=True, gpu_or_model_executed=False, real_failure_evidence_reviewed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root',type=Path);group.add_argument('--self-test',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    try:report=self_test() if args.self_test else review(args.root)
    except (Rejected, KeyError, ValueError, OSError) as error:
        report=dict(evidence_valid=False,functional_failure_gate_pass=False,error=str(error));status=1
    else:status=0
    if args.out:
        with args.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2));return status


if __name__=='__main__':
    raise SystemExit(main())
