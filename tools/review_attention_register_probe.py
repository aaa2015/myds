#!/usr/bin/env python3
"""Independently review complete private attention partial/stat/output blobs and saved A-B-A timings."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f'{value:016x}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stdout', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.stdout.read_text().splitlines() if line.startswith('{')]
    cases = [r for r in records if r.get('kind') == 'partial_parity']
    results = [r for r in records if r.get('kind') == 'result']
    goldens = [r for r in records if r.get('kind') == 'golden_library']
    require(len(results) == len(goldens) == 1, 'Missing or duplicated completion/golden identity')
    require(results[0].get('passed') is True, 'Native probe failed')
    require(len(cases) in (20, 240), 'Incomplete fixture matrix')
    quick = len(cases) == 20
    expected_cases = {
        f'keys{keys}-start{start}-mode{mode}-{path}-concurrent{concurrent}'
        for keys in ((128, 640) if quick else (1, 31, 32, 128, 512, 640))
        for start in ((127,) if quick else (0, 127))
        for mode in range(5)
        for path in (('direct',) if quick else ('direct', 'staged'))
        for concurrent in range(2)
    }
    actual_cases = {r['case'] for r in cases}
    require(actual_cases == expected_cases,
            'Fixture matrix differs; missing=' + repr(sorted(expected_cases - actual_cases)) +
            ', additional=' + repr(sorted(actual_cases - expected_cases)))
    require(results[0]['full_partial_parity_pairs'] == len(cases), 'Parity count differs')
    require(len({r['case'] for r in cases}) == len(cases), 'Duplicated fixture case')
    require(results[0]['full_output_parity_pairs'] == 4 * len(cases), 'Output count differs')
    expected, identities = set(), []
    for case in cases:
        for key in ('all_partial_and_stats_bytes_equal', 'actual_golden_matches_private_original_symbol',
                    'debug_and_uninstrumented_output_equal_to_production_packed8', 'canaries'):
            require(case.get(key) is True, 'Native parity gate false: ' + key)
        prefix = case['case']
        require(Path(prefix).name == prefix and prefix not in ('.', '..'), 'Unsafe case filename')
        golden_name = prefix + '-tg-actual-production-golden-output.bin'
        golden_path = args.artifacts / golden_name
        require(not golden_path.is_symlink(), 'Symlink artifact')
        golden = golden_path.read_bytes()
        require(len(golden) == 131072 and fnv(golden) == case['output_fnv1a64'], 'Golden output identity differs')
        expected.add(golden_name)
        identities.append({'name': golden_name, 'bytes': len(golden), 'sha256': hashlib.sha256(golden).hexdigest()})
        for kind, size in (('partial', 4194304), ('stats', 24576), ('output', 131072)):
            names = [prefix + '-' + arm + '-' + kind + '.bin' for arm in ('tg', 'register')]
            paths = [args.artifacts / name for name in names]
            require(all(p.is_file() and not p.is_symlink() for p in paths), 'Missing/unsafe artifact')
            payloads = [p.read_bytes() for p in paths]
            require(len(payloads[0]) == len(payloads[1]) == size, 'Artifact length differs: ' + kind)
            require(payloads[0] == payloads[1], 'Full raw payload bytes differ: ' + kind)
            require(fnv(payloads[0]) == case[kind + '_fnv1a64'], 'Raw payload differs from native identity: ' + kind)
            if kind == 'output':
                require(payloads[0] == golden, 'Debug output differs from actual production golden')
            expected.update(names)
            identities.extend({'name': name, 'bytes': size, 'sha256': hashlib.sha256(data).hexdigest()}
                              for name, data in zip(names, payloads))
    require({p.name for p in args.artifacts.iterdir()} == expected, 'Missing or additional files')
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get('kind') == 'gpu_timing':
            require(row['scope'] == 'uninstrumented_packed_kernel_plus_repeat_barriers', 'Wrong timing scope')
            require(row.get('full_output_bit_exact') is True, 'Timing output gate failed')
            groups[row['keys']][row['arm']].append(row)
    if groups:
        require(set(groups) == ({128, 640} if quick else {1, 31, 32, 128, 512, 640}), 'Incomplete timing shape set')
    timing = []
    for keys, arms in sorted(groups.items()):
        require(set(arms) == {'A1', 'B', 'A2'}, 'Incomplete A-B-A')
        summary = {}
        formal_count = None
        for arm, rows in arms.items():
            require(sum(r['warmup'] is True for r in rows) == 1, 'Exactly one warmup required')
            formal = [r for r in rows if r['warmup'] is False]
            require(len(formal) >= 3, 'At least three formal samples required')
            require(formal_count is None or formal_count == len(formal), 'Sample count differs')
            formal_count = len(formal)
            require(all(r['gpu_us_per_call'] > 0 for r in rows), 'Invalid GPU duration')
            expected_kernel = ('probe_attention_register' if arm == 'B' else
                               'kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512')
            require(all(r['kernel'] == expected_kernel for r in rows), 'Timing is not actual golden/candidate PSO')
            require(all(r['dynamic_bytes'] == (22400 if arm == 'B' else 24448) for r in rows), 'Wrong dynamic memory')
            require(sorted(r['sample'] for r in rows) == list(range(len(rows))), 'Timing samples missing/duplicated')
            require(len({r['repetitions'] for r in rows}) == 1, 'Timing repetition count differs')
            summary[arm] = statistics.median(r['gpu_us_per_call'] for r in formal)
        midpoint = (summary['A1'] + summary['A2']) / 2
        timing.append({'keys': keys, 'median_gpu_us': summary, 'formal_per_arm': formal_count,
                       'time_saved_vs_control_midpoint_percent': 100 * (1 - summary['B'] / midpoint),
                        'control_drift_percent': 100 * (summary['A2'] / summary['A1'] - 1),
                       'time_saved_vs_A1_percent': 100 * (1-summary['B']/summary['A1']),
                       'time_saved_vs_A2_percent': 100 * (1-summary['B']/summary['A2']),
                       'profit_exceeds_absolute_control_drift':
                           min(100 * (1-summary['B']/summary['A1']), 100 * (1-summary['B']/summary['A2'])) >
                           abs(100 * (summary['A2']/summary['A1']-1)),
                       'formal_values_us': {arm: [r['gpu_us_per_call'] for r in rows if not r['warmup']]
                                            for arm, rows in arms.items()}})
    report = {'raw_review_pass': True, 'fixture_not_fullmodel': True, 'golden': goldens[0],
              'cases': len(cases), 'exact_case_matrix': 'quick20' if quick else 'full240', 'files_checked': len(expected), 'timing': timing,
              'stdout_sha256': hashlib.sha256(args.stdout.read_bytes()).hexdigest(),
              'reviewer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'payloads': identities}
    require(not args.out.exists(), 'Refusing output overwrite')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'payloads'}))


if __name__ == '__main__':
    main()
