#!/usr/bin/env python3
"""Model-free Q2 down parity in separate processes. Executes GPU components.

--host-only exercises actual initialization rejection before Metal resources;
it neither submits GPU work nor loads a model. The normal suite is scheduled by
the root validation controller, not run implicitly by building the fixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(binary, argv, gate, down):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('MYDS_', 'DS4_'))}
    env.update(MYDS_DISABLE_MOE_MM='1', MYDS_METAL_CONCURRENT='1')
    if gate is not None:
        env['MYDS_MOE_NSG'] = gate
    if down is not None:
        env['MYDS_MOE_DOWN_NSG'] = down
    result = subprocess.run([str(binary), *argv], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f'Fixture failed (gate={gate!r}, down={down!r}):\n'
                           f'{result.stdout}\n{result.stderr}')
    records = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{')]
    if not records or not records[-1].get('pass'):
        raise RuntimeError('Fixture did not report completion')
    return records


def suite(args):
    binary = args.binary.resolve()
    rejected = []
    for value in ('', '0', '1', '3', '8', '-2', '2x', '4 ', 'invalid'):
        records = run(binary, ['--reject'], '4', value)
        if records != [{'kind': 'rejected_before_metal_resources', 'pass': True}]:
            raise RuntimeError('Invalid setting was not rejected before resources')
        rejected.append(value)
    report = {'schema': 'myds_moe_down_nsg_component_parity_v1',
              'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'host_only': args.host_only, 'rejected_values': rejected,
              'scope': 'Seeded valid nonzero IQ2/Q2 blocks, not captured model weights; no performance claim',
              'processes': []}
    if args.host_only:
        report['audit_pass'] = True
        return report
    baseline = None
    with tempfile.TemporaryDirectory(prefix='myds-moe-down-nsg-') as temp:
        for name, gate, down, effective_gate, effective_down, reinit in (
                ('explicit4', '4', '4', 4, 4, False),
                ('explicit2-reinit', '4', '2', 4, 2, True),
                ('inherit4', '4', None, 4, 4, False),
                ('inherit2', '2', None, 2, 2, False),
                ('default2', None, None, 2, 2, False),
                ('gate2-down4', '2', '4', 2, 4, False)):
            output = Path(temp) / name
            output.mkdir()
            argv = [str(ROOT / 'bin'), str(output), str(effective_gate), str(effective_down)]
            if reinit:
                argv.append('--reinit')
            records = run(binary, argv, gate, down)
            cases = [r for r in records if r['kind'] == 'case']
            phases = (0, 1, 2) if reinit else (0,)
            expected_downs = (effective_down, 4, effective_gate) if reinit else (effective_down,)
            seen = set()
            phase_hashes = []
            for phase, expected in zip(phases, expected_downs):
                phase_cases = [r for r in cases if r['phase'] == phase]
                if len(phase_cases) != 10:
                    raise RuntimeError('Expected ten nonzero cases per initialization')
                hashes = {}
                raw = {}
                for case in phase_cases:
                    if case['iq2_nsg'] != effective_gate or case['down_nsg'] != expected:
                        raise RuntimeError('Runtime did not apply/retain the selected geometry')
                    if not all(case[key] for key in ('finite', 'nonzero', 'canaries')):
                        raise RuntimeError('Invalid numeric fixture')
                    key = case['name']
                    if (phase, key) in seen:
                        raise RuntimeError('Duplicate case')
                    seen.add((phase, key))
                    data = (output / f'phase{phase}-{key}.bin').read_bytes()
                    if len(data) != case['first_bytes'] + case['second_bytes']:
                        raise RuntimeError('Incomplete full output export')
                    hashes[key] = hashlib.sha256(data).hexdigest()
                    raw[key] = data
                if baseline is None:
                    baseline = raw
                if raw != baseline:
                    mismatches = [key for key in raw if raw[key] != baseline.get(key)]
                    raise RuntimeError('Cross-process/reinit outputs are not byte exact: ' + ', '.join(mismatches))
                phase_hashes.append({'phase': phase, 'down_nsg': expected, 'output_sha256': hashes})
            report['processes'].append({'name': name, 'iq2_nsg': effective_gate,
                                        'phases': phase_hashes})
    report.update(audit_pass=True, byte_exact=True, cases_per_initialization=10,
                  tested_initializations=8, model_loaded=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/test_moe_down_nsg')
    parser.add_argument('--host-only', action='store_true')
    parser.add_argument('--out', type=Path, help='Optional JSON evidence path')
    args = parser.parse_args()
    report = suite(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
