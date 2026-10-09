#!/usr/bin/env python3
"""Isolated, model-free IQ2 fused-decode constant-LUT parity and GPU timings.

The normal command runs GPU components. --host-only rejects invalid options
before Metal resource creation; --build-missing-library compiles only. Building
the executable or metallib never runs this suite. GPU timings are warm repeated
component command-buffer spans, including output dependency barriers; they are
not full-model decode measurements or cold expert-weight measurements.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32'
CANDIDATE = 'kernel_mul_mv_id_iq2_xxs_pair_swiglu_constant_lut_f32'
REFERENCE_SHA256 = '265dff4968c3dd3482cf42289b68fa002239893cddc857ddb904fd19d4131976'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_contract():
    """Pin the unchanged reference, then compare every candidate math line.

    The allowlisted transform removes only the TG argument/table-copy block and
    changes table address spaces. This is additional review evidence; actual
    full-output GPU comparisons below remain the numerical acceptance gate.
    """
    source = (ROOT / 'metal/moe.metal').read_text()

    def kernel(name):
        start = source.index('kernel void ' + name + '(')
        return source[start:source.index('\n}', start) + 2]

    reference = kernel(REFERENCE)
    if hashlib.sha256(reference.encode()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError('Reference fused kernel changed from the frozen f5d4611 body')
    expected = reference.replace(REFERENCE, CANDIDATE, 1)
    expected = expected.replace('        threadgroup  char * shmem [[threadgroup(0)]],\n', '', 1)
    start = expected.index('    threadgroup uint64_t *svalues =')
    end = expected.index('\n    const int ix = tiisg;', start)
    expected = expected[:start] + expected[end:]
    expected = expected.replace('const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);',
                                'const constant uint8_t *gridg = (const constant uint8_t *)(ds4_metal_iq2xxs_grid + aux8g[l]);', 1)
    expected = expected.replace('const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);',
                                'const constant uint8_t *gridu = (const constant uint8_t *)(ds4_metal_iq2xxs_grid + aux8u[l]);', 1)
    expected = expected.replace('ssigns[(aux32g >> 7 * l) & 127]', 'ds4_metal_ksigns_iq2xs[(aux32g >> 7 * l) & 127]', 1)
    expected = expected.replace('ssigns[(aux32u >> 7 * l) & 127]', 'ds4_metal_ksigns_iq2xs[(aux32u >> 7 * l) & 127]', 1)
    if kernel(CANDIDATE) != expected:
        raise RuntimeError('Candidate changed more than the allowlisted LUT address-space transform')
    return {'reference_body_sha256': REFERENCE_SHA256,
            'candidate_body_sha256': hashlib.sha256(expected.encode()).hexdigest(),
            'math_source_unchanged': True}


def build_missing(directory):
    """Build a complete compatible library with only the candidate symbol absent.

    Does not initialize Metal. Never overwrites a nonempty output directory or
    writes repository build/bin. This fixture also verifies default-old-library
    compatibility and retry after explicit-candidate initialization failure.
    """
    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f'Refusing to overwrite nonempty fixture directory: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('myds_iq2_fixture_builder', ROOT / 'tools/build_metallib.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    combined = builder.PREAMBLE + '\n'
    for name in builder.SHADER_FILES:
        text = (ROOT / 'metal' / name).read_text()
        if name == 'moe.metal':
            if text.count('kernel void ' + CANDIDATE + '(') != 1:
                raise RuntimeError('Expected exactly one opt-in candidate definition')
            text = text.replace('kernel void ' + CANDIDATE + '(',
                                'kernel void kernel_fixture_removed_iq2_constant_lut_f32(', 1)
        combined += f'\n#line 1 "{name}"\n' + text
    combined += builder.COPY_KERNELS
    source = directory / 'fixture.metal'
    air = directory / 'fixture.air'
    library = directory / 'myds.metallib'
    source.write_text(combined)
    commands = [
        ['xcrun', '-sdk', 'macosx', 'metal', '-O3', '-ffast-math',
         '-DDS4_METAL_HC_STABLE=1', '-DDS4_METAL_NORM_RSQRT_DISABLE=1',
         '-c', str(source), '-o', str(air)],
        ['xcrun', '-sdk', 'macosx', 'metallib', str(air), '-o', str(library)],
    ]
    log = ''
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
        log += result.stdout + result.stderr
        if result.returncode:
            (directory / 'build.log').write_text(log)
            raise RuntimeError('Missing-symbol fixture static compilation failed:\n' + log)
    (directory / 'build.log').write_text(log)
    return {'library': str(library), 'library_sha256': sha(library), 'commands': commands,
            'model_loaded': False, 'gpu_executed': False}


def run(binary, argv, mode, nsg, stdout_file=None, stderr_file=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MYDS_', 'DS4_'))}
    env.update(MYDS_DISABLE_MOE_MM='1', MYDS_METAL_CONCURRENT='0', MYDS_MOE_NSG=str(nsg))
    if mode is not None:
        env['MYDS_MOE_IQ2_CONSTANT_LUT'] = mode
    def retain(stdout, stderr, code, timed_out=False):
        def decoded(text):
            return text.decode(errors='replace') if isinstance(text, bytes) else (text or '')
        stdout, stderr = decoded(stdout), decoded(stderr)
        if stdout_file is not None:
            stdout_file.write_text(stdout)
        if stderr_file is not None:
            stderr_file.write_text(stderr)
        if stdout_file is not None:
            stdout_file.with_name(stdout_file.name + '.process.json').write_text(json.dumps({
                'argv': [str(binary), *argv], 'constant_lut_env': mode, 'iq2_nsg': nsg,
                'returncode': code, 'timed_out': timed_out,
                'stdout': str(stdout_file), 'stderr': str(stderr_file),
            }, indent=2) + '\n')
        return stdout, stderr
    try:
        result = subprocess.run([str(binary), *argv], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        retain(error.stdout, error.stderr, None, timed_out=True)
        raise RuntimeError(f'Fixture timed out (lut={mode!r}, nsg={nsg}); retained logs: {stdout_file}, {stderr_file}') from error
    retain(result.stdout, result.stderr, result.returncode)
    if result.returncode:
        raise RuntimeError(f'Fixture failed (lut={mode!r}, nsg={nsg}):\n{result.stdout}\n{result.stderr}')
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if not records or not records[-1].get('pass'):
        raise RuntimeError('Fixture did not report completion')
    return records, result.stdout, result.stderr


def suite(args, output):
    binary = args.binary.resolve()
    contract, _, _ = run(binary, ['--args-contract'], None, 4,
                         output / 'args-contract.stdout', output / 'args-contract.stderr')
    expected_contract = {'kind': 'args_contract', 'pass': True, 'expert_token_shapes': 6,
                         'debug_full_gate_up_rows': True, 'debug_and_release_activation_offsets_equal': True,
                         'timing_uses_release_args': True}
    if contract != [expected_contract]:
        raise RuntimeError('Debug/release argument address contract failed')
    rejected = []
    for index, value in enumerate(('', '2', '-1', '01', '1 ', ' 1', 'true', 'invalid')):
        records, _, _ = run(binary, ['--reject'], value, 4,
                            output / f'rejected-{index}.stdout', output / f'rejected-{index}.stderr')
        if records != [{'kind': 'rejected_before_metal_resources', 'pass': True}]:
            raise RuntimeError('Invalid setting was not rejected before Metal resources')
        rejected.append(value)
    report = {'schema': 'myds_iq2_constant_lut_component_v1', 'source_contract': source_contract(),
              'binary_sha256': sha(binary), 'host_only': args.host_only,
              'args_contract': expected_contract,
              'rejected_values': rejected, 'processes': [], 'model_loaded': False,
              'full_blob_segments': ['reference_gate', 'reference_up', 'reference_mid', 'release_output_when_applicable'],
              'release_output_parity_scope': 'cross_process_full_f32_blob_comparison',
              'scope': 'Seeded nonzero quantized blocks; warm repeated component GPU spans include output barriers; no full-model performance claim'}
    if args.host_only:
        report.update(audit_pass=True, gpu_executed=False)
        return report

    shader_dir = args.shader_dir.resolve()
    report['metallib_sha256'] = sha(shader_dir / 'myds.metallib')
    if args.missing_library_dir:
        missing = args.missing_library_dir.resolve()
        missing_identity = {'library': str(missing / 'myds.metallib'), 'library_sha256': sha(missing / 'myds.metallib')}
    else:
        missing = output / 'missing-library'
        missing_identity = build_missing(missing)
    records, stdout, stderr = run(binary, ['--missing-candidate', str(missing), str(shader_dir)], '1', 4,
                                  output / 'missing-candidate.stdout', output / 'missing-candidate.stderr')
    if records != [{'kind': 'missing_candidate_rejected_and_retried', 'pass': True}]:
        raise RuntimeError('Missing candidate did not fail closed, preserve reference compatibility and reinitialize')
    report['missing_candidate'] = {**missing_identity, 'rejection_cleanup_and_retry_pass': True}
    baseline = None
    initializations = 0
    for name, mode, nsg, effective, reinit in (
            ('default4', None, 4, 0, False), ('explicit0-nsg4', '0', 4, 0, False),
            ('explicit1-nsg4-reinit', '1', 4, 1, True),
            ('explicit0-nsg2', '0', 2, 0, False), ('explicit1-nsg2', '1', 2, 1, False)):
        directory = output / name
        directory.mkdir()
        argv = [str(shader_dir), str(directory), str(effective), str(nsg), str(args.samples), str(args.repetitions)]
        if reinit:
            argv.append('--reinit')
        records, stdout, stderr = run(binary, argv, mode, nsg,
                                      directory / 'stdout.log', directory / 'stderr.log')
        cases = [r for r in records if r['kind'] == 'case']
        phases = (0, 1, 2) if reinit else (0,)
        modes = (effective, 0, 0) if reinit else (effective,)
        phase_reports = []
        for phase, effective_lut in zip(phases, modes):
            selected = [r for r in cases if r['phase'] == phase]
            if len(selected) != 9 or {r['index'] for r in selected} != set(range(9)):
                raise RuntimeError('Expected nine complete unique cases for every initialization')
            raw, hashes = {}, {}
            for case in selected:
                if case['constant_lut'] != bool(effective_lut) or case['iq2_nsg'] != nsg:
                    raise RuntimeError('Unexpected effective PSO configuration')
                if not all(case[k] for k in ('finite', 'nonzero', 'canaries', 'direct_full_f32_byte_exact')):
                    raise RuntimeError('Incomplete numerical gate')
                if case['release_path'] != (case['index'] in (0, 1, 2, 3, 7, 8)):
                    raise RuntimeError('Release host path coverage changed')
                if (case['direct_full_f32_byte_exact_scope'] != 'gate_up_mid' or
                        case['release_output_scope'] != 'finite_nonzero_here_full_f32_parity_in_driver' or
                        case['release_mid_bit_exact'] != case['release_path']):
                    raise RuntimeError('Direct/release numerical acceptance scope changed')
                data = (directory / f'phase{phase}-case{case["index"]}.bin').read_bytes()
                if len(data) != case['bytes']:
                    raise RuntimeError('Incomplete F32 export')
                raw[case['index']] = data
                hashes[case['index']] = hashlib.sha256(data).hexdigest()
            if baseline is None:
                baseline = raw
            if raw != baseline:
                raise RuntimeError(f'Cross-process/reinit complete F32 mismatch: {name} phase {phase}')
            phase_reports.append({'phase': phase, 'constant_lut': bool(effective_lut), 'output_sha256': hashes})
            initializations += 1
        if len(cases) != 9 * len(phases):
            raise RuntimeError('Unexpected additional cases')
        timings = [r for r in records if r['kind'] == 'timing']
        summary = {}
        if len(timings) != 3 * args.samples:
            raise RuntimeError('Incomplete component A-B-A timing matrix')
        for arm in ('A1', 'B', 'A2'):
            selected = [r for r in timings if r['arm'] == arm]
            if {r['sample'] for r in selected} != set(range(args.samples)):
                raise RuntimeError('Missing or duplicate timing samples')
            for r in selected:
                if (r['phase'] != 0 or r['iq2_nsg'] != nsg or r['repetitions'] != args.repetitions or
                        r['dynamic_tg_bytes'] != (0 if arm == 'B' else 2176) or
                        not math.isfinite(r['gpu_us_per_dispatch']) or r['gpu_us_per_dispatch'] <= 0):
                    raise RuntimeError('Invalid GPU timestamps or timing configuration')
            if selected:
                summary[arm] = statistics.median(r['gpu_us_per_dispatch'] for r in selected)
        report['processes'].append({'name': name, 'iq2_nsg': nsg, 'phases': phase_reports,
                                    'median_gpu_us_per_dispatch': summary, 'raw_timings': timings})
    report.update(audit_pass=True, byte_exact=True, cases_per_initialization=9,
                  tested_initializations=initializations, gpu_executed=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/test_moe_iq2_lut')
    parser.add_argument('--shader-dir', type=Path, default=ROOT / 'bin')
    parser.add_argument('--host-only', action='store_true')
    parser.add_argument('--samples', type=int, default=5, help='Formal samples per component A/B/A arm; 0 runs parity only')
    parser.add_argument('--repetitions', type=int, default=32)
    parser.add_argument('--artifacts', type=Path, help='New directory retaining every full F32 output and raw process log')
    parser.add_argument('--out', type=Path, help='Optional JSON evidence path')
    parser.add_argument('--missing-library-dir', type=Path, help='Existing complete library without the candidate symbol')
    parser.add_argument('--build-missing-library', type=Path, help='Static compilation only, then exit; creates a new fixture directory')
    args = parser.parse_args()
    if not 0 <= args.samples <= 100 or not 1 <= args.repetitions <= 10000:
        parser.error('samples must be 0..100, repetitions 1..10000')
    if args.build_missing_library:
        report = {'source_contract': source_contract(), **build_missing(args.build_missing_library)}
    elif args.artifacts:
        directory = args.artifacts.resolve()
        if directory.exists():
            parser.error('artifacts directory must not already exist')
        directory.mkdir(parents=True)
        try:
            report = suite(args, directory)
        except Exception as error:
            (directory / 'failure.json').write_text(json.dumps({'pass': False, 'error': str(error),
                                                              'retained_artifacts': str(directory)}, indent=2) + '\n')
            raise
    else:
        directory = Path(tempfile.mkdtemp(prefix='myds-iq2-lut-'))
        try:
            report = suite(args, directory)
        except Exception as error:
            (directory / 'failure.json').write_text(json.dumps({'pass': False, 'error': str(error),
                                                              'retained_artifacts': str(directory)}, indent=2) + '\n')
            print(f'Retained failed fixture artifacts: {directory}', file=sys.stderr)
            raise
        else:
            shutil.rmtree(directory)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
