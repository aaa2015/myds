#!/usr/bin/env python3
"""Independent CPU-only review of the first failed prefill multihead window.

This failed-mode reader never invokes the successful-window review, Metal,
inference, service controls, or a benchmark. It binds the executed archives,
reconstructs all three full KV datasets, and recomputes the first failure from
complete F32 output files using bounded NumPy blocks. Exit zero means valid
failure evidence, not numerical acceptance or a performance result.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tarfile

sys.dont_write_bytecode = True
import review_attention_multihead_artifacts as low
from review_qkv_dual_artifacts import gguf_descriptors

EXPECTED_MISMATCHES = 61341894
EXPECTED_SOURCE = '3052c07c61d358283758d98c23cd2b79a94760203dcc006ff5a4c226182711b4'
EXPECTED_ARTIFACTS = 'abc0289ace3f7101b1c74dac0194b80dc6dd902d5f799b94c78d71c298f24ebe'
LAYERS, ROWS, OUTPUT_BYTES = (2, 8, 19), 2048, 268435456
need, load, digest = low.require, low.load, low.digest


def file_stats(reference, candidate, block_words=1048576):
    """Compare every raw word; signed-zero bit differences remain mismatches.

    ULP distances cover finite pairs only. The monotone integer mapping merges
    +/-0 for numerical ULP distance while preserving their raw-bit distinction.
    No tolerance is used to decide acceptance.
    """
    import numpy as np
    reference, candidate = Path(reference), Path(candidate)
    size = reference.stat().st_size
    need(size == candidate.stat().st_size and size % 4 == 0, 'Output shape/alignment mismatch')
    count, first, maximum_ulp, maximum_abs = 0, None, 0, 0.0
    nonfinite = [0, 0]
    poison = [0, 0]
    first_values = None
    offset = 0
    with reference.open('rb') as a, candidate.open('rb') as b:
        while True:
            left, right = a.read(block_words * 4), b.read(block_words * 4)
            need(len(left) == len(right), 'Output read lengths differ')
            if not left:
                break
            u, v = np.frombuffer(left, dtype='<u4'), np.frombuffer(right, dtype='<u4')
            bad = u != v
            count += int(np.count_nonzero(bad))
            if first is None and bool(np.any(bad)):
                local = int(np.flatnonzero(bad)[0])
                first = offset + local
                rf, cf = float(u.view('<f4')[local]), float(v.view('<f4')[local])
                first_values = dict(reference_bits=f'{int(u[local]):08x}', candidate_bits=f'{int(v[local]):08x}',
                                    reference_f32=rf if math.isfinite(rf) else repr(rf),
                                    candidate_f32=cf if math.isfinite(cf) else repr(cf))
            fa = (u & np.uint32(0x7f800000)) != np.uint32(0x7f800000)
            fb = (v & np.uint32(0x7f800000)) != np.uint32(0x7f800000)
            nonfinite[0] += int(np.count_nonzero(~fa))
            nonfinite[1] += int(np.count_nonzero(~fb))
            poison[0] += int(np.count_nonzero(u == np.uint32(0xa5a5a5a5)))
            poison[1] += int(np.count_nonzero(v == np.uint32(0xa5a5a5a5)))
            valid = fa & fb
            if bool(np.any(valid)):
                x, y = u[valid], v[valid]
                def ordered(words):
                    magnitude = (words & np.uint32(0x7fffffff)).astype(np.uint64)
                    pivot = np.uint64(0x80000000)
                    return np.where((words & np.uint32(0x80000000)) != 0, pivot - magnitude, pivot + magnitude)
                oa, ob = ordered(x), ordered(y)
                distance = np.maximum(oa, ob) - np.minimum(oa, ob)
                maximum_ulp = max(maximum_ulp, int(distance.max()))
                difference = np.abs(x.view('<f4').astype(np.float64) - y.view('<f4').astype(np.float64))
                maximum_abs = max(maximum_abs, float(difference.max()))
            offset += len(u)
    need(offset == size // 4, 'Not every output word was compared')
    coordinates = None if first is None else dict(query=first // (64 * 512), head=(first // 512) % 64, dimension=first % 512)
    return dict(bytes=size, elements=offset, mismatches=count, first_mismatch=first,
                first_coordinates=coordinates, first_values=first_values,
                max_ulp_finite_pairs=maximum_ulp, max_abs_finite_pairs=maximum_abs,
                reference_nonfinite=nonfinite[0], candidate_nonfinite=nonfinite[1],
                reference_poison_words=poison[0], candidate_poison_words=poison[1],
                all_words_compared=True, comparison='raw F32 uint32 equality; no tolerance')


def verify_manifest(directory):
    manifest = load(directory / 'artifact-manifest.json')
    entries, seen, hashes = manifest['files'], set(), {}
    for entry in entries:
        relative = entry['path']
        need(relative not in seen, 'Duplicate logical manifest path')
        seen.add(relative)
        path = low.path_inside(directory, relative)
        st = path.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in hashes:
            hashes[key] = digest(path)
        need(st.st_size == entry['bytes'] and hashes[key] == entry['sha256'], 'Raw logical payload SHA/size: ' + relative)
        if entry.get('storage_alias_of'):
            other = low.path_inside(directory, entry['storage_alias_of']).stat()
            need((st.st_dev, st.st_ino) == (other.st_dev, other.st_ino), 'Declared hardlink identity mismatch')
    return manifest, seen, len(hashes)


def verify_archives(root, source_archive, artifacts_archive, manifest, capture):
    need(digest(source_archive) == EXPECTED_SOURCE, 'First-executed source archive SHA differs')
    need(digest(artifacts_archive) == EXPECTED_ARTIFACTS, 'First-executed artifact archive SHA differs')
    bindings = load(root / 'stage-inputs.json')
    stage_roots = [Path(path).parent for path in bindings if Path(path).name == 'Makefile']
    need(len(stage_roots) == 1, 'Unique original stage root required')
    stage = stage_roots[0]
    binaries, source_count = {}, 0
    with tarfile.open(source_archive, 'r:*') as archive:
        total_source_files = sum(member.isfile() for member in archive.getmembers())
        for path, expected in bindings.items():
            item = Path(path)
            if not item.is_relative_to(stage):
                need(item.name == 'protocol.json' and digest(root / 'protocol.json') == expected,
                     'Unexpected source outside frozen stage')
                continue
            relative = str(item.relative_to(stage))
            if relative.startswith('bin/'):
                binaries[item.name] = expected
            else:
                need(low.archive_member(archive, relative) == expected, 'Executed source member SHA: ' + relative)
                source_count += 1
    need(set(binaries) == {'bench_attention_multihead', 'capture_attention_multihead',
                           'count_chat_tokens', 'myds.metallib'}, 'Four executed artifacts required')
    with tarfile.open(artifacts_archive, 'r:*') as archive:
        for name, expected in binaries.items():
            need(low.archive_member(archive, 'bin/' + name) == expected, 'Executed binary/library archive SHA: ' + name)
    need(manifest['binary_sha256'] == binaries['bench_attention_multihead'] and
         manifest['probe_metallib_sha256'] == binaries['myds.metallib'] and
         capture['source']['capture_binary_sha256'] == binaries['capture_attention_multihead'] and
         capture['source']['capture_metallib_sha256'] == binaries['myds.metallib'], 'Executed archive provenance mismatch')
    return dict(source_archive_sha256=digest(source_archive), artifacts_archive_sha256=digest(artifacts_archive),
                prebound_source_members_verified=source_count, source_archive_regular_members=total_source_files,
                additional_archive_members_not_in_executed_input_bindings=total_source_files - source_count,
                executed_artifact_sha256=binaries)


def review(root, source_archive, artifacts_archive):
    root = root.resolve()
    directory = root / 'component'
    manifest, names, inode_count = verify_manifest(directory)
    need(manifest['status'] == 'failed' and manifest['failure'] == 'complete attention output bits/finite/written/guards' and
         manifest['total_cases'] == 3 and manifest['samples'] == 9 and manifest['repetitions_per_cb'] == 1 and
         manifest['mode'] == 'concurrent' and manifest['fullmodel'] is False, 'Expected first correctness failure manifest')
    before, restored, summary = [load(root / name) for name in ('before.json', 'restored.json', 'summary.json')]
    identity_keys = ('argv_raw_hex', 'myds_env_raw_hex', 'cwd', 'log', 'binary', 'binary_sha256',
                     'metallib', 'metallib_sha256', 'config', 'config_sha256', 'effective_myds_env', 'nice')
    need(all(before[key] == restored[key] for key in identity_keys) and
         before['health'] == restored['health'] == {'status': 'ok'}, 'Recorded exact restoration mismatch')
    need(summary['status'] == 'failed' and summary['correctness_pass'] is False and
         summary['fullmodel_run'] is False and summary['restoration']['restored'] is True and
         summary['restoration']['pid'] == restored['pid'], 'Runner did not reject/restore first failure')
    need(digest(root / 'original-myds.conf') == before['config_sha256'] and
         load(root / 'effective-env.json') == before['effective_myds_env'], 'Production configuration identity')
    for kind in ('binary', 'metallib'):
        need(digest(root / 'production-artifacts' / Path(before[kind]['path']).name) == before[kind + '_sha256'],
             'Frozen production artifact mismatch')
    need(manifest['production_metallib_sha256'] == before['metallib_sha256'], 'Wrong production golden library')
    environment = before['effective_myds_env']
    need(len(environment) == 51 and environment['MYDS_METAL_CONCURRENT'] == '1' and
         environment['MYDS_PREFILL_CHUNK_SIZE'] == '2048' and environment.get('MYDS_ATTN_GATHER', '1') == '1' and
         environment.get('MYDS_ATTN_VEC_HEAD_FIRST', '1') == '1' and
         environment.get('MYDS_PREFILL_ATTN', 'vec') == 'vec', 'Production prefill provider/config')
    steps = load(root / 'steps.json')
    need([row['name'] for row in steps] == ['tokenize', 'capture', 'host-validation', 'component'] and
         [row['returncode'] for row in steps] == [0, 0, 0, 1], 'Expected only the first component failure')
    host_manifest, _, _ = verify_manifest(root / 'host-validation')
    need(host_manifest['status'] == 'host_only', 'Host gate did not finish without Metal')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line.strip()]
    need(events[-1]['kind'] == 'window_complete' and
         not any(row['kind'] in ('monitor_failure', 'restoration_failure') for row in events) and
         sum(row['kind'] == 'window_failure' for row in events) == 1, 'Unexpected monitor/restoration failure')
    protocol = load(root / 'protocol.json')
    expected_protocol = dict(scope='prefill_gathered_head_first_vec', mode='concurrent', layers=list(LAYERS),
                             rows=ROWS, keys=640, cases=3, providers=4, samples=9, reps=1, variants=[2, 4],
                             base_pos=0, c_start=2048, ratio=2, n_comp=2048, top_k=512)
    need(all(protocol.get(key) == value for key, value in expected_protocol.items()), 'Original full-working-set protocol')
    capture = load(directory / 'capture.json')
    need(capture == load(root / 'packed-capture/capture.json') and digest(directory / 'capture.json') ==
         manifest['capture_json_sha256'] == summary['capture_json_sha256'], 'Captured JSON binding differs')
    need(capture['capture_kind'] == 'myds_prefill_attention_multihead' and
         [case['layer'] for case in capture['cases']] == list(LAYERS), 'Three actual capture cases')
    original = load(root / 'packed-capture/original-inputs.json')
    need(digest(root / 'packed-capture/original-inputs.json') == capture['source']['original_inputs_sha256'],
         'Original observer descriptor identity')
    provider = original['actual_provider']
    provider_path = root / 'raw-capture/capture-provider.json'
    need(load(provider_path) == provider and digest(provider_path) == original['provider']['sha256'] and
         provider['ordinary_production_prefill'] is True and provider['head_first_attempts'] == provider['head_first_encoded'] >= 3 and
         provider['gather_calls'] >= provider['head_first_encoded'] and
         all(provider[key] == 0 for key in ('head_first_failures', 'head_first_cached_fallbacks', 'matrix_encoded', 'matrix_fallbacks')),
         'Actual head-first adoption/fallback evidence')
    for source_key, name in (('effective_environment_sha256', 'effective-env.json'),
                             ('request_sha256', 'request.json'), ('token_file_sha256', 'capture-tokens.txt')):
        need(capture['source'][source_key] == digest(root / name), 'Request/token/env source identity')
    need(len((root / 'capture-tokens.txt').read_text().split()) == 4096 and
         capture['source']['performance_from_capture'] is False and capture['source']['teacher_forced_tokens_used'] == 4096,
         'Actual 4096-token observer provenance')
    token_words = [int(value) for value in (root / 'capture-tokens.txt').read_text().split()]
    tokenized = (root / 'tokenized.tokens.i32le.bin').read_bytes()
    need(len(tokenized) % 4 == 0 and list(struct.unpack('<4096i', tokenized[:16384])) == token_words and
         len(tokenized) // 4 == capture['source']['rendered_tokens_available'],
         'Observer token IDs differ from the actual tokenizer output prefix')
    model = Path(summary['model']['path'])
    st = model.stat()
    need(dict(path=str(model.resolve()), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
              inode=st.st_ino, device=st.st_dev) == summary['model'] == capture['source']['model_metadata'], 'Actual model artifact stat')
    with model.open('rb') as stream:
        prefix_sha = hashlib.sha256(stream.read(1048576)).hexdigest()
    need(prefix_sha == summary['model_header_sha256'] == capture['model_header_sha256'] == manifest['model']['header_sha256'],
         'Actual model prefix SHA identity')
    sink_names = [f'blk.{layer}.attn_sinks.weight' for layer in LAYERS]
    sink_descriptors, data_offset = gguf_descriptors(model, sink_names)
    need(data_offset == manifest['model']['tensor_data_offset'], 'Independent GGUF tensor-data offset')
    cases = []
    suffixes = {'args200': 'args200', 'q_f32': 'q.f32', 'kv_unique_f16': 'kv-unique.f16',
                'kv_row_ids_u32': 'kv-rows.u32', 'mask_f16': 'mask-compact.f16', 'sinks_f32': 'sinks.f32'}
    for index, case in enumerate(capture['cases']):
        need(all(case[key] == value for key, value in dict(base_pos=0, c_start=2048, rows=ROWS,
                 keys=640, ratio=2, n_comp=2048, top_k=512).items()), 'Real second full-chunk geometry')
        logical = {key: directory / f'capture-case-{index:03d}-{suffix}' for key, suffix in suffixes.items()}
        for key, path in logical.items():
            need(path.name in names and path.stat().st_size == case[key]['bytes'] and digest(path) == case[key]['sha256'],
                 'Captured logical payload differs from packed schema')
        stride = low.validate_args(logical['args200'].read_bytes())
        low.finite_written(logical['q_f32'], OUTPUT_BYTES, written=False)
        low.finite_written(logical['sinks_f32'], 256, written=False)
        coverage = low.mask_coverage(logical['mask_f16'])
        need(coverage == manifest['cases'][index]['mask_coverage'] and coverage['minimum_valid_picks'] == 512 and
             coverage['minimum_valid_window'] == 128 and coverage['full_512_pick_queries'] == ROWS, 'Actual complete mask coverage')
        expanded = low.expanded_kv_sha(logical['kv_unique_f16'], logical['kv_row_ids_u32'])
        need(expanded == case['kv_expanded_sha256'] == manifest['cases'][index]['kv_expanded_sha256'], 'Full expanded KV SHA')
        source = next(row for row in original['payloads'] if row['layer'] == case['layer'])
        for source_key, case_key in (('args', 'args200'), ('q', 'q_f32'), ('mask', 'mask_f16'), ('sinks', 'sinks_f32')):
            raw_path = Path(source[source_key]['path'])
            need(raw_path.stat().st_size == source[source_key]['bytes'] and
                 digest(raw_path) == source[source_key]['sha256'] == case[case_key]['sha256'], 'Original observer raw payload identity')
        raw_kv = Path(source['kv']['path'])
        need(raw_kv.stat().st_size == ROWS * 655360 and digest(raw_kv) == expanded == source['kv']['sha256'],
             'Retained actual 1.34GB KV bytes differ from reconstructed dictionary')
        name = sink_names[index]
        tensor = sink_descriptors[name]
        need(tensor['type'] == 0 and tensor['dimensions'] == [64] and
             data_offset + tensor['offset'] == manifest['cases'][index]['sink_file_offset'], 'Independent model sink descriptor')
        with model.open('rb') as stream:
            stream.seek(data_offset + tensor['offset'])
            need(stream.read(256) == logical['sinks_f32'].read_bytes(), 'Actual GGUF sink bytes differ')
        cases.append(dict(case=index, layer=case['layer'], rows=ROWS, keys=640, original_mask_stride=stride,
                          q_bytes=OUTPUT_BYTES, expanded_kv_bytes=ROWS * 655360, expanded_kv_sha256=expanded,
                          mask_coverage=coverage, actual_gguf_sinks_verified=True))
    need(len(manifest['bindings']) == 4, 'Four explicitly prepared PSOs required')
    for number, binding in enumerate(manifest['bindings']):
        heads = (1, 1, 2, 4)[number]
        need(binding['provider'] == number and binding['symbol'] == low.SYMBOLS[number] and
             binding['library'] == ('production' if number == 0 else 'probe') and binding['head_batch'] == heads and
             binding['groups_x'] == 64 // heads and binding['threads'] == [32, heads, 1] and binding['args_bytes'] == 200 and
             binding['ordered_constants'] == low.FC and binding['dynamic_threadgroup_bytes'] == (3328, 3328, 23040, 29696)[number] and
             binding['max_threads'] >= 32 * heads, 'Actual PSO/FC/geometry identity')
    rows = [json.loads(line) for line in (root / 'component.stdout').read_text().splitlines() if line.strip()]
    need(not any(row.get('kind') in ('timing', 'summary', 'complete') for row in rows), 'No formal timing or completion may be inferred')
    parity = [row for row in rows if row.get('kind') == 'parity']
    need([(row['case'], row['provider']) for row in parity] == [(0, 0), (0, 1), (0, 2)] and
         all(row['completed_cb'] and row['rows'] == ROWS and row['keys'] == 640 and row['output_bytes'] == OUTPUT_BYTES for row in parity),
         'Only case0 native/sameprobe/H2 were dispatched before failure')
    expected_outputs = {f'parity-case-000-provider-{provider}-output.f32' for provider in (0, 1, 2)}
    need({name for name in names if name.startswith('parity-') and name.endswith('.f32')} == expected_outputs,
         'Unexpected output implies another provider/case was run')
    outputs = [directory / f'parity-case-000-provider-{provider}-output.f32' for provider in (0, 1, 2)]
    need(all(path.stat().st_size == OUTPUT_BYTES for path in outputs), 'Complete actual output files required')
    guards = [path.with_suffix('.guards').read_bytes() == b'\xa5' * 128 for path in outputs]
    same = file_stats(outputs[0], outputs[1])
    failed = file_stats(outputs[0], outputs[2])
    need(same['mismatches'] == 0 and same['first_mismatch'] is None and guards[0] and guards[1] and
         same['reference_nonfinite'] == same['candidate_nonfinite'] == same['reference_poison_words'] == same['candidate_poison_words'] == 0,
         'Frozen production and same-library reference must be completely bit-exact, finite, written and guarded')
    need(failed['mismatches'] == EXPECTED_MISMATCHES == parity[2]['mismatches'] and
         failed['first_mismatch'] == parity[2]['first_mismatch'] == 0 and parity[2]['full_f32_bit_exact'] is False and
         parity[0]['full_f32_bit_exact'] is True and parity[1]['full_f32_bit_exact'] is True,
         'Independent mismatch count/first index does not reproduce the actual first failure')
    archives = verify_archives(root, source_archive, artifacts_archive, manifest, capture)
    return dict(schema_version=1, reviewer='independent_CPU_failed_mode', evidence_valid=True,
                correctness_pass=False, performance_measured=False, performance_gate_pass=False,
                decision='reject_correctness', gpu_work_submitted=False, fullmodel_run=False, production_adoption=False,
                raw_logical_files_verified=len(names), raw_unique_inodes_hashed=inode_count,
                recorded_restoration_exact_including_nice=True, original_pid=before['pid'], restored_pid=restored['pid'],
                cases=cases, native_vs_sameprobe=same, native_vs_H2=failed, guards_by_provider=guards,
                H4='not_run; pipeline creation is not execution', formal_timing_rows=0,
                limits=['No performance inference from capture/host/parity wall time.',
                        'Recorded restoration is verified; this offline reader does not query the current service.',
                        'Model identity uses actual stat, first1MiB SHA, independent GGUF descriptors and sink bytes; no full GGUF digest.'],
                manifest_sha256=digest(directory / 'artifact-manifest.json'), component_stdout_sha256=digest(root / 'component.stdout'),
                reviewer_sha256=digest(Path(__file__)), low_helper_sha256=digest(Path(low.__file__)), **archives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sourcearchive', '--source-archive', dest='source_archive', type=Path, required=True)
    parser.add_argument('--artifactsarchive', '--artifacts-archive', dest='artifacts_archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = review(args.root, args.source_archive, args.artifacts_archive)
    except Exception as error:
        result = dict(schema_version=1, evidence_valid=False, correctness_pass=False,
                      performance_measured=False, decision='invalid_evidence', error=repr(error), gpu_work_submitted=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(evidence_valid=result['evidence_valid'], decision=result['decision'],
                          out=str(args.out), sha256=digest(args.out))))
    return 0 if result['evidence_valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
