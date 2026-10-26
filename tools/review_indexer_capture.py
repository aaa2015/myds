#!/usr/bin/env python3
"""CPU-only verification and replay manifest for a complete real indexer capture."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(value, label):
    if not value:
        raise ValueError(label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-dir', 'build-manifest', 'tokens-file', 'config', 'model', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    root = a.capture_dir.resolve(strict=True)
    check(not a.out.exists(), 'new review output')
    files = {}
    for f in sorted(root.iterdir()):
        check(f.is_file() and not f.is_symlink(), 'regular capture payload')
        data = f.read_bytes()
        files[f.name] = dict(path=f.name, bytes=len(data), sha256=sha(data))
    meta = json.loads((root / 'capture-meta.json').read_text())
    check(meta['capture_kind'] == 'myds_real_prefill_indexer' and meta['tokens'] == 32768 and
          meta['real_model_forward'] is True and meta['performance_from_capture'] is False,
          'actual forward metadata')
    build = json.loads(a.build_manifest.read_text())
    check(build['compile_only_pass'] is True and build['normal_preprocessor_observer_absent'] is True,
          'frozen compile provenance')
    check(sha(Path(build['binary']).read_bytes()) == build['binary_sha256'], 'capture binary identity')
    # dump_env writes descriptive comment/blank lines as well as KEY=VALUE.
    # Their historical priority prose is not an environment assignment.
    env_lines = [s for s in (root / 'effective-myds.env').read_text().splitlines()
                 if s.strip() and not s.lstrip(' \t').startswith('#')]
    check(env_lines and all(re.fullmatch(r'MYDS_[A-Za-z0-9_]+=.*', s) for s in env_lines),
          'MYDS-only environment assignments')
    effective = dict(s.split('=', 1) for s in env_lines)
    check(len(effective) == len(env_lines), 'unique effective MYDS keys')
    configured = {}
    for line in a.config.read_text().splitlines():
        if not line.strip() or line.lstrip(' \t').startswith('#'):
            continue
        check(len(line.encode()) < 4095 and '=' in line, 'bounded config assignment')
        key, value = line.rstrip('\r\n \t').split('=', 1)
        key, value = key.strip(' \t'), value.lstrip(' \t')
        if not key.startswith('MYDS_'):
            continue
        value = re.split(r'(?<=[ \t])#', value, maxsplit=1)[0].rstrip(' \t')
        check(re.fullmatch(r'MYDS_[A-Za-z0-9_]+', key), 'config MYDS key')
        configured[key] = value
    check(configured and all(effective.get(k) == v for k, v in configured.items()),
          'all frozen config values overwrite environment and match effective bytes')
    check(effective.get('MYDS_INDEXER_BATCH_FLOATS') == '2097152', 'production 2M scratch budget')
    records = [json.loads(s) for s in (root / 'capture-records.jsonl').read_text().splitlines() if s]
    selected_records = [json.loads(s) for s in (root / 'selected-records.jsonl').read_text().splitlines() if s]
    check(len(records) == 3 and [r['layer'] for r in records] == [2, 8, 14], 'three actual owner/source layers')
    check(len(selected_records) == 3 and [r['layer'] for r in selected_records] == [2, 8, 14],
          'three complete actual TopK consumers')
    selected_by_layer = {r['layer']: r for r in selected_records}
    cases = []
    for r in records:
        layer, rows, cols = r['layer'], r['n_tokens'], r['n_comp']
        check(r['group_off'] == 0 and r['base_pos'] + r['c_start'] == 30720 and
              r['pos0'] == 30720 and r['ratio'] == 2 and r['n_heads'] == 32 and r['head_dim'] == 128 and
              8 <= rows <= 2048 and 15000 <= cols <= 16384 and rows * cols <= 2097152 and
              r['full_scores'] is True and r['synthetic'] is False and r['key_window'] == 0,
              'actual subgroup shape and scope')
        prefix = f'layer-{layer}-'
        payload = {}
        for kind, size in [('args72', 72), ('native-args', r['native_args_bytes']),
                           ('q-f32', rows * 32 * 128 * 4), ('weights-f32', rows * 32 * 4),
                           ('cache', cols * 128 * (2 if r['cache_bf16'] else 4)),
                           ('scores-f32', rows * cols * 4), ('selected-i32', rows * 512 * 4)]:
            name = prefix + kind + '.bin'
            check(files[name]['bytes'] == size, 'complete payload ' + name)
            payload[kind] = files[name]
        expected = struct.pack('<6I5Qf4x', cols, rows, 32, 128, r['pos0'], 2,
                               16384, 512, 128, 128 * (2 if r['cache_bf16'] else 4), cols * 4, 1 / 64)
        check((root / payload['args72']['path']).read_bytes() == expected, 'canonical 72-byte argument ABI')
        scores = (root / payload['scores-f32']['path']).read_bytes()
        values = struct.unpack('<' + 'I' * (len(scores) // 4), scores)
        invisible = 0
        for t in range(rows):
            visible = min(cols, (r['pos0'] + t + 1) // 2)
            row = values[t * cols:(t + 1) * cols]
            check(all((b & 0x7f800000) != 0x7f800000 for b in row[:visible]), 'finite visible scores')
            check(all(b == 0xff800000 for b in row[visible:]), 'all causal tail scores are minus infinity')
            invisible += cols - visible
        for kind in ('q-f32', 'weights-f32'):
            raw = (root / payload[kind]['path']).read_bytes()
            check(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw)), 'finite complete ' + kind)
        provider = r['provider']
        check(provider in ('kernel_myds_indexer_score_chunk_exact', 'kernel_glm_indexer_score_chunk_direct',
                           'kernel_dsv4_indexer_scores_tiled_bf16', 'kernel_dsv4_indexer_scores_tiled_f32'),
              'known original production provider')
        check(r['native_args_bytes'] == (72 if 'tiled' in provider else 36), 'provider argument size')
        native_args = (root / payload['native-args']['path']).read_bytes()
        if 'tiled' in provider:
            check(native_args[:68] == expected[:68], 'actual native tiled arguments')
        else:
            native_expected = struct.pack('<4If4I', cols, 32, 128, 2 if r['cache_bf16'] else 0,
                                          1 / 64, rows, r['pos0'], 2, 0)
            check(native_args == native_expected, 'actual native exact/direct arguments')
        selected = selected_by_layer[layer]
        check(selected['n_tokens'] == rows and selected['n_comp'] == cols and selected['pos0'] == r['pos0'] and
              selected['ratio'] == 2 and selected['top'] == 512 and selected['ordered_ids_complete'] is True and
              selected['mask_filter_between_scores_and_topk'] is False and selected['dispatches'],
              'actual ordered causal Top512 provenance')
        for call in selected['dispatches']:
            check(isinstance(call['kernel'], str) and call['kernel'].startswith('kernel_') and
                  len(call['grid']) == 3 and len(call['threads']) == 3 and call['host_accepted'] is True,
                  'actual successful TopK dispatch witness')
        ids_raw = (root / payload['selected-i32']['path']).read_bytes()
        ids = struct.unpack('<' + 'i' * (rows * 512), ids_raw)
        for t in range(rows):
            visible = min(cols, (r['pos0'] + t + 1) // 2)
            row_ids = ids[t * 512:(t + 1) * 512]
            check(len(set(row_ids)) == 512 and all(0 <= k < visible for k in row_ids),
                  'all ordered TopK IDs distinct and causally valid')
            chosen = [struct.unpack_from('<f', scores, (t * cols + k) * 4)[0] for k in row_ids]
            check(all(a >= b for a, b in zip(chosen, chosen[1:])), 'actual selected scores in descending order')
        cases.append(dict(**r, payloads=payload, invisible_minus_inf_count=invisible,
                          actual_topk_consumer=selected, candidate_topk_replayed=False))
    tokens = []
    for line in a.tokens_file.read_text().splitlines():
        for word in line.split('#', 1)[0].split():
            check(word.isascii() and word.isdigit(), 'exact-token file syntax')
            token = int(word)
            check(0 <= token <= 0x7fffffff, 'token integer bounds')
            tokens.append(token)
    check(len(tokens) == 32768 and struct.pack('<32768i', *tokens) ==
          (root / 'input-token-ids.i32').read_bytes(), 'capture consumed exact supplied token IDs')
    stat = a.model.stat()
    check(stat.st_size == meta['model_size'] and stat.st_ino == meta['model_inode'] and
          stat.st_mtime_ns == meta['model_mtime_s'] * 1000000000 + meta['model_mtime_ns'], 'model stat unchanged')
    with a.model.open('rb') as f:
        header = f.read(1048576)
    loaded = Path(meta['loaded_metallib_path'])
    check(loaded.is_absolute() and loaded.is_file(), 'actual loaded metallib path witness')
    result = dict(schema_version=1, capture_kind='myds_real_prefill_indexer', status='verified_cpu_only',
                  capture_directory=str(root), files=files, cases=cases, metadata=meta,
                  effective_myds=effective, environment_count=len(effective),
                  config_values_matched=True, config_declared_unique_count=len(configured),
                  effective_extra_myds_keys=sorted(set(effective) - set(configured)),
                  build_manifest_path=str(a.build_manifest.resolve()), build_manifest_sha256=sha(a.build_manifest.read_bytes()),
                  capture_binary_sha256=build['binary_sha256'],
                  loaded_metallib_path=str(loaded), loaded_metallib_sha256=sha(loaded.read_bytes()),
                  tokens_file_sha256=sha(a.tokens_file.read_bytes()), config_sha256=sha(a.config.read_bytes()),
                  model_path=str(a.model.resolve()), model_header_1MiB_sha256=sha(header),
                  whole_model_payload_sha256=False, replay_gpu_submitted=False, performance_from_capture=False)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(a.out), layers=[r['layer'] for r in cases],
                         actual_tuples=[[r['n_tokens'], r['n_comp'], r['pos0'], r['ratio']] for r in cases],
                         cpu_only_pass=True, gpu_replay_unrun=True), indent=2))


if __name__ == '__main__':
    main()
