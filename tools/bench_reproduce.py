#!/usr/bin/env python3
"""Verify saved evidence or run the greedy full-model harness on an idle Mac.

This tool does not stop, deploy, or restart services. It clears inherited MYDS_
settings, captures stdout/stderr separately, and audits the completed samples.
The historical fleet window scripts remain unchanged in docs/benchmarks.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/benchmarks/2026-10-05-optimization'
DEFAULT_ENV = {
    'MYDS_PREFIX_CACHE': '0', 'MYDS_KV_DSK': '0', 'MYDS_PHASE_PROFILE': '0',
    'MYDS_PREFILL_CHUNK_SIZE': '512', 'MYDS_MOE_CHUNKS': '9',
    'MYDS_SUFFIX_SHRINK_SHIFT': '512', 'MYDS_RESIDENCY_KEEP_ALIVE_S': '86400',
}
ARM_ENV = {
    'target': {'MYDS_PLD': '0', 'MYDS_BATCH2_SPEC': '0', 'MYDS_SPEC_SERIAL_PROBE': '0'},
    'pld': {'MYDS_PLD': '1', 'MYDS_BATCH2_SPEC': '0', 'MYDS_SPEC_SERIAL_PROBE': '1'},
    'pld-batch2': {'MYDS_PLD': '1', 'MYDS_BATCH2_SPEC': '1', 'MYDS_SPEC_SERIAL_PROBE': '0'},
    'dspark': {'MYDS_PLD': '0', 'MYDS_BATCH2_SPEC': '0', 'MYDS_SPEC_SERIAL_PROBE': '1'},
    'batch2': {'MYDS_PLD': '0', 'MYDS_BATCH2_SPEC': '1', 'MYDS_SPEC_SERIAL_PROBE': '0'},
    'hybrid': {'MYDS_PLD': '1', 'MYDS_BATCH2_SPEC': '1', 'MYDS_SPEC_SERIAL_PROBE': '0'},
}
DSPARK_ARMS = {'dspark', 'batch2', 'hybrid'}
OUTPUT_CAPS = (32, 64, 256, 512)
DECODE_COUNTERS = ('target_forward_calls', 'target_rows', 'batch2_calls', 'batch2_hits',
                  'batch2_misses', 'batch2_failures', 'batch2_fallbacks', 'draft_cycles',
                  'draft_proposed', 'draft_accepted', 'draft_rejects', 'confidence_rejects',
                  'margin_rejects', 'backoff_steps', 'pld_queries', 'pld_cycles',
                  'pld_proposed', 'pld_accepted', 'output_tokens', 'cycle_count')
DECODE_TIMERS = ('draft_ms', 'verify_ms', 'rollback_ms', 'snapshot_ms', 'pld_ms')
INDEXER_COMPACT_COUNTERS = ('requests', 'encoded', 'unavailable', 'dense_rows', 'scored_row_slots')
TOPK_PREFIX_COUNTERS = ('requests', 'encoded', 'unavailable', 'classified_rows', 'merge_dispatches',
                       'dense_write_slots', 'prefix_write_slots', 'decode_requests',
                       'decode_encoded', 'decode_unavailable')
TOPK_THRESHOLD_FIELDS = ('requests', 'encoded', 'errors', 'validated', 'adopted', 'fallback',
                         'nan', 'inf', 'insufficient', 'capacity', 'certificate', 'tie')
TOPK_THRESHOLD_COUNTERS = TOPK_THRESHOLD_FIELDS + tuple('decode_' + k for k in TOPK_THRESHOLD_FIELDS)
# Exact ordinary-target shapes; source providers exclude prefill and private verification.
DECODE_SHAPE_CANDIDATES = {
    'gpu_moe_down_overlap': ('MYDS_MOE_DOWN_OVERLAP', 'moe_down_overlap_mode', 40, 'require_moe_down_overlap'),
    'gpu_moe_iq2_v41': ('MYDS_MOE_IQ2_V41', 'moe_iq2_v41_mode', 40, 'require_moe_iq2_v41'),
    'gpu_dense_q4_v41': ('MYDS_DENSE_Q4_V41', 'dense_q4_v41_mode', 81, 'require_dense_q4_v41'),
    'gpu_dense_q4_v41_nb': ('MYDS_DENSE_Q4_V41_NB', 'dense_q4_v41_nb_mode', 80, 'require_dense_q4_v41_nb'),
    'gpu_dense_q4_v41_head_reference': ('MYDS_DENSE_Q4_V41_HEAD_REFERENCE', 'dense_q4_v41_head_reference_mode', 1, 'require_dense_q4_v41_head_reference'),
    'gpu_attention_rope_fuse': ('MYDS_ATTN_ROPE_FUSE', 'attention_rope_fuse_mode', 40, 'require_attention_rope_fuse'),
    'gpu_moe_hc_overlap': ('MYDS_MOE_HC_OVERLAP', 'moe_hc_overlap_mode', 40, 'require_moe_hc_overlap'),
    'gpu_attn_hc_overlap': ('MYDS_ATTN_HC_OVERLAP', 'attn_hc_overlap_mode', 40, 'require_attn_hc_overlap'),
    'gpu_indexer_decode_overlap': ('MYDS_INDEXER_DECODE_OVERLAP', 'indexer_decode_overlap_mode', 8, 'require_indexer_decode_overlap'),
    'gpu_attn_kv_reuse': ('MYDS_ATTN_KV_REUSE', 'attn_kv_reuse_mode', 40, 'require_attn_kv_reuse'),
    'gpu_ffn_hc_overlap': ('MYDS_FFN_HC_OVERLAP', 'ffn_hc_overlap_mode', 40, 'require_ffn_hc_overlap'),
    'gpu_shared_q4_v41_nb': ('MYDS_SHARED_Q4_V41_NB', 'shared_q4_v41_nb_mode', 80, 'require_shared_q4_v41_nb'),
    'gpu_attention_direct_pack_fuse': ('MYDS_ATTN_DIRECT_PACK_FUSE', 'attention_direct_pack_fuse_mode', 40, 'require_attention_direct_pack_fuse'),
}
CANDIDATE_PATHS = {
    'gpu_moe_down_overlap': ('MYDS_MOE_DOWN_OVERLAP', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_moe_iq2_v41': ('MYDS_MOE_IQ2_V41', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_dense_q4_v41': ('MYDS_DENSE_Q4_V41', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),

    'gpu_moe_q2_v41': ('MYDS_MOE_Q2_V41', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_attention_packed_register': ('MYDS_ATTN_PACKED_REGISTER', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures',
         'direct_encoded', 'staged_encoded')),
    'gpu_route_simd_shuffle': ('MYDS_ROUTE_SIMD_SHUFFLE', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_f16_gemv_bf16_fuse': ('MYDS_F16_GEMV_BF16_FUSE', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_indexer_rope_quant_fuse': ('MYDS_INDEXER_ROPE_QUANT_FUSE', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_dspark_capture': ('MYDS_DSPARK_CAPTURE_BYPASS', 'failures',
        tuple(k for f in ('requests','encoded','bypassed','failures','stale_rejections') for k in (f,'decode_'+f))),
    'gpu_topk_nan_shuffle': ('MYDS_INDEXER_TOPK_NAN_SHUFFLE', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_topk_threshold': ('MYDS_INDEXER_TOPK_THRESHOLD', 'errors', TOPK_THRESHOLD_COUNTERS),
    'gpu_indexer_head_slots': ('MYDS_INDEXER_HEAD_SLOTS', 'failures',
        ('requests', 'encoded', 'unavailable', 'failures', 'decode_requests', 'decode_encoded',
         'decode_unavailable', 'decode_failures')),
    'gpu_topk_nan_fuse': ('MYDS_INDEXER_TOPK_NAN_FUSE', 'unavailable',
        ('requests', 'encoded', 'unavailable', 'classified_partitions',
         'decode_requests', 'decode_encoded', 'decode_unavailable')),
    'gpu_batch2_q8_shared': ('MYDS_BATCH2_Q8_SHARED', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_lm_head_q8': ('MYDS_LM_HEAD_Q8_NR0', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_engram_project_round': ('MYDS_ENGRAM_PROJECT_ROUND_SKIP', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_moe_q2_sum_cache': ('MYDS_MOE_Q2_SUM_CACHE', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
    'gpu_moe_q2_sum_cache_fused': ('MYDS_MOE_Q2_SUM_CACHE_FUSED', 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')),
}
# Independent next-round ordinary-scalar candidates share the strict six-field contract.
for _name, (_flag, _mode, _count, _option) in DECODE_SHAPE_CANDIDATES.items():
    CANDIDATE_PATHS.setdefault(_name, (_flag, 'failures',
        ('requests', 'encoded', 'failures', 'decode_requests', 'decode_encoded', 'decode_failures')))
CANDIDATE_ENABLED_VALUES = {'gpu_lm_head_q8': '4'}
FORBIDDEN_ENV_PREFIXES = ('MYDS_PROFILE_', 'MYDS_SKIP_', 'MYDS_TRACE_', 'MYDS_BATCH_TRACE_')
FORBIDDEN_ENV_NAMES = {'MYDS_MOE_STEP_PROF', 'MYDS_PHASE_PROFILE_FILE', 'MYDS_DISABLE_V41_PREFILL',
                       'MYDS_PROF', 'MYDS_DISPATCH_HIST', 'MYDS_DEBUG_DISPATCH', 'MYDS_KPROF'}
GPU_PROCESSES = {'myds', 'llama-server', 'bench_phase', 'bench_qkv',
                 'bench_attention_group', 'bench_attention_vec',
                 'bench_decode_iq2', 'moe_prefill_replay'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path, supplied_digest=None):
    identity = {'path': str(path), 'bytes': path.stat().st_size,
                'mtime_ns': path.stat().st_mtime_ns,
                'identity_note': 'File metadata only; the complete GGUF was not SHA-hashed'}
    if supplied_digest is not None:
        if len(supplied_digest) != 64 or any(c not in '0123456789abcdef' for c in supplied_digest):
            raise ValueError('Model SHA256 must contain exactly 64 lowercase hexadecimal characters')
        identity.update(artifact_sha256=supplied_digest,
                        identity_note='Caller-supplied artifact SHA256; runner did not rehash model bytes')
    return identity


def source_state():
    """Bind the checkout and dirty changes without printing source or secrets."""
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                          capture_output=True, timeout=15)
    diff = subprocess.run(['git', '-C', str(ROOT), 'diff', '--binary', 'HEAD', '--', '.'],
                          capture_output=True, timeout=15)
    status = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain=v1', '-z', '--untracked-files=all'],
                            capture_output=True, timeout=15)
    if head.returncode or diff.returncode or status.returncode:
        raise RuntimeError('Could not bind benchmark source checkout identity')
    # Untracked file contents are bound too; paths are NUL-delimited, preserving spaces.
    untracked = {}
    for entry in status.stdout.split(b'\0'):
        if entry.startswith(b'?? '):
            relative = os.fsdecode(entry[3:])
            path = ROOT / relative
            if path.is_file():
                untracked[relative] = sha(path)
    return {'head': head.stdout.decode().strip(), 'dirty': bool(status.stdout),
            'diff_sha256': hashlib.sha256(diff.stdout).hexdigest(),
            'status_sha256': hashlib.sha256(status.stdout).hexdigest(),
            'untracked_sha256': untracked}


def fnv_bytes(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f'{value:016x}'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require_uninstrumented_environment(environment):
    # Several native diagnostics test getenv presence: string "0" and the empty
    # string still enable them. Apply this gate to requested and actual metadata.
    for name, value in environment.items():
        if value is not None and (name.startswith(FORBIDDEN_ENV_PREFIXES) or name in FORBIDDEN_ENV_NAMES):
            raise ValueError('Profiling/path-skipping setting is incompatible with this run: ' + name)


def verify(args):
    """校验已提交证据的 SHA。

    2026-10-25: 证据树（docs/benchmarks/…）未随源码仓库发布（见账本卷首与 §97.6），
    所以旧实现在 fresh clone 上不是“报告缺失”，而是直接 FileNotFoundError 崩溃。
    “没东西可校验” 与 “校验失败” 是两件不同的事：前者应当明确 SKIP 并退出 0，
    后者必须退出非 0。这里的区分很重要 —— 把缺失当失败会让 CI 永远红，
    把失败当缺失会让损坏悄悄流过去。
    """
    errors = []
    checked = 0
    missing_dirs = []
    for relative in ('2026-10-05-phase', '2026-10-05-optimization'):
        directory = ROOT / 'docs/benchmarks' / relative
        if not directory.is_dir():
            missing_dirs.append(relative)
            continue
        spec = 'files.sha256.json' if relative.endswith('optimization') else 'manifest.json'
        spec_path = directory / spec
        if not spec_path.is_file():
            errors.append(f'{relative}: 目录存在但缺少 {spec}')
            continue
        try:
            raw = json.loads(spec_path.read_text())
            entries = raw if relative.endswith('optimization') else raw['sha256']
        except Exception as exc:                       # noqa: BLE001
            errors.append(f'{relative}: 无法解析 {spec}: {exc}')
            continue
        for name, expected in entries.items():
            path = directory / name
            digest = expected['sha256'] if isinstance(expected, dict) else expected
            if not path.is_file() or sha(path) != digest:
                errors.append(str(path))
            elif isinstance(expected, dict) and path.stat().st_size != expected['bytes']:
                errors.append(str(path) + ': size')
            checked += 1

    if args.check_source:
        manifest_path = EVIDENCE / 'manifest.json'
        if not manifest_path.is_file():
            missing_dirs.append('evidence/manifest.json')
        else:
            try:
                manifest = json.loads(manifest_path.read_text())
                for name, digest in manifest['production_source_sha256'].items():
                    path = ROOT / name
                    if not path.is_file() or sha(path) != digest:
                        errors.append('source: ' + name)
                    checked += 1
            except Exception as exc:                   # noqa: BLE001
                errors.append(f'无法解析 evidence/manifest.json: {exc}')

    skipped = bool(missing_dirs) and checked == 0 and not errors
    out = {
        'verified': not errors,
        'files_checked': checked,
        'errors': errors,
    }
    if missing_dirs:
        out['skipped'] = True
        out['missing'] = missing_dirs
        out['reason'] = ('证据归档未随源码仓库发布；这里只做“有则校验”，'
                         '无则明确跳过。复现需用本仓库代码重跑基准，'
                         '不能仅凭本仓库数据验证历史数值。')
        del skipped
    print(json.dumps(out, ensure_ascii=False))
    return int(bool(errors))


def command_output(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def host_state():
    return {'platform': platform.platform(),
            'cpu': command_output(['sysctl', '-n', 'machdep.cpu.brand_string']),
            'memory_bytes': int(command_output(['sysctl', '-n', 'hw.memsize'])),
            'os': command_output(['sw_vers']),
            'swap': command_output(['sysctl', 'vm.swapusage']),
            'vm_stat': command_output(['vm_stat'])}


def check_idle(port):
    active = []
    for line in command_output(['ps', '-axo', 'pid=,comm=']).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and int(fields[0]) != os.getpid():
            if Path(fields[1]).name in GPU_PROCESSES:
                active.append(line.strip())
    if active:
        raise RuntimeError('Other inference/benchmark processes are running: ' + '; '.join(active))
    result = subprocess.run(['lsof', '-nP', '-t', '-a', f'-iTCP:{port}', '-sTCP:LISTEN'],
                            text=True, capture_output=True, timeout=15)
    if result.stdout.strip():
        raise RuntimeError(f'Port {port} has a listener; arrange an exclusive window before running')
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError('Could not check the server listener: ' + result.stderr.strip())


def load_audit():
    path = ROOT / 'tools/bench_phase_audit.py'
    spec = importlib.util.spec_from_file_location('myds_saved_phase_audit', path)
    module = importlib.util.module_from_spec(spec)
    # Evidence directories contain versioned raw bytes, not generated Python caches.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def validate_run(directory, state, audit):
    group, samples = audit.validate_group(directory, 'run', {'environment': state['environment']})
    metadata = group['metadata']
    validate_indexer_rope_quant_mode(state['environment'])
    validate_f16_fuse_mode(state['environment'])
    validate_indexer_rope_quant_mode(metadata['environment'])
    validate_f16_fuse_mode(metadata['environment'])
    validate_capture_modes(state['environment'])
    validate_capture_modes(metadata['environment'])
    if state['environment'].get('MYDS_INDEXER_ROPE_QUANT_FUSE') is not None:
        audit.require(metadata.get('protocol_version') == 2, 'Indexer RoPE quant fuse requires native protocol2')
    if state['environment'].get('MYDS_F16_GEMV_BF16_FUSE') is not None:
        audit.require(metadata.get('protocol_version') == 2, 'F16 BF16 fuse requires native protocol2')
    if state['environment'].get('MYDS_ENGINE_CAPTURE_BYPASS') is not None:
        audit.require(metadata.get('protocol_version') == 2, 'Engine capture bypass requires native protocol2')
    require_uninstrumented_environment(metadata['environment'])
    audit.require(metadata.get('protocol_version', 1) in (1, 2), 'Unsupported benchmark protocol version')
    fixed = {'temperature': 0, 'top_p': 1, 'top_k': 0, 'min_p': 0,
             'repetition_penalty': 1, 'seed': 42, 'warmup_runs': 1,
             'clock': 'CLOCK_MONOTONIC', 'prompt_renderer': 'parse_chat_prompt',
             'transport': 'direct_byte_callback', 'qos': 'user_interactive', 'qos_result': 0}
    for key, expected in fixed.items():
        audit.require(metadata.get(key) == expected, 'Fixed protocol differs: ' + key)
    audit.require(type(metadata.get('repeats')) is int and metadata['repeats'] in (3, 5),
                  'Fixed repeat protocol must be three or five formal samples')
    argv = state['argv']
    audit.require(isinstance(argv, list) and len(argv) in (6, 7), 'Invalid recorded harness argv')
    expected = {'model': argv[1], 'request_file': argv[2], 'repeats': int(argv[3]),
                'max_new_tokens': int(argv[4]), 'max_seq_len': int(argv[5]),
                'request_bytes': state['request_bytes'], 'request_fnv1a64': state['request_fnv1a64']}
    for key, value in expected.items():
        audit.require(metadata.get(key) == value, 'Metrics/input binding differs: ' + key)
    audit.require(state['model']['path'] == metadata['model'], 'Recorded model identity/path differs')
    audit.require(metadata['max_new_tokens'] in OUTPUT_CAPS and 0 < metadata['max_seq_len'] <= 0xffffffff,
                  'Recorded output/context range differs')
    if metadata.get('protocol_version') == 2:
        support = state.get('dspark_model')
        support_path = argv[6] if len(argv) == 7 else None
        audit.require(metadata.get('dspark_model') == support_path, 'DSpark model argv/metadata differs')
        audit.require((support['path'] if support else None) == support_path, 'DSpark identity/path differs')
        records = audit.read_records(directory / 'run-metrics.jsonl')
        loaded = next(record for record in records if record['kind'] == 'load')
        for mode,flag in (('topk_nan_shuffle_mode','MYDS_INDEXER_TOPK_NAN_SHUFFLE'),):
            if state['environment'].get(flag)=='1' or mode in loaded:
                actual=loaded.get(mode)
                audit.require(type(actual) is int and actual in (0,1) and
                    actual==int(state['environment'].get(flag)=='1'), 'Actual candidate getter differs: '+mode)
                if actual==1:
                    audit.require(loaded.get('native_v41') is True,'Candidate requires native V4.1')
        for mode, flag in (('route_simd_shuffle_mode', 'MYDS_ROUTE_SIMD_SHUFFLE'),
                           ('attention_packed_register_mode', 'MYDS_ATTN_PACKED_REGISTER'),
                           ('moe_q2_v41_mode', 'MYDS_MOE_Q2_V41')):
            if state['environment'].get(flag) == '1' or mode in loaded:
                actual = loaded.get(mode)
                audit.require(type(actual) is int and actual in (0, 1) and
                              actual == int(state['environment'].get(flag) == '1'),
                              'Actual candidate getter differs: ' + mode)
            if state['environment'].get(flag) == '1':
                audit.require(loaded.get('native_v41') is True and loaded.get('layers') == 40,
                              'Full decode candidate requires the native 40-layer V4.1 model')
        if any(state['environment'].get(values[0]) == '1' for values in DECODE_SHAPE_CANDIDATES.values()):
            actual_library = loaded.get('actual_metallib_path')
            audit.require(type(actual_library) is str and bool(actual_library),
                          'Required shape candidate lacks the actual loaded metallib path')
            expected_library = Path(argv[0]).parent / 'myds.metallib'
            audit.require(Path(actual_library).resolve() == expected_library.resolve(),
                          'Actual candidate loaded metallib path differs from the recorded build')
        for name, (flag, mode, _, _) in DECODE_SHAPE_CANDIDATES.items():
            if state['environment'].get(flag) == '1' or mode in loaded:
                actual = loaded.get(mode)
                audit.require(type(actual) is int and actual in (0, 1) and
                              actual == int(state['environment'].get(flag) == '1'),
                              'Actual candidate getter differs: ' + mode)
            if state['environment'].get(flag) == '1':
                audit.require(loaded.get('native_v41') is True and loaded.get('layers') == 40 and
                              loaded.get('dspark_loaded') is False,
                              'Shape candidate requires ordinary native 40-layer V4.1 without support')
        for sample in samples:
            sample['candidate_modes'] = {name: loaded.get(values[1]) for name, values in DECODE_SHAPE_CANDIDATES.items()}
            sample['candidate_layer_count'] = loaded.get('layers')
        if state['environment'].get('MYDS_F16_GEMV_BF16_FUSE') is not None or 'f16_gemv_bf16_fuse_mode' in loaded:
            actual = loaded.get('f16_gemv_bf16_fuse_mode')
            audit.require(type(actual) is int and actual in (0, 1) and
                          actual == int(state['environment'].get('MYDS_F16_GEMV_BF16_FUSE') == '1'),
                          'Actual candidate getter differs: f16_gemv_bf16_fuse_mode')
            if actual:
                audit.require(loaded.get('native_v41') is True, 'F16 BF16 fuse requires native V4.1')
        if state['environment'].get('MYDS_INDEXER_ROPE_QUANT_FUSE') is not None or 'indexer_rope_quant_fuse_mode' in loaded:
            actual = loaded.get('indexer_rope_quant_fuse_mode')
            audit.require(type(actual) is int and actual in (0, 1) and
                          actual == int(state['environment'].get('MYDS_INDEXER_ROPE_QUANT_FUSE') == '1'),
                          'Actual candidate getter differs: indexer_rope_quant_fuse_mode')
            if actual:
                audit.require(loaded.get('native_v41') is True and loaded.get('dspark_loaded') is False and support_path is None,
                              'Indexer RoPE quant fuse requires pure ordinary native V4.1 target')
        effective_capture = capture_bypass_requested(state['environment'])
        engine_capture = state['environment'].get('MYDS_ENGINE_CAPTURE_BYPASS') == '1'
        if effective_capture or 'dspark_capture_bypass_mode' in loaded:
            actual = loaded.get('dspark_capture_bypass_mode')
            audit.require(type(actual) is int and actual in (0, 1) and actual == int(effective_capture),
                          'Actual candidate getter differs: dspark_capture_bypass_mode')
            if actual:
                audit.require(loaded.get('native_v41') is True and support_path is not None,
                              'Capture bypass requires native V4.1 and loaded DSpark support')
        # Historical records predate this field. Explicit new-mode cohorts
        # require it even for A=0, while old archives remain unchanged.
        if state['environment'].get('MYDS_ENGINE_CAPTURE_BYPASS') is not None or 'engine_capture_bypass_mode' in loaded:
            actual = loaded.get('engine_capture_bypass_mode')
            audit.require(type(actual) is int and actual in (0, 1) and actual == int(engine_capture),
                          'Actual candidate getter differs: engine_capture_bypass_mode')
        if state['environment'].get('MYDS_INDEXER_TOPK_THRESHOLD') == '1' or 'topk_threshold_mode' in loaded:
            actual = loaded.get('topk_threshold_mode')
            audit.require(type(actual) is int and actual in (0, 1) and
                          actual == int(state['environment'].get('MYDS_INDEXER_TOPK_THRESHOLD') == '1'),
                          'Actual candidate context getter differs: topk_threshold_mode')
            if actual == 1:
                audit.require(loaded.get('native_v41') is True,
                              'TopK threshold candidate requires native V4.1 shape')
        if state['environment'].get('MYDS_INDEXER_HEAD_SLOTS') == '1' or 'indexer_head_slots_mode' in loaded:
            actual = loaded.get('indexer_head_slots_mode')
            audit.require(type(actual) is int and actual in (0, 1) and
                          actual == int(state['environment'].get('MYDS_INDEXER_HEAD_SLOTS') == '1'),
                          'Actual candidate context getter differs: indexer_head_slots_mode')
            if actual == 1:
                audit.require(loaded.get('native_v41') is True,
                              'Indexer head slots candidate requires native V4.1 shape')
        audit.require(loaded.get('dspark_loaded') is (support_path is not None),
                      'Requested DSpark support model was not loaded')
        if state['environment'].get('MYDS_MOE_Q2_SUM_CACHE_FUSED') == '1' or \
                'moe_q2_sum_cache_fused_mode' in loaded:
            for key, flag in (('moe_q2_sum_cache_mode', 'MYDS_MOE_Q2_SUM_CACHE'),
                              ('moe_q2_sum_cache_fused_mode', 'MYDS_MOE_Q2_SUM_CACHE_FUSED')):
                actual = loaded.get(key)
                audit.require(type(actual) is int and actual in (0, 1) and
                              actual == int(state['environment'].get(flag) == '1'),
                              'Actual candidate context getter differs: ' + key)
        for key, value in state['environment'].items():
            audit.require(metadata['environment'].get(key) == value, 'Unrecorded/different environment: ' + key)
        for sample in samples:
            logits = sample.get('final_logits')
            audit.require(isinstance(logits, dict) and logits.get('finite') is True,
                          'Missing/nonfinite final logits gate')
            audit.require(logits.get('elements') == loaded['vocab_size'], 'Final logits element count differs')
            digest = logits.get('fnv1a64_bytes')
            audit.require(isinstance(digest, str) and len(digest) == 16 and
                          all(c in '0123456789abcdef' for c in digest), 'Invalid final logits hash')
            audit.require(logits == samples[0]['final_logits'], 'Final logits differ from warmup')
            telemetry = sample.get('decode_telemetry')
            audit.require(isinstance(telemetry, dict), 'Missing per-request decode telemetry')
            for key in (*DECODE_COUNTERS, 'recorded_cycles'):
                value = telemetry.get(key)
                audit.require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                              'Invalid decode telemetry counter: ' + key)
            # Optional for older frozen archives; a new record must bind all six.
            depth2_keys = ('depth2_calls', 'depth2_hits1', 'depth2_hits2',
                           'depth2_misses', 'depth2_failures', 'depth2_fallbacks')
            if any(key in telemetry for key in depth2_keys):
                for key in depth2_keys:
                    audit.require(type(telemetry.get(key)) is int and telemetry[key] >= 0,
                                  'Invalid depth2 counter: ' + key)
                audit.require(sum(telemetry[key] for key in depth2_keys[1:5]) <= telemetry['depth2_calls'],
                              'Depth2 outcomes exceed calls')
                audit.require(telemetry['depth2_calls'] <= telemetry['batch2_calls'] and
                              telemetry['depth2_failures'] <= telemetry['batch2_failures'] and
                              telemetry['depth2_fallbacks'] <= telemetry['batch2_fallbacks'],
                              'Depth2 counters exceed compatibility totals')
            audit.require(telemetry['output_tokens'] == sample['output_tokens'], 'Decode telemetry output count differs')
            audit.require(isinstance(telemetry.get('timing_enabled'), bool), 'Missing timing-enabled marker')
            if state['environment'].get('MYDS_DECODE_METRICS') == '1':
                audit.require(telemetry['timing_enabled'], 'Requested decode timing was not enabled')
            audit.require(telemetry['recorded_cycles'] <= telemetry['cycle_count'], 'Too many recorded cycles')
            for key in DECODE_TIMERS:
                if telemetry['timing_enabled']:
                    audit.numeric(telemetry.get(key), key, positive=False)
                else:
                    audit.require(telemetry.get(key) is None, 'Disabled timer must be null: ' + key)
            for key in ('cycle_ms', 'cycle_output_tokens'):
                values = telemetry.get(key)
                audit.require(isinstance(values, list) and len(values) == telemetry['recorded_cycles'],
                              'Invalid recorded cycle array: ' + key)
            for duration in telemetry['cycle_ms']:
                audit.numeric(duration, 'cycle duration', positive=False)
            for count in telemetry['cycle_output_tokens']:
                audit.require(isinstance(count, int) and not isinstance(count, bool) and
                              0 <= count <= sample['output_tokens'],
                              'Invalid recorded cycle output count')
            audit.require(sum(telemetry['cycle_output_tokens']) <= telemetry['output_tokens'],
                          'Cycle outputs exceed total output tokens')
            audit.require(telemetry['batch2_hits'] + telemetry['batch2_misses'] <= telemetry['batch2_calls'],
                          'Batch2 result counters exceed calls')
            audit.require(telemetry['draft_accepted'] <= telemetry['draft_proposed'], 'Draft accepted exceeds proposed')
            audit.require(telemetry['pld_accepted'] <= telemetry['pld_proposed'], 'PLD accepted exceeds proposed')
    elif len(argv) == 7 or state.get('dspark_model'):
        audit.require(False, 'Historical protocol cannot establish DSpark execution')
    if state.get('arm') in ('batch2', 'pld-batch2', 'hybrid'):
        require_batch2_graph(samples)
    for sample in samples:
        if 'gpu_indexer_compact' not in sample:
            continue  # Earlier harnesses did not expose this path counter API.
        compact = sample['gpu_indexer_compact']
        audit.require(isinstance(compact, dict), 'Invalid compact indexer path counters')
        for key in INDEXER_COMPACT_COUNTERS:
            value = compact.get(key)
            audit.require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                          'Invalid compact indexer counter: ' + key)
        audit.require(compact['encoded'] + compact['unavailable'] <= compact['requests'],
                      'Compact indexer results exceed requests')
    for sample in samples:
        if 'gpu_topk_prefix' not in sample:
            continue  # Historical harnesses did not expose this path.
        counters = sample['gpu_topk_prefix']
        audit.require(isinstance(counters, dict), 'Invalid topk prefix path counters')
        for key in TOPK_PREFIX_COUNTERS:
            value = counters.get(key)
            audit.require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                          'Invalid topk prefix counter: ' + key)
        audit.require(counters['encoded'] + counters['unavailable'] == counters['requests'],
                      'Topk prefix results differ from requests')
        audit.require(counters['prefix_write_slots'] <= counters['dense_write_slots'],
                      'Topk prefix planned write slots exceed dense plan')
        audit.require(counters['decode_encoded'] + counters['decode_unavailable'] == counters['decode_requests'],
                      'Topk prefix decode results differ from requests')
        for key in ('requests', 'encoded', 'unavailable'):
            audit.require(counters['decode_' + key] <= counters[key],
                          'Topk prefix decode counters exceed request total')
    validate_candidate_paths(samples, state['environment'], audit)
    for name in CANDIDATE_PATHS:
        group[name] = candidate_paths(samples, name)
    group['callback_latency_ms'] = distribution(
        [value for sample in samples if not sample['warmup'] for value in sample['callback_intervals_ms']])
    group['decode_telemetry'] = telemetry_summary(
        samples, state['environment'].get('MYDS_DECODE_METRICS') == '1')
    group['gpu_indexer_compact'] = indexer_compact_paths(samples)
    group['gpu_topk_prefix'] = topk_prefix_paths(samples)
    return group, samples


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {'count': len(ordered), 'median': statistics.median(ordered),
            'p95': ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
            'p99': ordered[max(0, math.ceil(.99 * len(ordered)) - 1)],
            'min': ordered[0], 'max': ordered[-1], 'basis': 'nearest-rank'}


def telemetry_summary(samples, cycle_recording_requested=False):
    records = [{'run': sample['run'], 'warmup': sample['warmup'], **sample['decode_telemetry']}
               for sample in samples if isinstance(sample.get('decode_telemetry'), dict)]
    if not records:
        return None
    cycle_outputs = [count for record in records if not record['warmup']
                     for count in record.get('cycle_output_tokens', [])]
    return {'runs': records, 'scope': 'Per-request engine counters, including first target token',
            'batch2_executed_in_formal_runs': any(record.get('batch2_calls', 0) > 0
                                                for record in records if not record['warmup']),
            'cycle_recording_requested': cycle_recording_requested,
            'cycles_truncated': cycle_recording_requested and any(
                record.get('recorded_cycles', 0) < record.get('cycle_count', 0)
                for record in records if record.get('timing_enabled')),
            'any_batch2_failure_or_fallback': any(record.get('batch2_failures', 0) or record.get('batch2_fallbacks', 0)
                                                 for record in records if not record['warmup']),
            'mean_tokens_per_recorded_cycle': statistics.mean(cycle_outputs) if cycle_outputs else None,
            'cycle_output_tokens': distribution(cycle_outputs),
            'cycle_latency_ms': distribution([duration for record in records if not record['warmup']
                                             for duration in record.get('cycle_ms', [])])}


def attention_paths(samples):
    """Per-run attention path counters, or None for harnesses that predate them."""
    paths = [{'run': r['run'], 'warmup': r['warmup'], **r['attention_paths']}
             for r in samples if isinstance(r.get('attention_paths'), dict)]
    if not paths:
        return None
    fallbacks = sum(p.get('head_first_failures', 0) + p.get('head_first_cached_fallbacks', 0) +
                    p.get('matrix_fallbacks', 0) for p in paths)
    return {'runs': paths, 'any_fallback': bool(fallbacks)}


def indexer_compact_paths(samples):
    """Host command-encoding evidence, not a GPU timing or performance result."""
    records = [{'run': sample['run'], 'warmup': sample['warmup'], **sample['gpu_indexer_compact']}
               for sample in samples if isinstance(sample.get('gpu_indexer_compact'), dict)]
    if not records:
        return None
    formal = [record for record in records if not record['warmup']]
    return {'runs': records,
            'encoded_in_formal_runs': any(record['encoded'] > 0 for record in formal),
            'any_unavailable_in_formal_runs': any(record['unavailable'] > 0 for record in formal),
            'formal_totals': {key: sum(record[key] for record in formal) for key in INDEXER_COMPACT_COUNTERS},
            'scope': 'Per-request host path counters; encoded commands are not GPU duration or throughput proof'}


def topk_prefix_paths(samples):
    records = [{'run': sample['run'], 'warmup': sample['warmup'], **sample['gpu_topk_prefix']}
               for sample in samples if isinstance(sample.get('gpu_topk_prefix'), dict)]
    if not records:
        return None
    formal = [record for record in records if not record['warmup']]
    return {'runs': records,
            'encoded_in_formal_runs': any(record['encoded'] > 0 for record in formal),
            'any_unavailable_in_formal_runs': any(record['unavailable'] > 0 for record in formal),
            'formal_totals': {key: sum(record[key] for record in formal) for key in TOPK_PREFIX_COUNTERS},
            'scope': 'Per-request host encoding evidence. Write slots are planned finite-row work; '
                     'GPU NaN rows retain dense work. Decode counters exclude work through the first '
                     'callback. These are not actual GPU traffic or timing.'}


def validate_moe_cache_modes(environment):
    for flag in ('MYDS_MOE_Q2_SUM_CACHE', 'MYDS_MOE_Q2_SUM_CACHE_FUSED'):
        if environment.get(flag) not in (None, '0', '1'):
            raise ValueError('Invalid candidate 0/1 cohort: ' + flag + '=' + str(environment.get(flag)))
    if environment.get('MYDS_MOE_Q2_SUM_CACHE') == '1' and environment.get('MYDS_MOE_Q2_SUM_CACHE_FUSED') == '1':
        raise ValueError('Mutually exclusive candidate Q2 cache modes requested')
    if environment.get('MYDS_MOE_Q2_SUM_CACHE_FUSED') == '1' and \
            environment.get('MYDS_MOE_IQ2_CONSTANT_LUT') not in (None, '0'):
        raise ValueError('Fused candidate Q2 cache requires reference IQ2 LUT0')


def validate_capture_modes(environment):
    for flag in ('MYDS_DSPARK_CAPTURE_BYPASS', 'MYDS_ENGINE_CAPTURE_BYPASS'):
        if environment.get(flag) not in (None, '0', '1'):
            raise ValueError('Invalid candidate 0/1 cohort: ' + flag)
    if environment.get('MYDS_DSPARK_CAPTURE_BYPASS') == '1' and environment.get('MYDS_ENGINE_CAPTURE_BYPASS') == '1':
        raise ValueError('Mutually exclusive old and Engine capture bypass modes requested')


def capture_bypass_requested(environment):
    return any(environment.get(flag) == '1' for flag in
               ('MYDS_DSPARK_CAPTURE_BYPASS', 'MYDS_ENGINE_CAPTURE_BYPASS'))


def validate_f16_fuse_mode(environment):
    if environment.get('MYDS_F16_GEMV_BF16_FUSE') not in (None, '0', '1'):
        raise ValueError('Invalid candidate 0/1 cohort: MYDS_F16_GEMV_BF16_FUSE')


def require_f16_gemv_bf16_fuse(samples):
    # Host encoding evidence for every completed warm/formal ordinary graph.
    # Complete outputs/completion are audited elsewhere; encoded alone is not
    # a claim that a GPU command completed successfully.
    if not samples or not any(s.get('warmup') is False for s in samples):
        raise ValueError('F16 BF16 fuse requires complete warm/formal samples')
    for sample in samples:
        c = sample.get('gpu_f16_gemv_bf16_fuse')
        t = sample.get('decode_telemetry', {})
        if (not isinstance(c, dict) or any(type(c.get(k)) is not int or c[k] < 0
                for k in CANDIDATE_PATHS['gpu_f16_gemv_bf16_fuse'][2]) or
                c['requests'] <= 0 or c['encoded'] != c['requests'] or c['failures'] or
                c['decode_requests'] <= 0 or c['decode_encoded'] != c['decode_requests'] or c['decode_failures'] or
                c['decode_requests'] > c['requests'] or
                t.get('target_forward_calls') != sample.get('output_tokens') or
                t.get('target_rows') != sample.get('output_tokens') or
                any(t.get(k) != 0 for k in ('batch2_calls', 'batch2_failures', 'batch2_fallbacks',
                    'draft_cycles', 'draft_proposed', 'pld_queries', 'pld_cycles'))):
            raise ValueError('Required F16 BF16 fuse did not encode the full ordinary decode path')


def validate_indexer_rope_quant_mode(environment):
    if environment.get('MYDS_INDEXER_ROPE_QUANT_FUSE') not in (None, '0', '1'):
        raise ValueError('Invalid candidate 0/1 cohort: MYDS_INDEXER_ROPE_QUANT_FUSE')


def require_indexer_rope_quant_fuse(samples):
    # Host encoding evidence for every completed warm/formal ordinary graph.
    # Complete outputs/completion are audited elsewhere; encoded alone is not
    # a claim that a GPU command completed successfully.
    if not samples or not any(s.get('warmup') is False for s in samples):
        raise ValueError('Indexer RoPE quant fuse requires complete warm/formal samples')
    for sample in samples:
        c = sample.get('gpu_indexer_rope_quant_fuse')
        t = sample.get('decode_telemetry', {})
        if (not isinstance(c, dict) or any(type(c.get(k)) is not int or c[k] < 0
                for k in CANDIDATE_PATHS['gpu_indexer_rope_quant_fuse'][2]) or
                c['requests'] <= 0 or c['encoded'] != c['requests'] or c['failures'] or
                c['decode_requests'] <= 0 or c['decode_encoded'] != c['decode_requests'] or c['decode_failures'] or
                c['decode_requests'] > c['requests'] or
                t.get('target_forward_calls') != sample.get('output_tokens') or
                t.get('target_rows') != sample.get('output_tokens') or
                any(t.get(k) != 0 for k in ('batch2_calls', 'batch2_failures', 'batch2_fallbacks',
                    'draft_cycles', 'draft_proposed', 'pld_queries', 'pld_cycles'))):
            raise ValueError('Required Indexer RoPE quant fuse did not encode the full ordinary decode path')


def validate_candidate_paths(samples, environment, audit):
    """Validate native encoding records; absent historical records remain usable."""
    validate_moe_cache_modes(environment)
    validate_capture_modes(environment)
    validate_indexer_rope_quant_mode(environment)
    validate_f16_fuse_mode(environment)
    if environment.get('MYDS_MOE_Q2_SUM_CACHE_FUSED') == '1':
        for sample in samples:
            old = sample.get('gpu_moe_q2_sum_cache')
            audit.require(isinstance(old, dict) and all(type(old.get(key)) is int and old[key] == 0
                          for key in CANDIDATE_PATHS['gpu_moe_q2_sum_cache'][2]),
                          'Fused candidate requires zero independent-cache counters')
    for name, (flag, failure, keys) in CANDIDATE_PATHS.items():
        enabled_value = CANDIDATE_ENABLED_VALUES.get(name, '1')
        if name == 'gpu_lm_head_q8':
            audit.require(environment.get(flag) in (None, '2', '4'),
                          'Invalid LM head row cohort: ' + str(environment.get(flag)))
        if name in DECODE_SHAPE_CANDIDATES or name in ('gpu_moe_q2_v41','gpu_attention_packed_register','gpu_route_simd_shuffle','gpu_dspark_capture','gpu_topk_nan_shuffle','gpu_topk_threshold', 'gpu_indexer_head_slots', 'gpu_engram_project_round', 'gpu_moe_q2_sum_cache', 'gpu_moe_q2_sum_cache_fused', 'gpu_f16_gemv_bf16_fuse', 'gpu_indexer_rope_quant_fuse'):
            audit.require(environment.get(flag) in (None, '0', '1'),
                          'Invalid candidate 0/1 cohort: ' + str(environment.get(flag)))
        for sample in samples:
            counters = sample.get(name)
            if counters is None:
                enabled = capture_bypass_requested(environment) if name == 'gpu_dspark_capture' else environment.get(flag) == enabled_value
                audit.require(not enabled and not (name in ('gpu_f16_gemv_bf16_fuse','gpu_indexer_rope_quant_fuse') and environment.get(flag) is not None),
                              'Missing candidate path counters: ' + name)
                continue
            audit.require(isinstance(counters, dict), 'Invalid candidate path counters: ' + name)
            for key in keys:
                value = counters.get(key)
                audit.require(type(value) is int and value >= 0, 'Invalid candidate counter: ' + name + '.' + key)
            for key in ('requests', 'encoded', failure):
                audit.require(counters['decode_' + key] <= counters[key],
                              'Candidate decode counter exceeds request total: ' + name)
            audit.require(counters['encoded'] <= counters['requests'] and
                          counters[failure] <= counters['requests'] and
                          counters['decode_encoded'] <= counters['decode_requests'] and
                          counters['decode_' + failure] <= counters['decode_requests'],
                          'Candidate result exceeds requests: ' + name)
            if name == 'gpu_attention_packed_register':
                audit.require(counters['direct_encoded'] + counters['staged_encoded'] == counters['encoded'],
                              'Invalid candidate path attention register direct/staged accounting')
            if name == 'gpu_route_simd_shuffle':
                audit.require(counters['encoded'] + counters['failures'] == counters['requests'] and
                              counters['decode_encoded'] + counters['decode_failures'] == counters['decode_requests'],
                              'Invalid candidate path route shuffle encoding/failure counts differ from requests')
            if failure == 'unavailable':
                audit.require(counters['encoded'] + counters[failure] == counters['requests'] and
                              counters['decode_encoded'] + counters['decode_' + failure] == counters['decode_requests'],
                              'Candidate encoding/fallback counts differ from requests: ' + name)
            if name == 'gpu_indexer_head_slots':
                audit.require(counters['unavailable'] <= counters['failures'] and
                              counters['decode_unavailable'] <= counters['decode_failures'] and
                              counters['decode_unavailable'] <= counters['unavailable'],
                              'Candidate head-slot unavailable is not a failure subset')
            if name == 'gpu_topk_threshold':
                audit.require(counters.get('stats_completed') is True,
                              'TopK threshold GPU decisions were not observed after existing completion')
                for prefix in ('', 'decode_'):
                    audit.require(counters[prefix + 'validated'] == counters[prefix + 'encoded'] and
                                  counters[prefix + 'adopted'] + counters[prefix + 'fallback'] == counters[prefix + 'validated'],
                                  'TopK threshold GPU decision conservation differs')
                    audit.require(sum(counters[prefix + k] for k in
                                  ('nan', 'inf', 'insufficient', 'capacity', 'certificate', 'tie')) == counters[prefix + 'fallback'],
                                  'TopK threshold mutually exclusive fallback reasons differ')
                audit.require(all(counters['decode_' + k] <= counters[k] for k in TOPK_THRESHOLD_FIELDS),
                              'TopK threshold decode outcome exceeds total')
                audit.require(counters['errors'] == 0 and counters['encoded'] == counters['requests'],
                              'Completed TopK threshold graph has a host error or missing encoding')
                if environment.get(flag) == enabled_value:
                    expected_total, expected_decode = expected_topk_threshold_requests(sample['prompt_tokens'], sample['output_tokens'])
                    audit.require((counters['requests'], counters['decode_requests']) == (expected_total, expected_decode),
                                  'TopK threshold eligible ordinary row denominator differs')
            if name=='gpu_dspark_capture':
                for prefix in ('','decode_'):
                    audit.require(counters[prefix+'encoded']+counters[prefix+'bypassed']+counters[prefix+'failures']==counters[prefix+'requests'],
                                  'Capture result conservation differs')
                audit.require(all(counters['decode_'+k]<=counters[k] for k in
                    ('bypassed','stale_rejections')),'Capture decode counter exceeds total')
                if not capture_bypass_requested(environment):
                    audit.require(counters['bypassed']==0,'Reference capture path bypassed work')
            elif name=='gpu_topk_nan_shuffle' and environment.get(flag)==enabled_value:
                fused=sample.get('gpu_topk_nan_fuse')
                audit.require(isinstance(fused,dict) and all(counters[k]==fused[k] for k in
                    ('requests','encoded','decode_requests','decode_encoded')) and
                    fused.get('unavailable')==0 and fused.get('decode_unavailable')==0,
                    'Shuffle candidate path did not cover the complete eligible production NaN-fused path')
            elif environment.get(flag) != enabled_value:
                audit.require(not any(counters[key] for key in keys),
                              'Disabled candidate has execution counters: ' + name)


def expected_topk_threshold_requests(prompt_tokens, output_tokens):
    """Eight regular index rows, not the extra layer20 block TopK.

    First output uses pos=P-1; remaining N-1 outputs are the decode intervals.
    Three sources have compression ratio2 and five ratio1. Only top512 rows
    with n_cols8192..262144 enter the opt-in graph. No speculative rows here.
    """
    if type(prompt_tokens) is not int or type(output_tokens) is not int or prompt_tokens <= 0 or output_tokens < 0:
        raise ValueError('Invalid TopK threshold position/count')
    calls = [3 * int(8192 <= (prompt_tokens + j) // 2 <= 262144) +
             5 * int(8192 <= prompt_tokens + j <= 262144) for j in range(output_tokens)]
    return sum(calls), sum(calls[1:])


def candidate_paths(samples, name):
    _, failure, keys = CANDIDATE_PATHS[name]
    records = [{'run': s['run'], 'warmup': s['warmup'], **s[name]}
               for s in samples if isinstance(s.get(name), dict)]
    if not records:
        return None
    formal = [r for r in records if not r['warmup']]
    result = {'runs': records,
            'encoded_in_formal_decode': any(r['decode_encoded'] > 0 for r in formal),
            'any_failure_in_formal_runs': any(r[failure] > 0 for r in formal),
            'formal_totals': {k: sum(r[k] for r in formal) for k in keys},
            'scope': 'Host encoding counters, not GPU duration, physical weight traffic or NaN branch counts; '
                     'decode counters exclude work through the first callback.'}
    if name=='gpu_dspark_capture':
        result.update(bypassed_in_formal_decode=any(r['decode_bypassed']>0 for r in formal),
            any_stale_rejection_in_formal_runs=any(r['stale_rejections']>0 for r in formal),
            scope='Host capture request/encoding/bypass counters, not GPU duration; decode excludes first callback.')
    if name == 'gpu_topk_threshold':
        result.update(adopted_in_formal_decode=any(r['decode_adopted'] > 0 for r in formal),
                      fallback_in_formal_decode=any(r['decode_fallback'] > 0 for r in formal),
                      formal_decode_adoption_fraction=(sum(r['decode_adopted'] for r in formal) /
                          sum(r['decode_validated'] for r in formal)) if sum(r['decode_validated'] for r in formal) else None,
                      scope='Host graph encoding plus GPU completed validation/adoption/fallback decisions; '
                            'only ordinary top512 index rows with n_cols8192..262144. Extra layer20 block TopK excluded. '
                            'GPU stats observed at existing generation completion; no per-TopK CPU wait. '
                            'Decode excludes work through first callback; no GPU time or physical traffic claim.')
    return result


def require_candidate_path(samples, name):
    _, failure, _ = CANDIDATE_PATHS[name]
    if name in DECODE_SHAPE_CANDIDATES:
        _, _, per_target, _ = DECODE_SHAPE_CANDIDATES[name]
        if not samples or not any(not s['warmup'] for s in samples):
            raise ValueError('Required shape candidate has no warm/formal samples')
        for sample in samples:  # Warmup must also adopt the complete actual path.
            c = sample.get(name, {})
            t = sample.get('decode_telemetry', {})
            if (sample.get('candidate_modes', {}).get(name) != 1 or
                    sample.get('candidate_layer_count') != 40 or
                    c.get('requests') != per_target * sample['output_tokens'] or
                    c.get('encoded') != c.get('requests') or c.get('failures') != 0 or
                    c.get('decode_requests') != per_target * sample['decode_intervals'] or
                    c.get('decode_encoded') != c.get('decode_requests') or c.get('decode_failures') != 0 or
                    t.get('target_forward_calls') != sample['output_tokens'] or
                    t.get('target_rows') != sample['output_tokens'] or
                    any(t.get(k) != 0 for k in ('batch2_calls', 'batch2_failures', 'batch2_fallbacks',
                                              'draft_cycles', 'draft_proposed', 'pld_queries', 'pld_cycles'))):
                raise ValueError('Required shape candidate did not adopt every ordinary warm/formal target: ' + name)
        return
    formal = [s.get(name) for s in samples if not s['warmup']]
    if name=='gpu_dspark_capture':
        for sample in (s for s in samples if not s['warmup']):
            c=sample.get(name)
            if not c or c['requests']<=0 or c['bypassed']!=c['requests'] or c['encoded'] or c['failures'] or c['stale_rejections'] or \
                c['decode_requests']!=3*sample['decode_intervals'] or c['decode_bypassed']!=c['decode_requests'] or \
                c['decode_encoded'] or c['decode_failures'] or c['decode_stale_rejections'] or \
                sample['decode_telemetry']['target_forward_calls']!=sample['output_tokens'] or \
                sample['decode_telemetry']['target_rows']!=sample['output_tokens']:
                raise ValueError('Required capture bypass did not bypass the full ordinary decode path')
        if not formal: raise ValueError('Capture bypass has no formal samples')
        return
    if not formal or any(not c or c['requests'] <= 0 or c['encoded'] != c['requests'] or c[failure] or
                         c['decode_requests'] <= 0 or c['decode_encoded'] != c['decode_requests'] or
                         c['decode_' + failure] for c in formal):
        raise ValueError('Required candidate path did not encode every formal decode request: ' + name)
    if name in ('gpu_route_simd_shuffle', 'gpu_attention_packed_register', 'gpu_moe_q2_v41'):
        for sample in (s for s in samples if not s['warmup']):
            c = sample[name]
            t = sample.get('decode_telemetry', {})
            if (sample.get('candidate_layer_count') != 40 or
                    c['decode_requests'] != 40 * sample['decode_intervals'] or
                    t.get('target_forward_calls') != sample['output_tokens'] or
                    t.get('target_rows') != sample['output_tokens'] or
                    any(t.get(k) != 0 for k in ('batch2_calls', 'batch2_failures', 'batch2_fallbacks',
                                              'draft_cycles', 'draft_proposed', 'pld_queries', 'pld_cycles'))):
                raise ValueError('Required candidate path did not cover all 40 layers of ordinary decode: ' + name)
    if name == 'gpu_topk_threshold':
        for sample in (s for s in samples if not s['warmup']):
            c = sample[name]
            total, decode = expected_topk_threshold_requests(sample['prompt_tokens'], sample['output_tokens'])
            if (decode <= 0 or c['requests'] != total or c['decode_requests'] != decode or
                    c['stats_completed'] is not True or c['validated'] != total or c['decode_validated'] != decode or
                    c['decode_adopted'] <= 0 or
                    sample['decode_telemetry']['target_forward_calls'] != sample['output_tokens'] or
                    sample['decode_telemetry']['target_rows'] != sample['output_tokens']):
                raise ValueError('Required TopK threshold path lacks actual eligible ordinary GPU adoption')
    if name == 'gpu_indexer_head_slots':
        # Native V4.1 validates the eight index-source layers and 32x128
        # indexer shape. The pure-target protocol has one forward per output;
        # work through the first callback is excluded from N-1 decode periods.
        for sample in (s for s in samples if not s['warmup']):
            counters = sample[name]
            if (counters['unavailable'] or counters['decode_unavailable'] or
                    counters['requests'] != 8 * sample['output_tokens'] or
                    counters['decode_requests'] != 8 * sample['decode_intervals'] or
                    sample['decode_telemetry']['target_forward_calls'] != sample['output_tokens'] or
                    sample['decode_telemetry']['target_rows'] != sample['output_tokens']):
                raise ValueError('Required candidate path did not encode the full 32x128 eight-layer scalar shape: ' + name)


def require_batch2_graph(samples):
    """Reject successful scalar recovery masking a failed Batch2 graph."""
    formal = [s.get('decode_telemetry') for s in samples if not s['warmup']]
    if not formal or any(not c or c.get('batch2_calls', 0) <= 0 or c.get('batch2_hits', 0) <= 0 or
                         c.get('batch2_failures', 0) or c.get('batch2_fallbacks', 0) for c in formal):
        raise ValueError('Required Batch2 graph did not execute with hits and zero failures/fallbacks in every formal sample')


def compare(args):
    audit = load_audit()
    groups, raw, states = [], [], []
    for directory in (args.base.resolve(), args.candidate.resolve()):
        state = json.loads((directory / 'run-state.json').read_text())
        audit.require(state.get('status') == 'validated' and state.get('returncode') == 0,
                      f'Incomplete/failed run: {directory}')
        group, samples = validate_run(directory, state, audit)
        groups.append(group)
        raw.append(samples)
        states.append(state)
    base, candidate = groups
    for key in audit.MATCH_META:
        audit.require(base['metadata'][key] == candidate['metadata'][key], 'Input/config differs: ' + key)
    for key in ('request_sha256', 'model'):
        audit.require(states[0][key] == states[1][key], 'File identity differs: ' + key)
    audit.require(states[0].get('dspark_model') == states[1].get('dspark_model'),
                  'File identity differs: dspark_model')
    audit.require(states[0].get('arm', 'custom') == states[1].get('arm', 'custom'),
                  'Named measurement arm differs between base and candidate')
    if getattr(args, 'same_build', False):
        for key in ('binary_sha256', 'metallib_sha256', 'source'):
            audit.require(states[0].get(key) is not None and states[0].get(key) == states[1].get(key),
                          'Same-build comparison differs: ' + key)
    for key in ('cpu', 'memory_bytes', 'os'):
        audit.require(states[0]['host_before'][key] == states[1]['host_before'][key], 'Host differs: ' + key)
    audit.require((base['metadata']['environment'].get('MYDS_DECODE_METRICS') or '0') ==
                  (candidate['metadata']['environment'].get('MYDS_DECODE_METRICS') or '0'),
                  'Measurement setting differs: MYDS_DECODE_METRICS')
    output_signature = ('token_ids', 'output_bytes_fnv1a64', 'output_bytes', 'prompt_tokens',
                        'prefill_tokens', 'output_tokens', 'decode_intervals', 'stop_reason')
    for before, after in zip(*raw):
        for key in output_signature:
            audit.require(before[key] == after[key], f'Output/count differs in run {before["run"]}: {key}')
        if 'final_logits' in before or 'final_logits' in after:
            audit.require(before.get('final_logits') == after.get('final_logits'),
                          f'Final logits differ in run {before["run"]}')
    reductions = {}
    for phase in ('prefill', 'decode'):
        before = base['stats'][phase + '_ms_per_token']
        after = candidate['stats'][phase + '_ms_per_token']
        reductions[phase] = {'base': before, 'candidate': after,
                             'time_saved_percent': 100 * (1 - after['median'] / before['median'])}
    summary = {'audit_pass': True, 'all_warm_and_formal_outputs_match': True,
               'scope': f"Separate processes, one warmup plus {base['formal_runs']} formal samples; direct callback timing",
               'base': base, 'candidate': candidate, 'comparison': reductions,
               'attention_paths': {'base': attention_paths(raw[0]),
                                   'candidate': attention_paths(raw[1])},
               'gpu_indexer_compact': {'base': base['gpu_indexer_compact'],
                                       'candidate': candidate['gpu_indexer_compact']},
               'gpu_topk_prefix': {'base': base['gpu_topk_prefix'],
                                   'candidate': candidate['gpu_topk_prefix']},
               'builds': [{'binary_sha256': s['binary_sha256'], 'metallib_sha256': s['metallib_sha256'],
                           'environment': s['environment'], 'source': s.get('source'),
                           'dspark_model': s.get('dspark_model'), 'arm': s.get('arm', 'custom')}
                          for s in states],
               'environment_changes': {name: [states[0]['environment'].get(name), states[1]['environment'].get(name)]
                   for name in sorted(set(states[0]['environment']) | set(states[1]['environment']))
                   if states[0]['environment'].get(name) != states[1]['environment'].get(name)}}
    for name in CANDIDATE_PATHS:
        summary[name] = {'base': base[name], 'candidate': candidate[name]}
    control_directory = getattr(args, 'control', None)
    if control_directory is not None:
        control_directory = control_directory.resolve()
        audit.require(len({args.base.resolve(), args.candidate.resolve(), control_directory}) == 3,
                      'A–B–A requires three distinct run directories')
        control_state = json.loads((control_directory / 'run-state.json').read_text())
        audit.require(control_state.get('status') == 'validated' and control_state.get('returncode') == 0,
                      'Incomplete/failed final control')
        control, control_raw = validate_run(control_directory, control_state, audit)
        for key in ('binary_sha256', 'metallib_sha256', 'model', 'environment', 'dspark_model', 'source'):
            audit.require(states[0].get(key) == control_state.get(key), 'Final control identity differs: ' + key)
        for key in ('cpu', 'memory_bytes', 'os'):
            audit.require(states[0]['host_before'][key] == control_state['host_before'][key], 'Final control host differs: ' + key)
        for key in audit.MATCH_META:
            audit.require(base['metadata'][key] == control['metadata'][key], 'Final control input differs: ' + key)
        for before, after in zip(raw[0], control_raw):
            for key in output_signature:
                audit.require(before[key] == after[key],
                              f'Final control output/count differs in run {before["run"]}: {key}')
            if 'final_logits' in before or 'final_logits' in after:
                audit.require(before.get('final_logits') == after.get('final_logits'),
                              f'Final control logits differ in run {before["run"]}')
        summary['final_control'] = control
        summary['control_drift_percent'] = {
            phase: 100 * (control['stats'][phase + '_ms_per_token']['median'] /
                          base['stats'][phase + '_ms_per_token']['median'] - 1)
            for phase in ('prefill', 'decode')}
        summary['scope'] = f"A–B–A separate processes; one warmup and {base['formal_runs']} formal samples per arm; direct callback timing"
    output = args.out.resolve()
    if output.exists():
        raise ValueError('Comparison output already exists: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, summary)
    print(json.dumps({'audit_pass': True, 'output': str(output), 'comparison': reductions}))
    return 0


def stop_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def require_engine_capture_bypass(samples):
    """The new candidate must cover every complete ordinary request, including warmup."""
    if not samples or not any(not s['warmup'] for s in samples):
        raise ValueError('Engine capture bypass has no formal samples')
    for sample in samples:
        c = sample.get('gpu_dspark_capture')
        t = sample.get('decode_telemetry', {})
        if not c or c['requests'] <= 0 or c['bypassed'] != c['requests'] or c['encoded'] or c['failures'] or c['stale_rejections'] or \
                c['decode_requests'] != 3 * sample['decode_intervals'] or c['decode_bypassed'] != c['decode_requests'] or \
                c['decode_encoded'] or c['decode_failures'] or c['decode_stale_rejections'] or \
                t.get('target_forward_calls') != sample['output_tokens'] or t.get('target_rows') != sample['output_tokens'] or \
                any(t.get(k) != 0 for k in ('batch2_calls', 'batch2_failures', 'batch2_fallbacks',
                                           'draft_cycles', 'draft_proposed', 'pld_queries', 'pld_cycles')):
            raise ValueError('Required Engine capture bypass did not bypass the full ordinary decode graph')


def run(args):
    if platform.system() != 'Darwin':
        raise ValueError('The model runner requires macOS/Metal; verify works without a GPU')
    binary, model, request = (p.expanduser().resolve() for p in (args.binary, args.model, args.request))
    dspark_arg = getattr(args, 'dspark_model', None)
    support = dspark_arg.expanduser().resolve() if dspark_arg is not None else None
    arm = getattr(args, 'arm', 'custom')
    if arm in DSPARK_ARMS and support is None:
        raise ValueError('--dspark-model is required for the ' + arm + ' arm')
    if arm in ('target', 'pld', 'pld-batch2') and support is not None:
        raise ValueError('--dspark-model is incompatible with the ' + arm + ' arm')
    if support is None and getattr(args, 'dspark_sha256', None) is not None:
        raise ValueError('--dspark-sha256 requires --dspark-model')
    for path in (binary, model, request, binary.parent / 'myds.metallib', *([support] if support else [])):
        if not path.is_file():
            raise ValueError('Missing input: ' + str(path))
    if not os.access(binary, os.X_OK):
        raise ValueError('Harness is not executable: ' + str(binary))
    explicit_env = dict(DEFAULT_ENV)
    if arm != 'custom':
        explicit_env.update(MYDS_DECODE_SPLIT='4', MYDS_DECODE_METRICS='1', MYDS_SPEC_COST_SCHEDULER='0')
    explicit_env.update(ARM_ENV.get(arm, {}))
    for setting in args.env:
        name, separator, value = setting.partition('=')
        if not separator or not name.startswith('MYDS_'):
            raise ValueError('--env requires MYDS_NAME=value')
        explicit_env[name] = value
    for name, value in ARM_ENV.get(arm, {}).items():
        if explicit_env[name] != value:
            raise ValueError(f'{name}={explicit_env[name]} conflicts with named arm {arm}; use --arm custom')
    for name in ('MYDS_PREFIX_CACHE', 'MYDS_KV_DSK', 'MYDS_PHASE_PROFILE'):
        if explicit_env[name] != '0':
            raise ValueError(name + '=0 is required for this full-prefill protocol')
    require_uninstrumented_environment(explicit_env)
    validate_moe_cache_modes(explicit_env)
    validate_capture_modes(explicit_env)
    validate_indexer_rope_quant_mode(explicit_env)
    validate_f16_fuse_mode(explicit_env)
    if explicit_env.get('MYDS_ROUTE_SIMD_SHUFFLE') not in (None, '0', '1'):
        raise ValueError('Invalid candidate 0/1 cohort: MYDS_ROUTE_SIMD_SHUFFLE')
    if explicit_env.get('MYDS_ATTN_PACKED_REGISTER') not in (None, '0', '1'):
        raise ValueError('Invalid attention register 0/1 cohort')
    if explicit_env.get('MYDS_MOE_Q2_V41') not in (None, '0', '1'):
        raise ValueError('Invalid Q2 V4.1 0/1 cohort')
    for _, (flag, _, _, option) in DECODE_SHAPE_CANDIDATES.items():
        if explicit_env.get(flag) not in (None, '0', '1'):
            raise ValueError('Invalid candidate 0/1 cohort: ' + flag)
        if getattr(args, option, False):
            if (explicit_env.get(flag) != '1' or support is not None or arm not in ('target', 'custom') or
                    any(explicit_env.get(k) != '0' for k in ('MYDS_PLD', 'MYDS_BATCH2_SPEC', 'MYDS_SPEC_SERIAL_PROBE'))):
                raise ValueError('Required shape candidate needs explicit flag1 and ordinary target scope: ' + flag)
    if getattr(args, 'require_moe_q2_v41', False) and explicit_env.get('MYDS_MOE_Q2_V41') != '1':
        raise ValueError('Required Q2 V4.1 path needs explicit flag1')
    if getattr(args, 'require_attention_packed_register', False) and explicit_env.get('MYDS_ATTN_PACKED_REGISTER') != '1':
        raise ValueError('Required attention register path needs explicit flag1')
    if getattr(args, 'require_route_simd_shuffle', False) and explicit_env.get('MYDS_ROUTE_SIMD_SHUFFLE') != '1':
        raise ValueError('Required route SIMD shuffle path needs explicit flag1')
    if getattr(args, "require_f16_gemv_bf16_fuse", False) and (
            explicit_env.get("MYDS_F16_GEMV_BF16_FUSE") != "1" or
            any(explicit_env.get(k) != "0" for k in ("MYDS_PLD", "MYDS_BATCH2_SPEC", "MYDS_SPEC_SERIAL_PROBE"))):
        raise ValueError("Required F16 BF16 fuse needs explicit flag1 and ordinary SPEC0/SERIAL0 path")
    if getattr(args, "require_indexer_rope_quant_fuse", False) and (
            explicit_env.get("MYDS_INDEXER_ROPE_QUANT_FUSE") != "1" or arm != "target" or support is not None or
            any(explicit_env.get(k) != "0" for k in ("MYDS_PLD", "MYDS_BATCH2_SPEC", "MYDS_SPEC_SERIAL_PROBE"))):
        raise ValueError("Required Indexer RoPE quant fuse needs explicit flag1 and ordinary SPEC0/SERIAL0 path")
    for flag in ('MYDS_INDEXER_TOPK_NAN_SHUFFLE','MYDS_DSPARK_CAPTURE_BYPASS'):
        if explicit_env.get(flag) not in (None,'0','1'):
            raise ValueError('Invalid candidate 0/1 cohort: '+flag)
    if getattr(args,'require_topk_nan_shuffle',False) and explicit_env.get('MYDS_INDEXER_TOPK_NAN_SHUFFLE')!='1':
        raise ValueError('Required NaN shuffle path needs explicit flag1')
    if getattr(args,'require_dspark_capture_bypass',False) and (support is None or
        explicit_env.get('MYDS_DSPARK_CAPTURE_BYPASS')!='1' or
        any(explicit_env.get(k)!='0' for k in ('MYDS_PLD','MYDS_BATCH2_SPEC','MYDS_SPEC_SERIAL_PROBE'))):
        raise ValueError('Required capture bypass needs loaded support and ordinary SPEC0/SERIAL0 path')
    if getattr(args, 'require_engine_capture_bypass', False) and (support is None or
        explicit_env.get('MYDS_ENGINE_CAPTURE_BYPASS') != '1' or explicit_env.get('MYDS_DSPARK_CAPTURE_BYPASS') != '0' or
        any(explicit_env.get(k) != '0' for k in ('MYDS_PLD', 'MYDS_BATCH2_SPEC', 'MYDS_SPEC_SERIAL_PROBE'))):
        raise ValueError('Required Engine capture bypass needs old0/new1, loaded support and ordinary SPEC0/SERIAL0 path')
    if explicit_env.get('MYDS_INDEXER_TOPK_THRESHOLD') not in (None, '0', '1'):
        raise ValueError('Invalid candidate 0/1 cohort: MYDS_INDEXER_TOPK_THRESHOLD')
    if explicit_env.get('MYDS_INDEXER_TOPK_THRESHOLD') == '1' and (arm != 'target' or
            explicit_env.get('MYDS_INDEXER_COMPACT_SCORER') not in (None, '0')):
        raise ValueError('TopK threshold graph needs pure target and dense scorer')
    if getattr(args, 'require_topk_threshold', False) and explicit_env.get('MYDS_INDEXER_TOPK_THRESHOLD') != '1':
        raise ValueError('Required TopK threshold candidate needs flag1')
    if explicit_env.get('MYDS_INDEXER_HEAD_SLOTS') not in (None, '0', '1'):
        raise ValueError('Invalid candidate 0/1 cohort: MYDS_INDEXER_HEAD_SLOTS')
    if getattr(args, 'require_indexer_head_slots', False) and (
            explicit_env.get('MYDS_INDEXER_HEAD_SLOTS') != '1' or arm != 'target' or
            explicit_env.get('MYDS_INDEXER_COMPACT_SCORER') not in (None, '0')):
        raise ValueError('Required indexer head-slot candidate needs flag1, pure target and dense scorer')
    check_idle(args.server_port)
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {k: v for k, v in os.environ.items() if not k.startswith('MYDS_')}
    environment.update(explicit_env)
    argv = [str(binary), str(model), str(request), str(getattr(args,'repeats',3)), str(args.max_new_tokens), str(args.context)]
    if support is not None:
        argv.append(str(support))
    request_data = request.read_bytes()
    state = {'argv': argv, 'cwd': str(binary.parent.parent), 'environment': explicit_env,
             'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'binary_sha256': sha(binary), 'metallib_sha256': sha(binary.parent / 'myds.metallib'),
             'request_sha256': hashlib.sha256(request_data).hexdigest(),
             'request_bytes': len(request_data), 'request_fnv1a64': fnv_bytes(request_data),
             'model': file_identity(model, getattr(args, 'model_sha256', None)),
             'dspark_model': file_identity(support, getattr(args, 'dspark_sha256', None)) if support else None,
             'arm': arm, 'source': source_state(),
             'host_before': host_state(), 'status': 'started'}
    write_json(output / 'run-state.json', state)
    child = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise RuntimeError('Measurement interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / 'run-metrics.jsonl').open('w') as stdout, (output / 'run.log').open('w') as stderr:
            child = subprocess.Popen(argv, cwd=binary.parent.parent, env=environment,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            result = child.wait(timeout=args.timeout)
        state['returncode'] = result
        if result:
            raise RuntimeError(f'Harness exited {result}; inspect {output / "run.log"}')
        group, samples = validate_run(output, state, load_audit())
        if getattr(args, 'require_engine_capture_bypass', False):
            require_engine_capture_bypass(samples)
        if getattr(args, 'require_indexer_rope_quant_fuse', False):
            require_indexer_rope_quant_fuse(samples)
        if getattr(args, 'require_f16_gemv_bf16_fuse', False):
            require_f16_gemv_bf16_fuse(samples)
        if arm in ('batch2', 'pld-batch2', 'hybrid') or getattr(args, 'require_batch2_q8_shared', False):
            require_batch2_graph(samples)
        if getattr(args, 'require_topk_prefix', False):
            formal = [sample.get('gpu_topk_prefix') for sample in samples if not sample['warmup']]
            if not formal or any(not counters or counters['requests'] <= 0 or
                                 counters['encoded'] != counters['requests'] or counters['unavailable'] or
                                 counters['decode_requests'] <= 0 or
                                 counters['decode_encoded'] != counters['decode_requests'] or
                                 counters['decode_unavailable'] for counters in formal):
                raise ValueError('Required topk prefix path did not encode every formal decode request')
        for option, name in (('require_moe_down_overlap', 'gpu_moe_down_overlap'),
                             ('require_moe_iq2_v41', 'gpu_moe_iq2_v41'),
                             ('require_dense_q4_v41', 'gpu_dense_q4_v41'),
                             ('require_dense_q4_v41_nb', 'gpu_dense_q4_v41_nb'),
                             ('require_dense_q4_v41_head_reference', 'gpu_dense_q4_v41_head_reference'),
                             ('require_attention_rope_fuse', 'gpu_attention_rope_fuse'),
                             ('require_moe_hc_overlap', 'gpu_moe_hc_overlap'),
                             ('require_attn_hc_overlap', 'gpu_attn_hc_overlap'),
                             ('require_indexer_decode_overlap', 'gpu_indexer_decode_overlap'),
                             ('require_attn_kv_reuse', 'gpu_attn_kv_reuse'),
                             ('require_ffn_hc_overlap', 'gpu_ffn_hc_overlap'),
                             ('require_shared_q4_v41_nb', 'gpu_shared_q4_v41_nb'),
                             ('require_attention_direct_pack_fuse', 'gpu_attention_direct_pack_fuse'),
                             ('require_moe_q2_v41', 'gpu_moe_q2_v41'),
                             ('require_attention_packed_register', 'gpu_attention_packed_register'),
                             ('require_route_simd_shuffle', 'gpu_route_simd_shuffle'),
                             ('require_topk_threshold', 'gpu_topk_threshold'),
                             ('require_topk_nan_shuffle','gpu_topk_nan_shuffle'),
                             ('require_dspark_capture_bypass','gpu_dspark_capture'),
                             ('require_indexer_head_slots', 'gpu_indexer_head_slots'),
                             ('require_topk_nan_fuse', 'gpu_topk_nan_fuse'),
                             ('require_batch2_q8_shared', 'gpu_batch2_q8_shared'),
                             ('require_lm_head_q8', 'gpu_lm_head_q8'),
                             ('require_engram_project_round', 'gpu_engram_project_round'),
                             ('require_moe_q2_sum_cache', 'gpu_moe_q2_sum_cache'),
                             ('require_moe_q2_sum_cache_fused', 'gpu_moe_q2_sum_cache_fused')):
            if getattr(args, option, False):
                require_candidate_path(samples, name)
        group['attention_paths'] = attention_paths(samples)
        write_json(output / 'run-summary.json', group)
        state['status'] = 'validated'
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        try:
            if child:
                try:
                    stop_child(child)
                except (OSError, subprocess.SubprocessError) as exc:
                    state.update(status='failed', cleanup_error=str(exc))
            try:
                state['host_after'] = host_state()
            except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                state.update(status='failed', host_after_error=str(exc))
            write_json(output / 'run-state.json', state)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    if state['status'] != 'validated':
        raise RuntimeError('Final state capture/cleanup failed; inspect ' + str(output / 'run-state.json'))
    paths = group.get('attention_paths')
    print(json.dumps({'audit_pass': True, 'output': str(output),
                      'prefill_ms_per_token': group['stats']['prefill_ms_per_token'],
                      'decode_ms_per_token': group['stats']['decode_ms_per_token'],
                      'attention_fallback': paths['any_fallback'] if paths else None}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verification = commands.add_parser('verify', help='Check committed evidence SHA manifests; no GPU/model access')
    verification.add_argument('--check-source', action='store_true', help='Also check the packaged source snapshot')
    comparison = commands.add_parser('compare', help='Audit raw base/candidate samples and compare medians; no GPU')
    comparison.add_argument('--base', type=Path, required=True)
    comparison.add_argument('--candidate', type=Path, required=True)
    comparison.add_argument('--control', type=Path, help='Final A arm, identical to --base, for A–B–A drift')
    comparison.add_argument('--out', type=Path, required=True)
    comparison.add_argument('--same-build', action='store_true',
                            help='Require identical binary, metallib and source for an environment-only comparison')
    measurement = commands.add_parser('run', help='Run one model configuration; requires an already idle machine')
    measurement.add_argument('--binary', type=Path, default=ROOT / 'bin/bench_phase')
    measurement.add_argument('--model', type=Path, required=True)
    measurement.add_argument('--model-sha256', help='Caller-supplied immutable target artifact digest; not rehashed by runner')
    measurement.add_argument('--dspark-model', type=Path, help='Support GGUF; load failure is fatal')
    measurement.add_argument('--dspark-sha256', help='Caller-supplied immutable support artifact digest; not rehashed by runner')
    measurement.add_argument('--arm', choices=('target', 'pld', 'pld-batch2', 'dspark', 'batch2', 'hybrid', 'custom'),
                             default='target', help='Explicit path settings; custom permits arbitrary --env overrides')
    measurement.add_argument('--request', type=Path, required=True)
    measurement.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are refused')
    measurement.add_argument('--require-topk-prefix', action='store_true',
                             help='Require all formal runs to encode every eligible decode prefix request without fallback')
    measurement.add_argument('--require-moe-down-overlap', action='store_true',
                             help='Require the complete actual ordinary moe_down_overlap path in warmup and every formal target')
    measurement.add_argument('--require-moe-iq2-v41', action='store_true',
                             help='Require the complete actual ordinary moe_iq2_v41 path in warmup and every formal target')
    measurement.add_argument('--require-dense-q4-v41', action='store_true',
                             help='Require the complete actual ordinary dense_q4_v41 path in warmup and every formal target')
    measurement.add_argument('--require-dense-q4-v41-nb', action='store_true',
        help='Require complete ordinary-target dense_q4_v41_nb adoption')
    measurement.add_argument('--require-dense-q4-v41-head-reference', action='store_true',
        help='Require complete ordinary-target dense_q4_v41_head_reference adoption')
    measurement.add_argument('--require-attention-rope-fuse', action='store_true',
        help='Require complete ordinary-target attention_rope_fuse adoption')
    measurement.add_argument('--require-moe-hc-overlap', action='store_true',
        help='Require complete ordinary-target moe_hc_overlap adoption')
    measurement.add_argument('--require-attn-hc-overlap', action='store_true',
        help='Require complete ordinary-target attn_hc_overlap adoption')
    measurement.add_argument('--require-indexer-decode-overlap', action='store_true',
        help='Require complete ordinary-target indexer_decode_overlap adoption')
    measurement.add_argument('--require-attn-kv-reuse', action='store_true',
        help='Require complete ordinary-target attn_kv_reuse adoption')
    measurement.add_argument('--require-ffn-hc-overlap', action='store_true',
        help='Require complete ordinary-target ffn_hc_overlap adoption')
    measurement.add_argument('--require-shared-q4-v41-nb', action='store_true',
        help='Require complete fixed Shared Q4 GU/Down adoption in every ordinary target')
    measurement.add_argument('--require-attention-direct-pack-fuse', action='store_true',
        help='Require complete fused raw/selected attention packing in every ordinary target')
    measurement.add_argument('--require-route-simd-shuffle', action='store_true',
                             help='Require actual router SIMD shuffle encoding in every formal decode, with no failures')
    measurement.add_argument('--require-attention-packed-register', action='store_true',
                             help='Require actual packed attention register encoding in every formal decode, with no failures')
    measurement.add_argument('--require-moe-q2-v41', action='store_true',
                             help='Require actual specialized Q2 down encoding in every formal decode, with no failures')
    measurement.add_argument('--require-topk-nan-shuffle',action='store_true',
                             help='Require actual shuffle encoding in every eligible formal decode request')
    measurement.add_argument('--require-dspark-capture-bypass',action='store_true',
                             help='Require all three unconsumed target captures per ordinary decode to be bypassed')
    measurement.add_argument('--require-engine-capture-bypass', action='store_true',
                             help='Require actual Engine mode1 with old mode0, all ordinary capture work bypassed and no speculative graph')
    measurement.add_argument('--require-topk-threshold', action='store_true',
                             help='Require complete eligible ordinary top512 graph and actual GPU adoption in every formal decode')
    measurement.add_argument('--require-indexer-head-slots', action='store_true',
                             help='Require actual native 32x128 head-slot encoding in all eight scalar indexers per output, with no failures/unavailability')
    measurement.add_argument('--require-indexer-rope-quant-fuse', action='store_true',
                             help='Require actual fused indexer RoPE/quant in every warm/formal pure ordinary target decode, with no failures')
    measurement.add_argument('--require-f16-gemv-bf16-fuse', action='store_true',
                             help='Require actual F16 fused BF16 encoding in every warm/formal ordinary decode, with no failures')
    measurement.add_argument('--require-topk-nan-fuse', action='store_true',
                             help='Require native fused NaN sort counters in every formal decode, with no fallback')
    measurement.add_argument('--require-batch2-q8-shared', action='store_true',
                             help='Require native shared Q8 projection counters in every formal decode, with no failures')
    measurement.add_argument('--require-lm-head-q8', action='store_true',
                             help='Require native NR4 Q8 LM head encoding in every formal decode, with no failures')
    measurement.add_argument('--require-engram-project-round', action='store_true',
                             help='Require native Engram projection round-skip encoding in every formal decode, with no failures')
    measurement.add_argument('--require-moe-q2-sum-cache', action='store_true',
                             help='Require complete routed IQ2/precompute/Q2 encoding in every formal decode, with no failures')
    measurement.add_argument('--require-moe-q2-sum-cache-fused', action='store_true',
                             help='Require complete fused IQ2/cache/Q2 encoding in every formal decode, with no failures')
    measurement.add_argument('--env', action='append', default=[], help='Explicit MYDS_NAME=value, repeatable')
    measurement.add_argument('--max-new-tokens', type=int, choices=OUTPUT_CAPS, default=64)
    measurement.add_argument('--repeats', type=int, choices=(3,5), default=3,
                             help='Formal samples after one warmup; fixed before launching the native harness')
    measurement.add_argument('--context', type=int, default=262144)
    measurement.add_argument('--timeout', type=float, default=900)
    measurement.add_argument('--server-port', type=int, default=8181)
    args = parser.parse_args()
    try:
        if args.command == 'run' and (not 0 < args.context <= 0xffffffff or
                                     not math.isfinite(args.timeout) or args.timeout <= 0 or
                                     not 1 <= args.server_port <= 65535):
            raise ValueError('context must be 1..UINT32_MAX, timeout finite/positive, server-port 1..65535')
        if args.command == 'verify':
            return verify(args)
        if args.command == 'compare':
            return compare(args)
        return run(args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
