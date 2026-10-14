#!/usr/bin/env python3
"""Independent CPU audit of full RF component bytes and three library origins.

Never initializes Metal or invokes the producer. A successful component report
does not establish full-model output/state parity, speed, or production adoption.
"""
import argparse
import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import statistics
import struct
import sys

GOLDEN_SHA = '99b792d59f1f614065c87b1c1dd6b99ff4989d26fdc73c7bb55d9f770090d0c1'
REGISTER = 'kernel_myds_flash_attn_vec_packed8_register_reduce_f16_dk512_dv512'
ROPE = 'kernel_dsv41_rope_round'
FUSED = 'kernel_myds_attention_register_rope_fuse_f16_512'
DEBUG = ('probe_attention_rope_reference', 'probe_attention_rope_fused')
ROLES = {'cold_golden', 'private_debug', 'deployed_candidate'}
POSITIONS = (0, 11, 127, 128, 8191, 32768)
KEY_COUNTS = (1, 33, 128, 129, 639, 640)
REJECTIONS = ('selected_PSO_absent_in_real_golden_library', 'pos_limit', 'short_q', 'heads_alias_q',
              'scratch_writers_alias', 'null_consumed', 'nraw_zero', 'nraw_too_large',
              'raw_start_limit', 'selected_too_large', 'forged_cpu_pointer')
NEGATIVE_LENGTHS = (131072, 262144, 256, 19456, 4, 2048, 131104, 655360, 65600, 4210688, 1280)
CANARY = b'\xa5' * 16
RUNTIME_INVERSE_STORE_SHA = 'b4cfb0a2807550f924e4097709afa069a636acb3840890d6e718f96fe9de5dec'
PROVIDERS = {'production-unrotated': 'cold_golden', 'out0': 'cold_golden', 'out1': 'private_debug', 'out2': 'private_debug', 'out3': 'deployed_candidate'}


def need(ok, message):
    if not ok: raise ValueError(message)


def parse(data):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'Duplicate JSON key: ' + key); result[key] = value
        return result
    def real(value):
        value = float(value); need(math.isfinite(value), 'Nonfinite JSON number'); return value
    def bad(value): raise ValueError('Nonfinite JSON literal: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_float=real, parse_constant=bad)


def read(path):
    path = Path(path); need(path.is_file() and not path.is_symlink(), 'Missing/unsafe file: ' + str(path))
    return path.read_bytes()


def load(path): return parse(read(path))
def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(read(path))


def canon(value):
    need(type(value) is str and value.startswith('/'), 'Absolute original path required')
    value = posixpath.normpath(value)
    return '/private' + value if value == '/tmp' or value.startswith('/tmp/') else value


def native_bit(value):
    need(type(value) is bool or type(value) is int and value in (0, 1), 'Invalid native Boolean')
    return bool(value)


def integer(value, expected, label):
    need(type(value) is int and value == expected, 'Integer declaration differs: ' + label)


def finite_words(data, bf16=False):
    need(len(data) % 4 == 0, 'F32 payload alignment')
    words = array.array('I'); words.frombytes(data)
    if sys.byteorder != 'little': words.byteswap()
    need(all((u & 0x7f800000) != 0x7f800000 and u != 0xa5a5a5a5 and
             (not bf16 or u & 0xffff == 0) for u in words), 'Nonfinite/poison/non-BF16 F32 payload')


class Payloads:
    def __init__(self, root, files):
        need(type(files) is dict and bool(files), 'Missing complete file manifest')
        self.root, self.files, self.checked, self.bytes = root, files, set(), 0
        need({p.name for p in root.iterdir()} == set(files) | {'artifact-manifest.json'}, 'Exact raw file inventory differs')

    def get(self, name, length):
        need(type(name) is str and Path(name).name == name and name not in ('.', '..'), 'Unsafe payload name')
        need(name in self.files, 'Missing complete payload: ' + name)
        item = self.files[name]
        need(type(item) is dict and type(item.get('bytes')) is int and item['bytes'] == length and
             type(item.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Payload declared size/SHA differs: ' + name)
        data = read(self.root / name)
        need(len(data) == length and sha(data) == item['sha256'], 'Complete payload actual size/SHA differs: ' + name)
        if name not in self.checked: self.checked.add(name); self.bytes += len(data)
        return data

    def guarded(self, name, length, bf16=False):
        data = self.get(name, length + 32)
        need(data[:16] == data[-16:] == CANARY, 'Complete guards differ: ' + name)
        body = data[16:-16]; finite_words(body, bf16); return body


def extract(source, name, prefix='static inline void '):
    start = source.index(prefix + name + '('); brace = source.index('{', start); depth = 1; end = brace + 1
    while depth:
        need(end < len(source), 'Unclosed source helper')
        depth += (source[end] == '{') - (source[end] == '}'); end += 1
    return source[start:end]


def source_contract(root, identity, deployment):
    inputs = identity.get('input_sha256')
    required = {'metal/flash_attn.metal', 'metal/dsv41.metal', 'src/myds_v41_gpu.c',
                'src/myds_metal.m', 'tests/test_attention_rope_fuse.m', 'tests/test_attention_packed.m',
                'tools/build_attention_rope_fuse_probe.py', 'tools/build_metallib.py'}
    need(type(inputs) is dict and required <= set(inputs), 'Complete builder source identity missing')
    for name, digest in inputs.items():
        need(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source path')
        need(file_sha(root / name) == digest, 'Frozen source bytes differ: ' + name)
    deployed_sources = deployment.get('source_sha256')
    need(type(deployed_sources) is dict, 'Deployment source freeze missing')
    for name in inputs:
        if name.startswith('metal/'):
            need(deployed_sources.get(name) == inputs[name], 'Deployed/private shader source identity differs: ' + name)
    source = read(root / 'metal/flash_attn.metal').decode()
    original = extract(source, 'ds4_flash_attn_vec_packed_register_reduce_f16_512')
    candidate = extract(source, 'myds_attention_register_rope_fuse_impl')
    store = '''                const uint oc = (uint)quadrant * (uint)NW + (uint)out_lane;
                const float4 value = reduced * reduce_inv;
                dst4[oc] = myds_attention_rope_store(value, oc, rope);'''
    restored = candidate.replace('myds_attention_register_rope_fuse_impl', 'ds4_flash_attn_vec_packed_register_reduce_f16_512', 1)
    restored = restored.replace('        constant myds_attention_rope_fuse_args & rope,\n', '', 1)
    restored = restored.replace(store, '                dst4[quadrant * NW + out_lane] = reduced * reduce_inv;')
    need(sha(original.encode()) == 'fec1cfdf82104a001088c2fa38185f24fd0d1e40f85afdcb0fd509c48131c54d', 'Immutable original register helper changed')
    need(file_sha(root / 'metal/dsv41.metal') == '8011b57ff74868e57342cc0c9db78287db8f6b8e8fa1397648eb763ec111029b', 'Immutable original rope source changed')
    need(restored == original and sha(original.encode()) == identity['original_helper_sha256'] and
         sha(candidate.encode()) == identity['candidate_helper_sha256'], 'Reduction clone differs outside final store')
    host = read(root / 'src/myds_v41_gpu.c').decode(); fixture = read(root / 'tests/test_attention_rope_fuse.m').decode()
    for text in ('sizeof(struct attention_rope_fuse_args)==336u', 'offsetof(struct attention_rope_fuse_args,pos)==200u',
                 'offsetof(struct attention_rope_fuse_args,inverse)==204u', 'offsetof(struct attention_rope_fuse_args,frequencies)==208u'):
        need(text in host, 'Host ABI contract missing: ' + text)
    need(host.count('struct attention_rope_fuse_args fused={.attention=a,.pos=rope->pos,.inverse=1u};') == 2 and
         'struct attention_rope_fuse_args args={.attention=vector_args(f->keys),.pos=pos,.inverse=1u};' in fixture, 'Direct/staged/debug inverse=1 initialization missing')
    need('uint pos, inverse;' in source and source.count('rope.inverse ? -precise::sin(') == 2,
         'Runtime inverse sign selection missing')
    helper = extract(source, 'myds_attention_rope_store', 'static inline float4 ')
    variant = store_contract(helper)
    need('fused.inverse' not in extract(source, FUSED, 'kernel void '), 'Shader inverse guard may constant-fold sign select')
    return {'original_helper_sha256': sha(original.encode()), 'candidate_helper_sha256': sha(candidate.encode()),
            'reconstructed_original_exact': True, 'args_bytes': 336, 'pos_offset': 200, 'inverse_offset': 204, 'frequencies_offset': 208,
            'imag_arithmetic_variant': variant, 'rope_store_helper_sha256': sha(helper.encode())}


def store_contract(helper):
    old = ('myds_attention_rope_bf16(re0*s0+im0*c0)', 'myds_attention_rope_bf16(re1*s1+im1*c1)')
    new = ('myds_attention_rope_bf16(precise::fma(im0,c0,re0*s0))', 'myds_attention_rope_bf16(precise::fma(im1,c1,re1*s1))')
    if all(helper.count(value) == 1 for value in old) and all(value not in helper for value in new):
        restored = helper; variant = 'runtime_inverse_expression'
    elif all(helper.count(value) == 1 for value in new) and all(value not in helper for value in old):
        restored = helper
        for before, after in zip(new, old): restored = restored.replace(before, after, 1)
        annotation = '''        /* Pin the original scalar AIR's imaginary operand order (im*c,re*s).
         * This contraction direction is a parity candidate, not M2 ISA proof. */
'''
        need(restored.count(annotation) == 1, 'Exact new-variant explanatory comment differs')
        restored = restored.replace(annotation, '', 1)
        variant = 'runtime_inverse_explicit_imag_fma'
    else: raise ValueError('Both imaginary terms must use one exact allowed arithmetic contract')
    need(sha(restored.encode()) == RUNTIME_INVERSE_STORE_SHA, 'RoPE helper changed outside the two allowed imaginary terms')
    return variant


def expected_constants(pad):
    return [(400, True, 1), (401, True, 1), (402, True, 0), (403, True, 0), (404, True, int(pad)),
            (405, True, 0), (420, False, 512), (421, False, 512), (422, False, 1), (423, False, 32)]


def library_contract(manifest, records, args, deployment, process):
    need(file_sha(args.golden_metallib) == manifest['golden_metallib_sha256'] == GOLDEN_SHA, 'Immutable production99 golden differs')
    private_sha, deployed_sha = file_sha(args.private_metallib), file_sha(args.deployed_metallib)
    need(private_sha == manifest['private_metallib_sha256'] and deployed_sha == manifest['candidate_metallib_sha256'] ==
         deployment['metallib_sha256'] == process['metallib_sha256'], 'Actual deployed/private library identity differs')
    need(private_sha != deployed_sha and deployed_sha != GOLDEN_SHA, 'Three library roles are not distinct')
    need(type(process.get('pid')) is int and process['pid'] > 0 and process.get('health') == {'status': 'ok'} and
         type(process.get('argv')) is list and process['argv'] and process['argv_raw_hex'] == [x.encode().hex() for x in process['argv']], 'Actual deployment KERN capture missing')
    need(process['binary_sha256'] == deployment['production_binary_sha256'], 'Actual deployed host/freeze binary differs')
    libraries = manifest.get('libraries'); need(type(libraries) is list and len(libraries) == 3, 'Three original library identities missing')
    need([r for r in records if r.get('kind') == 'library_identity'] == libraries, 'Original provider stdout/manifest differs')
    by_role = {r['role']: r for r in libraries}; need(set(by_role) == ROLES, 'Missing/duplicate library role')
    functions = {'cold_golden': {REGISTER, ROPE}, 'private_debug': set(DEBUG), 'deployed_candidate': {FUSED}}
    shas = {'cold_golden': GOLDEN_SHA, 'private_debug': private_sha, 'deployed_candidate': deployed_sha}
    for role, item in by_role.items():
        need(item.get('kind') == 'library_identity' and item['sha256'] == shas[role] and
             type(item.get('selected_functions')) is list and len(item['selected_functions']) == len(functions[role]) and
             set(item['selected_functions']) == functions[role], 'Library selected functions differ: ' + role)
        canon(item['path'])
        actual_file = {'cold_golden': args.golden_metallib, 'private_debug': args.private_metallib, 'deployed_candidate': args.deployed_metallib}[role]
        integer(item['bytes'], len(read(actual_file)), 'complete library file size')
        bindings = item.get('pipeline_bindings'); need(type(bindings) is list and bindings, 'Actual pipeline bindings missing: ' + role)
        # Individual ordered-constant/cache/geometry fields are checked below;
        # this list cannot be supplied by an unrelated global g_library getter.
        validate_bindings(role, bindings, functions[role])
    need(canon(by_role['deployed_candidate']['path']) == canon(process['metallib']), 'Candidate PSO library is not actual deployed path')
    need(canon(manifest['loaded_runtime_library_path']) == canon(by_role['private_debug']['path']), 'Private staging/debug runtime library mislabeled')
    observed_inputs = [r for r in records if r.get('kind') == 'library_input']
    need(len(observed_inputs) == 3, 'Three original preinit library inputs missing')
    for original_input, library in zip(observed_inputs, libraries):
        need(original_input == dict(kind='library_input', role=library['role'], path=library['path'], sha256=library['sha256'], bytes=library['bytes'], before_metal_init=True), 'Original preinit library input differs')
    return by_role


def validate_bindings(role, bindings, functions):
    """Strict producer schema; updated only when the actual fixture is read."""
    seen = set()
    for item in bindings:
        need(type(item) is dict and item.get('symbol') in functions, 'Unexpected bound PSO')
        symbol = item['symbol']; constants = item.get('ordered_constants')
        need(type(constants) is list, 'Ordered PSO constants missing')
        values = []
        for value in constants:
            need(type(value) is dict and type(value.get('index')) is int and type(value.get('value')) is int, 'PSO constant types differ')
            values.append((value['index'], native_bit(value.get('is_bool')), value['value']))
        if symbol == ROPE:
            need(not values, 'Original rope has no function constants'); key = (symbol, None); threads, dynamic = 32, 0
        else:
            need(values in [expected_constants(False), expected_constants(True)], 'Exact ordered PSO constants differ')
            key = (symbol, bool(values[4][2])); threads, dynamic = 256, 22400
        need(key not in seen, 'Duplicate pipeline binding'); seen.add(key)
        need(item['threads'] == [threads, 1, 1] and all(type(v) is int for v in item['threads']), 'Pipeline thread dimensions differ')
        integer(item['dynamic_threadgroup_bytes'], dynamic, 'pipeline dynamic bytes')
        integer(item['thread_execution_width'], 32, 'SIMD width')
        integer(item['args_bytes'], 152 if symbol == ROPE else 200 if symbol == REGISTER else 336, 'bound PSO ABI')
        need(type(item['max_threads']) is int and item['max_threads'] >= threads and type(item['static_threadgroup_bytes']) is int and item['static_threadgroup_bytes'] >= 0 and native_bit(item['dictionary_matches_bound_pipeline']) and native_bit(item['resolver_matches_bound_pipeline']), 'Actual pipeline geometry/cache identity invalid')
        applicable = symbol != ROPE and len(symbol) < 64
        need(native_bit(item['specialized_cache_applicable']) == applicable and native_bit(item['specialized_cache_matches_bound_pipeline']) == applicable, 'Specialized cache pointer identity differs')
        cache_key = symbol + ''.join(f':{index}:{int(boolean)}:{value}' for index, boolean, value in values)
        need(item['cache_key'] == cache_key, 'Exact ordered cache key differs')
    expected = {(symbol, None) if symbol == ROPE else (symbol, pad) for symbol in functions for pad in ((False, True) if symbol != ROPE else (None,))}
    need(seen == expected, 'Missing selected pad variant/PSO binding: ' + role)


def seeded_values(count, seed):
    output = bytearray(count * 4)
    for i in range(count):
        x = (i * 747796405 + seed * 2891336453) & 0xffffffff
        x = (((x >> ((x >> 28) + 4)) ^ x) * 277803737) & 0xffffffff; x = (x >> 22) ^ x
        struct.pack_into('<f', output, i * 4, ((x % 255) - 127) / 128)
    return bytes(output)


def frequencies(compressed):
    libc = ctypes.CDLL(None); libc.powf.argtypes = (ctypes.c_float, ctypes.c_float); libc.powf.restype = ctypes.c_float
    f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
    base = 160000.0 if compressed else 10000.0
    low = math.floor(64 * math.log(65536 / (32 * 2 * math.pi)) / (2 * math.log(base)))
    high = math.ceil(64 * math.log(65536 / (2 * math.pi)) / (2 * math.log(base)))
    result = []
    for i in range(32):
        frequency = f32(1 / libc.powf(base, i / 32))
        if compressed:
            ramp = min(1., max(0., f32(f32(i - low) / f32(high - low)))); smooth = f32(1 - ramp)
            frequency = f32(f32(f32(frequency / 16) * f32(1 - smooth)) + f32(frequency * smooth))
        result.append(frequency)
    return struct.pack('<32f', *result)


def review(args):
    manifest = load(args.raw / 'artifact-manifest.json'); identity = load(args.source_identity)
    deployment, process = load(args.deployment_freeze), load(args.deployment_identity)
    need(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 2 and manifest['status'] == 'complete', 'Only full completed provenance-v2 fixture accepted')
    need(manifest['fullmodel_speedup_proven'] is False and manifest['gpu_physical_overlap_proven'] is False and
         native_bit(manifest['quick']) is False, 'Component scope/complete120 contract differs')
    need(file_sha(args.binary) == manifest['binary_sha256'] == identity['output_sha256']['test_attention_rope_fuse'] and
         file_sha(args.source_identity) == manifest['source_identity_sha256'] and
         file_sha(args.private_metallib) == identity['output_sha256']['myds.metallib'], 'Producer/private builder identities differ')
    proof = source_contract(args.source_root, identity, deployment)
    records = [parse(line) for line in read(args.stdout).splitlines() if line.startswith(b'{')]
    need(all(type(r) is dict and r.get('kind') in ('host_contract', 'fixture_input', 'library_input', 'library_identity', 'rejection', 'diagnostic_case', 'parity', 'timing', 'complete') for r in records), 'Unexpected producer JSON record')
    fixture_inputs = [r for r in records if r.get('kind') == 'fixture_input']
    need(fixture_inputs == [manifest['fixture_input']] and manifest['fixture_input']['before_metal_init'] is True and
         manifest['fixture_input']['binary_sha256'] == manifest['binary_sha256'] and manifest['fixture_input']['source_identity_sha256'] == manifest['source_identity_sha256'] and
         canon(manifest['fixture_input']['binary_path']) == canon(manifest['binary_path']) and canon(manifest['fixture_input']['source_identity_path']) == canon(manifest['source_identity_path']), 'Actual preinit producer/source identity differs')
    for record in (manifest, manifest['fixture_input']):
        integer(record['args_bytes'], 336, 'common ABI'); integer(record['inverse'], 1, 'common inverse')
    need(manifest['output_providers'] == PROVIDERS, 'Common output library providers differ')
    libraries = library_contract(manifest, records, args, deployment, process)
    need([r['kind'] for r in records[:8]] == ['host_contract', 'fixture_input'] + ['library_input'] * 3 + ['library_identity'] * 3 and
         records[-1]['kind'] == 'complete', 'Actual producer initialization/completion order differs')
    host_record = records[0]
    need(host_record['cpu_only'] is True and host_record['strict_flag'] is True, 'Producer host contract missing')
    integer(host_record['invalid_null_dispatch_delta'], 0, 'host null dispatch'); integer(host_record['args_bytes'], 336, 'host ABI')
    for kind, key in [('rejection', 'rejections'), ('diagnostic_case', 'diagnostics'), ('parity', 'cases'), ('timing', 'timings')]:
        need([r for r in records if r.get('kind') == kind] == manifest[key], 'Complete original stdout records differ: ' + kind)
    complete = [r for r in records if r.get('kind') == 'complete']
    need(len(complete) == 1 and complete[0]['pass'] is True and complete[0]['fullmodel_speedup_proven'] is False, 'Producer completion missing/overclaimed')
    integer(complete[0]['cases'], 120, 'complete case count'); integer(complete[0]['raw_files'], len(manifest['files']), 'complete raw count')
    need(len(manifest['cases']) == len(manifest['diagnostics']) == 120 and len(manifest['rejections']) == 11, 'Full120/11 predispatch gates missing')
    data = Payloads(args.raw, manifest['files'])
    for i, record in enumerate(manifest['rejections']):
        need(record['case'] == REJECTIONS[i] and record['all_observed_buffers_full_bytes_unchanged'] is True, 'Negative identity/byte declaration differs')
        for key, value in [('dispatch_delta', 0), ('encoded_delta', 0), ('requests_delta', 1), ('failures_delta', 1)]: integer(record[key], value, key)
        for j, length in enumerate(NEGATIVE_LENGTHS):
            need(data.get(f'reject{i}-buffer{j}-before.raw', length) == data.get(f'reject{i}-buffer{j}-after.raw', length), 'Entire negative buffer changed')
    samples, repetitions = manifest['samples'], manifest['repetitions']
    need(type(samples) is int and 3 <= samples <= 31 and type(repetitions) is int and 1 <= repetitions <= 64, 'Component timing protocol differs')
    timing = native_bit(manifest['timing_enabled']); times = {}; summaries = []; seeds = {}; freq = {v: frequencies(v) for v in (False, True)}
    expected_case = [(p, c, d, mask) for p in range(6) for c in (False, True) for d in (False, True) for mask in range(5)]
    expected_kinds = ['host_contract', 'fixture_input'] + ['library_input'] * 3 + ['library_identity'] * 3 + ['rejection'] * 11
    for _ in expected_case: expected_kinds += ['diagnostic_case', 'parity'] + (['timing'] * (3 * (samples + 1)) if timing else [])
    expected_kinds += ['complete']
    need([r['kind'] for r in records] == expected_kinds, 'Missing/duplicate/out-of-order complete producer record set')
    for i, (record, diagnostic, (p, compressed, direct, mask)) in enumerate(zip(manifest['cases'], manifest['diagnostics'], expected_case)):
        for key, value in [('case', i), ('pos', POSITIONS[p]), ('keys', KEY_COUNTS[p]), ('mask_mode', mask), ('raw_start', 127), ('inverse', 1), ('args_bytes', 336), ('requests_delta', 1), ('encoded_delta', 1), ('failures_delta', 0)]: integer(record[key], value, key)
        need(native_bit(record['compressed']) == compressed and native_bit(record['direct']) == direct and record['output_providers'] == PROVIDERS, 'Case path/library providers differ')
        need(all(record[key] is True for key in ('full_output_same', 'partial_stats_full_bytes_same', 'finite', 'guards', 'production_golden')), 'Case exact numerical gate missing')
        nraw = min(KEY_COUNTS[p], 128); selected = KEY_COUNTS[p] - nraw; source_rows = 2 * selected + 17 if selected else 0
        for key, value in [('n_raw', nraw), ('selected', selected), ('source_rows', source_rows)]: integer(record[key], value, key)
        prefix = f'case{i}'; golden = data.guarded(prefix + '-out0.guarded', 131072, True)
        if mask != 2 and not (mask == 3 and KEY_COUNTS[p] <= 32):
            need(any(u[0] & 0x7fffffff for u in struct.iter_unpack('<I', golden)), 'All-zero output despite active seeded attention rows')
        need(data.guarded(prefix + '-out1.guarded', 131072) == data.guarded(prefix + '-production-unrotated.guarded', 131072), 'Debug baseline differs from immutable cold production unrotated')
        need(data.guarded(prefix + '-out2.guarded', 131072, True) == golden and data.guarded(prefix + '-out3.guarded', 131072, True) == golden, 'Private or actual deployed fused complete output differs')
        for suffix, length in [('partial', 4194304), ('stats', 24576)]:
            need(data.guarded(prefix + '-' + suffix + '0.guarded', length) == data.guarded(prefix + '-' + suffix + '1.guarded', length), 'Complete debug intermediate bytes differ')
        exact_diff = dict(same=1, different_words=0, first_index=-1, reference_bits='none', candidate_bits='none')
        for key in ('reference_debug_unrotated', 'production_fused', 'debug_fused', 'partial', 'stats'):
            diff = diagnostic[key]; need(native_bit(diff['same']) and {**diff, 'same': 1} == exact_diff, 'Actual diagnostic raw comparison differs')
        need(diagnostic['case'] == i and diagnostic['actual_production_fused_ran_before_debug'] is True and diagnostic['case_exact_gate_passed'] is True and diagnostic['guards_finite_poison_bf16_valid'] is True and diagnostic['candidate_accepted'] is False and diagnostic['diagnostic_timings_accepted'] is False, 'Diagnostic run order/exact/scope contract differs')
        for key in ('pos', 'keys', 'mask_mode', 'compressed', 'direct', 'inverse', 'args_bytes', 'requests_delta', 'encoded_delta', 'failures_delta', 'output_providers'):
            need(type(diagnostic[key]) is type(record[key]) and diagnostic[key] == record[key], 'Diagnostic/parity case metadata differs')
        if mask == 4: need(diagnostic['mask_description'] == 'unmasked_no_effective_sink', 'Mode4 mislabeled')
        for suffix, count, seed in [('q.f32', 32768, 19 + mask), ('window.f32', 65536, 37 + mask)]:
            cache = (count, seed)
            if cache not in seeds: seeds[cache] = seeded_values(count, seed)
            need(data.get(prefix + '-' + suffix, count * 4) == seeds[cache], 'Entire seeded F32 input differs')
        sinks = struct.pack('<64f', *([-struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0] / 2] * 64 if mask == 4 else [(h - 31) / 8 for h in range(64)]))
        need(data.get(prefix + '-sinks.f32', 256) == sinks, 'Entire sink input differs')
        masks = struct.pack('<640H', *[0xfbff if mask == 2 or mask == 1 and j % 3 == 1 or mask == 3 and j < 32 else 0xbc00 if mask == 1 else 0 for j in range(640)])
        need(data.get(prefix + '-mask.f16', 1280) == masks and data.get(prefix + '-frequencies.f32', 128) == freq[compressed], 'Entire mask/frequency input differs')
        if selected:
            ids = struct.unpack('<' + str(selected) + 'i', data.get(prefix + '-ids.i32', selected * 4))
            need(list(ids) == [(j * 53 + 19) % source_rows for j in range(selected)] and len(set(ids)) == selected, 'Complete selected IDs order/uniqueness/range differs')
            cache = (source_rows * 512, 71 + mask)
            if cache not in seeds: seeds[cache] = seeded_values(*cache)
            expected_comp = b''.join(seeds[cache][j + 2:j + 4] for j in range(0, len(seeds[cache]), 4))
            need(data.get(prefix + '-comp.bf16', source_rows * 1024) == expected_comp, 'Entire source BF16 input differs')
            gathered = b''.join(seeds[cache][id * 2048:(id + 1) * 2048] for id in ids)
            need(data.get(prefix + '-gathered.f32', selected * 2048) == gathered, 'Entire selected gathered F32 input differs')
        case_times = [r for r in manifest['timings'] if r['case'] == i]
        expected_keys = {(arm, sample) for arm in ('A1', 'B', 'A2') for sample in range(samples + 1)} if timing else set()
        seen = set()
        for item in case_times:
            integer(item['case'], i, 'timing case')
            arm, sample = item['arm'], item['sample']; key = (arm, sample)
            need(type(sample) is int and key in expected_keys and key not in seen, 'Unexpected/duplicate timing'); seen.add(key)
            need(native_bit(item['warmup']) == (sample == 0) and all(item[k] is True for k in ('full_output_same', 'guards', 'finite')), 'Timing full output gate missing')
            integer(item['repetitions'], repetitions, 'timing repetitions'); integer(item['fused_encoded'], repetitions if arm == 'B' else 0, 'timing actual adoption')
            base = (2 if direct else 2 + (nraw > 128 - 127)) + (selected > 0) + (KEY_COUNTS[p] % 32 != 0)
            integer(item['dispatches'], repetitions * (base + (arm != 'B')), 'complete pair dispatches')
            start, end = item['gpu_start_s'], item['gpu_end_s']
            need(type(start) in (int, float) and type(end) in (int, float) and math.isfinite(start) and math.isfinite(end) and end > start, 'Invalid original GPU timestamp')
            us = (end - start) * 1e6 / repetitions
            need(type(item['gpu_us']) in (int, float) and math.isclose(us, item['gpu_us'], rel_tol=1e-12, abs_tol=1e-9), 'Complete pair timing normalization differs')
            need(data.guarded(f'{prefix}-timing-{arm}-s{sample}.guarded', 131072, True) == golden, 'Entire timed output differs')
            times[i, arm, sample] = us
        need(seen == expected_keys, 'Missing formal/warm component timing')
        if timing:
            values = [statistics.median(times[i, arm, sample] for sample in range(1, samples + 1)) for arm in ('A1', 'B', 'A2')]
            summaries.append(dict(case=i, median_component_gpu_us=values, midpoint_time_saved_percent=100 * (1 - values[1] / ((values[0] + values[2]) / 2)), conservative_time_saved_percent=100 * (1 - values[1] / min(values[0], values[2])), control_drift_percent=100 * (values[2] / values[0] - 1)))
    need(len(manifest['timings']) == len(times), 'Timing records outside complete case set')
    need(data.checked == set(data.files), 'Unexpected/unreviewed payload files')
    return dict(cpu_only=True, pass_=True, cases=120, negative_controls=11, raw_files=len(data.checked), raw_bytes=data.bytes,
        all_complete_payload_sha_size_recomputed=True, all_out0_3_unrotated_partial_stats_bytes_exact=True,
        actual_deployed_candidate_vs_immutable_production99_exact=True, all_seeded_inputs_reconstructed=True,
        golden_metallib_sha256=GOLDEN_SHA, deployed_metallib_sha256=file_sha(args.deployed_metallib), private_metallib_sha256=file_sha(args.private_metallib),
        libraries=libraries, source_contract=proof, component_timing_records=len(times), component_summaries=summaries,
        original_stdout_sha256=file_sha(args.stdout), manifest_sha256=file_sha(args.raw / 'artifact-manifest.json'),
        source_identity_sha256=file_sha(args.source_identity), deployment_identity_sha256=file_sha(args.deployment_identity), deployment_freeze_sha256=file_sha(args.deployment_freeze),
        reader_sha256=file_sha(Path(__file__)), production_feature_adoption_proven=False, fullmodel_speedup_proven=False,
        gpu_physical_overlap_proven=False, scope='Complete seeded component bytes and actual deployed candidate PSO provenance only. Private host allowance is distinct from production feature enabling. No full-model logits/tokens/KV/HC state or decode speed claim; component GPU timestamps are original producer observations, not CPU-measured GPU work.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'stdout', 'source-identity', 'source-root', 'binary', 'golden-metallib', 'private-metallib', 'deployed-metallib', 'deployment-identity', 'deployment-freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); need(not args.out.exists(), 'Never overwrite a review')
    result = review(args); args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(pass_=True, cases=120, output=str(args.out))))


if __name__ == '__main__': main()
