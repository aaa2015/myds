#!/usr/bin/env python3
"""Independent, CPU-only full-payload gate for register-only indexer MMA.

No builder/harness imports, no Metal API, and no performance interpretation.
A prospective protocol must bind the exact binary, source, full-TU build and
three libraries.  Actual-exact versus TILED differences are descriptive only;
three TILED controls/candidate must match, and the exact identity bridge must
match its separate deployed exact control.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
CASES = ((8, 513, 0, 0), (9, 543, 0, 1), (15, 1025, 1024, 1),
         (31, 519, 30720, 2), (33, 1023, 0, 2), (8, 16383, 32760, 0),
         (9, 16384, 32759, 1), (15, 513, 1000, 0), (128, 15424, 30720, 0),
         (2048, 513, 0, 1), (8, 1025, 2042, 2), (9, 543, 1092, 2))
SYMBOLS = ('kernel_dsv4_indexer_scores_tiled_bf16',) * 3 + (
    'kernel_myds_indexer_scores_register_only_bf16',
    'kernel_myds_indexer_score_chunk_exact', 'kernel_myds_indexer_score_chunk_exact')
LIBRARIES = (0, 1, 2, 2, 0, 1)
TG_BYTES = (21504, 21504, 21504, 0, 5120, 5120)
ARTIFACTS = ('harness', 'reader', 'binary', 'build_identity', 'deployed_library',
             'native_library', 'probe_library', 'host_fixture_manifest')
FIELDS = ('n_comp', 'n_tokens', 'n_head', 'head_dim', 'pos0', 'ratio',
          'q_token_stride', 'q_head_stride', 'weights_token_stride',
          'index_row_stride', 'score_token_stride', 'scale')
DATA_PROFILES = ('bf16_representable_f32', 'full_f32_signed_zero', 'full_f32_signed_zero_subnormal')
POISON = b'\xa5'
GUARD = POISON * 64
MARKER = b'\n#line 1 "myds_indexer_register_only.metal.inc"\n'


class EvidenceError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise EvidenceError(message)


def integer(v, minimum=0, maximum=(1 << 64)-1):
    need(type(v) is int and minimum <= v <= maximum, 'strict bounded integer')
    return v


def boolean(v):
    need(type(v) is bool, 'strict boolean')
    return v


def exact(actual, expected, label):
    """Python True == 1 is deliberately insufficient for producer contracts."""
    if isinstance(expected, dict):
        need(type(actual) is dict and set(actual) == set(expected), label + ': object keys')
        for k, v in expected.items():
            exact(actual[k], v, label + '.' + k)
    elif isinstance(expected, list):
        need(type(actual) is list and len(actual) == len(expected), label + ': list')
        for i, (a, e) in enumerate(zip(actual, expected)):
            exact(a, e, label + f'[{i}]')
    else:
        need(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def strict_pairs(items):
    d = {}
    for k, v in items:
        need(k not in d, 'duplicate JSON key')
        d[k] = v
    return d


def load_json(path):
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=strict_pairs,
                          parse_constant=lambda s: (_ for _ in ()).throw(EvidenceError('nonfinite JSON')))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise EvidenceError(str(e)) from e


def alias_name(path):
    s = str(path)
    return '/tmp/' + s[len('/private/tmp/'):] if s.startswith('/private/tmp/') else s


class Paths:
    def __init__(self, mappings=()):
        self.mappings = sorted(((str(a).rstrip('/'), Path(b)) for a, b in mappings), key=lambda p: -len(p[0]))

    def resolve(self, raw):
        need(type(raw) is str and Path(raw).is_absolute(), 'absolute provenance path')
        for prefix, dest in self.mappings:
            if raw == prefix or raw.startswith(prefix + '/'):
                return dest / raw[len(prefix):].lstrip('/')
        return Path(raw)

    def descriptor(self, d):
        need(type(d) is dict and {'path', 'bytes', 'sha256'} <= set(d), 'artifact descriptor')
        n = integer(d['bytes'], 1)
        need(type(d['sha256']) is str and len(d['sha256']) == 64, 'artifact SHA')
        p = self.resolve(d['path'])
        need(p.is_file() and not p.is_symlink(), 'regular frozen artifact: ' + str(p))
        b = p.read_bytes()
        need(len(b) == n and sha(b) == d['sha256'], 'frozen artifact length/SHA: ' + str(p))
        return p, b


class Payloads:
    def __init__(self, root, records):
        self.root, self.records, self.used = Path(root), {}, set()
        need(type(records) is list, 'files list')
        for r in records:
            need(type(r) is dict and set(r) in ({'path', 'bytes', 'sha256'},
                 {'path', 'bytes', 'sha256', 'storage_alias_of'}), 'raw inventory exact fields')
            p = r['path']
            need(type(p) is str and Path(p).name == p and p not in ('.', '..') and not p.startswith('.'), 'local raw filename')
            need(p not in self.records, 'duplicate raw filename')
            integer(r['bytes'], 1)
            need(type(r['sha256']) is str and len(r['sha256']) == 64, 'raw SHA string')
            self.records[p] = r
        for p, r in self.records.items():
            full = self.root / p
            need(full.is_file() and not full.is_symlink(), 'regular raw file: ' + p)
            b = full.read_bytes()
            need(len(b) == r['bytes'] and sha(b) == r['sha256'], 'raw bytes/SHA: ' + p)
            if 'storage_alias_of' in r:
                a = r['storage_alias_of']
                need(type(a) is str and a != p and a in self.records and
                     self.records[a]['bytes'] == r['bytes'] and self.records[a]['sha256'] == r['sha256'], 'valid identical payload alias')

    def read(self, d):
        need(type(d) is dict and type(d.get('path')) is str and d['path'] in self.records, 'referenced raw descriptor')
        exact(d, self.records[d['path']], 'raw descriptor matches inventory')
        self.used.add(d['path'])
        return (self.root / d['path']).read_bytes()

    def guarded(self, d, n):
        need(type(d) is dict and set(d) == {'body', 'prefix_guard', 'suffix_guard'}, 'guarded payload fields')
        b, pre, post = (self.read(d[k]) for k in ('body', 'prefix_guard', 'suffix_guard'))
        need(len(b) == n and len(pre) == len(post) == 64, 'guard/body exact extents')
        return b, pre, post

    def finish(self):
        need(self.used == set(self.records), 'every inventory payload referenced exactly by evidence')
        actual = {p.name for p in self.root.iterdir()}
        need(actual == set(self.records) | {'artifact-manifest.json'}, 'complete raw directory inventory')


def case_description(case):
    nt, nk, pos, pad = case
    qh = 512 + (16 if pad == 2 else 0)
    return dict(n_comp=nk, n_tokens=nt, n_head=32, head_dim=128, pos0=pos, ratio=2,
                q_token_stride=32*qh + (16 if pad else 0), q_head_stride=qh,
                weights_token_stride=128 + (16 if pad else 0),
                index_row_stride=256 + (16 if pad else 0),
                score_token_stride=4*(nk + (7 if pad else 0)), scale=1/64)


def scope(a):
    need(type(a) is dict and set(a) == set(FIELDS), 'argument fields')
    for k in FIELDS[:-1]:
        integer(a[k], 0, (1 << (64 if 'stride' in k else 32))-1)
    need(type(a['scale']) is float and a['scale'] == 1/64, 'exact F32 scale')
    need(a['n_head'] == 32 and a['head_dim'] == 128 and a['ratio'] == 2 and
         512 < a['n_comp'] <= 16384 and 8 <= a['n_tokens'] <= 2048 and
         a['pos0'] + a['n_tokens'] <= 32768, 'narrow BF16 prefill scope')
    q, h, w, k, s = (a[n] for n in FIELDS[6:11])
    need(all(stride <= ((1<<64)-129)//rows for stride,rows in ((q,a['n_tokens']),(w,a['n_tokens']),(k,a['n_comp']),(s,a['n_tokens']))), 'full guarded span multiplication cannot overflow')
    need(h >= 512 and h % 4 == 0 and q >= 31*h+512 and q % 4 == 0 and
         w >= 128 and w % 4 == 0 and k >= 256 and k % 2 == 0 and
         s >= a['n_comp']*4 and s % 4 == 0, 'aligned nonoverlapping strides')


def mismatch(a, b):
    need(len(a) == len(b) and len(a) % 4 == 0, 'equal F32 comparison extents')
    return sum(x != y for (x,), (y,) in zip(struct.iter_unpack('<I', a), struct.iter_unpack('<I', b)))


def check_inputs(before, a, profile):
    """Independently check finite logical data, all poison padding, and dense pack."""
    nt, nk = a['n_tokens'], a['n_comp']
    packed, features = [], []
    for role, rows, cols, elem, stride in ((0, nt, 32, 512, a['q_token_stride']),
          (1, nt, 1, 128, a['weights_token_stride']), (2, nk, 1, 256, a['index_row_stride'])):
        b = before['strided'][role][0]
        expected = bytearray(b'\xa5' * len(b))
        pieces = []
        for r in range(rows):
            for c in range(cols):
                off = r*stride + (c*a['q_head_stride'] if role == 0 else 0)
                piece = b[off:off+elem]
                expected[off:off+elem] = piece
                pieces.append(piece)
                if role < 2:
                    need(all(math.isfinite(v) for (v,) in struct.iter_unpack('<f', piece)), 'finite input F32')
                else:
                    need(all((v & 0x7f80) != 0x7f80 for (v,) in struct.iter_unpack('<H', piece)), 'finite BF16 input')
        need(bytes(expected) == b, 'all input row/head padding remains A5')
        need(any(v != 0 for piece in pieces for v in piece), 'nonzero logical fixture')
        pack = b''.join(pieces)
        need(pack == before['dense'][role][0], 'complete dense pack is exact logical input')
        words = struct.iter_unpack('<I' if role < 2 else '<H', pack)
        exp = 0x7f800000 if role < 2 else 0x7f80
        sign = 0x80000000 if role < 2 else 0x8000
        mant = 0x007fffff if role < 2 else 0x007f
        f = dict(positive_zero_count=0, negative_zero_count=0, subnormal_count=0, non_bf16_representable_f32_count=0)
        for (v,) in words:
            f['positive_zero_count'] += v == 0
            f['negative_zero_count'] += v == sign
            f['subnormal_count'] += (v & exp) == 0 and (v & mant) != 0
            f['non_bf16_representable_f32_count'] += role < 2 and v & 0xffff != 0
        if role < 2 and profile == 0:
            need(f['non_bf16_representable_f32_count'] == 0, 'BF16 representable F32 profile')
        if profile in (1, 2) and role < 2:
            need(f['non_bf16_representable_f32_count'] > 0 and f['positive_zero_count'] > 0 and f['negative_zero_count'] > 0, 'actual full-F32/signed-zero profile')
        if profile == 2:
            need(f['subnormal_count'] > 0 and f['positive_zero_count'] > 0 and f['negative_zero_count'] > 0, 'actual finite subnormal/signed-zero profile')
        features.append(f)
        packed.append(pack)
    return features


def score_result(data, stride, a):
    nt, nk = a['n_tokens'], a['n_comp']
    need(len(data) == nt*stride, 'complete score table extent')
    logical, padding, causal, finite, written, invisible = [], True, True, True, True, 0
    for t in range(nt):
        row = data[t*stride:t*stride+nk*4]
        logical.append(row)
        padding &= data[t*stride+nk*4:(t+1)*stride] == POISON*(stride-nk*4)
        visible = min((a['pos0']+t+1)//2, nk)
        for k, (v,) in enumerate(struct.iter_unpack('<I', row)):
            written &= v != 0xa5a5a5a5
            if k >= visible:
                invisible += 1
                causal &= v == 0xff800000
            else:
                finite &= v & 0x7f800000 != 0x7f800000
    return b''.join(logical), dict(padding_unchanged=bool(padding), causal_tail_valid=bool(causal),
           visible_finite=bool(finite), all_scores_written=bool(written), invisible_minus_inf_count=invisible)


def validate_unique_case_paths(c, host_only=False):
    """Each before/post/output/guard is its own named observation. Identical
    bytes may be hardlinked by the producer, but never collapse descriptors
    into a reused earlier observation filename.
    """
    named = [c['args72']['path'],c['exact_args36']['path']]
    def take(d):
        named.extend(d[k]['path'] for k in ('body','prefix_guard','suffix_guard'))
    for layout in ('strided','dense'):
        for d in c['inputs_before'][layout]:take(d)
    if not host_only:
        for pr in c['providers']:
            take(pr['scores'])
            for d in pr['inputs_after']:take(d)
    expected=20 if host_only else 92
    need(len(named) == expected and all(type(n) is str for n in named) and len(set(named)) == expected, 'distinct filename for every complete case observation (before/post/scores/guards)')


def review_case(c, payloads, expected, ci, host_only=False):
    need(type(c) is dict, 'case object')
    validate_unique_case_paths(c, host_only)
    exact(c['case'], ci, 'case ordinal')
    exact(c['padding_profile'], expected[3], 'frozen padding profile')
    exact(c['data_profile'], DATA_PROFILES[ci % 3], 'frozen actual data profile')
    exact(c['cache_type'], 'BF16', 'explicit BF16 cache')
    a = c['args']; scope(a); exact(a, case_description(expected), 'frozen case/strides')
    raw = payloads.read(c['args72'])
    need(len(raw) == 72 and raw[68:] == b'\0'*4, 'ABI72 trailing padding zero')
    vals = struct.unpack('<6I5Qf4x', raw)
    exact(dict(zip(FIELDS, vals)), a, 'ABI72 full decoded arguments')
    e = payloads.read(c['exact_args36'])
    expected_exact = (a['n_comp'], 32, 128, 2, 1/64, a['n_tokens'], a['pos0'], 2, 0)
    need(len(e) == 36 and struct.unpack('<4If4I', e) == expected_exact, 'distinct exact ABI36 cache_type2')
    sizes = {'strided': (a['n_tokens']*a['q_token_stride'], a['n_tokens']*a['weights_token_stride'], a['n_comp']*a['index_row_stride']),
             'dense': (a['n_tokens']*16384, a['n_tokens']*128, a['n_comp']*256)}
    need(type(c['inputs_before']) is dict and set(c['inputs_before']) == set(sizes), 'both input layouts retained')
    before = {}
    for layout, sizes_ in sizes.items():
        need(type(c['inputs_before'][layout]) is list and len(c['inputs_before'][layout]) == 3, 'three full input roles')
        before[layout] = [payloads.guarded(d, n) for d, n in zip(c['inputs_before'][layout], sizes_)]
        need(all(pre == post == GUARD for b, pre, post in before[layout]), 'initial input guards A5')
    features = check_inputs(before, a, ci % 3)
    if host_only:
        exact(c['providers'], [], 'host-only has no GPU outputs')
        return dict(case=ci, input_features=features, numerical_passed=False)
    need(type(c['providers']) is list and len(c['providers']) == 6, 'six providers all retained')
    logical, results = [], []
    for p, pr in enumerate(c['providers']):
        exact(pr['provider'], p, 'provider order')
        exact(pr['completion_ordinal'], ci*6+p+1, 'global completion ordinal')
        boolean(pr['gpu_completed'])
        exact(pr['symbol'], SYMBOLS[p], 'selected symbol')
        exact(pr['library_index'], LIBRARIES[p], 'selected library index')
        layout = 'strided' if p < 4 else 'dense'
        exact(pr['input_layout'], layout, 'provider input layout')
        exact(pr['dynamic_threadgroup_bytes'], TG_BYTES[p], 'actual TG memory')
        exact(pr['threads'], [32, 4, 1], 'actual threads')
        exact(pr['grid'], [(a['n_comp']+(31 if p < 4 else 15))//(32 if p < 4 else 16), (a['n_tokens']+7)//8, 1], 'actual grid')
        stride = a['score_token_stride'] if p < 4 else a['n_comp']*4
        exact(pr['score_stride'], stride, 'score stride')
        b, pre, post = payloads.guarded(pr['scores'], a['n_tokens']*stride)
        packed, stats = score_result(b, stride, a)
        logical.append(packed)
        inputs_ok = True
        need(type(pr['inputs_after']) is list and len(pr['inputs_after']) == 3, 'all three post-input witnesses')
        for r, (d, n) in enumerate(zip(pr['inputs_after'], sizes[layout])):
            inputs_ok &= payloads.guarded(d, n) == before[layout][r]
        exact(pr['inputs_unchanged'], inputs_ok, 'recomputed full inputs/guards unchanged')
        stats['guards_intact'] = pre == post == GUARD
        stats['tiled_mismatches'] = mismatch(packed, logical[0])
        stats['exact_bridge_mismatches'] = mismatch(packed, logical[4]) if p == 5 else None
        ok = pr['gpu_completed'] and inputs_ok and all(stats[k] for k in (
            'padding_unchanged', 'causal_tail_valid', 'visible_finite', 'all_scores_written', 'guards_intact'))
        ok &= p >= 4 or stats['tiled_mismatches'] == 0
        ok &= p != 5 or stats['exact_bridge_mismatches'] == 0
        stats['numerical_passed'] = bool(ok)
        for k, v in stats.items():
            exact(pr[k], v, f'recomputed provider {p} {k}')
        results.append(dict(provider=p, **stats))
    return dict(case=ci, args=a, input_features=features, all_numerical_passed=all(r['numerical_passed'] for r in results),
                candidate_mismatches_to_deployed_tiled=results[3]['tiled_mismatches'],
                actual_exact_mismatches_to_tiled_descriptive_only=results[4]['tiled_mismatches'], providers=results)


def assemble_frozen_production(source):
    raw = (source/'tools/build_metallib.py').read_bytes()
    tree = ast.parse(raw)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('SHADER_FILES','PREAMBLE','COPY_KERNELS'):
                    need(target.id not in values, 'unique production source literal')
                    values[target.id] = ast.literal_eval(node.value)
    need(set(values) == {'SHADER_FILES','PREAMBLE','COPY_KERNELS'}, 'three production assembly literals')
    order = values['SHADER_FILES']
    need(type(order) is list and len(order) == len(set(order)) == 29 and all(type(v) is str and Path(v).name == v for v in order), 'literal ordered 29 production shaders')
    need(type(values['PREAMBLE']) is str and type(values['COPY_KERNELS']) is str, 'literal production preamble/copy strings')
    result = values['PREAMBLE'].encode()+b'\n'
    for name in order:
        result += ('\n#line 1 "'+name+'"\n').encode()+(source/'metal'/name).read_bytes()
    return result + values['COPY_KERNELS'].encode()+b'\n', order


def verify_compiled_binary(paths, build, identity, expected):
    for obj in (build, identity):
        need(type(obj.get('artifacts')) is dict and 'bench_indexer_register_only' in obj['artifacts'], 'compiled harness artifact is mandatory in both manifests')
        d = obj['artifacts']['bench_indexer_register_only']
        _, raw = paths.descriptor(d)
        need(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'] and alias_name(d['path']) == alias_name(expected['path']), 'compiled harness binary equals prospective frozen binary')


def verify_build(paths, build, artifacts):
    need(type(build) is dict and build.get('kind') == 'indexer_register_only_full_library_build', 'complete-TU build identity kind')
    exact(build['schema_version'], 1, 'build schema')
    exact(build['compile_only_pass'], True, 'successful compile-only build')
    exact(build['gpu_work_submitted'], False, 'builder submitted no GPU work')
    exact(build['selected_symbol'], SYMBOLS[3], 'builder candidate symbol')
    exact(build['native_oracle_symbol'], SYMBOLS[0], 'builder native symbol')
    for label, art in (('native', 'native_library'), ('probe', 'probe_library')):
        _, raw = paths.descriptor(build['metallibs'][label])
        need(sha(raw) == artifacts[art]['sha256'] and len(raw) == artifacts[art]['bytes'], 'build library bound to protocol')
    ip, iraw = paths.descriptor(build['source_identity']); identity = load_json(ip)
    verify_compiled_binary(paths, build, identity, artifacts['binary'])
    source = ip.parent / 'source'
    inventory = identity['source_inventory']
    need(type(inventory) is dict and len(inventory) >= 32, 'complete source inventory')
    for name, d in inventory.items():
        need(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative source inventory path')
        integer(d['bytes'], 1)
        need((source/name).is_file() and not (source/name).is_symlink(), 'regular source inventory file')
        data = (source/name).read_bytes()
        need(len(data) == d['bytes'] and sha(data) == d['sha256'], 'full frozen source inventory: ' + name)
    need('tools/bench_indexer_register_only.m' in inventory and inventory['tools/bench_indexer_register_only.m']['sha256'] == artifacts['harness']['sha256'], 'compiled harness source in immutable build inventory')
    native, combined, candidate = ((ip.parent/n).read_bytes() for n in ('native.metal', 'all.metal', 'candidate.metal.inc'))
    need(len(native) == integer(build['native_source_bytes'], 1) and sha(native) == build['native_source_sha256'] and sha(combined) == build['combined_source_sha256'], 'full TU SHA/length')
    need(combined == native + MARKER + candidate, 'candidate-only append/inverse byte proof')
    need(candidate == (source/'tools/indexer_register_only.metal.inc').read_bytes(), 'compiled candidate equals frozen candidate source')
    reconstructed, order = assemble_frozen_production(source)
    need(reconstructed == native, 'independent literal AST assembly reproduces complete production TU byte for byte')
    exact(identity['original_shader_order'], order, 'production order recorded from literal')
    canonical = json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()
    need(sha(canonical) == identity['source_inventory_sha256'], 'canonical source inventory SHA')
    records = identity['original_shaders']
    need(type(records) is list and len(records) == 29 and len({r['path'] for r in records}) == 29, 'all 29 production shader sources')
    prior = -1
    for ordinal, r in enumerate(records):
        exact(r['index'], ordinal, 'production shader order')
        begin, end = integer(r['source_begin']), integer(r['source_end'])
        need(prior <= begin < end <= len(native), 'ordered embedded shader extents')
        data = (source/r['path']).read_bytes()
        need(native[begin:end] == data and len(data) == r['bytes'] and sha(data) == r['sha256'], 'unchanged complete embedded production shader')
        prior = end
    need(set(inventory) == {str(q.relative_to(source)) for q in source.rglob('*') if q.is_file()}, 'exact complete source inventory keyset')
    return dict(source_identity_sha256=sha(iraw), source_file_count=len(inventory), native_source_sha256=sha(native), combined_source_sha256=sha(combined), production_shader_count=29)


def common_manifest_identity(m, p, frozen, payloads):
    for k, prefix in (('harness','harness_source'),('binary','binary'),('build_identity','build_identity')):
        need(m[prefix+'_sha256'] == p['artifacts'][k]['sha256'] and alias_name(m[prefix+'_path']) == alias_name(p['artifacts'][k]['path']), 'actual '+k+' identity/path')
    need(payloads.read(m['harness_source_copy']) == frozen['harness'][1] and payloads.read(m['build_identity_copy']) == frozen['build_identity'][1], 'retained original source/build bytes')
    need(type(m['libraries']) is list and len(m['libraries']) == 3, 'three actual libraries')
    for i,(label,art) in enumerate(zip(('deployed','native_full_tu','probe_full_tu'),('deployed_library','native_library','probe_library'))):
        lib=m['libraries'][i]
        exact(lib,dict(index=i,label=label,path=lib['path'],bytes=p['artifacts'][art]['bytes'],sha256=p['artifacts'][art]['sha256']), 'loaded library')
        need(alias_name(lib['path']) == alias_name(p['artifacts'][art]['path']), 'prebound loaded library path')


def verify_host_fixture(p, frozen):
    root=frozen['host_fixture_manifest'][0].parent
    need(frozen['host_fixture_manifest'][0].name == 'artifact-manifest.json', 'host preimage manifest filename')
    m=load_json(frozen['host_fixture_manifest'][0])
    for k,v in dict(schema_version=1,probe='indexer_register_only_correctness',fixture_kind='deterministic_synthetic',correctness_only=True,timing_collected=False,fullmodel=False,production_registered=False,adoption_eligible=False,host_only=True,guard_bytes_per_side=64,poison_byte=165,provider_count=6,case_count=12,gpu_completion_count=0,all_numerical_passed=False,status='host_only_complete_not_numerical').items():
        exact(m[k],v,'frozen host fixture '+k)
    need(m.get('failure') is None and 'device' not in m, 'host fixture is CPU-only with no evidence failure')
    payloads=Payloads(root,m['files']);need(len(payloads.records) == 242, 'exact 242 host raw observation files');common_manifest_identity(m,p,frozen,payloads)
    need(type(m['cases']) is list and len(m['cases']) == 12, 'all host input cases')
    results=[review_case(c,payloads,CASES[i],i,True) for i,c in enumerate(m['cases'])]
    payloads.finish()
    return dict(host_fixture_manifest_sha256=sha(frozen['host_fixture_manifest'][1]),case_count=12,
                raw_file_count=len(payloads.records),gpu_completion_count=0,input_features=[v['input_features'] for v in results])


def compare_host_before(c, host_c, payloads, host_payloads):
    for field in ('case','padding_profile','data_profile','cache_type','args'):
        exact(c[field],host_c[field],'GPU input is prospectively frozen host input '+field)
    for field in ('args72','exact_args36'):
        need(payloads.read(c[field]) == host_payloads.read(host_c[field]), 'GPU argument bytes equal prospectively frozen host bytes')
    for layout in ('strided','dense'):
        for role in range(3):
            for field in ('body','prefix_guard','suffix_guard'):
                need(payloads.read(c['inputs_before'][layout][role][field]) == host_payloads.read(host_c['inputs_before'][layout][role][field]), 'complete GPU input preimage equals frozen host preimage')


def initialize(protocol_path, mappings=()):
    paths = Paths(mappings); p = load_json(protocol_path)
    for k, v in dict(schema_version=1, stage='synthetic_correctness', probe='indexer_register_only_correctness', correctness_only=True, timing_allowed=False, expected_gpu_completion_count=72).items():
        exact(p[k], v, 'protocol ' + k)
    exact(p['cases'], [dict(case=i, **case_description(c), padding_profile=c[3], data_profile=DATA_PROFILES[i % 3], cache_type='BF16') for i, c in enumerate(CASES)], 'prospectively frozen cases')
    need(type(p['artifacts']) is dict and set(p['artifacts']) == set(ARTIFACTS), 'all eight prospective artifacts')
    data = {}
    for k in ARTIFACTS:
        path, b = paths.descriptor(p['artifacts'][k]); data[k] = (path, b)
    need(sha(Path(__file__).read_bytes()) == p['artifacts']['reader']['sha256'], 'executed independent reader source is frozen')
    build = load_json(data['build_identity'][0]); proof = verify_build(paths, build, p['artifacts'])
    need(sha(data['harness'][1]) == p['artifacts']['harness']['sha256'], 'harness hash')
    proof['host_fixture'] = verify_host_fixture(p,data)
    return p, paths, data, proof


def read_stdout(path):
    records = []
    for line in Path(path).read_text().splitlines():
        need(line.strip(), 'no blank stdout record')
        try:
            r = json.loads(line, object_pairs_hook=strict_pairs, parse_constant=lambda _: (_ for _ in ()).throw(EvidenceError('nonfinite stdout')))
        except json.JSONDecodeError as e:
            raise EvidenceError('stdout JSON') from e
        need(type(r) is dict and type(r.get('kind')) is str, 'stdout object kind')
        records.append(r)
    return records


def verify_stdout(m, stdout, passed):
    records = read_stdout(stdout); it = iter(records)
    def take(kind):
        r = next(it, None); need(type(r) is dict and r.get('kind') == kind, 'strict stdout sequence: ' + kind); return r
    for i in range(3):
        r = take('loaded_library'); lib = m['libraries'][i]
        exact(r, dict(kind='loaded_library', **lib), 'loaded library stdout')
    for i in range(6):
        exact(take('pipeline'), dict(kind='pipeline', binding=m['bindings'][i]), 'actual pipeline stdout')
    gpu_all = True
    for ci, c in enumerate(m['cases']):
        for i, pr in enumerate(c['providers']):
            lib = m['libraries'][LIBRARIES[i]]
            exact(pr['library_sha256'], lib['sha256'], 'provider actual library SHA')
            completion = take('gpu_completion')
            status = integer(completion['status'], 0, 5)
            completed = boolean(completion['completed'])
            need(completed == (status == 4), 'GPU status/completion agreement')
            error = completion['error']
            need(error is None or type(error) is str, 'GPU error is text/null')
            need(not completed or error is None, 'completed GPU has no error')
            exact(completion, dict(kind='gpu_completion', ordinal=ci*6+i+1, case=ci, provider=i, status=status, completed=completed, symbol=SYMBOLS[i], library_sha256=lib['sha256'], error=error), 'GPU command-buffer completion')
            exact(pr['gpu_completed'], completed, 'provider GPU completed')
            gpu_all &= completed
            exact(take('provider_result'), dict(kind='provider_result', case=ci, provider=i, numerical_passed=pr['numerical_passed'], tiled_mismatches=pr['tiled_mismatches'], exact_bridge_mismatches=pr['exact_bridge_mismatches']), 'raw/stat result stdout')
        exact(take('case_complete'), dict(kind='case_complete', case=ci, args=c['args'], providers=6), 'case complete stdout')
    expected_status = 'correctness_complete_not_adoption' if passed else 'numerical_failed'
    exact(m['all_numerical_passed'], passed, 'full independent numerical joint gate')
    exact(m['status'], expected_status, 'completion classification')
    exact(take('complete'), dict(kind='complete', status=expected_status, case_count=12, gpu_completion_count=72, all_numerical_passed=passed, timing_collected=False, adoption_eligible=False), 'final complete stdout')
    need(next(it, None) is None, 'no extra timing/summary/unrecognized stdout records')
    return gpu_all


def review(protocol_path, root, stdout, mappings=()):
    p, paths, frozen, proof = initialize(protocol_path, mappings)
    need((Path(root)/'artifact-manifest.json').is_file() and not (Path(root)/'artifact-manifest.json').is_symlink(), 'regular artifact manifest')
    m = load_json(Path(root)/'artifact-manifest.json')
    for k, v in dict(schema_version=1, probe='indexer_register_only_correctness', fixture_kind='deterministic_synthetic', correctness_only=True, timing_collected=False, fullmodel=False, production_registered=False, adoption_eligible=False, host_only=False, guard_bytes_per_side=64, poison_byte=165, provider_count=6, case_count=12, gpu_completion_count=72).items():
        exact(m[k], v, 'artifact ' + k)
    need(m.get('failure') is None, 'no contradictory evidence failure')
    payloads = Payloads(root, m['files'])
    need(len(payloads.records) == 1106, 'exact 1106 GPU raw observation files')
    common_manifest_identity(m,p,frozen,payloads)
    need(type(m['device']) is str and m['device'], 'actual Metal device')
    need(type(m['bindings']) is list and len(m['bindings']) == 6, 'six actual PSO bindings')
    for i, b in enumerate(m['bindings']):
        lib = m['libraries'][LIBRARIES[i]]
        for k, v in dict(provider=i, symbol=SYMBOLS[i], library_index=LIBRARIES[i], library_path=lib['path'], library_sha256=lib['sha256'], args_bytes=72 if i < 4 else 36, dynamic_threadgroup_bytes=TG_BYTES[i], threads=[32, 4, 1], input_layout='strided' if i < 4 else 'dense', oracle_role='tiled_bit_exact_gate' if i < 4 else 'actual_exact_identity_bridge_not_candidate_oracle', thread_execution_width=32).items():
            exact(b[k], v, 'PSO ' + k)
        integer(b['max_threads'], 128)
        integer(b['static_threadgroup_bytes'])
        if i == 3:
            exact(b['static_threadgroup_bytes'], 0, 'register-only has no TG allocation')
    need(type(m['cases']) is list and len(m['cases']) == len(CASES), 'all twelve frozen cases')
    host_root=frozen['host_fixture_manifest'][0].parent
    host_m=load_json(frozen['host_fixture_manifest'][0])
    host_payloads=Payloads(host_root,host_m['files'])
    for i,c in enumerate(m['cases']):
        compare_host_before(c,host_m['cases'][i],payloads,host_payloads)
    results = [review_case(c, payloads, CASES[i], i) for i, c in enumerate(m['cases'])]
    payloads.finish()
    passed = all(c['all_numerical_passed'] for c in results)
    gpu_all = verify_stdout(m, stdout, passed)
    return dict(schema_version=1, evidence_valid=True, classification='synthetic_correctness_passed' if passed else 'synthetic_gpu_execution_failed' if not gpu_all else 'synthetic_numerical_failed',
                numerical_passed=passed, performance_evaluated=False, adoption_eligible=False, fullmodel_eligible=False,
                actual_exact_is_candidate_oracle=False, exact_symbol_invoked_explicitly=True, actual_default_runtime_provider_observed=False, real_capture_and_top512_bridge_completed=False,
                protocol_sha256=sha(Path(protocol_path).read_bytes()), artifact_manifest_sha256=sha((Path(root)/'artifact-manifest.json').read_bytes()),
                stdout_sha256=sha(Path(stdout).read_bytes()), reader_sha256=sha(Path(__file__).read_bytes()),
                device=m['device'], case_count=12, gpu_completion_count=72, raw_file_count=len(payloads.records), build_proof=proof, cases=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--artifact-root', type=Path)
    ap.add_argument('--stdout', type=Path)
    ap.add_argument('--initialize-only', action='store_true')
    ap.add_argument('--path-map', action='append', default=[], metavar='FROZEN_PREFIX=LOCAL_PREFIX')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        print('refusing to overwrite existing review report: ' + str(a.out), file=sys.stderr)
        return 2
    try:
        mappings = []
        for item in a.path_map:
            need('=' in item, 'path-map syntax'); mappings.append(item.split('=', 1))
        if a.initialize_only:
            _, _, _, proof = initialize(a.protocol, mappings)
            result = dict(schema_version=1, classification='cpu_prelaunch_identity_passed', evidence_valid=True,
                          numerical_passed=False, performance_evaluated=False, adoption_eligible=False,
                          gpu_work_submitted=False, build_proof=proof,
                          protocol_sha256=sha(a.protocol.read_bytes()), reader_sha256=sha(Path(__file__).read_bytes()))
        else:
            need(a.artifact_root is not None and a.stdout is not None, 'raw artifact-root/stdout required')
            result = review(a.protocol, a.artifact_root, a.stdout, mappings)
    except (EvidenceError, KeyError, OSError, ValueError, TypeError, struct.error) as e:
        result = dict(schema_version=1, classification='evidence_invalid', evidence_valid=False,
                      numerical_passed=False, performance_evaluated=False, adoption_eligible=False, error=str(e))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as report:
        report.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'build_proof')}, sort_keys=True))
    return 0 if result['evidence_valid'] and (a.initialize_only or result['numerical_passed']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
