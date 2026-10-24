#!/usr/bin/env python3
"""CPU-only audit of immutable experiment logs; never executes a GPU binary."""
import collections
import csv
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/tmp/myds-opt-20261005')
OUT = ROOT / 'evidence-next'
OUT.mkdir(exist_ok=True)
RESULTS = ROOT / 'results'
WIN = RESULTS / 'exclusive-20261005T021217Z'
VARIANTS = ['vec', 'transpose', 'group2', 'group4', 'group8']


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def percentile(values, p):
    a = sorted(values)
    pos = (len(a) - 1) * p
    i = int(pos)
    return a[i] if i == len(a)-1 else a[i] + (a[i+1]-a[i]) * (pos-i)


# A one-line, two-assignment source ablation. Reconstructing the original
# bytes after removing that one insertion is stricter than a loose diff scan.
original_path = Path('/tmp/myds-phase-20261005/unprofiled-src/src/myds_v41_gpu.c')
mask_path = ROOT / 'original-mask-only.c'
old = original_path.read_text()
patched = mask_path.read_text()
needle = '    if (use_gather) {\n'
assert old.count(needle) == 1
assert patched == old.replace(needle, needle + '        s->tmpl_len=0;s->tmpl_carry=0;\n', 1)
diff = ''.join(difflib.unified_diff(old.splitlines(keepends=True), patched.splitlines(keepends=True),
                                  fromfile=str(original_path), tofile=str(mask_path)))
(OUT / 'original-mask-only.patch').write_text(diff)
artifact_path = ROOT / 'original-mask-artifact-audit.json'
artifact = json.loads(artifact_path.read_text())
assert artifact['compiled_source']['original-mask-only.c'] == sha(mask_path)
assert len(artifact['objects']) == 17
changed_objects = [name for name, data in artifact['objects'].items()
                   if data['original_sha256'] != data['ablation_sha256']]
assert changed_objects == ['myds_v41_gpu.o']
for data in artifact['objects'].values():
    assert data['identical'] == (data['original_sha256'] == data['ablation_sha256'])
assert artifact['metallib']['identical']
assert artifact['metallib']['original_sha256'] == artifact['metallib']['ablation_sha256']
events = records(WIN / 'window.jsonl')
preflight = next(r for r in events if r['kind'] == 'preflight_passed')
assert preflight['hashes']['myds.metallib'] == artifact['metallib']['ablation_sha256']
loaded_path = '/tmp/myds-opt-20261005-run/original-mask-only/bin/myds.metallib'
mask_log = (WIN / 'mask-only-long.log').read_text()
assert 'loaded precompiled library from ' + loaded_path in mask_log
mask_event = next(r for r in events if r.get('name') == 'mask-only-long' and r['kind'] == 'measurement_start')
assert mask_event['argv'][0] == '/tmp/myds-opt-20261005-run/original-mask-only/bin/bench_phase'
assert next(r for r in events if r.get('name') == 'mask-only-long' and r['kind'] == 'measurement_end')['returncode'] == 0

model_paths = {
    'original_unfixed': RESULTS / 'exclusive-20261005T020515Z/original-long-metrics.jsonl',
    'original_mask_only': WIN / 'mask-only-long-metrics.jsonl',
    'current_base': RESULTS / 'exclusive-20261005T015621Z/base-long-metrics.jsonl',
    'current_exact_no_matrix': RESULTS / 'exclusive-20261005T020515Z/exact-long-metrics.jsonl',
    'combined_matrix': RESULTS / 'exclusive-20261005T015621Z/combined-long-metrics.jsonl',
}
model = {}
tokens = {}
metas = {}
for name, path in model_paths.items():
    rs = records(path)
    meta = next(r for r in rs if r['kind'] == 'metadata')
    formal = [r for r in rs if r['kind'] == 'sample' and not r['warmup']]
    warm = [r for r in rs if r['kind'] == 'sample' and r['warmup']]
    assert len(formal) == 3 and len(warm) == 1
    assert {r['run'] for r in formal} == {1, 2, 3}
    for r in formal + warm:
        assert r['generation_ok'] and r['sample_valid'] and r['output_equal_to_warmup']
        assert r['prompt_tokens'] == 7195 and r['prefill_tokens'] == 7194
        assert r['cached_tokens'] == 0 and r['output_tokens'] == 64
        assert r['decode_intervals'] == 63 and len(r['callback_intervals_ms']) == 63
        assert r['token_ids'] == formal[0]['token_ids']
    assert len({r['token_ids_fnv1a64'] for r in formal + warm}) == 1
    tokens[name] = formal[0]['token_ids']
    metas[name] = meta
    model[name] = {
        'path': str(path), 'sha256': sha(path), 'formal_runs': 3, 'warmup_runs': 1,
        'token_hash': formal[0]['token_ids_fnv1a64'],
        'output_bytes_hash': formal[0]['output_bytes_fnv1a64'],
        'formal_token_hashes': [r['token_ids_fnv1a64'] for r in formal],
        'all_outputs_identical_within_variant': True,
        'prefill_ms_per_token_median': statistics.median(r['prefill_ms_per_token'] for r in formal),
        'decode_ms_per_token_median': statistics.median(r['decode_ms_per_token'] for r in formal),
        'environment': meta['environment'],
    }

assert model['original_unfixed']['token_hash'] == '9c2affe72412a2f1'
assert model['original_mask_only']['token_hash'] == '39b46a154ab590cd'
assert tokens['original_mask_only'] == tokens['current_base'] == tokens['current_exact_no_matrix']
assert tokens['original_unfixed'] != tokens['original_mask_only']
match_keys = ['model', 'request_fnv1a64', 'prompt_fnv1a64', 'request_bytes', 'prompt_bytes',
              'max_new_tokens', 'max_seq_len', 'temperature', 'top_p', 'top_k', 'min_p',
              'repetition_penalty', 'seed', 'qos', 'prompt_renderer', 'transport', 'decode_basis']
for name in ('original_unfixed', 'current_base', 'current_exact_no_matrix'):
    assert all(metas[name][k] == metas['original_mask_only'][k] for k in match_keys)
    assert metas[name]['environment'] == metas['original_mask_only']['environment']
different_at = [i for i, (a, b) in enumerate(zip(tokens['original_unfixed'], tokens['original_mask_only'])) if a != b]

write('mask-attribution.json', {
    'conclusion': 'For the measured 7194-prefill-token request, the two template invalidation assignments alone change the original F32 deployment output to exactly the current-base 64-token sequence. This isolates the observed base/deployed output difference to the mask correctness fix, not BF16 storage rounding.',
    'scope': 'One deterministic prompt/model/configuration, warmup plus three formal runs. Output sequence equivalence does not assert all internal tensors or all prompts are bitwise equal. Cross-window wall timings are descriptive, not paired performance attribution.',
    'source_audit': {'original': str(original_path), 'original_sha256': sha(original_path),
                     'ablation': str(mask_path), 'ablation_sha256': sha(mask_path),
                     'sole_insertion': 's->tmpl_len=0;s->tmpl_carry=0;',
                     'location': 'use_gather entry, original line598/current ablation line599',
                     'exact_reconstruction_pass': True, 'diff': str(OUT / 'original-mask-only.patch')},
    'artifact_audit': {'path': str(artifact_path), 'sha256': sha(artifact_path),
                       'evidence_origin': 'Root agent saved the remote artifact hashes; this CPU audit validates internal consistency and compares the compiled source hash to the local file.',
                       'object_count': 17, 'identical_objects': 16, 'changed_objects': changed_objects,
                       'metallib_same_bytes_recorded': True,
                       'metallib_sha256': artifact['metallib']['ablation_sha256'],
                       'production_preflight_metallib_sha_matches': True,
                       'loaded_library_log': str(WIN / 'mask-only-long.log'),
                       'loaded_library_path': loaded_path,
                       'harness_note': 'Root independently reviewed deletion of a profiler prototype and an inert finish stub called after the measured wall interval; this audit has the harness hash, not its remote source bytes.',
                       'compiler_stdout_preserved': False},
    'request_configuration_equal': True, 'matched_metadata_fields': match_keys,
    'default_macroblock': 4096, 'chunk': 512, 'configured_moe_chunks': 9,
    'effective_moe_chunks_first_macroblock': 8,
    'first_different_output_index_zero_based': different_at[0],
    'different_output_positions': len(different_at),
    'mask_only_raw_token_array_equals_current_base': True,
    'mask_only_raw_token_array_equals_current_exact_no_matrix': True,
    'model_results': model,
    'cpu_mask_formula_evidence': str(ROOT / 'mask-causality-cpu.json'),
})


# Strict candidates: all full-array F32 and BF16 gates, four captured blocks.
strict_blocks = []
numeric_blocks = []
csv_rows = []
for start, rows, ncomp in [(0,512,256), (512,512,512), (1024,512,768), (1536,294,915)]:
    path = WIN / f'attention-group-{start}.jsonl'
    rs = records(path)
    counts = collections.Counter(r['kind'] for r in rs)
    assert counts == {'metadata':1, 'capture_verify':1, 'reference':1, 'verify':5, 'timing':65}
    meta = next(r for r in rs if r['kind'] == 'metadata')
    assert meta['device'] == 'Apple M2 Ultra' and meta['fixture']
    assert meta['fixture_metadata']['source'] == 'source-current capture, not performance run'
    assert meta['rows'] == rows and meta['start'] == start and meta['n_comp'] == ncomp
    assert meta['heads'] == 64 and meta['dk'] == meta['dv'] == 512
    assert meta['repeats'] == 13 and meta['warmups'] == 2
    assert meta['segments_separate_command_buffers'] is False
    cv = next(r for r in rs if r['kind'] == 'capture_verify')
    assert cv['pass'] is True and cv['bitwise_different_elements'] == cv['max_abs_error'] == cv['tolerance'] == 0
    assert cv['capture_output_fnv'] == cv['replay_output_fnv']
    reference = next(r for r in rs if r['kind'] == 'reference')
    assert reference['pass'] and reference['checked_elements'] == 8192
    assert math.isfinite(reference['max_abs_error']) and reference['max_abs_error'] <= reference['tolerance'] == 3e-6
    verifies = [r for r in rs if r['kind'] == 'verify']
    assert len(verifies) == 5 and {r['variant'] for r in verifies} == set(VARIANTS)
    for v in verifies:
        assert v['pass'] is True and v['bitwise_required'] is True
        for field in ['bitwise_different_elements', 'max_abs_error', 'max_rel_error_floor_1e_6',
                      'tolerance', 'bf16_rounded_different_elements', 'bf16_rounded_max_abs_error']:
            assert v[field] == 0
        assert len(v['bf16_different_by_query']) == rows and len(v['bf16_different_by_head']) == 64
        assert not any(v['bf16_different_by_query']) and not any(v['bf16_different_by_head'])
        assert v['output_fnv'] == cv['capture_output_fnv']
    # Also join the timing rows by unique iteration and candidate. Never pair
    # by adjacency or compare medians in place of per-iteration ratios.
    timing = {v:{} for v in VARIANTS}
    for r in rs:
        if r['kind'] != 'timing': continue
        assert r['stage'] == 'whole' and r['gpu_time_valid']
        assert r['rows'] == rows and r['start'] == start and r['n_comp'] == ncomp
        assert math.isfinite(r['gpu_ms']) and r['gpu_ms'] > 0
        assert math.isfinite(r['wall_ms']) and r['wall_ms'] > 0
        assert r['iteration'] not in timing[r['variant']]
        timing[r['variant']][r['iteration']] = r
    assert all(set(r) == set(range(13)) for r in timing.values())
    variants = []
    for v in VARIANTS:
        gpu = [timing[v][i]['gpu_ms'] for i in range(13)]
        wall = [timing[v][i]['wall_ms'] for i in range(13)]
        ratios = [timing[v][i]['gpu_ms'] / timing['vec'][i]['gpu_ms'] for i in range(13)]
        row = {'start':start, 'rows':rows, 'n_comp':ncomp, 'variant':v, 'samples':13,
               'gpu_ms_median':statistics.median(gpu), 'gpu_ms_p10':percentile(gpu,.1),
               'gpu_ms_p90':percentile(gpu,.9), 'wall_ms_median':statistics.median(wall),
               'paired_gpu_ratio_median':statistics.median(ratios),
               'paired_gpu_time_change_pct_median':100*(statistics.median(ratios)-1),
               'paired_gpu_time_saved_pct_median':100*(1-statistics.median(ratios)),
               'paired_gpu_ratio_p10':percentile(ratios,.1), 'paired_gpu_ratio_p90':percentile(ratios,.9),
               'f32_bitwise_differences':0, 'bf16_bitwise_differences':0,
               'full_output_elements':rows*64*512, 'capture_output_match':True, 'file':str(path)}
        variants.append(row)
        csv_rows.append(row)
    strict_blocks.append({'start':start, 'rows':rows, 'n_comp':ncomp,
                          'metadata':meta, 'source_file_sha256':sha(path),
                          'full_array_numeric_gate_pass':True,
                          'capture_baseline_bitwise_gate_pass':True,
                          'cpu_reference_subset_pass':True, 'variants':variants})
    numeric_blocks.append({'path':str(path), 'start':start, 'rows':rows,
                           'capture_verify':cv, 'reference':reference, 'verify_records':verifies})

total_medians = {v:sum(next(r['gpu_ms_median'] for r in b['variants'] if r['variant']==v)
                     for b in strict_blocks) for v in VARIANTS}
write('attention-strict-summary.json', {
    'scope':'Real captured layer14 attention inputs; four blocks cover1830 rows. Whole operation includes staging, window gather, compressed gather, flash and carry in one command buffer. This is component replay, not whole-model throughput.',
    'timing_protocol':{'warmups_per_variant':2, 'samples_per_variant_per_block':13,
                      'order':'Each round uses fixed vec,transpose,group2,group4,group8 order (round robin); it is not a rotated starting variant.',
                      'pairing':'candidate_gpu_ms(iteration)/vec_gpu_ms(same iteration), median over13 ratios',
                      'interval':'MTLCommandBuffer GPUStartTime to GPUEndTime',
                      'stages':['stage','window','comp','flash','carry']},
    'blocks':strict_blocks,
    'sum_of_independent_block_medians_ms':total_medians,
    'sum_of_independent_block_medians_change_pct':{v:100*(total_medians[v]/total_medians['vec']-1) for v in VARIANTS},
    'sum_interpretation':'Arithmetic sum of four separately replayed blocks; not a paired single-command-buffer1830-row measurement.',
    'all20_candidate_full_array_gates_pass':True,
    'all4_capture_baseline_gates_pass':True,
    'all4_cpu_reference_subset_gates_pass':True,
    'transpose_fastest_at_all4_measured_shapes':all(min(b['variants'],key=lambda r:r['gpu_ms_median'])['variant']=='transpose' for b in strict_blocks),
    'source_code':str(ROOT/'attention-group-src/tools/bench_attention_group.m'),
    'source_code_sha256':sha(ROOT/'attention-group-src/tools/bench_attention_group.m'),
    'excluded_from_real_qkv_evidence':['exclusive-20261005T014800Z/qkv-real-nr2.jsonl',
                                       'exclusive-20261005T014800Z/qkv-real-nr4.jsonl',
                                       '015042 synthetic QKV experiments'],
})
write('attention-strict-numeric-audit.json', {'blocks':numeric_blocks})
with (OUT/'attention-strict-summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)


# Retain earlier actual-capture attention results including matrix negative
# evidence. Tolerance pass must never be silently upgraded to a strict gate.
other = []
for directory, pattern in [('exclusive-20261005T014800Z','attention-real-*.jsonl'),
                           ('exclusive-20261005T015621Z','attention-maskscan-*.jsonl')]:
    for path in sorted((RESULTS/directory).glob(pattern)):
        rs = records(path)
        meta = next(r for r in rs if r['kind']=='metadata')
        assert meta['fixture'] and meta['fixture_metadata']['source']=='source-current capture, not performance run'
        verify = [r for r in rs if r['kind']=='verify']
        assert len(verify)==3 and {r['variant'] for r in verify}=={'vec','direct','matrix'}
        assert all(r['pass'] for r in verify)
        cv = next(r for r in rs if r['kind']=='capture_verify')
        assert cv['pass'] and cv['bitwise_different_elements']==cv['max_abs_error']==cv['tolerance']==0
        assert cv['capture_output_fnv']==cv['replay_output_fnv']
        reference = next(r for r in rs if r['kind']=='reference')
        assert reference['pass'] and reference['checked_elements']==8192
        assert math.isfinite(reference['max_abs_error']) and reference['max_abs_error']<=reference['tolerance']==3e-6
        for v in verify:
            assert math.isfinite(v['max_abs_error']) and math.isfinite(v['max_rel_error_floor_1e_6'])
            if v['bitwise_required']: assert v['bitwise_different_elements']==v['max_abs_error']==0
            else: assert v['max_abs_error']<=v['tolerance']
            if v['variant'] in ('vec','direct'): assert v['output_fnv']==cv['capture_output_fnv']
            if 'bf16_rounded_different_elements' in v:
                assert v['bf16_rounded_different_elements']>=0
                assert math.isfinite(v['bf16_rounded_max_abs_error'])
        timing = {v:{} for v in ('vec','direct','matrix')}
        for r in rs:
            if r['kind']!='timing':continue
            assert r['stage']=='whole' and r['gpu_time_valid']
            assert math.isfinite(r['gpu_ms']) and r['gpu_ms']>0
            assert r['iteration'] not in timing[r['variant']]
            timing[r['variant']][r['iteration']]=r
        assert all(set(rows)==set(range(meta['repeats'])) for rows in timing.values())
        paired=[]
        for v in timing:
            ratio=[timing[v][i]['gpu_ms']/timing['vec'][i]['gpu_ms'] for i in range(meta['repeats'])]
            paired.append({'variant':v,'gpu_ms_median':statistics.median(r['gpu_ms'] for r in timing[v].values()),
                           'paired_gpu_time_change_pct_median':100*(statistics.median(ratio)-1)})
        other.append({'path':str(path), 'sha256':sha(path), 'metadata':meta, 'verify_records':verify,
                      'capture_verify':cv, 'reference':reference, 'paired_timing_summary':paired,
                      'matrix_strict_f32_gate_pass':False,
                      'matrix_bf16_difference_known':next(r for r in verify if r['variant']=='matrix').get('bf16_rounded_different_elements'),
                      'earlier_missing_bf16_fields_are_unknown_not_zero':True})
write('attention-other-real-numeric-audit.json', {'datasets':other,
                                               'scope':'Actual capture evidence; matrix tolerance success is not bitwise equivalence. Missing numeric fields remain unknown.'})

# Other component data are classified explicitly. This audit does not relabel
# generated activations as captured model input because the weights are real.
components=[]
for directory, name in [('exclusive-20261005T014800Z','moe-real.jsonl'),
                         ('exclusive-20261005T015621Z','moe-half-real.jsonl')]:
    path=RESULTS/directory/name
    rs=records(path)
    config=next(r for r in rs if r['type']=='config')
    assert config['weight_kind']=='gguf_single_layer_iq2'
    route=next(r for r in rs if r['type']=='route_stats')
    assert route['case']=='capture' and route['gpu_map_exact']
    assert sum(route['counts'])==config['rows']*config['active']
    assert sum((n+31)//32 for n in route['counts'])==route['W']==456
    numeric=[r for r in rs if r['type']=='numeric']
    assert len(numeric)==4 and {r['variant_id'] for r in numeric}==set(range(4))
    for n in numeric:
        assert all(n[k]==0 for k in ('bitwise_different','nonfinite','max_abs','rms_error'))
        assert n['fnv1a64']==n['reference_fnv1a64']
        assert n['half_words']==config['output_bytes']//2==25297920
    components.append({'path':str(path),'sha256':sha(path),
                       'provenance':'Captured prefill FFN row matrix/routes plus actual GGUF single-layer weights',
                       'config':config,'numeric_gate_records':numeric,'gpu_map_exact':True,
                       'route_W':456,'performance_scope':'Isolated up/gate operation, not whole prefill/decode'})

path=WIN/'decode-iq2-real.jsonl'
rs=records(path)
config=next(r for r in rs if r['type']=='config')
inp=next(r for r in rs if r['type']=='input')
assert inp['weight_kind']=='gguf_selected_6_experts_iq2'
assert inp['source']=='prefill normalized FFN row replayed in single-token MV shape'
numeric=[r for r in rs if r['type']=='numeric']
assert len(numeric)==90
assert all(r['bitwise_equal'] and r['canaries'] and r['finite'] and
           all(r[k]>0 for k in ('gate_nonzero','up_nonzero','mid_nonzero')) for r in numeric)
coverage=collections.Counter((r['dim'],r['width']) for r in numeric)
assert len(coverage)==9 and all(n==10 for n in coverage.values())
assert next(r for r in rs if r['type']=='complete')['numeric']=='all gate/up/mid bitwise equal, finite and nonzero; all canaries intact'
components.append({'path':str(path),'sha256':sha(path),'config':config,'input':inp,
                   'provenance':'Actual GGUF selected experts and one captured prefill row; this is not captured decode input',
                   'numeric_record_count':90,'numeric_records':numeric,
                   'numeric_shape_coverage': [{'dim':k[0],'width':k[1],'records':n} for k,n in coverage.items()],
                   'numeric_coverage_scope':'Boundary and native shapes; do not call all90 records native-model shapes.',
                   'performance_scope':'Single-token MV replay of a prefill row, not whole decode throughput'})

qkv=[]
for nr in (2,4):
    path=RESULTS/'exclusive-20261005T015621Z'/f'qkv-actual-nr{nr}.jsonl'
    text=path.read_text()
    assert 'weights=actual GGUF selected tensors' in text
    assert f'MYDS_QKV_NR0={nr}' in text
    table=list(csv.reader(line for line in text.splitlines() if not line.startswith('#') and
                          not line.startswith('[') and not line.startswith('myds_metal:')))
    assert table[0]==['component','median_gpu_us','median_host_wall_us','checksum']
    values={r[0]:r for r in table[1:]}
    assert all(len(values[f'pair.sample.{i}.{v}'])==4 for i in range(9) for v in ('old','fused'))
    assert all(values[f'pair.sample.{i}.old'][3]==values[f'pair.sample.{i}.fused'][3] for i in range(9))
    qkv.append({'path':str(path),'sha256':sha(path),'nr0':nr,
                'provenance':'Actual GGUF weights, deterministic generated F32 activation; not model activation capture',
                'format':'CSV with startup log lines despite .jsonl suffix',
                'paired_output_checksums_all9_match':True,
                'paired_gpu_time_change_pct_median':100*(statistics.median(float(values[f'pair.sample.{i}.fused'][1])/float(values[f'pair.sample.{i}.old'][1]) for i in range(9))-1),
                'source_generated_input':str(ROOT/'qkv-src/tools/bench_qkv.m')+':29'})
write('component-provenance-audit.json',{
    'components':components,'qkv_actual_weight_but_generated_activation':qkv,
    'excluded_qkv':['014800 qkv-real-nr2/nr4 (synthetic weights)','015042 synthetic QKV'],
    'no_synthetic_qkv_used_in_real_attention_or_mask_attribution':True})

print(json.dumps({'mask_attribution_pass':True,
                  'mask_only_hash':model['original_mask_only']['token_hash'],
                  'strict_blocks':4, 'strict_numeric_records':20, 'strict_timing_records':260,
                  'transpose_gain_pct_by_start':{b['start']:next(r['paired_gpu_time_saved_pct_median'] for r in b['variants'] if r['variant']=='transpose') for b in strict_blocks},
                  'sum_medians_ms':total_medians, 'output':str(OUT)}))
