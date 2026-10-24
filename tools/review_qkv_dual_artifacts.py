#!/usr/bin/env python3
"""CPU-only independent audit of bench_qkv_dual raw files and GPU timestamps.

--root is the window directory containing component/, component.stdout and
capture.json. --model additionally parses the real GGUF descriptors and hashes
each actual Qa/KV weight interval; it never maps/loads the model into Metal.
Exit 0 means valid evidence, including a valid performance rejection. The JSON
decision and performance_gate_pass determine eligibility for the full-model gate.
--self-test creates synthetic CPU evidence; it is never GPU or quality evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import tarfile
import tempfile


NATIVE = 'kernel_mul_mv_q4_K_dense_f32_bf16'
DUAL = 'kernel_myds_qkv_dual_q4_k_bf16_nr2'
LAYERS, POSITIONS = (0, 19, 39), (0, 1, 7, 15)
MODES, CONTROLS, ARMS = ('serial', 'concurrent'), ('production', 'probe'), ('A1', 'B', 'A2')
OUTPUT = {'qa': 1280 * 4, 'kv': 512 * 4}
WEIGHT = {'qa': 1280 * 2880, 'kv': 512 * 2880}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path, offset=0, length=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        while length is None or length:
            data = stream.read(1048576 if length is None else min(length, 1048576))
            if not data:
                break
            h.update(data)
            if length is not None:
                length -= len(data)
    need(length in (None, 0), 'Short SHA interval: ' + str(path))
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    return isinstance(a, (int, float)) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def gguf_descriptors(path, wanted):
    """Parse CPU metadata, skip tokenizer arrays, and retain named tensors only."""
    size = Path(path).stat().st_size
    with Path(path).open('rb') as stream:
        def read(n):
            need(0 <= n <= size - stream.tell(), 'GGUF read outside file')
            data = stream.read(n)
            need(len(data) == n, 'Short GGUF read')
            return data
        def integer(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize('<' + fmt)))[0]
        def skip(n):
            need(0 <= n <= size - stream.tell(), 'GGUF skip outside file')
            stream.seek(n, 1)
        def string():
            n = integer('Q')
            need(n <= 1048576, 'Unbounded GGUF tensor/metadata name')
            return read(n).decode('utf-8')
        primitive = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def discard(kind):
            if kind in primitive:
                skip(struct.calcsize('<' + primitive[kind]))
            elif kind == 8:
                skip(integer('Q'))
            else:
                need(kind == 9, 'Unsupported GGUF metadata type')
                element, count = integer('I'), integer('Q')
                need(count <= 10000000 and element != 9, 'Unbounded/nested GGUF metadata array')
                if element in primitive:
                    skip(count * struct.calcsize('<' + primitive[element]))
                else:
                    need(element == 8, 'Unsupported GGUF array element type')
                    for _ in range(count):
                        skip(integer('Q'))
        need(read(4) == b'GGUF' and integer('I') in (2, 3), 'Expected GGUF v2/v3')
        nt, nk = integer('Q'), integer('Q')
        need(nt < 100000 and nk < 100000, 'Unbounded GGUF descriptor count')
        alignment = 32
        for _ in range(nk):
            key, kind = string(), integer('I')
            if key == 'general.alignment':
                need(kind == 4, 'GGUF alignment must be UINT32')
                alignment = integer('I')
            else:
                discard(kind)
        descriptors = {}
        for _ in range(nt):
            name, nd = string(), integer('I')
            need(0 < nd <= 4, 'Invalid GGUF tensor rank')
            dims = [integer('Q') for _ in range(nd)]
            kind, offset = integer('I'), integer('Q')
            if name in wanted:
                need(name not in descriptors, 'Duplicate target tensor')
                descriptors[name] = dict(type=kind, dimensions=dims, offset=offset)
        need(0 < alignment <= 1048576 and not alignment & (alignment - 1), 'Invalid GGUF alignment')
        data_offset = (stream.tell() + alignment - 1) // alignment * alignment
    need(set(descriptors) == set(wanted), 'Missing real GGUF Qa/KV descriptor')
    return descriptors, data_offset


def all_qkv_descriptors(model):
    """Fingerprint metadata for all 40 Qa/KV pairs, without reading weights."""
    need(model is not None, '--model is required for all Qa/KV descriptors')
    model = Path(model).resolve(strict=True)
    before = model.stat()
    wanted = {f'blk.{layer}.{part}.weight' for layer in range(40)
              for part in ('attn_q_a','attn_kv')}
    descriptors,data_offset = gguf_descriptors(model,wanted)
    tensors=[]
    for layer in range(40):
        for part,rows in (('attn_q_a',1280),('attn_kv',512)):
            name=f'blk.{layer}.{part}.weight';row=descriptors[name]
            count=rows*2880;absolute=data_offset+row['offset']
            need(row['type']==12 and row['dimensions']==[5120,rows], 'Unexpected whole-model Qa/KV type/shape: '+name)
            need(row['offset']>=0 and data_offset<=absolute<=before.st_size and
                 count<=before.st_size-absolute, 'Whole-model Qa/KV interval outside GGUF: '+name)
            tensors.append(dict(name=name,layer=layer,projection=part,type=12,type_name='Q4_K',
                                dimensions=row['dimensions'],relative_offset=row['offset'],
                                file_offset=absolute,bytes=count,complete_interval_in_file=True))
    prefix_bytes=min(1048576,before.st_size);prefix_sha=sha(model,length=prefix_bytes)
    after=model.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'Model changed during metadata audit')
    return dict(schema_version=1,reviewer='independent_cpu_qkv_metadata',gpu_work_submitted=False,
                metadata_only=True,weight_payload_hashes_read=0,all_40_layers_verified=True,
                all_80_expected_tensors_Q4_K=True,tensor_count=len(tensors),tensor_data_offset=data_offset,
                model=dict(path=str(model),bytes=before.st_size,inode=before.st_ino,device=before.st_dev,
                           mtime_ns=before.st_mtime_ns,prefix_bytes=prefix_bytes,prefix_sha256=prefix_sha),
                limitation='Metadata and interval bounds only; six real payload hashes are in the independent live-source component report',
                tensors=tensors)


def review(root, model=None, source_archive=None, artifacts_archive=None):
    root = Path(root).resolve()
    component = root / 'component'
    manifest = load(component / 'artifact-manifest.json')
    need(manifest.get('status') == 'complete' and manifest.get('probe') == 'qkv_dual_real_capture', 'Incomplete component manifest')
    need(manifest.get('nsg') == 4 and manifest.get('nr0') == 2 and manifest.get('total_cases') == 12,
         'Wrong production component geometry/case count')
    samples, reps = manifest['samples'], manifest['repetitions_per_cb']
    need(isinstance(samples, int) and 3 <= samples <= 64 and isinstance(reps, int) and 1 <= reps <= 256,
         'Invalid sample/repetition count')
    need(set(manifest['modes']) == set(MODES), 'Both encoder modes are required')
    entries = manifest['files']
    paths = [entry['path'] for entry in entries]
    need(len(paths) == len(set(paths)), 'Duplicate raw artifact paths')
    for entry in entries:
        path = (component / entry['path']).resolve()
        need(path.is_relative_to(component.resolve()) and path.is_file(), 'Raw path escapes component')
        need(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Raw file SHA/bytes: ' + entry['path'])
    expected = {'capture.json'} | {f'capture-case-{case:03d}-input.f32' for case in range(12)}
    expected |= {f'weight-layer-{layer}-{kind}.q4k' for layer in LAYERS for kind in OUTPUT}
    outputs = {f'parity-{mode}-case-{case:03d}-provider-{provider}-{kind}'
               for mode in MODES for case in range(12) for provider in range(3) for kind in OUTPUT}
    expected |= {name + '.f32' for name in outputs}
    guard_present = any(name.endswith('.guards') for name in paths)
    if guard_present:
        expected |= {name + '.guards' for name in outputs}
    need(set(paths) == expected, 'Raw manifest missing or adding unexpected schema payloads')
    capture = load(root / 'capture.json')
    need(sha(root / 'capture.json') == manifest['capture_json_sha256'] == sha(component / 'capture.json'), 'Capture JSON identity')
    need(capture['capture_kind'] == 'myds_decode_qkv' and len(capture['cases']) == 12, 'Capture type/count')
    need({(c['layer'], c['position']) for c in capture['cases']} == {(l,p) for l in LAYERS for p in POSITIONS}, 'Capture coverage')
    layer_counts = {layer: 0 for layer in LAYERS}
    trace_inputs_verified = 0
    for case, captured in enumerate(capture['cases']):
        layer_counts[captured['layer']] += 1
        path = component / f'capture-case-{case:03d}-input.f32'
        data = path.read_bytes()
        need(len(data) == 20480 and sha(path) == captured['input_f32']['sha256'], 'Captured attention norm SHA/bytes')
        need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', data)), 'Nonfinite captured input')
        trace = root/'trace'/f"myds_attn_norm-{captured['layer']}_pos{captured['position']}.bin"
        if trace.exists():
            need(trace.read_bytes() == data, 'Archived actual trace input differs from component input')
            trace_inputs_verified += 1
    bindings = manifest['bindings']
    need(len(bindings) == 3, 'Expected three explicit PSOs')
    for provider, row in enumerate(bindings):
        need(row['provider'] == provider and row['symbol'] == (DUAL if provider == 2 else NATIVE) and
             row['library'] == ('production' if provider == 0 else 'probe') and row['fc600'] == 4 and
             row['nr0'] == 2 and row['args_bytes'] == 112 and row['threads'] == [32,4,1] and
             row['groups'] == ([224] if provider == 2 else [160,64]) and
             row['dynamic_threadgroup_bytes'] == 256 and row['thread_execution_width'] == 32 and
             row['max_threads'] >= 128, 'Provider ABI/geometry mismatch')
    tensors = {row['name']: row for row in manifest['tensors']}
    need(len(tensors) == 6, 'Expected six actual weight tensors')
    for layer in LAYERS:
        for kind, rows in (('qa', 1280), ('kv', 512)):
            name = f"blk.{layer}.{'attn_q_a' if kind == 'qa' else 'attn_kv'}.weight"
            tensor = tensors[name]
            path = component / f'weight-layer-{layer}-{kind}.q4k'
            need(tensor['type'] == 12 and tensor['dimensions'] == [5120,rows] and
                 tensor['bytes'] == WEIGHT[kind] == path.stat().st_size and sha(path) == tensor['sha256'],
                 'Captured Q4_K full weight tensor identity')
            need(tensor['file_offset'] >= manifest['model']['tensor_data_offset'] and
                 tensor['file_offset'] + tensor['bytes'] <= manifest['model']['bytes'], 'GGUF interval outside recorded model')
    raw = {}
    floats = 0
    for name in outputs:
        kind = name.rsplit('-', 1)[-1]
        data = (component / (name + '.f32')).read_bytes()
        need(len(data) == OUTPUT[kind], 'Wrong Qa/KV output size')
        bits = [x[0] for x in struct.iter_unpack('<I', data)]
        need(all(v != 0xa5a5a5a5 for v in bits), 'Output contains unwritten poison')
        need(all(not v & 0xffff for v in bits), 'Output violates BF16 final rounding')
        need(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data)), 'Output has NaN/Inf')
        if guard_present:
            need((component / (name + '.guards')).read_bytes() == b'\xa5' * 128, 'Output guard differs')
        raw[name] = data
        floats += len(bits)
    for case in range(12):
        for kind in OUTPUT:
            oracle = raw[f'parity-serial-case-{case:03d}-provider-0-{kind}']
            for mode in MODES:
                for provider in range(3):
                    need(raw[f'parity-{mode}-case-{case:03d}-provider-{provider}-{kind}'] == oracle,
                         f'Non-0-ULP Qa/KV: mode={mode} case={case} provider={provider} kind={kind}')
    rows = [json.loads(line) for line in (root / 'component.stdout').read_text().splitlines() if line.strip()]
    parity = [row for row in rows if row.get('kind') == 'parity']
    need(len(parity) == 72 and {(r['mode'],r['case'],r['provider']) for r in parity} ==
         {(m,c,p) for m in MODES for c in range(12) for p in range(3)}, 'Incomplete parity command records')
    for row in parity:
        case = capture['cases'][row['case']]
        need(row['layer'] == case['layer'] and row['position'] == case['position'] and
             row['qa_mismatches'] == row['kv_mismatches'] == 0 and
             all(row.get(k) is True for k in ('full_f32_bit_exact','finite_guards_written','bf16_postround','completed_cb')),
             'Parity record disagrees with raw result')
    complete = [row for row in rows if row.get('kind') == 'complete']
    need(len(complete) == 1 and rows[-1] == complete[0] and complete[0].get('qa_kv_checked') is True and
         complete[0].get('full_f32_bit_exact') is True and complete[0].get('cases') == 12 and
         set(complete[0].get('modes',[])) == set(MODES), 'No complete correctness gate')
    timing = [row for row in rows if row.get('kind') == 'timing']
    need(len(timing) == len(LAYERS) * 2 * 2 * 3 * (samples + 1), 'Missing timing command buffers')
    seen, last_end = set(), 0
    for row in timing:
        key = (row['mode'], row['control_kind'], row['layer'], row['arm'], row['sample'])
        need(key not in seen, 'Duplicate timing command record')
        seen.add(key)
        need(row['mode'] in MODES and row['control_kind'] in CONTROLS and row['layer'] in LAYERS and
             row['arm'] in ARMS and 0 <= row['sample'] <= samples and row['warmup'] == (row['sample'] == 0),
             'Unexpected timing group/sample')
        provider = 2 if row['arm'] == 'B' else (0 if row['control_kind'] == 'production' else 1)
        pairs = layer_counts[row['layer']] * reps
        need(row['provider'] == provider and row['symbol'] == (DUAL if provider == 2 else NATIVE) and
             row['pairs'] == pairs and row['dispatches'] == pairs * (1 if provider == 2 else 2) and row['commits'] == 1,
             'Timing actual-provider/dispatch accounting mismatch')
        start, end = row['gpu_start_s'], row['gpu_end_s']
        need(all(isinstance(v,(int,float)) and math.isfinite(v) for v in (start,end)) and
             start > 0 and end > start and start >= last_end and
             row['completed_cb'] is True and row['full_f32_bit_exact'] is True, 'Invalid/overlapping GPU timestamps')
        last_end = end
        need(close(row['gpu_us_per_pair'], (end-start)*1e6/pairs), 'Timestamp-derived pair time differs')
    summaries = [row for row in rows if row.get('kind') == 'summary']
    need(len(summaries) == 12, 'Expected twelve mode/control/layer summaries')
    derived = []
    for mode in MODES:
        for control in CONTROLS:
            for layer in LAYERS:
                med = {}
                for arm in ARMS:
                    group = [r for r in timing if (r['mode'],r['control_kind'],r['layer'],r['arm']) ==
                             (mode,control,layer,arm) and not r['warmup']]
                    need(len(group) == samples and {r['sample'] for r in group} == set(range(1,samples+1)),
                         'Missing complete formal A1/B/A2 sample set')
                    med[arm] = statistics.median(r['gpu_us_per_pair'] for r in group)
                a1,b,a2 = (med[arm] for arm in ARMS)
                computed = dict(a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=(a2/a1-1)*100,
                                conservative_time_saved_percent=(1-b/min(a1,a2))*100,
                                time_saved_percent=(1-b/((a1+a2)/2))*100)
                published = [r for r in summaries if (r['mode'],r['control_kind'],r['layer']) == (mode,control,layer)]
                need(len(published) == 1 and published[0]['control_symbol'] == NATIVE and
                     published[0]['candidate_symbol'] == DUAL and all(close(published[0][k],v) for k,v in computed.items()),
                     'Published summary differs from formal GPU timestamps')
                derived.append(dict(mode=mode,control_kind=control,layer=layer,**computed,
                                    performance_gate_pass=abs(computed['control_drift_percent']) <= 2 and
                                    computed['conservative_time_saved_percent'] > 2))
    primary = [r for r in derived if r['mode'] == 'concurrent' and r['control_kind'] == 'production']
    won = all(r['performance_gate_pass'] for r in primary)
    result = dict(schema_version=1,evidence_valid=True,correctness_pass=True,performance_gate_pass=won,
                  decision='eligible_for_fullmodel_gate' if won else 'reject_component_performance',
                  gpu_work_submitted=False,fullmodel_run=False,production_adoption=False,cold_dram_proven=False,
                  raw_files_verified=len(entries),full_output_f32_values_verified=floats,parity_command_records=72,
                  guard_payloads_verified=len(outputs) if guard_present else 0,timing_command_buffers=len(timing),
                  protocol=dict(primary_mode='concurrent',primary_control='production',layers=list(LAYERS),
                                max_absolute_drift_percent=2,minimum_conservative_saving_percent_strict=2),
                  primary_performance=primary,supplemental_performance=[r for r in derived if r not in primary],
                  archived_weight_tensor_sha_verified=6,real_gguf_intervals_independently_checked=False,
                  archived_trace_inputs_verified=trace_inputs_verified,
                  model_fingerprint_scope='recorded file size + prefix SHA + six full weight tensor SHA; no whole-model-file SHA',
                  component_stdout_sha256=sha(root/'component.stdout'),
                  artifact_manifest_sha256=sha(component/'artifact-manifest.json'))
    if (root/'protocol.json').exists():
        frozen = load(root/'protocol.json')
        need(frozen['primary_mode'] == 'concurrent' and frozen['primary_control_kind'] == 'production' and
             frozen['max_absolute_control_drift_percent'] == 2 and
             frozen['minimum_conservative_time_saved_percent_strict'] == 2, 'Frozen runner gate differs from reviewed gate')
    if (root/'summary.json').exists():
        runner = load(root/'summary.json')
        need(runner.get('correctness_pass') is True and runner.get('performance_gate_pass') is won and
             runner.get('status') == ('component_passed' if won else 'component_rejected_performance'), 'Runner gate disagrees with raw-derived decision')
        result['runner_gate_agrees'] = True
        result['runner_summary_sha256'] = sha(root/'summary.json')
    if model:
        model = Path(model)
        descriptors,data_offset = gguf_descriptors(model,tensors)
        recorded = manifest['model']
        need(model.stat().st_size == recorded['bytes'] and data_offset == recorded['tensor_data_offset'] and
             sha(model,length=recorded['header_bytes']) == recorded['header_sha256'], 'Actual model size/prefix/data offset differs')
        for name,tensor in tensors.items():
            descriptor = descriptors[name]
            need(descriptor['type'] == tensor['type'] and descriptor['dimensions'] == tensor['dimensions'] and
                 descriptor['offset'] + data_offset == tensor['file_offset'], 'Real GGUF descriptor/interval mismatch')
            need(sha(model,tensor['file_offset'],tensor['bytes']) == tensor['sha256'], 'Actual GGUF weight interval SHA differs')
        result['real_gguf_intervals_independently_checked'] = True
        result['actual_gguf_weight_intervals_verified'] = 6
    if source_archive or artifacts_archive:
        stage = load(root/'stage-inputs.json')
        roots = [Path(path).parent for path in stage if Path(path).name == 'Makefile']
        need(len(roots) == 1, 'Expected one executed stage root')
        binary_hashes = {Path(path).name: expected for path,expected in stage.items()
                         if Path(path).parent == roots[0]/'bin'}
        need(manifest['binary_sha256'] == binary_hashes['bench_qkv_dual'] and
             manifest['probe_metallib_sha256'] == binary_hashes['myds.metallib'] and
             capture['source']['trace_binary_sha256'] == binary_hashes['trace_model'] and
             capture['source']['trace_metallib_sha256'] == binary_hashes['myds.metallib'],
             'Captured trace/component artifact identity differs from frozen stage')
        checked = 0
        for archive_path,is_binary in ((source_archive,False),(artifacts_archive,True)):
            if not archive_path:
                continue
            with tarfile.open(archive_path,'r:*') as archive:
                count = 0
                for path,expected_sha in stage.items():
                    path = Path(path)
                    if not path.is_relative_to(roots[0]):
                        need(path.name == 'protocol.json' and sha(root/'protocol.json') == expected_sha, 'Unknown external frozen stage input')
                        continue
                    relative = str(path.relative_to(roots[0]))
                    if relative.startswith('bin/') != is_binary:
                        continue
                    stream = archive.extractfile(relative)
                    need(stream is not None and hashlib.sha256(stream.read()).hexdigest() == expected_sha, 'Executed archive SHA differs: '+relative)
                    count += 1
                checked += count
                result['executed_artifact_members_verified' if is_binary else 'executed_source_members_verified'] = count
            result['executed_artifacts_archive_sha256' if is_binary else 'executed_source_archive_sha256'] = sha(archive_path)
        result['executed_archive_members_verified'] = checked
    if (root/'before.json').exists():
        before = load(root/'before.json')
        backup = root/'production-artifacts'/Path(before['metallib']['path']).name
        need(manifest['production_metallib_sha256'] == before['metallib_sha256'] == sha(backup),
             'Native golden does not match archived production library')
        result['native_library_bound_to_original_production'] = True
    return result


def self_test():
    """Build a real sparse GGUF-shaped CPU fixture, then exercise rejection paths."""
    with tempfile.TemporaryDirectory(prefix='myds-qkv-raw-review-') as temporary:
        root = Path(temporary); component = root/'component'; component.mkdir()
        def string(value):
            data=value.encode();return struct.pack('<Q',len(data))+data
        header=struct.pack('<IIQQ',0x46554747,3,6,0);offset=0;tensors=[]
        for layer in LAYERS:
            for kind,n in (('qa',1280),('kv',512)):
                name=f"blk.{layer}.{'attn_q_a' if kind=='qa' else 'attn_kv'}.weight"
                header+=string(name)+struct.pack('<IQQIQ',2,5120,n,12,offset)
                tensors.append(dict(name=name,type=12,dimensions=[5120,n],bytes=WEIGHT[kind],offset=offset,layer=layer,kind=kind))
                offset+=WEIGHT[kind]
        header+=b'\0'*(-len(header)%32);model=root/'synthetic-cpu-only.gguf'
        with model.open('wb') as stream:stream.write(header);stream.truncate(len(header)+offset)
        files=[]
        def raw(name,data):
            (component/name).write_bytes(data);files.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        cases=[]
        for case,(layer,position) in enumerate((l,p) for l in LAYERS for p in POSITIONS):
            data=struct.pack('<5120f',*([1.0]*5120));name=f'capture-case-{case:03d}-input.f32';raw(name,data)
            cases.append(dict(layer=layer,position=position,input_f32=dict(path=name,sha256=hashlib.sha256(data).hexdigest())))
        capture=dict(capture_kind='myds_decode_qkv',cases=cases,source=dict(kind='synthetic_cpu_only_fixture'))
        (root/'capture.json').write_text(json.dumps(capture));raw('capture.json',(root/'capture.json').read_bytes())
        for tensor in tensors:
            raw(f"weight-layer-{tensor['layer']}-{tensor['kind']}.q4k",b'\0'*tensor['bytes'])
            tensor['file_offset']=len(header)+tensor.pop('offset');tensor['sha256']=hashlib.sha256(b'\0'*tensor['bytes']).hexdigest()
            tensor.pop('layer');tensor.pop('kind')
        for mode in MODES:
            for case in range(12):
                for provider in range(3):
                    for kind,n in OUTPUT.items():
                        name=f'parity-{mode}-case-{case:03d}-provider-{provider}-{kind}'
                        raw(name+'.f32',struct.pack('<f',1.0+case/8)*(n//4));raw(name+'.guards',b'\xa5'*128)
        bindings=[dict(provider=p,symbol=DUAL if p==2 else NATIVE,library='production' if p==0 else 'probe',fc600=4,nr0=2,
                       args_bytes=112,threads=[32,4,1],groups=[224] if p==2 else [160,64],dynamic_threadgroup_bytes=256,
                       thread_execution_width=32,max_threads=512) for p in range(3)]
        manifest=dict(status='complete',probe='qkv_dual_real_capture',nsg=4,nr0=2,total_cases=12,samples=3,repetitions_per_cb=1,
                      modes=list(MODES),files=files,capture_json_sha256=sha(root/'capture.json'),bindings=bindings,tensors=tensors,
                      model=dict(bytes=model.stat().st_size,header_bytes=1048576,header_sha256=sha(model,length=1048576),tensor_data_offset=len(header)))
        manifest_path=component/'artifact-manifest.json';manifest_path.write_text(json.dumps(manifest))
        rows=[];clock=10.0
        for layer in LAYERS:
            for mode in MODES:
                for case,captured in enumerate(cases):
                    if captured['layer']==layer:
                        for p in range(3):rows.append(dict(kind='parity',mode=mode,case=case,provider=p,layer=layer,position=captured['position'],
                            qa_mismatches=0,kv_mismatches=0,full_f32_bit_exact=True,finite_guards_written=True,bf16_postround=True,completed_cb=True))
                for control in CONTROLS:
                    times={a:[] for a in ARMS}
                    for sample in range(4):
                        for arm in ARMS:
                            provider=2 if arm=='B' else (0 if control=='production' else 1)
                            us=90.0 if provider==2 else 100.0;start=clock;end=start+us*4/1e6;clock=end+0.0001
                            actual=(end-start)*1e6/4
                            rows.append(dict(kind='timing',mode=mode,layer=layer,control_kind=control,provider=provider,symbol=DUAL if provider==2 else NATIVE,
                                arm=arm,sample=sample,warmup=sample==0,pairs=4,dispatches=4 if provider==2 else 8,commits=1,gpu_start_s=start,gpu_end_s=end,
                                gpu_us_per_pair=actual,completed_cb=True,full_f32_bit_exact=True))
                            if sample:times[arm].append(actual)
                    a1,b,a2=(statistics.median(times[a]) for a in ARMS)
                    rows.append(dict(kind='summary',mode=mode,layer=layer,control_kind=control,control_symbol=NATIVE,candidate_symbol=DUAL,
                        a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=(a2/a1-1)*100,conservative_time_saved_percent=(1-b/min(a1,a2))*100,
                        time_saved_percent=(1-b/((a1+a2)/2))*100))
        rows.append(dict(kind='complete',qa_kv_checked=True,full_f32_bit_exact=True,cases=12,modes=list(MODES)))
        stdout=root/'component.stdout'
        def save_rows():stdout.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        save_rows();baseline=review(root,model)
        need(baseline['performance_gate_pass'] and baseline['actual_gguf_weight_intervals_verified']==6,'Valid CPU fixture rejected')
        controls=[]
        def reject(label,operation,restore):
            operation()
            try:
                review(root,model)
            except (ValueError,KeyError) as error:
                controls.append(dict(case=label,rejected=True,error=str(error)))
            else:
                raise ValueError('Negative CPU control incorrectly accepted: '+label)
            finally:restore()
        path=component/'parity-concurrent-case-000-provider-2-qa.f32';original=path.read_bytes()
        reject('raw_sha_corruption',lambda:path.write_bytes(bytes([original[0]^1])+original[1:]),lambda:path.write_bytes(original))
        def change_valid_hash(name,data):
            (component/name).write_bytes(data)
            for entry in files:
                if entry['path']==name:entry['sha256']=hashlib.sha256(data).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
        name=path.name
        reject('nonzero_ulp_with_matching_sha',lambda:change_valid_hash(name,struct.pack('<f',2.0)+original[4:]),lambda:change_valid_hash(name,original))
        guard=name[:-4]+'.guards'
        reject('guard_corruption_with_matching_sha',lambda:change_valid_hash(guard,b'\0'+b'\xa5'*127),lambda:change_valid_hash(guard,b'\xa5'*128))
        summary=next(row for row in rows if row['kind']=='summary');old=summary['b_us']
        reject('published_median_tampered',lambda:(summary.update(b_us=old+1),save_rows()),lambda:(summary.update(b_us=old),save_rows()))
        removed=rows.pop(next(i for i,row in enumerate(rows) if row['kind']=='timing'));save_rows()
        try:
            review(root,model)
        except ValueError as error:controls.append(dict(case='missing_timing_command',rejected=True,error=str(error)))
        else:raise ValueError('Missing timing command accepted')
        return dict(passed=True,gpu_work_submitted=False,fixture_kind='synthetic_cpu_only_not_model_quality_evidence',
                    valid_fixture_accepted=True,negative_controls=controls)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--model',type=Path)
    parser.add_argument('--source-archive',type=Path);parser.add_argument('--artifacts-archive',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--all-qkv-descriptors',action='store_true',help='CPU metadata audit of all 80 Qa/KV tensors; requires --model')
    args=parser.parse_args()
    try:
        need(sum((args.self_test,args.root is not None,args.all_qkv_descriptors))==1,
             'Choose exactly --root, --self-test or --all-qkv-descriptors')
        result=(all_qkv_descriptors(args.model) if args.all_qkv_descriptors else self_test() if args.self_test
                else review(args.root,args.model,args.source_archive,args.artifacts_archive))
        rc=0
    except (ValueError,KeyError,OSError,TypeError,tarfile.TarError) as error:
        result=dict(schema_version=1,evidence_valid=False,gpu_work_submitted=False,error=str(error));rc=2
    text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as stream:stream.write(text)
    print(text,end='');return rc


if __name__=='__main__':
    sys.exit(main())
