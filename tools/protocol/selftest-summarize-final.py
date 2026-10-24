#!/usr/bin/env python3
"""CPU-only synthetic fixtures, never model measurements."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

SCRIPT = Path(__file__).with_name('summarize-final.py')
spec = importlib.util.spec_from_file_location('summarize_final', SCRIPT)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
env = {'MYDS_PREFIX_CACHE':'0', 'MYDS_KV_DSK':'0', 'MYDS_PHASE_PROFILE':'0',
       'MYDS_PREFILL_CHUNK_SIZE':'512', 'MYDS_MOE_CHUNKS':'9'}


def fixture(directory):
    events = []
    def emit(kind, **kw):
        events.append(dict(kind=kind, utc=f'2026-10-05T00:00:{len(events):02d}+00:00', **kw))
    emit('window_created')
    emit('preflight_passed')
    emit('production_stop')
    for wi, workload in enumerate(s.WORKLOADS):
        out = 26 if workload == 'short' else 64
        pf = [128, 1830, 7194][wi]
        tokens = list(range(1, out+1))
        token_hash = s.token_hash(tokens)
        for variant in s.VARIANTS:
            name = variant+'-'+workload
            emit('measurement_start', name=name, environment=copy.deepcopy(env))
            meta = dict(kind='metadata', model='/synthetic/model.gguf', repeats=3, warmup_runs=1,
                        request_fnv1a64='1111111111111111', prompt_fnv1a64='2222222222222222',
                        request_bytes=100, prompt_bytes=100, max_new_tokens=64, max_seq_len=262144,
                        temperature=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1, seed=42,
                        qos='user_interactive', prompt_renderer='parse_chat_prompt', clock='CLOCK_MONOTONIC',
                        transport='direct_byte_callback', decode_basis='first_to_last_callback/(tokens-1)',
                        environment=dict(copy.deepcopy(env), MYDS_PHASE_PROFILE_FILE=None))
            records = [meta, dict(kind='load', load_s=1, context_create_s=.1)]
            for run in range(4):
                factor = (1 if variant=='base' else .8) * (1+.01*run)
                pf_ms, decode_ms = 3*factor, 40*factor
                pf_s, decode_s = pf_ms*pf/1000, decode_ms*(out-1)/1000
                first = pf_s+.1
                records.append(dict(kind='sample', run=run, warmup=(run==0), generation_ok=True,
                        sample_valid=True, output_equal_to_warmup=True, cached_tokens=0,
                        prefill_tokens=pf, prompt_tokens=pf+1, output_tokens=out, decode_intervals=out-1,
                        token_ids=tokens, token_ids_fnv1a64=token_hash, output_bytes_fnv1a64='3333333333333333',
                        output_bytes=100, stop_reason=('eos' if out<64 else 'length'),
                        callback_intervals_ms=[decode_ms]*(out-1), prefill_ms_per_token=pf_ms,
                        decode_ms_per_token=decode_ms, prefill_s=pf_s, decode_s=decode_s,
                        wall_s=first+decode_s+.1, ttft_s=first, first_callback_s=first,
                        last_callback_s=first+decode_s))
            write(directory/(name+'-metrics.jsonl'), records)
            emit('measurement_end', name=name, returncode=0, elapsed_s=2)
        emit('exact_output_pass', request=workload, samples=4, output_tokens=out, token_ids_fnv1a64=token_hash)
    emit('restoration_start')
    emit('restoration_health', ready=True, restore_returncode=None)
    emit('window_complete')
    write(directory/'window.jsonl', events)


def write(path, records):
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))


def alter(directory, file, change):
    path=directory/file
    original=path.read_bytes()
    records=[json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    write(path, records)
    try:
        s.summarize(directory)
    except s.AuditError as exc:
        return str(exc)
    finally:
        path.write_bytes(original)
    raise AssertionError('Invalid fixture accepted: '+file)


with tempfile.TemporaryDirectory(prefix='myds-summary-selftest-') as td:
    directory=Path(td)
    fixture(directory)
    summary, rows = s.summarize(directory)
    assert len(summary['groups'])==6 and len(rows)==6
    short=summary['comparisons'][0]
    assert short['output_tokens']==26 and short['decode_intervals']==25
    assert short['prefill_tokens']==128
    assert abs(short['phases']['decode']['time_saved_pct_from_medians']-20)<1e-10
    failures={}
    mutations = {
        'raw_token_changed':('final-short-metrics.jsonl',lambda rs:rs[2]['token_ids'].__setitem__(0,999)),
        'byte_hash_changed':('final-short-metrics.jsonl',lambda rs:rs[2].__setitem__('output_bytes_fnv1a64','4444444444444444')),
        'cached_prefill':('base-short-metrics.jsonl',lambda rs:rs[2].__setitem__('cached_tokens',1)),
        'invalid_generation':('base-short-metrics.jsonl',lambda rs:rs[2].__setitem__('generation_ok',False)),
        'invalid_sample':('base-short-metrics.jsonl',lambda rs:rs[2].__setitem__('sample_valid',False)),
        'warm_formal_missing':('base-short-metrics.jsonl',lambda rs:rs.pop()),
        'decode_denominator_assumed64':('base-short-metrics.jsonl',lambda rs:rs[2].__setitem__('decode_intervals',63)),
        'legacy_profiler_zero_present':('base-short-metrics.jsonl',lambda rs:rs[0]['environment'].__setitem__('MYDS_PROFILE_PREFILL','0')),
        'prefix_cache_enabled':('base-short-metrics.jsonl',lambda rs:rs[0]['environment'].__setitem__('MYDS_PREFIX_CACHE','1')),
        'ssd_enabled':('base-short-metrics.jsonl',lambda rs:rs[0]['environment'].__setitem__('MYDS_KV_DSK','1')),
        'profiler_enabled':('base-short-metrics.jsonl',lambda rs:rs[0]['environment'].__setitem__('MYDS_PHASE_PROFILE','1')),
        'restore_not_healthy':('window.jsonl',lambda rs:next(r for r in rs if r['kind']=='restoration_health').__setitem__('ready',False)),
        'nonzero_return':('window.jsonl',lambda rs:next(r for r in rs if r['kind']=='measurement_end').__setitem__('returncode',1)),
        'missing_exact_pair':('window.jsonl',lambda rs:rs.remove(next(r for r in rs if r['kind']=='exact_output_pass'))),
        'incomplete_window':('window.jsonl',lambda rs:rs.pop()),
        'window_error':('window.jsonl',lambda rs:rs.insert(-1,dict(kind='window_error',utc=rs[-1]['utc'],error='synthetic'))),
        'measurement_timeout':('window.jsonl',lambda rs:rs.insert(-1,dict(kind='measurement_timeout',utc=rs[-1]['utc']))),
        'derived_metric_mismatch':('base-short-metrics.jsonl',lambda rs:rs[2].__setitem__('decode_ms_per_token',999)),
    }
    for name,(file,change) in mutations.items():
        failures[name]=alter(directory,file,change)
    print(json.dumps(dict(selftest_only=True, real_results_read=False, valid_fixture_pass=True,
                         short_output_tokens=short['output_tokens'], short_decode_intervals=short['decode_intervals'],
                         six_groups=True, six_csv_rows=True, time_saved_pct=20,
                         rejected_cases=len(failures), failures=failures),indent=2))
