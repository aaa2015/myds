#!/usr/bin/env python3
"""Offline raw replay of actual Engine depth2 cancellation and continuations."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import struct
import sys
sys.dont_write_bytecode = True
from review_tree_benchmark import strict_json, require

ROLES = {'logits','residual','pre','previous_kv','previous_score','window','slots','pcache',
         'output_ids','output',*(f'{kind}{n}' for kind in ('compressed','index') for n in range(4))}
META = ('reason','output_tokens','pos','next_position','pcache_len','rng_state','engram_position','engram_history')

REQUEST_KEYS = {'kind','label','spec','cap','cancel_after','hit_output_cancel','hit_index',
                'continuation','temperature','prompt_tokens','cached_tokens','prompt_bytes','prompt_fnv1a64','pass'}
COUNTERS = {'target_forward_calls','target_rows','batch2_calls','batch2_hits','batch2_misses',
            'batch2_failures','batch2_fallbacks','draft_cycles','depth2_calls','depth2_hits1',
            'depth2_hits2','depth2_misses','depth2_failures','depth2_fallbacks','draft_proposed','draft_accepted'}
LABELS = {f'case{c}_{arm}{suffix}' for c in range(5) for arm in ('candidate','ordinary')
          for suffix in (('', '_continue') if c<2 else ('',))}

def exact(row, expected, message):
    require(all(row.get(key)==value and type(row.get(key)) is type(value)
                for key,value in expected.items()),message)

def uint(value, message, positive=False):
    require(type(value) is int and value >= int(positive),message)
    return value

def fnv(data):
    value=14695981039346656037
    for byte in data:value=((value^byte)*1099511628211)&0xffffffffffffffff
    return f'{value:016x}'

def validate_protocol(states, requests, shared3_mode=None):
    """Validate actual request scheduling; CPU selftests exercise this gate."""
    require(set(states)==set(requests)==LABELS,'Missing/extra lifecycle case identities')
    base_prompt=None
    continuation_bytes=b' Continue with the next point.'
    for c in range(5):
        for suffix in ('','_continue') if c<2 else ('',):
            cl=f'case{c}_candidate{suffix}';ol=f'case{c}_ordinary{suffix}'
            a,b=states[cl],states[ol]
            continuing=bool(suffix)
            cap=16 if continuing else 256 if c<2 else c-1
            for label,spec in ((cl,True),(ol,False)):
                request=requests[label];state=states[label];d=state['decode']
                require(set(request)==REQUEST_KEYS | ({'shared3'} if shared3_mode is not None else set()),'Request schema differs '+label)
                if shared3_mode is not None:
                    shared=request['shared3']
                    require(isinstance(shared,dict) and set(shared)=={'mode','dense','gu','routed'},'Missing shared3 counters '+label)
                    exact(shared,{'mode':shared3_mode},'Shared3 selector differs '+label)
                    for provider,multiplier in (('dense',200),('gu',40),('routed',40)):
                        values=shared[provider]
                        keys={'requests','encoded','failures'} | ({'q8_encoded','q4_encoded'} if provider=='dense' else set())
                        require(isinstance(values,dict) and set(values)==keys and all(type(v) is int and v>=0 for v in values.values()),'Invalid shared3 counters '+label)
                        count=d['depth2_calls']*multiplier if shared3_mode==1 else 0
                        exact(values,{'requests':count,'encoded':count,'failures':0},'Actual shared3 execution absent/failed '+label)
                        if provider=='dense':
                            require(values['q8_encoded']+values['q4_encoded']==count and (values['q4_encoded']>0 if count else values['q4_encoded']==0),'Actual shared Q4 absent '+label)
                hit=(c+1) if not continuing and c<2 and spec else 0
                cancel=a['output_tokens'] if not continuing and c<2 and not spec else 0
                exact(request,dict(kind='request',label=label,spec=spec,cap=cap,cancel_after=cancel,
                    hit_output_cancel=hit,hit_index=hit,continuation=continuing,temperature=0),
                    'Fixed request sampling/cancel/cap/continuation differs '+label)
                exact(request,{'pass':True},'Failed request '+label)
                uint(request['prompt_tokens'],'Invalid prompt token count',True)
                uint(request['cached_tokens'],'Invalid cached token count')
                uint(request['prompt_bytes'],'Invalid rendered prompt size',True)
                require(isinstance(request['prompt_fnv1a64'],str) and re.fullmatch('[0-9a-f]{16}',request['prompt_fnv1a64']),
                        'Invalid rendered prompt FNV '+label)
                require(COUNTERS<=set(d) and all(type(v) is int and v>=0 for v in d.values()),'Incomplete/invalid native counters '+label)
                require(d['depth2_failures']==d['depth2_fallbacks']==d['batch2_failures']==d['batch2_fallbacks']==0,
                        'Native failure/fallback '+label)
                require(d['depth2_calls']==d['depth2_hits1']+d['depth2_hits2']+d['depth2_misses'] and
                    d['batch2_calls']==d['batch2_hits']+d['batch2_misses'] and d['depth2_calls']<=d['batch2_calls'],
                    'Native outcomes differ '+label)
                require(d['target_rows']-d['target_forward_calls']-d['batch2_calls']==d['depth2_calls'],
                        'Actual native three-row work differs '+label)
                require(d['draft_accepted']<=d['draft_proposed'],'Accepted exceeds proposed '+label)
                if not spec:
                    require(all(d[k]==0 for k in ('depth2_calls','batch2_calls','draft_proposed','draft_accepted','draft_cycles')),
                            'Ordinary reference executed drafter '+label)
                if continuing:
                    previous=states[f'case{c}_{"candidate" if spec else "ordinary"}']
                    require(request['cached_tokens']==previous['pos'],'Continuation did not use own retained prefix '+label)
                    require(request['prompt_bytes']==len(continuation_bytes) and request['prompt_fnv1a64']==fnv(continuation_bytes),
                            'Continuation prompt identity differs '+label)
                    require((state['reason']==1 and state['output_tokens']==cap) or
                            (state['reason']==0 and 0<=state['output_tokens']<cap),'Continuation stop/length/EOS differs '+label)
                else:
                    require(request['cached_tokens']==0,'Cold request used cached prefix '+label)
                    identity=(request['prompt_tokens'],request['prompt_bytes'],request['prompt_fnv1a64'])
                    if base_prompt is None:base_prompt=identity
                    require(identity==base_prompt,'Fixed initial rendered prompt differs '+label)
                    if c<2:
                        require(state['reason']==4 and 0<state['output_tokens']<=cap,'Actual callback cancellation absent '+label)
                        if spec:require(d['depth2_hits2']>0,'No actual all-hit native cycle '+label)
                    else:
                        require(state['reason']==1 and state['output_tokens']==cap,'Short output tail stop/count differs '+label)
                        if spec and c<4:
                            require(d['depth2_calls']==d['batch2_calls']==d['draft_proposed']==d['draft_cycles']==0,
                                    'One/two-output tail executed speculative work '+label)
                        if spec and c==4:
                            require(d['depth2_calls']==0 and d['batch2_calls']>0 and d['draft_proposed']>0,
                                    'Three-output tail did not execute actual depth1 verifier '+label)
            require(requests[cl]['prompt_tokens']==requests[ol]['prompt_tokens'],'Candidate/reference prompt tokenization differs '+cl)
    return True

RES_UINT = ('allocation_count','allocated_size','device_allocated_size','live_mapped_allocations','pending_unregister')
RES_KEYS = {'kind','label','supported','active','heartbeat_running','pass',*RES_UINT}

def validate_residency(rows):
    expected=[('metadata',None),('residency','model_baseline')]
    for c in range(5):
        expected.append(('residency',f'case{c}_loaded'))
        for suffix in ('','_continue') if c<2 else ('',):
            for arm in ('candidate','ordinary'):
                label=f'case{c}_{arm}{suffix}'
                expected.extend((('request',label),('state',label)))
        if c==1:expected.append(('prefix_probe',None))
        expected.append(('residency',f'case{c}_freed'))
    expected.extend((('residency','model_freed'),('residency','runtime_cleanup'),('complete',None)))
    require([(row.get('kind'),row.get('label')) for row in rows]==expected,'Native request/resource order differs')
    records=[row for row in rows if row['kind']=='residency']
    require(len(records)==13,'Incomplete residency lifecycle evidence')
    base=records[0]
    for row in records:
        require(set(row)==RES_KEYS,'Residency schema differs')
        require(all(type(row[k]) is int and 0<=row[k]<(1<<64) for k in RES_UINT),'Invalid residency unsigned count')
        require(all(type(row[k]) is bool for k in ('supported','active','heartbeat_running','pass')),'Invalid residency flags')
        exact(row,{'supported':True,'pending_unregister':0,'pass':True},'Unsupported/pending/failed residency evidence')
    exact(base,{'active':True,'heartbeat_running':True},'Baseline residency inactive')
    require(base['allocation_count']>0,'Empty model baseline')
    baseline_keys=('allocation_count','allocated_size','live_mapped_allocations')
    for c in range(5):
        loaded,freed=records[1+c*2:3+c*2]
        for row in (loaded,freed):exact(row,{'active':True,'heartbeat_running':True},'Context residency inactive')
        require(all(loaded[k]>base[k] for k in baseline_keys),'Context load did not add residency allocations')
        require(all(freed[k]==base[k] for k in baseline_keys),'Context free did not return to exact model baseline')
    exact(records[-2],{'allocation_count':0,'live_mapped_allocations':0},'Model allocations remain registered')
    exact(records[-1],{'allocation_count':0,'allocated_size':0,'live_mapped_allocations':0,'active':False,'heartbeat_running':False},
          'Runtime cleanup incomplete')
    return records

def inspect(report, raw):
    require(report.is_file() and not report.is_symlink(), 'Missing original report')
    require(raw.is_dir() and not raw.is_symlink(),'Missing/linked raw directory')
    data=report.read_bytes(); require(data.endswith(b'\n'), 'Truncated report')
    rows=[strict_json(line) for line in data.splitlines()]
    require(rows and all(type(row) is dict and row.get('pass',True) is True for row in rows),'Failed report')
    meta=rows[0];require(meta.get('kind')=='metadata' and meta.get('real_support_loaded') is True and
        meta.get('capture_mode')==0 and meta.get('greedy') is True and meta.get('pairs')==5 and
        meta.get('files_per_state')==18 and meta.get('performance_measurement') is False,'Invalid real-model scope')
    shared3_mode=meta.get('shared3_mode')
    require(shared3_mode is None or type(shared3_mode) is int and shared3_mode in (0,1),'Invalid shared3 mode')
    require(set(meta)=={'kind','schema_version','target','support','real_support_loaded','capture_mode','greedy','performance_measurement','pairs','files_per_state'} | ({'shared3_mode'} if shared3_mode is not None else set()) and
        type(meta['schema_version']) is int and meta['schema_version']==1 and all(type(meta[k]) is str and meta[k] for k in ('target','support')),
        'Metadata schema/model identity differs')
    residency=validate_residency(rows)
    states={};requests={};digests={};payloads={};files=0
    for row in rows[1:-1]:
        kind=row.get('kind')
        if kind=='request':
            label=row['label'];require(label not in requests,'Duplicate request');requests[label]=row;continue
        if kind=='residency':
            continue
        require(kind in ('state','prefix_probe'),'Unexpected record')
        if kind=='state':
            require(set(row)=={'kind','label',*META,'decode','files','pass'} and row['pass'] is True,'State schema differs')
            require(isinstance(row['rng_state'],str) and re.fullmatch('0|[1-9][0-9]*',row['rng_state']) and int(row['rng_state'])<(1<<64),'Invalid RNG state')
            require(isinstance(row['engram_history'],list) and len(row['engram_history'])==3 and all(type(v) is int and 0<=v<129280 for v in row['engram_history']),'Invalid Engram history')
        else:
            require(set(row)=={'kind','first_slot','count','stage_slots','kv_rows','full_dispatches','prefix_dispatches','files','slot1_slot2_bitwise_equal','other_heads_nan','pass'} and row['pass'] is True,'Prefix schema differs')
        roles=set()
        require(isinstance(row.get('files'),list),'Missing raw descriptors')
        for f in row['files']:
            require(isinstance(f,dict) and set(f)=={'path','role','dtype','bytes','fnv1a64'},'Raw descriptor schema differs')
            name=f['path'];require(type(name) is str and Path(name).name==name,'Unsafe raw path')
            require(name not in digests,'Duplicate global raw path '+name)
            label=row['label'] if kind=='state' else 'prefix'
            require(name==label+'_'+f['role']+'.'+f['dtype'],'Raw path/role binding differs')
            p=raw/name;require(p.is_file() and not p.is_symlink(),'Missing raw file '+name)
            b=p.read_bytes();require(type(f['bytes']) is int and len(b)==f['bytes'],'Raw size differs '+name)
            h=14695981039346656037
            for byte in b:h=((h^byte)*1099511628211)&0xffffffffffffffff
            require(f['fnv1a64']==f'{h:016x}','Raw FNV differs '+name)
            require(f['dtype'] in ('f32','i32','bin'),'Invalid dtype')
            if f['dtype']=='f32':require(len(b)%4==0 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',b)),'Nonfinite tensor '+name)
            if f['dtype']=='i32':require(len(b)%4==0 and all(0<=v[0]<129280 for v in struct.iter_unpack('<i',b)),'Invalid token bytes')
            if f['role'].startswith(('compressed','index')):
                require(len(b)%2==0 and all((v[0]&0x7f80)!=0x7f80 for v in struct.iter_unpack('<H',b)),'Nonfinite BF16 cache')
            role=f['role'];require(role not in roles,'Duplicate role');roles.add(role)
            if kind=='state':
                pos=row['pos'];count=row['output_tokens']
                require(type(pos) is int and 0<pos<=4096 and type(count) is int and 0<=count<=256,'Invalid state bounds')
                sizes={'logits':129280*4,'residual':4*5120*4,'pre':16,'previous_kv':4*512*4,
                    'previous_score':4*512*4,'window':40*min(pos,128)*512*4,'slots':3*5120*4,
                    'pcache':pos*4,'output_ids':count*4}
                sizes.update({f'{k}{o}':(pos//(1 if o==3 else 2))*(1024 if k=='compressed' else 256)
                    for k in ('compressed','index') for o in range(4)})
                require(role=='output' or role in sizes and len(b)==sizes[role],'Wrong tensor geometry '+name)
                dtype='i32' if role in ('pcache','output_ids') else 'bin' if role=='output' or role.startswith(('compressed','index')) else 'f32'
                require(f['dtype']==dtype,'Wrong state dtype '+name)
            else:
                prefix_sizes={f'slot{slot}_role{r}':(4*5120*4,5120*4,5120*4,129280*4)[r]
                    for slot in (1,2) for r in range(4)}
                require(role in prefix_sizes and len(b)==prefix_sizes[role] and f['dtype']=='f32','Wrong prefix geometry')
            digests[name]=hashlib.sha256(b).hexdigest();payloads[name]=b;files+=1
        if kind=='state':
            require(roles==ROLES,'Incomplete state roles')
            label=row['label'];require(label not in states and label in requests,'State/request mismatch');states[label]=row
            require(row['pos']==row['next_position']==row['pcache_len']==row['engram_position'],'State positions differ')
            d=row['decode'];require(all(type(v) is int and v>=0 for v in d.values()),'Invalid decode counter')
            require(d['depth2_failures']==d['depth2_fallbacks']==0,'Failed native graph')
            require(d['depth2_calls']==d['depth2_hits1']+d['depth2_hits2']+d['depth2_misses'],'Depth2 outcomes differ')
            require(d['target_rows']-d['target_forward_calls']-d['batch2_calls']==d['depth2_calls'],'Actual linear three-row work differs')
            require(d['draft_accepted']<=d['draft_proposed'],'Accepted exceeds proposed')
            require(len(payloads[label+'_output_ids.i32'])==row['output_tokens']*4,'Output IDs count differs')
            require(len(payloads[label+'_pcache.i32'])==row['pcache_len']*4,'Pcache size differs')
        else:
            require(row.get('first_slot')==1 and row.get('count')==2 and row.get('stage_slots')==3 and row.get('kv_rows')==4 and
                row.get('slot1_slot2_bitwise_equal') is True and row.get('other_heads_nan') is True and
                0<row['prefix_dispatches']<row['full_dispatches'] and len(roles)==8,'Incomplete neural prefix proof')
    require(len(states)==14 and len(requests)==14 and files==260,'Incomplete lifecycle evidence')
    validate_protocol(states,requests,shared3_mode)
    comparisons=[]
    for c in range(5):
        for suffix in ('','_continue') if c<2 else ('',):
            cl=f'case{c}_candidate{suffix}';ol=f'case{c}_ordinary{suffix}';a=states[cl];b=states[ol]
            require(all(a[k]==b[k] for k in META),'Ordinary state metadata differs '+cl)
            for role in ROLES:
                af=next(f['path'] for f in a['files'] if f['role']==role);bf=next(f['path'] for f in b['files'] if f['role']==role)
                require(payloads[af]==payloads[bf],'Ordinary raw state differs '+cl+'/'+role)
                comparisons.append({'candidate':af,'ordinary':bf,'sha256':digests[af]})
            require(states[ol]['decode']['depth2_calls']==0 and states[ol]['decode']['draft_proposed']==0,'Ordinary reference used drafter')
            if not suffix and c<2:
                require(requests[cl]['hit_output_cancel']==c+1 and requests[cl]['hit_index']==c+1 and
                    a['decode']['depth2_hits2']>0 and a['reason']==4,'Actual all-hit truncation absent')
            if suffix:
                require(requests[cl]['continuation'] is True and requests[ol]['continuation'] is True and
                    requests[cl]['cached_tokens']==states[f'case{c}_candidate']['pos'] and
                    requests[ol]['cached_tokens']==states[f'case{c}_ordinary']['pos'] and
                    requests[cl]['prompt_tokens']==requests[ol]['prompt_tokens'],'Missing actual prefix resume')
    require(rows[-1].get('kind')=='complete' and rows[-1].get('states')==14 and rows[-1].get('raw_files')==files and
        rows[-1].get('native_all_hit_partial1_partial2') is True,'Missing completion')
    require(set(p.name for p in raw.iterdir())==set(digests),'Unrecorded raw file')
    require(set(rows[-1])=={'kind','states','raw_files','native_all_hit_partial1_partial2','pass'} and rows[-1]['pass'] is True,'Completion schema differs')
    return {'audit_pass':True,'cpu_replay_only':True,'actual_engine_native_all_hit_partial1_partial2':True,
        'all_ordinary_raw_state_exact':True,'cache_continuations_exact':True,'neural_prefix_slot1_slot2_exact':True,'prefix_parity_scope':'Native fixture memcmp against full inference; archive contains only prefix tensors',
        'short_tail_depth1_observed':True,'strict_request_protocol':True,'reviewer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'residency_churn_exact_baseline':True,'residency_records':residency,
        'residency_scope':'Metal SDK residency accounting; device size recorded; no physical-page or peak-RSS claim',
        'states':14,'raw_files':files,'comparisons':comparisons,'raw_sha256':digests,
        'report_sha256':hashlib.sha256(data).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('raw',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=inspect(a.report,a.raw);require(not a.output.exists(),'Output already exists');a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('audit_pass','states','raw_files','all_ordinary_raw_state_exact')}))
if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError,UnicodeError) as e:raise SystemExit('FAIL: '+str(e))
