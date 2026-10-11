#!/usr/bin/env python3
"""Offline raw replay of actual Engine depth2 cancellation and continuations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
sys.dont_write_bytecode = True
from review_tree_benchmark import strict_json, require

ROLES = {'logits','residual','pre','previous_kv','previous_score','window','slots','pcache',
         'output_ids','output',*(f'{kind}{n}' for kind in ('compressed','index') for n in range(4))}
META = ('reason','output_tokens','pos','next_position','pcache_len','rng_state','engram_position','engram_history')

def inspect(report, raw):
    require(report.is_file() and not report.is_symlink(), 'Missing original report')
    data=report.read_bytes(); require(data.endswith(b'\n'), 'Truncated report')
    rows=[strict_json(line) for line in data.splitlines()]
    require(rows and all(type(row) is dict and row.get('pass',True) is True for row in rows),'Failed report')
    meta=rows[0];require(meta.get('kind')=='metadata' and meta.get('real_support_loaded') is True and
        meta.get('capture_mode')==0 and meta.get('greedy') is True and meta.get('pairs')==5 and
        meta.get('files_per_state')==18 and meta.get('performance_measurement') is False,'Invalid real-model scope')
    states={};requests={};digests={};payloads={};files=0
    for row in rows[1:-1]:
        kind=row.get('kind')
        if kind=='request':
            label=row['label'];require(label not in requests,'Duplicate request');requests[label]=row;continue
        require(kind in ('state','prefix_probe'),'Unexpected record')
        roles=set()
        for f in row.get('files',[]):
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
            if f['dtype']=='i32':require(len(b)%4==0,'Invalid token bytes')
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
    require(set(p.name for p in raw.iterdir() if p.is_file())==set(digests),'Unrecorded raw file')
    return {'audit_pass':True,'cpu_replay_only':True,'actual_engine_native_all_hit_partial1_partial2':True,
        'all_ordinary_raw_state_exact':True,'cache_continuations_exact':True,'neural_prefix_slot1_slot2_exact':True,
        'states':14,'raw_files':files,'comparisons':comparisons,'raw_sha256':digests,
        'report_sha256':hashlib.sha256(data).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('raw',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=inspect(a.report,a.raw);require(not a.output.exists(),'Output already exists');a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('audit_pass','states','raw_files','all_ordinary_raw_state_exact')}))
if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError,UnicodeError) as e:raise SystemExit('FAIL: '+str(e))
