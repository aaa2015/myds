#!/usr/bin/env python3
"""Independent normal localgrow probe/full-sequence CPU reader.

Pure read_arm/compare_arms/growth_gate/performance_gate are controller-callable.
No inference, Metal, service, source mutation, or timing samples are generated.
"""
import argparse,array,ctypes,hashlib,json,math,statistics,struct,subprocess,sys,tarfile,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
FLAG='MYDS_PREFILL_MOE_WORKSPACE'
CLEAR_FLAG='MYDS_PREFILL_PROMPT_VIEW_CLEAR'
LAYOUT=('setup','grow','retain')
METRICS=('phys_footprint_bytes','rss_bytes','device_allocated_bytes')
SCALARS=['ctx_pos','rng','pcache_len','anchor_len','next','engram_position','v41_valid','max_seq_len','max_new_tokens','seed','continue_from_cache']

class ByteHashes:
    def __init__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='myds-fullmodel-cpu-fnv-')
        root=Path(self.temporary.name);source=root/'fnv.c';library=root/('fnv.dylib' if sys.platform=='darwin' else 'fnv.so')
        source.write_text('#include <stdint.h>\n#include <stddef.h>\n'
          'uint64_t hash_span(uint64_t h,const unsigned char *p,size_t n){'
          'for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}return h;}\n')
        command=['cc','-O3','-shared','-fPIC',str(source),'-o',str(library)]
        done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(done.returncode==0,'CPU byte hashing helper compilation failed: '+done.stderr)
        self.library=ctypes.CDLL(str(library));self.function=self.library.hash_span
        self.function.argtypes=[ctypes.c_uint64,ctypes.c_char_p,ctypes.c_size_t]
        self.function.restype=ctypes.c_uint64;self.cache={}

    def file(self,path):
        st=path.stat();key=(str(path.resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in self.cache:
            digest=hashlib.sha256();value=14695981039346656037
            with path.open('rb') as stream:
                while True:
                    block=stream.read(1048576)
                    if not block:break
                    digest.update(block);value=self.function(value,block,len(block))
            self.cache[key]=dict(bytes=st.st_size,sha256=digest.hexdigest(),fnv1a64_bytes=format(value,'016x'))
        return self.cache[key]

    def bytes(self,data):return format(self.function(14695981039346656037,data,len(data)),'016x')
    def close(self):self.temporary.cleanup()


def file_difference(left,right,f32=False):
    first=None;different=0;total=0;float_differences=0
    with left.open('rb') as a,right.open('rb') as b:
        while True:
            x,y=a.read(1048576),b.read(1048576)
            if not x and not y:break
            if x!=y:
                for offset,(u,v) in enumerate(zip(x,y)):
                    if u!=v:
                        if first is None:first=total+offset
                        different+=1
                different+=abs(len(x)-len(y))
                if first is None:first=total+min(len(x),len(y))
                if f32:
                    need(len(x)%4==len(y)%4==0,'Misaligned raw logits')
                    float_differences+=sum(u!=v for u,v in zip(struct.iter_unpack('<I',x),struct.iter_unpack('<I',y)))
                    float_differences+=abs(len(x)-len(y))//4
            total+=max(len(x),len(y))
    return dict(equal=different==0,different_bytes=different,first_different_byte=first,
                f32_bit_mismatches=float_differences if f32 else None)


def need(value, message):
    if not value:
        raise ValueError(message)


def parse(text):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'Duplicate JSON key: '+key); result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+v)))


def load(path): return parse(path.read_text())
def rows(path): return [parse(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path, count=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while count is None or count:
            block = stream.read(min(count,1048576) if count is not None else 1048576)
            if not block: break
            digest.update(block)
            if count is not None: count -= len(block)
    need(count in (None,0), 'Short original hashed interval')
    return digest.hexdigest()


def bounded(directory, name):
    need(type(name) is str and name not in ('','.','..') and Path(name).name == name,
         'Flat original raw path required')
    target = directory/name
    need(target.is_file() and not target.is_symlink() and target.resolve().parent == directory.resolve(),
         'Original state payload missing or escaped')
    return target


def workspace(value, setting, B):
    need(type(value) is dict and value['mode'] == setting and value['capacity'] == 2048 and value['moe_batch'] == B,
         'Actual retained/grown workspace capacity differs')
    need(value['failures'] == 0 and value['requests'] == value['mm_encoded']+value['mv_encoded'] and
         value['requests'] > 0 and value['mm_encoded'] > 0, 'Actual complete MoE wrapper counters differ')
    legacy = ('legacy_gate_bytes','legacy_up_bytes','legacy_mid_bytes')
    if setting == 0:
        need(all(value[k] == B*6*2304*4 for k in legacy) and value['scratch_owner_bytes'] == 3*B*6*2304*4,
             'Original F32 owners differ')
    else:
        a = value['arena']; F = a['f32_capacity']
        need(all(value[k] == 0 for k in legacy) and a['mm_capacity'] == B and a['mm_mid_bytes'] == B*6*2304*2 and
             type(F) is int and 0 <= F <= B and all(a[k] == F*6*2304*4 for k in
                 ('f32_gate_bytes','f32_up_bytes','f32_mid_bytes')), 'Compact real owner extents differ')
        need(a['requests']==a['mm_encoded']+a['mv_encoded'] and a['mm_encoded']>0 and a['failures']==0 and a['requests']<=value['requests'], 'Arena generation counters differ')
        total = sum(a[k] for k in ('mm_mid_bytes','f32_gate_bytes','f32_up_bytes','f32_mid_bytes','retired_bytes'))
        need(a['allocated_bytes'] == value['scratch_owner_bytes'] == total and
             a['peak_allocated_bytes'] >= total and value['workspace_owner_bytes'] >= total,
             'Actual active/retired owner byte accounting differs')


def state(directory, label, sample, hashes, outputs):
    manifest = load(directory/'manifest.json')
    need(manifest['schema_version'] == 1 and manifest['phase'] == label and
         manifest['scope'] == 'same_context_moe_workspace_localgrow_sequence' and manifest['complete'] is True and
         manifest['complete_live_state'] is True and manifest['scalar_order'] == SCALARS,
         'Complete same-context successful state export required')
    scalars = struct.unpack('<11Q', bounded(directory,'context-scalars.u64le').read_bytes())
    positions = list(struct.iter_unpack('<II', bounded(directory,'snapshot-positions.u32le').read_bytes()))
    need(len(positions) == 44 and scalars[7:10] == (327680,outputs,42) and
         scalars[6] == 1 and scalars[10] in (0,1) and 0 <= scalars[2] <= 327680 and 0 <= scalars[3] <= 1024,
         'Original complete context configuration differs')
    next_pos = scalars[4]
    need(0 < next_pos <= 327680 and scalars[0] == next_pos == scalars[5] and
         sample['prompt_tokens']-1 <= next_pos <= sample['prompt_tokens']+outputs,
         'Complete live position/Engram extent differs')
    for name, index in (('ctx_pos',0),('pcache_len',2),('anchor_len',3),('next_position',4),('engram_position',5)):
        need(manifest[name] == scalars[index], 'Manifest/raw context metadata differs: '+name)
    need(manifest['rng_state'] == format(scalars[1],'016x') and manifest['anchor_slot'] == 43 and
         manifest['requested_prefill_upper_bound'] == next_pos and len(manifest['limitations']) >= 2,
         'Raw RNG/published-state boundary differs')
    expected = {'context-scalars.u64le':88,'engram-tail.u32le':12,'snapshot-positions.u32le':352,
                'prefix-tokens.i32le':scalars[2]*4,'anchor-tokens.i32le':scalars[3]*4,
                'tokens.i32le':outputs*4,'output.bin':sample['output_bytes'],'final-logits.f32':129280*4}
    for layer in range(40): expected[f'layer-{layer:02d}-window-active.f32'] = min(next_pos,128)*2048
    for owner in range(4):
        if owner < 3:
            for kind in ('kv','score'): expected[f'owner-{owner}-previous-{kind}.f32'] = 2048
        count = next_pos//(1 if owner == 3 else 2)
        expected[f'owner-{owner}-compressed-active.bf16'] = count*1024
        expected[f'owner-{owner}-index-active.bf16'] = count*256
    published = []
    for slot, (position, staged) in enumerate(positions):
        need(not(position and staged) and staged <= next_pos and
             ((slot == 43 and (not position or position == scalars[3])) or (slot != 43 and position <= next_pos)),
             'Published/staged snapshot or persistent anchor identity differs')
        if not position: continue
        published.append(dict(slot=slot,position=position))
        name = f'snapshot-{slot:02d}-engram-position-tail.u32le'
        expected[name] = 16
        need(struct.unpack('<4I',bounded(directory,name).read_bytes())[0] == position,
             'Historical Engram position differs from its published snapshot')
        for layer in range(40): expected[f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32'] = min(position,128)*2048
        for owner in range(3):
            for kind in ('kv','score'): expected[f'snapshot-{slot:02d}-owner-{owner}-previous-{kind}.f32'] = 2048
    names = [r['path'] for r in manifest['files']]
    need(len(names) == len(set(names)) == manifest['file_count'] and set(names) == set(expected) and
         {p.name for p in directory.iterdir()} == set(expected)|{'manifest.json'}, 'Exact complete raw state file set differs')
    files = {}
    for item in manifest['files']:
        name = item['path']; value = hashes.file(bounded(directory,name))
        need(value['bytes'] == item['bytes'] == expected[name] and value['fnv1a64_bytes'] == item['fnv1a64_bytes'],
             'Complete original state bytes/FNV differ: '+name)
        files[name] = value
    need(files['tokens.i32le']['fnv1a64_bytes'] == sample['tokens_fnv1a64_bytes'] and
         files['output.bin']['fnv1a64_bytes'] == sample['output_fnv1a64_bytes'], 'Actual callback tokens/text bytes differ')
    logits = array.array('I'); logits.frombytes(bounded(directory,'final-logits.f32').read_bytes())
    if sys.byteorder != 'little': logits.byteswap()
    need(all((v & 0x7f800000) != 0x7f800000 and v != 0xa5a5a5a5 for v in logits), 'Complete vocabulary logits nonfinite/poison')
    return dict(path=str(directory),manifest_sha256=sha(directory/'manifest.json'),files=files,
                published_snapshots=published,rng_state=manifest['rng_state'])


def memory(data, trace):
    samples = rows(trace)
    need(samples and all(r['kind'] == 'memory_sample' and type(r['monotonic_s']) in (int,float) and
         math.isfinite(r['monotonic_s']) and r['run'] in range(3) and r['phase'] in range(4) and
         r['value']['task_valid'] is True and r['value']['device_valid'] is True and
         all(type(r['value'][k]) is int and r['value'][k] >= 0 for k in METRICS) for r in samples),
         'Complete valid original10ms memory trace required')
    need(all(a['monotonic_s'] <= b['monotonic_s'] for a,b in zip(samples,samples[1:])), 'Nonmonotonic raw memory trace')
    reports = [r for r in data if r['kind'] == 'memory_phase']
    need([(r['run'],r['phase'],r['label']) for r in reports] ==
         [(run,phase,label) for run in range(3) for phase,label in ((1,'request_prepare'),(2,'generation'),(3,'raw_export'))],
         'Exact all-stage memory phase order differs')
    out = []
    for report in reports:
        group = [r for r in samples if r['run'] == report['run'] and r['phase'] == report['phase']]
        need(report['interval_ms'] == 10 and report['sampling_ok'] is True and len(group) == report['observed_samples'] > 0,
             'Raw observed phase coverage differs')
        current = report['current']
        need(current['task_valid'] is True and current['device_valid'] is True and
             all(type(current[k]) is int and current[k]>=0 for k in METRICS), 'Invalid actual phase-end memory checkpoint')
        peaks = {key:max(r['value'][key] for r in group) for key in METRICS}
        for suffix,key in (('phys_footprint','phys_footprint_bytes'),('rss','rss_bytes'),('device_allocated','device_allocated_bytes')):
            need(report['observed_peak_'+suffix+'_bytes'] == peaks[key], 'Published peak differs from every original sample')
        gaps = [b['monotonic_s']-a['monotonic_s'] for a,b in zip(group,group[1:])]
        need(report['phase'] != 2 or len(group)>1, 'Generation needs multiple observed points')
        out.append(dict(run=report['run'],step=LAYOUT[report['run']],phase=report['phase'],label=report['label'],
            observed_samples=len(group),maximum_observed_gap_ms=max(gaps)*1000 if gaps else 0,peaks=peaks,
            current=report['current']))
    return dict(trace_sha256=sha(trace),original_samples=len(samples),phases=out,
                continuous_peak_coverage_proven=False,allocation_shadow_observed=False)


def close(a,b,tolerance=1e-8):
    return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance


def read_layout(path, hashes):
    value=load(path)
    need(value.get('schema_version')==1 and value.get('complete') is True and
         value.get('scope')=='process_local_actual_binding_identity','Complete actual layout descriptor')
    bindings=value['bindings'];need(len(bindings)==value['binding_count'] and len(bindings)<=131072,'All bounded bindings required')
    need(len({(b['name'],b['index']) for b in bindings})==len(bindings),'Unique binding identities')
    for b in bindings:
        need(type(b['name']) is str and b['name'] and b['scope'] in range(1,7) and b['kind'] in (1,2,3,4), 'Binding name/scope/kind')
        need(all(type(b[k]) is int and b[k]>=0 for k in ('index','wrapper_id','resource_id','offset','bytes','is_view')) and b['is_view'] in (0,1),'Bounded identity fields')
    if value['present']:
        need(value['implementation_version']==4 and value['capacity']==2048 and value['moe_batch'] in (8192,16384) and value['cw_id']>0 and value['moe_generation']>0,'Actual implementation4 CW/batch')
        owners=[b for b in bindings if b['scope']==1 and b['wrapper_id']]
        views=[b for b in bindings if b['scope'] in (2,5) and b['wrapper_id']]
        need(len(owners)==value['stable_owners'] and len(views)==value['stable_views'] and sum(b['bytes'] for b in owners)==value['stable_owner_bytes'],'Enumerated stable counts/bytes')
        owner_words=[x for b in owners for x in (b['wrapper_id'],b['resource_id'],b['bytes'])]
        oh=hashes.bytes(struct.pack('<'+'Q'*len(owner_words),*owner_words));vh=hashes.bytes(struct.pack('<'+'Q'*len(views),*(b['wrapper_id'] for b in views)))
        need(int(oh,16)==value['stable_owner_fingerprint'] and int(vh,16)==value['stable_view_fingerprint'],'Actual enumeration does not match cached stable fingerprints')
    else:need(not bindings and not value['cw_id'],'Absent layout may not claim bindings')
    return dict(path=str(path),sha256=sha(path),value=value)


def stable_bindings(before,after,retain=False):
    a,b=before['value'],after['value'];need(a['present'] and b['present'],'Actual before/after layouts')
    left={(v['name'],v['index']):v for v in a['bindings']};right={(v['name'],v['index']):v for v in b['bindings']}
    need(left.keys()==right.keys(),'Binding coverage changed')
    for key,x in left.items():
        y=right[key];need((x['kind'],x['scope'])==(y['kind'],y['scope']),'Binding semantics changed')
        if x['scope'] in (1,2) or (retain and x['scope'] in (3,4,6)):
            need(x==y,'Actual fixed resource/view tuple changed: '+str(key))
        elif x['scope']==5:
            need(x['wrapper_id']==y['wrapper_id'],'Stable dynamic wrapper replaced: '+str(key))
    for key in ('cw_id','capacity','stable_owner_fingerprint','stable_view_fingerprint','stable_owner_bytes','stable_owners','stable_views'):
        need(a[key]==b[key],'Non-MoE CW/layout changed: '+key)
    return True


def layout_transition(before,after,stage,setting):
    a,b=before['value'],after['value'];need(b['present'] and b['compact']==setting and b['moe_batch']==(8192 if stage==0 else 16384),'Stage actual batch/layout')
    if stage==0:
        need(not a['present'] and b['moe_generation']==1 and b['localgrow_attempts']==b['localgrow_successes']==b['localgrow_failures']==0,'Fresh sequence initial workspace')
    elif stage==1 and setting:
        stable_bindings(before,after)
        need(a['moe_batch']==8192 and b['moe_generation']==a['moe_generation']+1 and
             b['localgrow_attempts']==a['localgrow_attempts']+1 and b['localgrow_successes']==a['localgrow_successes']+1 and
             b['localgrow_failures']==a['localgrow_failures']==0,'Actual successful local transaction')
        need(b['down_id']!=a['down_id'] and b['arena_id']!=a['arena_id'] and all(x!=y for x,y in zip(a['acc_ids'],b['acc_ids'])),'All B-dependent pools replaced')
    elif stage==1:
        need(a['present'] and a['moe_batch']==8192 and a['cw_id']!=b['cw_id'] and b['localgrow_successes']==0,'Legacy control whole-CW growth')
    else:
        stable_bindings(before,after,True)
        need(all(a[k]==b[k] for k in ('moe_batch','moe_generation','down_id','arena_id','acc_ids','acc_bytes','localgrow_attempts','localgrow_successes','localgrow_failures')),'Retained B pools changed')
    return True


def counter_transition(previous,current,before_layout,after_layout,setting):
    if not setting:return
    a,b=before_layout['value'],after_layout['value'];keys=('requests','mm_encoded','mv_encoded')
    cw_delta={k:current[k]-(previous[k] if previous and a['cw_id']==b['cw_id'] else 0) for k in keys}
    arena=current['arena'];old=previous['arena'] if previous and a['arena_id']==b['arena_id'] else {}
    need(all(cw_delta[k]==arena[k]-old.get(k,0) and cw_delta[k]>=0 for k in keys),'CW lifetime and arena-generation encoding deltas disagree')


def observer(events_path,sequence,setting,gold,layouts):
    events=rows(events_path);need(not gold or not events,'Exact production gold must not use new observer')
    if gold:return dict(events=[],coverage=False,path=str(events_path),sha256=sha(events_path))
    for e in events:
        need(e['kind']=='growth_lifetime' and e['sequence']==sequence and e['stage'] in range(3) and
             e['in_generation'] is True and e['implementation_version']==4 and math.isfinite(e['monotonic_s']), 'Actual normal observer scope')
        v=e['actual'];need(v['task_valid'] is True and v['device_valid'] is True and all(type(v[k]) is int and v[k]>=0 for k in METRICS),'Actual observer process/device values')
    need(all(a['monotonic_s']<=b['monotonic_s'] and a['growth_sequence']<b['growth_sequence'] for a,b in zip(events,events[1:])),'Observer time/sequence order')
    for stage in range(3):
        selected=[e for e in events if e['stage']==stage]
        expected=[] if stage==2 else ([1,2] if stage==1 and setting else ([3,4] if setting else [5,6,7]))
        need([e['event'] for e in selected]==expected,'Known allocation lifetime checkpoints missing/extra')
        for e in selected:
            need(e['capacity']==2048 and e['new_batch']==(8192 if stage==0 else 16384) and e['old_batch']==(0 if stage==0 else 8192),'Observer actual growth extent')
        if stage==1 and setting:
            a,b=(x['value'] for x in layouts[stage]);first,last=selected
            need(last['growth_sequence']==first['growth_sequence']+1 and first['cw_id']==a['cw_id']==b['cw_id'] and
                 first['old_arena_id']==a['arena_id'] and first['new_arena_id']==b['arena_id'] and
                 first['old_down_id']==a['down_id'] and first['new_down_id']==b['down_id'] and
                 first['stable_owner_bytes']==a['stable_owner_bytes'] and first['old_pool_bytes']>0 and first['new_pool_bytes']>first['old_pool_bytes'], 'Simultaneous-live local allocation boundary identity')
    return dict(events=events,coverage=True,path=str(events_path),sha256=sha(events_path),scope='Known allocation-lifetime checkpoints plus10ms runtime; not exact hardware residency maximum')


def normal_scope(meta,setting,env,gold=False):
    need(type(setting)is int and setting in (0,1) and (not gold or setting==0),'Normal original-production or candidate setting')
    need(meta.get('implementation_version')==4 and type(meta.get('prompt_view_clear_mode'))is int and meta['prompt_view_clear_mode']==setting and meta.get('prompt_lifetime_observer')==(not gold) and meta.get('deferred_lifetime_observer')is False and meta.get('diagnostic_prompt_mode')is False,'Normal v4 mode and prompt-only lifetime observer')
    need(env.get(FLAG)==str(setting) and env.get(CLEAR_FLAG)==str(setting),'Both explicitly bound v4 flags')
    return True

def prompt_observer(path,sequence,setting,gold,layouts):
    path=Path(path)
    if gold:
        need(not path.exists(),'Exact-production gold must not fabricate prompt observer')
        return dict(events=[],coverage=False,path=str(path),scope='Gold has no new prompt observer')
    events=rows(path);need(len(events)==6,'All two prompt growth transactions need three known lifetime points')
    order=['RES','PRE','SELECTED','MASK','ENGRAM0','ENGRAM1']
    def identity(v):
        need(type(v)is dict and set(v)=={'wrapper_id','resource_id','offset','bytes','is_view'} and all(type(v[k])is int and v[k]>=0 for k in v) and v['is_view']in (0,1),'Complete prompt resource/view identity')
    for e in events:
        need(e['kind']=='prompt_growth' and e['sequence']==sequence and e['stage']in (0,1) and e['in_generation']is True and e['implementation_version']==4 and e['compact']==setting and e['clear_mode']==setting and e['quiescent']==1 and e['observer_CPU_only']is True and e['GPU_work_submitted_by_callback']is False and e['extra_gpu_waits']==0 and e['owner_order']==order,'Actual normal prompt observer scope')
        need(type(e['monotonic_s'])in (float,int) and math.isfinite(e['monotonic_s']),'Finite prompt timestamp')
        for name in ('old_owners','new_owners','views_before','views_current'):
            need(len(e[name])==6,'All six prompt identities');[identity(v)for v in e[name]]
        for name in ('old_allocated_bytes','new_allocated_bytes'):
            need(len(e[name])==6 and all(type(v)is int and v>=0 for v in e[name]),'All six actual Metal allocated sizes')
        need(all(n>=v['bytes'] and (v['resource_id']or n==0) for n,v in zip(e['old_allocated_bytes'],e['old_owners'])) and all(n>=v['bytes']for n,v in zip(e['new_allocated_bytes'],e['new_owners'])),'Actual allocated sizes cover uniquely-owned prompt lengths')
        need(sum(v['bytes']for v in e['old_owners'])==e['old_total_bytes'] and sum(v['bytes']for v in e['new_owners'])==e['new_total_bytes'],'Prompt requested-size accounting')
        need(all(v['wrapper_id']>0 and v['resource_id']>0 and v['bytes']>0 and v['is_view']==0 for v in e['new_owners']) and len({v['resource_id']for v in e['new_owners']})==6,'Complete new uniquely-owned prompt group')
        need(all(v['wrapper_id']>0 and v['is_view']==1 for v in e['views_before']),'Six dynamic wrappers stay allocated')
        a=e['actual'];need(a['task_valid']is True and a['device_valid']is True and all(type(a[k])is int and a[k]>=0 for k in METRICS),'Valid process/device prompt checkpoint')
        need(type(a['task_info_count_words'])is int and a['task_info_count_words']>0 and e['monotonic_s']<=a['snapshot_begin_s']<=a['snapshot_end_s'],'Actual API field/time availability')
    need(all(x['monotonic_s']<=y['monotonic_s'] and x['record_index']<y['record_index'] for x,y in zip(events,events[1:])),'Prompt callback chronology')
    for stage in range(3):
        selected=[e for e in events if e['stage']==stage]
        need([e['event']for e in selected]==([]if stage==2 else[1,2,3]),'Prompt READY/AFTER_CLEAR/AFTER_FREE coverage')
        if stage==2:continue
        ready,cleared,freed=selected
        need(ready['old_cap']==(0 if stage==0 else 7168) and ready['new_cap']==(7168 if stage==0 else 15360) and ready['old_mask_bytes']==(0 if stage==0 else 112) and ready['new_mask_bytes']==(112 if stage==0 else 240),'Actual prompt capacity/mask growth')
        immutable=('compact','clear_mode','old_cap','new_cap','cw_id','old_mask_bytes','new_mask_bytes','old_total_bytes','new_total_bytes','old_owners','new_owners','views_before','old_allocated_bytes','new_allocated_bytes')
        need(all(all(e[k]==ready[k]for k in immutable)for e in selected),'One complete prompt transaction identity')
        need([e['prompt_sequence']for e in selected]==list(range(ready['prompt_sequence'],ready['prompt_sequence']+3)),'Three consecutive prompt lifetime points')
        need(ready['cw_id']==layouts[stage][1]['value']['cw_id'],'Prompt event actual retained/new CW identity')
        need(ready['views_current']==ready['views_before'] and ready['clear_applied']==0 and cleared['clear_applied']==freed['clear_applied']==setting,'Clear applied only at the declared point')
        need(all(x['wrapper_id']==y['wrapper_id'] for x,y in zip(ready['views_before'],cleared['views_current'])) and freed['views_current']==cleared['views_current'],'Clear preserves six wrappers and postclear identity')
        if setting:
            need(all(not v['resource_id']and not v['bytes']and not v['offset'] for v in cleared['views_current']),'Candidate clears all six old references before release')
            if stage==1:need(all(v['resource_id']==o['resource_id']>0 for v,o in zip(ready['views_before'],ready['old_owners'])),'All old dynamic parents actually retained')
        else:need(cleared['views_current']==ready['views_before'],'Control must not clear dynamic views')
        if stage==0:need(all(not v['resource_id']and not v['bytes']for v in ready['old_owners']),'Fresh initial prompt group')
        if stage==0 or not setting:need(all(not v['resource_id']and not v['bytes']for v in ready['views_before']),'Initial or replacement legacy CW dynamic prompt views start empty')
        if stage==1:need(all(v['resource_id']>0 and v['bytes']>0 for v in ready['old_owners']) and not({v['resource_id']for v in ready['old_owners']}&{v['resource_id']for v in ready['new_owners']}),'Distinct complete old/new prompt groups')
    return dict(events=events,coverage=True,path=str(path),sha256=sha(path),scope='Known prompt all-new-live/after-clear/after-old-release CPU checkpoints, no exact physical ownership inference')

def read_arm(directory,expected_phase,expected_setting,requests,expected_env,*,gold=False,expected_model=None,request_original_paths=None):
    directory=Path(directory);data=rows(directory/'summary.jsonl');need(expected_phase in ('probe','correctness','performance'),'Known phase')
    repeats={'probe':0,'correctness':1,'performance':3}[expected_phase];outputs=8 if expected_phase=='probe' else 32
    meta=[v for v in data if v['kind']=='metadata'];seqs=[v for v in data if v['kind']=='sequence_complete'];complete=[v for v in data if v['kind']=='complete']
    need(len(meta)==len(complete)==1 and data[0]==meta[0] and data[-1]==complete[0] and len(data)==len(seqs)+2,'Exact complete sequence log')
    meta=meta[0];need(meta['schema_version']==1 and meta['probe']=='moe_workspace_promptclear_fullsequence' and meta['phase']==expected_phase and
      meta['workspace_mode']==expected_setting and meta['repeats']==repeats and meta['max_new_tokens']==outputs and meta['warmup_sequences']==int(repeats>0),'Actual native phase/count')
    need(meta['fresh_context_per_sequence'] is True and meta['same_context_within_sequence'] is True and meta['manual_cache_or_rng_reset'] is False and
      meta['extra_gpu_waits_inserted']==0 and meta['testing_api_used'] is False and meta['growth_lifetime_observer']==(not gold) and
      meta['observer_cpu_cost_in_generation_wall'] is True and meta['sampling_interval_ms']==10 and meta['performance_measured']==(expected_phase=='performance') and
      meta['total_wall_basis']=='sum_of_three_generation_calls_excluding_between_stage_exports' and (meta['max_seq_len'],meta['temperature'],meta['seed'])==(327680,0,42),'Normal production sequence scope')
    normal_scope(meta,expected_setting,expected_env,gold)
    need(meta['environment']==expected_env and (not expected_model or Path(meta['model']).resolve()==Path(expected_model).resolve()),'Complete actual environment/model')
    need([v['sequence'] for v in seqs]==list(range(repeats+1)) and complete[0]['sequences']==repeats+1 and
      complete[0]['generations']==3*(repeats+1) and complete[0]['max_new_tokens']==outputs and complete[0]['full_original_raw_exported'] is True and
      complete[0]['adoption_gate_pass'] is False and complete[0]['independent_review_required'] is True,'Complete native sequence coverage')
    hashes=ByteHashes();reviewed=[]
    try:
        need([v['stage'] for v in meta['requests']]==list(LAYOUT),'Request stage order')
        for v,stage in zip(meta['requests'],LAYOUT):
            p=Path(requests[stage]);actual=hashes.file(p);original=Path(request_original_paths[stage]) if request_original_paths else p;need(Path(v['path']).resolve()==original.resolve() and actual['bytes']==v['bytes'] and actual['fnv1a64_bytes']==v['fnv1a64_bytes'],'Actual original request binding')
        for seq in seqs:
            sequence=seq['sequence'];warmup=bool(sequence==0 and repeats);need(seq['warmup']==warmup and seq['fresh_context'] is True and seq['same_context_within_sequence'] is True and seq['generations']==3 and seq['generation_outputs']==outputs,'Sequence sample identity')
            sd=directory/f'seq-{sequence:03d}';raw=rows(sd/'summary.jsonl');samples=[v for v in raw if v['kind']=='sample'];need(len(samples)==3 and [v['stage'] for v in samples]==list(LAYOUT),'All three setup/primary stages')
            need(all(v['kind'] in ('sample','memory_phase')for v in raw),'Unexpected sequence record')
            layouts=[];stages=[];previous=None
            for stage,s in enumerate(samples):
                need(s['sequence']==sequence and s['warmup']==warmup and s['run']==stage and s['role']==('setup' if stage==0 else 'primary') and
                  s['same_context'] is True and s['generation_ok'] is True and s['raw_state_exported'] is True and s['stop_reason']=='length' and
                  s['prompt_tokens']==(6917,15187,6917)[stage] and s['output_tokens']==outputs and 0<=s['cached_tokens']<s['prompt_tokens'] and
                  s['prefill_tokens']==s['prompt_tokens']-1-s['cached_tokens'],'Complete actual generation/cache extent')
                need(s['decode_intervals']==outputs-1 and len(s['callback_intervals_ms'])==outputs-1 and
                  close(sum(s['callback_intervals_ms']),s['decode_s']*1000,1e-5) and close(s['last_callback_s']-s['first_callback_s'],s['decode_s'],3e-9) and
                  all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>0 for k in ('wall_s','prefill_s','decode_s')) and s['last_callback_s']<=s['wall_s']+1e-8,'Actual callback/wall timing')
                B=8192 if stage==0 else 16384;w=s['workspace']
                if gold:
                    need(expected_setting==0 and w.get('provider_counters_available') is False and w['mode']==0 and w['capacity']==2048 and w['moe_batch']==B and all(w[k]==B*6*2304*4 for k in ('legacy_gate_bytes','legacy_up_bytes','legacy_mid_bytes')),'Independent production golden legacy workspace')
                else:workspace(w,expected_setting,B)
                before=read_layout(sd/f'layout-{LAYOUT[stage]}-before.json',hashes);after=read_layout(sd/f'layout-{LAYOUT[stage]}-after.json',hashes)
                if gold:need(not before['value']['present'] and not after['value']['present'],'Gold must not fabricate v3 layout')
                else:
                    layout_transition(before,after,stage,expected_setting);counter_transition(previous,w,before,after,expected_setting)
                    if stage:need(before['value']==layouts[-1][1]['value'],'Same-context prior layout not retained')
                layouts.append((before,after));previous=w
                checked=state(sd/'state'/LAYOUT[stage],LAYOUT[stage],s,hashes,outputs)
                stages.append(dict(stage=LAYOUT[stage],sample=s,state=checked,before_layout=before,after_layout=after))
            need(close(seq['total_wall_s'],sum(s['wall_s']for s in samples),3e-8) and seq['elapsed_including_exports_s_diagnostic_only']>=seq['total_wall_s'] and
              seq['actual_B8192_to16384_growth'] is True and seq['actual_B16384_retained'] is True and seq['native_localgrow_identity_gate'] is True,'Complete total-wall/actual branch witness')
            mem=memory(raw,sd/'memory/samples.jsonl');obs=observer(sd/'growth-observer.jsonl',sequence,expected_setting,gold,layouts)
            prompt=prompt_observer(sd/'prompt-observer.jsonl',sequence,expected_setting,gold,layouts)
            need(seq['growth_observer_events']==len(obs['events']),'Complete observer record count')
            for i,v in enumerate(stages):
                peaks=next(x['peaks'] for x in mem['phases']if x['run']==i and x['phase']==2).copy()
                for e in obs['events']+prompt['events']:
                    if e['stage']==i:
                        peaks={k:max(peaks[k],e['actual'][k])for k in METRICS}
                v['actual_peak_with_lifetime_events']=peaks
                # Binding tuples have already been compared in this process. Keep their
                # original path/SHA and O(1) layout fields, not millions of JSON objects.
                for key in ('before_layout','after_layout'):
                    q=v[key];q['value']={k:x for k,x in q['value'].items() if k!='bindings'}
                    q['actual_bindings_fully_validated']=True
            reviewed.append(dict(sequence=sequence,warmup=warmup,total_wall_s=seq['total_wall_s'],stages=stages,memory=mem,observer=obs,prompt_observer=prompt))
    finally:hashes.close()
    return dict(directory=str(directory),phase=expected_phase,setting=expected_setting,gold=gold,metadata=meta,summary_sha256=sha(directory/'summary.jsonl'),sequences=reviewed,complete=True)


def compare_arms(reference,candidate):
    need(reference['metadata']['max_new_tokens']==candidate['metadata']['max_new_tokens'],'Matching output-count oracle')
    oracle=reference['sequences'][1 if len(reference['sequences'])>1 else 0];comparisons=[]
    for sequence in candidate['sequences']:
        for a,b in zip(oracle['stages'],sequence['stages']):
            need(a['stage']==b['stage'] and all(a['sample'][k]==b['sample'][k] for k in ('prompt_tokens','cached_tokens','prefill_tokens','output_tokens')),'Same-sequence cache/output contract')
            left,right=a['state'],b['state'];need(left['files'].keys()==right['files'].keys(),'Raw active state coverage')
            total=0
            for name in left['files']:
                x,y=Path(left['path'])/name,Path(right['path'])/name;delta=file_difference(x,y,name=='final-logits.f32');need(delta['equal'],'Full raw bytes differ: '+name);total+=x.stat().st_size
            comparisons.append(dict(sequence=sequence['sequence'],stage=b['stage'],full_original_bytes_equal=True,bytes_compared=total))
    return dict(full_original_bytes_equal=True,comparisons=comparisons)


def growth_gate(A1,B,A2):
    need(all(a['phase']=='probe' and len(a['sequences'])==1 for a in (A1,B,A2)) and [a['setting']for a in (A1,B,A2)]==[0,1,0],'One normal probe ABA sequence')
    comparisons=[compare_arms(A1,B),compare_arms(A1,A2)];values=[a['sequences'][0]['stages'][1]['actual_peak_with_lifetime_events']for a in (A1,B,A2)]
    peaks={k:dict(A1=values[0][k],B=values[1][k],A2=values[2][k],B_minus_A1=values[1][k]-values[0][k],B_minus_A2=values[1][k]-values[2][k],conservative_excess=values[1][k]-min(values[0][k],values[2][k]))for k in METRICS}
    coverage=all(a['sequences'][0]['observer']['coverage']and a['sequences'][0]['prompt_observer']['coverage']for a in (A1,B,A2));passed=coverage and all(peaks[k]['conservative_excess']<=134217728 for k in ('phys_footprint_bytes','device_allocated_bytes'))
    return dict(growth_gate_pass=passed,known_lifetime_coverage=coverage,actual_growth_peaks=peaks,raw_comparisons=comparisons,performance_measured=False,decision='pass_normal_local_growth'if passed else'reject_growth_peak_or_coverage')


def performance_gate(A1,B,A2):
    need(all(a['phase']=='performance' and len(a['sequences'])==4 for a in (A1,B,A2)) and [a['setting']for a in (A1,B,A2)]==[0,1,0],'Exactly one formal sequence ABA cohort')
    reports=[];passed=True
    for stage,name in enumerate(LAYOUT):
        med=lambda key:[statistics.median(q['stages'][stage]['sample'][key]for q in a['sequences'][1:])for a in (A1,B,A2)]
        times={k:med(k)for k in ('prefill_s','decode_s','wall_s')};drift={k:(v[2]/v[0]-1)*100 for k,v in times.items()};saved={k:(1-v[1]/min(v[0],v[2]))*100 for k,v in times.items()}
        peaks={k:[statistics.median(q['stages'][stage]['actual_peak_with_lifetime_events'][k]for q in a['sequences'][1:])for a in (A1,B,A2)]for k in METRICS}
        phys=peaks['phys_footprint_bytes'];device=peaks['device_allocated_bytes'];physsave=min(phys[0],phys[2])-phys[1];devsave=min(device[0],device[2])-device[1]
        timepass=all(abs(drift[k])<=2 and saved[k]>=-1 for k in ('prefill_s','decode_s'))
        mempass=abs(phys[2]-phys[0])<=268435456 and physsave>=-134217728 and devsave>0
        gain=physsave>=268435456 or(saved['prefill_s']>2 and physsave>=0);stagepass=timepass and mempass and (stage==0 or gain);passed=passed and stagepass
        reports.append(dict(stage=name,role='setup'if stage==0 else'primary',times=times,control_drift_percent=drift,conservative_time_saved_percent=saved,actual_peaks=peaks,conservative_physical_reduction_bytes=physsave,conservative_device_reduction_bytes=devsave,time_gate_pass=timepass,memory_gate_pass=mempass,material_gain_required=stage!=0,material_gain_pass=gain,stage_gate_pass=stagepass))
    total=[statistics.median(q['total_wall_s']for q in a['sequences'][1:])for a in (A1,B,A2)];tdrift=(total[2]/total[0]-1)*100;tsave=(1-total[1]/min(total[0],total[2]))*100;totalpass=abs(tdrift)<=2 and tsave>=-1
    coverage=all(q['observer']['coverage']and q['prompt_observer']['coverage']for a in (A1,B,A2)for q in a['sequences']);peak_excess=[max(q['stages'][1]['actual_peak_with_lifetime_events'][k]for q in B['sequences'][1:])-min(max(q['stages'][1]['actual_peak_with_lifetime_events'][k]for q in a['sequences'][1:])for a in (A1,A2))for k in ('phys_footprint_bytes','device_allocated_bytes')]
    shadowpass=coverage and all(x<=134217728 for x in peak_excess)
    return dict(performance_gate_pass=passed and totalpass and shadowpass,stages=reports,total_wall_seconds=dict(zip(('A1','B','A2'),total)),total_wall_control_drift_percent=tdrift,total_wall_conservative_saved_percent=tsave,total_wall_gate_pass=totalpass,growth_peak_coverage=coverage,growth_peak_excess_bytes=dict(zip(('phys_footprint_bytes','device_allocated_bytes'),peak_excess)),growth_peak_gate_pass=shadowpass,production_adoption=False)



GOLD_SOURCE_HASHES={'Makefile': '2292a6f4976d4e6adab1034cfda0710a0cccbbb632bf98b5dce76d1e2b6dbcc7',
 'include/myds.h': 'd73af31a68e379d1651239791d80538fd7191b4cf9072bfd98a84d834a64f6c5',
 'include/myds_align.h': '27db712ed10840fa27a2703dcd14c390f31cd0f666abb51db04e6f21d2a62476',
 'include/myds_configfile.h': 'b467e58f982c6dbeee12c14c55912716bc69d18e17e7596efeef4e7f291d27a3',
 'include/myds_cpu.h': '31d872795cc2ef1713f1a51e43d16c02cd3889b03ee8cc4ab440dc41e87fe7d0',
 'include/myds_dspark.h': '4e39abe8cbb8744a0f38a0975e6ce993c3f5f3f6945dfcc76029fd14a44d460f',
 'include/myds_engram.h': '57ee56c41399c41e255f9c232b1369579d9efd6bb771db15f6272a4d8717fb21',
 'include/myds_gguf.h': '8c8cd1d4de303725d8483c7085a433b84df0b48818a6ebf175dba8d27555cd7a',
 'include/myds_internal.h': 'b810de8a2c3e8a9ed81711728994e71bdca9695da33f2eb4552de67192607ac1',
 'include/myds_kvdsk.h': 'fc425edf81333e2308fa2a1cdb259aef7fe3ddbd609174ee4c3da04cae3a461c',
 'include/myds_log.h': '96da7e04b593b5a17ac4dfa6ba0fc694fb11928a167e3f7690e74228a71faa9e',
 'include/myds_lowkv_policy.h': 'def5dc1dec946fc06af0fa04338be57f42baad28e2ee65567d73f3682e282a6c',
 'include/myds_metal.h': 'ecb5191c4f5de1038d242e97e10732a3411102fc779c05e7ca2886ccc20e9ab1',
 'include/myds_moe_cpu.h': 'b4087827a191de2e4ae03d3a9682e58e1f5d92b941c2cbbebaf0c1f1a9dfddc4',
 'include/myds_pld.h': 'd5900389894ee34ab3189bd08a2eb38f284c0b6dae4060600fed71c73c4353b5',
 'include/myds_server.h': '2e5fa57fead8a0c92258683da2ae983e935425f1aee491492d4e29ff183cec40',
 'include/myds_server_internal.h': '133ab3b9c7cf13a2b2258e688ed3861eebaa73db25833b2d33780b8443ff73b8',
 'include/myds_snapshot_slots.h': '14616746226747af04c5cb65940d16af880f25888e6af6093fe6dfa5620a8f31',
 'include/myds_tokenizer.h': '292cc9c3ce212cdeb08d9beb4f92a84660b248e8c677c04790b93229e26b451c',
 'include/myds_v41.h': '6389d3aeaf3c12f35721df5653f03b526cac2796b85346e0d59efedf2a392bec',
 'include/myds_v41_decode.h': '616fffa71022b9342db70d3df54ffab93037c35e58a3441c42e577873e217348',
 'include/myds_v41_gpu.h': '17ab8f4c00662b0661af5ea0e0fb11b7abc4119df13dab77d098e71c046ff373',
 'include/myds_v41_moe.h': 'ac7fdc9cdeedd057452c277833fd8917d8d6ca00f68e7f93048cea71e2728aed',
 'include/myds_visfwd.h': 'e188c8e488591be257ec1170331b78f01b6db4132ded5a7938ef1c564d998d13',
 'include/myds_vispre.h': 'addb28dd5286289001717abe81b7995fa6599f5a69e65dd558019c12ce7711be',
 'include/myds_visw.h': 'd8425832a8d1425c7e0cfa13fc9c2ecca2e94e8fbc46e881ec03791617b0affa',
 'include/myds_vjob.h': '7710ea311702951206f183d762d12b89b867cf72b21ffcd98420565818b449b3',
 'metal/argsort.metal': '7bcdcb48366f71cb9adf6fdb27429b614eeab24821947a6710e2a65b3f88cc17',
 'metal/bin.metal': '66f7c42264b1cfb7b1e42380c94093f963dc1f0792f43e79b5588892f882d5b6',
 'metal/concat.metal': '04fb0fc2a3b51015444e330290b979a56e4cac82aee523afa2b42e361ab23da7',
 'metal/cpy.metal': 'df8cabe88c33076c16a54cfb88a00133b68ea5cbfc1742ac79608c84d4456cd4',
 'metal/dense.metal': 'efdc3e24cf13b6d8651f4dd4a728a453f2a4d1b2295cef41135516ea628bed55',
 'metal/dsv41.metal': '7a15599dc6d329715c610b581b2babcce46580187707b8d87565e795b41ed1a2',
 'metal/dsv4_hc.metal': '7f5e44ce6108285740cca97a4c81e8c271d0631ed784f459baa63b3d01260bfc',
 'metal/dsv4_kv.metal': '87334bd001a306dcdf8e344f7f47938d67d6887889a3da7282cba0d25b36d172',
 'metal/dsv4_misc.metal': '14dfce00954e374f1da677a174ae92c0c7bacbd46a5266a7665d736833252531',
 'metal/dsv4_rope.metal': 'a6e86dcf24d35a0014c92b7542c7c2ba8b3815c853f5d636da94cb1df1c05ccd',
 'metal/flash_attn.metal': '503fc1ec7fb94130f4e44c1a1147bf182d11c9993cc0688e8ed2f4993df7bdc6',
 'metal/generate_mxfp4_half_lut.py': 'a20b9b79d993b16b3bcb4a9658c86618ce4cd33739838fe6aa47bf947f42194e',
 'metal/get_rows.metal': 'bd3b352a6ffb138ae5e391db97b211f8386d417365eac2caa3730f55a075aa3a',
 'metal/glu.metal': '6d226a91566d625b3a4b141ecadcb4d796bdc02de19211d891e215c530c812ca',
 'metal/moe.metal': '78cb927e7884254832f3c37e97b1c4952c1a1863555616693f3b35fb787f5776',
 'metal/myds_chunk.metal': '2cdfbd0c96b47cfa81c26ca6b2194874f42bd766b8c02efacd86f15d2808bda1',
 'metal/myds_iq2_v41.metal': 'cbd703565fb9715c7a611e8e5ce15a81c9289cc7fbeedd3ca285ae8f5ba26aa5',
 'metal/myds_q4_v41.metal': '6487878f399cd24b3e3c357cc161a10944a558465f2d0eb680251418a4ab22d0',
 'metal/myds_v41_moe.metal': 'd731dfbe42b6ce94aabd70a5bc4472b4168b2913d4ce82c0d84982a2fd75c0e4',
 'metal/norm.metal': 'fd89688ae3ac8a473f731da0051121239bff2c6767674d091832f1e5cfd698c8',
 'metal/probe_attention_register.metal': '0bfd2bd9b52cf5bb1f54e12a7bca5721be0a92ca0dabd70b3023b8ef2248b13b',
 'metal/probe_dequant_cost.metal': '6ece56ba63f6f1d409a7be14f5897639bf203fd9c89672fb978126222596fbab',
 'metal/probe_q2_down.metal': '1883cbb1710ebdf270220f0f8e100452085f41c2d261f9758872a7b849bde791',
 'metal/probe_q4_sumy.metal': '4934774b49944258c9a729d41c1baf3a0c2abfe7b7d531c9994928d65cc775fa',
 'metal/probe_v41_route.metal': '92c715cd2436ccfb39125d8bac91a1e6a6983889e66a8aa6d4b84ec6aa1f6436',
 'metal/pureread.metal': '3c740ef1ad065e617c42e82c8b5f7f3c02a1011d5977b9d88aa62ee445a8117a',
 'metal/repeat.metal': '2c6142b47a2007d246f5eb6e7395047627bb6dfa02c649311d16620da35007cd',
 'metal/set_rows.metal': '25f597bf5f5fb5a3791efc68e3182539727609f09c45750ef9376ed430d7a029',
 'metal/softmax.metal': '6eaa6ce78c4b221b3c7fe45647d6eb42f165201992a243d77d04cf2668fb7664',
 'metal/sum_rows.metal': '7e1f3ad3f945b1f6f59aa2d8ee5bf464052d989a632c33ea95ce50d3c4fbf0fc',
 'metal/topk_threshold.metal': '8fb93312ccf29f934668954a202244df60e5e7cfd62680c7c74137161a369fc0',
 'metal/unary.metal': '4607dbea21501230a193f8a333b1bd543d566f405568a094d6faf2c17cb0c5cf',
 'metal/visenc.metal': 'dced00348fbf19e3d1d6c9e4bb39cd457114819bdd576468d3a1dbf22d807165',
 'src/main.c': 'e376ed9ef8b89d449c743d6af5d2d60221ace33fa41bf16fbd3ad77cbd50d942',
 'src/myds_configfile.c': 'a2dfc690288cb56e37afd4572d8c8739cba3f995faa59a78a1482eb131e67f56',
 'src/myds_cpu.c': 'd88b99249fc1a4b40c3f5630b4e7b780d4ff13eb671996401cded90525ece7fe',
 'src/myds_decode_candidates.inc': 'a11bc77045a1e01447e2c2fa67f0c5e743929b7f7db899f380e8322d7db6a286',
 'src/myds_dspark.c': '6acea67bea57390cf670695acfb00e4c39f4c2651bdfcaf0c32a3358187f6547',
 'src/myds_engine.c': '47caa3321c221d3bf7b786fa6d7fee29bafac25f402f4217f0eeefb17f6f31a9',
 'src/myds_engram.c': '54ef3cad362c9b0c49c03be6590bc6b2ca90aee5707fc3391e5acf75881ab885',
 'src/myds_engram_cpu.c': '11d143161f9f0a989b88abe666e733f67cf9b2fbe632e6733eebdab5e090cc20',
 'src/myds_gguf.c': '63c5ca489d8f9d1b87bd36c36c22f09f47498f904925aaa1b9ca982ad0a2ae6c',
 'src/myds_iq2_tables.h': 'fd92b447b6c38445129862f080162e6dcdb6caa1d4f76646f482d515a028cd86',
 'src/myds_kvdsk.c': '2221cd4ae07634fcc011a78c6b0185defe62ac282b233438461012723207ed56',
 'src/myds_log.c': 'f6eff9e828fc882e3d5f0ed7ccd033c0bcd91fefb345fe8634ef8512179082b5',
 'src/myds_metal.m': 'a36c6eb122f415e924037b402a3c78934dbc86f05538469adc604412b6a75a31',
 'src/myds_moe_cpu.c': '81902164f9a60d79aac8ce7f609ed4ad35263b6b5b0f72d063931b2c906cc937',
 'src/myds_server.c': '5aa609e8a0845c313444d062e93c38d04520c5f21bb7fe0e2ea552481b365a5a',
 'src/myds_tokenizer.c': '808c432023f5fa2534a4464fbbc7816bb54d86ac44921ba9b7ec2d45e5fcf144',
 'src/myds_tokenizer_unicode.h': 'bfb6f1a5149a813f18212f6216ce10468b962f7e7bd3e88329ef823de1854808',
 'src/myds_v41.c': '1a4614a71f3b425fc2181f7892bdd6a2118ea79e97183b58ed0e05488758601a',
 'src/myds_v41_decode.c': '9f9d89f84e83e54cda3d718aa748e3311af06fd8a7f1002918cae0de959829ae',
 'src/myds_v41_gpu.c': 'a4f34a543e8836212aacebe0a05bbdf4f5fbc40719e4d8efea24067a557e6f10',
 'src/myds_v41_moe.c': 'bfed3b37525bf0d7eeb2146a172a3072ef3ab44e1fd42631fa3f8c616ef5b9e2',
 'src/myds_v41_shared3_gu_shadow.inc': 'b33bb37bb19f18ee870e1d1eef6a6bbd4d649c7c02a4d8894e20277f25f6d713',
 'src/myds_v41_shared3_routed_shadow.inc': '9fb29d7128bb62f4b8d3945b2c54851c6b60de665036f98a52d23d274d76726a',
 'src/myds_visfwd.c': 'b9014c5f3248d524800c3cd6e67907e5dc858a27806a922aa0c6d71879800a95',
 'src/myds_vispre.c': 'bbd6e6caf1c5728c89420c080334917018fcbf2f7a0ca54d6ed404cee7426a0e',
 'src/myds_visw.c': '3f3b138ff904376c804c71e04f4ca2317b706394950442202bdd8c0604cf8089',
 'src/myds_vjob.c': '5acb00588396128b34e922d595b286d007e66bb363837d6bb886b62a13c50807',
 'third_party/iris/LICENSE': '663e51edcd9a3a8f858018ce6924ad2895a24c269c26c39ad75325bcd0f7e5bc',
 'third_party/iris/jpeg.h': '9a1d13b3fc3dfbe3af76ab35b650bfe9a54b418b254bf4bd9c7ed958b0c98403',
 'third_party/iris/png.h': 'b87dda8580af164c443e76ca073d77f1340e1a7584b73a8449a0f5bf14cd5018',
 'tools/build_metallib.py': 'd18dd1fd37cfac24fbd87db9d6ae08df3a1738e313aa627a327f0cb3e72365ce'}

PRODUCTION_BINARY='29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b'
PRODUCTION_LIBRARY='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81'
PLAN_SHA='a83127b1427b09b703883150f3ec6e8e90e7ecdead3f48b798c7755368bd9d9c'
IDENTITY_KEYS=('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary','metallib','binary_sha256','metallib_sha256','config','config_sha256','effective_myds_env')


def executed_archives(root,source_archive,artifacts_archive,bindings_path):
    """Verify original-path mappings and every byte. Retain only bounded JSON/input bytes."""
    inputs=load(root/'stage-inputs.json');manifest=load(bindings_path)
    need(manifest['schema_version']==1 and manifest['status']=='complete' and
         manifest['stage_inputs_sha256']==sha(root/'stage-inputs.json'),'Exact executed original input-map')
    entries=manifest['files'];need(len(entries)==len(inputs) and {v['original_path']:v['sha256']for v in entries}==inputs,'Complete archived original input map')
    index={};small={};reports=[]
    for name,path in (('source.tar.gz',source_archive),('artifacts.tar.gz',artifacts_archive)):
        selected={(v['tar_path']):v for v in entries if v['archive']==name};need(len(selected)==sum(v['archive']==name for v in entries),'Unique original archive mappings')
        desc=manifest['archives'][name];need(sha(path)==desc['sha256'] and path.stat().st_size==desc['bytes'],'Complete archive physical bytes/SHA')
        names=set()
        with tarfile.open(path,'r:*')as tar:
            for member in tar:
                need(member.isfile() and not member.islnk() and not member.issym() and not Path(member.name).is_absolute() and '..'not in Path(member.name).parts and member.name not in names,'Bounded unique regular archive member')
                names.add(member.name);v=selected.get(member.name);need(v is not None,'Unexpected archived byte source');stream=tar.extractfile(member);need(stream is not None,'Unreadable archived bytes');h=hashlib.sha256();count=0;kept=bytearray()
                capture=name=='source.tar.gz' and member.size<=4*1024*1024 and Path(v['original_path']).suffix=='.json'
                for block in iter(lambda:stream.read(1048576),b''):
                    h.update(block);count+=len(block)
                    if capture:kept.extend(block)
                need(count==member.size==v['bytes'] and h.hexdigest()==v['sha256'],'Executed archived logical bytes/SHA')
                index[v['original_path']]=dict(sha256=v['sha256'],bytes=count,archive=name,tar_path=member.name)
                if capture:small[v['original_path']]=bytes(kept)
        need(names==selected.keys() and len(names)==desc['members'],'Complete archive member set')
        reports.append(dict(path=str(path),sha256=desc['sha256'],bytes=desc['bytes'],members_verified=len(names)))
    need(index.keys()==inputs.keys(),'All frozen executed inputs archived')
    return dict(reports=reports,index=index,small=small,manifest_sha256=sha(bindings_path),input_map_sha256=sha(root/'stage-inputs.json'))


def frozen_protocol(plan):
    f=plan['fullmodel'];p=f['performance'];a=f['adoption'];n=plan['normal_success_probe'];c=f['correctness']
    need(PLAN_SHA is not None and plan['implementation_version']==4 and plan['flag']==FLAG and plan['prompt_view_clear_flag']==CLEAR_FLAG and plan['default_off'] is True and
         [v['stage']for v in f['sequence']]==list(LAYOUT) and [v['request']for v in f['sequence']]==['pf7k','pf16k','pf7k'] and
         f['max_seq_len']==327680 and f['max_new_tokens']==32 and f['temperature']==0 and f['seed']==42 and
         f['fresh_context_per_sequence'] is True and f['same_context_three_stages'] is True,'Actual frozen same-context v4 sequence')
    need(plan['schema_version']==4 and plan['prior_plan_sha256']=='fde7b74283dc680657f34ee00d74fdb704f4176ac1597d9cb20ac9544a66f9ea' and f['prompt_view_clear_flags']=={'A1':0,'B':1,'A2':0},'Frozen source/prompt-clear scope')
    need(c['order']==['gold','off','on'] and c['warmup']==c['formal_samples']==1 and c['total_generations']==18 and
         p['order']==['A1','B','A2'] and p['flags']=={'A1':0,'B':1,'A2':0} and p['warmup']==1 and p['formal_samples']==3 and p['total_generations']==36 and p['repeat_limit']==1 and
         p['maximum_absolute_prefill_and_decode_control_drift_percent']==2 and p['maximum_conservative_time_regression_percent']==1 and p['memory_sampling_interval_ms']==10 and p['physical_memory_control_drift_limit_bytes']==268435456,'Unchanged frozen counts/thresholds')
    need(a['minimum_conservative_peak_footprint_reduction_bytes']==268435456 and a['minimum_conservative_prefill_time_saved_percent_strict']==2 and a['max_footprint_regression_bytes']==134217728 and
         a['required_primary_workloads']==['grow','retain'] and a['setup_nonregression_required'] is True and a['setup_material_gain_threshold'] is False and
         a['total_sequence_wall_control_drift_limit_percent']==2 and a['total_sequence_wall_regression_limit_percent']==1 and
         a['known_growth_lifetime_checkpoint_required'] is True and a['localgrow_non_moe_identity_reuse_required'] is True,'Unchanged primary/setup/total/lifetime acceptance')
    need(n['order']==['A1','B','A2'] and n['sequences_per_arm']==1 and n['max_new_tokens']==8 and n['performance_measured'] is False and n['required_before_formal'] is True and n['all_state_bytes_equal'] is True and n['localgrow_id_reuse_and_checkpoint_required'] is True,'Separate normal probe prerequisite')
    d=plan['causal_diagnostic'];r=plan['recovery'];o=plan['memory_observation']
    need(n['prompt_view_lifetime_coverage_required'] is True and o['sample_interval_ms']==10 and o['growth_gate']['maximum_candidate_observed_device_or_phys_peak_regression_bytes']==134217728 and bool(o['normal_prompt_observer']),'Required normal prompt lifetime coverage')
    need(d['workspace_settings']==[1,1,1] and d['clear_settings']==[0,1,0] and d['total_generations']==9 and d['max_new_tokens']==8 and d['performance_measured'] is False and d['extra_gpu_waits_or_commits']==0 and d['deferred_resource_observer_registered_only_here'] is True,'Separate causal diagnostic scope')
    need(r['executions']==28 and len(r['scenarios'])==7 and r['prompt_failure_ordinals']==[1,6] and r['prompt_failure_moe_growth_may_already_commit'] is True,'Actual28run prompt-boundary recovery contract')


def recorded_identity(root):
    before,restored,summary=[load(root/name)for name in ('before.json','restored.json','summary.json')]
    need(all(k in before and before[k]==restored[k]for k in IDENTITY_KEYS) and before['health']==restored['health']=={'status':'ok'},'Recorded exact production argv/env/cwd/nice/artifact/config restore')
    need(before['binary_sha256']==PRODUCTION_BINARY and before['metallib_sha256']==PRODUCTION_LIBRARY and
         summary['restoration']['restored'] is True and summary['restoration']['pid']==restored['pid'] and summary['restoration']['exact_argv_myds_artifacts_config'] is True and summary['production_adoption'] is False,'Recorded production restore identities')
    need(sha(root/'production-artifacts/myds')==PRODUCTION_BINARY and sha(root/'production-artifacts/myds.metallib')==PRODUCTION_LIBRARY and
         sha(root/'original-myds.conf')==before['config_sha256'] and load(root/'effective-env.json')==before['effective_myds_env'],'Original recovery artifacts/config/environment bytes')
    env=before['effective_myds_env'];need(len(env)==51 and env.get('MYDS_METAL_CONCURRENT')==env.get('MYDS_PREFIX_CACHE')==env.get('MYDS_PLD')=='1','Actual complete production concurrency/cache/PLD setting')
    events=rows(root/'events.jsonl');need(events[-1]['kind']=='sequence_decision_complete' and events[-1]['status']==summary['status'] and not any(e['kind']in ('monitor_failure','restoration_failure')for e in events),'Actual final event and competition/restore monitor')
    return before,restored,summary


def long_reference_gate(link,index,label):
    need(type(link)is dict and set(link)in ({'path','sha256'},{'path','sha256','bytes'}) and type(link.get('path'))is str and Path(link['path']).is_absolute()and link['path']in index and index[link['path']]['sha256']==link.get('sha256'),'Full independent prerequisite report archived by exact path/SHA: '+label)
    if 'bytes'in link:need(type(link['bytes'])is int and link['bytes']>=0 and link['bytes']==index[link['path']]['bytes'],'Full independent prerequisite explicit byte extent: '+label)
    return True

def component_identity_gate(component,identity,environment,model,header):
    core=('src/myds_v41.c','src/myds_metal.m','include/myds_internal.h','include/myds_metal.h')
    need(component.get('candidate_core_sha256')=={k:identity['candidate'][k]for k in core} and component.get('metallib_sha256')==identity['candidate']['bin/myds.metallib']==PRODUCTION_LIBRARY,'Actual component core4/library identity')
    need(component.get('production_environment')==environment and component.get('model')==model and component.get('model_header_sha256')==header,'Actual component production environment/model/header identity')
    return True

def causal_identity_gate(causal,identity,environment,model,header):
    ci=causal.get('native_identity');need(type(ci)is dict and set(ci)=={'candidate'} and ci['candidate']==identity['candidate'],'Actual causal candidate native/core/library identity')
    need(causal.get('production_environment')==environment and causal.get('model')==model and causal.get('model_header_sha256')==header,'Actual causal production environment/model/header identity')
    return True

def native_record_gate(record,identity,kind):
    need(record['native_binary_sha256']==identity[kind]['bin/bench_moe_workspace_fullmodel_promptclear'] and record['metallib_sha256']==PRODUCTION_LIBRARY,'Actual selected original/candidate native/library identity')
    return True

def post_window_review(root,source_archive,artifacts_archive,bindings_path,model=None):
    root=Path(root);before,restored,summary=recorded_identity(root);plan=load(root/'protocol.json');frozen_protocol(plan)
    need(summary['plan_sha256']==PLAN_SHA,'Executed v2 recovery + sequence plan identity')
    archive=executed_archives(root,source_archive,artifacts_archive,bindings_path);index,small=archive['index'],archive['small'];approved=[];plans=[]
    for name,data in small.items():
        value=parse(data.decode())
        if type(value)is dict and value.get('experiment')==plan['experiment']:plans.append((name,index[name]['sha256'],value))
    matches=[v for v in plans if v[2]==plan];need(len(matches)==1 and matches[0][1]==PLAN_SHA,'Unique content-matching frozen v4 execution plan')
    required={'component_gate_pass','recovery_gate_pass','causal_lifetime_gate_pass'}
    prereq=summary['prerequisites'];need(type(prereq)is dict and set(prereq)==required,'All three frozen component/recovery/causal prerequisites')
    approved_values={}
    for key,item in prereq.items():
        path=item['path'];need(path in small and index[path]['sha256']==item['sha256'],'Exact archived prerequisite path/SHA: '+key)
        value=parse(small[path].decode());need(value.get('evidence_valid')is True and value.get(key)is True and value.get('implementation_version')==4 and value.get('plan_sha256')==PLAN_SHA,'Current implementation4 frozen prerequisite: '+key)
        approved.append(dict(key=key,path=path,sha256=item['sha256']));approved_values[key]=value
    mode=summary['mode'];need(mode in ('probe','full'),'Known executed mode');arm_records=load(root/'arms.json')
    expected=[('probe',n,m,False)for n,m in (('A1',0),('B',1),('A2',0))]if mode=='probe'else [('correctness','gold',0,True),('correctness','off',0,False),('correctness','on',1,False)]+[('performance',n,m,False)for n,m in (('A1',0),('B',1),('A2',0))]
    need([(v['phase'],v['label'],v['setting'],v['gold'])for v in arm_records]==expected,'Exact single cohort arms/order')
    identity=summary['native_identity'];core=('src/myds_v41.c','src/myds_metal.m','include/myds_internal.h','include/myds_metal.h');native='bin/bench_moe_workspace_fullmodel_promptclear'
    stage_by_kind={}
    for kind in ('candidate','gold'):
        candidates=[str(Path(name).parent.parent)for name,v in index.items()if name.endswith('/'+native)and v['sha256']==identity[kind][native] and all(index.get(str(Path(name).parent.parent/k),{}).get('sha256')==identity[kind][k]for k in (*core,'bin/myds.metallib'))]
        need(len(candidates)==1,'Unique archived native stage identity: '+kind);stage_by_kind[kind]=Path(candidates[0])
    need(stage_by_kind['candidate']!=stage_by_kind['gold'],'Independent candidate and exact-production gold stage')
    gold_stage=stage_by_kind['gold'];need(index[str(gold_stage/'bin/myds')]['sha256']==PRODUCTION_BINARY and index[str(gold_stage/'bin/myds.metallib')]['sha256']==PRODUCTION_LIBRARY and
         all(index.get(str(gold_stage/p),{}).get('sha256')==h for p,h in GOLD_SOURCE_HASHES.items()),'All91 exact-production golden source/build inputs')
    need(all(identity[k]['bin/myds.metallib']==PRODUCTION_LIBRARY for k in ('candidate','gold')),'Unchanged numerical shader library')
    component_identity_gate(approved_values['component_gate_pass'],identity,before['effective_myds_env'],summary['model'],summary['model_header_sha256'])
    need(approved_values['recovery_gate_pass'].get('component_review_sha256')==prereq['component_gate_pass']['sha256'],'Recovery prerequisite binds this exact independently reviewed component top')
    for key,field in (('component_gate_pass','long_report'),('recovery_gate_pass','long_report'),('causal_lifetime_gate_pass','full_report')):
        long_reference_gate(approved_values[key].get(field),index,key)
    causal_identity_gate(approved_values['causal_lifetime_gate_pass'],identity,before['effective_myds_env'],summary['model'],summary['model_header_sha256'])
    self_names=[name for name in index if name==str(stage_by_kind['candidate']/'tools'/Path(__file__).name)]
    need(len(self_names)==1 and index[self_names[0]]['sha256']==sha(Path(__file__)),'Execute the frozen independent reader version')
    # Materialize only independently SHA-verified request JSON (bounded under4MiB).
    with tempfile.TemporaryDirectory(prefix='localgrow-postreview-requests-')as temp:
        requests={};original={}
        for stage,request in zip(LAYOUT,('pf7k','pf16k','pf7k')):
            wanted=plan['fullmodel']['requests'][request]['sha256'];paths=[name for name,data in small.items()if index[name]['sha256']==wanted and Path(name).name==request+'.json'];need(len(paths)==1,'Unique archived original request JSON')
            original[stage]=paths[0];target=Path(temp)/(stage+'.json');target.write_bytes(small[paths[0]]);requests[stage]=target
        arms=[]
        for record in arm_records:
            relative=Path('arms')/(record['phase']+'-'+record['label']);directory=root/relative;need(sha(directory/'summary.jsonl')==record['native_summary_sha256'],'Actual recorded native summary bytes')
            expected_env=dict(before['effective_myds_env'],**{FLAG:str(record['setting']),CLEAR_FLAG:str(record['setting'])})
            need(load(root/'arm-environments'/(record['phase']+'-'+record['label']+'.json'))==expected_env,'Recorded exact arm environment')
            a=read_arm(directory,record['phase'],record['setting'],requests,expected_env,gold=record['gold'],expected_model=summary['model']['path'],request_original_paths=original)
            kind='gold'if record['gold']else'candidate';st=stage_by_kind[kind]
            need(Path(a['metadata']['actual_metallib_path']).resolve()==(st/'bin/myds.metallib').resolve(),'Actual selected library path');native_record_gate(record,identity,kind)
            arms.append(a)
    if mode=='probe':comparisons=[compare_arms(arms[0],v)for v in arms];gate=growth_gate(*arms);key='growth_gate_pass'
    else:
        comparisons=[compare_arms(arms[0],v)for v in arms];gate=performance_gate(*arms[3:]);key='performance_gate_pass'
        prior=[parse(v.decode())for v in small.values()if v[:1]==b'{'and b'"growth_gate_pass"'in v]
        prior=[v for v in prior if type(v)is dict and v.get('mode')=='probe'and v.get('evidence_valid')is True and v.get('growth_gate_pass')is True]
        need(any(v.get('native_identity')==identity and v.get('production_environment')==before['effective_myds_env'] and v.get('model')==summary['model'] and v.get('model_header_sha256')==summary['model_header_sha256'] and v.get('plan_sha256')==PLAN_SHA and {k:x.get('sha256')for k,x in v.get('prerequisites',{}).items()}=={k:x['sha256']for k,x in prereq.items()} for v in prior),'Archived prior actual normal probe exact source/model/env identity')
    need(summary[key]==gate[key] and summary['correctness_pass']is True and summary['normal_sequence_native_complete']is True,'Recomputed decision differs from original controller')
    actual_model=None
    if model is not None:
        model=Path(model);st=model.stat();need(st.st_size==summary['model']['bytes'] and sha(model,1048576)==summary['model_header_sha256'],'Actual model extent/header identity');actual_model=dict(path=str(model),bytes=st.st_size,header_sha256=sha(model,1048576))
    return dict(schema_version=1,evidence_valid=True,correctness_pass=True,mode=mode,implementation_version=4,**{key:gate[key]},decision=('pass_'if gate[key]else'reject_')+('growth_peak_or_coverage'if mode=='probe'else'formal_gain_or_controls'),production_adoption=False,performance_measured=mode=='full',plan_sha256=PLAN_SHA,raw_comparisons=comparisons,gate=gate,recorded_restoration_exact=True,before_pid=before['pid'],restored_pid=restored['pid'],live_production_sampled=False,archive_verification={k:v for k,v in archive.items()if k not in ('small','index')},frozen_logical_inputs_verified=len(index),all91_exact_production_source_inputs=True,native_identity=identity,prerequisites=approved,arms=arms,actual_model=actual_model,limitations=['All actual raw active/live/published snapshot bytes are rehashed and compared. Unpublished/invalid capacity, allocator padding and persistent disk-cache contents are excluded.','Actual resource IDs are compared only before/after within each process, never across processes.','Growth peak uses10ms process/device samples plus known synchronous allocation-lifetime checkpoints; it does not prove exact continuous physical residency.','Cumulative compact-arena candidate is compared with original legacyF32 production baseline; its total savings are not attributed solely to localgrow.','Recorded restoration is verified here. A separate later live audit is required for current service identity.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-archive','--sourcearchive',type=Path,required=True);p.add_argument('--artifacts-archive','--artifactsarchive',type=Path,required=True)
    p.add_argument('--archive-bindings',type=Path);p.add_argument('--model',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:result=post_window_review(a.root,a.source_archive,a.artifacts_archive,a.archive_bindings or a.source_archive.parent/'archive-bindings.json',a.model)
    except Exception as exc:result=dict(evidence_valid=False,correctness_pass=False,performance_measured=False,production_adoption=False,decision='reject_incomplete_or_inconsistent_promptclear_sequence_evidence',error=repr(exc))
    with a.out.open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));return 0 if result.get('evidence_valid')else 1

if __name__=='__main__':raise SystemExit(main())
