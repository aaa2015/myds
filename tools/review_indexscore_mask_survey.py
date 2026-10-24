#!/usr/bin/env python3
"""Independent CPU reader of real indexscore masks and a pre-frozen ROI screen.

No inference, Metal work, service operation or performance cohort is run here.
"""
import argparse,hashlib,json,math,struct,sys
from pathlib import Path
sys.dont_write_bytecode=True
ALL_LAYERS=(2,8,14,20,24,28,32,36)
ELIGIBLE=(24,28,32,36)
EXACT='kernel_myds_indexer_score_chunk_exact'
LIB='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81'

def need(v,m):
    if not v:raise ValueError(m)
def parse_json(raw):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'Duplicate JSON key: '+k);d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def load(p):return parse_json(Path(p).read_bytes())
def sha(p,n=None):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        if n is not None:
            data=f.read(n);need(len(data)==n,'Truncated prefix');h.update(data)
        else:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def u(v,m,positive=False):need(type(v)is int and (v>0 if positive else v>=0) and v<=0xffffffff,m);return v
def finite(v,m,positive=False):need(type(v)in (int,float) and math.isfinite(v) and (v>0 if positive else v>=0),m);return v
def reference(r):
    need(type(r)is dict and set(r)in ({'path','sha256'},{'path','bytes','sha256'}),'Complete source descriptor');p=Path(r['path']);need(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p)==r['sha256'],'Absolute original file/SHA')
    if 'bytes'in r:need(type(r['bytes'])is int and r['bytes']>=0 and p.stat().st_size==r['bytes'],'Original file extent')
    return p
def payload(root,r,dtype=None):
    need(type(r)is dict and set(r)=={'path','bytes','sha256','dtype'},'Exact raw payload descriptor');name=Path(r['path']);need(not name.is_absolute() and '..'not in name.parts and name.parts,'Bounded relative raw path');p=root/name;need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'Raw payload stays in actual capture');need(type(r['bytes'])is int and r['bytes']>=0 and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Complete raw payload SHA/extent');need(dtype is None or r['dtype']==dtype,'Raw payload type');return p.read_bytes()
def policy(v):
    need(v['probe']=='indexscore_mask_survey_v1' and v['nominal_prefix_tokens']==[48000,96000] and v['eligible_consumer_layers']==list(ELIGIBLE) and v['score_layers_all']==list(ALL_LAYERS),'Frozen two-length/eight-source scope')
    need(v['tile_keys']==16 and v['tile_queries']==8 and v['mask_block_keys']==8 and type(v['minimum_warm_strict_tile_coverage'])is float and v['minimum_warm_strict_tile_coverage']==0.20 and v['minimum_warm_diagnostic_opportunity_fraction']==0.03 and v['minimum_warm_cached_fraction_of_cold_prompt']==0.90,'Literal data-before20%/3%/90% screen thresholds')
    need(v['total_sequences']==4 and v['total_generations']==8 and v['normal_timing_samples_per_scenario']==1 and v['ordinary_and_diagnostic_sequences_per_length']==1 and v['same_context_phases']==['cold','warm_append'],'One ordinary and one diagnostic cold/warm sequence per length')
    need(all(v[k]is True for k in ('both_warm_lengths_must_pass','all_four_consumer_layers_complete_in_each_phase','all_eight_score_layers_disclosed_in_each_phase','all_score_records_sampled','producer_consumer_full_row_bytes_must_match','source_and_request_bytes_frozen_before_GPU')),'Complete raw/source coverage requirements')
    return v

def mask_rows(raw,nc,glen,bits,stride):
    need(len(raw)==glen*stride*4,'Complete original mask view extent');blocks=(nc+7)//8;need(stride>=((blocks+31)//32 if bits else blocks),'Actual mask row stride');rows=[]
    for t in range(glen):
        row=raw[t*stride*4:(t+1)*stride*4]
        if bits:value=int.from_bytes(row,'little')&((1<<blocks)-1)
        else:
            value=0
            for b in range(blocks):
                x=struct.unpack_from('<f',row,b*4)[0];need(x==0.0 or x==-math.inf,'Consumed float mask must be0/-INF')
                if x==0.0:value|=1<<b
        rows.append(value)
    return rows

def analyze_mask(raw,nc,glen,start,ratio,bits,stride):
    """Exact old causal predicate; conservative full8x16 opportunity only."""
    for name,value in (('nc',nc),('glen',glen),('ratio',ratio),('stride',stride)):u(value,name,True)
    u(start,'start');need(start+glen<=0xffffffff,'No32bit causal wrap');need(bits in (0,1) and type(bits)is int,'Bitset format')
    rows=mask_rows(raw,nc,glen,bits,stride);nt=(nc+15)//16;nq=(glen+7)//8;total=nt*nq;active=full=skips=causal_visible_skips=0;query_tail_active=key_tail_active=0;visible=allowed=0
    for t,row in enumerate(rows):
        vis=min(nc,(start+t+1)//ratio);n,rem=divmod(vis,8);visible+=vis;allowed+=8*(row&((1<<n)-1)).bit_count()+(rem if rem and (row>>n)&1 else 0)
    for tb in range(0,glen,8):
        last=min(tb+8,glen);max_vis=min(nc,(start+last)//ratio);count=(max_vis+15)//16;active+=count
        if last-tb<8:query_tail_active+=count;continue
        union=effective=0
        for t in range(tb,last):
            union|=rows[t];vis=min(nc,(start+t+1)//ratio);effective|=rows[t]&((1<<((vis+7)//8))-1)
        complete=min(count,nc//16);key_tail_active+=count-complete;full+=complete
        # Bits0,2,... are one per16-key tile. Pairwise OR detects any allowed
        # block without a Python loop over keys/cells or dispatched tile columns.
        even=((1<<(complete*2))-1)//3
        skips+=complete-((union|(union>>1))&even).bit_count()
        causal_visible_skips+=complete-((effective|(effective>>1))&even).bit_count()
    return dict(total_dispatched_tiles=total,causal_already_skipped_tiles=total-active,noncausal_dispatched_tiles=active,full8x16_noncausal_tiles=full,strict_full_mask_skip_tiles=skips,strict_tile_coverage=skips/active if active else 0.0,strict_fraction_of_full_tiles=skips/full if full else 0.0,causal_valid_cells_all_masked_full_tiles=causal_visible_skips,causal_boundary_only_extra_tiles=causal_visible_skips-skips,query_tail_noncausal_tiles=query_tail_active,key_tail_noncausal_tiles=key_tail_active,causal_visible_cells=visible,allowed_visible_cells=allowed,masked_visible_cells=visible-allowed,fine_grained_masked_cell_fraction=(visible-allowed)/visible if visible else 0.0)

def ordinal(v,m):need(type(v)is int and v>=0,m);return v

def span(v):
    need(type(v)is dict and set(v)=={'start_s','end_s','span_ms','commit_before','commit_ordinal'},'Actual completed current-CB timestamps/commit ordinals')
    a=finite(v['start_s'],'GPUstart',True);b=finite(v['end_s'],'GPUend',True);s=finite(v['span_ms'],'GPUspan',True)
    before=ordinal(v['commit_before'],'Original commit-before ordinal');after=ordinal(v['commit_ordinal'],'Original completed commit ordinal')
    need(after==before+1,'Exactly one ACTIVE CB committed and waited; no flush/no-active stale readback')
    need(b>a and abs(s-(b-a)*1000)<=max(1e-6,s*1e-9),'Original CB timestamp arithmetic');return b-a

def execution(r,previous=None):
    need(type(r['before_is_observer_empty_resume'])is bool and type(r['batch_active_before'])is int and r['batch_active_before']==1,'Actual active upstream CB state')
    need(all(type(r[k])is int and r[k]==1 for k in ('actual_auto_barrier_before','actual_auto_barrier_score','effective_concurrent_score')),'Production auto1 and actual scorer Concurrent1')
    mode=r['effective_concurrent_before'];need(type(mode)is int,'Actual before encoder state')
    if r['before_is_observer_empty_resume']:
        same=('scenario_id','phase','layer','owner','ratio','base_pos','chunk_start','chunk_rows','input_skip')
        need(previous is not None and mode==-1 and all(r[k]==previous[k] for k in same) and r['group_off']==previous['group_off']+previous['glen'] and r['g_start']==previous['g_start']+previous['glen'],'Only explicit observer-owned adjacent subgroup empty-CB resume permits actual before=-1')
    else:need(mode==1,'First production entry has an actual Concurrent encoder; no fabricated mode')
    up=ordinal(r['upstream_commit_ordinal'],'Completed upstream commit ordinal');previous_end=0.0;previous_ordinal=0
    if previous is not None:
        previous_end=previous['filter_topk_gpu_cb']['end_s'];previous_ordinal=previous['filter_topk_gpu_cb']['commit_ordinal']
        need(up>previous_ordinal,'Upstream completion must follow previous original filter CB')
    spans=[r['score_gpu_cb']]
    if r['layer']==20:spans.append(r['mask_producer_gpu_cb'])
    spans.append(r['filter_topk_gpu_cb'])
    for cb in spans:
        span(cb);need(cb['commit_before']==up and cb['commit_ordinal']==up+1,'Scorer/producer/filter each is the next actual completed CB')
        need(cb['start_s']>=previous_end and cb['end_s']>previous_end and cb['commit_ordinal']>previous_ordinal,'Strictly fresh monotonic CB span/ordinal; no stale or duplicated timings')
        up=previous_ordinal=cb['commit_ordinal'];previous_end=cb['end_s']
    return True

def read_run(directory,run,binding,ordinary=False):
    directory=Path(directory);v=load(directory/'survey.json');records=[parse_json(x) for x in (directory/'records.jsonl').read_text().splitlines() if x.strip()];need(v['schema_version']==1 and v['scenario_id']==run['scenario_id'],'Actual scenario identity');need(v['records']==records and v['record_count']==len(records) and v['records_sha256']==sha(directory/'records.jsonl'),'Every original record and JSONL byte identity')
    need(v['kind']==('ordinary_native_indexscore_reference'if ordinary else'actual_native_indexscore_mask_survey') and v['observer_commit_wait']is(not ordinary) and type(v['ordinary_uninstrumented_core'])is int and v['ordinary_uninstrumented_core']==int(ordinary) and v['performance_from_capture']is False,'Ordinary/perturbed diagnostic scope')
    role='ordinary'if ordinary else'diagnostic';need(v['binary_sha256']==binding['binaries'][role]['sha256'] and v['metallib_sha256']==binding['metallib']['sha256']==LIB and v['baseline_core_source_manifest_sha256']==binding['baseline_core_source_manifest_sha256'],'Actual source/binary/library identity');need(v['environment']==binding['expected_environment'] and len(v['environment'])==53,'Exact production53 environment');need(v['model_bytes']==binding['model']['bytes'] and v['model_header_bytes']==1048576 and v['model_header_sha256']==binding['model_header_sha256'],'Actual model identity')
    need(v['min_start']==0 and v['max_start']==0xffffffff and v['dump_inputs']is False,'No tail-only/range-selected opportunity extrapolation')
    need(len(v['phases'])==2 and [p['phase'] for p in v['phases']]==[0,1],'Same-context cold then warm');phases=[];output=[]
    for phase,p in enumerate(v['phases']):
        need(p['scenario_id']==run['scenario_id'] and p['observer_commit_wait']is(not ordinary) and type(p['ordinary_uninstrumented_core'])is int and p['ordinary_uninstrumented_core']==int(ordinary) and p['performance_from_capture']is False,'Actual phase scope')
        prefix='cold'if phase==0 else'warm';prompt=reference(run[prefix+'_prompt']);tokens=reference(run[prefix+'_tokens']);need(p['prompt_sha256']==sha(prompt) and p['prompt_bytes']==prompt.stat().st_size and p['tokens_sha256']==sha(tokens) and p['tokens_bytes']==tokens.stat().st_size==p['actual_tokens']*4,'Real frozen full prompt and actual tokenizer IDs');u(p['actual_tokens'],'Actual tokens',True);need(p['actual_tokens']>16384 and p['actual_tokens']<=327680,'Long real context');u(p['cached_tokens'],'cached');u(p['prefill_tokens'],'prefill',True);need(p['cached_tokens']==0 if phase==0 else p['cached_tokens']>0,'Actual cold/no-cache and warm/reused prefix');finite(p['prefill_wall_s'],'Original ordinary/diagnostic prefill wall',True);finite(p['generation_wall_s'],'Generation wall',True)
        need(type(p['dsp_present'])is int and p['dsp_present']==0 and all(p[k]==0 for k in ('draft_cycles','pld_cycles','batch2_calls')),'No speculative draft/PLD/batch2 forwards');need(p['pcache_len_before']==0 if phase==0 else p['pcache_len_before']>=v['phases'][0]['actual_tokens']-1,'Actual same-context cache sequence')
        text=payload(directory,p['output'],'bytes');ids=payload(directory,p['output_token_ids'],'i32le');need(len(ids)%4==0 and len(ids)<=4 and len(text)==p['output_bytes'] and hashlib.sha256(text).hexdigest()==p['output_sha256'],'Actual max1 callback output bytes/IDs');output.append((text,ids,p['stop_reason']))
        selected=records[p['first_record']:p['first_record']+p['record_count']];need(all(r['phase']==phase for r in selected),'Complete actual phase record interval');need(not ordinary or not selected,'Ordinary core has no observer records');phases.append(dict(metadata=p,records=selected))
    need(sum(p['record_count'] for p in v['phases'])==len(records) and v['phases'][0]['first_record']==0 and v['phases'][1]['first_record']==v['phases'][0]['record_count'],'Exact phase record partition');need([r['case'] for r in records]==list(range(len(records))),'Complete ordered unique record IDs');need(v['sampled_count']==len(records) and all(r['sampled']is True for r in records),'All actual score dispatches sampled')
    for i,r in enumerate(records):execution(r,records[i-1] if i else None)
    return dict(manifest=v,directory=directory,phases=phases,outputs=output)

def expected_intervals(metadata,tokens):
    """Source-derived batched-score domain, excluding final prompt-token decode.

    Exact53 defaults: chunk2048, auto macroblock<=106496, suffix shift512,
    anchor snapshots on, gather on, no guarded grid/per-token indexer override.
    """
    n=u(metadata['actual_tokens'],'Actual tokens',True);cached=u(metadata['cached_tokens'],'Actual cached tokens')
    need(metadata['prefill_tokens']==n-1-cached and len(tokens)==n*4,'Engine actual prefill domain excludes final prompt-token decode')
    calls=[(cached,n-1-cached)]
    if cached==0:
        ids=struct.unpack('<'+'i'*n,tokens)
        anchor=next((i+1 for i,x in enumerate(ids[:-1])if x==128803),0)
        if 4<=anchor<=1024 and anchor<n-1:calls=[(0,anchor),(anchor,n-1-anchor)]
    out=[]
    for base,count in calls:
        k=(count+106495)//106496;mb=min(131072,max(512,((count+k-1)//k+511)//512*512))
        for off in range(0,count,mb):
            a=base+off;b=a+min(mb,count-off)
            for layer in ALL_LAYERS:
                begin=max(a,b-(1+(39-layer)*127+512))if layer>=20 and base+count>=512 else a
                if layer<20 and begin==0:begin=1 # ratio2 scalar no-op first row
                if begin<b:out.append((layer,a,begin,b))
    return out

def complete_domain(run,phase,tokens):
    actual={}
    for r in run['phases'][phase]['records']:
        key=(r['layer'],r['macroblock_base_pos']);actual.setdefault(key,[]).append((r['g_start'],r['g_start']+r['glen']))
    expected={(layer,base):(lo,hi)for layer,base,lo,hi in expected_intervals(run['phases'][phase]['metadata'],tokens)}
    need(set(actual)==set(expected),'All source-derived actual prefill macroblock/layer domains captured; no missing scalar fallback')
    for key,intervals in actual.items():
        cursor,end=expected[key]
        for lo,hi in sorted(intervals):need(lo==cursor and hi>lo and hi<=end,'Complete query domain without gaps/overlap or double-added input skip');cursor=hi
        need(cursor==end,'Every actual batched-score query observed, excluding final prompt-token decode only')
    return dict(source_expected_query_intervals=[dict(layer=l,macroblock_base_pos=base,begin=lo,end=hi)for l,base,lo,hi in expected_intervals(run['phases'][phase]['metadata'],tokens)],source_complete_batched_score_domain=True,final_prompt_token_decode_excluded=True)

def analyze_phase(run,phase):
    directory=run['directory'];p=run['phases'][phase];records=p['records'];need({r['layer'] for r in records}==set(ALL_LAYERS),'All eight actual score-source layers complete');producer={};results=[];layer_times={l:0.0 for l in ALL_LAYERS};last_end=0.0
    for r in records:
        for k in ('layer','owner','ratio','base_pos','macroblock_base_pos','chunk_start','chunk_rows','input_skip','group_off','g_start','nc','nc_raw','glen','mask_stride_words','mask_bits'):u(r[k],k)
        need(r['layer']in ALL_LAYERS and r['owner']==(ALL_LAYERS.index(r['layer']) if r['layer']<20 else 3) and r['ratio']==(2 if r['layer']<20 else 1),'Real layer/owner/ratio');need(r['glen']>0 and r['nc_raw']==(r['g_start']+r['glen'])//r['ratio'] and r['nc']==max(512,r['nc_raw']) and r['g_start']==r['base_pos']+r['chunk_start']+r['input_skip']+r['group_off'] and r['macroblock_base_pos']==r['base_pos'] and r['group_off']+r['glen']<=r['chunk_rows']-r['input_skip'],'Actual already-offset subgroup geometry');need(r['g_start']+r['glen']<=p['metadata']['actual_tokens'] and r['input_skip']in (0,1),'Actual absolute query interval')
        need(r['scenario_id']==p['metadata']['scenario_id'] and r['phase']==phase and r['prefill_cached_tokens']==p['metadata']['cached_tokens'] and r['prefill_tokens']==p['metadata']['prefill_tokens'],'Actual phase counters per dispatch');need(r['tile_keys']==16 and r['tile_queries']==8 and r['selected_kernel']==EXACT  and r['observer_commit_wait']is True and r['performance_from_capture']is False,'Actual exact16x8 provider in production concurrent/auto1')
        dt=span(r['score_gpu_cb']);need(r['score_gpu_cb']['start_s']>=last_end,'Actual original CB chronological order');layer_times[r['layer']]+=dt;tail=r['filter_topk_gpu_cb'];span(tail);need(tail['start_s']>=r['score_gpu_cb']['end_s'],'Scorer then original consumer CB')
        if r['layer']==20:
            need(r.get('producer_mask_ready')is True and 'raw_mask'not in r,'Producer mask is unavailable before scoring');raw=payload(directory,r['produced_mask'],'u32le'if r['mask_bits'] else'f32le');need(len(raw)==r['glen']*r['mask_stride_words']*4,'Producer full rowstride span');span(r['mask_producer_gpu_cb']);need(r['mask_producer_gpu_cb']['start_s']>=r['score_gpu_cb']['end_s'] and tail['start_s']>=r['mask_producer_gpu_cb']['end_s'],'POST_MASK CB is separate, no scorer pollution')
            for t in range(r['glen']):
                key=(r['scenario_id'],phase,r['macroblock_base_pos'],r['owner'],r['g_start']+t);need(key not in producer,'Unique real producer logical query row');stride=r['mask_stride_words']*4;producer[key]=(raw[t*stride:(t+1)*stride],r['mask_bits'],r['mask_stride_words'])
        elif r['layer']in ELIGIBLE:
            raw=payload(directory,r['raw_mask'],'u32le'if r['mask_bits'] else'f32le');need(len(raw)==r['glen']*r['mask_stride_words']*4,'Complete consumer view')
            for t in range(r['glen']):
                key=(r['scenario_id'],phase,r['macroblock_base_pos'],r['owner'],r['g_start']+t);need(key in producer,'Every consumer row has the actual layer20 producer');stride=r['mask_stride_words']*4;need(producer[key]==(raw[t*stride:(t+1)*stride],r['mask_bits'],r['mask_stride_words']),'Complete producer/consumer raw row bytes, not subgroup index pairing')
            stats=analyze_mask(raw,r['nc'],r['glen'],r['g_start'],r['ratio'],r['mask_bits'],r['mask_stride_words']);results.append(dict(case=r['case'],layer=r['layer'],g_start=r['g_start'],glen=r['glen'],nc=r['nc'],isolated_score_gpu_s=dt,**stats))
        else:need('raw_mask'not in r and 'produced_mask'not in r,'Early sources have no block mask input')
        last_end=tail['end_s']
    # Every observed original chunk is exhaustively partitioned by actual subgroups.
    chunks={}
    for r in records:
        key=tuple(r[k] for k in ('layer','owner','ratio','base_pos','chunk_start','chunk_rows','input_skip'))
        need(r['group_off']==chunks.get(key,0),'Complete ordered subgroup partition; no omitted/duplicate subgroup')
        chunks[key]=r['group_off']+r['glen']
    need(all(end==key[-2]-key[-1] for key,end in chunks.items()),'Every observed chunk reaches its actual end')
    need({x['layer'] for x in results}==set(ELIGIBLE),'All four eligible consumers actually observed');active=sum(x['noncausal_dispatched_tiles'] for x in results);skip=sum(x['strict_full_mask_skip_tiles'] for x in results);need(active>0 and any(x['nc']>16384 for x in results),'Real nontrivial long-context scoring coverage')
    return dict(strict_skip_tiles=skip,noncausal_dispatched_tiles=active,strict_tile_coverage=skip/active,isolated_eligible_opportunity_s=sum(x['isolated_score_gpu_s']*x['strict_tile_coverage'] for x in results),all_eight_layer_score_gpu_s={str(k):v for k,v in layer_times.items()},eligible_dispatches=results,producer_rows_verified=len(producer),producer_consumer_full_row_bytes_equal=True,all_source_records=len(records))

def source_manifest(descriptor,root):
    value=load(reference(descriptor));rows=value.get('runtime_sha256',value.get('files',value.get('executed_inputs',value)))
    need(type(rows)is dict and rows,'Original source manifest map')
    allrows=dict(rows)
    for name,identity in value.get('extra_files',{}).items():
        need(name not in allrows or allrows[name]==identity,'Conflicting extra source identity');allrows[name]=identity
    root=Path(root).resolve(strict=True)
    for name,identity in allrows.items():
        rel=Path(name);need(not rel.is_absolute() and '..'not in rel.parts and rel.parts,'Bounded source member')
        p=root/rel;need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'Actual source member regular/bounded')
        digest=identity if type(identity)is str else identity['sha256'];need(sha(p)==digest,'Actual frozen source bytes: '+name)
        if type(identity)is dict and 'bytes'in identity:need(type(identity['bytes'])is int and p.stat().st_size==identity['bytes'],'Source member extent')
    return len(rows)

def warm_gate(phase,reuse,coverage,proxy):return phase==1 and reuse>=0.90 and coverage>=0.20 and proxy>=0.03

def review(a):
    need(sha(a.binding)==a.binding_sha256,'Explicit frozen binding SHA');b=load(a.binding);need(b['schema_version']==1 and b['probe']=='indexscore_mask_survey_v1','Actual survey binding');plan=load(reference(b['plan']));pol=policy(load(reference(plan['roi_policy'])));need(len(b['runs'])==2 and [r['nominal_prefix_tokens'] for r in b['runs']]==[48000,96000] and len({r['scenario_id'] for r in b['runs']})==2,'Frozen two scenario matrix');need(b['model_header_bytes']==1048576,'Model header extent')
    model=a.model.resolve(strict=True);st=model.stat();actual=dict(path=str(model),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev);need(actual==b['model'] and sha(model,1048576)==b['model_header_sha256'],'Actual current GGUF full stat/first1MiB identity');need(all(b['expected_environment'].get(k)=='1' for k in ('MYDS_PREFILL_MOE_WORKSPACE','MYDS_PREFILL_PROMPT_VIEW_CLEAR','MYDS_METAL_CONCURRENT','MYDS_PREFIX_CACHE','MYDS_PLD')),'Preserve adopted v4 baseline')
    baseline=b['baseline'];need(baseline['binary_sha256']=='9cf2a670a22632874829398eece031111e438b4ca803860a2e6bcbcc7e94c57a' and baseline['metallib_sha256']==LIB and baseline['config_sha256']=='79e119ae1a44e585305b1bd641989e150436284d68b6d9a7476ef9656776df09' and type(baseline['nice'])is int and baseline['nice']==0 and baseline['effective_myds_env']==b['expected_environment'] and baseline['initial_environment_count']==22 and baseline['initial_environment_sha256']=='e2727c8774cbeb42319e17af724ea8274a1bb38e7a596c90122a4db51687654d','Exact adopted source/environment reference')
    known=load(reference(plan['reader_baseline_binding']));need(all(baseline[k]==known[k] for k in ('binary_sha256','metallib_sha256','config_sha256','effective_myds_env','initial_environment_count','initial_environment_sha256')),'Independent exact known53/initial22 baseline')
    env=b['expected_environment'];need(env.get('MYDS_PREFILL_CHUNK_SIZE')=='2048' and env.get('MYDS_SUFFIX_SHRINK_SHIFT')=='512' and all(k not in env for k in ('MYDS_PREFILL_MACROBLOCK_SIZE','MYDS_SUFFIX_SHRINK','MYDS_ATTN_GATHER','MYDS_SUFFIX_SHRINK_MIN_TOKENS','MYDS_ANCHOR_SNAPSHOT','MYDS_INDEXER_PER_TOKEN','MYDS_PREFILL_GUARDED_GRID')),'Source-derived complete-domain exact baseline defaults')
    need([x['name']for x in plan['cases']]==['c48','c96'],'Frozen exact two-case order')
    for r,c in zip(b['runs'],plan['cases']):
        need(r['scenario_id']==('48k'if c['name']=='c48'else'96k') and all(r[prefix+'_'+kind]==c[role][kind]for prefix,role in (('cold','base'),('warm','append'))for kind in ('prompt','tokens')) and r['source_provenance']==c['source_provenance'],'Actual run requests/token/provenance exactly plan-bound')
    need(set(b['binaries'])=={'ordinary','diagnostic'} and reference(plan['reader']).resolve()==Path(__file__).resolve(),'Actual executed independent reader frozen in plan')
    need(plan['baseline_core_manifest']['sha256']==b['baseline_core_source_manifest_sha256'],'Baseline91 manifest binding')
    for role,r in b['binaries'].items():
        binary=reference(r);root=binary.parent.parent
        n=source_manifest(plan['baseline_core_manifest'if role=='ordinary'else'diagnostic_core_manifest'],root)
        if role=='ordinary':need(n==91,'Unmodified exact868+v4 source91')
        source_manifest(plan['harness_source_manifest'],root)
    reference(b['metallib']);reports=[]
    for spec in b['runs']:
        reference(spec['source_provenance']);ordinary=read_run(spec['ordinary_dir'],spec,b,True);diag=read_run(spec['diagnostic_dir'],spec,b,False);need(ordinary['outputs']==diag['outputs'],'Actual observer callback bytes/token IDs agree, no full-state claim');phases=[]
        for phase in range(2):
            op,dp=ordinary['phases'][phase]['metadata'],diag['phases'][phase]['metadata'];need(all(op[k]==dp[k] for k in ('prompt_bytes','prompt_sha256','tokens_bytes','tokens_sha256','actual_tokens','cached_tokens','prefill_tokens','pcache_len_before','pcache_len_after','stop_reason','output_sha256')),'Same real ordinary/diagnostic request/cache/output shape')
            domain=complete_domain(diag,phase,reference(spec[('cold'if phase==0 else'warm')+'_tokens']).read_bytes());stats=analyze_phase(diag,phase);stats.update(domain);reuse=dp['cached_tokens']/diag['phases'][0]['metadata']['actual_tokens'];fraction=stats['isolated_eligible_opportunity_s']/op['prefill_wall_s'];passed=warm_gate(phase,reuse,stats['strict_tile_coverage'],fraction);phases.append(dict(phase=phase,role='cold'if phase==0 else'warm_append',actual_tokens=dp['actual_tokens'],cached_tokens=dp['cached_tokens'],prefill_tokens=dp['prefill_tokens'],cached_fraction_of_setup=reuse,ordinary_prefill_wall_s=op['prefill_wall_s'],diagnostic_prefill_wall_s=dp['prefill_wall_s'],diagnostic_opportunity_fraction=fraction,warm_screen_pass=passed,**stats))
        reports.append(dict(scenario_id=spec['scenario_id'],nominal_prefix_tokens=spec['nominal_prefix_tokens'],phases=phases))
    passed=all(r['phases'][1]['warm_screen_pass'] for r in reports)
    return dict(schema_version=1,evidence_valid=True,roi_screen_pass=passed,candidate_implementation_eligible=passed,candidate_implementation_authorized=passed,performance_gate_pass=False,production_adoption=False,performance_measured=False,decision='eligible_for_separate_candidate_implementation_and_real_component_gates'if passed else'reject_insufficient_complete_tile_opportunity',binding_sha256=sha(a.binding),plan_sha256=b['plan']['sha256'],roi_policy_sha256=plan['roi_policy']['sha256'],reader_sha256=sha(__file__),actual_model=actual,model_header_sha256=b['model_header_sha256'],source_and_binary_binding=b,scenarios=reports,limits=['Diagnostic score CB spans include original barriers/dispatch overhead and a perturbed graph. The weighted ratio against N1 ordinary prefill wall is an opportunity proxy, not a strict upper bound or actual critical-path share.','Only strict complete8queryx16key mask tiles count. Existing causal early returns, query/key tails and causal-boundary-only extra tiles are excluded from the numerator.','Ordinary/diagnostic output-byte agreement is observer sanity, not full-state/F32-ULP candidate qualification.','Cold opportunities are disclosed even when zero. No performance, cold-nonregression, cancellation/recovery, full-model A-B-A or adoption gate is passed by this survey.'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'Preserve every independent report')
    try:r=review(a)
    except Exception as e:r=dict(schema_version=1,evidence_valid=False,roi_screen_pass=False,candidate_implementation_eligible=False,candidate_implementation_authorized=False,performance_gate_pass=False,production_adoption=False,error=repr(e),reader_sha256=sha(__file__))
    with a.out.open('x') as f:json.dump(r,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:r.get(k) for k in ('evidence_valid','roi_screen_pass','decision','error')},sort_keys=True));return 0 if r['evidence_valid']else 1
if __name__=='__main__':raise SystemExit(main())
