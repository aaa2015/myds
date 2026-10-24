#!/usr/bin/env python3
"""CPU-only independent survey reader controls; never loads a model or Metal."""
import argparse,copy,hashlib,importlib.util,json,math,random,struct,tempfile
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode=True
P=Path(__file__).with_name('review_indexscore_mask_survey.py')
spec=importlib.util.spec_from_file_location('survey_reader',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def desc(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=m.sha(p))
def rawdesc(root,name,raw,dtype):p=root/name;p.write_bytes(raw);return dict(path=name,bytes=len(raw),sha256=m.sha(p),dtype=dtype)
def rawmask(rows,nc,bits=1,stride=None):
    n=(nc+7)//8;stride=stride or ((n+31)//32 if bits else n)
    return b''.join(r.to_bytes(stride*4,'little') if bits else b''.join(struct.pack('<f',0.0 if (r>>i)&1 else -math.inf) for i in range(stride)) for r in rows),stride

def oracle(rows,nc,glen,start,ratio):
    total=causal=active=full=skip=effective=qt=kt=visible=allowed=0
    for t in range(glen):
        for k in range(min(nc,(start+t+1)//ratio)):
            visible+=1;allowed+=(rows[t]>>(k//8))&1
    for t0 in range(0,glen,8):
        last=min(t0+8,glen);vis=(start+last)//ratio
        for k0 in range(0,nc,16):
            total+=1
            if k0>=vis:causal+=1;continue
            active+=1
            if last-t0<8:qt+=1;continue
            if k0+16>nc:kt+=1;continue
            full+=1
            a=any((rows[t]>>(k//8))&1 for t in range(t0,last) for k in range(k0,k0+16))
            e=any((rows[t]>>(k//8))&1 for t in range(t0,last) for k in range(k0,min(k0+16,nc,(start+t+1)//ratio)))
            skip+=not a;effective+=not e
    return dict(total_dispatched_tiles=total,causal_already_skipped_tiles=causal,noncausal_dispatched_tiles=active,full8x16_noncausal_tiles=full,strict_full_mask_skip_tiles=skip,causal_valid_cells_all_masked_full_tiles=effective,causal_boundary_only_extra_tiles=effective-skip,query_tail_noncausal_tiles=qt,key_tail_noncausal_tiles=kt,causal_visible_cells=visible,allowed_visible_cells=allowed,masked_visible_cells=visible-allowed)

def cb(i,start):return dict(start_s=start,end_s=start+.04,span_ms=40.0,commit_before=i-1,commit_ordinal=i)
def rec(i,layer,start=20000,off=0,rows=8,skip=0):
    r=dict(case=i,scenario_id='48k',phase=0,prefill_cached_tokens=0,prefill_tokens=48000,layer=layer,owner=m.ALL_LAYERS.index(layer)if layer<20 else 3,ratio=2 if layer<20 else 1,base_pos=0,macroblock_base_pos=0,chunk_start=start-skip,chunk_rows=rows+skip,input_skip=skip,group_off=off,g_start=start+off,nc=max(512,(start+off+rows)//(2 if layer<20 else 1)),nc_raw=(start+off+rows)//(2 if layer<20 else 1),glen=rows,mask_stride_words=200,mask_bits=1,mask_view_offset=0,mask_view_bytes=rows*800,tile_keys=16,tile_queries=8,sampled=True,observer_commit_wait=True,performance_from_capture=False,actual_auto_barrier_before=1,effective_concurrent_before=1,before_is_observer_empty_resume=False,batch_active_before=1,actual_auto_barrier_score=1,effective_concurrent_score=1,selected_kernel=m.EXACT,upstream_commit_ordinal=i*5+1)
    j=r['upstream_commit_ordinal'];t=1+i
    r['score_gpu_cb']=cb(j+1,t)
    if layer==20:r['mask_producer_gpu_cb']=cb(j+2,t+.05);j+=1
    r['filter_topk_gpu_cb']=cb(j+2,t+.1)
    return r

def fixture(root):
    env={f'MYDS_CPU_FIXTURE_{i}':'0' for i in range(46)}
    env.update(MYDS_PREFILL_CHUNK_SIZE='2048',MYDS_SUFFIX_SHRINK_SHIFT='512')
    env.update({k:'1'for k in('MYDS_PREFILL_MOE_WORKSPACE','MYDS_PREFILL_PROMPT_VIEW_CLEAR','MYDS_METAL_CONCURRENT','MYDS_PREFIX_CACHE','MYDS_PLD')})
    model=root/'model-fixture';model.write_bytes(b'CPU_ONLY_NOT_A_GGUF\n'+b'0'*1048576);st=model.stat()
    b=dict(schema_version=1,probe='indexscore_mask_survey_v1',model=dict(path=str(model),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev),model_header_bytes=1048576,model_header_sha256=m.sha(model,1048576),expected_environment=env,binaries={},runs=[])
    b['baseline']=dict(binary_sha256='9cf2a670a22632874829398eece031111e438b4ca803860a2e6bcbcc7e94c57a',metallib_sha256=m.LIB,config_sha256='79e119ae1a44e585305b1bd641989e150436284d68b6d9a7476ef9656776df09',nice=0,effective_myds_env=env,initial_environment_count=22,initial_environment_sha256='e2727c8774cbeb42319e17af724ea8274a1bb38e7a596c90122a4db51687654d')
    source={f'src/cpu-fixture-{i}.c':hashlib.sha256(str(i).encode()).hexdigest()for i in range(91)}
    for role in ('ordinary','diagnostic'):
        tree=root/role;(tree/'bin').mkdir(parents=True)
        binary=tree/'bin/survey';binary.write_bytes(b'CPU fixture binary '+role.encode());b['binaries'][role]=desc(binary)
        for i,name in enumerate(source):p=tree/name;p.parent.mkdir(exist_ok=True);p.write_bytes(str(i).encode())
        p=tree/'tools/harness';p.parent.mkdir();p.write_bytes(b'CPU fixture source')
    core=root/'core.json';write(core,dict(files=source));harness=root/'harness.json';write(harness,dict(files={'tools/harness':hashlib.sha256(b'CPU fixture source').hexdigest()}));b['baseline_core_source_manifest_sha256']=m.sha(core)
    lib=root/'fixture-metallib';lib.write_bytes(b'CPU fixture library');b['metallib']=desc(lib);b['metallib']['sha256']=m.sha(lib)
    # The loader independently requires the production library SHA. Patch SHA
    # only for this explicit fake-library leaf, never production input files.
    original=m.sha
    def mocked_sha(p,n=None):return m.LIB if Path(p)==lib else original(p,n)
    m.sha=mocked_sha;b['metallib']['sha256']=m.LIB
    policy=root/'policy.json';write(policy,m.load(Path(__file__).parents[1]/'docs/benchmarks/2026-10-24-indexscore-mask-survey/evidence/roi-screen-policy-v1.json'))
    plan=root/'plan.json';write(plan,dict(roi_policy=desc(policy),reader=desc(P),baseline_core_manifest=desc(core),diagnostic_core_manifest=desc(core),harness_source_manifest=desc(harness)))
    known=root/'known.json';write(known,{k:v for k,v in b['baseline'].items() if k!='nice'})
    pv=m.load(plan);pv['reader_baseline_binding']=desc(known);pv['cases']=[];write(plan,pv)
    b['plan']=desc(plan)
    for scenario,nt in [('48k',48000),('96k',96000)]:
        run=dict(scenario_id=scenario,nominal_prefix_tokens=nt)
        for phase,prefix in enumerate(('cold','warm')):
            for kind,raw in [('prompt',b'cpu fixture rendered prompt '+str(nt+phase).encode()),('tokens',struct.pack('<i',1)*(nt+phase*2000))]:
                p=root/f'{scenario}-{prefix}-{kind}';p.write_bytes(raw);run[prefix+'_'+kind]=desc(p)
        prov=root/f'{scenario}-provenance';prov.write_text('New CPU fixture; not real inference.');run['source_provenance']=desc(prov)
        for ordinary in (True,False):
            role='ordinary'if ordinary else'diagnostic';directory=root/f'{scenario}-{role}';directory.mkdir();run[role+'_dir']=str(directory);records=[];phases=[]
            for phase,prefix in enumerate(('cold','warm')):
                first=len(records);n=nt+phase*2000;cached=nt if phase else 0
                if not ordinary:
                    meta=dict(actual_tokens=n,cached_tokens=cached,prefill_tokens=n-1-cached)
                    tokenraw=Path(run[prefix+'_tokens']['path']).read_bytes()
                    for layer,base,lo,hi in m.expected_intervals(meta,tokenraw):
                        for begin in range(lo,hi,2048):
                            rows=min(2048,hi-begin);skip=1 if begin==1 and base==0 and layer<20 else 0
                            r=rec(len(records),layer,start=begin,rows=rows,skip=skip)
                            r.update(scenario_id=scenario,phase=phase,prefill_cached_tokens=cached,prefill_tokens=n-1-cached,base_pos=base,macroblock_base_pos=base,chunk_start=begin-base-skip)
                            nblocks=(r['nc']+7)//8;stride=((nt+2000+7)//8+31)//32;r['mask_stride_words']=stride;r['mask_view_bytes']=rows*stride*4
                            masks=[(1<<(((r['g_start']+t+1)//r['ratio']+7)//8))-1 if phase==0 else 0 for t in range(rows)];raw,_=rawmask(masks,r['nc'],1,stride)
                            if layer==20:r.update(produced_mask=rawdesc(directory,f'{r["case"]}-producer',raw,'u32le'),producer_mask_ready=True)
                            elif layer in m.ELIGIBLE:r['raw_mask']=rawdesc(directory,f'{r["case"]}-consumer',raw,'u32le')
                            records.append(r)
                output=rawdesc(directory,f'{prefix}-output',b'X','bytes');ids=rawdesc(directory,f'{prefix}-ids',struct.pack('<i',42),'i32le')
                phases.append(dict(scenario_id=scenario,phase=phase,prompt_bytes=run[prefix+'_prompt']['bytes'],prompt_sha256=run[prefix+'_prompt']['sha256'],tokens_bytes=n*4,tokens_sha256=run[prefix+'_tokens']['sha256'],actual_tokens=n,cached_tokens=cached,prefill_tokens=n-1-cached,prefill_wall_s=.4,generation_wall_s=.5,pcache_len_before=nt if phase else 0,pcache_len_after=n,first_record=first,record_count=len(records)-first,stop_reason='max_new_tokens',output_bytes=1,output_sha256=output['sha256'],output=output,output_token_ids=ids,dsp_present=0,draft_cycles=0,pld_cycles=0,batch2_calls=0,observer_commit_wait=not ordinary,ordinary_uninstrumented_core=int(ordinary),performance_from_capture=False))
            lines=directory/'records.jsonl';lines.write_text(''.join(json.dumps(r)+'\n'for r in records))
            v=dict(schema_version=1,kind='ordinary_native_indexscore_reference'if ordinary else'actual_native_indexscore_mask_survey',scenario_id=scenario,records=records,record_count=len(records),records_sha256=m.sha(lines),sampled_count=len(records),phases=phases,observer_commit_wait=not ordinary,ordinary_uninstrumented_core=int(ordinary),performance_from_capture=False,binary_sha256=b['binaries'][role]['sha256'],metallib_sha256=m.LIB,baseline_core_source_manifest_sha256=b['baseline_core_source_manifest_sha256'],environment=env,model_bytes=b['model']['bytes'],model_header_bytes=1048576,model_header_sha256=b['model_header_sha256'],min_start=0,max_start=0xffffffff,dump_inputs=False)
            write(directory/'survey.json',v)
        b['runs'].append(run)
        pv['cases'].append(dict(name='c48'if scenario=='48k'else'c96',base={k:run['cold_'+k]for k in ('prompt','tokens')},append={k:run['warm_'+k]for k in ('prompt','tokens')},source_provenance=run['source_provenance']))
    write(plan,pv);b['plan']=desc(plan)
    binding=root/'binding.json';write(binding,b)
    return b,SimpleNamespace(binding=binding,binding_sha256=m.sha(binding),model=model),original

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();assert not a.out.exists();results=[]
    def check(name,fn,reject=False):
        try:fn()
        except (ValueError,KeyError,TypeError,AssertionError)as e:
            if not reject:raise
            results.append(dict(name=name,passed=True,expected_rejection=repr(e)));return
        assert not reject,name+' accepted invalid evidence';results.append(dict(name=name,passed=True))
    def eq(x,y):assert x==y,(x,y)
    # Small exhaustive and seeded-random per-cell oracle remains independent
    # of the production-size fast implementation and its pair bit operations.
    count=0
    for nc in (1,7,8,15,16,17,31,32,33):
        for glen in (1,7,8,9,16):
            for start in (0,1,7,15,32,100):
                for ratio in (1,2):
                    for pattern in (0,(1<<((nc+7)//8))-1,1,2):
                        rows=[pattern]*glen;raw,stride=rawmask(rows,nc);actual=m.analyze_mask(raw,nc,glen,start,ratio,1,stride);expect=oracle(rows,nc,glen,start,ratio);eq({k:actual[k]for k in expect},expect);count+=1
    rng=random.Random(20261024)
    for _ in range(400):
        nc=rng.randint(1,96);glen=rng.randint(1,24);start=rng.randint(0,128);ratio=rng.choice((1,2));rows=[rng.getrandbits((nc+7)//8)for _ in range(glen)]
        for bits in (0,1):
            raw,stride=rawmask(rows,nc,bits);actual=m.analyze_mask(raw,nc,glen,start,ratio,bits,stride);expect=oracle(rows,nc,glen,start,ratio);eq({k:actual[k]for k in expect},expect);count+=1
    results.append(dict(name='fast_integer_masks_equal_independent_per_cell_oracle',passed=True,cases=count))
    check('float_nonfinite_mask_rejected',lambda:m.analyze_mask(struct.pack('<f',math.nan)*16,16,8,100,1,0,2),True)
    check('truncated_mask_rejected',lambda:m.analyze_mask(b'\0',16,8,100,1,1,1),True)
    base=rec(0,24);check('actual_current_cb_ordinal_positive',lambda:m.execution(base))
    nxt=rec(1,24,start=20000,off=8);base['chunk_rows']=nxt['chunk_rows']=16;nxt.update(before_is_observer_empty_resume=True,effective_concurrent_before=-1)
    check('two_subgroups_observer_owned_empty_cb_positive',lambda:m.execution(nxt,base))
    for name,key,value in [('not_own_empty','before_is_observer_empty_resume',False),('no_active_cb','batch_active_before',0),('nonadjacent_query','g_start',20009),('wrong_layer','layer',28),('newphase','phase',1),('wrong_score_mode','effective_concurrent_score',-1)]:
        bad=copy.deepcopy(nxt);bad[key]=value;check('empty_cb_'+name,lambda bad=bad:m.execution(bad,base),True)
    check('firstentry_empty_rejected',lambda:m.execution(nxt),True)
    for name,mutate in [('stale_duplicate_span',lambda r:r.update(score_gpu_cb=copy.deepcopy(base['score_gpu_cb']))),('flush_noactive_commit_noincrement',lambda r:r['score_gpu_cb'].update(commit_ordinal=r['score_gpu_cb']['commit_before'])),('wrong_current_cb_ordinal',lambda r:r['score_gpu_cb'].update(commit_before=100,commit_ordinal=101)),('timestamp_arithmetic',lambda r:r['score_gpu_cb'].update(span_ms=41)),('scorer_serial',lambda r:r.update(effective_concurrent_score=0))]:
        bad=copy.deepcopy(nxt);mutate(bad);check(name,lambda bad=bad:m.execution(bad,base),True)
    check('frozen_gate_inclusive_boundaries',lambda:eq(m.warm_gate(1,.9,.2,.03),True))
    for name,args in [('cold_not_authorized',(0,1,1,1)),('reuse_below90',(1,.899999,.2,.03)),('coverage_below20',(1,.9,.199999,.03)),('proxy_below3',(1,.9,.2,.029999))]:check(name,lambda args=args:eq(m.warm_gate(*args),False))
    for nt,anchor in ((48000,33),(96000,512),(48000,1024)):
        tokens=[1]*nt;tokens[anchor-1]=128803;meta=dict(actual_tokens=nt,cached_tokens=0,prefill_tokens=nt-1);intervals=m.expected_intervals(meta,struct.pack('<'+'i'*nt,*tokens));check('source_anchor_split_'+str(anchor),lambda intervals=intervals,anchor=anchor:eq(sorted({x[1]for x in intervals}),[0,anchor]))
    check('duplicate_json_key',lambda:m.parse_json('{"a":1,"a":2}'),True)
    with tempfile.TemporaryDirectory(prefix='myds-indexscore-reader-cpu-')as tmp:
        b,args,original=fixture(Path(tmp).resolve());r=m.review(args);check('full_four_sequence_positive_and_cold_zero_disclosed',lambda:eq((r['evidence_valid'],r['roi_screen_pass'],r['performance_gate_pass'],r['production_adoption'],r['scenarios'][0]['phases'][0]['strict_tile_coverage']),(True,True,False,False,0.0)))
        # Freeze invalidity vs valid negative screens distinctly.
        p=Path(b['runs'][1]['ordinary_dir'])/'survey.json';v=m.load(p);saved=copy.deepcopy(v);v['phases'][1]['prefill_wall_s']=100;write(p,v);r=m.review(args);check('valid_negative_proxy_rc0_semantics',lambda:eq((r['evidence_valid'],r['roi_screen_pass'],r['candidate_implementation_authorized']),(True,False,False)));write(p,saved)
        p=Path(b['runs'][0]['diagnostic_dir'])/'survey.json';saved=m.load(p);eligible=next(i for i,r in enumerate(saved['records'])if r['layer']==24)
        def rewrite(v):
            records=p.parent/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n'for r in v['records']));v['records_sha256']=m.sha(records);write(p,v)
        bad=copy.deepcopy(saved);bad['records'][eligible]['g_start']+=1;rewrite(bad);check('raw_offset_geometry_not_double_shifted',lambda:m.review(args),True);rewrite(saved)
        bad=copy.deepcopy(saved);bad['records'][eligible]['raw_mask']['sha256']='0'*64;rewrite(bad);check('full_raw_payload_sha',lambda:m.review(args),True);rewrite(saved)
        bad=copy.deepcopy(saved);bad['records'][eligible]['selected_kernel']='direct';rewrite(bad);check('actual_provider_not_env_guess',lambda:m.review(args),True);rewrite(saved)
        bad=copy.deepcopy(saved);bad['records'][0]['input_skip']=0;rewrite(bad);check('original_chunk_peel_metadata_required',lambda:m.review(args),True);rewrite(saved)
        # Tamper one consumer payload and update every local SHA to require
        # actual producer/consumer row equality, not only descriptor hashes.
        rawrec=saved['records'][eligible]['raw_mask'];payload=p.parent/rawrec['path'];raw0=payload.read_bytes();payload.write_bytes(bytes([raw0[0]^1])+raw0[1:]);bad=copy.deepcopy(saved);bad['records'][eligible]['raw_mask']['sha256']=m.sha(payload);rewrite(bad);check('producer_consumer_full_row_bytes_detect_stale_mask',lambda:m.review(args),True);payload.write_bytes(raw0);rewrite(saved)
        bad=copy.deepcopy(saved);bad['records'][eligible]['glen']-=1;rewrite(bad);check('missing_query_domain_rejected',lambda:m.review(args),True);rewrite(saved)
        bad=copy.deepcopy(saved);bad['ordinary_uninstrumented_core']=False;rewrite(bad);check('native_integer_scope_type_not_bool',lambda:m.review(args),True);rewrite(saved)
        bad=copy.deepcopy(saved);bad['phases'][0]['dsp_present']=False;rewrite(bad);check('native_integer_dsp_type_not_bool',lambda:m.review(args),True);rewrite(saved)
        core=Path(b['binaries']['ordinary']['path']).parent.parent/'src/cpu-fixture-0.c';core.write_bytes(b'wrong');check('frozen_source_actual_bytes',lambda:m.review(args),True)
        m.sha=original
    report=dict(schema_version=1,scope='CPU fixtures only; no actual model, GPU, service or measured performance',reader_sha256=m.sha(P),test_sha256=m.sha(__file__),passed=True,control_count=len(results),independent_cell_oracle_cases=count,controls=results)
    write(a.out,report);print(json.dumps({k:report[k]for k in('passed','control_count','independent_cell_oracle_cases','reader_sha256')}))
if __name__=='__main__':main()
