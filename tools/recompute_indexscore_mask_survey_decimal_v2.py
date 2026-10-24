#!/usr/bin/env python3
"""New read-only independent replay and Decimal/byte-LUT recomputation.

This script never imports the survey reader and never runs native/model/Metal
programs. Its sole child is the frozen Python CPU reader on existing raw files.
"""
import argparse,hashlib,json,math,subprocess,sys,time
from decimal import Decimal,localcontext
from pathlib import Path
sys.dont_write_bytecode=True
LAYERS=(2,8,14,20,24,28,32,36);ELIGIBLE=(24,28,32,36)
OLD='6c85a5879ada3d86a4a71667fb85277987043ec8345e34bdaa113080eb1c33bb'
PORTABLE='f0edbe2166a7e8f4a1db8d926fa8d0ac084e233b6f2f9d54a5c39ea5d497b042'
BIND='e97b87285f2d962b549a01ba2ad4f8befd8d07692d507901f663fc9895543387'
PLAN='157a56fd05c337a445f3116d7f11bed40e57b691cd1083985cd36701a9185ebc'
ONES_LUT=tuple(bin(v).count('1')for v in range(256))
PAIR_LUT=tuple(sum(bool((v>>(2*i))&3)for i in range(4))for v in range(256))

def need(v,msg):
    if not v:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()
def parse(raw,decimal=True):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'Duplicate key '+k);d[k]=v
        return d
    return json.loads(raw,parse_float=(lambda value:Decimal.from_float(float(value))) if decimal else float,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def load(p,decimal=True):return parse(Path(p).read_bytes(),decimal)
def dump(p,v):
    def conv(x):
        if isinstance(x,Decimal):return str(x)
        raise TypeError(type(x).__name__)
    with Path(p).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,default=conv,allow_nan=False);f.write('\n')
def desc(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def verified(r):
    p=Path(r['path']);need(p.is_absolute()and p.is_file()and not p.is_symlink()and sha(p)==r['sha256'],'Original descriptor SHA/path')
    if 'bytes'in r:need(type(r['bytes'])is int and p.stat().st_size==r['bytes'],'Original descriptor bytes')
    return p
def near(actual,reported,label):
    actual=Decimal(actual);reported=Decimal(reported);error=abs(actual-reported);limit=max(Decimal('1e-12'),abs(reported)*Decimal('1e-12'))
    need(error<=limit,'Decimal arithmetic differs: '+label);return error
def payload(directory,r):
    rel=Path(r['path']);need(not rel.is_absolute()and '..'not in rel.parts,'Bounded raw descriptor');p=directory/rel
    need(p.is_file()and not p.is_symlink()and p.stat().st_size==r['bytes']and sha(p)==r['sha256'],'Complete raw SHA/extent');return p.read_bytes()

def pair_nonzero(flags,n):
    packed=flags.to_bytes(max((flags.bit_length()+7)//8,(n*2+7)//8),'little');whole,tail=divmod(n,4)
    count=sum(PAIR_LUT[v]for v in packed[:whole])
    if tail:count+=PAIR_LUT[packed[whole]&((1<<(tail*2))-1)]
    return count

def low_ones(flags,n):
    packed=flags.to_bytes(max((flags.bit_length()+7)//8,(n+7)//8),'little');whole,tail=divmod(n,8)
    count=sum(ONES_LUT[v]for v in packed[:whole])
    if tail:count+=ONES_LUT[packed[whole]&((1<<tail)-1)]
    return count

def strict_counts(raw,nc,glen,start,ratio,bits,stride):
    """Independent byte LUT counts16-key groups and actual visible cells."""
    need(len(raw)==glen*stride*4,'Original complete mask');blocks=(nc+7)//8;rowbytes=stride*4;need(type(bits)is int and bits in(0,1),'Actual mask format')
    rows=[];valid=allowed=0
    for t in range(glen):
        row=raw[t*rowbytes:(t+1)*rowbytes]
        if bits:flags=int.from_bytes(row,'little')
        else:
            import struct
            flags=0
            for b in range(blocks):
                f=struct.unpack_from('<f',row,b*4)[0];need(f==0.0 or f==-math.inf,'Float mask0/-INF')
                if f==0.0:flags|=1<<b
        rows.append(flags);vis=min(nc,(start+t+1)//ratio);n,rem=divmod(vis,8)
        valid+=vis;allowed+=8*low_ones(flags,n)+(rem if(flags>>n)&1 else 0)
    total=((glen+7)//8)*((nc+15)//16);active=full=skip=allskip=causalextra=qtail=ktail=activecells=0
    for first in range(0,glen,8):
        end=min(first+8,glen);visible=min(nc,(start+end)//ratio);tiles=(visible+15)//16;active+=tiles;activecells+=min(nc,tiles*16)*(end-first)
        union=effective=0
        for t in range(first,end):
            union|=rows[t];vis=min(nc,(start+t+1)//ratio);effective|=rows[t]&((1<<((vis+7)//8))-1)
        masked=tiles-pair_nonzero(effective,tiles);allskip+=masked
        if end-first!=8:qtail+=masked;continue
        n=min(tiles,nc//16);full+=n
        strict=n-pair_nonzero(union,n);complete_visible=n-pair_nonzero(effective,n)
        skip+=strict;causalextra+=complete_visible-strict;ktail+=masked-complete_visible
    need(allskip==skip+causalextra+qtail+ktail,'Disjoint strict/causal-boundary/query-tail/key-tail accounting')
    native=dict(total_tiles=total,causal_already_skipped_tiles=total-active,causal_active_tiles=active,new_mask_skippable_tiles=allskip,valid_cells=valid,allowed_valid_cells=allowed,rejected_valid_cells=valid-allowed,causal_cells_in_active_tiles=activecells-valid)
    return dict(total_dispatched_tiles=total,causal_already_skipped_tiles=total-active,noncausal_dispatched_tiles=active,full8x16_noncausal_tiles=full,strict_full_mask_skip_tiles=skip),native,dict(full8query_full16key_causal_boundary_extra_allmasked_tiles=causalextra,query_tail_allmasked_tiles_including_key_tail=qtail,full8query_key_tail_allmasked_tiles=ktail)

def recompute(binding,report):
    result=[];record_total=0;all_errors=[]
    for run,reported_run in zip(binding['runs'],report['scenarios']):
        need(run['scenario_id']==reported_run['scenario_id'],'Original scenario order');directory=Path(run['diagnostic_dir']);v=load(directory/'survey.json');ordinary=load(Path(run['ordinary_dir'])/'survey.json')
        lines=directory/'records.jsonl';records=[parse(x)for x in lines.read_bytes().splitlines()if x.strip()];need(v['records']==records and sha(lines)==v['records_sha256'] and len(records)==v['record_count'],'All original record bytes/JSON identity');record_total+=len(records)
        phases=[]
        for phase in (0,1):
            pm=v['phases'][phase];om=ordinary['phases'][phase];rp=reported_run['phases'][phase];group=records[pm['first_record']:pm['first_record']+pm['record_count']];need(all(r['phase']==phase for r in group),'Original phase partition')
            score={l:Decimal(0)for l in LAYERS};filter_time={l:Decimal(0)for l in LAYERS};producer_time=Decimal(0);eligible=Decimal(0);weighted=Decimal(0);active=skip=allskip=0;weighted_all=Decimal(0);extra_sums={};checks=[]
            expected={d['case']:d for d in rp['eligible_dispatches']};need(len(expected)==len(rp['eligible_dispatches']),'Unique reported eligible cases')
            for r in group:
                cb=r['score_gpu_cb'];dt=cb['end_s']-cb['start_s'];need(dt>0,'Actual positive original score CB');near(dt*1000,cb['span_ms'],'source span_ms');score[r['layer']]+=dt
                tail=r['filter_topk_gpu_cb'];filter_time[r['layer']]+=tail['end_s']-tail['start_s']
                if r['layer']==20:
                    prod=r['mask_producer_gpu_cb'];producer_time+=prod['end_s']-prod['start_s']
                if r['layer']not in ELIGIBLE:continue
                raw=payload(directory,r['raw_mask']);counts,native,extras=strict_counts(raw,r['nc'],r['glen'],r['g_start'],r['ratio'],r['mask_bits'],r['mask_stride_words']);need(native==r['coverage'],'Independent all-actual-visible-cells/native advisory counts differ');prior=expected.pop(r['case']);need(all(counts[k]==prior[k]for k in counts),'Independent byte-LUT strict count differs at case'+str(r['case']));coverage=Decimal(counts['strict_full_mask_skip_tiles'])/counts['noncausal_dispatched_tiles'];weighted_one=dt*coverage
                all_errors.append(near(dt,prior['isolated_score_gpu_s'],'per-dispatch score'));all_errors.append(near(coverage,prior['strict_tile_coverage'],'per-dispatch coverage'))
                eligible+=dt;weighted+=weighted_one;active+=counts['noncausal_dispatched_tiles'];skip+=counts['strict_full_mask_skip_tiles'];allskip+=native['new_mask_skippable_tiles'];weighted_all+=dt*Decimal(native['new_mask_skippable_tiles'])/counts['noncausal_dispatched_tiles']
                for key,value in extras.items():extra_sums[key]=extra_sums.get(key,0)+value
                checks.append(dict(case=r['case'],layer=r['layer'],score_gpu_s=dt,strict_coverage=coverage,weighted_opportunity_s=weighted_one,native_advisory_independently_equal=native,partial_and_causal_extra=extras,**counts))
            need(not expected,'Every reported eligible dispatch independently recomputed');need(set(r['layer']for r in group)==set(LAYERS),'All eight sources disclosed')
            coverage=Decimal(skip)/active;wall=om['prefill_wall_s'];proxy=weighted/wall;reuse=Decimal(pm['cached_tokens'])/v['phases'][0]['actual_tokens'];passed=phase==1 and reuse>=Decimal('.9')and coverage>=Decimal('.2')and proxy>=Decimal('.03')
            need(skip==rp['strict_skip_tiles']and active==rp['noncausal_dispatched_tiles']and passed==rp['warm_screen_pass'],'Phase count/screen differs')
            for l,dt in score.items():all_errors.append(near(dt,rp['all_eight_layer_score_gpu_s'][str(l)],'all-eight layer time'))
            for actual,key in((coverage,'strict_tile_coverage'),(weighted,'isolated_eligible_opportunity_s'),(wall,'ordinary_prefill_wall_s'),(proxy,'diagnostic_opportunity_fraction'),(reuse,'cached_fraction_of_setup')):all_errors.append(near(actual,rp[key],'phase '+key))
            phases.append(dict(phase=phase,role='cold'if phase==0 else'warm_append',all_source_records=len(group),actual_tokens=pm['actual_tokens'],cached_tokens=pm['cached_tokens'],prefill_tokens=pm['prefill_tokens'],all_eight_layer_score_gpu_s={str(k):x for k,x in score.items()},all_eight_layer_filter_topk_gpu_s={str(k):x for k,x in filter_time.items()},layer20_mask_producer_gpu_s=producer_time,all_eight_score_gpu_s=sum(score.values()),eligible_four_score_gpu_s=eligible,strict_skip_tiles=skip,noncausal_dispatched_tiles=active,strict_tile_coverage=coverage,isolated_eligible_opportunity_s=weighted,ordinary_prefill_wall_s=wall,diagnostic_opportunity_fraction=proxy,cached_fraction_of_setup=reuse,warm_screen_pass=passed,all_actual_visible_allmasked_tiles=allskip,all_actual_tile_coverage=Decimal(allskip)/active,all_actual_visible_allmasked_opportunity_s=weighted_all,all_actual_visible_allmasked_opportunity_fraction=weighted_all/wall,partial_and_causal_extra=extra_sums,supplemental_all_tile_metric_changes_strict_gate=False,independent_dispatches=checks))
        result.append(dict(scenario_id=run['scenario_id'],record_count=len(records),phases=phases))
    passed=all(r['phases'][1]['warm_screen_pass']for r in result);need(record_total==6153,'All actual6153 records required');need(passed==report['roi_screen_pass']and report['evidence_valid']is True and report['performance_gate_pass']is False and report['production_adoption']is False,'Original valid-negative scope')
    return dict(schema_version=1,evidence_valid=True,scope='Independent original raw byte-LUT strict counting and Decimal arithmetic; CPU-only',decimal_precision=60,input_float_representation='Exact binary64 recovered by JSON roundtrip then Decimal.from_float; all arithmetic below is Decimal',comparison_absolute_relative_tolerance='max(1e-12,abs(reported)*1e-12)',max_decimal_vs_reported_float_error=max(all_errors),all_records_reverified=record_total,roi_screen_pass=passed,candidate_implementation_authorized=passed,performance_gate_pass=False,production_adoption=False,scenarios=result)

def run(a):
    need(not a.out.exists(),'New independent output only');a.out.mkdir(parents=True);receipt=dict(schema_version=1,evidence_valid=False,no_gpu_or_native_rerun=True,reader_imported=False,processor_sha256=sha(__file__))
    try:
        d=a.root;reader=d/'portable-reader-v2/review_indexscore_mask_survey.py';binding=d/'replay-v2/binding.json';plan=d/'replay-v2/reader-replay-plan.json';original_plan=d/'frozen-survey-plan-v1.json';original_binding=d/'survey-v1/binding.json';prior=d/'replay-v2/review.json'
        need(sha(reader)==PORTABLE and sha(binding)==BIND and sha(plan)==PLAN,'Explicit frozen v2 identities');b=load(binding);np=load(plan);op=load(original_plan);ob=load(original_binding)
        need(set(np)==set(op)and np['reader']!=op['reader']and {k:v for k,v in np.items()if k!='reader'}=={k:v for k,v in op.items()if k!='reader'},'Only plan reader descriptor changed')
        need(np['reader']['sha256']==PORTABLE and op['reader']['sha256']==OLD and verified(np['reader']).resolve()==reader.resolve(),'Original/new reader exact identity');verified(op['reader']);need(set(b)==set(ob)and b['plan']!=ob['plan']and {k:v for k,v in b.items()if k!='plan'}=={k:v for k,v in ob.items()if k!='plan'},'Only binding plan descriptor changed');need(verified(b['plan']).resolve()==plan.resolve(),'Actual new plan bound');need(verified(ob['plan']).resolve()==original_plan.resolve(),'Original GPU plan preserved')
        old_failure=load(d/'survey-v1/review.json');window=load(d/'survey-v1/summary.json');need(old_failure['evidence_valid']is False and window['status']=='failed','Original failed window/report remain failed')
        receipt.update(source_plan_only_reader_changed=True,binding_only_plan_changed=True,original_plan=desc(original_plan),original_binding=desc(original_binding),original_failed_review=desc(d/'survey-v1/review.json'),original_failed_window_summary=desc(d/'survey-v1/summary.json'),reader=desc(reader),binding=desc(binding),replay_plan=desc(plan),prior_report=desc(prior))
        argv=['/usr/bin/python3',str(reader),'--binding',str(binding),'--binding-sha256',BIND,'--model',str(a.model),'--out',str(a.out/'review.json')];t=time.monotonic()
        with(a.out/'reader.stdout').open('xb')as stdout,(a.out/'reader.stderr').open('xb')as stderr:cp=subprocess.run(argv,stdout=stdout,stderr=stderr,check=False)
        receipt.update(reader_subprocess=dict(argv=argv,returncode=cp.returncode,wall_s_cpu_diagnostic_only=time.monotonic()-t));need(cp.returncode==0,'Frozen CPU reader returned failure');need(sha(a.out/'review.json')==sha(prior)and(a.out/'review.json').read_bytes()==prior.read_bytes(),'Full independent report must be byte/SHA identical to prior replay')
        receipt.update(independent_report=desc(a.out/'review.json'),report_full_bytes_equal_prior=True)
        report=load(a.out/'review.json');with_decimal=None
        with localcontext()as context:context.prec=60;value=recompute(b,report)
        value.update(independent_full_report=desc(a.out/'review.json'),prior_report=desc(prior),binding=desc(binding),processor_sha256=sha(__file__));dump(a.out/'decimal-recompute.json',value)
        receipt.update(evidence_valid=True,independent_report=desc(a.out/'review.json'),report_full_bytes_equal_prior=True,all_records_reverified=value['all_records_reverified'],decimal_recompute=desc(a.out/'decimal-recompute.json'),roi_screen_pass=value['roi_screen_pass'],candidate_implementation_authorized=value['candidate_implementation_authorized'],performance_gate_pass=False,production_adoption=False)
    except BaseException as exc:receipt['error']=repr(exc)
    dump(a.out/'independent-receipt.json',receipt);print(json.dumps({k:receipt.get(k)for k in('evidence_valid','all_records_reverified','report_full_bytes_equal_prior','roi_screen_pass','error')},sort_keys=True));return 0 if receipt['evidence_valid']else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(all(x.is_absolute()for x in(a.root,a.model,a.out)),'Absolute paths required');return run(a)
if __name__=='__main__':raise SystemExit(main())
