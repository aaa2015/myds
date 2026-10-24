#!/usr/bin/env python3
"""Synthetic localgrow layout and mocked controller controls; CPU only."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def module(name):
    p=Path(__file__).with_name(name+'.py');s=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    r=module('review_moe_workspace_localgrow_artifacts');c=module('run_moe_workspace_localgrow_component_window');checks=[]
    def test(name,fn,accept=True):
        error=None
        try:fn();actual=True
        except Exception as exc:actual=False;error=repr(exc)
        checks.append(dict(case=name,expected_accept=accept,accepted=actual,passed=actual==accept,error=error))
    def require(v,message='CPU fixture assertion'):
        if not v:raise AssertionError(message)
    def fixture(C,B,epoch=0):
        bindings=[]
        def entry(name,scope,kind,wrapper,resource,size,offset=0,index=4294967295,view=0):
            b=dict(name=name,scope=scope,kind=kind,index=index,wrapper_id=wrapper,resource_id=resource,bytes=size,offset=offset,is_view=view);bindings.append(b);return b
        names=('moe_norm_acc','moe_ids_acc','moe_w_acc','moe_shared_acc','moe_after_acc','moe_split_acc','moe_out_acc')
        strides=(20480,24,24,20480,81920,96,20480)
        for i,stride in enumerate(strides):entry('stable-'+str(i),1,1,10+i,110+i,C*stride)
        base=1000+epoch*100
        down=entry('moe_down_buf',3,1,base,base+10000,B*6*5120*4)
        acc=[]
        for i,(name,stride) in enumerate(zip(names,strides)):
            acc.append(entry(name,3,3 if B==C else 1,10+i if B==C else base+1+i,110+i if B==C else base+10001+i,B*stride))
        for i in range(C):entry('x['+str(i)+']',2,2,20000+i,110,20480,i*20480,i,1)
        for i in range(C):entry('moe_down['+str(i)+']',4,2,30000+epoch*100+i,down['resource_id'],6*5120*4,i*6*5120*4,i,1)
        entry('moe_view',5,2,50000,0,0,view=1)
        entry('arena.mm_mid',6,4,base+50,base+10050,B*6*2304*2)
        for name in ('arena.fallback_gate','arena.fallback_up','arena.fallback_mid'):entry(name,6,4,0,0,0)
        owners=[b for b in bindings if b['scope']==1];views=[b for b in bindings if b['scope'] in (2,5)]
        return dict(implementation_version=3,capacity=C,moe_batch=B,compact=1,cw_id=99,moe_generation=1+epoch,
          localgrow_attempts=epoch,localgrow_successes=epoch,localgrow_failures=0,
          stable_owner_fingerprint=r.id_hash([v for b in owners for v in (b['wrapper_id'],b['resource_id'],b['bytes'])]),
          stable_view_fingerprint=r.id_hash([b['wrapper_id'] for b in views]),stable_owner_bytes=sum(b['bytes'] for b in owners),
          stable_owners=len(owners),stable_views=len(views),down_id=down['wrapper_id'],down_bytes=down['bytes'],
          acc_ids=[b['wrapper_id'] for b in acc],acc_bytes=[b['bytes'] for b in acc],arena_id=60000+epoch,
          binding_count=len(bindings),bindings_complete=True,bindings=bindings)
    before=fixture(32,64);after=fixture(32,128,1)
    test('complete_actual_binding_shape_and_sameCW_growth',lambda:r.layout_transition(before,after,'grow'))
    test('alias_base_to_seven_owned_accumulators',lambda:r.layout_transition(fixture(32,32),fixture(32,64,1),'grow'))
    failed=copy.deepcopy(before);failed.update(localgrow_attempts=1,localgrow_failures=1)
    test('failure_preserves_every_pool_binding_and_payload_extent',lambda:r.layout_transition(before,failed,'failure'))
    test('short_retain_is_exact_layout_no_new_generation',lambda:r.layout_transition(after,copy.deepcopy(after),'retain'))
    mutations=[('stable_owner_replaced',lambda q:q['bindings'][0].update(resource_id=999)),
      ('one_down_view_wrong_offset',lambda q:next(b for b in q['bindings'] if b['scope']==4).update(offset=4)),
      ('one_down_view_old_resource',lambda q:next(b for b in q['bindings'] if b['scope']==4).update(resource_id=before['bindings'][7]['resource_id'])),
      ('CW_rebuilt_instead_of_localgrow',lambda q:q.update(cw_id=100)),
      ('missing_complete_binding',lambda q:q['bindings'].pop()),
      ('arena_not_replaced',lambda q:q.update(arena_id=before['arena_id'])),
      ('generation_not_advanced',lambda q:q.update(moe_generation=1)),
      ('grow_counter_not_advanced',lambda q:q.update(localgrow_successes=0))]
    for name,mutate in mutations:
        value=copy.deepcopy(after);mutate(value);test(name,lambda value=value:r.layout_transition(before,value,'grow'),False)
    value=copy.deepcopy(failed);value['acc_ids'][0]=999
    test('failed_transaction_partial_publication_rejected',lambda:r.layout_transition(before,value,'failure'),False)
    value=copy.deepcopy(after);value['bindings'][-1].update(wrapper_id=123,resource_id=before['bindings'][7]['resource_id'],bytes=4)
    test('retained_old_pool_reference_rejected',lambda:r.layout_transition(before,value,'grow'),False)
    with tempfile.TemporaryDirectory(prefix='moews-localgrow-CPU-only-') as tmp:
        base=Path(tmp);stage=base/'stage';(stage/'bin').mkdir(parents=True);(stage/'tools').mkdir()
        input_file=base/'frozen';input_file.write_text('CPU input never executed')
        env={'MYDS_CPU_FIXTURE':'1','MYDS_METAL_CONCURRENT':'1'};counter=[0]
        def exercise(kind='success',prepare=False):
            counter[0]+=1;out=base/str(counter[0]);out.mkdir();calls=[];ops=[]
            a=SimpleNamespace(stage=stage,gold_stage=base/'gold',out=out,model=base/'fake-model',inputs=base/'inputs',
              plan=base/'plan',port=8187,prepare=prepare,run=not prepare)
            def preflight(self):
                self.before={'pid':123456789};self.environment=dict(env);self.plan={'safety':{}}
                self.input_hashes={str(input_file):c.sha(input_file)};self.summary['model']={'CPU_mock':True}
            def stop(self):self.stopped=True;ops.append('stop_mock')
            def restore(self):
                ops.append('restore_mock')
                if kind=='restore-failure':raise ValueError('CPU restore failure')
                return {'restored':True,'pid':222}
            def unchanged(self):ops.append('unchanged_mock')
            def monitor(self):self.done.wait(10)
            def command(self,name,argv,env=None,timeout=None):
                calls.append(name);self.steps.append({'name':name,'returncode':0});c.write_json(out/'steps.json',self.steps)
                if name=='localgrow' and kind=='native-failure':raise ValueError('CPU native failure')
                directory=out/name
                if name in ('allocations','correctness'):directory.mkdir()
                if name=='allocations':(directory/'records.jsonl').write_text('{"all_pass":true,"implementation_version":3}\n')
                if name=='correctness':c.write_json(directory/'artifact-manifest.json',{'full_f32_bit_exact':True})
                if name=='failure-review':c.write_json(out/'failure-review.json',{'evidence_valid':True,'functional_failure_gate_pass':True})
                if name=='localgrow-review':c.write_json(out/'localgrow-review.json',{'evidence_valid':kind!='reader-negative','functional_localgrow_gate_pass':kind!='reader-negative'})
                if kind=='input-mutation':input_file.write_text('changed CPU frozen input')
            worker=c.LocalgrowComponentWindow(a);original=input_file.read_bytes()
            try:
                with patch.object(c.LocalgrowComponentWindow,'preflight',preflight),patch.object(c.LocalgrowComponentWindow,'stop',stop), \
                  patch.object(c.LocalgrowComponentWindow,'restore',restore),patch.object(c.LocalgrowComponentWindow,'unchanged',unchanged), \
                  patch.object(c.LocalgrowComponentWindow,'monitor',monitor),patch.object(c.LocalgrowComponentWindow,'run_command',command), \
                  patch.object(c.workspace.fcntl,'flock',lambda *a:None):code=worker.run()
            finally:input_file.write_bytes(original)
            return code,worker,calls,ops
        def positive():
            code,w,calls,ops=exercise();require(code==0 and calls==['allocations','capture','host','correctness','failures','failure-review','localgrow','localgrow-review','performance'] and ops==['stop_mock','restore_mock'])
            require(w.summary['localgrow_growth_time_is_not_gpu_execution_time'] is True)
        test('mock_all_nine_steps_functional_gate_before90CB_finally_restore',positive)
        def prepared():
            code,w,calls,ops=exercise(prepare=True);require(code==0 and not calls and ops==['unchanged_mock'])
        test('prepare_no_native_GPU_or_service_stop',prepared)
        for kind in ('native-failure','reader-negative','input-mutation','restore-failure'):
            def reject(kind=kind):
                code,w,calls,ops=exercise(kind);require(code==1 and ops==['stop_mock','restore_mock'])
                if kind in ('native-failure','reader-negative'):require('performance' not in calls)
            test(kind+'_reject_and_finally_restore',reject)
    result=dict(schema_version=1,scope='Synthetic CPU layout/ownership and mocked controller only',
      all_controls_pass=all(c['passed'] for c in checks),controls=checks,model_or_GPU_executed=False,service_operations=False,
      reader_sha256=r.sha(Path(r.__file__)),controller_sha256=r.sha(Path(c.__file__)))
    with args.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'all_controls_pass':result['all_controls_pass'],'controls':len(checks),'failed':[c['case'] for c in checks if not c['passed']]}));return int(not result['all_controls_pass'])


if __name__=='__main__':raise SystemExit(main())
