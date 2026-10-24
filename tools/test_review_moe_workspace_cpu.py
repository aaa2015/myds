#!/usr/bin/env python3
"""Synthetic CPU false-acceptance controls for the MoE component reader."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).with_name('review_moe_workspace_artifacts.py');spec=importlib.util.spec_from_file_location('moe_component_CPU_reader',source)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);checks=[]
    def check(name,function,accept):
        error=None
        try:function();accepted=True
        except (ValueError,KeyError,FileNotFoundError) as e:accepted=False;error=str(e)
        checks.append(dict(case=name,expected='accept' if accept else 'reject',accepted=accepted,passed=accepted==accept,error=error))
    with tempfile.TemporaryDirectory(prefix='moe-reader-CPU-fixture-') as tmp:
        root=Path(tmp).resolve();observed={}
        for layer in r.LAYERS:
            for i,(n,mm) in enumerate(((16384,True),(17,False)) if layer!=39 else ((1,False),)):
                observed[(layer,i,n,False)]=dict(layer=layer,rows=n,actual_mm=mm)
        def stats(layer,compact):
            actual=[c for c in observed.values() if c['layer']==layer];mm=sum(c['actual_mm'] for c in actual);mv=len(actual)-mm
            mid=16384*6*2304*2;f32=32*6*2304*4
            return dict(capacity=2048,moe_batch=16384,compact=compact,requests=len(actual),mm_encoded=mm,mv_encoded=mv,failures=0,
                legacy_gate_bytes=0 if compact else 16384*6*2304*4,legacy_up_bytes=0 if compact else 16384*6*2304*4,
                legacy_mid_bytes=0 if compact else 16384*6*2304*4,
                scratch=dict(requests=len(actual),mm_encoded=mm,mv_encoded=mv,failures=0,retired_bytes=0,retired_owners=0,
                    mm_mid_bytes=mid,f32_gate_bytes=f32,f32_up_bytes=f32,f32_mid_bytes=f32,allocated_bytes=mid+f32*3))
        plan=dict(component=dict(warmup=1,formal_samples=9,order=['A1','B','A2'],maximum_absolute_time_control_drift_percent=2,
            maximum_conservative_gpu_time_regression_percent=2,minimum_actual_device_allocation_reduction_bytes=536870912))
        def dataset(btime=1010.,a2time=1000.,saving=1073741824):
            data=[];index=0
            for layer in r.LAYERS:
                groups=[[],[],[]]
                for sample in range(10):
                    for arm_index,arm in enumerate(('A1','B','A2')):
                        us=(1000.,btime,a2time)[arm_index];start=100+index*.01;end=start+us/1e6;index+=1
                        compact=arm=='B';metal=12*1073741824-(saving if compact else 0)
                        m=dict(metal_current_allocated_size=metal,process_phys_footprint=6*1073741824-(536870912 if compact else 0),
                            process_resident_size=7*1073741824,workspace=stats(layer,compact))
                        row=dict(kind='timing',layer=layer,sample=sample,arm=arm,warmup=sample==0,gpu_start_s=start,gpu_end_s=end,
                            gpu_us=(end-start)*1e6,completed_cb=True,command_buffers=1,full_f32_bit_exact=True,finite_guards_written=True,
                            actual_cases_per_cb=sum(c['layer']==layer for c in observed.values()),factory_wall_us=50,host_encode_us=80,host_wait_us=90,
                            before_create=copy.deepcopy(m),after_create=copy.deepcopy(m),after_encode=copy.deepcopy(m),after_completed=copy.deepcopy(m),after_free=copy.deepcopy(m))
                        data.append(row)
                        if sample:groups[arm_index].append(row['gpu_us'])
                times=[r.statistics.median(g) for g in groups]
                data.append(dict(kind='summary',layer=layer,a1_us=times[0],b_us=times[1],a2_us=times[2],
                    control_drift_percent=(times[2]/times[0]-1)*100,
                    conservative_time_saved_percent=(1-times[1]/min(times[0],times[2]))*100))
            data.append(dict(kind='complete',timing_command_buffers=90));return data
        def write(data):(root/'performance.stdout').write_text(''.join(json.dumps(v)+'\n' for v in data))
        def verify():return r.performance(root,observed,plan)
        valid=dataset();write(valid)
        check('complete_90_CB_memory_gain_with_1pct_time_regression_positive',lambda:r.need(all(g['gate_pass'] for g in verify()),'Valid memory tradeoff rejected'),True)
        mutations=[('missing_CB',lambda d:d.pop(0)),('duplicate_CB',lambda d:d.__setitem__(1,copy.deepcopy(d[0]))),
            ('interleaving_changed',lambda d:d.__setitem__(slice(0,2),list(reversed(d[:2])))),
            ('zero_GPU_span',lambda d:d[0].update(gpu_end_s=d[0]['gpu_start_s'])),
            ('wrong_GPU_normalization',lambda d:d[0].update(gpu_us=d[0]['gpu_us']+1)),
            ('two_commits_disguised_as_one_CB',lambda d:d[0].update(command_buffers=2)),
            ('omitted_actual_tail',lambda d:d[0].update(actual_cases_per_cb=1)),
            ('missing_full_parity',lambda d:d[0].update(full_f32_bit_exact=False)),
            ('missing_observed_footprint',lambda d:d[0]['after_completed'].update(process_phys_footprint=0)),
            ('compact_retained_legacy_F32_pool',lambda d:d[1]['after_completed']['workspace'].update(legacy_mid_bytes=905969664)),
            ('native_factory_pool_manufactured_smaller',lambda d:d[0]['after_completed']['workspace'].update(legacy_mid_bytes=4096)),
            ('partial_actual_MM_adoption',lambda d:d[1]['after_completed']['workspace'].update(mm_encoded=0)),
            ('actual_provider_failure',lambda d:d[1]['after_completed']['workspace'].update(failures=1)),
            ('scratch_byte_accounting_fabricated',lambda d:d[1]['after_completed']['workspace']['scratch'].update(allocated_bytes=1)),
            ('published_summary_fabricated',lambda d:next(v for v in d if v['kind']=='summary').update(b_us=1))]
        for name,mutate in mutations:
            data=copy.deepcopy(valid);mutate(data);write(data);check(name,verify,False)
        for name,data in [('GPU_regression_above_2pct',dataset(btime=1030)),('control_drift_above_2pct',dataset(a2time=1030)),
                          ('actual_device_saving_only_256MiB',dataset(saving=268435456)),('requested_bytes_gain_but_no_actual_device_gain',dataset(saving=0))]:
            write(data);check(name,lambda:r.need(not any(g['gate_pass'] for g in verify()),'Numerical rejection incorrectly passed'),True)
        # The frozen native logger writes uncolored DEBUG/INFO lines to stdout.
        write(valid);original=(root/'performance.stdout').read_bytes()
        native_log=b'[2026-10-24 01:03:30.390] [INFO] myds_metal: initialized native GPU path\n'
        (root/'performance.stdout').write_bytes(native_log+original)
        evidence=[]
        check('native_logger_prefix_preserved_positive',lambda:r.need(all(g['gate_pass'] for g in r.performance(root,observed,plan,evidence)) and
             evidence[0]['native_log_lines']==1 and evidence[0]['native_log_bytes']==len(native_log) and
             evidence[0]['native_log_raw_sha256']==r.hashlib.sha256(native_log).hexdigest() and
             evidence[0]['stdout_sha256']==r.sha(root/'performance.stdout'),'Exact native log identity differs'),True)
        for name,prefix in (
            ('arbitrary_stdout_discard_forbidden',b'silent unexplained text\n'),
            ('malformed_logger_timestamp_rejected',b'[2026-10-24 01:03:30] [INFO] message\n'),
            ('impossible_logger_calendar_rejected',b'[2026-02-30 01:03:30.390] [INFO] message\n'),
            ('stderr_severity_cannot_appear_as_stdout',b'[2026-10-24 01:03:30.390] [WARN] message\n'),
            ('ANSI_not_in_frozen_stdout_grammar',b'[2026-10-24 01:03:30.390] [\x1b[32mINFO\x1b[0m] message\n'),
            ('malformed_JSON_cannot_be_ignored',b'{not JSON}\n')):
            (root/'performance.stdout').write_bytes(prefix+original);check(name,verify,False)
        (root/'strict-records.jsonl').write_bytes(native_log+b'{"kind":"record"}\n')
        check('capture_JSONL_cannot_hide_native_log',lambda:r.rows(root/'strict-records.jsonl'),False)
        # Delayed task ledger updates must remain visible and must not be used
        # as a substitute for the frozen actual-device allocation gate.
        deferred=copy.deepcopy(valid)
        for v in deferred:
            if v.get('kind')=='timing' and v['arm']=='B':
                v['before_create']['process_phys_footprint']+=2147483648
                v['after_completed']['process_resident_size']=v['before_create']['process_resident_size']
        write(deferred)
        def deferred_check():
            groups=verify();r.need(all(g['gate_pass'] for g in groups),'Physical ledger changed device-only gate')
            for g in groups:
                m=g['memory_checkpoint_observations'];r.need(m['actual_checkpoint_count']==5 and len(m['checkpoint_order'])==5,'Invented checkpoint')
                b=m['arms']['B']['checkpoints'];r.need(b['after_completed']['median_paired_delta_from_same_sample_before_create_bytes']['process_phys_footprint']==-2147483648 and
                  b['after_completed']['median_paired_delta_from_same_sample_before_create_bytes']['process_resident_size']==0,'Deferred negative task accounting concealed')
                r.need('conservative_completed_process_footprint_reduction_bytes' not in g,'Misleading exact physical gain field retained')
        check('deferred_negative_task_delta_observed_not_adoption_metric',deferred_check,True)
        checker=r.FloatReader()
        try:
            x=root/'a.f32';y=root/'b.f32';x.write_bytes(struct.pack('<4f',0,1,-1,2));y.write_bytes(x.read_bytes())
            check('all_original_F32_words_positive',lambda:r.need(checker.compare(x,y,16)['full_f32_bit_exact'],'Equal bytes rejected'),True)
            y.write_bytes(struct.pack('<4I',0,0x3f800001,0xbf800000,0x40000000))
            check('one_ULP_original_output_changed',lambda:r.need(checker.compare(x,y,16)['full_f32_bit_exact'],'Original bytes differ'),False)
            y.write_bytes(struct.pack('<4I',0,0x7fc00000,0xbf800000,0x40000000));check('NaN_original_output',lambda:checker.compare(x,y,16),False)
            y.write_bytes(b'\xa5'*16);check('unwritten_poison_original_output',lambda:checker.compare(x,y,16),False)
            y.write_bytes(b'\0'*12);check('short_original_output',lambda:checker.compare(x,y,16),False)
        finally:checker.close()
    report=dict(schema_version=1,reader_sha256=r.sha(source),scope='Synthetic CPU parser, original-byte and decision controls only',
        gpu_work_submitted=False,model_executed=False,service_operations=False,all_controls_pass=all(c['passed'] for c in checks),controls=checks)
    with a.out.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(all_controls_pass=report['all_controls_pass'],controls=len(checks),failed=[c['case'] for c in checks if not c['passed']])))
    return 0 if report['all_controls_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
