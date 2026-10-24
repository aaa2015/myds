#!/usr/bin/env python3
"""Separate steady Output_A measurement: strict 0-ULP then 1024-projection CBs.

Build-only observer and isolated probe; explicit prepare/run, finally exact
production restoration. Component wins merely qualify full-model A1/B/A2.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import sys
import threading
sys.dont_write_bytecode=True
import run_attention_multihead_window as prefill
import run_attention_indirect_window as common
import run_iq2_signed4_window as safety
need,sha,write_json=safety.need,safety.sha,safety.write_json
LAYERS,POSITIONS=(0,19,39),(0,1,7,15)
CONTROL='kernel_dsv4_attn_out_low_q4_K_bf16'
CANDIDATE='kernel_myds_outa_q4_v41_fixed_bf16'


def protocol():
    return dict(schema_version=1,scope='ordinary_decode_output_a_fixed_wrapper_steady',
        measurement_method='separate_steady_cb_1024_projections',
        required_control_kinds=['production','probe'],
        mode='concurrent',layers=list(LAYERS),positions=list(POSITIONS),cases=12,providers=3,
        samples=15,reps=256,projections_per_cb=1024,control_symbol=CONTROL,candidate_symbol=CANDIDATE,
        geometry=dict(groups=[128,1,8],threads=[32,4,1],threadgroup_bytes=256,nr=2,nsg=4),
        maximum_absolute_control_drift_percent=2.0,minimum_conservative_time_saved_percent_strict=2.0,
        correctness='All 12 inputs x 3 providers complete F32 bits/BF16/finite/guards before timing',
        fullmodel_run=False,production_adoption=False)


def analyse(directory,stdout,correctness=False):
    m=common.check_manifest(directory);rows=common.read_rows(stdout)
    if correctness:
        p=[r for r in rows if r.get('kind')=='parity']
        need(len(p)==36 and {(r['case'],r['provider']) for r in p}==
             {(c,provider) for c in range(12) for provider in range(3)},'36 exact real provider outputs required')
        need(all(r.get('full_f32_bit_exact') is True and r.get('mismatches')==0 and
             r.get('completed_cb') is True for r in p),'Full Output_A bits/completion gate')
        done=[r for r in rows if r.get('kind')=='complete']
        need(len(done)==1 and done[0].get('cases')==12 and done[0].get('providers')==3 and
             done[0].get('output_a_checked') is True and done[0].get('full_f32_bit_exact') is True and
             done[0].get('all_gpu_inputs_unchanged') is True,'Complete Output_A output/input gate required')
        need(not any(r.get('kind') in ('timing','summary') for r in rows),'Timing before complete 0-ULP gate')
        return dict(correctness_pass=True,logical_provider_outputs=36,logical_output_f32=294912,
                    correctness_manifest_sha256=sha(directory/'artifact-manifest.json'))
    ts=[r for r in rows if r.get('kind')=='timing'];ss=[r for r in rows if r.get('kind')=='summary']
    need(len(ts)==288 and len(ss)==6,'288 timing CB and six summaries required');reports=[]
    for layer in LAYERS:
        for kind in ('production','probe'):
            selected=[r for r in ts if r['layer']==layer and r['control_kind']==kind]
            need([(r['sample'],r['arm']) for r in selected]==[(s,a) for s in range(16) for a in ('A1','B','A2')],
                 'Strict interleaved samples required')
            times={a:[] for a in ('A1','B','A2')}
            for r in selected:
                need(r.get('completed_cb') is True and r.get('full_f32_bit_exact') is True and
                     r.get('mode')=='concurrent' and r.get('projections_per_cb')==1024,'Complete projection CB required')
                value=(r['gpu_end_s']-r['gpu_start_s'])*1e6/1024
                need(math.isfinite(value) and value>0 and math.isclose(value,r['gpu_us_per_projection'],abs_tol=1e-5,rel_tol=1e-8),
                     'Raw GPU timestamp arithmetic differs')
                need(r['symbol']==(CANDIDATE if r['arm']=='B' else CONTROL),'Selected Output_A symbol differs')
                if r['sample']:times[r['arm']].append(value)
            a1,b,a2=(statistics.median(times[a]) for a in ('A1','B','A2'))
            drift=(a2/a1-1)*100;saved=(1-b/min(a1,a2))*100
            values=dict(a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=drift,conservative_time_saved_percent=saved)
            pub=[r for r in ss if r['layer']==layer and r['control_kind']==kind]
            need(len(pub)==1 and all(math.isclose(pub[0][k],v,rel_tol=1e-8,abs_tol=1e-5) for k,v in values.items()),
                 'Published summary differs from actual timestamps')
            reports.append(dict(layer=layer,control_kind=kind,**values,performance_gate_pass=abs(drift)<=2 and saved>2))
    passed=all(r['performance_gate_pass'] for r in reports)
    return dict(performance_gate_pass=passed,performance=reports,timing_command_buffers=288,
                formal_command_buffers=270,warmup_command_buffers=18,
                performance_manifest_sha256=sha(directory/'artifact-manifest.json'))


class OutputAWindow(prefill.AttentionWindow):
    def scan(self):
        relevant=super().scan()
        inventory=[]
        for line in safety.command(['/bin/ps','-axww','-o','pid=,ppid=,comm=']).splitlines():
            fields=line.strip().split(None,2);need(len(fields)==3,'Process inventory parse')
            inventory.append((int(fields[0]),int(fields[1]),fields[2]))
        allowed={os.getpid(),*self.allowed}
        while True:
            more={pid for pid,parent,_ in inventory if parent in allowed}
            if more<=allowed:break
            allowed|=more
        for pid,_,executable in inventory:
            if Path(executable).name=='capture_output_a':
                need(pid in allowed,'Unrelated Output_A capture process competes with window')
                relevant.append(dict(pid=pid,executable=executable,permitted=True))
        return relevant

    def preflight(self):
        super().preflight()
        env=self.before['effective_myds_env']
        need(env.get('MYDS_DENSE_Q4_NR0','2')=='2' and env.get('MYDS_DENSE_Q4_NSG','4')=='4',
             'Fixed wrapper requires actual production NR2/NSG4')
        need(env.get('MYDS_ATTN_OUT_A_FUSE_BF16','1')=='1',
             'Native fused BF16 Output_A provider must match production')
        need(env.get('MYDS_ATTN_ROPE_FUSE','0')=='1' and env.get('MYDS_ATTN_PACKED_REGISTER','0')=='1',
             'Expected production RF/register provider')
        for n in ('capture_output_a','bench_output_a_fixed_steady'):
            p=self.a.stage/'bin'/n;need(p.is_file(),'Build required isolated executable: '+n);self.input_hashes[str(p)]=sha(p)
        self.frozen_protocol=protocol();write_json(self.a.out/'protocol.json',self.frozen_protocol)
        self.input_hashes[str(self.a.out/'protocol.json')]=sha(self.a.out/'protocol.json')
        write_json(self.a.out/'stage-inputs.json',self.input_hashes)
        self.summary.update(component='ordinary_decode_output_a_fixed_wrapper_steady',scope=self.frozen_protocol['scope'])

    def capture_cases(self):
        request=self.a.out/'request.json'
        write_json(request,dict(model='local-myds-deepseek-v4.1-flash-1017-q2-q4',stream=True,
            messages=[dict(role='user',content='请用 Python 写一个稳定排序函数，输入是整数列表，并解释重复元素如何保持顺序。')],
            max_tokens=32,temperature=0,top_p=1,thinking=False))
        prefix=self.a.out/'tokenized'
        self.run_command('tokenize',[self.a.stage/'bin/count_chat_tokens',self.a.model,request,prefix],env=self.environment,timeout=120)
        raw=prefix.with_suffix('.tokens.i32le.bin').read_bytes();need(len(raw)>=64 and len(raw)%4==0,'16 real rendered IDs required')
        ids=struct.unpack('<16i',raw[:64]);need(all(i>=0 for i in ids),'Invalid tokenizer IDs')
        token_file=self.a.out/'capture-tokens.txt';token_file.write_text(' '.join(map(str,ids))+'\n')
        raw_dir=self.a.out/'raw-capture'
        self.run_command('capture',[self.a.stage/'bin/capture_output_a',self.a.model,'--tokens-file',token_file,'--out-dir',raw_dir],
                         env=self.environment,timeout=1200)
        provider=json.loads((raw_dir/'capture-provider.json').read_text())
        observer=json.loads((raw_dir/'capture-observer-summary.json').read_text())
        need(provider['provider_contract_passed'] and provider['forward_tokens_completed']==16 and
             provider['rf_requests']==provider['rf_encoded']==provider['register_requests']==provider['register_encoded']==640 and
             provider['rf_failures']==provider['register_failures']==0,'Actual ordinary RF/register capture provider failed')
        need(observer['status']=='complete' and observer['observed_layer_calls']==640 and observer['selected_cases']==12,
             'Complete source observer required')
        records=common.read_rows(raw_dir/'capture-records.jsonl')
        need(len(records)==12 and {(r['layer'],r['position']) for r in records}=={(l,p) for l in LAYERS for p in POSITIONS},
             'Exact real layer/position coverage required')
        cases=[]
        for r in sorted(records,key=lambda x:(x['layer'],x['position'])):
            need(r['need_logits'] and r['ordinary_decode_allowed'] and not r['trace_on'] and
                 r['all_f32_finite'] and r['all_bf16_bits'] and r['attention_rope_consumed'] and r['weight_type']==12,
                 'Post inverse-RoPE/BF16 ordinary input scope differs')
            d=r['input_f32'];p=(raw_dir/d['path']).resolve(strict=True);data=p.read_bytes()
            need(len(data)==d['bytes']==131072 and sha(p)==d['sha256'],'Input actual bytes/SHA differ')
            words=struct.unpack('<32768I',data)
            need(all((w&0x7f800000)!=0x7f800000 and not(w&0xffff) for w in words),'Input finite/BF16 bit contract')
            cases.append(dict(r,input_f32=dict(path=str(p),bytes=131072,sha256=sha(p))))
        path=self.a.out/'capture.json'
        write_json(path,dict(schema_version=1,capture_kind='myds_decode_output_a',model_bytes=self.summary['model']['bytes'],
            model_header_bytes=1048576,model_header_sha256=self.summary['model_header_sha256'],cases=cases,
            source=dict(kind='actual_public_ordinary_decode_post_inverse_rope_bf16',model_metadata=self.summary['model'],
                provider=provider,observer=observer,request_sha256=sha(request),token_file_sha256=sha(token_file),
                records_sha256=sha(raw_dir/'capture-records.jsonl'),effective_environment_sha256=sha(self.a.out/'effective-env.json'),
                capture_binary_sha256=sha(self.a.stage/'bin/capture_output_a'),capture_metallib_sha256=sha(self.a.stage/'bin/myds.metallib'),
                trace_on=False,performance_from_capture=False)))
        return path

    def run(self):
        status,lock='failed',None
        def interrupted(number,frame):
            if self.recovering:self.event('deferred_signal_during_restoration',signal=number)
            else:raise KeyboardInterrupt('Output_A interrupted '+str(number))
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
        try:
            lock=open('/tmp/myds-codex-decode-window.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.preflight()
            if self.a.prepare:status='prepared';self.summary.update(service_operations=False,gpu_work_submitted=False,capture_run=False)
            else:
                self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start();self.stop()
                capture=self.capture_cases();self.summary['capture_json_sha256']=sha(capture)
                argv=[self.a.stage/'bin/bench_output_a_fixed_steady','--model',self.a.model,'--capture-json',capture,
                    '--probe-metallib',self.a.stage/'bin/myds.metallib','--production-metallib',self.a.out/'production-artifacts/myds.metallib']
                self.run_command('host-validation',[*argv,'--out-dir',self.a.out/'host-validation','--host-only'],env=self.environment)
                try:self.run_command('correctness',[*argv,'--out-dir',self.a.out/'correctness','--correctness-only'],env=self.environment,timeout=300)
                except BaseException:
                    f=self.a.out/'correctness.stdout';bad=[r for r in common.read_rows(f) if r.get('kind')=='parity' and r.get('mismatches',0)>0] if f.exists() else []
                    need(bad,'Correctness executable failed without recorded numerical rejection')
                    self.summary.update(correctness_pass=False,performance_measured=False,numerical_rejection=bad);status='component_rejected_correctness'
                else:
                    self.summary.update(**analyse(self.a.out/'correctness',self.a.out/'correctness.stdout',True))
                    cm=self.a.out/'correctness/artifact-manifest.json'
                    self.run_command('performance',[*argv,'--out-dir',self.a.out/'performance','--correctness-manifest',cm,
                        '--correctness-manifest-sha256',sha(cm),'--samples','15','--reps','256'],env=self.environment,timeout=600)
                    result=analyse(self.a.out/'performance',self.a.out/'performance.stdout');self.summary.update(**result,performance_measured=True)
                    status='component_passed_requires_fullmodel' if result['performance_gate_pass'] else 'component_rejected_performance'
                need(not self.fault.is_set(),'Competition during Output_A window')
                for p,h in self.input_hashes.items():need(sha(p)==h,'Frozen source input changed: '+p)
        except BaseException as exc:
            status='failed';self.summary['error']=repr(exc);self.event('window_failure',error=repr(exc))
        finally:
            self.recovering=True
            try:
                self.drain()
                if self.stopped:self.summary['restoration']=self.restore()
                else:
                    if self.before and 'model' in self.summary:self.unchanged()
                    self.summary['restoration']=dict(restored=True,service_not_stopped=True)
            except BaseException as exc:
                status='failed';self.summary['restoration']=dict(restored=False,error=repr(exc));self.event('restoration_failure',error=repr(exc))
            self.done.set()
            if self.thread:self.thread.join(timeout=15)
            if self.fault.is_set() or (self.thread and self.thread.is_alive()):status='failed'
            self.summary.update(status=status,fullmodel_run=False,production_adoption=False,
                fullmodel_not_run_reason=('Prepare only; no GPU gate run' if status=='prepared' else
                    'Separate full-model A1/B/A2 gate required' if status=='component_passed_requires_fullmodel' else
                    'Capture/component gate did not qualify full-model validation'))
            write_json(self.a.out/'summary.json',self.summary);self.event('window_complete',status=status)
            if lock:lock.close()
        return 1 if status=='failed' else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','model','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=8187);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');a=p.parse_args()
    need(sys.platform=='darwin','macOS host required')
    for n in ('stage','model','out'):
        v=getattr(a,n);need(v.is_absolute(),'Absolute paths required');setattr(a,n,v.resolve())
    need(Path('/tmp').resolve() in a.stage.parents,'Isolated /tmp stage required')
    a.out.mkdir(parents=True,exist_ok=False);os.chmod(a.out,0o700);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with (a.out/'runner.log').open('ab',buffering=0) as log,open(os.devnull,'rb') as null:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    return OutputAWindow(a).run()


if __name__=='__main__':sys.exit(main())
