#!/usr/bin/env python3
"""Explicit isolated source-slab layout experiment with separate 0-ULP gate.

Reuses the frozen real R2 captures, not the old multihead experiment. Candidate
F16 dictionary/row IDs are CPU-prepared input fixtures: this is a necessary
layout test only, never production preparation or full-model acceptance. A win
must next pass GPU-constructible source layout including preparation cost.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import threading

sys.dont_write_bytecode = True
import run_attention_multihead_window as prefill
import run_iq2_signed4_window as safety

SYMBOL = 'kernel_myds_prefill_attention_indirect_f16_dk512_dv512'
need, sha, write_json = safety.need, safety.sha, safety.write_json


def protocol():
    return dict(schema_version=1, scope='single_head_indirect_f16_layout_only',
        production_layout_preparation_tested=False, fullmodel_run=False,
        production_adoption=False, mode='concurrent', layers=[2,8,19],
        rows=2048, heads=64, keys=640, providers=3, samples=9, reps=1,
        candidate_symbol=SYMBOL, control_symbol=prefill.CONTROL,
        maximum_absolute_control_drift_percent=2.0,
        minimum_conservative_time_saved_percent_strict=2.0,
        correctness='Complete three-layer/all-provider 0-ULP phase before any timing CB',
        preparation='CPU exact row dictionary excluded; layout-only, not production performance proof',
        next_gate='GPU-constructible layout including preparation and guards before full-model A1/B/A2')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_manifest(directory):
    m=json.loads((directory/'artifact-manifest.json').read_text())
    need(m.get('status')=='complete','Component manifest is not complete')
    seen=set();cached={}
    for entry in m['files']:
        rel=entry['path'];p=(directory/rel).resolve(strict=True)
        need(rel not in seen and directory.resolve() in p.parents,'Invalid logical raw path')
        seen.add(rel);s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if key not in cached:cached[key]=sha(p)
        need(s.st_size==entry['bytes'] and cached[key]==entry['sha256'],'Raw bytes/SHA differs: '+rel)
        if entry.get('storage_alias_of'):
            a=(directory/entry['storage_alias_of']).stat()
            need((a.st_dev,a.st_ino)==(s.st_dev,s.st_ino),'Raw hardlink alias differs')
    return m


def check_correctness(directory, stdout):
    m=check_manifest(directory);rows=read_rows(stdout)
    parity=[r for r in rows if r.get('kind')=='parity']
    need(len(parity)==9 and {(r['case'],r['provider']) for r in parity}==
         {(c,p) for c in range(3) for p in range(3)},'All complete real outputs required')
    need(all(r.get('full_f32_bit_exact') is True and r.get('mismatches')==0 and
         r.get('rows')==2048 and r.get('output_bytes')==268435456 for r in parity),'Full 0-ULP gate failed')
    need(not any(r.get('kind') in ('timing','summary') for r in rows),'Timing occurred before complete correctness gate')
    complete=[r for r in rows if r.get('kind')=='complete']
    need(len(complete)==1 and complete[0].get('full_f32_bit_exact') is True,'Correctness completion required')
    return dict(correctness_pass=True,correctness_manifest_sha256=sha(directory/'artifact-manifest.json'),
                logical_outputs=9,logical_f32=603979776)


def analyse_performance(directory, stdout):
    m=check_manifest(directory);rows=read_rows(stdout)
    timings=[r for r in rows if r.get('kind')=='timing'];summaries=[r for r in rows if r.get('kind')=='summary']
    need(len(timings)==180 and len(summaries)==6,'Expected 180 complete timing CB and six summaries')
    reports=[]
    for case in range(3):
        for kind in ('production','probe'):
            ts=[r for r in timings if r['case']==case and r['control_kind']==kind]
            need(len(ts)==30,'Per case/control timing coverage')
            need([(r['sample'],r['arm']) for r in ts]==
                 [(s,a) for s in range(10) for a in ('A1','B','A2')],'Strict interleaved A1/B/A2 required')
            arms={a:[] for a in ('A1','B','A2')}
            for r in ts:
                need(r.get('completed_cb') is True and r.get('full_f32_bit_exact') is True and
                     r.get('mode')=='concurrent' and r.get('rows')==2048,'Timing completion/shape/output gate')
                t=(r['gpu_end_s']-r['gpu_start_s'])*1e6
                need(math.isfinite(t) and t>0 and math.isclose(t,r['gpu_us_per_chunk'],rel_tol=1e-8,abs_tol=1e-5),
                     'GPU raw timestamp arithmetic differs')
                need(r['symbol']==(SYMBOL if r['arm']=='B' else prefill.CONTROL),'Selected symbol differs')
                if r['sample']:arms[r['arm']].append(t)
            a1,b,a2=(statistics.median(arms[a]) for a in ('A1','B','A2'))
            drift=(a2/a1-1)*100;saved=(1-b/min(a1,a2))*100
            pub=[r for r in summaries if r['case']==case and r['control_kind']==kind]
            need(len(pub)==1,'Unique summary grouping')
            values=dict(a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=drift,conservative_time_saved_percent=saved)
            need(all(math.isclose(pub[0][k],v,abs_tol=1e-5,rel_tol=1e-8) for k,v in values.items()),'Published summary differs')
            reports.append(dict(case=case,layer=(2,8,19)[case],control_kind=kind,**values,
                performance_gate_pass=abs(drift)<=2 and saved>2))
    passed=all(r['performance_gate_pass'] for r in reports if r['control_kind']=='production')
    return dict(layout_performance_gate_pass=passed,performance=reports,
                timing_command_buffers=180,formal_command_buffers=162,warmup_command_buffers=18,
                performance_manifest_sha256=sha(directory/'artifact-manifest.json'),
                production_component_performance_proven=False,
                next_gate='GPU source layout/preparation including its cost' if passed else 'Reject this layout implementation')


class IndirectWindow(prefill.AttentionWindow):
    def preflight(self):
        super().preflight()
        need(self.before['metallib_sha256']=='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81',
             'Prior frozen real-input native library no longer matches production')
        capture=json.loads(self.a.capture_json.read_text())
        need([c['layer'] for c in capture['cases']]==[2,8,19] and
             capture['source']['teacher_forced_tokens_used']==4096,'Frozen real capture scope differs')
        need(capture['source']['model_metadata']==self.summary['model'],'Original real model identity changed')
        need(capture['source']['effective_environment_sha256']==sha(self.a.out/'effective-env.json'),
             'Real input capture used a different effective production environment')
        inputs=[self.a.capture_json,self.a.stage/'bin/bench_attention_indirect',Path(__file__).resolve()]
        for c in capture['cases']:
            for name in ('args200','q_f32','kv_unique_f16','kv_row_ids_u32','mask_f16','sinks_f32'):
                p=Path(c[name]['path']).resolve(strict=True)
                need(p.stat().st_size==c[name]['bytes'] and sha(p)==c[name]['sha256'],'Real input descriptor mismatch')
                inputs.append(p)
        for p in inputs:self.input_hashes[str(p)]=sha(p)
        self.frozen_protocol=protocol();write_json(self.a.out/'protocol.json',self.frozen_protocol)
        self.input_hashes[str(self.a.out/'protocol.json')]=sha(self.a.out/'protocol.json')
        write_json(self.a.out/'stage-inputs.json',self.input_hashes)
        self.summary.update(component='prefill_attention_indirect_layout',scope=self.frozen_protocol['scope'],
            capture_json_sha256=sha(self.a.capture_json),production_layout_preparation_tested=False,
            fullmodel_run=False,production_adoption=False)

    def run(self):
        status,lock='failed',None
        def interrupted(number,frame):
            if self.recovering:self.event('deferred_signal_during_restoration',signal=number)
            else:raise KeyboardInterrupt('Indirect window interrupted '+str(number))
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
        try:
            lock=open('/tmp/myds-codex-decode-window.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.preflight()
            if self.a.prepare:
                status='prepared';self.summary.update(service_operations=False,gpu_work_submitted=False)
            else:
                self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start();self.stop()
                argv=[self.a.stage/'bin/bench_attention_indirect','--model',self.a.model,'--capture-json',self.a.capture_json,
                      '--probe-metallib',self.a.stage/'bin/myds.metallib','--production-metallib',self.a.out/'production-artifacts/myds.metallib']
                self.run_command('host-validation',[*argv,'--out-dir',self.a.out/'host-validation','--host-only'],env=self.environment)
                try:
                    self.run_command('correctness',[*argv,'--out-dir',self.a.out/'correctness','--correctness-only'],env=self.environment,timeout=300)
                except BaseException:
                    output=self.a.out/'correctness.stdout'
                    mismatch=[r for r in read_rows(output) if r.get('kind')=='parity' and r.get('mismatches',0)>0] if output.exists() else []
                    need(mismatch,'Correctness process failed without a recorded numerical rejection')
                    self.summary.update(correctness_pass=False,performance_measured=False,numerical_rejection=mismatch)
                    status='component_rejected_correctness'
                else:
                    result=check_correctness(self.a.out/'correctness',self.a.out/'correctness.stdout');self.summary.update(**result)
                    cm=self.a.out/'correctness/artifact-manifest.json'
                    self.run_command('performance',[*argv,'--out-dir',self.a.out/'performance','--correctness-manifest',cm,
                        '--correctness-manifest-sha256',sha(cm),'--samples','9','--reps','1'],env=self.environment,timeout=600)
                    result=analyse_performance(self.a.out/'performance',self.a.out/'performance.stdout');self.summary.update(**result,performance_measured=True)
                    status='layout_passed_requires_production_preparation_gate' if result['layout_performance_gate_pass'] else 'component_rejected_performance'
                need(not self.fault.is_set(),'Competition observed')
                for p,h in self.input_hashes.items():need(sha(p)==h,'Frozen input changed: '+p)
        except BaseException as exc:
            status='failed'
            self.summary['error']=repr(exc);self.event('window_failure',error=repr(exc))
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
                fullmodel_not_run_reason='Layout fixture is not a validated GPU production preparation path')
            write_json(self.a.out/'summary.json',self.summary);self.event('window_complete',status=status)
            if lock:lock.close()
        return 1 if status=='failed' else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','model','out','capture-json'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=8187);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');a=p.parse_args()
    need(sys.platform=='darwin','macOS host required')
    for name in ('stage','model','out','capture_json'):
        value=getattr(a,name);need(value.is_absolute(),'Absolute paths required');setattr(a,name,value.resolve())
    need(Path('/tmp').resolve() in a.stage.parents,'Isolated /tmp stage required')
    a.out.mkdir(parents=True,exist_ok=False);os.chmod(a.out,0o700);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with (a.out/'runner.log').open('ab',buffering=0) as log,open(os.devnull,'rb') as null:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    return IndirectWindow(a).run()


if __name__=='__main__':sys.exit(main())
