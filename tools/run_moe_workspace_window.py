#!/usr/bin/env python3
"""Isolated provider-specific MoE scratch gate. Builds external; finally restores service."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import sys
import threading
import time
sys.dont_write_bytecode = True
import run_iq2_signed4_window as safety
need, sha, write_json = safety.need, safety.sha, safety.write_json
FLAG = 'MYDS_PREFILL_MOE_WORKSPACE'
PLAN_SHA = '55b43f634c43dd8c9c078cbc948bf41763fab9d6eba4cdbe725e758e002a9f20'

def source_inputs(stage):
    files = [stage / 'Makefile']
    for directory in ('src','include','metal','tools','third_party'):
        files += [p for p in (stage/directory).rglob('*') if p.is_file() and
                  (p.suffix in ('.c','.h','.m','.metal','.py','.inc','.mk') or p.name == 'LICENSE')]
    files += [p for p in (stage/'bin').iterdir() if p.is_file() and
              (p.name in ('myds','myds.metallib') or p.name.startswith(('bench_moe_workspace','capture_moe_workspace','test_moe_workspace')))]
    return files


def process_memory(pid):
    # rusage_info_v2 begins with UUID[16], then the ten rusage_info_v0 uint64s.
    # libproc fills a larger bounded buffer. ri_phys_footprint is byte offset72.
    lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(1024)
    if lib.proc_pid_rusage(pid,2,buffer):
        raise OSError(ctypes.get_errno(),'proc_pid_rusage')
    return dict(resident_bytes=struct.unpack_from('Q',buffer,64)[0],
                phys_footprint_bytes=struct.unpack_from('Q',buffer,72)[0])


def system_wired():
    raw=safety.command(['/usr/bin/vm_stat'])
    page=re.search(r'page size of (\d+) bytes',raw)
    wired=re.search(r'Pages wired down:\s*(\d+)\.',raw)
    need(page and wired,'Cannot parse actual wired memory')
    return int(page[1])*int(wired[1])


class WorkspaceWindow(safety.Window):
    def preflight(self):
        self.before,self.config_raw=safety.capture(self.a.port); self.allowed.add(self.before['pid'])
        need(os.getpriority(os.PRIO_PROCESS,os.getpid())<=self.before['nice'],'Nice restoration unavailable')
        need(Path(safety.option(self.before['argv'],'-m')).resolve(strict=True)==self.a.model,'Actual model differs')
        need(self.before['binary_sha256']=='29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b' and
             self.before['metallib_sha256']=='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81','Production baseline differs')
        need(sha(self.a.gold_stage/'bin/myds')==self.before['binary_sha256'] and
             sha(self.a.gold_stage/'bin/myds.metallib')==self.before['metallib_sha256'],'Gold must reproduce production byte-for-byte')
        need(sha(self.a.stage/'bin/myds.metallib')==self.before['metallib_sha256'],'Arithmetic/shader library must remain byte-exact')
        need(sha(self.a.plan)==PLAN_SHA,'Frozen plan changed')
        self.plan=json.loads(self.a.plan.read_text())
        self.environment={k:v for k,v in os.environ.items() if not k.startswith('MYDS_')}
        self.environment.update(self.before['effective_myds_env'])
        self.environment.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')
        need(len(self.before['effective_myds_env'])==51 and self.environment.get('MYDS_METAL_CONCURRENT')=='1' and
             self.environment.get('MYDS_PREFIX_CACHE')=='1' and self.environment.get('MYDS_PLD')=='1','Actual51 production settings required')
        for k in self.before['effective_myds_env']:
            need(not k.startswith(('MYDS_PROFILE_','MYDS_TRACE_','MYDS_SKIP_')),'Diagnostic production selector')
        self.summary.update(component='actual_prefill_moe_workspace',model=safety.metadata(self.a.model),
                            model_header_sha256=sha(self.a.model,1048576),plan_sha256=sha(self.a.plan))
        write_json(self.a.out/'before.json',self.before); (self.a.out/'original-myds.conf').write_bytes(self.config_raw)
        write_json(self.a.out/'effective-env.json',self.before['effective_myds_env'])
        backup=self.a.out/'production-artifacts'; backup.mkdir()
        for key in ('binary','metallib'):
            dst=backup/('myds' if key=='binary' else 'myds.metallib'); shutil.copy2(self.before[key]['path'],dst)
            need(sha(dst)==self.before[key+'_sha256'],'Production backup differs')
        for name,record in self.plan['fullmodel']['requests'].items():
            need(sha(self.a.inputs/(name+'.json'))==record['sha256'],'Request differs')
        raw=(self.a.inputs/'capture-tokens.i32le.bin').read_bytes()
        ids=list(map(int,(self.a.inputs/'capture-tokens.txt').read_text().split()))
        need(len(ids)==16401 and struct.pack('<16401i',*ids)==raw and
             sha(self.a.inputs/'capture-tokens.i32le.bin')==self.plan['capture']['raw_ids_sha256'],'Actual frozen tokenizerIDs differ')
        files=[*source_inputs(self.a.stage),*source_inputs(self.a.gold_stage),self.a.plan,
               *[p for p in self.a.inputs.rglob('*') if p.is_file()]]
        self.input_hashes={str(p):sha(p) for p in sorted(set(files))}
        need(str(Path(__file__).resolve()) in self.input_hashes and str(Path(safety.__file__).resolve()) in self.input_hashes,'Execute frozen runner/helper')
        write_json(self.a.out/'stage-inputs.json',self.input_hashes); write_json(self.a.out/'protocol.json',self.plan)
        self.event('preflight_processes',relevant=self.scan())
        # CPU sampler ABI sanity against own process before any service mutation.
        actual=process_memory(os.getpid());need(actual['phys_footprint_bytes']>0 and actual['resident_bytes']>0,'Memory ABI check')
        self.event('memory_guard_preflight',process=actual,system_wired_bytes=system_wired())

    def scan(self):
        rows = []
        for line in safety.command(['/bin/ps', '-axww', '-o', 'pid=,ppid=,comm=']).splitlines():
            parts = line.strip().split(None, 2)
            need(len(parts) == 3, 'Cannot parse process inventory')
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        allowed = {os.getpid(), *self.allowed}
        while True:
            more = {pid for pid, parent, _ in rows if parent in allowed}
            if more <= allowed:
                break
            allowed |= more
        relevant = []
        models = {'myds', 'myglm', 'ds4', 'llama-server', 'llama-cli', 'llama-bench',
                  'ollama', 'mlx_lm', 'vllm', 'lms', 'trace_model',
                  'capture_moe_workspace', 'test_moe_workspace_allocations', 'test_moe_workspace_cancel', 'test_moe_workspace_recovery'}
        for pid, _, exe in rows:
            name = Path(exe).name.lower()
            if name in models or name.startswith(('bench_', 'bench-', 'moe_prefill_replay')):
                relevant.append(dict(pid=pid, executable=exe, permitted=pid in allowed))
        need(all(row['permitted'] for row in relevant), 'Competing model/benchmark: ' + repr(relevant))
        return relevant

    def monitor(self):
        while not self.done.wait(.5):
            try:
                with self.guard:
                    rows=self.scan(); child=self.child
                    memory=process_memory(child.pid) if child and child.poll() is None else None
                wired=system_wired()
                if memory: need(memory['phys_footprint_bytes']<self.plan['safety']['stop_on_process_phys_footprint_bytes'],'Process footprint safety ceiling reached')
                need(wired<self.plan['safety']['stop_on_system_wired_bytes'],'System wired safety ceiling reached')
                self.event('monitor',relevant=rows,child_memory=memory,system_wired_bytes=wired)
            except ProcessLookupError:
                continue
            except OSError as exc:
                # A child may exit between poll and proc_pid_rusage; only ignore a proven exit.
                if child and child.poll() is not None: continue
                self.event('monitor_failure',error=repr(exc));self.fault.set();return
            except BaseException as exc:
                self.event('monitor_failure',error=repr(exc));self.fault.set();return

    def component(self):
        env=dict(self.environment,**{FLAG:'0'})
        self.run_command('allocations',[self.a.stage/'bin/test_moe_workspace_allocations','--run','--out-dir',self.a.out/'allocations'],env=env)
        complete=[json.loads(x) for x in (self.a.out/'allocations/records.jsonl').read_text().splitlines()]
        need(complete[-1].get('all_pass') is True,'Actual allocator diagnostics incomplete')
        capture=self.a.out/'raw-capture'
        self.run_command('capture',[self.a.stage/'bin/capture_moe_workspace',self.a.model,'--tokens-file',self.a.inputs/'capture-tokens.txt','--out-dir',capture],env=env,timeout=2400)
        capture_json=capture/'capture.json'
        common=[self.a.stage/'bin/bench_moe_workspace','--model',self.a.model,'--capture-json',capture_json]
        self.run_command('host',[*common,'--out-dir',self.a.out/'host','--host-only'],env=env)
        self.run_command('correctness',[*common,'--out-dir',self.a.out/'correctness','--correctness-only'],env=env,timeout=1800)
        self.run_command('failures',[self.a.stage/'bin/bench_moe_workspace_failures','--model',self.a.model,'--capture-json',capture_json,'--out-dir',self.a.out/'failures','--run'],env=env,timeout=1800)
        self.run_command('failure-review',['/usr/bin/python3',self.a.stage/'tools/test_moe_workspace_failures.py','--root',self.a.out/'failures','--out',self.a.out/'failure-review.json'],env=env)
        prior=self.a.out/'correctness/artifact-manifest.json'
        need(json.loads(prior.read_text()).get('full_f32_bit_exact') is True,'Prior complete real0ULP required')
        self.run_command('performance',[*common,'--out-dir',self.a.out/'performance','--correctness-manifest',prior,'--correctness-manifest-sha256',sha(prior)],env=env,timeout=1800)
        self.summary.update(component_executed=True,component_review_required=True,fullmodel_run=False)

    def run(self):
        status='failed';lock=None
        def interrupted(number,frame):
            if self.recovering:self.event('deferred_signal_during_restoration',signal=number)
            else:raise KeyboardInterrupt('Workspace window interrupted '+str(number))
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
        try:
            lock=open('/tmp/myds-codex-decode-window.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.preflight()
            if self.a.prepare:status='prepared';self.summary.update(service_operations=False,gpu_work_submitted=False)
            else:
                self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start();self.stop()
                self.component();status='component_complete_review_required'
                for path,digest in self.input_hashes.items():need(sha(path)==digest,'Frozen input changed: '+path)
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
            self.summary.update(status=status,production_adoption=False)
            write_json(self.a.out/'summary.json',self.summary);self.event('window_complete',status=status)
            if lock:lock.close()
        return int(status=='failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','gold-stage','model','out','plan','inputs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=8187)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();need(sys.platform=='darwin','macOS required')
    for k in ('stage','gold_stage','model','out','plan','inputs'):
        v=getattr(a,k);need(v.is_absolute(),'Absolute paths required');setattr(a,k,v.resolve())
    for stage in (a.stage,a.gold_stage):need(Path('/tmp').resolve() in stage.parents,'Separate tmp stage required')
    a.out.mkdir(parents=True,exist_ok=False);os.chmod(a.out,0o700);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with (a.out/'runner.log').open('ab',buffering=0) as log,open(os.devnull,'rb') as null:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    return WorkspaceWindow(a).run()

if __name__=='__main__':sys.exit(main())
