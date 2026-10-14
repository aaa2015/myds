#!/usr/bin/env python3
"""Exclusive prepared window; sampled all-machine competition rejects results.

The advisory flock spans preflight, steps, child cleanup and exact restoration.
It cannot prevent uncooperative GPU users or prove OS-level GPU exclusivity.
"""
import argparse, ctypes, fcntl, hashlib, json, math, os, pathlib, re, shlex
import shutil, signal, struct, subprocess, sys, threading, time, urllib.request
Path=pathlib.Path
LOCK='/tmp/myds-codex-decode-window.lock'
MODELS={'myds','myglm','ds4','llama-server','llama-cli','llama-bench','ollama','mlx_lm','vllm','lms'}
PROFILES={'run_prefill_prof.sh','run_clamp.sh','bench_reproduce.py','bench_decode_profile.py','run_decode_profile.py','run_kprof_window.sh'}

def require(ok,why):
    if not ok: raise RuntimeError(why)

def parse(raw):
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'Duplicate JSON key: '+k);result[k]=v
        return result
    def reject(v):raise ValueError('Nonfinite JSON value: '+v)
    def number(v):
        n=float(v);require(math.isfinite(n),'Nonfinite numeric value');return n
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=reject,parse_float=number)

def command(argv):return subprocess.check_output(argv,text=True,timeout=5).strip()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=True,allow_nan=False)+'\n')

def decode_kernel_args(data):
    argc=struct.unpack_from('i',data)[0];require(0<argc<4096,'Invalid KERN argc')
    cursor=data.index(b'\0',4)+1
    while cursor<len(data) and not data[cursor]:cursor+=1
    argv=[]
    for _ in range(argc):
        end=data.index(b'\0',cursor);argv.append(data[cursor:end]);cursor=end+1
    env,raw_env={},{}
    for raw in data[cursor:].split(b'\0'):
        if raw.startswith(b'MYDS_') and b'=' in raw:
            k,v=raw.split(b'=',1);name=k.decode('utf-8','surrogateescape')
            require(name not in env,'Duplicate KERN MYDS key')
            env[name]=v.decode('utf-8','surrogateescape');raw_env[k.hex()]=v.hex()
    return dict(argv=[v.decode('utf-8','surrogateescape') for v in argv],argv_raw_hex=[v.hex() for v in argv],myds_env=env,myds_env_raw_hex=raw_env)

def capture(pid):
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
    lib.sysctl.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_uint,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t]
    mib=(ctypes.c_int*3)(1,49,pid);size=ctypes.c_size_t()
    if lib.sysctl(mib,3,None,ctypes.byref(size),None,0):raise OSError(ctypes.get_errno(),'KERN size')
    buf=ctypes.create_string_buffer(size.value)
    if lib.sysctl(mib,3,buf,ctypes.byref(size),None,0):raise OSError(ctypes.get_errno(),'KERN read')
    value=decode_kernel_args(buf.raw[:size.value]);detail=command(['lsof','-a','-p',str(pid),'-d','cwd,1,2','-Fn'])
    names=re.findall(r'^n(.+)$',detail,re.M);require(len(names)==3 and names[1]==names[2],'Cannot bind cwd/common log')
    binary=Path(value['argv'][0]).resolve(strict=True);library=(binary.parent/'myds.metallib').resolve(strict=True)
    return dict(value,pid=pid,cwd=names[0],log=names[1],binary=str(binary),binary_sha256=digest(binary),metallib=str(library),metallib_sha256=digest(library))

def listener(required=True):
    r=subprocess.run(['lsof','-nP','-t','-a','-iTCP:8181','-sTCP:LISTEN'],capture_output=True,text=True,timeout=5)
    require(r.returncode in (0,1) and not r.stderr.strip(),'Listener check failed');pids=set(r.stdout.split())
    if not pids and not required:return None
    require(len(pids)==1,'Exactly one8181 listener required');return int(pids.pop())
def health():
    with urllib.request.urlopen('http://127.0.0.1:8181/health',timeout=3) as r:return parse(r.read())

def classification(p):
    name=Path(p.get('executable') or p.get('comm','')).name.lower();reasons=[]
    if name in MODELS or name.startswith(('bench_','bench-','llama-','profile_','test_')):reasons.append('model_or_native_benchmark')
    try:args=shlex.split(p.get('args',''))
    except ValueError:args=[]
    if name.startswith(('python','pypy')) or name in ('sh','bash','zsh','fish'):
        script=None
        for tok in args[1:]:
            if tok in ('-c','-m'):break
            if not tok.startswith('-'):script=tok;break
        if script:
            base=Path(script).name.lower()
            if base in PROFILES or base.startswith(('bench_','profile_')) or re.fullmatch(r'run_.*prof.*\.(py|sh)',base):reasons.append('profile_or_benchmark_script')
        if '-m' in args:
            i=args.index('-m')
            if i+1<len(args) and args[i+1].startswith(('mlx_lm','vllm','llama_cpp','torch.distributed')):reasons.append('model_python_module')
    if any(a.endswith(('.gguf','.safetensors')) for a in args[1:]):reasons.append('model_argument')
    if any(8080<=n<=8199 or n==6006 for n in p.get('listen_ports',[])):reasons.append('fleet_model_port')
    return reasons

class MachineScanner:
    def __init__(self):
        self.lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        self.lib.proc_pidpath.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32];self.lib.proc_pidpath.restype=ctypes.c_int
    def snapshot(self,listeners=False):
        rows=[]
        for line in command(['ps','-axww','-o','pid=,ppid=,pgid=,lstart=,args=']).splitlines():
            m=re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\w{3}\s+\w{3}\s+\d+\s+[\d:]+\s+\d{4})\s+(.*)$',line)
            require(m is not None,'Cannot parse machine ps snapshot');pid,ppid,pgid=map(int,m.group(1,2,3));buf=ctypes.create_string_buffer(4096)
            path=os.fsdecode(buf.value) if self.lib.proc_pidpath(pid,buf,len(buf))>0 else None
            rows.append(dict(pid=pid,ppid=ppid,pgid=pgid,start=m[4],comm=path or m[5].split(' ',1)[0],args=m[5],executable=path,listen_ports=[]))
        require(rows,'Empty machine ps snapshot')
        if listeners:
            r=subprocess.run(['lsof','-nP','-iTCP','-sTCP:LISTEN','-Fpn'],capture_output=True,text=True,timeout=5)
            require(r.returncode in (0,1) and not r.stderr.strip(),'Machine listeners unavailable');ports={};pid=None
            for line in r.stdout.splitlines():
                if line.startswith('p'):pid=int(line[1:])
                elif line.startswith('n') and pid is not None:
                    m=re.search(r':(\d+)$',line[1:])
                    if m:ports.setdefault(pid,set()).add(int(m[1]))
            require(set(ports)<={p['pid'] for p in rows},'Listener appeared during snapshot; refuse window')
            for p in rows:p['listen_ports']=sorted(ports.get(p['pid'],[]))
        return rows

class Window:
    def __init__(self,args,scanner=None):
        self.args=args;self.out=args.output;self.scanner=scanner;self.guard=threading.RLock()
        self.done=threading.Event();self.fault=threading.Event();self.monitor_error=None;self.thread=None
        self.before=None;self.stopped=False;self.recovering=False;self.permitted={};self.owned={};self.groups=set()
        self.child=None;self.outcomes=[];self.samples=0;self.original_env=None;self.artifacts={};self.artifacts_changed=False;self.lock=None;self.locked=False
    @staticmethod
    def key(p):return (p['start'],)
    def event(self,kind,**data):
        with self.guard:
            with (self.out/'exclusivity.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,monotonic_s=time.monotonic(),**data),ensure_ascii=True,allow_nan=False)+'\n')
    def inspect(self,phase,listeners=False):
        with self.guard:
            began=time.monotonic();rows=self.scanner.snapshot(listeners=listeners)
            by_pid={p['pid']:p for p in rows};allowed={os.getpid()}
            for pid,state in self.permitted.items():
                if pid in by_pid:
                    p=by_pid[pid];require(p.get('executable')==state['executable'],'Permitted executable changed')
                    if state.get('key') is None:state['key']=self.key(p)
                    require(state['key']==self.key(p),'Permitted PID identity changed');allowed.add(pid)
            owned_allowed=set()
            for p in rows:
                if p['pgid'] in self.groups or (p['pid'] in self.owned and self.owned[p['pid']]==self.key(p)):owned_allowed.add(p['pid'])
            while True:
                more={p['pid'] for p in rows if p['ppid'] in owned_allowed}
                if more<=owned_allowed:break
                owned_allowed|=more
            allowed|=owned_allowed
            for p in rows:
                if p['pid'] in owned_allowed:
                    if p['pid'] in self.owned:require(self.owned[p['pid']]==self.key(p),'Owned PID reused')
                    self.owned[p['pid']]=self.key(p)
                    if p['pgid']!=os.getpgrp():self.groups.add(p['pgid'])
            relevant=[]
            for p in rows:
                reasons=classification(p)
                if reasons:relevant.append({k:p[k] for k in ('pid','ppid','pgid','start','comm','executable','listen_ports')}|dict(reasons=reasons,permitted=p['pid'] in allowed))
            self.samples+=1;self.event('machine_snapshot',phase=phase,process_count=len(rows),relevant=relevant,listeners_checked=listeners,scan_s=time.monotonic()-began,unresolved_executable_count=sum(p['executable'] is None for p in rows))
            bad=[p for p in relevant if not p['permitted']];require(not bad,'Competing model/benchmark/profile PID: '+','.join(str(p['pid']) for p in bad))
            require(all(p['executable'] for p in relevant),'Relevant actual executable unavailable')
    def monitor(self):
        while not self.done.wait(self.args.monitor_interval):
            try:self.inspect('periodic')
            except BaseException as exc:
                self.monitor_error=repr(exc);self.fault.set()
                try:self.event('monitor_failure',error=repr(exc))
                except Exception:pass
                return
    def check(self):require(not self.fault.is_set(),'Machine monitor rejected window: '+str(self.monitor_error))
    def backup(self):
        folder=self.out/'production-artifacts';folder.mkdir()
        for kind in ('binary','metallib'):
            source=Path(self.before[kind]);target=folder/kind;stat=source.stat();shutil.copy2(source,target)
            require(digest(target)==self.before[kind+'_sha256'],'Production artifact changed during backup')
            self.artifacts[kind]=dict(path=str(source),backup=str(target),sha256=digest(target),mode=stat.st_mode&0o7777);target.chmod(0o600)
        write(self.out/'artifact-backups.json',self.artifacts)
    def restore_artifacts(self):
        for item in self.artifacts.values():
            source=Path(item['path']);backup=Path(item['backup']);require(digest(backup)==item['sha256'],'Original artifact backup changed')
            if not source.is_file() or digest(source)!=item['sha256'] or source.stat().st_mode&0o7777!=item['mode']:
                self.artifacts_changed=True
                temporary=source.with_name(source.name+'.window-restore-'+str(os.getpid()));require(not temporary.exists(),'Restore temp already exists')
                try:shutil.copy2(backup,temporary);temporary.chmod(item['mode']);os.replace(temporary,source)
                finally:
                    if temporary.exists():temporary.unlink()
                require(digest(source)==item['sha256'],'Original artifact restore failed');self.event('original_artifact_restored',path=str(source),sha256=item['sha256'])
    def group_alive(self,pgid):
        # Reap the leader before testing the group. Darwin can return EPERM for
        # a group containing the just-exited, unreaped leader instead of ESRCH.
        for attempt in range(2):
            if self.child:self.child.poll()
            try:os.killpg(pgid,0);return True
            except ProcessLookupError:return False
            except PermissionError:
                if attempt:raise
                time.sleep(.01)
    def drain(self):
        with self.guard:
            return self._drain_locked()
    def _drain_locked(self):
        for group in sorted(self.groups):
            require(group!=os.getpgrp(),'Refusing to signal controller group')
            if not self.group_alive(group):continue
            os.killpg(group,signal.SIGTERM);end=time.monotonic()+3
            while self.group_alive(group) and time.monotonic()<end:
                if self.child:self.child.poll()
                time.sleep(.05)
            if self.group_alive(group):
                os.killpg(group,signal.SIGKILL);end=time.monotonic()+3
                while self.group_alive(group) and time.monotonic()<end:
                    if self.child:self.child.poll()
                    time.sleep(.05)
            require(not self.group_alive(group),'Owned process group survived cleanup: '+str(group))
        if self.child:self.child.wait(timeout=2);self.child=None
        self.groups.clear();self.owned.clear()
    def stop_production(self):
        self.check();self.stopped=True;os.kill(self.before['pid'],signal.SIGTERM);end=time.monotonic()+40
        while time.monotonic()<end:
            self.check()
            try:os.kill(self.before['pid'],0)
            except ProcessLookupError:
                with self.guard:self.permitted.pop(self.before['pid'],None)
                self.inspect('production_stopped',listeners=True);return
            time.sleep(.1)
        raise RuntimeError('Original service did not terminate')
    def run_step(self,step,inputs):
        self.check();self.inspect('before_step:'+step['name'],listeners=True)
        for path,expected in inputs.items():require(digest(path)==expected,'Prepared input changed: '+path)
        self.check();env=dict(self.original_env);env.update(step.get('env',{}))
        for k in step.get('unset',[]):env.pop(k,None)
        began=time.monotonic();rc=None;error=None;print('Running '+step['name'],flush=True)
        try:
            with (self.out/(step['name']+'.stdout')).open('wb') as out,(self.out/(step['name']+'.stderr')).open('wb') as err:
                with self.guard:
                    self.child=subprocess.Popen(step['argv'],cwd=step.get('cwd',self.before['cwd']),env=env,stdout=out,stderr=err,start_new_session=True);self.groups.add(self.child.pid)
                end=began+step.get('timeout',900)
                while True:
                    self.check();rc=self.child.poll()
                    if rc is not None:break
                    if time.monotonic()>=end:raise subprocess.TimeoutExpired(step['argv'],step.get('timeout',900))
                    self.fault.wait(.05)
                self.inspect('after_step:'+step['name'],listeners=True)
                require(not any(self.group_alive(g) for g in self.groups),'Owned descendants remain after leader exited')
                require(rc==0,'Experiment failed: '+step['name']+' exited '+str(rc))
        except BaseException as exc:error=repr(exc);raise
        finally:
            try:self.drain()
            finally:self.outcomes.append(dict(name=step['name'],returncode=rc,seconds=time.monotonic()-began,error=error));write(self.out/'steps.json',self.outcomes)
    def restore(self):
        self.restore_artifacts();live=listener(required=False)
        if live is None:
            try:os.kill(self.before['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Original PID still alive without listener; refusing duplicate restore launch')
        if live is not None:
            require(live==self.before['pid'],'Unknown process owns8181; refusing to kill/adopt it');pid=live
        else:
            with Path(self.before['log']).open('ab') as log,self.guard:
                proc=subprocess.Popen(self.before['argv'],cwd=self.before['cwd'],env=self.original_env,stdout=log,stderr=log,start_new_session=True);pid=proc.pid
                self.permitted[pid]=dict(executable=self.before['binary'],key=None)
        end=time.monotonic()+180
        while time.monotonic()<end:
            try:
                h=health()
                if h.get('status')=='ok':
                    require(listener()==pid,'Restored listener is not original launched service');value=capture(pid)
                    for k in ('argv','argv_raw_hex','myds_env','myds_env_raw_hex','cwd','log','binary_sha256','metallib_sha256'):require(value[k]==self.before[k],'Restored identity differs: '+k)
                    value['health']=h;write(self.out/'restored.json',value);return value
            except (OSError,ValueError,subprocess.SubprocessError):pass
            time.sleep(.2)
        raise RuntimeError('Restore did not become healthy')
    def run(self):
        self.out.mkdir(parents=True,exist_ok=False);result=dict(status='failed',exclusive=False,restoration=dict(restored=False))
        handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        def interrupted(signum,frame):
            if self.recovering:self.event('signal_deferred_during_restore',signal=signum)
            else:raise KeyboardInterrupt('signal '+str(signum))
        try:
            self.lock=open(self.args.lock_path,'a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.locked=True
            for s in handlers:signal.signal(s,interrupted)
            self.scanner=self.scanner or MachineScanner();self.before=capture(listener());require(health().get('status')=='ok','Original service unhealthy')
            self.permitted[self.before['pid']]=dict(executable=self.before['binary'],key=None);self.inspect('preflight',listeners=True)
            active=subprocess.run(['lsof','-nP','-a','-iTCP:8181','-sTCP:ESTABLISHED'],capture_output=True,text=True,timeout=5)
            require(active.returncode in (0,1) and not active.stderr.strip() and not active.stdout.strip(),'Original connections/check failure')
            self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
            plan=parse(self.args.plan.read_text());require(type(plan) is list and plan,'Nonempty step array required');names=set();inputs={str(self.args.plan.resolve()):digest(self.args.plan)}
            for step in plan:
                name=step.get('name');require(type(name) is str and re.fullmatch(r'[A-Za-z0-9_-]+',name) and name not in names,'Unsafe/duplicate step name');names.add(name)
                require(type(step.get('argv')) is list and step['argv'] and all(type(a) is str and '\0' not in a for a in step['argv']),'Literal step argv required')
                timeout=step.get('timeout',900);require(type(timeout) in (int,float) and math.isfinite(timeout) and timeout>0,'Invalid timeout')
                for p in step.get('identities',[])+[step['argv'][0]]:
                    path=str(Path(p).resolve(strict=True));inputs[path]=digest(path)
            write(self.out/'before.json',self.before);write(self.out/'plan.json',plan);write(self.out/'inputs.json',inputs);self.backup()
            self.original_env={k:v for k,v in os.environ.items() if not k.startswith('MYDS_')};self.original_env.update(self.before['myds_env'])
            self.original_env.update(PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer');self.check()
            if self.args.preflight_only:self.inspect('readonly_preflight_complete',listeners=True);result.update(status='preflight_only',exclusive=False,service_operations=False,GPU_steps_executed=0)
            else:
                self.stop_production();print('Exclusive inference window acquired',flush=True)
                for step in plan:self.run_step(step,inputs)
                self.check();result.update(status='passed',exclusive=True)
        except BaseException as exc:result.update(status='failed',exclusive=False,error=repr(exc))
        finally:
            self.recovering=True
            try:
                self.drain()
                if self.stopped:
                    restored=self.restore();result['restoration']=dict(restored=True,pid=restored['pid']);print('Original argv, MYDS bytes, artifacts and health restored',flush=True)
                else:result['restoration']=dict(restored=True,mode='production_not_stopped')
            except BaseException as exc:
                result.update(status='failed',exclusive=False,restore_error=repr(exc));result['restoration']=dict(restored=False,error=repr(exc));write(self.out/'restored.json',dict(healthy=False,error=repr(exc)))
            self.done.set()
            if self.thread:self.thread.join(timeout=12)
            if self.fault.is_set() or (self.thread and self.thread.is_alive()):result.update(status='failed',exclusive=False,monitor_error=self.monitor_error or 'Monitor did not stop')
            if self.artifacts_changed:result.update(status='failed',exclusive=False,artifact_mutation_detected=True)
            result.update(monitor_samples=self.samples,monitor_interval_s=self.args.monitor_interval,lock_path=self.args.lock_path,lock_held_through_restore=self.locked,
                limits=['Advisory lock cannot prevent external uncooperative GPU work.','Periodic ps/proc_pidpath checks executable/script roles across ports; unknown renamed GPU programs and work shorter than sampling may evade detection.','Full TCP listeners checked before/between/after steps; periodic monitor does no lsof/artifact reads/hashing during GPU step.','Unresolved unrelated system executables are counted; inability to inspect relevant processes rejects the window.','Observed competition/monitor failure invalidates the entire window; only MYDS environment bytes retained.'])
            write(self.out/'window-result.json',result)
            for s,h in handlers.items():signal.signal(s,h)
            if self.lock:self.lock.close()
        return 0 if result['status'] in ('passed','preflight_only') else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--monitor-interval',type=float,default=1)
    p.add_argument('--lock-path',default=LOCK,help='All schedulers must share this lock')
    a=p.parse_args();require(math.isfinite(a.monitor_interval) and .1<=a.monitor_interval<=2,'Monitor interval must be0.1..2seconds');return Window(a).run()
if __name__=='__main__':sys.exit(main())
