#!/usr/bin/env python3
"""CPU-only ownership/competition/recovery injection. No real service or GPU."""
import contextlib, fcntl, importlib.util, io, json, os
from pathlib import Path
import signal, struct, subprocess, sys, tempfile, threading, types, unittest
from unittest import mock
SPEC=importlib.util.spec_from_file_location('decode_window_test',Path(__file__).resolve().parents[1]/'tools/run_decode_probe_window.py')
w=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(w)

def row(pid,exe,ppid=1,pgid=None,start='Wed Oct 14 00:00:00 2026',args='',ports=()):
    return dict(pid=pid,executable=exe,comm=exe or 'myds',ppid=ppid,pgid=pgid or pid,start=start,args=args or exe,listen_ports=list(ports))

class Scanner:
    def __init__(self):self.window=None;self.mode=None;self.calls=0
    def snapshot(self,listeners=False):
        self.calls+=1;win=self.window;rows=[]
        for pid,item in win.permitted.items():rows.append(row(pid,item['executable'],ports=[8181]))
        if win.child and win.child.poll() is None:
            rows.append(row(win.child.pid,'/fixture/bin/bench_phase',ppid=os.getpid(),pgid=win.child.pid))
            rows.append(row(win.child.pid+100000,'/fixture/bin/test_attention',ppid=win.child.pid,pgid=win.child.pid))
            if not win.recovering:
                if self.mode=='monitor-failure':raise OSError('CPU fixture monitor unavailable')
                if self.mode=='race':rows.append(row(799997,'/other/myds',ports=[8182]))
        elif self.mode=='preflight':rows.append(row(799998,'/bin/bash',args='/bin/bash /tmp/run_prefill_prof.sh'))
        return rows

class Fixture(w.Window):
    def stop_production(self):
        self.check();self.stopped=True;self.trace.append('stop')
        with self.guard:self.permitted.clear()
    def restore(self):
        self.trace.append('restore');self.restore_artifacts()
        # Another owner cannot obtain the lock while original service recovers.
        with open(self.args.lock_path,'a+') as f:
            with self.testcase.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if self.restore_failure:raise RuntimeError('CPU fixture restore failure')
        if self.second_signal:self.handle_fixture_signal()
        value=dict(self.before,pid=900002,health={'status':'ok'})
        w.write(self.out/'restored.json',value);return value

class Tests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='decode-window-CPU-only-',dir='/tmp');self.addCleanup(t.cleanup);self.root=Path(t.name)
        (self.root/'bin').mkdir();self.binary=self.root/'bin/myds';self.lib=self.root/'bin/myds.metallib'
        self.binary.write_bytes(b'CPU original binary');self.binary.chmod(0o755);self.lib.write_bytes(b'CPU original library')
        self.before=dict(pid=900001,argv=[str(self.binary),'--arg with space'],argv_raw_hex=[str(self.binary).encode().hex(),b'--arg with space'.hex()],
            myds_env={'MYDS_DECODE_SPLIT':'20','MYDS_SPACE':'a b=c'},myds_env_raw_hex={b'MYDS_SPACE'.hex():b'a b=c'.hex()},cwd=str(self.root),log=str(self.root/'log'),
            binary=str(self.binary),binary_sha256=w.digest(self.binary),metallib=str(self.lib),metallib_sha256=w.digest(self.lib))
        self.args=types.SimpleNamespace(plan=self.root/'plan.json',output=self.root/'window',lock_path=str(self.root/'lock'),monitor_interval=.01,preflight_only=False)
        self.plan([sys.executable,'-c',"import time;time.sleep(.12);print('CPU only')"])
    def plan(self,argv,timeout=2):self.args.plan.write_text(json.dumps([dict(name='component-and-ABA',argv=argv,timeout=timeout)]))
    def invoke(self,mode=None,restore_failure=False,mutate=None):
        scanner=Scanner();window=Fixture(self.args,scanner);scanner.window=window;scanner.mode=mode
        window.trace=[];window.testcase=self;window.restore_failure=restore_failure;window.second_signal=False
        if mutate:mutate(window)
        with mock.patch.object(w,'capture',return_value=self.before),mock.patch.object(w,'listener',return_value=900001),mock.patch.object(w,'health',return_value={'status':'ok'}), \
             mock.patch.object(w.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','')),contextlib.redirect_stdout(io.StringIO()):
            result=window.run()
        return result,window,json.loads((self.args.output/'window-result.json').read_text())
    def test_success_owned_child_and_descendants_accepted_restore_under_lock(self):
        rc,win,r=self.invoke();self.assertEqual(rc,0);self.assertTrue(r['exclusive']);self.assertEqual(r['status'],'passed')
        self.assertEqual(win.trace,['stop','restore']);self.assertTrue(r['lock_held_through_restore']);self.assertIsNone(win.child)
        records=[json.loads(s) for s in (win.out/'exclusivity.jsonl').read_text().splitlines()]
        self.assertTrue(any(p['permitted'] and p['executable'].endswith('test_attention') for e in records if e['kind']=='machine_snapshot' for p in e['relevant']))
    def test_preflight_competitor_refuses_stop(self):
        rc,win,r=self.invoke('preflight');self.assertEqual(rc,1);self.assertFalse(win.trace);self.assertFalse(r['exclusive'])
        self.assertIn('Competing',r['error']);self.assertEqual(r['restoration']['mode'],'production_not_stopped')
    def test_racing8182_competitor_kills_only_owned_child_and_restores(self):
        self.plan([sys.executable,'-c','import time;time.sleep(20)'])
        rc,win,r=self.invoke('race');self.assertEqual(rc,1);self.assertFalse(r['exclusive']);self.assertIsNone(win.child)
        self.assertEqual(win.trace,['stop','restore']);self.assertTrue(r['restoration']['restored']);self.assertIn('Competing',r['monitor_error'])
    def test_monitor_failure_rejects_and_restores(self):
        self.plan([sys.executable,'-c','import time;time.sleep(20)']);rc,win,r=self.invoke('monitor-failure')
        self.assertEqual(rc,1);self.assertIn('unavailable',r['monitor_error']);self.assertTrue(r['restoration']['restored']);self.assertIsNone(win.child)
    def test_timeout_reaps_real_CPU_child_and_restores(self):
        self.plan([sys.executable,'-c','import time;time.sleep(20)'],.03);rc,win,r=self.invoke()
        self.assertEqual(rc,1);self.assertIn('TimeoutExpired',r['error']);self.assertTrue(r['restoration']['restored']);self.assertIsNone(win.child)
    def test_nonzero_child_rejected_and_restored(self):
        self.plan([sys.executable,'-c','import sys;sys.exit(7)']);rc,win,r=self.invoke();self.assertEqual(rc,1);self.assertIn('exited 7',r['error']);self.assertTrue(r['restoration']['restored'])
    def test_restore_failure_remains_failed_not_healthy(self):
        rc,win,r=self.invoke(restore_failure=True);self.assertEqual(rc,1);self.assertFalse(r['exclusive']);self.assertFalse(r['restoration']['restored'])
        self.assertFalse(json.loads((win.out/'restored.json').read_text())['healthy'])
    def test_preflight_only_never_stop_or_launch_and_no_reuse_output(self):
        self.args.preflight_only=True
        with mock.patch.object(w.subprocess,'Popen',side_effect=AssertionError('Readonly preflight launched process')):rc,win,r=self.invoke()
        self.assertEqual(rc,0);self.assertFalse(win.trace);self.assertEqual(r['status'],'preflight_only');self.assertFalse(r['exclusive'])
        with self.assertRaises(FileExistsError):self.invoke()
    def test_artifact_mutation_atomic_original_restore(self):
        self.plan([sys.executable,'-c','from pathlib import Path;Path('+repr(str(self.binary))+').write_bytes(b"changed");Path('+repr(str(self.lib))+').write_bytes(b"changed")'])
        rc,win,r=self.invoke();self.assertEqual(rc,1);self.assertTrue(r['artifact_mutation_detected']);self.assertTrue(r['restoration']['restored']);self.assertEqual(self.binary.read_bytes(),b'CPU original binary');self.assertEqual(self.lib.read_bytes(),b'CPU original library')
        self.assertEqual(self.binary.stat().st_mode&0o777,0o755)
    def test_cleanup_failure_prevents_restore_overlap(self):
        def mutate(win):win.drain=mock.Mock(side_effect=RuntimeError('owned child remains'))
        rc,win,r=self.invoke(mutate=mutate);self.assertEqual(rc,1);self.assertNotIn('restore',win.trace);self.assertFalse(r['restoration']['restored'])
        if win.child and win.child.poll() is None:os.killpg(win.child.pid,signal.SIGKILL);win.child.wait()
    def test_lock_contention_refuses_capture_or_stop(self):
        with open(self.args.lock_path,'a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);rc,win,r=self.invoke()
        self.assertEqual(rc,1);self.assertFalse(win.trace);self.assertFalse(r['lock_held_through_restore'])
    def test_input_mutation_before_step_rejected_and_restored(self):
        def mutate(win):
            original=win.stop_production
            def stop():original();self.args.plan.write_text(self.args.plan.read_text()+' ')
            win.stop_production=stop
        rc,win,r=self.invoke(mutate=mutate);self.assertEqual(rc,1);self.assertIn('Prepared input changed',r['error']);self.assertTrue(r['restoration']['restored'])
    def test_actual_signal_interrupts_child_and_restores(self):
        self.plan([sys.executable,'-c','import time;time.sleep(20)'])
        def mutate(win):
            original=win.run_step
            def step(*args):
                timer=threading.Timer(.05,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.start()
                try:return original(*args)
                finally:timer.cancel();timer.join()
            win.run_step=step
        old=signal.getsignal(signal.SIGTERM);rc,win,r=self.invoke(mutate=mutate);self.assertEqual(rc,1);self.assertIn('signal 15',r['error']);self.assertTrue(r['restoration']['restored']);self.assertEqual(signal.getsignal(signal.SIGTERM),old)
    def test_identity_PID_reuse_rejected_but_exec_allowed(self):
        win=Fixture(self.args,Scanner());win.out.mkdir();win.groups={12345};win.owned={12345:('start1',)}
        win.scanner.snapshot=lambda **kw:[row(12345,'/fixture/bench_changed',start='start1')]
        win.inspect('exec')
        win.scanner.snapshot=lambda **kw:[row(12345,'/fixture/bench_changed',start='start2')]
        with self.assertRaisesRegex(RuntimeError,'Owned PID reused'):win.inspect('reuse')
    def test_classification_avoids_grep_self_and_catches_wrappers_ports(self):
        for value in [row(2,'/bin/zsh',args='/bin/zsh /tmp/run_prefill_prof.sh'),row(2,'/bin/zsh',args='/bin/zsh /tmp/run_clamp.sh'),row(2,'/other/myds',ports=[8182]),row(2,'/unknown/server',ports=[8182]),row(2,'/python3',args='python3 -m mlx_lm.server')]:self.assertTrue(w.classification(value))
        for value in [row(3,'/bin/grep',args='grep bench_phase run_clamp.sh'),row(4,'/python3',args='python3 -c "print(\'bench_phase\')"'),row(5,'/bin/ps',args='ps aux')]:self.assertFalse(w.classification(value))
    def test_kernel_raw_space_equals_unicode_and_no_secret_environment(self):
        args=[b'/fixture/my ds',b'--literal a=b',b''];env=[b'MYDS_SPACE=a b=c',b'MYDS_EMPTY=',b'MYDS_UNICODE=\xe4\xb8\xad',b'SECRET=no-save']
        raw=struct.pack('i',len(args))+b'/fixture/exe\0\0'+b'\0'.join(args)+b'\0'+b'\0'.join(env)+b'\0'
        value=w.decode_kernel_args(raw);self.assertEqual(value['argv_raw_hex'],[a.hex() for a in args]);self.assertEqual(value['myds_env']['MYDS_SPACE'],'a b=c');self.assertNotIn('SECRET',value['myds_env'])
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):w.decode_kernel_args(raw+b'MYDS_SPACE=again\0')
    def test_strict_JSON_negative_controls(self):
        for raw in ['{"x":NaN}','{"x":1e999}','{"x":1,"x":1}']:
            with self.subTest(raw=raw),self.assertRaises((RuntimeError,ValueError)):w.parse(raw)
    def test_leader_exit_still_drains_owned_process_group(self):
        win=Fixture(self.args,Scanner());win.groups={9876};win.child=mock.Mock(pid=9876);alive={'value':True};signals=[]
        def killpg(pid,sig):
            signals.append(sig)
            if sig==signal.SIGTERM:alive['value']=False
            elif sig==0 and not alive['value']:raise ProcessLookupError()
        with mock.patch.object(w.os,'killpg',side_effect=killpg):win.drain()
        self.assertIn(signal.SIGTERM,signals);win_child=None;self.assertIsNone(win.child)
    def test_snapshot_and_allowstate_serialized(self):
        win=Fixture(self.args,Scanner());win.out.mkdir();inside=threading.Event();release=threading.Event();changed=threading.Event()
        def snapshot(**kw):inside.set();release.wait(2);return []
        win.scanner.snapshot=snapshot
        observer=threading.Thread(target=lambda:win.inspect('stale-race'));observer.start();self.assertTrue(inside.wait(1))
        def changer():
            with win.guard:win.groups.clear();changed.set()
        mutation=threading.Thread(target=changer);mutation.start();self.assertFalse(changed.wait(.03));release.set();observer.join();mutation.join();self.assertTrue(changed.is_set())
    def test_second_signal_during_restore_deferred(self):
        self.plan([sys.executable,'-c','import time;time.sleep(20)'])
        def mutate(win):
            original=win.restore
            def restore():os.kill(os.getpid(),signal.SIGTERM);return original()
            win.restore=restore
            old_step=win.run_step
            def step(*args):
                timer=threading.Timer(.05,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.start()
                try:return old_step(*args)
                finally:timer.cancel();timer.join()
            win.run_step=step
        rc,win,r=self.invoke(mutate=mutate);self.assertEqual(rc,1);self.assertTrue(r['restoration']['restored'])
        self.assertIn('signal_deferred_during_restore',(win.out/'exclusivity.jsonl').read_text())
    def test_restore_refuses_original_alive_without_listener(self):
        win=w.Window(self.args,Scanner());win.before=self.before;win.artifacts={}
        with mock.patch.object(w,'listener',return_value=None),mock.patch.object(w.os,'kill',return_value=None),mock.patch.object(w.subprocess,'Popen',side_effect=AssertionError('duplicate launched')):
            with self.assertRaisesRegex(RuntimeError,'still alive'):win.restore()

if __name__=='__main__':unittest.main()
