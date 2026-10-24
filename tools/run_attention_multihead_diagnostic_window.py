#!/usr/bin/env python3
"""Explicit, exclusive arithmetic diagnostic; never a performance/adoption gate.

The command JSON is an exact argv array for a separate bench_* diagnostic.
All absolute file arguments are SHA-frozen before stopping production. Original
service identity and finally restoration use the prefill window's same helpers.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
sys.dont_write_bytecode=True
import run_attention_multihead_window as prefill
import run_iq2_signed4_window as safety


class DiagnosticWindow(prefill.AttentionWindow):
    def run(self):
        status,lock='failed',None
        def interrupted(number,frame):
            if self.recovering:self.event('deferred_signal_during_restoration',signal=number)
            else:raise KeyboardInterrupt('Diagnostic interrupted '+str(number))
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
        try:
            lock=open('/tmp/myds-codex-decode-window.lock','a+')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.preflight()
            argv=json.loads(self.a.command_json.read_text())
            safety.need(isinstance(argv,list) and argv and all(isinstance(s,str) for s in argv),
                        'Expected exact string argv array')
            safety.need(Path(argv[0]).is_absolute() and Path(argv[0]).name.startswith('bench_'),
                        'Explicit absolute bench diagnostic required')
            frozen=[self.a.command_json,Path(__file__).resolve()]
            for arg in argv:
                if arg.startswith('/') and Path(arg).is_file():frozen.append(Path(arg))
                elif arg.startswith('/') and Path(arg).is_dir():
                    frozen.extend(path for path in Path(arg).rglob('*') if path.is_file())
            for path in frozen:self.input_hashes[str(path)]=safety.sha(path)
            self.frozen_protocol=dict(schema_version=1,scope='arithmetic_diagnostic_only',
                performance_measured=False,fullmodel_run=False,production_adoption=False,
                require_observer_output_equal_to_release_before_stage_attribution=True)
            safety.write_json(self.a.out/'protocol.json',self.frozen_protocol)
            self.input_hashes[str(self.a.out/'protocol.json')]=safety.sha(self.a.out/'protocol.json')
            safety.write_json(self.a.out/'stage-inputs.json',self.input_hashes)
            safety.write_json(self.a.out/'diagnostic-inputs.json',self.input_hashes)
            safety.write_json(self.a.out/'diagnostic-command.json',argv)
            self.summary.update(scope='arithmetic_diagnostic_only',performance_measured=False,
                                diagnostic_argv=argv,fullmodel_run=False,production_adoption=False)
            if self.a.prepare:
                status='prepared'
                self.summary.update(service_operations=False,gpu_work_submitted=False)
            else:
                self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
                self.stop()
                self.run_command('arithmetic-diagnostic',argv,env=self.environment,timeout=300)
                for path,digest in self.input_hashes.items():
                    safety.need(safety.sha(path)==digest,'Frozen diagnostic input changed: '+path)
                safety.need(not self.fault.is_set(),'Competition during diagnostic')
                status='diagnostic_completed'
        except BaseException as exc:
            self.summary['error']=repr(exc);self.event('diagnostic_failure',error=repr(exc))
        finally:
            self.recovering=True
            try:
                self.drain()
                if self.stopped:self.summary['restoration']=self.restore()
                else:
                    if self.before and 'model' in self.summary:self.unchanged()
                    self.summary['restoration']=dict(restored=True,service_not_stopped=True)
            except BaseException as exc:
                status='failed';self.summary['restoration']=dict(restored=False,error=repr(exc))
                self.event('restoration_failure',error=repr(exc))
            self.done.set()
            if self.thread:self.thread.join(timeout=15)
            if self.fault.is_set() or (self.thread and self.thread.is_alive()):status='failed'
            self.summary.update(status=status,fullmodel_not_run=True,
                fullmodel_not_run_reason='Arithmetic diagnostic only; no performance/adoption gate')
            safety.write_json(self.a.out/'summary.json',self.summary)
            self.event('window_complete',status=status)
            if lock:lock.close()
        return 0 if status in ('prepared','diagnostic_completed') else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','model','out','command-json'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,default=8187)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    safety.need(sys.platform=='darwin','macOS diagnostic host required')
    for name in ('stage','model','out','command_json'):
        value=getattr(args,name);safety.need(value.is_absolute(),'Absolute paths required')
        setattr(args,name,value.resolve())
    safety.need(Path('/tmp').resolve() in args.stage.parents,'Isolated /tmp source stage required')
    args.out.mkdir(parents=True,exist_ok=False);os.chmod(args.out,0o700)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with (args.out/'runner.log').open('ab',buffering=0) as log,open(os.devnull,'rb') as null:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    return DiagnosticWindow(args).run()


if __name__=='__main__':sys.exit(main())
