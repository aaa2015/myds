#!/usr/bin/env python3
"""CPU-only fault tests; no server, process signal, network or GPU is used."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import copy
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('controlled_window', ROOT/'tools/run_controlled_service_window.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='controlled-window-test-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve(); self.root = root
        self.conf = root/'fleet.conf'; self.conf.write_text('fixture')
        self.log = root/'fleet.log'; self.log.write_text('fixture log')
        self.model = root/'model.gguf'; self.model.write_bytes(b'model')
        self.args = types.SimpleNamespace(root=root/'window', cwd=root, pid=12345, port=8187,
            stdin_path=self.conf, log_path=self.log, config=self.conf, model=self.model, library=root/'myds.metallib',
            expected_nice=5, lock=root/'lock', expected_identity='0'*64, protect=[],
            command=['/fixture/runner'], stop_timeout=.01, drain_timeout=.01, restore_timeout=.01)
        self.w = m.ControlledWindow(self.args)
        self.w.stdio = [os.open(self.conf, os.O_RDONLY), os.open(self.log, os.O_WRONLY|os.O_APPEND)]
        self.w.identity = {'fds': [self.fd(self.conf), self.fd(self.log), self.fd(self.log)],
                           'model':m.model_identity(self.model), 'uid':os.geteuid(), 'gid':os.getegid()}
        self.w.identity['fds'][1]['flags'] = self.w.identity['fds'][2]['flags'] = 0xa
        self.addCleanup(self.w.close_fds)

    def fd(self, path):
        s=path.stat(); return dict(path=str(path),dev=s.st_dev,inode=s.st_ino,mode=s.st_mode,
                                   flags=1,kind=1,guard=0,offset=0,status=0)

    def test_original_fd_accepted_and_path_replacement_rejected(self):
        self.w.verify_stdio()
        self.log.rename(self.root/'rotated.log'); self.log.write_text('replacement')
        with self.assertRaisesRegex(RuntimeError,'inode or path drift'):
            self.w.verify_stdio()

    @unittest.skipUnless(sys.platform == 'darwin','macOS proc_pidfdinfo required')
    def test_proc_pidfdinfo_reads_real_owned_child_stdio(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],
            stdin=self.w.stdio[0],stdout=self.w.stdio[1],stderr=self.w.stdio[1])
        try:
            # Wait only for the owned child to be observable; never touch a
            # server or send an inference request.
            time.sleep(.05)
            stdin=m.fd_info(child.pid,0); stdout=m.fd_info(child.pid,1); stderr=m.fd_info(child.pid,2)
            self.assertEqual(stdin['inode'],self.conf.stat().st_ino)
            self.assertEqual(stdout['inode'],self.log.stat().st_ino)
            self.assertEqual(stdout['flags'] & ~0x10000,0xa)
            self.assertEqual(stderr['inode'],stdout['inode'])
            self.assertTrue(stdout['status'] & 1 and stderr['status'] & 1)
            os.write(self.w.stdio[1],b'fixture write')
            self.assertEqual(m.fd_info(child.pid,1)['flags'],0x1000a)
        finally:
            child.terminate(); child.wait(timeout=5)

    def test_missing_append_rejected(self):
        import fcntl
        flags=fcntl.fcntl(self.w.stdio[1],fcntl.F_GETFL)
        fcntl.fcntl(self.w.stdio[1],fcntl.F_SETFL,flags & ~os.O_APPEND)
        with self.assertRaisesRegex(RuntimeError,'O_APPEND'):
            self.w.verify_stdio()

    def test_model_change_rejected(self):
        self.w.state['identities'] = {}
        self.w.verify()
        self.model.write_bytes(b'different model')
        with self.assertRaisesRegex(RuntimeError,'model identity drift'):
            self.w.verify()

    def test_snapshot_drift_before_signal_stops_without_kill(self):
        identity=self.w.identity
        def preflight(): self.w.identity=identity
        with mock.patch.object(self.w,'preflight',side_effect=preflight), \
             mock.patch.object(m,'snapshot',return_value={'changed':True}), \
             mock.patch.object(m.os,'kill') as kill, \
             mock.patch.object(m.signal,'signal'), mock.patch.object(m.signal,'getsignal'), \
             contextlib.redirect_stdout(io.StringIO()):
            code=self.w.run()
        self.assertEqual(code,1); kill.assert_not_called(); self.assertFalse(self.w.state['stop_requested'])

    def test_restore_blocked_when_experiment_does_not_exit(self):
        self.w.experiment=types.SimpleNamespace(poll=lambda:None,
            send_signal=mock.Mock(), wait=mock.Mock(side_effect=subprocess.TimeoutExpired('fixture',1)))
        with self.assertRaisesRegex(RuntimeError,'owned experiment remains'):
            self.w.restore()
        self.w.experiment.send_signal.assert_called_once_with(signal.SIGINT)

    def test_restore_blocks_artifact_drift_before_spawn(self):
        self.w.state['identities']={}; self.model.write_bytes(b'changed')
        with mock.patch.object(self.w,'drain_experiment'), mock.patch.object(self.w,'wait_original_exit'), \
             mock.patch.object(m.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError,'model identity drift'):
                self.w.restore()
            popen.assert_not_called()

    def test_restore_defers_signals_during_spawn(self):
        self.w.spawning_restore=True
        self.w.handle_signal(signal.SIGTERM,None)
        self.assertEqual(self.w.state['deferred_recovery_signals'],[signal.SIGTERM])

    def test_identity_digest_changes_with_captured_environment(self):
        one={'process':{'environment':{'A':'one'}},'nice':5}
        two={'process':{'environment':{'A':'two'}},'nice':5}
        self.assertNotEqual(m.canonical_digest(one),m.canonical_digest(two))

    def test_effective_environment_digest_uses_config_precedence_and_comments(self):
        import hashlib
        self.conf.write_bytes(b'# comment\nMYDS_A = config # comment\nMYDS_B=\nOTHER=ignored\n')
        env={'MYDS_A':'env','MYDS_C':'z','PATH':'ignored'}
        raw=b'MYDS_A=config\0MYDS_B=\0MYDS_C=z\0'
        self.assertEqual(m.effective_env_digest(env,self.conf),hashlib.sha256(raw).hexdigest())

    def test_priority_probe_fails_before_signal_when_nice_not_restorable(self):
        identity={'uid':os.geteuid(),'gid':os.getegid(),'nice':5}
        failed=types.SimpleNamespace(returncode=1,stdout='',stderr='permission denied')
        with mock.patch.object(m.subprocess,'run',return_value=failed), \
             mock.patch.object(m.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'cannot restore target uid/gid/nice'):
                m.verify_priority_can_be_restored(identity)
        kill.assert_not_called()

    def test_arbitrary_shell_cannot_be_harness(self):
        args=types.SimpleNamespace(harness=Path('/bin/sh'),harness_sha256='0'*64,harness_arg=[])
        with self.assertRaisesRegex(RuntimeError,'fixed mask-warm native harness'):
            m.harness_identity(args)

    def test_frozen_native_harness_inputs_and_digest(self):
        binary=self.root/'bench_indexer_mask_warm'
        binary.write_bytes(b'\xcf\xfa\xed\xfe' + b'fixture'); binary.chmod(0o700)
        library=self.root/'myds.metallib'; library.write_bytes(b'library')
        inputs=[]
        for i in range(4):
            path=self.root/f'input-{i}'; path.write_text(str(i)); inputs.append(str(path))
        args=types.SimpleNamespace(harness=binary,harness_sha256=m.base.sha(binary),model=self.model,
            experiment_library=library,
            harness_arg=['--run',str(self.model),*inputs,str(self.root/'new-output'),'262144',
                         str(self.conf),'a'*64,'off'])
        initial=m.harness_identity(args)
        Path(inputs[0]).write_text('changed')
        self.assertNotEqual(initial,m.harness_identity(args))
        binary.write_bytes(b'\xcf\xfa\xed\xfe' + b'changed')
        with self.assertRaisesRegex(RuntimeError,'SHA mismatch'):
            m.harness_identity(args)

    def test_three_arm_plan_requires_same_config_except_candidate_selector(self):
        binary=self.root/'bench_indexer_mask_warm'; binary.write_bytes(b'\xcf\xfa\xed\xfefixture'); binary.chmod(0o700)
        library=self.root/'myds.metallib'; library.write_bytes(b'library')
        self.conf.write_text('MYDS_PREFIX_CACHE=1\nMYDS_PLD=1\n')
        candidate=self.root/'candidate.conf'
        candidate.write_text(self.conf.read_text()+'MYDS_INDEXER_MASK_SKIP=1\n')
        inputs=[]
        for i in range(4):
            path=self.root/f'input-{i}'; path.write_text(str(i)); inputs.append(str(path))
        def arm(role,config,mode):
            return {'role':role,'argv':['--run',str(self.model),*inputs,str(self.root/role),
                                        '1048576',str(config),'a'*64,mode]}
        args=types.SimpleNamespace(harness=binary,harness_sha256=m.base.sha(binary),
            experiment_library=library,model=self.model,config=self.conf,harness_arg=[],
            harness_arms=[arm('A1',self.conf,'off'),arm('B',candidate,'on'),arm('A2',self.conf,'off')])
        self.assertEqual([x['role'] for x in m.harness_identity(args)['arms']],['A1','B','A2'])
        candidate.write_text('MYDS_PREFIX_CACHE=0\nMYDS_PLD=1\nMYDS_INDEXER_MASK_SKIP=1\n')
        with self.assertRaisesRegex(RuntimeError,'differs beyond mask selector'):
            m.harness_identity(args)

    def test_summary_library_mismatch_rejects_arm(self):
        import json
        output=self.root/'output'; output.mkdir()
        argv=['--run',str(self.model),'base','base_ids','append','append_ids',str(output),
              '262144',str(self.conf),'a'*64,'on']
        identity={'model':m.model_identity(self.model), 'harness':{
            'path':str(self.root/'bench_indexer_mask_warm'),'sha256':'b'*64,
            'selected_library':{'resolved':str(self.root/'myds.metallib'),'sha256':'c'*64},
            'arms':[{'argv':argv,'inputs':{str(self.conf):'d'*64}}]}}
        summary={'kind':'ordinary_warm_prefill_screen','observer':False,
                 'executable_path':identity['harness']['path'],'executable_sha256':'b'*64,
                 'metallib_path':str(self.root/'wrong.metallib'),'metallib_sha256':'c'*64}
        (output/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(RuntimeError,'actual metallib identity mismatch'):
            m.validate_arm_summary(identity,0)

    def test_arm_output_mismatch_is_rejected(self):
        phase={'actual_tokens':10,'cached_tokens':0,'prefill_tokens':9,'stop_reason':1,
               'output':{'sha256':'a','bytes':1},'output_token_ids':{'sha256':'b','bytes':4}}
        reference={'phases':[phase,phase]}; candidate=copy.deepcopy(reference)
        candidate['phases'][1]['output']['sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'arm output differs'):
            m.compare_arm_outputs(reference,candidate)

    def test_valid_summary_and_payload_corruption(self):
        import json
        output=self.root/'output'; output.mkdir()
        inputs=[]
        for i in range(4):
            path=self.root/f'input-{i}'; path.write_text(str(i)); inputs.append(str(path))
        argv=['--run',str(self.model),*inputs,str(output),'1048576',str(self.conf),'a'*64,'on']
        hashes={p:m.base.sha(p) for p in inputs}; hashes[str(self.conf)]=m.base.sha(self.conf)
        library=self.root/'myds.metallib'; exe=self.root/'bench_indexer_mask_warm'
        identity={'model':m.model_identity(self.model),'harness':{'path':str(exe),'sha256':'b'*64,
           'selected_library':{'resolved':str(library),'sha256':'c'*64},
           'arms':[{'argv':argv,'inputs':hashes}]}}
        phases=[]
        for i,role in enumerate(('cold','warm_append')):
            desc={}
            for name in ('output','output_token_ids'):
                payload=output/f'{role}-{name}'; payload.write_bytes(b'ok')
                desc[name]={'path':payload.name,'bytes':2,'sha256':m.base.sha(payload)}
            phases.append(dict(phase=role,cached_tokens=0 if i==0 else 5,
                prompt_sha256=hashes[inputs[i*2]],tokens_sha256=hashes[inputs[i*2+1]],**desc))
        summary={'kind':'ordinary_warm_prefill_screen','observer':False,
            'executable_path':str(exe),'executable_sha256':'b'*64,'metallib_path':str(library),
            'metallib_sha256':'c'*64,'config_path':str(self.conf),'config_sha256':hashes[str(self.conf)],
            'effective_myds_environment':{'sha256':'a'*64},'post_run_myds_environment':{'sha256':'a'*64},
            'mask_skip_expected':'on','model_path':str(self.model),'model_bytes':identity['model']['size'],
            'model_header_sha256':identity['model']['header_sha256'],'phases':phases,
            'mask_skip_warm_counters':{'failures':0,'encoded':1}}
        (output/'summary.json').write_text(json.dumps(summary))
        m.validate_arm_summary(identity,0)
        (output/phases[1]['output']['path']).write_bytes(b'no')
        with self.assertRaisesRegex(RuntimeError,'output payload mismatch'):
            m.validate_arm_summary(identity,0)

    def test_restore_uses_captured_fds_priority_and_exact_process(self):
        self.args.min_available_gib=1
        self.w.identity['fds'][0]['offset']=0
        process={'executable':'/fixture/myds','argv':['myds','--port','8187'],
                 'environment':{'PATH':'fixture'}}
        self.w.state.update(process, cwd=str(self.root), nice=5)
        child=types.SimpleNamespace(pid=777,poll=lambda:None)
        response=io.StringIO('{"status":"ok"}'); response.status=200
        rows=copy.deepcopy(self.w.identity['fds'])
        rows[1]['status']=rows[2]['status']=1
        rows[1]['flags']=0x1000a  # A written fd and a fresh fd have the same open mode.
        def query(*a):
            if a[0] != 'ps': return 'p777\nf1'
            return {'nice=':'5','uid=':str(os.geteuid()),'gid=':str(os.getegid())}[a[-1]]
        with mock.patch.object(self.w,'drain_experiment'), mock.patch.object(self.w,'wait_original_exit'), \
             mock.patch.object(self.w,'verify'), mock.patch.object(self.w,'persist'), \
             mock.patch.object(m,'check_available_memory') as memory, \
             mock.patch.object(m.base,'processes',side_effect=[[],[(777,'myds')]]), \
             mock.patch.object(m.subprocess,'Popen',return_value=child) as popen, \
             mock.patch.object(m.base,'process_arguments',return_value=process), \
             mock.patch.object(m.base,'process_cwd',return_value=self.root), \
             mock.patch.object(m.base,'query',side_effect=query), \
             mock.patch.object(m,'fd_info',side_effect=lambda pid,fd:rows[fd]), \
             mock.patch.object(m.urllib.request,'urlopen',return_value=response):
            self.w.restore()
        kwargs=popen.call_args.kwargs
        self.assertEqual(kwargs['stdin'],self.w.stdio[0])
        self.assertEqual(kwargs['stdout'],self.w.stdio[1])
        self.assertEqual(kwargs['stderr'],self.w.stdio[1])
        self.assertEqual(kwargs['env'],process['environment'])
        with mock.patch.object(m.os,'setpriority') as priority: kwargs['preexec_fn']()
        priority.assert_called_once_with(os.PRIO_PROCESS,0,5)
        self.assertTrue(self.w.state['restored'])
        memory.assert_not_called()

    def test_owned_group_remaining_blocks_restore(self):
        self.w.experiment=types.SimpleNamespace(pid=234,poll=lambda:0)
        with mock.patch.object(self.w,'drain_experiment'), mock.patch.object(self.w,'wait_original_exit'), \
             mock.patch.object(self.w,'verify'), mock.patch.object(m.os,'killpg',return_value=None), \
             mock.patch.object(m.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError,'process group still exists'):
                self.w.restore()
            popen.assert_not_called()

    def run_cli_sequence(self, exit_codes, memory_error=False):
        arms=[{'role':role,'argv':['--run',str(self.model),str(i)]}
              for i,role in enumerate(('A1','B','A2'))]
        sequence=self.root/'sequence.json'
        import json
        sequence.write_text(json.dumps({'arms':arms}))
        script=['window','--pid','12345','--cwd',str(self.root),'--port','8187',
          '--stdin-path',str(self.conf),'--log-path',str(self.log),'--config',str(self.conf),
          '--library',str(self.args.library),'--model',str(self.model),'--expected-nice','5',
          '--harness',str(self.root/'bench_indexer_mask_warm'),'--harness-sha256','0'*64,
          '--experiment-library',str(self.root/'myds.metallib'),
          '--min-available-gib','1','--sequence-json',str(sequence), '--execute',
          '--expected-identity','0'*64,'--root',str(self.args.root),'--lock',str(self.args.lock),
          '--acknowledge-idle-snapshot-only']
        identity={'harness':{'frozen':'fixture'},'process':{'fixture':'process'},'start':'start'}
        calls=[]
        def preflight(w):
            w.identity=identity; w.a.root.mkdir(); w.root_owned=True
            w.state.update(environment={'MYDS_FIXTURE':'1'},cwd=str(self.root))
        def launch(command,**kwargs):
            calls.append((command,kwargs)); code=exit_codes[len(calls)-1]
            return types.SimpleNamespace(pid=200+len(calls),wait=lambda:code,poll=lambda:code)
        self.restore_attempted=[]
        def restore(w): self.restore_attempted.append(True); w.state.update(restored=True)
        with mock.patch('sys.argv',script), \
             mock.patch.object(m.ControlledWindow,'preflight',preflight), \
             mock.patch.object(m.ControlledWindow,'verify_stdio'), \
             mock.patch.object(m.ControlledWindow,'verify'), \
             mock.patch.object(m.ControlledWindow,'wait_original_exit'), \
             mock.patch.object(m.ControlledWindow,'restore',restore), \
             mock.patch.object(m,'snapshot',return_value=identity), \
             mock.patch.object(m,'active_clients'), \
             mock.patch.object(m,'harness_identity',return_value=identity['harness']), \
             mock.patch.object(m,'validate_arm_summary'), \
             mock.patch.object(m,'compare_arm_outputs'), \
             mock.patch.object(m,'check_available_memory',side_effect=(RuntimeError('low memory') if memory_error else None)), \
             mock.patch.object(m.base,'processes',return_value=[]), \
             mock.patch.object(m.base,'process_arguments',return_value=identity.get('process')), \
             mock.patch.object(m.base,'query',return_value=identity.get('start')), \
             mock.patch.object(m.subprocess,'Popen',side_effect=launch), \
             mock.patch.object(m.os,'kill'), \
             mock.patch.object(m.os,'killpg',side_effect=ProcessLookupError()), \
             mock.patch.object(m.signal,'signal'), mock.patch.object(m.signal,'getsignal'), \
             contextlib.redirect_stdout(io.StringIO()):
            code=m.main()
        return code,calls

    def test_cli_initializes_and_runs_three_arms_in_order(self):
        code,calls=self.run_cli_sequence([0,0,0])
        self.assertEqual(code,0)
        self.assertEqual([call[0][-1] for call in calls],['0','1','2'])
        self.assertTrue(all(call[1]['env']=={'MYDS_FIXTURE':'1'} for call in calls))
        import json
        summary=json.loads((self.args.root/'summary.json').read_text())
        self.assertTrue(summary['screening_only'])
        self.assertTrue(summary['exclusivity_unproven'])

    def test_cli_stops_after_first_failed_arm(self):
        code,calls=self.run_cli_sequence([0,7,0])
        self.assertEqual(code,7)
        self.assertEqual([call[0][-1] for call in calls],['0','1'])

    def test_experiment_memory_gate_failure_still_attempts_restore(self):
        code,calls=self.run_cli_sequence([0,0,0],memory_error=True)
        self.assertEqual(code,1)
        self.assertEqual(calls,[])
        self.assertEqual(self.restore_attempted,[True])


if __name__ == '__main__': unittest.main()
