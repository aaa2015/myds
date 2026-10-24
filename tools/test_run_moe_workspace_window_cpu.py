#!/usr/bin/env python3
"""CPU orchestration controls; never capture/stop/restore a real service."""
import io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
import run_moe_workspace_window as module


def main():
    controls=[]
    with tempfile.TemporaryDirectory(prefix='myds-moe-ws-orchestration-cpu-') as temp:
        root=Path(temp)
        class Fake(module.WorkspaceWindow):
            def __init__(self,args,failure=None):super().__init__(args);self.calls=[];self.failure=failure
            def event(self,kind,**data):self.calls.append(kind)
            def preflight(self):
                self.calls.append('preflight');self.before={'fixture_only':True};self.summary['model']={};self.input_hashes={}
                if self.failure=='preflight':raise RuntimeError('CPU injected preflight')
            def stop(self):self.calls.append('stop');self.stopped=True
            def component(self):
                self.calls.append('component')
                if self.failure=='component':raise RuntimeError('CPU injected component')
                if self.failure=='monitor':self.fault.set()
                if self.failure=='frozen_input':self.input_hashes={str(root/'absent-frozen-input.c'):'0'*64}
            def monitor(self):pass
            def unchanged(self):self.calls.append('unchanged')
            def drain(self):self.calls.append('drain')
            def restore(self):
                self.calls.append('restore')
                if self.failure=='restore':raise RuntimeError('CPU injected restore')
                return {'restored':True,'fixture_only':True}
        for index,(prepare,failure,expected) in enumerate([(True,None,0),(False,None,0),(False,'preflight',1),(False,'component',1),(False,'restore',1),(False,'monitor',1),(False,'frozen_input',1)]):
            out=root/str(index);out.mkdir();w=Fake(SimpleNamespace(out=out,prepare=prepare),failure)
            with patch.object(module,'open',lambda *a,**k:io.StringIO(),create=True),patch.object(module.fcntl,'flock'),patch.object(module.signal,'signal'):
                code=w.run()
            assert code==expected,(index,code,w.summary)
            if prepare or failure=='preflight':assert 'stop' not in w.calls and 'restore' not in w.calls
            else:assert w.calls.index('stop')<w.calls.index('drain')<w.calls.index('restore')
            assert json.loads((out/'summary.json').read_text())['production_adoption'] is False
            controls.append({'case':str((prepare,failure)),'passed':True,'operations':w.calls})
        original=module.safety.command
        for name in ['capture_moe_workspace','test_moe_workspace_allocations','test_moe_workspace_cancel','bench_moe_workspace']:
            w=module.WorkspaceWindow(SimpleNamespace())
            with patch.object(module.safety,'command',lambda *a:'987654 1 /tmp/'+name):
                try:w.scan()
                except RuntimeError:pass
                else:raise AssertionError('Competing diagnostic accepted: '+name)
                w.allowed.add(987654);assert w.scan()[0]['permitted'] is True
            controls.append({'case':'external_'+name+'_rejected_allowed_child_accepted','passed':True})
    report=dict(schema_version=1,scope='CPU mock orchestration only; no service/Metal/model operations',all_pass=True,controls=controls,gpu_or_model_executed=False)
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
