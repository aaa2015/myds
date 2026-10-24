#!/usr/bin/env python3
"""Provider component plus actual local-growth functional gate in a safe window.

The new base owns the unique pre-data plan SHA, lock, monitoring and finally
restoration. Original component GPU timing stays90CB; localgrow host work and
lifetime checkpoints are recorded separately before that timing phase.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
sys.dont_write_bytecode=True
import run_moe_workspace_window_localgrow as workspace
need,sha,write_json=workspace.need,workspace.sha,workspace.write_json


class LocalgrowComponentWindow(workspace.WorkspaceWindow):
    def preflight(self):
        super().preflight()
        required=('bin/bench_moe_workspace_localgrow','bin/test_moe_workspace_localgrow_allocations',
                  'bin/bench_moe_workspace_localgrow_transactions','bin/bench_moe_workspace_failures',
                  'tools/run_moe_workspace_localgrow_component_window.py',
                  'tools/review_moe_workspace_localgrow_artifacts.py',
                  'tools/moe_workspace_probe_layout_localgrow.inc','tools/moe_workspace_localgrow_probes.mk')
        need(str(Path(__file__).resolve()) in self.input_hashes,'Execute the actual frozen component controller')
        need(all(str(self.a.stage/p) in self.input_hashes for p in required),'Every actual localgrow tool/source must be frozen')
        need(self.plan['implementation_version']==3 and self.plan['component']['localgrow_functional_required'] is True and
             self.plan['component']['warmup']==1 and self.plan['component']['formal_samples']==9 and
             self.plan['component']['maximum_absolute_time_control_drift_percent']==2 and
             self.plan['component']['maximum_conservative_gpu_time_regression_percent']==2 and
             self.plan['component']['minimum_actual_device_allocation_reduction_bytes']==536870912,
             'Unique v3 plan must preserve the original component gates')
        self.summary.update(component='actual_prefill_moe_localgrow_component',implementation_version=3,
                            localgrow_host_and_GPU_timing_are_separate=True)

    def component(self):
        env=dict(self.environment,**{workspace.FLAG:'0'})
        self.run_command('allocations',[self.a.stage/'bin/test_moe_workspace_localgrow_allocations','--run',
                          '--out-dir',self.a.out/'allocations'],env=env)
        records=[json.loads(line) for line in (self.a.out/'allocations/records.jsonl').read_text().splitlines() if line.strip()]
        need(records[-1].get('all_pass') is True and records[-1].get('implementation_version')==3,
             'Actual factory/localgrow allocation diagnostics must complete')
        capture=self.a.out/'raw-capture'
        self.run_command('capture',[self.a.stage/'bin/capture_moe_workspace',self.a.model,'--tokens-file',
                          self.a.inputs/'capture-tokens.txt','--out-dir',capture],env=env,timeout=2400)
        common=[self.a.stage/'bin/bench_moe_workspace_localgrow','--model',self.a.model,
                '--capture-json',capture/'capture.json']
        self.run_command('host',[*common,'--out-dir',self.a.out/'host','--host-only'],env=env)
        self.run_command('correctness',[*common,'--out-dir',self.a.out/'correctness','--correctness-only'],env=env,timeout=1800)
        self.run_command('failures',[self.a.stage/'bin/bench_moe_workspace_failures','--model',self.a.model,
                          '--capture-json',capture/'capture.json','--out-dir',self.a.out/'failures','--run'],env=env,timeout=1800)
        self.run_command('failure-review',['/usr/bin/python3',self.a.stage/'tools/test_moe_workspace_failures.py',
                          '--root',self.a.out/'failures','--out',self.a.out/'failure-review.json'],env=env)
        failure=json.loads((self.a.out/'failure-review.json').read_text())
        need(failure.get('evidence_valid') is True and failure.get('functional_failure_gate_pass') is True,
             'All original actual PSO/fallback functional controls must pass')
        self.run_command('localgrow',[self.a.stage/'bin/bench_moe_workspace_localgrow_transactions','--model',self.a.model,
                          '--capture-json',capture/'capture.json','--out-dir',self.a.out/'localgrow','--run'],env=env,timeout=1800)
        self.run_command('localgrow-review',['/usr/bin/python3',self.a.stage/'tools/review_moe_workspace_localgrow_artifacts.py',
                          '--root',self.a.out,'--model',self.a.model,'--functional-only',
                          '--out',self.a.out/'localgrow-review.json'],env=env,timeout=1800)
        functional=json.loads((self.a.out/'localgrow-review.json').read_text())
        need(functional.get('evidence_valid') is True and functional.get('functional_localgrow_gate_pass') is True,
             'Independent complete localgrow/rollback/raw-output gate must precede GPU timing')
        prior=self.a.out/'correctness/artifact-manifest.json'
        need(json.loads(prior.read_text()).get('full_f32_bit_exact') is True,'All real-layer correctness is required before timing')
        self.run_command('performance',[*common,'--out-dir',self.a.out/'performance','--correctness-manifest',prior,
                          '--correctness-manifest-sha256',sha(prior)],env=env,timeout=1800)
        self.summary.update(component_executed=True,component_review_required=True,fullmodel_run=False,
                            localgrow_functional_pass=True,localgrow_growth_time_is_not_gpu_execution_time=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','gold-stage','model','out','plan','inputs'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,default=8187)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args();need(sys.platform=='darwin','macOS required')
    for name in ('stage','gold_stage','model','out','plan','inputs'):
        path=getattr(args,name);need(path.is_absolute(),'Absolute paths required');setattr(args,name,path.resolve())
    need(all(Path('/tmp').resolve() in p.parents for p in (args.stage,args.gold_stage)),'Separate tmp stages required')
    args.out.mkdir(parents=True,exist_ok=False);os.chmod(args.out,0o700);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with (args.out/'runner.log').open('ab',buffering=0) as log,open(os.devnull,'rb') as null:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    return LocalgrowComponentWindow(args).run()


if __name__=='__main__':raise SystemExit(main())
