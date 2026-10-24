#!/usr/bin/env python3
"""Frozen same-build prompt lifetime intervention; no performance or adoption."""
import argparse,json,os,signal,sys
from pathlib import Path
sys.dont_write_bytecode=True
import run_moe_workspace_window_promptclear as workspace
import review_moe_prompt_lifetime_diagnostic as review
need,sha,write_json=workspace.need,workspace.sha,workspace.write_json
NATIVE='bin/bench_moe_workspace_fullmodel_promptclear'
CORE=('src/myds_v41.c','src/myds_metal.m','include/myds_internal.h','include/myds_metal.h')
ORDER=(('C0a',0),('C1',1),('C0b',0))
FORBIDDEN=('myds_metal_moe_workspace_test_control','myds_v41_moe_workspace_probe_','myds_v41_moe_workspace_cancel_control','myds_v41_prompt_workspace_test_','myds_v41_prompt_workspace_probe_')
class DiagnosticWindow(workspace.WorkspaceWindow):
    def preflight(self):
        super().preflight()
        p=self.plan['causal_diagnostic']
        need(self.plan['implementation_version']==4 and self.plan['prompt_view_clear_flag']==workspace.CLEAR_FLAG and p['order']==['C0a','C1','C0b'] and p['workspace_settings']==[1,1,1] and p['clear_settings']==[0,1,0] and p['total_generations']==9 and p['max_new_tokens']==8 and p['performance_measured']is False and p['extra_gpu_waits_or_commits']==0,'Frozen diagnostic scope differs')
        for name in (NATIVE,'tools/bench_moe_workspace_fullmodel_promptclear.c','tools/moe_prompt_lifetime_diagnostic.inc','tools/review_moe_prompt_lifetime_diagnostic.py','tools/run_moe_prompt_lifetime_diagnostic_window.py'):
            need(str(self.a.stage/name)in self.input_hashes,'Required actual source/native not frozen: '+name)
        need(str(Path(__file__).resolve())in self.input_hashes and str(Path(review.__file__).resolve())in self.input_hashes,'Use frozen actual controller/reader')
        symbols=workspace.safety.command(['/usr/bin/nm','-gU',str(self.a.stage/NATIVE)])
        need(not any(s in symbols for s in FORBIDDEN),'Normal diagnostic binary exposes TESTING controls')
        need(all(s in symbols for s in ('myds_v41_prompt_workspace_growth_observer_set','myds_metal_deferred_buffer_observer_set')),'Normal registration APIs absent')
        value=json.loads(self.a.component_review.read_text());workspace.validate_component_prerequisite(value,self.a.stage,self.summary,self.before)
        workspace.freeze_independent_long(value,'long_report',self.input_hashes)
        self.input_hashes[str(self.a.component_review)]=sha(self.a.component_review)
        self.prerequisites={'component_gate_pass':dict(path=str(self.a.component_review),sha256=sha(self.a.component_review))}
        self.requests={'setup':self.a.inputs/'pf7k.json','grow':self.a.inputs/'pf16k.json','retain':self.a.inputs/'pf7k.json'}
        self.native_identity={'candidate':{n:sha(self.a.stage/n)for n in (*CORE,NATIVE,'bin/myds.metallib')}}
        self.arms=[];self.read_arms=[]
        self.summary.update(mode='causal_diagnostic',implementation_version=4,performance_measured=False,fullmodel_run=False,correctness_pass=False,causal_lifetime_gate_pass=False,native_identity=self.native_identity,prerequisites=self.prerequisites)
        write_json(self.a.out/'normal-symbols.json',dict(binary_sha256=sha(self.a.stage/NATIVE),forbidden_present=False,nm_stdout=symbols))
        write_json(self.a.out/'stage-inputs.json',self.input_hashes)
    def component(self):
        self.summary.update(fullmodel_run=True,performance_measured=False)
        (self.a.out/'arms').mkdir();(self.a.out/'arm-environments').mkdir()
        for label,clear in ORDER:
            directory=self.a.out/'arms'/label;need(not directory.exists(),'Native original output must be new')
            env=dict(self.environment,**{workspace.FLAG:'1',workspace.CLEAR_FLAG:str(clear)})
            expected=dict(self.before['effective_myds_env'],**{workspace.FLAG:'1',workspace.CLEAR_FLAG:str(clear)})
            need({k:v for k,v in env.items()if k.startswith('MYDS_')}==expected,'All51 production settings plus two explicit selectors required')
            write_json(self.a.out/'arm-environments'/(label+'.json'),expected)
            self.run_command(label,[self.a.stage/NATIVE,'--run','--model',self.a.model,'--warm-request',self.requests['setup'],'--grow-request',self.requests['grow'],'--retain-request',self.requests['retain'],'--out-dir',directory,'--phase','probe','--repeats','0','--diagnostic-prompt-observer'],env=env,timeout=3600)
            arm=review.read_arm(directory,self.requests,expected,clear,expected_model=self.a.model)
            need(Path(arm['metadata']['actual_metallib_path']).resolve(strict=True)==(self.a.stage/'bin/myds.metallib').resolve(strict=True),'Actual first-priority library differs')
            self.read_arms.append(arm)
            self.arms.append(dict(label=label,workspace_mode=1,clear_mode=clear,directory=str(directory),native_binary_sha256=sha(self.a.stage/NATIVE),metallib_sha256=sha(self.a.stage/'bin/myds.metallib'),native_summary_sha256=sha(directory/'summary.jsonl')))
            write_json(directory/'controller-review.json',arm);write_json(self.a.out/'arms.json',self.arms)
        self.gate=review.causal_gate(*self.read_arms)
        need(self.gate.get('evidence_valid')is True and type(self.gate.get('causal_lifetime_gate_pass'))is bool,'Reader must explicitly validate causal evidence and decide')
        self.summary.update(correctness_pass=True,causal_lifetime_gate_pass=self.gate['causal_lifetime_gate_pass'],generations_completed=9,causal=self.gate,diagnostic_native_complete=True)
    def run(self):
        code=super().run()
        if self.a.prepare:return code
        complete=code==0 and 'error'not in self.summary and self.summary.get('diagnostic_native_complete')is True and self.summary.get('restoration',{}).get('restored')is True and len(getattr(self,'arms',[]))==3
        passed=complete and self.summary.get('causal_lifetime_gate_pass')is True
        status=('causal_passed_review_required'if passed else'causal_rejected_lifetime_evidence')if complete else'failed'
        self.summary.update(status=status,evidence_valid=complete,causal_lifetime_gate_pass=passed,production_adoption=False)
        if hasattr(self,'gate'):write_json(self.a.out/'causal-detail.json',self.gate)
        report=dict(schema_version=1,mode='causal_diagnostic',implementation_version=4,evidence_valid=complete,causal_lifetime_gate_pass=passed,correctness_pass=self.summary.get('correctness_pass',False),performance_measured=False,production_adoption=False,plan_sha256=self.summary.get('plan_sha256'),production_environment=self.before.get('effective_myds_env')if self.before else None,model=self.summary.get('model'),model_header_sha256=self.summary.get('model_header_sha256'),native_identity=getattr(self,'native_identity',{}),prerequisites=getattr(self,'prerequisites',{}),arms=getattr(self,'arms',[]),detail=dict(path='causal-detail.json',sha256=sha(self.a.out/'causal-detail.json'))if hasattr(self,'gate')else None,restoration=self.summary.get('restoration'))
        if 'error'in self.summary:report['error']=self.summary['error']
        write_json(self.a.out/'causal-review.json',report);write_json(self.a.out/'summary.json',self.summary)
        self.event('diagnostic_decision_complete',status=status,evidence_valid=complete,causal_lifetime_gate_pass=passed,production_adoption=False)
        return 0 if complete else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','gold-stage','model','out','plan','inputs','component-review'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=8187);m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true');a=p.parse_args();need(sys.platform=='darwin','macOS required')
    for n in ('stage','gold_stage','model','out','plan','inputs','component_review'):
        v=getattr(a,n);need(v.is_absolute(),'Absolute paths required');setattr(a,n,v.resolve())
    need(all(Path('/tmp').resolve()in v.parents for v in (a.stage,a.gold_stage)),'Separate tmp stages required')
    a.out.mkdir(parents=True,exist_ok=False);os.chmod(a.out,0o700);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:os.setsid()
    except PermissionError:pass
    with(a.out/'runner.log').open('ab',buffering=0)as log,open(os.devnull,'rb')as null:os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.dup2(null.fileno(),0)
    return DiagnosticWindow(a).run()
if __name__=='__main__':raise SystemExit(main())
