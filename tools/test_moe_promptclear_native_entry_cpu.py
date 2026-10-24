#!/usr/bin/env python3
"""CPU-only actual-main entry checks; never link or initialize the inference core.

The three actual tool mains are compiled, rather than copying their guards.
Only the two pure environment parser bodies are extracted from the candidate
core. A model-load stub starts with both initialized globals at zero, then
initializes them from those real parsers. Sentinels terminate before any model,
context, observer callback, Metal initialization, sampler, or generation work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'full':'bench_moe_workspace_fullmodel_promptclear.c',
    'cancel':'test_moe_workspace_cancel_promptclear.c',
    'recovery':'test_moe_workspace_recovery_promptclear.c',
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def body(text,name):
    m=re.search(r'\bint\s+'+re.escape(name)+r'\s*\(void\)\s*\{',text)
    if not m:raise ValueError('missing actual pure parser '+name)
    depth=1;i=m.end()
    while depth:
        if text[i]=='{':depth+=1
        elif text[i]=='}':depth-=1
        i+=1
    return text[m.start():i]

STUB=r'''
#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_server_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int initialized_workspace=0,initialized_clear=0;
PARSERS
int myds_metal_prefill_moe_workspace_enabled(void){return initialized_workspace;}
int myds_metal_prefill_prompt_view_clear_enabled(void){return initialized_clear;}
bool parse_chat_prompt(const char *body,char *out,size_t cap){
    size_t n=strlen(body);if(n>=cap)return false;memcpy(out,body,n+1);return true;
}
myds_model_t *myds_model_load(const char *path){
    (void)path;fprintf(stderr,"CPU_STUB_MODEL_LOAD globals=%d/%d\n",initialized_workspace,initialized_clear);fflush(stderr);
    if(getenv("CPU_ENTRY_STOP_AT_LOAD"))_Exit(73);
    initialized_workspace=myds_metal_prefill_moe_workspace_parse_mode();
    initialized_clear=myds_metal_prefill_prompt_view_clear_parse_mode();
    const char *m=getenv("CPU_ENTRY_MISMATCH");
    if(m&&!strcmp(m,"workspace"))initialized_workspace^=1;
    if(m&&!strcmp(m,"clear"))initialized_clear^=1;
    static myds_model_t model;model.v41=(myds_v41_model_t *)(uintptr_t)1;return &model;
}
void myds_model_free(myds_model_t *model){(void)model;}
static void accepted(void){fputs("CPU_STUB_POSTLOAD_ACCEPT\n",stderr);fflush(stderr);_Exit(74);}
myds_context_t *myds_context_create(myds_model_t *m,const myds_config_t *c){(void)m;(void)c;accepted();return NULL;}
int myds_v41_prompt_workspace_growth_observer_set(myds_v41_prompt_growth_observer_t fn,void *o){(void)o;if(fn)accepted();return 1;}
int myds_v41_moe_workspace_growth_observer_set(myds_v41_moe_growth_observer_t fn,void *o){(void)fn;(void)o;return 1;}
int myds_metal_init(const char *p){(void)p;fputs("FORBIDDEN_METAL_INIT\n",stderr);_Exit(99);}
'''
LAUNCHER=r'''
#include <dlfcn.h>
#include <stdio.h>
int main(int argc,char **argv){
    if(argc<2)return 98;void *h=dlopen(argv[1],RTLD_LAZY|RTLD_LOCAL);
    if(!h){fprintf(stderr,"dlopen: %s\n",dlerror());return 98;}
    int (*entry)(int,char **)=dlsym(h,"main");if(!entry)return 98;
    return entry(argc-1,argv+1);
}
'''

def run(args):
    stage=Path(args.stage).resolve();core=stage/'src/myds_metal.m'
    actual=core.read_text();parsers='\n'.join(body(actual,n) for n in
        ('myds_metal_prefill_moe_workspace_parse_mode','myds_metal_prefill_prompt_view_clear_parse_mode'))
    records=[];builds=[]
    with tempfile.TemporaryDirectory(prefix='myds-promptclear-entry-cpu-') as tmp:
        t=Path(tmp);(t/'stub.c').write_text(STUB.replace('PARSERS',parsers));(t/'launcher.c').write_text(LAUNCHER)
        subprocess.run(['cc',str(t/'launcher.c'),'-o',str(t/'entry')],check=True,capture_output=True,text=True)
        request=t/'request.json';request.write_text('{"messages":[{"role":"user","content":"CPU entry only"}]}')
        compiled={}
        for version,base in [('new',ROOT/'tools'),('old',Path(args.before).resolve())]:
            for kind,filename in SOURCES.items():
                for gold in ([False,True] if kind=='full' and version=='new' else [False]):
                    label=f'{version}-{kind}'+('-gold' if gold else '')
                    dylib=t/(label+'.dylib');cmd=['cc','-O0','-dynamiclib','-Wl,-undefined,dynamic_lookup',
                        '-Wno-unused-function','-I'+str(stage/'include'),'-I'+str(ROOT/'tools')]
                    if not gold:cmd+=['-DMYDS_BENCH_MOE_WS_API=1']
                    cmd +=[str(base/filename),str(t/'stub.c'),'-o',str(dylib)]
                    p=subprocess.run(cmd,capture_output=True,text=True)
                    if p.returncode:raise RuntimeError(label+' compile: '+p.stderr)
                    linked=subprocess.run(['otool','-L',str(dylib)],capture_output=True,text=True,check=True).stdout
                    if 'Metal.framework' in linked:raise ValueError('must not link Metal')
                    builds.append({'label':label,'source_sha256':sha(base/filename),'compiled':True,
                        'metal_framework_linked':False,'compiler_stderr':p.stderr,'linked_libraries':linked.splitlines()[1:]})
                    compiled[label]=dylib
        case_index=0
        def check(label,w,c,expect,*,stop=True,mismatch=None,name=None):
            nonlocal case_index
            out=t/f'case-{case_index:03}';case_index+=1
            kind='cancel' if '-cancel' in label else ('recovery' if '-recovery' in label else 'full')
            command=[str(t/'entry'),str(compiled[label]),'--run','--model','/CPU-STUB-NO-MODEL','--warm-request',str(request),'--out-dir',str(out)]
            if kind=='full':command+=['--grow-request',str(request),'--retain-request',str(request),'--phase','probe','--repeats','0']
            elif kind=='cancel':command+=['--cancel-request',str(request),'--recover-request',str(request),'--scenario','deterministic','--checkpoint','1','--temperature','0','--recover-tokens','8']
            else:command+=['--failure-request',str(request),'--recover-request',str(request),'--scenario','prompt-allocation-first','--temperature','0','--recover-tokens','8']
            env={k:v for k,v in os.environ.items() if not k.startswith('MYDS_') and not k.startswith('CPU_ENTRY_')}
            for key,value in [('MYDS_PREFILL_MOE_WORKSPACE',w),('MYDS_PREFILL_PROMPT_VIEW_CLEAR',c)]:
                if value is not None:env[key]=value
            if stop:env['CPU_ENTRY_STOP_AT_LOAD']='1'
            if mismatch:env['CPU_ENTRY_MISMATCH']=mismatch
            p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
            load='CPU_STUB_MODEL_LOAD globals=0/0' in p.stderr
            accepted='CPU_STUB_POSTLOAD_ACCEPT' in p.stderr
            good=(p.returncode==expect and ('FORBIDDEN_METAL_INIT' not in p.stderr) and
                (load if expect in (73,74) or mismatch else not load) and (accepted==(expect==74)))
            records.append({'name':name or f'{label} w={w!r} clear={c!r} mismatch={mismatch}',
                'returncode':p.returncode,'expected_returncode':expect,'model_load_stub_reached':load,
                'postload_guard_passed':accepted,'stderr':p.stderr,'passed':good})
            if not good:raise AssertionError(json.dumps(records[-1]))
        for kind in SOURCES:
            label='new-'+kind
            for w,c in [(None,None),('0','0'),('1','1')]:check(label,w,c,73)
            check(label,'1','1',74,stop=False,name=label+' initialized globals start zero; valid1/1 reaches postload')
            for mismatch in ('workspace','clear'):check(label,'1','1',1,stop=False,mismatch=mismatch)
            for w,c in [('0','1'),(None,'1')]:check(label,w,c,2 if kind=='full' else 1)
            for invalid in ('','2','01','garbage'):
                check(label,invalid,'0',2 if kind=='full' else 1)
                check(label,'0',invalid,2 if kind=='full' else 1)
            check('old-'+kind,'1','1',2 if kind=='full' else 1,name='reproduce old '+kind+' valid1/1 wrongly rejected before load')
        check('new-full','1','0',73,name='causal workspace1 clear0 remains valid')
        check('new-full','1','0',74,stop=False)
        for kind in ('cancel','recovery'):check('new-'+kind,'1','0',1)
        for w,c in [(None,None),('0','0')]:check('new-full-gold',w,c,73)
        for w,c in [('1','1'),('0','1'),('1','0'),('garbage','0'),('0','garbage')]:check('new-full-gold',w,c,2)
    return {'schema_version':1,'scope':'actual_native_main_CPU_entry_stubs_only','passed':all(r['passed'] for r in records),
        'controls_count':len(records),'controls':records,'builds':builds,'core_parser_source':str(core),
        'core_parser_source_sha256':sha(core),'actual_extracted_parser_bodies_sha256':hashlib.sha256(parsers.encode()).hexdigest(),
        'inference_core_linked':False,'metal_initialized':False,'model_file_opened':False,'gpu_work_submitted':False,
        'limits':['Renderer/model load/context/observer registration are CPU stubs; this proves entry and post-initialization predicates, not model execution or GPU correctness.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--before',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    report=run(a);Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'passed':report['passed'],'controls':report['controls_count'],'out':a.out}))
if __name__=='__main__':main()
