#!/usr/bin/env python3
"""Compile/run the real Shared Q4 host helpers with CPU Metal stubs only.
No MTLDevice creation, command submission, model load or GPU execution.
Also proves both existing shader files are unchanged prefixes and reconstructs
exact GU source after reversing only the declared specialization changes.
"""
import argparse, hashlib, json, os, struct, subprocess, tempfile
from pathlib import Path
R = Path(__file__).resolve().parents[1]
BASE = '467e5ac8a063afad026328e5eea83de20f5bf4e2'
def sha(b): return hashlib.sha256(b).hexdigest()
def need(x, why):
    if not x: raise AssertionError(why)
def function(s, name):
    p=s.index(name); a=s.rfind('\n',0,p)+1; b=s.index('{',p); d=1; e=b+1
    while d:
        d += (s[e]=='{')-(s[e]=='}'); e+=1
    return s[a:e]
def source_check():
    gu=(R/'metal/myds_v41_moe.metal').read_text(); down=(R/'metal/myds_q4_v41.metal').read_text()
    for name,new in [('metal/myds_v41_moe.metal',gu),('metal/myds_q4_v41.metal',down)]:
        old=subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=R).decode()
        need(new.startswith(old),f'original shader prefix changed: {name}')
    old=function(gu,'void kernel_myds_v41_shared_gate_up_swiglu_q4_impl(')
    new=function(gu,'static inline void myds_shared_q4_v41_nb_gu_impl(')
    new=new.replace('static inline void myds_shared_q4_v41_nb_gu_impl(', 'void kernel_myds_v41_shared_gate_up_swiglu_q4_impl(')
    new=new.replace('    constexpr short NR0 = 2;\n','')
    new=new.replace('const int nb = 20;', 'const int nb = (int)args.in_dim / QK_K;')
    new=new.replace('    const int first_row = (tgpig.x * NSG + sgitg) * NR0;','    const int first_row = (tgpig.x * NSG + sgitg) * NR0;\n    if (first_row >= (int)args.out_dim) return;')
    new=new.replace('const uint offset0 = uint(first_row + row) * 2880u;', 'const uint64_t offset0 = (uint64_t)(first_row + row) * args.row_bytes;\n        const bool active = (first_row + row < (int)args.out_dim);')
    new=new.replace('src_gate + offset0','src_gate + (active ? offset0 : 0)').replace('src_up   + offset0','src_up   + (active ? offset0 : 0)')
    new=new.replace('        FOR_UNROLL (short row = 0; row < NR0; ++row) {\n','        FOR_UNROLL (short row = 0; row < NR0; ++row) {\n            if (first_row + row >= (int)args.out_dim) break;\n')
    new=new.replace('for (short row = 0; row < NR0; ++row) {\n        float tot_g', 'for (short row = 0; row < NR0 && first_row + row < (int)args.out_dim; ++row) {\n        float tot_g')
    need(new==old,'exact GU specialization reconstruction failed')
    need('myds_q4_v41_body<5120,2304,true,true>(args,w,x,y,group,lane,sg)' in down,'Down specialization identity')
    return {'base_head':BASE,'gu_original_helper_sha256':sha(old.encode()),'gu_reconstructed_original_bytes_exact':True,'original_shader_prefixes_exact':True,'down_reuses_unchanged_fixed_nb_body':True}
def build_source():
    frozen=(R/'docs/benchmarks/2026-10-14-decode-round2/replay/q4-providers-cpu.m').read_text()
    prefix=frozen[:frozen.index('/* Included after both scalar GEMV argument ABIs.')]
    prefix=prefix.replace('enum {DC_OVERLAP,DC_IQ2,DC_Q4,DC_Q4_NB,DC_Q4_HEAD_REFERENCE,DC_COUNT}', 'enum {DC_OVERLAP,DC_IQ2,DC_Q4,DC_Q4_NB,DC_Q4_HEAD_REFERENCE,DC_SHARED_Q4_NB,DC_COUNT}')
    prefix=prefix.replace('static NSUInteger got_scratch;', 'static NSUInteger got_scratch;')
    prefix=prefix.replace('static BOOL finish_ok=YES;', 'static BOOL finish_ok=YES; static id last_pipeline;')
    prefix=prefix.replace('-(void)setComputePipelineState:(id)p {(void)p;}', '-(void)setComputePipelineState:(id)p {last_pipeline=p;}')
    prefix=prefix.replace('assert(n==112&&i==0);memcpy(&encoded_args,p,n);', 'assert((n==112||n==16)&&i==0); if(n==112)memcpy(&encoded_args,p,n);')
    prefix=prefix.replace('@interface MockLibrary:NSObject', 'static BOOL missing_gu,missing_down;\n@interface MockLibrary:NSObject\n-(id)newFunctionWithName:(NSString*)n;')
    prefix=prefix.replace('@implementation MockLibrary\n', '''@implementation MockLibrary
-(id)newFunctionWithName:(NSString*)n {if(missing_gu&&[n containsString:@"shared_gate_up"])return nil; return [NSObject new];}
''')
    prefix=prefix.replace('{(void)n;(void)v;(void)e;return [NSObject new];}', '{(void)v;(void)e;if(missing_down&&[n containsString:@"shared_down"])return nil;return [NSObject new];}')
    extra='''
static BOOL g_shared_q4_v41_nb,g_auto_barrier,g_kp_on;
static id<MTLComputePipelineState>g_pso_shared_q4_v41_nb[2];
static id<MTLComputeCommandEncoder>g_batch_enc;
static int concurrent_mode=1;
int myds_metal_get_concurrent(void){return concurrent_mode;}
'''
    native=(R/'src/myds_decode_candidates.inc').read_text()
    metal=(R/'src/myds_metal.m').read_text()
    sig=function(metal,'static int routed_moe_scoped_impl(').split('{',1)[0]
    stub=sig+'{return finish_ok ? encode_shared_down_overlap(mock_encoder,shared_down) : 0;}\n'
    wrapper='\n'.join(function(metal,n) for n in ['static int shared_down_scoped_failure(', 'static int routed_moe_down_overlap_scoped_impl(', 'int myds_metal_routed_moe_down_overlap_scoped(', 'int myds_metal_routed_moe_down_overlap_nbhc_scoped('])
    begin=metal.index('    if (shared_q4_v41_nb) {',metal.index('int myds_metal_init(')); end=metal.index('\n    const int moe_q2_sum_cache',begin)
    scope='''static int actual_init_scope(int shared_q4_v41_nb,int moe_down_overlap,int moe_iq2_v41,
 int dense_q4_v41,int dense_q4_v41_nb,int dense_q4_v41_head_reference,int moe_q2_v41) {
'''+metal[begin:end]+'\nreturn 1;\n}\n'
    return prefix+extra+native+'\n'+stub+wrapper+'\n'+scope+MAIN
MAIN=r'''
static myds_metal_buffer_t B(uintptr_t p,size_t n){MockBuffer*b=[MockBuffer new];b.device=g_device;b.contents=(void*)p;b.length=n;return(myds_metal_buffer_t){(id)b,0,n,(void*)p};}
static unsigned negative_count,down_negative_count;
static void rejected_gu(myds_metal_buffer_t*out,myds_metal_buffer_t*gate,myds_metal_buffer_t*up,myds_metal_buffer_t*in,uint32_t rows,uint32_t width,float clamp){
 unsigned before=dispatches;myds_metal_decode_candidate_counters_t a,b;myds_metal_shared_q4_v41_nb_counters(&a,false);
 assert(!myds_metal_shared_q4_v41_nb_gate_up(out,gate,up,in,rows,width,clamp));myds_metal_shared_q4_v41_nb_counters(&b,false);
 assert(dispatches==before&&b.requests==a.requests+1&&b.encoded==a.encoded&&b.failures==a.failures+1);negative_count++;
}
int main(void){@autoreleasepool{
 MockDevice*d=[MockDevice new];d.maxThreadsPerThreadgroup=MTLSizeMake(1024,1024,64);d.maxThreadgroupMemoryLength=32768;g_device=(id)d;g_library=(id)[MockLibrary new];g_batch_cb=(id)[NSObject new];mock_encoder=(id)[MockEncoder new];g_batch_enc=mock_encoder;
 assert(decode_candidate_parse("MYDS_SHARED_Q4_V41_NB")==0);
 const char*invalid[]={"","2","true","01"," 1","1 ","-1"};for(unsigned i=0;i<7;i++){setenv("MYDS_SHARED_Q4_V41_NB",invalid[i],1);assert(decode_candidate_parse("MYDS_SHARED_Q4_V41_NB")==-1);}setenv("MYDS_SHARED_Q4_V41_NB","1",1);assert(decode_candidate_parse("MYDS_SHARED_Q4_V41_NB")==1);
 setenv("MYDS_MOE_HC_OVERLAP","1",1);assert(actual_init_scope(1,1,0,1,1,0,1));
 assert(!actual_init_scope(1,0,0,1,1,0,1));assert(!actual_init_scope(1,1,1,1,1,0,1));assert(!actual_init_scope(1,1,0,0,1,0,1));assert(!actual_init_scope(1,1,0,1,0,0,1));assert(!actual_init_scope(1,1,0,1,1,1,1));assert(!actual_init_scope(1,1,0,1,1,0,0));
 setenv("MYDS_MOE_HC_OVERLAP","0",1);assert(!actual_init_scope(1,1,0,1,1,0,1));setenv("MYDS_MOE_HC_OVERLAP","1",1);
 const char*conflicts[]={"MYDS_MOE_CONCURRENT","MYDS_MOE_PIPELINE","MYDS_SHARED_FUSE","MYDS_MOE_FUSE_EXPAND","MYDS_METAL_CONCURRENT"};for(unsigned i=0;i<5;i++){setenv(conflicts[i],"0",1);assert(!actual_init_scope(1,1,0,1,1,0,1));unsetenv(conflicts[i]);}
 MockPipeline*ref=[MockPipeline new];ref.device=g_device;ref.threadExecutionWidth=32;ref.maxTotalThreadsPerThreadgroup=512;g_pso_mul_mv_q4_K_dense_bf16=(id)ref;
 missing_gu=YES;assert(!decode_candidates_initialize(1,0,1,1,0,1));missing_gu=NO;missing_down=YES;assert(!decode_candidates_initialize(1,0,1,1,0,1));missing_down=NO;assert(decode_candidates_initialize(1,0,1,1,0,1));
 assert(myds_metal_shared_q4_v41_nb_enabled()==1&&myds_metal_shared_q4_v41_nb_selected_kernel(0)&&myds_metal_shared_q4_v41_nb_selected_kernel(1)&&!myds_metal_shared_q4_v41_nb_selected_kernel(2));
 myds_metal_buffer_t gu=B(0x100000000ull,6635520),up=B(0x110000000ull,6635520),in=B(0x120000000ull,20480),out=B(0x130000000ull,9216);
 assert(myds_metal_shared_q4_v41_nb_gate_up(&out,&gu,&up,&in,2304,5120,10));assert(last_pipeline==g_pso_shared_q4_v41_nb[0]&&got_grid.width==288&&got_threads.width==128&&got_threads.height==1);
 rejected_gu(NULL,&gu,&up,&in,2304,5120,10);rejected_gu(&out,&gu,&up,&in,2303,5120,10);rejected_gu(&out,&gu,&up,&in,2304,5121,10);rejected_gu(&out,&gu,&up,&in,2304,5120,NAN);rejected_gu(&out,&gu,&up,&in,2304,5120,-1);
 myds_metal_buffer_t readers[]={gu,up,in};for(unsigned i=0;i<3;i++){myds_metal_buffer_t alias=readers[i];alias.size=9216;rejected_gu(&alias,&gu,&up,&in,2304,5120,10);alias=B((uintptr_t)readers[i].ptr+16,9216);rejected_gu(&alias,&gu,&up,&in,2304,5120,10);}
 myds_metal_buffer_t bad=out;bad.size--;rejected_gu(&bad,&gu,&up,&in,2304,5120,10);bad=out;bad.ptr=(char*)bad.ptr+4;rejected_gu(&bad,&gu,&up,&in,2304,5120,10);bad=out;bad.offset=16;rejected_gu(&bad,&gu,&up,&in,2304,5120,10);
 for(unsigned i=0;i<3;i++){myds_metal_buffer_t b=readers[i];b.size--;rejected_gu(&out,i==0?&b:&gu,i==1?&b:&up,i==2?&b:&in,2304,5120,10);}
 g_auto_barrier=YES;rejected_gu(&out,&gu,&up,&in,2304,5120,10);g_auto_barrier=NO;g_kp_on=YES;rejected_gu(&out,&gu,&up,&in,2304,5120,10);g_kp_on=NO;concurrent_mode=0;rejected_gu(&out,&gu,&up,&in,2304,5120,10);concurrent_mode=1;
 id cb=g_batch_cb;g_batch_cb=nil;rejected_gu(&out,&gu,&up,&in,2304,5120,10);g_batch_cb=cb;g_shared_q4_v41_nb=NO;rejected_gu(&out,&gu,&up,&in,2304,5120,10);g_shared_q4_v41_nb=YES;
 MockPipeline*p=(id)g_pso_shared_q4_v41_nb[0];p.threadExecutionWidth=16;rejected_gu(&out,&gu,&up,&in,2304,5120,10);p.threadExecutionWidth=32;p.device=(id)[NSObject new];rejected_gu(&out,&gu,&up,&in,2304,5120,10);p.device=g_device;p.maxTotalThreadsPerThreadgroup=127;rejected_gu(&out,&gu,&up,&in,2304,5120,10);p.maxTotalThreadsPerThreadgroup=512;
 myds_metal_buffer_t dout=B(0x140000000ull,20480),dw=B(0x150000000ull,6635520);shared_down_job_t job={&dout,&dw,&out,true};assert(encode_shared_down_overlap(mock_encoder,&job));assert(last_pipeline==g_pso_shared_q4_v41_nb[1]&&got_grid.width==640&&got_threads.width==32&&got_threads.height==4&&got_scratch==256&&encoded_args.ne00==2304&&encoded_args.ne01==5120&&encoded_args.nr0==2);
 job.shared_q4_nb=false;assert(encode_shared_down_overlap(mock_encoder,&job));assert(last_pipeline==g_pso_mul_mv_q4_K_dense_bf16);
 myds_metal_buffer_t all[15];const size_t sizes[]={20480,20480,1167851520ull,1167851520ull,1486356480ull,24,24,55296,55296,55296,122880,20480,6635520,9216,9216};
 for(unsigned i=0;i<15;i++)all[i]=B(0x200000000ull+(uint64_t)i*0x100000000ull,sizes[i]);
 bool allowed=true;
#define DOWN_CALL(F) F(&all[0],&all[1],&all[2],&all[3],&all[4],&all[5],&all[6],&all[7],&all[8],&all[9],&all[10],5120,2304,384,6,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,10,allowed,&all[11],&all[12],&all[13],GGML_TYPE_Q4_K)
#define DOWN_REJECT() do {unsigned before=dispatches;myds_metal_decode_candidate_counters_t ca,cb;myds_metal_shared_q4_v41_nb_counters(&ca,false);assert(!DOWN_CALL(myds_metal_routed_moe_down_overlap_nbhc_scoped));myds_metal_shared_q4_v41_nb_counters(&cb,false);assert(dispatches==before&&cb.requests==ca.requests+1&&cb.encoded==ca.encoded&&cb.failures==ca.failures+1);down_negative_count++;}while(0)
 assert(DOWN_CALL(myds_metal_routed_moe_down_overlap_nbhc_scoped));assert(last_pipeline==g_pso_shared_q4_v41_nb[1]);assert(DOWN_CALL(myds_metal_routed_moe_down_overlap_scoped));assert(last_pipeline==g_pso_mul_mv_q4_K_dense_bf16);
 for(unsigned i=0;i<14;i++){all[i].size--;DOWN_REJECT();all[i].size++;}
 for(unsigned i=0;i<14;i++){if(i==11)continue;myds_metal_buffer_t old=all[11];all[11]=B((uintptr_t)all[i].ptr+16,20480);DOWN_REJECT();all[11]=old;}
 allowed=false;DOWN_REJECT();allowed=true;g_shared_q4_v41_nb=NO;DOWN_REJECT();g_shared_q4_v41_nb=YES;g_auto_barrier=YES;DOWN_REJECT();g_auto_barrier=NO;g_kp_on=YES;DOWN_REJECT();g_kp_on=NO;concurrent_mode=0;DOWN_REJECT();concurrent_mode=1;
 id dp=g_pso_shared_q4_v41_nb[1];g_pso_shared_q4_v41_nb[1]=nil;DOWN_REJECT();g_pso_shared_q4_v41_nb[1]=dp;
 finish_ok=NO;DOWN_REJECT();finish_ok=YES;
 myds_metal_decode_candidate_counters_t c;myds_metal_shared_q4_v41_nb_counters(&c,false);assert(c.encoded==3&&c.failures==negative_count+down_negative_count);myds_metal_shared_q4_v41_nb_counters(&c,true);myds_metal_shared_q4_v41_nb_counters(&c,false);assert(!c.requests&&!c.encoded&&!c.failures);
 printf("PASS CPU-only real GU/Down helpers: %u GU + %u Down predispatch/failure negatives; strict env/init conflicts/missing PSOs; exact PSO/ABI/geometry; old provider preserved; counters/reset\n",negative_count,down_negative_count);
}return 0;}
'''
def probe_host_only(binary, d):
    # Deliberately fabricated GGUF/library bytes: host-only producer protocol,
    # no MTLDevice or numeric acceptance. Full real weights remain a GPU gate.
    header=struct.pack('<IIQQ',0x46554747,3,3,0);off=0
    for part,dims in [('ffn_gate_shexp.weight',(5120,2304)),('ffn_up_shexp.weight',(5120,2304)),('ffn_down_shexp.weight',(2304,5120))]:
        name=('blk.0.'+part).encode();header+=struct.pack('<Q',len(name))+name+struct.pack('<IQQIQ',2,*dims,12,off);off+=6635520
    header+=bytes((-len(header))%32);model=d/'fabricated.gguf';model.write_bytes(header+bytes(off))
    gold=d/'golden.not-metal';gold.write_bytes(b'fabricated-golden-CPU-only')
    cand=d/'candidate.not-metal';cand.write_bytes(b'fabricated-candidate-CPU-only')
    ident=d/'source.json';ident.write_text('{"cpu_only":true,"fabricated":true}')
    command=[str(binary),'--model',str(model),'--golden-metallib',str(gold),'--candidate-metallib',str(cand),'--source-identity',str(ident),'--out-dir',str(d/'host-good'),'--layers','0','--cases','3','--samples','3','--reps','1','--host-only']
    run=subprocess.run(command,text=True,capture_output=True);need(run.returncode==0,run.stdout+run.stderr)
    m=json.loads((d/'host-good/artifact-manifest.json').read_text());need(m['status']=='host_only' and m['total_cases']==3 and not m['files'],'host-only completeness')
    records=[json.loads(x) for x in run.stdout.splitlines()];need([r['kind'] for r in records]==['identity','complete'] and records[-1]['host_only'] is True,'host-only stdout')
    failures=[]
    for num,(key,value) in enumerate([('--layers','0,00'),('--cases','0'),('--samples','2'),('--reps','0'),('--golden-metallib',str(d/'missing'))]):
        bad=command.copy();bad[bad.index('--out-dir')+1]=str(d/f'bad-{num}');bad[bad.index(key)+1]=value
        cp=subprocess.run(bad,text=True,capture_output=True);need(cp.returncode!=0,'fabricated host negative accepted');failures.append({'key':key,'value':value,'returncode':cp.returncode})
    return {'cpu_only':True,'fabricated_inputs':True,'no_gpu':True,'host_only_manifest_sha256':sha((d/'host-good/artifact-manifest.json').read_bytes()),'protocol_negative_cases':failures,'numeric_acceptance':False}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--probe-binary',type=Path);a=p.parse_args();need(not a.out.exists(),'new report required')
    proof=source_check();source=build_source()
    with tempfile.TemporaryDirectory(prefix='shared-q4-v41-cpu-') as t:
        d=Path(t);m=d/'host.m';b=d/'host';m.write_text(source)
        cmd=['clang','-O1','-g','-fobjc-arc','-Wno-unused-function','-Wno-unused-parameter','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(R/'include'),str(m),'-framework','Foundation','-framework','Metal','-o',str(b)]
        cp=subprocess.run(cmd,text=True,capture_output=True);need(cp.returncode==0,cp.stderr)
        run=subprocess.run([str(b)],text=True,capture_output=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'});need(run.returncode==0,run.stdout+run.stderr)
        proof.update(cpu_only=True,no_gpu=True,pass_all=True,compile_argv=cmd,compile_stdout=cp.stdout,compile_stderr=cp.stderr,run_stdout=run.stdout,run_stderr=run.stderr,extracted_cpu_translation_unit_sha256=sha(source.encode()),script_sha256=sha(Path(__file__).read_bytes()),native_input_sha256={n:sha((R/n).read_bytes()) for n in ['src/myds_metal.m','src/myds_decode_candidates.inc','src/myds_v41_moe.c','include/myds_metal.h','metal/myds_v41_moe.metal','metal/myds_q4_v41.metal']})
        if a.probe_binary: proof['probe_host_only']=probe_host_only(a.probe_binary.resolve(),d)
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(proof,indent=2)+'\n');print(run.stdout,end='')
if __name__=='__main__':main()
