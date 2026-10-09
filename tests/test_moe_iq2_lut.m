/* Model-free component acceptance for the actual fused IQ2 decode kernel.
 * Includes the runtime only to inspect PSO selection/init state and retain GPU
 * command-buffer timestamps. Building does not initialize Metal or run GPU.
 * Data is seeded valid nonzero quantized blocks, not captured model weights. */
#include "myds_metal.m"
#include <inttypes.h>

static void failure(const char *file,int line,const char *condition);
#define REQUIRE(c) do { if (!(c)) failure(__FILE__,__LINE__,#c); } while (0)

struct guarded { myds_metal_buffer_t *owner,*view; size_t bytes; };
struct fixture {
    uint32_t dim,width,padded,active,tokens,experts;
    float clamp;
    struct guarded gate,up,down,x,ids,weights;
    struct guarded sg[2],su[2],mid[2],sd,out;
};
/* Failure evidence is raw, unvalidated data. In particular release gate/up and
 * scratch_down are deliberately untouched and never accepted as F32 outputs. */
static struct fixture *current_fixture;
static const char *current_directory,*current_stage="no_fixture";
static unsigned current_phase,current_case;
static bool current_completed;
static void failure(const char *file,int line,const char *condition) {
    fprintf(stderr,"FAIL %s:%d %s (phase=%u case=%u stage=%s)\n",file,line,condition,current_phase,current_case,current_stage);
    if(current_fixture&&current_directory&&current_completed) {
        const char *names[]={"reference-gate","reference-up","reference-mid","candidate-gate","candidate-up","candidate-mid","release-down-scratch","release-output"};
        struct fixture *f=current_fixture;
        struct guarded outputs[]={f->sg[0],f->su[0],f->mid[0],f->sg[1],f->su[1],f->mid[1],f->sd,f->out};
        for(unsigned i=0;i<sizeof(outputs)/sizeof(*outputs);i++) {
            char path[4096];int n=snprintf(path,sizeof(path),"%s/failure-phase%u-case%u-%s.bin",current_directory,current_phase,current_case,names[i]);
            if(n<=0||(size_t)n>=sizeof(path))continue;
            FILE *raw=fopen(path,"wb");if(!raw){fprintf(stderr,"cannot retain %s\n",path);continue;}
            size_t wrote=fwrite(myds_metal_buffer_data(outputs[i].view),1,outputs[i].bytes,raw);int closed=fclose(raw);
            fprintf(stderr,"FAILURE_RAW buffer=%s bytes=%zu wrote=%zu close=%d path=%s\n",names[i],outputs[i].bytes,wrote,closed,path);
        }
    }
    exit(1);
}
static struct guarded allocate(size_t bytes) {
    struct guarded b={.bytes=bytes};
    b.owner=myds_metal_buffer_alloc(bytes+32u); REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+32u);
    b.view=myds_metal_buffer_sub_view(b.owner,16u,bytes); REQUIRE(b.view); return b;
}
static void canary(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<16u;i++) REQUIRE(p[i]==0xa5&&p[16u+b.bytes+i]==0xa5);
}
static void release(struct guarded b) {
    canary(b); myds_metal_buffer_free(b.view); myds_metal_buffer_free(b.owner);
}
static void iq2_fill(struct guarded b,unsigned seed) {
    REQUIRE(b.bytes%66u==0u); uint8_t *p=myds_metal_buffer_data(b.view);
    for(size_t n=0;n<b.bytes/66u;n++) {
        const uint16_t d=(uint16_t)(0x2000u+(n%4u)*0x100u);
        memcpy(p+n*66u,&d,2u);
        for(size_t j=0;j<64u;j++) p[n*66u+2u+j]=(uint8_t)(n*37u+j*29u+seed);
    }
}
static void q2_fill(struct guarded b,unsigned seed) {
    REQUIRE(b.bytes%84u==0u); uint8_t *p=myds_metal_buffer_data(b.view);
    for(size_t n=0;n<b.bytes/84u;n++) {
        for(size_t j=0;j<16u;j++) p[n*84u+j]=(uint8_t)(1u+(n*13u+j*17u+seed)%255u);
        for(size_t j=0;j<64u;j++) p[n*84u+16u+j]=(uint8_t)(n*23u+j*41u+seed);
        const uint16_t d=(uint16_t)(0x1c00u+(n%3u)*0x80u),m=(uint16_t)(0x1000u+(n%5u)*0x80u);
        memcpy(p+n*84u+80u,&d,2u); memcpy(p+n*84u+82u,&m,2u);
    }
}
static void finite_nonzero(const char *name,struct guarded b) {
    const float *p=myds_metal_buffer_data(b.view); size_t nz=0;
    for(size_t i=0;i<b.bytes/4u;i++) {
        uint32_t bits; memcpy(&bits,p+i,4u);
        if(!isfinite(p[i])||bits==0xa5a5a5a5u)fprintf(stderr,"INVALID_F32 buffer=%s index=%zu bits=%08x finite=%d untouched=%d\n",name,i,bits,isfinite(p[i]),bits==0xa5a5a5a5u);
        REQUIRE(isfinite(p[i])&&bits!=0xa5a5a5a5u); nz+=p[i]!=0;
    }
    REQUIRE(nz>0); canary(b);
}
static void untouched(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.view);
    for(size_t i=0;i<b.bytes;i++) REQUIRE(p[i]==0xa5); canary(b);
}
static void exact(const char *name,struct guarded a,struct guarded b) {
    REQUIRE(a.bytes==b.bytes);
    const float *pa=myds_metal_buffer_data(a.view),*pb=myds_metal_buffer_data(b.view);
    if(memcmp(pa,pb,a.bytes))for(size_t i=0;i<a.bytes/4u;i++)if(memcmp(pa+i,pb+i,4u)){
        uint32_t ab,bb;memcpy(&ab,pa+i,4u);memcpy(&bb,pb+i,4u);fprintf(stderr,"PARITY_F32 buffer=%s index=%zu reference=%08x candidate=%08x\n",name,i,ab,bb);break;
    }
    REQUIRE(!memcmp(pa,pb,a.bytes));finite_nonzero(name,a);finite_nonzero(name,b);
}
static struct fixture create(uint32_t dim,uint32_t width,uint32_t active,uint32_t tokens,float clamp) {
    REQUIRE(dim%256u==0u&&active<=6u&&active>0u);
    struct fixture f={.dim=dim,.width=width,.padded=(width+15u)&~15u,.active=active,.tokens=tokens,.experts=13u,.clamp=clamp};
    f.gate=allocate((size_t)f.experts*f.padded*(dim/256u)*66u); f.up=allocate(f.gate.bytes);
    /* Down weights are used only for full release cases (width%256==0). */
    f.down=allocate((size_t)f.experts*dim*((width+255u)/256u)*84u);
    f.x=allocate((size_t)tokens*dim*4u); f.ids=allocate((size_t)tokens*active*4u); f.weights=allocate(f.ids.bytes);
    for(unsigned k=0;k<2u;k++) { f.sg[k]=allocate((size_t)tokens*active*width*4u); f.su[k]=allocate(f.sg[k].bytes); f.mid[k]=allocate(f.sg[k].bytes); }
    f.sd=allocate((size_t)tokens*active*dim*4u); f.out=allocate((size_t)tokens*dim*4u);
    iq2_fill(f.gate,17u); iq2_fill(f.up,71u); q2_fill(f.down,29u);
    float *x=myds_metal_buffer_data(f.x.view);
    for(size_t i=0;i<f.x.bytes/4u;i++) x[i]=((int)((i*19u+43u)%257u)-128)/512.0f;
    const uint32_t sel[]={11u,3u,0u,12u,5u,7u};
    int32_t *id=myds_metal_buffer_data(f.ids.view); float *w=myds_metal_buffer_data(f.weights.view);
    for(uint32_t t=0;t<tokens;t++) for(uint32_t s=0;s<active;s++) {
        id[(size_t)t*active+s]=(int32_t)((sel[s]+t*5u)%f.experts);
        w[(size_t)t*active+s]=(float)(s+1u)/(float)(active*(active+1u)/2u);
    }
    return f;
}
static void destroy(struct fixture *f) {
    struct guarded all[]={f->gate,f->up,f->down,f->x,f->ids,f->weights,f->sg[0],f->sg[1],f->su[0],f->su[1],f->mid[0],f->mid[1],f->sd,f->out};
    for(size_t i=0;i<sizeof(all)/sizeof(*all);i++) release(all[i]);
}
static id<MTLComputePipelineState> pipeline(bool constant_lut) {
    MTLFunctionConstantValues *c=[MTLFunctionConstantValues new]; short nsg=(short)g_moe_nsg;
    [c setConstantValue:&nsg type:MTLDataTypeShort atIndex:600]; NSError *err=nil;
    NSString *name=constant_lut?@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_constant_lut_f32":@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32";
    id<MTLFunction> fn=[g_library newFunctionWithName:name constantValues:c error:&err];
    if(!fn) fprintf(stderr,"Missing fixture kernel %s: %s\n",name.UTF8String,err.localizedDescription.UTF8String); REQUIRE(fn);
    id<MTLComputePipelineState> pso=[g_device newComputePipelineStateWithFunction:fn error:&err]; REQUIRE(pso);
    REQUIRE(pso.maxTotalThreadsPerThreadgroup>=(NSUInteger)32u*g_moe_nsg); return pso;
}
/* Debug outputs use i11=idx%ne11, while mid uses pair_row. With production
 * ne11=1 every expert's optional gate/up stores overlap the first row; release
 * disables those stores. For a complete diagnostic matrix use ne11=active and
 * nb11=0: each expert still reads the SAME x, but writes its own debug row.
 * Timing retains the exact release args, ne11=1/nb11=dim*sizeof(float).
 * Direct tail fixtures additionally pad weight rows to protect its NR=4 reads. */
static struct ds4_metal_args_mul_mv_id pair_args(const struct fixture *f,bool write_intermediates) {
    struct ds4_metal_args_mul_mv_id a={
      .nei0=f->active,.nei1=f->tokens,.nbi1=(uint64_t)f->active*4u,
      .ne00=f->dim,.ne01=f->width,.ne02=f->experts,.nb00=66u,.nb01=(uint64_t)(f->dim/256u)*66u,
      .nb02=(uint64_t)f->padded*(f->dim/256u)*66u,.ne10=f->dim,.ne11=write_intermediates?(int32_t)f->active:1,.ne12=1,.ne13=1,
      .nb10=4u,.nb11=write_intermediates?0u:(uint64_t)f->dim*4u,.nb12=(uint64_t)f->dim*4u,
      .ne0=f->width,.ne1=f->active,.nb1=(uint64_t)f->width*4u,.nr0=4,.tp_world=0};
    return a;
}
static void args_contract(void) {
    for(unsigned active=3u;active<=6u;active+=3u)for(unsigned tokens=1u;tokens<=3u;tokens++) {
        struct fixture f={.dim=5120u,.width=2304u,.padded=2304u,.active=active,.tokens=tokens,.experts=13u};
        struct ds4_metal_args_mul_mv_id debug=pair_args(&f,true),release_args=pair_args(&f,false);
        REQUIRE(debug.ne11==(int32_t)active&&debug.nb11==0u&&release_args.ne11==1&&release_args.nb11==5120u*4u);
        uint32_t seen=0;
        for(unsigned t=0;t<tokens;t++)for(unsigned expert=0;expert<active;expert++) {
            const unsigned debug_i11=expert%(unsigned)debug.ne11,release_i11=expert%(unsigned)release_args.ne11;
            const uint64_t debug_x=(uint64_t)debug_i11*debug.nb11+(uint64_t)t*debug.nb12;
            const uint64_t release_x=(uint64_t)release_i11*release_args.nb11+(uint64_t)t*release_args.nb12;
            const uint64_t debug_row=(uint64_t)t*debug.ne0*debug.ne1+(uint64_t)debug_i11*debug.ne0;
            const uint64_t pair_row=(uint64_t)t*active+expert;
            REQUIRE(debug_x==release_x&&debug_x==(uint64_t)t*f.dim*4u);
            REQUIRE(debug_row==pair_row*f.width&&pair_row<32u&&!(seen&(1u<<(unsigned)pair_row)));
            seen|=1u<<(unsigned)pair_row;
        }
        REQUIRE(seen==(1u<<(active*tokens))-1u);
    }
    puts("{\"kind\":\"args_contract\",\"pass\":true,\"expert_token_shapes\":6,\"debug_full_gate_up_rows\":true,\"debug_and_release_activation_offsets_equal\":true,\"timing_uses_release_args\":true}");
}
static void encode_pair(struct fixture *f,id<MTLComputePipelineState> pso,bool constant_lut,unsigned dst,bool write_intermediates) {
    struct ds4_metal_args_mul_mv_id a=pair_args(f,write_intermediates);
    struct ds4_metal_dsv4_moe_swiglu_weight_args act={
      .width=f->width,.rows=(uint64_t)f->active*f->tokens,
      .gate_row_stride=(uint64_t)f->width*4u,.up_row_stride=(uint64_t)f->width*4u,
      .mid_row_stride=(uint64_t)f->width*4u,.weight_stride=4u,.write_clamped=write_intermediates,.clamp_value=f->clamp};
    id<MTLComputeCommandEncoder> e=get_compute_encoder(g_batch_cb); REQUIRE(e);
    [e setComputePipelineState:pso]; [e setBytes:&a length:sizeof(a) atIndex:0]; [e setBytes:&act length:sizeof(act) atIndex:1];
    const myds_metal_buffer_t *b[]={f->gate.view,f->up.view,f->x.view,f->sg[dst].view,f->su[dst].view,f->mid[dst].view,f->ids.view,f->weights.view};
    for(unsigned i=0;i<8u;i++) [e setBuffer:b[i]->mtl_buf offset:b[i]->offset atIndex:i+2u];
    [e setThreadgroupMemoryLength:constant_lut?0u:2176u atIndex:0];
    const NSUInteger groups=(f->width+4u*(uint32_t)g_moe_nsg-1u)/(4u*(uint32_t)g_moe_nsg);
    metal_dispatch(e,"fixture_iq2_fused",MTLSizeMake(groups,1,(NSUInteger)f->active*f->tokens),MTLSizeMake(32,g_moe_nsg,1));
    const myds_metal_buffer_t *deps[]={f->sg[dst].view,f->su[dst].view,f->mid[dst].view};
    myds_metal_barrier_buffers(deps,3u); /* Repeated dispatch WAW, including concurrent mode. */
}
static void pair_once(struct fixture *f,id<MTLComputePipelineState> pso,bool constant_lut,unsigned dst) {
    current_completed=false;REQUIRE(myds_metal_begin_batch()); encode_pair(f,pso,constant_lut,dst,true); REQUIRE(myds_metal_commit_batch());
    current_completed=true;current_stage=constant_lut?"direct_candidate_completed":"direct_reference_completed";
}
static void export_case(const char *directory,unsigned phase,unsigned index,struct fixture *f,bool release_path) {
    char path[4096]; int n=snprintf(path,sizeof(path),"%s/phase%u-case%u.bin",directory,phase,index); REQUIRE(n>0&&(size_t)n<sizeof(path));
    FILE *file=fopen(path,"wb"); REQUIRE(file);
    struct guarded values[]={f->sg[0],f->su[0],f->mid[0],f->out}; size_t bytes=0;
    for(unsigned i=0;i<(release_path?4u:3u);i++) {
        finite_nonzero("validated_export",values[i]); REQUIRE(fwrite(myds_metal_buffer_data(values[i].view),1,values[i].bytes,file)==values[i].bytes); bytes+=values[i].bytes;
    }
    REQUIRE(fclose(file)==0);
    printf("{\"kind\":\"case\",\"phase\":%u,\"index\":%u,\"dim\":%u,\"width\":%u,\"active\":%u,\"tokens\":%u,\"clamp\":%.9g,\"bytes\":%zu,\"constant_lut\":%s,\"iq2_nsg\":%d,\"finite\":true,\"nonzero\":true,\"canaries\":true,\"direct_full_f32_byte_exact\":true,\"direct_full_f32_byte_exact_scope\":\"gate_up_mid\",\"release_output_scope\":\"finite_nonzero_here_full_f32_parity_in_driver\",\"release_mid_bit_exact\":%s,\"release_path\":%s}\n",
      phase,index,f->dim,f->width,f->active,f->tokens,f->clamp,bytes,g_moe_iq2_constant_lut?"true":"false",g_moe_nsg,release_path?"true":"false",release_path?"true":"false");
}
static double measure(struct fixture *f,id<MTLComputePipelineState> pso,bool lut,unsigned repetitions) {
    current_completed=false;REQUIRE(myds_metal_begin_batch()); id<MTLCommandBuffer> cb=g_batch_cb;
    for(unsigned i=0;i<repetitions;i++) encode_pair(f,pso,lut,1u,false);
    REQUIRE(myds_metal_commit_batch());
    current_completed=true;current_stage=lut?"timing_candidate_completed":"timing_reference_completed";
    REQUIRE(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime);
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/(double)repetitions;
}
static void timings(struct fixture *f,id<MTLComputePipelineState> ref,id<MTLComputePipelineState> candidate,unsigned phase,unsigned samples,unsigned reps) {
    id<MTLComputePipelineState> arms[]={ref,candidate,ref};
    const char *names[]={"A1","B","A2"};
    for(unsigned arm=0;arm<3u;arm++) {
        (void)measure(f,arms[arm],arm==1u,reps); /* Full repetitions, excluded warmup. */
        for(unsigned sample=0;sample<samples;sample++) {
            double us=measure(f,arms[arm],arm==1u,reps); exact("timing_mid",f->mid[0],f->mid[1]);
            printf("{\"kind\":\"timing\",\"phase\":%u,\"arm\":\"%s\",\"sample\":%u,\"repetitions\":%u,\"gpu_us_per_dispatch\":%.9f,\"iq2_nsg\":%d,\"dynamic_tg_bytes\":%u,\"threads\":%u,\"max_threads\":%lu,\"static_tg_bytes\":%lu}\n",
              phase,names[arm],sample,reps,us,g_moe_nsg,arm==1u?0u:2176u,32u*(unsigned)g_moe_nsg,
              (unsigned long)arms[arm].maxTotalThreadsPerThreadgroup,(unsigned long)arms[arm].staticThreadgroupMemoryLength);
        }
    }
}
static void cases(const char *directory,unsigned phase,unsigned samples,unsigned repetitions) {
    id<MTLComputePipelineState> ref=pipeline(false),candidate=pipeline(true);
    struct config {uint32_t dim,width,active,tokens;float clamp;bool concurrent;} cfg[]={
      {256,256,6,1,0.0f,false},{512,768,6,1,0.03125f,true},
      {512,768,3,2,10.0f,true},{512,768,6,3,0.0f,false},
      {512,1,6,1,10.0f,true},{512,17,6,2,0.03125f,true},{512,259,3,1,0.0f,false},
      {5120,2304,6,1,10.0f,true},{5120,2304,3,1,10.0f,true}};
    for(unsigned i=0;i<sizeof(cfg)/sizeof(*cfg);i++) {
        struct fixture f=create(cfg[i].dim,cfg[i].width,cfg[i].active,cfg[i].tokens,cfg[i].clamp);
        current_fixture=&f;current_directory=directory;current_phase=phase;current_case=i;current_completed=false;current_stage="fixture_created";
        myds_metal_set_concurrent(cfg[i].concurrent); myds_metal_set_auto_barrier(!cfg[i].concurrent);
        pair_once(&f,ref,false,0u); pair_once(&f,candidate,true,1u);
        exact("direct_gate",f.sg[0],f.sg[1]); exact("direct_up",f.su[0],f.su[1]); exact("direct_mid",f.mid[0],f.mid[1]);
        const bool release_path=f.width%256u==0u;
        if(release_path) {
            /* Release suppresses gate/up writes and dispatches IQ2 + Q2 exactly. */
            memset(myds_metal_buffer_data(f.sg[1].view),0xa5,f.sg[1].bytes);
            memset(myds_metal_buffer_data(f.su[1].view),0xa5,f.su[1].bytes);
            current_completed=false;REQUIRE(myds_metal_begin_batch()); myds_metal_dispatch_count_reset();
            REQUIRE(myds_metal_routed_moe_batch(f.out.view,f.x.view,f.gate.view,f.up.view,f.down.view,f.ids.view,f.weights.view,
              f.sg[1].view,f.su[1].view,f.mid[1].view,f.sd.view,f.dim,f.width,f.experts,f.active,f.tokens,
              GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,f.clamp));
            REQUIRE(myds_metal_dispatch_count()==2u); REQUIRE(myds_metal_commit_batch());
            current_completed=true;current_stage="release_completed";
            exact("release_mid",f.mid[0],f.mid[1]); untouched(f.sg[1]); untouched(f.su[1]); untouched(f.sd); finite_nonzero("release_output",f.out);
        }
        export_case(directory,phase,i,&f,release_path);
        if(i==7u&&samples) timings(&f,ref,candidate,phase,samples,repetitions);
        current_fixture=NULL;current_completed=false;destroy(&f);
    }
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc==2&&!strcmp(argv[1],"--args-contract")){
        REQUIRE(!g_initialized&&!g_device&&!g_queue&&!g_library);args_contract();
        REQUIRE(!g_initialized&&!g_device&&!g_queue&&!g_library);return 0;
    }
    if(argc==2&&!strcmp(argv[1],"--reject")) {
        REQUIRE(!myds_metal_init(NULL)); REQUIRE(!g_initialized&&!g_device&&!g_queue&&!g_library);
        REQUIRE(!g_pso_moe_pair_swiglu_iq2_xxs&&!g_moe_iq2_constant_lut);
        puts("{\"kind\":\"rejected_before_metal_resources\",\"pass\":true}"); return 0;
    }
    if(argc==4&&!strcmp(argv[1],"--missing-candidate")) {
        REQUIRE(setenv("MYDS_MOE_IQ2_CONSTANT_LUT","1",1)==0); REQUIRE(!myds_metal_init(argv[2]));
        REQUIRE(!g_initialized&&!g_device&&!g_queue&&!g_library&&!g_pso_moe_pair_swiglu_iq2_xxs&&!g_moe_iq2_constant_lut);
        REQUIRE(setenv("MYDS_MOE_IQ2_CONSTANT_LUT","0",1)==0); REQUIRE(myds_metal_init(argv[2])); REQUIRE(!g_moe_iq2_constant_lut); myds_metal_cleanup();
        REQUIRE(setenv("MYDS_MOE_IQ2_CONSTANT_LUT","1",1)==0); REQUIRE(myds_metal_init(argv[3])); REQUIRE(g_moe_iq2_constant_lut); myds_metal_cleanup();
        puts("{\"kind\":\"missing_candidate_rejected_and_retried\",\"pass\":true}"); return 0;
    }
    if(argc<7||argc>8) { fprintf(stderr,"Usage: test_moe_iq2_lut shader_dir output_dir expected_lut nsg samples repetitions [--reinit]\n"); return 2; }
    const int expected=atoi(argv[3]),nsg=atoi(argv[4]); const unsigned samples=(unsigned)atoi(argv[5]),repetitions=(unsigned)atoi(argv[6]);
    REQUIRE((expected==0||expected==1)&&(nsg==2||nsg==4)&&samples<=100u&&repetitions>0u&&repetitions<=10000u);
    REQUIRE(myds_metal_init(argv[1])); REQUIRE(g_moe_iq2_constant_lut==(expected!=0)&&g_moe_nsg==nsg&&g_moe_down_nsg==nsg);
    cases(argv[2],0u,samples,repetitions);
    REQUIRE(setenv("MYDS_MOE_IQ2_CONSTANT_LUT",expected?"0":"1",1)==0);
    REQUIRE(myds_metal_init(argv[1])); REQUIRE(g_moe_iq2_constant_lut==(expected!=0));
    myds_metal_cleanup(); REQUIRE(!g_initialized&&!g_moe_iq2_constant_lut&&!g_pso_moe_pair_swiglu_iq2_xxs);
    if(argc==8) {
        REQUIRE(!strcmp(argv[7],"--reinit")); REQUIRE(myds_metal_init(argv[1])); REQUIRE(g_moe_iq2_constant_lut==(expected==0));
        cases(argv[2],1u,0u,repetitions); myds_metal_cleanup();
        REQUIRE(unsetenv("MYDS_MOE_IQ2_CONSTANT_LUT")==0); REQUIRE(myds_metal_init(argv[1])); REQUIRE(!g_moe_iq2_constant_lut);
        cases(argv[2],2u,0u,repetitions); myds_metal_cleanup();
    }
    puts("{\"kind\":\"complete\",\"pass\":true,\"model_loaded\":false}"); return 0;
}
