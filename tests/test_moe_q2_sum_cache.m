/* Opt-in component gate. Uses the actual release routed-MoE helper, its
 * private init state and cache; does not replace either production selector.
 * Seeded IQ2/Q2 blocks are not captured GGUF weights. Only the selected six
 * experts of the complete 384-expert buffers are initialized/read. */
#include "myds_metal.m"
#include <inttypes.h>
#include <sys/stat.h>

#define DIM 5120u
#define WIDTH 2304u
#define ACTIVE 6u
#define EXPERTS 384u
#define MID_BYTES ((size_t)ACTIVE*WIDTH*4u)
#define OUT_BYTES ((size_t)DIM*4u)
#define CACHE_BYTES (6u*9u*32u*4u)
static void fail(const char *file,int line,const char *condition);
#define REQUIRE(x) do { if (!(x)) fail(__FILE__,__LINE__,#x); } while(0)
static const char *directory;
static unsigned raw_files;
static const uint32_t selected[ACTIVE]={383u,3u,0u,271u,5u,127u};

/* This is an independent copy of the ORIGINAL sum6 float4 accumulation loop,
 * not the new prepare kernel. The build target compiles it with the same
 * -O3 -ffast-math flags as the production metallib. Its output is diagnostic;
 * the unchanged original Q2 down PSO remains the full-output baseline. */
static const char oracle_source[]=
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void q2_original_sumy_oracle(device const float *mid [[buffer(0)]], device float *cache [[buffer(1)]], uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_threadgroup]]) {\n"
" const int nb=9; const int slot=tg.x; const short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;\n"
" device const float *y=mid+slot*2304; device const float *y4=y+ix*256+128*iq+8*ir;\n"
" for(int ib=ix;ib<nb;ib+=4) { float yl[32]; float4 sumy={0.f};\n"
" #pragma unroll\n for(short i=0;i<8;++i) { yl[i+0]=y4[i+0];sumy[0]+=yl[i+0];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yl[i+16]=y4[i+64];sumy[2]+=yl[i+16];yl[i+24]=y4[i+96];sumy[3]+=yl[i+24]; }\n"
" const uint b=(slot*nb+ib)*32+16*iq+ir;cache[b+0]=sumy[0];cache[b+4]=sumy[1];cache[b+8]=sumy[2];cache[b+12]=sumy[3]; y4+=4*256; } }\n";
static id<MTLComputePipelineState> oracle_pso;
struct guarded { myds_metal_buffer_t *owner,*view; size_t bytes; };
static struct guarded allocate(size_t bytes,bool sparse) {
    struct guarded b={.bytes=bytes}; b.owner=myds_metal_buffer_alloc(bytes+64); REQUIRE(b.owner);
    uint8_t *p=myds_metal_buffer_data(b.owner); REQUIRE(p); memset(p,0xa5,32);memset(p+32+bytes,0xa5,32);
    if(!sparse)memset(p+32,0xa5,bytes);
    b.view=myds_metal_buffer_sub_view(b.owner,32,bytes);REQUIRE(b.view);return b;
}
static void canary(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<32;i++)REQUIRE(p[i]==0xa5&&p[32+b.bytes+i]==0xa5);
}
static void release(struct guarded b){canary(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void poison(struct guarded b){memset(myds_metal_buffer_data(b.view),0xa5,b.bytes);}
static uint64_t fnv(const void *bytes,size_t length) {
    uint64_t h=UINT64_C(14695981039346656037);const uint8_t *p=bytes;
    for(size_t i=0;i<length;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static void finite(struct guarded b,bool nonzero) {
    const float *p=myds_metal_buffer_data(b.view);size_t nz=0;
    for(size_t i=0;i<b.bytes/4;i++){uint32_t u;memcpy(&u,p+i,4);REQUIRE(isfinite(p[i])&&u!=0xa5a5a5a5u);nz+=p[i]!=0;}
    REQUIRE(!nonzero||nz);canary(b);
}
static void untouched(struct guarded b){const uint8_t *p=myds_metal_buffer_data(b.view);for(size_t i=0;i<b.bytes;i++)REQUIRE(p[i]==0xa5);canary(b);}
static void equal(struct guarded a,struct guarded b,bool nonzero){REQUIRE(a.bytes==b.bytes);finite(a,nonzero);finite(b,nonzero);REQUIRE(!memcmp(myds_metal_buffer_data(a.view),myds_metal_buffer_data(b.view),a.bytes));}
static void file_item(const char *stem,const char *role,struct guarded b,const char *dtype,bool *first){
    char name[256],path[4096];REQUIRE(snprintf(name,sizeof(name),"%s-%s.%s",stem,role,!strcmp(dtype,"f32")?"f32":!strcmp(dtype,"i32")?"i32":"bin")<(int)sizeof(name));
    REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));FILE *f=fopen(path,"wbx");REQUIRE(f);
    REQUIRE(fwrite(myds_metal_buffer_data(b.view),1,b.bytes,f)==b.bytes);REQUIRE(!fclose(f));canary(b);
    printf("%s{\"path\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",*first?"":",",name,role,dtype,b.bytes,fnv(myds_metal_buffer_data(b.view),b.bytes));*first=false;raw_files++;
}
struct fixture {
    struct guarded gate,up,down,x,ids,weights;
    struct guarded sg[2],su[2],mid[2],sd[2],out[2],cache[2],dg[2],du[2],dm[2];float clamp;
};
static struct fixture *active_fixture;
static const char *active_stage="host";
static unsigned active_case;
/* Failure evidence is never replaced by a PASS record. If legal work was
 * encoded before a contract assertion, drain it before reading shared buffers.
 * Complete diagnostics survive even if a strict bit comparison rejects it. */
static void fail(const char *file,int line,const char *condition){
    static bool entered;
    fprintf(stderr,"FAIL %s:%d %s stage=%s case=%u\n",file,line,condition,active_stage,active_case);
    if(entered)exit(1);entered=true;
    bool drained=!g_batch_cb||myds_metal_commit_batch();
    fflush(stdout);
    if(directory&&active_fixture&&drained){
        struct fixture *f=active_fixture;
        struct guarded buffers[]={f->x,f->ids,f->weights,f->mid[0],f->mid[1],f->out[0],f->out[1],f->cache[0],f->cache[1],f->dg[0],f->dg[1],f->du[0],f->du[1],f->dm[0],f->dm[1],f->sg[0],f->sg[1],f->su[0],f->su[1]};
        const char *roles[]={"input","selected-ids","route-weights","reference-mid","candidate-mid","reference-output","candidate-output","reference-cache","candidate-cache","reference-debug-gate","candidate-debug-gate","reference-debug-up","candidate-debug-up","reference-debug-mid","candidate-debug-mid","reference-release-gate","candidate-release-gate","reference-release-up","candidate-release-up"};
        /* A failed record may have begun printing; separate the failure object
         * with a newline. The auditor still fails closed on malformed JSON. */
        printf("\n{\"schema_version\":1,\"kind\":\"native_failure\",\"stage\":\"%s\",\"case\":%u,\"drained\":true,\"files\":[",active_stage,active_case);
        bool first=true;
        for(unsigned i=0;i<19;i++){
            char path[4096],name[256];snprintf(name,sizeof(name),"failure-%s-case%u-%s.bin",active_stage,active_case,roles[i]);snprintf(path,sizeof(path),"%s/%s",directory,name);
            const void *data=myds_metal_buffer_data(buffers[i].view);if(!data)continue;
            FILE *out=fopen(path,"wb");if(!out)continue;size_t written=fwrite(data,1,buffers[i].bytes,out);fclose(out);
            printf("%s{\"path\":\"%s\",\"bytes\":%zu,\"written\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",first?"":",",name,buffers[i].bytes,written,fnv(data,buffers[i].bytes));first=false;
        }
        if(g_moe_q2_sum_cache_buffer.contents){
            char path[4096];snprintf(path,sizeof(path),"%s/failure-actual-private-cache.bin",directory);FILE *out=fopen(path,"wb");if(out){size_t written=fwrite(g_moe_q2_sum_cache_buffer.contents,1,CACHE_BYTES,out);fclose(out);printf("%s{\"path\":\"failure-actual-private-cache.bin\",\"bytes\":6912,\"written\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",first?"":",",written,fnv(g_moe_q2_sum_cache_buffer.contents,CACHE_BYTES));}
        }
        puts("]}");
    }
    fflush(stdout);exit(1);
}
static void fill_iq2(uint8_t *p,size_t length,unsigned seed){
    for(size_t b=0;b<length/66;b++){uint16_t d=(uint16_t)(0x1400u+(b%4u)*0x100u);memcpy(p+66*b,&d,2);for(size_t j=0;j<64;j++)p[66*b+2+j]=(uint8_t)(b*37u+j*29u+seed);}
}
static void fill_q2(uint8_t *p,size_t length,unsigned seed){
    for(size_t b=0;b<length/84;b++){for(size_t j=0;j<16;j++)p[b*84+j]=(uint8_t)(1u+(b*13u+j*17u+seed)%255u);for(size_t j=0;j<64;j++)p[b*84+16+j]=(uint8_t)(b*23u+j*41u+seed);uint16_t d=0x1c00u+(b%3u)*0x80u,m=0x1000u+(b%5u)*0x80u;memcpy(p+b*84+80,&d,2);memcpy(p+b*84+82,&m,2);}
}
static void input(struct fixture *f,unsigned seed){float *x=myds_metal_buffer_data(f->x.view);for(size_t i=0;i<DIM;i++)x[i]=((int)((i*19u+seed*23u)%1023u)-511)/1024.f;}
static struct fixture create(void){
    struct fixture f={.clamp=10.f};const size_t gate_exp=(size_t)WIDTH*(DIM/256u)*66u,down_exp=(size_t)DIM*(WIDTH/256u)*84u;
    f.gate=allocate(EXPERTS*gate_exp,true);f.up=allocate(f.gate.bytes,true);f.down=allocate(EXPERTS*down_exp,true);
    for(unsigned s=0;s<ACTIVE;s++){fill_iq2((uint8_t*)myds_metal_buffer_data(f.gate.view)+selected[s]*gate_exp,gate_exp,17+s);fill_iq2((uint8_t*)myds_metal_buffer_data(f.up.view)+selected[s]*gate_exp,gate_exp,71+s);fill_q2((uint8_t*)myds_metal_buffer_data(f.down.view)+selected[s]*down_exp,down_exp,29+s);}
    f.x=allocate(OUT_BYTES,false);f.ids=allocate(ACTIVE*4,false);f.weights=allocate(ACTIVE*4,false);
    int32_t *ids=myds_metal_buffer_data(f.ids.view);float *weights=myds_metal_buffer_data(f.weights.view);
    for(unsigned s=0;s<ACTIVE;s++){ids[s]=(int32_t)selected[s];weights[s]=(float)(s+1u)/21.f;}
    for(unsigned k=0;k<2;k++){f.sg[k]=allocate(MID_BYTES,false);f.su[k]=allocate(MID_BYTES,false);f.mid[k]=allocate(MID_BYTES,false);f.sd[k]=allocate(ACTIVE*OUT_BYTES,false);f.out[k]=allocate(OUT_BYTES,false);f.cache[k]=allocate(CACHE_BYTES,false);f.dg[k]=allocate(MID_BYTES,false);f.du[k]=allocate(MID_BYTES,false);f.dm[k]=allocate(MID_BYTES,false);}input(&f,31);return f;
}
static void destroy(struct fixture *f){
    struct guarded a[]={f->gate,f->up,f->down,f->x,f->ids,f->weights};for(unsigned i=0;i<6;i++)release(a[i]);
    for(unsigned k=0;k<2;k++){struct guarded a[]={f->sg[k],f->su[k],f->mid[k],f->sd[k],f->out[k],f->cache[k],f->dg[k],f->du[k],f->dm[k]};for(unsigned i=0;i<9;i++)release(a[i]);}
}
static void policy(unsigned concurrent,unsigned automatic){myds_metal_set_concurrent((int)concurrent);myds_metal_set_auto_barrier((int)automatic);}
static void actual_encoder(void){REQUIRE(g_batch_enc);REQUIRE(g_batch_enc.dispatchType==(myds_metal_get_concurrent()?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial));}
static int helper(struct fixture *f,unsigned dst,bool candidate){
    return myds_metal_routed_moe_scoped(f->out[dst].view,f->x.view,f->gate.view,f->up.view,f->down.view,f->ids.view,f->weights.view,f->sg[dst].view,f->su[dst].view,f->mid[dst].view,f->sd[dst].view,DIM,WIDTH,EXPERTS,ACTIVE,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,f->clamp,candidate);
}
static void caller_join(struct fixture *f,unsigned dst){const myds_metal_buffer_t *a[]={f->out[dst].view};myds_metal_barrier_buffers(a,1);}
static void oracle(struct guarded mid,struct guarded cache){
    id<MTLComputeCommandEncoder> e=get_compute_encoder(g_batch_cb);REQUIRE(e&&oracle_pso);[e setComputePipelineState:oracle_pso];[e setBuffer:mid.view->mtl_buf offset:mid.view->offset atIndex:0];[e setBuffer:cache.view->mtl_buf offset:cache.view->offset atIndex:1];metal_dispatch(e,"fixture_original_sumy_oracle",MTLSizeMake(ACTIVE,1,1),MTLSizeMake(32,1,1));const myds_metal_buffer_t *a[]={cache.view};myds_metal_barrier_buffers(a,1);
}
static struct ds4_metal_args_mul_mv_id down_args(void){
    return (struct ds4_metal_args_mul_mv_id){.nei0=ACTIVE,.nei1=1,.nbi1=ACTIVE*4u,.ne00=WIDTH,.ne01=DIM,.ne02=EXPERTS,.nb00=84,.nb01=(WIDTH/256u)*84u,.nb02=(uint64_t)DIM*(WIDTH/256u)*84u,.ne10=WIDTH,.ne11=ACTIVE,.ne12=1,.ne13=1,.nb10=4,.nb11=WIDTH*4u,.nb12=ACTIVE*WIDTH*4u,.ne0=DIM,.ne1=1,.nb1=DIM*4u,.nr0=2,.tp_world=1};
}
static void prepare(struct guarded mid,struct guarded cache,struct guarded ids){
    id<MTLComputeCommandEncoder> e=get_compute_encoder(g_batch_cb);struct ds4_metal_args_mul_mv_id a=down_args();[e setComputePipelineState:g_pso_moe_q2_sum_cache_prepare];[e setBytes:&a length:sizeof(a) atIndex:0];const myds_metal_buffer_t *b[]={mid.view,cache.view,ids.view};for(unsigned i=0;i<3;i++)[e setBuffer:b[i]->mtl_buf offset:b[i]->offset atIndex:i+1];metal_dispatch(e,"fixture_prepare",MTLSizeMake(4,1,1),MTLSizeMake(128,1,1));const myds_metal_buffer_t *d[]={cache.view};myds_metal_barrier_buffers(d,1);
}
static void debug(struct fixture *f,unsigned dst){
    struct ds4_metal_args_mul_mv_id a={.nei0=ACTIVE,.nei1=1,.nbi1=ACTIVE*4u,.ne00=DIM,.ne01=WIDTH,.ne02=EXPERTS,.nb00=66,.nb01=(DIM/256u)*66u,.nb02=(uint64_t)WIDTH*(DIM/256u)*66u,.ne10=DIM,.ne11=ACTIVE,.ne12=1,.ne13=1,.nb10=4,.nb11=0,.nb12=DIM*4u,.ne0=WIDTH,.ne1=ACTIVE,.nb1=WIDTH*4u,.nr0=4,.tp_world=0};
    struct ds4_metal_dsv4_moe_swiglu_weight_args act={.width=WIDTH,.rows=ACTIVE,.gate_row_stride=WIDTH*4u,.up_row_stride=WIDTH*4u,.mid_row_stride=WIDTH*4u,.weight_stride=4,.write_clamped=1,.clamp_value=f->clamp};
    id<MTLComputeCommandEncoder> e=get_compute_encoder(g_batch_cb);[e setComputePipelineState:g_pso_moe_pair_swiglu_iq2_xxs];[e setBytes:&a length:sizeof(a) atIndex:0];[e setBytes:&act length:sizeof(act) atIndex:1];const myds_metal_buffer_t *b[]={f->gate.view,f->up.view,f->x.view,f->dg[dst].view,f->du[dst].view,f->dm[dst].view,f->ids.view,f->weights.view};for(unsigned i=0;i<8;i++)[e setBuffer:b[i]->mtl_buf offset:b[i]->offset atIndex:i+2];[e setThreadgroupMemoryLength:2176 atIndex:0];metal_dispatch(e,"fixture_original_iq2_debug",MTLSizeMake((WIDTH+4u*g_moe_nsg-1u)/(4u*g_moe_nsg),1,ACTIVE),MTLSizeMake(32,g_moe_nsg,1));const myds_metal_buffer_t *deps[]={f->dg[dst].view,f->du[dst].view,f->dm[dst].view};myds_metal_barrier_buffers(deps,3);
}
static void counter_check(uint64_t n){myds_metal_moe_q2_sum_cache_counters_t c;myds_metal_moe_q2_sum_cache_counters(&c,false);REQUIRE(c.requests==n&&c.encoded==n&&!c.failures);}
static void copy_actual_cache(struct guarded to){REQUIRE(g_moe_q2_sum_cache_buffer.length==CACHE_BYTES);memcpy(myds_metal_buffer_data(to.view),g_moe_q2_sum_cache_buffer.contents,CACHE_BYTES);}
static void pair_once(struct fixture *f,unsigned dst,bool candidate,unsigned repetitions,double *start,double *end){
    myds_metal_moe_q2_sum_cache_counters(NULL,true);myds_metal_dispatch_count_reset();REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    for(unsigned j=0;j<repetitions;j++){REQUIRE(helper(f,dst,candidate));actual_encoder();caller_join(f,dst);}REQUIRE(myds_metal_dispatch_count()==(candidate?3u:2u)*repetitions);counter_check(candidate?repetitions:0);
    REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);if(start)*start=cb.GPUStartTime;if(end)*end=cb.GPUEndTime;
    if(candidate)copy_actual_cache(f->cache[dst]);
    untouched(f->sg[dst]);untouched(f->su[dst]);untouched(f->sd[dst]);finite(f->mid[dst],true);finite(f->out[dst],true);
}
static void reference_cache(struct fixture *f,unsigned dst){REQUIRE(myds_metal_begin_batch());oracle(f->mid[dst],f->cache[dst]);REQUIRE(myds_metal_commit_batch());finite(f->cache[dst],true);}
static void files_numeric(const char *stem,struct fixture *f,bool inputs,bool diag){bool first=true;printf(",\"files\":[");
    if(inputs){file_item(stem,"input",f->x,"f32",&first);file_item(stem,"selected-ids",f->ids,"i32",&first);file_item(stem,"route-weights",f->weights,"f32",&first);}
    const char *arm[]={"reference","candidate"};for(unsigned k=0;k<2;k++){char role[128];
#define EMIT(suffix,buf,type) do{snprintf(role,sizeof(role),"%s-%s",arm[k],suffix);file_item(stem,role,buf,type,&first);}while(0)
        EMIT("mid",f->mid[k],"f32");EMIT("cache",f->cache[k],"f32");EMIT("output",f->out[k],"f32");
        if(diag){EMIT("debug-gate",f->dg[k],"f32");EMIT("debug-up",f->du[k],"f32");EMIT("debug-mid",f->dm[k],"f32");EMIT("release-gate",f->sg[k],"bytes");EMIT("release-up",f->su[k],"bytes");}
#undef EMIT
    }printf("]");
}
static void parity_case(struct fixture *f,unsigned n){active_stage="parity";active_case=n;unsigned concurrent=n%4u/2u,automatic=n%2u;policy(concurrent,automatic);input(f,n/4u?73u:31u);f->clamp=n/4u?.03125f:10.f;
    pair_once(f,0,false,1,NULL,NULL);reference_cache(f,0);pair_once(f,1,true,1,NULL,NULL);REQUIRE(myds_metal_begin_batch());debug(f,0);debug(f,1);REQUIRE(myds_metal_commit_batch());
    equal(f->mid[0],f->mid[1],true);equal(f->cache[0],f->cache[1],true);equal(f->out[0],f->out[1],true);equal(f->dg[0],f->dg[1],true);equal(f->du[0],f->du[1],true);equal(f->dm[0],f->dm[1],true);equal(f->dm[0],f->mid[0],true);
    char stem[64];snprintf(stem,sizeof(stem),"case%u",n);printf("{\"schema_version\":1,\"kind\":\"parity\",\"case\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"clamp\":%.9g,\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"release_dispatches\":[2,3],\"requests\":1,\"encoded\":1,\"failures\":0,\"full_mid_equal\":true,\"full_cache_equal\":true,\"full_output_equal\":true,\"debug_sg_su_equal\":true,\"debug_mid_equals_release\":true,\"release_sg_su_untouched\":true,\"finite\":true,\"canaries\":true",n,concurrent,automatic,f->clamp);files_numeric(stem,f,true,true);puts("}");
}
static void cache_case(struct fixture *f,unsigned n){
    active_stage="cache";active_case=n;    policy(n/2u,n%2u);struct guarded mid=allocate(MID_BYTES,false),ref=allocate(CACHE_BYTES,false),cand=allocate(CACHE_BYTES,false);
    float *p=myds_metal_buffer_data(mid.view);static const float edge[]={0.f,-0.f,1.f,-1.f,0x1p24f,-0x1p24f,0x1p-126f,-0x1p-126f,0x1p-149f,-0x1p-149f,0x1.000002p0f,-0x1.000002p0f,0x1p-20f,-0x1p-20f,255.f,-255.f};
    for(unsigned i=0;i<ACTIVE*WIDTH;i++)p[i]=edge[(i*7u+n*3u)%16u];
    REQUIRE(myds_metal_begin_batch());oracle(mid,ref);prepare(mid,cand,f->ids);actual_encoder();REQUIRE(myds_metal_commit_batch());
    char stem[64];snprintf(stem,sizeof(stem),"cache%u",n);bool first=true;
    bool same=!memcmp(myds_metal_buffer_data(ref.view),myds_metal_buffer_data(cand.view),CACHE_BYTES);
    printf("{\"schema_version\":1,\"kind\":\"cache_contract\",\"case\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"encoder_dispatch_type_verified\":true,\"finite_adversarial\":true,\"original_sumy_oracle\":true,\"full_cache_equal\":%s,\"canaries\":true,\"files\":[",n,n/2u,n%2u,same?"true":"false");file_item(stem,"mid",mid,"f32",&first);file_item(stem,"selected-ids",f->ids,"i32",&first);file_item(stem,"reference-cache",ref,"f32",&first);file_item(stem,"candidate-cache",cand,"f32",&first);puts("]}");
    equal(ref,cand,false);release(mid);release(ref);release(cand);
}
static void reuse_case(struct fixture *f,unsigned n){
    active_stage="reuse";active_case=n;    policy(n/2u,n%2u);f->clamp=10.f;struct guarded sources[3];for(unsigned i=0;i<3;i++){sources[i]=allocate(OUT_BYTES,false);input(f,91+i*19);memcpy(myds_metal_buffer_data(sources[i].view),myds_metal_buffer_data(f->x.view),OUT_BYTES);}
    struct guarded snap[2][7][3];for(unsigned a=0;a<2;a++)for(unsigned i=0;i<7;i++)for(unsigned k=0;k<3;k++)snap[a][i][k]=allocate(k==0?MID_BYTES:k==1?CACHE_BYTES:OUT_BYTES,false);
    const unsigned seq[]={0,0,1,2,2,1,0};myds_metal_moe_q2_sum_cache_counters(NULL,true);REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    for(unsigned a=0;a<2;a++)for(unsigned i=0;i<7;i++){
        const myds_metal_buffer_t *before[]={f->x.view,f->mid[a].view,f->out[a].view};myds_metal_barrier_buffers(before,3);
        REQUIRE(myds_metal_buffer_copy(f->x.view,0,sources[seq[i]].view,0,OUT_BYTES));const myds_metal_buffer_t *xdep[]={f->x.view};myds_metal_barrier_buffers(xdep,1);
        REQUIRE(helper(f,a,a==1));caller_join(f,a);actual_encoder();
        if(!a)oracle(f->mid[a],f->cache[a]);
        myds_metal_buffer_t actual={.mtl_buf=g_moe_q2_sum_cache_buffer,.size=CACHE_BYTES,.offset=0,.ptr=g_moe_q2_sum_cache_buffer.contents,.is_view=true};
        REQUIRE(myds_metal_buffer_copy(snap[a][i][0].view,0,f->mid[a].view,0,MID_BYTES));REQUIRE(myds_metal_buffer_copy(snap[a][i][1].view,0,a?&actual:f->cache[a].view,0,CACHE_BYTES));REQUIRE(myds_metal_buffer_copy(snap[a][i][2].view,0,f->out[a].view,0,OUT_BYTES));
        const myds_metal_buffer_t *reads[]={f->mid[a].view,f->out[a].view,a?&actual:f->cache[a].view};myds_metal_barrier_buffers(reads,3);
        const myds_metal_buffer_t *written[]={snap[a][i][0].view,snap[a][i][1].view,snap[a][i][2].view};myds_metal_barrier_buffers(written,3);
    }
    counter_check(7);REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
    printf("{\"schema_version\":1,\"kind\":\"reuse\",\"case\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"same_command_buffer\":true,\"cb_count\":1,\"encoder_dispatch_type_verified\":true,\"sequence\":[1,3,3],\"snapshot_copies_add_dependencies\":true,\"both_arms_input_raw_and_snapshot_war\":true,\"both_arms_production_caller_output_join\":true,\"candidate_internal_output_join\":%s,\"missing_caller_join_safety_proven\":false,\"reference_cache_is_oracle\":true,\"requests\":7,\"encoded\":7,\"failures\":0,\"cb_status\":%lu,\"items\":[",n,n/2u,n%2u,n%2u?"true":"false",(unsigned long)cb.status);
    for(unsigned i=0;i<7;i++){bool first=true;char stem[64];snprintf(stem,sizeof(stem),"reuse%u-item%u",n,i);bool same=true;for(unsigned k=0;k<3;k++)same&=!memcmp(myds_metal_buffer_data(snap[0][i][k].view),myds_metal_buffer_data(snap[1][i][k].view),snap[0][i][k].bytes);
        printf("%s{\"item\":%u,\"input_index\":%u,\"full_mid_cache_output_equal\":%s,\"finite\":true,\"canaries\":true,\"files\":[",i?",":"",i,seq[i],same?"true":"false");file_item(stem,"input",sources[seq[i]],"f32",&first);file_item(stem,"selected-ids",f->ids,"i32",&first);file_item(stem,"route-weights",f->weights,"f32",&first);
        for(unsigned a=0;a<2;a++)for(unsigned k=0;k<3;k++){char role[64];snprintf(role,sizeof(role),"%s-%s",a?"candidate":"reference",k==0?"mid":k==1?"cache":"output");file_item(stem,role,snap[a][i][k],"f32",&first);}printf("]}");
    }puts("]}");
    for(unsigned i=0;i<7;i++)for(unsigned k=0;k<3;k++){equal(snap[0][i][k],snap[1][i][k],true);for(unsigned j=0;j<i;j++)if(seq[i]==seq[j])equal(snap[0][i][k],snap[0][j][k],true);}
    for(unsigned a=0;a<2;a++)for(unsigned i=0;i<7;i++)for(unsigned k=0;k<3;k++)release(snap[a][i][k]);for(unsigned i=0;i<3;i++)release(sources[i]);
}
static void rejection(struct fixture *f,unsigned n){
    active_stage="rejected";active_case=n;    static const char *names[]={"short-output","short-mid","short-input","unaligned-input","unaligned-mid","output-input-alias","mid-output-alias","negative-clamp","nan-clamp","missing-prepare-pso","missing-down-pso","missing-cache"};
    REQUIRE(n<12);policy(n%2u,0);f->clamp=10.f;poison(f->mid[1]);poison(f->out[1]);memset(g_moe_q2_sum_cache_buffer.contents,0xa5,CACHE_BYTES);
    myds_metal_buffer_t out=*f->out[1].view,mid=*f->mid[1].view,x=*f->x.view;
    switch(n){case 0:out.size-=4;break;case 1:mid.size-=4;break;case 2:x.size-=4;break;case 3:x.offset+=4;x.ptr=(uint8_t*)x.ptr+4;break;case 4:mid.offset+=1;mid.ptr=(uint8_t*)mid.ptr+1;break;case 5:out=*f->x.view;break;case 6:out=*f->mid[1].view;break;case 7:f->clamp=-1.f;break;case 8:f->clamp=NAN;break;default:break;}
    id<MTLComputePipelineState> prep=g_pso_moe_q2_sum_cache_prepare,down=g_pso_moe_q2_sum6_cached;id<MTLBuffer> cache=g_moe_q2_sum_cache_buffer;
    if(n==9)g_pso_moe_q2_sum_cache_prepare=nil;if(n==10)g_pso_moe_q2_sum6_cached=nil;if(n==11)g_moe_q2_sum_cache_buffer=nil;
    const uint64_t x_before=fnv(myds_metal_buffer_data(f->x.view),OUT_BYTES);myds_metal_dispatch_count_reset();myds_metal_moe_q2_sum_cache_counters(NULL,true);REQUIRE(myds_metal_begin_batch());
    int ok=myds_metal_routed_moe_scoped(&out,&x,f->gate.view,f->up.view,f->down.view,f->ids.view,f->weights.view,f->sg[1].view,f->su[1].view,&mid,f->sd[1].view,DIM,WIDTH,EXPERTS,ACTIVE,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,f->clamp,true);
    uint64_t dispatches=myds_metal_dispatch_count();g_pso_moe_q2_sum_cache_prepare=prep;g_pso_moe_q2_sum6_cached=down;g_moe_q2_sum_cache_buffer=cache;REQUIRE(myds_metal_commit_batch());
    myds_metal_moe_q2_sum_cache_counters_t c;myds_metal_moe_q2_sum_cache_counters(&c,false);copy_actual_cache(f->cache[1]);
    char stem[64];snprintf(stem,sizeof(stem),"rejected%u",n);bool first=true;
    printf("{\"schema_version\":1,\"kind\":\"rejected_call\",\"case\":%u,\"reason\":\"%s\",\"stage\":\"preflight\",\"returncode\":%d,\"dispatches\":%" PRIu64 ",\"partial_write\":false,\"fallback\":false,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"input_unchanged\":%s,\"files\":[",n,names[n],ok,dispatches,c.requests,c.encoded,c.failures,x_before==fnv(myds_metal_buffer_data(f->x.view),OUT_BYTES)?"true":"false");file_item(stem,"candidate-mid",f->mid[1],"bytes",&first);file_item(stem,"candidate-cache",f->cache[1],"bytes",&first);file_item(stem,"candidate-output",f->out[1],"bytes",&first);puts("]}");
    REQUIRE(!ok&&!dispatches&&c.requests==1&&!c.encoded&&c.failures==1);REQUIRE(x_before==fnv(myds_metal_buffer_data(f->x.view),OUT_BYTES));untouched(f->mid[1]);untouched(f->cache[1]);untouched(f->out[1]);f->clamp=10.f;
}
static void completion_contract(struct fixture *f){
    active_stage="completion";active_case=0;policy(1,0);f->clamp=10.f;input(f,31);
    pair_once(f,0,false,1,NULL,NULL);reference_cache(f,0);
    myds_metal_moe_q2_sum_cache_counters(NULL,true);myds_metal_dispatch_count_reset();REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    REQUIRE(helper(f,1,true));actual_encoder();caller_join(f,1);REQUIRE(myds_metal_dispatch_count()==3);counter_check(1);
    myds_metal_inject_commit_failure(1);int completed=myds_metal_commit_batch();myds_metal_inject_commit_failure(0);
    REQUIRE(!completed&&cb.status==MTLCommandBufferStatusCompleted);counter_check(1);copy_actual_cache(f->cache[1]);
    bool same=!memcmp(myds_metal_buffer_data(f->mid[0].view),myds_metal_buffer_data(f->mid[1].view),MID_BYTES)&&!memcmp(myds_metal_buffer_data(f->cache[0].view),myds_metal_buffer_data(f->cache[1].view),CACHE_BYTES)&&!memcmp(myds_metal_buffer_data(f->out[0].view),myds_metal_buffer_data(f->out[1].view),OUT_BYTES);
    printf("{\"schema_version\":1,\"kind\":\"completion_contract\",\"injected_host_commit_failure\":true,\"real_gpu_failure\":false,\"caller_returncode\":0,\"encoded_is_not_completion_success\":true,\"encoder_dispatch_type_verified\":true,\"cb_status\":4,\"dispatches\":3,\"requests\":1,\"encoded\":1,\"failures\":0,\"partial_write\":true,\"all_three_stages_actually_completed\":true,\"full_mid_cache_output_equal\":%s,\"finite\":true,\"canaries\":true",same?"true":"false");files_numeric("completion",f,true,false);puts("}");
    equal(f->mid[0],f->mid[1],true);equal(f->cache[0],f->cache[1],true);equal(f->out[0],f->out[1],true);
}
static void timing(struct fixture *f,unsigned samples,unsigned repetitions){
    policy(1,0);f->clamp=10.f;const unsigned sets[]={1,repetitions};unsigned count=repetitions==1?1:2;
    for(unsigned r=0;r<count;r++)for(unsigned arm=0;arm<3;arm++)for(unsigned s=0;s<=samples;s++){
        active_stage="timing";active_case=r*100u+arm*10u+s;input(f,1001+s*17);pair_once(f,0,false,1,NULL,NULL);reference_cache(f,0);
        double start=0,end=0;pair_once(f,1,arm==1,sets[r],&start,&end);if(arm!=1)reference_cache(f,1);
        bool same=!memcmp(myds_metal_buffer_data(f->mid[0].view),myds_metal_buffer_data(f->mid[1].view),MID_BYTES)&&!memcmp(myds_metal_buffer_data(f->cache[0].view),myds_metal_buffer_data(f->cache[1].view),CACHE_BYTES)&&!memcmp(myds_metal_buffer_data(f->out[0].view),myds_metal_buffer_data(f->out[1].view),OUT_BYTES);
        REQUIRE(isfinite(start)&&isfinite(end)&&start>0&&end>start);const char *name=arm==0?"A1":arm==1?"B":"A2";char stem[80];snprintf(stem,sizeof(stem),"timing-r%u-%s-s%u",sets[r],name,s);
        printf("{\"schema_version\":1,\"kind\":\"gpu_timing\",\"arm\":\"%s\",\"sample\":%u,\"warmup\":%s,\"repetitions\":%u,\"actual_sum_cache\":%u,\"concurrent\":1,\"auto_barrier\":0,\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"gpu_start_s\":%.17g,\"gpu_end_s\":%.17g,\"us_per_pair\":%.17g,\"dispatches\":%u,\"requests\":%u,\"encoded\":%u,\"failures\":0,\"full_mid_cache_output_equal\":%s,\"finite\":true,\"canaries\":true,\"hist_disabled\":true,\"both_arms_production_caller_output_join\":true,\"candidate_internal_output_join\":%s,\"missing_caller_join_safety_proven\":false,\"oracle_and_snapshot_outside_timed_cb\":true,\"actual_cache_available\":%s,\"fixture_not_fullmodel\":true,\"cold_dram_not_proven\":true",name,s,s?"false":"true",sets[r],arm==1,start,end,(end-start)*1e6/sets[r],(arm==1?3:2)*sets[r],arm==1?sets[r]:0,arm==1?sets[r]:0,same?"true":"false","false",arm==1?"true":"false");files_numeric(stem,f,true,false);puts("}");
        equal(f->mid[0],f->mid[1],true);equal(f->cache[0],f->cache[1],true);equal(f->out[0],f->out[1],true);
    }
}
static void clean_state(void){REQUIRE(!g_initialized&&!g_device&&!g_queue&&!g_library&&!g_pso_moe_q2_sum_cache_prepare&&!g_pso_moe_q2_sum6_cached&&!g_moe_q2_sum_cache_buffer&&!myds_metal_moe_q2_sum_cache_enabled());oracle_pso=nil;}
static void initialize(const char *shader,unsigned mode){REQUIRE(setenv("MYDS_MOE_Q2_SUM_CACHE",mode?"1":"0",1)==0);REQUIRE(myds_metal_init(shader));REQUIRE(myds_metal_moe_q2_sum_cache_enabled()==(int)mode);}
static void load_oracle(const char *shader){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/q2_sum_cache_oracle.metallib",shader)<(int)sizeof(path));NSError *err=nil;id<MTLLibrary> lib=[g_device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&err];if(!lib)fprintf(stderr,"oracle load: %s\n",err.localizedDescription.UTF8String);REQUIRE(lib);id<MTLFunction> fn=[lib newFunctionWithName:@"q2_original_sumy_oracle"];REQUIRE(fn);oracle_pso=[g_device newComputePipelineStateWithFunction:fn error:&err];REQUIRE(oracle_pso&&oracle_pso.threadExecutionWidth==32&&oracle_pso.maxTotalThreadsPerThreadgroup>=32);}
static void host_config(void){
    const char *bad[]={"","01","-1","2","true","false"," 1","1 ","+1","0x1","1\n","0\n"};
    REQUIRE(unsetenv("MYDS_MOE_Q2_SUM_CACHE")==0);REQUIRE(moe_q2_sum_cache_parse_mode()==0);
    for(unsigned i=0;i<12;i++){REQUIRE(setenv("MYDS_MOE_Q2_SUM_CACHE",bad[i],1)==0);REQUIRE(moe_q2_sum_cache_parse_mode()==-1);REQUIRE(!myds_metal_init(NULL));clean_state();}
    for(unsigned i=0;i<2;i++){REQUIRE(setenv("MYDS_MOE_Q2_SUM_CACHE",i?"1":"0",1)==0);REQUIRE(moe_q2_sum_cache_parse_mode()==(int)i);}REQUIRE(unsetenv("MYDS_MOE_Q2_SUM_CACHE")==0);
    unsigned seen[1728]={0};for(unsigned s=0;s<6;s++)for(unsigned b=0;b<9;b++)for(unsigned it=0;it<8;it++)for(unsigned k=0;k<4;k++){unsigned slot=(s*9+b)*32+16*(it/4)+it%4+4*k;REQUIRE(slot<1728&&!seen[slot]++);}for(unsigned i=0;i<1728;i++)REQUIRE(seen[i]==1);
    puts("{\"schema_version\":1,\"kind\":\"host_config\",\"invalid_flags\":12,\"actual_private_parser\":true,\"rejected_before_metal_resources\":true,\"cache_slots\":1728,\"unique_complete_cache_layout\":true,\"pass\":true}");
}
static void missing_library(const char *shader){
    REQUIRE(setenv("MYDS_MOE_Q2_SUM_CACHE","1",1)==0);int accepted=myds_metal_init(shader);REQUIRE(!accepted);clean_state();
    puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"missing-explicit1\",\"requested\":1,\"accepted\":false,\"cleanup_complete\":true,\"fallback\":false}");
    initialize(shader,0);REQUIRE(g_pso_moe_pair_swiglu_iq2_xxs&&g_pso_moe_sum6_q2_k);REQUIRE(![g_library newFunctionWithName:@"kernel_dsv41_q2_sum_cache_prepare_f32"]);REQUIRE(![g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_cached_f32"]);myds_metal_cleanup();clean_state();
    puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"missing-reference0\",\"requested\":0,\"accepted\":true,\"cleanup_complete\":true,\"original_pipelines_available\":true,\"both_candidate_symbols_missing\":true}");
    REQUIRE(unsetenv("MYDS_MOE_Q2_SUM_CACHE")==0);REQUIRE(myds_metal_init(shader));REQUIRE(!myds_metal_moe_q2_sum_cache_enabled());myds_metal_cleanup();clean_state();
    puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"missing-default-reinit\",\"requested\":0,\"accepted\":true,\"cleanup_complete\":true,\"original_pipelines_available\":true,\"both_candidate_symbols_missing\":true}");
    puts("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"missing-candidate\",\"initializations\":3,\"raw_files\":0,\"pass\":true}");
}
int main(int argc,char **argv){@autoreleasepool{
    if(argc==3&&!strcmp(argv[1],"--write-oracle-source")){FILE *f=fopen(argv[2],"wb");REQUIRE(f);REQUIRE(fwrite(oracle_source,1,strlen(oracle_source),f)==strlen(oracle_source));REQUIRE(!fclose(f));return 0;}
    const char *shader=NULL;unsigned samples=5,repetitions=32;bool host=false,no_timing=false,missing=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--host-only"))host=true;else if(!strcmp(argv[i],"--no-timing"))no_timing=true;else if(!strcmp(argv[i],"--expect-missing-candidate"))missing=true;
        else if(i+1<argc&&!strcmp(argv[i],"--shader-dir"))shader=argv[++i];else if(i+1<argc&&!strcmp(argv[i],"--output-dir"))directory=argv[++i];
        else if(i+1<argc&&(!strcmp(argv[i],"--samples")||!strcmp(argv[i],"--repetitions"))){bool s=!strcmp(argv[i],"--samples");char *end=NULL;unsigned long value=strtoul(argv[++i],&end,10);REQUIRE(end&&!*end&&value>=1&&value<=1024);if(s)samples=(unsigned)value;else repetitions=(unsigned)value;}
        else{fprintf(stderr,"Usage: %s --shader-dir DIR --output-dir NEWDIR [--samples 5 --repetitions 32] [--host-only|--no-timing|--expect-missing-candidate]\n",argv[0]);return 2;}
    }
    REQUIRE(unsetenv("MYDS_METAL_CONCURRENT")==0);REQUIRE(unsetenv("MYDS_DISPATCH_HIST")==0);REQUIRE(unsetenv("MYDS_PROF")==0);REQUIRE(unsetenv("MYDS_MOE_SPLIT")==0);REQUIRE(unsetenv("MYDS_MOE_STEP_PROF")==0);REQUIRE(unsetenv("MYDS_MOE_MM_MIN_TOKENS")==0);REQUIRE(unsetenv("MYDS_DISABLE_MOE_MM")==0);
    REQUIRE(setenv("MYDS_MOE_NSG","4",1)==0);REQUIRE(setenv("MYDS_MOE_DOWN_NSG","4",1)==0);REQUIRE(setenv("MYDS_MOE_IQ2_CONSTANT_LUT","0",1)==0);REQUIRE(setenv("MYDS_MOE_IQ2_TAIL_CULL","0",1)==0);g_hist_enabled=0;
    host_config();if(host)return 0;REQUIRE(shader&&directory);REQUIRE(!mkdir(directory,0700));if(missing){missing_library(shader);return 0;}
    initialize(shader,0);policy(1,0);struct fixture original=create();active_fixture=&original;active_stage="init0";pair_once(&original,0,false,1,NULL,NULL);
    void *oldmid=malloc(MID_BYTES),*oldout=malloc(OUT_BYTES);REQUIRE(oldmid&&oldout);memcpy(oldmid,myds_metal_buffer_data(original.mid[0].view),MID_BYTES);memcpy(oldout,myds_metal_buffer_data(original.out[0].view),OUT_BYTES);
    REQUIRE(!setenv("MYDS_MOE_Q2_SUM_CACHE","1",1));REQUIRE(myds_metal_init(shader));REQUIRE(!myds_metal_moe_q2_sum_cache_enabled());destroy(&original);active_fixture=NULL;myds_metal_cleanup();clean_state();
    puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"reference0\",\"requested\":0,\"accepted\":true,\"immutable_after_env_change\":true,\"cleanup_complete\":true}");
    initialize(shader,1);load_oracle(shader);REQUIRE(!setenv("MYDS_MOE_Q2_SUM_CACHE","0",1));REQUIRE(myds_metal_init(shader));REQUIRE(myds_metal_moe_q2_sum_cache_enabled()==1);struct fixture f=create();active_fixture=&f;active_stage="init1-bypass";pair_once(&f,0,false,1,NULL,NULL);REQUIRE(!memcmp(oldmid,myds_metal_buffer_data(f.mid[0].view),MID_BYTES)&&!memcmp(oldout,myds_metal_buffer_data(f.out[0].view),OUT_BYTES));free(oldmid);free(oldout);
    printf("{\"schema_version\":1,\"kind\":\"identity\",\"dim\":5120,\"width\":2304,\"active\":6,\"experts\":384,\"tokens\":1,\"gate_type\":\"IQ2_XXS\",\"up_type\":\"IQ2_XXS\",\"down_type\":\"Q2_K\",\"iq2_nsg\":%d,\"down_nsg\":%d,\"constant_lut\":false,\"gate_bytes\":%zu,\"up_bytes\":%zu,\"down_bytes\":%zu,\"selected_experts_only_initialized\":true,\"cache_bytes\":6912,\"mid_bytes\":55296,\"output_bytes\":20480,\"iq2_grid\":[144,1,6],\"iq2_threads\":[32,4,1],\"prepare_grid\":[4,1,1],\"prepare_threads\":[128,1,1],\"down_grid\":[640,1,1],\"down_threads\":[32,4,1],\"oracle_compile_flags\":\"-O3 -ffast-math\",\"oracle_kernel\":\"q2_original_sumy_oracle\",\"reference0_equals_scoped_bypass1\":true,\"actual_immutable_mode\":1,\"fullmodel\":false,\"cold_dram_not_proven\":true}\n",g_moe_nsg,g_moe_down_nsg,f.gate.bytes,f.up.bytes,f.down.bytes);
    for(unsigned n=0;n<8;n++)parity_case(&f,n);for(unsigned n=0;n<4;n++)cache_case(&f,n);for(unsigned n=0;n<4;n++)reuse_case(&f,n);for(unsigned n=0;n<12;n++)rejection(&f,n);completion_contract(&f);if(!no_timing)timing(&f,samples,repetitions);
    destroy(&f);active_fixture=NULL;myds_metal_cleanup();clean_state();puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"candidate1\",\"requested\":1,\"accepted\":true,\"immutable_after_env_change\":true,\"cleanup_complete\":true}");
    REQUIRE(unsetenv("MYDS_MOE_Q2_SUM_CACHE")==0);REQUIRE(myds_metal_init(shader));REQUIRE(!myds_metal_moe_q2_sum_cache_enabled());myds_metal_cleanup();clean_state();puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"default0-reinit\",\"requested\":0,\"accepted\":true,\"cleanup_complete\":true}");
    printf("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"component\",\"parity_cases\":8,\"cache_cases\":4,\"reuse_cases\":4,\"reuse_items\":28,\"rejected_calls\":12,\"completion_contracts\":1,\"timing_records\":%u,\"samples\":%u,\"repetitions\":%u,\"initializations\":3,\"raw_files\":%u,\"pass\":true}\n",no_timing?0:(repetitions==1?1:2)*3*(samples+1),samples,repetitions,raw_files);return 0;
}}
