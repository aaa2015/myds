/* Model-free acceptance of the production Batch2 TAIL selector and real
 * Metal helper. Includes the runtime to inspect PSOs/counters and exercise a
 * genuinely missing candidate function. Link the remaining engine objects.
 * Default invocation runs NR2 then NR4 in separate processes, respecting the
 * existing process-once MYDS_QKV_NR0 and decode-fuse policies. */
#include "myds_metal.m"
#include "myds_v41.h"
#include "myds_v41_decode.h"
#include "myds_internal.h"
#include <inttypes.h>
#include <sys/stat.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
struct guarded { myds_metal_buffer_t *owner,*view; size_t bytes,offset; };
struct fixture { uint32_t rows,width; struct guarded weight,input[2],ref[2],out[2]; };
static const char *output_dir,*shader_dir="bin",*policy="eligible";
static uint32_t samples=5,repetitions=32,nr;
static int timing_enabled=1,host_only,expect_missing;
static uint64_t fnv(const void *data,size_t size) {
    const uint8_t *p=data;uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<size;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static struct guarded allocate(size_t bytes,size_t offset) {
    struct guarded b={.bytes=bytes,.offset=offset};b.owner=myds_metal_buffer_alloc(bytes+offset+16u);REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+offset+16u);
    b.view=myds_metal_buffer_sub_view(b.owner,offset,bytes);REQUIRE(b.view);return b;
}
static void canary(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<b.offset;i++)REQUIRE(p[i]==0xa5);
    for(size_t i=0;i<16u;i++)REQUIRE(p[b.offset+b.bytes+i]==0xa5);
}
static void release(struct guarded b) {canary(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void reset_output(struct fixture *f) {for(unsigned t=0;t<2;t++)memset(myds_metal_buffer_data(f->out[t].view),0xa5,f->out[t].bytes);}
static void untouched(struct fixture *f) {
    for(unsigned t=0;t<2;t++){const uint8_t *p=myds_metal_buffer_data(f->out[t].view);for(size_t i=0;i<f->out[t].bytes;i++)REQUIRE(p[i]==0xa5);canary(f->out[t]);}
}
static void init_fixture(struct fixture *f,uint32_t rows,uint32_t width,unsigned seed,unsigned mode) {
    memset(f,0,sizeof(*f));f->rows=rows;f->width=width;
    f->weight=allocate((size_t)rows*(width/32u)*34u,18u);
    uint8_t *w=myds_metal_buffer_data(f->weight.view);
    for(size_t b=0;b<f->weight.bytes/34u;b++){
        uint16_t scale=(uint16_t)(0x1400u+(b%8u)*0x180u);memcpy(w+34u*b,&scale,2u);
        for(unsigned i=0;i<32u;i++)w[34u*b+2u+i]=(uint8_t)(int8_t)((int)((b*37u+i*29u+seed)%255u)-127);
    }
    for(unsigned t=0;t<2;t++){
        f->input[t]=allocate((size_t)width*4u,20u+4u*t);
        f->ref[t]=allocate((size_t)rows*4u,16u+16u*t);f->out[t]=allocate((size_t)rows*4u,32u+16u*t);
        float *x=myds_metal_buffer_data(f->input[t].view);
        for(uint32_t i=0;i<width;i++){
            uint32_t z=i*747796405u+(seed+113u*t)*2891336453u;z=((z>>((z>>28)+4))^z)*277803737u;z=(z>>22)^z;
            x[i]=mode==2?0.0f:((int)(z%1023u)-511)/(mode==3?128.0f:257.0f);
        }
        if(mode==1&&t==1) {const float *x0=myds_metal_buffer_data(f->input[0].view);for(uint32_t i=0;i<width;i++)x[i]=-x0[i];}
    }
}
static void free_fixture(struct fixture *f) {release(f->weight);for(unsigned t=0;t<2;t++){release(f->input[t]);release(f->ref[t]);release(f->out[t]);}}
static void export_pair(struct fixture *f,unsigned id) {
    if(!output_dir)return;
    for(unsigned t=0;t<2;t++)for(unsigned c=0;c<2;c++){
        char path[4096];int n=snprintf(path,sizeof(path),"%s/case%u-%s-token%u.f32",output_dir,id,c?"candidate":"reference",t);REQUIRE(n>0&&(size_t)n<sizeof(path));
        struct guarded b=c?f->out[t]:f->ref[t];FILE *file=fopen(path,"wb");REQUIRE(file);REQUIRE(fwrite(myds_metal_buffer_data(b.view),1,b.bytes,file)==b.bytes);REQUIRE(!fclose(file));
    }
}
static void compare(struct fixture *f) {
    for(unsigned t=0;t<2;t++){
        const float *a=myds_metal_buffer_data(f->ref[t].view),*b=myds_metal_buffer_data(f->out[t].view);
        if(memcmp(a,b,f->ref[t].bytes))for(uint32_t i=0;i<f->rows;i++)if(memcmp(a+i,b+i,4u)){
            uint32_t x,y;memcpy(&x,a+i,4u);memcpy(&y,b+i,4u);fprintf(stderr,"PARITY token=%u index=%u ref=%08x candidate=%08x rows=%u width=%u\n",t,i,x,y,f->rows,f->width);break;}
        REQUIRE(!memcmp(a,b,f->ref[t].bytes));
        for(uint32_t i=0;i<f->rows;i++){uint32_t bits;memcpy(&bits,b+i,4u);REQUIRE(isfinite(a[i])&&isfinite(b[i])&&bits!=0xa5a5a5a5u);REQUIRE((bits&0xffffu)==0);}
        canary(f->ref[t]);canary(f->out[t]);canary(f->input[t]);
    }canary(f->weight);
}
static uint64_t histogram(const char *name) {for(int i=0;i<g_hist_used;i++)if(!strcmp(g_hist[i].name,name))return g_hist[i].n;return 0;}
static int projected(struct fixture *f,myds_v41_batch2_projection_kind_t kind) {
    myds_metal_buffer_t *out[]={f->out[0].view,f->out[1].view},*input[]={f->input[0].view,f->input[1].view};
    return myds_v41_batch2_projection_probe(out,f->weight.view,input,f->rows,f->width,GGML_TYPE_Q8_0,kind);
}
static void reference(struct fixture *f,myds_v41_batch2_projection_kind_t kind) {
    REQUIRE(myds_metal_begin_batch());
    for(unsigned t=0;t<2;t++){
        if(kind==MYDS_V41_BATCH2_QKV)REQUIRE(myds_metal_v41_gemv_bf16(f->ref[t].view,f->weight.view,f->input[t].view,f->rows,f->width,GGML_TYPE_Q8_0));
        else REQUIRE(myds_v41_decode_gemv_bf16(f->ref[t].view,f->weight.view,f->input[t].view,f->rows,f->width,GGML_TYPE_Q8_0,myds_trace_active()));
    }REQUIRE(myds_metal_commit_batch());
}
static void numeric(unsigned case_id,uint32_t rows,uint32_t width,unsigned mode,myds_v41_batch2_projection_kind_t kind,int shared_expected) {
    struct fixture f;init_fixture(&f,rows,width,31u+case_id,mode);reference(&f,kind);
    myds_metal_batch2_q8_shared_counters_t before,after;myds_metal_batch2_q8_shared_counters(&before,false);
    const unsigned nsg=rows>65536u?8u:4u;char name[128];snprintf(name,sizeof(name),"kernel_dsv41_q8_shared2_bf16_r%u",nr);
    const uint64_t oldhist=histogram(name);REQUIRE(myds_metal_begin_batch());const int ok=projected(&f,kind),done=myds_metal_commit_batch();
    export_pair(&f,case_id);REQUIRE(ok&&done);compare(&f);myds_metal_batch2_q8_shared_counters(&after,false);
    REQUIRE(after.requests-before.requests==(uint64_t)shared_expected&&after.encoded-before.encoded==(uint64_t)shared_expected&&after.failures==before.failures);
    REQUIRE(histogram(name)-oldhist==(uint64_t)shared_expected);
    id<MTLComputePipelineState> p=shared_expected?batch2_q8_shared_pipeline(nsg,nr):nil;
    printf("{\"kind\":\"parity\",\"case\":%u,\"rows\":%u,\"width\":%u,\"nr0\":%u,\"nsg\":%u,\"projection_kind\":%u,\"policy\":\"%s\",\"actual_shared_dispatches\":%d,\"kernel\":\"%s\",\"pso_max_threads\":%lu,\"threads\":%u,\"dynamic_bytes\":%u,\"full_output_floats\":%u,\"bit_exact\":true,\"finite\":true,\"canaries\":true,\"fnv_token0\":\"%016" PRIx64 "\",\"fnv_token1\":\"%016" PRIx64 "\"}\n",case_id,rows,width,nr,nsg,kind,policy,shared_expected,shared_expected?name:"scalar_reference",(unsigned long)p.maxTotalThreadsPerThreadgroup,shared_expected?32u*nsg:0u,shared_expected?32u*nr*2u*4u:0u,2u*rows,fnv(myds_metal_buffer_data(f.out[0].view),f.out[0].bytes),fnv(myds_metal_buffer_data(f.out[1].view),f.out[1].bytes));
    free_fixture(&f);
}
static void reject_call(struct fixture *f,myds_metal_buffer_t *out[2],const myds_metal_buffer_t *w,myds_metal_buffer_t *in[2],uint32_t rows,uint32_t width) {
    reset_output(f);uint64_t dispatches=myds_metal_dispatch_count();myds_metal_batch2_q8_shared_counters_t a,b;myds_metal_batch2_q8_shared_counters(&a,false);
    REQUIRE(!myds_metal_v41_gemv_bf16_shared2(out,w,in,rows,width));myds_metal_batch2_q8_shared_counters(&b,false);
    REQUIRE(b.requests==a.requests+1u&&b.failures==a.failures+1u&&b.encoded==a.encoded&&myds_metal_dispatch_count()==dispatches);untouched(f);
}
static void rejections(void) {
    struct fixture f;init_fixture(&f,17,128,79,0);myds_metal_buffer_t *out[]={f.out[0].view,f.out[1].view},*in[]={f.input[0].view,f.input[1].view};
    myds_metal_buffer_t badout=*out[0],badin=*in[0],badw=*f.weight.view;
    badout.size-=4;myds_metal_buffer_t *o[]={&badout,out[1]};reject_call(&f,o,f.weight.view,in,17,128);
    badin.size-=4;myds_metal_buffer_t *x[]={&badin,in[1]};reject_call(&f,out,f.weight.view,x,17,128);
    badw.size-=2;reject_call(&f,out,&badw,in,17,128);
    badout=*out[0];badout.offset+=4;badout.ptr=(char*)badout.ptr+4;reject_call(&f,o,f.weight.view,in,17,128);
    o[0]=out[0];o[1]=out[0];reject_call(&f,o,f.weight.view,in,17,128);
    o[0]=in[0];o[1]=out[1]; /* Input offset20 also rejects alignment; use aligned subview below. */
    myds_metal_buffer_t alias=*in[0];alias.offset=16;alias.ptr=(char*)myds_metal_buffer_data(f.input[0].owner)+16;alias.size=68;o[0]=&alias;reject_call(&f,o,f.weight.view,in,17,128);
    alias=*f.weight.view;alias.offset=16;alias.ptr=(char*)myds_metal_buffer_data(f.weight.owner)+16;alias.size=68;reject_call(&f,o,f.weight.view,in,17,128);
    reject_call(&f,out,f.weight.view,in,0,128);reject_call(&f,out,f.weight.view,in,17,127);
    const char *invalid[]={"","00","01","2","true","+1"," 1","1 "};
    for(unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);i++){
        setenv("MYDS_BATCH2_Q8_SHARED",invalid[i],1);reset_output(&f);uint64_t count=myds_metal_dispatch_count();REQUIRE(!projected(&f,MYDS_V41_BATCH2_TAIL));REQUIRE(myds_metal_dispatch_count()==count);untouched(&f);
    }
    setenv("MYDS_BATCH2_Q8_SHARED","1",1);
    /* Actual library without the candidate symbol. Retain all previously
     * initialized scalar PSOs; no substitution can be accepted as success. */
    id<MTLLibrary> original=g_library;NSString *key=[NSString stringWithFormat:@"batch2_shared_q8_4_%u",nr];
    id<MTLComputePipelineState> saved=g_native_pipelines[key];REQUIRE(saved);[g_native_pipelines removeObjectForKey:key];REQUIRE(!g_native_pipelines[key]);
    NSError *error=nil;g_library=[g_device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void unrelated(device float *x [[buffer(0)]],uint i [[thread_position_in_grid]]) {x[i]=0;}\n" options:nil error:&error];REQUIRE(g_library);
    reset_output(&f);myds_metal_batch2_q8_shared_counters_t a,b;myds_metal_batch2_q8_shared_counters(&a,false);uint64_t count=myds_metal_dispatch_count();
    REQUIRE(!projected(&f,MYDS_V41_BATCH2_TAIL));myds_metal_batch2_q8_shared_counters(&b,false);REQUIRE(b.requests==a.requests+1&&b.failures==a.failures+1&&b.encoded==a.encoded&&count==myds_metal_dispatch_count());untouched(&f);
    g_library=original;g_native_pipelines[key]=saved;
    printf("{\"kind\":\"rejection\",\"nr0\":%u,\"range_alignment_dimension_cases\":6,\"alias_cases\":3,\"invalid_policy_values\":8,\"actual_missing_pso\":true,\"fallback_dispatches\":0,\"output_untouched\":true}\n",nr);free_fixture(&f);
}
static void reuse(void) {
    struct fixture f[3];const uint32_t widths[]={32,128,8192};
    for(unsigned j=0;j<3;j++){init_fixture(f+j,j==2?5120u:17u,widths[j],13u+j,0);reference(f+j,MYDS_V41_BATCH2_TAIL);}
    const unsigned counts[]={1,3,3};myds_metal_batch2_q8_shared_counters_t a,b;myds_metal_batch2_q8_shared_counters(&a,false);
    for(unsigned pass=0;pass<3;pass++){REQUIRE(myds_metal_begin_batch());for(unsigned j=0;j<counts[pass];j++)REQUIRE(projected(f+(pass==2?2u-j:j),MYDS_V41_BATCH2_TAIL));REQUIRE(myds_metal_commit_batch());for(unsigned j=0;j<counts[pass];j++)compare(f+(pass==2?2u-j:j));}
    myds_metal_batch2_q8_shared_counters(&b,false);REQUIRE(b.encoded-a.encoded==7&&b.requests-a.requests==7&&b.failures==a.failures);
    printf("{\"kind\":\"reuse\",\"nr0\":%u,\"calls_per_batch\":[1,3,3],\"auto_barrier\":%d,\"explicit_concurrent_mode\":%d,\"actual_shared_dispatches\":7,\"bit_exact\":true}\n",nr,myds_metal_get_auto_barrier(),myds_metal_get_concurrent());for(unsigned j=0;j<3;j++)free_fixture(f+j);
}
static void timing(const char *arm,unsigned shared) {
    struct fixture f;init_fixture(&f,5120,8192,42,0);reference(&f,MYDS_V41_BATCH2_TAIL);setenv("MYDS_BATCH2_Q8_SHARED",shared?"1":"0",1);
    REQUIRE(myds_metal_begin_batch());REQUIRE(projected(&f,MYDS_V41_BATCH2_TAIL));REQUIRE(myds_metal_commit_batch());compare(&f);
    for(unsigned sample=0;sample<=samples;sample++){
        myds_metal_batch2_q8_shared_counters_t a,b;myds_metal_batch2_q8_shared_counters(&a,false);uint64_t count=myds_metal_dispatch_count();
        REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned r=0;r<repetitions;r++){REQUIRE(projected(&f,MYDS_V41_BATCH2_TAIL));myds_metal_barrier();}
        REQUIRE(myds_metal_commit_batch());compare(&f);myds_metal_batch2_q8_shared_counters(&b,false);
        REQUIRE(b.requests-a.requests==(uint64_t)shared*repetitions&&b.encoded-a.encoded==(uint64_t)shared*repetitions&&b.failures==a.failures);
        REQUIRE(myds_metal_dispatch_count()-count==(uint64_t)repetitions*(shared?1u:2u));
        REQUIRE(cb.status==MTLCommandBufferStatusCompleted&&isfinite(cb.GPUStartTime)&&cb.GPUEndTime>cb.GPUStartTime);
        printf("{\"kind\":\"gpu_timing\",\"window\":%u,\"nr0\":%u,\"nsg\":4,\"arm\":\"%s\",\"shared\":%u,\"rows\":5120,\"width\":8192,\"warmup\":%s,\"sample\":%u,\"repetitions\":%u,\"encoded\":%" PRIu64 ",\"scalar_dispatches\":%u,\"gpu_us_per_pair\":%.9f,\"gpu_start_s\":%.9f,\"gpu_end_s\":%.9f,\"dispatch_hist_disabled\":true,\"bit_exact_full_output\":true}\n",nr==2?1u:2u,nr,arm,shared,sample==0?"true":"false",sample,repetitions,b.encoded-a.encoded,shared?0u:2u*repetitions,(cb.GPUEndTime-cb.GPUStartTime)*1e6/repetitions,cb.GPUStartTime,cb.GPUEndTime);
    }free_fixture(&f);
}
static void host_config(void) {
    unsetenv("MYDS_BATCH2_Q8_SHARED");REQUIRE(myds_metal_batch2_q8_shared_mode()==0);
    setenv("MYDS_BATCH2_Q8_SHARED","0",1);REQUIRE(!myds_metal_batch2_q8_shared_mode());setenv("MYDS_BATCH2_Q8_SHARED","1",1);REQUIRE(myds_metal_batch2_q8_shared_mode()==1);
    const char *bad[]={"","00","01","2","true","-1","+1"," 1","1 ","1\n","0x1","0 "};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++){setenv("MYDS_BATCH2_Q8_SHARED",bad[i],1);REQUIRE(!myds_metal_init("/path/never/read/for/invalid/config"));REQUIRE(!g_device&&!g_queue&&!g_library&&!g_initialized);}
    unsetenv("MYDS_BATCH2_Q8_SHARED");puts("{\"kind\":\"host_config\",\"valid_values\":[0,1],\"absent_default\":0,\"invalid_values_rejected_before_metal\":12}");
}
static uint32_t number(const char *text) {char *end=NULL;errno=0;unsigned long n=strtoul(text,&end,10);REQUIRE(!errno&&end!=text&&!*end&&n>=1&&n<=10000);return(uint32_t)n;}
static void directory(void) {if(!output_dir)return;struct stat st;errno=0;int e=lstat(output_dir,&st);REQUIRE(e!=0&&errno==ENOENT);REQUIRE(!mkdir(output_dir,0700));}
static int child(const char *program,unsigned row,const char *child_policy) {
    @autoreleasepool {NSTask *task=[NSTask new];task.executableURL=[NSURL fileURLWithPath:[NSString stringWithUTF8String:program]];
        NSMutableArray<NSString *> *args=[NSMutableArray arrayWithObjects:@"--nr0",[NSString stringWithFormat:@"%u",row],@"--policy",[NSString stringWithUTF8String:child_policy],@"--shader-dir",[NSString stringWithUTF8String:shader_dir],@"--samples",[NSString stringWithFormat:@"%u",samples],@"--repetitions",[NSString stringWithFormat:@"%u",repetitions],nil];
        if(output_dir){[args addObject:@"--output-dir"];[args addObject:[NSString stringWithFormat:@"%s/nr%u-%s",output_dir,row,child_policy]];}
        if(!timing_enabled)[args addObject:@"--no-timing"];if(expect_missing)[args addObject:@"--expect-missing-shared"];
        task.arguments=args;NSError *error=nil;fflush(stdout);fflush(stderr);REQUIRE([task launchAndReturnError:&error]);[task waitUntilExit];return task.terminationStatus;}
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--host-only"))host_only=1;
        else if(!strcmp(argv[i],"--no-timing"))timing_enabled=0;
        else if(!strcmp(argv[i],"--expect-missing-shared"))expect_missing=1;
        else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader_dir=argv[++i];
        else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)output_dir=argv[++i];
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=number(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)repetitions=number(argv[++i]);
        else if(!strcmp(argv[i],"--nr0")&&i+1<argc){nr=number(argv[++i]);REQUIRE(nr==2||nr==4);}
        else if(!strcmp(argv[i],"--policy")&&i+1<argc){policy=argv[++i];REQUIRE(!strcmp(policy,"eligible")||!strcmp(policy,"trace")||!strcmp(policy,"fuse-off"));}
        else{fprintf(stderr,"usage: %s [--shader-dir DIR] [--output-dir NEW_DIR] [--samples N] [--repetitions N] [--no-timing] [--host-only] [--nr0 2|4] [--policy eligible|trace|fuse-off] [--expect-missing-shared]\n",argv[0]);return 2;}
    }
    host_config();if(host_only){
        const char *value=getenv("MYDS_DECODE_Q8_FUSE_BF16");const int expected=!(value&&!strcmp(value,"0"));
        REQUIRE(myds_v41_decode_q8_fuse_bf16_enabled()==expected);
        setenv("MYDS_DECODE_Q8_FUSE_BF16",expected?"0":"1",1);REQUIRE(myds_v41_decode_q8_fuse_bf16_enabled()==expected);
        myds_metal_batch2_q8_shared_counters_t count;myds_metal_batch2_q8_shared_counters(NULL,true);
        REQUIRE(!myds_metal_v41_gemv_bf16_shared2(NULL,NULL,NULL,5120,8192));myds_metal_batch2_q8_shared_counters(&count,true);
        REQUIRE(count.requests==1&&count.encoded==0&&count.failures==1);myds_metal_batch2_q8_shared_counters(&count,false);REQUIRE(!count.requests&&!count.encoded&&!count.failures);
        REQUIRE(!g_device&&!g_queue&&!g_library&&!g_initialized);
        printf("{\"kind\":\"host_fuse_policy\",\"enabled\":%s,\"process_once_preserved\":true}\n",expected?"true":"false");return 0;
    }directory();
    if(!nr){REQUIRE(!child(argv[0],2,"eligible"));REQUIRE(!child(argv[0],4,"eligible"));if(!expect_missing){REQUIRE(!child(argv[0],4,"trace"));REQUIRE(!child(argv[0],4,"fuse-off"));}puts("{\"kind\":\"suite_complete\",\"nr0_processes\":[2,4],\"pass\":true}");return 0;}
    char text[8];snprintf(text,sizeof(text),"%u",nr);setenv("MYDS_QKV_NR0",text,1);setenv("MYDS_METAL_CONCURRENT","0",1);setenv("MYDS_QKV_FUSE_BF16","1",1);
    unsetenv("MYDS_TRACE_PREFIX");unsetenv("MYDS_BATCH_TRACE_PREFIX");if(!strcmp(policy,"trace"))setenv("MYDS_TRACE_PREFIX","fixture-no-trace-files",1);
    setenv("MYDS_DECODE_Q8_FUSE_BF16",!strcmp(policy,"fuse-off")?"0":"1",1);setenv("MYDS_BATCH2_Q8_SHARED","1",1);g_hist_enabled=1;
    if(expect_missing){REQUIRE(!myds_metal_init(shader_dir));REQUIRE(!g_initialized&&!g_device&&!g_library);setenv("MYDS_BATCH2_Q8_SHARED","0",1);REQUIRE(myds_metal_init(shader_dir));numeric(0,17,128,0,MYDS_V41_BATCH2_TAIL,0);puts("{\"kind\":\"missing_init\",\"explicit1_rejected\":true,\"default0_old_library_compatible\":true,\"pass\":true}");myds_metal_cleanup();return 0;}
    REQUIRE(myds_metal_init(shader_dir));myds_metal_batch2_q8_shared_counters(NULL,true);myds_metal_set_concurrent(0);
    printf("{\"kind\":\"identity\",\"device\":\"%s\",\"nr0\":%u,\"policy\":\"%s\",\"fixture_not_fullmodel\":true}\n",g_device.name.UTF8String,nr,policy);
    if(strcmp(policy,"eligible")){numeric(0,5120,8192,0,MYDS_V41_BATCH2_TAIL,0);myds_metal_cleanup();puts("{\"kind\":\"policy_bypass\",\"pass\":true}");return 0;}
    const uint32_t shapes[][2]={{1,32},{3,128},{5,256},{17,128},{5119,8192},{5120,8192},{5121,8192},{32768,1280},{65536,32},{65537,32}};
    unsigned id=0;
    for(unsigned c=0;c<2;c++){myds_metal_set_concurrent((int)c);myds_metal_set_auto_barrier(1);
        for(unsigned i=0;i<sizeof(shapes)/sizeof(*shapes);i++)numeric(id++,shapes[i][0],shapes[i][1],i%3u,MYDS_V41_BATCH2_TAIL,1);
        numeric(id++,5120,8192,0,MYDS_V41_BATCH2_TAIL,1);numeric(id++,5120,8192,1,MYDS_V41_BATCH2_TAIL,1);
        numeric(id++,5120,8192,3,MYDS_V41_BATCH2_TAIL,1);
        numeric(id++,17,128,0,MYDS_V41_BATCH2_QKV,1);numeric(id++,17,128,0,MYDS_V41_BATCH2_SHARED,1);rejections();
        for(unsigned a=0;a<2;a++){myds_metal_set_auto_barrier((int)a);reuse();}
    }
    myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);if(timing_enabled){g_hist_enabled=0;timing("A1",0);timing("B",1);timing("A2",0);}
    myds_metal_batch2_q8_shared_counters_t counters;myds_metal_batch2_q8_shared_counters(&counters,false);
    printf("{\"kind\":\"complete\",\"nr0\":%u,\"parity_cases\":%u,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures_expected\":%" PRIu64 ",\"pass\":true}\n",nr,id,counters.requests,counters.encoded,counters.failures);myds_metal_cleanup();return 0;
}
