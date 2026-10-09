/* Component-only packed attention gate. Includes the real runtime and GPU
 * entry points to inspect their private scratch/PSO/histogram without adding
 * production hooks. Build submits no GPU work. --host-only is CPU only.
 * Full mode submits GPU work and must run in an exclusive validation window.
 * Deterministic BF16-representable inputs are fixtures, not GGUF captures. */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <sys/stat.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); exit(1); } } while(0)
struct guarded { myds_metal_buffer_t *owner,*view; size_t bytes; };
struct fixture {
    uint32_t keys,nraw,start,selected,source,mode;
    myds_metal_buffer_t *q,*window,*comp,*gathered,*ids,*sinks;
    struct guarded out[5];
};
static const char *output_dir;
static const char *shader_dir="metal";
static uint32_t samples=3,repetitions=16;
static uint64_t fnv(const void *data,size_t n) {
    const uint8_t *p=data;uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;i++){ h^=p[i];h*=UINT64_C(1099511628211); }return h;
}
static struct guarded allocate(size_t bytes) {
    struct guarded b={.bytes=bytes};b.owner=myds_metal_buffer_alloc(bytes+32);REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+32);
    b.view=myds_metal_buffer_sub_view(b.owner,16,bytes);REQUIRE(b.view);return b;
}
static void canary(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<16;i++)REQUIRE(p[i]==0xa5 && p[16+b.bytes+i]==0xa5);
}
static float value(size_t i,unsigned seed) {
    uint32_t x=(uint32_t)i*747796405u+seed*2891336453u;x=((x>>((x>>28)+4))^x)*277803737u;x=(x>>22)^x;
    return ((int)(x%255u)-127)/128.0f; /* exact BF16 and F16 represented value */
}
static void fixture_init(struct fixture *f,uint32_t keys,uint32_t start,uint32_t mode) {
    memset(f,0,sizeof(*f));f->keys=keys;f->nraw=keys<128?keys:128;f->start=start;f->mode=mode;
    f->selected=keys-f->nraw;f->source=f->selected?2u*f->selected+17u:0;
#define BUF(field,bytes) do { f->field=myds_metal_buffer_alloc(bytes);REQUIRE(f->field); } while(0)
    BUF(q,64u*512u*4u);BUF(window,128u*512u*4u);BUF(sinks,64u*4u);
    if(f->selected){BUF(comp,(size_t)f->source*512u*2u);BUF(gathered,(size_t)f->selected*512u*4u);BUF(ids,(size_t)f->selected*4u);}
#undef BUF
    float *q=myds_metal_buffer_data(f->q),*w=myds_metal_buffer_data(f->window),*sink=myds_metal_buffer_data(f->sinks);
    for(size_t i=0;i<64u*512u;i++)q[i]=value(i,19u+mode);
    for(size_t i=0;i<128u*512u;i++)w[i]=value(i,37u+mode);
    for(unsigned h=0;h<64;h++)sink[h]=mode==4?-FLT_MAX/2:((int)h-31)/8.0f;
    if(f->selected){
        uint16_t *c=myds_metal_buffer_data(f->comp);float *g=myds_metal_buffer_data(f->gathered);int32_t *id=myds_metal_buffer_data(f->ids);
        for(size_t i=0;i<(size_t)f->source*512u;i++){float v=value(i,71u+mode);uint32_t bits;memcpy(&bits,&v,4);c[i]=(uint16_t)(bits>>16);}
        for(uint32_t j=0;j<f->selected;j++){
            id[j]=(int32_t)(((uint64_t)j*53u+19u)%f->source); /* noncontiguous, permuted selection */
            for(uint32_t k=0;k<j;k++)REQUIRE(id[k]!=id[j]);
            for(unsigned k=0;k<512;k++){uint32_t bits=(uint32_t)c[(size_t)id[j]*512u+k]<<16;memcpy(g+(size_t)j*512u+k,&bits,4);}
        }
    }
    for(unsigned i=0;i<5;i++)f->out[i]=allocate(64u*512u*4u);
}
static void fixture_free(struct fixture *f) {
    myds_metal_buffer_t *bs[]={f->q,f->window,f->comp,f->gathered,f->ids,f->sinks};
    for(size_t i=0;i<sizeof(bs)/sizeof(*bs);i++)myds_metal_buffer_free(bs[i]);
    for(unsigned i=0;i<5;i++){canary(f->out[i]);myds_metal_buffer_free(f->out[i].view);myds_metal_buffer_free(f->out[i].owner);}
}
static void prepare_mask(myds_v41_gpu_scratch_t *s,const struct fixture *f) {
    uint16_t *m=myds_metal_buffer_data(s->mask);
    for(unsigned j=0;j<640;j++){
        bool masked=f->mode==2 || (f->mode==1 && j%3==1) || (f->mode==3 && j<32);
        m[j]=masked?0xfbffu:(f->mode==1?0xbc00u:0u); /* -MAXHALF, -1, or zero */
    }
    memset(myds_metal_buffer_data(s->pad),0xa5,myds_metal_buffer_size(s->pad));
}
static const char *kernel_name(unsigned nsg) {
    switch(nsg){case 4:return "kernel_myds_flash_attn_vec_packed4_reduce_f16_dk512_dv512";
      case 8:return "kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512";
      case 16:return "kernel_myds_flash_attn_vec_packed16_reduce_f16_dk512_dv512";default:REQUIRE(0);return NULL;}
}
static uint64_t hist_count(const char *name) {
    for(int i=0;i<g_hist_used;i++)if(!strcmp(g_hist[i].name,name))return g_hist[i].n;return 0;
}
static void select_mode(unsigned nsg,int absent) {
    setenv("MYDS_ATTN_PACKED_REDUCE",nsg?"1":"0",1);
    if(absent)unsetenv("MYDS_ATTN_PACKED_NSG");
    else {char text[16];snprintf(text,sizeof(text),"%u",nsg?nsg:8u);setenv("MYDS_ATTN_PACKED_NSG",text,1);}
}
static int call(struct fixture *f,myds_v41_gpu_scratch_t *s,unsigned output,int direct) {
    return direct?myds_v41_gpu_attention_direct(f->out[output].view,f->sinks,f->q,f->window,f->nraw,f->start,
           f->comp,f->ids,f->source,f->selected,s):
       myds_v41_gpu_attention(f->out[output].view,f->sinks,f->q,f->window,f->nraw,f->start,f->gathered,f->selected,s);
}
static void compare(struct fixture *f,unsigned output) {
    const float *a=myds_metal_buffer_data(f->out[0].view),*b=myds_metal_buffer_data(f->out[output].view);
    if(memcmp(a,b,f->out[0].bytes)){
        for(size_t i=0;i<64u*512u;i++)if(memcmp(a+i,b+i,4)){
            uint32_t ab,bb;memcpy(&ab,a+i,4);memcpy(&bb,b+i,4);
            fprintf(stderr,"PARITY keys=%u start=%u mode=%u output=%u first_index=%zu ref=%08x candidate=%08x\n",f->keys,f->start,f->mode,output,i,ab,bb);break;}
        REQUIRE(0);
    }
    size_t nonzero=0;for(size_t i=0;i<64u*512u;i++){REQUIRE(isfinite(a[i])&&isfinite(b[i]));nonzero+=a[i]!=0;}
    if(f->mode!=2 && !(f->mode==3&&f->keys<=32))REQUIRE(nonzero);
    canary(f->out[0]);canary(f->out[output]);
}
static void export_out(struct fixture *f,unsigned output,unsigned nsg,int direct) {
    if(!output_dir)return;char path[4096];int n=snprintf(path,sizeof(path),"%s/keys%u-start%u-mode%u-%s-encoder%d-nsg%u-out%u.f32",output_dir,f->keys,f->start,f->mode,direct?"direct":"staged",myds_metal_get_concurrent(),nsg,output);
    REQUIRE(n>0&&(size_t)n<sizeof(path));FILE *file=fopen(path,"wb");REQUIRE(file);
    REQUIRE(fwrite(myds_metal_buffer_data(f->out[output].view),1,f->out[output].bytes,file)==f->out[output].bytes);REQUIRE(fclose(file)==0);
}
static void constants(uint32_t keys,myds_metal_constant_t c[10]) {
    const myds_metal_constant_t values[]={{400,true,1},{401,true,1},{402,true,0},{403,true,0},{404,true,keys%32!=0},
        {405,true,0},{420,false,512},{421,false,512},{422,false,1},{423,false,32}};memcpy(c,values,sizeof(values));
}
static id<MTLComputePipelineState> pipeline(unsigned nsg,uint32_t keys) {
    myds_metal_constant_t c[10];constants(keys,c);id<MTLComputePipelineState> p=specialized_pipeline_lookup(kernel_name(nsg),c,10);
    if(p)return p;MTLFunctionConstantValues *values=[MTLFunctionConstantValues new];
    for(unsigned i=0;i<10;i++){bool bv=c[i].value!=0;int32_t iv=c[i].value;
        [values setConstantValue:c[i].is_bool?(const void*)&bv:(const void*)&iv type:c[i].is_bool?MTLDataTypeBool:MTLDataTypeInt atIndex:c[i].index];}
    NSError *error=nil;id<MTLFunction> fn=[g_library newFunctionWithName:[NSString stringWithUTF8String:kernel_name(nsg)] constantValues:values error:&error];
    if(!fn)fprintf(stderr,"PSO function error: %s\n",error.localizedDescription.UTF8String);REQUIRE(fn);
    p=[g_device newComputePipelineStateWithFunction:fn error:&error];
    if(!p)fprintf(stderr,"PSO pipeline error: %s\n",error.localizedDescription.UTF8String);REQUIRE(p);return p;
}
static int supported(unsigned nsg) {
    id<MTLComputePipelineState> p=pipeline(nsg,31);size_t dynamic=ATTN_PACKED_SMEM(nsg);
    int ok=32u*nsg<=p.maxTotalThreadsPerThreadgroup && 32u*nsg<=g_device.maxThreadsPerThreadgroup.width &&
        dynamic<=g_device.maxThreadgroupMemoryLength && p.staticThreadgroupMemoryLength<=g_device.maxThreadgroupMemoryLength-dynamic;
    printf("{\"kind\":\"geometry\",\"kernel\":\"%s\",\"physical_nsg\":%u,\"logical_splits\":32,\"threads\":%u,\"dynamic_bytes\":%zu,\"pso_max_threads\":%lu,\"pso_static_bytes\":%lu,\"device_max_threadgroup_bytes\":%lu,\"supported\":%s}\n",
        kernel_name(nsg),nsg,32u*nsg,dynamic,(unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength,(unsigned long)g_device.maxThreadgroupMemoryLength,ok?"true":"false");
    if(!ok)REQUIRE(nsg==16);return ok;
}
static void host_configuration(void) {
    unsetenv("MYDS_ATTN_PACKED_NSG");REQUIRE(get_attn_packed_physical_nsg()==8);
    const char *valid[]={"4","8","16"};const int expected[]={4,8,16};
    for(unsigned i=0;i<3;i++){setenv("MYDS_ATTN_PACKED_NSG",valid[i],1);REQUIRE(get_attn_packed_physical_nsg()==expected[i]);}
    const char *invalid[]={"","0","2","32","04","08","016","4x","8x","16x","+4"," 4","4 ","-4"};
    for(unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);i++){setenv("MYDS_ATTN_PACKED_NSG",invalid[i],1);REQUIRE(!get_attn_packed_physical_nsg());}
    unsetenv("MYDS_ATTN_PACKED_NSG");puts("{\"kind\":\"host_config\",\"exact_valid_values\":3,\"invalid_values_rejected\":14,\"absent_is_original_8\":true}");
}
static void numeric(uint32_t keys,uint32_t start,uint32_t mode,int direct,int use16) {
    struct fixture f;fixture_init(&f,keys,start,mode);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);
    prepare_mask(s,&f);select_mode(0,0);REQUIRE(myds_metal_begin_batch());REQUIRE(call(&f,s,0,direct));REQUIRE(myds_metal_commit_batch());
    export_out(&f,0,0,direct);
    const unsigned ns[]={8,8,4,16};
    for(unsigned i=0;i<4;i++){
        if(ns[i]==16&&!use16)continue;select_mode(ns[i],i==0);prepare_mask(s,&f);
        uint64_t before=hist_count(kernel_name(ns[i]));REQUIRE(myds_metal_begin_batch());REQUIRE(call(&f,s,i+1,direct));REQUIRE(myds_metal_commit_batch());
        REQUIRE(hist_count(kernel_name(ns[i]))==before+1);export_out(&f,i+1,ns[i],direct);compare(&f,i+1);
        myds_metal_constant_t c[10];constants(keys,c);id<MTLComputePipelineState> p=specialized_pipeline_lookup(kernel_name(ns[i]),c,10);REQUIRE(p);
        printf("{\"kind\":\"parity\",\"keys\":%u,\"raw_start\":%u,\"mask_sink_mode\":%u,\"path\":\"%s\",\"explicit_concurrent_mode\":%d,\"kernel\":\"%s\",\"physical_nsg\":%u,\"default_env_absent\":%s,\"actual_kernel_dispatches\":1,\"output_floats\":32768,\"output_bytes\":131072,\"fnv1a64\":\"%016" PRIx64 "\",\"bit_exact\":true,\"finite\":true,\"canaries\":true}\n",
            keys,start,mode,direct?"production_direct":"staged",myds_metal_get_concurrent(),kernel_name(ns[i]),ns[i],i==0?"true":"false",fnv(myds_metal_buffer_data(f.out[i+1].view),f.out[i+1].bytes));
    }
    myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
/* 1 -> 3 -> 3 calls reuse the SAME scratch across command buffers. Within each
 * 3-call buffer, output bindings remain distinct while pad/KV/partial scratch
 * is reused. Queries/keys are immutable inputs so CPU writes never race GPU. */
static void reuse(unsigned nsg) {
    struct fixture f[3];const uint32_t keys[]={1,31,640};
    for(unsigned j=0;j<3;j++)fixture_init(f+j,keys[j],127,0);
    myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);select_mode(0,0);
    for(unsigned j=0;j<3;j++){prepare_mask(s,f+j);REQUIRE(myds_metal_begin_batch());REQUIRE(call(f+j,s,0,1));REQUIRE(myds_metal_commit_batch());}
    memset(myds_metal_buffer_data(s->mask),0,myds_metal_buffer_size(s->mask));select_mode(nsg,0);
    const unsigned counts[]={1,3,3};
    for(unsigned pass=0;pass<3;pass++){
        uint64_t before=hist_count(kernel_name(nsg));REQUIRE(myds_metal_begin_batch());
        for(unsigned j=0;j<counts[pass];j++){unsigned idx=pass==2?2-j:j;REQUIRE(call(f+idx,s,1+pass,1));}
        REQUIRE(myds_metal_commit_batch());REQUIRE(hist_count(kernel_name(nsg))==before+counts[pass]);
        for(unsigned j=0;j<counts[pass];j++){unsigned idx=pass==2?2-j:j;compare(f+idx,1+pass);}
    }
    printf("{\"kind\":\"scratch_reuse\",\"kernel\":\"%s\",\"physical_nsg\":%u,\"calls_per_batch\":[1,3,3],\"keys\":[1,31,640],\"actual_kernel_dispatches\":7,\"same_scratch\":true,\"same_command_buffer_reuse\":true,\"auto_barrier\":%d,\"explicit_concurrent_mode\":%d,\"bit_exact\":true}\n",kernel_name(nsg),nsg,myds_metal_get_auto_barrier(),myds_metal_get_concurrent());
    myds_v41_gpu_scratch_free(s);for(unsigned j=0;j<3;j++)fixture_free(f+j);
}
static void excluded16(void) {
    struct fixture f;fixture_init(&f,31,127,0);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);
    select_mode(16,0);prepare_mask(s,&f);uint64_t before=hist_count(kernel_name(16)),old8=hist_count(kernel_name(8));
    REQUIRE(myds_metal_begin_batch());REQUIRE(!call(&f,s,1,1));REQUIRE(myds_metal_commit_batch());
    REQUIRE(hist_count(kernel_name(16))==before && hist_count(kernel_name(8))==old8);
    const uint8_t *out=myds_metal_buffer_data(f.out[1].view);for(size_t i=0;i<f.out[1].bytes;i++)REQUIRE(out[i]==0xa5);
    puts("{\"kind\":\"hardware_exclusion\",\"physical_nsg\":16,\"actual_candidate_geometry_rejected\":true,\"fallback_packed8_dispatches\":0,\"output_untouched\":true}");
    myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
static struct flash_vec_args vector_args(uint32_t keys) {
    uint64_t kv=(uint64_t)keys*1024,mask=(uint64_t)keys*2;
    return (struct flash_vec_args){.ne01=1,.ne02=64,.ne03=1,.nb01=64u*2048,.nb02=2048,.nb03=64u*2048,
        .ne11=keys,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=1024,.nb12=kv,.nb13=kv,
        .ns20=512,.nb21=1024,.nb22=kv,.nb23=kv,.ne31=1,.ne32=1,.ne33=1,.nb31=mask,.nb32=mask,.nb33=mask,
        .ne1=64,.ne2=1,.ne3=1,.scale=1.0f/sqrtf(512)};
}
static void rejection(unsigned nsg) {
    struct fixture f;fixture_init(&f,31,127,0);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);
    const char *invalid[]={"","0","04","8x","16 ","32"};
    for(unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);i++){
        setenv("MYDS_ATTN_PACKED_NSG",invalid[i],1);uint64_t before=myds_metal_dispatch_count();
        uint64_t out=fnv(myds_metal_buffer_data(f.out[1].view),f.out[1].bytes);
        REQUIRE(!call(&f,s,1,0)&&!call(&f,s,1,1));REQUIRE(myds_metal_dispatch_count()==before);
        REQUIRE(fnv(myds_metal_buffer_data(f.out[1].view),f.out[1].bytes)==out);
    }
    if(nsg!=8){
        select_mode(nsg,0);setenv("MYDS_ATTN_PACKED_REDUCE","0",1);
        uint64_t before=myds_metal_dispatch_count();uint64_t out=fnv(myds_metal_buffer_data(f.out[1].view),f.out[1].bytes);
        REQUIRE(!call(&f,s,1,0)&&!call(&f,s,1,1));REQUIRE(myds_metal_dispatch_count()==before);
        REQUIRE(fnv(myds_metal_buffer_data(f.out[1].view),f.out[1].bytes)==out);
    }
    struct flash_vec_args a=vector_args(f.keys);myds_metal_constant_t c[10];constants(f.keys,c);
    const myds_metal_buffer_t *b[]={f.q,s->kv_f16,s->kv_f16,s->mask,f.sinks,s->pad,f.out[1].view};
    id<MTLComputePipelineState> p=pipeline(nsg,f.keys);uint64_t before=myds_metal_dispatch_count();
    REQUIRE(!specialized(kernel_name(nsg),&a,sizeof(a),b,7,64,1,1,(uint32_t)p.maxTotalThreadsPerThreadgroup+1u,1,1,ATTN_PACKED_SMEM(nsg),c,10));
    REQUIRE(!specialized(kernel_name(nsg),&a,sizeof(a),b,7,64,1,1,32u*nsg,1,1,g_device.maxThreadgroupMemoryLength+16u,c,10));
    REQUIRE(myds_metal_dispatch_count()==before);
    const uint8_t *out=myds_metal_buffer_data(f.out[1].view);for(size_t i=0;i<f.out[1].bytes;i++)REQUIRE(out[i]==0xa5);
    printf("{\"kind\":\"rejection\",\"kernel\":\"%s\",\"physical_nsg\":%u,\"invalid_config_before_staging\":true,\"actual_pso_over_threads_rejected\":true,\"device_over_memory_rejected\":true,\"dispatches\":0,\"output_untouched\":true}\n",kernel_name(nsg),nsg);
    myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
/* GPU timing distinguishes kernel-only cost (staging completed beforehand)
 * from the public production direct entry point (copy/gather/pad included).
 * Repetitions serialize shared scratch/output through a full resource barrier.
 * JSON emits one warmup + samples formal per arm, never a host enqueue timer. */
static void timing(uint32_t keys,unsigned nsg,int kernel_only,unsigned candidate,const char *arm,unsigned window) {
    struct fixture f;fixture_init(&f,keys,127,0);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);
    prepare_mask(s,&f);select_mode(0,0);REQUIRE(myds_metal_begin_batch());REQUIRE(call(&f,s,0,1));REQUIRE(myds_metal_commit_batch());
    select_mode(nsg,0);REQUIRE(myds_metal_begin_batch());REQUIRE(call(&f,s,1,1));REQUIRE(myds_metal_commit_batch());compare(&f,1);
    struct flash_vec_args a=vector_args(keys);myds_metal_constant_t c[10];constants(keys,c);
    const myds_metal_buffer_t *b[]={f.q,s->kv_f16,s->kv_f16,s->mask,f.sinks,s->pad,f.out[1].view};
    for(unsigned sample=0;sample<=samples;sample++){
        uint64_t dispatch_before=myds_metal_dispatch_count();
        REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned r=0;r<repetitions;r++){
            if(kernel_only)REQUIRE(specialized(kernel_name(nsg),&a,sizeof(a),b,7,64,1,1,32u*nsg,1,1,ATTN_PACKED_SMEM(nsg),c,10));
            else REQUIRE(call(&f,s,1,1));myds_metal_barrier();
        }
        REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted && isfinite(cb.GPUStartTime) && cb.GPUEndTime>cb.GPUStartTime);
        const uint32_t per_call=kernel_only?1u:2u+(f.selected?1u:0u)+(keys%32u?1u:0u);
        REQUIRE(myds_metal_dispatch_count()-dispatch_before==(uint64_t)repetitions*per_call);compare(&f,1);
        double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/repetitions;
        printf("{\"kind\":\"gpu_timing\",\"window\":%u,\"candidate_physical_nsg\":%u,\"arm\":\"%s\",\"scope\":\"%s\",\"keys\":%u,\"physical_nsg\":%u,\"kernel\":\"%s\",\"warmup\":%s,\"sample\":%u,\"repetitions\":%u,\"gpu_us_per_call\":%.9f,\"gpu_start_s\":%.9f,\"gpu_end_s\":%.9f,\"dispatch_hist_disabled\":true,\"auto_barrier\":%d,\"explicit_concurrent_mode\":%d,\"bit_exact_full_output\":true}\n",
            window,candidate,arm,kernel_only?"packed_kernel_only":"production_direct_including_staging",keys,nsg,kernel_name(nsg),sample==0?"true":"false",sample,repetitions,us,cb.GPUStartTime,cb.GPUEndTime,myds_metal_get_auto_barrier(),myds_metal_get_concurrent());
    }
    myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
static uint32_t parse_count(const char *v) { char *end=NULL;errno=0;unsigned long n=strtoul(v,&end,10);REQUIRE(!errno&&end!=v&&!*end&&n>=1&&n<=10000);return (uint32_t)n; }
int main(int argc,char **argv) {
    int host_only=0,timing_enabled=1;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--host-only"))host_only=1;
        else if(!strcmp(argv[i],"--no-timing"))timing_enabled=0;
        else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)output_dir=argv[++i];
        else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader_dir=argv[++i];
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=parse_count(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)repetitions=parse_count(argv[++i]);
        else{fprintf(stderr,"usage: %s [--host-only] [--no-timing] [--output-dir new-dir] [--shader-dir dir] [--samples N] [--repetitions N]\n",argv[0]);return 2;}
    }
    host_configuration();if(host_only)return 0;
    if(output_dir){
        struct stat st;errno=0;int exists=lstat(output_dir,&st);
        if(exists==0){fprintf(stderr,"output directory already exists; refusing overwrite: %s\n",output_dir);return 2;}
        REQUIRE(errno==ENOENT);REQUIRE(mkdir(output_dir,0700)==0);
    }
    setenv("MYDS_ATTN_VEC_HEAD_FIRST","0",1);setenv("MYDS_DEBUG_DISPATCH","1",1);
    /* Global env forces concurrent even when set_concurrent(0). Pin it to 0
     * BEFORE either runtime cache reads it, then explicitly test both modes. */
    setenv("MYDS_METAL_CONCURRENT","0",1);
    g_hist_enabled=1;REQUIRE(myds_metal_init(shader_dir));myds_metal_set_concurrent(0);
    REQUIRE([[g_queue commandBuffer] respondsToSelector:@selector(computeCommandEncoderWithDispatchType:)]);
    printf("{\"kind\":\"identity\",\"device\":\"%s\",\"compiled\":\"%s %s\",\"shape\":\"Q64x512,K=V512,F16,32logical-splits\",\"fixture_not_fullmodel\":true}\n",g_device.name.UTF8String,__DATE__,__TIME__);
    REQUIRE(supported(4)&&supported(8));int use16=supported(16);if(!use16)excluded16();
    const uint32_t keys[]={1,31,32,128,512,640};
    const unsigned ns[]={4,8,16};
    for(unsigned concurrent=0;concurrent<2;concurrent++){
        myds_metal_set_concurrent((int)concurrent);myds_metal_set_auto_barrier(1);
        for(unsigned k=0;k<6;k++)for(unsigned start=0;start<2;start++)for(unsigned mode=0;mode<5;mode++)for(unsigned direct=0;direct<2;direct++)numeric(keys[k],start?127:0,mode,direct,use16);
        for(unsigned i=0;i<3;i++)if(ns[i]!=16||use16){reuse(ns[i]);rejection(ns[i]);}
        myds_metal_set_auto_barrier(0);for(unsigned i=0;i<3;i++)if(ns[i]!=16||use16)reuse(ns[i]);
    }
    myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);
    if(timing_enabled){
        g_hist_enabled=0;unsigned window=0;
        const unsigned candidates[]={4,16};
        for(unsigned k=0;k<6;k++)for(unsigned scope=0;scope<2;scope++)for(unsigned i=0;i<2;i++)if(candidates[i]!=16||use16){
            ++window;timing(keys[k],8,scope,candidates[i],"A1",window);
            timing(keys[k],candidates[i],scope,candidates[i],"B",window);
            timing(keys[k],8,scope,candidates[i],"A2",window);
        }
    }
    myds_metal_cleanup();printf("{\"kind\":\"result\",\"passed\":true,\"completed_full_output_parity_pairs\":%u,\"nsg16_supported\":%s,\"bit_exact_full_outputs\":true,\"positions_not_fullmodel\":true}\n",use16?960u:720u,use16?"true":"false");return 0;
}
