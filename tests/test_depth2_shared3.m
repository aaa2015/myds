/* Real Metal, model-free shared3 component gate. Including the runtime
 * exposes the actual PSO cache for a genuinely missing-function test; GPU
 * arithmetic and dispatches are never mocked. Link without myds_metal.o.
 * NR2/NR4 run in separate processes because scalar QKV NR is process-once. */
#include "myds_metal.m"
#include <inttypes.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
struct guarded { myds_metal_buffer_t *owner, *view; size_t bytes, offset; };
struct fixture { uint32_t rows, width; ggml_type type; struct guarded weight, in[3], ref[3], separate[3], out[3]; };
static unsigned nr, parity_cases, reject_cases;
static const char *shader_dir = "bin";
static struct guarded allocate(size_t bytes, size_t offset) {
    struct guarded b = {.bytes=bytes,.offset=offset};
    b.owner=myds_metal_buffer_alloc(bytes+offset+16u); REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+offset+16u);
    b.view=myds_metal_buffer_sub_view(b.owner,offset,bytes); REQUIRE(b.view); return b;
}
static void canary(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<b.offset;i++) REQUIRE(p[i]==0xa5);
    for(size_t i=0;i<16;i++) REQUIRE(p[b.offset+b.bytes+i]==0xa5);
}
static void release(struct guarded b) { canary(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner); }
static void reset(struct fixture *f) { for(unsigned t=0;t<3;t++) memset(myds_metal_buffer_data(f->out[t].view),0xa5,f->out[t].bytes); }
static void untouched(struct fixture *f) {
    for(unsigned t=0;t<3;t++) { const uint8_t *p=myds_metal_buffer_data(f->out[t].view);
        for(size_t i=0;i<f->out[t].bytes;i++) REQUIRE(p[i]==0xa5); canary(f->out[t]); }
}
static void init_fixture(struct fixture *f,uint32_t rows,uint32_t width,ggml_type type,unsigned pattern,unsigned seed) {
    memset(f,0,sizeof(*f));f->rows=rows;f->width=width;f->type=type;
    const unsigned block=type==GGML_TYPE_Q4_K?256u:32u,bytes=type==GGML_TYPE_Q4_K?144u:34u;
    f->weight=allocate((size_t)rows*(width/block)*bytes,18);
    uint8_t *w=myds_metal_buffer_data(f->weight.view);
    for(size_t b=0;b<f->weight.bytes/bytes;b++) {
        uint16_t scale=(uint16_t)(0x1400u+(b%8u)*0x180u);memcpy(w+bytes*b,&scale,2);
        if(type==GGML_TYPE_Q4_K) {
            uint16_t minimum=(uint16_t)(0x1000u+(b%5u)*0x190u);memcpy(w+bytes*b+2,&minimum,2);
            /* Arbitrary packed scales exercise all six-bit scale/min bits. */
            for(unsigned i=4;i<144;i++) w[bytes*b+i]=(uint8_t)(b*37u+i*29u+seed);
        } else for(unsigned i=0;i<32;i++) w[bytes*b+2+i]=(uint8_t)(int8_t)((int)((b*37u+i*29u+seed)%255u)-127);
    }
    for(unsigned t=0;t<3;t++) {
        f->in[t]=allocate((size_t)width*4,20+4*t);
        f->ref[t]=allocate((size_t)rows*4,16+16*t);f->separate[t]=allocate((size_t)rows*4,64+16*t);
        f->out[t]=allocate((size_t)rows*4,32+16*t);
        float *x=myds_metal_buffer_data(f->in[t].view);
        for(uint32_t i=0;i<width;i++) {
            uint32_t z=i*747796405u+(seed+113u*t)*2891336453u;z=((z>>((z>>28)+4))^z)*277803737u;z=(z>>22)^z;
            x[i]=pattern==2?0.f:((int)(z%1023u)-511)/(pattern==3?128.f:257.f);
            if(pattern==4) x[i]=(i&1?-1.f:1.f)*(t+1.f)/256.f;
        }
        if(pattern==1&&t) { const float *x0=myds_metal_buffer_data(f->in[0].view);
            for(uint32_t i=0;i<width;i++) x[i]=t==1?-x0[i]:x0[i]*0.5f; }
    }
}
static void free_fixture(struct fixture *f) {
    release(f->weight);for(unsigned t=0;t<3;t++){release(f->in[t]);release(f->ref[t]);release(f->separate[t]);release(f->out[t]);}
}
static void equal(const float *a,const float *b,uint32_t n,const char *role,unsigned token,struct fixture *f) {
    if(memcmp(a,b,(size_t)n*4)) for(uint32_t i=0;i<n;i++) if(memcmp(a+i,b+i,4)) {
        uint32_t x,y;memcpy(&x,a+i,4);memcpy(&y,b+i,4);
        fprintf(stderr,"PARITY role=%s token=%u index=%u ref=%08x candidate=%08x type=%d rows=%u width=%u\n",role,token,i,x,y,f->type,f->rows,f->width);break;
    }
    REQUIRE(!memcmp(a,b,(size_t)n*4));
    for(uint32_t i=0;i<n;i++){uint32_t bits;memcpy(&bits,b+i,4);REQUIRE(isfinite(a[i])&&isfinite(b[i]));REQUIRE((bits&0xffffu)==0);}
}
static void compare(struct fixture *f) {
    for(unsigned t=0;t<3;t++) {
        equal(myds_metal_buffer_data(f->ref[t].view),myds_metal_buffer_data(f->out[t].view),f->rows,"shared3",t,f);
        if(f->type==GGML_TYPE_Q4_K) equal(myds_metal_buffer_data(f->ref[t].view),myds_metal_buffer_data(f->separate[t].view),f->rows,"separate_gemv_bf16",t,f);
        canary(f->ref[t]);canary(f->separate[t]);canary(f->in[t]);canary(f->out[t]);
    }canary(f->weight);
}
static void reference(struct fixture *f) {
    for(unsigned t=0;t<3;t++) {
        /* Separate command buffers give each reference row the ordinary
         * scalar dependency and reduction contract. */
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_metal_v41_gemv_bf16(f->ref[t].view,f->weight.view,f->in[t].view,f->rows,f->width,f->type));
        REQUIRE(myds_metal_commit_batch());
        if(f->type==GGML_TYPE_Q4_K) {
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_metal_gemv(f->separate[t].view,f->weight.view,f->in[t].view,f->rows,f->width,f->type));
            myds_metal_barrier();REQUIRE(myds_metal_v41_bf16(f->separate[t].view,f->rows));
            REQUIRE(myds_metal_commit_batch());
        }
    }
}
static int candidate(struct fixture *f) {
    myds_metal_buffer_t *out[]={f->out[0].view,f->out[1].view,f->out[2].view};
    myds_metal_buffer_t *input[]={f->in[0].view,f->in[1].view,f->in[2].view};
    return myds_metal_v41_gemv_bf16_shared3_type(out,f->weight.view,input,f->rows,f->width,f->type);
}
static void numeric(uint32_t rows,uint32_t width,ggml_type type,unsigned pattern) {
    struct fixture f;init_fixture(&f,rows,width,type,pattern,31+parity_cases);reference(&f);
    myds_metal_shared3_counters_t a,b;myds_metal_shared3_counters(&a,false);uint64_t dispatch=myds_metal_dispatch_count();
    REQUIRE(myds_metal_begin_batch());for(unsigned r=0;r<3;r++) REQUIRE(candidate(&f));REQUIRE(myds_metal_commit_batch());
    compare(&f);myds_metal_shared3_counters(&b,false);
    REQUIRE(b.requests-a.requests==3&&b.encoded-a.encoded==3&&b.failures==a.failures);
    REQUIRE(b.q4_encoded-a.q4_encoded==(type==GGML_TYPE_Q4_K?3u:0u));
    REQUIRE(b.q8_encoded-a.q8_encoded==(type==GGML_TYPE_Q8_0?3u:0u));REQUIRE(myds_metal_dispatch_count()-dispatch==3);
    printf("{\"kind\":\"parity\",\"case\":%u,\"type\":%d,\"rows\":%u,\"width\":%u,\"pattern\":%u,\"nr0\":%u,\"nsg\":%u,\"concurrent\":%d,\"auto_barrier\":%d,\"native_shared_dispatches\":3,\"full_output_floats\":%u,\"separate_q4_reference\":%s,\"bit_exact\":true,\"finite\":true,\"canaries\":true}\n",parity_cases++,type,rows,width,pattern,type==GGML_TYPE_Q4_K?2:nr,type==GGML_TYPE_Q4_K?4:(rows>65536?8:4),myds_metal_get_concurrent(),myds_metal_get_auto_barrier(),3*rows,type==GGML_TYPE_Q4_K?"true":"false");free_fixture(&f);
}
static void reject_call(struct fixture *f,myds_metal_buffer_t *out[3],const myds_metal_buffer_t *weight,myds_metal_buffer_t *in[3],uint32_t rows,uint32_t width,ggml_type type) {
    reset(f);myds_metal_shared3_counters_t a,b;myds_metal_shared3_counters(&a,false);uint64_t dispatch=myds_metal_dispatch_count();
    REQUIRE(!myds_metal_v41_gemv_bf16_shared3_type(out,weight,in,rows,width,type));myds_metal_shared3_counters(&b,false);
    REQUIRE(b.requests==a.requests+1&&b.failures==a.failures+1&&b.encoded==a.encoded&&b.q4_encoded==a.q4_encoded&&b.q8_encoded==a.q8_encoded);
    REQUIRE(dispatch==myds_metal_dispatch_count());untouched(f);reject_cases++;
}
static void rejections(ggml_type type) {
    struct fixture f;init_fixture(&f,17,256,type,0,77);reference(&f);
    myds_metal_buffer_t *out[]={f.out[0].view,f.out[1].view,f.out[2].view},*in[]={f.in[0].view,f.in[1].view,f.in[2].view};
    for(unsigned t=0;t<3;t++) {
        myds_metal_buffer_t bad=*out[t];bad.size-=4;myds_metal_buffer_t *o[]={out[0],out[1],out[2]};o[t]=&bad;reject_call(&f,o,f.weight.view,in,17,256,type);
        bad=*in[t];bad.size-=4;myds_metal_buffer_t *x[]={in[0],in[1],in[2]};x[t]=&bad;reject_call(&f,out,f.weight.view,x,17,256,type);
        bad=*out[t];bad.offset+=4;bad.ptr=(char*)bad.ptr+4;reject_call(&f,o,f.weight.view,in,17,256,type);
        bad=*in[t];bad.offset++;reject_call(&f,out,f.weight.view,x,17,256,type);
        bad=*out[t];bad.offset=bad.mtl_buf.length;reject_call(&f,o,f.weight.view,in,17,256,type);
        o[t]=NULL;reject_call(&f,o,f.weight.view,in,17,256,type);
    }
    myds_metal_buffer_t bad=*f.weight.view;bad.size-=2;reject_call(&f,out,&bad,in,17,256,type);
    bad=*f.weight.view;bad.offset++;reject_call(&f,out,&bad,in,17,256,type);
    reject_call(&f,out,f.weight.view,in,0,256,type);reject_call(&f,out,f.weight.view,in,17,255,type);
    reject_call(&f,out,f.weight.view,in,UINT32_MAX,256,type);reject_call(&f,out,f.weight.view,in,17,UINT32_MAX,type);
    reject_call(&f,NULL,f.weight.view,in,17,256,type);reject_call(&f,out,NULL,in,17,256,type);reject_call(&f,out,f.weight.view,NULL,17,256,type);
    reject_call(&f,out,f.weight.view,in,17,256,GGML_TYPE_F16);reject_call(&f,out,f.weight.view,in,17,256,GGML_TYPE_Q2_K);
    for(unsigned t=0;t<3;t++) for(unsigned s=t+1;s<3;s++) {myds_metal_buffer_t *o[]={out[0],out[1],out[2]};o[t]=out[s];reject_call(&f,o,f.weight.view,in,17,256,type);}
    for(unsigned t=0;t<3;t++) for(unsigned s=0;s<3;s++) {
        myds_metal_buffer_t alias=*in[s];alias.offset=16;alias.ptr=(char*)myds_metal_buffer_data(f.in[s].owner)+16;alias.size=68;
        myds_metal_buffer_t *o[]={out[0],out[1],out[2]};o[t]=&alias;reject_call(&f,o,f.weight.view,in,17,256,type);
    }
    bad=*f.weight.view;bad.offset=16;bad.ptr=(char*)myds_metal_buffer_data(f.weight.owner)+16;bad.size=68;
    myds_metal_buffer_t *o[]={out[0],out[1],&bad};reject_call(&f,o,f.weight.view,in,17,256,type);
    /* Independently registered native resources can alias the same physical
     * host range. A resource-identity-only guard would incorrectly accept. */
    void *mapped=NULL;REQUIRE(!posix_memalign(&mapped,16384,16384));memset(mapped,0x5a,16384);
    myds_metal_buffer_t *m0=myds_metal_buffer_from_ptr(mapped,16384),*m1=myds_metal_buffer_from_ptr(mapped,16384);
    REQUIRE(m0&&m1&&m0->mtl_buf!=m1->mtl_buf);
    myds_metal_buffer_t *mx=myds_metal_buffer_sub_view(m0,0,1024),*mo=myds_metal_buffer_sub_view(m1,16,68);REQUIRE(mx&&mo);
    myds_metal_buffer_t *aliasin[]={mx,in[1],in[2]},*aliasout[]={out[0],out[1],mo};
    reject_call(&f,aliasout,f.weight.view,aliasin,17,256,type);
    for(size_t i=0;i<16384;i++)REQUIRE(((uint8_t*)mapped)[i]==0x5a);
    myds_metal_buffer_free(mx);myds_metal_buffer_free(mo);myds_metal_buffer_free(m0);myds_metal_buffer_free(m1);free(mapped);
    /* Genuine library without the new symbol; cached scalar PSOs stay real. */
    unsigned nsg=4,row=type==GGML_TYPE_Q4_K?2:nr;
    NSString *key=[NSString stringWithFormat:@"shared3_%d_%u_%u",type,nsg,row];
    id<MTLComputePipelineState> saved=shared3_pipeline(nsg,row,type);REQUIRE(saved);
    id<MTLLibrary> original=g_library;[g_native_pipelines removeObjectForKey:key];
    g_library=[g_device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void unrelated(device float *x [[buffer(0)]],uint i [[thread_position_in_grid]]) {x[i]=0;}\n" options:nil error:nil];REQUIRE(g_library);
    reject_call(&f,out,f.weight.view,in,17,256,type);g_library=original;g_native_pipelines[key]=saved;
    /* Read-only input aliasing is legal and still has three distinct outputs. */
    in[1]=in[0];in[2]=in[0];REQUIRE(myds_metal_begin_batch());REQUIRE(myds_metal_v41_gemv_bf16_shared3_type(out,f.weight.view,in,17,256,type));REQUIRE(myds_metal_commit_batch());
    REQUIRE(!memcmp(myds_metal_buffer_data(out[0]),myds_metal_buffer_data(f.ref[0].view),68));
    REQUIRE(!memcmp(myds_metal_buffer_data(out[0]),myds_metal_buffer_data(out[1]),68));REQUIRE(!memcmp(myds_metal_buffer_data(out[0]),myds_metal_buffer_data(out[2]),68));
    free_fixture(&f);
}
static void ties(ggml_type type) {
    struct fixture f;init_fixture(&f,3,type==GGML_TYPE_Q4_K?256u:32u,type,0,11);
    const size_t stride=type==GGML_TYPE_Q4_K?144u:34u;
    uint8_t *w=myds_metal_buffer_data(f.weight.view);memset(w,0,f.weight.bytes);
    for(unsigned row=0;row<3;row++) {
        const uint16_t one=0x3c00u;memcpy(w+row*stride,&one,2);
        if(type==GGML_TYPE_Q4_K){w[row*stride+4]=1;w[row*stride+16]=1;}
        else w[row*stride+2]=1;
    }
    const float x[]={1.00390625f,1.01171875f,-1.01171875f};
    const uint32_t expected[]={0x3f800000u,0x3f820000u,0xbf820000u};
    for(unsigned t=0;t<3;t++) {float *in=myds_metal_buffer_data(f.in[t].view);memset(in,0,f.in[t].bytes);in[0]=x[t];}
    reference(&f);REQUIRE(myds_metal_begin_batch());REQUIRE(candidate(&f));REQUIRE(myds_metal_commit_batch());compare(&f);
    for(unsigned t=0;t<3;t++)for(unsigned row=0;row<3;row++){uint32_t bits;memcpy(&bits,(char*)myds_metal_buffer_data(f.out[t].view)+row*4,4);REQUIRE(bits==expected[t]);}
    printf("{\"kind\":\"bf16_ties\",\"type\":%d,\"positive_even_lower\":true,\"positive_odd_upper\":true,\"negative_odd_upper\":true,\"bit_exact\":true}\n",type);free_fixture(&f);
}
static void commit_boundary(ggml_type type) {
    struct fixture f;init_fixture(&f,17,1024,type,0,911);reference(&f);
    myds_metal_shared3_counters_t a,b;myds_metal_shared3_counters(&a,false);
    REQUIRE(myds_metal_begin_batch());REQUIRE(candidate(&f));
    myds_metal_inject_commit_failure(1);REQUIRE(!myds_metal_commit_batch());
    REQUIRE(myds_metal_get_commit_call_count()==1);myds_metal_inject_commit_failure(0);
    REQUIRE(myds_metal_synchronize());myds_metal_shared3_counters(&b,false);
    /* An encoded dispatch is not a successful surrounding transaction. The
     * helper cannot see a later caller commit failure; caller must reject it. */
    REQUIRE(b.requests==a.requests+1&&b.encoded==a.encoded+1&&b.failures==a.failures);
    reset(&f);REQUIRE(myds_metal_begin_batch());REQUIRE(candidate(&f));REQUIRE(myds_metal_commit_batch());compare(&f);
    printf("{\"kind\":\"commit_boundary\",\"type\":%d,\"caller_commit_failure_observed\":true,\"encoded_does_not_imply_commit\":true,\"recovery_bit_exact\":true}\n",type);free_fixture(&f);
}
static void dependency_chain(ggml_type type) {
    struct fixture f;init_fixture(&f,256,256,type,0,1947);reference(&f);
    struct guarded ref[3],out[3];myds_metal_buffer_t *second_out[3],*second_in[3];
    for(unsigned t=0;t<3;t++) {
        ref[t]=allocate(1024,16+16*t);out[t]=allocate(1024,32+16*t);
        REQUIRE(myds_metal_begin_batch());REQUIRE(myds_metal_v41_gemv_bf16(ref[t].view,f.weight.view,f.ref[t].view,256,256,type));REQUIRE(myds_metal_commit_batch());
        second_out[t]=out[t].view;second_in[t]=f.out[t].view;
    }
    const uint64_t dispatch=myds_metal_dispatch_count();
    REQUIRE(myds_metal_begin_batch());REQUIRE(candidate(&f));
    REQUIRE(myds_metal_v41_gemv_bf16_shared3_type(second_out,f.weight.view,second_in,256,256,type));
    /* Overwrite the first outputs after the second reads them, checking WAR
     * ordering as well as producer-to-consumer RAW ordering. */
    REQUIRE(candidate(&f));REQUIRE(myds_metal_commit_batch());REQUIRE(myds_metal_dispatch_count()-dispatch==3);compare(&f);
    for(unsigned t=0;t<3;t++) {equal(myds_metal_buffer_data(ref[t].view),myds_metal_buffer_data(out[t].view),256,"dependency_chain",t,&f);release(ref[t]);release(out[t]);}
    printf("{\"kind\":\"dependency_chain\",\"type\":%d,\"concurrent\":%d,\"auto_barrier\":%d,\"raw_war_same_cb\":true,\"native_shared_dispatches\":3,\"bit_exact\":true}\n",type,myds_metal_get_concurrent(),myds_metal_get_auto_barrier());free_fixture(&f);
}
static int child(const char *program,unsigned row) {
    @autoreleasepool { NSTask *task=[NSTask new];task.executableURL=[NSURL fileURLWithPath:[NSString stringWithUTF8String:program]];
        task.arguments=@[@"--nr0",[NSString stringWithFormat:@"%u",row],@"--shader-dir",[NSString stringWithUTF8String:shader_dir]];
        fflush(stdout);fflush(stderr);REQUIRE([task launchAndReturnError:nil]);[task waitUntilExit];return task.terminationStatus; }
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader_dir=argv[++i];
        else if(!strcmp(argv[i],"--nr0")&&i+1<argc){nr=(unsigned)strtoul(argv[++i],NULL,10);REQUIRE(nr==2||nr==4);}
        else {fprintf(stderr,"usage: %s [--shader-dir DIR] [--nr0 2|4]\n",argv[0]);return 2;}
    }
    if(!nr){REQUIRE(!child(argv[0],2));REQUIRE(!child(argv[0],4));puts("{\"kind\":\"suite_complete\",\"nr_processes\":2,\"pass\":true}");return 0;}
    char text[8];snprintf(text,sizeof(text),"%u",nr);setenv("MYDS_QKV_NR0",text,1);setenv("MYDS_METAL_CONCURRENT","0",1);
    REQUIRE(myds_metal_init(shader_dir));myds_metal_shared3_counters(NULL,true);
    printf("{\"kind\":\"identity\",\"device\":\"%s\",\"nr0\":%u,\"fixture_not_fullmodel\":true}\n",g_device.name.UTF8String,nr);
    const uint32_t q8[][2]={{1,32},{3,128},{5,256},{17,1024},{5119,8192},{5120,8192},{5121,8192},{32768,1280},{65536,32},{65537,32},{1280,5120},{512,5120},{5120,2304}};
    const uint32_t q4[][2]={{1,256},{3,256},{5,768},{17,1024},{5119,8192},{5120,8192},{5121,8192},{32768,1280},{32769,256},{1280,5120},{512,5120},{5120,2304}};
    for(unsigned concurrent=0;concurrent<2;concurrent++)for(unsigned automatic=0;automatic<2;automatic++) {
        myds_metal_set_concurrent((int)concurrent);myds_metal_set_auto_barrier((int)automatic);
        for(unsigned i=0;i<sizeof(q8)/sizeof(*q8);i++)numeric(q8[i][0],q8[i][1],GGML_TYPE_Q8_0,i%5);
        for(unsigned i=0;i<sizeof(q4)/sizeof(*q4);i++)numeric(q4[i][0],q4[i][1],GGML_TYPE_Q4_K,i%5);
        for(unsigned p=0;p<5;p++){numeric(17,1024,GGML_TYPE_Q8_0,p);numeric(17,1024,GGML_TYPE_Q4_K,p);}
        dependency_chain(GGML_TYPE_Q8_0);dependency_chain(GGML_TYPE_Q4_K);
    }
    ties(GGML_TYPE_Q8_0);ties(GGML_TYPE_Q4_K);
    rejections(GGML_TYPE_Q8_0);rejections(GGML_TYPE_Q4_K);
    commit_boundary(GGML_TYPE_Q8_0);commit_boundary(GGML_TYPE_Q4_K);
    myds_metal_shared3_counters_t count;myds_metal_shared3_counters(&count,false);
    printf("{\"kind\":\"complete\",\"nr0\":%u,\"parity_cases\":%u,\"rejection_cases\":%u,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"q8_encoded\":%" PRIu64 ",\"q4_encoded\":%" PRIu64 ",\"failures_expected\":%" PRIu64 ",\"pass\":true}\n",nr,parity_cases,reject_cases,count.requests,count.encoded,count.q8_encoded,count.q4_encoded,count.failures);
    myds_metal_cleanup();return 0;
}
