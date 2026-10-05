/* Actual-shape QKV component baseline. This standalone TU includes the runtime
 * so it can retain the ORIGINAL command buffer for GPUStartTime/GPUEndTime;
 * no observer encoder splits or extra submissions are introduced.
 * Compile: clang -O3 -Wall -Wextra -Iinclude -fobjc-arc tools/bench_qkv.m
 *   src/myds_v41_gpu.c src/myds_gguf.c src/myds_log.c
 *   -framework Foundation -framework Metal -lm -lpthread -o bin/bench_qkv
 * Root schedules GPU execution. --model reads ONLY selected 54MB Q8 tensors
 * with pread; no full-model mapping/residency or server/cache access.
 */
#include "../src/myds_metal.m"
#include "myds_v41_gpu.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <inttypes.h>

static void require(int ok,const char *why) {
    if(!ok){fprintf(stderr,"bench_qkv: %s\n",why);exit(1);}
}
static double wall_now(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}
struct fixture {
    myds_metal_buffer_t *w[3],*x,*qr,*qr_norm,*q,*kv,*window,*norm[2];
    uint32_t rows[3],width[3],pos;
    bool compressed;
};
static void fill_f32(myds_metal_buffer_t *b,unsigned seed) {
    float *p=myds_metal_buffer_data(b);
    for(size_t i=0;i<myds_metal_buffer_size(b)/4;i++)
        p[i]=((int)((i*19u+seed)%257u)-128)/512.0f;
}
static uint64_t checksum(myds_metal_buffer_t *b) {
    uint32_t *p=myds_metal_buffer_data(b);uint64_t h=1469598103934665603ULL;
    for(size_t i=0;i<myds_metal_buffer_size(b)/4;i++)h=(h^p[i])*1099511628211ULL;
    return h;
}
static void load_tensor(myds_metal_buffer_t *b,NSDictionary *tensor,int fd,bool q8) {
    size_t bytes=myds_metal_buffer_size(b);uint8_t *p=myds_metal_buffer_data(b);
    if(fd>=0) {
        require([[tensor objectForKey:@"bytes"] unsignedLongLongValue]==bytes,"tensor byte size mismatch");
        uint64_t off=[[tensor objectForKey:@"absolute_offset"] unsignedLongLongValue];
        struct stat st;require(!fstat(fd,&st)&&off<=(uint64_t)st.st_size&&bytes<=(uint64_t)st.st_size-off,"tensor file bounds");
        for(size_t done=0;done<bytes;) {
            ssize_t n=pread(fd,p+done,bytes-done,(off_t)(off+done));
            if(n<0&&errno==EINTR)continue;
            require(n>0,"tensor pread failed");done+=(size_t)n;
        }
    } else if(q8) {
        for(size_t i=0;i<bytes/34;i++) {
            uint16_t d=0x2400u;memcpy(p+i*34,&d,2);
            for(size_t j=0;j<32;j++)p[i*34+2+j]=(uint8_t)(int8_t)((int)((i*13+j*29)%255)-127);
        }
    } else {
        float *f=(float *)p;for(size_t i=0;i<bytes/4;i++)f[i]=1.0f+((int)(i%17)-8)/256.0f;
    }
}
static NSDictionary *lookup(NSArray *tensors,NSString *name) {
    for(NSDictionary *t in tensors)if([[t objectForKey:@"name"] isEqual:name])return t;
    return nil;
}
static void fixture_create(struct fixture *f,const char *layout,const char *model,unsigned layer) {
    memset(f,0,sizeof(*f));f->rows[0]=1280;f->rows[1]=32768;f->rows[2]=512;
    f->width[0]=5120;f->width[1]=1280;f->width[2]=5120;f->pos=127;
    NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:layout]];
    require(data!=nil,"cannot read tensor-layout JSON");
    NSDictionary *json=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    NSArray *tensors=[json objectForKey:@"tensors"];require(tensors!=nil,"bad tensor-layout JSON");
    int fd=model?open(model,O_RDONLY):-1;require(!model||fd>=0,"cannot open model");
    const char *suffix[]={"attn_q_a.weight","attn_q_b.weight","attn_kv.weight"};
    for(unsigned i=0;i<3;i++) {
        NSString *name=[NSString stringWithFormat:@"blk.%u.%s",layer,suffix[i]];
        NSDictionary *t=lookup(tensors,name);NSArray *dims=[t objectForKey:@"dims"];
        require(t&&[[t objectForKey:@"type_name"] isEqual:@"Q8_0"]&&[dims count]==2&&
            [dims[0] unsignedIntValue]==f->width[i]&&[dims[1] unsignedIntValue]==f->rows[i],"actual main-layer QKV type/shape mismatch");
        size_t bytes=(size_t)f->rows[i]*f->width[i]/32*34;
        f->w[i]=myds_metal_buffer_alloc(bytes);require(f->w[i]!=NULL,"weight allocation");
        load_tensor(f->w[i],t,fd,true);
    }
    const char *ns[]={"attn_q_a_norm.weight","attn_kv_a_norm.weight"};
    for(unsigned i=0;i<2;i++) {
        f->norm[i]=myds_metal_buffer_alloc((i?512:1280)*4u);require(f->norm[i]!=NULL,"norm allocation");
        NSDictionary *t=lookup(tensors,[NSString stringWithFormat:@"blk.%u.%s",layer,ns[i]]);
        require(t&&[[t objectForKey:@"type_name"] isEqual:@"F32"],"actual norm type mismatch");
        load_tensor(f->norm[i],t,fd,false);
    }
    if(fd>=0)close(fd);
#define ALLOC(m,n) do {f->m=myds_metal_buffer_alloc((n)*4u);require(f->m!=NULL,"activation allocation");fill_f32(f->m,7);}while(0)
    ALLOC(x,5120);ALLOC(qr,1280);ALLOC(qr_norm,1280);ALLOC(q,32768);ALLOC(kv,512);ALLOC(window,128*512);
#undef ALLOC
}
enum op { GEMV_QA,GEMV_QB,GEMV_KV,BF16_QA,BF16_QB,BF16_KV,NORM_QA,NORM_KV,ROPE_Q,ROPE_KV,QUANT_KV,COPY_WINDOW,CHAIN_OLD,
#ifdef MYDS_QKV_BENCH_FUSED
    GEMV_QA_FUSED,GEMV_QB_FUSED,GEMV_KV_FUSED,FINISH_FUSED,CHAIN_FUSED,
#endif
    OP_COUNT };
static const char *op_name[]={"gemv.qa","gemv.qb","gemv.kv","bf16.qa","bf16.qb","bf16.kv","rmsnorm.qa","rmsnorm.kv","rope.q","rope.kv","quant.kv","copy.window","chain.old",
#ifdef MYDS_QKV_BENCH_FUSED
    "gemv.qa_bf16","gemv.qb_bf16","gemv.kv_bf16","finish.fused","chain.fused",
#endif
};
static int issue(struct fixture *f,enum op op) {
    switch(op) {
    case GEMV_QA:myds_metal_barrier();return myds_metal_gemv(f->qr,f->w[0],f->x,1280,5120,GGML_TYPE_Q8_0);
    case GEMV_QB:myds_metal_barrier();return myds_metal_gemv(f->q,f->w[1],f->qr_norm,32768,1280,GGML_TYPE_Q8_0);
    case GEMV_KV:myds_metal_barrier();return myds_metal_gemv(f->kv,f->w[2],f->x,512,5120,GGML_TYPE_Q8_0);
    case BF16_QA:return myds_metal_v41_bf16(f->qr,1280);
    case BF16_QB:return myds_metal_v41_bf16(f->q,32768);
    case BF16_KV:return myds_metal_v41_bf16(f->kv,512);
    case NORM_QA:return myds_metal_v41_rmsnorm(f->qr_norm,f->qr,f->norm[0],1280,1e-6f,true);
    case NORM_KV:return myds_metal_v41_rmsnorm(f->kv,f->kv,f->norm[1],512,1e-6f,true);
    case ROPE_Q:return myds_v41_gpu_rope(f->q,512,64,f->pos,f->compressed,false);
    case ROPE_KV:return myds_v41_gpu_rope(f->kv,512,1,f->pos,f->compressed,false);
    case QUANT_KV:return myds_v41_gpu_quant(f->kv,512,1,1);
    case COPY_WINDOW:return myds_metal_buffer_copy(f->window,(size_t)(f->pos%128)*512*4,f->kv,0,512*4);
    case CHAIN_OLD:return issue(f,GEMV_QA)&&issue(f,BF16_QA)&&issue(f,NORM_QA)&&issue(f,GEMV_QB)&&issue(f,BF16_QB)&&
        issue(f,GEMV_KV)&&issue(f,BF16_KV)&&issue(f,NORM_KV)&&issue(f,ROPE_Q)&&issue(f,ROPE_KV)&&issue(f,QUANT_KV)&&issue(f,COPY_WINDOW);
#ifdef MYDS_QKV_BENCH_FUSED
    case GEMV_QA_FUSED:myds_metal_barrier();return myds_metal_v41_gemv_bf16(f->qr,f->w[0],f->x,1280,5120,GGML_TYPE_Q8_0);
    case GEMV_QB_FUSED:myds_metal_barrier();return myds_metal_v41_gemv_bf16(f->q,f->w[1],f->qr_norm,32768,1280,GGML_TYPE_Q8_0);
    case GEMV_KV_FUSED:myds_metal_barrier();return myds_metal_v41_gemv_bf16(f->kv,f->w[2],f->x,512,5120,GGML_TYPE_Q8_0);
    case FINISH_FUSED:return myds_v41_gpu_qkv_finish(f->q,f->kv,f->window,f->pos,f->compressed);
    case CHAIN_FUSED:return issue(f,GEMV_QA_FUSED)&&issue(f,NORM_QA)&&issue(f,GEMV_QB_FUSED)&&issue(f,GEMV_KV_FUSED)&&issue(f,NORM_KV)&&issue(f,FINISH_FUSED);
#endif
    default:return 0;
    }
}
static void measure(struct fixture *f,enum op op,unsigned reps,double *gpu,double *wall) {
    require(myds_metal_begin_batch(),"begin batch");
    id<MTLCommandBuffer> cb=g_batch_cb;double t=wall_now();
    for(unsigned i=0;i<reps;i++)require(issue(f,op),"component dispatch");
    require(myds_metal_commit_batch(),"commit batch");
    *wall=(wall_now()-t)*1e6/reps;
    require(cb.status==MTLCommandBufferStatusCompleted&&cb.GPUEndTime>cb.GPUStartTime,"GPU CB timestamps unavailable");
    *gpu=(cb.GPUEndTime-cb.GPUStartTime)*1e6/reps;
}
#ifdef MYDS_QKV_BENCH_FUSED
static void verify_chain(struct fixture *f) {
    myds_metal_buffer_t *b[]={f->qr,f->qr_norm,f->q,f->kv,f->window};void *old[5];double g,w;
    measure(f,CHAIN_OLD,1,&g,&w);
    for(unsigned i=0;i<5;i++) {
        size_t bytes=myds_metal_buffer_size(b[i]);old[i]=malloc(bytes);require(old[i]!=NULL,"reference snapshot");
        memcpy(old[i],myds_metal_buffer_data(b[i]),bytes);
    }
    measure(f,CHAIN_FUSED,1,&g,&w);
    for(unsigned i=0;i<5;i++) {
        require(!memcmp(old[i],myds_metal_buffer_data(b[i]),myds_metal_buffer_size(b[i])),"actual-weight full QKV chain not byte exact");
        const float *v=myds_metal_buffer_data(b[i]);size_t nz=0;
        for(size_t j=0;j<myds_metal_buffer_size(b[i])/4;j++){require(isfinite(v[j]),"chain nonfinite output");nz+=v[j]!=0;}
        require(nz!=0,"chain unexpectedly all zero");free(old[i]);
    }
    fprintf(stderr,"# old/fused QKV chain byte-exact for all QR/QRnorm/Q/KV/window values\n");
}
#endif
static int compare(const void *a,const void *b){double x=*(const double *)a,y=*(const double *)b;return(x>y)-(x<y);}
#ifdef MYDS_QKV_BENCH_FUSED
static void paired_chain(struct fixture *f,unsigned samples,unsigned reps) {
    double a[31],b[31],wa[31],wb[31],g,w;
    measure(f,CHAIN_OLD,4,&g,&w);measure(f,CHAIN_FUSED,4,&g,&w);
    for(unsigned i=0;i<samples;i++) {
        if(i%2){measure(f,CHAIN_FUSED,reps,&b[i],&wb[i]);measure(f,CHAIN_OLD,reps,&a[i],&wa[i]);}
        else {measure(f,CHAIN_OLD,reps,&a[i],&wa[i]);measure(f,CHAIN_FUSED,reps,&b[i],&wb[i]);}
        printf("pair.sample.%u.old,%.6f,%.6f,%016" PRIx64 "\n",i,a[i],wa[i],checksum(f->q)^checksum(f->kv)^checksum(f->qr));
        printf("pair.sample.%u.fused,%.6f,%.6f,%016" PRIx64 "\n",i,b[i],wb[i],checksum(f->q)^checksum(f->kv)^checksum(f->qr));
    }
    qsort(a,samples,sizeof(double),compare);qsort(b,samples,sizeof(double),compare);
    qsort(wa,samples,sizeof(double),compare);qsort(wb,samples,sizeof(double),compare);
    printf("pair.median.old,%.6f,%.6f,%016" PRIx64 "\n",a[samples/2],wa[samples/2],checksum(f->q)^checksum(f->kv)^checksum(f->qr));
    printf("pair.median.fused,%.6f,%.6f,%016" PRIx64 "\n",b[samples/2],wb[samples/2],checksum(f->q)^checksum(f->kv)^checksum(f->qr));
}
#endif
int main(int argc,char **argv) {
    const char *layout=NULL,*model=NULL,*shader="metal";unsigned layer=0,samples=9,reps=64;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--layout")&&i+1<argc)layout=argv[++i];
        else if(!strcmp(argv[i],"--model")&&i+1<argc)model=argv[++i];
        else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader=argv[++i];
        else if(!strcmp(argv[i],"--layer")&&i+1<argc)layer=(unsigned)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=(unsigned)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--reps")&&i+1<argc)reps=(unsigned)strtoul(argv[++i],NULL,10);
        else require(0,"unknown or incomplete argument");
    }
    require(layout&&layer<40&&samples>0&&samples<=31&&reps>0&&reps<=4096,"required --layout or invalid benchmark range");
    @autoreleasepool {
        require(myds_metal_init(shader),"Metal init");struct fixture f;fixture_create(&f,layout,model,layer);
        printf("# GPU=%s weights=%s layer=%u samples=%u reps=%u types=Q8_0 shapes=1280x5120,32768x1280,512x5120\n",g_device.name.UTF8String,model?"actual GGUF selected tensors":"synthetic nonzero",layer,samples,reps);
        printf("# MYDS_QKV_NR0=%s\n",getenv("MYDS_QKV_NR0")?getenv("MYDS_QKV_NR0"):"4 default");
#ifdef MYDS_QKV_BENCH_FUSED
        require(myds_metal_v41_qkv_fusion_available(),"fused shader unavailable; refusing fallback-only benchmark");
        verify_chain(&f);
#endif
        printf("component,median_gpu_us,median_host_wall_us,checksum\n");
        for(unsigned op=0;op<OP_COUNT;op++) {
            double g[31],w[31],a,b;measure(&f,(enum op)op,4,&a,&b);
            for(unsigned s=0;s<samples;s++)measure(&f,(enum op)op,reps,&g[s],&w[s]);
            qsort(g,samples,sizeof(double),compare);qsort(w,samples,sizeof(double),compare);
            printf("%s,%.6f,%.6f,%016" PRIx64 "\n",op_name[op],g[samples/2],w[samples/2],checksum(f.q)^checksum(f.kv)^checksum(f.qr));fflush(stdout);
        }
#ifdef MYDS_QKV_BENCH_FUSED
        paired_chain(&f,samples,reps);
#endif
        for(unsigned i=0;i<3;i++)myds_metal_buffer_free(f.w[i]);
        for(unsigned i=0;i<2;i++)myds_metal_buffer_free(f.norm[i]);
        myds_metal_buffer_t *b[]={f.x,f.qr,f.qr_norm,f.q,f.kv,f.window};
        for(unsigned i=0;i<sizeof(b)/sizeof(*b);i++)myds_metal_buffer_free(b[i]);
        myds_metal_cleanup();
    }
    return 0;
}
