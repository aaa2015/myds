/* CPU-only: includes the actual private production selector and scratch type.
 * No Metal framework/device/library is linked or used. */
#define _POSIX_C_SOURCE 200809L
#include "../src/myds_v41_gpu.c"
#include <assert.h>
#include <stdio.h>

struct myds_metal_buffer { uintptr_t base;size_t bytes,offset,native_bytes;int valid; };
static int batch=1,kprof,preflight_calls,dispatch_calls,native_range_calls,log_calls;
static const char *preflight_bad,*dispatch_bad;
static char dispatched[8][96];
static struct indexer_register_only_args got;
static const myds_metal_buffer_t *got_buffers[4];
static unsigned controls;
void myds_log(myds_log_level_t level,const char *format,...) {(void)level;(void)format;log_calls++;}
size_t myds_metal_buffer_size(const myds_metal_buffer_t *b){return b?b->bytes:0;}
size_t myds_metal_buffer_offset(const myds_metal_buffer_t *b){return b?b->offset:0;}
void *myds_metal_buffer_data(myds_metal_buffer_t *b){return b?(void*)(b->base+b->offset):NULL;}
int myds_metal_buffer_native_range(const myds_metal_buffer_t *b,uint64_t n,size_t align) {
    native_range_calls++;
    return b&&b->valid&&b->base&&n&&align&&b->offset<=b->native_bytes&&
        n<=b->bytes&&n<=b->native_bytes-b->offset&&!(b->offset%align)&&!((b->base+b->offset)%align);
}
int myds_metal_batch_active(void){return batch;}
int myds_metal_kprof_exclusive(void){return kprof;}
int myds_metal_kernel_constants_preflight(const char *name,const uint32_t threads[3],size_t smem,
    const myds_metal_constant_t *constants,size_t n) {
    preflight_calls++;
    assert(threads[0]==32&&threads[1]==4&&threads[2]==1&&smem==0&&!constants&&!n);
    return !preflight_bad||strcmp(preflight_bad,name);
}
int myds_metal_dispatch_kernel(const char *name,const void *args,size_t size,
    const myds_metal_buffer_t *const *buffers,size_t count,const uint32_t groups[3],
    const uint32_t threads[3],size_t smem) {
    assert(dispatch_calls<8);snprintf(dispatched[dispatch_calls++],96,"%s",name);
    if(strstr(name,"register_only")) {
        assert(size==72&&count==4&&threads[0]==32&&threads[1]==4&&threads[2]==1&&smem==0);
        memcpy(&got,args,72);memcpy(got_buffers,buffers,sizeof(got_buffers));
        assert(groups[0]==(got.n_comp+31)/32&&groups[1]==(got.n_tokens+7)/8&&groups[2]==1);
    }
    return !dispatch_bad||strcmp(dispatch_bad,name);
}
struct fixture {struct myds_v41_gpu_scratch s;struct myds_metal_buffer y,q,w,k;uint32_t nc,nt,start,ratio;bool bf16;};
static void configure(struct fixture *f,uint32_t nc,uint32_t nt,uint32_t start,uint32_t ratio,bool bf) {
    memset(f,0,sizeof(*f));f->s.max_comp=f->s.key_comp_max=327680;f->s.batch_floats=2097152;
    f->nc=nc;f->nt=nt;f->start=start;f->ratio=ratio;f->bf16=bf;
    uint64_t bytes[]={nc*(uint64_t)nt*4,nt*16384ull,nt*128ull,nc*(bf?256ull:512ull)};
    struct myds_metal_buffer *v[]={&f->y,&f->q,&f->w,&f->k};
    for(unsigned i=0;i<4;i++)*v[i]=(struct myds_metal_buffer){UINT64_C(0x100000000)+(uint64_t)i*UINT64_C(0x10000000),(size_t)bytes[i],0,(size_t)bytes[i]+256,1};
}
static void reset(void) {
    myds_v41_gpu_indexer_register_only_counters(NULL,true);preflight_calls=dispatch_calls=native_range_calls=log_calls=0;
    preflight_bad=dispatch_bad=NULL;batch=1;kprof=0;memset(dispatched,0,sizeof(dispatched));memset(&got,0,sizeof(got));
    unsetenv("MYDS_INDEXER_TILED");unsetenv("MYDS_INDEXER_SGBAR");unsetenv("MYDS_INDEXER_CHUNK_EXACT");
}
static int call(struct fixture *f,bool ordinary) {
    return myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(&f->s,&f->y,&f->q,&f->w,&f->k,
        f->nc,32,f->nt,f->start,f->ratio,f->bf16,ordinary);
}
static void counts(unsigned encoded,unsigned failed,unsigned bf) {
    myds_v41_gpu_indexer_register_only_counters_t c;
    myds_v41_gpu_indexer_register_only_counters(&c,false);
    assert(c.requests==1&&c.encoded==encoded&&!c.fallbacks&&c.failures==failed);
    assert(c.bf16_dispatches==(encoded&&bf)&&c.f32_dispatches==0);
    controls++;
}
static void fallback_counts(unsigned fallbacks,unsigned failed) {
    myds_v41_gpu_indexer_register_only_counters_t c;
    myds_v41_gpu_indexer_register_only_counters(&c,false);
    assert(c.requests==1&&c.encoded==0&&c.fallbacks==fallbacks&&c.failures==failed);
    controls++;
}
static void reference(struct fixture *f) {
    assert(call(f,true));counts(0,0,0);assert(dispatch_calls==1&&!preflight_calls&&!native_range_calls);
    assert(!strcmp(dispatched[0],"kernel_myds_indexer_score_chunk_exact"));
}
int main(int argc,char **argv) {
    assert(argc==2);int mode=myds_v41_gpu_indexer_register_only_enabled();struct fixture f;
    if(!strcmp(argv[1],"parse")) {
        const char *expected=getenv("EXPECT_MODE");assert(expected&&mode==atoi(expected));
        const char *f32=myds_v41_gpu_indexer_register_only_selected_kernel(false),*bf=myds_v41_gpu_indexer_register_only_selected_kernel(true);
        assert(f32==NULL&&(mode==1)==(bf!=NULL));
        if(mode==1){assert(!strcmp(bf,"kernel_myds_indexer_scores_register_only_bf16"));}
        assert(!preflight_calls&&!dispatch_calls);printf("{\"parse\":true,\"mode\":%d,\"gpu_work\":false}\n",mode);return 0;
    }
    configure(&f,15424,128,30720,2,true);reset();
    if(mode==0) {
        reference(&f);reset();setenv("MYDS_INDEXER_TILED","1",1);assert(call(&f,true));counts(0,0,0);
        assert(!strcmp(dispatched[0],"kernel_dsv4_indexer_scores_tiled_bf16")&&!preflight_calls&&!native_range_calls);
        reset();dispatch_bad="kernel_myds_indexer_score_chunk_exact";assert(call(&f,true));counts(0,0,0);
        assert(dispatch_calls==2&&!strcmp(dispatched[1],"kernel_glm_indexer_score_chunk_direct"));
        reset();assert(myds_v41_gpu_indexscores_chunk_bf16(&f.y,&f.q,&f.w,&f.k,f.nc,32,f.nt,f.start,f.ratio));
        myds_v41_gpu_indexer_register_only_counters_t c;myds_v41_gpu_indexer_register_only_counters(&c,false);assert(!c.requests&&!c.encoded&&!preflight_calls);
        printf("{\"controls\":%u,\"mode\":0,\"gpu_work\":false}\n",controls);return 0;
    }
    if(mode<0){assert(!call(&f,true));counts(0,1,0);assert(!dispatch_calls&&!preflight_calls);return 0;}
    assert(mode==1);
    if(getenv("MYDS_INDEXER_KEY_WINDOW")) {reference(&f);printf("{\"window_reference\":true,\"gpu_work\":false}\n");return 0;}
    assert(call(&f,true));counts(1,0,1);assert(preflight_calls==1&&dispatch_calls==1&&native_range_calls==4);
    assert(got.n_comp==15424&&got.n_tokens==128&&got.n_head==32&&got.head_dim==128&&got.pos0==30720&&got.ratio==2);
    assert(got.q_token_stride==16384&&got.q_head_stride==512&&got.weights_token_stride==128&&got.index_row_stride==256&&got.score_token_stride==61696&&got.scale==1.f/64.f);
    assert(got_buffers[0]==&f.q&&got_buffers[1]==&f.w&&got_buffers[2]==&f.k&&got_buffers[3]==&f.y);
    reset();configure(&f,1024,8,2040,2,false);reference(&f);
    reset();configure(&f,513,9,1017,2,true);assert(call(&f,true));counts(1,0,1);
    reset();configure(&f,16384,8,32760,2,true);assert(call(&f,true));counts(1,0,1);
    reset();configure(&f,15424,128,30720,2,true);assert(call(&f,false));counts(0,0,0);assert(!preflight_calls&&!native_range_calls);
    const uint32_t cases[][4]={{512,8,1016,2},{16385,8,32762,2},{2048,8,2040,1},{1024,7,2041,2},{16384,8,32761,2},{2048,2049,2047,2}};
    for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){reset();configure(&f,cases[i][0],cases[i][1],cases[i][2],cases[i][3],true);reference(&f);}
    reset();configure(&f,15424,128,30720,2,true);batch=0;reference(&f);
    reset();kprof=1;reference(&f);
    reset();setenv("MYDS_INDEXER_SGBAR","1",1);reference(&f);
    reset();setenv("MYDS_INDEXER_TILED","1",1);setenv("MYDS_INDEXER_SGBAR","1",1);assert(call(&f,true));counts(0,0,0);assert(!strcmp(dispatched[0],"kernel_dsv4_indexer_scores_tiled_sg_bf16"));
    reset();setenv("MYDS_INDEXER_TILED","1",1);assert(call(&f,true));counts(1,0,1);assert(!strcmp(dispatched[0],"kernel_myds_indexer_scores_register_only_bf16"));
    setenv("MYDS_INDEXER_REGISTER_ONLY_STRICT","1",1);
    for(unsigned i=0;i<4;i++){reset();configure(&f,15424,128,30720,2,true);struct myds_metal_buffer *v[]={&f.y,&f.q,&f.w,&f.k};v[i]->bytes--;assert(!call(&f,true));counts(0,1,0);assert(!dispatch_calls&&!preflight_calls);}
    for(unsigned i=0;i<4;i++){reset();configure(&f,15424,128,30720,2,true);struct myds_metal_buffer *v[]={&f.y,&f.q,&f.w,&f.k};v[i]->offset=1;assert(!call(&f,true));counts(0,1,0);assert(!dispatch_calls&&!preflight_calls);}
    for(unsigned i=0;i<4;i++){reset();configure(&f,15424,128,30720,2,true);struct myds_metal_buffer *v[]={&f.y,&f.q,&f.w,&f.k};v[i]->native_bytes=v[i]->bytes-1;assert(!call(&f,true));counts(0,1,0);assert(!dispatch_calls&&!preflight_calls);}
    for(unsigned i=0;i<3;i++)for(unsigned off=0;off<3;off++){reset();configure(&f,15424,128,30720,2,true);struct myds_metal_buffer *v[]={&f.q,&f.w,&f.k};f.y.base=v[i]->base+(off==0?0:off==1?4:v[i]->bytes-4);assert(!call(&f,true));counts(0,1,0);assert(!dispatch_calls&&!preflight_calls);}
    reset();configure(&f,15424,128,30720,2,true);f.s.batch_floats=15424u*128u-1;assert(!call(&f,true));counts(0,1,0);
    reset();configure(&f,15424,128,30720,2,true);f.s.max_comp=15423;assert(!call(&f,true));counts(0,1,0);
    reset();configure(&f,15424,128,30720,2,true);f.s.key_comp_max=15423;assert(!call(&f,true));counts(0,1,0);
    reset();configure(&f,15424,128,30718,2,true);assert(!call(&f,true));counts(0,1,0);
    reset();configure(&f,15424,128,30720,2,true);assert(!myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(NULL,&f.y,&f.q,&f.w,&f.k,f.nc,32,f.nt,f.start,f.ratio,true,true));counts(0,1,0);
    reset();assert(!myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(&f.s,&f.y,&f.q,&f.w,&f.k,f.nc,31,f.nt,f.start,f.ratio,true,true));counts(0,0,0);
    reset();configure(&f,15424,128,30720,2,true);preflight_bad="kernel_myds_indexer_scores_register_only_bf16";assert(!call(&f,true));counts(0,1,0);assert(!dispatch_calls&&preflight_calls==1);
    reset();configure(&f,15424,128,30720,2,true);dispatch_bad="kernel_myds_indexer_scores_register_only_bf16";assert(!call(&f,true));counts(0,1,0);assert(dispatch_calls==1&&preflight_calls==1);
    reset();assert(call(&f,true));counts(1,0,1);preflight_bad="kernel_myds_indexer_scores_register_only_bf16";
    myds_v41_gpu_indexer_register_only_counters(NULL,true);dispatch_calls=preflight_calls=0;assert(!call(&f,true));counts(0,1,0);assert(!dispatch_calls&&preflight_calls==1); /* no stale readiness after hypothetical library change */
    unsetenv("MYDS_INDEXER_REGISTER_ONLY_STRICT");
    /* Non-strict production graceful fallback tests */
    reset();configure(&f,15424,128,30720,2,true);preflight_bad="kernel_myds_indexer_scores_register_only_bf16";
    assert(call(&f,true));fallback_counts(1,0);assert(dispatch_calls==1&&preflight_calls==1);
    assert(!strcmp(dispatched[0],"kernel_myds_indexer_score_chunk_exact"));
    reset();configure(&f,15424,128,30720,2,true);dispatch_bad="kernel_myds_indexer_scores_register_only_bf16";
    assert(call(&f,true));fallback_counts(1,0);assert(dispatch_calls==2&&preflight_calls==1);
    assert(!strcmp(dispatched[1],"kernel_myds_indexer_score_chunk_exact"));
    reset();configure(&f,15424,128,30720,2,true);f.s.batch_floats=15424u*128u-1;
    assert(call(&f,true));fallback_counts(1,0);assert(dispatch_calls==1&&!preflight_calls);
    assert(!strcmp(dispatched[0],"kernel_myds_indexer_score_chunk_exact"));
    reset();assert(myds_v41_gpu_indexscores_chunk_bf16(&f.y,&f.q,&f.w,&f.k,f.nc,32,f.nt,f.start,f.ratio));
    myds_v41_gpu_indexer_register_only_counters_t c;myds_v41_gpu_indexer_register_only_counters(&c,false);assert(!c.requests&&!c.encoded&&!preflight_calls);
    int before=dispatch_calls;myds_v41_gpu_indexer_register_only_counters(NULL,true);myds_v41_gpu_indexer_register_only_counters(&c,false);assert(!c.requests&&!c.encoded&&dispatch_calls==before);
    printf("{\"controls\":%u,\"mode\":1,\"gpu_work\":false,\"actual_source_included\":true}\n",controls);return 0;
}
