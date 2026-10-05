#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); return 0; } } while (0)
struct fixture {
    uint32_t rows,start,ratio,ncomp;
    myds_metal_buffer_t *q,*kv,*comp,*packed,*ids,*sinks,*gather,*window;
    myds_metal_buffer_t *out[3],*owner[3];
};
static float sample(uint32_t i,uint32_t seed) {
    uint32_t x=(i+1u)*747796405u+seed*2891336453u;
    x=((x>>((x>>28)+4))^x)*277803737u;x=(x>>22)^x;
    float f=((int32_t)(x%8191u)-4095)/8192.0f;
    uint32_t b;memcpy(&b,&f,4);b=(b+0x7fffu+((b>>16)&1u))&0xffff0000u;memcpy(&f,&b,4);return f;
}
static uint32_t gcd(uint32_t a,uint32_t b) { while(b) { uint32_t t=a%b;a=b;b=t; } return a; }
static int init(struct fixture *f,uint32_t rows,uint32_t start,uint32_t ratio) {
    memset(f,0,sizeof(*f));f->rows=rows;f->start=start;f->ratio=ratio;
    f->ncomp=(start+rows)/ratio;if(!f->ncomp)f->ncomp=1;
#define ALLOC(k,n) do { f->k=myds_metal_buffer_alloc(n);CHECK(f->k); } while(0)
    ALLOC(q,(size_t)rows*64u*512u*4u);ALLOC(kv,(size_t)rows*512u*4u);
    ALLOC(comp,(size_t)f->ncomp*512u*4u);ALLOC(packed,(size_t)f->ncomp*512u*2u);
    ALLOC(ids,(size_t)rows*512u*4u);ALLOC(sinks,64u*4u);
    ALLOC(gather,(size_t)rows*640u*512u*2u);ALLOC(window,128u*512u*4u);
#undef ALLOC
    size_t bytes=(size_t)rows*64u*512u*4u;
    for(int k=0;k<3;k++) {
        f->owner[k]=myds_metal_buffer_alloc(bytes+32u);CHECK(f->owner[k]);
        memset(myds_metal_buffer_data(f->owner[k]),0xa5,bytes+32u);
        f->out[k]=myds_metal_buffer_sub_view(f->owner[k],16,bytes);CHECK(f->out[k]);
    }
    myds_metal_buffer_t *bs[]={f->q,f->kv,f->comp,f->window};
    for(unsigned k=0;k<4;k++) {
        float *p=myds_metal_buffer_data(bs[k]);
        for(size_t i=0;i<myds_metal_buffer_size(bs[k])/4u;i++)p[i]=sample((uint32_t)i,k==2?3u:start+k+1u);
    }
    const float *comp=myds_metal_buffer_data(f->comp);uint16_t *packed=myds_metal_buffer_data(f->packed);
    for(size_t i=0;i<(size_t)f->ncomp*512u;i++) { uint32_t b;memcpy(&b,comp+i,4);packed[i]=(uint16_t)(b>>16); }
    float *sink=myds_metal_buffer_data(f->sinks);for(unsigned h=0;h<64;h++)sink[h]=((int)h-31)/16.0f;
    int32_t *ids=myds_metal_buffer_data(f->ids);
    for(uint32_t r=0;r<rows;r++) {
        uint32_t count=(start+r+1u)/ratio,step=count>1u?count/2u+1u:1u;
        while(gcd(step,count)!=1u && count)step++;
        for(uint32_t j=0;j<512;j++)ids[(size_t)r*512u+j]=j<(count<512u?count:512u)?(int32_t)(((uint64_t)j*step+r*17u)%count):-1;
    }
    return 1;
}
static void cleanup(struct fixture *f) {
    myds_metal_buffer_t *bs[]={f->q,f->kv,f->comp,f->packed,f->ids,f->sinks,f->gather,f->window};
    for(size_t i=0;i<sizeof(bs)/sizeof(*bs);i++)myds_metal_buffer_free(bs[i]);
    for(int k=0;k<3;k++) { myds_metal_buffer_free(f->out[k]);myds_metal_buffer_free(f->owner[k]); }
}
static myds_v41_gpu_scratch_t *scratch(const char *variant,const char *head_first) {
    setenv("MYDS_PREFILL_ATTN",variant,1);
    if(head_first)setenv("MYDS_ATTN_VEC_HEAD_FIRST",head_first,1);else unsetenv("MYDS_ATTN_VEC_HEAD_FIRST");
    return myds_v41_gpu_scratch_create(8192);
}
/* mode 0 gathered compressed, 1 window only with a supplied gathered buffer,
 * 2 compressed slab. Both compressed formats use identical represented data. */
static int call(struct fixture *f,myds_v41_gpu_scratch_t *s,int out,int packed,int mode) {
    uint32_t n=mode==1?0u:f->ncomp;
    myds_metal_buffer_t *comp=n?(packed?f->packed:f->comp):NULL,*ids=n?f->ids:NULL;
    myds_metal_buffer_t *gather=mode==2?NULL:f->gather;
    return packed?myds_v41_gpu_attention_chunk_bf16(f->out[out],f->sinks,f->q,f->kv,comp,ids,gather,f->rows,n,512,f->start,0,n?f->ratio:0,s):
        myds_v41_gpu_attention_chunk(f->out[out],f->sinks,f->q,f->kv,comp,ids,gather,f->rows,n,512,f->start,0,n?f->ratio:0,s);
}
static int stage(struct fixture *f,myds_v41_gpu_scratch_t *s) {
    return !f->start||myds_v41_gpu_carry_stage(s,f->window,f->start);
}
static int compare(struct fixture *f,int out) {
    size_t count=(size_t)f->rows*64u*512u,bytes=count*4u,nonzero=0;
    const float *a=myds_metal_buffer_data(f->out[0]),*b=myds_metal_buffer_data(f->out[out]);
    CHECK(!memcmp(a,b,bytes));
    for(size_t i=0;i<count;i++) { CHECK(isfinite(a[i]));nonzero+=a[i]!=0; }
    CHECK(nonzero>0);
    for(int k=0;k<3;k++) {
        const uint8_t *p=myds_metal_buffer_data(f->owner[k]);
        for(size_t i=0;i<16;i++)CHECK(p[i]==0xa5&&p[16u+bytes+i]==0xa5);
    }
    return 1;
}
static int numeric(uint32_t rows,uint32_t start,uint32_t ratio) {
    struct fixture f;CHECK(init(&f,rows,start,ratio));
    myds_v41_gpu_scratch_t *s[]={scratch("vec","0"),scratch("vec",NULL),scratch("vec","1")};
    CHECK(s[0]&&s[1]&&s[2]);CHECK(myds_v41_gpu_attention_vec_head_first_status(s[0])==0);
    /* Policy is captured at creation, so later environment changes cannot alter a batch. */
    setenv("MYDS_ATTN_VEC_HEAD_FIRST","0",1);
    for(int packed=0;packed<2;packed++) {
        CHECK(myds_metal_begin_batch());
        for(int k=0;k<3;k++) { CHECK(stage(&f,s[k]));CHECK(call(&f,s[k],k,packed,0)); }
        CHECK(myds_metal_commit_batch());CHECK(compare(&f,1)&&compare(&f,2));
        /* Refuse fallback-only success: the compiled transpose PSO must have encoded. */
        CHECK(myds_v41_gpu_attention_vec_head_first_status(s[1])==1);
        CHECK(myds_v41_gpu_attention_vec_head_first_status(s[2])==1);
    }
    printf("PASS head-first exact rows=%u start=%u ratio=%u F32/BF16 default/explicit same-CB guarded outputs\n",rows,start,ratio);
    for(int k=0;k<3;k++)myds_v41_gpu_scratch_free(s[k]);cleanup(&f);return 1;
}
static int unavailable(const char *env) {
    struct fixture f;CHECK(init(&f,33,1536,2));
    myds_v41_gpu_scratch_t *vec=scratch("vec","0"),*hf=scratch("vec",env);CHECK(vec&&hf);
    myds_v41_gpu_inject_attention_vec_head_first_failure(true);
    CHECK(myds_metal_begin_batch());
    /* A failing optional PSO must not even be consulted for window/slab paths. */
    for(int mode=1;mode<=2;mode++) {
        CHECK(stage(&f,vec)&&stage(&f,hf));CHECK(call(&f,vec,0,1,mode)&&call(&f,hf,1,1,mode));
        CHECK(myds_v41_gpu_attention_vec_head_first_status(hf)==1);
    }
    CHECK(stage(&f,vec)&&stage(&f,hf));CHECK(call(&f,vec,0,1,0)&&call(&f,hf,1,1,0));
    CHECK(myds_metal_commit_batch());CHECK(compare(&f,1));CHECK(myds_v41_gpu_attention_vec_head_first_status(hf)==2);
    myds_v41_gpu_inject_attention_vec_head_first_failure(false);
    CHECK(myds_metal_begin_batch());CHECK(stage(&f,hf));CHECK(call(&f,hf,1,1,0));CHECK(myds_metal_commit_batch());
    CHECK(compare(&f,1));CHECK(myds_v41_gpu_attention_vec_head_first_status(hf)==2);
    printf("PASS missing head-first PSO %s fallback exact and cached; window/slab unchanged\n",env?"explicit=1":"default");
    myds_v41_gpu_scratch_free(vec);myds_v41_gpu_scratch_free(hf);cleanup(&f);return 1;
}
static int auto_fallback(int head_first_missing) {
    struct fixture f;CHECK(init(&f,128,1024,2));
    myds_v41_gpu_scratch_t *vec=scratch("vec","0"),*a=scratch("auto",NULL);CHECK(vec&&a);
    myds_v41_gpu_inject_attention_matrix_failure(1);
    myds_v41_gpu_inject_attention_vec_head_first_failure(head_first_missing);
    CHECK(myds_metal_begin_batch());CHECK(stage(&f,vec));CHECK(call(&f,vec,0,1,0));
    CHECK(stage(&f,a));uint64_t before=myds_metal_dispatch_count();CHECK(call(&f,a,1,1,0));uint64_t first=myds_metal_dispatch_count()-before;
    CHECK(myds_metal_commit_batch());CHECK(compare(&f,1));CHECK(myds_v41_gpu_attention_vec_head_first_status(a)==(head_first_missing?2u:1u));
    CHECK(myds_metal_begin_batch());CHECK(stage(&f,a));before=myds_metal_dispatch_count();CHECK(call(&f,a,1,1,0));uint64_t retry=myds_metal_dispatch_count()-before;
    CHECK(myds_metal_commit_batch());CHECK(compare(&f,1));CHECK(first==retry+1u);
    myds_v41_gpu_inject_attention_matrix_failure(0);myds_v41_gpu_inject_attention_vec_head_first_failure(false);
    printf("PASS matrix auto fallback chooses %s vec, byte exact with unchanged original args\n",head_first_missing?"original":"head-first");
    myds_v41_gpu_scratch_free(vec);myds_v41_gpu_scratch_free(a);cleanup(&f);return 1;
}
static void next_window(struct fixture *next,const struct fixture *prev) {
    float *w=myds_metal_buffer_data(next->window);const float *kv=myds_metal_buffer_data(prev->kv);
    memcpy(w,myds_metal_buffer_data(prev->window),128u*512u*4u);
    for(uint32_t r=0;r<prev->rows;r++)memcpy(w+(size_t)((prev->start+r)%128u)*512u,kv+(size_t)r*512u,512u*4u);
}
static int mixed_carry(void) {
    struct fixture f[5];const uint32_t rows[]={64,33,1,31,33};const int mode[]={0,0,1,2,0};uint32_t start=1536;
    for(int i=0;i<5;i++) { CHECK(init(f+i,rows[i],start,2));if(i)next_window(f+i,f+i-1);start+=rows[i]; }
    myds_v41_gpu_scratch_t *vec=scratch("vec","0"),*hf=scratch("vec",NULL),*fresh[5];CHECK(vec&&hf);
    for(int i=0;i<5;i++) { fresh[i]=scratch("vec","0");CHECK(fresh[i]); }
    CHECK(myds_metal_begin_batch());CHECK(stage(f,vec)&&stage(f,hf));
    for(int i=0;i<5;i++) {
        /* Persistent carry flows across overlapping short tails in one CB. A
         * separately staged ring provides an independent carry reference. */
        CHECK(call(f+i,vec,0,i%2,mode[i])&&call(f+i,hf,1,i%2,mode[i]));
        CHECK(stage(f+i,fresh[i])&&call(f+i,fresh[i],2,i%2,mode[i]));
    }
    CHECK(myds_metal_commit_batch());
    for(int i=0;i<5;i++) { CHECK(compare(f+i,1)&&compare(f+i,2));myds_v41_gpu_scratch_free(fresh[i]);cleanup(f+i); }
    CHECK(myds_v41_gpu_attention_vec_head_first_status(hf)==1);
    myds_v41_gpu_scratch_free(vec);myds_v41_gpu_scratch_free(hf);
    puts("PASS short-chunk carry + gather/window/slab/gather in one CB, independent ring reference byte exact");return 1;
}
int main(void) {
    printf("ATTENTION VEC compiled=%s %s\n",__DATE__,__TIME__);fflush(stdout);
    if(!myds_metal_init("metal"))return 1;
    setenv("MYDS_ATTN_GATHER","1",1);unsetenv("MYDS_ATTN_GATHER_MIN_KEYS");unsetenv("MYDS_ATTN_KEYS_CLAMP");
    const uint32_t cases[][3]={{1,0,2},{31,64,1},{33,1536,2},{126,127,2},{127,512,1},{128,0,2},{294,1536,2},{512,512,2}};
    int ok=1;for(size_t i=0;i<sizeof(cases)/sizeof(*cases)&&ok;i++)ok=numeric(cases[i][0],cases[i][1],cases[i][2]);
    if(ok)ok=unavailable(NULL)&&unavailable("1")&&auto_fallback(0)&&auto_fallback(1)&&mixed_carry();
    myds_v41_gpu_inject_attention_matrix_failure(0);myds_v41_gpu_inject_attention_vec_head_first_failure(false);
    myds_metal_cleanup();printf("ATTENTION VEC %s\n",ok?"PASS":"FAIL");return ok?0:1;
}
