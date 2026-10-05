#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);return 0;}}while(0)
struct fixture{uint32_t rows,start,ratio,ncomp;myds_metal_buffer_t *q,*kv,*comp,*packed,*ids,*sinks,*gather,*window,*out[3];};
static float sample(uint32_t i,uint32_t seed){uint32_t x=(i+1)*747796405u+seed*2891336453u;x=((x>>((x>>28)+4))^x)*277803737u;x=(x>>22)^x;float f=((int32_t)(x%8191)-4095)/8192.0f;uint32_t b;memcpy(&b,&f,4);b=(b+0x7fff+((b>>16)&1))&0xffff0000u;memcpy(&f,&b,4);return f;}
static uint32_t gcd(uint32_t a,uint32_t b){while(b){uint32_t t=a%b;a=b;b=t;}return a;}
static int init(struct fixture *f,uint32_t rows,uint32_t start,uint32_t ratio){memset(f,0,sizeof(*f));f->rows=rows;f->start=start;f->ratio=ratio;f->ncomp=(start+rows)/ratio;if(!f->ncomp)f->ncomp=1;
#define ALLOC(k,n) do{f->k=myds_metal_buffer_alloc(n);CHECK(f->k);}while(0)
    ALLOC(q,(size_t)rows*64*512*4);ALLOC(kv,(size_t)rows*512*4);ALLOC(comp,(size_t)f->ncomp*512*4);ALLOC(packed,(size_t)f->ncomp*512*2);ALLOC(ids,(size_t)rows*512*4);ALLOC(sinks,256);ALLOC(gather,(size_t)rows*640*512*2);ALLOC(window,128u*512*4);
#undef ALLOC
    for(int j=0;j<3;j++){f->out[j]=myds_metal_buffer_alloc((size_t)rows*64*512*4);CHECK(f->out[j]);}
    myds_metal_buffer_t *bs[]={f->q,f->kv,f->comp,f->window};for(int k=0;k<4;k++){float *p=myds_metal_buffer_data(bs[k]);for(size_t j=0;j<myds_metal_buffer_size(bs[k])/4;j++)p[j]=sample((uint32_t)j,k+1);}
    float *sink=myds_metal_buffer_data(f->sinks);for(int j=0;j<64;j++)sink[j]=(j-31)/16.0f;
    int32_t *ids=myds_metal_buffer_data(f->ids);for(uint32_t r=0;r<rows;r++){uint32_t count=(start+r+1)/ratio,step=count>1?count/2+1:1;while(gcd(step,count)!=1)step++;for(uint32_t j=0;j<512;j++)ids[(size_t)r*512+j]=j<(count<512?count:512)?(int32_t)(((uint64_t)j*step+r*17u)%count):-1;}
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_pack_bf16(f->packed,0,f->comp,0,f->ncomp*512));CHECK(myds_metal_commit_batch());return 1;
}
static void cleanup(struct fixture *f){myds_metal_buffer_t *b[]={f->q,f->kv,f->comp,f->packed,f->ids,f->sinks,f->gather,f->window,f->out[0],f->out[1],f->out[2]};for(size_t i=0;i<sizeof(b)/sizeof(*b);i++)myds_metal_buffer_free(b[i]);}
static myds_v41_gpu_scratch_t *scratch(const char *name){setenv("MYDS_PREFILL_ATTN",name,1);return myds_v41_gpu_scratch_create(8192);}
static int call(struct fixture *f,myds_v41_gpu_scratch_t *s,int out,int packed,int gather){return packed?myds_v41_gpu_attention_chunk_bf16(f->out[out],f->sinks,f->q,f->kv,f->packed,f->ids,gather?f->gather:NULL,f->rows,f->ncomp,512,f->start,0,f->ratio,s):myds_v41_gpu_attention_chunk(f->out[out],f->sinks,f->q,f->kv,f->comp,f->ids,gather?f->gather:NULL,f->rows,f->ncomp,512,f->start,0,f->ratio,s);}
static int compare(struct fixture *f,int candidate,int exact){const float *a=myds_metal_buffer_data(f->out[0]),*b=myds_metal_buffer_data(f->out[candidate]);double max=0;uint64_t diff=0;for(size_t i=0;i<(size_t)f->rows*64*512;i++){CHECK(isfinite(a[i])&&isfinite(b[i]));max=fmax(max,fabs((double)a[i]-b[i]));uint32_t x,y;memcpy(&x,a+i,4);memcpy(&y,b+i,4);diff+=x!=y;}CHECK(exact?diff==0:max<=3e-5);printf("PASS rows=%u start=%u ratio=%u %s maxabs=%.9g different=%llu\n",f->rows,f->start,f->ratio,exact?"direct exact":"matrix tolerance",max,(unsigned long long)diff);return 1;}
static int policy(uint32_t rows,uint32_t start,uint32_t ratio,int expected_matrix){
    struct fixture f;CHECK(init(&f,rows,start,ratio));
    myds_v41_gpu_scratch_t *s[]={scratch("vec"),scratch("auto"),scratch("matrix")};CHECK(s[0]&&s[1]&&s[2]);
    CHECK(myds_metal_begin_batch());for(int k=0;k<3;k++){if(start)CHECK(myds_v41_gpu_carry_stage(s[k],f.window,start));CHECK(call(&f,s[k],k,1,1));}CHECK(myds_metal_commit_batch());
    CHECK(compare(&f,2,0));CHECK(!memcmp(myds_metal_buffer_data(f.out[1]),myds_metal_buffer_data(f.out[expected_matrix?2:0]),(size_t)rows*64*512*4));
    printf("PASS auto policy rows=%u start=%u ratio=%u n_comp=%u chosen=%s byte exact\n",rows,start,ratio,f.ncomp,expected_matrix?"matrix":"vec");
    // A supplied gathered buffer with zero compressed keys still uses window-only vec.
    CHECK(myds_metal_begin_batch());for(int k=0;k<3;k++){if(start)CHECK(myds_v41_gpu_carry_stage(s[k],f.window,start));CHECK(myds_v41_gpu_attention_chunk(f.out[k],f.sinks,f.q,f.kv,NULL,NULL,f.gather,rows,0,512,start,0,0,s[k]));}CHECK(myds_metal_commit_batch());
    for(int k=1;k<3;k++)CHECK(!memcmp(myds_metal_buffer_data(f.out[0]),myds_metal_buffer_data(f.out[k]),(size_t)rows*64*512*4));
    // Slab fallback always uses the original vector path under all policies.
    CHECK(myds_metal_begin_batch());for(int k=0;k<3;k++){if(start)CHECK(myds_v41_gpu_carry_stage(s[k],f.window,start));CHECK(call(&f,s[k],k,1,0));}CHECK(myds_metal_commit_batch());
    for(int k=1;k<3;k++)CHECK(!memcmp(myds_metal_buffer_data(f.out[0]),myds_metal_buffer_data(f.out[k]),(size_t)rows*64*512*4));
    for(int k=0;k<3;k++)myds_v41_gpu_scratch_free(s[k]);cleanup(&f);return 1;
}
static int unavailable(uint32_t kind){
    struct fixture f;CHECK(init(&f,128,1024,2));
    myds_v41_gpu_scratch_t *vec=scratch("vec"),*auto_s=scratch("auto"),*explicit_s=scratch("matrix");CHECK(vec&&auto_s&&explicit_s);
    myds_v41_gpu_inject_attention_matrix_failure(kind);
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_carry_stage(vec,f.window,f.start));CHECK(call(&f,vec,0,1,1));
    CHECK(myds_v41_gpu_carry_stage(auto_s,f.window,f.start));uint64_t before=myds_metal_dispatch_count();CHECK(call(&f,auto_s,1,1,1));uint64_t first=myds_metal_dispatch_count()-before;
    CHECK(myds_metal_commit_batch());CHECK(!memcmp(myds_metal_buffer_data(f.out[0]),myds_metal_buffer_data(f.out[1]),(size_t)f.rows*64*512*4));
    // First failed head PSO follows an encoded scanner; cached retry uses vec only.
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_carry_stage(auto_s,f.window,f.start));before=myds_metal_dispatch_count();CHECK(call(&f,auto_s,1,1,1));uint64_t retry=myds_metal_dispatch_count()-before;CHECK(myds_metal_commit_batch());CHECK(first==retry+1);
    CHECK(!memcmp(myds_metal_buffer_data(f.out[0]),myds_metal_buffer_data(f.out[1]),(size_t)f.rows*64*512*4));
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_carry_stage(explicit_s,f.window,f.start));before=myds_metal_dispatch_count();CHECK(!call(&f,explicit_s,2,1,1));uint64_t strict=myds_metal_dispatch_count()-before;CHECK(myds_metal_commit_batch());
    CHECK(first==strict+3); // Only fallback vec and two carry copies distinguish auto.
    printf("PASS auto unavailable kind=%u first=%llu cached=%llu explicit_fail=%llu vec byte exact\n",kind,(unsigned long long)first,(unsigned long long)retry,(unsigned long long)strict);
    // Explicit matrix failure must remain retryable after the cause is removed.
    // Re-stage carry because a failed chunk deliberately does not publish its tail.
    myds_v41_gpu_inject_attention_matrix_failure(0);
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_carry_stage(explicit_s,f.window,f.start));
    before=myds_metal_dispatch_count();CHECK(call(&f,explicit_s,2,1,1));
    uint64_t recovered=myds_metal_dispatch_count()-before;CHECK(myds_metal_commit_batch());
    CHECK(recovered==strict+3);CHECK(compare(&f,2,0));
    printf("PASS explicit matrix same-scratch retry kind=%u recovered=%llu\n",kind,(unsigned long long)recovered);
    myds_v41_gpu_scratch_free(vec);myds_v41_gpu_scratch_free(auto_s);myds_v41_gpu_scratch_free(explicit_s);cleanup(&f);return 1;
}
int main(void){
    printf("ATTENTION POLICY compiled=%s %s\n",__DATE__,__TIME__);fflush(stdout);
    if(!myds_metal_init("metal"))return 1;setenv("MYDS_ATTN_GATHER","1",1);unsetenv("MYDS_ATTN_GATHER_MIN_KEYS");
    const uint32_t cases[][4]={{512,0,2,0},{512,512,2,1},{294,1536,2,1},{512,0,1,0},{256,256,1,1},{31,1024,2,0},{32,1024,2,1},{32,512,2,0},{512,510,2,0}};
    int ok=1;for(size_t i=0;i<sizeof(cases)/sizeof(*cases)&&ok;i++)ok=policy(cases[i][0],cases[i][1],cases[i][2],cases[i][3]);
    if(ok)ok=unavailable(1)&&unavailable(2);
    setenv("MYDS_PREFILL_ATTN","direct",1);myds_v41_gpu_scratch_t *bad=myds_v41_gpu_scratch_create(8192);if(bad){myds_v41_gpu_scratch_free(bad);ok=0;}
    myds_metal_cleanup();printf("ATTENTION POLICY %s\n",ok?"PASS":"FAIL");return ok?0:1;
}
