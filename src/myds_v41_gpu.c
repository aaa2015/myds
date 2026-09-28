#include "myds_v41_gpu.h"
#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct myds_v41_gpu_scratch {
    uint32_t max_comp;
    myds_metal_buffer_t *sort_a, *sort_b, *kv_f16, *mask, *pad, *partial;
};
static int floats(const myds_metal_buffer_t *b,uint64_t n) {
    return b && n && n<=myds_metal_buffer_size(b)/4 && myds_metal_buffer_offset(b)%4==0;
}
static int dispatch(const char *name,const void *args,size_t bytes,
    const myds_metal_buffer_t *const *b,size_t n,uint32_t x,uint32_t y,uint32_t z,
    uint32_t tx,uint32_t ty,uint32_t tz,size_t smem) {
    const uint32_t groups[]={x,y,z},threads[]={tx,ty,tz};
    return myds_metal_dispatch_kernel(name,args,bytes,b,n,groups,threads,smem);
}
static int specialized(const char *name,const void *args,size_t bytes,
    const myds_metal_buffer_t *const *b,size_t n,uint32_t x,uint32_t y,uint32_t z,
    uint32_t tx,uint32_t ty,uint32_t tz,size_t smem,
    const myds_metal_constant_t *constants,size_t nc) {
    const uint32_t groups[]={x,y,z},threads[]={tx,ty,tz};
    return myds_metal_dispatch_kernel_constants(name,args,bytes,b,n,groups,threads,smem,constants,nc);
}
myds_v41_gpu_scratch_t *myds_v41_gpu_scratch_create(uint32_t max_comp) {
    if(!max_comp || max_comp>1048576) return NULL;
    myds_v41_gpu_scratch_t *s=calloc(1,sizeof(*s));
    if(!s)return NULL;
    s->max_comp=max_comp;
    s->sort_a=myds_metal_buffer_alloc((size_t)max_comp*4);
    s->sort_b=myds_metal_buffer_alloc((size_t)max_comp*4);
    s->kv_f16=myds_metal_buffer_alloc(640u*512u*2u);
    s->mask=myds_metal_buffer_alloc(640u*2u);
    s->pad=myds_metal_buffer_alloc(2u*32u*512u*2u+32u*2u);
    s->partial=myds_metal_buffer_alloc(64u*(512u+2u)*32u*4u);
    if(!s->sort_a||!s->sort_b||!s->kv_f16||!s->mask||!s->pad||!s->partial){myds_v41_gpu_scratch_free(s);return NULL;}
    memset(myds_metal_buffer_data(s->mask),0,640u*2u);
    return s;
}
void myds_v41_gpu_scratch_free(myds_v41_gpu_scratch_t *s) {
    if(!s)return;
    myds_metal_buffer_free(s->sort_a);myds_metal_buffer_free(s->sort_b);
    myds_metal_buffer_free(s->kv_f16);myds_metal_buffer_free(s->mask);
    myds_metal_buffer_free(s->pad);myds_metal_buffer_free(s->partial);free(s);
}
static float rope_frequencies[2][32];
static pthread_once_t rope_once=PTHREAD_ONCE_INIT;
/* Match pinned ds4's powf/reciprocal and YaRN operation order. */
#pragma float_control(precise, on, push)
static void init_rope(void) {
    const double pi=3.14159265358979323846;
    for(int kind=0;kind<2;kind++) {
        float base=kind?160000.0f:10000.0f;
        float low=(float)floor(64.0*log(65536.0/(32.0*2.0*pi))/(2.0*log(base)));
        float high=(float)ceil(64.0*log(65536.0/(2.0*pi))/(2.0*log(base)));
        for(int i=0;i<32;i++) {
            float denominator=powf(base,(float)i/32.0f),f=1.0f/denominator;
            if(kind) {
                float ramp=fminf(1.0f,fmaxf(0.0f,(i-low)/(high-low)));
                float smooth=1.0f-ramp;
                float interpolate=(f/16.0f)*(1.0f-smooth),extrapolate=f*smooth;
                f=interpolate+extrapolate;
            }
            rope_frequencies[kind][i]=f;
        }
    }
}
#pragma float_control(pop)
int myds_v41_gpu_rope(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool compressed,bool inverse) {
    if(width<64||!heads||pos>=1048576||!floats(x,(uint64_t)width*heads))return 0;
    pthread_once(&rope_once,init_rope);
    struct {uint32_t width,heads,rows,start,inverse,stride;float frequencies[32];} a={width,heads,1,pos,inverse,1,{0}};
    memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={x};
    return dispatch("kernel_dsv41_rope",&a,sizeof(a),b,1,heads,1,1,32,1,1,0);
}
int myds_v41_gpu_quant(myds_metal_buffer_t *x,uint32_t width,uint32_t rows,uint32_t mode) {
    uint32_t block=mode==3?16:32;
    if(!width||!rows||mode>3||(mode&&width%block)||!floats(x,(uint64_t)width*rows))return 0;
    if(!mode && (uint64_t)width*rows<=UINT32_MAX && myds_metal_buffer_offset(x)%16==0)
        return myds_metal_v41_bf16(x,width*rows);
    const uint32_t a[]={width,rows,mode};const myds_metal_buffer_t *b[]={x};
    return dispatch("kernel_dsv41_quantize",a,sizeof(a),b,1,(width+block-1)/block,rows,1,32,1,1,0);
}
int myds_v41_gpu_pool2(myds_metal_buffer_t *out,const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores,myds_metal_buffer_t *previous_kv,
    myds_metal_buffer_t *previous_scores,uint32_t width,uint32_t pos) {
    if(!width||pos==UINT32_MAX||!floats(kv,width)||!floats(scores,width)||
        !floats(previous_kv,width)||!floats(previous_scores,width))return 0;
    if(!(pos&1))return myds_metal_buffer_copy(previous_kv,0,kv,0,(size_t)width*4)&&
        myds_metal_buffer_copy(previous_scores,0,scores,0,(size_t)width*4);
    if(!floats(out,width))return 0;
    const uint32_t a[]={width,1,1};const myds_metal_buffer_t *b[]={out,kv,scores,previous_kv,previous_scores};
    return dispatch("kernel_dsv41_pool2",a,sizeof(a),b,5,(width+255u)/256,1,1,256,1,1,0);
}
int myds_v41_gpu_indexscores(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,uint32_t n_heads) {
    if(!n_heads||!floats(scores,n_comp)||!floats(q,(uint64_t)n_heads*128)||
        !floats(weights,n_heads)||!floats(cache,(uint64_t)n_comp*128))return 0;
    struct {uint32_t n_rows,n_head,head_dim,cache_f16;float scale;} a={n_comp,n_heads,128,0,1.0f/64.0f};
    const myds_metal_buffer_t *b[]={q,weights,cache,scores};
    return n_heads==32 ? dispatch("kernel_glm_indexer_score_one_direct",&a,sizeof(a),b,4,n_comp,1,1,32,4,1,132u*4)
        : dispatch("kernel_glm_indexer_score_one",&a,sizeof(a),b,4,n_comp,1,1,32,1,1,32u*4);
}
static int candidates(myds_metal_buffer_t *out,const myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *mask,uint32_t n_comp,uint32_t pos,uint32_t ratio) {
    if(!n_comp||n_comp>UINT32_MAX-7||!ratio||pos==UINT32_MAX)return 0;
    uint32_t blocks=(n_comp+7)/8,width=mask?n_comp:blocks;
    if(!floats(out,width)||!floats(scores,n_comp)||(mask&&!floats(mask,blocks)))return 0;
    const uint32_t a[]={n_comp,1,pos,ratio};const myds_metal_buffer_t *b[]={scores,out,mask?mask:scores};
    return dispatch(mask?"kernel_dsv41_candidate_filter":"kernel_dsv41_candidate_blocks",
        a,sizeof(a),b,3,(width+255u)/256,1,1,256,1,1,0);
}
int myds_v41_gpu_candidates(myds_metal_buffer_t *blocks,const myds_metal_buffer_t *scores,
    uint32_t n_comp,uint32_t pos,uint32_t ratio) {return candidates(blocks,scores,NULL,n_comp,pos,ratio);}
int myds_v41_gpu_filter(myds_metal_buffer_t *scores,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t pos,uint32_t ratio) {return mask&&candidates(scores,scores,mask,n_comp,pos,ratio);}

struct sort_args {
    int32_t ne00,ne01,ne02,ne03;uint64_t nb00,nb01,nb02,nb03;
    int32_t ne0,ne1,ne2,ne3,top_k;uint32_t causal_start,causal_ratio;
};
struct merge_args {
    int64_t ne00,ne01,ne02,ne03;uint64_t nb00,nb01,nb02,nb03;
    int32_t ne0,ne1,ne2,ne3,top_k,len;uint32_t causal_start,causal_ratio,block_width,block_top_k;
};
int myds_v41_gpu_topk(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,myds_v41_gpu_scratch_t *s) {
    if(!s||n_cols>s->max_comp||n_cols>INT32_MAX||!top||top>n_cols||!floats(ids,top)||!floats(scores,n_cols))return 0;
    /* Match the reference bitonic partition as well as its comparator: ties
     * can otherwise select different cache rows after the merge. */
    uint32_t max_threads=myds_metal_kernel_max_threads("kernel_argsort_f32_i32_desc");
    if(!max_threads)return 0;
    uint32_t nth=1;while(nth<n_cols&&2u*nth<=max_threads)nth*=2;
    uint32_t npr=(n_cols+nth-1)/nth,blocktop=top<nth?top:nth;
    uint32_t last=n_cols-(npr-1)*nth;
    uint32_t work=npr==1?top:(npr-1)*blocktop+(last<blocktop?last:blocktop);
    struct sort_args a={.ne00=n_cols,.ne01=1,.ne02=1,.ne03=1,.nb00=4,
        .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*4,.nb03=(uint64_t)n_cols*4,
        .ne0=work,.ne1=1,.ne2=1,.ne3=1,.top_k=blocktop};
    const myds_metal_buffer_t *b[]={scores,npr==1?ids:s->sort_a};
    if(!dispatch("kernel_argsort_f32_i32_desc",&a,sizeof(a),b,2,npr,1,1,nth,1,1,((size_t)nth*8+15)&~(size_t)15))return 0;
    myds_metal_buffer_t *cur=s->sort_a,*next=s->sort_b;
    for(uint32_t len=blocktop;len<work;len*=2) {
        uint32_t nm=(work+2*len-1)/(2*len);
        struct merge_args m={.ne00=n_cols,.ne01=1,.ne02=1,.ne03=1,.nb00=4,
            .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*4,.nb03=(uint64_t)n_cols*4,
            .ne0=work,.ne1=1,.ne2=1,.ne3=1,.top_k=nm==1?top:work,.len=len,.block_width=nth,.block_top_k=blocktop};
        const myds_metal_buffer_t *mb[]={scores,cur,nm==1?ids:next};
        if(!dispatch("kernel_argsort_merge_f32_i32_desc",&m,sizeof(m),mb,3,nm,1,1,len<512?len:512,1,1,0))return 0;
        myds_metal_buffer_t *tmp=cur;cur=next;next=tmp;
    }
    return 1;
}
int myds_v41_gpu_topkmask(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top) {
    if(!top||top>n_cols||!floats(mask,n_cols)||!floats(ids,top))return 0;
    struct {int64_t ne00,ne01;uint64_t nb00,nb01;int64_t ne0,ne1;uint64_t nb0,nb1;} a={top,1,4,(uint64_t)top*4,n_cols,1,4,(uint64_t)n_cols*4};
    const myds_metal_buffer_t *b[]={ids,mask};
    return dispatch("kernel_dsv4_topk_mask",&a,sizeof(a),b,2,(n_cols+255u)/256,1,1,256,1,1,0)&&
        dispatch("kernel_dsv4_topk_mask_scatter",&a,sizeof(a),b,2,(top+255u)/256,1,1,256,1,1,0);
}
int myds_v41_gpu_gather(myds_metal_buffer_t *out,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *ids,uint32_t source_rows,uint32_t selected_rows) {
    if(!selected_rows||selected_rows>512||selected_rows>source_rows||
        !floats(out,(uint64_t)selected_rows*512)||!floats(cache,(uint64_t)source_rows*512)||!floats(ids,selected_rows))return 0;
    struct {int32_t ne00t,ne00;uint64_t nb01,nb02,nb03;int32_t ne10;uint64_t nb10,nb11,nb12,nb1,nb2,nb3;} a={
        .ne00t=512,.ne00=512,.nb01=2048,.nb02=(uint64_t)source_rows*2048,.nb03=(uint64_t)source_rows*2048,
        .ne10=selected_rows,.nb10=4,.nb11=(uint64_t)selected_rows*4,.nb12=(uint64_t)selected_rows*4,
        .nb1=2048,.nb2=(uint64_t)selected_rows*2048,.nb3=(uint64_t)selected_rows*2048};
    const myds_metal_buffer_t *b[]={cache,ids,out};
    return dispatch("kernel_get_rows_f32",&a,sizeof(a),b,3,2*selected_rows,1,1,256,1,1,0);
}

/* Use transient subviews only as binding handles; Metal retains the underlying
 * resources until queued dispatches finish. No CPU readback or synchronization. */
static int stage_f16(myds_metal_buffer_t *dst,size_t dstoff,const myds_metal_buffer_t *src,size_t srcoff,uint32_t count) {
    myds_metal_buffer_t *sv=myds_metal_buffer_sub_view(src,srcoff,(size_t)count*4);
    myds_metal_buffer_t *dv=myds_metal_buffer_sub_view(dst,dstoff,(size_t)count*2);
    if(!sv||!dv){myds_metal_buffer_free(sv);myds_metal_buffer_free(dv);return 0;}
    const myds_metal_buffer_t *b[]={sv,dv};
    int ok=dispatch("kernel_cpy_contig_f32_f16_4",&count,sizeof(count),b,2,(count+1023u)/1024,1,1,256,1,1,0);
    myds_metal_buffer_free(sv);myds_metal_buffer_free(dv);return ok;
}
struct flash_pad_args {
    int32_t ne11,ne_12_2,ne_12_3;uint64_t nb11,nb12,nb13,nb21,nb22,nb23;
    int32_t ne31,ne32,ne33;uint64_t nb31,nb32,nb33;
};
struct flash_vec_args {
    int32_t ne01,ne02,ne03;uint64_t nb01,nb02,nb03;int32_t ne11,ne_12_2,ne_12_3,ns10;
    uint64_t nb11,nb12,nb13;int32_t ns20;uint64_t nb21,nb22,nb23;
    int32_t ne31,ne32,ne33;uint64_t nb31,nb32,nb33;int32_t ne1,ne2,ne3;
    float scale,max_bias,m0,m1;int32_t n_head_log2;float logit_softcap;
};
int myds_v41_gpu_attention(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,myds_v41_gpu_scratch_t *s) {
    if(!s||!n_raw||n_raw>128||raw_start>=128||selected_rows>512||
        !floats(heads,64u*512)||!floats(q,64u*512)||!floats(sinks,64)||!floats(window,128u*512)||
        (selected_rows&&!floats(selected_kv,(uint64_t)selected_rows*512)))return 0;
    uint32_t tail=128-raw_start<n_raw?128-raw_start:n_raw;
    if(!stage_f16(s->kv_f16,0,window,(size_t)raw_start*2048,tail*512))return 0;
    if(n_raw>tail&&!stage_f16(s->kv_f16,(size_t)tail*1024,window,0,(n_raw-tail)*512))return 0;
    if(selected_rows&&!stage_f16(s->kv_f16,(size_t)n_raw*1024,selected_kv,0,selected_rows*512))return 0;
    uint32_t keys=n_raw+selected_rows;uint64_t kvbytes=(uint64_t)keys*1024,maskbytes=(uint64_t)keys*2;
    if(keys%32) {
        struct flash_pad_args a={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
            .nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,.nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes};
        const myds_metal_buffer_t *b[]={s->kv_f16,s->kv_f16,s->mask,s->pad};
        const myds_metal_constant_t c[]={{100,true,1},{125,false,32}};
        if(!specialized("kernel_flash_attn_ext_pad",&a,sizeof(a),b,4,32,1,1,32,1,1,0,c,2))return 0;
    }
    struct flash_vec_args a={.ne01=1,.ne02=64,.ne03=1,.nb01=64u*2048,.nb02=2048,.nb03=64u*2048,
        .ne11=keys,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
        .ns20=512,.nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,
        .nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes,.ne1=64,.ne2=1,.ne3=1,.scale=1.0f/sqrtf(512)};
    const myds_metal_buffer_t *b[]={q,s->kv_f16,s->kv_f16,s->mask,sinks,s->pad,s->partial};
    const myds_metal_constant_t c[]={{400,true,1},{401,true,1},{402,true,0},{403,true,0},{404,true,keys%32!=0},
        {405,true,0},{420,false,512},{421,false,512},{422,false,1},{423,false,32}};
    if(!specialized("kernel_flash_attn_ext_vec_f16_dk512_dv512",&a,sizeof(a),b,7,1,64,32,32,1,1,3328,c,10))return 0;
    const int32_t nrows=64;const myds_metal_buffer_t *rb[]={s->partial,heads};
    const myds_metal_constant_t rc[]={{500,false,512},{501,false,32}};
    return specialized("kernel_flash_attn_ext_vec_reduce",&nrows,sizeof(nrows),rb,2,64,1,1,1024,1,1,0,rc,2);
}
