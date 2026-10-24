/* Root schedules execution; this test never loads a full model. */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
#include "myds_v41_decode.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#c);return 0;}}while(0)
static int allow_fallback;
struct guarded {myds_metal_buffer_t *owner,*view;size_t bytes;};
static int guarded_create(struct guarded *b,size_t bytes) {
    b->bytes=bytes;b->owner=myds_metal_buffer_alloc(bytes+32);CHECK(b->owner);
    memset(myds_metal_buffer_data(b->owner),0xa5,bytes+32);
    b->view=myds_metal_buffer_sub_view(b->owner,16,bytes);CHECK(b->view);return 1;
}
static int guarded_ok(struct guarded *b) {
    const uint8_t *p=myds_metal_buffer_data(b->owner);
    for(size_t i=0;i<16;i++)CHECK(p[i]==0xa5&&p[16+b->bytes+i]==0xa5);
    return 1;
}
static void guarded_free(struct guarded *b){myds_metal_buffer_free(b->view);myds_metal_buffer_free(b->owner);}
static void fill(myds_metal_buffer_t *b,unsigned seed) {
    float *p=myds_metal_buffer_data(b);
    for(size_t i=0;i<myds_metal_buffer_size(b)/4;i++)p[i]=((int)((i*19u+seed)%257u)-128)/512.0f;
}
static void weights(myds_metal_buffer_t *b) {
    uint8_t *p=myds_metal_buffer_data(b);
    for(size_t i=0;i<myds_metal_buffer_size(b)/34;i++) {
        uint16_t d=(uint16_t)(0x2400u+(i%4)*0x100u);memcpy(p+i*34,&d,2);
        for(size_t j=0;j<32;j++)p[i*34+2+j]=(uint8_t)(int8_t)((int)((i*13+j*29)%255)-127);
    }
}
static int exact(myds_metal_buffer_t *a,myds_metal_buffer_t *b,size_t count) {
    const float *x=myds_metal_buffer_data(a),*y=myds_metal_buffer_data(b);size_t nonzero=0;
    for(size_t i=0;i<count;i++) {
        if(memcmp(x+i,y+i,4)){uint32_t u,v;memcpy(&u,x+i,4);memcpy(&v,y+i,4);fprintf(stderr,"mismatch i=%zu old=%08x new=%08x\n",i,u,v);return 0;}
        CHECK(isfinite(x[i]));nonzero+=x[i]!=0;
    }
    CHECK(nonzero>0);return 1;
}
static int test_gemv(uint32_t rows,uint32_t width,int batch) {
    struct guarded w,x,a,b;CHECK(guarded_create(&w,(size_t)rows*width/32*34));
    CHECK(guarded_create(&x,(size_t)width*4));CHECK(guarded_create(&a,(size_t)rows*4));CHECK(guarded_create(&b,(size_t)rows*4));
    weights(w.view);fill(x.view,17);
    if(batch)CHECK(myds_metal_begin_batch());
    myds_metal_dispatch_count_reset();myds_metal_barrier();CHECK(myds_metal_gemv(a.view,w.view,x.view,rows,width,GGML_TYPE_Q8_0));CHECK(myds_metal_v41_bf16(a.view,rows));CHECK(myds_metal_dispatch_count()==2);
    myds_metal_dispatch_count_reset();myds_metal_barrier();CHECK(myds_metal_v41_gemv_bf16(b.view,w.view,x.view,rows,width,GGML_TYPE_Q8_0));
    CHECK(allow_fallback||myds_metal_dispatch_count()==1);
    if(batch)CHECK(myds_metal_commit_batch());
    CHECK(exact(a.view,b.view,rows));CHECK(guarded_ok(&w)&&guarded_ok(&x)&&guarded_ok(&a)&&guarded_ok(&b));
    myds_metal_buffer_t *short_out=myds_metal_buffer_sub_view(b.view,0,(size_t)(rows-1)*4);
    if(rows>1)CHECK(!myds_metal_v41_gemv_bf16(short_out,w.view,x.view,rows,width,GGML_TYPE_Q8_0));
    CHECK(!myds_metal_v41_gemv_bf16(b.view,w.view,x.view,rows,width-1,GGML_TYPE_Q8_0));
    CHECK(!myds_metal_v41_gemv_bf16(NULL,w.view,x.view,rows,width,GGML_TYPE_Q8_0));
    CHECK(!myds_metal_v41_gemv_bf16(b.view,w.view,x.view,0,width,GGML_TYPE_Q8_0));
    CHECK(!myds_metal_v41_gemv_bf16(b.view,w.view,x.view,UINT32_MAX,width,GGML_TYPE_Q8_0));
    myds_metal_buffer_free(short_out);guarded_free(&w);guarded_free(&x);guarded_free(&a);guarded_free(&b);
    printf("PASS Q8 BF16-store exact rows=%u width=%u batch=%d\n",rows,width,batch);return 1;
}
static int test_float_fallback(ggml_type type) {
    const uint32_t rows=4,width=128;size_t elem=type==GGML_TYPE_F16?2:4;
    myds_metal_buffer_t *w=myds_metal_buffer_alloc(rows*width*elem),*x=myds_metal_buffer_alloc(width*4),
        *a=myds_metal_buffer_alloc(rows*4),*b=myds_metal_buffer_alloc(rows*4);CHECK(w&&x&&a&&b);
    if(type==GGML_TYPE_F16) {uint16_t *p=myds_metal_buffer_data(w);for(size_t i=0;i<rows*width;i++)p[i]=(uint16_t)(0x3000u+(i%17));}
    else fill(w,1);fill(x,19);
    CHECK(myds_metal_begin_batch());myds_metal_barrier();CHECK(myds_metal_v41_gemv(a,w,x,rows,width,type));CHECK(myds_metal_v41_bf16(a,rows));
    myds_metal_barrier();CHECK(myds_metal_v41_gemv_bf16(b,w,x,rows,width,type));CHECK(myds_metal_commit_batch());CHECK(exact(a,b,rows));
    myds_metal_buffer_free(w);myds_metal_buffer_free(x);myds_metal_buffer_free(a);myds_metal_buffer_free(b);
    printf("PASS original F16/F32 generic projection fallback type=%d\n",type);return 1;
}
static int original_finish(myds_metal_buffer_t *q,myds_metal_buffer_t *k,myds_metal_buffer_t *w,uint32_t pos,int compressed) {
    return myds_v41_gpu_rope(q,512,64,pos,compressed,false)&&myds_v41_gpu_rope(k,512,1,pos,compressed,false)&&
        myds_v41_gpu_quant(k,512,1,1)&&myds_metal_buffer_copy(w,(size_t)(pos%128)*512*4,k,0,512*4);
}
static int test_finish(uint32_t pos,int compressed,int batch,int edge) {
    struct guarded q0,k0,w0,q1,k1,w1;CHECK(guarded_create(&q0,64*512*4));CHECK(guarded_create(&q1,64*512*4));
    CHECK(guarded_create(&k0,512*4));CHECK(guarded_create(&k1,512*4));CHECK(guarded_create(&w0,128*512*4));CHECK(guarded_create(&w1,128*512*4));
    fill(q0.view,11);fill(k0.view,29);fill(w0.view,41);
    if(edge) {
        const uint32_t bits[]={0x00000000u,0x80000000u,0x00000001u,0x80000001u,0x007fffffu,0x00800000u,
            0x3f808000u,0x3f818000u,0x3affffffu,0x3b000000u,0x43dfffffu,0xc3dfffffu};
        uint32_t *k=myds_metal_buffer_data(k0.view),*q=myds_metal_buffer_data(q0.view);
        for(unsigned i=0;i<512;i++)k[i]=bits[i%(sizeof(bits)/sizeof(*bits))];
        for(unsigned h=0;h<64;h++)for(unsigned i=0;i<64;i++)q[h*512+448+i]=bits[i%(sizeof(bits)/sizeof(*bits))];
    }
    memcpy(myds_metal_buffer_data(q1.view),myds_metal_buffer_data(q0.view),q0.bytes);
    memcpy(myds_metal_buffer_data(k1.view),myds_metal_buffer_data(k0.view),k0.bytes);
    memcpy(myds_metal_buffer_data(w1.view),myds_metal_buffer_data(w0.view),w0.bytes);
    if(batch)CHECK(myds_metal_begin_batch());CHECK(original_finish(q0.view,k0.view,w0.view,pos,compressed));
    myds_metal_dispatch_count_reset();CHECK(myds_v41_gpu_qkv_finish(q1.view,k1.view,w1.view,pos,compressed));CHECK(allow_fallback||myds_metal_dispatch_count()==1);if(batch)CHECK(myds_metal_commit_batch());
    CHECK(exact(q0.view,q1.view,64*512));CHECK(exact(k0.view,k1.view,512));CHECK(exact(w0.view,w1.view,128*512));
    CHECK(guarded_ok(&q0)&&guarded_ok(&k0)&&guarded_ok(&w0)&&guarded_ok(&q1)&&guarded_ok(&k1)&&guarded_ok(&w1));
    CHECK(!myds_v41_gpu_qkv_finish(q1.view,k1.view,w1.view,1048576,compressed));
    CHECK(!myds_v41_gpu_qkv_finish(NULL,k1.view,w1.view,pos,compressed));
    myds_metal_buffer_t *small=myds_metal_buffer_sub_view(w1.view,0,w1.bytes-4);CHECK(small);
    CHECK(!myds_v41_gpu_qkv_finish(q1.view,k1.view,small,pos,compressed));myds_metal_buffer_free(small);
    /* The fusion capability must respect the existing injected-copy contract. */
    myds_metal_inject_copy_failure(1);CHECK(!myds_metal_v41_qkv_fusion_available());
    CHECK(myds_metal_begin_batch());CHECK(!myds_v41_gpu_qkv_finish(q1.view,k1.view,w1.view,pos,compressed));CHECK(myds_metal_commit_batch());
    myds_metal_inject_copy_failure(0);
    guarded_free(&q0);guarded_free(&q1);guarded_free(&k0);guarded_free(&k1);guarded_free(&w0);guarded_free(&w1);
    printf("PASS finish exact pos=%u compressed=%d batch=%d edge=%d incl copy-failure fallback\n",pos,compressed,batch,edge);return 1;
}

static void weights_f16(myds_metal_buffer_t *b) {
    uint16_t *p=myds_metal_buffer_data(b);
    for(size_t i=0;i<myds_metal_buffer_size(b)/2;i++)
        p[i]=(uint16_t)(0x2000u+(i%1024u)+((i%3u)==0?0x8000u:0u));
}

/* Nonzero, actual projection shapes; this compares component outputs, not
 * whole-model logits or performance. The two arms call the runtime scope. */
static int test_qkv_scope_graph(uint32_t pos,int finish_fallback) {
    enum {XN,XS,QR,QRN,KV,Q,WINDOW,FLAT,MIX,SPLIT,IQ,IW,SCORES,SELECT,NOUT};
    const uint32_t dim=5120,count=641;
    const size_t sizes[NOUT]={dim,dim,1280,1280,512,64*512,128*512,4*dim,24,24,32*128,32,count,512};
    struct guarded out[2][NOUT], qa,qb,kv,hc,iq,iw,res,pre,scale,base,norm,qrnorm,kvnorm,cache,mask;
    CHECK(guarded_create(&qa,(size_t)1280*dim/32*34));
    CHECK(guarded_create(&qb,(size_t)32768*1280/32*34));
    CHECK(guarded_create(&kv,(size_t)512*dim/32*34));
    CHECK(guarded_create(&hc,(size_t)24*4*dim*2));
    CHECK(guarded_create(&iq,(size_t)4096*1280*2));
    CHECK(guarded_create(&iw,(size_t)32*dim*2));
    CHECK(guarded_create(&res,4*dim*4));CHECK(guarded_create(&pre,24*4));
    CHECK(guarded_create(&scale,3*4));CHECK(guarded_create(&base,24*4));
    CHECK(guarded_create(&norm,dim*4));CHECK(guarded_create(&qrnorm,1280*4));
    CHECK(guarded_create(&kvnorm,512*4));CHECK(guarded_create(&cache,count*128*2));
    CHECK(guarded_create(&mask,((count+7)/8)*4));
    weights(qa.view);weights(qb.view);weights(kv.view);
    weights_f16(hc.view);weights_f16(iq.view);weights_f16(iw.view);fill(res.view,51);fill(base.view,3);
    const float pre_values[4]={0.7f,0.1f,-0.05f,0.25f};
    memcpy(myds_metal_buffer_data(pre.view),pre_values,sizeof(pre_values));
    for(unsigned i=0;i<3;i++)((float*)myds_metal_buffer_data(scale.view))[i]=0.01f;
    struct guarded *norms[]={&norm,&qrnorm,&kvnorm};
    for(unsigned b=0;b<3;b++)for(size_t i=0;i<norms[b]->bytes/4;i++)
        ((float*)myds_metal_buffer_data(norms[b]->view))[i]=0.8f+(float)(i%17)/32.0f;
    uint16_t *cp=myds_metal_buffer_data(cache.view);
    for(size_t i=0;i<cache.bytes/2;i++)cp[i]=(uint16_t)(0x3d00u+(i%128u));
    float *mp=myds_metal_buffer_data(mask.view);
    for(size_t i=0;i<mask.bytes/4;i++)mp[i]=(i%3u)==0?-INFINITY:0.0f;
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(count);CHECK(scratch);
    for(unsigned arm=0;arm<2;arm++) {
        for(unsigned b=0;b<NOUT;b++)CHECK(guarded_create(&out[arm][b],sizes[b]*4));
        fill(out[arm][WINDOW].view,41);
        setenv("MYDS_QKV_CONCURRENT",arm?"1":"0",1);
        myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_v41_hc_sum_rmsnorm(out[arm][XN].view,out[arm][XS].view,res.view,pre.view,norm.view,dim,1e-6f,true));
        myds_v41_decode_qkv_scope_t scope={0};myds_v41_decode_qkv_scope_begin(&scope,false);
        CHECK(scope.active==(arm==1));CHECK(myds_metal_get_concurrent()==(int)arm);
        myds_metal_barrier_buffer(out[arm][XN].view);
        CHECK(myds_metal_v41_hc_mix(out[arm][SPLIT].view,res.view,hc.view,GGML_TYPE_F16,
            scale.view,base.view,out[arm][FLAT].view,out[arm][MIX].view,dim,1e-6f,1e-6f));
        CHECK(myds_metal_v41_gemv_bf16(out[arm][QR].view,qa.view,out[arm][XN].view,1280,dim,GGML_TYPE_Q8_0));
        CHECK(myds_metal_v41_gemv_bf16(out[arm][KV].view,kv.view,out[arm][XN].view,512,dim,GGML_TYPE_Q8_0));
        const myds_metal_buffer_t *proj_deps[]={out[arm][QR].view,out[arm][KV].view};
        myds_metal_barrier_buffers(proj_deps,2);
        CHECK(myds_metal_v41_rmsnorm(out[arm][QRN].view,out[arm][QR].view,qrnorm.view,1280,1e-6f,true));
        CHECK(myds_metal_v41_rmsnorm(out[arm][KV].view,out[arm][KV].view,kvnorm.view,512,1e-6f,true));
        const myds_metal_buffer_t *norm_deps[]={out[arm][QRN].view,out[arm][KV].view};
        myds_metal_barrier_buffers(norm_deps,2);
        CHECK(myds_metal_v41_gemv_bf16(out[arm][Q].view,qb.view,out[arm][QRN].view,32768,1280,GGML_TYPE_Q8_0));
        CHECK(myds_metal_v41_gemv(out[arm][IQ].view,iq.view,out[arm][QRN].view,4096,1280,GGML_TYPE_F16));
        CHECK(myds_metal_v41_gemv(out[arm][IW].view,iw.view,out[arm][XN].view,32,dim,GGML_TYPE_F16));
        const myds_metal_buffer_t *index_deps[]={out[arm][IQ].view,out[arm][IW].view};
        myds_metal_barrier_buffers(index_deps,2);
        CHECK(myds_metal_v41_bf16(out[arm][IQ].view,4096));
        CHECK(myds_metal_v41_bf16(out[arm][IW].view,32));
        myds_metal_barrier_buffer(out[arm][IQ].view);
        CHECK(myds_v41_gpu_rope(out[arm][IQ].view,128,32,pos,true,false));
        myds_metal_barrier_buffer(out[arm][IQ].view);
        CHECK(myds_v41_gpu_quant(out[arm][IQ].view,128,32,2));
        myds_metal_barrier_buffers(index_deps,2);
        /* 合成 fixture：count 固定 641，与 pos/ratio 无对应关系，两臂只需同一个
         * 窗口下界 —— 传 0（无下界），不假装有真实几何。 */
        CHECK(myds_v41_gpu_indexscores_masked_bf16(out[arm][SCORES].view,out[arm][IQ].view,
            out[arm][IW].view,cache.view,mask.view,count,32,0));
        myds_metal_barrier_buffer(out[arm][SCORES].view);
        CHECK(myds_v41_gpu_topk(out[arm][SELECT].view,out[arm][SCORES].view,count,512,scratch));
        myds_metal_barrier_buffer(out[arm][Q].view);
        /* Force the original finish path while allowing its first copy to
         * succeed: this exercises RoPE -> quant -> ring-store dependencies. */
        if(finish_fallback)myds_metal_inject_copy_failure(2);
        CHECK(myds_v41_gpu_qkv_finish(out[arm][Q].view,out[arm][KV].view,out[arm][WINDOW].view,pos,true));
        myds_metal_inject_copy_failure(0);
        myds_v41_decode_qkv_scope_end(&scope);
        CHECK(!scope.active&&myds_metal_get_concurrent()==0&&myds_metal_get_auto_barrier()==1);
        CHECK(myds_metal_commit_batch());
    }
    for(unsigned b=0;b<NOUT;b++) {
        CHECK(memcmp(myds_metal_buffer_data(out[0][b].view),myds_metal_buffer_data(out[1][b].view),sizes[b]*4)==0);
        CHECK(guarded_ok(&out[0][b])&&guarded_ok(&out[1][b]));
        guarded_free(&out[0][b]);guarded_free(&out[1][b]);
    }
    struct guarded *inputs[]={&qa,&qb,&kv,&hc,&iq,&iw,&res,&pre,&scale,&base,&norm,&qrnorm,&kvnorm,&cache,&mask};
    for(unsigned b=0;b<sizeof(inputs)/sizeof(*inputs);b++){CHECK(guarded_ok(inputs[b]));guarded_free(inputs[b]);}
    myds_v41_gpu_scratch_free(scratch);unsetenv("MYDS_QKV_CONCURRENT");
    printf("PASS scoped concurrent QKV/HC/Indexer exact incl original IDs pos=%u fallback=%d\n",pos,finish_fallback);return 1;
}

static int test_qkv_scope_restore(void) {
    const char *values[]={NULL,"0","invalid","1"};
    for(unsigned auto_mode=0;auto_mode<2;auto_mode++)for(unsigned mode=0;mode<2;mode++)
        for(unsigned v=0;v<4;v++)for(unsigned trace=0;trace<2;trace++) {
            if(values[v])setenv("MYDS_QKV_CONCURRENT",values[v],1);else unsetenv("MYDS_QKV_CONCURRENT");
            myds_metal_set_concurrent((int)mode);myds_metal_set_auto_barrier((int)auto_mode);
            CHECK(myds_metal_begin_batch());
            myds_v41_decode_qkv_scope_t scope={0};myds_v41_decode_qkv_scope_begin(&scope,trace);
            const int enabled=v==3&&!trace;CHECK(scope.active==enabled);
            CHECK(myds_metal_get_concurrent()==(enabled?1:(int)mode));
            CHECK(myds_metal_get_auto_barrier()==(enabled?0:(int)auto_mode));
            CHECK(!myds_metal_v41_gemv_bf16(NULL,NULL,NULL,1,32,GGML_TYPE_Q8_0));
            myds_v41_decode_qkv_scope_end(&scope);myds_v41_decode_qkv_scope_end(&scope);
            CHECK(myds_metal_get_concurrent()==(int)mode&&myds_metal_get_auto_barrier()==(int)auto_mode);
            myds_metal_inject_commit_failure(1);CHECK(!myds_metal_commit_batch());myds_metal_inject_commit_failure(0);
            CHECK(myds_metal_get_concurrent()==(int)mode&&myds_metal_get_auto_barrier()==(int)auto_mode);
        }
    myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);unsetenv("MYDS_QKV_CONCURRENT");
    printf("PASS scope opt-in/trace gating and failure restoration of both modes\n");return 1;
}

static int test_finish_concurrent_alias(void) {
    struct guarded q[2],window[2];myds_metal_buffer_t *kv[2];
    for(unsigned arm=0;arm<2;arm++) {
        CHECK(guarded_create(&q[arm],64u*512u*4u));CHECK(guarded_create(&window[arm],128u*512u*4u));
        fill(q[arm].view,11);fill(window[arm].view,41);
        kv[arm]=myds_metal_buffer_sub_view(q[arm].view,0,512u*4u);CHECK(kv[arm]);
    }
    myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
    CHECK(myds_metal_begin_batch());CHECK(original_finish(q[0].view,kv[0],window[0].view,128,true));CHECK(myds_metal_commit_batch());
    myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);
    CHECK(myds_metal_begin_batch());CHECK(myds_v41_gpu_qkv_finish(q[1].view,kv[1],window[1].view,128,true));CHECK(myds_metal_commit_batch());
    myds_metal_set_concurrent(0);myds_metal_set_auto_barrier(1);
    CHECK(exact(q[0].view,q[1].view,64u*512u));CHECK(exact(window[0].view,window[1].view,128u*512u));
    for(unsigned arm=0;arm<2;arm++) {
        CHECK(guarded_ok(&q[arm])&&guarded_ok(&window[arm]));myds_metal_buffer_free(kv[arm]);guarded_free(&q[arm]);guarded_free(&window[arm]);
    }
    printf("PASS concurrent original-finish Q/KV alias ordering and canaries\n");return 1;
}
int main(int argc,char **argv) {
    const char *dir=argc>1?argv[1]:"metal";if(!myds_metal_init(dir))return 1;
    allow_fallback=getenv("MYDS_TEST_QKV_ALLOW_FALLBACK")!=NULL;
    if(!allow_fallback&&!myds_metal_v41_qkv_fusion_available()){fprintf(stderr,"fused shader missing; refusing fallback-only test success\n");return 1;}
    const uint32_t shape[][2]={{1,32},{3,64},{5,128},{65537,32},{1280,5120},{32768,1280},{512,5120}};
    for(size_t i=0;i<sizeof(shape)/sizeof(*shape);i++)for(int batch=0;batch<2;batch++)
        if(!test_gemv(shape[i][0],shape[i][1],batch))return 1;
    if(!test_float_fallback(GGML_TYPE_F16)||!test_float_fallback(GGML_TYPE_F32))return 1;
    if(!test_qkv_scope_restore()||!test_qkv_scope_graph(127,0)||!test_qkv_scope_graph(128,1)||!test_finish_concurrent_alias())return 1;
    const uint32_t pos[]={0,1,127,128,4095,262143,1048575};
    for(size_t i=0;i<sizeof(pos)/sizeof(*pos);i++)for(int compressed=0;compressed<2;compressed++)
        for(int edge=0;edge<2;edge++)if(!test_finish(pos[i],compressed,(int)(i%2),edge))return 1;
    printf("PASS QKV fusion nonzero/exact/shape/view/canary/failure fixtures\n");myds_metal_cleanup();return 0;
}
