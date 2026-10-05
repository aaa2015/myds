/* Root schedules execution; this test never loads a full model. */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_gpu.h"
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
int main(int argc,char **argv) {
    const char *dir=argc>1?argv[1]:"metal";if(!myds_metal_init(dir))return 1;
    allow_fallback=getenv("MYDS_TEST_QKV_ALLOW_FALLBACK")!=NULL;
    if(!allow_fallback&&!myds_metal_v41_qkv_fusion_available()){fprintf(stderr,"fused shader missing; refusing fallback-only test success\n");return 1;}
    const uint32_t shape[][2]={{1,32},{3,64},{5,128},{65537,32},{1280,5120},{32768,1280},{512,5120}};
    for(size_t i=0;i<sizeof(shape)/sizeof(*shape);i++)for(int batch=0;batch<2;batch++)
        if(!test_gemv(shape[i][0],shape[i][1],batch))return 1;
    if(!test_float_fallback(GGML_TYPE_F16)||!test_float_fallback(GGML_TYPE_F32))return 1;
    const uint32_t pos[]={0,1,127,128,4095,262143,1048575};
    for(size_t i=0;i<sizeof(pos)/sizeof(*pos);i++)for(int compressed=0;compressed<2;compressed++)
        for(int edge=0;edge<2;edge++)if(!test_finish(pos[i],compressed,(int)(i%2),edge))return 1;
    printf("PASS QKV fusion nonzero/exact/shape/view/canary/failure fixtures\n");myds_metal_cleanup();return 0;
}
