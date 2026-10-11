/* Production GU pair acceptance. Include the real implementation so a thin
 * native-dispatch interposer can fail row 0 or row 1 before encoding it. Link
 * the other engine objects, excluding myds_v41_moe.o. No GPU math is mocked. */
#define _POSIX_C_SOURCE 200809L
#include "myds_metal.h"
#include "myds_v41_moe.h"
#include "myds_v41_decode.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned fail_dispatch, native_calls;
static int pair_dispatch(const char *, const void *, size_t,
    const myds_metal_buffer_t *const *, size_t, const uint32_t[3],
    const uint32_t[3], size_t);
#define myds_metal_dispatch_kernel pair_dispatch
#include "../src/myds_v41_moe.c"
#undef myds_metal_dispatch_kernel
static int pair_dispatch(const char *name, const void *args, size_t bytes,
    const myds_metal_buffer_t *const *buffers, size_t count,
    const uint32_t groups[3], const uint32_t threads[3], size_t scratch) {
    ++native_calls;
    if (strstr(name,"shared2")) {fprintf(stderr,"unexpected shared2 dispatch: %s\n",name);exit(1);}
    if (fail_dispatch && native_calls == fail_dispatch) return 0;
    return myds_metal_dispatch_kernel(name,args,bytes,buffers,count,groups,threads,scratch);
}

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
enum { WIDTH=5120, ROWS=2304, GUARD=32 };
struct guarded { myds_metal_buffer_t *owner,*view; size_t bytes; };
static struct guarded alloc(size_t bytes) {
    struct guarded a={.bytes=bytes};a.owner=myds_metal_buffer_alloc(bytes+2u*GUARD);REQUIRE(a.owner);
    memset(myds_metal_buffer_data(a.owner),0xa5,bytes+2u*GUARD);
    a.view=myds_metal_buffer_sub_view(a.owner,GUARD,bytes);REQUIRE(a.view);return a;
}
static void canary(struct guarded a) {
    const uint8_t *p=myds_metal_buffer_data(a.owner);
    for(unsigned i=0;i<GUARD;++i)REQUIRE(p[i]==0xa5&&p[GUARD+a.bytes+i]==0xa5);
}
static void release(struct guarded a) {canary(a);myds_metal_buffer_free(a.view);myds_metal_buffer_free(a.owner);}
static uint32_t random_word(uint32_t x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static uint64_t hash(const void *data,size_t bytes) {
    const uint8_t *p=data;uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;++i)h=(h^p[i])*UINT64_C(1099511628211);return h;
}
static float bf16(float x) {uint32_t b;memcpy(&b,&x,4);b+=0x7fffu+((b>>16)&1u);b&=0xffff0000u;memcpy(&x,&b,4);return x;}
static const float ties[16]={1.00390625f,1.01171875f,-1.00390625f,-1.01171875f,
    9.96875f,10.03125f,-9.96875f,-10.03125f,0.501953125f,0.505859375f,
    -0.501953125f,-0.505859375f,31.9375f,32.125f,-31.9375f,-32.125f};
static void weights(struct guarded a,ggml_type type,unsigned seed,bool up,bool tie) {
    uint8_t *p=myds_metal_buffer_data(a.view);memset(p,0,a.bytes);
    const unsigned block=type==GGML_TYPE_Q4_K?256u:32u,bytes=type==GGML_TYPE_Q4_K?144u:34u;
    for(unsigned row=0;row<ROWS;++row)for(unsigned b=0;b<WIDTH/block;++b) {
        uint8_t *q=p+((size_t)row*(WIDTH/block)+b)*bytes;
        uint16_t d=tie?0x3c00u:(uint16_t)(0x0c00u+(random_word(row*93u+b*53u+seed+(up?111u:0u))%10u)*0x400u);
        memcpy(q,&d,2);
        if(tie) {
            if(!b) {
                const unsigned k=(row%16u)+(up?16u:0u);
                if(type==GGML_TYPE_Q4_K){q[4]=1;q[16+k]=1;}else q[2+k]=1;
            }
        } else if(type==GGML_TYPE_Q4_K) {
            uint16_t dm=(uint16_t)(0x0400u+(row+b)%7u*0x300u);memcpy(q+2,&dm,2);
            for(unsigned i=4;i<bytes;++i)q[i]=(uint8_t)random_word(seed+row*97u+b*31u+i*17u+(up?919u:0u));
        } else for(unsigned i=0;i<32u;++i)q[2+i]=(uint8_t)(int8_t)((int)(random_word(seed+row*97u+b*31u+i*17u+(up?919u:0u))%31u)-15);
    }
}
static void check_equal(myds_metal_buffer_t *a,myds_metal_buffer_t *b) {
    const float *x=myds_metal_buffer_data(a),*y=myds_metal_buffer_data(b);
    for(unsigned i=0;i<ROWS;++i) {
        if(memcmp(x+i,y+i,4))fprintf(stderr,"GU mismatch row=%u reference=%.9g pair=%.9g\n",i,x[i],y[i]);
        REQUIRE(!memcmp(x+i,y+i,4)&&isfinite(y[i]));uint32_t bits;memcpy(&bits,y+i,4);REQUIRE(!(bits&0xffffu)&&bits!=0xa5a5a5a5u);
    }
}
static void run(ggml_type type,int concurrent,int barrier,unsigned seed,bool tie,float limit,unsigned id) {
    const size_t wb=(size_t)ROWS*(WIDTH/(type==GGML_TYPE_Q4_K?256u:32u))*(type==GGML_TYPE_Q4_K?144u:34u);
    struct guarded g=alloc(wb),u=alloc(wb),x=alloc(2u*WIDTH*4u),out=alloc(2u*ROWS*4u),ref[2]={alloc(ROWS*4u),alloc(ROWS*4u)};
    myds_metal_buffer_t *in[2],*dest[2];
    weights(g,type,seed,false,tie);weights(u,type,seed,true,tie);
    const float scales[6]={0.015625f,0.0625f,0.25f,1.0f,4.0f,16.0f};
    for(unsigned t=0;t<2u;++t) {
        in[t]=myds_metal_buffer_sub_view(x.view,t*WIDTH*4u,WIDTH*4u);
        dest[t]=myds_metal_buffer_sub_view(out.view,t*ROWS*4u,ROWS*4u);REQUIRE(in[t]&&dest[t]);
        float *p=myds_metal_buffer_data(in[t]);
        for(unsigned i=0;i<WIDTH;++i)p[i]=tie?ties[(i+(t?7u:0u))%16u]:((int)(random_word(i*57u+seed*193u+t*619u)%4095u)-2047)*scales[seed%6u]/1024.0f;
    }
    const uint64_t gh=hash(myds_metal_buffer_data(g.view),g.bytes),uh=hash(myds_metal_buffer_data(u.view),u.bytes),xh=hash(myds_metal_buffer_data(x.view),x.bytes);
    myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(barrier);fail_dispatch=0;
    /* Independent scalar completion boundaries prevent observation copies
     * from supplying the paired graph's dependencies. */
    for(unsigned t=0;t<2u;++t){REQUIRE(myds_metal_begin_batch());REQUIRE(myds_v41_shared_gate_up_swiglu(ref[t].view,g.view,u.view,in[t],ROWS,WIDTH,type,type,limit));REQUIRE(myds_metal_commit_batch());}
    if(tie) {
        struct guarded projected=alloc(ROWS*4u);
        for(unsigned t=0;t<2u;++t) {
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_v41_decode_gemv_bf16(projected.view,g.view,in[t],ROWS,WIDTH,type,false));REQUIRE(myds_metal_commit_batch());
            const float *p=myds_metal_buffer_data(projected.view),*xi=myds_metal_buffer_data(in[t]);
            for(unsigned r=0;r<ROWS;++r){float expected=bf16(xi[r%16u]);REQUIRE(!memcmp(p+r,&expected,4));}
        }
        release(projected);
    }
    for(unsigned repeat=0;repeat<3u;++repeat) {
        memset(myds_metal_buffer_data(out.view),0xa5,out.bytes);native_calls=0;const uint64_t before=myds_metal_dispatch_count();
        REQUIRE(myds_metal_begin_batch());REQUIRE(myds_v41_shared_gate_up_swiglu_pair(dest,g.view,u.view,in,ROWS,WIDTH,type,type,limit));
        REQUIRE(native_calls==2u&&myds_metal_dispatch_count()-before==2u);REQUIRE(myds_metal_commit_batch());
        for(unsigned t=0;t<2u;++t)check_equal(ref[t].view,dest[t]);
    }
    /* Actual producer failures are returned even after the first row has
     * already been encoded. The enclosing transaction owns restoration. */
    if(tie&&limit==10.0f)for(unsigned fail=1;fail<=2u;++fail) {
        memset(myds_metal_buffer_data(out.view),0xa5,out.bytes);native_calls=0;fail_dispatch=fail;
        const uint64_t before=myds_metal_dispatch_count();REQUIRE(myds_metal_begin_batch());
        REQUIRE(!myds_v41_shared_gate_up_swiglu_pair(dest,g.view,u.view,in,ROWS,WIDTH,type,type,limit));
        REQUIRE(native_calls==fail&&myds_metal_dispatch_count()-before==fail-1u);REQUIRE(myds_metal_commit_batch());
        const uint8_t *q=myds_metal_buffer_data(dest[1]);for(unsigned i=0;i<ROWS*4u;++i)REQUIRE(q[i]==0xa5);
        if(fail==2u)check_equal(ref[0].view,dest[0]);
        fail_dispatch=0;
    }
    if(tie&&limit==10.0f) {
        myds_metal_buffer_t *short_row=myds_metal_buffer_sub_view(dest[1],0,ROWS*4u-4u);REQUIRE(short_row);
        myds_metal_buffer_t *invalid[][2]={{dest[0],short_row},{dest[0],dest[0]},{dest[0],in[1]}};
        for(unsigned i=0;i<3u;++i) {
            memset(myds_metal_buffer_data(out.view),0xa5,out.bytes);native_calls=0;
            const uint64_t before=myds_metal_dispatch_count();REQUIRE(myds_metal_begin_batch());
            REQUIRE(!myds_v41_shared_gate_up_swiglu_pair(invalid[i],g.view,u.view,in,ROWS,WIDTH,type,type,limit));
            REQUIRE(!native_calls&&myds_metal_dispatch_count()==before);REQUIRE(myds_metal_commit_batch());
            const uint8_t *p=myds_metal_buffer_data(out.view);for(size_t j=0;j<out.bytes;++j)REQUIRE(p[j]==0xa5);
        }
        myds_metal_buffer_free(short_row);
    }
    REQUIRE(gh==hash(myds_metal_buffer_data(g.view),g.bytes)&&uh==hash(myds_metal_buffer_data(u.view),u.bytes)&&xh==hash(myds_metal_buffer_data(x.view),x.bytes));
    printf("{\"kind\":\"production_gu_pair\",\"case\":%u,\"type\":%d,\"concurrent\":%d,\"auto_barrier\":%d,\"seed\":%u,\"bf16_ties\":%s,\"clamp\":%.1f,\"repeats\":3,\"scalar_psos_per_pair\":2,\"shared2_psos\":0,\"bit_exact\":true,\"unchanged_inputs\":true,\"canaries\":true,\"scalar_oracle_separate_cbs\":true,\"fixture_not_fullmodel\":true}\n",id,type,concurrent,barrier,seed,tie?"true":"false",limit);
    for(unsigned t=0;t<2u;++t){myds_metal_buffer_free(in[t]);myds_metal_buffer_free(dest[t]);release(ref[t]);}release(g);release(u);release(x);release(out);
}
int main(void) {
    unsetenv("MYDS_METAL_CONCURRENT");REQUIRE(myds_metal_init("bin"));unsigned id=0;
    for(unsigned type=0;type<2u;++type)for(int c=0;c<2;++c)for(int b=0;b<2;++b)
        for(unsigned seed=0;seed<7u;++seed)for(unsigned clamp=0;clamp<2u;++clamp)
            run(type?GGML_TYPE_Q8_0:GGML_TYPE_Q4_K,c,b,seed,seed==6u,clamp?10.0f:0.0f,id++);
    printf("[PASS] %u production GU pair cases, real scalar PSOs, Q4/Q8, BF16 halfway and dispatch failures.\n",id);
    myds_metal_cleanup();return 0;
}
