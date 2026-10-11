/* Small native Metal component oracle, not a model or speed acceptance gate.
 * Include the actual shared3 host provider to reject its first/second dispatch
 * independently; its reference uses the real production scalar PSOs. */
#define _POSIX_C_SOURCE 200809L
#include "myds_metal.h"
#include "myds_v41.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned reject_dispatch, candidate_dispatches;
static int shared3_dispatch(const char *, const void *, size_t,
    const myds_metal_buffer_t *const *, size_t, const uint32_t[3], const uint32_t[3], size_t);
#define myds_metal_dispatch_kernel shared3_dispatch
#include "../src/myds_v41_moe.c"
#undef myds_metal_dispatch_kernel
static int shared3_dispatch(const char *name, const void *args, size_t bytes,
    const myds_metal_buffer_t *const *buffers, size_t count,
    const uint32_t grid[3], const uint32_t threads[3], size_t scratch) {
    if (!strstr(name, "shared3")) { fprintf(stderr,"unexpected provider %s\n",name);exit(1); }
    ++candidate_dispatches;
    if (reject_dispatch == candidate_dispatches) return 0;
    return myds_metal_dispatch_kernel(name,args,bytes,buffers,count,grid,threads,scratch);
}
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#c);exit(1); } } while (0)
enum { GUARD=32, EXPERTS=21 };
struct guarded { myds_metal_buffer_t *owner,*view; size_t bytes; };
static struct guarded allocate(size_t bytes) {
    struct guarded b={.bytes=bytes};b.owner=myds_metal_buffer_alloc(bytes+2*GUARD);REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+2*GUARD);
    b.view=myds_metal_buffer_sub_view(b.owner,GUARD,bytes);REQUIRE(b.view);return b;
}
static void canary(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(unsigned i=0;i<GUARD;++i)REQUIRE(p[i]==0xa5&&p[GUARD+b.bytes+i]==0xa5);
}
static void release(struct guarded b) {canary(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static uint32_t word(uint32_t x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static uint64_t hash(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<b.bytes+2*GUARD;++i)h=(h^p[i])*UINT64_C(1099511628211);return h;
}
static float bf16(float x) {uint32_t bits;memcpy(&bits,&x,4);bits+=0x7fffu+((bits>>16)&1u);bits&=0xffff0000u;memcpy(&x,&bits,4);return x;}
static void quantized(struct guarded b,bool iq2,unsigned seed,unsigned value_profile) {
    const unsigned bytes=iq2?66u:84u;uint8_t *p=myds_metal_buffer_data(b.view);
    REQUIRE(b.bytes%bytes==0);
    for(size_t k=0;k<b.bytes/bytes;++k) {
        uint8_t *q=p+k*bytes;
        for(unsigned i=0;i<bytes;++i)q[i]=(uint8_t)word((uint32_t)k*97u+i*43u+seed);
        /* Non-power-of-two half scales exercise contraction/rounding edges. */
        uint16_t d=(uint16_t)(0x1400u+word((uint32_t)k+seed)%0x1400u);
        /* These profiles leave many IQ2 dots inside the +/-10 clamp, where
         * exp/division and route-weight multiplication remain observable. */
        if(iq2&&value_profile)d=(uint16_t)(0x1000u+word((uint32_t)k+seed)%0x800u);
        memcpy(q+(iq2?0u:80u),&d,2);
        if(!iq2) {uint16_t dm=(uint16_t)(0x1000u+word((uint32_t)k+seed*7u)%0x1000u);memcpy(q+82,&dm,2);}
    }
}
static void equal(const float *reference,myds_metal_buffer_t *actual,size_t count,const char *role) {
    const float *p=myds_metal_buffer_data(actual);
    for(size_t i=0;i<count;++i) {
        if(memcmp(reference+i,p+i,4))fprintf(stderr,"%s mismatch index=%zu scalar=%.9g shared3=%.9g\n",role,i,reference[i],p[i]);
        REQUIRE(isfinite(p[i])&&!memcmp(reference+i,p+i,4));
    }
}
static void run(uint32_t dim,uint32_t width,unsigned pattern,int concurrent,int barrier,float clamp_value,unsigned case_id,unsigned value_profile) {
    const size_t gu_bytes=(size_t)EXPERTS*width*(dim/256u)*66u;
    struct guarded gate=allocate(gu_bytes),up=allocate(gu_bytes),
        down=allocate((size_t)EXPERTS*dim*(width/256u)*84u),
        input=allocate((size_t)3*dim*4u),ids=allocate(18u*4u),weights=allocate(18u*4u),
        mid=allocate((size_t)18*width*4u),out=allocate((size_t)3*dim*4u),
        sg=allocate(mid.bytes),su=allocate(mid.bytes),sd=allocate((size_t)18*dim*4u),
        refmid=allocate(mid.bytes),refout=allocate(out.bytes);
    struct guarded all[]={gate,up,down,input,ids,weights,mid,out,sg,su,sd,refmid,refout};
    quantized(gate,true,17+value_profile*179u,value_profile);
    quantized(up,true,911+value_profile*683u,value_profile);
    quantized(down,false,73+value_profile*347u,value_profile);
    static const int32_t routes[5][18]={
        {0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5},
        {0,1,2,3,4,5, 5,4,3,2,1,0, 2,3,4,5,0,1},
        {0,2,4,6,8,10, 10,1,4,7,8,13, 13,2,7,0,15,8},
        {0,1,2,3,4,5, 6,7,8,9,10,11, 12,13,14,15,16,17},
        {0,0,1,1,2,2, 2,1,0,2,1,0, 0,2,0,1,2,1}
    };
    memcpy(myds_metal_buffer_data(ids.view),routes[pattern],ids.bytes);
    float *x=myds_metal_buffer_data(input.view),*w=myds_metal_buffer_data(weights.view);
    static const float scales[6]={0.0009765625f,0.0625f,0.25f,1.0f,16.0f,128.0f};
    for(size_t i=0;i<3u*dim;++i)x[i]=bf16(((int)(word((uint32_t)i*41u+pattern*79u)%4095u)-2047)*scales[(i/37u+pattern)%6u]/1024.0f);
    for(unsigned i=0;i<18u;++i)w[i]=(float)(1u+word(i*73u+pattern)%29u)/97.0f;
    if(value_profile) {
        for(size_t i=0;i<3u*dim;++i) {
            const uint32_t h=word((uint32_t)i*41u+pattern*79u+value_profile*997u);
            const float uniform=((int)(h%4095u)-2047)/1024.0f;
            if(value_profile==1)x[i]=bf16(uniform);
            else if(value_profile==2)x[i]=bf16(uniform*scales[(i/37u+pattern)%4u]);
            else if(value_profile==3)x[i]=bf16((i%2u?-1.0f:1.0f)+uniform/256.0f);
            else {
                /* The API accepts F32 input; non-BF16 values also exercise
                 * products and cancellation hidden by a canonical fixture. */
                uint32_t bits=0x3f000000u+(h&0x00ffffffu);
                memcpy(x+i,&bits,4);if(h&0x80000000u)x[i]=-x[i];
            }
        }
        for(unsigned i=0;i<18u;++i) {
            const uint32_t h=word(i*73u+pattern*31u+value_profile*619u);
            /* Positive route weights with arbitrary F32 mantissas. Profile2
             * additionally covers small finite weights, without flush ties. */
            const uint32_t base=value_profile==2?0x35800000u:0x3c800000u;
            const uint32_t bits=base+(h&0x02ffffffu);
            memcpy(w+i,&bits,4);
        }
    }
    uint64_t original[6];for(unsigned i=0;i<6u;++i)original[i]=hash(all[i]);
    myds_metal_buffer_t *in[3],*id[3],*weight[3],*g[3],*u[3],*m[3],*d[3],*o[3];
    for(unsigned t=0;t<3u;++t) {
#define VIEW(v,b,offset,bytes) do {v=myds_metal_buffer_sub_view(b.view,offset,bytes);REQUIRE(v);} while(0)
        VIEW(in[t],input,t*dim*4u,dim*4u);VIEW(id[t],ids,t*6u*4u,6u*4u);VIEW(weight[t],weights,t*6u*4u,6u*4u);
        VIEW(g[t],sg,t*6u*width*4u,6u*width*4u);VIEW(u[t],su,t*6u*width*4u,6u*width*4u);
        VIEW(m[t],refmid,t*6u*width*4u,6u*width*4u);VIEW(d[t],sd,t*6u*dim*4u,6u*dim*4u);VIEW(o[t],refout,t*dim*4u,dim*4u);
#undef VIEW
    }
    myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(barrier);
    /* Three separately completed scalar graphs supply an independent oracle. */
    for(unsigned t=0;t<3u;++t) {
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_metal_routed_moe(o[t],in[t],gate.view,up.view,down.view,id[t],weight[t],
            g[t],u[t],m[t],d[t],dim,width,EXPERTS,6,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,clamp_value));
        REQUIRE(myds_metal_commit_batch());
    }
    float *expected_mid=malloc(refmid.bytes),*expected_out=malloc(refout.bytes);REQUIRE(expected_mid&&expected_out);
    memcpy(expected_mid,myds_metal_buffer_data(refmid.view),refmid.bytes);memcpy(expected_out,myds_metal_buffer_data(refout.view),refout.bytes);
    /* The original scalar providers must also match in one shared encoder. */
    REQUIRE(myds_metal_begin_batch());
    for(unsigned t=0;t<3u;++t)REQUIRE(myds_metal_routed_moe(o[t],in[t],gate.view,up.view,down.view,id[t],weight[t],
        g[t],u[t],m[t],d[t],dim,width,EXPERTS,6,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,clamp_value));
    REQUIRE(myds_metal_commit_batch());equal(expected_mid,refmid.view,refmid.bytes/4u,"same_encoder_mid");equal(expected_out,refout.view,refout.bytes/4u,"same_encoder_out");
    reject_dispatch=0;candidate_dispatches=0;myds_v41_depth2_routed_shared3_counters(NULL,true);
    REQUIRE(myds_metal_begin_batch());myds_metal_dispatch_count_reset();
    REQUIRE(myds_v41_routed_moe_shared3(out.view,input.view,gate.view,up.view,down.view,ids.view,weights.view,mid.view,
        dim,width,EXPERTS,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,clamp_value));
    REQUIRE(myds_metal_dispatch_count()==2&&candidate_dispatches==2);REQUIRE(myds_metal_commit_batch());
    equal(expected_mid,mid.view,mid.bytes/4u,"mid");equal(expected_out,out.view,out.bytes/4u,"out");
    myds_v41_depth2_routed_shared3_counters_t counters;myds_v41_depth2_routed_shared3_counters(&counters,false);
    REQUIRE(counters.requests==1&&counters.encoded==1&&counters.failures==0);
    for(unsigned i=0;i<6u;++i)REQUIRE(original[i]==hash(all[i]));
    if(case_id==0) {
        const uint64_t before_mid=hash(mid),before_out=hash(out);
        candidate_dispatches=0;myds_v41_depth2_routed_shared3_counters(NULL,true);
        REQUIRE(!myds_v41_routed_moe_shared3(out.view,input.view,gate.view,up.view,down.view,ids.view,weights.view,mid.view,
            dim,width,EXPERTS,GGML_TYPE_Q2_K,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,clamp_value));
        REQUIRE(!myds_v41_routed_moe_shared3(out.view,input.view,gate.view,up.view,down.view,ids.view,weights.view,mid.view,
            dim,2560u,EXPERTS,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,clamp_value));
        REQUIRE(!myds_v41_routed_moe_shared3(out.view,input.view,gate.view,up.view,down.view,ids.view,weights.view,input.view,
            dim,width,EXPERTS,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,clamp_value));
        REQUIRE(candidate_dispatches==0&&hash(mid)==before_mid&&hash(out)==before_out);
        myds_v41_depth2_routed_shared3_counters(&counters,false);REQUIRE(counters.requests==3&&counters.encoded==0&&counters.failures==3);
        for(unsigned nth=1;nth<=2;++nth) {
            candidate_dispatches=0;reject_dispatch=nth;myds_v41_depth2_routed_shared3_counters(NULL,true);
            REQUIRE(myds_metal_begin_batch());
            REQUIRE(!myds_v41_routed_moe_shared3(out.view,input.view,gate.view,up.view,down.view,ids.view,weights.view,mid.view,
                dim,width,EXPERTS,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,clamp_value));
            REQUIRE(candidate_dispatches==nth);REQUIRE(myds_metal_commit_batch());
            myds_v41_depth2_routed_shared3_counters(&counters,false);REQUIRE(counters.requests==1&&counters.encoded==0&&counters.failures==1);
            REQUIRE(hash(out)==before_out); /* Down never encoded on either rejection. */
        }
        reject_dispatch=0;
        printf("{\"kind\":\"shared3_guard_and_dispatch_failures\",\"guards\":3,\"failed_dispatches\":2,\"no_failure_encoded\":true,\"out_unchanged\":true}\n");
    }
    printf("{\"kind\":\"routed_shared3\",\"case\":%u,\"dim\":%u,\"width\":%u,\"route_pattern\":%u,\"value_profile\":%u,\"concurrent\":%d,\"auto_barrier\":%d,\"clamp\":%.1f,\"dispatches\":2,\"iq2_scalar_nsg\":4,\"down_scalar_nsg\":4,\"down_candidate_nsg\":1,\"mid_bit_exact\":true,\"out_bit_exact\":true,\"unchanged_inputs\":true,\"canaries\":true,\"fixture_not_fullmodel\":true}\n",case_id,dim,width,pattern,value_profile,concurrent,barrier,clamp_value);
    free(expected_mid);free(expected_out);
    for(unsigned t=0;t<3u;++t) {myds_metal_buffer_free(in[t]);myds_metal_buffer_free(id[t]);myds_metal_buffer_free(weight[t]);myds_metal_buffer_free(g[t]);myds_metal_buffer_free(u[t]);myds_metal_buffer_free(m[t]);myds_metal_buffer_free(d[t]);myds_metal_buffer_free(o[t]);}
    for(unsigned i=0;i<sizeof(all)/sizeof(*all);++i)release(all[i]);
}
int main(int argc,char **argv) {
    unsetenv("MYDS_METAL_CONCURRENT");setenv("MYDS_MOE_NSG","4",1);setenv("MYDS_MOE_DOWN_NSG","4",1);
    setenv("MYDS_MOE_BATCH_DEDUP","0",1);setenv("MYDS_MOE_MM_MIN_TOKENS","0",1);setenv("MYDS_MOE_IQ2_LUT","threadgroup",1);
    setenv("MYDS_MOE_Q2_SUM_CACHE","0",1);setenv("MYDS_MOE_Q2_SUM_CACHE_FUSED","0",1);
    REQUIRE(myds_metal_init(argc>1?argv[1]:"bin"));unsigned id=0;
    for(unsigned shape=0;shape<2u;++shape)for(unsigned pattern=0;pattern<5u;++pattern)
        for(int concurrent=0;concurrent<2;++concurrent)for(int barrier=0;barrier<2;++barrier)
            run(shape?5120u:512u,shape?2304u:256u,pattern,concurrent,barrier,pattern==3u?0.0f:10.0f,id++,0);
    if(getenv("MYDS_MOE_SHARED3_RANGE_SWEEP")&&!strcmp(getenv("MYDS_MOE_SHARED3_RANGE_SWEEP"),"1"))
        for(unsigned profile=1;profile<=4u;++profile)for(unsigned shape=0;shape<2u;++shape)
            for(unsigned pattern=0;pattern<5u;++pattern)for(int concurrent=0;concurrent<2;++concurrent)
                for(int barrier=0;barrier<2;++barrier)
                    run(shape?5120u:512u,shape?2304u:256u,pattern,concurrent,barrier,10.0f,id++,profile);
    myds_metal_cleanup();printf("PASS: %u shared3 routed component cases\n",id);return 0;
}
