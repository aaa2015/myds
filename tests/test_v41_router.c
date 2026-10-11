/* Actual target 384-expert/top-6 GPU routing against independent double math.
 * Repeat identical inputs to detect instability, and compare all three fusion
 * levels and scalar/batch row addressing. This does not load a full model. */
#define _POSIX_C_SOURCE 200809L
#include "myds_metal.h"
#include "myds_v41_moe.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); \
} } while (0)
enum { EXPERTS=384, TOP_K=6, MAX_ROWS=5, GUARD=16, REPEATS=12 };
struct guarded { myds_metal_buffer_t *owner,*view;size_t bytes; };
static unsigned checked_rows;
static double max_weight_error;
static uint32_t rng_state=42u;

static struct guarded allocate(size_t bytes) {
    struct guarded g={.bytes=bytes};
    g.owner=myds_metal_buffer_alloc(bytes+2u*GUARD);CHECK(g.owner);
    memset(myds_metal_buffer_data(g.owner),0xa5,bytes+2u*GUARD);
    g.view=myds_metal_buffer_sub_view(g.owner,GUARD,bytes);CHECK(g.view);return g;
}
static void release(struct guarded g) {
    const uint8_t *p=myds_metal_buffer_data(g.owner);
    for(unsigned i=0;i<GUARD;i++)CHECK(p[i]==0xa5&&p[GUARD+g.bytes+i]==0xa5);
    myds_metal_buffer_free(g.view);myds_metal_buffer_free(g.owner);
}
static float random_float(float low,float high) {
    rng_state=rng_state*1664525u+1013904223u;
    return low+(high-low)*(float)(rng_state>>8)/16777216.0f;
}
static double probability(double x) {
    return sqrt(fmax(x,0.0)+log1p(exp(-fabs(x))));
}
static void oracle(const float *x,const float *bias,float scale,int32_t *ids,double *weights) {
    double p[EXPERTS],score[EXPERTS];int picked[EXPERTS]={0};
    for(unsigned e=0;e<EXPERTS;e++){p[e]=probability(x[e]);score[e]=p[e]+bias[e];}
    double sum=0.0;
    for(unsigned k=0;k<TOP_K;k++) {
        unsigned best=EXPERTS;
        for(unsigned e=0;e<EXPERTS;e++)
            if(!picked[e]&&(best==EXPERTS||score[e]>score[best]))best=e;
        CHECK(best<EXPERTS);picked[best]=1;ids[k]=(int32_t)best;weights[k]=p[best];sum+=weights[k];
    }
    sum=fmax(sum,0x1p-14);
    for(unsigned k=0;k<TOP_K;k++)weights[k]=weights[k]/sum*scale;
}
static void validate(const float *x,const float *bias,unsigned rows,float scale,
                     const int32_t *ids,const float *weights,const float *probs,unsigned mode) {
    for(unsigned r=0;r<rows;r++) {
        int32_t expected_ids[TOP_K];double expected_weights[TOP_K];
        oracle(x+r*EXPERTS,bias,scale,expected_ids,expected_weights);
        if(mode==3u)CHECK(expected_ids[0]==5&&expected_ids[5]==0);
        double sum=0.0;
        for(unsigned k=0;k<TOP_K;k++) {
            const unsigned i=r*TOP_K+k;const double error=fabs(weights[i]-expected_weights[k]);
            if(ids[i]!=expected_ids[k]||!isfinite(weights[i])||
               error>2e-6+2e-5*fabs(expected_weights[k])) {
                fprintf(stderr,"row=%u k=%u mode=%u id=%d expected=%d weight=%.9g expected=%.12g\n",
                        r,k,mode,ids[i],expected_ids[k],weights[i],expected_weights[k]);exit(1);
            }
            if(error>max_weight_error)max_weight_error=error;sum+=weights[i];
        }
        for(unsigned e=0;e<EXPERTS;e++) {
            const double p=probability(x[r*EXPERTS+e]);
            CHECK(isfinite(probs[r*EXPERTS+e]));
            CHECK(fabs(probs[r*EXPERTS+e]-p)<=2e-6+2e-5*fabs(p));
        }
        if(mode==2u||mode==4u)CHECK(sum<scale*0.05);
        else CHECK(fabs(sum-scale)<2e-5*scale);
        checked_rows++;
    }
}
static void run_case(unsigned rows,unsigned mode,float scale) {
    struct guarded logits=allocate(rows*EXPERTS*sizeof(float)),bias=allocate(EXPERTS*sizeof(float));
    struct guarded ids=allocate(rows*TOP_K*sizeof(int32_t)),weights=allocate(rows*TOP_K*sizeof(float));
    struct guarded probs=allocate(rows*EXPERTS*sizeof(float)),scores=allocate(rows*EXPERTS*sizeof(float));
    float *x=myds_metal_buffer_data(logits.view),*b=myds_metal_buffer_data(bias.view);
    for(unsigned e=0;e<EXPERTS;e++)b[e]=mode==0u?random_float(-2.0f,2.0f):0.0f;
    for(unsigned r=0;r<rows;r++)for(unsigned e=0;e<EXPERTS;e++) {
        if(mode==0u)x[r*EXPERTS+e]=random_float(-35.0f,100.0f);
        else if(mode==1u)x[r*EXPERTS+e]=0.0f;
        else if(mode==2u)x[r*EXPERTS+e]=-30.0f-e*0.1f-r*0.03f;
        else if(mode==3u)x[r*EXPERTS+e]=e<TOP_K?4.0f-e*2.0f:-50.0f;
        else x[r*EXPERTS+e]=-200.0f;
    }
    if(mode==3u)for(unsigned e=0;e<EXPERTS;e++)b[e]=e<TOP_K?e*4.0f:-10.0f;
    float input[MAX_ROWS*EXPERTS],bias_copy[EXPERTS];
    memcpy(input,x,logits.bytes);memcpy(bias_copy,b,bias.bytes);
    int32_t reference_ids[MAX_ROWS*TOP_K];float reference_weights[MAX_ROWS*TOP_K];
    float reference_probs[MAX_ROWS*EXPERTS];
    for(unsigned fusion=0;fusion<3;fusion++) {
        const char *level=fusion==0?"0":fusion==1?"1":"2";CHECK(!setenv("MYDS_MOE_ROUTE_FUSE",level,1));
        for(unsigned repeat=0;repeat<REPEATS;repeat++) {
            CHECK(myds_metal_begin_batch());
            const uint64_t dispatches=myds_metal_dispatch_count();
            CHECK(myds_v41_route_batch(ids.view,weights.view,probs.view,scores.view,
                                      logits.view,bias.view,EXPERTS,TOP_K,scale,rows));
            CHECK(myds_metal_dispatch_count()-dispatches==(fusion==0u?7u:fusion==1u?3u:1u));
            CHECK(myds_metal_commit_batch());
            const int32_t *got_ids=myds_metal_buffer_data(ids.view);
            const float *got_weights=myds_metal_buffer_data(weights.view),*got_probs=myds_metal_buffer_data(probs.view);
            validate(input,bias_copy,rows,scale,got_ids,got_weights,got_probs,mode);
            if(fusion==0u&&repeat==0u) {
                memcpy(reference_ids,got_ids,ids.bytes);memcpy(reference_weights,got_weights,weights.bytes);
                memcpy(reference_probs,got_probs,probs.bytes);
            } else {
                CHECK(!memcmp(reference_ids,got_ids,ids.bytes));
                CHECK(!memcmp(reference_weights,got_weights,weights.bytes));
                CHECK(!memcmp(reference_probs,got_probs,probs.bytes));
            }
        }
    }
    /* Scalar API on each row in one command buffer must match the batch.
     * Disjoint views exercise the target's row and shared-bias addressing. */
    myds_metal_buffer_t *views[MAX_ROWS][5];
    CHECK(myds_metal_begin_batch());
    const uint64_t scalar_dispatches=myds_metal_dispatch_count();
    for(unsigned r=0;r<rows;r++) {
        views[r][0]=myds_metal_buffer_sub_view(ids.view,r*TOP_K*sizeof(int32_t),TOP_K*sizeof(int32_t));
        views[r][1]=myds_metal_buffer_sub_view(weights.view,r*TOP_K*sizeof(float),TOP_K*sizeof(float));
        views[r][2]=myds_metal_buffer_sub_view(probs.view,r*EXPERTS*sizeof(float),EXPERTS*sizeof(float));
        views[r][3]=myds_metal_buffer_sub_view(scores.view,r*EXPERTS*sizeof(float),EXPERTS*sizeof(float));
        views[r][4]=myds_metal_buffer_sub_view(logits.view,r*EXPERTS*sizeof(float),EXPERTS*sizeof(float));
        for(unsigned v=0;v<5;v++)CHECK(views[r][v]);
        CHECK(myds_v41_route(views[r][0],views[r][1],views[r][2],views[r][3],
                            views[r][4],bias.view,EXPERTS,TOP_K,scale));
    }
    CHECK(myds_metal_dispatch_count()-scalar_dispatches==rows);
    CHECK(myds_metal_commit_batch());
    CHECK(!memcmp(reference_ids,myds_metal_buffer_data(ids.view),ids.bytes));
    CHECK(!memcmp(reference_weights,myds_metal_buffer_data(weights.view),weights.bytes));
    CHECK(!memcmp(reference_probs,myds_metal_buffer_data(probs.view),probs.bytes));
    CHECK(!memcmp(input,x,logits.bytes)&&!memcmp(bias_copy,b,bias.bytes));
    for(unsigned r=0;r<rows;r++)for(unsigned v=0;v<5;v++)myds_metal_buffer_free(views[r][v]);
    release(logits);release(bias);release(ids);release(weights);release(probs);release(scores);
}
int main(void) {
    /* Prevent an inherited environment override from forcing every mode to
     * use concurrent encoding while still reporting the explicit setter. */
    CHECK(!setenv("MYDS_METAL_CONCURRENT","0",1));
    CHECK(myds_metal_init(NULL));
    for(unsigned concurrent=0;concurrent<2;concurrent++)for(unsigned automatic=0;automatic<2;automatic++) {
        myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(automatic);
        CHECK(myds_metal_get_concurrent()==(int)concurrent);
        CHECK(myds_metal_get_auto_barrier()==(int)automatic);
        for(unsigned trial=0;trial<4;trial++)run_case(trial?5u:1u,0u,trial&1u?1.5f:2.5f);
        for(unsigned mode=1;mode<5;mode++)run_case(5u,mode,1.5f);
        printf("{\"kind\":\"target384_top6_gpu_router\",\"concurrent\":%u,\"auto_barrier\":%u,"
               "\"fusion_levels\":3,\"identical_input_repeats\":%u,\"pass\":true}\n",
               concurrent,automatic,REPEATS);
    }
    printf("{\"kind\":\"target384_top6_gpu_router_summary\",\"oracle_rows\":%u,"
           "\"max_weight_abs_error\":%.9g,\"fusion_and_scalar_batch_bit_exact\":true,"
           "\"full_model_correctness_or_performance_verified\":false,\"pass\":true}\n",
           checked_rows,max_weight_error);
    myds_metal_cleanup();return 0;
}
