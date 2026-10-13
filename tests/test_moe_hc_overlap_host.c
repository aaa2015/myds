/* Actual C scheduling entry with explicit CPU recording stubs only. No Metal
 * device, model execution, shader math or performance claim.
 * clang -O2 -Wall -Wextra -std=c11 -Iinclude -Ithird_party -Wl,-dead_strip
 *   tests/test_moe_hc_overlap_host.c src/myds_v41_moe.c -o /tmp/test_moe_hc_overlap_host
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_moe.h"
#include "myds_v41_decode.h"
#include "myds_log.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct myds_metal_buffer { size_t bytes,offset; void *data; };
static struct myds_metal_buffer buffers[64];
static unsigned used, events;
static const char *event_names[64],*failure;
static int auto_barrier=1,concurrent,batch=1,trace,overlap=1,q2=1,iq2,q4=1;
static int record(const char *name) {
    assert(events<64);event_names[events++]=name;
    return !failure || strcmp(failure,name);
}
static myds_metal_buffer_t *buffer(size_t bytes) {
    assert(used<64);struct myds_metal_buffer *b=&buffers[used++];
    *b=(struct myds_metal_buffer){bytes,0,(void *)(uintptr_t)(UINT64_C(0x100000000)+used*UINT64_C(0x1000000))};
    return b;
}
size_t myds_metal_buffer_size(const myds_metal_buffer_t *b) { return b?b->bytes:0; }
size_t myds_metal_buffer_offset(const myds_metal_buffer_t *b) { return b?b->offset:0; }
void *myds_metal_buffer_data(myds_metal_buffer_t *b) { return b?b->data:NULL; }
void myds_log(myds_log_level_t level,const char *fmt,...) { (void)level;(void)fmt; }
bool myds_trace_active(void) { return trace; }
bool myds_trace_f32(const char *s,uint32_t l,uint32_t p,const myds_metal_buffer_t *b,size_t n) {
    (void)s;(void)l;(void)p;(void)b;(void)n;return true;
}
int myds_metal_get_auto_barrier(void) { return auto_barrier; }
int myds_metal_get_concurrent(void) { return concurrent; }
void myds_metal_set_auto_barrier(int v) { auto_barrier=v; }
void myds_metal_set_concurrent(int v) { concurrent=v; }
int myds_metal_batch_active(void) { return batch; }
int myds_metal_moe_down_overlap_enabled(void) { return overlap; }
int myds_metal_moe_q2_v41_enabled(void) { return q2; }
int myds_metal_moe_iq2_v41_enabled(void) { return iq2; }
int myds_metal_dense_q4_v41_enabled(void) { return q4; }
void myds_metal_barrier(void) { record("barrier"); }
void myds_metal_barrier_buffer(const myds_metal_buffer_t *b) { assert(b);record("join_one"); }
void myds_metal_barrier_buffers(const myds_metal_buffer_t *const *b,size_t n) {
    assert(b && n);
    record(n==5?"join_cohort2":n==3?"join_three":n==4?"join_route":"join_router_output");
}
uint32_t myds_metal_kernel_max_threads(const char *name) { (void)name;return 1024; }
int myds_metal_dispatch_kernel(const char *name,const void *a,size_t size,
    const myds_metal_buffer_t *const *b,size_t nb,const uint32_t g[3],const uint32_t t[3],size_t scratch) {
    (void)a;(void)size;(void)b;(void)nb;(void)g;(void)t;(void)scratch;
    return record(strstr(name,"shared_gate_up")?"shared":"route");
}
int myds_metal_v41_gemv(myds_metal_buffer_t *o,const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x,uint32_t r,uint32_t d,ggml_type ty) {
    (void)o;(void)w;(void)x;(void)r;(void)d;(void)ty;return record("router");
}
int myds_v41_decode_gemv_bf16(myds_metal_buffer_t *o,const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x,uint32_t r,uint32_t d,ggml_type ty,bool tr) {
    (void)o;(void)w;(void)x;(void)r;(void)d;(void)ty;(void)tr;return record("shared_down");
}
int myds_metal_v41_hc_mix(myds_metal_buffer_t *s,const myds_metal_buffer_t *r,
    const myds_metal_buffer_t *f,ggml_type t,const myds_metal_buffer_t *sc,
    const myds_metal_buffer_t *b,myds_metal_buffer_t *n,myds_metal_buffer_t *m,uint32_t d,float e,float h) {
    (void)s;(void)r;(void)f;(void)t;(void)sc;(void)b;(void)n;(void)m;(void)d;(void)e;(void)h;
    return record("complete_hc");
}
int myds_metal_v41_hc_mix_stages_preflight(myds_metal_buffer_t *s,const myds_metal_buffer_t *r,
    const myds_metal_buffer_t *f,const myds_metal_buffer_t *sc,const myds_metal_buffer_t *b,
    myds_metal_buffer_t *n,myds_metal_buffer_t *m,uint32_t d,float e,float h) {
    assert(s && r && f && sc && b && n && m && d==5120 && e>=0 && h>=0);
    assert(!auto_barrier && concurrent && batch);return record("preflight");
}
int myds_metal_v41_hc_mix_norm_stage(myds_metal_buffer_t *n,const myds_metal_buffer_t *r,uint32_t d,float e) {
    assert(n && r && d==5120 && e>=0 && !auto_barrier && concurrent);return record("norm");
}
int myds_metal_v41_hc_mix_project_stage(myds_metal_buffer_t *m,const myds_metal_buffer_t *f,
    const myds_metal_buffer_t *n,uint32_t d) {
    assert(m && f && n && d==5120 && !auto_barrier && concurrent);return record("project");
}
int myds_metal_v41_hc_mix_split_stage(myds_metal_buffer_t *s,const myds_metal_buffer_t *m,
    const myds_metal_buffer_t *sc,const myds_metal_buffer_t *b,uint32_t d,float h) {
    assert(s && m && sc && b && d==5120 && h>=0 && !auto_barrier && concurrent);return record("split");
}
#define ROUTED_ARGS myds_metal_buffer_t *o,const myds_metal_buffer_t *x, \
    const myds_metal_buffer_t *g,const myds_metal_buffer_t *u,const myds_metal_buffer_t *d, \
    const myds_metal_buffer_t *ids,const myds_metal_buffer_t *w,myds_metal_buffer_t *sg, \
    myds_metal_buffer_t *su,myds_metal_buffer_t *sm,myds_metal_buffer_t *sd, \
    uint32_t dim,uint32_t width,uint32_t ne,uint32_t k,ggml_type gt,ggml_type ut,ggml_type dt,float clamp
#define IGNORE_ROUTED (void)o;(void)x;(void)g;(void)u;(void)d;(void)ids;(void)w;(void)sg;(void)su; \
    (void)sm;(void)sd;(void)dim;(void)width;(void)ne;(void)k;(void)gt;(void)ut;(void)dt;(void)clamp
int myds_metal_routed_moe(ROUTED_ARGS) { IGNORE_ROUTED;return record("routed_original"); }
int myds_metal_routed_moe_scoped(ROUTED_ARGS,bool ordinary) {
    IGNORE_ROUTED;(void)ordinary;return record("routed_original");
}
int myds_metal_routed_moe_down_overlap_scoped(ROUTED_ARGS,bool ordinary,myds_metal_buffer_t *so,
    const myds_metal_buffer_t *sw,const myds_metal_buffer_t *si,ggml_type st) {
    IGNORE_ROUTED;assert(ordinary && so && sw && si && st==GGML_TYPE_Q4_K);return record("routed");
}
int myds_metal_v41_moe_add_hc_expand(myds_metal_buffer_t *o,myds_metal_buffer_t *f,
    const myds_metal_buffer_t *m,const myds_metal_buffer_t *s,const myds_metal_buffer_t *r,
    const myds_metal_buffer_t *p,uint32_t d) {
    (void)o;(void)f;(void)m;(void)s;(void)r;(void)p;(void)d;return record("expand");
}
int myds_metal_v41_add_bf16(myds_metal_buffer_t *o,const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b,uint32_t n) { (void)o;(void)a;(void)b;(void)n;return record("add"); }
int myds_metal_v41_hc_expand(myds_metal_buffer_t *o,const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b,const myds_metal_buffer_t *s,uint32_t n) {
    (void)o;(void)a;(void)b;(void)s;(void)n;return record("expand_original");
}
static myds_context_t context;
static myds_model_t model;
static myds_layer_t layer;
static myds_metal_buffer_t *residual,*after_attn,*split,*fn,*scale,*base,*norm,*mix;
static int run(bool ordinary) {
    return myds_v41_moe_pipeline_scoped(&context,0,1,1,residual,after_attn,split,fn,scale,base,norm,mix,1e-5f,ordinary);
}
static void reset(void) {
    events=0;failure=NULL;auto_barrier=1;concurrent=0;batch=1;trace=0;overlap=q2=q4=1;iq2=0;
    myds_v41_moe_hc_overlap_counters(NULL,true);
    myds_v41_moe_hc_overlap_parse_mode();
}
static void counters(uint64_t r,uint64_t e,uint64_t f) {
    myds_metal_decode_candidate_counters_t c;myds_v41_moe_hc_overlap_counters(&c,false);
    assert(c.requests==r && c.encoded==e && c.failures==f);
}
int main(void) {
    context.model=&model;model.layers=&layer;model.n_layers=1;model.dim=5120;model.moe_expert_dim=2304;
    model.n_routed_experts=384;model.n_activated_experts=6;model.n_shared_experts=1;
    model.norm_eps=1e-5f;model.routed_scaling_factor=2.5f;
    layer.router_gate_type=GGML_TYPE_F16;layer.ffn_gate_exps_type=layer.ffn_up_exps_type=GGML_TYPE_IQ2_XXS;
    layer.ffn_down_exps_type=GGML_TYPE_Q2_K;
    layer.shared_gate_type=layer.shared_up_type=layer.shared_down_type=GGML_TYPE_Q4_K;
#define B(v) v=buffer(8u<<20)
    B(context.ffn_norm_out);B(context.router_logits);B(context.selected_ids);B(context.selected_weights);
    B(context.shared_gate_out);B(context.shared_up_out);B(context.shared_swiglu_out);B(context.shared_out);
    B(context.moe_gate);B(context.moe_up);B(context.moe_mid);B(context.moe_down);B(context.moe_out);B(context.ffn_out);
    B(layer.router_gate);B(layer.router_bias);B(layer.ffn_gate_exps);B(layer.ffn_up_exps);B(layer.ffn_down_exps);
    B(layer.shared_gate);B(layer.shared_up);B(layer.shared_down);
    B(residual);B(after_attn);B(split);B(fn);B(scale);B(base);B(norm);B(mix);
#undef B
    setenv("MYDS_ROUTE_SIMD_SHUFFLE","1",1);
    unsetenv("MYDS_MOE_HC_OVERLAP");reset();assert(myds_v41_moe_hc_overlap_enabled()==0 && run(true));
    assert(!strcmp(event_names[0],"complete_hc"));counters(0,0,0);
    setenv("MYDS_MOE_HC_OVERLAP","0",1);reset();assert(run(true));counters(0,0,0);
    setenv("MYDS_MOE_HC_OVERLAP","1",1);reset();assert(run(true));counters(1,1,0);
    const char *expected[]={"preflight","norm","router","shared","join_three","project","route",
        "join_router_output","join_cohort2","split","routed","join_three","expand","join_one"};
    assert(events==sizeof(expected)/sizeof(*expected));
    for(unsigned i=0;i<events;++i)assert(!strcmp(event_names[i],expected[i]));
    assert(auto_barrier==1 && concurrent==0);
    const char *failures[]={"preflight","norm","router","shared","project","route","split","routed","expand"};
    for(unsigned i=0;i<sizeof(failures)/sizeof(*failures);++i) {
        reset();failure=failures[i];assert(!run(true));counters(1,i>=7?1:0,1);
        assert(auto_barrier==1 && concurrent==0);
        for(unsigned j=0;j<events;++j)assert(strcmp(event_names[j],"complete_hc"));
    }
    const char *bad[]={"","01","true","2"," 1","1 ","-1"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);++i) {
        setenv("MYDS_MOE_HC_OVERLAP",bad[i],1);reset();
        assert(myds_v41_moe_hc_overlap_enabled()==-1 && !run(true) && events==0);counters(0,0,0);
    }
    setenv("MYDS_MOE_HC_OVERLAP","1",1);
    for(unsigned excluded=0;excluded<6;++excluded) {
        reset();if(excluded==0)batch=0;if(excluded==1)trace=1;if(excluded==2)iq2=1;
        if(excluded==3)q4=0;if(excluded==4)q2=0;if(excluded==5)overlap=0;
        if(trace) {assert(run(true));assert(!strcmp(event_names[0],"complete_hc"));counters(0,0,0);}
        else {assert(!run(true) && events==0);counters(1,0,1);assert(auto_barrier==1 && concurrent==0);}
    }
    reset();assert(run(false));counters(0,0,0);assert(!strcmp(event_names[0],"complete_hc"));
    reset();uint32_t saved_dim=model.dim;model.dim=4096;
    assert(!run(true) && events==0);counters(1,0,1);assert(auto_barrier==1 && concurrent==0);model.dim=saved_dim;
    reset();myds_metal_buffer_t *saved_split=split;split=NULL;
    assert(!run(true) && events==0);counters(1,0,1);assert(auto_barrier==1 && concurrent==0);split=saved_split;
    reset();void *saved=mix->data;mix->data=context.moe_mid->data;
    assert(!run(true) && events==0);counters(1,0,1);assert(auto_barrier==1 && concurrent==0);mix->data=saved;
    reset();size_t saved_size=norm->bytes;norm->bytes=4;
    assert(!run(true) && events==0);counters(1,0,1);norm->bytes=saved_size;
    reset();for(unsigned i=0;i<40;++i){events=0;assert(run(true));}counters(40,40,0);
    setenv("MYDS_MOE_HC_OVERLAP","invalid_after_parse",1);
    assert(myds_v41_moe_hc_overlap_enabled()==1); /* Cached context mode. */
    myds_v41_moe_hc_overlap_reset_mode();assert(myds_v41_moe_hc_overlap_enabled()==0);
    assert(myds_v41_moe_hc_overlap_parse_mode()==-1);
    puts("PASS: actual MoE C HC schedule, three cohorts, strict env,101 exclusions,scratch guards,counters,stage failures,caller restoration; CPU recording stubs only");
    return 0;
}
