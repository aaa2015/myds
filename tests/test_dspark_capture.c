/* CPU contract test: the real DSpark policy/capture/stale-consumer code,
 * with only the HC dispatch replaced. No Metal initialization or GPU work.
 * Link all project objects except myds_dspark.o, whose real source is included.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_dspark.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned dispatches, commits, model_opens;
static gguf_file_t *capture_gguf_open_stub(const char *path) { (void)path; model_opens++; return NULL; }
static int commit_ok=1;
static int capture_commit_stub(void) { commits++; return commit_ok; }
static int dispatch_ok=1;
static myds_metal_buffer_t *last_out;
static const myds_metal_buffer_t *last_in,*last_weight;
static int capture_hc_stub(myds_metal_buffer_t *out, const myds_metal_buffer_t *in,
    const myds_metal_buffer_t *weight, uint32_t dim) {
    assert(dim==MYDS_DSPARK_DIM);
    dispatches++;last_out=out;last_in=in;last_weight=weight;
    return dispatch_ok;
}
static int capture_hc_round_stub(myds_metal_buffer_t *out, const myds_metal_buffer_t *in,
    const myds_metal_buffer_t *weight, uint32_t dim, int round_bf16) {
    (void)round_bf16;
    return capture_hc_stub(out, in, weight, dim);
}
#define gguf_open capture_gguf_open_stub
#define myds_metal_commit_batch capture_commit_stub
#define myds_metal_v41_hc_weighted_sum capture_hc_stub
#define myds_metal_v41_hc_weighted_sum_round capture_hc_round_stub
#include "../src/myds_dspark.c"
#undef myds_metal_v41_hc_weighted_sum_round
#undef myds_metal_v41_hc_weighted_sum
#undef myds_metal_commit_batch
#undef gguf_open

static void init_policy(myds_dspark_t *ds,const char *mode,const char *engine_mode) {
    memset(ds,0,sizeof(*ds));
    if(mode)assert(!setenv("MYDS_DSPARK_CAPTURE_BYPASS",mode,1));
    else assert(!unsetenv("MYDS_DSPARK_CAPTURE_BYPASS"));
    if(engine_mode)assert(!setenv("MYDS_ENGINE_CAPTURE_BYPASS",engine_mode,1));
    else assert(!unsetenv("MYDS_ENGINE_CAPTURE_BYPASS"));
    assert(myds_dspark_capture_configure(ds));
    ds->main_proj=(myds_metal_buffer_t *)0x10;
    ds->target_hidden_buf=(myds_metal_buffer_t *)0x20;
    ds->hc_mean_weights=(myds_metal_buffer_t *)0x30;
    for(unsigned i=0;i<3;i++) {
        ds->target_layers[i]=37u+i;
        ds->target_slots[i]=(myds_metal_buffer_t *)(uintptr_t)(0x40+16*i);
    }
    ds->block_size=1;
}
static void init(myds_dspark_t *ds,const char *mode) {
    init_policy(ds,mode,NULL);
}
static void token(myds_dspark_t *ds) {
    myds_dspark_capture_invalidate(ds);
    for(unsigned i=0;i<3;i++)assert(myds_dspark_capture_layer_to(ds,37u+i,
        (myds_metal_buffer_t *)0x80,NULL));
    if(ds->capture_bypass_mode && ds->capture_valid_mask==7u) {
        assert(!myds_dspark_capture_ready(ds));
        myds_dspark_capture_publish(ds); /* authoritative target commit succeeded */
    }
}

/* Calls the actual linked tree function. A real CPU-only V4.1 context is
 * supplied, while ctx.model is deliberately NULL: the new guard must precede
 * every model dereference and all capture/position/Metal mutations. */
static unsigned engine_tree_guard(myds_dspark_t *ds) {
    assert(myds_dspark_engine_capture_bypass_enabled(ds));
    assert(!setenv("MYDS_ENGRAM_PROJECT_ROUND_SKIP","0",1));
    myds_model_t model={0};
    myds_v41_context_t *g=myds_v41_context_create_for_model(&model);
    assert(g);
    myds_v41_snapshot_set_positions(g,7u,11u);
    int32_t pcache[4]={2,3,5,7}, pcache_before[4];
    float logits[4]={1.f,2.f,3.f,4.f}, logits_before[4];
    myds_context_t ctx={.model=NULL,.v41=g,.dspark=ds,.pos=11u,
        .pcache_tokens=pcache,.pcache_len=4u,.pcache_cap=4u,
        .continue_from_cache=true,.rng_state=0x123456789abcdef0ull,
        .logits=(myds_metal_buffer_t *)logits};
    ctx.cfg.max_seq_len=4096u;
    myds_context_t ctx_before;
    myds_dspark_t ds_before;
    memcpy(&ctx_before,&ctx,sizeof(ctx));
    memcpy(&ds_before,ds,sizeof(*ds));
    memcpy(pcache_before,pcache,sizeof(pcache));
    memcpy(logits_before,logits,sizeof(logits));
    const uint32_t next_before=myds_v41_snapshot_get_next_position(g);
    const uint32_t engram_before=myds_v41_snapshot_get_engram_position(g);
    const bool valid_before=myds_v41_snapshot_get_valid(g);
    const uint64_t gpu_dispatch_before=myds_metal_dispatch_count();
    const unsigned dispatch_before=dispatches, commit_before=commits;
    const int barrier_before=myds_metal_get_auto_barrier();
    const int32_t tokens[3]={11,13,17};
    const uint32_t positions[3]={0u,11u,UINT32_MAX};
    unsigned cases=0;
    for(int auto_barrier=0;auto_barrier<=1;auto_barrier++) {
        myds_metal_set_auto_barrier(auto_barrier);
        for(unsigned p=0;p<3;p++) {
            int32_t out[2]={INT32_C(0x13579bdf),INT32_C(0x2468ace0)};
            const int32_t out_before[2]={out[0],out[1]};
            assert(myds_v41_forward_tree_spec(&ctx,tokens,positions[p],out)==-1);
            assert(!memcmp(out,out_before,sizeof(out)));
            assert(!memcmp(&ctx,&ctx_before,sizeof(ctx)));
            assert(!memcmp(ds,&ds_before,sizeof(*ds)));
            assert(!memcmp(pcache,pcache_before,sizeof(pcache)));
            assert(!memcmp(logits,logits_before,sizeof(logits)));
            assert(myds_v41_snapshot_get_next_position(g)==next_before);
            assert(myds_v41_snapshot_get_engram_position(g)==engram_before);
            assert(myds_v41_snapshot_get_valid(g)==valid_before);
            assert(myds_metal_dispatch_count()==gpu_dispatch_before);
            assert(dispatches==dispatch_before && commits==commit_before);
            assert(myds_metal_get_auto_barrier()==auto_barrier);
            cases++;
        }
    }
    myds_metal_set_auto_barrier(barrier_before);
    myds_v41_context_free(g);
    return cases;
}

int main(int argc,char **argv) {
    if(argc==2 && !strcmp(argv[1],"--markov-only")) {
        myds_dspark_t ds;setenv("MYDS_DSPARK_MARKOV_ONLY","1",1);init(&ds,"1");
        myds_dspark_capture_begin_request(&ds,true);token(&ds);
        assert(!dispatches && !commits && !myds_dspark_capture_ready(&ds));
        assert(ds.capture_counters.requests==3 && ds.capture_counters.bypassed==3);
        puts("{\"kind\":\"host_markov_scope\",\"pass\":true}");return 0;
    }
    assert(argc==1);unsetenv("MYDS_DSPARK_MARKOV_ONLY");
    assert(!unsetenv("MYDS_ENGINE_CAPTURE_BYPASS"));
    const char *bad[]={"","2","-1","01","true","1 "," 1","0.0","1x"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++) {
        myds_dspark_t ds={0};setenv("MYDS_DSPARK_CAPTURE_BYPASS",bad[i],1);
        assert(!myds_dspark_capture_configure(&ds));
        assert(!ds.capture_configured && !ds.capture_bypass_mode);
        assert(!myds_dspark_load("/must-not-open-invalid-config.gguf",1));
    }
    assert(!model_opens); /* strict invalid modes precede model resources */
    myds_dspark_t legacy,candidate;
    init(&legacy,NULL);assert(!myds_dspark_capture_bypass_enabled(&legacy));
    setenv("MYDS_DSPARK_CAPTURE_BYPASS","1",1);
    assert(myds_dspark_capture_configure(&legacy)); /* immutable */
    myds_dspark_capture_begin_request(&legacy,false);
    token(&legacy);assert(dispatches==3 && myds_dspark_capture_ready(&legacy));
    dispatch_ok=0;
    assert(myds_dspark_capture_layer_to(&legacy,37,(myds_metal_buffer_t *)0x80,NULL));
    assert(dispatches==4); /* mode0 keeps legacy unchecked failure behavior */
    dispatch_ok=1;dispatches=0;
    init(&candidate,"1");
    setenv("MYDS_DSPARK_CAPTURE_BYPASS","0",1);
    assert(myds_dspark_capture_configure(&candidate));
    assert(myds_dspark_capture_bypass_enabled(&candidate));
    /* Unmanaged/public API remains capture-active and forwards exact resources. */
    token(&candidate);assert(dispatches==3 && myds_dspark_capture_ready(&candidate));
    assert(last_out==candidate.target_slots[2] && last_in==(void *)0x80 &&
           last_weight==candidate.hc_mean_weights);
    myds_dspark_capture_counters_t c={0};
    myds_dspark_capture_counters(&candidate,&c,true);
    assert(c.requests==3 && c.encoded==3 && !c.bypassed && !c.failures);
    myds_dspark_capture_begin_request(&candidate,false);
    token(&candidate);assert(dispatches==3 && !myds_dspark_capture_ready(&candidate));
    myds_model_t model={0};myds_context_t ctx={.model=&model};
    assert(!myds_dspark_eval_stages(&candidate,&ctx,.8f,NULL));
    assert(!myds_dspark_propose(&candidate,&ctx,0,1));
    assert(dispatches==3);
    myds_dspark_capture_counters(&candidate,&c,true);
    assert(c.requests==3 && !c.encoded && c.bypassed==3 && c.stale_rejections==2);
    myds_dspark_capture_end_request(&candidate,false);
    assert(myds_dspark_capture_required(&candidate) && !myds_dspark_capture_ready(&candidate));
    /* A new SPEC request cannot consume the previous request's slots. */
    myds_dspark_capture_begin_request(&candidate,true);
    assert(!myds_dspark_capture_ready(&candidate));
    token(&candidate);assert(dispatches==6 && myds_dspark_capture_ready(&candidate));
    /* Workspace encoding never publishes real target slots prematurely. */
    myds_dspark_capture_invalidate(&candidate);
    for(unsigned i=0;i<3;i++)assert(myds_dspark_capture_layer_to(&candidate,37u+i,
        (void *)0x80,(void *)(uintptr_t)(0x100+16*i)));
    assert(!myds_dspark_capture_ready(&candidate));
    myds_dspark_capture_publish(&candidate);assert(myds_dspark_capture_ready(&candidate));
    /* Whole rollback/failure invalidates; partial first-state publishes only
     * after caller has restored all three slots. No GPU completion claim here. */
    myds_dspark_capture_invalidate(&candidate);assert(!myds_dspark_capture_ready(&candidate));
    dispatch_ok=0;
    assert(!myds_dspark_capture_layer_to(&candidate,37,(void *)0x80,NULL));
    assert(!myds_dspark_capture_ready(&candidate));
    dispatch_ok=1;token(&candidate);assert(myds_dspark_capture_ready(&candidate));
    myds_dspark_capture_begin_request(&candidate,false);
    myds_dspark_capture_publish(&candidate);assert(!myds_dspark_capture_ready(&candidate));
    myds_dspark_capture_end_request(&candidate,true);
    /* Direct API captures remain unavailable until the consumer completes its
     * pending CB. This test stubs completion, never claiming GPU correctness. */
    for(unsigned i=0;i<3;i++)myds_dspark_capture_layer(&candidate,37u+i,(void *)0x80);
    assert(!myds_dspark_capture_ready(&candidate));
    assert(!myds_dspark_eval_stages(&candidate,&ctx,.8f,NULL)); /* ws absent */
    assert(commits==1 && myds_dspark_capture_ready(&candidate));
    myds_dspark_capture_layer(&candidate,38,(void *)0x80);
    assert(!myds_dspark_capture_ready(&candidate) && candidate.capture_valid_mask==2u);
    assert(!myds_dspark_eval_stages(&candidate,&ctx,.8f,NULL));
    assert(commits==1); /* incomplete generation must not even commit/read */
    myds_dspark_capture_invalidate(&candidate);
    for(unsigned i=0;i<3;i++)myds_dspark_capture_layer(&candidate,37u+i,(void *)0x80);
    commit_ok=0;
    assert(!myds_dspark_eval_stages(&candidate,&ctx,.8f,NULL));
    assert(commits==2 && !myds_dspark_capture_ready(&candidate));commit_ok=1;
    token(&candidate);myds_dspark_capture_end_request(&candidate,false);
    assert(!myds_dspark_capture_ready(&candidate));
    candidate.main_proj=NULL;
    myds_dspark_capture_begin_request(&candidate,true);
    assert(!myds_dspark_capture_required(&candidate));
    assert(myds_dspark_capture_layer_to(&candidate,37,NULL,NULL));
    assert(!myds_dspark_capture_ready(&candidate));
    /* New mode shares the real strict parser and effective capture mechanism,
     * but cannot enter the experimental tree graph. Old-mode tests above are
     * unchanged; these cases never initialize Metal or execute a model. */
    assert(!myds_dspark_engine_capture_bypass_enabled(&legacy));
    assert(!myds_dspark_engine_capture_bypass_enabled(&candidate));
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++) {
        myds_dspark_t ds={0};
        assert(!setenv("MYDS_DSPARK_CAPTURE_BYPASS","0",1));
        assert(!setenv("MYDS_ENGINE_CAPTURE_BYPASS",bad[i],1));
        assert(myds_dspark_capture_parse_mode()==-1);
        assert(!myds_dspark_capture_configure(&ds));
        assert(!ds.capture_configured && !ds.capture_bypass_mode && !ds.capture_engine_only);
        assert(!myds_dspark_load("/must-not-open-invalid-engine-config.gguf",1));
    }
    myds_dspark_t conflict={0};
    assert(!setenv("MYDS_DSPARK_CAPTURE_BYPASS","1",1));
    assert(!setenv("MYDS_ENGINE_CAPTURE_BYPASS","1",1));
    assert(myds_dspark_capture_parse_mode()==-1);
    assert(!myds_dspark_capture_configure(&conflict));
    assert(!conflict.capture_configured && !conflict.capture_bypass_mode && !conflict.capture_engine_only);
    assert(!myds_dspark_load("/must-not-open-conflicting-config.gguf",1));
    assert(!model_opens);
    const char *valid_old[5]={NULL,"0","1",NULL,"0"};
    const char *valid_new[5]={NULL,"0",NULL,"1","1"};
    const int expected_mode[5]={0,0,1,2,2};
    for(unsigned i=0;i<5;i++) {
        myds_dspark_t ds;
        init_policy(&ds,valid_old[i],valid_new[i]);
        assert(myds_dspark_capture_parse_mode()==expected_mode[i]);
        assert(myds_dspark_capture_bypass_enabled(&ds)==(expected_mode[i]!=0));
        assert(myds_dspark_engine_capture_bypass_enabled(&ds)==(expected_mode[i]==2));
    }
    myds_dspark_t engine;
    init_policy(&engine,"0","1");
    assert(myds_dspark_capture_bypass_enabled(&engine) &&
        myds_dspark_engine_capture_bypass_enabled(&engine));
    /* An existing context keeps mode2 even after env becomes old-mode1. */
    assert(!setenv("MYDS_DSPARK_CAPTURE_BYPASS","1",1));
    assert(!setenv("MYDS_ENGINE_CAPTURE_BYPASS","0",1));
    assert(myds_dspark_capture_parse_mode()==1);
    assert(myds_dspark_capture_configure(&engine));
    assert(engine.capture_engine_only && engine.capture_bypass_mode);
    const unsigned engine_dispatch_start=dispatches;
    token(&engine);
    assert(dispatches==engine_dispatch_start+3 && myds_dspark_capture_ready(&engine));
    myds_dspark_capture_begin_request(&engine,false);
    token(&engine);
    assert(dispatches==engine_dispatch_start+3 && !myds_dspark_capture_ready(&engine));
    assert(engine.capture_counters.requests==6 && engine.capture_counters.encoded==3 &&
        engine.capture_counters.bypassed==3 && !engine.capture_counters.failures);
    myds_dspark_capture_end_request(&engine,true);
    assert(myds_dspark_capture_required(&engine) && !myds_dspark_capture_ready(&engine));
    myds_dspark_capture_begin_request(&engine,true);
    token(&engine);
    assert(dispatches==engine_dispatch_start+6 && myds_dspark_capture_ready(&engine));
    const unsigned tree_cases=engine_tree_guard(&engine);
    assert(tree_cases==6);
    printf("{\"kind\":\"host_engine_capture_contract\",\"strict_invalid_new_modes\":9,"
           "\"mutual_exclusion\":true,\"valid_parser_modes\":5,\"immutable_mode\":true,"
           "\"unmanaged_capture_active\":true,\"tree_guard_cases\":%u,"
           "\"tree_output_state_unchanged\":true,\"gpu_dispatch_delta\":0,"
           "\"gpu_arithmetic_verified\":false,\"pass\":true}\n",tree_cases);
    printf("{\"kind\":\"host_contract\",\"strict_invalid_modes\":9,\"immutable_mode\":true,"
           "\"dispatch_forwarding\":true,\"bypass_zero_dispatch\":true,\"stale_consumer_rejected\":true,"
           "\"failure_and_reenable\":true,\"gpu_arithmetic_verified\":false,\"pass\":true}\n");
    return 0;
}
