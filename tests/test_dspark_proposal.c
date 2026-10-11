/* CPU proposal contract test. The real selector, Markov math, top-2 tree,
 * publication and adaptive backoff run from myds_dspark.c. Only neural stage
 * evaluation and Metal buffer mapping are replaced by deterministic host
 * inputs. This fixture does not claim neural/GPU arithmetic correctness.
 * Link project objects except myds_dspark.o, as for test_dspark_capture. */
#define _POSIX_C_SOURCE 200809L
#include "myds_dspark.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VOCAB 8u
static unsigned eval_calls;
static int eval_ok=1;
static int32_t observed_seed;
static uint32_t observed_position,observed_slot;
static void *host_buffer_data(myds_metal_buffer_t *buffer) { return buffer; }
static myds_metal_buffer_t *unexpected_buffer_alloc(size_t bytes) {
    (void)bytes;assert(!"metadata rejection reached Metal allocation");return NULL;
}
static myds_metal_buffer_t *unexpected_buffer_from_ptr(const void *data,size_t bytes) {
    (void)data;(void)bytes;assert(!"metadata rejection reached Metal tensor mapping");return NULL;
}
static int fixture_eval(myds_dspark_t *ds,myds_context_t *ctx,float min_conf,float *confidence) {
    assert(ctx&&ctx->model&&ctx->model->vocab_size==VOCAB&&min_conf==0.0f);
    eval_calls++;
    observed_seed=ds->current_seed_token;
    observed_position=ds->current_seed_position;
    observed_slot=ds->current_output_slot;
    if(confidence)*confidence=0.9f;
    return eval_ok;
}
#define DSPARK_EVAL_STAGES fixture_eval
#define myds_metal_buffer_data host_buffer_data
#define myds_metal_buffer_alloc unexpected_buffer_alloc
#define myds_metal_buffer_from_ptr unexpected_buffer_from_ptr
#include "../src/myds_dspark.c"
#undef myds_metal_buffer_data
#undef myds_metal_buffer_alloc
#undef myds_metal_buffer_from_ptr
#undef DSPARK_EVAL_STAGES

typedef struct {
    myds_dspark_t ds;
    myds_model_t model;
    myds_context_t ctx;
    float logits[MYDS_DSPARK_DEFAULT_BLOCK_SIZE][VOCAB];
    uint16_t w1[VOCAB][MYDS_DSPARK_MARKOV_RANK],w2[VOCAB][MYDS_DSPARK_MARKOV_RANK];
    float state[MYDS_DSPARK_MARKOV_RANK],bias[VOCAB];
} fixture_t;

static void init(fixture_t *f,bool markov) {
    memset(f,0,sizeof(*f));
    f->model.vocab_size=VOCAB;
    f->ctx.model=&f->model;
    f->ctx.pos=23u;
    f->ds.block_size=MYDS_DSPARK_DEFAULT_BLOCK_SIZE;
    f->ds.markov_rank=MYDS_DSPARK_MARKOV_RANK;
    f->ds.main_proj=(myds_metal_buffer_t *)(uintptr_t)1;
    f->ds.target_hidden_buf=(myds_metal_buffer_t *)(uintptr_t)2;
    f->ds.draft_logits_buf=(myds_metal_buffer_t *)f->logits;
    for(unsigned d=0;d<MYDS_DSPARK_DEFAULT_BLOCK_SIZE;d++) {
        const unsigned primary=d?d+2u:1u;
        f->logits[d][primary]=10.0f;
        f->logits[d][primary+1u]=9.0f;
    }
    if(markov) {
        f->ds.markov_w1=(myds_metal_buffer_t *)f->w1;
        f->ds.markov_w2=(myds_metal_buffer_t *)f->w2;
        f->ds.markov_w1_type=f->ds.markov_w2_type=GGML_TYPE_F16;
        f->ds.markov_state_host=f->state;
        f->ds.markov_bias_host=f->bias;
        /* Exact IEEE half literals. The consumed token favours 7 while the
         * emitted seed favours 6, so wrong conditioning changes the answer. */
        f->w1[1][0]=UINT16_C(0xbc00); /* -1 */
        f->w1[4][0]=UINT16_C(0x3c00); /* +1 */
        f->w2[6][0]=UINT16_C(0x4900); /* +10 */
        f->w2[7][0]=UINT16_C(0xc900); /* -10 */
    }
}
static void stale(myds_dspark_t *ds) {
    ds->draft_count=5u;ds->draft_tree_count=2u;ds->is_high_confidence=true;
    ds->draft_tokens[0]=7;ds->draft_tree_tokens[0]=7;
}
static void unpublished(const myds_dspark_t *ds) {
    assert(!ds->draft_count&&!ds->draft_tree_count&&!ds->is_high_confidence);
}
static void environment(bool markov_only) {
    assert(!setenv("MYDS_DSPARK_MARKOV_ONLY",markov_only?"1":"0",1));
    assert(!setenv("MYDS_DSPARK_MIN_CONF","0",1));
    assert(!setenv("MYDS_DSPARK_MIN_MARGIN","0",1));
    assert(!setenv("MYDS_DSPARK_MARKOV_SCALE","1",1));
    assert(!unsetenv("MYDS_DSPARK_NO_MARKOV"));
    assert(!unsetenv("DS4_DSPARK_NO_MARKOV"));
}
static void markov_only_contract(void) {
    fixture_t f;init(&f,true);
    f.ctx.v41=(myds_v41_context_t *)(uintptr_t)1;
    assert(myds_dspark_propose_next(&f.ds,&f.ctx,1,4,1)==1u);
    assert(!eval_calls&&f.ds.current_seed_token==4&&f.ds.current_seed_position==23u);
    assert(f.ds.current_output_slot==0u&&f.ds.draft_tokens[0]==6);
    assert(f.ds.draft_tree_count==2u&&f.ds.draft_tree_tokens[0]==6);
    assert(f.ds.proposed_tokens==1u&&f.state[0]==1.0f);
    puts("{\"kind\":\"host_dspark_markov_seed_contract\",\"emitted_seed_used\":true,"
         "\"neural_eval_calls\":0,\"gpu_arithmetic_verified\":false,\"pass\":true}");
}
static void neural_contract(void) {
    fixture_t f;init(&f,false);
    assert(myds_dspark_propose(&f.ds,&f.ctx,1,5)==5u);
    assert(eval_calls==1u&&observed_seed==1&&observed_position==23u&&observed_slot==0u);
    assert(f.ds.draft_count==5u&&f.ds.draft_tokens[0]==1);
    assert(f.ds.draft_tree_count==2u&&f.ds.draft_tree_tokens[0]==1&&f.ds.draft_tree_tokens[1]==2);
    assert(f.ds.proposed_tokens==5u);

    f.ctx.v41=(myds_v41_context_t *)(uintptr_t)1;
    assert(myds_dspark_propose_next(&f.ds,&f.ctx,1,4,5)==4u);
    assert(eval_calls==2u&&observed_seed==1&&observed_position==22u&&observed_slot==1u);
    assert(f.ctx.pos==23u&&f.ds.draft_count==4u&&f.ds.draft_tokens[0]==3);
    assert(f.ds.draft_tokens[1]==4&&f.ds.draft_tokens[2]==5&&f.ds.draft_tokens[3]==6);
    assert(f.ds.draft_tree_count==2u&&f.ds.draft_tree_tokens[0]==3&&f.ds.draft_tree_tokens[1]==4);
    assert(f.ds.proposed_tokens==6u); /* V4.1 tests one continuation per cycle. */
    assert(myds_dspark_propose_next(&f.ds,&f.ctx,1,4,1)==1u);
    assert(f.ds.draft_count==1u&&f.ds.proposed_tokens==7u);

    /* Slot-1 conditioning must follow the emitted seed, independently of the
     * token that produced the target capture. Markov arithmetic is real. */
    fixture_t m;init(&m,true);memset(m.logits,0,sizeof(m.logits));
    m.ctx.v41=(myds_v41_context_t *)(uintptr_t)1;
    assert(myds_dspark_propose_next(&m.ds,&m.ctx,1,4,1)==1u);
    assert(observed_seed==1&&observed_position==22u&&observed_slot==1u);
    assert(m.ds.draft_tokens[0]==6&&m.state[0]==1.0f);
    assert(myds_dspark_propose(&m.ds,&m.ctx,1,1)==1u);
    assert(m.ds.draft_tokens[0]==7&&m.state[0]==-1.0f);

    /* A failed stage evaluation must not turn the usable Markov head into a
     * fallback proposal or publish leftover tree storage. */
    stale(&m.ds);eval_ok=0;
    uint64_t before=m.ds.proposed_tokens;
    const unsigned calls=eval_calls;
    assert(!myds_dspark_propose_next(&m.ds,&m.ctx,1,4,4));
    assert(eval_calls==calls+1u&&m.ds.proposed_tokens==before);unpublished(&m.ds);
    eval_ok=1;
    stale(&m.ds);m.ds.backoff_counter=2u;
    const uint64_t cycles=m.ds.cycles;
    const unsigned backoff_calls=eval_calls;
    assert(!myds_dspark_propose_next(&m.ds,&m.ctx,1,4,4));
    assert(m.ds.backoff_counter==1u&&m.ds.cycles==cycles&&m.ds.proposed_tokens==before);
    assert(eval_calls==backoff_calls);unpublished(&m.ds);

    /* An impossible consumed position fails before any neural work and clears
     * stale availability, instead of underflowing ctx.pos-1. */
    m.ds.backoff_counter=0;stale(&m.ds);m.ctx.pos=0;
    assert(!myds_dspark_propose_next(&m.ds,&m.ctx,1,4,1));
    assert(eval_calls==backoff_calls&&m.ds.proposed_tokens==before);unpublished(&m.ds);
    stale(&m.ds);m.ctx.pos=23u;m.ctx.v41=NULL;
    assert(!myds_dspark_propose_next(&m.ds,&m.ctx,1,4,1));
    assert(eval_calls==backoff_calls&&m.ds.proposed_tokens==before);unpublished(&m.ds);
    puts("{\"kind\":\"host_dspark_proposal_contract\",\"legacy_first_slot\":0,\"engine_first_slot\":1,"
         "\"consumed_position_checked\":true,\"top2_checked\":true,\"real_markov_conditioning\":true,"
         "\"backoff_and_eval_failure_unpublished\":true,\"zero_position_guard\":true,"
         "\"neural_gpu_arithmetic_verified\":false,\"pass\":true}");
}
static void metadata_u32(FILE *file,uint32_t value) { assert(fwrite(&value,4,1,file)==1); }
static void metadata_u64(FILE *file,uint64_t value) { assert(fwrite(&value,8,1,file)==1); }
static void metadata_string(FILE *file,const char *value) {
    const size_t bytes=strlen(value);metadata_u64(file,bytes);assert(fwrite(value,1,bytes,file)==bytes);
}
static void metadata_rejection_contract(void) {
    /* Real GGUF parser and loader, with every other geometry value defaulting
     * to the supported checkpoint. Trap any Metal resource allocation: a
     * missing-weight failure later in loading cannot satisfy these cases. */
    static const struct {const char *key;uint32_t value;bool floating;} cases[]={
        {"dspark.markov_rank",128,false},{"dspark.n_routed_experts",384,false},
        {"dspark.block_size",4,false},{"dspark.stage_count",4,false},
        {"deepseek41-dflash-draft.attention.layer_norm_rms_epsilon",UINT32_C(0x7fc00000),true},
        {"deepseek41-dflash-draft.hyper_connection.epsilon",UINT32_C(0x7f800000),true},
        {"deepseek41-dflash-draft.expert_weights_scale",UINT32_C(0xff800000),true},
        {"deepseek41-dflash-draft.swiglu_clamp_exp",UINT32_C(0x7fc00000),true}
    };
    assert(!setenv("MYDS_DSPARK_CAPTURE_BYPASS","0",1));
    assert(!setenv("MYDS_ENGINE_CAPTURE_BYPASS","0",1));
    for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
        char path[]="/tmp/myds-dspark-metadata-XXXXXX";
        const int fd=mkstemp(path);assert(fd>=0);
        FILE *file=fdopen(fd,"wb");assert(file);
        metadata_u32(file,GGUF_MAGIC);metadata_u32(file,3);metadata_u64(file,0);metadata_u64(file,2);
        metadata_string(file,"general.architecture");metadata_u32(file,GGUF_TYPE_STRING);
        metadata_string(file,"deepseek41-dflash-draft");metadata_string(file,cases[i].key);
        metadata_u32(file,cases[i].floating?GGUF_TYPE_FLOAT32:GGUF_TYPE_UINT32);metadata_u32(file,cases[i].value);
        assert(!fclose(file));
        gguf_file_t *parsed=gguf_open(path);assert(parsed&&parsed->tensor_count==0&&gguf_find_key(parsed,cases[i].key));
        gguf_close(parsed);assert(!myds_dspark_load(path,129280u));assert(!unlink(path));
    }
    puts("{\"kind\":\"host_dspark_metadata_rejection_contract\",\"malformed_checkpoints\":8,"
         "\"real_gguf_parser_and_loader\":true,\"metal_resource_allocation\":false,\"pass\":true}");
}
int main(int argc,char **argv) {
    const bool markov_only=argc==2&&!strcmp(argv[1],"--markov-only");
    assert(argc==1||markov_only);environment(markov_only);
    if(markov_only)markov_only_contract();else {neural_contract();metadata_rejection_contract();}
    return 0;
}
