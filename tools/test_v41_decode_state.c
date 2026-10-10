/* Full-model Batch-2 equivalence/state gate, not a speed benchmark or an
 * independent model oracle. One loaded model and one context are reused.
 * Run in an isolated source directory; never starts or contacts a server.
 * MODEL TOKEN_FILE OUTPUT_DIR, with >=256 exact input IDs in TOKEN_FILE.
 * The existing pinned ds4 trace harness can replay the emitted input files.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_v41_gpu.h"
#include "myds_dspark.h"
#include "reference/trace_input.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DIM 5120u
static FILE *gate_report;
static bool compare_capture_slots = true;
static bool chunk_prefix = false;
/* Private diagnostic accessors borrow already committed graph storage. */
extern const float *myds_v41_diagnostic_current_residual(const myds_v41_context_t *g);
extern const float *myds_v41_diagnostic_current_pre(const myds_v41_context_t *g);
extern const float *myds_v41_diagnostic_tree_logits(const myds_context_t *ctx,uint32_t row);
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#c); return 0; } } while (0)

typedef struct {
    myds_v41_spec_snapshot_t *snapshot;
    float *logits, *residual;
    float pre[4];
    float slots[3][DIM];
    uint64_t window_hash;
} state_t;

static uint64_t bytes_hash(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = data;
    for (size_t i=0;i<n;++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static uint64_t windows_hash(myds_context_t *ctx) {
    uint64_t h=UINT64_C(14695981039346656037);
    const uint32_t visible=ctx->pos<128u?ctx->pos:128u;
    for (uint32_t il=0;il<MYDS_V41_LAYERS_COUNT;++il) {
        myds_metal_buffer_t *w=myds_v41_snapshot_get_window(ctx->v41,il);
        const float *values=myds_metal_buffer_data(w);
        for(uint32_t k=0;k<visible;++k)
            h=bytes_hash(h,values+((ctx->pos-visible+k)%128u)*512u,512u*sizeof(float));
    }
    return h;
}
static int write_f32(const char *dir,uint32_t pos,const char *name,const float *p,size_t n) {
    char path[4096];
    REQUIRE(snprintf(path,sizeof(path),"%s/%s_pos%u.f32",dir,name,pos)>0);
    FILE *fp=fopen(path,"wb"); REQUIRE(fp);
    const int ok=fwrite(p,sizeof(*p),n,fp)==n;
    REQUIRE(fclose(fp)==0 && ok);
    return 1;
}
static int equal_f32(const char *label,const float *ref,const float *candidate,size_t n) {
    size_t mismatches=0,first=0; double maxabs=0;
    for(size_t i=0;i<n;++i) {
        REQUIRE(isfinite(ref[i]) && isfinite(candidate[i]));
        if(memcmp(ref+i,candidate+i,sizeof(float))) {
            if(!mismatches) first=i;
            ++mismatches;
            double error=fabs((double)ref[i]-candidate[i]); if(error>maxabs)maxabs=error;
        }
    }
    fprintf(gate_report,"{\"kind\":\"comparison\",\"label\":\"%s\",\"elements\":%zu,\"mismatches\":%zu,\"first\":%zu,\"max_abs\":%.17g,\"pass\":%s}\n",
        label,n,mismatches,first,maxabs,mismatches?"false":"true"); fflush(gate_report);
    return mismatches==0;
}
static int32_t argmax_f32(const float *values,size_t n) {
    int32_t best=-1;
    for(size_t i=0;i<n;++i)if(isfinite(values[i]) && (best<0 || values[i]>values[best]))best=(int32_t)i;
    return best;
}
/* Failure diagnostics run only after the actual verifier has returned. They
 * copy completed buffers; they never project a missing lazy second head or
 * advance the target state, so a ret=1 remains an honest one-step result. */
static int failure_state(myds_context_t *ctx,const myds_v41_spec_snapshot_t *first,
    const state_t *reference,uint32_t pos,const char *dir) {
    myds_v41_spec_snapshot_t *now=calloc(1,sizeof(*now)); REQUIRE(now);
    const int saved=myds_v41_spec_snapshot_save(ctx,now,1);
    fprintf(gate_report,"{\"kind\":\"failure_state\",\"pos\":%u,\"snapshot_saved\":%s,"
        "\"ctx_pos\":%u,\"pcache_len\":%u,\"next_position\":%u,\"rng_state\":\"%" PRIu64 "\","
        "\"engram_position\":%u,\"window_hash\":\"%016" PRIx64 "\",\"reference_window_hash\":\"%016" PRIx64 "\","
        "\"carry_matches_reference_first\":%s,\"engram_matches_reference_first\":%s,\"first_state_valid\":%s}\n",
        pos,saved?"true":"false",ctx->pos,ctx->pcache_len,now->next_position,now->rng_state,
        now->engram_position,windows_hash(ctx),reference->window_hash,
        saved && !memcmp(now->previous_kv,reference->snapshot->previous_kv,sizeof(now->previous_kv)) &&
            !memcmp(now->previous_score,reference->snapshot->previous_score,sizeof(now->previous_score))?"true":"false",
        saved && !memcmp(&now->engram_history,&reference->snapshot->engram_history,sizeof(now->engram_history))?"true":"false",
        first->first_state_valid?"true":"false");fflush(gate_report);
    int ok=write_f32(dir,pos,"failure_current_logits",myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size);
    if(saved)ok=write_f32(dir,pos,"failure_current_previous_kv",&now->previous_kv[0][0],4u*512u) &&
        write_f32(dir,pos,"failure_current_previous_score",&now->previous_score[0][0],4u*512u) && ok;
    if(first->first_state_valid) {
        ok=write_f32(dir,pos,"failure_first_hidden",myds_metal_buffer_data(first->first_residual),4u*DIM) &&
            write_f32(dir,pos,"failure_first_pre",myds_metal_buffer_data(first->first_pre),4u) && ok;
        for(uint32_t t=0;t<3;++t)if(first->first_dspark_slots[t]) {
            char name[64];snprintf(name,sizeof(name),"failure_first_target_slot%u",t);
            ok=write_f32(dir,pos,name,myds_metal_buffer_data(first->first_dspark_slots[t]),DIM) && ok;
        }
    }
    free(now);return ok;
}
static int state_init(state_t *s,uint32_t vocab) {
    s->snapshot=calloc(1,sizeof(*s->snapshot));
    s->logits=malloc(vocab*sizeof(float)); s->residual=malloc(4u*DIM*sizeof(float));
    REQUIRE(s->snapshot && s->logits && s->residual); return 1;
}
static void state_free(state_t *s) { free(s->snapshot);free(s->logits);free(s->residual); }
static int state_capture(myds_context_t *ctx,state_t *s,myds_v41_spec_snapshot_t *before) {
    REQUIRE(myds_v41_spec_snapshot_capture_first(ctx,before));
    REQUIRE(myds_v41_spec_snapshot_save(ctx,s->snapshot,1));
    memcpy(s->logits,myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size*sizeof(float));
    memcpy(s->residual,myds_metal_buffer_data(before->first_residual),4u*DIM*sizeof(float));
    memcpy(s->pre,myds_metal_buffer_data(before->first_pre),sizeof(s->pre));
    for(uint32_t t=0;t<3;++t)
        memcpy(s->slots[t],myds_metal_buffer_data(ctx->dspark->target_slots[t]),DIM*sizeof(float));
    s->window_hash=windows_hash(ctx); return 1;
}
static int state_capture_current(myds_context_t *ctx,state_t *s,const float *actual_logits) {
    REQUIRE(actual_logits && myds_v41_spec_snapshot_save(ctx,s->snapshot,1));
    const float *residual=myds_v41_diagnostic_current_residual(ctx->v41);
    const float *pre=myds_v41_diagnostic_current_pre(ctx->v41);
    REQUIRE(residual && pre && myds_dspark_capture_ready(ctx->dspark));
    memcpy(s->logits,actual_logits,ctx->model->vocab_size*sizeof(float));
    memcpy(s->residual,residual,4u*DIM*sizeof(float));
    memcpy(s->pre,pre,sizeof(s->pre));
    for(uint32_t t=0;t<3;++t)
        memcpy(s->slots[t],myds_metal_buffer_data(ctx->dspark->target_slots[t]),DIM*sizeof(float));
    s->window_hash=windows_hash(ctx);return 1;
}
static int state_equal_logits(myds_context_t *ctx,const state_t *ref,
    const myds_v41_spec_snapshot_t *live_first,const char *label,const float *actual_logits) {
    myds_v41_spec_snapshot_t *now=calloc(1,sizeof(*now)); REQUIRE(now);
    const int saved=myds_v41_spec_snapshot_save(ctx,now,1);
    int ok=saved && ctx->pos==ref->snapshot->pos && ctx->pcache_len==ref->snapshot->pos &&
        now->next_position==ref->snapshot->next_position &&
        now->rng_state==ref->snapshot->rng_state && now->engram_position==ref->snapshot->engram_position &&
        !memcmp(&now->engram_history,&ref->snapshot->engram_history,sizeof(now->engram_history)) &&
        !memcmp(now->previous_kv,ref->snapshot->previous_kv,sizeof(now->previous_kv)) &&
        !memcmp(now->previous_score,ref->snapshot->previous_score,sizeof(now->previous_score)) &&
        windows_hash(ctx)==ref->window_hash;
    free(now);
    if(!ok)fprintf(stderr,"FAIL %s: window/carry/Engram/RNG/position differs\n",label);
    REQUIRE(ok);
    REQUIRE(actual_logits && equal_f32(label,ref->logits,actual_logits,ctx->model->vocab_size));
    if(compare_capture_slots)for(uint32_t t=0;t<3;++t)
        REQUIRE(equal_f32("dspark_target_slot",ref->slots[t],myds_metal_buffer_data(ctx->dspark->target_slots[t]),DIM));
    if(live_first) {
        REQUIRE(live_first->first_state_valid);
        REQUIRE(equal_f32("first_hidden",ref->residual,myds_metal_buffer_data(live_first->first_residual),4u*DIM));
        REQUIRE(equal_f32("first_pre",ref->pre,myds_metal_buffer_data(live_first->first_pre),4));
    }
    return 1;
}
static int state_equal(myds_context_t *ctx,const state_t *ref,
    const myds_v41_spec_snapshot_t *live_first,const char *label) {
    return state_equal_logits(ctx,ref,live_first,label,myds_metal_buffer_data(ctx->logits));
}
static int consume(myds_context_t *ctx,int32_t token,uint32_t pos,bool sample) {
    ctx->pos=pos; ctx->pcache_tokens[pos]=token; ctx->pcache_len=pos+1u;
    const int next=myds_v41_forward(ctx,token,pos,sample);
    if(next>=0)ctx->pos=pos+1u;
    return next;
}
/* The long gate uses the public chunk prefill to establish a consecutive real
 * target state. Scalar short gates retain their original token-by-token path. */
static int establish_prefix(myds_context_t *ctx,const int32_t *ids,uint32_t pos) {
    ctx->pos=0;ctx->pcache_len=0;
    if(chunk_prefix) {
        REQUIRE(myds_v41_prefill_tokens(ctx,ids,pos,NULL)==1);
        REQUIRE(ctx->pos==pos);
        memcpy(ctx->pcache_tokens,ids,pos*sizeof(*ids));ctx->pcache_len=pos;
    } else {
        for(uint32_t i=0;i<pos;++i)REQUIRE(consume(ctx,ids[i],i,false)>=0);
    }
    return 1;
}
static int input_file(const char *dir,uint32_t pos,const int32_t *ids,int32_t next) {
    char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s/inputs_pos%u.txt",dir,pos)>0);
    FILE *fp=fopen(path,"w"); REQUIRE(fp);
    for(uint32_t i=0;i<=pos;++i)fprintf(fp,"%d\n",ids[i]);
    fprintf(fp,"%d\n",next); REQUIRE(fclose(fp)==0); return 1;
}
static int run_position(myds_context_t *ctx,const int32_t *ids,uint32_t pos,const char *dir) {
    fprintf(stderr,"FULLMODEL_STATE_BEGIN pos=%u\n",pos);
    ctx->rng_state=42;ctx->pcache_len=0;ctx->decode_metrics_active=false;
    REQUIRE(establish_prefix(ctx,ids,pos));
    myds_v41_spec_snapshot_t *base=calloc(1,sizeof(*base)); REQUIRE(base);
    state_t first={0},second={0};
    REQUIRE(state_init(&first,ctx->model->vocab_size) && state_init(&second,ctx->model->vocab_size));
    ctx->pos=pos; ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    REQUIRE(myds_v41_spec_snapshot_save(ctx,base,2));
    const int32_t next0=consume(ctx,ids[pos],pos,true); REQUIRE(next0>=0);
    REQUIRE(state_capture(ctx,&first,base));
    const int32_t next1=consume(ctx,next0,pos+1u,true); REQUIRE(next1>=0);
    REQUIRE(state_capture(ctx,&second,first.snapshot));
    REQUIRE(write_f32(dir,pos,"reference_first",first.logits,ctx->model->vocab_size));
    REQUIRE(write_f32(dir,pos,"reference_second",second.logits,ctx->model->vocab_size));
    REQUIRE(input_file(dir,pos,ids,next0));
    const int32_t eos=ctx->model->tokenizer->eos_id;
    /* A natural first EOS is a distinct no-second case. Preserve its numerical
     * contract; controlled EOS checks below also exercise the stop branch. */
    if(next0==eos) { fprintf(stderr,"Unexpected natural first EOS at pos=%u; use another token stream\n",pos); return 0; }
    ctx->decode_metrics_active=true;memset(&ctx->decode_stats,0,sizeof(ctx->decode_stats));
    ctx->decode_pld_candidate=true;
    int32_t tokens[2]={ids[pos],next0},out[2]={-1,-1};
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    const int caller_auto_barrier=myds_metal_get_auto_barrier();
    const int hit_result=myds_v41_forward_batch_spec_ex(ctx,tokens,2,pos,out,base);
    const int first_available=base->first_state_valid && base->first_logits;
    fprintf(gate_report,"{\"kind\":\"batch2_result\",\"pos\":%u,\"ret\":%d,\"expected_ret\":2,"
        "\"out\":[%d,%d],\"expected_out\":[%d,%d],\"first_logits_available\":%s,"
        "\"first_argmax\":%d,\"second_logits_available\":%s,\"second_argmax\":%d,"
        "\"snapshot_valid\":%s,\"first_state_valid\":%s,\"caller_auto_barrier_restored\":%s}\n",
        pos,hit_result,out[0],out[1],next0,next1,first_available?"true":"false",
        first_available?argmax_f32(myds_metal_buffer_data(base->first_logits),ctx->model->vocab_size):-1,
        hit_result==2?"true":"false",hit_result==2?argmax_f32(myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size):-1,
        base->valid?"true":"false",base->first_state_valid?"true":"false",
        myds_metal_get_auto_barrier()==caller_auto_barrier?"true":"false");fflush(gate_report);
    if(first_available)REQUIRE(write_f32(dir,pos,"candidate_first",myds_metal_buffer_data(base->first_logits),ctx->model->vocab_size));
    if(hit_result==2)REQUIRE(write_f32(dir,pos,"candidate_second",myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size));
    const int first_equal=first_available && equal_f32("batch2_first_logits",first.logits,myds_metal_buffer_data(base->first_logits),ctx->model->vocab_size);
    if(hit_result!=2 || out[0]!=next0 || out[1]!=next1 || !first_equal) {
        (void)failure_state(ctx,base,&first,pos,dir);
        fprintf(stderr,"BATCH2_HIT_FAILURE pos=%u ret=%d out=[%d,%d] expected=[%d,%d] first_available=%d first_equal=%d second_available=%d\n",
            pos,hit_result,out[0],out[1],next0,next1,first_available,first_equal,hit_result==2);
    }
    REQUIRE(hit_result==2 && out[0]==next0 && out[1]==next1 && first_equal);
    REQUIRE(myds_metal_get_auto_barrier()==caller_auto_barrier);
    REQUIRE(ctx->decode_stats.batch2_hits==1 && ctx->decode_stats.batch2_calls==1);
    ctx->pcache_tokens[pos+1u]=next0;ctx->pcache_len=pos+2u; /* caller commit */
    REQUIRE(state_equal(ctx,&second,NULL,"batch2_hit_second"));
    /* Equivalent to caller discard-second after its first callback cancels. */
    REQUIRE(myds_v41_spec_snapshot_restore_partial(ctx,base,1));
    REQUIRE(state_equal(ctx,&first,base,"cancel_after_first_state"));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(state_equal(ctx,&second,NULL,"continue_after_cancel"));
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    tokens[1]=(next0+1u)%ctx->model->vocab_size;
    ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    REQUIRE(myds_v41_forward_batch_spec_ex(ctx,tokens,2,pos,out,base)==1 && out[0]==next0);
    REQUIRE(myds_metal_get_auto_barrier()==caller_auto_barrier);
    REQUIRE(ctx->decode_stats.batch2_misses==1);
    REQUIRE(state_equal(ctx,&first,base,"batch2_miss_first"));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(state_equal(ctx,&second,NULL,"continue_after_miss"));
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    int32_t rescue[3]={ids[pos],tokens[1],next0};
    ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    REQUIRE(myds_v41_forward_batch_spec_ex(ctx,rescue,3,pos,out,base)==2);
    REQUIRE(out[0]==next0 && out[1]==next1);
    ctx->pcache_tokens[pos+1u]=next0;ctx->pcache_len=pos+2u;
    REQUIRE(state_equal(ctx,&second,NULL,"dual_branch_rescue"));
    REQUIRE(myds_v41_spec_snapshot_restore_partial(ctx,base,1));
    REQUIRE(state_equal(ctx,&first,base,"rescue_discard_second"));
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    tokens[1]=next0;
    ctx->model->tokenizer->eos_id=next0; /* Controlled stop-token identity only. */
    ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    const int eos_result=myds_v41_forward_batch_spec_ex(ctx,tokens,2,pos,out,base);
    ctx->model->tokenizer->eos_id=eos;
    REQUIRE(eos_result==1 && out[0]==next0);
    REQUIRE(myds_metal_get_auto_barrier()==caller_auto_barrier);
    REQUIRE(state_equal(ctx,&first,base,"controlled_eos_first"));
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    /* Call 3 reports failure after the encoded full graph has executed;
     * dispatcher must restore and recover via the actual scalar verifier. */
    myds_metal_inject_commit_failure(3);
    const int fallback_result=myds_v41_forward_batch_spec_ex(ctx,tokens,2,pos,out,base);
    myds_metal_inject_commit_failure(0);
    REQUIRE(fallback_result==2 && out[0]==next0 && out[1]==next1);
    REQUIRE(myds_metal_get_auto_barrier()==caller_auto_barrier);
    REQUIRE(ctx->decode_stats.batch2_failures==1 && ctx->decode_stats.batch2_fallbacks==1);
    ctx->pcache_tokens[pos+1u]=next0;ctx->pcache_len=pos+2u;
    REQUIRE(state_equal(ctx,&second,NULL,"batch2_failure_scalar_fallback"));
    REQUIRE(myds_v41_spec_snapshot_restore_partial(ctx,base,1));
    REQUIRE(state_equal(ctx,&first,base,"serial_fallback_first_checkpoint"));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(state_equal(ctx,&second,NULL,"continue_after_fallback"));
    myds_metal_inject_commit_failure(1);
    const int restore_failed=myds_v41_spec_snapshot_restore_partial(ctx,base,1);
    myds_metal_inject_commit_failure(0);
    REQUIRE(!restore_failed);
    /* Equivalent to engine STOP_ERROR invalidation, then a fresh request.
     * A stale nonzero continuation must be rejected before a graph executes. */
    ctx->pcache_len=0;
    REQUIRE(myds_v41_forward(ctx,next0,pos+1u,true)<0);
    REQUIRE(establish_prefix(ctx,ids,pos));
    REQUIRE(consume(ctx,ids[pos],pos,true)==next0);
    REQUIRE(state_equal(ctx,&first,NULL,"fresh_request_after_restore_failure"));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(state_equal(ctx,&second,NULL,"fresh_request_second_after_failure"));
    fprintf(gate_report,"{\"kind\":\"position_complete\",\"pos\":%u,\"next0\":%d,\"next1\":%d,\"batch2_calls\":%" PRIu64 ",\"hits\":%" PRIu64 ",\"misses\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"fallbacks\":%" PRIu64 ",\"pass\":true}\n",
        pos,next0,next1,ctx->decode_stats.batch2_calls,ctx->decode_stats.batch2_hits,
        ctx->decode_stats.batch2_misses,ctx->decode_stats.batch2_failures,ctx->decode_stats.batch2_fallbacks);fflush(gate_report);
    state_free(&first);state_free(&second);free(base);return 1;
}

/* Two small capture fixtures share one real target context/model. The dummy
 * entry buffer exists only to exercise neural-consumer policy; no drafter
 * stages or support weights are provided, and no real-drafter claim is made. */
static myds_dspark_t *capture_fixture(const char *mode) {
    const char *old=getenv("MYDS_DSPARK_CAPTURE_BYPASS");
    char *saved=old?strdup(old):NULL;
    if(old&&!saved)return NULL;
    if(mode)setenv("MYDS_DSPARK_CAPTURE_BYPASS",mode,1);
    myds_dspark_t *ds=calloc(1,sizeof(*ds));
    const bool configured=ds&&myds_dspark_capture_configure(ds);
    if(saved){setenv("MYDS_DSPARK_CAPTURE_BYPASS",saved,1);free(saved);}
    else unsetenv("MYDS_DSPARK_CAPTURE_BYPASS");
    if(!configured){free(ds);return NULL;}
    ds->target_hidden_buf=myds_metal_buffer_alloc(3u*DIM*sizeof(float));
    ds->hc_mean_weights=myds_metal_buffer_alloc(4u*sizeof(float));
    ds->main_proj=myds_metal_buffer_alloc(sizeof(float));
    const float mean[4]={.25f,.25f,.25f,.25f};
    if(!ds->target_hidden_buf||!ds->hc_mean_weights||!ds->main_proj||
       !myds_metal_buffer_write(ds->hc_mean_weights,0,mean,sizeof(mean)))goto fail;
    for(uint32_t i=0;i<3;i++) {
        ds->target_layers[i]=37u+i;
        ds->target_slots[i]=myds_metal_buffer_sub_view(ds->target_hidden_buf,i*DIM*sizeof(float),DIM*sizeof(float));
        if(!ds->target_slots[i])goto fail;
    }
    return ds;
fail:
    myds_dspark_free(ds);return NULL;
}
static int run_capture_position(myds_context_t *ctx,myds_dspark_t *reference,
    myds_dspark_t *candidate,const int32_t *ids,uint32_t pos,const char *dir) {
    fprintf(stderr,"CAPTURE_STATE_BEGIN pos=%u\n",pos);
    state_t first={0},second={0};
    myds_v41_spec_snapshot_t *base=calloc(1,sizeof(*base));REQUIRE(base);
    REQUIRE(state_init(&first,ctx->model->vocab_size)&&state_init(&second,ctx->model->vocab_size));
    ctx->dspark=reference;ctx->rng_state=42;ctx->pcache_len=0;
    for(uint32_t i=0;i<pos;i++)REQUIRE(consume(ctx,ids[i],i,false)>=0);
    REQUIRE(myds_v41_spec_snapshot_save(ctx,base,2));
    const int32_t next0=consume(ctx,ids[pos],pos,true);REQUIRE(next0>=0);
    REQUIRE(state_capture(ctx,&first,base));
    REQUIRE(write_f32(dir,pos,"capture_reference_slots",myds_metal_buffer_data(reference->target_hidden_buf),3u*DIM));
    const int32_t next1=consume(ctx,next0,pos+1u,true);REQUIRE(next1>=0);
    REQUIRE(state_capture(ctx,&second,first.snapshot));
    REQUIRE(write_f32(dir,pos,"reference_first",first.logits,ctx->model->vocab_size));
    REQUIRE(write_f32(dir,pos,"reference_second",second.logits,ctx->model->vocab_size));
    REQUIRE(input_file(dir,pos,ids,next0));

    ctx->dspark=candidate;
    float *poison=myds_metal_buffer_data(candidate->target_hidden_buf);
    REQUIRE(poison);for(uint32_t i=0;i<3u*DIM;i++)poison[i]=123.125f;
    myds_dspark_capture_counters(candidate,NULL,true);
    myds_dspark_capture_begin_request(candidate,false);
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    compare_capture_slots=false;
    REQUIRE(consume(ctx,ids[pos],pos,true)==next0);
    REQUIRE(myds_v41_spec_snapshot_capture_first(ctx,base));
    REQUIRE(state_equal(ctx,&first,base,"capture_disabled_first"));
    REQUIRE(write_f32(dir,pos,"candidate_first",myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(myds_v41_spec_snapshot_capture_first(ctx,first.snapshot));
    REQUIRE(state_equal(ctx,&second,first.snapshot,"capture_disabled_second"));
    REQUIRE(write_f32(dir,pos,"candidate_second",myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size));
    REQUIRE(!myds_dspark_capture_ready(candidate));
    REQUIRE(!myds_dspark_eval_stages(candidate,ctx,.8f,NULL));
    for(uint32_t i=0;i<3u*DIM;i++)REQUIRE(poison[i]==123.125f);
    REQUIRE(write_f32(dir,pos,"capture_disabled_slots",poison,3u*DIM));
    myds_dspark_capture_counters_t bypass={0};
    myds_dspark_capture_counters(candidate,&bypass,false);
    REQUIRE(bypass.requests==6&&bypass.bypassed==6&&!bypass.encoded&&!bypass.failures&&bypass.stale_rejections==1);
    myds_dspark_capture_end_request(candidate,false); /* cancelled/failed request */
    REQUIRE(myds_dspark_capture_required(candidate)&&!myds_dspark_capture_ready(candidate));

    /* Prefix/whole restore does not contain slot bytes: the first new target
     * step must repopulate them before a proposal, just as engine step0 does. */
    myds_dspark_capture_begin_request(candidate,true);
    REQUIRE(!myds_dspark_capture_ready(candidate));
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    REQUIRE(consume(ctx,ids[pos],pos,true)==next0);
    REQUIRE(myds_dspark_capture_ready(candidate)&&candidate->capture_committed);
    compare_capture_slots=true;
    REQUIRE(myds_v41_spec_snapshot_capture_first(ctx,base));
    REQUIRE(state_equal(ctx,&first,base,"capture_reenabled_first"));
    REQUIRE(write_f32(dir,pos,"capture_enabled_slots",myds_metal_buffer_data(candidate->target_hidden_buf),3u*DIM));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(state_equal(ctx,&second,NULL,"capture_reenabled_second"));
    REQUIRE(myds_v41_spec_snapshot_restore_partial(ctx,base,1));
    REQUIRE(myds_dspark_capture_ready(candidate));
    REQUIRE(state_equal(ctx,&first,base,"capture_cancel_keep_first"));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(state_equal(ctx,&second,NULL,"capture_continue_after_cancel"));
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    REQUIRE(!myds_dspark_capture_ready(candidate));
    REQUIRE(!myds_dspark_eval_stages(candidate,ctx,.8f,NULL));
    REQUIRE(consume(ctx,ids[pos],pos,true)==next0);
    /* Prewrite target error: this is not an injected GPU completion failure. */
    REQUIRE(myds_v41_forward(ctx,-1,pos+1u,true)<0);
    REQUIRE(!myds_dspark_capture_ready(candidate));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(myds_dspark_capture_ready(candidate));
    REQUIRE(state_equal(ctx,&second,NULL,"capture_after_prewrite_error"));
    myds_dspark_capture_counters_t total={0};
    myds_dspark_capture_counters(candidate,&total,false);
    REQUIRE(total.requests==21&&total.encoded==15&&total.bypassed==6&&
        !total.failures&&total.stale_rejections==2);
    fprintf(gate_report,"{\"kind\":\"capture_protocol\",\"pos\":%u,\"reference_mode\":0,\"candidate_mode\":1,"
        "\"bypass_requests\":6,\"bypass_encoded\":0,\"slots_untouched\":true,\"reenabled_slots_byte_exact\":true,"
        "\"partial_first_ready\":true,\"whole_rollback_stale_rejected\":true,\"prewrite_error_recovered\":true,"
        "\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"bypassed\":%" PRIu64 ",\"failures\":%" PRIu64 ","
        "\"stale_rejections\":%" PRIu64 ",\"real_drafter\":false,\"pass\":true}\n",
        pos,total.requests,total.encoded,total.bypassed,total.failures,total.stale_rejections);fflush(gate_report);
    myds_dspark_capture_end_request(candidate,true);
    ctx->dspark=candidate;state_free(&first);state_free(&second);free(base);return 1;
}
/* Tree is an independent legacy public graph. Compare its unchanged mode0
 * graph to mode1, rather than implying that its arithmetic is the scalar
 * oracle. The exported state contains slots, residual and pre; the separately
 * exported heads come from the actual tree workspace, never ctx->logits. */
static int export_tree_capture_state(const char *dir,uint32_t pos,const char *name,const state_t *s) {
    const size_t n=7u*DIM+4u;
    float *values=malloc(n*sizeof(*values));REQUIRE(values);
    memcpy(values,&s->slots[0][0],3u*DIM*sizeof(float));
    memcpy(values+3u*DIM,s->residual,4u*DIM*sizeof(float));
    memcpy(values+7u*DIM,s->pre,4u*sizeof(float));
    const int ok=write_f32(dir,pos,name,values,n);free(values);return ok;
}
static int run_capture_tree_position(myds_context_t *ctx,myds_dspark_t *reference,
    myds_dspark_t *candidate,const int32_t *ids,uint32_t pos,const char *dir) {
    fprintf(stderr,"CAPTURE_TREE_STATE_BEGIN pos=%u\n",pos);
    myds_v41_spec_snapshot_t *base=calloc(1,sizeof(*base));REQUIRE(base);
    state_t scalar={0},tree_reference={0},tree_candidate={0};
    REQUIRE(state_init(&scalar,ctx->model->vocab_size)&&state_init(&tree_reference,ctx->model->vocab_size)&&
        state_init(&tree_candidate,ctx->model->vocab_size));
    float *reference_heads[2]={malloc(ctx->model->vocab_size*sizeof(float)),
                              malloc(ctx->model->vocab_size*sizeof(float))};
    REQUIRE(reference_heads[0]&&reference_heads[1]);
    ctx->dspark=reference;ctx->rng_state=42;ctx->decode_metrics_active=false;
    REQUIRE(establish_prefix(ctx,ids,pos));
    REQUIRE(myds_v41_spec_snapshot_save(ctx,base,2));
    const int32_t scalar_next=consume(ctx,ids[pos],pos,true);REQUIRE(scalar_next>=0);
    REQUIRE(state_capture(ctx,&scalar,base));
    REQUIRE(myds_v41_spec_snapshot_restore(ctx,base));
    ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    int32_t probe[3]={ids[pos],scalar_next,(int32_t)(((uint32_t)scalar_next+1u)%ctx->model->vocab_size)},out[2]={-1,-1};
    REQUIRE(myds_v41_forward_tree_spec(ctx,probe,pos,out)>0);
    const int32_t tree_next=out[0];REQUIRE(tree_next>=0&&(uint32_t)tree_next<ctx->model->vocab_size);
    const int32_t wrong0=(int32_t)(((uint32_t)tree_next+1u)%ctx->model->vocab_size);
    const int32_t wrong1=(int32_t)(((uint32_t)tree_next+2u)%ctx->model->vocab_size);
    const int32_t branches[3][3]={{ids[pos],tree_next,wrong0},{ids[pos],wrong0,tree_next},{ids[pos],wrong0,wrong1}};
    const char *names[3]={"A","B","miss"};
    const int expected[3]={2,2,1};
    const int caller_auto=myds_metal_get_auto_barrier();
    for(uint32_t b=0;b<3;b++) {
        ctx->dspark=reference;REQUIRE(myds_v41_spec_snapshot_restore_partial(ctx,base,0));
        ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
        int32_t reference_out[2]={-1,-1};
        const int reference_ret=myds_v41_forward_tree_spec(ctx,branches[b],pos,reference_out);
        const int reference_barrier=myds_metal_get_auto_barrier();
        const bool reference_matches=reference_ret==expected[b]&&reference_out[0]==tree_next&&
            reference_barrier==caller_auto;
        if(!reference_matches) {
            /* The probe and this first reference both use mode0. Diagnose
             * their disagreement before any candidate executes, retaining
             * the original acceptance condition and immediate failure. */
            int computed_second=-1;
            if(reference_ret==2) {
                if(reference_out[0]==branches[b][1])computed_second=1;
                else if(reference_out[0]==branches[b][2])computed_second=2;
            }
            const uint32_t computed_count=reference_ret==1?1u:
                reference_ret==2?(computed_second>0?2u:1u):0u;
            char diagnostic_dir[4096],diagnostic_name[96];
            const int diagnostic_len=snprintf(diagnostic_dir,sizeof(diagnostic_dir),"%s/failure-diagnostics",dir);
            const bool diagnostic_dir_ready=diagnostic_len>0&&(size_t)diagnostic_len<sizeof(diagnostic_dir)&&
                (mkdir(diagnostic_dir,0700)==0||errno==EEXIST);
            bool exported=diagnostic_dir_ready&&(reference_ret!=2||computed_second>0);
            uint32_t exported_count=0;
            int32_t diagnostic_argmax[2]={-1,-1};
            for(uint32_t h=0;h<computed_count;++h) {
                const uint32_t row=h?(uint32_t)computed_second:0u;
                const float *actual=myds_v41_diagnostic_tree_logits(ctx,row);
                if(!actual){exported=false;continue;}
                diagnostic_argmax[h]=argmax_f32(actual,ctx->model->vocab_size);
                snprintf(diagnostic_name,sizeof(diagnostic_name),"failure_tree_%s_reference_logits%u",names[b],row);
                if(diagnostic_dir_ready&&write_f32(diagnostic_dir,pos,diagnostic_name,actual,ctx->model->vocab_size))
                    ++exported_count;
                else exported=false;
            }
            myds_dspark_capture_counters_t counter={0};
            myds_dspark_capture_counters(reference,&counter,false);
            fprintf(stderr,"CAPTURE_TREE_REFERENCE_FAILURE pos=%u branch=%s expected_ret=%d probe_next=%d "
                "reference_ret=%d reference_out=[%d,%d] caller_barrier=%d actual_barrier=%d "
                "ready=%d required=%d mode=%d disabled=%d mask=%u committed=%d computed_heads=%u exported_heads=%u\n",
                pos,names[b],expected[b],tree_next,reference_ret,reference_out[0],reference_out[1],
                caller_auto,reference_barrier,myds_dspark_capture_ready(reference),myds_dspark_capture_required(reference),
                myds_dspark_capture_bypass_enabled(reference),reference->capture_disabled,reference->capture_valid_mask,
                reference->capture_committed,computed_count,exported_count);
            fprintf(gate_report,"{\"kind\":\"capture_tree_reference_failure_diagnostic\",\"pos\":%u,\"branch\":\"%s\","
                "\"expected_ret\":%d,\"probe_next\":%d,\"reference_ret\":%d,\"reference_out\":[%d,%d],"
                "\"caller_auto_barrier\":%d,\"reference_auto_barrier\":%d,\"ready\":%s,\"required\":%s,"
                "\"reference_mode\":%d,\"capture_disabled\":%s,\"capture_valid_mask\":%u,\"capture_committed\":%s,"
                "\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"bypassed\":%" PRIu64 ",\"failures\":%" PRIu64 ","
                "\"stale_rejections\":%" PRIu64 ",\"computed_heads_proven\":%u,\"computed_heads_exported\":%u,"
                "\"selected_workspace_head\":%d,\"diagnostic_argmax\":[%d,%d],\"diagnostic_exports_complete\":%s,"
                "\"candidate_started\":false,\"original_failure_retained\":true,\"pass\":false}\n",
                pos,names[b],expected[b],tree_next,reference_ret,reference_out[0],reference_out[1],caller_auto,reference_barrier,
                myds_dspark_capture_ready(reference)?"true":"false",myds_dspark_capture_required(reference)?"true":"false",
                myds_dspark_capture_bypass_enabled(reference),reference->capture_disabled?"true":"false",
                reference->capture_valid_mask,reference->capture_committed?"true":"false",counter.requests,counter.encoded,
                counter.bypassed,counter.failures,counter.stale_rejections,computed_count,exported_count,computed_second,
                diagnostic_argmax[0],diagnostic_argmax[1],exported?"true":"false");
            fflush(gate_report);fflush(stderr);
        }
        REQUIRE(reference_matches);
        if(reference_ret==2){ctx->pcache_tokens[pos+1u]=tree_next;ctx->pcache_len=pos+2u;}
        const uint32_t selected_head=b<2u?b+1u:0u;
        const uint32_t head_indices[2]={0u,selected_head};
        const uint32_t head_count=reference_ret==2?2u:1u;
        REQUIRE(state_capture_current(ctx,&tree_reference,myds_v41_diagnostic_tree_logits(ctx,selected_head)));
        char name[96];snprintf(name,sizeof(name),"tree_%s_reference_capture_state",names[b]);
        REQUIRE(export_tree_capture_state(dir,pos,name,&tree_reference));
        int32_t head_argmax[2]={-1,-1};
        for(uint32_t h=0;h<head_count;++h) {
            const float *actual=myds_v41_diagnostic_tree_logits(ctx,head_indices[h]);REQUIRE(actual);
            memcpy(reference_heads[h],actual,ctx->model->vocab_size*sizeof(float));
            head_argmax[h]=argmax_f32(actual,ctx->model->vocab_size);
            REQUIRE(head_argmax[h]==reference_out[h]);
            snprintf(name,sizeof(name),"tree_%s_reference_logits%u",names[b],head_indices[h]);
            REQUIRE(write_f32(dir,pos,name,reference_heads[h],ctx->model->vocab_size));
        }
        ctx->dspark=candidate;REQUIRE(myds_v41_spec_snapshot_restore_partial(ctx,base,0));
        REQUIRE(!myds_dspark_capture_ready(candidate));
        ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
        int32_t candidate_out[2]={-1,-1};
        const int candidate_ret=myds_v41_forward_tree_spec(ctx,branches[b],pos,candidate_out);
        const int candidate_barrier=myds_metal_get_auto_barrier();
        const bool candidate_ready=myds_dspark_capture_ready(candidate);
        const bool tree_matches=candidate_ret==reference_ret&&
            !memcmp(reference_out,candidate_out,sizeof(reference_out))&&
            candidate_barrier==caller_auto&&candidate_ready;
        if(!tree_matches) {
            /* Record the actual failing values before cleanup/restore changes
             * freshness. Export only heads proven computed by a successful
             * return: ret1 means head0; ret2 identifies its selected branch.
             * A failed graph may have rolled back, so its workspace is not
             * claimed as completed diagnostic logits. */
            myds_dspark_capture_counters_t counter={0};
            myds_dspark_capture_counters(candidate,&counter,false);
            fprintf(stderr,"CAPTURE_TREE_FAILURE pos=%u branch=%s expected_ret=%d reference_ret=%d candidate_ret=%d "
                "reference_out=[%d,%d] candidate_out=[%d,%d] caller_barrier=%d actual_barrier=%d "
                "ready=%d required=%d mode=%d disabled=%d mask=%u committed=%d next_position=%u ctx_pos=%u\n",
                pos,names[b],expected[b],reference_ret,candidate_ret,reference_out[0],reference_out[1],
                candidate_out[0],candidate_out[1],caller_auto,candidate_barrier,candidate_ready,
                myds_dspark_capture_required(candidate),myds_dspark_capture_bypass_enabled(candidate),
                candidate->capture_disabled,candidate->capture_valid_mask,candidate->capture_committed,
                myds_v41_snapshot_get_next_position(ctx->v41),ctx->pos);
            int computed_second=-1;
            if(candidate_ret==2) {
                if(candidate_out[0]==branches[b][1])computed_second=1;
                else if(candidate_out[0]==branches[b][2])computed_second=2;
            }
            const uint32_t diagnostic_count=candidate_ret==1?1u:
                candidate_ret==2?(computed_second>0?2u:1u):0u;
            bool exported=true;
            int32_t diagnostic_argmax[2]={-1,-1};
            char diagnostic_dir[4096];
            const bool diagnostic_dir_ready=snprintf(diagnostic_dir,sizeof(diagnostic_dir),
                "%s/failure-diagnostics",dir)>0 &&
                (mkdir(diagnostic_dir,0700)==0||errno==EEXIST);
            if(!diagnostic_dir_ready)exported=false;
            for(uint32_t h=0;h<diagnostic_count;++h) {
                const uint32_t row=h?(uint32_t)computed_second:0u;
                const float *actual=myds_v41_diagnostic_tree_logits(ctx,row);
                if(!actual){exported=false;continue;}
                diagnostic_argmax[h]=argmax_f32(actual,ctx->model->vocab_size);
                snprintf(name,sizeof(name),"failure_tree_%s_candidate_logits%u",names[b],row);
                if(!diagnostic_dir_ready||!write_f32(diagnostic_dir,pos,name,actual,ctx->model->vocab_size))exported=false;
            }
            fprintf(gate_report,"{\"kind\":\"capture_tree_failure_diagnostic\",\"pos\":%u,\"branch\":\"%s\","
                "\"expected_ret\":%d,\"reference_ret\":%d,\"candidate_ret\":%d,"
                "\"reference_out\":[%d,%d],\"candidate_out\":[%d,%d],\"caller_auto_barrier\":%d,\"candidate_auto_barrier\":%d,"
                "\"ready\":%s,\"required\":%s,\"candidate_mode\":%d,\"capture_disabled\":%s,"
                "\"capture_valid_mask\":%u,\"capture_committed\":%s,\"next_position\":%u,\"ctx_pos\":%u,"
                "\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"bypassed\":%" PRIu64 ",\"failures\":%" PRIu64 ","
                "\"stale_rejections\":%" PRIu64 ",\"computed_heads_exported\":%u,\"selected_workspace_head\":%d,"
                "\"diagnostic_argmax\":[%d,%d],\"diagnostic_exports_complete\":%s,\"pass\":false}\n",
                pos,names[b],expected[b],reference_ret,candidate_ret,reference_out[0],reference_out[1],
                candidate_out[0],candidate_out[1],caller_auto,candidate_barrier,candidate_ready?"true":"false",
                myds_dspark_capture_required(candidate)?"true":"false",myds_dspark_capture_bypass_enabled(candidate),
                candidate->capture_disabled?"true":"false",candidate->capture_valid_mask,
                candidate->capture_committed?"true":"false",myds_v41_snapshot_get_next_position(ctx->v41),ctx->pos,
                counter.requests,counter.encoded,counter.bypassed,counter.failures,counter.stale_rejections,
                diagnostic_count,computed_second,diagnostic_argmax[0],diagnostic_argmax[1],exported?"true":"false");
            fflush(gate_report);fflush(stderr);
            /* One same-mode0 repeat of this very branch separates an unstable
             * legacy tree graph from a mode0/mode1 difference. It is evidence
             * gathering after failure, never a replacement acceptance arm.
             * The original tree_matches value remains false below. */
            ctx->dspark=reference;
            const bool repeat_restored=myds_v41_spec_snapshot_restore_partial(ctx,base,0);
            int repeat_ret=-1,repeat_barrier=myds_metal_get_auto_barrier();
            int32_t repeat_out[2]={-1,-1};
            if(repeat_restored) {
                ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
                repeat_ret=myds_v41_forward_tree_spec(ctx,branches[b],pos,repeat_out);
                repeat_barrier=myds_metal_get_auto_barrier();
            }
            int repeat_second=-1;
            if(repeat_ret==2) {
                if(repeat_out[0]==branches[b][1])repeat_second=1;
                else if(repeat_out[0]==branches[b][2])repeat_second=2;
            }
            const uint32_t repeat_count=repeat_ret==1?1u:
                repeat_ret==2?(repeat_second>0?2u:1u):0u;
            bool repeat_exports=diagnostic_dir_ready,repeat_heads_equal=repeat_count==head_count;
            uint32_t heads_compared=0;
            int32_t repeat_argmax[2]={-1,-1};
            for(uint32_t h=0;h<repeat_count;++h) {
                const uint32_t row=h?(uint32_t)repeat_second:0u;
                const float *actual=myds_v41_diagnostic_tree_logits(ctx,row);
                if(!actual){repeat_exports=false;repeat_heads_equal=false;continue;}
                repeat_argmax[h]=argmax_f32(actual,ctx->model->vocab_size);
                snprintf(name,sizeof(name),"repeat_mode0_tree_%s_logits%u",names[b],row);
                if(!diagnostic_dir_ready||!write_f32(diagnostic_dir,pos,name,actual,ctx->model->vocab_size))repeat_exports=false;
                if(row==0u||row==selected_head) {
                    const float *original=reference_heads[row==0u?0u:1u];
                    const bool same=equal_f32("capture_tree_failure_repeat_mode0_logits",original,actual,ctx->model->vocab_size);
                    repeat_heads_equal=repeat_heads_equal&&same;++heads_compared;
                } else repeat_heads_equal=false; /* Original lazy head was not computed. */
            }
            const bool repeat_outputs_equal=repeat_ret==reference_ret&&
                !memcmp(reference_out,repeat_out,sizeof(reference_out));
            fprintf(stderr,"CAPTURE_TREE_REPEAT_MODE0 pos=%u branch=%s restore=%d reference_ret=%d repeat_ret=%d "
                "reference_out=[%d,%d] repeat_out=[%d,%d] compared_heads=%u heads_equal=%d original_failure_retained=1\n",
                pos,names[b],repeat_restored,reference_ret,repeat_ret,reference_out[0],reference_out[1],
                repeat_out[0],repeat_out[1],heads_compared,repeat_heads_equal);
            fprintf(gate_report,"{\"kind\":\"capture_tree_failure_repeat_control\",\"pos\":%u,\"branch\":\"%s\","
                "\"mode\":0,\"restored_base\":%s,\"reference_ret\":%d,\"repeat_ret\":%d,"
                "\"reference_out\":[%d,%d],\"repeat_out\":[%d,%d],\"repeat_argmax\":[%d,%d],"
                "\"caller_auto_barrier\":%d,\"repeat_auto_barrier\":%d,\"computed_heads_exported\":%u,"
                "\"selected_workspace_head\":%d,\"heads_compared\":%u,\"full_computed_logits_byte_exact\":%s,"
                "\"outputs_equal\":%s,\"diagnostic_exports_complete\":%s,"
                "\"original_condition_failed\":true,\"original_failure_retained\":true,\"pass\":false}\n",
                pos,names[b],repeat_restored?"true":"false",reference_ret,repeat_ret,reference_out[0],reference_out[1],
                repeat_out[0],repeat_out[1],repeat_argmax[0],repeat_argmax[1],caller_auto,repeat_barrier,repeat_count,
                repeat_second,heads_compared,repeat_heads_equal?"true":"false",repeat_outputs_equal?"true":"false",
                repeat_exports?"true":"false");
            fflush(gate_report);fflush(stderr);
        }
        REQUIRE(tree_matches);
        if(candidate_ret==2){ctx->pcache_tokens[pos+1u]=tree_next;ctx->pcache_len=pos+2u;}
        REQUIRE(state_capture_current(ctx,&tree_candidate,myds_v41_diagnostic_tree_logits(ctx,selected_head)));
        REQUIRE(state_equal_logits(ctx,&tree_reference,NULL,"capture_tree_same_graph_state",tree_candidate.logits));
        REQUIRE(equal_f32("capture_tree_same_graph_hidden",tree_reference.residual,tree_candidate.residual,4u*DIM));
        REQUIRE(equal_f32("capture_tree_same_graph_pre",tree_reference.pre,tree_candidate.pre,4u));
        snprintf(name,sizeof(name),"tree_%s_candidate_capture_state",names[b]);
        REQUIRE(export_tree_capture_state(dir,pos,name,&tree_candidate));
        for(uint32_t h=0;h<head_count;++h) {
            const float *actual=myds_v41_diagnostic_tree_logits(ctx,head_indices[h]);REQUIRE(actual);
            REQUIRE(argmax_f32(actual,ctx->model->vocab_size)==candidate_out[h]);
            REQUIRE(equal_f32("capture_tree_same_graph_logits",reference_heads[h],actual,ctx->model->vocab_size));
            snprintf(name,sizeof(name),"tree_%s_candidate_logits%u",names[b],head_indices[h]);
            REQUIRE(write_f32(dir,pos,name,actual,ctx->model->vocab_size));
        }
        char argmax_json[64],out_json[64];
        if(head_count==2)snprintf(argmax_json,sizeof(argmax_json),"[%d,%d]",head_argmax[0],head_argmax[1]);
        else snprintf(argmax_json,sizeof(argmax_json),"[%d]",head_argmax[0]);
        snprintf(out_json,sizeof(out_json),"[%d,%d]",reference_out[0],reference_out[1]);
        fprintf(gate_report,"{\"kind\":\"capture_tree_branch\",\"pos\":%u,\"branch\":\"%s\",\"reference_ret\":%d,"
            "\"candidate_ret\":%d,\"tree_next\":%d,\"scalar_next\":%d,\"same_graph_capture_state_byte_exact\":true,"
            "\"scalar_tree_equivalence_claimed\":false,\"selected_head_index\":%u,\"computed_head_indices\":%s,"
            "\"head_argmax\":%s,\"reference_out\":%s,\"candidate_out\":%s,"
            "\"full_computed_logits_byte_exact\":true,\"current_state_capture\":true,"
            "\"ctx_logits_used\":false,\"ready\":true,\"pass\":true}\n",pos,names[b],reference_ret,candidate_ret,tree_next,scalar_next,
            selected_head,b==0?"[0,1]":b==1?"[0,2]":"[0]",argmax_json,out_json,out_json);
    }
    REQUIRE(myds_v41_spec_snapshot_restore_partial(ctx,base,0));
    ctx->pcache_tokens[pos]=ids[pos];ctx->pcache_len=pos+1u;
    myds_metal_inject_commit_failure(3);
    const int failed=myds_v41_forward_tree_spec(ctx,branches[0],pos,out);
    myds_metal_inject_commit_failure(0);
    REQUIRE(failed<0&&!myds_dspark_capture_ready(candidate)&&myds_metal_get_auto_barrier()==caller_auto);
    REQUIRE(consume(ctx,ids[pos],pos,true)==scalar_next);
    REQUIRE(myds_dspark_capture_ready(candidate)&&state_equal(ctx,&scalar,NULL,"capture_tree_failure_scalar_retry"));
    const float *retry_residual=myds_v41_diagnostic_current_residual(ctx->v41);
    const float *retry_pre=myds_v41_diagnostic_current_pre(ctx->v41);
    REQUIRE(retry_residual&&retry_pre);
    REQUIRE(equal_f32("capture_tree_failure_scalar_retry_hidden",scalar.residual,retry_residual,4u*DIM));
    REQUIRE(equal_f32("capture_tree_failure_scalar_retry_pre",scalar.pre,retry_pre,4u));
    fprintf(gate_report,"{\"kind\":\"capture_tree_complete\",\"pos\":%u,\"branches_completed\":3,"
        "\"encoded_graph_failure_invalidated\":true,\"scalar_retry_byte_exact\":true,\"pass\":true}\n",pos);
    state_free(&scalar);state_free(&tree_reference);state_free(&tree_candidate);
    free(reference_heads[0]);free(reference_heads[1]);free(base);return 1;
}
int main(int argc,char **argv) {
    const bool capture_only=argc==5&&!strcmp(argv[4],"--capture-bypass-only");
    const bool long_state=argc==5&&!strcmp(argv[4],"--long-indexer-state");
    if(argc!=4&&!capture_only&&!long_state) {fprintf(stderr,"Usage: %s MODEL TOKEN_FILE OUTPUT_DIR [--capture-bypass-only | --long-indexer-state]\n",argv[0]);return 2;}
    if(capture_only&&(!getenv("MYDS_DSPARK_CAPTURE_BYPASS")||strcmp(getenv("MYDS_DSPARK_CAPTURE_BYPASS"),"1")))return 2;
    const char *shuffle_env=getenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE");
    const bool expected_shuffle=shuffle_env&&!strcmp(shuffle_env,"1");
    if(long_state&&(!shuffle_env||(strcmp(shuffle_env,"0")&&strcmp(shuffle_env,"1"))||
        !getenv("MYDS_INDEXER_TOPK_PREFIX_MERGE")||strcmp(getenv("MYDS_INDEXER_TOPK_PREFIX_MERGE"),"1")||
        !getenv("MYDS_INDEXER_TOPK_NAN_FUSE")||strcmp(getenv("MYDS_INDEXER_TOPK_NAN_FUSE"),"1")))return 2;
    chunk_prefix=long_state;
    trace_tokens ids={0}; char *parsed_argv[]={argv[0],argv[1],"--tokens-file",argv[2]};
    if(!trace_tokens_parse(&ids,4,parsed_argv)||ids.count<(long_state?16386u:256u)) {free(ids.ids);return 2;}
    if(mkdir(argv[3],0700)&&(capture_only||long_state||errno!=EEXIST)) {free(ids.ids);return 2;}
    /* Explicit, isolated greedy arm. Avoid changing any running service. */
    setenv("MYDS_BATCH2_SPEC","1",1);setenv("MYDS_PREFIX_CACHE","0",1);setenv("MYDS_KV_DSK","0",1);
    setenv("MYDS_SPEC_SERIAL_PROBE","0",1);
    const char *disabled_by_default[]={"MYDS_QKV_CONCURRENT","MYDS_MOE_CONCURRENT",
        "MYDS_METAL_CONCURRENT","MYDS_INDEXER_COMPACT_SCORER"};
    for(size_t i=0;i<sizeof(disabled_by_default)/sizeof(*disabled_by_default);++i)
        if(!getenv(disabled_by_default[i]))setenv(disabled_by_default[i],"0",1);
    const char *forbidden[]={"MYDS_TRACE_PREFIX","MYDS_BATCH_TRACE_PREFIX","MYDS_SKIP_Q_A",
        "MYDS_SKIP_KV","MYDS_SKIP_NORM","MYDS_SKIP_KV_ROPE","MYDS_SKIP_KV_QUANT",
        "MYDS_SKIP_Q_B","MYDS_SKIP_Q_ROPE"};
    for(size_t i=0;i<sizeof(forbidden)/sizeof(*forbidden);++i)if(getenv(forbidden[i])) {
        fprintf(stderr,"Unset %s for production-path parity\n",forbidden[i]);free(ids.ids);return 2;
    }
    /* Keep the machine-readable gate report separate from engine stdout logs. */
    fflush(stdout);const int report_fd=dup(STDOUT_FILENO);
    gate_report=report_fd<0?NULL:fdopen(report_fd,"w");
    if(!gate_report||dup2(STDERR_FILENO,STDOUT_FILENO)<0) {
        if(gate_report)fclose(gate_report);else if(report_fd>=0)close(report_fd);free(ids.ids);return 2;
    }
    if(capture_only)fprintf(gate_report,"{\"kind\":\"capture_metadata\",\"contexts\":1,\"positions\":[126,127,128,255],\"configured_candidate_mode\":1,\"support_model_loaded\":false,\"real_drafter\":false,\"independent_oracle\":false,\"graph_subdirectory\":\"capture-graphs\",\"graph_log\":\"capture-graphs.stdout.log\"}\n");
    fprintf(gate_report,"{\"kind\":\"metadata\",\"contexts\":1,\"positions\":%s,"
        "\"temperature\":0,\"seed\":42,\"max_seq_len\":%u,\"batch2_opt_in\":true,"
        "\"target_capture_fixture\":true,\"independent_oracle\":false%s}\n",
        long_state?"[8190,8191,8192,16383]":"[126,127,128,255]",long_state?16386u:512u,
        long_state?(expected_shuffle?",\"long_indexer_state\":true,\"configured_shuffle_mode\":1,\"prefix_method\":\"public_chunk_prefill\"":
            ",\"long_indexer_state\":true,\"configured_shuffle_mode\":0,\"prefix_method\":\"public_chunk_prefill\""):"");fflush(gate_report);
    myds_model_t *model=myds_model_load(argv[1]);
    if(!model||!model->v41||model->n_layers!=40) {
        fputs("{\"kind\":\"complete\",\"exact_logits_and_state\":false,\"reason\":\"model_load_or_native_graph\"}\n",gate_report);
        if(model)myds_model_free(model);free(ids.ids);fclose(gate_report);return 1;
    }
    for(size_t i=0;i<ids.count;++i)if((uint32_t)ids.ids[i]>=model->vocab_size) {myds_model_free(model);free(ids.ids);return 2;}
    const myds_config_t cfg={.max_seq_len=long_state?16386u:512u,.max_new_tokens=8,.temperature=0,.top_p=1,.repetition_penalty=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);
    if(!ctx) {
        fputs("{\"kind\":\"complete\",\"exact_logits_and_state\":false,\"reason\":\"context_create\"}\n",gate_report);
        myds_model_free(model);free(ids.ids);fclose(gate_report);return 1;
    }
    if(!ctx->pcache_tokens) {ctx->pcache_tokens=calloc(cfg.max_seq_len,sizeof(int32_t));ctx->pcache_cap=cfg.max_seq_len;}
    /* Use actual target capture functions/buffers without loading an auxiliary
     * drafter model. This validates capture/rollback, not draft generation. */
    ctx->dspark=calloc(1,sizeof(*ctx->dspark));
    if(ctx->dspark&&!myds_dspark_capture_configure(ctx->dspark)){free(ctx->dspark);ctx->dspark=NULL;}
    if(ctx->dspark) {
        ctx->dspark->target_hidden_buf=myds_metal_buffer_alloc(3u*DIM*sizeof(float));
        ctx->dspark->hc_mean_weights=myds_metal_buffer_alloc(4u*sizeof(float));
        const float mean[4]={.25f,.25f,.25f,.25f};
        if(ctx->dspark->hc_mean_weights)myds_metal_buffer_write(ctx->dspark->hc_mean_weights,0,mean,sizeof(mean));
        for(uint32_t s=0;s<3 && ctx->dspark->target_hidden_buf;++s) {
            ctx->dspark->target_layers[s]=37u+s;
            ctx->dspark->target_slots[s]=myds_metal_buffer_sub_view(ctx->dspark->target_hidden_buf,s*DIM*sizeof(float),DIM*sizeof(float));
        }
    }
    const uint32_t short_positions[]={126,127,128,255};
    const uint32_t long_positions[]={8190,8191,8192,16383};
    const uint32_t *positions=long_state?long_positions:short_positions;
    int ok=ctx->pcache_tokens && ctx->dspark && ctx->dspark->hc_mean_weights &&
        ctx->dspark->target_slots[0] && ctx->dspark->target_slots[1] && ctx->dspark->target_slots[2];
    myds_dspark_t *capture_reference=NULL,*capture_candidate=NULL;
    if(capture_only) {
        myds_dspark_free(ctx->dspark);ctx->dspark=capture_fixture("1");
        capture_candidate=ctx->dspark;
        capture_reference=capture_fixture("0");
        ok=ctx->dspark&&capture_reference&&myds_dspark_capture_bypass_enabled(ctx->dspark);
    }
    size_t completed=0;
    /* Record real candidate use within the state gate, including public scalar
     * references. Private Batch2 work keeps its existing scorer contract. */
    myds_v41_gpu_indexer_head_slots_counters(NULL,true);
    myds_v41_gpu_topk_nan_shuffle_counters(NULL,true);
    for(size_t i=0;ok&&i<4u;++i) {
        ok=capture_only?run_capture_position(ctx,capture_reference,ctx->dspark,ids.ids,positions[i],argv[3]):
            run_position(ctx,ids.ids,positions[i],argv[3]);if(ok)++completed;
    }
    myds_v41_gpu_indexer_head_slots_counters_t head_slots={0};
    myds_v41_gpu_indexer_head_slots_counters(&head_slots,false);
    myds_v41_gpu_topk_nan_shuffle_counters_t shuffle={0};
    myds_v41_gpu_topk_nan_shuffle_counters(&shuffle,false);
    if(long_state) {
        ok=ok&&myds_v41_gpu_topk_nan_shuffle_enabled()==(int)expected_shuffle&&!shuffle.failures&&
            (expected_shuffle?(shuffle.requests>0&&shuffle.requests==shuffle.encoded):(!shuffle.requests&&!shuffle.encoded));
        fprintf(gate_report,"{\"kind\":\"long_indexer_complete\",\"positions_completed\":%zu,"
            "\"actual_mode\":%d,\"enabled\":%s,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64 ","
            "\"prefix_method\":\"public_chunk_prefill\",\"exact_target_state\":%s}\n",completed,
            myds_v41_gpu_topk_nan_shuffle_enabled(),
            myds_v41_gpu_topk_nan_shuffle_enabled()?"true":"false",shuffle.requests,shuffle.encoded,shuffle.failures,ok?"true":"false");
    }
    size_t graph_completed=0;
    if(capture_only&&ok) {
        char graph_dir[4096],graph_log[4096];
        const int paths_ok=snprintf(graph_dir,sizeof(graph_dir),"%s/capture-graphs",argv[3])>0&&
            snprintf(graph_log,sizeof(graph_log),"%s/capture-graphs.stdout.log",argv[3])>0;
        FILE *main_report=gate_report;
        FILE *graph_report=paths_ok&&!mkdir(graph_dir,0700)?fopen(graph_log,"wx"):NULL;
        ok=graph_report!=NULL;
        if(graph_report) {
            gate_report=graph_report;
            fprintf(gate_report,"{\"kind\":\"capture_graph_metadata\",\"contexts\":1,\"positions\":[126,127,128,255],"
                "\"configured_candidate_mode\":1,\"support_model_loaded\":false,\"real_drafter\":false,"
                "\"batch2_opt_in\":true,\"capture_required\":true}\n");
            myds_dspark_capture_begin_request(capture_candidate,true);
            ok=myds_dspark_capture_bypass_enabled(capture_candidate)&&myds_dspark_capture_required(capture_candidate);
            myds_dspark_capture_counters(capture_candidate,NULL,true);
            for(size_t i=0;ok&&i<4u;++i) {
                ok=run_position(ctx,ids.ids,short_positions[i],graph_dir);
                if(ok)++graph_completed;
            }
            if(ok)fprintf(gate_report,"{\"kind\":\"capture_tree_metadata\",\"reference_mode\":0,\"candidate_mode\":1,"
                "\"scalar_tree_equivalence_claimed\":false,\"capture_state_layout\":[15360,20480,4],\"raw_f32_files\":64,"
                "\"state_f32_files\":24,\"computed_logits_f32_files\":40,\"ctx_logits_used\":false}\n");
            for(size_t i=0;ok&&i<4u;++i)
                ok=run_capture_tree_position(ctx,capture_reference,capture_candidate,ids.ids,short_positions[i],graph_dir);
            myds_dspark_capture_counters_t capture={0};
            myds_dspark_capture_counters(capture_candidate,&capture,false);
            ok=ok&&graph_completed==4u&&capture.requests>0&&capture.requests==capture.encoded&&
                !capture.bypassed&&!capture.failures&&myds_dspark_capture_ready(capture_candidate);
            fprintf(gate_report,"{\"kind\":\"capture_graph_complete\",\"positions_completed\":%zu,\"candidate_mode\":1,"
                "\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"bypassed\":%" PRIu64 ",\"failures\":%" PRIu64 ","
                "\"ready\":%s,\"real_drafter\":false,\"exact_target_state\":%s}\n",graph_completed,
                capture.requests,capture.encoded,capture.bypassed,capture.failures,
                myds_dspark_capture_ready(capture_candidate)?"true":"false",ok?"true":"false");
            myds_dspark_capture_end_request(capture_candidate,ok);
            if(fclose(graph_report))ok=0;
            gate_report=main_report;
        }
    }
    fprintf(gate_report,"{\"kind\":\"complete\",\"native_v41\":true,\"contexts\":1,\"positions_completed\":%zu,\"exact_logits_and_state\":%s,"
        "\"indexer_head_slots\":{\"enabled\":%s,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"unavailable\":%" PRIu64 ",\"failures\":%" PRIu64 "}}\n",
        completed,ok?"true":"false",myds_v41_gpu_indexer_head_slots_enabled()?"true":"false",
        head_slots.requests,head_slots.encoded,head_slots.unavailable,head_slots.failures);
    if(capture_only) {
        fprintf(gate_report,"{\"kind\":\"capture_complete\",\"positions_completed\":%zu,\"candidate_mode\":1,"
            "\"raw_f32_files\":108,\"replay_files\":8,\"graph_positions_completed\":%zu,\"real_drafter\":false,\"exact_target_state\":%s}\n",completed,graph_completed,ok?"true":"false");
        if(ctx->dspark==capture_reference)ctx->dspark=NULL;
        myds_dspark_free(capture_reference);
        if(ctx->dspark!=capture_candidate)myds_dspark_free(capture_candidate);
    }
    myds_context_free(ctx);myds_model_free(model);free(ids.ids);fclose(gate_report);return ok?0:1;
}
