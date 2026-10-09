/* Full-model Batch-2 equivalence/state gate, not a speed benchmark or an
 * independent model oracle. One loaded model and one context are reused.
 * Run in an isolated source directory; never starts or contacts a server.
 * MODEL TOKEN_FILE OUTPUT_DIR, with >=256 exact input IDs in TOKEN_FILE.
 * The existing pinned ds4 trace harness can replay the emitted input files.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_internal.h"
#include "myds_v41.h"
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
static int state_equal(myds_context_t *ctx,const state_t *ref,
    const myds_v41_spec_snapshot_t *live_first,const char *label) {
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
    REQUIRE(equal_f32(label,ref->logits,myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size));
    for(uint32_t t=0;t<3;++t)
        REQUIRE(equal_f32("dspark_target_slot",ref->slots[t],myds_metal_buffer_data(ctx->dspark->target_slots[t]),DIM));
    if(live_first) {
        REQUIRE(live_first->first_state_valid);
        REQUIRE(equal_f32("first_hidden",ref->residual,myds_metal_buffer_data(live_first->first_residual),4u*DIM));
        REQUIRE(equal_f32("first_pre",ref->pre,myds_metal_buffer_data(live_first->first_pre),4));
    }
    return 1;
}
static int consume(myds_context_t *ctx,int32_t token,uint32_t pos,bool sample) {
    ctx->pos=pos; ctx->pcache_tokens[pos]=token; ctx->pcache_len=pos+1u;
    const int next=myds_v41_forward(ctx,token,pos,sample);
    if(next>=0)ctx->pos=pos+1u;
    return next;
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
    for(uint32_t i=0;i<pos;++i)REQUIRE(consume(ctx,ids[i],i,false)>=0);
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
    for(uint32_t i=0;i<pos;++i)REQUIRE(consume(ctx,ids[i],i,false)>=0);
    REQUIRE(consume(ctx,ids[pos],pos,true)==next0);
    REQUIRE(state_equal(ctx,&first,NULL,"fresh_request_after_restore_failure"));
    REQUIRE(consume(ctx,next0,pos+1u,true)==next1);
    REQUIRE(state_equal(ctx,&second,NULL,"fresh_request_second_after_failure"));
    fprintf(gate_report,"{\"kind\":\"position_complete\",\"pos\":%u,\"next0\":%d,\"next1\":%d,\"batch2_calls\":%" PRIu64 ",\"hits\":%" PRIu64 ",\"misses\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"fallbacks\":%" PRIu64 ",\"pass\":true}\n",
        pos,next0,next1,ctx->decode_stats.batch2_calls,ctx->decode_stats.batch2_hits,
        ctx->decode_stats.batch2_misses,ctx->decode_stats.batch2_failures,ctx->decode_stats.batch2_fallbacks);fflush(gate_report);
    state_free(&first);state_free(&second);free(base);return 1;
}
int main(int argc,char **argv) {
    if(argc!=4) {fprintf(stderr,"Usage: %s MODEL TOKEN_FILE OUTPUT_DIR\n",argv[0]);return 2;}
    trace_tokens ids={0}; char *parsed_argv[]={argv[0],argv[1],"--tokens-file",argv[2]};
    if(!trace_tokens_parse(&ids,4,parsed_argv)||ids.count<256) {free(ids.ids);return 2;}
    if(mkdir(argv[3],0700)&&errno!=EEXIST) {free(ids.ids);return 2;}
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
    fputs("{\"kind\":\"metadata\",\"contexts\":1,\"positions\":[126,127,128,255],"
        "\"temperature\":0,\"seed\":42,\"max_seq_len\":512,\"batch2_opt_in\":true,"
        "\"target_capture_fixture\":true,\"independent_oracle\":false}\n",gate_report);fflush(gate_report);
    myds_model_t *model=myds_model_load(argv[1]);
    if(!model||!model->v41||model->n_layers!=40) {
        fputs("{\"kind\":\"complete\",\"exact_logits_and_state\":false,\"reason\":\"model_load_or_native_graph\"}\n",gate_report);
        if(model)myds_model_free(model);free(ids.ids);fclose(gate_report);return 1;
    }
    for(size_t i=0;i<ids.count;++i)if((uint32_t)ids.ids[i]>=model->vocab_size) {myds_model_free(model);free(ids.ids);return 2;}
    const myds_config_t cfg={.max_seq_len=512,.max_new_tokens=8,.temperature=0,.top_p=1,.repetition_penalty=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);
    if(!ctx) {
        fputs("{\"kind\":\"complete\",\"exact_logits_and_state\":false,\"reason\":\"context_create\"}\n",gate_report);
        myds_model_free(model);free(ids.ids);fclose(gate_report);return 1;
    }
    if(!ctx->pcache_tokens) {ctx->pcache_tokens=calloc(cfg.max_seq_len,sizeof(int32_t));ctx->pcache_cap=cfg.max_seq_len;}
    /* Use actual target capture functions/buffers without loading an auxiliary
     * drafter model. This validates capture/rollback, not draft generation. */
    ctx->dspark=calloc(1,sizeof(*ctx->dspark));
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
    const uint32_t positions[]={126,127,128,255};
    int ok=ctx->pcache_tokens && ctx->dspark && ctx->dspark->hc_mean_weights &&
        ctx->dspark->target_slots[0] && ctx->dspark->target_slots[1] && ctx->dspark->target_slots[2];
    size_t completed=0;
    for(size_t i=0;ok&&i<sizeof(positions)/sizeof(*positions);++i) {
        ok=run_position(ctx,ids.ids,positions[i],argv[3]);if(ok)++completed;
    }
    fprintf(gate_report,"{\"kind\":\"complete\",\"native_v41\":true,\"contexts\":1,\"positions_completed\":%zu,\"exact_logits_and_state\":%s}\n",completed,ok?"true":"false");
    myds_context_free(ctx);myds_model_free(model);free(ids.ids);fclose(gate_report);return ok?0:1;
}
