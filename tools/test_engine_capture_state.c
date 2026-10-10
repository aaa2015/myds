/* Real target + real support, actual Engine request lifecycle gate.
 * MODEL SUPPORT NEW_OUTPUT_DIR. This is a correctness fixture, never timing.
 * New capture scope explicitly rejects the experimental tree API. The old
 * broad capture prototype is always disabled here; its failed gate is retained.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_dspark.h"
#include "myds_server_internal.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DIM 5120u
#define CASES 10u
#define FILES_PER_CASE 18u
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); return 0; } } while (0)
extern const float *myds_v41_diagnostic_current_residual(const myds_v41_context_t *g);
extern const float *myds_v41_diagnostic_current_pre(const myds_v41_context_t *g);
static FILE *report;
static const char *output;
static uint32_t raw_count;

static uint64_t fnv(const void *bytes,size_t n) {
    uint64_t h=UINT64_C(14695981039346656037);
    const unsigned char *p=bytes;
    for(size_t i=0;i<n;i++){h^=p[i];h*=UINT64_C(1099511628211);}
    return h;
}
static void string(FILE *f,const char *s) {
    fputc('"',f);
    for(const unsigned char *p=(const unsigned char *)s;*p;p++) {
        if(*p=='"'||*p=='\\')fprintf(f,"\\%c",*p);
        else if(*p<32)fprintf(f,"\\u%04x",*p);
        else fputc(*p,f);
    }
    fputc('"',f);
}
static int raw(const char *label,const char *role,const char *dtype,const void *p,size_t n,bool comma) {
    CHECK(p||n==0);
    if(!strcmp(dtype,"f32")) {
        CHECK(n%4==0);
        const float *f=p;for(size_t i=0;i<n/4;i++)CHECK(isfinite(f[i]));
    }
    char name[160],path[4096];
    CHECK(snprintf(name,sizeof(name),"%s_%s.%s",label,role,dtype)>0);
    const int len=snprintf(path,sizeof(path),"%s/%s",output,name);
    CHECK(len>0&&(size_t)len<sizeof(path));
    FILE *fp=fopen(path,"wb");CHECK(fp);
    const bool saved=fwrite(p?p:"",1,n,fp)==n;CHECK(fclose(fp)==0&&saved);
    fprintf(report,"%s{\"path\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",
        comma?",":"",name,role,dtype,n,fnv(p,n));
    raw_count++;return 1;
}

typedef struct {
    int32_t ids[16];uint32_t count,cancel_after;
    unsigned char *bytes;size_t n,capacity;
    volatile int cancelled;
    bool error;
    myds_context_t *ctx;
    uint32_t ready_callbacks;
} stream_t;
static void stream(const char *piece,size_t n,int32_t id,void *opaque) {
    stream_t *s=opaque;
    if(s->count>=16||n>SIZE_MAX-s->n){s->error=true;s->cancelled=1;return;}
    if(s->n+n>s->capacity) {
        size_t cap=s->n+n+4096;void *p=realloc(s->bytes,cap);
        if(!p){s->error=true;s->cancelled=1;return;}s->bytes=p;s->capacity=cap;
    }
    if(n)memcpy(s->bytes+s->n,piece,n);
    s->n+=n;s->ids[s->count++]=id;
    if(myds_dspark_capture_ready(s->ctx->dspark))s->ready_callbacks++;
    if(s->cancel_after&&s->count>=s->cancel_after)s->cancelled=1;
}
static int windows(myds_context_t *ctx,float **out,size_t *elements) {
    const uint32_t visible=ctx->pos<128u?ctx->pos:128u;
    CHECK(visible>0);
    *elements=(size_t)40u*visible*512u;
    *out=malloc(*elements*sizeof(float));CHECK(*out);
    for(uint32_t il=0;il<40u;il++) {
        const float *w=myds_metal_buffer_data(myds_v41_snapshot_get_window(ctx->v41,il));CHECK(w);
        for(uint32_t k=0;k<visible;k++)
            memcpy(*out+((size_t)il*visible+k)*512u,w+((ctx->pos-visible+k)%128u)*512u,512u*sizeof(float));
    }
    return 1;
}
static int tree_rejection(myds_context_t *ctx) {
    if(!myds_dspark_engine_capture_bypass_enabled(ctx->dspark)) {
        fputs("{\"schema_version\":1,\"kind\":\"tree_guard\",\"engine_mode\":0,\"executed\":false,\"pass\":true}\n",report);
        return 1;
    }
    CHECK(myds_metal_commit_batch());
    myds_context_t before=*ctx;myds_dspark_t ds_before=*ctx->dspark;
    float slots[3u*DIM];memcpy(slots,myds_metal_buffer_data(ctx->dspark->target_hidden_buf),sizeof(slots));
    myds_v41_spec_snapshot_t *a=calloc(1,sizeof(*a)),*b=calloc(1,sizeof(*b));CHECK(a&&b);
    CHECK(myds_v41_spec_snapshot_save(ctx,a,1));
    float *w_before=NULL,*w_after=NULL;size_t n_before=0,n_after=0;
    CHECK(windows(ctx,&w_before,&n_before));
    const uint64_t dispatch_before=myds_metal_dispatch_count();
    const int barrier=myds_metal_get_auto_barrier();
    const int32_t tokens[3]={1,2,3};int32_t out[3]={-101,-102,-103};
    const int ret=myds_v41_forward_tree_spec(ctx,tokens,ctx->pos,out);
    const bool no_dispatch=myds_metal_dispatch_count()==dispatch_before;
    const bool no_barrier=myds_metal_get_auto_barrier()==barrier;
    CHECK(ret==-1&&out[0]==-101&&out[1]==-102&&out[2]==-103);
    CHECK(no_dispatch&&no_barrier&&!memcmp(&before,ctx,sizeof(before))&&!memcmp(&ds_before,ctx->dspark,sizeof(ds_before)));
    CHECK(!memcmp(slots,myds_metal_buffer_data(ctx->dspark->target_hidden_buf),sizeof(slots)));
    CHECK(myds_v41_spec_snapshot_save(ctx,b,1));
    CHECK(!memcmp(a,b,sizeof(*a))&&windows(ctx,&w_after,&n_after)&&n_before==n_after&&
        !memcmp(w_before,w_after,n_before*sizeof(float)));
    fprintf(report,"{\"schema_version\":1,\"kind\":\"tree_guard\",\"engine_mode\":1,\"executed\":true,"
        "\"ret\":-1,\"output_unchanged\":true,\"ctx_ds_snapshot_slots_window_unchanged\":true,"
        "\"dispatch_before\":%" PRIu64 ",\"dispatch_after\":%" PRIu64 ",\"auto_barrier_unchanged\":true,\"pass\":true}\n",
        dispatch_before,dispatch_before);
    free(a);free(b);free(w_before);free(w_after);return 1;
}

/* A real neural consumer, independent of the Engine's legitimate confidence
 * rejects/backoff. min_conf=0 is this diagnostic API's complete-output mode;
 * Engine thresholds and adaptive state are left intact. NaN output poisoning
 * prevents an unchecked legacy dispatch failure from reusing finite old data.
 * This is numerical equivalence, not GPU failure-injection acceptance. */
static int neural_probe(myds_context_t *ctx) {
    myds_dspark_t *ds=ctx->dspark;
    CHECK(ds&&ds->ws&&myds_dspark_capture_ready(ds)&&myds_metal_commit_batch());
    myds_metal_buffer_t *buffers[4]={ds->draft_logits_buf,ds->ws->final_norm_out,ds->ws->final_embd,ds->stage_x_buf};
    const size_t sizes[4]={129280u,DIM,DIM,DIM};
    const char *roles[4]={"logits","final_norm","final_embd","entry_projection"};
    for(uint32_t i=0;i<4;i++) {
        CHECK(buffers[i]&&myds_metal_buffer_size(buffers[i])>=sizes[i]*4u);
        float *p=myds_metal_buffer_data(buffers[i]);CHECK(p);
        for(size_t j=0;j<sizes[i];j++)p[j]=NAN;
    }
    float *target=malloc(129280u*4u);CHECK(target);
    memcpy(target,myds_metal_buffer_data(ctx->logits),129280u*4u);
    myds_context_t before=*ctx;myds_dspark_t before_ds=*ds;
    const uint64_t dispatch_before=myds_metal_dispatch_count();
    float confidence=NAN;
    const int ret=myds_dspark_eval_stages(ds,ctx,0.0f,&confidence);
    CHECK(ret==1&&myds_metal_commit_batch());
    const uint64_t dispatch_after=myds_metal_dispatch_count();
    CHECK(dispatch_after>dispatch_before&&isfinite(confidence)&&confidence>=0&&confidence<=1);
    CHECK(!memcmp(&before,ctx,sizeof(before))&&!memcmp(&before_ds,ds,sizeof(before_ds))&&
        !memcmp(target,myds_metal_buffer_data(ctx->logits),129280u*4u)&&myds_dspark_capture_ready(ds));
    fprintf(report,"{\"schema_version\":1,\"kind\":\"neural_probe\",\"engine_mode\":%d,\"ret\":1,"
        "\"min_conf\":0,\"confidence\":%.9g,\"nan_output_poison\":true,\"actual_dispatch_increased\":true,"
        "\"dispatch_before\":%" PRIu64 ",\"dispatch_after\":%" PRIu64 ",\"ctx_ds_objects_and_target_logits_unchanged\":true,"
        "\"capture_ready\":true,\"batch2_coverage_claimed\":false,\"files\":[",
        myds_dspark_engine_capture_bypass_enabled(ds),confidence,dispatch_before,dispatch_after);
    for(uint32_t i=0;i<4;i++)CHECK(raw("neural_probe",roles[i],"f32",myds_metal_buffer_data(buffers[i]),sizes[i]*4u,i!=0));
    fputs("],\"pass\":true}\n",report);fflush(report);free(target);return 1;
}

static int request(myds_context_t *ctx,uint32_t i,const char *label,const char *prompt,
    uint32_t cap,uint32_t cancel_after,bool continuation,bool spec,bool active,float temperature,int32_t *first) {
    myds_config_t cfg=ctx->cfg;cfg.max_new_tokens=cap;cfg.temperature=temperature;
    cfg.top_p=1;cfg.top_k=0;cfg.min_p=0;cfg.repetition_penalty=1;
    myds_context_set_config(ctx,&cfg);
    CHECK(setenv("MYDS_BATCH2_SPEC",spec?"1":"0",1)==0);
    if(continuation)myds_set_continue_from_cache(ctx,true);
    /* A stale physical slot must not become a valid neural generation. */
    float *slots=myds_metal_buffer_data(ctx->dspark->target_hidden_buf);CHECK(slots);
    for(size_t j=0;j<3u*DIM;j++)slots[j]=-12345.25f;
    myds_dspark_capture_counters(ctx->dspark,NULL,true);
    stream_t s={.ctx=ctx,.cancel_after=cancel_after};myds_stop_reason_t reason=MYDS_STOP_ERROR;uint32_t prompt_tokens=0;
    const int ok=myds_generate_bytes_ex(ctx,prompt,stream,&s,&reason,&prompt_tokens,&s.cancelled);
    CHECK(myds_metal_commit_batch());
    myds_prefill_stats_t pf;myds_last_prefill_stats(ctx,&pf);
    myds_decode_stats_t d;myds_last_decode_stats(ctx,&d);
    myds_dspark_capture_counters_t c={0};myds_dspark_capture_counters(ctx->dspark,&c,false);
    const bool mode=myds_dspark_engine_capture_bypass_enabled(ctx->dspark);
    const bool bypass=mode&&!active;
    bool sentinel=true;for(size_t j=0;j<3u*DIM;j++)if(slots[j]!=-12345.25f)sentinel=false;
    const bool ready=myds_dspark_capture_ready(ctx->dspark);
    fprintf(report,"{\"schema_version\":1,\"kind\":\"request_status\",\"case\":%u,\"label\":\"%s\","
        "\"ok\":%d,\"reason\":%d,\"tokens\":%u,\"capture_requests\":%" PRIu64 ",\"capture_encoded\":%" PRIu64 ","
        "\"capture_bypassed\":%" PRIu64 ",\"capture_failures\":%" PRIu64 ",\"stale_rejections\":%" PRIu64 ","
        "\"draft_cycles\":%" PRIu64 ",\"batch2_calls\":%" PRIu64 ",\"ready\":%s}\n",
        i,label,ok,(int)reason,s.count,c.requests,c.encoded,c.bypassed,c.failures,c.stale_rejections,d.draft_cycles,d.batch2_calls,ready?"true":"false");fflush(report);
    CHECK(ok==1&&!s.error&&reason!=(int)MYDS_STOP_ERROR&&reason!=(int)MYDS_STOP_CTX_FULL);
    CHECK(s.count==(i==9u?0u:cancel_after?cancel_after:cap));
    CHECK(reason==(i==9u?MYDS_STOP_EOS:cancel_after?MYDS_STOP_CANCELLED:MYDS_STOP_LENGTH));
    CHECK(c.requests>0&&!c.failures&&!c.stale_rejections);
    CHECK(bypass?(c.bypassed==c.requests&&!c.encoded&&sentinel&&!ready):(c.encoded==c.requests&&!c.bypassed));
    if(mode&&active)CHECK(cancel_after?!ready:ready);
    if(continuation)CHECK(pf.cached_tokens>0);
    if(i==0u)CHECK(pf.cached_tokens==0);
    /* Confidence rejection and backoff may legally suppress proposals here.
     * Readiness at a target callback proves recapture; the separate neural
     * probe below proves a real completed consumer with complete outputs. */
    if(i==5u||i==6u||i==7u)CHECK(s.ready_callbacks>0);
    else CHECK(d.draft_cycles==0&&d.batch2_calls==0);
    if(first&&s.count)*first=s.ids[0];
    myds_v41_spec_snapshot_t *snap=calloc(1,sizeof(*snap));CHECK(snap&&myds_v41_spec_snapshot_save(ctx,snap,1));
    CHECK(ctx->pos==snap->next_position&&ctx->pcache_len==ctx->pos&&snap->engram_position==ctx->pos);
    float *w=NULL;size_t wn=0;CHECK(windows(ctx,&w,&wn));
    const uint32_t before_count=raw_count;
    fprintf(report,"{\"schema_version\":1,\"kind\":\"request\",\"case\":%u,\"label\":\"%s\",\"engine_mode\":%d,"
        "\"spec_requested\":%s,\"capture_active_scope\":%s,\"temperature\":%.9g,\"cap\":%u,\"cancel_after\":%u,\"continuation\":%s,"
        "\"ok\":1,\"reason\":%d,\"tokens\":%u,\"output_bytes\":%zu,\"pos\":%u,\"next_position\":%u,\"pcache_len\":%u,"
        "\"rng_state\":\"%" PRIu64 "\",\"engram_position\":%u,\"engram_history\":[%d,%d,%d],"
        "\"prompt_tokens\":%u,\"cached_tokens\":%u,\"prefill_tokens\":%u,"
        "\"capture\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"bypassed\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"stale_rejections\":%" PRIu64 ","
        "\"ready\":%s,\"disabled\":%s,\"mask\":%u,\"committed\":%s,\"sentinel_preserved\":%s,\"ready_callbacks\":%u},"
        "\"decode\":{\"target_forward_calls\":%" PRIu64 ",\"target_rows\":%" PRIu64 ",\"draft_cycles\":%" PRIu64 ","
        "\"draft_proposed\":%" PRIu64 ",\"draft_accepted\":%" PRIu64 ",\"batch2_calls\":%" PRIu64 ",\"batch2_hits\":%" PRIu64 ",\"batch2_misses\":%" PRIu64 ","
        "\"batch2_failures\":%" PRIu64 ",\"batch2_fallbacks\":%" PRIu64 ",\"output_tokens\":%" PRIu64 "},"
        "\"drafter\":{\"cycles\":%" PRIu64 ",\"proposed\":%" PRIu64 ",\"accepted\":%" PRIu64 ","
        "\"consecutive_misses\":%u,\"backoff_counter\":%u,\"high_confidence\":%s,\"draft_count\":%u,\"tree_count\":%u},\"files\":[",
        i,label,mode,spec?"true":"false",active?"true":"false",temperature,cap,cancel_after,continuation?"true":"false",
        (int)reason,s.count,s.n,ctx->pos,snap->next_position,ctx->pcache_len,snap->rng_state,snap->engram_position,
        snap->engram_history.tail[0],snap->engram_history.tail[1],snap->engram_history.tail[2],prompt_tokens,pf.cached_tokens,pf.prefill_tokens,
        c.requests,c.encoded,c.bypassed,c.failures,c.stale_rejections,ready?"true":"false",ctx->dspark->capture_disabled?"true":"false",
        ctx->dspark->capture_valid_mask,ctx->dspark->capture_committed?"true":"false",sentinel?"true":"false",s.ready_callbacks,
        d.target_forward_calls,d.target_rows,d.draft_cycles,d.draft_proposed,d.draft_accepted,d.batch2_calls,d.batch2_hits,d.batch2_misses,
        d.batch2_failures,d.batch2_fallbacks,d.output_tokens,ctx->dspark->cycles,ctx->dspark->proposed_tokens,ctx->dspark->accepted_tokens,
        ctx->dspark->consecutive_misses,ctx->dspark->backoff_counter,ctx->dspark->is_high_confidence?"true":"false",ctx->dspark->draft_count,ctx->dspark->draft_tree_count);
    CHECK(raw(label,"logits","f32",myds_metal_buffer_data(ctx->logits),129280u*4u,false));
    CHECK(raw(label,"residual","f32",myds_v41_diagnostic_current_residual(ctx->v41),4u*DIM*4u,true));
    CHECK(raw(label,"pre","f32",myds_v41_diagnostic_current_pre(ctx->v41),4u*4u,true));
    CHECK(raw(label,"previous_kv","f32",snap->previous_kv,sizeof(snap->previous_kv),true));
    CHECK(raw(label,"previous_score","f32",snap->previous_score,sizeof(snap->previous_score),true));
    CHECK(raw(label,"window","f32",w,wn*4u,true));
    CHECK(raw(label,"slots","f32",slots,3u*DIM*4u,true));
    for(uint32_t owner=0;owner<4u;owner++) {
        const size_t rows=ctx->pos/(owner==3u?1u:2u);char role[64];CHECK(rows>0);
        myds_metal_buffer_t *comp=myds_v41_snapshot_get_compressed(ctx->v41,owner);
        myds_metal_buffer_t *index=myds_v41_snapshot_get_index_cache(ctx->v41,owner);
        CHECK(comp&&index&&myds_metal_buffer_size(comp)>=rows*512u*2u&&myds_metal_buffer_size(index)>=rows*128u*2u);
        snprintf(role,sizeof(role),"compressed%u",owner);CHECK(raw(label,role,"bin",myds_metal_buffer_data(comp),rows*512u*2u,true));
        snprintf(role,sizeof(role),"index%u",owner);CHECK(raw(label,role,"bin",myds_metal_buffer_data(index),rows*128u*2u,true));
    }
    CHECK(raw(label,"pcache","i32",ctx->pcache_tokens,ctx->pcache_len*4u,true));
    CHECK(raw(label,"output_ids","i32",s.ids,s.count*4u,true));
    CHECK(raw(label,"output","bin",s.bytes,s.n,true));
    CHECK(raw_count-before_count==FILES_PER_CASE);
    fputs("],\"pass\":true}\n",report);fflush(report);
    free(w);free(snap);free(s.bytes);return 1;
}

static int run(const char *model_path,const char *support_path) {
    const int mode=myds_dspark_capture_parse_mode();CHECK(mode==0||mode==2);
    CHECK(getenv("MYDS_PREFIX_CACHE")&&!strcmp(getenv("MYDS_PREFIX_CACHE"),"1"));
    CHECK(getenv("MYDS_KV_DSK")&&!strcmp(getenv("MYDS_KV_DSK"),"0"));
    CHECK(getenv("MYDS_DSPARK_CAPTURE_BYPASS")&&!strcmp(getenv("MYDS_DSPARK_CAPTURE_BYPASS"),"0"));
    CHECK(getenv("MYDS_DSPARK_MARKOV_ONLY")&&!strcmp(getenv("MYDS_DSPARK_MARKOV_ONLY"),"0"));
    CHECK(getenv("MYDS_SPEC_COST_SCHEDULER")&&!strcmp(getenv("MYDS_SPEC_COST_SCHEDULER"),"0"));
    CHECK(setenv("MYDS_PLD","0",1)==0&&setenv("MYDS_SPEC_SERIAL_PROBE","0",1)==0&&setenv("MYDS_BATCH2_SPEC","0",1)==0);
    myds_model_t *m=myds_model_load(model_path);CHECK(m&&m->v41&&m->dim==DIM&&m->vocab_size==129280u);
    myds_config_t cfg={.max_seq_len=4096,.max_new_tokens=4,.temperature=0,.top_p=1,.top_k=0,.repetition_penalty=1,.seed=42};
    myds_context_t *ctx=myds_context_create(m,&cfg);CHECK(ctx&&ctx->v41);
    ctx->dspark=myds_dspark_load(support_path,m->vocab_size);
    CHECK(ctx->dspark&&ctx->dspark->gf&&ctx->dspark->main_proj&&ctx->dspark->spec_snap&&ctx->dspark->stage_count==3u);
    CHECK(myds_dspark_engine_capture_bypass_enabled(ctx->dspark)==(mode==2));
    /* Use the service's actual model chat format. Bare prose may legitimately
     * predict EOS on the first target step and cannot exercise these cases. */
    char body[8192]="{\"enable_thinking\":false,\"messages\":[{\"role\":\"user\",\"content\":\"";
    for(uint32_t i=0;i<16u;i++)strcat(body,"The production service handles requests, restores cached prefixes, and verifies each generated token. Describe a practical plan for improving software performance while preserving exact state and output. ");
    strcat(body,"\\nProvide a detailed numbered list of twenty practical optimization steps, with explanations and examples for every step.\"}]}");
    char prompt[8192];CHECK(parse_chat_prompt(body,prompt,sizeof(prompt)));
    const size_t prompt_bytes=strlen(prompt);CHECK(prompt_bytes>0);
    fprintf(report,"{\"schema_version\":1,\"kind\":\"metadata\",\"model\":");string(report,model_path);
    fputs(",\"support\":",report);string(report,support_path);
    fprintf(report,",\"engine_mode\":%d,\"effective_mode\":%d,\"vocab\":129280,\"dim\":5120,\"cases\":10,\"files_per_case\":18,"
        "\"support_loaded\":true,\"real_drafter_weights\":true,\"stages\":3,\"prefix_cache\":true,\"ssd_cache\":false,\"performance_measurement\":false,"
        "\"service_chat_renderer\":true,\"prompt_bytes\":%zu,\"prompt_fnv1a64\":\"%016" PRIx64 "\"}\n",
        myds_dspark_engine_capture_bypass_enabled(ctx->dspark),myds_dspark_capture_bypass_enabled(ctx->dspark),prompt_bytes,fnv(prompt,prompt_bytes));fflush(report);
    int32_t next=-1;
    CHECK(request(ctx,0,"ordinary_cold",prompt,4,0,false,false,false,0,NULL));
    CHECK(request(ctx,1,"ordinary_continue"," Continue with the next point.",4,0,true,false,false,0,NULL));
    CHECK(request(ctx,2,"ordinary_cancel",prompt,8,2,false,false,false,0,NULL));
    CHECK(request(ctx,3,"ordinary_resume"," Continue after the interruption.",4,0,true,false,false,0,NULL));
    CHECK(request(ctx,4,"spec_one",prompt,1,0,false,true,false,0,&next));CHECK(next>=0);
    CHECK(request(ctx,5,"neural",prompt,4,0,false,true,true,0,NULL));
    CHECK(request(ctx,6,"neural_cancel",prompt,8,2,false,true,true,0,NULL));
    CHECK(request(ctx,7,"neural_resume"," Continue after the interruption.",4,0,true,true,true,0,NULL));
    CHECK(neural_probe(ctx)&&tree_rejection(ctx));
    CHECK(request(ctx,8,"ordinary_again",prompt,4,0,false,false,false,0.7f,NULL));
    const int32_t original_eos=m->tokenizer->eos_id;m->tokenizer->eos_id=next;
    const int eos_ok=request(ctx,9,"controlled_eos",prompt,4,0,false,true,true,0,NULL);
    m->tokenizer->eos_id=original_eos;CHECK(eos_ok);
    CHECK(raw_count==CASES*FILES_PER_CASE+4u);
    fprintf(report,"{\"schema_version\":1,\"kind\":\"complete\",\"cases\":10,\"raw_files\":%u,\"engine_mode\":%d,"
        "\"controlled_eos\":true,\"tree_scope_checked\":true,\"pass\":true}\n",raw_count,mode==2);fflush(report);
    myds_context_free(ctx);myds_model_free(m);return 1;
}
int main(int argc,char **argv) {
    if(argc!=4){fputs("Usage: test_engine_capture_state MODEL SUPPORT NEW_OUTPUT_DIR\n",stderr);return 2;}
    output=argv[3];if(mkdir(output,0700)){perror("new output directory");return 2;}
    fflush(stdout);const int fd=dup(STDOUT_FILENO);report=fd<0?NULL:fdopen(fd,"w");
    if(!report||dup2(STDERR_FILENO,STDOUT_FILENO)<0)return 2;
    const int ok=run(argv[1],argv[2]);fclose(report);return ok?0:1;
}
