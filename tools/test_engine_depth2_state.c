/* Real target + real support: continuous depth2 Engine lifecycle gate.
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
    myds_context_t *ctx;
    int32_t ids[512]; uint32_t count, cancel_after, hit_output, hit_index;
    uint64_t hits_seen;
    unsigned char *bytes; size_t n, capacity;
    volatile int cancel; bool failed;
} stream_t;
static void stream(const char *piece,size_t n,int32_t id,void *opaque) {
    stream_t *s=opaque;
    if(s->count>=512||n>SIZE_MAX-s->n){s->failed=true;s->cancel=1;return;}
    if(s->n+n>s->capacity) {
        size_t cap=s->n+n+4096;void *p=realloc(s->bytes,cap);
        if(!p){s->failed=true;s->cancel=1;return;}s->bytes=p;s->capacity=cap;
    }
    if(n)memcpy(s->bytes+s->n,piece,n);
    s->n+=n;s->ids[s->count++]=id;
    if(s->hit_output) {
        const uint64_t h=s->ctx->decode_stats.depth2_hits2;
        if(h>s->hits_seen){s->hits_seen=h;s->hit_index=1;}
        else if(s->hit_index)s->hit_index++;
        if(s->hit_index==s->hit_output)s->cancel=1;
    }
    if(s->cancel_after&&s->count>=s->cancel_after)s->cancel=1;
}
static int windows(myds_context_t *ctx,float **out,size_t *elements) {
    const uint32_t visible=ctx->pos<128u?ctx->pos:128u;CHECK(visible>0);
    *elements=(size_t)40u*visible*512u;*out=malloc(*elements*4u);CHECK(*out);
    for(uint32_t il=0;il<40u;il++) {
        const float *w=myds_metal_buffer_data(myds_v41_snapshot_get_window(ctx->v41,il));CHECK(w);
        for(uint32_t k=0;k<visible;k++)memcpy(*out+((size_t)il*visible+k)*512u,
            w+((ctx->pos-visible+k)%128u)*512u,512u*4u);
    }return 1;
}
static int export_state(myds_context_t *ctx,const char *label,stream_t *s,myds_stop_reason_t reason) {
    myds_v41_spec_snapshot_t *snap=calloc(1,sizeof(*snap));CHECK(snap);
    CHECK(myds_metal_commit_batch()&&myds_v41_spec_snapshot_save(ctx,snap,1));
    CHECK(ctx->pos==snap->next_position&&ctx->pcache_len==ctx->pos&&snap->engram_position==ctx->pos);
    float *w=NULL;size_t wn=0;CHECK(windows(ctx,&w,&wn));
    myds_decode_stats_t d;myds_last_decode_stats(ctx,&d);
    CHECK(d.output_tokens==s->count&&d.draft_accepted<=d.draft_proposed);
    CHECK(d.depth2_failures==0&&d.depth2_fallbacks==0&&d.batch2_failures==0&&d.batch2_fallbacks==0);
    fprintf(report,"{\"kind\":\"state\",\"label\":\"%s\",\"reason\":%d,\"output_tokens\":%u,\"pos\":%u,"
        "\"next_position\":%u,\"pcache_len\":%u,\"rng_state\":\"%" PRIu64 "\",\"engram_position\":%u,\"engram_history\":[%d,%d,%d],"
        "\"decode\":{\"target_forward_calls\":%" PRIu64 ",\"target_rows\":%" PRIu64 ",\"batch2_calls\":%" PRIu64 ","
        "\"depth2_calls\":%" PRIu64 ",\"depth2_hits1\":%" PRIu64 ",\"depth2_hits2\":%" PRIu64 ",\"depth2_misses\":%" PRIu64 ","
        "\"depth2_failures\":%" PRIu64 ",\"depth2_fallbacks\":%" PRIu64 ",\"draft_proposed\":%" PRIu64 ",\"draft_accepted\":%" PRIu64 "},\"files\":[",
        label,reason,s->count,ctx->pos,snap->next_position,ctx->pcache_len,snap->rng_state,snap->engram_position,
        snap->engram_history.tail[0],snap->engram_history.tail[1],snap->engram_history.tail[2],d.target_forward_calls,d.target_rows,
        d.batch2_calls,d.depth2_calls,d.depth2_hits1,d.depth2_hits2,d.depth2_misses,d.depth2_failures,d.depth2_fallbacks,d.draft_proposed,d.draft_accepted);
    CHECK(raw(label,"logits","f32",myds_metal_buffer_data(ctx->logits),129280u*4u,false));
    CHECK(raw(label,"residual","f32",myds_v41_diagnostic_current_residual(ctx->v41),4u*DIM*4u,true));
    CHECK(raw(label,"pre","f32",myds_v41_diagnostic_current_pre(ctx->v41),16u,true));
    CHECK(raw(label,"previous_kv","f32",snap->previous_kv,sizeof(snap->previous_kv),true));
    CHECK(raw(label,"previous_score","f32",snap->previous_score,sizeof(snap->previous_score),true));
    CHECK(raw(label,"window","f32",w,wn*4u,true));
    CHECK(raw(label,"slots","f32",myds_metal_buffer_data(ctx->dspark->target_hidden_buf),3u*DIM*4u,true));
    for(uint32_t owner=0;owner<4u;owner++) {
        const size_t rows=ctx->pos/(owner==3u?1u:2u);char role[64];CHECK(rows>0);
        myds_metal_buffer_t *comp=myds_v41_snapshot_get_compressed(ctx->v41,owner);
        myds_metal_buffer_t *index=myds_v41_snapshot_get_index_cache(ctx->v41,owner);
        CHECK(comp&&index&&myds_metal_buffer_size(comp)>=rows*1024u&&myds_metal_buffer_size(index)>=rows*256u);
        snprintf(role,sizeof(role),"compressed%u",owner);CHECK(raw(label,role,"bin",myds_metal_buffer_data(comp),rows*1024u,true));
        snprintf(role,sizeof(role),"index%u",owner);CHECK(raw(label,role,"bin",myds_metal_buffer_data(index),rows*256u,true));
    }
    CHECK(raw(label,"pcache","i32",ctx->pcache_tokens,ctx->pcache_len*4u,true));
    CHECK(raw(label,"output_ids","i32",s->ids,s->count*4u,true));
    CHECK(raw(label,"output","bin",s->bytes,s->n,true));
    fputs("],\"pass\":true}\n",report);fflush(report);free(snap);free(w);return 1;
}
static int request(myds_context_t *ctx,const char *label,const char *prompt,bool spec,
    uint32_t cap,uint32_t cancel_after,uint32_t hit_output,bool continuation,float temperature,
    stream_t *s,myds_stop_reason_t *reason) {
    myds_config_t cfg=ctx->cfg;cfg.max_new_tokens=cap;cfg.temperature=temperature;
    cfg.top_p=1;cfg.top_k=0;cfg.min_p=0;cfg.repetition_penalty=1;myds_context_set_config(ctx,&cfg);
    CHECK(setenv("MYDS_BATCH2_SPEC",spec?"2":"0",1)==0&&setenv("MYDS_DSPARK_DEPTH",spec?"2":"1",1)==0);
    if(continuation)myds_set_continue_from_cache(ctx,true);
    *s=(stream_t){.ctx=ctx,.cancel_after=cancel_after,.hit_output=hit_output};
    uint32_t prompt_tokens=0;
    CHECK(myds_generate_bytes_ex(ctx,prompt,stream,s,reason,&prompt_tokens,&s->cancel)==1&&!s->failed);
    fprintf(stderr,"[depth2-engine] %s reason=%d outputs=%u cap=%u cached=%u hits2=%" PRIu64 "\n",label,(int)*reason,s->count,cap,ctx->last_cached_tokens,ctx->decode_stats.depth2_hits2);
    if(hit_output)CHECK(*reason==MYDS_STOP_CANCELLED&&s->hit_index==hit_output&&ctx->decode_stats.depth2_hits2>0);
    else if(cancel_after)CHECK(*reason==MYDS_STOP_CANCELLED&&s->count==cancel_after);
    else if(continuation)CHECK((*reason==MYDS_STOP_LENGTH&&s->count==cap)||(*reason==MYDS_STOP_EOS&&s->count<cap));
    else CHECK(*reason==MYDS_STOP_LENGTH&&s->count==cap);
    if(continuation)CHECK(ctx->last_cached_tokens>0);
    if(!spec)CHECK(ctx->decode_stats.depth2_calls==0&&ctx->decode_stats.draft_cycles==0);
    fprintf(report,"{\"kind\":\"request\",\"label\":\"%s\",\"spec\":%s,\"cap\":%u,\"cancel_after\":%u,"
        "\"hit_output_cancel\":%u,\"hit_index\":%u,\"continuation\":%s,\"temperature\":%.9g,\"prompt_tokens\":%u,\"cached_tokens\":%u,\"pass\":true}\n",
        label,spec?"true":"false",cap,cancel_after,hit_output,s->hit_index,continuation?"true":"false",temperature,prompt_tokens,ctx->last_cached_tokens);
    return export_state(ctx,label,s,*reason);
}
/* Compare actual slots1 and2 against full five-slot inference. Poison every
 * unprojected LM head; the prefix may not publish finite stale outputs there. */
static int prefix_probe(myds_context_t *ctx) {
    myds_dspark_t *ds=ctx->dspark;CHECK(myds_metal_commit_batch());
    const myds_dspark_t original=*ds;const myds_context_t context=*ctx;
    float markov[MYDS_DSPARK_MARKOV_RANK];if(ds->markov_state_host)memcpy(markov,ds->markov_state_host,sizeof(markov));
    ds->current_seed_token=ctx->pcache_tokens[ctx->pos-1u];ds->current_seed_position=ctx->pos-1u;ds->current_output_slot=1;
    myds_metal_buffer_t *bufs[8];size_t sizes[8];void *refs[8];
    for(unsigned row=1;row<=2;row++) {
        unsigned k=(row-1)*4;bufs[k]=ds->ws->v_stage_draft_hc[row];sizes[k]=4u*DIM*4u;
        bufs[k+1]=ds->ws->v_final_embd[row];sizes[k+1]=DIM*4u;
        bufs[k+2]=ds->ws->v_final_norm_out[row];sizes[k+2]=DIM*4u;
        bufs[k+3]=ds->ws->v_logits[row];sizes[k+3]=129280u*4u;
    }
    for(unsigned i=0;i<8;i++){refs[i]=malloc(sizes[i]);CHECK(refs[i]&&bufs[i]);}
    float cf=NAN,cp=NAN;uint64_t db=myds_metal_dispatch_count();
    CHECK(myds_dspark_eval_stages(ds,ctx,0,&cf)==1&&myds_metal_commit_batch());
    const uint64_t df=myds_metal_dispatch_count()-db;
    for(unsigned i=0;i<8;i++)memcpy(refs[i],myds_metal_buffer_data(bufs[i]),sizes[i]);
    float *draft=myds_metal_buffer_data(ds->draft_logits_buf);
    for(size_t i=0;i<5u*129280u;i++)draft[i]=NAN;
    db=myds_metal_dispatch_count();
    CHECK(myds_dspark_eval_stages_prefix(ds,ctx,0,&cp,1,2)==1&&myds_metal_commit_batch());
    const uint64_t dp=myds_metal_dispatch_count()-db;
    CHECK(dp>0&&dp<df&&!memcmp(&cf,&cp,4));
    fprintf(report,"{\"kind\":\"prefix_probe\",\"first_slot\":1,\"count\":2,\"stage_slots\":3,\"kv_rows\":4,"
        "\"full_dispatches\":%" PRIu64 ",\"prefix_dispatches\":%" PRIu64 ",\"files\":[",df,dp);
    for(unsigned i=0;i<8;i++) {
        const float *p=myds_metal_buffer_data(bufs[i]);CHECK(!memcmp(refs[i],p,sizes[i]));
        char role[32];snprintf(role,sizeof(role),"slot%u_role%u",i/4+1,i%4);CHECK(raw("prefix",role,"f32",p,sizes[i],i!=0));
        free(refs[i]);
    }
    for(size_t i=0;i<5u*129280u;i++)CHECK(i>=129280u&&i<3u*129280u?isfinite(draft[i]):isnan(draft[i]));
    *ds=original;if(ds->markov_state_host)memcpy(ds->markov_state_host,markov,sizeof(markov));
    CHECK(!memcmp(ctx,&context,sizeof(context)));
    fputs("],\"slot1_slot2_bitwise_equal\":true,\"other_heads_nan\":true,\"pass\":true}\n",report);fflush(report);return 1;
}
static myds_context_t *create(myds_model_t *m,const char *support) {
    myds_config_t cfg={.max_seq_len=4096,.max_new_tokens=256,.temperature=0,.top_p=1,.repetition_penalty=1,.seed=42};
    myds_context_t *ctx=myds_context_create(m,&cfg);if(!ctx)return NULL;
    ctx->dspark=myds_dspark_load(support,m->vocab_size);
    if(!ctx->dspark){myds_context_free(ctx);return NULL;}return ctx;
}
static int run(const char *model,const char *support) {
    CHECK(setenv("MYDS_PLD","0",1)==0&&setenv("MYDS_SPEC_SERIAL_PROBE","0",1)==0);
    CHECK(setenv("MYDS_DSPARK_CAPTURE_BYPASS","0",1)==0&&setenv("MYDS_PREFIX_CACHE","1",1)==0);
    CHECK(!myds_trace_active()&&!getenv("MYDS_TRACE_DIR"));
    CHECK(setenv("MYDS_TREE_SPEC","1",1)==0&&setenv("MYDS_ENGINE_CAPTURE_BYPASS","0",1)==0);
    CHECK(setenv("MYDS_DSPARK_BACKOFF_THRESH","0",1)==0);
    myds_model_t *m=myds_model_load(model);CHECK(m&&m->v41&&m->vocab_size==129280u&&m->dim==DIM);
    char prompt[8192];CHECK(parse_chat_prompt("{\"enable_thinking\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Write twenty numbered practical steps for software performance optimization, explaining each step with a detailed example.\"}]}",prompt,sizeof(prompt)));
    fprintf(report,"{\"kind\":\"metadata\",\"schema_version\":1,\"target\":");string(report,model);fputs(",\"support\":",report);string(report,support);
    fputs(",\"real_support_loaded\":true,\"capture_mode\":0,\"greedy\":true,\"performance_measurement\":false,\"pairs\":5,\"files_per_state\":18}\n",report);
    for(unsigned c=0;c<5;c++) {
        myds_context_t *candidate=create(m,support),*ordinary=create(m,support);CHECK(candidate&&ordinary);
        stream_t s={0},o={0};myds_stop_reason_t sr,orr;char cl[64],ol[64];
        snprintf(cl,sizeof(cl),"case%u_candidate",c);snprintf(ol,sizeof(ol),"case%u_ordinary",c);
        const uint32_t hit=c<2?c+1:0,cap=c<2?256:c-1;
        CHECK(request(candidate,cl,prompt,true,cap,0,hit,false,0,&s,&sr));
        CHECK(request(ordinary,ol,prompt,false,hit?256:cap,hit?s.count:0,0,false,0,&o,&orr));
        CHECK(s.count==o.count&&s.n==o.n&&!memcmp(s.ids,o.ids,s.count*4u)&&!memcmp(s.bytes,o.bytes,s.n)&&sr==orr);
        free(s.bytes);free(o.bytes);
        if(c<2) {
            snprintf(cl,sizeof(cl),"case%u_candidate_continue",c);snprintf(ol,sizeof(ol),"case%u_ordinary_continue",c);
            CHECK(request(candidate,cl," Continue with the next point.",true,16,0,0,true,0,&s,&sr));
            CHECK(request(ordinary,ol," Continue with the next point.",false,16,0,0,true,0,&o,&orr));
            CHECK(s.count==o.count&&s.n==o.n&&!memcmp(s.ids,o.ids,s.count*4u)&&!memcmp(s.bytes,o.bytes,s.n));
            if(c==1)CHECK(prefix_probe(candidate));
            free(s.bytes);free(o.bytes);
        }
        myds_context_free(candidate);myds_context_free(ordinary);
    }
    fprintf(report,"{\"kind\":\"complete\",\"states\":14,\"raw_files\":%u,\"native_all_hit_partial1_partial2\":true,\"pass\":true}\n",raw_count);
    myds_model_free(m);return 1;
}
int main(int argc,char **argv) {
    if(argc!=4){fputs("Usage: test_engine_depth2_state MODEL SUPPORT NEW_OUTPUT_DIR\n",stderr);return 2;}
    output=argv[3];if(mkdir(output,0700)){perror("new output directory");return 2;}
    fflush(stdout);int fd=dup(STDOUT_FILENO);report=fd<0?NULL:fdopen(fd,"w");
    if(!report||dup2(STDERR_FILENO,STDOUT_FILENO)<0)return 2;
    int ok=run(argv[1],argv[2]);fclose(report);return ok?0:1;
}
