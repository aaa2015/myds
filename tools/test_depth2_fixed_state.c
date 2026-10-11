/* Continuous greedy depth2 target-state fixture. No drafter or performance.
 * MODEL TOKEN_FILE_OR_DASH NEW_OUTPUT_DIR; MYDS_FIXED_STATE_POSITION=11..2047.
 * Default: miss1/partial1, miss2/partial1+partial2, hit3/partial1+partial2.
 * MYDS_FIXED_STATE_FAULTS=1: second/third head and final publication commit failures, second EOS.
 * Borrowed checkpoint views are consumed before any subsequent forward. */
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
#define ITEMS 17u
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);return 0;} } while(0)
extern const float *myds_v41_diagnostic_current_residual(const myds_v41_context_t *g);
extern const float *myds_v41_diagnostic_current_pre(const myds_v41_context_t *g);
extern char **environ;
static FILE *report;
static const char *output;
static uint32_t raw_count;
static const int32_t default_ids[10]={0,128799,15061,2229,15102,4071,6598,1468,320,3655};

typedef struct {char role[40];const char *dtype;void *data;size_t bytes;} payload_t;
typedef struct {
    myds_v41_spec_snapshot_t *snap;
    uint32_t pos,pcache_len;
    payload_t items[ITEMS];uint32_t count;
} state_t;
static uint64_t fnv(const void *p,size_t n) {
    uint64_t hash=UINT64_C(14695981039346656037);const unsigned char *bytes=p;
    for(size_t i=0;i<n;i++){hash^=bytes[i];hash*=UINT64_C(1099511628211);}return hash;
}
static void string(FILE *file,const char *value) {
    fputc('"',file);for(const unsigned char *p=(const unsigned char *)value;*p;p++) {
        if(*p=='"'||*p=='\\')fprintf(file,"\\%c",*p);else if(*p<32)fprintf(file,"\\u%04x",*p);else fputc(*p,file);
    }fputc('"',file);
}
static int append(state_t *s,const char *role,const char *dtype,const void *data,size_t bytes) {
    CHECK(s->count<ITEMS&&data&&bytes);payload_t *item=&s->items[s->count++];
    CHECK(strlen(role)<sizeof(item->role));strcpy(item->role,role);item->dtype=dtype;item->bytes=bytes;
    item->data=malloc(bytes);CHECK(item->data);memcpy(item->data,data,bytes);return 1;
}
static const payload_t *find(const state_t *s,const char *role) {
    for(uint32_t i=0;i<s->count;i++)if(!strcmp(s->items[i].role,role))return &s->items[i];return NULL;
}
static int capture(myds_context_t *ctx,state_t *s) {
    CHECK(myds_metal_commit_batch());s->snap=calloc(1,sizeof(*s->snap));CHECK(s->snap);
    CHECK(myds_v41_spec_snapshot_save(ctx,s->snap,0));s->pos=ctx->pos;s->pcache_len=ctx->pcache_len;
    CHECK(s->pos==s->pcache_len&&s->pos==s->snap->next_position);
    CHECK(append(s,"logits","f32",myds_metal_buffer_data(ctx->logits),ctx->model->vocab_size*sizeof(float))&&
        append(s,"residual","f32",myds_v41_diagnostic_current_residual(ctx->v41),4u*DIM*sizeof(float))&&
        append(s,"pre","f32",myds_v41_diagnostic_current_pre(ctx->v41),4u*sizeof(float))&&
        append(s,"previous_kv","f32",s->snap->previous_kv,sizeof(s->snap->previous_kv))&&
        append(s,"previous_score","f32",s->snap->previous_score,sizeof(s->snap->previous_score))&&
        append(s,"slots","f32",myds_metal_buffer_data(ctx->dspark->target_hidden_buf),3u*DIM*sizeof(float)));
    const uint32_t visible=ctx->pos<128u?ctx->pos:128u;const size_t elements=(size_t)40u*visible*512u;
    float *window=malloc(elements*sizeof(float));CHECK(window);
    for(uint32_t layer=0;layer<40u;layer++) {
        const float *w=myds_metal_buffer_data(myds_v41_snapshot_get_window(ctx->v41,layer));CHECK(w);
        for(uint32_t i=0;i<visible;i++)memcpy(window+((size_t)layer*visible+i)*512u,
            w+((ctx->pos-visible+i)%128u)*512u,512u*sizeof(float));
    }
    CHECK(append(s,"window","f32",window,elements*sizeof(float)));free(window);
    for(uint32_t owner=0;owner<4u;owner++) {
        const size_t rows=ctx->pos/(owner==3u?1u:2u);char role[40];
        myds_metal_buffer_t *compressed=myds_v41_snapshot_get_compressed(ctx->v41,owner);
        myds_metal_buffer_t *index=myds_v41_snapshot_get_index_cache(ctx->v41,owner);
        CHECK(rows&&compressed&&index&&myds_metal_buffer_size(compressed)>=rows*512u*2u&&myds_metal_buffer_size(index)>=rows*128u*2u);
        snprintf(role,sizeof(role),"compressed%u",owner);CHECK(append(s,role,"bf16",myds_metal_buffer_data(compressed),rows*512u*2u));
        snprintf(role,sizeof(role),"index%u",owner);CHECK(append(s,role,"bf16",myds_metal_buffer_data(index),rows*128u*2u));
    }
    CHECK(append(s,"pcache","i32",ctx->pcache_tokens,ctx->pcache_len*sizeof(int32_t))&&
        append(s,"snapshot","bin",s->snap,sizeof(*s->snap))&&s->count==ITEMS);return 1;
}
static void free_state(state_t *s) {
    for(uint32_t i=0;i<s->count;i++)free(s->items[i].data);free(s->snap);memset(s,0,sizeof(*s));
}
static int export_payload(uint32_t repeat,const char *label,const payload_t *item,bool comma) {
    char name[192],path[4096];const int n=snprintf(name,sizeof(name),"r%u_%s_%s.%s",repeat,label,item->role,item->dtype);
    CHECK(n>0&&(size_t)n<sizeof(name));const int p=snprintf(path,sizeof(path),"%s/%s",output,name);CHECK(p>0&&(size_t)p<sizeof(path));
    FILE *file=fopen(path,"wb");CHECK(file);const bool saved=fwrite(item->data,1,item->bytes,file)==item->bytes;
    CHECK(fclose(file)==0&&saved);raw_count++;
    fprintf(report,"%s{\"path\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",
        comma?",":"",name,item->role,item->dtype,item->bytes,fnv(item->data,item->bytes));return 1;
}
static int export_state(uint32_t repeat,const char *label,const state_t *s) {
    fprintf(report,"{\"schema_version\":1,\"kind\":\"state\",\"repeat\":%u,\"label\":\"%s\",\"pos\":%u,\"pcache_len\":%u,"
        "\"next_position\":%u,\"rng_state\":\"%" PRIu64 "\",\"engram_position\":%u,\"engram_history\":[%d,%d,%d],\"files\":[",
        repeat,label,s->pos,s->pcache_len,s->snap->next_position,s->snap->rng_state,s->snap->engram_position,
        s->snap->engram_history.tail[0],s->snap->engram_history.tail[1],s->snap->engram_history.tail[2]);
    for(uint32_t i=0;i<s->count;i++)CHECK(export_payload(repeat,label,&s->items[i],i!=0));
    fputs("]}\n",report);fflush(report);return 1;
}
static uint32_t ordered(float value) {
    uint32_t bits;memcpy(&bits,&value,4);return bits&UINT32_C(0x80000000)?~bits:bits|UINT32_C(0x80000000);
}
static float bf16_float(const void *data,size_t i) {
    uint16_t h;memcpy(&h,(const unsigned char *)data+i*2u,2);uint32_t bits=(uint32_t)h<<16;float value;memcpy(&value,&bits,4);return value;
}
static int compare(uint32_t repeat,const char *label,const payload_t *a,const payload_t *b) {
    if(!a||!b||a->bytes!=b->bytes||strcmp(a->dtype,b->dtype))return 0;
    const bool f32=!strcmp(a->dtype,"f32"),bf16=!strcmp(a->dtype,"bf16");
    const size_t width=f32?4u:bf16?2u:1u,elements=a->bytes/width;
    size_t mismatch=0,first=0,nonfinite=0;double max_abs=0;uint32_t max_ulp=0,first_ref_bits=0,first_actual_bits=0;
    for(size_t i=0;i<elements;i++) {
        const unsigned char *left=(const unsigned char *)a->data+i*width,*right=(const unsigned char *)b->data+i*width;
        const bool different=memcmp(left,right,width)!=0;
        if(different){if(!mismatch){first=i;memcpy(&first_ref_bits,left,width);memcpy(&first_actual_bits,right,width);}mismatch++;}
        if(f32||bf16) {
            float x,y;if(f32){memcpy(&x,left,4);memcpy(&y,right,4);}else{x=bf16_float(a->data,i);y=bf16_float(b->data,i);}
            if(!isfinite(x)||!isfinite(y)){nonfinite++;continue;}
            const double error=fabs((double)x-y);if(error>max_abs)max_abs=error;
            const uint32_t ox=ordered(x),oy=ordered(y),ulp=ox>oy?ox-oy:oy-ox;if(ulp>max_ulp)max_ulp=ulp;
        }
    }
    const bool pass=!mismatch&&!nonfinite;
    fprintf(report,"{\"schema_version\":1,\"kind\":\"comparison\",\"repeat\":%u,\"label\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\","
        "\"bytes\":%zu,\"elements\":%zu,\"mismatches\":%zu,\"nonfinite\":%zu,\"first_index\":%zu,"
        "\"first_reference_bits\":\"%08x\",\"first_actual_bits\":\"%08x\",\"max_abs\":%.17g,\"max_ulp\":%u,\"pass\":%s}\n",
        repeat,label,a->role,a->dtype,a->bytes,elements,mismatch,nonfinite,first,first_ref_bits,first_actual_bits,max_abs,max_ulp,pass?"true":"false");
    fflush(report);return pass;
}
static int equal_states(uint32_t repeat,const char *label,const state_t *a,const state_t *b) {
    bool pass=a->count==b->count&&a->pos==b->pos&&a->pcache_len==b->pcache_len;
    for(uint32_t i=0;i<a->count;i++)pass=compare(repeat,label,&a->items[i],find(b,a->items[i].role))&&pass;
    return pass;
}
static int consume(myds_context_t *ctx,int32_t token,uint32_t pos) {
    if(!ctx->pcache_tokens||pos>=ctx->pcache_cap)return -1;
    ctx->pos=pos;ctx->pcache_tokens[pos]=token;ctx->pcache_len=pos+1u;
    const int32_t next=myds_v41_forward(ctx,token,pos,true);if(next>=0)ctx->pos=pos+1u;return next;
}
static int32_t miss_candidate(const myds_model_t *model,int32_t truth,int32_t avoid,uint32_t offset) {
    for(uint32_t i=0;i<model->vocab_size;i++) {
        const int32_t candidate=(int32_t)(((uint64_t)(uint32_t)truth+offset+i)%model->vocab_size);
        if(candidate!=truth&&candidate!=avoid&&candidate!=model->tokenizer->eos_id)return candidate;
    }
    return -1;
}
static myds_dspark_t *capture_fixture(void) {
    myds_dspark_t *ds=calloc(1,sizeof(*ds));if(!ds)return NULL;
    if(!myds_dspark_capture_configure(ds)){free(ds);return NULL;}
    ds->target_hidden_buf=myds_metal_buffer_alloc(3u*DIM*sizeof(float));ds->hc_mean_weights=myds_metal_buffer_alloc(4u*sizeof(float));
    const float mean[4]={.25f,.25f,.25f,.25f};
    if(!ds->target_hidden_buf||!ds->hc_mean_weights||!myds_metal_buffer_write(ds->hc_mean_weights,0,mean,sizeof(mean)))goto failed;
    for(uint32_t i=0;i<3;i++){ds->target_layers[i]=36u+i;ds->target_slots[i]=myds_metal_buffer_sub_view(ds->target_hidden_buf,i*DIM*sizeof(float),DIM*sizeof(float));if(!ds->target_slots[i])goto failed;}
    return ds;
failed:myds_dspark_free(ds);return NULL;
}

static bool disjoint(myds_metal_buffer_t *a,myds_metal_buffer_t *b) {
    if(!a||!b)return false;
    const uintptr_t p=(uintptr_t)myds_metal_buffer_data(a),q=(uintptr_t)myds_metal_buffer_data(b);
    const size_t n=myds_metal_buffer_size(a),m=myds_metal_buffer_size(b);
    return p&&q&&n&&m&&(p<q?n<=q-p:m<=p-q);
}
static bool checkpoints_disjoint(myds_context_t *ctx,const myds_v41_spec_snapshot_t *s) {
    for(uint32_t o=0;o<4u;o++) {
        myds_metal_buffer_t *live_kv=myds_v41_snapshot_get_previous_kv(ctx->v41,o);
        myds_metal_buffer_t *live_score=myds_v41_snapshot_get_previous_score(ctx->v41,o);
        if(!disjoint(s->first_previous_kv[o],live_kv)||!disjoint(s->first_previous_score[o],live_score))return false;
        if(s->second_state_valid&&(!disjoint(s->second_previous_kv[o],live_kv)||!disjoint(s->second_previous_score[o],live_score)))return false;
    }
    if(!s->second_state_valid)return true;
    if(!disjoint(s->first_residual,s->second_residual)||!disjoint(s->first_pre,s->second_pre)||
       !disjoint(s->first_logits,s->second_logits))return false;
    for(uint32_t o=0;o<4u;o++)if(!disjoint(s->first_previous_kv[o],s->second_previous_kv[o])||
       !disjoint(s->first_previous_score[o],s->second_previous_score[o]))return false;
    for(uint32_t i=0;i<3u;i++)if(!disjoint(s->first_dspark_slots[i],s->second_dspark_slots[i]))return false;
    return true;
}
static void label_case(char label[96],uint32_t arm,uint32_t keep,const char *suffix) {
    snprintf(label,96,"depth2_arm%u_keep%u_%s",arm,keep,suffix);
}
static int checkpoint(uint32_t repeat,const char *label,uint32_t arm,uint32_t keep,uint32_t count,
    const state_t *reference,const myds_v41_spec_snapshot_t *snapshot,bool *pass) {
    CHECK(count==1u||count==2u);
    CHECK(count==1u?snapshot->first_state_valid:snapshot->second_state_valid);
    const char *roles[3]={"logits","residual","pre"};
    myds_metal_buffer_t *views[3]={count==1u?snapshot->first_logits:snapshot->second_logits,
        count==1u?snapshot->first_residual:snapshot->second_residual,count==1u?snapshot->first_pre:snapshot->second_pre};
    for(uint32_t i=0;i<3u;i++) {
        const payload_t *ref=find(reference,roles[i]);CHECK(ref&&views[i]);
        payload_t item={.dtype="f32",.data=myds_metal_buffer_data(views[i]),.bytes=ref->bytes};
        CHECK(item.data&&myds_metal_buffer_size(views[i])>=item.bytes);strcpy(item.role,roles[i]);
        fprintf(report,"{\"schema_version\":1,\"kind\":\"checkpoint_file\",\"repeat\":%u,\"arm\":%u,\"keep_count\":%u,\"checkpoint_count\":%u,\"label\":\"%s\",\"files\":[",
            repeat,arm,keep,count,label);
        CHECK(export_payload(repeat,label,&item,false));fputs("]}\n",report);
        *pass=compare(repeat,label,ref,&item)&&*pass;
    }
    return 1;
}
static int reference_states(myds_context_t *ctx,int32_t seed,uint32_t pos,state_t refs[3],int32_t truth[3]) {
    int32_t token=seed;
    for(uint32_t i=0;i<3u;i++) {
        truth[i]=consume(ctx,token,pos+i);CHECK(truth[i]>=0);
        if(i<2u)CHECK(truth[i]!=ctx->model->tokenizer->eos_id);
        CHECK(capture(ctx,&refs[i]));token=truth[i];
    }
    return 1;
}
static void commit_count_begin(uint32_t fail_at) {
    /* The existing hook counts commit calls only when fail_at is nonzero.
     * UINT32_MAX observes normal head stages without triggering failure. */
    myds_metal_inject_commit_failure(fail_at?fail_at:UINT32_MAX);
}
static uint32_t commit_count_end(void) {
    const uint32_t result=myds_metal_get_commit_call_count();myds_metal_inject_commit_failure(0u);return result;
}
static bool expected_counters(const myds_context_t *ctx,int ret,bool failure) {
    const myds_decode_stats_t *d=&ctx->decode_stats;
    return d->depth2_calls==1u&&d->depth2_hits1==(!failure&&ret==2?1u:0u)&&
        d->depth2_hits2==(!failure&&ret==3?1u:0u)&&d->depth2_misses==(!failure&&ret==1?1u:0u)&&
        d->depth2_failures==(failure?1u:0u)&&d->depth2_fallbacks==0u&&
        d->batch2_calls==1u&&d->batch2_hits==(!failure&&ret>=2?1u:0u)&&
        d->batch2_misses==(!failure&&ret==1?1u:0u)&&d->batch2_failures==(failure?1u:0u)&&
        d->batch2_fallbacks==0u&&d->target_forward_calls==1u&&d->target_rows==3u;
}
static void execution_counters(const myds_context_t *ctx) {
    const myds_decode_stats_t *d=&ctx->decode_stats;
#define COUNT(name) fprintf(report,",\"" #name "\":%" PRIu64,d->name)
    COUNT(depth2_calls);COUNT(depth2_hits1);COUNT(depth2_hits2);COUNT(depth2_misses);
    COUNT(depth2_failures);COUNT(depth2_fallbacks);COUNT(batch2_calls);COUNT(batch2_hits);
    COUNT(batch2_misses);COUNT(batch2_failures);COUNT(batch2_fallbacks);COUNT(target_forward_calls);COUNT(target_rows);
#undef COUNT
}
static void execution_identity(const char *kind,uint32_t repeat,uint32_t arm,uint32_t keep,uint32_t pos,
    int32_t emitted0,int32_t seed,const int32_t truth[3],const int32_t inputs[3],int expected,int ret,
    const int32_t actual[3],uint32_t commits) {
    fprintf(report,"{\"schema_version\":1,\"kind\":\"%s\",\"repeat\":%u,\"arm\":%u,\"keep_count\":%u,\"position\":%u,\"emitted0\":%d,\"seed\":%d,"
        "\"truth\":[%d,%d,%d],\"inputs\":[%d,%d,%d],\"n_inputs\":3,\"expected_ret\":%d,\"ret\":%d,\"actual\":[%d,%d,%d],\"commit_call_count\":%u",
        kind,repeat,arm,keep,pos,emitted0,seed,truth[0],truth[1],truth[2],inputs[0],inputs[1],inputs[2],expected,ret,actual[0],actual[1],actual[2],commits);
}
static int start_native(myds_context_t *ctx,const myds_v41_spec_snapshot_t *base,uint32_t pos,int32_t seed) {
    CHECK(myds_v41_spec_snapshot_restore(ctx,base));ctx->decode_metrics_active=true;ctx->decode_pld_candidate=true;
    memset(&ctx->decode_stats,0,sizeof(ctx->decode_stats));ctx->pcache_tokens[pos]=seed;ctx->pcache_len=pos+1u;return 1;
}
static int export_compare_state(myds_context_t *ctx,uint32_t repeat,const char *label,const state_t *reference,bool *pass) {
    state_t state={0};CHECK(capture(ctx,&state)&&export_state(repeat,label,&state));
    *pass=equal_states(repeat,label,reference,&state)&&*pass;free_state(&state);return 1;
}
static int normal_case(myds_context_t *ctx,const myds_v41_spec_snapshot_t *base,myds_v41_spec_snapshot_t *snap,
    uint32_t repeat,uint32_t arm,uint32_t keep,uint32_t pos,int32_t emitted0,int32_t seed,
    const int32_t truth[3],const state_t refs[3],bool *pass) {
    CHECK(start_native(ctx,base,pos,seed));
    const int32_t wrong1=miss_candidate(ctx->model,truth[0],seed,1u+repeat*3u);
    const int32_t wrong2=miss_candidate(ctx->model,truth[1],wrong1,2u+repeat*3u);CHECK(wrong1>=0&&wrong2>=0);
    const int32_t inputs[3]={seed,arm?truth[0]:wrong1,arm==2u?truth[1]:wrong2};
    const int expected=(int)arm+1;int32_t actual[3]={-1,-1,-1};const int barrier=myds_metal_get_auto_barrier();
    commit_count_begin(0u);const int ret=myds_v41_forward_linear_spec_ex(ctx,inputs,3u,pos,actual,snap);
    const uint32_t commits=commit_count_end();
    bool success=ret==expected&&snap->valid&&snap->first_state_valid&&snap->second_state_valid==(ret>=2)&&
        checkpoints_disjoint(ctx,snap)&&expected_counters(ctx,ret,false)&&commits==(uint32_t)ret+3u&&myds_metal_get_auto_barrier()==barrier;
    for(uint32_t i=0;i<3u;i++)success=success&&actual[i]==(i<(uint32_t)expected?truth[i]:-1);
    execution_identity("execution",repeat,arm,keep,pos,emitted0,seed,truth,inputs,expected,ret,actual,commits);
    execution_counters(ctx);
    fprintf(report,",\"first_state_valid\":%s,\"second_state_valid\":%s,\"checkpoint_buffers_disjoint\":%s,\"caller_auto_barrier_restored\":%s,\"pass\":%s}\n",
        snap->first_state_valid?"true":"false",snap->second_state_valid?"true":"false",checkpoints_disjoint(ctx,snap)?"true":"false",
        myds_metal_get_auto_barrier()==barrier?"true":"false",success?"true":"false");fflush(report);*pass=success&&*pass;CHECK(ret==expected);
    /* Read borrowed checkpoints before partial restoration or scalar forward. */
    char label[96];label_case(label,arm,keep,"checkpoint_first");CHECK(checkpoint(repeat,label,arm,keep,1u,&refs[0],snap,pass));
    if(ret>=2){label_case(label,arm,keep,"checkpoint_second");CHECK(checkpoint(repeat,label,arm,keep,2u,&refs[1],snap,pass));}
    for(uint32_t i=1;i<(uint32_t)ret;i++)ctx->pcache_tokens[pos+i]=truth[i-1];
    ctx->pcache_len=pos+(uint32_t)ret;
    label_case(label,arm,keep,"full");CHECK(export_compare_state(ctx,repeat,label,&refs[ret-1],pass));
    CHECK(myds_v41_spec_snapshot_restore_partial(ctx,snap,keep));
    label_case(label,arm,keep,"partial");CHECK(export_compare_state(ctx,repeat,label,&refs[keep-1],pass));
    ctx->decode_metrics_active=false;
    for(uint32_t count=keep+1u;count<=3u;count++) {
        *pass=consume(ctx,truth[count-2u],pos+count-1u)==truth[count-1u]&&*pass;
        char suffix[32];snprintf(suffix,sizeof(suffix),"resume_%u",count);label_case(label,arm,keep,suffix);
        CHECK(export_compare_state(ctx,repeat,label,&refs[count-1u],pass));
    }
    return 1;
}

/* Whole rollback preserves the legal persistent prefix; its unpublished
 * activation/logit/capture scratch has no pre-step snapshot contract. */
static bool rollback_invalid(const char *role) {
    return !strcmp(role,"logits")||!strcmp(role,"residual")||!strcmp(role,"pre")||!strcmp(role,"slots");
}
static bool equal_prefix(uint32_t repeat,const char *label,const state_t *a,const state_t *b) {
    bool pass=a->count==b->count&&a->pos==b->pos&&a->pcache_len==b->pcache_len;
    for(uint32_t i=0;i<a->count;i++)if(!rollback_invalid(a->items[i].role))
        pass=compare(repeat,label,&a->items[i],find(b,a->items[i].role))&&pass;
    return pass;
}
static int commit_fault_case(myds_context_t *ctx,const myds_v41_spec_snapshot_t *base,
    myds_v41_spec_snapshot_t *snap,uint32_t repeat,uint32_t stage,uint32_t pos,int32_t emitted0,
    int32_t seed,const int32_t truth[3],const state_t *base_ref,const state_t refs[3],bool *pass) {
    CHECK(start_native(ctx,base,pos,seed));const int32_t inputs[3]={seed,truth[0],truth[1]};
    int32_t actual[3]={-1,-1,-1};const uint32_t fail_at=stage+2u;const int barrier=myds_metal_get_auto_barrier();
    commit_count_begin(fail_at);const int ret=myds_v41_forward_linear_spec_ex(ctx,inputs,3u,pos,actual,snap);
    const uint32_t commits=commit_count_end();
    const bool ready=myds_v41_linear_spec_retry_ready(ctx,snap);
    const bool invalid=!ctx->dspark->capture_valid_mask&&!ctx->dspark->capture_committed;
    const bool success=ret==-1&&actual[0]==-1&&actual[1]==-1&&actual[2]==-1&&expected_counters(ctx,ret,true)&&
        commits>=fail_at&&ready&&invalid&&!myds_dspark_capture_ready(ctx->dspark)&&!snap->first_state_valid&&!snap->second_state_valid&&
        ctx->pos==pos&&ctx->pcache_len==pos&&snap->next_position==pos&&myds_metal_get_auto_barrier()==barrier;
    execution_identity("fault_execution",repeat,stage,0u,pos,emitted0,seed,truth,inputs,-1,ret,actual,commits);
    execution_counters(ctx);
    fprintf(report,",\"commit_at\":%u,\"retry_ready\":%s,\"capture_invalid\":%s,\"capture_ready\":false,\"first_state_valid\":%s,\"second_state_valid\":%s,\"caller_auto_barrier_restored\":%s,\"pass\":%s}\n",
        fail_at,ready?"true":"false",invalid?"true":"false",snap->first_state_valid?"true":"false",snap->second_state_valid?"true":"false",
        myds_metal_get_auto_barrier()==barrier?"true":"false",success?"true":"false");fflush(report);*pass=success&&*pass;CHECK(ret==-1&&ready);
    char label[96];snprintf(label,sizeof(label),"fault_stage%u_rolled_base",stage);state_t rolled={0};
    CHECK(capture(ctx,&rolled)&&export_state(repeat,label,&rolled));*pass=equal_prefix(repeat,label,base_ref,&rolled)&&*pass;free_state(&rolled);
    ctx->decode_metrics_active=false;
    for(uint32_t count=1u;count<=3u;count++) {
        *pass=consume(ctx,count==1u?seed:truth[count-2u],pos+count-1u)==truth[count-1u]&&*pass;
        snprintf(label,sizeof(label),"fault_stage%u_retry_%u",stage,count);
        CHECK(export_compare_state(ctx,repeat,label,&refs[count-1u],pass));
    }
    return 1;
}
static int eos_case(myds_context_t *ctx,const myds_v41_spec_snapshot_t *base,myds_v41_spec_snapshot_t *snap,
    uint32_t repeat,uint32_t pos,int32_t emitted0,int32_t seed,const int32_t truth[3],const state_t refs[3],bool *pass) {
    CHECK(start_native(ctx,base,pos,seed));CHECK(truth[0]!=truth[1]);
    const int32_t inputs[3]={seed,truth[0],truth[1]};int32_t actual[3]={-1,-1,-1};
    const int32_t original_eos=ctx->model->tokenizer->eos_id;ctx->model->tokenizer->eos_id=truth[1];
    const int barrier=myds_metal_get_auto_barrier();commit_count_begin(0u);
    const int ret=myds_v41_forward_linear_spec_ex(ctx,inputs,3u,pos,actual,snap);const uint32_t commits=commit_count_end();
    ctx->model->tokenizer->eos_id=original_eos;
    const bool success=ret==2&&actual[0]==truth[0]&&actual[1]==truth[1]&&actual[2]==-1&&expected_counters(ctx,ret,false)&&
        commits==5u&&snap->first_state_valid&&snap->second_state_valid&&checkpoints_disjoint(ctx,snap)&&myds_metal_get_auto_barrier()==barrier;
    execution_identity("eos_execution",repeat,3u,2u,pos,emitted0,seed,truth,inputs,2,ret,actual,commits);execution_counters(ctx);
    fprintf(report,",\"temporary_eos\":%d,\"original_eos\":%d,\"temperature\":0,\"first_state_valid\":%s,\"second_state_valid\":%s,\"checkpoint_buffers_disjoint\":%s,\"caller_auto_barrier_restored\":%s,\"pass\":%s}\n",
        truth[1],original_eos,snap->first_state_valid?"true":"false",snap->second_state_valid?"true":"false",checkpoints_disjoint(ctx,snap)?"true":"false",
        myds_metal_get_auto_barrier()==barrier?"true":"false",success?"true":"false");fflush(report);*pass=success&&*pass;CHECK(ret==2);
    CHECK(checkpoint(repeat,"fault_eos_checkpoint_first",3u,2u,1u,&refs[0],snap,pass)&&
        checkpoint(repeat,"fault_eos_checkpoint_second",3u,2u,2u,&refs[1],snap,pass));
    ctx->pcache_tokens[pos+1u]=truth[0];ctx->pcache_len=pos+2u;
    CHECK(export_compare_state(ctx,repeat,"fault_eos_full",&refs[1],pass));
    CHECK(myds_v41_spec_snapshot_restore_partial(ctx,snap,2u)&&export_compare_state(ctx,repeat,"fault_eos_partial",&refs[1],pass));
    ctx->decode_metrics_active=false;*pass=consume(ctx,truth[1],pos+2u)==truth[2]&&*pass;
    CHECK(export_compare_state(ctx,repeat,"fault_eos_resume_3",&refs[2],pass));return 1;
}
static int run(const char *model_path,const int32_t *ids,const char *token_source) {
    CHECK(!myds_trace_active()&&!getenv("MYDS_TRACE_DIR"));
    const char *fault_env=getenv("MYDS_FIXED_STATE_FAULTS"),*position_env=getenv("MYDS_FIXED_STATE_POSITION");
    CHECK(!fault_env||!strcmp(fault_env,"0")||!strcmp(fault_env,"1"));
    const bool faults=fault_env&&!strcmp(fault_env,"1");uint32_t position=11u;
    if(position_env) {
        char *end=NULL;errno=0;const unsigned long value=strtoul(position_env,&end,10);
        CHECK(!errno&&end!=position_env&&!*end&&value>=11u&&value<=2047u);position=(uint32_t)value;
    }
    const uint32_t capacity=position+3u>512u?position+3u:512u;
    CHECK(setenv("MYDS_TREE_SPEC","0",1)==0&&setenv("MYDS_BATCH2_SPEC","1",1)==0&&
        setenv("MYDS_BATCH2_LAZY_HEAD","1",1)==0&&setenv("MYDS_BATCH2_TRACE_GPU","0",1)==0&&
        setenv("MYDS_DSPARK_CAPTURE_BYPASS",faults?"1":"0",1)==0&&setenv("MYDS_ENGINE_CAPTURE_BYPASS","0",1)==0&&setenv("MYDS_PLD","0",1)==0);
    myds_model_t *model=myds_model_load(model_path);CHECK(model&&model->v41&&model->dim==DIM&&model->n_layers==40u&&model->vocab_size==129280u);
    CHECK(!myds_trace_active());for(uint32_t i=0;i<10u;i++)CHECK(ids[i]>=0&&(uint32_t)ids[i]<model->vocab_size);
    const myds_config_t cfg={.max_seq_len=capacity,.max_new_tokens=8,.temperature=0,.top_p=1,.repetition_penalty=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);CHECK(ctx&&ctx->v41&&!myds_trace_active());
    if(!ctx->pcache_tokens){ctx->pcache_tokens=calloc(capacity,sizeof(int32_t));ctx->pcache_cap=capacity;}CHECK(ctx->pcache_tokens&&ctx->pcache_cap>=capacity);
    ctx->dspark=capture_fixture();CHECK(ctx->dspark);
    fprintf(report,"{\"schema_version\":1,\"kind\":\"metadata\",\"model\":");string(report,model_path);fputs(",\"token_source\":",report);string(report,token_source);
    fputs(",\"prefix_ids\":[",report);for(uint32_t i=0;i<10u;i++)fprintf(report,"%s%d",i?",":"",ids[i]);
    fprintf(report,"],\"prefix_ids_fnv1a64\":\"%016" PRIx64 "\",\"vocab\":%u,\"dim\":5120,\"snapshot_bytes\":%zu,\"prompt_tokens\":10,\"public_prefill_tokens\":9,"
        "\"position\":%u,\"repeats\":3,\"trace_active\":false,\"forced_pld_candidate\":true,\"real_target\":true,\"support_model_loaded\":false,"
        "\"capture_layers\":[36,37,38],\"performance_measurement\":false,\"depth2_mode\":true,\"cases_per_repeat\":%u,"
        "\"unused_third_perturbed_each_repeat\":%s,\"fault_mode\":%s,\"capture_mode\":%u,\"head_commit_indices\":[4,5],\"final_publication_commit_index\":6,\"temperature\":0,\"rollback_invalid_roles\":%s,\"environment\":{",
        fnv(ids,10u*sizeof(int32_t)),model->vocab_size,sizeof(myds_v41_spec_snapshot_t),position,faults?4u:5u,
        faults?"false":"true",faults?"true":"false",faults?1u:0u,faults?"[\"logits\",\"residual\",\"pre\",\"slots\"]":"[]");
    bool comma=false;
    for(char **entry=environ;entry&&*entry;entry++) {
        if(strncmp(*entry,"MYDS_",5))continue;const char *split=strchr(*entry,'=');CHECK(split);
        const size_t length=(size_t)(split-*entry);char key[512];CHECK(length<sizeof(key));memcpy(key,*entry,length);key[length]=0;
        if(comma)fputc(',',report);string(report,key);fputc(':',report);string(report,split+1);comma=true;
    }
    fputs("}}\n",report);fflush(report);
    CHECK(myds_v41_prefill_tokens(ctx,ids,9u,NULL)==1&&ctx->pos==9u);
    memcpy(ctx->pcache_tokens,ids,9u*sizeof(int32_t));ctx->pcache_len=9u;ctx->decode_metrics_active=false;
    int32_t emitted0=consume(ctx,ids[9],9u);CHECK(emitted0>=0);int32_t seed=consume(ctx,emitted0,10u);CHECK(seed>=0);
    while(ctx->pos<position){const int32_t input=seed;seed=consume(ctx,input,ctx->pos);CHECK(seed>=0);emitted0=input;}
    CHECK(ctx->pos==position&&ctx->pcache_len==position);
    myds_v41_spec_snapshot_t *base=calloc(1,sizeof(*base)),*native=calloc(1,sizeof(*native));CHECK(base&&native&&myds_v41_spec_snapshot_save(ctx,base,3u));
    state_t initial[4]={0};bool pass=true;uint32_t completed=0,cases=0;
    const char *scalar_labels[3]={"scalar_first","scalar_second","scalar_third"};
    for(uint32_t repeat=0;repeat<3u;repeat++) {
        state_t base_state={0},refs[3]={0};int32_t truth[3];
        CHECK(myds_v41_spec_snapshot_restore(ctx,base));ctx->decode_metrics_active=false;
        if(faults)CHECK(capture(ctx,&base_state)&&export_state(repeat,"scalar_base",&base_state));
        CHECK(reference_states(ctx,seed,position,refs,truth));
        for(uint32_t i=0;i<3u;i++)CHECK(export_state(repeat,scalar_labels[i],&refs[i]));
        if(repeat) {
            for(uint32_t i=0;i<3u;i++){char label[64];snprintf(label,sizeof(label),"scalar_repeat_%s",i==0u?"first":i==1u?"second":"third");pass=equal_states(repeat,label,&initial[i+1u],&refs[i])&&pass;}
            if(faults)pass=equal_prefix(repeat,"scalar_repeat_base",&initial[0],&base_state)&&pass;
        }
        if(faults) {
            for(uint32_t stage=2u;stage<=4u;stage++){CHECK(commit_fault_case(ctx,base,native,repeat,stage,position,emitted0,seed,truth,&base_state,refs,&pass));cases++;}
            CHECK(eos_case(ctx,base,native,repeat,position,emitted0,seed,truth,refs,&pass));cases++;
        } else for(uint32_t arm=0;arm<3u;arm++)for(uint32_t keep=1u;keep<=(arm?2u:1u);keep++) {
            CHECK(normal_case(ctx,base,native,repeat,arm,keep,position,emitted0,seed,truth,refs,&pass));cases++;
        }
        if(!repeat){initial[0]=base_state;memset(&base_state,0,sizeof(base_state));for(uint32_t i=0;i<3u;i++){initial[i+1u]=refs[i];memset(&refs[i],0,sizeof(refs[i]));}}
        free_state(&base_state);for(uint32_t i=0;i<3u;i++)free_state(&refs[i]);completed++;
    }
    fprintf(report,"{\"schema_version\":1,\"kind\":\"%s\",\"repeats_completed\":%u,\"cases_completed\":%u,\"raw_files\":%u,\"same_base_snapshot\":true,"
        "\"depth2_mode\":true,\"trace_active\":false,\"performance_measurement\":false,\"bitwise_logits_and_visible_state\":%s,\"pass\":%s}\n",
        faults?"depth2_fault_complete":"depth2_complete",completed,cases,raw_count,pass?"true":"false",pass?"true":"false");fflush(report);
    for(uint32_t i=0;i<4u;i++)free_state(&initial[i]);free(base);free(native);myds_context_free(ctx);myds_model_free(model);return pass;
}
int main(int argc,char **argv) {
    if(argc!=4){fputs("Usage: test_depth2_fixed_state MODEL TOKEN_FILE_OR_DASH NEW_OUTPUT_DIR\n",stderr);return 2;}
    int32_t ids[10];trace_tokens tokens={0};
    if(!strcmp(argv[2],"-"))memcpy(ids,default_ids,sizeof(ids));
    else {if(!trace_tokens_file(&tokens,argv[2])||tokens.count<10u){free(tokens.ids);return 2;}memcpy(ids,tokens.ids,sizeof(ids));free(tokens.ids);}
    output=argv[3];if(mkdir(output,0700)){perror("new output directory");return 2;}
    fflush(stdout);const int fd=dup(STDOUT_FILENO);report=fd<0?NULL:fdopen(fd,"w");if(!report||dup2(STDERR_FILENO,STDOUT_FILENO)<0)return 2;
    const int ok=run(argv[1],ids,!strcmp(argv[2],"-")?"archived_state_fixture_first_10_ids":argv[2]);fclose(report);return ok?0:1;
}
