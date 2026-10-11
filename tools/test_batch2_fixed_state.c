/* Fixed-base real target scalar/Batch2 equivalence gate. No drafter, trace,
 * scheduler or benchmark timing. MODEL TOKEN_FILE_OR_DASH NEW_OUTPUT_DIR.
 * '-' selects the first ten IDs of the archived target state fixture. */
#define _POSIX_C_SOURCE 200809L
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_dspark.h"
#include "reference/trace_input.h"
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
static int run(const char *model_path,const int32_t *ids,const char *token_source) {
    CHECK(!myds_trace_active()&&!getenv("MYDS_TRACE_DIR"));
    CHECK(setenv("MYDS_BATCH2_SPEC","1",1)==0&&setenv("MYDS_DSPARK_CAPTURE_BYPASS","0",1)==0&&setenv("MYDS_ENGINE_CAPTURE_BYPASS","0",1)==0);
    myds_model_t *model=myds_model_load(model_path);CHECK(model&&model->v41&&model->dim==DIM&&model->n_layers==40);
    CHECK(!myds_trace_active());
    for(uint32_t i=0;i<10u;i++)CHECK(ids[i]>=0&&(uint32_t)ids[i]<model->vocab_size);
    const myds_config_t cfg={.max_seq_len=512,.max_new_tokens=8,.temperature=0,.top_p=1,.repetition_penalty=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);CHECK(ctx&&ctx->v41);
    CHECK(!myds_trace_active());
    if(!ctx->pcache_tokens){ctx->pcache_tokens=calloc(512u,sizeof(int32_t));ctx->pcache_cap=512u;}CHECK(ctx->pcache_tokens);
    ctx->dspark=capture_fixture();CHECK(ctx->dspark);
    fprintf(report,"{\"schema_version\":1,\"kind\":\"metadata\",\"model\":");string(report,model_path);
    fputs(",\"token_source\":",report);string(report,token_source);
    fprintf(report,",\"prefix_ids\":[");for(uint32_t i=0;i<10u;i++)fprintf(report,"%s%d",i?",":"",ids[i]);
    fprintf(report,"],\"prefix_ids_fnv1a64\":\"%016" PRIx64 "\",\"vocab\":%u,\"dim\":5120,\"snapshot_bytes\":%zu,\"prompt_tokens\":10,\"public_prefill_tokens\":9,"
        "\"position\":11,\"repeats\":3,\"trace_active\":false,\"forced_pld_candidate\":true,\"real_target\":true,"
        "\"support_model_loaded\":false,\"capture_layers\":[36,37,38],\"performance_measurement\":false,\"environment\":{",fnv(ids,10u*sizeof(int32_t)),model->vocab_size,sizeof(myds_v41_spec_snapshot_t));
    bool comma=false;
    for(char **entry=environ;entry&&*entry;entry++) {
        if(strncmp(*entry,"MYDS_",5))continue;
        const char *split=strchr(*entry,'=');CHECK(split);const size_t length=(size_t)(split-*entry);
        char key[512];CHECK(length<sizeof(key));memcpy(key,*entry,length);key[length]=0;
        if(comma)fputc(',',report);string(report,key);fputc(':',report);string(report,split+1);comma=true;
    }
    fputs("}}\n",report);fflush(report);
    CHECK(myds_v41_prefill_tokens(ctx,ids,9u,NULL)==1&&ctx->pos==9u);
    memcpy(ctx->pcache_tokens,ids,9u*sizeof(int32_t));ctx->pcache_len=9u;
    ctx->decode_metrics_active=false;
    const int32_t emitted0=consume(ctx,ids[9],9u);CHECK(emitted0>=0);
    const int32_t seed=consume(ctx,emitted0,10u);CHECK(seed>=0&&ctx->pos==11u);
    myds_v41_spec_snapshot_t *base=calloc(1,sizeof(*base)),*batch=calloc(1,sizeof(*batch));CHECK(base&&batch);
    CHECK(myds_v41_spec_snapshot_save(ctx,base,2u));
    state_t first0={0},second0={0};bool pass=true;uint32_t completed=0;
    for(uint32_t repeat=0;repeat<3u;repeat++) {
        state_t first={0},second={0},full={0},partial={0},continued={0};
        CHECK(myds_v41_spec_snapshot_restore(ctx,base));ctx->decode_metrics_active=false;
        const int32_t truth=consume(ctx,seed,11u);CHECK(truth>=0&&truth!=model->tokenizer->eos_id);
        CHECK(capture(ctx,&first)&&export_state(repeat,"scalar_first",&first));
        const int32_t next=consume(ctx,truth,12u);CHECK(next>=0);
        CHECK(capture(ctx,&second)&&export_state(repeat,"scalar_second",&second));
        if(repeat){pass=equal_states(repeat,"scalar_repeat_first",&first0,&first)&&pass;pass=equal_states(repeat,"scalar_repeat_second",&second0,&second)&&pass;}
        CHECK(myds_v41_spec_snapshot_restore(ctx,base));ctx->decode_pld_candidate=true;ctx->decode_metrics_active=true;
        memset(&ctx->decode_stats,0,sizeof(ctx->decode_stats));ctx->pcache_tokens[11]=seed;ctx->pcache_len=12u;
        const int barrier=myds_metal_get_auto_barrier();const int32_t inputs[2]={seed,truth};int32_t actual[2]={-1,-1};
        const int result=myds_v41_forward_batch_spec_ex(ctx,inputs,2u,11u,actual,batch);
        const bool execution=result==2&&actual[0]==truth&&actual[1]==next&&batch->first_state_valid&&batch->first_logits&&
            ctx->decode_stats.batch2_calls==1&&ctx->decode_stats.batch2_hits==1&&ctx->decode_stats.batch2_failures==0&&
            ctx->decode_stats.batch2_fallbacks==0&&ctx->decode_stats.target_forward_calls==1&&ctx->decode_stats.target_rows==2&&
            myds_metal_get_auto_barrier()==barrier;
        fprintf(report,"{\"schema_version\":1,\"kind\":\"execution\",\"repeat\":%u,\"position\":11,\"emitted0\":%d,\"seed\":%d,\"truth\":%d,\"next\":%d,"
            "\"ret\":%d,\"actual\":[%d,%d],\"batch2_calls\":%" PRIu64 ",\"batch2_hits\":%" PRIu64 ",\"batch2_failures\":%" PRIu64 ",\"batch2_fallbacks\":%" PRIu64 ","
            "\"target_forward_calls\":%" PRIu64 ",\"target_rows\":%" PRIu64 ",\"caller_auto_barrier_restored\":%s,\"pass\":%s}\n",
            repeat,emitted0,seed,truth,next,result,actual[0],actual[1],ctx->decode_stats.batch2_calls,ctx->decode_stats.batch2_hits,
            ctx->decode_stats.batch2_failures,ctx->decode_stats.batch2_fallbacks,ctx->decode_stats.target_forward_calls,ctx->decode_stats.target_rows,
            myds_metal_get_auto_barrier()==barrier?"true":"false",execution?"true":"false");fflush(report);
        pass=execution&&pass;
        /* A successful ret2 exposes the exact first checkpoint. Compare it
         * before a later operation can overwrite its borrowed workspace. */
        if(result>0&&batch->first_state_valid&&batch->first_logits) {
            const char *roles[3]={"logits","residual","pre"};myds_metal_buffer_t *views[3]={batch->first_logits,batch->first_residual,batch->first_pre};
            for(uint32_t i=0;i<3u;i++) {
                const payload_t *ref=find(&first,roles[i]);payload_t item={.dtype="f32",.data=myds_metal_buffer_data(views[i]),.bytes=ref->bytes};strcpy(item.role,roles[i]);
                fprintf(report,"{\"schema_version\":1,\"kind\":\"checkpoint_file\",\"repeat\":%u,\"files\":[",repeat);CHECK(export_payload(repeat,"checkpoint_first",&item,false));fputs("]}\n",report);
                pass=compare(repeat,"batch2_checkpoint_first",ref,&item)&&pass;
            }
        }
        if(result==2) {
            ctx->pcache_tokens[12]=truth;ctx->pcache_len=13u;CHECK(capture(ctx,&full)&&export_state(repeat,"batch2_second",&full));
            pass=equal_states(repeat,"batch2_second",&second,&full)&&pass;
            CHECK(myds_v41_spec_snapshot_restore_partial(ctx,batch,1u));
            CHECK(capture(ctx,&partial)&&export_state(repeat,"partial_first",&partial));pass=equal_states(repeat,"partial_first",&first,&partial)&&pass;
            ctx->decode_metrics_active=false;const int32_t resumed=consume(ctx,truth,12u);pass=resumed==next&&pass;
            CHECK(capture(ctx,&continued)&&export_state(repeat,"continued_second",&continued));pass=equal_states(repeat,"continued_second",&second,&continued)&&pass;
        } else if(result==1) {
            /* A differing first argmax cannot satisfy the forced scalar truth
             * candidate. Preserve its real miss state and still diagnose the
             * same explicit continuation input, without pretending ret2. */
            CHECK(capture(ctx,&partial)&&export_state(repeat,"batch2_miss_first",&partial));
            pass=equal_states(repeat,"batch2_miss_first",&first,&partial)&&pass;
            ctx->decode_metrics_active=false;const int32_t resumed=consume(ctx,truth,12u);pass=resumed==next&&pass;
            CHECK(capture(ctx,&continued)&&export_state(repeat,"continued_after_miss",&continued));
            pass=equal_states(repeat,"continued_after_miss",&second,&continued)&&pass;
        }
        if(!repeat){first0=first;second0=second;memset(&first,0,sizeof(first));memset(&second,0,sizeof(second));}
        free_state(&first);free_state(&second);free_state(&full);free_state(&partial);free_state(&continued);completed++;
    }
    fprintf(report,"{\"schema_version\":1,\"kind\":\"complete\",\"repeats_completed\":%u,\"raw_files\":%u,\"same_base_snapshot\":true,"
        "\"trace_active\":false,\"performance_measurement\":false,\"bitwise_logits_and_visible_state\":%s,\"pass\":%s}\n",completed,raw_count,pass?"true":"false",pass?"true":"false");fflush(report);
    free_state(&first0);free_state(&second0);free(base);free(batch);myds_context_free(ctx);myds_model_free(model);return pass;
}
int main(int argc,char **argv) {
    if(argc!=4){fputs("Usage: test_batch2_fixed_state MODEL TOKEN_FILE_OR_DASH NEW_OUTPUT_DIR\n",stderr);return 2;}
    int32_t ids[10];trace_tokens tokens={0};
    if(!strcmp(argv[2],"-"))memcpy(ids,default_ids,sizeof(ids));
    else {if(!trace_tokens_file(&tokens,argv[2])||tokens.count<10u){free(tokens.ids);return 2;}memcpy(ids,tokens.ids,sizeof(ids));free(tokens.ids);}
    output=argv[3];if(mkdir(output,0700)){perror("new output directory");return 2;}
    fflush(stdout);const int fd=dup(STDOUT_FILENO);report=fd<0?NULL:fdopen(fd,"w");if(!report||dup2(STDERR_FILENO,STDOUT_FILENO)<0)return 2;
    const int ok=run(argv[1],ids,!strcmp(argv[2],"-")?"archived_state_fixture_first_10_ids":argv[2]);fclose(report);return ok?0:1;
}
