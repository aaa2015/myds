/* Same-context cancellation/recovery correctness only. The included harness
 * supplies byte capture/state-file helpers, not its main or measurement path.
 * No extra commit/wait/trace operation is inserted into generation. */
#define MYDS_MOE_WORKSPACE_TESTING 1
#define MYDS_BENCH_MOE_WS_API 1
#define main moe_workspace_fullmodel_inherited_main
#include "bench_moe_workspace_fullmodel.c"
#undef main
#include <stdatomic.h>
#include "moe_workspace_layout_promptclear.inc"

#ifndef MYDS_MOE_WS_CANCEL_HELPERS_ONLY
typedef struct {
    volatile int *flag;
    atomic_int done;
    uint64_t target,observed;
    double delay_s,started,flag_time;
    bool set,timed_out;
} cancel_thread_t;
#endif
static void cancel_need(bool value,const char *message) {
    if(!value){fprintf(stderr,"test_moe_workspace_cancel: %s\n",message);exit(1);}
}
#include "moe_prompt_snapshot_promptclear.inc"
#ifndef MYDS_MOE_WS_CANCEL_HELPERS_ONLY
static void *cancel_worker(void *opaque) {
    cancel_thread_t *c=opaque;const struct timespec interval={0,1000000};
    while(!atomic_load_explicit(&c->done,memory_order_acquire)) {
        uint64_t progress=myds_v41_moe_workspace_progress();double elapsed=now_s()-c->started;
        if((progress>=c->target&&elapsed>=c->delay_s)||elapsed>=120.0) {
            c->observed=progress;c->flag_time=now_s();c->timed_out=elapsed>=120.0;c->set=true;
            *c->flag=1;return NULL; /* Existing production cancellation API. */
        }
        nanosleep(&interval,NULL);
    }
    return NULL;
}
#endif
static char *render(const char *path,size_t *bytes,uint64_t *hash) {
    size_t n=0;char *body=read_request(path,&n);cancel_need(body&&n<(SIZE_MAX-8192u)/4u,"request bytes");
    *bytes=n;*hash=hash_bytes(UINT64_C(14695981039346656037),body,n);
    size_t capacity=n*4u+8192u;if(capacity<65536u)capacity=65536u;
    char *prompt=malloc(capacity);cancel_need(prompt&&parse_chat_prompt(body,prompt,capacity),"production prompt renderer");free(body);return prompt;
}
/* A cancelled prefill does not advance next_position at its success epilogue.
 * Its live rings are partly updated at different layers. Export only metadata,
 * live CPU Engram/RNG, and published snapshot payload at each snapshot's p.
 * Complete live ring/cache exports resume only after recovery succeeds. */
static bool cancel_state(const char *root,const char *label,myds_context_t *ctx,
                         const capture_t *c,bool complete,uint32_t requested_upper) {
    /* Pure status query, never a repair/drain. Refuse to read shared GPU memory
     * if an abnormal existing generation return still has work in flight. */
    if(!myds_metal_moe_workspace_is_quiescent())return false;
    char directory[4096],path[4096],name[128];
    int n=snprintf(directory,sizeof(directory),"%s/%s",root,label);if(n<0||(size_t)n>=sizeof(directory)||mkdir(directory,0700))return false;
    n=snprintf(path,sizeof(path),"%s/manifest.json",directory);if(n<0||(size_t)n>=sizeof(path))return false;
    FILE *out=fopen(path,"wbx");if(!out)return false;
    uint32_t next=myds_v41_snapshot_get_next_position(ctx->v41),history[3]={0};myds_v41_snapshot_get_engram_history(ctx->v41,history);
    uint32_t positions[MYDS_SNAP_SLOTS*2u];
    for(uint32_t s=0;s<MYDS_SNAP_SLOTS;s++){positions[2*s]=myds_v41_snapshot_get_pos(ctx->v41,s);positions[2*s+1]=myds_v41_snapshot_get_staged_pos(ctx->v41,s);}
    uint64_t scalars[]={ctx->pos,ctx->rng_state,ctx->pcache_len,ctx->anchor_len,next,
        myds_v41_snapshot_get_engram_position(ctx->v41),myds_v41_snapshot_get_valid(ctx->v41)?1u:0u,
        ctx->cfg.max_seq_len,ctx->cfg.max_new_tokens,(uint32_t)ctx->cfg.seed,ctx->continue_from_cache?1u:0u};
    bool ok=ctx->pcache_len<=ctx->pcache_cap&&ctx->anchor_len<=1024u&&next<=ctx->cfg.max_seq_len;unsigned files_count=0;
    fprintf(out,"{\"schema_version\":1,\"phase\":");json_string(out,label);
    fprintf(out,",\"scope\":\"same_context_prefill_recovery\",\"complete_live_state\":%s,\"ctx_pos\":%u,\"next_position\":%u,\"pcache_len\":%u,\"anchor_len\":%u,\"anchor_slot\":%u,\"requested_prefill_upper_bound\":%u,\"rng_state\":\"%016" PRIx64 "\",\"engram_position\":%u,\"limitations\":[\"Interrupted partially updated live window/cache/carry, inactive and unpublished staged capacity are not compared. Published snapshots use their own positions.\",\"Persistent disk-cache payload and allocator padding are not exported.\"],\"scalar_order\":[\"ctx_pos\",\"rng\",\"pcache_len\",\"anchor_len\",\"next\",\"engram_position\",\"v41_valid\",\"max_seq_len\",\"max_new_tokens\",\"seed\",\"continue_from_cache\"],\"files\":[",
        complete?"true":"false",ctx->pos,next,ctx->pcache_len,ctx->anchor_len,(unsigned)MYDS_V41_ANCHOR_SLOT,requested_upper,ctx->rng_state,myds_v41_snapshot_get_engram_position(ctx->v41));
#define CSF(file,data,bytes) do{if(ok)ok=state_file(out,directory,file,data,bytes,&files_count);}while(0)
#define CSM(file,buffer,bytes) do{if(ok)ok=state_metal(out,directory,file,buffer,bytes,&files_count);}while(0)
    CSF("context-scalars.u64le",scalars,sizeof(scalars));CSF("engram-tail.u32le",history,sizeof(history));
    CSF("snapshot-positions.u32le",positions,sizeof(positions));
    CSF("prefix-tokens.i32le",ctx->pcache_tokens,(size_t)ctx->pcache_len*4u);CSF("anchor-tokens.i32le",ctx->anchor_tokens,(size_t)ctx->anchor_len*4u);
    if(c){CSF("tokens.i32le",c->ids,(size_t)c->count*4u);CSF("output.bin",c->raw_bytes,(size_t)c->byte_count);}
    if(complete) {
        for(uint32_t layer=0;ok&&layer<40;layer++){snprintf(name,sizeof(name),"layer-%02u-window-active.f32",layer);ok=state_window(out,directory,name,myds_v41_snapshot_get_window(ctx->v41,layer),0,next,&files_count);}
        for(uint32_t owner=0;ok&&owner<4;owner++) {
            uint32_t rows=next/(owner==3?1u:2u);
            if(owner<3){snprintf(name,sizeof(name),"owner-%u-previous-kv.f32",owner);CSM(name,myds_v41_snapshot_get_previous_kv(ctx->v41,owner),2048u);snprintf(name,sizeof(name),"owner-%u-previous-score.f32",owner);CSM(name,myds_v41_snapshot_get_previous_score(ctx->v41,owner),2048u);}
            snprintf(name,sizeof(name),"owner-%u-compressed-active.bf16",owner);CSM(name,myds_v41_snapshot_get_compressed(ctx->v41,owner),(size_t)rows*1024u);
            snprintf(name,sizeof(name),"owner-%u-index-active.bf16",owner);CSM(name,myds_v41_snapshot_get_index_cache(ctx->v41,owner),(size_t)rows*256u);
        }
        const float *logits=myds_metal_buffer_data(ctx->logits);bool finite=logits!=NULL;
        for(uint32_t i=0;finite&&i<ctx->model->vocab_size;i++)if(!isfinite(logits[i]))finite=false;
        if(!finite)ok=false;CSF("final-logits.f32",logits,(size_t)ctx->model->vocab_size*4u);
    }
    for(uint32_t s=0;ok&&s<MYDS_SNAP_SLOTS;s++) {
        uint32_t p=positions[2*s],staged=positions[2*s+1];
        /* The persistent anchor is intentionally retained across requests and
         * can be beyond a subsequent shorter request's current next position. */
        bool anchor=s==MYDS_V41_ANCHOR_SLOT;
        if((p&&staged)||(!anchor&&p>requested_upper)||staged>requested_upper||
           (anchor&&p&&p!=ctx->anchor_len)){ok=false;break;}if(!p)continue;
        uint32_t tail[3]={0},position=0,h[4]={0};
        if(!myds_v41_snapshot_get_slot_engram(ctx->v41,s,tail,&position)||position!=p){ok=false;break;}
        h[0]=position;memcpy(h+1,tail,sizeof(tail));snprintf(name,sizeof(name),"snapshot-%02u-engram-position-tail.u32le",s);CSF(name,h,sizeof(h));
        myds_metal_buffer_t *window=myds_v41_snapshot_get_snap_window(ctx->v41,s);
        for(uint32_t layer=0;ok&&layer<40;layer++){snprintf(name,sizeof(name),"snapshot-%02u-layer-%02u-window-active.f32",s,layer);ok=state_window(out,directory,name,window,(size_t)layer*128u*512u*4u,p,&files_count);}
        for(uint32_t owner=0;ok&&owner<3;owner++){snprintf(name,sizeof(name),"snapshot-%02u-owner-%u-previous-kv.f32",s,owner);CSM(name,myds_v41_snapshot_get_snap_prev_kv(ctx->v41,s,owner),2048u);snprintf(name,sizeof(name),"snapshot-%02u-owner-%u-previous-score.f32",s,owner);CSM(name,myds_v41_snapshot_get_snap_prev_score(ctx->v41,s,owner),2048u);}
    }
#undef CSF
#undef CSM
    fprintf(out,"],\"file_count\":%u,\"complete\":%s}\n",files_count,ok?"true":"false");if(fclose(out))ok=false;return ok;
}
#ifndef MYDS_MOE_WS_CANCEL_HELPERS_ONLY
int main(int argc,char **argv) {
    const char *model_path=NULL,*warm_request=NULL,*cancel_request=NULL,*recovery_request=NULL,*out_dir=NULL,*scenario=NULL;
    uint32_t checkpoint=0,recover_tokens=0,delay_ms=0;float temperature=-1;bool run=false;
    for(int i=1;i<argc;i++) {
        const char *key=argv[i];if(!strcmp(key,"--run")){run=true;continue;}
        cancel_need(++i<argc,"option value");const char *value=argv[i];
        if(!strcmp(key,"--model"))model_path=value;else if(!strcmp(key,"--warm-request"))warm_request=value;else if(!strcmp(key,"--cancel-request"))cancel_request=value;
        else if(!strcmp(key,"--recover-request"))recovery_request=value;else if(!strcmp(key,"--out-dir"))out_dir=value;
        else if(!strcmp(key,"--scenario"))scenario=value;else if(!strcmp(key,"--checkpoint"))checkpoint=positive(value,100000,"checkpoint");
        else if(!strcmp(key,"--recover-tokens"))recover_tokens=positive(value,32,"recovery tokens");
        else if(!strcmp(key,"--delay-ms")){char *end=NULL;unsigned long x=strtoul(value,&end,10);cancel_need(*value&&!*end&&x<=10000,"delay ms");delay_ms=(uint32_t)x;}
        else if(!strcmp(key,"--temperature")){if(!strcmp(value,"0"))temperature=0;else if(!strcmp(value,"0.7"))temperature=.7f;else cancel_need(false,"temperature 0 or 0.7");}
        else cancel_need(false,"unknown option");
    }
    cancel_need(run&&model_path&&warm_request&&cancel_request&&recovery_request&&out_dir&&out_dir[0]=='/'&&checkpoint&&temperature>=0&&(recover_tokens==8||recover_tokens==32)&&scenario&&(!strcmp(scenario,"deterministic")||!strcmp(scenario,"async")),"explicit --run model/requests/new absolute out/scenario/checkpoint/temperature/recover-tokens required");
    const int clear_mode=promptclear_mode();
    cancel_need(mkdir(out_dir,0700)==0,"new output directory");char path[4096];int n=snprintf(path,sizeof(path),"%s/summary.json",out_dir);cancel_need(n>0&&(size_t)n<sizeof(path),"summary path");FILE *summary=fopen(path,"wbx");cancel_need(summary!=NULL,"new summary");
    size_t warm_request_bytes=0,cancel_request_bytes=0,recover_request_bytes=0;uint64_t warm_request_hash=0,cancel_request_hash=0,recover_request_hash=0;
    char *warm_prompt=render(warm_request,&warm_request_bytes,&warm_request_hash);
    char *cancel_prompt=render(cancel_request,&cancel_request_bytes,&cancel_request_hash),*recover_prompt=render(recovery_request,&recover_request_bytes,&recover_request_hash);
    const char *forbidden[]={"MYDS_TRACE_PREFIX","MYDS_BATCH_TRACE_PREFIX","MYDS_PROFILE_PREFILL","MYDS_PROFILE_SPLIT","MYDS_PROFILE_SPLIT2","MYDS_PROFILE_SPLIT3","MYDS_MOE_STEP_PROF","MYDS_KPROF","MYDS_MOE_TILE_PROF"};
    for(unsigned i=0;i<sizeof(forbidden)/sizeof(*forbidden);i++)cancel_need(!getenv(forbidden[i]),"diagnostic trace/profiling environment forbidden");
    myds_model_t *model=myds_model_load(model_path);cancel_need(model&&model->v41,"real V4.1 model");
    promptclear_initialized_mode(clear_mode);
    myds_config_t cfg={.max_seq_len=327680,.max_new_tokens=recover_tokens,.temperature=temperature,.top_p=1,.top_k=0,.min_p=0,.repetition_penalty=1,.seed=42,.stream=false};
    myds_context_t *ctx=myds_context_create(model,&cfg);cancel_need(ctx&&ctx->v41&&!ctx->dspark,"one context, ordinary target only");
    cancel_need(myds_metal_moe_workspace_test_control(0,0),"idle clear faults before same-context warm");
    capture_t warm={.bytes_hash=UINT64_C(14695981039346656037),.ids_hash=UINT64_C(14695981039346656037)};
    warm.raw_bytes=malloc(MAX_OUTPUT_BYTES);cancel_need(warm.raw_bytes!=NULL,"warm capture");
    myds_stop_reason_t warm_reason=MYDS_STOP_ERROR;uint32_t warm_tokens=0;
    int warm_api=myds_generate_bytes_ex(ctx,warm_prompt,capture,&warm,&warm_reason,&warm_tokens,NULL);
    bool warm_ok=warm_api&&warm_reason==MYDS_STOP_LENGTH&&warm.count==recover_tokens&&!warm.overflow&&warm_tokens==6917;
    bool warm_export=cancel_state(out_dir,"before-cancel",ctx,&warm,true,myds_v41_snapshot_get_next_position(ctx->v41));
    myds_v41_prompt_growth_event_t pw={0},pf={0},pr={0};
    cancel_need(promptclear_snapshot(out_dir,"before-cancel",ctx,&pw),"warm prompt identity");
    localgrow_layout_t warm_layout={0},cancel_layout={0},recover_layout={0};bool wp=false,cp=false,rp=false;
    n=snprintf(path,sizeof(path),"%s/layout-before-cancel.json",out_dir);
    cancel_need(n>0&&(size_t)n<sizeof(path)&&localgrow_dump_layout(path,ctx,&warm_layout,&wp),"warm actual layout");
    cancel_need(warm_ok&&warm_export&&wp&&warm_layout.capacity==2048&&warm_layout.moe_batch==8192,"actual same-context warm before cancellation");
    fputs("{\"schema_version\":1,\"probe\":\"same_context_moe_workspace_cancel_promptclear\",\"scenario\":",summary);json_string(summary,scenario);
    fprintf(summary,",\"warm_generation_ok\":%s,\"warm_state_exported\":%s,\"warm_prompt_tokens\":%u,\"warm_output_tokens\":%u,\"warm_workspace\":",warm_ok?"true":"false",warm_export?"true":"false",warm_tokens,warm.count);bench_ws_json(summary,ctx);fflush(summary);free(warm.raw_bytes);
    volatile int cancel_flag=0;bool async=!strcmp(scenario,"async");cancel_thread_t thread={.flag=&cancel_flag,.target=checkpoint,.delay_s=delay_ms/1000.0};atomic_init(&thread.done,0);pthread_t worker;
    cancel_need(myds_v41_moe_workspace_cancel_control(async?0u:checkpoint,async?NULL:&cancel_flag),"idle progress reset/cancel hook registration");
    capture_t cancelled={.bytes_hash=UINT64_C(14695981039346656037),.ids_hash=UINT64_C(14695981039346656037)};cancelled.raw_bytes=malloc(MAX_OUTPUT_BYTES);cancel_need(cancelled.raw_bytes!=NULL,"cancel capture memory");
    thread.started=now_s();if(async)cancel_need(pthread_create(&worker,NULL,cancel_worker,&thread)==0,"async flag thread");
    myds_stop_reason_t reason=MYDS_STOP_ERROR;uint32_t prompt_tokens=0;
    int ok=myds_generate_bytes_ex(ctx,cancel_prompt,capture,&cancelled,&reason,&prompt_tokens,&cancel_flag);
    uint64_t observed=myds_v41_moe_workspace_progress();
    if(async){atomic_store_explicit(&thread.done,1,memory_order_release);cancel_need(pthread_join(worker,NULL)==0,"join CPU cancellation thread");}
    bool cancel_ok=ok&&reason==MYDS_STOP_CANCELLED&&cancelled.count==0&&cancel_flag&&prompt_tokens>1&&ctx->pcache_len==0&&observed>=checkpoint&&(!async||(thread.set&&!thread.timed_out));
    cancel_need(promptclear_snapshot(out_dir,"after-cancel",ctx,&pf),"cancel prompt identity");
    bool cancelled_state=cancel_state(out_dir,"after-cancel",ctx,&cancelled,false,prompt_tokens?prompt_tokens-1u:0);
    n=snprintf(path,sizeof(path),"%s/layout-after-cancel.json",out_dir);
    bool cancel_layout_ok=n>0&&(size_t)n<sizeof(path)&&localgrow_dump_layout(path,ctx,&cancel_layout,&cp)&&cp&&cancel_layout.moe_batch==16384;
    bool actual_growth=cancel_layout_ok&&(!bench_ws_mode()||(localgrow_stable(&warm_layout,&cancel_layout)&&cancel_layout.moe_generation==warm_layout.moe_generation+1&&cancel_layout.localgrow_successes==warm_layout.localgrow_successes+1));
    cancel_ok=cancel_ok&&prompt_tokens==15187&&actual_growth;
    /* Always preserve cancellation facts before accepting or rejecting recovery. */
    
    fprintf(summary,",\"vocab_size\":%u,\"snapshot_slots\":%u,\"request_inputs\":{\"warm\":{\"bytes\":%zu,\"fnv1a64_bytes\":\"%016" PRIx64 "\"},\"cancel\":{\"bytes\":%zu,\"fnv1a64_bytes\":\"%016" PRIx64 "\"},\"recover\":{\"bytes\":%zu,\"fnv1a64_bytes\":\"%016" PRIx64 "\"}}",model->vocab_size,(unsigned)MYDS_SNAP_SLOTS,warm_request_bytes,warm_request_hash,cancel_request_bytes,cancel_request_hash,recover_request_bytes,recover_request_hash);
    fprintf(summary,",\"implementation_version\":4,\"prompt_view_clear_mode\":%d",clear_mode);
    fprintf(summary,",\"workspace_mode\":%d,\"temperature\":%.9g,\"seed\":42,\"recovery_tokens\":%u,\"same_context\":true,\"rng_manually_reset\":false,\"extra_gpu_waits_inserted\":0,\"performance_measured\":false,\"target_checkpoint\":%u,\"observed_progress\":%" PRIu64 ",\"thread_flag_progress\":%" PRIu64 ",\"async_thread_set_flag\":%s,\"thread_timed_out\":%s,\"cancel_api_ok\":%s,\"cancel_stop_reason\":",bench_ws_mode(),temperature,recover_tokens,checkpoint,observed,thread.observed,thread.set?"true":"false",thread.timed_out?"true":"false",ok?"true":"false");json_string(summary,stop_name(reason));
    fprintf(summary,",\"cancel_output_tokens\":%u,\"cancel_prompt_tokens\":%u,\"cancel_gate_pass\":%s,\"cancel_state_exported\":%s,\"cancel_workspace\":",cancelled.count,prompt_tokens,cancel_ok?"true":"false",cancelled_state?"true":"false");bench_ws_json(summary,ctx);fprintf(summary,",\"same_context_warm_before_cancel\":true,\"actual_B8192_to16384_growth_before_cancel\":%s",actual_growth?"true":"false");fflush(summary);
    cancel_need(myds_v41_moe_workspace_cancel_control(0,NULL),"unregister hook while idle");free(cancelled.raw_bytes);
    int status=1;
    if(cancel_ok&&cancelled_state) {
        capture_t recovered={.bytes_hash=UINT64_C(14695981039346656037),.ids_hash=UINT64_C(14695981039346656037)};recovered.raw_bytes=malloc(MAX_OUTPUT_BYTES);cancel_need(recovered.raw_bytes!=NULL,"recovery capture memory");
        uint32_t recovery_prompt_tokens=0;reason=MYDS_STOP_ERROR;
        ok=myds_generate_bytes_ex(ctx,recover_prompt,capture,&recovered,&reason,&recovery_prompt_tokens,NULL);
        bool generation_ok=ok&&reason==MYDS_STOP_LENGTH&&recovered.count==recover_tokens&&!recovered.overflow;
        cancel_need(promptclear_snapshot(out_dir,"recovery",ctx,&pr),"recovery prompt identity");
        bool exported=cancel_state(out_dir,"recovery",ctx,&recovered,true,myds_v41_snapshot_get_next_position(ctx->v41));
        n=snprintf(path,sizeof(path),"%s/layout-recovery.json",out_dir);
        bool layout_ok=n>0&&(size_t)n<sizeof(path)&&localgrow_dump_layout(path,ctx,&recover_layout,&rp)&&rp&&recover_layout.moe_batch==16384&&localgrow_stable(&cancel_layout,&recover_layout)&&recover_layout.moe_generation==cancel_layout.moe_generation;
        generation_ok=generation_ok&&recovery_prompt_tokens==6917&&layout_ok;
        fprintf(summary,",\"recovery_api_ok\":%s,\"recovery_stop_reason\":",ok?"true":"false");json_string(summary,stop_name(reason));
        fprintf(summary,",\"recovery_output_tokens\":%u,\"recovery_prompt_tokens\":%u,\"recovery_generation_ok\":%s,\"recovery_state_exported\":%s",recovered.count,recovery_prompt_tokens,generation_ok?"true":"false",exported?"true":"false");
        fputs(",\"recovery_workspace\":",summary);bench_ws_json(summary,ctx);
        status=generation_ok&&exported?0:1;free(recovered.raw_bytes);
    }
    fprintf(summary,",\"status\":\"%s\",\"fullmodel_recovery_within_arm_completed\":%s}\n",status?"failed":"complete",status?"false":"true");cancel_need(fclose(summary)==0,"summary close");
    myds_context_free(ctx);myds_model_free(model);free(warm_prompt);free(cancel_prompt);free(recover_prompt);return status;
}
#endif
