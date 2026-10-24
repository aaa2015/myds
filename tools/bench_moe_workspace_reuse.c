/* Supplemental same-context success reuse diagnostic; normal API only.
 * One process/model/context: warm pf7k -> grow pf16k -> retain pf7k.
 * Existing cache/PLD settings remain active; fixed8 outputs/temp0/seed42.
 * No fault injection, cancellation checkpoint, extra GPU wait or factory
 * observer. Wall times are diagnostic only, never a latency adoption result. */
#if defined(MYDS_MOE_WORKSPACE_TESTING) || defined(MYDS_MOE_WORKSPACE_CAPTURE)
#error "Reuse diagnostic must link the normal production core, without TESTING or CAPTURE"
#endif
#define MYDS_BENCH_MOE_WS_API 1
#define main moe_workspace_reuse_unused_fullmodel_main
#include "bench_moe_workspace_fullmodel.c"
#undef main

static char *reuse_prompt(const char *path,size_t *bytes,uint64_t *hash) {
    char *body=read_request(path,bytes);if(!body||*bytes>(SIZE_MAX-8192u)/4u){free(body);return NULL;}
    *hash=hash_bytes(UINT64_C(14695981039346656037),body,*bytes);
    size_t capacity=*bytes*4u+8192u;if(capacity<65536u)capacity=65536u;
    char *prompt=malloc(capacity);
    if(!prompt||!parse_chat_prompt(body,prompt,capacity)){free(prompt);prompt=NULL;}
    free(body);return prompt;
}

static bool reuse_state(const char *root,const char *label,myds_context_t *ctx,
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
    fprintf(out,",\"scope\":\"same_context_moe_workspace_reuse\",\"complete_live_state\":%s,\"ctx_pos\":%u,\"next_position\":%u,\"pcache_len\":%u,\"anchor_len\":%u,\"anchor_slot\":%u,\"requested_prefill_upper_bound\":%u,\"rng_state\":\"%016" PRIx64 "\",\"engram_position\":%u,\"limitations\":[\"Complete successful live state is exported. Published snapshots use their own positions; the persistent anchor can outlive a shorter current request.\",\"Persistent disk-cache payload and allocator padding are not exported.\"],\"scalar_order\":[\"ctx_pos\",\"rng\",\"pcache_len\",\"anchor_len\",\"next\",\"engram_position\",\"v41_valid\",\"max_seq_len\",\"max_new_tokens\",\"seed\",\"continue_from_cache\"],\"files\":[",
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

int main(int argc,char **argv) {
    const char *model_path=NULL,*requests[3]={NULL,NULL,NULL},*out=NULL;bool execute=false;
    for(int i=1;i<argc;i++) {
        const char *key=argv[i];if(!strcmp(key,"--run")){execute=true;continue;}
        if(++i>=argc){fputs("Missing CLI value\n",stderr);return 2;}const char *value=argv[i];
        if(!strcmp(key,"--model"))model_path=value;else if(!strcmp(key,"--warm-request"))requests[0]=value;
        else if(!strcmp(key,"--grow-request"))requests[1]=value;else if(!strcmp(key,"--retain-request"))requests[2]=value;
        else if(!strcmp(key,"--out-dir"))out=value;else {fputs("Unknown CLI option\n",stderr);return 2;}
    }
    if(!execute||!model_path||!requests[0]||!requests[1]||!requests[2]||!out||out[0]!='/') {
        fputs("Usage: bench_moe_workspace_reuse --run --model ABS --warm-request ABS --grow-request ABS --retain-request ABS --out-dir NEW_ABSOLUTE\n",stderr);return 2;
    }
    if(myds_metal_prefill_moe_workspace_parse_mode()<0)return 2;
    const char *forbidden[]={"MYDS_TRACE_PREFIX","MYDS_BATCH_TRACE_PREFIX","MYDS_PROFILE_PREFILL","MYDS_PROFILE_SPLIT","MYDS_PROFILE_SPLIT2","MYDS_PROFILE_SPLIT3","MYDS_MOE_STEP_PROF","MYDS_KPROF","MYDS_MOE_TILE_PROF"};
    for(unsigned i=0;i<sizeof(forbidden)/sizeof(*forbidden);i++)if(getenv(forbidden[i])){fputs("Trace/profiling is forbidden\n",stderr);return 2;}
    if(mkdir(out,0700)){perror("new output directory");return 2;}
    int status=1;FILE *json=NULL;myds_model_t *model=NULL;myds_context_t *ctx=NULL;
    ws_memory_sampler_t sampler;bool sampling=false;char *prompts[3]={NULL,NULL,NULL};
    size_t request_bytes[3]={0};uint64_t request_hash[3]={0};capture_t captured={0};char *before_json=NULL;
    char path[4096],state_dir[4096],memory_dir[4096];const char *labels[]={"warm","grow","retain"};
#define REQUIRE(value,message) do{if(!(value)){fprintf(stderr,"reuse: %s\n",message);goto cleanup;}}while(0)
    int n=snprintf(path,sizeof(path),"%s/summary.jsonl",out);REQUIRE(n>0&&(size_t)n<sizeof(path),"summary path");
    json=fopen(path,"wbx");REQUIRE(json,"new summary file");
    n=snprintf(state_dir,sizeof(state_dir),"%s/state",out);REQUIRE(n>0&&(size_t)n<sizeof(state_dir)&&mkdir(state_dir,0700)==0,"state directory");
    n=snprintf(memory_dir,sizeof(memory_dir),"%s/memory",out);REQUIRE(n>0&&(size_t)n<sizeof(memory_dir)&&mkdir(memory_dir,0700)==0,"memory directory");
    for(unsigned i=0;i<3;i++){prompts[i]=reuse_prompt(requests[i],&request_bytes[i],&request_hash[i]);REQUIRE(prompts[i],"original request prompt renderer");}
    model=myds_model_load(model_path);REQUIRE(model&&model->v41,"actual V4.1 model");
    const myds_config_t cfg={.max_seq_len=327680,.max_new_tokens=8,.temperature=0,.top_p=1,.top_k=0,.min_p=0,.repetition_penalty=1,.seed=42,.stream=false};
    REQUIRE(ws_memory_start(&sampler,memory_dir),"memory sampler start");sampling=true;
    ws_memory_phase(&sampler,0,1);ctx=myds_context_create(model,&cfg);REQUIRE(ctx&&ctx->v41&&!ctx->dspark,"one normal context");
    fputs("{\"kind\":\"metadata\",\"schema_version\":1,\"probe\":\"same_context_moe_workspace_success_reuse\",\"model\":",json);json_string(json,model_path);
    fprintf(json,",\"workspace_mode\":%d,\"max_seq_len\":327680,\"max_new_tokens\":8,\"temperature\":0,\"seed\":42,\"same_context\":true,\"fresh_context_per_sample\":false,\"manual_cache_or_rng_reset\":false,\"extra_gpu_waits_inserted\":0,\"testing_api_used\":false,\"factory_lifetime_observer\":false,\"sampling_interval_ms\":10,\"performance_measured\":false,\"wall_time_scope\":\"diagnostic_only\",\"sampling_coverage_requires_independent_decision\":true,\"actual_metallib_path\":",bench_ws_mode());
    json_string(json,myds_metal_loaded_library_path());fputs(",\"environment\":{",json);bool comma=false;
    for(char **entry=environ;*entry;entry++)if(!strncmp(*entry,"MYDS_",5)) {
        char *eq=strchr(*entry,'=');if(!eq)continue;size_t size=(size_t)(eq-*entry);char *key=malloc(size+1);REQUIRE(key,"environment key");memcpy(key,*entry,size);key[size]=0;
        if(comma)fputc(',',json);comma=true;json_string(json,key);fputc(':',json);json_string(json,eq+1);free(key);
    }
    fputs("},\"requests\":[",json);
    for(unsigned i=0;i<3;i++){if(i)fputc(',',json);fputs("{\"step\":",json);json_string(json,labels[i]);fputs(",\"path\":",json);json_string(json,requests[i]);fprintf(json,",\"bytes\":%zu,\"fnv1a64_bytes\":\"%016" PRIx64 "\"}",request_bytes[i],request_hash[i]);}
    fputs("]}\n",json);fflush(json);
    myds_v41_moe_workspace_probe_stats_t prior={0},after={0};bool growth=false,retained=false;
    for(unsigned step=0;step<3;step++) {
        if(step)ws_memory_phase(&sampler,step,1);
        bool before_present=myds_v41_prefill_moe_workspace_stats(ctx,&prior)!=0;
        size_t before_bytes=0;FILE *before=open_memstream(&before_json,&before_bytes);REQUIRE(before,"before-workspace serialization");
        bench_ws_json(before,ctx);REQUIRE(fclose(before)==0,"before-workspace close");
        ws_memory_report(&sampler,json,"request_prepare");
        captured=(capture_t){.bytes_hash=UINT64_C(14695981039346656037),.ids_hash=UINT64_C(14695981039346656037),.dspark=ctx->dspark};
        captured.raw_bytes=malloc(MAX_OUTPUT_BYTES);REQUIRE(captured.raw_bytes,"raw byte capture");
        myds_stop_reason_t reason=MYDS_STOP_ERROR;uint32_t prompt_tokens=0;
        ws_memory_phase(&sampler,step,2);double begin=now_s();
        int ok=myds_generate_bytes_ex(ctx,prompts[step],capture,&captured,&reason,&prompt_tokens,NULL);
        double wall=now_s()-begin;ws_memory_report(&sampler,json,"generation");ws_memory_phase(&sampler,step,3);
        bool after_present=myds_v41_prefill_moe_workspace_stats(ctx,&after)!=0;
        myds_prefill_stats_t pf={0};myds_last_prefill_stats(ctx,&pf);
        bool complete=ok&&reason==MYDS_STOP_LENGTH&&captured.count==8&&!captured.overflow&&after_present;
        bool exported=complete&&reuse_state(state_dir,labels[step],ctx,&captured,true,myds_v41_snapshot_get_next_position(ctx->v41));
        if(step==1)growth=before_present&&after_present&&prior.capacity==after.capacity&&prior.capacity==2048&&prior.moe_batch==8192&&after.moe_batch==16384;
        if(step==2)retained=before_present&&after_present&&prior.capacity==after.capacity&&prior.moe_batch==after.moe_batch&&after.moe_batch==16384;
        fprintf(json,"{\"kind\":\"sample\",\"run\":%u,\"step\":",step);json_string(json,labels[step]);
        fprintf(json,",\"same_context\":true,\"fresh_context\":%s,\"generation_ok\":%s,\"raw_state_exported\":%s,\"prompt_tokens\":%u,\"cached_tokens\":%u,\"prefill_tokens\":%u,\"output_tokens\":%u,\"stop_reason\":",step==0?"true":"false",complete?"true":"false",exported?"true":"false",prompt_tokens,pf.cached_tokens,pf.prefill_tokens,captured.count);json_string(json,stop_name(reason));
        fprintf(json,",\"wall_s_diagnostic_only\":%.9f,\"prefill_s_diagnostic_only\":%.9f,\"output_bytes\":%" PRIu64 ",\"output_fnv1a64_bytes\":\"%016" PRIx64 "\",\"tokens_fnv1a64_bytes\":\"%016" PRIx64 "\",\"before_workspace\":%s,\"after_workspace\":",wall,pf.prefill_seconds,captured.byte_count,captured.bytes_hash,captured.ids_hash,before_json);bench_ws_json(json,ctx);
        fputs("}\n",json);fflush(json);free(before_json);before_json=NULL;free(captured.raw_bytes);captured.raw_bytes=NULL;
        ws_memory_report(&sampler,json,"raw_export");ws_memory_phase(&sampler,step,0);
        REQUIRE(complete&&exported,"successful complete original raw state required at all three steps");
    }
    fprintf(json,"{\"kind\":\"complete\",\"three_generations_completed\":true,\"actual_C2048_B8192_to16384_grow\":%s,\"actual_B16384_retained_after_short\":%s,\"performance_measured\":false,\"sampling_coverage_proven\":false,\"adoption_gate_pass\":false,\"requires_independent_peak_coverage_and_raw_review\":true}\n",growth?"true":"false",retained?"true":"false");
    REQUIRE(growth&&retained,"actual requested grow/retain geometry was not exercised");status=0;
cleanup:
    if(sampling&&!ws_memory_stop(&sampler))status=1;
    free(before_json);free(captured.raw_bytes);if(ctx)myds_context_free(ctx);if(model)myds_model_free(model);
    for(unsigned i=0;i<3;i++)free(prompts[i]);if(json&&fclose(json))status=1;
#undef REQUIRE
    return status;
}
