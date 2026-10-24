/* One formal sequence cohort: each sequence owns a fresh context; its setup7k,
 * grow16k and retain7k generations share that context. No testing API, reset,
 * cache/RNG reset or extra GPU wait is inserted. Both normal arms register the
 * same CPU MoE and prompt growth observers; their cost is included in generation wall. Deferred-resource observation is diagnostic-only. Context
 * creation and between-stage state/layout exports are separately reported. */
#if defined(MYDS_MOE_WORKSPACE_TESTING) || defined(MYDS_MOE_WORKSPACE_CAPTURE)
#error "Formal sequence must use the normal production core"
#endif
#define main promptclear_unused_original_fullmodel_main
#include "bench_moe_workspace_fullmodel.c"
#undef main
#include "moe_workspace_sequence_state_localgrow.inc"
#include "moe_workspace_layout_promptclear.inc"
#include "moe_prompt_lifetime_diagnostic.inc"

typedef struct { FILE *file;uint32_t sequence,stage;bool in_generation;int error;uint64_t events; } localgrow_observer_t;
#ifdef MYDS_BENCH_MOE_WS_API
static void localgrow_growth_observer(void *opaque,const myds_v41_moe_growth_event_t *e) {
    localgrow_observer_t *o=opaque;if(!o||!o->file){if(o)o->error=1;return;}
    ws_memory_value_t value=ws_memory_read();if(!value.task_valid||!value.device_valid)o->error=1;
    ++o->events;
    fprintf(o->file,"{\"kind\":\"growth_lifetime\",\"sequence\":%u,\"stage\":%u,\"in_generation\":%s,\"monotonic_s\":%.9f,\"implementation_version\":%u,\"event\":%u,\"capacity\":%u,\"old_batch\":%u,\"new_batch\":%u,\"growth_sequence\":%" PRIu64 ",\"cw_id\":%" PRIu64 ",\"old_arena_id\":%" PRIu64 ",\"new_arena_id\":%" PRIu64 ",\"old_down_id\":%" PRIu64 ",\"new_down_id\":%" PRIu64 ",\"stable_owner_bytes\":%" PRIu64 ",\"old_pool_bytes\":%" PRIu64 ",\"new_pool_bytes\":%" PRIu64 ",\"actual\":",
      o->sequence,o->stage,o->in_generation?"true":"false",now_s(),e->implementation_version,e->event,e->capacity,e->old_batch,e->new_batch,e->sequence,e->cw_id,e->old_arena_id,e->new_arena_id,e->old_down_id,e->new_down_id,e->stable_owner_bytes,e->old_pool_bytes,e->new_pool_bytes);
    ws_memory_json(o->file,value);if(fputs("}\n",o->file)<0||fflush(o->file))o->error=1;
}
#endif

static bool localgrow_directory(char *path,size_t cap,const char *base,const char *name) {
    int n=snprintf(path,cap,"%s/%s",base,name);return n>0&&(size_t)n<cap&&mkdir(path,0700)==0;
}
static bool localgrow_workspace_geometry(const myds_context_t *ctx,uint32_t *C,uint32_t *B) {
#ifdef MYDS_BENCH_MOE_WS_API
    myds_v41_moe_workspace_probe_stats_t q={0};if(!myds_v41_prefill_moe_workspace_stats(ctx,&q))return false;
    *C=q.capacity;*B=q.moe_batch;
#else
    uint64_t q[5]={0};if(!myds_v41_gold_moe_scratch(ctx,q))return false;*C=(uint32_t)q[0];*B=(uint32_t)q[1];
#endif
    return true;
}
int main(int argc,char **argv) {
    const char *model_path=NULL,*requests[3]={0},*out=NULL,*phase=NULL;uint32_t repeats=0;bool execute=false,diagnostic_prompt=false;
    const char *clear_env=getenv("MYDS_PREFILL_PROMPT_VIEW_CLEAR");
    if(clear_env&&strcmp(clear_env,"0")&&strcmp(clear_env,"1")){fputs("MYDS_PREFILL_PROMPT_VIEW_CLEAR must be unset, 0 or 1\n",stderr);return 2;}
    const int clear_mode=clear_env&&!strcmp(clear_env,"1");
    for(int i=1;i<argc;i++) {
        const char *key=argv[i];if(!strcmp(key,"--run")){execute=true;continue;}if(!strcmp(key,"--diagnostic-prompt-observer")){diagnostic_prompt=true;continue;}
        if(++i>=argc){fputs("Missing option value\n",stderr);return 2;}const char *v=argv[i];
        if(!strcmp(key,"--model"))model_path=v;else if(!strcmp(key,"--warm-request"))requests[0]=v;
        else if(!strcmp(key,"--grow-request"))requests[1]=v;else if(!strcmp(key,"--retain-request"))requests[2]=v;
        else if(!strcmp(key,"--out-dir"))out=v;else if(!strcmp(key,"--phase"))phase=v;
        else if(!strcmp(key,"--repeats")){char *end=NULL;unsigned long value=strtoul(v,&end,10);if(!*v||*end||value>3)return 2;repeats=(uint32_t)value;}
        else {fputs("Unknown option\n",stderr);return 2;}
    }
    if(!execute||!model_path||!requests[0]||!requests[1]||!requests[2]||!out||out[0]!='/'||!phase||
       !((!strcmp(phase,"probe")&&repeats==0)||(!strcmp(phase,"correctness")&&repeats==1)||(!strcmp(phase,"performance")&&repeats==3))) {
        fputs("Required: --run --model ABS --warm-request ABS --grow-request ABS --retain-request ABS --out-dir NEW --phase probe|correctness|performance --repeats 0|1|3\n",stderr);return 2;
    }
#ifdef MYDS_BENCH_MOE_WS_API
    const int requested_workspace_mode=myds_metal_prefill_moe_workspace_parse_mode();
#else
    const char *workspace_env=getenv("MYDS_PREFILL_MOE_WORKSPACE");
    const int requested_workspace_mode=(!workspace_env||!strcmp(workspace_env,"0"))?0:(!strcmp(workspace_env,"1")?1:-1);
#endif
    if(requested_workspace_mode<0){fputs("MYDS_PREFILL_MOE_WORKSPACE must be unset, 0 or 1\n",stderr);return 2;}
#ifndef MYDS_BENCH_MOE_WS_API
    if(requested_workspace_mode||clear_mode){fputs("Production gold requires workspace0/clear0\n",stderr);return 2;}
#endif
    if(diagnostic_prompt&&strcmp(phase,"probe")){fputs("Prompt observer is restricted to diagnostic probe mode\n",stderr);return 2;}
    if(clear_mode&&!requested_workspace_mode){fputs("Prompt clearing requires compact workspace mode\n",stderr);return 2;}
    const char *forbidden[]={"MYDS_TRACE_PREFIX","MYDS_BATCH_TRACE_PREFIX","MYDS_PROFILE_PREFILL","MYDS_PROFILE_SPLIT","MYDS_PROFILE_SPLIT2","MYDS_PROFILE_SPLIT3","MYDS_MOE_STEP_PROF","MYDS_KPROF","MYDS_MOE_TILE_PROF"};
    for(unsigned i=0;i<sizeof(forbidden)/sizeof(*forbidden);i++)if(getenv(forbidden[i])){fputs("Trace/profiling forbidden\n",stderr);return 2;}
    if(mkdir(out,0700)){perror("new output directory");return 2;}
    int status=1;FILE *summary=NULL,*seq_json=NULL;myds_model_t *model=NULL;myds_context_t *ctx=NULL;
    ws_memory_sampler_t sampler;bool sampling=false;char *prompts[3]={0};capture_t captured={0};localgrow_observer_t observer={0};promptdiag_state_t pd={0};bool pd_registered=false;FILE *prompt_file=NULL;
    const uint32_t output_tokens=!strcmp(phase,"probe")?8u:32u;
    size_t request_bytes[3]={0};uint64_t request_hashes[3]={0};char path[4096],seqdir[4096],statedir[4096],memdir[4096];
    const char *labels[]={"setup","grow","retain"};const uint32_t expected_prompt_tokens[]={6917,15187,6917};uint32_t completed=0;
#define LG_REQUIRE(v,m) do{if(!(v)){fprintf(stderr,"localgrow sequence: %s\n",m);goto cleanup;}}while(0)
    int n=snprintf(path,sizeof(path),"%s/summary.jsonl",out);LG_REQUIRE(n>0&&(size_t)n<sizeof(path),"summary path");summary=fopen(path,"wbx");LG_REQUIRE(summary,"summary open");
    for(unsigned i=0;i<3;i++){prompts[i]=localgrow_prompt(requests[i],&request_bytes[i],&request_hashes[i]);LG_REQUIRE(prompts[i],"production renderer");}
    model=myds_model_load(model_path);LG_REQUIRE(model&&model->v41,"real ordinary V4.1 model");
    LG_REQUIRE(bench_ws_mode()==requested_workspace_mode,"initialized workspace mode differs from requested environment");
#ifdef MYDS_BENCH_MOE_WS_API
    LG_REQUIRE(myds_metal_prefill_prompt_view_clear_enabled()==clear_mode,"initialized prompt clear mode differs from requested environment");
#endif
    #ifdef MYDS_BENCH_MOE_WS_API
    LG_REQUIRE(promptdiag_register_mode(&pd,diagnostic_prompt),"normal prompt observer register");pd_registered=true;
#endif
#ifdef MYDS_BENCH_MOE_WS_API
    LG_REQUIRE(myds_v41_moe_workspace_growth_observer_set(localgrow_growth_observer,&observer),"idle register normal observer");
    const bool has_observer=true;
#else
    const bool has_observer=false;
#endif
    fputs("{\"kind\":\"metadata\",\"schema_version\":1,\"probe\":\"moe_workspace_promptclear_fullsequence\",\"phase\":",summary);json_string(summary,phase);
    fputs(",\"model\":",summary);json_string(summary,model_path);
    fprintf(summary,",\"workspace_mode\":%d,\"repeats\":%u,\"warmup_sequences\":%u,\"max_new_tokens\":%u,\"max_seq_len\":327680,\"temperature\":0,\"seed\":42,\"fresh_context_per_sequence\":true,\"same_context_within_sequence\":true,\"manual_cache_or_rng_reset\":false,\"extra_gpu_waits_inserted\":0,\"testing_api_used\":false,\"growth_lifetime_observer\":%s,\"observer_cpu_cost_in_generation_wall\":true,\"sampling_interval_ms\":10,\"performance_measured\":%s,\"total_wall_basis\":\"sum_of_three_generation_calls_excluding_between_stage_exports\",\"actual_metallib_path\":",bench_ws_mode(),repeats,repeats?1u:0u,output_tokens,has_observer?"true":"false",!strcmp(phase,"performance")?"true":"false");json_string(summary,myds_metal_loaded_library_path());
    fprintf(summary,",\"implementation_version\":4,\"prompt_view_clear_mode\":%d,\"prompt_lifetime_observer\":%s",clear_mode,pd_registered?"true":"false");
    fprintf(summary,",\"deferred_lifetime_observer\":%s,\"diagnostic_prompt_mode\":%s",diagnostic_prompt?"true":"false",diagnostic_prompt?"true":"false");
    fputs(",\"environment\":{",summary);bool comma=false;
    for(char **entry=environ;*entry;entry++)if(!strncmp(*entry,"MYDS_",5)) {
        char *eq=strchr(*entry,'=');if(!eq)continue;size_t z=(size_t)(eq-*entry);char *key=malloc(z+1);LG_REQUIRE(key,"environment key");memcpy(key,*entry,z);key[z]=0;
        if(comma)fputc(',',summary);comma=true;json_string(summary,key);fputc(':',summary);json_string(summary,eq+1);free(key);
    }
    fputs("},\"requests\":[",summary);
    for(unsigned i=0;i<3;i++){if(i)fputc(',',summary);fputs("{\"stage\":",summary);json_string(summary,labels[i]);fputs(",\"path\":",summary);json_string(summary,requests[i]);fprintf(summary,",\"bytes\":%zu,\"fnv1a64_bytes\":\"%016" PRIx64 "\"}",request_bytes[i],request_hashes[i]);}
    fputs("]}\n",summary);fflush(summary);
    const myds_config_t cfg={.max_seq_len=327680,.max_new_tokens=output_tokens,.temperature=0,.top_p=1,.top_k=0,.min_p=0,.repetition_penalty=1,.seed=42,.stream=false};
    for(uint32_t sequence=0;sequence<=repeats;sequence++) {
        char seqname[32];snprintf(seqname,sizeof(seqname),"seq-%03u",sequence);LG_REQUIRE(localgrow_directory(seqdir,sizeof(seqdir),out,seqname),"sequence directory");
        LG_REQUIRE(localgrow_directory(statedir,sizeof(statedir),seqdir,"state"),"state directory");LG_REQUIRE(localgrow_directory(memdir,sizeof(memdir),seqdir,"memory"),"memory directory");
        n=snprintf(path,sizeof(path),"%s/summary.jsonl",seqdir);LG_REQUIRE(n>0&&(size_t)n<sizeof(path),"sequence log path");seq_json=fopen(path,"wbx");LG_REQUIRE(seq_json,"sequence log");
        n=snprintf(path,sizeof(path),"%s/growth-observer.jsonl",seqdir);LG_REQUIRE(n>0&&(size_t)n<sizeof(path),"observer path");observer.file=fopen(path,"wbx");LG_REQUIRE(observer.file,"new observer file");observer.sequence=sequence;observer.events=0;observer.error=0;
        if(pd_registered){n=snprintf(path,sizeof(path),"%s/%s",seqdir,diagnostic_prompt?"prompt-lifetime.jsonl":"prompt-observer.jsonl");LG_REQUIRE(n>0&&(size_t)n<sizeof(path),"prompt observer path");prompt_file=fopen(path,"wbx");LG_REQUIRE(prompt_file,"new prompt observer file");promptdiag_stream(&pd,prompt_file,sequence);}
        LG_REQUIRE(ws_memory_start(&sampler,memdir),"normal memory sampler");sampling=true;ws_memory_phase(&sampler,0,1);
        ctx=myds_context_create(model,&cfg);LG_REQUIRE(ctx&&ctx->v41&&!ctx->dspark,"fresh sequence context");
        double total_wall=0,sequence_begin=now_s();bool actual_growth=false,actual_retention=false,stable_growth=true;
        for(unsigned stage=0;stage<3;stage++) {
            observer.stage=stage;observer.in_generation=false;
            if(stage)ws_memory_phase(&sampler,stage,1);
            localgrow_layout_t before={0},after={0};bool bp=false,ap=false;char lname[128];
            snprintf(lname,sizeof(lname),"layout-%s-before.json",labels[stage]);n=snprintf(path,sizeof(path),"%s/%s",seqdir,lname);LG_REQUIRE(n>0&&(size_t)n<sizeof(path)&&localgrow_dump_layout(path,ctx,&before,&bp),"before actual layout");
            ws_memory_report(&sampler,seq_json,"request_prepare");
            captured=(capture_t){.bytes_hash=UINT64_C(14695981039346656037),.ids_hash=UINT64_C(14695981039346656037),.dspark=ctx->dspark};captured.raw_bytes=malloc(MAX_OUTPUT_BYTES);LG_REQUIRE(captured.raw_bytes,"raw capture");
            myds_stop_reason_t reason=MYDS_STOP_ERROR;uint32_t prompt_tokens=0;ws_memory_phase(&sampler,stage,2);double begin=now_s();
            if(pd_registered)promptdiag_position(&pd,stage,true);
            observer.in_generation=true;int ok=myds_generate_bytes_ex(ctx,prompts[stage],capture,&captured,&reason,&prompt_tokens,NULL);observer.in_generation=false;if(pd_registered)promptdiag_position(&pd,stage,false);
            double wall=now_s()-begin;total_wall+=wall;ws_memory_report(&sampler,seq_json,"generation");ws_memory_phase(&sampler,stage,3);
            snprintf(lname,sizeof(lname),"layout-%s-after.json",labels[stage]);n=snprintf(path,sizeof(path),"%s/%s",seqdir,lname);LG_REQUIRE(n>0&&(size_t)n<sizeof(path)&&localgrow_dump_layout(path,ctx,&after,&ap),"after actual layout");
            uint32_t C=0,B=0;bool geometry=localgrow_workspace_geometry(ctx,&C,&B);myds_prefill_stats_t pf={0};myds_last_prefill_stats(ctx,&pf);
            bool good=ok&&reason==MYDS_STOP_LENGTH&&captured.count==output_tokens&&!captured.overflow&&prompt_tokens==expected_prompt_tokens[stage]&&geometry&&C==2048&&B==(stage?16384u:8192u)&&!observer.error&&(!pd_registered||!pd.error);
            bool exported=good&&localgrow_state(statedir,labels[stage],ctx,&captured,true,myds_v41_snapshot_get_next_position(ctx->v41));
            if(stage==1)actual_growth=geometry&&B==16384;
            if(stage==2)actual_retention=geometry&&B==16384;
#ifdef MYDS_BENCH_MOE_WS_API
            good=good&&ap&&after.implementation_version==4;
            if(stage==1&&bench_ws_mode())stable_growth=bp&&localgrow_stable(&before,&after)&&after.moe_generation==before.moe_generation+1&&after.localgrow_successes==before.localgrow_successes+1;
            if(stage==2)stable_growth=stable_growth&&bp&&localgrow_stable(&before,&after)&&after.moe_generation==before.moe_generation;
#endif
            fprintf(seq_json,"{\"kind\":\"sample\",\"sequence\":%u,\"warmup\":%s,\"run\":%u,\"stage\":",sequence,sequence==0&&repeats?"true":"false",stage);json_string(seq_json,labels[stage]);
            fprintf(seq_json,",\"role\":\"%s\",\"same_context\":true,\"generation_ok\":%s,\"raw_state_exported\":%s,\"prompt_tokens\":%u,\"cached_tokens\":%u,\"prefill_tokens\":%u,\"output_tokens\":%u,\"stop_reason\":",stage?"primary":"setup",good?"true":"false",exported?"true":"false",prompt_tokens,pf.cached_tokens,pf.prefill_tokens,captured.count);json_string(seq_json,stop_name(reason));
            fprintf(seq_json,",\"wall_s\":%.9f,\"prefill_s\":%.9f,\"first_callback_s\":%.9f,\"last_callback_s\":%.9f,\"decode_s\":%.9f,\"decode_intervals\":%u,\"output_bytes\":%" PRIu64 ",\"output_fnv1a64_bytes\":\"%016" PRIx64 "\",\"tokens_fnv1a64_bytes\":\"%016" PRIx64 "\",\"workspace\":",wall,pf.prefill_seconds,captured.count?captured.first-begin:-1,captured.count?captured.last-begin:-1,captured.count>1?captured.last-captured.first:0,captured.count?captured.count-1:0,captured.byte_count,captured.bytes_hash,captured.ids_hash);bench_ws_json(seq_json,ctx);
            fputs(",\"callback_intervals_ms\":[",seq_json);for(uint32_t i=1;i<captured.count&&i<MAX_OUTPUT_TOKENS;i++)fprintf(seq_json,"%s%.9f",i>1?",":"",1000*(captured.times[i]-captured.times[i-1]));fputs("]}\n",seq_json);fflush(seq_json);
            free(captured.raw_bytes);captured.raw_bytes=NULL;ws_memory_report(&sampler,seq_json,"raw_export");ws_memory_phase(&sampler,stage,0);
            LG_REQUIRE(good&&exported&&stable_growth,"complete output/geometry/local growth witness");
        }
        double elapsed=now_s()-sequence_begin;int stopped=ws_memory_stop(&sampler);sampling=false;LG_REQUIRE(stopped,"sampler stop");
        FILE *closing=seq_json;seq_json=NULL;LG_REQUIRE(fclose(closing)==0,"sequence log close");
        fprintf(summary,"{\"kind\":\"sequence_complete\",\"sequence\":%u,\"warmup\":%s,\"fresh_context\":true,\"same_context_within_sequence\":true,\"generations\":3,\"generation_outputs\":%u,\"total_wall_s\":%.9f,\"elapsed_including_exports_s_diagnostic_only\":%.9f,\"actual_B8192_to16384_growth\":%s,\"actual_B16384_retained\":%s,\"native_localgrow_identity_gate\":%s,\"growth_observer_events\":%" PRIu64 ",\"sampling_coverage_proven\":false}\n",sequence,sequence==0&&repeats?"true":"false",output_tokens,total_wall,elapsed,actual_growth?"true":"false",actual_retention?"true":"false",stable_growth?"true":"false",observer.events);fflush(summary);
        myds_context_free(ctx);ctx=NULL;if(pd_registered){promptdiag_stream(&pd,NULL,sequence);closing=prompt_file;prompt_file=NULL;LG_REQUIRE(fclose(closing)==0,"prompt observer close");}closing=observer.file;observer.file=NULL;LG_REQUIRE(fclose(closing)==0,"observer close");completed++;
    }
    fprintf(summary,"{\"kind\":\"complete\",\"sequences\":%u,\"generations\":%u,\"max_new_tokens\":%u,\"full_original_raw_exported\":true,\"adoption_gate_pass\":false,\"independent_review_required\":true}\n",completed,completed*3,output_tokens);status=0;
cleanup:
    if(sampling&&!ws_memory_stop(&sampler))status=1;if(seq_json&&fclose(seq_json))status=1;free(captured.raw_bytes);
    if(ctx)myds_context_free(ctx);
#ifdef MYDS_BENCH_MOE_WS_API
    if(!myds_v41_moe_workspace_growth_observer_set(NULL,NULL))status=1;
#endif
    if(pd_registered&&!promptdiag_unregister(&pd))status=1;if(prompt_file&&fclose(prompt_file))status=1;
    if(observer.file&&fclose(observer.file))status=1;if(model)myds_model_free(model);for(unsigned i=0;i<3;i++)free(prompts[i]);if(summary&&fclose(summary))status=1;
#undef LG_REQUIRE
    return status;
}
