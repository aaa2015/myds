/* Real-context fault -> same-context recovery. Never execute without an
 * exclusive authorized window. Includes only the cancellation tool's helpers;
 * neither its main nor the inherited fullmodel main is invoked. */
#define MYDS_MOE_WS_CANCEL_HELPERS_ONLY 1
#include "test_moe_workspace_cancel_localgrow.c"

static capture_t recovery_capture(void) {
    capture_t c={.bytes_hash=UINT64_C(14695981039346656037),.ids_hash=UINT64_C(14695981039346656037)};
    c.raw_bytes=malloc(MAX_OUTPUT_BYTES);cancel_need(c.raw_bytes!=NULL,"raw capture allocation");return c;
}
static void recovery_trace(FILE *f,const myds_metal_moe_workspace_allocation_t *trace,uint64_t n) {
    fputc('[',f);
    for(uint64_t i=0;i<n;i++) {
        if(i)fputc(',',f);fprintf(f,"{\"ordinal\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"kind\":%u,\"index\":%d,\"success\":%s,\"site\":",trace[i].ordinal,trace[i].bytes,trace[i].kind,trace[i].index,trace[i].success?"true":"false");
        json_string(f,trace[i].site);fputc('}',f);
    }
    fputc(']',f);
}
int main(int argc,char **argv) {
    const char *model_path=NULL,*warm_request=NULL,*failure_request=NULL,*recover_request=NULL,*out=NULL,*scenario=NULL;
    uint32_t count=0;float temperature=-1;bool run=false;
    for(int i=1;i<argc;i++) {
        const char *key=argv[i];if(!strcmp(key,"--run")){run=true;continue;}
        cancel_need(++i<argc,"option value");const char *v=argv[i];
        if(!strcmp(key,"--model"))model_path=v;else if(!strcmp(key,"--warm-request"))warm_request=v;
        else if(!strcmp(key,"--failure-request"))failure_request=v;else if(!strcmp(key,"--recover-request"))recover_request=v;
        else if(!strcmp(key,"--out-dir"))out=v;else if(!strcmp(key,"--scenario"))scenario=v;
        else if(!strcmp(key,"--recover-tokens"))count=positive(v,32,"recovery tokens");
        else if(!strcmp(key,"--temperature")){if(!strcmp(v,"0"))temperature=0;else if(!strcmp(v,"0.7"))temperature=.7f;else cancel_need(false,"temperature");}
        else cancel_need(false,"unknown option");
    }
    cancel_need(run&&model_path&&warm_request&&failure_request&&recover_request&&out&&out[0]=='/'&&
                scenario&&(!strcmp(scenario,"allocation")||!strcmp(scenario,"allocation-owner")||!strcmp(scenario,"shared-pso"))&&temperature>=0&&(count==8||count==32),
                "explicit --run, model, warm/failure/recover requests, new absolute out, scenario, temperature, recover-tokens required");
    cancel_need(mkdir(out,0700)==0,"new output directory");char summary_path[4096];int n=snprintf(summary_path,sizeof(summary_path),"%s/summary.json",out);
    cancel_need(n>0&&(size_t)n<sizeof(summary_path),"summary path");FILE *f=fopen(summary_path,"wbx");cancel_need(f!=NULL,"new summary");
    const char *forbidden[]={"MYDS_TRACE_PREFIX","MYDS_BATCH_TRACE_PREFIX","MYDS_PROFILE_PREFILL","MYDS_PROFILE_SPLIT","MYDS_PROFILE_SPLIT2","MYDS_PROFILE_SPLIT3","MYDS_MOE_STEP_PROF","MYDS_KPROF","MYDS_MOE_TILE_PROF"};
    for(unsigned i=0;i<sizeof(forbidden)/sizeof(*forbidden);i++)cancel_need(!getenv(forbidden[i]),"trace/profiling forbidden");
    const char *requests[]={warm_request,failure_request,recover_request};char *prompts[3]={0};size_t request_bytes[3]={0};uint64_t request_hashes[3]={0};
    for(unsigned i=0;i<3;i++)prompts[i]=render(requests[i],&request_bytes[i],&request_hashes[i]);
    myds_model_t *model=myds_model_load(model_path);cancel_need(model&&model->v41,"real V4.1 model");
    myds_config_t cfg={.max_seq_len=327680,.max_new_tokens=count,.temperature=temperature,.top_p=1,.top_k=0,.min_p=0,.repetition_penalty=1,.seed=42,.stream=false};
    myds_context_t *ctx=myds_context_create(model,&cfg);cancel_need(ctx&&ctx->v41&&!ctx->dspark,"one real ordinary context");
    cancel_need(myds_metal_moe_workspace_test_control(0,0),"idle clear fault before warm");
    capture_t warm=recovery_capture();uint32_t warm_tokens=0;myds_stop_reason_t reason=MYDS_STOP_ERROR;
    int ok=myds_generate_bytes_ex(ctx,prompts[0],capture,&warm,&reason,&warm_tokens,NULL);
    bool warm_ok=ok&&reason==MYDS_STOP_LENGTH&&warm.count==count&&!warm.overflow&&warm_tokens==6917;
    bool warm_state=cancel_state(out,"before-failure",ctx,&warm,true,myds_v41_snapshot_get_next_position(ctx->v41));
    localgrow_layout_t warm_layout={0},failed_layout={0},recovered_layout={0};bool wp=false,fp=false,rp=false;
    char layout_path[4096];n=snprintf(layout_path,sizeof(layout_path),"%s/layout-before-failure.json",out);
    cancel_need(n>0&&(size_t)n<sizeof(layout_path)&&localgrow_dump_layout(layout_path,ctx,&warm_layout,&wp)&&wp&&warm_layout.moe_batch==8192&&warm_layout.capacity==2048,"complete warm actual layout");
    myds_v41_moe_workspace_probe_stats_t old={0};bool old_present=myds_v41_prefill_moe_workspace_stats(ctx,&old)!=0;
    fprintf(f,"{\"schema_version\":1,\"probe\":\"same_context_moe_workspace_failure_recovery_localgrow\",\"scenario\":");json_string(f,scenario);
    fprintf(f,",\"workspace_mode\":%d,\"temperature\":%.9g,\"seed\":42,\"recovery_tokens\":%u,\"vocab_size\":%u,\"snapshot_slots\":%u,\"same_context\":true,\"rng_manually_reset\":false,\"extra_gpu_waits_inserted\":0,\"performance_measured\":false,\"real_oom\":false,\"warm_generation_ok\":%s,\"warm_state_exported\":%s,\"warm_prompt_tokens\":%u,\"warm_output_tokens\":%u,\"warm_workspace\":",bench_ws_mode(),temperature,count,model->vocab_size,(unsigned)MYDS_SNAP_SLOTS,warm_ok?"true":"false",warm_state?"true":"false",warm_tokens,warm.count);
    bench_ws_json(f,ctx);fputs(",\"request_inputs\":{",f);const char *names[]={"warm","failure","recover"};
    for(unsigned i=0;i<3;i++)fprintf(f,"%s\"%s\":{\"bytes\":%zu,\"fnv1a64_bytes\":\"%016" PRIx64 "\"}",i?",":"",names[i],request_bytes[i],request_hashes[i]);fputc('}',f);fflush(f);free(warm.raw_bytes);
    cancel_need(warm_ok&&warm_state&&old_present&&old.capacity>0&&old.moe_batch>0,"initial real successful workspace and complete state");
    bool allocation=!strcmp(scenario,"allocation")||!strcmp(scenario,"allocation-owner");uint64_t ordinal=!strcmp(scenario,"allocation-owner")?3u:(allocation?1u:0u);uint32_t mask=allocation?0u:8u;
    cancel_need(myds_v41_moe_workspace_cancel_control(0,NULL),"idle reset progress only");
    cancel_need(myds_metal_moe_workspace_test_control(ordinal,mask),"idle arm fault before actual prefill entry");
    int selected=myds_metal_moe_workspace_provider(32);
    capture_t failed=recovery_capture();uint32_t failed_tokens=0;reason=MYDS_STOP_ERROR;
    ok=myds_generate_bytes_ex(ctx,prompts[1],capture,&failed,&reason,&failed_tokens,NULL);
    uint64_t progress=myds_v41_moe_workspace_progress();
    uint64_t trace_n=myds_metal_moe_workspace_allocation_trace(NULL,0);cancel_need(trace_n<131072u,"bounded actual allocation trace");
    myds_metal_moe_workspace_allocation_t *trace=calloc(trace_n?trace_n:1,sizeof(*trace));cancel_need(trace!=NULL,"trace copy");
    cancel_need(myds_metal_moe_workspace_allocation_trace(trace,trace_n)==trace_n,"complete actual trace");
    unsigned injected=0;for(uint64_t i=0;i<trace_n;i++)if(!trace[i].success)injected++;
    bool fault_proven=selected==MYDS_MOE_WS_NONE&&progress==0;
    if(allocation) {
        fault_proven=injected==1&&trace_n==ordinal&&trace[ordinal-1].ordinal==ordinal&&!trace[ordinal-1].success&&trace[ordinal-1].bytes>0;
        if(ordinal==3u)fault_proven=fault_proven&&trace[ordinal-1].kind==1;
        else fault_proven=fault_proven&&trace[0].kind==(bench_ws_mode()?3u:1u);
    }
    bool failure_ok=!ok&&reason==MYDS_STOP_ERROR&&failed.count==0&&!failed.overflow&&ctx->pcache_len==0&&failed_tokens==15187&&failed_tokens>warm_tokens&&fault_proven;
    bool failed_state=cancel_state(out,"after-failure",ctx,&failed,false,failed_tokens?failed_tokens-1u:0);
    n=snprintf(layout_path,sizeof(layout_path),"%s/layout-after-failure.json",out);
    cancel_need(n>0&&(size_t)n<sizeof(layout_path)&&localgrow_dump_layout(layout_path,ctx,&failed_layout,&fp),"actual failed layout");
    bool retained=!bench_ws_mode()||(fp&&localgrow_stable(&warm_layout,&failed_layout)&&failed_layout.moe_batch==8192&&failed_layout.moe_generation==warm_layout.moe_generation&&failed_layout.down_id==warm_layout.down_id&&failed_layout.arena_id==warm_layout.arena_id&&memcmp(failed_layout.acc_ids,warm_layout.acc_ids,sizeof(warm_layout.acc_ids))==0&&failed_layout.localgrow_attempts==warm_layout.localgrow_attempts+1&&failed_layout.localgrow_failures==warm_layout.localgrow_failures+1);
    failure_ok=failure_ok&&retained;
    myds_v41_moe_workspace_probe_stats_t after={0};bool after_present=myds_v41_prefill_moe_workspace_stats(ctx,&after)!=0;
    fprintf(f,",\"failure_api_ok\":%s,\"failure_stop_reason\":",ok?"true":"false");json_string(f,stop_name(reason));
    fprintf(f,",\"failure_prompt_tokens\":%u,\"failure_output_tokens\":%u,\"failure_gate_pass\":%s,\"failure_state_exported\":%s,\"fault_ordinal\":%" PRIu64 ",\"mm_pso_mask\":%u,\"selected_provider\":%d,\"failed_moe_progress\":%" PRIu64 ",\"injected_nulls\":%u,\"allocation_trace\":",failed_tokens,failed.count,failure_ok?"true":"false",failed_state?"true":"false",ordinal,mask,selected,progress,injected);
    recovery_trace(f,trace,trace_n);free(trace);fputs(",\"failed_workspace\":",f);bench_ws_json(f,ctx);
    fprintf(f,",\"old_workspace_present_before_failure\":true,\"workspace_present_after_failure\":%s,\"old_capacity\":%u,\"old_moe_batch\":%u,\"old_generation\":%" PRIu64 ",\"after_capacity\":%u,\"after_moe_batch\":%u,\"after_generation\":%" PRIu64,after_present?"true":"false",old.capacity,old.moe_batch,old.generation,after.capacity,after.moe_batch,after.generation);fflush(f);free(failed.raw_bytes);
    /* No hidden drain: this rejects if generation's existing failure path did
     * not leave the diagnostic controls idle. It never repairs that condition. */
    bool clear=myds_metal_moe_workspace_test_control(0,0)!=0;fprintf(f,",\"fault_cleared_while_idle\":%s",clear?"true":"false");
    int status=1;
    if(failure_ok&&failed_state&&clear) {
        capture_t recovered=recovery_capture();uint32_t tokens=0;reason=MYDS_STOP_ERROR;
        ok=myds_generate_bytes_ex(ctx,prompts[2],capture,&recovered,&reason,&tokens,NULL);
        bool valid=ok&&reason==MYDS_STOP_LENGTH&&recovered.count==count&&!recovered.overflow&&tokens==6917;
        bool exported=cancel_state(out,"recovery",ctx,&recovered,true,myds_v41_snapshot_get_next_position(ctx->v41));
        n=snprintf(layout_path,sizeof(layout_path),"%s/layout-recovery.json",out);
        bool layout_ok=n>0&&(size_t)n<sizeof(layout_path)&&localgrow_dump_layout(layout_path,ctx,&recovered_layout,&rp)&&rp&&recovered_layout.moe_batch==8192;
        if(bench_ws_mode())layout_ok=layout_ok&&localgrow_stable(&warm_layout,&recovered_layout)&&recovered_layout.moe_generation==warm_layout.moe_generation;
        valid=valid&&layout_ok;
        fprintf(f,",\"recovery_api_ok\":%s,\"recovery_stop_reason\":",ok?"true":"false");json_string(f,stop_name(reason));
        fprintf(f,",\"recovery_generation_ok\":%s,\"recovery_state_exported\":%s,\"recovery_prompt_tokens\":%u,\"recovery_output_tokens\":%u,\"recovery_workspace\":",valid?"true":"false",exported?"true":"false",tokens,recovered.count);bench_ws_json(f,ctx);
        status=valid&&exported?0:1;free(recovered.raw_bytes);
    }
    fprintf(f,",\"status\":\"%s\",\"fullmodel_recovery_within_arm_completed\":%s,\"ownership_contract\":\"Compact failed local growth must preserve old CW/B/pools and stable bindings; legacy control actual retention or release is recorded.\"}\n",status?"failed":"complete",status?"false":"true");
    cancel_need(fclose(f)==0,"summary close");myds_context_free(ctx);myds_model_free(model);for(unsigned i=0;i<3;i++)free(prompts[i]);return status;
}
