/* Additional actual local-growth transaction tests; original factory/C-growth
 * suite is included unchanged and remains a separately identified case group.
 * --run allocates real Metal objects; no model inference or GPU math is timed. */
#define main moews_localgrow_original_allocation_main
#include "test_moe_workspace_allocations.c"
#undef main
#include "moe_workspace_probe_layout_promptclear.inc"
#include <mach/mach.h>
#include <time.h>

#ifndef MYDS_MOE_WS_SELECTION_ONLY
static uint32_t lg_old_B,lg_new_B;static uint64_t lg_ordinal;static const char *lg_phase;
static void lg_checkpoint(void *unused,const myds_v41_moe_growth_event_t *e) {
    (void)unused;myds_metal_residency_stats_t m={0};task_vm_info_data_t v={0};mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    require(myds_metal_residency_diagnostics(&m)&&task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&v,&n)==KERN_SUCCESS,"actual localgrow resource/task checkpoint");
    struct timespec t;require(!clock_gettime(CLOCK_MONOTONIC,&t),"checkpoint clock");
    fprintf(records,"{\"kind\":\"localgrow_checkpoint\",\"phase\":");text(records,lg_phase);
    fprintf(records,",\"old_B\":%u,\"new_B\":%u,\"ordinal\":%" PRIu64 ",\"event\":%u,\"sequence\":%" PRIu64 ",\"cw_id\":%" PRIu64 ",\"monotonic_s\":%.9f,\"metal_current_allocated_size\":%" PRIu64 ",\"process_phys_footprint\":%" PRIu64 ",\"process_resident_size\":%" PRIu64 ",\"old_pool_bytes\":%" PRIu64 ",\"new_pool_bytes\":%" PRIu64 ",\"cpu_only\":true,\"gpu_waits\":0}\n",lg_old_B,lg_new_B,lg_ordinal,e->event,e->sequence,e->cw_id,t.tv_sec+t.tv_nsec*1e-9,m.device_allocated_size,v.phys_footprint,v.resident_size,e->old_pool_bytes,e->new_pool_bytes);fflush(records);
}
static void lg_layout(myds_v41_moe_workspace_probe_t *w,const char *phase) {
    fputs("{\"kind\":\"localgrow_layout\",\"phase\":",records);text(records,phase);fputs(",\"layout\":",records);
    require(lg_snapshot_json(records,w),"complete actual layout enumeration");fputs("}\n",records);fflush(records);
}
static uint64_t lg_pattern(myds_v41_moe_workspace_probe_t *w,int initialize) {
    myds_v41_moe_workspace_layout_t q={0};require(myds_v41_moe_workspace_probe_layout(w,&q),"pattern layout");uint64_t total=0;
    for(uint64_t i=0;i<q.binding_count;i++) {
        myds_v41_moe_workspace_binding_t b={0};require(myds_v41_moe_workspace_probe_binding(w,i,&b),"pattern binding");
        if(b.scope!=1||!b.identity.bytes)continue;
        unsigned char *p=myds_v41_moe_workspace_probe_binding_data(w,i);require(p!=NULL,"actual stable owner CPU data");
        for(uint64_t j=0;j<b.identity.bytes;j++) {
            unsigned char expected=(unsigned char)(0x63u+i*17u+j*29u);
            if(initialize)p[j]=expected;else require(p[j]==expected,"complete stable owner bytes survive local transaction");
        }total+=b.identity.bytes;
    }require(total==q.stable_owner_bytes,"all stable owner bytes patterned exactly once");return total;
}
static myds_v41_moe_workspace_binding_t *lg_bindings(myds_v41_moe_workspace_probe_t *w,myds_v41_moe_workspace_layout_t *q) {
    require(myds_v41_moe_workspace_probe_layout(w,q)&&q->implementation_version==4&&q->binding_count<100000,"v4 actual layout");
    myds_v41_moe_workspace_binding_t *b=calloc(q->binding_count,sizeof(*b));require(b!=NULL,"CPU binding snapshot");
    for(uint64_t i=0;i<q->binding_count;i++)require(myds_v41_moe_workspace_probe_binding(w,i,&b[i]),"complete actual binding snapshot");return b;
}
static void lg_compare(myds_v41_moe_workspace_probe_t *w,const myds_v41_moe_workspace_layout_t *before,
                       const myds_v41_moe_workspace_binding_t *bindings,int failure,uint32_t target) {
    myds_v41_moe_workspace_layout_t q={0};require(myds_v41_moe_workspace_probe_layout(w,&q),"actual after layout");
    require(q.cw_id==before->cw_id&&q.capacity==before->capacity&&q.stable_owner_bytes==before->stable_owner_bytes&&
      q.stable_owner_fingerprint==before->stable_owner_fingerprint&&q.stable_view_fingerprint==before->stable_view_fingerprint&&
      q.binding_count==before->binding_count,"C workspace and stable resources retained");
    if(failure)require(q.moe_batch==before->moe_batch&&q.moe_generation==before->moe_generation&&q.down_id==before->down_id&&
      q.arena_id==before->arena_id&&!memcmp(q.acc_ids,before->acc_ids,sizeof(q.acc_ids)),"all original batch owners retained on failure");
    else require(q.moe_batch==target&&q.moe_generation>before->moe_generation&&q.down_id!=before->down_id&&q.arena_id!=before->arena_id,
                 "same CW publishes new batch owners exactly through production helper");
    for(uint64_t i=0;i<q.binding_count;i++) {
        myds_v41_moe_workspace_binding_t b={0};require(myds_v41_moe_workspace_probe_binding(w,i,&b),"after binding query");
        require(!strcmp(b.name,bindings[i].name)&&b.scope==bindings[i].scope,"stable binding schema/order");
        if(failure||b.scope==1||b.scope==2)require(!memcmp(&b.identity,&bindings[i].identity,sizeof(b.identity)),"all original/stable resource bindings retained");
        if(b.scope==5)require(b.identity.wrapper_id==bindings[i].identity.wrapper_id,"dynamic binding wrapper retained");
        if(!failure&&b.scope==4)require(b.identity.resource_id&&b.identity.resource_id!=bindings[i].identity.resource_id&&
          b.identity.bytes==6u*5120u*4u&&b.identity.offset==(uint64_t)b.index*6u*5120u*4u,"each C down view rebased to new batch resource");
    }
}
static void lg_sweep(uint32_t oldB,uint32_t newB) {
    lg_old_B=oldB;lg_new_B=newB;lg_ordinal=0;lg_phase="dry";
    myds_metal_moe_workspace_allocation_stats_t initial;allocation_stats(&initial);
    require(myds_metal_moe_workspace_test_control(0,0),"localgrow idle reset");
    myds_v41_moe_workspace_probe_t *w=myds_v41_moe_workspace_probe_create(32,2304,oldB,true);require(w!=NULL,"localgrow initial real factory");
    require(myds_metal_moe_workspace_test_control(0,0),"reset trace before actual localgrow");
    require(myds_v41_moe_workspace_growth_observer_set(lg_checkpoint,NULL),"register normal checkpoint observer");
    require(myds_v41_moe_workspace_probe_resize(&w,32,2304,newB,true),"successful same-C localgrow dry trace");
    size_t count=0;myds_metal_moe_workspace_allocation_t *trace=read_trace(&count);
    fprintf(records,"{\"kind\":\"localgrow_coverage\",\"capacity\":32,\"old_B\":%u,\"new_B\":%u,\"selection\":\"every_actual_relative_allocation\",\"trace\":[",oldB,newB);
    for(size_t i=0;i<count;i++){if(i)fputc(',',records);require(trace[i].ordinal==i+1&&trace[i].success,"complete actual successful localgrow trace");fprintf(records,"{\"ordinal\":%" PRIu64 ",\"kind\":%u,\"index\":%d,\"bytes\":%" PRIu64 ",\"site\":",trace[i].ordinal,trace[i].kind,trace[i].index,trace[i].bytes);text(records,trace[i].site);fputc('}',records);}fputs("]}\n",records);fflush(records);
    free(trace);myds_v41_moe_workspace_probe_free(w);myds_metal_moe_workspace_allocation_stats_t now;allocation_stats(&now);require(live_equal(&initial,&now),"dry localgrow free accounting");
    for(uint64_t ordinal=1;ordinal<=count;ordinal++) {
        lg_ordinal=ordinal;lg_phase="failure";
        require(myds_metal_moe_workspace_test_control(0,0),"clear before localgrow NULL case");
        w=myds_v41_moe_workspace_probe_create(32,2304,oldB,true);require(w!=NULL,"original case factory");myds_v41_moe_workspace_probe_t *original=w;
        uint64_t patterned=lg_pattern(w,1);myds_v41_moe_workspace_layout_t prior={0};myds_v41_moe_workspace_binding_t *bindings=lg_bindings(w,&prior);
        lg_layout(w,"before_localgrow_null");myds_metal_moe_workspace_allocation_stats_t before;allocation_stats(&before);
        require(myds_metal_moe_workspace_test_control(ordinal,0),"arm relative localgrow NULL");
        uint64_t dispatch_before=myds_metal_dispatch_count();
        require(!myds_v41_moe_workspace_probe_resize(&w,32,2304,newB,true)&&w==original,"localgrow NULL returns failure without replacing handle");
        uint64_t dispatch_after=myds_metal_dispatch_count();require(dispatch_before==dispatch_after,"failed localgrow encodes no dispatch");
        lg_compare(w,&prior,bindings,1,newB);require(lg_pattern(w,0)==patterned,"all stable bytes preserved on allocation failure");
        lg_layout(w,"after_localgrow_null");failed_trace(ordinal,oldB==32?"localgrow_alias32_to64":"localgrow64_to128",&before);
        require(myds_metal_moe_workspace_test_control(0,0),"clear localgrow NULL before same-handle retry");
        lg_phase="retry";
        require(myds_v41_moe_workspace_probe_resize(&w,32,2304,newB,true)&&w==original,"same-handle real localgrow retry succeeds");
        lg_compare(w,&prior,bindings,0,newB);require(lg_pattern(w,0)==patterned,"all stable bytes preserved on retry");lg_layout(w,"after_localgrow_retry");
        fprintf(records,"{\"kind\":\"localgrow_retry\",\"old_B\":%u,\"new_B\":%u,\"ordinal\":%" PRIu64 ",\"same_handle\":true,\"stable_payload_bytes_checked\":%" PRIu64 ",\"stable_payload_mismatches\":0,\"dispatches_before_failure\":%" PRIu64 ",\"dispatches_after_failure\":%" PRIu64 ",\"success\":true}\n",oldB,newB,ordinal,patterned,dispatch_before,dispatch_after);fflush(records);
        free(bindings);myds_v41_moe_workspace_probe_free(w);allocation_stats(&now);require(live_equal(&initial,&now),"localgrow NULL retry/free full ownership restored");
        cleanup_record("localgrow_retry_free",&initial,&now);
    }
    require(myds_v41_moe_workspace_growth_observer_set(NULL,NULL),"unregister idle observer");
}
#endif
int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--host-only")){selection_host_test();return 0;}
#ifdef MYDS_MOE_WS_SELECTION_ONLY
    return 2;
#else
    require(argc==4&&!strcmp(argv[1],"--run")&&!strcmp(argv[2],"--out-dir")&&argv[3][0]=='/'&&mkdir(argv[3],0700)==0,"explicit run and new absolute output");
    char path[4096];int n=snprintf(path,sizeof(path),"%s/records.jsonl",argv[3]);require(n>0&&(size_t)n<sizeof(path),"output path");records=fopen(path,"wx");require(records!=NULL,"new records");
    require(myds_metal_init("metal"),"Metal init");provider_and_idle_guards();sweep(false);sweep(true);lg_sweep(64,128);lg_sweep(32,64);
    fputs("{\"kind\":\"complete\",\"all_pass\":true,\"implementation_version\":4,\"localgrow_matrices\":2,\"actual_factory_and_Cgrow_coverage\":true,\"every_localgrow_allocation_failed_and_retried\":true,\"real_oom\":false,\"gpu_math_or_timing_tested\":false}\n",records);
    require(fclose(records)==0,"close records");myds_metal_cleanup();return 0;
#endif
}
