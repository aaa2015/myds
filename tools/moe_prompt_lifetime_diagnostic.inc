/* Read-only normal prompt/deferred lifetime records. Include after now_s(),
 * json_string(), myds_internal.h and moe_workspace_memory.inc. These callbacks
 * never read resource payloads, submit, synchronize, wait or mutate resources. */
#include <stddef.h>

typedef struct {
    FILE *file;
    uint32_t sequence,stage;
    bool in_generation,registered,include_deferred;
    int error;
    uint64_t records,prompt_events,deferred_events;
} promptdiag_state_t;

static void promptdiag_position(promptdiag_state_t *s,uint32_t stage,bool active) {
    if(!s)return;
    if(stage>2u)s->error=1;
    s->stage=stage;s->in_generation=active;
}
static void promptdiag_stream(promptdiag_state_t *s,FILE *file,uint32_t sequence) {
    if(!s)return;
    s->file=file;s->sequence=sequence;s->in_generation=false;
    if(file){s->records=0;s->prompt_events=0;s->deferred_events=0;s->stage=0;}
}

#ifdef MYDS_BENCH_MOE_WS_API
static void promptdiag_actual(FILE *file,promptdiag_state_t *s) {
    const double begin=now_s();
    task_vm_info_data_t vm={0};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    kern_return_t kr=task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&vm,&count);
    myds_metal_residency_stats_t metal={0};bool device=myds_metal_residency_diagnostics(&metal);
    const double end=now_s();
    const bool valid=kr==KERN_SUCCESS&&
       (size_t)count*sizeof(natural_t)>=offsetof(task_vm_info_data_t,phys_footprint)+sizeof(vm.phys_footprint);
    if(!valid||!device)s->error=1;
    fprintf(file,"{\"task_valid\":%s,\"device_valid\":%s,\"task_info_return\":%d,\"task_info_count_words\":%u,"
        "\"task_info_requested_words\":%u,\"snapshot_begin_s\":%.9f,\"snapshot_end_s\":%.9f,"
        "\"phys_footprint_bytes\":%" PRIu64 ",\"rss_bytes\":%" PRIu64 ",\"device_allocated_bytes\":%" PRIu64 ",\"task_vm\":{",
        valid?"true":"false",device?"true":"false",kr,count,(unsigned)TASK_VM_INFO_COUNT,begin,end,
        (uint64_t)vm.phys_footprint,(uint64_t)vm.resident_size,metal.device_allocated_size);
    bool comma=false;
#define PD_FIELD(field,fmt,cast) do { \
    if(comma)fputc(',',file);comma=true;fputs("\"" #field "\":",file); \
    if(kr==KERN_SUCCESS&&(size_t)count*sizeof(natural_t)>=offsetof(task_vm_info_data_t,field)+sizeof(vm.field)) \
        fprintf(file,fmt,(cast)vm.field);else fputs("null",file); \
}while(0)
#define PD_U(field) PD_FIELD(field,"%" PRIu64,uint64_t)
#define PD_I(field) PD_FIELD(field,"%" PRId64,int64_t)
    PD_U(virtual_size);PD_I(region_count);PD_I(page_size);PD_U(resident_size);PD_U(resident_size_peak);
    PD_U(device);PD_U(device_peak);PD_U(internal);PD_U(internal_peak);PD_U(external);PD_U(external_peak);
    PD_U(reusable);PD_U(reusable_peak);PD_U(purgeable_volatile_pmap);PD_U(purgeable_volatile_resident);
    PD_U(purgeable_volatile_virtual);PD_U(compressed);PD_U(compressed_peak);PD_U(compressed_lifetime);PD_U(phys_footprint);
    PD_I(ledger_phys_footprint_peak);PD_I(ledger_purgeable_nonvolatile);PD_I(ledger_purgeable_novolatile_compressed);
    PD_I(ledger_purgeable_volatile);PD_I(ledger_purgeable_volatile_compressed);
    PD_I(ledger_tag_network_nonvolatile);PD_I(ledger_tag_network_nonvolatile_compressed);
    PD_I(ledger_tag_network_volatile);PD_I(ledger_tag_network_volatile_compressed);
    PD_I(ledger_tag_media_footprint);PD_I(ledger_tag_media_footprint_compressed);
    PD_I(ledger_tag_media_nofootprint);PD_I(ledger_tag_media_nofootprint_compressed);
    PD_I(ledger_tag_graphics_footprint);PD_I(ledger_tag_graphics_footprint_compressed);
    PD_I(ledger_tag_graphics_nofootprint);PD_I(ledger_tag_graphics_nofootprint_compressed);
    PD_I(ledger_tag_neural_footprint);PD_I(ledger_tag_neural_footprint_compressed);
    PD_I(ledger_tag_neural_nofootprint);PD_I(ledger_tag_neural_nofootprint_compressed);
#undef PD_I
#undef PD_U
#undef PD_FIELD
    fputs("}}",file);
}
static bool promptdiag_prefix(promptdiag_state_t *s,const char *kind) {
    if(!s||!s->file){if(s)s->error=1;return false;}
    fputs("{\"kind\":",s->file);json_string(s->file,kind);
    fprintf(s->file,",\"sequence\":%u,\"stage\":%u,\"in_generation\":%s,\"record_index\":%" PRIu64
        ",\"monotonic_s\":%.9f,\"observer_CPU_only\":true,\"GPU_work_submitted_by_callback\":false,\"extra_gpu_waits\":0",
        s->sequence,s->stage,s->in_generation?"true":"false",++s->records,now_s());
    return true;
}
static void promptdiag_finish(promptdiag_state_t *s) {
    fputs(",\"actual\":",s->file);promptdiag_actual(s->file,s);
    if(fputs("}\n",s->file)<0||fflush(s->file)||ferror(s->file))s->error=1;
}
static void promptdiag_id(FILE *file,const myds_metal_buffer_identity_t *v) {
    fprintf(file,"{\"wrapper_id\":%" PRIu64 ",\"resource_id\":%" PRIu64 ",\"offset\":%" PRIu64
        ",\"bytes\":%" PRIu64 ",\"is_view\":%u}",v->wrapper_id,v->resource_id,v->offset,v->bytes,v->is_view);
}
static void promptdiag_ids(FILE *file,const char *key,const myds_metal_buffer_identity_t values[6]) {
    fprintf(file,",\"%s\":[",key);
    for(unsigned i=0;i<6;i++){if(i)fputc(',',file);promptdiag_id(file,&values[i]);}fputc(']',file);
}
static void promptdiag_sizes(FILE *file,const char *key,const uint64_t values[6]) {
    fprintf(file,",\"%s\":[",key);for(unsigned i=0;i<6;i++)fprintf(file,"%s%" PRIu64,i?",":"",values[i]);fputc(']',file);
}
static void promptdiag_prompt(void *opaque,const myds_v41_prompt_growth_event_t *e) {
    promptdiag_state_t *s=opaque;if(!promptdiag_prefix(s,"prompt_growth"))return;++s->prompt_events;
    fprintf(s->file,",\"implementation_version\":%u,\"event\":%u,\"compact\":%u,\"clear_mode\":%u,"
       "\"quiescent\":%u,\"clear_applied\":%u,\"old_cap\":%u,\"new_cap\":%u,\"prompt_sequence\":%" PRIu64
       ",\"cw_id\":%" PRIu64 ",\"old_mask_bytes\":%" PRIu64 ",\"new_mask_bytes\":%" PRIu64
       ",\"old_total_bytes\":%" PRIu64 ",\"new_total_bytes\":%" PRIu64,
       e->implementation_version,e->event,e->compact,e->clear_mode,e->quiescent,e->clear_applied,e->old_cap,e->new_cap,
       e->sequence,e->cw_id,e->old_mask_bytes,e->new_mask_bytes,e->old_total_bytes,e->new_total_bytes);
    fputs(",\"owner_order\":[\"RES\",\"PRE\",\"SELECTED\",\"MASK\",\"ENGRAM0\",\"ENGRAM1\"]",s->file);
    promptdiag_ids(s->file,"old_owners",e->old_owners);promptdiag_ids(s->file,"new_owners",e->new_owners);
    promptdiag_ids(s->file,"views_before",e->views_before);promptdiag_ids(s->file,"views_current",e->views_current);
    promptdiag_sizes(s->file,"old_allocated_bytes",e->old_allocated_bytes);promptdiag_sizes(s->file,"new_allocated_bytes",e->new_allocated_bytes);
    promptdiag_finish(s);
}
static void promptdiag_deferred(void *opaque,const myds_metal_deferred_buffer_event_t *e) {
    promptdiag_state_t *s=opaque;if(!promptdiag_prefix(s,"deferred_buffer"))return;++s->deferred_events;
    fprintf(s->file,",\"implementation_version\":%u,\"event\":%u,\"batch_active\":%u,\"inflight_count\":%u,"
       "\"origin\":%u,\"deferred_sequence\":%" PRIu64 ",\"commit_epoch\":%" PRIu64 ",\"resource_id\":%" PRIu64
       ",\"resource_bytes\":%" PRIu64 ",\"resource_allocated_size\":%" PRIu64 ",\"entries\":%" PRIu64,
       e->implementation_version,e->event,e->batch_active,e->inflight_count,e->origin,e->sequence,e->commit_epoch,
       e->resource_id,e->resource_bytes,e->resource_allocated_size,e->entries);
    promptdiag_finish(s);
}
static bool promptdiag_register_mode(promptdiag_state_t *s,bool include_deferred) {
    if(!s||s->registered)return false;
    if(!myds_v41_prompt_workspace_growth_observer_set(promptdiag_prompt,s))return false;
    if(include_deferred&&!myds_metal_deferred_buffer_observer_set(promptdiag_deferred,s)) {
        (void)myds_v41_prompt_workspace_growth_observer_set(NULL,NULL);return false;
    }
    s->registered=true;s->include_deferred=include_deferred;return true;
}
static inline bool promptdiag_register(promptdiag_state_t *s) { return promptdiag_register_mode(s,true); }
static bool promptdiag_unregister(promptdiag_state_t *s) {
    if(!s||!s->registered)return false;
    bool deferred=!s->include_deferred||myds_metal_deferred_buffer_observer_set(NULL,NULL);
    bool prompt=myds_v41_prompt_workspace_growth_observer_set(NULL,NULL);
    if(deferred&&prompt){s->registered=false;s->include_deferred=false;return true;}return false;
}
#else
static inline bool promptdiag_register_mode(promptdiag_state_t *s,bool include_deferred) {(void)s;(void)include_deferred;return false;}
static inline bool promptdiag_register(promptdiag_state_t *s) {(void)s;return false;}
static bool promptdiag_unregister(promptdiag_state_t *s) {(void)s;return false;}
#endif
