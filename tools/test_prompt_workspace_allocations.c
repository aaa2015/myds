/* Actual Metal allocations, actual v4 prompt transaction. Explicit --run only.
 * No model load/inference; six scoped NULL ordinals, clear rejection, retry and
 * cleanup. Normal production objects never compile these fault APIs. */
#define _POSIX_C_SOURCE 200809L
#include "myds_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef MYDS_MOE_WORKSPACE_TESTING
#error This standalone diagnostic requires isolated TESTING objects
#endif
static FILE *raw;
static unsigned records;
static void identity(const myds_metal_buffer_identity_t *x) {
    fprintf(raw,"{\"wrapper_id\":%llu,\"resource_id\":%llu,\"bytes\":%llu,\"offset\":%llu,\"is_view\":%u}",
        (unsigned long long)x->wrapper_id,(unsigned long long)x->resource_id,
        (unsigned long long)x->bytes,(unsigned long long)x->offset,(unsigned)x->is_view);
}
static void record(const char *name,const char *point,unsigned ordinal,const myds_v41_prompt_growth_event_t *s) {
    fprintf(raw,"{\"kind\":\"snapshot\",\"case\":\"%s\",\"point\":\"%s\",\"ordinal\":%u,\"allocation_count\":%u,\"injected_ordinal\":%u,\"injected_hits\":%u,\"implementation_version\":%u,\"cap\":%u,\"mask_bytes\":%llu,\"total_bytes\":%llu,\"quiescent\":%u,\"owners\":[",
        name,point,ordinal,myds_v41_prompt_workspace_test_allocation_count(),myds_v41_prompt_workspace_test_injected_ordinal(),myds_v41_prompt_workspace_test_injected_hits(),s->implementation_version,s->old_cap,
        (unsigned long long)s->old_mask_bytes,(unsigned long long)s->old_total_bytes,s->quiescent);
    for(unsigned i=0;i<6u;i++){if(i)fputc(',',raw);identity(&s->old_owners[i]);}
    fputs("],\"views\":[",raw);for(unsigned i=0;i<6u;i++){if(i)fputc(',',raw);identity(&s->views_current[i]);}
    fputs("],\"allocated_bytes\":[",raw);for(unsigned i=0;i<6u;i++)fprintf(raw,"%s%llu",i?",":"",(unsigned long long)s->old_allocated_bytes[i]);
    fputs("]}\n",raw);fflush(raw);records++;
}
static int snapshot(myds_v41_prompt_workspace_probe_t *p,myds_v41_prompt_growth_event_t *s,const char *name,const char *point,unsigned ordinal) {
    if(!myds_v41_prompt_workspace_probe_snapshot(p,s))return 0;record(name,point,ordinal,s);return 1;
}
static int same(const myds_v41_prompt_growth_event_t *a,const myds_v41_prompt_growth_event_t *b) {
    return a->old_cap==b->old_cap&&a->old_mask_bytes==b->old_mask_bytes&&a->old_total_bytes==b->old_total_bytes&&
        !memcmp(a->old_owners,b->old_owners,sizeof(a->old_owners))&&
        !memcmp(a->views_current,b->views_current,sizeof(a->views_current))&&
        !memcmp(a->old_allocated_bytes,b->old_allocated_bytes,sizeof(a->old_allocated_bytes));
}
static uint64_t device(void){myds_metal_residency_stats_t s;if(!myds_metal_residency_diagnostics(&s))return UINT64_MAX;return s.device_allocated_size;}
static int trial(unsigned ordinal,bool clear_failure,bool clear_success) {
    const char *name=ordinal?"allocation_null":clear_failure?"clear_rejected":clear_success?"clear_success":"clear_disabled";
    uint64_t baseline=device();myds_v41_prompt_workspace_probe_t *p=NULL;int ok=0;
    if(baseline==UINT64_MAX)return 0;
    myds_v41_prompt_growth_event_t before,after;
    if(!myds_v41_prompt_workspace_test_control(0,false))goto done;
    p=myds_v41_prompt_workspace_probe_create(4u,4u);if(!p)goto done;
    if(!snapshot(p,&before,name,"before",ordinal)||!myds_v41_prompt_workspace_test_control(ordinal,clear_failure))goto done;
    int rc=myds_v41_prompt_workspace_probe_resize(p,8u,8u,clear_success||clear_failure);
    if(ordinal||clear_failure) {
        if(rc||myds_v41_prompt_workspace_test_allocation_count()!=6u||
           !snapshot(p,&after,name,"failed",ordinal)||!same(&before,&after))goto done;
        if(!myds_v41_prompt_workspace_test_control(0,false)||
           !myds_v41_prompt_workspace_probe_resize(p,8u,8u,true)||
           !snapshot(p,&after,name,"retry",ordinal))goto done;
    } else {
        if(!rc||!snapshot(p,&after,name,"success",ordinal))goto done;
    }
    if(after.old_cap!=8u||after.old_mask_bytes!=8u)goto done;
    for(unsigned i=0;i<6u;i++) {
        if(!after.old_owners[i].resource_id||after.old_owners[i].resource_id==before.old_owners[i].resource_id)goto done;
        if(clear_success||clear_failure||ordinal) {
            if(after.views_current[i].resource_id||after.views_current[i].bytes||after.views_current[i].offset)goto done;
        } else if(memcmp(&after.views_current[i],&before.views_current[i],sizeof(after.views_current[i])))goto done;
    }
    ok=1;
done:
    myds_v41_prompt_workspace_probe_free(p);(void)myds_v41_prompt_workspace_test_control(0,false);
    uint64_t end=device();
    fprintf(raw,"{\"kind\":\"case_result\",\"case\":\"%s\",\"ordinal\":%u,\"pass\":%s,\"device_before\":%llu,\"device_after_free\":%llu,\"cleanup_exact\":%s}\n",name,ordinal,ok&&baseline==end?"true":"false",(unsigned long long)baseline,(unsigned long long)end,baseline==end?"true":"false");fflush(raw);records++;
    return ok&&baseline==end;
}
int main(int argc,char **argv) {
    if(argc!=4||strcmp(argv[1],"--run")||strcmp(argv[2],"--out-dir")||argv[3][0]!='/') {
        fprintf(stderr,"usage: %s --run --out-dir NEW_ABSOLUTE_DIRECTORY\n",argv[0]);return 2;
    }
    struct stat st;if(!lstat(argv[3],&st)||mkdir(argv[3],0700)){perror("new out-dir");return 2;}
    char path[4096];snprintf(path,sizeof(path),"%s/records.jsonl",argv[3]);raw=fopen(path,"wx");if(!raw)return 2;
    int ok=myds_metal_init(NULL)&&myds_metal_moe_workspace_is_quiescent();
    for(unsigned i=1u;i<=6u&&ok;i++)ok=trial(i,false,true);
    if(ok)ok=trial(0u,true,true);
    if(ok)ok=trial(0u,false,true);
    if(ok)ok=trial(0u,false,false);
    fprintf(raw,"{\"kind\":\"complete\",\"status\":\"%s\",\"implementation_version\":4,\"scope\":\"actual_metal_prompt_group_transaction_no_model\",\"cases\":9,\"snapshot_or_result_records\":%u,\"gate_pass\":%s}\n",ok?"complete":"failed",records,ok?"true":"false");fclose(raw);
    snprintf(path,sizeof(path),"%s/summary.json",argv[3]);FILE *f=fopen(path,"wx");if(!f)return 2;
    fprintf(f,"{\"status\":\"%s\",\"implementation_version\":4,\"prompt_allocations_gate_pass\":%s,\"cases\":9}\n",ok?"complete":"failed",ok?"true":"false");fclose(f);
    return ok?0:1;
}
