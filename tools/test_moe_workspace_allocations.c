/* Independent diagnostic factory/failure test. No model or dispatch timing.
 * Compile with MYDS_MOE_WORKSPACE_TESTING against the isolated candidate.
 * --host-only is a pure selection fixture; --run explicitly creates Metal
 * objects. Injected NULL is a simulated allocation failure, never real OOM. */
#define _POSIX_C_SOURCE 200809L
#ifndef MYDS_MOE_WORKSPACE_TESTING
#error "Diagnostic object build required"
#endif
#include "myds_internal.h"
#include "myds_metal.h"
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void require(bool value,const char *message) {
    if(!value){fprintf(stderr,"FAIL: %s\n",message);exit(1);}
}
static int ordinal_cmp(const void *a,const void *b) {
    uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b;return x<y?-1:x>y;
}
/* Sweep distinct allocation sites, not each of C token views. Array sites
 * receive first/middle/last coverage. The trace comes from the actual factory. */
static size_t representatives(const myds_metal_moe_workspace_allocation_t *trace,
                             size_t count,uint64_t *out) {
    bool *used=calloc(count?count:1,sizeof(*used));require(used!=NULL,"selection allocation");
    size_t selected=0;
    for(size_t i=0;i<count;i++) {
        require(trace[i].ordinal==i+1u&&trace[i].success==1&&
                memchr(trace[i].site,0,sizeof(trace[i].site))&&trace[i].site[0],"complete successful relative allocation trace");
        if(used[i])continue;
        size_t first=i,middle=i,last=i,n=0;
        for(size_t j=i;j<count;j++)if(trace[j].kind==trace[i].kind&&!strcmp(trace[j].site,trace[i].site)){used[j]=true;last=j;n++;}
        size_t target=n/2u,k=0;
        for(size_t j=i;j<count;j++)if(trace[j].kind==trace[i].kind&&!strcmp(trace[j].site,trace[i].site))if(k++==target){middle=j;break;}
        out[selected++]=trace[first].ordinal;
        if(middle!=first)out[selected++]=trace[middle].ordinal;
        if(last!=middle&&last!=first)out[selected++]=trace[last].ordinal;
    }
    free(used);qsort(out,selected,sizeof(*out),ordinal_cmp);return selected;
}
static void selection_host_test(void) {
    myds_metal_moe_workspace_allocation_t trace[9]={0};
    const char *sites[]={"owner_a","view_x","view_y","view_x","view_y","view_x","view_y","view_x","owner_b"};
    for(size_t i=0;i<9;i++){trace[i].ordinal=i+1;trace[i].kind=i==0||i==8?1:2;trace[i].success=1;trace[i].index=(int32_t)i;strcpy(trace[i].site,sites[i]);}
    uint64_t got[27]={0};size_t n=representatives(trace,9,got);
    const uint64_t expected[]={1,2,3,5,6,7,8,9};
    require(n==sizeof(expected)/sizeof(*expected)&&!memcmp(got,expected,sizeof(expected)),"first/middle/last independent site representatives");
    puts("{\"kind\":\"complete\",\"scope\":\"cpu_selection_fixture_only\",\"gpu_or_model_executed\":false,\"selected\":[1,2,3,5,6,7,8,9],\"all_pass\":true}");
}
#ifndef MYDS_MOE_WS_SELECTION_ONLY
static FILE *records;
static void text(FILE *out,const char *s) {
    fputc('"',out);for(const unsigned char *p=(const unsigned char*)s;*p;p++) {
        if(*p=='"'||*p=='\\')fputc('\\',out);
        if(*p<32)fprintf(out,"\\u%04x",*p);else fputc(*p,out);
    }fputc('"',out);
}
static void allocation_stats(myds_metal_moe_workspace_allocation_stats_t *s){myds_metal_moe_workspace_allocation_stats(s);}
static bool live_equal(const myds_metal_moe_workspace_allocation_stats_t *a,const myds_metal_moe_workspace_allocation_stats_t *b) {
    return a->live_owners==b->live_owners&&a->live_views==b->live_views&&a->owner_bytes==b->owner_bytes;
}
static void live_json(const myds_metal_moe_workspace_allocation_stats_t *s) {
    fprintf(records,"{\"owners\":%" PRIu64 ",\"views\":%" PRIu64 ",\"owner_bytes\":%" PRIu64 "}",s->live_owners,s->live_views,s->owner_bytes);
}
static void cleanup_record(const char *phase,const myds_metal_moe_workspace_allocation_stats_t *before,
                           const myds_metal_moe_workspace_allocation_stats_t *after) {
    fputs("{\"kind\":\"cleanup_accounting\",\"phase\":",records);text(records,phase);
    fputs(",\"before\":",records);live_json(before);fputs(",\"after\":",records);live_json(after);
    fprintf(records,",\"delta_owners\":%" PRId64 ",\"delta_views\":%" PRId64 ",\"delta_owner_bytes\":%" PRId64
        ",\"live_accounting_restored\":%s}\n",
        (int64_t)after->live_owners-(int64_t)before->live_owners,
        (int64_t)after->live_views-(int64_t)before->live_views,
        (int64_t)after->owner_bytes-(int64_t)before->owner_bytes,live_equal(before,after)?"true":"false");
    fflush(records);
}
static bool workspace_equal(const myds_v41_moe_workspace_probe_stats_t *a,const myds_v41_moe_workspace_probe_stats_t *b) {
    bool same=a->capacity==b->capacity&&a->moe_batch==b->moe_batch&&a->compact==b->compact&&
        a->generation==b->generation&&a->owner_bytes==b->owner_bytes&&a->owners==b->owners&&a->views==b->views&&
        a->legacy_gate_bytes==b->legacy_gate_bytes&&a->legacy_up_bytes==b->legacy_up_bytes&&a->legacy_mid_bytes==b->legacy_mid_bytes&&
        a->requests==b->requests&&a->mm_encoded==b->mm_encoded&&a->mv_encoded==b->mv_encoded&&a->failures==b->failures;
#define SAME_SCRATCH(field) same=same&&a->scratch.field==b->scratch.field
    SAME_SCRATCH(capacity);SAME_SCRATCH(expert_dim);SAME_SCRATCH(mm_capacity);SAME_SCRATCH(f32_capacity);
    SAME_SCRATCH(last_provider);SAME_SCRATCH(generation);SAME_SCRATCH(mm_mid_bytes);
    SAME_SCRATCH(f32_gate_bytes);SAME_SCRATCH(f32_up_bytes);SAME_SCRATCH(f32_mid_bytes);
    SAME_SCRATCH(retired_bytes);SAME_SCRATCH(retired_owners);SAME_SCRATCH(allocated_bytes);SAME_SCRATCH(peak_allocated_bytes);
    SAME_SCRATCH(requests);SAME_SCRATCH(mm_encoded);SAME_SCRATCH(mv_encoded);
    SAME_SCRATCH(fallback_preencode);SAME_SCRATCH(failures);
#undef SAME_SCRATCH
    return same;
}
static myds_metal_moe_workspace_allocation_t *read_trace(size_t *count) {
    uint64_t n=myds_metal_moe_workspace_allocation_trace(NULL,0);require(n>0&&n<100000,"bounded allocation trace");
    myds_metal_moe_workspace_allocation_t *t=calloc((size_t)n,sizeof(*t));require(t!=NULL,"trace allocation");
    require(myds_metal_moe_workspace_allocation_trace(t,n)==n,"complete trace retrieval");*count=(size_t)n;return t;
}
static void failed_trace(uint64_t ordinal,const char *phase,const myds_metal_moe_workspace_allocation_stats_t *before) {
    size_t n=0;myds_metal_moe_workspace_allocation_t *t=read_trace(&n);unsigned failed=0;
    for(size_t i=0;i<n;i++)if(!t[i].success){failed++;require(t[i].ordinal==ordinal,"requested simulated allocation failed");}
    require(failed==1,"exactly one injected NULL, not systemic OOM");
    myds_metal_moe_workspace_allocation_stats_t after;allocation_stats(&after);bool restored=live_equal(before,&after);
    fprintf(records,"{\"kind\":\"allocation_failure\",\"phase\":");text(records,phase);
    fprintf(records,",\"ordinal\":%" PRIu64 ",\"site\":",ordinal);text(records,t[ordinal-1u].site);
    fprintf(records,",\"site_index\":%d,\"allocation_kind\":%u,\"simulated_null\":true,\"real_oom\":false,\"failure_hits\":%u,\"before\":",t[ordinal-1u].index,t[ordinal-1u].kind,failed);
    live_json(before);fputs(",\"after\":",records);live_json(&after);
    fprintf(records,",\"live_accounting_restored\":%s}\n",restored?"true":"false");fflush(records);free(t);require(restored,"partial factory/grow resources fully released");
}
static void sweep(bool grow) {
    myds_metal_moe_workspace_allocation_stats_t initial;allocation_stats(&initial);
    require(myds_metal_moe_workspace_test_control(0,0),"idle diagnostic control");
    myds_v41_moe_workspace_probe_t *probe=myds_v41_moe_workspace_probe_create(32,2304,64,true);require(probe!=NULL,"dry factory C32 B64");
    if(grow){require(myds_metal_moe_workspace_test_control(0,0),"reset before dry grow");require(myds_v41_moe_workspace_probe_resize(&probe,64,2304,128,true),"dry grow C64 B128");}
    size_t count=0;myds_metal_moe_workspace_allocation_t *trace=read_trace(&count);
    uint64_t *targets=calloc(count*3u,sizeof(*targets));require(targets!=NULL,"targets allocation");size_t selected=representatives(trace,count,targets);
    fprintf(records,"{\"kind\":\"coverage\",\"phase\":\"%s\",\"actual_trace_count\":%zu,\"representatives\":%zu,\"selection\":\"all_sites_first_middle_last\",\"trace\":[",grow?"grow32_to64":"create32",count,selected);
    for(size_t i=0;i<count;i++){if(i)fputc(',',records);fprintf(records,"{\"ordinal\":%" PRIu64 ",\"kind\":%u,\"index\":%d,\"bytes\":%" PRIu64 ",\"site\":",trace[i].ordinal,trace[i].kind,trace[i].index,trace[i].bytes);text(records,trace[i].site);fputc('}',records);}
    fputs("]}\n",records);fflush(records);free(trace);myds_v41_moe_workspace_probe_free(probe);
    myds_metal_moe_workspace_allocation_stats_t live;allocation_stats(&live);
    cleanup_record(grow?"dry_grow32_to64":"dry_create32",&initial,&live);
    require(live_equal(&initial,&live),"dry factory cleanup");
    for(size_t i=0;i<selected;i++) {
        myds_v41_moe_workspace_probe_stats_t before_ws={0},after_ws={0};myds_v41_moe_workspace_probe_t *old=NULL;
        require(myds_metal_moe_workspace_test_control(0,0),"reset diagnostic state");probe=NULL;
        if(grow){probe=myds_v41_moe_workspace_probe_create(32,2304,64,true);require(probe!=NULL,"old grow workspace");old=probe;myds_v41_moe_workspace_probe_stats(probe,&before_ws);}
        myds_metal_moe_workspace_allocation_stats_t before;allocation_stats(&before);
        require(myds_metal_moe_workspace_test_control(targets[i],0),"arm injected NULL while idle");
        if(grow){require(!myds_v41_moe_workspace_probe_resize(&probe,64,2304,128,true),"grow reports failure");require(probe==old,"grow preserves old workspace handle");myds_v41_moe_workspace_probe_stats(probe,&after_ws);require(workspace_equal(&before_ws,&after_ws),"grow preserves old generation/capacity/views");}
        else {probe=myds_v41_moe_workspace_probe_create(32,2304,64,true);require(probe==NULL,"factory reports failure");}
        failed_trace(targets[i],grow?"grow32_to64":"create32",&before);
        require(myds_metal_moe_workspace_test_control(0,0),"disable injected NULL before retry");
        if(grow)require(myds_v41_moe_workspace_probe_resize(&probe,64,2304,128,true),"grow retry succeeds");
        else {probe=myds_v41_moe_workspace_probe_create(32,2304,64,true);require(probe!=NULL,"factory retry succeeds");}
        myds_v41_moe_workspace_probe_free(probe);allocation_stats(&live);
        cleanup_record(grow?"grow_retry_free":"create_retry_free",&initial,&live);
        require(live_equal(&initial,&live),"retry/free restores live ownership");
        fprintf(records,"{\"kind\":\"retry\",\"phase\":\"%s\",\"ordinal\":%" PRIu64 ",\"success\":true,\"old_generation_preserved_on_failure\":%s,\"live_accounting_restored\":true}\n",grow?"grow32_to64":"create32",targets[i],grow?"true":"null");fflush(records);
    }free(targets);
}
static void provider_and_idle_guards(void) {
    require(myds_metal_moe_workspace_test_control(0,0),"clear diagnostic mask");
    require(myds_metal_moe_workspace_provider(31)==MYDS_MOE_WS_MV_F32,"31 token MV provider");
    require(myds_metal_moe_workspace_provider(32)==MYDS_MOE_WS_MM_F16,"32 token MM provider");
    for(uint32_t mask=1;mask<=8;mask<<=1){require(myds_metal_moe_workspace_test_control(0,mask),"arm missing provider PSO mask");require(myds_metal_moe_workspace_provider(32)==(mask==8?MYDS_MOE_WS_NONE:MYDS_MOE_WS_MV_F32),"missing shared sum rejects; MM-only requirements select MV");fprintf(records,"{\"kind\":\"provider_precheck\",\"mask\":%u,\"tokens\":32,\"provider\":\"%s\",\"gpu_output_tested\":false}\n",mask,mask==8?"NONE":"MV_F32");}
    require(myds_metal_moe_workspace_test_control(0,0),"clear mask");
    require(myds_metal_begin_batch(),"open empty diagnostic batch");
    require(!myds_metal_moe_workspace_test_control(1,1),"fault controls reject non-idle batch without waiting");
    require(myds_metal_commit_batch(),"complete empty diagnostic batch");
    require(myds_metal_moe_workspace_test_control(0,0),"idle fault control restored");
    fputs("{\"kind\":\"idle_guard\",\"passed\":true,\"scope\":\"empty_batch_diagnostic_only\"}\n",records);
}
#endif
int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--host-only")){selection_host_test();return 0;}
#ifdef MYDS_MOE_WS_SELECTION_ONLY
    fputs("This CPU-only build accepts only --host-only\n",stderr);return 2;
#else
    if(argc!=4||strcmp(argv[1],"--run")||strcmp(argv[2],"--out-dir")){fputs("Usage: test_moe_workspace_allocations --host-only | --run --out-dir NEW_ABSOLUTE\n",stderr);return 2;}
    require(argv[3][0]=='/'&&mkdir(argv[3],0700)==0,"new absolute output directory");char path[4096];int n=snprintf(path,sizeof(path),"%s/records.jsonl",argv[3]);require(n>0&&(size_t)n<sizeof(path),"output path");records=fopen(path,"wx");require(records!=NULL,"exclusive records file");
    fputs("{\"kind\":\"identity\",\"scope\":\"actual_factory_simulated_null_failure_diagnostic\",\"real_oom\":false,\"model_or_weights_loaded\":false,\"gpu_math_or_timing_tested\":false}\n",records);fflush(records);
    require(myds_metal_init("metal"),"Metal initialization");provider_and_idle_guards();sweep(false);sweep(true);
    fputs("{\"kind\":\"complete\",\"all_pass\":true,\"real_oom\":false,\"component_arithmetic_proven\":false,\"fullmodel_cancel_recovery_proven\":false}\n",records);require(fclose(records)==0,"records close");myds_metal_cleanup();return 0;
#endif
}
