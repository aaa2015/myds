/* Process-local prompt owner/view identity only, outside generation. No wait. */
static int promptclear_mode(void) {
    int mode=myds_metal_prefill_prompt_view_clear_parse_mode();
    int workspace_mode=myds_metal_prefill_moe_workspace_parse_mode();
    cancel_need(mode>=0&&workspace_mode>=0&&mode==workspace_mode,"two explicitly aligned strict candidate flags");return mode;
}
static void promptclear_initialized_mode(int requested) {
    cancel_need(bench_ws_mode()==requested&&myds_metal_prefill_prompt_view_clear_enabled()==requested,"initialized candidate modes differ from requested environment");
}
static inline bool promptclear_identity_equal(const myds_metal_buffer_identity_t *a,const myds_metal_buffer_identity_t *b) {
    return a->wrapper_id==b->wrapper_id&&a->resource_id==b->resource_id&&a->offset==b->offset&&a->bytes==b->bytes&&a->is_view==b->is_view;
}
static inline bool promptclear_owners_equal(const myds_v41_prompt_growth_event_t *a,const myds_v41_prompt_growth_event_t *b) {
    if(a->old_cap!=b->old_cap||a->old_mask_bytes!=b->old_mask_bytes||a->old_total_bytes!=b->old_total_bytes)return false;
    for(unsigned i=0;i<6;i++)if(!promptclear_identity_equal(&a->old_owners[i],&b->old_owners[i]))return false;return true;
}
static inline bool promptclear_views_equal(const myds_v41_prompt_growth_event_t *a,const myds_v41_prompt_growth_event_t *b) {
    for(unsigned i=0;i<6;i++)if(!promptclear_identity_equal(&a->views_before[i],&b->views_before[i]))return false;return true;
}
static bool promptclear_snapshot(const char *directory,const char *label,const myds_context_t *ctx,myds_v41_prompt_growth_event_t *copy) {
    myds_v41_prompt_growth_event_t q={0};
    if(!myds_metal_moe_workspace_is_quiescent()||!myds_v41_prefill_prompt_workspace_snapshot(ctx,&q)||!q.quiescent)return false;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/prompt-%s.json",directory,label);if(n<0||(size_t)n>=sizeof(path))return false;
    FILE *f=fopen(path,"wbx");if(!f)return false;
    fprintf(f,"{\"schema_version\":1,\"scope\":\"process_local_prompt_group_readonly_identity\",\"implementation_version\":%u,\"compact\":%u,\"clear_mode\":%u,\"quiescent\":%u,\"cw_id\":%" PRIu64 ",\"cap\":%u,\"mask_bytes\":%" PRIu64 ",\"total_bytes\":%" PRIu64 ",\"owner_order\":[\"RES\",\"PRE\",\"SELECTED\",\"MASK\",\"ENGRAM0\",\"ENGRAM1\"],\"owners\":[",q.implementation_version,q.compact,q.clear_mode,q.quiescent,q.cw_id,q.old_cap,q.old_mask_bytes,q.old_total_bytes);
    for(unsigned k=0;k<2;k++) {
        if(k)fputs("],\"views\":[",f);
        for(unsigned i=0;i<6;i++) {
            const myds_metal_buffer_identity_t *v=k?&q.views_before[i]:&q.old_owners[i];
            fprintf(f,"%s{\"wrapper_id\":%" PRIu64 ",\"resource_id\":%" PRIu64 ",\"offset\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"is_view\":%u}",i?",":"",v->wrapper_id,v->resource_id,v->offset,v->bytes,v->is_view);
        }
    }
    fprintf(f,"],\"complete\":true}\n");bool ok=!ferror(f);if(fclose(f))ok=false;*copy=q;return ok;
}
