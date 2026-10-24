/* Normal read-only layout evidence, outside measured generation. Resource IDs
 * have meaning only before/after within one process, never as cross-arm state. */
#ifdef MYDS_BENCH_MOE_WS_API
typedef myds_v41_moe_workspace_layout_t localgrow_layout_t;
static int localgrow_layout(const myds_context_t *ctx,localgrow_layout_t *out) {
    return myds_v41_prefill_moe_workspace_layout(ctx,out);
}
#else
typedef struct {
    uint32_t implementation_version,capacity,moe_batch,compact;
    uint64_t cw_id,moe_generation,localgrow_attempts,localgrow_successes,localgrow_failures;
    uint64_t stable_owner_fingerprint,stable_view_fingerprint,stable_owner_bytes,stable_owners,stable_views;
    uint64_t down_id,down_bytes,acc_ids[7],acc_bytes[7],arena_id,binding_count;
} localgrow_layout_t;
static int localgrow_layout(const myds_context_t *ctx,localgrow_layout_t *out) {
    (void)ctx;memset(out,0,sizeof(*out));return 0;
}
#endif
static bool localgrow_dump_layout(const char *path,const myds_context_t *ctx,localgrow_layout_t *copy,bool *present) {
    localgrow_layout_t q={0};*present=localgrow_layout(ctx,&q)!=0;*copy=q;
    FILE *f=fopen(path,"wbx");if(!f)return false;
    fprintf(f,"{\"schema_version\":1,\"scope\":\"process_local_actual_binding_identity\",\"present\":%s,\"implementation_version\":%u,\"capacity\":%u,\"moe_batch\":%u,\"compact\":%u,\"cw_id\":%" PRIu64 ",\"moe_generation\":%" PRIu64 ",\"localgrow_attempts\":%" PRIu64 ",\"localgrow_successes\":%" PRIu64 ",\"localgrow_failures\":%" PRIu64 ",\"stable_owner_fingerprint\":%" PRIu64 ",\"stable_view_fingerprint\":%" PRIu64 ",\"stable_owner_bytes\":%" PRIu64 ",\"stable_owners\":%" PRIu64 ",\"stable_views\":%" PRIu64 ",\"down_id\":%" PRIu64 ",\"down_bytes\":%" PRIu64 ",\"arena_id\":%" PRIu64 ",\"binding_count\":%" PRIu64 ",\"acc_ids\":[",
        *present?"true":"false",q.implementation_version,q.capacity,q.moe_batch,q.compact,q.cw_id,q.moe_generation,q.localgrow_attempts,q.localgrow_successes,q.localgrow_failures,q.stable_owner_fingerprint,q.stable_view_fingerprint,q.stable_owner_bytes,q.stable_owners,q.stable_views,q.down_id,q.down_bytes,q.arena_id,q.binding_count);
    for(unsigned i=0;i<7;i++)fprintf(f,"%s%" PRIu64,i?",":"",q.acc_ids[i]);fputs("],\"acc_bytes\":[",f);
    for(unsigned i=0;i<7;i++)fprintf(f,"%s%" PRIu64,i?",":"",q.acc_bytes[i]);fputs("],\"bindings\":[",f);
    bool ok=q.binding_count<=131072u;
#ifdef MYDS_BENCH_MOE_WS_API
    for(uint64_t i=0;ok&&i<q.binding_count;i++) {
        myds_v41_moe_workspace_binding_t b={0};if(!myds_v41_prefill_moe_workspace_binding(ctx,i,&b)){ok=false;break;}
        if(!memchr(b.name,0,sizeof(b.name))){ok=false;break;}if(i)fputc(',',f);
        fputs("{\"name\":",f);json_string(f,b.name);
        fprintf(f,",\"kind\":%u,\"scope\":%u,\"index\":%u,\"wrapper_id\":%" PRIu64 ",\"resource_id\":%" PRIu64 ",\"offset\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"is_view\":%u}",b.kind,b.scope,b.index,b.identity.wrapper_id,b.identity.resource_id,b.identity.offset,b.identity.bytes,b.identity.is_view);
    }
    if(ok&&*present){myds_v41_moe_workspace_binding_t extra={0};if(myds_v41_prefill_moe_workspace_binding(ctx,q.binding_count,&extra))ok=false;}
#endif
    fprintf(f,"],\"complete\":%s}\n",ok?"true":"false");if(fclose(f))ok=false;return ok;
}
#ifdef MYDS_BENCH_MOE_WS_API
static bool localgrow_stable(const localgrow_layout_t *a,const localgrow_layout_t *b) {
    return a->cw_id==b->cw_id&&a->capacity==b->capacity&&a->stable_owner_fingerprint==b->stable_owner_fingerprint&&
      a->stable_view_fingerprint==b->stable_view_fingerprint&&a->stable_owner_bytes==b->stable_owner_bytes&&
      a->stable_owners==b->stable_owners&&a->stable_views==b->stable_views;
}
#endif
