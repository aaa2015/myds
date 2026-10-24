/* Bounded TESTING handle layout snapshots. Full binding enumeration is kept
 * outside GPU timing; timed memory checkpoints use the O(1) summary only. */
static void lg_string(FILE *f,const char *s){fputc('"',f);for(const unsigned char *p=(const unsigned char *)s;*p;p++){if(*p=='"'||*p=='\\'){fputc('\\',f);fputc(*p,f);}else if(*p<32)fprintf(f,"\\u%04x",*p);else fputc(*p,f);}fputc('"',f);}
static int lg_snapshot_json_mode(FILE *f,const myds_v41_moe_workspace_probe_t *w,int full){
 myds_v41_moe_workspace_layout_t q={0};if(!myds_v41_moe_workspace_probe_layout(w,&q)||q.implementation_version!=3||q.binding_count>131072)return 0;
 fprintf(f,"{\"implementation_version\":%u,\"capacity\":%u,\"moe_batch\":%u,\"compact\":%u,\"cw_id\":%" PRIu64 ",\"moe_generation\":%" PRIu64 ",\"localgrow_attempts\":%" PRIu64 ",\"localgrow_successes\":%" PRIu64 ",\"localgrow_failures\":%" PRIu64 ",\"stable_owner_fingerprint\":%" PRIu64 ",\"stable_view_fingerprint\":%" PRIu64 ",\"stable_owner_bytes\":%" PRIu64 ",\"stable_owners\":%" PRIu64 ",\"stable_views\":%" PRIu64 ",\"down_id\":%" PRIu64 ",\"down_bytes\":%" PRIu64 ",\"arena_id\":%" PRIu64 ",\"acc_ids\":[",q.implementation_version,q.capacity,q.moe_batch,q.compact,q.cw_id,q.moe_generation,q.localgrow_attempts,q.localgrow_successes,q.localgrow_failures,q.stable_owner_fingerprint,q.stable_view_fingerprint,q.stable_owner_bytes,q.stable_owners,q.stable_views,q.down_id,q.down_bytes,q.arena_id);
 for(unsigned i=0;i<7;i++)fprintf(f,"%s%" PRIu64,i?",":"",q.acc_ids[i]);fputs("],\"acc_bytes\":[",f);for(unsigned i=0;i<7;i++)fprintf(f,"%s%" PRIu64,i?",":"",q.acc_bytes[i]);
 fprintf(f,"],\"binding_count\":%" PRIu64 ",\"bindings_complete\":%s,\"bindings\":[",q.binding_count,full?"true":"false");
 for(uint64_t i=0;full&&i<q.binding_count;i++){
  myds_v41_moe_workspace_binding_t b={0};if(!myds_v41_moe_workspace_probe_binding(w,i,&b)||!memchr(b.name,0,sizeof(b.name)))return 0;
  if(i)fputc(',',f);fputs("{\"name\":",f);lg_string(f,b.name);
  fprintf(f,",\"kind\":%u,\"scope\":%u,\"index\":%u,\"wrapper_id\":%" PRIu64 ",\"resource_id\":%" PRIu64 ",\"offset\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"is_view\":%u}",b.kind,b.scope,b.index,b.identity.wrapper_id,b.identity.resource_id,b.identity.offset,b.identity.bytes,b.identity.is_view);
 }
 if(full){myds_v41_moe_workspace_binding_t extra={0};if(myds_v41_moe_workspace_probe_binding(w,q.binding_count,&extra))return 0;}
 fputs("]}",f);return !ferror(f);
}
static inline int lg_snapshot_json(FILE *f,const myds_v41_moe_workspace_probe_t *w){return lg_snapshot_json_mode(f,w,1);}
static inline int lg_snapshot_summary_json(FILE *f,const myds_v41_moe_workspace_probe_t *w){return lg_snapshot_json_mode(f,w,0);}
