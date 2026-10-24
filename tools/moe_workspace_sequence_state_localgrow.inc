/* Original complete same-context raw export, separate localgrow scope. No wait. */
static char *localgrow_prompt(const char *path,size_t *bytes,uint64_t *hash) {
    char *body=read_request(path,bytes);if(!body||*bytes>(SIZE_MAX-8192u)/4u){free(body);return NULL;}
    *hash=hash_bytes(UINT64_C(14695981039346656037),body,*bytes);
    size_t capacity=*bytes*4u+8192u;if(capacity<65536u)capacity=65536u;
    char *prompt=malloc(capacity);
    if(!prompt||!parse_chat_prompt(body,prompt,capacity)){free(prompt);prompt=NULL;}
    free(body);return prompt;
}

static bool localgrow_state(const char *root,const char *label,myds_context_t *ctx,
                         const capture_t *c,bool complete,uint32_t requested_upper) {
    /* Pure status query, never a repair/drain. Refuse to read shared GPU memory
     * if an abnormal existing generation return still has work in flight. */
#ifdef MYDS_BENCH_MOE_WS_API
    if(!myds_metal_moe_workspace_is_quiescent())return false;
#endif
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
    fprintf(out,",\"scope\":\"same_context_moe_workspace_localgrow_sequence\",\"complete_live_state\":%s,\"ctx_pos\":%u,\"next_position\":%u,\"pcache_len\":%u,\"anchor_len\":%u,\"anchor_slot\":%u,\"requested_prefill_upper_bound\":%u,\"rng_state\":\"%016" PRIx64 "\",\"engram_position\":%u,\"limitations\":[\"Complete successful live state is exported. Published snapshots use their own positions; the persistent anchor can outlive a shorter current request.\",\"Persistent disk-cache payload and allocator padding are not exported.\"],\"scalar_order\":[\"ctx_pos\",\"rng\",\"pcache_len\",\"anchor_len\",\"next\",\"engram_position\",\"v41_valid\",\"max_seq_len\",\"max_new_tokens\",\"seed\",\"continue_from_cache\"],\"files\":[",
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
