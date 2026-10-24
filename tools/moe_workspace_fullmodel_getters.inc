/* Read-only diagnostics for isolated harness custom v41 object only. */
#ifndef MYDS_MOE_WORKSPACE_TESTING
int myds_v41_snapshot_get_slot_engram(const myds_v41_context_t *g,uint32_t slot,uint32_t out[3],uint32_t *position){
    if(!g||slot>=V41_SNAP_SLOTS||!out||!position||!g->snap_pos[slot])return 0;
    memcpy(out,g->snap_engram[slot].tail,3u*sizeof(uint32_t));
    *position=g->snap_pos[slot];return 1;
}
#endif
#ifndef MYDS_BENCH_MOE_WS_API
int myds_v41_gold_moe_scratch(const myds_context_t *ctx,uint64_t out[5]){
    if(!ctx||!ctx->v41||!ctx->v41->chunk_ws||!out)return 0;
    const v41_chunk_workspace_t *cw=ctx->v41->chunk_ws;
    out[0]=cw->capacity;out[1]=cw->moe_batch;
    out[2]=myds_metal_buffer_size(cw->moe_gate_buf);
    out[3]=myds_metal_buffer_size(cw->moe_up_buf);
    out[4]=myds_metal_buffer_size(cw->moe_mid_buf);return 1;
}
#endif
