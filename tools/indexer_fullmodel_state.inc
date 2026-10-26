/* Current multi-slot anchor export. Included at the legacy export_state point;
 * uses that driver's unchanged state_file/state_metal/state_window helpers. */
static bool bench_indexer_anchor_bounds(const myds_context_t *ctx) {
    if (!ctx || ctx->anchor_slots > MYDS_ANCHOR_SLOTS_MAX || ctx->anchor_n > ctx->anchor_slots ||
        (ctx->anchor_slots && !ctx->anchor_cap) ||
        ctx->pcache_len > ctx->pcache_cap || (ctx->pcache_len && !ctx->pcache_tokens)) return false;
    uint32_t occupied = 0;
    for (uint32_t slot = 0; slot < MYDS_ANCHOR_SLOTS_MAX; ++slot) {
        const uint32_t count = ctx->anchor_len[slot];
        if (slot >= ctx->anchor_slots && count) return false;
        if (count > ctx->anchor_cap || (count && !ctx->anchor_tokens[slot])) return false;
        occupied += count != 0;
    }
    return occupied == ctx->anchor_n;
}

static bool export_state(uint32_t run, myds_context_t *ctx, const capture_t *capture) {
    const char *sink = getenv("MYDS_BENCH_STATE_DIR");
    if (!sink || !*sink || !ctx || !ctx->v41 || !capture ||
        !bench_indexer_anchor_bounds(ctx) || capture->count > MAX_OUTPUT_TOKENS ||
        capture->byte_count > MAX_OUTPUT_BYTES || (capture->byte_count && !capture->raw_bytes)) return false;
    uint32_t next = myds_v41_snapshot_get_next_position(ctx->v41);
    if (next > ctx->cfg.max_seq_len) return false;
    char directory[4096], path[4096];
    int n = snprintf(directory, sizeof(directory), "%s/run-%u", sink, run);
    if (n < 0 || (size_t)n >= sizeof(directory) || mkdir(directory, 0700) != 0) return false;
    n = snprintf(path, sizeof(path), "%s/manifest.json", directory);
    if (n < 0 || (size_t)n >= sizeof(path)) return false;
    FILE *out = fopen(path, "wbx");
    if (!out) return false;
    uint32_t history[3] = {0};
    myds_v41_snapshot_get_engram_history(ctx->v41, history);
    uint32_t snapshot_positions[MYDS_SNAP_SLOTS * 2u];
    for (uint32_t slot = 0; slot < MYDS_SNAP_SLOTS; ++slot) {
        snapshot_positions[slot * 2u] = myds_v41_snapshot_get_pos(ctx->v41, slot);
        snapshot_positions[slot * 2u + 1u] = myds_v41_snapshot_get_staged_pos(ctx->v41, slot);
    }
    uint64_t scalars[] = {ctx->pos, ctx->rng_state, ctx->cfg.max_seq_len, ctx->cfg.max_new_tokens,
        (uint32_t)ctx->cfg.seed, ctx->kv_capacity, ctx->pcache_len, ctx->anchor_n,
        ctx->continue_from_cache ? 1u : 0u, myds_v41_snapshot_get_valid(ctx->v41) ? 1u : 0u,
        next, myds_v41_snapshot_get_engram_position(ctx->v41),
        myds_v41_snapshot_allocated_bytes(ctx->v41), ctx->prefill_capacity,
        myds_v41_low_kv_enabled(ctx->v41) ? 1u : 0u, ctx->anchor_slots, ctx->anchor_cap};
    fprintf(out, "{\"schema_version\":2,\"subject\":\"prefill_indexer_k_resident\",\"run\":%u,"
        "\"scope\":\"fresh_context_full_prefill_final_state\",\"metadata\":{\"byte_order\":\"little\","
        "\"fresh_context\":true,\"ctx_pos\":%u,\"next_position\":%u,\"window_rows\":%u,"
        "\"layers\":40,\"window_width\":512,\"window_order\":\"chronological_active_rows\","
        "\"compressed_row_bytes\":1024,\"index_row_bytes\":256,\"rng_state\":\"%016" PRIx64 "\","
        "\"engram_position\":%u,\"engram_tail\":[%u,%u,%u],\"anchor_layout\":\"per_slot_active_tokens\","
        "\"anchor_slots\":%u,\"anchor_cap\":%u,\"anchor_n\":%u,"
        "\"scalar_order\":[\"ctx_pos\",\"rng_state\",\"max_seq_len\",\"max_new_tokens\",\"seed_u32\","
        "\"kv_capacity\",\"pcache_len\",\"anchor_n\",\"continue_from_cache\",\"v41_valid\","
        "\"next_position\",\"engram_position\",\"snapshot_allocated_bytes\",\"prefill_capacity\","
        "\"low_kv_enabled\",\"anchor_slots\",\"anchor_cap\"]},"
        "\"limitations\":[\"Final live state and published snapshot arrays only; historical snapshot Engram tails are not exported.\","
        "\"No persistent disk-cache payload, draft-model state, allocator padding, inactive capacity, scratch or intermediate logits.\"],\"files\":[",
        run, ctx->pos, next, next < 128u ? next : 128u, ctx->rng_state,
        myds_v41_snapshot_get_engram_position(ctx->v41), history[0], history[1], history[2],
        ctx->anchor_slots, ctx->anchor_cap, ctx->anchor_n);
    unsigned files = 0;
    bool ok = true;
#define SF(name,data,bytes) do { if (ok) ok=state_file(out,directory,name,data,bytes,&files); } while (0)
#define SM(name,buffer,bytes) do { if (ok) ok=state_metal(out,directory,name,buffer,bytes,&files); } while (0)
    SF("context-scalars.u64le", scalars, sizeof(scalars));
    SF("engram-tail.u32le", history, sizeof(history));
    SF("snapshot-positions.u32le", snapshot_positions, sizeof(snapshot_positions));
    SF("prefix-tokens.i32le", ctx->pcache_tokens, (size_t)ctx->pcache_len * sizeof(int32_t));
    SF("anchor-lengths.u32le", ctx->anchor_len, (size_t)ctx->anchor_slots * sizeof(uint32_t));
    char name[128];
    for (uint32_t slot = 0; ok && slot < ctx->anchor_slots; ++slot) {
        snprintf(name, sizeof(name), "anchor-%02u-tokens.i32le", slot);
        SF(name, ctx->anchor_tokens[slot], (size_t)ctx->anchor_len[slot] * sizeof(int32_t));
    }
    SF("tokens.i32le", capture->ids, (size_t)capture->count * sizeof(int32_t));
    SF("output.bin", capture->raw_bytes, (size_t)capture->byte_count);
    for (uint32_t layer = 0; ok && layer < 40u; ++layer) {
        snprintf(name, sizeof(name), "layer-%02u-window-active.f32", layer);
        ok = state_window(out, directory, name, myds_v41_snapshot_get_window(ctx->v41, layer), 0, next, &files);
    }
    for (uint32_t owner = 0; ok && owner < 4u; ++owner) {
        uint32_t rows = next / (owner == 3u ? 1u : 2u);
        if (owner < 3u) {
            snprintf(name, sizeof(name), "owner-%u-previous-kv.f32", owner);
            SM(name, myds_v41_snapshot_get_previous_kv(ctx->v41, owner), 512u * sizeof(float));
            snprintf(name, sizeof(name), "owner-%u-previous-score.f32", owner);
            SM(name, myds_v41_snapshot_get_previous_score(ctx->v41, owner), 512u * sizeof(float));
        }
        snprintf(name, sizeof(name), "owner-%u-compressed-active.bf16", owner);
        SM(name, myds_v41_snapshot_get_compressed(ctx->v41, owner), (size_t)rows * 1024u);
        snprintf(name, sizeof(name), "owner-%u-index-active.bf16", owner);
        SM(name, myds_v41_snapshot_get_index_cache(ctx->v41, owner), (size_t)rows * 256u);
    }
    for (uint32_t slot = 0; ok && slot < MYDS_SNAP_SLOTS; ++slot) {
        uint32_t pos = snapshot_positions[slot * 2u];
        if (!pos) continue;
        if (pos > next) { ok = false; break; }
        myds_metal_buffer_t *window = myds_v41_snapshot_get_snap_window(ctx->v41, slot);
        for (uint32_t layer = 0; ok && layer < 40u; ++layer) {
            snprintf(name, sizeof(name), "snapshot-%02u-layer-%02u-window-active.f32", slot, layer);
            ok = state_window(out, directory, name, window, (size_t)layer * 128u * 512u * sizeof(float), pos, &files);
        }
        for (uint32_t owner = 0; ok && owner < 3u; ++owner) {
            snprintf(name, sizeof(name), "snapshot-%02u-owner-%u-previous-kv.f32", slot, owner);
            SM(name, myds_v41_snapshot_get_snap_prev_kv(ctx->v41, slot, owner), 512u * sizeof(float));
            snprintf(name, sizeof(name), "snapshot-%02u-owner-%u-previous-score.f32", slot, owner);
            SM(name, myds_v41_snapshot_get_snap_prev_score(ctx->v41, slot, owner), 512u * sizeof(float));
        }
    }
#undef SF
#undef SM
    fprintf(out, "],\"file_count\":%u,\"complete\":%s}\n", files, ok ? "true" : "false");
    if (fclose(out) != 0) ok = false;
    return ok;
}
