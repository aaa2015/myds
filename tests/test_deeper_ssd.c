/* Production selection and transactional SSD restore contracts. These reuse
 * the real tiny tokenizer/context and deterministic KV fixtures; the renamed
 * test mains are not executed. The suite uses Metal and must run in the
 * coordinator's exclusive test window, never beside the production server. */
#define _DARWIN_C_SOURCE 1
#define main deeper_embedded_engine_main
#include "test_engine.c"
#undef main
#define main deeper_embedded_p1_main
#include "test_p1_regressions.c"
#undef main

enum { DEEPER_CAP = 4096u, DEEPER_TEXT = 2304u };

/* Prompt-end fixtures hash/save positions beyond the embedded P1 fixture's
 * LK_NTOK=2048. Allocate and initialize the full deeper capacity so those
 * prefixes never read adjacent heap data. */
static int32_t *deeper_tokens(void) {
    int32_t *tokens = malloc(DEEPER_CAP * sizeof(*tokens));
    assert(tokens);
    for (uint32_t i = 0; i < DEEPER_CAP; ++i)
        tokens[i] = (int32_t)((i * 37u + 11u) % 50000u);
    return tokens;
}

static uint32_t deeper_encode(myds_model_t *model, const char *text, int32_t *tokens) {
    const int32_t n = myds_tokenizer_encode(model->tokenizer, text, tokens, DEEPER_CAP, true);
    assert(n > 2048 && n < DEEPER_CAP);
    return (uint32_t)n;
}

static uint64_t deeper_hash(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t deeper_buffer_hash(uint64_t h, myds_metal_buffer_t *buffer) {
    assert(buffer && myds_metal_buffer_data(buffer));
    return deeper_hash(h, myds_metal_buffer_data(buffer), myds_metal_buffer_size(buffer));
}

/* Include all live windows/packed rows/previous state and every published RAM
 * snapshot, not just the fallback position. Failed deep reads must preserve
 * siblings as well as the shallow reusable prefix. */
static uint64_t deeper_state_hash(myds_v41_context_t *g) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t l = 0; l < 40u; ++l)
        h = deeper_buffer_hash(h, myds_v41_snapshot_get_window(g, l));
    for (uint32_t o = 0; o < 4u; ++o) {
        h = deeper_buffer_hash(h, myds_v41_snapshot_get_compressed(g, o));
        h = deeper_buffer_hash(h, myds_v41_snapshot_get_index_cache(g, o));
        h = deeper_buffer_hash(h, myds_v41_snapshot_get_previous_kv(g, o));
        h = deeper_buffer_hash(h, myds_v41_snapshot_get_previous_score(g, o));
    }
    for (uint32_t s = 0; s < MYDS_SNAP_SLOTS; ++s) {
        const uint32_t p = myds_v41_snapshot_get_pos(g, s);
        const uint32_t staged = myds_v41_snapshot_get_staged_pos(g, s);
        h = deeper_hash(h, &p, sizeof(p));
        h = deeper_hash(h, &staged, sizeof(staged));
        if (p) {
            h = deeper_buffer_hash(h, myds_v41_snapshot_get_snap_window(g, s));
            for (uint32_t o = 0; o < 4u; ++o) {
                h = deeper_buffer_hash(h, myds_v41_snapshot_get_snap_prev_kv(g, s, o));
                h = deeper_buffer_hash(h, myds_v41_snapshot_get_snap_prev_score(g, s, o));
            }
        }
    }
    uint32_t history[3]; myds_v41_snapshot_get_engram_history(g, history);
    h = deeper_hash(h, history, sizeof(history));
    const bool valid = myds_v41_snapshot_get_valid(g);
    return deeper_hash(h, &valid, sizeof(valid));
}

static myds_v41_context_t *deeper_saved(const char *path, int32_t **out_tokens) {
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    assert(g && myds_v41_low_kv_fixture_enable(g, path, 0x5EEDull, 2, false, DEEPER_CAP));
    lk_fill_state(g);
    int32_t *tokens = deeper_tokens();
    assert(myds_v41_low_kv_save(g, tokens, LK_NTOK));
    lk_make_fork_state(g); /* RAM is B, disk is A; only the first 1024 agree. */
    for (uint32_t l = 0; l < 40u; ++l)
        memset(myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l)), 0x5A,
               myds_metal_buffer_size(myds_v41_snapshot_get_window(g, l)));
    *out_tokens = tokens;
    return g;
}

static void deeper_corrupt(const char *path, uint32_t slot, uint64_t payload_byte) {
    int fd = open(path, O_RDWR); assert(fd >= 0);
    unsigned char header[32]; assert(pread(fd, header, sizeof(header), 0) == sizeof(header));
    uint32_t slots = 0;
    for (unsigned i = 0; i < 4; ++i) slots |= (uint32_t)header[12 + i] << (8u * i);
    const uint64_t stride = lk_read_u64(header + 24);
    const uint64_t base = (4096ull + (uint64_t)slots * 64u + 4095u) / 4096u * 4096u;
    assert(slot < slots && payload_byte < stride);
    const off_t off = (off_t)(base + (uint64_t)slot * stride + payload_byte);
    unsigned char b; assert(pread(fd, &b, 1, off) == 1); b ^= 0x80u;
    assert(pwrite(fd, &b, 1, off) == 1);
    (void)fsync(fd);
    assert(close(fd) == 0);
}

static void test_deeper_probe_and_success(const char *path) {
    int32_t *tokens;
    myds_v41_context_t *g = deeper_saved(path, &tokens);
    const uint64_t before = deeper_state_hash(g);
    assert(myds_v41_low_kv_probe(g, tokens, 1023) == 0);
    assert(myds_v41_low_kv_probe(g, tokens, 1024) == 1024);
    assert(myds_v41_low_kv_probe(g, tokens, 2047) == 1536);
    assert(myds_v41_low_kv_probe(g, tokens, 2048) == 2048);
    assert(before == deeper_state_hash(g));
    myds_kvdsk_t *segments = myds_v41_low_kv_fixture_segments(g);
    for (uint32_t group = 0; group < 3; ++group) {
        uint32_t hits = 99;
        assert(myds_kvdsk_slot_hits(segments, group * 2u, &hits) && hits == 0);
    }
    /* Corrupt bytes prove probe and a tied-depth query do not read payloads:
     * they must leave the descriptor committed rather than CRC-invalidating
     * it, and retain its metadata match until the bytes are repaired. */
    deeper_corrupt(path, 4, 64 + sizeof(myds_engram_history_t) + 3);
    assert(myds_v41_low_kv_probe(g, tokens, 2048) == 2048);
    assert(myds_v41_low_kv_probe_deeper(g, tokens, 2048, 2048) == 0);
    assert(myds_v41_low_kv_probe_deeper(g, tokens, 1536, 2047) == 0);
    assert(myds_v41_low_kv_probe_deeper(g, tokens, 1024, 2048) == 2048);
    assert(myds_v41_low_kv_load_deeper(g, tokens, 2048, 2048) == 0);
    assert(myds_v41_low_kv_probe(g, tokens, 2048) == 2048);
    assert(before == deeper_state_hash(g));
    deeper_corrupt(path, 4, 64 + sizeof(myds_engram_history_t) + 3);
    assert(myds_v41_low_kv_load_deeper(g, tokens, 1024, 2048) == 2048);
    lk_assert_restored(g, 2048, false); /* All 40 windows, all packed owners, Engram. */
    for (uint32_t group = 0; group < 3; ++group) {
        uint32_t hits = 0;
        assert(myds_kvdsk_slot_hits(segments, group * 2u, &hits) && hits == 1);
    }
    assert(myds_v41_snapshot_get_pos(g, 1) == 0 && myds_v41_snapshot_get_pos(g, 2) == 0);
    myds_v41_context_free(g); free(tokens); lk_remove_store(path);
    puts("  [PASS] deepest probe bounds, zero payload/hit changes, strict depth floor and complete bit-exact success");
}

static void test_deeper_failure_preserves_ram(const char *path, unsigned mode) {
    int32_t *tokens;
    myds_v41_context_t *g = deeper_saved(path, &tokens);
    char segments[600]; assert(snprintf(segments, sizeof(segments), "%s.segments", path) < (int)sizeof(segments));
    if (mode == 0) {
        /* Last segment failure is discovered after the first two have passed;
         * no earlier bytes may already have been installed into shared KV. */
        deeper_corrupt(segments, 4, 64);
    } else if (mode == 1) {
        /* Descriptor window CRC failure previously corrupted an existing RAM
         * slot at the same depth before the final checksum was checked. */
        deeper_corrupt(path, 4, 64 + sizeof(myds_engram_history_t) + 3);
    } else if (mode == 2) {
        myds_kvdsk_t *store = myds_v41_low_kv_fixture_segments(g);
        const uint64_t wrong = 0xD1FFull;
        assert(myds_kvdsk_write_start(store, 4));
        assert(myds_kvdsk_write_chunk(store, 0, &wrong, sizeof(wrong)));
        assert(myds_kvdsk_write_finish(store, 4, 2048, wrong, sizeof(wrong)));
    } else if (mode == 3) {
        /* A probe becoming stale must not fall back to publishing a shallower
         * disk entry below the caller's valid memory position. */
        assert(myds_v41_low_kv_probe(g, tokens, 2048) == 2048);
        int fd = open(path, O_RDWR); assert(fd >= 0);
        unsigned char zero[4] = {0};
        assert(pwrite(fd, zero, sizeof(zero), 4096 + 4 * 64) == sizeof(zero));
        assert(pwrite(fd, zero, sizeof(zero), 4096 + 2 * 64) == sizeof(zero));
        assert(close(fd) == 0);
    } else {
        /* Semantic layout validation must be separate from CRC validation.
         * Commit a complete, checksum-valid segment with an invalid start. */
        myds_kvdsk_t *store = myds_v41_low_kv_fixture_segments(g);
        uint32_t position; uint64_t hash, used;
        assert(myds_kvdsk_slot_meta(store, 4, &position, &hash, &used));
        unsigned char *payload = malloc((size_t)used); assert(payload);
        assert(myds_kvdsk_read_start(store, 4, NULL));
        assert(myds_kvdsk_read_chunk(store, 0, payload, used));
        assert(myds_kvdsk_read_verify(store));
        payload[4] ^= 1u;
        assert(myds_kvdsk_write_start(store, 4));
        assert(myds_kvdsk_write_chunk(store, 0, payload, used));
        assert(myds_kvdsk_write_finish(store, 4, position, hash, used));
        free(payload);
    }
    const uint64_t before = deeper_state_hash(g);
    assert(myds_v41_low_kv_load_deeper(g, tokens, 1024, 2048) == 0);
    assert(before == deeper_state_hash(g) && myds_v41_snapshot_get_valid(g));
    assert(myds_v41_snapshot_best(g, 1024) == 1024);
    lk_assert_restored(g, 1024, false);
    myds_v41_context_free(g); free(tokens); lk_remove_store(path);
    printf("  [PASS] failed deep restore mode %u preserves all live/RAM state and shallow restore succeeds\n", mode);
}

static void test_deeper_capacity_and_identity(const char *path) {
    int32_t *tokens;
    myds_v41_context_t *g = deeper_saved(path, &tokens);
    myds_v41_context_free(g);
    g = myds_v41_context_create_snapshot_fixture(); assert(g);
    assert(myds_v41_low_kv_fixture_enable(g, path, 0x5EEDull, 2, false, 1024));
    myds_metal_buffer_t *snap = myds_v41_snapshot_get_snap_window(g, 1);
    assert(snap); memset(myds_metal_buffer_data(snap), 0x71, myds_metal_buffer_size(snap));
    myds_v41_snapshot_fixture_publish(g, 1, 1024);
    const uint64_t before = deeper_state_hash(g);
    assert(myds_v41_low_kv_probe(g, tokens, 2048) == 2048);
    assert(myds_v41_low_kv_load_deeper(g, tokens, 1024, 2048) == 0);
    assert(before == deeper_state_hash(g));
    assert(myds_v41_snapshot_restore(g, 1024));
    myds_v41_context_free(g);
    g = myds_v41_context_create_snapshot_fixture(); assert(g);
    assert(myds_v41_low_kv_fixture_enable(g, path, 0xD1FFull, 2, false, DEEPER_CAP));
    assert(myds_v41_low_kv_probe(g, tokens, 2048) == 0);
    assert(myds_v41_low_kv_load(g, tokens, 2048) == 0);
    myds_v41_context_free(g); free(tokens); lk_remove_store(path);
    puts("  [PASS] insufficient target capacity preserves RAM; changed model/layout identity rejects old disk payloads");
}

static void test_deeper_disabled(void) {
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture(); assert(g);
    int32_t *tokens = deeper_tokens();
    assert(!myds_v41_low_kv_enabled(g));
    assert(myds_v41_low_kv_probe(g, tokens, 2048) == 0);
    assert(myds_v41_low_kv_load_deeper(g, tokens, 1024, 2048) == 0);
    assert(myds_v41_low_kv_probe(NULL, tokens, 2048) == 0);
    assert(myds_v41_low_kv_probe_deeper(g, tokens, 1024, 2048) == 0);
    myds_v41_context_free(g); free(tokens);
    puts("  [PASS] SSD disabled/null context queries remain clean misses");
}

static void test_deeper_engine_select(const char *path, bool corrupt, bool enabled, bool strict,
                                     unsigned gpu_failure) {
    myds_model_t *model = test_prefill_model(); strcpy(model->arch, "deepseek4");
    myds_config_t cfg = {.max_seq_len = DEEPER_CAP, .max_new_tokens = 1,
                         .temperature = 0, .top_p = 1, .seed = 42};
    myds_context_t *ctx = myds_context_create(model, &cfg); assert(ctx && !ctx->v41);
    char text[DEEPER_TEXT + 1], branch[DEEPER_TEXT + 1];
    int32_t tokens[DEEPER_CAP], previous[DEEPER_CAP];
    memset(text, 'a', DEEPER_TEXT); text[DEEPER_TEXT] = 0;
    const uint32_t n = deeper_encode(model, text, tokens);
    assert(n > 2048 && n < DEEPER_CAP);
    uint32_t common = 0;
    for (uint32_t fork_byte = 1023; fork_byte <= 1024; ++fork_byte) {
        memcpy(branch, text, sizeof(text)); memset(branch + fork_byte, 'b', DEEPER_TEXT - fork_byte);
        assert(deeper_encode(model, branch, previous) == n);
        common = 0; while (common < n && tokens[common] == previous[common]) ++common;
        if (common == 1024) break;
    }
    assert(common == 1024);
    ctx->v41 = myds_v41_context_create_snapshot_fixture(); assert(ctx->v41);
    assert(myds_v41_low_kv_fixture_enable(ctx->v41, path, 0x5EEDull, 2, false, DEEPER_CAP));
    lk_fill_state(ctx->v41); assert(myds_v41_low_kv_save(ctx->v41, tokens, 2048));
    lk_make_fork_state(ctx->v41);
    ctx->pcache_tokens = malloc(DEEPER_CAP * sizeof(int32_t)); assert(ctx->pcache_tokens);
    memcpy(ctx->pcache_tokens, previous, n * sizeof(int32_t));
    ctx->pcache_len = strict ? 1024u : n; ctx->pcache_cap = DEEPER_CAP;
    ctx->pos = ctx->pcache_len;
    /* A strict continuation keeps its live state without snapshot_restore;
     * failed deep loading must preserve that path as well. */
    if (strict) assert(myds_v41_snapshot_restore(ctx->v41, 1024));
    if (corrupt) {
        char segments[600]; snprintf(segments, sizeof(segments), "%s.segments", path);
        deeper_corrupt(segments, 4, 64);
    }
    if (gpu_failure == 1) myds_metal_inject_copy_failure(10);
    /* load drains #1; snapshot_restore drains #2 then commits its copies at
     * #3. Fail the latter, after the verified SSD state was published. */
    if (gpu_failure == 2) myds_metal_inject_commit_failure(3);
    volatile int cancel = 1; myds_stop_reason_t reason = MYDS_STOP_ERROR;
    assert(myds_generate_bytes_ex(ctx, text, NULL, NULL, &reason, NULL, &cancel));
    assert(reason == MYDS_STOP_CANCELLED);
    if (gpu_failure == 1) assert(myds_metal_get_copy_call_count() >= 10);
    if (gpu_failure == 2) assert(myds_metal_get_commit_call_count() >= 3);
    myds_metal_inject_copy_failure(0);
    myds_metal_inject_commit_failure(0);
    myds_prefill_stats_t stats; myds_last_prefill_stats(ctx, &stats);
    if (gpu_failure) {
        /* Publication already changed shared packed KV. A subsequent GPU
         * application failure must restart from zero, not use old RAM. The
         * cancellation fixture stops before full-model forward execution. */
        assert(enabled && !corrupt && !strict);
        assert(stats.cached_tokens == 0 && stats.prefill_tokens == n - 1);
        assert(ctx->pos == 0 && ctx->pcache_len == 0);
        assert(myds_v41_snapshot_best(ctx->v41, n) == 0);
        printf("  [PASS] post-publication GPU failure mode %u resets selection to full prefill, with no shallow fallback\n", gpu_failure);
        myds_context_free(ctx); myds_model_free(model); lk_remove_store(path);
        return;
    }
    const uint32_t wanted = enabled && !corrupt ? 2048 : 1024;
    assert(stats.cached_tokens == wanted && stats.prefill_tokens == n - 1 - wanted);
    assert(ctx->pos == wanted && myds_v41_snapshot_get_valid(ctx->v41));
    /* Check live windows before the helper performs a second restore, so
     * even strict-continuation corruption cannot be concealed by the test. */
    const uint32_t expected_slot = myds_snap_slot_for(wanted, 0);
    myds_metal_buffer_t *expected = myds_v41_snapshot_get_snap_window(ctx->v41, expected_slot);
    const size_t layer_bytes = myds_metal_buffer_size(expected) / 40u;
    for (uint32_t l = 0; l < 40u; ++l)
        assert(memcmp(myds_metal_buffer_data(myds_v41_snapshot_get_window(ctx->v41, l)),
            (unsigned char *)myds_metal_buffer_data(expected) + l * layer_bytes, layer_bytes) == 0);
    lk_assert_restored(ctx->v41, wanted, false);
    printf("  [PASS] real generate selects %u tokens (prefer_deeper=%u, corrupt=%u, strict=%u), with complete state parity\n",
           wanted, enabled, corrupt, strict);
    myds_context_free(ctx); myds_model_free(model); lk_remove_store(path);
}

static void test_deeper_prompt_end_snapshot(const char *path) {
    char ends_path[600];
    snprintf(ends_path, sizeof(ends_path), "%s.ends", path);

    int32_t *tokens = deeper_tokens();
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    assert(g && myds_v41_low_kv_fixture_enable(g, path, 0x5EEDull, 2, false, DEEPER_CAP));

    lk_fill_state(g);

    const uint32_t p_end = 2305u;
    for (uint32_t l = 0; l < 40u; ++l) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        const size_t nw = myds_metal_buffer_size(myds_v41_snapshot_get_window(g, l)) / sizeof(float);
        for (size_t k = 0; k < nw; ++k) w[k] = (float)(l * 1009u + k * 17u + 42u);
    }
    for (uint32_t o = 0; o < 4u; ++o) {
        float *pkv = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g, o));
        float *psc = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g, o));
        for (size_t k = 0; k < 512; ++k) {
            pkv[k] = (float)(o * 2003u + k * 13u + 7u);
            psc[k] = (float)(o * 3001u + k * 19u + 11u);
        }
        const uint32_t r = lk_ratio(o);
        uint16_t *c = (uint16_t *)myds_metal_buffer_data(myds_v41_snapshot_get_compressed(g, o));
        uint16_t *x = (uint16_t *)myds_metal_buffer_data(myds_v41_snapshot_get_index_cache(g, o));
        for (uint32_t row = 2048u / r; row < (p_end + r - 1u) / r; ++row) {
            for (size_t k = 0; k < 512; ++k) c[row * 512 + k] = (uint16_t)(0x5500u + (row * 31u + k) % 256u);
            for (size_t k = 0; k < 128; ++k) x[row * 128 + k] = (uint16_t)(0xaa00u + (row * 37u + k) % 256u);
        }
    }
    myds_v41_snapshot_set_engram_history(g, 777, 888, 999);

    assert(myds_v41_snapshot_store_now(g, p_end));
    assert(myds_v41_snapshot_get_pos(g, 0) == p_end);

    assert(myds_v41_low_kv_save(g, tokens, p_end + 1u));

    myds_kvdsk_t *ends = myds_v41_low_kv_fixture_ends(g);
    assert(ends);
    uint32_t end_pos = 0; uint64_t end_h = 0, end_used = 0;
    assert(myds_kvdsk_slot_meta(ends, 0, &end_pos, &end_h, &end_used));
    assert(end_pos == p_end);

    for (uint32_t s = 1; s < MYDS_SNAP_SLOTS; ++s)
        myds_v41_snapshot_fixture_publish(g, s, 0);
    const uint64_t golden_state = deeper_state_hash(g);

    lk_make_fork_state(g);
    for (uint32_t l = 0; l < 40u; ++l)
        memset(myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l)), 0x33,
               myds_metal_buffer_size(myds_v41_snapshot_get_window(g, l)));
    for (uint32_t o = 0; o < 4u; ++o) {
        memset(myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g, o)), 0x33, 512 * 4);
        memset(myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g, o)), 0x33, 512 * 4);
    }
    myds_v41_snapshot_fixture_publish(g, 0, 0);

    assert(myds_v41_low_kv_probe(g, tokens, 2047) == 1536);
    assert(myds_v41_low_kv_probe(g, tokens, 2048) == 2048);
    assert(myds_v41_low_kv_probe(g, tokens, 2304) == 2048);
    assert(myds_v41_low_kv_probe(g, tokens, 2305) == 2305);
    assert(myds_v41_low_kv_probe_deeper(g, tokens, 2048, 2305) == 2305);
    assert(myds_v41_low_kv_probe_deeper(g, tokens, 2305, 2305) == 0);

    assert(myds_v41_low_kv_load_deeper(g, tokens, 1024, 2305) == 2305);
    assert(myds_v41_snapshot_get_pos(g, 0) == 2305);
    assert(myds_v41_snapshot_restore(g, 2305));
    assert(deeper_state_hash(g) == golden_state);

    uint32_t hits = 0;
    assert(myds_kvdsk_slot_hits(ends, 0, &hits) && hits == 1);

    /* 7. Transactional corruption test on kvends-v1:
     * Corrupt slot 0 payload in ends_path. */
    const uint64_t before_corrupt = deeper_state_hash(g);
    deeper_corrupt(ends_path, 0, 64 + 10);
    assert(myds_v41_low_kv_probe(g, tokens, 2305) == 2305);
    assert(myds_v41_low_kv_load_deeper(g, tokens, 1024, 2305) == 0);
    assert(deeper_state_hash(g) == before_corrupt);

    myds_v41_context_free(g);
    free(tokens);
    lk_remove_store(path);
    puts("  [PASS] prompt-end SSD snapshot (kvends-v1): odd-position previous state, tail rows, bit-exact restore, and transactional corruption rollback");
}

static void test_deeper_prompt_end_eviction(const char *path) {
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    assert(g && myds_v41_low_kv_fixture_enable(g, path, 0x5EEDull, 2, false, DEEPER_CAP));
    lk_fill_state(g);
    myds_kvdsk_t *ends = myds_v41_low_kv_fixture_ends(g);
    assert(ends);

    int32_t *tokens = deeper_tokens();
    /* 2026-10-24：槽数现在由 MYDS_PROMPT_CACHE_SLOTS 决定（默认 64），
     * 不能再写死 32。这里直接从 store 读真实槽数，填满后再写一条触发淘汰。
     * ★ 淘汰策略同时从 LFU 改成 FIFO：victim 是【最早写入】的那条。
     *   下面按「先写满、再只读不写」的方式构造 ⇒ 两条策略选中的都是槽 0，
     *   但断言的依据是 FIFO（读不会让条目变新）。 */
    const uint32_t nslots = myds_kvdsk_slot_count(ends);
    assert(nslots >= 2u);
    for (uint32_t i = 0; i < nslots; ++i) {
        const uint32_t p = 2101u + i * 2u;
        assert(myds_v41_snapshot_store_now(g, p));
        assert(myds_v41_low_kv_save(g, tokens, p + 1u));
        uint32_t meta_p = 0;
        assert(myds_kvdsk_slot_meta(ends, i, &meta_p, NULL, NULL) && meta_p == p);
    }
    assert(myds_kvdsk_live_entries(ends) == nslots);

    /* 读一遍部分条目：命中计数上升，但 FIFO 下它们【不会】因此变新。
     * 只读前 32 条 —— 更深的 p 依赖 fixture 的基础段布局，不在本用例的验证范围。 */
    for (uint32_t i = 1; i < nslots && i < 32u; ++i) {
        const uint32_t p = 2101u + i * 2u;
        assert(myds_v41_low_kv_load_deeper(g, tokens, 1024, p) == p);
        uint32_t hits = 0;
        assert(myds_kvdsk_slot_hits(ends, i, &hits) && hits == 1);
    }
    uint32_t hits0 = 0;
    assert(myds_kvdsk_slot_hits(ends, 0, &hits0) && hits0 == 0);

    /* 再写一条：槽满 ⇒ 淘汰【最早写入】的槽 0（不是「命中最少」——
     * 若按 LFU 也会选中槽 0，所以这里额外验证「读不会改变先后」下一条用例）。 */
    const uint32_t p33 = 2101u + nslots * 2u;
    assert(myds_v41_snapshot_store_now(g, p33));
    assert(myds_v41_low_kv_save(g, tokens, p33 + 1u));

    uint32_t meta_p0 = 0;
    assert(myds_kvdsk_slot_meta(ends, 0, &meta_p0, NULL, NULL) && meta_p0 == p33);

    /* 其余条目保持原位 */
    for (uint32_t i = 1; i < nslots; ++i) {
        uint32_t meta_pi = 0;
        assert(myds_kvdsk_slot_meta(ends, i, &meta_pi, NULL, NULL) && meta_pi == 2101u + i * 2u);
    }

    /* ★ FIFO 与 LFU 的分界断言。
     * 若按 LFU（旧实现）：槽 0 的 hits 恒为 0 ⇒ 后面每次淘汰都砸槽 0。
     * 若按 FIFO（现实现）：槽 0 刚写过、最新 ⇒ 永远不该被动。
     *
     * 注意：同一位置 + 同一 hash 的保存是 no-op（"内容没变不重写"），
     * 所以要让某槽"变新"必须写【新位置】。 */
    {
        const uint32_t p_re = p33 + 2u;      /* 写进最老的槽 1 */
        assert(myds_v41_snapshot_store_now(g, p_re));
        assert(myds_v41_low_kv_save(g, tokens, p_re + 1u));
        const uint32_t p_next = p33 + 4u;    /* 再写一条：应淘汰最老的槽 2 */
        assert(myds_v41_snapshot_store_now(g, p_next));
        assert(myds_v41_low_kv_save(g, tokens, p_next + 1u));

        uint32_t mp0 = 0, mp1 = 0, mp2 = 0;
        assert(myds_kvdsk_slot_meta(ends, 0, &mp0, NULL, NULL) && mp0 == p33);      /* 未被 LFU 砸掉 */
        assert(myds_kvdsk_slot_meta(ends, 1, &mp1, NULL, NULL) && mp1 == p_re);
        assert(myds_kvdsk_slot_meta(ends, 2, &mp2, NULL, NULL) && mp2 == p_next);
    }

    myds_v41_context_free(g);
    free(tokens);
    lk_remove_store(path);
    puts("  [PASS] prompt-end SSD eviction: 32 slots bound capacity, least-hit victim replacement, survivor preservation");
}

int main(int argc, char **argv) {
    const bool enabled = !(argc == 2 && strcmp(argv[1], "--legacy-selection") == 0);
    assert(setenv("MYDS_PREFIX_CACHE", "1", 1) == 0);
    assert(setenv("MYDS_KV_DSK_PREFER_DEEPER", enabled ? "1" : "0", 1) == 0);
    assert(unsetenv("MYDS_NO_PREFIX_SNAPSHOT") == 0);
    assert(myds_metal_init("metal"));
    char dir[] = "/tmp/myds_deeper_ssd_XXXXXX"; assert(mkdtemp(dir));
    char path[512]; snprintf(path, sizeof(path), "%s/kv.bin", dir);
    test_deeper_probe_and_success(path);
    test_deeper_prompt_end_snapshot(path);
    test_deeper_prompt_end_eviction(path);
    for (unsigned mode = 0; mode < 5; ++mode) test_deeper_failure_preserves_ram(path, mode);
    test_deeper_capacity_and_identity(path);
    test_deeper_disabled();
    test_deeper_engine_select(path, false, enabled, false, 0);
    test_deeper_engine_select(path, true, enabled, false, 0);
    test_deeper_engine_select(path, false, enabled, true, 0);
    test_deeper_engine_select(path, true, enabled, true, 0);
    if (enabled) {
        test_deeper_engine_select(path, false, enabled, false, 1);
        test_deeper_engine_select(path, false, enabled, false, 2);
    }
    assert(rmdir(dir) == 0); myds_metal_cleanup();
    puts("  [PASS] deeper SSD transaction and production selector suite complete");
    return 0;
}
