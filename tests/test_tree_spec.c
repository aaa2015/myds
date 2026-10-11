/* Model-free public guards and snapshot restoration. No target model is
 * loaded or evaluated. Host-written state fixtures do not prove tree branch
 * arbitration, compressor isolation, target arithmetic or performance. */
#define _POSIX_C_SOURCE 200809L
#include "myds.h"
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_metal.h"
#include "myds_dspark.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { \
    if (!(c)) { \
        fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
        return 0; \
    } \
} while (0)

enum { WINDOW = 128, ROW = 512, CACHE_CAP = 256, VOCAB = 8 };

static void fill_floats(myds_metal_buffer_t *buffer, unsigned seed) {
    float *p = myds_metal_buffer_data(buffer);
    for (size_t i = 0; i < myds_metal_buffer_size(buffer) / sizeof(float); ++i)
        p[i] = ((int)((i * 13u + seed) % 113u) - 56) / 64.0f;
}
static void fill_windows(myds_v41_context_t *g) {
    for (unsigned l = 0; l < MYDS_V41_LAYERS_COUNT; ++l)
        fill_floats(myds_v41_snapshot_get_window(g, l), l + 1u);
}
static void write_window_row(myds_v41_context_t *g, unsigned pos, float value) {
    for (unsigned l = 0; l < MYDS_V41_LAYERS_COUNT; ++l) {
        float *w = myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        for (unsigned d = 0; d < ROW; ++d)
            w[(pos % WINDOW) * ROW + d] = value + (float)l / 64.0f;
    }
}
static uint64_t state_buffer_hash(myds_v41_context_t *g) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (unsigned i = 0; i < MYDS_V41_LAYERS_COUNT + 8u; ++i) {
        myds_metal_buffer_t *b = i < MYDS_V41_LAYERS_COUNT
            ? myds_v41_snapshot_get_window(g, i)
            : (i < MYDS_V41_LAYERS_COUNT + 4u
                ? myds_v41_snapshot_get_previous_kv(g, i - MYDS_V41_LAYERS_COUNT)
                : myds_v41_snapshot_get_previous_score(g, i - MYDS_V41_LAYERS_COUNT - 4u));
        const unsigned char *p = myds_metal_buffer_data(b);
        for (size_t n = 0; n < myds_metal_buffer_size(b); ++n)
            hash = (hash ^ p[n]) * UINT64_C(1099511628211);
    }
    return hash;
}

/* Reject-before-mutation, including capture and output sentinels. This fixture
 * has no graph weights and is used only for invalid public calls. */
static int rejected_without_mutation_mode(myds_context_t *ctx, const int32_t *tokens,
    uint32_t pos, int have_out, bool linear, uint32_t count) {
    myds_context_t before;
    myds_dspark_t ds_before;
    int32_t cache_before[CACHE_CAP], out[3] = {123456789, -987654321, 111222333};
    const int32_t out_before[3] = {out[0], out[1], out[2]};
    float logits_before[VOCAB];
    const int barrier = myds_metal_get_auto_barrier();
    const uint64_t dispatches = myds_metal_dispatch_count();
    uint32_t next = 0, engram_pos = 0;
    bool valid = false;
    uint64_t buffers = 0;
    if (ctx) {
        memcpy(&before, ctx, sizeof(before));
        if (ctx->dspark) memcpy(&ds_before, ctx->dspark, sizeof(ds_before));
        if (ctx->pcache_tokens) memcpy(cache_before, ctx->pcache_tokens, sizeof(cache_before));
        if (ctx->logits) memcpy(logits_before, myds_metal_buffer_data(ctx->logits), sizeof(logits_before));
        if (ctx->v41) {
            next = myds_v41_snapshot_get_next_position(ctx->v41);
            engram_pos = myds_v41_snapshot_get_engram_position(ctx->v41);
            valid = myds_v41_snapshot_get_valid(ctx->v41);
            buffers = state_buffer_hash(ctx->v41);
        }
    }
    if (linear) {
        myds_v41_spec_snapshot_t *snap = calloc(1, sizeof(*snap));
        CHECK(snap);
        snap->valid = snap->first_state_valid = snap->second_state_valid = snap->linear_retry_valid = true;
        CHECK(myds_v41_forward_linear_spec_ex(ctx, tokens, count, pos, have_out ? out : NULL, snap) == -1);
        CHECK(!snap->valid && !snap->first_state_valid && !snap->second_state_valid && !snap->linear_retry_valid);
        CHECK(!myds_v41_linear_spec_retry_ready(ctx, snap));
        free(snap);
    } else CHECK(myds_v41_forward_tree_spec(ctx, tokens, pos, have_out ? out : NULL) == -1);
    CHECK(!memcmp(out, out_before, sizeof(out)));
    CHECK(myds_metal_dispatch_count() == dispatches);
    CHECK(myds_metal_get_auto_barrier() == barrier);
    if (ctx) {
        CHECK(!memcmp(ctx, &before, sizeof(before)));
        if (ctx->dspark) CHECK(!memcmp(ctx->dspark, &ds_before, sizeof(ds_before)));
        if (ctx->pcache_tokens) CHECK(!memcmp(ctx->pcache_tokens, cache_before, sizeof(cache_before)));
        if (ctx->logits) CHECK(!memcmp(myds_metal_buffer_data(ctx->logits), logits_before, sizeof(logits_before)));
        if (ctx->v41) {
            CHECK(myds_v41_snapshot_get_next_position(ctx->v41) == next);
            CHECK(myds_v41_snapshot_get_engram_position(ctx->v41) == engram_pos);
            CHECK(myds_v41_snapshot_get_valid(ctx->v41) == valid);
            CHECK(state_buffer_hash(ctx->v41) == buffers);
        }
    }
    return 1;
}

static int rejected_without_mutation(myds_context_t *ctx, const int32_t *tokens,
    uint32_t pos, int have_out) {
    return rejected_without_mutation_mode(ctx, tokens, pos, have_out, false, 3u);
}

/* This model-free capacity0 fixture proves public rejection is atomic. It
 * cannot prove which overlapping invalid precondition triggered the guard. */
static int test_public_linear_invalid_calls(void) {
    puts("[TEST] Public continuous depth2 invalid calls preserve caller state");
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    CHECK(g);
    myds_model_t model = {.vocab_size = VOCAB};
    model.v41 = (myds_v41_model_t *)(void *)&model;
    int32_t cache[CACHE_CAP] = {0};
    myds_dspark_t ds = {.capture_configured = true, .capture_bypass_mode = true};
    myds_dspark_capture_publish(&ds);
    myds_context_t ctx = {.model=&model, .v41=g, .dspark=&ds, .pos=7,
        .pcache_tokens=cache, .pcache_cap=CACHE_CAP, .pcache_len=7,
        .rng_state=UINT64_C(0x123456789abcdef0)};
    ctx.cfg.max_seq_len = CACHE_CAP;
    ctx.logits = myds_metal_buffer_alloc(VOCAB * sizeof(float));
    CHECK(ctx.logits);
    fill_floats(ctx.logits, 7); fill_windows(g);
    for (unsigned o=0;o<4;++o) {
        fill_floats(myds_v41_snapshot_get_previous_kv(g,o),11u+o);
        fill_floats(myds_v41_snapshot_get_previous_score(g,o),17u+o);
    }
    myds_v41_snapshot_set_positions(g,7,7);
    const int32_t tokens[3]={1,2,3};
    const int saved_barrier=myds_metal_get_auto_barrier();
#define REJECT_LINEAR(c,t,p,o,n) CHECK(rejected_without_mutation_mode(c,t,p,o,true,n))
    for(int barrier=0;barrier<=1;++barrier) {
        myds_metal_set_auto_barrier(barrier);
        REJECT_LINEAR(NULL,tokens,7,1,3);
        REJECT_LINEAR(&ctx,NULL,7,1,3);
        REJECT_LINEAR(&ctx,tokens,7,0,3);
        for(uint32_t n=0;n<=4;++n) if(n!=3) REJECT_LINEAR(&ctx,tokens,7,1,n);
        ctx.model=NULL; REJECT_LINEAR(&ctx,tokens,7,1,3);ctx.model=&model;
        model.v41=NULL;REJECT_LINEAR(&ctx,tokens,7,1,3);model.v41=(myds_v41_model_t *)(void *)&model;
        ctx.v41=NULL;REJECT_LINEAR(&ctx,tokens,7,1,3);ctx.v41=g;
        for(unsigned row=0;row<3;++row) {
            int32_t bad[3]={1,2,3};bad[row]=-1;REJECT_LINEAR(&ctx,bad,7,1,3);
            bad[row]=VOCAB;REJECT_LINEAR(&ctx,bad,7,1,3);
        }
        ctx.cfg.temperature=.7f;REJECT_LINEAR(&ctx,tokens,7,1,3);
        ctx.cfg.temperature=NAN;REJECT_LINEAR(&ctx,tokens,7,1,3);ctx.cfg.temperature=0;
        ctx.img_emb_n=1;REJECT_LINEAR(&ctx,tokens,7,1,3);ctx.img_emb_n=0;
        ctx.cfg.max_seq_len=9;REJECT_LINEAR(&ctx,tokens,7,1,3);ctx.cfg.max_seq_len=CACHE_CAP;
        REJECT_LINEAR(&ctx,tokens,6,1,3);REJECT_LINEAR(&ctx,tokens,UINT32_MAX,1,3);
        myds_v41_snapshot_set_positions(g,7,8);REJECT_LINEAR(&ctx,tokens,7,1,3);
        myds_v41_snapshot_set_positions(g,7,7);
        ds.capture_engine_only=true;REJECT_LINEAR(&ctx,tokens,7,1,3);ds.capture_engine_only=false;
    }
#undef REJECT_LINEAR
    myds_metal_set_auto_barrier(saved_barrier);
    myds_metal_buffer_free(ctx.logits);myds_v41_context_free(g);
    return 1;
}

static int test_public_tree_invalid_calls(void) {
    puts("[TEST] Public tree invalid calls preserve caller state");
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    CHECK(g);
    /* A non-NULL opaque marker is never dereferenced: all calls are invalid
     * before graph access. This is not a usable target model. */
    myds_model_t model = {.vocab_size = VOCAB};
    model.v41 = (myds_v41_model_t *)(void *)&model;
    int32_t cache[CACHE_CAP] = {0};
    myds_dspark_t ds = {.capture_configured = true, .capture_bypass_mode = true};
    myds_dspark_capture_publish(&ds);
    myds_context_t ctx = {.model = &model, .v41 = g, .dspark = &ds, .pos = 7,
        .pcache_tokens = cache, .pcache_cap = CACHE_CAP, .pcache_len = 7,
        .rng_state = UINT64_C(0x123456789abcdef0)};
    ctx.cfg.max_seq_len = CACHE_CAP;
    ctx.logits = myds_metal_buffer_alloc(VOCAB * sizeof(float));
    CHECK(ctx.logits);
    fill_floats(ctx.logits, 7);
    fill_windows(g);
    for (unsigned o = 0; o < 4; ++o) {
        fill_floats(myds_v41_snapshot_get_previous_kv(g, o), 11u + o);
        fill_floats(myds_v41_snapshot_get_previous_score(g, o), 17u + o);
    }
    myds_v41_snapshot_set_positions(g, 7, 7);
    const int32_t tokens[3] = {1, 2, 3};
    const int saved_barrier = myds_metal_get_auto_barrier();
    for (int barrier = 0; barrier <= 1; ++barrier) {
        myds_metal_set_auto_barrier(barrier);
        CHECK(rejected_without_mutation(NULL, tokens, 7, 1));
        CHECK(rejected_without_mutation(&ctx, NULL, 7, 1));
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 0));
        ctx.model = NULL;
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        ctx.model = &model;
        model.v41 = NULL;
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        model.v41 = (myds_v41_model_t *)(void *)&model;
        ctx.v41 = NULL;
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        ctx.v41 = g;
        for (unsigned slot = 0; slot < 3; ++slot) {
            int32_t bad[3] = {1, 2, 3};
            bad[slot] = -1;
            CHECK(rejected_without_mutation(&ctx, bad, 7, 1));
            bad[slot] = VOCAB;
            CHECK(rejected_without_mutation(&ctx, bad, 7, 1));
        }
        /* Snapshot fixture capacity is zero: these prove rejection and
         * atomicity, not independent coverage of the internal guard branches. */
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        ctx.cfg.max_seq_len = 8; /* only one position remains */
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        ctx.cfg.max_seq_len = 0;
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        ctx.cfg.max_seq_len = CACHE_CAP;
        CHECK(rejected_without_mutation(&ctx, tokens, 6, 1));
        CHECK(rejected_without_mutation(&ctx, tokens, UINT32_MAX, 1));
        myds_v41_snapshot_set_positions(g, 7, 8);
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        myds_v41_snapshot_set_positions(g, 7, 7);
        ds.capture_engine_only = true;
        CHECK(rejected_without_mutation(&ctx, tokens, 7, 1));
        ds.capture_engine_only = false;
    }
    myds_metal_set_auto_barrier(saved_barrier);
    myds_metal_buffer_free(ctx.logits);
    myds_v41_context_free(g);
    return 1;
}

static int test_host_window_partial_restore(uint32_t pos) {
    printf("[TEST] Host-written window snapshot partial1 at %u\n", pos);
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    myds_v41_spec_snapshot_t *snap = calloc(1, sizeof(*snap));
    CHECK(g && snap);
    int32_t cache[CACHE_CAP] = {0};
    myds_context_t ctx = {.v41 = g, .pos = pos, .pcache_tokens = cache,
        .pcache_cap = CACHE_CAP, .pcache_len = pos, .rng_state = 1234};
    fill_windows(g);
    myds_v41_snapshot_set_positions(g, pos, pos);
    CHECK(myds_v41_spec_snapshot_save(&ctx, snap, 2));
    CHECK(snap->valid && !snap->first_state_valid && snap->window_count == 2);
    write_window_row(g, pos, 71.0f);
    write_window_row(g, pos + 1, 89.0f);
    ctx.pos = pos + 2; ctx.pcache_len = pos + 2;
    cache[pos] = 3; cache[pos + 1] = 5;
    myds_v41_snapshot_set_positions(g, pos + 2, pos + 2);
    CHECK(myds_v41_spec_snapshot_restore_partial(&ctx, snap, 1));
    CHECK(ctx.pos == pos + 1 && ctx.pcache_len == pos + 1);
    CHECK(myds_v41_snapshot_get_next_position(g) == pos + 1);
    for (unsigned l = 0; l < MYDS_V41_LAYERS_COUNT; ++l) {
        const float *w = myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        CHECK(!memcmp(w + ((pos + 1) % WINDOW) * ROW, snap->window_saved[l][1], ROW * sizeof(float)));
        for (unsigned d = 0; d < ROW; ++d)
            CHECK(w[(pos % WINDOW) * ROW + d] == 71.0f + (float)l / 64.0f);
    }
    CHECK(myds_v41_spec_snapshot_restore(&ctx, snap));
    CHECK(ctx.pos == pos && ctx.pcache_len == pos && ctx.rng_state == 1234);
    for (unsigned l = 0; l < MYDS_V41_LAYERS_COUNT; ++l) {
        const float *w = myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        for (unsigned k = 0; k < 2; ++k)
            CHECK(!memcmp(w + ((pos + k) % WINDOW) * ROW, snap->window_saved[l][k], ROW * sizeof(float)));
    }
    free(snap);
    myds_v41_context_free(g);
    return 1;
}

/* The production first-checkpoint producer creates its own borrowed views.
 * Host values represent a completed first step, not an actual target forward.
 * The retain-one API is also used for EOS/cancel after a verifier hit; this
 * fixture tests restoration, not the engine's EOS/cancel decision. */
static int test_host_first_checkpoint_restore(uint32_t pos) {
    printf("[TEST] First checkpoint RNG/carry/logits/Engram/capture at %u\n", pos);
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    myds_v41_spec_snapshot_t *snap = calloc(1, sizeof(*snap));
    CHECK(g && snap);
    myds_model_t model = {.vocab_size = VOCAB};
    int32_t cache[CACHE_CAP] = {0};
    myds_dspark_t ds = {.capture_configured = true, .capture_bypass_mode = true};
    myds_context_t ctx = {.model = &model, .v41 = g, .dspark = &ds, .pos = pos,
        .pcache_tokens = cache, .pcache_cap = CACHE_CAP, .pcache_len = pos,
        .rng_state = UINT64_C(0xabcdef1234567890)};
    ctx.cfg.temperature = 0.7f;
    ctx.logits = myds_metal_buffer_alloc(VOCAB * sizeof(float));
    CHECK(ctx.logits);
    for (unsigned s = 0; s < 3; ++s) {
        ds.target_slots[s] = myds_metal_buffer_alloc(MYDS_DSPARK_DIM * sizeof(float));
        CHECK(ds.target_slots[s]);
        fill_floats(ds.target_slots[s], 19u + s);
    }
    fill_windows(g);
    for (unsigned o = 0; o < 4; ++o) {
        fill_floats(myds_v41_snapshot_get_previous_kv(g, o), 7u + o);
        fill_floats(myds_v41_snapshot_get_previous_score(g, o), 11u + o);
    }
    myds_v41_snapshot_set_engram_history(g, 31, 37, 41);
    myds_v41_snapshot_set_positions(g, pos, pos);
    CHECK(myds_v41_spec_snapshot_save(&ctx, snap, 2));
    const uint64_t first_rng = UINT64_C(0x778899aabbccddee);
    CHECK(first_rng != snap->rng_state * UINT64_C(6364136223846793005) + 1);
    ctx.pos = pos + 1; ctx.pcache_len = pos + 1; ctx.rng_state = first_rng;
    cache[pos] = 3;
    write_window_row(g, pos, 73.0f);
    fill_floats(ctx.logits, 23);
    for (unsigned o = 0; o < 4; ++o) {
        fill_floats(myds_v41_snapshot_get_previous_kv(g, o), 43u + o);
        fill_floats(myds_v41_snapshot_get_previous_score(g, o), 53u + o);
    }
    myds_v41_snapshot_set_engram_history(g, 47, 53, 59);
    myds_v41_snapshot_set_positions(g, pos + 1, pos + 1);
    float first_logits[VOCAB], first_kv[4][ROW], first_score[4][ROW];
    float first_slots[3][MYDS_DSPARK_DIM];
    memcpy(first_logits, myds_metal_buffer_data(ctx.logits), sizeof(first_logits));
    for (unsigned o = 0; o < 4; ++o) {
        memcpy(first_kv[o], myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g, o)), sizeof(first_kv[o]));
        memcpy(first_score[o], myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g, o)), sizeof(first_score[o]));
    }
    for (unsigned s = 0; s < 3; ++s)
        memcpy(first_slots[s], myds_metal_buffer_data(ds.target_slots[s]), sizeof(first_slots[s]));
    myds_dspark_capture_publish(&ds);
    CHECK(myds_dspark_capture_ready(&ds));
    CHECK(myds_v41_spec_snapshot_capture_first(&ctx, snap));
    CHECK(snap->first_state_valid && snap->first_rng_state == first_rng);
    CHECK(snap->first_logits && snap->first_residual && snap->first_pre);
    for (unsigned s = 0; s < 3; ++s) CHECK(snap->first_dspark_slots[s]);
    const uint64_t dispatches = myds_metal_dispatch_count();
    const int saved_barrier = myds_metal_get_auto_barrier();
    for (int barrier = 0; barrier <= 1; ++barrier) {
        myds_metal_set_auto_barrier(barrier);
        ctx.pos = pos + 2; ctx.pcache_len = pos + 2; ctx.rng_state = 9;
        cache[pos + 1] = 5;
        write_window_row(g, pos + 1, 97.0f);
        fill_floats(ctx.logits, 67);
        for (unsigned o = 0; o < 4; ++o) {
            fill_floats(myds_v41_snapshot_get_previous_kv(g, o), 71u + o);
            fill_floats(myds_v41_snapshot_get_previous_score(g, o), 79u + o);
        }
        for (unsigned s = 0; s < 3; ++s) fill_floats(ds.target_slots[s], 83u + s);
        myds_dspark_capture_invalidate(&ds);
        myds_v41_snapshot_set_engram_history(g, 61, 67, 71);
        myds_v41_snapshot_set_positions(g, pos + 2, pos + 2);
        /* Detect accidental borrowed-view aliasing before restoring: the
         * completed first checkpoint must survive later live-buffer writes. */
        CHECK(!memcmp(myds_metal_buffer_data(snap->first_logits), first_logits, sizeof(first_logits)));
        for (unsigned o = 0; o < 4; ++o) {
            CHECK(!memcmp(myds_metal_buffer_data(snap->first_previous_kv[o]), first_kv[o], sizeof(first_kv[o])));
            CHECK(!memcmp(myds_metal_buffer_data(snap->first_previous_score[o]), first_score[o], sizeof(first_score[o])));
        }
        for (unsigned s = 0; s < 3; ++s)
            CHECK(!memcmp(myds_metal_buffer_data(snap->first_dspark_slots[s]), first_slots[s], sizeof(first_slots[s])));
        CHECK(myds_v41_spec_snapshot_restore_partial(&ctx, snap, 1));
        CHECK(ctx.pos == pos + 1 && ctx.pcache_len == pos + 1 && ctx.rng_state == first_rng);
        CHECK(myds_v41_snapshot_get_next_position(g) == pos + 1);
        CHECK(myds_v41_snapshot_get_engram_position(g) == pos + 1);
        uint32_t history[3];
        const uint32_t expected_history[3] = {47, 53, 59};
        myds_v41_snapshot_get_engram_history(g, history);
        CHECK(!memcmp(history, expected_history, sizeof(history)));
        CHECK(!memcmp(myds_metal_buffer_data(ctx.logits), first_logits, sizeof(first_logits)));
        for (unsigned o = 0; o < 4; ++o) {
            CHECK(!memcmp(myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g, o)),
                          first_kv[o], sizeof(first_kv[o])));
            CHECK(!memcmp(myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g, o)),
                          first_score[o], sizeof(first_score[o])));
        }
        CHECK(myds_dspark_capture_ready(&ds));
        for (unsigned s = 0; s < 3; ++s)
            CHECK(!memcmp(myds_metal_buffer_data(ds.target_slots[s]),
                          first_slots[s], sizeof(first_slots[s])));
        for (unsigned l = 0; l < MYDS_V41_LAYERS_COUNT; ++l) {
            const float *w = myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
            CHECK(!memcmp(w + ((pos + 1) % WINDOW) * ROW, snap->window_saved[l][1], ROW * sizeof(float)));
            for (unsigned d = 0; d < ROW; ++d)
                CHECK(w[(pos % WINDOW) * ROW + d] == 73.0f + (float)l / 64.0f);
        }
        CHECK(myds_metal_get_auto_barrier() == barrier);
    }
    myds_metal_set_auto_barrier(saved_barrier);
    CHECK(myds_metal_dispatch_count() == dispatches);
    /* Base snapshot lacks pre-step hidden capture payloads. Whole rollback
     * must invalidate capture, while restoring the base fields it stores. */
    CHECK(myds_v41_spec_snapshot_restore(&ctx, snap));
    CHECK(ctx.pos == pos && ctx.pcache_len == pos && ctx.rng_state == snap->rng_state);
    CHECK(!myds_dspark_capture_ready(&ds));
    CHECK(myds_v41_snapshot_get_engram_position(g) == pos);
    uint32_t history[3];
    const uint32_t base_history[3] = {31, 37, 41};
    myds_v41_snapshot_get_engram_history(g, history);
    CHECK(!memcmp(history, base_history, sizeof(history)));
    for (unsigned o = 0; o < 4; ++o) {
        CHECK(!memcmp(myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g, o)), snap->previous_kv[o], ROW * sizeof(float)));
        CHECK(!memcmp(myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g, o)), snap->previous_score[o], ROW * sizeof(float)));
    }
    for (unsigned s = 0; s < 3; ++s) myds_metal_buffer_free(ds.target_slots[s]);
    myds_metal_buffer_free(ctx.logits);
    free(snap);
    myds_v41_context_free(g);
    return 1;
}

/* Completed first/second steps are host-written, then captured by the real
 * checkpoint producer. This checks serialization, borrowed-view independence
 * and partial1/2 restoration across ring wrap; it does not evaluate a model. */
static int test_host_second_checkpoint_restore(uint32_t pos) {
    printf("[TEST] Independent first/second checkpoint and partial2 at %u\n",pos);
    myds_v41_context_t *g=myds_v41_context_create_snapshot_fixture();
    myds_v41_spec_snapshot_t *snap=calloc(1,sizeof(*snap));
    CHECK(g&&snap);
    myds_model_t model={.vocab_size=VOCAB};
    int32_t cache[CACHE_CAP]={0};
    myds_dspark_t ds={.capture_configured=true,.capture_bypass_mode=true};
    myds_context_t ctx={.model=&model,.v41=g,.dspark=&ds,.pos=pos,
        .pcache_tokens=cache,.pcache_cap=CACHE_CAP,.pcache_len=pos,.rng_state=1234};
    ctx.cfg.temperature=.7f;
    ctx.logits=myds_metal_buffer_alloc(VOCAB*sizeof(float));CHECK(ctx.logits);
    for(unsigned s=0;s<3;++s) {
        ds.target_slots[s]=myds_metal_buffer_alloc(MYDS_DSPARK_DIM*sizeof(float));
        CHECK(ds.target_slots[s]);fill_floats(ds.target_slots[s],19u+s);
    }
    fill_windows(g);myds_v41_snapshot_set_positions(g,pos,pos);
    myds_v41_snapshot_set_engram_history(g,31,37,41);
    for(unsigned o=0;o<4;++o) {
        fill_floats(myds_v41_snapshot_get_previous_kv(g,o),7u+o);
        fill_floats(myds_v41_snapshot_get_previous_score(g,o),11u+o);
    }
    CHECK(myds_v41_spec_snapshot_save(&ctx,snap,3));
    CHECK(!snap->first_state_valid&&!snap->second_state_valid&&!snap->linear_retry_valid);
    const uint64_t expected_rng[2]={UINT64_C(0x123456789abcdef0),UINT64_C(0xfedcba9876543210)};
    float expected_logits[2][VOCAB],expected_kv[2][4][ROW],expected_score[2][4][ROW];
    float expected_slots[2][3][MYDS_DSPARK_DIM];
    for(unsigned step=1;step<=2;++step) {
        ctx.pos=pos+step;ctx.pcache_len=pos+step;ctx.rng_state=expected_rng[step-1];
        cache[pos+step-1]=(int32_t)(step+1);
        write_window_row(g,pos+step-1,71.0f+(float)step);
        fill_floats(ctx.logits,23u+step);
        memcpy(expected_logits[step-1],myds_metal_buffer_data(ctx.logits),sizeof(expected_logits[0]));
        for(unsigned o=0;o<4;++o) {
            fill_floats(myds_v41_snapshot_get_previous_kv(g,o),43u+o+step);
            fill_floats(myds_v41_snapshot_get_previous_score(g,o),53u+o+step);
            memcpy(expected_kv[step-1][o],myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g,o)),sizeof(expected_kv[0][0]));
            memcpy(expected_score[step-1][o],myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g,o)),sizeof(expected_score[0][0]));
        }
        for(unsigned s=0;s<3;++s) {
            fill_floats(ds.target_slots[s],29u+s+step);
            memcpy(expected_slots[step-1][s],myds_metal_buffer_data(ds.target_slots[s]),sizeof(expected_slots[0][0]));
        }
        myds_v41_snapshot_set_engram_history(g,47u+step,53u+step,59u+step);
        myds_v41_snapshot_set_positions(g,pos+step,pos+step);myds_dspark_capture_publish(&ds);
        CHECK(step==1?myds_v41_spec_snapshot_capture_first(&ctx,snap):myds_v41_spec_snapshot_capture_second(&ctx,snap));
    }
    CHECK(snap->first_state_valid&&snap->second_state_valid);
    CHECK(snap->first_logits!=snap->second_logits&&snap->first_residual!=snap->second_residual&&snap->first_pre!=snap->second_pre);
    for(unsigned o=0;o<4;++o) CHECK(snap->first_previous_kv[o]!=snap->second_previous_kv[o]&&snap->first_previous_score[o]!=snap->second_previous_score[o]);
    for(unsigned s=0;s<3;++s) CHECK(snap->first_dspark_slots[s]!=snap->second_dspark_slots[s]);
    const uint64_t dispatches=myds_metal_dispatch_count();
    const int saved_barrier=myds_metal_get_auto_barrier();
    for(int barrier=0;barrier<=1;++barrier) {
        myds_metal_set_auto_barrier(barrier);
        ctx.pos=pos+3;ctx.pcache_len=pos+3;ctx.rng_state=9;cache[pos+2]=5;
        for(unsigned t=0;t<2;++t) write_window_row(g,pos+t,72.0f+(float)t);
        write_window_row(g,pos+2,97.0f);fill_floats(ctx.logits,67);
        for(unsigned o=0;o<4;++o) {
            fill_floats(myds_v41_snapshot_get_previous_kv(g,o),71u+o);
            fill_floats(myds_v41_snapshot_get_previous_score(g,o),79u+o);
        }
        for(unsigned s=0;s<3;++s) fill_floats(ds.target_slots[s],83u+s);
        myds_dspark_capture_invalidate(&ds);myds_v41_snapshot_set_engram_history(g,61,67,71);
        myds_v41_snapshot_set_positions(g,pos+3,pos+3);
        for(unsigned keep=2;keep>=1;--keep) {
            const unsigned e=keep-1;
            CHECK(myds_v41_spec_snapshot_restore_partial(&ctx,snap,keep));
            CHECK(ctx.pos==pos+keep&&ctx.pcache_len==pos+keep&&ctx.rng_state==expected_rng[e]);
            CHECK(myds_v41_snapshot_get_next_position(g)==pos+keep&&myds_v41_snapshot_get_engram_position(g)==pos+keep);
            uint32_t history[3],expected_history[3]={47u+keep,53u+keep,59u+keep};
            myds_v41_snapshot_get_engram_history(g,history);CHECK(!memcmp(history,expected_history,sizeof(history)));
            CHECK(!memcmp(myds_metal_buffer_data(ctx.logits),expected_logits[e],sizeof(expected_logits[0])));
            for(unsigned o=0;o<4;++o) {
                CHECK(!memcmp(myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g,o)),expected_kv[e][o],sizeof(expected_kv[0][0])));
                CHECK(!memcmp(myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g,o)),expected_score[e][o],sizeof(expected_score[0][0])));
            }
            CHECK(myds_dspark_capture_ready(&ds));
            for(unsigned s=0;s<3;++s) CHECK(!memcmp(myds_metal_buffer_data(ds.target_slots[s]),expected_slots[e][s],sizeof(expected_slots[0][0])));
            for(unsigned l=0;l<MYDS_V41_LAYERS_COUNT;++l) {
                const float *w=myds_metal_buffer_data(myds_v41_snapshot_get_window(g,l));
                for(unsigned k=keep;k<3;++k) CHECK(!memcmp(w+((pos+k)%WINDOW)*ROW,snap->window_saved[l][k],ROW*sizeof(float)));
                for(unsigned k=0;k<keep;++k) for(unsigned d=0;d<ROW;++d) CHECK(w[((pos+k)%WINDOW)*ROW+d]==72.0f+(float)k+(float)l/64.0f);
            }
            CHECK(myds_metal_get_auto_barrier()==barrier);
        }
        CHECK(!memcmp(myds_metal_buffer_data(snap->second_logits),expected_logits[1],sizeof(expected_logits[0])));
        CHECK(!memcmp(myds_metal_buffer_data(snap->first_logits),expected_logits[0],sizeof(expected_logits[0])));
    }
    myds_metal_set_auto_barrier(saved_barrier);CHECK(myds_metal_dispatch_count()==dispatches);
    CHECK(myds_v41_spec_snapshot_restore(&ctx,snap));CHECK(!myds_dspark_capture_ready(&ds));
    /* Validity alone must not permit a partial checkpoint with a missing
     * owner mapping to publish a reusable target state. */
    myds_metal_buffer_t *saved_owner=snap->second_previous_kv[0];
    snap->second_previous_kv[0]=NULL;
    CHECK(!myds_v41_spec_snapshot_restore_partial(&ctx,snap,2));
    CHECK(ctx.pos==pos&&!myds_v41_snapshot_get_valid(g)&&!myds_dspark_capture_ready(&ds));
    snap->second_previous_kv[0]=saved_owner;
    CHECK(myds_v41_spec_snapshot_restore(&ctx,snap));
    snap->second_state_valid=false;
    CHECK(!myds_v41_spec_snapshot_restore_partial(&ctx,snap,2));
    CHECK(!myds_v41_snapshot_get_valid(g)&&!myds_dspark_capture_ready(&ds));
    CHECK(!myds_v41_linear_spec_retry_ready(&ctx,snap));
    for(unsigned s=0;s<3;++s) myds_metal_buffer_free(ds.target_slots[s]);
    myds_metal_buffer_free(ctx.logits);free(snap);myds_v41_context_free(g);
    return 1;
}

int main(void) {
    puts("Model-free tree API guards and snapshot restoration tests");
    if (!myds_metal_init(NULL)) {
        fputs("FAIL: Metal buffer runtime initialization\n", stderr);
        return EXIT_FAILURE;
    }
    if (!test_public_tree_invalid_calls() || !test_public_linear_invalid_calls()) return EXIT_FAILURE;
    const uint32_t positions[] = {50, 126, 127, 128};
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); ++i)
        if (!test_host_window_partial_restore(positions[i])) return EXIT_FAILURE;
    if (!test_host_first_checkpoint_restore(126) || !test_host_first_checkpoint_restore(127))
        return EXIT_FAILURE;
    if (!test_host_second_checkpoint_restore(126) || !test_host_second_checkpoint_restore(127) ||
        !test_host_second_checkpoint_restore(128)) return EXIT_FAILURE;
    puts("PASS: model-free guards and snapshot restoration (no target arithmetic)");
    return EXIT_SUCCESS;
}
