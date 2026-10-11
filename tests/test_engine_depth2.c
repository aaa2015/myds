/* Host lifecycle oracle: execute the real Engine generation loop, replacing
 * tokenizer, target graph and snapshot boundaries with deterministic CPU
 * state. This proves publication/rollback/cache contracts, not Metal math.
 * Link project objects except myds_engine.o, as test_dspark_proposal does for
 * its selector. Never initialize Metal or load model weights in this test. */
#define _POSIX_C_SOURCE 200809L
#include "myds.h"
#include "myds_dspark.h"
#include "myds_v41.h"
#include "myds_tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t next;
    int32_t inputs[64];
    int32_t logits_next, capture_input;
} host_state_t;
typedef struct {
    myds_context_t ctx;
    myds_model_t model;
    myds_tokenizer_t tokenizer;
    myds_dspark_t ds;
    myds_v41_spec_snapshot_t snap;
    host_state_t state, base, first, second;
    int ret, fail_bytes_token;
    bool fail_graph, retry_ready, fail_recovery, fail_restore, fail_capture;
    unsigned scalar_calls, linear_calls, batch_calls, batch_n, proposal_calls;
    unsigned keep1, keep2, whole, max_cap, emitted_count, cancel_after;
    int32_t emitted[32], last_consumed_seen, emitted_seed_seen;
    uint32_t proposal_pos_seen;
    volatile int cancelled;
} fixture_t;
static fixture_t *active;

static void consume(myds_context_t *ctx, int32_t token, uint32_t pos) {
    fixture_t *f = active;
    assert(ctx == &f->ctx && pos == f->state.next && pos < 64u);
    assert(token == (int32_t)(10u + pos));
    f->state.inputs[pos] = token;
    f->state.next = pos + 1u;
    f->state.logits_next = token + 1;
    f->state.capture_input = token;
}
static int32_t host_scalar(myds_context_t *ctx, int32_t token, uint32_t pos, bool logits) {
    fixture_t *f = active;
    f->scalar_calls++;
    if (f->fail_recovery && f->linear_calls) return -1;
    consume(ctx, token, pos);
    if (!f->fail_capture) {
        f->ds.capture_valid_mask = 7u;
        myds_dspark_capture_publish(&f->ds);
    }
    if (ctx->decode_metrics_active) {
        ctx->decode_stats.target_forward_calls++;
        ctx->decode_stats.target_rows++;
    }
    return logits ? token + 1 : 0;
}
static int host_prefill(myds_context_t *ctx, const int32_t *tokens,
    uint32_t count, volatile int *cancel) {
    (void)cancel;
    for (uint32_t i = 0; i < count; ++i) {
        consume(ctx, tokens[i], ctx->pos);
        ctx->pos++;
    }
    return 1;
}
static int host_encode(const myds_tokenizer_t *tok, const char *text,
    int32_t *tokens, uint32_t capacity, bool bos) {
    (void)tok; (void)text; (void)bos;
    assert(capacity > 0u); tokens[0] = 10; return 1;
}
static const char *host_bytes(const myds_tokenizer_t *tok, int32_t token, size_t *length) {
    (void)tok;
    static const char piece[3] = {'x', '\0', 'y'};
    if (token == active->fail_bytes_token) return NULL;
    if (length) *length = sizeof(piece);
    return piece;
}
static void host_clear(myds_v41_context_t *g, uint32_t keep) {
    (void)g; assert(keep == 0u);
    memset(&active->state, 0, sizeof(active->state));
    for (unsigned i = 0; i < 64u; ++i) active->state.inputs[i] = -1;
    active->state.logits_next = active->state.capture_input = -1;
}
static bool host_store(myds_v41_context_t *g, uint32_t pos) {
    (void)g; (void)pos; return true;
}
static bool host_disk_save(myds_v41_context_t *g, const int32_t *tokens, uint32_t count) {
    (void)g; (void)tokens; (void)count; return false;
}
static uint32_t host_propose(myds_dspark_t *ds, myds_context_t *ctx,
    int32_t consumed, int32_t seed, uint32_t cap) {
    fixture_t *f = active;
    assert(ds == &f->ds && ctx == &f->ctx && consumed == seed - 1);
    assert(ctx->pos == f->state.next && f->state.capture_input == consumed);
    f->proposal_calls++;
    f->last_consumed_seen = consumed; f->emitted_seed_seen = seed;
    f->proposal_pos_seen = ctx->pos;
    if (cap > f->max_cap) f->max_cap = cap;
    ds->cycles++; ds->proposed_tokens += cap;
    ds->draft_count = cap;
    ds->draft_tokens[0] = f->ret == 1 ? seed + 8 : seed + 1;
    ds->draft_tokens[1] = f->ret == 2 ? seed + 8 : seed + 2;
    ds->draft_tree_tokens[0] = ds->draft_tokens[0];
    ds->draft_tree_tokens[1] = seed + 20;
    ds->draft_tree_count = 2u; ds->is_high_confidence = true;
    return cap;
}
static bool host_whole(myds_context_t *ctx, const myds_v41_spec_snapshot_t *snap) {
    fixture_t *f = active;
    f->whole++;
    if (f->fail_restore) return false;
    f->state = f->base; ctx->pos = snap->pos; ctx->pcache_len = snap->pos;
    return true;
}
static bool host_partial(myds_context_t *ctx, const myds_v41_spec_snapshot_t *snap,
    uint32_t keep) {
    fixture_t *f = active;
    assert(keep == 1u || keep == 2u);
    if (keep == 1u) f->keep1++; else f->keep2++;
    if (f->fail_restore) return false;
    assert(snap->valid && snap->first_state_valid &&
        (keep == 1u || snap->second_state_valid));
    f->state = keep == 1u ? f->first : f->second;
    ctx->pos = snap->pos + keep; ctx->pcache_len = ctx->pos;
    return true;
}
static bool host_retry(const myds_context_t *ctx, const myds_v41_spec_snapshot_t *snap) {
    return active->retry_ready && snap->linear_retry_valid && snap->valid &&
        ctx->pos == snap->pos && active->state.next == snap->pos;
}
static int host_linear(myds_context_t *ctx, const int32_t *tokens, uint32_t count,
    uint32_t pos, int32_t *out, myds_v41_spec_snapshot_t *snap) {
    fixture_t *f = active;
    assert(count == 3u && ctx->cfg.temperature <= 0.0f && !ctx->img_emb_n);
    assert(pos == ctx->pos && tokens[0] == (int32_t)(10u + pos));
    assert(tokens[1] == f->ds.draft_tokens[0] && tokens[2] == f->ds.draft_tokens[1]);
    f->linear_calls++;
    ctx->decode_stats.depth2_calls++;
    ctx->decode_stats.batch2_calls++;
    ctx->decode_stats.target_forward_calls++; ctx->decode_stats.target_rows += 3u;
    f->base = f->state;
    snap->valid = true; snap->pos = snap->next_position = pos;
    snap->window_count = 3u; snap->linear_retry_valid = false;
    snap->first_state_valid = snap->second_state_valid = false;
    if (f->fail_graph) {
        consume(ctx, tokens[0], pos);
        if (f->retry_ready) { f->state = f->base; ctx->pos = pos; snap->linear_retry_valid = true; }
        myds_dspark_capture_poison(&f->ds);
        ctx->decode_stats.depth2_failures++; ctx->decode_stats.batch2_failures++;
        return -1;
    }
    unsigned ret = (unsigned)f->ret;
    for (unsigned i = 0; i < ret; ++i) {
        consume(ctx, tokens[i], pos + i);
        if (i == 0u) { f->first = f->state; snap->first_state_valid = true; }
        if (i == 1u) { f->second = f->state; snap->second_state_valid = true; }
        out[i] = tokens[i] + 1;
        if (out[i] == f->tokenizer.eos_id) { ret = i + 1u; break; }
    }
    ctx->pos = pos + ret;
    if (ret == 1u) { ctx->decode_stats.depth2_misses++; ctx->decode_stats.batch2_misses++; }
    else {
        if (ret == 2u) ctx->decode_stats.depth2_hits1++; else ctx->decode_stats.depth2_hits2++;
        ctx->decode_stats.batch2_hits++;
    }
    return (int)ret;
}
static int host_batch(myds_context_t *ctx, const int32_t *tokens, uint32_t count,
    uint32_t pos, int32_t *out, myds_v41_spec_snapshot_t *snap) {
    fixture_t *f = active;
    f->batch_calls++; f->batch_n = count;
    assert(count == 2u || count == 3u);
    f->base = f->state; snap->valid = true; snap->pos = pos;
    consume(ctx, tokens[0], pos); out[0] = tokens[0] + 1;
    f->first = f->state; snap->first_state_valid = true;
    if (out[0] != tokens[1] || out[0] == f->tokenizer.eos_id) { ctx->pos = pos + 1u; return 1; }
    consume(ctx, tokens[1], pos + 1u); out[1] = tokens[1] + 1;
    ctx->pos = pos + 2u; return 2;
}

#define myds_v41_forward host_scalar
#define myds_v41_prefill_tokens host_prefill
#define myds_tokenizer_encode host_encode
#define myds_tokenizer_decode_bytes host_bytes
#define myds_v41_snapshot_clear host_clear
#define myds_v41_snapshot_store_now host_store
#define myds_v41_low_kv_save host_disk_save
#define myds_dspark_propose_next host_propose
#define myds_v41_spec_snapshot_restore host_whole
#define myds_v41_spec_snapshot_restore_partial host_partial
#define myds_v41_linear_spec_retry_ready host_retry
#define myds_v41_forward_linear_spec_ex host_linear
#define myds_v41_forward_batch_spec_ex host_batch
#include "../src/myds_engine.c"
static void callback(const char *bytes, size_t length, int32_t token, void *opaque) {
    fixture_t *f = opaque;
    assert(length == 3u && bytes[0] == 'x' && bytes[1] == '\0' && bytes[2] == 'y');
    assert(token == (int32_t)(11u + f->emitted_count));
    f->emitted[f->emitted_count++] = token;
    if (f->cancel_after && f->emitted_count == f->cancel_after) f->cancelled = 1;
}
static void init(fixture_t *f, int ret) {
    memset(f, 0, sizeof(*f)); active = f;
    f->ret = ret; f->fail_bytes_token = -1;
    f->model.tokenizer = &f->tokenizer; f->model.vocab_size = 64u;
    f->model.v41 = (myds_v41_model_t *)(uintptr_t)1;
    f->tokenizer.vocab_size = 64u; f->tokenizer.eos_id = 63;
    f->ctx.model = &f->model; f->ctx.v41 = (myds_v41_context_t *)f;
    f->ctx.dspark = &f->ds; f->ds.spec_snap = &f->snap;
    f->ctx.cfg.max_seq_len = 64u; f->ctx.cfg.max_new_tokens = 4u;
    f->ctx.cfg.temperature = 0.0f;
    f->ds.block_size = 5u; f->ds.capture_configured = true;
    assert(!setenv("MYDS_DSPARK_DEPTH", "2", 1));
    assert(!setenv("MYDS_TREE_SPEC", "1", 1));
    assert(!setenv("MYDS_BATCH2_SPEC", "1", 1));
    assert(!unsetenv("MYDS_BATCH2_SPEC_FORCE"));
}
static int run(fixture_t *f, myds_stop_reason_t *reason) {
    active = f;
    uint32_t prompt_count = 0;
    int ok = myds_generate_bytes_ex(&f->ctx, "host", callback, f, reason,
        &prompt_count, &f->cancelled);
    assert(prompt_count == 1u);
    return ok;
}
static void exact_state(const fixture_t *f, unsigned next, bool cache) {
    host_state_t expected = { .next = next,
        .logits_next = (int32_t)(10u + next), .capture_input = (int32_t)(9u + next) };
    for (unsigned i = 0; i < 64u; ++i) expected.inputs[i] = i < next ? (int32_t)(10u + i) : -1;
    assert(f->ctx.pos == next && !memcmp(&f->state, &expected, sizeof(expected)));
    assert(f->ctx.pcache_len == (cache ? next : 0u));
    if (cache) for (unsigned i = 0; i < next; ++i) assert(f->ctx.pcache_tokens[i] == (int32_t)(10u + i));
    for (unsigned i = 0; i < f->emitted_count; ++i) assert(f->emitted[i] == (int32_t)(11u + i));
}
static void finish(fixture_t *f) { free(f->ctx.pcache_tokens); free(f->ctx.decode_cycles); }

int main(void) {
    assert(!setenv("MYDS_PREFIX_CACHE", "0", 1));
    assert(!setenv("MYDS_PLD", "0", 1));
    assert(!setenv("MYDS_SPEC_COST_SCHEDULER", "0", 1));
    assert(!setenv("MYDS_SPEC_SERIAL_PROBE", "0", 1));
    assert(!setenv("MYDS_DECODE_METRICS", "0", 1));
    assert(!unsetenv("MYDS_TRACE_PREFIX")); assert(!unsetenv("MYDS_DEBUG_TAIL"));
    unsigned cases = 0;
    fixture_t *f = calloc(1, sizeof(*f)); assert(f);
    myds_stop_reason_t reason;
    for (int ret = 1; ret <= 3; ++ret) {
        init(f, ret); if (ret < 3) f->cancel_after = (unsigned)ret + 1u;
        assert(run(f, &reason));
        assert(f->linear_calls == 1u && !f->batch_calls && f->max_cap == 2u);
        assert(f->last_consumed_seen == 10 && f->emitted_seed_seen == 11 && f->proposal_pos_seen == 1u);
        assert(f->ctx.decode_stats.draft_proposed == 2u && f->ctx.decode_stats.draft_accepted == (unsigned)ret - 1u);
        assert(f->ds.accepted_tokens == (unsigned)ret - 1u);
        assert(f->ctx.decode_stats.depth2_misses == (ret == 1));
        assert(f->ctx.decode_stats.depth2_hits1 == (ret == 2));
        assert(f->ctx.decode_stats.depth2_hits2 == (ret == 3));
        exact_state(f, (unsigned)ret + 1u, true); finish(f); cases++;
    }
    for (unsigned keep = 1; keep <= 2; ++keep) {
        init(f, 3); f->cancel_after = keep + 1u;
        assert(run(f, &reason) && reason == MYDS_STOP_CANCELLED);
        assert(f->linear_calls == 1u && (keep == 1u ? f->keep1 : f->keep2) == 1u);
        assert(f->ctx.decode_stats.draft_accepted == keep - 1u);
        exact_state(f, keep + 1u, true); finish(f); cases++;
    }
    for (unsigned keep = 0; keep <= 2; ++keep) {
        init(f, 3); f->fail_bytes_token = (int)(12u + keep);
        assert(!run(f, &reason) && reason == MYDS_STOP_ERROR);
        assert(f->emitted_count == keep + 1u && f->ctx.decode_stats.draft_accepted == (keep ? keep - 1u : 0u));
        assert(keep == 0u ? f->whole == 1u : (keep == 1u ? f->keep1 : f->keep2) == 1u);
        exact_state(f, keep + 1u, false); finish(f); cases++;
    }
    for (unsigned eos_row = 1; eos_row <= 3; ++eos_row) {
        init(f, 3); f->tokenizer.eos_id = (int32_t)(11u + eos_row);
        assert(run(f, &reason) && reason == MYDS_STOP_EOS);
        assert(f->emitted_count == eos_row && f->ctx.decode_stats.draft_accepted == eos_row - 1u);
        exact_state(f, eos_row + 1u, true); finish(f); cases++;
    }
    init(f, 3); f->fail_graph = f->retry_ready = true; f->cancel_after = 2u;
    assert(run(f, &reason));
    assert(f->scalar_calls == 2u && f->ctx.decode_stats.depth2_failures == 1u &&
        f->ctx.decode_stats.depth2_fallbacks == 1u && f->ctx.decode_stats.batch2_fallbacks == 1u &&
        !f->ctx.decode_stats.draft_accepted && !f->ctx.decode_stats.depth2_hits2);
    exact_state(f, 2u, true); finish(f); cases++;
    for (unsigned failure = 0; failure < 5u; ++failure) {
        init(f, 3); f->fail_graph = true; f->retry_ready = failure != 0u;
        f->fail_recovery = failure == 1u; f->fail_restore = failure == 2u || failure == 3u;
        f->fail_capture = failure == 4u;
        if (failure == 2u || failure == 3u) { f->fail_graph = false; f->cancel_after = failure; }
        assert(!run(f, &reason) && reason == MYDS_STOP_ERROR);
        assert(f->ctx.pcache_len == 0u && !f->ctx.decode_stats.depth2_fallbacks);
        assert(f->linear_calls == 1u && f->scalar_calls == (failure == 1u || failure == 4u ? 2u : 1u));
        assert(f->emitted_count == (failure == 2u || failure == 3u ? failure : 1u));
        finish(f); cases++;
    }
    for (unsigned mode = 0; mode < 4u; ++mode) {
        init(f, 3); f->ctx.cfg.max_new_tokens = 3u;
        if (mode == 1u) { f->ctx.cfg.temperature = 0.8f; assert(!setenv("MYDS_BATCH2_SPEC_FORCE", "1", 1)); }
        if (mode == 2u) { f->ds.capture_engine_only = true; f->ds.capture_bypass_mode = true; }
        if (mode == 3u) assert(!setenv("MYDS_DSPARK_DEPTH", "1", 1));
        assert(run(f, &reason));
        assert(!f->linear_calls && f->max_cap == 1u && f->batch_calls == 1u);
        assert(f->batch_n == (mode == 0u ? 2u : 3u));
        exact_state(f, 3u, true); finish(f); cases++;
    }
    init(f, 3); f->ctx.cfg.max_seq_len = 3u;
    assert(run(f, &reason)); assert(!f->linear_calls && f->max_cap == 1u && f->batch_n == 2u);
    exact_state(f, 3u, true); finish(f); cases++;
    init(f, 3); assert(!unsetenv("MYDS_BATCH2_SPEC"));
    assert(run(f, &reason)); assert(f->linear_calls == 1u && !f->batch_calls);
    exact_state(f, 4u, true); finish(f); cases++;
    /* A second proposal must observe the previous graph's final consumed
     * input, rather than the last emitted bonus token or the original seed. */
    init(f, 3); f->ctx.cfg.max_new_tokens = 7u;
    assert(run(f, &reason));
    assert(f->linear_calls == 2u && f->proposal_calls == 2u && !f->batch_calls);
    assert(f->last_consumed_seen == 13 && f->emitted_seed_seen == 14 && f->proposal_pos_seen == 4u);
    assert(f->ctx.decode_stats.draft_proposed == 4u && f->ctx.decode_stats.draft_accepted == 4u);
    exact_state(f, 7u, true); finish(f); cases++;
    init(f, 2); f->ctx.cfg.max_new_tokens = 5u;
    assert(run(f, &reason));
    assert(f->linear_calls == 1u && f->batch_calls == 1u && f->batch_n == 2u);
    assert(f->last_consumed_seen == 12 && f->emitted_seed_seen == 13 && f->proposal_pos_seen == 3u);
    exact_state(f, 5u, true); finish(f); cases++;
    init(f, 3);
    const char *bad_depth[] = { "", "0", "2x", "3", "-2" };
    for (unsigned i = 0; i < sizeof(bad_depth) / sizeof(bad_depth[0]); ++i) {
        assert(!setenv("MYDS_DSPARK_DEPTH", bad_depth[i], 1));
        assert(myds_dspark_decode_depth(&f->ctx) == 1u); cases++;
    }
    assert(!unsetenv("MYDS_DSPARK_DEPTH")); assert(myds_dspark_decode_depth(&f->ctx) == 1u); cases++;
    assert(!setenv("MYDS_DSPARK_DEPTH", "2", 1));
    f->ctx.cfg.temperature = NAN; assert(myds_dspark_decode_depth(&f->ctx) == 1u); cases++;
    f->ctx.cfg.temperature = 0.0f; f->ctx.img_emb_n = 1u;
    assert(myds_dspark_decode_depth(&f->ctx) == 1u); cases++;
    free(f);
    printf("{\"kind\":\"host_engine_depth2\",\"cases\":%u,\"real_engine_loop\":true,"
        "\"partial1_partial2\":true,\"cache_and_consumed_input\":true,"
        "\"retry_fail_closed\":true,\"gpu_arithmetic_verified\":false,\"pass\":true}\n", cases);
    return 0;
}
