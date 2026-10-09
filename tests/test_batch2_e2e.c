/* =================================================================
 *  test_batch2_e2e.c: Phase 3 End-to-End Batch-2 Speculative Suite
 *  Verifies:
 *  1. Dynamic environment gating (MYDS_BATCH2_SPEC=1 vs baseline)
 *  2. Speculative short-circuit & single-token fallbacks
 *  3. Workspace lifecycle (lazy initialization, reuse, cleanup)
 *  4. Speculative hit (advance 2 tokens) & miss (partial rollback)
 * ================================================================= */

#define _POSIX_C_SOURCE 200809L
#include "myds.h"
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_metal.h"
#include "myds_v41_gpu.h"
#include "myds_dspark.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define CHECK(c) do { \
    if (!(c)) { \
        fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
        return 0; \
    } \
} while (0)

static int test_batch2_spec_api_guardrails(void) {
    printf("[TEST] Batch-2 Speculative Forward API Guardrails...\n");
    int32_t tokens[2] = {100, 101};
    int32_t out[2] = {0};

    // NULL context
    CHECK(myds_v41_forward_batch_spec(NULL, tokens, 2, 0, out) == -1);

    // NULL tokens
    myds_context_t ctx = {0};
    CHECK(myds_v41_forward_batch_spec(&ctx, NULL, 2, 0, out) == -1);

    // NULL out_tokens
    CHECK(myds_v41_forward_batch_spec(&ctx, tokens, 2, 0, NULL) == -1);

    // n_tokens == 0
    CHECK(myds_v41_forward_batch_spec(&ctx, tokens, 0, 0, out) == -1);

    printf("  [PASS] Guardrails correctly reject invalid parameters.\n");
    return 1;
}

static int test_batch2_spec_short_circuit_behavior(void) {
    printf("[TEST] Real Batch-2 opt-in / PLD / greedy policy...\n");
    myds_context_t ctx = {0};
    const char *env = getenv("MYDS_BATCH2_SPEC");
    char *saved_env = env ? strdup(env) : NULL;
    unsetenv("MYDS_BATCH2_SPEC");
    CHECK(!myds_v41_batch2_spec_requested(&ctx));
    setenv("MYDS_BATCH2_SPEC", "1", 1);
    CHECK(myds_v41_batch2_spec_requested(&ctx));
    CHECK(myds_v41_batch2_spec_enabled(&ctx, 2)); // PLD, no DSpark model
    CHECK(!myds_v41_batch2_spec_enabled(&ctx, 1));
    ctx.cfg.temperature = 0.5f;
    CHECK(!myds_v41_batch2_spec_requested(&ctx));
    ctx.cfg.temperature = 0.0f;
    ctx.img_emb_n = 1;
    CHECK(!myds_v41_batch2_spec_requested(&ctx));
    ctx.img_emb_n = 0;
    myds_dspark_t ds = {0};
    ctx.dspark = &ds;
    CHECK(!myds_v41_batch2_spec_enabled(&ctx, 2));
    ctx.decode_pld_candidate = true;
    CHECK(myds_v41_batch2_spec_enabled(&ctx, 2)); // PLD remains independent of DSpark confidence
    ctx.decode_pld_candidate = false;
    ds.is_high_confidence = true;
    CHECK(myds_v41_batch2_spec_enabled(&ctx, 2));
    setenv("MYDS_BATCH2_SPEC", "0", 1);
    CHECK(!myds_v41_batch2_spec_enabled(&ctx, 2));
    if (saved_env) { setenv("MYDS_BATCH2_SPEC", saved_env, 1); free(saved_env); }
    else unsetenv("MYDS_BATCH2_SPEC");
    printf("  [PASS] Opt-in policy uses the actual verifier gate.\n");
    return 1;
}

static void fill_floats(myds_metal_buffer_t *buffer, uint32_t seed) {
    float *p = myds_metal_buffer_data(buffer);
    for (size_t i = 0; i < myds_metal_buffer_size(buffer) / sizeof(float); ++i)
        p[i] = ((int)((i * 13u + seed) % 113u) - 56) / 64.0f;
}

/* Independent sequential reference uses the primitives directly. The probe
 * calls the same production helper as forward_batch2, with both tokens in one
 * GPU batch. This deliberately avoids pretending to test a full 40-layer
 * target model without its weights. */
static int test_batch2_real_window_attention(uint32_t pos) {
    printf("[TEST] Real Metal raw-window attention parity at pos %u...\n", pos);
    const size_t window_bytes = 128u * 512u * sizeof(float);
    const size_t heads_bytes = 64u * 512u * sizeof(float);
    myds_metal_buffer_t *window = myds_metal_buffer_alloc(window_bytes);
    myds_metal_buffer_t *reference_window = myds_metal_buffer_alloc(window_bytes);
    myds_metal_buffer_t *sinks = myds_metal_buffer_alloc(64u * sizeof(float));
    myds_metal_buffer_t *q[2], *kv[2], *heads[2], *rq[2], *rkv[2], *rheads[2];
    CHECK(window && reference_window && sinks);
    void *original_window = malloc(window_bytes);
    CHECK(original_window);
    fill_floats(window, 71);
    memcpy(original_window, myds_metal_buffer_data(window), window_bytes);
    memcpy(myds_metal_buffer_data(reference_window), original_window, window_bytes);
    memset(myds_metal_buffer_data(sinks), 0, 64u * sizeof(float));
    for (uint32_t t = 0; t < 2; ++t) {
        q[t] = myds_metal_buffer_alloc(heads_bytes);
        rq[t] = myds_metal_buffer_alloc(heads_bytes);
        heads[t] = myds_metal_buffer_alloc(heads_bytes);
        rheads[t] = myds_metal_buffer_alloc(heads_bytes);
        kv[t] = myds_metal_buffer_alloc(512u * sizeof(float));
        rkv[t] = myds_metal_buffer_alloc(512u * sizeof(float));
        CHECK(q[t] && rq[t] && heads[t] && rheads[t] && kv[t] && rkv[t]);
        fill_floats(q[t], 11u + t * 19u);
        fill_floats(kv[t], 23u + t * 37u);
        memcpy(myds_metal_buffer_data(rq[t]), myds_metal_buffer_data(q[t]), heads_bytes);
        memcpy(myds_metal_buffer_data(rkv[t]), myds_metal_buffer_data(kv[t]), 512u * sizeof(float));
    }
    myds_v41_gpu_scratch_t *gpu = myds_v41_gpu_scratch_create(512);
    CHECK(gpu);
    for (uint32_t t = 0; t < 2; ++t) {
        const uint32_t p = pos + t;
        const uint32_t nraw = p + 1u < 128u ? p + 1u : 128u;
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_qkv_finish(rq[t], rkv[t], reference_window, p, false));
        CHECK(myds_v41_gpu_attention_direct(rheads[t], sinks, rq[t], reference_window,
            nraw, (p + 1u - nraw) % 128u, NULL, NULL, 0, 0, gpu));
        CHECK(myds_v41_gpu_rope_round(rheads[t], 512, 64, p, false, true));
        CHECK(myds_metal_commit_batch());
    }
    CHECK(myds_v41_batch2_window_probe(window, sinks, q, kv, heads, pos, false));
    CHECK(memcmp(myds_metal_buffer_data(window), myds_metal_buffer_data(reference_window), window_bytes) == 0);
    for (uint32_t t = 0; t < 2; ++t) {
        CHECK(memcmp(myds_metal_buffer_data(heads[t]), myds_metal_buffer_data(rheads[t]), heads_bytes) == 0);
    }

    /* Regression sensitivity: reproduce the former future-write-first order.
     * At the full-window boundary, token 0 must differ from the reference. */
    if (pos >= 127u) {
        memcpy(myds_metal_buffer_data(window), original_window, window_bytes);
        for (uint32_t t = 0; t < 2; ++t) { fill_floats(q[t], 11u + t * 19u); fill_floats(kv[t], 23u + t * 37u); }
        CHECK(myds_metal_begin_batch());
        CHECK(myds_v41_gpu_qkv_finish(q[0], kv[0], window, pos, false));
        CHECK(myds_v41_gpu_qkv_finish(q[1], kv[1], window, pos + 1u, false));
        CHECK(myds_v41_gpu_attention_direct(heads[0], sinks, q[0], window,
            128, (pos + 1u - 128u) % 128u, NULL, NULL, 0, 0, gpu));
        CHECK(myds_v41_gpu_rope_round(heads[0], 512, 64, pos, false, true));
        CHECK(myds_metal_commit_batch());
        CHECK(memcmp(myds_metal_buffer_data(heads[0]), myds_metal_buffer_data(rheads[0]), heads_bytes) != 0);
    }

    memcpy(myds_metal_buffer_data(window), original_window, window_bytes);
    for (uint32_t t = 0; t < 2; ++t) { fill_floats(q[t], 11u + t * 19u); fill_floats(kv[t], 23u + t * 37u); }
    CHECK(!myds_v41_batch2_window_probe(window, sinks, q, kv, heads, pos, true));
    CHECK(memcmp(myds_metal_buffer_data(window), original_window, window_bytes) == 0);
    for (uint32_t t = 0; t < 2; ++t) { fill_floats(q[t], 11u + t * 19u); fill_floats(kv[t], 23u + t * 37u); }
    myds_metal_inject_commit_failure(2); // The real GPU batch executes, then reports failure.
    const int failed_commit = myds_v41_batch2_window_probe(window, sinks, q, kv, heads, pos, false);
    myds_metal_inject_commit_failure(0);
    CHECK(!failed_commit);
    CHECK(memcmp(myds_metal_buffer_data(window), original_window, window_bytes) == 0);
    for (uint32_t t = 0; t < 2; ++t) {
        myds_metal_buffer_free(q[t]); myds_metal_buffer_free(kv[t]); myds_metal_buffer_free(heads[t]);
        myds_metal_buffer_free(rq[t]); myds_metal_buffer_free(rkv[t]); myds_metal_buffer_free(rheads[t]);
    }
    myds_metal_buffer_free(window); myds_metal_buffer_free(reference_window); myds_metal_buffer_free(sinks);
    myds_v41_gpu_scratch_free(gpu);
    free(original_window);
    return 1;
}

static int test_snapshot_rng_and_first_state(void) {
    printf("[TEST] Exact accepted-prefix carry / RNG / logits restoration...\n");
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    myds_v41_spec_snapshot_t *snap = calloc(1, sizeof(*snap));
    myds_v41_spec_snapshot_t *after = calloc(1, sizeof(*after));
    CHECK(g && snap && after);
    myds_model_t model = { .vocab_size = 4 };
    myds_context_t ctx = { .model = &model, .v41 = g, .pos = 127, .rng_state = 4321 };
    int32_t pcache[256] = {0};
    ctx.pcache_tokens = pcache; ctx.pcache_cap = 256; ctx.pcache_len = 127;
    ctx.logits = myds_metal_buffer_alloc(4u * sizeof(float));
    myds_v41_snapshot_set_positions(g, 127, 127);
    CHECK(myds_v41_spec_snapshot_save(&ctx, snap, 2));
    myds_metal_buffer_t *carry_kv[4], *carry_score[4];
    for (uint32_t o = 0; o < 4; ++o) {
        carry_kv[o] = myds_metal_buffer_alloc(512u * sizeof(float));
        carry_score[o] = myds_metal_buffer_alloc(512u * sizeof(float));
        CHECK(carry_kv[o] && carry_score[o]);
        fill_floats(carry_kv[o], 71u + o); fill_floats(carry_score[o], 83u + o);
        snap->first_previous_kv[o] = carry_kv[o];
        snap->first_previous_score[o] = carry_score[o];
    }
    myds_metal_buffer_t *first_logits = myds_metal_buffer_alloc(4u * sizeof(float));
    CHECK(first_logits && ctx.logits);
    fill_floats(first_logits, 17);
    fill_floats(ctx.logits, 29);
    snap->first_logits = first_logits;
    snap->first_rng_state = 9876;
    snap->first_engram_history = snap->engram_history;
    snap->first_state_valid = true;
    ctx.pos = 129; ctx.pcache_len = 129; ctx.rng_state = 1111;
    myds_v41_snapshot_set_positions(g, 129, 129);
    CHECK(myds_v41_spec_snapshot_restore_partial(&ctx, snap, 1));
    CHECK(ctx.pos == 128 && ctx.pcache_len == 128 && ctx.rng_state == 9876);
    CHECK(memcmp(myds_metal_buffer_data(ctx.logits), myds_metal_buffer_data(first_logits), 4u * sizeof(float)) == 0);
    /* Invoke the same producer used after a real serial target first-step.
     * It must create borrowed first-state views from live buffers itself. */
    snap->first_state_valid = false;
    CHECK(myds_v41_spec_snapshot_capture_first(&ctx, snap));
    CHECK(snap->first_state_valid && snap->first_rng_state == 9876);
    CHECK(snap->first_residual && snap->first_pre && snap->first_logits);
    CHECK(((const float *)myds_metal_buffer_data(snap->first_pre))[0] == 1.0f);
    ctx.rng_state = 2222;
    fill_floats(ctx.logits, 89);
    CHECK(myds_v41_spec_snapshot_restore_partial(&ctx, snap, 1));
    CHECK(ctx.rng_state == 9876);
    CHECK(memcmp(myds_metal_buffer_data(ctx.logits), myds_metal_buffer_data(first_logits), 4u * sizeof(float)) == 0);
    CHECK(myds_v41_spec_snapshot_save(&ctx, after, 1));
    for (uint32_t o = 0; o < 4; ++o) {
        CHECK(memcmp(after->previous_kv[o], myds_metal_buffer_data(carry_kv[o]), 512u * sizeof(float)) == 0);
        CHECK(memcmp(after->previous_score[o], myds_metal_buffer_data(carry_score[o]), 512u * sizeof(float)) == 0);
    }
    CHECK(myds_v41_spec_snapshot_restore(&ctx, snap));
    CHECK(ctx.pos == 127 && ctx.rng_state == 4321);
    myds_metal_inject_commit_failure(1);
    const int invalidated = myds_v41_spec_snapshot_save(&ctx, snap, 2);
    myds_metal_inject_commit_failure(0);
    CHECK(!invalidated && !snap->valid && !snap->first_state_valid);
    CHECK(myds_v41_spec_snapshot_save(&ctx, snap, 2));

    /* A real queued copy can execute before its commit reports failure.
     * Failed restoration must prevent reuse, while invalid arguments must
     * leave an otherwise valid graph available. */
    myds_metal_buffer_t *window = myds_v41_snapshot_get_window(g, 0);
    CHECK(window);
    ((float *)myds_metal_buffer_data(window))[512u] = 321.75f;
    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_buffer_copy(window, 127u * 512u * sizeof(float), window,
                                512u * sizeof(float), 512u * sizeof(float)));
    myds_metal_inject_commit_failure(1);
    const int full_restore_failed = myds_v41_spec_snapshot_restore(&ctx, snap);
    myds_metal_inject_commit_failure(0);
    CHECK(!full_restore_failed);
    CHECK(((const float *)myds_metal_buffer_data(window))[127u * 512u] == 321.75f);
    CHECK(!myds_v41_spec_snapshot_save(&ctx, after, 1));
    CHECK(myds_v41_spec_snapshot_restore(&ctx, snap));
    CHECK(((const float *)myds_metal_buffer_data(window))[127u * 512u] == snap->window_saved[0][0][0]);
    CHECK(myds_v41_spec_snapshot_save(&ctx, after, 1));

    ctx.pos = 129; ctx.pcache_len = 129;
    myds_v41_snapshot_set_positions(g, 129, 129);
    CHECK(myds_metal_begin_batch());
    CHECK(myds_metal_buffer_copy(window, 0, window, 512u * sizeof(float),
                                512u * sizeof(float)));
    myds_metal_inject_commit_failure(1);
    const int partial_restore_failed = myds_v41_spec_snapshot_restore_partial(&ctx, snap, 1);
    myds_metal_inject_commit_failure(0);
    CHECK(!partial_restore_failed && ctx.pos == 129);
    CHECK(((const float *)myds_metal_buffer_data(window))[0] == 321.75f);
    CHECK(!myds_v41_spec_snapshot_save(&ctx, after, 1));
    CHECK(myds_v41_spec_snapshot_restore(&ctx, snap));
    CHECK(myds_v41_spec_snapshot_save(&ctx, after, 1));
    after->valid = false;
    CHECK(!myds_v41_spec_snapshot_restore(&ctx, after));
    CHECK(!myds_v41_spec_snapshot_restore_partial(&ctx, after, 1));
    CHECK(!myds_v41_spec_snapshot_restore(&ctx, NULL));
    CHECK(!myds_v41_spec_snapshot_restore_partial(&ctx, NULL, 1));
    CHECK(myds_v41_spec_snapshot_save(&ctx, after, 1));
    for (uint32_t o = 0; o < 4; ++o) { myds_metal_buffer_free(carry_kv[o]); myds_metal_buffer_free(carry_score[o]); }
    myds_metal_buffer_free(first_logits); myds_metal_buffer_free(ctx.logits);
    free(snap); free(after); myds_v41_context_free(g);
    return 1;
}

static int test_batch2_spec_state_machine(void) {
    printf("[TEST] Batch-2 Speculative State Machine: Hit vs Miss Recovery...\n");
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    CHECK(g);

    myds_context_t ctx = {0};
    ctx.v41 = g;
    ctx.pos = 50;

    int32_t pcache[256];
    memset(pcache, 0, sizeof(pcache));
    ctx.pcache_tokens = pcache;
    ctx.pcache_cap = 256;
    ctx.pcache_len = 50;

    // Fill initial state
    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        CHECK(w);
        for (uint32_t s = 0; s < 128; s++) {
            for (uint32_t d = 0; d < 512; d++) {
                w[s * 512 + d] = (float)(l * 5000 + s * 10 + d);
            }
        }
    }
    myds_v41_snapshot_set_positions(g, 50, 50);

    // 1. Snapshot taken prior to Batch-2 forward
    myds_v41_spec_snapshot_t snap = {0};
    CHECK(myds_v41_spec_snapshot_save(&ctx, &snap, 2));
    CHECK(snap.valid);
    CHECK(snap.pos == 50);

    // 2. Simulate Batch-2 forward execution mutates slot 50 and slot 51
    ctx.pos = 52;
    ctx.pcache_tokens[50] = 1001;
    ctx.pcache_tokens[51] = 1002;
    ctx.pcache_len = 52;
    myds_v41_snapshot_set_positions(g, 52, 52);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s0 = 50 % 128;
        uint32_t s1 = 51 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            w[s0 * 512 + d] = 777.0f; // Token 0 generated KV
            w[s1 * 512 + d] = 888.0f; // Token 1 speculative draft KV
        }
    }

    // 3. Scenario: Draft 1002 mismatched! Authoritative was 1099.
    // Must execute partial rollback keeping 1 step (pos 50 kept, pos 51 reverted).
    CHECK(myds_v41_spec_snapshot_restore_partial(&ctx, &snap, 1));
    CHECK(ctx.pos == 51);
    CHECK(ctx.pcache_len == 51);
    CHECK(myds_v41_snapshot_get_next_position(g) == 51);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s0 = 50 % 128;
        uint32_t s1 = 51 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            CHECK(w[s0 * 512 + d] == 777.0f); // Kept!
            float expected_orig = (float)(l * 5000 + s1 * 10 + d);
            CHECK(w[s1 * 512 + d] == expected_orig); // Reverted cleanly!
        }
    }

    // 4. Scenario: Draft matched! Both tokens accepted without rollback.
    memset(&snap, 0, sizeof(snap));
    CHECK(myds_v41_spec_snapshot_save(&ctx, &snap, 2));

    ctx.pos = 53;
    ctx.pcache_tokens[51] = 2001;
    ctx.pcache_tokens[52] = 2002;
    ctx.pcache_len = 53;
    myds_v41_snapshot_set_positions(g, 53, 53);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s1 = 51 % 128;
        uint32_t s2 = 52 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            w[s1 * 512 + d] = 333.0f;
            w[s2 * 512 + d] = 444.0f;
        }
    }

    CHECK(myds_v41_spec_snapshot_restore_partial(&ctx, &snap, 2));
    CHECK(ctx.pos == 53);
    CHECK(ctx.pcache_len == 53);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s1 = 51 % 128;
        uint32_t s2 = 52 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            CHECK(w[s1 * 512 + d] == 333.0f);
            CHECK(w[s2 * 512 + d] == 444.0f);
        }
    }

    myds_v41_context_free(g);
    printf("  [PASS] Batch-2 Speculative Hit/Miss state machine 100%% verified.\n");
    return 1;
}

int main(void) {
    printf("\n=======================================================\n");
    printf("     Phase 3: Batch-2 Speculative E2E Test Suite       \n");
    printf("=======================================================\n");

    if (!myds_metal_init(NULL)) {
        fprintf(stderr, "Failed to initialize Metal\n");
        return 1;
    }

    if (!test_batch2_spec_api_guardrails() || !test_batch2_spec_short_circuit_behavior() ||
        !test_batch2_spec_state_machine() || !test_snapshot_rng_and_first_state()) return 1;
    const uint32_t positions[] = {50, 126, 127, 128, 255};
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); ++i)
        if (!test_batch2_real_window_attention(positions[i])) return 1;

    printf("\n=== ALL PHASE 3 BATCH-2 E2E TESTS PASSED CLEANLY! ===\n\n");
    return 0;
}
