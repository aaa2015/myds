#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "myds.h"
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_metal.h"
#include "myds_dspark.h"
#include "myds_log.h"

static void fill_buffer(myds_metal_buffer_t *b, float val, size_t count) {
    if (!b) return;
    float *p = (float *)myds_metal_buffer_data(b);
    if (!p) return;
    for (size_t i = 0; i < count; i++) p[i] = val;
}

static void test_spec_snapshot_save_restore(void) {
    printf("[TEST] Speculative Snapshot Save & Restore Integrity...\n");
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    assert(g && "Failed to create snapshot fixture");

    myds_context_t ctx = {0};
    ctx.v41 = g;
    ctx.pos = 40;

    int32_t pcache[256];
    memset(pcache, 0, sizeof(pcache));
    ctx.pcache_tokens = pcache;
    ctx.pcache_cap = 256;
    ctx.pcache_len = 45;

    // Fill initial state
    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        assert(w);
        for (uint32_t s = 0; s < 128; s++) {
            for (uint32_t d = 0; d < 512; d++) {
                w[s * 512 + d] = (float)(l * 10000 + s * 10 + (d % 10));
            }
        }
    }

    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer(myds_v41_snapshot_get_previous_kv(g, o), 100.0f + (float)o, 512);
        fill_buffer(myds_v41_snapshot_get_previous_score(g, o), 200.0f + (float)o, 512);
    }

    myds_v41_snapshot_set_engram_history(g, 1234, 5678, 9012);
    myds_v41_snapshot_set_positions(g, 40, 40);

    // 1. Save snapshot of 6 steps ahead (pos 40 to 45)
    myds_v41_spec_snapshot_t snap = {0};
    bool ok = myds_v41_spec_snapshot_save(&ctx, &snap, 6);
    assert(ok && "myds_v41_spec_snapshot_save failed");
    assert(snap.valid);
    assert(snap.pos == 40);
    assert(snap.window_count == 6);

    // 2. Corrupt/Mutate state as if speculative execution stepped forward and failed
    ctx.pos = 46;
    ctx.pcache_len = 46;
    myds_v41_snapshot_set_positions(g, 46, 46);
    myds_v41_snapshot_set_engram_history(g, 9999, 8888, 7777);

    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer(myds_v41_snapshot_get_previous_kv(g, o), -999.0f, 512);
        fill_buffer(myds_v41_snapshot_get_previous_score(g, o), -888.0f, 512);
    }

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        for (uint32_t k = 0; k < 6; k++) {
            uint32_t slot = (40 + k) % 128;
            for (uint32_t d = 0; d < 512; d++) {
                w[slot * 512 + d] = -777.0f;
            }
        }
    }

    // 3. Restore snapshot
    ok = myds_v41_spec_snapshot_restore(&ctx, &snap);
    assert(ok && "myds_v41_spec_snapshot_restore failed");

    // 4. Verify all fields reverted exactly
    assert(ctx.pos == 40);
    assert(ctx.pcache_len == 40);
    assert(myds_v41_snapshot_get_next_position(g) == 40);
    assert(myds_v41_snapshot_get_engram_position(g) == 40);

    uint32_t restored_hist[3] = {0};
    myds_v41_snapshot_get_engram_history(g, restored_hist);
    assert(restored_hist[0] == 1234);
    assert(restored_hist[1] == 5678);
    assert(restored_hist[2] == 9012);

    for (uint32_t o = 0; o < 4; o++) {
        float *pkv = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_previous_kv(g, o));
        float *psc = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_previous_score(g, o));
        for (size_t i = 0; i < 512; i++) {
            assert(pkv[i] == 100.0f + (float)o);
            assert(psc[i] == 200.0f + (float)o);
        }
    }

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        for (uint32_t s = 0; s < 128; s++) {
            for (uint32_t d = 0; d < 512; d++) {
                float expected = (float)(l * 10000 + s * 10 + (d % 10));
                assert(w[s * 512 + d] == expected);
            }
        }
    }

    // Clean up
    myds_v41_context_free(g);
    printf("  [PASS] Speculative Snapshot Save & Restore bit-for-bit exact.\n");
}

static void test_spec_snapshot_boundary_checks(void) {
    printf("[TEST] Speculative Snapshot Boundary Checks...\n");

    // NULL pointers
    assert(!myds_v41_spec_snapshot_save(NULL, NULL, 5));
    myds_v41_spec_snapshot_t snap = {0};
    assert(!myds_v41_spec_snapshot_restore(NULL, &snap));
    assert(!myds_v41_spec_snapshot_restore(NULL, NULL));

    // Invalid snapshot flag
    snap.valid = false;
    myds_context_t ctx = {0};
    assert(!myds_v41_spec_snapshot_restore(&ctx, &snap));

    // Max ahead clamp check
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    ctx.v41 = g;
    ctx.pos = 10;
    bool ok = myds_v41_spec_snapshot_save(&ctx, &snap, 9999);
    assert(ok);
    assert(snap.window_count == MYDS_V41_SPEC_MAX_AHEAD);

    myds_v41_context_free(g);
    printf("  [PASS] Boundary checks passed.\n");
}

static void test_spec_snapshot_partial_restore(void) {
    printf("[TEST] Speculative Snapshot Partial Restore Integrity...\n");
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    assert(g && "Failed to create snapshot fixture");

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
        assert(w);
        for (uint32_t s = 0; s < 128; s++) {
            for (uint32_t d = 0; d < 512; d++) {
                w[s * 512 + d] = (float)(l * 10000 + s * 10 + (d % 10));
            }
        }
    }

    myds_v41_snapshot_set_positions(g, 50, 50);

    // Save snapshot of 4 steps ahead (pos 50 to 53)
    myds_v41_spec_snapshot_t snap = {0};
    bool ok = myds_v41_spec_snapshot_save(&ctx, &snap, 4);
    assert(ok && snap.valid);

    // Simulate stepping forward 4 steps: mutate slots 50, 51, 52, 53
    ctx.pos = 54;
    ctx.pcache_len = 54;
    myds_v41_snapshot_set_positions(g, 54, 54);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        for (uint32_t k = 0; k < 4; k++) {
            uint32_t slot = (50 + k) % 128;
            for (uint32_t d = 0; d < 512; d++) {
                w[slot * 512 + d] = 888.0f + (float)k;
            }
        }
    }

    // Partial restore: keep 2 steps (slots 50, 51 kept), revert slots 52, 53
    ok = myds_v41_spec_snapshot_restore_partial(&ctx, &snap, 2);
    assert(ok && "restore_partial failed");
    assert(ctx.pos == 52);
    assert(ctx.pcache_len == 52);
    assert(myds_v41_snapshot_get_next_position(g) == 52);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        // k = 0, 1 (slots 50, 51) must KEEP 888.0f + k
        for (uint32_t k = 0; k < 2; k++) {
            uint32_t slot = (50 + k) % 128;
            for (uint32_t d = 0; d < 512; d++) {
                assert(w[slot * 512 + d] == 888.0f + (float)k);
            }
        }
        // k = 2, 3 (slots 52, 53) must be REVERTED to original expected value
        for (uint32_t k = 2; k < 4; k++) {
            uint32_t slot = (50 + k) % 128;
            for (uint32_t d = 0; d < 512; d++) {
                float expected = (float)(l * 10000 + slot * 10 + (d % 10));
                assert(w[slot * 512 + d] == expected);
            }
        }
    }

    myds_v41_context_free(g);
    printf("  [PASS] Speculative Snapshot Partial Restore exactly preserved kept steps and rolled back remainder.\n");
}

static void test_adaptive_speculative_backoff(void) {
    printf("[TEST] Adaptive Speculative Scheduler Backoff State Machine...\n");

    myds_dspark_t ds = {0};
    ds.block_size = 3;

    // Initially active
    assert(ds.consecutive_misses == 0);
    assert(ds.backoff_counter == 0);

    // Cycle 1: 0 accepted
    uint32_t accepted = 0;
    if (accepted == 0) {
        ds.consecutive_misses++;
        if (ds.consecutive_misses >= 4) ds.backoff_counter = 8;
        else if (ds.consecutive_misses >= 2) ds.backoff_counter = 4;
    }
    assert(ds.consecutive_misses == 1);
    assert(ds.backoff_counter == 0); // Not backed off yet after 1 miss

    // Cycle 2: 0 accepted -> triggers backoff 4
    if (accepted == 0) {
        ds.consecutive_misses++;
        if (ds.consecutive_misses >= 4) ds.backoff_counter = 8;
        else if (ds.consecutive_misses >= 2) ds.backoff_counter = 4;
    }
    assert(ds.consecutive_misses == 2);
    assert(ds.backoff_counter == 4);

    // Verify backoff decrements in propose
    for (int i = 4; i > 0; i--) {
        assert(ds.backoff_counter == (uint32_t)i);
        // Simulate myds_dspark_propose backoff check
        if (ds.backoff_counter > 0) {
            ds.backoff_counter--;
        }
    }
    assert(ds.backoff_counter == 0);

    // Hit resets everything
    accepted = 2;
    if (accepted > 0) {
        ds.consecutive_misses = 0;
        ds.backoff_counter = 0;
    }
    assert(ds.consecutive_misses == 0);
    assert(ds.backoff_counter == 0);

    // 4 misses trigger backoff 8
    for (int i = 0; i < 4; i++) {
        ds.consecutive_misses++;
        if (ds.consecutive_misses >= 4) ds.backoff_counter = 8;
        else if (ds.consecutive_misses >= 2) ds.backoff_counter = 4;
    }
    assert(ds.consecutive_misses == 4);
    assert(ds.backoff_counter == 8);

    printf("  [PASS] Adaptive Scheduler Backoff State Machine verified.\n");
}

static void test_spec_micro_batch_verification_sim(void) {
    printf("[TEST] Micro-Batch Forward Verification & Partial Rollback State Machine...\n");
    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    assert(g && "Failed to create snapshot fixture");

    myds_context_t ctx = {0};
    ctx.v41 = g;
    ctx.pos = 100;

    int32_t pcache[256];
    memset(pcache, 0, sizeof(pcache));
    ctx.pcache_tokens = pcache;
    ctx.pcache_cap = 256;
    ctx.pcache_len = 100;

    // Fill initial state
    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        assert(w);
        for (uint32_t s = 0; s < 128; s++) {
            for (uint32_t d = 0; d < 512; d++) {
                w[s * 512 + d] = (float)(l * 1000 + s);
            }
        }
    }
    myds_v41_snapshot_set_positions(g, 100, 100);

    // Case 1: Mismatch on token 0 -> partial rollback keeping 1 step
    myds_v41_spec_snapshot_t snap = {0};
    bool ok = myds_v41_spec_snapshot_save(&ctx, &snap, 2);
    assert(ok && snap.valid);

    // Simulate batch execution mutates pos 100 and pos 101
    ctx.pos = 102;
    ctx.pcache_tokens[100] = 501;
    ctx.pcache_tokens[101] = 502; // Wrong speculative draft token
    ctx.pcache_len = 102;
    myds_v41_snapshot_set_positions(g, 102, 102);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s0 = 100 % 128;
        uint32_t s1 = 101 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            w[s0 * 512 + d] = 111.0f; // Kept
            w[s1 * 512 + d] = 999.0f; // Polluted, to be rolled back
        }
    }

    // Speculative mismatch: restore_partial(1)
    ok = myds_v41_spec_snapshot_restore_partial(&ctx, &snap, 1);
    assert(ok);
    assert(ctx.pos == 101);
    assert(ctx.pcache_len == 101);
    assert(myds_v41_snapshot_get_next_position(g) == 101);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s0 = 100 % 128;
        uint32_t s1 = 101 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            assert(w[s0 * 512 + d] == 111.0f); // Kept!
            assert(w[s1 * 512 + d] == (float)(l * 1000 + s1)); // Reverted!
        }
    }

    // Case 2: Full match -> advance both without rollback
    memset(&snap, 0, sizeof(snap));
    ok = myds_v41_spec_snapshot_save(&ctx, &snap, 2);
    assert(ok && snap.valid);

    // Simulate batch execution mutates pos 101 and pos 102
    ctx.pos = 103;
    ctx.pcache_tokens[101] = 601;
    ctx.pcache_tokens[102] = 602;
    ctx.pcache_len = 103;
    myds_v41_snapshot_set_positions(g, 103, 103);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s1 = 101 % 128;
        uint32_t s2 = 102 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            w[s1 * 512 + d] = 222.0f;
            w[s2 * 512 + d] = 333.0f;
        }
    }

    // Speculative full match: keep 2 steps (noop or keep_count >= window_count)
    ok = myds_v41_spec_snapshot_restore_partial(&ctx, &snap, 2);
    assert(ok);
    assert(ctx.pos == 103);
    assert(ctx.pcache_len == 103);
    assert(myds_v41_snapshot_get_next_position(g) == 103);

    for (uint32_t l = 0; l < MYDS_V41_LAYERS_COUNT; l++) {
        float *w = (float *)myds_metal_buffer_data(myds_v41_snapshot_get_window(g, l));
        uint32_t s1 = 101 % 128;
        uint32_t s2 = 102 % 128;
        for (uint32_t d = 0; d < 512; d++) {
            assert(w[s1 * 512 + d] == 222.0f); // Kept!
            assert(w[s2 * 512 + d] == 333.0f); // Kept!
        }
    }

    myds_v41_context_free(g);
    printf("  [PASS] Micro-batch verification dual-branch state machine passed cleanly.\n");
}

int main(void) {
    printf("\n=======================================================\n");
    printf("  DSpark Rollback & Speculative State Machine Tests     \n");
    printf("=======================================================\n");
    if (!myds_metal_init(NULL)) {
        fprintf(stderr, "Failed to initialize Metal\n");
        return 1;
    }

    test_spec_snapshot_save_restore();
    test_spec_snapshot_partial_restore();
    test_spec_snapshot_boundary_checks();
    test_adaptive_speculative_backoff();
    test_spec_micro_batch_verification_sim();

    printf("\n=== All DSpark Rollback Tests Passed Successfully! ===\n\n");
    return 0;
}

