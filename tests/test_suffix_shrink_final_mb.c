/* [CAND-KV2-P1] Host unit test for MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY.
 * Verifies fail-closed env parsing, /debug/state counters, exact total_eff,
 * sh_s_global, sh_full_rel, sh_loop_rel, kv_only_skip_win, anchor preservation,
 * and cross-macroblock sliding-window causality across 64K+64K, 64K+512,
 * 64K+2000, seam boundaries, 128K+128K, and anchor-in-mb0 cases. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/myds_v41.c"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static void test_env_parsing_fail_closed(void) {
    unsetenv("MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY");
    REQUIRE(myds_v41_gpu_suffix_shrink_final_mb_only_enabled() == 0);

    setenv("MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY", "0", 1);
    REQUIRE(myds_v41_gpu_suffix_shrink_final_mb_only_enabled() == 0);

    setenv("MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY", "1", 1);
    REQUIRE(myds_v41_gpu_suffix_shrink_final_mb_only_enabled() == 1);

    const char *bad[] = {"", "2", "-1", "true", "false", " 1", "1 ", "01", "yes"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
        setenv("MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY", bad[i], 1);
        REQUIRE(myds_v41_gpu_suffix_shrink_final_mb_only_enabled() == -1);
    }
    unsetenv("MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY");
}

static void test_debug_state_counters(void) {
    myds_v41_gpu_suffix_shrink_final_mb_counters_t c;
    myds_v41_gpu_suffix_shrink_final_mb_counters(NULL, true);

    myds_v41_gpu_suffix_shrink_final_mb_record(false, false); /* deferred */
    myds_v41_gpu_suffix_shrink_final_mb_record(false, true);  /* anchor */
    myds_v41_gpu_suffix_shrink_final_mb_record(true, false);  /* final */

    myds_v41_gpu_suffix_shrink_final_mb_counters(&c, false);
    REQUIRE(c.requests == 3u);
    REQUIRE(c.deferred_mbs == 1u);
    REQUIRE(c.final_mbs == 1u);
    REQUIRE(c.anchor_mbs == 1u);

    /* Non-reset read must not subtract evidence. */
    myds_v41_gpu_suffix_shrink_final_mb_counters(&c, false);
    REQUIRE(c.requests == 3u);

    myds_v41_gpu_suffix_shrink_final_mb_counters(&c, true);
    REQUIRE(c.requests == 3u);
    myds_v41_gpu_suffix_shrink_final_mb_counters(&c, false);
    REQUIRE(c.requests == 0u && c.deferred_mbs == 0u && c.final_mbs == 0u && c.anchor_mbs == 0u);
}

/* Simulate the exact prefill macroblock + layer + chunk loop and verify:
 *  1) Every token p in [base_pos, prompt_end) produces L20 compressed KV (kv_only or full).
 *  2) For every layer il in [20..39] and every FULL token p in [base_pos, prompt_end):
 *     - Every preceding token q in [max(base_pos, p-127), p-1] was written to g->window[il]
 *       (i.e., ran with !kv_only_skip_win and was not skipped by sh_loop_rel / opt_skip_dead_kv).
 *     - Every preceding token q in [max(base_pos, p-127), p-1] that lies in the same macroblock
 *       as its window-producing kv_only/full pass had layer il-1 computed FULL.
 *  3) Every token q in [prompt_end - shrink_shift, prompt_end) at il=39 is computed FULL
 *     (preserving final prompt-end and final-512-grid snapshots).
 *  4) On deferred non-final MBs with opt_skip_dead_kv=false, zero dead kv_only chunks execute
 *     on L21..39 (because sh_loop_rel = skv skips dead prefix directly). */
typedef struct {
    uint64_t full_token_layers;
    uint64_t dead_kv_only_chunks_l21_39;
    uint64_t mb0_l21_39_chunks;
    uint32_t mb0_total_eff;
    uint32_t mb1_total_eff;
    bool     mb0_deferred;
    bool     mb1_deferred;
} sim_stats_t;

static sim_stats_t simulate_case(
    uint32_t base_pos, uint32_t n_tokens, uint32_t macroblock_size,
    uint32_t chunk_size, uint32_t shrink_shift,
    bool opt_shrink_final_mb_only, bool opt_skip_dead_kv,
    const myds_context_t *ctx) {
    const uint32_t prompt_end = base_pos + n_tokens;
    const uint32_t n_layers = 20u; /* L20..39 */
    /* Per-token bitmaps relative to base_pos for L20..39:
     *   bit 0: full output computed
     *   bit 1: valid KV produced (!kv_only_skip_win && prev_full)
     *   bit 2: L20 compressor run (any chunk visiting token) */
    uint8_t *state = (uint8_t *)calloc((size_t)n_layers * (size_t)n_tokens, 1u);
    REQUIRE(state != NULL);
    /* Exact 128-slot window ring mirror per layer L20..39: stores global pos written,
     * or UINT32_MAX if never validly written. */
    uint32_t ring[20][V41_WINDOW];
    for (uint32_t li = 0; li < 20u; ++li)
        for (uint32_t s = 0; s < V41_WINDOW; ++s)
            ring[li][s] = UINT32_MAX;

    sim_stats_t st;
    memset(&st, 0, sizeof(st));

    uint32_t mb_idx = 0;
    for (uint32_t mb_start = 0; mb_start < n_tokens; mb_start += macroblock_size, ++mb_idx) {
        uint32_t mb_len = n_tokens - mb_start;
        if (mb_len > macroblock_size) mb_len = macroblock_size;
        const uint32_t mb_base_pos = base_pos + mb_start;
        const uint32_t mb_end = mb_base_pos + mb_len;
        const bool mb_has_anchor = opt_shrink_final_mb_only &&
                                   v41_mb_has_anchor(ctx, mb_base_pos, mb_end);
        const v41_mb_shrink_plan_t plan = v41_suffix_shrink_mb_plan(
            base_pos, n_tokens, mb_start, mb_len,
            shrink_shift, false, opt_shrink_final_mb_only, mb_has_anchor);
        if (mb_idx == 0u) {
            st.mb0_total_eff = plan.total_eff;
            st.mb0_deferred = plan.mb_deferred;
        } else if (mb_idx == 1u) {
            st.mb1_total_eff = plan.total_eff;
            st.mb1_deferred = plan.mb_deferred;
        }

        const uint32_t cone_shift = shrink_shift;
        /* Track per-MB full output availability for L19..39 (L19 is always full on this MB). */
        for (uint32_t il = 20u; il < V41_LAYERS; ++il) {
            const v41_layer_shrink_bounds_t sb = v41_suffix_shrink_layer_bounds(
                il, mb_base_pos, mb_len, plan.total_eff, cone_shift,
                shrink_shift, plan.mb_deferred, false, V41_SNAP_STRIDE, false, 0u);

            for (uint32_t c_start = sb.sh_loop_rel; c_start < mb_len; ) {
                uint32_t c_len = mb_len - c_start;
                if (c_len > chunk_size) c_len = chunk_size;
                if (shrink_shift) {
                    const uint32_t abs0 = mb_base_pos + c_start;
                    const uint32_t grid = ((abs0 + c_len) / V41_SNAP_STRIDE) * V41_SNAP_STRIDE;
                    if (grid > abs0) c_len = grid - abs0;
                }
                if (c_start < sb.sh_full_rel && c_start + c_len > sb.sh_full_rel) {
                    c_len = sb.sh_full_rel - c_start;
                }
                REQUIRE(c_len > 0u);
                const bool kv_only = (c_start + c_len <= sb.sh_full_rel);
                const bool kv_only_skip_win = v41_kv_only_skip_win(
                    kv_only, false, mb_base_pos, c_start, c_len, sb.sh_s_global);

                if (opt_skip_dead_kv && kv_only_skip_win && !kv_source(il)) {
                    c_start += c_len;
                    continue;
                }
                if (mb_idx == 0u && il >= 21u) st.mb0_l21_39_chunks++;
                if (kv_only_skip_win && il >= 21u) st.dead_kv_only_chunks_l21_39++;

                /* Mirror exact carry_stage check at the first FULL chunk of this MB:
                 * every carry position in [max(base_pos, carry_pos - 127), carry_pos)
                 * MUST be present and valid in the 128-slot window ring! */
                const uint32_t carry_pos = mb_base_pos + c_start;
                if (c_start == sb.sh_full_rel && carry_pos != 0u) {
                    const uint32_t need_lo = (carry_pos > 127u) ? (carry_pos - 127u) : 0u;
                    const uint32_t check_lo = (need_lo < base_pos) ? base_pos : need_lo;
                    for (uint32_t wp = check_lo; wp < carry_pos; ++wp) {
                        REQUIRE(ring[il - 20u][wp % V41_WINDOW] == wp);
                    }
                }

                /* Track per-token outputs and exact 128-slot window ring writes. */
                const uint32_t win_t0 = (c_len > V41_WINDOW) ? (c_len - V41_WINDOW) : 0u;
                for (uint32_t t = 0; t < c_len; ++t) {
                    const uint32_t rel_tok = mb_start + c_start + t;
                    const uint32_t gpos = base_pos + rel_tok;
                    const bool prev_full = (il == 20u) ||
                        ((state[(size_t)(il - 21u) * n_tokens + rel_tok] & 1u) != 0u);
                    if (!kv_only) {
                        REQUIRE(prev_full);
                    }
                    uint8_t *cell = &state[(size_t)(il - 20u) * n_tokens + rel_tok];
                    if (il == 20u) *cell |= 4u;
                    if (!kv_only_skip_win && prev_full) *cell |= 2u;
                    if (!kv_only_skip_win && t >= win_t0) {
                        ring[il - 20u][gpos % V41_WINDOW] = prev_full ? gpos : UINT32_MAX;
                    }
                    if (!kv_only) {
                        *cell |= 1u;
                        st.full_token_layers++;
                    }
                }
                c_start += c_len;
            }
        }
    }

    /* Check invariant 1: L20 compressed KV populated for every single token. */
    for (uint32_t t = 0; t < n_tokens; ++t) {
        REQUIRE((state[t] & 4u) != 0u);
    }

    /* Check invariant 2: Sliding-window causality across macroblock seams!
     * For every layer il=20..39 and every FULL token p in [base_pos, prompt_end),
     * every window dependency q in [max(base_pos, p-127), p-1] MUST have written
     * its window KV (bit 1). */
    for (uint32_t il = 20u; il < V41_LAYERS; ++il) {
        const uint8_t *row = state + (size_t)(il - 20u) * n_tokens;
        for (uint32_t rel = 0; rel < n_tokens; ++rel) {
            if (!(row[rel] & 1u)) continue;
            const uint32_t w_lo = (rel > 127u) ? (rel - 127u) : 0u;
            for (uint32_t q = w_lo; q < rel; ++q) {
                REQUIRE((row[q] & 2u) != 0u);
            }
        }
    }

    /* Check invariant 3: Final layer-39 suffix cone [prompt_end - (1 + shrink_shift), prompt_end)
     * is 100% computed FULL (preserving prompt-end & final-512-grid snapshots). */
    {
        const uint32_t sub39 = 1u + shrink_shift;
        const uint32_t s39 = (prompt_end > sub39) ? (prompt_end - sub39) : 0u;
        const uint32_t rel_s39 = (s39 > base_pos) ? (s39 - base_pos) : 0u;
        const uint8_t *row39 = state + (size_t)19u * n_tokens;
        for (uint32_t rel = rel_s39; rel < n_tokens; ++rel) {
            REQUIRE((row39[rel] & 1u) != 0u);
        }
    }

    free(state);
    return st;
}

static void test_64k_plus_64k(void) {
    const uint32_t mb = 65536u;
    const uint32_t n = 131072u;
    const uint32_t shift = 512u;

    /* Baseline (MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY=0) */
    const sim_stats_t base = simulate_case(0u, n, mb, 2048u, shift, false, true, NULL);
    REQUIRE(!base.mb0_deferred && base.mb0_total_eff == 65536u);
    REQUIRE(!base.mb1_deferred && base.mb1_total_eff == 131072u);

    /* Enabled (MYDS_SUFFIX_SHRINK_FINAL_MB_ONLY=1), opt_skip_dead_kv=true */
    const sim_stats_t opt = simulate_case(0u, n, mb, 2048u, shift, true, true, NULL);
    REQUIRE(opt.mb0_deferred && opt.mb0_total_eff == 131072u);
    REQUIRE(!opt.mb1_deferred && opt.mb1_total_eff == 131072u);
    REQUIRE(opt.mb0_l21_39_chunks == 0u);
    REQUIRE(opt.dead_kv_only_chunks_l21_39 == 0u);
    /* Exact 34,390 full token-layers saved on mb0:
     * sum_{il=20..39} (1 + (39-il)*127 + 512) = 20*513 + 190*127 = 10260 + 24130 = 34390 */
    REQUIRE(base.full_token_layers - opt.full_token_layers == 34390u);

    /* Enabled with MYDS_PREFILL_SKIP_DEAD_KV=0: sh_loop_rel=skv must STILL execute 0 chunks on mb0 L21..39! */
    const sim_stats_t opt_noskip = simulate_case(0u, n, mb, 2048u, shift, true, false, NULL);
    REQUIRE(opt_noskip.mb0_l21_39_chunks == 0u);
    REQUIRE(opt_noskip.full_token_layers == opt.full_token_layers);

    /* Verify exact layer bounds on mb0 for L20 and L21..39. */
    const v41_layer_shrink_bounds_t b20 = v41_suffix_shrink_layer_bounds(
        20u, 0u, mb, n, shift, shift, true, false, V41_SNAP_STRIDE, false, 0u);
    REQUIRE(b20.sh_s_global == 131072u - 2926u);
    REQUIRE(b20.sh_full_rel == mb);
    REQUIRE(b20.sh_loop_rel == 0u);
    REQUIRE(v41_kv_only_skip_win(true, false, 0u, 63488u, 2048u, b20.sh_s_global));

    const v41_layer_shrink_bounds_t b21 = v41_suffix_shrink_layer_bounds(
        21u, 0u, mb, n, shift, shift, true, false, V41_SNAP_STRIDE, false, 0u);
    REQUIRE(b21.sh_s_global == 131072u - 2799u);
    REQUIRE(b21.sh_full_rel == mb);
    REQUIRE(b21.sh_loop_rel == mb);
}

static void test_64k_plus_short_mb1(void) {
    const uint32_t mb = 65536u;
    const uint32_t shift = 512u;

    /* 64K + 512 (mb_len_1 = 512 < 2926) */
    {
        const uint32_t n = mb + 512u; /* 66048 */
        const sim_stats_t opt = simulate_case(0u, n, mb, 2048u, shift, true, true, NULL);
        const sim_stats_t opt_noskip = simulate_case(0u, n, mb, 2048u, shift, true, false, NULL);
        REQUIRE(opt.mb0_deferred && opt.mb0_total_eff == n);
        REQUIRE(!opt.mb1_deferred && opt.mb1_total_eff == n);
        REQUIRE(opt_noskip.full_token_layers == opt.full_token_layers);

        /* At L39: s_global = 66048 - 513 = 65535 < 65536, skv = 65408. */
        const v41_layer_shrink_bounds_t b39 = v41_suffix_shrink_layer_bounds(
            39u, 0u, mb, n, shift, shift, true, false, V41_SNAP_STRIDE, false, 0u);
        REQUIRE(b39.sh_s_global == 65535u);
        REQUIRE(b39.sh_full_rel == 65535u);
        REQUIRE(b39.sh_loop_rel == 65408u);
        /* Seam chunk [65408, 65535): 65535 + 128 = 65663 > 65535 => MUST NOT skip window! */
        REQUIRE(!v41_kv_only_skip_win(true, false, 0u, 65408u, 127u, b39.sh_s_global));
    }

    /* 64K + 2000 (mb_len_1 = 2000 < 2926): seam straddles L27/L28/L29 */
    {
        const uint32_t n = mb + 2000u; /* 67536 */
        const sim_stats_t opt = simulate_case(0u, n, mb, 2048u, shift, true, true, NULL);
        const sim_stats_t opt_noskip = simulate_case(0u, n, mb, 2048u, shift, true, false, NULL);
        REQUIRE(opt.mb0_deferred && opt.mb0_total_eff == n);
        REQUIRE(opt_noskip.full_token_layers == opt.full_token_layers);
        REQUIRE(opt_noskip.dead_kv_only_chunks_l21_39 == 0u);

        /* L28: sub = 1 + 11*127 + 512 = 1910 -> s_global = 65626 > 65536, skv = 65499 < 65536. */
        const v41_layer_shrink_bounds_t b28 = v41_suffix_shrink_layer_bounds(
            28u, 0u, mb, n, shift, shift, true, false, V41_SNAP_STRIDE, false, 0u);
        REQUIRE(b28.sh_s_global == 65626u);
        REQUIRE(b28.sh_full_rel == mb);
        REQUIRE(b28.sh_loop_rel == 65499u);
        REQUIRE(!v41_kv_only_skip_win(true, false, 0u, 65499u, 37u, b28.sh_s_global));

        /* L29: sub = 1 + 10*127 + 512 = 1783 -> s_global = 65753, skv = 65626 >= 65536. */
        const v41_layer_shrink_bounds_t b29 = v41_suffix_shrink_layer_bounds(
            29u, 0u, mb, n, shift, shift, true, false, V41_SNAP_STRIDE, false, 0u);
        REQUIRE(b29.sh_s_global == 65753u);
        REQUIRE(b29.sh_full_rel == mb);
        REQUIRE(b29.sh_loop_rel == mb);
    }

    /* Sweep every tricky seam length around 128, 512, 2926, and 2926+128. */
    const uint32_t tails[] = {1u, 127u, 128u, 129u, 511u, 512u, 513u,
                              2925u, 2926u, 2927u, 3053u, 3054u, 3055u};
    for (size_t i = 0; i < sizeof(tails) / sizeof(*tails); ++i) {
        (void)simulate_case(0u, mb + tails[i], mb, 2048u, shift, true, true, NULL);
        (void)simulate_case(0u, mb + tails[i], mb, 2048u, shift, true, false, NULL);
    }
}

static void test_128k_plus_128k(void) {
    /* Cold 256K with 128K macroblocks */
    const sim_stats_t cold = simulate_case(0u, 262144u, 131072u, 2048u, 512u, true, false, NULL);
    REQUIRE(cold.mb0_deferred && cold.mb0_total_eff == 262144u);
    REQUIRE(!cold.mb1_deferred && cold.mb1_total_eff == 262144u);
    REQUIRE(cold.mb0_l21_39_chunks == 0u);

    /* Warm 128K -> 256K with 64K macroblocks (base_pos = 131072, n_tokens = 131072) */
    const sim_stats_t warm = simulate_case(131072u, 131072u, 65536u, 2048u, 512u, true, false, NULL);
    REQUIRE(warm.mb0_deferred && warm.mb0_total_eff == 262144u);
    REQUIRE(!warm.mb1_deferred && warm.mb1_total_eff == 262144u);
    REQUIRE(warm.mb0_l21_39_chunks == 0u);
}

static void test_anchor_in_mb0_preservation(void) {
    const uint32_t mb = 65536u;
    const uint32_t n = 131072u;
    const uint32_t shift = 512u;

    /* Case 1: Matching system-prompt anchor inside mb0 (ap = 32768). */
    myds_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.anchor_slots = 4u;
    ctx.anchor_len[0] = 32768u;
    REQUIRE(v41_mb_has_anchor(&ctx, 0u, mb));
    REQUIRE(!v41_mb_has_anchor(&ctx, mb, n));

    const sim_stats_t anc = simulate_case(0u, n, mb, 2048u, shift, true, true, &ctx);
    const sim_stats_t base = simulate_case(0u, n, mb, 2048u, shift, false, true, NULL);
    REQUIRE(!anc.mb0_deferred && anc.mb0_total_eff == mb);
    REQUIRE(!anc.mb1_deferred && anc.mb1_total_eff == n);
    REQUIRE(anc.full_token_layers == base.full_token_layers);

    /* Case 2: Matching anchor at exact mb0 end (ap = 65536). */
    ctx.anchor_len[0] = mb;
    REQUIRE(v41_mb_has_anchor(&ctx, 0u, mb));
    REQUIRE(!v41_mb_has_anchor(&ctx, mb, n));

    /* Case 3: Anchor at mb_base_pos (already captured in Phase 1 before Phase 2 starts). */
    ctx.anchor_len[0] = 16384u;
    REQUIRE(!v41_mb_has_anchor(&ctx, 16384u, 16384u + mb));

    /* Case 4: Stale anchor from another session (tokens mismatch pcache_tokens) MUST NOT block deferral. */
    int32_t tok_a[64], tok_b[64];
    for (int i = 0; i < 64; ++i) { tok_a[i] = i + 1; tok_b[i] = i + 100; }
    ctx.anchor_len[0] = 64u;
    ctx.anchor_tokens[0] = tok_a;
    ctx.pcache_tokens = tok_b;
    ctx.pcache_len = 64u;
    REQUIRE(!v41_mb_has_anchor(&ctx, 0u, mb));

    /* Case 5: Matching anchor tokens DOES preserve mb0. */
    ctx.pcache_tokens = tok_a;
    REQUIRE(v41_mb_has_anchor(&ctx, 0u, mb));
}

int main(void) {
    test_env_parsing_fail_closed();
    test_debug_state_counters();
    test_64k_plus_64k();
    test_64k_plus_short_mb1();
    test_128k_plus_128k();
    test_anchor_in_mb0_preservation();
    printf("[PASS] test_suffix_shrink_final_mb: env fail-closed, counters, 64K+64K (34,390 rows saved), "
           "64K+512, 64K+2000, seam sweep, 128K+128K, and anchor-in-mb0 all verified\n");
    return 0;
}
