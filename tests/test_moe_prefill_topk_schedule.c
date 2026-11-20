/* [P-L2] Host unit test for MYDS_MOE_PREFILL_TOPK_SCHEDULE.
 * Verifies:
 *   1) Strict fail-closed env parsing (absent/"0"/"1"/"2" vs invalid strings)
 *      and conflict rejection against non-zero/invalid MYDS_MOE_PREFILL_TOPK,
 *      including context creation rejection.
 *   2) kvdisk / myds_v41_model_identity isolation across off, TOPK=3/4,
 *      SCHEDULE=1, SCHEDULE=2, and fail-closed on invalid/conflicting env.
 *   3) Exact per-chunk policy decisions across prompt lengths (512, 2047, 2048,
 *      16384, 32768, 131072, 262144), layers (L0..L39; all 40 V4.1 layers are MoE),
 *      interior vs tail boundaries, and warm-append (past_len > 0: warm+310,
 *      warm+2K, warm_128k_to_256k).
 *   4) Full production macroblock + suffix-shrink + chunk + MoE-batch-flush
 *      simulation proving:
 *        - ZERO mixed-k MoE batches (every flush has homogeneous k and fits moe_batch),
 *        - 100% k=6 on short prompts (<2048) and warm+310 follow-ups,
 *        - 100% k=6 on early protected layers (L0..L3 preset 1, L0..L1 preset 2),
 *        - 100% k=6 on final protect_tail=512 tokens of every prompt,
 *        - Exact % of MoE FLOPs at k=4 for 16K/32K/128K/256K cold & warm_128k_to_256k.
 *   5) Live /debug/state counters & JSON serialization (non-reset read, reset,
 *      enabled/preset reflection, zero truncation). */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/myds_v41.c"
#include "myds_debugstate.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static void clear_topk_env(void) {
    unsetenv("MYDS_MOE_PREFILL_TOPK");
    unsetenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE");
    unsetenv("MYDS_PREFILL_MACROBLOCK_SIZE");
}

static void test_env_parsing_and_conflicts(void) {
    clear_topk_env();

    REQUIRE(myds_v41_gpu_moe_prefill_topk_parse() == 0);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 0);

    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "0", 1);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 0);

    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "1", 1);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 1);

    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "2", 1);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 2);

    /* Preset parameter verification */
    myds_v41_moe_prefill_topk_schedule_cfg_t cfg;
    REQUIRE(!myds_v41_gpu_moe_prefill_topk_schedule_config(0, &cfg));
    REQUIRE(!myds_v41_gpu_moe_prefill_topk_schedule_config(-1, &cfg));
    REQUIRE(!myds_v41_gpu_moe_prefill_topk_schedule_config(3, &cfg));

    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_config(1, &cfg));
    REQUIRE(cfg.preset == 1u && cfg.k_interior == 4u &&
            cfg.min_prompt_tokens == 2048u && cfg.protect_layers_lt == 4u &&
            cfg.protect_tail_tokens == 512u);

    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_config(2, &cfg));
    REQUIRE(cfg.preset == 2u && cfg.k_interior == 4u &&
            cfg.min_prompt_tokens == 2048u && cfg.protect_layers_lt == 2u &&
            cfg.protect_tail_tokens == 512u);

    /* Invalid values must fail closed (-1) */
    const char *bad[] = {
        "", "-1", "3", "4", "6", "01", "true", "false", " 1", "1 ", "auto", "k4"
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
        setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", bad[i], 1);
        REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == -1);
    }

    /* Compatibility with MYDS_MOE_PREFILL_TOPK=0 / unset */
    setenv("MYDS_MOE_PREFILL_TOPK", "0", 1);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "1", 1);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 1);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "2", 1);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 2);

    /* Explicit uniform MYDS_MOE_PREFILL_TOPK remains compatible when schedule is off */
    unsetenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE");
    setenv("MYDS_MOE_PREFILL_TOPK", "4", 1);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_parse() == 4);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 0);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "0", 1);
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == 0);

    /* Conflicts when both schedule > 0 and non-zero/invalid MYDS_MOE_PREFILL_TOPK are set */
    const char *conflicts[] = { "3", "4", "6", "2", "bad" };
    for (int p = 1; p <= 2; ++p) {
        char ps[4];
        snprintf(ps, sizeof(ps), "%d", p);
        setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", ps, 1);
        for (size_t i = 0; i < sizeof(conflicts) / sizeof(*conflicts); ++i) {
            setenv("MYDS_MOE_PREFILL_TOPK", conflicts[i], 1);
            REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_mode() == -1);
        }
    }

    /* Context creation must fail closed on invalid schedule or conflict */
    myds_model_t dummy_model;
    memset(&dummy_model, 0, sizeof(dummy_model));
    dummy_model.n_activated_experts = 6u;
    dummy_model.n_routed_experts = 384u;
    dummy_model.n_layers = V41_LAYERS;

    unsetenv("MYDS_MOE_PREFILL_TOPK");
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "invalid", 1);
    REQUIRE(myds_v41_context_create_for_model(&dummy_model) == NULL);

    setenv("MYDS_MOE_PREFILL_TOPK", "4", 1);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "1", 1);
    REQUIRE(myds_v41_context_create_for_model(&dummy_model) == NULL);

    unsetenv("MYDS_MOE_PREFILL_TOPK");
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "1", 1);
    myds_v41_context_t *ok_ctx = myds_v41_context_create_for_model(&dummy_model);
    REQUIRE(ok_ctx != NULL);
    myds_v41_context_free(ok_ctx);

    clear_topk_env();
    printf("PASS env_parsing_and_conflicts\n");
}

static void test_kvdisk_identity_isolation(void) {
    clear_topk_env();

    gguf_file_t dummy_gf;
    memset(&dummy_gf, 0, sizeof(dummy_gf));
    dummy_gf.fd = -1;

    myds_model_t dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.gf = &dummy_gf;
    snprintf(dummy.arch, sizeof(dummy.arch), "deepseek41");
    dummy.n_activated_experts = 6u;
    dummy.n_routed_experts = 384u;
    dummy.n_layers = V41_LAYERS;

    const uint64_t id_base = myds_v41_model_identity(&dummy);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "0", 1);
    const uint64_t id_zero = myds_v41_model_identity(&dummy);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "1", 1);
    const uint64_t id_s1 = myds_v41_model_identity(&dummy);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "2", 1);
    const uint64_t id_s2 = myds_v41_model_identity(&dummy);

    unsetenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE");
    setenv("MYDS_MOE_PREFILL_TOPK", "4", 1);
    const uint64_t id_k4 = myds_v41_model_identity(&dummy);
    setenv("MYDS_MOE_PREFILL_TOPK", "3", 1);
    const uint64_t id_k3 = myds_v41_model_identity(&dummy);

    /* Invalid / conflict fails closed to 0 */
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "1", 1);
    REQUIRE(myds_v41_model_identity(&dummy) == 0ull);
    unsetenv("MYDS_MOE_PREFILL_TOPK");
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "bad", 1);
    REQUIRE(myds_v41_model_identity(&dummy) == 0ull);
    clear_topk_env();

    REQUIRE(id_base != 0ull && id_s1 != 0ull && id_s2 != 0ull && id_k4 != 0ull && id_k3 != 0ull);
    REQUIRE(id_base == id_zero);
    REQUIRE(id_s1 != id_base && id_s1 != id_k4 && id_s1 != id_k3);
    REQUIRE(id_s2 != id_base && id_s2 != id_k4 && id_s2 != id_k3 && id_s2 != id_s1);

    printf("PASS kvdisk_identity_isolation (base=0x%016llx s1=0x%016llx s2=0x%016llx k4=0x%016llx)\n",
           (unsigned long long)id_base, (unsigned long long)id_s1,
           (unsigned long long)id_s2, (unsigned long long)id_k4);
}

static void test_direct_policy_boundaries(void) {
    myds_v41_prefill_topk_reason_t r = MYDS_PREFILL_TOPK_SCHED_DISABLED;

    /* Disabled preset always keeps model_k=6 */
    REQUIRE(myds_v41_gpu_moe_prefill_topk_schedule_select(0, 6u, 131072u, 10u, 0u, 2048u, &r) == 6u);
    REQUIRE(r == MYDS_PREFILL_TOPK_SCHED_DISABLED);

    /* Short prompts (<2048): 512 and 2047 stay k=6 across ALL 40 layers */
    const uint32_t short_lens[] = { 1u, 51u, 310u, 512u, 1024u, 2047u };
    for (int preset = 1; preset <= 2; ++preset) {
        for (size_t i = 0; i < sizeof(short_lens) / sizeof(*short_lens); ++i) {
            const uint32_t plen = short_lens[i];
            for (uint32_t il = 0; il < V41_LAYERS; ++il) {
                const uint32_t k = myds_v41_gpu_moe_prefill_topk_schedule_select(
                    preset, 6u, plen, il, 0u, plen, &r);
                REQUIRE(k == 6u && r == MYDS_PREFILL_TOPK_SCHED_SHORT);
            }
        }
    }

    /* Threshold prompt = 2048 (tail_start = 1536):
     * - Early layers protected (preset 1: L0..L3; preset 2: L0..L1)
     * - Deep layers: interior chunks <= 1536 get k=4; any overlap > 1536 gets k=6 (TAIL) */
    for (int preset = 1; preset <= 2; ++preset) {
        const uint32_t prot_lt = (preset == 1) ? 4u : 2u;
        for (uint32_t il = 0; il < V41_LAYERS; ++il) {
            uint32_t k0 = myds_v41_gpu_moe_prefill_topk_schedule_select(
                preset, 6u, 2048u, il, 0u, 512u, &r);
            if (il < prot_lt) {
                REQUIRE(k0 == 6u && r == MYDS_PREFILL_TOPK_SCHED_LAYER);
            } else {
                REQUIRE(k0 == 4u && r == MYDS_PREFILL_TOPK_SCHED_K4);
            }

            /* Exact boundary [1024, 1536): ends right at tail_start=1536 -> still interior! */
            uint32_t k_edge = myds_v41_gpu_moe_prefill_topk_schedule_select(
                preset, 6u, 2048u, il, 1024u, 1536u, &r);
            if (il < prot_lt) {
                REQUIRE(k_edge == 6u && r == MYDS_PREFILL_TOPK_SCHED_LAYER);
            } else {
                REQUIRE(k_edge == 4u && r == MYDS_PREFILL_TOPK_SCHED_K4);
            }

            /* 1-token overlap [1025, 1537): overlaps token 1536 -> TAIL protected! */
            uint32_t k_ov1 = myds_v41_gpu_moe_prefill_topk_schedule_select(
                preset, 6u, 2048u, il, 1025u, 1537u, &r);
            REQUIRE(k_ov1 == 6u);
            REQUIRE(r == (il < prot_lt ? MYDS_PREFILL_TOPK_SCHED_LAYER : MYDS_PREFILL_TOPK_SCHED_TAIL));

            /* Tail chunk [1536, 2048): TAIL protected! */
            uint32_t k_tail = myds_v41_gpu_moe_prefill_topk_schedule_select(
                preset, 6u, 2048u, il, 1536u, 2048u, &r);
            REQUIRE(k_tail == 6u);
            REQUIRE(r == (il < prot_lt ? MYDS_PREFILL_TOPK_SCHED_LAYER : MYDS_PREFILL_TOPK_SCHED_TAIL));

            /* Full single 2048-token chunk [0, 2048): overlaps tail -> TAIL protected! */
            uint32_t k_full = myds_v41_gpu_moe_prefill_topk_schedule_select(
                preset, 6u, 2048u, il, 0u, 2048u, &r);
            REQUIRE(k_full == 6u);
            REQUIRE(r == (il < prot_lt ? MYDS_PREFILL_TOPK_SCHED_LAYER : MYDS_PREFILL_TOPK_SCHED_TAIL));
        }
    }

    /* Warm-append: warm_128k+310 (base_pos=128512, n_tokens=310 -> total=128822, tail_start=128310)
     * Every single new token lies inside [tail_start, total) -> 100% k=6 across ALL 40 layers! */
    for (int preset = 1; preset <= 2; ++preset) {
        const uint32_t prot_lt = (preset == 1) ? 4u : 2u;
        for (uint32_t il = 0; il < V41_LAYERS; ++il) {
            uint32_t k = myds_v41_gpu_moe_prefill_topk_schedule_select(
                preset, 6u, 128822u, il, 128512u, 128822u, &r);
            REQUIRE(k == 6u);
            REQUIRE(r == (il < prot_lt ? MYDS_PREFILL_TOPK_SCHED_LAYER : MYDS_PREFILL_TOPK_SCHED_TAIL));
        }
    }

    printf("PASS direct_policy_boundaries\n");
}

typedef struct {
    uint64_t total_token_layers;
    uint64_t k4_token_layers;
    uint64_t k6_short_token_layers;
    uint64_t k6_layer_token_layers;
    uint64_t k6_tail_token_layers;
    uint64_t k4_dispatches;
    uint64_t k6_short_dispatches;
    uint64_t k6_layer_dispatches;
    uint64_t k6_tail_dispatches;
    uint64_t total_dispatches;
} sched_sim_t;

/* Mirror the exact macroblock + suffix-shrink + chunk + MoE-batch-flush loop
 * from myds_v41_prefill_tokens() and verify all safety & uniformity invariants. */
static sched_sim_t simulate_prefill_schedule(
    int preset,
    uint32_t base_pos,
    uint32_t n_tokens,
    uint32_t chunk_size_cfg,
    uint32_t moe_chunks_cfg,
    uint32_t moe_batch_max,
    bool suffix_shrink,
    uint32_t shrink_shift,
    bool final_mb_only) {
    sched_sim_t st;
    memset(&st, 0, sizeof(st));

    myds_v41_moe_prefill_topk_schedule_cfg_t cfg;
    const bool sched_on = (preset > 0) &&
                          myds_v41_gpu_moe_prefill_topk_schedule_config(preset, &cfg);
    const uint32_t total_prompt_len = base_pos + n_tokens;
    const uint32_t tail_start = (sched_on && total_prompt_len > cfg.protect_tail_tokens)
        ? (total_prompt_len - cfg.protect_tail_tokens) : 0u;

    uint32_t chunk_size = chunk_size_cfg;
    if (n_tokens < chunk_size) {
        chunk_size = n_tokens;
        if (chunk_size % 2u != 0u) chunk_size++;
        if (chunk_size < 2u) chunk_size = 2u;
    }
    const uint32_t macroblock_size = myds_v41_resolve_macroblock_size(n_tokens);
    const uint32_t mb_cap_needed = n_tokens < macroblock_size ? n_tokens : macroblock_size;

    uint32_t moe_chunks = moe_chunks_cfg;
    if (moe_chunks > 1u && (size_t)chunk_size * moe_chunks > moe_batch_max)
        moe_chunks = (uint32_t)(moe_batch_max / chunk_size);
    if (moe_chunks < 1u) moe_chunks = 1u;
    if (moe_chunks > 1u) {
        uint32_t need = (uint32_t)(((size_t)mb_cap_needed + chunk_size - 1u) / chunk_size);
        if (moe_chunks > need) moe_chunks = need;
        if (moe_chunks < 1u) moe_chunks = 1u;
    }
    const uint32_t moe_batch = chunk_size * moe_chunks;
    const bool shrink_active = suffix_shrink && total_prompt_len >= 512u;

    for (uint32_t mb_start = 0; mb_start < n_tokens; mb_start += macroblock_size) {
        uint32_t mb_len = n_tokens - mb_start;
        if (mb_len > macroblock_size) mb_len = macroblock_size;
        const uint32_t mb_base_pos = base_pos + mb_start;
        const v41_mb_shrink_plan_t mb_plan = v41_suffix_shrink_mb_plan(
            base_pos, n_tokens, mb_start, mb_len,
            shrink_shift, false, final_mb_only, false);

        for (uint32_t il = 0; il < V41_LAYERS; ++il) {
            const bool shrink_layer = shrink_active && il >= 20u;
            uint32_t sh_loop_rel = 0u, sh_full_rel = 0u, sh_s_global = 0u;
            if (shrink_layer) {
                const v41_layer_shrink_bounds_t sb = v41_suffix_shrink_layer_bounds(
                    il, mb_base_pos, mb_len, mb_plan.total_eff, shrink_shift,
                    shrink_shift, mb_plan.mb_deferred, false, V41_SNAP_STRIDE, false, 0u);
                sh_s_global = sb.sh_s_global;
                sh_full_rel = sb.sh_full_rel;
                sh_loop_rel = sb.sh_loop_rel;
            }

            uint32_t moe_off = 0u;
            uint32_t batch_k = 0u;
            myds_v41_prefill_topk_reason_t batch_reason = MYDS_PREFILL_TOPK_SCHED_DISABLED;

            for (uint32_t c_start = sh_loop_rel; c_start < mb_len; ) {
                const uint32_t c_len = v41_prefill_chunk_len(
                    c_start, mb_len, chunk_size, mb_base_pos, shrink_shift,
                    false, shrink_layer, 0u, sh_full_rel);
                REQUIRE(c_len > 0u);
                const bool kv_only = shrink_layer && (c_start + c_len <= sh_full_rel);
                const bool kv_only_skip_win = v41_kv_only_skip_win(
                    kv_only, false, mb_base_pos, c_start, c_len, sh_s_global);
                if (kv_only_skip_win && !kv_source(il)) {
                    c_start += c_len;
                    continue;
                }
                if (!kv_only) {
                    const uint32_t abs_s = mb_base_pos + c_start;
                    const uint32_t abs_e = abs_s + c_len;
                    myds_v41_prefill_topk_reason_t reason = MYDS_PREFILL_TOPK_SCHED_DISABLED;
                    const uint32_t chunk_k = sched_on
                        ? myds_v41_gpu_moe_prefill_topk_schedule_select(
                              preset, 6u, total_prompt_len, il, abs_s, abs_e, &reason)
                        : 6u;

                    /* Safety invariants on every chunk */
                    if (sched_on) {
                        if (total_prompt_len < cfg.min_prompt_tokens) {
                            REQUIRE(chunk_k == 6u && reason == MYDS_PREFILL_TOPK_SCHED_SHORT);
                        }
                        if (il < cfg.protect_layers_lt) {
                            REQUIRE(chunk_k == 6u);
                        }
                        if (abs_e > tail_start) {
                            REQUIRE(chunk_k == 6u);
                        }
                    }

                    /* Batch uniformity invariant: no mixed k or mixed reason inside one MoE batch! */
                    if (moe_off == 0u) {
                        batch_k = chunk_k;
                        batch_reason = reason;
                    } else {
                        REQUIRE(chunk_k == batch_k);
                        REQUIRE(reason == batch_reason);
                    }
                    moe_off += c_len;
                    REQUIRE(moe_off <= moe_batch);

                    st.total_token_layers += c_len;
                    if (chunk_k == 4u) st.k4_token_layers += c_len;
                    else if (reason == MYDS_PREFILL_TOPK_SCHED_SHORT) st.k6_short_token_layers += c_len;
                    else if (reason == MYDS_PREFILL_TOPK_SCHED_LAYER) st.k6_layer_token_layers += c_len;
                    else if (reason == MYDS_PREFILL_TOPK_SCHED_TAIL)  st.k6_tail_token_layers += c_len;

                    uint32_t next_chunk_k = chunk_k;
                    if (sched_on && c_start + c_len < mb_len) {
                        const uint32_t next_c_start = c_start + c_len;
                        const uint32_t next_c_len = v41_prefill_chunk_len(
                            next_c_start, mb_len, chunk_size, mb_base_pos, shrink_shift,
                            false, shrink_layer, 0u, sh_full_rel);
                        next_chunk_k = myds_v41_gpu_moe_prefill_topk_schedule_select(
                            preset, 6u, total_prompt_len, il,
                            mb_base_pos + next_c_start, mb_base_pos + next_c_start + next_c_len,
                            NULL);
                    }
                    if (myds_v41_should_flush_moe_batch_k(
                            moe_off, moe_batch, c_start, c_len, mb_len, chunk_size,
                            chunk_k, next_chunk_k)) {
                        st.total_dispatches++;
                        if (sched_on) {
                            myds_v41_gpu_moe_prefill_topk_schedule_record(reason);
                            if (reason == MYDS_PREFILL_TOPK_SCHED_K4) st.k4_dispatches++;
                            else if (reason == MYDS_PREFILL_TOPK_SCHED_SHORT) st.k6_short_dispatches++;
                            else if (reason == MYDS_PREFILL_TOPK_SCHED_LAYER) st.k6_layer_dispatches++;
                            else if (reason == MYDS_PREFILL_TOPK_SCHED_TAIL)  st.k6_tail_dispatches++;
                        }
                        moe_off = 0u;
                    }
                }
                c_start += c_len;
            }
            /* Every layer must finish with a cleanly flushed MoE batch. */
            REQUIRE(moe_off == 0u);
        }
    }
    return st;
}

static void test_workloads_and_warm_appends(void) {
    clear_topk_env();

    /* 1) Short prompts: 512 and 2047 -> 0% k=4, 100% k6_protected_short */
    for (int preset = 1; preset <= 2; ++preset) {
        const uint32_t shorts[] = { 512u, 2047u };
        for (size_t i = 0; i < sizeof(shorts) / sizeof(*shorts); ++i) {
            const sched_sim_t s = simulate_prefill_schedule(
                preset, 0u, shorts[i], 2048u, 9u, 16384u, true, 512u, true);
            REQUIRE(s.k4_token_layers == 0u && s.k4_dispatches == 0u);
            REQUIRE(s.k6_short_token_layers == s.total_token_layers);
            REQUIRE(s.k6_short_dispatches == s.total_dispatches);
            REQUIRE(s.k6_layer_dispatches == 0u && s.k6_tail_dispatches == 0u);
        }
    }

    /* 2) Crucial warm-append check: warm_128k+310 and warm_256k+310 -> 0% k=4, 100% k=6! */
    for (int preset = 1; preset <= 2; ++preset) {
        const uint32_t prot_lt = (preset == 1) ? 4u : 2u;
        const uint32_t bases[] = { 128512u, 257536u };
        for (size_t i = 0; i < sizeof(bases) / sizeof(*bases); ++i) {
            const sched_sim_t s = simulate_prefill_schedule(
                preset, bases[i], 310u, 2048u, 9u, 16384u, true, 512u, true);
            REQUIRE(s.k4_token_layers == 0u && s.k4_dispatches == 0u);
            REQUIRE(s.k6_short_dispatches == 0u);
            REQUIRE(s.k6_layer_dispatches == prot_lt);
            REQUIRE(s.k6_tail_dispatches == V41_LAYERS - prot_lt);
            REQUIRE(s.k6_layer_token_layers + s.k6_tail_token_layers == s.total_token_layers);
        }
    }

    /* 3) warm_128k+2k:
     *    - chunk=2048: single chunk overlaps final 512 -> 100% k=6 (tail protected)
     *    - chunk=512:  3 interior chunks [128512..130048) at k=4, final 512 chunk at k=6,
     *                  and batch flush cleanly splits interior k=4 from tail k=6! */
    for (int preset = 1; preset <= 2; ++preset) {
        const uint32_t prot_lt = (preset == 1) ? 4u : 2u;
        const sched_sim_t w2k_c2048 = simulate_prefill_schedule(
            preset, 128512u, 2048u, 2048u, 9u, 16384u, true, 512u, true);
        REQUIRE(w2k_c2048.k4_dispatches == 0u && w2k_c2048.k4_token_layers == 0u);
        REQUIRE(w2k_c2048.k6_layer_dispatches == prot_lt);
        REQUIRE(w2k_c2048.k6_tail_dispatches == V41_LAYERS - prot_lt);

        const sched_sim_t w2k_c512 = simulate_prefill_schedule(
            preset, 128512u, 2048u, 512u, 9u, 16384u, true, 512u, true);
        REQUIRE(w2k_c512.k4_dispatches > 0u);
        REQUIRE(w2k_c512.k6_tail_dispatches == V41_LAYERS - prot_lt);
        REQUIRE(w2k_c512.k6_layer_dispatches == prot_lt);
    }

    /* 4) Cold 16K, 32K, 128K, 256K + real non-512-aligned 128,889 + warm_128k_to_256k (129,298) */
    const struct {
        const char *name;
        uint32_t base_pos;
        uint32_t n_tokens;
    } cases[] = {
        { "cold_16k",          0u,      16384u  },
        { "cold_32k",          0u,      32768u  },
        { "cold_128k",         0u,      131072u },
        { "cold_128k_real",    0u,      128889u },
        { "cold_256k",         0u,      262144u },
        { "warm_128k_to_256k", 128512u, 129298u },
    };

    for (int preset = 1; preset <= 2; ++preset) {
        printf("--- Preset %d expected k=4 MoE FLOP share (chunk=2048, moe_batch=16384, shrink_shift=512, final_mb_only=1) ---\n", preset);
        for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); ++i) {
            const sched_sim_t sh = simulate_prefill_schedule(
                preset, cases[i].base_pos, cases[i].n_tokens, 2048u, 9u, 16384u, true, 512u, true);
            const sched_sim_t nosh = simulate_prefill_schedule(
                preset, cases[i].base_pos, cases[i].n_tokens, 2048u, 9u, 16384u, false, 0u, false);

            const double pct_sh = 100.0 * (double)sh.k4_token_layers / (double)sh.total_token_layers;
            const double pct_nosh = 100.0 * (double)nosh.k4_token_layers / (double)nosh.total_token_layers;
            printf("  %-18s: suffix_shrink=%.2f%% (%llu/%llu tok-layers, disp k4=%llu layer=%llu tail=%llu) | unshrunk=%.2f%%\n",
                   cases[i].name, pct_sh,
                   (unsigned long long)sh.k4_token_layers, (unsigned long long)sh.total_token_layers,
                   (unsigned long long)sh.k4_dispatches, (unsigned long long)sh.k6_layer_dispatches,
                   (unsigned long long)sh.k6_tail_dispatches, pct_nosh);

            REQUIRE(sh.k4_dispatches > 0u);
            REQUIRE(sh.k6_short_dispatches == 0u);
            REQUIRE(sh.k6_layer_dispatches > 0u);
            REQUIRE(sh.k6_tail_dispatches > 0u);
            REQUIRE(sh.k4_token_layers + sh.k6_layer_token_layers + sh.k6_tail_token_layers == sh.total_token_layers);

            /* Verify exact coverage expectations for 128K/256K */
            if (cases[i].n_tokens >= 128889u) {
                if (preset == 1) {
                    REQUIRE(pct_sh > 77.0 && pct_sh < 80.0);
                    REQUIRE(pct_nosh > 87.5 && pct_nosh < 90.0);
                } else {
                    REQUIRE(pct_sh > 86.5 && pct_sh < 89.5);
                    REQUIRE(pct_nosh > 92.5 && pct_nosh < 95.0);
                }
            }
        }
    }
    printf("PASS workloads_and_warm_appends\n");
}

static void test_debug_state_counters_and_json(void) {
    clear_topk_env();
    myds_v41_gpu_moe_prefill_topk_schedule_counters(NULL, true);

    /* Default disabled state */
    myds_v41_gpu_moe_prefill_topk_schedule_counters_t c;
    myds_v41_gpu_moe_prefill_topk_schedule_counters(&c, false);
    REQUIRE(c.enabled == 0 && c.preset == 0);
    REQUIRE(c.k4_dispatches == 0u && c.k6_protected_short == 0u &&
            c.k6_protected_layer == 0u && c.k6_protected_tail == 0u);

    /* Enable preset 1 and drive simulated dispatches */
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "1", 1);
    (void)simulate_prefill_schedule(1, 0u, 512u, 2048u, 9u, 16384u, true, 512u, true);
    (void)simulate_prefill_schedule(1, 0u, 32768u, 2048u, 9u, 16384u, true, 512u, true);

    myds_v41_gpu_moe_prefill_topk_schedule_counters(&c, false);
    REQUIRE(c.enabled == 1 && c.preset == 1);
    REQUIRE(c.k6_protected_short == 40u); /* 512-tok short prompt across all 40 layers */
    REQUIRE(c.k6_protected_layer == 8u);  /* L0..L3 × 2 MoE batches of 16384 */
    REQUIRE(c.k4_dispatches == 39u);      /* L4..L19 × 2 (=32) + L20..L26 × 1 (=7) */
    REQUIRE(c.k6_protected_tail == 36u);  /* L4..L39 × 1 tail batch */

    /* Non-reset read must not clear counters */
    myds_v41_gpu_moe_prefill_topk_schedule_counters_t c2;
    myds_v41_gpu_moe_prefill_topk_schedule_counters(&c2, false);
    REQUIRE(c2.k4_dispatches == c.k4_dispatches &&
            c2.k6_protected_short == c.k6_protected_short &&
            c2.k6_protected_layer == c.k6_protected_layer &&
            c2.k6_protected_tail == c.k6_protected_tail);

    /* Verify /debug/state JSON serialization */
    char json[4096];
    myds_debug_state_json(json, sizeof(json), NULL);
    REQUIRE(strstr(json, "truncated") == NULL);
    const char *expected =
        "\"moe_prefill_topk_schedule\":{\"enabled\":1,\"preset\":1,"
        "\"k4_dispatches\":39,\"k6_protected_short\":40,"
        "\"k6_protected_layer\":8,\"k6_protected_tail\":36}";
    if (strstr(json, expected) == NULL) {
        fprintf(stderr, "FAIL: expected JSON fragment not found:\n  want: %s\n  got:  %s\n",
                expected, json);
        exit(1);
    }

    /* Reset clears counters and preset 2 reflects accurately */
    myds_v41_gpu_moe_prefill_topk_schedule_counters(&c, true);
    REQUIRE(c.k4_dispatches == 39u);
    setenv("MYDS_MOE_PREFILL_TOPK_SCHEDULE", "2", 1);
    myds_v41_gpu_moe_prefill_topk_schedule_counters(&c, false);
    REQUIRE(c.enabled == 1 && c.preset == 2);
    REQUIRE(c.k4_dispatches == 0u && c.k6_protected_short == 0u &&
            c.k6_protected_layer == 0u && c.k6_protected_tail == 0u);

    clear_topk_env();
    printf("PASS debug_state_counters_and_json (json_len=%zu)\n", strlen(json));
}

int main(void) {
    test_env_parsing_and_conflicts();
    test_kvdisk_identity_isolation();
    test_direct_policy_boundaries();
    test_workloads_and_warm_appends();
    test_debug_state_counters_and_json();
    printf("ALL test_moe_prefill_topk_schedule PASSED\n");
    return 0;
}
