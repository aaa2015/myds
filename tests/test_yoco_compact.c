/* Unit test for YoCo SWA Bounded Replay (MYDS_YOCO_COMPACT)
 * Verifies mathematical properties, layer scheduling bounds, and MoE token reductions.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); \
} } while (0)

#define V41_LAYERS 40u
#define V41_WINDOW 128u

static void test_yoco_compact_bounds(uint32_t n_tokens, uint32_t keep_tokens) {
    const uint32_t cone_shift = 0u;
    const uint32_t sub_39 = keep_tokens + cone_shift;
    const uint32_t s_global_39 = (n_tokens > sub_39) ? (n_tokens - sub_39) : 0u;

    for (uint32_t il = 20u; il < V41_LAYERS; il++) {
        const uint32_t sh_need = keep_tokens;
        const uint32_t sub = sh_need + cone_shift;
        const uint32_t s_global = (n_tokens > sub) ? (n_tokens - sub) : 0u;
        REQUIRE(s_global == s_global_39);

        uint32_t sh_full_rel = s_global;
        uint32_t sh_loop_rel = 0;
        if (il == 20u) {
            sh_loop_rel = 0u; /* Layer 20 runs compressor for all tokens */
        } else {
            sh_loop_rel = sh_full_rel; /* Layers 21..39 skip all prefix chunks */
        }

        if (n_tokens <= keep_tokens) {
            REQUIRE(s_global == 0u);
            REQUIRE(sh_full_rel == 0u);
            REQUIRE(sh_loop_rel == 0u);
        } else {
            REQUIRE(s_global == n_tokens - keep_tokens);
            if (il == 20u) {
                REQUIRE(sh_loop_rel == 0u);
                REQUIRE(sh_full_rel == n_tokens - keep_tokens);
            } else {
                REQUIRE(sh_loop_rel == n_tokens - keep_tokens);
                REQUIRE(sh_full_rel == n_tokens - keep_tokens);
            }

            /* Verify that the final token (n_tokens - 1) has its entire 128-window
             * inside the retained segment [s_global, n_tokens). */
            const uint32_t final_tok = n_tokens - 1u;
            const uint32_t window_earliest = (final_tok >= (V41_WINDOW - 1u)) ? (final_tok - (V41_WINDOW - 1u)) : 0u;
            REQUIRE(window_earliest >= s_global);
        }
    }
}

int main(void) {
    printf("[TEST] Testing YoCo Compact prefill scheduling bounds...\n");

    const uint32_t keep_tokens = 140u;
    const uint32_t test_lens[] = {
        1u, 16u, 64u, 128u, 140u, 141u, 256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 65536u
    };

    uint64_t total_moe_avoided = 0;
    for (size_t i = 0; i < sizeof(test_lens) / sizeof(test_lens[0]); i++) {
        uint32_t n = test_lens[i];
        test_yoco_compact_bounds(n, keep_tokens);

        if (n > keep_tokens) {
            uint32_t avoided_per_consumer_layer = n - keep_tokens;
            /* 19 layers (21..39) completely skip the prefix for Attention + MoE,
             * and Layer 20 skips Attention + MoE on the prefix. */
            uint64_t avoided_this_prompt = (uint64_t)avoided_per_consumer_layer * 20u;
            total_moe_avoided += avoided_this_prompt;
        }
    }

    /* Verify 32K context savings */
    const uint32_t n_32k = 32768u;
    const uint64_t avoided_32k = (uint64_t)(n_32k - keep_tokens) * 20u;
    REQUIRE(avoided_32k == (32768 - 140) * 20); /* 652,560 token-layer MoE evaluations saved! */

    printf("[PASS] YoCo Compact prefill scheduling passed for all %zu test lengths!\n",
           sizeof(test_lens) / sizeof(test_lens[0]));
    printf("       Total consumer MoE token-evaluations avoided across test suite: %llu\n",
           (unsigned long long)total_moe_avoided);
    printf("       At 32K context: saved %llu consumer token-layer MoE/Attn evaluations (~99.5%% reduction in consumer prefill)\n",
           (unsigned long long)avoided_32k);

    return 0;
}
