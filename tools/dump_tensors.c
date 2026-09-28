#include "myds_gguf.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <path.gguf>\n", argv[0]);
        return 1;
    }
    gguf_file_t *gf = gguf_open(argv[1]);
    if (!gf) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", argv[1]);
        return 1;
    }
    printf("Architecture: %s\n", gguf_get_str(gf, "general.architecture", "unknown"));
    printf("Total tensors: %llu, KV count: %llu\n\n",
           (unsigned long long)gf->tensor_count,
           (unsigned long long)gf->metadata_kv_count);

    printf("--- Sample Tensors (blk.0.* and top-level) ---\n");
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const char *name = gf->tensors[i].name;
        if (strstr(name, "blk.0.") || strstr(name, "token") || strstr(name, "output")) {
            printf("  [%3llu] %-36s (type=%2d, size=%10zu bytes)\n",
                   (unsigned long long)i, name, gf->tensors[i].type, gf->tensors[i].size_bytes);
        }
    }

    /* Check myds expected names */
    printf("\n--- Myds Expected Names in blk.0 ---\n");
    const char *expected[] = {
        "token_embd.weight", "output_norm.weight", "output.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight", "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight",
        NULL
    };
    for (int j = 0; expected[j]; j++) {
        const gguf_tensor_t *t = gguf_find_tensor(gf, expected[j]);
        printf("  %-32s : %s\n", expected[j], t ? "FOUND" : "MISSING");
    }

    /* Check V4.1 MLA / DSpark Expected Names in blk.0 */
    printf("\n--- DeepSeek V4.1 MLA / DSpark Expected Names in blk.0 ---\n");
    const char *mla_expected[] = {
        "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight", "blk.0.attn_q_b.weight",
        "blk.0.attn_kv.weight", "blk.0.attn_kv_a_norm.weight", "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight",
        "blk.0.attn_output_a.weight", "blk.0.attn_output_b.weight",
        "blk.0.attn_sinks.weight", "blk.0.exp_probs_b.bias",
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
        NULL
    };
    for (int j = 0; mla_expected[j]; j++) {
        const gguf_tensor_t *t = gguf_find_tensor(gf, mla_expected[j]);
        printf("  %-32s : %s\n", mla_expected[j], t ? "FOUND" : "MISSING");
    }

    gguf_close(gf);
    return 0;
}
