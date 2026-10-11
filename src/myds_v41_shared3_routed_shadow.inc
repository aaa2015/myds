/* Included after v41_shared3_shadow_write and the batch workspace definition.
 * Explicit same-operand diagnostic: completed GPU joins change scheduling and
 * forbid using this run as a state/fault acceptance or performance sample.
 * Candidate MID/output stay untouched. Only independent reference buffers are
 * written; the first discrepancy terminates after saving a replay payload. */
static int v41_shared3_routed_shadow(v41_batch2_workspace_t *ws,
    const myds_layer_t *base, const myds_model_t *m, uint32_t pos, uint32_t layer) {
    const char *dir = getenv("MYDS_DEPTH2_SHARED3_SHADOW");
    if (!dir || !*dir) return 1;
    const char *mm = getenv("MYDS_MOE_MM_MIN_TOKENS");
    if (*dir != '/' || !ws || !base || !m || !mm || strcmp(mm, "0")) {
        MYDS_LOG_E("SHARED3_SHADOW routed requires absolute output path and MYDS_MOE_MM_MIN_TOKENS=0");
        return 0;
    }
    /* With n_tok=3 the old provider cannot enter the two-row dedup path.
     * MM_MIN=0 selects the original scalar IQ2 GU / Q2 sum6 PSOs for each
     * row, with their original per-row expert/block association. */
    enum { RSH_GATE, RSH_UP, RSH_MID, RSH_DOWN, RSH_OUT, RSH_COUNT };
    myds_metal_buffer_t *ref[RSH_COUNT] = {NULL};
    const myds_metal_buffer_t *geometry[RSH_COUNT] = {
        ws->moe_gate_b2, ws->moe_up_b2, ws->moe_mid_b2, ws->moe_down_b2, ws->moe_out_b2};
    const size_t expected[RSH_COUNT] = {
        3u * 6u * 2304u * sizeof(float), 3u * 6u * 2304u * sizeof(float),
        3u * 6u * 2304u * sizeof(float), 3u * 6u * V41_DIM * sizeof(float),
        3u * V41_DIM * sizeof(float)};
    size_t bytes[RSH_COUNT] = {0};
    int ok = base->ffn_gate_exps_type == GGML_TYPE_IQ2_XXS &&
        base->ffn_up_exps_type == GGML_TYPE_IQ2_XXS && base->ffn_down_exps_type == GGML_TYPE_Q2_K;
    for (unsigned i = 0; i < RSH_COUNT && ok; ++i) {
        bytes[i] = myds_metal_buffer_size(geometry[i]);
        if (bytes[i] != expected[i]) { ok = 0; break; }
        ref[i] = myds_metal_buffer_alloc(bytes[i]);
        if (!ref[i]) { ok = 0; break; }
    }
    if (ok) {
        const myds_metal_buffer_t *deps[] = {ws->ffn_norm_b2, ws->selected_ids_b2, ws->selected_w_b2};
        myds_metal_barrier_buffers(deps, 3);
        ok = myds_metal_routed_moe_batch(ref[RSH_OUT], ws->ffn_norm_b2,
            base->ffn_gate_exps, base->ffn_up_exps, base->ffn_down_exps,
            ws->selected_ids_b2, ws->selected_w_b2, ref[RSH_GATE], ref[RSH_UP],
            ref[RSH_MID], ref[RSH_DOWN], V41_DIM, 2304, 384, 6, 3,
            base->ffn_gate_exps_type, base->ffn_up_exps_type, base->ffn_down_exps_type,
            m->swiglu_clamp_exp);
    }
    /* Even a partial reference-encoding failure must join before release.
     * A second blocking drain after a reported commit error is cleanup only;
     * neither return value can turn the failed diagnostic into a success. */
    const int committed = myds_metal_commit_batch();
    if (!committed) {
        const int drained = myds_metal_synchronize();
        MYDS_LOG_E("SHARED3_SHADOW routed commit failed (cleanup drain=%d)", drained);
        ok = 0;
    }
    if (!ok) {
        for (unsigned i = 0; i < RSH_COUNT; ++i) myds_metal_buffer_free(ref[i]);
        return 0;
    }
    const uint32_t *candidate[2] = {myds_metal_buffer_data(ws->moe_mid_b2),
                                    myds_metal_buffer_data(ws->moe_out_b2)};
    const uint32_t *scalar[2] = {myds_metal_buffer_data(ref[RSH_MID]),
                                 myds_metal_buffer_data(ref[RSH_OUT])};
    const size_t counts[2] = {bytes[RSH_MID] / sizeof(float), bytes[RSH_OUT] / sizeof(float)};
    size_t first[2] = {counts[0], counts[1]}, differences[2] = {0, 0};
    for (unsigned role = 0; role < 2; ++role)
        for (size_t i = 0; i < counts[role]; ++i)
            if (candidate[role][i] != scalar[role][i]) {
                if (!differences[role]) first[role] = i;
                ++differences[role];
            }
    if (differences[0] || differences[1]) {
        if (mkdir(dir, 0700)) {
            MYDS_LOG_E("SHARED3_SHADOW routed requires a new payload directory: %s", dir);
            for (unsigned i = 0; i < RSH_COUNT; ++i) myds_metal_buffer_free(ref[i]);
            exit(87);
        }
        int saved = 1;
        saved &= v41_shared3_shadow_write(dir, "input.f32", myds_metal_buffer_data(ws->ffn_norm_b2),
            3u * V41_DIM * sizeof(float));
        saved &= v41_shared3_shadow_write(dir, "ids.i32", myds_metal_buffer_data(ws->selected_ids_b2),
            18u * sizeof(int32_t));
        saved &= v41_shared3_shadow_write(dir, "weights.f32", myds_metal_buffer_data(ws->selected_w_b2),
            18u * sizeof(float));
        saved &= v41_shared3_shadow_write(dir, "candidate-mid.f32", candidate[0], bytes[RSH_MID]);
        saved &= v41_shared3_shadow_write(dir, "scalar-mid.f32", scalar[0], bytes[RSH_MID]);
        saved &= v41_shared3_shadow_write(dir, "candidate-out.f32", candidate[1], bytes[RSH_OUT]);
        saved &= v41_shared3_shadow_write(dir, "scalar-out.f32", scalar[1], bytes[RSH_OUT]);
        const int32_t *ids = myds_metal_buffer_data(ws->selected_ids_b2);
        const size_t gate_expert_bytes = (size_t)2304u * (V41_DIM / 256u) * 66u;
        const size_t down_expert_bytes = (size_t)V41_DIM * (2304u / 256u) * 84u;
        const myds_metal_buffer_t *blobs[] = {base->ffn_gate_exps, base->ffn_up_exps, base->ffn_down_exps};
        const size_t expert_bytes[] = {gate_expert_bytes, gate_expert_bytes, down_expert_bytes};
        const char *prefix[] = {"gate", "up", "down"};
        unsigned unique_experts = 0;
        for (unsigned pair = 0; pair < 18u; ++pair) {
            const int32_t expert = ids[pair];
            if (expert < 0 || expert >= 384) { saved = 0; continue; }
            bool duplicate = false;
            for (unsigned previous = 0; previous < pair; ++previous)
                if (ids[previous] == expert) { duplicate = true; break; }
            if (duplicate) continue;
            ++unique_experts;
            for (unsigned kind = 0; kind < 3; ++kind) {
                char name[64];
                snprintf(name, sizeof(name), "%s-%d.bin", prefix[kind], expert);
                const size_t offset = (size_t)expert * expert_bytes[kind];
                if (offset > myds_metal_buffer_size(blobs[kind]) ||
                    expert_bytes[kind] > myds_metal_buffer_size(blobs[kind]) - offset) {
                    saved = 0; continue;
                }
                saved &= v41_shared3_shadow_write(dir, name,
                    (const char *)myds_metal_buffer_data((myds_metal_buffer_t *)blobs[kind]) + offset,
                    expert_bytes[kind]);
            }
        }
        const unsigned role = differences[0] ? 0u : 1u;
        const size_t index = first[role];
        const unsigned row = (unsigned)(index / (role == 0u ? 6u * 2304u : V41_DIM));
        const unsigned slot = role == 0u ? (unsigned)((index / 2304u) % 6u) : 6u;
        const size_t channel = index % (role == 0u ? 2304u : V41_DIM);
        char metadata[2048];
        const int n = snprintf(metadata, sizeof(metadata),
            "{\"provider\":\"routed\",\"pos\":%u,\"layer\":%u,\"dim\":%u,\"expert_dim\":2304,"
            "\"experts\":384,\"active\":6,\"rows\":3,\"gate_type\":%u,\"up_type\":%u,\"down_type\":%u,"
            "\"clamp\":%.9g,\"first_role\":\"%s\",\"first_row\":%u,\"first_slot\":%u,\"first_index\":%zu,"
            "\"first_flat_index\":%zu,\"candidate_bits\":%u,\"scalar_bits\":%u,"
            "\"mid_differences\":%zu,\"mid_first_flat_index\":%zu,\"out_differences\":%zu,\"out_first_flat_index\":%zu,"
            "\"unique_experts\":%u,\"gate_expert_bytes\":%zu,\"down_expert_bytes\":%zu,"
            "\"reference\":\"original_scalar_psos_batch3\",\"moe_mm_min_tokens\":0,\"performance_measurement\":false}\n",
            pos, layer, V41_DIM, (unsigned)base->ffn_gate_exps_type, (unsigned)base->ffn_up_exps_type,
            (unsigned)base->ffn_down_exps_type, m->swiglu_clamp_exp, role == 0u ? "mid" : "out",
            row, slot, channel, index, candidate[role][index], scalar[role][index],
            differences[0], first[0], differences[1], first[1], unique_experts, gate_expert_bytes, down_expert_bytes);
        saved &= n > 0 && (size_t)n < sizeof(metadata) &&
            v41_shared3_shadow_write(dir, "metadata.json", metadata, (size_t)n);
        MYDS_LOG_E("SHARED3_SHADOW first routed discrepancy pos=%u layer=%u role=%s row=%u slot=%u channel=%zu bits=%08x/%08x mid_diff=%zu out_diff=%zu saved=%d dir=%s",
            pos, layer, role == 0u ? "mid" : "out", row, slot, channel,
            candidate[role][index], scalar[role][index], differences[0], differences[1], saved, dir);
        for (unsigned i = 0; i < RSH_COUNT; ++i) myds_metal_buffer_free(ref[i]);
        exit(saved ? 86 : 87);
    }
    for (unsigned i = 0; i < RSH_COUNT; ++i) myds_metal_buffer_free(ref[i]);
    return myds_metal_begin_batch();
}
