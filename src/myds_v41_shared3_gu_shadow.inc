/* Included after v41_projection_shadow in myds_v41.c. Diagnostic only:
 * compare the original shared3 output against the actual scalar GU PSOs on
 * the same operands. Per-layer GPU joins invalidate acceptance and timing.
 * Keep the candidate outputs connected to the graph; no reference replaces
 * them. On discrepancy, preserve the inputs before a later layer reuses them.
 */
static int v41_shared3_gu_shadow(v41_batch2_workspace_t *ws,
    const myds_layer_t *base, const myds_model_t *m,
    uint32_t pos, uint32_t layer) {
    const char *dir = getenv("MYDS_DEPTH2_SHARED3_SHADOW");
    if (!dir || !*dir) return 1;
    if (*dir != '/' || !ws || !base || !m ||
        base->shared_gate_type != base->shared_up_type ||
        (base->shared_gate_type != GGML_TYPE_Q4_K &&
         base->shared_gate_type != GGML_TYPE_Q8_0)) return 0;

    const uint32_t rows = 2304u, width = V41_DIM;
    myds_metal_buffer_t *ref[3] = {NULL, NULL, NULL};
    int ok = 1;
    for (unsigned t = 0; t < 3 && ok; ++t) {
        ref[t] = myds_metal_buffer_alloc((size_t)rows * sizeof(float));
        if (!ref[t]) { ok = 0; break; }
        ok = myds_v41_shared_gate_up_swiglu(ref[t],
            base->shared_gate, base->shared_up, ws->v_ffn_norm[t],
            rows, width, base->shared_gate_type, base->shared_up_type,
            m->swiglu_clamp_exp);
    }
    /* Join every encoded reference, including allocation/dispatch failure,
     * before reading or freeing it. commit also drains older submissions. */
    if (!myds_metal_commit_batch()) ok = 0;

    unsigned first_row = 3u;
    size_t first_index = 0;
    if (ok) for (unsigned t = 0; t < 3 && first_row == 3u; ++t) {
        const uint32_t *candidate = myds_metal_buffer_data(ws->v_shared_swiglu[t]);
        const uint32_t *scalar = myds_metal_buffer_data(ref[t]);
        if (!candidate || !scalar) { ok = 0; break; }
        for (size_t i = 0; i < rows; ++i)
            if (candidate[i] != scalar[i]) {
                first_row = t; first_index = i; break;
            }
    }
    if (first_row < 3u) {
        if (mkdir(dir, 0700)) {
            MYDS_LOG_E("SHARED3_SHADOW GU requires a new payload directory: %s", dir);
            for (unsigned t = 0; t < 3; ++t) myds_metal_buffer_free(ref[t]);
            exit(87);
        }
        const size_t row_bytes = base->shared_gate_type == GGML_TYPE_Q4_K
            ? (size_t)(width / 256u) * 144u : (size_t)(width / 32u) * 34u;
        int saved = v41_shared3_shadow_write(dir, "gate.bin",
            myds_metal_buffer_data((myds_metal_buffer_t *)base->shared_gate), rows * row_bytes);
        saved &= v41_shared3_shadow_write(dir, "up.bin",
            myds_metal_buffer_data((myds_metal_buffer_t *)base->shared_up), rows * row_bytes);
        for (unsigned t = 0; t < 3; ++t) {
            char name[64];
            snprintf(name, sizeof(name), "input-%u.f32", t);
            saved &= v41_shared3_shadow_write(dir, name,
                myds_metal_buffer_data(ws->v_ffn_norm[t]), (size_t)width * sizeof(float));
            snprintf(name, sizeof(name), "candidate-%u.f32", t);
            saved &= v41_shared3_shadow_write(dir, name,
                myds_metal_buffer_data(ws->v_shared_swiglu[t]), (size_t)rows * sizeof(float));
            snprintf(name, sizeof(name), "scalar-%u.f32", t);
            saved &= v41_shared3_shadow_write(dir, name,
                myds_metal_buffer_data(ref[t]), (size_t)rows * sizeof(float));
        }
        const uint32_t *candidate = myds_metal_buffer_data(ws->v_shared_swiglu[first_row]);
        const uint32_t *scalar = myds_metal_buffer_data(ref[first_row]);
        char metadata[1024];
        const int n = snprintf(metadata, sizeof(metadata),
            "{\"provider\":\"gu\",\"pos\":%u,\"layer\":%u,\"type\":%u,\"up_type\":%u,"
            "\"rows\":%u,\"width\":%u,\"clamp\":%.9g,\"first_row\":%u,\"first_index\":%zu,"
            "\"candidate_bits\":%u,\"scalar_bits\":%u,\"performance_measurement\":false,"
            "\"acceptance_evidence\":false}\n",
            pos, layer, (unsigned)base->shared_gate_type, (unsigned)base->shared_up_type,
            rows, width, (double)m->swiglu_clamp_exp, first_row, first_index,
            candidate[first_index], scalar[first_index]);
        saved &= n > 0 && (size_t)n < sizeof(metadata) &&
            v41_shared3_shadow_write(dir, "metadata.json", metadata, (size_t)n);
        MYDS_LOG_E("SHARED3_SHADOW first GU discrepancy pos=%u layer=%u row=%u index=%zu bits=%08x/%08x saved=%d dir=%s",
            pos, layer, first_row, first_index,
            candidate[first_index], scalar[first_index], saved, dir);
        for (unsigned t = 0; t < 3; ++t) myds_metal_buffer_free(ref[t]);
        /* A complete diagnostic payload exits86; every save failure exits87. */
        exit(saved ? 86 : 87);
    }
    for (unsigned t = 0; t < 3; ++t) myds_metal_buffer_free(ref[t]);
    return ok && myds_metal_begin_batch();
}
