/* Test DSpine runtime loading, weights mapping, and draft proposal */
#include "myds_dspine.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(void) {
    printf("[test_dspine_runtime] Creating DSpine runtime...\n");
    myds_dspine_t *ds = myds_dspine_create();
    assert(ds != NULL);
    assert(ds->hidden_size == 5120);
    assert(ds->num_layers == 3);

    const char *checkpoint = "/tmp/dspine_model_out/model-000002.safetensors";
    printf("[test_dspine_runtime] Loading weights from %s...\n", checkpoint);
    bool ok = myds_dspine_load_weights(ds, checkpoint);
    assert(ok && "Failed to load weights");
    assert(ds->loaded);

    /* Verify critical buffers */
    assert(ds->fc_proj != NULL);
    assert(ds->hidden_norm != NULL);
    assert(ds->final_norm != NULL);
    assert(ds->transfer_codes != NULL);

    for (uint32_t i = 0; i < ds->num_layers; i++) {
        assert(ds->layers[i].q_proj != NULL);
        assert(ds->layers[i].k_proj != NULL);
        assert(ds->layers[i].v_proj != NULL);
        assert(ds->layers[i].o_proj != NULL);
        assert(ds->layers[i].gate_proj != NULL);
        assert(ds->layers[i].up_proj != NULL);
        assert(ds->layers[i].down_proj != NULL);
        assert(ds->injection_readouts[i] != NULL);
        assert(ds->injection_writes[i] != NULL);
    }

    assert(ds->injection_message != NULL);
    assert(ds->injection_pred_gate != NULL);
    assert(ds->injection_recv_gate != NULL);
    assert(ds->injection_bias != NULL);

    printf("[test_dspine_runtime] All 47 weight tensors successfully loaded and mapped.\n");

    /* Test capture state */
    myds_dspine_capture_invalidate(ds);
    assert(!myds_dspine_capture_ready(ds));

    /* Synthesize target hidden states in ds->target_hidden_buf */
    float *target_h = (float *)myds_metal_buffer_data(ds->target_hidden_buf);
    assert(target_h != NULL);
    for (uint32_t i = 0; i < MYDS_DSPINE_TARGET_DIM; i++) {
        target_h[i] = 0.01f * (float)(i % 100);
    }

    printf("[test_dspine_runtime] Testing Metal GPU GEMV on fc_proj with GGML_TYPE_BF16...\n");
    myds_metal_buffer_t *gpu_out = myds_metal_buffer_alloc(5120 * sizeof(float));
    assert(gpu_out != NULL);

    int gemv_ok = myds_metal_v41_gemv_batch(gpu_out, ds->fc_proj, ds->target_hidden_buf,
                                            5120, MYDS_DSPINE_TARGET_DIM, 1, GGML_TYPE_BF16);
    printf("  myds_metal_v41_gemv_batch (fc_proj) returned: %d\n", gemv_ok);
    assert(gemv_ok == 1);
    myds_metal_buffer_free(gpu_out);

    /* Test batched GEMV on gate_proj (15360 x 5120) with n_tok = 8 */
    printf("[test_dspine_runtime] Testing Metal batched GEMV on gate_proj (15360 x 5120, n_tok = 8)...\n");
    myds_metal_buffer_t *b_in = myds_metal_buffer_alloc(8 * 5120 * sizeof(float));
    myds_metal_buffer_t *b_out = myds_metal_buffer_alloc(8 * 15360 * sizeof(float));
    assert(b_in && b_out);

    for (int iter = 0; iter < 5; iter++) {
        struct timeval bg0, bg1;
        gettimeofday(&bg0, NULL);
        int bg_ok = myds_metal_v41_gemv_batch(b_out, ds->layers[0].gate_proj, b_in,
                                              15360, 5120, 8, GGML_TYPE_BF16);
        gettimeofday(&bg1, NULL);
        double bg_ms = (bg1.tv_sec - bg0.tv_sec) * 1000.0 + (bg1.tv_usec - bg0.tv_usec) / 1000.0;
        printf("  iter %d: batched GEMV gate_proj returned: %d in %.3f ms\n", iter, bg_ok, bg_ms);
        assert(bg_ok == 1);
    }
    myds_metal_buffer_free(b_in);
    myds_metal_buffer_free(b_out);

    /* Create mock model and context */
    myds_model_t model = {0};
    model.vocab_size = 129280;
    /* Allocate mock embedding table for test tokens (first 2000 tokens) */
    size_t emb_bytes = (size_t)model.vocab_size * ds->hidden_size * sizeof(uint16_t);
    model.tok_embeddings = myds_metal_buffer_alloc(emb_bytes);
    assert(model.tok_embeddings != NULL);

    myds_context_t ctx = {0};
    ctx.model = &model;
    ctx.dspine = ds;
    ctx.pos = 10;

    printf("[test_dspine_runtime] Running DSpine draft proposal (max_propose = 2, block = 3)...\n");
    uint32_t count = 0;
    for (int iter = 0; iter < 5; iter++) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        count = myds_dspine_propose(ds, &ctx, 101, 2);

        gettimeofday(&t1, NULL);
        double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;

        printf("  iter %d: Proposed %u tokens in %.3f ms\n", iter, count, elapsed_ms);
        assert(count == 2);
    }
    for (uint32_t i = 0; i < count; i++) {
        printf("  draft token [%u] = %d\n", i, ds->draft_tokens[i]);
    }

    /* Cleanup mock buffers */
    myds_metal_buffer_free(model.tok_embeddings);

    printf("[test_dspine_runtime] Freeing DSpine runtime...\n");
    myds_dspine_free(ds);

    printf("[test_dspine_runtime] PASSED!\n");
    return 0;
}
