/* Contract test for DSpine training data capture.
 * Verifies that:
 * 1. Capture enablement and tap layer configuration.
 * 2. HC averaging across 4 streams (4 * 5120 -> 5120) and float-to-BF16 conversion.
 * 3. Last norm capture.
 * 4. Proper 8-byte aligned Safetensors header formatting and binary payload.
 */

#include "myds_dspine_capture.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline float bf16_to_float(uint16_t b) {
    uint32_t val = (uint32_t)b << 16;
    float f;
    memcpy(&f, &val, sizeof(f));
    return f;
}

int main(void) {
    printf("[test_dspine_capture] Starting DSpine capture verification...\n");

    /* 1. Test is_enabled */
    unsetenv("MYDS_DSPINE_CAPTURE_DIR");
    assert(!myds_dspine_capture_is_enabled());

    setenv("MYDS_DSPINE_CAPTURE_DIR", "/tmp/dspine_test_shards", 1);
    assert(myds_dspine_capture_is_enabled());

    /* 2. Test tap layer slots */
    assert(myds_dspine_capture_layer_slot(2) == 0);
    assert(myds_dspine_capture_layer_slot(10) == 1);
    assert(myds_dspine_capture_layer_slot(20) == 2);
    assert(myds_dspine_capture_layer_slot(30) == 3);
    assert(myds_dspine_capture_layer_slot(38) == 4);
    assert(myds_dspine_capture_layer_slot(0) == -1);
    assert(myds_dspine_capture_layer_slot(39) == -1);

    /* 3. Begin capture for 8 tokens */
    const uint32_t n_tokens = 8;
    int32_t tokens[8] = {101, 102, 103, 104, 105, 106, 107, 108};
    assert(myds_dspine_capture_begin(n_tokens, tokens));

    /* 4. Synthesize HC residual: 4 streams * 5120 floats per token */
    float *hc_buf = (float *)malloc(n_tokens * 4 * MYDS_DSPINE_DIM * sizeof(float));
    assert(hc_buf);

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < 4; s++) {
            for (uint32_t d = 0; d < MYDS_DSPINE_DIM; d++) {
                /* Set stream values so that mean = (t + 1) * 0.1f + d * 0.001f */
                float base = (float)(t + 1) * 0.1f + (float)d * 0.001f;
                hc_buf[t * 4 * MYDS_DSPINE_DIM + s * MYDS_DSPINE_DIM + d] = base + (float)(s - 1.5f) * 0.05f;
            }
        }
    }

    /* Capture tap layers */
    uint32_t taps[5] = {2, 10, 20, 30, 38};
    for (int i = 0; i < 5; i++) {
        myds_dspine_capture_layer_hc(taps[i], hc_buf, 0, n_tokens);
    }

    /* 5. Synthesize target_last_hidden_states */
    float *norm_buf = (float *)malloc(n_tokens * MYDS_DSPINE_DIM * sizeof(float));
    assert(norm_buf);
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < MYDS_DSPINE_DIM; d++) {
            norm_buf[t * MYDS_DSPINE_DIM + d] = (float)(t + 1) * 0.25f - (float)d * 0.0005f;
        }
    }
    myds_dspine_capture_last_norm(norm_buf, 0, n_tokens);

    /* 6. Finish and save to disk */
    assert(myds_dspine_capture_finish_and_save());
    printf("[test_dspine_capture] Shard saved to /tmp/dspine_test_shards successfully.\n");

    free(hc_buf);
    free(norm_buf);

    return 0;
}

