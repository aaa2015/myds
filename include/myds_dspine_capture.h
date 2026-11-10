#ifndef MYDS_DSPINE_CAPTURE_H
#define MYDS_DSPINE_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYDS_DSPINE_MAX_TAPS 8
#define MYDS_DSPINE_DIM 5120

/* Check if capture is enabled (via MYDS_DSPINE_CAPTURE_DIR) */
bool myds_dspine_capture_is_enabled(void);

/* Check if layer_idx is one of our tapped target layers; returns slot index or -1 */
int  myds_dspine_capture_layer_slot(uint32_t layer_idx);

/* Optional per-sequence override for loss_mask and deterministic shard filename */
void myds_dspine_capture_set_override(const float *loss_mask, uint32_t n_tokens,
                                      const char *exact_filepath);

/* Begin capturing for a sequence of n_tokens */
bool myds_dspine_capture_begin(uint32_t n_tokens, const int32_t *tokens);

/* Capture a tapped layer's output (4 HC streams averaged into 5120 floats -> BF16)
 * residual_hc has 4 * 5120 floats per token.
 * count is the number of tokens in this block.
 * offset is the token index offset within the sequence. */
void myds_dspine_capture_layer_hc(uint32_t layer_idx, const float *residual_hc,
                                  uint32_t offset, uint32_t count);

/* Capture target last hidden states (after output_norm)
 * x_norm has 5120 floats per token. */
void myds_dspine_capture_last_norm(const float *x_norm, uint32_t offset, uint32_t count);

/* Finish and save to .safetensors */
bool myds_dspine_capture_finish_and_save(void);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_DSPINE_CAPTURE_H */
