#ifndef MYDS_V41_DECODE_H
#define MYDS_V41_DECODE_H

#include "myds_metal.h"
#include <stdbool.h>

/* Single-token output-B/shared-expert projection only. The caller retains its
 * original barrier placement. Q8 may combine GEMV and the original BF16 store;
 * other formats, trace and MYDS_DECODE_Q8_FUSE_BF16=0 retain separate dispatches.
 * Output views at offsets 4/8/12 modulo 16 retain the original separate path
 * and its BF16 API's alignment rejection. They are not supported BF16 outputs.
 * A failed fused call is propagated and never retried with another kernel.
 * The environment switch is read once per process, shared by both call sites.
 * Existing MYDS_QKV_NR0 selects NR4 (default) or NR2 in the Metal implementation.
 * Missing fused PSOs are handled by that implementation's original fallback.
 */
int myds_v41_decode_gemv_bf16(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t rows, uint32_t width, ggml_type type, bool trace_on);

#endif
