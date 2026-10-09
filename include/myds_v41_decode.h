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
/* Query the same process-once fuse policy used by the scalar wrapper. */
bool myds_v41_decode_q8_fuse_bf16_enabled(void);

/* Experimental, local QKV/Indexer encoder scope. MYDS_QKV_CONCURRENT=1
 * explicitly opts in; the default and trace path keep the caller's mode.
 * End must run on success and failure, before draining/freeing GPU work.
 * It closes the dependency region and restores both caller settings. */
typedef struct {
    int auto_barrier, concurrent;
    bool active;
} myds_v41_decode_qkv_scope_t;
void myds_v41_decode_qkv_scope_begin(myds_v41_decode_qkv_scope_t *scope, bool trace_on);
void myds_v41_decode_qkv_scope_end(myds_v41_decode_qkv_scope_t *scope);

#endif
