#ifndef MYDS_CPU_H
#define MYDS_CPU_H

#include <stdbool.h>
#include <stdint.h>

/* Decode IEEE 754 binary16 bits, preserving signed zero, infinities and NaN
 * payloads. GGML_TYPE_F16 uses binary16, not BF16. */
float myds_fp16_to_fp32(uint16_t h);

/* Tail RoPE shared by decode and prefill. Angles are evaluated once per
 * position/pair, with the same recurrence and per-element arithmetic as the
 * original head-major implementation. Invalid rotation shapes are a no-op. */
void myds_rope_tail_inplace(float *x, uint32_t n_head, uint32_t head_dim,
                            uint32_t n_rot, uint32_t pos, float freq_base,
                            float freq_scale, bool inverse);

#endif
