#include "myds_cpu.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

float myds_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            /* A binary16 subnormal is mantissa * 2^-24. Normalize it before
             * rebiasing; merely shifting its exponent gives the wrong scale. */
            exponent = 113;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

void myds_rope_tail_inplace(float *x, uint32_t n_head, uint32_t head_dim,
                            uint32_t n_rot, uint32_t pos, float freq_base,
                            float freq_scale, bool inverse) {
    if (!x || n_head == 0 || n_rot == 0 || n_rot > head_dim || (n_rot & 1)) return;
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float theta_extrap = (float)pos;

    /* Every head has the same angles. Moving this loop outside the heads
     * removes n_head-fold redundant sin/cos calls without a lookup table or
     * a different floating-point recurrence. */
    for (uint32_t i = 0; i < n_rot; i += 2) {
        const float theta = freq_scale * theta_extrap;
        const float c = cosf(theta);
        const float s = sin_sign * sinf(theta);
        for (uint32_t h = 0; h < n_head; h++) {
            float *tail = x + (size_t)h * head_dim + n_nope;
            const float x0 = tail[i];
            const float x1 = tail[i + 1];
            tail[i] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
        }
        theta_extrap *= theta_scale;
    }
}
