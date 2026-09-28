#include "myds_cpu.h"

#include <math.h>
#include <stddef.h>

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
