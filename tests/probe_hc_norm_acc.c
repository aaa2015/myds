/* Dump the pre-BF16-rounding accumulator of kernel_myds_v41_hc_sum_rmsnorm.
 *
 * Why this exists (ledger §120): the 0-ULP logits comparison proved blind to two
 * classes of shader change because the kernel rounds to BF16 before the value
 * reaches the model. `reorder` (accumulating the four HC terms c3..c0 instead of
 * c0..c3) produced bit-identical final logits, so a logits-only gate would pass
 * it. Whether that is safe depends on how far the reorder moves the pre-rounding
 * value relative to the BF16 quantum -- close to the quantum means another input
 * could tip it over a rounding boundary.
 *
 * This probe isolates the kernel: synthetic inputs, no model, no weights, so it
 * runs anywhere with a Metal device and cannot disturb a running service.
 *
 * The library is loaded from the working directory (myds_metal_init searches
 * "./bin/myds.metallib" first), and the resolved path is printed so the caller can
 * assert which arm actually ran -- that mistake already cost one false PASS.
 *
 * When built against a library whose `sum_out[i00] = v` was patched to
 * `sum_out[i00] = acc`, the sum buffer holds the raw pre-rounding accumulator.
 * The patch is applied identically to every arm by tools/hc_norm_acc_margin.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "myds_metal.h"

static void fill(myds_metal_buffer_t *b, unsigned seed) {
    float *p = myds_metal_buffer_data(b);
    const size_t n = myds_metal_buffer_size(b) / 4;
    for (size_t i = 0; i < n; i++) {
        /* Deterministic, mixed magnitude, both signs: rounding margins are
         * magnitude dependent and a single-scale fill would hide that. */
        const float mag = (float)(1u + ((uint32_t)(i * 2654435761u + seed) % 97u)) / 64.0f;
        const float sign = ((i + seed) & 1u) ? 1.0f : -1.0f;
        p[i] = sign * mag * ((float)((int)((i * 7u + seed) % 31u) - 15) / 16.0f);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.f32> [dim] [n_tok] [tag]\n", argv[0]);
        return 2;
    }
    const char *out_path = argv[1];
    const uint32_t dim = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 10) : 5120u;
    const uint32_t n_tok = (argc > 3) ? (uint32_t)strtoul(argv[3], NULL, 10) : 1u;
    const char *tag = (argc > 4) ? argv[4] : "arm";

    const uint32_t flat = 4u * dim;
    myds_metal_buffer_t *res = myds_metal_buffer_alloc((size_t)n_tok * flat * 4u);
    myds_metal_buffer_t *pre = myds_metal_buffer_alloc((size_t)n_tok * 24u * 4u);
    myds_metal_buffer_t *nw  = myds_metal_buffer_alloc((size_t)dim * 4u);
    myds_metal_buffer_t *xn  = myds_metal_buffer_alloc((size_t)n_tok * dim * 4u);
    myds_metal_buffer_t *sum = myds_metal_buffer_alloc((size_t)n_tok * dim * 4u);
    if (!res || !pre || !nw || !xn || !sum) {
        fprintf(stderr, "buffer alloc failed\n");
        return 2;
    }
    fill(res, 3u); fill(pre, 7u); fill(nw, 13u);

    /* round_bf16 = true: the production path. Only the sum_out content is
     * changed by the patch, so the comparison isolates the accumulator. */
    if (!myds_metal_v41_hc_sum_rmsnorm_batch(xn, sum, res, pre, nw, dim, n_tok,
                                             96u, 1e-5f, true)) {
        fprintf(stderr, "kernel dispatch rejected\n");
        return 3;
    }
    myds_metal_barrier();

    const char *lib = myds_metal_loaded_library_path();
    printf("{\"tag\":\"%s\",\"loaded_library\":\"%s\",\"dim\":%u,\"n_tok\":%u}\n",
           tag, lib ? lib : "", dim, n_tok);

    const float *src = myds_metal_buffer_data(sum);
    const size_t n = (size_t)n_tok * dim;
    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return 4; }
    if (fwrite(src, sizeof(float), n, f) != n) { fclose(f); return 4; }
    fclose(f);
    printf("{\"tag\":\"%s\",\"wrote\":\"%s\",\"floats\":%zu}\n", tag, out_path, n);
    return 0;
}
