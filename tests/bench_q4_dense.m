#include "myds_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static void fill_q4_k(void *ptr, size_t bytes, unsigned seed) {
    uint8_t *p = (uint8_t *)ptr;
    for (size_t i = 0; i < bytes; i++) {
        p[i] = (uint8_t)((i * 37 + seed * 101) & 0xff);
    }
    size_t n_blocks = bytes / 144;
    for (size_t b = 0; b < n_blocks; b++) {
        uint16_t *d_ptr = (uint16_t *)(p + b * 144);
        uint16_t *dmin_ptr = (uint16_t *)(p + b * 144 + 2);
        *d_ptr = 0x3c00;
        *dmin_ptr = 0x3800;
    }
}

static void fill_f32(void *ptr, size_t count, unsigned seed) {
    float *p = (float *)ptr;
    for (size_t i = 0; i < count; i++) {
        p[i] = (float)(((int)(i * 43 + seed * 97) % 201) - 100) / 100.0f;
    }
}

static void bench_shape(const char *name, uint32_t out_dim, uint32_t in_dim) {
    size_t row_bytes = (in_dim / 256u) * 144u;
    size_t w_bytes = row_bytes * out_dim;
    size_t in_bytes = in_dim * sizeof(float);
    size_t out_bytes = out_dim * sizeof(float);

    myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *in = myds_metal_buffer_alloc(in_bytes);
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(out_bytes);

    fill_q4_k(myds_metal_buffer_data(w), w_bytes, 42);
    fill_f32(myds_metal_buffer_data(in), in_dim, 99);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        myds_metal_begin_batch();
        myds_metal_v41_gemv_bf16(out, w, in, out_dim, in_dim, GGML_TYPE_Q4_K);
        myds_metal_commit_batch();
    }

    const int iters = 200;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        myds_metal_begin_batch();
        myds_metal_v41_gemv_bf16(out, w, in, out_dim, in_dim, GGML_TYPE_Q4_K);
        myds_metal_commit_batch();
    }
    double t1 = now_us();
    double avg_us = (t1 - t0) / iters;
    double gb = (double)(w_bytes + in_bytes + out_bytes) * 1e-9;
    double bw_gbs = gb / (avg_us * 1e-6);

    printf("%-15s [%5u x %5u Q4_K]: avg = %6.2f us (%6.2f GB/s)\n", name, out_dim, in_dim, avg_us, bw_gbs);

    myds_metal_buffer_free(w);
    myds_metal_buffer_free(in);
    myds_metal_buffer_free(out);
}

static void bench_attn_output_a(void) {
    const uint32_t n_groups = 8, group_dim = 1024, rank = 512, n_tok = 2;
    size_t a_group_bytes = (uint64_t)rank * (group_dim / 256u) * 144u;
    size_t w_bytes = a_group_bytes * n_groups;
    size_t heads_bytes = (uint64_t)n_groups * group_dim * n_tok * sizeof(float);
    size_t low_bytes = (uint64_t)n_groups * rank * n_tok * sizeof(float);

    myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *heads = myds_metal_buffer_alloc(heads_bytes);
    myds_metal_buffer_t *low = myds_metal_buffer_alloc(low_bytes);

    fill_q4_k(myds_metal_buffer_data(w), w_bytes, 77);
    fill_f32(myds_metal_buffer_data(heads), heads_bytes / 4, 88);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        myds_metal_begin_batch();
        myds_metal_attn_output_a_bf16_batch(low, w, heads, n_groups, group_dim, rank, n_tok, GGML_TYPE_Q4_K);
        myds_metal_commit_batch();
    }

    const int iters = 200;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        myds_metal_begin_batch();
        myds_metal_attn_output_a_bf16_batch(low, w, heads, n_groups, group_dim, rank, n_tok, GGML_TYPE_Q4_K);
        myds_metal_commit_batch();
    }
    double t1 = now_us();
    double avg_us = (t1 - t0) / iters;
    double gb = (double)(w_bytes + heads_bytes + low_bytes) * 1e-9;
    double bw_gbs = gb / (avg_us * 1e-6);

    printf("%-15s [ 8x%4ux%4u Q4_K]: avg = %6.2f us (%6.2f GB/s)\n", "attn_output_a", rank, group_dim, avg_us, bw_gbs);

    myds_metal_buffer_free(w);
    myds_metal_buffer_free(heads);
    myds_metal_buffer_free(low);
}

int main(int argc, char **argv) {
    const char *metallib = (argc > 1) ? argv[1] : NULL;
    if (!myds_metal_init(metallib)) {
        fprintf(stderr, "Failed to init metal\n");
        return 1;
    }

    bench_shape("attn_output_b", 5120, 8192);
    bench_shape("shared_down",   5120, 2304);
    bench_shape("attn_q_a",      1280, 5120);
    bench_shape("attn_kv",        512, 5120);
    bench_shape("attn_q_b",     32768, 1280);
    bench_attn_output_a();

    myds_metal_cleanup();
    return 0;
}
