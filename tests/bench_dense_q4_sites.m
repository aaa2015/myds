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

static double measure_dense_q4(uint32_t rows, uint32_t width, bool round_bf16,
                              const char *nr0_val, const char *mask_val) {
    setenv("MYDS_DENSE_Q4_V41", "1", 1);
    setenv("MYDS_DENSE_Q4_V41_NB", "1", 1);
    setenv("MYDS_DENSE_Q4_NR0", nr0_val, 1);
    setenv("MYDS_DENSE_Q4_NR4_MASK", mask_val, 1);

    if (!myds_metal_init(NULL)) {
        fprintf(stderr, "Failed to init metal for NR0=%s mask=%s\n", nr0_val, mask_val);
        exit(1);
    }

    size_t row_bytes = (width / 256u) * 144u;
    size_t w_bytes = row_bytes * rows;
    size_t in_bytes = width * sizeof(float);
    size_t out_bytes = rows * sizeof(float);

    myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *in = myds_metal_buffer_alloc(in_bytes);
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(out_bytes);

    fill_q4_k(myds_metal_buffer_data(w), w_bytes, 42);
    fill_f32(myds_metal_buffer_data(in), width, 99);

    // Warmup
    for (int i = 0; i < 20; i++) {
        myds_metal_begin_batch();
        myds_metal_dense_q4_v41(out, w, in, rows, width, round_bf16);
        myds_metal_commit_batch();
    }

    const int iters = 1000;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        myds_metal_begin_batch();
        myds_metal_dense_q4_v41(out, w, in, rows, width, round_bf16);
        myds_metal_commit_batch();
    }
    double t1 = now_us();
    double avg_us = (t1 - t0) / iters;

    myds_metal_buffer_free(w);
    myds_metal_buffer_free(in);
    myds_metal_buffer_free(out);
    myds_metal_cleanup();

    return avg_us;
}

static void test_dense_site(const char *name, uint32_t rows, uint32_t width, bool round_bf16, unsigned site_id) {
    char mask_str[16];
    snprintf(mask_str, sizeof(mask_str), "0x%x", 1u << site_id);

    double us_nr2 = measure_dense_q4(rows, width, round_bf16, "2", mask_str);
    double us_nr4 = measure_dense_q4(rows, width, round_bf16, "4", mask_str);

    size_t row_bytes = (width / 256u) * 144u;
    size_t total_bytes = row_bytes * rows + (width + rows) * sizeof(float);
    double gb = (double)total_bytes * 1e-9;
    double bw_nr2 = gb / (us_nr2 * 1e-6);
    double bw_nr4 = gb / (us_nr4 * 1e-6);
    double diff_pct = (us_nr4 - us_nr2) / us_nr2 * 100.0;

    printf("%-16s [site %u, %6ux%5u]: NR2 = %6.2f us (%5.1f GB/s) | NR4 = %6.2f us (%5.1f GB/s) | delta = %+6.2f%%\n",
           name, site_id, rows, width, us_nr2, bw_nr2, us_nr4, bw_nr4, diff_pct);
}

static double measure_attn_out_a(const char *nr0_val, const char *mask_val) {
    setenv("MYDS_DENSE_Q4_NR0", nr0_val, 1);
    setenv("MYDS_DENSE_Q4_NR4_MASK", mask_val, 1);

    if (!myds_metal_init(NULL)) {
        fprintf(stderr, "Failed to init metal for attn_out_a\n");
        exit(1);
    }

    const uint32_t n_groups = 8, group_dim = 4096, rank = 1024, n_tok = 1;
    size_t a_group_bytes = (uint64_t)rank * (group_dim / 256u) * 144u;
    size_t w_bytes = a_group_bytes * n_groups;
    size_t heads_bytes = (uint64_t)n_groups * group_dim * n_tok * sizeof(float);
    size_t low_bytes = (uint64_t)n_groups * rank * n_tok * sizeof(float);

    myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *heads = myds_metal_buffer_alloc(heads_bytes);
    myds_metal_buffer_t *low = myds_metal_buffer_alloc(low_bytes);

    fill_q4_k(myds_metal_buffer_data(w), w_bytes, 77);
    fill_f32(myds_metal_buffer_data(heads), heads_bytes / 4, 88);

    // Warmup
    for (int i = 0; i < 20; i++) {
        myds_metal_begin_batch();
        myds_metal_attn_output_a_bf16_batch(low, w, heads, n_groups, group_dim, rank, n_tok, GGML_TYPE_Q4_K);
        myds_metal_commit_batch();
    }

    const int iters = 1000;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        myds_metal_begin_batch();
        myds_metal_attn_output_a_bf16_batch(low, w, heads, n_groups, group_dim, rank, n_tok, GGML_TYPE_Q4_K);
        myds_metal_commit_batch();
    }
    double t1 = now_us();
    double avg_us = (t1 - t0) / iters;

    myds_metal_buffer_free(w);
    myds_metal_buffer_free(heads);
    myds_metal_buffer_free(low);
    myds_metal_cleanup();

    return avg_us;
}

static void test_attn_out_a_site(void) {
    double us_nr2 = measure_attn_out_a("2", "0x20");
    double us_nr4 = measure_attn_out_a("4", "0x20");

    const uint32_t n_groups = 8, group_dim = 4096, rank = 1024, n_tok = 1;
    size_t a_group_bytes = (uint64_t)rank * (group_dim / 256u) * 144u;
    size_t w_bytes = a_group_bytes * n_groups;
    size_t total_bytes = w_bytes + (n_groups * group_dim + n_groups * rank) * sizeof(float);
    double gb = (double)total_bytes * 1e-9;
    double bw_nr2 = gb / (us_nr2 * 1e-6);
    double bw_nr4 = gb / (us_nr4 * 1e-6);
    double diff_pct = (us_nr4 - us_nr2) / us_nr2 * 100.0;

    printf("%-16s [site 5, 8x%4ux%4u]: NR2 = %6.2f us (%5.1f GB/s) | NR4 = %6.2f us (%5.1f GB/s) | delta = %+6.2f%%\n",
           "attn_output_a", rank, group_dim, us_nr2, bw_nr2, us_nr4, bw_nr4, diff_pct);
}

int main(void) {
    printf("=== Dense Q4_K Isolated Per-Site Microbenchmarks (1000 iters) ===\n");
    test_dense_site("attn_output_b", 5120, 8192, true, 2);
    test_dense_site("attn_q_b", 32768, 1280, true, 1);
    test_dense_site("lm_head", 129280, 5120, false, 0);
    test_attn_out_a_site();
    printf("=== Benchmark Completed ===\n");
    return 0;
}
