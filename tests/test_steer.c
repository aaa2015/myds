#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <unistd.h>

#include "myds_metal.h"
#include "myds_steer.h"

int main(int argc, char **argv) {
    printf("=== Testing GLP Steering on Metal ===\n");

    char p1[PATH_MAX];
    char p2[PATH_MAX];
    const char *home = getenv("HOME");
    if (!home) home = "/Users/fangqingyuan";

    if (argc >= 3) {
        snprintf(p1, sizeof(p1), "%s", argv[1]);
        snprintf(p2, sizeof(p2), "%s", argv[2]);
    } else {
        snprintf(p1, sizeof(p1), "%s/.lmstudio/models/msuiche/glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf", home);
        snprintf(p2, sizeof(p2), "%s/.lmstudio/models/msuiche/glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf", home);
    }

    if (access(p1, R_OK) != 0 || access(p2, R_OK) != 0) {
        printf("[SKIP] Steering vector files not found at:\n  %s\n  %s\n", p1, p2);
        return 0;
    }

    // 1. Test loading GLP-39
    printf("\n--- Loading GLP-39 ---\n");
    myds_steer_t *st1 = myds_steer_load(p1, -1.0f, MYDS_STEER_MODE_AUTO, NULL);
    assert(st1 != NULL);
    assert(st1->spec_version == 1);
    assert(st1->mode == MYDS_STEER_MODE_V2_RESIDUAL_ONLY);
    assert(fabsf(st1->alpha_effective - 0.75f) < 1e-4f);
    assert(st1->active_layer_count == 39);
    assert(st1->layer_active[1] && st1->layer_active[39]);
    assert(!st1->layer_active[0]); // Layer 0 not in file
    printf("GLP-39 loaded successfully!\n");

    // 2. Test loading GLP-27
    printf("\n--- Loading GLP-27 ---\n");
    myds_steer_t *st2 = myds_steer_load(p2, -1.0f, MYDS_STEER_MODE_AUTO, NULL);
    assert(st2 != NULL);
    assert(st2->spec_version == 3);
    assert(st2->mode == MYDS_STEER_MODE_V3_DUAL);
    assert(fabsf(st2->alpha_effective - 3.5f) < 1e-4f);
    assert(st2->active_layer_count == 27);
    assert(st2->layer_active[11] && st2->layer_active[37]);
    assert(!st2->layer_active[10] && !st2->layer_active[38]);
    printf("GLP-27 loaded successfully!\n");

    // 3. Test Metal Kernel Execution on GLP-39 (HC Residual)
    printf("\n--- Testing Metal Kernel: HC Residual Steering ---\n");
    uint32_t dim = 5120;
    myds_metal_buffer_t *res_buf = myds_metal_buffer_alloc(4 * dim * sizeof(float));
    float *host_res = (float *)malloc(4 * dim * sizeof(float));
    for (uint32_t i = 0; i < 4 * dim; i++) {
        host_res[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    myds_metal_buffer_write(res_buf, 0, host_res, 4 * dim * sizeof(float));

    // Direction vector from layer 20
    const myds_metal_buffer_t *dir20 = st1->directions[20];
    assert(dir20 != NULL);
    float *host_dir = (float *)malloc(dim * sizeof(float));
    myds_metal_buffer_read(dir20, 0, host_dir, dim * sizeof(float));

    // Compute expected dot product before steering
    float dot_before = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float mean = 0.25f * (host_res[i] + host_res[dim + i] + host_res[2*dim + i] + host_res[3*dim + i]);
        dot_before += mean * host_dir[i];
    }
    printf("Before steering: dot(h_mean, dir) = %f\n", dot_before);

    // Run GPU kernel with alpha = 0.75
    int ok = myds_metal_steer_hc_residual(res_buf, dir20, 0.75f, dim, 1);
    assert(ok == 1);

    // Read back and verify
    float *host_res_after = (float *)malloc(4 * dim * sizeof(float));
    myds_metal_buffer_read(res_buf, 0, host_res_after, 4 * dim * sizeof(float));
    float dot_after = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float mean = 0.25f * (host_res_after[i] + host_res_after[dim + i] + host_res_after[2*dim + i] + host_res_after[3*dim + i]);
        dot_after += mean * host_dir[i];
    }
    printf("After steering (alpha=0.75): dot(h_mean, dir) = %f\n", dot_after);
    float expected_after = (1.0f - 0.75f) * dot_before;
    printf("Expected dot: %f, Actual dot: %f (diff: %e)\n", expected_after, dot_after, fabsf(dot_after - expected_after));
    assert(fabsf(dot_after - expected_after) < 1e-4f);
    printf("HC Residual Steering test PASSED!\n");

    // 4. Test Metal Kernel Execution on GLP-27 (Vector Steering)
    printf("\n--- Testing Metal Kernel: Single Vector Steering ---\n");
    myds_metal_buffer_t *vec_buf = myds_metal_buffer_alloc(dim * sizeof(float));
    float *host_vec = (float *)malloc(dim * sizeof(float));
    for (uint32_t i = 0; i < dim; i++) {
        host_vec[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    myds_metal_buffer_write(vec_buf, 0, host_vec, dim * sizeof(float));

    const myds_metal_buffer_t *dir20_v3 = st2->directions[20];
    assert(dir20_v3 != NULL);
    myds_metal_buffer_read(dir20_v3, 0, host_dir, dim * sizeof(float));

    float dot_vec_before = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot_vec_before += host_vec[i] * host_dir[i];
    }
    printf("Before vector steering: dot(h, dir) = %f\n", dot_vec_before);

    // Run GPU kernel with alpha = 1.0 (should produce exactly zero projection)
    ok = myds_metal_steer_vec(vec_buf, dir20_v3, 1.0f, dim, 1);
    assert(ok == 1);

    float *host_vec_after = (float *)malloc(dim * sizeof(float));
    myds_metal_buffer_read(vec_buf, 0, host_vec_after, dim * sizeof(float));
    float dot_vec_after = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot_vec_after += host_vec_after[i] * host_dir[i];
    }
    printf("After vector steering (alpha=1.0): dot(h, dir) = %f (expected: 0.0)\n", dot_vec_after);
    assert(fabsf(dot_vec_after) < 1e-4f);
    printf("Single Vector Steering test PASSED!\n");

    // 5. Test Steer Profile Names
    printf("\n--- Testing Profile Naming ---\n");
    assert(strcmp(st1->name, "glp39") == 0);
    assert(strcmp(st2->name, "glp27") == 0);
    printf("st1 name: %s, st2 name: %s\n", st1->name, st2->name);
    printf("Profile naming test PASSED!\n");

    // Cleanup
    free(host_res); free(host_res_after); free(host_dir);
    free(host_vec); free(host_vec_after);
    myds_metal_buffer_free(res_buf);
    myds_metal_buffer_free(vec_buf);
    myds_steer_free(st1);
    myds_steer_free(st2);

    printf("\n=== All GLP Steering Metal Tests Passed 100%% Successfully! ===\n");
    return 0;
}
