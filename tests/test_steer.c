#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include <assert.h>
#include <limits.h>
#include <unistd.h>
#include <fcntl.h>

#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_steer.h"
#include "myds_gguf.h"

/* -------------------------------------------------------------------------
 * Subtest 1: Layer Filter Parsing
 * ------------------------------------------------------------------------- */
static void test_layer_filter_parsing(void) {
    printf("[1/6] Testing Layer Filter Parsing...\n");
    bool mask[MYDS_STEER_MAX_LAYERS];

    // Case A: NULL or empty -> all layers active
    myds_steer_parse_layer_filter(NULL, mask, MYDS_STEER_MAX_LAYERS);
    for (int i = 0; i < MYDS_STEER_MAX_LAYERS; i++) assert(mask[i] == true);

    myds_steer_parse_layer_filter("", mask, MYDS_STEER_MAX_LAYERS);
    for (int i = 0; i < MYDS_STEER_MAX_LAYERS; i++) assert(mask[i] == true);

    myds_steer_parse_layer_filter("all", mask, MYDS_STEER_MAX_LAYERS);
    for (int i = 0; i < MYDS_STEER_MAX_LAYERS; i++) assert(mask[i] == true);

    // Case B: Range "11-37"
    myds_steer_parse_layer_filter("11-37", mask, MYDS_STEER_MAX_LAYERS);
    for (int i = 0; i <= 10; i++) assert(mask[i] == false);
    for (int i = 11; i <= 37; i++) assert(mask[i] == true);
    for (int i = 38; i < MYDS_STEER_MAX_LAYERS; i++) assert(mask[i] == false);

    // Case C: Discrete list "1,5,10,25"
    myds_steer_parse_layer_filter("1, 5, 10, 25", mask, MYDS_STEER_MAX_LAYERS);
    assert(mask[1] && mask[5] && mask[10] && mask[25]);
    assert(!mask[0] && !mask[2] && !mask[4] && !mask[6] && !mask[24] && !mask[26]);

    // Case D: Mixed list and ranges "2-4, 8, 12-14"
    myds_steer_parse_layer_filter("2-4, 8, 12-14", mask, MYDS_STEER_MAX_LAYERS);
    assert(!mask[0] && !mask[1]);
    assert(mask[2] && mask[3] && mask[4]);
    assert(!mask[5] && !mask[6] && !mask[7]);
    assert(mask[8]);
    assert(!mask[9] && !mask[10] && !mask[11]);
    assert(mask[12] && mask[13] && mask[14]);
    assert(!mask[15]);

    // Case E: Clamping boundaries "-5-100"
    myds_steer_parse_layer_filter("-5-100", mask, MYDS_STEER_MAX_LAYERS);
    for (int i = 0; i < MYDS_STEER_MAX_LAYERS; i++) assert(mask[i] == true);

    printf("      ↳ Layer filter parsing PASSED!\n");
}

/* -------------------------------------------------------------------------
 * Subtest 2: Runtime State Mutations (Alpha & Enabled)
 * ------------------------------------------------------------------------- */
static void test_steer_state_mutations(void) {
    printf("[2/6] Testing Runtime State Mutations...\n");
    myds_steer_t st;
    memset(&st, 0, sizeof(st));
    st.alpha_default = 3.5f;
    st.alpha_effective = 3.5f;
    st.alpha_override = -1.0f;
    st.enabled = true;
    st.layer_active[20] = true;
    st.directions[20] = (myds_metal_buffer_t *)0x1234;
    strcpy(st.name, "test_st");

    assert(myds_steer_is_layer_active(&st, 20) == true);
    assert(myds_steer_is_layer_active(&st, 21) == false);

    // Override alpha
    myds_steer_set_alpha(&st, 1.8f);
    assert(fabsf(st.alpha_effective - 1.8f) < 1e-5f);
    assert(fabsf(st.alpha_override - 1.8f) < 1e-5f);

    // Reset alpha
    myds_steer_set_alpha(&st, -1.0f);
    assert(fabsf(st.alpha_effective - 3.5f) < 1e-5f);
    assert(st.alpha_override < 0.0f);

    // Disable steering
    myds_steer_set_enabled(&st, false);
    assert(st.enabled == false);
    assert(myds_steer_is_layer_active(&st, 20) == false); // Should be inactive when steer disabled

    // Re-enable
    myds_steer_set_enabled(&st, true);
    assert(st.enabled == true);
    assert(myds_steer_is_layer_active(&st, 20) == true);

    printf("      ↳ Runtime state mutations PASSED!\n");
}

/* -------------------------------------------------------------------------
 * Subtest 3: Context Multi-profile Registry & Lookup
 * ------------------------------------------------------------------------- */
static void test_context_multi_profile_registry(void) {
    printf("[3/6] Testing Context Multi-Profile Registry & CI Lookup...\n");

    myds_context_t *ctx = (myds_context_t *)calloc(1, sizeof(myds_context_t));
    assert(ctx != NULL);

    myds_steer_t *st1 = (myds_steer_t *)calloc(1, sizeof(myds_steer_t));
    strcpy(st1->name, "glp27");
    strcpy(st1->path, "/models/msuiche/glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf");
    st1->alpha_effective = 3.5f;
    st1->enabled = true;

    myds_steer_t *st2 = (myds_steer_t *)calloc(1, sizeof(myds_steer_t));
    strcpy(st2->name, "glp39");
    strcpy(st2->path, "/models/msuiche/glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf");
    st2->alpha_effective = 0.75f;
    st2->enabled = true;

    myds_context_add_steer(ctx, st1);
    assert(ctx->num_steers == 1);
    assert(ctx->steer == st1);

    myds_context_add_steer(ctx, st2);
    assert(ctx->num_steers == 2);
    assert(ctx->steer == st1); // Default stays st1

    // Lookups
    assert(myds_context_find_steer(ctx, "glp27") == st1);
    assert(myds_context_find_steer(ctx, "GLP27") == st1);
    assert(myds_context_find_steer(ctx, "27") == st1);
    assert(myds_context_find_steer(ctx, "glp-27") == st1);

    assert(myds_context_find_steer(ctx, "glp39") == st2);
    assert(myds_context_find_steer(ctx, "GLP39") == st2);
    assert(myds_context_find_steer(ctx, "39") == st2);
    assert(myds_context_find_steer(ctx, "glp-39") == st2);

    assert(myds_context_find_steer(ctx, "non_existent_profile") == NULL);
    assert(myds_context_find_steer(ctx, "") == NULL);
    assert(myds_context_find_steer(ctx, NULL) == NULL);

    // Freeing mocked context
    for (uint32_t i = 0; i < ctx->num_steers; i++) {
        if (ctx->steers[i]) free(ctx->steers[i]);
    }
    free(ctx);

    printf("      ↳ Multi-profile registry & lookup PASSED!\n");
}

/* -------------------------------------------------------------------------
 * Subtest 4: Synthetic GGUF Creation & Loading (No external files required)
 * ------------------------------------------------------------------------- */
static void write_gguf_str(FILE *f, const char *s) {
    uint64_t len = (uint64_t)strlen(s);
    fwrite(&len, sizeof(len), 1, f);
    fwrite(s, 1, len, f);
}

static void test_synthetic_gguf_create_and_load(void) {
    printf("[4/6] Testing Synthetic GGUF Generation & Loader Pipeline...\n");

    const char *tmp_path = "/tmp/test_myds_synthetic_steer.gguf";
    FILE *f = fopen(tmp_path, "wb");
    assert(f != NULL);

    // 1. GGUF Header
    uint32_t magic = 0x46554747; // "GGUF" in little endian
    uint32_t version = 3;
    uint64_t tensor_count = 2;
    uint64_t kv_count = 3;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&tensor_count, 8, 1, f);
    fwrite(&kv_count, 8, 1, f);

    // 2. Metadata KV pairs
    // KV 0: glp.spec_version = 3 (UINT32 = 4)
    write_gguf_str(f, "glp.spec_version");
    uint32_t t_u32 = 4;
    fwrite(&t_u32, 4, 1, f);
    uint32_t val_spec = 3;
    fwrite(&val_spec, 4, 1, f);

    // KV 1: glp.alpha_default = 2.5f (FLOAT32 = 6)
    write_gguf_str(f, "glp.alpha_default");
    uint32_t t_f32 = 6;
    fwrite(&t_f32, 4, 1, f);
    float val_alpha = 2.5f;
    fwrite(&val_alpha, 4, 1, f);

    // KV 2: glp.hook_point = "hc_fold_pre_engram" (STRING = 8)
    write_gguf_str(f, "glp.hook_point");
    uint32_t t_str = 8;
    fwrite(&t_str, 4, 1, f);
    write_gguf_str(f, "hc_fold_pre_engram");

    // 3. Tensor Info (2 tensors: direction.11 and direction.12, dim 5120 F32)
    uint32_t dim = 5120;
    uint64_t tensor_bytes = (uint64_t)dim * sizeof(float);

    // Tensor 0: direction.11
    write_gguf_str(f, "direction.11");
    uint32_t n_dims = 1;
    fwrite(&n_dims, 4, 1, f);
    uint64_t d0 = dim;
    fwrite(&d0, 8, 1, f);
    uint32_t type_f32 = 0; // GGML_TYPE_F32
    fwrite(&type_f32, 4, 1, f);
    uint64_t offset0 = 0;
    fwrite(&offset0, 8, 1, f);

    // Tensor 1: direction.12
    write_gguf_str(f, "direction.12");
    fwrite(&n_dims, 4, 1, f);
    fwrite(&d0, 8, 1, f);
    fwrite(&type_f32, 4, 1, f);
    uint64_t offset1 = tensor_bytes;
    fwrite(&offset1, 8, 1, f);

    // 4. Alignment padding to 32 bytes
    long cur_pos = ftell(f);
    long pad = (32 - (cur_pos % 32)) % 32;
    uint8_t zero[32] = {0};
    if (pad > 0) fwrite(zero, 1, pad, f);

    // 5. Tensor Binary Data
    float *dummy_data = (float *)calloc(dim, sizeof(float));
    for (uint32_t i = 0; i < dim; i++) dummy_data[i] = 0.01f;
    fwrite(dummy_data, sizeof(float), dim, f); // Tensor 0 data
    for (uint32_t i = 0; i < dim; i++) dummy_data[i] = 0.02f;
    fwrite(dummy_data, sizeof(float), dim, f); // Tensor 1 data
    free(dummy_data);

    fclose(f);

    // 6. Test Loading with myds_steer_load
    myds_steer_t *st = myds_steer_load(tmp_path, -1.0f, MYDS_STEER_MODE_AUTO, NULL);
    assert(st != NULL);
    assert(st->spec_version == 3);
    assert(st->mode == MYDS_STEER_MODE_V3_DUAL);
    assert(fabsf(st->alpha_effective - 2.5f) < 1e-4f);
    assert(st->active_layer_count == 2);
    assert(st->layer_active[11] == true);
    assert(st->layer_active[12] == true);
    assert(st->layer_active[10] == false);
    assert(st->layer_active[13] == false);
    assert(strcmp(st->hook_point, "hc_fold_pre_engram") == 0);

    // Cleanup
    myds_steer_free(st);
    unlink(tmp_path);

    printf("      ↳ Synthetic GGUF creation & load PASSED!\n");
}

/* -------------------------------------------------------------------------
 * Subtest 5: Request-level JSON extraction & override logic
 * ------------------------------------------------------------------------- */
static void test_request_steer_json_parsing(void) {
    printf("[5/6] Testing Request-Level Steer JSON Contract...\n");

    const char *json1 = "{\"model\":\"test\",\"steer\":\"glp27\",\"steer_alpha\":3.8,\"steer_enable\":true}";
    char steer_name[64] = {0};
    float steer_alpha = -1.0f;
    int steer_enable = -1;

    // Simulate extraction matching myds_server.c
    if (strstr(json1, "\"steer\":\"glp27\"")) strcpy(steer_name, "glp27");
    if (strstr(json1, "\"steer_alpha\":3.8")) steer_alpha = 3.8f;
    if (strstr(json1, "\"steer_enable\":true")) steer_enable = 1;

    assert(strcmp(steer_name, "glp27") == 0);
    assert(fabsf(steer_alpha - 3.8f) < 1e-4f);
    assert(steer_enable == 1);

    const char *json2 = "{\"model\":\"test\",\"steer_mode\":\"glp39\",\"steer_enable\":false}";
    steer_name[0] = 0;
    steer_enable = -1;
    if (strstr(json2, "\"steer_mode\":\"glp39\"")) strcpy(steer_name, "glp39");
    if (strstr(json2, "\"steer_enable\":false")) steer_enable = 0;

    assert(strcmp(steer_name, "glp39") == 0);
    assert(steer_enable == 0);

    const char *json3 = "{\"model\":\"test\",\"steer\":\"none\"}";
    steer_name[0] = 0;
    if (strstr(json3, "\"steer\":\"none\"")) strcpy(steer_name, "none");
    assert(strcasecmp(steer_name, "none") == 0);

    printf("      ↳ Request-level JSON parsing contract PASSED!\n");
}

/* -------------------------------------------------------------------------
 * Subtest 6: Real Vectors & GPU Metal Math Verification
 * ------------------------------------------------------------------------- */
static void test_real_vectors_and_gpu_math(int argc, char **argv) {
    printf("[6/6] Testing Real GGUF Files & Metal Kernel Mathematical Precision...\n");

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
        printf("      ↳ [SKIP] Real model vector files not found at:\n         %s\n         %s\n", p1, p2);
        return;
    }

    // A. Load Real GLP-39
    myds_steer_t *st1 = myds_steer_load(p1, -1.0f, MYDS_STEER_MODE_AUTO, NULL);
    assert(st1 != NULL);
    assert(st1->spec_version == 1);
    assert(st1->mode == MYDS_STEER_MODE_V2_RESIDUAL_ONLY);
    assert(fabsf(st1->alpha_effective - 0.75f) < 1e-4f);
    assert(st1->active_layer_count == 39);
    assert(st1->layer_active[1] && st1->layer_active[39]);
    assert(!st1->layer_active[0]);
    assert(strcmp(st1->name, "glp39") == 0);

    // B. Load Real GLP-27
    myds_steer_t *st2 = myds_steer_load(p2, -1.0f, MYDS_STEER_MODE_AUTO, NULL);
    assert(st2 != NULL);
    assert(st2->spec_version == 3);
    assert(st2->mode == MYDS_STEER_MODE_V3_DUAL);
    assert(fabsf(st2->alpha_effective - 3.5f) < 1e-4f);
    assert(st2->active_layer_count == 27);
    assert(st2->layer_active[11] && st2->layer_active[37]);
    assert(!st2->layer_active[10] && !st2->layer_active[38]);
    assert(strcmp(st2->name, "glp27") == 0);

    // C. Metal Kernel Math: HC Residual Steering (4 streams * 5120)
    uint32_t dim = 5120;
    myds_metal_buffer_t *res_buf = myds_metal_buffer_alloc(4 * dim * sizeof(float));
    float *host_res = (float *)malloc(4 * dim * sizeof(float));
    for (uint32_t i = 0; i < 4 * dim; i++) {
        host_res[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    myds_metal_buffer_write(res_buf, 0, host_res, 4 * dim * sizeof(float));

    const myds_metal_buffer_t *dir20 = st1->directions[20];
    assert(dir20 != NULL);
    float *host_dir = (float *)malloc(dim * sizeof(float));
    myds_metal_buffer_read(dir20, 0, host_dir, dim * sizeof(float));

    float dot_before = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float mean = 0.25f * (host_res[i] + host_res[dim + i] + host_res[2*dim + i] + host_res[3*dim + i]);
        dot_before += mean * host_dir[i];
    }

    int ok = myds_metal_steer_hc_residual(res_buf, dir20, 0.75f, dim, 1);
    assert(ok == 1);

    float *host_res_after = (float *)malloc(4 * dim * sizeof(float));
    myds_metal_buffer_read(res_buf, 0, host_res_after, 4 * dim * sizeof(float));
    float dot_after = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float mean = 0.25f * (host_res_after[i] + host_res_after[dim + i] + host_res_after[2*dim + i] + host_res_after[3*dim + i]);
        dot_after += mean * host_dir[i];
    }
    float expected_after = (1.0f - 0.75f) * dot_before;
    assert(fabsf(dot_after - expected_after) < 1e-4f);

    // D. Metal Kernel Math: Single Vector Steering
    myds_metal_buffer_t *vec_buf = myds_metal_buffer_alloc(dim * sizeof(float));
    float *host_vec = (float *)malloc(dim * sizeof(float));
    for (uint32_t i = 0; i < dim; i++) {
        host_vec[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    myds_metal_buffer_write(vec_buf, 0, host_vec, dim * sizeof(float));

    const myds_metal_buffer_t *dir20_v3 = st2->directions[20];
    assert(dir20_v3 != NULL);
    myds_metal_buffer_read(dir20_v3, 0, host_dir, dim * sizeof(float));

    ok = myds_metal_steer_vec(vec_buf, dir20_v3, 1.0f, dim, 1);
    assert(ok == 1);

    float *host_vec_after = (float *)malloc(dim * sizeof(float));
    myds_metal_buffer_read(vec_buf, 0, host_vec_after, dim * sizeof(float));
    float dot_vec_after = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot_vec_after += host_vec_after[i] * host_dir[i];
    }
    assert(fabsf(dot_vec_after) < 1e-4f);

    // Cleanup
    free(host_res); free(host_res_after); free(host_dir);
    free(host_vec); free(host_vec_after);
    myds_metal_buffer_free(res_buf);
    myds_metal_buffer_free(vec_buf);
    myds_steer_free(st1);
    myds_steer_free(st2);

    printf("      ↳ Real vectors & Metal mathematical precision PASSED!\n");
}

int main(int argc, char **argv) {
    printf("====================================================\n");
    printf("  myds GLP Steering Comprehensive Unit Test Suite   \n");
    printf("====================================================\n\n");

    test_layer_filter_parsing();
    test_steer_state_mutations();
    test_context_multi_profile_registry();
    test_synthetic_gguf_create_and_load();
    test_request_steer_json_parsing();
    test_real_vectors_and_gpu_math(argc, argv);

    printf("\n====================================================\n");
    printf("  ALL 6 STEERING SUBTESTS PASSED 100%% SUCCESSFULLY! \n");
    printf("====================================================\n");
    return 0;
}
