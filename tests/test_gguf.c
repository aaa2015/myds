#include "myds_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        return 0; \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    if (strcmp((a), (b)) != 0) { \
        fprintf(stderr, "[FAIL] %s:%d: '%s' != '%s'\n", __FILE__, __LINE__, (a), (b)); \
        return 0; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    float diff = (float)((a) > (b) ? ((a) - (b)) : ((b) - (a))); \
    if (diff > (tol)) { \
        fprintf(stderr, "[FAIL] %s:%d: %.6f != %.6f\n", __FILE__, __LINE__, (float)(a), (float)(b)); \
        return 0; \
    } \
} while(0)

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }
static void write_f32(FILE *f, float v) { fwrite(&v, sizeof(v), 1, f); }
static void write_str(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    write_u64(f, len);
    fwrite(s, 1, len, f);
}

/* Helper to generate a minimal valid synthetic GGUF v3 file */
static const char *create_synthetic_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;

    /* 1. Header: magic, version, n_tensors, n_kvs */
    uint32_t magic = GGUF_MAGIC;
    uint32_t version = 3;
    uint64_t n_tensors = 2;
    uint64_t n_kvs = 7;

    fwrite(&magic, 4, 1, f);
    write_u32(f, version);
    write_u64(f, n_tensors);
    write_u64(f, n_kvs);

    /* 2. Key-Values */
    // kv 0: general.architecture (string) = "deepseek2"
    write_str(f, "general.architecture");
    write_u32(f, GGUF_TYPE_STRING);
    write_str(f, "deepseek2");

    // kv 1: deepseek2.embedding_length (u32) = 2048
    write_str(f, "deepseek2.embedding_length");
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 2048);

    // kv 2: deepseek2.block_count (u32) = 28
    write_str(f, "deepseek2.block_count");
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 28);

    // kv 3: deepseek2.rope.freq_base (f32) = 10000.0f
    write_str(f, "deepseek2.rope.freq_base");
    write_u32(f, GGUF_TYPE_FLOAT32);
    write_f32(f, 10000.0f);

    // kv 4: general.alignment (u32) = 32
    write_str(f, "general.alignment");
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 32);

    // kv 5: test.flag (bool) = true (1)
    write_str(f, "test.flag");
    write_u32(f, GGUF_TYPE_BOOL);
    uint8_t b_val = 1;
    fwrite(&b_val, 1, 1, f);

    // kv 6: test.array (array of u32) = [10, 20, 30]
    write_str(f, "test.array");
    write_u32(f, GGUF_TYPE_ARRAY);
    write_u32(f, GGUF_TYPE_UINT32);
    write_u64(f, 3);
    uint32_t arr[3] = {10, 20, 30};
    fwrite(arr, sizeof(uint32_t), 3, f);

    /* 3. Tensor infos */
    // tensor 0: "token_embd.weight", 2 dims: [2048, 4], type: F32 (0), offset: 0
    write_str(f, "token_embd.weight");
    write_u32(f, 2);
    write_u64(f, 2048);
    write_u64(f, 4);
    write_u32(f, GGML_TYPE_F32);
    write_u64(f, 0);

    // tensor 1: "output_norm.weight", 1 dim: [2048], type: F32 (0), offset: 2048*4*sizeof(float) = 32768
    write_str(f, "output_norm.weight");
    write_u32(f, 1);
    write_u64(f, 2048);
    write_u32(f, GGML_TYPE_F32);
    write_u64(f, 32768);

    /* 4. Pad to 32-byte alignment */
    long pos = ftell(f);
    long pad = (32 - (pos % 32)) % 32;
    for (long i = 0; i < pad; i++) fputc(0, f);

    /* 5. Tensor Binary Data */
    // tensor 0 data (2048 * 4 floats)
    float *t0_data = (float *)calloc(2048 * 4, sizeof(float));
    for (int i = 0; i < 2048 * 4; i++) t0_data[i] = 1.25f;
    fwrite(t0_data, sizeof(float), 2048 * 4, f);
    free(t0_data);

    // tensor 1 data (2048 floats)
    float *t1_data = (float *)calloc(2048, sizeof(float));
    for (int i = 0; i < 2048; i++) t1_data[i] = 2.5f;
    fwrite(t1_data, sizeof(float), 2048, f);
    free(t1_data);

    fclose(f);
    return path;
}

static int test_gguf_metadata_parsing(const char *path) {
    gguf_file_t *gf = gguf_open(path);
    ASSERT_TRUE(gf != NULL);

    ASSERT_TRUE(gf->version == 3);
    ASSERT_TRUE(gf->tensor_count == 2);
    ASSERT_TRUE(gf->metadata_kv_count == 7);
    ASSERT_TRUE(gf->alignment == 32);

    /* Test key lookups */
    const char *arch = gguf_get_str(gf, "general.architecture", "");
    ASSERT_STR_EQ(arch, "deepseek2");

    uint32_t emb_len = gguf_get_u32(gf, "deepseek2.embedding_length", 0);
    ASSERT_TRUE(emb_len == 2048);

    uint32_t n_layers = gguf_get_u32(gf, "deepseek2.block_count", 0);
    ASSERT_TRUE(n_layers == 28);

    float freq_base = gguf_get_f32(gf, "deepseek2.rope.freq_base", 0.0f);
    ASSERT_NEAR(freq_base, 10000.0f, 1e-4f);

    bool flag = gguf_get_bool(gf, "test.flag", false);
    ASSERT_TRUE(flag == true);

    /* Test fallback default values on missing keys */
    ASSERT_TRUE(gguf_get_u32(gf, "nonexistent.key", 999) == 999);
    ASSERT_STR_EQ(gguf_get_str(gf, "nonexistent.key", "default_val"), "default_val");

    gguf_close(gf);
    printf("  [PASS] GGUF metadata parsing\n");
    return 1;
}

static int test_gguf_tensor_parsing(const char *path) {
    gguf_file_t *gf = gguf_open(path);
    ASSERT_TRUE(gf != NULL);

    /* Test tensor lookup */
    const gguf_tensor_t *t0 = gguf_find_tensor(gf, "token_embd.weight");
    ASSERT_TRUE(t0 != NULL);
    ASSERT_STR_EQ(t0->name, "token_embd.weight");
    ASSERT_TRUE(t0->n_dims == 2);
    ASSERT_TRUE(t0->dims[0] == 2048);
    ASSERT_TRUE(t0->dims[1] == 4);
    ASSERT_TRUE(t0->type == GGML_TYPE_F32);
    ASSERT_TRUE(t0->offset == 0);
    ASSERT_TRUE(t0->data != NULL);

    /* Check mapped content */
    const float *f0 = (const float *)t0->data;
    ASSERT_NEAR(f0[0], 1.25f, 1e-5f);
    ASSERT_NEAR(f0[2048 * 4 - 1], 1.25f, 1e-5f);

    const gguf_tensor_t *t1 = gguf_find_tensor(gf, "output_norm.weight");
    ASSERT_TRUE(t1 != NULL);
    ASSERT_STR_EQ(t1->name, "output_norm.weight");
    ASSERT_TRUE(t1->n_dims == 1);
    ASSERT_TRUE(t1->dims[0] == 2048);
    ASSERT_TRUE(t1->data != NULL);

    const float *f1 = (const float *)t1->data;
    ASSERT_NEAR(f1[0], 2.5f, 1e-5f);
    ASSERT_NEAR(f1[2047], 2.5f, 1e-5f);

    /* Test nonexistent tensor */
    ASSERT_TRUE(gguf_find_tensor(gf, "nonexistent.tensor") == NULL);

    gguf_close(gf);
    printf("  [PASS] GGUF tensor memory-mapping and lookup\n");
    return 1;
}

static int test_gguf_error_handling(void) {
    /* Test opening nonexistent file */
    gguf_file_t *gf = gguf_open("/tmp/definitely_not_existing_file.gguf");
    ASSERT_TRUE(gf == NULL);

    /* Test opening file with invalid magic */
    const char *bad_path = "/tmp/test_bad_magic.bin";
    FILE *f = fopen(bad_path, "wb");
    uint32_t bad_magic = 0x12345678;
    fwrite(&bad_magic, sizeof(bad_magic), 1, f);
    fclose(f);

    gf = gguf_open(bad_path);
    ASSERT_TRUE(gf == NULL);
    unlink(bad_path);

    printf("  [PASS] GGUF corrupt and nonexistent file error handling\n");
    return 1;
}

int main(void) {
    printf("====================================================\n");
    printf("        Running myds GGUF Parser Unit Tests         \n");
    printf("====================================================\n");

    const char *temp_file = "/tmp/test_myds_synthetic.gguf";
    if (!create_synthetic_gguf(temp_file)) {
        fprintf(stderr, "Failed to create synthetic test GGUF file\n");
        return 1;
    }

    int passed = 0;
    passed += test_gguf_metadata_parsing(temp_file);
    passed += test_gguf_tensor_parsing(temp_file);
    passed += test_gguf_error_handling();

    unlink(temp_file);

    printf("====================================================\n");
    if (passed == 3) {
        printf("  ALL GGUF PARSER UNIT TESTS PASSED!\n");
        printf("====================================================\n");
        return 0;
    } else {
        printf("  GGUF TESTS FAILED (passed %d/3)\n", passed);
        return 1;
    }
}
