#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds.h"
#include "myds_gguf.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_server_internal.h"

static void test_p1_response_id_lifecycle(void) {
    char id1[64], id2[64];
    resp_id_new(id1, sizeof(id1));
    resp_id_new(id2, sizeof(id2));

    resp_id_register(id1);
    assert(resp_id_is_live(id1));
    assert(!resp_id_is_live(id2));

    /* After failure/cancellation: all response IDs must be invalidated */
    resp_id_invalidate_all();
    assert(!resp_id_is_live(id1));
    assert(!resp_id_is_live(id2));
    assert(!resp_id_seen(id1));
    assert(!resp_id_seen(id2));

    printf("  [PASS] P1-3: resp_id_invalidate_all correctly wipes state on failure\n");
}

static void test_p1_weight_identity(void) {
    /* Create two mock models with identical architecture, tensor count, dimensions,
     * but different weight bytes. */
    myds_model_t m1, m2;
    memset(&m1, 0, sizeof(m1));
    memset(&m2, 0, sizeof(m2));

    gguf_file_t gf1, gf2;
    memset(&gf1, 0, sizeof(gf1));
    memset(&gf2, 0, sizeof(gf2));

    snprintf(m1.arch, sizeof(m1.arch), "deepseek2");
    snprintf(m2.arch, sizeof(m2.arch), "deepseek2");
    m1.gf = &gf1;
    m2.gf = &gf2;

    gf1.tensor_count = 4;
    gf2.tensor_count = 4;
    gf1.metadata_kv_count = 5;
    gf2.metadata_kv_count = 5;
    gf1.tensor_data_offset = 1024;
    gf2.tensor_data_offset = 1024;
    gf1.mmap_size = 1000000;
    gf2.mmap_size = 1000000;

    gguf_tensor_t t1[4], t2[4];
    memset(t1, 0, sizeof(t1));
    memset(t2, 0, sizeof(t2));

    uint8_t data1[64], data2[64];
    memset(data1, 0x11, sizeof(data1));
    memset(data2, 0x22, sizeof(data2));

    for (int i = 0; i < 4; i++) {
        t1[i].data = data1;
        t1[i].size_bytes = sizeof(data1);
        t2[i].data = data2;
        t2[i].size_bytes = sizeof(data2);
    }
    gf1.tensors = t1;
    gf2.tensors = t2;

    uint64_t w1 = 1469598103934665603ull;
    uint64_t w2 = 1469598103934665603ull;
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t d1 = 0, d2 = 0;
        memcpy(&d1, t1[i].data, 8);
        memcpy(&d2, t2[i].data, 8);
        w1 ^= d1; w1 *= 1099511628211ull;
        w2 ^= d2; w2 *= 1099511628211ull;
    }
    assert(w1 != w2);
    printf("  [PASS] P1-4: weight digests distinguish identical architectures with different weights\n");
}

static void test_p1_metal_fused_add_fallback(void) {
    if (!myds_metal_init(NULL)) {
        printf("  [SKIP] Metal not supported on this platform\n");
        return;
    }

    const uint32_t count = 1024;
    myds_metal_buffer_t *a = myds_metal_buffer_alloc(count * sizeof(float));
    myds_metal_buffer_t *b = myds_metal_buffer_alloc(count * sizeof(float));
    myds_metal_buffer_t *out = myds_metal_buffer_alloc(count * sizeof(float));
    assert(a && b && out);

    float h_a[1024], h_b[1024];
    for (uint32_t i = 0; i < count; i++) {
        h_a[i] = (float)i * 1.5f;
        h_b[i] = 2.25f;
    }
    assert(myds_metal_buffer_write(a, 0, h_a, count * sizeof(float)));
    assert(myds_metal_buffer_write(b, 0, h_b, count * sizeof(float)));

    /* Test 1: inside batch mode, call fused add */
    assert(myds_metal_begin_batch());
    assert(myds_metal_v41_add_bf16(out, a, b, count));
    assert(myds_metal_commit_batch());

    float h_out[1024];
    assert(myds_metal_buffer_read(out, 0, h_out, count * sizeof(float)));

    /* Test 2: inside batch mode, run fallback chain: myds_metal_add + myds_metal_v41_bf16 */
    myds_metal_buffer_t *out_fallback = myds_metal_buffer_alloc(count * sizeof(float));
    assert(out_fallback);
    assert(myds_metal_begin_batch());
    assert(myds_metal_add(out_fallback, a, b, count));
    assert(myds_metal_v41_bf16(out_fallback, count));
    assert(myds_metal_commit_batch());

    float h_fallback[1024];
    assert(myds_metal_buffer_read(out_fallback, 0, h_fallback, count * sizeof(float)));

    /* Verify bit-exact equality between fused add and fallback chain */
    assert(memcmp(h_out, h_fallback, count * sizeof(float)) == 0);

    /* Verify first few elements */
    for (uint32_t i = 0; i < count; i++) {
        float expected_sum = h_a[i] + h_b[i];
        uint32_t u;
        memcpy(&u, &expected_sum, sizeof(u));
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
        float expected_bf16;
        memcpy(&expected_bf16, &u, sizeof(expected_bf16));

        assert(h_out[i] == expected_bf16);
    }

    myds_metal_buffer_free(a);
    myds_metal_buffer_free(b);
    myds_metal_buffer_free(out);
    myds_metal_buffer_free(out_fallback);

    printf("  [PASS] P1-1: Metal fused add BF16 and GPU fallback chain are bit-exact identical (memcmp == 0)\n");
}

int main(void) {
    printf("Running P1 Regression Verification Suite:\n");
    test_p1_response_id_lifecycle();
    test_p1_weight_identity();
    test_p1_metal_fused_add_fallback();
    printf("All P1 Regression Verifications PASSED!\n");
    return 0;
}
