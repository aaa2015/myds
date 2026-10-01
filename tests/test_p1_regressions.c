#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>

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

    /* Test late response failure: if ID is generated but transport fails */
    char id3[64];
    resp_id_new(id3, sizeof(id3));
    resp_id_invalidate_all();
    assert(!resp_id_is_live(id3));
    assert(!resp_id_seen(id3));

    printf("  [PASS] P1-3 & P2: resp_id lifecycle and late transport failure invalidation\n");
}

static void *worker_hold_slot(void *arg) {
    (void)arg;
    server_vjob_gpu_lock();
    usleep(60000); /* hold for 60ms */
    server_vjob_gpu_unlock();
    return NULL;
}

static void test_p1_vision_queue_slot(void) {
    /* Test 1: Recursive locking in the same thread */
    server_vjob_gpu_lock();
    server_vjob_gpu_lock();
    server_vjob_gpu_unlock();
    server_vjob_gpu_unlock();

    /* Test 2: Timeout when slot is held by another thread */
    setenv("MYDS_QUEUE_TIMEOUT_MS", "15", 1);
    setenv("MYDS_MAX_QUEUE_DEPTH", "1", 1);

    pthread_t th;
    assert(pthread_create(&th, NULL, worker_hold_slot, NULL) == 0);
    usleep(5000); /* ensure worker thread has acquired the slot */

    /* Current thread tries to acquire: timeout is 15ms, worker holds for 60ms */
    server_vjob_gpu_lock();
    assert(server_vjob_gpu_last_error() != 0); /* must fail with timeout (-2) or busy (-1) */

    pthread_join(th, NULL);
    unsetenv("MYDS_QUEUE_TIMEOUT_MS");
    unsetenv("MYDS_MAX_QUEUE_DEPTH");

    /* Test 3: After worker releases, slot can be cleanly acquired again */
    server_vjob_gpu_lock();
    assert(server_vjob_gpu_last_error() == 0);
    server_vjob_gpu_unlock();

    printf("  [PASS] P2: Vision GPU lock integrates with queue timeout and depth limits\n");
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
    gf1.metadata_kv_count = 0;
    gf2.metadata_kv_count = 0;
    gf1.tensor_data_offset = 1024;
    gf2.tensor_data_offset = 1024;
    gf1.mmap_size = 1000000;
    gf2.mmap_size = 1000000;

    gguf_tensor_t t1[4], t2[4];
    memset(t1, 0, sizeof(t1));
    memset(t2, 0, sizeof(t2));

    uint8_t data1[64], data2[64];
    memset(data1, 0x11, sizeof(data1));
    memset(data2, 0x11, sizeof(data2)); /* identical contents */

    for (int i = 0; i < 4; i++) {
        t1[i].data = data1;
        t1[i].size_bytes = sizeof(data1);
        t2[i].data = data2;
        t2[i].size_bytes = sizeof(data2);
    }
    gf1.tensors = t1;
    gf2.tensors = t2;

    /* 1. Same weights at different memory addresses: identity must be identical */
    uint64_t id1 = myds_v41_model_identity(&m1);
    uint64_t id2 = myds_v41_model_identity(&m2);
    assert(id1 == id2);

    /* 2. Modify sampled offset 0: identity must change */
    data2[0] = 0x99;
    uint64_t id_mut0 = myds_v41_model_identity(&m2);
    assert(id_mut0 != id1);

    /* 3. Reset offset 0, modify ONLY unsampled offset 16: identity must change! */
    data2[0] = data1[0];
    data2[16] = 0x77;
    uint64_t id_mut16 = myds_v41_model_identity(&m2);
    assert(id_mut16 != id1);
    assert(id_mut16 != id_mut0);

    /* 4. Reset offset 16, modify offset 33: identity must change */
    data2[16] = data1[16];
    data2[33] = 0x55;
    uint64_t id_mut33 = myds_v41_model_identity(&m2);
    assert(id_mut33 != id1);

    /* 5. With manifest digest, identity is driven by manifest */
    gguf_kv_t kv;
    memset(&kv, 0, sizeof(kv));
    char manifest_key[] = "models.manifest.digest";
    char manifest_val[] = "sha256-verified-manifest-digest-v1";
    kv.key = manifest_key;
    kv.type = GGUF_TYPE_STRING;
    kv.value.val_str = manifest_val;
    gf1.kvs = &kv;
    gf1.metadata_kv_count = 1;
    uint64_t id_manifest = myds_v41_model_identity(&m1);
    assert(id_manifest != id1);

    printf("  [PASS] P1-1: myds_v41_model_identity distinguishes modifications at offset 0, 16, 33 and manifest\n");
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

static void test_p1_unmapped_engram_identity(void) {
    char path[] = "/tmp/test_myds_engram_unmapped_XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);

    /* Write 1024 bytes header + 256 bytes dummy tensor data */
    uint8_t zeros[1024 + 256];
    memset(zeros, 0x42, sizeof(zeros));
    assert(pwrite(fd, zeros, sizeof(zeros), 0) == (ssize_t)sizeof(zeros));

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

    gf1.tensor_count = 1;
    gf2.tensor_count = 1;
    gf1.tensor_data_offset = 1024;
    gf2.tensor_data_offset = 1024;
    gf1.fd = fd;
    gf2.fd = fd;

    gguf_tensor_t t1, t2;
    memset(&t1, 0, sizeof(t1));
    memset(&t2, 0, sizeof(t2));

    /* Unmapped tensor: data is NULL, read via gf->fd and pread */
    t1.data = NULL;
    t1.size_bytes = 256;
    t1.offset = 0;

    t2.data = NULL;
    t2.size_bytes = 256;
    t2.offset = 0;

    gf1.tensors = &t1;
    gf2.tensors = &t2;

    uint64_t id1 = myds_v41_model_identity(&m1);
    uint64_t id2 = myds_v41_model_identity(&m2);
    assert(id1 == id2);

    /* Mutate a single byte in the file inside the unmapped tensor */
    uint8_t mut_byte = 0x99;
    assert(pwrite(fd, &mut_byte, 1, 1024 + 16) == 1);

    uint64_t id_mut = myds_v41_model_identity(&m2);
    assert(id_mut != id1);

    /* Read failure case: close fd or use invalid offset */
    gf2.fd = -1;
    uint64_t id_fail1 = myds_v41_model_identity(&m2);
    uint64_t id_fail2 = myds_v41_model_identity(&m2);
    /* Upon read error, unique non-repeatable fail_salt is injected, disabling disk reuse */
    assert(id_fail1 != id1 && id_fail1 != id_mut);
    assert(id_fail1 != id_fail2);

    close(fd);
    unlink(path);
    printf("  [PASS] P1: Unmapped tensor streamed via fd detects byte changes and disables disk reuse on error\n");
}

static void test_p1_same_process_large_model_isolation(void) {
    /* Two models with > 256MB tensors and no manifest */
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

    gf1.tensor_count = 1;
    gf2.tensor_count = 1;
    gf1.fd = -1;
    gf2.fd = -1;

    gguf_tensor_t t1, t2;
    memset(&t1, 0, sizeof(t1));
    memset(&t2, 0, sizeof(t2));

    const size_t sz = 300ull * 1024ull * 1024ull; /* 300 MB > 256 MB */
    void *buf = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(buf != MAP_FAILED);
    memset(buf, 0x33, 1024);
    memset((uint8_t *)buf + sz / 2, 0x55, 1024);

    t1.data = buf;
    t1.size_bytes = sz;
    t2.data = buf;
    t2.size_bytes = sz;

    gf1.tensors = &t1;
    gf2.tensors = &t2;

    /* 1. Distinct load generations */
    m1.load_generation = 0x10001ull;
    m2.load_generation = 0x20002ull;

    uint64_t id1 = myds_v41_model_identity(&m1);
    uint64_t id2 = myds_v41_model_identity(&m2);
    assert(id1 != id2);

    /* 2. Same model instance preserves identity across multiple identity checks */
    uint64_t id1_again = myds_v41_model_identity(&m1);
    assert(id1 == id1_again);

    /* 3. Even if load_generation is 0 (fallback to model instance pointer),
     * two distinct model pointers in the same process must have different identities */
    m1.load_generation = 0;
    m2.load_generation = 0;
    uint64_t id_ptr1 = myds_v41_model_identity(&m1);
    uint64_t id_ptr2 = myds_v41_model_identity(&m2);
    assert(id_ptr1 != id_ptr2);

    munmap(buf, sz);
    printf("  [PASS] P1: Same-process large models have isolated identities across distinct instances\n");
}

int main(void) {
    printf("Running P1 Regression Verification Suite:\n");
    test_p1_response_id_lifecycle();
    test_p1_vision_queue_slot();
    test_p1_weight_identity();
    test_p1_unmapped_engram_identity();
    test_p1_same_process_large_model_isolation();
    test_p1_metal_fused_add_fallback();
    printf("All P1 Regression Verifications PASSED!\n");
    return 0;
}
