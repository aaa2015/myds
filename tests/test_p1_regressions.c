#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <math.h>

#include "myds.h"
#include "myds_gguf.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_v41_gpu.h"
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
    /* Upon read error, identity explicitly returns 0 */
    assert(id_fail1 == 0);
    assert(id_fail2 == 0);

    /* Verify low_kv_enable on a valid non-null context safely refuses to enable disk reuse when identity is 0 */
    myds_v41_context_t *ctx = myds_v41_context_create_for_model(&m2);
    assert(ctx != NULL);
    assert(!myds_v41_low_kv_enable(ctx, "/tmp"));
    assert(!myds_v41_low_kv_enabled(ctx));

    /* Restore file descriptor so tensor reading succeeds */
    gf2.fd = fd;
    char tmp_kvdsk_dir[] = "/tmp/test_myds_kvdsk_recover_XXXXXX";
    assert(mkdtemp(tmp_kvdsk_dir) != NULL);
    assert(myds_v41_low_kv_enable(ctx, tmp_kvdsk_dir));
    assert(myds_v41_low_kv_enabled(ctx));
    myds_v41_context_free(ctx);

    /* Clean up temporary test files */
    char bin_path[1200];
    snprintf(bin_path, sizeof(bin_path), "%s/kvlow-%016llx.bin", tmp_kvdsk_dir, (unsigned long long)id_mut);
    unlink(bin_path);
    rmdir(tmp_kvdsk_dir);

    close(fd);
    unlink(path);
    printf("  [PASS] P1 & P2: Unmapped tensor streamed via fd detects byte changes, returns 0 on read failure, and recovers on valid fd\n");
}

static void test_p1_tensor_descriptor_binding_identity(void) {
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

    gf1.tensor_count = 2;
    gf2.tensor_count = 2;
    gf1.tensor_data_offset = 1024;
    gf2.tensor_data_offset = 1024;
    gf1.mmap_size = 20000;
    gf2.mmap_size = 20000;

    gguf_tensor_t t1[2], t2[2];
    memset(t1, 0, sizeof(t1));
    memset(t2, 0, sizeof(t2));

    float w_gate[16], w_up[16];
    for (int i = 0; i < 16; i++) {
        w_gate[i] = 1.0f + 0.1f * i;
        w_up[i] = 2.0f + 0.2f * i;
    }

    /* Model 1:
     * Tensor 0: name="ffn_gate_shexp.weight", data=w_gate
     * Tensor 1: name="ffn_up_shexp.weight",   data=w_up
     *
     * Model 2: (weights and layout in memory identical, only tensor names swapped!)
     * Tensor 0: name="ffn_up_shexp.weight",   data=w_gate
     * Tensor 1: name="ffn_gate_shexp.weight", data=w_up
     */
    snprintf(t1[0].name, sizeof(t1[0].name), "ffn_gate_shexp.weight");
    t1[0].n_dims = 1;
    t1[0].dims[0] = 16;
    t1[0].type = GGML_TYPE_F32;
    t1[0].offset = 0;
    t1[0].size_bytes = sizeof(w_gate);
    t1[0].data = w_gate;

    snprintf(t1[1].name, sizeof(t1[1].name), "ffn_up_shexp.weight");
    t1[1].n_dims = 1;
    t1[1].dims[0] = 16;
    t1[1].type = GGML_TYPE_F32;
    t1[1].offset = sizeof(w_gate);
    t1[1].size_bytes = sizeof(w_up);
    t1[1].data = w_up;

    snprintf(t2[0].name, sizeof(t2[0].name), "ffn_up_shexp.weight");
    t2[0].n_dims = 1;
    t2[0].dims[0] = 16;
    t2[0].type = GGML_TYPE_F32;
    t2[0].offset = 0;
    t2[0].size_bytes = sizeof(w_gate);
    t2[0].data = w_gate;

    snprintf(t2[1].name, sizeof(t2[1].name), "ffn_gate_shexp.weight");
    t2[1].n_dims = 1;
    t2[1].dims[0] = 16;
    t2[1].type = GGML_TYPE_F32;
    t2[1].offset = sizeof(w_gate);
    t2[1].size_bytes = sizeof(w_up);
    t2[1].data = w_up;

    gf1.tensors = t1;
    gf2.tensors = t2;

    /* 1. In standard mode: swapping tensor names with identical weights MUST yield different identities */
    uint64_t id1 = myds_v41_model_identity(&m1);
    uint64_t id2 = myds_v41_model_identity(&m2);
    assert(id1 != 0);
    assert(id2 != 0);
    assert(id1 != id2);

    /* 2. In MYDS_FORCE_FULL_DIGEST=1 mode: identities must also differ */
    setenv("MYDS_FORCE_FULL_DIGEST", "1", 1);
    uint64_t id1_force = myds_v41_model_identity(&m1);
    uint64_t id2_force = myds_v41_model_identity(&m2);
    unsetenv("MYDS_FORCE_FULL_DIGEST");
    assert(id1_force != id2_force);

    /* 3. Mutating dimensions must change identity */
    t2[0].dims[0] = 32;
    uint64_t id_dim = myds_v41_model_identity(&m2);
    assert(id_dim != id2);
    t2[0].dims[0] = 16;

    /* 4. Mutating type must change identity */
    t2[0].type = GGML_TYPE_BF16;
    uint64_t id_type = myds_v41_model_identity(&m2);
    assert(id_type != id2);

    printf("  [PASS] P1: Tensor name, dimension, and type descriptors bound to weights alter model identity\n");
}

static void test_p1_engram_metadata_identity(void) {
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
    gf1.metadata_kv_count = 1;
    gf2.metadata_kv_count = 1;
    gf1.tensor_data_offset = 1024;
    gf2.tensor_data_offset = 1024;
    gf1.mmap_size = 10000;
    gf2.mmap_size = 10000;

    gguf_tensor_t t1, t2;
    memset(&t1, 0, sizeof(t1));
    memset(&t2, 0, sizeof(t2));

    uint8_t tensor_data[64];
    memset(tensor_data, 0x5a, sizeof(tensor_data));
    t1.data = tensor_data;
    t1.size_bytes = sizeof(tensor_data);
    t2.data = tensor_data;
    t2.size_bytes = sizeof(tensor_data);

    gf1.tensors = &t1;
    gf2.tensors = &t2;

    /* Two metadata layouts with identical tensor bytes and keys but swapped token_map values at positions 0 and 6 */
    uint32_t map1[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint32_t map2[16] = {6, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    char key_token_map[] = "deepseek41.engram.token_map";
    gguf_kv_t kv1, kv2;
    memset(&kv1, 0, sizeof(kv1));
    memset(&kv2, 0, sizeof(kv2));

    kv1.key = key_token_map;
    kv1.type = GGUF_TYPE_ARRAY;
    kv1.value.val_arr.elem_type = GGUF_TYPE_UINT32;
    kv1.value.val_arr.count = 16;
    kv1.value.val_arr.data = map1;

    kv2.key = key_token_map;
    kv2.type = GGUF_TYPE_ARRAY;
    kv2.value.val_arr.elem_type = GGUF_TYPE_UINT32;
    kv2.value.val_arr.count = 16;
    kv2.value.val_arr.data = map2;

    gf1.kvs = &kv1;
    gf2.kvs = &kv2;

    /* 1. Normal mode (small model, no manifest): swapped token_map must yield different identity */
    uint64_t id1 = myds_v41_model_identity(&m1);
    uint64_t id2 = myds_v41_model_identity(&m2);
    assert(id1 != 0);
    assert(id2 != 0);
    assert(id1 != id2);

    /* 2. Under MYDS_FORCE_FULL_DIGEST=1: must also yield different identity */
    setenv("MYDS_FORCE_FULL_DIGEST", "1", 1);
    uint64_t id1_force = myds_v41_model_identity(&m1);
    uint64_t id2_force = myds_v41_model_identity(&m2);
    unsetenv("MYDS_FORCE_FULL_DIGEST");
    assert(id1_force != id2_force);

    /* 3. Mutating primes array must also change identity */
    uint32_t primes1[4] = {10007, 10009, 10037, 10039};
    uint32_t primes2[4] = {10007, 10009, 10037, 10069};
    char key_primes[] = "deepseek41.engram.primes";
    kv1.key = key_primes;
    kv1.value.val_arr.count = 4;
    kv1.value.val_arr.data = primes1;
    kv2.key = key_primes;
    kv2.value.val_arr.count = 4;
    kv2.value.val_arr.data = primes2;
    uint64_t id_p1 = myds_v41_model_identity(&m1);
    uint64_t id_p2 = myds_v41_model_identity(&m2);
    assert(id_p1 != id_p2);

    printf("  [PASS] P1: GGUF metadata (token_map swap, primes mutation) changes model identity\n");
}

static void test_p2_moe_kernel_and_pso_validation(void) {
    if (!myds_metal_init(NULL)) {
        printf("  [SKIP] Metal not supported on this platform\n");
        return;
    }

    /* Valid MoE test with 256-wide dimensions aligned to IQ2_XXS and Q2_K 256-quantization blocks */
    const uint32_t dim = 256;
    const uint32_t expert_dim = 256;
    const uint32_t n_routed_experts = 16;
    const uint32_t n_activated_experts = 6;

    myds_metal_buffer_t *moe_out = myds_metal_buffer_alloc(dim * sizeof(float));
    myds_metal_buffer_t *x_in = myds_metal_buffer_alloc(dim * sizeof(float));

    /* IQ2_XXS has 66 bytes per 256 weights; Q2_K has 84 bytes per 256 weights */
    uint64_t gate_row_bytes = (dim / 256u) * 66u;
    uint64_t gate_exp_bytes = (uint64_t)expert_dim * gate_row_bytes;
    uint64_t down_row_bytes = (expert_dim / 256u) * 84u;
    uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;

    myds_metal_buffer_t *gate_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * gate_exp_bytes));
    myds_metal_buffer_t *up_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * gate_exp_bytes));
    myds_metal_buffer_t *down_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * down_exp_bytes));
    myds_metal_buffer_t *selected_ids = myds_metal_buffer_alloc(n_activated_experts * sizeof(int32_t));
    myds_metal_buffer_t *selected_weights = myds_metal_buffer_alloc(n_activated_experts * sizeof(float));
    myds_metal_buffer_t *scratch_gate = myds_metal_buffer_alloc(n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_up = myds_metal_buffer_alloc(n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_mid = myds_metal_buffer_alloc(n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_down = myds_metal_buffer_alloc(n_activated_experts * dim * sizeof(float));

    assert(moe_out && x_in && gate_exps && up_exps && down_exps && selected_ids &&
           selected_weights && scratch_gate && scratch_up && scratch_mid && scratch_down);

    float h_x[256];
    for (uint32_t i = 0; i < dim; i++) h_x[i] = 1.0f + 0.01f * (float)(i % 17);
    assert(myds_metal_buffer_write(x_in, 0, h_x, sizeof(h_x)));

    uint8_t q_data[344064];
    memset(q_data, 0x22, sizeof(q_data));
    assert(myds_metal_buffer_write(gate_exps, 0, q_data, (size_t)(n_routed_experts * gate_exp_bytes)));
    assert(myds_metal_buffer_write(up_exps, 0, q_data, (size_t)(n_routed_experts * gate_exp_bytes)));

    /* Avoid cancellation by zeroing dmin in Q2_K down-projection blocks (scale only) */
    uint8_t q_down[344064];
    memset(q_down, 0x22, sizeof(q_down));
    for (uint32_t row = 0; row < n_routed_experts * dim; row++) {
        q_down[row * 84 + 82] = 0;
        q_down[row * 84 + 83] = 0;
    }
    assert(myds_metal_buffer_write(down_exps, 0, q_down, (size_t)(n_routed_experts * down_exp_bytes)));

    int32_t ids[6] = {0, 1, 2, 3, 4, 5};
    float weights[6] = {0.2f, 0.2f, 0.2f, 0.2f, 0.1f, 0.1f};
    assert(myds_metal_buffer_write(selected_ids, 0, ids, sizeof(ids)));
    assert(myds_metal_buffer_write(selected_weights, 0, weights, sizeof(weights)));

    myds_metal_dispatch_count_reset();
    int ret = myds_metal_routed_moe(
        moe_out, x_in, gate_exps, up_exps, down_exps,
        selected_ids, selected_weights,
        scratch_gate, scratch_up, scratch_mid, scratch_down,
        dim, expert_dim, n_routed_experts, n_activated_experts,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K,
        0.0f
    );
    assert(ret == 1);
    uint64_t dispatches = myds_metal_dispatch_count();
    assert(dispatches == 2);

    float h_out[256];
    assert(myds_metal_buffer_read(moe_out, 0, h_out, sizeof(h_out)));
    const double expected_out = 985.310926;
    for (uint32_t i = 0; i < dim; i++) {
        assert(isfinite(h_out[i]));
        double diff = fabs((double)h_out[i] - expected_out);
        assert(diff / expected_out < 1e-4);
    }

    myds_metal_buffer_free(moe_out);
    myds_metal_buffer_free(x_in);
    myds_metal_buffer_free(gate_exps);
    myds_metal_buffer_free(up_exps);
    myds_metal_buffer_free(down_exps);
    myds_metal_buffer_free(selected_ids);
    myds_metal_buffer_free(selected_weights);
    myds_metal_buffer_free(scratch_gate);
    myds_metal_buffer_free(scratch_up);
    myds_metal_buffer_free(scratch_mid);
    myds_metal_buffer_free(scratch_down);

    printf("  [PASS] P2: MoE IQ2/Q2K 256-block quantized execution verified successfully on GPU\n");
}

static void test_p2_moe_batch_mm_tail_cull_validation(void) {
    if (!myds_metal_init(NULL)) {
        printf("  [SKIP] Metal not supported on this platform\n");
        return;
    }

    const uint32_t dim = 256;
    const uint32_t expert_dim = 256;
    const uint32_t n_routed_experts = 16;
    const uint32_t n_activated_experts = 6;
    const uint32_t n_tok = 40;

    uint64_t gate_row_bytes = (dim / 256u) * 66u;
    uint64_t gate_exp_bytes = (uint64_t)expert_dim * gate_row_bytes;
    uint64_t down_row_bytes = (expert_dim / 256u) * 84u;
    uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;

    myds_metal_buffer_t *moe_out = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
    myds_metal_buffer_t *x_in = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
    myds_metal_buffer_t *gate_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * gate_exp_bytes));
    myds_metal_buffer_t *up_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * gate_exp_bytes));
    myds_metal_buffer_t *down_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * down_exp_bytes));
    myds_metal_buffer_t *selected_ids = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * sizeof(int32_t));
    myds_metal_buffer_t *selected_weights = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * sizeof(float));
    myds_metal_buffer_t *scratch_gate = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_up = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_mid = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_down = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * dim * sizeof(float));

    assert(moe_out && x_in && gate_exps && up_exps && down_exps && selected_ids &&
           selected_weights && scratch_gate && scratch_up && scratch_mid && scratch_down);

    float *h_x = malloc((size_t)n_tok * dim * sizeof(float));
    assert(h_x);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < dim; i++) {
            h_x[t * dim + i] = 1.0f + 0.01f * (float)(i % 17);
        }
    }
    assert(myds_metal_buffer_write(x_in, 0, h_x, (size_t)n_tok * dim * sizeof(float)));
    free(h_x);

    uint8_t q_data[344064];
    memset(q_data, 0x22, sizeof(q_data));
    assert(myds_metal_buffer_write(gate_exps, 0, q_data, (size_t)(n_routed_experts * gate_exp_bytes)));
    assert(myds_metal_buffer_write(up_exps, 0, q_data, (size_t)(n_routed_experts * gate_exp_bytes)));

    uint8_t q_down[344064];
    memset(q_down, 0x22, sizeof(q_down));
    for (uint32_t row = 0; row < n_routed_experts * dim; row++) {
        q_down[row * 84 + 82] = 0;
        q_down[row * 84 + 83] = 0;
    }
    assert(myds_metal_buffer_write(down_exps, 0, q_down, (size_t)(n_routed_experts * down_exp_bytes)));

    /* Deterministic non-uniform distribution across 16 experts:
     * Expert active row counts precisely cover:
     * Expert 0: 1 row, Expert 1: 4 rows, Expert 2: 8 rows, Expert 3: 9 rows,
     * Expert 4: 16 rows, Expert 5: 17 rows, Expert 6: 24 rows, Expert 7: 31 rows,
     * Expert 8: 33 rows, Experts 9..15: 13..14 rows each.
     * This thoroughly exercises all dynamic 8-row slice bucketing and tail culling branches. */
    static const int32_t s_tail_cull_expert_ids[240] = {
         8,  7,  6,  5,  4,  9,  8,  7,  6,  5,  4, 10,
         8,  7,  6,  5,  4, 11,  8,  7,  6,  5, 12, 13,
         8,  7,  6, 14,  4,  5,  8,  7,  6,  9, 10, 11,
         8,  7,  6, 12, 13, 14,  8,  7,  6, 15,  4,  5,
         8,  7,  6,  9, 10, 11,  8,  7,  6, 12, 13, 14,
         8,  7,  6, 15,  4,  5,  8,  7,  6,  9, 10, 11,
         8,  7,  6, 12, 13, 14,  8,  7,  6, 15,  4,  5,
         8,  7,  6,  9, 10, 11,  8,  7, 12, 13, 14, 15,
         8,  7,  3,  4,  5,  6,  8,  7,  9, 10, 11, 12,
         8,  7, 13, 14, 15,  2,  8,  7,  3,  4,  5,  6,
         8,  7,  9, 10, 11, 12,  8,  7, 13, 14, 15,  2,
         8,  7,  3,  4,  5,  6,  8,  7,  9, 10, 11, 12,
         8,  7, 13, 14, 15,  2,  8,  3,  4,  5,  6,  7,
         8,  9, 10, 11, 12, 13,  8, 14, 15,  2,  3,  4,
         5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  1,
         2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
        14, 15,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  6,  7,
         8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,
         4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    };

    float *h_weights = malloc((size_t)n_tok * n_activated_experts * sizeof(float));
    assert(h_weights);
    for (uint32_t i = 0; i < n_tok * n_activated_experts; i++) {
        h_weights[i] = (i % 6 < 4) ? 0.2f : 0.1f;
    }
    assert(myds_metal_buffer_write(selected_ids, 0, s_tail_cull_expert_ids, sizeof(s_tail_cull_expert_ids)));
    assert(myds_metal_buffer_write(selected_weights, 0, h_weights, (size_t)n_tok * n_activated_experts * sizeof(float)));
    free(h_weights);

    int ret = myds_metal_routed_moe_batch(
        moe_out, x_in, gate_exps, up_exps, down_exps,
        selected_ids, selected_weights,
        scratch_gate, scratch_up, scratch_mid, scratch_down,
        dim, expert_dim, n_routed_experts, n_activated_experts, n_tok,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K,
        0.0f
    );
    assert(ret == 1);

    float *h_out = malloc((size_t)n_tok * dim * sizeof(float));
    assert(h_out);
    assert(myds_metal_buffer_read(moe_out, 0, h_out, (size_t)n_tok * dim * sizeof(float)));

    /* Verify all 40 tokens output valid, positive, finite values matching golden reference */
    double sum = 0.0;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < dim; i++) {
            float val = h_out[t * dim + i];
            assert(isfinite(val));
            assert(val > 0.0f);
            sum += (double)val;
        }
    }
    uint64_t fnv = 0xcbf29ce484222325ULL;
    const uint8_t *p_bytes = (const uint8_t *)h_out;
    for (size_t b = 0; b < (size_t)n_tok * dim * sizeof(float); b++) {
        fnv = ((fnv ^ p_bytes[b]) * 0x100000001b3ULL);
    }
    /* Bit-exact reference assertions against un-culled baseline */
    assert(fabsf(h_out[0] - 985.08129883f) < 1e-4f);
    assert(fabsf(h_out[1] - 985.08129883f) < 1e-4f);
    assert(fabs(sum - 10087232.5) < 1.0);
    assert(fnv == 0x503770f9a4c89325ULL);
    free(h_out);

    myds_metal_buffer_free(moe_out);
    myds_metal_buffer_free(x_in);
    myds_metal_buffer_free(gate_exps);
    myds_metal_buffer_free(up_exps);
    myds_metal_buffer_free(down_exps);
    myds_metal_buffer_free(selected_ids);
    myds_metal_buffer_free(selected_weights);
    myds_metal_buffer_free(scratch_gate);
    myds_metal_buffer_free(scratch_up);
    myds_metal_buffer_free(scratch_mid);
    myds_metal_buffer_free(scratch_down);

    printf("  [PASS] P2: MoE batch MM path (n_tok=40, non-uniform 1..33 rows) bit-exact to golden reference\n");
}

/* Multi-case non-uniform expert row coverage fixture (tested bit-exact on M2 Ultra vs un-culled kernel) */
static void test_p2_moe_tail_cull_multi_case_golden_equivalence(void) {
    if (!myds_metal_init(NULL)) {
        printf("  [SKIP] Metal not supported on this platform\n");
        return;
    }

    const uint32_t dim = 256;
    const uint32_t expert_dim = 256;
    const uint32_t n_routed_experts = 8;
    const uint32_t n_activated_experts = 6;
    const uint32_t n_tok = 64;

    uint64_t gate_row_bytes = (dim / 256u) * 66u;
    uint64_t gate_exp_bytes = (uint64_t)expert_dim * gate_row_bytes;
    uint64_t down_row_bytes = (expert_dim / 256u) * 84u;
    uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;

    myds_metal_buffer_t *moe_out = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
    myds_metal_buffer_t *x_in = myds_metal_buffer_alloc((size_t)n_tok * dim * sizeof(float));
    myds_metal_buffer_t *gate_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * gate_exp_bytes));
    myds_metal_buffer_t *up_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * gate_exp_bytes));
    myds_metal_buffer_t *down_exps = myds_metal_buffer_alloc((size_t)(n_routed_experts * down_exp_bytes));
    myds_metal_buffer_t *selected_ids = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * sizeof(int32_t));
    myds_metal_buffer_t *selected_weights = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * sizeof(float));
    myds_metal_buffer_t *scratch_gate = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_up = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_mid = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * expert_dim * sizeof(float));
    myds_metal_buffer_t *scratch_down = myds_metal_buffer_alloc((size_t)n_tok * n_activated_experts * dim * sizeof(float));

    assert(moe_out && x_in && gate_exps && up_exps && down_exps && selected_ids &&
           selected_weights && scratch_gate && scratch_up && scratch_mid && scratch_down);

    float *h_x = malloc((size_t)n_tok * dim * sizeof(float));
    assert(h_x);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < dim; i++) {
            h_x[t * dim + i] = 1.0f + 0.01f * (float)(i % 17) + .001f * t;
        }
    }
    assert(myds_metal_buffer_write(x_in, 0, h_x, (size_t)n_tok * dim * sizeof(float)));
    free(h_x);

    uint8_t q_data[344064];
    memset(q_data, 0x22, sizeof(q_data));
    for (unsigned z = 0; z < n_routed_experts * expert_dim; z++) {
        __fp16 h = (__fp16)(.013f + .0001f * (z % 13));
        memcpy(q_data + z * 66, &h, 2);
    }
    assert(myds_metal_buffer_write(gate_exps, 0, q_data, (size_t)(n_routed_experts * gate_exp_bytes)));
    assert(myds_metal_buffer_write(up_exps, 0, q_data, (size_t)(n_routed_experts * gate_exp_bytes)));

    uint8_t q_down[344064];
    memset(q_down, 0x22, sizeof(q_down));
    for (uint32_t row = 0; row < n_routed_experts * dim; row++) {
        __fp16 d = (__fp16)(.007f + .00007f * (row % 17));
        memcpy(q_down + row * 84 + 80, &d, 2);
        q_down[row * 84 + 82] = 0;
        q_down[row * 84 + 83] = 0;
    }
    assert(myds_metal_buffer_write(down_exps, 0, q_down, (size_t)(n_routed_experts * down_exp_bytes)));

    const unsigned cases[] = {1, 4, 8, 9, 16, 17, 24, 31, 33};
    const uint64_t golden_fnv[9] = {
        0xd0d94abd21752177ULL, /* case  1 */
        0x1f7cbd4c5028c4caULL, /* case  4 */
        0x5185f9997de3b766ULL, /* case  8 */
        0xa4073ee7411ad74dULL, /* case  9 */
        0x61c8c0db813324a7ULL, /* case 16 */
        0x5ce5956dcabc4fceULL, /* case 17 */
        0x1beab66e7157c06bULL, /* case 24 */
        0xc411339798b00279ULL, /* case 31 */
        0xedbd0bc4a7c5c802ULL, /* case 33 */
    };

    int32_t *h_ids = malloc((size_t)n_tok * n_activated_experts * sizeof(int32_t));
    float *h_weights = malloc((size_t)n_tok * n_activated_experts * sizeof(float));
    float *h_out = malloc((size_t)n_tok * dim * sizeof(float));
    assert(h_ids && h_weights && h_out);

    for (unsigned c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        unsigned case_rows = cases[c];
        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint32_t k = 0; k < n_activated_experts; k++) {
                h_ids[t * n_activated_experts + k] = (t < case_rows && k == t % 6) ? 6 : (int32_t)k;
                h_weights[t * n_activated_experts + k] = .1f + .01f * ((t + 2 * k) % 11);
            }
        }
        assert(myds_metal_buffer_write(selected_ids, 0, h_ids, (size_t)n_tok * n_activated_experts * sizeof(int32_t)));
        assert(myds_metal_buffer_write(selected_weights, 0, h_weights, (size_t)n_tok * n_activated_experts * sizeof(float)));

        myds_metal_dispatch_count_reset();
        int ret = myds_metal_routed_moe_batch(
            moe_out, x_in, gate_exps, up_exps, down_exps,
            selected_ids, selected_weights,
            scratch_gate, scratch_up, scratch_mid, scratch_down,
            dim, expert_dim, n_routed_experts, n_activated_experts, n_tok,
            GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K,
            0.0f
        );
        assert(ret == 1);
        assert(myds_metal_dispatch_count() == 4);
        assert(myds_metal_buffer_read(moe_out, 0, h_out, (size_t)n_tok * dim * sizeof(float)));

        /* Bitwise hash check against M2 Ultra / un-culled reference run */
        uint64_t fnv = 0xcbf29ce484222325ULL;
        const uint8_t *p_bytes = (const uint8_t *)h_out;
        for (size_t b = 0; b < (size_t)n_tok * dim * sizeof(float); b++) {
            fnv = ((fnv ^ p_bytes[b]) * 0x100000001b3ULL);
        }
        assert(fnv == golden_fnv[c]);
        assert(fabsf(h_out[0] - 692.66223145f) < 1e-4f);

        for (uint32_t i = 0; i < n_tok * dim; i++) {
            assert(isfinite(h_out[i]));
            assert(h_out[i] >= 649.0f && h_out[i] <= 888.0f);
        }
    }

    free(h_ids);
    free(h_weights);
    free(h_out);

    myds_metal_buffer_free(moe_out);
    myds_metal_buffer_free(x_in);
    myds_metal_buffer_free(gate_exps);
    myds_metal_buffer_free(up_exps);
    myds_metal_buffer_free(down_exps);
    myds_metal_buffer_free(selected_ids);
    myds_metal_buffer_free(selected_weights);
    myds_metal_buffer_free(scratch_gate);
    myds_metal_buffer_free(scratch_up);
    myds_metal_buffer_free(scratch_mid);
    myds_metal_buffer_free(scratch_down);

    printf("  [PASS] P2: MoE tail-cull multi-case golden equivalence (rows 1,4,8,9,16,17,24,31,33 bit-exact to un-culled reference)\n");
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

static void fill_buffer_f32(myds_metal_buffer_t *buf, float val) {
    assert(buf);
    float *ptr = (float *)myds_metal_buffer_data(buf);
    size_t words = myds_metal_buffer_size(buf) / sizeof(float);
    for (size_t i = 0; i < words; i++) ptr[i] = val;
}

static void check_buffer_f32(myds_metal_buffer_t *buf, float expected) {
    assert(buf);
    const float *ptr = (const float *)myds_metal_buffer_data(buf);
    size_t words = myds_metal_buffer_size(buf) / sizeof(float);
    for (size_t i = 0; i < words; i++) assert(ptr[i] == expected);
}

static void test_p1_snapshot_overwrite_failure(void) {
    if (!myds_metal_init(NULL)) {
        printf("  [SKIP] Metal not supported on this platform\n");
        return;
    }

    myds_v41_context_t *g = myds_v41_context_create_snapshot_fixture();
    assert(g != NULL);

    /* ── Case 1: Existing valid slot 0 -> partial copy failure during overwrite ── */
    /* 1. Setup slot 0 with valid snapshot at pos 1024 */
    for (uint32_t l = 0; l < 40; l++) {
        fill_buffer_f32(myds_v41_snapshot_get_window(g, l), 7.0f);
    }
    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer_f32(myds_v41_snapshot_get_previous_kv(g, o), 2.0f);
        fill_buffer_f32(myds_v41_snapshot_get_previous_score(g, o), 0.0f);
    }
    myds_v41_snapshot_set_engram_history(g, 11, 22, 33);
    myds_v41_snapshot_set_positions(g, 1024, 1024);

    assert(myds_v41_snapshot_store_now(g, 1024));
    assert(myds_v41_snapshot_get_pos(g, 0) == 1024);
    assert(myds_v41_snapshot_get_staged_pos(g, 0) == 0);
    assert(myds_v41_snapshot_best(g, 1024) == 1024);

    /* 2. Advance live state to pos 1025 with new contents */
    for (uint32_t l = 0; l < 40; l++) {
        fill_buffer_f32(myds_v41_snapshot_get_window(g, l), 9.0f);
    }
    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer_f32(myds_v41_snapshot_get_previous_kv(g, o), 4.0f);
        fill_buffer_f32(myds_v41_snapshot_get_previous_score(g, o), 5.0f);
    }
    myds_v41_snapshot_set_engram_history(g, 44, 55, 66);
    myds_v41_snapshot_set_positions(g, 1025, 1025);

    /* Clear up to 1025 keeps old snapshot at 1024 intact */
    myds_v41_snapshot_clear(g, 1025);
    assert(myds_v41_snapshot_get_pos(g, 0) == 1024);

    /* 3. Inject copy failure at copy #10 */
    myds_metal_inject_copy_failure(10);
    bool stored = myds_v41_snapshot_store_now(g, 1025);
    assert(!stored);
    assert(myds_metal_get_copy_call_count() == 10);

    /* Slot 0 MUST be invalidated so corrupted mixed data is NEVER selectable */
    assert(myds_v41_snapshot_get_pos(g, 0) == 0);
    assert(myds_v41_snapshot_get_staged_pos(g, 0) == 0);
    assert(myds_v41_snapshot_best(g, 1024) == 0);
    assert(myds_v41_snapshot_best(g, 1025) == 0);
    assert(!myds_v41_snapshot_restore(g, 1024));
    assert(!myds_v41_snapshot_restore(g, 1025));

    myds_metal_inject_copy_failure(0);

    /* ── Case 2: Existing valid slot 0 -> final commit_batch failure ── */
    /* 1. Re-populate slot 0 with valid snapshot at pos 1024 */
    for (uint32_t l = 0; l < 40; l++) {
        fill_buffer_f32(myds_v41_snapshot_get_window(g, l), 7.0f);
    }
    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer_f32(myds_v41_snapshot_get_previous_kv(g, o), 2.0f);
        fill_buffer_f32(myds_v41_snapshot_get_previous_score(g, o), 0.0f);
    }
    myds_v41_snapshot_set_engram_history(g, 11, 22, 33);
    myds_v41_snapshot_set_positions(g, 1024, 1024);

    assert(myds_v41_snapshot_store_now(g, 1024));
    assert(myds_v41_snapshot_get_pos(g, 0) == 1024);
    assert(myds_v41_snapshot_best(g, 1024) == 1024);

    /* 2. Advance live state to pos 1026 */
    for (uint32_t l = 0; l < 40; l++) {
        fill_buffer_f32(myds_v41_snapshot_get_window(g, l), 8.0f);
    }
    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer_f32(myds_v41_snapshot_get_previous_kv(g, o), 3.0f);
        fill_buffer_f32(myds_v41_snapshot_get_previous_score(g, o), 1.0f);
    }
    myds_v41_snapshot_set_engram_history(g, 77, 88, 99);
    myds_v41_snapshot_set_positions(g, 1026, 1026);

    /* 3. Inject commit failure on the 2nd commit_batch call (the batch commit) */
    myds_metal_inject_commit_failure(2);
    stored = myds_v41_snapshot_store_now(g, 1026);
    assert(!stored);
    assert(myds_metal_get_commit_call_count() == 2);

    /* Slot 0 MUST be invalidated */
    assert(myds_v41_snapshot_get_pos(g, 0) == 0);
    assert(myds_v41_snapshot_get_staged_pos(g, 0) == 0);
    assert(myds_v41_snapshot_best(g, 1024) == 0);
    assert(myds_v41_snapshot_best(g, 1026) == 0);
    assert(!myds_v41_snapshot_restore(g, 1024));
    assert(!myds_v41_snapshot_restore(g, 1026));

    myds_metal_inject_commit_failure(0);

    /* ── Case 3: Subsequent successful store recovers normal operation ── */
    /* 1. Advance to 1027 and store without faults */
    for (uint32_t l = 0; l < 40; l++) {
        fill_buffer_f32(myds_v41_snapshot_get_window(g, l), 10.0f);
    }
    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer_f32(myds_v41_snapshot_get_previous_kv(g, o), 5.0f);
        fill_buffer_f32(myds_v41_snapshot_get_previous_score(g, o), 6.0f);
    }
    myds_v41_snapshot_set_engram_history(g, 101, 102, 103);
    myds_v41_snapshot_set_positions(g, 1027, 1027);

    stored = myds_v41_snapshot_store_now(g, 1027);
    assert(stored);
    assert(myds_v41_snapshot_get_pos(g, 0) == 1027);
    assert(myds_v41_snapshot_get_staged_pos(g, 0) == 0);
    assert(myds_v41_snapshot_best(g, 1027) == 1027);

    /* 2. Zero out live buffers and restore from 1027 */
    for (uint32_t l = 0; l < 40; l++) {
        fill_buffer_f32(myds_v41_snapshot_get_window(g, l), 0.0f);
    }
    for (uint32_t o = 0; o < 4; o++) {
        fill_buffer_f32(myds_v41_snapshot_get_previous_kv(g, o), 0.0f);
        fill_buffer_f32(myds_v41_snapshot_get_previous_score(g, o), 0.0f);
    }
    myds_v41_snapshot_set_engram_history(g, 0, 0, 0);
    myds_v41_snapshot_set_positions(g, 0, 0);

    assert(myds_v41_snapshot_restore(g, 1027));
    assert(myds_v41_snapshot_get_valid(g));

    uint32_t hist[3];
    myds_v41_snapshot_get_engram_history(g, hist);
    assert(hist[0] == 101 && hist[1] == 102 && hist[2] == 103);

    for (uint32_t l = 0; l < 40; l++) {
        check_buffer_f32(myds_v41_snapshot_get_window(g, l), 10.0f);
    }
    for (uint32_t o = 0; o < 4; o++) {
        check_buffer_f32(myds_v41_snapshot_get_previous_kv(g, o), 5.0f);
        check_buffer_f32(myds_v41_snapshot_get_previous_score(g, o), 6.0f);
    }

    myds_v41_context_free(g);
    printf("  [PASS] P1: Snapshot slot 0 invalidation on copy/commit failure prevents selecting corrupted state, and recovers cleanly on next store\n");
}

static void test_p1_macroblock_prefill_invariants(void) {
    /* 1. Test macroblock size environment resolution & clamping */
    unsetenv("MYDS_PREFILL_MACROBLOCK_SIZE");
    assert(myds_v41_resolve_macroblock_size() == 4096u);

    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "200", 1);
    assert(myds_v41_resolve_macroblock_size() == 512u); /* clamp min */

    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "1000", 1);
    assert(myds_v41_resolve_macroblock_size() == 1024u); /* round up to multiple of 512 */

    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "2048", 1);
    assert(myds_v41_resolve_macroblock_size() == 2048u);

    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "32768", 1);
    assert(myds_v41_resolve_macroblock_size() == 16384u); /* clamp max */

    /* Negative, zero, and malformed inputs must safely fall back to default without integer underflow */
    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "-1", 1);
    assert(myds_v41_resolve_macroblock_size() == 4096u);

    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "-100", 1);
    assert(myds_v41_resolve_macroblock_size() == 4096u);

    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "0", 1);
    assert(myds_v41_resolve_macroblock_size() == 4096u);

    setenv("MYDS_PREFILL_MACROBLOCK_SIZE", "invalid", 1);
    assert(myds_v41_resolve_macroblock_size() == 4096u);

    unsetenv("MYDS_PREFILL_MACROBLOCK_SIZE");
    assert(myds_v41_resolve_macroblock_size() == 4096u);

    /* 2. Test carry staging across macroblock boundaries */
    if (!myds_metal_init(NULL)) {
        printf("  [SKIP] Metal not supported for macroblock carry test\n");
        return;
    }

    myds_v41_gpu_scratch_t *scr = myds_v41_gpu_scratch_create(2048);
    assert(scr != NULL);

    myds_metal_buffer_t *window = myds_metal_buffer_alloc(128u * 512u * sizeof(float));
    assert(window != NULL);
    fill_buffer_f32(window, 123.45f);

    /* Stage carry at boundary 512 (which is > 128 carry tokens) */
    int ok = myds_v41_gpu_carry_stage(scr, window, 512u);
    assert(ok == 1);

    /* Stage carry at boundary 1024 */
    ok = myds_v41_gpu_carry_stage(scr, window, 1024u);
    assert(ok == 1);

    myds_metal_buffer_free(window);
    myds_v41_gpu_scratch_free(scr);

    printf("  [PASS] P1: Macroblock configuration bounds, stride snapping, and carry staging invariants\n");
}

static void test_p2_tiled_indexer_causal_equivalence(void) {
    if (!myds_metal_init(NULL)) {
        printf("  [SKIP] Metal not supported for tiled indexer test\n");
        return;
    }

    const uint32_t n_tokens = 32;
    const uint32_t n_comp = 64;
    const uint32_t n_head = 32;
    const uint32_t head_dim = 128;
    const uint32_t pos0 = 100;
    const uint32_t ratio = 2;

    size_t q_bytes = (size_t)n_tokens * n_head * head_dim * sizeof(float);
    size_t w_bytes = (size_t)n_tokens * n_head * sizeof(float);
    size_t k_bytes = (size_t)n_comp * head_dim * sizeof(float);
    size_t s_bytes = (size_t)n_tokens * n_comp * sizeof(float);

    myds_metal_buffer_t *q_buf = myds_metal_buffer_alloc(q_bytes);
    myds_metal_buffer_t *w_buf = myds_metal_buffer_alloc(w_bytes);
    myds_metal_buffer_t *k_buf = myds_metal_buffer_alloc(k_bytes);
    myds_metal_buffer_t *s_direct = myds_metal_buffer_alloc(s_bytes);
    myds_metal_buffer_t *s_tiled = myds_metal_buffer_alloc(s_bytes);
    assert(q_buf && w_buf && k_buf && s_direct && s_tiled);

    float *q_ptr = (float *)myds_metal_buffer_data(q_buf);
    float *w_ptr = (float *)myds_metal_buffer_data(w_buf);
    float *k_ptr = (float *)myds_metal_buffer_data(k_buf);
    for (size_t i = 0; i < n_tokens * n_head * head_dim; i++) {
        q_ptr[i] = ((float)(i % 19) - 9.0f) * 0.05f;
    }
    for (size_t i = 0; i < n_tokens * n_head; i++) {
        w_ptr[i] = 1.0f + 0.02f * (float)(i % 11);
    }
    for (size_t i = 0; i < n_comp * head_dim; i++) {
        k_ptr[i] = ((float)(i % 23) - 11.0f) * 0.04f;
    }

    /* 1. Run direct indexer */
    setenv("MYDS_INDEXER_TILED", "0", 1);
    assert(myds_metal_begin_batch());
    assert(myds_v41_gpu_indexscores_chunk(s_direct, q_buf, w_buf, k_buf,
                                          n_comp, n_head, n_tokens, pos0, ratio));
    assert(myds_metal_commit_batch());

    /* 2. Run hardware tiled indexer */
    setenv("MYDS_INDEXER_TILED", "1", 1);
    assert(myds_metal_begin_batch());
    assert(myds_v41_gpu_indexscores_chunk(s_tiled, q_buf, w_buf, k_buf,
                                          n_comp, n_head, n_tokens, pos0, ratio));
    assert(myds_metal_commit_batch());

    unsetenv("MYDS_INDEXER_TILED");

    /* 3. Compare direct and tiled */
    const float *sd = (const float *)myds_metal_buffer_data(s_direct);
    const float *st = (const float *)myds_metal_buffer_data(s_tiled);
    float max_diff = 0.0f;
    for (size_t i = 0; i < n_tokens * n_comp; i++) {
        float vd = sd[i], vt = st[i];
        if (isinf(vd) || isinf(vt)) {
            assert(vd == vt);
        } else {
            float diff = fabsf(vd - vt);
            if (diff > max_diff) max_diff = diff;
            assert(diff < 1e-4f);
        }
    }

    myds_metal_buffer_free(q_buf);
    myds_metal_buffer_free(w_buf);
    myds_metal_buffer_free(k_buf);
    myds_metal_buffer_free(s_direct);
    myds_metal_buffer_free(s_tiled);

    printf("  [PASS] P2: Hardware-tiled indexer causal equivalence to direct indexer (max_diff=%e)\n", max_diff);
}

int main(void) {
    printf("Running P1 Regression Verification Suite:\n");
    test_p1_response_id_lifecycle();
    test_p1_vision_queue_slot();
    test_p1_weight_identity();
    test_p1_tensor_descriptor_binding_identity();
    test_p1_engram_metadata_identity();
    test_p1_unmapped_engram_identity();
    test_p1_same_process_large_model_isolation();
    test_p1_metal_fused_add_fallback();
    test_p2_moe_kernel_and_pso_validation();
    test_p2_moe_batch_mm_tail_cull_validation();
    test_p2_moe_tail_cull_multi_case_golden_equivalence();
    test_p1_snapshot_overwrite_failure();
    test_p1_macroblock_prefill_invariants();
    test_p2_tiled_indexer_causal_equivalence();
    printf("All P1 Regression Verifications PASSED!\n");
    return 0;
}
