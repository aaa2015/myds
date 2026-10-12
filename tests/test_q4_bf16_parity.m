#include "myds_metal.h"
#include "myds_v41_moe.h"
#include "myds_v41_decode.h"
#include "myds_v41.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while(0)

// Helper to fill Q4_K block with pseudo-random valid data
static void fill_q4_k(void *ptr, size_t bytes, unsigned seed) {
    uint8_t *p = (uint8_t *)ptr;
    for (size_t i = 0; i < bytes; i++) {
        p[i] = (uint8_t)((i * 37 + seed * 101) & 0xff);
    }
    // ensure scales and d are non-zero / finite
    // block_q4_K is 144 bytes:
    // half d (2 bytes)
    // half dmin (2 bytes)
    // uint8_t scales[12]
    // uint8_t qs[256/2 = 128]
    size_t n_blocks = bytes / 144;
    for (size_t b = 0; b < n_blocks; b++) {
        uint16_t *d_ptr = (uint16_t *)(p + b * 144);
        uint16_t *dmin_ptr = (uint16_t *)(p + b * 144 + 2);
        *d_ptr = 0x3c00; // 1.0 in FP16
        *dmin_ptr = 0x3800; // 0.5 in FP16
    }
}

static void fill_f32(void *ptr, size_t count, unsigned seed) {
    float *p = (float *)ptr;
    for (size_t i = 0; i < count; i++) {
        p[i] = (float)(((int)(i * 43 + seed * 97) % 201) - 100) / 100.0f;
    }
}

int main(void) {
    printf("=== Testing Dense Q4_K Direct BF16 Store Parity ===\n");
    REQUIRE(myds_metal_init(NULL));

    // Test 1: myds_metal_v41_gemv_bf16 with Q4_K vs Reference
    {
        uint32_t out_dim = 1280;
        uint32_t in_dim = 5120;
        size_t row_bytes = (in_dim / 256u) * 144u;
        size_t w_bytes = row_bytes * out_dim;
        size_t in_bytes = in_dim * sizeof(float);
        size_t out_bytes = out_dim * sizeof(float);

        myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
        myds_metal_buffer_t *in = myds_metal_buffer_alloc(in_bytes);
        myds_metal_buffer_t *out_ref = myds_metal_buffer_alloc(out_bytes);
        myds_metal_buffer_t *out_test = myds_metal_buffer_alloc(out_bytes);
        REQUIRE(w && in && out_ref && out_test);

        fill_q4_k(myds_metal_buffer_data(w), w_bytes, 123);
        fill_f32(myds_metal_buffer_data(in), in_dim, 456);

        // Reference: myds_metal_gemv + myds_metal_v41_bf16
        memset(myds_metal_buffer_data(out_ref), 0, out_bytes);
        REQUIRE(myds_metal_gemv(out_ref, w, in, out_dim, in_dim, GGML_TYPE_Q4_K));
        REQUIRE(myds_metal_v41_bf16(out_ref, out_dim));

        // Test NR2: myds_metal_v41_gemv_bf16 with NR2
        setenv("MYDS_DENSE_Q4_NR0", "2", 1);
        memset(myds_metal_buffer_data(out_test), 0, out_bytes);
        REQUIRE(myds_metal_v41_gemv_bf16(out_test, w, in, out_dim, in_dim, GGML_TYPE_Q4_K));
        const uint32_t *ref_u32 = (const uint32_t *)myds_metal_buffer_data(out_ref);
        const uint32_t *test_u32 = (const uint32_t *)myds_metal_buffer_data(out_test);
        int diff = 0;
        for (uint32_t i = 0; i < out_dim; i++) {
            if (ref_u32[i] != test_u32[i]) diff++;
        }
        REQUIRE(diff == 0);
        printf("[PASS] myds_metal_v41_gemv_bf16 (Q4_K, NR2): %u x %u bit-exact parity with GEMV+BF16!\n", out_dim, in_dim);

        // Test NR4: myds_metal_v41_gemv_bf16 with NR4
        setenv("MYDS_DENSE_Q4_NR0", "4", 1);
        memset(myds_metal_buffer_data(out_test), 0, out_bytes);
        REQUIRE(myds_metal_v41_gemv_bf16(out_test, w, in, out_dim, in_dim, GGML_TYPE_Q4_K));
        test_u32 = (const uint32_t *)myds_metal_buffer_data(out_test);
        diff = 0;
        for (uint32_t i = 0; i < out_dim; i++) {
            if (ref_u32[i] != test_u32[i]) diff++;
        }
        REQUIRE(diff == 0);
        printf("[PASS] myds_metal_v41_gemv_bf16 (Q4_K, NR4): %u x %u bit-exact parity with GEMV+BF16!\n", out_dim, in_dim);

        myds_metal_buffer_free(w);
        myds_metal_buffer_free(in);
        myds_metal_buffer_free(out_ref);
        myds_metal_buffer_free(out_test);
    }

    // Test 2: myds_metal_attn_output_a_bf16 with Q4_K vs Reference across ranks
    {
        uint32_t ranks[] = { 16, 1024 };
        for (size_t r = 0; r < sizeof(ranks)/sizeof(ranks[0]); r++) {
            uint32_t n_groups = (ranks[r] == 1024) ? 8 : 128;
            uint32_t group_dim = 256;
            uint32_t rank = ranks[r];
            size_t a_group_bytes = (size_t)rank * (group_dim / 256u) * 144u;
            size_t w_bytes = a_group_bytes * n_groups;
            size_t heads_bytes = (size_t)n_groups * group_dim * sizeof(float);
            size_t low_bytes = (size_t)n_groups * rank * sizeof(float);

            myds_metal_buffer_t *out_a = myds_metal_buffer_alloc(w_bytes);
            myds_metal_buffer_t *heads = myds_metal_buffer_alloc(heads_bytes);
            myds_metal_buffer_t *low_ref = myds_metal_buffer_alloc(low_bytes);
            myds_metal_buffer_t *low_test = myds_metal_buffer_alloc(low_bytes);
            REQUIRE(out_a && heads && low_ref && low_test);

            fill_q4_k(myds_metal_buffer_data(out_a), w_bytes, 789 + (unsigned)r);
            fill_f32(myds_metal_buffer_data(heads), n_groups * group_dim, 101 + (unsigned)r);

            // Reference: myds_metal_attn_output_a + myds_metal_v41_bf16
            setenv("MYDS_ATTN_OUT_A_FUSE_BF16", "0", 1);
            memset(myds_metal_buffer_data(low_ref), 0, low_bytes);
            REQUIRE(myds_metal_attn_output_a_bf16(low_ref, out_a, heads, n_groups, group_dim, rank, GGML_TYPE_Q4_K));

            // Test NR2 fused
            setenv("MYDS_ATTN_OUT_A_FUSE_BF16", "1", 1);
            setenv("MYDS_DENSE_Q4_NR0", "2", 1);
            memset(myds_metal_buffer_data(low_test), 0, low_bytes);
            REQUIRE(myds_metal_attn_output_a_bf16(low_test, out_a, heads, n_groups, group_dim, rank, GGML_TYPE_Q4_K));
            const uint32_t *ref_u32 = (const uint32_t *)myds_metal_buffer_data(low_ref);
            const uint32_t *test_u32 = (const uint32_t *)myds_metal_buffer_data(low_test);
            int diff = 0;
            for (uint32_t i = 0; i < n_groups * rank; i++) {
                if (ref_u32[i] != test_u32[i]) diff++;
            }
            REQUIRE(diff == 0);
            printf("[PASS] myds_metal_attn_output_a_bf16 (Q4_K, NR2): %u groups x rank %u bit-exact parity!\n", n_groups, rank);

            // Test NR4 fused
            setenv("MYDS_DENSE_Q4_NR0", "4", 1);
            memset(myds_metal_buffer_data(low_test), 0, low_bytes);
            REQUIRE(myds_metal_attn_output_a_bf16(low_test, out_a, heads, n_groups, group_dim, rank, GGML_TYPE_Q4_K));
            test_u32 = (const uint32_t *)myds_metal_buffer_data(low_test);
            diff = 0;
            for (uint32_t i = 0; i < n_groups * rank; i++) {
                if (ref_u32[i] != test_u32[i]) diff++;
            }
            REQUIRE(diff == 0);
            printf("[PASS] myds_metal_attn_output_a_bf16 (Q4_K, NR4): %u groups x rank %u bit-exact parity!\n", n_groups, rank);

            myds_metal_buffer_free(out_a);
            myds_metal_buffer_free(heads);
            myds_metal_buffer_free(low_ref);
            myds_metal_buffer_free(low_test);
        }
    }

    // Test 3: myds_v41_shared_gate_up_swiglu with Q4_K vs Reference
    {
        uint32_t dim = 7168;
        uint32_t width = 2048;
        size_t row_bytes = (dim / 256u) * 144u;
        size_t w_bytes = row_bytes * width;
        size_t in_bytes = dim * sizeof(float);
        size_t out_bytes = width * sizeof(float);

        myds_metal_buffer_t *gate = myds_metal_buffer_alloc(w_bytes);
        myds_metal_buffer_t *up = myds_metal_buffer_alloc(w_bytes);
        myds_metal_buffer_t *input = myds_metal_buffer_alloc(in_bytes);
        myds_metal_buffer_t *gate_out = myds_metal_buffer_alloc(out_bytes);
        myds_metal_buffer_t *up_out = myds_metal_buffer_alloc(out_bytes);
        myds_metal_buffer_t *swiglu_ref = myds_metal_buffer_alloc(out_bytes);
        myds_metal_buffer_t *swiglu_test = myds_metal_buffer_alloc(out_bytes);
        REQUIRE(gate && up && input && gate_out && up_out && swiglu_ref && swiglu_test);

        fill_q4_k(myds_metal_buffer_data(gate), w_bytes, 202);
        fill_q4_k(myds_metal_buffer_data(up), w_bytes, 303);
        fill_f32(myds_metal_buffer_data(input), dim, 404);

        // Reference: separate gemv(gate) + gemv(up) + shared_swiglu
        REQUIRE(myds_v41_decode_gemv_bf16(gate_out, gate, input, width, dim, GGML_TYPE_Q4_K, false));
        REQUIRE(myds_v41_decode_gemv_bf16(up_out, up, input, width, dim, GGML_TYPE_Q4_K, false));
        REQUIRE(myds_v41_shared_swiglu(swiglu_ref, gate_out, up_out, width, 5.0f));

        // Test: fused kernel_myds_v41_shared_gate_up_swiglu_q4
        memset(myds_metal_buffer_data(swiglu_test), 0, out_bytes);
        REQUIRE(myds_v41_shared_gate_up_swiglu(swiglu_test, gate, up, input, width, dim,
                                               GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 5.0f));

        const uint32_t *ref_u32 = (const uint32_t *)myds_metal_buffer_data(swiglu_ref);
        const uint32_t *test_u32 = (const uint32_t *)myds_metal_buffer_data(swiglu_test);
        int diff = 0;
        for (uint32_t i = 0; i < width; i++) {
            if (ref_u32[i] != test_u32[i]) {
                if (diff < 5) {
                    printf("SharedSwiGLU Mismatch at %u: ref=0x%08x (%.6f) test=0x%08x (%.6f)\n",
                           i, ref_u32[i], *(float*)&ref_u32[i], test_u32[i], *(float*)&test_u32[i]);
                }
                diff++;
            }
        }
        REQUIRE(diff == 0);
        printf("[PASS] myds_v41_shared_gate_up_swiglu (Q4_K): width=%u dim=%u bit-exact parity with separate GEMV+SwiGLU!\n", width, dim);

        myds_metal_buffer_free(gate);
        myds_metal_buffer_free(up);
        myds_metal_buffer_free(input);
        myds_metal_buffer_free(gate_out);
        myds_metal_buffer_free(up_out);
        myds_metal_buffer_free(swiglu_ref);
        myds_metal_buffer_free(swiglu_test);
    }

    // Test 4: Batch-2 Q4_K Shared GEMV Parity across multiple dimensions
    {
        struct {
            uint32_t out_dim;
            uint32_t in_dim;
            const char *desc;
            myds_v41_batch2_projection_kind_t kind;
        } cases[] = {
            { 1280, 2048, "attn_q_a (1280x2048)", MYDS_V41_BATCH2_QKV },
            { 512,  2048, "attn_kv (512x2048)", MYDS_V41_BATCH2_QKV },
            { 40960, 1280, "attn_q_b (40960x1280)", MYDS_V41_BATCH2_QKV },
            { 2048, 8192, "attn_output_b (2048x8192)", MYDS_V41_BATCH2_TAIL },
            { 2048, 2304, "shared_down (2048x2304)", MYDS_V41_BATCH2_SHARED },
        };

        for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
            uint32_t out_dim = cases[c].out_dim;
            uint32_t in_dim = cases[c].in_dim;
            size_t row_bytes = (in_dim / 256u) * 144u;
            size_t w_bytes = row_bytes * out_dim;
            size_t in_bytes = in_dim * sizeof(float);
            size_t out_bytes = out_dim * sizeof(float);

            myds_metal_buffer_t *w = myds_metal_buffer_alloc(w_bytes);
            myds_metal_buffer_t *in[2] = {
                myds_metal_buffer_alloc(in_bytes),
                myds_metal_buffer_alloc(in_bytes)
            };
            myds_metal_buffer_t *ref[2] = {
                myds_metal_buffer_alloc(out_bytes),
                myds_metal_buffer_alloc(out_bytes)
            };
            myds_metal_buffer_t *test[2] = {
                myds_metal_buffer_alloc(out_bytes),
                myds_metal_buffer_alloc(out_bytes)
            };
            REQUIRE(w && in[0] && in[1] && ref[0] && ref[1] && test[0] && test[1]);

            fill_q4_k(myds_metal_buffer_data(w), w_bytes, (unsigned)(501 + c));
            fill_f32(myds_metal_buffer_data(in[0]), in_dim, (unsigned)(601 + c));
            fill_f32(myds_metal_buffer_data(in[1]), in_dim, (unsigned)(701 + c));

            // Reference: Two separate single-token GEMV calls
            for (int t = 0; t < 2; t++) {
                memset(myds_metal_buffer_data(ref[t]), 0, out_bytes);
                REQUIRE(myds_metal_v41_gemv_bf16(ref[t], w, in[t], out_dim, in_dim, GGML_TYPE_Q4_K));
            }

            // Test 4a: Direct myds_metal_v41_gemv_bf16_shared2_type dispatch
            memset(myds_metal_buffer_data(test[0]), 0, out_bytes);
            memset(myds_metal_buffer_data(test[1]), 0, out_bytes);
            REQUIRE(myds_metal_v41_gemv_bf16_shared2_type(test, w, in, out_dim, in_dim, GGML_TYPE_Q4_K));

            for (int t = 0; t < 2; t++) {
                const uint32_t *ref_u32 = (const uint32_t *)myds_metal_buffer_data(ref[t]);
                const uint32_t *test_u32 = (const uint32_t *)myds_metal_buffer_data(test[t]);
                int diff = 0;
                for (uint32_t i = 0; i < out_dim; i++) {
                    if (ref_u32[i] != test_u32[i]) {
                        if (diff < 5) {
                            printf("Shared2 Direct Mismatch [%s, tok %d] at %u: ref=0x%08x (%.6f) test=0x%08x (%.6f)\n",
                                   cases[c].desc, t, i, ref_u32[i], *(float*)&ref_u32[i], test_u32[i], *(float*)&test_u32[i]);
                        }
                        diff++;
                    }
                }
                REQUIRE(diff == 0);
            }
            printf("[PASS] myds_metal_v41_gemv_bf16_shared2_type: %s bit-exact parity for tokens 0 & 1!\n", cases[c].desc);

            // Test 4b: Through myds_v41_batch2_projection_probe
            memset(myds_metal_buffer_data(test[0]), 0, out_bytes);
            memset(myds_metal_buffer_data(test[1]), 0, out_bytes);
            REQUIRE(myds_v41_batch2_projection_probe(test, w, in, out_dim, in_dim, GGML_TYPE_Q4_K, cases[c].kind));

            for (int t = 0; t < 2; t++) {
                const uint32_t *ref_u32 = (const uint32_t *)myds_metal_buffer_data(ref[t]);
                const uint32_t *test_u32 = (const uint32_t *)myds_metal_buffer_data(test[t]);
                int diff = 0;
                for (uint32_t i = 0; i < out_dim; i++) {
                    if (ref_u32[i] != test_u32[i]) {
                        if (diff < 5) {
                            printf("Batch2 Probe Mismatch [%s, tok %d] at %u: ref=0x%08x (%.6f) test=0x%08x (%.6f)\n",
                                   cases[c].desc, t, i, ref_u32[i], *(float*)&ref_u32[i], test_u32[i], *(float*)&test_u32[i]);
                        }
                        diff++;
                    }
                }
                REQUIRE(diff == 0);
            }
            printf("[PASS] myds_v41_batch2_projection_probe: %s bit-exact parity for tokens 0 & 1!\n", cases[c].desc);

            myds_metal_buffer_free(w);
            myds_metal_buffer_free(in[0]);
            myds_metal_buffer_free(in[1]);
            myds_metal_buffer_free(ref[0]);
            myds_metal_buffer_free(ref[1]);
            myds_metal_buffer_free(test[0]);
            myds_metal_buffer_free(test[1]);
        }
    }

    myds_metal_cleanup();
    printf("=== ALL DENSE Q4_K FUSED KERNEL PARITY TESTS PASSED! ===\n");
    return 0;
}
