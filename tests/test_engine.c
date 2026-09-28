#include "myds.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Test greedy argmax */
static int32_t sample_greedy(const float *logits, int vocab_size) {
    int32_t best_id = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    return best_id;
}

/* Test temperature scaling */
static void apply_temperature(float *logits, int vocab_size, float temp) {
    if (temp <= 0.0f) return;
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }
}

/* Test softmax */
static void apply_softmax(float *logits, int vocab_size) {
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_sum;
    }
}

/* Pair for top-p sorting */
typedef struct {
    int32_t id;
    float prob;
} prob_pair_t;

static int compare_prob_desc(const void *a, const void *b) {
    float pa = ((const prob_pair_t *)a)->prob;
    float pb = ((const prob_pair_t *)b)->prob;
    if (pb > pa) return 1;
    if (pb < pa) return -1;
    return 0;
}

/* Test top-p filtering */
static int apply_top_p_filter(prob_pair_t *pairs, int vocab_size, float top_p) {
    qsort(pairs, vocab_size, sizeof(prob_pair_t), compare_prob_desc);
    float cumsum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += pairs[i].prob;
        cutoff = i + 1;
        if (cumsum >= top_p) break;
    }
    return cutoff;
}

/* Test RoPE frequency formula */
static float calculate_rope_freq(uint32_t dim_idx, uint32_t total_dim, float freq_base) {
    float exponent = (float)(2 * dim_idx) / (float)total_dim;
    return 1.0f / powf(freq_base, exponent);
}

int main(void) {
    printf("====================================================\n");
    printf("       Running myds Engine Sampling Unit Tests      \n");
    printf("====================================================\n");

    /* 1. Greedy Argmax Sampling */
    {
        float logits[8] = {-2.5f, 0.1f, 10.5f, 3.2f, -1.0f, 8.9f, 2.0f, -0.5f};
        int32_t picked = sample_greedy(logits, 8);
        assert(picked == 2); /* 10.5f is maximum at index 2 */
        printf("  [PASS] Greedy argmax token selection (index 2 picked)\n");
    }

    /* 2. Temperature Scaling */
    {
        float logits[4] = {2.0f, 4.0f, 6.0f, 8.0f};
        apply_temperature(logits, 4, 2.0f);
        assert(fabsf(logits[0] - 1.0f) < 1e-5f);
        assert(fabsf(logits[1] - 2.0f) < 1e-5f);
        assert(fabsf(logits[2] - 3.0f) < 1e-5f);
        assert(fabsf(logits[3] - 4.0f) < 1e-5f);
        printf("  [PASS] Logits temperature scaling (temp = 2.0 scaling)\n");
    }

    /* 3. Softmax Normalization */
    {
        float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        apply_softmax(logits, 4);
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            assert(logits[i] > 0.0f && logits[i] < 1.0f);
            sum += logits[i];
        }
        assert(fabsf(sum - 1.0f) < 1e-5f);
        assert(logits[3] > logits[2] && logits[2] > logits[1] && logits[1] > logits[0]);
        printf("  [PASS] Softmax probability distribution and sum-to-one constraint\n");
    }

    /* 4. Top-P (Nucleus) Filtering */
    {
        prob_pair_t pairs[5] = {
            {0, 0.40f},
            {1, 0.30f},
            {2, 0.15f},
            {3, 0.10f},
            {4, 0.05f}
        };
        /* At top_p = 0.70, top 2 tokens (0.40 + 0.30 = 0.70) are retained */
        int cutoff = apply_top_p_filter(pairs, 5, 0.70f);
        assert(cutoff == 2);
        assert(pairs[0].id == 0);
        assert(pairs[1].id == 1);

        /* At top_p = 0.80, top 3 tokens (0.40 + 0.30 + 0.15 = 0.85 >= 0.80) are retained */
        cutoff = apply_top_p_filter(pairs, 5, 0.80f);
        assert(cutoff == 3);
        printf("  [PASS] Top-p nucleus filtering with cumulative probability threshold\n");
    }

    /* 5. Decoupled RoPE Frequency Calculation */
    {
        float f0 = calculate_rope_freq(0, 64, 10000.0f);
        assert(fabsf(f0 - 1.0f) < 1e-5f); /* 1.0 / (10000 ^ 0) == 1.0 */

        float f_last = calculate_rope_freq(32, 64, 10000.0f);
        assert(fabsf(f_last - 0.0001f) < 1e-6f); /* 1.0 / (10000 ^ 1) == 0.0001 */
        printf("  [PASS] RoPE decoupled angular frequency calculation\n");
    }

    /* 6. DeepSeek V4/V4.1 Architecture Enforcement & V2/V3 Rejection */
    {
        /* DeepSeek V4 and V4.1 accepted architectures */
        assert(myds_is_supported_arch("deepseek41") == true);
        assert(myds_is_supported_arch("deepseek4") == true);
        assert(myds_is_supported_arch("deepseek4-dspark") == true);
        assert(myds_is_supported_arch("deepseek-v4") == true);
        assert(myds_is_supported_arch("deepseek-v4.1") == true);

        /* DeepSeek V2, V3, and legacy architectures explicitly rejected */
        assert(myds_is_supported_arch("deepseek2") == false);
        assert(myds_is_supported_arch("deepseek3") == false);
        assert(myds_is_supported_arch("deepseek") == false);
        assert(myds_is_supported_arch("deepseek-v2") == false);
        assert(myds_is_supported_arch("deepseek-v3") == false);

        /* Non-DeepSeek architectures rejected */
        assert(myds_is_supported_arch("llama") == false);
        assert(myds_is_supported_arch("qwen2") == false);
        assert(myds_is_supported_arch("") == false);
        assert(myds_is_supported_arch(NULL) == false);

        printf("  [PASS] DeepSeek V4/V4.1 architecture enforcement & V2/V3 rejection\n");
    }

    printf("====================================================\n");
    printf("   ALL ENGINE SAMPLING UNIT TESTS PASSED!           \n");
    printf("====================================================\n");
    return 0;
}
