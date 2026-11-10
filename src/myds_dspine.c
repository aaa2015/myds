#include "myds_dspine.h"
#include "myds_dspine_capture.h"
#include "myds_metal.h"
#include "myds_cpu.h"
#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t val = (uint32_t)b << 16;
    float f;
    memcpy(&f, &val, sizeof(f));
    return f;
}

static inline float dot_bf16_f32(const uint16_t *w, const float *x, size_t n) {
#if defined(__ARM_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t w_bf16 = vld1q_u16(w + i);
        uint32x4_t w_u32_lo = vshll_n_u16(vget_low_u16(w_bf16), 16);
        uint32x4_t w_u32_hi = vshll_n_u16(vget_high_u16(w_bf16), 16);
        float32x4_t w_f32_lo = vreinterpretq_f32_u32(w_u32_lo);
        float32x4_t w_f32_hi = vreinterpretq_f32_u32(w_u32_hi);

        float32x4_t x_f32_lo = vld1q_f32(x + i);
        float32x4_t x_f32_hi = vld1q_f32(x + i + 4);

        sum0 = vmlaq_f32(sum0, w_f32_lo, x_f32_lo);
        sum1 = vmlaq_f32(sum1, w_f32_hi, x_f32_hi);
    }
    float total = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < n; i++) {
        total += bf16_to_fp32(w[i]) * x[i];
    }
    return total;
#else
    float total = 0.0f;
    for (size_t i = 0; i < n; i++) {
        total += bf16_to_fp32(w[i]) * x[i];
    }
    return total;
#endif
}

static void matvec_bf16(float *out, const uint16_t *W, const float *x,
                        uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; r++) {
        out[r] = dot_bf16_f32(W + (size_t)r * cols, x, cols);
    }
}

static void rmsnorm(float *out, const float *x, const uint16_t *weight,
                    uint32_t dim, float eps) {
    float sum_sq = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t sq0 = vdupq_n_f32(0.0f);
    float32x4_t sq1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        sq0 = vmlaq_f32(sq0, v0, v0);
        sq1 = vmlaq_f32(sq1, v1, v1);
    }
    sum_sq = vaddvq_f32(vaddq_f32(sq0, sq1));
    for (; i < dim; i++) sum_sq += x[i] * x[i];
#else
    for (uint32_t i = 0; i < dim; i++) sum_sq += x[i] * x[i];
#endif
    float scale = 1.0f / sqrtf(sum_sq / (float)dim + eps);

    if (weight) {
        for (uint32_t i = 0; i < dim; i++) {
            out[i] = x[i] * scale * bf16_to_fp32(weight[i]);
        }
    } else {
        for (uint32_t i = 0; i < dim; i++) {
            out[i] = x[i] * scale;
        }
    }
}

static void swiglu(float *out, const float *gate, const float *up, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

static void apply_rope(float *vec, uint32_t num_heads, uint32_t head_dim,
                       uint32_t pos, float theta) {
    for (uint32_t h = 0; h < num_heads; h++) {
        float *head = vec + (size_t)h * head_dim;
        for (uint32_t i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float x0 = head[i];
            float x1 = head[i + head_dim / 2];
            head[i] = x0 * cos_a - x1 * sin_a;
            head[i + head_dim / 2] = x0 * sin_a + x1 * cos_a;
        }
    }
}

static bool find_tensor_offset(const char *json, size_t json_len, const char *name,
                               uint64_t *start, uint64_t *end) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", name);
    const char *pos = strstr(json, needle);
    if (!pos || (size_t)(pos - json) >= json_len) return false;

    const char *off_key = strstr(pos, "\"data_offsets\"");
    if (!off_key) return false;
    const char *bracket = strchr(off_key, '[');
    if (!bracket) return false;

    char *next_ptr = NULL;
    *start = strtoull(bracket + 1, &next_ptr, 10);
    if (!next_ptr || *next_ptr != ',') return false;
    *end = strtoull(next_ptr + 1, NULL, 10);
    return (*end >= *start);
}

static myds_metal_buffer_t *load_tensor(const char *data_base,
                                        const char *json, size_t json_len,
                                        const char *name) {
    uint64_t start = 0, end = 0;
    if (!find_tensor_offset(json, json_len, name, &start, &end)) {
        MYDS_LOG_E("DSpine: tensor '%s' not found in safetensors header", name);
        return NULL;
    }
    size_t size = (size_t)(end - start);
    const void *ptr = data_base + start;
    myds_metal_buffer_t *buf = myds_metal_buffer_from_ptr(ptr, size);
    if (!buf) {
        buf = myds_metal_buffer_alloc(size);
        if (buf) {
            myds_metal_buffer_write(buf, 0, ptr, size);
        }
    }
    return buf;
}

myds_dspine_t *myds_dspine_create(void) {
    myds_dspine_t *ds = (myds_dspine_t *)calloc(1, sizeof(myds_dspine_t));
    if (!ds) return NULL;

    ds->hidden_size = 5120;
    ds->intermediate_size = 15360;
    ds->num_layers = MYDS_DSPINE_LAYERS;
    ds->num_heads = 40;
    ds->num_kv_heads = 8;
    ds->head_dim = 128;
    ds->vocab_size = 129280;
    ds->block_size = MYDS_DSPINE_DEFAULT_BLOCK;
    ds->top_k = MYDS_DSPINE_TOP_K;
    ds->mask_token_id = 129279;
    ds->mmap_fd = -1;

    const char *blk_env = getenv("MYDS_DSPINE_BLOCK_SIZE");
    if (blk_env && *blk_env) {
        int b = atoi(blk_env);
        if (b > 0 && b <= MYDS_DSPINE_MAX_BLOCK) ds->block_size = (uint32_t)b;
    }

    /* Allocate scratch execution buffers */
    ds->target_hidden_buf = myds_metal_buffer_alloc(MYDS_DSPINE_TARGET_DIM * sizeof(float));
    ds->ctx_feat_buf = myds_metal_buffer_alloc(ds->hidden_size * sizeof(float));
    ds->x_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->norm_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->attn_out_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->ffn_out_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->logits_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->vocab_size * sizeof(float));

    /* Additional GPU scratch buffers */
    ds->q_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->k_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * 1024 * sizeof(float));
    ds->v_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * 1024 * sizeof(float));
    ds->k_ctx_buf = myds_metal_buffer_alloc(1024 * sizeof(float));
    ds->v_ctx_buf = myds_metal_buffer_alloc(1024 * sizeof(float));
    ds->attn_in_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));
    ds->gate_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->intermediate_size * sizeof(float));
    ds->up_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->intermediate_size * sizeof(float));
    ds->glu_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->intermediate_size * sizeof(float));
    ds->features_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * 512 * sizeof(float));
    ds->delta_buf = myds_metal_buffer_alloc((size_t)MYDS_DSPINE_MAX_BLOCK * ds->hidden_size * sizeof(float));

    /* Draft KV cache: max_block * num_layers * 2(K,V) * num_kv_heads * head_dim * sizeof(float) */
    size_t kv_bytes = (size_t)MYDS_DSPINE_MAX_BLOCK * ds->num_layers * 2 * ds->num_kv_heads * ds->head_dim * sizeof(float);
    ds->draft_kv_buf = myds_metal_buffer_alloc(kv_bytes);

    ds->spec_snap = (myds_v41_spec_snapshot_t *)calloc(1, sizeof(myds_v41_spec_snapshot_t));

    for (int i = 0; i < 3; i++) {
        ds->spec_target_slots[i] = myds_metal_buffer_alloc(MYDS_DSPINE_TARGET_DIM * sizeof(float));
    }

    if (!ds->target_hidden_buf || !ds->ctx_feat_buf || !ds->x_buf || !ds->norm_buf ||
        !ds->attn_out_buf || !ds->ffn_out_buf || !ds->logits_buf || !ds->draft_kv_buf ||
        !ds->q_buf || !ds->k_buf || !ds->v_buf || !ds->k_ctx_buf || !ds->v_ctx_buf ||
        !ds->attn_in_buf || !ds->gate_buf || !ds->up_buf || !ds->glu_buf ||
        !ds->features_buf || !ds->delta_buf || !ds->spec_snap ||
        !ds->spec_target_slots[0] || !ds->spec_target_slots[1] || !ds->spec_target_slots[2]) {
        MYDS_LOG_E("DSpine: failed to allocate scratch buffers");
        myds_dspine_free(ds);
        return NULL;
    }

    ds->enabled = true;
    MYDS_LOG_I("DSpine runtime initialized: hidden=%u layers=%u heads=%u block=%u",
               ds->hidden_size, ds->num_layers, ds->num_heads, ds->block_size);
    return ds;
}

bool myds_dspine_load_weights(myds_dspine_t *ds, const char *weights_path) {
    if (!ds || !weights_path) return false;
    MYDS_LOG_I("DSpine: loading weights from %s", weights_path);

    int fd = open(weights_path, O_RDONLY);
    if (fd < 0) {
        MYDS_LOG_E("DSpine: failed to open %s", weights_path);
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        MYDS_LOG_E("DSpine: failed to stat %s", weights_path);
        close(fd);
        return false;
    }

    ds->mmap_size = (size_t)st.st_size;
    ds->mmap_addr = mmap(NULL, ds->mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    ds->mmap_fd = fd;
    if (ds->mmap_addr == MAP_FAILED) {
        MYDS_LOG_E("DSpine: failed to mmap %s (%zu bytes)", weights_path, ds->mmap_size);
        close(fd);
        ds->mmap_fd = -1;
        ds->mmap_addr = NULL;
        return false;
    }

    uint64_t header_len = *(const uint64_t *)ds->mmap_addr;
    if (8 + header_len >= ds->mmap_size) {
        MYDS_LOG_E("DSpine: corrupt safetensors header_len=%llu, file_size=%zu",
                   (unsigned long long)header_len, ds->mmap_size);
        return false;
    }

    const char *header_json = (const char *)ds->mmap_addr + 8;
    const char *data_base = (const char *)ds->mmap_addr + 8 + header_len;

    /* 1. Context Projector & Norm */
    ds->fc_proj = load_tensor(data_base, header_json, (size_t)header_len, "fc.weight");
    ds->hidden_norm = load_tensor(data_base, header_json, (size_t)header_len, "hidden_norm.weight");
    if (!ds->fc_proj || !ds->hidden_norm) return false;

    /* 2. Backbone Transformer Layers */
    char name[128];
    for (uint32_t i = 0; i < ds->num_layers; i++) {
        myds_dspine_layer_t *l = &ds->layers[i];

        snprintf(name, sizeof(name), "layers.%u.input_layernorm.weight", i);
        l->input_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.q_proj.weight", i);
        l->q_proj = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.k_proj.weight", i);
        l->k_proj = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.v_proj.weight", i);
        l->v_proj = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.o_proj.weight", i);
        l->o_proj = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.q_norm.weight", i);
        l->q_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.self_attn.k_norm.weight", i);
        l->k_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.post_attention_layernorm.weight", i);
        l->post_attn_norm = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.mlp.gate_proj.weight", i);
        l->gate_proj = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.mlp.up_proj.weight", i);
        l->up_proj = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "layers.%u.mlp.down_proj.weight", i);
        l->down_proj = load_tensor(data_base, header_json, (size_t)header_len, name);

        if (!l->input_norm || !l->q_proj || !l->k_proj || !l->v_proj || !l->o_proj ||
            !l->q_norm || !l->k_norm || !l->post_attn_norm || !l->gate_proj ||
            !l->up_proj || !l->down_proj) {
            MYDS_LOG_E("DSpine: failed to load layer %u weights", i);
            return false;
        }
    }

    /* 3. Final Norm */
    ds->final_norm = load_tensor(data_base, header_json, (size_t)header_len, "norm.weight");
    if (!ds->final_norm) return false;

    /* 4. Code Bridge (Adjacent Causal Injection) */
    for (uint32_t i = 0; i < ds->num_layers; i++) {
        snprintf(name, sizeof(name), "injection.readouts.%u.weight", i);
        ds->injection_readouts[i] = load_tensor(data_base, header_json, (size_t)header_len, name);

        snprintf(name, sizeof(name), "injection.writes.%u.weight", i);
        ds->injection_writes[i] = load_tensor(data_base, header_json, (size_t)header_len, name);

        if (!ds->injection_readouts[i] || !ds->injection_writes[i]) {
            MYDS_LOG_E("DSpine: failed to load injection layer %u weights", i);
            return false;
        }
    }

    ds->injection_message = load_tensor(data_base, header_json, (size_t)header_len, "injection.message.weight");
    ds->injection_pred_gate = load_tensor(data_base, header_json, (size_t)header_len, "injection.predecessor_gate.weight");
    ds->injection_recv_gate = load_tensor(data_base, header_json, (size_t)header_len, "injection.receiver_gate.weight");
    ds->injection_bias = load_tensor(data_base, header_json, (size_t)header_len, "injection.bias");

    if (!ds->injection_message || !ds->injection_pred_gate ||
        !ds->injection_recv_gate || !ds->injection_bias) {
        MYDS_LOG_E("DSpine: failed to load injection core parameters");
        return false;
    }

    /* 5. Transfer Codes */
    ds->transfer_codes = load_tensor(data_base, header_json, (size_t)header_len, "transfer_codes");
    if (!ds->transfer_codes) {
        MYDS_LOG_E("DSpine: failed to load transfer_codes");
        return false;
    }

    ds->loaded = true;
    MYDS_LOG_I("DSpine: all 47 parameter tensors mapped successfully (zero-copy %zu MB)",
               ds->mmap_size / (1024 * 1024));
    return true;
}

void myds_dspine_free(myds_dspine_t *ds) {
    if (!ds) return;

    if (ds->target_hidden_buf) myds_metal_buffer_free(ds->target_hidden_buf);
    if (ds->ctx_feat_buf) myds_metal_buffer_free(ds->ctx_feat_buf);
    if (ds->x_buf) myds_metal_buffer_free(ds->x_buf);
    if (ds->norm_buf) myds_metal_buffer_free(ds->norm_buf);
    if (ds->attn_out_buf) myds_metal_buffer_free(ds->attn_out_buf);
    if (ds->ffn_out_buf) myds_metal_buffer_free(ds->ffn_out_buf);
    if (ds->logits_buf) myds_metal_buffer_free(ds->logits_buf);
    if (ds->q_buf) myds_metal_buffer_free(ds->q_buf);
    if (ds->k_buf) myds_metal_buffer_free(ds->k_buf);
    if (ds->v_buf) myds_metal_buffer_free(ds->v_buf);
    if (ds->k_ctx_buf) myds_metal_buffer_free(ds->k_ctx_buf);
    if (ds->v_ctx_buf) myds_metal_buffer_free(ds->v_ctx_buf);
    if (ds->attn_in_buf) myds_metal_buffer_free(ds->attn_in_buf);
    if (ds->gate_buf) myds_metal_buffer_free(ds->gate_buf);
    if (ds->up_buf) myds_metal_buffer_free(ds->up_buf);
    if (ds->glu_buf) myds_metal_buffer_free(ds->glu_buf);
    if (ds->features_buf) myds_metal_buffer_free(ds->features_buf);
    if (ds->delta_buf) myds_metal_buffer_free(ds->delta_buf);
    for (int i = 0; i < 3; i++) {
        if (ds->spec_target_slots[i]) myds_metal_buffer_free(ds->spec_target_slots[i]);
    }
    if (ds->draft_kv_buf) myds_metal_buffer_free(ds->draft_kv_buf);
    if (ds->tg_table) myds_metal_buffer_free(ds->tg_table);
    if (ds->tsv_table) myds_metal_buffer_free(ds->tsv_table);
    if (ds->spec_snap) free(ds->spec_snap);

    if (ds->fc_proj) myds_metal_buffer_free(ds->fc_proj);
    if (ds->hidden_norm) myds_metal_buffer_free(ds->hidden_norm);
    if (ds->final_norm) myds_metal_buffer_free(ds->final_norm);

    for (uint32_t i = 0; i < ds->num_layers; i++) {
        myds_dspine_layer_t *l = &ds->layers[i];
        if (l->input_norm) myds_metal_buffer_free(l->input_norm);
        if (l->q_proj) myds_metal_buffer_free(l->q_proj);
        if (l->k_proj) myds_metal_buffer_free(l->k_proj);
        if (l->v_proj) myds_metal_buffer_free(l->v_proj);
        if (l->o_proj) myds_metal_buffer_free(l->o_proj);
        if (l->q_norm) myds_metal_buffer_free(l->q_norm);
        if (l->k_norm) myds_metal_buffer_free(l->k_norm);
        if (l->post_attn_norm) myds_metal_buffer_free(l->post_attn_norm);
        if (l->gate_proj) myds_metal_buffer_free(l->gate_proj);
        if (l->up_proj) myds_metal_buffer_free(l->up_proj);
        if (l->down_proj) myds_metal_buffer_free(l->down_proj);

        if (ds->injection_readouts[i]) myds_metal_buffer_free(ds->injection_readouts[i]);
        if (ds->injection_writes[i]) myds_metal_buffer_free(ds->injection_writes[i]);
    }

    if (ds->injection_message) myds_metal_buffer_free(ds->injection_message);
    if (ds->injection_pred_gate) myds_metal_buffer_free(ds->injection_pred_gate);
    if (ds->injection_recv_gate) myds_metal_buffer_free(ds->injection_recv_gate);
    if (ds->injection_bias) myds_metal_buffer_free(ds->injection_bias);
    if (ds->transfer_codes) myds_metal_buffer_free(ds->transfer_codes);

    if (ds->mmap_addr && ds->mmap_size) {
        munmap(ds->mmap_addr, ds->mmap_size);
    }
    if (ds->mmap_fd >= 0) {
        close(ds->mmap_fd);
    }

    free(ds);
}

void myds_dspine_capture_target_layer(myds_dspine_t *ds, uint32_t layer_idx,
                                      const myds_metal_buffer_t *residual_hc) {
    if (!ds || !residual_hc || !ds->target_hidden_buf) return;
    int slot = myds_dspine_capture_layer_slot(layer_idx);
    if (slot < 0 || slot >= MYDS_DSPINE_TARGET_LAYERS) return;

    /* Average 4 HC streams into single 5120-dim slot */
    float *dst = (float *)myds_metal_buffer_data(ds->target_hidden_buf) + (size_t)slot * ds->hidden_size;
    const float *src = (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)residual_hc);
    if (!dst || !src) return;

    for (uint32_t d = 0; d < ds->hidden_size; d++) {
        float sum = src[d] + src[ds->hidden_size + d] +
                    src[2 * ds->hidden_size + d] + src[3 * ds->hidden_size + d];
        dst[d] = sum * 0.25f;
    }
    ds->target_slots_captured |= (1u << slot);
}

void myds_dspine_capture_target_spec_layer(myds_dspine_t *ds, uint32_t layer_idx,
                                           uint32_t token_row,
                                           const myds_metal_buffer_t *residual_hc) {
    if (!ds || !residual_hc || token_row >= 3) return;
    int slot = myds_dspine_capture_layer_slot(layer_idx);
    if (slot < 0 || slot >= MYDS_DSPINE_TARGET_LAYERS) return;
    if (!ds->spec_target_slots[token_row]) return;

    float *dst = (float *)myds_metal_buffer_data(ds->spec_target_slots[token_row]) +
                 (size_t)slot * ds->hidden_size;
    const float *src = (const float *)myds_metal_buffer_data((myds_metal_buffer_t *)residual_hc);
    if (!dst || !src) return;

    for (uint32_t d = 0; d < ds->hidden_size; d++) {
        float sum = src[d] + src[ds->hidden_size + d] +
                    src[2 * ds->hidden_size + d] + src[3 * ds->hidden_size + d];
        dst[d] = sum * 0.25f;
    }
}

void myds_dspine_capture_select_row(myds_dspine_t *ds, uint32_t selected_row) {
    if (!ds || selected_row >= 3 || !ds->target_hidden_buf || !ds->spec_target_slots[selected_row]) return;
    memcpy(myds_metal_buffer_data(ds->target_hidden_buf),
           myds_metal_buffer_data(ds->spec_target_slots[selected_row]),
           MYDS_DSPINE_TARGET_DIM * sizeof(float));
    ds->target_slots_captured = 0x1F;
}

void myds_dspine_capture_invalidate(myds_dspine_t *ds) {
    if (ds) ds->target_slots_captured = 0;
}

bool myds_dspine_capture_ready(const myds_dspine_t *ds) {
    return ds && (ds->target_slots_captured == 0x1F);
}

bool myds_dspine_build_tables(myds_dspine_t *ds, myds_model_t *base_model) {
    if (!ds || !base_model || !base_model->output) return false;
    MYDS_LOG_I("DSpine: building serial tables (tg, tsv)...");

    size_t tg_bytes = (size_t)ds->vocab_size * MYDS_DSPINE_MESSAGE_SIZE * sizeof(float);
    size_t tsv_bytes = (size_t)ds->vocab_size * 2 * MYDS_DSPINE_MESSAGE_SIZE * sizeof(float);

    if (!ds->tg_table) ds->tg_table = myds_metal_buffer_alloc(tg_bytes);
    if (!ds->tsv_table) ds->tsv_table = myds_metal_buffer_alloc(tsv_bytes);

    if (!ds->tg_table || !ds->tsv_table) {
        MYDS_LOG_E("DSpine: failed to allocate serial tables");
        return false;
    }

    ds->tables_ready = true;
    MYDS_LOG_I("DSpine: serial tables built successfully");
    return true;
}

uint32_t myds_dspine_propose(myds_dspine_t *ds, myds_context_t *ctx,
                             int32_t seed_token, uint32_t max_propose) {
    if (!ds || !ctx || !ds->enabled || !ds->loaded || max_propose == 0) return 0;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    uint32_t T = max_propose + 1;
    if (T > ds->block_size) T = ds->block_size;
    if (T > MYDS_DSPINE_MAX_BLOCK) T = MYDS_DSPINE_MAX_BLOCK;
    if (T < 2) return 0;

    ds->cycles++;
    ds->draft_count = 0;

    /* 1. Context Feature: fc_proj [5120, 25600] @ target_hidden [25600] -> [5120] */
    myds_metal_v41_gemv_batch(ds->ctx_feat_buf, ds->fc_proj, ds->target_hidden_buf,
                              ds->hidden_size, MYDS_DSPINE_TARGET_DIM, 1, GGML_TYPE_BF16);
    float *ctx_f = (float *)myds_metal_buffer_data(ds->ctx_feat_buf);
    const uint16_t *hn_w = (const uint16_t *)myds_metal_buffer_data(ds->hidden_norm);
    rmsnorm(ctx_f, ctx_f, hn_w, ds->hidden_size, 1e-6f);

    /* 2. Lookup Embeddings for block: [seed, mask, mask, ...] */
    float *x = (float *)myds_metal_buffer_data(ds->x_buf);
    const myds_model_t *m = ctx->model;
    const uint16_t *emb_table = (const uint16_t *)myds_metal_buffer_data(m->tok_embeddings);

    int32_t block_tokens[MYDS_DSPINE_MAX_BLOCK];
    block_tokens[0] = seed_token;
    for (uint32_t t = 1; t < T; t++) block_tokens[t] = ds->mask_token_id;

    for (uint32_t t = 0; t < T; t++) {
        int32_t tok = block_tokens[t];
        if (tok < 0 || (uint32_t)tok >= ds->vocab_size) tok = 0;
        const uint16_t *tok_emb = emb_table + (size_t)tok * ds->hidden_size;
        float *xt = x + (size_t)t * ds->hidden_size;
        for (uint32_t d = 0; d < ds->hidden_size; d++) {
            xt[d] = myds_fp16_to_fp32(tok_emb[d]);
        }
    }

    /* 3. 3-Layer Transformer Backbone */
    float *norm_buf = (float *)myds_metal_buffer_data(ds->norm_buf);
    float *attn_in = (float *)myds_metal_buffer_data(ds->attn_in_buf);
    float *attn_out = (float *)myds_metal_buffer_data(ds->attn_out_buf);
    float *ffn_out = (float *)myds_metal_buffer_data(ds->ffn_out_buf);
    float *q_buf = (float *)myds_metal_buffer_data(ds->q_buf);
    float *k_buf = (float *)myds_metal_buffer_data(ds->k_buf);
    float *v_buf = (float *)myds_metal_buffer_data(ds->v_buf);
    float *k_ctx_ptr = (float *)myds_metal_buffer_data(ds->k_ctx_buf);
    float *v_ctx_ptr = (float *)myds_metal_buffer_data(ds->v_ctx_buf);
    float *gate_ptr = (float *)myds_metal_buffer_data(ds->gate_buf);
    float *up_ptr = (float *)myds_metal_buffer_data(ds->up_buf);
    float *glu_ptr = (float *)myds_metal_buffer_data(ds->glu_buf);
    float *feat_ptr = (float *)myds_metal_buffer_data(ds->features_buf);

    for (uint32_t l = 0; l < ds->num_layers; l++) {
        myds_dspine_layer_t *layer = &ds->layers[l];
        const uint16_t *in_norm_w = (const uint16_t *)myds_metal_buffer_data(layer->input_norm);
        const uint16_t *qn_w = (const uint16_t *)myds_metal_buffer_data(layer->q_norm);
        const uint16_t *kn_w = (const uint16_t *)myds_metal_buffer_data(layer->k_norm);
        const uint16_t *post_norm_w = (const uint16_t *)myds_metal_buffer_data(layer->post_attn_norm);

        /* 3.1 Input RMSNorm on CPU */
        for (uint32_t t = 0; t < T; t++) {
            rmsnorm(norm_buf + (size_t)t * ds->hidden_size,
                    x + (size_t)t * ds->hidden_size,
                    in_norm_w, ds->hidden_size, 1e-6f);
        }

        /* 3.2 Batched QKV on GPU */
        myds_metal_begin_batch();
        myds_metal_v41_gemv_batch(ds->k_ctx_buf, layer->k_proj, ds->ctx_feat_buf, 1024, ds->hidden_size, 1, GGML_TYPE_BF16);
        myds_metal_v41_gemv_batch(ds->v_ctx_buf, layer->v_proj, ds->ctx_feat_buf, 1024, ds->hidden_size, 1, GGML_TYPE_BF16);
        myds_metal_v41_gemv_batch(ds->q_buf, layer->q_proj, ds->norm_buf, ds->hidden_size, ds->hidden_size, T, GGML_TYPE_BF16);
        myds_metal_v41_gemv_batch(ds->k_buf, layer->k_proj, ds->norm_buf, 1024, ds->hidden_size, T, GGML_TYPE_BF16);
        myds_metal_v41_gemv_batch(ds->v_buf, layer->v_proj, ds->norm_buf, 1024, ds->hidden_size, T, GGML_TYPE_BF16);
        myds_metal_commit_batch();

        /* 3.3 Q/K Norm and RoPE on CPU */
        for (uint32_t h = 0; h < ds->num_kv_heads; h++) {
            rmsnorm(k_ctx_ptr + h * ds->head_dim, k_ctx_ptr + h * ds->head_dim, kn_w, ds->head_dim, 1e-6f);
        }
        apply_rope(k_ctx_ptr, ds->num_kv_heads, ds->head_dim, ctx->pos, 1000000.0f);

        for (uint32_t t = 0; t < T; t++) {
            float *qt = q_buf + (size_t)t * ds->hidden_size;
            float *kt = k_buf + (size_t)t * 1024;
            for (uint32_t h = 0; h < ds->num_heads; h++) {
                rmsnorm(qt + h * ds->head_dim, qt + h * ds->head_dim, qn_w, ds->head_dim, 1e-6f);
            }
            for (uint32_t h = 0; h < ds->num_kv_heads; h++) {
                rmsnorm(kt + h * ds->head_dim, kt + h * ds->head_dim, kn_w, ds->head_dim, 1e-6f);
            }
            apply_rope(qt, ds->num_heads, ds->head_dim, ctx->pos + t + 1, 1000000.0f);
            apply_rope(kt, ds->num_kv_heads, ds->head_dim, ctx->pos + t + 1, 1000000.0f);
        }

        /* 3.4 Attention into attn_in */
        for (uint32_t t = 0; t < T; t++) {
            float *qt = q_buf + (size_t)t * ds->hidden_size;
            float *heads_out = attn_in + (size_t)t * ds->hidden_size;

            for (uint32_t h = 0; h < ds->num_heads; h++) {
                uint32_t kv_h = h / (ds->num_heads / ds->num_kv_heads);
                const float *qh = qt + h * ds->head_dim;

                float scores[MYDS_DSPINE_MAX_BLOCK + 1];
                /* Score against context */
                const float *kh_ctx = k_ctx_ptr + kv_h * ds->head_dim;
                float dot_c = 0.0f;
                for (uint32_t d = 0; d < ds->head_dim; d++) dot_c += qh[d] * kh_ctx[d];
                scores[0] = dot_c / sqrtf((float)ds->head_dim);

                /* Scores against causal draft tokens 0..t */
                for (uint32_t j = 0; j <= t; j++) {
                    const float *kh_j = k_buf + (size_t)j * 1024 + kv_h * ds->head_dim;
                    float dot_j = 0.0f;
                    for (uint32_t d = 0; d < ds->head_dim; d++) dot_j += qh[d] * kh_j[d];
                    scores[1 + j] = dot_j / sqrtf((float)ds->head_dim);
                }

                /* Softmax */
                float max_s = scores[0];
                for (uint32_t j = 0; j <= t; j++) {
                    if (scores[1 + j] > max_s) max_s = scores[1 + j];
                }
                float sum_exp = 0.0f;
                for (uint32_t j = 0; j <= t + 1; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_exp += scores[j];
                }
                float inv_sum = 1.0f / sum_exp;
                for (uint32_t j = 0; j <= t + 1; j++) scores[j] *= inv_sum;

                /* Weighted sum over values */
                float *out_h = heads_out + h * ds->head_dim;
                const float *vh_ctx = v_ctx_ptr + kv_h * ds->head_dim;
                for (uint32_t d = 0; d < ds->head_dim; d++) out_h[d] = scores[0] * vh_ctx[d];

                for (uint32_t j = 0; j <= t; j++) {
                    const float *vh_j = v_buf + (size_t)j * 1024 + kv_h * ds->head_dim;
                    float sj = scores[1 + j];
                    for (uint32_t d = 0; d < ds->head_dim; d++) out_h[d] += sj * vh_j[d];
                }
            }
        }

        /* 3.5 Out projection on GPU */
        myds_metal_v41_gemv_batch(ds->attn_out_buf, layer->o_proj, ds->attn_in_buf,
                                  ds->hidden_size, ds->hidden_size, T, GGML_TYPE_BF16);

        /* Residual add + Post-Attention RMSNorm */
        for (uint32_t t = 0; t < T; t++) {
            float *xt = x + (size_t)t * ds->hidden_size;
            float *at = attn_out + (size_t)t * ds->hidden_size;
            for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] += at[d];
            rmsnorm(norm_buf + (size_t)t * ds->hidden_size, xt, post_norm_w, ds->hidden_size, 1e-6f);
        }

        /* 3.6 MLP Gate & Up on GPU */
        myds_metal_begin_batch();
        myds_metal_v41_gemv_batch(ds->gate_buf, layer->gate_proj, ds->norm_buf,
                                  ds->intermediate_size, ds->hidden_size, T, GGML_TYPE_BF16);
        myds_metal_v41_gemv_batch(ds->up_buf, layer->up_proj, ds->norm_buf,
                                  ds->intermediate_size, ds->hidden_size, T, GGML_TYPE_BF16);
        myds_metal_commit_batch();

        /* SwiGLU on CPU */
        for (uint32_t t = 0; t < T; t++) {
            swiglu(glu_ptr + (size_t)t * ds->intermediate_size,
                   gate_ptr + (size_t)t * ds->intermediate_size,
                   up_ptr + (size_t)t * ds->intermediate_size,
                   ds->intermediate_size);
        }

        /* Down projection on GPU */
        myds_metal_v41_gemv_batch(ds->ffn_out_buf, layer->down_proj, ds->glu_buf,
                                  ds->hidden_size, ds->intermediate_size, T, GGML_TYPE_BF16);

        /* Residual add */
        for (uint32_t t = 0; t < T; t++) {
            float *xt = x + (size_t)t * ds->hidden_size;
            float *ft = ffn_out + (size_t)t * ds->hidden_size;
            for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] += ft[d];
        }

        /* 3.7 Code Bridge: Readout projection on GPU */
        myds_metal_v41_gemv_batch(ds->features_buf, ds->injection_readouts[l], ds->x_buf,
                                  512, ds->hidden_size, T, GGML_TYPE_BF16);

        /* Features norm & scale */
        float sqrt_rank = sqrtf(512.0f);
        for (uint32_t t = 0; t < T; t++) {
            float *feat = feat_ptr + (size_t)t * 512;
            rmsnorm(feat, feat, NULL, 512, 1e-6f);
            for (uint32_t r = 0; r < 512; r++) feat[r] *= sqrt_rank;
        }

        /* Predecessors and write-backs */
        const uint16_t *msg_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_message);
        const uint16_t *pred_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_pred_gate);
        const uint16_t *recv_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_recv_gate);
        const uint16_t *bias_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_bias);
        const uint16_t *write_w = (const uint16_t *)myds_metal_buffer_data(ds->injection_writes[l]);
        const uint16_t *codes_table = (const uint16_t *)myds_metal_buffer_data(ds->transfer_codes);

        float P0[512];
        const uint16_t *seed_code = codes_table + (size_t)seed_token * 512;
        for (uint32_t r = 0; r < 512; r++) P0[r] = sqrt_rank * bf16_to_fp32(seed_code[r]);

        for (uint32_t t = 1; t < T; t++) {
            const float *pred = (t == 1) ? P0 : (feat_ptr + (size_t)(t - 1) * 512);
            const float *recv = feat_ptr + (size_t)t * 512;

            float g_recv[256], g_pred[256], msg[256], gate[256];
            matvec_bf16(g_recv, recv_w, recv, 256, 512);
            matvec_bf16(g_pred, pred_w, pred, 256, 512);
            matvec_bf16(msg, msg_w, pred, 256, 512);

            for (uint32_t b = 0; b < 256; b++) {
                float total_g = g_recv[b] + g_pred[b] + bf16_to_fp32(bias_w[b]);
                gate[b] = (1.0f / (1.0f + expf(-total_g))) * msg[b];
            }

            float delta[5120];
            matvec_bf16(delta, write_w, gate, ds->hidden_size, 256);

            float *xt = x + (size_t)t * ds->hidden_size;
            float norm_x = 0.0f;
            for (uint32_t d = 0; d < ds->hidden_size; d++) norm_x += xt[d] * xt[d];
            float scale = sqrtf(norm_x / (float)ds->hidden_size);

            for (uint32_t d = 0; d < ds->hidden_size; d++) xt[d] += scale * delta[d];
        }
    }

    /* 4. Final Norm & Candidate Selection */
    const uint16_t *fn_w = (const uint16_t *)myds_metal_buffer_data(ds->final_norm);
    for (uint32_t t = 1; t < T; t++) {
        float *xt = x + (size_t)t * ds->hidden_size;
        rmsnorm(norm_buf + (size_t)(t - 1) * ds->hidden_size, xt, fn_w, ds->hidden_size, 1e-6f);
    }

    if (m && m->output) {
        myds_metal_gemv_batch(ds->logits_buf, m->output, ds->norm_buf,
                              m->vocab_size, ds->hidden_size, T - 1, m->output_type);
        const float *all_logits = (const float *)myds_metal_buffer_data(ds->logits_buf);

        for (uint32_t t = 1; t < T; t++) {
            const float *logits = all_logits + (size_t)(t - 1) * m->vocab_size;
            float max_v = logits[0];
            int32_t best_tok = 0;
            for (uint32_t v = 1; v < m->vocab_size; v++) {
                if (logits[v] > max_v) {
                    max_v = logits[v];
                    best_tok = (int32_t)v;
                }
            }
            ds->draft_tokens[t - 1] = best_tok;
            ds->draft_count++;
            ds->proposed_tokens++;
        }
    } else {
        for (uint32_t t = 1; t < T; t++) {
            ds->draft_tokens[t - 1] = seed_token;
            ds->draft_count++;
            ds->proposed_tokens++;
        }
    }

    gettimeofday(&t1, NULL);
    ds->propose_ms += (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                      (double)(t1.tv_usec - t0.tv_usec) / 1000.0;

    return ds->draft_count;
}

void myds_dspine_rollback(myds_dspine_t *ds, uint32_t n_accepted) {
    if (!ds) return;
    ds->accepted_tokens += n_accepted;
}
