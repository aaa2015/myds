#include "myds_dspine_capture.h"
#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>

/* DeepSeek-V4.1-Flash-1017 special token IDs for automatic supervised loss_mask */
#define V41_TOK_EOS       1
#define V41_TOK_SYSTEM    128799
#define V41_TOK_USER      128803
#define V41_TOK_ASSISTANT 128804
#define V41_TOK_THINK_BEG 128821
#define V41_TOK_THINK_END 128822

static int s_target_layers[MYDS_DSPINE_MAX_TAPS] = {2, 10, 20, 30, 38};
static int s_num_taps = 5;
static int s_init_done = 0;

static uint32_t s_curr_tokens = 0;
static int32_t *s_tokens_buf = NULL;
static float *s_mask_buf = NULL;
static uint16_t *s_hidden_buf = NULL;     /* [curr_tokens, num_taps * 5120] in BF16 */
static uint16_t *s_last_norm_buf = NULL;  /* [curr_tokens, 5120] in BF16 */
static uint32_t s_shard_counter = 0;

static float *s_override_mask = NULL;
static uint32_t s_override_n = 0;
static char s_override_path[512] = {0};

static inline uint16_t float_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t lsb = (x >> 16) & 1;
    uint32_t bias = 0x7fff + lsb;
    x += bias;
    return (uint16_t)(x >> 16);
}

static void ensure_init(void) {
    if (s_init_done) return;
    s_init_done = 1;
    const char *custom = getenv("MYDS_DSPINE_TARGET_LAYERS");
    if (custom && *custom) {
        int n = 0;
        char buf[256];
        strncpy(buf, custom, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char *tok = strtok(buf, ", ");
        while (tok && n < MYDS_DSPINE_MAX_TAPS) {
            int l = atoi(tok);
            if (l >= 0 && l < 40) {
                s_target_layers[n++] = l;
            }
            tok = strtok(NULL, ", ");
        }
        if (n > 0) s_num_taps = n;
    }
}

bool myds_dspine_capture_is_enabled(void) {
    const char *dir = getenv("MYDS_DSPINE_CAPTURE_DIR");
    return dir != NULL && *dir != '\0';
}

int myds_dspine_capture_layer_slot(uint32_t layer_idx) {
    ensure_init();
    for (int i = 0; i < s_num_taps; i++) {
        if ((uint32_t)s_target_layers[i] == layer_idx) return i;
    }
    return -1;
}

void myds_dspine_capture_set_override(const float *loss_mask, uint32_t n_tokens,
                                      const char *exact_filepath) {
    if (s_override_mask) {
        free(s_override_mask);
        s_override_mask = NULL;
    }
    s_override_n = 0;
    s_override_path[0] = '\0';

    if (loss_mask && n_tokens > 0) {
        s_override_mask = (float *)malloc(n_tokens * sizeof(float));
        if (s_override_mask) {
            memcpy(s_override_mask, loss_mask, n_tokens * sizeof(float));
            s_override_n = n_tokens;
        }
    }
    if (exact_filepath && *exact_filepath) {
        strncpy(s_override_path, exact_filepath, sizeof(s_override_path) - 1);
        s_override_path[sizeof(s_override_path) - 1] = '\0';
    }
}

static void fill_supervised_mask(float *mask, const int32_t *tokens, uint32_t n_tokens) {
    if (!mask) return;
    if (!tokens) {
        for (uint32_t i = 0; i < n_tokens; i++) mask[i] = 1.0f;
        return;
    }
    bool has_assistant = false;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] == V41_TOK_ASSISTANT) {
            has_assistant = true;
            break;
        }
    }
    if (!has_assistant) {
        for (uint32_t i = 0; i < n_tokens; i++) mask[i] = 1.0f;
        return;
    }

    bool on = false;
    bool just_started = false;
    for (uint32_t i = 0; i < n_tokens; i++) {
        int32_t t = tokens[i];
        if (t == V41_TOK_ASSISTANT) {
            on = true;
            just_started = true;
            mask[i] = 0.0f;
            continue;
        }
        if (just_started) {
            just_started = false;
            if (t == V41_TOK_THINK_BEG || t == V41_TOK_THINK_END) {
                mask[i] = 0.0f;
                continue;
            }
        }
        if (t == V41_TOK_EOS) {
            mask[i] = on ? 1.0f : 0.0f;
            on = false;
            continue;
        }
        if (t == V41_TOK_USER || t == V41_TOK_SYSTEM) {
            on = false;
            mask[i] = 0.0f;
            continue;
        }
        mask[i] = on ? 1.0f : 0.0f;
    }
}

bool myds_dspine_capture_begin(uint32_t n_tokens, const int32_t *tokens) {
    if (!myds_dspine_capture_is_enabled() || n_tokens == 0) return false;
    ensure_init();

    /* Clean any previous buffer */
    if (s_tokens_buf) { free(s_tokens_buf); s_tokens_buf = NULL; }
    if (s_mask_buf) { free(s_mask_buf); s_mask_buf = NULL; }
    if (s_hidden_buf) { free(s_hidden_buf); s_hidden_buf = NULL; }
    if (s_last_norm_buf) { free(s_last_norm_buf); s_last_norm_buf = NULL; }

    s_curr_tokens = n_tokens;
    s_tokens_buf = (int32_t *)malloc(n_tokens * sizeof(int32_t));
    if (tokens && s_tokens_buf) {
        memcpy(s_tokens_buf, tokens, n_tokens * sizeof(int32_t));
    }
    s_mask_buf = (float *)malloc(n_tokens * sizeof(float));
    if (s_mask_buf) {
        if (s_override_mask && s_override_n == n_tokens) {
            memcpy(s_mask_buf, s_override_mask, n_tokens * sizeof(float));
        } else {
            fill_supervised_mask(s_mask_buf, tokens, n_tokens);
        }
    }
    if (s_override_mask) {
        free(s_override_mask);
        s_override_mask = NULL;
        s_override_n = 0;
    }

    size_t hidden_elems = (size_t)n_tokens * s_num_taps * MYDS_DSPINE_DIM;
    s_hidden_buf = (uint16_t *)calloc(hidden_elems, sizeof(uint16_t));

    size_t norm_elems = (size_t)n_tokens * MYDS_DSPINE_DIM;
    s_last_norm_buf = (uint16_t *)calloc(norm_elems, sizeof(uint16_t));

    if (!s_tokens_buf || !s_mask_buf || !s_hidden_buf || !s_last_norm_buf) {
        MYDS_LOG_E("DSpine capture: failed to allocate memory for %u tokens", n_tokens);
        if (s_tokens_buf) { free(s_tokens_buf); s_tokens_buf = NULL; }
        if (s_mask_buf) { free(s_mask_buf); s_mask_buf = NULL; }
        if (s_hidden_buf) { free(s_hidden_buf); s_hidden_buf = NULL; }
        if (s_last_norm_buf) { free(s_last_norm_buf); s_last_norm_buf = NULL; }
        s_curr_tokens = 0;
        return false;
    }

    MYDS_LOG_I("DSpine capture: started recording for %u tokens (taps=%d)", n_tokens, s_num_taps);
    return true;
}

void myds_dspine_capture_layer_hc(uint32_t layer_idx, const float *residual_hc,
                                  uint32_t offset, uint32_t count) {
    if (!s_hidden_buf || !residual_hc) return;
    int slot = myds_dspine_capture_layer_slot(layer_idx);
    if (slot < 0) return;

    const size_t concat_dim = (size_t)s_num_taps * MYDS_DSPINE_DIM;
    const size_t slot_offset = (size_t)slot * MYDS_DSPINE_DIM;

    for (uint32_t t = 0; t < count; t++) {
        uint32_t tok_idx = offset + t;
        if (tok_idx >= s_curr_tokens) break;

        const float *hc = residual_hc + (size_t)t * 4u * MYDS_DSPINE_DIM;
        uint16_t *dst = s_hidden_buf + (size_t)tok_idx * concat_dim + slot_offset;

        for (uint32_t j = 0; j < MYDS_DSPINE_DIM; j++) {
            float mean = 0.25f * (hc[j] + hc[MYDS_DSPINE_DIM + j] +
                                  hc[2u * MYDS_DSPINE_DIM + j] + hc[3u * MYDS_DSPINE_DIM + j]);
            dst[j] = float_to_bf16(mean);
        }
    }
}

void myds_dspine_capture_last_norm(const float *x_norm, uint32_t offset, uint32_t count) {
    if (!s_last_norm_buf || !x_norm) return;

    for (uint32_t t = 0; t < count; t++) {
        uint32_t tok_idx = offset + t;
        if (tok_idx >= s_curr_tokens) break;

        const float *src = x_norm + (size_t)t * MYDS_DSPINE_DIM;
        uint16_t *dst = s_last_norm_buf + (size_t)tok_idx * MYDS_DSPINE_DIM;

        for (uint32_t j = 0; j < MYDS_DSPINE_DIM; j++) {
            dst[j] = float_to_bf16(src[j]);
        }
    }
}

bool myds_dspine_capture_finish_and_save(void) {
    if (!s_hidden_buf || !s_last_norm_buf || !s_tokens_buf || s_curr_tokens == 0) return false;

    const char *dir = getenv("MYDS_DSPINE_CAPTURE_DIR");
    if (!dir || !*dir) return false;

    /* Ensure directory exists */
    mkdir(dir, 0755);

    char filepath[512];
    char tmp_filepath[544];
    if (s_override_path[0] != '\0') {
        strncpy(filepath, s_override_path, sizeof(filepath) - 1);
        filepath[sizeof(filepath) - 1] = '\0';
        s_override_path[0] = '\0';
        ++s_shard_counter;
    } else {
        time_t now = time(NULL);
        struct tm tm_buf;
        localtime_r(&now, &tm_buf);
        snprintf(filepath, sizeof(filepath), "%s/shard-%04d%02d%02d-%02d%02d%02d-%05u.safetensors",
                 dir, tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                 tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, ++s_shard_counter);
    }
    snprintf(tmp_filepath, sizeof(tmp_filepath), "%s.tmp", filepath);

    FILE *f = fopen(tmp_filepath, "wb");
    if (!f) {
        MYDS_LOG_E("DSpine capture: failed to open %s for writing", tmp_filepath);
        return false;
    }

    uint32_t T = s_curr_tokens;
    size_t concat_dim = (size_t)s_num_taps * MYDS_DSPINE_DIM;

    size_t ids_bytes = (size_t)T * sizeof(int32_t);
    size_t mask_bytes = (size_t)T * sizeof(float);
    size_t hidden_bytes = (size_t)T * concat_dim * sizeof(uint16_t);
    size_t last_norm_bytes = (size_t)T * MYDS_DSPINE_DIM * sizeof(uint16_t);

    size_t off0 = 0;
    size_t off1 = off0 + ids_bytes;
    size_t off2 = off1 + mask_bytes;
    size_t off3 = off2 + hidden_bytes;
    size_t off4 = off3 + last_norm_bytes;

    /* Prepare JSON header */
    char header_buf[2048];
    int hlen = snprintf(header_buf, sizeof(header_buf),
        "{\"input_ids\":{\"dtype\":\"I32\",\"shape\":[%u],\"data_offsets\":[%zu,%zu]},"
        "\"loss_mask\":{\"dtype\":\"F32\",\"shape\":[%u],\"data_offsets\":[%zu,%zu]},"
        "\"hidden_states\":{\"dtype\":\"BF16\",\"shape\":[%u,%zu],\"data_offsets\":[%zu,%zu]},"
        "\"target_last_hidden_states\":{\"dtype\":\"BF16\",\"shape\":[%u,%u],\"data_offsets\":[%zu,%zu]},"
        "\"__metadata__\":{\"format\":\"pt\"}}",
        T, off0, off1,
        T, off1, off2,
        T, concat_dim, off2, off3,
        T, MYDS_DSPINE_DIM, off3, off4);

    if (hlen <= 0 || (size_t)hlen >= sizeof(header_buf)) {
        fclose(f);
        remove(tmp_filepath);
        return false;
    }

    /* 8-byte alignment padding */
    size_t pad = (8u - ((size_t)hlen % 8u)) % 8u;
    for (size_t i = 0; i < pad; i++) {
        header_buf[hlen++] = ' ';
    }
    header_buf[hlen] = '\0';

    uint64_t n_header = (uint64_t)hlen;
    bool ok = (fwrite(&n_header, sizeof(uint64_t), 1, f) == 1) &&
              (fwrite(header_buf, 1, (size_t)hlen, f) == (size_t)hlen) &&
              (fwrite(s_tokens_buf, 1, ids_bytes, f) == ids_bytes) &&
              (fwrite(s_mask_buf, 1, mask_bytes, f) == mask_bytes) &&
              (fwrite(s_hidden_buf, 1, hidden_bytes, f) == hidden_bytes) &&
              (fwrite(s_last_norm_buf, 1, last_norm_bytes, f) == last_norm_bytes);

    if (fclose(f) != 0) ok = false;

    if (!ok || rename(tmp_filepath, filepath) != 0) {
        MYDS_LOG_E("DSpine capture: failed to write/rename shard %s", filepath);
        remove(tmp_filepath);
        ok = false;
    } else {
        MYDS_LOG_I("DSpine capture: successfully saved %u tokens to %s (%.2f MB)",
                   T, filepath, (double)(sizeof(uint64_t) + (size_t)hlen + off4) / (1024.0 * 1024.0));
    }

    /* Clean memory */
    free(s_tokens_buf); s_tokens_buf = NULL;
    free(s_mask_buf); s_mask_buf = NULL;
    free(s_hidden_buf); s_hidden_buf = NULL;
    free(s_last_norm_buf); s_last_norm_buf = NULL;
    s_curr_tokens = 0;

    return ok;
}
