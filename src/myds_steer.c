/* myds_steer.c — GLP Activation Steering & Control Vector Subsystem */

#include "myds_steer.h"
#include "myds_gguf.h"
#include "myds_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static void parse_layer_filter(const char *filter, bool *mask, uint32_t max_layers) {
    for (uint32_t i = 0; i < max_layers; i++) mask[i] = true;
    if (!filter || !*filter || strcmp(filter, "all") == 0 || strcmp(filter, "auto") == 0) return;

    for (uint32_t i = 0; i < max_layers; i++) mask[i] = false;

    char buf[256];
    strncpy(buf, filter, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *saveptr = NULL;
    char *token = strtok_r(buf, ",", &saveptr);
    while (token) {
        while (*token && isspace((unsigned char)*token)) token++;
        char *dash = strchr(token, '-');
        if (dash) {
            *dash = '\0';
            int start = atoi(token);
            int end = atoi(dash + 1);
            if (start < 0) start = 0;
            if (end >= (int)max_layers) end = (int)max_layers - 1;
            for (int i = start; i <= end; i++) {
                if (i >= 0 && i < (int)max_layers) mask[i] = true;
            }
        } else {
            int l = atoi(token);
            if (l >= 0 && l < (int)max_layers) mask[l] = true;
        }
        token = strtok_r(NULL, ",", &saveptr);
    }
}

myds_steer_t *myds_steer_load(const char *path, float alpha_override,
                              myds_steer_mode_t mode, const char *layers_filter)
{
    if (!path || !*path) return NULL;

    gguf_file_t *gf = gguf_open(path);
    if (!gf) {
        MYDS_LOG_E("GLP steer: failed to open control vector file %s", path);
        return NULL;
    }

    myds_steer_t *st = (myds_steer_t *)calloc(1, sizeof(myds_steer_t));
    if (!st) {
        gguf_close(gf);
        return NULL;
    }

    strncpy(st->path, path, sizeof(st->path) - 1);

    /* 1. Read Metadata */
    st->spec_version = gguf_get_u32(gf, "glp.spec_version", 1);
    st->alpha_default = gguf_get_f32(gf, "glp.alpha_default", 1.0f);
    const char *hp = gguf_get_str(gf, "glp.hook_point", "residual_stream_post_layer");
    strncpy(st->hook_point, hp, sizeof(st->hook_point) - 1);

    /* 2. Determine Mode */
    if (mode == MYDS_STEER_MODE_AUTO) {
        if (st->spec_version >= 3 || strcmp(st->hook_point, "hc_fold_pre_engram") == 0) {
            st->mode = MYDS_STEER_MODE_V3_DUAL;
        } else {
            st->mode = MYDS_STEER_MODE_V2_RESIDUAL_ONLY;
        }
    } else {
        st->mode = mode;
    }

    st->alpha_override = alpha_override;
    st->alpha_effective = (alpha_override >= 0.0f) ? alpha_override : st->alpha_default;
    st->enabled = true;
    st->dim = 5120;

    /* 3. Layer Filter */
    bool allowed_layers[MYDS_STEER_MAX_LAYERS];
    parse_layer_filter(layers_filter, allowed_layers, MYDS_STEER_MAX_LAYERS);

    /* 4. Load Direction Tensors */
    uint32_t min_active = MYDS_STEER_MAX_LAYERS, max_active = 0;
    st->active_layer_count = 0;

    for (uint32_t l = 0; l < MYDS_STEER_MAX_LAYERS; l++) {
        st->layer_active[l] = false;
        st->directions[l] = NULL;
        if (!allowed_layers[l]) continue;

        char tname[64];
        snprintf(tname, sizeof(tname), "direction.%u", l);
        const gguf_tensor_t *t = gguf_find_tensor(gf, tname);
        if (!t) continue;

        if (t->n_dims != 1 || t->dims[0] != st->dim || t->type != GGML_TYPE_F32) {
            MYDS_LOG_W("GLP steer: tensor %s has unexpected shape [%llu] type %u (expected [%u] F32), skipping",
                       tname, (unsigned long long)t->dims[0], (unsigned)t->type, st->dim);
            continue;
        }

        const size_t bytes = st->dim * sizeof(float);
        myds_metal_buffer_t *mb = myds_metal_buffer_alloc(bytes);
        if (!mb) {
            MYDS_LOG_E("GLP steer: failed to allocate Metal buffer for layer %u", l);
            continue;
        }

        /* Copy data into unified memory buffer */
        if (!myds_metal_buffer_write(mb, 0, t->data, bytes)) {
            MYDS_LOG_E("GLP steer: failed to write tensor %s to Metal buffer", tname);
            myds_metal_buffer_free(mb);
            continue;
        }

        st->directions[l] = mb;
        st->layer_active[l] = true;
        st->active_layer_count++;
        if (l < min_active) min_active = l;
        if (l > max_active) max_active = l;
    }

    gguf_close(gf);

    if (st->active_layer_count == 0) {
        MYDS_LOG_W("GLP steer: no active direction vectors loaded from %s", path);
        myds_steer_free(st);
        return NULL;
    }

    const char *mode_str = (st->mode == MYDS_STEER_MODE_V3_DUAL) ? "Spec-v3 (Dual Hook)" : "Spec-v2 (Residual Only)";
    MYDS_LOG_I("GLP Steering loaded: %s | Mode: %s | Spec: v%u | Alpha: %.3f%s | Active layers: %u (L%u..L%u)",
               st->path, mode_str, st->spec_version, st->alpha_effective,
               (st->alpha_override >= 0.0f) ? " [override]" : " [file default]",
               st->active_layer_count, min_active, max_active);

    return st;
}

void myds_steer_free(myds_steer_t *st) {
    if (!st) return;
    for (uint32_t l = 0; l < MYDS_STEER_MAX_LAYERS; l++) {
        if (st->directions[l]) {
            myds_metal_buffer_free(st->directions[l]);
            st->directions[l] = NULL;
        }
        st->layer_active[l] = false;
    }
    free(st);
}

void myds_steer_set_alpha(myds_steer_t *st, float alpha) {
    if (!st) return;
    if (alpha >= 0.0f) {
        st->alpha_effective = alpha;
        st->alpha_override = alpha;
    } else {
        st->alpha_effective = st->alpha_default;
        st->alpha_override = -1.0f;
    }
}

void myds_steer_set_enabled(myds_steer_t *st, bool enabled) {
    if (!st) return;
    st->enabled = enabled;
}
