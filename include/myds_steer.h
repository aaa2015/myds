#ifndef MYDS_STEER_H
#define MYDS_STEER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "myds_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MYDS_STEER_MAX_LAYERS 64

typedef enum {
    MYDS_STEER_MODE_AUTO = 0,             /* Auto-detect from GGUF metadata (spec v3 -> DUAL, spec v1/v2 -> RESIDUAL_ONLY) */
    MYDS_STEER_MODE_V2_RESIDUAL_ONLY = 1, /* Hook only at post-layer residual fold (e.g. GLP-39) */
    MYDS_STEER_MODE_V3_DUAL = 2           /* Hook at both attention o_proj output and residual fold (e.g. GLP-27) */
} myds_steer_mode_t;

typedef struct myds_steer {
    bool enabled;
    char path[512];
    char name[64];           /* Identifier name, e.g. "glp27", "glp39" */
    float alpha_default;     /* Alpha from GGUF file */
    float alpha_override;    /* CLI/env override (-1 if not set) */
    float alpha_effective;   /* Currently effective alpha */
    myds_steer_mode_t mode;  /* Effective steer mode */
    uint32_t spec_version;   /* Read from glp.spec_version */
    char hook_point[64];     /* Read from glp.hook_point */
    uint32_t dim;            /* Vector dimension (5120) */

    /* Layer active mask */
    bool layer_active[MYDS_STEER_MAX_LAYERS];
    uint32_t active_layer_count;

    /* Metal buffers holding direction vectors for each layer */
    myds_metal_buffer_t *directions[MYDS_STEER_MAX_LAYERS];
} myds_steer_t;

/* Load control vector GGUF file.
 * alpha_override: < 0 to use alpha_default from file.
 * mode: MYDS_STEER_MODE_AUTO to detect from metadata.
 * layers_filter: e.g. "11-37" or NULL/empty for all layers in file.
 * Returns allocated myds_steer_t on success, NULL on failure.
 */
myds_steer_t *myds_steer_load(const char *path, float alpha_override,
                              myds_steer_mode_t mode, const char *layers_filter);

/* Parse layer filter string (e.g. "11-37", "1,5,10-15", "all") into mask. */
void myds_steer_parse_layer_filter(const char *filter, bool *mask, uint32_t max_layers);

/* Free steer instance and its metal buffers. */
void myds_steer_free(myds_steer_t *st);

/* Set runtime override for alpha (e.g. per-request). < 0 resets to default. */
void myds_steer_set_alpha(myds_steer_t *st, float alpha);

/* Enable / disable steering at runtime. */
void myds_steer_set_enabled(myds_steer_t *st, bool enabled);

/* Check if a specific layer is active. */
static inline bool myds_steer_is_layer_active(const myds_steer_t *st, uint32_t layer) {
    if (!st || !st->enabled || layer >= MYDS_STEER_MAX_LAYERS) return false;
    return st->layer_active[layer] && st->directions[layer] != NULL;
}

#ifdef __cplusplus
}
#endif

#endif /* MYDS_STEER_H */
