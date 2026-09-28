#ifndef MYDS_H
#define MYDS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct myds_model myds_model_t;
typedef struct myds_context myds_context_t;

typedef struct {
    uint32_t max_seq_len;      /* Maximum context window (default: 4096) */
    uint32_t max_new_tokens;    /* Maximum generated tokens (default: 256) */
    float    temperature;       /* Sampling temperature (0.0 = greedy argmax) */
    float    top_p;             /* Nucleus top-p sampling (default: 0.9) */
    int32_t  seed;              /* Random seed */
    bool     stream;            /* Stream output tokens to callback */
} myds_config_t;

typedef void (*myds_stream_callback_t)(const char *piece, int32_t token_id, void *user_data);

/* Model lifecycle */
myds_model_t   *myds_model_load(const char *model_path);
void            myds_model_free(myds_model_t *model);
void            myds_model_print_info(const myds_model_t *model);

/* Inference context lifecycle */
myds_context_t *myds_context_create(myds_model_t *model, const myds_config_t *cfg);
void            myds_context_free(myds_context_t *ctx);
myds_config_t   myds_context_get_config(const myds_context_t *ctx);
void            myds_context_set_config(myds_context_t *ctx, const myds_config_t *cfg);

/* Generation */
int             myds_generate(myds_context_t *ctx,
                              const char     *prompt,
                              myds_stream_callback_t cb,
                              void           *user_data);

/* Single token forward step */
int32_t         myds_forward_token(myds_context_t *ctx, int32_t token_id, uint32_t pos);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_H */
