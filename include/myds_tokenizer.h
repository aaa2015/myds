#ifndef MYDS_TOKENIZER_H
#define MYDS_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "myds_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char    *text;
    float    score;
    uint32_t len;
} myds_token_entry_t;

typedef struct {
    uint32_t           vocab_size;
    myds_token_entry_t *vocab;
    int32_t            bos_id;
    int32_t            eos_id;
    int32_t            pad_id;
} myds_tokenizer_t;

myds_tokenizer_t *myds_tokenizer_load(const gguf_file_t *gf);
void              myds_tokenizer_free(myds_tokenizer_t *tok);

/* Tokenize UTF-8 input string to an array of token IDs.
 * Returns the number of tokens written into out_tokens.
 * If out_tokens is NULL, returns the total required token count. */
int32_t           myds_tokenizer_encode(const myds_tokenizer_t *tok,
                                        const char             *text,
                                        int32_t                *out_tokens,
                                        int32_t                 max_tokens,
                                        bool                    add_bos);

/* Decode a single token ID into a string. Returns pointer to internal buffer or literal. */
const char       *myds_tokenizer_decode_piece(const myds_tokenizer_t *tok, int32_t token_id);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_TOKENIZER_H */
