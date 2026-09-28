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
    char    *decoded; /* Owned, NUL-terminated storage; use decoded_len for bytes. */
    uint32_t decoded_len;
    bool     special;
} myds_token_entry_t;

typedef struct {
    uint32_t           vocab_size;
    myds_token_entry_t *vocab;
    int32_t            bos_id;
    int32_t            eos_id;
    int32_t            pad_id;
    void              *impl; /* Immutable lookup tables, owned by tokenizer. */
} myds_tokenizer_t;

myds_tokenizer_t *myds_tokenizer_load(const gguf_file_t *gf);
void              myds_tokenizer_free(myds_tokenizer_t *tok);

enum {
    MYDS_TOKENIZER_ERROR_INVALID = -1,
    MYDS_TOKENIZER_ERROR_UNKNOWN_BYTE = -2,
    MYDS_TOKENIZER_ERROR_NOMEM = -3
};

/* Tokenize UTF-8 input string to an array of token IDs.
 * Returns the number of tokens written into out_tokens.
 * If out_tokens is NULL, returns the total required token count.
 * A negative result is an error; discard any partial output. Never drops bytes.
 * If the return value exceeds max_tokens, the caller must retry with more space.
 * Recognizes the vocabulary's special tokens in rendered chat prompts. */
int32_t           myds_tokenizer_encode(const myds_tokenizer_t *tok,
                                        const char             *text,
                                        int32_t                *out_tokens,
                                        int32_t                 max_tokens,
                                        bool                    add_bos);

/* Explicit-length input variant, including embedded NUL bytes. */
int32_t myds_tokenizer_encode_bytes(const myds_tokenizer_t *tok,
                                    const char *text, size_t len,
                                    int32_t *out_tokens, int32_t max_tokens,
                                    bool add_bos);

/* Stable, tokenizer-owned bytes. Pieces may contain NUL or partial UTF-8; join
 * them before interpreting UTF-8. Invalid IDs return "" and a zero length. */
const char *myds_tokenizer_decode_bytes(const myds_tokenizer_t *tok,
                                        int32_t token_id, size_t *len);

/* Compatibility string API. Use decode_bytes for lossless binary output. */
const char       *myds_tokenizer_decode_piece(const myds_tokenizer_t *tok, int32_t token_id);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_TOKENIZER_H */
