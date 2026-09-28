#include "myds_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASH_TABLE_SIZE 131072

typedef struct hash_node {
    int32_t id;
    struct hash_node *next;
} hash_node_t;

typedef struct {
    hash_node_t *buckets[HASH_TABLE_SIZE];
} vocab_hash_t;

static uint32_t hash_str(const char *str, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)str[i];
        h *= 16777619u;
    }
    return h % HASH_TABLE_SIZE;
}

static void hash_insert(vocab_hash_t *ht, const char *str, size_t len, int32_t id) {
    uint32_t b = hash_str(str, len);
    hash_node_t *node = (hash_node_t *)malloc(sizeof(hash_node_t));
    node->id = id;
    node->next = ht->buckets[b];
    ht->buckets[b] = node;
}

static int32_t hash_find(const vocab_hash_t *ht, const myds_token_entry_t *vocab, const char *str, size_t len) {
    uint32_t b = hash_str(str, len);
    hash_node_t *curr = ht->buckets[b];
    while (curr) {
        if (vocab[curr->id].len == len && memcmp(vocab[curr->id].text, str, len) == 0) {
            return curr->id;
        }
        curr = curr->next;
    }
    return -1;
}

static void hash_free(vocab_hash_t *ht) {
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        hash_node_t *curr = ht->buckets[i];
        while (curr) {
            hash_node_t *tmp = curr->next;
            free(curr);
            curr = tmp;
        }
    }
    free(ht);
}

myds_tokenizer_t *myds_tokenizer_load(const gguf_file_t *gf) {
    if (!gf) return NULL;
    const gguf_kv_t *tok_kv = gguf_find_key(gf, "tokenizer.ggml.tokens");
    if (!tok_kv || tok_kv->type != GGUF_TYPE_ARRAY || tok_kv->value.val_arr.elem_type != GGUF_TYPE_STRING) {
        /* Fallback: create a byte-level 256-token vocabulary */
        uint32_t vocab_size = 256;
        myds_tokenizer_t *tok = (myds_tokenizer_t *)calloc(1, sizeof(myds_tokenizer_t));
        tok->vocab_size = vocab_size;
        tok->vocab = (myds_token_entry_t *)calloc(vocab_size, sizeof(myds_token_entry_t));
        for (uint32_t i = 0; i < vocab_size; i++) {
            char b[2] = { (char)i, '\0' };
            tok->vocab[i].text = strdup(b);
            tok->vocab[i].len = 1;
            tok->vocab[i].score = -(float)i;
        }
        tok->bos_id = 1;
        tok->eos_id = 0;
        tok->pad_id = 0;
        return tok;
    }

    uint32_t vocab_size = (uint32_t)tok_kv->value.val_arr.count;
    char **tokens = (char **)tok_kv->value.val_arr.data;

    const gguf_kv_t *score_kv = gguf_find_key(gf, "tokenizer.ggml.scores");
    float *scores = NULL;
    if (score_kv && score_kv->type == GGUF_TYPE_ARRAY && score_kv->value.val_arr.elem_type == GGUF_TYPE_FLOAT32) {
        scores = (float *)score_kv->value.val_arr.data;
    }

    myds_tokenizer_t *tok = (myds_tokenizer_t *)calloc(1, sizeof(myds_tokenizer_t));
    tok->vocab_size = vocab_size;
    tok->vocab = (myds_token_entry_t *)calloc(vocab_size, sizeof(myds_token_entry_t));

    for (uint32_t i = 0; i < vocab_size; i++) {
        tok->vocab[i].text = strdup(tokens[i] ? tokens[i] : "");
        tok->vocab[i].len = (uint32_t)strlen(tok->vocab[i].text);
        tok->vocab[i].score = scores ? scores[i] : -(float)i;
    }

    tok->bos_id = (int32_t)gguf_get_u32(gf, "tokenizer.ggml.bos_token_id", 1);
    tok->eos_id = (int32_t)gguf_get_u32(gf, "tokenizer.ggml.eos_token_id", 2);
    tok->pad_id = (int32_t)gguf_get_u32(gf, "tokenizer.ggml.padding_token_id", 0);

    return tok;
}

void myds_tokenizer_free(myds_tokenizer_t *tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i].text);
        }
        free(tok->vocab);
    }
    free(tok);
}

/* Fast BPE encode with greedy longest-match & byte fallback */
int32_t myds_tokenizer_encode(const myds_tokenizer_t *tok,
                             const char             *text,
                             int32_t                *out_tokens,
                             int32_t                 max_tokens,
                             bool                    add_bos) {
    if (!tok || !text) return 0;

    vocab_hash_t *ht = (vocab_hash_t *)calloc(1, sizeof(vocab_hash_t));
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        hash_insert(ht, tok->vocab[i].text, tok->vocab[i].len, (int32_t)i);
    }

    int32_t count = 0;
    if (add_bos && tok->bos_id >= 0) {
        if (out_tokens && count < max_tokens) out_tokens[count] = tok->bos_id;
        count++;
    }

    size_t len = strlen(text);
    size_t i = 0;

    while (i < len) {
        int32_t best_id = -1;
        size_t best_len = 0;

        /* Look for longest matching token up to 32 bytes */
        size_t max_match = (len - i > 64) ? 64 : (len - i);
        for (size_t l = max_match; l >= 1; l--) {
            int32_t id = hash_find(ht, tok->vocab, text + i, l);
            if (id >= 0) {
                best_id = id;
                best_len = l;
                break;
            }
        }

        if (best_id >= 0) {
            if (out_tokens && count < max_tokens) out_tokens[count] = best_id;
            count++;
            i += best_len;
        } else {
            /* Byte fallback token: <0xHH> */
            char hex_tok[8];
            snprintf(hex_tok, sizeof(hex_tok), "<0x%02X>", (uint8_t)text[i]);
            int32_t id = hash_find(ht, tok->vocab, hex_tok, strlen(hex_tok));
            if (id >= 0) {
                if (out_tokens && count < max_tokens) out_tokens[count] = id;
                count++;
            }
            i++;
        }
    }

    hash_free(ht);
    return count;
}

const char *myds_tokenizer_decode_piece(const myds_tokenizer_t *tok, int32_t token_id) {
    if (!tok || token_id < 0 || (uint32_t)token_id >= tok->vocab_size) return "";

    const char *s = tok->vocab[token_id].text;
    if (!s) return "";

    /* Parse byte token format: <0xHH> */
    if (strlen(s) == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
        static char byte_str[2] = {0, 0};
        char hex[3] = { s[3], s[4], '\0' };
        byte_str[0] = (char)strtol(hex, NULL, 16);
        return byte_str;
    }

    return s;
}
