#include "myds_tokenizer.h"
#include "myds_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        return 0; \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    if (strcmp((a), (b)) != 0) { \
        fprintf(stderr, "[FAIL] %s:%d: '%s' != '%s'\n", __FILE__, __LINE__, (a), (b)); \
        return 0; \
    } \
} while(0)

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }
static void write_f32(FILE *f, float v) { fwrite(&v, sizeof(v), 1, f); }
static void write_str(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    write_u64(f, len);
    fwrite(s, 1, len, f);
}

/* Create a GGUF file with test vocabulary */
static const char *create_tokenizer_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;

    uint32_t magic = GGUF_MAGIC;
    uint32_t version = 3;
    uint64_t n_tensors = 0;
    uint64_t n_kvs = 4;

    fwrite(&magic, 4, 1, f);
    write_u32(f, version);
    write_u64(f, n_tensors);
    write_u64(f, n_kvs);

    // Tokens list
    const char *vocab_tokens[] = {
        "<unk>", "<s>", "</s>", "hello", "world", "Deep", "Seek", " ", "<0x41>", "<0xFF>"
    };
    uint64_t n_vocab = sizeof(vocab_tokens) / sizeof(vocab_tokens[0]);

    // kv 0: tokenizer.ggml.tokens (array of strings)
    write_str(f, "tokenizer.ggml.tokens");
    write_u32(f, GGUF_TYPE_ARRAY);
    write_u32(f, GGUF_TYPE_STRING);
    write_u64(f, n_vocab);
    for (uint64_t i = 0; i < n_vocab; i++) {
        write_str(f, vocab_tokens[i]);
    }

    // kv 1: tokenizer.ggml.scores (array of f32)
    write_str(f, "tokenizer.ggml.scores");
    write_u32(f, GGUF_TYPE_ARRAY);
    write_u32(f, GGUF_TYPE_FLOAT32);
    write_u64(f, n_vocab);
    for (uint64_t i = 0; i < n_vocab; i++) {
        write_f32(f, (float)(100 - i));
    }

    // kv 2: tokenizer.ggml.bos_token_id (u32) = 1
    write_str(f, "tokenizer.ggml.bos_token_id");
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 1);

    // kv 3: tokenizer.ggml.eos_token_id (u32) = 2
    write_str(f, "tokenizer.ggml.eos_token_id");
    write_u32(f, GGUF_TYPE_UINT32);
    write_u32(f, 2);

    fclose(f);
    return path;
}

static int test_tokenizer_load_and_vocab(const char *path) {
    gguf_file_t *gf = gguf_open(path);
    ASSERT_TRUE(gf != NULL);

    myds_tokenizer_t *tok = myds_tokenizer_load(gf);
    ASSERT_TRUE(tok != NULL);
    ASSERT_TRUE(tok->vocab_size == 10);
    ASSERT_TRUE(tok->bos_id == 1);
    ASSERT_TRUE(tok->eos_id == 2);

    ASSERT_STR_EQ(tok->vocab[3].text, "hello");
    ASSERT_STR_EQ(tok->vocab[5].text, "Deep");
    ASSERT_STR_EQ(tok->vocab[6].text, "Seek");

    myds_tokenizer_free(tok);
    gguf_close(gf);
    printf("  [PASS] Tokenizer vocabulary loading and metadata\n");
    return 1;
}

static int test_tokenizer_encode(const char *path) {
    gguf_file_t *gf = gguf_open(path);
    ASSERT_TRUE(gf != NULL);

    myds_tokenizer_t *tok = myds_tokenizer_load(gf);
    ASSERT_TRUE(tok != NULL);

    int32_t tokens[32];

    /* 1. Encode with BOS */
    int32_t n = myds_tokenizer_encode(tok, "DeepSeek", tokens, 32, true);
    ASSERT_TRUE(n == 3); /* [<s>, "Deep", "Seek"] */
    ASSERT_TRUE(tokens[0] == 1); /* <s> */
    ASSERT_TRUE(tokens[1] == 5); /* Deep */
    ASSERT_TRUE(tokens[2] == 6); /* Seek */

    /* 2. Encode without BOS */
    n = myds_tokenizer_encode(tok, "hello world", tokens, 32, false);
    ASSERT_TRUE(n == 3); /* ["hello", " ", "world"] */
    ASSERT_TRUE(tokens[0] == 3); /* hello */
    ASSERT_TRUE(tokens[1] == 7); /* space */
    ASSERT_TRUE(tokens[2] == 4); /* world */

    /* 3. Byte fallback: character 'A' (0x41) has token <0x41> (id 8) */
    n = myds_tokenizer_encode(tok, "A", tokens, 32, false);
    ASSERT_TRUE(n == 1);
    ASSERT_TRUE(tokens[0] == 8); /* <0x41> */

    myds_tokenizer_free(tok);
    gguf_close(gf);
    printf("  [PASS] Tokenizer BPE encoding and byte-fallback\n");
    return 1;
}

static int test_tokenizer_decode(const char *path) {
    gguf_file_t *gf = gguf_open(path);
    ASSERT_TRUE(gf != NULL);

    myds_tokenizer_t *tok = myds_tokenizer_load(gf);
    ASSERT_TRUE(tok != NULL);

    /* 1. Decode standard piece */
    ASSERT_STR_EQ(myds_tokenizer_decode_piece(tok, 3), "hello");
    ASSERT_STR_EQ(myds_tokenizer_decode_piece(tok, 4), "world");
    ASSERT_STR_EQ(myds_tokenizer_decode_piece(tok, 5), "Deep");

    /* 2. Decode byte token <0x41> -> "A" */
    const char *byte_piece = myds_tokenizer_decode_piece(tok, 8);
    ASSERT_TRUE(byte_piece != NULL);
    ASSERT_TRUE(byte_piece[0] == 'A');

    /* 3. Out-of-bounds token returns "" */
    ASSERT_STR_EQ(myds_tokenizer_decode_piece(tok, -1), "");
    ASSERT_STR_EQ(myds_tokenizer_decode_piece(tok, 9999), "");

    myds_tokenizer_free(tok);
    gguf_close(gf);
    printf("  [PASS] Tokenizer decoding & byte token translation\n");
    return 1;
}

static int test_tokenizer_fallback_mode(void) {
    /* Test fallback when GGUF has no tokens */
    const char *empty_path = "/tmp/test_empty_tok.gguf";
    FILE *f = fopen(empty_path, "wb");
    uint32_t magic = GGUF_MAGIC;
    uint32_t version = 3;
    uint64_t n_tensors = 0;
    uint64_t n_kvs = 0;
    fwrite(&magic, 4, 1, f);
    write_u32(f, version);
    write_u64(f, n_tensors);
    write_u64(f, n_kvs);
    fclose(f);

    gguf_file_t *gf = gguf_open(empty_path);
    ASSERT_TRUE(gf != NULL);

    myds_tokenizer_t *tok = myds_tokenizer_load(gf);
    ASSERT_TRUE(tok != NULL);
    ASSERT_TRUE(tok->vocab_size == 256);

    int32_t tokens[16];
    int32_t n = myds_tokenizer_encode(tok, "XYZ", tokens, 16, false);
    ASSERT_TRUE(n == 3);
    ASSERT_TRUE(tokens[0] == 'X');
    ASSERT_TRUE(tokens[1] == 'Y');
    ASSERT_TRUE(tokens[2] == 'Z');

    ASSERT_STR_EQ(myds_tokenizer_decode_piece(tok, 'X'), "X");

    myds_tokenizer_free(tok);
    gguf_close(gf);
    unlink(empty_path);

    printf("  [PASS] Tokenizer 256-byte fallback mode\n");
    return 1;
}

int main(void) {
    printf("====================================================\n");
    printf("      Running myds BPE Tokenizer Unit Tests         \n");
    printf("====================================================\n");

    const char *temp_file = "/tmp/test_myds_tok.gguf";
    if (!create_tokenizer_gguf(temp_file)) {
        fprintf(stderr, "Failed to create synthetic test tokenizer GGUF\n");
        return 1;
    }

    int passed = 0;
    passed += test_tokenizer_load_and_vocab(temp_file);
    passed += test_tokenizer_encode(temp_file);
    passed += test_tokenizer_decode(temp_file);
    passed += test_tokenizer_fallback_mode();

    unlink(temp_file);

    printf("====================================================\n");
    if (passed == 4) {
        printf("  ALL TOKENIZER UNIT TESTS PASSED!\n");
        printf("====================================================\n");
        return 0;
    } else {
        printf("  TOKENIZER TESTS FAILED (passed %d/4)\n", passed);
        return 1;
    }
}
