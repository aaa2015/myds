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

    ASSERT_TRUE(myds_tokenizer_encode(tok, "?", tokens, 32, false) == MYDS_TOKENIZER_ERROR_UNKNOWN_BYTE);
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


/* A synthetic byte-BPE vocabulary with independently assigned byte IDs and
 * deliberately conflicting merge ranks. It does not depend on encode output. */
static int create_byte_bpe_gguf(const char *path, bool missing_byte, bool gpt2) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    write_u32(f, GGUF_MAGIC); write_u32(f, 3); write_u64(f, 0); write_u64(f, 7);
    write_str(f, "tokenizer.ggml.model"); write_u32(f, GGUF_TYPE_STRING); write_str(f, "gpt2");
    write_str(f, "tokenizer.ggml.pre"); write_u32(f, GGUF_TYPE_STRING); write_str(f, gpt2 ? "gpt-2" : "joyai-llm");
    write_str(f, "tokenizer.ggml.tokens"); write_u32(f, GGUF_TYPE_ARRAY); write_u32(f, GGUF_TYPE_STRING); write_u64(f, 273);
    unsigned extra = 256;
    for (unsigned b = 0; b < 256; ++b) {
        unsigned cp = ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) ? b : extra++;
        char bytes[3] = {0};
        if (cp < 128) bytes[0] = (char)cp;
        else { bytes[0] = (char)(0xc0 | (cp >> 6)); bytes[1] = (char)(0x80 | (cp & 63)); }
        write_str(f, missing_byte && b == 'Z' ? "notZ" : bytes);
    }
    const char *extra_tokens[] = {"<S>", "</S>", "<P>", "Ġb", "bc", "ab", "12", "123", "1234"};
    for (size_t i = 0; i < sizeof(extra_tokens)/sizeof(extra_tokens[0]); ++i) write_str(f, extra_tokens[i]);
    char long_token[257]; memset(long_token, 'a', sizeof(long_token));
    for (int n = 2; n <= 256; n *= 2) { long_token[n] = 0; write_str(f, long_token); long_token[n] = 'a'; }
    write_str(f, "tokenizer.ggml.merges"); write_u32(f, GGUF_TYPE_ARRAY); write_u32(f, GGUF_TYPE_STRING); write_u64(f, 14);
    const char *merges[] = {"Ġ b", "b c", "a b", "1 2", "12 3", "123 4"};
    for (size_t i = 0; i < sizeof(merges)/sizeof(merges[0]); ++i) write_str(f, merges[i]);
    for (int n = 1; n <= 128; n *= 2) {
        char merge[258]; memset(merge, 'a', (size_t)n * 2 + 1); merge[n] = ' '; merge[n * 2 + 1] = 0; write_str(f, merge);
    }
    write_str(f, "tokenizer.ggml.token_type"); write_u32(f, GGUF_TYPE_ARRAY); write_u32(f, GGUF_TYPE_INT32); write_u64(f, 273);
    for (unsigned i = 0; i < 273; ++i) write_u32(f, i >= 256 && i <= 257 ? 3 : 1);
    write_str(f, "tokenizer.ggml.bos_token_id"); write_u32(f, GGUF_TYPE_UINT32); write_u32(f, 256);
    write_str(f, "tokenizer.ggml.eos_token_id"); write_u32(f, GGUF_TYPE_UINT32); write_u32(f, 257);
    fclose(f); return 1;
}
static int test_byte_bpe_known_answers(void) {
    const char *path = "/tmp/test_myds_byte_bpe.gguf";
    ASSERT_TRUE(create_byte_bpe_gguf(path, false, false));
    gguf_file_t *gf = gguf_open(path); ASSERT_TRUE(gf != NULL);
    myds_tokenizer_t *tok = myds_tokenizer_load(gf); ASSERT_TRUE(tok != NULL);
    int32_t ids[512];
    ASSERT_TRUE(myds_tokenizer_encode(tok, "a b", ids, 512, false) == 2);
    ASSERT_TRUE(ids[0] == 'a' && ids[1] == 259); /* Ġb, never drop the space. */
    ASSERT_TRUE(myds_tokenizer_encode(tok, "abc", ids, 512, false) == 2);
    ASSERT_TRUE(ids[0] == 'a' && ids[1] == 260); /* b+c outranks a+b. */
    const int32_t chinese[] = {0xe4, 0xb8, 0xad, 0xe6, 0x96, 0x87};
    ASSERT_TRUE(myds_tokenizer_encode(tok, "中文", ids, 512, false) == 6);
    ASSERT_TRUE(!memcmp(ids, chinese, sizeof(chinese)));
    ASSERT_TRUE(myds_tokenizer_encode(tok, "1234", ids, 512, false) == 2);
    ASSERT_TRUE(ids[0] == 263 && ids[1] == '4'); /* JoyAI splits digits in threes. */
    ASSERT_TRUE(myds_tokenizer_encode(tok, "<S>a b", ids, 512, true) == 3);
    ASSERT_TRUE(ids[0] == 256 && ids[1] == 'a' && ids[2] == 259); /* BOS once. */
    ASSERT_TRUE(myds_tokenizer_encode(tok, "a</S>", ids, 512, false) == 2);
    ASSERT_TRUE(ids[0] == 'a' && ids[1] == 257);
    ASSERT_TRUE(myds_tokenizer_encode(tok, "<P>", ids, 512, false) == 1);
    ASSERT_TRUE(ids[0] == 258); /* Added token mislabeled NORMAL in GGUF. */
    char long_word[257]; memset(long_word, 'a', 256); long_word[256] = 0;
    ASSERT_TRUE(myds_tokenizer_encode(tok, long_word, ids, 512, false) == 1);
    ASSERT_TRUE(ids[0] == 272); /* No arbitrary 64-byte longest-match limit. */
    ASSERT_TRUE(myds_tokenizer_encode(tok, "a b", NULL, 0, false) == 2);
    ids[1] = -999;
    ASSERT_TRUE(myds_tokenizer_encode(tok, "a b", ids, 1, false) == 2);
    ASSERT_TRUE(ids[0] == 'a' && ids[1] == -999);
    myds_tokenizer_free(tok); gguf_close(gf); unlink(path);
    printf("  [PASS] Byte-BPE known IDs, rank order, pre-tokenization, and special tokens\n");
    return 1;
}
static int test_byte_bpe_binary_decode(void) {
    const char *path = "/tmp/test_myds_binary_bpe.gguf";
    ASSERT_TRUE(create_byte_bpe_gguf(path, false, false));
    gguf_file_t *gf = gguf_open(path); ASSERT_TRUE(gf != NULL);
    myds_tokenizer_t *tok = myds_tokenizer_load(gf); ASSERT_TRUE(tok != NULL);
    size_t len = 99;
    const char *zero = myds_tokenizer_decode_bytes(tok, 0, &len);
    ASSERT_TRUE(len == 1 && zero[0] == 0);
    const char *lead = myds_tokenizer_decode_bytes(tok, 0xe4, &len);
    ASSERT_TRUE(len == 1 && (unsigned char)lead[0] == 0xe4); /* Incomplete UTF-8. */
    const char *other = myds_tokenizer_decode_bytes(tok, 0xff, &len);
    ASSERT_TRUE(len == 1 && (unsigned char)other[0] == 0xff);
    ASSERT_TRUE((unsigned char)lead[0] == 0xe4 && zero[0] == 0); /* No mutable static buffer. */
    const char *space_b = myds_tokenizer_decode_bytes(tok, 259, &len);
    ASSERT_TRUE(len == 2 && !memcmp(space_b, " b", 2));
    char bytes[256]; for (int i = 0; i < 256; ++i) bytes[i] = (char)i;
    int32_t ids[512];
    int32_t n = myds_tokenizer_encode_bytes(tok, bytes, sizeof(bytes), ids, 512, false);
    ASSERT_TRUE(n > 0 && n <= 256);
    size_t offset = 0;
    for (int32_t i = 0; i < n; ++i) {
        const char *piece = myds_tokenizer_decode_bytes(tok, ids[i], &len);
        ASSERT_TRUE(offset + len <= sizeof(bytes) && !memcmp(piece, bytes + offset, len));
        offset += len;
    }
    ASSERT_TRUE(offset == sizeof(bytes));
    ASSERT_TRUE(myds_tokenizer_decode_bytes(tok, -1, &len)[0] == 0 && len == 0);
    myds_tokenizer_free(tok); gguf_close(gf); unlink(path);
    printf("  [PASS] Binary decoding preserves all 256 bytes, NUL, and split UTF-8\n");
    return 1;
}
static int test_byte_bpe_unknown_and_gpt2(void) {
    const char *path = "/tmp/test_myds_missing_bpe.gguf";
    ASSERT_TRUE(create_byte_bpe_gguf(path, true, true));
    gguf_file_t *gf = gguf_open(path); ASSERT_TRUE(gf != NULL);
    myds_tokenizer_t *tok = myds_tokenizer_load(gf); ASSERT_TRUE(tok != NULL);
    int32_t ids[16];
    ASSERT_TRUE(myds_tokenizer_encode(tok, "aZ", ids, 16, false) == MYDS_TOKENIZER_ERROR_UNKNOWN_BYTE);
    ASSERT_TRUE(myds_tokenizer_encode(tok, "aZ", NULL, 0, false) == MYDS_TOKENIZER_ERROR_UNKNOWN_BYTE);
    ASSERT_TRUE(myds_tokenizer_encode(tok, "1234", ids, 16, false) == 1);
    ASSERT_TRUE(ids[0] == 264); /* GPT-2 numbers are not split every three digits. */
    myds_tokenizer_free(tok); gguf_close(gf); unlink(path);
    printf("  [PASS] Missing-byte errors are explicit; GPT-2 metadata selects its split rules\n");
    return 1;
}

static int test_byte_bpe_piece_reuse(void) {
    const char *path = "/tmp/test_myds_piece_reuse.gguf";
    ASSERT_TRUE(create_byte_bpe_gguf(path, false, false));
    gguf_file_t *gf = gguf_open(path); ASSERT_TRUE(gf != NULL);
    myds_tokenizer_t *tok = myds_tokenizer_load(gf); ASSERT_TRUE(tok != NULL);
    /* Grow on powers of two and their boundaries, then revisit short pieces.
     * Special tokens divide encode_bpe calls within this same encode request. */
    static const size_t lengths[] = {1, 31, 32, 33, 255, 256, 257, 4097, 2, 65, 3, 1};
    char text[8192]; int32_t expected[1024], ids[1025];
    size_t bytes = 0; int32_t count = 0;
    for (size_t j = 0; j < sizeof(lengths)/sizeof(lengths[0]); ++j) {
        size_t n = lengths[j];
        memset(text + bytes, 'a', n); bytes += n;
        while (n >= 256) { expected[count++] = 272; n -= 256; }
        for (unsigned power = 7; power > 0; --power) {
            size_t size = (size_t)1 << power;
            if (n >= size) { expected[count++] = 264 + (int32_t)power; n -= size; }
        }
        if (n) expected[count++] = 'a';
        memcpy(text + bytes, "\nabc<S>1234</S>", 15); bytes += 15;
        expected[count++] = '\n'; expected[count++] = 'a'; expected[count++] = 260;
        expected[count++] = 256; expected[count++] = 263; expected[count++] = '4';
        expected[count++] = 257;
    }
    ASSERT_TRUE(myds_tokenizer_encode_bytes(tok, text, bytes, ids, 1024, false) == count);
    ASSERT_TRUE(!memcmp(ids, expected, (size_t)count * sizeof(*ids)));
    ASSERT_TRUE(myds_tokenizer_encode_bytes(tok, text, bytes, NULL, 0, false) == count);
    for (int32_t capacity = 0; capacity <= count; ++capacity) {
        ids[capacity] = -999;
        ASSERT_TRUE(myds_tokenizer_encode_bytes(tok, text, bytes, ids, capacity, false) == count);
        ASSERT_TRUE(!memcmp(ids, expected, (size_t)capacity * sizeof(*ids)));
        ASSERT_TRUE(ids[capacity] == -999);
    }
    myds_tokenizer_free(tok); gguf_close(gf); unlink(path);
    printf("  [PASS] Long/short BPE pieces and specials preserve IDs, counts, and capacity guards\n");
    return 1;
}

/* Metadata-only reference probe: no model tensors or GPU are evaluated. */
static int dump_reference(const char *path, const char *text) {
    gguf_file_t *gf = gguf_open(path);
    if (!gf) return 1;
    myds_tokenizer_t *tok = myds_tokenizer_load(gf);
    if (!tok) { gguf_close(gf); return 1; }
    int32_t count = myds_tokenizer_encode(tok, text, NULL, 0, false);
    int32_t *ids = count >= 0 ? malloc(((size_t)count + 1) * sizeof(*ids)) : NULL;
    if (!ids) { myds_tokenizer_free(tok); gguf_close(gf); return 1; }
    int32_t result = myds_tokenizer_encode(tok, text, ids, count, false);
    printf("{\"ids\":[");
    for (int32_t i = 0; i < result; ++i) printf("%s%d", i ? "," : "", ids[i]);
    printf("],\"decoded_hex\":\"");
    for (int32_t i = 0; i < result; ++i) {
        size_t len; const char *piece = myds_tokenizer_decode_bytes(tok, ids[i], &len);
        for (size_t j = 0; j < len; ++j) printf("%02x", (unsigned char)piece[j]);
    }
    printf("\"}\n");
    free(ids); myds_tokenizer_free(tok); gguf_close(gf);
    return result < 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--dump")) return dump_reference(argv[2], argv[3]);

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
    passed += test_byte_bpe_known_answers();
    passed += test_byte_bpe_binary_decode();
    passed += test_byte_bpe_unknown_and_gpt2();
    passed += test_byte_bpe_piece_reuse();

    unlink(temp_file);

    printf("====================================================\n");
    if (passed == 8) {
        printf("  ALL TOKENIZER UNIT TESTS PASSED!\n");
        printf("====================================================\n");
        return 0;
    } else {
        printf("  TOKENIZER TESTS FAILED (passed %d/8)\n", passed);
        return 1;
    }
}
