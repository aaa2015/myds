/* CPU-only lifecycle tests. Include the implementation to inject allocation
 * failures and compare the ASCII fast path with the unchanged Unicode table.
 * clang -O2 -Wall -Wextra -std=c11 -Iinclude tests/test_tokenizer_workspace.c \
 *   src/myds_gguf.c -lpthread -o /tmp/test_tokenizer_workspace
 */
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool track_allocations;
static size_t allocation_calls, fail_at, live_allocations;
static bool allocation_fails(void) {
    return track_allocations && ++allocation_calls == fail_at;
}
static void *test_malloc(size_t n) {
    if (allocation_fails()) return NULL;
    void *p = malloc(n);
    if (track_allocations && p) ++live_allocations;
    return p;
}
static void *test_calloc(size_t n, size_t size) {
    if (allocation_fails()) return NULL;
    void *p = calloc(n, size);
    if (track_allocations && p) ++live_allocations;
    return p;
}
static void *test_realloc(void *p, size_t n) {
    if (allocation_fails()) return NULL;
    const bool is_new = p == NULL;
    void *result = realloc(p, n);
    if (track_allocations && result && is_new) ++live_allocations;
    return result;
}
static void test_free(void *p) {
    if (track_allocations && p) { assert(live_allocations); --live_allocations; }
    free(p);
}
#define malloc test_malloc
#define calloc test_calloc
#define realloc test_realloc
#define free test_free
#include "../src/myds_tokenizer.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

static void check_unicode(void) {
    /* The original range-table search is the independent oracle. */
    for (uint32_t cp = 0; cp <= 0x110000; ++cp)
        assert(tokenizer_unicode_class(cp) == myds_unicode_class(cp));
    assert(tokenizer_unicode_class(UINT32_MAX) == myds_unicode_class(UINT32_MAX));
    for (unsigned byte = 0; byte < 128; ++byte) {
        char s = (char)byte;
        character_t c = char_at(&s, 1, 0);
        assert(c.cp == byte && c.next == 1 && c.cls == myds_unicode_class(byte));
        c = char_at(&s, 1, 1);
        assert(c.cp == UINT32_MAX && c.next == 1 && c.cls == 0);
    }
}

typedef struct {
    const myds_tokenizer_t *tok;
    const char *text;
    size_t bytes;
    const int32_t *expected;
    int32_t count;
} encode_job_t;
static void *encode_thread(void *arg) {
    const encode_job_t *job = arg;
    int32_t ids[8192];
    for (unsigned i = 0; i < 64; ++i) {
        assert(myds_tokenizer_encode_bytes(job->tok, job->text, job->bytes,
                                          ids, 8192, false) == job->count);
        assert(!memcmp(ids, job->expected, (size_t)job->count * sizeof(*ids)));
    }
    return NULL;
}

int main(void) {
    check_unicode();
    /* Independently specified byte IDs and merges: a+a -> 256, aa+aa -> 257.
     * A special token forces multiple encode_bpe calls to share one workspace. */
    tokenizer_impl_t impl = {0};
    myds_token_entry_t vocab[259] = {0};
    myds_tokenizer_t tok = {.impl = &impl, .vocab = vocab, .vocab_size = 259, .bos_id = -1};
    merge_t merges[16]; memset(merges, 0xff, sizeof(merges));
    impl.byte_bpe = true; impl.joyai = true; impl.merges = merges; impl.merge_cap = 16;
    for (int i = 0; i < 256; ++i) impl.byte_ids[i] = i;
    merge_t pairs[] = {{'a', 'a', 256, 0}, {256, 256, 257, 1}};
    for (size_t i = 0; i < sizeof(pairs)/sizeof(pairs[0]); ++i) {
        size_t slot = hash_pair(pairs[i].left, pairs[i].right) & 15;
        while (merges[slot].rank >= 0) slot = (slot + 1) & 15;
        merges[slot] = pairs[i];
    }
    int32_t special = 258;
    vocab[258] = (myds_token_entry_t){.text = "<S>", .len = 3, .special = true};
    impl.special_ids = &special; impl.n_special = 1; impl.special_prefix['<'] = true;
    char text[8192]; int32_t expected[8192], ids[8192];
    size_t bytes = 0; int32_t count = 0;
    const size_t lengths[] = {1, 65, 4, 513, 2, 4097, 3, 32};
    for (size_t j = 0; j < sizeof(lengths)/sizeof(lengths[0]); ++j) {
        size_t n = lengths[j];
        memset(text + bytes, 'a', n); bytes += n;
        while (n >= 4) { expected[count++] = 257; n -= 4; }
        if (n >= 2) { expected[count++] = 256; n -= 2; }
        if (n) expected[count++] = 'a';
        memcpy(text + bytes, "<S>", 3); bytes += 3;
        expected[count++] = 258;
    }
    track_allocations = true;
    assert(myds_tokenizer_encode_bytes(&tok, text, bytes, ids, 8192, false) == count);
    assert(!memcmp(ids, expected, (size_t)count * sizeof(*ids)));
    assert(live_allocations == 0);
    const size_t allocations = allocation_calls;
    assert(allocations > 2); /* Exercise both initial allocations and growth. */
    for (size_t failure = 1; failure <= allocations; ++failure) {
        allocation_calls = 0; fail_at = failure;
        assert(myds_tokenizer_encode_bytes(&tok, text, bytes, ids, 8192, false) == MYDS_TOKENIZER_ERROR_NOMEM);
        assert(allocation_calls == failure && live_allocations == 0);
        allocation_calls = 0; fail_at = 0;
        assert(myds_tokenizer_encode_bytes(&tok, text, bytes, ids, 8192, false) == count);
        assert(!memcmp(ids, expected, (size_t)count * sizeof(*ids)) && live_allocations == 0);
    }
    /* Error after earlier pieces have allocated and reused both scratch arrays. */
    impl.byte_ids['Z'] = -1; text[bytes] = 'Z';
    assert(myds_tokenizer_encode_bytes(&tok, text, bytes + 1, ids, 8192, false) == MYDS_TOKENIZER_ERROR_UNKNOWN_BYTE);
    assert(live_allocations == 0);
    assert(myds_tokenizer_encode_bytes(&tok, text, bytes, ids, 8192, false) == count);
    assert(!memcmp(ids, expected, (size_t)count * sizeof(*ids)) && live_allocations == 0);
    track_allocations = false;

    /* All threads read the same tokenizer; only request-local scratch mutates. */
    encode_job_t job = {&tok, text, bytes, expected, count};
    pthread_t threads[4];
    for (size_t i = 0; i < 4; ++i) assert(!pthread_create(&threads[i], NULL, encode_thread, &job));
    for (size_t i = 0; i < 4; ++i) assert(!pthread_join(threads[i], NULL));
    printf("PASS: Unicode oracle, %zu allocation failures, cleanup, and concurrent encoding\n", allocations);
    return 0;
}
