/* Shared, bounded exact-token input for the two independent trace harnesses.
 * Only diagnostic input plumbing is shared; no inference implementation is. */
#ifndef MYDS_TRACE_INPUT_H
#define MYDS_TRACE_INPUT_H
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRACE_TOKEN_LIMIT 1048576u
typedef struct { int *ids; size_t count, capacity; } trace_tokens;

/* An unset filter or "all" matches everything. A malformed filter matches
 * nothing, never a silently truncated first number from strtoul("1,8",...). */
static inline int trace_filter_valid(const char *filter) {
    if (!filter || !*filter || !strcmp(filter, "all")) return 1;
    const char *p = filter;
    for (;;) {
        if (!isdigit((unsigned char)*p)) return 0;
        errno = 0;
        char *end;
        unsigned long n = strtoul(p, &end, 10);
        if (errno || n > UINT32_MAX) return 0;
        if (!*end) return 1;
        if (*end != ',' || !end[1]) return 0;
        p = end + 1;
    }
}

static inline int trace_filter_matches(const char *filter, uint32_t value) {
    if (!trace_filter_valid(filter)) return 0;
    if (!filter || !*filter || !strcmp(filter, "all")) return 1;
    const char *p = filter;
    do {
        char *end;
        if (strtoul(p, &end, 10) == value) return 1;
        p = *end ? end + 1 : end;
    } while (*p);
    return 0;
}

static inline int trace_tokens_add(trace_tokens *tokens, const char *word) {
    if (!word || !*word || !isdigit((unsigned char)*word)) return 0;
    errno = 0;
    char *end;
    long id = strtol(word, &end, 10);
    if (errno || *end || id < 0 || id > INT_MAX || tokens->count >= TRACE_TOKEN_LIMIT)
        return 0;
    if (tokens->count == tokens->capacity) {
        size_t capacity = tokens->capacity ? tokens->capacity * 2 : 256;
        int *ids = realloc(tokens->ids, capacity * sizeof(*ids));
        if (!ids) return 0;
        tokens->ids = ids;
        tokens->capacity = capacity;
    }
    tokens->ids[tokens->count++] = (int)id;
    return 1;
}

static inline int trace_tokens_file(trace_tokens *tokens, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 0; }
    char word[32];
    size_t length = 0;
    int ch, ok = 1, comment = 0;
    while ((ch = fgetc(fp)) != EOF) {
        if (comment) { if (ch == '\n') comment = 0; continue; }
        if (isspace((unsigned char)ch) || ch == '#') {
            if (length) {
                word[length] = '\0';
                if (!trace_tokens_add(tokens, word)) { ok = 0; break; }
                length = 0;
            }
            if (ch == '#') comment = 1;
        } else {
            if (length + 1 >= sizeof(word) || !isdigit((unsigned char)ch)) { ok = 0; break; }
            word[length++] = (char)ch;
        }
    }
    if (ok && length) { word[length] = '\0'; ok = trace_tokens_add(tokens, word); }
    if (ferror(fp)) ok = 0;
    if (fclose(fp)) ok = 0;
    if (!ok) fprintf(stderr, "Invalid or unreadable token file: %s\n", path);
    return ok;
}

static inline int trace_tokens_parse(trace_tokens *tokens, int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[2], "--tokens-file"))
        return trace_tokens_file(tokens, argv[3]) && tokens->count > 0;
    if (argc < 3) return 0;
    for (int i = 2; i < argc; i++) {
        if (!trace_tokens_add(tokens, argv[i])) {
            fprintf(stderr, "Invalid token ID: %s\n", argv[i]);
            return 0;
        }
    }
    return tokens->count > 0;
}
#endif
