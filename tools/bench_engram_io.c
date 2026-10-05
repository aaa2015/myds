/* CPU-only measurement of the production Engram prepare-token path.
 * Input is an actual tokenizer sequence, never uniformly random table rows.
 * Timed intervals include hash, worker dispatch/barrier, pread and FP8 decode.
 * JSON, validation and checksums are outside each token's timed interval.
 * This program does not establish coldness. The companion runner opens a
 * fresh APFS clone and brackets the lookup phase with device read counters.
 * It does not initialize Metal or read projection/backbone tensor payloads. */
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "myds_engram.h"
#include "myds_tokenizer.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { MAX_TOKENS = 32768, MAX_TEXT_BYTES = 4 * 1024 * 1024 };
static const uint64_t FNV_INIT = UINT64_C(14695981039346656037);
static uint64_t hash_bytes(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = data;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static uint64_t hash_u32(uint64_t h, uint32_t v) {
    unsigned char raw[4];
    for (unsigned j = 0; j < 4; ++j) raw[j] = (unsigned char)(v >> (8 * j));
    return hash_bytes(h, raw, sizeof(raw));
}
static double now_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) { perror("clock_gettime"); exit(1); }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') putchar('\\');
        if (*p < 32) printf("\\u%04x", *p); else putchar(*p);
    }
    putchar('"');
}
static unsigned positive(const char *s, unsigned max) {
    if (!s || !*s || *s == '+' || *s == '-' || *s == ' ' || *s == '\t') return 0;
    char *end; errno = 0; unsigned long x = strtoul(s, &end, 10);
    return errno || *end || !x || x > max ? 0 : (unsigned)x;
}
static unsigned char *read_file(const char *path, size_t cap, size_t *length) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || (unsigned long)n > cap || fseek(f, 0, SEEK_SET)) {
        fclose(f); errno = EINVAL; return NULL;
    }
    unsigned char *data = malloc((size_t)n + 1);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data); fclose(f); return NULL;
    }
    if (fclose(f)) { free(data); return NULL; }
    data[n] = 0; *length = (size_t)n; return data;
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static bool gate(bool enabled) {
    if (!enabled) return true;
    fflush(stdout);
    int c = getchar();
    if (c != '\n') { fputs("gate needs newline on stdin\n", stderr); return false; }
    return true;
}
static void usage(void) {
    fputs("usage: bench_engram_io --model MODEL (--tokens-bin LE_I32_FILE | --text UTF8_FILE) [--workers 1|4|8] [--runs 1..8] [--mode cached|nocache] [--gate]\n", stderr);
}

int main(int argc, char **argv) {
    const char *model = NULL, *binary = NULL, *text = NULL, *mode = "cached";
    unsigned workers = 1, runs = 1; bool gated = false;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--gate")) { gated = true; continue; }
        if (i + 1 == argc) { usage(); return 2; }
        const char *value = argv[++i];
        if (!strcmp(arg, "--model")) model = value;
        else if (!strcmp(arg, "--tokens-bin")) binary = value;
        else if (!strcmp(arg, "--text")) text = value;
        else if (!strcmp(arg, "--mode")) mode = value;
        else if (!strcmp(arg, "--workers")) workers = positive(value, 8);
        else if (!strcmp(arg, "--runs")) runs = positive(value, 8);
        else { usage(); return 2; }
    }
    if (!model || (!binary == !text) || (workers != 1 && workers != 4 && workers != 8) ||
        !runs || (strcmp(mode, "cached") && strcmp(mode, "nocache"))) { usage(); return 2; }
    char nworkers[16]; snprintf(nworkers, sizeof(nworkers), "%u", workers);
    if (setenv("MYDS_ENGRAM_IO_THREADS", nworkers, 1) ||
        (!strcmp(mode, "nocache") ? setenv("MYDS_ENGRAM_NOCACHE", "1", 1) : unsetenv("MYDS_ENGRAM_NOCACHE"))) {
        perror("environment"); return 1;
    }
    int status = 1;
    gguf_file_t *gf = NULL; myds_engram_t *e = NULL; myds_tokenizer_t *tok = NULL;
    int32_t *tokens = NULL; uint32_t *expected_rows = NULL;
    uint64_t *expected_values = NULL; double *times = NULL;
    unsigned char *input = NULL;
    size_t input_size = 0, count = 0;
    struct stat before, after;
    if (stat(model, &before)) { perror("stat model"); goto done; }
    gf = gguf_open(model);
    if (!gf) { fputs("GGUF load failed\n", stderr); goto done; }
    const gguf_kv_t *map = gguf_find_key(gf, "deepseek41.engram.token_map");
    const gguf_tensor_t *projection = gguf_find_tensor(gf, "blk.1.engram_kv.weight");
    if (!map || map->type != GGUF_TYPE_ARRAY || map->value.val_arr.elem_type != GGUF_TYPE_UINT32 ||
        !map->value.val_arr.count || map->value.val_arr.count > INT32_MAX || !projection ||
        projection->n_dims != 2 || !projection->dims[1] || projection->dims[1] % 5 ||
        projection->dims[1] / 5 > UINT32_MAX) { fputs("Missing Engram layout\n", stderr); goto done; }
    e = myds_engram_create(gf, model, (uint32_t)map->value.val_arr.count,
                          (uint32_t)(projection->dims[1] / 5));
    if (!e || !myds_engram_set_io_threads(e, workers)) { perror("Engram creation"); goto done; }
    if (!myds_engram_unmap_tables(gf, e)) { perror("unmap tables"); goto done; }
    input = read_file(binary ? binary : text, binary ? MAX_TOKENS * 4u : MAX_TEXT_BYTES, &input_size);
    if (!input || !input_size) { perror("input file"); goto done; }
    tokens = malloc(MAX_TOKENS * sizeof(*tokens));
    if (!tokens) goto done;
    if (binary) {
        if (input_size % 4) { fputs("Token binary must contain complete little-endian int32 values\n", stderr); goto done; }
        count = input_size / 4;
        for (size_t i = 0; i < count; ++i) {
            uint32_t value = 0;
            for (unsigned j = 0; j < 4; ++j) value |= (uint32_t)input[4 * i + j] << (8 * j);
            if (value >= e->layout.vocab_size) { fputs("Invalid input token ID\n", stderr); goto done; }
            tokens[i] = (int32_t)value;
        }
    } else {
        tok = myds_tokenizer_load(gf);
        if (!tok) { fputs("Tokenizer load failed\n", stderr); goto done; }
        int32_t n = myds_tokenizer_encode_bytes(tok, (const char *)input, input_size, tokens, MAX_TOKENS, false);
        if (n <= 0 || n > MAX_TOKENS) { fputs("Tokenization failed or exceeds 32768 tokens\n", stderr); goto done; }
        count = (size_t)n;
    }
    expected_rows = malloc(count * MYDS_ENGRAM_LAYERS * MYDS_ENGRAM_COLS * sizeof(*expected_rows));
    expected_values = calloc(count, sizeof(*expected_values)); times = malloc(count * sizeof(*times));
    if (!expected_rows || !expected_values || !times) goto done;
    myds_engram_history_t oracle;
    myds_engram_history_reset(&oracle);
    if (!myds_engram_hash(&e->layout, &oracle, tokens, NULL, count, expected_rows)) goto done;
    /* Prepare the immutable decode LUT before timing without accessing a table. */
    uint8_t zero[MYDS_ENGRAM_ROW_BYTES] = {0}; float decoded[MYDS_ENGRAM_DIM];
    memset(zero + MYDS_ENGRAM_DIM, 127, 8);
    if (!myds_engram_decode_row(zero, decoded)) goto done;
    uint64_t token_hash = FNV_INIT;
    for (size_t i = 0; i < count; ++i) token_hash = hash_u32(token_hash, (uint32_t)tokens[i]);
    printf("{\"kind\":\"engram_io_ready\",\"model\":"); string(model);
    printf(",\"workers\":%u,\"runs\":%u,\"mode\":", workers, runs); string(mode);
    printf(",\"input_kind\":"); string(binary ? "little_endian_int32_tokens" : "tokenized_utf8_without_added_bos");
    printf(",\"input_bytes_fnv1a64\":\"%016" PRIx64 "\",\"token_ids_fnv1a64\":\"%016" PRIx64 "\",\"token_count\":%zu,\"device\":%llu,\"inode\":%llu,\"file_bytes\":%llu,\"table_offsets\":[%" PRIu64 ",%" PRIu64 "],\"table_rows\":[%u,%u],\"scope\":\"production prepare-token CPU path; no Metal or backbone payload load; coldness requires external device counters\"}\n",
        hash_bytes(FNV_INIT, input, input_size), token_hash, count,
        (unsigned long long)before.st_dev, (unsigned long long)before.st_ino, (unsigned long long)before.st_size,
        e->table[0].offset, e->table[1].offset, e->table[0].rows, e->table[1].rows);
    for (unsigned run = 0; run < runs; ++run) {
        printf("{\"kind\":\"engram_io_lookup_ready\",\"run\":%u}\n", run);
        if (!gate(gated)) goto done;
        myds_engram_reset(e); myds_engram_history_t history;
        myds_engram_history_reset(&history);
        uint64_t rows_hash = FNV_INIT, values_hash = FNV_INIT;
        double sum = 0;
        for (size_t p = 0; p < count; ++p) {
            double start = now_s();
            bool ok = myds_engram_prepare_token(e, tokens[p], p);
            times[p] = (now_s() - start) * 1000;
            if (!ok) { perror("prepare token"); goto done; }
            if (memcmp(&e->row_ids[0][0], expected_rows + p * 48, 48 * sizeof(uint32_t))) {
                fputs("row IDs differ from bulk hash oracle\n", stderr); goto done;
            }
            for (unsigned j = 2; j; --j) history.tail[j] = history.tail[j - 1];
            history.tail[0] = (int32_t)e->layout.token_map[tokens[p]];
            if (!e->prepared || e->position != p + 1 || memcmp(&e->history, &history, sizeof(history))) {
                fputs("history/position transaction mismatch\n", stderr); goto done;
            }
            uint64_t one_rows = FNV_INIT, one_values = FNV_INIT;
            for (unsigned layer = 0; layer < 2; ++layer) {
                for (unsigned col = 0; col < 24; ++col) {
                    uint32_t value = e->row_ids[layer][col];
                    one_rows = hash_u32(one_rows, value); rows_hash = hash_u32(rows_hash, value);
                }
                for (unsigned j = 0; j < MYDS_ENGRAM_INPUT_DIM; ++j) {
                    uint32_t bits; memcpy(&bits, &e->row_values[layer][j], 4);
                    one_values = hash_u32(one_values, bits); values_hash = hash_u32(values_hash, bits);
                }
            }
            if (run && expected_values[p] != one_values) { fputs("lookup repeat values mismatch\n", stderr); goto done; }
            expected_values[p] = one_values; sum += times[p];
            printf("{\"kind\":\"engram_io_token\",\"run\":%u,\"pos\":%zu,\"token\":%d,\"prepare_ms\":%.9f,\"rows_fnv1a64\":\"%016" PRIx64 "\",\"values_fnv1a64\":\"%016" PRIx64 "\",\"history\":[%d,%d,%d],\"row_ids\":[",
                run, p, tokens[p], times[p], one_rows, one_values, history.tail[0], history.tail[1], history.tail[2]);
            for (unsigned j = 0; j < 48; ++j) printf("%s%u", j ? "," : "", e->row_ids[j / 24][j % 24]);
            fputs("]}\n", stdout);
        }
        qsort(times, count, sizeof(*times), compare_double);
        double median = count % 2 ? times[count / 2] : (times[count / 2 - 1] + times[count / 2]) / 2;
        printf("{\"kind\":\"engram_io_lookup_done\",\"run\":%u,\"cold_candidate\":%s,\"token_count\":%zu,\"prepare_ms_sum\":%.9f,\"prepare_ms_per_token\":%.9f,\"prepare_ms_token_median\":%.9f,\"prepare_ms_token_min\":%.9f,\"prepare_ms_token_max\":%.9f,\"rows_fnv1a64\":\"%016" PRIx64 "\",\"values_fnv1a64\":\"%016" PRIx64 "\",\"logical_row_bytes_upper_bound\":%zu,\"history\":[%d,%d,%d],\"valid\":true}\n",
            run, run ? "false" : "true", count, sum, sum / count, median, times[0], times[count - 1],
            rows_hash, values_hash, count * 48 * MYDS_ENGRAM_ROW_BYTES, history.tail[0], history.tail[1], history.tail[2]);
        if (!gate(gated)) goto done;
    }
    if (stat(model, &after) || before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size || before.st_mtime != after.st_mtime) {
        fputs("Model stat changed during measurement\n", stderr); goto done;
    }
    fputs("{\"kind\":\"engram_io_complete\",\"valid\":true}\n", stdout); status = 0;
done:
    free(times); free(expected_values); free(expected_rows); free(tokens); free(input);
    myds_tokenizer_free(tok); myds_engram_free(e); gguf_close(gf);
    return status;
}
