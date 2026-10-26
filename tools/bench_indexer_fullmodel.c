/* Build-only full-model driver; no device is created by --preflight.
 *
 * Build with tools/indexer_fullmodel.mk. The checked generated include retains
 * the original driver's timing/context lifecycle; only evidence hooks and its
 * obsolete single-anchor exporter are adapted. No production code is edited.
 *
 * Future runtime adapter: -DMYDS_BENCH_INDEXER_ADAPTER_HEADER='"/ABS/file.inc"'
 * must define static bench_indexer_adapter_mode(), _symbol(bool bf16), and
 * _counters(bench_indexer_provider_counters_t *,bool reset). Counter reads must
 * be host-only, with no dispatch, wait or state mutation except idle reset.
 * encoded means actual candidate encodings. --run requires this adapter.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_configfile.h"
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
extern char **environ;

typedef struct {
    uint64_t requests, encoded, fallbacks, failures, f32_dispatches, bf16_dispatches;
} bench_indexer_provider_counters_t;
#ifdef MYDS_BENCH_INDEXER_ADAPTER_HEADER
#include MYDS_BENCH_INDEXER_ADAPTER_HEADER
#define BENCH_INDEXER_API_AVAILABLE 1
#else
#define BENCH_INDEXER_API_AVAILABLE 0
static int bench_indexer_adapter_mode(void) { return -1; }
static const char *bench_indexer_adapter_symbol(bool bf16) { (void)bf16; return NULL; }
static void bench_indexer_adapter_counters(bench_indexer_provider_counters_t *out, bool reset) {
    (void)reset;
    if (out) memset(out, 0, sizeof(*out));
}
#endif

static void bench_indexer_before_sample(void);
static void bench_indexer_first_callback(void);
static bool bench_indexer_provider_valid(void);
static void bench_indexer_sample_json(FILE *json, uint32_t run);

#ifndef MYDS_BENCH_INDEXER_BASE_INCLUDE
#error "Use tools/indexer_fullmodel.mk to create the checked base include"
#endif
#define main bench_indexer_base_main
#include MYDS_BENCH_INDEXER_BASE_INCLUDE
#undef main

static bench_indexer_provider_counters_t indexer_first;
static void bench_indexer_before_sample(void) {
    memset(&indexer_first, 0, sizeof(indexer_first));
    bench_indexer_adapter_counters(NULL, true);
}
static void bench_indexer_first_callback(void) {
    bench_indexer_adapter_counters(&indexer_first, false);
}
static bool bench_indexer_provider_valid(void) {
    bench_indexer_provider_counters_t counters = {0};
    bench_indexer_adapter_counters(&counters, false);
    const int mode = bench_indexer_adapter_mode();
    const char *f32_symbol = bench_indexer_adapter_symbol(false);
    const char *bf16_symbol = bench_indexer_adapter_symbol(true);
    /* This candidate is prefill-only. A late decode encoding cannot stand in
     * for adoption during the prefill interval, and no candidate work may leak
     * into the N-1 decode intervals. Exact symbol identity belongs to the
     * frozen outer reader; absent/empty selected symbols fail here as well. */
    const bool prefill_adoption = indexer_first.encoded > 0 &&
        counters.encoded == indexer_first.encoded &&
        counters.f32_dispatches == indexer_first.f32_dispatches &&
        counters.bf16_dispatches == indexer_first.bf16_dispatches &&
        f32_symbol && *f32_symbol && bf16_symbol && *bf16_symbol;
    return BENCH_INDEXER_API_AVAILABLE && (mode == 0 || mode == 1) &&
        !counters.failures && !counters.fallbacks &&
        (mode == 0 ? counters.encoded == 0 : prefill_adoption) &&
        counters.requests >= counters.encoded &&
        counters.encoded == counters.f32_dispatches + counters.bf16_dispatches &&
        counters.requests >= indexer_first.requests && counters.encoded >= indexer_first.encoded &&
        counters.fallbacks >= indexer_first.fallbacks && counters.failures >= indexer_first.failures &&
        counters.f32_dispatches >= indexer_first.f32_dispatches &&
        counters.bf16_dispatches >= indexer_first.bf16_dispatches;
}
static void bench_indexer_sample_json(FILE *json, uint32_t run) {
    bench_indexer_provider_counters_t counters = {0};
    bench_indexer_adapter_counters(&counters, false);
    fprintf(json, ",\"subject\":\"prefill_indexer_k_resident\",\"gpu_indexer_provider\":{"
        "\"api_available\":%s,\"mode\":%d,\"run\":%u,\"adoption_valid\":%s,\"symbol_f32\":",
        BENCH_INDEXER_API_AVAILABLE ? "true" : "false", bench_indexer_adapter_mode(), run,
        bench_indexer_provider_valid() ? "true" : "false");
    json_string(json, bench_indexer_adapter_symbol(false));
    fputs(",\"symbol_bf16\":", json); json_string(json, bench_indexer_adapter_symbol(true));
#define IC(field) do { \
    fprintf(json, ",\"" #field "\":%" PRIu64 ",\"through_first_callback_" #field "\":%" PRIu64 ",\"decode_" #field "\":", counters.field, indexer_first.field); \
    if (counters.field >= indexer_first.field) fprintf(json, "%" PRIu64, counters.field - indexer_first.field); \
    else fputs("null", json); \
} while (0)
    IC(requests); IC(encoded); IC(fallbacks); IC(failures); IC(f32_dispatches); IC(bf16_dispatches);
#undef IC
    fputs(",\"first_callback_scope\":\"prefill_plus_first_output_callback\","
          "\"counters_count_encoding_not_gpu_completion\":true}", json);
}

/* Evidence roots must be distinct, existing, empty absolute directories. These
 * checks and all configuration loading precede any model/context creation. */
static bool indexer_empty_directory(const char *name, struct stat *identity) {
    if (!name || name[0] != '/' || stat(name, identity) != 0 || !S_ISDIR(identity->st_mode) ||
        access(name, W_OK | X_OK) != 0) return false;
    DIR *directory = opendir(name);
    if (!directory) return false;
    bool empty = true;
    struct dirent *entry;
    while ((entry = readdir(directory))) {
        if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) { empty = false; break; }
    }
    if (closedir(directory) != 0) empty = false;
    return empty;
}
static int indexer_compare_environment(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}
static bool indexer_configuration_json(const char *path, const char *raw, size_t bytes, int loaded, bool execute) {
    size_t count = 0;
    for (char **entry = environ; entry && *entry; ++entry) if (!strncmp(*entry, "MYDS_", 5)) ++count;
    const char **settings = malloc((count ? count : 1u) * sizeof(*settings));
    if (!settings) return false;
    size_t at = 0;
    for (char **entry = environ; entry && *entry; ++entry) if (!strncmp(*entry, "MYDS_", 5)) settings[at++] = *entry;
    qsort(settings, count, sizeof(*settings), indexer_compare_environment);
    fputs("{\"kind\":\"indexer_metadata\",\"schema_version\":1,\"subject\":\"prefill_indexer_k_resident\","
          "\"fresh_context_per_sample\":true,\"cache_configuration_preserved\":true,\"config_load_override_env\":true,\"config_file\":", stdout);
    json_string(stdout, path);
    fprintf(stdout, ",\"config_loaded_entries\":%d,\"config_file_bytes\":%zu,\"config_file_fnv1a64\":\"%016" PRIx64 "\",\"config_raw\":",
        loaded, bytes, hash_bytes(UINT64_C(14695981039346656037), raw, bytes));
    json_string(stdout, raw);
    fprintf(stdout, ",\"execute_requested\":%s,\"provider_adapter_available\":%s,"
        "\"effective_configuration_scope\":\"all_MYDS_environment_after_config_load; absent_selectors_use_source_defaults\","
        "\"effective_configuration_count\":%zu,\"effective_configuration\":{",
        execute ? "true" : "false", BENCH_INDEXER_API_AVAILABLE ? "true" : "false", count);
    for (size_t i = 0; i < count; ++i) {
        const char *equals = strchr(settings[i], '=');
        if (!equals) { free(settings); return false; }
        size_t length = (size_t)(equals - settings[i]);
        char *key = malloc(length + 1u);
        if (!key) { free(settings); return false; }
        memcpy(key, settings[i], length); key[length] = 0;
        if (i) fputc(',', stdout);
        json_string(stdout, key); fputc(':', stdout); json_string(stdout, equals + 1); free(key);
    }
    fputs("}}\n", stdout); free(settings);
    return fflush(stdout) == 0;
}

int main(int argc, char **argv) {
    const char *config_path = NULL;
    bool execute = false, preflight = false;
    char **base_argv = calloc((size_t)argc + 1u, sizeof(*base_argv));
    if (!base_argv) return 2;
    int base_argc = 1;
    base_argv[0] = argv[0];
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--config")) {
            if (config_path || ++i >= argc) { free(base_argv); return 2; }
            config_path = argv[i];
        } else if (!strcmp(argv[i], "--run")) execute = true;
        else if (!strcmp(argv[i], "--preflight")) preflight = true;
        else base_argv[base_argc++] = argv[i];
    }
    if (!config_path || config_path[0] != '/' || execute == preflight ||
        (execute && (base_argc < 3 || base_argc > 7)) || (preflight && base_argc != 1)) {
        fputs("Usage: bench_indexer_fullmodel --config ABS --preflight\n"
              "   or: bench_indexer_fullmodel --config ABS --run model.gguf request.json [repeats=3] [tokens=64] [ctx=262144] [dspark.gguf]\n", stderr);
        free(base_argv); return 2;
    }
    size_t raw_bytes = 0;
    char *raw_config = read_request(config_path, &raw_bytes);
    if (!raw_config) { free(base_argv); return 2; }
    /* The loader logs INFO to stdout. Keep stdout JSONL-only even before the
     * legacy main installs its own output redirection. */
    fflush(stdout);
    int saved_stdout = dup(STDOUT_FILENO);
    if (saved_stdout < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        if (saved_stdout >= 0) close(saved_stdout);
        free(raw_config); free(base_argv); return 2;
    }
    const int loaded = myds_configfile_load(config_path, true);
    fflush(stdout);
    int restored_stdout = dup2(saved_stdout, STDOUT_FILENO);
    close(saved_stdout);
    if (restored_stdout < 0) { free(raw_config); free(base_argv); return 2; }
    size_t after_bytes = 0;
    char *after = read_request(config_path, &after_bytes);
    bool stable = after && loaded > 0 && after_bytes == raw_bytes && !memcmp(raw_config, after, raw_bytes);
    free(after);
    if (!stable || !getenv("MYDS_PREFIX_CACHE") || strcmp(getenv("MYDS_PREFIX_CACHE"), "1") ||
        !getenv("MYDS_PLD") || strcmp(getenv("MYDS_PLD"), "1")) {
        fputs("A stable explicit configuration with PREFIX_CACHE=1 and PLD=1 is required\n", stderr);
        free(raw_config); free(base_argv); return 2;
    }
    struct stat logits, state;
    if (!indexer_empty_directory(getenv("MYDS_BENCH_LOGITS_DIR"), &logits) ||
        !indexer_empty_directory(getenv("MYDS_BENCH_STATE_DIR"), &state) ||
        (logits.st_dev == state.st_dev && logits.st_ino == state.st_ino)) {
        fputs("Distinct empty absolute raw logits and state directories are required\n", stderr);
        free(raw_config); free(base_argv); return 2;
    }
    if (!indexer_configuration_json(config_path, raw_config, raw_bytes, loaded, execute)) {
        free(raw_config); free(base_argv); return 2;
    }
    free(raw_config);
    if (!execute) { free(base_argv); return 0; }
    if (!BENCH_INDEXER_API_AVAILABLE) {
        fputs("Runtime provider witness adapter is unavailable; refusing a full-model run\n", stderr);
        free(base_argv); return 2;
    }
    int status = bench_indexer_base_main(base_argc, base_argv);
    free(base_argv);
    return status;
}
