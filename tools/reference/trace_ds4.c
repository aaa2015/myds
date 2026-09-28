/* Exact-token reference harness. MIT; see the pinned ds4 source LICENSE. */
#include "ds4.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(FILE *fp) {
    fprintf(fp, "Usage: trace_ds4 MODEL.gguf TOKEN_ID [TOKEN_ID ...]\n"
                "Exact token IDs, no automatic BOS/chat template. Context is at least 128.\n"
                "Run from the isolated source directory so Metal files match the binary.\n"
                "Use DS4_METAL_GRAPH_DUMP_PREFIX/LAYER/POS to select binary tensor dumps.\n");
    return fp == stdout ? 0 : 2;
}

/* Session sync may calculate the head after advancing its internal position.
 * The public token frontier is the unambiguous position for a logits capture. */
static int write_frontier_logits(const float *logits, int vocab, int pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    const char *layer = getenv("DS4_METAL_GRAPH_DUMP_LAYER");
    const char *position = getenv("DS4_METAL_GRAPH_DUMP_POS");
    const char *names = getenv("DS4_METAL_GRAPH_DUMP_NAME");
    if (!prefix || !prefix[0]) return 1;
    if (layer && layer[0] && strcmp(layer, "all") && strtoul(layer, NULL, 10) != 0) return 1;
    if (position && position[0] && strtoul(position, NULL, 10) != (unsigned long)pos) return 1;
    if (names && names[0] && !strstr(names, "logits")) return 1;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s_logits-0_pos%d.bin", prefix, pos);
    if (n < 0 || (size_t)n >= sizeof(path)) return 0;
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror(path); return 0; }
    const size_t written = fwrite(logits, sizeof(*logits), (size_t)vocab, fp);
    const int closed = fclose(fp);
    if (written != (size_t)vocab || closed) {
        fprintf(stderr, "Cannot finish logit trace: %s\n", path);
        return 0;
    }
    fprintf(stderr, "ds4: dumped frontier logits pos=%d to %s\n", pos, path);
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--help")) return usage(stdout);
    if (argc < 3 || argc > 1048576) return usage(stderr);
    ds4_tokens tokens = {0};
    for (int i = 2; i < argc; i++) {
        char *end = NULL;
        errno = 0;
        long id = strtol(argv[i], &end, 10);
        if (errno || !argv[i][0] || *end || id < 0 || id > INT_MAX) {
            fprintf(stderr, "Invalid token ID: %s\n", argv[i]);
            ds4_tokens_free(&tokens);
            return 2;
        }
        ds4_tokens_push(&tokens, (int)id);
    }
    const int ctx = tokens.len + 16 < 128 ? 128 : tokens.len + 16;
    ds4_engine_options options = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .n_threads = 8, .context_size = ctx, .prefill_chunk = 1,
        .power_percent = 100, .placement_session_count_hint = 1,
    };
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    int result = 1;
    if (ds4_engine_open(&engine, &options)) goto done;
    if (!ds4_engine_is_deepseek41(engine)) {
        fprintf(stderr, "This trace harness requires DeepSeek V4.1\n");
        goto done;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    for (int i = 0; i < tokens.len; i++) {
        if (tokens.v[i] >= vocab) {
            fprintf(stderr, "Token %d is outside vocabulary size %d\n", tokens.v[i], vocab);
            goto done;
        }
    }
    if (ds4_session_create(&session, engine, ctx)) goto done;
    char error[256] = {0};
    if (ds4_session_sync(session, &tokens, error, sizeof(error))) {
        fprintf(stderr, "Reference inference failed: %s\n", error);
        goto done;
    }
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (!logits) goto done;
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        free(logits);
        goto done;
    }
    if (!write_frontier_logits(logits, vocab, tokens.len - 1)) {
        free(logits);
        goto done;
    }
    int nonfinite = 0;
    for (int i = 0; i < vocab; i++) nonfinite += !isfinite(logits[i]);
    const int best = ds4_session_argmax(session);
    printf("{\"source\":\"ds4\",\"prompt_tokens\":%d,\"vocab\":%d,"
           "\"argmax\":%d,\"argmax_logit\":%.9g,\"nonfinite\":%d}\n",
           tokens.len, vocab, best, logits[best], nonfinite);
    free(logits);
    result = nonfinite ? 1 : 0;
done:
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    ds4_tokens_free(&tokens);
    return result;
}
