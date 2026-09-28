/* Exact-token reference harness. MIT; see the pinned ds4 source LICENSE. */
#include "ds4.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "trace_input.h"

static int usage(FILE *fp) {
    fprintf(fp, "Usage: trace_ds4 MODEL.gguf TOKEN_ID [TOKEN_ID ...]\n"
                "       trace_ds4 MODEL.gguf --tokens-file PATH\n"
                "Exact token IDs, no automatic BOS/chat template. Context is at least 128.\n"
                "Run from the isolated source directory so Metal files match the binary.\n"
                "Use DS4_METAL_GRAPH_DUMP_PREFIX/LAYER/POS to select binary tensor dumps.\n"
                "LAYER/POS accept decimal CSV or all. Positions are zero-based.\n");
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
    if (!trace_filter_matches(layer, 0) || !trace_filter_matches(position, (uint32_t)pos)) return 1;
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
    trace_tokens tokens = {0};
    if (!trace_tokens_parse(&tokens, argc, argv) ||
        !trace_filter_valid(getenv("DS4_METAL_GRAPH_DUMP_LAYER")) ||
        !trace_filter_valid(getenv("DS4_METAL_GRAPH_DUMP_POS"))) {
        fprintf(stderr, "Invalid token sequence or trace filter\n");
        free(tokens.ids);
        return 2;
    }
    const int ctx = tokens.count + 16 < 128 ? 128 : (int)tokens.count + 16;
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
    for (size_t i = 0; i < tokens.count; i++) {
        if (tokens.ids[i] >= vocab) {
            fprintf(stderr, "Token %d is outside vocabulary size %d\n", tokens.ids[i], vocab);
            goto done;
        }
    }
    if (ds4_session_create(&session, engine, ctx)) goto done;
    char error[256] = {0};
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (!logits) goto done;
    for (size_t pos = 0; pos < tokens.count; pos++) {
        /* The first sync creates a valid checkpoint. Every following step uses
         * the exact supplied ID, regardless of the reference's prediction. */
        ds4_tokens first = {.v = tokens.ids, .len = 1, .cap = 1};
        int rc = pos ? ds4_session_eval(session, tokens.ids[pos], error, sizeof(error))
                     : ds4_session_sync(session, &first, error, sizeof(error));
        if (rc || ds4_session_copy_logits(session, logits, vocab) != vocab) {
            fprintf(stderr, "Reference inference failed at pos %zu: %s\n", pos, error);
            free(logits); goto done;
        }
        if (!write_frontier_logits(logits, vocab, (int)pos)) { free(logits); goto done; }
        int nonfinite = 0;
        for (int i = 0; i < vocab; i++) nonfinite += !isfinite(logits[i]);
        const int best = ds4_session_argmax(session);
        if (best < 0 || best >= vocab || nonfinite) {
            fprintf(stderr, "Invalid logits at pos %zu (argmax=%d, nonfinite=%d)\n", pos, best, nonfinite);
            free(logits); goto done;
        }
        printf("{\"source\":\"ds4\",\"position\":%zu,\"input_token\":%d,"
               "\"prompt_tokens\":%zu,\"vocab\":%d,\"argmax\":%d,"
               "\"argmax_logit\":%.9g,\"nonfinite\":%d}\n",
               pos, tokens.ids[pos], pos + 1, vocab, best, logits[best], nonfinite);
        fflush(stdout);
    }
    free(logits);
    result = 0;
done:
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    free(tokens.ids);
    return result;
}
