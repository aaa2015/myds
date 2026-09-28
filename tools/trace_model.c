/* Diagnostic entry point: exact token input, no chat template or sampling
 * dependence. MYDS_TRACE_PREFIX selects F32 stage files in the engine. */
#include "myds_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include "reference/trace_input.h"

int main(int argc, char **argv) {
    if (argc < 3 || (argc == 2 && !strcmp(argv[1], "--help"))) {
        fprintf(stderr, "Usage: %s MODEL TOKEN_ID [TOKEN_ID ...]\n"
                        "       %s MODEL --tokens-file PATH\n", argv[0], argv[0]);
        return argc == 2 && !strcmp(argv[1], "--help") ? 0 : 2;
    }
    trace_tokens tokens = {0};
    if (!trace_tokens_parse(&tokens, argc, argv) ||
        !trace_filter_valid(getenv("MYDS_TRACE_POS")) ||
        !trace_filter_valid(getenv("MYDS_TRACE_LAYER"))) {
        free(tokens.ids);
        fprintf(stderr, "Invalid token sequence or trace filter\n");
        return 2;
    }
    myds_model_t *model = myds_model_load_diagnostic(argv[1]);
    if (!model) { free(tokens.ids); return 1; }
    for (size_t i = 0; i < tokens.count; i++) {
        if ((uint32_t)tokens.ids[i] >= model->vocab_size) {
            fprintf(stderr, "Token %d is outside vocabulary size %u\n", tokens.ids[i], model->vocab_size);
            myds_model_free(model); free(tokens.ids); return 2;
        }
    }
    myds_config_t cfg = {.max_seq_len = (uint32_t)tokens.count + 1,
                         .max_new_tokens = 1, .temperature = 0,
                         .top_p = 1, .seed = 42};
    myds_context_t *ctx = myds_context_create(model, &cfg);
    if (!ctx) { myds_model_free(model); free(tokens.ids); return 1; }
    fprintf(stderr, "TRACE_CONTEXT_READY\n");
    int result = 0;
    for (size_t i = 0; i < tokens.count; i++) {
        fprintf(stderr, "TRACE_FORWARD_BEGIN pos=%zu\n", i);
        int next = myds_forward_token(ctx, tokens.ids[i], (uint32_t)i);
        if (next < 0) { result = 1; break; }
        printf("TRACE_TOKEN pos=%zu input=%d argmax=%d\n", i, tokens.ids[i], next);
        fflush(stdout);
    }
    myds_context_free(ctx);
    myds_model_free(model);
    free(tokens.ids);
    return result;
}
