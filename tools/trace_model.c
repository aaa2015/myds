/* Diagnostic entry point: exact token input, no chat template or sampling
 * dependence. MYDS_TRACE_PREFIX selects F32 stage files in the engine. */
#include "myds_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s MODEL TOKEN_ID [TOKEN_ID ...]\n", argv[0]);
        return 2;
    }
    myds_model_t *model = myds_model_load_diagnostic(argv[1]);
    if (!model) return 1;
    myds_config_t cfg = {.max_seq_len = (uint32_t)argc + 1,
                         .max_new_tokens = 1, .temperature = 0,
                         .top_p = 1, .seed = 42};
    myds_context_t *ctx = myds_context_create(model, &cfg);
    if (!ctx) { myds_model_free(model); return 1; }
    int result = 0;
    for (int i = 2; i < argc; i++) {
        char *end;
        errno = 0;
        long id = strtol(argv[i], &end, 10);
        if (errno || *end || end == argv[i] || id < 0 ||
            id > INT_MAX || (uint32_t)id >= model->vocab_size) {
            fprintf(stderr, "Invalid token ID: %s\n", argv[i]);
            result = 2;
            break;
        }
        int next = myds_forward_token(ctx, (int32_t)id, (uint32_t)i - 2);
        if (next < 0) { result = 1; break; }
        printf("TRACE_TOKEN pos=%d input=%ld argmax=%d\n", i - 2, id, next);
    }
    myds_context_free(ctx);
    myds_model_free(model);
    return result;
}
