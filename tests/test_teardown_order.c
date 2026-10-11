/* CPU-only oracle for the real Engine destructors. Include Engine and replace
 * only the GPU drain / resource-release boundaries; never load model weights
 * or initialize Metal. Link project objects except myds_engine.o / main.o.
 * The independent public DSpark destructor is not exercised by this fixture. */
#define _POSIX_C_SOURCE 200809L
#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_dspark.h"
#include "myds_v41.h"
#include "myds_gguf.h"
#include "myds_tokenizer.h"
#include "myds_log.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum event { SYNC, ERROR_LOG, DSPARK_FREE, V41_CONTEXT_FREE, V41_MODEL_FREE,
             BUFFER_FREE, TOKENIZER_FREE, GGUF_CLOSE };
typedef struct {
    enum event events[32];
    unsigned count, sync_calls, errors, expected_sync, expected_errors;
    int sync_results[2];
    bool drained;
} fixture_t;
static fixture_t *active;

static void record(enum event event) {
    assert(active && active->count < 32u);
    active->events[active->count++] = event;
}
static int host_synchronize(void) {
    record(SYNC);
    const unsigned call = active->sync_calls++;
    assert(call < active->expected_sync);
    /* First failure may leave older inflight submissions to drain. On the
     * second call every wait has finished even if those commands failed. */
    if (call + 1u == active->expected_sync) active->drained = true;
    return active->sync_results[call];
}
static void host_log(myds_log_level_t level, const char *format, ...) {
    assert(level == MYDS_LOG_ERROR);
    char message[512];
    va_list args;
    va_start(args, format);
    const int n = vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    assert(n > 0 && (size_t)n < sizeof(message));
    assert(strstr(message, "teardown:") && strstr(message, "failed"));
    if (active->errors == 0u) {
        assert(strstr(message, "draining prior submissions"));
    } else {
        assert(strstr(message, "after blocking wait") &&
               strstr(message, "without recovery"));
    }
    active->errors++;
    record(ERROR_LOG);
}
static void host_gguf_close(gguf_file_t *gf) {
    assert(active->drained && gf);
    record(GGUF_CLOSE);
    free(gf);
}
static void host_dspark_free(myds_dspark_t *ds) {
    assert(active->drained && ds);
    record(DSPARK_FREE);
    host_gguf_close(ds->gf);
    free(ds);
}
static void host_v41_context_free(myds_v41_context_t *ctx) {
    assert(active->drained && ctx);
    record(V41_CONTEXT_FREE);
    free(ctx);
}
static void host_v41_model_free(myds_v41_model_t *model) {
    assert(active->drained && model);
    record(V41_MODEL_FREE);
    free(model);
}
static void host_buffer_free(myds_metal_buffer_t *buffer) {
    if (!buffer) return;
    assert(active->drained);
    record(BUFFER_FREE);
    free(buffer);
}
static void host_tokenizer_free(myds_tokenizer_t *tokenizer) {
    assert(active->drained && tokenizer);
    record(TOKENIZER_FREE);
    free(tokenizer);
}

#define myds_metal_synchronize host_synchronize
#define myds_dspark_free host_dspark_free
#define myds_v41_context_free host_v41_context_free
#define myds_v41_model_free host_v41_model_free
#define gguf_close host_gguf_close
#define myds_metal_buffer_free host_buffer_free
#define myds_tokenizer_free host_tokenizer_free
#define myds_log host_log
#include "../src/myds_engine.c"
#undef myds_metal_synchronize
#undef myds_dspark_free
#undef myds_v41_context_free
#undef myds_v41_model_free
#undef gguf_close
#undef myds_metal_buffer_free
#undef myds_tokenizer_free
#undef myds_log

static void *allocated(size_t bytes) {
    void *p = calloc(1, bytes);
    assert(p);
    return p;
}
static myds_metal_buffer_t *buffer(void) {
    /* Release stub owns this sentinel; no real Metal object exists. */
    return allocated(1u);
}
static fixture_t fixture(unsigned mode) {
    assert(mode < 3u);
    return (fixture_t){ .sync_results = { mode == 0u, mode == 1u },
        .expected_sync = mode == 0u ? 1u : 2u,
        .expected_errors = mode };
}
static void check_trace(const fixture_t *f, const enum event *tail, unsigned n) {
    assert(f->drained && f->sync_calls == f->expected_sync &&
           f->errors == f->expected_errors);
    unsigned offset = 0u;
    assert(f->events[offset++] == SYNC);
    if (f->expected_errors) {
        assert(f->events[offset++] == ERROR_LOG);
        assert(f->events[offset++] == SYNC);
        if (f->expected_errors == 2u) assert(f->events[offset++] == ERROR_LOG);
    }
    assert(f->count == offset + n);
    for (unsigned i = 0; i < n; ++i) assert(f->events[offset + i] == tail[i]);
}
static void context_case(unsigned mode) {
    fixture_t f = fixture(mode); active = &f;
    myds_context_t *ctx = allocated(sizeof(*ctx));
    ctx->dspark = allocated(sizeof(*ctx->dspark));
    ctx->dspark->gf = allocated(sizeof(*ctx->dspark->gf));
    ctx->v41 = allocated(1u);
    ctx->x = buffer(); ctx->logits = buffer();
    ctx->decode_cycles = allocated(1u);
    ctx->pcache_tokens = allocated(4u); ctx->pcache_len = ctx->pcache_cap = 1u;
    ctx->sample_probs = allocated(4u); ctx->sample_indices = allocated(4u);
    ctx->sample_cands = allocated(1u);
    myds_context_free(ctx);
    const enum event tail[] = { DSPARK_FREE, GGUF_CLOSE, V41_CONTEXT_FREE,
                                BUFFER_FREE, BUFFER_FREE };
    check_trace(&f, tail, sizeof(tail) / sizeof(*tail));
    active = NULL;
}
static void model_case(unsigned mode, bool tied_output) {
    fixture_t f = fixture(mode); active = &f;
    myds_model_t *model = allocated(sizeof(*model));
    model->v41 = allocated(1u);
    model->gf = allocated(sizeof(*model->gf));
    model->tokenizer = allocated(sizeof(*model->tokenizer));
    model->tok_embeddings = buffer(); model->output_norm = buffer();
    model->output = tied_output ? model->tok_embeddings : buffer();
    model->n_layers = 1u; model->layers = allocated(sizeof(*model->layers));
    model->layers[0].attn_norm = buffer();
    myds_model_free(model);
    const enum event tied[] = { V41_MODEL_FREE, BUFFER_FREE, BUFFER_FREE,
        BUFFER_FREE, TOKENIZER_FREE, GGUF_CLOSE };
    const enum event separate[] = { V41_MODEL_FREE, BUFFER_FREE, BUFFER_FREE,
        BUFFER_FREE, BUFFER_FREE, TOKENIZER_FREE, GGUF_CLOSE };
    check_trace(&f, tied_output ? tied : separate,
                tied_output ? sizeof(tied) / sizeof(*tied) :
                              sizeof(separate) / sizeof(*separate));
    active = NULL;
}
int main(void) {
    /* NULL destructors must not initialize/drain a runtime. */
    myds_context_free(NULL); myds_model_free(NULL);
    for (unsigned mode = 0; mode < 3u; ++mode) {
        context_case(mode);
        model_case(mode, false);
        model_case(mode, true);
    }
    puts("PASS teardown order: 9 CPU cases; sync 1, 0->1, 0->0; context/model mmap release follows drain; errors remain observable");
    return 0;
}
