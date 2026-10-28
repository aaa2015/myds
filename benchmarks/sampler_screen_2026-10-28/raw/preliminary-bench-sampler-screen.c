/* CPU-only screening of the real V4.1 sampler entry point.  The input must be
 * a single native-endian array of 129,280 float32 logits captured elsewhere.
 * It never opens a model or initializes Metal.  Compile with the commands in
 * the companion experiment record; this includes the actual V4.1 entry rather
 * than a copied approximation of the greedy/finite-check dispatch. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/myds_v41.c"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB 129280u
#define SAMPLES 201u
#define WARMUP 32u
static volatile uint64_t sink;

static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static int compare_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr,"usage: %s logits-f32\n",argv[0]); return 2; }
    FILE *f = fopen(argv[1],"rb");
    if (!f) { perror("open logits"); return 2; }
    float *logits = malloc((size_t)VOCAB * sizeof(*logits));
    if (!logits || fread(logits, sizeof(*logits), VOCAB, f) != VOCAB || fgetc(f) != EOF) {
        fprintf(stderr,"input must be exactly %u float32 values\n",VOCAB); return 2;
    }
    if (fclose(f)) { perror("close logits"); return 2; }
    float max = -INFINITY;
    for (uint32_t i=0;i<VOCAB;++i) {
        if (!isfinite(logits[i])) { fprintf(stderr,"non-finite logits\n"); return 2; }
        if (logits[i] > max) max = logits[i];
    }
    myds_context_t ctx = {0};
    myds_config_t cfg = {0};
    ctx.sample_probs = malloc((size_t)VOCAB * sizeof(float));
    ctx.sample_indices = malloc((size_t)VOCAB * sizeof(int32_t));
    /* myds_cand_t in myds_engine.c consists of float prob + int32_t id. */
    ctx.sample_cands = malloc((size_t)VOCAB * (sizeof(float) + sizeof(int32_t)));
    if (!ctx.sample_probs || !ctx.sample_indices || !ctx.sample_cands) return 2;
    ctx.sampler_trace_once = 1;
    ctx.rng_state = 42;
    ctx.recent_count = 64;
    for (uint32_t i=0;i<ctx.recent_count;++i) ctx.recent_tokens[i] = (int32_t)i;
    cfg.repetition_penalty = 1.0f;
    cfg.top_p = 1.0f;
    cfg.top_k = 0;
    cfg.min_p = 0.0f;
    const float temps[] = {0.0f, 0.4f, 1.0f};
    for (size_t j=0;j<sizeof(temps)/sizeof(temps[0]);++j) {
        cfg.temperature = temps[j];
        uint32_t eligible = 0;
        if (temps[j] > 0.0f)
            for (uint32_t i=0;i<VOCAB;++i) if ((logits[i]-max)/temps[j] >= -20.0f) ++eligible;
        for (uint32_t k=0;k<WARMUP;++k) {
            int32_t id = sample_and_verify_token(logits,VOCAB,&cfg,&ctx);
            if (id < 0 || (uint32_t)id >= VOCAB) return 3;
            sink += (uint32_t)id;
        }
        uint64_t raw[SAMPLES], sorted[SAMPLES];
        for (uint32_t k=0;k<SAMPLES;++k) {
            uint64_t start = now_ns();
            int32_t id = sample_and_verify_token(logits,VOCAB,&cfg,&ctx);
            raw[k] = now_ns()-start;
            if (id < 0 || (uint32_t)id >= VOCAB) return 3;
            sink += (uint32_t)id;
        }
        for (uint32_t k=0;k<SAMPLES;++k) {
            sorted[k]=raw[k];
            printf("{\"kind\":\"sample\",\"temperature\":%.1f,\"index\":%u,\"elapsed_ns\":%" PRIu64 "}\n",(double)temps[j],k,raw[k]);
        }
        qsort(sorted,SAMPLES,sizeof(sorted[0]),compare_u64);
        printf("{\"kind\":\"summary\",\"temperature\":%.1f,\"top_p\":1,\"top_k\":0,\"min_p\":0,\"eligible\":%u,\"vocab\":%u,\"warmup\":%u,\"samples\":%u,\"median_ns\":%" PRIu64 ",\"p10_ns\":%" PRIu64 ",\"p90_ns\":%" PRIu64 "}\n",(double)temps[j],eligible,VOCAB,WARMUP,SAMPLES,sorted[100],sorted[20],sorted[180]);
    }
    free(ctx.sample_probs);free(ctx.sample_indices);free(ctx.sample_cands);free(logits);
    return 0;
}
