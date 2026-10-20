/* 测我们的 tokenizer 在不同长度下的编码耗时（串行 vs 理论并行上限）。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "myds_tokenizer.h"

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <model.gguf>\n", argv[0]); return 2; }
    gguf_file_t *gf = gguf_open(argv[1]);
    if (!gf) { printf("  gguf open failed\n"); return 1; }
    myds_tokenizer_t *tok = myds_tokenizer_load(gf);
    if (!tok) { printf("  tokenizer load failed\n"); return 1; }
    printf("  vocab=%u\n\n", (unsigned)tok->vocab_size);

    /* 构造不同长度的英文/中文混合文本 */
    const char *unit = "The compressor reads rows from the table and folds them "
                       "into the residual stream before the next sublayer. ";
    size_t ulen = strlen(unit);
    size_t lens[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    printf("  %-12s %-10s %-12s %-14s\n", "字符数", "token数", "耗时 ms", "MB/s");
    for (int k = 0; k < 6; k++) {
        size_t want = lens[k];
        char *text = malloc(want + 1);
        for (size_t i = 0; i < want; i++) text[i] = unit[i % ulen];
        text[want] = 0;

        int32_t *ids = malloc(sizeof(int32_t) * (want + 16));
        double best = 0;
        int n = 0;
        for (int rep = 0; rep < 3; rep++) {
            double t0 = now();
            n = myds_tokenizer_encode_bytes(tok, text, strlen(text), ids, (int32_t)(want + 16), false);
            double dt = now() - t0;
            if (rep == 0 || (dt > 0 && dt < best)) best = dt;
        }
        printf("  %-12zu %-10d %-12.1f %-14.1f\n", want, n, best * 1000.0,
               best > 0 ? (want / best / 1e6) : 0.0);
        free(text); free(ids);
    }
    myds_tokenizer_free(tok);
    return 0;
}
