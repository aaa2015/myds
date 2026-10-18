/*
 * score.c — 逐 token 打分：perplexity 与 logits 导出。
 *
 * 为什么要它：myds 与 ds4 都没有 logprobs / 打分端点，bench_phase 只在生成
 * 结束时导出【最后一个位置】的 logits。要比较两个量化版本（例如同一批文本上
 * imatrix 量化 vs 原量化）的差异，必须拿到【每个位置】的全词表 logits。
 *
 * myds_forward_token() 每步都会把全词表 logits 算进 ctx->logits（need_logits=true），
 * 所以本工具不改引擎，只读它：
 *
 *   for i in 0..n-2:
 *       forward(tokens[i], pos=i)      # 位置 i 的分布,预测 tokens[i+1]
 *       read ctx->logits               # 全词表 float32
 *       nll += -(logits[tokens[i+1]] - logsumexp(logits))
 *
 * 不做采样：temperature=0 只影响返回值，我们只读 logits 本身。
 *
 * 用法:
 *   score <model.gguf> --text <file>            [--dump <dir>] [--max-tokens N]
 *   score <model.gguf> --prompt "raw text"      [--dump <dir>] [--max-tokens N]
 *
 * 协议模式（复刻 llama.cpp `perplexity --chunks`，用于与公开数字同口径对比）:
 *   score <model.gguf> --text <file> --context 512 --chunks 8
 *   把 token 流按 context 切成 chunks 段，每段只计分 [context/2, context-2]
 *   这些位置（跳过前半段做预热），于是计分位置数 =
 *   (context - 1 - context//2) * chunks，与 llama.cpp 的 perplexity 一致。
 *
 * --dump 会为每个位置写 <dir>/pos-%06u.f32（全词表 float32，原生字节序），
 * 供外部 Python 算 KL 等指标。体积 = vocab_size * 4 B / token。
 *
 * 终止：文本不足 2 个 token 时拒绝（没有任何可预测的位置）。
 */
#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_tokenizer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)n + 1u);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return NULL; }
    buf[got] = '\0';
    *out_len = got;
    return buf;
}

static int dump_logits(const char *dir, uint32_t pos, const float *logits, uint32_t count) {
    if (!dir) return 1;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/pos-%06u.f32", dir, pos);
    if (n < 0 || (size_t)n >= sizeof(path)) return 0;
    FILE *f = fopen(path, "wbx");   /* 绝不覆盖：同名即视为调用方错误 */
    if (!f) { perror(path); return 0; }
    int ok = fwrite(logits, sizeof(*logits), count, f) == count;
    if (fclose(f) != 0) ok = 0;
    if (!ok) fprintf(stderr, "short write: %s\n", path);
    return ok;
}

typedef struct {
    double   nll;
    uint32_t scored;
    uint32_t top1;
} seg_stats;

/* 跑一段 token：对 pos = 0..n-2 做 forward（KV 必须连续），
 * 但只对 pos >= score_from 的位置计分。
 * dump_base 是这段起点在全局编号里的偏移，协议模式下各段编号连续。*/
static int run_segment(myds_model_t *model, const int32_t *tok, uint32_t n,
                       uint32_t score_from, const char *dump_dir, uint32_t dump_base,
                       seg_stats *out) {
    const myds_config_t cfg = {
        .max_seq_len = n,
        .max_new_tokens = 0,
        .temperature = 0,
        .top_p = 1,
        .top_k = 0,
        .min_p = 0,
        .repetition_penalty = 1,
        .seed = 42,
        .stream = false,
    };
    myds_context_t *ctx = myds_context_create(model, &cfg);
    if (!ctx) { fputs("Context creation failed\n", stderr); return 0; }

    const uint32_t vocab = model->vocab_size;
    int ok = 1;
    for (uint32_t pos = 0; pos + 1 < n; ++pos) {
        if (myds_forward_token(ctx, tok[pos], pos) < 0) {
            fprintf(stderr, "forward failed at pos %u\n", pos);
            ok = 0; break;
        }
        if (pos < score_from) continue;   /* 预热段：照跑 forward，不计分 */

        const float *logits = (const float *)myds_metal_buffer_data(ctx->logits);
        if (!logits) { fputs("no logits buffer\n", stderr); ok = 0; break; }

        float mx = -INFINITY;
        for (uint32_t v = 0; v < vocab; ++v) {
            if (!isfinite(logits[v])) {
                fprintf(stderr, "non-finite logit at pos %u\n", pos);
                ok = 0; break;
            }
            if (logits[v] > mx) mx = logits[v];
        }
        if (!ok) break;

        double sum = 0.0;
        for (uint32_t v = 0; v < vocab; ++v) sum += exp((double)logits[v] - (double)mx);
        const double lse = (double)mx + log(sum);

        const int32_t target = tok[pos + 1];
        out->nll += -(double)logits[target] + lse;   /* -log p(target) */
        ++out->scored;

        /* top-1 命中率：argmax 是否等于真实的下一个 token（不依赖采样） */
        float best = -INFINITY;
        uint32_t best_i = 0;
        for (uint32_t v = 0; v < vocab; ++v) {
            if (logits[v] > best) { best = logits[v]; best_i = v; }
        }
        if ((int32_t)best_i == target) ++out->top1;

        if (!dump_logits(dump_dir, dump_base + pos, logits, vocab)) { ok = 0; break; }
    }

    myds_context_free(ctx);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <model.gguf> (--text <file> | --prompt <string>) "
                "[--dump <dir>] [--max-tokens N]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *text_path = NULL;
    const char *prompt_arg = NULL;
    const char *dump_dir = NULL;
    uint32_t max_tokens = 4096;
    uint32_t ctx_len = 0;      /* 协议模式: 每段 token 数 */
    uint32_t chunks = 0;       /* 协议模式: 段数 */

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_arg = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v < 2 || v > 262144) { fprintf(stderr, "bad --max-tokens: %ld\n", v); return 2; }
            max_tokens = (uint32_t)v;
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v < 2 || v > 262144) { fprintf(stderr, "bad --context: %ld\n", v); return 2; }
            ctx_len = (uint32_t)v;
        } else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v < 1 || v > 100000) { fprintf(stderr, "bad --chunks: %ld\n", v); return 2; }
            chunks = (uint32_t)v;
        } else {
            fprintf(stderr, "unexpected argument: %s\n", argv[i]);
            return 2;
        }
    }
    if ((text_path != NULL) == (prompt_arg != NULL)) {
        fputs("give exactly one of --text / --prompt\n", stderr);
        return 2;
    }
    if ((ctx_len == 0) != (chunks == 0)) {
        fputs("--context and --chunks must be given together\n", stderr);
        return 2;
    }

    size_t text_len = 0;
    char *text = text_path ? read_file(text_path, &text_len) : NULL;
    if (text_path && !text) return 1;
    if (prompt_arg) { text = (char *)prompt_arg; text_len = strlen(prompt_arg); }
    if (text_len == 0) { fputs("empty input\n", stderr); return 2; }

    /* stdout 只放结果 JSON；引擎的原生日志改走 stderr。 */
    fflush(stdout);
    int json_fd = dup(STDOUT_FILENO);
    FILE *json = json_fd < 0 ? NULL : fdopen(json_fd, "w");
    if (!json || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) { perror("redirect stdout"); return 2; }

    myds_model_t *model = myds_model_load(model_path);
    if (!model) { fputs("Model load failed\n", stderr); return 1; }
    if (!model->tokenizer) { fputs("Model has no tokenizer\n", stderr); myds_model_free(model); return 1; }

    /* 先编码：缓冲区按【最坏情况】分配（每个字节一个 token），再检查返回的
     * 真实长度。myds_tokenizer_encode* 在空间不足时返回所需长度且会继续写
     * max_tokens 个槽位，所以绝不能用“预期 token 数”去分配 —— 那会堆溢出。 */
    if (text_len > (size_t)INT32_MAX - 16u) { fputs("input too large\n", stderr); myds_model_free(model); return 2; }
    int32_t cap = (int32_t)text_len + 16;
    int32_t *tokens = malloc((size_t)cap * sizeof(*tokens));
    if (!tokens) { myds_model_free(model); return 1; }
    int32_t n_tokens = myds_tokenizer_encode_bytes(model->tokenizer, text, text_len,
                                                   tokens, cap, true);
    if (n_tokens < 0) {
        fputs("tokenizer encode failed\n", stderr);
        free(tokens); myds_model_free(model);
        return 2;
    }
    if (n_tokens >= cap) {
        fprintf(stderr, "tokenizer needed more than %d tokens\n", cap);
        free(tokens); myds_model_free(model);
        return 2;
    }
    if (chunks == 0 && (uint32_t)n_tokens > max_tokens) {
        fprintf(stderr, "input encodes to %d tokens; raise --max-tokens (currently %u)\n",
                n_tokens, max_tokens);
        free(tokens); myds_model_free(model);
        return 2;
    }
    if (n_tokens < 2) {
        fprintf(stderr, "input encodes to %d token(s); need at least 2\n", n_tokens);
        free(tokens); myds_model_free(model);
        return 2;
    }

    const uint32_t vocab = model->vocab_size;
    double nll = 0.0;
    uint32_t scored = 0;
    uint32_t top1_hits = 0;
    int status = 0;

    if (chunks > 0) {
        if (ctx_len < 2) { fputs("--context must be at least 2\n", stderr); status = 2; }
        const uint64_t need = (uint64_t)ctx_len * (uint64_t)chunks;
        if (status == 0 && (uint64_t)n_tokens < need) {
            fprintf(stderr, "input encodes to %d tokens; protocol needs %u x %u = %llu\n",
                    n_tokens, ctx_len, chunks, (unsigned long long)need);
            status = 2;
        }
        /* llama.cpp perplexity 口径：每段只计分 [context/2, context-2]。 */
        const uint32_t score_from = ctx_len / 2;
        for (uint32_t c = 0; c < chunks && status == 0; ++c) {
            seg_stats s = {0, 0, 0};
            if (!run_segment(model, tokens + (size_t)c * ctx_len, ctx_len, score_from,
                             dump_dir, c * ctx_len, &s)) { status = 1; break; }
            nll += s.nll; scored += s.scored; top1_hits += s.top1;
        }
    } else {
        seg_stats s = {0, 0, 0};
        if (!run_segment(model, tokens, (uint32_t)n_tokens, 0, dump_dir, 0, &s)) status = 1;
        else { nll = s.nll; scored = s.scored; top1_hits = s.top1; }
    }

    if (status == 0) {
        const double mean_nll = nll / (double)scored;
        fprintf(json,
                "{\"kind\":\"score\",\"model\":\"%s\",\"tokens\":%d,\"scored\":%u,"
                "\"vocab\":%u,\"nll_sum\":%.9f,\"nll_mean\":%.9f,\"ppl\":%.9f,"
                "\"top1_hits\":%u,\"top1_rate\":%.6f,\"dump_dir\":%s,"
                "\"context\":%u,\"chunks\":%u,\"score_from\":%u}\n",
                model_path, n_tokens, scored, vocab, nll, mean_nll, exp(mean_nll),
                top1_hits, scored ? (double)top1_hits / (double)scored : 0.0,
                dump_dir ? "true" : "null",
                ctx_len, chunks, chunks > 0 ? ctx_len / 2 : 0u);
        fflush(json);
    }

    myds_model_free(model);
    free(tokens);
    return status;
}
