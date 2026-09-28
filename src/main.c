#include "myds.h"
#include "myds_server.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void print_usage(const char *prog) {
    printf("myds - Minimal DeepSeek Inference Engine on Apple Silicon Metal\n\n");
    printf("Usage:\n");
    printf("  %s -m <model.gguf> [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>      Path to DeepSeek GGUF model (required)\n");
    printf("  --port <port>           Run HTTP server on specified port\n");
    printf("  --host <host>           Host to listen on (default: 0.0.0.0)\n");
    printf("  --alias <name>          Model alias for API responses (default: myds)\n");
    printf("  -p, --prompt <string>   Prompt string (if omitted and no --port, starts interactive chat)\n");
    printf("  -n, --max-tokens <int>  Max tokens to generate (default: 256)\n");
    printf("  -c, --ctx, --ctx-size   Context window size (default: 4096)\n");
    printf("  --temp <float>          Sampling temperature (default: 0.7, 0.0 for greedy)\n");
    printf("  --top-p <float>         Nucleus top-p sampling cutoff (default: 0.9)\n");
    printf("  --seed <int>            Random seed (default: random)\n");
    printf("  -h, --help              Show this help message\n");
}

static void token_stream_printer(const char *piece, int32_t token_id, void *user_data) {
    (void)token_id;
    uint32_t *token_counter = (uint32_t *)user_data;
    (*token_counter)++;
    fputs(piece, stdout);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    const char *host = "0.0.0.0";
    const char *alias = "myds";
    int port = 0;

    myds_config_t cfg = {
        .max_seq_len = 4096,
        .max_new_tokens = 256,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .seed = (int32_t)time(NULL),
        .stream = true,
    };

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--alias") == 0 && i + 1 < argc) {
            alias = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            cfg.max_new_tokens = (uint32_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx-size") == 0 || strcmp(argv[i], "--ctx") == 0) && i + 1 < argc) {
            cfg.max_seq_len = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            cfg.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            cfg.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: missing model path (-m <model.gguf>)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("====================================================\n");
    printf("       myds: Minimal DeepSeek Inference Engine      \n");
    printf("         Accelerated by Apple Silicon Metal         \n");
    printf("====================================================\n");

    MYDS_LOG_I("Starting myds (ctx_size=%u, max_new_tokens=%u, temp=%.2f, top_p=%.2f)",
               cfg.max_seq_len, cfg.max_new_tokens, cfg.temperature, cfg.top_p);
    MYDS_LOG_I("Loading model file: %s", model_path);

    clock_t t0 = clock();
    myds_model_t *model = myds_model_load(model_path);
    if (!model) {
        MYDS_LOG_E("Failed to load model from %s", model_path);
        return 1;
    }
    clock_t t1 = clock();
    double load_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Model loaded in %.2f s\n", load_sec);
    MYDS_LOG_I("Model loaded successfully in %.2fs", load_sec);

    myds_model_print_info(model);

    myds_context_t *ctx = myds_context_create(model, &cfg);
    if (!ctx) {
        MYDS_LOG_E("Failed to create inference context");
        myds_model_free(model);
        return 1;
    }
    MYDS_LOG_I("Inference context initialized");

    if (port > 0) {
        /* HTTP Server Mode for Fleet deployment */
        int rc = myds_server_start(ctx, host, port, alias);
        myds_context_free(ctx);
        myds_model_free(model);
        return rc;
    }

    if (prompt) {
        /* Single prompt generation */
        MYDS_LOG_I("Generating response for prompt (%zu chars)...", strlen(prompt));
        printf("\n[Prompt]: %s\n\n[Response]: ", prompt);
        uint32_t token_count = 0;
        clock_t gen_t0 = clock();

        myds_generate(ctx, prompt, token_stream_printer, &token_count);

        clock_t gen_t1 = clock();
        double elapsed = (double)(gen_t1 - gen_t0) / CLOCKS_PER_SEC;
        double speed = (elapsed > 0) ? ((double)token_count / elapsed) : 0.0;
        printf("\n\n----------------------------------------------------\n");
        printf("Generated %u tokens in %.2f s (%.2f tokens/sec)\n", token_count, elapsed, speed);
        printf("----------------------------------------------------\n");
        MYDS_LOG_I("Generation completed: %u tokens in %.2fs (%.2f tokens/s)", token_count, elapsed, speed);
    } else {
        /* Interactive Chat Mode */
        printf("\nEntering interactive chat mode (type 'exit' or Ctrl+D to quit):\n\n");
        char line[2048];
        while (1) {
            printf("\nUser >>> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
            if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) break;
            if (strlen(line) == 0) continue;

            printf("\nDeepSeek >>> ");
            uint32_t token_count = 0;
            clock_t gen_t0 = clock();

            myds_generate(ctx, line, token_stream_printer, &token_count);

            clock_t gen_t1 = clock();
            double elapsed = (double)(gen_t1 - gen_t0) / CLOCKS_PER_SEC;
            double speed = (elapsed > 0) ? ((double)token_count / elapsed) : 0.0;
            printf("\n  [%.1f tokens/s]\n", speed);
        }
    }

    myds_context_free(ctx);
    myds_model_free(model);
    return 0;
}
