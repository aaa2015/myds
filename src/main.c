#include "myds.h"
#include "myds_server.h"
#include "myds_vjob.h"
#include "myds_dspark.h"
#include "myds_metal.h"
#include "myds_log.h"
#include "myds_configfile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

static void print_usage(const char *prog) {
    printf("myds - Minimal DeepSeek V4 / V4.1 Inference Engine on Apple Silicon Metal\n\n");
    printf("Usage:\n");
    printf("  %s -m <model.gguf> [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>      Path to DeepSeek V4/V4.1 GGUF model (required)\n");
    printf("  --port <port>           Run HTTP server on specified port\n");
    printf("  --host <host>           Host to listen on (default: 0.0.0.0)\n");
    printf("  --alias <name>          Model alias for API responses (default: myds)\n");
    printf("  --vision <file>         Vision encoder GGUF (enables image input; omit = text only)\n");
    printf("  --dspark, --mtp-model <path>  DSpark support model GGUF for speculative decoding\n");
    printf("  --dspark-draft <int>    Draft tokens per speculative cycle (default: 5)\n");
    printf("  -p, --prompt <string>   Prompt string (if omitted and no --port, starts interactive chat)\n");
    printf("  -n, --max-tokens <int>  Max tokens to generate (default: 256)\n");
    printf("  -c, --ctx, --ctx-size   Context window size (default: 4096)\n");
    printf("  --temp <float>          Sampling temperature (default: 0.7, 0.0 for greedy)\n");
    printf("  --top-p <float>         Nucleus top-p sampling cutoff (default: 0.9)\n");
    printf("  --thinking [on|off]     Enable thinking / reasoning mode (default: off)\n");
    printf("  --no-thinking           Disable thinking / reasoning mode\n");
    printf("  --pld                   Enable Prompt-Lookup Decoding (speculative draft)\n");
    printf("  --no-pld                Disable Prompt-Lookup Decoding\n");
    printf("  -h, --help              Show this help message\n");
}

static inline double myds_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct {
    uint32_t token_count;
    double t_start;
    double t_first;
    double t_last;
    bool print_tokens;
    int32_t *tokens;
    size_t tokens_cap;
} cli_stream_ctx_t;

static void token_stream_printer(const char *piece, size_t length, int32_t token_id, void *user_data) {
    cli_stream_ctx_t *sctx = (cli_stream_ctx_t *)user_data;
    double now = myds_now_s();
    if (sctx->token_count == 0) {
        sctx->t_first = now;
    }
    sctx->t_last = now;
    if (sctx->print_tokens && sctx->tokens && sctx->token_count < sctx->tokens_cap) {
        sctx->tokens[sctx->token_count] = token_id;
    }
    sctx->token_count++;
    fwrite(piece, 1, length, stdout);
    fflush(stdout);
}

/* 配置文件的默认搜索顺序。第一个存在的会被加载。
 * 相对路径按当前工作目录解析——生产由 start_fleet.sh 传绝对路径，不依赖 cwd。 */
static const char *const kConfigCandidates[] = {
    "myds.conf",                       /* 当前目录 */
    "./etc/myds.conf",
    NULL
};

/* 在解析任何环境变量之前把配置文件读成环境变量。
 * 这是整个方案的关键：180 处 getenv 一行都不用改。
 * 返回真正生效的条数（0 = 没找到配置文件，走纯环境变量 + 代码默认值）。 */
static int load_config_early(int argc, char *argv[]) {
    const char *path = NULL;
    /* --config <path> 优先 */
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--config") == 0) { path = argv[i + 1]; break; }
    }
    if (!path) {
        const char *e = getenv("MYDS_CONFIG");
        if (e && *e) path = e;
    }
    if (!path) {
        for (int i = 0; kConfigCandidates[i]; i++) {
            FILE *f = fopen(kConfigCandidates[i], "r");
            if (f) { fclose(f); path = kConfigCandidates[i]; break; }
        }
    }
    if (!path) return 0;   /* 没有配置文件不是错误 */

    /* 2026-10-24：配置文件优先。
     *
     * 原设计允许环境变量盖住配置（"临时覆盖一次"），实测里这会造成两个真相源：
     * 同一个开关在不同地方看到不同的值，排查时先要确认谁生效。
     * 现在配置文件是唯一来源；环境变量里的 MYDS_* 会被配置值覆盖。
     * 唯一例外是 MYDS_CONFIG 本身（它在上方已读过，只用于定位本文件）。 */
    return myds_configfile_load(path, true);
}

int main(int argc, char *argv[]) {
    /* ★ 必须最先执行：下面所有函数都会 getenv */
    const int cfg_n = load_config_early(argc, argv);

    /* --dump-config <path>：把当前生效的 MYDS_* 全量导出（用于从生产进程抓权威配置） */
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--dump-config") == 0) {
            const int n = myds_configfile_dump_env(argv[i + 1]);
            return n >= 0 ? 0 : 1;
        }
    }
    (void)cfg_n;

    const int capture_mode = myds_dspark_capture_parse_mode();
    if (capture_mode < 0) return 1;
    if (myds_metal_f16_gemv_bf16_fuse_parse_mode() < 0) return 1;
    if (myds_metal_indexer_rope_quant_fuse_parse_mode() < 0) return 1;
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    const char *model_path = NULL;
    const char *prompt = NULL;
    const char *host = "0.0.0.0";
    const char *alias = "myds";
    const char *vision_path = NULL;
    const char *dspark_path = NULL;
    uint32_t dspark_draft = 0;
    int port = 0;
    bool print_tokens = false;

    myds_config_t cfg = {
        .max_seq_len = 4096,
        .max_new_tokens = 256,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
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
        } else if (strcmp(argv[i], "--vision") == 0 && i + 1 < argc) {
            vision_path = argv[++i];
        } else if ((strcmp(argv[i], "--dspark") == 0 || strcmp(argv[i], "--mtp-model") == 0) && i + 1 < argc) {
            dspark_path = argv[++i];
        } else if (strcmp(argv[i], "--dspark-draft") == 0 && i + 1 < argc) {
            dspark_draft = (uint32_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
            const char *pfile = argv[++i];
            FILE *f = fopen(pfile, "rb");
            if (!f) {
                fprintf(stderr, "Error: cannot open prompt file '%s'\n", pfile);
                return 1;
            }
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = malloc(sz + 1);
            if (!buf) { fclose(f); return 1; }
            if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return 1; }
            buf[sz] = '\0';
            fclose(f);
            prompt = buf;
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            cfg.max_new_tokens = (uint32_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx-size") == 0 || strcmp(argv[i], "--ctx") == 0) && i + 1 < argc) {
            cfg.max_seq_len = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            cfg.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            cfg.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            cfg.top_k = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--thinking") == 0) {
            if (i + 1 < argc && (!strcmp(argv[i+1], "on") || !strcmp(argv[i+1], "true") || !strcmp(argv[i+1], "1"))) {
                cfg.thinking_mode = true;
                i++;
            } else if (i + 1 < argc && (!strcmp(argv[i+1], "off") || !strcmp(argv[i+1], "false") || !strcmp(argv[i+1], "0"))) {
                cfg.thinking_mode = false;
                i++;
            } else {
                cfg.thinking_mode = true;
            }
        } else if (strcmp(argv[i], "--no-thinking") == 0) {
            cfg.thinking_mode = false;
        } else if (strcmp(argv[i], "--print-tokens") == 0) {
            print_tokens = true;
        } else if (strcmp(argv[i], "--pld") == 0) {
            setenv("MYDS_PLD", "1", 1);
        } else if (strcmp(argv[i], "--no-pld") == 0) {
            setenv("MYDS_PLD", "0", 1);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!dspark_path) {
        const char *dsp_env = getenv("MYDS_DSPARK");
        if (!dsp_env) dsp_env = getenv("DS4_DSPARK");
        if (dsp_env && atoi(dsp_env) != 0) {
            const char *dsp_file = getenv("MYDS_DSPARK_FILE");
            if (!dsp_file) dsp_file = getenv("DS4_DSPARK_FILE");
            if (dsp_file && access(dsp_file, R_OK) == 0) {
                dspark_path = dsp_file;
            }
        }
    }

    if (capture_mode != 0 && !dspark_path) {
        MYDS_LOG_E("DSpark capture bypass requires a support model path (--dspark or MYDS_DSPARK_FILE)");
        return 1;
    }

    if (!model_path) {
        fprintf(stderr, "Error: missing model path (-m <model.gguf>)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("====================================================\n");
    printf("    myds: Minimal DeepSeek V4 / V4.1 Inference Engine   \n");
    printf("         Accelerated by Apple Silicon Metal         \n");
    printf("====================================================\n");

    MYDS_LOG_I("Starting myds (ctx_size=%u, max_new_tokens=%u, temp=%.2f, top_p=%.2f)",
               cfg.max_seq_len, cfg.max_new_tokens, cfg.temperature, cfg.top_p);
    MYDS_LOG_I("Loading model file: %s", model_path);

    double load_t0 = myds_now_s();
    myds_model_t *model = myds_model_load(model_path);
    if (!model) {
        MYDS_LOG_E("Failed to load model from %s", model_path);
        return 1;
    }
    double load_t1 = myds_now_s();
    double load_sec = load_t1 - load_t0;
    MYDS_LOG_I("Model loaded successfully in %.2fs", load_sec);

    myds_model_print_info(model);

    myds_context_t *ctx = myds_context_create(model, &cfg);
    if (!ctx) {
        MYDS_LOG_E("Failed to create inference context");
        myds_model_free(model);
        return 1;
    }
    MYDS_LOG_I("Inference context initialized");

    /* 视觉编码器（可选）。★ 初始化失败【不致命】—— 服务照常起，只是不接受图片。
     * 这样配置写错一个路径不会把整个服务拖down；错误在日志里说清楚。 */
    if (vision_path) {
        if (!myds_vjob_init(ctx, vision_path)) {
            MYDS_LOG_E("视觉编码器初始化失败（%s）—— 本次以【纯文本】模式运行", vision_path);
        } else {
            MYDS_LOG_I("多模态已启用：接受 input_image（image_url 需为 data: base64 URL）");
        }
    }

    if (dspark_path) {
        ctx->dspark = myds_dspark_load(dspark_path, model->vocab_size);
        if (!ctx->dspark) {
            MYDS_LOG_E("Failed to load DSpark support model from %s", dspark_path);
            if (capture_mode != 0) {
                myds_context_free(ctx);
                myds_model_free(model);
                return 1;
            }
        } else {
            if (dspark_draft > 0) ctx->dspark->block_size = dspark_draft;
            MYDS_LOG_I("DSpark speculative decoding enabled: block_size=%u, stages=%u, markov_rank=%u",
                       ctx->dspark->block_size, ctx->dspark->stage_count, ctx->dspark->markov_rank);
        }
    }
    if (capture_mode != 0 && !ctx->dspark) {
        MYDS_LOG_E("DSpark capture bypass requires a loaded support model");
        myds_context_free(ctx);
        myds_model_free(model);
        return 1;
    }

    if (port > 0) {
        /* HTTP Server Mode for Fleet deployment */
        int rc = myds_server_start(ctx, host, port, alias);
        myds_context_free(ctx);
        myds_model_free(model);
        return rc;
    }

    int exit_code = 0;
    if (prompt) {
        /* Single prompt generation */
        MYDS_LOG_I("Generating response for prompt (%zu chars)...", strlen(prompt));
        printf("\n[Prompt]: %s\n\n[Response]: ", prompt);
        int32_t *tokens_buf = print_tokens ? (int32_t *)malloc(cfg.max_new_tokens * sizeof(int32_t)) : NULL;
        cli_stream_ctx_t sctx = {
            .token_count = 0,
            .t_start = myds_now_s(),
            .t_first = 0.0,
            .t_last = 0.0,
            .print_tokens = print_tokens,
            .tokens = tokens_buf,
            .tokens_cap = cfg.max_new_tokens,
        };

        myds_stop_reason_t stop_reason = MYDS_STOP_EOS;
        uint32_t prompt_tokens = 0;
        int generated = myds_generate_bytes_ex(ctx, prompt, token_stream_printer, &sctx,
                                               &stop_reason, &prompt_tokens, NULL);

        double t_end = myds_now_s();
        double wall_s = t_end - sctx.t_start;
        double prefill_s = (sctx.token_count > 0 && sctx.t_first > sctx.t_start) ? (sctx.t_first - sctx.t_start) : wall_s;
        double ttft_s = prefill_s;
        double decode_s = (sctx.token_count > 1 && sctx.t_last > sctx.t_first) ? (sctx.t_last - sctx.t_first) : 0.0;
        double decode_tps = (decode_s > 0.0 && sctx.token_count > 1) ? ((double)(sctx.token_count - 1) / decode_s) : ((wall_s > 0.0) ? ((double)sctx.token_count / wall_s) : 0.0);

        if (print_tokens && tokens_buf && sctx.token_count > 0) {
            uint32_t n_print = sctx.token_count < cfg.max_new_tokens ? sctx.token_count : cfg.max_new_tokens;
            printf("\nTOKEN_IDS: [");
            for (uint32_t ti = 0; ti < n_print; ti++) {
                printf("%d%s", tokens_buf[ti], (ti + 1 < n_print) ? "," : "");
            }
            printf("]\n");
        }
        free(tokens_buf);

        printf("\n\n----------------------------------------------------\n");
        printf("load %.2f s | prefill %.2f s (prompt %u tok, TTFT %.2f s) | decode %u tok in %.2f s = %.2f t/s | wall %.2f s\n",
               load_sec, prefill_s, prompt_tokens, ttft_s, sctx.token_count, decode_s, decode_tps, wall_s);
        if (ctx->dspark && ctx->dspark->proposed_tokens > 0) {
            double acc_rate = 100.0 * (double)ctx->dspark->accepted_tokens / (double)ctx->dspark->proposed_tokens;
            printf("DSpark speculative: cycles=%llu proposed=%llu accepted=%llu (%.2f%%) propose_time=%.2fms (avg %.2fms/cycle)\n",
                   (unsigned long long)ctx->dspark->cycles,
                   (unsigned long long)ctx->dspark->proposed_tokens,
                   (unsigned long long)ctx->dspark->accepted_tokens,
                   acc_rate,
                   ctx->dspark->propose_ms,
                   ctx->dspark->cycles > 0 ? ctx->dspark->propose_ms / (double)ctx->dspark->cycles : 0.0);
        }
        printf("----------------------------------------------------\n");
        MYDS_LOG_I("Generation metrics: load %.2fs | prefill %.2fs (%u tok, TTFT %.2fs) | decode %u tok in %.2fs = %.2f t/s | wall %.2fs | stop_reason: %d",
                   load_sec, prefill_s, prompt_tokens, ttft_s, sctx.token_count, decode_s, decode_tps, wall_s, (int)stop_reason);
        if (!generated) {
            MYDS_LOG_E("Generation failed (stop_reason: %d)", (int)stop_reason);
            exit_code = 1;
        }
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
            cli_stream_ctx_t sctx = {
                .token_count = 0,
                .t_start = myds_now_s(),
                .t_first = 0.0,
                .t_last = 0.0,
            };

            myds_stop_reason_t stop_reason = MYDS_STOP_EOS;
            uint32_t prompt_tokens = 0;
            int generated = myds_generate_bytes_ex(ctx, line, token_stream_printer, &sctx,
                                                   &stop_reason, &prompt_tokens, NULL);

            double t_end = myds_now_s();
            double wall_s = t_end - sctx.t_start;
            double prefill_s = (sctx.token_count > 0 && sctx.t_first > sctx.t_start) ? (sctx.t_first - sctx.t_start) : wall_s;
            double decode_s = (sctx.token_count > 1 && sctx.t_last > sctx.t_first) ? (sctx.t_last - sctx.t_first) : 0.0;
            double decode_tps = (decode_s > 0.0 && sctx.token_count > 1) ? ((double)(sctx.token_count - 1) / decode_s) : ((wall_s > 0.0) ? ((double)sctx.token_count / wall_s) : 0.0);

            printf("\n  [decode %.1f t/s | TTFT %.2fs | %u prompt tok | %u gen tok | wall %.2fs]\n",
                   decode_tps, prefill_s, prompt_tokens, sctx.token_count, wall_s);
            if (!generated) {
                MYDS_LOG_E("Generation failed (stop_reason: %d)", (int)stop_reason);
                exit_code = 1;
                break;
            }
        }
    }

    myds_context_free(ctx);
    myds_model_free(model);
    return exit_code;
}
