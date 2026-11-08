#include "myds.h"
#include <errno.h>
#include "myds_server.h"
#include "myds_vjob.h"
#include "myds_dspark.h"
#include "myds_steer.h"
#include "myds_steer_alpha.h"
#include "myds_metal.h"
#include "myds_log.h"
#include "myds_configfile.h"
#include "myds_knobs.h"
#include "myds_debugstate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

/* ── 2026-11-01：模型加载【之前】的两道门 ─────────────────────────────
 *
 * 背景：原来的顺序是 model_load(148.4 GB) → context_create → server_start(bind)。
 * 端口冲突只有在最后一步才发现，而那时内存已经压进去了 —— 两个实例同时在
 * page in 同一份模型正是 10-18 / 10-22 两次 macOS Wired Panic 的成因。
 * 这里把两道门提到最前，任何一道不过就 exit，分配 0 字节。
 *
 * 为什么是「探测端口」而不是「预绑定」：真要预绑定要把 listen fd 一路传进
 * server_start，改动面大且容易和现有 accept 循环打架。探测 + 单实例锁已经
 * 覆盖了「误起第二个」这个唯一真实场景；纯 TOCTOU 竞态在这里不构成风险，
 * 因为触发它需要两个进程在同一毫秒内启动。
 */
#define MYDS_GPU_LOCK_PATH "/tmp/myds-gpu.lock"
#define MYDS_INSTANCE_LOCK_PATH "/tmp/myds-instance.lock"

static int g_instance_lock_fd = -1;

/* 单实例锁。返回 0 = 拿到，-1 = 已有实例。
 * 每个 --port 各一把锁，不同端口的实例互不干扰（8187 与 8181 是合法共存的）。 */
static int take_instance_lock(int port) {
    static char path[128];
    if (port > 0) snprintf(path, sizeof(path), "%s-%d", MYDS_INSTANCE_LOCK_PATH, port);
    else snprintf(path, sizeof(path), "%s", MYDS_INSTANCE_LOCK_PATH);
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        MYDS_LOG_W("instance lock: 无法打开 %s (%s)，跳过单实例检查", path, strerror(errno));
        return 0;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return -1;
    }
    g_instance_lock_fd = fd;   /* 故意不释放：进程退出时内核自动解 */
    return 0;
}

/* 配置门禁：环境里存在【没有任何代码读取】的 MYDS_* 键 ⇒ 直接拒绝启动。
 *
 * 为什么要在正常启动路径上做，而不是只在 --print-config 里：MYDS_TOPK_THRESHOLD=1
 * 曾在生产 conf 里躺了 4 天，配置声称开、实际全关，而这类错误不会报错、只会安静地
 * 改掉对照实验的前提。只在人工跑 --print-config 时才检查，等于指望人记得跑。
 *
 * 退出码用 2（与 --print-config 一致），MYDS_STRICT_CONFIG=0 可显式降级为告警。
 */
static int verify_config_gate(void) {
    static myds_knob_seen_t seen[1024];
    const int n = myds_knobs_from_environ(seen, sizeof(seen) / sizeof(seen[0]));
    if (n < 0) {
        MYDS_LOG_E("config: MYDS_* 变量过多，无法核对（上限 %zu）", sizeof(seen) / sizeof(seen[0]));
        return 2;
    }
    int unread = 0;
    for (int i = 0; i < n; ++i) if (!seen[i].knob) ++unread;
    if (!unread) return 0;

    const char *soft = getenv("MYDS_STRICT_CONFIG");
    const bool strict = !(soft && *soft && atoi(soft) == 0);
    for (int i = 0; i < n; ++i) {
        if (seen[i].knob) continue;
        if (strict)
            MYDS_LOG_E("config: %s=%s 没有任何代码读取（拼写错误？已被删除的开关？）",
                       seen[i].name, seen[i].value);
        else
            MYDS_LOG_W("config: %s=%s 没有任何代码读取（MYDS_STRICT_CONFIG=0，仅告警）",
                       seen[i].name, seen[i].value);
    }
    return strict ? 2 : 0;
}

/* 端口是否已被占用（尝试 connect）。返回 1 = 占用。 */
static int port_in_use(const char *host, int port) {
    if (port <= 0) return 0;
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!host || !*host || strcmp(host, "0.0.0.0") == 0) sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    else if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    const int rc = connect(fd, (struct sockaddr *)&sa, sizeof(sa));
    const int busy = (rc == 0);
    close(fd);
    return busy;
}

static void print_usage(const char *prog) {
    printf("myds - Minimal DeepSeek V4 / V4.1 Inference Engine on Apple Silicon Metal\n\n");
    printf("Usage:\n");
    printf("  %s -m <model.gguf> [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>      Path to DeepSeek V4/V4.1 GGUF model (required)\n");
    printf("  --port <port>           Run HTTP server on specified port\n");
    printf("  --host <host>           Host to listen on (default: 127.0.0.1; no auth, use a proxied 0.0.0.0 only)\n");
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
    printf("  --steer-vector <path>   Path to GLP control vector GGUF file\n");
    printf("  --steer-alpha <float>   Steering strength multiplier (default: from file)\n");
    printf("  --steer-mode <auto|v2|v3> Steering mode (auto, v2=residual_only, v3=dual)\n");
    printf("  --steer-layers <range>  Steering active layer range (e.g. 11-37, default: all in file)\n");
    printf("  --steer-enable [0|1]    Enable or disable steering (default: 1 if vector provided)\n");
    printf("  --no-steer              Disable steering\n");
    printf("  --pld                   Enable Prompt-Lookup Decoding (speculative draft)\n");
    printf("  --no-pld                Disable Prompt-Lookup Decoding\n");
    printf("  -h, --help              Show this help message\n");
    printf("  --print-config [-v]     Report which MYDS_* keys this build reads;\n");
    printf("                          exit 2 if any set key is read by nothing\n");
    printf("  --dump-config <path>    Write the effective MYDS_* environment to a file\n");
    printf("  --knob-fingerprint      Print this build's knob-name set fingerprint\n");
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
/* 2026-10-24：把「配置文件里的键」读成 cfg 的初值。
 *
 * 为什么不直接 getenv：配置文件（--config）在 load_config_early 里先加载，
 * 它是【唯一来源】——无条件覆盖进程环境里的同名项（见 myds_configfile.c 的说明）。
 * 这两个辅助只负责「取值 + 校验 + 越界时明确告警」，不做任何写入。
 *
 * 为什么需要它们：CTX / MAX_OUT / TOP_K / MIN_P / REP_PEN 此前只能来自命令行，
 * 配置文件里写不了（没有读取点），于是「配置单一来源」对这几项是空的。
 * 命令行在下面稍后解析，仍然覆盖这里 —— 与既定的优先级一致。 */
static uint32_t cfg_u32_env(const char *key, uint32_t def, uint32_t lo, uint32_t hi) {
    const char *e = getenv(key);
    if (!e || !*e) return def;
    char *end = NULL;
    errno = 0;
    const long long v = strtoll(e, &end, 10);
    if (errno || end == e || (end && *end) || v < (long long)lo || v > (long long)hi) {
        MYDS_LOG_W("config: %s=%s 非法（需 %u..%u）⇒ 用默认 %u", key, e, lo, hi, def);
        return def;
    }
    return (uint32_t)v;
}

static float cfg_f32_env(const char *key, float def, float lo, float hi) {
    const char *e = getenv(key);
    if (!e || !*e) return def;
    char *end = NULL;
    errno = 0;
    const double v = strtod(e, &end);
    if (errno || end == e || (end && *end) || v < (double)lo || v > (double)hi) {
        MYDS_LOG_W("config: %s=%s 非法（需 %.3g..%.3g）⇒ 用默认 %.3g", key, e, (double)lo, (double)hi, (double)def);
        return def;
    }
    return (float)v;
}

static int load_config_early(int argc, char *argv[]) {
    /* ★ 2026-10-25：--config 可重复，按出现顺序叠加（后者覆盖前者）。
     *
     * 为什么不满足于单文件：近似开关（如 MYDS_YOCO_COMPACT）必须按【服务】区分，
     * 而配置文件是 MYDS_* 的唯一来源（下面 override_env 的那段说明）。
     * 如果只能给一个文件，想让一个臂开近似就得整份复制 myds.conf ——
     * 两份全量配置必然慢慢漂移，而漂移的后果是“对比实验的变量不止一个”。
     * 允许叠加后，覆盖文件可以只写差异行：
     *     --config myds.conf --config myds.conf.non0ulp
     * 主文件仍是唯一全量真源，覆盖文件只表达本臂的偏离。
     *
     * 注意：这里刻意【不】因为某个文件读失败就中止 —— 缺文件时打警告并继续，
     * 与下面候选路径的宽容度保持一致；真正需要硬失败的是“一个都没读到”。 */
    int loaded = 0;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--config") != 0) continue;
        if (myds_configfile_load(argv[i + 1], true) > 0) loaded++;
        i++;   /* 跳过路径值，避免把它误当成下一个 flag */
    }
    if (loaded > 0) return loaded;

    const char *path = NULL;
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

/* §165：退出时是否拆除 Metal 驱动层。
 *
 * 默认 0（不拆）。原因：2026-11-08 实测，即使心跳线程已经停掉，
 * `myds_metal_cleanup()` 走到 `_MTLCommandQueue dealloc → -[IOGPUMetalDevice _purgeDevice]
 * → IOGPUResourceListDestroy` 时仍会 EXC_BAD_ACCESS（6 次退出崩 1 次，
 * 报告 myds-2026-11-08-120125.ips）。这是驱动层内部的释放时序问题，
 * 我们改不了；但可以【选择不进入】它 —— 进程马上就结束了，内核会回收。
 *
 * 这与 §153 的取舍一致：宁可泄漏，也不要在关机时序上争。
 * =1 恢复旧行为（完整 teardown），只用于复现/对照，不给生产用。
 *
 * 注意：一旦跳过 teardown，必须用 `_exit()` 而不是 `return`，
 * 否则 atexit / __cxa_finalize / CoreAnalytics 仍会跑，把竞态又请回来。 */
static int exit_metal_teardown_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("MYDS_EXIT_METAL_TEARDOWN");
        cached = (e && *e && atoi(e) != 0) ? 1 : 0;
    }
    return cached;
}

/* 统一的正常退出：停心跳 → 释放自己的内存 → 有选择地拆驱动层。 */
static void exit_after_cleanup(int rc, int free_ctx,
                              myds_context_t *ctx, myds_model_t *model) {
    myds_metal_residency_stop_at_exit();     /* §164.6：心跳必须先停 */
    if (free_ctx) {
        if (ctx) myds_context_free(ctx);
        if (model) myds_model_free(model);
    }
    if (exit_metal_teardown_enabled()) {
        myds_metal_cleanup();
        return;                              /* 让 main 正常 return，保留 atexit */
    }
    fflush(NULL);
    _exit(rc);                               /* §165：跳过驱动层拆除与 atexit */
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
    /* --print-config：不加载模型，回答「配置写的键，到底有没有代码在读」。
     * 存在的理由：MYDS_TOPK_THRESHOLD=1 在生产 conf 里躺了 4 天而无人读取，
     * 配置声称开、实际全关。这类错误不会报错，只会安静地改掉对照实验的前提。
     * 未知键一律 exit 2，让它进 CI 门而不是进复盘文档。 */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--knob-fingerprint") == 0) {
            /* 只打印指纹，不加载模型、不读 conf。部署脚本用它判断
             * 「这份 conf 里的键，这个二进制是否都认识」。 */
            printf("%016llx\n", (unsigned long long)myds_knobs_fingerprint());
            return 0;
        }
        if (strcmp(argv[i], "--print-config") != 0) continue;
        bool verbose = false;
        for (int j = 1; j < argc; j++)
            if (!strcmp(argv[j], "--verbose") || !strcmp(argv[j], "-v")) verbose = true;
        const int unread = myds_knobs_report(1u | 2u | (verbose ? 4u : 0u), verbose);
        return unread < 0 ? 1 : (unread ? 2 : 0);
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
    /* 2026-10-25: 默认只监听回环。服务端没有任何认证，绑定 0.0.0.0 等于
     * 把推理接口连同模型暴露给整个网段。要对外提供服务需显式 --host 0.0.0.0，
     * 并在前面放一层带鉴权的反代。 */
    const char *host = "127.0.0.1";
    const char *alias = "myds";
    const char *vision_path = NULL;
    const char *dspark_path = NULL;
    uint32_t dspark_draft = 0;
    int port = 0;
    bool print_tokens = false;

    /* 初值来自配置文件（键可写入 myds.conf）；随后命令行覆盖。
     * 各键的合法区间按采样语义收紧，越界一律回退默认并告警 ——
     * 静默夹取会让「配置写了 5% 实际按别的值采样」这类问题无法发现。 */
    myds_config_t cfg = {
        .max_seq_len        = cfg_u32_env("MYDS_CTX_SIZE", 4096u, 256u, 4194304u),
        .max_new_tokens     = cfg_u32_env("MYDS_MAX_NEW_TOKENS", 256u, 1u, 4194304u),
        .temperature        = cfg_f32_env("MYDS_TEMPERATURE", 0.7f, 0.0f, 4.0f),
        .top_p              = cfg_f32_env("MYDS_TOP_P", 0.9f, 0.0f, 1.0f),
        .top_k              = cfg_u32_env("MYDS_TOP_K", 40u, 0u, 1048576u),
        .min_p              = cfg_f32_env("MYDS_MIN_P", 0.0f, 0.0f, 1.0f),
        .repetition_penalty = cfg_f32_env("MYDS_REPETITION_PENALTY", 1.0f, 0.0f, 4.0f),
        .presence_penalty   = cfg_f32_env("MYDS_PRESENCE_PENALTY", 0.0f, -2.0f, 2.0f),
        .seed               = (int32_t)time(NULL),
        .stream             = true,
    };

    const char *steer_path = getenv("MYDS_STEER_PATH");
    float steer_alpha = -1.0f;          /* CLI --steer-alpha 的总覆盖（优先于 env） */
    const char *steer_alpha_env = getenv("MYDS_STEER_ALPHA");
    myds_steer_mode_t steer_mode = MYDS_STEER_MODE_AUTO;
    const char *steer_mode_env = getenv("MYDS_STEER_MODE");
    if (steer_mode_env) {
        if (!strcmp(steer_mode_env, "v2") || !strcmp(steer_mode_env, "residual_only")) steer_mode = MYDS_STEER_MODE_V2_RESIDUAL_ONLY;
        else if (!strcmp(steer_mode_env, "v3") || !strcmp(steer_mode_env, "dual")) steer_mode = MYDS_STEER_MODE_V3_DUAL;
    }
    const char *steer_layers = getenv("MYDS_STEER_LAYERS");
    int steer_enable = 1;
    const char *steer_enable_env = getenv("MYDS_STEER_ENABLE");
    if (steer_enable_env) {
        steer_enable = atoi(steer_enable_env);
    }

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
        } else if (strcmp(argv[i], "--steer-vector") == 0 && i + 1 < argc) {
            steer_path = argv[++i];
        } else if (strcmp(argv[i], "--steer-alpha") == 0 && i + 1 < argc) {
            steer_alpha = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--steer-mode") == 0 && i + 1 < argc) {
            const char *mstr = argv[++i];
            if (!strcmp(mstr, "v2") || !strcmp(mstr, "residual_only")) steer_mode = MYDS_STEER_MODE_V2_RESIDUAL_ONLY;
            else if (!strcmp(mstr, "v3") || !strcmp(mstr, "dual")) steer_mode = MYDS_STEER_MODE_V3_DUAL;
            else steer_mode = MYDS_STEER_MODE_AUTO;
        } else if (strcmp(argv[i], "--steer-layers") == 0 && i + 1 < argc) {
            steer_layers = argv[++i];
        } else if (strcmp(argv[i], "--steer-enable") == 0 && i + 1 < argc) {
            steer_enable = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-steer") == 0) {
            steer_enable = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            i++;   /* 已在 load_config_early 里消费过，这里只跳过它的值 */
        } else {
            /* ★ 2026-11-01：原来这里没有任何兜底分支，未识别的参数被静默忽略。
             * 实测后果：`--temperature 0` / `--max_tokens 512` / `--ctx_size 8192`
             * 三个「看起来对」的写法都不生效，而进程正常启动、日志只显示代码默认值。
             * 这种失败比崩溃难查得多 —— 崩溃至少有人看。 */
            MYDS_LOG_E("未知命令行参数: %s", argv[i]);
            if (argv[i][0] == '-' && argv[i][1] == '-') {
                MYDS_LOG_E("（注意：采样参数走配置文件或 MYDS_* 环境变量；"
                           "命令行只认 --temp/--top-p/--max-tokens/--ctx-size 这几个写法）");
            }
            print_usage(argv[0]);
            return 2;
        }
    }

    /* ── 模型加载前的三道门（任何一道不过 ⇒ 0 字节 GPU 分配就退出）──
     * 顺序有讲究：先验配置（不碰任何资源），再验端口（不碰 GPU），最后取锁。
     * 反过来会让一次配置拼写错误白等 40 秒的模型加载。 */
    {
        const int cfg_rc = verify_config_gate();
        if (cfg_rc != 0) {
            MYDS_LOG_E("拒绝启动：配置门禁未通过（退出码 %d）", cfg_rc);
            return cfg_rc;
        }
        const char *allow_multi = getenv("MYDS_ALLOW_MULTI_INSTANCE");
        const bool multi_ok = allow_multi && *allow_multi && atoi(allow_multi) != 0;
        if (!multi_ok) {
            if (port_in_use(host, port)) {
                MYDS_LOG_E("拒绝启动：端口 %d 已在监听。另一个实例可能正在跑。", port);
                MYDS_LOG_E("  已加载模型再发现端口冲突会让两个进程同时 page in 同一份权重 ——");
                MYDS_LOG_E("  那正是 10-18 / 10-22 两次 macOS Wired Panic 的成因。");
                MYDS_LOG_E("  确认无实例后重试；确需多实例（如轻量单测）设 MYDS_ALLOW_MULTI_INSTANCE=1。");
                return 3;
            }
            if (take_instance_lock(port) != 0) {
                MYDS_LOG_E("拒绝启动：端口 %d 的单实例锁已被占用（另一个 myds 正在启动或运行）。", port);
                MYDS_LOG_E("  确需多实例设 MYDS_ALLOW_MULTI_INSTANCE=1。");
                return 3;
            }
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

    /* 2026-10-24：把全部采样参数打出来。
     * 此前只打 ctx/max_new/temp/top_p，top_k / min_p / repetition_penalty
     * 即使配了也无从确认 —— 「配了但不生效」正是这类配置最难发现的地方。 */
    MYDS_LOG_I("Starting myds (ctx_size=%u, max_new_tokens=%u, temp=%.2f, top_p=%.2f, "
               "top_k=%u, min_p=%.3f, rep_pen=%.3f, pres_pen=%.3f, thinking=%s)",
               cfg.max_seq_len, cfg.max_new_tokens, cfg.temperature, cfg.top_p,
               cfg.top_k, (double)cfg.min_p, (double)cfg.repetition_penalty,
               (double)cfg.presence_penalty, cfg.thinking_mode ? "on" : "off");
    MYDS_LOG_I("Loading model file: %s", model_path);

    double load_t0 = myds_now_s();
    myds_model_t *model = myds_model_load(model_path);
    if (!model) {
        MYDS_LOG_E("Failed to load model from %s", model_path);
        return 1;
    }
    /* §164.6 兜底：无论从哪条路径退出，都要先停 residency 心跳。
     * atexit 是 LIFO —— 我们比系统框架晚注册，所以【先】于它们执行，
     * 正好抢在 CoreAnalytics 的 atExitHandler 之前把线程收掉。
     * 正常路径已经显式调用过，本函数幂等。 */
    atexit(myds_metal_residency_stop_at_exit);

    double load_t1 = myds_now_s();
    double load_sec = load_t1 - load_t0;
    MYDS_LOG_I("Model loaded successfully in %.2fs", load_sec);

    myds_model_print_info(model);

    myds_context_t *ctx = myds_context_create(model, &cfg);
    if (!ctx) {
        MYDS_LOG_E("Failed to create inference context");
        exit_after_cleanup(1, 0, NULL, model);
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

    /* GLP 控制向量 / 隐空间消融子系统 */
    if (steer_path && *steer_path) {
        char path_buf[1024];
        strncpy(path_buf, steer_path, sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
        char *saveptr = NULL;
        char *p = strtok_r(path_buf, ",", &saveptr);
        while (p) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p) {
                /* 优先级：CLI --steer-alpha > MYDS_STEER_ALPHA > profile 文件默认 */
                float a = (steer_alpha >= 0.0f) ? steer_alpha
                                                : myds_steer_alpha_for(steer_alpha_env, p);
                myds_steer_t *st = myds_steer_load(p, a, steer_mode, steer_layers);
                if (!st) {
                    MYDS_LOG_E("GLP 控制向量加载失败（%s）", p);
                } else {
                    st->enabled = (steer_enable != 0);
                    myds_context_add_steer(ctx, st);
                    MYDS_LOG_I("GLP 控制向量已注册: [%s] (enabled=%d, alpha=%.3f, mode=%d)",
                               st->name, st->enabled ? 1 : 0, st->alpha_effective, st->mode);
                }
            }
            p = strtok_r(NULL, ",", &saveptr);
        }
    }

    if (dspark_path) {
        ctx->dspark = myds_dspark_load(dspark_path, model->vocab_size);
        if (!ctx->dspark) {
            MYDS_LOG_E("Failed to load DSpark support model from %s", dspark_path);
            if (capture_mode != 0) {
                exit_after_cleanup(1, 1, ctx, model);
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
        exit_after_cleanup(1, 1, ctx, model);
        return 1;
    }

    if (port > 0) {
        /* HTTP Server Mode for Fleet deployment */
        /* Register /debug/state. Kept out of myds_server.c so that file does not
         * depend on the knob inventory, the GPU counters or the Metal library
         * path; see include/myds_debugstate.h. */
        myds_server_set_debug_provider(myds_debug_state_json, myds_debug_enabled,
                                       myds_debug_is_loopback, myds_debug_token_ok);
        int rc = myds_server_start(ctx, host, port, alias);
        /* rc == 2 表示关机时还有请求线程没退完（见 myds_server_start 的 drain 段）。
         * 那种情况下 ctx/model 里仍有线程在读写，绝不能 free —— 交给进程退出回收。
         * 这是 2026-11-07 定位的那个崩溃：detach 的客户端线程 + 立即 free ctx。 */
        /* §164.6 + §165：停心跳、按需释放、有选择地拆驱动层。
         * rc==2 时不能 free ctx/model（线程还在用），但【可以且必须】停心跳 ——
         * 它不碰 ctx/model。 */
        if (rc == 2) {
            MYDS_LOG_E("main: request threads still running at shutdown; "
                       "leaving context/model allocated (process exit reclaims)");
            exit_after_cleanup(0, 0, NULL, NULL);
            return 0;
        }
        exit_after_cleanup(rc, 1, ctx, model);
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

    exit_after_cleanup(exit_code, 1, ctx, model);
    return exit_code;
}
