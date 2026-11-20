/* 隔离基准：decode 的 engram 行读取，串行 vs 线程池。
 *
 * 背景（2026-11-07）：生产 decode 每 token 有 949 µs 的 GPU 空隙，
 * 其中 73% 是 engram。查代码发现 prepare_batch_step 的并行池分支条件是
 * `count == 1` 就走串行 —— 而 decode 每次正好只有 1 个 token。
 * 也就是说这个池只服务 prefill，decode 从来没并行过。
 *
 * 这个程序回答一个问题：把那 48 次随机读并行起来，到底省多少。
 * 不加载模型、不跑推理，只按真实偏移随机读行。
 *
 * 2026-11-20（[D-E1] 硬件门控）：支持将基准池上限放宽至 48（不改动生产 src/myds_engram.c
 * 的 16 线程上限），并支持逗号分隔的 QD 扫描（如 16,24,48）：
 *   - pool <= 16：直接走生产 myds_engram_reader_create / read_token
 *   - 17 <= pool <= 48：走本基准内置的同构持久化线程池（offset 排序 pread + 生产 decode_row）
 *
 * 用法：
 *   bench_engram_rows <model.gguf> [rounds=200] [pool=8|16,24,48] [hot|decode]
 *   bench_engram_rows --selftest   （无模型自测：验证 QD=16/24/48 0-ULP 一致性）
 * 输出：CSV，便于和其他基准一起归档。
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_gguf.h"
#include "myds_engram.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    BENCH_ENGRAM_PROD_MAX_THREADS = 16,
    BENCH_ENGRAM_MAX_THREADS      = 48,
    BENCH_ENGRAM_JOBS             = MYDS_ENGRAM_LAYERS * MYDS_ENGRAM_COLS
};

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ── Bench-only wide thread pool (17..48 workers, mirrors src/myds_engram.c) ── */
typedef struct {
    int fd, error;
    const uint8_t *map;
    uint64_t offset;
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
} bench_io_job_t;

struct bench_engram_pool;
typedef struct {
    struct bench_engram_pool *pool;
    unsigned index;
} bench_io_worker_t;

typedef struct bench_engram_pool {
    unsigned workers, started, remaining, job_count;
    bool stopping;
    uint64_t generation;
    pthread_mutex_t state;
    pthread_cond_t work, done;
    pthread_t threads[BENCH_ENGRAM_MAX_THREADS];
    bench_io_worker_t worker[BENCH_ENGRAM_MAX_THREADS];
    bench_io_job_t jobs[BENCH_ENGRAM_JOBS];
} bench_engram_pool_t;

static void bench_read_raw_job(bench_io_job_t *job) {
    if (job->map) {
        memcpy(job->raw, job->map + job->offset, sizeof(job->raw));
        job->error = 0;
        return;
    }
    size_t done = 0;
    while (done < sizeof(job->raw)) {
        ssize_t n = pread(job->fd, job->raw + done, sizeof(job->raw) - done,
                          (off_t)(job->offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { job->error = (n == 0) ? EIO : errno; return; }
        done += (size_t)n;
    }
    job->error = 0;
}

static void *bench_engram_worker(void *opaque) {
    bench_io_worker_t *w = (bench_io_worker_t *)opaque;
    bench_engram_pool_t *p = w->pool;
    uint64_t seen = 0;
    pthread_mutex_lock(&p->state);
    for (;;) {
        while (!p->stopping && seen == p->generation)
            pthread_cond_wait(&p->work, &p->state);
        if (p->stopping) break;
        seen = p->generation;
        const unsigned count = p->job_count;
        pthread_mutex_unlock(&p->state);

        unsigned my_jobs[BENCH_ENGRAM_JOBS];
        unsigned n_my = 0;
        for (unsigned i = w->index; i < count; i += p->workers)
            my_jobs[n_my++] = i;
        for (unsigned i = 0; i < n_my; ++i) {
            for (unsigned j = i + 1; j < n_my; ++j) {
                if (p->jobs[my_jobs[j]].offset < p->jobs[my_jobs[i]].offset) {
                    unsigned tmp = my_jobs[i];
                    my_jobs[i] = my_jobs[j];
                    my_jobs[j] = tmp;
                }
            }
        }
        for (unsigned i = 0; i < n_my; ++i)
            bench_read_raw_job(&p->jobs[my_jobs[i]]);

        pthread_mutex_lock(&p->state);
        if (--p->remaining == 0) pthread_cond_signal(&p->done);
    }
    pthread_mutex_unlock(&p->state);
    return NULL;
}

static bench_engram_pool_t *bench_engram_pool_create(unsigned workers) {
    if (workers < 2 || workers > BENCH_ENGRAM_MAX_THREADS) {
        errno = EINVAL;
        return NULL;
    }
    bench_engram_pool_t *p = (bench_engram_pool_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->workers = workers;
    if (pthread_mutex_init(&p->state, NULL) != 0) { free(p); return NULL; }
    if (pthread_cond_init(&p->work, NULL) != 0) { pthread_mutex_destroy(&p->state); free(p); return NULL; }
    if (pthread_cond_init(&p->done, NULL) != 0) {
        pthread_cond_destroy(&p->work);
        pthread_mutex_destroy(&p->state);
        free(p);
        return NULL;
    }
    for (unsigned i = 0; i < workers; ++i) {
        p->worker[i] = (bench_io_worker_t){ .pool = p, .index = i };
        if (pthread_create(&p->threads[i], NULL, bench_engram_worker, &p->worker[i]) != 0) {
            pthread_mutex_lock(&p->state);
            p->stopping = true;
            pthread_cond_broadcast(&p->work);
            pthread_mutex_unlock(&p->state);
            for (unsigned j = 0; j < p->started; ++j) pthread_join(p->threads[j], NULL);
            pthread_cond_destroy(&p->done);
            pthread_cond_destroy(&p->work);
            pthread_mutex_destroy(&p->state);
            free(p);
            return NULL;
        }
        ++p->started;
    }
    return p;
}

static void bench_engram_pool_free(bench_engram_pool_t *p) {
    if (!p) return;
    pthread_mutex_lock(&p->state);
    p->stopping = true;
    pthread_cond_broadcast(&p->work);
    pthread_mutex_unlock(&p->state);
    for (unsigned i = 0; i < p->started; ++i) pthread_join(p->threads[i], NULL);
    pthread_cond_destroy(&p->done);
    pthread_cond_destroy(&p->work);
    pthread_mutex_destroy(&p->state);
    free(p);
}

static bool bench_engram_pool_read_token(
    bench_engram_pool_t *p,
    const myds_engram_table_t tables[MYDS_ENGRAM_LAYERS],
    const uint32_t rows[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS],
    float out[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM])
{
    unsigned map[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];
    unsigned jobs = 0;
    for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer) {
        for (int col = 0; col < MYDS_ENGRAM_COLS; ++col) {
            int previous = 0;
            while (previous < col && rows[layer][previous] != rows[layer][col]) ++previous;
            if (previous < col) { map[layer][col] = map[layer][previous]; continue; }
            map[layer][col] = jobs;
            p->jobs[jobs].fd = tables[layer].fd;
            p->jobs[jobs].map = tables[layer].map;
            p->jobs[jobs].offset = tables[layer].offset +
                                   (uint64_t)rows[layer][col] * MYDS_ENGRAM_ROW_BYTES;
            p->jobs[jobs].error = 0;
            ++jobs;
        }
    }
    pthread_mutex_lock(&p->state);
    p->job_count = jobs;
    p->remaining = p->workers;
    ++p->generation;
    pthread_cond_broadcast(&p->work);
    while (p->remaining) pthread_cond_wait(&p->done, &p->state);
    pthread_mutex_unlock(&p->state);

    for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer) {
        for (int col = 0; col < MYDS_ENGRAM_COLS; ++col) {
            bench_io_job_t *job = &p->jobs[map[layer][col]];
            if (job->error) { errno = job->error; return false; }
            int previous = 0;
            while (previous < col && map[layer][previous] != map[layer][col]) ++previous;
            if (previous < col) {
                memcpy(out[layer] + col * MYDS_ENGRAM_DIM,
                       out[layer] + previous * MYDS_ENGRAM_DIM,
                       MYDS_ENGRAM_DIM * sizeof(float));
            } else if (!myds_engram_decode_row(job->raw, out[layer] + col * MYDS_ENGRAM_DIM)) {
                return false;
            }
        }
    }
    return true;
}

/* Unified bench reader handle: uses production reader for <=16, bench pool for 17..48. */
typedef struct {
    unsigned workers;
    myds_engram_reader_t *prod;
    bench_engram_pool_t  *wide;
} bench_reader_t;

static bool bench_reader_init(bench_reader_t *br, unsigned workers) {
    memset(br, 0, sizeof(*br));
    br->workers = workers;
    if (workers <= 1) return true;
    if (workers <= BENCH_ENGRAM_PROD_MAX_THREADS) {
        br->prod = myds_engram_reader_create(workers);
        return br->prod != NULL;
    }
    if (workers <= BENCH_ENGRAM_MAX_THREADS) {
        br->wide = bench_engram_pool_create(workers);
        return br->wide != NULL;
    }
    errno = EINVAL;
    return false;
}

static void bench_reader_free(bench_reader_t *br) {
    if (br->prod) myds_engram_reader_free(br->prod);
    if (br->wide) bench_engram_pool_free(br->wide);
    memset(br, 0, sizeof(*br));
}

static bool bench_reader_read_token(
    bench_reader_t *br,
    const myds_engram_table_t tables[MYDS_ENGRAM_LAYERS],
    const uint32_t rows[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS],
    float out[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM])
{
    if (br->prod) return myds_engram_reader_read_token(br->prod, tables, rows, out);
    if (br->wide) return bench_engram_pool_read_token(br->wide, tables, rows, out);
    for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l)
        if (!myds_engram_read(&tables[l], rows[l], MYDS_ENGRAM_COLS, out[l]))
            return false;
    return true;
}

static int parse_pool_list(const char *arg, unsigned out_pools[16], int *out_count) {
    *out_count = 0;
    const char *p = arg;
    while (*p) {
        if (*out_count >= 16) return 0;
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p || v < 1ul || v > BENCH_ENGRAM_MAX_THREADS) return 0;
        out_pools[(*out_count)++] = (unsigned)v;
        if (*end == ',') p = end + 1;
        else if (*end == '\0') break;
        else return 0;
    }
    return *out_count > 0;
}

static int run_selftest(void) {
    char tmpl[] = "/tmp/bench_engram_rows_selftest_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) { perror("mkstemp"); return 1; }
    unlink(tmpl);

    const uint32_t n_rows = 256u;
    uint8_t row_buf[MYDS_ENGRAM_ROW_BYTES];
    for (uint32_t r = 0; r < n_rows * 2u; ++r) {
        for (int i = 0; i < MYDS_ENGRAM_DIM; ++i)
            row_buf[i] = (uint8_t)((r * 13u + (uint32_t)i * 7u) % 127u);
        for (int s = 0; s < MYDS_ENGRAM_DIM / 32; ++s)
            row_buf[MYDS_ENGRAM_DIM + s] = (uint8_t)(120u + ((r + (uint32_t)s) % 30u));
        if (write(fd, row_buf, sizeof(row_buf)) != (ssize_t)sizeof(row_buf)) {
            perror("write");
            close(fd);
            return 1;
        }
    }

    myds_engram_table_t t[MYDS_ENGRAM_LAYERS];
    memset(t, 0, sizeof(t));
    if (!myds_engram_table_open_fd(&t[0], fd, 0, n_rows) ||
        !myds_engram_table_open_fd(&t[1], fd, (uint64_t)n_rows * MYDS_ENGRAM_ROW_BYTES, n_rows)) {
        fprintf(stderr, "selftest: table open failed\n");
        close(fd);
        return 1;
    }
    close(fd);

    const unsigned sweep[] = { 16u, 24u, 48u };
    uint32_t rows[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];
    float ref[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM];
    float got[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM];

    srand(20261120);
    for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l) {
        for (int c = 0; c < MYDS_ENGRAM_COLS; ++c)
            rows[l][c] = (uint32_t)(rand() % (int)n_rows);
        /* Exercise duplicate-row dedup inside a token too */
        rows[l][5] = rows[l][2];
    }
    for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l) {
        if (!myds_engram_read(&t[l], rows[l], MYDS_ENGRAM_COLS, ref[l])) {
            fprintf(stderr, "selftest: serial read failed\n");
            return 1;
        }
    }

    for (size_t i = 0; i < sizeof(sweep) / sizeof(sweep[0]); ++i) {
        bench_reader_t br;
        if (!bench_reader_init(&br, sweep[i])) {
            fprintf(stderr, "selftest: reader init failed for QD=%u\n", sweep[i]);
            return 1;
        }
        memset(got, 0x55, sizeof(got));
        if (!bench_reader_read_token(&br, t, (const uint32_t (*)[MYDS_ENGRAM_COLS])rows, got)) {
            fprintf(stderr, "selftest: read failed for QD=%u\n", sweep[i]);
            bench_reader_free(&br);
            return 1;
        }
        if (memcmp(ref, got, sizeof(ref)) != 0) {
            fprintf(stderr, "selftest: 0-ULP mismatch for QD=%u\n", sweep[i]);
            bench_reader_free(&br);
            return 1;
        }
        printf("[selftest PASS] QD=%-2u (%s): 48-row read memcmp==0\n",
               sweep[i], br.prod ? "prod_pool" : "bench_wide_pool");
        bench_reader_free(&br);
    }

    for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l) myds_engram_table_close(&t[l]);
    printf("PASS: bench_engram_rows selftest passed for QD=16, 24, 48.\n");
    return 0;
}

/* 真实的两个 engram 表在 blk.1 / blk.14。取法照抄 src/myds_engram.c:713 附近。 */
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) {
        return run_selftest();
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [rounds=200] [pool=8|16,24,48] [hot|decode]\n"
                        "       %s --selftest\n", argv[0], argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const int rounds = argc > 2 ? atoi(argv[2]) : 200;
    unsigned pools[16];
    int n_pools = 0;
    const char *pool_arg = argc > 3 ? argv[3] : "8";
    if (!parse_pool_list(pool_arg, pools, &n_pools)) {
        fprintf(stderr, "invalid pool spec '%s' (expected 1..%d or comma list e.g. 16,24,48)\n",
                pool_arg, BENCH_ENGRAM_MAX_THREADS);
        return 2;
    }

    gguf_file_t *gf = gguf_open(model);
    if (!gf) { fprintf(stderr, "cannot open %s\n", model); return 1; }

    static const char *names[MYDS_ENGRAM_LAYERS] = {
        "blk.1.engram_embd.weight", "blk.14.engram_embd.weight"
    };
    myds_engram_table_t t[MYDS_ENGRAM_LAYERS];
    memset(t, 0, sizeof(t));
    for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l) {
        const gguf_tensor_t *tn = gguf_find_tensor(gf, names[l]);
        if (!tn) { fprintf(stderr, "missing %s\n", names[l]); return 1; }
        const uint32_t rows = (uint32_t)tn->dims[1];
        /* 与生产同一条路：gguf_tensor_source 给出正确的分片 fd 与文件内绝对偏移，
         * 再用 open_fd（即 pread），不是 mmap。基准必须复现生产口径。 */
        int src_fd = -1;
        uint64_t src_offset = 0;
        if (!gguf_tensor_source(gf, tn, &src_fd, &src_offset)) {
            fprintf(stderr, "tensor_source failed for %s\n", names[l]);
            return 1;
        }
        if (!myds_engram_table_open_fd(&t[l], src_fd, src_offset, rows)) {
            fprintf(stderr, "table open failed for %s\n", names[l]);
            return 1;
        }
        fprintf(stderr, "%s: rows=%u dims=%llux%llu type=%d\n", names[l], rows,
                (unsigned long long)tn->dims[0],
                (unsigned long long)tn->dims[1], (int)tn->type);
    }

    const int hot = argc > 4 && strcmp(argv[4], "hot") == 0;
    const int decode_only = argc > 4 && strcmp(argv[4], "decode") == 0;

    if (decode_only) {
        static uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
        float out[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM];
        for (int i = 0; i < MYDS_ENGRAM_DIM; ++i) {
            raw[i] = (uint8_t)(i % 127);
            raw[MYDS_ENGRAM_DIM + i / 32] = (uint8_t)(128 + (i / 32));
        }
        const int reps = 20000;
        const double t0 = now_us();
        for (int i = 0; i < reps; ++i)
            if (!myds_engram_decode_row(raw, out[0])) { fprintf(stderr, "decode failed\n"); return 1; }
        const double row_us = (now_us() - t0) / reps;
        printf("decode_us_per_ROW=%.3f  (x48 rows = %.1f us/token)\n", row_us, row_us * 48.0);
        for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l) myds_engram_table_close(&t[l]);
        gguf_close(gf);
        return 0;
    }

    printf("mode=%s\n", hot ? "hot(same 48 rows)" : "cold(random 48 rows)");
    printf("rounds,serial_us_per_token,pool_workers,pool_us_per_token,speedup\n");

    float out_ser[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM];
    float out_par[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM];
    uint32_t rows[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];

    for (int pi = 0; pi < n_pools; ++pi) {
        const unsigned pool = pools[pi];
        bench_reader_t br;
        if (!bench_reader_init(&br, pool)) {
            fprintf(stderr, "reader create failed for pool=%u\n", pool);
            return 1;
        }

        srand(20261107 + (unsigned)pi * 97u);
        double ser_us = 0, par_us = 0;
        int ser_n = 0, par_n = 0;
        bool parity_checked = false;

        for (int round = 0; round < rounds; ++round) {
            if (!hot || round == 0) {
                for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l)
                    for (int c = 0; c < MYDS_ENGRAM_COLS; ++c)
                        rows[l][c] = (uint32_t)(((uint64_t)rand() * 2654435761u) % t[l].rows);
            }

            if (round < 4) {
                for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l)
                    if (myds_engram_read(&t[l], rows[l], MYDS_ENGRAM_COLS, out_ser[l])) { }
                continue;
            }

            {
                const double t0 = now_us();
                for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l)
                    if (!myds_engram_read(&t[l], rows[l], MYDS_ENGRAM_COLS, out_ser[l])) {
                        fprintf(stderr, "serial read failed\n");
                        bench_reader_free(&br);
                        return 1;
                    }
                ser_us += now_us() - t0;
                ++ser_n;
            }

            if (pool > 1) {
                const double t0 = now_us();
                if (!bench_reader_read_token(&br, t, (const uint32_t (*)[MYDS_ENGRAM_COLS])rows, out_par)) {
                    fprintf(stderr, "pool read failed (pool=%u)\n", pool);
                    bench_reader_free(&br);
                    return 1;
                }
                par_us += now_us() - t0;
                ++par_n;
                if (!parity_checked) {
                    if (memcmp(out_ser, out_par, sizeof(out_ser)) != 0) {
                        fprintf(stderr, "0-ULP parity mismatch for pool=%u\n", pool);
                        bench_reader_free(&br);
                        return 1;
                    }
                    parity_checked = true;
                }
            }
        }

        if (par_n) {
            printf("%d,%.1f,%u,%.1f,%.2fx\n", ser_n, ser_us / ser_n, pool,
                   par_us / par_n, (ser_us / ser_n) / (par_us / par_n));
        } else {
            printf("%d,%.1f,%u,n/a,n/a\n", ser_n, ser_n ? ser_us / ser_n : 0.0, pool);
        }
        bench_reader_free(&br);
    }

    for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l) myds_engram_table_close(&t[l]);
    gguf_close(gf);
    return 0;
}
