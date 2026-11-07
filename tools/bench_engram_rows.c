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
 * 用法：bench_engram_rows <model.gguf> [rounds=200] [pool=8]
 * 输出：CSV 一行，便于和其他基准一起归档。
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_gguf.h"
#include "myds_engram.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* 真实的两个 engram 表在 blk.1 / blk.14。取法照抄 src/myds_engram.c:713 附近。 */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [rounds=200] [pool=8]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const int rounds = argc > 2 ? atoi(argv[2]) : 200;
    const unsigned pool = argc > 3 ? (unsigned)atoi(argv[3]) : 8u;

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

    myds_engram_reader_t *r = NULL;
    if (pool > 1) {
        r = myds_engram_reader_create(pool);
        if (!r) { fprintf(stderr, "reader create failed\n"); return 1; }
    }

    /* 每次读 48 行（2 层 × 24 列），与 decode 完全同形。
     * 行号用固定种子随机生成：既不是连续行（不会吃到 readahead），
     * 也不可重复（不会命中去重逻辑）。 */
    srand(20261107);
    double ser_us = 0, par_us = 0;
    int ser_n = 0, par_n = 0;
    float out[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM];
    uint32_t rows[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];

    /* 预热：让线程池起来、页表建立，不计时。 */
    for (int round = 0; round < rounds; ++round) {
        for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l)
            for (int c = 0; c < MYDS_ENGRAM_COLS; ++c)
                rows[l][c] = (uint32_t)(((uint64_t)rand() * 2654435761u) % t[l].rows);

        if (round < 4) {
            for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l)
                if (myds_engram_read(&t[l], rows[l], MYDS_ENGRAM_COLS, out[l])) { }
            continue;
        }

        /* 串行（生产的 decode 现状） */
        {
            const double t0 = now_us();
            for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l)
                if (!myds_engram_read(&t[l], rows[l], MYDS_ENGRAM_COLS, out[l])) {
                    fprintf(stderr, "serial read failed\n");
                    return 1;
                }
            ser_us += now_us() - t0;
            ++ser_n;
        }

        /* 并行（池） */
        if (r) {
            const double t0 = now_us();
            if (!myds_engram_reader_read_token(r, t, (const uint32_t (*)[MYDS_ENGRAM_COLS])rows, out)) {
                fprintf(stderr, "pool read failed\n");
                return 1;
            }
            par_us += now_us() - t0;
            ++par_n;
        }
    }

    printf("rounds,serial_us_per_token,pool_workers,pool_us_per_token,speedup\n");
    printf("%d,%.1f,%u,%s,%s\n", ser_n, ser_n ? ser_us / ser_n : 0.0, pool,
           par_n ? "" : "n/a",
           par_n ? "" : "n/a");
    if (par_n) {
        /* 第二次打印带数值的完整行，避免上面的占位符在不同 locale 下解析出错 */
        printf("%d,%.1f,%u,%.1f,%.2fx\n", ser_n, ser_us / ser_n, pool,
               par_us / par_n, (ser_us / ser_n) / (par_us / par_n));
    }

    if (r) myds_engram_reader_free(r);
    for (int l = 0; l < MYDS_ENGRAM_LAYERS; ++l) myds_engram_table_close(&t[l]);
    gguf_close(gf);
    return 0;
}
