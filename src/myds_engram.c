/* Hash and E4M3/E8M0 row semantics adapted from ds4_engram.c and
 * metal/dsv41.metal at 6f873ad7f5940ab8556abc41ae510853118ecc4a.
 *
 * MIT License
 * Copyright (c) 2026 The ds4.c authors
 * Copyright (c) 2023-2026 The ggml authors
 * Copyright (c) 2023 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "myds_engram.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static bool fail(int error) { errno = error; return false; }

bool myds_engram_layout_valid(const myds_engram_layout_t *l) {
    if (!l || !l->token_map || !l->vocab_size || !l->compressed_vocab_size ||
        l->compressed_vocab_size > INT32_MAX || l->pad_id >= l->compressed_vocab_size)
        return false;
    for (uint32_t i = 0; i < l->vocab_size; ++i)
        if (l->token_map[i] >= l->compressed_vocab_size) return false;
    for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer) {
        for (int i = 0; i < MYDS_ENGRAM_NGRAM; ++i) {
            uint64_t multiplier = l->multipliers[layer][i];
            if (!(multiplier & 1) || multiplier > (uint64_t)INT64_MAX / l->compressed_vocab_size)
                return false;
        }
        uint64_t sum = 0;
        for (int i = 0; i < MYDS_ENGRAM_COLS; ++i) {
            if (l->primes[layer][i] < 2) return false;
            sum += l->primes[layer][i];
        }
        if (sum != l->rows[layer]) return false;
    }
    return true;
}

void myds_engram_history_reset(myds_engram_history_t *history) {
    if (history)
        for (int i = 0; i < MYDS_ENGRAM_NGRAM - 1; ++i) history->tail[i] = MYDS_ENGRAM_DEAD;
}

bool myds_engram_hash(const myds_engram_layout_t *l, myds_engram_history_t *h,
                     const int32_t *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows) {
    if (!l || !h || !l->token_map || (count && (!tokens || !rows)) ||
        count > SIZE_MAX / (MYDS_ENGRAM_LAYERS * MYDS_ENGRAM_COLS * sizeof(*rows)))
        return fail(EINVAL);
    for (int i = 0; i < MYDS_ENGRAM_NGRAM - 1; ++i)
        if (h->tail[i] < MYDS_ENGRAM_DEAD ||
            (h->tail[i] >= 0 && (uint32_t)h->tail[i] >= l->compressed_vocab_size))
            return fail(EINVAL);
    /* Validate the entire request before changing the caller's history. */
    for (size_t i = 0; i < count; ++i)
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= l->vocab_size)
            return fail(EINVAL);
    for (size_t i = 0; i < count; ++i) {
        int32_t current = mask && !mask[i] ? MYDS_ENGRAM_DEAD : (int32_t)l->token_map[tokens[i]];
        uint32_t ids[MYDS_ENGRAM_NGRAM];
        bool blocked = false;
        for (int j = 0; j < MYDS_ENGRAM_NGRAM; ++j) {
            int32_t id = j ? h->tail[j - 1] : current;
            blocked |= id == MYDS_ENGRAM_DEAD;
            ids[j] = blocked ? l->pad_id : (uint32_t)id;
        }
        for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer) {
            uint64_t hash = (uint64_t)ids[0] * l->multipliers[layer][0];
            uint32_t offset = 0;
            for (int n = 1; n < MYDS_ENGRAM_NGRAM; ++n) {
                hash ^= (uint64_t)ids[n] * l->multipliers[layer][n];
                for (int head = 0; head < MYDS_ENGRAM_HEADS; ++head) {
                    int col = (n - 1) * MYDS_ENGRAM_HEADS + head;
                    uint32_t prime = l->primes[layer][col];
                    *rows++ = (uint32_t)(hash % prime) + offset;
                    offset += prime;
                }
            }
        }
        for (int j = MYDS_ENGRAM_NGRAM - 2; j > 0; --j) h->tail[j] = h->tail[j - 1];
        h->tail[0] = current;
    }
    return true;
}

bool myds_engram_table_open(myds_engram_table_t *table, const char *path,
                           uint64_t offset, uint32_t rows) {
    if (!table) return fail(EINVAL);
    *table = (myds_engram_table_t){.fd = -1};
    uint64_t bytes = (uint64_t)rows * MYDS_ENGRAM_ROW_BYTES;
    if (!path || !rows || offset > INT64_MAX || bytes > INT64_MAX - offset)
        return fail(EINVAL);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) goto error;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || offset + bytes > (uint64_t)st.st_size) {
        errno = EINVAL;
        goto error;
    }
#ifdef __APPLE__
    if (getenv("MYDS_ENGRAM_NOCACHE")) {
        if (fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0) goto error;
    }
#endif
    *table = (myds_engram_table_t){.fd = fd, .offset = offset, .rows = rows};
    return true;
error: {
        int saved = errno;
        close(fd);
        return fail(saved);
    }
}

/* 2026-10-25 split GGUF：从【已经打开的】gf 分片 fd 派生一个独立的 table fd。
 *
 * 为什么 dup 而不是直接复用 gf->parts[i].fd：table 的关闭语义是「关掉自己这一份」，
 * 直接共用会让 gguf_close 与 engram_close 争相关同一个 fd（double close）。
 * dup 保留同一个文件偏移语义（pread 本来就无状态），成本只是一个 fd。
 *
 * 为什么需要这个变体：拆片后张量分散在多个文件里，路径也不再是"模型路径"，
 * 所以不能再走 path 版本。 */
bool myds_engram_table_open_fd(myds_engram_table_t *table, int src_fd,
                              uint64_t offset, uint32_t rows) {
    if (!table) return fail(EINVAL);
    *table = (myds_engram_table_t){.fd = -1};
    const uint64_t bytes = (uint64_t)rows * MYDS_ENGRAM_ROW_BYTES;
    if (src_fd < 0 || !rows || offset > INT64_MAX || bytes > INT64_MAX - offset)
        return fail(EINVAL);
    int fd = fcntl(src_fd, F_DUPFD_CLOEXEC, 0);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) goto error;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || offset + bytes > (uint64_t)st.st_size) {
        errno = EINVAL;
        goto error;
    }
#ifdef __APPLE__
    if (getenv("MYDS_ENGRAM_NOCACHE")) {
        if (fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0) goto error;
    }
#endif
    *table = (myds_engram_table_t){.fd = fd, .offset = offset, .rows = rows};
    return true;
error: {
        int saved = errno;
        close(fd);
        return fail(saved);
    }
}

/* 2026-10-25 byte-split：行数据只能从复合映射读。
 *
 * 字节分段的张量跨越多个分片，一行 264 字节也可能跨边界 ⇒ 没有单一
 * (fd, offset) 能表达它。映射已经是连续的，所以直接 memcpy。
 *
 * 仍然 dup 一份 src_fd：表的关闭语义、clone、以及「fd < 0 即为未打开」的
 * 既有判断都依赖它，少这一份会让两条路径的行为分叉。 */
bool myds_engram_table_open_map(myds_engram_table_t *table, int src_fd,
                               const uint8_t *map, uint64_t map_size,
                               uint64_t offset, uint32_t rows) {
    if (!table) return fail(EINVAL);
    *table = (myds_engram_table_t){.fd = -1};
    const uint64_t bytes = (uint64_t)rows * MYDS_ENGRAM_ROW_BYTES;
    if (!map || src_fd < 0 || !rows || map_size > INT64_MAX ||
        offset > map_size || bytes > map_size - offset)
        return fail(EINVAL);
    int fd = fcntl(src_fd, F_DUPFD_CLOEXEC, 0);
    if (fd < 0) return false;
    *table = (myds_engram_table_t){.fd = fd, .map = map, .offset = offset, .rows = rows};
    return true;
}

void myds_engram_table_close(myds_engram_table_t *table) {
    if (!table) return;
    if (table->fd >= 0) close(table->fd);
    *table = (myds_engram_table_t){.fd = -1};
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Every decoded value depends only on two bytes. Populate these tables once,
 * then never mutate them; pthread_once publishes them to concurrent readers.
 * Keep the original ldexpf and BF16 operation order during initialization,
 * including signed zero, subnormal rounding and both overflow checks. Inference
 * uses round-to-nearest; other caller rounding modes use the scalar fallback.
 * Initialization must not leak overflow/underflow flags from unrelated entries. */
static float decode_values[256][256];
static unsigned char decode_range_error[256][256];
static pthread_once_t decode_once = PTHREAD_ONCE_INIT;

static void init_decode_values(void) {
    const int saved_errno = errno;
    fenv_t saved_env;
    const bool restore_env = feholdexcept(&saved_env) == 0;
    if (restore_env) fesetround(FE_TONEAREST);
    for (int scale = 0; scale < 256; ++scale) {
        for (int code = 0; code < 256; ++code) {
            float value = NAN;
            errno = 0;
            if ((code & 127) != 127 && scale != 255) {
                const int exponent = (code >> 3) & 15, fraction = code & 7;
                value = exponent ? ldexpf((float)(8 + fraction), exponent - 10) :
                                   ldexpf((float)fraction, -9);
                if (code & 128) value = -value;
                value = ldexpf(value, scale - 127);
                if (isfinite(value)) value = bf16(value);
            }
            decode_values[scale][code] = value;
            decode_range_error[scale][code] = errno == ERANGE;
        }
    }
    if (restore_env) fesetenv(&saved_env);
    errno = saved_errno;
}

static bool decode_row_scalar(const uint8_t *raw, float *out) {
    for (int i = 0; i < MYDS_ENGRAM_DIM; ++i) {
        const unsigned code = raw[i], scale = raw[MYDS_ENGRAM_DIM + i / 32];
        if ((code & 127u) == 127u || scale == 255u) return fail(EDOM);
        const int exponent = (code >> 3) & 15, fraction = code & 7;
        float value = exponent ? ldexpf((float)(8 + fraction), exponent - 10) :
                                 ldexpf((float)fraction, -9);
        if (code & 128u) value = -value;
        value = ldexpf(value, (int)scale - 127);
        if (!isfinite(value)) return fail(EDOM);
        value = bf16(value);
        if (!isfinite(value)) return fail(EDOM);
        out[i] = value;
    }
    return true;
}

bool myds_engram_decode_row(const uint8_t raw[MYDS_ENGRAM_ROW_BYTES],
                           float out[MYDS_ENGRAM_DIM]) {
    if (!raw || !out) return fail(EINVAL);
    if (fegetround() != FE_TONEAREST) return decode_row_scalar(raw, out);
    const int saved_errno = errno;
    pthread_once(&decode_once, init_decode_values);
    errno = saved_errno; /* Building unrelated combinations must not leak errno. */
    for (int i = 0; i < MYDS_ENGRAM_DIM; ++i) {
        const uint8_t code = raw[i], scale = raw[MYDS_ENGRAM_DIM + i / 32];
        const float value = decode_values[scale][code];
        if (!isfinite(value)) return fail(EDOM);
        if (decode_range_error[scale][code]) errno = ERANGE;
        out[i] = value;
    }
    return true;
}

bool myds_engram_read(const myds_engram_table_t *table, const uint32_t *rows,
                     size_t count, float *out) {
    if (!table || table->fd < 0 || (count && (!rows || !out)) ||
        count > SIZE_MAX / (MYDS_ENGRAM_DIM * sizeof(*out))) return fail(EINVAL);
    for (size_t i = 0; i < count; ++i)
        if (rows[i] >= table->rows) return fail(EINVAL);
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    for (size_t i = 0; i < count; ++i) {
        /* Repeated IDs within a small Engram lookup need one physical read. */
        size_t previous = i > MYDS_ENGRAM_COLS ? i - MYDS_ENGRAM_COLS : 0;
        for (; previous < i; ++previous) if (rows[previous] == rows[i]) break;
        if (previous < i) {
            memcpy(out + i * MYDS_ENGRAM_DIM, out + previous * MYDS_ENGRAM_DIM,
                   MYDS_ENGRAM_DIM * sizeof(*out));
            continue;
        }
        uint64_t offset = table->offset + (uint64_t)rows[i] * sizeof(raw);
        if (table->map) {
            memcpy(raw, table->map + offset, sizeof(raw));
        } else {
            size_t done = 0;
            while (done < sizeof(raw)) {
                ssize_t n = pread(table->fd, raw + done, sizeof(raw) - done, (off_t)(offset + done));
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) return n == 0 ? fail(EIO) : false;
                done += (size_t)n;
            }
        }
        if (!myds_engram_decode_row(raw, out + i * MYDS_ENGRAM_DIM)) return false;
    }
    return true;
}

enum {
    ENGRAM_IO_MAX_THREADS = 16,
    ENGRAM_BATCH_MAX = 64,
    ENGRAM_IO_JOBS = MYDS_ENGRAM_LAYERS * MYDS_ENGRAM_COLS,
    ENGRAM_IO_BATCH_JOBS = ENGRAM_BATCH_MAX * ENGRAM_IO_JOBS,
    ENGRAM_BATCH_FLAG = 0x80000000u
};
typedef struct {
    int fd, error, retry_errno;
    const uint8_t *map;   /* 非空 ⇒ 从复合映射拷贝，忽略 fd（byte-split） */
    uint64_t offset;
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
} engram_io_job_t;
typedef struct {
    struct myds_engram_reader *reader;
    unsigned index;
} engram_io_worker_t;

struct myds_engram_reader {
    unsigned workers, started, remaining, job_count;
    bool stopping;
    uint64_t generation;
    /* submit serializes callers through decoding; state protects dispatch and
     * the barrier. A worker owns each strided raw/error slot until the barrier. */
    pthread_mutex_t submit, state;
    pthread_cond_t work, done;
    pthread_t threads[ENGRAM_IO_MAX_THREADS];
    engram_io_worker_t worker[ENGRAM_IO_MAX_THREADS];
    engram_io_job_t jobs[ENGRAM_IO_JOBS];
    engram_io_job_t *batch_jobs;
};

static void read_raw_job(engram_io_job_t *job) {
    if (job->map) {
        memcpy(job->raw, job->map + job->offset, sizeof(job->raw));
        job->error = 0;
        return;
    }
    size_t done = 0;
    while (done < sizeof(job->raw)) {
        ssize_t n = pread(job->fd, job->raw + done, sizeof(job->raw) - done,
                          (off_t)(job->offset + done));
        if (n < 0 && errno == EINTR) { job->retry_errno = EINTR; continue; }
        if (n <= 0) { job->error = n == 0 ? EIO : errno; return; }
        done += (size_t)n;
    }
    job->error = 0;
}

static __attribute__((noinline)) void engram_io_worker_batch(
    myds_engram_reader_t *r, unsigned worker_index, unsigned count) {
    unsigned my_jobs[ENGRAM_IO_BATCH_JOBS];
    unsigned n_my_jobs = 0;
    for (unsigned i = worker_index; i < count; i += r->workers)
        my_jobs[n_my_jobs++] = i;
    for (unsigned i = 1; i < n_my_jobs; ++i) {
        unsigned tmp = my_jobs[i];
        uint64_t off = r->batch_jobs[tmp].offset;
        unsigned j = i;
        while (j > 0 && r->batch_jobs[my_jobs[j - 1]].offset > off) {
            my_jobs[j] = my_jobs[j - 1];
            --j;
        }
        my_jobs[j] = tmp;
    }
    for (unsigned i = 0; i < n_my_jobs; ++i)
        read_raw_job(&r->batch_jobs[my_jobs[i]]);
}

static void *engram_io_worker(void *opaque) {
    engram_io_worker_t *worker = opaque;
    myds_engram_reader_t *r = worker->reader;
    uint64_t seen = 0;
    pthread_mutex_lock(&r->state);
    for (;;) {
        while (!r->stopping && seen == r->generation)
            pthread_cond_wait(&r->work, &r->state);
        if (r->stopping) break;
        seen = r->generation;
        const unsigned count = r->job_count;
        pthread_mutex_unlock(&r->state);
        if (__builtin_expect(!(count & ENGRAM_BATCH_FLAG), 1)) {
            /* Sequentialize I/O: sort assigned jobs by offset to avoid random disk/page thrashing */
            unsigned my_jobs[ENGRAM_IO_JOBS];
            unsigned n_my_jobs = 0;
            for (unsigned i = worker->index; i < count; i += r->workers)
                my_jobs[n_my_jobs++] = i;
            for (unsigned i = 0; i < n_my_jobs; ++i) {
                for (unsigned j = i + 1; j < n_my_jobs; ++j) {
                    if (r->jobs[my_jobs[j]].offset < r->jobs[my_jobs[i]].offset) {
                        unsigned tmp = my_jobs[i];
                        my_jobs[i] = my_jobs[j];
                        my_jobs[j] = tmp;
                    }
                }
            }
            for (unsigned i = 0; i < n_my_jobs; ++i)
                read_raw_job(&r->jobs[my_jobs[i]]);
        } else {
            engram_io_worker_batch(r, worker->index, count & ~ENGRAM_BATCH_FLAG);
        }
        pthread_mutex_lock(&r->state);
        if (--r->remaining == 0) pthread_cond_signal(&r->done);
    }
    pthread_mutex_unlock(&r->state);
    return NULL;
}

myds_engram_reader_t *myds_engram_reader_create(unsigned workers) {
    if (!workers || workers > ENGRAM_IO_MAX_THREADS) { errno = EINVAL; return NULL; }
    const int saved_errno = errno;
    myds_engram_reader_t *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->workers = workers;
    if (workers == 1) { errno = saved_errno; return r; }
    int error = pthread_mutex_init(&r->submit, NULL);
    if (error) goto allocation;
    error = pthread_mutex_init(&r->state, NULL);
    if (error) goto submit;
    error = pthread_cond_init(&r->work, NULL);
    if (error) goto state;
    error = pthread_cond_init(&r->done, NULL);
    if (error) goto work;
    for (unsigned i = 0; i < workers; ++i) {
        r->worker[i] = (engram_io_worker_t){.reader = r, .index = i};
        error = pthread_create(&r->threads[i], NULL, engram_io_worker, &r->worker[i]);
        if (error) {
            pthread_mutex_lock(&r->state);
            r->stopping = true;
            pthread_cond_broadcast(&r->work);
            pthread_mutex_unlock(&r->state);
            for (unsigned j = 0; j < r->started; ++j) pthread_join(r->threads[j], NULL);
            pthread_cond_destroy(&r->done);
            goto work;
        }
        ++r->started;
    }
    errno = saved_errno;
    return r;
work:
    pthread_cond_destroy(&r->work);
state:
    pthread_mutex_destroy(&r->state);
submit:
    pthread_mutex_destroy(&r->submit);
allocation:
    free(r);
    errno = error;
    return NULL;
}

void myds_engram_reader_free(myds_engram_reader_t *r) {
    if (!r) return;
    if (r->workers > 1) {
        pthread_mutex_lock(&r->state);
        r->stopping = true;
        pthread_cond_broadcast(&r->work);
        pthread_mutex_unlock(&r->state);
        for (unsigned i = 0; i < r->started; ++i) pthread_join(r->threads[i], NULL);
        pthread_cond_destroy(&r->done);
        pthread_cond_destroy(&r->work);
        pthread_mutex_destroy(&r->state);
        pthread_mutex_destroy(&r->submit);
    }
    free(r->batch_jobs);
    free(r);
}

bool myds_engram_reader_read_token(myds_engram_reader_t *r,
    const myds_engram_table_t tables[MYDS_ENGRAM_LAYERS],
    const uint32_t rows[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS],
    float out[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM]) {
    if (!tables || !rows || !out) return fail(EINVAL);
    if (!r || r->workers == 1) {
        for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer)
            if (!myds_engram_read(&tables[layer], rows[layer], MYDS_ENGRAM_COLS, out[layer]))
                return false;
        return true;
    }
    /* Invalid inputs use the scalar path, retaining its validation/error order
     * even when the first table also contains a decode or physical-read error. */
    for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer) {
        if (tables[layer].fd < 0)
            return myds_engram_reader_read_token(NULL, tables, rows, out);
        for (int col = 0; col < MYDS_ENGRAM_COLS; ++col)
            if (rows[layer][col] >= tables[layer].rows)
                return myds_engram_reader_read_token(NULL, tables, rows, out);
    }
    pthread_mutex_lock(&r->submit);
    unsigned map[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];
    unsigned jobs = 0;
    for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer) {
        for (int col = 0; col < MYDS_ENGRAM_COLS; ++col) {
            int previous = 0;
            while (previous < col && rows[layer][previous] != rows[layer][col]) ++previous;
            if (previous < col) { map[layer][col] = map[layer][previous]; continue; }
            map[layer][col] = jobs;
            r->jobs[jobs].fd = tables[layer].fd;
            r->jobs[jobs].map = tables[layer].map;
            r->jobs[jobs].offset = tables[layer].offset +
                                  (uint64_t)rows[layer][col] * MYDS_ENGRAM_ROW_BYTES;
            r->jobs[jobs].error = 0;
            r->jobs[jobs].retry_errno = 0;
            ++jobs;
        }
    }
    pthread_mutex_lock(&r->state);
    r->job_count = jobs;
    r->remaining = r->workers;
    ++r->generation;
    pthread_cond_broadcast(&r->work);
    while (r->remaining) pthread_cond_wait(&r->done, &r->state);
    pthread_mutex_unlock(&r->state);
    /* Drain all workers first, then reproduce scalar I/O/decode error priority.
     * Keeping decode here also retains caller errno and floating-point flags. */
    bool ok = true;
    for (int layer = 0; layer < MYDS_ENGRAM_LAYERS && ok; ++layer) {
        for (int col = 0; col < MYDS_ENGRAM_COLS; ++col) {
            engram_io_job_t *job = &r->jobs[map[layer][col]];
            if (job->error) { errno = job->error; ok = false; break; }
            int previous = 0;
            while (previous < col && map[layer][previous] != map[layer][col]) ++previous;
            if (previous < col)
                memcpy(out[layer] + col * MYDS_ENGRAM_DIM,
                       out[layer] + previous * MYDS_ENGRAM_DIM, MYDS_ENGRAM_DIM * sizeof(float));
            else {
                /* Scalar pread leaves EINTR after a successful retry. Apply it
                 * once for the physical read, never again for duplicate IDs. */
                if (job->retry_errno) errno = job->retry_errno;
                if (!myds_engram_decode_row(job->raw, out[layer] + col * MYDS_ENGRAM_DIM)) {
                    ok = false;
                    break;
                }
            }
        }
    }
    pthread_mutex_unlock(&r->submit);
    return ok;
}

bool myds_engram_set_io_threads(myds_engram_t *e, unsigned workers) {
    if (!e || !workers || workers > ENGRAM_IO_MAX_THREADS) return fail(EINVAL);
    myds_engram_reader_t *next = workers == 1 ? NULL : myds_engram_reader_create(workers);
    if (workers > 1 && !next) return false;
    myds_engram_reader_free(e->reader);
    e->reader = next;
    e->io_threads = workers;
    /* Keep the adaptive controller's view honest when a caller pins the
     * worker count through this API. */
    if (e->io_adaptive) e->adapt_pool = workers > 1;
    return true;
}

/*
 * Adaptive raw-read controller (MYDS_ENGRAM_IO_THREADS=auto).
 *
 * Measured per-token prepare cost on this host:
 *   hot cache   serial 25 us   pool(8) 83 us   -> serial wins by 58 us
 *   cold table  serial 1197 us pool(8) 249 us  -> pool wins by 948 us
 * The two regimes are an order of magnitude apart, so one mark separates them
 * for both directions. Consecutive-window counters plus a cooldown keep the
 * controller from flapping when a prompt mixes cold and warm n-grams.
 */
enum {
    ENGRAM_ADAPT_WINDOW_TOKENS   = 8,
    ENGRAM_ADAPT_MARK_US         = 150,
    ENGRAM_ADAPT_UP_WINDOWS      = 2,
    ENGRAM_ADAPT_DOWN_WINDOWS    = 24,
    ENGRAM_ADAPT_COOLDOWN_WINDOWS= 64,
    ENGRAM_ADAPT_WORKERS         = 8
};

static double engram_now_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static void engram_adapt_note(myds_engram_t *e, double us, size_t tokens) {
    if (!e->io_adaptive || !tokens) return;
    e->adapt_us += us;
    e->adapt_win += (unsigned)tokens;
    if (e->adapt_win < ENGRAM_ADAPT_WINDOW_TOKENS) return;
    const double avg = e->adapt_us / (double)e->adapt_win;
    e->adapt_us = 0.0;
    e->adapt_win = 0;
    if (e->adapt_cooldown) { --e->adapt_cooldown; return; }
    if (!e->adapt_pool) {
        if (avg > ENGRAM_ADAPT_MARK_US) {
            if (++e->adapt_up >= ENGRAM_ADAPT_UP_WINDOWS) {
                /* A failed switch leaves the serial path in place: still correct. */
                if (myds_engram_set_io_threads(e, ENGRAM_ADAPT_WORKERS)) {
                    e->adapt_cooldown = ENGRAM_ADAPT_COOLDOWN_WINDOWS;
                }
                e->adapt_up = e->adapt_down = 0;
            }
        } else {
            e->adapt_up = 0;
        }
    } else {
        if (avg < ENGRAM_ADAPT_MARK_US) {
            if (++e->adapt_down >= ENGRAM_ADAPT_DOWN_WINDOWS) {
                if (myds_engram_set_io_threads(e, 1u)) {
                    e->adapt_cooldown = ENGRAM_ADAPT_COOLDOWN_WINDOWS;
                }
                e->adapt_up = e->adapt_down = 0;
            }
        } else {
            e->adapt_down = 0;
        }
    }
}

static unsigned configured_io_threads(void) {
    const char *text = getenv("MYDS_ENGRAM_IO_THREADS");
    if (!text) return 8;
    unsigned value = 0;
    if (!*text) return 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9') return 0;
        value = value * 10u + (unsigned)(*text - '0');
        if (value > ENGRAM_IO_MAX_THREADS) return 0;
    }
    return value;
}

static const void *array(const gguf_file_t *gf, const char *key, gguf_type type, size_t count) {
    const gguf_kv_t *kv = gguf_find_key(gf, key);
    return kv && kv->type == GGUF_TYPE_ARRAY && kv->value.val_arr.elem_type == type &&
           kv->value.val_arr.count == count ? kv->value.val_arr.data : NULL;
}

static bool shape(const gguf_tensor_t *t, ggml_type type, uint64_t a, uint64_t b) {
    if (!t || t->type != type || t->n_dims != 2 || t->dims[0] != a || t->dims[1] != b)
        return false;
    size_t row_bytes = ggml_type_row_size(type, (size_t)a);
    return row_bytes && b <= SIZE_MAX / row_bytes && t->size_bytes == b * row_bytes;
}

myds_engram_t *myds_engram_create(const gguf_file_t *gf, const char *path,
                                 uint32_t vocab_size, uint32_t width) {
    if (!gf || !path || !vocab_size || !width || width > UINT32_MAX / 5u) {
        errno = EINVAL;
        return NULL;
    }
    myds_engram_t *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    for (int i = 0; i < MYDS_ENGRAM_LAYERS; ++i) e->table[i].fd = -1;
    {
        const char *io = getenv("MYDS_ENGRAM_IO_THREADS");
        e->io_adaptive = io && strcmp(io, "auto") == 0;
    }
    e->io_threads = e->io_adaptive ? 1u : configured_io_threads();
    if (!e->io_threads) {
        fputs("Invalid MYDS_ENGRAM_IO_THREADS: expected an integer in 1..16\n", stderr);
        goto invalid;
    }
    const char *encoding = gguf_get_str(gf, "deepseek41.engram.encoding", "");
    const uint32_t *layer_ids = array(gf, "deepseek41.engram.layer_ids", GGUF_TYPE_UINT32, 2);
    const uint32_t *rows = array(gf, "deepseek41.engram.rows", GGUF_TYPE_UINT32, 2);
    const uint32_t *primes = array(gf, "deepseek41.engram.primes", GGUF_TYPE_UINT32, 48);
    const uint64_t *multipliers = array(gf, "deepseek41.engram.multipliers", GGUF_TYPE_UINT64, 8);
    e->layout.token_map = array(gf, "deepseek41.engram.token_map", GGUF_TYPE_UINT32, vocab_size);
    if (strcmp(encoding, "e4m3_e8m0_32_row264") || !layer_ids || !rows || !primes ||
        !multipliers || !e->layout.token_map || layer_ids[0] != 1 || layer_ids[1] != 14)
        goto invalid;
    memcpy(e->layer_ids, layer_ids, sizeof(e->layer_ids));
    memcpy(e->layout.rows, rows, sizeof(e->layout.rows));
    memcpy(e->layout.primes, primes, sizeof(e->layout.primes));
    memcpy(e->layout.multipliers, multipliers, sizeof(e->layout.multipliers));
    e->layout.vocab_size = vocab_size;
    e->layout.compressed_vocab_size = gguf_get_u32(gf, "deepseek41.engram.compressed_vocab_size", 0);
    e->layout.pad_id = gguf_get_u32(gf, "deepseek41.engram.pad_id", UINT32_MAX);
    if (!myds_engram_layout_valid(&e->layout)) goto invalid;
    e->width = width;
    for (int i = 0; i < MYDS_ENGRAM_LAYERS; ++i) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%u.engram_embd.weight", e->layer_ids[i]);
        const gguf_tensor_t *table = gguf_find_tensor(gf, name);
        if (!shape(table, GGML_TYPE_I8, MYDS_ENGRAM_ROW_BYTES, e->layout.rows[i]))
            goto invalid;
        /* 2026-10-25 split GGUF：不能再自己拼 gf->fd + gf->tensor_data_offset。
         * 拆片后这个张量可能在 part 3，而 gf->fd 只是 part 0 —— 那样会从错误的
         * 文件读、且偏移也是错的（甚至越界）。gguf_tensor_source 给出正确的
         * 分片 fd 与文件内绝对偏移。 */
        int src_fd = -1;
        uint64_t src_offset = 0;
        if (gf->byte_split) {
            /* 字节分段：表可能横跨分片，行也可能跨边界 ⇒ 只能从复合映射读。
             * 偏移仍用「逻辑文件内绝对偏移」，与单文件语义一致。 */
            if (!table->data || !gf->mmap_addr) goto invalid;
            if (!myds_engram_table_open_map(&e->table[i], gf->fd, gf->mmap_addr,
                                            (uint64_t)gf->mmap_size,
                                            gf->tensor_data_offset + table->offset,
                                            e->layout.rows[i])) goto error;
        } else {
            if (!gguf_tensor_source(gf, table, &src_fd, &src_offset)) goto invalid;
            if (!myds_engram_table_open_fd(&e->table[i], src_fd, src_offset,
                                           e->layout.rows[i])) goto error;
        }
        snprintf(name, sizeof(name), "blk.%u.engram_kv.weight", e->layer_ids[i]);
        e->projection[i] = gguf_find_tensor(gf, name);
        snprintf(name, sizeof(name), "blk.%u.engram_q_norm.weight", e->layer_ids[i]);
        e->q_norm[i] = gguf_find_tensor(gf, name);
        snprintf(name, sizeof(name), "blk.%u.engram_k_norm.weight", e->layer_ids[i]);
        e->k_norm[i] = gguf_find_tensor(gf, name);
        if (!shape(e->projection[i], GGML_TYPE_F16, MYDS_ENGRAM_INPUT_DIM, 5u * width) ||
            !shape(e->q_norm[i], GGML_TYPE_F32, width, 4) ||
            !shape(e->k_norm[i], GGML_TYPE_F32, width, 4)) goto invalid;
    }
    myds_engram_reset(e);
    return e;
invalid:
    errno = EINVAL;
error: {
        int saved = errno;
        myds_engram_free(e);
        errno = saved;
        return NULL;
    }
}

void myds_engram_free(myds_engram_t *e) {
    if (!e) return;
    myds_engram_reader_free(e->reader);
    for (int i = 0; i < MYDS_ENGRAM_LAYERS; ++i) myds_engram_table_close(&e->table[i]);
    free(e);
}

myds_engram_t *myds_engram_clone(const myds_engram_t *source) {
    if (!source || source->table[0].fd < 0 || source->table[1].fd < 0) {
        errno = EINVAL;
        return NULL;
    }
    myds_engram_t *copy = malloc(sizeof(*copy));
    if (!copy) return NULL;
    memcpy(copy, source, sizeof(*copy));
    copy->reader = NULL;
    /* map 指针随结构体 memcpy 复制，指向同一段映射（生命周期由 gguf_file_t 持有）。 */
    for (int i = 0; i < MYDS_ENGRAM_LAYERS; ++i) copy->table[i].fd = -1;
    for (int i = 0; i < MYDS_ENGRAM_LAYERS; ++i) {
        copy->table[i].fd = fcntl(source->table[i].fd, F_DUPFD_CLOEXEC, 0);
        if (copy->table[i].fd < 0) {
            int saved = errno;
            myds_engram_free(copy);
            errno = saved;
            return NULL;
        }
    }
    if (copy->io_threads > 1 && !myds_engram_set_io_threads(copy, copy->io_threads)) {
        int saved = errno;
        myds_engram_free(copy);
        errno = saved;
        return NULL;
    }
    myds_engram_reset(copy);
    return copy;
}

bool myds_engram_unmap_tables(gguf_file_t *gf, const myds_engram_t *e) {
    if (!gf || !e || !gf->mmap_addr || !gf->mmap_size) return fail(EINVAL);
    const long page = sysconf(_SC_PAGESIZE);
    uint64_t start = e->table[0].offset;
    if (page <= 0 || !start || start % (uint64_t)page || start > gf->mmap_size ||
        e->table[1].offset < start + (uint64_t)e->table[0].rows * MYDS_ENGRAM_ROW_BYTES)
        return fail(EINVAL);
    gguf_tensor_t *tables[2] = {NULL, NULL};
    for (uint64_t i = 0; i < gf->tensor_count; ++i) {
        gguf_tensor_t *t = &gf->tensors[i];
        int table = strcmp(t->name, "blk.1.engram_embd.weight") == 0 ? 0 :
                    strcmp(t->name, "blk.14.engram_embd.weight") == 0 ? 1 : -1;
        if (t->offset > UINT64_MAX - gf->tensor_data_offset) return fail(EINVAL);
        uint64_t offset = gf->tensor_data_offset + t->offset;
        if (table >= 0) {
            if (tables[table] || offset != e->table[table].offset ||
                !shape(t, GGML_TYPE_I8, MYDS_ENGRAM_ROW_BYTES, e->table[table].rows))
                return fail(EINVAL);
            tables[table] = t;
        } else if (offset > start || t->size_bytes > start - offset) {
            return fail(EINVAL);
        }
    }
    if (!tables[0] || !tables[1]) return fail(EINVAL);
    /* An already-unmapped model is safe for repeated loader validation. */
    if (start == gf->mmap_size)
        return !tables[0]->data && !tables[1]->data ? true : fail(EINVAL);
    if (e->table[1].offset > gf->mmap_size ||
        tables[1]->size_bytes > gf->mmap_size - e->table[1].offset) return fail(EINVAL);
    if (gf->byte_split) {
        /* 字节分段下这段映射【不是】disk-only 的：engram 的行就从它读，
         * 因为一行 264 字节也可能跨越分片边界，没有单一 (fd, offset) 能表达。
         * 布局已按上面的规则校验过，这里保留映射并返回成功 —— 回收它会让
         * 后续行读取 SIGSEGV。 */
        return true;
    }
    if (munmap(gf->mmap_addr + start, gf->mmap_size - (size_t)start) != 0) return false;
    gf->mmap_size = (size_t)start;
    tables[0]->data = tables[1]->data = NULL;
    return true;
}

void myds_engram_reset(myds_engram_t *e) {
    if (!e) return;
    myds_engram_history_reset(&e->history);
    e->position = 0;
    e->prepared = false;
    /* A new sequence re-hashes from scratch, so drop the window in progress but
     * keep whatever engagement the controller already decided on. */
    e->adapt_us = 0.0;
    e->adapt_win = 0;
    e->adapt_up = e->adapt_down = 0;
}

static bool prepare_token_step(myds_engram_t *e, int32_t token, uint64_t position) {
    if (!e) return fail(EINVAL);
    e->prepared = false;
    if (position != e->position || position == UINT64_MAX) return fail(EINVAL);
    myds_engram_history_t next = e->history;
    if (!myds_engram_hash(&e->layout, &next, &token, NULL, 1, &e->row_ids[0][0])) return false;
    if (e->io_threads > 1 && !e->reader &&
        !myds_engram_set_io_threads(e, e->io_threads)) return false;
    if (!myds_engram_reader_read_token(e->reader, e->table, e->row_ids, e->row_values))
        return false;
    e->history = next;
    ++e->position;
    e->prepared = true;
    return true;
}

bool myds_engram_prepare_token(myds_engram_t *e, int32_t token, uint64_t position) {
    if (!e || !e->io_adaptive) return prepare_token_step(e, token, position);
    const double t0 = engram_now_us();
    const bool ok = prepare_token_step(e, token, position);
    /* Timed here, not inside the step: the controller may only swap readers
     * between lookups, never while one is in flight. */
    engram_adapt_note(e, engram_now_us() - t0, 1);
    return ok;
}

static bool prepare_batch_step(myds_engram_t *e, const int32_t *tokens,
                              size_t count, uint64_t start_position,
                              float *out_layer0, float *out_layer1) {
    if (!e || (count && (!tokens || !out_layer0 || !out_layer1))) return fail(EINVAL);
    if (!count) return true;
    if (start_position != e->position || count > UINT64_MAX - start_position) {
        e->prepared = false;
        return fail(EINVAL);
    }
    if (e->io_threads > 1 && !e->reader &&
        !myds_engram_set_io_threads(e, e->io_threads)) {
        e->prepared = false;
        return false;
    }
    myds_engram_reader_t *r = e->reader;
    if (!r || r->workers <= 1 || count == 1 ||
        e->table[0].fd < 0 || e->table[1].fd < 0 ||
        fegetround() != FE_TONEAREST) {
        for (size_t t = 0; t < count; ++t) {
            if (!prepare_token_step(e, tokens[t], start_position + t)) return false;
            memcpy(out_layer0 + t * MYDS_ENGRAM_INPUT_DIM,
                   e->row_values[0], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
            memcpy(out_layer1 + t * MYDS_ENGRAM_INPUT_DIM,
                   e->row_values[1], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        }
        return true;
    }

    for (size_t off = 0; off < count; ) {
        size_t chunk = count - off;
        if (chunk > ENGRAM_BATCH_MAX) chunk = ENGRAM_BATCH_MAX;

        uint32_t batch_rows[ENGRAM_BATCH_MAX][MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];
        myds_engram_history_t next = e->history;
        bool valid = myds_engram_hash(&e->layout, &next, tokens + off, NULL, chunk, &batch_rows[0][0][0]);
        if (valid) {
            for (size_t t = 0; t < chunk && valid; ++t)
                for (int layer = 0; layer < MYDS_ENGRAM_LAYERS && valid; ++layer)
                    for (int col = 0; col < MYDS_ENGRAM_COLS; ++col)
                        if (batch_rows[t][layer][col] >= e->table[layer].rows) { valid = false; break; }
        }
        if (!valid) {
            for (size_t t = 0; t < chunk; ++t) {
                if (!prepare_token_step(e, tokens[off + t], start_position + off + t)) return false;
                memcpy(out_layer0 + (off + t) * MYDS_ENGRAM_INPUT_DIM,
                       e->row_values[0], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
                memcpy(out_layer1 + (off + t) * MYDS_ENGRAM_INPUT_DIM,
                       e->row_values[1], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
            }
            off += chunk;
            continue;
        }

        pthread_mutex_lock(&r->submit);
        if (!r->batch_jobs) {
            r->batch_jobs = (engram_io_job_t *)calloc(ENGRAM_IO_BATCH_JOBS, sizeof(engram_io_job_t));
            if (!r->batch_jobs) {
                pthread_mutex_unlock(&r->submit);
                for (size_t t = 0; t < chunk; ++t) {
                    if (!prepare_token_step(e, tokens[off + t], start_position + off + t)) return false;
                    memcpy(out_layer0 + (off + t) * MYDS_ENGRAM_INPUT_DIM,
                           e->row_values[0], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
                    memcpy(out_layer1 + (off + t) * MYDS_ENGRAM_INPUT_DIM,
                           e->row_values[1], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
                }
                off += chunk;
                continue;
            }
        }
        unsigned map[ENGRAM_BATCH_MAX][MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];
        unsigned jobs = 0;
        uint32_t ht_row[4096];
        int16_t ht_job[4096];
        for (int layer = 0; layer < MYDS_ENGRAM_LAYERS; ++layer) {
            memset(ht_job, -1, sizeof(ht_job));
            for (size_t t = 0; t < chunk; ++t) {
                for (int col = 0; col < MYDS_ENGRAM_COLS; ++col) {
                    const uint32_t rid = batch_rows[t][layer][col];
                    uint32_t h = (rid * 2654435761u) & 4095u;
                    while (ht_job[h] >= 0 && ht_row[h] != rid)
                        h = (h + 1u) & 4095u;
                    if (ht_job[h] >= 0) {
                        map[t][layer][col] = (unsigned)ht_job[h];
                        continue;
                    }
                    ht_row[h] = rid;
                    ht_job[h] = (int16_t)jobs;
                    map[t][layer][col] = jobs;
                    r->batch_jobs[jobs].fd = e->table[layer].fd;
                    r->batch_jobs[jobs].map = e->table[layer].map;
                    r->batch_jobs[jobs].offset = e->table[layer].offset +
                                                 (uint64_t)rid * MYDS_ENGRAM_ROW_BYTES;
                    r->batch_jobs[jobs].error = 0;
                    r->batch_jobs[jobs].retry_errno = 0;
                    ++jobs;
                }
            }
        }
        pthread_mutex_lock(&r->state);
        r->job_count = jobs | ENGRAM_BATCH_FLAG;
        r->remaining = r->workers;
        ++r->generation;
        pthread_cond_broadcast(&r->work);
        while (r->remaining) pthread_cond_wait(&r->done, &r->state);
        pthread_mutex_unlock(&r->state);

        float *job_decoded[ENGRAM_IO_BATCH_JOBS];
        memset(job_decoded, 0, jobs * sizeof(float *));
        bool ok = true;
        for (size_t t = 0; t < chunk && ok; ++t) {
            for (int layer = 0; layer < MYDS_ENGRAM_LAYERS && ok; ++layer) {
                float *dst_base = (layer == 0 ? out_layer0 : out_layer1) +
                                  (off + t) * MYDS_ENGRAM_INPUT_DIM;
                for (int col = 0; col < MYDS_ENGRAM_COLS; ++col) {
                    const unsigned jid = map[t][layer][col];
                    engram_io_job_t *job = &r->batch_jobs[jid];
                    if (job->error) { ok = false; break; }
                    float *dst = dst_base + col * MYDS_ENGRAM_DIM;
                    if (job_decoded[jid]) {
                        memcpy(dst, job_decoded[jid], MYDS_ENGRAM_DIM * sizeof(float));
                    } else {
                        if (job->retry_errno) errno = job->retry_errno;
                        if (!myds_engram_decode_row(job->raw, dst)) { ok = false; break; }
                        job_decoded[jid] = dst;
                    }
                }
            }
        }
        pthread_mutex_unlock(&r->submit);

        if (!ok) {
            for (size_t t = 0; t < chunk; ++t) {
                if (!prepare_token_step(e, tokens[off + t], start_position + off + t)) return false;
                memcpy(out_layer0 + (off + t) * MYDS_ENGRAM_INPUT_DIM,
                       e->row_values[0], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
                memcpy(out_layer1 + (off + t) * MYDS_ENGRAM_INPUT_DIM,
                       e->row_values[1], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
            }
            off += chunk;
            continue;
        }

        memcpy(e->row_ids, batch_rows[chunk - 1], sizeof(e->row_ids));
        memcpy(e->row_values[0], out_layer0 + (off + chunk - 1) * MYDS_ENGRAM_INPUT_DIM,
               MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        memcpy(e->row_values[1], out_layer1 + (off + chunk - 1) * MYDS_ENGRAM_INPUT_DIM,
               MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        e->history = next;
        e->position += chunk;
        e->prepared = true;
        off += chunk;
    }
    return true;
}

bool myds_engram_prepare_batch(myds_engram_t *e, const int32_t *tokens,
                              size_t count, uint64_t start_position,
                              float *out_layer0, float *out_layer1) {
    if (!e || !e->io_adaptive || !count)
        return prepare_batch_step(e, tokens, count, start_position, out_layer0, out_layer1);
    const double t0 = engram_now_us();
    const bool ok = prepare_batch_step(e, tokens, count, start_position, out_layer0, out_layer1);
    engram_adapt_note(e, engram_now_us() - t0, count);
    return ok;
}

bool myds_engram_add_reference(float *residual, const float *kv,
                               const float *q_weight, const float *k_weight,
                               uint32_t width, float eps) {
    if (!residual || !kv || !q_weight || !k_weight || !width || !isfinite(eps) || eps <= 0)
        return fail(EINVAL);
    for (size_t i = 0; i < (size_t)width * 4; ++i)
        if (!isfinite(residual[i]) || !isfinite(q_weight[i]) || !isfinite(k_weight[i]))
            return fail(EDOM);
    for (size_t i = 0; i < (size_t)width * 5; ++i)
        if (!isfinite(kv[i]) || !isfinite(bf16(kv[i]))) return fail(EDOM);
    for (unsigned stream = 0; stream < 4; ++stream) {
        size_t offset = (size_t)stream * width;
        double h2 = 0, k2 = 0, dot = 0;
        for (uint32_t i = 0; i < width; ++i) {
            double h = residual[offset + i], k = bf16(kv[offset + i]);
            float weights = q_weight[offset + i] * k_weight[offset + i];
            h2 += h * h;
            k2 += k * k;
            dot += h * weights * k;
        }
        dot /= sqrt(h2 / width + eps) * sqrt(k2 / width + eps) * sqrt((double)width);
        float score = (float)dot;
        float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(score), 1e-6f)), score)));
        for (uint32_t i = 0; i < width; ++i)
            residual[offset + i] = bf16(residual[offset + i] + gate * bf16(kv[(size_t)4 * width + i]));
    }
    return true;
}
