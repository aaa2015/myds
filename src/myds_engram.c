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
        size_t done = 0;
        while (done < sizeof(raw)) {
            ssize_t n = pread(table->fd, raw + done, sizeof(raw) - done, (off_t)(offset + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return n == 0 ? fail(EIO) : false;
            done += (size_t)n;
        }
        if (!myds_engram_decode_row(raw, out + i * MYDS_ENGRAM_DIM)) return false;
    }
    return true;
}

enum { ENGRAM_IO_MAX_THREADS = 16, ENGRAM_IO_JOBS = MYDS_ENGRAM_LAYERS * MYDS_ENGRAM_COLS };
typedef struct {
    int fd, error, retry_errno;
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
};

static void read_raw_job(engram_io_job_t *job) {
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
        for (unsigned i = worker->index; i < count; i += r->workers)
            read_raw_job(&r->jobs[i]);
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
    return true;
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
    e->io_threads = configured_io_threads();
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
    struct stat model_stat;
    if (fstat(gf->fd, &model_stat) != 0) goto error;
    for (int i = 0; i < MYDS_ENGRAM_LAYERS; ++i) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%u.engram_embd.weight", e->layer_ids[i]);
        const gguf_tensor_t *table = gguf_find_tensor(gf, name);
        if (!shape(table, GGML_TYPE_I8, MYDS_ENGRAM_ROW_BYTES, e->layout.rows[i]) ||
            table->offset > UINT64_MAX - gf->tensor_data_offset) goto invalid;
        if (!myds_engram_table_open(&e->table[i], path, gf->tensor_data_offset + table->offset,
                                   e->layout.rows[i])) goto error;
        struct stat table_stat;
        if (fstat(e->table[i].fd, &table_stat) != 0) goto error;
        if (model_stat.st_dev != table_stat.st_dev || model_stat.st_ino != table_stat.st_ino)
            goto invalid;
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
}

bool myds_engram_prepare_token(myds_engram_t *e, int32_t token, uint64_t position) {
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
