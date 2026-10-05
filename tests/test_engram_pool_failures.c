#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "myds_engram.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Only the separately compiled myds_engram.c object redirects these APIs.
 * The wrappers call the real pthread/pread functions; production has no hook.
 * Initialization/destruction occurs on the test thread, so resource counters
 * need no synchronization. Physical I/O counters are shared by pool workers. */
static unsigned create_calls, created, joined, fail_create;
static unsigned mutex_calls, mutex_live, fail_mutex;
static unsigned cond_calls, cond_live, fail_cond;
static atomic_bool interrupt_once;
static atomic_bool interrupt_selected_once;
static atomic_int_fast64_t interrupt_offset;
static atomic_bool short_reads;
static pthread_mutex_t io_audit = PTHREAD_MUTEX_INITIALIZER;
static unsigned io_calls;
static size_t io_payload;

int myds_test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                             void *(*start)(void *), void *arg) {
    if (++create_calls == fail_create) return EAGAIN;
    int result = pthread_create(thread, attr, start, arg);
    if (!result) ++created;
    return result;
}

int myds_test_pthread_join(pthread_t thread, void **value) {
    int result = pthread_join(thread, value);
    if (!result) ++joined;
    return result;
}

int myds_test_pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr) {
    if (++mutex_calls == fail_mutex) return ENOMEM;
    int result = pthread_mutex_init(mutex, attr);
    if (!result) ++mutex_live;
    return result;
}

int myds_test_pthread_mutex_destroy(pthread_mutex_t *mutex) {
    int result = pthread_mutex_destroy(mutex);
    assert(!result && mutex_live);
    --mutex_live;
    return result;
}

int myds_test_pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr) {
    if (++cond_calls == fail_cond) return ENOMEM;
    int result = pthread_cond_init(cond, attr);
    if (!result) ++cond_live;
    return result;
}

int myds_test_pthread_cond_destroy(pthread_cond_t *cond) {
    int result = pthread_cond_destroy(cond);
    assert(!result && cond_live);
    --cond_live;
    return result;
}

ssize_t myds_test_pread(int fd, void *buffer, size_t count, off_t offset) {
    assert(pthread_mutex_lock(&io_audit) == 0);
    ++io_calls;
    assert(pthread_mutex_unlock(&io_audit) == 0);
    if (atomic_exchange(&interrupt_once, false) ||
        (offset == atomic_load(&interrupt_offset) && atomic_exchange(&interrupt_selected_once, false))) {
        errno = EINTR; return -1;
    }
    if (atomic_load(&short_reads) && count > 37) count = 37;
    ssize_t result = pread(fd, buffer, count, offset);
    if (result > 0) {
        assert(pthread_mutex_lock(&io_audit) == 0);
        io_payload += (size_t)result;
        assert(pthread_mutex_unlock(&io_audit) == 0);
    }
    return result;
}

float myds_test_ldexpf(float value, int exponent) {
    float result = ldexpf(value, exponent);
    /* macOS does not report ERANGE for these exactly representable finite
     * F32 subnormals. Preserve the mathematical result and inject a portable
     * libm range notification, testing errno ordering independently of libc. */
    if (value == 1.0f && exponent == -127) errno = ERANGE;
    return result;
}

static void reset_attempts(void) {
    create_calls = mutex_calls = cond_calls = 0;
    fail_create = fail_mutex = fail_cond = 0;
}

static void assert_empty(void) {
    assert(created == joined && !mutex_live && !cond_live);
}

static unsigned count_low_fds(void) {
    unsigned count = 0;
    for (int fd = 0; fd < 1024; ++fd)
        if (fcntl(fd, F_GETFD) >= 0) ++count;
    return count;
}

static myds_engram_t *fixture(int fd) {
    static const uint32_t map[] = {0, 3, 5, 7, 2};
    myds_engram_t *e = calloc(1, sizeof(*e));
    assert(e);
    e->layout = (myds_engram_layout_t){
        .token_map = map, .vocab_size = 5, .compressed_vocab_size = 8, .pad_id = 1,
        .rows = {264, 264}, .multipliers = {{1, 3, 5, 7}, {11, 13, 17, 19}},
    };
    for (unsigned layer = 0; layer < 2; ++layer) {
        for (unsigned col = 0; col < MYDS_ENGRAM_COLS; ++col)
            e->layout.primes[layer][col] = 11;
        e->table[layer] = (myds_engram_table_t){
            .fd = dup(fd), .rows = 264,
            .offset = (uint64_t)layer * 264 * MYDS_ENGRAM_ROW_BYTES,
        };
        assert(e->table[layer].fd >= 0 && e->table[layer].fd < 1024);
    }
    e->io_threads = 1;
    myds_engram_reset(e);
    return e;
}

static void write_row(int fd, off_t offset, uint8_t code, uint8_t scale) {
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    memset(raw, code, MYDS_ENGRAM_DIM);
    memset(raw + MYDS_ENGRAM_DIM, scale, MYDS_ENGRAM_ROW_BYTES - MYDS_ENGRAM_DIM);
    assert(pwrite(fd, raw, sizeof(raw), offset) == (ssize_t)sizeof(raw));
}

static void test_retry_errno(myds_engram_t *e, int fd) {
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    const uint8_t underflow_code = 0x38;
    float one[MYDS_ENGRAM_DIM];
    /* The isolated libm wrapper supplies the range notification; values are
     * unchanged. The regular test exhaustively verifies the real math path. */
    memset(raw + MYDS_ENGRAM_DIM, 0, MYDS_ENGRAM_ROW_BYTES - MYDS_ENGRAM_DIM);
    memset(raw, underflow_code, MYDS_ENGRAM_DIM); errno = EBUSY;
    assert(myds_engram_decode_row(raw, one) && errno == ERANGE);
    uint32_t ids[2][MYDS_ENGRAM_COLS];
    float expected[2][MYDS_ENGRAM_INPUT_DIM], actual[2][MYDS_ENGRAM_INPUT_DIM];
    for (unsigned col = 0; col < MYDS_ENGRAM_COLS; ++col) {
        ids[0][col] = col & 1u;
        ids[1][col] = 2;
    }
    for (unsigned early_underflow = 0; early_underflow < 2; ++early_underflow) {
        write_row(fd, 0, early_underflow ? underflow_code : 0x38, early_underflow ? 0 : 127);
        write_row(fd, MYDS_ENGRAM_ROW_BYTES, early_underflow ? 0x38 : underflow_code,
                  early_underflow ? 127 : 0);
        write_row(fd, (off_t)(e->table[1].offset + 2 * MYDS_ENGRAM_ROW_BYTES), 0x38, 127);
        off_t interrupted = early_underflow ? MYDS_ENGRAM_ROW_BYTES : 0;
        atomic_store(&interrupt_offset, interrupted);
        atomic_store(&interrupt_selected_once, true);
        errno = EBUSY;
        assert(myds_engram_reader_read_token(NULL, e->table, ids, expected));
        int oracle_errno = errno;
        assert(oracle_errno == (early_underflow ? EINTR : ERANGE));
        atomic_store(&interrupt_selected_once, true);
        errno = EBUSY;
        assert(myds_engram_reader_read_token(e->reader, e->table, ids, actual));
        assert(errno == oracle_errno && memcmp(expected, actual, sizeof(actual)) == 0);
    }
    puts("Engram pool: EINTR/injected-libm-ERANGE follow logical order; duplicate rows do not replay errno");
}

typedef struct {
    myds_engram_reader_t *reader;
    const myds_engram_table_t *tables;
    uint32_t ids[2][MYDS_ENGRAM_COLS];
    float expected[2][MYDS_ENGRAM_INPUT_DIM];
    myds_engram_t *context, *oracle;
    unsigned id;
} caller_t;

static void *concurrent_caller(void *opaque) {
    caller_t *caller = opaque;
    for (unsigned iteration = 0; iteration < 64; ++iteration) {
        if (caller->context) {
            int32_t token = (int32_t)((iteration + caller->id) % 5);
            assert(myds_engram_prepare_token(caller->oracle, token, iteration));
            assert(myds_engram_prepare_token(caller->context, token, iteration));
            assert(caller->context->prepared && caller->context->position == caller->oracle->position);
            assert(memcmp(&caller->context->history, &caller->oracle->history,
                          sizeof(caller->context->history)) == 0);
            assert(memcmp(caller->context->row_ids, caller->oracle->row_ids,
                          sizeof(caller->context->row_ids)) == 0);
            assert(memcmp(caller->context->row_values, caller->oracle->row_values,
                          sizeof(caller->context->row_values)) == 0);
        } else {
            float actual[2][MYDS_ENGRAM_INPUT_DIM];
            assert(myds_engram_reader_read_token(caller->reader, caller->tables, caller->ids, actual));
            assert(memcmp(caller->expected, actual, sizeof(actual)) == 0);
        }
    }
    return NULL;
}

static void test_concurrent_callers(myds_engram_t *e, int fd) {
    for (unsigned layer = 0; layer < 2; ++layer)
        for (unsigned id = 0; id < e->table[layer].rows; ++id)
            write_row(fd, (off_t)(e->table[layer].offset + (uint64_t)id * MYDS_ENGRAM_ROW_BYTES),
                      (uint8_t)(0x38 + id % 7), (uint8_t)(124 + layer * 3 + id % 5));
    caller_t *callers = calloc(2, sizeof(*callers));
    assert(callers);
    pthread_t threads[2];
    for (unsigned id = 0; id < 2; ++id) {
        callers[id].reader = e->reader; callers[id].tables = e->table;
        for (unsigned layer = 0; layer < 2; ++layer)
            for (unsigned col = 0; col < MYDS_ENGRAM_COLS; ++col)
                callers[id].ids[layer][col] = (col * 13 + layer * 7 + id * 11) % 31;
        assert(myds_engram_reader_read_token(NULL, e->table, callers[id].ids, callers[id].expected));
    }
    for (unsigned id = 0; id < 2; ++id)
        assert(pthread_create(&threads[id], NULL, concurrent_caller, callers + id) == 0);
    for (unsigned id = 0; id < 2; ++id) assert(pthread_join(threads[id], NULL) == 0);
    for (unsigned id = 0; id < 2; ++id) {
        callers[id].context = myds_engram_clone(e);
        callers[id].oracle = myds_engram_clone(e);
        callers[id].id = id;
        assert(callers[id].context && callers[id].oracle);
        assert(myds_engram_set_io_threads(callers[id].oracle, 1));
    }
    assert(callers[0].context->reader != callers[1].context->reader);
    for (unsigned id = 0; id < 2; ++id)
        assert(pthread_create(&threads[id], NULL, concurrent_caller, callers + id) == 0);
    for (unsigned id = 0; id < 2; ++id) assert(pthread_join(threads[id], NULL) == 0);
    for (unsigned id = 0; id < 2; ++id) {
        myds_engram_free(callers[id].context); myds_engram_free(callers[id].oracle);
    }
    free(callers);
    puts("Engram pool: two concurrent reader callers and two independently advancing clones match serial oracles");
}

int main(void) {
    /* Fail each stage, including after several real workers have started. */
    for (unsigned attempt = 1; attempt <= 4; ++attempt) {
        reset_attempts(); fail_create = attempt;
        unsigned old_created = created, old_joined = joined;
        assert(!myds_engram_reader_create(4) && errno == EAGAIN);
        assert(created - old_created == attempt - 1);
        assert(joined - old_joined == attempt - 1);
        assert_empty();
    }
    for (unsigned attempt = 1; attempt <= 2; ++attempt) {
        reset_attempts(); fail_mutex = attempt;
        assert(!myds_engram_reader_create(4) && errno == ENOMEM);
        assert_empty();
        reset_attempts(); fail_cond = attempt;
        assert(!myds_engram_reader_create(4) && errno == ENOMEM);
        assert_empty();
    }
    puts("Engram pool: every mutex/condition/worker initialization failure releases acquired resources");

    char path[] = "/tmp/myds-engram-pool-failure.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0 && ftruncate(fd, 2 * 264 * MYDS_ENGRAM_ROW_BYTES) == 0);
    myds_engram_t *e = fixture(fd);
    reset_attempts();
    assert(myds_engram_set_io_threads(e, 4));
    myds_engram_reader_t *original = e->reader;
    for (unsigned attempt = 1; attempt <= 8; ++attempt) {
        reset_attempts(); fail_create = attempt;
        assert(!myds_engram_set_io_threads(e, 8) && errno == EAGAIN);
        assert(e->reader == original && e->io_threads == 4);
        assert(mutex_live == 2 && cond_live == 2 && created - joined == 4);
        reset_attempts();
        myds_engram_reset(e);
        assert(myds_engram_prepare_token(e, 1, 0));
    }
    for (unsigned attempt = 1; attempt <= 4; ++attempt) {
        reset_attempts(); fail_create = attempt;
        unsigned before = count_low_fds();
        assert(!myds_engram_clone(e) && errno == EAGAIN);
        assert(count_low_fds() == before);
        assert(e->reader == original && mutex_live == 2 && cond_live == 2 && created - joined == 4);
        reset_attempts();
        myds_engram_reset(e);
        assert(myds_engram_prepare_token(e, 2, 0));
    }
    puts("Engram pool: failed replacement preserves its working pool; failed clone joins workers and closes duplicate fds");

    uint32_t ids[2][MYDS_ENGRAM_COLS];
    float out[2][MYDS_ENGRAM_INPUT_DIM];
    for (unsigned layer = 0; layer < 2; ++layer)
        for (unsigned col = 0; col < MYDS_ENGRAM_COLS; ++col) ids[layer][col] = col % 3;
    io_calls = 0; io_payload = 0;
    atomic_store(&short_reads, true); atomic_store(&interrupt_once, true);
    errno = EBUSY;
    assert(myds_engram_reader_read_token(e->reader, e->table, ids, out));
    assert(errno == EINTR);
    /* Six unique physical rows, eight positive 37-byte-or-smaller reads each,
     * plus exactly one injected EINTR retry. No duplicate row I/O. */
    assert(io_payload == 6 * MYDS_ENGRAM_ROW_BYTES && io_calls == 6 * 8 + 1);
    for (unsigned layer = 0; layer < 2; ++layer)
        for (unsigned i = 0; i < MYDS_ENGRAM_INPUT_DIM; ++i) assert(out[layer][i] == 0.0f);
    atomic_store(&short_reads, false);
    test_retry_errno(e, fd);
    test_concurrent_callers(e, fd);
    myds_engram_free(e); close(fd); unlink(path);
    assert_empty();
    puts("Engram pool: positive short reads/EINTR retry, exact duplicate-row I/O and final resource balance passed");
    return 0;
}
