#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "myds_engram.h"

#include <assert.h>
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
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

static const uint32_t token_map[] = {0, 3, 5, 7, 2};

static myds_engram_layout_t layout(void) {
    myds_engram_layout_t l = {
        .token_map = token_map, .vocab_size = 5, .compressed_vocab_size = 8, .pad_id = 1,
        .rows = {264, 264}, .multipliers = {{1, 3, 5, 7}, {11, 13, 17, 19}},
    };
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 24; ++j) l.primes[i][j] = 11;
    return l;
}

static void test_hash(void) {
    myds_engram_layout_t l = layout();
    assert(myds_engram_layout_valid(&l));
    /* Independent, fixed XOR/modulo results, before column partition offsets.
     * Covers 2-, 3- and 4-grams in both layers, including padding and a gap. */
    static const uint32_t expected[2][4][6] = {
        {{0,5,2,0,6,2}, {1,9,3,5,1,7}, {8,7,0,1,8,0}, {1,3,5,0,2,0}},
        {{0,5,2,0,6,2}, {2,7,0,6,1,4}, {4,1,6,9,4,0}, {1,7,10,0,4,2}},
    };
    int32_t tokens[] = {1, 2, 3, 4};
    uint8_t mask[] = {1, 0, 1, 1};
    uint32_t rows[4][2][24], single[4][2][24];
    for (int masked = 0; masked < 2; ++masked) {
        myds_engram_history_t history, one;
        myds_engram_history_reset(&history);
        myds_engram_history_reset(&one);
        assert(myds_engram_hash(&l, &history, tokens, masked ? mask : NULL, 4, &rows[0][0][0]));
        for (int token = 0; token < 4; ++token) {
            assert(myds_engram_hash(&l, &one, tokens + token, masked ? mask + token : NULL,
                                    1, &single[token][0][0]));
            for (int layer = 0; layer < 2; ++layer)
                for (int column = 0; column < 24; ++column)
                    assert(rows[token][layer][column] ==
                           expected[masked][token][layer * 3 + column / 8] + 11u * column);
        }
        assert(memcmp(rows, single, sizeof(rows)) == 0);
        assert(memcmp(&history, &one, sizeof(one)) == 0);
        myds_engram_history_t saved = history;
        int32_t invalid[] = {1, 99};
        assert(!myds_engram_hash(&l, &history, invalid, NULL, 2, &rows[0][0][0]));
        assert(memcmp(&history, &saved, sizeof(saved)) == 0);
    }
    l.multipliers[1][1] = 2;
    assert(!myds_engram_layout_valid(&l));
    l = layout(); l.rows[0] = 263;
    assert(!myds_engram_layout_valid(&l));
    l = layout(); l.pad_id = 8;
    assert(!myds_engram_layout_valid(&l));
    printf("Engram: known-answer hashes, chunking, masked boundaries, invalid-input transaction passed\n");
}

static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, sizeof(u)); return u; }

static void row(uint8_t *raw, uint8_t value, uint8_t scale) {
    memset(raw, value, 256);
    memset(raw + 256, scale, 8);
}

/* Independent copy of the pre-LUT scalar contract. This intentionally uses
 * libm and explicit F32 bit rounding rather than the implementation's table. */
static bool decode_scalar(const uint8_t *raw, float *out) {
    if (!raw || !out) { errno = EINVAL; return false; }
    for (int i = 0; i < MYDS_ENGRAM_DIM; ++i) {
        const unsigned code = raw[i], scale = raw[MYDS_ENGRAM_DIM + i / 32];
        if ((code & 127u) == 127u || scale == 255u) { errno = EDOM; return false; }
        const int exponent = (code >> 3) & 15, fraction = code & 7;
        float value = exponent ? ldexpf((float)(8 + fraction), exponent - 10) :
                                 ldexpf((float)fraction, -9);
        if (code & 128u) value = -value;
        value = ldexpf(value, (int)scale - 127);
        if (!isfinite(value)) { errno = EDOM; return false; }
        uint32_t u = bits(value);
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
        memcpy(&value, &u, sizeof(value));
        if (!isfinite(value)) { errno = EDOM; return false; }
        out[i] = value;
    }
    return true;
}

static void *decode_first_reader(void *arg) {
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    float out[MYDS_ENGRAM_DIM];
    const unsigned id = (unsigned)(uintptr_t)arg;
    row(raw, id & 1u ? 0xb8 : 0x38, (uint8_t)(127 + id % 3));
    assert(feclearexcept(FE_ALL_EXCEPT) == 0);
    assert(feraiseexcept(FE_INVALID) == 0);
    errno = EBUSY;
    assert(myds_engram_decode_row(raw, out) && errno == EBUSY);
    assert(fetestexcept(FE_ALL_EXCEPT) == FE_INVALID);
    assert(feclearexcept(FE_ALL_EXCEPT) == 0);
    const float expected = (id & 1u ? -1.0f : 1.0f) * (float)(1u << (id % 3));
    for (int i = 0; i < MYDS_ENGRAM_DIM; ++i) assert(bits(out[i]) == bits(expected));
    return NULL;
}

static void test_decode_table(void) {
    /* Exercise concurrent first use before any other decode test. */
    pthread_t readers[8];
    for (unsigned i = 0; i < 8; ++i)
        assert(pthread_create(&readers[i], NULL, decode_first_reader, (void *)(uintptr_t)i) == 0);
    for (unsigned i = 0; i < 8; ++i) assert(pthread_join(readers[i], NULL) == 0);
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    float actual[MYDS_ENGRAM_DIM], expected[MYDS_ENGRAM_DIM];
    for (unsigned scale = 0; scale < 256; ++scale) {
        for (unsigned code = 0; code < 256; ++code) {
            row(raw, (uint8_t)code, (uint8_t)scale);
            memset(actual, 0x5a, sizeof(actual));
            memset(expected, 0x5a, sizeof(expected));
            errno = EBUSY;
            const bool oracle_ok = decode_scalar(raw, expected);
            const int oracle_errno = errno;
            errno = EBUSY;
            assert(myds_engram_decode_row(raw, actual) == oracle_ok);
            assert(errno == oracle_errno);
            assert(memcmp(actual, expected, sizeof(actual)) == 0);
        }
    }
    /* Mixed rows ensure each scale is selected by its own 32-value block.
     * Also match partial output and EDOM when corruption appears late. */
    for (unsigned trial = 0; trial < 512; ++trial) {
        for (unsigned block = 0; block < 8; ++block) {
            raw[256 + block] = (uint8_t)((trial * 17u + block * 31u) % 255u);
            for (unsigned j = 0; j < 32; ++j) {
                unsigned code = (trial * 13u + block * 29u + j * 7u) & 255u;
                uint8_t one[MYDS_ENGRAM_ROW_BYTES];
                row(one, (uint8_t)code, raw[256 + block]);
                if (!decode_scalar(one, expected)) code = j & 1u ? 128u : 0u;
                raw[block * 32 + j] = (uint8_t)code;
            }
        }
        errno = EBUSY;
        assert(decode_scalar(raw, expected));
        const int oracle_errno = errno;
        errno = EBUSY;
        assert(myds_engram_decode_row(raw, actual) && errno == oracle_errno);
        assert(memcmp(actual, expected, sizeof(actual)) == 0);
        raw[255] = 127;
        memset(actual, 0x5a, sizeof(actual)); memset(expected, 0x5a, sizeof(expected));
        assert(!decode_scalar(raw, expected));
        assert(!myds_engram_decode_row(raw, actual) && errno == EDOM);
        assert(memcmp(actual, expected, sizeof(actual)) == 0);
    }
    assert(!myds_engram_decode_row(NULL, actual) && errno == EINVAL);
    assert(!myds_engram_decode_row(raw, NULL) && errno == EINVAL);
    puts("Engram: concurrent LUT initialization, all 65536 code/scale pairs, mixed rows and errno parity passed");
}

static void test_decode_rounding_modes(void) {
    fenv_t saved_env;
    assert(fegetenv(&saved_env) == 0);
    static const int modes[] = {FE_DOWNWARD, FE_UPWARD};
    static const struct { uint8_t code, scale; } cases[] = {
        {0x01,0}, {0x04,0}, {0x05,0}, {0x84,0}, {0x85,0},
        {0x38,127}, {0x7e,254}, {0xfe,254}, {0x7f,127}, {0x38,255},
    };
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    float actual[MYDS_ENGRAM_DIM], expected[MYDS_ENGRAM_DIM];
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); ++m) {
        assert(fesetround(modes[m]) == 0);
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
            row(raw, cases[c].code, cases[c].scale);
            memset(expected, 0x5a, sizeof(expected)); memset(actual, 0x5a, sizeof(actual));
            assert(feclearexcept(FE_ALL_EXCEPT) == 0);
            assert(feraiseexcept(FE_INVALID) == 0);
            errno = EBUSY;
            const bool oracle_ok = decode_scalar(raw, expected);
            const int oracle_errno = errno, oracle_flags = fetestexcept(FE_ALL_EXCEPT);
            assert(feclearexcept(FE_ALL_EXCEPT) == 0);
            assert(feraiseexcept(FE_INVALID) == 0);
            errno = EBUSY;
            assert(myds_engram_decode_row(raw, actual) == oracle_ok);
            assert(errno == oracle_errno && fetestexcept(FE_ALL_EXCEPT) == oracle_flags);
            assert(fegetround() == modes[m] && memcmp(actual, expected, sizeof(actual)) == 0);
        }
    }
    assert(fesetenv(&saved_env) == 0);
    puts("Engram: downward/upward scalar fallback preserves values, errors and caller floating-point environment");
}

static void test_decode(void) {
    uint8_t raw[264];
    float out[256];
    static const struct { uint8_t code, scale; uint32_t expected; } known[] = {
        {0x00,127,0x00000000}, {0x80,127,0x80000000},
        {0x01,127,0x3b000000}, {0x07,127,0x3c600000},
        {0x08,127,0x3c800000}, {0x38,127,0x3f800000},
        {0xb8,127,0xbf800000}, {0x7e,127,0x43e00000},
        {0xfe,127,0xc3e00000}, {0x38,128,0x40000000},
        {0x38,126,0x3f000000}, {0x38,0,0x00400000},
        {0x04,0,0x00000000}, {0x05,0,0x00010000},
    };
    for (size_t k = 0; k < sizeof(known) / sizeof(known[0]); ++k) {
        row(raw, known[k].code, known[k].scale);
        assert(myds_engram_decode_row(raw, out));
        for (int i = 0; i < 256; ++i) assert(bits(out[i]) == known[k].expected);
    }
    /* All finite E4M3 values have only four significant bits and are exactly
     * representable in BF16 at scale 1. Mathematical oracle uses frexp form. */
    row(raw, 0, 127);
    for (unsigned code = 0; code < 256; ++code) {
        memset(raw, code, 256);
        bool valid = (code & 127) != 127;
        assert(myds_engram_decode_row(raw, out) == valid);
        if (!valid) continue;
        unsigned exponent = (code & 127) / 8, fraction = code & 7;
        float expected = exponent ? (1.0f + fraction / 8.0f) * powf(2.0f, (int)exponent - 7) :
                                    fraction * (1.0f / 512.0f);
        if (code & 128) expected = -expected;
        assert(bits(out[0]) == bits(expected));
    }
    row(raw, 0x38, 255);
    assert(!myds_engram_decode_row(raw, out) && errno == EDOM);
    row(raw, 0x7e, 254);
    assert(!myds_engram_decode_row(raw, out) && errno == EDOM);
    row(raw, 0x38, 127);
    raw[256] = 128;
    assert(myds_engram_decode_row(raw, out));
    assert(out[0] == 2 && out[31] == 2 && out[32] == 1);
    printf("Engram: all E4M3 bytes, E8M0 scales, BF16 underflow, NaN/overflow rejection passed\n");
}

static void test_disk(void) {
    char path[] = "/tmp/myds-engram-test.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    const uint64_t offset = UINT64_C(1) << 30;
    assert(ftruncate(fd, (off_t)(offset + 4 * 264)) == 0);
    uint8_t raw[264];
    for (int i = 0; i < 4; ++i) {
        row(raw, 0x38, (uint8_t)(127 + i));
        assert(pwrite(fd, raw, sizeof(raw), (off_t)(offset + (uint64_t)i * 264)) == sizeof(raw));
    }
    struct stat st;
    assert(fstat(fd, &st) == 0 && (uint64_t)st.st_blocks * 512 < (1u << 20));
    myds_engram_table_t table;
    assert(myds_engram_table_open(&table, path, offset, 4));
    uint32_t ids[] = {2, 0, 2};
    float values[3 * 256];
    assert(myds_engram_read(&table, ids, 3, values));
    for (int i = 0; i < 256; ++i) {
        assert(values[i] == 4 && values[256 + i] == 1 && values[512 + i] == 4);
    }
    ids[0] = 4;
    assert(!myds_engram_read(&table, ids, 1, values) && errno == EINVAL);
    ids[0] = 3;
    assert(ftruncate(fd, (off_t)(offset + 3 * 264 + 80)) == 0);
    assert(!myds_engram_read(&table, ids, 1, values) && errno == EIO);
    myds_engram_table_close(&table);
    assert(table.fd == -1);
    assert(!myds_engram_table_open(&table, path, offset, 4));
    assert(!myds_engram_table_open(&table, path, UINT64_MAX, 4));
    close(fd);
    unlink(path);
    printf("Engram: sparse 1-GiB offset row reads, deduplication, bounds, truncated row failure passed\n");
}

/* Distinct finite encodings in every row and 32-value block. Unlike a zero
 * fixture, this detects reordered columns, swapped tables and stale requests.
 * The mathematical expected values come from decode_scalar above. */
static void patterned_row(uint8_t raw[MYDS_ENGRAM_ROW_BYTES], unsigned layer, unsigned id) {
    for (unsigned block = 0; block < 8; ++block) {
        raw[MYDS_ENGRAM_DIM + block] = (uint8_t)(120 + (id * 3 + layer * 5 + block) % 15);
        for (unsigned j = 0; j < 32; ++j) {
            unsigned code = (id * 29 + layer * 71 + block * 11 + j * 7) % 127;
            raw[block * 32 + j] = (uint8_t)(code | (((id + layer + block + j) & 1u) << 7));
        }
    }
}

static void write_patterned_tables(int fd, const myds_engram_table_t tables[2]) {
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    uint64_t end = tables[1].offset + (uint64_t)tables[1].rows * sizeof(raw);
    assert(ftruncate(fd, (off_t)end) == 0);
    for (unsigned layer = 0; layer < 2; ++layer)
        for (uint32_t id = 0; id < tables[layer].rows; ++id) {
            patterned_row(raw, layer, id);
            off_t off = (off_t)(tables[layer].offset + (uint64_t)id * sizeof(raw));
            assert(pwrite(fd, raw, sizeof(raw), off) == (ssize_t)sizeof(raw));
        }
}

static void test_parallel_disk(void) {
    char path[] = "/tmp/myds-engram-parallel.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    const uint64_t offset = (UINT64_C(1) << 30) + 3;
    myds_engram_table_t tables[2] = {
        {.fd = fd, .offset = offset, .rows = 64},
        {.fd = fd, .offset = offset + 64 * MYDS_ENGRAM_ROW_BYTES + 19, .rows = 64},
    };
    write_patterned_tables(fd, tables);
    for (unsigned layer = 0; layer < 2; ++layer)
        assert(myds_engram_table_open(&tables[layer], path, tables[layer].offset, tables[layer].rows));
    uint32_t ids[2][MYDS_ENGRAM_COLS];
    float expected[2][MYDS_ENGRAM_INPUT_DIM], actual[2][MYDS_ENGRAM_INPUT_DIM];
    uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
    for (unsigned layer = 0; layer < 2; ++layer)
        for (unsigned column = 0; column < MYDS_ENGRAM_COLS; ++column) {
            /* Repeated, shuffled IDs exercise deduplication and output order. */
            ids[layer][column] = (column * 13 + layer * 7) % 19;
            patterned_row(raw, layer, ids[layer][column]);
            assert(decode_scalar(raw, expected[layer] + column * MYDS_ENGRAM_DIM));
        }
    assert(myds_engram_reader_read_token(NULL, tables, ids, actual));
    assert(memcmp(expected, actual, sizeof(actual)) == 0);
    static const unsigned workers[] = {1, 2, 4, 8, 16};
    for (size_t w = 0; w < sizeof(workers) / sizeof(workers[0]); ++w) {
        myds_engram_reader_t *reader = myds_engram_reader_create(workers[w]);
        assert(reader);
        for (unsigned iteration = 0; iteration < 32; ++iteration) {
            errno = EBUSY;
            assert(myds_engram_reader_read_token(reader, tables, ids, actual));
            assert(errno == EBUSY && memcmp(expected, actual, sizeof(actual)) == 0);
        }
        /* Decode still runs with the caller's rounding mode and sticky flags. */
        fenv_t saved_env;
        assert(fegetenv(&saved_env) == 0 && fesetround(FE_UPWARD) == 0);
        assert(feclearexcept(FE_ALL_EXCEPT) == 0 && feraiseexcept(FE_INVALID) == 0);
        assert(myds_engram_reader_read_token(reader, tables, ids, actual));
        assert(fegetround() == FE_UPWARD && fetestexcept(FE_ALL_EXCEPT) == FE_INVALID);
        assert(memcmp(expected, actual, sizeof(actual)) == 0);
        assert(fesetenv(&saved_env) == 0);

        /* The first table validates its complete row list before any writes.
         * Output after a later-table failure is invalid, as in serial reads. */
        uint32_t invalid[2][MYDS_ENGRAM_COLS];
        memcpy(invalid, ids, sizeof(invalid)); invalid[0][23] = tables[0].rows;
        memset(actual, 0x5a, sizeof(actual));
        float untouched[2][MYDS_ENGRAM_INPUT_DIM]; memcpy(untouched, actual, sizeof(actual));
        assert(!myds_engram_reader_read_token(reader, tables, invalid, actual) && errno == EINVAL);
        assert(memcmp(untouched, actual, sizeof(actual)) == 0);
        memcpy(invalid, ids, sizeof(invalid)); invalid[1][23] = tables[1].rows;
        assert(!myds_engram_reader_read_token(reader, tables, invalid, actual) && errno == EINVAL);

        /* Actual partial pread followed by EOF; the same pool must recover. */
        uint64_t short_end = tables[1].offset + (uint64_t)ids[1][0] * sizeof(raw) + 80;
        assert(ftruncate(fd, (off_t)short_end) == 0);
        assert(!myds_engram_reader_read_token(reader, tables, ids, actual) && errno == EIO);

        /* Both faults exist concurrently. Serial logical order selects the
         * earlier corrupt encoding, even if a later EOF finishes first. */
        patterned_row(raw, 0, ids[0][0]); raw[255] = 127;
        off_t first = (off_t)(tables[0].offset + (uint64_t)ids[0][0] * sizeof(raw));
        assert(pwrite(fd, raw, sizeof(raw), first) == (ssize_t)sizeof(raw));
        assert(!myds_engram_reader_read_token(reader, tables, ids, actual) && errno == EDOM);
        /* Malformed later input cannot hide an earlier decode error either. */
        assert(!myds_engram_reader_read_token(reader, tables, invalid, actual) && errno == EDOM);
        myds_engram_table_t later_bad[2]; memcpy(later_bad, tables, sizeof(later_bad));
        later_bad[1].fd = -1;
        assert(!myds_engram_reader_read_token(reader, later_bad, ids, actual) && errno == EDOM);
        write_patterned_tables(fd, tables);
        assert(myds_engram_reader_read_token(reader, tables, ids, actual));
        assert(memcmp(expected, actual, sizeof(actual)) == 0);

        /* A positive descriptor number that has been closed reaches pread,
         * so EBADF must be transferred back from a worker thread. */
        myds_engram_table_t bad[2]; memcpy(bad, tables, sizeof(bad));
        int closed_fd = dup(tables[0].fd);
        assert(closed_fd >= 0 && close(closed_fd) == 0); bad[0].fd = closed_fd;
        assert(!myds_engram_reader_read_token(reader, bad, ids, actual) && errno == EBADF);
        assert(myds_engram_reader_read_token(reader, tables, ids, actual));
        assert(memcmp(expected, actual, sizeof(actual)) == 0);
        myds_engram_reader_free(reader);
    }
    assert(!myds_engram_reader_create(0) && errno == EINVAL);
    assert(!myds_engram_reader_create(17) && errno == EINVAL);
    myds_engram_reader_free(NULL);
    /* Repeated lifecycle coverage catches stale requests and unjoined worker
     * access to freed storage under ASan/TSan, without timing-based assertions. */
    for (unsigned i = 0; i < 32; ++i) {
        myds_engram_reader_t *reader = myds_engram_reader_create(4);
        assert(reader && myds_engram_reader_read_token(reader, tables, ids, actual));
        assert(memcmp(expected, actual, sizeof(actual)) == 0);
        myds_engram_reader_free(reader);
    }
    myds_engram_table_close(&tables[0]); myds_engram_table_close(&tables[1]);
    close(fd); unlink(path);
    puts("Engram: serial/1/2/4/8/16-worker byte parity, repeated IDs, ordered errors, EOF/EBADF recovery and lifecycle passed");
}

static void test_gate(void) {
    float residual[16], kv[20], q[16], k[16];
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < 4; ++i) {
            residual[s * 4 + i] = (float[]){1, -1, 0, 2}[s];
            kv[s * 4 + i] = q[s * 4 + i] = k[s * 4 + i] = 1;
        }
    for (int i = 16; i < 20; ++i) kv[i] = 2;
    assert(myds_engram_add_reference(residual, kv, q, k, 4, 1e-20f));
    const float expected[] = {2.609375f, -0.609375f, 1.0f, 3.609375f};
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < 4; ++i) assert(residual[s * 4 + i] == expected[s]);
    kv[19] = NAN;
    float before[16]; memcpy(before, residual, sizeof(before));
    assert(!myds_engram_add_reference(residual, kv, q, k, 4, 1e-20f));
    assert(memcmp(before, residual, sizeof(before)) == 0);
    printf("Engram: independent four-stream signed gate/add known answers passed\n");
}

static void assert_prepared_equal(const myds_engram_t *a, const myds_engram_t *b) {
    assert(a->prepared && b->prepared && a->position == b->position);
    assert(memcmp(&a->history, &b->history, sizeof(a->history)) == 0);
    assert(memcmp(a->row_ids, b->row_ids, sizeof(a->row_ids)) == 0);
    assert(memcmp(a->row_values, b->row_values, sizeof(a->row_values)) == 0);
}

static void test_model_readers(const gguf_file_t *gf, const char *path, int fd) {
    myds_engram_t *serial = myds_engram_create(gf, path, 5, 1);
    assert(serial && myds_engram_set_io_threads(serial, 1));
    write_patterned_tables(fd, serial->table);
    static const int32_t tokens[] = {1, 2, 3, 4, 4, 0, 1, 3, 2, 0, 4, 1};
    static const unsigned workers[] = {1, 2, 4, 8, 16};
    for (size_t w = 0; w < sizeof(workers) / sizeof(workers[0]); ++w) {
        myds_engram_t *parallel = myds_engram_clone(serial);
        assert(parallel && myds_engram_set_io_threads(parallel, workers[w]));
        myds_engram_reader_t *saved_reader = parallel->reader;
        assert(!myds_engram_set_io_threads(parallel, 0) && errno == EINVAL);
        assert(parallel->io_threads == workers[w] && parallel->reader == saved_reader);
        assert(!myds_engram_set_io_threads(parallel, 17) && errno == EINVAL);
        assert(parallel->io_threads == workers[w] && parallel->reader == saved_reader);
        for (unsigned pass = 0; pass < 3; ++pass) {
            myds_engram_reset(serial); myds_engram_reset(parallel);
            for (size_t p = 0; p < sizeof(tokens) / sizeof(tokens[0]); ++p) {
                int32_t token = tokens[(p + pass) % (sizeof(tokens) / sizeof(tokens[0]))];
                assert(myds_engram_prepare_token(serial, token, p));
                assert(myds_engram_prepare_token(parallel, token, p));
                assert_prepared_equal(serial, parallel);
            }
        }

        /* Fresh clone owns descriptors and a pool independently of its source.
         * Advancing or destroying it cannot change the source's request/state. */
        myds_engram_history_t before_clone = parallel->history;
        myds_engram_t *clone = myds_engram_clone(parallel);
        assert(clone && clone->io_threads == parallel->io_threads);
        if (workers[w] > 1) assert(clone->reader && clone->reader != parallel->reader);
        assert(myds_engram_prepare_token(clone, 2, 0));
        assert(memcmp(&before_clone, &parallel->history, sizeof(before_clone)) == 0);
        int clone_fds[2] = {clone->table[0].fd, clone->table[1].fd};
        myds_engram_free(clone);
        for (unsigned layer = 0; layer < 2; ++layer)
            assert(fcntl(clone_fds[layer], F_GETFD) == -1 && errno == EBADF);

        /* Failed duplicate must release its partial ownership while preserving
         * the source pool. The source remains usable after fixing its fd. */
        int valid_fd = parallel->table[1].fd;
        /* A recently closed low fd could be reused by the first duplication. */
        parallel->table[1].fd = INT_MAX;
        assert(!myds_engram_clone(parallel) && errno == EBADF);
        parallel->table[1].fd = valid_fd;

        const uint64_t position = parallel->position;
        const int32_t token = 3;
        myds_engram_history_t before = parallel->history, next = before;
        uint32_t rows[2][MYDS_ENGRAM_COLS];
        assert(myds_engram_hash(&parallel->layout, &next, &token, NULL, 1, &rows[0][0]));
        assert(ftruncate(fd, (off_t)(parallel->table[1].offset +
            (uint64_t)rows[1][0] * MYDS_ENGRAM_ROW_BYTES + 80)) == 0);
        assert(!myds_engram_prepare_token(parallel, token, position) && errno == EIO);
        assert(!parallel->prepared && parallel->position == position);
        assert(memcmp(&before, &parallel->history, sizeof(before)) == 0);
        write_patterned_tables(fd, parallel->table);

        uint8_t raw[MYDS_ENGRAM_ROW_BYTES];
        patterned_row(raw, 1, rows[1][0]); raw[255] = 127;
        off_t corrupt = (off_t)(parallel->table[1].offset +
            (uint64_t)rows[1][0] * sizeof(raw));
        assert(pwrite(fd, raw, sizeof(raw), corrupt) == (ssize_t)sizeof(raw));
        assert(!myds_engram_prepare_token(parallel, token, position) && errno == EDOM);
        assert(!parallel->prepared && parallel->position == position);
        assert(memcmp(&before, &parallel->history, sizeof(before)) == 0);
        write_patterned_tables(fd, parallel->table);

        /* Recover at the original position, with no skipped token or stale
         * rows from either failed request. Both complete outputs must agree. */
        assert(myds_engram_prepare_token(serial, token, position));
        assert(myds_engram_prepare_token(parallel, token, position));
        assert_prepared_equal(serial, parallel);
        int parallel_fds[2] = {parallel->table[0].fd, parallel->table[1].fd};
        myds_engram_free(parallel);
        for (unsigned layer = 0; layer < 2; ++layer)
            assert(fcntl(parallel_fds[layer], F_GETFD) == -1 && errno == EBADF);
    }
    myds_engram_free(serial);
    puts("Engram: token/history/row byte parity, reset, clone ownership, configuration rollback and failure/retry transactions passed");
}

static void test_io_configuration(const gguf_file_t *gf, const char *path) {
    const char *existing = getenv("MYDS_ENGRAM_IO_THREADS");
    char *saved = existing ? strdup(existing) : NULL;
    assert(!existing || saved);
    assert(unsetenv("MYDS_ENGRAM_IO_THREADS") == 0);
    myds_engram_t *e = myds_engram_create(gf, path, 5, 1);
    assert(e && e->io_threads == 8 && !e->reader);
    myds_engram_free(e);
    assert(setenv("MYDS_ENGRAM_IO_THREADS", "1", 1) == 0);
    e = myds_engram_create(gf, path, 5, 1);
    assert(e && e->io_threads == 1 && !e->reader);
    myds_engram_free(e);
    static const char *invalid[] = {"0", "17", "-1", "abc", "8x"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        assert(setenv("MYDS_ENGRAM_IO_THREADS", invalid[i], 1) == 0);
        assert(!myds_engram_create(gf, path, 5, 1) && errno == EINVAL);
    }
    if (saved) { assert(setenv("MYDS_ENGRAM_IO_THREADS", saved, 1) == 0); free(saved); }
    else assert(unsetenv("MYDS_ENGRAM_IO_THREADS") == 0);
    puts("Engram: default/lazy template, explicit serial and invalid environment configuration passed");
}

/* Small descriptor fixture exercises loader contracts and transactional reads
 * without a real model, GPU allocation or mmap of the table. */
static void test_model(void) {
    assert(ggml_type_row_size(GGML_TYPE_I8, 264) == 264);
    assert(strcmp(ggml_type_name(GGML_TYPE_I8), "I8") == 0);
    myds_engram_layout_t l = layout();
    char path[] = "/tmp/myds-engram-model.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    assert(ftruncate(fd, 2 * 264 * 264) == 0);
    uint32_t layer_ids[] = {1, 14};
    gguf_kv_t kvs[8] = {
        {.key="deepseek41.engram.encoding", .type=GGUF_TYPE_STRING,
         .value.val_str="e4m3_e8m0_32_row264"},
        {.key="deepseek41.engram.layer_ids", .type=GGUF_TYPE_ARRAY,
         .value.val_arr={GGUF_TYPE_UINT32,2,layer_ids}},
        {.key="deepseek41.engram.rows", .type=GGUF_TYPE_ARRAY,
         .value.val_arr={GGUF_TYPE_UINT32,2,l.rows}},
        {.key="deepseek41.engram.primes", .type=GGUF_TYPE_ARRAY,
         .value.val_arr={GGUF_TYPE_UINT32,48,l.primes}},
        {.key="deepseek41.engram.multipliers", .type=GGUF_TYPE_ARRAY,
         .value.val_arr={GGUF_TYPE_UINT64,8,l.multipliers}},
        {.key="deepseek41.engram.token_map", .type=GGUF_TYPE_ARRAY,
         .value.val_arr={GGUF_TYPE_UINT32,5,(void *)token_map}},
        {.key="deepseek41.engram.compressed_vocab_size", .type=GGUF_TYPE_UINT32,
         .value.val_u32=8},
        {.key="deepseek41.engram.pad_id", .type=GGUF_TYPE_UINT32, .value.val_u32=1},
    };
    gguf_tensor_t tensors[8] = {0};
    for (int i = 0; i < 2; ++i) {
        gguf_tensor_t *t = tensors + i * 4;
        snprintf(t[0].name, sizeof(t[0].name), "blk.%u.engram_embd.weight", layer_ids[i]);
        t[0].type=GGML_TYPE_I8; t[0].dims[0]=264; t[0].dims[1]=264;
        t[0].size_bytes=264*264; t[0].offset=(uint64_t)i*264*264;
        snprintf(t[1].name, sizeof(t[1].name), "blk.%u.engram_kv.weight", layer_ids[i]);
        t[1].type=GGML_TYPE_F16; t[1].dims[0]=6144; t[1].dims[1]=5; t[1].size_bytes=6144*5*2;
        snprintf(t[2].name, sizeof(t[2].name), "blk.%u.engram_q_norm.weight", layer_ids[i]);
        snprintf(t[3].name, sizeof(t[3].name), "blk.%u.engram_k_norm.weight", layer_ids[i]);
        for (int j = 0; j < 4; ++j) t[j].n_dims=2;
        for (int j = 2; j < 4; ++j) {
            t[j].type=GGML_TYPE_F32; t[j].dims[0]=1; t[j].dims[1]=4; t[j].size_bytes=16;
        }
    }
    gguf_file_t gf = {.fd=fd, .kvs=kvs, .metadata_kv_count=8, .tensors=tensors, .tensor_count=8};
    test_io_configuration(&gf, path);
    test_model_readers(&gf, path, fd);
    myds_engram_t *e = myds_engram_create(&gf, path, 5, 1);
    assert(e && !e->prepared && e->position == 0);
    assert(myds_engram_prepare_token(e, 1, 0) && e->prepared && e->position == 1);
    myds_engram_t *clone = myds_engram_clone(e);
    assert(clone && !clone->prepared && clone->position == 0 && clone->table[0].fd != e->table[0].fd);
    assert(myds_engram_prepare_token(clone, 2, 0) && e->history.tail[0] == 3);
    myds_engram_free(clone);
    assert(!myds_engram_prepare_token(e, 1, 0) && !e->prepared && e->position == 1);
    myds_engram_history_t before = e->history;
    assert(ftruncate(fd, 264*264) == 0);
    assert(!myds_engram_prepare_token(e, 2, 1) && !e->prepared && e->position == 1);
    assert(memcmp(&before, &e->history, sizeof(before)) == 0);
    assert(ftruncate(fd, 2*264*264) == 0);
    assert(myds_engram_prepare_token(e, 2, 1));
    myds_engram_reset(e);
    assert(e->position == 0 && e->history.tail[0] == MYDS_ENGRAM_DEAD);
    myds_engram_free(e);
    kvs[4].value.val_arr.elem_type=GGUF_TYPE_UINT32;
    assert(!myds_engram_create(&gf, path, 5, 1));
    close(fd); unlink(path);
    printf("Engram: metadata/tensor validation, two-table transaction, reset passed\n");
}

static void test_unmap(void) {
    long page = sysconf(_SC_PAGESIZE);
    assert(page > 0);
    char path[] = "/tmp/myds-engram-unmap.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0 && ftruncate(fd, 3 * page) == 0);
    uint8_t *data = mmap(NULL, 3 * (size_t)page, PROT_READ, MAP_SHARED, fd, 0);
    assert(data != MAP_FAILED);
    gguf_tensor_t tensors[3] = {
        {.name="resident", .n_dims=2, .dims={1,1}, .type=GGML_TYPE_F32,
         .offset=0, .data=data, .size_bytes=4},
        {.name="blk.1.engram_embd.weight", .n_dims=2, .dims={264,1}, .type=GGML_TYPE_I8,
         .offset=(uint64_t)page, .data=data+page, .size_bytes=264},
        {.name="blk.14.engram_embd.weight", .n_dims=2, .dims={264,1}, .type=GGML_TYPE_I8,
         .offset=(uint64_t)page+264, .data=data+page+264, .size_bytes=264},
    };
    gguf_file_t gf = {.fd=fd, .mmap_addr=data, .mmap_size=3*(size_t)page,
                     .tensors=tensors, .tensor_count=3};
    myds_engram_t e = {0};
    e.table[0]=(myds_engram_table_t){.fd=fd, .offset=(uint64_t)page, .rows=1};
    e.table[1]=(myds_engram_table_t){.fd=fd, .offset=(uint64_t)page+264, .rows=1};
    tensors[0].offset=(uint64_t)page+528;
    assert(!myds_engram_unmap_tables(&gf, &e));
    assert(gf.mmap_size == 3*(size_t)page && tensors[1].data);
    tensors[0].offset=0;
    assert(myds_engram_unmap_tables(&gf, &e));
    assert(gf.mmap_size == (size_t)page && !tensors[1].data && !tensors[2].data);
    assert(data[0] == 0 && myds_engram_unmap_tables(&gf, &e));
#ifdef __APPLE__
    mach_vm_address_t region = (mach_vm_address_t)(data + page);
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object;
    kern_return_t result = mach_vm_region(mach_task_self(), &region, &region_size,
        VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &count, &object);
    assert(result == KERN_INVALID_ADDRESS ||
           (result == KERN_SUCCESS && region >= (mach_vm_address_t)(data + 3 * page)));
    if (result == KERN_SUCCESS) mach_port_deallocate(mach_task_self(), object);
#else
    unsigned char residency;
    errno=0;
    assert(mincore(data+page, (size_t)page, &residency) == -1 && errno == ENOMEM);
#endif
    assert(munmap(data, gf.mmap_size) == 0);
    close(fd); unlink(path);
    printf("Engram: reject overlapping resident weights, unmap table pages, retain main mapping passed\n");
}

int main(void) {
    test_decode_table(); test_decode_rounding_modes(); test_hash(); test_decode(); test_disk(); test_parallel_disk(); test_gate(); test_model(); test_unmap();
    puts("Engram: all tests passed");
    return 0;
}
