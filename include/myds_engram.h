#ifndef MYDS_ENGRAM_H
#define MYDS_ENGRAM_H

#include "myds_gguf.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    MYDS_ENGRAM_LAYERS = 2,
    MYDS_ENGRAM_NGRAM = 4,
    MYDS_ENGRAM_HEADS = 8,
    MYDS_ENGRAM_COLS = 24,
    MYDS_ENGRAM_DIM = 256,
    MYDS_ENGRAM_ROW_BYTES = 264,
    MYDS_ENGRAM_INPUT_DIM = MYDS_ENGRAM_COLS * MYDS_ENGRAM_DIM,
    MYDS_ENGRAM_DEAD = -1
};

typedef struct {
    const uint32_t *token_map;
    uint32_t vocab_size, compressed_vocab_size, pad_id;
    uint32_t rows[MYDS_ENGRAM_LAYERS];
    uint64_t multipliers[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_NGRAM];
    uint32_t primes[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];
} myds_engram_layout_t;

/* The three newest compressed IDs, newest first. A masked token breaks every
 * n-gram crossing it. A copied history is a complete transaction snapshot. */
typedef struct { int32_t tail[MYDS_ENGRAM_NGRAM - 1]; } myds_engram_history_t;

bool myds_engram_layout_valid(const myds_engram_layout_t *layout);
void myds_engram_history_reset(myds_engram_history_t *history);
/* Validate layout once before calling. Invalid tokens leave history unchanged.
 * Output is [token][layer][column]. Mask zero suppresses the additive graph at
 * that position; callers must not apply Engram for a masked token. */
bool myds_engram_hash(const myds_engram_layout_t *layout,
                     myds_engram_history_t *history,
                     const int32_t *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows);

typedef struct { int fd; uint64_t offset; uint32_t rows; } myds_engram_table_t;
/* Separate pread descriptor; no full-table read or GPU map. On macOS,
 * MYDS_ENGRAM_NOCACHE requests F_NOCACHE/no-readahead for diagnostics. That
 * switch alone does not prove a cache miss or physical SSD transfer. */
bool myds_engram_table_open(myds_engram_table_t *table, const char *path,
                           uint64_t offset, uint32_t rows);
void myds_engram_table_close(myds_engram_table_t *table);
/* Each row is 256 E4M3FN bytes followed by eight E8M0 scale bytes. Invalid
 * encodings, nonfinite results and short reads fail; output is then invalid.
 * Scratch storage is one 264-byte row, irrespective of table/count size. */
bool myds_engram_decode_row(const uint8_t row[MYDS_ENGRAM_ROW_BYTES],
                           float out[MYDS_ENGRAM_DIM]);
bool myds_engram_read(const myds_engram_table_t *table, const uint32_t *rows,
                     size_t count, float *out);

/* Persistent raw-I/O workers. Only pread is parallel; decoding runs on the
 * calling thread in table/column order, preserving rounding and error order.
 * Workers are joined before free returns. Do not free during a read. A NULL
 * reader, or a reader created with one worker, uses the serial oracle above. */
typedef struct myds_engram_reader myds_engram_reader_t;
myds_engram_reader_t *myds_engram_reader_create(unsigned workers);
void myds_engram_reader_free(myds_engram_reader_t *reader);
bool myds_engram_reader_read_token(myds_engram_reader_t *reader,
    const myds_engram_table_t tables[MYDS_ENGRAM_LAYERS],
    const uint32_t rows[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS],
    float out[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM]);

typedef struct {
    myds_engram_layout_t layout;
    myds_engram_history_t history;
    myds_engram_table_t table[MYDS_ENGRAM_LAYERS];
    const gguf_tensor_t *projection[MYDS_ENGRAM_LAYERS];
    const gguf_tensor_t *q_norm[MYDS_ENGRAM_LAYERS];
    const gguf_tensor_t *k_norm[MYDS_ENGRAM_LAYERS];
    uint32_t layer_ids[MYDS_ENGRAM_LAYERS], width;
    uint64_t position;
    bool prepared;
    unsigned io_threads;
    myds_engram_reader_t *reader;
    uint32_t row_ids[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_COLS];
    float row_values[MYDS_ENGRAM_LAYERS][MYDS_ENGRAM_INPUT_DIM];
} myds_engram_t;

/* GGUF descriptors/metadata are borrowed; gf must outlive this object. Loader
 * validates exact tensor dimensions, metadata and descriptor/file identity. */
myds_engram_t *myds_engram_create(const gguf_file_t *gf, const char *path,
                                 uint32_t vocab_size, uint32_t width);
/* Clone model's immutable template into a context. Each clone owns duplicate
 * pread descriptors and fresh history, so contexts cannot share hash state. */
myds_engram_t *myds_engram_clone(const myds_engram_t *engram);
/* 1 selects serial I/O; 2..16 create a private persistent pool. Failure leaves
 * the previous configuration intact. Like prepare/reset/free, this requires
 * exclusive access to the context. Model load reads MYDS_ENGRAM_IO_THREADS. */
bool myds_engram_set_io_threads(myds_engram_t *engram, unsigned workers);
/* Reject overlapping/non-tail layouts; unmap only page-aligned disk-only tail.
 * Call before publishing model pointers/GPU views. Sets table data pointers to
 * NULL and shrinks mmap_size; projection/norm/metadata remain valid. */
bool myds_engram_unmap_tables(gguf_file_t *gf, const myds_engram_t *engram);
void myds_engram_free(myds_engram_t *engram);
void myds_engram_reset(myds_engram_t *engram);
/* Call once per text token before any layer, with consecutive positions.
 * Both tables' rows are fetched before history advances; failure cannot commit
 * partial hash state. Projection is native Metal F16 GEMV 6144 -> 5*width,
 * BF16 rounded, followed by the native Metal Engram gate/add at layers 1/14. */
bool myds_engram_prepare_token(myds_engram_t *engram, int32_t token,
                              uint64_t position);

/* Small CPU mathematical oracle for the native GPU gate/add. It is not a
 * production projection path. residual=[4,width], kv=[5,width], q/k=[4,width]. */
bool myds_engram_add_reference(float *residual, const float *kv,
                               const float *q_weight, const float *k_weight,
                               uint32_t width, float eps);

#endif
