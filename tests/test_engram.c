#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "myds_engram.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
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
    test_hash(); test_decode(); test_disk(); test_gate(); test_model(); test_unmap();
    puts("Engram: all tests passed");
    return 0;
}
