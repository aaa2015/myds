#ifndef MYDS_GGUF_H
#define MYDS_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGUF_MAGIC 0x46554747u /* "GGUF" in little-endian */
#define GGUF_MAX_DIMS 8

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 29,
    GGML_TYPE_MXFP4   = 39,
} ggml_type;

typedef struct {
    char *key;
    gguf_type type;
    union {
        uint8_t   val_u8;
        int8_t    val_i8;
        uint16_t  val_u16;
        int16_t   val_i16;
        uint32_t  val_u32;
        int32_t   val_i32;
        float     val_f32;
        bool      val_bool;
        char     *val_str;
        uint64_t  val_u64;
        int64_t   val_i64;
        double    val_f64;
        struct {
            gguf_type elem_type;
            uint64_t  count;
            void     *data;
        } val_arr;
    } value;
} gguf_kv_t;

typedef struct {
    char        name[128];
    uint32_t    n_dims;
    uint64_t    dims[GGUF_MAX_DIMS];
    ggml_type   type;
    uint64_t    offset;
    const void *data;      /* Direct pointer into mmap file */
    size_t      size_bytes;
} gguf_tensor_t;

typedef struct {
    int            fd;
    uint8_t       *mmap_addr;
    size_t         mmap_size;
    uint32_t       version;
    uint64_t       tensor_count;
    uint64_t       metadata_kv_count;
    gguf_kv_t     *kvs;
    gguf_tensor_t *tensors;
    uint64_t       alignment;
    uint64_t       tensor_data_offset;
} gguf_file_t;

gguf_file_t *gguf_open(const char *path);
void         gguf_close(gguf_file_t *f);

const gguf_kv_t     *gguf_find_key(const gguf_file_t *f, const char *key);
const gguf_tensor_t *gguf_find_tensor(const gguf_file_t *f, const char *name);

/* Helper getters */
uint32_t     gguf_get_u32(const gguf_file_t *f, const char *key, uint32_t def);
uint64_t     gguf_get_u64(const gguf_file_t *f, const char *key, uint64_t def);
float        gguf_get_f32(const gguf_file_t *f, const char *key, float def);
const char  *gguf_get_str(const gguf_file_t *f, const char *key, const char *def);
bool         gguf_get_bool(const gguf_file_t *f, const char *key, bool def);

size_t       ggml_type_row_size(ggml_type type, size_t row_len);
const char  *ggml_type_name(ggml_type type);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_GGUF_H */
