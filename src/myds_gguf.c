#include "myds_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
} reader_t;

static inline bool read_bytes(reader_t *r, void *dst, size_t n) {
    if (r->pos + n > r->size) return false;
    memcpy(dst, r->data + r->pos, n);
    r->pos += n;
    return true;
}

static inline bool read_u32(reader_t *r, uint32_t *v) {
    return read_bytes(r, v, sizeof(*v));
}

static inline bool read_u64(reader_t *r, uint64_t *v) {
    return read_bytes(r, v, sizeof(*v));
}

static inline bool read_i32(reader_t *r, int32_t *v) {
    return read_bytes(r, v, sizeof(*v));
}

static inline bool read_f32(reader_t *r, float *v) {
    return read_bytes(r, v, sizeof(*v));
}

static char *read_string(reader_t *r) {
    uint64_t len = 0;
    if (!read_u64(r, &len)) return NULL;
    if (r->pos + len > r->size) return NULL;
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, r->data + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

static bool read_kv_value(reader_t *r, gguf_type type, gguf_kv_t *kv) {
    kv->type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:   return read_bytes(r, &kv->value.val_u8, 1);
        case GGUF_TYPE_INT8:    return read_bytes(r, &kv->value.val_i8, 1);
        case GGUF_TYPE_UINT16:  return read_bytes(r, &kv->value.val_u16, 2);
        case GGUF_TYPE_INT16:   return read_bytes(r, &kv->value.val_i16, 2);
        case GGUF_TYPE_UINT32:  return read_u32(r, &kv->value.val_u32);
        case GGUF_TYPE_INT32:   return read_i32(r, &kv->value.val_i32);
        case GGUF_TYPE_FLOAT32: return read_f32(r, &kv->value.val_f32);
        case GGUF_TYPE_BOOL:    return read_bytes(r, &kv->value.val_bool, 1);
        case GGUF_TYPE_STRING: {
            kv->value.val_str = read_string(r);
            return kv->value.val_str != NULL;
        }
        case GGUF_TYPE_UINT64:  return read_u64(r, &kv->value.val_u64);
        case GGUF_TYPE_INT64:   return read_bytes(r, &kv->value.val_i64, 8);
        case GGUF_TYPE_FLOAT64: return read_bytes(r, &kv->value.val_f64, 8);
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!read_u32(r, &elem_type) || !read_u64(r, &count)) return false;
            kv->value.val_arr.elem_type = (gguf_type)elem_type;
            kv->value.val_arr.count = count;
            if (elem_type == GGUF_TYPE_STRING) {
                char **strs = (char **)calloc(count, sizeof(char *));
                if (!strs) return false;
                for (uint64_t i = 0; i < count; i++) {
                    strs[i] = read_string(r);
                    if (!strs[i]) {
                        for (uint64_t j = 0; j < i; j++) free(strs[j]);
                        free(strs);
                        return false;
                    }
                }
                kv->value.val_arr.data = strs;
                return true;
            } else if (elem_type == GGUF_TYPE_FLOAT32) {
                float *arr = (float *)malloc(count * sizeof(float));
                if (!arr || !read_bytes(r, arr, count * sizeof(float))) {
                    free(arr);
                    return false;
                }
                kv->value.val_arr.data = arr;
                return true;
            } else if (elem_type == GGUF_TYPE_INT32) {
                int32_t *arr = (int32_t *)malloc(count * sizeof(int32_t));
                if (!arr || !read_bytes(r, arr, count * sizeof(int32_t))) {
                    free(arr);
                    return false;
                }
                kv->value.val_arr.data = arr;
                return true;
            } else {
                /* Generic array skip */
                size_t elem_sz = 1;
                switch (elem_type) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:
                    case GGUF_TYPE_BOOL: elem_sz = 1; break;
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16: elem_sz = 2; break;
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_FLOAT32: elem_sz = 4; break;
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64:
                    case GGUF_TYPE_FLOAT64: elem_sz = 8; break;
                    default: return false;
                }
                void *buf = malloc(count * elem_sz);
                if (!buf || !read_bytes(r, buf, count * elem_sz)) {
                    free(buf);
                    return false;
                }
                kv->value.val_arr.data = buf;
                return true;
            }
        }
    }
    return false;
}

size_t ggml_type_row_size(ggml_type type, size_t row_len) {
    switch (type) {
        case GGML_TYPE_F32:   return row_len * sizeof(float);
        case GGML_TYPE_F16:   return row_len * sizeof(uint16_t);
        case GGML_TYPE_BF16:  return row_len * sizeof(uint16_t);
        case GGML_TYPE_Q8_0:  return (row_len / 32) * (sizeof(uint16_t) + 32);
        case GGML_TYPE_Q4_0:  return (row_len / 32) * (sizeof(uint16_t) + 16);
        case GGML_TYPE_Q4_K:  return (row_len / 256) * 144;
        case GGML_TYPE_Q5_K:  return (row_len / 256) * 176;
        case GGML_TYPE_Q6_K:  return (row_len / 256) * 210;
        case GGML_TYPE_Q2_K:  return (row_len / 256) * 84;
        case GGML_TYPE_IQ2_XXS: return (row_len / 256) * 66;
        case GGML_TYPE_MXFP4: return (row_len / 32) * 17;
        default:              return row_len * 2; /* fallback */
    }
}

const char *ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:   return "F32";
        case GGML_TYPE_F16:   return "F16";
        case GGML_TYPE_BF16:  return "BF16";
        case GGML_TYPE_Q8_0:  return "Q8_0";
        case GGML_TYPE_Q4_0:  return "Q4_0";
        case GGML_TYPE_Q4_K:  return "Q4_K";
        case GGML_TYPE_Q5_K:  return "Q5_K";
        case GGML_TYPE_Q6_K:  return "Q6_K";
        case GGML_TYPE_Q2_K:  return "Q2_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_MXFP4: return "MXFP4";
        default:              return "UNKNOWN";
    }
}

gguf_file_t *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("gguf_open: open");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }

    size_t size = (size_t)st.st_size;
    size_t page_sz = (size_t)getpagesize();
    size_t mmap_size = ((size + page_sz - 1) / page_sz) * page_sz;
    uint8_t *addr = (uint8_t *)mmap(NULL, mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        perror("gguf_open: mmap");
        close(fd);
        return NULL;
    }

    reader_t r = { .data = addr, .size = size, .pos = 0 };
    uint32_t magic = 0;
    if (!read_u32(&r, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf_open: invalid magic (expected 0x%08x, got 0x%08x)\n", GGUF_MAGIC, magic);
        munmap(addr, size);
        close(fd);
        return NULL;
    }

    uint32_t version = 0;
    if (!read_u32(&r, &version) || (version != 2 && version != 3)) {
        fprintf(stderr, "gguf_open: unsupported GGUF version %u\n", version);
        munmap(addr, size);
        close(fd);
        return NULL;
    }

    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_u64(&r, &tensor_count) || !read_u64(&r, &kv_count)) {
        munmap(addr, size);
        close(fd);
        return NULL;
    }

    gguf_file_t *gf = (gguf_file_t *)calloc(1, sizeof(gguf_file_t));
    gf->fd = fd;
    gf->mmap_addr = addr;
    gf->mmap_size = mmap_size;
    gf->version = version;
    gf->tensor_count = tensor_count;
    gf->metadata_kv_count = kv_count;
    gf->alignment = 32; /* default */
    gf->kvs = (gguf_kv_t *)calloc(kv_count, sizeof(gguf_kv_t));

    for (uint64_t i = 0; i < kv_count; i++) {
        gf->kvs[i].key = read_string(&r);
        uint32_t type = 0;
        if (!read_u32(&r, &type) || !read_kv_value(&r, (gguf_type)type, &gf->kvs[i])) {
            fprintf(stderr, "gguf_open: failed reading metadata kv %llu\n", (unsigned long long)i);
            gguf_close(gf);
            return NULL;
        }
        if (strcmp(gf->kvs[i].key, "general.alignment") == 0) {
            gf->alignment = gf->kvs[i].value.val_u32 ? gf->kvs[i].value.val_u32 : gf->kvs[i].value.val_u64;
        }
    }

    gf->tensors = (gguf_tensor_t *)calloc(tensor_count, sizeof(gguf_tensor_t));
    for (uint64_t i = 0; i < tensor_count; i++) {
        char *name = read_string(&r);
        if (!name) {
            gguf_close(gf);
            return NULL;
        }
        strncpy(gf->tensors[i].name, name, sizeof(gf->tensors[i].name) - 1);
        free(name);

        uint32_t n_dims = 0;
        if (!read_u32(&r, &n_dims) || n_dims > GGUF_MAX_DIMS) {
            gguf_close(gf);
            return NULL;
        }
        gf->tensors[i].n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; d++) {
            if (!read_u64(&r, &gf->tensors[i].dims[d])) {
                gguf_close(gf);
                return NULL;
            }
        }

        uint32_t type = 0;
        uint64_t offset = 0;
        if (!read_u32(&r, &type) || !read_u64(&r, &offset)) {
            gguf_close(gf);
            return NULL;
        }
        gf->tensors[i].type = (ggml_type)type;
        gf->tensors[i].offset = offset;
    }

    /* Calculate data alignment */
    size_t align = gf->alignment;
    gf->tensor_data_offset = (r.pos + align - 1) / align * align;

    for (uint64_t i = 0; i < tensor_count; i++) {
        size_t abs_offset = gf->tensor_data_offset + gf->tensors[i].offset;
        gf->tensors[i].data = (const void *)(gf->mmap_addr + abs_offset);

        /* Calculate tensor size */
        size_t n_elements = 1;
        for (uint32_t d = 0; d < gf->tensors[i].n_dims; d++) {
            n_elements *= gf->tensors[i].dims[d];
        }
        gf->tensors[i].size_bytes = ggml_type_row_size(gf->tensors[i].type, n_elements);
    }

    return gf;
}

void gguf_close(gguf_file_t *f) {
    if (!f) return;
    if (f->kvs) {
        for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
            free(f->kvs[i].key);
            if (f->kvs[i].type == GGUF_TYPE_STRING) {
                free(f->kvs[i].value.val_str);
            } else if (f->kvs[i].type == GGUF_TYPE_ARRAY) {
                if (f->kvs[i].value.val_arr.elem_type == GGUF_TYPE_STRING) {
                    char **strs = (char **)f->kvs[i].value.val_arr.data;
                    if (strs) {
                        for (uint64_t j = 0; j < f->kvs[i].value.val_arr.count; j++) {
                            free(strs[j]);
                        }
                        free(strs);
                    }
                } else {
                    free(f->kvs[i].value.val_arr.data);
                }
            }
        }
        free(f->kvs);
    }
    if (f->tensors) free(f->tensors);
    if (f->mmap_addr) munmap(f->mmap_addr, f->mmap_size);
    if (f->fd >= 0) close(f->fd);
    free(f);
}

const gguf_kv_t *gguf_find_key(const gguf_file_t *f, const char *key) {
    if (!f || !key) return NULL;
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
        if (strcmp(f->kvs[i].key, key) == 0) return &f->kvs[i];
    }
    return NULL;
}

const gguf_tensor_t *gguf_find_tensor(const gguf_file_t *f, const char *name) {
    if (!f || !name) return NULL;
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    return NULL;
}

uint32_t gguf_get_u32(const gguf_file_t *f, const char *key, uint32_t def) {
    const gguf_kv_t *kv = gguf_find_key(f, key);
    if (!kv) return def;
    if (kv->type == GGUF_TYPE_UINT32) return kv->value.val_u32;
    if (kv->type == GGUF_TYPE_INT32)  return (uint32_t)kv->value.val_i32;
    if (kv->type == GGUF_TYPE_UINT64) return (uint32_t)kv->value.val_u64;
    return def;
}

uint64_t gguf_get_u64(const gguf_file_t *f, const char *key, uint64_t def) {
    const gguf_kv_t *kv = gguf_find_key(f, key);
    if (!kv) return def;
    if (kv->type == GGUF_TYPE_UINT64) return kv->value.val_u64;
    if (kv->type == GGUF_TYPE_UINT32) return kv->value.val_u32;
    return def;
}

float gguf_get_f32(const gguf_file_t *f, const char *key, float def) {
    const gguf_kv_t *kv = gguf_find_key(f, key);
    if (!kv) return def;
    if (kv->type == GGUF_TYPE_FLOAT32) return kv->value.val_f32;
    if (kv->type == GGUF_TYPE_FLOAT64) return (float)kv->value.val_f64;
    return def;
}

const char *gguf_get_str(const gguf_file_t *f, const char *key, const char *def) {
    const gguf_kv_t *kv = gguf_find_key(f, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return def;
    return kv->value.val_str;
}

bool gguf_get_bool(const gguf_file_t *f, const char *key, bool def) {
    const gguf_kv_t *kv = gguf_find_key(f, key);
    if (!kv || kv->type != GGUF_TYPE_BOOL) return def;
    return kv->value.val_bool;
}
