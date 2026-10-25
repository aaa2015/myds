#include "myds_gguf.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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
    /* 2026-10-25: 溢出安全。原写法 `r->pos + n > r->size` 在 n 接近 SIZE_MAX 时
     * 会回绕成小数从而通过检查，随后 memcpy 一次巨量拷贝。公开仓库会迎来 fuzz 输入，
     * 所以改成减法形式：只有 n <= size-pos 才成立，不产生回绕。 */
    if (n > r->size || r->pos > r->size - n) return false;
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
    /* 2026-10-25: 三处都要查，缺一不可。
     *   ① 剩余空间：用减法形式，`r->pos + len` 会回绕。
     *      len=UINT64_MAX 时 pos+len 回绕成 31，旧写法通过 ⇒ 下一步崩。
     *   ② len+1 回绕：len=UINT64_MAX 时 malloc(len+1) 等价于 malloc(0),
     *      返回一个小块但有效，随后 memcpy 巨量 ⇒ ASan negative-size-param。
     *   ③ malloc 上限：len 合法但巨大时不要让 malloc 直接失败。
     * ★ 这个函数不经过 read_bytes()，所以那边修了这里依然会崩——
     *   上一版就是漏了这一点。 */
    if (len >= r->size || len > r->size - r->pos) return NULL;
    if (len > (uint64_t)SIZE_MAX - 1) return NULL;
    char *s = (char *)malloc((size_t)len + 1);
    if (!s) return NULL;
    memcpy(s, r->data + r->pos, (size_t)len);
    s[len] = '\0';
    r->pos += (size_t)len;
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
            /* 2026-10-25: count 必须能放进 size_t，且与元素尺寸相乘不得回绕。
             * 否则 `calloc(count, sizeof(char*))` 会得到一个小块，
             * 而循环里的 count 次写入直接跑出块外。 */
            if (count > (uint64_t)SIZE_MAX) return false;
            if (elem_type == GGUF_TYPE_STRING) {
                if (count > (uint64_t)SIZE_MAX / sizeof(char *)) return false;
                char **strs = (char **)calloc((size_t)count, sizeof(char *));
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
                if (count > (uint64_t)SIZE_MAX / sizeof(float)) return false;
                float *arr = (float *)malloc((size_t)count * sizeof(float));
                if (!arr || !read_bytes(r, arr, (size_t)count * sizeof(float))) {
                    free(arr);
                    return false;
                }
                kv->value.val_arr.data = arr;
                return true;
            } else if (elem_type == GGUF_TYPE_INT32) {
                if (count > (uint64_t)SIZE_MAX / sizeof(int32_t)) return false;
                int32_t *arr = (int32_t *)malloc((size_t)count * sizeof(int32_t));
                if (!arr || !read_bytes(r, arr, (size_t)count * sizeof(int32_t))) {
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
                void *buf = malloc((size_t)count * elem_sz);
                if (!buf || !read_bytes(r, buf, (size_t)count * elem_sz)) {
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
        case GGML_TYPE_I8:    return row_len;
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

bool ggml_type_row_size_checked(ggml_type type, size_t row_len, size_t *out) {
    if (!out) return false;
    /* 2026-10-25: 非分块类型是直接乘法，元素数极大时会回绕。
     * 例：F32 且 row_len = 2^62 ⇒ 2^62*4 = 0 ⇒ size_bytes=0，
     * 边界检查全部通过，但其实指向一个巨大张量。
     * 分块类型（Q*_K / Q8_0 等）先除以块长，不会回绕，直接放行。 */
    switch (type) {
        case GGML_TYPE_F32:
            if (row_len > SIZE_MAX / sizeof(float)) return false;
            *out = row_len * sizeof(float);
            return true;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            if (row_len > SIZE_MAX / sizeof(uint16_t)) return false;
            *out = row_len * sizeof(uint16_t);
            return true;
        case GGML_TYPE_I8:
            *out = row_len;
            return true;
        case GGML_TYPE_Q8_0:
            if (row_len / 32u > SIZE_MAX / (sizeof(uint16_t) + 32u)) return false;
            *out = (row_len / 32u) * (sizeof(uint16_t) + 32u);
            return true;
        case GGML_TYPE_Q4_0:
            if (row_len / 32u > SIZE_MAX / (sizeof(uint16_t) + 16u)) return false;
            *out = (row_len / 32u) * (sizeof(uint16_t) + 16u);
            return true;
        default:
            /* 其余类型都是块量化，块长 ≥ 32，不会回绕 */
            *out = ggml_type_row_size(type, row_len);
            return true;
    }
}

const char *ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:   return "F32";
        case GGML_TYPE_F16:   return "F16";
        case GGML_TYPE_BF16:  return "BF16";
        case GGML_TYPE_I8:    return "I8";
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


/* ── split GGUF 支持（2026-10-25）──────────────────────────────────
 *
 * llama.cpp 的 `gguf-split` 产出 `<prefix>-00001-of-00005.gguf` 形式的多个文件：
 * 每个都是【完整合法的 GGUF】，各带自己的 header / KV / 张量子集。
 * HF 上超过单文件上限的权重都用这种形式发布，所以 myds 必须能直接吃第一个分片。
 *
 * 设计取舍：
 *   · metadata 只取 part 0 的 —— llama.cpp 在每个分片都写了同一份 KV，
 *     但只有 part 0 是权威；多余的直接释放。
 *   · 一致性校验只做「版本 / alignment / 张量总数 == 各分片之和 / 名字不重复」。
 *     逐字段比对所有分片的 KV 代价太大且没有实际收益。
 *   · 每个张量记下所属分片与文件内绝对偏移（gguf_tensor_t.part / file_offset）。
 *     engram 的 pread 路径必须用 gguf_tensor_source()，否则会从 part 0 读错位置。
 *   · n_parts == 1 时所有路径与改动前完全一致（单文件仍是单次 mmap）。
 */

static void free_kvs(gguf_kv_t *kvs, uint64_t n) {
    if (!kvs) return;
    for (uint64_t i = 0; i < n; i++) {
        free(kvs[i].key);
        if (kvs[i].type == GGUF_TYPE_STRING) {
            free(kvs[i].value.val_str);
        } else if (kvs[i].type == GGUF_TYPE_ARRAY) {
            if (kvs[i].value.val_arr.elem_type == GGUF_TYPE_STRING) {
                char **strs = (char **)kvs[i].value.val_arr.data;
                if (strs) {
                    for (uint64_t j = 0; j < kvs[i].value.val_arr.count; j++) free(strs[j]);
                    free(strs);
                }
            } else {
                free(kvs[i].value.val_arr.data);
            }
        }
    }
    free(kvs);
}

/* 解析 split 文件名 <dir><prefix>-<idx>-of-<total>.gguf。
 * 成功时把「目录 + prefix + '-'」写进 prefix_out，方便直接拼其它分片路径。 */
static bool split_parse_name(const char *path, char *prefix_out, size_t prefix_cap,
                            uint32_t *idx, uint32_t *total, int *idx_width) {
    if (!path || !prefix_out || prefix_cap == 0) return false;
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    const size_t blen = strlen(base);
    if (blen < 6) return false;
    if (strcasecmp(base + blen - 5, ".gguf") != 0) return false;
    const char *end = base + blen - 5;         /* 指向 ".gguf" 前，即 total 数字之后 */

    /* total：结尾的连续数字 */
    const char *tot_end = end;
    const char *tot_start = tot_end;
    while (tot_start > base && isdigit((unsigned char)tot_start[-1])) tot_start--;
    if (tot_start == tot_end) return false;
    if (tot_end - tot_start > 6) return false;
    if (tot_start - base < 4 || strncmp(tot_start - 4, "-of-", 4) != 0) return false;

    /* idx：位于 "-of-" 之前 */
    const char *ix_end = tot_start - 4;
    const char *ix_start = ix_end;
    while (ix_start > base && isdigit((unsigned char)ix_start[-1])) ix_start--;
    if (ix_start == ix_end) return false;
    if (ix_end - ix_start > 6) return false;
    if (ix_start == base || ix_start[-1] != '-') return false;

    char tmp[32];
    const size_t n = (size_t)(tot_end - tot_start);
    if (n >= sizeof(tmp)) return false;
    memcpy(tmp, tot_start, n);
    tmp[n] = '\0';
    const long t = strtol(tmp, NULL, 10);
    if (t < 2 || t > 99999) return false;

    const size_t m = (size_t)(ix_end - ix_start);
    memcpy(tmp, ix_start, m);
    tmp[m] = '\0';
    const long i = strtol(tmp, NULL, 10);
    if (i < 1 || i > t) return false;

    /* ix_start 指向 idx 的首位数字；它前面那个字符是分隔符 '-'，
     * 拼接时要把它排掉，否则会得到 "prefix--00001-of-..."。 */
    const size_t pre_len = (size_t)(ix_start - path) - 1u;
    if (pre_len + 1 >= prefix_cap) return false;
    memcpy(prefix_out, path, pre_len);
    prefix_out[pre_len] = '\0';

    if (idx) *idx = (uint32_t)i;
    if (total) *total = (uint32_t)t;
    if (idx_width) *idx_width = (int)m;
    return true;
}

typedef struct {
    gguf_kv_t     *kvs;
    uint64_t       kv_count;
    gguf_tensor_t *tensors;
    uint64_t       tensor_count;
    uint32_t       version;
    uint64_t       alignment;
} part_parse_t;

/* mmap 一个分片并解析它的 header / KV / 张量，同时做边界校验。
 * 返回的张量 offset 仍是文件内相对值，data 指向该分片自己的映射。 */
static bool parse_mapped(uint8_t *addr, size_t size, const char *path,
                         gguf_part_t *pt, part_parse_t *out);

/* 只做映射，解析交给 parse_mapped。拆开是为了让字节分段（多个文件拼成
 * 一个连续 VA）能复用同一套解析与边界校验。 */
static bool parse_part(const char *path, gguf_part_t *pt, part_parse_t *out) {
    memset(pt, 0, sizeof(*pt));
    pt->fd = -1;
    memset(out, 0, sizeof(*out));

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("gguf_open: open");
        return false;
    }
#if defined(__APPLE__) && defined(F_NOCACHE)
    (void)fcntl(fd, F_NOCACHE, 1);
#endif
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    const size_t size = (size_t)st.st_size;
    const size_t page_sz = (size_t)getpagesize();
    const size_t mmap_size = ((size + page_sz - 1) / page_sz) * page_sz;
    uint8_t *addr = (uint8_t *)mmap(NULL, mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { perror("gguf_open: mmap"); close(fd); return false; }

    pt->fd = fd;
    pt->mmap_addr = addr;
    pt->mmap_size = mmap_size;
    pt->file_size = size;

    if (!parse_mapped(addr, size, path, pt, out)) {
        munmap(addr, mmap_size);
        close(fd);
        memset(pt, 0, sizeof(*pt));
        pt->fd = -1;
        return false;
    }
    return true;
}

/* 从已映射的缓冲区解析 header/KV/张量，并做全部边界校验。
 * addr 可以来自单文件、某个分片，或字节分段拼出的连续映射。 */
static bool parse_mapped(uint8_t *addr, size_t size, const char *path,
                         gguf_part_t *pt, part_parse_t *out) {
    memset(out, 0, sizeof(*out));
    pt->mmap_addr = addr;
    pt->file_size = size;
    reader_t r = { .data = addr, .size = size, .pos = 0 };
    uint32_t magic = 0, version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_u32(&r, &magic) || magic != GGUF_MAGIC ||
        !read_u32(&r, &version) || (version != 2 && version != 3) ||
        !read_u64(&r, &tensor_count) || !read_u64(&r, &kv_count)) {
        fprintf(stderr, "gguf_open: %s: invalid header\n", path);
        goto fail;
    }

    out->version = version;
    out->kv_count = kv_count;
    out->tensor_count = tensor_count;
    out->alignment = 32;
    out->kvs = (gguf_kv_t *)calloc(kv_count ? kv_count : 1, sizeof(gguf_kv_t));
    if (!out->kvs) goto fail;

    for (uint64_t i = 0; i < kv_count; i++) {
        out->kvs[i].key = read_string(&r);
        if (!out->kvs[i].key) {
            fprintf(stderr, "gguf_open: %s: failed reading metadata key %llu\n",
                    path, (unsigned long long)i);
            goto fail;
        }
        uint32_t type = 0;
        if (!read_u32(&r, &type) || !read_kv_value(&r, (gguf_type)type, &out->kvs[i])) {
            fprintf(stderr, "gguf_open: %s: failed reading metadata kv %llu\n",
                    path, (unsigned long long)i);
            goto fail;
        }
        if (strcmp(out->kvs[i].key, "general.alignment") == 0) {
            const uint32_t a = out->kvs[i].value.val_u32;
            if (a != 0) {
                if (a > (1u << 20)) {
                    fprintf(stderr, "gguf_open: %s: general.alignment=%u too large\n", path, a);
                    goto fail;
                }
                out->alignment = a;
            } else {
                fprintf(stderr, "gguf_open: %s: general.alignment=0 ignored, keeping %llu\n",
                        path, (unsigned long long)out->alignment);
            }
        }
    }

    out->tensors = (gguf_tensor_t *)calloc(tensor_count ? tensor_count : 1,
                                          sizeof(gguf_tensor_t));
    if (!out->tensors) goto fail;

    for (uint64_t i = 0; i < tensor_count; i++) {
        char *name = read_string(&r);
        if (!name) goto fail;
        strncpy(out->tensors[i].name, name, sizeof(out->tensors[i].name) - 1);
        free(name);

        uint32_t n_dims = 0;
        if (!read_u32(&r, &n_dims) || n_dims > GGUF_MAX_DIMS) goto fail;
        out->tensors[i].n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; d++) {
            if (!read_u64(&r, &out->tensors[i].dims[d])) goto fail;
        }
        uint32_t type = 0;
        uint64_t offset = 0;
        if (!read_u32(&r, &type) || !read_u64(&r, &offset)) goto fail;
        out->tensors[i].type = (ggml_type)type;
        out->tensors[i].offset = offset;
    }

    {
        size_t align = (size_t)out->alignment;
        if (align == 0) align = 1;
        pt->tensor_data_offset = ((uint64_t)r.pos + align - 1) / align * align;
    }

    if (pt->tensor_data_offset > pt->file_size && tensor_count > 0) {
        fprintf(stderr, "gguf_open: %s: tensor data offset %llu beyond file size %zu\n",
                path, (unsigned long long)pt->tensor_data_offset, pt->file_size);
        goto fail;
    }
    const size_t data_avail = (pt->tensor_data_offset <= pt->file_size)
                              ? (size_t)(pt->file_size - (size_t)pt->tensor_data_offset) : 0u;

    for (uint64_t i = 0; i < tensor_count; i++) {
        if (out->tensors[i].offset > (uint64_t)data_avail) {
            fprintf(stderr, "gguf_open: %s: tensor %s offset %llu beyond data region (%zu)\n",
                    path, out->tensors[i].name,
                    (unsigned long long)out->tensors[i].offset, data_avail);
            goto fail;
        }
        const size_t rel = (size_t)out->tensors[i].offset;

        size_t n_elements = 1;
        for (uint32_t d = 0; d < out->tensors[i].n_dims; d++) {
            const uint64_t dim = out->tensors[i].dims[d];
            if (dim != 0 && n_elements > SIZE_MAX / (size_t)dim) {
                fprintf(stderr, "gguf_open: %s: tensor %s element count overflows\n",
                        path, out->tensors[i].name);
                goto fail;
            }
            n_elements *= (size_t)dim;
        }
        size_t row = 0;
        if (!ggml_type_row_size_checked(out->tensors[i].type, n_elements, &row)) {
            fprintf(stderr, "gguf_open: %s: tensor %s byte size overflows\n",
                    path, out->tensors[i].name);
            goto fail;
        }
        if (row > (size_t)data_avail - rel) {
            fprintf(stderr, "gguf_open: %s: tensor %s out of bounds (%zu > %zu)\n",
                    path, out->tensors[i].name, row, (size_t)data_avail - rel);
            goto fail;
        }
        out->tensors[i].size_bytes = row;
        out->tensors[i].file_offset = pt->tensor_data_offset + (uint64_t)rel;
        out->tensors[i].data = (const void *)(pt->mmap_addr + out->tensors[i].file_offset);
    }
    return true;

fail:
    /* 只释放解析出来的数据结构；映射的所有权属于调用方。 */
    free_kvs(out->kvs, out->kv_count);
    free(out->tensors);
    memset(out, 0, sizeof(*out));
    pt->mmap_addr = NULL;
    pt->mmap_size = 0;
    pt->tensor_data_offset = 0;
    return false;
}

/* ── 字节分段（contiguous byte-range parts）─────────────────────────────
 *
 * 为什么需要它：llama.cpp 的 split 按【张量边界】切，每个分片都是合法 GGUF。
 * 但我们的权重里有两个 94.42 GiB 的 engram 张量，单个就超过托管方的单文件
 * 上限 ⇒ 按张量切也传不上去。这种情况下只能按【固定字节长度】切，每片不是
 * 合法 GGUF，合起来才是。
 *
 * 设计：把各分片映射进一段【连续的虚拟地址】，对上层完全等价于一个文件。
 *   · 分片用 MAP_FIXED 逐个盖到预留区间上，偏移 = (i-1)*chunk。
 *   · chunk 必须是页对齐的（1 MiB 对齐就满足），否则相邻分片的页会重叠。
 *   · 因此 parsemapped 拿到的就是一个 size = total 的连续缓冲区，
 *     所有既有边界校验与张量指针逻辑一行都不用改。
 *
 * 校验（缺片绝不能静默通过 —— 少了中间一片会让后半段整体错位）：
 *   · 清单文件存在且字段自洽
 *   · 除最后一片外，每片大小都等于清单里的 chunk
 *   · 各片大小之和等于清单里的 total
 *   · chunk 页对齐
 *
 * 清单格式（`<base>.parts`，纯文本，几十字节）：
 *     myds-split 1
 *     parts 8
 *     chunk 45268074496
 *     total 362142629888
 */
typedef struct {
    uint32_t parts;
    uint64_t chunk;
    uint64_t total;
} byte_split_t;

/* 从 "<...>.partNN" 推出 base；失败返回 false（说明不是这种命名）。 */
static bool byte_split_base(const char *path, char *base, size_t cap) {
    if (!path || !base || cap == 0) return false;
    const size_t n = strlen(path);
    const char *dot = NULL;
    for (size_t i = n; i > 0; i--) {
        if (path[i - 1] == '.') { dot = path + i - 1; break; }
    }
    if (!dot) return false;
    /* dot 指向 ".part01" 的那个点，所以从 dot 开始比。 */
    if (strncmp(dot, ".part", 5) != 0) return false;
    /* digit 从 ".part" 之后开始（dot+5），不是 dot+1 —— 否则会把 "part01"
     * 当成数字串，isdigit('p') 为假，整个识别就永远失败。 */
    const char *digits = dot + 5;
    if (!*digits) return false;
    for (const char *q = digits; *q; q++) if (!isdigit((unsigned char)*q)) return false;
    if (strlen(digits) < 2) return false;
    const size_t pre = (size_t)(dot - path);
    if (pre + 1 >= cap) return false;
    memcpy(base, path, pre);
    base[pre] = '\0';
    return pre > 0;
}

static bool byte_split_manifest(const char *base, byte_split_t *m) {
    char path[4096];
    const int n = snprintf(path, sizeof(path), "%s.parts", base);
    if (n <= 0 || (size_t)n >= sizeof(path)) return false;
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char line[256];
    uint32_t ver = 0, parts = 0;
    uint64_t chunk = 0, total = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long a = 0, b = 0;
        if (sscanf(line, "myds-split %u", &ver) == 1) continue;
        if (sscanf(line, "parts %llu", &a) == 1) { parts = (uint32_t)a; continue; }
        if (sscanf(line, "chunk %llu", &a) == 1) { chunk = a; continue; }
        if (sscanf(line, "total %llu", &a) == 1) { total = a; continue; }
        (void)b;
    }
    fclose(f);
    if (ver != 1 || parts < 2 || !chunk || !total) return false;
    if (chunk % (uint64_t)getpagesize() != 0) {
        fprintf(stderr, "gguf_open: %s.parts: chunk %llu not page aligned\n", base,
                (unsigned long long)chunk);
        return false;
    }
    if ((uint64_t)(parts - 1) * chunk >= total + chunk) return false;
    m->parts = parts;
    m->chunk = chunk;
    m->total = total;
    return true;
}

static gguf_file_t *gguf_open_byte_split(const char *base, const byte_split_t *m) {
    if (m->parts > 65535u) return NULL;
    const size_t page = (size_t)getpagesize();
    const size_t total_size = (size_t)m->total;
    if (total_size != m->total) return NULL;   /* 32 位下放不下 */
    const size_t reserve = (total_size + page - 1) / page * page;

    /* 预留整段地址空间，再逐片 MAP_FIXED 覆盖。预留本身不占物理内存。 */
    uint8_t *root = (uint8_t *)mmap(NULL, reserve, PROT_NONE,
                                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (root == MAP_FAILED) {
        fprintf(stderr, "gguf_open: cannot reserve %zu bytes of address space\n", reserve);
        return NULL;
    }

    gguf_file_t *gf = (gguf_file_t *)calloc(1, sizeof(gguf_file_t));
    if (!gf) goto fail_root;
    gf->fd = -1;
    gf->n_parts = 1;
    gf->parts = (gguf_part_t *)calloc(1, sizeof(gguf_part_t));
    if (!gf->parts) { free(gf); goto fail_root; }
    gf->parts[0].fd = -1;

    struct { int fd; uint8_t *addr; size_t len; } segs[65536];
    uint32_t nseg = 0;
    uint64_t covered = 0;
    for (uint32_t i = 1; i <= m->parts; i++) {
        char path[4096];
        const int n = snprintf(path, sizeof(path), "%s.part%02u", base, i);
        if (n <= 0 || (size_t)n >= sizeof(path)) goto fail_segs;
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "gguf_open: byte-split part %u/%u missing: %s\n",
                    i, m->parts, path);
            goto fail_segs;
        }
#if defined(__APPLE__) && defined(F_NOCACHE)
        (void)fcntl(fd, F_NOCACHE, 1);
#endif
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); goto fail_segs; }
        const size_t sz = (size_t)st.st_size;
        const size_t want = (i < m->parts) ? (size_t)m->chunk : (size_t)(m->total - covered);
        if (sz != want) {
            fprintf(stderr, "gguf_open: byte-split part %u/%u size %zu, expected %zu\n",
                    i, m->parts, sz, want);
            close(fd);
            goto fail_segs;
        }
        const size_t maplen = (sz + page - 1) / page * page;
        uint8_t *dst = root + (size_t)covered;
        void *got = mmap(dst, maplen, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0);
        if (got == MAP_FAILED) {
            fprintf(stderr, "gguf_open: byte-split part %u/%u cannot map at %p\n",
                    i, m->parts, (void *)dst);
            close(fd);
            goto fail_segs;
        }
        segs[nseg].fd = fd;
        segs[nseg].addr = dst;
        segs[nseg].len = maplen;
        nseg++;
        covered += sz;
    }
    if (covered != m->total) {
        fprintf(stderr, "gguf_open: byte-split covered %llu of %llu bytes\n",
                (unsigned long long)covered, (unsigned long long)m->total);
        goto fail_segs;
    }

    part_parse_t pp;
    if (!parse_mapped(root, total_size, base, &gf->parts[0], &pp)) goto fail_segs;
    gf->parts[0].fd = segs[0].fd;
    gf->parts[0].mmap_addr = root;
    gf->parts[0].mmap_size = reserve;

    gf->fd = segs[0].fd;
    gf->mmap_addr = root;
    gf->mmap_size = reserve;
    gf->file_size = total_size;
    gf->tensor_data_offset = gf->parts[0].tensor_data_offset;
    gf->version = pp.version;
    gf->alignment = pp.alignment;
    gf->tensor_count = pp.tensor_count;
    gf->metadata_kv_count = pp.kv_count;
    gf->kvs = pp.kvs;
    gf->tensors = pp.tensors;
    /* 标记：字节分段模式下 gguf_tensor_source 不可用（engram 必须走映射）。 */
    gf->byte_split = 1;

    /* 其余分片的 fd 保留打开：gguf_close 统一关闭。
     * 记录方式：用 parts[0] 之后的空间不够（n_parts==1），所以这里
     * 复用 fd 字段把首个 fd 存下，其余 fd 由 segs 的局部数组承载 ——
     * 为避免复杂化，直接把多余 fd 关掉：映射已建立，pread 不再需要
     * （engram 在 byte_split 模式走映射读取）。 */
    for (uint32_t i = 1; i < nseg; i++) close(segs[i].fd);
    fprintf(stderr, "gguf: byte-split %u parts, %llu bytes, %llu tensors\n",
            m->parts, (unsigned long long)m->total,
            (unsigned long long)gf->tensor_count);
    return gf;

fail_segs:
    for (uint32_t i = 0; i < nseg; i++) {
        if (segs[i].addr) munmap(segs[i].addr, segs[i].len);
        if (segs[i].fd >= 0) close(segs[i].fd);
    }
    free(gf->parts);
    free(gf);
fail_root:
    munmap(root, reserve);
    return NULL;
}

static gguf_file_t *gguf_open_split(const char *any_part, const char *prefix,
                                    uint32_t total, int width) {
    gguf_file_t *gf = (gguf_file_t *)calloc(1, sizeof(gguf_file_t));
    if (!gf) return NULL;
    gf->fd = -1;
    gf->n_parts = total;
    gf->parts = (gguf_part_t *)calloc(total, sizeof(gguf_part_t));
    if (!gf->parts) { free(gf); return NULL; }
    for (uint32_t i = 0; i < total; i++) gf->parts[i].fd = -1;

    /* 先收集所有分片的张量数与名字，最后一次性分配合并数组 */
    uint64_t merged_count = 0;
    uint32_t version = 0;
    uint64_t alignment = 0;
    for (uint32_t i = 1; i <= total; i++) {
        char path[4096];
        const int n = snprintf(path, sizeof(path), "%s-%0*u-of-%0*u.gguf",
                               prefix, width, i, width, total);
        if (n <= 0 || (size_t)n >= sizeof(path)) {
            fprintf(stderr, "gguf_open: split path too long\n");
            goto fail;
        }
        part_parse_t pp;
        if (!parse_part(path, &gf->parts[i - 1], &pp)) {
            fprintf(stderr, "gguf_open: split part %u/%u failed: %s\n", i, total, path);
            goto fail;
        }
        if (i == 1) {
            version = pp.version;
            alignment = pp.alignment;
            gf->kvs = pp.kvs;
            gf->metadata_kv_count = pp.kv_count;
            gf->version = version;
            gf->alignment = alignment;
        } else {
            /* 其余分片的 KV 丢掉：只有 part 0 是权威 */
            free_kvs(pp.kvs, pp.kv_count);
            if (pp.version != version) {
                fprintf(stderr, "gguf_open: split part %u version mismatch (%u vs %u)\n",
                        i, pp.version, version);
                free(pp.tensors);
                goto fail;
            }
            if (pp.alignment != alignment) {
                fprintf(stderr, "gguf_open: split part %u alignment mismatch (%llu vs %llu)\n",
                        i, (unsigned long long)pp.alignment, (unsigned long long)alignment);
                free(pp.tensors);
                goto fail;
            }
        }
        merged_count += pp.tensor_count;
        /* 张量先暂存到该 part 的临时槽：借用 parts[i-1] 的字段存不下，
         * 所以用一个链表式临时数组挂在 gf 上不合适 —— 这里改用
         * 二次遍历：先把每个分片的张量数量记下来，再统一拷贝。 */
        free(pp.tensors);   /* 见下方 second pass，这里只做校验 */
        pp.tensors = NULL;
    }

    /* 二次遍历：分配合并数组并填充 */
    gf->tensors = (gguf_tensor_t *)calloc(merged_count ? merged_count : 1,
                                          sizeof(gguf_tensor_t));
    if (!gf->tensors) goto fail;
    gf->tensor_count = merged_count;

    uint64_t w = 0;
    for (uint32_t i = 1; i <= total; i++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s-%0*u-of-%0*u.gguf", prefix, width, i, width, total);
        part_parse_t pp;
        gguf_part_t tmp;
        if (!parse_part(path, &tmp, &pp)) goto fail;
        /* 复用第一次已经映射好的分片：把 tmp 的映射还给 gf->parts[i-1] 之外的释放 */
        munmap(tmp.mmap_addr, tmp.mmap_size);
        close(tmp.fd);
        free_kvs(pp.kvs, pp.kv_count);

        for (uint64_t k = 0; k < pp.tensor_count; k++) {
            gguf_tensor_t t = pp.tensors[k];
            t.part = (uint16_t)(i - 1);
            t.data = (const void *)(gf->parts[i - 1].mmap_addr + t.file_offset);
            gf->tensors[w++] = t;
        }
        free(pp.tensors);
    }

    gf->fd = gf->parts[0].fd;
    gf->mmap_addr = gf->parts[0].mmap_addr;
    gf->mmap_size = gf->parts[0].mmap_size;
    gf->file_size = gf->parts[0].file_size;
    gf->tensor_data_offset = gf->parts[0].tensor_data_offset;

    /* 名字唯一性 */
    for (uint64_t a = 0; a < gf->tensor_count; a++) {
        for (uint64_t b = a + 1; b < gf->tensor_count; b++) {
            if (strcmp(gf->tensors[a].name, gf->tensors[b].name) == 0) {
                fprintf(stderr, "gguf_open: duplicate tensor across split parts: %s\n",
                        gf->tensors[a].name);
                goto fail;
            }
        }
    }

    /* 不用 MYDS_LOG_*：gguf 层被 test_gguf 单独链接，那里没有 myds_log.o。 */
    fprintf(stderr, "gguf: %u-part split, %llu tensors, prefix %s\n",
            total, (unsigned long long)gf->tensor_count, prefix);
    (void)any_part;
    return gf;

fail:
    gguf_close(gf);
    return NULL;
}

gguf_file_t *gguf_open(const char *path) {
    /* ① 字节分段（我们自己上传用的形式） */
    {
        char base[4096];
        if (byte_split_base(path, base, sizeof(base))) {
            byte_split_t m;
            if (byte_split_manifest(base, &m)) {
                return gguf_open_byte_split(base, &m);
            }
            fprintf(stderr,
                    "gguf_open: %s looks like a byte-split part but %s.parts is missing "
                    "or invalid; refusing rather than guessing the layout\n",
                    path, base);
            return NULL;
        }
    }
    /* ② llama.cpp 的按张量 split：<prefix>-00001-of-00005.gguf */
    char prefix[4096];
    uint32_t idx = 0, total = 0;
    int width = 0;
    if (split_parse_name(path, prefix, sizeof(prefix), &idx, &total, &width) && total > 1) {
        return gguf_open_split(path, prefix, total, width);
    }

    gguf_file_t *gf = (gguf_file_t *)calloc(1, sizeof(gguf_file_t));
    if (!gf) return NULL;
    gf->fd = -1;
    gf->n_parts = 1;
    gf->parts = (gguf_part_t *)calloc(1, sizeof(gguf_part_t));
    if (!gf->parts) { free(gf); return NULL; }
    gf->parts[0].fd = -1;

    part_parse_t pp;
    if (!parse_part(path, &gf->parts[0], &pp)) {
        free(gf->parts);
        free(gf);
        return NULL;
    }

    gf->fd = gf->parts[0].fd;
    gf->mmap_addr = gf->parts[0].mmap_addr;
    gf->mmap_size = gf->parts[0].mmap_size;
    gf->file_size = gf->parts[0].file_size;
    gf->tensor_data_offset = gf->parts[0].tensor_data_offset;
    gf->version = pp.version;
    gf->alignment = pp.alignment;
    gf->tensor_count = pp.tensor_count;
    gf->metadata_kv_count = pp.kv_count;
    gf->kvs = pp.kvs;
    gf->tensors = pp.tensors;
    return gf;
}

bool gguf_tensor_source(const gguf_file_t *f, const gguf_tensor_t *t,
                        int *fd, uint64_t *file_offset) {
    if (!f || !t) return false;
    if (f->parts && f->n_parts && t->part < f->n_parts) {
        const gguf_part_t *pt = &f->parts[t->part];
        if (pt->fd < 0 || t->file_offset >= pt->file_size) return false;
        if (fd) *fd = pt->fd;
        if (file_offset) *file_offset = t->file_offset;
        return true;
    }
    /* 兼容路径：调用方可以手工构造 gguf_file_t（单测就是这么做的，
     * 只填 fd/kvs/tensors，没有 parts）。那种结构只有单文件语义，
     * 所以按改动前的算法拼 fd + (tensor_data_offset + tensor.offset)。 */
    if (f->fd < 0) return false;
    if (fd) *fd = f->fd;
    if (file_offset) *file_offset = f->tensor_data_offset + t->offset;
    return true;
}
/* 读取张量内部的一段字节。
 *
 * 为什么必须集中到一个函数：拆片后「张量在哪个文件、偏移多少」不再等于
 * 「gf->fd + tensor_data_offset + offset」。直接 pread 的调用方会读到 part 0 的
 * 错误位置 —— 如果偏移恰好落在 part 0 范围内，还会【静默读到错数据】，
 * 那比报错更糟。
 *
 *   · 张量已映射（单文件 / 按张量分片 / 字节分段的复合映射）⇒ 直接 memcpy；
 *   · 未映射（调用方做过 unmap）⇒ 用 gguf_tensor_source 取正确分片 fd 再 pread。
 *
 * 两种情况都做边界校验，offset+len 不得越过张量末尾。 */
bool gguf_tensor_read(const gguf_file_t *f, const gguf_tensor_t *t,
                      void *dst, uint64_t offset, size_t len) {
    if (!f || !t || (len && !dst)) { errno = EINVAL; return false; }
    if (offset > t->size_bytes || (uint64_t)len > t->size_bytes - offset) {
        errno = EINVAL;
        return false;
    }
    if (!len) return true;
    if (t->data) {
        memcpy(dst, (const uint8_t *)t->data + offset, len);
        return true;
    }
    int fd = -1;
    uint64_t base = 0;
    if (!gguf_tensor_source(f, t, &fd, &base)) { errno = EINVAL; return false; }
    uint8_t *out = (uint8_t *)dst;
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, out + done, len - done, (off_t)(base + offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) { errno = EIO; return false; }
        done += (size_t)n;
    }
    return true;
}

void gguf_close(gguf_file_t *f) {
    if (!f) return;
    free_kvs(f->kvs, f->metadata_kv_count);
    free(f->tensors);
    /* 2026-10-25 split：逐个分片解映射与关 fd。
     * 单文件时 n_parts == 1，行为与改动前一致（它只映射一次）。 */
    if (f->parts) {
        for (uint32_t i = 0; i < f->n_parts; i++) {
            if (f->parts[i].mmap_addr) munmap(f->parts[i].mmap_addr, f->parts[i].mmap_size);
            if (f->parts[i].fd >= 0) close(f->parts[i].fd);
        }
        free(f->parts);
    }
    free(f);
}

const gguf_kv_t *gguf_find_key(const gguf_file_t *f, const char *key) {
    if (!f || !f->kvs || !key) return NULL;
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
        if (strcmp(f->kvs[i].key, key) == 0) return &f->kvs[i];
    }
    return NULL;
}

const gguf_tensor_t *gguf_find_tensor(const gguf_file_t *f, const char *name) {
    if (!f || !f->tensors || !name) return NULL;
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
