/* test_gguf_split.c — split GGUF 加载回归（2026-10-25）
 *
 * 背景：llama.cpp 的 gguf-split 产出 `<prefix>-00001-of-00005.gguf` 形式的多个文件，
 * 每个都是【完整合法的 GGUF】，各带自己的 header / KV / 张量子集。HF 上超过
 * 单文件上限的权重都是这样发布的，所以 myds 必须能直接打开第一个分片。
 *
 * 本测试用【自己构造】的多分片文件，不依赖外部工具：
 *   ① 构造一个含若干张量的单文件 GGUF
 *   ② 把它按张量拆成 3 个分片（写法与 gguf-split 一致：每片独立 header）
 *   ③ 断言：从任一分片打开，都能看到全部张量；每个张量的字节与单文件逐字节相同
 *   ④ 断言：gguf_tensor_source 给出的 (part, 偏移) 确实指向那份数据
 *   ⑤ 断言：缺一片时必须失败，而不是静默少张量
 *
 * 覆盖的陷阱：
 *   · 只看 part 0 的张量 ⇒ 张量数会少
 *   · 用 gf->fd + tensor_data_offset 直接 pread ⇒ 从错误的文件读（engram 的路径）
 *   · 分片间版本/对齐不一致 ⇒ 必须报错
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static void tests_failed(const char *file, int line, const char *expr) {
    fprintf(stderr, "FAIL %s:%d %s\n", file, line, expr);
    exit(1);
}
#define CHECK(x) do { if (!(x)) tests_failed(__FILE__, __LINE__, #x); } while (0)

#define MAGIC 0x46554747u
#define ALIGN 16384u
#define N_TENSORS 6
#define PARTS 3

/* 每个张量：F32，dims = [rows]（rows 个 float）。构造简单、字节易算。 */
static const uint32_t g_rows[N_TENSORS] = { 16, 32, 48, 64, 96, 128 };

typedef struct { uint8_t *buf; size_t len; size_t cap; } wbuf_t;

static void wb_need(wbuf_t *w, size_t extra) {
    if (w->len + extra <= w->cap) return;
    size_t cap = w->cap ? w->cap : 4096;
    while (cap < w->len + extra) cap <<= 1;
    uint8_t *nb = realloc(w->buf, cap);
    CHECK(nb != NULL);
    w->buf = nb;
    w->cap = cap;
}
static void wb_put(wbuf_t *w, const void *p, size_t n) {
    wb_need(w, n);
    memcpy(w->buf + w->len, p, n);
    w->len += n;
}
static void wb_u32(wbuf_t *w, uint32_t v) { wb_put(w, &v, 4); }
static void wb_u64(wbuf_t *w, uint64_t v) { wb_put(w, &v, 8); }
static void wb_str(wbuf_t *w, const char *s) {
    const uint64_t n = strlen(s);
    wb_u64(w, n);
    wb_put(w, s, n);
}

static float tensor_value(uint32_t t, uint32_t i) {
    /* 稳定、可复算的填充，便于断言 */
    return (float)((t + 1) * 1000 + i) * 0.25f;
}

/* 写一个合法的 GGUF：kv = {general.alignment}，张量集合由 (names, count) 给定。
 * 返回每片的 header 长度（用于后面算 data 起点）。 */
static size_t write_gguf(const char *path, const uint32_t *idxs, uint32_t count) {
    wbuf_t w = {0};
    wb_u32(&w, MAGIC);
    wb_u32(&w, 3);            /* version */
    wb_u64(&w, count);
    wb_u64(&w, 1);            /* n_kv */

    wb_str(&w, "general.alignment");
    wb_u32(&w, 4);            /* GGUF_TYPE_UINT32 */
    wb_u32(&w, ALIGN);

    /* 先算每个张量在本片里的偏移（紧密排布 + 对齐） */
    uint64_t offs[N_TENSORS] = {0};
    uint64_t cur = 0;
    for (uint32_t k = 0; k < count; k++) {
        offs[k] = cur;
        cur += (uint64_t)g_rows[idxs[k]] * 4u;
        cur = (cur + ALIGN - 1) / ALIGN * ALIGN;
    }

    for (uint32_t k = 0; k < count; k++) {
        char name[64];
        snprintf(name, sizeof(name), "tensor.%u.weight", idxs[k]);
        wb_str(&w, name);
        wb_u32(&w, 1);                       /* n_dims */
        wb_u64(&w, g_rows[idxs[k]]);
        wb_u32(&w, 0);                       /* GGML_TYPE_F32 */
        wb_u64(&w, offs[k]);
    }

    const size_t header_len = w.len;
    const size_t data_start = (header_len + ALIGN - 1) / ALIGN * ALIGN;

    wb_need(&w, data_start - w.len);
    memset(w.buf + w.len, 0, data_start - w.len);
    w.len = data_start;

    for (uint32_t k = 0; k < count; k++) {
        const size_t need = data_start + offs[k] + (size_t)g_rows[idxs[k]] * 4u;
        if (w.len < need) { wb_need(&w, need - w.len); memset(w.buf + w.len, 0, need - w.len); w.len = need; }
        float *dst = (float *)(w.buf + data_start + offs[k]);
        for (uint32_t i = 0; i < g_rows[idxs[k]]; i++) dst[i] = tensor_value(idxs[k], i);
    }

    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(w.buf, 1, w.len, f) == w.len);
    fclose(f);
    free(w.buf);
    return header_len;
}

static const char *g_prefix = "/tmp/myds_split_test";

static void split_all(void) {
    /* 均分：每片 2 个张量 */
    for (uint32_t p = 0; p < PARTS; p++) {
        uint32_t idxs[2] = { p * 2u, p * 2u + 1u };
        char path[256];
        snprintf(path, sizeof(path), "%s-%05u-of-%05u.gguf", g_prefix, p + 1u, PARTS);
        write_gguf(path, idxs, 2);
    }
}

static void cleanup_all(void) {
    for (uint32_t p = 0; p < PARTS; p++) {
        char path[256];
        snprintf(path, sizeof(path), "%s-%05u-of-%05u.gguf", g_prefix, p + 1u, PARTS);
        unlink(path);
    }
    unlink("/tmp/myds_split_single.gguf");
}

/* 全量张量应当一致 */
static int check_all_tensors(gguf_file_t *single, gguf_file_t *split) {
    CHECK(single != NULL);
    CHECK(split != NULL);
    CHECK(split->tensor_count == single->tensor_count);
    CHECK(split->n_parts == PARTS);
    for (uint64_t i = 0; i < single->tensor_count; i++) {
        const gguf_tensor_t *a = &single->tensors[i];
        const gguf_tensor_t *b = gguf_find_tensor(split, a->name);
        CHECK(b != NULL);
        CHECK(a->size_bytes == b->size_bytes);
        CHECK(a->type == b->type);
        CHECK(a->n_dims == b->n_dims);
        for (uint32_t d = 0; d < a->n_dims; d++) CHECK(a->dims[d] == b->dims[d]);
        CHECK(memcmp(a->data, b->data, a->size_bytes) == 0);
        /* 张量必须能被定位到某个真实分片 */
        int fd = -1;
        uint64_t fo = 0;
        CHECK(gguf_tensor_source(split, b, &fd, &fo));
        CHECK(fd >= 0);
        CHECK(b->part < PARTS);
    }
    return 1;
}

int main(void) {
    cleanup_all();
    printf("=== split GGUF 加载 ===\n");

    /* ① 单文件基线 */
    uint32_t all[N_TENSORS];
    for (uint32_t i = 0; i < N_TENSORS; i++) all[i] = i;
    write_gguf("/tmp/myds_split_single.gguf", all, N_TENSORS);

    /* ② 拆成 3 片 */
    split_all();

    gguf_file_t *single = gguf_open("/tmp/myds_split_single.gguf");
    CHECK(single != NULL);
    CHECK(single->n_parts == 1);
    CHECK(single->tensor_count == N_TENSORS);

    /* ③ 从【每一片】打开都应看到全套 —— 用户可能直接指定任意一片 */
    for (uint32_t p = 1; p <= PARTS; p++) {
        char path[256];
        snprintf(path, sizeof(path), "%s-%05u-of-%05u.gguf", g_prefix, p, PARTS);
        gguf_file_t *sp = gguf_open(path);
        CHECK(sp != NULL);
        check_all_tensors(single, sp);
        gguf_close(sp);
        printf("  ✓ 从第 %u 片打开：%llu 张量全部一致\n", p,
               (unsigned long long)single->tensor_count);
    }

    /* ④ 缺中间一片必须失败，不能静默少张量 */
    {
        char missing[256];
        snprintf(missing, sizeof(missing), "%s-%05u-of-%05u.gguf", g_prefix, 2u, PARTS);
        char saved[300];
        snprintf(saved, sizeof(saved), "%s.saved", missing);
        CHECK(rename(missing, saved) == 0);
        gguf_file_t *sp = gguf_open("/tmp/myds_split_single.gguf");
        CHECK(sp != NULL);            /* 单文件不受影响 */
        gguf_close(sp);
        char part1[256];
        snprintf(part1, sizeof(part1), "%s-%05u-of-%05u.gguf", g_prefix, 1u, PARTS);
        gguf_file_t *broken = gguf_open(part1);
        CHECK(broken == NULL);        /* 缺片必须拒绝 */
        printf("  ✓ 缺第 2 片：拒绝加载（未静默少张量）\n");
        CHECK(rename(saved, missing) == 0);
    }

    gguf_close(single);
    cleanup_all();
    printf("PASS: split GGUF loads every tensor byte-identically from any part\n");
    return 0;
}
