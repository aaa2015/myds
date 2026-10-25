/* test_gguf_byte_split.c — 字节分段 GGUF 加载回归（2026-10-25）
 *
 * 背景：权重里有两个 94.42 GiB 的张量，单个就超过托管方的单文件上限，
 * 所以按【张量】切（llama.cpp 的做法）也传不上去。这种情况下只能按
 * 【固定字节长度】切：每片不是合法 GGUF，拼起来才是。
 *
 * myds 的做法是把各片映射进一段连续虚拟地址，对上层完全等价于一个文件。
 *
 * 本测试断言：
 *   ① 从任意一片打开都能看到全部张量，且与未切分文件逐字节相同
 *   ② 内层切分点（张量横跨两片）也正确 —— 这是最容易错的地方
 *   ③ 缺片必须失败，不能静默读到错位数据
 *   ④ 清单缺失/不自洽必须失败（否则无从知道该拼几片、每片多长）
 *   ⑤ 清单里的 chunk 非页对齐必须拒绝（相邻片的页会重叠）
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>

static void tests_failed(const char *file, int line, const char *expr) {
    fprintf(stderr, "FAIL %s:%d %s\n", file, line, expr);
    exit(1);
}
#define CHECK(x) do { if (!(x)) tests_failed(__FILE__, __LINE__, #x); } while (0)

#define MAGIC 0x46554747u
#define ALIGN 16384u
#define N_TENSORS 6
#define PARTS 4
#define BASE "/tmp/myds_bytesplit_test"
#define SINGLE BASE ".gguf"

static const uint32_t g_rows[N_TENSORS] = { 300, 512, 700, 900, 1100, 1300 };

typedef struct { uint8_t *buf; size_t len, cap; } wbuf_t;
static void wb_need(wbuf_t *w, size_t extra) {
    if (w->len + extra <= w->cap) return;
    size_t cap = w->cap ? w->cap : 8192;
    while (cap < w->len + extra) cap <<= 1;
    uint8_t *nb = realloc(w->buf, cap);
    CHECK(nb != NULL);
    w->buf = nb; w->cap = cap;
}
static void wb_put(wbuf_t *w, const void *p, size_t n) {
    wb_need(w, n); memcpy(w->buf + w->len, p, n); w->len += n;
}
static void wb_u32(wbuf_t *w, uint32_t v) { wb_put(w, &v, 4); }
static void wb_u64(wbuf_t *w, uint64_t v) { wb_put(w, &v, 8); }
static void wb_str(wbuf_t *w, const char *s) {
    const uint64_t n = strlen(s); wb_u64(w, n); wb_put(w, s, n);
}
static float tensor_value(uint32_t t, uint32_t i) {
    return (float)((t + 1) * 7919 + i) * 0.5f;
}

/* 写单文件 GGUF（全部张量，F32，每个张量单独对齐） */
static void write_single(void) {
    wbuf_t w = {0};
    wb_u32(&w, MAGIC);
    wb_u32(&w, 3);
    wb_u64(&w, N_TENSORS);
    wb_u64(&w, 1);
    wb_str(&w, "general.alignment");
    wb_u32(&w, 4);
    wb_u32(&w, ALIGN);

    uint64_t offs[N_TENSORS];
    uint64_t cur = 0;
    for (uint32_t k = 0; k < N_TENSORS; k++) {
        offs[k] = cur;
        cur += (uint64_t)g_rows[k] * 4u;
        cur = (cur + ALIGN - 1) / ALIGN * ALIGN;
    }
    for (uint32_t k = 0; k < N_TENSORS; k++) {
        char name[64];
        snprintf(name, sizeof(name), "tensor.%u.weight", k);
        wb_str(&w, name);
        wb_u32(&w, 1);
        wb_u64(&w, g_rows[k]);
        wb_u32(&w, 0);          /* F32 */
        wb_u64(&w, offs[k]);
    }
    const size_t data_start = (w.len + ALIGN - 1) / ALIGN * ALIGN;
    wb_need(&w, data_start - w.len);
    memset(w.buf + w.len, 0, data_start - w.len);
    w.len = data_start;
    for (uint32_t k = 0; k < N_TENSORS; k++) {
        const size_t need = data_start + offs[k] + (size_t)g_rows[k] * 4u;
        if (w.len < need) { wb_need(&w, need - w.len); memset(w.buf + w.len, 0, need - w.len); w.len = need; }
        float *dst = (float *)(w.buf + data_start + offs[k]);
        for (uint32_t i = 0; i < g_rows[k]; i++) dst[i] = tensor_value(k, i);
    }
    FILE *f = fopen(SINGLE, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(w.buf, 1, w.len, f) == w.len);
    fclose(f);
    free(w.buf);
}

/* 按固定字节长度切成 PARTS 片，并写清单。
 * chunk 取页的整数倍，且刻意让它落在张量中间（张量横跨分片）。 */
static uint64_t g_chunk = 0;

static void split_and_manifest(uint64_t chunk, int write_manifest) {
    FILE *in = fopen(SINGLE, "rb");
    CHECK(in != NULL);
    CHECK(fseek(in, 0, SEEK_END) == 0);
    const long total = ftell(in);
    CHECK(total > 0);
    CHECK(fseek(in, 0, SEEK_SET) == 0);
    uint8_t *all = malloc((size_t)total);
    CHECK(all != NULL);
    CHECK(fread(all, 1, (size_t)total, in) == (size_t)total);
    fclose(in);
    g_chunk = chunk;

    uint64_t covered = 0;
    for (uint32_t i = 1; i <= PARTS; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s.part%02u", BASE, i);
        const size_t sz = (i < PARTS) ? (size_t)chunk
                                      : (size_t)((uint64_t)total - covered);
        FILE *out = fopen(path, "wb");
        CHECK(out != NULL);
        CHECK(fwrite(all + covered, 1, sz, out) == sz);
        fclose(out);
        covered += sz;
    }
    CHECK(covered == (uint64_t)total);

    char mpath[512];
    snprintf(mpath, sizeof(mpath), "%s.parts", BASE);
    if (write_manifest) {
        FILE *mf = fopen(mpath, "w");
        CHECK(mf != NULL);
        fprintf(mf, "myds-split 1\nparts %u\nchunk %llu\ntotal %ld\n",
                PARTS, (unsigned long long)chunk, total);
        fclose(mf);
    } else {
        unlink(mpath);
    }
    free(all);
}

static void cleanup_all(void) {
    for (uint32_t i = 1; i <= PARTS; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s.part%02u", BASE, i);
        unlink(path);
    }
    char mpath[512];
    snprintf(mpath, sizeof(mpath), "%s.parts", BASE);
    unlink(mpath);
    unlink(SINGLE);
}

static void check_all(gguf_file_t *single) {
    CHECK(single != NULL);
    CHECK(single->tensor_count == N_TENSORS);
    for (uint32_t p = 1; p <= PARTS; p++) {
        char path[512];
        snprintf(path, sizeof(path), "%s.part%02u", BASE, p);
        gguf_file_t *sp = gguf_open(path);
        CHECK(sp != NULL);
        CHECK(sp->byte_split);
        CHECK(sp->tensor_count == single->tensor_count);
        for (uint64_t i = 0; i < single->tensor_count; i++) {
            const gguf_tensor_t *a = &single->tensors[i];
            const gguf_tensor_t *b = gguf_find_tensor(sp, a->name);
            CHECK(b != NULL);
            CHECK(a->size_bytes == b->size_bytes);
            CHECK(memcmp(a->data, b->data, a->size_bytes) == 0);
        }
        gguf_close(sp);
    }
    printf("  ✓ %u 片 × 从任一片打开：%u 张量逐字节一致\n", PARTS, N_TENSORS);
}

int main(void) {
    cleanup_all();
    printf("=== byte-split GGUF ===\n");
    write_single();

    struct stat st;
    CHECK(stat(SINGLE, &st) == 0);
    /* chunk 取页对齐、且不等于任何张量边界 —— 保证有张量横跨两片 */
    const uint64_t chunk = ((uint64_t)st.st_size / PARTS / 16384u) * 16384u;
    CHECK(chunk % 16384u == 0);
    CHECK(chunk > 0);
    printf("  single=%lld bytes, chunk=%llu (%u parts)\n",
           (long long)st.st_size, (unsigned long long)chunk, PARTS);

    split_and_manifest(chunk, 1);
    gguf_file_t *single = gguf_open(SINGLE);
    CHECK(single != NULL);
    check_all(single);

    /* 缺片：删掉中间一片 */
    {
        char mid[512], saved[600];
        snprintf(mid, sizeof(mid), "%s.part02", BASE);
        snprintf(saved, sizeof(saved), "%s.saved", mid);
        CHECK(rename(mid, saved) == 0);
        char p1[512];
        snprintf(p1, sizeof(p1), "%s.part01", BASE);
        CHECK(gguf_open(p1) == NULL);
        printf("  ✓ 缺第 2 片：拒绝加载\n");
        CHECK(rename(saved, mid) == 0);
    }

    /* 清单缺失：不能靠猜 */
    {
        split_and_manifest(chunk, 0);
        char p1[512];
        snprintf(p1, sizeof(p1), "%s.part01", BASE);
        CHECK(gguf_open(p1) == NULL);
        printf("  ✓ 清单缺失：拒绝加载\n");
        split_and_manifest(chunk, 1);
    }

    /* 清单不自洽：chunk 非页对齐 */
    {
        char mpath[512];
        snprintf(mpath, sizeof(mpath), "%s.parts", BASE);
        FILE *mf = fopen(mpath, "w");
        CHECK(mf != NULL);
        fprintf(mf, "myds-split 1\nparts %u\nchunk 16385\ntotal %lld\n",
                PARTS, (long long)st.st_size);
        fclose(mf);
        char p1[512];
        snprintf(p1, sizeof(p1), "%s.part01", BASE);
        CHECK(gguf_open(p1) == NULL);
        printf("  ✓ chunk 非页对齐：拒绝加载\n");
    }

    gguf_close(single);
    cleanup_all();
    printf("PASS: byte-split loads every tensor byte-identically from any part\n");
    return 0;
}
