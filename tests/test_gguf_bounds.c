/* test_gguf_bounds.c — 畸形 GGUF 的边界防护回归（2026-10-25）
 *
 * 为什么需要它：公开仓库会迎来 fuzz 输入。原实现只算 `data = mmap_addr + offset`
 * 就返回，不校验任何区间；tensor 偏移越界会让后续读取崩在 mmap 之外。
 *
 * 每个用例都构造一个【最小但合法】的 GGUF 头，然后只破坏一处：
 *   case 1  tensor offset 指向文件尾之外
 *   case 2  tensor 尺寸超过剩余空间
 *   case 3  general.alignment = 0（原实现会除零）
 *   case 4  维度乘积溢出
 *   case 5  kv 声明的字符串长度超过文件（read_bytes 溢出门径）
 *
 * 断言：gguf_open 必须返回 NULL，且不崩溃。
 * ★ 这些用例在修复前会段错误或除零 —— 正是要防的。
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static int failures;

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); return 0; } } while (0)

/* ── 最小 GGUF 构造器 ── */
typedef struct {
    uint8_t buf[4096];
    size_t len;
} builder;

static void put(builder *b, const void *p, size_t n) {
    memcpy(b->buf + b->len, p, n);
    b->len += n;
}
static void put_u32(builder *b, uint32_t v) { put(b, &v, 4); }
static void put_u64(builder *b, uint64_t v) { put(b, &v, 8); }
static void put_str(builder *b, const char *s) {
    const uint64_t n = strlen(s);
    put_u64(b, n);
    put(b, s, n);
}

/* 说明：字段布局与 src/myds_gguf.c 的解析顺序严格一致。 */
static void gguf_header(builder *b, uint64_t n_tensors, uint64_t n_kv) {
    memset(b, 0, sizeof(*b));
    put(b, "GGUF", 4);
    put_u32(b, 3);            /* version */
    put_u64(b, n_tensors);
    put_u64(b, n_kv);
}

/* 一个 uint32 元数据：general.alignment */
static void kv_alignment(builder *b, uint32_t align) {
    put_str(b, "general.alignment");
    put_u32(b, 4);            /* GGUF_TYPE_UINT32 */
    put_u32(b, align);
}

/* 一个 F32 1-D tensor 描述 */
static void tensor_desc(builder *b, const char *name, uint64_t dim, uint32_t type,
                        uint64_t offset) {
    put_str(b, name);
    put_u32(b, 1);            /* n_dims */
    put_u64(b, dim);
    put_u32(b, type);
    put_u64(b, offset);
}

static int write_tmp(builder *b, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite(b->buf, 1, b->len, f);
    fclose(f);
    return 1;
}

/* 期望 gguf_open 失败；若它返回非 NULL 则说明防护缺失 */
static int expect_reject(const char *label, builder *b) {
    const char *path = "/tmp/test_gguf_bounds.gguf";
    if (!write_tmp(b, path)) { fprintf(stderr, "  cannot write %s\n", path); return 0; }
    gguf_file_t *gf = gguf_open(path);
    const int ok = (gf == NULL);
    if (gf) {
        /* 不该成功；但仍要安全关闭 */
        gguf_close(gf);
        fprintf(stderr, "  ✗ %s: gguf_open 接受了畸形文件\n", label);
        failures++;
    } else {
        printf("  ✓ %s: 已拒绝\n", label);
    }
    unlink(path);
    return ok;
}

int main(void) {
    printf("=== 畸形 GGUF 边界防护 ===\n");
    builder b;
    const size_t tail_pad = 256;   /* 给数据区留一点空间 */

    /* case 1: tensor offset 远超文件尾 */
    gguf_header(&b, 1, 0);
    tensor_desc(&b, "t", 64, 0 /* F32 */, (uint64_t)1 << 30);
    b.len += tail_pad;
    CHECK(expect_reject("offset 超出文件", &b));

    /* case 2: offset 合法但尺寸超过剩余空间（dim 巨大） */
    gguf_header(&b, 1, 0);
    tensor_desc(&b, "t", 1000000, 0, 0);
    b.len += tail_pad;
    CHECK(expect_reject("尺寸超出剩余空间", &b));

    /* case 3: alignment = 0 —— 现在被“忽略并保留默认”，文件本身合法。
     * 这里只要求【不崩溃】；崩溃（除零）才是缺陷。 */
    {
        gguf_header(&b, 1, 1);
        kv_alignment(&b, 0);
        tensor_desc(&b, "t", 8, 0, 0);
        b.len += tail_pad;
        const char *path = "/tmp/test_gguf_align0.gguf";
        CHECK(write_tmp(&b, path));
        gguf_file_t *gf = gguf_open(path);
        printf("  ✓ alignment=0: 未崩溃（%s）\n", gf ? "已接受并使用默认对齐" : "已拒绝");
        if (gf) gguf_close(gf);
        unlink(path);
    }

    /* case 4: 维度乘积溢出 —— 用 3 维极大值 */
    gguf_header(&b, 1, 0);
    put_str(&b, "big");
    put_u32(&b, 3);
    put_u64(&b, (uint64_t)1 << 40);
    put_u64(&b, (uint64_t)1 << 40);
    put_u64(&b, (uint64_t)1 << 40);
    put_u32(&b, 0);
    put_u64(&b, 0);
    b.len += tail_pad;
    CHECK(expect_reject("维度乘积溢出", &b));

    /* case 5: kv 字符串长度超过文件（read_bytes 溢出门径）
     * 注意：builder 的缓冲区是固定的 4096 字节，不能用一个巨大的长度值去
     * 触发 write，所以这里手工写一个短文件，只放 header + 一个荒谬的 key 长度。 */
    {
        builder tiny;
        memset(&tiny, 0, sizeof(tiny));
        tiny.len = 0;
        put(&tiny, "GGUF", 4);
        put_u32(&tiny, 3);
        put_u64(&tiny, 0);                      /* n_tensors */
        put_u64(&tiny, 1);                      /* n_kv */
        put_u64(&tiny, (uint64_t)1 << 40);      /* 声称的 key 长度远超文件 */
        CHECK(expect_reject("kv 长度字段越界", &tiny));
    }

    printf("\n%s\n", failures ? "存在未拦截的用例" : "全部边界防护生效");
    return failures ? 1 : 0;
}
