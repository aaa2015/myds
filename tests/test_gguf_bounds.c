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

/* 2026-10-25: CHECK 在 main 里直接 `return 0`，而 main 的成功返回值也是 0
 * ⇒ 断言失败时进程仍然以 0 退出，CI 会把失败当通过。已实测：与未修复的解析器
 * 链接时它确实打印了 FAIL，但退出码是 0。
 *
 * 现在统一走 tests_failed()：打印后 exit(1)。helper 函数里的 CHECK 也一样。 */
static void tests_failed(const char *file, int line, const char *expr) {
    fprintf(stderr, "FAIL %s:%d %s\n", file, line, expr);
    exit(1);
}

#define CHECK(x) do { if (!(x)) tests_failed(__FILE__, __LINE__, #x); } while (0)

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

/* 期望 gguf_open 失败；若它返回非 NULL 则说明防护缺失。
 * 失败时直接 exit(1)（经 CHECK），不再靠 main 的返回值——那是上一版的缺陷。 */
static int expect_reject(const char *label, builder *b) {
    const char *path = "/tmp/test_gguf_bounds.gguf";
    if (!write_tmp(b, path)) {
        fprintf(stderr, "  cannot write %s\n", path);
        return 0;
    }
    gguf_file_t *gf = gguf_open(path);
    const int ok = (gf == NULL);
    if (gf) {
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

    /* case 5: kv 字符串长度超过文件（read_string 没有经过 read_bytes）
     * 报告里复现的具体形状：len = 2^64-1 ⇒ len+1 回绕成 0，
     * malloc(0) 返回有效小块，随后 memcpy 已为负大小 ⇒ ASan negative-size-param。
     * 旧实现下本用例应触发 ASan 或崩溃。 */
    {
        builder tiny;
        memset(&tiny, 0, sizeof(tiny));
        tiny.len = 0;
        put(&tiny, "GGUF", 4);
        put_u32(&tiny, 3);
        put_u64(&tiny, 0);                      /* n_tensors */
        put_u64(&tiny, 1);                      /* n_kv */
        put_u64(&tiny, UINT64_MAX);             /* len = 2^64-1 */
        CHECK(expect_reject("kv 长度 = UINT64_MAX（len+1 回绕）", &tiny));
    }

    /* case 6: 只有 header、完全没有 tensor payload。
     * mmap_size 是按页取整的，拿它校验会把这个垃圾文件当成合法。 */
    {
        builder hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.len = 0;
        put(&hdr, "GGUF", 4);
        put_u32(&hdr, 3);
        put_u64(&hdr, 1);                       /* 1 个张量 */
        put_u64(&hdr, 0);                       /* 0 个 kv */
        tensor_desc(&hdr, "t", 8, 0, 0);       /* 声明 8 个 F32 = 32 字节 */
        /* 不追加任何 payload，文件结束 */
        CHECK(expect_reject("无 tensor payload", &hdr));
    }

    /* case 7: 张量 offset 极大，与数据区起点相加回绕。
     * 旧实现下 data 指针会指到文件头。 */
    {
        gguf_header(&b, 1, 0);
        tensor_desc(&b, "wrap", 8, 0, (UINT64_MAX - 16384 + 1));
        b.len += 64;
        CHECK(expect_reject("offset 加法回绕", &b));
    }

    /* case 8: F32 元素数 = 2^62，乘 4 回绕成 0 字节。
     * 旧实现返回 size_bytes=0，边界检查全部通过。 */
    {
        gguf_header(&b, 1, 0);
        tensor_desc(&b, "f32huge", ((uint64_t)1 << 62), 0 /* F32 */, 0);
        b.len += 64;
        CHECK(expect_reject("F32 元素数×4 回绕", &b));
    }

    printf("\n%s\n", failures ? "存在未拦截的用例" : "全部边界防护生效");
    return failures ? 1 : 0;
}
