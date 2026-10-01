/*
 * tok_probe.c — 取某个 token id 的文本（用于图片占位符）
 *
 * 为什么需要它：图片块在提示词里要占 N 个位置，这些位置的 token id 都是同一个
 * （V4.1 是 129264）。服务端把提示词当【字符串】交给引擎，所以必须知道
 * 「哪个字符串会被编成这个 id」—— 否则无法让 tokenizer 把图片块放到正确的位置。
 *
 * 用法: tok_probe <model.gguf> <token_id> [<token_id> ...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_gguf.h"
#include "myds_tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <token_id> [<token_id> ...]\n", argv[0]);
        return 2;
    }
    gguf_file_t *gf = gguf_open(argv[1]);
    if (!gf) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }
    myds_tokenizer_t *tok = myds_tokenizer_load(gf);
    if (!tok) { fprintf(stderr, "tokenizer 创建失败\n"); gguf_close(gf); return 1; }

    for (int i = 2; i < argc; i++) {
        int32_t id = (int32_t)strtol(argv[i], NULL, 10);
        const char *piece = myds_tokenizer_decode_piece(tok, id);
        printf("  token %d 的文本 = ", id);
        if (!piece) { printf("(无)\n"); continue; }
        printf("\"");
        for (const unsigned char *p = (const unsigned char *)piece; *p; p++) {
            if (*p >= 32 && *p < 127) putchar(*p);
            else printf("\\x%02x", *p);
        }
        printf("\"  (字节长 %zu)\n", strlen(piece));

        /* 验证：把这个文本编回去，应当得到【恰好一个】该 id */
        int32_t back[8] = {0};
        int32_t n = myds_tokenizer_encode(tok, piece, back, 8, false);
        printf("    编码回去 → %d 个 token:", n);
        for (int32_t k = 0; k < n && k < 8; k++) printf(" %d", back[k]);
        printf("%s\n", (n == 1 && back[0] == id) ? "   ✓ 往返一致（可安全用作占位符）"
                                                : "   ✗ 往返不一致！");
    }
    myds_tokenizer_free(tok);
    gguf_close(gf);
    return 0;
}
