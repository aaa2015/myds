/*
 * vis_parse_probe.c — 本地验证：图片内容块是否被解析到「进图片分支」那一步
 *
 * 为什么需要它：端到端在远端只回一句「不支持的内容块」，而我看不到是哪一步。
 * 这个探针在本地（不需要模型/权重）直接调提示词解析器，把失败原因打出来，
 * 并且能区分两种情况：
 *   · 「图片分支根本没进去」 —— 解析层没认出 input_image
 *   · 「进去了但图片处理失败」—— 视觉编码器没加载（本地就是这个预期结果）
 *
 * 用法: vis_parse_probe
 */
#include <stdio.h>
#include <string.h>

#include "myds_server_internal.h"
#include "myds_vjob.h"

/* 最小 1x1 PNG（base64），只用来验证「解出来了、走到分支里」 */
static const char *TINY_PNG_B64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

static void try_body(const char *label, const char *body) {
    static char prompt[65536];
    const char *why = NULL;
    myds_vjob_sink_t sink;
    memset(&sink, 0, sizeof(sink));
    int ok = parse_chat_prompt_vis(body, prompt, sizeof(prompt), &why, &sink);
    printf("  【%s】ok=%d", label, ok);
    if (!ok) printf("  原因: %s", why ? why : "(无)");
    printf("\n");
    if (ok) {
        printf("       渲染出的提示词（前 220 字节）: %.220s\n", prompt);
        printf("       sink: %u 张图\n", sink.count);
    }
    myds_vjob_sink_free(&sink);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("====================================================\n");
    printf("   图片内容块解析探针\n");
    printf("====================================================\n");

    char body[8192];
    /* ① 基线：纯文本 */
    try_body("纯文本", "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]}]}");

    /* ② 图片（Responses 风格：image_url 是字符串） */
    snprintf(body, sizeof(body),
             "{\"input\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,%s\"},"
             "{\"type\":\"input_text\",\"text\":\"描述这张图\"}]}]}", TINY_PNG_B64);
    try_body("input_image(image_url 字符串)", body);

    /* ③ 图片（Chat 风格：image_url 是对象） */
    snprintf(body, sizeof(body),
             "{\"input\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"input_image\",\"image_url\":{\"url\":\"data:image/png;base64,%s\"}}]}]}",
             TINY_PNG_B64);
    try_body("input_image(image_url 对象)", body);

    /* ④ 图片在 messages 下（Chat Completions 端点走这条） */
    snprintf(body, sizeof(body),
             "{\"messages\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,%s\"}]}]}", TINY_PNG_B64);
    try_body("messages + input_image", body);

    /* ⑤ 历史里【多张图】—— 网关每轮重发完整历史时就是这个形状。
     *   第一版这里会被拒（"一次请求只支持一张图"），
     *   导致「发过图之后的任何对话」全部 400。 */
    {
        char img_part[4096];
        snprintf(img_part, sizeof(img_part),
                 "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,%s\"}",
                 TINY_PNG_B64);
        char *w = body;
        w += sprintf(w, "{\"messages\":[");
        w += sprintf(w, "{\"role\":\"user\",\"content\":[%s,{\"type\":\"text\",\"text\":\"first\"}]},", img_part);
        w += sprintf(w, "{\"role\":\"assistant\",\"content\":\"ok\"},");
        w += sprintf(w, "{\"role\":\"user\",\"content\":[%s,{\"type\":\"text\",\"text\":\"second\"}]},", img_part);
        w += sprintf(w, "{\"role\":\"assistant\",\"content\":\"ok\"},");
        w += sprintf(w, "{\"role\":\"user\",\"content\":\"now just talking\"}]}");
        *w = 0;
    }
    try_body("历史里两张图 + 后续对话", body);

    /* ⑥ 带 type:message 的消息壳（Responses 里也常见） */
    snprintf(body, sizeof(body),
             "{\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":["
             "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,%s\"}]}]}", TINY_PNG_B64);
    try_body("type=message 壳", body);
    return 0;
}
