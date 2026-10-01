/*
 * vis_dup_probe.c — 量「系统提示词 / tools 是否在渲染后重复出现」
 *
 * 背景（2026-10-01 线上现象）：
 *   同一批请求里，有的 has_instructions=0/has_tools=0、prompt_chars≈21073；
 *   有的 has_instructions=1/has_tools=1、prompt_chars≈42428（≈两倍）。
 *   怀疑：当请求【同时】带 instructions/tools，而 input 历史里【又有】system 消息时，
 *   服务端会把系统提示词与工具定义各渲染一遍。
 *
 * 本探针直接数渲染结果里出现次数 —— 数字说话，不靠推断。
 */
#include <stdio.h>
#include <string.h>

#include "myds_server_internal.h"
#include "myds_vjob.h"

static int count_occurrences(const char *hay, const char *needle) {
    int n = 0;
    if (!needle || !*needle) return 0;
    for (const char *p = hay; (p = strstr(p, needle)) != NULL; p += strlen(needle)) n++;
    return n;
}

static const char *SYS = "SYS_MARKER";
static const char *TOOLNAME = "TOOLNAME_MARKER";

static void try_body(const char *label, const char *body) {
    static char prompt[262144];
    const char *why = NULL;
    myds_vjob_sink_t sink;
    memset(&sink, 0, sizeof(sink));
    int ok = parse_chat_prompt_vis(body, prompt, sizeof(prompt), &why, &sink);
    printf("  【%s】ok=%d\n", label, ok);
    if (!ok) {
        printf("       原因: %s\n", why ? why : "(无)");
    } else {
        printf("       渲染长度 %zu 字节\n", strlen(prompt));
        printf("       系统提示词出现 %d 次   工具名出现 %d 次\n",
               count_occurrences(prompt, SYS), count_occurrences(prompt, TOOLNAME));
    }
    myds_vjob_sink_free(&sink);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("==========================================================\n");
    printf("   系统提示词 / tools 重复渲染探针\n");
    printf("==========================================================\n");

    char body[65536];

    /* ① instructions + tools，input 里【没有】system 消息 —— 基准 */
    snprintf(body, sizeof(body),
             "{\"instructions\":\"%s\","
             "\"tools\":[{\"type\":\"function\",\"name\":\"%s\",\"description\":\"d\","
             "\"parameters\":{\"type\":\"object\",\"properties\":{}}}],"
             "\"input\":[{\"role\":\"user\",\"content\":\"hello\"}]}", SYS, TOOLNAME);
    try_body("① instructions+tools，历史无 system", body);

    /* ② instructions + tools，且 input 历史里【也有】system 消息（线上那种） */
    snprintf(body, sizeof(body),
             "{\"instructions\":\"%s\","
             "\"tools\":[{\"type\":\"function\",\"name\":\"%s\",\"description\":\"d\","
             "\"parameters\":{\"type\":\"object\",\"properties\":{}}}],"
             "\"input\":[{\"role\":\"system\",\"content\":\"%s\"},"
             "{\"role\":\"user\",\"content\":\"hello\"}]}", SYS, TOOLNAME, SYS);
    try_body("② instructions+tools，历史【有】system", body);

    /* ③ 只有 tools，历史里有 system —— tools 会不会重复 */
    snprintf(body, sizeof(body),
             "{\"tools\":[{\"type\":\"function\",\"name\":\"%s\",\"description\":\"d\","
             "\"parameters\":{\"type\":\"object\",\"properties\":{}}}],"
             "\"input\":[{\"role\":\"system\",\"content\":\"%s\"},"
             "{\"role\":\"user\",\"content\":\"hello\"}]}", TOOLNAME, SYS);
    try_body("③ 只有 tools，历史有 system", body);

    /* ④ instructions 与历史 system 的文本【不同】—— 该不该都保留？ */
    snprintf(body, sizeof(body),
             "{\"instructions\":\"%s\",\"input\":[{\"role\":\"system\",\"content\":\"OTHER_SYS\"},"
             "{\"role\":\"user\",\"content\":\"hello\"}]}", SYS);
    try_body("④ instructions 与历史 system 文本不同", body);
    return 0;
}
