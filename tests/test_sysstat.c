/* tests/test_sysstat.c — 系统资源采样
 *
 * 这个文件关心的是「取不到 vs 真的是 0」这条界线。
 *
 * 页面上的 0% 会被读成「空闲」，0 GB 会被读成「没用内存」。如果采样失败也输出 0，
 * 一次权限/接口变更就会让监控长期显示「一切正常」—— 比报错难查得多。
 * 所以接口契约是：取不到 → 负值 → JSON null，且必须肉眼可分辨。
 */
#include "myds_sysstat.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    printf("[FAIL] %s:%d  %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

int main(void) {
    myds_sysstat_reset_for_test();
    myds_sysstat_t s;
    myds_sysstat_sample(&s);

    /* 这台机器上五项都应该取得到；取不到是可诊断的真实故障，不该静默通过。 */
    CHECK(s.valid == 1);
    CHECK(s.cpu_percent >= 0.0 && s.cpu_percent <= 100.0);
    CHECK(s.gpu_percent >= 0.0 && s.gpu_percent <= 100.0);
    CHECK(s.mem_used_gb > 0.0);
    CHECK(s.mem_total_gb > 0.0);
    CHECK(s.self_rss_gb > 0.0);
    /* 已用不可能超过总量：这条能抓住「用了错误的页大小」这类错误 */
    CHECK(s.mem_used_gb <= s.mem_total_gb);
    /* RSS 是本进程，必须小于整机已用 */
    CHECK(s.self_rss_gb <= s.mem_used_gb);

    printf("  cpu=%.1f%% gpu=%.1f%% mem=%.1f/%.1fGB rss=%.2fGB\n",
           s.cpu_percent, s.gpu_percent, s.mem_used_gb, s.mem_total_gb, s.self_rss_gb);

    /* 缓存生效：1 秒内两次采样应完全一致，否则页面 3 秒轮询会每次触发系统调用 */
    myds_sysstat_t a, b;
    myds_sysstat_sample(&a);
    myds_sysstat_sample(&b);
    CHECK(a.cpu_percent == b.cpu_percent);

    /* JSON 形状 */
    char json[512];
    CHECK(myds_sysstat_json(json, sizeof(json)));
    CHECK(strstr(json, "\"cpu_percent\":") != NULL);
    CHECK(strstr(json, "\"gpu_percent\":") != NULL);
    CHECK(strstr(json, "\"mem_total_gb\":") != NULL);
    CHECK(strstr(json, "\"self_rss_gb\":") != NULL);
    /* 五项都取得到时不该出现 null —— 出现就说明某条采样路径挂了 */
    CHECK(strstr(json, "null") == NULL);
    printf("  json=%s\n", json);

    /* 缓冲不足必须失败，不能截断 */
    char tiny[16];
    CHECK(!myds_sysstat_json(tiny, sizeof(tiny)));

    if (failures) { printf("\n%d sysstat test(s) FAILED\n", failures); return 1; }
    printf("PASS: cpu/gpu/mem sampled, cached, and never conflated with zero\n");
    return 0;
}
