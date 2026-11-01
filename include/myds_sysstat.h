#ifndef MYDS_SYSSTAT_H
#define MYDS_SYSSTAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 整机资源占用快照。所有「取不到」的字段都是负值，JSON 里输出 null。
 *
 * 绝不用 0 表示失败：0% 的 CPU 是真的空闲，0 GB 的内存是真的没用 ——
 * 把「读不到」画成 0 会让页面在采样失败时看起来一切正常。 */
typedef struct {
    int    valid;
    double cpu_percent;    /* 整机，两次采样求差；< 0 = 取不到 */
    double gpu_percent;    /* IOAccelerator Device Utilization；< 0 = 取不到 */
    double mem_used_gb;    /* active + wired + compressor；< 0 = 取不到 */
    double mem_total_gb;   /* sysctl hw.memsize；< 0 = 取不到 */
    double self_rss_gb;    /* 本进程 RSS；< 0 = 取不到 */
} myds_sysstat_t;

/* 采样，1 秒内重复调用返回缓存（采样本身有成本，页面每 3 秒轮询一次）。 */
void myds_sysstat_sample(myds_sysstat_t *out);

/* JSON 对象。缓冲不足返回 false —— 不发截断的 JSON。 */
bool myds_sysstat_json(char *out, size_t cap);

void myds_sysstat_reset_for_test(void);

#ifdef __cplusplus
}
#endif
#endif /* MYDS_SYSSTAT_H */
