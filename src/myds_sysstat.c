/* myds_sysstat.c — 进程与整机的 CPU / GPU / 内存占用采样
 *
 * 供 /trace/api/system 使用。纯观测，不参与推理。
 *
 * 三条数据来源，都实测过在这台机器上可用：
 *   CPU   host_processor_info(PROCESSOR_CPU_LOAD_INFO) —— 两次采样求差得到占用率
 *   GPU   IOKit IOAccelerator 的 PerformanceStatistics["Device Utilization %"]
 *         （不需要 root；powermetrics 需要 sudo，不能用在普通服务里）
 *   MEM   host_statistics64(HOST_VM_INFO64) + sysctl hw.memsize
 *
 * 为什么内存要 sysctl：host_statistics64 只给页数，没有总物理内存。
 * 第一版用 active+inactive+wire+compressor+free 反推总量，在小内存机器上
 * 凑得出 15.5GB 而实际是 16GB —— 差 3%，够让「用了 77.9%」这种读数不可信。
 *
 * 采样有成本（host_processor_info 要分配并释放一块内核内存），所以对外只有一个
 * 带缓存的接口：1 秒内重复调用返回同一份快照。
 */
#include "myds_sysstat.h"
#include "myds_log.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/processor_info.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>

#define CACHE_MS 1000

static myds_sysstat_t g_cache;
static double g_cache_at = -1.0;   /* 单调秒 */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/* 返回 -1 表示取不到（不是 0）。0 是「真的空闲」，两者不能混。 */
static double cpu_ticks_total(unsigned long long *busy, unsigned long long *total) {
    natural_t n = 0;
    processor_info_array_t info = NULL;
    mach_msg_type_number_t count = 0;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO,
                            &n, &info, &count) != KERN_SUCCESS) return -1.0;
    unsigned long long b = 0, t = 0;
    processor_cpu_load_info_t p = (processor_cpu_load_info_t)info;
    for (natural_t i = 0; i < n; ++i) {
        const unsigned long long b1 = p[i].cpu_ticks[CPU_STATE_USER] +
                                      p[i].cpu_ticks[CPU_STATE_SYSTEM] +
                                      p[i].cpu_ticks[CPU_STATE_NICE];
        b += b1;
        t += b1 + p[i].cpu_ticks[CPU_STATE_IDLE];
    }
    vm_deallocate(mach_task_self(), (vm_address_t)info, count * sizeof(integer_t));
    *busy = b;
    *total = t;
    return 0.0;
}

/* CPU 占用需要两个时间点。没有历史时先做一次 50ms 的短采样：
 * 读一次返回「暂无」会让首屏永远显示 —，而 50ms 对一次页面请求可以接受。 */
static double cpu_percent(void) {
    static unsigned long long prev_busy = 0, prev_total = 0;
    static int have_prev = 0;
    unsigned long long busy = 0, total = 0;
    if (cpu_ticks_total(&busy, &total) < 0.0) return -1.0;
    if (!have_prev) {
        usleep(50000);
        prev_busy = busy; prev_total = total;
        if (cpu_ticks_total(&busy, &total) < 0.0) return -1.0;
        have_prev = 1;
    }
    const unsigned long long db = busy - prev_busy;
    const unsigned long long dt = total - prev_total;
    prev_busy = busy;
    prev_total = total;
    if (dt == 0) return -1.0;
    return 100.0 * (double)db / (double)dt;
}

static double gpu_percent(void) {
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
            IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS) return -1.0;
    double best = -1.0;
    io_object_t svc;
    while ((svc = IOIteratorNext(it))) {
        CFMutableDictionaryRef props = NULL;
        if (IORegistryEntryCreateCFProperties(svc, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS && props) {
            CFTypeRef perf = CFDictionaryGetValue(props, CFSTR("PerformanceStatistics"));
            if (perf && CFGetTypeID(perf) == CFDictionaryGetTypeID()) {
                CFTypeRef util = CFDictionaryGetValue((CFDictionaryRef)perf,
                                                      CFSTR("Device Utilization %"));
                if (util && CFGetTypeID(util) == CFNumberGetTypeID()) {
                    int v = 0;
                    if (CFNumberGetValue((CFNumberRef)util, kCFNumberIntType, &v) && (double)v > best)
                        best = (double)v;
                }
            }
            CFRelease(props);
        }
        IOObjectRelease(svc);
    }
    IOObjectRelease(it);
    return best;
}

static uint64_t mem_total_bytes(void) {
    uint64_t v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname("hw.memsize", &v, &len, NULL, 0) != 0) return 0;
    return v;
}

/* 已用 = active + wired + compressor（不含 inactive/file-backed cache：
 * 那部分可回收，算进去会让「内存占用」长期虚高，看不出真实压力）。 */
static double mem_used_gb(void) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &c) != KERN_SUCCESS) return -1.0;
    const unsigned long long pages = (unsigned long long)vm.active_count +
                                     (unsigned long long)vm.wire_count +
                                     (unsigned long long)vm.compressor_page_count;
    return (double)pages * 16384.0 / (1024.0 * 1024.0 * 1024.0);
}

static double self_rss_gb(void) {
    mach_task_basic_info_data_t ti;
    mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&ti, &c) != KERN_SUCCESS)
        return -1.0;
    return (double)ti.resident_size / (1024.0 * 1024.0 * 1024.0);
}

void myds_sysstat_sample(myds_sysstat_t *out) {
    if (!out) return;
    const double t = now_s();
    if (g_cache_at > 0.0 && (t - g_cache_at) * 1000.0 < CACHE_MS) {
        *out = g_cache;
        return;
    }
    memset(&g_cache, 0, sizeof(g_cache));
    g_cache.cpu_percent = cpu_percent();
    g_cache.gpu_percent = gpu_percent();
    g_cache.mem_used_gb = mem_used_gb();
    g_cache.mem_total_gb = (double)mem_total_bytes() / (1024.0 * 1024.0 * 1024.0);
    g_cache.self_rss_gb = self_rss_gb();
    g_cache.valid = 1;
    g_cache_at = now_s();
    *out = g_cache;
}

void myds_sysstat_reset_for_test(void) {
    g_cache_at = -1.0;
    memset(&g_cache, 0, sizeof(g_cache));
}

/* -1 输出 null：取不到不是 0。这个文件里每一条都遵守它。 */
static void put_num(char *out, size_t cap, size_t *w, const char *key, double v, int last) {
    if (v < 0.0)
        *w += (size_t)snprintf(out + *w, cap - *w, "\"%s\":null%s", key, last ? "" : ",");
    else
        *w += (size_t)snprintf(out + *w, cap - *w, "\"%s\":%.2f%s", key, v, last ? "" : ",");
}

bool myds_sysstat_json(char *out, size_t cap) {
    if (!out || cap < 128u) return false;
    myds_sysstat_t s;
    myds_sysstat_sample(&s);
    size_t w = 0;
    w += (size_t)snprintf(out + w, cap - w, "{");
    put_num(out, cap, &w, "cpu_percent", s.cpu_percent, 0);
    put_num(out, cap, &w, "gpu_percent", s.gpu_percent, 0);
    put_num(out, cap, &w, "mem_used_gb", s.mem_used_gb, 0);
    put_num(out, cap, &w, "mem_total_gb", s.mem_total_gb, 0);
    put_num(out, cap, &w, "self_rss_gb", s.self_rss_gb, 1);
    w += (size_t)snprintf(out + w, cap - w, "}");
    return w < cap;
}
