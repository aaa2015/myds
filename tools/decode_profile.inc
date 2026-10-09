/* Derived from the archived CB diagnostic; current source is bound by prepare_decode_profile.
 * Isolated diagnostic only. One request owner records its existing operations.
 * No callback runs on a driver or Engram worker thread; no new GPU work/wait. */
#include <pthread.h>
#include <time.h>
#define RP_CAP 65536u
#define RP_HOST_CAP 4096u
typedef struct {
    const char *phase;
    uint32_t pos, layer, rows, kv_only, moe_rows;
    const char *stage;
} rp_context_t;
typedef struct {
    rp_context_t context;
    double created, commit_before, commit_after, wait_s;
    uint64_t dispatch_before, dispatch_count;
    uint32_t waits;
    rp_context_t last_context;
} rp_cb_t;
typedef struct { const char *name, *phase; uint32_t position; double seconds; uint64_t calls, bytes; } rp_host_t;
/* Activation is thread local so the existing residency heartbeat never reads
 * mutable request state. bench_phase has one generation owner at a time. */
static _Thread_local bool rp_active;
static rp_context_t rp_context;
static rp_cb_t rp_cbs[RP_CAP];
static rp_host_t rp_hosts[RP_HOST_CAP];
static uint32_t rp_n, rp_host_n, rp_overflow;
static NSMutableArray<id<MTLCommandBuffer>> *rp_objects;
static double rp_epoch, rp_event_wait_s;
static uint32_t rp_run;
static double rp_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static bool rp_on(void) { return rp_active; }
double myds_remaining_tick(void) { return rp_on() ? rp_now() : 0; }
void myds_remaining_host(const char *name, double start, uint64_t bytes) {
    if (!rp_on() || !start) return;
    const double elapsed = rp_now() - start;
    uint32_t i = 0;
    for (; i < rp_host_n; ++i) if (rp_hosts[i].position == rp_context.pos && !strcmp(rp_hosts[i].phase, rp_context.phase) && !strcmp(rp_hosts[i].name, name)) break;
    if (i == rp_host_n) {
        if (rp_host_n == RP_HOST_CAP) { ++rp_overflow; return; }
        rp_hosts[i].name = name; rp_hosts[i].phase = rp_context.phase;
        rp_hosts[i].position = rp_context.pos; ++rp_host_n;
    }
    rp_hosts[i].seconds += elapsed; ++rp_hosts[i].calls; rp_hosts[i].bytes += bytes;
}
void myds_remaining_context(const char *phase, uint32_t pos, uint32_t layer,
                            uint32_t rows, uint32_t kv_only, uint32_t moe_rows) {
    if (!rp_on()) return;
    rp_context = (rp_context_t){phase, pos, layer, rows, kv_only, moe_rows, "setup"};
}
static NSInteger rp_index(id<MTLCommandBuffer> cb) {
    /* The active CB is almost always the latest one. Flush waits may be older. */
    for (NSInteger i = (NSInteger)rp_n - 1; i >= 0; --i)
        if (rp_objects[(NSUInteger)i] == cb) return i;
    return -1;
}
static id<MTLCommandBuffer> rp_created(id<MTLCommandBuffer> cb) {
    if (rp_on() && cb) {
        if (rp_n == RP_CAP) ++rp_overflow;
        else {
            rp_cbs[rp_n] = (rp_cb_t){.context = rp_context, .created = rp_now(),
                .dispatch_before = myds_metal_dispatch_count()};
            [rp_objects addObject:cb]; ++rp_n;
        }
    }
    return cb;
}
static void rp_commit(id<MTLCommandBuffer> cb) {
    const NSInteger i = rp_on() ? rp_index(cb) : -1;
    if (i >= 0) {
        rp_cbs[i].commit_before = rp_now();
        rp_cbs[i].dispatch_count = myds_metal_dispatch_count() - rp_cbs[i].dispatch_before;
    }
    [cb commit];
    if (i >= 0) rp_cbs[i].commit_after = rp_now();
}
static void rp_wait(id<MTLCommandBuffer> cb) {
    const NSInteger i = rp_on() ? rp_index(cb) : -1;
    const double start = i >= 0 ? rp_now() : 0;
    [cb waitUntilCompleted];
    if (i >= 0) { rp_cbs[i].wait_s += rp_now() - start; ++rp_cbs[i].waits; }
}
#define RP_KERNEL_CAP 4096u
typedef struct { const char *name, *phase; MTLSize grid, threads; bool dispatch_threads; uint64_t calls; } rp_kernel_t;
static rp_kernel_t rp_kernels[RP_KERNEL_CAP];
static uint32_t rp_kernel_n;
static bool rp_trace_marked;
void myds_decode_diag_stage(uint32_t pos, uint32_t layer, const char *stage) {
    if (!rp_on()) return;
    rp_context.pos = pos; rp_context.layer = layer; rp_context.stage = stage;
}
static void rp_dispatch(const char *name, MTLSize grid, MTLSize threads, bool dispatch_threads) {
    if (!rp_on()) return;
    const NSInteger cb = g_batch_cb ? rp_index(g_batch_cb) : (NSInteger)rp_n - 1;
    if (cb >= 0) {
        rp_cb_t *r = &rp_cbs[cb];
        if (myds_metal_dispatch_count() == r->dispatch_before) {
            r->context = rp_context;
            id<MTLCommandBuffer> object = rp_objects[(NSUInteger)cb];
            object.label = [NSString stringWithFormat:@"decode-diag run=%u phase=%s pos=%u cb=%ld layer=%u",
                rp_run,rp_context.phase,rp_context.pos,(long)cb,rp_context.layer];
        }
        r->last_context = rp_context;
    }
    uint32_t i = 0;
    for (; i < rp_kernel_n; ++i) {
        rp_kernel_t *k = &rp_kernels[i];
        if (!strcmp(k->name,name) && !strcmp(k->phase,rp_context.phase) &&
            k->dispatch_threads==dispatch_threads && k->grid.width==grid.width &&
            k->grid.height==grid.height && k->grid.depth==grid.depth &&
            k->threads.width==threads.width && k->threads.height==threads.height &&
            k->threads.depth==threads.depth) break;
    }
    if (i==rp_kernel_n) {
        if (i==RP_KERNEL_CAP) { ++rp_overflow; return; }
        rp_kernels[i]=(rp_kernel_t){name,rp_context.phase,grid,threads,dispatch_threads,0}; ++rp_kernel_n;
    }
    ++rp_kernels[i].calls;
}
void myds_decode_diag_callback(uint32_t count) {
    if (!rp_on() || rp_run != 1 || count != 32 || rp_trace_marked || !getenv("MYDS_PROFILE_DECODE_TRACE")) return;
    rp_trace_marked = true;
    fprintf(stderr,"DECODE_DIAG_TRACE_READY run=%u position=%u monotonic=%.9f\n",rp_run,rp_context.pos,rp_now());
    fflush(stderr);
}
void myds_remaining_request_begin(uint32_t run) {
    const char *on = getenv("MYDS_PROFILE_DECODE");
    rp_active = on && !strcmp(on, "cb");
    if (!rp_active) return;
    rp_epoch = rp_now(); rp_run = run;
    rp_n = rp_host_n = rp_overflow = 0; rp_event_wait_s = 0;
    memset(rp_hosts, 0, sizeof(rp_hosts));
    rp_context = (rp_context_t){"prefill",0,0,0,0,0,"prefill"};
    rp_objects = [NSMutableArray new];
    rp_kernel_n = 0; rp_trace_marked = false;
    memset(rp_kernels, 0, sizeof(rp_kernels));
}
void myds_remaining_request_end(void) {
    if (!rp_on()) return;
    /* Generation already returned. Never add a synchronization here. */
    const char *path = getenv("MYDS_PROFILE_DECODE_FILE");
    FILE *out = path && *path ? fopen(path, "a") : stderr;
    if (!out) { perror("remaining profile fopen"); rp_active = false; rp_objects = nil; return; }
    double gpu_s = 0, waits = 0, commits = 0, encode = 0;
    uint32_t pending = 0, invalid_gpu = 0, submitted = 0;
    for (uint32_t i = 0; i < rp_n; ++i) {
        id<MTLCommandBuffer> cb = rp_objects[i];
        rp_cb_t *r = &rp_cbs[i];
        if (!r->commit_before) continue;
        ++submitted;
        const bool complete = cb.status == MTLCommandBufferStatusCompleted;
        const double gs = complete ? cb.GPUStartTime : 0;
        const double ge = complete ? cb.GPUEndTime : 0;
        const bool valid_gpu = complete && gs > 0 && ge >= gs;
        if (!complete) ++pending;
        if (!valid_gpu) ++invalid_gpu;
        const double gpu = valid_gpu ? ge - gs : 0;
        const double enc = r->commit_before - r->created;
        const double commit = r->commit_after - r->commit_before;
        gpu_s += gpu; waits += r->wait_s; commits += commit; encode += enc;
        fprintf(out, "{\"kind\":\"decode_diag_cb\",\"run\":%u,\"cb\":%u,\"phase\":\"%s\","
            "\"position\":%u,\"layer\":%u,\"rows\":%u,\"kv_only\":%u,\"moe_rows\":%u,"
            "\"created_ms\":%.9f,\"commit_before_ms\":%.9f,\"commit_after_ms\":%.9f,"
            "\"encode_wall_ms\":%.9f,\"commit_cpu_ms\":%.9f,\"wait_cpu_ms\":%.9f,\"wait_calls\":%u,"
            "\"dispatches\":%llu,\"status\":%lu,\"gpu_valid\":%s,\"gpu_start_s\":%.9f,\"gpu_end_s\":%.9f,\"gpu_span_ms\":%.9f,\"first_layer\":%u,\"last_layer\":%u}\n",
            rp_run,i,r->context.phase,r->context.pos,r->context.layer,r->context.rows,r->context.kv_only,r->context.moe_rows,
            1000*(r->created-rp_epoch),1000*(r->commit_before-rp_epoch),1000*(r->commit_after-rp_epoch),
            1000*enc,1000*commit,1000*r->wait_s,r->waits,(unsigned long long)r->dispatch_count,
            (unsigned long)cb.status,valid_gpu?"true":"false",gs,ge,1000*gpu,r->context.layer,r->last_context.layer);
    }
    for (uint32_t i = 0; i < rp_host_n; ++i)
        fprintf(out, "{\"kind\":\"decode_diag_host\",\"run\":%u,\"stage\":\"%s\",\"phase\":\"%s\",\"position\":%u,\"cpu_ms\":%.9f,\"calls\":%llu,\"bytes\":%llu}\n",
            rp_run,rp_hosts[i].name,rp_hosts[i].phase,rp_hosts[i].position,1000*rp_hosts[i].seconds,(unsigned long long)rp_hosts[i].calls,(unsigned long long)rp_hosts[i].bytes);
    for (uint32_t i = 0; i < rp_kernel_n; ++i) {
        rp_kernel_t *k = &rp_kernels[i];
        fprintf(out, "{\"kind\":\"decode_diag_dispatch_shape\",\"run\":%u,\"phase\":\"%s\","
            "\"kernel\":\"%s\",\"grid\":[%llu,%llu,%llu],\"threads\":[%llu,%llu,%llu],"
            "\"dispatch_threads\":%s,\"calls\":%llu}\n", rp_run,k->phase,k->name,
            (unsigned long long)k->grid.width,(unsigned long long)k->grid.height,(unsigned long long)k->grid.depth,
            (unsigned long long)k->threads.width,(unsigned long long)k->threads.height,(unsigned long long)k->threads.depth,
            k->dispatch_threads?"true":"false",(unsigned long long)k->calls);
    }
    fprintf(out, "{\"kind\":\"decode_diag_summary\",\"run\":%u,\"created_cbs\":%u,\"submitted_cbs\":%u,"
        "\"overflow\":%u,\"pending\":%u,\"invalid_gpu\":%u,\"gpu_span_ms\":%.9f,\"encode_wall_ms\":%.9f,"
        "\"commit_cpu_ms\":%.9f,\"wait_cpu_ms\":%.9f,\"event_wait_cpu_ms\":%.9f}\n",
        rp_run,rp_n,submitted,rp_overflow,pending,invalid_gpu,1000*gpu_s,1000*encode,1000*commits,1000*waits,1000*rp_event_wait_s);
    if (out != stderr) fclose(out);
    rp_active = false; rp_objects = nil;
}
