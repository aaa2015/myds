/* CPU-only process/Metal allocation observation. No Metal initialization,
 * command submission or waits. Peaks are 10ms observed values, not an exact
 * hardware maximum. Include after now_s(), with myds_metal.h and pthread.h. */
#include <mach/mach.h>
#include <stdatomic.h>
#include <inttypes.h>
#include <time.h>

typedef struct {
    uint64_t resident_bytes,phys_footprint_bytes,device_allocated_bytes;
    int task_valid,device_valid;
} ws_memory_value_t;
typedef struct {
    pthread_t thread; pthread_mutex_t lock;
    atomic_int stop;
    FILE *trace;
    unsigned run,phase;
    uint64_t samples,phys_peak,rss_peak,device_peak;
    int running,error;
} ws_memory_sampler_t;

static ws_memory_value_t ws_memory_read(void) {
    ws_memory_value_t v={0};
    task_vm_info_data_t info={0};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&count)==KERN_SUCCESS) {
        v.task_valid=1;v.phys_footprint_bytes=info.phys_footprint;v.resident_bytes=info.resident_size;
    }
    myds_metal_residency_stats_t stats={0};
    if(myds_metal_residency_diagnostics(&stats)) {
        v.device_valid=1;v.device_allocated_bytes=stats.device_allocated_size;
    }
    return v;
}
static void ws_memory_json(FILE *f,ws_memory_value_t v) {
    fprintf(f,"{\"task_valid\":%s,\"device_valid\":%s,\"rss_bytes\":%" PRIu64
        ",\"phys_footprint_bytes\":%" PRIu64 ",\"device_allocated_bytes\":%" PRIu64 "}",
        v.task_valid?"true":"false",v.device_valid?"true":"false",v.resident_bytes,
        v.phys_footprint_bytes,v.device_allocated_bytes);
}
static void ws_memory_sample_locked(ws_memory_sampler_t *s) {
    ws_memory_value_t v=ws_memory_read();
    if(!v.task_valid)s->error=1;
    if(s->phase) {
        if(v.phys_footprint_bytes>s->phys_peak)s->phys_peak=v.phys_footprint_bytes;
        if(v.resident_bytes>s->rss_peak)s->rss_peak=v.resident_bytes;
        if(v.device_allocated_bytes>s->device_peak)s->device_peak=v.device_allocated_bytes;
        ++s->samples;
    }
    if(fprintf(s->trace,"{\"kind\":\"memory_sample\",\"monotonic_s\":%.9f,\"run\":%u,\"phase\":%u,\"value\":",
        now_s(),s->run,s->phase)<0)s->error=1;
    ws_memory_json(s->trace,v);if(fputs("}\n",s->trace)<0)s->error=1;
}
static void *ws_memory_worker(void *opaque) {
    ws_memory_sampler_t *s=opaque;const struct timespec interval={0,10000000};
    while(!atomic_load_explicit(&s->stop,memory_order_relaxed)) {
        pthread_mutex_lock(&s->lock);ws_memory_sample_locked(s);pthread_mutex_unlock(&s->lock);
        nanosleep(&interval,NULL);
    }
    return NULL;
}
static int ws_memory_start(ws_memory_sampler_t *s,const char *directory) {
    memset(s,0,sizeof(*s));atomic_init(&s->stop,0);
    if(!directory||!*directory)return 0;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/samples.jsonl",directory);
    if(n<0||(size_t)n>=sizeof(path))return 0;
    s->trace=fopen(path,"wbx");if(!s->trace)return 0;
    if(pthread_mutex_init(&s->lock,NULL)!=0){fclose(s->trace);return 0;}
    if(pthread_create(&s->thread,NULL,ws_memory_worker,s)!=0) {
        pthread_mutex_destroy(&s->lock);fclose(s->trace);return 0;
    }
    s->running=1;return 1;
}
static void ws_memory_phase(ws_memory_sampler_t *s,unsigned run,unsigned phase) {
    pthread_mutex_lock(&s->lock);s->run=run;s->phase=phase;
    s->samples=s->phys_peak=s->rss_peak=s->device_peak=0;
    ws_memory_sample_locked(s);pthread_mutex_unlock(&s->lock);
}
static void ws_memory_report(ws_memory_sampler_t *s,FILE *out,const char *label) {
    pthread_mutex_lock(&s->lock);ws_memory_sample_locked(s);
    fprintf(out,"{\"kind\":\"memory_phase\",\"run\":%u,\"phase\":%u,\"label\":\"%s\","
        "\"interval_ms\":10,\"observed_samples\":%" PRIu64 ",\"observed_peak_phys_footprint_bytes\":%" PRIu64
        ",\"observed_peak_rss_bytes\":%" PRIu64 ",\"observed_peak_device_allocated_bytes\":%" PRIu64
        ",\"sampling_ok\":%s,\"current\":",s->run,s->phase,label,s->samples,s->phys_peak,s->rss_peak,s->device_peak,
        s->error?"false":"true");
    ws_memory_json(out,ws_memory_read());fputs("}\n",out);fflush(out);
    /* Close the reported phase while still holding the sampling lock. Any
     * worker observation before the next phase belongs to transition phase0. */
    s->phase=0;
    pthread_mutex_unlock(&s->lock);
}
static int ws_memory_stop(ws_memory_sampler_t *s) {
    if(!s->running)return 0;
    atomic_store_explicit(&s->stop,1,memory_order_relaxed);
    if(pthread_join(s->thread,NULL)!=0)s->error=1;
    if(fclose(s->trace)!=0)s->error=1;
    pthread_mutex_destroy(&s->lock);s->running=0;return !s->error;
}
