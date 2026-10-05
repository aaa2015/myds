#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Independent CPU-only random-row SSD reference. No GPU, mmap, full scan,
 * cache purge or model write. Distinct rows in every warm/formal sample.
 * Cached/nocache descriptor modes do not prove coldness: verify physical
 * storage-driver counters separately. Uniform table rows are not natural-
 * language Engram hashes and cannot predict complete-model throughput. */
#define ROW_BYTES 264u
#define COLS_PER_TABLE 24u
#define FNV_INIT UINT64_C(14695981039346656037)
#define FNV_PRIME UINT64_C(1099511628211)
typedef struct { uint32_t table,row; uint64_t offset; } row_t;
static uint64_t rng_state;
static uint64_t rng64(void) {
    uint64_t z=(rng_state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
static uint32_t uniform(uint32_t bound) {
    uint64_t x, threshold=(-((uint64_t)bound))%bound;
    do { x=rng64(); } while (x<threshold);
    return (uint32_t)(x%bound);
}
static uint64_t mix64(uint64_t z) {
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
static uint64_t fnv_byte(uint64_t h,uint8_t x) { return (h^x)*FNV_PRIME; }
static uint64_t fnv_u64(uint64_t h,uint64_t x) {
    for (unsigned i=0;i<8;i++) h=fnv_byte(h,(uint8_t)(x>>(i*8)));
    return h;
}
static double now_s(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) { perror("clock_gettime");exit(1); }
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}
static void json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p=(const unsigned char *)s;*p;p++) {
        if (*p=='"'||*p=='\\') putchar('\\');
        if (*p<32) printf("\\u%04x",*p); else putchar(*p);
    }
    putchar('"');
}
static void usage(FILE *f) {
    fprintf(f,"usage: bench-engram-cold --model PATH --offset0 U64 --rows0 U32 --offset1 U64 --rows1 U32 [--mode cached|nocache] [--tokens 64] [--warmup 3] [--runs 9] [--seed 42] [--row-log PATH]\n");
}
static int u64(const char *s,uint64_t *v) {
    if (!s||!*s||*s=='-'||*s=='+'||*s==' '||*s=='\t') return 0;
    char *end=NULL;errno=0;unsigned long long n=strtoull(s,&end,0);
    if (errno||!end||*end) return 0;
    *v=(uint64_t)n;return 1;
}
int main(int argc,char **argv) {
    const char *path=NULL,*mode="cached",*row_log=NULL;
    uint64_t offsets[2]={0},rows64[2]={0},tokens64=64,warm64=3,runs64=9,seed=42;
    int have_offsets[2]={0};
    for (int i=1;i<argc;i++) {
        const char *a=argv[i];
        if (!strcmp(a,"--help")) { usage(stdout);return 0; }
        if (i+1==argc) { usage(stderr);return 2; }
        const char *v=argv[++i];uint64_t *dst=NULL;
        if (!strcmp(a,"--model")) path=v;
        else if (!strcmp(a,"--mode")) mode=v;
        else if (!strcmp(a,"--row-log")) row_log=v;
        else if (!strcmp(a,"--offset0")) { dst=&offsets[0];have_offsets[0]=1; }
        else if (!strcmp(a,"--offset1")) { dst=&offsets[1];have_offsets[1]=1; }
        else if (!strcmp(a,"--rows0")) dst=&rows64[0];
        else if (!strcmp(a,"--rows1")) dst=&rows64[1];
        else if (!strcmp(a,"--tokens")) dst=&tokens64;
        else if (!strcmp(a,"--warmup")) dst=&warm64;
        else if (!strcmp(a,"--runs")) dst=&runs64;
        else if (!strcmp(a,"--seed")) dst=&seed;
        else { fprintf(stderr,"unknown option: %s\n",a);return 2; }
        if (dst&&!u64(v,dst)) { fprintf(stderr,"invalid unsigned value for %s\n",a);return 2; }
    }
    if (!path||!*path||!have_offsets[0]||!have_offsets[1]||!rows64[0]||!rows64[1]||
        rows64[0]>UINT32_MAX||rows64[1]>UINT32_MAX||!tokens64||tokens64>8192||
        !runs64||runs64>100||warm64>100||
        (strcmp(mode,"cached")&&strcmp(mode,"nocache"))) { usage(stderr);return 2; }
    uint64_t n_samples=warm64+runs64,per_sample=tokens64*48u,total=per_sample*n_samples;
    if (total>1000000u||rows64[0]<total/2||rows64[1]<total/2) {
        fprintf(stderr,"workload too large or insufficient unique rows (cap 1000000 total rows)\n");return 2;
    }
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    if (fd<0) { perror("open model read-only");return 1; }
    struct stat st;
    if (fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0) { perror("fstat regular file");close(fd);return 1; }
    uint64_t ends[2];
    for (unsigned t=0;t<2;t++) {
        if (offsets[t]>(uint64_t)INT64_MAX||rows64[t]>(UINT64_MAX-offsets[t])/ROW_BYTES||
            (ends[t]=offsets[t]+rows64[t]*ROW_BYTES)>(uint64_t)st.st_size) {
            fprintf(stderr,"table %u outside file/off_t bounds\n",t);close(fd);return 2;
        }
    }
    if (offsets[0]<ends[1]&&offsets[1]<ends[0]) { fprintf(stderr,"table ranges overlap\n");close(fd);return 2; }
    if (!strcmp(mode,"nocache")) {
#ifdef __APPLE__
        if (fcntl(fd,F_NOCACHE,1)||fcntl(fd,F_RDAHEAD,0)) { perror("F_NOCACHE/F_RDAHEAD");close(fd);return 1; }
#else
        fprintf(stderr,"nocache mode needs macOS F_NOCACHE\n");close(fd);return 2;
#endif
    }
    FILE *rowf=NULL;
    if (row_log) {
        /* Prevent a row log alias/symlink from overwriting the read-only model.
         * Exclusively create a new path instead of truncating an existing file. */
        int rfd=open(row_log,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if (rfd<0||(rowf=fdopen(rfd,"w"))==NULL) { perror("exclusive row-log create");if(rfd>=0)close(rfd);close(fd);return 1; }
    }
    size_t cap=1;
    while (cap<(size_t)total*2u) cap*=2;
    uint64_t *seen=calloc(cap,sizeof(*seen));row_t *sel=malloc((size_t)per_sample*sizeof(*sel));
    if (!seen||!sel) { fprintf(stderr,"allocation failed\n");free(seen);free(sel);if(rowf)fclose(rowf);close(fd);return 1; }
    long page_size=sysconf(_SC_PAGESIZE);
    if (page_size<=0) { fprintf(stderr,"invalid system page size\n");return 1; }
    printf("{\"kind\":\"random_engram_reference_config\",\"model\":");json_string(path);
    printf(",\"mode\":");json_string(mode);
    printf(",\"file_bytes\":%" PRIu64 ",\"offsets\":[%" PRIu64 ",%" PRIu64 "],\"rows\":[%" PRIu64 ",%" PRIu64 "],\"tokens_per_sample\":%" PRIu64 ",\"warmup\":%" PRIu64 ",\"runs\":%" PRIu64 ",\"seed\":%" PRIu64 ",\"total_unique_rows\":%" PRIu64 ",\"page_size\":%ld,\"stat_blksize_hint\":%ld,\"row_bytes\":264,\"rows_per_token_equivalent\":48,\"boundary\":\"timed serial pread plus raw checksum; row generation and row-log writes excluded; default cached/nocache does not certify physical cold reads; uniform table rows differ from natural-language hash locality\"}\n",
        (uint64_t)st.st_size,offsets[0],offsets[1],rows64[0],rows64[1],tokens64,warm64,runs64,seed,total,page_size,(long)st.st_blksize);
    fflush(stdout);rng_state=seed;
    uint64_t all_ids=FNV_INIT,all_data=FNV_INIT;
    for (uint64_t s=0;s<n_samples;s++) {
        uint64_t ids_hash=FNV_INIT,cross_page=0,cross4k=0;
        for (uint64_t j=0;j<per_sample;j++) {
            uint32_t t=(uint32_t)((j/COLS_PER_TABLE)%2u),r;uint64_t key;size_t slot;
            do {
                r=uniform((uint32_t)rows64[t]);key=(((uint64_t)t<<32)|r)+1u;
                slot=(size_t)mix64(key)&(cap-1u);
                while (seen[slot]&&seen[slot]!=key) slot=(slot+1u)&(cap-1u);
            } while (seen[slot]);
            seen[slot]=key;uint64_t off=offsets[t]+(uint64_t)r*ROW_BYTES;
            sel[j]=(row_t){t,r,off};
            ids_hash=fnv_u64(fnv_u64(fnv_u64(ids_hash,t),r),off);
            all_ids=fnv_u64(fnv_u64(fnv_u64(all_ids,t),r),off);
            cross_page+=(off%(uint64_t)page_size)+ROW_BYTES>(uint64_t)page_size;
            cross4k+=(off%4096u)+ROW_BYTES>4096u;
            if (rowf) fprintf(rowf,"%" PRIu64 ",%" PRIu64 ",%u,%u,%" PRIu64 "\n",s,j,t,r,off);
        }
        if (rowf&&fflush(rowf)) { perror("row-log fflush");return 1; }
        uint64_t payload=0,calls=0,requested=0,checksum=FNV_INIT;uint8_t raw[ROW_BYTES];
        const double start=now_s();
        for (uint64_t j=0;j<per_sample;j++) {
            size_t done=0;
            while (done<ROW_BYTES) {
                const size_t want=ROW_BYTES-done;calls++;requested+=want;
                ssize_t n=pread(fd,raw+done,want,(off_t)(sel[j].offset+done));
                if (n<0&&errno==EINTR) continue;
                if (n<=0) { fprintf(stderr,"pread failed sample=%" PRIu64 " row=%" PRIu64 " table=%u row_id=%u offset=%" PRIu64 " errno=%d returned=%zd\n",s,j,sel[j].table,sel[j].row,sel[j].offset+done,errno,n);return 1; }
                done+=(size_t)n;payload+=(uint64_t)n;
            }
            for (unsigned b=0;b<ROW_BYTES;b++) checksum=fnv_byte(checksum,raw[b]);
        }
        const double elapsed=now_s()-start;
        if (!checksum||payload!=per_sample*ROW_BYTES) { fprintf(stderr,"checksum/payload verification failed\n");return 1; }
        all_data=fnv_u64(all_data,checksum);
        printf("{\"kind\":\"random_engram_reference_sample\",\"sample\":%" PRIu64 ",\"warmup\":%s,\"token_equivalents\":%" PRIu64 ",\"unique_rows\":%" PRIu64 ",\"wall_ms\":%.9f,\"wall_ms_per_token_equivalent\":%.9f,\"logical_read_bytes\":%" PRIu64 ",\"pread_calls\":%" PRIu64 ",\"pread_requested_bytes\":%" PRIu64 ",\"row_ids_offsets_fnv1a64\":\"%016" PRIx64 "\",\"raw_rows_fnv1a64\":\"%016" PRIx64 "\",\"rows_crossing_system_page\":%" PRIu64 ",\"rows_crossing_4k_reference_block\":%" PRIu64 ",\"system_pages_touched_before_dedup\":%" PRIu64 ",\"reference_4k_blocks_touched_before_dedup\":%" PRIu64 "}\n",
            s,s<warm64?"true":"false",tokens64,per_sample,elapsed*1e3,elapsed*1e3/(double)tokens64,payload,calls,requested,ids_hash,checksum,cross_page,cross4k,per_sample+cross_page,per_sample+cross4k);
        fflush(stdout);
    }
    printf("{\"kind\":\"random_engram_reference_complete\",\"samples\":%" PRIu64 ",\"global_unique_rows\":%" PRIu64 ",\"logical_read_bytes\":%" PRIu64 ",\"all_row_ids_offsets_fnv1a64\":\"%016" PRIx64 "\",\"sample_checksums_fnv1a64\":\"%016" PRIx64 "\"}\n",n_samples,total,total*ROW_BYTES,all_ids,all_data);
    free(seen);free(sel);if(rowf&&fclose(rowf)) { perror("row-log close");close(fd);return 1; }close(fd);return 0;
}
