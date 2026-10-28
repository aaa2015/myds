/* CPU-only tokenizer A/B worker. Invoke through bench_tokenizer.py. */
#define _POSIX_C_SOURCE 200809L
#include "myds_tokenizer.h"
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

myds_tokenizer_t *baseline_tokenizer_load(const gguf_file_t *);
void baseline_tokenizer_free(myds_tokenizer_t *);
int32_t baseline_tokenizer_encode_bytes(const myds_tokenizer_t *, const char *, size_t,
                                      int32_t *, int32_t, bool);
typedef int32_t (*encode_fn)(const myds_tokenizer_t *,const char *,size_t,int32_t *,int32_t,bool);
static encode_fn encoders[2]={baseline_tokenizer_encode_bytes,myds_tokenizer_encode_bytes};
static volatile uint64_t sink;
static unsigned correctness_calls, special_cases;
static uint64_t now_ns(void) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t)) { perror("clock_gettime");exit(2); }
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
static void *alloc(size_t n) {
    void *p=malloc(n?n:1);if(!p){perror("malloc");exit(2);}return p;
}
static int check(myds_tokenizer_t **t,const char *s,size_t n) {
    correctness_calls++;
    if(n>(size_t)INT32_MAX-2)return 0;
    int32_t cap=(int32_t)n+2;
    int32_t *a=alloc((size_t)cap*4),*b=alloc((size_t)cap*4);
    for(int bos=0;bos<2;bos++) {
        int32_t na=encoders[0](t[0],s,n,a,cap,bos),nb=encoders[1](t[1],s,n,b,cap,bos);
        if(na!=nb||na>cap||(na>=0&&memcmp(a,b,(size_t)na*4)))goto fail;
        for(int arm=0;arm<2;arm++)
            if(encoders[arm](t[arm],s,n,NULL,0,bos)!=na)goto fail;
        if(na<0)continue; /* Error parity for a vocabulary with missing bytes. */
        int32_t limits[]={0,1,na/2,na?na-1:0};
        for(size_t k=0;k<sizeof(limits)/sizeof(limits[0]);k++) {
            int32_t limit=limits[k];
            memset(a,0xa5,(size_t)cap*4);memset(b,0xa5,(size_t)cap*4);
            int32_t ra=encoders[0](t[0],s,n,a,limit,bos),rb=encoders[1](t[1],s,n,b,limit,bos);
            if(ra!=na||rb!=na||memcmp(a,b,(size_t)cap*4))goto fail;
            for(int32_t j=limit;j<cap;j++)
                if((uint32_t)a[j]!=UINT32_C(0xa5a5a5a5))goto fail;
        }
    }
    free(a);free(b);return 1;
fail:
    free(a);free(b);return 0;
}
static uint32_t random32(uint32_t *s) {*s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return *s;}
static int random_checks(myds_tokenizer_t **t,unsigned count,uint32_t seed) {
    static const char *pieces[]={"hello world ","  \t\r\n","1234567890","中文测试", "e\xcc\x81", "😀", "𠀀", "don't can't", "<|im_start|>","\xe2\x80\x83"};
    char s[4096];uint32_t state=seed?seed:1;
    if(!check(t,"",0)||!check(t,NULL,0))return 0;
    for(unsigned c=0;c<count;c++) {
        size_t n=0;
        if(c%2==0) {n=random32(&state)%sizeof(s);for(size_t j=0;j<n;j++)s[j]=(char)random32(&state);}
        else {unsigned parts=random32(&state)%100;for(unsigned j=0;j<parts;j++) {
            const char *p=pieces[random32(&state)%(sizeof(pieces)/sizeof(pieces[0]))];
            size_t len=strlen(p);if(n+len>sizeof(s))break;memcpy(s+n,p,len);n+=len;
        }}
        if(!check(t,s,n)){fprintf(stderr,"random correctness failed case %u\n",c);return 0;}
    }
    /* Real vocabulary special tokens, including adjacent specials and BOS. */
    for(uint32_t id=0;id<t[0]->vocab_size;id++)if(t[0]->vocab[id].special) {
        const myds_token_entry_t *v=&t[0]->vocab[id];
        if(v->len>(sizeof(s)-4)/2)continue;
        memcpy(s,v->text,v->len);memcpy(s+v->len,v->text,v->len);
        memcpy(s+2*v->len," x\0z",4);
        special_cases++;
        if(!check(t,s,2*v->len+4)){fprintf(stderr,"special correctness failed id %u\n",id);return 0;}
    }
    return 1;
}
static uint64_t run(int arm,myds_tokenizer_t **t,const char *s,size_t n,int32_t *out,int32_t cap,
                    unsigned repetitions,int32_t expected) {
    uint64_t start=now_ns();
    for(unsigned i=0;i<repetitions;i++) {
        int32_t got=encoders[arm](t[arm],s,n,out,cap,false);
        if(got!=expected){fprintf(stderr,"timed token count changed\n");exit(3);}
        sink+=(uint64_t)got+(got>0?(uint32_t)out[got-1]:0u);
    }
    return now_ns()-start;
}
int main(int argc,char **argv) {
    if(argc<7){fprintf(stderr,"use tools/bench_tokenizer.py\n");return 2;}
    unsigned samples=(unsigned)strtoul(argv[2],NULL,10),random_count=(unsigned)strtoul(argv[4],NULL,10);
    uint64_t minimum_ns=(uint64_t)(strtod(argv[3],NULL)*1e6);
    uint32_t seed=(uint32_t)strtoul(argv[5],NULL,10);
    if(samples<9||!minimum_ns)return 2;
    gguf_file_t *gf=gguf_open(argv[1]);if(!gf)return 2;
    myds_tokenizer_t *t[2]={baseline_tokenizer_load(gf),myds_tokenizer_load(gf)};
    if(!t[0]||!t[1]){fprintf(stderr,"tokenizer load failed\n");return 2;}
    if(!random_checks(t,random_count,seed))return 3;
    printf("{\"kind\":\"correctness\",\"scope\":\"random_and_specials\",\"random_cases\":%u,\"special_cases\":%u,\"input_cases\":%u,\"seed\":%u,\"pass\":true}\n",random_count,special_cases,correctness_calls,seed);fflush(stdout);
    for(int arg=6;arg<argc;arg++) {
        FILE *f=fopen(argv[arg],"rb");if(!f){perror(argv[arg]);return 2;}
        if(fseek(f,0,SEEK_END)){fclose(f);return 2;}long size=ftell(f);
        if(size<0||size>INT32_MAX-2){fclose(f);return 2;}rewind(f);
        size_t n=(size_t)size;char *s=alloc(n+1);
        if(fread(s,1,n,f)!=n){fclose(f);free(s);return 2;}fclose(f);s[n]=0;
        if(!check(t,s,n)){fprintf(stderr,"prompt correctness failed: %s\n",argv[arg]);return 3;}
        int32_t cap=(int32_t)n+2,*out=alloc((size_t)cap*4);
        int32_t expected=encoders[0](t[0],s,n,out,cap,false);
        if(expected<0){fprintf(stderr,"prompt cannot be encoded: %s (%d)\n",argv[arg],expected);return 3;}
        printf("{\"kind\":\"correctness\",\"scope\":\"prompt\",\"prompt_index\":%d,\"bytes\":%zu,\"tokens\":%d,\"pass\":true}\n",arg-6,n,expected);fflush(stdout);
        uint64_t warm[2];
        for(int arm=0;arm<2;arm++)warm[arm]=run(arm,t,s,n,out,cap,1,expected);
        uint64_t fastest=warm[0]<warm[1]?warm[0]:warm[1];if(!fastest)fastest=1;
        uint64_t wanted=(minimum_ns+fastest-1)/fastest;
        /* A quadratic baseline may be orders of magnitude slower. Keep the
         * common batch within roughly one second when possible rather than
         * multiplying a slow baseline by the fastest arm's huge batch count.
         * The raw calibration and actual durations make this cap visible. */
        uint64_t slowest=warm[0]>warm[1]?warm[0]:warm[1];if(!slowest)slowest=1;
        uint64_t max_batch_ns=minimum_ns>UINT64_C(1000000000)?minimum_ns:UINT64_C(1000000000);
        uint64_t bounded=max_batch_ns/slowest;if(!bounded)bounded=1;
        if(wanted>bounded)wanted=bounded;
        unsigned reps=(unsigned)(wanted>100000?100000:wanted);if(!reps)reps=1;
        /* Both arms execute the same repetition count, fixed before timing. */
        printf("{\"kind\":\"warmup\",\"prompt_index\":%d,\"baseline_ns\":%" PRIu64 ",\"candidate_ns\":%" PRIu64 ",\"repetitions\":%u}\n",arg-6,warm[0],warm[1],reps);fflush(stdout);
        for(unsigned sample=0;sample<samples;sample++)for(unsigned order=0;order<2;order++) {
            int arm=(int)((sample+order)%2);
            uint64_t elapsed=run(arm,t,s,n,out,cap,reps,expected);
            printf("{\"kind\":\"sample\",\"prompt_index\":%d,\"sample\":%u,\"order\":%u,\"arm\":\"%s\",\"repetitions\":%u,\"elapsed_ns\":%" PRIu64 ",\"ns_per_encode\":%.6f}\n",arg-6,sample,order,arm?"candidate":"baseline",reps,elapsed,(double)elapsed/reps);fflush(stdout);
        }
        free(out);free(s);
    }
    baseline_tokenizer_free(t[0]);myds_tokenizer_free(t[1]);gguf_close(gf);return 0;
}
