/* Opt-in full top-k component fixture. Inputs are seeded score bit patterns,
 * not captured model activations. Includes the real runtime for completed CB
 * timestamps. No benchmark operation installs files or changes a service. */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <sys/stat.h>
#include <errno.h>
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#c); exit(1); } } while (0)
struct guarded_nan { myds_metal_buffer_t *owner,*view; size_t bytes; };
static struct guarded_nan alloc_nan(size_t bytes) {
    struct guarded_nan b={.bytes=bytes};
    b.owner=myds_metal_buffer_alloc(bytes+32u);REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+32u);
    b.view=myds_metal_buffer_sub_view(b.owner,16u,bytes);REQUIRE(b.view);return b;
}
static void guard_nan(struct guarded_nan b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<16;i++)REQUIRE(p[i]==0xa5&&p[16u+b.bytes+i]==0xa5);
}
static void free_nan(struct guarded_nan b) {
    guard_nan(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);
}
static uint64_t fnv_nan(const void *data,size_t bytes) {
    const uint8_t *p=data;uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static void write_bytes_nan(const char *dir,unsigned index,const char *suffix,const void *data,size_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/case-%03u-%s.bin",dir,index,suffix);
    REQUIRE(n>0&&(size_t)n<sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);
    REQUIRE(fwrite(data,1,bytes,f)==bytes);REQUIRE(!fclose(f));
}
static void write_nan(const char *dir,unsigned index,const char *suffix,struct guarded_nan b) {
    write_bytes_nan(dir,index,suffix,myds_metal_buffer_data(b.view),b.bytes);
}
static const char *modes_nan[]={"dense","separate","fused"};
static unsigned positive_nan(const char *s,unsigned max) {
    char *end=NULL;errno=0;unsigned long n=strtoul(s,&end,10);
    REQUIRE(s[0]&&s[0]!='-'&&!errno&&end&&!*end&&n>0&&n<=max);return (unsigned)n;
}
static void mode_nan(unsigned mode) {
    REQUIRE(mode<3u);REQUIRE(!setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE",mode?"1":"0",1));
    REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE",mode==2u?"1":"0",1));
}
struct shape_nan {uint32_t cols,rows,start,ratio;};
static int call_nan(struct guarded_nan ids,struct guarded_nan score,struct shape_nan s,myds_v41_gpu_scratch_t *scratch) {
    return s.ratio?myds_v41_gpu_topk_causal(ids.view,score.view,s.cols,512u,s.rows,s.start,s.ratio,scratch):
        myds_v41_gpu_topk(ids.view,score.view,s.cols,512u,scratch);
}
static uint32_t nth_nan(struct shape_nan s) {
    uint32_t max=myds_metal_kernel_max_threads(s.ratio?"kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc");
    REQUIRE(max);uint32_t nth=1u;while(nth<s.cols&&2u*nth<=max)nth*=2u;return nth;
}
static uint32_t parts_nan(struct shape_nan s) {const uint32_t nth=nth_nan(s);return (s.cols+nth-1u)/nth;}
static void flags_nan(const uint32_t *actual,struct guarded_nan score,struct shape_nan s) {
    const uint32_t nth=nth_nan(s),parts=parts_nan(s),*bits=myds_metal_buffer_data(score.view);
    for(uint32_t row=0;row<s.rows;row++)for(uint32_t part=0;part<parts;part++) {
        uint32_t width=s.ratio?(s.start+row+1u)/s.ratio:s.cols;if(width>s.cols)width=s.cols;
        uint32_t end=(part+1u)*nth;if(end>width)end=width;
        uint32_t expected=0u;
        for(uint32_t col=part*nth;col<end;col++)expected|=(bits[(size_t)row*s.cols+col]&0x7fffffffu)>0x7f800000u;
        REQUIRE(actual[(size_t)row*parts+part]==expected);
    }
}
static void fill_nan(struct guarded_nan score,struct shape_nan s,unsigned pattern) {
    uint32_t *bits=myds_metal_buffer_data(score.view);
    for(uint32_t row=0;row<s.rows;row++)for(uint32_t col=0;col<s.cols;col++) {
        float v=((int)((col*2654435761u+row*739u+17u)%4001u)-2000)/256.f;
        if(pattern==1u)v=col%2u?0.f:-0.f;
        if(pattern==2u)v=-INFINITY;
        if(pattern==3u&&col%19u==0u)v=col%2u?INFINITY:-INFINITY;
        uint32_t b;memcpy(&b,&v,4u);
        const uint32_t width=s.ratio?(s.start+row+1u)/s.ratio:s.cols;
        if(pattern==4u&&(col%511u==0u||col+1u==s.cols))b=col%2u?0x7fc12345u:0xff800001u;
        if(pattern==5u&&col>=width)b=0x7fc12345u;
        bits[(size_t)row*s.cols+col]=b;
    }
}
static void check_ids_nan(struct guarded_nan ids,struct shape_nan s) {
    const uint32_t *p=myds_metal_buffer_data(ids.view);
    for(uint32_t row=0;row<s.rows;row++) {
        uint32_t width=s.ratio?(s.start+row+1u)/s.ratio:s.cols;if(width>s.cols)width=s.cols;
        uint32_t written=width<512u?width:512u;
        for(uint32_t i=0;i<written;i++)REQUIRE(p[(size_t)row*512u+i]<width);
        for(uint32_t i=written;i<512u;i++)REQUIRE(p[(size_t)row*512u+i]==0xa5a5a5a5u);
    }
    guard_nan(ids);
}
static void parity_nan(const char *dir,unsigned index,struct shape_nan s,unsigned pattern,int concurrent) {
    struct guarded_nan score=alloc_nan((size_t)s.cols*s.rows*4u),ids[3];
    fill_nan(score,s,pattern);uint64_t score_hash=fnv_nan(myds_metal_buffer_data(score.view),score.bytes);
    write_nan(dir,index,"scores",score);
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(131077u);REQUIRE(scratch);
    for(unsigned mode=0;mode<3;mode++) {
        ids[mode]=alloc_nan((size_t)s.rows*512u*4u);mode_nan(mode);
        myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(!concurrent);
        myds_v41_gpu_topk_nan_fuse_counters(NULL,true);myds_metal_dispatch_count_reset();
        REQUIRE(myds_metal_begin_batch());myds_metal_barrier();
        REQUIRE(call_nan(ids[mode],score,s,scratch));REQUIRE(myds_metal_commit_batch());
        check_ids_nan(ids[mode],s);guard_nan(score);
        REQUIRE(score_hash==fnv_nan(myds_metal_buffer_data(score.view),score.bytes));
        REQUIRE(!memcmp(myds_metal_buffer_data(ids[0].view),myds_metal_buffer_data(ids[mode].view),ids[0].bytes));
        myds_v41_gpu_topk_nan_fuse_counters_t c;myds_v41_gpu_topk_nan_fuse_counters(&c,false);
        if(mode==2u&&parts_nan(s)>=16u)REQUIRE(c.requests==1u&&c.encoded==1u&&c.unavailable==0u&&c.classified_partitions==(uint64_t)parts_nan(s)*s.rows);
        else REQUIRE(c.requests==0u&&c.encoded==0u&&c.unavailable==0u);
        if(mode==2u&&c.encoded) {
            const size_t bytes=(size_t)parts_nan(s)*s.rows*4u;
            REQUIRE(scratch->topk_nan_partitions&&myds_metal_buffer_size(scratch->topk_nan_partitions)>=bytes);
            const uint32_t *flags=myds_metal_buffer_data(scratch->topk_nan_partitions);
            flags_nan(flags,score,s);write_bytes_nan(dir,index,"partition-flags",flags,bytes);
            printf("{\"kind\":\"partition_flags\",\"case\":%u,\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"nth\":%u,\"parts\":%u,\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\",\"cpu_ieee_or_equal\":true}\n",
                index,s.cols,s.rows,s.start,s.ratio,nth_nan(s),parts_nan(s),bytes,fnv_nan(flags,bytes));
        }
        write_nan(dir,index,modes_nan[mode],ids[mode]);
        printf("{\"kind\":\"case\",\"case\":%u,\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"pattern\":%u,\"concurrent\":%d,\"mode\":\"%s\",\"id_bytes\":%zu,\"ids_fnv1a64\":\"%016" PRIx64 "\",\"score_fnv1a64\":\"%016" PRIx64 "\",\"dispatches\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"classified_partitions\":%" PRIu64 ",\"canaries\":true,\"full_ids_equal\":true}\n",
            index,s.cols,s.rows,s.start,s.ratio,pattern,concurrent,modes_nan[mode],ids[mode].bytes,
            fnv_nan(myds_metal_buffer_data(ids[mode].view),ids[mode].bytes),score_hash,myds_metal_dispatch_count(),c.encoded,c.classified_partitions);
        fflush(stdout);
    }
    for(unsigned mode=0;mode<3;mode++)free_nan(ids[mode]);free_nan(score);myds_v41_gpu_scratch_free(scratch);
}
/* Copy every live flags plane before its next writer in the SAME command buffer.
 * CPU reads after the final commit cannot by themselves prove prior snapshots. */
static void reuse_nan(const char *dir,unsigned first,int concurrent) {
    const struct shape_nan shapes[]={{32768,1,0,0},{32768,3,32765,1},{32768,3,8190,1}};
    const unsigned patterns[]={0,4,5};struct guarded_nan scores[3],ids[3],refs[3],flags[3];
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(131077u);REQUIRE(scratch);
    myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(!concurrent);
    for(unsigned step=0;step<3;step++) {
        struct shape_nan s=shapes[step];scores[step]=alloc_nan((size_t)s.cols*s.rows*4u);
        ids[step]=alloc_nan((size_t)s.rows*512u*4u);refs[step]=alloc_nan(ids[step].bytes);
        flags[step]=alloc_nan((size_t)parts_nan(s)*s.rows*4u);fill_nan(scores[step],s,patterns[step]);
        mode_nan(0);REQUIRE(myds_metal_begin_batch());myds_metal_barrier();
        REQUIRE(call_nan(refs[step],scores[step],s,scratch));REQUIRE(myds_metal_commit_batch());
    }
    mode_nan(2);myds_v41_gpu_topk_nan_fuse_counters(NULL,true);
    REQUIRE(myds_metal_begin_batch());myds_metal_barrier();
    for(unsigned step=0;step<3;step++) {
        REQUIRE(call_nan(ids[step],scores[step],shapes[step],scratch));
        /* The fixture-only snapshot adds a RAW and a copy-reader WAR dependency. */
        myds_metal_barrier();REQUIRE(myds_metal_buffer_copy(flags[step].view,0,scratch->topk_nan_partitions,0,flags[step].bytes));
        myds_metal_barrier();
    }
    REQUIRE(myds_metal_commit_batch());myds_v41_gpu_topk_nan_fuse_counters_t c;
    myds_v41_gpu_topk_nan_fuse_counters(&c,false);REQUIRE(c.requests==3&&c.encoded==3&&c.unavailable==0);
    for(unsigned step=0;step<3;step++) {
        struct shape_nan s=shapes[step];const unsigned index=first+step;
        REQUIRE(!memcmp(myds_metal_buffer_data(ids[step].view),myds_metal_buffer_data(refs[step].view),ids[step].bytes));
        flags_nan(myds_metal_buffer_data(flags[step].view),scores[step],s);check_ids_nan(ids[step],s);
        write_nan(dir,index,"reuse-scores",scores[step]);write_nan(dir,index,"reuse-dense",refs[step]);
        write_nan(dir,index,"reuse-fused",ids[step]);write_nan(dir,index,"reuse-flags",flags[step]);
        printf("{\"kind\":\"reuse_flags\",\"case\":%u,\"step\":%u,\"concurrent\":%d,\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"pattern\":%u,\"nth\":%u,\"parts\":%u,\"flag_bytes\":%zu,\"flags_fnv1a64\":\"%016" PRIx64 "\",\"ids_fnv1a64\":\"%016" PRIx64 "\",\"score_fnv1a64\":\"%016" PRIx64 "\",\"same_command_buffer\":true,\"cpu_ieee_or_equal\":true,\"full_ids_equal\":true,\"canaries\":true}\n",
            index,step,concurrent,s.cols,s.rows,s.start,s.ratio,patterns[step],nth_nan(s),parts_nan(s),flags[step].bytes,
            fnv_nan(myds_metal_buffer_data(flags[step].view),flags[step].bytes),fnv_nan(myds_metal_buffer_data(ids[step].view),ids[step].bytes),fnv_nan(myds_metal_buffer_data(scores[step].view),scores[step].bytes));
        free_nan(scores[step]);free_nan(ids[step]);free_nan(refs[step]);free_nan(flags[step]);
    }
    myds_v41_gpu_scratch_free(scratch);fflush(stdout);
}
static void timing_nan(unsigned index,struct shape_nan s,unsigned samples,unsigned reps) {
    struct guarded_nan score=alloc_nan((size_t)s.cols*s.rows*4u),ids=alloc_nan((size_t)s.rows*512u*4u),reference=alloc_nan(ids.bytes);
    fill_nan(score,s,0);myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(131077u);REQUIRE(scratch);
    myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);
    const uint64_t score_hash=fnv_nan(myds_metal_buffer_data(score.view),score.bytes);
    mode_nan(0);REQUIRE(myds_metal_begin_batch());myds_metal_barrier();
    REQUIRE(call_nan(reference,score,s,scratch));REQUIRE(myds_metal_commit_batch());check_ids_nan(reference,s);
    for(unsigned baseline=0;baseline<2;baseline++)for(unsigned arm=0;arm<3;arm++) {
        const unsigned mode=arm==1u?2u:baseline;mode_nan(mode);
        for(unsigned sample=0;sample<=samples;sample++) {
            myds_v41_gpu_topk_nan_fuse_counters(NULL,true);myds_metal_dispatch_count_reset();
            REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;REQUIRE(cb);
            for(unsigned rep=0;rep<reps;rep++){myds_metal_barrier();REQUIRE(call_nan(ids,score,s,scratch));}
            REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
            const double start=cb.GPUStartTime,end=cb.GPUEndTime;REQUIRE(isfinite(start)&&isfinite(end)&&start>0&&end>start);
            guard_nan(score);guard_nan(ids);check_ids_nan(ids,s);
            REQUIRE(!memcmp(myds_metal_buffer_data(reference.view),myds_metal_buffer_data(ids.view),ids.bytes));
            REQUIRE(score_hash==fnv_nan(myds_metal_buffer_data(score.view),score.bytes));
            myds_v41_gpu_topk_nan_fuse_counters_t c;myds_v41_gpu_topk_nan_fuse_counters(&c,false);
            if(mode==2u)REQUIRE(c.requests==reps&&c.encoded==reps&&c.unavailable==0u);
            printf("{\"kind\":\"timing\",\"case\":%u,\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"baseline\":\"%s\",\"arm\":\"%s\",\"mode\":\"%s\",\"sample\":%u,\"warmup\":%s,\"repetitions\":%u,\"gpu_start_s\":%.12f,\"gpu_end_s\":%.12f,\"gpu_s\":%.12f,\"dispatches\":%" PRIu64 ",\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"unavailable\":%" PRIu64 ",\"ids_fnv1a64\":\"%016" PRIx64 "\",\"score_fnv1a64\":\"%016" PRIx64 "\",\"full_ids_equal\":true,\"score_unchanged\":true}\n",
                index,s.cols,s.rows,s.start,s.ratio,modes_nan[baseline],arm==0u?"a1":arm==1u?"b":"a2",modes_nan[mode],sample,sample?"false":"true",reps,start,end,end-start,myds_metal_dispatch_count(),c.requests,c.encoded,c.unavailable,fnv_nan(myds_metal_buffer_data(ids.view),ids.bytes),score_hash);
            fflush(stdout);
        }
    }
    free_nan(score);free_nan(ids);free_nan(reference);myds_v41_gpu_scratch_free(scratch);
}
int main(int argc,char **argv) {
    const char *dir=NULL,*shader=NULL;unsigned samples=5,reps=32;
    for(int i=1;i<argc;i+=2){REQUIRE(i+1<argc);
        if(!strcmp(argv[i],"--shader-dir"))shader=argv[i+1];
        else if(!strcmp(argv[i],"--output-dir"))dir=argv[i+1];
        else if(!strcmp(argv[i],"--samples"))samples=positive_nan(argv[i+1],20u);
        else if(!strcmp(argv[i],"--repetitions"))reps=positive_nan(argv[i+1],1024u);
        else REQUIRE(0);
    }
    REQUIRE(dir&&shader&&samples>=3u&&samples<=20u&&reps>=1u&&reps<=1024u);
    REQUIRE(!mkdir(dir,0700));REQUIRE(myds_metal_init(shader));
    const struct shape_nan shapes[]={{8192,1,0,0},{32768,1,0,0},{131077,1,0,0},
        {32768,3,32765,1},{32768,7,0,2},{32768,3,16382,2}};
    unsigned index=0;
    for(size_t i=0;i<sizeof(shapes)/sizeof(*shapes);i++)for(unsigned pattern=0;pattern<6;pattern++)for(int conc=0;conc<2;conc++)
        parity_nan(dir,index++,shapes[i],pattern,conc);
    for(int conc=0;conc<2;conc++)reuse_nan(dir,72u+(unsigned)conc*3u,conc);
    for(size_t i=0;i<sizeof(shapes)/sizeof(*shapes);i++)if(parts_nan(shapes[i])>=16u)timing_nan((unsigned)i,shapes[i],samples,reps);
    myds_metal_cleanup();printf("{\"kind\":\"complete\",\"cases\":%u,\"pass\":true}\n",index);return 0;
}
