/* Real-runtime full ordered-ID acceptance for partition NaN SIMD shuffle.
 * Scores are seeded synthetic IEEE patterns, not model captures. --host-only
 * initializes no Metal resources. Other modes require an exclusive GPU window.
 * Original production helper is called directly for the A reference even when
 * immutable mode1 is initialized; no runtime selector mutation is used. */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <sys/stat.h>

#define NS_MAX_COLS 131077u
#define NS_TOP 512u
static const char *ns_dir,*ns_shader="bin";
static unsigned ns_samples=5,ns_repetitions=32,ns_files_count,ns_parities,ns_timings;
static int ns_host_only,ns_missing;
struct ns_buf {myds_metal_buffer_t *owner,*view;size_t bytes;};
static struct ns_buf ns_failure_scores,ns_failure_ids[3];
static void ns_dump_failure(void);
static void ns_fail(const char *file,int line,const char *condition) {
    fprintf(stderr,"FAIL %s:%d %s\n",file,line,condition);ns_dump_failure();fflush(stdout);fflush(stderr);exit(1);
}
#define NS_REQUIRE(c) do {if(!(c))ns_fail(__FILE__,__LINE__,#c);} while(0)
static struct {char name[192];size_t bytes;uint64_t hash;} ns_files[16384];
static uint64_t ns_fnv(const void *data,size_t bytes) {
    uint64_t h=UINT64_C(14695981039346656037);const uint8_t *p=data;
    for(size_t i=0;i<bytes;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static struct ns_buf ns_alloc(size_t bytes) {
    struct ns_buf b={.bytes=bytes};b.owner=myds_metal_buffer_alloc(bytes+64u);NS_REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+64u);
    b.view=myds_metal_buffer_sub_view(b.owner,32u,bytes);NS_REQUIRE(b.view);return b;
}
static void ns_guard(struct ns_buf b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(unsigned i=0;i<32;i++)NS_REQUIRE(p[i]==0xa5&&p[32u+b.bytes+i]==0xa5);
}
static void ns_free(struct ns_buf b) {ns_guard(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void ns_poison(struct ns_buf b) {memset(myds_metal_buffer_data(b.view),0xa5,b.bytes);}
static void ns_untouched(struct ns_buf b) {
    const uint8_t *p=myds_metal_buffer_data(b.view);for(size_t i=0;i<b.bytes;i++)NS_REQUIRE(p[i]==0xa5);ns_guard(b);
}
static void ns_dump(const char *role,struct ns_buf b) {
    if(!ns_dir||!b.view)return;char path[4096];if(snprintf(path,sizeof(path),"%s/failure-%s.bin",ns_dir,role)>=(int)sizeof(path))return;
    FILE *f=fopen(path,"wb");if(!f)return;size_t n=fwrite(myds_metal_buffer_data(b.view),1,b.bytes,f);int rc=fclose(f);
    if(n==b.bytes&&!rc)fprintf(stderr,"FAILURE_RAW %s bytes=%zu unverified=true\n",path,b.bytes);
}
static void ns_dump_failure(void) {
    ns_dump("scores",ns_failure_scores);const char *names[]={"dense-ids","production-ids","candidate-ids"};
    for(unsigned i=0;i<3;i++)ns_dump(names[i],ns_failure_ids[i]);
}
static void ns_exact(struct ns_buf a,struct ns_buf b) {
    NS_REQUIRE(a.bytes==b.bytes);ns_guard(a);ns_guard(b);
    if(memcmp(myds_metal_buffer_data(a.view),myds_metal_buffer_data(b.view),a.bytes)){
        ns_dump("comparison-reference",a);ns_dump("comparison-candidate",b);
        const uint32_t *x=myds_metal_buffer_data(a.view),*y=myds_metal_buffer_data(b.view);
        for(size_t i=0;i<a.bytes/4u;i++)if(x[i]!=y[i]){fprintf(stderr,"PARITY index=%zu reference=%08x candidate=%08x\n",i,x[i],y[i]);break;}
        NS_REQUIRE(0);
    }
}
static void ns_file(const char *stem,const char *role,const char *dtype,const void *data,size_t bytes,bool *first) {
    char name[192],path[4096];NS_REQUIRE(snprintf(name,sizeof(name),"%s-%s.%s",stem,role,dtype)<(int)sizeof(name));
    uint64_t hash=ns_fnv(data,bytes);unsigned found=ns_files_count;
    for(unsigned i=0;i<ns_files_count;i++)if(!strcmp(ns_files[i].name,name)){found=i;break;}
    if(found==ns_files_count){
        NS_REQUIRE(ns_files_count<sizeof(ns_files)/sizeof(ns_files[0]));
        NS_REQUIRE(snprintf(path,sizeof(path),"%s/%s",ns_dir,name)<(int)sizeof(path));FILE *f=fopen(path,"wbx");NS_REQUIRE(f);
        NS_REQUIRE(fwrite(data,1,bytes,f)==bytes);NS_REQUIRE(!fclose(f));
        strcpy(ns_files[found].name,name);ns_files[found].bytes=bytes;ns_files[found].hash=hash;ns_files_count++;
    }else NS_REQUIRE(ns_files[found].bytes==bytes&&ns_files[found].hash==hash);
    printf("%s{\"path\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",*first?"":",",name,role,dtype,bytes,hash);*first=false;
}
static void ns_raw(const char *stem,const char *role,const char *dtype,struct ns_buf b,bool *first) {
    ns_guard(b);ns_file(stem,role,dtype,myds_metal_buffer_data(b.view),b.bytes,first);
    uint8_t raw[64];const uint8_t *p=myds_metal_buffer_data(b.owner);memcpy(raw,p,32);memcpy(raw+32,p+32+b.bytes,32);
    char r[96];NS_REQUIRE(snprintf(r,sizeof(r),"%s-guards",role)<(int)sizeof(r));ns_file(stem,r,"bin",raw,sizeof(raw),first);
}
static uint32_t ns_random(uint32_t i,uint32_t seed) {
    uint32_t z=i*747796405u+seed*2891336453u;z=((z>>((z>>28u)+4u))^z)*277803737u;return (z>>22u)^z;
}
struct ns_shape {uint32_t cols,rows,start,ratio;};
static unsigned ns_reuses,ns_negatives,ns_stages;
static const char *ns_patterns[]={"seeded_finite","all_equal","signed_zero","negative_inf","mixed_inf","visible_ieee_nan","future_ieee_nan","signed_subnormal"};
static const char *ns_sort(bool causal,bool candidate) {
    return candidate?(causal?"kernel_myds_argsort_f32_i32_desc_causal_partition_nan_shuffle":"kernel_myds_argsort_f32_i32_desc_partition_nan_shuffle"):
        (causal?"kernel_myds_argsort_f32_i32_desc_causal_partition_nan":"kernel_myds_argsort_f32_i32_desc_partition_nan");
}
static uint32_t ns_width(struct ns_shape s,uint32_t row) {
    uint32_t w=s.ratio?(s.start+row+1u)/s.ratio:s.cols;return w<s.cols?w:s.cols;
}
static uint32_t ns_nth(struct ns_shape s) {
    uint32_t max=myds_metal_kernel_max_threads(s.ratio?"kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc"),nth=1u;
    NS_REQUIRE(max);while(nth<s.cols&&2u*nth<=max)nth*=2u;return nth;
}
static uint32_t ns_parts(struct ns_shape s) {return (s.cols+ns_nth(s)-1u)/ns_nth(s);}
static bool ns_eligible(struct ns_shape s) {
    const uint32_t nth=ns_nth(s),parts=ns_parts(s),last=s.cols-(parts-1u)*nth;
    const uint32_t work=(parts-1u)*512u+(last<512u?last:512u),layout=s.ratio?s.cols:work;
    return parts>=16u&&layout>1024u;
}
static void ns_fill(struct ns_buf score,struct ns_shape s,unsigned pattern,uint32_t seed) {
    NS_REQUIRE(pattern<8u);uint32_t *p=myds_metal_buffer_data(score.view);
    for(uint32_t row=0;row<s.rows;row++)for(uint32_t col=0;col<s.cols;col++){
        const uint32_t z=ns_random(col+row*s.cols,seed);float f=((int)(z%4001u)-2000)/256.f;uint32_t b;memcpy(&b,&f,4u);
        if(pattern==1u)b=0x3f800000u;
        if(pattern==2u)b=col%2u?0u:0x80000000u;
        if(pattern==3u)b=0xff800000u;
        if(pattern==4u&&col%19u==0u)b=col%2u?0x7f800000u:0xff800000u;
        if(pattern==5u&&(col==0u||col==1023u||col==1024u||col==2047u||col+1u==s.cols)){
            const uint32_t nan[]={0x7fc12345u,0xff800001u,0x7f800001u,0xffc00001u};b=nan[col%4u];
        }
        if(pattern==6u&&col>=ns_width(s,row))b=col%2u?0x7f800001u:0xffc12345u;
        if(pattern==7u)b=(col%2u?0u:0x80000000u)|(1u+z%0x007fffffu);
        p[(size_t)row*s.cols+col]=b;
    }
}
static void ns_policy(unsigned c,unsigned a) {
    myds_metal_set_concurrent((int)c);myds_metal_set_auto_barrier((int)a);
    NS_REQUIRE(myds_metal_get_concurrent()==(int)c&&myds_metal_get_auto_barrier()==(int)a);
}
static void ns_encoder(void) {NS_REQUIRE(g_batch_enc&&g_batch_enc.dispatchType==(myds_metal_get_concurrent()?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial));}
static void ns_mode(unsigned path) {
    NS_REQUIRE(path<3u);NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE",path?"1":"0",1));NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE",path?"1":"0",1));
}
static void ns_deps(struct ns_buf score,struct ns_buf ids) {
    if(!myds_metal_get_auto_barrier()){const myds_metal_buffer_t *b[]={score.view,ids.view};myds_metal_barrier_buffers(b,2u);}
}
static int ns_call(struct ns_buf ids,struct ns_buf score,struct ns_shape s,unsigned path,myds_v41_gpu_scratch_t *scratch) {
    ns_deps(score,ids);
    if(path!=2u)return topk_rows_causal(ids.view,score.view,s.cols,512u,s.rows,s.start,s.ratio,scratch);
    return s.ratio?myds_v41_gpu_topk_causal(ids.view,score.view,s.cols,512u,s.rows,s.start,s.ratio,scratch):myds_v41_gpu_topk(ids.view,score.view,s.cols,512u,scratch);
}
static void ns_reset(void) {
    myds_v41_gpu_topk_nan_shuffle_counters(NULL,true);myds_v41_gpu_topk_nan_fuse_counters(NULL,true);myds_v41_gpu_topk_prefix_counters(NULL,true);myds_metal_dispatch_count_reset();
}
static uint64_t ns_hist(const char *name) {for(int i=0;i<g_hist_used;i++)if(!strcmp(g_hist[i].name,name))return g_hist[i].n;return 0;}
static void ns_counts(unsigned calls,unsigned fuse_calls,uint64_t parts) {
    myds_v41_gpu_topk_nan_shuffle_counters_t c;myds_v41_gpu_topk_nan_shuffle_counters(&c,false);
    NS_REQUIRE(c.requests==calls&&c.encoded==calls&&!c.failures);
    myds_v41_gpu_topk_nan_fuse_counters_t f;myds_v41_gpu_topk_nan_fuse_counters(&f,false);
    NS_REQUIRE(f.requests==fuse_calls&&f.encoded==fuse_calls&&!f.unavailable&&f.classified_partitions==parts);
}
static void ns_count_json(void) {
    myds_v41_gpu_topk_nan_shuffle_counters_t c;myds_v41_gpu_topk_nan_shuffle_counters(&c,false);
    myds_v41_gpu_topk_nan_fuse_counters_t f;myds_v41_gpu_topk_nan_fuse_counters(&f,false);
    printf(",\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":%" PRIu64 ",\"nan_fuse_requests\":%" PRIu64 ",\"nan_fuse_encoded\":%" PRIu64 ",\"nan_fuse_unavailable\":%" PRIu64 ",\"classified_partitions\":%" PRIu64,c.requests,c.encoded,c.failures,f.requests,f.encoded,f.unavailable,f.classified_partitions);
}
static void ns_geometry_json(struct ns_shape s,uint32_t nth) {
    id<MTLComputePipelineState> p=g_native_pipelines[[NSString stringWithUTF8String:ns_sort(s.ratio!=0u,true)]];NS_REQUIRE(p);
    printf(",\"candidate_max_threads\":%zu,\"candidate_simd_width\":%zu,\"candidate_static_tg_bytes\":%zu,\"dynamic_tg_bytes\":%zu",(size_t)p.maxTotalThreadsPerThreadgroup,(size_t)p.threadExecutionWidth,(size_t)p.staticThreadgroupMemoryLength,((size_t)nth*8u+15u)&~(size_t)15u);
}
static void ns_ids(struct ns_buf b,struct ns_shape s) {
    const uint32_t *p=myds_metal_buffer_data(b.view);for(uint32_t row=0;row<s.rows;row++){
        const uint32_t width=ns_width(s,row),n=width<512u?width:512u;
        for(uint32_t i=0;i<n;i++)NS_REQUIRE(p[(size_t)row*512u+i]<width);
        for(uint32_t i=n;i<512u;i++)NS_REQUIRE(p[(size_t)row*512u+i]==0xa5a5a5a5u);
    }ns_guard(b);
}
static void ns_flags(const uint32_t *flags,struct ns_buf scores,struct ns_shape s,uint32_t nth) {
    const uint32_t parts=(s.cols+nth-1u)/nth,*p=myds_metal_buffer_data(scores.view);
    for(uint32_t row=0;row<s.rows;row++)for(uint32_t part=0;part<parts;part++){
        uint32_t expected=0u,end=(part+1u)*nth,width=ns_width(s,row);if(end>width)end=width;
        for(uint32_t col=part*nth;col<end;col++)expected|=(p[(size_t)row*s.cols+col]&0x7fffffffu)>0x7f800000u;
        NS_REQUIRE(flags[(size_t)row*parts+part]==expected);
    }
}
static struct ns_buf ns_flags_copy(myds_v41_gpu_scratch_t *scratch,struct ns_buf scores,struct ns_shape s) {
    const size_t bytes=(size_t)ns_parts(s)*s.rows*4u;NS_REQUIRE(scratch->topk_nan_partitions&&myds_metal_buffer_size(scratch->topk_nan_partitions)>=bytes);
    struct ns_buf b=ns_alloc(bytes);memcpy(myds_metal_buffer_data(b.view),myds_metal_buffer_data(scratch->topk_nan_partitions),bytes);
    ns_flags(myds_metal_buffer_data(b.view),scores,s,ns_nth(s));return b;
}
static void ns_run(struct ns_buf out,struct ns_buf score,struct ns_shape s,unsigned path,myds_v41_gpu_scratch_t *scratch) {
    ns_mode(path);ns_reset();const bool eligible=path&&ns_eligible(s);const char *name=ns_sort(s.ratio!=0,path==2u);const uint64_t hist=ns_hist(name);
    NS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    NS_REQUIRE(ns_call(out,score,s,path,scratch));ns_encoder();ns_deps(score,out);
    NS_REQUIRE(myds_metal_commit_batch());NS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);ns_ids(out,s);
    ns_counts(path==2u&&eligible,eligible,eligible?(uint64_t)ns_parts(s)*s.rows:0u);
    if(g_hist_enabled&&eligible)NS_REQUIRE(ns_hist(name)-hist==1u);
}
static void ns_parity(unsigned index,struct ns_shape s,unsigned pattern,unsigned policy,myds_v41_gpu_scratch_t *scratch) {
    struct ns_buf scores=ns_alloc((size_t)s.cols*s.rows*4u),out[3],flags[2]={{0}};ns_fill(scores,s,pattern,47u);
    ns_failure_scores=scores;ns_policy(policy/2u,policy%2u);const uint64_t hash=ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes);uint64_t dispatches[3];
    for(unsigned path=0;path<3;path++){
        out[path]=ns_alloc((size_t)s.rows*512u*4u);ns_failure_ids[path]=out[path];ns_run(out[path],scores,s,path,scratch);dispatches[path]=myds_metal_dispatch_count();
        ns_exact(out[0],out[path]);ns_guard(scores);NS_REQUIRE(hash==ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes));
        if(path&&ns_eligible(s))flags[path-1u]=ns_flags_copy(scratch,scores,s);
    }
    if(ns_eligible(s))ns_exact(flags[0],flags[1]);char stem[64];snprintf(stem,sizeof(stem),"case%u",index);
    printf("{\"schema_version\":1,\"kind\":\"parity\",\"case\":%u,\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"pattern\":%u,\"pattern_name\":\"%s\",\"seed\":47,\"concurrent\":%u,\"auto_barrier\":%u,\"nth\":%u,\"parts\":%u,\"reference_max_threads\":%u,\"runtime_mode\":1,\"candidate_eligible\":%s,\"actual_kernel\":\"%s\",\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"three_paths\":[\"dense\",\"production_nan_fuse\",\"candidate_shuffle\"],\"path_dispatches\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "],\"full_ids_byte_exact\":true,\"flags_ieee_exact\":true,\"input_unchanged\":true,\"canaries\":true",index,s.cols,s.rows,s.start,s.ratio,pattern,ns_patterns[pattern],policy/2u,policy%2u,ns_nth(s),ns_parts(s),myds_metal_kernel_max_threads(s.ratio?"kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc"),ns_eligible(s)?"true":"false",ns_eligible(s)?ns_sort(s.ratio!=0,true):"original_dense",dispatches[0],dispatches[1],dispatches[2]);ns_count_json();ns_geometry_json(s,ns_nth(s));
    bool first=true;printf(",\"files\":[");ns_raw(stem,"scores","f32",scores,&first);const char *names[]={"dense-ids","production-ids","candidate-ids"};
    for(unsigned i=0;i<3;i++)ns_raw(stem,names[i],"i32",out[i],&first);
    if(ns_eligible(s)){ns_raw(stem,"production-flags","u32",flags[0],&first);ns_raw(stem,"candidate-flags","u32",flags[1],&first);}puts("]}");fflush(stdout);ns_parities++;
    memset(&ns_failure_scores,0,sizeof(ns_failure_scores));memset(ns_failure_ids,0,sizeof(ns_failure_ids));
    for(unsigned i=0;i<3;i++)ns_free(out[i]);for(unsigned i=0;i<2;i++)if(flags[i].view)ns_free(flags[i]);ns_free(scores);
}
/* Snapshot dependencies are fixture-only readers, and are explicitly joined
 * before flags/scratch are overwritten. They do not prove safe unjoined reuse. */
static void ns_reuse(unsigned policy) {
    const struct ns_shape shapes[]={{32768u,1u,0u,0u},{32768u,3u,32765u,1u},{32768u,3u,8190u,1u}};
    const unsigned patterns[]={5u,6u,0u};struct ns_buf scores[3],out[3],reference[3],flags[3];
    myds_v41_gpu_scratch_t *a=myds_v41_gpu_scratch_create(NS_MAX_COLS);NS_REQUIRE(a);ns_policy(policy/2u,policy%2u);
    for(unsigned step=0;step<3;step++){
        scores[step]=ns_alloc((size_t)shapes[step].cols*shapes[step].rows*4u);ns_fill(scores[step],shapes[step],patterns[step],73u+step);
        out[step]=ns_alloc((size_t)shapes[step].rows*512u*4u);reference[step]=ns_alloc(out[step].bytes);flags[step]=ns_alloc((size_t)ns_parts(shapes[step])*shapes[step].rows*4u);
        ns_run(reference[step],scores[step],shapes[step],1u,a);
    }
    myds_v41_gpu_scratch_free(a);myds_v41_gpu_scratch_t *b=myds_v41_gpu_scratch_create(NS_MAX_COLS);NS_REQUIRE(b&&!b->topk_nan_partition_slots);
    ns_mode(2u);ns_reset();uint64_t part_count=0;NS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    for(unsigned step=0;step<3;step++){
        ns_failure_scores=scores[step];ns_failure_ids[0]=reference[step];ns_failure_ids[2]=out[step];NS_REQUIRE(ns_call(out[step],scores[step],shapes[step],2u,b));ns_encoder();ns_deps(scores[step],out[step]);
        const myds_metal_buffer_t *deps[]={b->topk_nan_partitions,flags[step].view};myds_metal_barrier_buffers(deps,2u);
        NS_REQUIRE(myds_metal_buffer_copy(flags[step].view,0u,b->topk_nan_partitions,0u,flags[step].bytes));myds_metal_barrier_buffers(deps,2u);
        NS_REQUIRE(b->topk_nan_partition_slots==(uint64_t)ns_parts(shapes[step])*shapes[step].rows);part_count+=(uint64_t)ns_parts(shapes[step])*shapes[step].rows;
    }
    const uint64_t dispatches=myds_metal_dispatch_count();NS_REQUIRE(myds_metal_commit_batch());NS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);ns_counts(3u,3u,part_count);
    printf("{\"schema_version\":1,\"kind\":\"reuse\",\"case\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"runtime_mode\":1,\"same_command_buffer\":true,\"rows_sequence\":[1,3,3],\"flags_slots_sequence\":[%u,%u,%u],\"copies_add_dependencies\":true,\"missing_snapshot_join_safety_proven\":false,\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"dispatches\":%" PRIu64 ",\"full_ids_byte_exact\":true,\"flags_ieee_exact\":true,\"canaries\":true",policy,policy/2u,policy%2u,ns_parts(shapes[0])*shapes[0].rows,ns_parts(shapes[1])*shapes[1].rows,ns_parts(shapes[2])*shapes[2].rows,dispatches);ns_count_json();printf(",\"items\":[");
    for(unsigned step=0;step<3;step++){
        ns_failure_scores=scores[step];ns_failure_ids[0]=reference[step];ns_failure_ids[2]=out[step];ns_exact(reference[step],out[step]);ns_ids(out[step],shapes[step]);ns_flags(myds_metal_buffer_data(flags[step].view),scores[step],shapes[step],ns_nth(shapes[step]));char stem[64];snprintf(stem,sizeof(stem),"reuse-p%u-step%u",policy,step);
        printf("%s{\"step\":%u,\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"pattern\":%u,\"seed\":%u,\"nth\":%u,\"parts\":%u,\"files\":[",step?",":"",step,shapes[step].cols,shapes[step].rows,shapes[step].start,shapes[step].ratio,patterns[step],73u+step,ns_nth(shapes[step]),ns_parts(shapes[step]));bool first=true;
        ns_raw(stem,"scores","f32",scores[step],&first);ns_raw(stem,"production-ids","i32",reference[step],&first);ns_raw(stem,"candidate-ids","i32",out[step],&first);ns_raw(stem,"candidate-flags","u32",flags[step],&first);printf("]}");
    }puts("]}");fflush(stdout);ns_reuses++;
    memset(&ns_failure_scores,0,sizeof(ns_failure_scores));memset(ns_failure_ids,0,sizeof(ns_failure_ids));
    for(unsigned step=0;step<3;step++){ns_free(scores[step]);ns_free(out[step]);ns_free(reference[step]);ns_free(flags[step]);}myds_v41_gpu_scratch_free(b);
}
static void ns_timing(struct ns_shape s,unsigned pattern,unsigned reps,myds_v41_gpu_scratch_t *scratch) {
    struct ns_buf scores=ns_alloc((size_t)s.cols*s.rows*4u),reference=ns_alloc((size_t)s.rows*2048u),out=ns_alloc(reference.bytes);
    ns_policy(1u,0u);g_hist_enabled=0;const char *arms[]={"A1","B","A2"};ns_failure_scores=scores;ns_failure_ids[0]=reference;ns_failure_ids[2]=out;
    for(unsigned arm=0;arm<3;arm++)for(unsigned sample=0;sample<=ns_samples;sample++){
        const uint32_t seed=9001u+sample;ns_fill(scores,s,pattern,seed);ns_poison(reference);ns_poison(out);ns_run(reference,scores,s,1u,scratch);
        const uint64_t hash=ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes);const unsigned path=arm==1u?2u:1u;ns_mode(path);ns_reset();
        NS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned rep=0;rep<reps;rep++){NS_REQUIRE(ns_call(out,scores,s,path,scratch));ns_encoder();ns_deps(scores,out);}
        const uint64_t dispatches=myds_metal_dispatch_count();NS_REQUIRE(myds_metal_commit_batch());NS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
        NS_REQUIRE(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime);
        ns_counts(arm==1u?reps:0u,reps,(uint64_t)ns_parts(s)*s.rows*reps);ns_exact(reference,out);ns_ids(out,s);ns_guard(scores);NS_REQUIRE(hash==ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes));struct ns_buf flags=ns_flags_copy(scratch,scores,s);
        char stem[128];snprintf(stem,sizeof(stem),"timing-c%u-r%u-st%u-ratio%u-p%u-reps%u-%s-s%u",s.cols,s.rows,s.start,s.ratio,pattern,reps,arms[arm],sample);
        printf("{\"schema_version\":1,\"kind\":\"gpu_timing\",\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"pattern\":%u,\"pattern_name\":\"%s\",\"input_seed\":%u,\"scope\":\"complete_prefix_topk_graph\",\"required_production_shape\":%s,\"arm\":\"%s\",\"sample\":%u,\"warmup\":%s,\"repetitions\":%u,\"concurrent\":1,\"auto_barrier\":0,\"nth\":%u,\"parts\":%u,\"runtime_mode\":1,\"path\":\"%s\",\"actual_kernel\":\"%s\",\"encoder_dispatch_type_verified\":true,\"caller_input_output_join\":true,\"prefix\":1,\"nan_fuse\":1,\"gpu_start_s\":%.12f,\"gpu_end_s\":%.12f,\"gpu_us_per_call\":%.9f,\"dispatches\":%" PRIu64 ",\"completed_cb\":true,\"cb_status\":4,\"dispatch_hist_disabled\":true,\"full_ids_byte_exact\":true,\"flags_ieee_exact\":true,\"input_unchanged\":true,\"canaries\":true,\"oracle_prepared_before_each_measured_cb\":true,\"synthetic_distribution\":true,\"cold_dram_not_proven\":true,\"fixture_not_fullmodel\":true",s.cols,s.rows,s.start,s.ratio,pattern,ns_patterns[pattern],seed,pattern==0u?"true":"false",arms[arm],sample,sample?"false":"true",reps,ns_nth(s),ns_parts(s),arm==1u?"candidate_shuffle":"production_nan_fuse",ns_sort(s.ratio!=0,arm==1u),cb.GPUStartTime,cb.GPUEndTime,(cb.GPUEndTime-cb.GPUStartTime)*1e6/reps,dispatches);ns_count_json();ns_geometry_json(s,ns_nth(s));
        bool first=true;printf(",\"files\":[");ns_raw(stem,"scores","f32",scores,&first);ns_raw(stem,"production-ids","i32",reference,&first);ns_raw(stem,"measured-ids","i32",out,&first);ns_raw(stem,"partition-flags","u32",flags,&first);puts("]}");fflush(stdout);ns_timings++;ns_free(flags);
    }
    memset(&ns_failure_scores,0,sizeof(ns_failure_scores));memset(ns_failure_ids,0,sizeof(ns_failure_ids));ns_free(scores);ns_free(reference);ns_free(out);g_hist_enabled=1;
}
@interface NSRestrictedPipeline : NSProxy {id<MTLComputePipelineState> actual;unsigned limitation;}
+(id)withPipeline:(id<MTLComputePipelineState>)p limitation:(unsigned)n;
@end
@implementation NSRestrictedPipeline
+(id)withPipeline:(id<MTLComputePipelineState>)p limitation:(unsigned)n {NSRestrictedPipeline *v=[NSRestrictedPipeline alloc];v->actual=p;v->limitation=n;return v;}
-(NSUInteger)maxTotalThreadsPerThreadgroup {return limitation==0u?64u:actual.maxTotalThreadsPerThreadgroup;}
-(NSUInteger)threadExecutionWidth {return limitation==1u?16u:actual.threadExecutionWidth;}
-(NSUInteger)staticThreadgroupMemoryLength {return limitation==2u?g_device.maxThreadgroupMemoryLength:actual.staticThreadgroupMemoryLength;}
-(NSMethodSignature *)methodSignatureForSelector:(SEL)s {return [(NSObject *)actual methodSignatureForSelector:s];}
-(void)forwardInvocation:(NSInvocation *)i {[i invokeWithTarget:actual];}
@end
static void ns_rejected(const char *reason,struct ns_buf primary,myds_metal_buffer_t *dst,struct ns_buf scores,const myds_metal_buffer_t *src,myds_v41_gpu_scratch_t *scratch,bool causal) {
    const uint64_t hash=ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes);const size_t bytes=dst?myds_metal_buffer_size(dst):0u;uint8_t *saved=bytes?malloc(bytes):NULL;NS_REQUIRE(!bytes||saved);if(saved)memcpy(saved,myds_metal_buffer_data(dst),bytes);
    ns_mode(2u);ns_reset();NS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;NS_REQUIRE(!(causal?myds_v41_gpu_topk_causal(dst,src,32768u,512u,1u,32767u,1u,scratch):myds_v41_gpu_topk(dst,src,32768u,512u,scratch)));NS_REQUIRE(!myds_metal_dispatch_count());NS_REQUIRE(myds_metal_commit_batch());NS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);ns_untouched(primary);ns_guard(scores);NS_REQUIRE(hash==ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes));if(saved)NS_REQUIRE(!memcmp(saved,myds_metal_buffer_data(dst),bytes));
    myds_v41_gpu_topk_nan_shuffle_counters_t c;myds_v41_gpu_topk_nan_shuffle_counters(&c,false);NS_REQUIRE(c.requests==1u&&!c.encoded&&c.failures==1u);char stem[64];snprintf(stem,sizeof(stem),"reject%u",ns_negatives);
    printf("{\"schema_version\":1,\"kind\":\"rejected_call\",\"case\":%u,\"reason\":\"%s\",\"causal\":%s,\"dispatches\":0,\"no_fallback\":true,\"output_untouched\":true,\"actual_writer_untouched\":true,\"actual_writer_bytes\":%zu,\"input_unchanged\":true,\"canaries\":true",ns_negatives,reason,causal?"true":"false",bytes);ns_count_json();bool first=true;printf(",\"files\":[");ns_raw(stem,"output","bin",primary,&first);ns_raw(stem,"scores","f32",scores,&first);if(saved){ns_file(stem,"writer-before","bin",saved,bytes,&first);ns_file(stem,"writer-after","bin",myds_metal_buffer_data(dst),bytes,&first);}puts("]}");fflush(stdout);ns_negatives++;free(saved);
}
static void ns_rejections(myds_v41_gpu_scratch_t *scratch) {
    struct ns_buf scores=ns_alloc(32768u*4u),out=ns_alloc(2048u);ns_fill(scores,(struct ns_shape){32768,1,0,0},0u,91u);ns_policy(1u,0u);
#define NS_REJECT(label,dst,src,sc) ns_rejected(label,out,dst,scores,src,sc,false)
    NS_REJECT("null_output",NULL,scores.view,scratch);NS_REJECT("null_scores",out.view,NULL,scratch);NS_REJECT("null_scratch",out.view,scores.view,NULL);
    myds_metal_buffer_t *v=myds_metal_buffer_sub_view(out.view,0u,2044u);NS_REQUIRE(v);NS_REJECT("short_output",v,scores.view,scratch);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(scores.view,0u,scores.bytes-4u);NS_REQUIRE(v);NS_REJECT("short_scores",out.view,v,scratch);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(out.owner,33u,out.bytes);NS_REQUIRE(v);NS_REJECT("unaligned_output",v,scores.view,scratch);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(scores.owner,33u,scores.bytes);NS_REQUIRE(v);NS_REJECT("unaligned_scores",out.view,v,scratch);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(scores.view,32u,2048u);NS_REQUIRE(v);NS_REJECT("overlapping_parent_views",v,scores.view,scratch);myds_metal_buffer_free(v);
    myds_v41_gpu_inject_topk_prefix_unavailable(true);NS_REJECT("prefix_unavailable",out.view,scores.view,scratch);myds_v41_gpu_inject_topk_prefix_unavailable(false);
    myds_v41_gpu_inject_topk_nan_fuse_unavailable(MYDS_TOPK_NAN_FUSE_UNAVAILABLE_SORT);NS_REJECT("sort_unavailable",out.view,scores.view,scratch);
    myds_v41_gpu_inject_topk_nan_fuse_unavailable(MYDS_TOPK_NAN_FUSE_UNAVAILABLE_MERGE);NS_REJECT("merge_unavailable",out.view,scores.view,scratch);myds_v41_gpu_inject_topk_nan_fuse_unavailable(0u);
    for(unsigned causal=0;causal<2;causal++){
        NSString *key=[NSString stringWithUTF8String:ns_sort(causal,true)];id<MTLComputePipelineState> saved=g_native_pipelines[key];NS_REQUIRE(saved);
        const char *names[]={"restricted_threads","restricted_simd","restricted_tg_memory"};
        for(unsigned k=0;k<3;k++){g_native_pipelines[key]=(id<MTLComputePipelineState>)[NSRestrictedPipeline withPipeline:saved limitation:k];
            ns_rejected(names[k],out,out.view,scores,scores.view,scratch,causal!=0u);g_native_pipelines[key]=saved;
        }
    }
#undef NS_REJECT
    ns_free(scores);ns_free(out);
}
/* Direct new and old partition PSOs: diagnostic nth is explicit and does not
 * claim the public whole-chain selector adopts a small ineligible row. */
static void ns_stage(uint32_t nth,unsigned pattern,bool causal) {
    struct ns_shape shape={nth>1u?2u*nth-1u:1u,causal?3u:1u,0u,causal?1u:0u};shape.start=shape.cols>3u?shape.cols-3u:0u;
    const uint32_t parts=(shape.cols+nth-1u)/nth;struct ns_buf scores=ns_alloc((size_t)shape.cols*shape.rows*4u),out[2],flags[2];ns_fill(scores,shape,pattern,131u);ns_policy(1u,0u);
    const uint64_t hash=ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes);uint64_t count[2];ns_failure_scores=scores;memset(ns_failure_ids,0,sizeof(ns_failure_ids));
    for(unsigned p=0;p<2;p++){
        out[p]=ns_alloc((size_t)shape.cols*shape.rows*4u);ns_failure_ids[p?2u:0u]=out[p];flags[p]=ns_alloc((size_t)parts*shape.rows*4u);ns_reset();const uint64_t hist=ns_hist(ns_sort(causal,p!=0u));
        struct sort_args a={.ne00=shape.cols,.ne01=shape.rows,.ne02=1,.ne03=1,.nb00=4,.nb01=(uint64_t)shape.cols*4u,.nb02=(uint64_t)shape.cols*shape.rows*4u,.nb03=(uint64_t)shape.cols*shape.rows*4u,.ne0=shape.cols,.ne1=shape.rows,.ne2=1,.ne3=1,.top_k=nth,.causal_start=shape.start,.causal_ratio=shape.ratio};
        const myds_metal_buffer_t *b[]={scores.view,out[p].view,flags[p].view};NS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;myds_metal_barrier_buffers(b,3u);
        NS_REQUIRE(dispatch(ns_sort(causal,p!=0u),&a,sizeof(a),b,3u,parts*shape.rows,1u,1u,nth,1u,1u,((size_t)nth*8u+15u)&~(size_t)15u));ns_encoder();myds_metal_barrier_buffers(b,3u);count[p]=myds_metal_dispatch_count();NS_REQUIRE(count[p]==1u);NS_REQUIRE(myds_metal_commit_batch());NS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);NS_REQUIRE(ns_hist(ns_sort(causal,p!=0u))-hist==1u);ns_flags(myds_metal_buffer_data(flags[p].view),scores,shape,nth);ns_counts(0u,0u,0u);ns_guard(scores);NS_REQUIRE(hash==ns_fnv(myds_metal_buffer_data(scores.view),scores.bytes));
    }
    ns_exact(out[0],out[1]);ns_exact(flags[0],flags[1]);char stem[64];snprintf(stem,sizeof(stem),"stage%u",ns_stages);
    printf("{\"schema_version\":1,\"kind\":\"sort_stage_parity\",\"case\":%u,\"cols\":%u,\"rows\":%u,\"start\":%u,\"ratio\":%u,\"nth\":%u,\"parts\":%u,\"pattern\":%u,\"pattern_name\":\"%s\",\"seed\":131,\"concurrent\":1,\"auto_barrier\":0,\"explicit_partition_geometry\":true,\"public_candidate_eligibility_claimed\":false,\"reference_kernel\":\"%s\",\"actual_kernel\":\"%s\",\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"path_dispatches\":[1,1],\"full_partition_ids_byte_exact\":true,\"flags_ieee_exact\":true,\"input_unchanged\":true,\"canaries\":true",ns_stages,shape.cols,shape.rows,shape.start,shape.ratio,nth,parts,pattern,ns_patterns[pattern],ns_sort(causal,false),ns_sort(causal,true));ns_count_json();ns_geometry_json(shape,nth);bool first=true;printf(",\"files\":[");ns_raw(stem,"scores","f32",scores,&first);ns_raw(stem,"production-partition-ids","i32",out[0],&first);ns_raw(stem,"candidate-partition-ids","i32",out[1],&first);ns_raw(stem,"production-flags","u32",flags[0],&first);ns_raw(stem,"candidate-flags","u32",flags[1],&first);puts("]}");fflush(stdout);ns_stages++;memset(&ns_failure_scores,0,sizeof(ns_failure_scores));memset(ns_failure_ids,0,sizeof(ns_failure_ids));for(unsigned p=0;p<2;p++){ns_free(out[p]);ns_free(flags[p]);}ns_free(scores);
}
/* Host injected commit failure happens after real GPU completion. Encoding
 * counters describe the graph, and never upgrade the outer failed return. */
static void ns_completion(myds_v41_gpu_scratch_t *scratch) {
    const struct ns_shape shape={32768,1,0,0};struct ns_buf scores=ns_alloc(32768u*4u),reference=ns_alloc(2048u),out=ns_alloc(2048u);ns_fill(scores,shape,0u,417u);ns_policy(1u,0u);ns_run(reference,scores,shape,1u,scratch);ns_mode(2u);ns_reset();
    NS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;NS_REQUIRE(ns_call(out,scores,shape,2u,scratch));ns_encoder();ns_deps(scores,out);myds_metal_inject_commit_failure(1u);NS_REQUIRE(!myds_metal_commit_batch());myds_metal_inject_commit_failure(0u);NS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);ns_counts(1u,1u,ns_parts(shape));ns_exact(reference,out);struct ns_buf flags=ns_flags_copy(scratch,scores,shape);
    printf("{\"schema_version\":1,\"kind\":\"completion_contract\",\"cols\":32768,\"rows\":1,\"start\":0,\"ratio\":0,\"pattern\":0,\"seed\":417,\"nth\":%u,\"parts\":%u,\"runtime_mode\":1,\"encoded_is_not_completion_success\":true,\"injected_host_commit_failure\":true,\"real_gpu_failure\":false,\"outer_commit_return\":0,\"completed_cb\":true,\"cb_status\":4,\"full_ids_byte_exact\":true,\"flags_ieee_exact\":true,\"canaries\":true",ns_nth(shape),ns_parts(shape));ns_count_json();bool first=true;printf(",\"files\":[");ns_raw("completion","scores","f32",scores,&first);ns_raw("completion","production-ids","i32",reference,&first);ns_raw("completion","candidate-ids","i32",out,&first);ns_raw("completion","candidate-flags","u32",flags,&first);puts("]}");fflush(stdout);ns_free(scores);ns_free(reference);ns_free(out);ns_free(flags);
}
static void ns_clean(void) {NS_REQUIRE(!g_initialized&&!g_device&&!g_library&&!g_queue&&!g_indexer_topk_nan_shuffle&&!myds_v41_gpu_topk_nan_shuffle_enabled());}
static void ns_ambient(void) {
    const char *off[]={"MYDS_BATCH2_Q8_SHARED","MYDS_MOE_Q2_SUM_CACHE","MYDS_MOE_Q2_SUM_CACHE_FUSED","MYDS_MOE_IQ2_CONSTANT_LUT","MYDS_INDEXER_HEAD_SLOTS","MYDS_INDEXER_COMPACT_SCORER","MYDS_INDEXER_TOPK_THRESHOLD","MYDS_RESIDENCY_BYPASS","MYDS_V41_WEIGHT_STAGE","MYDS_INDEXER_BATCH_FLOATS","MYDS_ATTN_KEYS_CLAMP","MYDS_ATTN_GATHER_MIN_KEYS","MYDS_PREFILL_ATTN","MYDS_CHUNK_KEY_MAX"};
    for(unsigned i=0;i<sizeof(off)/sizeof(off[0]);i++)unsetenv(off[i]);NS_REQUIRE(!setenv("MYDS_METAL_CONCURRENT","0",1));NS_REQUIRE(!setenv("MYDS_LM_HEAD_Q8_NR0","2",1));ns_mode(1u);g_hist_enabled=1;
}
static void ns_host(void) {
    ns_clean();unsetenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE");NS_REQUIRE(indexer_topk_nan_shuffle_parse_mode()==0);
    NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE","0",1));NS_REQUIRE(indexer_topk_nan_shuffle_parse_mode()==0);NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE","1",1));NS_REQUIRE(indexer_topk_nan_shuffle_parse_mode()==1);
    const char *bad[]={"","2","-1","+1","01","0x1"," 1","1 ","1\n","true","false","00"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);i++){NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE",bad[i],1));NS_REQUIRE(indexer_topk_nan_shuffle_parse_mode()<0);NS_REQUIRE(!myds_metal_init(ns_shader));ns_clean();}
    NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE","1",1));NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD","1",1));NS_REQUIRE(!myds_metal_init(ns_shader));ns_clean();unsetenv("MYDS_INDEXER_TOPK_THRESHOLD");
    myds_v41_gpu_topk_nan_shuffle_counters(NULL,true);myds_v41_gpu_topk_nan_shuffle_counters_t c;myds_v41_gpu_topk_nan_shuffle_counters(&c,false);NS_REQUIRE(!c.requests&&!c.encoded&&!c.failures);unsetenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE");
    puts("{\"schema_version\":1,\"kind\":\"host_config\",\"valid_values\":[0,1],\"absent_default\":0,\"invalid_values_rejected_before_metal_resources\":12,\"threshold_mutual_exclusion_rejected\":true,\"counter_reset\":true,\"no_gpu_initialization\":true,\"pass\":true}");
}
static void ns_initialization(const char *phase,unsigned mode,bool missing) {
    NS_REQUIRE(myds_v41_gpu_topk_nan_shuffle_enabled()==(int)mode);NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE",mode?"0":"1",1));NS_REQUIRE(myds_v41_gpu_topk_nan_shuffle_enabled()==(int)mode);
    struct ns_shape shape={32768,1,0,0};struct ns_buf scores=ns_alloc(shape.cols*4u),reference=ns_alloc(2048u),out=ns_alloc(2048u);ns_fill(scores,shape,0u,47u);ns_policy(0u,1u);myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(NS_MAX_COLS);NS_REQUIRE(scratch);
    ns_run(reference,scores,shape,1u,scratch);ns_mode(2u);ns_reset();NS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;NS_REQUIRE(ns_call(out,scores,shape,2u,scratch));ns_encoder();ns_deps(scores,out);NS_REQUIRE(myds_metal_commit_batch());NS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);ns_counts(mode,1u,ns_parts(shape));ns_exact(reference,out);
    const bool scalar=[g_library.functionNames containsObject:[NSString stringWithUTF8String:ns_sort(false,true)]],causal=[g_library.functionNames containsObject:[NSString stringWithUTF8String:ns_sort(true,true)]];NS_REQUIRE(missing?!(scalar&&causal):(scalar&&causal));
    printf("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"%s\",\"initialized\":true,\"actual_mode\":%u,\"immutable_after_env_change\":true,\"reference_success\":true,\"missing_library\":%s,\"candidate_symbols\":[%s,%s],\"completed_cb\":true,\"cb_status\":4,\"full_ids_byte_exact\":true,\"canaries\":true,\"cols\":32768,\"pattern\":0,\"seed\":47",phase,mode,missing?"true":"false",scalar?"true":"false",causal?"true":"false");ns_count_json();bool first=true;printf(",\"files\":[");ns_raw(phase,"scores","f32",scores,&first);ns_raw(phase,"production-ids","i32",reference,&first);ns_raw(phase,"selected-ids","i32",out,&first);puts("]}");fflush(stdout);myds_v41_gpu_scratch_free(scratch);ns_free(scores);ns_free(reference);ns_free(out);
}
static unsigned ns_number(const char *s) {char *end=NULL;errno=0;unsigned long n=strtoul(s,&end,10);NS_REQUIRE(s[0]&&s[0]!='-'&&s[0]!='+'&&!errno&&end&&!*end&&n&&n<=1024u);return (unsigned)n;}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--host-only"))ns_host_only=1;
        else if(!strcmp(argv[i],"--expect-missing-candidate"))ns_missing=1;
        else if(i+1<argc&&!strcmp(argv[i],"--shader-dir"))ns_shader=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--output-dir"))ns_dir=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--samples"))ns_samples=ns_number(argv[++i]);
        else if(i+1<argc&&!strcmp(argv[i],"--repetitions"))ns_repetitions=ns_number(argv[++i]);
        else {fprintf(stderr,"usage: %s --shader-dir DIR --output-dir NEWDIR [--samples 5 --repetitions 32] [--host-only] [--expect-missing-candidate]\n",argv[0]);return 2;}
    }
    NS_REQUIRE(ns_samples>=3u&&ns_samples<=20u);ns_ambient();ns_host();if(ns_host_only)return 0;NS_REQUIRE(ns_dir&&!mkdir(ns_dir,0700));
    if(ns_missing){
        NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE","1",1));NS_REQUIRE(!myds_metal_init(ns_shader));ns_clean();puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"missing-explicit1\",\"initialized\":false,\"actual_mode\":0,\"cleanup_complete\":true,\"no_fallback\":true}");
        NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE","0",1));NS_REQUIRE(myds_metal_init(ns_shader));ns_initialization("missing-reference0",0u,true);myds_metal_cleanup();ns_clean();unsetenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE");NS_REQUIRE(myds_metal_init(ns_shader));ns_initialization("missing-default0-reinit",0u,true);myds_metal_cleanup();ns_clean();printf("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"missing-library\",\"initializations\":3,\"native_files\":%u,\"pass\":true}\n",ns_files_count);return 0;
    }
    NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE","0",1));NS_REQUIRE(myds_metal_init(ns_shader));ns_initialization("explicit0",0u,false);myds_metal_cleanup();ns_clean();
    NS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE","1",1));NS_REQUIRE(myds_metal_init(ns_shader));ns_initialization("explicit1",1u,false);
    printf("{\"schema_version\":1,\"kind\":\"identity\",\"device\":\"%s\",\"runtime_mode\":1,\"candidate_scalar_kernel\":\"%s\",\"candidate_causal_kernel\":\"%s\",\"reference_scalar_max_threads\":%u,\"reference_causal_max_threads\":%u,\"new_flag_default\":0,\"prefix\":1,\"nan_fuse\":1,\"barrier_coalescing\":false,\"synthetic_distribution\":true}\n",g_device.name.UTF8String,ns_sort(false,true),ns_sort(true,true),myds_metal_kernel_max_threads("kernel_argsort_f32_i32_desc"),myds_metal_kernel_max_threads("kernel_argsort_f32_i32_desc_causal"));
    const struct ns_shape shapes[]={{15360,1,0,0},{15361,1,0,0},{16384,1,0,0},{32768,1,0,0},{131077,1,0,0},{32768,3,32765,1},{32768,3,8190,1},{32768,7,0,2},{32768,3,16382,2},{131077,3,262149,2}};
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(NS_MAX_COLS);NS_REQUIRE(scratch);
    for(unsigned shape=0;shape<sizeof(shapes)/sizeof(shapes[0]);shape++)for(unsigned pattern=0;pattern<8;pattern++)for(unsigned policy=0;policy<4;policy++)ns_parity(ns_parities,shapes[shape],pattern,policy,scratch);
    const unsigned patterns[]={0u,1u,5u,7u};for(uint32_t nth=1u;nth<=1024u;nth*=2u)for(unsigned p=0;p<4;p++)for(unsigned c=0;c<2;c++)ns_stage(nth,patterns[p],c!=0u);
    for(unsigned p=0;p<4;p++)ns_reuse(p);ns_rejections(scratch);ns_completion(scratch);
    const struct ns_shape timed[]={{16384,1,0,0},{32768,1,0,0},{32768,3,32765,1}};
    for(unsigned i=0;i<3;i++){ns_timing(timed[i],0u,1u,scratch);if(ns_repetitions!=1u)ns_timing(timed[i],0u,ns_repetitions,scratch);}
    ns_timing(timed[1],1u,1u,scratch);if(ns_repetitions!=1u)ns_timing(timed[1],1u,ns_repetitions,scratch);
    myds_v41_gpu_scratch_free(scratch);myds_metal_cleanup();ns_clean();unsetenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE");NS_REQUIRE(myds_metal_init(ns_shader));ns_initialization("default0-reinit",0u,false);myds_metal_cleanup();ns_clean();
    printf("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"full\",\"parity_cases\":%u,\"partition_stage_cases\":%u,\"reuse_cases\":%u,\"rejections\":%u,\"completion_contracts\":1,\"timing_records\":%u,\"samples\":%u,\"repetitions\":%u,\"native_files\":%u,\"pass\":true}\n",ns_parities,ns_stages,ns_reuses,ns_negatives,ns_timings,ns_samples,ns_repetitions,ns_files_count);return 0;
}
