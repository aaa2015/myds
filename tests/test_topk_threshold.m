/* Full-byte acceptance for the actual inclusive F32 threshold selector.
 * Building and --host-only submit no GPU work. Other modes must run in an
 * exclusive GPU window. Scores are synthetic seeded inputs, not model traces. */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <sys/stat.h>

#define TK_MAX_COLS 262144u
#define TK_TOP 512u
static const char *tk_dir,*tk_shader="bin";
static unsigned tk_samples=5,tk_repetitions=32,tk_files_count,tk_parities,tk_timings;
static int tk_host_only,tk_missing;
struct tk_buf {myds_metal_buffer_t *owner,*view;size_t bytes;};
static struct tk_buf tk_failure_scores,tk_failure_ids[3];
static void tk_dump_failure(void);
static void tk_fail(const char *file,int line,const char *condition) {
    fprintf(stderr,"FAIL %s:%d %s\n",file,line,condition);tk_dump_failure();fflush(stdout);fflush(stderr);exit(1);
}
#define TK_REQUIRE(c) do {if(!(c))tk_fail(__FILE__,__LINE__,#c);} while(0)
static struct {char name[192];size_t bytes;uint64_t hash;} tk_files[8192];
static uint64_t tk_fnv(const void *data,size_t bytes) {
    uint64_t h=UINT64_C(14695981039346656037);const uint8_t *p=data;
    for(size_t i=0;i<bytes;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static struct tk_buf tk_alloc(size_t bytes) {
    struct tk_buf b={.bytes=bytes};b.owner=myds_metal_buffer_alloc(bytes+64u);TK_REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+64u);
    b.view=myds_metal_buffer_sub_view(b.owner,32u,bytes);TK_REQUIRE(b.view);return b;
}
static void tk_guard(struct tk_buf b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(unsigned i=0;i<32;i++)TK_REQUIRE(p[i]==0xa5&&p[32u+b.bytes+i]==0xa5);
}
static void tk_free(struct tk_buf b) {tk_guard(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void tk_poison(struct tk_buf b) {memset(myds_metal_buffer_data(b.view),0xa5,b.bytes);}
static void tk_untouched(struct tk_buf b) {
    const uint8_t *p=myds_metal_buffer_data(b.view);for(size_t i=0;i<b.bytes;i++)TK_REQUIRE(p[i]==0xa5);tk_guard(b);
}
static void tk_dump(const char *role,struct tk_buf b) {
    if(!tk_dir||!b.view)return;char path[4096];if(snprintf(path,sizeof(path),"%s/failure-%s.bin",tk_dir,role)>=(int)sizeof(path))return;
    FILE *f=fopen(path,"wb");if(!f)return;size_t n=fwrite(myds_metal_buffer_data(b.view),1,b.bytes,f);int rc=fclose(f);
    if(n==b.bytes&&!rc)fprintf(stderr,"FAILURE_RAW %s bytes=%zu unverified=true\n",path,b.bytes);
}
static void tk_dump_failure(void) {
    tk_dump("scores",tk_failure_scores);const char *names[]={"dense-ids","production-ids","candidate-ids"};
    for(unsigned i=0;i<3;i++)tk_dump(names[i],tk_failure_ids[i]);
}
static void tk_exact(struct tk_buf a,struct tk_buf b) {
    TK_REQUIRE(a.bytes==b.bytes);tk_guard(a);tk_guard(b);
    if(memcmp(myds_metal_buffer_data(a.view),myds_metal_buffer_data(b.view),a.bytes)){
        tk_dump("comparison-reference",a);tk_dump("comparison-candidate",b);
        const uint32_t *x=myds_metal_buffer_data(a.view),*y=myds_metal_buffer_data(b.view);
        for(size_t i=0;i<a.bytes/4u;i++)if(x[i]!=y[i]){fprintf(stderr,"PARITY index=%zu reference=%08x candidate=%08x\n",i,x[i],y[i]);break;}
        TK_REQUIRE(0);
    }
}
static void tk_file(const char *stem,const char *role,const char *dtype,const void *data,size_t bytes,bool *first) {
    char name[192],path[4096];TK_REQUIRE(snprintf(name,sizeof(name),"%s-%s.%s",stem,role,dtype)<(int)sizeof(name));
    uint64_t hash=tk_fnv(data,bytes);unsigned found=tk_files_count;
    for(unsigned i=0;i<tk_files_count;i++)if(!strcmp(tk_files[i].name,name)){found=i;break;}
    if(found==tk_files_count){
        TK_REQUIRE(tk_files_count<sizeof(tk_files)/sizeof(tk_files[0]));
        TK_REQUIRE(snprintf(path,sizeof(path),"%s/%s",tk_dir,name)<(int)sizeof(path));FILE *f=fopen(path,"wbx");TK_REQUIRE(f);
        TK_REQUIRE(fwrite(data,1,bytes,f)==bytes);TK_REQUIRE(!fclose(f));
        strcpy(tk_files[found].name,name);tk_files[found].bytes=bytes;tk_files[found].hash=hash;tk_files_count++;
    }else TK_REQUIRE(tk_files[found].bytes==bytes&&tk_files[found].hash==hash);
    printf("%s{\"path\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",*first?"":",",name,role,dtype,bytes,hash);*first=false;
}
static void tk_raw(const char *stem,const char *role,const char *dtype,struct tk_buf b,bool *first) {
    tk_guard(b);tk_file(stem,role,dtype,myds_metal_buffer_data(b.view),b.bytes,first);
    uint8_t raw[64];const uint8_t *p=myds_metal_buffer_data(b.owner);memcpy(raw,p,32);memcpy(raw+32,p+32+b.bytes,32);
    char r[96];TK_REQUIRE(snprintf(r,sizeof(r),"%s-guards",role)<(int)sizeof(r));tk_file(stem,r,"bin",raw,sizeof(raw),first);
}
static uint32_t tk_random(uint32_t i,uint32_t seed) {
    uint32_t z=i*747796405u+seed*2891336453u;z=((z>>((z>>28u)+4u))^z)*277803737u;return (z>>22u)^z;
}
static const char *tk_patterns[]={"unique_positive","masked_unique","zero_low","subnormal_low","boundary_tie","interior_tie","selected_tie","all_tie","insufficient_positive","all_zero","positive_inf","visible_nan","capacity","all_masked","negative_inf_low","unique_513"};
static int tk_expected(unsigned p) {return p==0u||p==1u||p==2u||p==3u||p==14u||p==15u;}
static unsigned tk_expected_reason(unsigned p) {return tk_expected(p)?0u:p==11u?1u:p==10u?2u:p==8u||p==9u||p==13u?3u:p==7u||p==12u?4u:6u;}
/* High 640 scores occupy one inclusive histogram bin; all other values are
 * far lower. This guarantees candidate capacity/adoption for timing inputs,
 * and explicitly does not claim a captured-model score distribution. */
static void tk_fill(struct tk_buf score,uint32_t n,unsigned pattern,uint32_t seed) {
    TK_REQUIRE(pattern<sizeof(tk_patterns)/sizeof(*tk_patterns));uint32_t *v=myds_metal_buffer_data(score.view);
    for(uint32_t i=0;i<n;i++){
        v[i]=0x3d800000u+tk_random(i,seed)%0x000fffffu;
        if(pattern==1u&&(i/8u)%2u)v[i]=0xff800000u;
        if(pattern==2u||pattern==9u)v[i]=(i%2u)?0u:0x80000000u;
        if(pattern==3u)v[i]=(i%2u?0u:0x80000000u)|(1u+tk_random(i,seed)%0x007fffffu);
        if(pattern==7u)v[i]=0x3f800000u;
        if(pattern==8u||pattern==13u||pattern==14u)v[i]=0xff800000u;
    }
    if(pattern==7u||pattern==9u||pattern==13u)return;
    unsigned high=pattern==8u?400u:pattern==15u?513u:pattern==12u?1536u:640u,rank=0;if(high>n)high=n;
    for(uint32_t j=n;j&&rank<high;){uint32_t i=--j;if(pattern==1u&&(i/8u)%2u)continue;
        unsigned r=rank;if(pattern==4u&&rank==512u)r=511u;if(pattern==5u&&rank==41u)r=40u;
        v[i]=pattern==6u?0x43000000u:0x43010000u-r*32u;rank++;
    }
    TK_REQUIRE(rank==high);
    if(pattern==10u)v[n/2u]=0x7f800000u;
    if(pattern==11u){v[0]=0x7f800001u;v[n/2u]=0xffc12345u;v[n-1u]=0x7fc12345u;}
}
static void tk_policy(unsigned c,unsigned a) {
    myds_metal_set_concurrent((int)c);myds_metal_set_auto_barrier((int)a);
    TK_REQUIRE(myds_metal_get_concurrent()==(int)c&&myds_metal_get_auto_barrier()==(int)a);
}
static void tk_encoder(void) {TK_REQUIRE(g_batch_enc&&g_batch_enc.dispatchType==(myds_metal_get_concurrent()?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial));}
static void tk_mode(unsigned mode) {
    TK_REQUIRE(mode<3u);TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE",mode?"1":"0",1));TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE",mode?"1":"0",1));
}
static void tk_prepare(struct tk_buf score,struct tk_buf ids) {
    if(!myds_metal_get_auto_barrier()){const myds_metal_buffer_t *deps[]={score.view,ids.view};myds_metal_barrier_buffers(deps,2);}
}
static void tk_join(struct tk_buf score,struct tk_buf ids) {
    if(!myds_metal_get_auto_barrier()){const myds_metal_buffer_t *deps[]={score.view,ids.view};myds_metal_barrier_buffers(deps,2);}
}
static int tk_call(struct tk_buf ids,struct tk_buf score,uint32_t n,unsigned mode,myds_v41_gpu_scratch_t *s) {
    tk_prepare(score,ids);return mode==2u?(n<8192u?myds_v41_gpu_topk_threshold_candidate(ids.view,score.view,n,s):myds_v41_gpu_topk_threshold(ids.view,score.view,n,TK_TOP,s)):myds_v41_gpu_topk(ids.view,score.view,n,TK_TOP,s);
}
static void tk_reset(void) {myds_v41_gpu_topk_threshold_counters(NULL,true);TK_REQUIRE(myds_v41_gpu_topk_threshold_stats(NULL,true));myds_metal_dispatch_count_reset();}
static myds_v41_gpu_topk_threshold_stats_t tk_stats(unsigned requested,unsigned adopted) {
    myds_v41_gpu_topk_threshold_counters_t h;myds_v41_gpu_topk_threshold_counters(&h,false);myds_v41_gpu_topk_threshold_stats_t g;
    TK_REQUIRE(myds_v41_gpu_topk_threshold_stats(&g,false));TK_REQUIRE(h.requests==requested&&h.encoded==requested&&!h.errors);
    TK_REQUIRE(g.validated==requested&&g.adopted==adopted&&g.fallback==requested-adopted);
    const uint32_t *words=myds_metal_buffer_data(g_indexer_topk_threshold_stats);
    const uint64_t decoded[]={g.validated,g.adopted,g.fallback,g.nan,g.inf,g.insufficient,g.capacity,g.certificate,g.tie};
    TK_REQUIRE(words);for(unsigned i=0;i<9;i++)TK_REQUIRE(words[i]==decoded[i]);return g;
}
static void tk_stats_json(myds_v41_gpu_topk_threshold_stats_t g) {
    myds_v41_gpu_topk_threshold_counters_t h;myds_v41_gpu_topk_threshold_counters(&h,false);
    printf(",\"host\":{\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"errors\":%" PRIu64 "},\"gpu\":{\"validated\":%" PRIu64 ",\"adopted\":%" PRIu64 ",\"fallback\":%" PRIu64 ",\"nan\":%" PRIu64 ",\"inf\":%" PRIu64 ",\"insufficient\":%" PRIu64 ",\"capacity\":%" PRIu64 ",\"certificate\":%" PRIu64 ",\"tie\":%" PRIu64 "}",h.requests,h.encoded,h.errors,g.validated,g.adopted,g.fallback,g.nan,g.inf,g.insufficient,g.capacity,g.certificate,g.tie);
}
static uint32_t tk_nth(uint32_t n) {
    uint32_t max=myds_metal_kernel_max_threads("kernel_argsort_f32_i32_desc"),nth=1;TK_REQUIRE(max);while(nth<n&&2u*nth<=max)nth*=2u;return nth;
}
static void tk_ids(struct tk_buf ids,uint32_t n) {
    const uint32_t *p=myds_metal_buffer_data(ids.view);for(unsigned i=0;i<TK_TOP;i++)TK_REQUIRE(p[i]<n);tk_guard(ids);
}
static void tk_diagnostics(const char *stem,myds_v41_gpu_scratch_t *s,myds_v41_gpu_topk_threshold_stats_t g,bool *first) {
    (void)g;TK_REQUIRE(s->threshold_control&&s->threshold_pairs);
    uint32_t control[12],pairs[2048];TK_REQUIRE(myds_v41_gpu_topk_threshold_debug(s,control,pairs));
    TK_REQUIRE(!memcmp(control,myds_metal_buffer_data(s->threshold_control),48u)&&!memcmp(pairs,myds_metal_buffer_data(s->threshold_pairs),8192u));tk_file(stem,"control","u32",myds_metal_buffer_data(s->threshold_control),48u,first);
    tk_file(stem,"pairs","u32",myds_metal_buffer_data(s->threshold_pairs),1024u*8u,first);
    tk_file(stem,"gpu-stats","u32",myds_metal_buffer_data(g_indexer_topk_threshold_stats),36u,first);
}
static void tk_run(struct tk_buf ids,struct tk_buf score,uint32_t n,unsigned mode,myds_v41_gpu_scratch_t *s) {
    tk_mode(mode);tk_reset();TK_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    TK_REQUIRE(tk_call(ids,score,n,mode,s));tk_encoder();tk_join(score,ids);TK_REQUIRE(myds_metal_commit_batch());TK_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);tk_ids(ids,n);
}
static bool tk_pairs_valid(const uint32_t c[12]) {return c[1]==0u||c[1]==6u||(c[1]==5u&&c[8]>=512u&&c[8]<=1024u&&c[10]==c[8]);}
static void tk_live_files(const char *,myds_v41_gpu_scratch_t *,uint32_t,bool *);
static void tk_parity(unsigned index,uint32_t n,unsigned pattern,unsigned policy,myds_v41_gpu_scratch_t *s) {
    struct tk_buf score=tk_alloc(n*4u),ids[3];for(unsigned i=0;i<3;i++)ids[i]=tk_alloc(TK_TOP*4u);
    tk_failure_scores=score;memcpy(tk_failure_ids,ids,sizeof(ids));tk_fill(score,n,pattern,101u+n);const uint64_t hash=tk_fnv(myds_metal_buffer_data(score.view),score.bytes);tk_policy(policy/2u,policy%2u);
    for(unsigned m=0;m<3;m++){tk_run(ids[m],score,n,m,s);tk_exact(ids[0],ids[m]);TK_REQUIRE(hash==tk_fnv(myds_metal_buffer_data(score.view),score.bytes));tk_guard(score);}
    myds_v41_gpu_topk_threshold_stats_t g=tk_stats(1u,tk_expected(pattern));
    char stem[64],input[80];snprintf(stem,sizeof(stem),"case%u",index);snprintf(input,sizeof(input),"inputs-n%u-p%u",n,pattern);
    printf("{\"schema_version\":1,\"kind\":\"parity\",\"case\":%u,\"cols\":%u,\"top\":512,\"pattern\":%u,\"pattern_name\":\"%s\",\"seed\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"reference_nth\":%u,\"expected_adopt\":%s,\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"dense_production_candidate_full_ids_equal\":true,\"scores_unchanged\":true,\"canaries\":true,\"actual_gpu_stats\":true,\"prefix\":1,\"nan_fuse\":1",index,n,pattern,tk_patterns[pattern],101u+n,policy/2u,policy%2u,tk_nth(n),tk_expected(pattern)?"true":"false");tk_stats_json(g);
    const uint32_t *ctl=myds_metal_buffer_data(s->threshold_control);TK_REQUIRE(ctl[0]==(uint32_t)tk_expected(pattern)&&ctl[1]==tk_expected_reason(pattern));
    const uint64_t reasons[]={g.nan,g.inf,g.insufficient,g.capacity,g.certificate,g.tie};
    for(unsigned k=0;k<6;k++)TK_REQUIRE(reasons[k]==(uint64_t)(tk_expected_reason(pattern)==k+1u));
    if(ctl[0]){const uint32_t *values=myds_metal_buffer_data(score.view),*selected=myds_metal_buffer_data(ids[2].view);
        for(unsigned i=0;i<512;i++){TK_REQUIRE(values[selected[i]]>=0x00800000u&&values[selected[i]]<0x7f800000u);if(i)TK_REQUIRE(values[selected[i-1u]]>values[selected[i]]);}}printf(",\"gpu_reason\":%u,\"pairs_valid\":%s",ctl[1],tk_pairs_valid(ctl)?"true":"false");
    bool first=true;printf(",\"files\":[");tk_raw(input,"scores","f32",score,&first);const char *roles[]={"dense-ids","production-ids","candidate-ids"};for(unsigned i=0;i<3;i++)tk_raw(stem,roles[i],"i32",ids[i],&first);tk_diagnostics(stem,s,g,&first);tk_live_files(stem,s,n,&first);puts("]}");fflush(stdout);tk_parities++;
    tk_failure_scores=(struct tk_buf){0};memset(tk_failure_ids,0,sizeof(tk_failure_ids));tk_free(score);for(unsigned i=0;i<3;i++)tk_free(ids[i]);
}
struct tk_diag {struct tk_buf control,pairs,parts,hist,offsets;};
static struct tk_diag tk_diag_alloc(uint32_t n) {
    const uint32_t parts=(n+1023u)/1024u;
    return (struct tk_diag){tk_alloc(48u),tk_alloc(8192u),tk_alloc(parts*16u),tk_alloc(parts*256u*4u),tk_alloc(parts*8u)};
}
static void tk_diag_free(struct tk_diag d) {tk_free(d.control);tk_free(d.pairs);tk_free(d.parts);tk_free(d.hist);tk_free(d.offsets);}
static void tk_diag_copy(struct tk_diag d,myds_v41_gpu_scratch_t *s) {
    const myds_metal_buffer_t *src[]={s->threshold_control,s->threshold_pairs,s->threshold_parts,s->threshold_hist,s->threshold_offsets};
    const struct tk_buf dst[]={d.control,d.pairs,d.parts,d.hist,d.offsets};
    if(!myds_metal_get_auto_barrier())myds_metal_barrier_buffers(src,5u);
    for(unsigned i=0;i<5;i++)TK_REQUIRE(myds_metal_buffer_copy(dst[i].view,0,src[i],0,dst[i].bytes));
    if(!myds_metal_get_auto_barrier()){const myds_metal_buffer_t *deps[]={d.control.view,d.pairs.view,d.parts.view,d.hist.view,d.offsets.view};myds_metal_barrier_buffers(deps,5u);}
}
static void tk_diag_files(const char *stem,struct tk_diag d,bool *first) {
    const struct tk_buf b[]={d.control,d.pairs,d.parts,d.hist,d.offsets};const char *roles[]={"control","pairs","parts","hist","offsets"};
    for(unsigned i=0;i<5;i++)tk_raw(stem,roles[i],"u32",b[i],first);
}
static void tk_live_files(const char *stem,myds_v41_gpu_scratch_t *s,uint32_t n,bool *first) {
    uint32_t parts=(n+1023u)/1024u;
    tk_file(stem,"parts","u32",myds_metal_buffer_data(s->threshold_parts),parts*16u,first);
    tk_file(stem,"hist","u32",myds_metal_buffer_data(s->threshold_hist),parts*256u*4u,first);
    tk_file(stem,"offsets","u32",myds_metal_buffer_data(s->threshold_offsets),parts*8u,first);
}
static void tk_reuse(unsigned policy,myds_v41_gpu_scratch_t *s) {
    const unsigned order[]={0,0,1,2,2,1,0},counts[]={1,3,3},patterns[]={0,4,1};
    struct tk_buf input[3],reference[3],shared=tk_alloc(32768u*4u),out=tk_alloc(2048u),snap_scores[7],snap_ids[7];struct tk_diag diag[7];
    for(unsigned i=0;i<3;i++){input[i]=tk_alloc(32768u*4u);reference[i]=tk_alloc(2048u);tk_fill(input[i],32768u,patterns[i],701u+i*43u);tk_policy(0,1);tk_run(reference[i],input[i],32768u,1u,s);}
    for(unsigned i=0;i<7;i++){snap_scores[i]=tk_alloc(32768u*4u);snap_ids[i]=tk_alloc(2048u);diag[i]=tk_diag_alloc(32768u);}
    tk_policy(policy/2u,policy%2u);tk_mode(2);tk_reset();unsigned item=0;
    for(unsigned pass=0;pass<3;pass++){
        TK_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        /* Stats API must refuse both reads and resets while encoding, without
         * waiting or modifying the buffer consumed by the pending graph. */
        TK_REQUIRE(!myds_v41_gpu_topk_threshold_stats(NULL,false));TK_REQUIRE(!myds_v41_gpu_topk_threshold_stats(NULL,true));
        uint32_t debug_control[12],debug_pairs[2048];memset(debug_control,0xa5,sizeof(debug_control));memset(debug_pairs,0xa5,sizeof(debug_pairs));
        TK_REQUIRE(!myds_v41_gpu_topk_threshold_debug(s,debug_control,debug_pairs));
        for(unsigned k=0;k<12;k++)TK_REQUIRE(debug_control[k]==0xa5a5a5a5u);for(unsigned k=0;k<2048;k++)TK_REQUIRE(debug_pairs[k]==0xa5a5a5a5u);
        for(unsigned j=0;j<counts[pass];j++,item++){
            if(!myds_metal_get_auto_barrier())myds_metal_barrier_buffer(shared.view);
            TK_REQUIRE(myds_metal_buffer_copy(shared.view,0,input[order[item]].view,0,shared.bytes));
            TK_REQUIRE(tk_call(out,shared,32768u,2u,s));tk_encoder();tk_join(shared,out);
            TK_REQUIRE(myds_metal_buffer_copy(snap_scores[item].view,0,shared.view,0,shared.bytes));
            TK_REQUIRE(myds_metal_buffer_copy(snap_ids[item].view,0,out.view,0,out.bytes));tk_diag_copy(diag[item],s);
            tk_join(snap_scores[item],snap_ids[item]);
        }
        tk_encoder();TK_REQUIRE(myds_metal_commit_batch());TK_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
    }
    TK_REQUIRE(item==7u);myds_v41_gpu_topk_threshold_stats_t g=tk_stats(7u,5u);
    printf("{\"schema_version\":1,\"kind\":\"reuse\",\"case\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"calls_per_batch\":[1,3,3],\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"shared_input_output\":true,\"caller_input_output_join\":true,\"snapshot_copies_add_dependencies\":true,\"active_stats_read_reset_rejected\":true,\"full_ids_equal\":true,\"scores_unchanged\":true,\"canaries\":true",policy,policy/2u,policy%2u);tk_stats_json(g);
    printf(",\"items\":[");item=0;
    for(unsigned pass=0;pass<3;pass++)for(unsigned j=0;j<counts[pass];j++,item++){
        const unsigned ii=order[item];tk_exact(input[ii],snap_scores[item]);tk_exact(reference[ii],snap_ids[item]);tk_ids(snap_ids[item],32768u);
        const uint32_t *c=myds_metal_buffer_data(diag[item].control.view);TK_REQUIRE(c[0]==(uint32_t)tk_expected(patterns[ii]));
        char stem[96],in[96];snprintf(stem,sizeof(stem),"reuse%u-pass%u-item%u",policy,pass,j);snprintf(in,sizeof(in),"reuse-input%u",ii);
        printf("%s{\"pass\":%u,\"item\":%u,\"input_index\":%u,\"cols\":32768,\"pattern\":%u,\"seed\":%u,\"gpu_adopt\":%u,\"gpu_reason\":%u,\"pairs_valid\":%s,\"files\":[",item?",":"",pass,j,ii,patterns[ii],701u+ii*43u,c[0],c[1],tk_pairs_valid(c)?"true":"false");
        bool first=true;tk_raw(in,"scores","f32",input[ii],&first);tk_raw(in,"reference-ids","i32",reference[ii],&first);
        tk_raw(stem,"observed-scores","f32",snap_scores[item],&first);tk_raw(stem,"candidate-ids","i32",snap_ids[item],&first);tk_diag_files(stem,diag[item],&first);printf("]}");
    }
    bool first=true;printf("],\"files\":[");char stem[64];snprintf(stem,sizeof(stem),"reuse%u",policy);tk_file(stem,"gpu-stats","u32",myds_metal_buffer_data(g_indexer_topk_threshold_stats),36u,&first);puts("]}");fflush(stdout);
    for(unsigned i=0;i<3;i++){tk_free(input[i]);tk_free(reference[i]);}for(unsigned i=0;i<7;i++){tk_free(snap_scores[i]);tk_free(snap_ids[i]);tk_diag_free(diag[i]);}tk_free(shared);tk_free(out);
}
static void tk_timing(uint32_t n,unsigned pattern,unsigned reps,myds_v41_gpu_scratch_t *s) {
    const char *arms[]={"A1","B","A2"};struct tk_buf score=tk_alloc(n*4u),reference=tk_alloc(2048u),out=tk_alloc(2048u);
    tk_failure_scores=score;tk_failure_ids[0]=reference;tk_failure_ids[2]=out;tk_policy(1,0);g_hist_enabled=0;
    for(unsigned arm=0;arm<3;arm++)for(unsigned sample=0;sample<=tk_samples;sample++){
        tk_fill(score,n,pattern,9001u+sample);const uint64_t hash=tk_fnv(myds_metal_buffer_data(score.view),score.bytes);
        /* All arms perform this separate production oracle before the measured
         * CB. This deliberately warms inputs; cold DRAM is not proven. */
        tk_run(reference,score,n,1u,s);tk_poison(out);tk_mode(arm==1u?2u:1u);tk_reset();
        TK_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned rep=0;rep<reps;rep++){TK_REQUIRE(tk_call(out,score,n,arm==1u?2u:1u,s));tk_encoder();tk_join(score,out);}
        const uint64_t dispatches=myds_metal_dispatch_count();tk_encoder();TK_REQUIRE(myds_metal_commit_batch());TK_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
        TK_REQUIRE(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime);
        tk_exact(reference,out);TK_REQUIRE(hash==tk_fnv(myds_metal_buffer_data(score.view),score.bytes));tk_guard(score);tk_ids(out,n);
        myds_v41_gpu_topk_threshold_stats_t g=tk_stats(arm==1u?reps:0u,arm==1u?reps:0u);
        char stem[128],input[96];snprintf(stem,sizeof(stem),"timing-n%u-p%u-r%u-%s-s%u",n,pattern,reps,arms[arm],sample);snprintf(input,sizeof(input),"timing-inputs-n%u-p%u-s%u",n,pattern,sample);
        printf("{\"schema_version\":1,\"kind\":\"gpu_timing\",\"cols\":%u,\"pattern\":%u,\"pattern_name\":\"%s\",\"scope\":\"complete_topk_graph\",\"required_production_shape\":%s,\"arm\":\"%s\",\"sample\":%u,\"warmup\":%s,\"input_seed\":%u,\"repetitions\":%u,\"concurrent\":1,\"auto_barrier\":0,\"prefix\":1,\"nan_fuse\":1,\"encoder_dispatch_type_verified\":true,\"caller_input_output_join\":true,\"guarded_reference_graph_included\":%s,\"gpu_start_s\":%.12f,\"gpu_end_s\":%.12f,\"gpu_us_per_call\":%.9f,\"dispatches\":%" PRIu64 ",\"cb_status\":4,\"completed_cb\":true,\"dispatch_hist_disabled\":true,\"full_ids_equal\":true,\"scores_unchanged\":true,\"canaries\":true,\"actual_gpu_stats\":true,\"synthetic_distribution\":true,\"oracle_prepared_before_each_measured_cb\":true,\"cold_dram_not_proven\":true,\"fixture_not_fullmodel\":true",n,pattern,tk_patterns[pattern],pattern==0u||n==32768u?"true":"false",arms[arm],sample,sample?"false":"true",9001u+sample,reps,arm==1u?"true":"false",cb.GPUStartTime,cb.GPUEndTime,(cb.GPUEndTime-cb.GPUStartTime)*1e6/reps,dispatches);tk_stats_json(g);
        bool first=true;printf(",\"files\":[");tk_raw(input,"scores","f32",score,&first);tk_raw(input,"reference-ids","i32",reference,&first);tk_raw(stem,"output-ids","i32",out,&first);if(arm==1u){tk_diagnostics(stem,s,g,&first);tk_live_files(stem,s,n,&first);}else tk_file(stem,"gpu-stats","u32",myds_metal_buffer_data(g_indexer_topk_threshold_stats),36u,&first);puts("]}");fflush(stdout);tk_timings++;
    }
    tk_failure_scores=(struct tk_buf){0};memset(tk_failure_ids,0,sizeof(tk_failure_ids));tk_free(score);tk_free(reference);tk_free(out);g_hist_enabled=1;
}
@interface TKRestrictedPipeline : NSProxy {id<MTLComputePipelineState> actual;unsigned limitation;}
+(id)withPipeline:(id<MTLComputePipelineState>)p limitation:(unsigned)n;
@end
@implementation TKRestrictedPipeline
+(id)withPipeline:(id<MTLComputePipelineState>)p limitation:(unsigned)n {TKRestrictedPipeline *v=[TKRestrictedPipeline alloc];v->actual=p;v->limitation=n;return v;}
-(NSUInteger)maxTotalThreadsPerThreadgroup {return limitation==0u?64u:actual.maxTotalThreadsPerThreadgroup;}
-(NSUInteger)threadExecutionWidth {return limitation==1u?16u:actual.threadExecutionWidth;}
-(NSUInteger)staticThreadgroupMemoryLength {return limitation==2u?g_device.maxThreadgroupMemoryLength:actual.staticThreadgroupMemoryLength;}
-(NSMethodSignature *)methodSignatureForSelector:(SEL)s {return [(NSObject *)actual methodSignatureForSelector:s];}
-(void)forwardInvocation:(NSInvocation *)i {[i invokeWithTarget:actual];}
@end
static void tk_rejected(unsigned index,const char *reason,struct tk_buf primary,myds_metal_buffer_t *ids,
    struct tk_buf score,const myds_metal_buffer_t *input,uint32_t n,myds_v41_gpu_scratch_t *s) {
    const uint64_t input_hash=tk_fnv(myds_metal_buffer_data(score.view),score.bytes);const size_t actual_bytes=ids?myds_metal_buffer_size(ids):0u;
    uint8_t *saved=actual_bytes?malloc(actual_bytes):NULL;TK_REQUIRE(!actual_bytes||saved);if(saved)memcpy(saved,myds_metal_buffer_data(ids),actual_bytes);
    tk_reset();TK_REQUIRE(myds_metal_begin_batch());TK_REQUIRE(!myds_v41_gpu_topk_threshold_candidate(ids,input,n,s));
    TK_REQUIRE(myds_metal_dispatch_count()==0u);TK_REQUIRE(myds_metal_commit_batch());tk_untouched(primary);tk_guard(score);
    TK_REQUIRE(input_hash==tk_fnv(myds_metal_buffer_data(score.view),score.bytes));if(saved)TK_REQUIRE(!memcmp(saved,myds_metal_buffer_data(ids),actual_bytes));
    myds_v41_gpu_topk_threshold_counters_t h;myds_v41_gpu_topk_threshold_counters(&h,false);TK_REQUIRE(h.requests==1u&&!h.encoded&&h.errors==1u);
    myds_v41_gpu_topk_threshold_stats_t g;TK_REQUIRE(myds_v41_gpu_topk_threshold_stats(&g,false));TK_REQUIRE(!g.validated&&!g.adopted&&!g.fallback);
    char stem[64];snprintf(stem,sizeof(stem),"reject%u",index);
    printf("{\"schema_version\":1,\"kind\":\"rejected_call\",\"case\":%u,\"reason\":\"%s\",\"dispatches\":0,\"no_fallback\":true,\"output_untouched\":true,\"actual_writer_bytes\":%zu,\"actual_writer_untouched\":true,\"scores_unchanged\":true,\"canaries\":true",index,reason,actual_bytes);tk_stats_json(g);
    bool first=true;printf(",\"files\":[");tk_raw(stem,"output","bin",primary,&first);tk_file(stem,"gpu-stats","u32",myds_metal_buffer_data(g_indexer_topk_threshold_stats),36u,&first);
    if(saved){tk_file(stem,"writer-before","bin",saved,actual_bytes,&first);tk_file(stem,"writer-after","bin",myds_metal_buffer_data(ids),actual_bytes,&first);}puts("]}");fflush(stdout);free(saved);
}
static unsigned tk_rejections(myds_v41_gpu_scratch_t *s) {
    struct tk_buf score=tk_alloc(8192u*4u),ids=tk_alloc(2048u);tk_fill(score,8192u,0u,91u);tk_policy(1,0);unsigned n=0;
#define TK_REJECT(reason,o,x,cols,scratch) tk_rejected(n++,reason,ids,o,score,x,cols,scratch)
    TK_REJECT("null_output",NULL,score.view,8192u,s);TK_REJECT("null_scores",ids.view,NULL,8192u,s);TK_REJECT("null_scratch",ids.view,score.view,8192u,NULL);
    TK_REJECT("zero_cols",ids.view,score.view,0u,s);TK_REJECT("below_diagnostic_min",ids.view,score.view,511u,s);TK_REJECT("above_max",ids.view,score.view,262145u,s);
    myds_metal_buffer_t *v=myds_metal_buffer_sub_view(ids.view,0,2044u);TK_REQUIRE(v);TK_REJECT("short_output",v,score.view,8192u,s);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(score.view,0,score.bytes-4u);TK_REQUIRE(v);TK_REJECT("short_scores",ids.view,v,8192u,s);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(ids.owner,33u,ids.bytes);TK_REQUIRE(v);TK_REJECT("unaligned_output",v,score.view,8192u,s);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(score.owner,33u,score.bytes);TK_REQUIRE(v);TK_REJECT("unaligned_scores",ids.view,v,8192u,s);myds_metal_buffer_free(v);
    v=myds_metal_buffer_sub_view(score.view,32u,2048u);TK_REQUIRE(v);TK_REJECT("output_scores_overlap",v,score.view,8192u,s);myds_metal_buffer_free(v);
    myds_v41_gpu_inject_topk_threshold_unavailable(true);TK_REJECT("injected_preflight_unavailable",ids.view,score.view,8192u,s);myds_v41_gpu_inject_topk_threshold_unavailable(false);
    NSString *key=@"kernel_myds_topk_threshold_final";id<MTLComputePipelineState> saved=g_native_pipelines[key];TK_REQUIRE(saved);
    const char *names[]={"restricted_max_threads","restricted_simd_width","restricted_tg_memory"};
    for(unsigned k=0;k<3;k++){g_native_pipelines[key]=(id<MTLComputePipelineState>)[TKRestrictedPipeline withPipeline:saved limitation:k];TK_REJECT(names[k],ids.view,score.view,8192u,s);g_native_pipelines[key]=saved;}
#undef TK_REJECT
    tk_free(score);tk_free(ids);return n;
}

static void tk_completion(myds_v41_gpu_scratch_t *s) {
    struct tk_buf score=tk_alloc(8192u*4u),reference=tk_alloc(2048u),out=tk_alloc(2048u);tk_fill(score,8192u,0u,417u);tk_policy(1,0);tk_run(reference,score,8192u,1u,s);tk_mode(2);tk_reset();
    TK_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;TK_REQUIRE(tk_call(out,score,8192u,2u,s));tk_encoder();tk_join(score,out);
    myds_metal_inject_commit_failure(1u);TK_REQUIRE(!myds_metal_commit_batch());myds_metal_inject_commit_failure(0u);
    TK_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);tk_exact(reference,out);myds_v41_gpu_topk_threshold_stats_t g=tk_stats(1u,1u);
    printf("{\"schema_version\":1,\"kind\":\"completion_contract\",\"cols\":8192,\"pattern\":0,\"seed\":417,\"encoded_is_not_completion_success\":true,\"injected_host_commit_failure\":true,\"real_gpu_failure\":false,\"outer_commit_return\":0,\"completed_cb\":true,\"cb_status\":4,\"full_ids_equal\":true,\"scores_unchanged\":true,\"canaries\":true");tk_stats_json(g);
    bool first=true;printf(",\"files\":[");tk_raw("completion","scores","f32",score,&first);tk_raw("completion","reference-ids","i32",reference,&first);tk_raw("completion","candidate-ids","i32",out,&first);tk_diagnostics("completion",s,g,&first);tk_live_files("completion",s,8192u,&first);puts("]}");fflush(stdout);tk_free(score);tk_free(reference);tk_free(out);
}
static void tk_clean(void) {TK_REQUIRE(!g_initialized&&!g_device&&!g_library&&!g_queue&&!g_indexer_topk_threshold&&!g_indexer_topk_threshold_stats&&!myds_v41_gpu_topk_threshold_enabled());}
static void tk_ambient(void) {
    const char *off[]={"MYDS_BATCH2_Q8_SHARED","MYDS_MOE_Q2_SUM_CACHE","MYDS_MOE_Q2_SUM_CACHE_FUSED","MYDS_MOE_IQ2_CONSTANT_LUT","MYDS_INDEXER_HEAD_SLOTS","MYDS_INDEXER_COMPACT_SCORER","MYDS_RESIDENCY_BYPASS","MYDS_V41_WEIGHT_STAGE"};
    for(unsigned i=0;i<sizeof(off)/sizeof(off[0]);i++)unsetenv(off[i]);TK_REQUIRE(!setenv("MYDS_METAL_CONCURRENT","0",1));tk_mode(1u);g_hist_enabled=1;
}
static void tk_host(void) {
    tk_clean();unsetenv("MYDS_INDEXER_TOPK_THRESHOLD");TK_REQUIRE(indexer_topk_threshold_parse_mode()==0);
    TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD","0",1));TK_REQUIRE(indexer_topk_threshold_parse_mode()==0);
    TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD","1",1));TK_REQUIRE(indexer_topk_threshold_parse_mode()==1);
    const char *bad[]={"","2","-1","+1","01","0x1"," 1","1 ","1\n","true","false","00"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);i++){TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD",bad[i],1));TK_REQUIRE(indexer_topk_threshold_parse_mode()<0);TK_REQUIRE(!myds_metal_init(tk_shader));tk_clean();}
    myds_v41_gpu_topk_threshold_counters(NULL,true);myds_v41_gpu_topk_threshold_counters_t h;myds_v41_gpu_topk_threshold_counters(&h,false);TK_REQUIRE(!h.requests&&!h.encoded&&!h.errors);
    myds_v41_gpu_topk_threshold_stats_t g;TK_REQUIRE(myds_v41_gpu_topk_threshold_stats(&g,true));TK_REQUIRE(!memcmp(&g,&(myds_v41_gpu_topk_threshold_stats_t){0},sizeof(g)));
    unsetenv("MYDS_INDEXER_TOPK_THRESHOLD");puts("{\"schema_version\":1,\"kind\":\"host_config\",\"valid_values\":[0,1],\"absent_default\":0,\"invalid_values_rejected_before_metal_resources\":12,\"counter_reset\":true,\"no_gpu_initialization\":true,\"pass\":true}");
}
static void tk_init_record(const char *phase,unsigned mode,bool missing) {
    TK_REQUIRE(myds_v41_gpu_topk_threshold_enabled()==(int)mode);TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD",mode?"0":"1",1));TK_REQUIRE(myds_v41_gpu_topk_threshold_enabled()==(int)mode);
    struct tk_buf score=tk_alloc(8192u*4u),out=tk_alloc(2048u);tk_fill(score,8192u,0u,47u);tk_policy(0,1);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(TK_MAX_COLS);TK_REQUIRE(s);tk_run(out,score,8192u,1u,s);
    const bool symbol=[g_library.functionNames containsObject:@"kernel_myds_topk_threshold_final"];TK_REQUIRE(symbol!=missing);
    printf("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"%s\",\"initialized\":true,\"actual_mode\":%u,\"immutable_after_env_change\":true,\"reference_success\":true,\"candidate_symbol_missing\":%s,\"completed_cb\":true,\"cb_status\":4,\"scores_unchanged\":true,\"canaries\":true,\"cols\":8192,\"pattern\":0,\"seed\":47,\"files\":[",phase,mode,missing?"true":"false");
    bool first=true;tk_raw("initialization-inputs","scores","f32",score,&first);tk_raw(phase,"reference-ids","i32",out,&first);puts("]}");fflush(stdout);myds_v41_gpu_scratch_free(s);tk_free(score);tk_free(out);
}
static void tk_missing_library(void) {
    TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD","1",1));TK_REQUIRE(!myds_metal_init(tk_shader));tk_clean();
    puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"missing-explicit1\",\"initialized\":false,\"actual_mode\":0,\"cleanup_complete\":true,\"no_fallback\":true}");
    TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD","0",1));TK_REQUIRE(myds_metal_init(tk_shader));tk_init_record("missing-reference0",0u,true);myds_metal_cleanup();tk_clean();
    unsetenv("MYDS_INDEXER_TOPK_THRESHOLD");TK_REQUIRE(myds_metal_init(tk_shader));tk_init_record("missing-default0-reinit",0u,true);myds_metal_cleanup();tk_clean();
    printf("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"missing-library\",\"initializations\":3,\"native_files\":%u,\"pass\":true}\n",tk_files_count);
}
static unsigned tk_number(const char *value) {
    char *end=NULL;errno=0;unsigned long n=strtoul(value,&end,10);TK_REQUIRE(value[0]&&value[0]!='-'&&value[0]!='+'&&!errno&&end&&!*end&&n&&n<=1024u);return (unsigned)n;
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--host-only"))tk_host_only=1;
        else if(!strcmp(argv[i],"--expect-missing-candidate"))tk_missing=1;
        else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)tk_shader=argv[++i];
        else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)tk_dir=argv[++i];
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)tk_samples=tk_number(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)tk_repetitions=tk_number(argv[++i]);
        else {fprintf(stderr,"usage: %s --shader-dir DIR --output-dir NEW_DIR [--samples 5 --repetitions 32] [--host-only] [--expect-missing-candidate]\n",argv[0]);return 2;}
    }
    TK_REQUIRE(tk_samples==5u&&tk_repetitions==32u);tk_ambient();tk_host();if(tk_host_only)return 0;
    TK_REQUIRE(tk_dir);struct stat st;errno=0;TK_REQUIRE(lstat(tk_dir,&st)!=0&&errno==ENOENT);TK_REQUIRE(!mkdir(tk_dir,0700));if(tk_missing){tk_missing_library();return 0;}
    TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD","0",1));TK_REQUIRE(myds_metal_init(tk_shader));tk_init_record("reference0",0u,false);myds_metal_cleanup();tk_clean();
    TK_REQUIRE(!setenv("MYDS_INDEXER_TOPK_THRESHOLD","1",1));TK_REQUIRE(myds_metal_init(tk_shader));tk_init_record("candidate1",1u,false);
    printf("{\"schema_version\":1,\"kind\":\"identity\",\"device\":\"%s\",\"runtime_mode\":1,\"top\":512,\"capacity\":1024,\"bins\":256,\"partition_cols\":1024,\"view_offset\":32,\"prefix\":1,\"nan_fuse\":1,\"actual_gpu_stats\":true,\"ambient_env_cleared\":true,\"synthetic_distribution\":true,\"fixture_not_fullmodel\":true}\n",g_device.name.UTF8String);
    myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(TK_MAX_COLS);TK_REQUIRE(s);unsigned index=0;
    const uint32_t small[]={512u,513u,1023u,1025u,4095u,8191u};
    for(unsigned k=0;k<sizeof(small)/sizeof(small[0]);k++)for(unsigned policy=0;policy<4;policy++)tk_parity(index++,small[k],0u,policy,s);
    const uint32_t small_fallback[]={513u,1023u};
    for(unsigned k=0;k<2;k++)for(unsigned p=4;p<=5;p++)for(unsigned policy=0;policy<4;policy++)tk_parity(index++,small_fallback[k],p,policy,s);
    const uint32_t full[]={8192u,8193u};
    for(unsigned k=0;k<2;k++)for(unsigned p=0;p<16;p++)for(unsigned policy=0;policy<4;policy++)tk_parity(index++,full[k],p,policy,s);
    const uint32_t large[]={16407u,32768u,131077u,262144u};const unsigned patterns[]={0u,1u,4u,11u,12u};
    for(unsigned k=0;k<4;k++)for(unsigned p=0;p<5;p++)for(unsigned policy=0;policy<4;policy++)tk_parity(index++,large[k],patterns[p],policy,s);
    TK_REQUIRE(index==248u&&tk_parities==248u);for(unsigned p=0;p<4;p++)tk_reuse(p,s);const unsigned rejections=tk_rejections(s);TK_REQUIRE(rejections==15u);tk_completion(s);
    const uint32_t timings[]={16384u,32768u};
    for(unsigned k=0;k<2;k++)for(unsigned p=0;p<2;p++){tk_timing(timings[k],p,1u,s);tk_timing(timings[k],p,32u,s);}TK_REQUIRE(tk_timings==144u);
    myds_v41_gpu_scratch_free(s);myds_metal_cleanup();tk_clean();unsetenv("MYDS_INDEXER_TOPK_THRESHOLD");TK_REQUIRE(myds_metal_init(tk_shader));tk_init_record("default0-reinit",0u,false);myds_metal_cleanup();tk_clean();
    printf("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"main\",\"parity_cases\":248,\"reuse_cases\":4,\"reuse_items\":28,\"rejected_calls\":15,\"completion_contracts\":1,\"timing_records\":144,\"initializations\":3,\"native_files\":%u,\"full_ids_equal\":true,\"scores_unchanged\":true,\"pass\":true}\n",tk_files_count);return 0;
}
