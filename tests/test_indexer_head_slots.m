/* Model-free component acceptance for the actual indexer selector.
 * The unchanged generic scorer is the oracle. No arithmetic mirror is used.
 * --host-only does not initialize Metal; all other execution belongs in an
 * exclusive GPU window. Building never submits GPU work. */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <sys/stat.h>

#define HS_Q_FLOATS (32u*128u)
#define HS_MAX_ROWS 131077u
#define HS_REFERENCE "kernel_glm_indexer_score_one_direct"
#define HS_CANDIDATE "kernel_myds_indexer_score_one_head_slots"
static const char *hs_dir,*hs_shader="bin";
static unsigned hs_samples=5,hs_repetitions=32,hs_native_files,hs_parities,hs_timings;
static int hs_host_only,hs_expect_missing;
static void hs_failure_dump(void);
static void hs_fail(const char *file,int line,const char *condition) {
    fprintf(stderr,"FAIL %s:%d %s\n",file,line,condition);hs_failure_dump();fflush(stdout);fflush(stderr);exit(1);
}
#define HS_REQUIRE(c) do {if(!(c))hs_fail(__FILE__,__LINE__,#c);} while(0)
struct hs_guard {myds_metal_buffer_t *owner,*view;size_t bytes;};
struct hs_fixture {
    uint32_t rows,seed;bool bf16;unsigned mask_mode,pattern;
    struct hs_guard q,weights,cache,mask,scores[2],ids[2];
    uint64_t input_hash[4];
};
static struct hs_fixture *hs_active_fixture;
static struct {char name[192];size_t bytes;uint64_t hash;} hs_files[16384];
static uint64_t hs_fnv(const void *data,size_t bytes) {
    uint64_t h=UINT64_C(14695981039346656037);const uint8_t *p=data;
    for(size_t i=0;i<bytes;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static void hs_dump_one(const char *role,const void *data,size_t bytes) {
    if(!hs_dir||!data||!bytes)return;
    char path[4096];if(snprintf(path,sizeof(path),"%s/failure-%s.bin",hs_dir,role)>=(int)sizeof(path))return;
    FILE *file=fopen(path,"wb");if(!file)return;
    const size_t written=fwrite(data,1,bytes,file);const int closed=fclose(file);
    if(written==bytes&&!closed)fprintf(stderr,"FAILURE_RAW %s bytes=%zu fnv1a64=%016" PRIx64 " unverified=true\n",path,bytes,hs_fnv(data,bytes));
}
static void hs_failure_dump(void) {
    if(!hs_active_fixture)return;
    struct hs_fixture *f=hs_active_fixture;hs_active_fixture=NULL;
    struct hs_guard b[]={f->q,f->weights,f->cache,f->mask,f->scores[0],f->scores[1],f->ids[0],f->ids[1]};
    const char *roles[]={"q","weights","cache","mask","reference-scores","candidate-scores","reference-ids","candidate-ids"};
    for(unsigned i=0;i<8;i++)if(b[i].view)hs_dump_one(roles[i],myds_metal_buffer_data(b[i].view),b[i].bytes);
}
static struct hs_guard hs_allocate(size_t bytes) {
    struct hs_guard b={.bytes=bytes};b.owner=myds_metal_buffer_alloc(bytes+64u);HS_REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+64u);
    b.view=myds_metal_buffer_sub_view(b.owner,32u,bytes);HS_REQUIRE(b.view);return b;
}
static void hs_guard_check(struct hs_guard b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(unsigned i=0;i<32u;i++)HS_REQUIRE(p[i]==0xa5&&p[32u+b.bytes+i]==0xa5);
}
static void hs_destroy_guard(struct hs_guard b) {
    hs_guard_check(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);
}
static void hs_poison(struct hs_guard b){memset(myds_metal_buffer_data(b.view),0xa5,b.bytes);}
static void hs_untouched(struct hs_guard b) {
    const uint8_t *p=myds_metal_buffer_data(b.view);for(size_t i=0;i<b.bytes;i++)HS_REQUIRE(p[i]==0xa5);hs_guard_check(b);
}
static uint32_t hs_random(uint32_t i,uint32_t seed) {
    uint32_t z=i*747796405u+seed*2891336453u;z=((z>>((z>>28u)+4u))^z)*277803737u;return (z>>22u)^z;
}
static void hs_fill(struct hs_fixture *f,uint32_t seed) {
    f->seed=seed;float *q=myds_metal_buffer_data(f->q.view),*w=myds_metal_buffer_data(f->weights.view);
    for(unsigned i=0;i<HS_Q_FLOATS;i++)q[i]=((int)(hs_random(i,seed)%1023u)-511)/2048.f;
    for(unsigned h=0;h<32u;h++){q[h*128u]=.125f+(h%5u)/64.f;w[h]=.125f+(hs_random(h,seed+7u)%256u)/256.f;}
    for(uint32_t row=0;row<f->rows;row++)for(unsigned d=0;d<128u;d++){
        float v=((int)(hs_random(row*128u+d,seed+13u)%1023u)-511)/512.f;
        uint32_t bits;memcpy(&bits,&v,4);
        if(f->pattern==1u)bits=(d%2u)?0u:0x80000000u;
        if(f->pattern==2u&&d==0u&&row%7u==0u)bits=0x7f800000u;
        if(f->pattern==2u&&d==0u&&row%7u==3u)bits=0xff800000u;
        if(f->pattern==3u&&d==5u&&row%7u==0u)bits=row%2u?0xffc10000u:0x7f810000u;
        if(f->bf16)((uint16_t *)myds_metal_buffer_data(f->cache.view))[(size_t)row*128u+d]=(uint16_t)(bits>>16u);
        else memcpy((float *)myds_metal_buffer_data(f->cache.view)+(size_t)row*128u+d,&bits,4);
    }
    float *mask=myds_metal_buffer_data(f->mask.view);
    for(uint32_t i=0;i<(f->rows+7u)/8u;i++)mask[i]=f->mask_mode==1u?1.f:f->mask_mode==2u?(float)(i%2u):0.f;
    struct hs_guard in[]={f->q,f->weights,f->cache,f->mask};
    for(unsigned i=0;i<4u;i++){hs_guard_check(in[i]);f->input_hash[i]=hs_fnv(myds_metal_buffer_data(in[i].view),in[i].bytes);}
}
static struct hs_fixture hs_create(uint32_t rows,bool bf16,unsigned mask,unsigned pattern,uint32_t seed) {
    struct hs_fixture f={.rows=rows,.bf16=bf16,.mask_mode=mask,.pattern=pattern};
    f.q=hs_allocate(HS_Q_FLOATS*4u);f.weights=hs_allocate(32u*4u);
    f.cache=hs_allocate((size_t)rows*128u*(bf16?2u:4u));f.mask=hs_allocate((size_t)((rows+7u)/8u)*4u);
    for(unsigned arm=0;arm<2;arm++){f.scores[arm]=hs_allocate((size_t)rows*4u);f.ids[arm]=hs_allocate((size_t)(rows<512u?rows:512u)*4u);}
    hs_fill(&f,seed);return f;
}
static void hs_destroy(struct hs_fixture *f) {
    hs_destroy_guard(f->q);hs_destroy_guard(f->weights);hs_destroy_guard(f->cache);hs_destroy_guard(f->mask);
    for(unsigned arm=0;arm<2;arm++){hs_destroy_guard(f->scores[arm]);hs_destroy_guard(f->ids[arm]);}
}
static void hs_inputs_unchanged(struct hs_fixture *f) {
    struct hs_guard in[]={f->q,f->weights,f->cache,f->mask};
    for(unsigned i=0;i<4;i++){hs_guard_check(in[i]);HS_REQUIRE(f->input_hash[i]==hs_fnv(myds_metal_buffer_data(in[i].view),in[i].bytes));}
}
static void hs_exact(struct hs_guard a,struct hs_guard b) {
    HS_REQUIRE(a.bytes==b.bytes);hs_guard_check(a);hs_guard_check(b);
    if(memcmp(myds_metal_buffer_data(a.view),myds_metal_buffer_data(b.view),a.bytes)){
        hs_dump_one("comparison-reference",myds_metal_buffer_data(a.view),a.bytes);
        hs_dump_one("comparison-candidate",myds_metal_buffer_data(b.view),b.bytes);
        const uint32_t *x=myds_metal_buffer_data(a.view),*y=myds_metal_buffer_data(b.view);
        for(size_t i=0;i<a.bytes/4u;i++)if(x[i]!=y[i]){fprintf(stderr,"PARITY index=%zu reference=%08x candidate=%08x\n",i,x[i],y[i]);break;}
        HS_REQUIRE(0);
    }
}
static size_t hs_count_nonfinite(struct hs_guard b) {
    const uint32_t *p=myds_metal_buffer_data(b.view);size_t n=0;
    for(size_t i=0;i<b.bytes/4u;i++){HS_REQUIRE(p[i]!=0xa5a5a5a5u);n+=(p[i]&0x7fffffffu)>=0x7f800000u;}return n;
}
static void hs_score_contract(struct hs_fixture *f,struct hs_guard b) {
    const float *s=myds_metal_buffer_data(b.view);const uint32_t *bits=myds_metal_buffer_data(b.view);
    const float *mask=myds_metal_buffer_data(f->mask.view);size_t nonzero=0,active=0;
    for(uint32_t row=0;row<f->rows;row++){
        HS_REQUIRE(bits[row]!=0xa5a5a5a5u);
        if(f->mask_mode&&mask[row/8u]!=0.f)HS_REQUIRE(bits[row]==0xff800000u);
        else {active++;if(f->pattern==0u){HS_REQUIRE(isfinite(s[row]));nonzero+=s[row]!=0.f;}
            if(f->pattern==1u)HS_REQUIRE(s[row]==0.f);}
    }
    if(f->pattern==0u&&active)HS_REQUIRE(nonzero);hs_guard_check(b);
}
static void hs_ids_contract(struct hs_fixture *f,struct hs_guard b) {
    const uint32_t *p=myds_metal_buffer_data(b.view);
    for(size_t i=0;i<b.bytes/4u;i++)HS_REQUIRE(p[i]<f->rows);hs_guard_check(b);
}
static void hs_file(const char *stem,const char *role,const char *dtype,const void *data,size_t bytes,bool *first) {
    char name[192],path[4096];HS_REQUIRE(snprintf(name,sizeof(name),"%s-%s.%s",stem,role,dtype)<(int)sizeof(name));
    const uint64_t hash=hs_fnv(data,bytes);unsigned found=hs_native_files;
    for(unsigned i=0;i<hs_native_files;i++)if(!strcmp(hs_files[i].name,name)){found=i;break;}
    if(found==hs_native_files){
        HS_REQUIRE(hs_native_files<sizeof(hs_files)/sizeof(hs_files[0]));
        HS_REQUIRE(snprintf(path,sizeof(path),"%s/%s",hs_dir,name)<(int)sizeof(path));FILE *file=fopen(path,"wbx");HS_REQUIRE(file);
        HS_REQUIRE(fwrite(data,1,bytes,file)==bytes);HS_REQUIRE(!fclose(file));
        strcpy(hs_files[found].name,name);hs_files[found].bytes=bytes;hs_files[found].hash=hash;hs_native_files++;
    } else HS_REQUIRE(hs_files[found].bytes==bytes&&hs_files[found].hash==hash);
    printf("%s{\"path\":\"%s\",\"role\":\"%s\",\"dtype\":\"%s\",\"bytes\":%zu,\"fnv1a64\":\"%016" PRIx64 "\"}",*first?"":",",name,role,dtype,bytes,hash);*first=false;
}
static void hs_file_guard(const char *stem,const char *role,struct hs_guard b,bool *first) {
    hs_guard_check(b);hs_file(stem,role,"f32",myds_metal_buffer_data(b.view),b.bytes,first);
}
static void hs_guard_file(const char *stem,const char *role,struct hs_guard b,bool *first) {
    uint8_t raw[64];const uint8_t *p=myds_metal_buffer_data(b.owner);
    memcpy(raw,p,32);memcpy(raw+32,p+32+b.bytes,32);hs_guard_check(b);
    char guard_role[96];HS_REQUIRE(snprintf(guard_role,sizeof(guard_role),"%s-guards",role)<(int)sizeof(guard_role));
    hs_file(stem,guard_role,"bin",raw,sizeof(raw),first);
}
static void hs_input_files(struct hs_fixture *f,const char *stem,bool *first) {
    struct hs_guard b[]={f->q,f->weights,f->cache,f->mask};const char *roles[]={"q","weights","cache","mask"};
    for(unsigned i=0;i<4;i++){hs_file(stem,roles[i],i==2u&&f->bf16?"bf16":"f32",myds_metal_buffer_data(b[i].view),b[i].bytes,first);hs_guard_file(stem,roles[i],b[i],first);}
}
static void hs_outputs(struct hs_fixture *f,const char *stem,bool topk,bool *first) {
    for(unsigned arm=0;arm<2;arm++){
        const char *score=arm?"candidate-scores":"reference-scores",*ids=arm?"candidate-ids":"reference-ids";
        hs_file_guard(stem,score,f->scores[arm],first);hs_guard_file(stem,score,f->scores[arm],first);
        if(topk){hs_file(stem,ids,"i32",myds_metal_buffer_data(f->ids[arm].view),f->ids[arm].bytes,first);hs_guard_file(stem,ids,f->ids[arm],first);}
    }
}
static void hs_policy(unsigned concurrent,unsigned automatic) {
    myds_metal_set_concurrent((int)concurrent);myds_metal_set_auto_barrier((int)automatic);
    HS_REQUIRE(myds_metal_get_concurrent()==(int)concurrent&&myds_metal_get_auto_barrier()==(int)automatic);
}
static void hs_encoder(void) {
    HS_REQUIRE(g_batch_enc&&g_batch_enc.dispatchType==(myds_metal_get_concurrent()?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial));
}
static void hs_prepare(struct hs_fixture *f,struct hs_guard out) {
    if(!myds_metal_get_auto_barrier()){
        const myds_metal_buffer_t *deps[]={f->q.view,f->weights.view,f->cache.view,f->mask.view,out.view};
        myds_metal_barrier_buffers(deps,5u);
    }
}
static int hs_call(struct hs_fixture *f,struct hs_guard out,bool candidate) {
    hs_prepare(f,out);
    const myds_metal_buffer_t *mask=f->mask_mode?f->mask.view:NULL;
    return candidate?myds_v41_gpu_indexscores_head_slots(out.view,f->q.view,f->weights.view,f->cache.view,mask,f->rows,32u,f->bf16):
        indexscores(out.view,f->q.view,f->weights.view,f->cache.view,mask,f->rows,32u,f->bf16);
}
static void hs_join(struct hs_guard scores,struct hs_guard ids,bool topk) {
    const myds_metal_buffer_t *deps[]={scores.view,ids.view};
    if(!myds_metal_get_auto_barrier())myds_metal_barrier_buffers(deps,topk?2u:1u);
}
static uint64_t hs_hist(const char *name) {
    for(int i=0;i<g_hist_used;i++)if(!strcmp(g_hist[i].name,name))return g_hist[i].n;return 0;
}
static void hs_counter_check(unsigned calls) {
    myds_v41_gpu_indexer_head_slots_counters_t c;myds_v41_gpu_indexer_head_slots_counters(&c,false);
    HS_REQUIRE(c.requests==calls&&c.encoded==calls&&!c.unavailable&&!c.failures);
}
static void hs_counter_json(void) {
    myds_v41_gpu_indexer_head_slots_counters_t c;myds_v41_gpu_indexer_head_slots_counters(&c,false);
    printf(",\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"unavailable\":%" PRIu64 ",\"failures\":%" PRIu64,c.requests,c.encoded,c.unavailable,c.failures);
}
static void hs_run(struct hs_fixture *f,unsigned arm,bool topk,myds_v41_gpu_scratch_t *scratch) {
    myds_v41_gpu_indexer_head_slots_counters(NULL,true);myds_metal_dispatch_count_reset();
    const char *name=arm?HS_CANDIDATE:HS_REFERENCE;const uint64_t before=hs_hist(name);
    HS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    HS_REQUIRE(hs_call(f,f->scores[arm],arm!=0));hs_encoder();hs_join(f->scores[arm],f->ids[arm],false);
    HS_REQUIRE(myds_metal_dispatch_count()==1u);
    if(topk)HS_REQUIRE(myds_v41_gpu_topk(f->ids[arm].view,f->scores[arm].view,f->rows,f->rows<512u?f->rows:512u,scratch));
    hs_join(f->scores[arm],f->ids[arm],topk);hs_encoder();HS_REQUIRE(myds_metal_commit_batch());
    HS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);if(g_hist_enabled)HS_REQUIRE(hs_hist(name)-before==1u);hs_counter_check(arm);
    hs_score_contract(f,f->scores[arm]);if(topk)hs_ids_contract(f,f->ids[arm]);hs_inputs_unchanged(f);
}
static void hs_parity(unsigned n,uint32_t rows,bool bf16,unsigned mask,unsigned pattern,unsigned policy,myds_v41_gpu_scratch_t *scratch) {
    struct hs_fixture f=hs_create(rows,bf16,mask,pattern,31u+rows+mask*73u+pattern*41u);
    hs_active_fixture=&f;
    hs_policy(policy/2u,policy%2u);hs_run(&f,0,true,scratch);hs_run(&f,1,true,scratch);
    hs_exact(f.scores[0],f.scores[1]);hs_exact(f.ids[0],f.ids[1]);
    char stem[64],inputs[128];snprintf(stem,sizeof(stem),"case%u",n);
    snprintf(inputs,sizeof(inputs),"inputs-r%u-bf%u-mask%u-pattern%u",rows,bf16,mask,pattern);
    printf("{\"schema_version\":1,\"kind\":\"parity\",\"case\":%u,\"rows\":%u,\"heads\":32,\"head_dim\":128,\"cache_bf16\":%s,\"mask_mode\":%u,\"pattern\":%u,\"seed\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"reference_kernel\":\"%s\",\"actual_kernel\":\"%s\",\"threads\":[32,4,1],\"candidate_dynamic_tg_bytes\":640,\"reference_dynamic_tg_bytes\":528,\"nonfinite_scores\":%zu,\"full_scores_byte_exact\":true,\"full_ids_byte_exact\":true,\"input_unchanged\":true,\"canaries\":true,\"reference_candidate_scorer_dispatches\":[1,1],\"topk_prefix\":1,\"topk_nan_fuse\":1,\"runtime_mode\":1",
        n,rows,bf16?"true":"false",mask,pattern,f.seed,policy/2u,policy%2u,HS_REFERENCE,HS_CANDIDATE,hs_count_nonfinite(f.scores[1]));
    hs_counter_json();bool first=true;printf(",\"files\":[");hs_input_files(&f,inputs,&first);hs_outputs(&f,stem,true,&first);puts("]}");fflush(stdout);
    hs_parities++;hs_active_fixture=NULL;hs_destroy(&f);
}
static void hs_timing(uint32_t rows,bool bf16,bool topk,unsigned reps,myds_v41_gpu_scratch_t *scratch) {
    struct hs_fixture f=hs_create(rows,bf16,0,0,9001u);hs_policy(1,0);g_hist_enabled=0;
    hs_active_fixture=&f;
    const char *arms[]={"A1","B","A2"},*scope=topk?"score_topk":"scorer_join";
    for(unsigned arm=0;arm<3;arm++)for(unsigned sample=0;sample<=hs_samples;sample++){
        hs_fill(&f,9001u+sample);hs_run(&f,0,topk,scratch);hs_poison(f.scores[1]);hs_poison(f.ids[1]);
        myds_v41_gpu_indexer_head_slots_counters(NULL,true);myds_metal_dispatch_count_reset();
        HS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned rep=0;rep<reps;rep++){
            HS_REQUIRE(hs_call(&f,f.scores[1],arm==1u));hs_encoder();hs_join(f.scores[1],f.ids[1],false);
            if(topk)HS_REQUIRE(myds_v41_gpu_topk(f.ids[1].view,f.scores[1].view,rows,512u,scratch));
            hs_join(f.scores[1],f.ids[1],topk);
        }
        hs_encoder();const uint64_t dispatches=myds_metal_dispatch_count();HS_REQUIRE(dispatches>=reps);
        if(!topk)HS_REQUIRE(dispatches==reps);
        HS_REQUIRE(myds_metal_commit_batch());HS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
        HS_REQUIRE(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime);
        hs_counter_check(arm==1u?reps:0u);hs_exact(f.scores[0],f.scores[1]);hs_score_contract(&f,f.scores[1]);hs_inputs_unchanged(&f);
        if(topk){hs_exact(f.ids[0],f.ids[1]);hs_ids_contract(&f,f.ids[1]);}else hs_untouched(f.ids[1]);
        char stem[128],inputs[128];snprintf(stem,sizeof(stem),"timing-r%u-bf%u-%s-reps%u-%s-s%u",rows,bf16,scope,reps,arms[arm],sample);
        snprintf(inputs,sizeof(inputs),"timing-inputs-r%u-bf%u-s%u",rows,bf16,sample);
        printf("{\"schema_version\":1,\"kind\":\"gpu_timing\",\"rows\":%u,\"heads\":32,\"head_dim\":128,\"cache_bf16\":%s,\"scope\":\"%s\",\"arm\":\"%s\",\"sample\":%u,\"warmup\":%s,\"input_seed\":%u,\"repetitions\":%u,\"concurrent\":1,\"auto_barrier\":0,\"encoder_dispatch_type_verified\":true,\"actual_kernel\":\"%s\",\"threads\":[32,4,1],\"dynamic_tg_bytes\":%u,\"runtime_mode\":1,\"scorer_dispatches\":%u,\"dispatches\":%" PRIu64 ",\"caller_input_output_join\":true,\"score_raw_join\":true,\"ids_scores_output_join\":%s,\"topk_prefix\":1,\"topk_nan_fuse\":1,\"gpu_start_s\":%.12f,\"gpu_end_s\":%.12f,\"gpu_us_per_iteration\":%.9f,\"cb_status\":4,\"completed_cb\":true,\"dispatch_hist_disabled\":true,\"full_scores_byte_exact\":true,\"full_ids_byte_exact\":%s,\"input_unchanged\":true,\"canaries\":true,\"fixture_not_fullmodel\":true,\"cold_dram_not_proven\":true",
            rows,bf16?"true":"false",scope,arms[arm],sample,sample?"false":"true",f.seed,reps,arm==1u?HS_CANDIDATE:HS_REFERENCE,arm==1u?640u:528u,reps,dispatches,topk?"true":"false",cb.GPUStartTime,cb.GPUEndTime,(cb.GPUEndTime-cb.GPUStartTime)*1e6/reps,topk?"true":"false");
        hs_counter_json();bool first=true;printf(",\"files\":[");hs_input_files(&f,inputs,&first);hs_outputs(&f,stem,topk,&first);puts("]}");fflush(stdout);hs_timings++;
    }
    hs_active_fixture=NULL;hs_destroy(&f);g_hist_enabled=1;
}
static void hs_reuse(unsigned n,myds_v41_gpu_scratch_t *scratch) {
    struct hs_fixture inputs[3],shared=hs_create(16384u,true,0,0,71u);
    hs_active_fixture=&shared;
    for(unsigned i=0;i<3;i++)inputs[i]=hs_create(16384u,true,i==2?2u:0u,0,71u+i*43u);
    struct hs_guard scores[2][7],ids[2][7];for(unsigned a=0;a<2;a++)for(unsigned i=0;i<7;i++){scores[a][i]=hs_allocate(16384u*4u);ids[a][i]=hs_allocate(512u*4u);}
    const unsigned counts[]={1u,3u,3u},order[]={0u,0u,1u,2u,2u,1u,0u};uint64_t arm_dispatches[2];
    hs_policy(n/2u,n%2u);
    for(unsigned arm=0;arm<2;arm++){
        myds_v41_gpu_indexer_head_slots_counters(NULL,true);myds_metal_dispatch_count_reset();
        const uint64_t hist=hs_hist(arm?HS_CANDIDATE:HS_REFERENCE);unsigned item=0;
        for(unsigned pass=0;pass<3;pass++){
            HS_REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
            for(unsigned j=0;j<counts[pass];j++,item++){
                struct hs_fixture *in=&inputs[order[item]];shared.mask_mode=in->mask_mode;
                if(!myds_metal_get_auto_barrier()){
                    const myds_metal_buffer_t *d[]={shared.q.view,shared.weights.view,shared.cache.view,shared.mask.view};
                    myds_metal_barrier_buffers(d,4u);
                }
                HS_REQUIRE(myds_metal_buffer_copy(shared.q.view,0,in->q.view,0,in->q.bytes));
                HS_REQUIRE(myds_metal_buffer_copy(shared.weights.view,0,in->weights.view,0,in->weights.bytes));
                HS_REQUIRE(myds_metal_buffer_copy(shared.cache.view,0,in->cache.view,0,in->cache.bytes));
                HS_REQUIRE(myds_metal_buffer_copy(shared.mask.view,0,in->mask.view,0,in->mask.bytes));
                HS_REQUIRE(hs_call(&shared,shared.scores[arm],arm!=0));hs_encoder();hs_join(shared.scores[arm],shared.ids[arm],false);
                HS_REQUIRE(myds_v41_gpu_topk(shared.ids[arm].view,shared.scores[arm].view,16384u,512u,scratch));
                hs_join(shared.scores[arm],shared.ids[arm],true);
                HS_REQUIRE(myds_metal_buffer_copy(scores[arm][item].view,0,shared.scores[arm].view,0,scores[arm][item].bytes));
                HS_REQUIRE(myds_metal_buffer_copy(ids[arm][item].view,0,shared.ids[arm].view,0,ids[arm][item].bytes));
                /* Observation copies get their own RAW/final join. The real
                 * scorer->TopK consumer join precedes these snapshots. */
                hs_join(scores[arm][item],ids[arm][item],true);
            }
            hs_encoder();HS_REQUIRE(myds_metal_commit_batch());HS_REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
        }
        HS_REQUIRE(item==7u&&hs_hist(arm?HS_CANDIDATE:HS_REFERENCE)-hist==7u);hs_counter_check(arm?7u:0u);
        arm_dispatches[arm]=myds_metal_dispatch_count();
    }
    printf("{\"schema_version\":1,\"kind\":\"reuse\",\"case\":%u,\"concurrent\":%u,\"auto_barrier\":%u,\"calls_per_batch\":[1,3,3],\"encoder_dispatch_type_verified\":true,\"completed_cb\":true,\"cb_status\":4,\"actual_kernel\":\"%s\",\"reference_candidate_scorer_dispatches\":[7,7],\"reference_candidate_dispatches\":[%" PRIu64 ",%" PRIu64 "],\"shared_input_output\":true,\"caller_input_output_join\":true,\"score_raw_join\":true,\"ids_scores_output_join\":true,\"snapshot_copies_add_dependencies\":true,\"topk_prefix\":1,\"topk_nan_fuse\":1,\"full_scores_byte_exact\":true,\"full_ids_byte_exact\":true,\"input_unchanged\":true,\"canaries\":true",n,n/2u,n%2u,HS_CANDIDATE,arm_dispatches[0],arm_dispatches[1]);
    hs_counter_json();printf(",\"items\":[");unsigned item=0;
    for(unsigned pass=0;pass<3;pass++)for(unsigned j=0;j<counts[pass];j++,item++){
        const unsigned input_index=order[item];struct hs_fixture *in=&inputs[input_index];
        hs_exact(scores[0][item],scores[1][item]);hs_exact(ids[0][item],ids[1][item]);hs_inputs_unchanged(in);
        hs_score_contract(in,scores[1][item]);hs_ids_contract(in,ids[1][item]);
        char stem[96],inputstem[96];snprintf(stem,sizeof(stem),"reuse%u-pass%u-item%u",n,pass,j);snprintf(inputstem,sizeof(inputstem),"reuse-input%u",input_index);
        printf("%s{\"pass\":%u,\"item\":%u,\"input_index\":%u,\"rows\":16384,\"cache_bf16\":true,\"mask_mode\":%u,\"seed\":%u,\"files\":[",item?",":"",pass,j,input_index,in->mask_mode,in->seed);
        bool first=true;hs_input_files(in,inputstem,&first);
        for(unsigned a=0;a<2;a++){const char *r=a?"candidate-scores":"reference-scores",*ri=a?"candidate-ids":"reference-ids";
            hs_file_guard(stem,r,scores[a][item],&first);hs_guard_file(stem,r,scores[a][item],&first);
            hs_file(stem,ri,"i32",myds_metal_buffer_data(ids[a][item].view),ids[a][item].bytes,&first);hs_guard_file(stem,ri,ids[a][item],&first);}
        printf("]}");
    }
    puts("]}");fflush(stdout);
    for(unsigned a=0;a<2;a++)for(unsigned i=0;i<7;i++){hs_destroy_guard(scores[a][i]);hs_destroy_guard(ids[a][i]);}
    for(unsigned i=0;i<3;i++)hs_destroy(&inputs[i]);hs_active_fixture=NULL;hs_destroy(&shared);
}
@interface HSRestrictedPipeline : NSProxy {id<MTLComputePipelineState> actual;unsigned limitation;}
+(id)withPipeline:(id<MTLComputePipelineState>)pipeline limitation:(unsigned)kind;
@end
@implementation HSRestrictedPipeline
+(id)withPipeline:(id<MTLComputePipelineState>)pipeline limitation:(unsigned)kind {
    HSRestrictedPipeline *p=[HSRestrictedPipeline alloc];p->actual=pipeline;p->limitation=kind;return p;
}
-(NSUInteger)maxTotalThreadsPerThreadgroup {return limitation==0u?64u:actual.maxTotalThreadsPerThreadgroup;}
-(NSUInteger)threadExecutionWidth {return limitation==1u?16u:actual.threadExecutionWidth;}
-(NSUInteger)staticThreadgroupMemoryLength {return limitation==2u?g_device.maxThreadgroupMemoryLength:actual.staticThreadgroupMemoryLength;}
-(NSMethodSignature *)methodSignatureForSelector:(SEL)selector{return [(NSObject *)actual methodSignatureForSelector:selector];}
-(void)forwardInvocation:(NSInvocation *)invocation{[invocation invokeWithTarget:actual];}
@end
static void hs_rejected(unsigned n,const char *reason,struct hs_fixture *f,myds_metal_buffer_t *out,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *w,const myds_metal_buffer_t *cache,const myds_metal_buffer_t *mask,
    uint32_t rows,uint32_t heads,bool unavailable) {
    uint8_t *saved=malloc(f->scores[1].bytes);HS_REQUIRE(saved);memcpy(saved,myds_metal_buffer_data(f->scores[1].view),f->scores[1].bytes);
    const size_t actual_bytes=out?myds_metal_buffer_size(out):0u;
    uint8_t *actual_saved=actual_bytes?malloc(actual_bytes):NULL;HS_REQUIRE(!actual_bytes||actual_saved);
    if(actual_bytes)memcpy(actual_saved,myds_metal_buffer_data(out),actual_bytes);
    myds_v41_gpu_indexer_head_slots_counters(NULL,true);myds_metal_dispatch_count_reset();
    HS_REQUIRE(myds_metal_begin_batch());HS_REQUIRE(!myds_v41_gpu_indexscores_head_slots(out,q,w,cache,mask,rows,heads,f->bf16));
    HS_REQUIRE(myds_metal_dispatch_count()==0u);HS_REQUIRE(myds_metal_commit_batch());
    myds_v41_gpu_indexer_head_slots_counters_t c;myds_v41_gpu_indexer_head_slots_counters(&c,false);
    HS_REQUIRE(c.requests==1u&&!c.encoded&&c.unavailable==(unsigned)unavailable&&c.failures==1u);
    HS_REQUIRE(!memcmp(saved,myds_metal_buffer_data(f->scores[1].view),f->scores[1].bytes));free(saved);hs_untouched(f->scores[1]);hs_inputs_unchanged(f);
    if(actual_bytes)HS_REQUIRE(!memcmp(actual_saved,myds_metal_buffer_data(out),actual_bytes));free(actual_saved);
    char stem[64];snprintf(stem,sizeof(stem),"reject%u",n);
    printf("{\"schema_version\":1,\"kind\":\"rejected_call\",\"case\":%u,\"reason\":\"%s\",\"dispatches\":0,\"no_fallback\":true,\"output_untouched\":true,\"actual_writer_bytes\":%zu,\"actual_writer_untouched\":true,\"actual_writer_is_primary\":%s,\"input_unchanged\":true,\"canaries\":true",n,reason,actual_bytes,out==f->scores[1].view?"true":"false");hs_counter_json();
    bool first=true;printf(",\"files\":[");hs_file(stem,"output","bin",myds_metal_buffer_data(f->scores[1].view),f->scores[1].bytes,&first);hs_guard_file(stem,"output",f->scores[1],&first);
    if(actual_bytes)hs_file(stem,"actual-writer","bin",myds_metal_buffer_data(out),actual_bytes,&first);
    puts("]}");fflush(stdout);
}
static unsigned hs_rejections(void) {
    struct hs_fixture f=hs_create(31u,true,2,0,91u);unsigned n=0;hs_policy(1,0);
#define HS_REJECT(reason,o,q,w,k,m,r,h,u) hs_rejected(n++,reason,&f,o,q,w,k,m,r,h,u)
    HS_REJECT("null_output",NULL,f.q.view,f.weights.view,f.cache.view,f.mask.view,31,32,false);
    HS_REJECT("null_query",f.scores[1].view,NULL,f.weights.view,f.cache.view,f.mask.view,31,32,false);
    HS_REJECT("null_weights",f.scores[1].view,f.q.view,NULL,f.cache.view,f.mask.view,31,32,false);
    HS_REJECT("null_cache",f.scores[1].view,f.q.view,f.weights.view,NULL,f.mask.view,31,32,false);
    HS_REJECT("zero_rows",f.scores[1].view,f.q.view,f.weights.view,f.cache.view,f.mask.view,0,32,false);
    HS_REJECT("unsupported_heads",f.scores[1].view,f.q.view,f.weights.view,f.cache.view,f.mask.view,31,31,false);
    struct hs_guard b[]={f.scores[1],f.q,f.weights,f.cache,f.mask};
    const char *short_names[]={"short_output","short_query","short_weights","short_cache","short_mask"};
    for(unsigned i=0;i<5;i++){
        const size_t less=i==3?2u:4u;myds_metal_buffer_t *v=myds_metal_buffer_sub_view(b[i].view,0,b[i].bytes-less);HS_REQUIRE(v);
        HS_REJECT(short_names[i],i==0?v:b[0].view,i==1?v:b[1].view,i==2?v:b[2].view,i==3?v:b[3].view,i==4?v:b[4].view,31,32,false);myds_metal_buffer_free(v);
    }
    const char *align_names[]={"unaligned_output","unaligned_query","unaligned_weights","unaligned_cache","unaligned_mask"};
    for(unsigned i=0;i<5;i++){
        const size_t delta=i==1?4u:1u;myds_metal_buffer_t *v=myds_metal_buffer_sub_view(b[i].owner,32u+delta,b[i].bytes);HS_REQUIRE(v);
        HS_REJECT(align_names[i],i==0?v:b[0].view,i==1?v:b[1].view,i==2?v:b[2].view,i==3?v:b[3].view,i==4?v:b[4].view,31,32,false);myds_metal_buffer_free(v);
    }
    const char *aliases[]={"output_alias_query","output_alias_weights","output_alias_cache","output_alias_mask"};
    for(unsigned i=0;i<4;i++){
        /* For the tiny mask extend its view into the real owner, so rejection
         * proves overlap rather than merely the score-size precondition. */
        myds_metal_buffer_t *out=i==3?NULL:b[i+1].view;
        if(i==3){
            /* A separate, adequately sized owner makes the mask reader and
             * score writer ranges genuinely overlap while preserving input. */
            struct hs_guard large=hs_allocate(31u*4u);float *m=myds_metal_buffer_data(large.view);for(unsigned j=0;j<31;j++)m[j]=j%2u;
            const uint64_t hash=hs_fnv(m,large.bytes);
            HS_REJECT(aliases[i],large.view,f.q.view,f.weights.view,f.cache.view,large.view,31,32,false);
            HS_REQUIRE(hash==hs_fnv(m,large.bytes));hs_destroy_guard(large);
        }else HS_REJECT(aliases[i],out,f.q.view,f.weights.view,f.cache.view,f.mask.view,31,32,false);
    }
    id<MTLComputePipelineState> saved=g_pso_indexer_head_slots;
    g_pso_indexer_head_slots=nil;HS_REJECT("actual_missing_candidate_pso",f.scores[1].view,f.q.view,f.weights.view,f.cache.view,f.mask.view,31,32,true);g_pso_indexer_head_slots=saved;
    const char *geometry[]={"restricted_max_threads","restricted_simd_width","restricted_tg_memory"};
    for(unsigned kind=0;kind<3;kind++){g_pso_indexer_head_slots=(id<MTLComputePipelineState>)[HSRestrictedPipeline withPipeline:saved limitation:kind];
        HS_REJECT(geometry[kind],f.scores[1].view,f.q.view,f.weights.view,f.cache.view,f.mask.view,31,32,true);g_pso_indexer_head_slots=saved;}
#undef HS_REJECT
    hs_destroy(&f);return n;
}
static void hs_clean(void) {
    HS_REQUIRE(!g_initialized&&!g_device&&!g_library&&!g_queue&&!g_pso_indexer_head_slots&&!myds_v41_gpu_indexer_head_slots_enabled());
}
static void hs_host_config(void) {
    hs_clean();unsetenv("MYDS_INDEXER_HEAD_SLOTS");HS_REQUIRE(indexer_head_slots_parse_mode()==0);
    HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS","0",1));HS_REQUIRE(indexer_head_slots_parse_mode()==0);
    HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS","1",1));HS_REQUIRE(indexer_head_slots_parse_mode()==1);
    const char *bad[]={"","2","-1","+1","01","0x1"," 1","1 ","1\n","true","false","00"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);i++){
        HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS",bad[i],1));HS_REQUIRE(indexer_head_slots_parse_mode()<0);
        HS_REQUIRE(!myds_metal_init(hs_shader));hs_clean();
    }
    myds_v41_gpu_indexer_head_slots_counters(NULL,true);
    HS_REQUIRE(!myds_v41_gpu_indexscores_head_slots(NULL,NULL,NULL,NULL,NULL,1u,32u,true));
    myds_v41_gpu_indexer_head_slots_counters_t c;myds_v41_gpu_indexer_head_slots_counters(&c,true);
    HS_REQUIRE(c.requests==1u&&!c.encoded&&c.unavailable==1u&&c.failures==1u);hs_counter_check(0);
    unsetenv("MYDS_INDEXER_HEAD_SLOTS");
    puts("{\"schema_version\":1,\"kind\":\"host_config\",\"valid_values\":[0,1],\"absent_default\":0,\"invalid_values_rejected_before_metal_resources\":12,\"counter_reset\":true,\"no_gpu_initialization\":true,\"pass\":true}");
}
static void hs_ambient(void) {
    const char *off[]={"MYDS_BATCH2_Q8_SHARED","MYDS_MOE_Q2_SUM_CACHE","MYDS_MOE_Q2_SUM_CACHE_FUSED","MYDS_MOE_IQ2_CONSTANT_LUT","MYDS_INDEXER_COMPACT_SCORER","MYDS_RESIDENCY_BYPASS","MYDS_V41_WEIGHT_STAGE"};
    for(unsigned i=0;i<sizeof(off)/sizeof(off[0]);i++)unsetenv(off[i]);
    HS_REQUIRE(!setenv("MYDS_METAL_CONCURRENT","0",1));
    HS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE","1",1));HS_REQUIRE(!setenv("MYDS_INDEXER_TOPK_NAN_FUSE","1",1));
    g_hist_enabled=1;
}
static void hs_smoke_initialization(const char *phase,int mode,bool missing) {
    HS_REQUIRE(myds_v41_gpu_indexer_head_slots_enabled()==mode);
    struct hs_fixture f=hs_create(65u,true,0,0,47u);hs_policy(0,1);hs_run(&f,0,false,NULL);
    hs_run(&f,0,false,NULL);hs_score_contract(&f,f.scores[0]);
    if(!mode)HS_REQUIRE(!g_pso_indexer_head_slots);
    /* A changed environment cannot alter this initialized selector. */
    HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS",mode?"0":"1",1));HS_REQUIRE(myds_v41_gpu_indexer_head_slots_enabled()==mode);
    if(!mode){
        myds_v41_gpu_indexer_head_slots_counters(NULL,true);myds_metal_dispatch_count_reset();
        HS_REQUIRE(!myds_v41_gpu_indexscores_head_slots(f.scores[1].view,f.q.view,f.weights.view,f.cache.view,NULL,65u,32u,true));
        HS_REQUIRE(myds_metal_dispatch_count()==0u);hs_untouched(f.scores[1]);
        myds_v41_gpu_indexer_head_slots_counters_t c;myds_v41_gpu_indexer_head_slots_counters(&c,false);
        HS_REQUIRE(c.requests==1u&&!c.encoded&&c.unavailable==1u&&c.failures==1u);
    }
    printf("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"%s\",\"initialized\":true,\"actual_mode\":%d,\"immutable_after_env_change\":true,\"reference_success\":true,\"candidate_symbol_missing\":%s,\"candidate_pso_available\":%s,\"reference_kernel\":\"%s\",\"completed_cb\":true,\"cb_status\":4,\"canaries\":true,\"input_unchanged\":true",phase,mode,missing?"true":"false",g_pso_indexer_head_slots?"true":"false",HS_REFERENCE);
    bool first=true;printf(",\"files\":[");hs_input_files(&f,"initialization-inputs",&first);hs_file_guard(phase,"reference-scores",f.scores[0],&first);hs_guard_file(phase,"reference-scores",f.scores[0],&first);puts("]}");fflush(stdout);
    hs_destroy(&f);
}
static void hs_missing(void) {
    HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS","1",1));HS_REQUIRE(!myds_metal_init(hs_shader));hs_clean();
    puts("{\"schema_version\":1,\"kind\":\"initialization\",\"phase\":\"missing-explicit1\",\"initialized\":false,\"actual_mode\":0,\"cleanup_complete\":true,\"no_fallback\":true}");
    HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS","0",1));HS_REQUIRE(myds_metal_init(hs_shader));
    HS_REQUIRE(![g_library.functionNames containsObject:@HS_CANDIDATE]);hs_smoke_initialization("missing-reference0",0,true);myds_metal_cleanup();hs_clean();
    unsetenv("MYDS_INDEXER_HEAD_SLOTS");HS_REQUIRE(myds_metal_init(hs_shader));
    HS_REQUIRE(![g_library.functionNames containsObject:@HS_CANDIDATE]);hs_smoke_initialization("missing-default0-reinit",0,true);myds_metal_cleanup();hs_clean();
    printf("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"missing-library\",\"initializations\":3,\"native_files\":%u,\"pass\":true}\n",hs_native_files);
}
static unsigned hs_number(const char *s) {
    char *end=NULL;errno=0;unsigned long n=strtoul(s,&end,10);
    HS_REQUIRE(s[0]&&s[0]!='-'&&s[0]!='+'&&!errno&&end&&!*end&&n>0&&n<=1024u);return (unsigned)n;
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--host-only"))hs_host_only=1;
        else if(!strcmp(argv[i],"--expect-missing-candidate"))hs_expect_missing=1;
        else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)hs_shader=argv[++i];
        else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)hs_dir=argv[++i];
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)hs_samples=hs_number(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)hs_repetitions=hs_number(argv[++i]);
        else {fprintf(stderr,"usage: %s --shader-dir DIR --output-dir NEW_DIR [--samples 5 --repetitions 32] [--host-only] [--expect-missing-candidate]\n",argv[0]);return 2;}
    }
    HS_REQUIRE(hs_samples==5u&&hs_repetitions==32u);hs_host_config();if(hs_host_only)return 0;
    HS_REQUIRE(hs_dir);struct stat st;errno=0;HS_REQUIRE(lstat(hs_dir,&st)!=0&&errno==ENOENT);HS_REQUIRE(!mkdir(hs_dir,0700));hs_ambient();
    if(hs_expect_missing){hs_missing();return 0;}
    HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS","0",1));HS_REQUIRE(myds_metal_init(hs_shader));hs_smoke_initialization("reference0",0,false);myds_metal_cleanup();hs_clean();
    HS_REQUIRE(!setenv("MYDS_INDEXER_HEAD_SLOTS","1",1));HS_REQUIRE(myds_metal_init(hs_shader));hs_smoke_initialization("candidate1",1,false);
    HS_REQUIRE(g_pso_indexer_head_slots&&g_pso_indexer_head_slots.threadExecutionWidth==32u&&indexer_head_slots_geometry());
    printf("{\"schema_version\":1,\"kind\":\"identity\",\"device\":\"%s\",\"candidate_kernel\":\"%s\",\"reference_kernel\":\"%s\",\"threads\":[32,4,1],\"candidate_dynamic_tg_bytes\":640,\"candidate_static_tg_bytes\":%lu,\"candidate_max_threads\":%lu,\"candidate_simd_width\":%lu,\"nonzero_aligned_view_offset\":32,\"runtime_mode\":1,\"topk_prefix\":1,\"topk_nan_fuse\":1,\"ambient_env_cleared\":true,\"fixture_not_fullmodel\":true}\n",
        g_device.name.UTF8String,HS_CANDIDATE,HS_REFERENCE,(unsigned long)g_pso_indexer_head_slots.staticThreadgroupMemoryLength,(unsigned long)g_pso_indexer_head_slots.maxTotalThreadsPerThreadgroup,(unsigned long)g_pso_indexer_head_slots.threadExecutionWidth);
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(HS_MAX_ROWS);HS_REQUIRE(scratch);
    const uint32_t small[]={1u,31u,65u,512u},large[]={16384u,32768u};unsigned n=0;
    for(unsigned s=0;s<4;s++)for(unsigned bf=0;bf<2;bf++)for(unsigned p=0;p<4;p++)for(unsigned mask=0;mask<3;mask++)for(unsigned policy=0;policy<4;policy++)
        hs_parity(n++,small[s],bf,mask,p,policy,scratch);
    for(unsigned s=0;s<2;s++)for(unsigned bf=0;bf<2;bf++)for(unsigned m=0;m<2;m++)for(unsigned policy=0;policy<4;policy++)
        hs_parity(n++,large[s],bf,m?2u:0u,0u,policy,scratch);
    hs_parity(n++,HS_MAX_ROWS,true,0,0,2,scratch);HS_REQUIRE(n==417u&&hs_parities==417u);
    for(unsigned policy=0;policy<4;policy++)hs_reuse(policy,scratch);
    const unsigned rejections=hs_rejections();HS_REQUIRE(rejections==24u);
    for(unsigned s=0;s<2;s++)for(unsigned bf=0;bf<2;bf++)for(unsigned topk=0;topk<2;topk++){
        hs_timing(large[s],bf,topk,1u,scratch);hs_timing(large[s],bf,topk,hs_repetitions,scratch);
    }
    HS_REQUIRE(hs_timings==288u);myds_v41_gpu_scratch_free(scratch);myds_metal_cleanup();hs_clean();
    unsetenv("MYDS_INDEXER_HEAD_SLOTS");HS_REQUIRE(myds_metal_init(hs_shader));hs_smoke_initialization("default0-reinit",0,false);myds_metal_cleanup();hs_clean();
    printf("{\"schema_version\":1,\"kind\":\"suite_complete\",\"mode\":\"main\",\"parity_cases\":%u,\"reuse_cases\":4,\"rejected_calls\":%u,\"timing_records\":%u,\"initializations\":3,\"native_files\":%u,\"full_scores_byte_exact\":true,\"full_ids_byte_exact\":true,\"pass\":true}\n",hs_parities,rejections,hs_timings,hs_native_files);
    return 0;
}
