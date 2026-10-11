/* Real, model-free V4.1 Engram projection + consumer acceptance. The public
 * probe links the production C helper. The runtime inclusion only retains
 * completed CB timestamps and injects missing existing PSOs. --host-only is
 * CPU-only; all other modes belong in an exclusive GPU validation window. */
#include "myds_metal.m"
#include "myds_v41.h"
#include "myds_internal.h"
#include <inttypes.h>
#include <sys/stat.h>

#define DIM 5120u
#define WIDTH 6144u
#define KV_FLOATS (5u * DIM)
#define RES_FLOATS (4u * DIM)
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); fflush(stdout); fflush(stderr); exit(1); } } while (0)
struct guarded { myds_metal_buffer_t *owner,*view; size_t bytes,offset; };
struct fixture { struct guarded weight,input,initial,q,k,kv[2],res[2]; };
static const char *shader_dir="bin",*output_dir;
static unsigned samples=5,repetitions=32;
static int host_only,no_timing;
static myds_model_t blank_model;
static myds_v41_context_t *policies[2];

static uint64_t fnv(const void *data,size_t bytes) {
    const uint8_t *p=data; uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;++i){h^=p[i];h*=UINT64_C(1099511628211);} return h;
}
static uint32_t rounded(uint32_t bits) {
    if((bits&0x7f800000u)!=0x7f800000u) bits+=0x7fffu+((bits>>16u)&1u);
    return bits&0xffff0000u;
}
static uint32_t random_bits(uint32_t i,unsigned seed) {
    uint32_t z=i*747796405u+seed*2891336453u;
    z=((z>>((z>>28)+4))^z)*277803737u; return (z>>22)^z;
}
static struct guarded allocate(size_t bytes,size_t offset) {
    struct guarded b={.bytes=bytes,.offset=offset};
    b.owner=myds_metal_buffer_alloc(bytes+offset+16u); REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+offset+16u);
    b.view=myds_metal_buffer_sub_view(b.owner,offset,bytes); REQUIRE(b.view); return b;
}
static void canary(struct guarded b) {
    const uint8_t *p=myds_metal_buffer_data(b.owner);
    for(size_t i=0;i<b.offset;++i) REQUIRE(p[i]==0xa5);
    for(size_t i=0;i<16u;++i) REQUIRE(p[b.offset+b.bytes+i]==0xa5);
}
static void release(struct guarded b) {canary(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void finite(struct guarded b) {
    const float *p=myds_metal_buffer_data(b.view);size_t nonzero=0;
    for(size_t i=0;i<b.bytes/4u;++i){uint32_t bits;memcpy(&bits,p+i,4u);REQUIRE(isfinite(p[i])&&bits!=0xa5a5a5a5u);nonzero+=p[i]!=0;}
    REQUIRE(nonzero);canary(b);
}
static void export_blob(const char *prefix,const char *suffix,struct guarded b) {
    if(!output_dir)return;char path[4096];
    int n=snprintf(path,sizeof(path),"%s/%s-%s.f32",output_dir,prefix,suffix);REQUIRE(n>0&&(size_t)n<sizeof(path));
    FILE *file=fopen(path,"wx");REQUIRE(file);REQUIRE(fwrite(myds_metal_buffer_data(b.view),1,b.bytes,file)==b.bytes);REQUIRE(!fclose(file));
}
static void fill_inputs(struct fixture *f,unsigned seed,int boundary) {
    static const uint32_t ties[]={0x3f807fffu,0x3f808000u,0x3f808001u,0x3f817fffu,0x3f818000u,0x3f818001u,
        0xbf807fffu,0xbf808000u,0xbf808001u,0xbf817fffu,0xbf818000u,0xbf818001u,0x3d807fffu,0x3d808000u,0xbd808000u,0x3e818000u};
    float *x=myds_metal_buffer_data(f->input.view),*r=myds_metal_buffer_data(f->initial.view);
    for(uint32_t i=0;i<WIDTH;++i){if(boundary){uint32_t bits=ties[i%(sizeof(ties)/sizeof(*ties))];memcpy(x+i,&bits,4u);}else x[i]=((int)(random_bits(i,seed)%1023u)-511)/257.0f;}
    for(uint32_t i=0;i<RES_FLOATS;++i){float v=((int)(random_bits(i,seed+71u)%1023u)-511)/513.0f;uint32_t bits;memcpy(&bits,&v,4u);bits=rounded(bits);memcpy(r+i,&bits,4u);}
}
static struct fixture create(int boundary) {
    struct fixture f={0};f.weight=allocate((size_t)KV_FLOATS*WIDTH*2u,16u);
    f.input=allocate(WIDTH*4u,32u);f.initial=allocate(RES_FLOATS*4u,48u);
    f.q=allocate(RES_FLOATS*4u,64u);f.k=allocate(RES_FLOATS*4u,80u);
    for(unsigned i=0;i<2u;++i){f.kv[i]=allocate(KV_FLOATS*4u,96u+16u*i);f.res[i]=allocate(RES_FLOATS*4u,128u+16u*i);}
    uint16_t *w=myds_metal_buffer_data(f.weight.view);
    if(boundary){memset(w,0,f.weight.bytes);for(uint32_t row=0;row<KV_FLOATS;++row)w[(size_t)row*WIDTH+row%WIDTH]=0x3c00u;}
    else for(size_t i=0;i<f.weight.bytes/2u;++i){uint32_t z=random_bits((uint32_t)i,17u);w[i]=(uint16_t)(0x2400u+(z%0x800u)|((z>>16u)&0x8000u));}
    float *q=myds_metal_buffer_data(f.q.view),*k=myds_metal_buffer_data(f.k.view);
    for(uint32_t i=0;i<RES_FLOATS;++i){q[i]=0.5f+(random_bits(i,29u)%1024u)/1024.0f;k[i]=0.5f+(random_bits(i,43u)%1024u)/1024.0f;}
    fill_inputs(&f,31u,boundary);return f;
}
static void destroy(struct fixture *f) {
    release(f->weight);release(f->input);release(f->initial);release(f->q);release(f->k);
    for(unsigned i=0;i<2u;++i){release(f->kv[i]);release(f->res[i]);}
}
static void reset_arm(struct fixture *f,unsigned arm) {
    memcpy(myds_metal_buffer_data(f->res[arm].view),myds_metal_buffer_data(f->initial.view),f->initial.bytes);
    memset(myds_metal_buffer_data(f->kv[arm].view),0xa5,f->kv[arm].bytes);
}
static int pair(const myds_v41_context_t *policy,struct fixture *f,unsigned arm,
                bool ordinary,bool logits,bool trace) {
    return myds_v41_engram_project_round_probe(policy,f->res[arm].view,f->kv[arm].view,
        f->weight.view,f->input.view,f->q.view,f->k.view,1e-6f,ordinary,logits,trace);
}
static void verify_encoder(void) {
    REQUIRE(g_batch_enc);
    REQUIRE(g_batch_enc.dispatchType==(myds_metal_get_concurrent()?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial));
}
static id<MTLCommandBuffer> run(const myds_v41_context_t *policy,struct fixture *f,unsigned arm,
                             unsigned reps,bool ordinary,bool logits,bool trace) {
    REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
    for(unsigned i=0;i<reps;++i)REQUIRE(pair(policy,f,arm,ordinary,logits,trace));
    verify_encoder();
    REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);return cb;
}
static size_t check(struct guarded ref_kv,struct guarded candidate_kv,struct guarded ref_res,struct guarded candidate_res) {
    REQUIRE(ref_kv.bytes==candidate_kv.bytes&&ref_res.bytes==candidate_res.bytes);
    finite(ref_kv);finite(candidate_kv);finite(ref_res);finite(candidate_res);
    const uint32_t *a=myds_metal_buffer_data(ref_kv.view),*b=myds_metal_buffer_data(candidate_kv.view);size_t differences=0;
    for(size_t i=0;i<ref_kv.bytes/4u;++i){REQUIRE((a[i]&0xffffu)==0&&rounded(b[i])==a[i]);differences+=a[i]!=b[i];}
    if(memcmp(myds_metal_buffer_data(ref_res.view),myds_metal_buffer_data(candidate_res.view),ref_res.bytes)){
        const uint32_t *x=myds_metal_buffer_data(ref_res.view),*y=myds_metal_buffer_data(candidate_res.view);
        for(size_t i=0;i<ref_res.bytes/4u;++i)if(x[i]!=y[i]){fprintf(stderr,"PARITY residual index=%zu reference=%08x candidate=%08x\n",i,x[i],y[i]);break;}
    }
    REQUIRE(!memcmp(myds_metal_buffer_data(ref_res.view),myds_metal_buffer_data(candidate_res.view),ref_res.bytes));return differences;
}
static void export_pair(const char *prefix,struct fixture *f,unsigned candidate_arm) {
    export_blob(prefix,"input",f->input);export_blob(prefix,"initial-residual",f->initial);
    export_blob(prefix,"reference-kv",f->kv[0]);export_blob(prefix,"candidate-kv",f->kv[candidate_arm]);
    export_blob(prefix,"reference-residual",f->res[0]);export_blob(prefix,"candidate-residual",f->res[candidate_arm]);
}
static void numeric(struct fixture *f,unsigned case_id,const char *name,bool ordinary,bool logits,bool trace) {
    const bool skip=ordinary&&logits&&!trace;reset_arm(f,0);reset_arm(f,1);
    run(policies[0],f,0,1u,ordinary,logits,trace);
    myds_v41_engram_project_round_counters_t a,b;myds_v41_engram_project_round_counters(&a,false);
    uint64_t dispatch=myds_metal_dispatch_count();run(policies[1],f,1,1u,ordinary,logits,trace);
    myds_v41_engram_project_round_counters(&b,false);uint64_t calls=myds_metal_dispatch_count()-dispatch;
    char prefix[64];snprintf(prefix,sizeof(prefix),"case%u",case_id);export_pair(prefix,f,1);
    size_t diff=check(f->kv[0],f->kv[1],f->res[0],f->res[1]);
    REQUIRE(skip?diff>0:diff==0);REQUIRE(calls==(skip?2u:3u));
    REQUIRE(b.requests-a.requests==skip&&b.encoded-a.encoded==skip&&b.failures==a.failures);
    printf("{\"kind\":\"parity\",\"case\":%u,\"name\":\"%s\",\"dim\":5120,\"input_width\":6144,\"projected_floats\":25600,\"residual_floats\":20480,\"weight_bytes\":%zu,\"concurrent\":%d,\"auto_barrier\":%d,\"ordinary_decode\":%s,\"need_logits\":%s,\"trace_on\":%s,\"actual_round_skip\":%u,\"requests\":%u,\"encoded\":%u,\"failures\":0,\"dispatches\":%llu,\"raw_projection_differences\":%zu,\"input_fnv1a64\":\"%016" PRIx64 "\",\"initial_residual_fnv1a64\":\"%016" PRIx64 "\",\"reference_kv_fnv1a64\":\"%016" PRIx64 "\",\"candidate_kv_fnv1a64\":\"%016" PRIx64 "\",\"residual_fnv1a64\":\"%016" PRIx64 "\",\"rounded_projection_bit_exact\":true,\"full_f32_residual_bit_exact\":true,\"finite\":true,\"nonpoison\":true,\"canaries\":true,\"completed_cb\":true,\"encoder_dispatch_type_verified\":true,\"helper_consumer_join\":true}\n",
        case_id,name,f->weight.bytes,myds_metal_get_concurrent(),myds_metal_get_auto_barrier(),ordinary?"true":"false",logits?"true":"false",trace?"true":"false",skip,skip,skip,(unsigned long long)calls,diff,
        fnv(myds_metal_buffer_data(f->input.view),f->input.bytes),fnv(myds_metal_buffer_data(f->initial.view),f->initial.bytes),
        fnv(myds_metal_buffer_data(f->kv[0].view),f->kv[0].bytes),fnv(myds_metal_buffer_data(f->kv[1].view),f->kv[1].bytes),fnv(myds_metal_buffer_data(f->res[1].view),f->res[1].bytes));
}

static void reuse(struct fixture *f,unsigned case_id) {
    const unsigned counts[]={1u,3u,3u};struct guarded inputs[3],kv[2][3][3],res[2][3][3];
    for(unsigned i=0;i<3u;++i){inputs[i]=allocate(WIDTH*4u,16u+16u*i);fill_inputs(f,113u+i,0);memcpy(myds_metal_buffer_data(inputs[i].view),myds_metal_buffer_data(f->input.view),inputs[i].bytes);}
    fill_inputs(f,47u,0);myds_v41_engram_project_round_counters_t before,after;myds_v41_engram_project_round_counters(&before,false);
    for(unsigned arm=0;arm<2u;++arm){reset_arm(f,arm);
        for(unsigned pass=0;pass<3u;++pass){
            for(unsigned j=0;j<counts[pass];++j){kv[arm][pass][j]=allocate(KV_FLOATS*4u,16u);res[arm][pass][j]=allocate(RES_FLOATS*4u,32u);}
            REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
            for(unsigned j=0;j<counts[pass];++j){unsigned index=pass==2u?2u-j:j;
                /* The preceding pair's own consumer join completed its input
                 * read. The producer's own input fence orders this GPU copy
                 * and the preceding snapshot readers before projected writes. */
                REQUIRE(myds_metal_buffer_copy(f->input.view,0,inputs[index].view,0,inputs[index].bytes));
                REQUIRE(pair(policies[arm],f,arm,true,true,false));
                /* These snapshots follow the real helper's consumer join.
                 * Copies add observation dependencies, disclosed below. */
                REQUIRE(myds_metal_buffer_copy(kv[arm][pass][j].view,0,f->kv[arm].view,0,kv[arm][pass][j].bytes));
                REQUIRE(myds_metal_buffer_copy(res[arm][pass][j].view,0,f->res[arm].view,0,res[arm][pass][j].bytes));
            }
            verify_encoder();REQUIRE(myds_metal_commit_batch());REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
        }
    }
    myds_v41_engram_project_round_counters(&after,false);REQUIRE(after.requests-before.requests==7u&&after.encoded-before.encoded==7u&&after.failures==before.failures);
    printf("{\"kind\":\"reuse\",\"case\":%u,\"concurrent\":%d,\"auto_barrier\":%d,\"calls_per_batch\":[1,3,3],\"requests\":7,\"encoded\":7,\"failures\":0,\"helper_consumer_join\":true,\"caller_projected_fence_added\":false,\"snapshot_copies_add_dependencies\":true,\"completed_cb\":true,\"encoder_dispatch_type_verified\":true,\"items\":[",case_id,myds_metal_get_concurrent(),myds_metal_get_auto_barrier());
    int first=1;for(unsigned pass=0;pass<3u;++pass)for(unsigned j=0;j<counts[pass];++j){
        char prefix[96];snprintf(prefix,sizeof(prefix),"reuse%u-pass%u-item%u",case_id,pass,j);
        export_blob(prefix,"reference-kv",kv[0][pass][j]);export_blob(prefix,"candidate-kv",kv[1][pass][j]);
        export_blob(prefix,"reference-residual",res[0][pass][j]);export_blob(prefix,"candidate-residual",res[1][pass][j]);
        REQUIRE(check(kv[0][pass][j],kv[1][pass][j],res[0][pass][j],res[1][pass][j])>0);
        printf("%s{\"pass\":%u,\"item\":%u,\"input_index\":%u,\"reference_kv_fnv1a64\":\"%016" PRIx64 "\",\"candidate_kv_fnv1a64\":\"%016" PRIx64 "\",\"residual_fnv1a64\":\"%016" PRIx64 "\"}",first?"":",",pass,j,pass==2u?2u-j:j,
            fnv(myds_metal_buffer_data(kv[0][pass][j].view),kv[0][pass][j].bytes),fnv(myds_metal_buffer_data(kv[1][pass][j].view),kv[1][pass][j].bytes),fnv(myds_metal_buffer_data(res[1][pass][j].view),res[1][pass][j].bytes));first=0;
        for(unsigned arm=0;arm<2u;++arm){release(kv[arm][pass][j]);release(res[arm][pass][j]);}
    }
    puts("],\"rounded_projection_bit_exact\":true,\"full_f32_residual_bit_exact\":true,\"finite\":true,\"nonpoison\":true,\"canaries\":true}");
    for(unsigned i=0;i<3u;++i)release(inputs[i]);
}

/* Match Batch2's two distinct inputs/residual views and shared projected KV.
 * The oracle completes each scalar token in its own CB. The paired arm has
 * no copies, observations, caller barriers, or encoder breaks between adds
 * and HC consumers: only the production helper supplies the dependency. */
static void batch2_hc_dependency(struct fixture *f,unsigned case_id) {
    struct guarded inputs=allocate(2u*WIDTH*4u,32u),pre=allocate(8u*4u,16u),norm=allocate(DIM*4u,32u);
    struct guarded res[2],sum[2],out[2];
    myds_metal_buffer_t *input_rows[2],*pre_rows[2],*res_rows[2][2],*sum_rows[2][2],*out_rows[2][2];
    for(unsigned t=0;t<2u;++t){
        fill_inputs(f,211u+31u*t,0);
        memcpy((uint8_t *)myds_metal_buffer_data(inputs.view)+t*WIDTH*4u,myds_metal_buffer_data(f->input.view),WIDTH*4u);
        input_rows[t]=myds_metal_buffer_sub_view(inputs.view,t*WIDTH*4u,WIDTH*4u);REQUIRE(input_rows[t]);
        pre_rows[t]=myds_metal_buffer_sub_view(pre.view,t*4u*4u,4u*4u);REQUIRE(pre_rows[t]);
        float *p=myds_metal_buffer_data(pre_rows[t]);
        for(unsigned h=0;h<4u;++h)p[h]=(float)(h+1u+t)/16.0f;
    }
    float *nw=myds_metal_buffer_data(norm.view);
    for(unsigned i=0;i<DIM;++i)nw[i]=0.5f+(float)(random_bits(i,313u)%128u)/128.0f;
    for(unsigned arm=0;arm<2u;++arm){
        res[arm]=allocate(2u*RES_FLOATS*4u,32u);sum[arm]=allocate(2u*DIM*4u,48u);out[arm]=allocate(2u*DIM*4u,64u);
        for(unsigned t=0;t<2u;++t){
            res_rows[arm][t]=myds_metal_buffer_sub_view(res[arm].view,t*RES_FLOATS*4u,RES_FLOATS*4u);
            sum_rows[arm][t]=myds_metal_buffer_sub_view(sum[arm].view,t*DIM*4u,DIM*4u);
            out_rows[arm][t]=myds_metal_buffer_sub_view(out[arm].view,t*DIM*4u,DIM*4u);
            REQUIRE(res_rows[arm][t]&&sum_rows[arm][t]&&out_rows[arm][t]);
        }
    }
    uint64_t input_hash=fnv(myds_metal_buffer_data(inputs.view),inputs.bytes);
    uint64_t norm_hash=fnv(myds_metal_buffer_data(norm.view),norm.bytes),pre_hash=fnv(myds_metal_buffer_data(pre.view),pre.bytes);
    myds_v41_engram_project_round_counters_t before,after;myds_v41_engram_project_round_counters(&before,false);
    for(unsigned arm=0;arm<2u;++arm)for(unsigned repeat=0;repeat<(arm?8u:1u);++repeat){
        for(unsigned t=0;t<2u;++t){
            fill_inputs(f,337u+19u*t,0);
            memcpy(myds_metal_buffer_data(res_rows[arm][t]),myds_metal_buffer_data(f->initial.view),RES_FLOATS*4u);
        }
        memset(myds_metal_buffer_data(sum[arm].view),0xa5,sum[arm].bytes);
        memset(myds_metal_buffer_data(out[arm].view),0xa5,out[arm].bytes);
        if(arm)REQUIRE(myds_metal_begin_batch());
        for(unsigned t=0;t<2u;++t){
            if(!arm)REQUIRE(myds_metal_begin_batch());
            REQUIRE(myds_v41_engram_project_round_probe(policies[1],res_rows[arm][t],f->kv[arm].view,
                f->weight.view,input_rows[t],f->q.view,f->k.view,1e-6f,false,false,false));
            if(!arm){
                REQUIRE(myds_metal_v41_hc_sum_rmsnorm(out_rows[arm][t],sum_rows[arm][t],res_rows[arm][t],pre_rows[t],norm.view,DIM,1e-6f,true));
                verify_encoder();REQUIRE(myds_metal_commit_batch());
            }
        }
        if(arm){
            for(unsigned t=0;t<2u;++t)REQUIRE(myds_metal_v41_hc_sum_rmsnorm(out_rows[arm][t],sum_rows[arm][t],res_rows[arm][t],pre_rows[t],norm.view,DIM,1e-6f,true));
            verify_encoder();REQUIRE(myds_metal_commit_batch());
            REQUIRE(!memcmp(myds_metal_buffer_data(res[0].view),myds_metal_buffer_data(res[1].view),res[0].bytes));
            REQUIRE(!memcmp(myds_metal_buffer_data(sum[0].view),myds_metal_buffer_data(sum[1].view),sum[0].bytes));
            REQUIRE(!memcmp(myds_metal_buffer_data(out[0].view),myds_metal_buffer_data(out[1].view),out[0].bytes));
        }
        finite(res[arm]);finite(sum[arm]);finite(out[arm]);
    }
    myds_v41_engram_project_round_counters(&after,false);
    REQUIRE(!memcmp(&before,&after,sizeof(before)));
    REQUIRE(input_hash==fnv(myds_metal_buffer_data(inputs.view),inputs.bytes));
    REQUIRE(norm_hash==fnv(myds_metal_buffer_data(norm.view),norm.bytes)&&pre_hash==fnv(myds_metal_buffer_data(pre.view),pre.bytes));
    printf("{\"kind\":\"batch2_hc_dependency\",\"case\":%u,\"concurrent\":%d,\"auto_barrier\":%d,\"tokens\":2,\"paired_repeats\":8,\"scalar_oracle_separate_cbs\":true,\"paired_same_cb\":true,\"shared_projected_kv\":true,\"residual_subviews_shared_owner\":true,\"helper_consumer_join\":true,\"caller_barriers\":false,\"observation_copies\":false,\"round_skip_requests\":0,\"full_f32_residual_bit_exact\":true,\"hc_sum_bit_exact\":true,\"hc_norm_bit_exact\":true,\"inputs_unchanged\":true,\"canaries\":true}\n",case_id,myds_metal_get_concurrent(),myds_metal_get_auto_barrier());
    for(unsigned t=0;t<2u;++t){myds_metal_buffer_free(input_rows[t]);myds_metal_buffer_free(pre_rows[t]);}
    for(unsigned arm=0;arm<2u;++arm){
        for(unsigned t=0;t<2u;++t){myds_metal_buffer_free(res_rows[arm][t]);myds_metal_buffer_free(sum_rows[arm][t]);myds_metal_buffer_free(out_rows[arm][t]);}
        release(res[arm]);release(sum[arm]);release(out[arm]);
    }
    release(inputs);release(pre);release(norm);
}

static void rejected(struct fixture *f,const char *name,myds_metal_buffer_t *residual,myds_metal_buffer_t *kv,
                     const myds_metal_buffer_t *weight,const myds_metal_buffer_t *input,
                     const myds_metal_buffer_t *q,const myds_metal_buffer_t *k,float eps,int partial) {
    uint8_t *saved_res=malloc(f->res[1].bytes),*saved_kv=malloc(f->kv[1].bytes);REQUIRE(saved_res&&saved_kv);
    memcpy(saved_res,myds_metal_buffer_data(f->res[1].view),f->res[1].bytes);memcpy(saved_kv,myds_metal_buffer_data(f->kv[1].view),f->kv[1].bytes);
    myds_v41_engram_project_round_counters_t a,b;myds_v41_engram_project_round_counters(&a,false);uint64_t old=myds_metal_dispatch_count();
    REQUIRE(myds_metal_begin_batch());
    if(!partial){REQUIRE(!myds_v41_engram_project_round_probe(policies[0],residual,kv,weight,input,q,k,eps,true,true,false));REQUIRE(myds_metal_dispatch_count()==old);}
    REQUIRE(!myds_v41_engram_project_round_probe(policies[1],residual,kv,weight,input,q,k,eps,true,true,false));REQUIRE(myds_metal_commit_batch());
    myds_v41_engram_project_round_counters(&b,false);REQUIRE(b.requests-a.requests==1u&&b.encoded==a.encoded&&b.failures-a.failures==1u);
    REQUIRE(myds_metal_dispatch_count()-old==(unsigned)partial);REQUIRE(!memcmp(saved_res,myds_metal_buffer_data(f->res[1].view),f->res[1].bytes));
    if(!partial)REQUIRE(!memcmp(saved_kv,myds_metal_buffer_data(f->kv[1].view),f->kv[1].bytes));
    printf("{\"kind\":\"rejected_call\",\"reason\":\"%s\",\"requests\":1,\"encoded\":0,\"failures\":1,\"dispatches\":%d,\"residual_untouched\":true,\"projection_may_be_written\":%s,\"reference_prewrite_rejection\":%s,\"no_fallback\":true}\n",name,partial,partial?"true":"false",partial?"false":"true");free(saved_res);free(saved_kv);
}
static void rejections(struct fixture *f) {
    reset_arm(f,1);
#define REJECT(name,r,v,w,x,q,k,e) rejected(f,name,r,v,w,x,q,k,e,0)
    REJECT("null_output",NULL,f->kv[1].view,f->weight.view,f->input.view,f->q.view,f->k.view,1e-6f);
    REJECT("invalid_eps",f->res[1].view,f->kv[1].view,f->weight.view,f->input.view,f->q.view,f->k.view,NAN);
    myds_metal_buffer_t *short_res=myds_metal_buffer_sub_view(f->res[1].view,0,f->res[1].bytes-4u);REQUIRE(short_res);
    REJECT("short_residual",short_res,f->kv[1].view,f->weight.view,f->input.view,f->q.view,f->k.view,1e-6f);myds_metal_buffer_free(short_res);
    myds_metal_buffer_t *short_weight=myds_metal_buffer_sub_view(f->weight.view,0,f->weight.bytes-2u);REQUIRE(short_weight);
    REJECT("short_projection_weight",f->res[1].view,f->kv[1].view,short_weight,f->input.view,f->q.view,f->k.view,1e-6f);myds_metal_buffer_free(short_weight);
    REJECT("projected_residual_alias",f->kv[1].view,f->kv[1].view,f->weight.view,f->input.view,f->q.view,f->k.view,1e-6f);
    myds_metal_buffer_t *overlap=myds_metal_buffer_sub_view(f->kv[1].view,16u,f->res[1].bytes);REQUIRE(overlap);
    REJECT("projected_residual_overlap",overlap,f->kv[1].view,f->weight.view,f->input.view,f->q.view,f->k.view,1e-6f);myds_metal_buffer_free(overlap);
    REJECT("projected_input_alias",f->res[1].view,f->kv[1].view,f->weight.view,f->kv[1].view,f->q.view,f->k.view,1e-6f);
    myds_metal_buffer_t *unaligned=myds_metal_buffer_sub_view(f->input.owner,f->input.offset+4u,f->input.bytes);REQUIRE(unaligned);
    REJECT("unaligned_input",f->res[1].view,f->kv[1].view,f->weight.view,unaligned,f->q.view,f->k.view,1e-6f);myds_metal_buffer_free(unaligned);
    unaligned=myds_metal_buffer_sub_view(f->kv[1].owner,f->kv[1].offset+4u,f->kv[1].bytes);REQUIRE(unaligned);
    REJECT("unaligned_projected",f->res[1].view,unaligned,f->weight.view,f->input.view,f->q.view,f->k.view,1e-6f);myds_metal_buffer_free(unaligned);
#undef REJECT
    id<MTLComputePipelineState> saved=g_pso_v41_engram_add;g_pso_v41_engram_add=nil;
    rejected(f,"missing_consumer_pso",f->res[1].view,f->kv[1].view,f->weight.view,f->input.view,f->q.view,f->k.view,1e-6f,1);g_pso_v41_engram_add=saved;
    canary(f->weight);canary(f->input);canary(f->kv[1]);canary(f->res[1]);
}

static void timing(struct fixture *f,unsigned reps) {
    const char *arms[]={"A1","B","A2"};
    for(unsigned arm=0;arm<3u;++arm)for(unsigned sample=0;sample<=samples;++sample){
        fill_inputs(f,1001u+sample,0);reset_arm(f,0);reset_arm(f,1);
        run(policies[0],f,0,reps,true,true,false);
        myds_v41_engram_project_round_counters_t a,b;myds_v41_engram_project_round_counters(&a,false);uint64_t old=myds_metal_dispatch_count();
        id<MTLCommandBuffer> cb=run(policies[arm==1u],f,1,reps,true,true,false);
        myds_v41_engram_project_round_counters(&b,false);uint64_t expected=(uint64_t)(arm==1u)*reps;
        REQUIRE(b.requests-a.requests==expected&&b.encoded-a.encoded==expected&&b.failures==a.failures);
        REQUIRE(myds_metal_dispatch_count()-old==(uint64_t)(arm==1u?2u:3u)*reps);
        REQUIRE(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime);
        char prefix[96];snprintf(prefix,sizeof(prefix),"timing-r%u-%s-s%u",reps,arms[arm],sample);export_pair(prefix,f,1);
        size_t diff=check(f->kv[0],f->kv[1],f->res[0],f->res[1]);REQUIRE(arm==1u?diff>0:diff==0);
        printf("{\"kind\":\"gpu_timing\",\"arm\":\"%s\",\"sample\":%u,\"warmup\":%s,\"input_seed\":%u,\"repetitions\":%u,\"dim\":5120,\"input_width\":6144,\"weight_bytes\":%zu,\"actual_round_skip\":%u,\"requests\":%llu,\"encoded\":%llu,\"failures\":0,\"dispatches\":%llu,\"gpu_start_s\":%.9f,\"gpu_end_s\":%.9f,\"us_per_pair\":%.9f,\"input_fnv1a64\":\"%016" PRIx64 "\",\"initial_residual_fnv1a64\":\"%016" PRIx64 "\",\"reference_kv_fnv1a64\":\"%016" PRIx64 "\",\"candidate_kv_fnv1a64\":\"%016" PRIx64 "\",\"residual_fnv1a64\":\"%016" PRIx64 "\",\"rounded_projection_bit_exact\":true,\"full_f32_residual_bit_exact\":true,\"finite\":true,\"nonpoison\":true,\"canaries\":true,\"completed_cb\":true,\"encoder_dispatch_type_verified\":true,\"cb_status\":%lu,\"helper_consumer_join\":true,\"dispatch_hist_disabled\":true,\"fixture_not_fullmodel\":true,\"cold_dram_not_proven\":true}\n",
            arms[arm],sample,sample==0?"true":"false",1001u+sample,reps,f->weight.bytes,arm==1u,(unsigned long long)expected,(unsigned long long)expected,(unsigned long long)(myds_metal_dispatch_count()-old),cb.GPUStartTime,cb.GPUEndTime,(cb.GPUEndTime-cb.GPUStartTime)*1e6/reps,
            fnv(myds_metal_buffer_data(f->input.view),f->input.bytes),fnv(myds_metal_buffer_data(f->initial.view),f->initial.bytes),fnv(myds_metal_buffer_data(f->kv[0].view),f->kv[0].bytes),fnv(myds_metal_buffer_data(f->kv[1].view),f->kv[1].bytes),fnv(myds_metal_buffer_data(f->res[1].view),f->res[1].bytes),(unsigned long)cb.status);
    }
}
static void host_config(void) {
    REQUIRE(!g_device&&!g_initialized);myds_v41_engram_project_round_counters(NULL,true);
    unsetenv("MYDS_ENGRAM_PROJECT_ROUND_SKIP");myds_v41_context_t *zero=myds_v41_context_create_for_model(&blank_model);REQUIRE(zero&&myds_v41_engram_project_round_skip_enabled(zero)==0);
    setenv("MYDS_ENGRAM_PROJECT_ROUND_SKIP","1",1);myds_v41_context_t *one=myds_v41_context_create_for_model(&blank_model);REQUIRE(one&&myds_v41_engram_project_round_skip_enabled(one)==1&&myds_v41_engram_project_round_skip_enabled(zero)==0);
    setenv("MYDS_ENGRAM_PROJECT_ROUND_SKIP","0",1);REQUIRE(myds_v41_engram_project_round_skip_enabled(one)==1);
    const char *invalid[]={"","2","3","01","00"," 1","1 ","+1","-1","true","0x1","1\n"};
    for(unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);++i){setenv("MYDS_ENGRAM_PROJECT_ROUND_SKIP",invalid[i],1);REQUIRE(!myds_v41_context_create_for_model(&blank_model));}
    REQUIRE(!myds_v41_engram_project_round_probe(one,NULL,NULL,NULL,NULL,NULL,NULL,1e-6f,true,true,false));
    myds_v41_engram_project_round_counters_t counts;myds_v41_engram_project_round_counters(&counts,true);REQUIRE(counts.requests==1u&&counts.encoded==0&&counts.failures==1u);
    REQUIRE(!myds_v41_engram_project_round_probe(zero,NULL,NULL,NULL,NULL,NULL,NULL,1e-6f,true,true,false));
    REQUIRE(!myds_v41_engram_project_round_probe(one,NULL,NULL,NULL,NULL,NULL,NULL,1e-6f,false,true,false));
    myds_v41_engram_project_round_counters(&counts,false);REQUIRE(!counts.requests&&!counts.encoded&&!counts.failures);
    for(uint32_t high=0;high<65536u;++high){uint32_t canonical=high<<16u;REQUIRE(rounded(canonical)==canonical);}
    const uint32_t lows[]={0u,1u,0x7fffu,0x8000u,0x8001u,0xffffu};
    for(uint32_t high=0;high<65536u;++high)for(unsigned i=0;i<sizeof(lows)/sizeof(*lows);++i){uint32_t bits=(high<<16u)|lows[i];REQUIRE(rounded(rounded(bits))==rounded(bits));}
    myds_v41_context_free(zero);myds_v41_context_free(one);unsetenv("MYDS_ENGRAM_PROJECT_ROUND_SKIP");REQUIRE(!g_device&&!g_initialized);
    puts("{\"kind\":\"host_config\",\"pass\":true,\"absent_default\":0,\"valid_values\":[0,1],\"invalid_values_rejected\":12,\"context_policy_immutable\":true,\"canonical_patterns\":65536,\"idempotence_inputs\":393216,\"counter_reset\":true,\"no_gpu_initialization\":true}");
}
static unsigned number(const char *value) {char *end=NULL;unsigned long n=strtoul(value,&end,10);REQUIRE(*value&&end&&!*end&&n>=1u&&n<=10000u);return(unsigned)n;}
int main(int argc,char **argv) {
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader_dir=argv[++i];
        else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)output_dir=argv[++i];
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=number(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)repetitions=number(argv[++i]);
        else if(!strcmp(argv[i],"--host-only"))host_only=1;
        else if(!strcmp(argv[i],"--no-timing"))no_timing=1;
        else {fprintf(stderr,"usage: %s --shader-dir DIR --output-dir NEWDIR --samples 5 --repetitions 32 [--host-only|--no-timing]\n",argv[0]);return 2;}
    }
    host_config();if(host_only)return 0;
    if(output_dir){struct stat state;REQUIRE(lstat(output_dir,&state)!=0&&errno==ENOENT);REQUIRE(!mkdir(output_dir,0700));}
    for(unsigned i=0;i<2u;++i){setenv("MYDS_ENGRAM_PROJECT_ROUND_SKIP",i?"1":"0",1);policies[i]=myds_v41_context_create_for_model(&blank_model);REQUIRE(policies[i]&&myds_v41_engram_project_round_skip_enabled(policies[i])==(int)i);}
    /* Runtime's actual encoder choice ORs the setter with this environment
     * override. Clear it before init so the four setter modes are genuine. */
    unsetenv("MYDS_METAL_CONCURRENT");
    REQUIRE(myds_metal_init(shader_dir));struct fixture f=create(0);myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);reset_arm(&f,0);run(policies[0],&f,0,1u,true,true,false);
    printf("{\"kind\":\"identity\",\"device\":\"%s\",\"dim\":5120,\"input_width\":6144,\"weight_bytes\":%zu,\"projection_kernel\":\"kernel_mul_mv_f16_f32_4\",\"projection_nsg\":8,\"projection_threads\":256,\"projection_groups\":12800,\"projection_dynamic_tg_bytes\":256,\"round_kernel\":\"kernel_dsv41_bf16_linear\",\"consumer_kernel\":\"kernel_dsv41_engram_add\",\"actual_context_modes\":[0,1],\"context_policy_immutable\":true,\"setup_completed_cb\":true,\"encoder_dispatch_type_verified\":true,\"ambient_concurrent_env_cleared\":true,\"fixture_not_fullmodel\":true}\n",g_device.name.UTF8String,f.weight.bytes);
    myds_v41_engram_project_round_counters(NULL,true);unsigned case_id=0;
    for(unsigned c=0;c<2u;++c)for(unsigned a=0;a<2u;++a){myds_metal_set_concurrent(c);myds_metal_set_auto_barrier(a);fill_inputs(&f,31u,0);numeric(&f,case_id++,"seeded_dense",true,true,false);reuse(&f,2u*c+a);batch2_hc_dependency(&f,2u*c+a);}
    myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);
    numeric(&f,8u,"nonordinary_reference",false,true,false);numeric(&f,9u,"no_logits_reference",true,false,false);numeric(&f,10u,"trace_reference",true,true,true);rejections(&f);
    g_hist_enabled=0;if(!no_timing){timing(&f,1u);if(repetitions!=1u)timing(&f,repetitions);}destroy(&f);
    f=create(1);case_id=4u;for(unsigned c=0;c<2u;++c)for(unsigned a=0;a<2u;++a){myds_metal_set_concurrent(c);myds_metal_set_auto_barrier(a);fill_inputs(&f,31u,1);numeric(&f,case_id++,"sparse_bf16_boundary",true,true,false);}destroy(&f);
    for(unsigned i=0;i<2u;++i)myds_v41_context_free(policies[i]);myds_metal_cleanup();
    printf("{\"kind\":\"suite_complete\",\"pass\":true,\"parity_cases\":11,\"same_batch_reuse_modes\":4,\"reuse_calls_per_mode\":7,\"batch2_hc_dependency_modes\":4,\"batch2_hc_paired_repeats\":8,\"rejected_calls\":10,\"samples_per_arm\":%u,\"timing_enabled\":%s,\"timing_repetitions\":[1,%u],\"production_setter_added\":false}\n",samples,no_timing?"false":"true",repetitions);return 0;
}
