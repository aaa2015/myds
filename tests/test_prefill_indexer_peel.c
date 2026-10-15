/* Real prefill host helper versus its unchanged scalar indexer. This fixture
 * includes the production source so row peeling and its fallback contract are
 * exercised directly, with small synthetic GPU buffers and no model weights.
 * Build/run explicitly: make bin/test_prefill_indexer_peel && bin/test_prefill_indexer_peel
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/myds_v41.c"

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); \
} } while (0)

enum { CACHE_ROWS = 1024, MASK_STRIDE = 256 };
static unsigned cases_checked;
static uint64_t scalar_dispatches, batched_dispatches;

struct peel_fixture {
    myds_context_t ctx;
    myds_model_t model;
    myds_v41_model_t model_v41;
    myds_v41_context_t state;
    myds_metal_buffer_t *q, *ref_q, *weights, *selected, *ref_selected, *mask;
    myds_metal_buffer_t *q_view, *w_view, *sel_view, *mask_view;
    uint32_t tokens, prompt_rows, cache_rows, layer;
};

static myds_metal_buffer_t *allocate(size_t bytes) {
    myds_metal_buffer_t *b = myds_metal_buffer_alloc(bytes);
    REQUIRE(b);
    memset(myds_metal_buffer_data(b),0xa5,bytes);
    return b;
}

static void create_fixture(struct peel_fixture *f, uint32_t tokens, uint32_t c_start,
                           uint32_t cache_rows, bool ties) {
    memset(f,0,sizeof(*f));
    f->ctx.model=&f->model;f->ctx.v41=&f->state;f->model.v41=&f->model_v41;
    f->tokens=tokens;f->prompt_rows=c_start+tokens+2u;f->cache_rows=cache_rows;f->layer=2u;
    f->state.gpu=myds_v41_gpu_scratch_create(CACHE_ROWS);REQUIRE(f->state.gpu);
    f->state.index_cache[0]=allocate((size_t)cache_rows*128u*sizeof(uint16_t));
    /* All three ratio-2 owners may read this immutable fixture cache. */
    f->state.index_cache[1]=f->state.index_cache[0];
    f->state.index_cache[2]=f->state.index_cache[0];
    f->state.index_scores=allocate((size_t)CACHE_ROWS*sizeof(float));
    f->q=allocate((size_t)tokens*V41_INDEX_HEADS*128u*sizeof(float));
    f->ref_q=allocate(myds_metal_buffer_size(f->q));
    f->weights=allocate((size_t)tokens*V41_INDEX_HEADS*sizeof(float));
    f->selected=allocate((size_t)f->prompt_rows*V41_TOPK*sizeof(int32_t));
    f->ref_selected=allocate(myds_metal_buffer_size(f->selected));
    f->mask=allocate((size_t)f->prompt_rows*MASK_STRIDE*sizeof(float));
    f->q_view=myds_metal_buffer_create_view();f->w_view=myds_metal_buffer_create_view();
    f->sel_view=myds_metal_buffer_create_view();f->mask_view=myds_metal_buffer_create_view();
    REQUIRE(f->q_view&&f->w_view&&f->sel_view&&f->mask_view);
    float *q=myds_metal_buffer_data(f->q),*w=myds_metal_buffer_data(f->weights);
    uint16_t *cache=myds_metal_buffer_data(f->state.index_cache[0]);
    for (size_t i=0;i<myds_metal_buffer_size(f->q)/sizeof(float);i++)
        q[i]=(float)((int)((i*19u+7u)%127u)-63)/64.0f;
    memcpy(myds_metal_buffer_data(f->ref_q),q,myds_metal_buffer_size(f->q));
    for (size_t i=0;i<(size_t)tokens*V41_INDEX_HEADS;i++)
        w[i]=ties?0.0f:(float)((i*11u+3u)%37u)/32.0f;
    for (size_t i=0;i<(size_t)cache_rows*128u;i++) {
        float x=(float)((int)((i*7u+13u)%53u)-26)/32.0f;
        uint32_t bits;memcpy(&bits,&x,sizeof(bits));cache[i]=(uint16_t)(bits>>16);
    }
}

static void free_fixture(struct peel_fixture *f) {
    myds_metal_buffer_t *buffers[]={f->q,f->ref_q,f->weights,f->selected,f->ref_selected,
        f->mask,f->q_view,f->w_view,f->sel_view,f->mask_view,
        f->state.index_cache[0],f->state.index_scores};
    for (size_t i=0;i<sizeof(buffers)/sizeof(*buffers);i++)myds_metal_buffer_free(buffers[i]);
    myds_v41_gpu_scratch_free(f->state.gpu);
}

static int run_batched(struct peel_fixture *f,uint32_t base_pos,uint32_t c_start) {
    return publish_indexer_chunk(&f->ctx,f->layer,c_start,f->tokens,base_pos,f->q,f->weights,
        f->selected,V41_TOPK*sizeof(int32_t),f->mask,MASK_STRIDE*sizeof(float),
        f->q_view,f->w_view,f->sel_view,f->mask_view);
}

static void run_scalar(struct peel_fixture *f,uint32_t base_pos,uint32_t c_start) {
    for (uint32_t t=0;t<f->tokens;t++) {
        REQUIRE(myds_metal_buffer_update_sub_view(f->q_view,f->ref_q,
            (size_t)t*V41_INDEX_HEADS*128u*sizeof(float),V41_INDEX_HEADS*128u*sizeof(float)));
        REQUIRE(myds_metal_buffer_update_sub_view(f->w_view,f->weights,
            (size_t)t*V41_INDEX_HEADS*sizeof(float),V41_INDEX_HEADS*sizeof(float)));
        REQUIRE(myds_metal_buffer_update_sub_view(f->sel_view,f->ref_selected,
            (size_t)(c_start+t)*V41_TOPK*sizeof(int32_t),V41_TOPK*sizeof(int32_t)));
        REQUIRE(publish_indexer_one(&f->ctx,f->layer,base_pos+c_start+t,NULL,NULL,
            f->q_view,f->w_view,f->sel_view,NULL,NULL));
    }
}

static void assert_bytes(const myds_metal_buffer_t *b,unsigned char value) {
    const unsigned char *p=myds_metal_buffer_data((myds_metal_buffer_t *)b);
    for (size_t i=0;i<myds_metal_buffer_size(b);i++)REQUIRE(p[i]==value);
}

static void compare_case(uint32_t tokens,uint32_t base_pos,uint32_t c_start,bool ties,
                         uint32_t layer,const char *peel_mode) {
    struct peel_fixture f;create_fixture(&f,tokens,c_start,CACHE_ROWS,ties);
    f.layer=layer;
    const size_t q_bytes=myds_metal_buffer_size(f.q);
    unsigned char *q_before=malloc(q_bytes);REQUIRE(q_before);
    memcpy(q_before,myds_metal_buffer_data(f.q),q_bytes);
    if (peel_mode) REQUIRE(setenv("MYDS_PREFILL_INDEXER_PEEL",peel_mode,1)==0);
    else REQUIRE(unsetenv("MYDS_PREFILL_INDEXER_PEEL")==0);
    REQUIRE(myds_metal_begin_batch());
    uint64_t before=myds_metal_dispatch_count();
    REQUIRE(run_batched(&f,base_pos,c_start)==1);
    batched_dispatches+=myds_metal_dispatch_count()-before;
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(myds_metal_begin_batch());before=myds_metal_dispatch_count();
    run_scalar(&f,base_pos,c_start);
    scalar_dispatches+=myds_metal_dispatch_count()-before;
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(!memcmp(myds_metal_buffer_data(f.q),myds_metal_buffer_data(f.ref_q),q_bytes));
    const int32_t *a=myds_metal_buffer_data(f.selected),*b=myds_metal_buffer_data(f.ref_selected);
    if (memcmp(a,b,myds_metal_buffer_size(f.selected))) {
        for(size_t i=0;i<myds_metal_buffer_size(f.selected)/sizeof(int32_t);i++)
            if(a[i]!=b[i]){fprintf(stderr,"selected mismatch tokens=%u base=%u start=%u ties=%d row=%zu col=%zu batch=%d scalar=%d\n",
                tokens,base_pos,c_start,ties,i/V41_TOPK,i%V41_TOPK,a[i],b[i]);break;}
        REQUIRE(0);
    }
    assert_bytes(f.mask,0xa5);
    if (!base_pos&&!c_start) {
        REQUIRE(!memcmp(myds_metal_buffer_data(f.q),q_before,V41_INDEX_HEADS*128u*sizeof(float)));
        for(uint32_t i=0;i<V41_TOPK;i++)REQUIRE((uint32_t)a[i]==UINT32_C(0xa5a5a5a5));
    }
    free(q_before);free_fixture(&f);cases_checked++;
}

static void fallback_case(uint32_t tokens,uint32_t cache_rows,uint32_t base_pos,
                          const char *mode,bool short_q) {
    struct peel_fixture f;create_fixture(&f,tokens,0u,cache_rows,false);
    if (mode) REQUIRE(setenv("MYDS_PREFILL_INDEXER_PEEL",mode,1)==0);
    else REQUIRE(unsetenv("MYDS_PREFILL_INDEXER_PEEL")==0);
    if (short_q) {
        myds_metal_buffer_free(f.q);
        f.q=allocate(V41_INDEX_HEADS*128u*sizeof(float));
    }
    const size_t bytes=myds_metal_buffer_size(f.q);
    unsigned char *before_q=malloc(bytes);REQUIRE(before_q);
    memcpy(before_q,myds_metal_buffer_data(f.q),bytes);
    REQUIRE(myds_metal_begin_batch());uint64_t before=myds_metal_dispatch_count();
    REQUIRE(run_batched(&f,base_pos,0u)==0);
    REQUIRE(myds_metal_dispatch_count()==before);
    REQUIRE(myds_metal_commit_batch());
    REQUIRE(!memcmp(before_q,myds_metal_buffer_data(f.q),bytes));
    assert_bytes(f.selected,0xa5);assert_bytes(f.mask,0xa5);
    free(before_q);free_fixture(&f);cases_checked++;
}

int main(void) {
    REQUIRE(setenv("MYDS_INDEXER_BATCH_FLOATS","65536",1)==0);
    REQUIRE(unsetenv("MYDS_INDEXER_TILED")==0);
    REQUIRE(myds_metal_init("metal"));
    const uint32_t tails[]={1u,2u,3u,9u,31u,32u,33u,65u,255u,256u,511u,512u};
    for(unsigned default_mode=0;default_mode<2;default_mode++) {
        const char *peel_mode=default_mode?NULL:"1";
        for(unsigned exact=0;exact<2;exact++) {
            REQUIRE(setenv("MYDS_INDEXER_CHUNK_EXACT",exact?"1":"0",1)==0);
            for (size_t i=0;i<sizeof(tails)/sizeof(*tails);i++)
                for(unsigned ties=0;ties<2;ties++)compare_case(tails[i],0u,0u,ties!=0u,2u,peel_mode);
            compare_case(31u,64u,0u,false,8u,peel_mode);compare_case(33u,0u,6u,true,14u,peel_mode);
            compare_case(65u,1u,0u,false,8u,peel_mode);compare_case(9u,0u,1u,true,14u,peel_mode);
            compare_case(65u,0u,0u,false,8u,peel_mode);compare_case(65u,0u,0u,true,14u,peel_mode);
        }
    }
    fallback_case(33u,CACHE_ROWS,0u,"invalid",false);
    fallback_case(33u,CACHE_ROWS,0u,"0",false);
    fallback_case(33u,127u,0u,"1",false);
    fallback_case(33u,CACHE_ROWS,0u,"1",true);
    fallback_case(2u,CACHE_ROWS,131074u,"1",false); /* width exceeds score budget */
    myds_metal_cleanup();
    printf("[PASS] prefill indexer peel: %u cases, q/selected byte parity, row-0 canaries, zero-dispatch fallbacks; scalar=%llu batch=%llu dispatches\n",
        cases_checked,(unsigned long long)scalar_dispatches,(unsigned long long)batched_dispatches);
    return 0;
}
