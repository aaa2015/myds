/* CPU-only actual publisher failure propagation, no Metal device/library. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "../src/myds_v41.c"
#include <assert.h>
#pragma clang diagnostic ignored "-Wunused-parameter"
struct myds_metal_buffer {size_t size;};
static struct myds_metal_buffer scores={8388608u*4u},blocks={8388608u*4u},sel={8388608u*4u};
static uint32_t stub_batch_floats=2097152u;
static int scorer_ret=0;
static uint32_t expected_start=30720u,expected_nc=15424u,expected_ratio=2u,expected_rows=2048u;
static unsigned rope_calls,scoped_calls,after_scorer_calls,fallback_rope_calls;
static uint32_t sum_scoped_nt,min_scoped_nt,max_scoped_nt;
static bool trace_on;
void myds_log(myds_log_level_t level,const char *fmt,...) {}
bool myds_trace_active(void){return trace_on;}
size_t myds_metal_buffer_size(const myds_metal_buffer_t *b){return b?b->size:0;}
int myds_metal_buffer_update_sub_view(myds_metal_buffer_t*v,const myds_metal_buffer_t*p,size_t off,size_t n){assert(p&&off<=p->size&&n<=p->size-off);v->size=n;return 1;}
uint32_t myds_v41_gpu_indexer_batch_floats(const myds_v41_gpu_scratch_t*s){return stub_batch_floats;}
myds_metal_buffer_t *myds_v41_gpu_indexer_scores(const myds_v41_gpu_scratch_t*s){return &scores;}
myds_metal_buffer_t *myds_v41_gpu_indexer_blocks(const myds_v41_gpu_scratch_t*s){return &blocks;}
myds_metal_buffer_t *myds_v41_gpu_indexer_block_selected(const myds_v41_gpu_scratch_t*s){return &sel;}
int myds_v41_gpu_indexer_rope_quant_batch(myds_metal_buffer_t*x,myds_metal_buffer_t*d,size_t off,uint32_t h,uint32_t start,uint32_t rows,uint32_t stride,bool compressed){assert(h==32&&start==expected_start&&rows==expected_rows&&stride==1&&compressed);rope_calls++;return 1;}
int myds_v41_gpu_rope_batch(myds_metal_buffer_t*x,uint32_t d,uint32_t h,uint32_t start,uint32_t rows,bool c,bool inv){fallback_rope_calls++;return 1;}
int myds_v41_gpu_quant(myds_metal_buffer_t*x,uint32_t width,uint32_t rows,uint32_t mode){fallback_rope_calls++;return 1;}
int myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(myds_v41_gpu_scratch_t*s,myds_metal_buffer_t*y,const myds_metal_buffer_t*q,const myds_metal_buffer_t*w,const myds_metal_buffer_t*k,uint32_t nc,uint32_t nh,uint32_t nt,uint32_t start,uint32_t ratio,bool bf,bool ordinary){
    const uint32_t exp_nc=(start+nt)/expected_ratio;
    assert(rope_calls==1&&nc==(exp_nc<512u?512u:exp_nc)&&ratio==expected_ratio&&nh==32&&bf&&ordinary==!trace_on);
    assert(start==expected_start+sum_scoped_nt);
    assert(nt>=8u&&(uint64_t)nc*nt<=stub_batch_floats);
    scoped_calls++;sum_scoped_nt+=nt;
    if(nt<min_scoped_nt)min_scoped_nt=nt;
    if(nt>max_scoped_nt)max_scoped_nt=nt;
    return scorer_ret;
}
int myds_v41_gpu_candidates_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g){after_scorer_calls++;return 1;}
int myds_v41_gpu_filter_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g){after_scorer_calls++;return 1;}
int myds_v41_gpu_filter_rows_bits(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g){after_scorer_calls++;return 1;}
int myds_v41_gpu_topk_causal(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g,myds_v41_gpu_scratch_t*s){after_scorer_calls++;return 1;}
int myds_v41_gpu_maskbits_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f){after_scorer_calls++;return 1;}
int myds_v41_gpu_topkmask_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f){after_scorer_calls++;return 1;}
static int compact_ret=0;
static unsigned compact_calls;
static uint32_t sum_compact_nt,min_compact_nt,max_compact_nt;
uint32_t myds_v41_gpu_indexer_prefill_compact_max_glen(const myds_v41_gpu_scratch_t*s){return stub_batch_floats>=131072u?((stub_batch_floats/16384u>512u)?512u:((stub_batch_floats/16384u)&~7u)):0u;}
int myds_v41_gpu_indexer_prefill_compact_scoped(myds_v41_gpu_scratch_t*s,myds_metal_buffer_t*ids,const myds_metal_buffer_t*q,const myds_metal_buffer_t*w,const myds_metal_buffer_t*k,const myds_metal_buffer_t*m,uint32_t nc,uint32_t nt,uint32_t start,uint32_t ratio,uint32_t mu,uint32_t tb,bool ord){
    assert(rope_calls==1&&ratio==1u&&tb==2048u&&start>=16384u&&nt>=8u&&nc==start+nt&&ord==!trace_on);
    compact_calls++;
    if(compact_ret>0){
        sum_compact_nt+=nt;
        if(nt<min_compact_nt)min_compact_nt=nt;
        if(nt>max_compact_nt)max_compact_nt=nt;
    }
    return compact_ret;
}
int main(void) {
    myds_context_t ctx={0};myds_v41_context_t g={0};ctx.v41=&g;int opaque;g.gpu=(void*)&opaque;
    struct myds_metal_buffer cache={1048576u*256u},q={2048u*16384u},w={2048u*128u},selected={2048u*2048u},mask={2048u*524288u};
    struct myds_metal_buffer qv={0},wv={0},sv={0},mv={0};
    for(unsigned i=0;i<8;i++)g.index_cache[i]=&cache;
    for(unsigned t=0;t<2;t++) {
        trace_on=t!=0;rope_calls=scoped_calls=after_scorer_calls=fallback_rope_calls=0;
        sum_scoped_nt=0;min_scoped_nt=UINT32_MAX;max_scoped_nt=0;
        stub_batch_floats=2097152u;scorer_ret=0;expected_start=30720u;expected_nc=15424u;expected_ratio=2u;
        int result=publish_indexer_chunk(&ctx,2,0,2048,30720,&q,&w,&selected,2048,&mask,32768,false,&qv,&wv,&sv,&mv);
        assert(result==-1&&rope_calls==1&&scoped_calls==1&&sum_scoped_nt==128u&&!after_scorer_calls&&!fallback_rope_calls);
    }
    /* [P-I1] Verify indexer_group_len + publish_indexer_chunk across 512K < nc <= 1M with batch_floats=8388608:
     * every subgroup of a 2048-token chunk must keep nt >= 8 so register-only never silently falls back to TILED. */
    stub_batch_floats=8388608u;trace_on=false;scorer_ret=1;
    for(uint32_t nc=524289u;nc<=1048576u;nc=(nc+1024u<=1048576u?nc+1024u:(nc<1048576u?1048576u:nc+1u))) {
        uint32_t g0=0;
        while(g0<2048u) {
            uint32_t glen=indexer_group_len(2048u,g0,nc-2048u,0u,1u,stub_batch_floats);
            assert(glen>=8u&&(uint64_t)glen*nc<=stub_batch_floats&&g0+glen<=2048u);
            g0+=glen;
        }
        assert(g0==2048u);
        if(nc==1048576u)break;
    }
    const uint32_t base_positions[]={522241u,524288u,580000u,630000u,750000u,920000u,1046528u};
    for(unsigned i=0;i<sizeof(base_positions)/sizeof(base_positions[0]);i++) {
        rope_calls=scoped_calls=after_scorer_calls=fallback_rope_calls=0;
        sum_scoped_nt=0;min_scoped_nt=UINT32_MAX;max_scoped_nt=0;
        expected_start=base_positions[i];expected_nc=base_positions[i]+2048u;expected_ratio=1u;
        int result=publish_indexer_chunk(&ctx,20,0,2048,base_positions[i],&q,&w,&selected,2048,&mask,524288,false,&qv,&wv,&sv,&mv);
        assert(result==1&&rope_calls==1&&sum_scoped_nt==2048u&&min_scoped_nt>=8u&&!fallback_rope_calls);
    }
    /* [P-L1] Verify il>20 mask_bits=true compact sub-batching (4x512 for 2048, 504+11 for 515),
     * graceful fallback (compact_ret=0 -> dense once), and fail-closed (compact_ret=-1). */
    rope_calls=scoped_calls=after_scorer_calls=fallback_rope_calls=compact_calls=0;
    sum_compact_nt=0;min_compact_nt=UINT32_MAX;max_compact_nt=0;compact_ret=1;
    expected_start=129024u;expected_rows=2048u;
    assert(publish_indexer_chunk(&ctx,24,0,2048,129024,&q,&w,&selected,2048,&mask,65536,true,&qv,&wv,&sv,&mv)==1);
    assert(rope_calls==1&&compact_calls==4&&sum_compact_nt==2048u&&min_compact_nt==512u&&max_compact_nt==512u&&!scoped_calls&&!after_scorer_calls);

    /* Remainder balancing: 515 -> 504 + 11 (never leaves <8 tail) */
    rope_calls=scoped_calls=after_scorer_calls=compact_calls=0;
    sum_compact_nt=0;min_compact_nt=UINT32_MAX;max_compact_nt=0;compact_ret=1;expected_rows=515u;
    assert(publish_indexer_chunk(&ctx,24,0,515,129024,&q,&w,&selected,2048,&mask,65536,true,&qv,&wv,&sv,&mv)==1);
    assert(rope_calls==1&&compact_calls==2&&sum_compact_nt==515u&&min_compact_nt==11u&&max_compact_nt==504u&&!scoped_calls&&!after_scorer_calls);
    expected_rows=2048u;

    rope_calls=scoped_calls=after_scorer_calls=compact_calls=0;sum_scoped_nt=0;
    expected_start=129024u;expected_ratio=1u;compact_ret=0;scorer_ret=1;
    assert(publish_indexer_chunk(&ctx,24,0,2048,129024,&q,&w,&selected,2048,&mask,65536,true,&qv,&wv,&sv,&mv)==1);
    assert(rope_calls==1&&compact_calls==1&&sum_scoped_nt==2048u&&after_scorer_calls==scoped_calls*2u);

    rope_calls=scoped_calls=after_scorer_calls=compact_calls=0;compact_ret=-1;
    assert(publish_indexer_chunk(&ctx,24,0,2048,129024,&q,&w,&selected,2048,&mask,65536,true,&qv,&wv,&sv,&mv)==-1);
    assert(rope_calls==1&&compact_calls==1&&!scoped_calls&&!after_scorer_calls);
    puts("{\"actual_publisher_scorer_failure_is_minus1\":true,\"trace_scope_forwarded\":true,\"no_repeat_rope_or_consumers\":true,\"p_i1_gt_512k_no_fallback\":true,\"p_l1_publisher_verified\":true,\"gpu_work\":false}");return 0;
}
