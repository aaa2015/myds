/* CPU-only actual publisher failure propagation, no Metal device/library. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "../src/myds_v41.c"
#include <assert.h>
#pragma clang diagnostic ignored "-Wunused-parameter"
struct myds_metal_buffer {size_t size;};
static struct myds_metal_buffer scores={2097152u*4u},blocks={2097152u*4u},sel={2097152u*4u};
static unsigned rope_calls,scoped_calls,after_scorer_calls,fallback_rope_calls;
static bool trace_on;
void myds_log(myds_log_level_t level,const char *fmt,...) {}
bool myds_trace_active(void){return trace_on;}
size_t myds_metal_buffer_size(const myds_metal_buffer_t *b){return b?b->size:0;}
int myds_metal_buffer_update_sub_view(myds_metal_buffer_t*v,const myds_metal_buffer_t*p,size_t off,size_t n){assert(p&&off<=p->size&&n<=p->size-off);v->size=n;return 1;}
uint32_t myds_v41_gpu_indexer_batch_floats(const myds_v41_gpu_scratch_t*s){return 2097152;}
myds_metal_buffer_t *myds_v41_gpu_indexer_scores(const myds_v41_gpu_scratch_t*s){return &scores;}
myds_metal_buffer_t *myds_v41_gpu_indexer_blocks(const myds_v41_gpu_scratch_t*s){return &blocks;}
myds_metal_buffer_t *myds_v41_gpu_indexer_block_selected(const myds_v41_gpu_scratch_t*s){return &sel;}
int myds_v41_gpu_indexer_rope_quant_batch(myds_metal_buffer_t*x,myds_metal_buffer_t*d,size_t off,uint32_t h,uint32_t start,uint32_t rows,uint32_t stride,bool compressed){assert(h==32&&start==30720&&rows==2048&&stride==1&&compressed);rope_calls++;return 1;}
int myds_v41_gpu_rope_batch(myds_metal_buffer_t*x,uint32_t d,uint32_t h,uint32_t start,uint32_t rows,bool c,bool inv){fallback_rope_calls++;return 1;}
int myds_v41_gpu_quant(myds_metal_buffer_t*x,uint32_t width,uint32_t rows,uint32_t mode){fallback_rope_calls++;return 1;}
int myds_v41_gpu_indexscores_register_only_prefill_chunk_scoped(myds_v41_gpu_scratch_t*s,myds_metal_buffer_t*y,const myds_metal_buffer_t*q,const myds_metal_buffer_t*w,const myds_metal_buffer_t*k,uint32_t nc,uint32_t nh,uint32_t nt,uint32_t start,uint32_t ratio,bool bf,bool ordinary){assert(rope_calls==1&&nc==15424&&nt==128&&start==30720&&ratio==2&&nh==32&&bf&&ordinary==!trace_on);scoped_calls++;return 0;}
int myds_v41_gpu_candidates_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g){after_scorer_calls++;return 1;}
int myds_v41_gpu_filter_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g){after_scorer_calls++;return 1;}
int myds_v41_gpu_filter_rows_bits(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g){after_scorer_calls++;return 1;}
int myds_v41_gpu_topk_causal(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f,uint32_t g,myds_v41_gpu_scratch_t*s){after_scorer_calls++;return 1;}
int myds_v41_gpu_maskbits_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f){after_scorer_calls++;return 1;}
int myds_v41_gpu_topkmask_rows(myds_metal_buffer_t*a,const myds_metal_buffer_t*b,uint32_t c,uint32_t d,uint32_t e,uint32_t f){after_scorer_calls++;return 1;}
int main(void) {
    myds_context_t ctx={0};myds_v41_context_t g={0};ctx.v41=&g;int opaque;g.gpu=(void*)&opaque;
    struct myds_metal_buffer cache={327680u*256u},q={2048u*16384u},w={2048u*128u},selected={2048u*2048u},mask={2048u*32768u};
    struct myds_metal_buffer qv={0},wv={0},sv={0},mv={0};g.index_cache[0]=&cache;
    for(unsigned t=0;t<2;t++) {
        trace_on=t!=0;rope_calls=scoped_calls=after_scorer_calls=fallback_rope_calls=0;
        int result=publish_indexer_chunk(&ctx,2,0,2048,30720,&q,&w,&selected,2048,&mask,32768,false,&qv,&wv,&sv,&mv);
        assert(result==-1&&rope_calls==1&&scoped_calls==1&&!after_scorer_calls&&!fallback_rope_calls);
    }
    puts("{\"actual_publisher_scorer_failure_is_minus1\":true,\"trace_scope_forwarded\":true,\"no_repeat_rope_or_consumers\":true,\"gpu_work\":false}");return 0;
}
