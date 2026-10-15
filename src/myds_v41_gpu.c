#include "myds_v41_gpu.h"
#include "myds_log.h"
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* Chunked prefill geometry.  The slab holds, in order, the carry of the previous
 * 127 KV vectors, this chunk's tokens, and the compressed table prefix; the mask
 * block holds one uint16 row per token. */
#define MYDS_CHUNK_CARRY      127u
#define MYDS_CHUNK_MAXC       MYDS_V41_CHUNK_MAX
/* Compressed-key coverage is independent of the allocated attention workspace.
 * Gather uses only carry + current chunk; a dense slab is allocated on demand. */
#define MYDS_CHUNK_KEY_MAX_DEFAULT 262144u
#define MYDS_CHUNK_KEY_MAX_LIMIT  4194304u


struct myds_v41_gpu_scratch {
    uint32_t max_comp;
    myds_metal_buffer_t *sort_a, *sort_b, *kv_f16, *mask, *pad, *partial;
    myds_metal_buffer_t *topk_prefix_nan; /* lazy row flags, optional path only */
    uint32_t topk_prefix_rows;
    myds_metal_buffer_t *topk_nan_partitions; /* independent fused-path flags */
    uint64_t topk_nan_partition_slots;
    /* Independent Top512 candidate; fixed maxima avoid reallocating in-flight
     * scratch on same-command-buffer reuse. Allocated only on explicit use. */
    myds_metal_buffer_t *threshold_parts, *threshold_control, *threshold_hist;
    myds_metal_buffer_t *threshold_offsets, *threshold_pairs;
    myds_metal_buffer_t *sv_view, *dv_view;
    /* 批量 indexer（prefill Step 4）的临时缓冲：分数表、top-k 的中间排序结果两份、
     * 块分数、块选择结果。每个都是 batch_floats 个 float（或等
     * 字节数的 int），由调用方按“组内 token 数 × 本组最大压缩键数 ≤ 预算”切组。
     * 放在 scratch 里（而不是每次 prefill 分配）只是为了所有权清楚、不引入新的失败点。 */
    uint32_t batch_floats;
    myds_metal_buffer_t *chunk_scores, *chunk_tmp_a, *chunk_tmp_b, *chunk_blocks, *chunk_block_sel;
    /* Chunked prefill works on one f16 slab and one mask row block for a whole
     * chunk, because a per-token staging and a per-token attention dispatch are
     * what dominated prefill.  The carry keeps the previous chunk's last 127 KV
     * vectors so a query's 128 wide window is contiguous. */
    myds_metal_buffer_t *kv_chunk, *mask_chunk;
    uint32_t tmpl_len, tmpl_carry;   /* which window mask template mask_chunk holds */
    /* chunk attention 的三个运行时几何量，由【配置的上下文】推出（见 scratch_create）。
     * 以前是编译期常量 ⇒ 「配置的上下文」与「实际能跑的上下文」会静默分叉。 */
    uint32_t key_comp_max;   /* n_comp 的上界；超了必须在 prefill 开始前就拒绝 */
    uint32_t mask_u16;       /* 掩码行宽（u16），32 对齐 ⇒ keys_pad 落在行内 */
    uint32_t slab_rows;      /* allocated KV rows, rounded to a 32-key block */
    uint32_t keys_clamp;     /* 诊断：MYDS_ATTN_KEYS_CLAMP 缓存 */
    uint32_t gather_min_keys; /* explicit opt-in policy; zero preserves default */
    uint32_t prefill_attn_variant; /* 0=vec, 1=matrix, 2=auto */
    uint32_t attn_matrix_unavailable; /* auto caches failure for this scratch */
    uint32_t attn_vec_head_first; /* gather-only prefill vec geometry, default on */
    uint32_t attn_vec_head_first_unavailable; /* optional PSO failed for this scratch */
    myds_metal_buffer_t *attn_head_blocks; /* matrix tile mask block flags, lazy */
    bool direct_window_prestaged, direct_scratch_fenced;
    bool direct_populate_kv_cache, direct_expect_kv_reuse;
    bool direct_kv_cache_valid, direct_pad_cache_valid;
    bool direct_deferred_q_compressed;
    myds_metal_buffer_t *direct_deferred_q;
    const myds_metal_buffer_t *direct_deferred_qb_weight, *direct_deferred_qb_input;
    uint32_t direct_deferred_q_pos, direct_deferred_qb_done, direct_deferred_q_rope_done;
    const myds_metal_buffer_t *direct_kv_comp_buf, *direct_kv_ids_buf;
    size_t direct_kv_comp_offset, direct_kv_ids_offset;
    uint32_t direct_kv_n_raw, direct_kv_source_rows, direct_kv_selected_rows, direct_kv_pos;
};
static uint32_t attn_matrix_test_failure;
void myds_v41_gpu_inject_attention_matrix_failure(uint32_t kind) {
    attn_matrix_test_failure=kind;
}
static uint32_t attn_vec_head_first_test_failure;
void myds_v41_gpu_inject_attention_vec_head_first_failure(bool enabled) {
    attn_vec_head_first_test_failure=enabled;
}
uint32_t myds_v41_gpu_attention_vec_head_first_status(const myds_v41_gpu_scratch_t *s) {
    return !s||!s->attn_vec_head_first ? 0u : s->attn_vec_head_first_unavailable ? 2u : 1u;
}
/* Process-wide prefill attention path counters. Host-only relaxed increments at
 * points that already exist on the encode path; no GPU work or ordering change.
 * The order matches myds_v41_gpu_attention_counters_t field order. */
enum { AC_CALLS, AC_GATHER, AC_SLAB, AC_HF_ATTEMPTS, AC_HF_ENCODED, AC_HF_FAILURES,
       AC_HF_CACHED, AC_ORIGINAL, AC_MATRIX, AC_MATRIX_FALLBACKS, AC_COUNT };
static _Atomic uint64_t attn_counters[AC_COUNT];
static inline void attn_count(unsigned k) {
    atomic_fetch_add_explicit(&attn_counters[k], 1u, memory_order_relaxed);
}
void myds_v41_gpu_attention_counters(myds_v41_gpu_attention_counters_t *out, bool reset) {
    uint64_t v[AC_COUNT];
    for (unsigned k = 0; k < AC_COUNT; ++k)
        v[k] = reset ? atomic_exchange_explicit(&attn_counters[k], 0u, memory_order_relaxed)
                     : atomic_load_explicit(&attn_counters[k], memory_order_relaxed);
    if (!out) return;
    *out = (myds_v41_gpu_attention_counters_t){
        .calls = v[AC_CALLS], .gather_calls = v[AC_GATHER], .slab_calls = v[AC_SLAB],
        .head_first_attempts = v[AC_HF_ATTEMPTS], .head_first_encoded = v[AC_HF_ENCODED],
        .head_first_failures = v[AC_HF_FAILURES], .head_first_cached_fallbacks = v[AC_HF_CACHED],
        .original_vec_encoded = v[AC_ORIGINAL], .matrix_encoded = v[AC_MATRIX],
        .matrix_fallbacks = v[AC_MATRIX_FALLBACKS]};
}
static int floats(const myds_metal_buffer_t *b,uint64_t n) {
    return b && n && n<=myds_metal_buffer_size(b)/4 && myds_metal_buffer_offset(b)%4==0;
}
static int bf16s(const myds_metal_buffer_t *b,uint64_t n) {
    return b && n && n<=myds_metal_buffer_size(b)/2 && myds_metal_buffer_offset(b)%2==0;
}
static int dispatch(const char *name,const void *args,size_t bytes,
    const myds_metal_buffer_t *const *b,size_t n,uint32_t x,uint32_t y,uint32_t z,
    uint32_t tx,uint32_t ty,uint32_t tz,size_t smem) {
    const uint32_t groups[]={x,y,z},threads[]={tx,ty,tz};
    return myds_metal_dispatch_kernel(name,args,bytes,b,n,groups,threads,smem);
}
static int specialized(const char *name,const void *args,size_t bytes,
    const myds_metal_buffer_t *const *b,size_t n,uint32_t x,uint32_t y,uint32_t z,
    uint32_t tx,uint32_t ty,uint32_t tz,size_t smem,
    const myds_metal_constant_t *constants,size_t nc) {
    const uint32_t groups[]={x,y,z},threads[]={tx,ty,tz};
    return myds_metal_dispatch_kernel_constants(name,args,bytes,b,n,groups,threads,smem,constants,nc);
}
int myds_v41_gpu_pack_bf16(myds_metal_buffer_t *dst,size_t dst_offset_words,
    const myds_metal_buffer_t *src,size_t src_offset_floats,uint32_t count) {
    if(!bf16s(dst,count)||!floats(src,count)||
       dst_offset_words>myds_metal_buffer_size(dst)/2-count||
       src_offset_floats>myds_metal_buffer_size(src)/4-count)return 0;
    struct {uint64_t dst_offset,src_offset;uint32_t count,pad;} a={
        dst_offset_words,src_offset_floats,count,0};
    const myds_metal_buffer_t *b[]={src,dst};
    return dispatch("kernel_dsv41_pack_bf16",&a,sizeof(a),b,2,
                    (count-1u)/256u+1u,1,1,256,1,1,0);
}
uint32_t myds_v41_gpu_chunk_max_c(const myds_v41_gpu_scratch_t *s) {
    return s ? MYDS_CHUNK_MAXC : 0u;
}
uint32_t myds_v41_gpu_chunk_key_max(const myds_v41_gpu_scratch_t *s) {
    return s ? s->key_comp_max : 0u;
}
size_t myds_v41_gpu_chunk_workspace_bytes(const myds_v41_gpu_scratch_t *s) {
    return s ? myds_metal_buffer_size(s->kv_chunk)+myds_metal_buffer_size(s->mask_chunk) : 0;
}
myds_v41_gpu_scratch_t *myds_v41_gpu_scratch_create(uint32_t max_comp) {
    if (myds_v41_gpu_attention_packed_register_enabled()<0 || myds_v41_attention_rope_fuse_enabled()<0 ||
        myds_v41_gpu_attention_direct_pack_fuse_enabled()<0) return NULL;
    if(!max_comp || max_comp>1048576) return NULL;
    myds_v41_gpu_scratch_t *s=calloc(1,sizeof(*s));
    if(!s)return NULL;
    s->max_comp=max_comp;
    const char *kc = getenv("MYDS_ATTN_KEYS_CLAMP");
    if(kc) s->keys_clamp = (uint32_t)strtoul(kc, NULL, 10);
    /* Cover the configured context without eagerly allocating a dense mask. */
    {
        uint32_t key_max=MYDS_CHUNK_KEY_MAX_DEFAULT;
        const char *km=getenv("MYDS_CHUNK_KEY_MAX");
        if(km){ long v=atol(km); if(v>0&&(unsigned long)v<=MYDS_CHUNK_KEY_MAX_LIMIT) key_max=(uint32_t)v; }
        if(key_max>max_comp) key_max=max_comp;   /* 不必超过配置的上下文 */
        s->key_comp_max=key_max;
        if(s->key_comp_max<2u||s->key_comp_max>MYDS_CHUNK_KEY_MAX_LIMIT){ free(s); return NULL; }
        s->slab_rows=(MYDS_CHUNK_CARRY+MYDS_CHUNK_MAXC+31u)/32u*32u;
        s->mask_u16=s->slab_rows;
    }
    const char *gm=getenv("MYDS_ATTN_GATHER_MIN_KEYS");
    if(gm){
        char *end=NULL;errno=0;
        unsigned long long v=strtoull(gm,&end,10);
        if(errno||end==gm||*end||v>s->key_comp_max){
            MYDS_LOG_E("MYDS_ATTN_GATHER_MIN_KEYS must be an integer in [0,%u]",s->key_comp_max);
            free(s);return NULL;
        }
        s->gather_min_keys=(uint32_t)v;
    }
    const char *av=getenv("MYDS_PREFILL_ATTN");
    if(av && strcmp(av,"vec")){
        if(!strcmp(av,"matrix"))s->prefill_attn_variant=1;
        else if(!strcmp(av,"auto"))s->prefill_attn_variant=2;
        else{MYDS_LOG_E("MYDS_PREFILL_ATTN must be vec, matrix, or auto");free(s);return NULL;}
    }
    /* Capture the policy once per scratch. Both default and explicit enablement
     * are best effort: an older complete metallib falls back to the original vec. */
    const char *hf=getenv("MYDS_ATTN_VEC_HEAD_FIRST");
    s->attn_vec_head_first=!hf||strcmp(hf,"0")!=0;
    s->sort_a=myds_metal_buffer_alloc((size_t)max_comp*4);
    s->sort_b=myds_metal_buffer_alloc((size_t)max_comp*4);
    s->kv_f16=myds_metal_buffer_alloc(640u*512u*2u);
    s->mask=myds_metal_buffer_alloc(640u*2u);
    s->pad=myds_metal_buffer_alloc(2u*32u*512u*2u+32u*2u);
    s->partial=myds_metal_buffer_alloc(64u*(512u+2u)*32u*4u);
    s->sv_view=myds_metal_buffer_create_view();
    s->dv_view=myds_metal_buffer_create_view();
    s->kv_chunk=myds_metal_buffer_alloc((size_t)s->slab_rows*512u*2u);
    s->mask_chunk=myds_metal_buffer_alloc((size_t)MYDS_CHUNK_MAXC*s->mask_u16*2u);
    uint32_t batch_floats = MYDS_V41_INDEXER_BATCH_FLOATS;
    const char *ib_env = getenv("MYDS_INDEXER_BATCH_FLOATS");
    if (ib_env) {
        long v = atol(ib_env);
        if (v >= 65536 && v <= 16777216) batch_floats = (uint32_t)v;
    } else if (max_comp > 32768u) {
        uint64_t target = (uint64_t)max_comp * 8ull;
        if (target > 2097152ull) target = 2097152ull;
        if (target > (uint64_t)batch_floats) batch_floats = (uint32_t)target;
    }
    s->batch_floats = batch_floats;
    s->chunk_scores=myds_metal_buffer_alloc((size_t)s->batch_floats*4u);
    s->chunk_tmp_a=myds_metal_buffer_alloc((size_t)s->batch_floats*4u);
    s->chunk_tmp_b=myds_metal_buffer_alloc((size_t)s->batch_floats*4u);
    s->chunk_blocks=myds_metal_buffer_alloc((size_t)s->batch_floats*4u);
    s->chunk_block_sel=myds_metal_buffer_alloc((size_t)s->batch_floats*4u);
    if(!s->sort_a||!s->sort_b||!s->kv_f16||!s->mask||!s->pad||!s->partial||!s->sv_view||!s->dv_view||
       !s->kv_chunk||!s->mask_chunk||!s->chunk_scores||!s->chunk_tmp_a||!s->chunk_tmp_b||
       !s->chunk_blocks||!s->chunk_block_sel){myds_v41_gpu_scratch_free(s);return NULL;}
    memset(myds_metal_buffer_data(s->mask),0,640u*2u);
    /* Zero the slab once: rows that are never written by a chunk must read as 0,
     * not as leftover data, because masked entries still compute a score. */
    memset(myds_metal_buffer_data(s->kv_chunk),0,(size_t)s->slab_rows*512u*2u);
    memset(myds_metal_buffer_data(s->mask_chunk),0xff,(size_t)MYDS_CHUNK_MAXC*s->mask_u16*2u);
    return s;
}
void myds_v41_gpu_scratch_free(myds_v41_gpu_scratch_t *s) {
    if(!s)return;
    myds_metal_buffer_free(s->sort_a);myds_metal_buffer_free(s->sort_b);
    myds_metal_buffer_free(s->topk_prefix_nan);
    myds_metal_buffer_free(s->topk_nan_partitions);
    myds_metal_buffer_free(s->threshold_parts);myds_metal_buffer_free(s->threshold_control);
    myds_metal_buffer_free(s->threshold_hist);myds_metal_buffer_free(s->threshold_offsets);
    myds_metal_buffer_free(s->threshold_pairs);
    myds_metal_buffer_free(s->kv_f16);myds_metal_buffer_free(s->mask);
    myds_metal_buffer_free(s->kv_chunk);myds_metal_buffer_free(s->mask_chunk);
    myds_metal_buffer_free(s->attn_head_blocks);
    myds_metal_buffer_free(s->pad);myds_metal_buffer_free(s->partial);
    myds_metal_buffer_free(s->chunk_scores);myds_metal_buffer_free(s->chunk_tmp_a);
    myds_metal_buffer_free(s->chunk_tmp_b);myds_metal_buffer_free(s->chunk_blocks);
    myds_metal_buffer_free(s->chunk_block_sel);
    myds_metal_buffer_free(s->sv_view);myds_metal_buffer_free(s->dv_view);free(s);
}
myds_metal_buffer_t *myds_v41_gpu_indexer_scores(const myds_v41_gpu_scratch_t *s) {
    return s?s->chunk_scores:NULL;
}
myds_metal_buffer_t *myds_v41_gpu_indexer_blocks(const myds_v41_gpu_scratch_t *s) {
    return s?s->chunk_blocks:NULL;
}
myds_metal_buffer_t *myds_v41_gpu_indexer_block_selected(const myds_v41_gpu_scratch_t *s) {
    return s?s->chunk_block_sel:NULL;
}
uint32_t myds_v41_gpu_indexer_batch_floats(const myds_v41_gpu_scratch_t *s) {
    return s ? s->batch_floats : MYDS_V41_INDEXER_BATCH_FLOATS;
}
uint32_t myds_v41_gpu_indexer_batch_rows(const myds_v41_gpu_scratch_t *s,uint32_t n_comp) {
    if(!s||!n_comp)return 0;
    uint32_t rows=(uint32_t)(s->batch_floats/n_comp);
    if(rows>MYDS_CHUNK_MAXC)rows=MYDS_CHUNK_MAXC;
    return rows;
}
static float rope_frequencies[2][32];
static pthread_once_t rope_once=PTHREAD_ONCE_INIT;
/* Match pinned ds4's powf/reciprocal and YaRN operation order. */
#pragma float_control(precise, on, push)
static void init_rope(void) {
    const double pi=3.14159265358979323846;
    for(int kind=0;kind<2;kind++) {
        float base=kind?160000.0f:10000.0f;
        float low=(float)floor(64.0*log(65536.0/(32.0*2.0*pi))/(2.0*log(base)));
        float high=(float)ceil(64.0*log(65536.0/(2.0*pi))/(2.0*log(base)));
        for(int i=0;i<32;i++) {
            float denominator=powf(base,(float)i/32.0f),f=1.0f/denominator;
            if(kind) {
                float ramp=fminf(1.0f,fmaxf(0.0f,(i-low)/(high-low)));
                float smooth=1.0f-ramp;
                float interpolate=(f/16.0f)*(1.0f-smooth),extrapolate=f*smooth;
                f=interpolate+extrapolate;
            }
            rope_frequencies[kind][i]=f;
        }
    }
}
#pragma float_control(pop)
int myds_v41_gpu_rope_batch_stride(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t start_pos,uint32_t rows,uint32_t stride,bool compressed,bool inverse) {
    if(width<64||!heads||!rows||!stride||start_pos>=1048576||
        (uint64_t)start_pos+(uint64_t)(rows-1u)*stride>=1048576||
        !floats(x,(uint64_t)width*heads*rows))return 0;
    pthread_once(&rope_once,init_rope);
    struct {uint32_t width,heads,rows,start,inverse,stride;float frequencies[32];} a={width,heads,rows,start_pos,inverse,stride,{0}};
    memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={x};
    return dispatch("kernel_dsv41_rope",&a,sizeof(a),b,1,heads,rows,1,32,1,1,0);
}
int myds_v41_gpu_rope_batch(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t start_pos,uint32_t rows,bool compressed,bool inverse) {
    return myds_v41_gpu_rope_batch_stride(x,width,heads,start_pos,rows,1u,compressed,inverse);
}
int myds_v41_gpu_rope(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool compressed,bool inverse) {
    return myds_v41_gpu_rope_batch(x,width,heads,pos,1,compressed,inverse);
}
int myds_v41_gpu_rope_heads_slice(myds_v41_gpu_scratch_t *s,
    myds_metal_buffer_t *x,uint32_t head_offset,uint32_t head_count,
    uint32_t pos,bool compressed) {
    if(!head_count)return 1;
    if(!s||!s->dv_view||!x||head_offset>=64u||head_count>64u-head_offset||
       pos>=1048576u||!floats(x,64u*512u))return 0;
    if(head_offset==0u&&head_count==64u)
        return myds_v41_gpu_rope(x,512u,64u,pos,compressed,false);
    const size_t byte_off=(size_t)head_offset*512u*sizeof(float);
    const size_t byte_len=(size_t)head_count*512u*sizeof(float);
    if(!myds_metal_buffer_update_sub_view(s->dv_view,x,byte_off,byte_len))return 0;
    pthread_once(&rope_once,init_rope);
    struct {uint32_t width,heads,rows,start,inverse,stride;float frequencies[32];} a={
        512u,head_count,1u,pos,0u,1u,{0}};
    memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={s->dv_view};
    return dispatch("kernel_dsv41_rope",&a,sizeof(a),b,1,head_count,1,1,32,1,1,0);
}
int myds_v41_gpu_rope_batch_round(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t start_pos,uint32_t rows,bool compressed,bool inverse) {
    if(width<64||!heads||!rows||start_pos>=1048576||
        (uint64_t)start_pos+(uint64_t)(rows-1u)>=1048576||
        !floats(x,(uint64_t)width*heads*rows))return 0;
    pthread_once(&rope_once,init_rope);
    struct {uint32_t width,heads,rows,start,inverse,stride;float frequencies[32];} a={width,heads,rows,start_pos,inverse,1u,{0}};
    memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={x};
    if(rows>1u&&(width&3u)==0u&&myds_metal_buffer_offset(x)%16u==0u&&
       dispatch("kernel_dsv41_rope_round_batch",&a,sizeof(a),b,1,heads,rows,1,32,1,1,0))
        return 1;
    return dispatch("kernel_dsv41_rope_round",&a,sizeof(a),b,1,heads,rows,1,32,1,1,0);
}
int myds_v41_gpu_rope_round(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool compressed,bool inverse) {
    return myds_v41_gpu_rope_batch_round(x,width,heads,pos,1,compressed,inverse);
}
int myds_dspark_gpu_rope(myds_metal_buffer_t *x,uint32_t width,uint32_t heads,
    uint32_t pos,bool inverse) {
    if(width<64u||!heads||pos>=1048576u||!floats(x,(uint64_t)width*heads))return 0;
    pthread_once(&rope_once,init_rope);
    struct {uint32_t width,heads,rows,start,inverse,stride;float frequencies[32];} a={
        width,heads,1u,pos,inverse,1u,{0}};
    memcpy(a.frequencies,rope_frequencies[0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={x};
    if(!myds_metal_get_auto_barrier())myds_metal_barrier_buffer(x);
    return dispatch("kernel_myds_dspark_rope_f32",&a,sizeof(a),b,1,heads,1,1,32,1,1,0);
}
int myds_dspark_gpu_quant_kv(myds_metal_buffer_t *x,uint32_t rows) {
    if(!rows||!floats(x,(uint64_t)rows*512u))return 0;
    const myds_metal_buffer_t *b[]={x};
    if(!myds_metal_get_auto_barrier())myds_metal_barrier_buffer(x);
    return dispatch("kernel_myds_dspark_kv_fp8_f32",&rows,sizeof(rows),b,1,7,rows,1,64,1,1,256u);
}
int myds_v41_gpu_quant(myds_metal_buffer_t *x,uint32_t width,uint32_t rows,uint32_t mode) {
    uint32_t block=mode==3?16:32;
    if(!width||!rows||mode>3||(mode&&width%block)||!floats(x,(uint64_t)width*rows))return 0;
    if(!mode && (uint64_t)width*rows<=UINT32_MAX && myds_metal_buffer_offset(x)%16==0)
        return myds_metal_v41_bf16(x,width*rows);
    const uint32_t a[]={width,rows,mode};const myds_metal_buffer_t *b[]={x};
    return dispatch("kernel_dsv41_quantize",a,sizeof(a),b,1,(width+block-1)/block,rows,1,32,1,1,0);
}
int myds_v41_gpu_indexer_rope_quant(myds_metal_buffer_t *x,uint32_t heads,uint32_t pos) {
    /* Reuse the exact once-initialized compressed frequency table; the
     * explicit native bridge owns complete preflight/fail-closed counters. */
    pthread_once(&rope_once,init_rope);
    return myds_metal_indexer_rope_quant_dispatch(x,heads,pos,rope_frequencies[1]);
}
int myds_v41_gpu_indexer_rope_quant_batch(myds_metal_buffer_t *x,
    myds_metal_buffer_t *packed_dst,size_t dst_offset_words,
    uint32_t heads,uint32_t start_pos,uint32_t rows,uint32_t stride,bool compressed) {
    if(!heads||!rows||!stride||start_pos>=1048576u||
       (uint64_t)start_pos+(uint64_t)(rows-1u)*stride>=1048576u||
       !floats(x,(uint64_t)128u*heads*rows))return 0;
    const uint64_t total_elems=(uint64_t)128u*heads*rows;
    if(packed_dst&&(!bf16s(packed_dst,total_elems)||
       dst_offset_words>myds_metal_buffer_size(packed_dst)/2u-total_elems))return 0;
    pthread_once(&rope_once,init_rope);
    struct {
        uint32_t heads,rows,start,stride,mode,pack_bf16;
        uint64_t dst_offset;
        float frequencies[32];
    } a={heads,rows,start_pos,stride,2u,packed_dst?1u:0u,(uint64_t)dst_offset_words,{0}};
    memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={x,packed_dst?packed_dst:x};
    return dispatch("kernel_dsv41_indexer_rope_quant_batch",&a,sizeof(a),b,2,heads,rows,1,32,1,1,0);
}
int myds_v41_gpu_kv_rope_quant_batch(myds_metal_buffer_t *x,
    myds_metal_buffer_t *packed_dst,size_t dst_offset_words,
    uint32_t start_pos,uint32_t rows,uint32_t stride,bool compressed,uint32_t mode) {
    if(!rows||!stride||(mode!=1u&&mode!=3u)||start_pos>=1048576u||
       (uint64_t)start_pos+(uint64_t)(rows-1u)*stride>=1048576u||
       !floats(x,(uint64_t)512u*rows))return 0;
    const uint64_t total_elems=(uint64_t)512u*rows;
    if(packed_dst&&(!bf16s(packed_dst,total_elems)||
       dst_offset_words>myds_metal_buffer_size(packed_dst)/2u-total_elems))return 0;
    pthread_once(&rope_once,init_rope);
    struct {
        uint32_t heads,rows,start,stride,mode,pack_bf16;
        uint64_t dst_offset;
        float frequencies[32];
    } a={1u,rows,start_pos,stride,mode,packed_dst?1u:0u,(uint64_t)dst_offset_words,{0}};
    memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={x,packed_dst?packed_dst:x};
    return dispatch("kernel_dsv41_kv_rope_quant_batch",&a,sizeof(a),b,2,1,rows,1,32,1,1,0);
}
static int buffer_ranges_overlap(const myds_metal_buffer_t *a,size_t a_bytes,
                                 const myds_metal_buffer_t *b,size_t b_bytes) {
    uintptr_t x=(uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)a),
              y=(uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)b);
    return x<=y?y-x<a_bytes:x-y<b_bytes;
}
int myds_v41_gpu_qkv_finish(myds_metal_buffer_t *q,myds_metal_buffer_t *kv,
    myds_metal_buffer_t *window,uint32_t pos,bool compressed) {
    if(pos>=1048576u||!floats(q,64u*512u)||!floats(kv,512u)||
       !floats(window,128u*512u))return 0;
    if(!myds_metal_v41_qkv_fusion_available()||
       buffer_ranges_overlap(q,64u*512u*4u,kv,512u*4u)||
       buffer_ranges_overlap(q,64u*512u*4u,window,128u*512u*4u)||
       buffer_ranges_overlap(kv,512u*4u,window,128u*512u*4u)) {
        if (!myds_v41_gpu_rope(q,512,64,pos,compressed,false)) return 0;
        /* The fallback also handles aliased Q/KV/window views. Preserve the
         * reference Q-before-KV order even inside a concurrent encoder. */
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(q);
        if (!myds_v41_gpu_rope(kv,512,1,pos,compressed,false)) return 0;
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(kv);
        if (!myds_v41_gpu_quant(kv,512,1,1)) return 0;
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(kv);
        return myds_metal_buffer_copy(window,(size_t)(pos%128u)*512u*4u,kv,0,512u*4u);
    }
    pthread_once(&rope_once,init_rope);
    struct {uint32_t pos,pad;float frequencies[32];} a={pos,0,{0}};
    memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));
    const myds_metal_buffer_t *b[]={q,kv,window};
    return dispatch("kernel_dsv41_qkv_finish",&a,sizeof(a),b,3,80,1,1,32,1,1,0);
}
/* 整 chunk 的 pool2：tail=0 ⇒ 第 p 对读 kv/scores 的第 2p、2p+1 行，
 * 与逐 token 版（previous_* 保存前一个偶数位）结果逐位相同：每个输出元素独立计算。 */
static int stage_f16(myds_metal_buffer_t *dst,size_t dstoff,const myds_metal_buffer_t *src,
    size_t srcoff,uint32_t count,myds_v41_gpu_scratch_t *s);   /* 定义在后面 */
/* 见头文件。位置 p ∈ [base_pos-carry, base_pos) → slab 行 (p + CARRY - base_pos)、
 * 窗槽位 (p % 128)；carry = min(127, base_pos)。 */
int myds_v41_gpu_carry_stage(myds_v41_gpu_scratch_t *s,const myds_metal_buffer_t *window,
    uint32_t base_pos) {
    if(!s||!window||!base_pos)return 0;
    const uint32_t carry=base_pos<MYDS_CHUNK_CARRY?base_pos:MYDS_CHUNK_CARRY;
    if(!carry)return 1;
    const uint32_t p0=base_pos-carry;
    const uint32_t s0=p0%128u;
    uint32_t run1=128u-s0; if(run1>carry)run1=carry;
    const uint32_t row0=MYDS_CHUNK_CARRY-carry;
    if(!stage_f16(s->kv_chunk,(size_t)row0*1024u,window,(size_t)s0*512u*4u,run1*512u,s))return 0;
    if(carry>run1&&!stage_f16(s->kv_chunk,(size_t)(row0+run1)*1024u,window,0,
                             (carry-run1)*512u,s))return 0;
    return 1;
}
int myds_v41_gpu_pool2_pairs(myds_metal_buffer_t *out,const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores,uint32_t width,uint32_t pairs) {
    if(!width||!pairs||!floats(out,(uint64_t)width*pairs)||
        !floats(kv,(uint64_t)width*2u*pairs)||!floats(scores,(uint64_t)width*2u*pairs))return 0;
    const uint32_t a[]={width,pairs,0u};
    const myds_metal_buffer_t *b[]={out,kv,scores,kv,kv};
    return dispatch("kernel_dsv41_pool2",a,sizeof(a),b,5,(width+255u)/256,pairs,1,256,1,1,0);
}
int myds_v41_gpu_pool2(myds_metal_buffer_t *out,const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores,myds_metal_buffer_t *previous_kv,
    myds_metal_buffer_t *previous_scores,uint32_t width,uint32_t pos) {
    if(!width||pos==UINT32_MAX||!floats(kv,width)||!floats(scores,width)||
        !floats(previous_kv,width)||!floats(previous_scores,width))return 0;
    if(!(pos&1))return myds_metal_buffer_copy(previous_kv,0,kv,0,(size_t)width*4)&&
        myds_metal_buffer_copy(previous_scores,0,scores,0,(size_t)width*4);
    if(!floats(out,width))return 0;
    const uint32_t a[]={width,1,1};const myds_metal_buffer_t *b[]={out,kv,scores,previous_kv,previous_scores};
    return dispatch("kernel_dsv41_pool2",a,sizeof(a),b,5,(width+255u)/256,1,1,256,1,1,0);
}
static _Atomic uint64_t indexer_head_slots_requests;
static _Atomic uint64_t indexer_head_slots_encoded;
static _Atomic uint64_t indexer_head_slots_unavailable;
static _Atomic uint64_t indexer_head_slots_failures;
int myds_v41_gpu_indexer_head_slots_enabled(void) {
    return myds_metal_indexer_head_slots_enabled();
}
void myds_v41_gpu_indexer_head_slots_counters(
    myds_v41_gpu_indexer_head_slots_counters_t *out,bool reset) {
#define HEAD_SLOTS_COUNT(field) (reset ? atomic_exchange_explicit(&indexer_head_slots_##field,0u,memory_order_relaxed) \
                                     : atomic_load_explicit(&indexer_head_slots_##field,memory_order_relaxed))
    const myds_v41_gpu_indexer_head_slots_counters_t value = {
        .requests=HEAD_SLOTS_COUNT(requests), .encoded=HEAD_SLOTS_COUNT(encoded),
        .unavailable=HEAD_SLOTS_COUNT(unavailable), .failures=HEAD_SLOTS_COUNT(failures)};
    if(out)*out=value;
#undef HEAD_SLOTS_COUNT
}
int myds_v41_gpu_indexscores_head_slots(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t n_heads,bool cache_bf16) {
    atomic_fetch_add_explicit(&indexer_head_slots_requests,1u,memory_order_relaxed);
    bool encoded=false,unavailable=false;
    const int ok=myds_metal_indexer_head_slots_dispatch(scores,q,weights,cache,mask,
        n_comp,n_heads,cache_bf16,&encoded,&unavailable);
    if(encoded)atomic_fetch_add_explicit(&indexer_head_slots_encoded,1u,memory_order_relaxed);
    if(unavailable)atomic_fetch_add_explicit(&indexer_head_slots_unavailable,1u,memory_order_relaxed);
    if(!ok)atomic_fetch_add_explicit(&indexer_head_slots_failures,1u,memory_order_relaxed);
    return ok;
}
static int indexscores(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *mask,uint32_t n_comp,uint32_t n_heads,bool cache_bf16) {
    if(!n_heads||!floats(scores,n_comp)||!floats(q,(uint64_t)n_heads*128)||
        !floats(weights,n_heads)||!(cache_bf16?bf16s(cache,(uint64_t)n_comp*128):
        floats(cache,(uint64_t)n_comp*128)))return 0;
    if(mask&&!floats(mask,(n_comp+7u)/8u))return 0;
    struct {uint32_t n_rows,n_head,head_dim,cache_f16;float scale;uint32_t has_mask;} a={
        n_comp,n_heads,128,cache_bf16?2u:0u,1.0f/64.0f,mask?1u:0u};
    const myds_metal_buffer_t *b[]={q,weights,cache,scores,mask?mask:scores};
    return n_heads==32 ? dispatch("kernel_glm_indexer_score_one_direct",&a,sizeof(a),b,5,n_comp,1,1,32,4,1,132u*4)
        : dispatch("kernel_glm_indexer_score_one",&a,sizeof(a),b,5,n_comp,1,1,32,1,1,32u*4);
}
int myds_v41_gpu_indexscores(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,uint32_t n_heads) {
    return indexscores(scores,q,weights,cache,NULL,n_comp,n_heads,false);
}
int myds_v41_gpu_indexscores_bf16(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,uint32_t n_heads) {
    return indexscores(scores,q,weights,cache,NULL,n_comp,n_heads,true);
}
int myds_v41_gpu_indexscores_masked(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *mask,uint32_t n_comp,uint32_t n_heads) {
    return indexscores(scores,q,weights,cache,mask,n_comp,n_heads,false);
}
int myds_v41_gpu_indexscores_masked_bf16(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *mask,uint32_t n_comp,uint32_t n_heads) {
    return indexscores(scores,q,weights,cache,mask,n_comp,n_heads,true);
}

enum { IC_REQUESTS, IC_ENCODED, IC_UNAVAILABLE, IC_DENSE_ROWS, IC_ROW_SLOTS, IC_COUNT };
static _Atomic uint64_t indexer_compact_counters[IC_COUNT];
static bool indexer_compact_test_unavailable;
void myds_v41_gpu_inject_indexer_compact_unavailable(bool enabled) {
    indexer_compact_test_unavailable=enabled;
}
void myds_v41_gpu_indexer_compact_counters(myds_v41_gpu_indexer_compact_counters_t *out,bool reset) {
    uint64_t v[IC_COUNT];
    for(unsigned i=0;i<IC_COUNT;i++)v[i]=reset?
        atomic_exchange_explicit(&indexer_compact_counters[i],0,memory_order_relaxed):
        atomic_load_explicit(&indexer_compact_counters[i],memory_order_relaxed);
    if(out)*out=(myds_v41_gpu_indexer_compact_counters_t){v[0],v[1],v[2],v[3],v[4]};
}
int myds_v41_gpu_indexscores_selected_blocks_bf16(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache,const myds_metal_buffer_t *mask,
    const myds_metal_buffer_t *block_ids,uint32_t n_comp,uint32_t n_selected_blocks) {
    if(!n_comp||n_comp>1048576u||n_selected_blocks>(n_comp+7u)/8u||
       !floats(scores,n_comp)||!floats(q,32u*128u)||!floats(weights,32u)||
       !bf16s(cache,(uint64_t)n_comp*128u)||!floats(mask,(n_comp+7u)/8u)||
       (n_selected_blocks&&!floats(block_ids,n_selected_blocks)))return -1;
    const size_t score_bytes=(size_t)n_comp*4u;
    if(buffer_ranges_overlap(scores,score_bytes,q,32u*128u*4u)||
       buffer_ranges_overlap(scores,score_bytes,weights,32u*4u)||
       buffer_ranges_overlap(scores,score_bytes,cache,(size_t)n_comp*128u*2u)||
       buffer_ranges_overlap(scores,score_bytes,mask,(size_t)((n_comp+7u)/8u)*4u)||
       (n_selected_blocks&&buffer_ranges_overlap(scores,score_bytes,block_ids,(size_t)n_selected_blocks*4u)))return -1;
    atomic_fetch_add_explicit(&indexer_compact_counters[IC_REQUESTS],1,memory_order_relaxed);
    /* Probe both optional pipelines before filling any score. Missing kernels
     * can use the original scorer without observing partly changed buffers. */
    if(indexer_compact_test_unavailable||
       myds_metal_kernel_max_threads("kernel_myds_indexer_scores_fill")<256u||
       (n_selected_blocks&&myds_metal_kernel_max_threads("kernel_myds_indexer_score_selected_blocks_direct")<128u)) {
        atomic_fetch_add_explicit(&indexer_compact_counters[IC_UNAVAILABLE],1,memory_order_relaxed);
        return 0;
    }
    const myds_metal_buffer_t *inputs[]={q,weights,cache,mask,block_ids};
    myds_metal_barrier_buffers(inputs,5);
    const myds_metal_buffer_t *fill_b[]={scores};
    if(!dispatch("kernel_myds_indexer_scores_fill",&n_comp,sizeof(n_comp),fill_b,1,
                 (n_comp+255u)/256u,1,1,256,1,1,0))return -1;
    myds_metal_barrier_buffer(scores);
    if(n_selected_blocks) {
        struct {uint32_t n_rows,n_head,head_dim,cache_f16;float scale;
            uint32_t has_mask,n_selected_blocks,pad;} a={n_comp,32u,128u,2u,1.0f/64.0f,1u,n_selected_blocks,0};
        const myds_metal_buffer_t *b[]={q,weights,cache,scores,mask,block_ids};
        if(!dispatch("kernel_myds_indexer_score_selected_blocks_direct",&a,sizeof(a),b,6,
                     n_selected_blocks*8u,1,1,32,4,1,132u*4))return -1;
    }
    myds_metal_barrier_buffer(scores);
    atomic_fetch_add_explicit(&indexer_compact_counters[IC_ENCODED],1,memory_order_relaxed);
    atomic_fetch_add_explicit(&indexer_compact_counters[IC_DENSE_ROWS],n_comp,memory_order_relaxed);
    atomic_fetch_add_explicit(&indexer_compact_counters[IC_ROW_SLOTS],(uint64_t)n_selected_blocks*8u,memory_order_relaxed);
    return 1;
}
/* 整 chunk 批量版：kernel_glm_indexer_score_chunk_direct 与 _direct 的线程/网格/smem
 * 相同，只是多一维 token（grid.y）与逐行因果门。只支持 n_heads == 32，
 * 否则返回 0 让调用方回退到逐 token 路径。 */
static int indexscores_chunk(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,
    uint32_t n_heads,uint32_t n_tokens,uint32_t causal_start,uint32_t causal_ratio,bool cache_bf16) {
    if(n_heads!=32u||!n_comp||!n_tokens||!causal_ratio||causal_start==UINT32_MAX)return 0;
    if(!floats(scores,(uint64_t)n_comp*n_tokens)||
       !floats(q,(uint64_t)n_tokens*n_heads*128u)||
       !floats(weights,(uint64_t)n_tokens*n_heads)||!(cache_bf16?bf16s(cache,(uint64_t)n_comp*128u):
       floats(cache,(uint64_t)n_comp*128u)))return 0;
    const char *tiled_env = getenv("MYDS_INDEXER_TILED");
    int opt_tiled = (tiled_env && atoi(tiled_env) > 0) ? 1 : 0;
    const myds_metal_buffer_t *b[]={q,weights,cache,scores};
    if (opt_tiled && n_tokens >= 8) {
        struct {
            uint32_t n_comp, n_tokens, n_head, head_dim, pos0, ratio;
            uint64_t q_token_stride, q_head_stride, weights_token_stride;
            uint64_t index_row_stride, score_token_stride;
            float    scale;
        } ta = {
            .n_comp = n_comp,
            .n_tokens = n_tokens,
            .n_head = n_heads,
            .head_dim = 128,
            .pos0 = causal_start,
            .ratio = causal_ratio,
            .q_token_stride = (uint64_t)n_heads * 128u * sizeof(float),
            .q_head_stride = (uint64_t)128u * sizeof(float),
            .weights_token_stride = (uint64_t)n_heads * sizeof(float),
            .index_row_stride = (uint64_t)128u * (cache_bf16?2u:4u),
            .score_token_stride = (uint64_t)n_comp * sizeof(float),
            .scale = 1.0f / 64.0f
        };
        const uint32_t gx = (n_comp + 31u) / 32u;
        const uint32_t gy = (n_tokens + 7u) / 8u;
        const size_t smem = (8u * 128u + 32u * 128u + 8u * 32u) * sizeof(float);
        return dispatch(cache_bf16?"kernel_dsv4_indexer_scores_tiled_bf16":
                        "kernel_dsv4_indexer_scores_tiled_f32", &ta, sizeof(ta), b, 4,
                        gx, gy, 1, 32, 4, 1, smem);
    }
    struct {uint32_t n_rows,n_head,head_dim,cache_f16;float scale;
            uint32_t n_tokens,causal_start,causal_ratio,pad;} a={
        n_comp,n_heads,128,cache_bf16?2u:0u,1.0f/64.0f,n_tokens,causal_start,causal_ratio,0};
    const char *exact_env = getenv("MYDS_INDEXER_CHUNK_EXACT");
    if ((!exact_env || strcmp(exact_env, "0") != 0) &&
        dispatch("kernel_myds_indexer_score_chunk_exact", &a, sizeof(a), b, 4,
                 (n_comp + 15u) / 16u, (n_tokens + 7u) / 8u, 1, 32, 4, 1, 1280u * 4u)) {
        return 1;
    }
    return dispatch("kernel_glm_indexer_score_chunk_direct",&a,sizeof(a),b,4,
        n_comp,n_tokens,1,32,4,1,132u*4);
}
int myds_v41_gpu_indexscores_chunk(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,
    uint32_t n_heads,uint32_t n_tokens,uint32_t causal_start,uint32_t causal_ratio) {
    return indexscores_chunk(scores,q,weights,cache,n_comp,n_heads,n_tokens,causal_start,causal_ratio,false);
}
int myds_v41_gpu_indexscores_chunk_bf16(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,
    uint32_t n_heads,uint32_t n_tokens,uint32_t causal_start,uint32_t causal_ratio) {
    return indexscores_chunk(scores,q,weights,cache,n_comp,n_heads,n_tokens,causal_start,causal_ratio,true);
}
/* 按行批量（rows 行）。mask_stride 是 block_mask 的行步长（float）；
 * 逐 token 调用传 rows=1、start=pos、mask_stride=(n_comp+7)/8（与改前一致）。 */
static int candidates(myds_metal_buffer_t *out,const myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *mask,uint32_t n_comp,uint32_t rows,uint32_t start,
    uint32_t ratio,uint32_t mask_stride) {
    if(!n_comp||n_comp>UINT32_MAX-7||!rows||!ratio||start==UINT32_MAX)return 0;
    uint32_t blocks=(n_comp+7)/8,width=mask?n_comp:blocks;
    if(mask&&(!mask_stride||blocks>mask_stride))return 0;
    if(!floats(out,(uint64_t)width*rows)||!floats(scores,(uint64_t)n_comp*rows)||
       (mask&&!floats(mask,(uint64_t)mask_stride*rows)))return 0;
    const uint32_t a[]={n_comp,rows,start,ratio,mask_stride};
    const myds_metal_buffer_t *b[]={scores,out,mask?mask:scores};
    return dispatch(mask?"kernel_dsv41_candidate_filter":"kernel_dsv41_candidate_blocks",
        a,sizeof(a),b,3,(width+255u)/256,rows,1,256,1,1,0);
}
int myds_v41_gpu_candidates(myds_metal_buffer_t *blocks,const myds_metal_buffer_t *scores,
    uint32_t n_comp,uint32_t pos,uint32_t ratio) {
    return candidates(blocks,scores,NULL,n_comp,1,pos,ratio,(n_comp+7)/8);
}
int myds_v41_gpu_filter(myds_metal_buffer_t *scores,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t pos,uint32_t ratio) {
    return mask&&candidates(scores,scores,mask,n_comp,1,pos,ratio,(n_comp+7)/8);
}
int myds_v41_gpu_candidates_rows(myds_metal_buffer_t *blocks,const myds_metal_buffer_t *scores,
    uint32_t n_comp,uint32_t rows,uint32_t start,uint32_t ratio,uint32_t mask_stride) {
    return candidates(blocks,scores,NULL,n_comp,rows,start,ratio,mask_stride);
}
int myds_v41_gpu_filter_rows(myds_metal_buffer_t *scores,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t rows,uint32_t start,uint32_t ratio,uint32_t mask_stride) {
    return mask&&candidates(scores,scores,mask,n_comp,rows,start,ratio,mask_stride);
}

struct sort_args {
    int32_t ne00,ne01,ne02,ne03;uint64_t nb00,nb01,nb02,nb03;
    int32_t ne0,ne1,ne2,ne3,top_k;uint32_t causal_start,causal_ratio;
};
struct merge_args {
    int64_t ne00,ne01,ne02,ne03;uint64_t nb00,nb01,nb02,nb03;
    int32_t ne0,ne1,ne2,ne3,top_k,len;uint32_t causal_start,causal_ratio,block_width,block_top_k;
};
struct prefix_merge_args { struct merge_args merge; uint32_t keep; };
struct nan_fuse_merge_args { struct merge_args merge; uint32_t keep,parts; };
enum { TP_REQUESTS, TP_ENCODED, TP_UNAVAILABLE, TP_ROWS, TP_MERGES,
       TP_DENSE_WRITES, TP_PREFIX_WRITES, TP_COUNT };
static _Atomic uint64_t topk_prefix_counters[TP_COUNT];
static bool topk_prefix_test_unavailable;
enum { TN_REQUESTS, TN_ENCODED, TN_UNAVAILABLE, TN_PARTITIONS, TN_COUNT };
static _Atomic uint64_t topk_nan_fuse_counters[TN_COUNT];
static uint32_t topk_nan_fuse_test_unavailable;
void myds_v41_gpu_inject_topk_nan_fuse_unavailable(uint32_t mask) {
    topk_nan_fuse_test_unavailable=mask;
}
void myds_v41_gpu_topk_nan_fuse_counters(myds_v41_gpu_topk_nan_fuse_counters_t *out,bool reset) {
    uint64_t v[TN_COUNT];
    for(unsigned i=0;i<TN_COUNT;i++)v[i]=reset?
        atomic_exchange_explicit(&topk_nan_fuse_counters[i],0,memory_order_relaxed):
        atomic_load_explicit(&topk_nan_fuse_counters[i],memory_order_relaxed);
    if(out)*out=(myds_v41_gpu_topk_nan_fuse_counters_t){v[0],v[1],v[2],v[3]};
}
void myds_v41_gpu_inject_topk_prefix_unavailable(bool enabled) {
    topk_prefix_test_unavailable=enabled;
}
void myds_v41_gpu_topk_prefix_counters(myds_v41_gpu_topk_prefix_counters_t *out,bool reset) {
    uint64_t v[TP_COUNT];
    for(unsigned i=0;i<TP_COUNT;i++)v[i]=reset?
        atomic_exchange_explicit(&topk_prefix_counters[i],0,memory_order_relaxed):
        atomic_load_explicit(&topk_prefix_counters[i],memory_order_relaxed);
    if(out)*out=(myds_v41_gpu_topk_prefix_counters_t){v[0],v[1],v[2],v[3],v[4],v[5],v[6]};
}
/* Planned layout writes only. Unordered (NaN) rows use all dense destinations
 * in the guarded GPU kernel; the host does not read back classification. */
static void topk_prefix_count_writes(uint32_t n_cols,uint32_t top,uint32_t rows,
    uint32_t start,uint32_t ratio,uint32_t nth,uint32_t blocktop,uint32_t layout,uint32_t len) {
    uint64_t dense=0,prefix=0;
    const uint32_t limit=(layout+2u*len-1u)/(2u*len)==1u?top:layout;
    for(uint32_t row=0;row<rows;row++) {
        uint32_t width=ratio?(start+row+1u)/ratio:n_cols;
        if(width>n_cols)width=n_cols;
        if(!width)continue;
        uint32_t work=ratio?((width-1u)/nth)*blocktop+
            ((width-1u)%nth+1u<blocktop?(width-1u)%nth+1u:blocktop):layout;
        dense+=(layout+2u*len-1u)/(2u*len)==1u?(work<limit?work:limit):work;
        const uint32_t full=work/(2u*len),tail=work%(2u*len);
        prefix+=(uint64_t)full*(2u*len<top?2u*len:top)+(tail<top?tail:top);
    }
    atomic_fetch_add_explicit(&topk_prefix_counters[TP_DENSE_WRITES],dense,memory_order_relaxed);
    atomic_fetch_add_explicit(&topk_prefix_counters[TP_PREFIX_WRITES],prefix,memory_order_relaxed);
}
/* 按行 top-k：n_rows 行，行 stride = n_cols（scores）/ top（ids）。
 * causal_ratio != 0 时用 causal 变体：每行的可见列数 = (causal_start + row + 1)/ratio，
 * 行号就是 grid 里的 i01（= row），与 kernel_argsort_f32_i32_desc_causal 的语义一致。
 * 逐 token 调用走 n_rows=1、causal_ratio=0（非 causal 变体）—— 与改前完全相同。
 * 中间结果：非 causal 用 sort_a/sort_b（整行 stride = work，n_rows*work 个 float）；
 * 批量（causal）用 chunk_tmp_a/b，行 stride = npr==1 ? top : n_cols。
 * 批量路径要求 ids 的行步长 == top（调用方按 top 逐行读）。 */
static int topk_rows_causal(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,uint32_t n_rows,uint32_t causal_start,uint32_t causal_ratio,
    myds_v41_gpu_scratch_t *s) {
    if(!s||!n_rows||n_cols>INT32_MAX||!top||top>n_cols||!floats(ids,(uint64_t)top*n_rows)||
       !floats(scores,(uint64_t)n_cols*n_rows))return 0;
    if(causal_ratio&&causal_start==UINT32_MAX)return 0;
    const char *nan_fuse_env=getenv("MYDS_INDEXER_TOPK_NAN_FUSE");
    if(nan_fuse_env&&strcmp(nan_fuse_env,"0")&&strcmp(nan_fuse_env,"1")) {
        MYDS_LOG_E("MYDS_INDEXER_TOPK_NAN_FUSE must be exactly 0 or 1");
        return 0;
    }
    /* Match the reference bitonic partition as well as its comparator: ties
     * can otherwise select different cache rows after the merge. */
    uint32_t max_threads=myds_metal_kernel_max_threads(causal_ratio?
        "kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc");
    if(!max_threads)return 0;
    uint32_t nth=1;while(nth<n_cols&&2u*nth<=max_threads)nth*=2;
    uint32_t npr=(n_cols+nth-1)/nth,blocktop=top<nth?top:nth;
    uint32_t last=n_cols-(npr-1)*nth;
    uint32_t work=npr==1?top:(npr-1)*blocktop+(last<blocktop?last:blocktop);
    /* 一行的中间结果长度：非 causal（逐 token）是 work；causal（批量）则是内核按
     * 【该行自己的宽度】算出来的 work_width = ((w-1)/nth)*blocktop + min((w-1)%nth+1, blocktop)。
     * 它对 w 不单调（w 越接近 nth 的整数倍，右边那项越大），实测出现过比 work 还大的例子
     * （n_cols=5304 ⇒ work=2664，但 width=5267 的行要写 2707 个），那样会越界覆盖下一行。
     * 注意 work_width(w) ≤ w（因为 blocktop ≤ nth），所以批量路径的行步长直接取满宽
     * n_cols：每行一定落在自己行内，且缓冲需求与分数表同为 n_rows*n_cols（调用方已保证）。
     *
     * 但 dst 的行步长在两种情况下不同，必须分开看：
     *   - npr > 1：sort 写中间缓冲（步长 = ne0 = n_cols），最后一级 merge 写 ids
     *     （步长 = top_k = top）；
     *   - npr == 1：没有 merge，sort 直接把 ids 当 dst（步长 = ne0），所以此时
     *     ne0 必须 = top。
     * 调用方（prompt_selected / 块掩码 scatter）按 top 一行行读，所以批量路径要求
     * top == 调用方给 ids 的行步长；host 侧通过把分数表宽度补到 ≥ V41_TOPK 来保证
     * top == V41_TOPK。 */
    uint32_t layout=causal_ratio?(npr==1?top:n_cols):work;
    /* 中间结果缓冲：非 causal 的逐 token 路径沿用 sort_a/sort_b（大小 = max_comp，
     * 与旧行为一致）；批量（causal）路径改用专门的 chunk_tmp_a/chunk_tmp_b
     * （大小 = MYDS_V41_INDEXER_BATCH_FLOATS）—— 多行时需要的 n_rows*layout 个 float
     * 远超 max_comp，直接用 sort_a 会越界写坏堆（实测：n_rows*layout 超 max_comp 后
     * 相邻缓冲被覆盖，结果随机错乱）。 */
    myds_metal_buffer_t *tmp_a=causal_ratio?s->chunk_tmp_a:s->sort_a;
    myds_metal_buffer_t *tmp_b=causal_ratio?s->chunk_tmp_b:s->sort_b;
    if(!tmp_a||!tmp_b)return 0;
    if(causal_ratio) {
        if((uint64_t)n_rows*layout>(uint64_t)s->batch_floats)return 0;
    } else if(n_cols>s->max_comp||(uint64_t)n_rows*work>(uint64_t)s->max_comp||!n_rows)return 0;
    const char *prefix_env=getenv("MYDS_INDEXER_TOPK_PREFIX_MERGE");
    /* A lone final merge already truncates to top. Classification scans every
     * visible score and adds a dispatch/barriers: require at least 16 sort
     * partitions so larger intermediate merges can amortize that cost. */
    bool prefix=npr>=16u&&layout>2u*blocktop&&top<=512u&&prefix_env&&strcmp(prefix_env,"1")==0;
    const char *prefix_kernel=causal_ratio?
        "kernel_myds_argsort_merge_f32_i32_desc_causal_prefix_guarded":
        "kernel_myds_argsort_merge_f32_i32_desc_prefix_guarded";
    const bool nan_fuse_requested=prefix&&nan_fuse_env&&strcmp(nan_fuse_env,"1")==0;
    const char *nan_fuse_sort=causal_ratio?
        "kernel_myds_argsort_f32_i32_desc_causal_partition_nan":
        "kernel_myds_argsort_f32_i32_desc_partition_nan";
    const char *nan_fuse_merge=causal_ratio?
        "kernel_myds_argsort_merge_f32_i32_desc_causal_partition_prefix_guarded":
        "kernel_myds_argsort_merge_f32_i32_desc_partition_prefix_guarded";
    if(prefix) {
        atomic_fetch_add_explicit(&topk_prefix_counters[TP_REQUESTS],1,memory_order_relaxed);
        /* Probe both optional PSOs and allocate flags before sort writes.
         * Dispatch failure after classification starts fails closed. */
        if(nan_fuse_requested) {
            atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_REQUESTS],1,memory_order_relaxed);
            /* Preserve the reference nth even if the optional PSO has a lower
             * limit: changing the partition would change tie IDs. The clone
             * reserves 32 SIMD partials, supporting at most 1024 threads. */
            const char *sort_probe=(topk_nan_fuse_test_unavailable&MYDS_TOPK_NAN_FUSE_UNAVAILABLE_SORT)?
                "kernel_myds_argsort_partition_nan_missing_sort":nan_fuse_sort;
            const char *merge_probe=(topk_nan_fuse_test_unavailable&MYDS_TOPK_NAN_FUSE_UNAVAILABLE_MERGE)?
                "kernel_myds_argsort_partition_nan_missing_merge":nan_fuse_merge;
            prefix=!topk_prefix_test_unavailable&&nth<=1024u&&
                myds_metal_kernel_max_threads(sort_probe)>=nth&&
                myds_metal_kernel_max_threads(merge_probe)>=512u;
            const uint64_t slots=(uint64_t)npr*n_rows;
            if(prefix&&s->topk_nan_partition_slots<slots) {
                myds_metal_buffer_t *flags=slots<=SIZE_MAX/4u?
                    myds_metal_buffer_alloc((size_t)slots*4u):NULL;
                if(flags) {
                    myds_metal_buffer_free(s->topk_nan_partitions);
                    s->topk_nan_partitions=flags;s->topk_nan_partition_slots=slots;
                } else prefix=false;
            }
            if(!prefix)atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_UNAVAILABLE],1,memory_order_relaxed);
        } else prefix=!topk_prefix_test_unavailable&&
                myds_metal_kernel_max_threads("kernel_myds_argsort_visible_nan_rows")>=256u&&
                myds_metal_kernel_max_threads(prefix_kernel)>=512u;
        if(prefix&&!nan_fuse_requested&&s->topk_prefix_rows<n_rows) {
            myds_metal_buffer_t *flags=myds_metal_buffer_alloc((size_t)n_rows*4u);
            if(flags) {
                myds_metal_buffer_free(s->topk_prefix_nan);
                s->topk_prefix_nan=flags;s->topk_prefix_rows=n_rows;
            } else prefix=false;
        }
        if(!prefix)atomic_fetch_add_explicit(&topk_prefix_counters[TP_UNAVAILABLE],1,memory_order_relaxed);
        else if(!nan_fuse_requested) {
            struct {uint32_t cols,rows,start,ratio;} na={n_cols,n_rows,causal_start,causal_ratio};
            const myds_metal_buffer_t *nb[]={scores,s->topk_prefix_nan};
            /* Closes WAR when the same scratch is reused within a batch. */
            myds_metal_barrier_buffers(nb,2);
            if(!dispatch("kernel_myds_argsort_visible_nan_rows",&na,sizeof(na),nb,2,
                         n_rows,1,1,256,1,1,0))return 0;
            myds_metal_barrier_buffer(s->topk_prefix_nan);
            atomic_fetch_add_explicit(&topk_prefix_counters[TP_ROWS],n_rows,memory_order_relaxed);
            const myds_metal_buffer_t *sort_deps[]={tmp_a,tmp_b};
            myds_metal_barrier_buffers(sort_deps,2);
        }
        if(nan_fuse_requested) {
            /* WAR on scratch/flags reused in a concurrent command batch; the
             * upcoming sort overwrites every partition slot, including empty
             * causal groups. Also order tmp writes after a preflight failure:
             * dense fallback may follow a prior prefix call in this same batch.
             * No reset dispatch or inter-group atomics are needed. */
            const myds_metal_buffer_t *deps[]={scores,tmp_a,tmp_b,s->topk_nan_partitions};
            myds_metal_barrier_buffers(deps,s->topk_nan_partitions?4u:3u);
        }
    }
    struct sort_args a={.ne00=n_cols,.ne01=n_rows,.ne02=1,.ne03=1,.nb00=4,
        .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*n_rows*4,.nb03=(uint64_t)n_cols*n_rows*4,
        .ne0=layout,.ne1=n_rows,.ne2=1,.ne3=1,.top_k=blocktop,
        .causal_start=causal_start,.causal_ratio=causal_ratio};
    const bool nan_fuse=prefix&&nan_fuse_requested;
    const bool overlap_topk_qb = s && !myds_metal_get_auto_barrier() &&
        s->direct_deferred_q && s->direct_deferred_qb_weight && s->direct_deferred_qb_input &&
        !causal_ratio && n_rows == 1u && !prefix && npr >= 1u &&
        s->direct_deferred_qb_done > 0u && s->direct_deferred_qb_done < 64u;
    const myds_metal_buffer_t *qb_weight = overlap_topk_qb ? s->direct_deferred_qb_weight : NULL;
    const myds_metal_buffer_t *qb_input = overlap_topk_qb ? s->direct_deferred_qb_input : NULL;
    uint32_t qb_done = overlap_topk_qb ? s->direct_deferred_qb_done : 0u;
    uint32_t rope_done = 0u;
    if (s) {
        s->direct_deferred_qb_weight = NULL;
        s->direct_deferred_qb_input = NULL;
    }
    const myds_metal_buffer_t *b[]={scores,npr==1?ids:tmp_a,s->topk_nan_partitions};
    if(!dispatch(nan_fuse?nan_fuse_sort:causal_ratio?"kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc",
        &a,sizeof(a),b,nan_fuse?3u:2u,npr*n_rows,1,1,nth,1,1,((size_t)nth*8+15)&~(size_t)15))return 0;
    if(nan_fuse) {
        myds_metal_barrier_buffer(s->topk_nan_partitions);
        atomic_fetch_add_explicit(&topk_prefix_counters[TP_ROWS],n_rows,memory_order_relaxed);
        atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_PARTITIONS],(uint64_t)npr*n_rows,memory_order_relaxed);
    }
    if (overlap_topk_qb) {
        if (!myds_v41_gpu_rope_heads_slice(s, s->direct_deferred_q, 0u, qb_done,
                s->direct_deferred_q_pos, s->direct_deferred_q_compressed)) return 0;
        rope_done = qb_done;
        const uint32_t pass0_qb_target = (npr == 1u) ? 64u
            : (qb_done + 16u < 64u ? qb_done + 16u : 64u);
        if (pass0_qb_target > qb_done) {
            if (!myds_metal_dense_q4_v41_qb_slice(s->direct_deferred_q, qb_weight, qb_input,
                    qb_done * 512u, (pass0_qb_target - qb_done) * 512u, false)) return 0;
            qb_done = pass0_qb_target;
        }
    }
    myds_metal_buffer_t *cur=tmp_a,*next=tmp_b;
    for(uint32_t len=blocktop;len<layout;len*=2) {
        if (!myds_metal_get_auto_barrier()) {
            if (overlap_topk_qb) {
                const myds_metal_buffer_t *merge_deps[] = {cur, s->direct_deferred_q};
                myds_metal_barrier_buffers(merge_deps, 2u);
            } else {
                myds_metal_barrier_buffer(cur);
            }
        }
        uint32_t nm=(layout+2*len-1)/(2*len);
        struct merge_args m={.ne00=n_cols,.ne01=n_rows,.ne02=1,.ne03=1,.nb00=4,
            .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*n_rows*4,.nb03=(uint64_t)n_cols*n_rows*4,
            .ne0=layout,.ne1=n_rows,.ne2=1,.ne3=1,.top_k=nm==1?top:layout,.len=len,.block_width=nth,
            .block_top_k=blocktop,.causal_start=causal_start,.causal_ratio=causal_ratio};
        const myds_metal_buffer_t *mb[]={scores,cur,nm==1?ids:next,
            nan_fuse?s->topk_nan_partitions:s->topk_prefix_nan};
        if(prefix) {
            if(nan_fuse) {
                struct nan_fuse_merge_args fm={m,top,npr};
                if(!dispatch(nan_fuse_merge,&fm,sizeof(fm),mb,4,nm*n_rows,1,1,len<512?len:512,1,1,0))return 0;
            } else {
                struct prefix_merge_args pm={m,top};
                if(!dispatch(prefix_kernel,&pm,sizeof(pm),mb,4,nm*n_rows,1,1,len<512?len:512,1,1,0))return 0;
            }
            atomic_fetch_add_explicit(&topk_prefix_counters[TP_MERGES],1,memory_order_relaxed);
            topk_prefix_count_writes(n_cols,top,n_rows,causal_start,causal_ratio,nth,blocktop,layout,len);
        } else if(!dispatch(causal_ratio?"kernel_argsort_merge_f32_i32_desc_causal":
                            "kernel_argsort_merge_f32_i32_desc",
                            &m,sizeof(m),mb,3,nm*n_rows,1,1,len<512?len:512,1,1,0))return 0;
        if (overlap_topk_qb) {
            if (qb_done > rope_done) {
                if (!myds_v41_gpu_rope_heads_slice(s, s->direct_deferred_q, rope_done,
                        qb_done - rope_done, s->direct_deferred_q_pos,
                        s->direct_deferred_q_compressed)) return 0;
                rope_done = qb_done;
            }
            if (qb_done < 64u) {
                const uint32_t qb_target = (nm == 1u) ? 64u
                    : (qb_done + 10u < 64u ? qb_done + 10u : 64u);
                if (!myds_metal_dense_q4_v41_qb_slice(s->direct_deferred_q, qb_weight, qb_input,
                        qb_done * 512u, (qb_target - qb_done) * 512u, false)) return 0;
                qb_done = qb_target;
            }
        }
        myds_metal_buffer_t *tmp=cur;cur=next;next=tmp;
    }
    if (overlap_topk_qb) {
        s->direct_deferred_qb_done = qb_done;
        s->direct_deferred_q_rope_done = rope_done;
    }
    if (!myds_metal_get_auto_barrier() && !(s && s->direct_deferred_q)) {
        myds_metal_barrier_buffer(ids);
    }
    if(prefix)atomic_fetch_add_explicit(&topk_prefix_counters[TP_ENCODED],1,memory_order_relaxed);
    if(nan_fuse)atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_ENCODED],1,memory_order_relaxed);
    return 1;
}

/* Opt-in strict clone: default and all ineligible calls retain the original
 * helper. Successful encoding is not GPU completion or a performance claim. */
static _Atomic uint64_t topk_nan_shuffle_requests,topk_nan_shuffle_encoded,topk_nan_shuffle_failures;
int myds_v41_gpu_topk_nan_shuffle_enabled(void) {
    return myds_metal_indexer_topk_nan_shuffle_enabled();
}
void myds_v41_gpu_topk_nan_shuffle_counters(myds_v41_gpu_topk_nan_shuffle_counters_t *out,bool reset) {
#define NS_COUNT(field) (reset?atomic_exchange_explicit(&topk_nan_shuffle_##field,0u,memory_order_relaxed):atomic_load_explicit(&topk_nan_shuffle_##field,memory_order_relaxed))
    const myds_v41_gpu_topk_nan_shuffle_counters_t value={NS_COUNT(requests),NS_COUNT(encoded),NS_COUNT(failures)};
#undef NS_COUNT
    if(out)*out=value;
}

static int topk_rows_causal_nan_shuffle_encode(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,uint32_t n_rows,uint32_t causal_start,uint32_t causal_ratio,
    myds_v41_gpu_scratch_t *s) {
    if(!s||!n_rows||n_cols>INT32_MAX||!top||top>n_cols||!floats(ids,(uint64_t)top*n_rows)||
       !floats(scores,(uint64_t)n_cols*n_rows))return 0;
    if(causal_ratio&&causal_start==UINT32_MAX)return 0;
    const char *nan_fuse_env=getenv("MYDS_INDEXER_TOPK_NAN_FUSE");
    if(nan_fuse_env&&strcmp(nan_fuse_env,"0")&&strcmp(nan_fuse_env,"1")) {
        MYDS_LOG_E("MYDS_INDEXER_TOPK_NAN_FUSE must be exactly 0 or 1");
        return 0;
    }
    /* Match the reference bitonic partition as well as its comparator: ties
     * can otherwise select different cache rows after the merge. */
    uint32_t max_threads=myds_metal_kernel_max_threads(causal_ratio?
        "kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc");
    if(!max_threads)return 0;
    uint32_t nth=1;while(nth<n_cols&&2u*nth<=max_threads)nth*=2;
    uint32_t npr=(n_cols+nth-1)/nth,blocktop=top<nth?top:nth;
    uint32_t last=n_cols-(npr-1)*nth;
    uint32_t work=npr==1?top:(npr-1)*blocktop+(last<blocktop?last:blocktop);
    /* 一行的中间结果长度：非 causal（逐 token）是 work；causal（批量）则是内核按
     * 【该行自己的宽度】算出来的 work_width = ((w-1)/nth)*blocktop + min((w-1)%nth+1, blocktop)。
     * 它对 w 不单调（w 越接近 nth 的整数倍，右边那项越大），实测出现过比 work 还大的例子
     * （n_cols=5304 ⇒ work=2664，但 width=5267 的行要写 2707 个），那样会越界覆盖下一行。
     * 注意 work_width(w) ≤ w（因为 blocktop ≤ nth），所以批量路径的行步长直接取满宽
     * n_cols：每行一定落在自己行内，且缓冲需求与分数表同为 n_rows*n_cols（调用方已保证）。
     *
     * 但 dst 的行步长在两种情况下不同，必须分开看：
     *   - npr > 1：sort 写中间缓冲（步长 = ne0 = n_cols），最后一级 merge 写 ids
     *     （步长 = top_k = top）；
     *   - npr == 1：没有 merge，sort 直接把 ids 当 dst（步长 = ne0），所以此时
     *     ne0 必须 = top。
     * 调用方（prompt_selected / 块掩码 scatter）按 top 一行行读，所以批量路径要求
     * top == 调用方给 ids 的行步长；host 侧通过把分数表宽度补到 ≥ V41_TOPK 来保证
     * top == V41_TOPK。 */
    uint32_t layout=causal_ratio?(npr==1?top:n_cols):work;
    /* 中间结果缓冲：非 causal 的逐 token 路径沿用 sort_a/sort_b（大小 = max_comp，
     * 与旧行为一致）；批量（causal）路径改用专门的 chunk_tmp_a/chunk_tmp_b
     * （大小 = MYDS_V41_INDEXER_BATCH_FLOATS）—— 多行时需要的 n_rows*layout 个 float
     * 远超 max_comp，直接用 sort_a 会越界写坏堆（实测：n_rows*layout 超 max_comp 后
     * 相邻缓冲被覆盖，结果随机错乱）。 */
    myds_metal_buffer_t *tmp_a=causal_ratio?s->chunk_tmp_a:s->sort_a;
    myds_metal_buffer_t *tmp_b=causal_ratio?s->chunk_tmp_b:s->sort_b;
    if(!tmp_a||!tmp_b)return 0;
    const myds_metal_buffer_t *native[]={ids,scores,tmp_a,tmp_b};
    const uint64_t words[]={ (uint64_t)top*n_rows, (uint64_t)n_cols*n_rows,
        (uint64_t)layout*n_rows, (uint64_t)layout*n_rows };
    for(unsigned i=0;i<4u;i++) {
        if(words[i]>SIZE_MAX/4u||!myds_metal_buffer_native_valid(native[i])||
           !floats(native[i],words[i]))return 0;
        for(unsigned j=0;j<i;j++)if(buffer_ranges_overlap(native[i],(size_t)words[i]*4u,
            native[j],(size_t)words[j]*4u))return 0;
    }
    if(!myds_metal_indexer_topk_nan_shuffle_preflight(causal_ratio!=0u,nth))return 0;
    if(causal_ratio) {
        if((uint64_t)n_rows*layout>(uint64_t)s->batch_floats)return 0;
    } else if(n_cols>s->max_comp||(uint64_t)n_rows*work>(uint64_t)s->max_comp||!n_rows)return 0;
    const char *prefix_env=getenv("MYDS_INDEXER_TOPK_PREFIX_MERGE");
    /* A lone final merge already truncates to top. Classification scans every
     * visible score and adds a dispatch/barriers: require at least 16 sort
     * partitions so larger intermediate merges can amortize that cost. */
    bool prefix=npr>=16u&&layout>2u*blocktop&&top<=512u&&prefix_env&&strcmp(prefix_env,"1")==0;
    const char *prefix_kernel=causal_ratio?
        "kernel_myds_argsort_merge_f32_i32_desc_causal_prefix_guarded":
        "kernel_myds_argsort_merge_f32_i32_desc_prefix_guarded";
    const bool nan_fuse_requested=prefix&&nan_fuse_env&&strcmp(nan_fuse_env,"1")==0;
    const char *nan_fuse_sort=causal_ratio?
        "kernel_myds_argsort_f32_i32_desc_causal_partition_nan_shuffle":
        "kernel_myds_argsort_f32_i32_desc_partition_nan_shuffle";
    const char *nan_fuse_merge=causal_ratio?
        "kernel_myds_argsort_merge_f32_i32_desc_causal_partition_prefix_guarded":
        "kernel_myds_argsort_merge_f32_i32_desc_partition_prefix_guarded";
    if(prefix) {
        atomic_fetch_add_explicit(&topk_prefix_counters[TP_REQUESTS],1,memory_order_relaxed);
        /* Probe both optional PSOs and allocate flags before sort writes.
         * Dispatch failure after classification starts fails closed. */
        if(nan_fuse_requested) {
            atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_REQUESTS],1,memory_order_relaxed);
            /* Preserve the reference nth even if the optional PSO has a lower
             * limit: changing the partition would change tie IDs. The clone
             * reserves 32 SIMD partials, supporting at most 1024 threads. */
            const char *sort_probe=(topk_nan_fuse_test_unavailable&MYDS_TOPK_NAN_FUSE_UNAVAILABLE_SORT)?
                "kernel_myds_argsort_partition_nan_missing_sort":nan_fuse_sort;
            const char *merge_probe=(topk_nan_fuse_test_unavailable&MYDS_TOPK_NAN_FUSE_UNAVAILABLE_MERGE)?
                "kernel_myds_argsort_partition_nan_missing_merge":nan_fuse_merge;
            prefix=!topk_prefix_test_unavailable&&nth<=1024u&&
                myds_metal_kernel_max_threads(sort_probe)>=nth&&
                myds_metal_kernel_max_threads(merge_probe)>=512u;
            const uint64_t slots=(uint64_t)npr*n_rows;
            if(prefix&&s->topk_nan_partition_slots<slots) {
                myds_metal_buffer_t *flags=slots<=SIZE_MAX/4u?
                    myds_metal_buffer_alloc((size_t)slots*4u):NULL;
                if(flags) {
                    myds_metal_buffer_free(s->topk_nan_partitions);
                    s->topk_nan_partitions=flags;s->topk_nan_partition_slots=slots;
                } else prefix=false;
            }
            if(!prefix)atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_UNAVAILABLE],1,memory_order_relaxed);
        } else prefix=!topk_prefix_test_unavailable&&
                myds_metal_kernel_max_threads("kernel_myds_argsort_visible_nan_rows")>=256u&&
                myds_metal_kernel_max_threads(prefix_kernel)>=512u;
        if(prefix&&!nan_fuse_requested&&s->topk_prefix_rows<n_rows) {
            myds_metal_buffer_t *flags=myds_metal_buffer_alloc((size_t)n_rows*4u);
            if(flags) {
                myds_metal_buffer_free(s->topk_prefix_nan);
                s->topk_prefix_nan=flags;s->topk_prefix_rows=n_rows;
            } else prefix=false;
        }
        if(!prefix) { atomic_fetch_add_explicit(&topk_prefix_counters[TP_UNAVAILABLE],1,memory_order_relaxed);return 0; }
        else if(!nan_fuse_requested) {
            struct {uint32_t cols,rows,start,ratio;} na={n_cols,n_rows,causal_start,causal_ratio};
            const myds_metal_buffer_t *nb[]={scores,s->topk_prefix_nan};
            /* Closes WAR when the same scratch is reused within a batch. */
            myds_metal_barrier_buffers(nb,2);
            if(!dispatch("kernel_myds_argsort_visible_nan_rows",&na,sizeof(na),nb,2,
                         n_rows,1,1,256,1,1,0))return 0;
            myds_metal_barrier_buffer(s->topk_prefix_nan);
            atomic_fetch_add_explicit(&topk_prefix_counters[TP_ROWS],n_rows,memory_order_relaxed);
            const myds_metal_buffer_t *sort_deps[]={tmp_a,tmp_b};
            myds_metal_barrier_buffers(sort_deps,2);
        }
        if(nan_fuse_requested) {
            /* WAR on scratch/flags reused in a concurrent command batch; the
             * upcoming sort overwrites every partition slot, including empty
             * causal groups. Also order tmp writes after a preflight failure:
             * dense fallback may follow a prior prefix call in this same batch.
             * No reset dispatch or inter-group atomics are needed. */
            const myds_metal_buffer_t *deps[]={scores,tmp_a,tmp_b,s->topk_nan_partitions};
            myds_metal_barrier_buffers(deps,s->topk_nan_partitions?4u:3u);
        }
    }
    struct sort_args a={.ne00=n_cols,.ne01=n_rows,.ne02=1,.ne03=1,.nb00=4,
        .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*n_rows*4,.nb03=(uint64_t)n_cols*n_rows*4,
        .ne0=layout,.ne1=n_rows,.ne2=1,.ne3=1,.top_k=blocktop,
        .causal_start=causal_start,.causal_ratio=causal_ratio};
    const bool nan_fuse=prefix&&nan_fuse_requested;
    const uint64_t flag_words=(uint64_t)npr*n_rows;
    if(!nan_fuse||flag_words>SIZE_MAX/4u||
       !myds_metal_buffer_native_valid(s->topk_nan_partitions)||
       !floats(s->topk_nan_partitions,flag_words))return 0;
    for(unsigned i=0;i<4u;i++)if(buffer_ranges_overlap(s->topk_nan_partitions,
        (size_t)flag_words*4u,native[i],(size_t)words[i]*4u))return 0;
    const myds_metal_buffer_t *b[]={scores,npr==1?ids:tmp_a,s->topk_nan_partitions};
    if(!dispatch(nan_fuse?nan_fuse_sort:causal_ratio?"kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc",
        &a,sizeof(a),b,nan_fuse?3u:2u,npr*n_rows,1,1,nth,1,1,((size_t)nth*8+15)&~(size_t)15))return 0;
    if(nan_fuse) {
        myds_metal_barrier_buffer(s->topk_nan_partitions);
        atomic_fetch_add_explicit(&topk_prefix_counters[TP_ROWS],n_rows,memory_order_relaxed);
        atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_PARTITIONS],(uint64_t)npr*n_rows,memory_order_relaxed);
    }
    myds_metal_buffer_t *cur=tmp_a,*next=tmp_b;
    for(uint32_t len=blocktop;len<layout;len*=2) {
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(cur);
        uint32_t nm=(layout+2*len-1)/(2*len);
        struct merge_args m={.ne00=n_cols,.ne01=n_rows,.ne02=1,.ne03=1,.nb00=4,
            .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*n_rows*4,.nb03=(uint64_t)n_cols*n_rows*4,
            .ne0=layout,.ne1=n_rows,.ne2=1,.ne3=1,.top_k=nm==1?top:layout,.len=len,.block_width=nth,
            .block_top_k=blocktop,.causal_start=causal_start,.causal_ratio=causal_ratio};
        const myds_metal_buffer_t *mb[]={scores,cur,nm==1?ids:next,
            nan_fuse?s->topk_nan_partitions:s->topk_prefix_nan};
        if(prefix) {
            if(nan_fuse) {
                struct nan_fuse_merge_args fm={m,top,npr};
                if(!dispatch(nan_fuse_merge,&fm,sizeof(fm),mb,4,nm*n_rows,1,1,len<512?len:512,1,1,0))return 0;
            } else {
                struct prefix_merge_args pm={m,top};
                if(!dispatch(prefix_kernel,&pm,sizeof(pm),mb,4,nm*n_rows,1,1,len<512?len:512,1,1,0))return 0;
            }
            atomic_fetch_add_explicit(&topk_prefix_counters[TP_MERGES],1,memory_order_relaxed);
            topk_prefix_count_writes(n_cols,top,n_rows,causal_start,causal_ratio,nth,blocktop,layout,len);
        } else if(!dispatch(causal_ratio?"kernel_argsort_merge_f32_i32_desc_causal":
                            "kernel_argsort_merge_f32_i32_desc",
                            &m,sizeof(m),mb,3,nm*n_rows,1,1,len<512?len:512,1,1,0))return 0;
        myds_metal_buffer_t *tmp=cur;cur=next;next=tmp;
    }
    if (!myds_metal_get_auto_barrier() && !(s && s->direct_deferred_q)) {
        myds_metal_barrier_buffer(ids);
    }
    if(prefix)atomic_fetch_add_explicit(&topk_prefix_counters[TP_ENCODED],1,memory_order_relaxed);
    if(nan_fuse)atomic_fetch_add_explicit(&topk_nan_fuse_counters[TN_ENCODED],1,memory_order_relaxed);
    return 1;
}

static int topk_rows_causal_nan_shuffle(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t cols,uint32_t top,uint32_t rows,uint32_t start,uint32_t ratio,myds_v41_gpu_scratch_t *s) {
    const char *prefix=getenv("MYDS_INDEXER_TOPK_PREFIX_MERGE"),*nan=getenv("MYDS_INDEXER_TOPK_NAN_FUSE");
    if(!prefix||strcmp(prefix,"1")||!nan||strcmp(nan,"1")||!rows||!cols||cols>INT32_MAX||
       !top||top>cols||top>512u||(ratio&&start==UINT32_MAX))
        return topk_rows_causal(ids,scores,cols,top,rows,start,ratio,s);
    const uint32_t max=myds_metal_kernel_max_threads(ratio?
        "kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc");
    uint32_t nth=1u;
    if(max)while(nth<cols&&2u*nth<=max)nth*=2u;
    const uint32_t npr=(cols+nth-1u)/nth,block=top<nth?top:nth,last=cols-(npr-1u)*nth;
    const uint32_t work=npr==1u?top:(npr-1u)*block+(last<block?last:block);
    const uint32_t layout=ratio?(npr==1u?top:cols):work;
    if(max&&(npr<16u||layout<=2u*block))return topk_rows_causal(ids,scores,cols,top,rows,start,ratio,s);
    atomic_fetch_add_explicit(&topk_nan_shuffle_requests,1u,memory_order_relaxed);
    const int ok=max&&topk_rows_causal_nan_shuffle_encode(ids,scores,cols,top,rows,start,ratio,s);
    atomic_fetch_add_explicit(ok?&topk_nan_shuffle_encoded:&topk_nan_shuffle_failures,1u,memory_order_relaxed);
    return ok;
}

static _Atomic uint64_t threshold_host_requests,threshold_host_encoded,threshold_host_errors;
static bool threshold_test_unavailable;
int myds_v41_gpu_topk_threshold_enabled(void) { return myds_metal_indexer_topk_threshold_enabled(); }
void myds_v41_gpu_inject_topk_threshold_unavailable(bool enabled) { threshold_test_unavailable=enabled; }
void myds_v41_gpu_topk_threshold_counters(myds_v41_gpu_topk_threshold_counters_t *out,bool reset) {
#define TH_COUNT(field) (reset?atomic_exchange_explicit(&threshold_host_##field,0u,memory_order_relaxed):atomic_load_explicit(&threshold_host_##field,memory_order_relaxed))
    const myds_v41_gpu_topk_threshold_counters_t value={TH_COUNT(requests),TH_COUNT(encoded),TH_COUNT(errors)};
#undef TH_COUNT
    if(out)*out=value;
}
int myds_v41_gpu_topk_threshold_stats(myds_v41_gpu_topk_threshold_stats_t *out,bool reset) {
    uint32_t v[9]={0};const int ok=myds_metal_indexer_topk_threshold_stats(v,reset);
    if(out)*out=(myds_v41_gpu_topk_threshold_stats_t){v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]};
    return ok;
}
int myds_v41_gpu_topk_threshold_debug(const myds_v41_gpu_scratch_t *s,
    uint32_t control[12],uint32_t pairs[2048]) {
    uint32_t completed[9];
    if(!s||!control||!pairs||!s->threshold_control||!s->threshold_pairs||
       !myds_metal_indexer_topk_threshold_stats(completed,false)||
       !myds_metal_buffer_native_valid(s->threshold_control)||
       !myds_metal_buffer_native_valid(s->threshold_pairs)||
       myds_metal_buffer_size(s->threshold_control)<48u||
       myds_metal_buffer_size(s->threshold_pairs)<8192u)return 0;
    memcpy(control,myds_metal_buffer_data(s->threshold_control),48u);
    memcpy(pairs,myds_metal_buffer_data(s->threshold_pairs),8192u);
    return 1;
}
static int threshold_workspace(myds_v41_gpu_scratch_t *s) {
    if(!s->threshold_parts)s->threshold_parts=myds_metal_buffer_alloc(256u*16u);
    if(!s->threshold_control)s->threshold_control=myds_metal_buffer_alloc(12u*4u);
    if(!s->threshold_hist)s->threshold_hist=myds_metal_buffer_alloc(256u*256u*4u);
    if(!s->threshold_offsets)s->threshold_offsets=myds_metal_buffer_alloc(256u*8u);
    if(!s->threshold_pairs)s->threshold_pairs=myds_metal_buffer_alloc(1024u*8u);
    return s->threshold_parts&&s->threshold_control&&s->threshold_hist&&s->threshold_offsets&&s->threshold_pairs;
}
static int threshold_encode(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
                            uint32_t cols,myds_v41_gpu_scratch_t *s) {
    if(!s||cols<512u||cols>262144u||cols>s->max_comp||!floats(ids,512u)||!floats(scores,cols)||
       threshold_test_unavailable)return 0;
    const char *nan_env=getenv("MYDS_INDEXER_TOPK_NAN_FUSE");
    if(nan_env&&strcmp(nan_env,"0")&&strcmp(nan_env,"1"))return 0;
    // Preflight every candidate/reference PSO and resource before any GPU write.
    if(!myds_metal_indexer_topk_threshold_prepare()||!threshold_workspace(s))return 0;
    uint32_t max_threads=myds_metal_kernel_max_threads("kernel_argsort_f32_i32_desc"),nth=1u;
    if(!max_threads)return 0;
    while(nth<cols&&2u*nth<=max_threads)nth*=2u;
    const uint32_t npr=(cols+nth-1u)/nth,blocktop=nth<512u?nth:512u;
    const uint32_t last=cols-(npr-1u)*nth;
    const uint32_t work=npr==1u?512u:(npr-1u)*blocktop+(last<blocktop?last:blocktop);
    const uint32_t parts=(cols+1023u)/1024u;
    myds_metal_buffer_t *stats=myds_metal_indexer_topk_threshold_stats_buffer();
    const myds_metal_buffer_t *all[]={scores,ids,s->sort_a,s->sort_b,s->threshold_parts,
        s->threshold_control,s->threshold_hist,s->threshold_offsets,s->threshold_pairs,stats};
    const size_t sizes[]={(size_t)cols*4u,512u*4u,(size_t)work*4u,(size_t)work*4u,
        (size_t)parts*16u,48u,(size_t)parts*256u*4u,(size_t)parts*8u,8192u,36u};
    const size_t align[]={4u,4u,4u,4u,16u,4u,4u,8u,8u,4u};
    for(size_t i=0;i<10u;i++) {
        if(!myds_metal_buffer_native_valid(all[i])||myds_metal_buffer_size(all[i])<sizes[i]||
           myds_metal_buffer_offset(all[i])%align[i])return 0;
        for(size_t j=0;j<i;j++)if(buffer_ranges_overlap(all[i],sizes[i],all[j],sizes[j]))return 0;
    }
    const char *prefix_env=getenv("MYDS_INDEXER_TOPK_PREFIX_MERGE");
    const bool prefix=npr>=16u&&work>2u*blocktop&&prefix_env&&strcmp(prefix_env,"1")==0;
    struct {uint32_t cols,parts;} a={cols,parts};
    // Joins previous readers before overwriting every reused scratch plane and
    // orders upstream scores. All later RAW joins occur without CPU flag reads.
    myds_metal_barrier_buffers(all,10u);
    const myds_metal_buffer_t *range[]={scores,s->threshold_parts};
    if(!dispatch("kernel_myds_topk_threshold_range_parts",&a,sizeof(a),range,2,parts,1,1,256,1,1,0))return 0;
    myds_metal_barrier_buffer(s->threshold_parts);
    const myds_metal_buffer_t *reduce[]={s->threshold_parts,s->threshold_control};
    if(!dispatch("kernel_myds_topk_threshold_range_reduce",&a,sizeof(a),reduce,2,1,1,1,256,1,1,0))return 0;
    myds_metal_barrier_buffer(s->threshold_control);
    const myds_metal_buffer_t *hist[]={scores,s->threshold_control,s->threshold_hist};
    if(!dispatch("kernel_myds_topk_threshold_hist_parts",&a,sizeof(a),hist,3,parts,1,1,256,1,1,0))return 0;
    myds_metal_barrier_buffer(s->threshold_hist);
    const myds_metal_buffer_t *hr[]={s->threshold_hist,s->threshold_control,s->threshold_offsets};
    if(!dispatch("kernel_myds_topk_threshold_hist_reduce",&a,sizeof(a),hr,3,1,1,1,256,1,1,0))return 0;
    const myds_metal_buffer_t *hdeps[]={s->threshold_control,s->threshold_offsets};
    myds_metal_barrier_buffers(hdeps,2u);
    const myds_metal_buffer_t *compact[]={scores,s->threshold_control,s->threshold_offsets,s->threshold_pairs};
    if(!dispatch("kernel_myds_topk_threshold_compact",&a,sizeof(a),compact,4,parts,1,1,256,1,1,0))return 0;
    const myds_metal_buffer_t *cdeps[]={s->threshold_control,s->threshold_pairs};
    myds_metal_barrier_buffers(cdeps,2u);
    const myds_metal_buffer_t *final[]={s->threshold_control,s->threshold_pairs,ids,stats};
    if(!dispatch("kernel_myds_topk_threshold_final",&a,sizeof(a),final,4,1,1,1,512,1,1,8192u))return 0;
    myds_metal_barrier_buffer(s->threshold_control);
    struct sort_args sa={.ne00=cols,.ne01=1,.ne02=1,.ne03=1,.nb00=4,
        .nb01=(uint64_t)cols*4,.nb02=(uint64_t)cols*4,.nb03=(uint64_t)cols*4,
        .ne0=work,.ne1=1,.ne2=1,.ne3=1,.top_k=blocktop,.causal_start=0,.causal_ratio=0};
    const myds_metal_buffer_t *sb[]={scores,npr==1u?ids:s->sort_a,s->threshold_control};
    if(!dispatch("kernel_myds_argsort_f32_i32_desc_threshold_guarded",&sa,sizeof(sa),sb,3,
                 npr,1,1,nth,1,1,((size_t)nth*8u+15u)&~(size_t)15u))return 0;
    myds_metal_buffer_t *cur=s->sort_a,*next=s->sort_b;
    for(uint32_t len=blocktop;len<work;len*=2u) {
        myds_metal_barrier_buffer(cur);
        const uint32_t nm=(work+2u*len-1u)/(2u*len);
        struct {struct merge_args merge;uint32_t keep,prefix;} ma={
            {.ne00=cols,.ne01=1,.ne02=1,.ne03=1,.nb00=4,.nb01=(uint64_t)cols*4,
             .nb02=(uint64_t)cols*4,.nb03=(uint64_t)cols*4,.ne0=work,.ne1=1,.ne2=1,.ne3=1,
             .top_k=nm==1u?512u:work,.len=len,.block_width=nth,.block_top_k=blocktop,
             .causal_start=0,.causal_ratio=0},512u,prefix?1u:0u};
        const myds_metal_buffer_t *mb[]={scores,cur,nm==1u?ids:next,s->threshold_control};
        if(!dispatch("kernel_myds_argsort_merge_f32_i32_desc_threshold_guarded",&ma,sizeof(ma),mb,4,
                     nm,1,1,len<512u?len:512u,1,1,0))return 0;
        myds_metal_buffer_t *tmp=cur;cur=next;next=tmp;
    }
    myds_metal_barrier_buffer(ids);
    return 1;
}
int myds_v41_gpu_topk_threshold_candidate(myds_metal_buffer_t *ids,
    const myds_metal_buffer_t *scores,uint32_t cols,myds_v41_gpu_scratch_t *s) {
    atomic_fetch_add_explicit(&threshold_host_requests,1u,memory_order_relaxed);
    const int ok=threshold_encode(ids,scores,cols,s);
    atomic_fetch_add_explicit(ok?&threshold_host_encoded:&threshold_host_errors,1u,memory_order_relaxed);
    return ok;
}
int myds_v41_gpu_topk_threshold(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t cols,uint32_t top,myds_v41_gpu_scratch_t *s) {
    if(!myds_v41_gpu_topk_threshold_enabled()||top!=512u||cols<8192u||cols>262144u)
        return topk_rows_causal(ids,scores,cols,top,1,0,0,s);
    return myds_v41_gpu_topk_threshold_candidate(ids,scores,cols,s);
}
int myds_v41_gpu_topk(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,myds_v41_gpu_scratch_t *s) {
    if(myds_v41_gpu_topk_nan_shuffle_enabled())return topk_rows_causal_nan_shuffle(ids,scores,n_cols,top,1,0,0,s);
    return topk_rows_causal(ids,scores,n_cols,top,1,0,0,s);
}
int myds_v41_gpu_topk_causal(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,uint32_t n_rows,uint32_t causal_start,uint32_t causal_ratio,
    myds_v41_gpu_scratch_t *s) {
    if(myds_v41_gpu_topk_nan_shuffle_enabled())return topk_rows_causal_nan_shuffle(ids,scores,n_cols,top,n_rows,causal_start,causal_ratio,s);
    return topk_rows_causal(ids,scores,n_cols,top,n_rows,causal_start,causal_ratio,s);
}
int myds_v41_gpu_topkmask(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top) {
    return myds_v41_gpu_topkmask_rows(mask,ids,n_cols,top,1,n_cols);
}
int myds_v41_gpu_topkmask_rows(myds_metal_buffer_t *mask,const myds_metal_buffer_t *ids,
    uint32_t n_cols,uint32_t top,uint32_t rows,uint32_t mask_stride) {
    if(!rows||!mask_stride||n_cols>mask_stride||!top||top>n_cols||
       !floats(mask,(uint64_t)mask_stride*rows)||!floats(ids,(uint64_t)top*rows))return 0;
    struct {int64_t ne00,ne01;uint64_t nb00,nb01;int64_t ne0,ne1;uint64_t nb0,nb1;} a={
        top,rows,4,(uint64_t)top*4,n_cols,rows,4,(uint64_t)mask_stride*4};
    const myds_metal_buffer_t *b[]={ids,mask};
    /* 清整行（ne0×ne1 个元素）再 scatter 打开的 picks：两个 dispatch 都在同一个
     * batch 里、顺序执行，与逐 token 时相同。 */
    const uint64_t clear=(uint64_t)n_cols*rows;
    const uint64_t scatter=(uint64_t)top*rows;
    if(clear>UINT32_MAX*256ull||scatter>UINT32_MAX*256ull)return 0;
    if (!dispatch("kernel_dsv4_topk_mask",&a,sizeof(a),b,2,
               (uint32_t)((clear+255u)/256u),1,1,256,1,1,0)) return 0;
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(mask);
    return dispatch("kernel_dsv4_topk_mask_scatter",&a,sizeof(a),b,2,
               (uint32_t)((scatter+255u)/256u),1,1,256,1,1,0);
}
int myds_v41_gpu_gather(myds_metal_buffer_t *out,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *ids,uint32_t source_rows,uint32_t selected_rows) {
    if(!selected_rows||selected_rows>512||selected_rows>source_rows||
        !floats(out,(uint64_t)selected_rows*512)||!floats(cache,(uint64_t)source_rows*512)||!floats(ids,selected_rows))return 0;
    struct {int32_t ne00t,ne00;uint64_t nb01,nb02,nb03;int32_t ne10;uint64_t nb10,nb11,nb12,nb1,nb2,nb3;} a={
        .ne00t=512,.ne00=512,.nb01=2048,.nb02=(uint64_t)source_rows*2048,.nb03=(uint64_t)source_rows*2048,
        .ne10=selected_rows,.nb10=4,.nb11=(uint64_t)selected_rows*4,.nb12=(uint64_t)selected_rows*4,
        .nb1=2048,.nb2=(uint64_t)selected_rows*2048,.nb3=(uint64_t)selected_rows*2048};
    const myds_metal_buffer_t *b[]={cache,ids,out};
    return dispatch("kernel_get_rows_f32",&a,sizeof(a),b,3,2*selected_rows,1,1,256,1,1,0);
}
int myds_v41_gpu_gather_bf16(myds_metal_buffer_t *out,const myds_metal_buffer_t *cache,
    const myds_metal_buffer_t *ids,uint32_t source_rows,uint32_t selected_rows) {
    if(!selected_rows||selected_rows>512||selected_rows>source_rows||
       !floats(out,(uint64_t)selected_rows*512)||!bf16s(cache,(uint64_t)source_rows*512)||
       !floats(ids,selected_rows))return 0;
    const uint32_t a[]={source_rows,selected_rows};
    const myds_metal_buffer_t *b[]={cache,ids,out};
    return dispatch("kernel_dsv41_get_rows_bf16",a,sizeof(a),b,3,2,selected_rows,1,256,1,1,0);
}

/* Use transient subviews only as binding handles; Metal retains the underlying
 * resources until queued dispatches finish. No CPU readback or synchronization. */
static int stage_f16(myds_metal_buffer_t *dst,size_t dstoff,const myds_metal_buffer_t *src,size_t srcoff,uint32_t count,myds_v41_gpu_scratch_t *s) {
    if(!s||!src||!dst||!myds_metal_buffer_update_sub_view(s->sv_view,src,srcoff,(size_t)count*4)||
       !myds_metal_buffer_update_sub_view(s->dv_view,dst,dstoff,(size_t)count*2)) return 0;
    const myds_metal_buffer_t *b[]={s->sv_view,s->dv_view};
    return dispatch("kernel_cpy_contig_f32_f16_4",&count,sizeof(count),b,2,(count+1023u)/1024,1,1,256,1,1,0);
}
static int stage_bf16_f16(myds_metal_buffer_t *dst,size_t dstoff,const myds_metal_buffer_t *src,
    size_t srcoff,uint32_t count,myds_v41_gpu_scratch_t *s) {
    if(!s||!src||!dst||!myds_metal_buffer_update_sub_view(s->sv_view,src,srcoff,(size_t)count*2u)||
       !myds_metal_buffer_update_sub_view(s->dv_view,dst,dstoff,(size_t)count*2u))return 0;
    const myds_metal_buffer_t *b[]={s->sv_view,s->dv_view};
    return dispatch("kernel_dsv41_stage_bf16_f16",&count,sizeof(count),b,2,
                    (count+255u)/256u,1,1,256,1,1,0);
}
struct flash_pad_args {
    int32_t ne11,ne_12_2,ne_12_3;uint64_t nb11,nb12,nb13,nb21,nb22,nb23;
    int32_t ne31,ne32,ne33;uint64_t nb31,nb32,nb33;
};
struct flash_vec_args {
    int32_t ne01,ne02,ne03;uint64_t nb01,nb02,nb03;int32_t ne11,ne_12_2,ne_12_3,ns10;
    uint64_t nb11,nb12,nb13;int32_t ns20;uint64_t nb21,nb22,nb23;
    int32_t ne31,ne32,ne33;uint64_t nb31,nb32,nb33;int32_t ne1,ne2,ne3;
    float scale,max_bias,m0,m1;int32_t n_head_log2;float logit_softcap;
    uint64_t nb_kvq;   /* 每 query 的 K/V 基址步长；0 = 旧行为 */
};
/* 两侧手工镜像（宿主 C ↔ Metal），无编译器交叉校验：加字段必须同步改两边。
 * 这里钉死尺寸，任何一侧改动而另一边没跟就会在编译期炸，而不是跑出垃圾 K 地址。 */
_Static_assert(sizeof(struct flash_vec_args) == 200,
               "flash_vec_args 尺寸变了 —— 内核侧 struct ds4_metal_args_flash_attn_ext_vec 必须同步");
_Static_assert(sizeof(struct flash_vec_args) - 8 == 192, "nb_kvq 必须在尾部偏移 192");
struct attention_rope_fuse_args {
    struct flash_vec_args attention;
    uint32_t pos, inverse;
    float frequencies[32];
};
_Static_assert(sizeof(struct attention_rope_fuse_args)==336u,"attention+rope ABI");
_Static_assert(offsetof(struct attention_rope_fuse_args,pos)==200u,"attention+rope position ABI");
_Static_assert(offsetof(struct attention_rope_fuse_args,inverse)==204u,"attention+rope inverse ABI");
_Static_assert(offsetof(struct attention_rope_fuse_args,frequencies)==208u,"attention+rope frequency ABI");
struct attention_rope_scope { uint32_t pos; bool compressed; };

/* B 方案：见 metal/myds_chunk.metal 里的 myds_attn_gather_args（必须逐字段对齐）。 */
struct myds_attn_gather_args {
    uint32_t rows, top_k, n_comp, valid_from, mask_stride, c_start, base_pos, ratio;
};

/* 进程级开关：默认【开】。MYDS_ATTN_GATHER=0 可关回旧路径。 */
static int attn_gather_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("MYDS_ATTN_GATHER");
        cached = (v && atoi(v) == 0) ? 0 : 1;
    }
    return cached;
}
/* 建窗口/因果 mask 模板。列 [0,127) 是 carry，[127,127+c_len) 是本 chunk。
 * 行 r（chunk 内第 r 个 token）能看的是 slab 列 [r, r+127]：位置不超过自己（因果）
 * 且不早于自己往前 128 个（窗口）。carry 部分还要满足"那个位置真实存在"。
 * 模板只在 (c_len, carry) 变化时重建 —— 完整的 chunk 形状相同，所以整个 prefill 只建一两次。 */
static int build_chunk_mask_template(myds_v41_gpu_scratch_t *s,uint32_t c_len,uint32_t carry) {
    if(c_len==0||c_len>MYDS_CHUNK_MAXC||carry>MYDS_CHUNK_CARRY)return 0;
    const uint32_t cols=MYDS_CHUNK_CARRY+c_len;
    const uint32_t a[]={c_len,s->mask_u16,carry};
    const myds_metal_buffer_t *b[]={s->mask_chunk};
    /* Build on the GPU so earlier encoded attention calls can still read the
     * previous mask, including several mixed paths in a single command batch. */
    if(!dispatch("kernel_myds_mask_chunk_window",a,sizeof(a),b,1,
                 (cols+63u)/64u,c_len,1,64,1,1,0))return 0;
    s->tmpl_len=c_len;s->tmpl_carry=carry;
    return 1;
}

static int ensure_chunk_workspace(myds_v41_gpu_scratch_t *s,uint32_t rows) {
    if(rows<=s->slab_rows)return 1;
    const uint32_t full_rows=(MYDS_CHUNK_CARRY+MYDS_CHUNK_MAXC+s->key_comp_max+31u)/32u*32u;
    if(rows>full_rows)return 0;
    myds_metal_buffer_t *kv=myds_metal_buffer_alloc((size_t)full_rows*512u*2u);
    myds_metal_buffer_t *mask=myds_metal_buffer_alloc((size_t)MYDS_CHUNK_MAXC*full_rows*2u);
    if(!kv||!mask){myds_metal_buffer_free(kv);myds_metal_buffer_free(mask);return 0;}
    memset(myds_metal_buffer_data(kv),0,(size_t)full_rows*512u*2u);
    memset(myds_metal_buffer_data(mask),0xff,(size_t)MYDS_CHUNK_MAXC*full_rows*2u);
    /* Preserve the carry in GPU order. Encoders retain the old MTLBuffers until
     * prior work finishes, so replacing their host handles does not free an
     * in-flight mask or read the carry prematurely on the CPU. */
    if(!myds_metal_buffer_copy(kv,0,s->kv_chunk,0,(size_t)MYDS_CHUNK_CARRY*512u*2u)){
        myds_metal_buffer_free(kv);myds_metal_buffer_free(mask);return 0;
    }
    myds_metal_buffer_free(s->kv_chunk);myds_metal_buffer_free(s->mask_chunk);
    s->kv_chunk=kv;s->mask_chunk=mask;s->slab_rows=full_rows;s->mask_u16=full_rows;
    s->tmpl_len=0;s->tmpl_carry=0;
    return 1;
}

/* 整个 chunk 的 attention，一次 dispatch。省掉的正是每 token 的 staging/gather/pad/vec/reduce。 */
static int attention_chunk(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *chunk_kv,
    const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *ids,
    myds_metal_buffer_t *gathered,
    uint32_t c_len,uint32_t n_comp,uint32_t top_k,
    uint32_t base_pos,uint32_t c_start,uint32_t ratio,myds_v41_gpu_scratch_t *s,bool comp_bf16) {
    myds_v41_gpu_attn_kv_reuse_reset(s);
    if(!s||!heads||!q||!chunk_kv||c_len==0||c_len>MYDS_CHUNK_MAXC||top_k>512u||
       n_comp>s->key_comp_max||
       !floats(heads,(uint64_t)c_len*64u*512)||!floats(q,(uint64_t)c_len*64u*512)||
       !floats(sinks,64)||!floats(chunk_kv,(uint64_t)c_len*512)||
       (uint64_t)base_pos+c_start+c_len>UINT32_MAX||
       (n_comp&&!ratio)||
       (n_comp&&(!comp_kv||!(comp_bf16?bf16s(comp_kv,(uint64_t)n_comp*512):floats(comp_kv,(uint64_t)n_comp*512))||
                 !ids||!floats(ids,(uint64_t)c_len*top_k))))return 0;

    /* 诊断：把压缩键数强行调小，【只影响 attention 读取的 K/V 量，不影响 dispatch 结构】，
     * 用来判断这 8 秒是不是 K/V 带宽 bound。结果不完整，仅计时。 */
    if(__builtin_expect(s->keys_clamp > 0, 0)) {
        if(s->keys_clamp < n_comp) n_comp = s->keys_clamp;
    }

    const uint32_t p0=base_pos+c_start;
    const uint32_t carry=p0<MYDS_CHUNK_CARRY?p0:MYDS_CHUNK_CARRY;

    /* B 方案（docs/B_MAX_SPARSE_ATTENTION_SPEC.md）：把本 query 可见的 640 个键
     * （128 滑窗 + 512 picks）gather 成紧凑段，内核直接只算这 20 块。
     *
     * 开了它，下面三件事就全是【死重】，直接跳过：
     *   ① 掩码模板 —— 640 列全部由两个 gather 内核显式写（窗口段 / picks 段），
     *      模板写的 [0,128) 与 [128,640) 都会被覆盖，没有别的读者；
     *   ② 压缩前缀 staging —— 只有旧内核走 slab 里的压缩列才需要；
     *      comp gather 是直接从 comp_kv 读 f32，不经 slab（省掉 ∝n_comp 的拷贝）；
     *   ③ mask fill / scatter 两次 dispatch。 */
    const int use_gather = attn_gather_enabled() && gathered != NULL && n_comp != 0 &&
                          n_comp>=s->gather_min_keys;
    /* The matrix experiment visits all 640 gathered slots. Auto keeps vec for
     * early causal prefixes, small chunks, and every slab/window-only call.
     * The threshold is a policy for tested shapes, not a numerical guarantee. */
    const int use_matrix = use_gather && (s->prefill_attn_variant==1 ||
        (s->prefill_attn_variant==2 && !s->attn_matrix_unavailable && c_len>=32u && p0/ratio>=256u && n_comp>=512u));
    if(use_gather&&(myds_metal_buffer_offset(gathered)%2u||
       myds_metal_buffer_size(gathered)<(size_t)c_len*640u*512u*2u))return 0;
    attn_count(AC_CALLS);attn_count(use_gather?AC_GATHER:AC_SLAB);

    if(!use_gather&&!ensure_chunk_workspace(s,MYDS_CHUNK_CARRY+c_len+n_comp))return 0;

    if(!use_gather && (s->tmpl_len!=c_len||s->tmpl_carry!=carry))
        if(!build_chunk_mask_template(s,c_len,carry))return 0;

    const uint32_t comp_base=MYDS_CHUNK_CARRY+c_len;
    const uint32_t keys=comp_base+n_comp;
    /* The kernel walks keys in blocks of 32 and skips a block whose mask row is all
     * invisible, so the key count is rounded up and the extra columns stay masked
     * instead of being padded into a separate buffer. */
    const uint32_t keys_pad=(keys+31u)/32u*32u;

    if(!stage_f16(s->kv_chunk,(size_t)MYDS_CHUNK_CARRY*1024u,chunk_kv,0,c_len*512u,s))return 0;
    if(!use_gather && n_comp && !(comp_bf16?
       stage_bf16_f16(s->kv_chunk,(size_t)comp_base*1024u,comp_kv,0,n_comp*512u,s):
       stage_f16(s->kv_chunk,(size_t)comp_base*1024u,comp_kv,0,n_comp*512u,s)))return 0;

    if(!use_gather) {
    /* 填充覆盖 [comp_base, keys_pad)：n_comp 只是【本 chunk 真实的压缩键数】，而内核读的
     * 是 32 对齐后的 keys_pad，多出来那几列必须显式置为不可见
     * （以前靠模板的清尾巴兜底，现在模板不再清 —— 见 build_chunk_mask_template）。 */
    {
        const uint32_t pad_cols=keys_pad-comp_base;
        const uint32_t fill_groups=(pad_cols+63u)/64u;
        if(fill_groups) {
            struct {uint32_t base,stride,n_comp,top_k,rows,c_start,base_pos,ratio;} mf={
                comp_base,s->mask_u16,pad_cols,top_k,c_len,c_start,base_pos,ratio};
            const myds_metal_buffer_t *fb[]={s->mask_chunk};
            /* 位置单调递增 ⇒ 上一个 chunk 打开的 picks 一定落在 [comp_base, comp_base+pad_cols) 内，
             * 所以这个区间清成不可见就够。线程组按 64 线程/组折算（拿「编译期的列跨度」当组数
             * 会让 64 倍线程空转，且随行宽一起变大 —— 抬上限时实测慢了 2.1s）。 */
            if(!dispatch("kernel_myds_mask_chunk_fill",&mf,sizeof(mf),fb,1,
                         fill_groups,c_len,1,64,1,1,0))return 0;
        }
    }
    if(n_comp&&top_k) {
        struct {uint32_t base,stride,n_comp,top_k,rows,c_start,base_pos,ratio;} ma={
            comp_base,s->mask_u16,n_comp,top_k,c_len,c_start,base_pos,ratio};
        const myds_metal_buffer_t *sb[]={s->mask_chunk,ids};
        if(!dispatch("kernel_myds_mask_chunk_scatter",&ma,sizeof(ma),sb,2,
                     (top_k+63u)/64u,c_len,1,64,1,1,0))return 0;
    }
    }   /* !use_gather */

    if (use_gather) {
        /* Both gather kernels overwrite the shared mask, even when their chunk
         * has a different length. A later window-only layer must rebuild it. */
        s->tmpl_len=0;s->tmpl_carry=0;
        struct myds_attn_gather_args ga={
            c_len,top_k,n_comp,MYDS_CHUNK_CARRY-carry,s->mask_u16,c_start,base_pos,ratio};
        {   const myds_metal_buffer_t *g1[]={gathered,s->mask_chunk,s->kv_chunk};
            if(!dispatch("kernel_myds_attn_gather_window",&ga,sizeof(ga),g1,3,
                         128u,c_len,1,128,1,1,0))return 0; }
        {   const myds_metal_buffer_t *g2[]={gathered,s->mask_chunk,comp_kv,ids};
            if(!dispatch(comp_bf16?"kernel_myds_attn_gather_comp_bf16":
                         "kernel_myds_attn_gather_comp",&ga,sizeof(ga),g2,4,
                         512u,c_len,1,128,1,1,0))return 0; }
    }

    const uint32_t kv_keys = use_gather ? 640u : keys_pad;
    struct flash_vec_args a={
        .ne01=(int32_t)c_len,.ne02=64,.ne03=1,.nb01=64u*2048u,.nb02=2048u,.nb03=64u*2048u,
        .ne11=(int32_t)kv_keys,.ne_12_2=1,.ne_12_3=1,.ns10=512,
        .nb11=1024u,.nb12=(uint64_t)kv_keys*1024u,.nb13=(uint64_t)kv_keys*1024u,
        .ns20=512,.nb21=1024u,.nb22=(uint64_t)kv_keys*1024u,.nb23=(uint64_t)kv_keys*1024u,
        .ne31=(int32_t)c_len,.ne32=1,.ne33=1,
        .nb31=(uint64_t)s->mask_u16*2u,
        .nb32=(uint64_t)s->mask_u16*2u*(uint64_t)c_len,
        .nb33=(uint64_t)s->mask_u16*2u*(uint64_t)c_len,
        .ne1=64,.ne2=(int32_t)c_len,.ne3=1,.scale=1.0f/sqrtf(512),
        .nb_kvq= use_gather ? 640u*1024u : 0u };
    const myds_metal_buffer_t *kv_sel = use_gather ? gathered : s->kv_chunk;
    const myds_metal_buffer_t *b[]={q,kv_sel,kv_sel,s->mask_chunk,sinks,s->pad,heads};
    const myds_metal_constant_t c[]={{400,true,1},{401,true,1},{402,true,0},{403,true,0},{404,true,0},
        {405,true,0},{420,false,512},{421,false,512},{422,false,1},{423,false,1}};
    int matrix_encoded=0;
    if(use_matrix){
        /* Keep a unchanged: an unavailable matrix PSO/32KiB workspace must fall
         * back with the original query/head axes and gathered-KV strides. */
        struct flash_vec_args ma=a;
        int ready=1;
        if(!s->attn_head_blocks){
            s->attn_head_blocks=myds_metal_buffer_alloc((size_t)MYDS_CHUNK_MAXC*8u*10u);
            if(!s->attn_head_blocks)ready=0;
        }
        ma.ne01=64;ma.ne02=(int32_t)c_len;ma.nb01=2048u;ma.nb02=64u*2048u;
        ma.ne_12_2=(int32_t)c_len;ma.ne31=64;ma.ne32=(int32_t)c_len;
        ma.nb31=0;ma.nb32=(uint64_t)s->mask_u16*2u;
        struct {int32_t ne01,ne30,ne31,ne32,ne33;uint64_t nb31,nb32,nb33;} ba={
            64,640,64,(int32_t)c_len,1,0,(uint64_t)s->mask_u16*2u,
            (uint64_t)s->mask_u16*2u*c_len};
        _Static_assert(sizeof(ba)==48,"flash block scanner Metal ABI");
        const myds_metal_buffer_t *bb[]={s->mask_chunk,s->attn_head_blocks};
        const myds_metal_constant_t bc[]={{224,false,8},{225,false,64}};
        if(ready)ready=specialized("kernel_flash_attn_ext_blk",&ba,sizeof(ba),bb,2,
                                  10,8,c_len,32,1,1,0,bc,2);
        const myds_metal_buffer_t *mb[]={q,kv_sel,kv_sel,s->mask_chunk,sinks,s->pad,s->attn_head_blocks,heads};
        const myds_metal_constant_t mc[]={{300,true,1},{301,true,1},{302,true,0},{303,true,0},
            {304,true,0},{310,true,0},{320,false,512},{321,false,512},{322,false,4}};
        /* Both diagnostic failures use the real PSO/dispatch validation path.
         * It returns before encoding a matrix dispatch for absent functions or
         * excessive shared memory. A prior scanner only modifies its flags. */
        const char *kernel=attn_matrix_test_failure==1?
            "kernel_myds_attention_test_missing_matrix_pso":
            "kernel_myds_flash_attn_heads_f16_dk512_dv512";
        const size_t smem=attn_matrix_test_failure==2?SIZE_MAX-15u:32768u;
        if(ready)matrix_encoded=specialized(kernel,&ma,192,mb,8,8,c_len,1,32,4,1,smem,mc,9);
        if(matrix_encoded)attn_count(AC_MATRIX);
        else{
            if(s->prefill_attn_variant!=2){ /* explicit matrix is strict */
                MYDS_LOG_E("attention: MYDS_PREFILL_ATTN=matrix could not encode "
                           "(c_len=%u n_comp=%u pos=%u); set MYDS_DEBUG_DISPATCH=1 for the reason",
                           c_len,n_comp,p0);
                return 0;
            }
            attn_count(AC_MATRIX_FALLBACKS);
            s->attn_matrix_unavailable=1;
            MYDS_LOG_W("attention: matrix auto unavailable (c_len=%u n_comp=%u pos=%u); "
                       "scratch %p uses vec from now on. MYDS_DEBUG_DISPATCH=1 shows the reason",
                       c_len,n_comp,p0,(void *)s);
        }
    }
    if(!matrix_encoded){
        int vec_encoded=0;
        if(use_gather && s->attn_vec_head_first){
            if(s->attn_vec_head_first_unavailable) attn_count(AC_HF_CACHED);
            else {
                const char *kernel=attn_vec_head_first_test_failure?
                    "kernel_myds_attention_test_missing_head_first_pso":
                    "kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512";
                /* Only the physical query/head grid changes. Keep a, the buffers,
                 * constants, one SIMD group and 3328-byte private scratch unchanged. */
                attn_count(AC_HF_ATTEMPTS);
                vec_encoded=specialized(kernel,&a,sizeof(a),b,7,64,c_len,1,32,1,1,3328,c,10);
                if(vec_encoded)attn_count(AC_HF_ENCODED);
                else{
                    /* Sticky for this scratch (the process lifetime of a context):
                     * say so once, because every later gathered chunk silently
                     * pays the original grid's cost. */
                    attn_count(AC_HF_FAILURES);
                    s->attn_vec_head_first_unavailable=1;
                    MYDS_LOG_W("attention: head-first vec could not encode (c_len=%u n_comp=%u pos=%u); "
                               "scratch %p uses the original grid from now on. "
                               "MYDS_DEBUG_DISPATCH=1 shows the reason",c_len,n_comp,p0,(void *)s);
                }
            }
        }
        if(!vec_encoded){
            if(!specialized("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                            &a,sizeof(a),b,7,c_len,64,1,32,1,1,3328,c,10))return 0;
            attn_count(AC_ORIGINAL);
        }
    }

    /* This chunk's tail becomes the next chunk's carry: rows [c_len, c_len + 127)
     * move down to [0, 127).  When the chunk is shorter than the carry window the
     * source and destination overlap, and myds_metal_buffer_copy rejects overlapping
     * ranges on one buffer (a dispatch based copy is unordered across them), so the
     * move happens on the CPU.  It is 127 rows, once per chunk.  Without this, any
     * run whose last chunk is shorter than 127 rows fails outright, which is what the
     * default chunk size of 256 produces for a prompt that is not a multiple of it. */
    {
        /* Two copies through a scratch buffer rather than one overlapping copy:
         * myds_metal_buffer_copy rejects overlapping ranges on the same buffer, and
         * a CPU move would read the slab while the dispatch that filled it may still
         * be in flight.  Both copies below go between different buffers, and the copy
         * helper brackets each one with a buffer memory barrier. */
        const size_t carry_bytes=(size_t)MYDS_CHUNK_CARRY*512u*2u;
        if(!myds_metal_buffer_copy(s->kv_f16,0,s->kv_chunk,(size_t)c_len*1024u,carry_bytes))return 0;
        if(!myds_metal_buffer_copy(s->kv_chunk,0,s->kv_f16,0,carry_bytes))return 0;
    }
    return 1;
}
int myds_v41_gpu_attention_chunk(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *chunk_kv,
    const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *ids,myds_metal_buffer_t *gathered,
    uint32_t c_len,uint32_t n_comp,uint32_t top_k,uint32_t base_pos,uint32_t c_start,
    uint32_t ratio,myds_v41_gpu_scratch_t *s) {
    return attention_chunk(heads,sinks,q,chunk_kv,comp_kv,ids,gathered,c_len,n_comp,top_k,
                           base_pos,c_start,ratio,s,false);
}
int myds_v41_gpu_attention_chunk_bf16(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *chunk_kv,
    const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *ids,myds_metal_buffer_t *gathered,
    uint32_t c_len,uint32_t n_comp,uint32_t top_k,uint32_t base_pos,uint32_t c_start,
    uint32_t ratio,myds_v41_gpu_scratch_t *s) {
    return attention_chunk(heads,sinks,q,chunk_kv,comp_kv,ids,gathered,c_len,n_comp,top_k,
                           base_pos,c_start,ratio,s,true);
}

enum { AR_REQUESTS, AR_ENCODED, AR_FAILURES, AR_DIRECT, AR_STAGED, AR_COUNT };
static _Atomic uint64_t attention_register_counters[AR_COUNT];
static inline void attention_register_count(unsigned field) {
    atomic_fetch_add_explicit(&attention_register_counters[field],1u,memory_order_relaxed);
}
int myds_v41_gpu_attention_packed_register_enabled(void) {
    const char *value=getenv("MYDS_ATTN_PACKED_REGISTER");
    if(!value || !strcmp(value,"0")) return 0;
    if(strcmp(value,"1")) {
        MYDS_LOG_E("MYDS_ATTN_PACKED_REGISTER must be exactly 0 or 1");return -1;
    }
    const char *nsg=getenv("MYDS_ATTN_PACKED_NSG");
    const char *reduce=getenv("MYDS_ATTN_PACKED_REDUCE");
    if((nsg && strcmp(nsg,"8")) || (reduce && strcmp(reduce,"1"))) {
        MYDS_LOG_E("MYDS_ATTN_PACKED_REGISTER=1 requires physical NSG8 and packed reduce1");return -1;
    }
    return 1;
}
void myds_v41_gpu_attention_packed_register_counters(
    myds_v41_gpu_attention_packed_register_counters_t *out,bool reset) {
    uint64_t values[AR_COUNT];
    for(unsigned i=0;i<AR_COUNT;i++) values[i]=reset?
        atomic_exchange_explicit(&attention_register_counters[i],0u,memory_order_relaxed):
        atomic_load_explicit(&attention_register_counters[i],memory_order_relaxed);
    if(out) *out=(myds_v41_gpu_attention_packed_register_counters_t){
        .requests=values[AR_REQUESTS],.encoded=values[AR_ENCODED],.failures=values[AR_FAILURES],
        .direct_encoded=values[AR_DIRECT],.staged_encoded=values[AR_STAGED]};
}

enum { RF_REQUESTS, RF_ENCODED, RF_FAILURES, RF_COUNT };
static _Atomic uint64_t attention_rope_counters[RF_COUNT];
static void attention_rope_count(unsigned field) {
    atomic_fetch_add_explicit(&attention_rope_counters[field],1u,memory_order_relaxed);
}
int myds_v41_attention_rope_fuse_enabled(void) {
    const char *value=getenv("MYDS_ATTN_ROPE_FUSE");
    if (!value || !strcmp(value,"0")) return 0;
    if (strcmp(value,"1")) {
        MYDS_LOG_E("MYDS_ATTN_ROPE_FUSE must be exactly 0 or 1");return -1;
    }
    if (myds_v41_gpu_attention_packed_register_enabled()!=1) {
        MYDS_LOG_E("MYDS_ATTN_ROPE_FUSE=1 requires packed register1/physical8/reduce1");return -1;
    }
    return 1;
}
void myds_v41_attention_rope_fuse_counters(myds_metal_decode_candidate_counters_t *out,bool reset) {
    uint64_t v[RF_COUNT];
    for (unsigned i=0;i<RF_COUNT;i++) v[i]=reset?
        atomic_exchange_explicit(&attention_rope_counters[i],0u,memory_order_relaxed):
        atomic_load_explicit(&attention_rope_counters[i],memory_order_relaxed);
    if (out) *out=(myds_metal_decode_candidate_counters_t){v[0],v[1],v[2]};
}
static int attention_rope_failure(void) {
    attention_rope_count(RF_FAILURES);
    return 0;
}
static int attention_rope_range(const myds_metal_buffer_t *b,uint64_t bytes,size_t alignment) {
    return myds_metal_buffer_native_range(b,bytes,alignment);
}
static _Atomic uint64_t attn_kv_reuse_requests, attn_kv_reuse_encoded, attn_kv_reuse_failures;
static _Atomic int attn_kv_reuse_mode;
int myds_v41_attn_kv_reuse_parse_mode(void) {
    const char *v = getenv("MYDS_ATTN_KV_REUSE");
    const int mode = (!v || !strcmp(v, "0")) ? 0 : !strcmp(v, "1") ? 1 : -1;
    if (mode < 0) MYDS_LOG_E("MYDS_ATTN_KV_REUSE must be exactly 0 or 1");
    atomic_store_explicit(&attn_kv_reuse_mode, mode, memory_order_relaxed);
    return mode;
}
int myds_v41_attn_kv_reuse_enabled(void) {
    return atomic_load_explicit(&attn_kv_reuse_mode, memory_order_relaxed);
}
void myds_v41_attn_kv_reuse_reset_mode(void) {
    atomic_store_explicit(&attn_kv_reuse_mode, 0, memory_order_relaxed);
}
void myds_v41_attn_kv_reuse_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
#define ATTN_KV_REUSE_COUNT(field) \
    (reset ? atomic_exchange_explicit(&attn_kv_reuse_##field, 0u, memory_order_relaxed) \
           : atomic_load_explicit(&attn_kv_reuse_##field, memory_order_relaxed))
    const myds_metal_decode_candidate_counters_t value = {
        .requests = ATTN_KV_REUSE_COUNT(requests),
        .encoded = ATTN_KV_REUSE_COUNT(encoded),
        .failures = ATTN_KV_REUSE_COUNT(failures)};
    if (out) *out = value;
#undef ATTN_KV_REUSE_COUNT
}
void myds_v41_attn_kv_reuse_record_prestage_failure(void) {
    atomic_fetch_add_explicit(&attn_kv_reuse_requests, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
}
void myds_v41_gpu_attn_kv_reuse_reset(myds_v41_gpu_scratch_t *s) {
    if (!s) return;
    s->direct_window_prestaged = false;
    s->direct_scratch_fenced = false;
    s->direct_populate_kv_cache = false;
    s->direct_expect_kv_reuse = false;
    s->direct_kv_cache_valid = false;
    s->direct_pad_cache_valid = false;
    s->direct_deferred_q_compressed = false;
    s->direct_deferred_q = NULL;
    s->direct_deferred_qb_weight = NULL;
    s->direct_deferred_qb_input = NULL;
    s->direct_deferred_q_pos = 0;
    s->direct_deferred_qb_done = 0;
    s->direct_deferred_q_rope_done = 0;
    s->direct_kv_comp_buf = NULL;
    s->direct_kv_ids_buf = NULL;
    s->direct_kv_comp_offset = 0;
    s->direct_kv_ids_offset = 0;
    s->direct_kv_n_raw = 0;
    s->direct_kv_source_rows = 0;
    s->direct_kv_selected_rows = 0;
    s->direct_kv_pos = 0;
}
int myds_v41_gpu_attn_kv_reuse_prepare_layer(myds_v41_gpu_scratch_t *s,
    uint32_t layer, uint32_t count, bool is_index_source) {
    if (!s || !s->kv_f16 || !s->pad || layer >= 40u) return 0;
    s->direct_window_prestaged = false;
    s->direct_scratch_fenced = false;
    s->direct_deferred_q_compressed = false;
    s->direct_deferred_q = NULL;
    s->direct_deferred_qb_weight = NULL;
    s->direct_deferred_qb_input = NULL;
    s->direct_deferred_q_pos = 0;
    s->direct_deferred_qb_done = 0;
    s->direct_deferred_q_rope_done = 0;
    if (layer < 2u || is_index_source) {
        s->direct_kv_cache_valid = false;
        s->direct_pad_cache_valid = false;
        s->direct_expect_kv_reuse = false;
        s->direct_populate_kv_cache = (layer >= 2u && is_index_source && count > 0u);
    } else {
        s->direct_populate_kv_cache = false;
        s->direct_expect_kv_reuse = (count > 0u);
    }
    return 1;
}
int myds_v41_gpu_attn_kv_reuse_prestage_window(myds_v41_gpu_scratch_t *s,
    const myds_metal_buffer_t *window, uint32_t pos) {
    if (!s || !s->kv_f16 || !floats(window, 128u * 512u) || pos >= 1048576u) return 0;
    const uint32_t n_raw = pos + 1u < 128u ? pos + 1u : 128u;
    const uint32_t raw_start = (pos + 1u - n_raw) % 128u;
    const uint32_t w_args[] = {n_raw, raw_start};
    const myds_metal_buffer_t *wb[] = {window, s->kv_f16};
    if (!dispatch("kernel_dsv41_copy_window_f32_to_f16", w_args, sizeof(w_args), wb, 2, 1, n_raw, 1, 128, 1, 1, 0)) return 0;
    s->direct_window_prestaged = true;
    s->direct_scratch_fenced = true;
    return 1;
}
void myds_v41_gpu_attn_kv_reuse_mark_fenced(myds_v41_gpu_scratch_t *s) {
    if (s) s->direct_scratch_fenced = true;
}
void myds_v41_gpu_attn_kv_reuse_defer_q_rope(myds_v41_gpu_scratch_t *s,
    myds_metal_buffer_t *q, bool compressed) {
    if (!s) return;
    s->direct_deferred_q = q;
    s->direct_deferred_q_compressed = compressed;
    s->direct_deferred_qb_weight = NULL;
    s->direct_deferred_qb_input = NULL;
    s->direct_deferred_q_pos = 0;
    s->direct_deferred_qb_done = 0;
    s->direct_deferred_q_rope_done = 0;
}
void myds_v41_gpu_attn_kv_reuse_bind_topk_qb(myds_v41_gpu_scratch_t *s,
    const myds_metal_buffer_t *qb_weight, const myds_metal_buffer_t *qb_input,
    uint32_t pos, uint32_t qb_done_heads) {
    if (!s || !s->direct_deferred_q) return;
    s->direct_deferred_qb_weight = qb_weight;
    s->direct_deferred_qb_input = qb_input;
    s->direct_deferred_q_pos = pos;
    s->direct_deferred_qb_done = qb_done_heads;
    s->direct_deferred_q_rope_done = 0;
}
uint32_t myds_v41_gpu_attn_kv_reuse_qb_done(const myds_v41_gpu_scratch_t *s) {
    return s ? s->direct_deferred_qb_done : 0u;
}
const myds_metal_buffer_t *myds_v41_gpu_scratch_kv_f16(const myds_v41_gpu_scratch_t *s) {
    return s ? s->kv_f16 : NULL;
}
const myds_metal_buffer_t *myds_v41_gpu_scratch_pad(const myds_v41_gpu_scratch_t *s) {
    return s ? s->pad : NULL;
}
myds_metal_buffer_t *myds_v41_gpu_scratch_partial(const myds_v41_gpu_scratch_t *s) {
    return s ? s->partial : NULL;
}
enum { AP_REQUESTS, AP_ENCODED, AP_FAILURES, AP_COUNT };
static _Atomic uint64_t attention_pack_counters[AP_COUNT];
static pthread_once_t attention_pack_once=PTHREAD_ONCE_INIT;
static int attention_pack_mode;
static const char attention_pack_kernel[]="kernel_myds_dsv41_direct_pack_fuse";
static void attention_pack_init(void) {
    const char *value=getenv("MYDS_ATTN_DIRECT_PACK_FUSE");
    if (!value || !strcmp(value,"0")) return;
    if (strcmp(value,"1")) {
        MYDS_LOG_E("MYDS_ATTN_DIRECT_PACK_FUSE must be exactly 0 or 1");
        attention_pack_mode=-1;return;
    }
    if (myds_v41_attention_rope_fuse_enabled()!=1) {
        MYDS_LOG_E("MYDS_ATTN_DIRECT_PACK_FUSE=1 requires RF1/packed register1/physical8/reduce1");
        attention_pack_mode=-1;return;
    }
    const char *direct=getenv("MYDS_ATTN_DIRECT_PACK");
    if(direct&&strcmp(direct,"1")) {
        MYDS_LOG_E("MYDS_ATTN_DIRECT_PACK_FUSE=1 requires MYDS_ATTN_DIRECT_PACK absent or exactly1");
        attention_pack_mode=-1;return;
    }
    attention_pack_mode=1;
}
int myds_v41_gpu_attention_direct_pack_fuse_enabled(void) {
    pthread_once(&attention_pack_once,attention_pack_init);
    return attention_pack_mode;
}
const char *myds_v41_gpu_attention_direct_pack_fuse_selected_kernel(void) {
    return myds_v41_gpu_attention_direct_pack_fuse_enabled()==1 ? attention_pack_kernel : NULL;
}
void myds_v41_gpu_attention_direct_pack_fuse_counters(myds_metal_decode_candidate_counters_t *out,bool reset) {
    uint64_t v[AP_COUNT];
    for(unsigned i=0;i<AP_COUNT;i++)v[i]=reset?
        atomic_exchange_explicit(&attention_pack_counters[i],0u,memory_order_relaxed):
        atomic_load_explicit(&attention_pack_counters[i],memory_order_relaxed);
    if(out)*out=(myds_metal_decode_candidate_counters_t){v[0],v[1],v[2]};
}
static int attention_pack_failure(void) {
    atomic_fetch_add_explicit(&attention_pack_counters[AP_FAILURES],1u,memory_order_relaxed);
    return 0;
}
struct attention_direct_pack_args {uint32_t n_raw,raw_start,source_rows,selected_rows;};
_Static_assert(sizeof(struct attention_direct_pack_args)==16u,"direct pack ABI");
/* Call only before scratch WAR joins/writes. The scoped RF entry has already
 * validated heads/pad/partial and every reader/writer cross-alias. Keep an
 * independent complete pack guard and PSO preflight here as well. */
static int attention_direct_pack_preflight(const myds_metal_buffer_t *dst,
    const myds_metal_buffer_t *window,uint32_t n_raw,uint32_t raw_start,
    const myds_metal_buffer_t *comp,const myds_metal_buffer_t *ids,
    uint32_t source_rows,uint32_t selected_rows) {
    if(!n_raw||n_raw>128u||raw_start>=128u||selected_rows>512u||
       (selected_rows&&(!source_rows||selected_rows>source_rows)))return 0;
    const uint64_t dn=(uint64_t)(n_raw+selected_rows)*512u*2u,wn=128u*512u*4u;
    if(!myds_metal_buffer_native_range(dst,dn,16u)||
       !myds_metal_buffer_native_range(window,wn,16u)||
       buffer_ranges_overlap(dst,(size_t)dn,window,(size_t)wn))return 0;
    if(selected_rows) {
        const uint64_t cn=(uint64_t)source_rows*512u*2u,in=(uint64_t)selected_rows*4u;
        if(!myds_metal_buffer_native_range(comp,cn,16u)||
           !myds_metal_buffer_native_range(ids,in,4u)||
           buffer_ranges_overlap(dst,(size_t)dn,comp,(size_t)cn)||
           buffer_ranges_overlap(dst,(size_t)dn,ids,(size_t)in))return 0;
    }
    const uint32_t threads[]={128u,1u,1u};
    return myds_metal_kernel_constants_preflight(attention_pack_kernel,threads,0u,NULL,0u);
}
/* Counts this selected pack dispatch only. A later RF failure propagates from
 * the enclosing forward; it is not a pack failure after pack encoded success. */
static int attention_direct_pack_encode(myds_metal_buffer_t *dst,
    const myds_metal_buffer_t *window,uint32_t n_raw,uint32_t raw_start,
    const myds_metal_buffer_t *comp,const myds_metal_buffer_t *ids,
    uint32_t source_rows,uint32_t selected_rows) {
    const struct attention_direct_pack_args a={n_raw,raw_start,source_rows,selected_rows};
    const myds_metal_buffer_t *b[]={window,selected_rows?comp:window,selected_rows?ids:window,dst};
    const uint64_t before=myds_metal_dispatch_count();
    const int ok=dispatch(attention_pack_kernel,&a,sizeof(a),b,4,1,n_raw+selected_rows,1,128,1,1,0);
    if(myds_metal_dispatch_count()>before)
        atomic_fetch_add_explicit(&attention_pack_counters[AP_ENCODED],1u,memory_order_relaxed);
    return ok ? 1 : attention_pack_failure();
}
/* Complete predispatch guards: no scratch staging, joins, CB or dispatch may
 * precede failure. Read-only selected IDs retain the original GPU range check;
 * this host never synchronizes or inspects their payload. */
static int attention_rope_validate(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,uint32_t raw_start,
    const myds_metal_buffer_t *selected,const myds_metal_buffer_t *ids,uint32_t source_rows,
    uint32_t selected_rows,myds_v41_gpu_scratch_t *s,uint32_t pos,bool direct) {
    if (!s || pos>=1048576u || !n_raw || n_raw>128u || raw_start>=128u || selected_rows>512u ||
        (direct && selected_rows && selected_rows>source_rows)) return 0;
    const myds_metal_buffer_t *writers[]={heads,s->kv_f16,s->pad,s->partial};
    const uint64_t wn[]={64u*512u*4u,640u*512u*2u,2u*32u*512u*2u+32u*2u,64u*(512u+2u)*32u*4u};
    const myds_metal_buffer_t *readers[]={sinks,q,window,s->mask,selected,ids};
    const uint64_t rn[]={64u*4u,64u*512u*4u,128u*512u*4u,640u*2u,
        direct?(uint64_t)source_rows*512u*2u:(uint64_t)selected_rows*512u*4u,(uint64_t)selected_rows*4u};
    const size_t align[]={4u,16u,16u,2u,16u,4u};
    for (unsigned i=0;i<4u;i++) {
        if (!attention_rope_range(writers[i],wn[i],16u)) return 0;
        for (unsigned j=0;j<i;j++) if(buffer_ranges_overlap(writers[i],(size_t)wn[i],writers[j],(size_t)wn[j])) return 0;
    }
    for (unsigned j=0;j<6u;j++) {
        if (j>=4u && (!selected_rows || (j==5u && !direct))) continue;
        if (!attention_rope_range(readers[j],rn[j],align[j])) return 0;
        for (unsigned i=0;i<4u;i++) if(buffer_ranges_overlap(writers[i],(size_t)wn[i],readers[j],(size_t)rn[j])) return 0;
    }
    const uint32_t threads[]={256u,1u,1u};
    const uint32_t keys=n_raw+selected_rows;
    const myds_metal_constant_t c[]={{400,true,1},{401,true,1},{402,true,0},{403,true,0},{404,true,keys%32u!=0},
        {405,true,0},{420,false,512},{421,false,512},{422,false,1},{423,false,32}};
    return myds_metal_kernel_constants_preflight("kernel_myds_attention_register_rope_fuse_f16_512",threads,22400u,c,10u);
}

/* Physical SIMD packing is separate from the legacy logical NSG=1/NWG=32
 * constants. Absent / exact 8 preserves the existing packed8 path. Explicit
 * candidates are experiments: malformed values and unavailable geometry fail
 * before they can silently benchmark the original implementation. */
static int get_attn_packed_physical_nsg(void) {
    const char *v = getenv("MYDS_ATTN_PACKED_NSG");
    if (!v || !strcmp(v, "8")) return 8;
    if (!strcmp(v, "4")) return 4;
    if (!strcmp(v, "16")) return 16;
    MYDS_LOG_E("MYDS_ATTN_PACKED_NSG must be exactly 4, 8, or 16");
    return 0;
}

/* Shared Q + physical half score banks + 32 logical split weights/stats +
 * sink scales + padded 32x33 float4 partial plane. Match the shader layout. */
#define ATTN_PACKED_SMEM(nsg) (22400u + 256u * (nsg))
_Static_assert(ATTN_PACKED_SMEM(8u) == 24448u, "packed8 allocation changed");
_Static_assert(ATTN_PACKED_SMEM(4u) == 23424u, "packed4 allocation mismatch");
_Static_assert(ATTN_PACKED_SMEM(16u) == 26496u, "packed16 allocation mismatch");

static void log_attn_packed_candidate_encoded(int nsg, const char *kernel) {
    static _Atomic uint32_t logged;
    const uint32_t bit = nsg == 4 ? 1u : 2u;
    if (!(atomic_load_explicit(&logged, memory_order_relaxed) & bit) &&
        !(atomic_fetch_or_explicit(&logged, bit, memory_order_relaxed) & bit)) {
        MYDS_LOG_I("packed attention candidate encoded: kernel=%s physical_nsg=%d logical_splits=32 groups=64x1x1 threads=%ux1x1 dynamic_bytes=%u",
            kernel, nsg, 32u * (uint32_t)nsg, ATTN_PACKED_SMEM((uint32_t)nsg));
    }
}

static inline int get_attn_packed_reduce_mode(void) {
    const char *v = getenv("MYDS_ATTN_PACKED_REDUCE");
    if (!v) return 1;
    return (strcmp(v, "0") == 0) ? 0 : 1;
}

static int attention_staged_impl(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,myds_v41_gpu_scratch_t *s,int register_mode,
    const struct attention_rope_scope *rope) {
    myds_v41_gpu_attn_kv_reuse_reset(s);
    if(!s||!n_raw||n_raw>128||raw_start>=128||selected_rows>512||
        !floats(heads,64u*512)||!floats(q,64u*512)||!floats(sinks,64)||!floats(window,128u*512)||
        (selected_rows&&!floats(selected_kv,(uint64_t)selected_rows*512)))return 0;
    const int packed_nsg = get_attn_packed_physical_nsg();
    if (!packed_nsg) return 0;
    if (packed_nsg != 8 && !get_attn_packed_reduce_mode()) {
        MYDS_LOG_E("explicit MYDS_ATTN_PACKED_NSG=%d requires packed reduce enabled", packed_nsg);
        return 0;
    }
    if (!myds_metal_get_auto_barrier()) {
        /* The previous attention may still READ these shared buffers in a
         * concurrent encoder. Fence before staging/pad/partial writes (WAR). */
        const myds_metal_buffer_t *scratch_buffers[] = {s->kv_f16, s->pad, s->partial};
        myds_metal_barrier_buffers(scratch_buffers, 3);
    }
    uint32_t tail=128-raw_start<n_raw?128-raw_start:n_raw;
    if(!stage_f16(s->kv_f16,0,window,(size_t)raw_start*2048,tail*512,s))return 0;
    if(n_raw>tail&&!stage_f16(s->kv_f16,(size_t)tail*1024,window,0,(n_raw-tail)*512,s))return 0;
    if(selected_rows&&!stage_f16(s->kv_f16,(size_t)n_raw*1024,selected_kv,0,selected_rows*512,s))return 0;
    if(!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(s->kv_f16);
    uint32_t keys=n_raw+selected_rows;uint64_t kvbytes=(uint64_t)keys*1024,maskbytes=(uint64_t)keys*2;
    if(keys%32) {
        struct flash_pad_args a={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
            .nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,.nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes};
        const myds_metal_buffer_t *b[]={s->kv_f16,s->kv_f16,s->mask,s->pad};
        const myds_metal_constant_t c[]={{100,true,1},{125,false,32}};
        if(!specialized("kernel_flash_attn_ext_pad",&a,sizeof(a),b,4,32,1,1,32,1,1,0,c,2))return 0;
        /* Packed and original vec both READ the generated pad K/V and mask. */
        if(!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(s->pad);
    }
    struct flash_vec_args a={.ne01=1,.ne02=64,.ne03=1,.nb01=64u*2048,.nb02=2048,.nb03=64u*2048,
        .ne11=keys,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
        .ns20=512,.nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,
        .nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes,.ne1=64,.ne2=1,.ne3=1,.scale=1.0f/sqrtf(512)};
    const myds_metal_buffer_t *b[]={q,s->kv_f16,s->kv_f16,s->mask,sinks,s->pad,s->partial};
    const myds_metal_constant_t c[]={{400,true,1},{401,true,1},{402,true,0},{403,true,0},{404,true,keys%32!=0},
        {405,true,0},{420,false,512},{421,false,512},{422,false,1},{423,false,32}};
    if (register_mode) {
        const myds_metal_buffer_t *register_buffers[]={q,s->kv_f16,s->kv_f16,s->mask,sinks,s->pad,heads};
        const uint64_t dispatch_before=myds_metal_dispatch_count();
        int ok;
        if (rope) {
            struct attention_rope_fuse_args fused={.attention=a,.pos=rope->pos,.inverse=1u};
            memcpy(fused.frequencies,rope_frequencies[rope->compressed?1:0],sizeof(fused.frequencies));
            ok=specialized("kernel_myds_attention_register_rope_fuse_f16_512",
                &fused,sizeof(fused),register_buffers,7,64,1,1,256,1,1,22400,c,10);
        } else ok=specialized("kernel_myds_flash_attn_vec_packed8_register_reduce_f16_dk512_dv512",
            &a,sizeof(a),register_buffers,7,64,1,1,256,1,1,22400,c,10);
        if(myds_metal_dispatch_count()>dispatch_before) {
            if (rope) attention_rope_count(RF_ENCODED);
            attention_register_count(AR_ENCODED);
            attention_register_count(AR_STAGED);
        }
        if(!ok) MYDS_LOG_E("explicit packed attention register candidate unavailable or failed");
        return ok; /* Explicit candidate never retries another attention kernel. */
    }
    if ((packed_nsg != 8 || get_attn_packed_reduce_mode()) && keys <= 1024) {
        const myds_metal_buffer_t *pb[] = {q, s->kv_f16, s->kv_f16, s->mask, sinks, s->pad, heads};
        if (packed_nsg != 8) {
            const char *kernel = packed_nsg == 4
                ? "kernel_myds_flash_attn_vec_packed4_reduce_f16_dk512_dv512"
                : "kernel_myds_flash_attn_vec_packed16_reduce_f16_dk512_dv512";
            /* Generic dispatch validates the actual specialized PSO thread
             * limit and static + dynamic memory before encoding any work. */
            const int ok = specialized(kernel, &a, sizeof(a), pb, 7, 64, 1, 1,
                32u * (uint32_t)packed_nsg, 1, 1, ATTN_PACKED_SMEM((uint32_t)packed_nsg), c, 10);
            if (!ok) MYDS_LOG_E("packed attention NSG=%d candidate unavailable; refusing fallback", packed_nsg);
            else log_attn_packed_candidate_encoded(packed_nsg, kernel);
            return ok;
        }
        if (specialized("kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512",
                        &a, sizeof(a), pb, 7, 64, 1, 1, 256, 1, 1, 24448, c, 10)) {
            return 1;
        }
    }
    int vec_ok = 0;
    if(s->attn_vec_head_first && !s->attn_vec_head_first_unavailable) {
        vec_ok = specialized("kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",&a,sizeof(a),b,7,64,1,32,32,1,1,3328,c,10);
    }
    if(!vec_ok) {
        if(!specialized("kernel_flash_attn_ext_vec_f16_dk512_dv512",&a,sizeof(a),b,7,1,64,32,32,1,1,3328,c,10))return 0;
    }
    if(!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(s->partial);
    const int32_t nrows=64;const myds_metal_buffer_t *rb[]={s->partial,heads};
    const myds_metal_constant_t rc[]={{500,false,512},{501,false,32}};
    return specialized("kernel_flash_attn_ext_vec_reduce",&nrows,sizeof(nrows),rb,2,64,1,1,1024,1,1,0,rc,2);
}

int myds_v41_gpu_attention(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,myds_v41_gpu_scratch_t *s) {
    const int mode=myds_v41_gpu_attention_packed_register_enabled();
    if(mode) attention_register_count(AR_REQUESTS);
    if(mode<0) {attention_register_count(AR_FAILURES);return 0;}
    const int ok=attention_staged_impl(heads,sinks,q,window,n_raw,raw_start,selected_kv,selected_rows,s,mode,NULL);
    if(mode && !ok) attention_register_count(AR_FAILURES);
    return ok;
}

static int attention_direct_impl(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *selected_ids,
    uint32_t source_rows,uint32_t selected_rows,myds_v41_gpu_scratch_t *s,int register_mode,
    const struct attention_rope_scope *rope) {
    const int kv_reuse_mode = rope ? myds_v41_attn_kv_reuse_enabled() : 0;
    if (kv_reuse_mode < 0) return 0;
    const bool want_kv_reuse = (kv_reuse_mode == 1);
    if (!want_kv_reuse) {
        myds_v41_gpu_attn_kv_reuse_reset(s);
    } else {
        atomic_fetch_add_explicit(&attn_kv_reuse_requests, 1u, memory_order_relaxed);
    }
    if(!s||!n_raw||n_raw>128||raw_start>=128||selected_rows>512||
        !floats(heads,64u*512)||!floats(q,64u*512)||!floats(sinks,64)||!floats(window,128u*512)||
        (selected_rows&&(selected_rows>source_rows||!bf16s(comp_kv,(uint64_t)source_rows*512)||!floats(selected_ids,selected_rows)))) {
        if (want_kv_reuse) atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
        return 0;
    }

    const bool window_prestaged = want_kv_reuse && s->direct_window_prestaged;
    const bool scratch_fenced = want_kv_reuse && s->direct_scratch_fenced;
    const bool populate_kv_cache = want_kv_reuse && s->direct_populate_kv_cache;
    const bool expect_kv_reuse = want_kv_reuse && s->direct_expect_kv_reuse;
    myds_metal_buffer_t *deferred_q = want_kv_reuse ? s->direct_deferred_q : NULL;
    const bool deferred_q_compressed = want_kv_reuse && s->direct_deferred_q_compressed;
    const uint32_t deferred_q_rope_done = (want_kv_reuse && deferred_q) ? s->direct_deferred_q_rope_done : 0u;
    if (s) {
        s->direct_window_prestaged = false;
        s->direct_scratch_fenced = false;
        s->direct_populate_kv_cache = false;
        s->direct_expect_kv_reuse = false;
        s->direct_deferred_q_compressed = false;
        s->direct_deferred_q = NULL;
        s->direct_deferred_qb_weight = NULL;
        s->direct_deferred_qb_input = NULL;
        s->direct_deferred_q_pos = 0;
        s->direct_deferred_qb_done = 0;
        s->direct_deferred_q_rope_done = 0;
    }

    /* Only an actual ordinary direct RF scope selects this candidate. Public
     * direct/staged/trace callers continue to use the two original shaders. */
    const int pack_fuse=rope ? myds_v41_gpu_attention_direct_pack_fuse_enabled() : 0;
    if(pack_fuse) {
        atomic_fetch_add_explicit(&attention_pack_counters[AP_REQUESTS],1u,memory_order_relaxed);
        if(pack_fuse<0 || register_mode!=1 || want_kv_reuse ||
           !attention_direct_pack_preflight(s->kv_f16,window,n_raw,raw_start,
                                            comp_kv,selected_ids,source_rows,selected_rows)) {
            if (want_kv_reuse) atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
            return attention_pack_failure();
        }
    }

    const int packed_nsg = get_attn_packed_physical_nsg();
    if (!packed_nsg) {
        if (want_kv_reuse) atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
        return 0;
    }
    if (packed_nsg != 8 && !get_attn_packed_reduce_mode()) {
        MYDS_LOG_E("explicit MYDS_ATTN_PACKED_NSG=%d requires packed reduce enabled", packed_nsg);
        if (want_kv_reuse) atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
        return 0;
    }
    if (want_kv_reuse && (register_mode != 1 || packed_nsg != 8 || myds_metal_get_auto_barrier() ||
        !scratch_fenced || (deferred_q && (deferred_q != q || expect_kv_reuse || !selected_rows || !rope || deferred_q_rope_done > 64u)))) {
        myds_v41_gpu_attn_kv_reuse_reset(s);
        atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
        return 0;
    }

    uint32_t keys=n_raw+selected_rows;uint64_t kvbytes=(uint64_t)keys*1024,maskbytes=(uint64_t)keys*2;
    if (want_kv_reuse) {
        if (expect_kv_reuse) {
            const bool cache_hit = selected_rows > 0u &&
                s->direct_kv_cache_valid &&
                s->direct_kv_comp_buf == comp_kv &&
                s->direct_kv_comp_offset == myds_metal_buffer_offset(comp_kv) &&
                s->direct_kv_ids_buf == selected_ids &&
                s->direct_kv_ids_offset == myds_metal_buffer_offset(selected_ids) &&
                s->direct_kv_n_raw == n_raw &&
                s->direct_kv_source_rows == source_rows &&
                s->direct_kv_selected_rows == selected_rows &&
                s->direct_kv_pos == rope->pos;
            if (!cache_hit) {
                myds_v41_gpu_attn_kv_reuse_reset(s);
                atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                return 0;
            }
            if (!window_prestaged) {
                const uint32_t w_args[] = {n_raw, raw_start};
                const myds_metal_buffer_t *wb[] = {window, s->kv_f16};
                if (!dispatch("kernel_dsv41_copy_window_f32_to_f16", w_args, sizeof(w_args), wb, 2, 1, n_raw, 1, 128, 1, 1, 0)) {
                    myds_v41_gpu_attn_kv_reuse_reset(s);
                    atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                    return 0;
                }
                myds_metal_barrier_buffer(s->kv_f16);
            }
            if (keys % 32u) {
                if (!((keys & ~31u) >= n_raw && s->direct_pad_cache_valid)) {
                    struct flash_pad_args a={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
                        .nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,.nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes};
                    const myds_metal_buffer_t *b[]={s->kv_f16,s->kv_f16,s->mask,s->pad};
                    const myds_metal_constant_t c[]={{100,true,1},{125,false,32}};
                    if (!specialized("kernel_flash_attn_ext_pad",&a,sizeof(a),b,4,32,1,1,32,1,1,0,c,2)) {
                        myds_v41_gpu_attn_kv_reuse_reset(s);
                        atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                        return 0;
                    }
                    myds_metal_barrier_buffer(s->pad);
                }
            }
        } else {
            if (!window_prestaged) {
                const uint32_t w_args[] = {n_raw, raw_start};
                const myds_metal_buffer_t *wb[] = {window, s->kv_f16};
                if (!dispatch("kernel_dsv41_copy_window_f32_to_f16", w_args, sizeof(w_args), wb, 2, 1, n_raw, 1, 128, 1, 1, 0)) {
                    myds_v41_gpu_attn_kv_reuse_reset(s);
                    atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                    return 0;
                }
            }
            if (selected_rows) {
                struct {
                    uint32_t source_rows;
                    uint32_t selected_rows;
                    uint32_t dst_row_offset;
                    uint32_t pad;
                } g_args = {source_rows, selected_rows, n_raw, 0};
                const myds_metal_buffer_t *gb[] = {comp_kv, selected_ids, s->kv_f16};
                if (!dispatch("kernel_dsv41_get_rows_bf16_to_f16", &g_args, sizeof(g_args), gb, 3, 1, selected_rows, 1, 128, 1, 1, 0)) {
                    myds_v41_gpu_attn_kv_reuse_reset(s);
                    atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                    return 0;
                }
            }
            const uint32_t rem_rope = (deferred_q && deferred_q_rope_done < 64u)
                ? (64u - deferred_q_rope_done) : 0u;
            if (keys % 32u) {
                if (!window_prestaged || selected_rows) {
                    myds_metal_barrier_buffer(s->kv_f16);
                }
                if (rem_rope) {
                    if (!myds_v41_gpu_rope_heads_slice(s, deferred_q, deferred_q_rope_done,
                            rem_rope, rope->pos, deferred_q_compressed)) {
                        myds_v41_gpu_attn_kv_reuse_reset(s);
                        atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                        return 0;
                    }
                }
                struct flash_pad_args a={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
                    .nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,.nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes};
                const myds_metal_buffer_t *b[]={s->kv_f16,s->kv_f16,s->mask,s->pad};
                const myds_metal_constant_t c[]={{100,true,1},{125,false,32}};
                if (!specialized("kernel_flash_attn_ext_pad",&a,sizeof(a),b,4,32,1,1,32,1,1,0,c,2)) {
                    myds_v41_gpu_attn_kv_reuse_reset(s);
                    atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                    return 0;
                }
                if (deferred_q) {
                    const myds_metal_buffer_t *pad_deps[] = {s->pad, deferred_q};
                    myds_metal_barrier_buffers(pad_deps, 2u);
                } else {
                    myds_metal_barrier_buffer(s->pad);
                }
            } else {
                if (rem_rope) {
                    if (!myds_v41_gpu_rope_heads_slice(s, deferred_q, deferred_q_rope_done,
                            rem_rope, rope->pos, deferred_q_compressed)) {
                        myds_v41_gpu_attn_kv_reuse_reset(s);
                        atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
                        return 0;
                    }
                }
                if (deferred_q) {
                    const myds_metal_buffer_t *pack_deps[] = {s->kv_f16, deferred_q};
                    myds_metal_barrier_buffers(pack_deps, 2u);
                } else if (!window_prestaged || selected_rows) {
                    myds_metal_barrier_buffer(s->kv_f16);
                }
            }
            if (populate_kv_cache && selected_rows > 0u) {
                s->direct_kv_cache_valid = true;
                s->direct_pad_cache_valid = (keys % 32u != 0u) && ((keys & ~31u) >= n_raw);
                s->direct_kv_comp_buf = comp_kv;
                s->direct_kv_comp_offset = myds_metal_buffer_offset(comp_kv);
                s->direct_kv_ids_buf = selected_ids;
                s->direct_kv_ids_offset = myds_metal_buffer_offset(selected_ids);
                s->direct_kv_n_raw = n_raw;
                s->direct_kv_source_rows = source_rows;
                s->direct_kv_selected_rows = selected_rows;
                s->direct_kv_pos = rope->pos;
            }
        }
    } else {
        if (!myds_metal_get_auto_barrier()) {
            /* The previous attention may still READ these shared buffers in a
             * concurrent encoder. Fence before staging/pad/partial writes (WAR). */
            const myds_metal_buffer_t *scratch_buffers[] = {s->kv_f16, s->pad, s->partial};
            myds_metal_barrier_buffers(scratch_buffers, 3);
        }
        if(pack_fuse) {
            /* Unused inputs for selected_rows0 bind a valid window resource. The
             * shader's uniform raw-row branch never reads these dummy bindings. */
            if(!attention_direct_pack_encode(s->kv_f16,window,n_raw,raw_start,
                                            comp_kv,selected_ids,source_rows,selected_rows))return 0;
        } else {
            const uint32_t w_args[] = {n_raw, raw_start};
            const myds_metal_buffer_t *wb[] = {window, s->kv_f16};
            if(!dispatch("kernel_dsv41_copy_window_f32_to_f16", w_args, sizeof(w_args), wb, 2, 1, n_raw, 1, 128, 1, 1, 0)) return 0;

            if(selected_rows) {
            struct {
                uint32_t source_rows;
                uint32_t selected_rows;
                uint32_t dst_row_offset;
                uint32_t pad;
            } g_args = {source_rows, selected_rows, n_raw, 0};
            const myds_metal_buffer_t *gb[] = {comp_kv, selected_ids, s->kv_f16};
                if(!dispatch("kernel_dsv41_get_rows_bf16_to_f16", &g_args, sizeof(g_args), gb, 3, 1, selected_rows, 1, 128, 1, 1, 0)) return 0;
            }
        }
        if(!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(s->kv_f16);

        if(keys%32) {
            struct flash_pad_args a={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
                .nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,.nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes};
            const myds_metal_buffer_t *b[]={s->kv_f16,s->kv_f16,s->mask,s->pad};
            const myds_metal_constant_t c[]={{100,true,1},{125,false,32}};
            if(!specialized("kernel_flash_attn_ext_pad",&a,sizeof(a),b,4,32,1,1,32,1,1,0,c,2))return 0;
            /* Packed and original vec both READ the generated pad K/V and mask. */
            if(!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(s->pad);
        }
    }
    struct flash_vec_args a={.ne01=1,.ne02=64,.ne03=1,.nb01=64u*2048,.nb02=2048,.nb03=64u*2048,
        .ne11=keys,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
        .ns20=512,.nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,
        .nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes,.ne1=64,.ne2=1,.ne3=1,.scale=1.0f/sqrtf(512)};
    const myds_metal_buffer_t *b[]={q,s->kv_f16,s->kv_f16,s->mask,sinks,s->pad,s->partial};
    const myds_metal_constant_t c[]={{400,true,1},{401,true,1},{402,true,0},{403,true,0},{404,true,keys%32!=0},
        {405,true,0},{420,false,512},{421,false,512},{422,false,1},{423,false,32}};
    if (register_mode) {
        const myds_metal_buffer_t *register_buffers[]={q,s->kv_f16,s->kv_f16,s->mask,sinks,s->pad,heads};
        const uint64_t dispatch_before=myds_metal_dispatch_count();
        int ok;
        if (rope) {
            struct attention_rope_fuse_args fused={.attention=a,.pos=rope->pos,.inverse=1u};
            memcpy(fused.frequencies,rope_frequencies[rope->compressed?1:0],sizeof(fused.frequencies));
            ok=specialized("kernel_myds_attention_register_rope_fuse_f16_512",
                &fused,sizeof(fused),register_buffers,7,64,1,1,256,1,1,22400,c,10);
        } else ok=specialized("kernel_myds_flash_attn_vec_packed8_register_reduce_f16_dk512_dv512",
            &a,sizeof(a),register_buffers,7,64,1,1,256,1,1,22400,c,10);
        if(myds_metal_dispatch_count()>dispatch_before) {
            if (rope) attention_rope_count(RF_ENCODED);
            if (want_kv_reuse) atomic_fetch_add_explicit(&attn_kv_reuse_encoded, 1u, memory_order_relaxed);
            attention_register_count(AR_ENCODED);
            attention_register_count(AR_DIRECT);
        } else if (want_kv_reuse) {
            myds_v41_gpu_attn_kv_reuse_reset(s);
            atomic_fetch_add_explicit(&attn_kv_reuse_failures, 1u, memory_order_relaxed);
        }
        if(!ok) MYDS_LOG_E("explicit packed attention register candidate unavailable or failed");
        return ok; /* Explicit candidate never retries another attention kernel. */
    }
    if ((packed_nsg != 8 || get_attn_packed_reduce_mode()) && keys <= 1024) {
        const myds_metal_buffer_t *pb[] = {q, s->kv_f16, s->kv_f16, s->mask, sinks, s->pad, heads};
        if (packed_nsg != 8) {
            const char *kernel = packed_nsg == 4
                ? "kernel_myds_flash_attn_vec_packed4_reduce_f16_dk512_dv512"
                : "kernel_myds_flash_attn_vec_packed16_reduce_f16_dk512_dv512";
            /* Generic dispatch validates the actual specialized PSO thread
             * limit and static + dynamic memory before encoding any work. */
            const int ok = specialized(kernel, &a, sizeof(a), pb, 7, 64, 1, 1,
                32u * (uint32_t)packed_nsg, 1, 1, ATTN_PACKED_SMEM((uint32_t)packed_nsg), c, 10);
            if (!ok) MYDS_LOG_E("packed attention NSG=%d candidate unavailable; refusing fallback", packed_nsg);
            else log_attn_packed_candidate_encoded(packed_nsg, kernel);
            return ok;
        }
        if (specialized("kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512",
                        &a, sizeof(a), pb, 7, 64, 1, 1, 256, 1, 1, 24448, c, 10)) {
            return 1;
        }
    }
    int vec_ok = 0;
    if(s->attn_vec_head_first && !s->attn_vec_head_first_unavailable) {
        vec_ok = specialized("kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",&a,sizeof(a),b,7,64,1,32,32,1,1,3328,c,10);
    }
    if(!vec_ok) {
        if(!specialized("kernel_flash_attn_ext_vec_f16_dk512_dv512",&a,sizeof(a),b,7,1,64,32,32,1,1,3328,c,10))return 0;
    }
    if(!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(s->partial);
    const int32_t nrows=64;const myds_metal_buffer_t *rb[]={s->partial,heads};
    const myds_metal_constant_t rc[]={{500,false,512},{501,false,32}};
    return specialized("kernel_flash_attn_ext_vec_reduce",&nrows,sizeof(nrows),rb,2,64,1,1,1024,1,1,0,rc,2);
}

int myds_v41_gpu_attention_direct(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *selected_ids,
    uint32_t source_rows,uint32_t selected_rows,myds_v41_gpu_scratch_t *s) {
    const int mode=myds_v41_gpu_attention_packed_register_enabled();
    if(mode) attention_register_count(AR_REQUESTS);
    if(mode<0) {attention_register_count(AR_FAILURES);return 0;}
    const int ok=attention_direct_impl(heads,sinks,q,window,n_raw,raw_start,comp_kv,selected_ids,source_rows,selected_rows,s,mode,NULL);
    if(mode && !ok) attention_register_count(AR_FAILURES);
    return ok;
}

int myds_v41_gpu_attention_rope_fuse(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,
    myds_v41_gpu_scratch_t *s,uint32_t pos,bool compressed,
    bool ordinary_allowed,bool trace_on,bool *rope_consumed) {
    const int mode=myds_v41_attention_rope_fuse_enabled();
    if (rope_consumed) *rope_consumed=false;
    if (!mode || (mode>0 && (!ordinary_allowed || trace_on))) {
        if (!rope_consumed) return 0;
        return myds_v41_gpu_attention(heads,sinks,q,window,n_raw,raw_start,selected_kv,selected_rows,s);
    }
    attention_rope_count(RF_REQUESTS);
    if (mode<0 || !rope_consumed || !attention_rope_validate(heads,sinks,q,window,n_raw,raw_start,
        selected_kv,NULL,selected_rows,selected_rows,s,pos,false)) return attention_rope_failure();
    pthread_once(&rope_once,init_rope);
    const struct attention_rope_scope rope={pos,compressed};
    attention_register_count(AR_REQUESTS);
    const int ok=attention_staged_impl(heads,sinks,q,window,n_raw,raw_start,selected_kv,selected_rows,s,1,&rope);
    if (!ok) {attention_register_count(AR_FAILURES);return attention_rope_failure();}
    *rope_consumed=true;
    return 1;
}
int myds_v41_gpu_attention_direct_rope_fuse(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *selected_ids,
    uint32_t source_rows,uint32_t selected_rows,myds_v41_gpu_scratch_t *s,
    uint32_t pos,bool compressed,bool ordinary_allowed,bool trace_on,bool *rope_consumed) {
    const int mode=myds_v41_attention_rope_fuse_enabled();
    const int kv_reuse_mode=myds_v41_attn_kv_reuse_enabled();
    if (rope_consumed) *rope_consumed=false;
    if (kv_reuse_mode < 0) return 0;
    if (!mode || (mode>0 && (!ordinary_allowed || trace_on))) {
        if (kv_reuse_mode == 1 && ordinary_allowed && !trace_on) {
            myds_v41_gpu_attn_kv_reuse_reset(s);
            myds_v41_attn_kv_reuse_record_prestage_failure();
            return 0;
        }
        if (!rope_consumed) return 0;
        return myds_v41_gpu_attention_direct(heads,sinks,q,window,n_raw,raw_start,comp_kv,selected_ids,source_rows,selected_rows,s);
    }
    attention_rope_count(RF_REQUESTS);
    if (mode<0 || !rope_consumed || !attention_rope_validate(heads,sinks,q,window,n_raw,raw_start,
        comp_kv,selected_ids,source_rows,selected_rows,s,pos,true)) {
        if (kv_reuse_mode == 1) {
            myds_v41_gpu_attn_kv_reuse_reset(s);
            myds_v41_attn_kv_reuse_record_prestage_failure();
        }
        return attention_rope_failure();
    }
    pthread_once(&rope_once,init_rope);
    const struct attention_rope_scope rope={pos,compressed};
    attention_register_count(AR_REQUESTS);
    const int ok=attention_direct_impl(heads,sinks,q,window,n_raw,raw_start,comp_kv,selected_ids,source_rows,selected_rows,s,1,&rope);
    if (!ok) {attention_register_count(AR_FAILURES);return attention_rope_failure();}
    *rope_consumed=true;
    return 1;
}
