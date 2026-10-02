#include "myds_v41_gpu.h"
#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* Chunked prefill geometry.  The slab holds, in order, the carry of the previous
 * 127 KV vectors, this chunk's tokens, and the compressed table prefix; the mask
 * block holds one uint16 row per token. */
#define MYDS_CHUNK_CARRY      127u
/* 单个 chunk 的 token 上界。MoE 每层要读【全部 384 个 expert】的权重（3.79 GB），
 * 而这份开销与 chunk 内的 token 数无关 —— 只与 chunk 的【个数】有关。
 * 512 时一个 4404 token 的 prompt 要 9 个 chunk（=9 遍 expert 权重，1.38 TB），
 * 2048 只需 3 个。所以这个值直接决定 MoE 的总量，不该卡在 512。 */
#define MYDS_CHUNK_MAXC       MYDS_V41_CHUNK_MAX
/* 压缩键的上界（键数，含 carry 与本 chunk）。
 * ratio_for 对 l>=20 的层返回 1，此时一个 chunk 的 n_comp 就是【绝对位置】，
 * 所以键范围 = CARRY + c_len + n_comp 会随上下文线性增长。
 * 历史上这个值被写死过两次（4608 → 16384），每次都是一个「宣称的上下文 > 实际能跑的」
 * 缺口：超了之后第 20 层 attention_chunk 返回 0 ⇒ 整个 prefill 失败，而且注释还谎称
 * 「会回退逐 token」（代码里根本没有回退）。
 * 现在：默认【完整覆盖配置的上下文】（不再有「配置 N、实际只能跑 N-2175」的小缺口），
 * 只封顶在 KEY_MAX_DEFAULT，env MYDS_CHUNK_KEY_MAX 可调（调小省内存，调大覆盖更长的上下文）。
 * 代价是内存：mask_chunk = MAXC * mask_u16 * 2 字节，kv slab = slab_rows * 1024 字节。
 *   ctx 262144 ⇒ mask 约 1.03 GB + slab 约 258 MB（配置了这么长上下文就该付这个钱）
 *   ctx   8192 ⇒ 合计约 42 MB
 * 超上限的 prompt 在 prefill【开始前】被明确拒绝（不再跑到第 20 层才炸）。 */
#define MYDS_CHUNK_KEY_MAX_DEFAULT 262144u
#define MYDS_CHUNK_KEY_MAX_LIMIT  4194304u


struct myds_v41_gpu_scratch {
    uint32_t max_comp;
    myds_metal_buffer_t *sort_a, *sort_b, *kv_f16, *mask, *pad, *partial;
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
    uint32_t slab_rows;      /* kv slab 行数 = CARRY + MAXC + key_comp_max */
    uint32_t keys_clamp;     /* 诊断：MYDS_ATTN_KEYS_CLAMP 缓存 */
};
static int floats(const myds_metal_buffer_t *b,uint64_t n) {
    return b && n && n<=myds_metal_buffer_size(b)/4 && myds_metal_buffer_offset(b)%4==0;
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
uint32_t myds_v41_gpu_chunk_max_c(const myds_v41_gpu_scratch_t *s) {
    return s ? MYDS_CHUNK_MAXC : 0u;
}
uint32_t myds_v41_gpu_chunk_key_max(const myds_v41_gpu_scratch_t *s) {
    return s ? s->key_comp_max : 0u;
}
myds_v41_gpu_scratch_t *myds_v41_gpu_scratch_create(uint32_t max_comp) {
    if(!max_comp || max_comp>1048576) return NULL;
    myds_v41_gpu_scratch_t *s=calloc(1,sizeof(*s));
    if(!s)return NULL;
    s->max_comp=max_comp;
    const char *kc = getenv("MYDS_ATTN_KEYS_CLAMP");
    if(kc) s->keys_clamp = (uint32_t)strtoul(kc, NULL, 10);
    /* chunk attention 的几何量：键上界跟随配置的上下文，并封顶（env 可覆盖）。
     * 封顶的理由是内存：mask_chunk = MAXC * mask_u16 * 2 字节，而 mask_u16 ≈ 键上界。 */
    {
        uint32_t key_max=MYDS_CHUNK_KEY_MAX_DEFAULT;
        const char *km=getenv("MYDS_CHUNK_KEY_MAX");
        if(km){ long v=atol(km); if(v>0&&(unsigned long)v<=MYDS_CHUNK_KEY_MAX_LIMIT) key_max=(uint32_t)v; }
        if(key_max>max_comp) key_max=max_comp;   /* 不必超过配置的上下文 */
        s->key_comp_max=key_max;
        if(s->key_comp_max<2u||s->key_comp_max>MYDS_CHUNK_KEY_MAX_LIMIT){ free(s); return NULL; }
        s->slab_rows=MYDS_CHUNK_CARRY+MYDS_CHUNK_MAXC+s->key_comp_max;
        s->mask_u16=(s->slab_rows+31u)/32u*32u;  /* 32 对齐 ⇒ keys_pad 永远落在行内 */
        if(s->mask_u16<s->slab_rows){ free(s); return NULL; }   /* 溢出保护 */
    }
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
    myds_metal_buffer_free(s->kv_f16);myds_metal_buffer_free(s->mask);
    myds_metal_buffer_free(s->kv_chunk);myds_metal_buffer_free(s->mask_chunk);
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
int myds_v41_gpu_quant(myds_metal_buffer_t *x,uint32_t width,uint32_t rows,uint32_t mode) {
    uint32_t block=mode==3?16:32;
    if(!width||!rows||mode>3||(mode&&width%block)||!floats(x,(uint64_t)width*rows))return 0;
    if(!mode && (uint64_t)width*rows<=UINT32_MAX && myds_metal_buffer_offset(x)%16==0)
        return myds_metal_v41_bf16(x,width*rows);
    const uint32_t a[]={width,rows,mode};const myds_metal_buffer_t *b[]={x};
    return dispatch("kernel_dsv41_quantize",a,sizeof(a),b,1,(width+block-1)/block,rows,1,32,1,1,0);
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
int myds_v41_gpu_indexscores(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,uint32_t n_heads) {
    if(!n_heads||!floats(scores,n_comp)||!floats(q,(uint64_t)n_heads*128)||
        !floats(weights,n_heads)||!floats(cache,(uint64_t)n_comp*128))return 0;
    struct {uint32_t n_rows,n_head,head_dim,cache_f16;float scale;} a={n_comp,n_heads,128,0,1.0f/64.0f};
    const myds_metal_buffer_t *b[]={q,weights,cache,scores};
    return n_heads==32 ? dispatch("kernel_glm_indexer_score_one_direct",&a,sizeof(a),b,4,n_comp,1,1,32,4,1,132u*4)
        : dispatch("kernel_glm_indexer_score_one",&a,sizeof(a),b,4,n_comp,1,1,32,1,1,32u*4);
}
/* 整 chunk 批量版：kernel_glm_indexer_score_chunk_direct 与 _direct 的线程/网格/smem
 * 相同，只是多一维 token（grid.y）与逐行因果门。只支持 n_heads == 32，
 * 否则返回 0 让调用方回退到逐 token 路径。 */
int myds_v41_gpu_indexscores_chunk(myds_metal_buffer_t *scores,const myds_metal_buffer_t *q,
    const myds_metal_buffer_t *weights,const myds_metal_buffer_t *cache,uint32_t n_comp,
    uint32_t n_heads,uint32_t n_tokens,uint32_t causal_start,uint32_t causal_ratio) {
    if(n_heads!=32u||!n_comp||!n_tokens||!causal_ratio||causal_start==UINT32_MAX)return 0;
    if(!floats(scores,(uint64_t)n_comp*n_tokens)||
       !floats(q,(uint64_t)n_tokens*n_heads*128u)||
       !floats(weights,(uint64_t)n_tokens*n_heads)||!floats(cache,(uint64_t)n_comp*128u))return 0;
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
            .index_row_stride = (uint64_t)128u * sizeof(float),
            .score_token_stride = (uint64_t)n_comp * sizeof(float),
            .scale = 1.0f / 64.0f
        };
        const uint32_t gx = (n_comp + 31u) / 32u;
        const uint32_t gy = (n_tokens + 7u) / 8u;
        const size_t smem = (8u * 128u + 32u * 128u + 8u * 32u) * sizeof(float);
        return dispatch("kernel_dsv4_indexer_scores_tiled_f32", &ta, sizeof(ta), b, 4,
                        gx, gy, 1, 32, 4, 1, smem);
    }
    struct {uint32_t n_rows,n_head,head_dim,cache_f16;float scale;
            uint32_t n_tokens,causal_start,causal_ratio,pad;} a={
        n_comp,n_heads,128,0,1.0f/64.0f,n_tokens,causal_start,causal_ratio,0};
    return dispatch("kernel_glm_indexer_score_chunk_direct",&a,sizeof(a),b,4,
        n_comp,n_tokens,1,32,4,1,132u*4);
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
    struct sort_args a={.ne00=n_cols,.ne01=n_rows,.ne02=1,.ne03=1,.nb00=4,
        .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*n_rows*4,.nb03=(uint64_t)n_cols*n_rows*4,
        .ne0=layout,.ne1=n_rows,.ne2=1,.ne3=1,.top_k=blocktop,
        .causal_start=causal_start,.causal_ratio=causal_ratio};
    const myds_metal_buffer_t *b[]={scores,npr==1?ids:tmp_a};
    if(!dispatch(causal_ratio?"kernel_argsort_f32_i32_desc_causal":"kernel_argsort_f32_i32_desc",
        &a,sizeof(a),b,2,npr*n_rows,1,1,nth,1,1,((size_t)nth*8+15)&~(size_t)15))return 0;
    myds_metal_buffer_t *cur=tmp_a,*next=tmp_b;
    for(uint32_t len=blocktop;len<layout;len*=2) {
        uint32_t nm=(layout+2*len-1)/(2*len);
        struct merge_args m={.ne00=n_cols,.ne01=n_rows,.ne02=1,.ne03=1,.nb00=4,
            .nb01=(uint64_t)n_cols*4,.nb02=(uint64_t)n_cols*n_rows*4,.nb03=(uint64_t)n_cols*n_rows*4,
            .ne0=layout,.ne1=n_rows,.ne2=1,.ne3=1,.top_k=nm==1?top:layout,.len=len,.block_width=nth,
            .block_top_k=blocktop,.causal_start=causal_start,.causal_ratio=causal_ratio};
        const myds_metal_buffer_t *mb[]={scores,cur,nm==1?ids:next};
        if(!dispatch(causal_ratio?"kernel_argsort_merge_f32_i32_desc_causal":
                     "kernel_argsort_merge_f32_i32_desc",
            &m,sizeof(m),mb,3,nm*n_rows,1,1,len<512?len:512,1,1,0))return 0;
        myds_metal_buffer_t *tmp=cur;cur=next;next=tmp;
    }
    return 1;
}
int myds_v41_gpu_topk(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,myds_v41_gpu_scratch_t *s) {
    return topk_rows_causal(ids,scores,n_cols,top,1,0,0,s);
}
int myds_v41_gpu_topk_causal(myds_metal_buffer_t *ids,const myds_metal_buffer_t *scores,
    uint32_t n_cols,uint32_t top,uint32_t n_rows,uint32_t causal_start,uint32_t causal_ratio,
    myds_v41_gpu_scratch_t *s) {
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
    return dispatch("kernel_dsv4_topk_mask",&a,sizeof(a),b,2,
               (uint32_t)((clear+255u)/256u),1,1,256,1,1,0)&&
        dispatch("kernel_dsv4_topk_mask_scatter",&a,sizeof(a),b,2,
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

/* Use transient subviews only as binding handles; Metal retains the underlying
 * resources until queued dispatches finish. No CPU readback or synchronization. */
static int stage_f16(myds_metal_buffer_t *dst,size_t dstoff,const myds_metal_buffer_t *src,size_t srcoff,uint32_t count,myds_v41_gpu_scratch_t *s) {
    if(!s||!src||!dst||!myds_metal_buffer_update_sub_view(s->sv_view,src,srcoff,(size_t)count*4)||
       !myds_metal_buffer_update_sub_view(s->dv_view,dst,dstoff,(size_t)count*2)) return 0;
    const myds_metal_buffer_t *b[]={s->sv_view,s->dv_view};
    return dispatch("kernel_cpy_contig_f32_f16_4",&count,sizeof(count),b,2,(count+1023u)/1024,1,1,256,1,1,0);
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

/* B 方案：见 metal/myds_chunk.metal 里的 myds_attn_gather_args（必须逐字段对齐）。 */
struct myds_attn_gather_args {
    uint32_t rows, top_k, n_comp, valid_from, mask_stride, c_start, base_pos, ratio;
};

/* 进程级开关：默认【关】。开了才走 gather 路径。 */
static int attn_gather_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("MYDS_ATTN_GATHER");
        cached = (v && atoi(v) > 0) ? 1 : 0;
    }
    return cached;
}
/* 建窗口/因果 mask 模板。列 [0,127) 是 carry，[127,127+c_len) 是本 chunk。
 * 行 r（chunk 内第 r 个 token）能看的是 slab 列 [r, r+127]：位置不超过自己（因果）
 * 且不早于自己往前 128 个（窗口）。carry 部分还要满足"那个位置真实存在"。
 * 模板只在 (c_len, carry) 变化时重建 —— 完整的 chunk 形状相同，所以整个 prefill 只建一两次。 */
static int build_chunk_mask_template(myds_v41_gpu_scratch_t *s,uint32_t c_len,uint32_t carry) {
    if(c_len==0||c_len>MYDS_CHUNK_MAXC||carry>MYDS_CHUNK_CARRY)return 0;
    uint16_t *m=(uint16_t *)myds_metal_buffer_data(s->mask_chunk);
    if(!m)return 0;
    const uint32_t cols=MYDS_CHUNK_CARRY+c_len;
    for(uint32_t r=0;r<c_len;r++){
        uint16_t *row=m+(size_t)r*s->mask_u16;
        for(uint32_t c=0;c<cols;c++){
            int ok=(c>=r)&&(c<=MYDS_CHUNK_CARRY+r);
            if(ok&&c<MYDS_CHUNK_CARRY)ok=(c>=MYDS_CHUNK_CARRY-carry);  /* carry 里只放了真实存在的位置 */
            row[c]=ok?0u:0xfc00u;
        }
        /* 压缩区（含 32 对齐的尾部）【不在这里】清理：它每个 chunk 都由
         * kernel_myds_mask_chunk_fill 先置为不可见、再由 scatter 打开本 chunk 的 picks。
         * 以前这里有一句「把 cols 之后全部清成不可见」—— mask_u16 还是编译期常量
         * （18560）时只是两万次/行；改成跟随上下文的运行时值（最大 264320）后变成
         * 24 万次/行 × 2048 行 ≈ 5 亿次，ttft 直接 +0.7s（实测 13.83 → 14.52）。
         * keys_pad 之后的列没有任何读者，所以现在只写 [0, cols) 就够。 */
    }
    s->tmpl_len=c_len;s->tmpl_carry=carry;
    return 1;
}

/* 整个 chunk 的 attention，一次 dispatch。省掉的正是每 token 的 staging/gather/pad/vec/reduce。 */
int myds_v41_gpu_attention_chunk(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *chunk_kv,
    const myds_metal_buffer_t *comp_kv,const myds_metal_buffer_t *ids,
    myds_metal_buffer_t *gathered,
    uint32_t c_len,uint32_t n_comp,uint32_t top_k,
    uint32_t base_pos,uint32_t c_start,uint32_t ratio,myds_v41_gpu_scratch_t *s) {
    if(!s||!heads||!q||!chunk_kv||c_len==0||c_len>MYDS_CHUNK_MAXC||top_k>512u||
       n_comp>s->key_comp_max||
       !floats(heads,(uint64_t)c_len*64u*512)||!floats(q,(uint64_t)c_len*64u*512)||
       !floats(sinks,64)||!floats(chunk_kv,(uint64_t)c_len*512)||
       (n_comp&&(!comp_kv||!floats(comp_kv,(uint64_t)n_comp*512)||
                 !ids||!floats(ids,(uint64_t)c_len*top_k))))return 0;

    /* 诊断：把压缩键数强行调小，【只影响 attention 读取的 K/V 量，不影响 dispatch 结构】，
     * 用来判断这 8 秒是不是 K/V 带宽 bound。结果不完整，仅计时。 */
    if(__builtin_expect(s->keys_clamp > 0, 0)) {
        if(s->keys_clamp < n_comp) n_comp = s->keys_clamp;
    }

    const uint32_t p0=base_pos+c_start;
    const uint32_t carry=p0<MYDS_CHUNK_CARRY?p0:MYDS_CHUNK_CARRY;
    if(s->tmpl_len!=c_len||s->tmpl_carry!=carry)
        if(!build_chunk_mask_template(s,c_len,carry))return 0;

    const uint32_t comp_base=MYDS_CHUNK_CARRY+c_len;
    const uint32_t keys=comp_base+n_comp;
    /* The kernel walks keys in blocks of 32 and skips a block whose mask row is all
     * invisible, so the key count is rounded up and the extra columns stay masked
     * instead of being padded into a separate buffer. */
    const uint32_t keys_pad=(keys+31u)/32u*32u;

    if(!stage_f16(s->kv_chunk,(size_t)MYDS_CHUNK_CARRY*1024u,chunk_kv,0,c_len*512u,s))return 0;
    if(n_comp&&!stage_f16(s->kv_chunk,(size_t)comp_base*1024u,comp_kv,0,n_comp*512u,s))return 0;

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
    if(n_comp) {
        struct {uint32_t base,stride,n_comp,top_k,rows,c_start,base_pos,ratio;} ma={
            comp_base,s->mask_u16,n_comp,top_k,c_len,c_start,base_pos,ratio};
        const myds_metal_buffer_t *sb[]={s->mask_chunk,ids};
        if(!dispatch("kernel_myds_mask_chunk_scatter",&ma,sizeof(ma),sb,2,top_k,c_len,1,64,1,1,0))return 0;
    }

    /* B 方案（docs/B_MAX_SPARSE_ATTENTION_SPEC.md）：把本 query 可见的 640 个键
     * （128 滑窗 + 512 picks）gather 成各自的紧凑段，并重写 [0,640) 的掩码。
     * 在上面那套旧掩码【之后】跑，直接覆盖它 —— 内核只看 [0,640)，其余列无读者。 */
    const int use_gather = attn_gather_enabled() && gathered != NULL && n_comp != 0;
    if (use_gather) {
        struct myds_attn_gather_args ga={
            c_len,top_k,n_comp,MYDS_CHUNK_CARRY-carry,s->mask_u16,c_start,base_pos,ratio};
        {   const myds_metal_buffer_t *g1[]={gathered,s->mask_chunk,s->kv_chunk};
            if(!dispatch("kernel_myds_attn_gather_window",&ga,sizeof(ga),g1,3,
                         128u,c_len,1,128,1,1,0))return 0; }
        {   const myds_metal_buffer_t *g2[]={gathered,s->mask_chunk,comp_kv,ids};
            if(!dispatch("kernel_myds_attn_gather_comp",&ga,sizeof(ga),g2,4,
                         top_k,c_len,1,128,1,1,0))return 0; }
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
    if(!specialized("kernel_flash_attn_ext_vec_f16_dk512_dv512",&a,sizeof(a),b,7,
                    c_len,64,1,32,1,1,3328,c,10))return 0;

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

int myds_v41_gpu_attention(myds_metal_buffer_t *heads,const myds_metal_buffer_t *sinks,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *window,uint32_t n_raw,
    uint32_t raw_start,const myds_metal_buffer_t *selected_kv,uint32_t selected_rows,myds_v41_gpu_scratch_t *s) {
    if(!s||!n_raw||n_raw>128||raw_start>=128||selected_rows>512||
        !floats(heads,64u*512)||!floats(q,64u*512)||!floats(sinks,64)||!floats(window,128u*512)||
        (selected_rows&&!floats(selected_kv,(uint64_t)selected_rows*512)))return 0;
    uint32_t tail=128-raw_start<n_raw?128-raw_start:n_raw;
    if(!stage_f16(s->kv_f16,0,window,(size_t)raw_start*2048,tail*512,s))return 0;
    if(n_raw>tail&&!stage_f16(s->kv_f16,(size_t)tail*1024,window,0,(n_raw-tail)*512,s))return 0;
    if(selected_rows&&!stage_f16(s->kv_f16,(size_t)n_raw*1024,selected_kv,0,selected_rows*512,s))return 0;
    uint32_t keys=n_raw+selected_rows;uint64_t kvbytes=(uint64_t)keys*1024,maskbytes=(uint64_t)keys*2;
    if(keys%32) {
        struct flash_pad_args a={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
            .nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,.nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes};
        const myds_metal_buffer_t *b[]={s->kv_f16,s->kv_f16,s->mask,s->pad};
        const myds_metal_constant_t c[]={{100,true,1},{125,false,32}};
        if(!specialized("kernel_flash_attn_ext_pad",&a,sizeof(a),b,4,32,1,1,32,1,1,0,c,2))return 0;
    }
    struct flash_vec_args a={.ne01=1,.ne02=64,.ne03=1,.nb01=64u*2048,.nb02=2048,.nb03=64u*2048,
        .ne11=keys,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=1024,.nb12=kvbytes,.nb13=kvbytes,
        .ns20=512,.nb21=1024,.nb22=kvbytes,.nb23=kvbytes,.ne31=1,.ne32=1,.ne33=1,
        .nb31=maskbytes,.nb32=maskbytes,.nb33=maskbytes,.ne1=64,.ne2=1,.ne3=1,.scale=1.0f/sqrtf(512)};
    const myds_metal_buffer_t *b[]={q,s->kv_f16,s->kv_f16,s->mask,sinks,s->pad,s->partial};
    const myds_metal_constant_t c[]={{400,true,1},{401,true,1},{402,true,0},{403,true,0},{404,true,keys%32!=0},
        {405,true,0},{420,false,512},{421,false,512},{422,false,1},{423,false,32}};
    if(!specialized("kernel_flash_attn_ext_vec_f16_dk512_dv512",&a,sizeof(a),b,7,1,64,32,32,1,1,3328,c,10))return 0;
    const int32_t nrows=64;const myds_metal_buffer_t *rb[]={s->partial,heads};
    const myds_metal_constant_t rc[]={{500,false,512},{501,false,32}};
    return specialized("kernel_flash_attn_ext_vec_reduce",&nrows,sizeof(nrows),rb,2,64,1,1,1024,1,1,0,rc,2);
}
