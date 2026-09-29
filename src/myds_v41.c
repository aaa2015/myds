/* Native DeepSeek V4.1 text graph. Numerical ordering is checked against the
 * separately built, pinned ds4 oracle (MIT); no reference engine is linked.
 * Keep this graph diagnostic until the recorded whole-model gates pass. */
#include "myds_internal.h"
#include "myds_engram.h"
#include "myds_v41_gpu.h"
#include "myds_v41_moe.h"
#include "myds_cpu.h"
#include "myds_log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V41_LAYERS 40u
#define V41_DIM 5120u
#define V41_HEADS 64u
#define V41_HEAD_DIM 512u
#define V41_INDEX_HEADS 32u
#define V41_INDEX_DIM 128u
#define V41_WINDOW 128u
#define V41_TOPK 512u

typedef struct {
    myds_metal_buffer_t *attn_fn, *attn_scale, *attn_base;
    myds_metal_buffer_t *ffn_fn, *ffn_scale, *ffn_base;
    myds_metal_buffer_t *compressor_kv, *compressor_gate, *compressor_norm;
    myds_metal_buffer_t *index_k, *index_k_norm, *index_q, *index_proj;
} v41_layer_t;

struct myds_v41_model {
    v41_layer_t layers[V41_LAYERS];
    myds_engram_t *engram;
    myds_metal_buffer_t *engram_proj[2], *engram_q[2], *engram_k[2];
    float hc_eps;
};

struct myds_v41_context {
    myds_engram_t *engram;
    myds_v41_gpu_scratch_t *gpu;
    uint32_t capacity, next_position;
    bool valid;
    myds_metal_buffer_t *residual, *after_attn, *flat_norm, *mix;
    myds_metal_buffer_t *attn_split, *ffn_split, *pre;
    myds_metal_buffer_t *engram_rows[2], *engram_kv;
    myds_metal_buffer_t *window[V41_LAYERS];
    myds_metal_buffer_t *compressed[4], *index_cache[4];
    myds_metal_buffer_t *previous_kv[4], *previous_score[4];
    myds_metal_buffer_t *pool_kv, *pool_score, *latent;
    myds_metal_buffer_t *index_k, *index_q, *index_weights, *index_scores;
    myds_metal_buffer_t *selected, *selected_kv;
    myds_metal_buffer_t *block_scores, *block_selected, *block_mask;
};

static uint32_t ratio_for(uint32_t l) { return l < 2 ? 0 : l < 20 ? 2 : 1; }
static uint32_t owner_for(uint32_t l) { return l < 8 ? 0 : l < 14 ? 1 : l < 20 ? 2 : 3; }
static bool kv_source(uint32_t l) { return l == 2 || l == 8 || l == 14 || l == 20; }
static bool index_source(uint32_t l) { return kv_source(l) || (l >= 24 && l % 4 == 0); }

static bool metadata_u32(const gguf_file_t *gf, const char *suffix, uint32_t expected) {
    char key[160]; snprintf(key, sizeof(key), "deepseek41.%s", suffix);
    const gguf_kv_t *kv = gguf_find_key(gf, key);
    if (!kv || (kv->type != GGUF_TYPE_UINT32 && kv->type != GGUF_TYPE_INT32) ||
        gguf_get_u32(gf, key, UINT32_MAX) != expected) {
        MYDS_LOG_E("V4.1 unsupported/missing metadata %s (expected %u)", key, expected);
        return false;
    }
    return true;
}

static bool metadata_float(const gguf_file_t *gf, const char *suffix, float expected) {
    char key[160]; snprintf(key, sizeof(key), "deepseek41.%s", suffix);
    const gguf_kv_t *kv = gguf_find_key(gf, key);
    if (!kv || kv->type != GGUF_TYPE_FLOAT32 || kv->value.val_f32 != expected) {
        MYDS_LOG_E("V4.1 unsupported/missing metadata %s", key); return false;
    }
    return true;
}

static bool metadata_array(const gguf_file_t *gf, const char *suffix,
                           const uint32_t *expected, uint32_t count) {
    char key[160]; snprintf(key, sizeof(key), "deepseek41.%s", suffix);
    const gguf_kv_t *kv = gguf_find_key(gf, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY || kv->value.val_arr.count != count ||
        (kv->value.val_arr.elem_type != GGUF_TYPE_UINT32 &&
         kv->value.val_arr.elem_type != GGUF_TYPE_INT32) ||
        !kv->value.val_arr.data || memcmp(kv->value.val_arr.data, expected, count * 4u)) {
        MYDS_LOG_E("V4.1 unsupported/missing metadata %s", key); return false;
    }
    return true;
}

static myds_metal_buffer_t *wrap(const gguf_tensor_t *t) {
    return t && t->data ? myds_metal_buffer_from_ptr(t->data, t->size_bytes) : NULL;
}

static myds_metal_buffer_t *weight(const gguf_file_t *gf, uint32_t layer,
                                   const char *suffix, ggml_type type,
                                   uint32_t width, uint32_t rows) {
    char name[128]; snprintf(name, sizeof(name), "blk.%u.%s.weight", layer, suffix);
    const gguf_tensor_t *t = gguf_find_tensor(gf, name);
    if (!t || t->type != type || t->n_dims != (rows ? 2u : 1u) ||
        t->dims[0] != width || (rows && t->dims[1] != rows)) {
        MYDS_LOG_E("V4.1 missing/invalid tensor %s", name); return NULL;
    }
    myds_metal_buffer_t *b = wrap(t);
    if (!b) MYDS_LOG_E("V4.1 cannot map tensor %s", name);
    return b;
}

myds_v41_model_t *myds_v41_model_create(myds_model_t *m, const char *path) {
    if (!m || !path || strcmp(m->arch, "deepseek41") || m->dim != V41_DIM ||
        m->n_layers != V41_LAYERS || m->n_heads != V41_HEADS ||
        m->head_dim != V41_HEAD_DIM || m->q_lora_rank != 1280 ||
        m->n_out_groups != 8 || m->out_group_rank != 1024 ||
        m->n_routed_experts != 384 || m->n_activated_experts != 6 ||
        m->n_shared_experts != 1 || m->moe_expert_dim != 2304 ||
        m->tok_embeddings_type != GGML_TYPE_F16) {
        MYDS_LOG_E("Native V4.1 requires the validated Flash 4-stream text layout");
        return NULL;
    }
    const gguf_file_t *gf = m->gf;
    const struct { const char *key; uint32_t value; } ints[] = {
        {"sliding_window",128}, {"hc_mult",4}, {"hc_sinkhorn_iters",20},
        {"index_n_heads",32}, {"index_head_dim",128}, {"index_topk",512},
        {"candidate_source_layer_id",20}, {"candidate_topk_blocks",2048},
        {"candidate_block_size",8}, {"rope_theta",10000}, {"compress_rope_theta",160000},
    };
    for (size_t i = 0; i < sizeof(ints)/sizeof(*ints); i++)
        if (!metadata_u32(gf, ints[i].key, ints[i].value)) return NULL;
    const struct { const char *key; float value; } floats[] = {
        {"rms_norm_eps",1e-20f}, {"hc_eps",1e-6f}, {"swiglu_limit",10.0f},
        {"routed_scaling_factor",1.5f}, {"rope_scaling.factor",16.0f},
        {"rope_scaling.beta_fast",32.0f}, {"rope_scaling.beta_slow",1.0f},
        {"rope_scaling.original_max_position_embeddings",65536.0f},
    };
    for (size_t i = 0; i < sizeof(floats)/sizeof(*floats); i++)
        if (!metadata_float(gf, floats[i].key, floats[i].value)) return NULL;
    uint32_t ratios[V41_LAYERS];
    for (uint32_t i = 0; i < V41_LAYERS; i++) ratios[i] = ratio_for(i);
    const uint32_t kv[] = {2,8,14,20}, index[] = {2,8,14,20,24,28,32,36};
    if (!metadata_array(gf,"kv_source_layer_ids",kv,4) ||
        !metadata_array(gf,"index_source_layer_ids",index,8) ||
        !metadata_array(gf,"compress_ratios",ratios,V41_LAYERS) ||
        strcmp(gguf_get_str(gf,"deepseek41.scoring_func",""),"sqrtsoftplus") ||
        strcmp(gguf_get_str(gf,"deepseek41.topk_method",""),"noaux_tc") ||
        !gguf_get_bool(gf,"deepseek41.norm_topk_prob",false)) return NULL;
    myds_v41_model_t *v = calloc(1,sizeof(*v));
    if (!v) return NULL;
    v->hc_eps = 1e-6f;
#define W(member, name, type, width, rows) \
    do { l->member = weight(gf,i,name,type,width,rows); if (!l->member) goto fail; } while (0)
    for (uint32_t i = 0; i < V41_LAYERS; i++) {
        v41_layer_t *l = &v->layers[i];
        W(attn_fn,"hc_attn_fn",GGML_TYPE_F16,4*V41_DIM,24);
        W(attn_scale,"hc_attn_scale",GGML_TYPE_F32,3,0);
        W(attn_base,"hc_attn_base",GGML_TYPE_F32,24,0);
        W(ffn_fn,"hc_ffn_fn",GGML_TYPE_F16,4*V41_DIM,24);
        W(ffn_scale,"hc_ffn_scale",GGML_TYPE_F32,3,0);
        W(ffn_base,"hc_ffn_base",GGML_TYPE_F32,24,0);
        if (kv_source(i)) {
            W(compressor_kv,"attn_compressor_kv",GGML_TYPE_F16,V41_DIM,512);
            W(compressor_norm,"attn_compressor_norm",GGML_TYPE_F32,512,0);
            if (ratio_for(i)==2) W(compressor_gate,"attn_compressor_gate",GGML_TYPE_F16,V41_DIM,512);
            W(index_k,"indexer.attn_k",GGML_TYPE_F16,512,128);
            W(index_k_norm,"indexer.k_norm",GGML_TYPE_F32,128,0);
        }
        if (index_source(i)) {
            W(index_q,"indexer.attn_q_b",GGML_TYPE_F16,1280,V41_INDEX_HEADS*128);
            W(index_proj,"indexer.proj",GGML_TYPE_F16,V41_DIM,V41_INDEX_HEADS);
        }
        const myds_layer_t *base = &m->layers[i];
        if (!base->router_bias || !base->attn_sinks || !base->shared_gate ||
            base->attn_output_a_type != GGML_TYPE_Q8_0 ||
            base->attn_output_b_type != GGML_TYPE_Q8_0) goto fail;
    }
#undef W
    v->engram = myds_engram_create(gf,path,m->vocab_size,m->dim);
    if (!v->engram) goto fail;
    for (uint32_t i=0;i<2;i++) {
        v->engram_proj[i]=wrap(v->engram->projection[i]);
        v->engram_q[i]=wrap(v->engram->q_norm[i]);
        v->engram_k[i]=wrap(v->engram->k_norm[i]);
        if (!v->engram_proj[i] || !v->engram_q[i] || !v->engram_k[i]) goto fail;
    }
    if (!myds_engram_unmap_tables(m->gf,v->engram)) goto fail;
    MYDS_LOG_I("Native V4.1 diagnostic graph: HC4, Engram disk rows, SWA128, 4 KV owners");
    return v;
fail:
    myds_v41_model_free(v); return NULL;
}

void myds_v41_model_free(myds_v41_model_t *v) {
    if (!v) return;
    for (uint32_t i=0;i<V41_LAYERS;i++) {
        v41_layer_t *l=&v->layers[i];
#define F(x) myds_metal_buffer_free(l->x)
        F(attn_fn); F(attn_scale); F(attn_base); F(ffn_fn); F(ffn_scale); F(ffn_base);
        F(compressor_kv); F(compressor_gate); F(compressor_norm);
        F(index_k); F(index_k_norm); F(index_q); F(index_proj);
#undef F
    }
    for (uint32_t i=0;i<2;i++) {
        myds_metal_buffer_free(v->engram_proj[i]); myds_metal_buffer_free(v->engram_q[i]);
        myds_metal_buffer_free(v->engram_k[i]);
    }
    myds_engram_free(v->engram); free(v);
}

myds_v41_context_t *myds_v41_context_create(myds_context_t *ctx) {
    if (!ctx || !ctx->model->v41 || !ctx->cfg.max_seq_len || ctx->cfg.max_seq_len>1048576u)
        return NULL;
    /* Basic buffers are allocated by the shared context constructor. */
    myds_metal_buffer_t *required[] = {ctx->x,ctx->x_norm,ctx->q,ctx->qr,ctx->qr_norm,
        ctx->kv_act,ctx->attn_heads,ctx->attn_low,ctx->attn_out,ctx->ffn_norm_out,
        ctx->ffn_out,ctx->logits,ctx->router_logits,ctx->selected_ids,ctx->selected_weights,
        ctx->moe_gate,ctx->moe_up,ctx->moe_mid,ctx->moe_down,ctx->moe_out,
        ctx->shared_gate_out,ctx->shared_up_out,ctx->shared_swiglu_out,ctx->shared_out};
    for (size_t i=0;i<sizeof(required)/sizeof(*required);i++) if (!required[i]) return NULL;
    if (!ctx->sample_probs || !ctx->sample_indices || !ctx->sample_cands) return NULL;
    myds_v41_context_t *g=calloc(1,sizeof(*g));
    if (!g) return NULL;
    g->capacity=ctx->cfg.max_seq_len;
    g->engram=myds_engram_clone(ctx->model->v41->engram);
    if (!g->engram) goto fail;
    g->gpu=myds_v41_gpu_scratch_create(g->capacity);
    if (!g->gpu) goto fail;
#define A(member, words) do { g->member=myds_metal_buffer_alloc((size_t)(words)*4u); if (!g->member) goto fail; } while (0)
    A(residual,4*V41_DIM); A(after_attn,4*V41_DIM); A(flat_norm,4*V41_DIM);
    A(mix,24); A(attn_split,24); A(ffn_split,24); A(pre,4);
    A(engram_rows[0],MYDS_ENGRAM_INPUT_DIM); A(engram_rows[1],MYDS_ENGRAM_INPUT_DIM);
    A(engram_kv,5*V41_DIM);
    for (uint32_t i=0;i<V41_LAYERS;i++) A(window[i],V41_WINDOW*V41_HEAD_DIM);
    for (uint32_t i=0;i<4;i++) {
        const size_t rows=i==3 ? g->capacity : (g->capacity+1u)/2u;
        A(compressed[i],rows*512); A(index_cache[i],rows*128);
        A(previous_kv[i],512); A(previous_score[i],512);
    }
    A(pool_kv,512); A(pool_score,512); A(latent,512); A(index_k,128);
    A(index_q,V41_INDEX_HEADS*128); A(index_weights,V41_INDEX_HEADS);
    A(index_scores,g->capacity); A(selected,V41_TOPK); A(selected_kv,V41_TOPK*512);
    A(block_scores,(g->capacity+7u)/8u); A(block_mask,(g->capacity+7u)/8u);
    A(block_selected,2048);
#undef A
    g->valid=true;
    return g;
fail:
    myds_v41_context_free(g); return NULL;
}

void myds_v41_context_free(myds_v41_context_t *g) {
    if (!g) return;
#define F(x) myds_metal_buffer_free(g->x)
    F(residual); F(after_attn); F(flat_norm); F(mix); F(attn_split); F(ffn_split); F(pre);
    F(engram_rows[0]); F(engram_rows[1]); F(engram_kv);
    for (uint32_t i=0;i<V41_LAYERS;i++) F(window[i]);
    for (uint32_t i=0;i<4;i++) { F(compressed[i]); F(index_cache[i]); F(previous_kv[i]); F(previous_score[i]); }
    F(pool_kv); F(pool_score); F(latent); F(index_k); F(index_q); F(index_weights);
    F(index_scores); F(selected); F(selected_kv); F(block_scores); F(block_selected); F(block_mask);
#undef F
    myds_v41_gpu_scratch_free(g->gpu); myds_engram_free(g->engram); free(g);
}

static int project(myds_metal_buffer_t *out, const myds_metal_buffer_t *w,
                   const myds_metal_buffer_t *in, uint32_t rows, uint32_t width,
                   ggml_type type, bool round) {
    myds_metal_barrier();
    return ((type == GGML_TYPE_F16 || type == GGML_TYPE_F32)
        ? myds_metal_v41_gemv(out,w,in,rows,width,type)
        : myds_metal_gemv(out,w,in,rows,width,type)) &&
        (!round || myds_metal_v41_bf16(out,rows));
}

static int publish_and_select(myds_context_t *ctx, uint32_t il, uint32_t pos) {
    myds_v41_context_t *g=ctx->v41;
    const v41_layer_t *l=&ctx->model->v41->layers[il];
    const uint32_t ratio=ratio_for(il), owner=owner_for(il);
    if (!ratio) return 1;
    const uint32_t count=(pos+1u)/ratio;
#define CHECK(x) do { if (!(x)) return 0; } while (0)
    if (kv_source(il)) {
        CHECK(project(g->pool_kv,l->compressor_kv,ctx->x_norm,512,V41_DIM,GGML_TYPE_F16,ratio==1));
        if (ratio==2) {
            CHECK(project(g->pool_score,l->compressor_gate,ctx->x_norm,512,V41_DIM,GGML_TYPE_F16,false));
            CHECK(myds_v41_gpu_pool2(g->latent,g->pool_kv,g->pool_score,
                g->previous_kv[owner],g->previous_score[owner],512,pos));
        } else CHECK(myds_metal_buffer_copy(g->latent,0,g->pool_kv,0,512*4u));
        if ((pos+1u)%ratio==0) {
            const uint32_t start=pos+1u-ratio;
            CHECK(myds_metal_v41_rmsnorm(g->latent,g->latent,l->compressor_norm,512,ctx->model->norm_eps,true));
            CHECK(project(g->index_k,l->index_k,g->latent,128,512,GGML_TYPE_F16,true));
            CHECK(myds_metal_v41_rmsnorm(g->index_k,g->index_k,l->index_k_norm,128,ctx->model->norm_eps,true));
            CHECK(myds_v41_gpu_rope(g->index_k,128,1,start,true,false));
            CHECK(myds_v41_gpu_quant(g->index_k,128,1,2));
            CHECK(myds_metal_buffer_copy(g->index_cache[owner],(size_t)(count-1)*128*4u,g->index_k,0,128*4u));
            CHECK(myds_v41_gpu_rope(g->latent,512,1,start,true,false));
            CHECK(myds_v41_gpu_quant(g->latent,512,1,3));
            CHECK(myds_metal_buffer_copy(g->compressed[owner],(size_t)(count-1)*512*4u,g->latent,0,512*4u));
        }
    }
    if (count && index_source(il)) {
        CHECK(project(g->index_q,l->index_q,ctx->qr_norm,V41_INDEX_HEADS*128,1280,GGML_TYPE_F16,true));
        CHECK(myds_v41_gpu_rope(g->index_q,128,V41_INDEX_HEADS,pos,true,false));
        CHECK(myds_v41_gpu_quant(g->index_q,128,V41_INDEX_HEADS,2));
        CHECK(project(g->index_weights,l->index_proj,ctx->x_norm,V41_INDEX_HEADS,V41_DIM,GGML_TYPE_F16,true));
        CHECK(myds_v41_gpu_indexscores(g->index_scores,g->index_q,g->index_weights,g->index_cache[owner],count,V41_INDEX_HEADS));
        if (il==20) {
            uint32_t blocks=(count+7)/8,top=blocks<2048?blocks:2048;
            CHECK(myds_v41_gpu_candidates(g->block_scores,g->index_scores,count,pos,ratio));
            CHECK(myds_v41_gpu_topk(g->block_selected,g->block_scores,blocks,top,g->gpu));
            CHECK(myds_v41_gpu_topkmask(g->block_mask,g->block_selected,blocks,top));
        } else if (il>20) CHECK(myds_v41_gpu_filter(g->index_scores,g->block_mask,count,pos,ratio));
        CHECK(myds_v41_gpu_topk(g->selected,g->index_scores,count,count<V41_TOPK?count:V41_TOPK,g->gpu));
    }
#undef CHECK
    return 1;
}

int32_t myds_v41_forward(myds_context_t *ctx, int32_t token, uint32_t pos, bool need_logits) {
    myds_model_t *m=ctx->model;
    myds_v41_context_t *g=ctx->v41;
    myds_v41_model_t *v=m->v41;
    uint32_t il=0;
    const char *phase="input";
    if (!g || token<0 || (uint32_t)token>=m->vocab_size || pos>=g->capacity ||
        pos>=ctx->cfg.max_seq_len) return -1;
    if (!myds_metal_commit_batch()) return -1;
    if (pos==0) { g->next_position=0; g->valid=true; myds_engram_reset(g->engram); }
    if (!g->valid || pos!=g->next_position) {
        MYDS_LOG_E("V4.1 expected consecutive token position %u, received %u",g->next_position,pos);
        return -1;
    }
#define CHECK(x) do { if (!(x)) goto fail; } while (0)
#define TRACE(stage,buf,n) CHECK(myds_trace_f32(stage,il,pos,buf,n))
    CHECK(myds_engram_prepare_token(g->engram,token,pos));
    for (uint32_t i=0;i<2;i++) CHECK(myds_metal_buffer_write(g->engram_rows[i],0,
        g->engram->row_values[i],MYDS_ENGRAM_INPUT_DIM*4u));
    const uint16_t *row=(const uint16_t *)myds_metal_buffer_data(m->tok_embeddings)+(size_t)token*V41_DIM;
    float *res=myds_metal_buffer_data(g->residual);
    for (uint32_t j=0;j<V41_DIM;j++) res[j]=myds_fp16_to_fp32(row[j]);
    for (uint32_t j=1;j<4;j++) memcpy(res+j*V41_DIM,res,V41_DIM*4u);
    const float initial_pre[4]={1,0,0,0};
    CHECK(myds_metal_buffer_write(g->pre,0,initial_pre,sizeof(initial_pre)));
    TRACE("embedding",g->residual,4*V41_DIM);
    for (il=0;il<V41_LAYERS;il++) {
        const myds_layer_t *base=&m->layers[il];
        const v41_layer_t *l=&v->layers[il];
        const bool compressed=ratio_for(il)!=0;
        phase="HC attention input";
        CHECK(myds_metal_begin_batch());
        if (il==1 || il==14) {
            uint32_t e=il==1?0:1;
            CHECK(project(g->engram_kv,v->engram_proj[e],g->engram_rows[e],5*V41_DIM,MYDS_ENGRAM_INPUT_DIM,GGML_TYPE_F16,true));
            CHECK(myds_metal_v41_engram_add(g->residual,g->engram_kv,v->engram_q[e],v->engram_k[e],V41_DIM,m->norm_eps));
            TRACE("engram_out",g->residual,4*V41_DIM);
        }
        CHECK(myds_metal_v41_hc_mix(g->attn_split,g->residual,l->attn_fn,GGML_TYPE_F16,l->attn_scale,l->attn_base,g->flat_norm,g->mix,V41_DIM,m->norm_eps,v->hc_eps));
        TRACE("hc_attn_mix",g->mix,24); TRACE("hc_attn_split",g->attn_split,24);
        TRACE("hc_attn_pre",g->pre,4);
        CHECK(myds_metal_v41_hc_weighted_sum(ctx->x,g->residual,g->pre,V41_DIM));
        TRACE("hc_attn_input",ctx->x,V41_DIM);
        CHECK(myds_metal_v41_rmsnorm(ctx->x_norm,ctx->x,base->attn_norm,V41_DIM,m->norm_eps,true));
        TRACE("attn_norm",ctx->x_norm,V41_DIM);
        phase="attention projections";
        CHECK(project(ctx->qr,base->attn_q_a,ctx->x_norm,1280,V41_DIM,base->attn_q_a_type,true));
        TRACE("q_a_raw",ctx->qr,1280);
        CHECK(myds_metal_v41_rmsnorm(ctx->qr_norm,ctx->qr,base->attn_q_a_norm,1280,m->norm_eps,true));
        TRACE("q_a_norm",ctx->qr_norm,1280);
        CHECK(project(ctx->q,base->attn_q_b,ctx->qr_norm,V41_HEADS*512,1280,base->attn_q_b_type,true));
        TRACE("q_b",ctx->q,V41_HEADS*512);
        CHECK(project(ctx->kv_act,base->attn_kv,ctx->x_norm,512,V41_DIM,base->attn_kv_type,true));
        TRACE("kv_raw",ctx->kv_act,512);
        CHECK(myds_metal_v41_rmsnorm(ctx->kv_act,ctx->kv_act,base->attn_kv_a_norm,512,m->norm_eps,true));
        TRACE("kv_norm",ctx->kv_act,512);
        CHECK(myds_v41_gpu_rope(ctx->q,512,V41_HEADS,pos,compressed,false));
        CHECK(myds_v41_gpu_rope(ctx->kv_act,512,1,pos,compressed,false));
        TRACE("kv_rope",ctx->kv_act,512); TRACE("q_rope",ctx->q,V41_HEADS*512);
        CHECK(myds_v41_gpu_quant(ctx->kv_act,512,1,1));
        TRACE("kv_quant",ctx->kv_act,512);
        CHECK(myds_metal_buffer_copy(g->window[il],(size_t)(pos%128)*512*4u,ctx->kv_act,0,512*4u));
        phase="compressed KV / indexer";
        CHECK(publish_and_select(ctx,il,pos));
        uint32_t count=compressed?(pos+1u)/ratio_for(il):0;
        uint32_t top=count<V41_TOPK?count:V41_TOPK;
        if (count) CHECK(myds_v41_gpu_gather(g->selected_kv,g->compressed[owner_for(il)],g->selected,count,top));
        uint32_t nraw=pos+1u<128?pos+1u:128;
        phase="attention reduction";
        CHECK(myds_v41_gpu_attention(ctx->attn_heads,base->attn_sinks,ctx->q,g->window[il],nraw,(pos+1u-nraw)%128,g->selected_kv,top,g->gpu));
        CHECK(myds_metal_v41_bf16(ctx->attn_heads,V41_HEADS*512));
        CHECK(myds_v41_gpu_rope(ctx->attn_heads,512,V41_HEADS,pos,compressed,true));
        CHECK(myds_metal_attn_output_a(ctx->attn_low,base->attn_output_a,ctx->attn_heads,8,4096,1024,base->attn_output_a_type));
        CHECK(myds_metal_v41_bf16(ctx->attn_low,8192));
        CHECK(project(ctx->attn_out,base->attn_output_b,ctx->attn_low,V41_DIM,8192,base->attn_output_b_type,true));
        TRACE("attn_out",ctx->attn_out,V41_DIM);
        phase="HC FFN input";
        CHECK(myds_metal_v41_hc_expand(g->after_attn,ctx->attn_out,g->residual,g->attn_split,V41_DIM));
        TRACE("after_attn",g->after_attn,4*V41_DIM);
        CHECK(myds_metal_v41_hc_mix(g->ffn_split,g->after_attn,l->ffn_fn,GGML_TYPE_F16,l->ffn_scale,l->ffn_base,g->flat_norm,g->mix,V41_DIM,m->norm_eps,v->hc_eps));
        TRACE("hc_ffn_mix",g->mix,24); TRACE("hc_ffn_split",g->ffn_split,24);
        CHECK(myds_metal_v41_hc_weighted_sum(ctx->x,g->after_attn,g->attn_split,V41_DIM));
        TRACE("hc_ffn_input",ctx->x,V41_DIM);
        CHECK(myds_metal_v41_rmsnorm(ctx->ffn_norm_out,ctx->x,base->ffn_norm,V41_DIM,m->norm_eps,true));
        TRACE("ffn_norm",ctx->ffn_norm_out,V41_DIM);
        phase="MoE";
        CHECK(myds_v41_moe(ctx,il,(uint32_t)token,pos));
        CHECK(myds_metal_v41_hc_expand(g->residual,ctx->ffn_out,g->after_attn,g->ffn_split,V41_DIM));
        CHECK(myds_metal_buffer_copy(g->pre,0,g->ffn_split,0,4*4u));
        TRACE("layer_out",g->residual,4*V41_DIM);
        /* A bounded command batch per layer retains all scratch dependencies. */
        CHECK(myds_metal_commit_batch());
    }
    phase="output head";
    if (need_logits) {
        CHECK(myds_metal_begin_batch());
        CHECK(myds_metal_v41_hc_weighted_sum(ctx->x,g->residual,g->pre,V41_DIM));
        CHECK(myds_metal_v41_rmsnorm(ctx->x_norm,ctx->x,m->output_norm,V41_DIM,m->norm_eps,true));
        CHECK(project(ctx->logits,m->output,ctx->x_norm,m->vocab_size,V41_DIM,m->output_type,false));
        CHECK(myds_metal_commit_batch());
        CHECK(myds_trace_f32("logits",0,pos,ctx->logits,m->vocab_size));
        const float *logits=myds_metal_buffer_data(ctx->logits);
        for (uint32_t i=0;i<m->vocab_size;i++) if (!isfinite(logits[i])) goto fail;
    }
    g->next_position=pos+1u;
    return need_logits ? myds_sample_next_token(myds_metal_buffer_data(ctx->logits),m->vocab_size,&ctx->cfg,ctx,NULL,0) : 0;
fail:
    MYDS_LOG_E("V4.1 native graph failed at pos %u layer %u: %s",pos,il,phase);
    (void)myds_metal_commit_batch(); g->valid=false; return -1;
#undef CHECK
#undef TRACE
}

int myds_v41_prefill_tokens(myds_context_t *ctx, const int32_t *tokens,
                            uint32_t n_tokens, volatile int *cancel_flag) {
    if (!ctx || !tokens || n_tokens == 0) return 1;
    myds_model_t *m = ctx->model;
    myds_v41_context_t *g = ctx->v41;
    myds_v41_model_t *v = m ? m->v41 : NULL;
    if (!m || !g || !v) return -1;
    if (!myds_metal_commit_batch()) return -1;

    uint32_t base_pos = ctx->pos;
    if (base_pos + n_tokens > g->capacity || base_pos + n_tokens > ctx->cfg.max_seq_len) {
        MYDS_LOG_E("V4.1 prefill prompt length exceeds context capacity (%u + %u > %u)",
                   base_pos, n_tokens, g->capacity);
        return -1;
    }

    if (base_pos == 0) {
        g->next_position = 0;
        g->valid = true;
        myds_engram_reset(g->engram);
    }
    if (!g->valid || base_pos != g->next_position) {
        MYDS_LOG_E("V4.1 expected consecutive token position %u, received %u",
                   g->next_position, base_pos);
        return -1;
    }

    const char *chunk_env = getenv("MYDS_PREFILL_CHUNK_SIZE");
    uint32_t chunk_size = chunk_env ? (uint32_t)atoi(chunk_env) : 256u;
    if (chunk_size < 2u) chunk_size = 2u;
    if (chunk_size % 2u != 0) chunk_size = (chunk_size / 2u) * 2u;
    if (chunk_size > 2048u) chunk_size = 2048u;

    size_t res_size = (size_t)n_tokens * 4u * V41_DIM;
    float *prompt_res = (float *)malloc(res_size * sizeof(float));
    float *prompt_pre = (float *)malloc((size_t)n_tokens * 4u * sizeof(float));
    int32_t *prompt_selected = (int32_t *)malloc((size_t)n_tokens * V41_TOPK * sizeof(int32_t));
    size_t mask_bytes = (size_t)myds_metal_buffer_size(g->block_mask);
    uint8_t *prompt_block_mask = (uint8_t *)malloc((size_t)n_tokens * mask_bytes);
    float *engram_all[2] = {NULL, NULL};
    engram_all[0] = (float *)malloc((size_t)n_tokens * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    engram_all[1] = (float *)malloc((size_t)n_tokens * MYDS_ENGRAM_INPUT_DIM * sizeof(float));

    if (!prompt_res || !prompt_pre || !prompt_selected || !prompt_block_mask ||
        !engram_all[0] || !engram_all[1]) {
        MYDS_LOG_E("V4.1 prefill failed to allocate workspace buffers");
        free(prompt_res);
        free(prompt_pre);
        free(prompt_selected);
        free(prompt_block_mask);
        free(engram_all[0]);
        free(engram_all[1]);
        return -1;
    }

    uint32_t il = 0;
    const char *phase = "engram prepare";

    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t pos = base_pos + t;
        if (!myds_engram_prepare_token(g->engram, tokens[t], (uint64_t)pos)) {
            MYDS_LOG_E("V4.1 engram prepare failed at pos %u", pos);
            goto fail;
        }
        memcpy(engram_all[0] + (size_t)t * MYDS_ENGRAM_INPUT_DIM,
               g->engram->row_values[0], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        memcpy(engram_all[1] + (size_t)t * MYDS_ENGRAM_INPUT_DIM,
               g->engram->row_values[1], MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    }

    phase = "embedding lookup";
    const uint16_t *embd_table = (const uint16_t *)myds_metal_buffer_data(m->tok_embeddings);
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t tok = tokens[t];
        if (tok < 0 || (uint32_t)tok >= m->vocab_size) {
            MYDS_LOG_E("V4.1 invalid token %d at pos %u", tok, base_pos + t);
            goto fail;
        }
        const uint16_t *row = embd_table + (size_t)tok * V41_DIM;
        float *r = prompt_res + (size_t)t * 4u * V41_DIM;
        for (uint32_t j = 0; j < V41_DIM; j++) {
            r[j] = myds_fp16_to_fp32(row[j]);
        }
        for (uint32_t s = 1; s < 4; s++) {
            memcpy(r + s * V41_DIM, r, V41_DIM * sizeof(float));
        }
        float *p = prompt_pre + (size_t)t * 4u;
        p[0] = 1.0f; p[1] = 0.0f; p[2] = 0.0f; p[3] = 0.0f;

        if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
            memcpy(myds_metal_buffer_data(g->residual), r, 4u * V41_DIM * sizeof(float));
            if (!myds_trace_f32("embedding", 0, base_pos + t, g->residual, 4 * V41_DIM)) goto fail;
        }
    }

    phase = "layer loop";
#define CHECK(x) do { if (!(x)) goto fail; } while (0)
#define TRACE(stage,buf,n) CHECK(myds_trace_f32(stage,il,pos,buf,n))

    for (il = 0; il < V41_LAYERS; il++) {
        if (cancel_flag && *cancel_flag) goto cancelled;
        const myds_layer_t *base = &m->layers[il];
        const v41_layer_t *l = &v->layers[il];
        const bool compressed = ratio_for(il) != 0;

        for (uint32_t c_start = 0; c_start < n_tokens; c_start += chunk_size) {
            if (cancel_flag && *cancel_flag) goto cancelled;
            uint32_t c_len = n_tokens - c_start;
            if (c_len > chunk_size) c_len = chunk_size;

            for (uint32_t t = 0; t < c_len; t++) {
                uint32_t idx = c_start + t;
                uint32_t pos = base_pos + idx;
                float *tok_res = prompt_res + (size_t)idx * 4u * V41_DIM;
                float *tok_pre = prompt_pre + (size_t)idx * 4u;

                memcpy(myds_metal_buffer_data(g->residual), tok_res, 4u * V41_DIM * sizeof(float));
                memcpy(myds_metal_buffer_data(g->pre), tok_pre, 4u * sizeof(float));

                phase = "HC attention input";
                CHECK(myds_metal_begin_batch());

                if (il == 1 || il == 14) {
                    uint32_t e = (il == 1 ? 0 : 1);
                    float *erow = engram_all[e] + (size_t)idx * MYDS_ENGRAM_INPUT_DIM;
                    CHECK(myds_metal_buffer_write(g->engram_rows[e], 0, erow, MYDS_ENGRAM_INPUT_DIM * 4u));
                    CHECK(project(g->engram_kv, v->engram_proj[e], g->engram_rows[e], 5 * V41_DIM, MYDS_ENGRAM_INPUT_DIM, GGML_TYPE_F16, true));
                    CHECK(myds_metal_v41_engram_add(g->residual, g->engram_kv, v->engram_q[e], v->engram_k[e], V41_DIM, m->norm_eps));
                    TRACE("engram_out", g->residual, 4 * V41_DIM);
                }

                CHECK(myds_metal_v41_hc_mix(g->attn_split, g->residual, l->attn_fn, GGML_TYPE_F16, l->attn_scale, l->attn_base, g->flat_norm, g->mix, V41_DIM, m->norm_eps, v->hc_eps));
                TRACE("hc_attn_mix", g->mix, 24); TRACE("hc_attn_split", g->attn_split, 24);
                TRACE("hc_attn_pre", g->pre, 4);
                CHECK(myds_metal_v41_hc_weighted_sum(ctx->x, g->residual, g->pre, V41_DIM));
                TRACE("hc_attn_input", ctx->x, V41_DIM);
                CHECK(myds_metal_v41_rmsnorm(ctx->x_norm, ctx->x, base->attn_norm, V41_DIM, m->norm_eps, true));
                TRACE("attn_norm", ctx->x_norm, V41_DIM);

                phase = "attention projections";
                CHECK(project(ctx->qr, base->attn_q_a, ctx->x_norm, 1280, V41_DIM, base->attn_q_a_type, true));
                TRACE("q_a_raw", ctx->qr, 1280);
                CHECK(myds_metal_v41_rmsnorm(ctx->qr_norm, ctx->qr, base->attn_q_a_norm, 1280, m->norm_eps, true));
                TRACE("q_a_norm", ctx->qr_norm, 1280);
                CHECK(project(ctx->q, base->attn_q_b, ctx->qr_norm, V41_HEADS * 512, 1280, base->attn_q_b_type, true));
                TRACE("q_b", ctx->q, V41_HEADS * 512);
                CHECK(project(ctx->kv_act, base->attn_kv, ctx->x_norm, 512, V41_DIM, base->attn_kv_type, true));
                TRACE("kv_raw", ctx->kv_act, 512);
                CHECK(myds_metal_v41_rmsnorm(ctx->kv_act, ctx->kv_act, base->attn_kv_a_norm, 512, m->norm_eps, true));
                TRACE("kv_norm", ctx->kv_act, 512);

                CHECK(myds_v41_gpu_rope(ctx->q, 512, V41_HEADS, pos, compressed, false));
                CHECK(myds_v41_gpu_rope(ctx->kv_act, 512, 1, pos, compressed, false));
                TRACE("kv_rope", ctx->kv_act, 512); TRACE("q_rope", ctx->q, V41_HEADS * 512);
                CHECK(myds_v41_gpu_quant(ctx->kv_act, 512, 1, 1));
                TRACE("kv_quant", ctx->kv_act, 512);
                CHECK(myds_metal_buffer_copy(g->window[il], (size_t)(pos % 128) * 512 * 4u, ctx->kv_act, 0, 512 * 4u));

                phase = "compressed KV / indexer";
                if (il > 20 && index_source(il)) {
                    uint8_t *src_mask = prompt_block_mask + (size_t)idx * mask_bytes;
                    CHECK(myds_metal_buffer_write(g->block_mask, 0, src_mask, mask_bytes));
                }
                CHECK(publish_and_select(ctx, il, pos));
                if (il == 20) {
                    uint8_t *dst_mask = prompt_block_mask + (size_t)idx * mask_bytes;
                    memcpy(dst_mask, myds_metal_buffer_data(g->block_mask), mask_bytes);
                }
                if (index_source(il)) {
                    int32_t *dst_sel = prompt_selected + (size_t)idx * V41_TOPK;
                    memcpy(dst_sel, myds_metal_buffer_data(g->selected), V41_TOPK * sizeof(int32_t));
                } else if (compressed) {
                    int32_t *src_sel = prompt_selected + (size_t)idx * V41_TOPK;
                    memcpy(myds_metal_buffer_data(g->selected), src_sel, V41_TOPK * sizeof(int32_t));
                }

                uint32_t count = compressed ? (pos + 1u) / ratio_for(il) : 0;
                uint32_t top = count < V41_TOPK ? count : V41_TOPK;
                if (count) {
                    CHECK(myds_v41_gpu_gather(g->selected_kv, g->compressed[owner_for(il)], g->selected, count, top));
                }
                uint32_t nraw = pos + 1u < 128 ? pos + 1u : 128;

                phase = "attention reduction";
                CHECK(myds_v41_gpu_attention(ctx->attn_heads, base->attn_sinks, ctx->q, g->window[il], nraw, (pos + 1u - nraw) % 128, g->selected_kv, top, g->gpu));
                CHECK(myds_metal_v41_bf16(ctx->attn_heads, V41_HEADS * 512));
                CHECK(myds_v41_gpu_rope(ctx->attn_heads, 512, V41_HEADS, pos, compressed, true));
                CHECK(myds_metal_attn_output_a(ctx->attn_low, base->attn_output_a, ctx->attn_heads, 8, 4096, 1024, base->attn_output_a_type));
                CHECK(myds_metal_v41_bf16(ctx->attn_low, 8192));
                CHECK(project(ctx->attn_out, base->attn_output_b, ctx->attn_low, V41_DIM, 8192, base->attn_output_b_type, true));
                TRACE("attn_out", ctx->attn_out, V41_DIM);

                phase = "HC FFN input";
                CHECK(myds_metal_v41_hc_expand(g->after_attn, ctx->attn_out, g->residual, g->attn_split, V41_DIM));
                TRACE("after_attn", g->after_attn, 4 * V41_DIM);
                CHECK(myds_metal_v41_hc_mix(g->ffn_split, g->after_attn, l->ffn_fn, GGML_TYPE_F16, l->ffn_scale, l->ffn_base, g->flat_norm, g->mix, V41_DIM, m->norm_eps, v->hc_eps));
                TRACE("hc_ffn_mix", g->mix, 24); TRACE("hc_ffn_split", g->ffn_split, 24);
                CHECK(myds_metal_v41_hc_weighted_sum(ctx->x, g->after_attn, g->attn_split, V41_DIM));
                TRACE("hc_ffn_input", ctx->x, V41_DIM);
                CHECK(myds_metal_v41_rmsnorm(ctx->ffn_norm_out, ctx->x, base->ffn_norm, V41_DIM, m->norm_eps, true));
                TRACE("ffn_norm", ctx->ffn_norm_out, V41_DIM);

                phase = "MoE";
                CHECK(myds_v41_moe(ctx, il, (uint32_t)tokens[idx], pos));
                CHECK(myds_metal_v41_hc_expand(g->residual, ctx->ffn_out, g->after_attn, g->ffn_split, V41_DIM));
                CHECK(myds_metal_buffer_copy(g->pre, 0, g->ffn_split, 0, 4 * 4u));
                TRACE("layer_out", g->residual, 4 * V41_DIM);

                CHECK(myds_metal_commit_batch());

                memcpy(tok_res, myds_metal_buffer_data(g->residual), 4u * V41_DIM * sizeof(float));
                memcpy(tok_pre, myds_metal_buffer_data(g->pre), 4u * sizeof(float));
            }
        }
    }
#undef CHECK
#undef TRACE

    ctx->pos = base_pos + n_tokens;
    g->next_position = ctx->pos;
    g->valid = true;

    float *last_res = prompt_res + (size_t)(n_tokens - 1) * 4u * V41_DIM;
    float *last_pre = prompt_pre + (size_t)(n_tokens - 1) * 4u;
    memcpy(myds_metal_buffer_data(g->residual), last_res, 4u * V41_DIM * sizeof(float));
    memcpy(myds_metal_buffer_data(g->pre), last_pre, 4u * sizeof(float));

    if (ctx->cfg.temperature > 0.0f) {
        for (uint32_t i = 0; i < n_tokens; i++) {
            ctx->rng_state = ctx->rng_state * 6364136223846793005ULL + 1ULL;
        }
    }

    free(prompt_res);
    free(prompt_pre);
    free(prompt_selected);
    free(prompt_block_mask);
    free(engram_all[0]);
    free(engram_all[1]);
    return 1;

cancelled:
    free(prompt_res);
    free(prompt_pre);
    free(prompt_selected);
    free(prompt_block_mask);
    free(engram_all[0]);
    free(engram_all[1]);
    return -2;

fail:
    MYDS_LOG_E("V4.1 prefill failed at layer %u: %s", il, phase);
    (void)myds_metal_commit_batch();
    g->valid = false;
    free(prompt_res);
    free(prompt_pre);
    free(prompt_selected);
    free(prompt_block_mask);
    free(engram_all[0]);
    free(engram_all[1]);
    return -1;
}
