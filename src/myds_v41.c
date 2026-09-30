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
#include <sys/time.h>

#define V41_LAYERS 40u
#define V41_DIM 5120u
#define V41_HEADS 64u
#define V41_HEAD_DIM 512u
#define V41_INDEX_HEADS 32u
#define V41_INDEX_DIM 128u
#define V41_WINDOW 128u
#define V41_TOPK 512u
/* Per-token workspace arrays below are sized by this, NOT by 512.  They used to
 * be hardcoded 512-wide while the scalar fields (cur_res_view, cur_pre_view,
 * hc_flat_norm_buf, hc_mix_buf) sat after them in the struct, so any capacity
 * above 512 made the assembly loop write arr[t] past the arrays and silently
 * overwrite those fields -- hc_flat_norm_buf came back as size 0 and prefill
 * died at layer 0 with "HC attention input".  Raising the cap therefore needs
 * this sizing, not just a relaxed clamp. */
#define V41_MAX_CHUNK 4096u

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

static int project_batch(myds_metal_buffer_t *out, const myds_metal_buffer_t *w,
                         const myds_metal_buffer_t *in, uint32_t rows, uint32_t width,
                         uint32_t n_tok, ggml_type type, bool round) {
    myds_metal_barrier();
    int use_mm = (n_tok >= 32 && getenv("MYDS_DISABLE_MUL_MM") == NULL);
    int ok = 0;
    if (use_mm) {
        ok = myds_metal_mul_mm(out, w, in, rows, width, n_tok, type);
    }
    if (!ok) {
        ok = ((type == GGML_TYPE_F16 || type == GGML_TYPE_F32)
            ? myds_metal_v41_gemv_batch(out, w, in, rows, width, n_tok, type)
            : myds_metal_gemv_batch(out, w, in, rows, width, n_tok, type));
    }
    return ok && (!round || myds_metal_v41_bf16(out, rows * n_tok));
}

static int dispatch_k(const char *kernel, const void *args, size_t size,
                      const myds_metal_buffer_t *const *buffers, size_t count,
                      uint32_t groups, uint32_t threads, size_t scratch) {
    const uint32_t grid[3] = {groups, 1, 1}, block[3] = {threads, 1, 1};
    return myds_metal_dispatch_kernel(kernel, args, size, buffers, count, grid, block, scratch);
}

static int publish_and_select(myds_context_t *ctx, uint32_t il, uint32_t pos,
                              const myds_metal_buffer_t *x_norm, const myds_metal_buffer_t *qr_norm,
                              const myds_metal_buffer_t *pre_pool_kv,
                              const myds_metal_buffer_t *pre_pool_score,
                              myds_metal_buffer_t *pre_index_q,
                              const myds_metal_buffer_t *pre_index_weights,
                              myds_metal_buffer_t *out_selected,
                              const myds_metal_buffer_t *in_block_mask,
                              myds_metal_buffer_t *out_block_mask) {
    myds_v41_context_t *g=ctx->v41;
    const v41_layer_t *l=&ctx->model->v41->layers[il];
    const uint32_t ratio=ratio_for(il), owner=owner_for(il);
    if (!ratio) return 1;
    const uint32_t count=(pos+1u)/ratio;
#define CHECK(x) do { if (!(x)) return 0; } while (0)
    if (kv_source(il)) {
        const myds_metal_buffer_t *p_kv = pre_pool_kv;
        if (!p_kv) {
            CHECK(project(g->pool_kv,l->compressor_kv,x_norm,512,V41_DIM,GGML_TYPE_F16,ratio==1));
            p_kv = g->pool_kv;
        }
        if (ratio==2) {
            const myds_metal_buffer_t *p_score = pre_pool_score;
            if (!p_score) {
                CHECK(project(g->pool_score,l->compressor_gate,x_norm,512,V41_DIM,GGML_TYPE_F16,false));
                p_score = g->pool_score;
            }
            CHECK(myds_v41_gpu_pool2(g->latent,p_kv,p_score,
                g->previous_kv[owner],g->previous_score[owner],512,pos));
        } else CHECK(myds_metal_buffer_copy(g->latent,0,p_kv,0,512*4u));
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
        myds_metal_buffer_t *q_buf = pre_index_q;
        if (!q_buf) {
            CHECK(project(g->index_q,l->index_q,qr_norm,V41_INDEX_HEADS*128,1280,GGML_TYPE_F16,true));
            q_buf = g->index_q;
        }
        CHECK(myds_v41_gpu_rope(q_buf,128,V41_INDEX_HEADS,pos,true,false));
        CHECK(myds_v41_gpu_quant(q_buf,128,V41_INDEX_HEADS,2));

        const myds_metal_buffer_t *w_buf = pre_index_weights;
        if (!w_buf) {
            CHECK(project(g->index_weights,l->index_proj,x_norm,V41_INDEX_HEADS,V41_DIM,GGML_TYPE_F16,true));
            w_buf = g->index_weights;
        }

        CHECK(myds_v41_gpu_indexscores(g->index_scores,q_buf,w_buf,g->index_cache[owner],count,V41_INDEX_HEADS));
        if (il==20) {
            uint32_t blocks=(count+7)/8,top=blocks<2048?blocks:2048;
            CHECK(myds_v41_gpu_candidates(g->block_scores,g->index_scores,count,pos,ratio));
            CHECK(myds_v41_gpu_topk(g->block_selected,g->block_scores,blocks,top,g->gpu));
            myds_metal_buffer_t *b_mask = out_block_mask ? out_block_mask : g->block_mask;
            CHECK(myds_v41_gpu_topkmask(b_mask,g->block_selected,blocks,top));
        } else if (il>20) {
            const myds_metal_buffer_t *b_mask = in_block_mask ? in_block_mask : g->block_mask;
            CHECK(myds_v41_gpu_filter(g->index_scores,b_mask,count,pos,ratio));
        }
        myds_metal_buffer_t *sel = out_selected ? out_selected : g->selected;
        CHECK(myds_v41_gpu_topk(sel,g->index_scores,count,count<V41_TOPK?count:V41_TOPK,g->gpu));
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
    CHECK(myds_metal_begin_batch());
    for (il=0;il<V41_LAYERS;il++) {
        const myds_layer_t *base=&m->layers[il];
        const v41_layer_t *l=&v->layers[il];
        const bool compressed=ratio_for(il)!=0;
        phase="HC attention input";
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
        CHECK(publish_and_select(ctx,il,pos,ctx->x_norm,ctx->qr_norm,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
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
        myds_metal_barrier();
    }
    phase="output head";
    if (need_logits) {
        CHECK(myds_metal_v41_hc_weighted_sum(ctx->x,g->residual,g->pre,V41_DIM));
        CHECK(myds_metal_v41_rmsnorm(ctx->x_norm,ctx->x,m->output_norm,V41_DIM,m->norm_eps,true));
        CHECK(project(ctx->logits,m->output,ctx->x_norm,m->vocab_size,V41_DIM,m->output_type,false));
        CHECK(myds_metal_commit_batch());
        CHECK(myds_trace_f32("logits",0,pos,ctx->logits,m->vocab_size));
        const float *logits=myds_metal_buffer_data(ctx->logits);
        for (uint32_t i=0;i<m->vocab_size;i++) if (!isfinite(logits[i])) goto fail;
    } else {
        CHECK(myds_metal_commit_batch());
    }
    g->next_position=pos+1u;
    return need_logits ? myds_sample_next_token(myds_metal_buffer_data(ctx->logits),m->vocab_size,&ctx->cfg,ctx,NULL,0) : 0;
fail:
    MYDS_LOG_E("V4.1 native graph failed at pos %u layer %u: %s",pos,il,phase);
    (void)myds_metal_commit_batch(); g->valid=false; return -1;
#undef CHECK
#undef TRACE
}

typedef struct {
    uint32_t capacity;
    myds_metal_buffer_t *x_buf;
    myds_metal_buffer_t *x_norm_buf;
    myds_metal_buffer_t *qr_buf;
    myds_metal_buffer_t *qr_norm_buf;
    myds_metal_buffer_t *q_buf;
    myds_metal_buffer_t *kv_act_buf;
    myds_metal_buffer_t *attn_heads_buf;
    myds_metal_buffer_t *attn_low_buf;
    myds_metal_buffer_t *attn_out_buf;
    myds_metal_buffer_t *after_attn_buf;
    myds_metal_buffer_t *attn_split_buf;
    myds_metal_buffer_t *ffn_split_buf;
    myds_metal_buffer_t *ffn_norm_buf;
    myds_metal_buffer_t *router_logits_buf;
    myds_metal_buffer_t *shared_gate_buf;
    myds_metal_buffer_t *shared_up_buf;
    myds_metal_buffer_t *shared_swiglu_buf;
    myds_metal_buffer_t *shared_out_buf;
    myds_metal_buffer_t *engram_rows_buf[2];
    myds_metal_buffer_t *selected_ids_buf;
    myds_metal_buffer_t *selected_weights_buf;
    myds_metal_buffer_t *route_probs_buf;
    myds_metal_buffer_t *route_scores_buf;
    myds_metal_buffer_t *route_ids_f32_buf;
    myds_metal_buffer_t *pool_kv_buf;
    myds_metal_buffer_t *pool_score_buf;
    myds_metal_buffer_t *index_q_buf;
    myds_metal_buffer_t *index_weights_buf;
    myds_metal_buffer_t *moe_gate_buf;
    myds_metal_buffer_t *moe_up_buf;
    myds_metal_buffer_t *moe_mid_buf;
    myds_metal_buffer_t *moe_down_buf;
    myds_metal_buffer_t *moe_out_buf;
    myds_metal_buffer_t *ffn_out_buf;
    myds_metal_buffer_t *layer_out_buf;

    myds_metal_buffer_t *x[V41_MAX_CHUNK];
    myds_metal_buffer_t *x_norm[V41_MAX_CHUNK];
    myds_metal_buffer_t *qr[V41_MAX_CHUNK];
    myds_metal_buffer_t *qr_norm[V41_MAX_CHUNK];
    myds_metal_buffer_t *q[V41_MAX_CHUNK];
    myds_metal_buffer_t *kv_act[V41_MAX_CHUNK];
    myds_metal_buffer_t *attn_heads[V41_MAX_CHUNK];
    myds_metal_buffer_t *attn_low[V41_MAX_CHUNK];
    myds_metal_buffer_t *attn_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *after_attn[V41_MAX_CHUNK];
    myds_metal_buffer_t *attn_split[V41_MAX_CHUNK];
    myds_metal_buffer_t *ffn_split[V41_MAX_CHUNK];
    myds_metal_buffer_t *ffn_norm[V41_MAX_CHUNK];
    myds_metal_buffer_t *router_logits[V41_MAX_CHUNK];
    myds_metal_buffer_t *shared_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *engram_rows[2][V41_MAX_CHUNK];
    myds_metal_buffer_t *selected_ids[V41_MAX_CHUNK];
    myds_metal_buffer_t *selected_weights[V41_MAX_CHUNK];
    myds_metal_buffer_t *route_probs[V41_MAX_CHUNK];
    myds_metal_buffer_t *route_scores[V41_MAX_CHUNK];
    myds_metal_buffer_t *route_ids_f32[V41_MAX_CHUNK];
    myds_metal_buffer_t *pool_kv[V41_MAX_CHUNK];
    myds_metal_buffer_t *pool_score[V41_MAX_CHUNK];
    myds_metal_buffer_t *index_q[V41_MAX_CHUNK];
    myds_metal_buffer_t *index_weights[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_gate[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_up[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_mid[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_down[V41_MAX_CHUNK];
    myds_metal_buffer_t *moe_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *ffn_out[V41_MAX_CHUNK];
    myds_metal_buffer_t *layer_out[V41_MAX_CHUNK];

    myds_metal_buffer_t *res_view[V41_MAX_CHUNK];
    myds_metal_buffer_t *pre_view[V41_MAX_CHUNK];
    myds_metal_buffer_t *cur_res_view;
    myds_metal_buffer_t *cur_pre_view;
    myds_metal_buffer_t *cur_selected_view;
    myds_metal_buffer_t *cur_mask_view;
    myds_metal_buffer_t *hc_flat_norm_buf;
    myds_metal_buffer_t *hc_mix_buf;
} v41_chunk_workspace_t;

static void v41_chunk_workspace_free(v41_chunk_workspace_t *cw) {
    if (!cw) return;
    for (uint32_t t = 0; t < cw->capacity; t++) {
        if (cw->x[t]) myds_metal_buffer_free(cw->x[t]);
        if (cw->x_norm[t]) myds_metal_buffer_free(cw->x_norm[t]);
        if (cw->qr[t]) myds_metal_buffer_free(cw->qr[t]);
        if (cw->qr_norm[t]) myds_metal_buffer_free(cw->qr_norm[t]);
        if (cw->q[t]) myds_metal_buffer_free(cw->q[t]);
        if (cw->kv_act[t]) myds_metal_buffer_free(cw->kv_act[t]);
        if (cw->attn_heads[t]) myds_metal_buffer_free(cw->attn_heads[t]);
        if (cw->attn_low[t]) myds_metal_buffer_free(cw->attn_low[t]);
        if (cw->attn_out[t]) myds_metal_buffer_free(cw->attn_out[t]);
        if (cw->after_attn[t]) myds_metal_buffer_free(cw->after_attn[t]);
        if (cw->attn_split[t]) myds_metal_buffer_free(cw->attn_split[t]);
        if (cw->ffn_split[t]) myds_metal_buffer_free(cw->ffn_split[t]);
        if (cw->ffn_norm[t]) myds_metal_buffer_free(cw->ffn_norm[t]);
        if (cw->router_logits[t]) myds_metal_buffer_free(cw->router_logits[t]);
        if (cw->shared_out[t]) myds_metal_buffer_free(cw->shared_out[t]);
        if (cw->engram_rows[0][t]) myds_metal_buffer_free(cw->engram_rows[0][t]);
        if (cw->engram_rows[1][t]) myds_metal_buffer_free(cw->engram_rows[1][t]);
        if (cw->selected_ids[t]) myds_metal_buffer_free(cw->selected_ids[t]);
        if (cw->selected_weights[t]) myds_metal_buffer_free(cw->selected_weights[t]);
        if (cw->route_probs[t]) myds_metal_buffer_free(cw->route_probs[t]);
        if (cw->route_scores[t]) myds_metal_buffer_free(cw->route_scores[t]);
        if (cw->route_ids_f32[t]) myds_metal_buffer_free(cw->route_ids_f32[t]);
        if (cw->pool_kv[t]) myds_metal_buffer_free(cw->pool_kv[t]);
        if (cw->pool_score[t]) myds_metal_buffer_free(cw->pool_score[t]);
        if (cw->index_q[t]) myds_metal_buffer_free(cw->index_q[t]);
        if (cw->index_weights[t]) myds_metal_buffer_free(cw->index_weights[t]);
        if (cw->moe_gate[t]) myds_metal_buffer_free(cw->moe_gate[t]);
        if (cw->moe_up[t]) myds_metal_buffer_free(cw->moe_up[t]);
        if (cw->moe_mid[t]) myds_metal_buffer_free(cw->moe_mid[t]);
        if (cw->moe_down[t]) myds_metal_buffer_free(cw->moe_down[t]);
        if (cw->moe_out[t]) myds_metal_buffer_free(cw->moe_out[t]);
        if (cw->ffn_out[t]) myds_metal_buffer_free(cw->ffn_out[t]);
        if (cw->layer_out[t]) myds_metal_buffer_free(cw->layer_out[t]);
        if (cw->res_view[t]) myds_metal_buffer_free(cw->res_view[t]);
        if (cw->pre_view[t]) myds_metal_buffer_free(cw->pre_view[t]);
    }
    if (cw->x_buf) myds_metal_buffer_free(cw->x_buf);
    if (cw->x_norm_buf) myds_metal_buffer_free(cw->x_norm_buf);
    if (cw->qr_buf) myds_metal_buffer_free(cw->qr_buf);
    if (cw->qr_norm_buf) myds_metal_buffer_free(cw->qr_norm_buf);
    if (cw->q_buf) myds_metal_buffer_free(cw->q_buf);
    if (cw->kv_act_buf) myds_metal_buffer_free(cw->kv_act_buf);
    if (cw->attn_heads_buf) myds_metal_buffer_free(cw->attn_heads_buf);
    if (cw->attn_low_buf) myds_metal_buffer_free(cw->attn_low_buf);
    if (cw->attn_out_buf) myds_metal_buffer_free(cw->attn_out_buf);
    if (cw->after_attn_buf) myds_metal_buffer_free(cw->after_attn_buf);
    if (cw->attn_split_buf) myds_metal_buffer_free(cw->attn_split_buf);
    if (cw->ffn_split_buf) myds_metal_buffer_free(cw->ffn_split_buf);
    if (cw->ffn_norm_buf) myds_metal_buffer_free(cw->ffn_norm_buf);
    if (cw->router_logits_buf) myds_metal_buffer_free(cw->router_logits_buf);
    if (cw->shared_gate_buf) myds_metal_buffer_free(cw->shared_gate_buf);
    if (cw->shared_up_buf) myds_metal_buffer_free(cw->shared_up_buf);
    if (cw->shared_swiglu_buf) myds_metal_buffer_free(cw->shared_swiglu_buf);
    if (cw->shared_out_buf) myds_metal_buffer_free(cw->shared_out_buf);
    if (cw->engram_rows_buf[0]) myds_metal_buffer_free(cw->engram_rows_buf[0]);
    if (cw->engram_rows_buf[1]) myds_metal_buffer_free(cw->engram_rows_buf[1]);
    if (cw->selected_ids_buf) myds_metal_buffer_free(cw->selected_ids_buf);
    if (cw->selected_weights_buf) myds_metal_buffer_free(cw->selected_weights_buf);
    if (cw->route_probs_buf) myds_metal_buffer_free(cw->route_probs_buf);
    if (cw->route_scores_buf) myds_metal_buffer_free(cw->route_scores_buf);
    if (cw->route_ids_f32_buf) myds_metal_buffer_free(cw->route_ids_f32_buf);
    if (cw->pool_kv_buf) myds_metal_buffer_free(cw->pool_kv_buf);
    if (cw->pool_score_buf) myds_metal_buffer_free(cw->pool_score_buf);
    if (cw->index_q_buf) myds_metal_buffer_free(cw->index_q_buf);
    if (cw->index_weights_buf) myds_metal_buffer_free(cw->index_weights_buf);
    if (cw->moe_gate_buf) myds_metal_buffer_free(cw->moe_gate_buf);
    if (cw->moe_up_buf) myds_metal_buffer_free(cw->moe_up_buf);
    if (cw->moe_mid_buf) myds_metal_buffer_free(cw->moe_mid_buf);
    if (cw->moe_down_buf) myds_metal_buffer_free(cw->moe_down_buf);
    if (cw->moe_out_buf) myds_metal_buffer_free(cw->moe_out_buf);
    if (cw->ffn_out_buf) myds_metal_buffer_free(cw->ffn_out_buf);
    if (cw->layer_out_buf) myds_metal_buffer_free(cw->layer_out_buf);
    if (cw->cur_res_view) myds_metal_buffer_free(cw->cur_res_view);
    if (cw->cur_pre_view) myds_metal_buffer_free(cw->cur_pre_view);
    if (cw->cur_selected_view) myds_metal_buffer_free(cw->cur_selected_view);
    if (cw->cur_mask_view) myds_metal_buffer_free(cw->cur_mask_view);
    if (cw->hc_flat_norm_buf) myds_metal_buffer_free(cw->hc_flat_norm_buf);
    if (cw->hc_mix_buf) myds_metal_buffer_free(cw->hc_mix_buf);
    free(cw);
}

static v41_chunk_workspace_t *v41_chunk_workspace_create(uint32_t capacity, uint32_t exp_dim) {
    if (!capacity || capacity > V41_MAX_CHUNK) return NULL;
    v41_chunk_workspace_t *cw = calloc(1, sizeof(*cw));
    if (!cw) return NULL;
    cw->capacity = capacity;
    size_t C = capacity;

#define ALLOC_BUF(name, bytes) do { cw->name = myds_metal_buffer_alloc(bytes); if (!cw->name) goto fail; } while(0)
    ALLOC_BUF(x_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(x_norm_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(qr_buf, C * 1280 * sizeof(float));
    ALLOC_BUF(qr_norm_buf, C * 1280 * sizeof(float));
    ALLOC_BUF(q_buf, C * V41_HEADS * 512 * sizeof(float));
    ALLOC_BUF(kv_act_buf, C * 512 * sizeof(float));
    ALLOC_BUF(attn_heads_buf, C * V41_HEADS * 512 * sizeof(float));
    ALLOC_BUF(attn_low_buf, C * 8192 * sizeof(float));
    ALLOC_BUF(attn_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(after_attn_buf, C * 4 * V41_DIM * sizeof(float));
    ALLOC_BUF(attn_split_buf, C * 24 * sizeof(float));
    ALLOC_BUF(ffn_split_buf, C * 24 * sizeof(float));
    ALLOC_BUF(ffn_norm_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(router_logits_buf, C * 384 * sizeof(float));
    ALLOC_BUF(shared_gate_buf, C * exp_dim * sizeof(float));
    ALLOC_BUF(shared_up_buf, C * exp_dim * sizeof(float));
    ALLOC_BUF(shared_swiglu_buf, C * exp_dim * sizeof(float));
    ALLOC_BUF(shared_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(engram_rows_buf[0], C * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    ALLOC_BUF(engram_rows_buf[1], C * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    ALLOC_BUF(selected_ids_buf, C * 6 * sizeof(int32_t));
    ALLOC_BUF(selected_weights_buf, C * 6 * sizeof(float));
    ALLOC_BUF(route_probs_buf, C * 384 * sizeof(float));
    ALLOC_BUF(route_scores_buf, C * 384 * sizeof(float));
    ALLOC_BUF(route_ids_f32_buf, C * 6 * sizeof(float));
    ALLOC_BUF(pool_kv_buf, C * 512 * sizeof(float));
    ALLOC_BUF(pool_score_buf, C * 512 * sizeof(float));
    ALLOC_BUF(index_q_buf, C * V41_INDEX_HEADS * 128 * sizeof(float));
    ALLOC_BUF(index_weights_buf, C * V41_INDEX_HEADS * sizeof(float));
    ALLOC_BUF(moe_gate_buf, C * 6 * exp_dim * sizeof(float));
    ALLOC_BUF(moe_up_buf, C * 6 * exp_dim * sizeof(float));
    ALLOC_BUF(moe_mid_buf, C * 6 * exp_dim * sizeof(float));
    ALLOC_BUF(moe_down_buf, C * 6 * V41_DIM * sizeof(float));
    ALLOC_BUF(moe_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(ffn_out_buf, C * V41_DIM * sizeof(float));
    ALLOC_BUF(layer_out_buf, C * 4 * V41_DIM * sizeof(float));
    ALLOC_BUF(hc_flat_norm_buf, C * 4 * V41_DIM * sizeof(float));
    ALLOC_BUF(hc_mix_buf, C * 24 * sizeof(float));
#undef ALLOC_BUF

    for (size_t t = 0; t < C; t++) {
#define SUB_V(arr, buf, offset_elems, count_elems) do { \
    cw->arr[t] = myds_metal_buffer_sub_view(cw->buf, (t * (offset_elems)) * sizeof(float), (count_elems) * sizeof(float)); \
    if (!cw->arr[t]) goto fail; \
} while(0)
        SUB_V(x, x_buf, V41_DIM, V41_DIM);
        SUB_V(x_norm, x_norm_buf, V41_DIM, V41_DIM);
        SUB_V(qr, qr_buf, 1280, 1280);
        SUB_V(qr_norm, qr_norm_buf, 1280, 1280);
        SUB_V(q, q_buf, V41_HEADS * 512, V41_HEADS * 512);
        SUB_V(kv_act, kv_act_buf, 512, 512);
        SUB_V(attn_heads, attn_heads_buf, V41_HEADS * 512, V41_HEADS * 512);
        SUB_V(attn_low, attn_low_buf, 8192, 8192);
        SUB_V(attn_out, attn_out_buf, V41_DIM, V41_DIM);
        SUB_V(after_attn, after_attn_buf, 4 * V41_DIM, 4 * V41_DIM);
        SUB_V(attn_split, attn_split_buf, 24, 24);
        SUB_V(ffn_split, ffn_split_buf, 24, 24);
        SUB_V(ffn_norm, ffn_norm_buf, V41_DIM, V41_DIM);
        SUB_V(router_logits, router_logits_buf, 384, 384);
        SUB_V(shared_out, shared_out_buf, V41_DIM, V41_DIM);
#undef SUB_V
        cw->engram_rows[0][t] = myds_metal_buffer_sub_view(cw->engram_rows_buf[0], (t * MYDS_ENGRAM_INPUT_DIM) * sizeof(float), MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        cw->engram_rows[1][t] = myds_metal_buffer_sub_view(cw->engram_rows_buf[1], (t * MYDS_ENGRAM_INPUT_DIM) * sizeof(float), MYDS_ENGRAM_INPUT_DIM * sizeof(float));
        if (!cw->engram_rows[0][t] || !cw->engram_rows[1][t]) goto fail;

        cw->selected_ids[t] = myds_metal_buffer_sub_view(cw->selected_ids_buf, t * 6 * sizeof(int32_t), 6 * sizeof(int32_t));
        if (!cw->selected_ids[t]) goto fail;
        cw->selected_weights[t] = myds_metal_buffer_sub_view(cw->selected_weights_buf, t * 6 * sizeof(float), 6 * sizeof(float));
        if (!cw->selected_weights[t]) goto fail;
        cw->route_probs[t] = myds_metal_buffer_sub_view(cw->route_probs_buf, t * 384 * sizeof(float), 384 * sizeof(float));
        if (!cw->route_probs[t]) goto fail;
        cw->route_scores[t] = myds_metal_buffer_sub_view(cw->route_scores_buf, t * 384 * sizeof(float), 384 * sizeof(float));
        if (!cw->route_scores[t]) goto fail;
        cw->route_ids_f32[t] = myds_metal_buffer_sub_view(cw->route_ids_f32_buf, t * 6 * sizeof(float), 6 * sizeof(float));
        if (!cw->route_ids_f32[t]) goto fail;

        cw->pool_kv[t] = myds_metal_buffer_sub_view(cw->pool_kv_buf, t * 512 * sizeof(float), 512 * sizeof(float));
        if (!cw->pool_kv[t]) goto fail;
        cw->pool_score[t] = myds_metal_buffer_sub_view(cw->pool_score_buf, t * 512 * sizeof(float), 512 * sizeof(float));
        if (!cw->pool_score[t]) goto fail;
        cw->index_q[t] = myds_metal_buffer_sub_view(cw->index_q_buf, t * V41_INDEX_HEADS * 128 * sizeof(float), V41_INDEX_HEADS * 128 * sizeof(float));
        if (!cw->index_q[t]) goto fail;
        cw->index_weights[t] = myds_metal_buffer_sub_view(cw->index_weights_buf, t * V41_INDEX_HEADS * sizeof(float), V41_INDEX_HEADS * sizeof(float));
        if (!cw->index_weights[t]) goto fail;

        cw->moe_gate[t] = myds_metal_buffer_sub_view(cw->moe_gate_buf, t * 6 * exp_dim * sizeof(float), 6 * exp_dim * sizeof(float));
        if (!cw->moe_gate[t]) goto fail;
        cw->moe_up[t] = myds_metal_buffer_sub_view(cw->moe_up_buf, t * 6 * exp_dim * sizeof(float), 6 * exp_dim * sizeof(float));
        if (!cw->moe_up[t]) goto fail;
        cw->moe_mid[t] = myds_metal_buffer_sub_view(cw->moe_mid_buf, t * 6 * exp_dim * sizeof(float), 6 * exp_dim * sizeof(float));
        if (!cw->moe_mid[t]) goto fail;
        cw->moe_down[t] = myds_metal_buffer_sub_view(cw->moe_down_buf, t * 6 * V41_DIM * sizeof(float), 6 * V41_DIM * sizeof(float));
        if (!cw->moe_down[t]) goto fail;

        cw->moe_out[t] = myds_metal_buffer_sub_view(cw->moe_out_buf, (t * V41_DIM) * sizeof(float), V41_DIM * sizeof(float));
        cw->ffn_out[t] = myds_metal_buffer_sub_view(cw->ffn_out_buf, (t * V41_DIM) * sizeof(float), V41_DIM * sizeof(float));
        cw->layer_out[t] = myds_metal_buffer_sub_view(cw->layer_out_buf, (t * 4 * V41_DIM) * sizeof(float), 4 * V41_DIM * sizeof(float));
        cw->res_view[t] = myds_metal_buffer_create_view();
        cw->pre_view[t] = myds_metal_buffer_create_view();
        if (!cw->moe_out[t] || !cw->ffn_out[t] || !cw->layer_out[t] || !cw->res_view[t] || !cw->pre_view[t]) goto fail;
    }
    cw->cur_res_view = myds_metal_buffer_create_view();
    cw->cur_pre_view = myds_metal_buffer_create_view();
    cw->cur_selected_view = myds_metal_buffer_create_view();
    cw->cur_mask_view = myds_metal_buffer_create_view();
    if (!cw->cur_res_view || !cw->cur_pre_view || !cw->cur_selected_view || !cw->cur_mask_view) goto fail;
    return cw;
fail:
    v41_chunk_workspace_free(cw);
    return NULL;
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
    if (chunk_size > V41_MAX_CHUNK) chunk_size = V41_MAX_CHUNK;
    if (n_tokens < chunk_size) {
        chunk_size = n_tokens;
        if (chunk_size % 2u != 0) chunk_size++;
        if (chunk_size < 2u) chunk_size = 2u;
        if (chunk_size > V41_MAX_CHUNK) chunk_size = V41_MAX_CHUNK;
    }

    size_t res_size = (size_t)n_tokens * 4u * V41_DIM;
    size_t res_tok_bytes = 4u * V41_DIM * sizeof(float);
    size_t pre_tok_bytes = 24u * sizeof(float);
    size_t sel_bytes = V41_TOPK * sizeof(int32_t);
    size_t mask_bytes = (size_t)myds_metal_buffer_size(g->block_mask);

    myds_metal_buffer_t *prompt_res = myds_metal_buffer_alloc(res_size * sizeof(float));
    myds_metal_buffer_t *prompt_pre = myds_metal_buffer_alloc((size_t)n_tokens * pre_tok_bytes);
    myds_metal_buffer_t *prompt_selected = myds_metal_buffer_alloc((size_t)n_tokens * sel_bytes);
    myds_metal_buffer_t *prompt_block_mask = myds_metal_buffer_alloc((size_t)n_tokens * mask_bytes);
    float *engram_all[2] = {NULL, NULL};
    engram_all[0] = (float *)malloc((size_t)n_tokens * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    engram_all[1] = (float *)malloc((size_t)n_tokens * MYDS_ENGRAM_INPUT_DIM * sizeof(float));
    v41_chunk_workspace_t *cw = v41_chunk_workspace_create(chunk_size, m->moe_expert_dim);

    if (!prompt_res || !prompt_pre || !prompt_selected || !prompt_block_mask ||
        !engram_all[0] || !engram_all[1] || !cw) {
        MYDS_LOG_E("V4.1 prefill failed to allocate workspace buffers");
        v41_chunk_workspace_free(cw);
        myds_metal_buffer_free(prompt_res);
        myds_metal_buffer_free(prompt_pre);
        myds_metal_buffer_free(prompt_selected);
        myds_metal_buffer_free(prompt_block_mask);
        free(engram_all[0]);
        free(engram_all[1]);
        return -1;
    }

    int do_prof = getenv("MYDS_PROFILE_PREFILL") ? 1 : 0;
    double prof_engram = 0, prof_embedding = 0, prof_total_layers = 0;
    double prof_step1 = 0, prof_step2 = 0, prof_step3 = 0, prof_step4 = 0, prof_step5 = 0;
    double prof_step6 = 0, prof_step7 = 0, prof_step8 = 0;
    double prof_wait_qkv = 0, prof_wait_swa = 0, prof_wait_ffn_router = 0, prof_wait_moe = 0;
    uint64_t disp_step14 = 0, disp_step58 = 0, disp_layer_all = 0;  /* 诊断 */
    uint64_t disp_step4loop = 0;  /* 诊断：仅 Step4 逐 token 循环 */
    struct timeval ptv0, ptv1, ptv_all0, ptv_all1;
#define TIME_START() do { if (do_prof) gettimeofday(&ptv0, NULL); } while(0)
#define TIME_ACC(acc) do { if (do_prof) { gettimeofday(&ptv1, NULL); acc += (ptv1.tv_sec - ptv0.tv_sec) * 1000.0 + (ptv1.tv_usec - ptv0.tv_usec) / 1000.0; } } while(0)

    uint32_t il = 0;
    const char *phase = "engram prepare";

    TIME_START();
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
    TIME_ACC(prof_engram);

    phase = "embedding lookup";
    TIME_START();
    const uint16_t *embd_table = (const uint16_t *)myds_metal_buffer_data(m->tok_embeddings);
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t tok = tokens[t];
        if (tok < 0 || (uint32_t)tok >= m->vocab_size) {
            MYDS_LOG_E("V4.1 invalid token %d at pos %u", tok, base_pos + t);
            goto fail;
        }
        const uint16_t *row = embd_table + (size_t)tok * V41_DIM;
        float r[4 * V41_DIM];
        for (uint32_t j = 0; j < V41_DIM; j++) {
            float v_emb = myds_fp16_to_fp32(row[j]);
            r[j] = v_emb;
            r[j + V41_DIM] = v_emb;
            r[j + 2 * V41_DIM] = v_emb;
            r[j + 3 * V41_DIM] = v_emb;
        }
        const float initial_pre[24] = {1.0f, 0.0f, 0.0f, 0.0f};
        if (!myds_metal_buffer_write(prompt_res, (size_t)t * res_tok_bytes, r, res_tok_bytes)) goto fail;
        if (!myds_metal_buffer_write(prompt_pre, (size_t)t * pre_tok_bytes, initial_pre, pre_tok_bytes)) goto fail;

        if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
            memcpy(myds_metal_buffer_data(g->residual), r, res_tok_bytes);
            if (!myds_trace_f32("embedding", 0, base_pos + t, g->residual, 4 * V41_DIM)) goto fail;
        }
    }
    TIME_ACC(prof_embedding);

    phase = "layer loop";
#define CHECK(x) do { if (!(x)) goto fail; } while (0)
#define TRACE(stage,buf,n) CHECK(myds_trace_f32(stage,il,pos,buf,n))

    if (do_prof) gettimeofday(&ptv_all0, NULL);

    for (il = 0; il < V41_LAYERS; il++) {
        if (cancel_flag && *cancel_flag) goto cancelled;
        const myds_layer_t *base = &m->layers[il];
        const v41_layer_t *l = &v->layers[il];
        const bool compressed = ratio_for(il) != 0;

        for (uint32_t c_start = 0; c_start < n_tokens; c_start += chunk_size) {
            if (cancel_flag && *cancel_flag) goto cancelled;
            uint32_t c_len = n_tokens - c_start;
            if (c_len > chunk_size) c_len = chunk_size;

            if (il == 1 || il == 14) {
                uint32_t e = (il == 1 ? 0 : 1);
                CHECK(myds_metal_buffer_write(cw->engram_rows_buf[e], 0,
                    engram_all[e] + (size_t)c_start * MYDS_ENGRAM_INPUT_DIM,
                    (size_t)c_len * MYDS_ENGRAM_INPUT_DIM * sizeof(float)));
            }

            CHECK(myds_metal_begin_batch());

            myds_metal_buffer_update_sub_view(cw->cur_res_view, prompt_res, (size_t)c_start * res_tok_bytes, (size_t)c_len * res_tok_bytes);
            myds_metal_buffer_update_sub_view(cw->cur_pre_view, prompt_pre, (size_t)c_start * pre_tok_bytes, (size_t)c_len * pre_tok_bytes);

            for (uint32_t t = 0; t < c_len; t++) {
                uint32_t idx = c_start + t;
                myds_metal_buffer_update_sub_view(cw->res_view[t], prompt_res, (size_t)idx * res_tok_bytes, res_tok_bytes);
                myds_metal_buffer_update_sub_view(cw->pre_view[t], prompt_pre, (size_t)idx * pre_tok_bytes, 4 * sizeof(float));
            }

            uint64_t disp_layer0 = myds_metal_dispatch_count();  /* 诊断 */

            /* Step 1: HC Attention Inputs for all tokens in chunk */
            TIME_START();
            phase = "HC attention input";

            if (il == 1 || il == 14) {
                uint32_t e = (il == 1 ? 0 : 1);
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;
                    CHECK(project(g->engram_kv, v->engram_proj[e], cw->engram_rows[e][t], 5 * V41_DIM, MYDS_ENGRAM_INPUT_DIM, GGML_TYPE_F16, true));
                    CHECK(myds_metal_v41_engram_add(cw->res_view[t], g->engram_kv, v->engram_q[e], v->engram_k[e], V41_DIM, m->norm_eps));
                    TRACE("engram_out", cw->res_view[t], 4 * V41_DIM);
                }
            }

            /* Batched HC Mix: RMSNorm + GEMM + Split across the entire chunk */
            CHECK(myds_metal_v41_rmsnorm_batch(cw->hc_flat_norm_buf, cw->cur_res_view, NULL, 4 * V41_DIM, c_len, m->norm_eps, false));
            CHECK(project_batch(cw->hc_mix_buf, l->attn_fn, cw->hc_flat_norm_buf, 24, 4 * V41_DIM, c_len, GGML_TYPE_F16, false));
            CHECK(myds_metal_v41_hc_split_batch(cw->attn_split_buf, cw->hc_mix_buf, l->attn_scale, l->attn_base, c_len, v->hc_eps));

            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;
                    TRACE("hc_attn_split", cw->attn_split[t], 24);
                    TRACE("hc_attn_pre", cw->pre_view[t], 4);
                }
            }

            CHECK(myds_metal_v41_hc_weighted_sum_batch(cw->x_buf, cw->cur_res_view, cw->cur_pre_view, V41_DIM, c_len, pre_tok_bytes));
            CHECK(myds_metal_v41_rmsnorm_batch(cw->x_norm_buf, cw->x_buf, base->attn_norm, V41_DIM, c_len, m->norm_eps, true));

            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;
                    TRACE("hc_attn_input", cw->x[t], V41_DIM);
                    TRACE("attn_norm", cw->x_norm[t], V41_DIM);
                }
            }
            TIME_ACC(prof_step1);

            /* Step 2: Batched Q_A and KV projections (reads weights ONCE for all c_len tokens) */
            TIME_START();
            phase = "batched attention projections";
            CHECK(project_batch(cw->qr_buf, base->attn_q_a, cw->x_norm_buf, 1280, V41_DIM, c_len, base->attn_q_a_type, true));
            CHECK(project_batch(cw->kv_act_buf, base->attn_kv, cw->x_norm_buf, 512, V41_DIM, c_len, base->attn_kv_type, true));

            CHECK(myds_metal_v41_rmsnorm_batch(cw->qr_norm_buf, cw->qr_buf, base->attn_q_a_norm, 1280, c_len, m->norm_eps, true));
            CHECK(myds_metal_v41_rmsnorm_batch(cw->kv_act_buf, cw->kv_act_buf, base->attn_kv_a_norm, 512, c_len, m->norm_eps, true));

            /* Batched RoPE & Quant for KV */
            CHECK(myds_v41_gpu_rope_batch(cw->kv_act_buf, 512, 1, base_pos + c_start, c_len, compressed, false));
            CHECK(myds_v41_gpu_quant(cw->kv_act_buf, 512, c_len, 1));

            for (uint32_t t = 0; t < c_len; t++) {
                uint32_t idx = c_start + t;
                uint32_t pos = base_pos + idx;
                TRACE("q_a_raw", cw->qr[t], 1280);
                TRACE("q_a_norm", cw->qr_norm[t], 1280);
                TRACE("kv_raw", cw->kv_act[t], 512);
                TRACE("kv_norm", cw->kv_act[t], 512);
                TRACE("kv_rope", cw->kv_act[t], 512);
                TRACE("kv_quant", cw->kv_act[t], 512);
                /* The sliding-window ring is deliberately NOT written here.  Writing the
                 * whole chunk first and reading it afterwards made each token read the
                 * last tokens of the chunk instead of its own window, because the ring
                 * only holds V41_WINDOW slots.  It is written per token in the attention
                 * loop below, immediately before it is read, which is the order the
                 * per-token path uses. */
            }
            TIME_ACC(prof_step2);

            /* Step 3: Batched Q_B projection (reads 21 MB weight ONCE for all c_len tokens) */
            TIME_START();
            CHECK(project_batch(cw->q_buf, base->attn_q_b, cw->qr_norm_buf, V41_HEADS * 512, 1280, c_len, base->attn_q_b_type, true));

            /* Batched RoPE for Q */
            CHECK(myds_v41_gpu_rope_batch(cw->q_buf, 512, V41_HEADS, base_pos + c_start, c_len, compressed, false));

            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;
                    TRACE("q_b", cw->q[t], V41_HEADS * 512);
                    TRACE("q_rope", cw->q[t], V41_HEADS * 512);
                }
            }
            TIME_ACC(prof_step3);

            if (do_prof) {
                TIME_START();
                CHECK(myds_metal_commit_batch());
                TIME_ACC(prof_wait_qkv);
                CHECK(myds_metal_begin_batch());
            }

            /* Step 4: SWA Attention reduction & indexer */
            TIME_START();
            uint32_t ratio = ratio_for(il);
            if (ratio) {
                if (kv_source(il)) {
                    CHECK(project_batch(cw->pool_kv_buf, l->compressor_kv, cw->x_norm_buf, 512, V41_DIM, c_len, GGML_TYPE_F16, ratio == 1));
                    if (ratio == 2) {
                        CHECK(project_batch(cw->pool_score_buf, l->compressor_gate, cw->x_norm_buf, 512, V41_DIM, c_len, GGML_TYPE_F16, false));
                    }
                }
                if (index_source(il)) {
                    CHECK(project_batch(cw->index_q_buf, l->index_q, cw->qr_norm_buf, V41_INDEX_HEADS * 128, 1280, c_len, GGML_TYPE_F16, true));
                    CHECK(project_batch(cw->index_weights_buf, l->index_proj, cw->x_norm_buf, V41_INDEX_HEADS, V41_DIM, c_len, GGML_TYPE_F16, true));
                }
            }

            uint64_t disp_loop0 = myds_metal_dispatch_count();  /* 诊断 */
            for (uint32_t t = 0; t < c_len; t++) {
                uint32_t idx = c_start + t;
                uint32_t pos = base_pos + idx;
                uint32_t count = compressed ? (pos + 1u) / ratio_for(il) : 0;
                uint32_t top = count < V41_TOPK ? count : V41_TOPK;

                phase = "compressed KV / indexer";
                if (!myds_metal_buffer_update_sub_view(cw->cur_selected_view, prompt_selected, (size_t)idx * sel_bytes, sel_bytes) ||
                    !myds_metal_buffer_update_sub_view(cw->cur_mask_view, prompt_block_mask, (size_t)idx * mask_bytes, mask_bytes)) {
                    goto fail;
                }

                CHECK(publish_and_select(ctx, il, pos, cw->x_norm[t], cw->qr_norm[t],
                                         kv_source(il) ? cw->pool_kv[t] : NULL,
                                         (kv_source(il) && ratio == 2) ? cw->pool_score[t] : NULL,
                                         index_source(il) ? cw->index_q[t] : NULL,
                                         index_source(il) ? cw->index_weights[t] : NULL,
                                         (count && index_source(il)) ? cw->cur_selected_view : NULL,
                                         (il > 20 && index_source(il)) ? cw->cur_mask_view : NULL,
                                         (il == 20) ? cw->cur_mask_view : NULL));

                if (count) {
                    CHECK(myds_v41_gpu_gather(g->selected_kv, g->compressed[owner_for(il)], cw->cur_selected_view, count, top));
                }
                uint32_t nraw = pos + 1u < 128 ? pos + 1u : 128;

                /* Publish this token's KV into the ring immediately before reading it,
                 * so the nraw keys attention sees are exactly the tokens in
                 * [pos+1-nraw, pos].  myds_metal_buffer_copy brackets itself with buffer
                 * memory barriers, so no extra barrier is needed between the two. */
                CHECK(myds_metal_buffer_copy(g->window[il], (size_t)(pos % 128) * 512 * 4u, cw->kv_act[t], 0, 512 * 4u));

                phase = "attention reduction";
                CHECK(myds_v41_gpu_attention(cw->attn_heads[t], base->attn_sinks, cw->q[t], g->window[il], nraw, (pos + 1u - nraw) % 128, g->selected_kv, top, g->gpu));
            }

            /* bf16 + inverse RoPE are elementwise over (token, head) and each token's slice of
             * attn_heads_buf is disjoint, so they run once for the whole chunk instead of once per
             * token.  That removes two dispatches per token per layer.  Positions in the chunk are
             * consecutive, which is what the batched RoPE wants. */
            CHECK(myds_metal_v41_bf16(cw->attn_heads_buf, V41_HEADS * 512 * c_len));
            CHECK(myds_v41_gpu_rope_batch(cw->attn_heads_buf, 512, V41_HEADS, base_pos + c_start, c_len, compressed, true));

            if (do_prof) { disp_step4loop += myds_metal_dispatch_count() - disp_loop0; }  /* 诊断 */

            /* Batched Attention Output A for all c_len tokens in chunk (reads 4.2 MB weight ONCE) */
            CHECK(myds_metal_attn_output_a_batch(cw->attn_low_buf, base->attn_output_a, cw->attn_heads_buf, 8, 4096, 1024, c_len, base->attn_output_a_type));
            CHECK(myds_metal_v41_bf16(cw->attn_low_buf, 8192 * c_len));
            TIME_ACC(prof_step4);
            if (do_prof) { disp_step14 += myds_metal_dispatch_count() - disp_layer0; }  /* 诊断 */

            if (do_prof) {
                TIME_START();
                CHECK(myds_metal_commit_batch());
                TIME_ACC(prof_wait_swa);
                CHECK(myds_metal_begin_batch());
            }

            /* Step 5: Batched Attention Output B (reads 17.8 MB weight ONCE for all c_len tokens) */
            TIME_START();
            phase = "batched attention output b";
            CHECK(project_batch(cw->attn_out_buf, base->attn_output_b, cw->attn_low_buf, V41_DIM, 8192, c_len, base->attn_output_b_type, true));
            TIME_ACC(prof_step5);

            /* Step 6: HC FFN Input for all tokens in chunk */
            TIME_START();
            phase = "HC FFN input";
            CHECK(myds_metal_v41_hc_expand_batch(cw->after_attn_buf, cw->attn_out_buf, cw->cur_res_view, cw->attn_split_buf, V41_DIM, c_len));

            /* Batched HC Mix: RMSNorm + GEMM + Split across the entire chunk */
            CHECK(myds_metal_v41_rmsnorm_batch(cw->hc_flat_norm_buf, cw->after_attn_buf, NULL, 4 * V41_DIM, c_len, m->norm_eps, false));
            CHECK(project_batch(cw->hc_mix_buf, l->ffn_fn, cw->hc_flat_norm_buf, 24, 4 * V41_DIM, c_len, GGML_TYPE_F16, false));
            CHECK(myds_metal_v41_hc_split_batch(cw->ffn_split_buf, cw->hc_mix_buf, l->ffn_scale, l->ffn_base, c_len, v->hc_eps));

            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;
                    TRACE("attn_out", cw->attn_out[t], V41_DIM);
                    TRACE("after_attn", cw->after_attn[t], 4 * V41_DIM);
                    TRACE("hc_ffn_split", cw->ffn_split[t], 24);
                }
            }

            CHECK(myds_metal_v41_hc_weighted_sum_batch(cw->x_buf, cw->after_attn_buf, cw->attn_split_buf, V41_DIM, c_len, 96));
            CHECK(myds_metal_v41_rmsnorm_batch(cw->ffn_norm_buf, cw->x_buf, base->ffn_norm, V41_DIM, c_len, m->norm_eps, true));

            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;
                    TRACE("hc_ffn_input", cw->x[t], V41_DIM);
                    TRACE("ffn_norm", cw->ffn_norm[t], V41_DIM);
                }
            }
            TIME_ACC(prof_step6);

            /* Step 7: Batched Router Gate & Shared Experts (reads 10 MB shared weights + 1.5 MB router ONCE) */
            TIME_START();
            phase = "batched MoE shared and router";
            uint32_t width = m->moe_expert_dim;
            CHECK(project_batch(cw->router_logits_buf, base->router_gate, cw->ffn_norm_buf, m->n_routed_experts, V41_DIM, c_len, base->router_gate_type, false));
            CHECK(project_batch(cw->shared_gate_buf, base->shared_gate, cw->ffn_norm_buf, width, V41_DIM, c_len, base->shared_gate_type, true));
            CHECK(project_batch(cw->shared_up_buf, base->shared_up, cw->ffn_norm_buf, width, V41_DIM, c_len, base->shared_up_type, true));
            CHECK(myds_v41_shared_swiglu(cw->shared_swiglu_buf, cw->shared_gate_buf, cw->shared_up_buf, width * c_len, m->swiglu_clamp_exp));
            CHECK(project_batch(cw->shared_out_buf, base->shared_down, cw->shared_swiglu_buf, V41_DIM, width, c_len, base->shared_down_type, true));

            /* Step 7b: Batched MoE routing for all tokens in chunk on GPU */
            phase = "batched MoE routing";
            uint32_t k = m->n_activated_experts;
            /* Routing is independent per token, so the whole chunk is routed in one pass.
             * This replaces seven dispatches per token (three transform, argsort, sum,
             * divide, scale) with seven dispatches per chunk.  The trace calls are kept
             * per token because they read the same per-token rows the batched kernels write. */
            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;   /* TRACE 需要 pos */
                    (void)pos;
                    TRACE("route_logits", cw->router_logits[t], m->n_routed_experts);
                }
            }
            CHECK(myds_v41_route_batch(cw->selected_ids_buf, cw->selected_weights_buf,
                                       cw->route_probs_buf, cw->route_scores_buf,
                                       cw->router_logits_buf, base->router_bias,
                                       m->n_routed_experts, k, m->routed_scaling_factor, c_len));
            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t pos = base_pos + c_start + t;   /* TRACE 需要 pos */
                    (void)pos;
                    TRACE("route_probs", cw->route_probs[t], m->n_routed_experts);
                    TRACE("route_weights", cw->selected_weights[t], k);
                    const myds_metal_buffer_t *id_buffers[] = {cw->selected_ids[t], cw->route_ids_f32[t]};
                    CHECK(dispatch_k("kernel_myds_v41_ids_f32", &k, sizeof(k), id_buffers, 2, 1, 32, 0));
                    TRACE("route_ids", cw->route_ids_f32[t], k);
                }
            }
            TIME_ACC(prof_step7);

            /* Single commit for all Attention + FFN + Shared + Routing in the chunk */
            if (do_prof) {
                TIME_START();
                CHECK(myds_metal_commit_batch());
                TIME_ACC(prof_wait_ffn_router);
                CHECK(myds_metal_begin_batch());
            } else {
                myds_metal_barrier();
            }

            /* Step 8: Routed MoE & Layer output across all tokens in chunk */
            TIME_START();
            CHECK(myds_metal_routed_moe_batch(cw->moe_out_buf, cw->ffn_norm_buf,
                base->ffn_gate_exps, base->ffn_up_exps, base->ffn_down_exps,
                cw->selected_ids_buf, cw->selected_weights_buf,
                cw->moe_gate_buf, cw->moe_up_buf, cw->moe_mid_buf, cw->moe_down_buf,
                V41_DIM, width, m->n_routed_experts, k, c_len,
                base->ffn_gate_exps_type, base->ffn_up_exps_type, base->ffn_down_exps_type,
                m->swiglu_clamp_exp));

            myds_metal_barrier();
            CHECK(myds_metal_add(cw->ffn_out_buf, cw->moe_out_buf, cw->shared_out_buf, V41_DIM * c_len));
            CHECK(myds_metal_v41_bf16(cw->ffn_out_buf, V41_DIM * c_len));
            myds_metal_barrier();

            CHECK(myds_metal_v41_hc_expand_batch(cw->layer_out_buf, cw->ffn_out_buf, cw->after_attn_buf, cw->ffn_split_buf, V41_DIM, c_len));

            if (getenv("MYDS_BATCH_TRACE_PREFIX")) {
                for (uint32_t t = 0; t < c_len; t++) {
                    uint32_t idx = c_start + t;
                    uint32_t pos = base_pos + idx;
                    TRACE("shared_out", cw->shared_out[t], V41_DIM);
                    TRACE("routed_out", cw->moe_out[t], V41_DIM);
                    TRACE("ffnblock", cw->ffn_out[t], V41_DIM);
                    TRACE("layer_out", cw->layer_out[t], 4 * V41_DIM);
                }
            }

            CHECK(myds_metal_buffer_copy(prompt_res, (size_t)c_start * res_tok_bytes,
                                         cw->layer_out_buf, 0, (size_t)c_len * res_tok_bytes));
            CHECK(myds_metal_buffer_copy(prompt_pre, (size_t)c_start * pre_tok_bytes,
                                         cw->ffn_split_buf, 0, (size_t)c_len * pre_tok_bytes));
            TIME_ACC(prof_step8);

            TIME_START();
            CHECK(myds_metal_commit_batch());
            TIME_ACC(prof_wait_moe);
            if (do_prof) {  /* 诊断 */
                disp_layer_all += myds_metal_dispatch_count() - disp_layer0;
                disp_step58 += myds_metal_dispatch_count() - disp_layer0 - disp_step14;
            }
        }
    }

    if (do_prof) {
        gettimeofday(&ptv_all1, NULL);
        prof_total_layers = (ptv_all1.tv_sec - ptv_all0.tv_sec) * 1000.0 + (ptv_all1.tv_usec - ptv_all0.tv_usec) / 1000.0;
        printf("--- PREFILL PROFILE (%u tokens, %u layers) ---\n", n_tokens, V41_LAYERS);
        printf("Engram Prepare        : %8.2f ms\n", prof_engram);
        printf("Embedding Lookup      : %8.2f ms\n", prof_embedding);
        printf("Step 1 (HC Attn In)   : %8.2f ms\n", prof_step1);
        printf("Step 2 (Q_A & KV GEMV): %8.2f ms\n", prof_step2);
        printf("Step 3 (Q_B GEMV)     : %8.2f ms\n", prof_step3);
        printf("Wait Proj (GPU)       : %8.2f ms\n", prof_wait_qkv);
        printf("Step 4 (SWA Attention): %8.2f ms\n", prof_step4);
        printf("Wait SWA (GPU)        : %8.2f ms\n", prof_wait_swa);
        printf("Step 5 (Attn Out B)   : %8.2f ms\n", prof_step5);
        printf("Step 6 (HC FFN In)    : %8.2f ms\n", prof_step6);
        printf("Step 7 (Shared+Route) : %8.2f ms\n", prof_step7);
        printf("Wait FFN+R (GPU)      : %8.2f ms\n", prof_wait_ffn_router);
        printf("Step 8 (MoE dispatch) : %8.2f ms\n", prof_step8);
        printf("Wait MoE (GPU)        : %8.2f ms\n", prof_wait_moe);
        printf("Total Layers Wall     : %8.2f ms\n", prof_total_layers);
        {   /* 诊断：dispatch 次数 */
            uint64_t layer_iters = (uint64_t)V41_LAYERS * (uint64_t)((n_tokens + chunk_size - 1u) / (chunk_size ? chunk_size : 1u));
            double kvpl = layer_iters ? (double)disp_layer_all / (double)layer_iters : 0.0;
            printf("--- DISPATCH COUNT (诊断) ---\n");
            printf("Step1-4 dispatches    : %10llu\n", (unsigned long long)disp_step14);
            printf("Step5-8 dispatches    : %10llu\n", (unsigned long long)disp_step58);
            printf("Layer dispatches total: %10llu\n", (unsigned long long)disp_layer_all);
            printf("Layer iterations      : %10llu\n", (unsigned long long)layer_iters);
            printf("Step4 loop dispatches : %10llu\n", (unsigned long long)disp_step4loop);
            printf("dispatches / layer-iter: %8.1f  (Step1-4: %.1f, Step4loop: %.1f)\n",
                   kvpl, layer_iters ? (double)disp_step14 / (double)layer_iters : 0.0,
                   layer_iters ? (double)disp_step4loop / (double)layer_iters : 0.0);
        }
        printf("Total Steps Sum       : %8.2f ms\n",
               prof_step1 + prof_step2 + prof_step3 + prof_wait_qkv + prof_step4 + prof_wait_swa +
               prof_step5 + prof_step6 + prof_step7 + prof_wait_ffn_router + prof_step8 + prof_wait_moe);
        fflush(stdout);
    }

    ctx->pos = base_pos + n_tokens;
    g->next_position = ctx->pos;
    g->valid = true;

    size_t last_idx = n_tokens - 1;
    CHECK(myds_metal_buffer_copy(g->residual, 0, prompt_res, last_idx * res_tok_bytes, res_tok_bytes));
    CHECK(myds_metal_buffer_copy(g->pre, 0, prompt_pre, last_idx * pre_tok_bytes, 4 * sizeof(float)));
    CHECK(myds_metal_commit_batch());
#undef CHECK
#undef TRACE

    if (ctx->cfg.temperature > 0.0f) {
        for (uint32_t i = 0; i < n_tokens; i++) {
            ctx->rng_state = ctx->rng_state * 6364136223846793005ULL + 1ULL;
        }
    }

    v41_chunk_workspace_free(cw);
    myds_metal_buffer_free(prompt_res);
    myds_metal_buffer_free(prompt_pre);
    myds_metal_buffer_free(prompt_selected);
    myds_metal_buffer_free(prompt_block_mask);
    free(engram_all[0]);
    free(engram_all[1]);
    return 1;

cancelled:
    (void)myds_metal_commit_batch();
    v41_chunk_workspace_free(cw);
    myds_metal_buffer_free(prompt_res);
    myds_metal_buffer_free(prompt_pre);
    myds_metal_buffer_free(prompt_selected);
    myds_metal_buffer_free(prompt_block_mask);
    free(engram_all[0]);
    free(engram_all[1]);
    return -2;

fail:
    MYDS_LOG_E("V4.1 prefill failed at layer %u: %s", il, phase);
    (void)myds_metal_commit_batch();
    g->valid = false;
    v41_chunk_workspace_free(cw);
    myds_metal_buffer_free(prompt_res);
    myds_metal_buffer_free(prompt_pre);
    myds_metal_buffer_free(prompt_selected);
    myds_metal_buffer_free(prompt_block_mask);
    free(engram_all[0]);
    free(engram_all[1]);
    return -1;
}
