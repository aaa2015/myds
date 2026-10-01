/*
 * myds_vjob.c — 视觉作业（见 include/myds_vjob.h）
 */
#include "myds_vjob.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_internal.h"      /* 取 model->tokenizer */
#include "myds_log.h"
#include "myds_metal.h"
#include "myds_tokenizer.h"
#include "myds_visfwd.h"
#include "myds_vispre.h"
#include "myds_visw.h"

/* 视觉状态。一个进程一套（与引擎上下文同寿命）。 */
static myds_vis_weights *g_w = NULL;
static char             *g_placeholder = NULL;
static uint32_t          g_image_token = 0;
static void (*g_vjob_gpu_lock)(void) = NULL;
static void (*g_vjob_gpu_unlock)(void) = NULL;

void myds_vjob_set_gpu_lock(void (*lock_fn)(void), void (*unlock_fn)(void)) {
    g_vjob_gpu_lock = lock_fn;
    g_vjob_gpu_unlock = unlock_fn;
}

static void vjob_error(char *error, size_t cap, const char *msg) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", msg);
}

int myds_vjob_init(myds_context_t *ctx, const char *vision_gguf) {
    if (g_w) return 1;
    if (!ctx || !vision_gguf || !*vision_gguf) return 0;

    char err[512] = {0};
    if (!myds_vis_weights_load(vision_gguf, &g_w, err, sizeof(err))) {
        MYDS_LOG_E("视觉编码器加载失败: %s", err);
        return 0;
    }
    g_image_token = g_w->spec.image_token_id;
    MYDS_LOG_I("视觉编码器已就绪: %u 层 · projection %u · image token %u · F32 %.2f GiB",
               g_w->spec.block_count, g_w->spec.projection, g_image_token,
               (double)g_w->spec.f32_bytes / 1073741824.0);

    /* ── 取占位 token 的文本，并【验证往返】──
     * 服务端把提示词当字符串交给引擎，所以占位符必须满足：
     * 它的文本编码回来恰好是 1 个 image token。不满足就宁可不用视觉，
     * 也不要让 tokenizer 把图片块编成别的长度 —— 那会让位置对不上、静默出错。 */
    const myds_tokenizer_t *tok = ctx->model ? ctx->model->tokenizer : NULL;
    if (!tok) {
        MYDS_LOG_E("视觉: 拿不到 tokenizer，拒绝启用");
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    const char *piece = myds_tokenizer_decode_piece(tok, (int32_t)g_image_token);
    if (!piece) {
        MYDS_LOG_E("视觉: 词表里没有 token %u 的文本", g_image_token);
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    int32_t back[4] = {0};
    int32_t n = myds_tokenizer_encode(tok, piece, back, 4, false);
    if (n != 1 || back[0] != (int32_t)g_image_token) {
        MYDS_LOG_E("视觉: 占位符往返不一致（%d 个 token），拒绝启用", n);
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    g_placeholder = strdup(piece);
    if (!g_placeholder) {
        myds_vis_weights_free(g_w); g_w = NULL;
        return 0;
    }
    MYDS_LOG_I("视觉占位符: %s（往返一致，1 个 token）", g_placeholder);
    return 1;
}

bool myds_vjob_ready(void) { return g_w != NULL && g_placeholder != NULL; }
const char *myds_vjob_placeholder(void) { return g_placeholder; }
uint32_t myds_vjob_image_token_id(void) { return g_image_token; }

/* 把 F32 缓冲里的哨兵向量读出来（image_start / image_newline / image_end）。
 * 这些是模型学出来的向量，对每个位置都一样，用来标记块的首、行尾、末。 */
static int read_sentinel(const myds_metal_buffer_t *buf, float *out, uint32_t dim) {
    return buf && myds_metal_buffer_read(buf, 0, out, (size_t)dim * sizeof(float));
}

int myds_vjob_plan(const uint8_t *encoded, size_t len, myds_vjob_plan_t *out,
                   char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!myds_vjob_ready()) {
        vjob_error(error, error_cap, "视觉编码器未启用");
        return 0;
    }
    const uint32_t dim = g_w->spec.projection;

    myds_vis_image image;
    if (!myds_vis_image_decode_memory(&image, encoded, len, error, error_cap)) return 0;
    myds_vis_patches pat;
    if (!myds_vis_preprocess(&pat, &image, error, error_cap)) {
        myds_vis_image_free(&image);
        return 0;
    }
    /* ★ 布局用的是【下采样后的 llm 网格】，不是 patch 网格（V4.1 的规矩） */
    myds_vis_layout lay;
    if (!myds_vis_layout_build(&lay, pat.llm_grid_height, pat.llm_grid_width,
                               error, error_cap)) {
        myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }

    /* ★ 编码器输出的行数是【下采样后】的位置数，不是 patch 数：
     *   39×39 个 patch 经 aligner 的 3×3 unfold ⇒ ceil(39/3)² = 13×13 = 169 行。
     *   这与布局里的 image_count 一致（也是 169）。
     *   我第一版拿 patch_count(1521) 去比，于是每张图都报「编码失败」——
     *   而错误信息还是我自己那句笼统的 catch-all，看不出是这里比错了。 */
    float *emb = (float *)malloc((size_t)lay.image_count * dim * sizeof(float));
    if (!emb) {
        vjob_error(error, error_cap, "图片嵌入分配失败");
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }
    uint32_t rows = 0;
    if (g_vjob_gpu_lock) g_vjob_gpu_lock();
    int enc_ok = myds_vis_encode(g_w, &pat, emb, &rows, error, error_cap);
    if (g_vjob_gpu_unlock) g_vjob_gpu_unlock();
    if (!enc_ok) {
        if (error && !error[0]) vjob_error(error, error_cap, "视觉编码失败");
        free(emb);
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }
    if (rows != lay.image_count) {
        /* 位置数对不上 ⇒ 布局与编码器对同一张图的理解不一致，绝不放行（会让向量贴错位置） */
        if (error && error_cap)
            snprintf(error, error_cap, "编码输出 %u 行，但布局需要 %u 个图片位置",
                     rows, lay.image_count);
        free(emb);
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }
    

    /* ── 按布局拼出逐位置的向量 ──
     * IMAGE 位置取编码器输出（按布局给的排列序）；三处哨兵取模型学出来的向量。 */
    const uint32_t n = lay.token_count;
    float *block = (float *)malloc((size_t)n * dim * sizeof(float));
    float *sent_start = (float *)malloc((size_t)dim * sizeof(float));
    float *sent_nl    = (float *)malloc((size_t)dim * sizeof(float));
    float *sent_end   = (float *)malloc((size_t)dim * sizeof(float));
    int ok = block && sent_start && sent_nl && sent_end
             && read_sentinel(g_w->image_start, sent_start, dim)
             && read_sentinel(g_w->image_newline, sent_nl, dim)
             && read_sentinel(g_w->image_end, sent_end, dim);
    if (!ok) {
        vjob_error(error, error_cap, "哨兵向量读取失败");
    } else {
        uint32_t img_pos = 0;
        for (uint32_t i = 0; i < n; i++) {
            float *dst = block + (size_t)i * dim;
            switch (lay.types[i]) {
                case MYDS_VIS_TOKEN_IMAGE: {
                    const uint32_t src = lay.perm[img_pos++];
                    memcpy(dst, emb + (size_t)src * dim, (size_t)dim * sizeof(float));
                    break;
                }
                case MYDS_VIS_TOKEN_START:   memcpy(dst, sent_start, (size_t)dim * sizeof(float)); break;
                case MYDS_VIS_TOKEN_NEWLINE: memcpy(dst, sent_nl,    (size_t)dim * sizeof(float)); break;
                case MYDS_VIS_TOKEN_END:     memcpy(dst, sent_end,   (size_t)dim * sizeof(float)); break;
                default:                     memset(dst, 0, (size_t)dim * sizeof(float)); break;
            }
        }
        if (img_pos != lay.image_count) {
            vjob_error(error, error_cap, "内部错误：布局的图片数与编码输出不符");
            ok = 0;
        }
    }
    free(sent_start); free(sent_nl); free(sent_end); free(emb);
    if (!ok) {
        free(block);
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }

    /* 占位文本 = 占位符重复 n 次（连续，构成一块） */
    const size_t plen = strlen(g_placeholder);
    char *text = (char *)malloc((size_t)n * plen + 1);
    if (!text) {
        vjob_error(error, error_cap, "占位文本分配失败");
        free(block);
        myds_vis_layout_free(&lay); myds_vis_patches_free(&pat); myds_vis_image_free(&image);
        return 0;
    }
    for (uint32_t i = 0; i < n; i++) memcpy(text + (size_t)i * plen, g_placeholder, plen);
    text[(size_t)n * plen] = 0;

    out->token_count = n;
    out->dim = dim;
    out->block = block;
    out->text = text;
    out->grid_height = pat.grid_height;
    out->grid_width = pat.grid_width;
    out->ok = true;

    myds_vis_layout_free(&lay);
    myds_vis_patches_free(&pat);
    myds_vis_image_free(&image);
    return 1;
}

void myds_vjob_plan_free(myds_vjob_plan_t *plan) {
    if (!plan) return;
    free(plan->block);
    free(plan->text);
    memset(plan, 0, sizeof(*plan));
}

int64_t myds_vjob_token_offset_at(myds_context_t *ctx, const char *prompt_base,
                                  const char *block_at) {
    if (!ctx || !prompt_base || !block_at || block_at < prompt_base) return -1;
    const myds_tokenizer_t *tok = ctx->model ? ctx->model->tokenizer : NULL;
    if (!tok) return -1;
    /* 编码「占位块之前的那段」，token 数就是该块的起始下标。
     * ★ add_bos 必须与引擎内部分词时一致（myds_generate_bytes_ex 传 true），
     *   否则会整体差 1 个位置 —— 那会静默地把图片向量贴到错的 token 上。 */
    const size_t prefix_len = (size_t)(block_at - prompt_base);
    char *prefix = (char *)malloc(prefix_len + 1);
    if (!prefix) return -1;
    memcpy(prefix, prompt_base, prefix_len);
    prefix[prefix_len] = 0;
    int32_t count = myds_tokenizer_encode(tok, prefix, NULL, 0, true);
    free(prefix);
    return count < 0 ? -1 : (int64_t)count;
}

/* ── 编码缓存 ──
 * 网关每轮重发完整历史 ⇒ 同一张图会被反复送进来。缓存最近若干张的【整份计划】，
 * 避免每轮把每张图重编一遍（实测 ~6s/图；历史里 3 张图就是每轮 18s）。
 * 键 = 内容哈希（FNV-1a，够用；不需要抗碰撞的强度）。 */
#define VJOB_CACHE_SLOTS 6
typedef struct {
    uint64_t         hash;
    size_t           len;
    myds_vjob_plan_t plan;
    bool             used;
} vjob_cache_t;
static vjob_cache_t   g_cache[VJOB_CACHE_SLOTS];
static uint32_t       g_cache_next;
static uint32_t       g_hits, g_misses;
static pthread_mutex_t g_vjob_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint64_t vjob_hash(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h ^ (uint64_t)n;
}

static int vjob_plan_copy(const myds_vjob_plan_t *src, myds_vjob_plan_t *dst) {
    memset(dst, 0, sizeof(*dst));
    const size_t bbytes = (size_t)src->token_count * src->dim * sizeof(float);
    float *block = (float *)malloc(bbytes);
    char  *text  = src->text ? strdup(src->text) : NULL;
    if (!block || !text) { free(block); free(text); return 0; }
    memcpy(block, src->block, bbytes);
    dst->token_count = src->token_count;
    dst->dim = src->dim;
    dst->block = block;
    dst->text = text;
    dst->grid_height = src->grid_height;
    dst->grid_width = src->grid_width;
    dst->ok = true;
    return 1;
}

uint32_t myds_vjob_cache_hits(void) {
    pthread_mutex_lock(&g_vjob_cache_mutex);
    uint32_t v = g_hits;
    pthread_mutex_unlock(&g_vjob_cache_mutex);
    return v;
}

uint32_t myds_vjob_cache_misses(void) {
    pthread_mutex_lock(&g_vjob_cache_mutex);
    uint32_t v = g_misses;
    pthread_mutex_unlock(&g_vjob_cache_mutex);
    return v;
}

int myds_vjob_sink_add(myds_vjob_sink_t *sink, const uint8_t *encoded, size_t len,
                       char *error, size_t error_cap) {
    if (!sink) return 0;
    if (sink->count == sink->cap) {
        const uint32_t cap = sink->cap ? sink->cap * 2u : 4u;
        myds_vjob_plan_t *np = (myds_vjob_plan_t *)realloc(sink->plans, cap * sizeof(*np));
        if (!np) { vjob_error(error, error_cap, "图片计划表扩容失败"); sink->failed = true; return 0; }
        sink->plans = np;
        sink->cap = cap;
    }
    const uint64_t h = vjob_hash(encoded, len);
    pthread_mutex_lock(&g_vjob_cache_mutex);
    for (uint32_t i = 0; i < VJOB_CACHE_SLOTS; i++) {
        if (g_cache[i].used && g_cache[i].hash == h && g_cache[i].len == len) {
            if (vjob_plan_copy(&g_cache[i].plan, &sink->plans[sink->count])) {
                sink->count++;
                g_hits++;
                pthread_mutex_unlock(&g_vjob_cache_mutex);
                MYDS_LOG_I("图片命中缓存（第 %u 张，%zu 字节）—— 跳过编码", sink->count, len);
                return 1;
            }
            break;   /* 拷贝失败就退回重新编码 */
        }
    }
    pthread_mutex_unlock(&g_vjob_cache_mutex);

    myds_vjob_plan_t fresh;
    if (!myds_vjob_plan(encoded, len, &fresh, error, error_cap)) {
        sink->failed = true;
        if (error) snprintf(sink->why, sizeof(sink->why), "%.190s", error);
        return 0;
    }

    pthread_mutex_lock(&g_vjob_cache_mutex);
    g_misses++;
    /* 存入缓存（覆盖最老的一格）。★ 先存再拷：两份独立的数据，谁都不许 free 到对方。 */
    vjob_cache_t *slot = &g_cache[g_cache_next];
    g_cache_next = (g_cache_next + 1u) % VJOB_CACHE_SLOTS;
    myds_vjob_plan_free(&slot->plan);
    slot->used = false;
    if (vjob_plan_copy(&fresh, &slot->plan)) {
        slot->hash = h;
        slot->len = len;
        slot->used = true;
    }
    pthread_mutex_unlock(&g_vjob_cache_mutex);

    sink->plans[sink->count++] = fresh;   /* 承接所有权 */
    return 1;
}

void myds_vjob_sink_free(myds_vjob_sink_t *sink) {
    if (!sink) return;
    for (uint32_t i = 0; i < sink->count; i++) myds_vjob_plan_free(&sink->plans[i]);
    free(sink->plans);
    memset(sink, 0, sizeof(*sink));
}
