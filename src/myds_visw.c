/*
 * myds_visw.c — 视觉编码器权重加载（见 include/myds_visw.h 的设计说明）
 */
#include "myds_visw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_gguf.h"

/* V4.1 flash 的期望指纹（与 ds4 同一口径） */
#define VIS_EXPECT_ARCH      "deepseek4-vision"
#define VIS_EXPECT_ARCH      "deepseek4-vision"
#define VIS_EXPECT_VARIANT   "v4.1-flash"
#define VIS_EXPECT_TENSORS   266u
#define VIS_EXPECT_BLOCKS    32u
#define VIS_EXPECT_DIM       1024u
#define VIS_EXPECT_FFN       2816u
#define VIS_EXPECT_HEADS     16u
#define VIS_EXPECT_PROJECTION 5120u
#define VIS_EXPECT_PATCH     14u
#define VIS_EXPECT_DOWNSAMPLE 3u
#define VIS_EXPECT_MAX_TOKENS 1024u
#define VIS_EXPECT_MIN_PIXELS 295936u
#define VIS_EXPECT_TOKEN_ID  129264u
#define VIS_EXPECT_RMS_EPS   1.0e-6f

static void visw_error(char *error, size_t cap, const char *msg) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", msg);
}

/* 带张量名的错误信息（比全局信息好查） */
static void visw_error_t(char *error, size_t cap, const char *msg, const char *name) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s: %s", msg, name);
}

static uint64_t tensor_elements(const gguf_tensor_t *t) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->dims[i];
    return n;
}

/* 检查张量：存在、类型正确、维数与期望一致。返回元素数，0 表示失败。 */
static uint64_t require_tensor(const gguf_file_t *gf, const char *name,
                              ggml_type want_type, uint32_t want_rank,
                              const uint64_t *want_dims,
                              char *error, size_t error_cap) {
    const gguf_tensor_t *t = gguf_find_tensor(gf, name);
    if (!t) {
        visw_error_t(error, error_cap, "vision tensor missing", name);
        return 0;
    }
    if (t->type != want_type) {
        if (error && error_cap)
            snprintf(error, error_cap, "vision tensor %s has type %s, expected %s",
                     name, ggml_type_name(t->type), ggml_type_name(want_type));
        return 0;
    }
    if (want_rank && t->n_dims != want_rank) {
        if (error && error_cap)
            snprintf(error, error_cap, "vision tensor %s has %u dims, expected %u",
                     name, t->n_dims, want_rank);
        return 0;
    }
    if (want_dims) {
        for (uint32_t i = 0; i < t->n_dims; i++) {
            if (t->dims[i] != want_dims[i]) {
                if (error && error_cap)
                    snprintf(error, error_cap,
                             "vision tensor %s dim[%u]=%llu, expected %llu",
                             name, i, (unsigned long long)t->dims[i],
                             (unsigned long long)want_dims[i]);
                return 0;
            }
        }
    }
    return tensor_elements(t);
}

/* 两个数都查（含“两个之一”的情形，如 image 向量只需长度对） */
#define REQUIRE(gf, name, type, rank, dims)                                   \
    do {                                                                      \
        uint64_t n_ = require_tensor((gf), (name), (type), (rank), (dims),     \
                                     error, error_cap);                        \
        if (!n_) goto fail;                                                    \
        total += n_;                                                          \
    } while (0)

int myds_vis_weights_probe(const char *path, const char *expected_revision,
                           myds_vis_spec *out,
                           char *error, size_t error_cap) {
    if (!path || !out) return 0;
    memset(out, 0, sizeof(*out));

    gguf_file_t *gf = gguf_open(path);
    if (!gf) {
        visw_error(error, error_cap, "cannot open vision GGUF");
        return 0;
    }

    const char *arch = gguf_get_str(gf, "general.architecture", "");
    if (strcmp(arch, VIS_EXPECT_ARCH) != 0) {
        if (error && error_cap)
            snprintf(error, error_cap,
                     "not a DeepSeek vision encoder: architecture is '%s', expected '%s'",
                     arch, VIS_EXPECT_ARCH);
        goto fail;
    }
    /* 与语言模型同源校验：V4.1 有多个已发布的 checkpoint（df42c109 / 1bae0e10 …），
     * 写死某一个哈希会造成双向错误：新 checkpoint 的正确文件被拒，而错贴旧标签的文件反而放行。
     * 这里只要求【侧车与语言模型声明的 revision 一致】。
     * expected_revision == NULL 表示调用方没有语言模型上下文（单元测试 / 独立工具），
     * 此时跳过同源校验，其余命名/维数/指纹校验照常执行。 */
    if (expected_revision) {
        const char *rev = gguf_get_str(gf, "general.source.revision", "");
        if (!*expected_revision) {
            if (error && error_cap)
                snprintf(error, error_cap,
                         "language checkpoint did not declare a source revision, so the vision "
                         "encoder (revision '%s') cannot be verified", rev);
            goto fail;
        }
        if (strcmp(rev, expected_revision) != 0) {
            if (error && error_cap)
                snprintf(error, error_cap,
                         "vision encoder revision is '%s', language checkpoint is '%s' "
                         "(the sidecar does not match this checkpoint)",
                         rev, expected_revision);
            goto fail;
        }
    }
    const char *variant = gguf_get_str(gf, "deepseek4-vision.checkpoint_variant", "");
    if (strcmp(variant, VIS_EXPECT_VARIANT) != 0) {
        if (error && error_cap)
            snprintf(error, error_cap,
                     "vision encoder variant is '%s', expected '%s'", variant, VIS_EXPECT_VARIANT);
        goto fail;
    }
    if (gf->tensor_count != VIS_EXPECT_TENSORS) {
        if (error && error_cap)
            snprintf(error, error_cap, "vision GGUF has %llu tensors, expected %u",
                     (unsigned long long)gf->tensor_count, VIS_EXPECT_TENSORS);
        goto fail;
    }

    myds_vis_spec *s = out;
    s->block_count   = gguf_get_u32(gf, "deepseek4-vision.block_count", 0);
    s->dim           = gguf_get_u32(gf, "deepseek4-vision.embedding_length", 0);
    s->ffn           = gguf_get_u32(gf, "deepseek4-vision.feed_forward_length", 0);
    s->heads         = gguf_get_u32(gf, "deepseek4-vision.attention.head_count", 0);
    s->projection    = gguf_get_u32(gf, "deepseek4-vision.projection_length", 0);
    s->patch         = gguf_get_u32(gf, "deepseek4-vision.patch_size", 0);
    s->downsample    = gguf_get_u32(gf, "deepseek4-vision.downsample_ratio", 0);
    s->max_tokens    = gguf_get_u32(gf, "deepseek4-vision.image.max_tokens", 0);
    s->min_pixels    = gguf_get_u32(gf, "deepseek4-vision.image.min_pixels", 0);
    s->image_token_id = gguf_get_u32(gf, "deepseek4-vision.image.token_id", 0);
    s->rms_eps       = gguf_get_f32(gf, "deepseek4-vision.attention.layer_norm_rms_epsilon", 0.0f);
    s->tensor_count  = (uint32_t)gf->tensor_count;

#define EXPECT_U(field, key, want)                                            \
    do {                                                                      \
        if (s->field != (want)) {                                              \
            if (error && error_cap)                                            \
                snprintf(error, error_cap, "%s is %u, expected %u",            \
                         (key), s->field, (unsigned)(want));                   \
            goto fail;                                                        \
        }                                                                     \
    } while (0)
    EXPECT_U(block_count, "deepseek4-vision.block_count", VIS_EXPECT_BLOCKS);
    EXPECT_U(dim, "deepseek4-vision.embedding_length", VIS_EXPECT_DIM);
    EXPECT_U(ffn, "deepseek4-vision.feed_forward_length", VIS_EXPECT_FFN);
    EXPECT_U(heads, "deepseek4-vision.attention.head_count", VIS_EXPECT_HEADS);
    EXPECT_U(projection, "deepseek4-vision.projection_length", VIS_EXPECT_PROJECTION);
    EXPECT_U(patch, "deepseek4-vision.patch_size", VIS_EXPECT_PATCH);
    EXPECT_U(downsample, "deepseek4-vision.downsample_ratio", VIS_EXPECT_DOWNSAMPLE);
    EXPECT_U(max_tokens, "deepseek4-vision.image.max_tokens", VIS_EXPECT_MAX_TOKENS);
    EXPECT_U(min_pixels, "deepseek4-vision.image.min_pixels", VIS_EXPECT_MIN_PIXELS);
    EXPECT_U(image_token_id, "deepseek4-vision.image.token_id", VIS_EXPECT_TOKEN_ID);
#undef EXPECT_U
    if (s->rms_eps != VIS_EXPECT_RMS_EPS) {
        if (error && error_cap)
            snprintf(error, error_cap, "vision rms epsilon is %g, expected %g",
                     (double)s->rms_eps, (double)VIS_EXPECT_RMS_EPS);
        goto fail;
    }
    if (s->block_count > MYDS_VIS_MAX_BLOCKS) {
        visw_error(error, error_cap, "vision block count exceeds the engine limit");
        goto fail;
    }

    /* ── 逐个张量校验命名与维数，同时累加元素数 ── */
    const uint32_t D = s->dim, F = s->ffn, P = s->projection, B = s->block_count;
    const uint32_t PS = s->patch * s->patch * 3u;       /* 588 */
    uint64_t total = 0;

    const uint64_t d_ps_d[]   = {PS, D};
    const uint64_t d_d[]      = {D};
    const uint64_t d_3d[]     = {3u * D};
    const uint64_t d_9216_p[] = {9216u, P};
    const uint64_t d_p[]      = {P};
    const uint64_t d_p_p[]    = {P, P};
    const uint64_t d_d_3d[]   = {D, 3u * D};
    const uint64_t d_d_d[]    = {D, D};
    /* ★ SwiGLU：w1 的输出是 2*ffn（gate 与 up 拼在一起），后面才 split 成 ffn。
     *   我第一版按 ffn=2816 写，被维数校验当场抓住（实际 5632）。 */
    const uint64_t d_d_2f[]   = {D, 2u * F};
    const uint64_t d_f_d[]    = {F, D};

    REQUIRE(gf, "vision.patch_embed.proj.weight",     GGML_TYPE_BF16, 2, d_ps_d);
    REQUIRE(gf, "vision.patch_embed.proj.bias",       GGML_TYPE_BF16, 1, d_d);
    REQUIRE(gf, "vision.norm.weight",                 GGML_TYPE_BF16, 1, d_d);
    REQUIRE(gf, "aligner.w1.weight",                  GGML_TYPE_BF16, 2, d_9216_p);
    REQUIRE(gf, "aligner.w1.bias",                    GGML_TYPE_BF16, 1, d_p);
    REQUIRE(gf, "aligner.w2.weight",                  GGML_TYPE_BF16, 2, d_p_p);
    REQUIRE(gf, "aligner.w2.bias",                    GGML_TYPE_BF16, 1, d_p);
    REQUIRE(gf, "image_start",                        GGML_TYPE_BF16, 1, d_p);
    REQUIRE(gf, "image_newline",                      GGML_TYPE_BF16, 1, d_p);
    REQUIRE(gf, "image_end",                          GGML_TYPE_BF16, 1, d_p);

    for (uint32_t il = 0; il < B; il++) {
        char name[128];
#define REQUIRE_LAYER(suffix, type, rank, dims)                               \
        do {                                                                   \
            snprintf(name, sizeof(name), "vision.blocks.%u.%s", il, (suffix));  \
            uint64_t n_ = require_tensor(gf, name, (type), (rank), (dims),      \
                                         error, error_cap);                     \
            if (!n_) goto fail;                                                  \
            total += n_;                                                         \
        } while (0)
        REQUIRE_LAYER("norm1.weight",      GGML_TYPE_BF16, 1, d_d);
        REQUIRE_LAYER("attn.wqkv.weight",  GGML_TYPE_BF16, 2, d_d_3d);
        REQUIRE_LAYER("attn.wqkv.bias",    GGML_TYPE_BF16, 1, d_3d);
        REQUIRE_LAYER("attn.wo.weight",    GGML_TYPE_BF16, 2, d_d_d);
        REQUIRE_LAYER("attn.wo.bias",      GGML_TYPE_BF16, 1, d_d);
        REQUIRE_LAYER("norm2.weight",      GGML_TYPE_BF16, 1, d_d);
        REQUIRE_LAYER("mlp.w1.weight",     GGML_TYPE_BF16, 2, d_d_2f);
        REQUIRE_LAYER("mlp.w2.weight",     GGML_TYPE_BF16, 2, d_f_d);
#undef REQUIRE_LAYER
    }

    /* 266 个张量应当【恰好】被上面覆盖：多出来说明有没接上的东西 */
    if (total == 0) {
        visw_error(error, error_cap, "no vision tensors matched");
        goto fail;
    }
    s->f32_bytes = total * sizeof(float);
    s->param_count_m = (uint32_t)(total / 1000000ull);

    gguf_close(gf);
    return 1;

fail:
    gguf_close(gf);
    return 0;
}

/* ─────────────────── 全量加载 ─────────────────── */

/* BF16 → F32 是【精确】的：BF16 就是 F32 的高 16 位。
 * 不做任何舍入、不做范围检查 —— 一旦带条件分支，就可能在某个值上引入偏差。 */
static void bf16_block_to_f32(const uint16_t *src, float *dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &bits, 4);
    }
}

static myds_metal_buffer_t *buffer_from_bf16_tensor(const gguf_tensor_t *t,
                                                    char *error, size_t error_cap) {
    const uint64_t count = tensor_elements(t);
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        visw_error_t(error, error_cap, "vision tensor element count out of range", t->name);
        return NULL;
    }
    if (t->size_bytes < count * 2u) {
        visw_error_t(error, error_cap, "vision tensor is smaller than its shape implies", t->name);
        return NULL;
    }
    myds_metal_buffer_t *buf = myds_metal_buffer_alloc((size_t)count * sizeof(float));
    if (!buf) {
        visw_error_t(error, error_cap, "cannot allocate vision weight buffer", t->name);
        return NULL;
    }
    /* 分块转换，避免给大张量再开一份等大的中转缓冲 */
    enum { CHUNK = 1u << 16 };
    float *stage = malloc(CHUNK * sizeof(float));
    if (!stage) {
        myds_metal_buffer_free(buf);
        visw_error(error, error_cap, "cannot allocate vision conversion stage");
        return NULL;
    }
    const uint16_t *src = (const uint16_t *)t->data;
    uint64_t done = 0;
    while (done < count) {
        size_t n = (size_t)((count - done) < CHUNK ? (count - done) : CHUNK);
        bf16_block_to_f32(src + done, stage, n);
        if (!myds_metal_buffer_write(buf, (size_t)done * sizeof(float),
                                     stage, n * sizeof(float))) {
            free(stage);
            myds_metal_buffer_free(buf);
            visw_error_t(error, error_cap, "cannot upload vision weight", t->name);
            return NULL;
        }
        done += n;
    }
    free(stage);
    return buf;
}

static myds_metal_buffer_t *load_tensor(const gguf_file_t *gf, const char *name,
                                        char *error, size_t error_cap) {
    const gguf_tensor_t *t = gguf_find_tensor(gf, name);
    if (!t) {
        visw_error_t(error, error_cap, "vision tensor missing", name);
        return NULL;
    }
    return buffer_from_bf16_tensor(t, error, error_cap);
}

int myds_vis_weights_load(const char *path, const char *expected_revision,
                          myds_vis_weights **out,
                          char *error, size_t error_cap) {
    if (!path || !out) return 0;
    *out = NULL;

    myds_vis_spec spec;
    /* 先 probe：把命名/维数/指纹全查一遍，再动手分配 */
    if (!myds_vis_weights_probe(path, expected_revision, &spec, error, error_cap)) return 0;

    gguf_file_t *gf = gguf_open(path);
    if (!gf) {
        visw_error(error, error_cap, "cannot open vision GGUF");
        return 0;
    }

    myds_vis_weights *w = calloc(1, sizeof(*w));
    if (!w) {
        gguf_close(gf);
        visw_error(error, error_cap, "cannot allocate vision weights");
        return 0;
    }
    w->spec = spec;

    w->patch_weight     = load_tensor(gf, "vision.patch_embed.proj.weight", error, error_cap);
    w->patch_bias       = load_tensor(gf, "vision.patch_embed.proj.bias", error, error_cap);
    w->post_norm        = load_tensor(gf, "vision.norm.weight", error, error_cap);
    w->aligner_w1       = load_tensor(gf, "aligner.w1.weight", error, error_cap);
    w->aligner_w1_bias  = load_tensor(gf, "aligner.w1.bias", error, error_cap);
    w->aligner_w2       = load_tensor(gf, "aligner.w2.weight", error, error_cap);
    w->aligner_w2_bias  = load_tensor(gf, "aligner.w2.bias", error, error_cap);
    w->image_start      = load_tensor(gf, "image_start", error, error_cap);
    w->image_newline    = load_tensor(gf, "image_newline", error, error_cap);
    w->image_end        = load_tensor(gf, "image_end", error, error_cap);
    if (!w->patch_weight || !w->patch_bias || !w->post_norm ||
        !w->aligner_w1 || !w->aligner_w1_bias || !w->aligner_w2 ||
        !w->aligner_w2_bias || !w->image_start || !w->image_newline || !w->image_end) {
        goto fail;
    }

    for (uint32_t il = 0; il < spec.block_count; il++) {
        myds_vis_layer_weights *L = &w->layer[il];
        char name[128];
#define LOAD_LAYER(field, suffix)                                             \
        do {                                                                  \
            snprintf(name, sizeof(name), "vision.blocks.%u.%s", il, (suffix)); \
            L->field = load_tensor(gf, name, error, error_cap);                \
            if (!L->field) goto fail;                                          \
        } while (0)
        LOAD_LAYER(norm1,            "norm1.weight");
        LOAD_LAYER(qkv_weight,       "attn.wqkv.weight");
        LOAD_LAYER(qkv_bias,         "attn.wqkv.bias");
        LOAD_LAYER(attn_proj_weight, "attn.wo.weight");
        LOAD_LAYER(attn_proj_bias,   "attn.wo.bias");
        LOAD_LAYER(norm2,            "norm2.weight");
        LOAD_LAYER(mlp_w1,           "mlp.w1.weight");
        LOAD_LAYER(mlp_w2,           "mlp.w2.weight");
#undef LOAD_LAYER
    }

    gguf_close(gf);
    *out = w;
    return 1;

fail:
    gguf_close(gf);
    myds_vis_weights_free(w);
    return 0;
}

void myds_vis_weights_free(myds_vis_weights *w) {
    if (!w) return;
    myds_metal_buffer_free(w->patch_weight);
    myds_metal_buffer_free(w->patch_bias);
    myds_metal_buffer_free(w->post_norm);
    myds_metal_buffer_free(w->aligner_w1);
    myds_metal_buffer_free(w->aligner_w1_bias);
    myds_metal_buffer_free(w->aligner_w2);
    myds_metal_buffer_free(w->aligner_w2_bias);
    myds_metal_buffer_free(w->image_start);
    myds_metal_buffer_free(w->image_newline);
    myds_metal_buffer_free(w->image_end);
    for (uint32_t il = 0; il < MYDS_VIS_MAX_BLOCKS; il++) {
        myds_metal_buffer_free(w->layer[il].norm1);
        myds_metal_buffer_free(w->layer[il].qkv_weight);
        myds_metal_buffer_free(w->layer[il].qkv_bias);
        myds_metal_buffer_free(w->layer[il].attn_proj_weight);
        myds_metal_buffer_free(w->layer[il].attn_proj_bias);
        myds_metal_buffer_free(w->layer[il].norm2);
        myds_metal_buffer_free(w->layer[il].mlp_w1);
        myds_metal_buffer_free(w->layer[il].mlp_w2);
    }
    free(w);
}
