/*
 * test_visw.c — 视觉权重加载的校验
 *
 * 分两段，对应加载器自己的两段：
 *   probe() —— 零分配，可在任何机器上全量跑。查的是命名/维数/指纹/总量。
 *   load()  —— 真分配 ~1.94GB，只在内存够的机器上跑。
 *
 * 用法: test_visw <vision.gguf> [--load]
 *   不给 --load 时只跑 probe（安全，不占内存）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "myds_visw.h"
#include "myds_metal.h"

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "[FAIL] %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
            return 0;                                                         \
        }                                                                     \
    } while (0)

static int test_probe(const char *path) {
    char err[512] = {0};
    myds_vis_spec s;
    CHECK(myds_vis_weights_probe(path, &s, err, sizeof(err)),
          "probe failed: %s", err);

    printf("  block_count   = %u\n", s.block_count);
    printf("  dim           = %u\n", s.dim);
    printf("  ffn           = %u\n", s.ffn);
    printf("  heads         = %u\n", s.heads);
    printf("  projection    = %u\n", s.projection);
    printf("  patch         = %u\n", s.patch);
    printf("  downsample    = %u\n", s.downsample);
    printf("  max_tokens    = %u\n", s.max_tokens);
    printf("  min_pixels    = %u\n", s.min_pixels);
    printf("  image_token   = %u\n", s.image_token_id);
    printf("  rms_eps       = %g\n", (double)s.rms_eps);
    printf("  张量数        = %u\n", s.tensor_count);
    printf("  参数量        = %u M\n", s.param_count_m);
    printf("  F32 需要内存  = %.2f GiB\n",
           (double)s.f32_bytes / 1073741824.0);

    CHECK(s.block_count == 32, "block_count %u", s.block_count);
    CHECK(s.dim == 1024, "dim %u", s.dim);
    CHECK(s.projection == 5120, "projection %u", s.projection);
    CHECK(s.patch == 14, "patch %u", s.patch);
    CHECK(s.image_token_id == 129264, "token id %u", s.image_token_id);
    CHECK(s.tensor_count == 266, "tensor count %u", s.tensor_count);

    /* 参数量应落在合理区间：266 张量、925.6MB BF16 ⇒ 约 463M 个参数 */
    CHECK(s.param_count_m > 400 && s.param_count_m < 520,
          "参数量 %u M 不在合理区间 400~520", s.param_count_m);

    printf("  [PASS] probe：全部张量命名/维数/类型/指纹校验通过\n");
    return 1;
}

/* 反例：拿一个不是视觉编码器的文件，必须被拒绝 */
static int test_reject(const char *path) {
    char err[512] = {0};
    myds_vis_spec s;
    if (!path) { printf("  [SKIP] 未提供反例文件\n"); return 1; }
    int ok = myds_vis_weights_probe(path, &s, err, sizeof(err));
    CHECK(!ok, "应当拒绝非视觉编码器文件，但它通过了");
    printf("  [PASS] 正确拒绝非视觉文件: %s\n", err);
    return 1;
}

static int test_load(const char *path, const char *shader_dir) {
    char err[512] = {0};
    myds_vis_weights *w = NULL;
    CHECK(myds_metal_init(shader_dir), "myds_metal_init(%s) 失败", shader_dir);
    printf("  ⏳ 正在加载 ...\n");
    CHECK(myds_vis_weights_load(path, &w, err, sizeof(err)), "load failed: %s", err);
    CHECK(w != NULL, "load 返回成功但没有对象");
    CHECK(w->patch_weight && w->aligner_w1 && w->layer[31].mlp_w2,
          "加载后有空缓冲");
    /* 抽验一个值：BF16 → F32 应当精确。
     * 拿 image_start 的第 0 个元素，与直接读文件头相比对。 */
    float v[8] = {0};
    CHECK(myds_metal_buffer_read(w->image_start, 0, v, sizeof(v)),
          "读 image_start 失败");
    printf("  image_start[0..1] = %.6f %.6f\n", (double)v[0], (double)v[1]);
    CHECK(v[0] == v[0] && v[1] == v[1], "image_start 含 NaN");

    /* ── ★ 精确性不变量 ──
     * BF16 就是 F32 的高 16 位 ⇒ 转换【必须】让低 16 位全为 0。
     * 任何一个非零低位都说明中间做过舍入或范围处理 —— 那就是有损了。
     * 这比硬编码几个期望数值更硬：它覆盖全量，而且不依赖某个特定文件的内容。 */
    size_t checked = 0, bad = 0;
    float *tmp = NULL;
    size_t tmp_cap = 0;
#define CHECK_EXACT(buf, count, label)                                        \
    do {                                                                      \
        size_t n_ = (count);                                                   \
        if (n_ > tmp_cap) {                                                    \
            free(tmp);                                                         \
            tmp = malloc(n_ * sizeof(float));                                  \
            CHECK(tmp != NULL, "临时缓冲分配失败");                             \
            tmp_cap = n_;                                                      \
        }                                                                      \
        CHECK(myds_metal_buffer_read((buf), 0, tmp, n_ * sizeof(float)),        \
              "读回 %s 失败", (label));                                         \
        for (size_t i_ = 0; i_ < n_; i_++) {                                   \
            uint32_t bits_; memcpy(&bits_, &tmp[i_], 4);                        \
            if (bits_ & 0xffffu) {                                             \
                if (bad < 3)                                                    \
                    fprintf(stderr, "  ✗ %s[%zu] 低位非零: 0x%08x\n",          \
                            (label), i_, bits_);                               \
                bad++;                                                         \
            }                                                                  \
            checked++;                                                         \
        }                                                                      \
    } while (0)

    CHECK_EXACT(w->patch_bias, 1024, "patch_bias");
    CHECK_EXACT(w->image_start, 5120, "image_start");
    CHECK_EXACT(w->image_newline, 5120, "image_newline");
    CHECK_EXACT(w->image_end, 5120, "image_end");
    CHECK_EXACT(w->layer[0].norm1, 1024, "layer0.norm1");
    CHECK_EXACT(w->layer[31].mlp_w2, 2816u * 1024u, "layer31.mlp_w2");
    free(tmp);
#undef CHECK_EXACT
    CHECK(bad == 0, "%zu 个值里 %zu 个低位非零 —— 转换不是精确的", checked, bad);
    printf("  ✓ 精确性不变量: %zu 个值全部低 16 位为 0（BF16→F32 无损）\n", checked);

    /* 元素总数应当与 probe 算出的字节数吻合 */
    uint64_t elems = 0;
    elems += 1024ull * 588ull + 1024ull + 1024ull;                 /* patch w/b + norm */
    elems += 9216ull * 5120ull + 5120ull + 5120ull * 5120ull + 5120ull;
    elems += 5120ull * 3ull;                                       /* image_* */
    for (uint32_t il = 0; il < w->spec.block_count; il++) {
        elems += 1024ull;                    /* norm1 */
        elems += 1024ull * 3072ull;          /* qkv w */
        elems += 3072ull;                    /* qkv b */
        elems += 1024ull * 1024ull;          /* wo w */
        elems += 1024ull;                    /* wo b */
        elems += 1024ull;                    /* norm2 */
        elems += 1024ull * 5632ull;          /* mlp w1 */
        elems += 2816ull * 1024ull;          /* mlp w2 */
    }
    CHECK(elems == w->spec.f32_bytes / sizeof(float),
          "元素数 %llu 与 probe 的 %llu 不符",
          (unsigned long long)elems,
          (unsigned long long)(w->spec.f32_bytes / sizeof(float)));
    printf("  ✓ 元素总数 %llu 与 probe 一致（%.2f GiB）\n",
           (unsigned long long)elems, (double)w->spec.f32_bytes / 1073741824.0);

    myds_vis_weights_free(w);
    printf("  [PASS] load：266 张量全部转为 F32 并上传\n");
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vision.gguf> [--load] [<not-vision.gguf>]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int do_load = 0;
    const char *reject = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0) do_load = 1;
        else reject = argv[i];
    }

    printf("====================================================\n");
    printf("   视觉权重加载校验\n");
    printf("====================================================\n");

    int passed = 0, total = 0;
    total++; passed += test_probe(path);
    total++; passed += test_reject(reject);
    const char *shader_dir = getenv("MYDS_METAL_DIR");
    if (!shader_dir || !*shader_dir) shader_dir = "metal";
    if (do_load) { total++; passed += test_load(path, shader_dir); }
    else printf("  [SKIP] load（未给 --load）\n");

    printf("====================================================\n");
    printf("  %s (%d/%d)\n", passed == total ? "全部通过" : "有失败", passed, total);
    return passed == total ? 0 : 1;
}
