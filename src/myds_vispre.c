/*
 * myds_vispre.c — 图片解码 / 预处理 / 图片 token 布局
 *
 * 忠实移植自 ds4 的 ds4_image.c + ds4_image.h（MIT，见 third_party/iris/LICENSE）。
 * ★ 移植原则：算术次序、舍入方式、常量【一字不改】—— 这样 tests/test_vispre.c
 *   才能拿 ds4 的原始实现当参照做【逐字节】对拍。任何"顺手优化"都会破坏这个门禁。
 *
 * 与 ds4 的差异（仅限不影响数值的部分）：
 *   · 不移植 sha256 指纹（ds4 用它做图片去重；本文件不涉及，且在 rgb 之后计算）
 *   · 错误信息前缀改为 myds
 *   · 不移植 glm53 / qwen4 两条别的视觉通路（本仓库只用 deepseek4）
 */
#include "myds_vispre.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JPEG_IMPLEMENTATION
#define PNG_IMPLEMENTATION
#include "iris/jpeg.h"
#include "iris/png.h"

/* ─────────────────────────── 工具 ─────────────────────────── */

static void vis_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message);
}

static uint32_t vis_align_u32(uint32_t value, uint32_t factor) {
    return (value + factor - 1) / factor * factor;
}

/* ─────────────────── JPEG EXIF 方向 ─────────────────── */

static uint16_t exif_u16(const uint8_t *p, int little) {
    return little ? (uint16_t)(p[0] | (p[1] << 8))
                  : (uint16_t)((p[0] << 8) | p[1]);
}

static uint32_t exif_u32(const uint8_t *p, int little) {
    if (little) {
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
               ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int jpeg_orientation(const uint8_t *data, size_t len) {
    size_t pos = 2;
    while (pos + 4 <= len && data[pos] == 0xff) {
        uint8_t marker = data[pos + 1];
        pos += 2;
        if (marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd9)) continue;
        if (pos + 2 > len) break;
        size_t segment = ((size_t)data[pos] << 8) | data[pos + 1];
        if (segment < 2 || segment > len - pos) break;
        if (marker == 0xe1 && segment >= 14 &&
            memcmp(data + pos + 2, "Exif\0\0", 6) == 0) {
            const uint8_t *tiff = data + pos + 8;
            size_t tiff_len = segment - 8;
            if (tiff_len < 8) break;
            int little = tiff[0] == 'I' && tiff[1] == 'I';
            int big = tiff[0] == 'M' && tiff[1] == 'M';
            if ((!little && !big) || exif_u16(tiff + 2, little) != 42) break;
            uint32_t ifd = exif_u32(tiff + 4, little);
            if (ifd > tiff_len || tiff_len - ifd < 2) break;
            uint16_t count = exif_u16(tiff + ifd, little);
            size_t entries = (size_t)count * 12;
            if (entries > tiff_len - ifd - 2) break;
            for (uint16_t i = 0; i < count; i++) {
                const uint8_t *entry = tiff + ifd + 2 + (size_t)i * 12;
                if (exif_u16(entry, little) == 0x0112 &&
                    exif_u16(entry + 2, little) == 3 &&
                    exif_u32(entry + 4, little) == 1) {
                    int orientation = exif_u16(entry + 8, little);
                    return orientation >= 1 && orientation <= 8 ? orientation : 1;
                }
            }
        }
        if (marker == 0xda) break;
        pos += segment;
    }
    return 1;
}

/* ─────────────────── 解码 ─────────────────── */

static int oriented_rgb(myds_vis_image *out, const uint8_t *pixels,
                        uint32_t width, uint32_t height, uint32_t channels,
                        int orientation) {
    uint32_t out_width = orientation >= 5 ? height : width;
    uint32_t out_height = orientation >= 5 ? width : height;
    size_t bytes = (size_t)out_width * out_height * 3;
    uint8_t *rgb = malloc(bytes);
    if (!rgb) return 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t ox = x, oy = y;
            switch (orientation) {
                case 2: ox = width - 1 - x; break;
                case 3: ox = width - 1 - x; oy = height - 1 - y; break;
                case 4: oy = height - 1 - y; break;
                case 5: ox = y; oy = x; break;
                case 6: ox = height - 1 - y; oy = x; break;
                case 7: ox = height - 1 - y; oy = width - 1 - x; break;
                case 8: ox = y; oy = width - 1 - x; break;
                default: break;
            }
            const uint8_t *src = pixels + ((size_t)y * width + x) * channels;
            uint8_t *dst = rgb + ((size_t)oy * out_width + ox) * 3;
            if (channels < 3) {
                dst[0] = dst[1] = dst[2] = src[0];
            } else {
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
    }

    out->width = out_width;
    out->height = out_height;
    out->rgb = rgb;
    return 1;
}

int myds_vis_image_decode_memory(myds_vis_image *out, const uint8_t *encoded,
                                 size_t encoded_len, char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!encoded || encoded_len == 0 || encoded_len > MYDS_VIS_MAX_ENCODED_BYTES) {
        vis_error(error, error_cap, "image is empty or exceeds the 64 MiB encoded limit");
        return 0;
    }

    if (encoded_len >= 8 && memcmp(encoded, "\x89PNG\r\n\x1a\n", 8) == 0) {
        png_image *decoded = png_load_mem(encoded, encoded_len);
        if (!decoded) {
            vis_error(error, error_cap, "invalid or unsupported PNG image");
            return 0;
        }
        int ok = oriented_rgb(out, decoded->data, (uint32_t)decoded->width,
                              (uint32_t)decoded->height,
                              (uint32_t)decoded->channels, 1);
        png_free(decoded);
        if (!ok) vis_error(error, error_cap, "unable to allocate decoded PNG pixels");
        return ok;
    }
    if (encoded_len >= 2 && encoded[0] == 0xff && encoded[1] == 0xd8) {
        jpeg_image *decoded = jpeg_load_mem(encoded, encoded_len);
        if (!decoded) {
            vis_error(error, error_cap, "invalid or unsupported JPEG image");
            return 0;
        }
        int orientation = jpeg_orientation(encoded, encoded_len);
        int ok = oriented_rgb(out, decoded->data, (uint32_t)decoded->width,
                              (uint32_t)decoded->height,
                              (uint32_t)decoded->channels, orientation);
        jpeg_free(decoded);
        if (!ok) vis_error(error, error_cap, "unable to allocate decoded JPEG pixels");
        return ok;
    }

    vis_error(error, error_cap, "image must be JPEG or PNG");
    return 0;
}

int myds_vis_image_decode_file(myds_vis_image *out, const char *path,
                               char *error, size_t error_cap) {
    if (!out || !path || !path[0]) {
        vis_error(error, error_cap, "image path is empty");
        return 0;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (error && error_cap) snprintf(error, error_cap, "%s: %s", path, strerror(errno));
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) goto io_error;
    long end = ftell(fp);
    if (end <= 0 || (unsigned long)end > MYDS_VIS_MAX_ENCODED_BYTES) {
        fclose(fp);
        vis_error(error, error_cap, "image is empty or exceeds the 64 MiB encoded limit");
        return 0;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) goto io_error;
    size_t len = (size_t)end;
    uint8_t *data = malloc(len);
    if (!data) {
        fclose(fp);
        vis_error(error, error_cap, "unable to allocate encoded image buffer");
        return 0;
    }
    if (fread(data, 1, len, fp) != len) {
        free(data);
        goto io_error;
    }
    fclose(fp);
    int ok = myds_vis_image_decode_memory(out, data, len, error, error_cap);
    free(data);
    return ok;

io_error:
    if (error && error_cap) snprintf(error, error_cap, "%s: %s", path, strerror(errno));
    fclose(fp);
    return 0;
}

void myds_vis_image_free(myds_vis_image *image) {
    if (!image) return;
    free(image->rgb);
    memset(image, 0, sizeof(*image));
}

/* ─────────────────── 双三次缩放 ─────────────────── */

static double vis_cubic(double x, double a) {
    x = fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

static void resize_rgb_bicubic(const uint8_t *src, uint32_t src_width,
                               uint32_t src_height, float *dst, uint32_t dst_width,
                               uint32_t dst_height, uint32_t dst_stride) {
    double scale_x = (double)src_width / dst_width;
    double scale_y = (double)src_height / dst_height;
    double filter_x = scale_x >= 1.0 ? 1.0 / scale_x : 1.0;
    double filter_y = scale_y >= 1.0 ? 1.0 / scale_y : 1.0;
    double support_x = scale_x >= 1.0 ? 2.0 * scale_x : 2.0;
    double support_y = scale_y >= 1.0 ? 2.0 * scale_y : 2.0;

    for (uint32_t dy = 0; dy < dst_height; dy++) {
        double center_y = scale_y * ((double)dy + 0.5);
        int y0 = (int)(center_y - support_y + 0.5);
        int y1 = (int)(center_y + support_y + 0.5);
        if (y0 < 0) y0 = 0;
        if (y1 > (int)src_height) y1 = (int)src_height;
        for (uint32_t dx = 0; dx < dst_width; dx++) {
            double center_x = scale_x * ((double)dx + 0.5);
            int x0 = (int)(center_x - support_x + 0.5);
            int x1 = (int)(center_x + support_x + 0.5);
            if (x0 < 0) x0 = 0;
            if (x1 > (int)src_width) x1 = (int)src_width;
            double sum[3] = {0, 0, 0};
            double weight_sum = 0;
            for (int iy = y0; iy < y1; iy++) {
                double wy = vis_cubic(((double)iy + 0.5 - center_y) * filter_y, -0.5);
                for (int ix = x0; ix < x1; ix++) {
                    double wx = vis_cubic(((double)ix + 0.5 - center_x) * filter_x, -0.5);
                    double weight = wx * wy;
                    const uint8_t *pixel = src +
                        ((size_t)iy * src_width + (uint32_t)ix) * 3;
                    sum[0] += pixel[0] * weight;
                    sum[1] += pixel[1] * weight;
                    sum[2] += pixel[2] * weight;
                    weight_sum += weight;
                }
            }
            float *pixel = dst + ((size_t)dy * dst_stride + dx) * 3;
            for (unsigned c = 0; c < 3; c++) {
                double value = round(sum[c] / weight_sum);
                if (value < 0.0) value = 0.0;
                if (value > 255.0) value = 255.0;
                pixel[c] = (float)value;
            }
        }
    }
}

/* ─────────────────── 尺寸规划 ─────────────────── */

static uint32_t vis_grid_tokens(uint32_t height, uint32_t width,
                                uint32_t *grid_height, uint32_t *grid_width) {
    const uint32_t h = (height / MYDS_VIS_PATCH + 2u) / MYDS_VIS_DOWNSAMPLE;
    const uint32_t w = (width / MYDS_VIS_PATCH + 2u) / MYDS_VIS_DOWNSAMPLE;
    uint64_t count = (uint64_t)h * (w + 1u) + 2u;
    if (grid_height) *grid_height = h;
    if (grid_width) *grid_width = w;
    return count <= UINT32_MAX ? (uint32_t)count : UINT32_MAX;
}

static int vis_solve_resize(uint32_t height, uint32_t width, uint32_t budget,
                            uint32_t *best_height, uint32_t *best_width) {
    const double ratio = (double)height / width;
    const double max_w_float = sqrt(((double)budget - 2.0) / ratio + 0.25) - 0.5;
    const double max_h_float = max_w_float * ratio;
    uint32_t max_h, max_w;
    if (max_w_float < 1.0) {
        max_w = 1;
        max_h = (budget - 2u) / (max_w + 1u);
        if (max_h == 0) return 0;
        *best_width = max_w * 42u;
        *best_height = max_h * 42u;
    } else if (max_h_float < 1.0) {
        max_h = 1u;
        max_w = (budget - 2u) / max_h - 1u;
        if (max_w <= 1u) return 0;
        *best_width = max_w * 42u;
        *best_height = max_h * 42u;
    } else {
        max_w = (uint32_t)floor(max_w_float);
        max_h = (uint32_t)floor(max_h_float);
        if (max_w == 0 || max_h == 0) return 0;
        const double scale = fmin((double)max_w * 42.0 / width,
                                  (double)max_h * 42.0 / height);
        *best_width = (uint32_t)floor(width * scale / 14.0) * 14u;
        *best_height = (uint32_t)floor(height * scale / 14.0) * 14u;
    }
    return *best_width != 0 && *best_height != 0;
}

static int vis_safe_resize(uint32_t height, uint32_t width, uint32_t *best_height,
                           uint32_t *best_width, uint32_t *llm_height,
                           uint32_t *llm_width) {
    const uint32_t max_tokens = 1024u;          /* V4.1 的 deepseek4-vision.image.max_tokens */
    uint32_t budget = max_tokens;
    uint32_t tokens = vis_grid_tokens(*best_height, *best_width, llm_height, llm_width);
    while (tokens > max_tokens) {
        if (budget <= 4u ||
            !vis_solve_resize(height, width, budget, best_height, best_width)) {
            return 0;
        }
        tokens = vis_grid_tokens(*best_height, *best_width, llm_height, llm_width);
        budget--;
    }
    return 1;
}

/* ─────────────────── 预处理（V4.1）─────────────────── */

int myds_vis_preprocess(myds_vis_patches *out, const myds_vis_image *image,
                        char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!image || !image->rgb || image->width == 0 || image->height == 0) {
        vis_error(error, error_cap, "invalid DeepSeek V4 image preprocessing input");
        return 0;
    }

    uint32_t planned_width = image->width;
    uint32_t planned_height = image->height;
    const uint64_t planned_pixels = (uint64_t)planned_width * planned_height;
    const uint32_t min_pixels = 295936u;        /* V4.1 的 deepseek4-vision.image.min_pixels */
    if (planned_pixels < min_pixels) {
        const double scale = sqrt((double)min_pixels / planned_pixels);
        planned_width = (uint32_t)(planned_width * scale);
        planned_height = (uint32_t)(planned_height * scale);
        if (planned_width == 0) planned_width = 1;
        if (planned_height == 0) planned_height = 1;
    }

    uint32_t best_width = vis_align_u32(planned_width, MYDS_VIS_PATCH);
    uint32_t best_height = vis_align_u32(planned_height, MYDS_VIS_PATCH);
    uint32_t llm_height = 0, llm_width = 0;
    if (!vis_safe_resize(planned_height, planned_width, &best_height, &best_width,
                         &llm_height, &llm_width)) {
        vis_error(error, error_cap, "unable to fit image within the vision token budget");
        return 0;
    }
    if (best_width > MYDS_VIS_MAX_DIMENSION || best_height > MYDS_VIS_MAX_DIMENSION ||
        (uint64_t)best_width * best_height > MYDS_VIS_MAX_PIXELS) {
        vis_error(error, error_cap, "resized image exceeds the decoded image limits");
        return 0;
    }

    size_t canvas_values = (size_t)best_width * best_height * 3u;
    float *canvas = malloc(canvas_values * sizeof(*canvas));
    if (!canvas) {
        vis_error(error, error_cap, "unable to allocate resized image");
        return 0;
    }
    for (size_t i = 0; i < canvas_values; i++) canvas[i] = 127.0f;

    const uint32_t content_width_full = best_width;
    const uint32_t content_height_full = best_height;
    uint32_t content_width = content_width_full;
    uint32_t content_height = content_height_full;
    uint32_t offset_x = 0, offset_y = 0;
    {
        const double scale = fmin((double)best_width / image->width,
                                  (double)best_height / image->height);
        content_width = (uint32_t)lrint(image->width * scale);
        content_height = (uint32_t)lrint(image->height * scale);
        if (content_width < 1) content_width = 1;
        if (content_height < 1) content_height = 1;
        if (content_width > best_width) content_width = best_width;
        if (content_height > best_height) content_height = best_height;
        offset_x = (uint32_t)lrint((best_width - content_width) * 0.5);
        offset_y = (uint32_t)lrint((best_height - content_height) * 0.5);
    }

    float *resized = canvas + ((size_t)offset_y * best_width + offset_x) * 3u;
    resize_rgb_bicubic(image->rgb, image->width, image->height, resized,
                       content_width, content_height, best_width);
    for (size_t i = 0; i < canvas_values; i++) {
        canvas[i] = canvas[i] / 127.5f - 1.0f;
    }

    const uint32_t grid_height = best_height / MYDS_VIS_PATCH;
    const uint32_t grid_width = best_width / MYDS_VIS_PATCH;
    const uint32_t patch_count = grid_height * grid_width;
    const size_t patch_values = (size_t)patch_count * 3u * 14u * 14u;
    float *patches = malloc(patch_values * sizeof(*patches));
    if (!patches) {
        free(canvas);
        vis_error(error, error_cap, "unable to allocate vision patches");
        return 0;
    }
    size_t index = 0;
    for (uint32_t patch_y = 0; patch_y < grid_height; patch_y++) {
        for (uint32_t patch_x = 0; patch_x < grid_width; patch_x++) {
            for (uint32_t channel = 0; channel < 3u; channel++) {
                for (uint32_t y = 0; y < 14u; y++) {
                    for (uint32_t x = 0; x < 14u; x++) {
                        const float *pixel = canvas +
                            ((size_t)(patch_y * 14u + y) * best_width +
                             patch_x * 14u + x) * 3u;
                        patches[index++] = pixel[channel];
                    }
                }
            }
        }
    }
    free(canvas);
    if (index != patch_values) {
        free(patches);
        vis_error(error, error_cap, "internal DeepSeek vision patch layout mismatch");
        return 0;
    }

    out->content_width = content_width;
    out->content_height = content_height;
    out->padded_width = best_width;
    out->padded_height = best_height;
    out->grid_width = grid_width;
    out->grid_height = grid_height;
    out->llm_grid_width = llm_width;
    out->llm_grid_height = llm_height;
    out->patch_count = patch_count;
    out->patches = patches;
    return 1;
}

void myds_vis_patches_free(myds_vis_patches *patches) {
    if (!patches) return;
    free(patches->patches);
    memset(patches, 0, sizeof(*patches));
}

/* ─────────────────── 图片 token 布局 ─────────────────── */

int myds_vis_layout_build(myds_vis_layout *out, uint32_t llm_grid_height,
                          uint32_t llm_grid_width,
                          char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    const uint64_t count = (uint64_t)llm_grid_height * ((uint64_t)llm_grid_width + 1u) + 2u;
    if (!llm_grid_height || !llm_grid_width || count > 1024u) {
        vis_error(error, error_cap, "invalid V4.1 image token grid");
        return 0;
    }
    uint8_t *types = malloc((size_t)count);
    uint32_t *perm = malloc((size_t)llm_grid_height * llm_grid_width * sizeof(*perm));
    if (!types || !perm) {
        free(types); free(perm);
        vis_error(error, error_cap, "unable to allocate V4.1 image layout");
        return 0;
    }
    uint32_t token = 0, image = 0;
    types[token++] = MYDS_VIS_TOKEN_START;
    for (uint32_t row = 0; row < llm_grid_height; row++) {
        for (uint32_t col = 0; col < llm_grid_width; col++) {
            types[token++] = MYDS_VIS_TOKEN_IMAGE;
            perm[image] = image;            /* 恒等：编码器输出序 = 排布序 */
            image++;
        }
        types[token++] = MYDS_VIS_TOKEN_NEWLINE;
    }
    types[token++] = MYDS_VIS_TOKEN_END;
    out->token_count = token;
    out->image_count = image;
    out->types = types;
    out->perm = perm;
    return 1;
}

void myds_vis_layout_free(myds_vis_layout *layout) {
    if (!layout) return;
    free(layout->types);
    free(layout->perm);
    memset(layout, 0, sizeof(*layout));
}

/* FNV-1a：只为对拍时把几十 MB 的 patches 压成一个数 */
uint64_t myds_vis_patches_hash(const myds_vis_patches *p) {
    if (!p || !p->patches) return 0;
    uint64_t h = 1469598103934665603ull;
    const size_t n = (size_t)p->patch_count * 3u * 14u * 14u;
    const uint8_t *b = (const uint8_t *)p->patches;
    for (size_t i = 0; i < n * sizeof(float); i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}
