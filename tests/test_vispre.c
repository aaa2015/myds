/*
 * test_vispre.c — 视觉预处理回归测试与多线程双三次缩放 0-ULP / A-B 验证
 */
#include "myds_vispre.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void check_uniform_resample(uint32_t width, uint32_t height) {
    const uint8_t rgb_value[3] = {23u, 141u, 229u};
    const float expected[3] = {
        23.0f / 127.5f - 1.0f,
        141.0f / 127.5f - 1.0f,
        229.0f / 127.5f - 1.0f,
    };
    const float pad = 127.0f / 127.5f - 1.0f;
    size_t pixels = (size_t)width * height;
    uint8_t *rgb = malloc(pixels * 3u);
    assert(rgb != NULL);
    for (size_t i = 0; i < pixels; i++) {
        rgb[i * 3u + 0u] = rgb_value[0];
        rgb[i * 3u + 1u] = rgb_value[1];
        rgb[i * 3u + 2u] = rgb_value[2];
    }

    myds_vis_image img = {.width = width, .height = height, .rgb = rgb};
    myds_vis_patches pat;
    char err[256] = {0};
    assert(myds_vis_preprocess(&pat, &img, err, sizeof(err)) == 1);

    uint32_t offset_x = (uint32_t)lrint((pat.padded_width - pat.content_width) * 0.5);
    uint32_t offset_y = (uint32_t)lrint((pat.padded_height - pat.content_height) * 0.5);
    size_t index = 0;
    for (uint32_t patch_y = 0; patch_y < pat.grid_height; patch_y++) {
        for (uint32_t patch_x = 0; patch_x < pat.grid_width; patch_x++) {
            for (uint32_t channel = 0; channel < 3u; channel++) {
                for (uint32_t y = 0; y < 14u; y++) {
                    for (uint32_t x = 0; x < 14u; x++) {
                        uint32_t gx = patch_x * 14u + x;
                        uint32_t gy = patch_y * 14u + y;
                        int in_content = gx >= offset_x &&
                                         gx < offset_x + pat.content_width &&
                                         gy >= offset_y &&
                                         gy < offset_y + pat.content_height;
                        float want = in_content ? expected[channel] : pad;
                        assert(fabsf(pat.patches[index++] - want) < 1e-6f);
                    }
                }
            }
        }
    }
    myds_vis_patches_free(&pat);
    free(rgb);
}

static void fill_synthetic_rgb(uint8_t *rgb, uint32_t width, uint32_t height) {
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t idx = ((size_t)y * width + x) * 3u;
            rgb[idx + 0u] = (uint8_t)((x * 7u + y * 3u) & 255u);
            rgb[idx + 1u] = (uint8_t)((x * 3u + y * 11u) & 255u);
            rgb[idx + 2u] = (uint8_t)((x * 13u + y * 5u) & 255u);
        }
    }
}

static void check_multithread_0ulp(uint32_t width, uint32_t height) {
    size_t bytes = (size_t)width * height * 3u;
    uint8_t *rgb = malloc(bytes);
    assert(rgb != NULL);
    fill_synthetic_rgb(rgb, width, height);

    myds_vis_image img = {.width = width, .height = height, .rgb = rgb};
    myds_vis_patches pat1, pat8;
    char err[256] = {0};
    assert(myds_vis_preprocess_ex(&pat1, &img, 1u, err, sizeof(err)) == 1);
    assert(myds_vis_preprocess_ex(&pat8, &img, 8u, err, sizeof(err)) == 1);

    assert(pat1.content_width == pat8.content_width);
    assert(pat1.content_height == pat8.content_height);
    assert(pat1.padded_width == pat8.padded_width);
    assert(pat1.padded_height == pat8.padded_height);
    assert(pat1.grid_width == pat8.grid_width);
    assert(pat1.grid_height == pat8.grid_height);
    assert(pat1.llm_grid_width == pat8.llm_grid_width);
    assert(pat1.llm_grid_height == pat8.llm_grid_height);
    assert(pat1.patch_count == pat8.patch_count);

    size_t patch_bytes = (size_t)pat1.patch_count * 3u * 14u * 14u * sizeof(float);
    assert(memcmp(pat1.patches, pat8.patches, patch_bytes) == 0);
    assert(myds_vis_patches_hash(&pat1) == myds_vis_patches_hash(&pat8));

    myds_vis_patches_free(&pat1);
    myds_vis_patches_free(&pat8);
    free(rgb);
}

static void run_ab_bench(uint32_t width, uint32_t height, int rounds) {
    size_t bytes = (size_t)width * height * 3u;
    uint8_t *rgb = malloc(bytes);
    assert(rgb != NULL);
    fill_synthetic_rgb(rgb, width, height);
    myds_vis_image img = {.width = width, .height = height, .rgb = rgb};
    char err[256] = {0};

    /* Warmup */
    for (int w = 1; w <= 8; w += 7) {
        myds_vis_patches p;
        assert(myds_vis_preprocess_ex(&p, &img, (uint32_t)w, err, sizeof(err)) == 1);
        myds_vis_patches_free(&p);
    }

    double t1_min = 1e9, t1_max = 0.0, t1_sum = 0.0;
    double t8_min = 1e9, t8_max = 0.0, t8_sum = 0.0;
    uint32_t dst_w = 0, dst_h = 0;

    for (int r = 0; r < rounds; r++) {
        myds_vis_patches p1, p8;
        double s1, e1, s8, e8;
        if (r % 2 == 0) {
            s1 = now_ms();
            assert(myds_vis_preprocess_ex(&p1, &img, 1u, err, sizeof(err)) == 1);
            e1 = now_ms();
            s8 = now_ms();
            assert(myds_vis_preprocess_ex(&p8, &img, 8u, err, sizeof(err)) == 1);
            e8 = now_ms();
        } else {
            s8 = now_ms();
            assert(myds_vis_preprocess_ex(&p8, &img, 8u, err, sizeof(err)) == 1);
            e8 = now_ms();
            s1 = now_ms();
            assert(myds_vis_preprocess_ex(&p1, &img, 1u, err, sizeof(err)) == 1);
            e1 = now_ms();
        }
        size_t patch_bytes = (size_t)p1.patch_count * 3u * 14u * 14u * sizeof(float);
        assert(memcmp(p1.patches, p8.patches, patch_bytes) == 0);
        dst_w = p1.padded_width;
        dst_h = p1.padded_height;
        double d1 = e1 - s1, d8 = e8 - s8;
        if (d1 < t1_min) t1_min = d1;
        if (d1 > t1_max) t1_max = d1;
        t1_sum += d1;
        if (d8 < t8_min) t8_min = d8;
        if (d8 > t8_max) t8_max = d8;
        t8_sum += d8;
        myds_vis_patches_free(&p1);
        myds_vis_patches_free(&p8);
    }
    printf("  [BENCH] %ux%u -> %ux%u (%d alternating rounds): "
           "1-thread mean=%.2f ms [%.2f, %.2f] | 8-thread mean=%.2f ms [%.2f, %.2f] "
           "(delta=%.2f ms, %.2fx)\n",
           width, height, dst_w, dst_h, rounds,
           t1_sum / rounds, t1_min, t1_max,
           t8_sum / rounds, t8_min, t8_max,
           (t1_sum - t8_sum) / rounds, t1_sum / t8_sum);
    free(rgb);
}

int main(int argc, char **argv) {
    printf("====================================================\n");
    printf("     Running myds Vision Preprocess Unit Tests      \n");
    printf("====================================================\n");

    check_uniform_resample(613u, 409u);
    check_uniform_resample(90u, 70u);
    check_uniform_resample(1900u, 137u);
    printf("  [PASS] Uniform flat-canvas bicubic resample across thread boundaries (613x409, 90x70, 1900x137)\n");

    const uint32_t cases[][2] = {
        {1u, 1u}, {90u, 70u}, {256u, 256u}, {512u, 512u},
        {613u, 409u}, {1024u, 768u}, {1900u, 137u}, {4032u, 3024u},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        check_multithread_0ulp(cases[i][0], cases[i][1]);
    }
    printf("  [PASS] 1-thread vs 8-thread bicubic resize 0-ULP byte-for-byte identity across 8 image geometries (up to 4032x3024)\n");

    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        run_ab_bench(256u, 256u, 5);
        run_ab_bench(512u, 512u, 5);
        run_ab_bench(1024u, 768u, 5);
        run_ab_bench(4032u, 3024u, 5);
    }

    printf("====================================================\n");
    printf("      ALL VISION PREPROCESS TESTS PASSED!           \n");
    printf("====================================================\n");
    return 0;
}

