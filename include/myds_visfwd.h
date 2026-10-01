/*
 * myds_visfwd.h — 视觉编码器前向（patch → 语言模型维度的嵌入序列）
 *
 * 与 ds4 的 ds4_gpu_deepseek4_vision_encode 是同一套流程，
 * 差别只在实现落点：ds4 用它的 GPU 张量抽象，本引擎用 myds_metal 的缓冲与派发。
 *
 * 数值路线（见 tests/test_metal.c 的实测结论）：
 *   权重在加载期已无损展宽成 F32，矩阵乘走 F32 路径（累加器 simdgroup_float8x8）。
 *   F16 路径的累加器是 half，九千长度的点积会漂到 4e-3，不能用。
 *   编码器本身很小（每张图约 1 GFLOP），F32 的算力代价可以忽略。
 *
 * 「每步舍回 BF16」是这套编码器的核心性格 —— 见 metal/visenc.metal 的说明。
 */
#ifndef MYDS_VISFWD_H
#define MYDS_VISFWD_H

#include <stddef.h>
#include <stdint.h>

#include "myds_vispre.h"
#include "myds_visw.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 编码一张图。out 需能容纳 out_rows * spec.projection 个 float。
 * out_rows 传回实际输出行数（= ceil(gh/3) * ceil(gw/3)）。 */
int myds_vis_encode(const myds_vis_weights *w, const myds_vis_patches *patches,
                    float *out, uint32_t *out_rows, char *error, size_t error_cap);

/* 单步诊断：把某个中间量转储到文件（对拍用）。
 * stage 取值见 src/myds_visfwd.c 的实现；未启用时零开销。 */
void myds_vis_fwd_set_dump(const char *prefix);
int  myds_vis_fwd_dump_enabled(void);

#ifdef __cplusplus
}
#endif
#endif /* MYDS_VISFWD_H */
