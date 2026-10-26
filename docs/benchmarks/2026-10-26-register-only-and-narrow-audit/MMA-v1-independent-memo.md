# M4 register-only v1 独立 CPU 数值失败审计

本轮原始 72 CB 的失败保留：candidate 未通过相对本地 TILED 的 full-score bit-exact 门。所有后续 CPU 分析只解释失败，不重判原 cohort，不包含 GPU 提交、计时或性能采纳。

独立运行冻结 d4 reader 的输出与 root 原 review 逐 byte 相同，分类为 `synthetic_numerical_failed`、`evidence_valid=true`。另用不导入 builder/harness/reader 的 `raw_audit.py` 全量复核 1106 个 raw 文件的 bytes/SHA 和全部完整 score/input/guard，72 个 completion 均 ordinal 正确、status=4、completed=true、error=null。

全部 provider 的因果 -INF、可见 finite、完整写入、padding、output guards、input body/prefix/suffix 前后镜像均通过。每个 case 的 provider 0/1/2 TILED 全 bit 相同，provider 4/5 exact 全 bit 相同；只有 provider 3 的 score bits 违反候选门。

| Case | Candidate 相对 TILED 的完整分数位差 | Exact 相对 TILED 的描述性位差 |
|---:|---:|---:|
| 0 | 11 | 13 |
| 1 | 14 | 17 |
| 2 | 4494 | 6280 |
| 3 | 8862 | 10860 |
| 4 | 148 | 216 |
| 5 | 72205 | 103677 |
| 6 | 82278 | 99388 |
| 7 | 4296 | 6016 |
| 8 | 1133893 | 1597209 |
| 9 | 443727 | 539194 |
| 10 | 4649 | 6524 |
| 11 | 2793 | 3967 |

## 前两组完整可见 cell 的 ULP 描述

ULP 使用 IEEE F32 uint32 单调排序距离（包含正负零的相邻编码），只统计可见有限分数；完整不可见区域的 -INF 已另全量核对。

| Case | 可见 cell | C↔T 位差 / 最大 ULP / 总 ULP | C↔E 位差 / 最大 ULP / 总 ULP | 更近 T / 更近 E / 等距 |
|---:|---:|---|---|---|
| 0 | 16 | 11 / 6 / 20 | 8 / 4 / 16 | 4 / 6 / 6 |
| 1 | 20 | 14 / 72 / 103 | 19 / 54 / 121 | 11 / 5 / 4 |

case0 首差 token1/key0：TILED `3e212ceb` (0.15739791095256805)，candidate 与 exact 均为 `3e212cec` (0.15739792585372925)。case1 同一坐标首差：TILED `be0f766a`，candidate `be0f766b`，exact `be0f7669`。因此不能说 candidate 统一更像 exact。

## 可证伪的 head contraction 线索

读取原始 Q/W/BF16 cache 后，CPU 枚举了 8 个合理 dot 累加模型与 6 个 head 累加模型，共 48 组合；这是在 v1 失败后进行的解释性模型筛查，不是候选性能门或预先冻结的采纳实验。所有预测和每 cell 结果完整保存。

- 同一 d=0..127 顺序 F32 dot 模型，按 head 0..31 执行分开的 F32 `mul(weight,reluDot) → mul(scale) → add(acc)`，case0/1 全部 36 可见 cell 均逐 bit 匹配 TILED。两组中 scalar FMA dot 和 scalar mul/add dot 的最终预测也一致。
- 保持上述 dot 模型，仅 head 改成 `fmaf(reluDot, f32(weight/64), acc)`，全部 36 cell 逐 bit 匹配 candidate。
- balanced/float4-simd-sum dot 模型加相同 head FMA，又能在全部 36 cell 逐 bit 匹配 exact。
- 仅用首差 case0/token1/key0 也有明确中间值预言：head0..25 的两种 head 累加相同，head26 首次分叉。dot `4085b32f`，weight `3f5b0000`，separate accumulator `3e2a22c8`，fused accumulator `3e2a22c9`；最终分别为原 TILED/candidate bits。
- case1/token1/key0 在 head5 首次分叉：dot `405fabba`，weight `3f3d3000`，separate `3b9ab788`，fused `3b9ab784`。

这些 endpoint 匹配强烈支持 head contraction 的定位方向，但没有读取 GPU 的 per-head dot/accumulator，不能排除其它产生相同 endpoint 的 lowering，也不能凭 AIR 中同一串 fast fmul/fadd 证明最终设备指令相同。CPU 使用 host libSystem `fmaf`；其余 primitive 显式 pack 成 F32，当前 `fegetround()==0`。

下一最窄可证伪诊断是新冻结的两条 per-head trace（trace 版各自最终分数首先必须完整重现各自 v1，否则 observer 已改变目标 lowering），然后对比同一坐标全部 32 个 dot/head-acc bits；或仅隔离 head contraction 的单一新版本消融。不能放松 bit-exact 门，也不能把本轮误差改成允许 ULP 后采纳。

## 证据文件

- `independent-reader-review.json`：冻结 reader 对原 raw 的完整复核。
- `raw-audit.json`：独立全 raw 审计与 case0/1 全部差异 cell。
- `raw_audit.py`：不导入原 producer/consumer 的复核脚本。
- `cpu-rounding-models.json` / `cpu_rounding_models.py`：完整 48 数学组合及每 cell 预测。
- `cpu-head-trace-predictions.json`：两组 token1/key0 全 32 head 的事先诊断预言。

原 harness c16、reader d4、tests 0348、协议、build-v2、host fixture 和 v1 GPU raw 均未修改。
