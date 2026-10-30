# 2026-10-30：BF16 舍入前累加器探针

账本 §121。目的：§120 的 logits 比对看不见舍入前的变化，需要一个更深的观测点。

## 它做了什么

给 `kernel_myds_v41_hc_sum_rmsnorm_batch` 打「`sum_out[i00] = v` → `= acc`」补丁
（**所有臂都打**，故补丁不是变量），用合成输入调度 kernel，dump 累加器为 f32。
不需要模型、不需要独占窗口；GPU 只跑几微秒，**不是生产测量**。

## 结果（M4，5120 元素）

| 臂 | 舍入前差异 | max |Δ| / BF16 量子 |
|---|---|---|
| `gross`（dump 值 ×1.0009765625） | **5120/5120** | **0.2499** |
| `reorder`（c0..c3 → c3..c0） | **0/5120** | **0.0000** |

`gross` 是有效性门——它不动就说明探针没看到被执行的那段代码。

## 结论：`reorder` 臂什么都没测

反转累加顺序后**舍入前就没有差异**。不是 BF16 抹平，是 `-ffast-math`
让编译器把源码级换序规范成同一段机器码。
⇒ §120 里「观测面盲于重排」不成立；真正被证实的盲区只有亚量子扰动（`f32ulp`）。

## 三次踩错，同一形状

1. 补丁打在 `kernel_myds_v41_hc_sum_rmsnorm`（`dsv4_hc.metal`），但 host 入口
   `..._hc_sum_rmsnorm_batch` 调度的是 **`kernel_myds_v41_hc_sum_rmsnorm_batch`**
   （`dsv4_misc.metal`）。dump 100% 是 BF16 可表示值 ⇒ 拿到的还是舍入后的值。
2. 补丁在整个文件命中 2 次 ⇒ 改为在 kernel 体内定位。
3. 有效性门把扰动加在 `v`（dump 之后）⇒ dump 到的 `acc` 当然不变，读数 0/5120。

三条现在都是断言：kernel 名/文件写死；kernel 体内必须恰好命中 1 次；
dump 若 100% BF16 可表示则 abort。

## 局限

合成填充是 dyadic 值，**81/5120（1.58%）恰好落在 BF16 tie 上**，真实数据不会这样聚集。
「到边界裕度分布」不能代表真实分布；两臂对比不受影响。

## 环境

macmini **也没有 Metal Toolchain**（真实 `.metal` 文件直接报错）⇒ §117 那句「macmini 的
Xcode 路径有 metal」是错的，已在账本更正。
本轮绕法：库在 macstudio 构建、探针在 macmini 跑（库是数据，不绑定机器）。

## 复现

```bash
# 构建（需要 Metal 工具链的机器）
python3 tools/hc_norm_acc_margin.py --root /tmp/margin --out /tmp/margin.jsonl \
  --probe bin/probe_hc_norm_acc

# 或复用别处构建好的臂
python3 tools/hc_norm_acc_margin.py --root /tmp/m --out /tmp/m.jsonl \
  --probe bin/probe_hc_norm_acc \
  --prebuilt baseline=/path/baseline --prebuilt gross=/path/gross \
  --prebuilt reorder=/path/reorder
```

`make bin/probe_hc_norm_acc` 构建探针（不在 `make test` 内）。
