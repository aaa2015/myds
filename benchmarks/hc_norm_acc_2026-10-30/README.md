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

## ⛔ 上面那张表用的是有缺陷的输入（2026-10-30 22:2x 修正）

第一版填充是 dyadic 值（`±(k·j)/1024`），**四项和在 f32 里精确**：

```
f32 正序 == f32 反序        0/5120
f32 正序 == 精确双精度和  5120/5120
```

精确加法与顺序无关 ⇒ 这个输入**在构造上测不出换序**，那个 `0/5120` 是输入的性质，
不是编译器的行为。原先归因给 `-ffast-math` 是错的，已撤回（账本 §121/§122）。

## 修正后的结果（全精度随机填充）

| 臂 | 舍入前差异 | max \|Δ\|/量子 | BF16 舍入后翻转 |
|---|---|---|---|
| `gross` ×1.0009765625 | 5120/5120 | 0.249985 | — |
| `reorder` c0..c3→c3..c0 | **2400/5120** | **0.003296** | **0/5120** |

**结论：换序确实改变了累加器（2400/5120），但幅度只有约 0.3% 个 BF16 量子，
舍入把它们全部吸收 ⇒ 最终 logits 相同。§120 的观察成立，且现在有了量级。**

## 三道前置门（缺一不可）

1. `ordering_gate`：模拟 f32 正序 vs 反序，全同则 abort（当前 39.6% 不同）。
   **这个门就是本次事故的产物。**
2. `gross`：扰动被 dump 的量，必须出现差异，否则说明探针没看到执行路径。
3. `pre_rounding`：dump 若 100% 是 BF16 可表示值则 abort（说明拿到的还是舍入后的值）。

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
