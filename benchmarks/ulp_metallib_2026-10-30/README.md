# 2026-10-30：两套 metallib 的 0-ULP 对照

账本 §118。

## 问题

修复 macstudio 的 Metal 工具链后，重编同一份 `.metal` 源码得到**不同的库**：

| 库 | SHA256 前缀 |
|---|---|
| 已部署 | `fe55793346e15119` |
| 新工具链重编 | `719242c9adf3cb48` |

同大小（4,283,287 B）、MTLB 头相同、18,978 字节（0.44%）不同。
**换库等于换 GPU 代码，所以必须先问：数值上等价吗？**

## 方法

同一个 `score` 可执行文件、同一个模型，只让 `<arm>/myds.metallib` 不同；
`score --dump` 导出**每个位置的完整词表 logits**，逐字节哈希比对。

两条断言，顺序不能换（§C.13 Q9）：
1. **同一臂重复两次必须相同** —— 否则任何对比都无意义；
2. 两臂之间相同。

## 结果

同一输入（91 B 文本，23 个位置），8 次运行（含预热）：

```
old warmup#0  a993fd95c4929717
new warmup#0  a993fd95c4929717
old#0         a993fd95c4929717
new#0         a993fd95c4929717
old#1         a993fd95c4929717
new#1         a993fd95c4929717
old#2         a993fd95c4929717
new#2         a993fd95c4929717
VERDICT pass=True
```

**两套库在本次输入上逐位相同。**

## 三个必须记录的坑（都真实发生过）

**1. 工作目录陷阱 —— 曾导致一次假 PASS。**
`myds_metal_init` 的搜索顺序是 `metal/myds.metallib` → **`./bin/myds.metallib`** →
`./build/...` → `./metal/...` → `<exec_dir>/myds.metallib`。
第 2 项**相对当前工作目录**。从仓库根目录跑时，两个臂都加载了仓库的
`bin/myds.metallib`（部署版），比对的是同一个文件。
已在驱动里改为 `cwd=arm_dir`，并**逐次断言实际加载路径**等于该臂的库
（记录里的 `loaded_ok` 字段）。

**2. 冷启动效应。**
第一次跑这一轮实验时，第 1 个进程得 `fe98df1d`，其后三个都得 `a993fd95`。
已加丢弃式预热；不加的话会报出假差异。

**3. 一次未复现的异常。**
`/tmp/ulp-neg3.jsonl` 里 `new warmup#0` 得 `8093312c`，随后的 `new#0` 与
`old#0` 都是 `a993fd95`。同一库、同一输入、两次结果不同。**未能复现**，
后续 6 次独立运行全部 `a993fd95`。**这是未解释的方差，不是已排除的问题。**

## 未证明的事（重要）

**没能构造出能让比对失败的负向对照**，所以「这个比对能检出 shader 级数值差异」
**未被证明**：

| 尝试 | 结果 |
|---|---|
| 翻转库内 1 字节（偏移 100000） | 仍 `a993fd95`（该字节落在未被使用的区域） |
| `-DDS4_MOE_REVERSE_EXPERTS=1` 重编（反转专家累加顺序，库变 `7ce69b76`） | 仍 `a993fd95`（该路径未被本输入触发） |
| 两个臂指向同一目录 | **被正确拒绝**（同臂守卫） |

另外单独验证了 dump 本身**对输入敏感**：换一段文本得 `0f1d49d8`（12 个位置）
≠ `a993fd95`（23 个位置）。

**⇒ 结论的正确表述**：在本输入上，两套库的输出逐位相同（8/8，加载路径已逐次核实）；
但「本方法能检出真实的 shader 数值差异」未获证明，因为一个能改变结果的对照没造出来。

## 复现

```bash
# 必须在窗口内（模型 ~150 GB wired）
python3 tools/run_register_only_service_window.py \
  --root /tmp/myds-window-ulp3 --pid $(lsof -nP -t -a -iTCP:8187 -sTCP:LISTEN) \
  --cwd ~/Projects/myfleet --port 8187 --protect myds.conf \
  -- python3 tools/ulp_metallib_compare.py \
       --out /tmp/ulp3.jsonl --old-dir /tmp/ulp-old --new-dir /tmp/ulp-new \
       --text /tmp/ulp-input/probe.txt --warmup 1 --repeats 3
```

原始记录：`ulp.jsonl`（首轮，含冷启动误判）、`ulp2.jsonl`（加预热后 PASS）、
`ulp-neg3.jsonl`（负向对照 + 异常）、`ulp-stab.jsonl`（6 次稳定）、
`ulp-sens2.jsonl`（输入敏感性）。
