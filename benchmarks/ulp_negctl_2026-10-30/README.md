# 2026-10-30：0-ULP 比对的负向对照（剂量梯度）

账本 §120。补齐 §118 的缺口：**§118 证明了「两套库输出相同」，但没证明「这个比对能看出差异」。**

## 设计

不改 `metal/` 正式源码：`prepare` 把 `metal/` 与 `tools/build_metallib.py` 复制进隔离树，
每臂打一处**文本补丁**（先断言目标在目标 kernel 内恰好出现 1 次），
再用同一工具链、同一 session 构建；未打补丁的那份作 baseline ⇒ **补丁是唯一变量**。

目标 kernel：`kernel_myds_v41_hc_sum_rmsnorm`（`metal/dsv4_hc.metal`）。
`score` 的逐 token forward 每次注意力/FFN 前和最终输出 norm 都走它
（`v41_hc_sum_rmsnorm_scoped`，融合路径默认开启）。

一条关键细节：该路径输出**经过 BF16 舍入**（`round_bf16=true`），
所以舍入前的 1 个 f32 ULP 通常会被吸收——这很可能就是 §118 几个对照都打不出差异的原因。
因此 `bf16ulp` 臂改成**在舍入之后**扰动 1 个 BF16 ULP。

## 四个臂

| 臂 | 扰动 | 预期 |
|---|---|---|
| `gross` | 输出整体 ×(1+2⁻⁶) | **必须 differ**（有效性门） |
| `bf16ulp` | 某元素舍入后 +1 BF16 ULP | differ（灵敏度） |
| `f32ulp` | 舍入前 +1 f32 ULP | 记录（展示被掩蔽） |
| `reorder` | HC 加权和四项累加顺序 c3..c0 | 记录（真实 bug 类别） |

## 结果（`run.jsonl`，SHA `854daa6e…`）

```
valid              : True
baseline runs      : 18       distinct digests: 1     deterministic: True
detects_bf16_ulp   : True

arm        expect         observed
gross      differ         differ      9c79e12e3305f124  (2/2)
bf16ulp    differ         differ      7b5cf057bf292a62  (2/2)
f32ulp     informational  identical   a993fd95c4929717  (2/2)
reorder    informational  identical   a993fd95c4929717  (2/2)
```

### 这证明了什么

1. **基线确定性成立**：同一库启动 18 个进程，digest 只有一个值。
   ⇒ §118 里那两次「同库不同结果」**不是基线不稳**，是别的原因（未能复现，仍记为未解释）。
2. **比对能检出 shader 级数值改动**：`gross` 改了 → 立刻不同。
   这也是有效性门：如果 gross 也相同，说明 kernel 根本不在执行路径上、或 dump 是瞎的。
3. **灵敏度到 1 个 BF16 ULP**：`bf16ulp` 只动一个元素的一个 BF16 单位，仍被检出。

### 这同时揭示了一个限制

`f32ulp`（舍入前 1 f32 ULP）与 `reorder`（累加顺序反转）**都是 identical**。
不是比对失效——是这两类改动**真的没改变最终 logits**，因为 BF16 舍入把差异抹掉了。
⇒ **0-ULP 门拦不住这两类改动**。它们的数值影响存在但落不进这个观测面。
判断 shader 改动是否安全时，只靠 logits 比对会漏掉这一类；需要配合 BF16 前的中间值对照。

## 复现

```bash
# 构建（不需要模型，只需 Metal 工具链）
python3 tools/ulp_negative_control.py prepare \
  --repo ~/Projects/myds --root /tmp/negctl

# 跑（独占窗口，~150 GB wired）
python3 tools/run_register_only_service_window.py \
  --root /tmp/myds-window-negctl --pid $(lsof -nP -t -a -iTCP:8187 -sTCP:LISTEN) \
  --cwd ~/Projects/myfleet --port 8187 --protect myds.conf \
  -- python3 tools/ulp_negative_control.py run --root /tmp/negctl \
       --out /tmp/negctl-run.jsonl --text /tmp/ulp-input/probe.txt \
       --baseline-repeats 10 --arm-repeats 2
```

`--repo` 必须显式给：本工具会被复制到 `/tmp` 与窗口基类并排，
`__file__` 的 `parent.parent` 那时不是仓库根（这个坑在本项目已出现三次）。

## 边界

- 只覆盖 `score` 的**逐 token forward 路径**；**批处理 prefill 未覆盖**。
- 单一 91 B 输入、23 个位置、单一模型。
- `gross` 是 1.6% 的乘性扰动，不是真实 bug 形态；它只用来验证观测面有效。
