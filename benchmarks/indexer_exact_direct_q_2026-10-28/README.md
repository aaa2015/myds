# Exact indexer direct-Q 候选：M4 组件筛查

**结果：保留结果，不接入生产。** 当前生产配置未启用 TILED，超出原 register-only
范围后源码首选 `kernel_myds_indexer_score_chunk_exact`。本候选从该 exact 内核派生，
将 Q/weights 的 threadgroup 暂存改为各 SIMD group 直接读取，并删除两道每 head
的 TG 屏障。每对输入的 `float4 dot → simd_sum → scale → ReLU → weight → head
顺序累加` 保持不变。它不同于此前 §92 的 TILED 屏障变体和已否决的 TILED
K-resident：后者使用 MMA 算术，本候选沿用 actual exact 的 SIMT 算术。

预先的停止条件为：完整 scores/生产 Top512 任一数值门失败，或组件没有清晰正收益，
即不接入 runtime、不启动昂贵的真实模型试验。真实 indexer 占比和长期测量上下文见
账本 §84、§104；组件局部变化不能直接外推成 prefill 加速。

## 身份与验证范围

- 机器：本机 Apple M4；没有加载模型或触碰 macstudio。合成输入，BF16 K、32 heads、
  head_dim 128、ratio=2；三组 `(n_comp, n_tokens, start)` 为
  `(16385,8,32762)`、`(32768,31,65505)`、`(65536,32,131040)`。组末
  `n_comp=(start+n_tokens)/2`，且 `n_comp*n_tokens ≤ 2097152`，匹配现有 score 预算。
- 在同一完整编译单元中追加候选；原 native 单元作为字节不变的前缀保留。构建器对源
  的更改逆变换后必须逐字节还原原 exact 内核。
- 部署 metallib exact `fe55793346e15119fce4e0f40d4bf6d3023fc3694ffc231e66b8a89e3ae0f74d`；
  候选完整 TU metallib `9cbe9746e47c2726eb08464406e3fbd1089fcf63e473d0bc1b23a31dd99047f4`；
  候选 shader `ee6aa55cdce5caa85c6b4add03a2074217f3c0f8eb6d39a3bd03e047e6d0964a`。
  完整生成源码、命令、源码及公共 header 哈希见 `raw/first-identity.json`。
  `src/myds_decode_candidates.inc` 等传递 include 未由该清单独立记录，故不能把清单称为
  全部编译输入的完整快照；工作树基线是 `f829591`，测试工具不在该提交中。
- 原性能 cohort 使用 `raw/first-source/` 所存探针和 helper。`raw/first-results.log`
  保留原始 stdout（其中含生产 Metal 初始化文本和 JSON 行）。后来发现最初的比较
  没有显式检查全部 score 是否被写入，故补了该断言并以 `--correctness-only`
  运行独立 cohort，**没有重复性能计时**。新旧候选 shader 与完整 TU metallib
  哈希相同，harness 二进制因新增断言而不同；其身份和结果是 `raw/check-*`。

## 数值门

三组中 deployed exact vs 同 TU exact、deployed exact vs direct-Q 的完整 score 位
差异均为 0；因果不可见列为 `-Inf`，可见分数有限，输入与输出 guard 未变。
生产 `myds_v41_gpu_topk_causal` 的 prefix merge 与 NaN fuse 每臂实际编码，
exact A/A 和 exact/candidate 的**有序 Top512 ID**差异均为 0。追加的独立
correctness-only cohort 还确认三臂每个 score 都已从 poison 改写。

仅证明上述合成输入、M4 设备与组件的完整 score/Top512 一致，不证明 M2、真实模型
或完整 logits/state 一致。检查事件记录差异数和 Top512 SHA，不保存完整 score/ID，
不能只靠归档独立重算全部 score；要重新运行 fixture。

## GPU 组件耗时（非全模型）

每组预热三次，以实际 GPU command-buffer 起止时间测量；每个 CB 用 concurrent
compute pass 提交三个相同 dispatch，三个输出缓冲互不别名。7 轮按 A1（同 TU exact）
→ B（direct-Q）→ A2（同 TU exact）夹测。下表为每 dispatch 的 GPU 中位耗时；
候选回退按 `(B / min(A1,A2) - 1)*100`：

| n_comp × n_tokens | A1 ms | B ms | A2 ms | B 相对较快对照 | A2/A1 漂移 |
|---|---:|---:|---:|---:|---:|
| 16385 × 8 | 2.021820 | 2.172528 | 1.997806 | +8.746% | −1.188% |
| 32768 × 31 | 15.662292 | 16.932000 | 15.613944 | +8.442% | −0.309% |
| 65536 × 32 | 32.219792 | 34.973792 | 32.209819 | +8.581% | −0.031% |

第一组 A1/A2 中位漂移超过参考 1% 控制门，不把其数值作为可靠回退判定；
后两组对照漂移较小，候选中位数明确慢于对照。三次 dispatch 在每个 CB 内可以并发，
除以 3 只是相同工作量下的组件对比单位，不是每次独占 GPU 的延迟。这里没有复现
真实模型的其他 concurrent 内核、真实 tensor 分布或 M2 GPU，不能将组件比例当作
生产端到端退化，也不证明变慢的原因是额外 device Q 读取。数值门通过不代表值得采纳。

第一次编译曾因 Objective-C ARC 的对象指针 ownership 标注报错，尚未执行 GPU；
修正签名后用新目录运行了上述唯一性能 cohort，错误日志保留在 `raw/compile-failure.log`。
其后补 score 写入校验的独立 cohort 不含 timing 记录。没有因负结果删除或替换样本。

从原始 GPU 时间重算：

```sh
python3 - <<'PY'
import json, statistics
from pathlib import Path
rows=[]
for line in Path('benchmarks/indexer_exact_direct_q_2026-10-28/raw/first-results.log').read_text().splitlines():
    try: rows.append(json.loads(line))
    except json.JSONDecodeError: pass # Metal 初始化日志不是 JSON
for case in range(3):
    vals={arm:[r['gpu_seconds']*1000/r['dispatches'] for r in rows
               if r.get('kind')=='timing' and r['case']==case and r['arm']==arm]
          for arm in ('A1','B','A2')}
    assert all(len(v)==7 for v in vals.values())
    m={arm:statistics.median(v) for arm,v in vals.items()}
    print(case,m,'B_vs_faster_control_pct',100*(m['B']/min(m['A1'],m['A2'])-1),
          'A2_vs_A1_pct',100*(m['A2']/m['A1']-1))
PY
```

## 重现

构建与 CPU 源变换检查，不触碰 GPU：

```sh
python3 tools/run_indexer_exact_direct_q_probe.py --out-dir /tmp/myds-exact-direct-q-build
```

组件运行必须在无并发推理/benchmark 的机器使用独占 GPU，输出目录必须为仓库外
新路径；只校验数值，不重复原性能计时：

```sh
python3 tools/run_indexer_exact_direct_q_probe.py --out-dir /tmp/myds-exact-direct-q-check \
  --run --correctness-only --deployed-metallib bin/myds.metallib
```

去掉 `--correctness-only` 会自行产生新的性能样本；这些是新实验，不应覆盖本次原始
数据或据此选择更有利的一轮。本工具没有生产接入，不读取权重、不对服务发请求。
