# 2026-10-30：capacity 与 CHUNK_KEY_MAX 的两维对照（E3）

**结论：两个旋钮对 32K prefill 都没有可测影响（≤0.5%）。§105.7 的成因候选 1 被证否，
`B1 KV 缓冲按需增长` 不立项。**

账本记录在 §111。本目录只保存原始数据和重算脚本。

## 为什么不是单维 A/B

§105.7 把 `--ctx-size`（524288→1048576）和 `MYDS_CHUNK_KEY_MAX`（524288→1048576）
当成一组 A/B 同时改动。两个变量、一对臂，无法归因。本轮扫 2×2：

| 臂 | `--ctx-size` | `MYDS_CHUNK_KEY_MAX` | 实际 `key_comp_max` |
|---|---:|---:|---:|
| A | 1048576 | 1048576 | 1048576 |
| B | 1048576 | 524288 | 524288 |
| D | 524288 | 524288 | 524288 |
| C | 524288 | 1048576 | 524288（clamp） |

`myds_v41_gpu_scratch_create` 把 `key_max > max_comp` 夹到 `max_comp`
（`src/myds_v41_gpu.c`），且无任何日志输出 `key_comp_max`，所以 **C 与 B 是同一配置**。
C 仍跑，作为 A/A 对照：B/C 不一致即说明 clamp 推理或换臂流程有错，整轮作废。

## 结果

| 臂 | 8K t/s | 32K t/s |
|---|---:|---:|
| A ctx1M key1M | 535.1 | **614.8** |
| B ctx1M key512K | 535.2 | **615.0** |
| D ctx512K key512K | 535.5 | 609.2 |
| C ctx512K key1M(clamped) | 517.2 * | 612.5 |

| 问题 | 8K | 32K |
|---|---:|---:|
| `CHUNK_KEY_MAX` 1M→512K（ctx 固定） | **+0.02%** | **+0.02%** |
| ctx 1M→512K（key 固定） | −3.36% * | −0.39% |

\* C 的 8K 只有一个异常值（497.9，同臂另一样本 536.6，格内离散 7.48%），
不足以支撑 −3.36%。除该格外，所有格内离散 ≤0.50%，即噪声底线约 0.5%。

## decode 的口径陷阱

HTTP wall time 会得到 27.1 ms/tok，因为它含 prefill 与传输；服务端日志自报
`decode: 160 tok in 3.8s` = **23.67 ms/tok**。两者差 14%。
**decode 只能取服务端日志行**，原样保存在 `raw/decode-lines.txt`。

## 复现

```bash
python3 benchmarks/context_arm_2026-10-30/analyze.py \
  benchmarks/context_arm_2026-10-30/raw/e3-cycle1.jsonl \
  benchmarks/context_arm_2026-10-30/raw/decode-lines.txt
```

在独占窗口内重跑整轮：

```bash
python3 tools/run_register_only_service_window.py \
  --root /tmp/myds-window-e3b --pid $(lsof -nP -t -a -iTCP:8187 -sTCP:LISTEN) \
  --cwd ~/Projects/myfleet --port 8187 --protect myds.conf \
  -- python3 tools/bench_context_arm.py --out /tmp/e3-run2.jsonl \
    --cycles 3 --arms A,B,D,C --sizes 8k,32k --samples 3
```

## 边界

- n=2/格，只够判定「效应 <0.5%」，不足以给每格定精确值。要判亚百分比差异需 ≥9 样本/臂并交错。
- 未测 90K 以上，未测 warm 追加（E2），未改任何生产配置。
- 制品 SHA `bbe8aa66…`，与 §105.6 记录的 `a97976e0…` 不同（期间有过一次部署）；
  本轮 32K 与 decode 均与 §105.7 一致，未观察到由此产生的退化。
