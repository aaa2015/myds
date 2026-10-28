# 2026-10-28 register-only 长范围候选：有序 Top512 门失败

**结论：停止该候选，未接入运行时或生产。** 从冻结的 register-only shader
派生一个仅扩大两个上限的隔离版本，在本机 Apple M4 上进行合成组件验证：
候选与 TILED 的完整 score 位模式均相同，但与当前实际 exact scorer 的
**有序 Top512** 在 8 组中的 5 组不同，不满足预定的有序输出门。

## 身份与范围

- 原 shader SHA256 `a62922c6f5937236bd0e354dd06943b39bfa5f788419c16a8700da793769d616`；
  只在隔离副本把 `n_comp` 上限 16384→65536、结束位置上限 32768→131072。
  候选源 SHA256 `824f83a16d7cbe4352e5cc88446a7a5ecdbf472543d621d3713e4c3c905e05e7`。
- 运行设备是 **Apple M4、本机**，不是 macstudio 的 M2 Ultra；未加载模型权重。
  运行前检查未发现 myds/benchmark 推理进程。原生产源、shader 和 macstudio 服务未触碰。
- 每组 BF16 cache，32 heads、128 维、ratio 2；变换 0/1/2 表示紧密及不同 stride/padding，
  输入轮流包含 BF16 可表示浮点、全 F32 与带符号零、全 F32 与次正规数。
  各组 `n_comp × n_tokens ≤ 2097152`，覆盖因果边界附近的行。
- 总计 **48 个 scorer GPU command buffer 全部完成**，每组 6 个 provider；输出 guard、
  padding、因果尾 `-Inf`、有效分数有限、输入未被修改均通过。候选与 deployed、native、
  probe 的三个 TILED provider 对照，全部完整 score 位模式为 0 差异；两份 actual exact
  彼此也为 0 差异。actual exact 与 TILED 并非逐位等价。
- 最终验证用实际 `myds_v41_gpu_topk_causal`，每组先对同一 exact scores 独立执行两次
  Top512，再对候选 scores 执行一次；三臂均打开 `MYDS_INDEXER_TOPK_PREFIX_MERGE=1`
  和 `MYDS_INDEXER_TOPK_NAN_FUSE=1`。每组 3 个 Top512 GPU batch 完成，两类实际 provider
  计数均为 3，ID guard、因果可见范围及无重复检查通过。逐位置比较两个**有序** ID 列表。
  8 组 exact 自身 A/A 均为 0 差异，候选与 exact 的差异与首轮一致。
- 最终验证 fail-closed 检查 `myds_metal_loaded_library_path()` 的规范路径及 SHA；实际
  Top512 runtime 加载 `/Users/fangqingyuan/Projects/myds/bin/myds.metallib`，SHA 与 scorer
  使用的 deployed 库一致。并在构建前和构建后核对 runtime 源及头文件 SHA。

## 组件结果

“exact vs TILED score 位差”统计矩形内的不同 score 位模式，不是 Top512 差异。
每组候选 vs TILED score 位差均为 0；“Top512 A/A”比较两次独立的 exact 有序列表，
“Top512 A/B”比较 actual exact 与候选有序列表，统计值不同的槽位数。

| case | tokens | n_comp | pos0 | 布局 | 输入 | exact vs TILED score 位差 | Top512 A/A | Top512 A/B |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 8 | 16385 | 32762 | 0 | BF16可表示 | 91,205 | 0 | 0 |
| 1 | 9 | 16385 | 32761 | 1 | F32、signed zero | 120,325 | 0 | 0 |
| 2 | 15 | 32767 | 65519 | 2 | F32、signed zero、subnormal | 401,349 | 0 | 0 |
| 3 | 31 | 32768 | 65505 | 0 | BF16可表示 | 687,436 | 0 | **2** |
| 4 | 63 | 32769 | 65475 | 1 | F32、signed zero | 1,674,797 | 0 | **6** |
| 5 | 31 | 65535 | 131039 | 2 | F32、signed zero、subnormal | 1,607,705 | 0 | **2** |
| 6 | 32 | 65536 | 131040 | 0 | BF16可表示 | 1,423,832 | 0 | **2** |
| 7 | 9 | 65536 | 131063 | 1 | F32、signed zero | 472,750 | 0 | **2** |

例如 case 3，第一处差异是 row 13、排名 233（从 0 开始）：actual exact ID 16638，
候选 ID 12454。**有序 ID 不同不能单凭此结果认定选中集合不同或最终模型输出不同；**
但它已违反本次预先指定的有序门。当前生产配置未启用 `MYDS_INDEXER_TILED`；
源码在超出旧 register-only 准入时按此配置首选 actual exact，本次未捕获生产请求
实际执行的 provider。因此不能只据 TILED 逐位相同就将新范围接入。没有测组件或
全模型性能。

## 实际加载制品

| 标签 | SHA256 |
|---|---|
| deployed TILED/exact（本机 `bin/myds.metallib`） | `fe55793346e15119fce4e0f40d4bf6d3023fc3694ffc231e66b8a89e3ae0f74d` |
| native 完整 TU | `db7f9b071feb102a47d17051c2f5b6ec7de0cf3e18df46e5beaa912f8cb334f0` |
| probe 完整 TU | `ca65e99eb422851cfca547a5494433f7bc9bf363216f65425fbc3feabbf68332` |
| 最终三臂 GPU fixture 二进制 | `79306374f268b4f5a19faa9ae19f4c5e5fb3b4053958d89ffe1317119cd4cc56` |
| 最终三臂 fixture 源 | `c46c9719af9ef9fc767b598146fa3f68b1d3cb15ec2f48f61b058c371d88ec24` |

最终三臂原始记录是 [`raw/aa-gpu-manifest.json`](raw/aa-gpu-manifest.json) 和
[`raw/aa-gpu-events.jsonl`](raw/aa-gpu-events.jsonl)；原样 stdout 在 `raw/aa-gpu-stdout.log`。
manifest 中每组的 `providers` 有 score/guard/input 哈希与比较结果，
`actual_exact_vs_long_top512` 有三臂 Top512 哈希、两类 provider 计数、首差异和不同槽位数。
输入和分数只保存 SHA256 与大小，不保存大张量；它们可由冻结 fixture 重新生成。

最终 cohort 的源码派生、编译命令和身份见 `raw/aa-derivation.json`、
`raw/aa-source-identity.json`、`raw/aa-build-manifest.json` 和 `raw/aa-compile.log`；
当次实际生成的 fixture、候选 shader、构建器分别保存在 `raw/aa-source/`。
`raw/aa-runtime-proof.json` 把最终 cohort 的运行时源码/头文件哈希、Git HEAD、
编译命令、fixture 二进制和两个库的实际身份关联核验。

完整记录了三次依次执行的组件 cohort，未覆盖旧数据：

1. **初次两臂**：exact/candidate 有序差异 0/0/0/2/6/2/2/2，fixture 退出 2；
   `raw/gpu-manifest.json` 和 `raw/initial-source/` 保存当次身份与生成源。
   当时 runtime 日志指向相同 metallib，但 fixture 没有路径与 SHA 的 fail-closed 断言，
   不能把最终 cohort 的身份门追溯赋予首轮。失败后进行了仅编译重建，未提交 GPU，
   结果见 `raw/rebuild-check.json`。
2. **准备性三臂 A/A**：exact A/A 全 0，exact/candidate 仍为 0/0/0/2/6/2/2/2，
   fixture 退出 2、外层退出 1；其时尚未加 runtime 路径身份门。证据以
   `raw/preliminary-aa-*` 命名，生成源在 `raw/preliminary-aa-source/`。
3. **最终三臂 A/A + runtime 路径及 SHA 门**：本 README 的表格；fixture 退出 2、
   外层退出 1。证据以 `raw/aa-*` 命名，生成源在 `raw/aa-source/`。当前工具与
   当前 helper 的 SHA 对应此 cohort，并非首轮或准备性 cohort 的 helper。

## 复现

仅构建和 CPU scope 检查，不提交 GPU 工作：

```bash
python3 tools/run_indexer_register_only_long_probe.py \
  --out-dir /tmp/myds-indexer-long-build
```

确认本机没有其他推理或 benchmark、并拥有独占 GPU 测试窗口后：

```bash
python3 tools/run_indexer_register_only_long_probe.py \
  --out-dir /tmp/myds-indexer-long-replay --run \
  --deployed-metallib bin/myds.metallib
```

输出目录必须全新且在仓库外。该命令会构建隔离 Metal 库、运行组件、保存 manifest 和
结构化事件；不会加载大模型或更改生产文件。如果重现本次门失败，**GPU fixture
退出码为 2，Python 外层命令退出码为 1**，manifest 的状态为 `numerical_failed`。
如果另一设备给出不同结果，应保留其原始记录，不能因此改写本次 M4 结果。

本证据仅说明合成输入下的组件结果；不证明 M2 Ultra、真实模型、完整 logits/state
或端到端性能。模型侧质量与性能均未评估。
