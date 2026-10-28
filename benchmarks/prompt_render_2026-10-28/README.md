# 2026-10-28：工具结果的请求渲染优化

**结论：仅优化工具结果的连续普通字节；保留原 JSON schema 字符串实现。**
原 `prompt_tool_text` 为每个普通字节执行一次容量检查和追加；候选合并连续字节，
仅在需要转义时分段。`</tool_result>` 和多层 `&amp;...lt;/tool_result>` 的转义规则
不变。余量耗尽立即停止扫描，并按原代码先写入能放下的普通字节；转义序列仍为
原有原子追加。没有修改模板、tokenizer、模型计算或 GPU 调度。

## 测量与身份

- macstudio M2 Ultra，192 GiB，macOS 27.2，Apple clang 21；CPU-only，
  没有加载模型、初始化 Metal、发送推理请求或修改生产服务。生产进程保持运行，
  CPU 不独占；结果是组件计时，不是 HTTP TTFT 或模型 prefill/decode 测量。
- 基线是 Git `f82959151f1f29026eb991e096dc4741339f72bd` 的 `src/myds_server.c`，
  SHA256 `c49d25b48c72c69a399a174ff4644b3c36a7d09f2308041178502e3d7dc74ed7`；
  最终候选 SHA256 `b675ff077181319728319e78406a03ff1d0a2f7e5e24063edfe9929c62bce96b`。
- 每臂以相同 clang `-O3 -std=c11 -mmacosx-version-min=26.0` 编译单独程序，
  调用实际 `parse_chat_prompt_ex`（含 JSON 解析和渲染），文件读取和进程启动不计时。
  每轮预热 8 次，固定两臂相同重复次数，采 9 对 AB/BA 交错样本。
- 9 个合成工作负载为工具结果、工具 schema 与普通 user 文本各三档；另将仓库自身
  `myds_server.c` 和 `myds_v41.c` 作为工具结果，前者固定为首版快照（随本目录保存），
  后者直接来自本仓库。附加原文和完整 JSON 输入的字节数及 SHA 均记录在 JSONL。
  所有输入均为合成或仓库源码，不包含用户请求。
- `tool_524288` 中的数字是构造时的**字符数**，实际 HTTP JSON body 为 647,104 字节；
  `extra_myds_server.c` 和 `extra_myds_v41.c` 的 body 分别为 241,622 和 484,116 字节。
  都低于当前配置缺省的 8 MiB body 限额；本实验没有使用 HTTP 接口。
- `raw/v4-ab-macstudio.jsonl` 保存源码、编译命令、全部预热与逐对计时；
  `raw/v4-aa-macstudio.jsonl` 两臂均为相同基线，用于观察构建布局与环境偏差。
  数据可由 `elapsed_ns / repetitions` 重算，见下方代码。

## 最终结果

| 输入 | JSON body 字节 | 基线中位 ms | 候选中位 ms | 组件耗时降低 |
|---|---:|---:|---:|---:|
| tool_8192 | 10,217 | 0.09281 | 0.05026 | 45.85% |
| schema_8192 | 10,324 | 0.08187 | 0.08189 | -0.03% |
| user_8192 | 10,205 | 0.04401 | 0.04403 | -0.03% |
| tool_65536 | 80,988 | 0.73272 | 0.39368 | 46.27% |
| schema_65536 | 81,095 | 0.63797 | 0.63824 | -0.04% |
| user_65536 | 80,976 | 0.34325 | 0.34439 | -0.33% |
| tool_524288 | 647,104 | 5.84087 | 3.15373 | 46.01% |
| schema_524288 | 647,211 | 5.09711 | 5.09356 | 0.07% |
| user_524288 | 647,092 | 2.77156 | 2.76050 | 0.40% |
| extra_myds_server.c | 241,622 | 2.29272 | 1.28069 | 44.14% |
| extra_myds_v41.c | 484,116 | 4.24673 | 2.18686 | 48.50% |

以上只测本次渲染阶段。大源码工具结果在该组件中节约约 1.01 和 2.06 ms；
普通 user 与 schema 场景在本轮基本持平。不能将百分比称为整体推理或 TTFT 收益。
同机 A/A 的 11 个场景最大绝对中位差约 0.25%，不把它当作可直接相减的校正值。

另用独立 stress 语料测试短请求、密集 `</tool_result>`/层叠实体，以及密集 schema
控制字符；最终工具版 `raw/v4-stress-macstudio.jsonl` 的 64KiB 密集工具结果
0.610→0.314 ms，64KiB 密集 schema 1.118→1.118 ms；短请求约 1–2 µs，读数仅
作组件观察。stress 的语料来自独立 `tools/bench_prompt_render_stress.py`，其 SHA 在
`raw/stress-script.sha256`；JSONL 的 `script_sha256` 是**共享 A/B 驱动**，并非该 wrapper。

## 差分与失败路径

最终 A/B、A/A 均包含 11 个计时输入、48 个随机混合输入与 11 个非法输入：
非法输入覆盖 UTF-8 overlong、截断、surrogate、越界码点、原始 NUL、`\\u0000`、
孤立 UTF-16 surrogate、非法转义和未闭合 JSON。每个输入比较两臂 success/failure、
reason、**整个初始化输出缓冲区**和两端 guard，覆盖容量 0、1、小容量、恰好不足、
刚好足够及完整容量；输出字节与失败行为均一致。随机种子为 20261028。

`tests/test_server_api.c` 对 4KiB 连续段、相邻/多层工具标记、schema 控制字符、
UTF-8 和每一个输出容量使用独立已知答案检查；`tests/test_prompt_render_bounds.c`
使用保护页验证输出容量耗尽后不会扫描后续输入，无依赖计时的阈值。最终版本地
server API、bounds 和针对 server/test 对象的 AddressSanitizer + UndefinedBehaviorSanitizer
均通过；链接的其他工程对象未做 sanitizer 插桩。复现命令为 `raw/v4-local-sanitizers.sh`。

测量工具还执行了负向对照：临时错误候选将 `<tool_result>` 改成 `<tool_resultX>`，
差分在正确性阶段以 exit 1 拒绝，未进入采样；见 `raw/harness-negative.jsonl` 和
`raw/harness-negative.stderr`。

## 实验取舍

旧版本和所有逐次样本均保留，不从多轮里只挑较好读数：

| 版本 | 处理 | 密集 schema 64KiB 观察 | 结论 |
|---|---|---|---|
| v1 `ed2b891d…` | tool + schema span | 1.115→1.510 ms，慢 35.4% | 淘汰；保留 AB、AA、stress 和源码 |
| v2 `b69aae68…` | 跳过空 run | 1.115→1.413 ms，慢 26.7% | 仅做 stress，未做正式 AB |
| v3 `18f4c10a…` | 转义字面量直接追加 | 1.115→1.198 ms，慢 7.45% | 仍超过 5% 筛查条件，停止 schema 路径 |
| v4 `b675ff07…` | 只优化 tool | 1.118→1.118 ms | 保留；完成完整差分、AB、AA、stress |

各版 server 源快照与 `v1/v2/v3/v4-*-macstudio.jsonl` 在 `raw/`。
v2、v3 的单次 stress 是用于拒绝坏例的筛查，并非正式收益结论。

## 复现

```sh
make bin/test_server_api bin/test_prompt_render_bounds
./bin/test_server_api
./bin/test_prompt_render_bounds
sh benchmarks/prompt_render_2026-10-28/raw/v4-local-sanitizers.sh

python3 tools/bench_prompt_render.py \
  --baseline-ref f82959151f1f29026eb991e096dc4741339f72bd \
  --extra-text benchmarks/prompt_render_2026-10-28/raw/v1-myds_server.c \
  --extra-text src/myds_v41.c --samples 9 --min-batch-ms 50 \
  --output /tmp/myds-prompt-render-ab.jsonl

python3 tools/bench_prompt_render_stress.py \
  --baseline-ref f82959151f1f29026eb991e096dc4741339f72bd \
  --samples 9 --min-batch-ms 50 --output /tmp/myds-prompt-render-stress.jsonl
```

`--output` 必须是新路径。`--baseline-source` 可用于 Git 历史不同的隔离机器；
A/A 时可给 `--baseline-source` 和 `--candidate-source` 指向同一基线源文件。
附加文件 basename 会成为 case 名，但原文字节 SHA 不受路径影响。

从原始计时重算，不读取保存的 summary：

```sh
python3 - <<'PY'
import json, statistics
from pathlib import Path
path = Path('benchmarks/prompt_render_2026-10-28/raw/v4-ab-macstudio.jsonl')
rows = [json.loads(line) for line in path.read_text().splitlines()]
for name in dict.fromkeys(r['case'] for r in rows if r['kind'] == 'sample'):
    medians = []
    for arm in ('baseline', 'candidate'):
        samples = [r['elapsed_ns'] / r['repetitions'] / 1e6 for r in rows
                   if r['kind'] == 'sample' and r['case'] == name and r['arm'] == arm]
        assert len(samples) == 9
        medians.append(statistics.median(samples))
    old, new = medians
    print(name, f'{old:.5f} -> {new:.5f} ms', f'{100*(old-new)/old:.2f}%')
PY
```
