# 2026-10-28：V4.1 采样器 CPU 筛查

**结论：这两份历史 logits 上，采样器不满足本轮 ≥2% 端到端优化的预筛条件，不实现候选。**
当前 macstudio 服务日志中的有效采样参数为 `temperature=1, top_p=1, top_k=0,
min_p=0, repetition_penalty=1, presence_penalty=0`。这种配置会对保留的候选排序，
此前账本中贪婪采样的 10 µs 读数不能直接套用。

macstudio M2 Ultra、macOS 27.2、Apple clang 21、部署目标 26.0；下表是实际 V4.1
`sample_and_verify_token` 的组件 CPU 耗时，包含有限值检查和实际 sampler 函数：

| 轮次 | 历史 logits | 温度 | 候选数 | 中位数 | P10–P90 | 样本 |
|---|---|---:|---:|---:|---:|---:|
| 第一次 | business96 | 0.0 | — | 33 µs | 32–45 µs | 201 |
| 第一次 | business96 | 0.4 | 27 | 312 µs | 253–476 µs | 201 |
| 第一次 | business96 | 1.0 | 1,661 | 337 µs | 336–340 µs | 201 |
| 第一次 | code38404 | 0.0 | — | 8 µs | 7–8 µs | 201 |
| 第一次 | code38404 | 0.4 | 14 | 253 µs | 252–253 µs | 201 |
| 第一次 | code38404 | 1.0 | 1,927 | 356 µs | 356–362 µs | 201 |
| 第二次 | business96 | 0.0 | — | 33 µs | 28–38 µs | 201 |
| 第二次 | business96 | 0.4 | 27 | 313 µs | 253–479 µs | 201 |
| 第二次 | business96 | 1.0 | 1,661 | 341 µs | 341–348 µs | 201 |
| 第二次 | code38404 | 0.0 | — | 8 µs | 7–8 µs | 201 |
| 第二次 | code38404 | 0.4 | 14 | 253 µs | 252–257 µs | 201 |
| 第二次 | code38404 | 1.0 | 1,927 | 355 µs | 355–361 µs | 201 |

第一次试跑后发现，**非计时**候选计数使用除以温度，而真实 sampler 先计算倒数、
再执行乘法；浮点边界附近两者可能分类不同。于是将计数表达式改为与 sampler 一致，
重新编译并按原流程完整执行第二次；采样入口和计时代码未变。两次结果都如上列出，
温度 1 的四个中位数均低于约 0.472 ms（23.6 ms 的 2%），不依赖选择较好的一轮。

每个输入和温度预热 32 次，记录 201 次，每次用 `CLOCK_MONOTONIC` 测实际入口的
一次调用；缓冲区预分配，使用非空近期 token 环但惩罚关闭、bitmask 为空，RNG 从
42 开始，逐次自然推进。计时不含加载文件、构造 ctx 或记录结果。计时分辨率在本次
日志中约为 1 µs。进程保持运行，CPU 不是独占；其间没有加载模型、初始化 Metal、
使用 GPU、发送推理请求或更改生产。一次实验结束后的只读 `/health` 检查是唯一
服务请求，结果为 ok；当时 PID 13659、CPU 约 0.1%，`raw/service.txt` 保存进程读数。

以账本历史 23.6 ms/token 为**粗略参考**，即使将四个 337–356 µs 中位数全部消除也仅相当于
约 1.43%–1.51%，而实际上不可能消除全部采样成本。该分母来自历史不同请求口径，
所以这不是本轮端到端收益或经过独立验证的严格上界；它只是继续深入实现前的成本筛查。
business96 的温度 0.4 样本分布较宽，不应解释为稳定收益或回退。

表中两份都是**历史确定性生成末尾的 logits**。只检查两个 129,280 维状态，将它们
用于模拟非贪婪的一次采样；它们不代表随机生成过程中的全部 token、全部请求、
结构化输出 bitmask、top-p 截断、惩罚或其他机器。当前结论是这两个输入上值得
停止进一步实现，不是采样器在所有负载下都不可能成为瓶颈。

温度 0 的 V4.1 默认路径会调用独立 NEON greedy，而非通用采样器；它的样本也保存
在 JSONL，但 business96 与 code38404 的观测差异明显，不能从两份向量外推固定成本。
之前直接调用通用 sampler 所见的贪婪约 490 µs 不适用于 V4.1 默认路径。

## 原始证据与身份

- 正式入口源：`src/myds_v41.c` SHA256 `b5951c3af234d142fad444c112d6c9e823dbd63d2b10b536976c9dc28d335d99`；
  通用 sampler 源：`src/myds_engine.c` SHA256 `a8a5169ec08581c31ba34e3631c07a406fb641e560420655c8578075ed3cf600`。
- 实验入口：`tools/bench_sampler_screen.c` SHA256
  `ffe39568f86fc024dbaeefa9d1fe1c24dba650375e973bfa7ab312c9eedc7bf2`；macstudio
  实验二进制 SHA256 `96664656449595c26de3d9ae1d13aca8b0ee4bba493af0365367355dae5ecf1c`。
- 输入：`business96.f32` SHA256
  `0f81a7a6fc9258de3fba15c348568ed9bd88ed43eb0209c844e43ecee7319de9`；
  `code38404.f32` SHA256 `16b107ffb472adc0cddd0a83483dac2dd63b03c516d5bc82a82bcf1225090253`。
  每份长度均为 517,120 字节，内容**不入库**。
- 源自历史归档 `fullmodel-evidence-v1.tar.gz`，SHA256
  `82cc45e62a87ca1fafcf35d484ade137b58b956b3aae801ad0e35e0289c38f73`；
  该归档的生产来源及冻结证据见既有 `mydsdoc` 的 register-only 实验记录。
- 最终 1,206 条逐次耗时样本（2 输入 × 3 温度 × 201）见 `raw/*-final.jsonl`；
  编译输出、源码/二进制/输入哈希、include 文件哈希及机器信息见同一目录。
  源码构建于独立临时目录，未改生产二进制。
- `raw/*-raw.jsonl` 保留第一次正式入口试跑，源文件
  `raw/preliminary-bench-sampler-screen.c` 的哈希与第一次记录对应。最终版本只把
  非计时的候选计数表达式从除以温度改为与 sampler 相同的乘倒数；两轮都保留。
  最终结果使用最终版本，未按结果选择最佳轮次。

## 复现（仅 CPU）

需要本仓库及本机已有历史私有归档，命令提取两份 logits 到全新临时目录；
它不会把 logits、请求文本或采样 token 放入 Git 仓库。

```sh
bench_archive="$HOME/Benchmarks/myds/2026-10-26-register-only-and-narrow/fullmodel-evidence-v1.tar.gz"
bench_work="$(mktemp -d /tmp/myds-sampler-screen.XXXXXX)"
shasum -a 256 "$bench_archive"
# 应为 82cc45e62a87ca1fafcf35d484ade137b58b956b3aae801ad0e35e0289c38f73
tar -xOf "$bench_archive" \
  origin/M2/private/tmp/myds-register-only-P-freeze-20261026-v1/P-window/business96/A1/logits/run-0.f32 \
  > "$bench_work/business96.f32"
tar -xOf "$bench_archive" \
  origin/M2/private/tmp/myds-register-only-P-freeze-20261026-v1/P-window/code38404/A1/logits/run-0.f32 \
  > "$bench_work/code38404.f32"
shasum -a 256 "$bench_work"/*.f32
clang -O3 -Wall -Wextra -std=c11 -mmacosx-version-min=26.0 -Iinclude -Ithird_party \
  tools/bench_sampler_screen.c src/myds_engine.c src/myds_log.c -Wl,-dead_strip \
  -framework Foundation -framework Metal -lpthread -lm -o "$bench_work/probe"
"$bench_work/probe" "$bench_work/business96.f32" > "$bench_work/business96.jsonl"
"$bench_work/probe" "$bench_work/code38404.f32" > "$bench_work/code38404.jsonl"
```

重算已保存的最终表格（每组应为 201 行，不从 summary 抄数）：

```sh
python3 - <<'PY'
import json, statistics
from pathlib import Path
for p in sorted(Path('benchmarks/sampler_screen_2026-10-28/raw').glob('*-final.jsonl')):
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    for temperature in (0.4, 1.0):
        xs = sorted(r['elapsed_ns'] for r in rows if r['kind'] == 'sample'
                    and r['temperature'] == temperature)
        assert len(xs) == 201
        print(p.name, temperature, 'median/p10/p90 us:',
              statistics.median(xs) / 1000, xs[20] / 1000, xs[180] / 1000)
PY
```
