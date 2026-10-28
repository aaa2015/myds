# 2026-10-28：请求分词的 CPU 优化

本轮先阅读优化账本 §A/§B、§0、§100–§106，再检查当前实现。已有 GPU 实验的
失败结论按原范围保留。本次改动只缩短分词阶段：ASCII 分类改为等价查表，BPE 的
symbols/候选 heap 在一次 encode 调用内复用，结束或失败时释放。没有修改分词规则、
merge 排名、模型算术、权重或 GPU 调度。

## 测量范围与身份

- 机器：macstudio，M2 Ultra，192 GiB；macOS 27.2；Apple clang 21.0.0。
- 基线：`15f9174c7ba451491070fb17ea49aa0b36d12753` 的 `src/myds_tokenizer.c`。
  SHA256：`808c432023f5fa2534a4464fbbc7816bb54d86ac44921ba9b7ec2d45e5fcf144`。
- 候选源 SHA256：`17bc8fbe8c4d91f84686db1b67f82c0a8a11dc5d05fe00ad43f4eed9fcf5db7e`。
- 两臂使用同一编译器、`-O3 -std=c11 -mmacosx-version-min=26.0`、同一 Unicode 表，
  编译为同一进程内两套独立函数和 tokenizer 实例。
- 实际模型是 `DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf`，词表 129,280。
  正式计时工具仅提取约 5.76 MB 的 GGUF header/metadata；不读 tensor payload、不初始化 GPU。
  元数据 SHA256：`54ae6209c1bc4babbccd2ffc1d627f7ab1058501b2bda951293f994542b77b84`。
- 7 条公开生成的合成输入，不含用户请求；输入字节及 SHA 见 `corpus-manifest.json`。
  计时只包含 `encode_bytes(add_bos=false)`，输出缓冲区预分配，排除加载和文件读取。
- 每个输入预热后固定两臂相同的重复次数，正式 15 对样本，AB/BA 交错；
  每个候选批次目标约 100 ms。全部原始 batch/repetition/timing 在 JSONL 中。
- 生产服务保持运行且空闲，属于 CPU 组件实测，不属于账本的 GPU 独占整机测试。
  前后 PID 均为 13659，制品 SHA 不变，结束时 `/health` 为 ok；见 `raw/service-after.json`。

## 实测结果

| 输入 | token数 | 基线中位 ms | 优化中位 ms | 分词耗时降低 |
|---|---:|---:|---:|---:|
| 短英文 | 8 | 0.001313 | 0.000764 | 41.80% |
| 短中英混合 | 11 | 0.003219 | 0.002952 | 8.30% |
| 英文段落（约256 KiB） | 48,525 | 9.742250 | 4.979400 | 48.89% |
| Python代码（约256 KiB） | 75,411 | 10.103864 | 4.678409 | 53.70% |
| 中文段落（约128 KiB） | 21,670 | 6.523438 | 6.305813 | 3.34% |
| 多语言/代码/特殊token（约512 KiB） | 147,807 | 22.487571 | 14.977857 | 33.39% |
| 超长单词片交错 | 6,148 | 3.411290 | 3.325290 | 2.52% |

耗时降低按 `100 × (baseline_median - candidate_median) / baseline_median` 计算。
这是分词耗时，不能当作 GPU prefill/decode 或整次请求的加速比例。主三档英文、代码、
混合输入分别仅节约约 4.76、5.43、7.51 ms；短输入节约不足 1 µs。长上下文整体推理
仍受原有 GPU 工作主导，本轮没有测量新的端到端吞吐。

额外 A/A 对照将两臂都编译为相同基线，测量英文、代码、中文和长词片，每臂 9 样本。
两臂中位差分别为 0.024%、0.339%、0.976%、0.559%；原始数据见 `raw/macstudio-aa.jsonl`。
因此 33–54% 的主场景收益明显大于本轮计时偏差。中文 3.34%、长词片 2.52% 的改善
较小，仅作为这组输入和这次会话的观测，不外推到所有中文/超长词片。

## 正确性与失败路径

- 真实词表：256 个有种子的二进制/Unicode 输入、1,283 个特殊 token 组合、空字符串与
  NULL 空输入、7 个计时输入，合计 **1,548 个输入**。每个输入比较所有 token IDs，
  同时覆盖 BOS 开/关、count-only、截断输出与未写入区哨兵；全部通过。
- 独立 Hugging Face 参考：真实 `tokenizer.json` 与实际 GGUF metadata，**118/118**
  token ID 及解码字节用例通过，详见 `raw/macstudio-hf-reference.json`。
- 本地及 macstudio tokenizer 8 项已知答案测试通过；新增长短词片交错、特殊 token
  两侧复用和各输出容量检查。
- 生命周期测试穷举 Unicode 分类与旧表对照；逐一注入 12 个分配失败点，检查 NOMEM、
  零存活分配、后续正常编码；检查 unknown-byte 错误清理、4 线程共享 tokenizer。
  本地 AddressSanitizer + UndefinedBehaviorSanitizer 通过。
- 本地 `bin/myds` 构建、server API 和 engine gate 测试通过。没有运行全模型 GPU 数值门；
  数值计算路径本轮未改，所验证的是模型输入 token 序列及分词 API 契约。
- 测量工具用正常 synthetic GGUF 自检通过；另将输出 ID 故意 xor 1，工具在正确性阶段
  以 worker exit 3 / Python exit 1 拒绝，未生成正式计时行。见 `raw/harness-*.jsonl`。
  JSONL 的 `summary.pass` 表示正确性与采集完成，不自动表示有性能收益。

## 复现

不需要 192 GiB 内存即可运行该 CPU 工具，但需要可读取的模型 GGUF 元数据。
从项目根目录运行，输出文件必须是新路径（工具拒绝覆盖既有证据）：

```bash
make test-tokenizer
python3 tools/make_tokenizer_perf_corpus.py --out scratch/tokenizer-corpus
python3 tools/make_tokenizer_perf_corpus.py --out scratch/tokenizer-corpus --check
python3 tools/bench_tokenizer.py \
  --model /path/to/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf \
  --baseline-ref 15f9174c7ba451491070fb17ea49aa0b36d12753 \
  --prompt scratch/tokenizer-corpus/short_ascii.txt \
  --prompt scratch/tokenizer-corpus/short_mixed.txt \
  --prompt scratch/tokenizer-corpus/english_256k.txt \
  --prompt scratch/tokenizer-corpus/code_256k.txt \
  --prompt scratch/tokenizer-corpus/chinese_128k.txt \
  --prompt scratch/tokenizer-corpus/mixed_512k.txt \
  --prompt scratch/tokenizer-corpus/long_piece.txt \
  --samples 15 --random-cases 256 --min-batch-ms 100 \
  --output scratch/tokenizer-ab.jsonl
```

macstudio 的工作副本 Git 历史不同，本轮将上述 revision 的旧源单独导出，用
`--baseline-source baseline.c` 测量；原始记录中的源码哈希与上述 revision 一致。
工具支持 `--candidate-source`，A/A 时两项同时指向同一基线 C 文件即可。

独立参考检查可复用仓库已有工具（需要 Python `tokenizers` 库）：

```bash
python3 tests/test_tokenizer_reference.py /path/to/tokenizer.json \
  --probe ./bin/test_tokenizer --gguf /path/to/model.gguf \
  --output scratch/tokenizer-hf-reference.json
```

`raw/macstudio-ab.jsonl` 是正式性能结论的来源；`raw/pilot-*.txt` 只保留早期试跑，
其取最小值的旧工具不参与正式结论。本轮未部署新生产可执行文件。

从原始 batch 耗时重算表格（不依赖保存的 summary）：

```bash
python3 - <<'PY'
import json, statistics
from pathlib import Path
p = Path('benchmarks/tokenizer_2026-10-28/raw/macstudio-ab.jsonl')
rows = [json.loads(line) for line in p.read_text().splitlines()]
for item in (r for r in rows if r['kind'] == 'input'):
    index = item['prompt_index']
    medians = []
    for arm in ('baseline', 'candidate'):
        samples = [r['elapsed_ns'] / r['repetitions'] / 1e6 for r in rows
                   if r['kind'] == 'sample' and r['prompt_index'] == index
                   and r['arm'] == arm]
        assert len(samples) == 15
        medians.append(statistics.median(samples))
    a, b = medians
    print(Path(item['path']).name, f'{a:.6f} ms -> {b:.6f} ms',
          f'{100 * (a-b) / a:.2f}% lower latency')
PY
```
