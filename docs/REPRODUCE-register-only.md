# 复现 register-only indexer 的性能数据

本文档的目标是让**另一个模型或另一个工程师**，在只有这份仓库和一个符合条件的机器的前提下，
能独立得到与账本 §105 一致的数字，并且**能判断自己失败了**。

不以「读完觉得有道理」为通过标准。通过标准只有一条：`verify_register_only_bench.py` 返回 `PASS`。

---

## 0. 你不需要读完全部账本

需要知道的只有三条：

1. 被测对象是 `MYDS_INDEXER_REGISTER_ONLY` 这个开关（默认开），它对 indexer 的 **prefill**
   打分算子生效，对 decode 无影响。
2. 结论是 32K prefill 上 **+4.35%**，且有逐字节相同的输出作为等价证据。
3. 结论只在**特定配置 + 特定机器**上成立，换配置会变（§105.7 明列了未定位的 3.9–5.6% 差异）。

---

## 1. 可分三个层次复现，按代价从低到高

| 层次 | 需要什么 | 能证明什么 | 命令 |
|---|---|---|---|
| **L1 算术** | 只要 Python 3 | 账本里每个数字与原始 JSONL 一致 | `make audit-ledger` |
| **L2 判定逻辑** | 只要 Python 3 | 验证器能接受真数据、拒绝假数据 | `make verify-register-only` |
| **L3 实测** | M2/M3 Ultra 192 GB + 362 GB 权重 + 本仓库 | 在该机器上能否重现 593→619 t/s | `make bench-register-only` 然后验证 |

**L1 和 L2 在任何笔记本上都能跑，几十毫秒。** 如果你只想检查「这个项目是不是在编数字」，
跑完 L1+L2 就够了，不需要任何 GPU。这是刻意设计的：让最低成本的检查也有意义。

---

## 2. L1：算术审计（无 GPU）

```bash
make audit-ledger
# checks: 166  passed: 166  failed: 0
```

`tools/audit_ledger_105.py` 从 `benchmarks/register_only_2026-10-27/raw/*.jsonl`
**重算**全部统计量（中位数、极值、离散度、逐周期值、配对估计、标准差、SEM、z、
两臂 sha、计数器差分、GPU 六种残值组合、线性拟合参数与残差），
再逐项与账本 §105 的**文字**比对。任何一项不符即 `exit 1`。

它还会把账本 §106 的六个容差与
`expected.json` 逐项对齐（改了一边忘了另一边就报错），并检查账本里**不许再出现**已撤回的数字（`591.7` / `617.7` / `4.39` / `1,620`），
仅允许它们出现在明确的撤回语句里。

**为什么这层重要**：账本数字是手工从 JSON 抄进 Markdown 的。手抄就是错误入口。
这一层把「抄错」变成会红的检查。

---

## 3. L2：判定逻辑自检（无 GPU）

```bash
make verify-register-only
# 36 checks, 36 passed, 0 failed
# VERDICT: PASS — the measured run is consistent with the recorded results
```

用归档的历史数据喂给验证器，证明它**能接受**真数据。

再验证它**会拒绝**假数据（负向对照）：

```bash
python3 - <<'EOF'
import json, pathlib
src = pathlib.Path('benchmarks/register_only_2026-10-27/raw/rig.jsonl')
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
for r in rows:
    if r.get('kind')=='run' and r.get('arm')=='ON' and r.get('sample')=='nonce':
        r['tps'] *= 0.97          # 模拟一台慢 3% 的机器
pathlib.Path('/tmp/rig_corrupt.jsonl').write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
EOF
python3 tools/verify_register_only_bench.py --ab /tmp/rig_corrupt.jsonl
# 应输出 2 项 FAIL 并 exit 1
```

一个从没见过失败用例的验证器不算验证器。这条负向对照就是它的自证。

---

## 4. L3：实测（需要真机器）

> **L3 必须在持有模型权重的那台机器上本地执行。** 脚本用 `pgrep -x myds`
> 找进程、用 `127.0.0.1:8187` 发请求、读本机日志——它不是在远端操纵另一台机器。
> 从笔记本 ssh 进去跑是错的：那样 `pgrep` 找不到进程，`/health` 也连不上。

### 4.0 先跑预检（不动任何东西）

```bash
python3 tools/bench_register_only_ab.py --out /tmp/x.jsonl --boots 1 --timed-per-tboot 1
```

（注意上面故意写错一个参数名以便快速失败；也可以用真实参数跑，预检先于任何重启。）

预检会硬拒绝以下情况，**且不碰任何服务**：

| 检查 | 不通过时 |
|---|---|
| `bin/myds` / 配置 / 模型 / vision 权重 / 语料存在 | 列出缺失路径 |
| **总内存 < 180 GiB** | 拒绝启动——本基准会钉住约 150 GB，小机器会因 wired 耗尽 **内核 panic**，不是 OOM kill |
| `sudo -n true` 通过 | 拒绝——没有它就读不到计数器，两臂是否真有区别就无法证明 |

在 16 GiB 的笔记本上实测预检输出（这是正确的行为，不是故障）：

```
PREFLIGHT FAILED — nothing was started, no service was touched:
  - model not found: …DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
  - this machine has 16 GiB RAM; the benchmark pins ~150 GB and a 64/96/128 GiB host
    can panic from wired exhaustion. Refusing to start.
```

### 4.0.1 工具与语料必须先到那台机器上

`deploy.sh` 的 `SYNC_DIRS` 包含 `tools` 与 `docs`，**不包含 `benchmarks`**。所以要么在目标机器上
从仓库直接 clone，要么手动补一次：

```bash
rsync -a tools/ <host>:~/Projects/myds/tools/
rsync -a benchmarks/ <host>:~/Projects/myds/benchmarks/
```

然后在那台机器上 `python3 tools/make_register_only_corpus.py --check` 确认语料完整。

### 4.1 硬件与软件前置条件

| 项 | 要求 | 为什么 |
|---|---|---|
| 内存 | **≥192 GiB 统一内存** | 模型常驻 148–154 GB，工作区另有需求；低于此会在 prefill 时 OOM 或 panic |
| 权重 | `DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf`（362 GB） | 数字是这份权重量出来的，换量化结果不可比 |
| 独占 | 无其它 myds / bench 进程 | 脚本启动前会检查，发现 >1 个即拒绝运行 |
| Xcode | 命令行工具齐全（`xcrun metal` 可用） | 需要 `make metallib` |
| sudo | **免密**（`sudo -n true` 通过） | 读原子计数器要 `lldb` attach，采 GPU 要 `powermetrics` |
| 配置 | `MYDS_CHUNK_KEY_MAX=1048576`（本次口径） | 该值直接改变 chunk 工作区大小，是吞吐差异的主要嫌疑 |

如果 `sudo` 没有免密，脚本会在计数器那一步返回 `None`，验证器会报
`counter deltas present` 失败——**这是设计如此**：没有计数器就无法证明两臂真的分开跑了，
此时结论不成立，而不是"跳过也能过"。

### 4.2 构建

```bash
make -j8            # 产出 bin/myds 与 bin/myds.metallib
shasum -a 256 bin/myds bin/myds.metallib
```

本次数字对应的制品：

```
bin/myds           a97976e0a2e2aced06349c203ee7d4d4244688178eee907281a08e3aded2f1dd
bin/myds.metallib  fe55793346e15119fce4e0f40d4bf6d3023fc3694ffc231e66b8a89e3ae0f74d
```

**哈希不同不一定是错**（编译器版本差异会导致二进制不同），但要先确认
`strings bin/myds.metallib | grep kernel_myds_indexer_scores_register_only_bf16` 能找到内核。
找不到就别往下跑。

### 4.3 数据集合

```bash
make corpus-check
# corpus OK: 30 files match manifest
```

语料在 `benchmarks/register_only_2026-10-27/prompts/`，30 个文件（6 个 prefill 尺寸
+ 1 个 A/B 尺寸 + 3 个 decode 场景，各 3 个变体），共 1.9 MB，随仓库发布。

- 变体存在的原因：服务端有 prefix cache，重复的 prompt 会命中缓存（`cached != 0`），
  那样**根本没有发生 prefill**，测出来的数无意义。三个变体绕开缓存，不需要改服务端配置。
- `manifest.json` 记录每个文件的字符数、字节数、sha256。
  `tools/make_register_only_corpus.py` 是生成器，`--check` 模式验证磁盘上的文件与生成结果逐字节一致。
  **语料可审计，而不是一坨不可追溯的二进制。**

**已知偏差**：语料的标签（`[r0] `）与历史跑批用的 nonce（`[编号 p52k-r0-<epoch>] `）不同，
所以 token 数会差几个。`expected.json` 因此在 token 数上给了 ±2% 的容差，而不是要求精确相等。
吞吐的 ±2% 容差与这个无关，那是热漂移的量级。

### 4.4 跑 A/B

```bash
mkdir -p benchmarks/register_only_2026-10-27/runs
python3 tools/bench_register_only_ab.py \
    --out benchmarks/register_only_2026-10-27/runs/ab.jsonl
```

预计 **35–45 分钟**（3 个开机周期 × 2 臂 × (1 预热 + 4 次 32K 请求)），
期间服务会重启 6 次、每次约 10 秒不可用。**别在有人的时候跑。**

脚本会做的事（每一步都有对应的失败判据）：

1. 停掉所有 myds，等内存回到 ≥110 GB（不等就拒绝，避免双进程把内存翻倍）
2. 以 `MYDS_INDEXER_REGISTER_ONLY=0/1` 启动，等 `/health`
3. 读开机前的原子计数器
4. 发 1 次预热（丢弃）
5. 发 1 次**固定 prompt**（缓存必空）→ 用于两臂逐字节比对
6. 发 3 次语料变体，每次并发采 GPU（58 点）
7. 读计数器，算差分
8. 全部结束后**恢复部署默认**（无 env 覆盖 = 默认开）

### 4.5 跑扫频

```bash
python3 tools/bench_register_only_sweep.py \
    --out benchmarks/register_only_2026-10-27/runs/sweep.jsonl
```

预计 **20–25 分钟**。前提是服务已经在跑（本脚本不重启服务）。

### 4.6 判定

```bash
python3 tools/verify_register_only_bench.py \
    --ab    benchmarks/register_only_2026-10-27/runs/ab.jsonl \
    --sweep benchmarks/register_only_2026-10-27/runs/sweep.jsonl
```

它会逐条打印 `[PASS]`/`[FAIL]`，并在最后给一句 `VERDICT`。

---

## 5. 容差是怎么定的（以及为什么不该更严）

| 量 | 容差 | 依据 |
|---|---|---|
| prefill / decode 吞吐 | **±2%** | 单次会话内实测离散 0.135%（关）/ 0.065%（开）；跨会话要覆盖热漂移与页缓存状态 |
| A/B 收益 | **±1.5 个百分点** | 是两个 ±2% 量之差的组合，误差会放大 |
| prompt token 数 | **±2%** | 语料标签与历史 nonce 不同，见 4.3 |
| 两臂输出 | **必须完全相等** | 这是等价性主张，不是性能量，没有容差 |
| GPU 驻留 | **≥95%** | 低于此时比较无效（一臂在等），不是"更快" |
| GPU 功耗 | **85–125 W** | 本次两臂 104.2 / 106.0 W；越界说明负载形态变了 |

**不要因为"想让它通过"而放宽容差。** 若实测超出容差，正确做法是记录差异并说明原因，
而不是改 `expected.json`。放宽容差等于取消判据。

---

## 6. 这次复现**不能**证明什么

写清楚边界，比写清楚结论更重要：

1. **不能证明在其他机器上也是 +4.35%**。M4、M3 Ultra、内存更小的机器都没有数据。
   本项目的所有性能数字都限定在这台 M2 Ultra 192 GB 上。
2. **不能证明 logits 级等价**。A/B 的两臂比对是**输出文本**逐字节相同，
   强于无检查，但弱于 §104 的 129,280 维 logits 逐位比对。两者不能互相替代。
3. **不能解释与旧账本 3.9–5.6% 的差距**。`CHUNK_KEY_MAX` 等四项是**候选原因**，
   本轮没有做同口径对照实验，所以账本只记差异、不给归因。
4. **不能证明 90K 以上也有收益**。门限是 `end ≤ 32768`，长上下文场景收益≈0。
5. **不能证明生产上不会回退**。优雅回退存在，且计数器显示 `fallbacks=0`，
   但那是这台机器这段时间的观测，不是保证。

---

## 7. 如果你得到的结果不一样

按顺序排查，每一步都有对应的检查点：

| 现象 | 先查 | 常见原因 |
|---|---|---|
| 预检报 `model not found` | 是否在持有权重的机器上跑 | 在笔记本上跑了（L3 必须本机执行） |
| 预检报 `16 GiB RAM` | 同上 | 小机器会 panic，脚本故意不让跑 |
| `cached != 0` | 语料是否被重复使用 | 手动改过 prompt；两个脚本同时跑 |
| 两臂 `encoded` 都是 0 | `strings bin/myds.metallib \| grep register_only_bf16` | metallib 没重编；内核名写错 |
| 两臂 `encoded` 都是 2025 | 启动 env 是否生效 | `myfleet/myds.conf` 里写了 `MYDS_INDEXER_REGISTER_ONLY`，它**无条件覆盖**进程环境 |
| 两臂输出 sha 不同 | 是否真的从空缓存开始 | 预热 prompt 与固定 prompt 用成了同一个文件 |
| GPU 驻留只有 2% | 采样时机 | 遥测如果在请求结束后采，只能读到排空残值（0–2.31%、0–386 mW） |
| 整体慢 5% | `MYDS_CHUNK_KEY_MAX` | 与本次口径不同；这是已知会改变吞吐的配置项 |
| 计数器读不到 | `sudo -n true` | 没有免密 sudo；**此时结论不成立**，不能当"通过" |

---

## 8. 文件清单

```
benchmarks/register_only_2026-10-27/
  README.md                      这一层的入口
  prompts/                       31 个语料文件 + manifest.json（sha256 逐一记录）
  raw/                           rig.jsonl / sweep2.jsonl / dec2.jsonl —— 账本数字的原始出处
  expected.json                  期望值 + 容差 + 为什么这样定
  runs/                          自己跑的结果放这里
tools/
  make_register_only_corpus.py   语料生成器，--check 验证磁盘与生成一致
  bench_register_only_ab.py      A/B 测量（交错、9 样本/臂、并发 GPU、计数器）
  bench_register_only_sweep.py   扫频 + decode（每点 3 次）
  verify_register_only_bench.py  判定器，--historical 自检，可作负向对照
  audit_ledger_105.py            账本数字 vs 原始 JSONL，147 项
docs/
  REPRODUCE-register-only.md     本文
```

## 8.1 落地过程中修掉的两个真实缺陷

这两个都不是风格问题，而是会让结果无效的缺陷：

1. **固定样本与第一个计时样本曾共用同一个语料文件**（`p32k_ab__v0.txt`）。
   后果：固定样本把 prefix cache 填满，第一个计时样本 0.17 s 返回、`cached != 0`，
   **根本没有发生 prefill**，却被记成了一行空数据。
   修正：新增独立的 `ab_fixed.txt`，并在脚本里加「同一 boot 内 prompt 不得重复」的断言；
   同时把「无服务端行」与「`cached != 0`」从静默记录空值改成**中止该 boot**——
   无效数据不应该进入平均。
2. **没有预检就启动**。小机器上会先花七分钟再失败，或者直接因 wired 耗尽 panic。
   修正：启动前检查产物/配置/模型/语料是否存在、总内存 ≥180 GiB、`sudo -n true` 是否通过；
   任一项不满足则列出原因并退出，**不碰任何服务**。

---

## 9. 一句话

复现的难点从来不是跑一遍，而是**知道自己跑错了吗**。
所以这份仓库里最重要的两个文件不是测量脚本，是 `expected.json`（容差写在明处）
和 `verify_register_only_bench.py`（能返回 FAIL）。
