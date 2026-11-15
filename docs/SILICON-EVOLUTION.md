# Silicon Self-Evolution Protocol (`myds`)

`myds` 的全部代码由 LLM 编写，并在真实 Apple Silicon 硬件上通过物理闭环迭代演进。为了让本地运行在目标 Mac Studio 上的 LLM Agent 在**无人类干预**条件下直接对硬件做假设、改核、验证与演进，仓库内置四件硅基自进化基础设施。

## 目标硬件范围（仅限可运行全量 `337 GiB` V4.1-Flash 的 5 款 Mac Studio）

本协议仅面向具备足够统一内存运行 `DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf`（`362.14 GB` 文件，约 `148.4 GB` wired + `188.8 GB` 稀疏 `pread` Engram 表）的五款机型：

| 机型 | 统一内存 | GPU 家族 | Threadgroup / 寄存器分配架构 | 标称峰值带宽 | 当前调优深度 |
|---|---:|---|---|---:|---|
| **M2 Ultra** | **192 GB** | `MTLGPUFamilyApple8` | **静态硬切分**（`floor(32768 / static_smem)` 硬卡每核并发 TG 数） | `800 GB/s` | 主力在产精调（`§1–§176`） |
| **M3 Ultra** | **256 GB** | `MTLGPUFamilyApple9` | **Dynamic Caching**（smem、寄存器与 L1 硬件动态共享） | `819 GB/s` | 待持机 Agent 本地实测演进 |
| **M3 Ultra** | **512 GB** | `MTLGPUFamilyApple9` | **Dynamic Caching**（全量 `362 GB` 含 Engram 可纯内存常驻） | `819 GB/s` | 待持机 Agent 本地实测演进 |
| **M5 Ultra** | **256 GB** | `MTLGPUFamilyApple10+` | **Dynamic Caching** | `1024 GB/s` | 待持机 Agent 本地实测演进 |
| **M5 Ultra** | **512 GB** | `MTLGPUFamilyApple10+` | **Dynamic Caching**（全量常驻 + TB/s 级带宽） | `1024 GB/s` | 待持机 Agent 本地实测演进 |

> **铁律**：仓库绝不为未持有的机型编造未经实测的启发式分支。持机用户只需在本地运行下述闭环工具，由本地或外部 LLM Agent 基于本机真实 Metal PSO 指标与 `ABABAB` 物理判决提交 PR。

---

## 四件硅基自进化基础设施

### 1. 硅基显微镜（Silicon Microscope）
- **文件**：`tools/evo_pso_probe.m`（编译为 `bin/evo_pso_probe`）+ `tools/evo_silicon_probe.py`
- **职责**：
  1. 直接调用本机 Metal 驱动 `MTLComputePipelineState` 编译全部 `533` 个内核，导出每个内核在**当前芯片**上的 `staticThreadgroupMemoryLength`、`max_tgs_per_core_by_smem`、`maxTotalThreadsPerThreadgroup` 与 `Apple8` 静态分配 vs `Apple9+` Dynamic Caching 架构属性；支持 `--diff-metallib` 在编译期毫秒级抓出占用率退化。
  2. 解析 `/trace` 请求记录并严格执行 [`docs/MEASUREMENT-DISCIPLINE.md`](MEASUREMENT-DISCIPLINE.md) `§6.1`（按 `cbs / layers` 还原真实采样 token 数，严禁除 `out_tokens`）与 `§5.1`（标注 `gpu_gap` 不可在首个 CB 前异步重叠回收），给出相对本机带宽屋顶的真实利用率 `%`。
  3. 关联 `bench_kprof` 输出与 PSO 占用率属性，并强制附带 `§6.2` 串行失真封条。

```bash
make evo-probe
python3 tools/evo_silicon_probe.py --metallib bin/myds.metallib --top 20
```

### 2. 结构化演进记忆库（Evolution Memory）
- **文件**：`tools/evo_memory.py` + `docs/ledger-index.jsonl`
- **职责**：将两万余行的[`性能账本`](2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md)（`§0.3` 在产登记表、`§0.4`/`§0.6` 已证否红线、`§1–§176` 全部章节）及自治实验记录确定性编译为 `docs/ledger-index.jsonl`。Agent 在提出任何改核假设前必须先检索死胡同清单，避免重踩已证否方向（如 MoE `dot(float4,float4)` 非 0-ULP、Decode 浅层 `Top-2` EOS 死循环等）。

```bash
make evo-memory-check
python3 tools/evo_memory.py --query --subsystem indexer --limit 10
python3 tools/evo_memory.py --query --subsystem moe --text float4
```

### 3. 原子化物理裁判机（Atomic Physical Judge）
- **文件**：`tools/evo_judge.py`
- **职责**：把 `docs/MEASUREMENT-DISCIPLINE.md`（`§1–§12`）的全部防伪规则固化为不可绕过的机器退出码（`0 = ADOPT/PASS`、`1 = REJECT`、`3 = VOID_MEASUREMENT`）：
  - **`--mode offline`**（毫秒至秒级静态+单测门禁，无需停在线服务）：校验 `make check-knobs`、`evo_memory.py --check`、候选构建、Metal PSO 占用率 diff、全套单测及 `.ips` 崩溃计数增量为 `0`（`§9`）。
  - **`--mode full`**（全量权重独占窗口 `ABABAB` 交错裁判）：
    - 强制校验两臂二进制/metallib SHA256 或独立配置文件确实存在差异（`§3`）；
    - 为每次启动在独立临时目录分段暂存 `./bin/myds` 与 `./bin/myds.metallib`（防止 `src/myds_metal.m` 的 `./bin/myds.metallib` 搜索顺序遮蔽候选 shader），生成隔离 `arm.conf`（保留 `MYDS_PREFIX_CACHE=1`，清空隔离 `MYDS_KV_DSK` 与 `MYDS_RESP_STATE`，并经 `--print-config` 校验无拼写废键，`§1`/`§2`/`§12`）；
    - 在生产 `auto_barrier=0` 整图下跨 `A1→B1→A2→B2→A3→B3` 六次冷启动校验多条确定性提示词输出 SHA256 **100% 逐位一致（0-ULP）**（`§4`）；
    - 通过 `/trace/api/requests` 权威核验 `cached_tokens == 0` 且日志无 OOM/回退（`§2.1`/`§A.5.4`）；
    - 要求全部配对轮次同向正收益、中位增益 `>= --min-gain-pct` 且严格大于单臂组内极差（`§10`）。

```bash
make evo-judge
python3 tools/evo_judge.py --mode offline --unit-cmd ""
```

### 4. 分时自举守护器（Time-Sliced Self-Hosting Supervisor）
- **文件**：`tools/evo_supervisor.py`
- **职责**：解决 `192 GB / 256 GB` 单机上「正在给 Agent 提供推理的 Golden `myds` 与全模型 A/B 评测无法同时常驻」的自指死锁：
  1. **零停机预检**：在不动在线 `8187` 服务的前提下校验候选 Mach-O/metallib、通过 `KERN_PROCARGS2` 提取在线模型与配置路径，并执行候选 `--print-config` 与 `--mode offline` 门禁。
  2. **连接排空**：等待在线 `ESTABLISHED` TCP 客户端（含自举 Agent 当前回合连接）自然归零。
  3. **唯一权威独占窗口**：严格复用 `tools/run_register_only_service_window.py`（`Window`）与 `tools/gpu_window.py` 停产备份、等待 XNU 释放原进程 `resident_gib` 物理页后再启动 `evo_judge.py --mode full`，评测结束后精确恢复原 Golden 进程并核验 `gpu_window.diff()`。
  4. **可选原子晋升（`--promote-on-adopt`）**：仅当裁判给出 `ADOPT` 时，以临时文件 + `os.replace()` 原子替换目录项 inode（严守 `§7` 绝不原地 `cp` 覆写映射页以防 `SIGBUS`），验证新进程 `/health` 正常否则自动回滚，并落盘 `surgery-receipt.json` 供 Agent 下一轮苏醒读取。

```bash
python3 tools/evo_supervisor.py status --port 8187
make evo-check
```

---

## Agent 自治演进标准闭环

1. **查记忆**：运行 `python3 tools/evo_memory.py --query --subsystem <target>` 排除历史已证否方向。
2. **看硬件**：运行 `make evo-probe` 查看当前芯片（`Apple8` 静态 smem 或 `Apple9+` Dynamic Caching）下各热点内核的 smem、每核 TG 上限与寄存器分档。
3. **编候选**：将修改后的候选产物输出到隔离目录（例如 `/tmp/cand/myds` 与 `/tmp/cand/myds.metallib`），严禁直接覆盖在线运行中的二进制。
4. **过离线门**：运行 `python3 tools/evo_judge.py --mode offline --candidate-bin /tmp/cand/myds --candidate-metallib /tmp/cand/myds.metallib` 确认 PSO 占用率无意外退化且单测全绿。
5. **进物理窗**：调用 `python3 tools/evo_supervisor.py evaluate ... --record-memory` 执行分时独占 `ABABAB` 实测；只有 `ADOPT`（`exit 0`）的改动才允许提交代码与账本证据，被否决的改动同样记入演进记忆以阻止后续世代重复试错。
