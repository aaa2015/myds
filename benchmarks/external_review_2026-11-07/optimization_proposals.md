# myds prefill / decode 优化建议（2026-11-07）

> 写法依账本 §C.13：每条都写**仓库现状（Q1）**、**与已证否路径的区别（Q2）**、**上界及依据（Q4）**、**0-ULP 状态**、**什么情况下会失败（Q8）**。
> 标注：【实测】= 账本或 trace 文档里的原始数字；【代码】= 我读源码确认的事实；【推断】= 未测，必须先测。
> **本文所有收益都没有经过我实测**——我没有运行权限，也没碰生产。

---

## 总览（按预计省下的绝对秒数排序，Q10）

| # | 方向 | 场景 | 上界依据 | 0-ULP | 工作量 |
|---|---|---|---|---|---|
| **P0** | prefill：engram 准备与 GPU 流水化 | 所有冷 prefill | trace 实测 GPU 忙 36.76 s / 墙钟 46.5 s；io8→16 带来 +3.7% | **构造保证**（数据不变，只改时序） | 中 |
| **P1** | prefill flash-attn：O 累加器放寄存器 → Q=16 | 冷 prefill，attention 占 20.9% | §89 roofline 差 4.3×；§89.5 漏掉了这条路 | **预期保持**（需门禁实测） | 中-高 |
| **P2** | prefill engram 读取本身：深队列 + 多线程解码 | 冷 prefill，内容越随机越明显 | 依 P0 的测量结果定 | 构造保证 | 低 |
| **P3** | decode：blk.14 的 engram 行与 layer 0–12 重叠 | decode | gpu_gap 中 engram 949 µs（io8）【实测】 | 构造保证 | 低 |
| **P4** | 短 prompt / 热追加的固定成本：先补归因 | 多轮 agent、短问答 | 1.22 s/请求、1.89 s/轮【实测】，其中 56% 尚未分解 | 待定 | 先做测量 |

decode 的 kernel 微调方向我**不建议重开**：§19（norm→GEMV 融合、`MYDS_FUSE_HC_NORM_MIX`）、§4.3、§65、§143.2 已经覆盖了我能想到的融合点。

---

## P0　prefill：engram 准备与 GPU 流水化（最推荐）

### 现状【代码】
- 每个 macroblock 内部，engram 准备在 CPU 上**同步**跑完，之后才开始 embedding 和任何 GPU 工作：[myds_v41.c:7690-7721](file:///Users/fangqingyuan/Projects/myds/src/myds_v41.c#L7690-L7721)。准备按 ≤64 token 分步，每步都等 I/O 全部完成（[myds_engram.c:860-1010](file:///Users/fangqingyuan/Projects/myds/src/myds_engram.c#L860-L1010)，`while (r->remaining) pthread_cond_wait`）。
- 上一个 macroblock 以 `commit_batch`（提交并等待）结束（v41.c:8758-8768）。所以**下一个 macroblock 准备 engram 时 GPU 完全空闲**。32K 以内的 prompt 只有 1 个 macroblock，GPU 在整段 engram 准备期间空闲。
- 两张 engram 表被消费的时间点相差很远：表 0 在 **layer 1** 用，表 1 在 **layer 14** 用（v41.c:7903-7963）。prefill 在 macroblock 内是**层优先**的顺序（layer → chunk，v41.c:7782/7854）。

### 为什么判断这里有空间【实测】
1. trace 文档 §10（[2026-11-03-trace-phase-breakdown.md:206-213](file:///Users/fangqingyuan/Projects/myds/docs/2026-11-03-trace-phase-breakdown.md#L206-L213)）：22.8K prefill **墙钟 46.5 s，GPU 忙（Σ CB 时间戳）36.76 s**。CB 之间会并发（decode 平均 1.4 个 CB 同时在跑，trace §12.2），所以 Σ 只会**多算**忙时，不会少算 ⇒ **GPU 空闲 ≥ 9.7 s（≥21%）**。账本 §0.0「GPU 驻留 99.9–100%」是 10-23 以前在稳态计算窗口里测的，覆盖不到 engram 准备这一段。
2. §143.5：只把 engram 读线程从 8 提到 16，prefill 32K 就 **+3.7%**。说明 engram 准备在 prefill 墙钟里占好几个百分点，而且就在关键路径上。
3. 两条记录有冲突，需要先厘清：§0.0 表里 48K 的 Engram 是 1.16 s（1.5%），可是 io8→16 带来的 +3.7% 至少要求 engram 占比 >3.7%。可能是语料不同，也可能是 10-24 以后 split GGUF / page cache 的条件变了。

### 做法
- **A（单 macroblock）**：开一个后台线程，按 chunk 顺序先产出表 0 的行，再产出表 1 的行。主线程照常编码 layer 0 的全部 chunk（GPU 开始跑）。编码 layer 1 的第 c 个 chunk 之前，在 CPU 侧等「表 0 第 c 块已就绪」；编码 layer 14 之前等表 1 就绪。buffer 是 shared storage，**CPU 在 commit 之前写完就可见，不需要 MTLSharedEvent**。表 1 的读取可以被 layer 0–13（约 35% 的 GPU 时间）完全盖住。
- **B（多 macroblock，>106K）**：GPU 跑第 k 块时，后台准备第 k+1 块的 engram（`prompt_engram_buf` 需要双缓冲）。history 只是最后 3 个 token，第 k+1 块的起始状态在准备开始前就能确定。
- embedding 查表（v41.c:7722+）一起放进后台线程。

### 上界与验证
- **上界 = engram 准备 + embedding 的墙钟 − layer 0 的 GPU 时间。** 先用 `MYDS_PROFILE_PREFILL=1` 读 `prof_engram`。**真实语料（`p32k_ab__v0.txt`）和随机词语料各测一次**。
- 顺带检验一个假设：§144.4 把「随机词比真实语料慢 25%」归因于 MoE 路由局部性，但这一点**没有做过只改这一个变量的 A/B**（违反 §0.5.1）。随机词的 n-gram 几乎全部唯一 ⇒ engram 行全部 page-cache miss，这也能解释这 25%。如果两种语料的 `prof_engram` 差几秒，归因就要改写。
- 0-ULP：engram 数值、kernel、归约顺序都不变，**只是时序变了**。仍然要按 §A.4 四查，并跑 `bin/score --dump`。

### 会失败的情况（Q8）
- 真实语料下 `prof_engram` < 1%，那么只对随机或冷内容有收益，达不到 2% 门。
- 后台读线程和主线程的 CB 编码抢 P-core，导致宿主侧编码变慢。a897cec 已经证明宿主侧时间对墙钟敏感（−1.9 s）。

---

## P1　prefill flash-attn：O 累加器放寄存器，从而解开 Q=16

### 现状【代码】
- 生产 kernel `kernel_myds_flash_attn_heads_f16_dk512_dv512`（Q=8，C=64，NSG=4）。O 累加器 `so` 放在 threadgroup memory，类型是 **float**（`FA_NONVEC_TYPES` 的 o_t = float，[flash_attn.metal:931-937](file:///Users/fangqingyuan/Projects/myds/metal/flash_attn.metal#L931-L937)）。
- 每处理 64 个 key，都要做下面这些事（[flash_attn.metal:665-756](file:///Users/fangqingyuan/Projects/myds/metal/flash_attn.metal#L665-L756)）：
  1. 在 smem 里对 `so4 *= ms` 做一遍整行缩放；
  2. `threadgroup_barrier`；
  3. 把 16 个 `simdgroup_float8x8` 从 `so` 读进寄存器 `lo[]`，做 MMA，再写回 `so`；
  4. 再一个 barrier。

  也就是说，**O 本来就在寄存器里参与计算**，smem 只是在两次迭代之间暂存它。

### 和 §89 的区别（Q2）
§89.4/89.5 只考虑了「把 O 改成 half」（得 45 KB，仍然超）。**没有考虑 O 完全不进 smem**。按 §89.2 的同一套记账（单位 half）：

| 配置 | q | ss（分数） | sk | 合计 |
|---|---:|---:|---:|---:|
| 现状 Q=8，O 在 smem | 4,096 + O 8,192 | 2,048 | 2,048 | **32,768 B** |
| Q=8，O 在寄存器 | 4,096 | 2,048 | 2,048 | **16,384 B** |
| **Q=16、NSG=8，O 在寄存器** | 8,192 | 4,096 | 4,096 | **32,768 B ✓** |
| Q=16、NSG=4，O 在寄存器 | 8,192 | 4,096 | 2,048 | 28,672 B ✓（但每线程 O 要 64 个 float，可能溢出） |

Q=16、NSG=8 时，每个 SG 持有 16 行 × 64 列 = 16 个 float8x8，**和现状每个 SG 的 `lo[NO=16]` 一样多**，寄存器压力不增加。

### 为什么预期能保持 0-ULP【推断，必须过门】
- O 是 float，存进 smem 是无损的；寄存器里的缩放 `lo *= ms[row]` 和 smem 里的 `so4 *= ms` 是同一次 float 乘法，操作数相同。
- Q=8→16 只改变「一个 TG 里放几个头」。每个头的 QK 点积、softmax（`simd_max`/`simd_sum` 仍然是一个 SG 处理一行 64 个 key）、PV 累加的顺序都不变。**唯一的风险**是 `simdgroup_multiply_accumulate` 在不同行 tile 上的内部求和顺序是否完全相同，这只能由 `score --dump` 判定。
- 所以 §89.6「换 tiling 必破 0-ULP」的结论对这条路不成立：这条路没有改 key 的分块方式。

### 分两步做，每步都设门
1. **Q=8 + O 在寄存器**：只省掉每 64 个 key 一次的 smem 往返、整行缩放和一个 barrier。先做组件门（真实 48K 的 layer 2/8/19，concurrent 口径）。组件收益 <5% 就停（§0.5.1）。
2. **Q=16、NSG=8**：K/V 复用翻倍。grid 从 `(8, c_len)` 变成 `(4, c_len)`，host 侧改 [myds_v41_gpu.c:1684](file:///Users/fangqingyuan/Projects/myds/src/myds_v41_gpu.c#L1684) 附近的 `specialized(...)` 参数。

### 上界
flash-attn 本体在 48K 是 16.35 s / 79 s = **20.9%**（§0.0）。§89 的 roofline 差距是 4.3×，§0.6 给的「Q=16 上界」是 **prefill +11~14%**。保守按 attention 本体快 25% 算，约 **+5%**。

### 会失败的情况
- 每个 TG 的 grid 数从 8 降到 4，c_len 很小时占用率不够（2K chunk × 4 = 8K 个 TG，应该够）。
- §75 的多头共享 K tile 慢了 2–6×。但那是另一种 kernel 结构（多头共享、分块不同），**这里的 kernel 主体不变**。真遇到同样的变慢，要先查是不是寄存器溢出。

---

## P2　prefill engram 读取本身（P0 的补充，做 P0 时一起改）

【代码】现在的问题：
- 每 ≤64 token 同步一次，I/O 队列每次都排空。32K prompt 约 500 次。
- `ENGRAM_IO_MAX_THREADS = 16`（[myds_engram.c:323](file:///Users/fangqingyuan/Projects/myds/src/myds_engram.c#L323)），队列深度最多 16。
- 读完以后，E4M3 解码在**调用线程上单线程**跑（每 64 token 3,072 行 × 256 值）。
- 行只有 264 B，但没有设 F_NOCACHE，每次 miss 都会拉进一个 16 KB 页（§143.3 `fs_usage`：每次 16 KB）。page cache 的有效利用率约 1.6%。

建议（都不改数值）：
1. 一次算出整个 macroblock 的 row id（纯哈希），去重后连续投递，队列深度提到 32–64。可以把上限改大，或者改用 `dispatch_io` / `aio`。§25.3 当年测得「16 比 8 慢」，§143 又测得「16 比 8 快」，说明环境已经变了，这个参数要重新扫。
2. 行解码分摊到 worker 上。每行的解码互相独立，结果逐位相同；为了保留 errno 语义，按 job 记录错误，再按原顺序扫描取第一个。
3. 【推断，需离线评估】用户态行缓存（264 B 粒度）：同样多的内存，命中率可能比 16 KB 页缓存高一个数量级。先在 trace 里导出 row_ids 序列，离线模拟 LRU 命中率，再决定做不做。

---

## P3　decode：blk.14 的 engram 读取与 layer 0–12 重叠

【代码】decode 每个 token 在 `begin_batch` **之前**同步读完两张表共 48 行（[myds_v41.c:4180-4182](file:///Users/fangqingyuan/Projects/myds/src/myds_v41.c#L4180-L4182)）。表 0 的 GEMV 在 layer 0 就预置了，所以必须先读；**表 1 要到 layer 13 才用**。

做法：先读表 0 的 24 行，然后开始编码 layer 0。表 1 的 24 行交给后台线程读，主线程在编码 layer 13 所在的 CB（`DECODE_SPLIT=20`，每 2 层一个 CB）之前 join 并写入 `engram_rows[1]`。同样**不需要 shared event**。

与已证否路径的区别：§25.4 否掉的是「采样后、forward 前提前读」（窗口只有 90 µs，而且当时 engram 只要 142 µs）。这里是 token **内部**的重叠，窗口约 7 ms。

上界【实测 + 推断】：§143.3 io8 下 engram 是 949 µs；io16 以后按隔离基准约 530 µs（推断）。藏掉一半约 0.25–0.3 ms / 23.3 ms ≈ **+1.1–1.3% decode**。量级与 §143.5 已采纳的那一条相当，低于 2% 门。**能不能做，取决于你们对 decode 门槛的取舍。**

---

## P4　短 prompt / 热追加的固定成本：先补归因，不先给修法

按 Q10，这是**用户最能感受到的秒数**：
- 多轮 agent 每轮固定 **1.89 s** + 2.68 ms/tok（§114）；
- 短 prompt 固定 **1.22 s**（§115）；
- 其中 **56%（Step1–4/indexer/compressor）从未分解**（§119 的修正）。

建议的第一步，只做测量：
1. 用现成的 `MYDS_PHASE_SPLIT=1` + `MYDS_PHASE_SPLIT_ATTN=1` + `MYDS_PHASE_SPLIT_MOE=1`，在 n = 228 / 802 / 2,129 三档各跑 3 次，把这 56% 拆到标签。trace 文档 §9 已证明开启后输出逐位一致，代价约 1%。
2. 查清 §113 那个「Step4 每 layer-iter 44.6 次 dispatch」到底对应什么划分。
3. 小批量 MoE：n=228 时 MoE 要 645 ms，而全部专家权重按 724 GB/s 流一遍的下限约 0.21 s（152.88 GB / 724 GB/s）。如果 mm 路径在小批量下只达到 244 GB/s，可以考虑小 n 走类似 decode 的 mv-id 路径。但 mm 和 mv 的归约顺序不同，**这一点必然破 0-ULP**，要先定质量策略。

---

## 我不确定的三件事（§C.13 附录要求）

1. **P0 的核心前提**是「Σ CB 时间 < 墙钟 ⇒ GPU 空闲」。我没有用 powermetrics 在 prefill **全程**（含 engram 段）采样复核。动手前先跑一次 `sudo powermetrics --samplers gpu_power -i 200`，覆盖整个 32K 请求，看开头那一段的驻留率是不是明显掉下去。
2. **P1** 的 MMA 行 tile 内部求和顺序是否完全一致，只有实测才能定。如果不一致，Q=16 就退化成「放弃 0-ULP」那一档（§0.6 已列）。但第一步（Q=8 + O 在寄存器）不受影响。
3. §0.0 的 Engram 1.16 s（1.5%）和 §143.5 的 +3.7% 互相矛盾，P0 的收益落在哪个量级，取决于这个矛盾的答案。
