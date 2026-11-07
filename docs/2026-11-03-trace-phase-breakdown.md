# /trace 分阶段拆解：口径与实现（2026-11-03）

**背景**：页面上只有「排队 / 解析+渲染 / 进引擎前 / 引擎总 / prefill / decode / 未归因」七段。
对一次 43 秒的请求，这些只回答「哪一段大」，不回答「为什么大」。
本次把 **prefill、decode、未归因** 三条各自拆开，全部常开、零额外同步点。

## 一、口径（先读这一节，否则数字会读错）

| 字段 | 含义 | 怎么来的 |
|---|---|---|
| `墙钟` | 该阶段的真实耗时 | 已有的起止时间戳 |
| `GPU 忙` | GPU 真正在算的时间 | Σ(command buffer 的 `GPUEndTime - GPUStartTime`) |
| `阻塞等待` | CPU 卡在 `waitUntilCompleted` 的墙钟 | 只在那两个本来就阻塞的函数里取两次时间 |
| `宿主` | 提交 kernel / 采样 / 分词 | `墙钟 - 等待` |

**关键：`GPU 忙` 与 `阻塞等待` 会重叠。异步提交时 CPU 与 GPU 并行，
所以 `GPU 忙 + 等待 ≠ 墙钟`。页面分开列，不能相加去凑满。**

这不是估算：GPU 时间戳是 Metal 自己记录的，wait 是真实的阻塞区间。

## 二、每段怎么看

### prefill 内部

- `GPU 忙` 接近墙钟 ⇒ prefill 是算力受限，优化只能动 kernel/量化。
- `GPU 忙` 远小于墙钟 ⇒ 瓶颈在 CPU 提交或 Python 侧准备，改 kernel 没用。

### decode 内部

- `GPU 忙 ≈ 墙钟` ⇒ 纯 GPU 受限。2026-11-03 实测 997.8ms / 1010ms = 98.8%。
- `回调写出` 大 ⇒ 客户端慢或网络拥塞，**不是模型变慢**。这一条最容易误判。

### 未归因分解

`引擎总 - prefill - decode`，拆成：

- `分词`：tokenizer。长 prompt 时值得看。
- `KV 落盘加载`：磁盘快照 `load+apply`。
- `其余`：首字前的 head+采样、快照写入等。

## 三、常开为什么不影响性能

- GPU 忙时：只读 command buffer 完成时**本来就存在**的两个时间戳。
- wait：只在 `waitUntilCompleted` 前后各取一次 `mach_absolute_time`，那里本来就在阻塞。
- decode 每 token 约 2 次系统调用级开销，实测对 40 t/s 无影响。

**没有新增 commit、没有加锁、没有改变批处理方式。**
逐层计时仍需要 `MYDS_PROFILE_PREFILL=1`，因为那个开关会把 chunk 退化成串行，
测出来的数不代表生产口径 —— 所以页面继续显示「未测量」，不画平线冒充。

## 四、实现位置

| 文件 | 内容 |
|---|---|
| `src/myds_metal.m` | `g_ph_gpu_busy_s / g_ph_wait_s / g_ph_cbs`，`myds_metal_phase_take/reset` |
| `include/myds_internal.h` | `ctx->ph_*` 每请求归因字段 |
| `src/myds_engine.c` | 入口清零；prefill 后 take；generate 末尾 take；分词与 kvdsk 计时 |
| `src/myds_http_io.c` | 回调 `emit_s`（SSE 与缓冲两条路） |
| `src/myds_server.c` | `myds_context_reset_phase_detail` + 填入 trace 记录 |
| `src/myds_trace.c` | JSON `detail` 对象 + 页面三张表 |

## 五、坑（踩过的）

1. **异步 flush 的 command buffer 必须在 `wait_inflight_commands` 里记账。**
   只记 `wait_command_buffer` 时，decode 的 GPU 忙时只有 1.5 ms/token，
   而墙钟是 20 ms/token —— 差了 13 倍，差点据此得出「CPU 是瓶颈」的错误结论。
   `flush_batch` 提交的那批只会在 `wait_inflight_commands` 完成。

2. **未测量必须是 null，不能是 0。** `-1` 在 JSON 里输出 `null`，页面显示「—」。

3. **顶部瀑布图不能同时画「引擎总」和它的子项。**
   原来 7 段一起画，一条 1 秒的请求能画出 2 秒的宽度，比例全错。
   现在 `bar` 只用互不包含的段：queue / render / setup / prefill / decode / unattributed。

4. **`engine - prefill - decode` 归因到的「其余」包含首字前的 head+采样**，
   不要把整块当成"未知开销"去找 bug。

## 六、更细的一层：按阶段标签归集 GPU 时间（第二阶段）

上面只到「prefill 的 GPU 忙 / decode 的 GPU 忙」。要知道**是哪一步在算**，
需要在提交的那一刻就知道这个 command buffer 属于哪个阶段。

做法：`myds_metal_phase_tag_set(tag)` 只是写一个 int。CB 提交时把当前标签存进
与 inflight 数组对齐的表，完成时按标签把 `GPUEndTime-GPUStartTime` 累加进去。
**不新增 commit、不加锁、不加等待。**

标签：engram / embedding / 归一化+Mix / q_a,kv / q_b / attention / attn out /
ffn 输入 / 共享专家+路由 / MoE / head+lm_head / 其它。

### prefill 需要 `MYDS_PHASE_SPLIT=1`

生产路径原本把整个 chunk 编进少数几个 CB，步骤之间没有提交点，标签无从生效。
`MYDS_PHASE_SPLIT=1` 在步骤边界插入 **flush（只提交、不等待）**，让每个步骤
各自成为一个 CB。仍是同一个 queue，顺序不变。

**实测代价（macstudio，556 token prompt，各 3 次，交替重跑以避开前缀缓存）**

| | prefill 墙钟 | CB 数 | decode |
|---|---|---|---|
| 关 | 2.135 / 2.136 / 2.191 s | 82 | 41.5 / 41.5 / 41.4 t/s |
| 开 | 2.129 / 2.134 / 2.190 s | 723 | 41.4 / 41.5 / 41.5 t/s |

中位数 2.136 vs 2.134 s，decode 完全相同 ⇒ **开销在噪声内**，**说明（2026-11-06 更正）**：代码里 `MYDS_PHASE_SPLIT` 是 `getenv(...) != NULL` 才生效，
仓库内不设默认；线上由 myfleet 的 `myds.conf`（`MYDS_PHASE_SPLIT=1`）设置。
所以「默认开启」只对 myfleet 部署成立，直接运行 `bin/myds` 需自己设置。

### 实测归因（同上那轮 prefill，2.13s / GPU 2.05s）

| 阶段 | GPU 忙 | CB |
|---|---|---|
| MoE 专家 | 1170.0 ms (58%) | 80 |
| attention | 381.7 ms (19%) | 81 |
| 共享专家+路由 | 144.4 ms | 80 |
| attn out | 122.2 ms | 80 |
| q_b GEMV | 111.0 ms | 81 |
| q_a/kv | 55.7 ms | 81 |
| head+lm_head | 39.0 ms | 158 |
| ffn 输入 | 28.2 ms | 80 |

### decode 分段

`MYDS_DECODE_SPLIT=20` 本来就把每个 token 切成 20 段（每段 2 层）。
现在每段各带标签，页面直接列出「段 0..19 + head」的 GPU 时间。
实测 200 token 时各段 200–280 ms，head 308 ms —— head 单独成桶，
避免「最后一段总是最慢」的假象。

## 七、又踩到的坑

**引擎侧 `uint64_t` 计数拷进 trace 的 `uint32_t` 数组，整段错位。**
`memcpy` 按目标大小拷，把 `uint64` 数组的前一半当成 52 个 `uint32`，
于是「其它（OTHER, 12）」的 41 个 CB 显示成了 `#12 ... 41` 而 GPU 时间是 0 ——
看起来像「有个神秘的 24 号阶段」。两边类型必须一致。

## 八、MoE 内部细分（2026-11-06，`MYDS_PHASE_SPLIT_MOE=1`）

需要同时设置 `MYDS_PHASE_SPLIT`。在 prefill 的 mm 路径（`n_tok >= MoE mm 阈值`）里，
于 map0 / gate+up+SwiGLU / down / sum6 之间各做一次只提交、不等待的 flush，
新增标签 `MoE·路由分发`、`MoE·gate+up+SwiGLU`、`MoE·down`、`MoE·求和`
（编号 13–16，放在 OTHER 之后，不改变既有编号）。求和之后的 kernel 仍归原 `MoE 专家` 标签。

- 量的是 Metal GPU 时间戳，不同于 `MYDS_MOE_STEP_PROF`（那个量的是提交耗时）。
- 每次 MoE 调用多 3 次 flush，×40 层。**尚未实测其对 prefill 墙钟的影响**，
  开启后不要拿墙钟与账本数字比较，须先按本文第六节的方法交替重测。
- 只覆盖 prefill 的 mm 路径；decode 的 v41 路径（单 token）未细分。
- **`a897cec` 起改为异步 flush**：`commit_batch()`（提交并等待）换成 `flush_batch()`
  （只提交），inflight 达到 256 个 CB 时才真正排空。所以上面「多 3 次 flush」的代价
  已经不再是逐次同步等待。9581fb2 里那个「prefill +1.56%」是在**同步版本**上测的，
  换成异步后**作废**，不要再引用；当前数见第十节。

## 九、attention 与 decode 层内细分（2026-11-06）

### 开关
| 开关 | 作用 | 前提 |
|---|---|---|
| `MYDS_PHASE_SPLIT_ATTN=1` | prefill 的 attention 拆成 6 个标签 | 需 `MYDS_PHASE_SPLIT` |
| `MYDS_PHASE_SPLIT_DECODE=N` | decode 每 N 个 token 采样一次，层内按 3 个子层切 | 独立；被采样的 token 不做 `DECODE_SPLIT` 分段 |

均默认关闭，线上 `myds.conf` 里是注释掉的。

### 新标签（编号 17–25，放在既有编号之后）
prefill：`attn·compressor/indexer 投影`、`attn·压缩KV发布`、`attn·indexer发布`、`attn·carry+窗口环`、
`attn·chunk attention`、`attn·rope+out_a`。
decode：`decode·attn前(HC+投影+indexer)`、`decode·attention归约+输出`、`decode·FFN(HC+MoE)`。

decode 只在层内【没有并发作用域】的三处切（层开头、attention 归约前、HC FFN 前）：
生产路径的 qkv 投影、indexer、MoE 都在内部带并发作用域的函数里，在其中插 flush 会改变调度，所以没切。
被采样 token 的 CB 数 = 采样 token 数 × 40，换算每 token 耗时时用它做分母。

### 实测（macstudio，M2 Ultra，2026-11-06）
- **输出逐位一致**：`temperature=0` 固定 prompt（2536 token，生成 112 token），开关全开 vs 全关，
  两次都是冷启动（临时指向空的 `MYDS_KV_DSK` 目录，避免命中磁盘 KV），输出 sha256 相同（`63da14be…`）。
  注意：第一次对比时 KV 来自磁盘缓存（"all 2535 cached"），prefill 并没有重算，所以不算数；以冷启动那次为准。
- **代价**（`ATTN`+`DECODE` 全开 vs 基线，基线已含 `PHASE_SPLIT`+`PHASE_SPLIT_MOE`；各 6 次 prefill / 4 次 decode）：
  prefill（约 8.4K token）18,189 → 18,371 ms，**+1.0%**；decode 40.83 → 40.61 t/s，**-0.55%**。
- **读数示例**（2.5K token prompt 冷 prefill，GPU 忙 6.69 s）：`attn·chunk attention` 1.53 s，
  `attn·rope+out_a` 347 ms，compressor/indexer/ring/投影合计约 140 ms。
  decode 每 token（7 个采样 token 的均值）：attn 前 4.6 ms、attention 归约+输出 6.9 ms、FFN 11.1 ms。
- 样本量小（decode 采样仅 7 个 token），只能看量级，不要拿来做精细结论。

## 十、性能验收：异步 flush 与 MoE 批拆分（2026-11-07）

`a897cec` 改了三处会影响墙钟的东西：MoE 段内 flush 从 `commit_batch()`（同步等待）
换成 `flush_batch()`（只提交，inflight ≥256 才排空）、`should_flush_moe_batch()` 用
`next_chunk_max` 而非 `chunk_size` 判溢出、以及预热时预分配 chunk/prompt 工作区。

### 口径（先读，否则数字会读错）

- 机器：macstudio（M2 Ultra，192 GB），独占，无其它 myds 进程。
- 两只二进制：`base` = `162d368`（改动前）、`new` = `a897cec` + 本节配套的 trace 修复。
  `metal/` 源码在两个提交之间**零改动**，两只用同一份 `myds.metallib`（sha256 `787f9d7d…`）。
- 配置：`myds.conf` 原样，仅把 `MYDS_KV_DSK` 注释掉（避免每请求写 GB 级快照混进墙钟）；
  `MYDS_PHASE_SPLIT=1`、`MYDS_PHASE_SPLIT_MOE=1`（线上取值），`MYDS_MOE_CHUNKS=9`、
  `MYDS_PREFILL_CHUNK_SIZE=2048`。
- 请求：固定 prompt **22,800 token**，`temperature=0`、`max_tokens=32`、非流式；
  每轮先发一次 `X-Myds-Warmup: 1` 的 306 token 探针，再发 A、B 两个**不同内容**的
  22.8K prompt（内容不同 ⇒ 前缀缓存必然 miss）。
- 交替：`base1, new1, base2, new2, base3, new3`，每只 3 轮 × 2 个请求 = 6 个样本。
- 指标取自服务端自己的 `Completed` 日志行（`prefill: N tok in X s`），
  明细（GPU 忙 / CB 数）取自 `/trace/api/requests`。
- ⚠️ **本节的绝对吞吐（约 480–490 t/s）不可与台账 §105 的 618.9 相比**：本节用的是
  生成的英文随机词 prompt，而 §105 用的是真实语料。同 token 数下两者差 **25%**
  （见第十一节）。**本节唯一成立的是 A/B 两臂之间的相对差**（同一 prompt 文件、交替各 3 轮）。

### 结果

| 指标（每只 3 轮 × 2 请求 = 6 样本） | base `162d368` | new `a897cec`+修复 | Δ |
|---|---|---|---|
| prefill 墙钟 · A（22,800 tok） | 47.5 s（47.5–47.7） | 46.5 s（46.4–46.5） | **−1.0 s / −2.1%** |
| prefill 墙钟 · B（22,767 tok） | 47.3 s（47.3–47.6） | 46.3 s（46.3–46.7） | **−1.0 s / −2.1%** |
| prefill 吞吐 | 479.9 t/s | 490.5 t/s | +2.2% |
| GPU 忙（Σ CB 时间戳） | 36.75 s | 36.76 s | 0（噪声内） |
| 阻塞等待 | 33.2 s | 33.9 s | +0.7 s（见下） |
| 宿主（墙钟 − 等待） | 14.4 s | 12.5 s | **−1.9 s** |
| prefill CB 数 | 2,925–2,935 | 2,657–2,671 | −9.2% |
| decode | 38.3–38.9 t/s | 38.5–38.8 t/s | 无差别 |
| 输出文本 | — | — | **12/12 逐字节相同** |

两臂的样本区间**不重叠**（new 最大 46.7 s < base 最小 47.3 s），所以这 1.0 s 不是噪声。
另：计时用的 new 二进制还差一处**只影响显示**的修复（in-flight 记录的 detail 也置 -1）。
最终二进制（+该修复）另跑了第 4 轮 `new4`：prefill 46.5 s / 490.8 t/s / decode 38.8 t/s，
落在上面 new 的区间内 ⇒ 该修复对耗时无影响。
复现脚本与原始产物在 macstudio `~/bench/ab/`（`run_one.sh` / `driver.sh` / `results/`）。

### 怎么读

- **增益来自宿主侧提交，不是 GPU。** 两只的 `GPU 忙` 都是 36.7 s（逐次一致到 10 ms），
  只有 CB 数与 prefill 墙钟变了 ⇒ 省下的是「提交后立刻等它完成」的那段串行时间。
- **MoE 批拆分的修复在这组口径下没有体现**：22,800 token 的 macroblock 是 23,040，
  MoE 批上限 16,384，8 个 chunk 攒满就必然要 flush 一次，走不到
  「尾块放得进剩余批空间」那条被修的分支。它只在 `moe_batch` 接近 macroblock
  长度时有效（如 5014 / 6144 那组），由 `tests/test_prefill_shrink_grid.c` 的穷举覆盖，
  本轮**没有**对应的实机 A/B —— 不要把这轮的 −2.7% 算到它头上。
- **预热预分配对 >16,384 token 的 prompt 不完整**：预热按
  `cfg_chunk_size × cfg_moe_chunks = 2048 × 8 = 16384` 分配，而 22.8K prompt 的
  macroblock 要 23,040，日志里仍会出现一次
  `prefill prompt workspace expanded: tokens_cap=16384->23040`。要真正「预热完就不扩」，
  预分配的上界应取 macroblock，而不是一个 MoE 批。
- decode 速率 38.4–38.9 t/s 两臂无差别 —— 这个提交没碰 decode。

## 十一、prefill「掉下来」的复核：不是代码，是我的口径（2026-11-07 08:10）

2026-11-07 上午我用生成的英文随机词 prompt（22.8K / 39K token）测得 prefill 约 490 t/s，
据此怀疑相对台账 §105（618.9 t/s @32K）掉了 20%。**这个结论是错的，已撤回。**

### 决定性对照

用 §105 自己的语料文件 `benchmarks/register_only_2026-10-27/prompts/p32k_ab__v0.txt`
（52,032 字符 ⇒ 32,224 token）跑当前 HEAD：

| 臂 | prompt | token | prefill |
|---|---|---|---|
| 台账 §105（2026-10-27） | `p32k_ab__v0.txt` | 32,210 | **618.9 t/s**（54.3–54.4 s） |
| `691bc26`（2026-10-27 构建） | 同一文件 | 32,225 | **615.2 t/s**（52.4 s） |
| 今天 HEAD `b0b7b77` | 同一文件 | 32,224 | **619.4 t/s**（52.0 s） |

**同一份语料，三个数在 0.7% 以内。** 既没有掉，也没有靠改配置找回来。

证据与复现脚本：`benchmarks/prefill_claims_check_2026-11-07/`
（`extract.py --check` 与 `verify.py` 都不需要 GPU）。

### 为什么我先前测出 490

| 我用的 prompt | token | prefill | 二进制 |
|---|---|---|---|
| 英文随机词（A/B） | 22,800 | 490.8 | HEAD |
| 英文随机词（阶梯） | 38,980 | 496.4 | HEAD |
| 英文随机词（阶梯） | 38,980 | **497.0** | 10-27 `691bc26` |
| 英文随机词（阶梯） | 77,767 | 480.5 | HEAD |
| 台账 Markdown 原文 | 61,757 | 464.7 | HEAD |
| **§105 语料（中文）** | **32,224** | **619.4** | HEAD |

⇒ **同 token 数下，prompt 内容能把 prefill 拉开 25%**（619 vs 496）。
随机词表的 MoE 路由几乎均匀铺满所有专家、且没有可利用的局部性；
真实语料的路由集中，权重驻留命中率高。生成文本做 A/B 会系统性低估。

### 顺带否掉的两个「嫌疑人」

| 假设 | 实测 | 结论 |
|---|---|---|
| 10-27 → HEAD 代码回归 | 同一 38.9K prompt：`691bc26` 497.0 / HEAD 496.4；台账语料上 615.2 / 619.4 | 否 |
| `MYDS_CHUNK_KEY_MAX` 524288→1048576（234bf98 的首要候选） | 498.0（512K） vs 498.9（1M） | 否 |

### 规则（加到本文件的硬约束）

**prefill 吞吐 A/B 必须用 `benchmarks/register_only_2026-10-27/prompts/` 里的语料，
并标注文件名。** 用生成文本得到的数只能和自己比，不能和台账比，也不能拿来说明回归。

这条已经变成机器检查，不再靠记性：

- **寄存器** `benchmarks/prefill_claims.json` —— 本文件里出现的每个 `t/s` 字面量
  都必须在册，且写明 `prompt_file` 与 `comparable_to_ledger_105`。
- **检查器** `tools/prefill_claims.py` —— `comparable_to_ledger_105: true` 必须指到
  `ledger_corpus` 里的文件，且 sha256 与 manifest 一致；没有语料文件却声称可比，
  直接失败（就是上面那次误判的形状）。
- **门** `make audit-ledger` —— 跑它就会跑上面两条，外加三个自测，确认检查器本身有牙。
