# DeepSeek V4.1 Flash prefill 与 decode 性能续查

> **10月26日完整全模型结果**：新观察者组件门通过后，K-resident 已完成私有接入、三方数值/状态资格和唯一24代正式窗口。两负载 prefill 保守节约 **1.591% /1.514%**，未达冻结 >2% 采纳门，decode 无回退门也失败，保持私有default-off，无HTTP/生产采用。[最新完整结果](2026-10-25-indexer-observer-and-qb-rope-review.md)。本文早先R2失败按原cohort保留。

> **同日后续**：对原观察者的同 PSO 对照复现了顺序偏移；显式匹配控制输出复用距离、先过 null 校准的新 cohort 已通过六组组件门（actual exact 保守节约 40.17%–40.49%，TILED 12.76%–13.02%，最大控制漂移 0.06935%）。候选已进入独立源码快照的 default-off runtime 资格验证。本文以下 R2 失败结论保持不变；后续状态以[观察者与私有全模型续查](2026-10-25-indexer-observer-and-qb-rope-review.md)为准，尚不能由组件结果推断生产收益。

2026 年 10 月 25 日。**indexer 的 K 矩阵跨 head 驻留候选已经推进到 M2 Ultra 的真实 32K prefill 输入**：相对实际默认 exact scorer，三个目标层的组件保守耗时节约为 40.10%–40.32%；相对同构建 TILED，额外节约为 12.47%–12.88%。完整生产编译单元关闭了此前的外部原生制品身份差异，真实 scores、完整有序 Top512 和输入 canary 门也已通过。

**冻结的联合性能门仍未通过。** layer 2 / TILED 控制的 A1→A2 漂移为 −2.08460747%，超过 ±2% 上限。其余五组通过，但不能代替六组全部通过。本轮保留唯一正式 cohort，没有替换样本、放宽阈值、接入 runtime 或进入全模型/HTTP 验收，尚无新的全模型或生产提速。详细结果见[真实输入续查](#续查完整编译单元与真实-32k-输入)。

decode 本轮没有得到可采纳的新实现。SharedGU 迟汇合原型经 Metal 屏障语义复核后撤回。prefill 的 attention 输出累加器驻寄存器原型未过 M4 数值门。两个 HTTP benchmark 的 token 计量错误已修复，并加入 13 个测试。

## 当前基线与账本边界

本轮开始时源码为 `59e7989`。工作期间同一 checkout 的其他工作提交了 `224dc70`，其中包含尚未完成的候选文件；本轮没有执行 Git 提交。撤回 decode 候选时按初始源码逐文件还原，没有按新 HEAD 恢复未完成的代码。六个相关运行时、header 和原生 benchmark 文件与初始版本的完整 SHA 一致，见[撤回核对](benchmarks/2026-10-25-performance-reopen/decode-candidate-withdrawal.json)。Engram、GGUF 及其他并发修改保留。

现场读取的部署配置 SHA256 为 `0a3524b030a73916f7684d841b912c6e978c680825a3c3520ca18188f35a2cff`。未设 `MYDS_PREFILL_ATTN`，当前源码默认 `vec`；未设 `MYDS_ATTN_VEC_HEAD_FIRST`，默认开启。也未设 `MYDS_INDEXER_TILED`，在无其他环境覆盖时默认 chunk scorer 是 `kernel_myds_indexer_score_chunk_exact`。首轮只读服务观察没有获得该服务的实际 provider counters，不能把历史 §85 的 TILED 收益当作当前已经启用的基线；后续隔离真实 forward 已取得 exact 的实际 dispatch 见证。首轮配置、制品及各组件运行前后的服务状态见[现场记录](benchmarks/2026-10-25-performance-reopen/m2-deployment-observation.json)。

账本有三处需要限定推论范围：

- **§89 的 32 KiB 上限针对 matrix attention 实验。** 当前 `src/myds_v41_gpu.c` 默认走 vec/head-first，matrix 需要显式 `MYDS_PREFILL_ATTN=matrix/auto`。其 Q=8 的共享内存账不能证明该 vec 路径没有优化空间。
- **§71 的 0.07% 是 command buffer 之间的间隙。** `gp_note_cb()` 读取整 CB 的起止时间，不能观察 CB 内部 dispatch 依赖或约束同 cohort 并发压缩 busy span 的收益。Apple 对 [GPUStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime) 的定义也只覆盖整 CB。
- **§92 的 3.05% 是 SGBAR 这个具体变体的组件结果。** 它不能作为所有 indexer 加载、寄存器分配和指令调度方案的收益上界。本轮 K 驻留实验与降低屏障粒度不同。

这些边界修正不改变旧实验的具体负结果，也不证明存在两位数的全模型空间。插入 commit/wait 的分段 profile 可以帮助定位，但其时间份额不能直接当普通异步图的关键路径上界。

## Indexer K 矩阵驻留实验

原 `kernel_dsv4_indexer_scores_tiled_storage` 每个 TG 处理 8 token × 32 key。在 32 个 head 的循环内，它反复把相同 K 的 16 个 `simdgroup_float8x8` 从 threadgroup 内存加载到矩阵寄存器。

候选只把这 16 次加载移到 head 循环外，保留每个 head 内 16 次 MMA 的顺序、ReLU、权重乘法及 head 累加表达式。每 TG 的 K 矩阵逻辑加载从 `32 × 16 KiB` 变为 `1 × 16 KiB`，少约 496 KiB 的 threadgroup 读取，代价是每 lane 约多 30 个常驻 F32 值。这里没有减少同等数量的 DRAM 流量；尚无 spill、occupancy 或硬件 LDS 计数证据来确定收益归因。

候选以独立符号构建，不接入运行时或生产 shader 清单。相关工具：

- [构建器](../tools/build_indexer_k_resident_probe.py)：精确抽取原模板作为同构建控制，逐项记录变换，并验证逆变换可逐字节恢复原模板。
- [GPU harness](../tools/bench_indexer_k_resident.m)：检查实际选中符号、完整 scores、poison、因果 `-Infinity`、NaN、stride padding、输入 SHA 和 canary。支持外部原生 metallib 控制。
- [独立读者](../tools/review_indexer_k_resident.py)：不导入构建器或 harness，直接解析 72 字节 ABI 与保存的完整输出，核验 SHA 并从 GPU timestamps 重算中位数。

### 同构建组件结果

两台设备分别执行 F32/BF16 共 16 组确定性合成输入。输入包含非整 tile、全部不可见的因果块、尾部遮罩、NaN 和带 padding 的 stride。普通输入值采用 BF16 可表示的网格；这不是任意 F32 分布或真实模型输入的覆盖证明。

每台设备的完整同构建实验各保存 16 组数值 case、48 条 parity；其中 4 个大 shape case（F32/BF16 × 两种 shape）分别对两个 control 计时。每个 case/control 配对执行一次热身加五组正式交错 A1/B/A2，共 144 条 GPU timing。每臂一个 CB、一次 dispatch，编码器为 concurrent。每台设备的完整复核分别检查 160 个 raw 文件、28,343,598 个 score cell、113,450,160 个输出字节、10 个可见 NaN 和 1,152 个因果 `-Infinity`。

下表为 **BF16 组件**，单位 ms/次 scorer dispatch。保守节约为 `1 − B/min(A1,A2)`，不是 tok/s 提升。

| 设备 | token × key | 控制 | A1 | B | A2 | 保守耗时节约 | 控制漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M4 | 128 × 4096 | 0 | 5.890 | 5.156 | 5.885 | 12.394% | −0.079% |
| M4 | 128 × 4096 | 1 | 5.905 | 5.309 | 5.850 | 9.263% | −0.925% |
| M4 | 256 × 16384 | 0 | 47.618 | 41.785 | 47.660 | 12.251% | +0.088% |
| M4 | 256 × 16384 | 1 | 46.008 | 40.259 | 46.555 | 12.496% | +1.189% |
| M2 Ultra | 128 × 4096 | 0 | 1.518 | 1.376 | 1.564 | 9.401% | **+3.002%** |
| M2 Ultra | 128 × 4096 | 1 | 1.569 | 1.376 | 1.534 | 10.335% | **−2.183%** |
| M2 Ultra | 256 × 16384 | 0 | 10.907 | 9.440 | 10.871 | 13.164% | −0.332% |
| M2 Ultra | 256 × 16384 | 1 | 10.879 | 9.424 | 10.830 | 12.980% | −0.458% |

本表两个 control 在各自同构建实验中使用同一库，不能当作两个独立制品控制。M4 所有组最大控制漂移为 1.462%；M2 的小 BF16 shape 两组都超过 2%，完整保留，不能据大 shape 正值判联合门通过。M2 保留了已有 8099 服务，属于 **非独占组件筛选**。

全部 F32 结果、每条原始 timestamp 和独立复核分别见 [M4 读者结果](benchmarks/2026-10-25-performance-reopen/m4-indexer-review.json)、[M4 JSONL](benchmarks/2026-10-25-performance-reopen/m4-indexer.jsonl)、[M2 读者结果](benchmarks/2026-10-25-performance-reopen/m2-samebuild-review.json)及 [M2 JSONL](benchmarks/2026-10-25-performance-reopen/m2-samebuild.jsonl)。

### 首轮外部原生制品身份差异（后续已闭合）

本段保留首轮失败及当时的判断；完整生产编译单元的关闭证据见下方续查。旧失败数据没有覆盖。

M2 首次使用当前部署路径的 metallib（SHA256 `3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9`）作第三方控制。case 1 的 7 token × 31 key F32 输入中，**外部原生核与同构建原生核已经有 127/217 个 scores 的位模式不同**。canary、因果尾部和 padding 均正常，candidate 尚未执行该 case。读者正确分类为 `native_identity_gate_failed`，不能把它记为 candidate 数值失败。

初次隔离构建使用 Metal minver 26.0，原构建脚本默认 15.0。保留初次失败后，单独将 Metal minver 改为 15.0 再做一次身份检查，仍在同 case、同 provider 失败。因此 minver 差异不能解释或关闭该问题。两次原始失败见 [26.0 身份门](benchmarks/2026-10-25-performance-reopen/m2-deployed-native-gate-review.json)和 [15.0 身份门](benchmarks/2026-10-25-performance-reopen/m2-deployed-native-min15-gate-review.json)。没有反复重测到通过，也没有把外部控制替换为同构建控制后宣称金库门通过。

同构建筛选是为判断这个源代码变换是否值得继续，不能替代部署制品等价性。本轮控制均为 TILED；所读配置与当前源码推导的默认 provider 是 exact，两者不能混用。下一步要使用完整生产 shader 编译流程建立可信原生对照，核验实际启用的 provider，并把实际 exact 或其他基线与 candidate 的完整 scores 和耗时直接对照，再检查真实模型的排序后 TopK IDs、下游 attention、logits、tokens 与状态。

按历史 48k 诊断中 indexer 约 9.3% 的份额作粗略优先级估算，若 12.5% 的组件节约完整传递，整段时间只约减少 1.16%。这个估算不构成全模型上界或收益，且不能保证通过既有 >2% 采纳门；更长上下文的 indexer 占比需在当前基线重测。

## 续查：完整编译单元与真实 32K 输入

### 完整生产编译单元恢复原生函数身份

[构建器](../tools/build_indexer_k_resident_probe.py)新增 `--full-library`：按生产 `build_metallib.py` 中的原顺序，保留全部 29 份 shader、PREAMBLE、COPY_KERNELS 和原编译参数，完整 native prefix 不改，只在末尾追加候选模板与两个独立 symbol。完整 native 源为 1,574,167 字节，SHA256 `bc6603c97ec24af7597f6b10c7ed2a7bfe1344379733a0a4df780250a13f3224`，与部署目录的 `build/myds_all.metal` 相同。Metal minver 为生产默认 15.0，CPU harness minver 为 26.0。全部 shader 字节、偏移、参数和编译命令见[来源清单](benchmarks/2026-10-25-indexer-real-input/full-library-source-identity.json)。

M2 新完整编译单元与冻结的部署库 `383710…` 做三方比较：部署原生 TILED、完整 probe 中的原生 TILED、同 probe 中的 K 驻留候选。16 个 F32/BF16 case、48 条 parity 全部通过。独立读者核验了 160 个 raw 文件、28,343,598 个 score cell 和 113,450,160 个输出字节；144 条 timing 的最大绝对控制漂移为 1.1966%，四组 BF16 的保守节约为 12.1948%、12.9690%、13.5040%、13.5046%。这些仍是合成组件结果。[完整复核](benchmarks/2026-10-25-indexer-real-input/full-library-independent-review.json)、[原始 JSONL](benchmarks/2026-10-25-indexer-real-input/full-library-synthetic.jsonl)。

原失败 case 1 的 args、Q、W、K 在旧、新实验中逐字节相同，部署 provider 的输出也相同；新完整编译单元的原生及候选均与部署输出一致。[实际 AIR 审计](benchmarks/2026-10-25-indexer-real-input/full-library-ir-audit.json)进一步确认，新完整 native 与 probe 中原 native 的 F32/BF16 函数 IR 在规范化 metadata 编号后与部署库相同，旧隔离 native 则不同。原模板、ABI、字段偏移和 symbol 均未变化，因此原问题已定位到隔离构建没有保持原生函数的编译身份。

AIR 中可见乘法关联次序差异，但本例 scale 为精确的 `1/64`；FMA 收缩或其他代码生成差异没有被单独隔离。这里只能确认完整编译单元关闭了原生身份门，不能声称某一个编译器变换是唯一根因，也不能外推任意 F32 输入或全模型正确性。两次旧隔离失败继续保留。

### 真实模型捕获及数值门

通过目标 GGUF 的真实 tokenizer 和原生 `parse_chat_prompt` 渲染工程代码审阅请求。请求包含重复的代码片段，共得到 38,404 个 token，本次按协议取前 32,768 个实际 token ID 执行真实模型 prefill。它提供真实 forward activation，**不代表生产业务流量分布**。请求、渲染结果、全部 token 与截取 token 的身份分别冻结，没有用随机张量代替模型输出。[tokenizer 记录](benchmarks/2026-10-25-indexer-real-input/tokenize.json)、[预冻结协议](benchmarks/2026-10-25-indexer-real-input/protocol-v2.json)。

模型路径为 `DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf`，362,142,629,888 字节；保存 stat 及首 1 MiB SHA，未重算整个模型 payload 的 SHA。配置保持原字节，62 项实际 `MYDS_*` 与冻结配置全部匹配，`PREFIX_CACHE=1`、`PLD=1`，context capacity 为 327,680。实际加载 metallib SHA 为 `3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9`。[身份记录](benchmarks/2026-10-25-indexer-real-input/baseline-identity.json)、[完整捕获复核](benchmarks/2026-10-25-indexer-real-input/capture-verified.json)。

在 global chunk start 30,720 的 2,048-token chunk 中，捕获 layer 2 / 8 / 14 的首个实际 scorer batch。三层均为 **128 token × 15,424 key、32 head × 128 维、ratio=2、BF16 cache、key window=0**；2 Mi-float 的 score budget 决定了实际子组大小。实际 dispatch 见证确认当前默认 provider 为 **`kernel_myds_indexer_score_chunk_exact`**，没有强制开启 TILED。每层保存完整 Q、weights、cache、native args、canonical args72、1,974,272 个 scores 和 65,536 个有序 Top512 IDs，以及实际 TopK dispatch 和 provider counters。仅覆盖三个指定层的这个 batch，不覆盖整个请求的全部 batch。

捕获由 `MYDS_INDEXER_CAPTURE` 编译期开关启用，普通编译预处理结果中不存在 observer 或新增 wait。捕获插入了 commit/wait，因此其 wall time 不作为性能数据。[捕获工具](../tools/build_indexer_capture.py)、[捕获 observer](../tools/indexer_capture.inc)、[CPU 读者](../tools/review_indexer_capture.py)。捕获 reader 首次因配置 dump 的注释行解析错误而拒绝，修正 parser 后在原 raw 上复核通过；旧 reader 错误也保留，没有重跑捕获来抹掉记录。

[真实回放 harness](../tools/bench_indexer_k_capture.m)先完成三层全部数值门，再允许任何计时；actual exact 使用冻结部署库，TILED 与候选使用完整 probe。原生 `myds_v41_gpu_topk_causal` 继续从冻结部署库加载。完整 raw 复核确认：

- actual exact 的 scores 全部逐位复现原捕获。
- 候选与同构建 TILED 的全部 5,922,816 个 scores 逐位相同。
- **exact 与 TILED 并非全部逐位相同**：layer 2 有 4 个 score cell 不同，最大绝对差 `9.5367431640625e-7`；layers 8 / 14 没有差异。这属于原 provider 切换的差异，不能记成 K 驻留引入的误差，也不能声称已通过 exact→TILED 的逐位分数门。
- 三层、三 provider 的全部 **589,824 个有序 Top512 IDs** 与原捕获逐字节相同，实际 TopK 的 provider counters 路径也一致。这是本次输入的消费者门，不替代下游 attention、logits、输出或状态门。
- 90 份 raw、109,811,328 个逻辑 payload 字节的 size/SHA 均正确。17,768,448 个保存 score cell 的可见有限值、全部因果 `-Infinity`、poison、输出和 TopK canary 均通过；phase1 / phase2 后全部 18 份输入 post raw 与原 Q/W/cache 完整相同，两侧 canary 正常。

[逐条原始 JSONL](benchmarks/2026-10-25-indexer-real-input/replay.jsonl)、[完整 artifact 清单](benchmarks/2026-10-25-indexer-real-input/replay-artifact-manifest.json)、[root 独立 raw 复算](benchmarks/2026-10-25-indexer-real-input/root-raw-review.json)。新增[独立读者](../tools/review_indexer_k_capture.py)没有导入 builder 或 harness，独立核验了 29 份 Metal、59 份 replay 和 60 份 capture 源码、实际 binary/library、全部 raw、21 行初始化日志严格白名单，以及全部时间戳；[最终复核报告](benchmarks/2026-10-25-indexer-real-input/real-independent-review.json)确认数值门通过、冻结性能门失败、`fullmodel_eligible=false`。7 个 CPU 反例 / 边界测试覆盖算术与 guard 失败、伪造 pass、时间戳 / summary 矛盾和严格阈值，未修改 GPU raw。

本次捕获与回放都持有共享实验锁，前、中、后的识别推理进程检查未发现竞争；这不等价于证明整台机器没有其他 GPU 使用者。运行前后 canonical binary、metallib、配置 SHA 与冻结身份相同，没有启动或停止服务。[实际 argv 与制品](benchmarks/2026-10-25-indexer-real-input/replay-execution-plan.json)、[进程检查](benchmarks/2026-10-25-indexer-real-input/replay-monitor.json)、[窗口结果](benchmarks/2026-10-25-indexer-real-input/replay-window.json)。[结束时复核](benchmarks/2026-10-25-indexer-real-input/post-gates-identity.json)另确认模型 stat / 首 1 MiB SHA 仍与冻结值相同。

### 唯一正式真实组件 cohort：收益明显，联合门失败

协议 v2 在任何真实回放 GPU 提交前冻结，保留了 v1 及修订原因。要求每层分别对实际 exact 和同构建 TILED 两个 control 进行 A1/B/A2；每臂一次热身加九个正式样本，共 180 条 scorer 计时 CB，其中 18 条热身、162 条正式。数值门和 TopK 另外提交的 CB 不包含在这 180 条中。每条计时 CB 一次 scorer dispatch，concurrent encoder；score 全量检查和 TopK 不计入 GPU scorer 时间。正式样本逐次核对完整 scores SHA，未替换、丢弃或补样。每层 / control 均要求绝对漂移 ≤2%，且 `1 − B/min(A1,A2)` 严格 >2%。

下表单位为 ms / scorer dispatch。**40% 包含 exact→TILED 与 K 驻留的组合变化；约 13% 才是相对 TILED 的新增候选收益。** 这些数据来自已捕获输入的重复回放，没有证明 cold DRAM 性能或真实全模型关键路径收益。

| layer | A control | A1 | B | A2 | 保守耗时节约 | 控制漂移 | 冻结门 |
|---:|---|---:|---:|---:|---:|---:|---|
| 2 | 当前 exact | 7.686833 | 4.556125 | 7.608542 | 40.1183% | −1.0185% | 通过 |
| 2 | 同构建 TILED | 5.314750 | 4.533958 | 5.203958 | 12.8748% | **−2.0846%** | **失败** |
| 8 | 当前 exact | 7.721500 | 4.541625 | 7.609958 | 40.3200% | −1.4446% | 通过 |
| 8 | 同构建 TILED | 5.297125 | 4.552250 | 5.200750 | 12.4694% | −1.8194% | 通过 |
| 14 | 当前 exact | 7.726750 | 4.557375 | 7.608083 | 40.0983% | −1.5358% | 通过 |
| 14 | 同构建 TILED | 5.303625 | 4.559875 | 5.234250 | 12.8839% | −1.3081% | 通过 |

harness exit 0 表示数据采集与数值检查完成，**不表示冻结性能门通过**。layer 2 / TILED 的绝对漂移超上限 0.08460747 个百分点，六组联合门因此失败。不能用其余五组、较大的收益幅度或平均结果覆盖。没有为取得通过而重跑这组，也没有将门改为 2.1%。当前记录尚不足以将漂移唯一归因于热态、缓存、GPU 频率或测时器。

本轮到此停止采纳流程：候选未接入 runtime 或生产 shader 注册，未执行全模型 native/HTTP、下游 attention/logits/output/state 或取消恢复验收。后续若继续，应先用独立控制实验定位 A1→A2 偏移，解决测量稳定性后重新冻结实验；本 cohort 永久保留为联合门未通过。

### 全模型验证工具的准备范围

新增[全模型 driver](../tools/bench_indexer_fullmodel.c)、[独立构建目标](../tools/indexer_fullmodel.mk)及[受校验生成器](../tools/prepare_indexer_fullmodel.py)，复用旧 driver 的原生 prompt renderer、callback 和 fresh-context 生命周期。旧 `bench_output_a_fullmodel.c` 原字节保留。配置在 model/context 创建前 `load(true)`，保持 `PREFIX_CACHE=1` / `PLD=1`；每次样本仍要求真实 stats 证明 cache miss。无 runtime provider adapter、无完整 logits/state sinks 或 sinks 不为空时拒绝运行。

[状态 exporter](../tools/indexer_fullmodel_state.inc)适配当前多槽 anchor，检查实际 `anchor_cap`，保存每槽长度及真实 token 内容，替代旧的 scalar 解释及 1,024 上限。准备覆盖最终 vocab logits、完整 tokens/output bytes、RNG、target active window/cache、Engram/prefix/anchor 和 published snapshots；没有 intermediate logits、draft state、持久磁盘 payload 或缺少 getter 的历史 snapshot Engram tails 证据。

provider 合同要求候选在首 callback 前实际编码、decode 不产生候选编码，拒绝 fallback/failure、回退计数及 NULL/空 symbol。12 个纯 CPU fixture 通过；完整引擎/driver CPU 编译链接通过，最终合同收紧后 wrapper 再次编译通过。实际二进制仅执行无模型加载的 `--preflight`，没有把工具准备写成全模型验收。

本轮小型来源、协议、配置、原始 JSONL、复核及窗口记录位于 [evidence-index.json](benchmarks/2026-10-25-indexer-real-input/evidence-index.json)。完整 shader、二进制、模型捕获与 GPU raw 保留在 M2 的 `/private/tmp/myds-indexer-r2-20261025`；本机保留完整真实捕获 / replay raw 副本于 `/tmp/myds-indexer-r2-real-local-20261025`，完整 IR 审计于 `/tmp/myds-indexer-r2-independent-full-20261025`。仓库中的小文件不能替代这些大 payload 做独立 GPU raw 复算。

本轮已执行的 CPU 检查共 38 项：builder 6、真实 reader 7、全模型 driver 12、HTTP stream 13，全部通过；已加入 `make test`。`git diff --check` 通过。这些检查用于验证工具合同，GPU 数值和组件裁定仍来自上述冻结窗口及完整 raw。

## 已停止的两个候选

### Attention 输出累加器驻寄存器

把 gathered head-first vec 中每 lane 的四个 `so4` 累加器移入寄存器，保留 Q、score、PV 和 20 轮 32-key online softmax 顺序。R1 使用 `contract(off)`，R2 另用 `precise::fma` 固定 scale/add 的舍入边界。

两个版本在 M4 首个 1-row 合成输入均未过逐位门：32,768 个 F32 中 30,196 个位模式不同，首差 `0xbcf03830 → 0xbcf03824`，最大绝对差 `1.043081283569336e-7`。finite 和 canary 正常，R1/R2 的 candidate 输出 SHA 相同。没有性能计时，不将小绝对误差代替逐位门，也不推断目标 M2 上的数值行为。[独立 raw 比较](benchmarks/2026-10-25-performance-reopen/so4-m4-independent-review.json)。

[独立构建器](../tools/build_prefill_output_register_probe.py)和 [synthetic harness](../tools/bench_prefill_output_register.m)保留，生产不调用。构建器还可生成完整真实 capture 重放程序；本轮未完成目标机真实 capture 门。首次目标机独占预检发现其他 `myds` 进程后即拒绝运行，未停止该服务。

### Decode SharedGU 迟汇合

原型试图把 SharedGU 提前至 HC mix/router 所在 cohort，并从后续 resource barrier 的列表移除 Shared MID，使其跨越 route 阶段继续运行。

本机 Xcode SDK 的 `MTLComputeCommandEncoder.h:324` 明确规定 `memoryBarrierWithResources:count:` 等待 encoder 中此前全部 dispatch 完成；资源列表约束副作用可见性。[Apple WWDC22 官方说明](https://developer.apple.com/videos/play/wwdc2022/10101/)同样说明 memory barrier 会让后续命令等待此前命令结束。因此移除 MID 不能实现预期的跨 cohort 迟汇合。

候选已完整撤回。它只剩与 cohort 1 的 mix/router 同时执行这一更窄机会，原先按横跨多个 cohort 推出的收益估算撤销。成熟路径在同一 cohort 内的已有并发不受此结论影响。本轮没有新 decode GPU 性能结果。

## HTTP 基准计量修复

旧 `eval/decode_fixed.py` 以非空 SSE 文本块计 token，并用 `total − TTFT` 作分母。服务端先逐 token 记数，再经 UTF-8 等缓冲输出；文本块与 token 不一一对应。128 个 token 可能只有 120 个非空块。即使一块恰好一个 token，分母已排除首 token 的等待而分子仍包含它，也会产生约 `128/127 − 1 = 0.787%` 的偏差。usage/DONE 传输尾巴还会污染分母。

[共享 reader](../eval/stream_bench.py)现在从最终 `usage.completion_tokens` 取数，要求达到请求上限、`finish_reason=length`、完整 usage 和 `[DONE]`。缺失、提前停止、timeout、HTTP 截断均拒收。首末非空文本时刻与总结束时刻分开保存。

输出明确称为 HTTP decode **估计值**，采用 `(completion_tokens − 1)/(last_visible − first_visible)`。UTF-8、reasoning 缓冲和不可见 token 仍会使首个 visible delta 偏离首个生成 token；正式采纳仍应使用原生回调的 N−1 个 decode interval。prefill 输出改称 TTFT 与 tokens/TTFT，usage 的完整 prompt 长度不能单独证明 cache miss。

两份 CLI 支持可配置输出数和 `--jsonl`，保留 warmup、所有原始时刻、usage、文本块数、finish、DONE 及拒收记录。[13 个 CPU 测试](../tests/test_stream_bench.py)覆盖这些计量与故障边界，已接入 `make test`。当前修复不能倒推出历史每次读数偏差多少，也没有把历史吞吐重算成新基线。

## 复现与后续验证

独立合成组件无需模型文件，不会修改服务。先构建再运行：

```sh
python3 tools/build_indexer_k_resident_probe.py \
  --source-root "$PWD" --out-dir /tmp/myds-indexer-k-new --full-library
/tmp/myds-indexer-k-new/bench_indexer_k_resident \
  --metallib /tmp/myds-indexer-k-new/probe.metallib \
  --native-metallib /tmp/myds-indexer-k-new/native.metallib \
  --out-dir /tmp/myds-indexer-k-new/results > /tmp/myds-indexer-k-new/results.jsonl
python3 tools/review_indexer_k_resident.py \
  --build-dir /tmp/myds-indexer-k-new \
  --result-dir /tmp/myds-indexer-k-new/results \
  --jsonl /tmp/myds-indexer-k-new/results.jsonl \
  --out /tmp/myds-indexer-k-new/review.json
python3 tests/test_stream_bench.py
```

每次必须用新的输出目录。新构建器默认 Metal minver 15.0，与原构建脚本一致，可显式用 `--metal-minver 26.0` 复现本轮初始参数；CPU executable minver 保持 26.0。外部原生控制用 `--native-metallib /absolute/path/to/myds.metallib` 加入，不应删除失败控制后冒充通过。

保留的真实输入可只用 CPU 重新审计，不提交 GPU 工作；输出路径必须为新文件：

```sh
REAL=/tmp/myds-indexer-r2-real-local-20261025
python3 -B tools/review_indexer_k_capture.py \
  --capture-verified "$REAL/capture-verified.json" \
  --protocol "$REAL/real-component-protocol-v2.json" \
  --build-manifest "$REAL/replay-build-v4/build-manifest.json" \
  --metal-manifest "$REAL/full-build/source-identity.json" \
  --window-dir "$REAL/real-replay-v4" --input-root "$REAL" \
  --capture-root "$REAL/capture-window/raw" \
  --path-map "/private/tmp/myds-indexer-r2-20261025=$REAL" \
  --path-map "/tmp/myds-indexer-r2-20261025=$REAL" \
  --out "$REAL/new-independent-review.json"
```

首轮定义的后续顺序及当前状态：

1. 完整原生编译、部署制品身份与实际 provider 已核验，旧隔离身份差异已关闭；实际 exact 与候选的直接对照已完成。
2. 已完成三个指定层的真实 BF16 batch、完整 scores 与 ordered Top512；任意真实 F32 cache、其他 chunk/shape、业务分布和全链路行为仍未覆盖。
3. 唯一正式 M2 组件 cohort 已完成；识别推理任务检查无竞争，保留全部样本。六组中的一组漂移超门，因此不能进入第 4 步。下一实验先定位控制偏移，不能把当前失败记录重写为通过。
4. 全模型与 HTTP 未执行。将来通过前门后，仍需短、业务与长上下文，一次热身加至少三次正式样本，分开记录 prefill、TTFT、decode，比较完整 logits、tokens、输出和 cache/Engram/snapshot 状态。
5. 数值、漂移、收益和恢复门共同通过后才能决定启用。任何组件节约都不能替代全模型 >2% 的采纳判据。

本轮隔离构建了完整引擎，编译通过；现有 warning 保留，没有以编译通过替代 GPU 证据。两个 host overlap 合同 fixture 修复了落后的 scorer/kprof stub 接口并通过，它们只证明 host 合同。首轮 M2 组件筛选前后部署 binary、metallib 和配置 SHA 完全相同，8099 服务 PID 保持 52300；后续真实输入窗口开始和结束时均无识别的推理进程。两轮中本实验均未启动或停止服务。

完整 score/output buffers 保存在原机器的 `/private/tmp/myds-indexer-k-resident-m4-v1`、`/private/tmp/myds-indexer-k-screen-20261025`、`/private/tmp/myds-indexer-k-samebuild-screen-20261025` 和 `/private/tmp/myds-indexer-k-min15-screen-20261025`，未把这些大文件随源码发布。仓库保存小型原始 JSONL、独立复核结果和[最终文件身份](benchmarks/2026-10-25-performance-reopen/final-source-identity.json)，合成输入可用上述工具重新生成。
