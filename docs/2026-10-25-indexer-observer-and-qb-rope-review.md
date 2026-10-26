# Indexer 计时观察者与 Q_B / forward RoPE 续查

2026-10-25 至26日。**K-resident 已完成私有 runtime、两轮全模型资格和唯一24代正式性能窗口：两条长请求的 prefill 保守节约为1.591%和1.514%，未达预声明严格 >2% 门；decode 无回退门也失败，因此不采纳。** 全部数值/状态/实际采用门通过，保留所有原始样本，不追加重试。生产制品、配置与工作树 C/header/Metal 未纳入该实现。

此前保留旧观察方式的同 PSO 对照复现了1.31%–1.84%的 A1/A2 差异；显式匹配输出复用距离的新测量随后通过六组组件门。相对 actual exact 的组件保守节约40.17%–40.49%、相对 TILED 的12.76%–13.02%，不能当作全模型收益。另一个[删除无用 dummy-Q RoPE 的 decode 候选](2026-10-26-decode-kv-only-finish-review.md)独立推进，不与 K-resident 联合开启。

所有先前结论保留：[R2 正式组件六组联合门失败](2026-10-25-prefill-decode-performance-review.md)，本轮第一批 468 条诊断也未通过预声明的工程修正规则。它们不由后续记录重判。Q_B / forward RoPE 冻结 v1 在 M4 和实际 M2 目标均未过逐位门，未进行性能计时或 runtime 接入。

## 原计时序列的确定事实

[完整时间取证](benchmarks/2026-10-25-indexer-observer-and-qb-rope/old-cohort-timing-forensics.json)重新分析了原 180 条记录，其中 18 热身、162 正式。六组中共 **54/54 个正式 A1/A2 配对为 A2 更快**；配对百分比中位数 −1.45954%，范围 −3.40368% 至 −0.24201%。这个跨层、跨 control 的顺序信号不能只解释成 layer2 的一次随机越门点，也不能据相关、重复样本计算独立性 p 值。

原固定序列为 `[A,B,A][A,B,A]…`。A1 的直接前驱是同一个 A，且立刻复用刚被 A2 写过的控制输出；A2 的前驱是 B，控制输出在中间隔了一次 B。**arm、前驱和输出 buffer 复用距离混淆。** GPU timestamp 之间的间隙中位数 6.583 ms，包含 CPU poison、验证、SHA、JSON、编码提交及其他等待，不能直接称为测得的纯 CPU 时间。保存的 GPU spans 之和占首尾 timestamp 范围 46.61%，只是这个重验证 harness 的时间口径，不是生产 GPU 利用率或 occupancy。

每次原 scorer CB 后，harness 直接读取 shared output，完整比较 actual/TILED、扫描 finite/causal/poison，再做 memcmp、SHA、JSON/flush。这些工作不在保存的 GPU span 内，但可能改变下一次 dispatch 的前置状态。其实现见 [原 replay](../tools/bench_indexer_k_capture.m)，本轮没有修改该文件。

## 第一批有限诊断：数值通过，工程规则失败

[schema3 协议](benchmarks/2026-10-25-indexer-observer-and-qb-rope/diagnostic-protocol-v3.json)在任何诊断 GPU 提交前冻结。它包括三个同 provider sham、TILED/candidate 的 legacy 与匹配前驱对照，以及延后验证和不同输出槽的描述对照，共九个 phase。一个 warmup round 加十二个分析 round，六次正序、六次反序；每个 phase 有 conditioner、A1、B、A2 四次 dispatch，共 **468 条记录**。

新工具每 CB 后复制完整 scores / guards 到快照，再立即验证或在四步结束后验证，JSON 和 raw 在组末写入。因此它是一个受控的新观察者，不能冒充旧 binary 的原样重放。[builder](../tools/build_indexer_stability.py)冻结原 helper 前缀而不改旧文件；[GPU 工具](../tools/bench_indexer_stability.m)与[独立 CPU reader](../tools/review_indexer_stability.py)分别实现数据采集与裁定。

| 运行前冻结的规则 | 实测 | 结果 |
|---|---:|---|
| exact sham 三位置 span ≤0.5% | 0.0290307% | 通过 |
| TILED sham 三位置 span ≤0.5% | 0.1623387% | 通过 |
| candidate sham 三位置 span ≤0.5% | 0.4583374% | 通过 |
| legacy inline 复现 A2 加速 ≥0.5% | 0.0031910% | **失败** |
| balanced inline A1/A2 绝对差 ≤0.5% | 0.1904702% | 通过 |
| 配对绝对偏移减少量中位数 ≥0.5 pp | 0.120103 pp | **失败** |
| 减少量严格为正的配对 ≥10/12 | 7/12 | **失败** |

[独立完整复核](benchmarks/2026-10-25-indexer-observer-and-qb-rope/diagnostic-independent-review.json)核验 966 个 raw、3,732,500,864 个逻辑字节，468 份完整输出及 guards、输入前后见证、源码、实际库、顺序和进程窗口。相同 payload 以透明硬链接保存，只有九个独立物理 payload；逻辑字节数不代表物理 I/O。全部数值通过，最终为 `diagnostic_rule_not_satisfied_no_next_stage`。没有据某个描述 phase 的好读数救回决定门，也没有追加样本。

该结果不支持直接采用第一版前驱修正方案。快照 / 组末输出的新观察者未复现旧偏移，不能证明旧偏移不存在。

## 恢复旧观察方式的独立同 PSO 对照

随后冻结[新的 literal observer 协议](benchmarks/2026-10-25-indexer-observer-and-qb-rope/observer-protocol.json)，四个新进程固定为 **R→S→S→R**。R 使用真实 candidate，S 仅将 phase2 的 TILED control / 中间 B 改为与 A 相同的 TILED PSO object；B 的 logical provider、output2、72 字节 args、grid 与共享内存保留。actual-exact 组中 B 仍为 candidate，不能将其称作 null。

[独立 builder](../tools/build_indexer_observer_probe.py)只作六项可逆变换，保留原三层初始化、PSO 创建、phase1、TopK、分配释放、direct shared-output 校验以及逐 CB SHA/JSON。R/S 同 binary；增加了 CLI、诊断 metadata 和两个 timing 绑定字段，因此是“最小改动的旧观察者对照”，并非声称与 R2 executable 逐字节相同。

四次运行各 180 条，共 **720 条记录＝72 热身＋648 正式**。全部 numerical、TopK、输入、canary 和制品窗口检查通过。下表是 TILED 组，保守 B 差值仍按原公式计算；S 的 B 是同一个 kernel，其数值不是优化收益。

| 运行 | layer2 A2/A1 | layer8 A2/A1 | layer14 A2/A1 | 中间 B 的保守耗时差 |
|---|---:|---:|---:|---:|
| R0，真实 candidate | −1.8653% | −1.3166% | −2.0748% | 节约 12.6844%–13.3357% |
| S1，同 TILED PSO | −1.5323% | −1.5392% | −1.4041% | −0.3069% 至 +0.0016% |
| S2，同 TILED PSO | −1.8413% | −1.5093% | −1.3093% | −0.0904% 至 +0.5257% |
| R3，真实 candidate | −1.6645% | −1.2746% | −1.5353% | 节约 12.9219%–13.3112% |

S 的六个 null 组中 **54/54 个正式配对仍是 A2 更快**。因此该偏移能够在同 PSO、同输入、同算术下出现，不能全归给 candidate 的不同计算。它缩小了观察者 / 输出槽 / 顺序的排查范围，没有单独隔离 JSON、CPU shared-memory 扫描、分配、缓存或频率中的某一个因素。R0 的 layer14 漂移也越过原 2% 线，完整保留；本次四个进程全部是诊断，不从 R3 的读数挑选一个作为新正式结果。

完整独立复核见本目录最终证据索引。后续工程假设是显式匹配控制输出的复用距离，并先验证同 PSO null 是否稳定；这不改写上述任何 cohort 的判定。

## 匹配输出复用距离的新组件门

新观察者保留旧 replay 的 direct shared-output 完整验证、SHA 与逐 CB JSON，在每个 A1/B/A2 前增加一次 conditioner C：**C→A1→B→A2**。C 与 B 使用同一实际 PSO、args 和 output2，使两个控制臂的直接前驱一致、控制输出复用距离均为两次 CB。原始 replay 文件没有修改；[新 builder](../tools/build_indexer_matched_replay.py)的十一项变换可逆恢复冻结的原源码。

[新协议](benchmarks/2026-10-25-indexer-observer-and-qb-rope/matched-protocol.json)在数据产生前冻结，先执行唯一 S 同 PSO 校准，三层 null span 均需 ≤0.5%；只有独立 reader 通过后才启动唯一新进程 R。S 的三层 span 为 0.143839%、0.345076%、0.406536%，全部通过。R 绑定 S 的复核 SHA，使用相同 binary、源码、reader 和输入。每阶段 240 条记录，包括 60 conditioner、18 热身臂和 162 正式臂。

| 层 | 控制 | 保守耗时节约 | A2/A1 漂移 |
|---|---|---:|---:|
| 2 | actual exact | 40.19911% | −0.008761% |
| 2 | TILED | 12.75707% | +0.004800% |
| 8 | actual exact | 40.17019% | +0.008761% |
| 8 | TILED | 13.01557% | +0.003190% |
| 14 | actual exact | 40.49098% | −0.011498% |
| 14 | TILED | 12.85853% | −0.069345% |

六组全部满足严格 >2% 的保守节约和绝对控制漂移 ≤2%；完整 scores、有序 Top512、18 份输入见证及每阶段 90 个 raw / 109,811,328 字节通过各自的数值合同。具体是 candidate 与 TILED 全 scores 逐位一致；layer2 的 actual exact 与它们仍有原先四个 score cell 差异，完整有序 Top512 一致，不能泛读成三 provider 全 scores 逐位相同。[校准独立复核](benchmarks/2026-10-25-indexer-observer-and-qb-rope/matched-calibration-review.json)、[正式独立复核](benchmarks/2026-10-25-indexer-observer-and-qb-rope/matched-formal-review.json)。这支持进入另行冻结的全模型资格门，没有把 R2 或第一批诊断改判通过，也没有确定唯一的硬件偏移原因。

## 私有 runtime 与全模型验证边界

候选只接入 `/private/tmp/myds-indexer-k-runtime-20261025/source` 的独立源码快照。`MYDS_INDEXER_K_RESIDENT` 缺席或为 0 时关闭，严格只接受 0/1。启用范围限定 ordinary nontrace prefill、ratio2、512 < ncomp ≤16384、8≤ntokens≤2048、end≤32768，以及原生范围、别名、预算、PSO 和 threadgroup 几何门。实际 eligible dispatch 失败立即报错，不以 native 或 CPU 回退掩盖；ineligible 请求继续原路径。

新增 host counters 记录请求、实际编码、F32/BF16 和失败；这是 CPU 编码见证，不是 GPU 完成证明。decode 的 encoded 增量必须为零。私有源码控制与实际失败发布路径的 CPU 检查通过；真实目标编译成功。生产 checkout 的 C/header/Metal 清单及 canonical 服务没有纳入此实现。全模型 G_lib/off/on、长请求、正式 A1/B/A2、HTTP、取消恢复均按各自阶段独立记录，组件门不能代替后续验收。

### Q0：7,540-token 全模型资格门通过

固定 G_lib→off→on，各一 warmup 加一 verify，共六次全模型生成，每次独立新 context、完整 prefill、64 输出 token。G_lib 使用相同私有 C binary 配部署 metallib，是 deployed-library bridge，不是 deployed executable 或 HTTP bridge。配置原有 62 项保持，额外只有候选开关与两项证据目录；prefix cache / PLD 等生产设置没有为测速改掉。

六代的完整 129,280 维最终 logits 均 finite，169 个状态 payload 加一份 logits、每代 56,370,503 字节，合计 **338,223,018 raw payload 字节全部逐字节一致**。覆盖输出 bytes / token IDs、live cache、Engram / RNG、prefix / anchor 和已发布 snapshot 数组；不含各步中间 logits、snapshot 历史 Engram tail、磁盘 payload 或 HTTP wire。G_lib/off 的候选 encoded 为零；on 每代请求 77、实际编码 27，全部 BF16、在首次 callback 前完成编码，decode 六项 counter 增量均为零，无失败或回退。

原 Q0 第一次启动在 child 创建前因备份 executable 权限拒绝，未执行模型或 GPU；只恢复 executable mode 后以新目录和仅路径不同的 v2 协议运行，旧窗口留存。原冻结 reader 随后因 macOS `/tmp` 与 `/private/tmp` 的字符串差异拒绝，保留原源码和失败报告；[v2 reader](../tools/review_indexer_fullmodel_v2.py)仅接受这个已知别名，仍要求解析到冻结的同一库且完整 SHA 一致，21 个 CPU 测试通过。没有为修复 reader 重新运行 GPU。[原 reader 拒绝](benchmarks/2026-10-25-indexer-observer-and-qb-rope/fullmodel-q0-v2-review.json)、[修复后独立完整复核](benchmarks/2026-10-25-indexer-observer-and-qb-rope/fullmodel-q0-v2-canonical-review.json)、[第二份 raw-only 比较](benchmarks/2026-10-25-indexer-observer-and-qb-rope/q0-root-raw-byte-review.json)。

Q0 通过只允许准备独立冻结的长请求资格协议，不是性能或生产采纳结论。此处的请求由业务记录生成器构造，不代表真实生产流量分布。

### Qlong：双长请求资格门通过

[独立协议](benchmarks/2026-10-25-indexer-observer-and-qb-rope/fullmodel-qlong-protocol.json)绑定 Q0 修复后实际报告，固定 business96 29,609 tokens 与 code38404 38,404 tokens，各 G_lib→off→on、warmup1+verify1、64 输出，共 **12 次生成**。沿用同一私有 binary、两份库、62 项配置与 fresh context，不开 profiler、capture 或 trace。业务请求由生成器构造，代码请求含重复工程片段，都不代表生产流量分布。

[冻结 Qlong reader](../tools/review_indexer_fullmodel_long.py)完整复核通过：2,064 个 inventory 文件、1,694,479,263 字节；其中实际 logits/状态 payload 合计 **1,694,041,062 字节**。两负载各自三 provider 的输出 bytes、token IDs、完整最终 logits 与 schema2 状态逐字节一致，全部最终 logits finite。business96 的 on 每代 519 requests / 342 BF16 encodings，code38404 为 782 / 417；编码全部发生在首次 callback 前，decode 六项计数均为零，off/G_lib encoded 均为零。代码负载越过 32K 窄门后继续走原路径，未以 fallback 计数掩盖实际采用范围。

来源、输入、实际加载库、配置、source/target 和完整窗口门均通过。[联合正式复核](benchmarks/2026-10-25-indexer-observer-and-qb-rope/fullmodel-qlong-review.json)，SHA `eb59b34c86c39fc5cb91d793c3c0dc95fc04254c150b8e3ebb824847d9099262`；第二份 raw-only 比较分别见 [business96](benchmarks/2026-10-25-indexer-observer-and-qb-rope/qlong-business96-root-raw-review.json) 与 [code38404](benchmarks/2026-10-25-indexer-observer-and-qb-rope/qlong-code38404-root-raw-review.json)。Qlong 通过允许另行准备 P，不代表耗时门已通过。

### P：双长请求正式性能门

[P 数值门](benchmarks/2026-10-25-indexer-observer-and-qb-rope/fullmodel-p-predeclared-gates.json)在 Qlong 数据前冻结；[P 协议](benchmarks/2026-10-25-indexer-observer-and-qb-rope/fullmodel-p-protocol.json)与[独立 reader](../tools/review_indexer_fullmodel_performance.py)在 P 数据前冻结，并在 prelaunch plan 锁定实际 Qlong 报告 SHA。每负载 A1(off)→B(on)→A2(off)，各 warmup1+formal3，共 24 次生成，全部 raw 保留，不修剪、不替换样本。

每臂取正式 run1/2/3 的中位数，prefill 节约 `100*(1-B/min(A1,A2))` 严格 >2%；prefill、TTFT 和 decode 各自 A2/A1 绝对漂移 ≤1%，TTFT / decode 相对较快 A 的回退 ≤0.5%。reader 从原 JSON 十进制数构造有理数，不用 epsilon 或百分比预舍入处理边界。完整数值/状态/采用门先于性能裁定。

唯一24代窗口已完整结束，六次 warmup 和18次正式均保留。独立 reader 审过4,116个 inventory 文件、3,388,834,915字节，全部数值/状态/输出、来源、实际库、完整65项有效配置（原62 + mode + 两个sink）、符号/计数和窗口门通过。on 的每代实际编码分别为342和417 BF16，全部在 first callback 前；decode 六项 counter 均为零，off 两臂 encoded 为零。

| 请求 | A1 prefill s | B prefill s | A2 prefill s | 保守节约 | 控制漂移 |
|---|---:|---:|---:|---:|---:|
| business96 /29,609 tokens |65.262557|64.224137|65.287753|**1.591142%**|+0.038607%|
| code38404 /38,404 tokens |84.136947|82.818263|84.091642|**1.514275%**|−0.053847%|

prefill 的两个控制漂移通过，两个节约均未达 >2%。TTFT 的保守节约为1.595729% /1.519979%，其无回退门通过。decode 的正式中位数及门如下，单位 ms/token：

| 请求 | A1 | B | A2 | B 对较快 A 的回退 | 控制漂移 |
|---|---:|---:|---:|---:|---:|
| business96 |26.371635|26.191937|25.915095|**+1.068263%**|**−1.731177%**|
| code38404 |26.555603|26.766159|26.648619|**+0.792886%**|+0.350268%|

两个 decode 无回退门失败，business 控制漂移也越过1%线。候选没有在 decode dispatch，且本轮没有单独隔离频率、缓存或其他顺序效应，不能仅据这些读数断言某个唯一的 decode 回退原因。联合结果为 `P_performance_gate_failed_no_retry`，`native_fullmodel_gate_passed=false`，没有进入 HTTP、取消恢复或生产采纳。

[完整独立正式结果](benchmarks/2026-10-25-indexer-observer-and-qb-rope/fullmodel-p-review.json)，SHA `5fc89462d60ceb08d58e684ea380affe85d049b950b134a0a05976195bd86c64`。原始十进制样本、精确有理数、每臂窗口与全部 raw 都保留；没有删掉 A1 的慢点或 A2 的快点，没有放宽门槛，也没有重跑该 cohort。K-resident 保持私有 default-off。

## Decode：Q_B 与 forward RoPE 融合 v1 拒收

本轮先排除重复工作：late-layer post-score filter 已从当前默认 scoped/head-slots 主路径消失，剩余调用在其他 legacy / reference 路径；Output_A 固定 wrapper 已有两负载全模型约 0.36% / 0.40% 的旧结果，没有通过 >2% 采纳门。没有把它们重新列作新机会。

新的窄实验针对 32 个非 index-source decode 层的 full Q_B→forward RoPE。候选在 NR2 epilogue 内保留两次 uniform `simd_sum`，先做原 Q_B 的 BF16 舍入，再对每头尾部 64 维执行原 precise trig / 仿射表达式 / 第二次 BF16 舍入。理论可省 32 次 dispatch 和 1 MiB/token 的尾部读写；这是源代码工作量，不能转换为已测收益。稀疏 lane0 三角函数也可能延长原 Q_B cohort。

[构建器](../tools/build_qb_rope_probe.py)使用完整生产 Metal 前缀，只追加私有符号，并记录可逆 Q4 body 变换和冻结的原 host 频率初始化。[harness](../tools/bench_qb_rope_probe.m)覆盖 32 层的 pos0 / 1 / 327679，加若干边界位置，共 114 case；每 case 比较实际 deployed gold、完整 probe 中原 Q_B＋RoPE、候选三份 32,768 值完整输出。

M4 唯一 screen 的 342 个 CB 均完成；所有输入、finite/BF16 格式和 guards 通过。实际 gold 与完整 TU 原生对照 114/114 全位一致，候选 111/114 全位一致，三组各有一个 rotated `im` 值不同：

| case / 位置 | Gold → candidate | 绝对差 |
|---|---|---:|
| 8，L3 / pos327679 / head63 dim505 | `0x3fe10000` 1.7578125 → `0x3fe00000` 1.75 | 0.0078125 |
| 92，L38 / pos327679 / head42 dim485 | `0x41b00000` 22 → `0x41af0000` 21.875 | 0.125 |
| 99，L0 / pos16384 / head11 dim483 | `0xbf860000` −1.046875 → `0xbf870000` −1.0546875 | 0.0078125 |

三者均为一个 BF16 ULP，即 65,536 个 F32 ULP，最大相对误差 0.746269%。第三个使用 uncompressed 频率表，不能说成全部是最大位置 / compressed。无 unrotated 差异；[完整 raw 独立审计](benchmarks/2026-10-25-indexer-observer-and-qb-rope/qb-rope-independent-review.json)核过 1,715 文件、11,206,656 个输出 F32 和全部输入 / guards。数值门失败后 timing records 为零。

实际 AIR 的原生身份审计通过。候选 `im` 末式的加法顺序与原核不同，二者均带 fast 标记；加法交换本身不能证明这些误差的原因，也没有机器级证据证明某个 FMA lowering 是唯一根因。因此保持 v1 拒收，没有盲改 FMA 后重复寻找通过。

同一冻结 v1 又在实际 M2 上做了一次 correctness-only 目标检查，没有调整源码、参数或生成器，也没有开启计时。原生对照仍为 114/114 通过，候选 cases 8、29、38、56、92、105 各有一个 rotated 值未过门，均为一个 BF16 ULP，最大绝对差 0.125、相对差 0.568182%；M4 的三项失败独立保留。[M2 完整 raw 独立复核](benchmarks/2026-10-25-indexer-observer-and-qb-rope/qb-rope-M2-independent-review.json)。

两设备与 host-only 的 344 份 before-payload 逐字节相同；deployed native gold 自身在两设备间另有六个单值差异（cases 29、38、44、56、59、108），probe native 也有相同的跨设备差异集合。因此每台设备都用自身 deployed gold 作为 oracle，不能用 M4 gold 审判 M2 candidate。目标差异没有覆盖任何旧失败，也没有证明某个 FMA lowering 是唯一根因。

Q_B / RoPE v1 未接入 runtime；K-resident 只有上述私有接入。完整来源、协议、计时、窗口与独立复核保存在 [evidence-index.json](benchmarks/2026-10-25-indexer-observer-and-qb-rope/evidence-index.json)；大 raw 仍保留在原实验目录和本机归档。

完整M2 raw、源码/bin/metallib及M4补充证据另存到`macstudio:/Users/fangqingyuan/Benchmarks/myds/2026-10-26-metal-performance/`，两份成功包均完成逐payload/硬链接校验与完整archive SHA验证。原始证据未删改，GGUF模型payload未打包。[归档清单与只读复核说明](benchmarks/2026-10-25-indexer-observer-and-qb-rope/durable-archive.md)。
