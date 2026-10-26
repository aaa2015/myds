# 私有 indexer runtime 合同与 Q_B/RoPE v1 拒收补充

2026-10-25。本页只裁定已冻结源码、CPU 合同与已有 Q_B 数值证据。没有新增 GPU、远端操作或服务启动。K-resident 的 matched 正式组件门已通过并获准实施私有 runtime；完整模型/HTTP 的最终采纳不能从本页推导。root 已报告 Q0 六次生成完成，本页不将“生成完成”替代正在进行的独立 reader 裁定。

## K-resident 私有接入：source/CPU 合同通过

[本次 source audit](private-runtime-source-contract-audit.json)重新全量核验私有源码 **548 个文件**，与冻结 build manifest 完全一致；同时直接核对本地镜像中的 M2 target/gold 两个 executable 与两个 library SHA。完整 [seed manifest](private-runtime-seed-manifest.json)、[build manifest](private-runtime-build-manifest.json)、[目标 CPU build manifest](private-runtime-target-build-manifest.json)与[目标 source check](private-runtime-target-source-check.json)保留原始 bytes。

- Seed 为当前 dirty 工作树的 545 个构建源码文件导出，不声称等价 clean HEAD。完整 seed inventory=`c39bd664eb6dd35316a092af3b63771da799fc65fc37dd3eb772c9f4270fb0af`。
- 548 文件 final inventory=`7791166d7a9a6ef29eaa2336c665919ee041ec1eae47e32027d6a656b44adec7`。55 份 matched R3 的 `src/include/third_party` 与 seed 逐 bytes 相同。
- 接入只改 GPU C、唯一 prefill publisher C call、GPU header，加 adapter 与两份 CPU fixtures。原 `indexscores_chunk` 及两个 public chunk wrapper 的**完整函数文本逐 bytes 不变**；所有 Metal source/backend/注册列表均未修改。
- [核心 patch](private-runtime-runtime-core.patch)和[含实际源 fixtures 的完整 patch](private-runtime-runtime-with-host-fixtures.patch)均可复核。[正反向 byte proof](private-runtime-patch-byteproof.json)在第三个临时副本证明：正向得到全部 548 文件，反向恢复全部 545 seed 文件并移除新增文件。

本次明确授权的窄作用域如下；它收紧了[原只读计划](private-runtime-original-plan.md)，没有重写计划历史：

| 条件 | 运行时合同 |
|---|---|
| 模式 | `MYDS_INDEXER_K_RESIDENT` absent/0/1，默认0；其他值为−1并拒绝；配置载入后首次 getter 通过 pthread_once 固定 |
| 调用点 | 新 ordinary-prefill scoped API，唯一生产调用传 `!myds_trace_active()`；旧 public APIs/decode 不进入候选 |
| Shape | heads32、dim128、ratio2、`512<n_comp<=16384`、`8<=n_tokens<=2048`、`end<=32768`、`n_comp=end/2` |
| 上下文 | active batch、无 exclusive kprof、key window=0、SGBAR absent/0 |
| 资源 | 当前 scratch max/key/budget；完整 native range/alignment；score writer 与 Q/weights/cache 的完整 reader spans 不重叠 |
| PSO | F32/BF16 两个具名候选均预检 SIMD32/128threads/21504B+static TG memory；不跨 library 生命周期缓存成功 readiness |
| 不适用 | 小表、大表、ratio1及其他合法非作用域请求继续原 selector，不计 candidate fallback |
| 失败 | eligible 参数/资源/PSO/dispatch 失败立即0→publisher−1；不重做原地 Q RoPE/quant，不试 exact/direct/scalar fallback |

六项 host atomic 为 `requests, encoded, fallbacks, failures, f32_dispatches, bf16_dispatches`。requests 包含 off/ineligible scoped 调用，因此只能要求 `requests>=encoded`。encoded 在 active-batch dispatch 被接受后增加，**不是 GPU completion**；普通 ineligible/reference 不记 fallback。正常 fallback/failure均0，`encoded=f32+bf16`。adapter逐字段真实映射既有 fullmodel driver；mode1必须首 callback 前 encoded>0、两个 symbol 精确且非空，final encoded/dtype计数等于首 callback，decode编码增量0。不能用短请求不适用去放宽采用门。

两 symbol 固定为 `kernel_myds_indexer_scores_k_resident_f32` / `kernel_myds_indexer_scores_k_resident_bf16`。执行的 CPU 验证是：

- [实际 selector fixture](private-runtime-selector-host-fixture-results.json)：48个mode1控制，17个独立进程的 parser/default/window/非法实际调用检查。
- [实际 publisher fixture](private-runtime-publisher-host-fixture.stdout.jsonl)：scorer故障传播−1、trace scope正确、无重复RoPE和后续消费者。
- [原 fullmodel driver 12项 CPU tests](private-runtime-fullmodel-driver-cpu.stderr.log)、[完整 CPU build 日志](private-runtime-cpu-build.stdout.log)、[保留的7条seed warnings](private-runtime-cpu-build.stderr.log)。
- [mode0/1 host preflight](private-runtime-preflight-results.json)不进入 model/device 创建；[普通预处理](private-runtime-normal-preprocessor-review.json)没有 capture observer标识。不能把这些 CPU 检查写成 GPU 数值、PSO实际可用或性能证明。

| 制品 | 完整 SHA256 |
|---|---|
| 授权的 matched 正式 review | `7a50c7122d8d70905b7f77ead07b52e59bea87acf4af199ee109fecb4cc20508` |
| 私有 runtime build manifest | `e21a9a2dfd27fb9c771bf3d6f9c49f534a9689db0c3fa7e66bdcaf461e7b0896` |
| Local/M2 target executable | `e8cc04b81300bec6a8b8c671f1477723dfbaf67581edfbb7338094f2d48d8afa` |
| Candidate full-TU library | `9df7ed2a5212d54d4475480bf6388d2f66ce24c93bb3a8b003f2b789806d6982` |
| Actual deployed Gold library | `3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9` |

候选 full-TU library 的原 prefix=`bc6603c97ec24af7597f6b10c7ed2a7bfe1344379733a0a4df780250a13f3224`，完整原 prefix/append 与获准组件 build逐 bytes一致。此源码审计不调用加载器，不声称库已在某个完整模型样本实际加载；该 witness属于 Q/P/HTTP 的冻结窗口。组件通过和私有默认关闭接入均不等于生产采纳。actual exact→TILED/candidate 已知存在少量 score 差异，后续必须核完整 ordered TopK、mask/filter、attention、logits/tokens与active/published状态；不能仅凭 TILED→candidate scores0ULP通过宣称全图等价。

## Q_B→forward RoPE 冻结 v1：M4与M2均拒收

该 probe 只考虑32个非index-source decode层的 full32768×1280 Q4 NR2/NSG4/NB epilogue。保留两次 uniform simd_sum和第一层BF16，再执行原前向RoPE/第二层BF16。理论减少32dispatch和1MiB/token tail往返仅是工作量；由于数值门失败，两设备均没有性能计时、runtime接入或生产采纳。

| 设备 | 完整 cases / CB | Actual Gold vs full-TU native | Candidate vs本设备Gold | Timing |
|---|---:|---|---|---:|
| Apple M4 | 114 / 342 | 114/114全位一致 | 111/114全位一致，3个单值差异 | 0 |
| Apple M2 Ultra | 114 / 342 | 114/114全位一致 | 108/114全位一致，6个单值差异 | 0 |

[M4独立完整审计](qb-rope-independent-review.json)与[M2独立完整审计](qb-rope-M2-independent-review.json)分别核对1715文件、11,206,656个F32输出、914份64B guard和115份post inputs。每设备的inputs/guards/finite/BF16格式均通过。M2的**344份before payload**与M4及host-only逐bytes/SHA全同，包含完整weights、inputs、mv112和rope152参数；不能只靠同generator/source SHA断言输入相同。

差异全部位于rotated `im`，每项为1 BF16 ULP，即65536 F32 ULP；无unrotated差异。M4 cases=`{8,92,99}`，M2 cases=`{8,29,38,56,92,105}`；共同2项，M2新增4项而M4独有case99。M2不是M4失败集合的超集。

| M2 case | layer / position / head-dim | Gold bits → Candidate bits | 绝对误差 |
|---|---|---|---:|
| 8 | 3 / 327679 / 63-505 | `3fe10000 → 3fe00000` | .0078125 |
| 29 | 11 / 327679 / 31-483 | `c1f30000 → c1f20000` | .125 |
| 38 | 15 / 327679 / 1-483 | `bdb90000 → bdba0000` | .00048828125 |
| 56 | 22 / 327679 / 4-479 | `3eef0000 → 3eee0000` | .001953125 |
| 92 | 38 / 327679 / 42-485 | `41b00000 → 41af0000` | .125 |
| 105 | 3 / 16384 / 28-465 | `3dfb0000 → 3dfa0000` | .00048828125 |

M2六项均compressed，最大相对误差0.568182%；M4的case99为uncompressed/pos16384，最大相对误差0.746269%。同Gold库和同输入的M2/M4原生Gold输出本身还有6个单值跨设备差异，probe-native相同；这不推翻**每设备内部**的Gold/native身份通过，也说明不能用M4 Gold取代M2 oracle。

[actual Gold RoPE AIR](qb-rope-ir-actual_gold.kernel_dsv41_rope.ll)与[probe原RoPE AIR](qb-rope-ir-probe.kernel_dsv41_rope.ll)逐bytes相同；原Q_B AIR仅有TU constructor文件名符号差异，标准化这一处后完全一致。[候选AIR](qb-rope-ir-probe.kernel_myds_probe_qb_rope_nb_bf16.ll)的im加法operand顺序与Gold不同，均带fast标记。独立有限F32加法交换本身不解释差异；FMA/lowering是假设，尚无机器级证据或独立第一层BF16 tail捕获证明确定修复。没有改FMA、源码或阈值来寻找通过。

M2运行前的[协议](qb-rope-M2-target-protocol.json)、[execution plan](qb-rope-M2-execution-plan.json)、[window](qb-rope-M2-window.json)、[完整stdout](qb-rope-M2.jsonl)及[artifact inventory](qb-rope-M2-artifact-manifest.json)原样保存。所有小证据副本的原始路径、size/SHA与byte一致性见独立[copy manifest](decode-source-audit-evidence-copies.json)。大raw、完整源码archive和compiled binary/library保留在原`/tmp`阶段目录；本页未复制这些payload，也未更改主报告、Makefile或总evidence-index。
