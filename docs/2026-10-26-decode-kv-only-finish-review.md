# Decode KV-only finish：删除占位 Q 的 RoPE

2026-10-26。**真实默认调用链中确有无用工作，私有实现通过两设备组件和全模型数值/采用门，但唯一24代正式窗口的decode保守节约仅−0.088% /+0.214%，未达冻结 >2% 门，不采纳。** 所有控制漂移和secondary门通过；该负结果不是因本次控制漂移超限，也没有重测。候选保持私有default-off，未进入HTTP/生产。它与已拒收的 Q_B→forward RoPE 融合是两个不同候选：此处保留真实 Q_B 和真实 Q 的原 RoPE，只删除占位 scratch 的计算。

## 默认调用链与窄范围

`src/myds_v41_moe.c` 的 KV reuse 路径在约 2441 行将 attention `partial_scratch` 作为 dummy Q 传给 `myds_v41_gpu_qkv_finish()`。原 `kernel_dsv41_qkv_finish` 的前 64 个 TG 仍为这块 scratch 执行 Q-tail RoPE，后 16 个 TG 才完成实际 KV 的 RoPE、BF16/FP8 舍入及 window 写入。真实 `ctx->q` 随后由原 Q_B GEMV 生成并单独做原 RoPE。

初版仅考虑 32 个 non-index-source 层、普通无 trace 的 RF/register、FLASH_DEC=0 路径。这条 attention 后续分支根本不绑定或读取 partial，因此删除 dummy 写入不依赖推测“稍后会覆盖”。FLASH split / generic vec 路径另有 partial 生产再归约，初版全部排除；约 2560 行的 index-source 调用也保留。原 partial 分配、所有资源列表和屏障均保留，不能据资源列表直接声称并发 overlap。

按当前 32 层的源代码工作量，每 token 可删除 65,536 对无用旋转、131,072 次 precise trig 调用与 1 MiB 逻辑读写；dispatch 数没有减少，Q_B 可能仍决定同 cohort 的完成时间。这些数量不是实测 GPU 带宽或全模型收益。[完整调用链、失败路径和复制边界](benchmarks/2026-10-26-kv-only-finish/source-contract-and-reproduction.md)。

## 冻结的组件与 M4 结果

新 symbol `kernel_myds_probe_kv_only_finish` 使用完整生产 Metal 前缀，仅将原函数的 `group<64` 分支改为 early return；其余 KV 分支逐字节保持。原 80 TG×32 lane、args136、Q/KV/window 的 binding 与三资源 join 全部保留。更早的 16-TG 草稿只完成 CPU 编译，没有提交 GPU，也不是本次测试制品。

[correctness-only 协议](benchmarks/2026-10-26-kv-only-finish/component-correctness-protocol-v1.json)冻结 114 cases、部署 gold / 完整 TU 原生 / KV-only 三 provider，共 **342 CB**，不开 timing。32 层各测 pos0、1、327679，并加窗口 wrap、频率边界等 18 cases。输入含零、极小/大 BF16 KV，以及 dummy Q 的 NaN/Inf/非 BF16 特殊值，后者有意不要求 finite。

M4 唯一窗口全部完成，51 份冻结输入和制品的 SHA 在前后相同。独立重读 **3,535 个 raw 文件、180,401,269 字节**；1,026 份完整输出和 2,052 份 guard 满足：

- 部署 gold 与完整 TU native 的完整 dummy Q、KV、window 逐字节一致。
- candidate 的完整 512 值 KV 和 65,536 值 window 与 gold 逐字节一致，22,588,416 个 KV/window F32 全部 finite 且 BF16 可表示。
- candidate 的完整 128 KiB dummy Q 等于 before；它与 native 不同属于删除无用写入的预期行为。
- 所有非当前 window 槽等于 before，当前槽等于完整 KV；全部两侧 guard 为 A5。
- 342 个 completion ordinal/status、两次实际库加载、三个实际 PSO 和原 geometry 通过，无 fallback、无 timing。

原独立 reader 因 `finite_bf16_required` 的声明类型拒收：producer 的 `@(r!=0)` 是 C int，实际 JSON 为 0/1，reader 误按 bool 重算。原 reader、拒收报告、协议、raw 和窗口全部保留。v2 **仅**将这个声明字段改为严格 int0/1，其他 bool / 数值 / 来源 / 完成门不变；17 个 CPU 测试含独立手写的真实 JSON 字段及 bool 冒充反例，通过后在同一 raw 上复核，没有重跑 GPU。

[原 reader 拒收](benchmarks/2026-10-26-kv-only-finish/M4.original-reader-failure.json)、[v2 独立正式复核](benchmarks/2026-10-26-kv-only-finish/M4.independent-review-v2.json)、[第二份 raw-only 比较](benchmarks/2026-10-26-kv-only-finish/M4.root-raw-review.json)。正式复核 SHA 为 `2339cda8330db430efd0e905a7a359a35cbea92c4373f01f5f0c4eb646f37660`。

## 私有 runtime 和剩余门

[接入设计](benchmarks/2026-10-26-kv-only-finish/private-runtime-plan.md)仅在新独立源码快照中实施，原 K-resident 的 548 文件快照及正在运行的 Q/P 制品保持冻结。严格 default-off 的 `MYDS_DECODE_KV_ONLY_FINISH` 只触发上述 32 层和 `pos<327680`；eligible 的缺 PSO / preflight / dispatch 失败必须立即报错，不静默 reference 重跑。正常 ineligible 走原路径。

新的 driver 必须如实使用 decode subject、单 selected symbol 与请求/eligible/encoded/reference/fallback/failure 六计数。64 个正常输出、无 speculative 路径时，最后一个 prompt token 的 scalar forward 位于首次 callback 前，故预期 first encoded=32、后续 decode=32×63=2016、总计2048；不能沿用 prefill 候选的 decode encoded=0 规则。raw state exporter 保留全部 schema2 数据，适配 subject/scope 需可逆证明。

私有 CPU 实施已完成并冻结为 556 文件：只改 header、GPU C 和 MoE 的一个调用点，新增八个 driver / generator / adapter / exporter / CPU target / 测试文件；全部 548 seed 文件的原始 SHA 保留。完整 patch 正向重建556、逆向恢复548并删除八个新文件通过。235 项 actual GPU C selector / ABI / span / alias / PSO / 失败测试、20 个独立 flag 进程、真实 caller 的逐层/长位置/失败/alias fixture 和20项 driver CPU 测试通过。N256 动态计数正例对应 first32 / decode8160 / total8192，实际 callback 数不等于请求 N、N1 或 speculative telemetry 均拒收。

实际62项配置的 mode0/1 无设备 preflight 通过；额外四项为固定 `MYDS_INDEXER_K_RESIDENT=0`、新候选 mode 和两个证据 sink，共66项，未联合启用旧 prefill 候选。CPU build/link 与普通预处理通过，capture observer 未编入。这些仍是 CPU / mock 合同证据，不是 GPU runtime 或 fullmodel 证明。[实施细节](benchmarks/2026-10-26-kv-only-finish/runtime-IMPLEMENTATION.md)、[完整身份](benchmarks/2026-10-26-kv-only-finish/runtime-build-manifest.json)、[patch 正逆证明](benchmarks/2026-10-26-kv-only-finish/runtime-patch-byteproof.json)。

目标M2使用同一冻结binary/library/protocol另跑唯一342CB、无timing，原生gold由M2自身执行。完整KV/window、dummyQ-before、guards、实际PSO/库和完成门再次通过；56份冻结输入/制品及canonical身份前后相同，manifest.failure不存在。[M2独立正式复核](benchmarks/2026-10-26-kv-only-finish/M2.independent-review-v2.json)，SHA `2c5433f72107d3461e5f4cb2cde9171aa747068c58e7b74908c60005d4211edc`。没有以M4输出作M2 oracle。

目标CPU构建验证了完整556 source前后不变，binary SHA仍为 `a4288dd88714bc32fcc2f54897fb636e5ed6660caca229c7cf391ba709dfaf05`，target-bin与target-gold相同，分别加载d12与3837库的实际见证留给真实 --run。两个mode的66项无设备preflight通过。[目标构建身份](benchmarks/2026-10-26-kv-only-finish/target-build-manifest.json)。新请求由目标GGUF元数据和原生renderer实际得到11/883 tokens，原正文先行冻结，未为凑名义10/800而调整。

唯一真实模型 Q 已通过：两负载各G_lib→off→on、warm1+verify1、N256，共12代；source/target/输入/M2组件/reader均在数据前绑定。1512个inventory文件、227,608,039字节完整复核，每代123个schema2 state payload与完整129,280维final logits、tokens、output原始bytes全部一致且logits finite。四次mode1均first32/decode8160/final8192、reference0；mode0的candidate encoded均0、reference8192。真实target forward calls/rows/output/cycle均256，spec/draft/batch2/PLD计数0，timing profiling关闭。

[Q正式独立复核](benchmarks/2026-10-26-kv-only-finish/decode-fullmodel-Q-review.json)，SHA `fdfe6c3d7b11d12e8219b5216fb809a2ea0bd22e1aa38376ea141b2bff3a933e`；[peer同raw复核](benchmarks/2026-10-26-kv-only-finish/decode-fullmodel-Q-peer-review.json)只在记录的path_maps不同，其余证据与判定一致。G_lib是同一private C binary配部署库，不是部署executable/source或HTTP bridge。

独立冻结的24代正式P已完成：每负载A1off→Bon→A2off，每臂warm1+formal3，N256。数据前门要求两负载decode保守节约严格>2%，prefill/TTFT/decode控制漂移分别≤1%，secondary prefill/TTFT对较快A回退≤0.5%；取三formal中位数、原JSON小数转精确有理数，无epsilon/删点/替换/重试。

完整24代含6次warmup和18次formal，3012个inventory文件、455,092,539字节。全部final logits、123个schema2 state payload、tokens和output的跨臂/跨run逐字节比较通过；logits均finite。八次B各first32/decode8160/final8192、reference0；16次A各encoded0/reference8192；fallback/failure全零。真实target forward/rows/output/cycle均256，spec/draft/batch2/PLD全零、profiling关闭。

| 输入 | A1 decode ms/token | B | A2 | 保守节约 | 控制漂移 |
|---|---:|---:|---:|---:|---:|
| short /11 tokens |23.567207843|23.577505882|23.556698039|**−0.088331%**|−0.044595%|
| business2 /883 tokens |24.481505882|24.427835294|24.480282353|**+0.214242%**|−0.004998%|

六组metric的控制漂移均≤1%；四组secondary TTFT/prefill均通过≤0.5%的无回退门。短输入TTFT保守节约0.110828%、prefill0.043139%；业务为0.024363% /0.017904%。失败来自两个primary decode收益都未超过2%。删除了源代码中的无用计算，也实际编码了候选，但本轮没有测到达到采纳要求的整体收益；没有据此唯一定位Q_B、带宽、CPU编码或某种重叠机制为原因。

[正式独立P结果](benchmarks/2026-10-26-kv-only-finish/decode-fullmodel-P-review.json)与[peer同raw复核](benchmarks/2026-10-26-kv-only-finish/decode-fullmodel-P-peer-review.json)逐字节相同，SHA `6f8641c81b18d54df950132234e0bfe167cd309c35fdace0ae5e067e450c666c`，结论 `P_performance_gate_failed_no_retry`。保留全部样本和原门，不进入HTTP、取消恢复或生产采纳。

[最终只读现场核验](benchmarks/2026-10-26-kv-only-finish/final-postrun-identity.json)再次确认五个canonical制品/配置的完整SHA、模型stat与首1MiB身份均未变，已识别模型benchmark进程为空。模型整个GGUF payload未重算SHA，本文不声称运行中服务健康或部署executable/source的HTTP等价。全部小型证据见[evidence-index.json](benchmarks/2026-10-26-kv-only-finish/evidence-index.json)，完整raw及耐久归档位置另列索引。

完整M2 raw和源码制品、M4组件/seed补充包已完成lossless归档与SHA验证，持久保存于`macstudio:/Users/fangqingyuan/Benchmarks/myds/2026-10-26-metal-performance/`；M4包和完整manifest在本机同路径也保留。原始证据未删改。[归档说明](benchmarks/2026-10-26-kv-only-finish/durable-archive.md)。
