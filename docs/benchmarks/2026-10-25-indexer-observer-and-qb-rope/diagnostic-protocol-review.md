# Prefill indexer R3 稳定性诊断：独立 gate review

本文件仅为 CPU 侧设计审查。没有 GPU、远程操作或对冻结文件的修改。诊断目标是判断旧 harness 是否存在需要修复的测量依赖，并为新的前瞻性性能门提供依据；**诊断 cohort 永远不用于采纳**。R2 原 cohort 的六组联合性能门失败永久保留，不替换样本、不改 ±2% 漂移门、不将本轮诊断当作补样或“重试到通过”。

## 从旧源码与 180 条 raw 可以直接确认什么

读取对象：`tools/bench_indexer_k_capture.m` 和 `docs/benchmarks/2026-10-25-indexer-real-input/replay.jsonl`。旧记录共 180 条计时 CB，其中 18 热身、162 正式；下表只分析原 162 正式记录，未删样。

| layer / control | 原 A2/A1 中位数之比漂移 | 配对 A2 比 A1 快的次数 | A1 前 GPU 间隙中位数 | A2 前 GPU 间隙中位数 |
|---|---:|---:|---:|---:|
| 2 / actual exact | −1.0185168% | 9/9 | 6.414375 ms | 6.542792 ms |
| 2 / TILED | −2.0846075% | 9/9 | 6.546625 ms | 6.623167 ms |
| 8 / actual exact | −1.4445598% | 9/9 | 6.589000 ms | 6.574958 ms |
| 8 / TILED | −1.8193827% | 9/9 | 6.526333 ms | 6.611750 ms |
| 14 / actual exact | −1.5357903% | 9/9 | 6.547875 ms | 6.618375 ms |
| 14 / TILED | −1.3080671% | 9/9 | 6.577625 ms | 6.616250 ms |

六组共 **54/54 次配对 A2 都快于 A1**。这些重复不独立，不能据此套二项分布给 p 值；但它是跨层、跨 control 一致的顺序信号，不能只解释为 layer2 的一个随机越门点。表中漂移保持原公式 `median(A2)/median(A1)-1`，没有改用配对比值的中位数冒充原 gate 结果。

这里的“GPU 间隙”是相邻计时 CB 的 `GPUStartTime[i]−GPUEndTime[i−1]`，**不是已经测明的纯 CPU gap**。旧 raw 没有 poison、validation、SHA、JSON、encode/commit/wait 各段 CPU 时刻，不能从它唯一归因到 host、缓存、GPU 频率或 driver 排队。18 个 layer/control/arm 子组的该间隙中位数约 6.38–6.64 ms，已经大于候选本体约 4.54 ms。

源码中几个确定的混淆来源：

- `real_index_run` 每次先在 shared output 上 CPU `memset(0xA5)`，再编码一个 concurrent encoder、一个 dispatch、commit、wait。
- 每次 wait 后，`real_index_validate` 做 actual capture 与 TILED 两次全 score diff、完整 finite/causal/poison 扫描；随后 timing emit 又做一次完整 memcmp、SHA 和同步 JSON/fflush。
- 固定流为 `[A,B,A][A,B,A]…`。分析样本的 A1 前驱是同一个 A，A2 前驱是 B；A1/A2 共用同一个 output allocation。**arm 标签、前驱 kernel、output 复用史与组内位置完全混淆。**
- 检查与 I/O 不直接进入保存的 GPU span，但它们处于连续 CB 之间，可能改变下一次 dispatch 的输入条件。不能因为计时是硬件 timestamp 就宣称 observer 无影响。

因此本轮优先检验“顺序 / 前驱 / shared output 复用 / per-dispatch CPU 取证”的测量依赖，不先宣布唯一根因。

## 只运行一次的有限诊断矩阵

主诊断固定使用 layer2 的原始 **128×15424 BF16、pos0=30720、ratio=2** 输入，不改 shader、PSO、参数、输入或编译单元。A 固定为完整 TU 的 TILED，B 固定为同一库的 K-resident。选择 layer2 是因为它触发原门且 signal 最强；诊断结论不自动扩大到其他层。后续正式新 gate 仍须包含 layers2/8/14 与 actual exact/TILED 两种控制。

保留完整生产 TU 与 deployed native 的身份绑定。先完成 layer2 的原 numerical/consumer sanity；任何数值、input、canary 或库身份失败都停止，不补跑。记录所有 job、原始 CPU/GPU 时刻和完整输出见证。

| cell | 真实执行 kernel | 调度 | output 映射 | 每 CB 之后的 CPU 工作 | 分离的问题 |
|---|---|---|---|---|---|
| C1 | A=TILED，B=K | 原 ABA | 两个 provider output，A1/A2 共用 A output | 原完整 validation+hash+JSON/fflush | 固定一次旧模式诊断参照，不是 R2 的补样 |
| C2 | A 与伪 B **均为同一个 TILED PSO object、相同 args/input** | 原 ABA 标签 | 与 C1 相同的两个 output 映射 | 与 C1 相同 | 同 kernel 伪三臂；任何“优化收益”都是测量/状态依赖，不能来自 K 优化 |
| C3 | A=TILED，B=K | 平衡 AABB | 与 C1 相同的两个 provider output | 原完整 validation+hash+JSON/fflush | C1→C3 只改变顺序，解开前驱与 A1/A2 标签 |
| C4 | A=TILED，B=K | 平衡 AABB | 每 job 独立 output slot | 原完整 validation+hash+JSON/fflush | C3→C4 观察 output 地址复用 / 工作集的影响，不宣称只隔离某一种 cache |
| C5 | A=TILED，B=K | 平衡 AABB | 与 C4 同样的独立 slot pool | 仅 timestamp/状态记录；全部 score 验证、hash、JSON 在整段之后 | C4→C5 只移动 per-dispatch CPU 读回 / 取证工作；不删 correctness 检查 |
| C6 | A=TILED，B=K | 平衡 AABB | 与 C4/C5 同样 slot pool | 与 C5 相同，另在每次 wait 返回后固定 idle 6.6 ms | C5→C6 检验 gap dose；实际 GPU 间隙仍须测量，不冒称精确复现旧 6.6 ms |

C2 的“伪 B”必须绑定 A 的**同一 PSO object**，不能用重新编译的 alias、另一个 native wrapper 或 candidate shader 冒充 null control。逻辑 arm 标签与 actual provider/symbol 分别记录，不能因标签写 B 就把它计为候选执行。

固定两遍 block 顺序：第一遍 `C1,C2,C3,C4,C5,C6`，第二遍 `C6,C5,C4,C3,C2,C1`。不得根据第一遍结果调整第二遍、补 warmup 或更换顺序。

- 每遍 C1/C2：固定 1 组 warmup ABA + 9 组分析 ABA，即各 30 个计时 dispatch。
- 每遍 C3–C6：固定 1 个 warmup AABB cycle + 6 个分析 AABB cycle，即各 28 个计时 dispatch。warmup 最后一个 B 是首个分析 A 的前驱；每个分析 cycle 恰好包含 `B→A、A→A、A→B、B→B` 各一次。
- 两遍合计 **344 个诊断计时 dispatch = 44 个预热 + 300 个分析样本**。numerical / TopK 的额外 CB 单独计数，不混入 344。所有 job ID、provider、slot、前驱和 warmup 标记在 GPU 前生成并写入 plan。
- 不设置“跑到稳定”的条件，不以结束时 gate 是否好看决定追加轮次。

独立 output pool 只需 28 个约 7.9 MB 的 body，约 211 MiB；在所有 cell 之前一次分配并保持，所有 cell 使用同一 pool 身份。shared 模式只复用固定两个 slot，独立模式每 job 一槽。每次 dispatch 前的 poison 仍保留，C4/C5/C6 的 poison 位置、范围、方法相同。不要同时把 poison 移到整段之外，否则 C4→C5 又混入一项干预。

C5/C6 每次仍保留原来的一个 dispatch/CB、commit、wait，不改为异步多 CB 提交，不加入 GPU cache flush、dummy kernel、额外 encoder 或 per-dispatch GPU counter sampler。区别仅在 wait 后的 CPU 取证位置。独立 slot 在验证前不得覆盖；每次 score 完整保留到段末，全量校验与 guard、inputs pre/post 均必须通过并保存，失败输出也保留。完整 bytes 或无损可物化的去重存储均可，不能仅留 checksum 后宣称可独立复算。

固定 idle 6.6 ms 不是频率/缓存根因证明，也不是测量修复方案的候选参数扫描。记录请求的 sleep、实际 sleep、实际 commit/launch gap；若 C6 的实际 GPU gap 与旧模式不同，照实报告，不再调 sleep 重跑至吻合。

## 必须记录的原始观察

每 job 记录这些字段，全部使用与 MTL GPU 时间相容的 host monotonic clock（例如项目现有 `myds_metal_host_time` 的 mach_absolute_time 时基）；禁止混用 Unix wall clock 与 GPU timestamp 作减法：

- plan hash、pass/cell/block/cycle/job ID、真实 provider/PSO symbol、logical label、input/args/library SHA、output slot 和资源身份。
- `previous_actual_provider`、`previous_output_slot`、previous CB GPU end、当前 GPU start/end/span。
- CPU poison begin/end、encoder begin/end、commit call begin/end、wait begin/return。
- validation begin/end、memcmp/hash begin/end、JSON/flush begin/end；deferred 模式明确这些字段在 segment 后，不伪造成每 job 都发生过同步检查。
- C6 requested idle / actual idle；每 job 的 submission→GPU start 与 GPU end→host wait return。
- 输出完整 bytes/SHA、expected provider-specific table、ordered IDs/consumer sanity（段外）、input pre/post bytes、canary、finite/`-Infinity`/poison 结果。

L 模式的 JSON/flush 本身是干预的一部分，仍应真实执行；完整 after-emit 时刻可先留在内存，段末另存 timing-detail，避免为了“记录 after”再插一次额外同步 emit。D 模式在段内不做 full-output SHA、完整 memcmp、磁盘 raw 写入或 stdout flush。段末再执行完整检查，不把少做检查当提速。

设备/OS、编译器、完整 source/library/harness/plan/reader SHA、有效 config、进程检查、共享实验锁和前后制品身份照旧保留。频率/功耗若有既有低侵入采样可以附录，但不把缺少频率数据的实验解释成已测明 DVFS，也不要为取 frequency 引入逐 dispatch profiler。

## CPU reader 的预冻结分析与判定

所有分析保留两遍结果，不只合并出一个总体 median，不挑最快 cell。诊断不输出 `production_adoption=true` 或 `fullmodel_eligible=true`。

1. **完整性先于机制解释。** 恰好 344 个 timing job、预注册序列/slot、全部 warmup 和 input/output bytes 完整；分 pass/cell 校验实际 PSO、shader/args/inputs 不变。任何缺项或数值失败则该诊断不支持工程结论，不补样。
2. **旧模式复现描述。** C1 每遍分别给 A1/B/A2 median、A2/A1 比和全部九对符号；不能把本次 C1 通过或失败写成 R2 旧 cohort 被重判。
3. **伪三臂测量地板。** C2 每遍给相同 kernel 的 arm/slot/位置差异。若它也出现稳定非零 A1→A2 偏移，先认定 harness/状态依赖存在，不能把同量级变化当优化收益。
4. **前驱分层。** 对 C3–C6 分别列 `A|prevA、A|prevB、B|prevA、B|prevB` 的完整值、median、IQR、min/max；每遍每个 transition 有 6 个分析点。不要将固定 ABA 的 A1/A2 直接命名为“随时间漂移”，也不要把两个前驱层平均后隐藏一个仍偏移的层。
5. **真实优化与顺序偏差分开。** 在同一 pass/cell 下，对相同 predecessor、同一 output policy 和相同 host policy 分别比较 A 与 B。只有两个 predecessor 层都保持 B 更快，才可描述“收益方向没有完全由顺序制造”；它仍是诊断观察，不是新的采纳 gate。
6. **host 影响看实际时间。** C4→C5 检查 CPU 取证时间移出后，GPU 间隙和相同 provider 的 span 是否共同变化；C5→C6 检查纯 idle dose 是否改变相同 provider span。C4 与 C6 不能因 nominal 6.6 ms 相同就视为 CPU 活动等价。
7. **forward/reverse 一致性。** 每项位置、前驱、slot 和 host-policy 对比都分别显示两遍。方向反转或仅一遍出现的现象记为未定，不替它造唯一机制；不对 344 个相关样本做虚假的独立性显著检验。

工程诊断可在 GPU 前冻结一个较小的**观察分辨线**（建议 1.0%）用于标识 arm/前驱依赖，而不是采纳阈值。举例：同 provider 的同向差异两遍均超过 1% 时标为“可复现测量依赖”；低于该线只能写本次未分辨，不能证明无依赖。这个观察线不能替代或放宽原 ±2% / >2% 性能门。

## 什么时候允许提出测量工程修复

优先候选是**保留全量数值证据，把每 CB 后的 CPU output 扫描/hash/JSON 移出计时序列**，而不是选择哪种顺序让候选读数最好看。只依据 control/null 的稳定性和计量定义选择测量方式，不以 candidate 的最大收益选择 cell。

- C1→C3 发生变化而 C3→C4/C5 变化很小：顺序/前驱是首要待处理变量；不能把 deferred validation 单独宣传为修复根因。
- C3→C4 明显变化：output 复用/工作集条件有影响；新 gate 必须明确并平衡资源条件，不能默换 allocation 再当同口径旧基线。
- C4→C5 在两遍中一致改变 bias，且 C5 全量 bytes 正确：可以提出“取证出窗口”的工程修复。它改变的是测量观察方式，不是模型算术；不能将缩短 host gap 算为真实引擎提速。
- C5→C6 使差异重新出现：说明 dispatch 间 idle 相关的设备状态值得进一步控制，**不能唯一归因到 DVFS**。
- 若 C5 仍有大于原 2% 的 control/前驱/时间偏移，或两遍结论冲突：不进入新的真实性能 gate。本轮诊断如实结束，另立可证伪假设，而不是更换样本数、sleep、warmup 或 threshold 再试。

若要采纳测量修复，应另有 CPU mutation tests：误把 warmup 计入正式、交换 actual provider、缺少 slot、GPU/host 时基错配、漏掉 deferred 输出、伪 B 实际绑定 candidate、计数/CRC 声称正确但 bytes 改变均应拒收。新 reader 必须从 raw 重算，不能相信 harness 的 `passed` 或 exit 0。

## 与后续新的真实性能 gate 分离

诊断结束后先发布结果和工程修复差异，再由主线程冻结全新的 source/harness/plan/reader SHA、明确统计量和固定样本数。**不得把 C1–C6 的任一正值直接用于采纳，或把它们与 R2 合并。**

新 gate 至少保持以下边界，不因诊断而放松：

- 同一真实捕获，layers2/8/14、actual exact 和 TILED 两种控制，全部六组都必须通过；若因平衡前驱增加分层约束，则各预注册层也都要通过，不能只看加权平均。
- actual exact 逐 byte 复现捕获；candidate 与 TILED 的全部 scores 位一致；完整 ordered Top512 及 inputs/guards 通过。原 exact→TILED 的 4 个 cell 差异继续报告，不改写成 score 0-ULP。
- 绝对 control drift 仍 ≤2%，保守耗时节约仍严格 >2%；固定 warmup 与正式数量，在 GPU 前冻结，不根据新数据追加。
- 资源布局、CPU 取证位置、前驱平衡和统计定义显式记录。若以新模式替代旧 fixed ABA，必须先说明新 A1/A2 控制如何匹配 predecessor/slot/host 条件；不能沿用 ABA 名称却改变分母。
- 仅在新的真实组件联合门通过后才考虑 fullmodel；fullmodel 仍需完整 logits/tokens/output/state、单独 prefill/decode、真实业务/长上下文与原收益/恢复门。组件诊断不开放 runtime 或生产采纳。

本设计的产出可以是“有明确测量修复依据”，也可以是“依赖尚未定位，继续停留组件阶段”。两种结果都不改变 R2 的永久失败记录。

## 对主线程当前 schema2 / 9-phase 计划的独立结论

主线程选择先执行更窄的前驱 conditioner 诊断，计划文件为 `/tmp/myds-indexer-r3-local-20261025/diagnostic-protocol.json`，schema2。前文 6-cell / 344-dispatch AABB/idle-dose 方案保留为尚未定位时的备选设计，**不追加进当前计划、不变相增加 GPU 次数**。

当前 9-phase 分别覆盖 exact/TILED/candidate 的三个同 provider null 三臂，以及 conditioner=A 的 legacy、conditioner=B 的 balanced 三种 observer/output 变体。每 CB 后统一 snapshot 完整 score/guard；inline 执行完整双 diff、scan、SHA，deferred 只把这些验证移到四 CB 后；所有 phase 的 raw/JSON 都在四 CB 后保存。因此它是受控的前驱干预，不是对 R2 原 observer 的完整重放，也不是“完全没有 CPU output 触碰”的 deferred 模式。

计数为 13 rounds × 9 phases × 4 steps = **468 条计时 CB**：round0 的 36 条预热、round1–12 的 432 条诊断分析记录。432 中包含 108 条 conditioner 和 324 条 A1/B/A2；不能把 468 全称为候选性能样本。初始 numerical/TopK 的额外 CB 继续单列。

在“只诊断是否值得另立工程测量假设、不直接采纳”的范围内，**未发现阻断当前 GPU 数据收集的问题**。schema2 的 sham 公式已检查 A1/B/A2 三位置最大/最小 median 之比，而非只检查 A1/A2，能够防止中间 B 位置产生假收益却躲过 null gate。其数值、sham、legacy 现象复现与 conditioner reduction 公式均事先列明；deferred/distinct 四相位明确只作描述，不能事后择优替换 primary balanced-inline 决策相位。

仍须保留两个解释限制：

1. `phase_index=(round+order)%9` 的循环轮换并未平衡所有相位对的相对先后。对正式 rounds1–12，legacy phase3 仍有 11/12 次先于 balanced phase4；phase5/6、phase7/8 也都是 11/12。因此结果最多支持受控干预下的新测量假设，不能声称已排除全部随时间的相位偏差或唯一证明前驱 kernel 是根因。schema2 已将“不完全位置平衡”列为 limitation。
2. 当前 helper `wall()` 使用 `CLOCK_MONOTONIC`。在没有证明它与 GPU timestamp 共享 epoch 前，reader 可以分别计算 host-host 时差与 GPU-GPU 间隙，不能直接作 `GPUStartTime−host_commit` 并称为 launch 延迟。若需要该跨时基指标，应使用项目已有 mach_absolute_time host clock 或先记录明确的时基校验；不应靠事后平移去美化数据。

即使该 schema2 的 measurement-correction-support 规则通过，也只允许主线程公开诊断并**另外冻结**新的真实组件计划。conditioner=B 使两个 native A 测量有相同前驱，但 B 本身仍前接 A；不能把诊断中 A/B 的比直接视为已消除所有 predecessor confound。新的 gate 应显式定义各 measured provider 的前驱、output slot 和 observer 条件如何匹配，或分层比较同一前驱；原 ≤2% 漂移、严格 >2% 收益、三个层/两个 control 联合门不变。R2 原失败、该诊断和未来新 gate 三者分别归档。

## GPU 前最终 schema3 复核

当前实际计划已升级 schema3，完整 SHA-256 为 `76c032bda7af0673eb3d7de77762d546aa9f8b1c3cd8b6bb9bc596a0536a111f`。warmup round0 使用升序 0..8；正式奇数 round 使用降序 8..0、偶数 round 使用升序 0..8。只读核对 `tools/bench_indexer_stability.m:157–158` 与协议一致，独立 CPU 枚举确认三组 legacy/balanced 相位对（3/4、5/6、7/8）均为 legacy 先行 6 次、balanced 先行 6 次。schema2 的 11/12 相对先后混淆已在任何该诊断 GPU 工作之前修正，不再是 schema3 的问题。

schema3 仍为 468 条计时 CB（36 warmup、432 analysis），数值证据与所有预冻结判定公式不变。六正六反能平衡一阶 phase 位置与 pair 的相对先后，但仍是确定性顺序，不能消除全部前史或证明独立微架构根因。主线程也明确仅作 host-host 与 GPU-GPU 差分，不以 `CLOCK_MONOTONIC` 与 GPU 时间跨 epoch 相减；这一解释限制已落实。

**最终结论：在上述有限诊断范围内，没有发现阻断 schema3 数据收集的设计问题。** 即使诊断支持 measurement correction，也只支持公开证据后另立冻结组件实验；不重新裁定旧 R2 cohort、不从本诊断采纳、不放宽原门。未来独立组件 gate 仍须预先匹配或分层前驱/slot/observer 条件。
