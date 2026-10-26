# Gemini ds41-perf 方案的当前 decode 映射（CPU 只读）

2026-10-26。审阅文档为 `/Users/fangqingyuan/.gemini/antigravity-ide/brain/3d5b3923-370d-4918-a651-b95c0d4253f9/ds41_perf_analysis_and_adoption_plan.md`，SHA e0314a33d21d5058aa8f98640883f07d7b001a76d199211d5f1c806722a4759b。这里只核当前 myds 源码、冻结62项配置和已有账本/26日真实结果，不把文档的136提交或bit-identical自述当作已验证事实。未GPU、未修改src/header/Metal、未部署；upstream具体commit由另一路独立审阅。

## 一、逐项映射与建议结论

| 方案位置 | 当前实际情况 | 分类与动作 |
|---|---|---|
| 153–157 literal QA/KV Q8 pair、MXFP4 MoE、Q8 Output_B+HC | 当前模型的已测QA/KV为Q4_K（账本§74的真实GGUF独立双输出原型）；当前普通FFN明确要求GU=IQ2_XXS、Down=Q2_K、shared=Q4_K；Output_B当前默认选择Q4 NB。 | 这些指定Q8/MXFP4 symbol不能直接替换当前类型。Q4已有部分literal provider；只能审未覆盖的Q4残余，不能标记整项全新。 |
| 175–180 BF16 router / FP8 index-QB与Engram窄副本+16B load | 当前未找到这些紧凑副本、对应BF16/E4M3 GEMV provider或初始化桥；现有F16 GEMV→BF16是输出舍入融合。 | 是本轮decode主要新增候选，但先做真实权重完整CPU bit-roundtrip和类型/规模清单；通过后才值得私有组件。窄存储无损不等于新GEMV累加树无损。 |
| 159–164 42-TG deferred collapse | target HC collapse+RMSNorm当前是1TG×1024，不存在该part/norm调用链；现有HC mix12TG/cluster6TG是另一算子且融合flag默认0。 | 未实现同一42TG合同；不能认定“只占1核=空泡可收回”。需要重建现有每lane的累加/两级归约树，且已有精确split norm显著负结果。低于窄副本优先级。 |
| 166–169 单token rope_tab与inverse RoPE epilogue | 当前只有once初始化frequencies[2][32]；没有position cos/sin全局表。Attention register inverse-RoPE/BF16 epilogue已经实际启用。 | inverse融合属于已有；校正后的position table是未测新机会，需覆盖两频率和compressor的pos−1等位置，并按现有GPU precise trig/FMA/BF16合同验证。不能宣称新增40次inverse dispatch收益。 |

## 二、literal/static shapes 的准确源码边界

- `src/myds_decode_candidates.inc:188–196,215–221` 按完整shape选择HEAD129280×5120、Q_B32768×1280、Output_B5120×8192，严格几何/范围/alias门后派发实际PSO。`metal/myds_q4_v41.metal:29–42,105–117`已有模板ROWS/WIDTH、固定row bytes、FIXED_NB和NR2/NR4变体。冻结配置`/tmp/myds-indexer-r2-real-local-20261025/deployed/myds.conf:63–66`为DENSE_Q4_V41=1、NB=1、NR0=2，属于实际启用而非仅shader存在。
- QA/KV本身仍由`src/myds_v41_moe.c:2414–2415,2497–2498`分别调用带真实type的BF16输出GEMV。模型loader `src/myds_engine.c:682–694,721–753`保存原GGUFtype，未自动改成Q8。
- 文档Q8 pair对当前Q4模型类型不适用；但Q4版同一机制已经做过：账本§74 `9952–9968,10034–10062`是独立权重/输出、单dispatch双区的零copy原型，三层组件0-ULP、实际concurrent门无有效收益，不接fullmodel。不能只引用更早§65“subview未做”而忽略§74补齐的负结果；也不能把该结果扩大成所有literal优化不可能。
- Output_B当前`src/myds_v41.c:4510–4531`为Q4固定shape BF16投影，之后原HC expand。文档Q8+HC epilogue不能直接用。Q4适配若保持dot、原BF16边界、HC四残差计算顺序和全部依赖，仍是一个不同的新实现假设，但没有当下速度证据；不要将Q8上游结果当成本模型收益。
- Output_A固定wrapper另已有账本§77组件约7.5%、完整两负载仅0.356890%/0.404154%的结果（`10493–10507,10657–10672`），保持default-off。
- MXFP4不能靠注册一个kernel接到当前路由专家：`src/myds_engine.c:245–256`的支持类型不含MXFP4；`src/myds_v41_moe.c:1755–1759`普通已采用FFNHC分支明确锁定IQ2_XXS/Q2_K/Q4_K。`include/myds_gguf.h:67`有MXFP4枚举、GGUF parser知道大小，并不等于native inference provider支持。把当前IQ2专家转成MXFP4属于新模型artifact/量化合同，不能预设权重或logits0-ULP。

## 三、窄副本：当前确实未接，但必须先证明可逆

当前加载是原tensor内存wrap：`src/myds_engine.c:259–270,905–913`保留router原GGUFtype；target附加权重`src/myds_v41.c:2731–2741`严格核type/shape后wrap。Indexer的`indexer.attn_q_b`在`2808`锁定F16、1280×4096，只在8个index-source层存在；Engram `src/myds_engram.c:722–730`锁定2个F16矩阵，每个6144×25600（维度来源`include/myds_engram.h:10–16`和target width5120）。没有为它们构造E4M3 buffer。

当前dense桥`src/myds_metal.m:4700–4711,4753–4758`只给该类F16/F32权重选择原dense GEMV。Index-QB目前使用`src/myds_v41_moe.c:2563`的F16→BF16输出fuse，bridge在`src/myds_metal.m:4909–4951`；Engram prestage在`src/myds_v41.c:4488–4517`仍走F16 GEMV与原输出BF16处理。它们都不意味着FP8权重窄存储已经实现。

CPU第一门应该对**实际冻结GGUF的全部目标tensor**记录descriptor/type/extent/SHA，执行原bits→候选编码→原bits逐元素可逆验证，报告全部失败值而非只看量化误差。Router先确认实际F32及BF16可表示；源码的F32/3.9MB注释不是可靠的实际type证明。F16标签不能证明index-QB/Engram全部来自无scale的E4M3：原FP8可能带分块scale，或widen过程中值已改变；signed zero、subnormal、NaN/Inf及负零处理也必须明确。失败则不能以“来源本来FP8”强制窄化。

即使编码全可逆，新kernel必须保持原lane读序、逐lane乘加顺序、每SG归约与跨SG归约和BF16输出。`metal/dense.metal:82–120`是两次simd_sum加32-slot中间树；Q8示例`179–248`保持原block/scale摆放，普通F16/F32也有自己的归约合同。宽读取可以只改变load表示，不能暗中变成dot16、新unroll/K分片、MMA或另一括号结构。必须完整生产TU原prefix+实际部署库控制，先完整中间投影/router logits/路由IDs和后续state/logits门，再正式两负载decode计时。

### 按当前target尺寸的逻辑payload预算

以下是维度与格式的计算，**不是当前page-cache/phys/wired可用空间结论**；router行只在实际type为F32且可逆门通过时成立。

| 目标 | 原始payload | 窄副本payload/新增独立allocation |
|---|---:|---:|
| 40 router ×384×5120，F32→BF16（条件项） |314,572,800B=300MiB|157,286,400B=150MiB|
| 8 index-QB ×4096×1280，F16→E4M3 |83,886,080B=80MiB|41,943,040B=40MiB|
| 2 Engram ×25600×6144，F16→E4M3 |629,145,600B=600MiB|314,572,800B=300MiB|
| 三项都成立 |980MiB|490MiB=513,802,240B|

这些窄副本若仅用于decode，prefill仍需原F16/F32权重，因此旧mapping/Metal wrappers未必释放，490MiB通常是**新增**独立payload，不是整机内存下降490MiB。decode对这些矩阵的逻辑权重读取减半，不代表全模型读取或时间减半；router/indexer工作可能与主cohort重叠，FP8 decode指令/宽load安排也有代价。内存安全裕度交由root的实际现场预算判断。

## 四、HC collapse 和 Engram gate 多TG不是“零风险”

target默认`src/myds_v41.c:3977–4000`融合HC四stream weighted collapse与RMSNorm；`src/myds_metal.m:3783–3822`是grid1×1×1、1024threads，不是文档的42TG。其关键数值链`metal/dsv4_hc.metal:1155–1188`为：原i+=1024的每lane遍历→c0*w0、c1*w1、c2*w2、c3*w3按序acc→BF16边界→dot(v,v)逐lane累计→simd_sum→原32个SG slot→simd_sum→sqrt/weight。任意按连续K块切42组并把partial相加都不能仅用实数等式证明0-ULP。

若未来评估同类分布式方案，应先让每原lane产生与原来完全相同的partial，再按原SG/32slot树归并，显式保留原中间BF16。TG数量必须从此模型dim5120与原虚拟lane映射推导，不能照抄42。已有12TG/6TG `hc_mix_norm_project_fused`是HC mix的24维投影（`src/myds_metal.m:3650–3673`），不是collapse；其mode在`3630–3635`默认0，不能因shader存在说当前全开。

账本§15 `2541–2566`的同类norm split在正确性修好后完整0-ULP，但M2 3.739→8.107us，慢116.8%；多一dispatch、join和partial往返抵消并行。这不在数学上否定任意HC新实现，但明确反驳“单TG只用1/76核心，所以拆开必快”。当前HCsum也已有1024threads采用记录（§4.5 `1676–1684`），不能把原256thread前提继续套用。

Engram gate目前不是40层的同一collapse：`src/myds_metal.m:4189–4217`按4个residual stream分派4TG×32lane；`metal/dsv41.metal:312–327`每lane顺序算h²/k²/dot，再simd_sum、rsqrt、sqrt/exp gate和残差BF16。跨TG拆此dot/平方和同样改变树，需独立合同和完整输出证明，不能与HC collapse凭名字合并为已保证无损的phase1。

## 五、RoPE：已有inverse融合与真正新table机会必须拆开

真实attention head宽512、index head宽128（`src/myds_v41.c:32–38`），但两者**都只对末64维做RoPE**：`metal/dsv41.metal:83–95`使用width−64+2*lane，lane0..31。因此每种frequency是32对。文档168“float2[64]对应128维head64对cos/sin”是错误解释；若上游table总64项，应解释为raw32+compressed32两种frequency，而非本target旋转128维。

本仓已有`src/myds_v41_gpu.c:262–294`的once frequencies[2][32]（powf/reciprocal/YaRN operation order受precise float-control保护），仍由shader逐调用计算pos*freq和precise trig，未接position cos/sin table。这个table可以保留为一个**校正后的新候选**，但要解决：

- 同token的raw/compressed两表、forward/inverse符号与原precise GPU trig，不能直接改CPU sinf/cosf或half表。
- Compressor commit不总用当前pos：`src/myds_v41_moe.c:2601`的start=pos+1−ratio；ratio2的commit用pos−1，不能拿当前pos的comp表替代。prefill rows/stride和其他generic API也不能顺手改。
- 保留两次BF16边界和当前accepted imaginary FMA方向，表producer→consumer依赖、跨splitCB/复用lifetime与原barriers；新table生产可能多一个dispatch和RAW join。

Attention inverse epilogue已启用：冻结config51–52为PACKED_REGISTER1/ROPE_FUSE1；`src/myds_v41.c:4409–4438`实际调用direct RF、通过rope_consumed跳过standalone inverse；`src/myds_v41_gpu.c:2250–2266`派发`kernel_myds_attention_register_rope_fuse_f16_512`，`metal/flash_attn.metal:2355–2374`已有pre-BF16、precise trig、原imaginary FMA和post-BF16。因此不能再把新增inverse融合算作这次table迁移节省的40个dispatch。

数值警示已有真实目标结果：`docs/2026-10-25-indexer-observer-and-qb-rope-review.md:125–143`的完整TU Q_B→forward-RoPE v1在M4/M2分别3/6个tail单值失败，均1BF16 ULP，native gold/fullTU native自身通过；源码表达式看起来接近并不保证编译后位一致。它不是table的直接证否，但禁止把table列作零风险phase1。

性能警示为26日闭合结果：`docs/2026-10-26-decode-kv-only-finish-review.md:51–60`，删无用dummy-Q的131,072次trig和1MiB逻辑读写且实际每B采用8192次，完整数值/状态/控制门通过，双负载decode仍仅−0.088331%/+0.214242%，未达>2%。这没有唯一归因到Q_B或带宽，也不证明table必败，但“消除大量trig/工作量→显著decode提升”已无足够推理依据。table应在窄副本CPU可逆核查之后，作为有完整目标组件和图级门的低优先级实验。

## 末段建议

把方案的decode部分重写成三类：已启用的Q4 literal providers、RF inverse融合和BF16输出融合应记录为baseline；类型错配的Q8/MXFP4 symbol与已测负结果不重复迁入；真正新增优先做router/index-QB/Engram窄权重的完整CPU可逆性与预算审计。可逆性通过后再准备保持原K walk/归约的私有宽load组件，不先改生产模型或默认flag。校正raw32+comp32及compressor位置合同后的RoPE表可保留次级假设；42TG collapse先完成现有1024lane/32slot数学和source映射，不能以“零风险”排在前面。所有性能主张均等真实两负载、完整tensor/logit/state和实际provider门，不能继承上游提交信息中的bit-identical或时间百分比。
