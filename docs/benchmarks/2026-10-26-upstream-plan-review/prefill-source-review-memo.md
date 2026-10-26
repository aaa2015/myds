# Gemini 方案的 prefill/indexer 独立只读审查

2026-10-26。审查对象为 `/Users/fangqingyuan/.gemini/antigravity-ide/brain/3d5b3923-370d-4918-a651-b95c0d4253f9/ds41_perf_analysis_and_adoption_plan.md`。当前仓库 HEAD 为 `224dc70a839af1468c68c277384fa489ea55afbd`，工作树已有大量未提交差异，因此以下结论绑定实际读到的文件；相关文件 SHA 见同目录 `cpu-audit.json`。本审查没有修改源码、编译/运行 GPU、启动模型或访问远端。

## 结论

**register-only MMA 是值得下一轮最小原型验证的新机会；它和上一轮仅 hoist K fragments 的候选不是同一个变换。** Gemini 的完整优化清单混有已有实现、条件错误和过时状态，不能按“零风险 Phase1”直接迁入。尤其不能将本地 K-resident 全模型约1.5%的结果当作 register-only 的收益上界，也不能将上游 M3 Ultra 的 scorer 局部1.31→0.77秒写成 M2 Ultra 全模型41%收益。

BF16 index cache 已是当前常规路径；decode 的 masked scorer 已在读取 K 前跳过非候选行。额外的 selected-block 紧凑网格 scorer 也已存在且 default-off。真正仍不同的后备方向是长 prefill 的 mask-aware tile 跳过/调度，它只有实际 mask 非全选时才有空间，应先统计真实 tile 覆盖。

## 1. 新 register-only 与既有 K-hoist 的区别

- 当前 native 模板位于 `metal/dsv4_misc.metal:6618`。每 TG 8 token×32 key，4个 SG，每 SG 是8 token×8 key；`6637–6639` 的 qtg/ktg/dot 共21504B threadgroup存储。`6691–6740` 在每head复制Q、执行16次MMA、把dot写入TG后读回并累加，每head三道屏障。
- 既有 builder `tools/build_indexer_k_resident_probe.py:37–78` 的三个可逆编辑，只把16个 `simdgroup_load(mk)` hoist到head循环外；`45–47` 保留 `resident_k`，`58–61` 改MMA使用它。qtg、ktg、dot及原屏障仍在，动态TG存储仍21504B。它减少K的TG逻辑读取，不能称为消除TG往返。
- 官方提交 `0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4` 的 `metal/dsv4_misc.metal:7272–7341` 直接由device Q/K填充matrix `thread_elements()`，head循环无TG存储/屏障，dot也直接取fragment元素。该变换可移除当前K-hoist尚保留的Q staging、dot回写与三道head屏障。
- 代价也不同：满8-token tile下，原Q device逻辑读取约4KiB/head/TG；四个SG各自direct Q后约16KiB/head/TG。缓存可合并多少、实际DRAM字节、occupancy与spill均未测。不能把“去TG”直接等同减少所有访存或一定提速。
- 当前默认provider也需正确分层：`src/myds_v41_gpu.c:673–681` 只有显式 `MYDS_INDEXER_TILED>0` 才选TILED；`727–735` 默认尝试 `kernel_myds_indexer_score_chunk_exact`。最新正式K窗口的metadata中TILED为空，HEAD_SLOTS=1。历史ledger §91在当时TILED=1的描述不能当当前配置见证。

官方原始来源：[0f9b03c 完整提交](https://github.com/kernelpool/ds4/commit/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4)。本机核对的是只读bare mirror `/tmp/myds-upstream-ds41-perf-audit-20261026.git` 的该完整SHA，未将Gemini示例当作官方源码。该commit消息称M3 Ultra、bit-identical、scorer1.31→0.77秒，并同时包含另一个Q8 prefill变化；“Together”32K/64K吞吐不能只归给此scorer。

## 2. Gemini 示例的具体缺口

1. **完全不可见subtile没有写出−Infinity，是明确正确性缺口。** Gemini文件 `111–147` 只有 `if(row_base<max_visible)`；不进入时整块不写。当前native `metal/dsv4_misc.metal:6645–6657` 会写全部live不可见score为−INF。官方0f9原文件 `7336–7340` 也有 `else if(live)` 写row与row+1的−INF，该分支被Gemini展示稿漏掉。例：8 tokens、64 rows、max_visible16，行16..63的384个合法score cell会保留旧值/poison，而不是原合同的−INF。即使当前TopK另外限制visible宽度，也不能满足完整scores/重复buffer复用门。

2. **坐标映射本身正确，不应误判为官方bug；展示稿缺少必需helper。** Gemini `103` 调用 `dsv41_mma_coord()`，当前myds源码没有该helper。官方 `7267–7269` 的定义为 `qid=lane/4; x=(qid&2)*2+(lane%2)*2; y=(qid&4)+((lane/2)%4)`。CPU枚举32 lane的 `(y,x)`/`(y,x+1)` 恰好无重复覆盖8×8的64个cell。官方Q/K转置加载与dot取元素均依此一致。Gemini `87` 把每SG写成8×32有误，实际是每TG8×32、每SG8×8。CPU枚举只确认此源码映射的覆盖，不能替代目标Metal compiler/GPU输出门。

3. **旧GLM和当前DSV4参数同为72B，却不是同一ABI。** Gemini `93` 使用 `ds4_metal_args_glm_indexer_scores_batch`，本地定义在 `metal/dsv4_misc.metal:305–319`；当前TILED使用 `ds4_metal_args_dsv4_indexer_scores_fused`，定义在 `84–97`，host组装在 `src/myds_v41_gpu.c:682–700`。旧GLM多出cache_f16/row_group_size/pad，当前则有ratio及index_row_stride。若只替换symbol、复用当前ta，代表性Qtoken_stride16384、Qhead_stride512、weight_stride128会被误读为512、128、256；row_group_size会误读成16384。ratio2恰好被当cache_f162，sizeof及部分字段都像正确，反而更容易漏验。详见CPU JSON的完整pack/unpack结果。必须显式适配当前ABI，不能靠sizeof72或symbol存在通过。

4. **原head累加表达式不相同，0-ULP不能只靠“同16次MMA”。** 本地DSV4 `metal/dsv4_misc.metal:6729–6736` 是 `acc += max(s,0)*(w[head]*scale)`；Gemini `139–140`/官方0f9 `7328–7329` 是 `acc += max(s*scale,0)*w[head]`，它对齐的是上游GLM TILED表达式（本地旧GLM见 `2495–2502`）。scale=1/64是精确2次幂，但在subnormal、overflow、FMA收缩与fast-math编译情况下不能由表面等式推得全域逐bit。CPU IEEE-F32示例 `s=2^-149,w=64,scale=1/64` 前式保留最小subnormal、后式先缩放成0；这只是CPU合同反例，不声称当前M2真实输入或Metal FTZ实测会出现该差异。最窄原型应先保留本地表达式、head/16block次序及舍入边界，再通过完整raw门。

5. **host和边界合同缺失。** 直接fragment代码硬编码D128、16blocks、每SG8keys；需要明确32heads、128dim、128threads/4SG、非零token/ratio、F32/BF16缓存类型与行stride、合法Q/W/score stride、alias/range、尾token/尾key、全部不可见tile和每行visible因果门。Gemini `151` 仅说接入口，`212–217` 的新测试target与 `227` 的myds-bench并非当前可直接执行的既有验收命令。FNV-only也不能替代完整raw/metadata/finite/state及路径/制品绑定。

## 3. BF16 cache 与候选块过滤：已有、默认和剩余边界

- **BF16 cache已实现，不是新的>8192切换。** `src/myds_v41.c:939` 的 `V41_CACHE_BYTES=2`，`2932–2934` 为compressed/index分配2B元素持久缓存；逐token `3316–3320` 是rope→quant(mode2)→pack_bf16，batch `3455–3460` 已融合或回退到相同pack。`metal/dsv4_misc.metal:8078–8086` 明确先完成FP4/dequant/BF16舍入再以高16bit写ushort；reader `1669–1679` 对cache_f162按BF16精确展开。decode `src/myds_v41.c:3555–3559`、prefill `3779–3780` 已显式使用BF16缓存。Gemini `184–185` 的“再同步一份BF16、8192后才启用”会重复已有设计。

- **候选mask和提前跳过评分也已实现。** `src/myds_v41.c:3561–3573` 在layer20生成Top2048 *blocks* mask，后层使用它并保留原filter。当前head-slots实现 `metal/dsv4_misc.metal:7449–7469` 在加载K之前，对mask非零行写−INF并return；reference direct同样在 `2086–2105` 提前跳过。`src/myds_v41_moe.c:2644–2656` 在实际pipeline传入这个mask；`2637–2638` 在≤2048个block且KVreuse时可省全选mask。因此不能声称当前默认总是在所有非候选行完成dot后才过滤。

- **真正减少TG网格的selected-block scorer也已经有代码，但default-off。** `src/myds_v41.c:3545–3553` 仅普通单token、il>20、count>16384、无trace且 `MYDS_INDEXER_COMPACT_SCORER=1` 时选择；结合 `2689–2692`，实际是layer24/28/32/36。README `509` 明确关闭和适用边界。host `src/myds_v41_gpu.c:554–595` 先preflight两个PSO，再fill dense scores为−INF，按原block IDs派发selected_blocks×8行。shader `metal/dsv4_misc.metal:2026–2068` 保留原行ID、dot/simd_sum及head累加，原dense TopK网络/同分规则不动。缺PSO在写前返回0可走原路径；dispatch开始后失败返回−1，不应重复原路径。

- **Gemini的阈值单位错误。** `2048`是block budget，每block8rows；默认全选情况下，压缩行数≤16384时并没有过滤收益。不是“rows>2048”即可跳掉大部分。其理论机会从count>16384且实际mask非全选才开始；普通后层ratio1对应位置也须按真实count解释。

- **组件测试代码存在，不等于当前GPU/全模型采纳已闭合。** `tests/test_v41_gpu.c:242–350` 覆盖1/7/8/9/33/16385/32769/131077行、全选/全拒/部分mask、反序block IDs、完整scores+orderedTopK、非零subview/guards、窗口下界以及缺PSO/alias/shape拒收。本次未运行它，也未找到compact scorer新的冻结全模型采纳证据；最新两批正式P均不以它为候选。不能把default-off工具存在写成已获收益。

- **长prefill mask-aware仍是后备不同机会。** 当前batch path `src/myds_v41.c:3779–3801` 先dense score，之后filter；TILED args不带mask。账本 `docs/2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md:11653–11656` 已列出该机会和>16384/实际mask非全选前提。先只读记录每层每query mask及8×32 tile覆盖，估计可跳tile/额外调度，再决定实现。不能把已有scalar compactor直接搬入prefill，也不能减少TopK输入宽度而改变dense tie网络。

## 3A. 4ff3af6 的 Top512 与当前 myds 的对应路径

官方完整提交 [4ff3af604facebc487772613751f63319bd7dfa8](https://github.com/kernelpool/ds4/commit/4ff3af604facebc487772613751f63319bd7dfa8) 合并了几项不同变换：block内SIMD shuffle排序、merge只保留前512、radix-select的512赢家单pass排序，以及每token复用候选block list。提交消息明确只保证**无NaN输入**的选择ID及顺序相同，V4 Flash路径不变；这不能转换成myds含NaN/同分/所有backend的一般0-ULP合同。

- **prefix512 merge已有，而且最新冻结配置显式开启。** 实际路径是 `myds_v41_gpu_topk()`→`topk_rows_causal()`，入口 `src/myds_v41_gpu.c:1470–1478`。`960–974` 严格要求 `MYDS_INDEXER_TOPK_PREFIX_MERGE=1`、npr≥16、layout>2×blocktop、top≤512；`975–1011` 还要optional PSO可用、scratch flag分配成功。不是设置flag后每次TopK都走它。
- **NaN分类融合已有。** 同时 `MYDS_INDEXER_TOPK_NAN_FUSE=1` 时由原partition sort写NaN flags，而不是另加完整visible-row分类dispatch；PSO的thread上限不得改变原nth，见 `src/myds_v41_gpu.c:968–1000`。guarded merge在 `metal/argsort.metal:533–554` 对整个row所有原partition的flag做OR；任一可见NaN就从首级merge走dense原网络。未融合分类时的同样gate见 `365–377`。最新K P metadata两flag为1，可查 `docs/benchmarks/2026-10-25-indexer-observer-and-qb-rope/P-business96-A1.jsonl:2`；README `510–511` 记载代码默认0。应分别写“程序默认关闭、当前冻结配置显式开启”。
- **SIMD shuffle实现也已有，但不是已全局替换的生产新收益。** `metal/argsort.metal:51,94–108` 原模板已有shuffle分支，现有host-visible普通desc/causal在 `150–152` 默认shuffle=false；partition-NaN的shuffle新符号在 `691–694`。独立候选 `MYDS_INDEXER_TOPK_NAN_SHUFFLE` 严格0/1、缺席默认0，见 `src/myds_metal.m:207–215`；显式开启后，`src/myds_v41_gpu.c:1472,1478` 才路由到 `topk_rows_causal_nan_shuffle`，其native分区、geometry、range/alias和scratch门见 `1143–1199`。本次没有执行该候选或找到新的正式全模型采纳结果，不能把代码存在写成已获提速。
- **上游“radix选出512赢家再单pass sort512”不是当前prefix merge的同义词。** 当前库没有官方 `kernel_topk_select_sort512` 符号；本地另有 threshold 原型，但它是 `MYDS_INDEXER_TOPK_THRESHOLD=1`（缺席默认0，`src/myds_metal.m:277–286`），只在top=512、8192≤cols≤262144时选，其他走原topk，见 `src/myds_v41_gpu.c:1464–1468`。不能把配置中的通用 `MYDS_TOPK_THRESHOLD=1` 当作这个indexer开关；也不能把两个算法的同分总序假设混用。
- **候选block list复用的思想亦非空白。** 当前compact scorer直接复用layer20原 `g->block_selected`，`src/myds_v41.c:3550–3551`，没有每个后层从mask重建list。它仍受上文default-off/单token/>16384/原denseTopK合同限制。官方构建紧凑候选分数和winner ID映射的整条路径不应直接替代本地dense tie网络。

因此Top512应列为“部分机制已有、部分default-off候选已有、官方sort512/radix路线尚需单列评估”，不能把4ff整项作为全新必收益迁移。任何后续应先用相同旧partition/有序IDs比较，覆盖全同分、signedzero、±INF、每个分区的quiet/signaling NaN与causal padding，明确无NaN快速路径和有NaN原路径。它不与下一轮register-only scorer同时开启，避免无法归因。

## 4. 最新失败结论与它的适用范围

- K-hoist完整身份问题已经关闭，不能继续写成“部署原生身份未解决”。`docs/2026-10-25-prefill-decode-performance-review.md:76–84` 记录完整29shader TU恢复native与部署库的源/IR/完整输出身份；初始失败仍保留。
- 最新K-hoist私有接入仅ordinary nontrace prefill、ratio2、512<ncomp≤16384、8≤tokens≤2048、end≤32768，见 `docs/2026-10-25-indexer-observer-and-qb-rope-review.md:75–77`。正式P `103–119` 数值/状态/采用通过，但两负载prefill只1.591142%/1.514275%，decode门也失败，因此未采纳。它限制的是该实际实现、输入和cohort，不是所有register-only/长上下文优化的上界。
- 最新KV-only decode正式P见 `docs/2026-10-26-decode-kv-only-finish-review.md:49–60`，主decode仅−0.088331%/+0.214242%，也失败。它只说明删除dummy-Q工作没有在本轮达到整体门，不能用来证明MMA无价值或唯一归因频率/带宽/Q_B。

## 5. 下一轮 register-only MMA 的最窄 scope 与 gates

1. **先只做独立组件，不接runtime。** 从当前完整production TU保留原native prefix和全部29shader顺序，末尾追加新symbol；使用当前DSV4 ABI72、F32 Q/W、BF16 cache、32heads、D128、ratio2、8≤tokens≤2048、512<ncomp≤16384、end≤32768、no key-window。保持现有K-hoist和原native/scorer不变。首轮只移除qtg/ktg/dot中转，直接正确坐标fragment读写，保留本地head累加表达式及16block/head次序；不联合改变TopK、mask、预算、dispatch数或无关参数。

2. **先CPU合同再GPU gate。** 验完整ABI offsets/strides、官方coord覆盖、逆编辑恢复native、完整source/args/commands/SHA。允许CPU静态最小原型准备，所有GPU由root独占另冻结。编译后外部部署TILED与同完整TU native身份门必须先过；不能用same-build positive替代外部gold。

3. **GPU synthetic必须比较全scores位模式，不只BF16舍入/TopK。** 重点非8token/32key尾、row_group/pos causal变化、整个SG/TG不可见、全zero/signedzero、极小大值、非BF16F32 Q/W、F32/BF16显式格式边界、NaN/-INF不可见尾、poison、score stride padding、输入before/after和全部canary。窄scope外要明确拒收，不静默计算另一算法。初次bit失败保留raw，不进入timing。

4. **再回放现有真实capture与完整orderedTop512。** 用原真实capture的实际Q/W/cache，全部输出分数、输入raw/guards、IDs一致；保留actual exact、native TILED、K-hoist和新register-only的分层对照。最初可把K-hoist作为描述性消融，但不可用它替代真实默认exact或部署native oracle。

5. **性能观察方式沿已修复匹配复用距离合同。** 新symbol/物理输出布局先独立same-PSO null校准；数据前冻结三臂、conditioner与B同PSO/args/output、C→A1→B→A2顺序及完整逐CB raw。不能复用曾有1.31–1.84%顺序偏移的旧ABA观察者直接宣称小收益，也不能把旧失败cohort重写。收益/漂移阈值由root在新数据前冻结，不从上游41%或旧K的1.5%倒推。

6. **组件通过后才准备独立default-off runtime与Q/P。** 沿现有G_lib/off/on deployed-library bridge、真实selected-symbol与pre-first encoding/no-decode counters、最终full-vocab logits/tokens/output/schema2 state与完整身份门。再独立冻结至少两负载native A1/B/A2，分开prefill/TTFT/decode、保留warm/formal，严格>2%及既定控制/无回退门。native通过也不等于生产；HTTP deployed-executable bridge、取消/恢复等另列，没有自动后续授权。

本memo只建议下轮最窄验证顺序，没有批准迁入生产或任何GPU执行。
