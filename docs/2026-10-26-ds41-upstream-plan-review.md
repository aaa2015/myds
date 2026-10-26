# ds41-perf 性能建议审阅与修订路线

2026-10-26。结论：**上游引用真实，方案有可用的新方向，但不能按现稿的“零风险三阶段”直接迁入。优先保留 prefill 的 register-only MMA，以及 decode 的精确窄权重副本；其余先去重、纠正适用条件，再按独立候选验证。** 本次完成原始提交、当前源码和 CPU 合同复核，没有进行新的 GPU 测速、模型转换或服务变更。

审阅对象为 [Gemini 原稿的冻结副本](benchmarks/2026-10-26-upstream-plan-review/proposal-reviewed.md)，SHA256 `e0314a33d21d5058aa8f98640883f07d7b001a76d199211d5f1c806722a4759b`。原路径为 `/Users/fangqingyuan/.gemini/antigravity-ide/brain/3d5b3923-370d-4918-a651-b95c0d4253f9/ds41_perf_analysis_and_adoption_plan.md`，没有覆盖原稿。

当前工作树 HEAD 为 `224dc70a839af1468c68c277384fa489ea55afbd`，带已有未提交修改，不能将 HEAD 当作本次读取文件的完整身份。[CPU 证据](benchmarks/2026-10-26-upstream-plan-review/review-evidence.json)记录了相关文件 SHA、命令原始输出及退出码。上游从官方 Git remote 只读获取，固定到 `ds41-perf` 的 `72ccf363174c3e1acf840b3661a956d4fb321071`；`main..ds41-perf` 确为 136 个提交，但其中包括 merge、修复与 revert，不能据数量推导 136 个独立通过的性能实验。本次逐项核对原稿引用的 12 个提交，完整 SHA、官方 URL、patch SHA 及原始 patch 已保存在[上游清单](benchmarks/2026-10-26-upstream-plan-review/upstream-source-manifest.json)。GitHub 网页读取遇到限制/超时，依据为官方 remote 的 Git objects 与 patch。

## 1. 对路线选择影响最大的修正

### R1：MMA 值得继续，但 -41% 是局部数字，示例代码还有接入阻断项

原稿第 40 行和第 85–151 行。官方 [0f9b03c](https://github.com/kernelpool/ds4/commit/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4) 的 `1.31→0.77s` 是 M3 Ultra 的 16K scorer 局部结果。提交中的整体 cold-prefill 数字为 32K `860→892 t/s`、64K `853→898 t/s`，并且与另一项 Q8 prefill 改动联合测得，不能分别归给 MMA，更不能当作 M2 Ultra 的预期收益。

提交中的 bit-identical 是作者报告，本次未独立执行上游 GPU 验证。更具体地，该提交没有新增测试 diff；其当时的 [Apple indexer fixture](https://github.com/kernelpool/ds4/blob/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4/tests/test_deepseek41_metal.c#L835-L890) 对 double oracle 用 `<1e-5` 相对误差门，`memcmp` reference 段位于 `#ifndef __APPLE__`。因此不能说该提交本身附带了 Apple MMA↔TILED 全 scores 的逐位测试证据。原稿“每个优化均有独立逐位测试”应改成按提交列出作者报告、测试实际合同和本机未验部分。

这项设计与我们刚测过的 K-resident **确实不同**。我们的 [builder](../tools/build_indexer_k_resident_probe.py) 将 K fragments 移到 head 循环外，但仍保留 Q/dot 的 threadgroup 中转及屏障；上游从 device 直接装载 Q 到 matrix fragments，并标量读取 weights，head 循环完全没有 threadgroup 内存往返。K-resident 全模型的约 1.5% 负采纳结果，没有否定这个不同候选。

但原稿的 kernel 不能直接粘贴：

- **丢失必需写入。** 原稿第 111–147 行只有 `row_base < max_visible` 分支；上游 `metal/dsv4_misc.metal:7336–7339` 还有 `else if (live)`，把整块不可见的输出写成 `-INFINITY`。没有该分支，复用 scores 中的旧值或 poison 会留下来。CPU 枚举示例 `n_tokens=8, n_rows=64, max_visible=16` 有 384 个应写 `-INF` 的 cell 未写。这是示例合同缺陷，尚未声称在 GPU 复现了输出错误。
- **同为 72 字节不代表 ABI 兼容。** 原稿使用旧 `ds4_metal_args_glm_indexer_scores_batch`；myds 当前 tiled 分支的 `DSV4` batch 结构字段不同。将当前参数块按旧结构解释，示例会把 `q_token_stride=16384` 读成 `row_group_size`，并把真实 `q_head_stride=512` 读成 `q_token_stride`。需要新的明确结构、字段 offset 检查和 host/Metal 两侧适配。
- **旧加载器不能读当前 BF16。** 该提交的 `glm_cache_load_f32_or_f16` 把非零类型当作 half；myds 使用 type 2 表示 BF16。照抄会把 BF16 位模式当 F16 解码。需要保留真实 BF16→F32 精确加载语义以及当前 producer 的 Q/W 布局。
- **head 累加表达式也不同。** 当前 DSV4 用 `max(s,0)*(w[head]*scale)`，上游旧 GLM 用 `max(s*scale,0)*w[head]`。即使 scale 是 1/64，也不能据实数等式或源码相似就保证 subnormal/overflow/FMA/fast-math 下逐位相同。首版应保持当前表达式和 block/head 顺序，而非顺手复制旧括号结构。
- **几何说明写大了四倍。** 每个 TG 处理 `8 tokens × 32 keys`，四个 SIMD groups 各负责 `8×8`。原稿写成每个 SIMD group `8×32`。原始 lane-coordinate helper 是正确映射；CPU 枚举确认 32 lanes×2 elements 恰好覆盖 64 个不同 cell，不能把缺 helper 误诊为坐标算法错误。
- 当前还有 causal ratio、key-window、stride/span/alias 和适用范围合同。第一版应排除非零 key-window；它本身会改变检索语义，当前默认关闭。空 batch 也必须由 host 拒绝，不能让 `last_token-1` 下溢。

[ABI/覆盖 CPU 证据](benchmarks/2026-10-26-upstream-plan-review/prefill-abi-coverage-audit.json)保留了具体字段和枚举结果。当前 selector 见 [src/myds_v41_gpu.c](../src/myds_v41_gpu.c):665 起，不能绕过真实默认 provider 而只测新建的 synthetic PSO。

此外，原稿的基线描述已过时。[10月25–26日 K-resident 正式结论](2026-10-25-indexer-observer-and-qb-rope-review.md)已经完成：native 身份、实际采用、全模型 logits/state/output 门通过，两条长请求的 prefill 保守节约为 **1.591% / 1.514%**，未达冻结的严格 >2% 门，decode 无回退门也失败，故未采纳。不是仍卡在制品身份或数值资格。组件相对 actual-exact 约 40% 的改善没有转成同幅度全模型改善，正说明新候选仍需要独立完整验证。

### R2：Engram 全表 warmer 的硬件条件被外推错了

原稿第 30、47、192–196 行。官方 [bdae764](https://github.com/kernelpool/ds4/commit/bdae7640ef05954fffa11d87360dc8c130aa85fc) 的显著收益来自 **512GB Mac 的 Q2 模型**。`ds4.c:41122–41138` 在启动前计算模型、graph、表跨度及 **64GiB reserve** 是否一起放得下；不满足即跳过。189GiB 表加 reserve 本身已超过 192GB 级机器的容量，不能因“外置 SSD + 192GB”就推导 2 倍收益。

这也不是免成本的首请求加速。原提交描述 USB 顺序预热约需 3.5 分钟，测速可显式等待 warmer 完成；`cold prompts` 不等于页缓存仍冷、也不等于首请求含预热时间的用户体验。页缓存可回收，压力退出不能保证已读页长期保留。在内存放不下时，顺序扫描还可能与模型页及实际随机请求争用资源；这是风险推断，本次未测实际淘汰量或吞吐影响。

当前 myds 已有独立 Engram `pread` descriptor、持久 worker、默认多线程及 split/byte-split 表合同，见 [include/myds_engram.h](../include/myds_engram.h):49–84、[src/myds_engram.c](../src/myds_engram.c)。这不等于已有全表 warmer，但意味着迁移需适配现有 reader 和分片语义。`F_NOCACHE` 也不能单独证明物理 SSD miss。**在当前 192GB 目标上撤下全表 warmer 的优先级**；以后若研究有界/需求式预读，应另立不同候选，先证明实际 I/O 瓶颈及完整内存预算。

### R3：窄权重是新的 decode 机会，但“整个 decode 带宽减半”不成立

原稿第 22、42、175–180 行。上游 [7d9903b](https://github.com/kernelpool/ds4/commit/7d9903b2ed6bd74c14aed06a89a591a14d613e9e) 和 [55b1e75](https://github.com/kernelpool/ds4/commit/55b1e75e64ec66b458e5cd5cf0944bec20590ae8) 值得保留。前者报告 M3 Ultra 的 Engram projection `849→658us`，整体 decode `38.5→39.2 t/s`；后者增加 indexer Q_B 窄副本和宽加载。减半的是对应权重表示的字节量，不是 MoE、attention、KV、其他投影及调度在内的整个 decode。

当前源码尚未接入这组 BF16 Router / E4M3 权重副本：

- [src/myds_engine.c](../src/myds_engine.c):259–270、905–913 直接包装 GGUF tensor，并保留原 router type。
- [src/myds_v41.c](../src/myds_v41.c):2731–2741、2807–2809 对 indexer Q_B 要求 F16，再包装原始 tensor。
- [src/myds_engram.c](../src/myds_engram.c):722–730 要求 Engram projection 为 F16。
- 已有 F16 GEMV→BF16 **输出**融合不等于 E4M3 **权重**读取。

这里适合先做不碰 GPU 的真实模型表示资格审计：逐元素/逐 block 检查 router 的 F32→BF16→F32、projection 的 F16→E4M3+scale→F16 是否原始 bits 完全一致，记录全部相关层、张量类型/形状及失败位置。不能只因 GGUF type 或源 checkpoint 的一般描述就认定本文件可无损转换。任一张量不满足时必须保持原表示，不能自动近似量化。接入 16B load 时还要保留 K 遍历、乘加、SIMD 归约和输出舍入顺序。

“额外数百 MB 完全安全”也应改成预算门：原权重是否仍映射、窄副本与 scales 的实际大小、graph/KV/并发 scratch、page cache、模型和 context 生命周期均需计算并验证，不能由 192GB 名义容量减去一个历史 resident 数字得出。

按当前目标形状计算，8个 index-QB 的 E4M3 code payload 为40MiB，2个 Engram projection 为300MiB；若40个 router 实际为F32且全部可逆，再加150MiB BF16 payload，共 **490MiB（513,802,240 B）**，另计 block scales、元数据、分配对齐及转换时的 staging/临时双份缓冲。原矩阵若继续服务 prefill，这些是新增 allocation，不是整机内存减少。这里是维度/格式计算，没有验证目标权重可逆性或当前物理可用空间。

### R4：固定形状方案混入了已实现路径和不支持的格式

原稿第 153–157 行。myds 已有 Q4 literal shapes，包括 LM head `129280×5120`、attention Q_B `32768×1280`、output-B `5120×8192` 及固定 NB，见 [metal/myds_q4_v41.metal](../metal/myds_q4_v41.metal):29–42、105–117 和 [src/myds_decode_candidates.inc](../src/myds_decode_candidates.inc):188–221。最近真实模型使用 dense-Q4 的 NB 路径；不能把这些重新算作待获得的收益。

原稿举的 QA/KV Q8 pair 和 output-B Q8+HC 不直接覆盖当前 Q4 模型。MXFP4 MoE 更不适合直接接入：当前 gate/up 支持 IQ2_XXS 等，down 支持 Q2_K 等，[src/myds_engine.c](../src/myds_engine.c):245–256 没有 MXFP4 类型。真实普通 FFN/HC 路径还有 IQ2_XXS GU、Q2_K down、Q4 shared 的具体准入合同。

固定维度的思路可以用于尚未特化的真实热点，但应先列出实际 selected symbols、类型和残余动态寻址。现有账本的 QA/KV 并发融合负结果、Output_A wrapper 全模型约 0.36%/0.40% 结果也要列入，见[优化账本](2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md) §74 及后续 Output_A 记录。编译常量化不自动减少 CPU dispatch，更不保证整体提速。

### R5：RoPE 与多 TG 规约仍有新部分，但都不是零风险

原稿第 159–169 行。myds 已有默认采用的 attention inverse-RoPE 融合；[src/myds_v41.c](../src/myds_v41.c):4405–4438 根据 `attention_rope_consumed` 跳过独立 inverse，register/RF shader 保留 BF16 边界及显式 FMA。因此应该审计新增的**每步系数共享表**，不要重复实施已存在的逆旋转融合。

上游 [f0b4f55](https://github.com/kernelpool/ds4/commit/f0b4f55fcc3164be9637879fbbc635385ca213d9) 的 64 个 `float2` 为 **raw 32 + compressed 32**，各服务于 64 维 rotary tail 的 32 对；不是原稿所写“128维 head 的64组”。myds attention head 为512、indexer head 为128，二者只旋转末64维。表需保持原 GPU `precise` trig、频率和位置来源、inverse 符号、BF16→旋转→BF16 次序，并处理 ratio2 compressor commit 的 `pos-1`。不能把所有 consumer 都绑定当前 `pos`。新增 producer dispatch 和资源依赖也必须计入成本。

已有反例提示收益必须实测：Q_B→forward RoPE v1 在 M4/M2 数值失败；刚完成的 dummy-Q 删除候选虽删除每 token 131,072 次源码层面 trig 调用，完整 decode 仅 **−0.088% / +0.214%**，未达采纳门，见 [KV-only 正式报告](2026-10-26-decode-kv-only-finish-review.md):53–60。这没有证明共享表无效，只否定“少算 trig 必然大幅加速”的推理。

HC 方面，现有默认是 fused weighted-sum+RMSNorm，见 [src/myds_v41.c](../src/myds_v41.c):3977–4000；[metal/dsv4_hc.metal](../metal/dsv4_hc.metal):1155–1188 保持四路累加→BF16→原 lane 的平方和→32 SIMD partials 的顺序。尚未接入上游 [e99dc21](https://github.com/kernelpool/ds4/commit/e99dc2175175f2eb8af317b09f34989d8db96c5e) 的 42TG collapse。它是另一个可测候选，必须保留当前线程到元素/partial 的映射，不能任意拆 K 或直接套用上游的旧 reduction tree。多 kernel、scratch 和 barrier 的开销也可能超过并行收益。42 个 TG 的数量不能证明 76 个 GPU 核心满载；提交未给 M2 全模型收益。

### R6：BF16 cache 已有，candidate 阈值单位写错，Top512 需要按已有实现去重

原稿第 184–186 行。myds 已经全程用两字节 BF16 cache：`V41_CACHE_BYTES=2`、index cache 分配及 `pack_bf16` 写入，见 [src/myds_v41.c](../src/myds_v41.c):939、2610–2611、3258 起。不存在等待 ≥8192 才能首次获得的 F32→BF16 带宽减半。

decode masked scorer 也已在读 K/cache 之前跳过非候选行。进一步压缩 dispatch 网格的 `MYDS_INDEXER_COMPACT_SCORER` 已存在、默认关闭，见 [src/myds_v41_gpu.c](../src/myds_v41_gpu.c):554–594 和 [README](../README.md):509。上游 [2d5ef0d](https://github.com/kernelpool/ds4/commit/2d5ef0d3c12885e67b99860daa0032ce366bf419) 判断的是 `(n_comp+7)/8 > 2048`，即 **超过2048个8-key block，至少16385条压缩行**，不是超过2048条行。myds 当前 compact 限于 layer24/28/32/36、count>16384 的普通单 token decode；不能扩成所有 Layer>20 或 prefill。

Top512 方面，上游 [4ff3af6](https://github.com/kernelpool/ds4/commit/4ff3af604facebc487772613751f63319bd7dfa8) 的比较保证明确限定无 NaN。myds 已有 SIMD shuffle 模板、prefix merge、NaN 分类/融合、threshold 等实现，见 [metal/argsort.metal](../metal/argsort.metal):94 起、[src/myds_v41_gpu.c](../src/myds_v41_gpu.c):960 起、[README](../README.md):510–511。**prefix merge / NaN-fuse 在代码中默认0，但最近冻结配置显式设为1**，见 [P-business96-A1 metadata](benchmarks/2026-10-25-indexer-observer-and-qb-rope/P-business96-A1.jsonl) 第2行；它们仍受分区数、shape、PSO/scratch 等实际分发门限制。独立 NaN-shuffle 和 indexer threshold 候选仍关闭，不能把通用 `MYDS_TOPK_THRESHOLD` 当成 `MYDS_INDEXER_TOPK_THRESHOLD`。

上游 radix 赢家的单次最终512排序在本地没有同名 provider；prefix merge 也不是它的同义词。candidate block-list 复用思想则已出现在本地 compact scorer 中。下一步应只列剩余算法差异，单项测试；ties、±Inf、signed-zero、NaN 和有序 IDs 都属于合同。不能把“源码已有”写成“每次实际采用”，也不能把整组当全新必收益大改。

### R7：哈希适合定位分歧，不能替代逐字节门；当前验证命令不完整

原稿第 198–230 行。上游 [187f46b](https://github.com/kernelpool/ds4/commit/187f46b4fa1fe358a802832a0c19dffd7759a0e2) 有逐 step logits hash **和 raw dump**。myds 现有 [fullmodel driver](../tools/bench_output_a_fullmodel.c):741–796 已有 final-logits FNV、finite 与 raw 导出；逐 step 观察仍可补充，但不能只加一个 64-bit hash 就声称“100%逐位一致”。哈希有碰撞；最后一步一致也不能证明中间每一步/缓存一致。

实现时应绑定 step/frontier、元素数、provider、实际符号和完成状态，缺行、分配/复制/写盘失败必须拒收；异常时保留完整 raw 以定位首个分歧，继续保留 logits/state/output 的 byte comparison。不同实现还需统一 seed/算法版本：该上游提交的 seed 为 `1469598103934665603`，当前 myds 为 `14695981039346656037`，不能直接比较同名哈希。

把复制/打印放在单步 timer 后，也可能改变下一步前的 host 间隙。因此逐 step observer 用于数值诊断；正式性能窗口必须关闭它，沿用普通异步路径。

本次实际 CPU 执行结果：

| 原稿命令/路径 | 当前结果 | 需要修改 |
|---|---|---|
| `make -n test-indexer-mma` | exit2，No rule | 明确列为待新增测试工具，而非现有 target |
| `make -n test-decode-literal-shapes` | exit2，No rule | 同上，并按当前 Q4/IQ2/Q2 格式命名/构建 |
| `make -n test-split-collapse` | exit2，No rule | 同上，必须包含当前归约树的 oracle |
| `./bin/myds-bench` | 当前文件不存在 | 为真实 native driver 明确 build、argv 与输入 |
| 原稿 `python3 eval/stream_bench.py ...` | exit0，stdout/stderr均空 | 这是被 import 的库，无 main/argparse，不发 HTTP 请求 |

`eval/decode_fixed.py --help` 和 `eval/prefill_decode_bench.py --help` 均可运行，其 CLI 使用端口位置参数和 `--tokens` / `--decode-tokens`，不是原稿的 `--url` / `--max-tokens`。这些 HTTP 工具只适合筛查；TTFT 和 visible SSE decode estimate 不能代替 native prefill/decode 的正式计时，也不能用跨工程的 `ds4 oracle` 绕过同设备/同模型/同语义的 baseline 身份桥。

原稿开关清单也尚未实现，而且 warmer 段落的 `MYDS_DISABLE_ENGRAM_WARM` 与末尾 `MYDS_OPT_ENGRAM_WARM` 命名不一致。下一版需明确开关、解析时机、默认值、actual counters 和失败策略，不能用环境变量存在作为候选已执行的证明。

## 2. 修订后的优先级

| 原建议 | 当前判断 | 下一步 |
|---|---|---|
| register-only MMA scorer | 新的 prefill 候选，优先 | 先修 ABI/BF16/masking 合同，再独立私有 probe |
| Router BF16 / projection E4M3 | 新的 decode 候选，优先 | 先全张量 exact round-trip 审计，再预算与 reader 接入 |
| Decode literal shapes | 部分已采用，部分类型不适用 | 只查实际热点中剩余动态工作；不迁 MXFP4 代替 IQ2/Q2 |
| 每步 RoPE 系数表 | 新增部分可研究；inverse 融合已有 | 限定正确位置/频率族，保留舍入，统计新增 producer 成本 |
| HC 42TG / Engram gate 分摊 | 未接入，存在收益与回退两种可能 | 按当前 reduction tree 做单独候选，排在前两项之后 |
| BF16 index cache | 已实现 | 不再作为新收益项 |
| candidate scorer / Top512 | prefix/NaN-fuse当前配置开启；compact/shuffle等有关闭候选 | 按实际符号、eligibility与负结果去重后，再决定剩余单项 |
| Engram 全表 warmer | 当前192GB目标不满足上游预算门 | 撤下当前实施路线；有界 I/O 假设另立项目 |
| 逐步 hash/raw witness | 有工程价值，final部分已有 | 补诊断完整性，保留 raw/state byte gate，性能测量关闭 observer |
| pre-encode spin revert | 负例有效，外推过大 | 保留具体失败经验，不据此证明全部异步编排无优化空间 |

DSpark 可继续维持独立低优先级。具备完整 pipeline 或支持 MXFP4，不代表当前模型能减少 target forward，也不代表本机的候选接受率/最终时延已通过；应保留此前真实负结果。

## 3. 可执行的下一轮实验边界

**Prefill 第一候选：当前 BF16 布局上的 register-only MMA。** 第一版只覆盖已捕获且可复用的 ordinary nontrace、ratio2、32 heads×128 dimensions、`512<n_comp≤16384`、`8≤tokens≤2048`、`end≤32768` 和原实际 batch/stride；key-window=0，排除 prefix/spec/trace 等未覆盖路径。采用新 symbol，保持关闭为默认，不能与旧 K-resident 联合开启。保持16个dim-block和32个head的加法次序，显式写完整 future/masked `-INF`，验证 tail、短batch、边界、alias、guard、缺PSO和dispatch failure。

在 any GPU window 前冻结数值 oracle。上一次 candidate/TILED 全 scores 一致，但 layer2 的 actual-exact 仍有4个原有 score cell 差异、有序Top512相同；这必须明示，不能把与TILED的0-ULP偷换成与当前 actual-exact 的0-ULP。若本轮冻结门要求 actual-exact 全 scores 逐位一致，就按该门裁定，不能事后放宽。全模型 logits/state/output bytes 继续要求完全一致。

组件测量沿用已验证的 matched-output reuse 校准：先同PSO null，再做实际baseline、新MMA和必要TILED桥的独立比较。保留所有原始样本、actual PSO/geometry/counters、输入前后见证及库/源码身份。M4只做设备自身数值资格，M2用M2自身baseline；两设备不能互作0-ULP oracle。

**Decode 第一候选：精确窄权重 reader。** 将表示资格、内存预算、reader数值合同、真实selector/counters分开完成；按权重家族独立开关/实验，不一次叠加全部副本、宽加载、RoPE、HC。保持实际 IQ2/Q2/Q4 模型和当前并发图，证明优化权重确实被读取，而不只是构建出副本或因准入条件回到原路径。

两条候选各自通过组件/全模型资格后，才进入同binary的 native A1(off)→B(on)→A2(off)。每臂 warmup1+formal3，使用数据前冻结的两类工作负载，并明确标注合成或生产来源；此前的 business/repeated-code benchmark 不代表真实生产流量。prefill 与decode分别计量，绑定完整 argv/env、binary/metallib/source/model/request，记录实际 target forwards、fallback/error、raw logits/state/tokens/output。沿用对应采纳协议的严格 >2% primary收益、≤1%控制漂移及secondary ≤0.5%回退要求；失败保留原样，不靠删样、阈值调整或多轮挑优救回。通过后再安排HTTP及相关状态/恢复门，最后才谈生产采用。

以上是本次源码审阅支持的下一步，**尚无新 GPU 性能结果，也没有据上游数字承诺 myds 提速幅度**。本次只新增审阅文档与证据副本，未把任何候选写入运行时。完整文件身份见[证据索引](benchmarks/2026-10-26-upstream-plan-review/evidence-index.json)，分项复核见[prefill memo](benchmarks/2026-10-26-upstream-plan-review/prefill-source-review-memo.md)、[decode memo](benchmarks/2026-10-26-upstream-plan-review/decode-source-review-memo.md)、[上游原始提交审阅](benchmarks/2026-10-26-upstream-plan-review/upstream-review.md)。
