# kernelpool/ds4 ds41-perf 官方来源核验

本轮只做官方GitHub remote的只读git取证和CPU源码审计，没有运行benchmark、Metal编译或GPU。网页工具restricted/timeout后，依据官方 `https://github.com/kernelpool/ds4.git` clone到独立bare mirror；主工作区及其Git refs未写入。所有12个提交均存在且是当前分支祖先，完整patch/源码/SHA另存本目录。

分支HEAD为 `72ccf363174c3e1acf840b3661a956d4fb321071`（2026-10-22），main和merge-base同为 `0aaea5a238fb41a35106a551e73c8409dfb751ac`。以这两个固定SHA计算，main..ds41-perf恰136，反向0；分支全祖先933。136包含8个merge及修复/回滚，并不是136份独立逐位实验。13个commit message显式使用bit-identical/bit-exact/bitwise词语；这不证明其余不精确，也不足单凭计数确立“大多已逐位验证”。本轮审计是指定12提交和136历史范围核验，不声称本机执行过这些GPU测试。

| 完整官方提交 | 原数字/适用条件/差异 | 测试证据范围 |
|---|---|---|
| [0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4](https://github.com/kernelpool/ds4/commit/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4) | M3 Ultra：scorer16K1.31→0.77s；Q8 unpack合并改动后32K860→892、64K853→898t/s。只有局部scorer约41.22%耗时节约，组合全prefill约+3.72%/+5.28%。 | 无新增test diff；当时Apple index-batch采用double oracle相对误差<1e-5，逐memcmp段仅非Apple。 |
| [b66c926046683242238744510250d5ef3461e7c3](https://github.com/kernelpool/ds4/commit/b66c926046683242238744510250d5ef3461e7c3) | Q8 QA5120→1280/KV512；output-B8192→5120；HC4stream；MoE是MXFP4 5120→2304、384experts/选6。没有提交中独立速度数值。 | 新增literal-vs-generic完整memcmp与kernel-run计数；不能把MXFP4直接套当前IQ2/Q2。 |
| [55b1e75e64ec66b458e5cd5cf0944bec20590ae8](https://github.com/kernelpool/ds4/commit/55b1e75e64ec66b458e5cd5cf0944bec20590ae8) | IndexerQ_B额外精确E4M3副本；16B vector loads；一个layer不能exact则全indexer family回F16。只减少这些weight reads，不是全decode带宽减半。 | 新增6144→256和1280→4096 E4M3两passes，subnormal/small codes及BF16router5120→384。 |
| [7d9903b2ed6bd74c14aed06a89a591a14d613e9e](https://github.com/kernelpool/ds4/commit/7d9903b2ed6bd74c14aed06a89a591a14d613e9e) | M3 Ultra、2K decode38.5→39.2t/s约+1.82%；Engram projection849→658us约-22.50%，局部权重读减半。 | 提交无新增test diff；声称logits bit-identical，未附可独立重放raw/cohort。 |
| [f0b4f55fcc3164be9637879fbbc635385ca213d9](https://github.com/kernelpool/ds4/commit/f0b4f55fcc3164be9637879fbbc635385ca213d9) | 每step64 float2=raw32+compressed32，共512B；只旋转64维tail/32pairs，不是head128的64pairs。无提交内速度数字。 | 新增Apple rope-tab synthetic：pos0/1/8191/1048575 × raw/comp，attn64heads×512、NWG32、Q32×128；fullmemcmp。 |
| [64ceeea4f51408a00a7e9f69b0171e1b71292de9](https://github.com/kernelpool/ds4/commit/64ceeea4f51408a00a7e9f69b0171e1b71292de9) | Metal-only额外BF16 index view，wide mode rows>=8192；不是prompt>=8192。保持原F32并增加BF16view，restore必须验证exact，否则F32。 | 测试key counts8200/65535/65536/98303/131077、guard、view on/off/fullmemcmp及non-BF16恢复拒用。 |
| [2d5ef0d3c12885e67b99860daa0032ce366bf419](https://github.com/kernelpool/ds4/commit/2d5ef0d3c12885e67b99860daa0032ce366bf419) | il>20 && ceil(n_comp/8)>2048，即n_comp>=16385；选择2048个8-key blocks，方案文字“n_comp>2048”错误。 | 新增candidate score/select回归；没有提交内M2速度数值。 |
| [e99dc2175175f2eb8af317b09f34989d8db96c5e](https://github.com/kernelpool/ds4/commit/e99dc2175175f2eb8af317b09f34989d8db96c5e) | Engram gate从每stream32threads到256；HC collapse42TG=32+10，两个kernel。工作划分不是76GPUcores满载证明。 | 新增8次fresh random输入HC完整memcmp，gate在unmasked width5120、rows<=16比对；无提交内速度数字。 |
| [4ff3af604facebc487772613751f63319bd7dfa8](https://github.com/kernelpool/ds4/commit/4ff3af604facebc487772613751f63319bd7dfa8) | V4.1 decodeTop512专用shuffle/prefix merge/radix winners sort；原声明相同ID/顺序限无NaN，V4 Flash保持旧路径。 | 新增513/1024/1025/10240/10241/16383/65536 widths、ties/±inf/signedzero/NaN与candidate list reuse；无提交内速度数字。 |
| [bdae7640ef05954fffa11d87360dc8c130aa85fc](https://github.com/kernelpool/ds4/commit/bdae7640ef05954fffa11d87360dc8c130aa85fc) | Q2/512GB Mac、USB SSD、整span约189GiB可装入pagecache，2K198→420、8K214→518t/s。不是192GB只要外置盘就2x。 | 无新增test diff或原始argv/raw；提供WARM_WAIT benchmark开关，无法据消息确定warm3.5min是否计入原数字。 |
| [187f46b4fa1fe358a802832a0c19dffd7759a0e2](https://github.com/kernelpool/ds4/commit/187f46b4fa1fe358a802832a0c19dffd7759a0e2) | 非speculative loop每step可选logits hash/dump在token_t1后；seed1469598103934665603不同于myds14695981039346656037。 | malloc/copy/fopen缺失可静默跳过；DUMP嵌在HASH分支；无完整性失败门，诊断仍会改变后续host gap。 |
| [72ccf363174c3e1acf840b3661a956d4fb321071](https://github.com/kernelpool/ds4/commit/72ccf363174c3e1acf840b3661a956d4fb321071) | 当前HEAD，真实撤回e1d8114预编码下一token方案；held command buffer/per-token GPU spin是原title。 | 删除131行preencode测试；不能把held直接等同已证实deadlock，或推广成所有异步overlap无效。 |

## MMA 与本地 K-resident 的实质差异

官方0f9的MMA直接从device取Q/head weights，以每simdgroup的16个K矩阵fragment驻留，完全不使用原qtg/ktg/dot threadgroup storage或head loop barriers。本地 `tools/build_indexer_k_resident_probe.py:40–66` 只hoist原K simdgroup_load并保留Q-TG与屏障；两者不是同一候选。官方MMA的测量硬件是M3 Ultra，MMA选择本身未设置M3-only guard；不能据此认为M2已获同样逐位和速度结果。

原代码确有dsv41_mma_coord以及Gemini显示的thread_elements/xy映射，不能把这个映射本身判成官方错误；Gemini摘录漏掉helper定义及末尾else写-infinity分支。因此该snippet不是可原样复制的完整kernel。旧官方ABI是GLM batch结构、cache_f16为F16语义；当前myds的BF16 cache mode/72B布局不能直接拿来接它。

原kernel/helper：[0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4 misc:7267–7341](https://github.com/kernelpool/ds4/blob/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4/metal/dsv4_misc.metal#L7267-L7341)；[loader F16语义1640–1648](https://github.com/kernelpool/ds4/blob/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4/metal/dsv4_misc.metal#L1640-L1648)；[selector/ABI/dispatch36994起](https://github.com/kernelpool/ds4/blob/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4/ds4_metal.m#L36994-L37055)；[当时Apple测试范围798–895](https://github.com/kernelpool/ds4/blob/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4/tests/test_deepseek41_metal.c#L798-L895)。

## RoPE table 的尺寸与数学顺序

64个float2是两套频率各32pairs；attention head512和index head128都只对尾64维旋转。table由float(pos)*frequency和precise::sincos计算，compressed consumer使用256B偏移。attention reduce先将v*S舍入BF16，再对尾部逆旋转，使用显式fma，最后再次BF16；indexer新kernel同样先RoPE→BF16，再BF16/FP4 quantize，不能跨过这些舍入边界。表只在TP1、非streaming、非quality、非layer-resident、active commands且需要logits的ordinary step建立，pos<1048576，并带并发encoder内存屏障。它不是把任意当前RoPE表达式换成sincos/FMA就自动0ULP。

[table kernel3578–3590](https://github.com/kernelpool/ds4/blob/f0b4f55fcc3164be9637879fbbc635385ca213d9/metal/moe.metal#L3578-L3590)、[host512B/offset/dispatch49654–49717](https://github.com/kernelpool/ds4/blob/f0b4f55fcc3164be9637879fbbc635385ca213d9/ds4_metal.m#L49654-L49717)、[适用条件43149–43154](https://github.com/kernelpool/ds4/blob/f0b4f55fcc3164be9637879fbbc635385ca213d9/ds4.c#L43149-L43154)、[attention BF16/FMA顺序1440–1465](https://github.com/kernelpool/ds4/blob/f0b4f55fcc3164be9637879fbbc635385ca213d9/metal/flash_attn.metal#L1440-L1465)、[indexer次序211–250](https://github.com/kernelpool/ds4/blob/f0b4f55fcc3164be9637879fbbc635385ca213d9/metal/dsv41.metal#L211-L250)、[synthetic tests771–826](https://github.com/kernelpool/ds4/blob/f0b4f55fcc3164be9637879fbbc635385ca213d9/tests/test_deepseek41_metal.c#L771-L826)。

## Warmer 的2x条件与内存

实际feasibility公式为saturating(m->size + graph_bytes + span + 64GiB) <= physical_memory；只在非streaming graph调用，一process一次尝试，disable env存在即停（值为0也算存在）。代码没有检测“是否外置SSD”，USB只是所报告实验的存储条件。仅189GiB span加64GiB reserve已经超过192GB，所以原gate不会因为192GB机有外置盘就启动。4个utility-QoS reader各用8MiB scratch（总约32MiB），以pread向OS cache读完整span；缓存页约189GiB可回收，不是只花32MiB内存，也没有wire/新表allocation。压力是每64个8MiB块轮询sysctl，不是即时通知回调。

官方消息报告USB约1GB/s、warm约3.5分钟和512GB/Q2数字；没有原始run argv、cache状态或计时边界raw，不能把2x声明为已核cold-start end-to-end latency改善。Q4在512GB排除，原实现也明确保留64GiB reserve。启用benchmark等待开关会在graph allocation等待warm完成，不能省略该成本后仍称整个冷启动耗时。

[fit/once/WAIT gate41114–41147](https://github.com/kernelpool/ds4/blob/bdae7640ef05954fffa11d87360dc8c130aa85fc/ds4.c#L41114-L41147)、[4threads/8MiB/pressure/pread380–457](https://github.com/kernelpool/ds4/blob/bdae7640ef05954fffa11d87360dc8c130aa85fc/ds4_engram.c#L380-L457)、[graph nonstreaming call41198](https://github.com/kernelpool/ds4/blob/bdae7640ef05954fffa11d87360dc8c130aa85fc/ds4.c#L41198)。

## 其余要避免外推的口径

weight narrow只针对通过逐元素/逐block exact重建的router、Engram和indexer投影；复制本身新增resident memory，F32/F16原weights并未据这个提交自动消失。7d的849→658us不是整个decode -22.5%；实际作者给出的整个decode为38.5→39.2t/s。BF16 index view同样是F32 cache之外的额外副本，而且8192是keys/rows，不是prompt长度。candidate筛选2048是8-key blocks，门为ceil(n_comp/8)>2048。多TG/静态shape/Top512这些提交没有各自M2全模型收益数字，应当保持候选和当地适配/测量边界。

187f的FNV-style诊断不检查每一步hash/dump是否齐全，不能代替fullraw/state；即便per-token计时戳在诊断之前，读回/输出仍可能影响下一次调度。方案提供的myds-bench/check-golden或make test-indexer-mma等迁移命令，应当标为拟实现接口，不能因上游commit真实就当成本地已存在验证工具。

源码快照和每commit的完整官方URL在source-manifest.json；136范围与全部message在branch-scope-summary.json。该方案应先修正硬件、维度、阈值、测试和内存前提，再按本地冻结baseline/counter/fullbytes/组件/全模型门评估；这里没有承诺本地收益或宣布采纳。
