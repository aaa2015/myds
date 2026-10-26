# Register-only MMA 与窄权重 CPU 审计

2026-10-26，进行中。承接[上游方案审阅](2026-10-26-ds41-upstream-plan-review.md)，先完成最小原型与数值资格，再决定实机全模型验证。所有候选保持私有/默认关闭，生产推理文件和配置没有由本任务改动。

## 当前结果

| 项目 | 已完成的证据 | 当前结论 |
|---|---|---|
| 窄权重 CPU 表示资格 | 实际 M2 模型全部50张量、435,159,040个值、1,027,604,480源字节 | 全部 finite，原 bits→窄编码→原 bits 完全一致 |
| MMA v1 | M4，12case×6provider＝72CB，1106完整raw，独立reader有效 | 12case均有候选分数位差，失败保留，不做性能计时 |
| MMA head-control v2 | M4与M2各自12case×6provider＝72CB、1106raw | 两设备分别通过完整TILED逐位门及exact身份桥 |
| 真实capture＋原Top512消费链 | CPU工具准备中 | 尚未完成GPU，不具备全模型资格 |
| 私有runtime/全模型Q/P | CPU准备中 | 尚未加载模型或运行性能窗口 |

当前记录尚无新性能收益，也没有生产采纳。后续阶段的实际结果将另列，不能由前面组件门推导。

## 基线与边界

工作树 HEAD `224dc70a839af1468c68c277384fa489ea55afbd` 带既有未提交修改。先冻结557文件副本，完整29shader TU 为1,574,167B，SHA `bc6603c97ec24af7597f6b10c7ed2a7bfe1344379733a0a4df780250a13f3224`，与目标canonical Metal源码一致。[源清单](benchmarks/2026-10-26-register-only-and-narrow-audit/baseline-source-manifest.json)。

开始时M2 binary/metallib/Metal源码/AIR/62项配置均匹配上一轮身份。随后只读核验发现外部工作将canonical配置改为63项、默认1M上下文：SHA `735e0dc73fbbf7483dba0bffe4ad379a4dab910bb734d3a6708df65c8edc16d7`；binary `2b4d5695…`、metallib `38371023…`、Metal源码 `bc6603c9…` 和AIR `09d67ef8…` 没有变化。原PID69692服务随后已不存在，本任务未向它发送停止/重启信号，也不声称运行服务健康或已恢复。

组件固定输入不受此配置变化影响。真实capture回放继续使用捕获时的62项配置 `0a3524b0…`，保证原consumer桥；新的全模型阶段将另行冻结最新63项配置、context1048576和两个有明确来源的长请求，不把不同配置的时延混算。

## 窄权重：实际模型的完整CPU审计

工具为 [audit_v41_narrow_weights.py](../tools/audit_v41_narrow_weights.py) 与 [CPU C worker](../tools/v41_narrow_weights_worker.c)，仅Python标准库和系统clang，不链接Metal或推理运行时。按原上游encoder及32×32 block的 `e0/e0+1/e0−1` 次序做exact转换，另检查shader等价的F32权重重建；有失败时继续扫描余下块/张量，不以量化误差阈值替代逐位比较。

模型：`/Users/fangqingyuan/hf-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf`，362,142,629,888B。只读GGUF头部和全部目标payload，**未重算整个GGUF payload SHA**。

首版工具在读取头部后拒收：使用标准`block_count`而漏掉真实loader支持的`num_hidden_layers`。该次未扫描权重，不是权重数值失败。原[拒收JSON](benchmarks/2026-10-26-register-only-and-narrow-audit/narrow-v1-header-rejection.json)保留；v2按真实loader加入alias支持、冲突/类型/wrap拒收，并收紧精确architecture，原C encoder完全未改。

v2完成实际M2全量扫描，exit0，用时6.33秒。25项CPU fixture通过，包含独立原上游block selector、signed zero/subnormal、非finite、全量失败统计、GGUF和split、输出alias及保留实际编译制品。实际结果如下：

| 家族 | 实际类型/数量 | 元素数 | 源payload | 窄code/BF16 payload | scales |
|---|---|---:|---:|---:|---:|
| Router | 40×F32，5120×384 |78,643,200|300MiB|150MiB|0|
| Indexer Q_B | 8×F16，1280×4096 |41,943,040|80MiB|40MiB|163,840B|
| Engram projection | 2×F16，6144×25600 |314,572,800|600MiB|300MiB|1,228,800B|
| 总计 |50张量|435,159,040|980MiB|490MiB|1,392,640B|

348,160个FP8 block全部成功。所有张量的完整源流SHA与重建流SHA一致；失败块、roundtrip差异、shader-F32重建差异、非finite均0。包含12,226,883个subnormal和7,300个负零，非仅零值/普通值抽样。代码/BF16加scales合计 **515,194,880B＝491.328125MiB**；原矩阵若继续用于prefill，属于新增allocation，另计metadata、对齐和转换暂存。

完整[审计报告](benchmarks/2026-10-26-register-only-and-narrow-audit/narrow-v2-audit.json)、[实际argv/退出状态](benchmarks/2026-10-26-register-only-and-narrow-audit/narrow-v2-execution.json)、[root复核](benchmarks/2026-10-26-register-only-and-narrow-audit/narrow-v2-root-review.json)、[独立复核说明](benchmarks/2026-10-26-register-only-and-narrow-audit/narrow-v2-independent-REVIEW.md)。完整工具/实际worker dylib/源码/patch/编译日志包为M2 `/private/tmp/myds-register-only-20261026/narrow-v2-complete.tar.gz`，SHA `824785f76223d7f4fb650224d6896345b29bf0d39191ed1bbbee3bc59c1e4eea`，本机也有副本。独立复核检查报告、完整artifact和统计的一致性，没有再次从远端读取1GB权重，不能称第二次独立payload重哈希。

这关闭的是**目标权重可无损窄表示**这一前置条件。GPU reader的乘加/归约、真实采用、内存预算和整体性能仍未验证；没有据此把decode带宽或时延写成减半。

## MMA：首轮失败与有依据的单一修正

首版新增独立symbol `kernel_myds_indexer_scores_register_only_bf16`，使用当前DSV4 ABI72和BF16位展开；8tokens×32keys/TG、4SG×32lane，candidate静态/动态TG memory均0。范围为H32/D128、ratio2、`512<n_comp≤16384`、`8≤tokens≤2048`、end≤32768、key-window0，未接到生产runtime。

原始 [v1片段](../tools/indexer_register_only.metal.inc) 保留本地source表达式，却仍在M4发生数值位差。12case轮换BF16可表示F32、完整F32、负零和subnormal，覆盖dense/padded strides、整块不可见、尾token/key。每case6provider：部署TILED、完整native TU TILED、追加probe TU原TILED、candidate、部署exact、完整native TU exact。首轮三个TILED控制全相同，两个exact全相同；72CB全部完成，inputs/guards/padding/causal/finite均通过，失败只来自candidate分数bits。

[原始独立裁定](benchmarks/2026-10-26-register-only-and-narrow-audit/M4-synthetic-v1.review.json)、[root分类](benchmarks/2026-10-26-register-only-and-narrow-audit/M4-synthetic-v1-root-summary.json)、[第二份raw审计](benchmarks/2026-10-26-register-only-and-narrow-audit/MMA-v1-independent-memo.md)。不把该轮重判为通过。

失败后的CPU定位固定dot，只改变head舍入方式：前两case全部36个可见cell，分开`mul→mul(scale)→add`复现TILED，head FMA复现candidate。AIR中二者head仍带fast flags，单凭源码括号不能约束backend。此为诊断线索，未称GPU中间值已证明唯一根因。

v2只改两处head epilogue：局部`contract(off) reassociate(off)`，显式`relu(dot)*weight→*scale→+acc`；MMA/加载/维度/16-block次序、完整native前缀和全局flags不变。两处[逆变换证明](benchmarks/2026-10-26-register-only-and-narrow-audit/MMA-source-transform.json)恢复v1精确bytes；[AIR检查](benchmarks/2026-10-26-register-only-and-narrow-audit/MMA-head-control-v2-ir-review.json)确认head的4mul/2add不含contract/reassoc。v2 candidate SHA `a62922c6f5937236bd0e354dd06943b39bfa5f788419c16a8700da793769d616`，probe库 SHA `fe55793346e15119fce4e0f40d4bf6d3023fc3694ffc231e66b8a89e3ae0f74d`。

v2在M4和M2各自重跑完整12case/72CB。每台都以本机执行的原部署库和native控制为oracle；未用M4输出作M2 oracle。两台candidate均与三TILED控制完整分数逐位相同，exact身份桥也通过；全部1106raw、冻结host输入、参数、PSO/geometry、输入后镜像、guards及CB完成门通过。两台窗口均未收集timing，库/源码/协议前后身份未变。[M4独立结果](benchmarks/2026-10-26-register-only-and-narrow-audit/M4-head-control-v2.review.json)、[M2独立结果](benchmarks/2026-10-26-register-only-and-narrow-audit/M2-head-control-v2.review.json)。

完整库/binary/harness和33个compiler inputs在两设备同路径的 `/private/tmp/myds-register-only-head-control-build-v2-20261026`，实际GPU raw分别位于各自 `/private/tmp/myds-register-only-mma-20261026/M4-head-control-v2`、`M2-head-control-v2`。

## 后续门

下一步验证原32K模型capture的三层完整scores和原GPU有序Top512。必须明确actual-exact与TILED既有4-cell差异，不将TILED逐位一致偷换为actual-exact逐位一致；有序IDs、全模型logits/state/tokens/output仍需完全一致。

私有runtime仅CPU准备，默认关闭且单BF16 symbol；eligible缺PSO/dispatch错误fail closed，正常不适用继续原路径，decode candidate计数必须0。全模型新Q/P将使用新63项配置/1M容量、两类注明合成来源的长请求、N256。Q为G_lib/off/on各warm1+verify1，共12代；Q通过才冻结P，每臂warm1+formal3，共24代。P primary prefill保守节约严格>2%，prefill/TTFT/decode控制漂移≤1%，secondary无回退≤0.5%，所有raw数值/状态和实际采用必须通过。此处仅记录数据前计划，不代表已运行或通过。
