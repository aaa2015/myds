# Prefill indexer K-resident：仅接入设计，等待新组件门

2026-10-25。本文件是只读源码审查后的**拟实施方案**；没有据此修改 `src/`、header、生产 shader 注册或配置，没有构建新 runtime，也没有执行 GPU。只有 root 明确确认新的组件 R 门通过并触发实施后，才可进行下述改动。已有 Q_B v1 与其 build 全部保持冻结。

当前实际默认 chunk provider 是 `kernel_myds_indexer_score_chunk_exact`，冻结部署库 SHA256 为 `3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9`。不能先开启 `MYDS_INDEXER_TILED` 来改变控制。候选是 exact 默认图中显式选用已验证结构的 TILED/K-resident provider；约 40% 的旧组件读数包含 exact→TILED，而约 13% 才是 TILED→K-resident 的增量。旧正式 cohort 的联合漂移门未过，不能作为本次接入授权。

## 1. 当前真实调用链与不可改变的边界

- `src/myds_v41.c:6996` `myds_v41_prefill_tokens` → Step 4 `:7867` `publish_indexer_chunk`。
- `publish_indexer_chunk` 在 `:3696` 读取 scratch 的实际 `batch_floats`；`:3648–3660` `indexer_group_len` 按 `glen × max(512,last_visible) <= batch_floats` 分组。配置 2,097,152 floats 原值保持不动。真实 capture 的 `128×15424, pos0=30720, ratio=2` 来自这个分组，不切成假想 `256×16384`。
- `:3704–3712` 保留 ratio2 首个零可见行 peel；`:3749–3752` Q 的原地 RoPE/quant 已经开始，因此 scorer 后续失败必须返回 `-1`，不得回退重做。
- 唯一 production chunk scorer 调用是 `:3779–3780`，目前调用 `myds_v41_gpu_indexscores_chunk_bf16`。失败立即 `return -1`，在 `:7872–7874` 到达 prefill fatal path。
- `src/myds_v41_gpu.c:665–747` `indexscores_chunk` 当前先校验基本范围，再按 TILED / exact / direct 选择。`:681` 的原 TILED 条件为 `TILED>0 && n_tokens>=8 && key_window==0`；SG 屏障变体另由 `:705–706` 决定。
- 原 TILED ABI 是 72B（有效字段至 offset64 的 float scale，尾部 padding4B），grid=`ceil(n_comp/32)×ceil(n_tokens/8)×1`，threads=`32×4×1`，dynamic TG memory=`21504B`。字段与 shader `metal/dsv4_misc.metal:84–97` 对齐。
- 原生 TILED 模板在 `metal/dsv4_misc.metal:6617–6767`；现有 private probe library 已追加 `kernel_myds_indexer_scores_k_resident_f32` 和 `_bf16`。本方案不改模板，不注册生产 shader，也不生成其他 shader 变种。
- 后续 layer20 block/mask 发布、late-layer filtering、同组最终 `myds_v41_gpu_topk_causal`、attention、缓存与 snapshot 行为全部保持当前 source 调用顺序。不得因三个 capture 层没有 filter 而删除其他层的真实消费者。

## 2. 建议的最小 API 与文件改动

建议只在私有 runtime source snapshot 中改以下内容；主工作树的生产注册保持不动。

1. `include/myds_v41_gpu.h`：声明 strict mode getter、selected symbol getter、六字段 host counters/getter，以及一个明确的普通 prefill scoped entry。
2. `src/myds_v41_gpu.c`：strict once-parse、host counters、scoped entry 与 `indexscores_chunk` 中的私有 candidate 分支，复用当前 args/geometry/dispatch bridge。
3. `src/myds_v41.c:3779`：唯一实际 prefill 调用改用 scoped entry，并传 `g->gpu` 和 `ordinary_prefill_allowed = !myds_trace_active()`。当前唯一上游来自普通 `myds_v41_prefill_tokens`；speculative forward 的 API 在另一条调用链。
4. 新增 CPU-only host fixture 与私有 fullmodel adapter include。无需改 `src/myds_metal.m` 或 `include/myds_metal.h`：当前已存在 PSO preflight 与 generic bridge。

建议 API 形状（名称可在实施时统一，不改变合同）：

```c
int myds_v41_gpu_indexer_k_resident_enabled(void); /* absent/0 -> 0, 1 -> 1, invalid -> -1 */
const char *myds_v41_gpu_indexer_k_resident_selected_kernel(bool cache_bf16);
typedef struct {
    uint64_t requests, encoded, fallbacks, failures;
    uint64_t f32_dispatches, bf16_dispatches;
} myds_v41_gpu_indexer_k_resident_counters_t;
void myds_v41_gpu_indexer_k_resident_counters(
    myds_v41_gpu_indexer_k_resident_counters_t *out, bool reset);
int myds_v41_gpu_indexscores_prefill_chunk_scoped(
    myds_v41_gpu_scratch_t *scratch, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q, const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache, uint32_t n_comp, uint32_t n_heads,
    uint32_t n_tokens, uint32_t causal_start, uint32_t causal_ratio,
    bool cache_bf16, bool ordinary_prefill_allowed);
```

引入明确 scoped entry 的理由：现有两个 public chunk API 没有普通请求/trace 上下文，也没有 scratch budget 参数；仅根据 `n_tokens>=8` 猜上下文不足以实现严格 scope。原 `myds_v41_gpu_indexscores_chunk[_bf16]` 保持原 provider selector，不触发候选，供既有 direct API 测试/其他调用方使用。不要添加全局/TLS “当前在 prefill” setter，避免失败退出或并发请求泄漏作用域。

## 3. Selector 与 default-off 合同

`MYDS_INDEXER_K_RESIDENT` 只接受 absent / `0` / `1`，默认0；空串、`01`、`true`、`2`、负数都拒绝。`pthread_once` 解析必须发生在显式 `myds_configfile_load(path,true)` 之后。不同 arm 用不同进程，不在运行中 setenv 切换缓存 mode。invalid mode 在 context/scratch 建立或首次 scoped 请求前明确失败，不当作0。

mode0：不查询 candidate PSO、不调用 candidate、不改原 TILED/exact/direct 选择、不加入 barrier/commit/wait/新 buffer。最多保留一次 logical request 的 host atomic 观测；gold、same-build off/on 都明确记录这一 source 差异。旧库缺少两个候选 symbol 时 mode0 仍可正常执行。

mode1：在原 scorer 选择之前检查普通 scoped 请求是否 eligible；eligible 时不要求 `MYDS_INDEXER_TILED=1`，直接使用 K-resident。两个 symbol 分别为：

- F32 cache：`kernel_myds_indexer_scores_k_resident_f32`。
- BF16 cache：`kernel_myds_indexer_scores_k_resident_bf16`。

ineligible 请求继续执行原 selector，因此其行为仍服从原 config 的 TILED/exact/direct 设置。不要自动改 env，不对 SGBAR 或 key window 做语义覆盖。

初版建议 eligibility 为以下条件的交集：

| 条件 | 具体界限 |
|---|---|
| 调用 scope | 新 scoped entry，ordinary=true，无 trace/exclusive kprof，已有 active Metal batch |
| 固定 shape | n_heads=32，head_dim=128；cache storage 仅原 F32/BF16 两种 |
| Token 行 | `8 <= n_tokens <= MYDS_V41_CHUNK_MAX`，当前上限2048 |
| Ratio | 仅实际 V4.1 的1或2；其他合法通用输入仍走 reference |
| Position | `causal_start < 1048576`，以 uint64 算 `end=causal_start+n_tokens <=1048576` |
| 可见范围 | `(causal_start+1)/ratio >=1`；`last_visible=end/ratio`；`n_comp=max(512,last_visible)`，与 production group exact tuple 相符 |
| Keys | `1 <= n_comp <= scratch->max_comp`，并处于当前 chunk/key 配置与实际 cache capacity 内，不凭常量扩容 |
| Score 预算 | `(uint64_t)n_tokens*n_comp <= scratch->batch_floats`，保持生产2M设置，不调分组 |
| 原实验开关 | `indexer_key_window()==0`；SGBAR absent/`0`（按旧 selector 的实际非零语义判定） |
| 完整 spans | scores/Q/W/cache 的真实 size、offset alignment 与各自乘积全部合法；score writer 与所有三个完整 reader 区间不相交 |
| GPU 能力 | 两个 candidate PSO 都通过已有 exact preflight：SIMD32、128 threads、static+21504B TG memory/device 等约束 |

size/position 溢出、短 buffer、坏 alignment、writer-reader overlap 是非法请求，应明确 failure 而非拿 reference 掩盖。形状合法但不足8行、非普通作用域、keywindow/SG模式非零、无 active batch等是正常不适用，不算 candidate fallback。

这里的结构支持范围不等于已获实模正确性证明：真实组件仅覆盖过 ratio2/128×15424/layers2/8/14；ratio1、其他真实行宽、首块和末尾 remainder 必须由后续完整模型门覆盖。不得将 synthetic 模板覆盖等同于真实全图验收。

## 4. 缺失 PSO / dispatch 失败必须 fail closed

当前 `myds_metal_kernel_constants_preflight(name,threads,21504,NULL,0)` 已在 `src/myds_metal.m:3241–3257` 解析**同一个** dispatch PSO，检查 SIMD32、线程与总 TG memory，不碰 CB/encoder。mode1 首个 eligible 请求可先 preflight 两个 symbol，任何缺失都写明确 symbol/reason 日志、failure++、立即返回0。不得将 missing PSO 编为 ineligible，也不得进入 exact/direct retry。

可直接使用当前 preflight/cache，不在 C 层永久缓存“库已准备好”：同进程 Metal cleanup/reinit 后可能更换 library；跨库 readiness 必须重新预检。若后续要缓存，需先增加受实际 loaded-library 生命周期约束的 generation 合同，不能只靠一次 static bool。

active-batch 限定解决 host `encoded` 的准确性：当前 generic bridge 在 `metal_dispatch` 之前失败返回0；在 active batch 真正调用 `metal_dispatch` 后 `v41_finish` 直接返回1（`src/myds_metal.m:3050–3055,3260–3299`）。因此本场景 `dispatch==1` 可以准确记“实际编码被接受”，但仍**不是 GPU completion/success**。不允许 standalone completion failure 后模糊地把 encoded 写成0；standalone 不进入该候选。

控制流应为：

```text
strict mode invalid -> configuration error
ordinary scoped scorer call -> requests++
original basic invalid input -> failures++ (mode1 candidate request), return0
mode0 / normal ineligible -> original selector, no candidate counters other than request
mode1 eligible -> complete span validation -> both PSO preflight
preflight failure -> failures++, log named reason, return0 (no native retry)
dispatch candidate
  return0 -> failures++, return0 (no native retry)
  return1 -> encoded++, f32_dispatches++ OR bf16_dispatches++, return1
```

`publish_indexer_chunk` 将0转为−1；不回退逐token，不重做已经原地旋转/量化的 Q。保持原取消、错误清理及结束 batch 合同。getter/reset 不提交任何 GPU 工作，不加入等待，不清理 cache/model state。

## 5. 六个计数器与 fullmodel adapter

| 字段 | 定义 |
|---|---|
| requests | 新 ordinary scoped entry 实际接到的 logical scorer 调用数，mode0也可记录；不是 getter 次数、token数或编译数 |
| encoded | eligible candidate dispatch 被真实 encoder 接受的次数；后续 enclosing CB 失败由上层 GPU completion/state gate判定 |
| fallbacks | 已进入 eligible candidate 请求后，实际改跑原生/reference 的次数；设计中这一分支禁止，正常值恒0，保留作退化回归见证 |
| failures | 候选请求的非法 spans/PSO预检/dispatch失败次数，每次调用最多加1；配置解析错误单列日志，不造假的 encoding |
| f32_dispatches | 真正使用 F32 candidate symbol 的编码次数 |
| bf16_dispatches | 真正使用 BF16 candidate symbol 的编码次数 |

正常 ineligible→reference 不记 fallback，否则短 batch 等合法路径会被 fullmodel driver 错拒。host-only relaxed atomic；六字段 reset 只允许 idle sample 边界，不承诺运行中跨字段原子快照。process-wide 数字不冒称 per-context；正式测量要求独占。

不变量：`encoded=f32_dispatches+bf16_dispatches`，`requests>=encoded`；正常 `fallbacks=failures=0`。默认关闭 `encoded/f32/bf16=0`。symbol getter 返回 selector 名称，不单独冒称 actual GPU success；actual encoded 计数、直接 dispatch 通路和 frozen library identity共同作采用见证。

`tools/bench_indexer_fullmodel.c:31–109` 已有恰好匹配这六字段的 adapter 合同。私有 include 必须显式逐字段复制（不要靠不安全 struct cast）：

```c
static int bench_indexer_adapter_mode(void) {
    return myds_v41_gpu_indexer_k_resident_enabled();
}
static const char *bench_indexer_adapter_symbol(bool bf16) {
    return myds_v41_gpu_indexer_k_resident_selected_kernel(bf16);
}
static void bench_indexer_adapter_counters(bench_indexer_provider_counters_t *out,bool reset) {
    myds_v41_gpu_indexer_k_resident_counters_t v;
    myds_v41_gpu_indexer_k_resident_counters(out?&v:NULL,reset);
    if(out) *out=(bench_indexer_provider_counters_t){v.requests,v.encoded,v.fallbacks,
                                                   v.failures,v.f32_dispatches,v.bf16_dispatches};
}
```

该 driver 在每个 sample 前 reset，首个 output callback 保存 `indexer_first`。mode1 必须 `first.encoded>0`、两个 symbol 非空，final encoded/F32/BF16 都等于 first，即 **decode encoded=0**；fallback/failure全0。mode0 encoded=0。没有 adapter 的 `--run` 已拒绝，不要用 fabricated getter 绕过。`--preflight` 本身不得创建设备。

## 6. 必须先做的 CPU host fixture

新增独立 mock C fixture，编入实际 GPU selector 源码并 stub generic preflight/dispatch/buffer API；使用 function sections+dead stripping 或项目既有 host fixture方式，不连接真实 Metal/不创建 GPU。不能以复制一份 selector 的 mock 算通过。

至少覆盖：

1. absent/0 保留 exact（TILED未设）；TILED=1保留原TILED；direct/SG/窗口原行为完全由旧selector决定；default0不探测两个候选PSO。
2. mode=1 的实际 tuple `128×15424,pos30720,ratio2,bf16`，args72有效字段与capture一致，grid482×16、threads32×4、smem21504、buffers顺序Q/W/cache/scores；仅一次candidate dispatch，`requests=encoded=bf16=1`，其他0。
3. F32同形状使用F32 symbol，dtype计数互斥；ratio1、首块padded512、8/9 tokens、partial key tile/row tile边界。
4. `n_tokens=1/7`、ordinary=false、trace/exclusive kprof、keywindow>0、SGBAR非零、无batch：保留reference，不计fallback，不编码candidate；旧public chunk API保持reference。
5. heads31、ratio0、pos/end越界、乘积/实际budget越界、shortQ/W/cache/scores、错offset、scores与每个reader前后/子view别名：非法请求明确拒绝且未派发任何原生/candidate。
6. 缺F32、缺BF16、TE不32、maxthreads<128、static+dynamic内存超界、dispatch返回0：每次 failure恰好1，encoded0，fallback0；没有 exact/direct/scalar重试。
7. 预检与dispatch顺序、selector不调用commit/wait、不改auto/concurrent、不新增buffer/alias、reset读取无GPU副作用；cleanup/reinit后不能复用前库成功readiness。
8. caller级注入：scorer失败发生在Q RoPE/quant以后时，`publish_indexer_chunk` 返回−1且不触发逐tokenfallback，不重复旋转/量化。
9. adapter mode0/1、首callback前0编码、decode-only编码、prefill后decode泄漏、counter倒退、NULL/空symbol、fallback/failure均拒收。保留已有 `tests/test_indexer_fullmodel_driver.py` 的12个CPU负控。

CPU fixture只能证明selector/错误/观测合同；新的实际GPU数值、consumer与性能门另算。existing窗口GPU测试不能当作这个host fixture，也不能让一条GPU target依赖在CPU检查中被自动执行。

## 7. 私有 binary + lib 构建方法（未来实施后使用，当前未执行）

所有源先snapshot到新的独立stage，按完整原始文件bytes/SHA冻结；HEAD仅上下文，snapshot可以没有.git。保存旧R1/R2/v1所有目录，不覆盖任何旧manifest/source。仅在这个snapshot应用授权接入patch，并保存最小patch与反向byteproof。

已有 private full-TU library build可重用其完整冻结产物，或者用同参数重新生成后要求全部原prefix/shader/候选 fragment一致。参考命令：

```sh
python3 "$STAGE/source/tools/build_indexer_k_resident_probe.py" \
  --source-root "$STAGE/source" --out-dir "$STAGE/shader-build" --full-library
make -C "$STAGE/source" -f tools/indexer_fullmodel.mk \
  BUILD_DIR="$STAGE/obj" BIN_DIR="$STAGE/bin" \
  INDEXER_ADAPTER_HEADER="$STAGE/indexer_adapter.inc" \
  "$STAGE/bin/bench_indexer_fullmodel"
```

这两个命令仅构建；禁止将命令替换为默认 `make` 以免触发生产bin或shader目标。实际Make flags、minver/std/defines、compiler/SDK路径、source SHA、binary SHA、library SHA、完整命令均入manifest。目标M2需要同源、同flags CPU重建并核实际产物，而不是默认本机tuple等于目标。

在stage的exe旁放**唯一**冻结 `myds.metallib`（来自private probe），运行cwd用新的空目录，预检stderr和 `myds_metal_loaded_library_path()` 实际路径，严格resolve后核SHA。gold library对照用实际冻结deployed库；same-build off/on使用同一个private full-TU probe库和同binary，只差新mode。mode0加载缺候选symbol的旧gold库必须成功。不要用cwd或另一个同名库shadow，也不要把“预期路径”当已加载路径。

fullmodel运行前只执行 `--config ABS --preflight`：stable raw config、load(true)、62项实际MYDS原值及唯一新增实验mode的来源、不同空的logits/state目录、adapter是否编入均先核。config load(true)会覆盖已有env；若用私有overlay或新增env，必须记录实际生效值/完整MYDS环境，不能依旧header注释推断优先级。主冻结myds.conf不改。

## 8. 数值/消费者/性能与状态风险门

- TILED→candidate完整scores必须0-ULP；actual-default exact→TILED/candidate不应虚称score全同。旧真实L2有4cells、maxabs9.5367431640625e-7的provider切换差异已单列，不能擦掉。
- 全部真实score finite/因果−INF、完整有序Top512、layer20 block_selected/mask、late-layer filter后的scores/IDs、attention结果、完整词表logits、tokens/text和active/published KV/Engram/RNG/snapshot按冻结门逐bytes比较。两个provider选择不等价的风险由真实消费者门决定。
- 组件R门只覆盖该输入与测量合同，不证明完整图关键路径、coldDRAM或最终速度；resident K约增加每lane32个F32寄存器量，spills/occupancy以及与其他prefill内核的资源竞争都可能抵消组件优势。
- 不改变batch_floats、chunk大小、cache配置、模型/请求/token流身份、不插observer commit/wait到正常计时路径。capture宏仅诊断编译，普通build须预处理确认完全剥离。
- fullmodel首callback计数只是采用见证；必须同时核native GPU完成、完整状态/输出raw和独立reader。harness exit0不等于冻结性能门通过。
- 任何正式性能协议需在GPU前独立冻结controls、顺序、样本数、漂移/收益阈值、最多窗口数及失败处置；不从本方案补写一个事后阈值。当前任务没有获准运行fullmodel或HTTP，也没有改变生产部署。

**交付状态：仅方案，等待 root 在新组件R门完成后明确触发实施。**
