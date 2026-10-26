# 架构方案：借鉴 kernelpool/ds4 (ds41-perf) 优化 myds 推理引擎

## 1. 目标与背景 (Goal Description)

### 1.1 背景
[myds](file:///Users/fangqingyuan/Projects/myds) 是从 upstream `ds4` 派生、专注在 **Apple Silicon (M2 Ultra 192GB / M4)** 上运行 **DeepSeek-V4.1 Flash** 的高性能推理引擎。其项目特点是遵循极其严苛的 **0-ULP 逐位对齐（bit-identical）**、严谨的性能账本与门禁准则。在 `docs/2026-10-25-prefill-decode-performance-review.md` 中，团队刚尝试了 Indexer 的 K 矩阵跨 head 驻留寄存器等方向，但面临部署制品身份门与数值对齐的挑战。

本次针对开源项目 [kernelpool/ds4](https://github.com/kernelpool/ds4) 的核心性能分支 [`ds41-perf`](https://github.com/kernelpool/ds4/commits/ds41-perf/) 进行全量深度分析（136 个专项提交）。该分支由 Tarjei Mandt (`kernelpool`)、Ivan Fioravanti、Mark Shank 等工程师主导，专门对 DeepSeek V4.1 Flash 在 Metal (Apple Silicon) 以及 CUDA/ROCm 上的端到端性能进行了极致调优。

### 1.2 本方案目标
本方案旨在系统性梳理 `ds41-perf` 中的核心成果与失败反例，提炼出与 `myds` 当前硬件（M2 Ultra 76核 GPU / 统一内存）、架构约束（0-ULP 数值门禁）完全契合的技术方案，并制定可落地的渐进式重构路线图。

---

## 2. 需用户审阅的关键决策 (User Review Required)

> [!IMPORTANT]
> **数值一致性优先级高于吞吐量**：`ds41-perf` 的大部分优化均严格遵循 `bit-identical` 原则，每个优化项均配有独立的测试与回滚开关（如 `DS4_METAL_DISABLE_V41_*`）。本方案建议在 `myds` 中沿用该机制，新增对应的 `MYDS_OPT_*` 环境变量，未经验证前不作为硬编码默认行为。

> [!WARNING]
> **关于内存开销与权重窄副本（Narrow Weights）**：
> `ds41-perf` 中通过在模型载入时为 Router 生成 BF16 副本、为 Indexer $Q_B$ 和 Engram Projection 生成精确的 E4M3 (FP8) 副本，使 Decode 阶段的访存带宽直接降低 50%。这会在 Host/Metal 统一内存中额外占用数百 MB 空间。鉴于 M2 Ultra 拥有 192 GB 统一内存，且 Q2/Q4 权重的常驻内存约为 148 GB，该内存开销完全在安全裕度内。

---

## 3. 待澄清问题 (Open Questions)

> [!NOTE]
> 1. **DSpark 投机解码优先级**：`myds` 之前实测 PLD 投机为负收益（-6.6%），而在 `ds41-perf` 中已具备成熟的 V4.1 DSpark 完整管线（含原生 MXFP4 专家算子）。是否优先将资源集中在 Deterministic 单步内核加速（Prefill MMA Indexer + Decode Static Shapes），随后再考虑 DSpark？（**推荐先做单步内核**）
> 2. **外置硬盘存储场景**：当前测试机 M2 Ultra 的模型权重与 Engram 表是否挂载在外置 SSD (USB-C / Thunderbolt)？若是，`bdae764` 的 Engram Page Cache Warmer 将带来 2x+ 的冷启动 Prefill 吞吐提升。

---

## 4. `kernelpool/ds4:ds41-perf` 核心技术亮点与价值对比

经过对 136 个 commit 的逐项审计，以下是对于 `myds` 最具借鉴价值的技术项与负面教训总结：

| 序号 | ds41-perf 提交 | 核心技术点 | 收益与实测数据 | 对 myds 的借鉴意义 |
|---|---|---|---|---|
| **1** | `0f9b03c` | **MMA 寄存器驻留 Indexer 打分**<br>`kernel_dsv41_indexer_scores_mma` | 16K 下从 1.31s → 0.77s（**耗时 -41%**），Bit-identical | **直接解决 myds 10-25 续查未闭合痛点**，消灭 TG 内存往返与屏障 |
| **2** | `b66c926` | **编译期常量化静态 Decode 形状**<br>`_static_v41` 系列算子 | 消除动态维度寻址，编译器完全展开循环与指令调度 | Decode 吞吐稳步提升，消除 CPU/GPU 编排冗余 |
| **3** | `55b1e75`<br>`7d9903b` | **权重瘦身与 16 字节宽读取**<br>E4M3 $Q_B$ / Engram, BF16 Router | Engram 投影 **849us → 658us (-22.5%)**，访存字节减半 | 压榨 M2 Ultra 显存带宽，消除 GGUF 冗余精度拓展 |
| **4** | `f0b4f55` | **单 Token 全局 RoPE 缓存表**<br>`rope_tab` + 融合逆旋转/量化 | 消除每层重复计算，融合 Attention 输出写回与 Indexer 量化 | 减少 Decode 每步的 Dispatch 数量与冗余三角函数开销 |
| **5** | `64ceeea`<br>`2d5ef0d` | **BF16 Indexer 键缓存与候选块过滤** | 长上下文（>8K）Indexer 访存减半；Layer>20 仅打分候选块 | 大幅降低长序列（32K~150K）Decode 的计算与带宽开销 |
| **6** | `e99dc21` | **多 Threadgroup 分布式规约**<br>HC Collapse 拆分 42 个 TG | 消除单 SIMD/单 TG 造成的 GPU 计算空泡 | 让 M2 Ultra 的 76 个 GPU 核心充分满载，避免大核饥饿 |
| **7** | `4ff3af6` | **Decode Top-512 ArgSort 极致优化** | SIMD shuffle 局部排序 + 512 截断归并 + 单 pass 排序 | 提升每步 Decode 中挑选 512 个胜出 Indexer 候选的速度 |
| **8** | `bdae764` | **Engram 表物理 Page Cache 顺序预热** | 冷启动 USB SSD 2K: 198→420 t/s, 8K: 214→518 t/s (**+142%**) | 若权重置于外置固态，彻底消除 189GB 随机 4K IO 瓶颈 |
| **9** | `187f46b` | **ds4-bench 对拍工具链** | 每步 Decode Logits 的 FNV-1a Hash 校验 | 极其契合 myds 的 0-ULP 门禁自动化测试 |
| **避坑** | `72ccf36` | **Pre-encode 自旋保持 (已回滚)** | 造成 Command Buffer 挂起与每 Token GPU 空转耗电 | 印证了 myds 账本结论：Apple Silicon 异步空泡无法靠 GPU 自旋掩盖 |

---

## 5. 拟在 `myds` 实施的技术方案 (Proposed Changes)

方案按依赖关系与风险等级分为三期实施：

```mermaid
graph TD
    subgraph Phase 1: 零风险核心算子对齐
        A[0f9b03c: MMA Indexer Kernel] --> A1[合入 metal/dsv4_misc.metal]
        B[b66c926: Compile Literal Shapes] --> B1[特化 QA/KV/MoE/HC 静态尺寸]
        C[e99dc21: Multi-TG Reduction] --> C1[拆分 HC Collapse & Engram Gate]
        D[f0b4f55: Token RoPE Table] --> D1[单步生成 RoPE Table + 逆旋转融合]
    end
    
    subgraph Phase 2: 显存带宽与访存优化
        E[55b1e75 & 7d9903b: Narrow Weights] --> E1[Graph Alloc 生成 FP8/BF16 瘦拷贝]
        F[64ceeea: BF16 Indexer Key Cache] --> F1[>8K 自动切换 BF16 打分]
        G[2d5ef0d: Indexer Candidate Scoring] --> G1[Layer>20 仅打分 Candidate 块]
    end
    
    subgraph Phase 3: I/O 预热与测试套件
        H[bdae764: Engram Page Warmer] --> H1[后台顺序流式填充 Page Cache]
        I[187f46b: FNV-1a Hash Bench] --> I1[集成至 myds_test 门禁系统]
    end

    Phase 1 --> Phase 2
    Phase 2 --> Phase 3
```

---

### 阶段一：高价值单步算子加速与融合 (Phase 1)

#### 1. [NEW] MMA 寄存器驻留 Indexer 打分算子
在 [metal/dsv4_misc.metal](file:///Users/fangqingyuan/Projects/myds/metal/dsv4_misc.metal) 中引入 `kernel_dsv41_indexer_scores_mma`：
- **原理**：每个 simdgroup 处理 8 tokens $\times$ 32 keys，利用 `simdgroup_float8x8 kt[16]` 在寄存器中维持 $K^T$ 矩阵分片。
- **改动点**：彻底省去 threadgroup 共享内存的存储和屏障，在 32 个 head 循环内部利用 `simdgroup_multiply_accumulate` 执行点积并累加。

```metal
// metal/dsv4_misc.metal
kernel void kernel_dsv41_indexer_scores_mma(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token_base = tgpig.y * 8u;
    const uint row_base = tgpig.x * 32u + (uint)sg * 8u;
    const short2 xy = dsv41_mma_coord(lane);
    const uint token = token_base + (uint)xy.y;
    const uint row = row_base + (uint)xy.x;
    const bool live = token < args.n_tokens;
    const uint visible = live ? glm_indexer_batch_visible_rows(args, token) : 0u;
    const uint last_token = min(token_base + 8u, args.n_tokens);
    const uint max_visible = glm_indexer_batch_visible_rows(args, last_token - 1u);

    if (row_base < max_visible) {
        simdgroup_float8x8 kt[16];
        for (uint db = 0; db < 16u; db++) {
            float k0 = 0.0f, k1 = 0.0f;
            const uint d = db * 8u + (uint)xy.y;
            if (row_base + (uint)xy.x < args.n_rows)
                k0 = glm_cache_load_f32_or_f16(indexer_key_cache,
                    (uint64_t)(row_base + (uint)xy.x) * 128u + d, args.cache_f16);
            if (row_base + (uint)xy.x + 1u < args.n_rows)
                k1 = glm_cache_load_f32_or_f16(indexer_key_cache,
                    (uint64_t)(row_base + (uint)xy.x + 1u) * 128u + d, args.cache_f16);
            kt[db].thread_elements()[0] = k0;
            kt[db].thread_elements()[1] = k1;
        }
        device const float *qrow = (device const float *)(q + (uint64_t)min(token, args.n_tokens - 1u) * args.q_token_stride);
        device const float *w = (device const float *)(weights + (uint64_t)min(token, args.n_tokens - 1u) * args.weights_token_stride);
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint head = 0; head < args.n_head; head++) {
            device const float *qh = qrow + head * (args.q_head_stride / sizeof(float));
            simdgroup_float8x8 dot = make_filled_simdgroup_matrix<float, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint db = 0; db < 16u; db++) {
                simdgroup_float8x8 mq;
                mq.thread_elements()[0] = qh[db * 8u + (uint)xy.x];
                mq.thread_elements()[1] = qh[db * 8u + (uint)xy.x + 1u];
                simdgroup_multiply_accumulate(dot, mq, kt[db], dot);
            }
            const float wh = w[head];
            acc0 += max(dot.thread_elements()[0] * args.scale, 0.0f) * wh;
            acc1 += max(dot.thread_elements()[1] * args.scale, 0.0f) * wh;
        }
        if (live) {
            device float *dst = (device float *)(scores + (uint64_t)token * args.score_token_stride);
            if (row < args.n_rows) dst[row] = row < visible ? acc0 : -INFINITY;
            if (row + 1u < args.n_rows) dst[row + 1u] = row + 1u < visible ? acc1 : -INFINITY;
        }
    }
}
```

在 [src/myds_v41_gpu.c](file:///Users/fangqingyuan/Projects/myds/src/myds_v41_gpu.c) 中将 `kernel_dsv41_indexer_scores_mma` 接入分发入口。

#### 2. [MODIFY] 编译期静态 Decode 形状特化
在 [metal/dense.metal](file:///Users/fangqingyuan/Projects/myds/metal/dense.metal) 和 [metal/moe.metal](file:///Users/fangqingyuan/Projects/myds/metal/moe.metal) 中引入固定尺寸模板特化：
- `kernel_mul_mv_q8_0_f32_bf16_pair_qa_kv`：将 `q_a` (5120 $\to$ 1280) 与 `kv` (5120 $\to$ 512) 融合在单次 dispatch 中分两段执行，常量化所有行列步长。
- `kernel_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_v41_f32`：特化 MoE Gate/Up (5120 $\to$ 2304, 384 专家选 6) 的参数常量。
- `kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_static`：Attention Output (8192 $\to$ 5120) 融合写回 HC 4 个残差流。

#### 3. [MODIFY] 分布式规约展开 (Multi-TG Reduction)
在 [metal/dsv41.metal](file:///Users/fangqingyuan/Projects/myds/metal/dsv41.metal) 中：
- 将原本单 threadgroup 串行跑的 `kernel_dsv41_deferred_collapse` 重构为两阶段：
  - Stage 1: `kernel_dsv41_deferred_collapse_part`（分派 42 个 threadgroups，每组 32 线程，并行对 4 个残差流进行加权折叠和 partial sum）。
  - Stage 2: `kernel_dsv41_deferred_collapse_norm`（汇总 42 个 partial sum，完成 RMSNorm 缩放和 BF16 输出写回）。
- 保证加法树结构和操作数顺序与原实现一致，做到 0-ULP 逐位对齐。

#### 4. [MODIFY] 全局 RoPE Table 与 Attention 尾部逆旋转融合
在 [src/myds_v41_decode.c](file:///Users/fangqingyuan/Projects/myds/src/myds_v41_decode.c) 与 [metal/flash_attn.metal](file:///Users/fangqingyuan/Projects/myds/metal/flash_attn.metal) 中：
- 在 decode 单步前统一生成一份长为 64 的 `float2 rope_tab`（对应 128 维 head 的 64 组 cos/sin 对）。
- `kernel_dsv41_flash_attn_vec_reduce_heads` 直接读取该表，在注意力规约写回时同步执行逆旋转与 BF16 舍入，后续投影无需再调用 RoPE 内核。

---

### 阶段二：显存带宽削减与长上下文优化 (Phase 2)

#### 1. [NEW] 模型权重瘦身副本 (Narrow Decode Weights)
在 [src/myds_v41.c](file:///Users/fangqingyuan/Projects/myds/src/myds_v41.c) 初始化阶段：
- 识别 GGUF 中因格式规范被 widen 的权重：
  - Router 权重（GGUF 为 F32，源数据为 BF16） $\to$ 构建并缓存 BF16 紧凑副本。
  - Indexer $Q_B$ 与 Engram Projection 权重（GGUF 为 F16，源数据为 E4M3/FP8） $\to$ 构建无损 E4M3 紧凑副本。
- 在 Decode 阶段调用 `kernel_mul_mv_bf16_f32_16x` 与 `kernel_mul_mv_e4m3_f32`，使用 16 字节宽访存指令读取。

#### 2. [MODIFY] 超长上下文 Indexer BF16 键缓存与候选块过滤
在 [src/myds_v41_gpu.c](file:///Users/fangqingyuan/Projects/myds/src/myds_v41_gpu.c)：
- Prefill/Decode 写入 Indexer Key Cache 时，同步写入一份紧凑的 `bf16` 格式（由于原始 key 即为 FP4 乘 scale 并在写入时做过 BF16 截断，此转换完全无精度损失）。
- 当已存 key 数量 $\ge 8192$ 时，Decode 阶段自动切入 `kernel_dsv41_indexer_scores_bf16`，内存带宽开销直接减半。
- 对于 Layer $> 20$ 且压缩行数 $> 2048$ 的情况，启用 `ds4_gpu_dsv41_indexer_candidates_tensor`，只打分上一层通过 Top-2048 块掩码筛选出的候选块。

---

### 阶段三：存储优化与工程门禁提升 (Phase 3)

#### 1. [NEW] Engram 物理 Page Cache 流式预热器
在 [src/myds_engram.c](file:///Users/fangqingyuan/Projects/myds/src/myds_engram.c)：
- 针对搭载大内存（如当前 M2 Ultra 192GB）且 Engram 存放在外置存储设备的场景：
- 启动后台低优先级 pthread，以系统可用的最高顺序读带宽（1~3 GB/s）流式预读 Engram 文件内容并丢弃，使 OS Page Cache 驻留热页。
- 遇系统内存压力通知或手动开关 `MYDS_DISABLE_ENGRAM_WARM=1` 时随时优雅退出。

#### 2. [NEW] ds4-bench 风格的 Logits FNV-1a Hash 检验机制
在 [tools/](file:///Users/fangqingyuan/Projects/myds/tools/) 与测试套件中：
- 引入逐 step 输出的 logits FNV-1a 64-bit 哈希打印功能（`MYDS_DECODE_LOGITS_HASH=1`）。
- 在修改任何 Metal 算子时，通过双分支脚本自动化对比新老内核输出的哈希值，确保 100% 逐位一致。

---

## 6. 验证计划 (Verification Plan)

### 6.1 单元测试与逐位门禁 (Automated Bit-Identity Tests)
每个迁移的内核必须通过独立的合成数据逐位对拍测试（严禁任何 1-ULP 差异）：

```bash
# 1. 编译并运行 indexer MMA 对拍单测
make test-indexer-mma

# 2. 运行 Metal Decode 静态尺寸与规约对拍测试
make test-decode-literal-shapes
make test-split-collapse

# 3. 运行全量现存单元测试
make test
```

### 6.2 端到端模型基准验证 (Full Model Golden Tests)
使用真实模型权重进行 Golden 对齐与性能复测：

```bash
# 验证端到端 logits 与 ds4 oracle 逐位对齐
./bin/myds-bench --model /path/to/DeepSeek-V4.1-Flash-Q2.gguf --prompt-len 8192 --check-golden

# 测试 Prefill 与 Decode 吞吐改善情况
python3 eval/stream_bench.py --url http://127.0.0.1:8099/v1 --max-tokens 128 --jsonl /tmp/bench.jsonl
```

### 6.3 回滚开关验证 (Feature Flag Verification)
验证所有新增开关在禁用状态下能精确回退到原有逻辑：
- `MYDS_OPT_INDEX_MMA=0`
- `MYDS_OPT_LITERAL_SHAPES=0`
- `MYDS_OPT_SPLIT_REDUCE=0`
- `MYDS_OPT_ROPE_TABLE=0`
- `MYDS_OPT_INDEX_BF16=0`
- `MYDS_OPT_ENGRAM_WARM=0`

---
*注：按照规划模式（`/plan`）指令，本方案给出后即停止，不自动执行下一步代码修改。*
