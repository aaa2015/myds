# myds: Apple Silicon Metal 推理引擎实验

**当前 native engine 已完整实现 DeepSeek V4.1 文本计算图（HC、Engram、压缩 KV、indexer），并通过与 ds4 oracle 的逐位验收；生产 8181 正在运行 DeepSeek-V4.1-Flash Q2。** 正确性门禁默认放行已验收特性，`MYDS_DENY_V41=1` 可不重编译地恢复「拒绝加载」（紧急刹车）。

验收依据与门禁规则见 [正确性修复与验证状态](docs/CORRECTNESS_STATUS.md)；性能结论以 `docs/verification/` 下带日期的实测记录为准。单元测试通过不等于真实模型逐位对齐，改动数值路径仍须过同等门禁。

---

## 已有算子与实验基础

- **围绕 DeepSeek V4 / V4.1 构建的算子基础（完整计算图尚未实现）**：
  - **RMSNorm** 归一化加速
  - **MLA (Multi-head Latent Attention)** 解耦多头潜在注意力与低秩压缩 KV Cache
  - **Decoupled RoPE** 解耦旋转位置编码
  - **DeepSeekMoE** 混合专家网络（Shared Experts 共享专家 + Top-K 路由专家，Sigmoid 门控与权重重整化）
  - **SwiGLU** 激活函数融合计算
  - **模型加载检查**：架构名称仅用于识别模型家族；另外检查必要操作和词表维度，遇到未实现的 HC、Engram、压缩 KV、indexer 会拒绝推理
- **Apple Silicon Metal 算子**：
  - 仓库包含从 `ds4` 迁移的 `.metal` 源文件；文件存在不代表相应模型步骤已接入计算图
  - 利用统一内存架构（Unified Memory）通过 `MTLResourceStorageModeShared` 实现 CPU 与 GPU 之间零拷贝
  - 运行时即时编译优化，原生发挥 M 系列芯片（M1/M2/M3/M4/M5 等）的最大能效
- **极简架构与零依赖**：
  - 核心推理引擎仅由几个高内聚、低耦合的模块组成，无需任何外部第三方库（仅依赖系统级 Foundation 与 Metal 框架）
  - 内置极简 GGUF v2/v3 解析器，支持 `mmap` 秒级映射大模型权重
  - 内置 merge-rank byte BPE、JoyAI 预分词、显式字节长度解码与 UTF-8 流式输出
  - `make` 构建本地程序与测试；另有独立 HF tokenizer 与 ds4 tensor 对照工具

---

## 项目结构

```
myds/
├── Makefile                          # 极简构建规则 (make, make test, make clean)
├── README.md                         # 项目设计与使用文档
├── metal/                            # 迁移自 ds4 的 20 个 DeepSeek Metal 核心 Shader
│   ├── argsort.metal                 # Top-K 路由排序
│   ├── bin.metal                     # 标量与张量二元计算
│   ├── concat.metal                  # 张量拼接
│   ├── cpy.metal                     # 快速连续内存与格式转换
│   ├── dense.metal                   # 量化矩阵向量/矩阵乘法 (Q8_0, Q4_K, F16, F32)
│   ├── dsv4_hc.metal                 # HC 多路残差算子源码（未接入 native 计算图）
│   ├── dsv4_kv.metal                 # DeepSeek KV Cache 状态操作
│   ├── dsv4_misc.metal               # DeepSeek 辅助算子
│   ├── dsv4_rope.metal               # 解耦 RoPE 旋转位置编码
│   ├── dsv41.metal                   # DeepSeek V4.1 专有优化算子
│   ├── flash_attn.metal              # 闪速注意力核心计算
│   ├── get_rows.metal                # 嵌入层查表操作
│   ├── glu.metal                     # SwiGLU 激活函数
│   ├── moe.metal                     # MoE 专家路由与输出累加
│   ├── norm.metal                    # RMSNorm 算子
│   ├── repeat.metal                  # 广播重复
│   ├── set_rows.metal                # 行级写入
│   ├── softmax.metal                 # Softmax 与缩放
│   ├── sum_rows.metal                # 行求和归约
│   └── unary.metal                   # 一元运算
├── include/                          # 头文件定义
│   ├── myds.h                        # 推理引擎主接口与 V4/V4.1 架构判定
│   ├── myds_metal.h                  # Metal GPU 运行时与算子声明
│   ├── myds_gguf.h                   # 极简 GGUF 格式解析器
│   ├── myds_tokenizer.h              # DeepSeek BPE 分词器
│   ├── myds_server.h                 # HTTP 服务端
│   └── myds_log.h                    # 微秒级线程安全结构化日志
├── src/                              # 核心源码
│   ├── main.c                        # 命令行与交互终端入口
│   ├── myds_engine.c                 # DeepSeek Transformer 前向传播流程与模型加载
│   ├── myds_cpu.c                    # CPU 数值算子（复用 RoPE 角度计算）
│   ├── myds_metal.m                  # Metal 运行时与 GPU 算子分发
│   ├── myds_gguf.c                   # GGUF 格式解析与 mmap 内存映射
│   ├── myds_tokenizer.c              # BPE 编码与流式解码
│   ├── myds_server.c                 # POSIX HTTP Server (OpenAI Responses / Chat)
│   └── myds_log.c                    # 结构化日志实现
└── tests/
    ├── test_metal.c                  # GPU 核心算子金标对齐测试
    ├── test_gguf.c                   # GGUF 解析与 V4/V4.1 架构校验/V2/V3 拦截测试
    ├── test_tokenizer.c              # BPE 分词与特殊 Token 测试
    ├── test_server.c                 # HTTP 端点与并发连接测试
    ├── test_server_api.c             # Responses API 与协议解析金标测试
    ├── test_engine.c                 # 采样算法与 V4 架构判定单元测试
    ├── test_cpu.c                    # CPU RoPE 逐位回归与微基准
    └── test_log.c                    # 结构化日志线程安全与过滤测试
```

---

## 编译与测试

最新 decode 实施状态见 [2026-10-09 实施记录](docs/DECODE_IMPLEMENTATION_2026-10-09.md)，
包含默认 target 路径、batch2 状态修复、QKV 并发与长上下文 scorer 候选。
[验证协议](docs/benchmarks/2026-10-09-decode/README.md) 提供真实 DSpark 加载及 A–B–A 命令；
[Mac Studio 实测](docs/benchmarks/2026-10-09-decode/macstudio/README.md) 记录完整模型状态门禁、
普通256输出及长上下文对照，包含负收益和失败后修复记录。历史软件与硬件路线见
[2026-10-08 优化方案](docs/DECODE_OPTIMIZATION_PLAN_2026-10-08.md)。

2026-10-05 的 Mac Studio 独占测试、逐 token 耗时和组件回放证据见
[本轮实测汇总](docs/benchmarks/2026-10-05-optimization/final-summary.json)。
测试关闭 profiler、内存前缀命中和 KV 磁盘命中，每组预热一次后正式运行三次。
prefill 按实际处理的输入 token 计时，decode 按首次到末次输出回调之间的 token 间隔计时。
基准固定 `temperature=0`、`top_p=1`、`top_k=0`、`min_p=0`、
`repetition_penalty=1`、`seed=42`；greedy argmax 路径不读取采样随机数。
长输入的耗时波动及无 GPU 额外提交的路径诊断见
[补充验证](docs/benchmarks/2026-10-05-optimization/head-first-long-validation.json)，
两轮变慢的原始数据仍保留，不将复测的稳定收益当作所有运行的保证。
[早期性能记录](docs/PERFORMANCE.md) 保留历史状态说明，不作为本轮优化的基线。

### 1. 编译项目

在项目根目录下直接运行：

```bash
make
```

将在 `bin/` 目录下生成 `myds` 主程序以及独立的测试程序。

### 2. 运行单元测试套件

```bash
make test
```

测试流程将自动编译并依次运行单元测试套件，包含下列基础测试以及
Engram、KV 磁盘存储、前缀恢复、V4.1 GPU 和 HTTP 字节输出回归：

1. **GGUF 模型解析器测试套件 (`bin/test_gguf`)**：
   - [x] GGUF v2/v3 标量、字符串、布尔及动态数组元数据提取
   - [x] 张量描述信息校验与 32 字节对齐 mmap 内存映射验证
   - [x] 非法魔数（Magic）、损坏文件及不存在文件的容错处理
   - [x] **DeepSeek V2 & V3 架构拦截校验**（严格拒绝 legacy V2/V3 模型）

2. **DeepSeek BPE 分词器测试套件 (`bin/test_tokenizer`)**：
   - [x] 词表加载与 BOS/EOS 特殊 Token 识别
   - [x] Byte BPE 映射、merge rank、JoyAI 预分词与已知答案
   - [x] Byte-Fallback `<0xHH>` 未登录字节回退处理
   - [x] 逐 Token 还原文本与字节 Token 还原
   - [x] 缺失词表时的 256 字节兜底模式验证

3. **Metal GPU 核心算子测试套件 (`bin/test_metal`)**：
   - [x] **RMSNorm** 算子（与 CPU 基准对齐，浮点误差 < 1e-4）
   - [x] **SwiGLU** 激活函数（与 CPU 基准对齐，浮点误差 < 1e-4）
   - [x] **Decoupled RoPE** 解耦旋转编码（旋转前后向量模长守恒）
   - [x] **DeepSeek-MoE Router** 专家路由（Sigmoid 门控打分、Top-K 选取与权重重整化）
   - [x] **GEMV F32** 浮点矩阵向量乘法
   - [x] **GEMV Q8_0** 量化反量化点积精度测试
   - [x] **DeepSeek-MoE Sum** 多专家加权累加测试
   - [x] **Add** 残差加法与广播
   - [x] **Softmax** 归一化与单调性校验
   - [x] **Buffer R/W** 统一内存边界读写安全测试

4. **HTTP 服务器并发测试套件 (`bin/test_server`)**：
   - [x] Socket 生命周期与 HTTP 报文收发
   - [x] 并发非阻塞请求测试

5. **微秒级结构化日志测试套件 (`bin/test_log`)**：
   - [x] 日志级别过滤 (DEBUG / INFO / WARN / ERROR)
   - [x] 多线程并发无锁写入安全性测试

6. **OpenAI Responses API 协议测试套件 (`bin/test_server_api`)**：
   - [x] 端点路由匹配 (`/responses`, `/chat/completions`, `/health`, `/models`)
   - [x] 复杂输入参数优先级解析 (`max_output_tokens` 优先于 `max_tokens`)
   - [x] Responses API 流式 SSE 事件与非流式 JSON 响应规范金标对齐

7. **推理引擎采样与架构判定测试套件 (`bin/test_engine`)**：
   - [x] Greedy Argmax / Temperature / Softmax / Top-p 采样算法测试
   - [x] **DeepSeek V4/V4.1 架构专有支持与 V2/V3 拦截测试**（`myds_is_supported_arch`）
   - [x] Prefill 工作区扩容/复用、跨 512-token 边界、取消与恢复、KV 容量检查

8. **CPU 数值回归 (`bin/test_cpu`)**：
   - [x] RoPE 在 288 组形状/位置/缩放/正反向组合下与原实现逐位一致
   - [x] 逆旋转还原、无旋转及非法形状边界

运行可复现的算子与合成模型基准：

```bash
make bench
```

其中 `bench_metal` 在同一进程比较逐 token 与批量 GEMV，并检查逐位一致；
`bench_synthetic` 生成小型 MLA + MoE GGUF，测量真实加载、预填充和解码路径。
合成模型用于验证调度开销，真实模型性能请使用 `tools/bench.sh` 或 HTTP 基准。

---

## 快速上手

### 命令行生成

以下展示 CLI 接口：

```bash
./bin/myds -m /path/to/deepseek.gguf -p "你好，请介绍一下你自己。"
```

### 交互式对话模式

如果未指定 `-p` 参数，`myds` 会自动进入交互式对话模式：

```bash
./bin/myds -m /path/to/deepseek.gguf
```

### HTTP 服务器模式 (OpenAI 兼容 API)

`myds` 内置 POSIX HTTP 服务端。只有模型加载通过后才会启动监听：

```bash
./bin/myds -m /path/to/deepseek.gguf --port 8181 --host 0.0.0.0 --alias local-myds
```

支持的标准端点：
- `GET /health`：服务健康检查
- `GET /v1/models`：获取已加载的模型列表
- `POST /v1/responses` 或 `/responses`：原生支持 OpenAI Responses API 格式（`mycode` 客户端默认协议），完整支持流式 SSE (`event: response.output_text.delta`, `response.completed` 等) 与非流式 JSON (`output` 消息数组)
- `POST /v1/chat/completions` 或 `/chat/completions`：支持标准 OpenAI Chat Completions 协议（流式 SSE 与非流式 JSON）
- 请求体参数兼容性：全面支持 `instructions`（系统指令）、`input`（消息数组或纯文本，自动展开 `input_text` 复杂结构）、`messages`、`max_output_tokens` / `max_tokens`、`temperature` 与 `top_p`

### 算力舰队部署与启动

生产实例（8181）的替换必须经停产窗口（模型重载约 1~2 分钟）；改码只在 macmini，macstudio 只做靶机。

`deploy` 中已集成 `myds` 算力引擎支持：
- `start_fleet.sh`：定义 `SERVER_TYPE=15` 或 `SERVER_TYPE=myds` 即可调度 `myds` 作为后端引擎
- `sync_myds.sh`：以 `macmini` 为唯一权威源码，自动增量同步到 `macstudio` 并完成远程构建
- `deploy.sh`：一键自动执行同步、编译与舰队点火：

```bash
cd ~/Projects/deploy
./deploy.sh
```

### 常用参数说明

```
Usage:
  ./bin/myds -m <model.gguf> [options]

Options:
  -m, --model <path>      Path to DeepSeek GGUF model (required)
  --port <port>           Run HTTP server on specified port
  --host <host>           Host to listen on (default: 0.0.0.0)
  --alias <name>          Model alias for API responses (default: myds)
  -p, --prompt <string>   Prompt string (if omitted and no --port, starts interactive chat)
  -n, --max-tokens <int>  Max tokens to generate (default: 256)
  -c, --ctx, --ctx-size   Context window size (default: 4096)
  --temp <float>          Sampling temperature (default: 0.7, 0.0 for greedy)
  --top-p <float>         Nucleus top-p sampling cutoff (default: 0.9)
  --seed <int>            Random seed (default: random)
  -h, --help              Show this help message
```

### V4.1 推理工作区与流式传输

chunk attention 的 KV slab 和 mask 初始只分配窗口路径所需的 10.625 MiB。
压缩键容量仍覆盖配置的上下文；走 dense slab 且现有工作区不足时才扩容。
内存前缀快照在首次捕获对应槽位时分配，已分配的槽位继续复用。
compressed KV 与 index cache 保存已完成 BF16 舍入的上 16 位，读取时还原 F32；
计算中间量保持原精度与归约顺序。

| 环境变量 | 默认值 | 行为 |
| --- | --- | --- |
| `MYDS_ATTN_GATHER_MIN_KEYS` | `0` | scratch 创建时捕获。压缩键数小于阈值时走 slab，达到阈值时在 gather 已启用且提供缓冲的条件下走 gather。合法范围为 `0..chunk_key_max`；非法值会拒绝工作区创建。阈值须按本机与负载测量。 |
| `MYDS_ATTN_GATHER` | 启用 | 设为 `0` 关闭 gather，阈值不覆盖此开关。 |
| `MYDS_ATTN_VEC_HEAD_FIRST` | 启用 | gathered prefill 的 vec attention 将 query/head 物理 grid 调换，保留每个 head 的算术与归约。设为 `0` 使用原排列。旧 metallib 缺新 PSO 时自动回退，并在 scratch 中缓存不可用状态；首次回退输出一条 WARN 日志，进程级计数可由 `myds_v41_gpu_attention_counters()` 读取。 |
| `MYDS_PREFILL_ATTN` | `vec` | `matrix` 与 `auto` 是 compressed gather 的实验选项。矩阵归约会产生浮点差异，实测长输入生成 token 可变化；默认关闭。`auto` 的矩阵 PSO/共享内存不可用时回退 vec，显式 `matrix` 则返回失败。 |
| `MYDS_QKV_FUSE_BF16` | 启用 | decode 的 Q8 QKV 投影把 BF16 舍入合并到 GEMV 写出；设为 `0` 恢复独立舍入 dispatch。trace 使用原路径。 |
| `MYDS_QKV_FUSE_POST` | 启用 | decode 合并 Q/K RoPE、KV 量化及窗口写入，保留原舍入边界；设为 `0` 恢复原 dispatch。缺融合 PSO、别名或 copy 故障注入时回退原路径。 |
| `MYDS_QKV_NR0` | `4` | 融合 Q8 GEMV 每组处理四个独立输出行；`2` 使用两个行的配置。归约顺序不变。 |
| `MYDS_BATCH2_SPEC` | 关闭 | 显式 `1` 请求 greedy、无图片的真实 batch2；剩余输出/context 至少两步，DSpark 主候选仍要求 high-confidence。 |
| `MYDS_SPEC_SERIAL_PROBE` | 关闭 | 显式 `1` 允许 V4.1 逐 token 验证的 DSpark/PLD 对照；默认未启用 batch2 时跳过这类草稿与快照工作。 |
| `MYDS_SPEC_COST_SCHEDULER` | 关闭 | 显式 `1` 按整个周期成本与纯 target 基线比较，三个不划算周期后 backoff 16 个候选周期；每请求重置，需独立 A/B。 |
| `MYDS_DECODE_METRICS` | 关闭 | 显式 `1` 收集 host 分项时间与原始周期；请求 counters 始终可读。成本调度器也启用累计计时，但不会单独开启原始周期数组。 |
| `MYDS_QKV_CONCURRENT` | 关闭 | 显式设为 `1` 才在单 token QKV/Indexer 局部图创建 concurrent encoder，按资源同步独立分支；正常退出与失败都恢复进入前的 encoder/auto-barrier 设置。trace 与关闭 QKV pipeline 时保留参考路径。已有 `MYDS_METAL_CONCURRENT=1` 仍有原全局语义。 |
| `MYDS_INDEXER_COMPACT_SCORER` | 关闭 | 显式设为 `1` 才在单 token decode 的 layer 24/28/32/36、历史行数超过 16384 时，按 layer 20 原 selected block IDs 派发评分。先填充 dense scores 为 `-INFINITY`，有效行写回原 ID；原 filter 与 dense top-k 分区/同分网络保持一致。缺可选 PSO 在写入前回退；开始派发后失败终止。prefill、trace 与批量投机保持参考评分路径。 |
| `MYDS_MOE_IQ2_TAIL_CULL` | 启用 | prefill MoE IQ2 up/gate 的尾 tile 跳过没有有效 token 的半 tile。`0` 恢复原 kernel；默认缺新 PSO 时回退原 kernel，显式 `1` 缺 PSO 时初始化失败并清理所有部分状态。 |
| `MYDS_SSE_DRAIN_TIMEOUT_MS` | `1000` | SSE 完成时的整体排空期限，合法范围 `1..30000` ms；非法值使用默认值。 |

两项 decode 实验均默认关闭。本地 M4 的非零组件检查验证 QKV/HC/Indexer 中间量、
有效与屏蔽行分数及原 top-k IDs 逐位一致；Mac Studio 普通输入的 QKV A–B–A 未见额外收益，
完整结果和配置见上述实测报告。
compact scorer 只减少屏蔽行的空 threadgroup，并增加一次 dense `-INFINITY` 填充；
它仍扫描原 dense top-k，不能承诺 top-k 加速。采纳前分别测量实际 GPU 重叠及
长上下文整模型 A–B–A，报告路径计数、fallback、数值/状态回归和 P95/P99；本轮128k仍未测量。
`myds_v41_gpu_indexer_compact_counters()` 提供请求、实际派发、缺 PSO 回退、dense 行数及
实际派发行槽数，避免把请求开关当作已执行路径。

生产 SSE 输出使用每请求独立的 FIFO writer。队列上限为 1 MiB，包含包头和正在发送的包；
入队不等待客户端 socket 或空闲队列容量。积压超限或断连会取消该请求，socket 在 writer
退出并 join 后关闭。尾部排空目前仍持有推理槽，等待受上述整体期限限制。

低位 KV 磁盘缓存改为窗口快照加共享 BF16 段：第一段为 `[0,1024)`，后续每段 512 token。
窗口文件为 `kvlow-v3-<identity>.bin`，段文件为 `kvsegments-v1-<identity>.bin`，
二者的总文件容量共同计入 `MYDS_KV_DSK_BUDGET_MB`。旧格式文件不再复用，也不会自动删除。
快照仅在相关段提交后发布；恢复校验位置、前缀哈希、布局和完整 payload 校验和，
引用的段被覆盖或损坏时返回缓存 miss。相同内容再次保存不重写窗口或共享段，但仍读取
并校验已有段。此存储改动不改变 `MYDS_KV_DSK_ASYNC` 的默认开关。

针对这些改动的本地门禁为 `make test`。V4.1 GPU 回归分别运行默认 gather、关闭 gather、
显式阈值三种配置，并覆盖同一 command batch 内的路径切换、工作区扩容和 BF16 缓存对照。
这些测试不替代真实 GGUF 模型的 logits、吞吐与 TTFT 验收。

`make test` 还包含 QKV 两种行配置的非零逐位对照，以及 attention 的短块、carry、
F32/BF16 输入、子视图 canary、同一 batch 混用和缺 PSO 回退。独立完整模型计时工具
可用 `make bin/bench_phase` 构建；它不会由默认测试自动加载模型。

### 2026-10-05 测试、日志与测量复现

本轮完整模型实测机器是 Apple M2 Ultra / 192 GiB / macOS 26.5.2，模型为
`DeepSeek-V4.1-Flash-Q2.gguf`，上下文容量 262144。本地 Apple M4 / 16 GiB
运行小规模 GPU fixture 和单元测试；完整模型吞吐数据来自 Mac Studio。
需要 Python 3、Clang、macOS SDK 和 Metal 编译工具；项目的
`tools/build_metallib.py` 负责生成 metallib。完整模型复测需要本机准备相同 GGUF。

#### 单元测试与日志

从仓库根目录运行，保存 stdout/stderr，并通过 `pipefail` 保留测试失败状态：

```sh
MYDS_REPRO_OUT="$(mktemp -d /tmp/myds-review-XXXXXX)"
set -o pipefail
make -B -j8 all bin/bench_phase 2>&1 | tee "$MYDS_REPRO_OUT/build.log"
make test 2>&1 | tee "$MYDS_REPRO_OUT/unit-tests.log"
make test-tail-init 2>&1 | tee "$MYDS_REPRO_OUT/tail-init.log"
python3 docs/benchmarks/2026-10-05-optimization/protocol/selftest-summarize-final.py \
  > "$MYDS_REPRO_OUT/measurement-audit-selftest.json"
python3 tools/bench_reproduce.py verify
# 仅在 afcc05c 检出上：python3 tools/bench_reproduce.py verify --check-source
```

`make -B` 强制重编译，避免复制源码或切换测量版本时复用时间戳较新的旧对象文件。
`make test-tail-init` 是独立 GPU 测试，不由 `make test` 自动执行；
`make build-tail-init` 只构建缺少可选 tail PSO 的真实 metallib fixture。
测量汇总脚本的 selftest 只处理 CPU 数据，验证有效样本及 18 种错误记录的拒绝行为。

| 测试入口 | 主要断言 |
| --- | --- |
| `bin/test_v41_gpu`、`--slab`、`--threshold` | gather/slab 选择、阈值、lazy 工作区、BF16 缓存、同一 batch 内路径切换；compact scorer 在 1..128k 行比较 dense scores/原 top-k IDs，覆盖全允许/屏蔽、有限及 `-INFINITY` 同分、尾块、反序 block IDs、子视图 canary、concurrent 依赖和缺 PSO 未写入回退；三种配置均在 `make test` 中执行。 |
| `bin/test_attention_policy` | matrix/auto/vec 策略、缺 PSO 和共享内存失败回退、恢复后的重试。 |
| `bin/test_attention_vec` | 非零 F32/BF16 逐位对照、短块和 carry、子视图 canary、head-first 成功与缺 PSO 回退及对应路径计数。 |
| `MYDS_QKV_NR0=4 bin/test_qkv_fusion`、`MYDS_QKV_NR0=2 bin/test_qkv_fusion` | 两种独立行配置、BF16 舍入边界、RoPE/量化/窗口写入、奇数输出行、别名与 copy 失败回退；局部 concurrent scope 比较实际投影形状的 QKV/HC/32-head Indexer，验证 opt-in/trace 门禁及失败后恢复两种设置；均在 `make test` 中执行。 |
| `make test-tail-init` | 真实缺 tail PSO 和必需 PSO 创建失败后的全状态清理、重复失败、恢复完整库及 lazy 初始化重试。 |
| `bin/test_engine`、`bin/test_p1` | 生成、前缀恢复与快照边界、缓存失败路径回归。 |
| `bin/test_server_api`、`tests/test_http_bytes.py` | FIFO SSE 输出、背压/断连取消、排空期限与实际 HTTP 字节协议；由 `make test` 执行。 |
| `bin/test_engram`、`bin/test_kvdsk`、`bin/test_metal` | LUT 数值、共享段提交/覆盖/损坏/budget、Metal 特化缓存与基础算子。 |
| `python3 tests/test_bench_reproduce.py` | 全 CPU：使用保存的真实 metrics 验证汇总，用 subprocess stub 验证输入/日志绑定、非 greedy 拒绝、超时/失败清理；由 `make test` 执行，stub 不运行 GGUF 或 Metal。 |

本轮原始日志收录于 `docs/benchmarks/2026-10-05-optimization/validation/`：
`workspace-build.log`、`workspace-test.log`、`workspace-tail-init-test.log`、
`workspace-policy-retry-test.log`。`mandatory-pso-before-test.log` 是修复前故障复现，
其失败是预期证据；`mandatory-pso-after-test.log` 是修复后的通过记录。
打包时新增的测量入口 CPU 回归单独记录在 `reproduction-tool-test.log`，
探针构建与补丁检查记录在 `replay-tools/cpu-validation.json`。
本地 fixture 通过的范围是这些断言，完整模型的输出与性能另由下面的工具验证。

#### 完整模型每 token 测量

先在 Mac Studio 安排独占窗口：确认生产请求结束，暂停推理服务和其他 GPU benchmark，
完成后通过 deploy 的 `start_fleet.sh` 恢复并检查 `/health`。
`tools/bench_reproduce.py run` 会检查已知推理/benchmark 进程与 8181 监听，
发现仍在运行时退出；服务暂停与恢复由窗口管理者负责。

```sh
MYDS_REPRO_MODEL="~/.lmstudio/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q2.gguf"
python3 tools/bench_reproduce.py run \
  --binary "$PWD/bin/bench_phase" \
  --model "$MYDS_REPRO_MODEL" \
  --request "$PWD/docs/benchmarks/2026-10-05-optimization/requests/normal.json" \
  --out "$MYDS_REPRO_OUT/candidate-normal"
```

输出目录必须是新的。`run-metrics.jsonl` 保存 metadata、模型加载信息、预热和三轮正式样本，
包括实际 token 数、原始 token 数组、输出字节哈希、逐 token 回调间隔；
`run.log` 保存引擎 stdout/stderr；`run-state.json` 保存完整 argv、显式环境、
二进制/metallib/请求 SHA、模型文件大小与 mtime、运行前后硬件/内存/swap 状态及退出码；
`run-summary.json` 仅在样本校验通过后产生。模型身份记录为文件元数据，未对整个 GGUF 做 SHA。
metadata 的 environment 额外记录 `MYDS_ATTN_VEC_HEAD_FIRST`、`MYDS_ATTN_GATHER_MIN_KEYS`、
`MYDS_PREFILL_ATTN`、`MYDS_QKV_FUSE_BF16`、`MYDS_QKV_FUSE_POST`、`MYDS_QKV_NR0`、
`MYDS_MOE_IQ2_TAIL_CULL`。每个样本含 `attention_paths`（本轮 attention 调用、gather/slab、
head-first 尝试/成功/失败/缓存回退、原 vec、matrix 及其回退次数）；出现回退时 stderr 另有一行提示。
Engram I/O 配置另记录 `MYDS_ENGRAM_IO_THREADS`、`MYDS_ENGRAM_NOCACHE`。
直接调用底层工具时，`MYDS_BENCH_STORAGE_COUNTERS=1` 在每次请求计时前后记录
`IOBlockStorageDriver` 的全局计数器；采集时间不计入请求耗时。这些是操作系统块设备
读计数，不能单独归因到本进程，也不能当作 NAND 流量。
`run-summary.json` 与 `compare` 输出 `attention_paths`，`run` 结束时打印 `attention_fallback`。
2026-10-05 证据由加入这些字段之前的 bench_phase 生成，样本中没有该字段，汇总显示为 `null`。

入口清除全部继承的 `MYDS_*` 环境，并显式设置 prefix cache、KV disk、phase profiler 为 0，
chunk=512、MoE chunks=9、suffix shift=512、residency keep-alive=86400。
`--env MYDS_NAME=value` 可以重复传入，例如 `--env MYDS_ATTN_VEC_HEAD_FIRST=0`；
缓存命中、profiler 和跳过计算的配置会被拒绝。三个请求样本为
`requests/short.json`、`normal.json`、`long.json`，分别实测 prefill 356、1830、7194 token。
默认生成上限 64；短请求实际 EOS 为 26 token，decode 分母因此为 25。

底层工具也可以直接调用：
`bin/bench_phase model.gguf request.json [repeats=3] [max_new_tokens=64] [max_seq_len=262144]`。
输出上限支持 32/64；温度、top-p、top-k、min-p、重复惩罚和 seed 在工具中固定为
0/1/0/0/1/42，greedy 路径不读取采样 RNG，请求 JSON 中的 sampling 字段不覆盖它。
prefill 口径为 `1000 * prefill_seconds / prefill_tokens`，最终 prompt token 在首次 decode
forward 消耗；decode 为 `1000 * (last_callback-first_callback) / (output_tokens-1)`，
不含首次 forward/TTFT。加载、HTTP 传输、队列等待不计入这两个指标。

#### Engram SSD 并行读取

默认使用 8 个读取线程；`MYDS_ENGRAM_IO_THREADS=1` 使用原串行读取；`2..16` 使用每个 context 独立拥有的
常驻线程池。只并行当前 token 的两张表共 48 行 raw `pread`，解码仍在调用线程按
原顺序执行。全部 worker 结束、读取与解码全部成功后才提交 history/position；
任意失败均返回错误，池初始化失败不会静默改变实际线程数。模型加载日志记录所选配置。

Mac Studio M2 Ultra 的同输入独占实测：Engram 冷首轮 CPU 路径由
1.197 降到 0.249 ms/token，完整模型冷首请求 prefill 由 4.339 降到
3.346 ms/token、decode 由 41.779 降到 37.460 ms/token。缓存热态存在调度成本：
普通 prompt 的 prefill 增加 0.059 ms/token、decode 增加 0.414 ms/token。
默认 8 线程优先降低 SSD 未命中时延；已知长期热缓存的负载可选择 1 线程。
这些是固定 prompt 的实测，冷首请求的 decode 已经过 prefill 暖页，不能解释为全部 lookup 都冷。

CPU 工具走真实 GGUF 的 `myds_engram_prepare_token`，包含 hash、派发、等待、读取和解码；
它不加载主干/GPU。`--tokens-bin` 接受 little-endian int32，`--text` 用模型分词器且
不额外加 BOS；完整 HTTP prompt 的精确输入导出工具见本轮协议。

```sh
make bin/bench_engram_io
python3 tools/run_engram_io.py --bench "$PWD/bin/bench_engram_io" \
  --model "$MYDS_REPRO_MODEL" --tokens-bin /absolute/path/to/tokens.bin \
  --output /absolute/path/to/new-results --order 1,4,8,8,4,1 --warm-runs 2
python3 tools/summarize_engram_io.py /absolute/path/to/new-results
python3 tools/test_engram_pool_failures.py --sanitizer address
python3 tools/test_engram_pool_failures.py --sanitizer thread
```

冷读候选样本使用同内容 APFS clone；每轮在模型/线程池初始化完成后，通过 stdin 闸门
采集读取阶段的设备计数器。不同 inode、`F_NOCACHE` 或应用层逻辑字节都不能独立证明冷读。
首轮与重复热态轮分别汇总，逐 token 行 ID、解码向量 hash、history 和跨配置一致性必须通过。
普通单元测试、线程/同步初始化故障注入、EINTR/正短读、并发、测量工具证据校验和
mock 独占窗口恢复测试纳入 `make test`。
完整模型与 CPU 读取的日志、复现步骤和结果见
[Engram I/O 实测结论](docs/benchmarks/2026-10-05-engram-io/assessment.txt)及
[复现协议](docs/benchmarks/2026-10-05-engram-io/protocol.txt)。

#### 构建同轮基线并比较

同轮基线是本轮优化前的工作区，已含 causal mask 修复；使用已提交快照补丁恢复，
不能直接用旧部署版本代替：

```sh
MYDS_REPRO_BASE="$MYDS_REPRO_OUT/base-source"
mkdir "$MYDS_REPRO_BASE"
git archive f0a9c228ed1bac8e72b8a78e50f87ea89d807b73 | tar -x -C "$MYDS_REPRO_BASE"
git -C "$MYDS_REPRO_BASE" apply --check \
  "$PWD/docs/benchmarks/2026-10-05-optimization/protocol/baseline-working-changes.patch"
git -C "$MYDS_REPRO_BASE" apply \
  "$PWD/docs/benchmarks/2026-10-05-optimization/protocol/baseline-working-changes.patch"
make -C "$MYDS_REPRO_BASE" -B -j8 all 2>&1 | tee "$MYDS_REPRO_OUT/base-build.log"
xcrun clang -O3 -Wall -Wextra -std=c11 -I"$MYDS_REPRO_BASE/include" \
  -c tools/bench_phase.c -o "$MYDS_REPRO_BASE/build/bench_phase.o"
xcrun clang "$MYDS_REPRO_BASE"/build/myds_*.o "$MYDS_REPRO_BASE/build/bench_phase.o" \
  -framework Foundation -framework Metal -lm -lpthread -o "$MYDS_REPRO_BASE/bin/bench_phase"
python3 tools/bench_reproduce.py run \
  --binary "$MYDS_REPRO_BASE/bin/bench_phase" --model "$MYDS_REPRO_MODEL" \
  --request "$PWD/docs/benchmarks/2026-10-05-optimization/requests/normal.json" \
  --out "$MYDS_REPRO_OUT/base-normal"
python3 tools/bench_reproduce.py compare \
  --base "$MYDS_REPRO_OUT/base-normal" --candidate "$MYDS_REPRO_OUT/candidate-normal" \
  --out "$MYDS_REPRO_OUT/normal-comparison.json"
```

每个版本使用独立进程，预热一次后正式三次。compare 重新校验原始记录，要求输入配置、
模型文件元数据、机器配置、全部 token 数组及输出字节哈希一致，再比较三轮中位数并保留
最小/最大值。实验 matrix 路径若改变输出，严格 compare 会失败，不能按严格一致优化报收益。
本轮长输入两轮变慢的数据在 `final-summary.json` 和
`head-first-long-validation.json` 中均保留；重新测试也应保留全部正式样本。

#### 层级归因、组件回放与已保存证据

| 文件/目录 | 用途与复现边界 |
| --- | --- |
| `optimization/raw/exclusive-20261005T022233Z/` | 主对照的六组 `*-metrics.jsonl`、引擎 `.log`、独占窗口事件 `window.jsonl`、服务恢复 `restore.log`。 |
| `optimization/raw/exclusive-20261005T023349Z/` | head-first 长输入补充对照，包含 host 诊断计数，独立于主对照。 |
| `optimization/protocol/summarize-final.py` | 对主窗口做 CPU 审计，验证完整结束、服务恢复、退出码、实际分母和输出，再生成 JSON/CSV。 |
| `optimization/replay-tools/recipe.json` | attention/MoE/decode 独立探针、实验 Metal 补丁、实际中间量采集、GGUF tensor layout 提取的源码来源、构建和运行命令。 |
| `optimization/component-provenance-audit.json` | 区分真实权重、真实中间量和合成输入；QKV 使用真实选定权重及合成非零 activation，decode IQ2 使用捕获的 prefill 末行。 |
| `optimization/replay-input-manifest.json` | 真实捕获文件的 SHA/大小。约 503 MB 的 activation 数据需按 recipe 重新采集，不包含在 Git 中。 |
| `phase/instrumentation.patch`、`phase/layer-profile.patch`、`phase/aggregate-layer.py` | 历史分层仪器及聚合脚本。GPU timestamp/counter 可用性与 profiler 扰动记录在 phase manifest；仪器结果用于归因，整体耗时采用关闭仪器的对照。 |
| `optimization/protocol/run-*.py`、`phase/run-exclusive.py` | 本轮执行过的历史窗口脚本，固定了当时的临时目录、fleet/model 路径、端口和生产 SHA。新窗口使用参数化 `bench_reproduce.py`；复用历史脚本前须准备其依赖并逐项核对现场配置。 |

上表中的 `optimization/`、`phase/` 分别指
`docs/benchmarks/2026-10-05-optimization/` 和 `docs/benchmarks/2026-10-05-phase/`。
仅复算已保存主对照数据，无需 GPU 或 GGUF：

```sh
python3 docs/benchmarks/2026-10-05-optimization/protocol/summarize-final.py \
  docs/benchmarks/2026-10-05-optimization/raw/exclusive-20261005T022233Z \
  --out "$MYDS_REPRO_OUT/saved-result-audit"
```

QKV 探针单独编译，`MYDS_QKV_BENCH_FUSED` 宏和 `--model` 均需提供，才能复现本轮融合对照：

```sh
xcrun clang -O3 -Wall -Wextra -Iinclude -fobjc-arc -DMYDS_QKV_BENCH_FUSED \
  tools/bench_qkv.m src/myds_v41_gpu.c src/myds_gguf.c src/myds_log.c \
  -framework Foundation -framework Metal -lm -lpthread -o bin/bench_qkv
MYDS_QKV_NR0=4 bin/bench_qkv --layout /path/to/tensor-layout.json \
  --model "$MYDS_REPRO_MODEL" --shader-dir "$PWD/bin" --layer 0 --samples 9 --reps 64 \
  > "$MYDS_REPRO_OUT/qkv-nr4.csv" 2> "$MYDS_REPRO_OUT/qkv-nr4.log"
```

layout 按 replay recipe 从同一 GGUF 提取。QKV 的 stdout 含启动信息及 CSV，历史文件虽用
`.jsonl` 后缀，审计按 CSV 处理；新复现采用 `.csv`。组件 probe 只给出该组件、该输入的计时，
不可将多个独立回放中位数相加后当作整层或整模型耗时。
`.gitattributes` 保留证据的原始换行，`bench_reproduce.py verify` 校验两轮证据 manifest；
`--check-source` 同时验证本次打包源码，基准是 `afcc05c` 中的测量快照；之后的提交
（例如 attention 回退计数）会被报告为漂移，可用 `git diff afcc05c -- <file>` 查看，证据 manifest 不随之改写。
服务部署后 SHA、模型或环境变化时，应另存新的结果目录。

## 视觉（图片输入）

支持图片输入（DeepSeek-V4.1-Flash-Vision），启动加 `--vision <编码器.gguf>`。
使用、原理、验证与**明确不做**的范围：见 **[docs/VISION.md](docs/VISION.md)**。
