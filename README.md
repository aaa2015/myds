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
| `MYDS_ATTN_VEC_HEAD_FIRST` | 启用 | gathered prefill 的 vec attention 将 query/head 物理 grid 调换，保留每个 head 的算术与归约。设为 `0` 使用原排列。旧 metallib 缺新 PSO 时自动回退，并在 scratch 中缓存不可用状态。 |
| `MYDS_PREFILL_ATTN` | `vec` | `matrix` 与 `auto` 是 compressed gather 的实验选项。矩阵归约会产生浮点差异，实测长输入生成 token 可变化；默认关闭。`auto` 的矩阵 PSO/共享内存不可用时回退 vec，显式 `matrix` 则返回失败。 |
| `MYDS_QKV_FUSE_BF16` | 启用 | decode 的 Q8 QKV 投影把 BF16 舍入合并到 GEMV 写出；设为 `0` 恢复独立舍入 dispatch。trace 使用原路径。 |
| `MYDS_QKV_FUSE_POST` | 启用 | decode 合并 Q/K RoPE、KV 量化及窗口写入，保留原舍入边界；设为 `0` 恢复原 dispatch。缺融合 PSO、别名或 copy 故障注入时回退原路径。 |
| `MYDS_QKV_NR0` | `4` | 融合 Q8 GEMV 每组处理四个独立输出行；`2` 使用两个行的配置。归约顺序不变。 |
| `MYDS_MOE_IQ2_TAIL_CULL` | 启用 | prefill MoE IQ2 up/gate 的尾 tile 跳过没有有效 token 的半 tile。`0` 恢复原 kernel；默认缺新 PSO 时回退原 kernel，显式 `1` 缺 PSO 时初始化失败并清理所有部分状态。 |
| `MYDS_SSE_DRAIN_TIMEOUT_MS` | `1000` | SSE 完成时的整体排空期限，合法范围 `1..30000` ms；非法值使用默认值。 |

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

## 视觉（图片输入）

支持图片输入（DeepSeek-V4.1-Flash-Vision），启动加 `--vision <编码器.gguf>`。
使用、原理、验证与**明确不做**的范围：见 **[docs/VISION.md](docs/VISION.md)**。
