# myds: Minimal DeepSeek Inference Engine on Apple Silicon Metal

`myds` 是一个专为 **DeepSeek** 系列模型量身打造的极简、高性能本地推理引擎。它完全基于 C / Objective-C 编写，去除了非 DeepSeek 架构的一切冗余代码，直接调用 Apple Silicon 原生 **Metal** 图形与计算框架加速，具备毫秒级冷启动、零拷贝内存映射与流式生成能力。

---

## 核心特性

- **专注于 DeepSeek 架构**：
  - **RMSNorm** 归一化加速
  - **MLA (Multi-head Latent Attention)** 解耦多头潜在注意力与低秩压缩 KV Cache
  - **Decoupled RoPE** 解耦旋转位置编码
  - **DeepSeekMoE** 混合专家网络（Shared Experts 共享专家 + Top-K 路由专家，Sigmoid 门控与权重重整化）
  - **SwiGLU** 激活函数融合计算
- **纯正 Apple Silicon Metal 驱动**：
  - 从 `ds4` 迁移并提炼出 20 个 DeepSeek 专有 `.metal` 着色器核函数
  - 利用统一内存架构（Unified Memory）通过 `MTLResourceStorageModeShared` 实现 CPU 与 GPU 之间零拷贝
  - 运行时即时编译优化，原生发挥 M 系列芯片（M1/M2/M3/M4/M5 等）的最大能效
- **极简架构与零依赖**：
  - 核心推理引擎仅由几个高内聚、低耦合的模块组成，无需任何外部第三方库（仅依赖系统级 Foundation 与 Metal 框架）
  - 内置极简 GGUF v2/v3 解析器，支持 `mmap` 秒级映射大模型权重
  - 内置 DeepSeek BPE 分词器，支持流式文本解码与交互式 Chat 模式
  - `make` 2 秒内完成全量构建，内置完整的单元测试套件

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
│   ├── dsv4_hc.metal                 # DeepSeek 混合缓存矩阵运算
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
│   ├── myds.h                        # 推理引擎主接口 (模型加载、上下文、生成)
│   ├── myds_metal.h                  # Metal GPU 运行时与算子声明
│   ├── myds_gguf.h                   # 极简 GGUF 格式解析器
│   └── myds_tokenizer.h              # DeepSeek BPE 分词器
├── src/                              # 核心源码
│   ├── main.c                        # 极简命令行与交互终端入口
│   ├── myds_engine.c                 # DeepSeek Transformer 前向传播流程
│   ├── myds_metal.m                  # Metal 运行时与 GPU 算子分发
│   ├── myds_gguf.c                   # GGUF 格式解析与 mmap 内存映射
│   └── myds_tokenizer.c              # BPE 编码与流式解码
└── tests/
    └── test_metal.c                  # GPU 核心算子金标对齐与正确性测试
```

---

## 编译与测试

### 1. 编译项目

在项目根目录下直接运行：

```bash
make
```

将在 `bin/` 目录下生成 `myds` 主程序以及 `test_metal` 算子测试程序。

### 2. 运行单元测试套件

```bash
make test
```

测试流程将自动编译并依次运行 3 个独立的单元测试套件，共计 17 项金标对齐测试：

1. **GGUF 模型解析器测试套件 (`bin/test_gguf`)**：
   - [x] GGUF v2/v3 标量、字符串、布尔及动态数组元数据提取
   - [x] 张量描述信息校验与 32 字节对齐 mmap 内存映射验证
   - [x] 非法魔数（Magic）、损坏文件及不存在文件的容错处理

2. **DeepSeek BPE 分词器测试套件 (`bin/test_tokenizer`)**：
   - [x] 词表加载与 BOS/EOS 特殊 Token 识别
   - [x] BPE 贪婪分词与前缀最长匹配
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

---

## 快速上手

### 命令行生成

指定 DeepSeek GGUF 模型文件并输入提示词：

```bash
./bin/myds -m /path/to/deepseek.gguf -p "你好，请介绍一下你自己。"
```

### 交互式对话模式

如果未指定 `-p` 参数，`myds` 会自动进入交互式对话模式：

```bash
./bin/myds -m /path/to/deepseek.gguf
```

### HTTP 服务器模式 (OpenAI 兼容 API)

`myds` 内置轻量级、低延迟的 POSIX HTTP 服务端，可直接接入算力舰队或作为 OpenAI 兼容端点：

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