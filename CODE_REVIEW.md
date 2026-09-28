# myds 代码评审报告 — 架构 / 性能 / 正确性

> **用途**：本文件是 myds 的评审结论与优化任务清单，供后续优化（如 Gemini）直接消费。
> **评审对象**：本仓库（`~/Projects/myds`，从 `ds4` 抽取的 DeepSeek V4/V4.1 推理引擎）
> **日期**：2026-09-28 ｜ **性质**：只读评审（评审过程未改动本仓库任何文件）
> **复核环境**：macmini（Apple M4 / 10 核 / 16 GB / macOS 27.0）。本文所有标「实测」的数字都在该机器上取得。

## 如何使用本文

1. §0 给结论与定级；§3 是**必须修**的缺陷（每条含 现象/证据/根因/修法/验收）；§5 是**可直接执行**的任务清单。
2. 每个任务都有**验收门禁**——请按门禁验证，不要只跑通 `make`。
3. 引用格式 `file:line`，行号为 2026-09-28 的仓库状态；改动后行号会漂移，请按**函数名/符号名**定位。
4. 术语：**实测** = 本环境跑出来的数字；**源码** = 读码确认；**推测** = 未证（文中已标注）。

---

## 0. 结论速览

**一句话**：这不是「Metal 推理引擎」，而是**一个 CPU 标量推理引擎外挂了 2 个 Metal 算子**；同时它**装载不了真正的 DeepSeek V4.1 权重**，也**算不对 MoE**。**当前状态不可进舰队。**

### 定级

| 维度 | 定级 | 关键依据 |
|---|---|---|
| 权重装载 | 🔴 阻断 | 每个张量整份拷贝一份到 MTLBuffer ⇒ 常驻 **≈ 2× 模型**（§3.1） |
| 计算正确性 | 🔴 阻断 | MoE 路由算完**丢弃**；MLA / attention sinks / hyper-connection 完全缺失（§3.2/3.3） |
| GPU 利用率 | 🔴 阻断 | 运行期只 dispatch **2 个** kernel；matmul/attention/RoPE/softmax/add **全在 CPU 单线程**（§4.1） |
| 吞吐 | 🔴 阻断 | GEMV 实测 **4.33 GB/s**（同机 8 线程流读 77.5 GB/s）；attention 单层单 token **7.3 ms @ctx2048**（§4.2） |
| 服务端可用性 | 🟠 严重 | `Content-Length` 解析**永远失败**；无 recv 超时；SSE 可产出非法 UTF-8（§3.5） |
| 测试可信度 | 🟠 严重 | 3 个测试套件测的是**自己复制的实现**，不链接生产代码（§2.8） |
| 工程/部署 | 🟡 中 | shader 运行期编译 **1.9 s**；硬编码绝对路径；依赖 cwd；无 `-D` 通道（§2.9） |

### 一个关键事实（决定优化策略）

**20 个 `.metal` 与上游 `ds4` 逐字节相同（md5 20/20 SAME，实测）。**
⇒ 抽出来的只是 **host 层**，而 host 层重建时「抄了参数、漏了语义」。
⇒ **正解是移植 `ds4` 的 host 调度层**（`~/Projects/ds4/ds4_metal.m` 的批命令缓冲 + 参数契约 + 权重视图），**不要继续逆向重建**。

### 开工前必须知道的三条禁令

1. **绝不要把 myds 指向 macstudio**，直到 §3.1 修完。（现状 = 2× 模型内存，会复现内存风暴 → Jetson → 内核 panic 类事故）
2. **不要改 `.metal` 文件**，除非确有必要并同步回 `ds4`（两边逐字节相同是当前唯一的一致性保障）。
3. **不要为了让测试通过而改测试**去迁就实现（§2.8 已经因为"测复制品"而失真）。

---

## 1. 评审方法与证据来源

| 来源 | 内容 |
|---|---|
| 逐文件读码 | `include/*` + `src/*`（engine/metal/server/gguf/tokenizer/main/log）+ `Makefile` + `tests/*` |
| 本机实测 | 运行期 shader 编译耗时、GEMV 有效带宽、buffer churn、RoPE/attention 单 token 成本、不支持量化类型的行为 |
| 内核走查（并行子代理） | `moe/dsv4_misc/argsort/set_rows/sum_rows` 与 `dense/flash_attn/dsv4_hc/dsv4_kv/dsv4_rope/dsv41` 两组；含 kernel 清单、host 契约、`DS4_METAL_HAS_TENSOR` 裁剪面、351 个 kernel 符号实测、RMSNorm 派发网格实验 |
| 对照上游 `ds4`（仓库外只读） | host 调度面、批命令缓冲、零拷贝、真实张量名、线程池、批预填充 |
| 对照 `deploy` | 舰队配置（8181 行）、点火方式、cwd 语义 |

**实测装置**：`/tmp` 侧独立程序，编译本仓库的 `src/myds_metal.m`，不修改仓库。复现命令见 §8。

---

## 2. 架构评审

### 2.1 分层与依赖

```
main.c ──> myds.h (model/context 生命周期 + generate)
   ├─ myds_server.c ──> myds.h            HTTP/SSE + 手写 JSON 解析
   ├─ myds_engine.c ──> myds_metal.h      Transformer 前向 + 模型装载 + 采样
   │        └─ myds_gguf.c (mmap)  myds_tokenizer.c  myds_log.c
   └─ myds_metal.m ──> Metal/Foundation   运行期编译 20 个 .metal + 缓冲管理
```

- ✅ **边界方向干净**：engine 不掺 ObjC、metal 层不掺模型语义，`include/myds_metal.h` 的算子契约清晰。**这部分设计应当保留。**
- ❌ **契约实现与头文件声明不符**：`include/myds_metal.h:76` 声明 `myds_metal_moe_sum`、`:91` 声明 `myds_metal_softmax`，两者都是**纯 CPU 实现**（`src/myds_metal.m:590`、`:629`），而 GPU pipeline 建了却不用（§4.1）。**头文件在撒谎**，是后续接线最大的认知陷阱。
- ❌ **引擎层没有「层」的抽象**：`src/myds_engine.c:13-34` 的 `myds_layer_t` 只有 `attn_q/k/v/attn_output + ffn_gate/up/down` 这套「MHA+GQA 稠密」字段。DeepSeek V4/V4.1 的层结构（MLA 低秩 + 融合潜 KV + attention sinks + hyper-connection + 路由专家）**没有对应载体**：加一个能力要同时改结构体、加开关、改装载、改前向。
  → **建议**：改成**名字表驱动**的权重描述（上游 `ds4.c` 用字面量表驱动，`"blk.%u.attn_q_a.weight"` 之类），新增能力只加表项。

### 2.2 内存架构：★最严重★

`src/myds_engine.c:95-101`

```c
static myds_metal_buffer_t *create_metal_buffer_from_tensor(const gguf_tensor_t *t) {
    if (!t || !t->data) return NULL;
    myds_metal_buffer_t *buf = myds_metal_buffer_alloc(t->size_bytes);   /* newBufferWithLength */
    if (!buf) return NULL;
    myds_metal_buffer_write(buf, 0, t->data, t->size_bytes);            /* :99 整份 memcpy */
    return buf;
}
```

- GGUF 已经 `mmap(PROT_READ, MAP_SHARED)`（`src/myds_gguf.c:179`），`t->data` 直接指向映射区（`src/myds_gguf.c:272`）——**天然零拷贝**。这里却对**每个张量**再 `newBufferWithLength`（`src/myds_metal.m:215`）+ `memcpy`。
- **后果**：进程常驻 ≈ `mmap(全模型)` + `MTLBuffer(全模型)` ≈ **2× 模型**。
- 舰队 `fleet.conf:102` 的 8181 行指向的就是这个 V4.1 Flash Q2（上游 ds4 侧常驻 153 GB / 机器 192 GB）⇒ **myds 一起就会 OOM**。
- **修法**：`[device newBufferWithBytesNoCopy:length:options:MTLResourceStorageModeShared deallocator:nil]`（要求页对齐；把 GGUF 张量区按 `vm_page_size` 对齐映射即可）。上游 `ds4` 用 2 处 `newBufferWithBytesNoCopy` 做同样的事。
- **验收**：`ps -o rss= -p <pid>` 常驻 ≈ 模型文件大小 ±5%（而不是 2×）。

### 2.3 GPU 使用面：声明 ≠ 执行

`src/myds_metal.m:84-120` 编译了 20 个文件（**实测库里有 351 个 `kernel*` 符号**），但只取 6 个函数（`:152-189`）、真正 dispatch **2 个**（`:301` rmsnorm、`:412` swiglu）：

| 算子 | 头文件声明 | 实现 | 实际执行 |
|---|---|---|---|
| RMSNorm | `myds_metal.h:33` | GPU（`kernel_rms_norm_mul_f32_4`） | ✅ GPU |
| SwiGLU | `myds_metal.h:60` | GPU（`kernel_swiglu_f32`） | ✅ GPU |
| **GEMV** | `myds_metal.h:51` | **CPU 标量三重循环**（`myds_metal.m:465`） | ❌ `g_pso_mul_mv_q8_0` 建了 PSO，**全仓零 dispatch** |
| Q4_K GEMV | — | 无（CPU 路径也没有 case） | ❌ 且 `kernel_mul_mv_q4_K_f32`（`myds_metal.m:185`）**库里不存在**（实测 0 命中） |
| RoPE | `myds_metal.h:41` | CPU（逐头逐对 `powf/cosf/sinf`） | ❌ |
| Attention | — | **CPU**：`myds_engine.c:530-578` 三层循环 | ❌ 明明有 `metal/flash_attn.metal`（1692 行） |
| Softmax | `myds_metal.h:91` | CPU | ❌ 且 `newFunctionWithName:@"kernel_soft_max"`（`myds_metal.m:170`）**返回 nil**（实测；真实名是 `kernel_soft_max_f32` / `_f32_4`） |
| Add | `myds_metal.h:84` | CPU | ❌ `g_pso_add`（`myds_metal.m:164`）建了不用 |

⇒ 每个 token 的算力热点（GEMV 占 >90% 时间）**一行 GPU 代码都没跑**。

### 2.4 同步与命令缓冲模型

- **现状**：每个算子 = 新建 command buffer + `commit` + **`waitUntilCompleted`**（`src/myds_metal.m:304-305`、`:415-416`、`:661-662`），CPU/GPU 全串行、零重叠。
- 更糟：**CPU 与 GPU 交替写同一块 shared buffer**，把 rmsnorm/swiglu 的加速收益又吃掉一半。
- **接 GPU 后会成为新瓶颈**：28 层 × 每层 8~10 个算子 ≈ **250~280 次全同步/token**。
- 上游 `ds4` 的模型是**一个 batch command buffer 承载整个前向**（`ds4_metal.m` 的 `g_batch_cb` / `g_batch_enc`，271 处 dispatch 共用）。
- **修法**：先落「批命令编码器」（`begin/commit` 一对 CB 编一整 token），再谈算子接线。**顺序不能反。**

### 2.5 上下文与会话模型

- `myds_generate` 入口 `ctx->pos = 0`（`src/myds_engine.c:646`）⇒ **每次都从零重算**：无 KV 复用、无 prefix cache、无多轮上下文（`src/main.c` 的交互模式每轮都重置，等于每轮都是新会话）。
- **预填充无批处理**：prompt 逐 token 走完整前向（`src/myds_engine.c:650-654`，还顺带采样一次），TTFT ≈ N × 单 token 成本。
  上游 `ds4` 有 `prefill_chunk` + `matmul_q8_0_batch_worker` + `ds4_parallel_for` 线程池（`ds4.c` 里 `n_tokens` 出现 1127 次）——**批预填充收益通常 10~50×**，这层完全没抽出来。
- **KV cache 是 fp32 稠密**（`src/myds_engine.c:369-372`：`calloc(layers)` + `calloc(seq*kv_heads*head_dim)`），无量化、无 MLA 潜向量压缩。
  对照舰队配置 **CTX=131072**：按 `2 × n_layers × n_kv_heads × head_dim × ctx × 4B` 直接爆。真 V4.1 的 KV 是**潜向量 + 量化 tail**（`metal/dsv4_rope.metal:508` 的 `kernel_dsv4_kv_rope_fp8_store_f32` 里能看到 FP8 前缀 + 512 维 head_dim 的布局）。

### 2.6 服务端架构

- **同步单线程**：`src/myds_server.c` 的 accept 循环里直接调 `handle_client()`，推理全局互斥（`g_infer_mutex`）⇒ 并发=串行；**慢客户端会锁死整个引擎**（无 `SO_RCVTIMEO`、无请求体上限）。
- `Content-Length` 处理（`src/myds_server.c:505`）：`json_extract_int_field(req_buf, "Content-Length", 0)` 内部拼的是**带引号的 needle**（`"Content-Length"`），而 HTTP 头没有引号 ⇒ **永远返回 0** ⇒ 补读 body 的循环 `while (body_have < content_len …)` 永不执行。分片到达的大 body 被**静默截断**（退化成 `"Hello"`）。
- **SSE**：每个 token 片段单独 `json_escape` + 单次 `send`（`src/myds_server.c:357-361`、`:379-384`）。
  ① `send` 返回值未处理（14 处 `send(`，无一处检查，大响应/慢客户端会静默丢字节）；
  ② **byte-fallback token 产生的半个 UTF-8 序列会原样进 JSON**（中日韩输出必现，客户端可能直接报错）——上游 llama.cpp / ds4 都做跨片段 UTF-8 缓冲。
- `usage.input_tokens` **硬编码 10**（`src/myds_server.c:609`、`:637`、`:699`）；`finish_reason` 恒为 `"stop"`（`:676`、`:698`，撞 max_tokens 也报 stop）。
- 手写 JSON 解析（`strstr` + 引号 needle）在**值里含 key 名**时会误命中；无 `tool_calls` / `tool` role 支持。
- **10 处 `cur_len += snprintf(out+cur_len, max_prompt_len-cur_len, …)` 无溢出保护**：一旦 `cur_len > cap`，`max_prompt_len - cur_len` 因 `size_t` **下溢**为巨值 ⇒ **越界写**。目前靠 cap≈2×body 侥幸压住，属**潜在远程堆溢出**，必须改成统一 append 助手。

### 2.7 错误处理

- **算子返回值全线被忽略**：`myds_metal_gemv` 对不支持的量化类型 `return 0`（`src/myds_metal.m:534`），引擎侧不检查 ⇒ **权重被静默跳过、输出垃圾、零日志**。
- **张量缺失也静默**：装载时 `if (tq) {…}` 拿不到就留 NULL（`src/myds_engine.c:210-260`），前向再 `if (layer->attn_q)` 跳过（`:503-511`）⇒ **模型少一半权重也照跑不报错**。
- `myds_metal_buffer_alloc` 的 `calloc` 未判空（`src/myds_metal.m:215-220`）；`myds_context_create` 不检查任何分配失败；`malloc` 未判空见 `src/myds_server.c:620/626/630/685/691/695`。

### 2.8 测试架构（信任度问题）

`Makefile` 的链接关系暴露了真相：

```
test_engine:     仅 test_engine.o      # 不链接 myds_engine.o ⇒ 测不到真采样器
test_server:     仅 test_server.o      # 不链接 myds_server.o ⇒ 测的是复制品
test_server_api: 仅 test_server_api.o  # 同上
```

`tests/test_server.c:8` 自带一份 `test_find_matching_brace`，`tests/test_engine.c:9` 自带 `sample_greedy/apply_softmax` —— **测的是自己抄的实现**。而真正的 `sample_next_token` 是 `static`（`src/myds_engine.c:420`），**结构上不可测**。
只有 `test_metal / test_gguf / test_tokenizer / test_log` 链接了生产对象。

⇒ README 里那份「7 套件全绿」的真实语义是：**4 套覆盖部分生产代码，3 套自我循环**。**测试全绿不能作为「引擎正确」的证据。**

### 2.9 构建与部署

- **shader 运行期编译**：`newLibraryWithSource`（`src/myds_metal.m:142`）现场编译 1.1 MB Metal ⇒ **实测 1.907 s**（README「毫秒级冷启动」不成立）。收益点是**离线 metallib + 嵌入二进制**（上游 `ds4` Makefile 已把 `metal/*.metal` 作为 `ds4_metal.o` 的依赖，见 `METAL_SRCS`）。
- **路径解析**（`src/myds_metal.m:110-118`）：`"metal/x"` → `"./metal/x"` → `~/Projects/myds/metal/x`（**硬编码绝对路径**）。所以从别的 cwd 起也能跑，**但换机器/换用户名/换目录就必然启动失败**（舰队 `start_fleet.sh` 是 `env … $CURRENT_BIN … &`，**不 cd**）。
- **关键常量只存在于 host 的 C 字符串 preamble 里**（`src/myds_metal.m:27-80`，如 `:40 #define N_R0_Q8_0 2`、`:43 #define FC_MUL_MV 600`），且**没有 `-D`/`preprocessorMacros` 通道**（`:133` 只设了 `mathMode`）⇒ 这 20 个 `.metal` **无法独立编译**，任何调参都要改 host 的 C 字符串。
- `Makefile`：`-O3 -Wall -Wextra` 但无 `-ffast-math`、无 `-mcpu`、无 LTO、无 `-arch`（上游 `ds4` 有 `NATIVE_CPU_FLAG` + `-ffast-math`）；**无任何 Metal 离线构建目标**。

---

## 3. 正确性缺陷清单

### P0-1 权重双份 ⇒ 2× 内存 🔴

- **现象**：常驻内存 ≈ 2× 模型大小。
- **证据**：`src/myds_engine.c:95-101`（`memcpy` 在 `:99`）+ `src/myds_gguf.c:179/272`（已 mmap）。
- **根因**：没用 `newBufferWithBytesNoCopy`。
- **修法**：见 §5 T0-2。
- **验收**：`ps -o rss=` ≈ 模型大小 ±5%。

### P0-2 MoE 路由算完即丢 🔴

- **现象**：MoE 层只贡献 shared expert，**输出确定性错误，零日志**。
- **证据**：`src/myds_engine.c:586-608`

  ```c
  if (m->n_routed_experts > 0 && layer->router_gate) {
      …
      myds_metal_gemv(router_logits, layer->router_gate, …);
      int32_t exp_ids[32];            /* :591 */
      float  exp_weights[32];         /* :592 */
      myds_metal_moe_route(exp_ids, exp_weights, …);   /* :595 算了 */
      if (layer->shared_gate && layer->shared_up && layer->shared_down) {
          … 只跑 shared expert …      /* :598-605 */
      }
      /* exp_ids / exp_weights 之后再未被使用 —— 被编译器优化掉也不会有警告 */
  }
  ```
- **附加**：路由专家权重（`ffn_gate_exps/ffn_up_exps/ffn_down_exps`）**根本没装载**（`grep exps src/*.c` = 空）；`exp_ids[32]` 硬上限 32，`m->n_activated_experts` 未做越界校验。
- **修法**：见 §5 T0-4。
- **验收**：MoE 模型的 logits 与上游 `ds4` 同 prompt 同 seed **逐位一致**；或**显式拒绝** MoE 模型（不许静默）。

### P0-3 真实 V4.1 张量名零匹配 🔴

- **现象**：指向真模型时注意力权重全 NULL、MLA 全缺，**引擎仍「成功加载」并输出垃圾**。
- **证据**：
  - myds 只认（`src/myds_engine.c:205-260`）：`blk.N.{attn_norm,attn_q,attn_k,attn_v,attn_output,ffn_norm,ffn_gate,ffn_up,ffn_down,ffn_gate_inp,ffn_gate_shexp,ffn_up_shexp,ffn_down_shexp}`
  - V4.1 真实名字（上游 `ds4.c` 名字表 + 本机 `DeepSeek-V4.1-Flash-DSpark-support-v2.gguf` 实测 dump）：

    ```
    attn_q_a / attn_q_a_norm / attn_q_b        ← MLA 低秩 Q
    attn_kv / attn_kv_a_norm / attn_kv_a_mqa    ← 融合潜 KV
    attn_k_b / attn_v_b / attn_output_a / attn_output_b
    attn_sinks                                   ← attention sinks
    hc_attn_{base,fn,scale,down} / hc_ffn_*      ← hyper-connection
    exp_probs_b.bias / ffn_gate_inp_shexp
    ffn_{gate,up,down}_exps / ffn_gate_tid2eid
    ```
  - **实测 dump**：myds 要找的 25 个名字在该文件里 **25/25 MISSING**。
- **注意（尚需一次实测才能定稿）**：上游 `ds4` 的 loader **两套名字都接受**（`attn_q` 与 `attn_q_a` 并存），说明社区量化版可能存在两种命名。**部署前必须实测目标 GGUF**（§8-1），这是一条**上线硬门禁**。
- **修法**：见 §5 T0-3。
- **验收**：装载期打印"已装载张量 / 缺失张量"，缺失必需项 ⇒ **退出码非 0**。

### P0-4 不支持的量化类型静默 no-op 🔴

- **现象**：Q4_K / Q6_K / MXFP4 权重被**静默跳过**，输出垃圾。
- **证据**（本机实测）：

  ```
  GEMV GGML_TYPE_Q4_K  -> rc=0, y[0] 未被写入
  GEMV GGML_TYPE_Q6_K  -> rc=0
  GEMV GGML_TYPE_MXFP4 -> rc=0
  ```
  代码：`src/myds_metal.m:465-534`（只有 Q8_0 / F32 / F16 三个分支，其余落到 `:534 return 0`）。
- **背景**：舰队主力权重是 **Q2_K / MXFP4 / Q4_K**。README 只宣称支持 Q8_0/Q4_K/F16/F32 —— **既不支持也不报错**（且 Q4_K 分支根本不存在）。
- **修法**：见 §5 T0-1（fail-fast）+ T2-5（真正支持）。
- **验收**：未知类型 ⇒ 装载期报错退出；已支持类型 ⇒ 与 CPU 参考逐位对齐。

### P1-1 RoPE 只旋转第一个 KV head 🟠

- **证据**：`src/myds_metal.m:356`

  ```c
  if (k && h == 0) { /* 注释：decoupled 只有 1 个 head */ … }
  ```
  `ctx->k` 是 `n_kv_heads*head_dim`（`src/myds_engine.c:333`）。除第 0 头外**其余 KV 头的 K 完全未旋转** ⇒ GQA 下位置信息错乱（除非 `n_kv_heads==1`）。
- **另有 3 处语义问题**：
  1. 调用方把 `rope_dim = head_dim` 传入（`src/myds_engine.c:513`），而 V4.1 只旋转**尾部 64 维**（`metal/dsv4_rope.metal` 的 `n_dims` / `rope.tail` 才是权威）。
  2. `freq_scale` 被当作乘在 `pos` 上的线性缩放，ggml 语义是 `1/scale_linear`。
  3. 每头每对都重算 `powf/cosf/sinf`，且没用 `metal/dsv4_rope.metal` 的原地版（`kernel_dsv4_rope_tail_f32_inplace_pair`）。
- **修法**：见 §5 T2-1。
- **验收**：RoPE 后向量模长守恒 + 与参考实现逐位一致（`tests/test_metal.c` 已有模长守恒测试，但**只覆盖了 h==0**，需补多头用例）。

### P1-2 派发参数「侥幸正确」🟠

- **证据**：`src/myds_metal.m:283-300` 手搓 `ds4_metal_args_norm`：`memset 0` 后**只填** `ne00/ne00_t/nb1/eps`，`nbf1[0] / nef1[1] / nbf1[1]` 等全 0，且只派发**1 个 threadgroup**（`:301`）。
- **实测（子代理）**：
  - 单行场景 **dim=2048/128/64 全部 MATCH**（最大误差 3.6e-7 / 0 / 0）⇒ 现在能跑。
  - **把 grid 改成 2 个 threadgroup，第 1 行输出会等于 row0 的 rmsnorm**（对 row0 误差 0、对正确 row1 误差 2.07）⇒ **静默行混叠**。
- **风险**：等哪天要做批预填充，错误会以「结果不对但日志全绿」的形式出现。
- **同类**：swiglu（`:388-420`）、`kernel_add2_f32`（`:164`）、`kernel_soft_max`（`:170`，**函数名不存在**）的 host 参数实测 MATCH，但都是**照抄**而非按契约填写。契约在 `metal/norm.metal:1-18` 写得清清楚楚，host 侧没有对应常量/注释。
- **修法**：见 §5 T1-2。

### P1-3 采样与舰队配置不一致 🟠

- **证据**：`src/myds_engine.c:420-470`
- **问题**：
  1. 无 `top_k` / `min_p` / repetition / presence penalty。舰队 `fleet.conf:102` 配了 **TOP_K=40 / MIN_P=0 / REP_PEN=1.0**（其余服务由 ds4 执行 top_k）⇒ **同一 alias 家族在 myds 上行为不同**。
  2. `top_p` 实现是「`r = rand()*top_p` 后在**全量分布**上找累积」（未截断后重归一化），与 OpenAI / ds4 语义有偏差。
  3. 无 per-request RNG（全局 `srand`/`rand()`，且 `src/myds_server.c` 内是**线程不安全**调用）。
  4. 每个 token `malloc(vocab_size*4)`（32000 词表 ≈ 128 KB/token）再 `free` —— 纯浪费。
- **修法**：见 §5 T3-4。

### P1-4 服务端健壮性 🟠

见 §2.6。要点：`Content-Length` 永不生效、无 recv 超时（slowloris 锁死唯一线程）、`send` 返回值全不查、SSE 半 UTF-8、usage/finish_reason 造假、10 处 snprintf 累积无溢出保护。
**修法**：见 §5 T3-3。

### P2 其它 🟡

- `include/myds.h:9-27` 架构判定用 `strncmp(arch,"deepseek4",9)==0` ⇒ `deepseek49`、`deepseek4whatever` 也会放行；`"deepseek4-dspark"` 放行但其权重结构需要 MTP/hc 支持。
- `src/myds_engine.c:642` `int32_t prompt_tokens[4096]`（16 KB 栈）**硬编码上限**，完全忽略 `-c/--ctx-size`（舰队给它 131072）。
- `src/myds_engine.c:535` `ctx->attn_scores` 只按 `max_seq_len` 分配，attention 却在 CPU 逐头复用它 —— 接 GPU 后这个 scratch 设计要重做。
- `src/main.c` 用 `clock()`（CPU 时间）算 tokens/s，GPU 引擎上指标会失真；stdout 全缓冲（实测：`Initialized successfully` 在进程退出时才刷出，混在错误之后）。
- **每 token 每层 `myds_metal_buffer_alloc/free` 两次**（`src/myds_engine.c:571`、`:589`）—— 实测 56 次 alloc+free = **0.220 ms/token**，纯开销。
- `Makefile clean` 会 `rm -rf bin`（连带 9 个已构建二进制）。

---

## 4. 性能评审

### 4.1 实测数据（macmini / Apple M4 / macOS 27.0）

| 项 | 结果 | 说明 |
|---|---|---|
| 运行期 shader 编译 | **1.907 s** | `myds_metal_init("metal")`；每个进程启动一次 |
| GEMV Q8_0 4096×4096 | **4.12 ms/call → 4.33 GB/s** | 生产路径（CPU 标量） |
| GEMV F16 4096×4096 | **8.27 ms/call → 4.06 GB/s** | 同上 |
| GEMV Q4_K / Q6_K / MXFP4 | **rc=0，不写输出** | 静默 no-op |
| 56× MTLBuffer(16 KB) alloc+free | **0.220 ms**（3.9 µs/次） | 每 token 的层内 churn |
| CPU RoPE(128 头, dim 128) | **0.037 ms/token** | 不是热点（但语义错，见 P1-1） |
| **CPU attention（1 层, 128 头, head_dim 128, ctx 2048）** | **7.3 ms/token/层** | ×28 层 ≈ **204 ms/token ⇒ 上限 ~5 t/s**，随 ctx **线性**恶化 |
| 同机 8 线程流式读（参照） | **77.5 GB/s** | 说明 4.33 GB/s 距内存带宽有 **~18×** 空间 |

### 4.2 归因与量级

每 token 成本 ≈ `权重流量 / 有效带宽` + `attention 线性项`：

- 权重流量 = 单 token 活跃权重字节（MoE ≈ 每层激活专家 + 共享专家 + 注意力）。以 **4.33 GB/s** 计：
  - 活跃 20 GB ⇒ **~4.6 s/token ≈ 0.22 t/s**
  - 活跃 40 GB ⇒ **0.11 t/s**
- **参照**：同机房 ds4 在 macstudio 上跑 V4.1-Q2 实测 **25.34 t/s**、MXFP4 **35.43 t/s**。
- ⇒ **当前实现与「可用」之间差 2 个数量级**。三处主因：
  1. **GEMV 在 CPU 单线程**（4.33 vs 可行 50~100 GB/s ⇒ **10~20×**）
  2. **attention 在 CPU 且 fp32 稠密 KV、无 MLA**（204 ms/token 上限 ⇒ 架构级）
  3. **无批预填充**（TTFT 随 prompt 线性，且每 token 还多做一次全采样）

### 4.3 内核侧可用资产（子代理发现，选摘）

库里有 **351 个 kernel**，其中大量已编译可用的加速核处于闲置状态：

- `dense.metal`：`kernel_mul_mv_q8_0_f32`(:187)、`..._pair`(:351)、`dsv4_shared_gate_up_swiglu_q8_0`(:599)、`router_shared_gate_up`(:937)、`mul_mm_*`（预填充，:2929-2933）、`mul_mv_f32_f32`/`_f16_f32`(:1244-1245)
- `flash_attn.metal`：`kernel_flash_attn_ext_vec_f16_dk512_dv512`(:1391) + `_reduce`(:1442) —— **DS4 生产解码主核**
- `dsv4_rope.metal`：`kernel_dsv4_rope_tail_f32`(:102)、`_inplace_pair`(:208)、`dsv4_kv_rope_fp8_store_f32`(:508)
- `moe.metal`：`kernel_mul_mv_id_{q8_0,q2_K,q4_K,q8_K,iq2_xxs,mxfp4}_f32`(:3372-3377)

**两个必须知道的裁剪事实**：

1. **`DS4_METAL_HAS_TENSOR` 全仓无定义** ⇒ `moe.metal:9196-9911`（18 个 `host_name`）、`dsv4_misc.metal:6572+`、`dense.metal:2376`、`dsv41.metal:218` **根本没编译**——不是「没接线」，是「不存在」。
2. **变体爆炸**：`moe.metal` 里同一段 ~90 行 K 循环被复制 **约 18 份**（q4_K 的 8 份 sum6 逐字相同：:7152/7275/7390/7501/7618/7750/7886/8033）。这是长期维护债，**不是当前瓶颈**——不要在这一阶段去重。

---

## 5. 优化任务清单（可执行）

> 每项含 **改什么 / 验收门禁**。请**按阶段顺序**做：阶段 0 是阻断项，阶段 1 是最大单点收益，顺序反了会白干。

### 阶段 0 — 阻断项（先做，1~2 天量级）

- [ ] **T0-1 装载期 fail-fast**
  - 改：`src/myds_metal.m` 的 `myds_metal_gemv` 不再对不支持类型 `return 0`（`:534`），改为返回错误码；`src/myds_engine.c` 的调用点检查返回值并**报错退出**。缺必需张量 ⇒ 退出。
  - 门禁：故意用 Q4_K/Q6_K/MXFP4 权重启动 ⇒ **退出码非 0 且日志有明确错误**。

- [ ] **T0-2 权重零拷贝**（收益：内存 2× → 1.0×）
  - 改：`src/myds_engine.c:95-101` 改用 `newBufferWithBytesNoCopy`；`src/myds_gguf.c` 把张量区按**页对齐**映射（现 `alignment` 来自文件头，通常 32；需按 `vm_page_size` 对齐）。
  - 门禁：`ps -o rss= -p <pid>` ≈ 模型大小 ±5%；输出与改前**逐位一致**。

- [ ] **T0-3 目标 GGUF 张量名门禁**（**上线硬门禁**）
  - 改：装载后打印「已装载 / 缺失」张量清单；必需张量缺失 ⇒ 退出。
  - 前置：先跑 §8-1 确认真实张量名。**名字不匹配 ⇒ 先做阶段 2 的 MLA 支持，别急着做阶段 1**（否则加速的是错的路径）。
  - 门禁：文档化目标模型的张量名清单，随代码入库。

- [ ] **T0-4 MoE 修正（或显式拒绝）**
  - 改：装载 `ffn_{gate,up,down}_exps` + `exp_probs_b`；按 `exp_ids/exp_weights` 真正累加（先 CPU 版本，与 shared expert 输出相加）。若暂不支持 ⇒ **显式拒绝 MoE 模型**。
  - 注意：`exp_ids[32]/exp_weights[32]` 硬上限要改成按 `n_routed_experts` 分配。
  - 门禁：MoE 模型输出与上游 `ds4` 同 prompt 同 seed **逐位一致**。

### 阶段 1 — 把 GPU 用起来（**最大单点收益**，1~2 周）

- [ ] **T1-1 批命令缓冲**（**必须先做**，否则 T1-2 收益被同步吃掉）
  - 改：`src/myds_metal.m` 增加 `begin_token()/commit_token()`：整个 token 的算子编码进**一个** command buffer，**单次** `waitUntilCompleted`；把 `:304-305`、`:415-416` 的 per-op 同步去掉。
  - 门禁：每 token 的 `commit` 次数从 ~250 降到 1（加计数器打印）；输出逐位一致。

- [ ] **T1-2 移植 ds4 host 调度层**
  - 改：照上游 `~/Projects/ds4/ds4_metal.m` 的 pipeline 持有结构 + 参数填充 + grid 公式，实现 `ds4_metal_args_mul_mv`（定义在 `metal/dense.metal:6-25`）的正规填充。**不要继续逆向重建。**
  - 接线顺序（收益/风险比）：`kernel_mul_mv_q8_0_f32` → `kernel_mul_mv_f32_f32` / `_f16_f32`（覆盖当前 F32/F16 路径）→ `kernel_add2_f32` → `kernel_soft_max_f32`（**注意真名带后缀**）→ 把已接线的 swiglu 并入 CB。
  - 注意：函数常量 `FC_mul_mv_nsg`（preamble `:43 #define FC_MUL_MV 600`）必须设；threadgroup 内存 = `N_R0_Q8_0 * 32 * 4` 字节；grid = `(ceil(out_dim/N_R0_Q8_0), 1, 1)`，tptg = `(32*nsg, 1, 1)`。**`N_R0_Q8_0` 改值必须与 grid/shmem 同步改。**
  - 门禁：**每个 kernel 先与 CPU 参考逐提示词位对齐**（≥3 提示词，含一个确定性记忆型），再进主链（见 §7 护栏 1）。
  - 预期：GEMV 从 **4.33 GB/s → 50~90 GB/s**（10~20×）。

### 阶段 2 — 补齐架构能力（与 ds4 对齐，2~6 周）

- [ ] **T2-1 RoPE 修正**：多头 K 全部旋转；`rope_dim` 改为尾部 n_dims（V4.1 是 64）；`freq_scale` 语义改 `1/scale`；改用 `metal/dsv4_rope.metal` 的原地/affine 版。门禁：多头模长守恒 + 逐位对齐。
- [ ] **T2-2 MLA**：装载 `attn_q_a/attn_q_a_norm/attn_q_b/attn_kv/attn_kv_a_norm/attn_kv_a_mqa/attn_output_a/b`，接 `kernel_mul_mv_q8_0_f32_pair` + `dsv4_kv.metal` / `dsv4_hc.metal`。**收益：KV 从 fp32 稠密 → 潜向量（内存降一个数量级）。**
- [ ] **T2-3 FlashAttention 解码核**：用 `kernel_flash_attn_ext_vec_f16_dk512_dv512` + `_reduce` 替换 CPU 三层循环（`src/myds_engine.c:530-578`）。**收益：204 ms/token → 亚毫秒级。**
- [ ] **T2-4 批预填充**：接 `mul_mm_*`（64×32 tile，`dense.metal:2929-2933`）+ 激活量化，TTFT 从 N×decode 降到近似常数项。
- [ ] **T2-5 更多量化类型**：`myds_metal_gemv` 增加 Q4_K / Q2_K / MXFP4 分支（对应 kernel 见 §4.3）；或复用 `kernel_mul_mv_id_*` / `mul_mv_ext_*`。
- [ ] **T2-6 attention sinks / hyper-connection**：`attn_sinks`、`hc_*` 的 host 支持（否则数值分布与官方不一致）。

### 阶段 3 — 系统与服务（可与 1/2 并行）

- [ ] **T3-1 KV 量化**：fp32 → F16/FP8（参考 `metal/dsv4_rope.metal:508`）；CTX=131072 的可行性才谈得上。
- [ ] **T3-2 prefix cache / 多轮上下文**：`ctx->pos` 不再复位（`src/myds_engine.c:646`），按会话维护 KV（上游有 `ds4_kvstore.c` / `ds4_prompt_prefix.c` 可移植）。
- [ ] **T3-3 服务端加固**：`Content-Length` 改为标准头部解析（别用 JSON 提取器，`src/myds_server.c:505`）；`SO_RCVTIMEO`；请求体上限；`send` 全写循环（14 处）；跨片段 UTF-8 缓冲；真实 `usage` / `finish_reason`；统一 append 助手消掉 10 处 snprintf 累积。
- [ ] **T3-4 采样对齐舰队**：支持 `top_k` / `min_p` / `rep_pen`；per-request RNG；采样缓冲区复用（别每 token `malloc`）。
- [ ] **T3-5 构建**：离线 `.metallib` 嵌入（消掉 1.9 s 运行期编译 + 绝对路径 cwd 依赖）；`-ffast-math` / `-mcpu`；preamble 常量改为 `preprocessorMacros` 或直接回灌 `.metal`（实现单源真相）。
- [ ] **T3-6 测试重构**：让 `test_engine` / `test_server` / `test_server_api` **链接生产对象**（把 `sample_next_token`、JSON 助手提为可测函数），删除测试内的复制实现。补一条端到端金标准用例（真 GGUF + 固定 prompt + 固定 seed + **逐 token 哈希**）。

---

## 6. 设计边界（非目标 —— 请勿越过）

myds 的定位是「**极简**、专注 DeepSeek V4/V4.1」。因此：

1. **不要把 `ds4` 整个搬进来**（`ds4.c` 3.9 MB / `ds4_metal.m` 50881 行）。目标是用**最少代码把 GPU 用起来**，而不是复刻上游。
2. **不要做通用模型抽象**（llama.cpp 式的 `ggml` 通用图）。只服务 V4/V4.1。
3. **不要为了「支持更多模型」放松架构校验**（`include/myds.h:9-27` 的拒绝逻辑是对的，只是前缀匹配要收紧）。
4. **不要动 `.metal` 文件**（与 `ds4` 逐字节相同是唯一的一致性保障）；确需改动，必须同步回 `ds4` 并逐文件比对 md5。
5. **不要引入第三方依赖**（README 承诺零依赖，仅 Foundation + Metal）。
6. **不要保留「未接线的加速核」这种状态**：要么接线、要么删掉、要么在注释里显式标注「未接线」——现在是**头文件声明与实现不符**，最坏的一种。

---

## 7. 护栏（吸取上游两次静默错误的教训）

1. **任何改数值路径的改动（kernel 接线、量化、fused 算子）⇒ ≥3 提示词与金标准逐位一致，含一个确定性记忆型 prompt。**
   **「日志无 error/failed」是必要条件，不是充分条件——静默错误恰恰是零日志的。**
2. **跨档位门禁**：`N_R0_Q8_0 ∈ {2,4,8}` 必须三者结果一致（上游 nr0>2 需要「单一真源」修复；myds preamble 是 2，动它要连带改 grid/shmem）。
3. **端到端验证要落到进程**：`ps eww -p <pid>` 看 env、`ps -o command=` 看 argv。别只信「我改了脚本」。
4. **改配置用精确相等匹配**（`fleet.conf` 里 `-iq4` / `-iq4xs` 同前缀曾误命中），改完 **diff 全文**，不能只看目标行。
5. **在生产推理机上跑任何脚本前先问**：「这个脚本最坏会申请多少内存？」答不上来就别跑。
   **绝不在推理机上整文件读大模型**（GGUF 动辄 100 GB+；读 metadata 用**有界前缀**，不要无 size 参数的 `read()`）。
6. **跨机操作要么拆成两条命令，要么在脚本里显式标注执行主机**；关键落盘要**读回确认**（`wc -l` / `grep -c` / `md5`），别信脚本自己打印的 ✓。

---

## 8. 附录：复现命令（只读）

```bash
# 1) 目标 GGUF 到底有哪些张量名（决定 myds 能否装载）—— 上线硬门禁
cd ~/Projects/myds
# 用 include/myds_gguf.h 写一个 ~30 行的 dumper（只读 mmap 元数据，不碰张量数据）：
#   遍历 gf->tensors[i].name 打印；对 myds 需要的名字逐个 gguf_find_tensor 并打印 FOUND/MISSING
# 期望：blk.0.{attn_q,attn_k,attn_v,attn_output,...} 与 blk.0.{attn_q_a,attn_q_b,attn_kv,attn_sinks,hc_*} 二选一出现
# 注：本机 DeepSeek-V4.1-Flash-DSpark-support-v2.gguf 上，myds 要求的 25 个名字 25/25 MISSING。

# 2) 权重是否被拷贝（P0-1 是否已修）
ps -o rss=,vsz= -p <pid>          # rss ≈ 模型大小 2 倍 ⇒ 未修

# 3) 运行期 kernel 符号 & 名字可寻址性
#    实测：库内 351 个 kernel 符号；kernel_soft_max 与 kernel_mul_mv_q4_K_f32 均【不存在】

# 4) 本机 shader 编译耗时 + GEMV 有效带宽（自建 /tmp 侧 harness）
#    编译 src/myds_metal.m → 调 myds_metal_init("metal") 计时；再调 myds_metal_gemv 计时
#    期望（M4 实测基线）：init ≈ 1.9 s；GEMV Q8_0 4096x4096 ≈ 4.1 ms（4.33 GB/s）

# 5) 20 个 .metal 与上游 ds4 是否同源（应全部 SAME）
for f in metal/*.metal; do
  printf "%s " "$(basename "$f")"
  [ "$(md5 -q "$f")" = "$(md5 -q ~/Projects/ds4/metal/$(basename "$f"))" ] && echo SAME || echo DIFF
done
```

---

## 9. 舰队侧待办（不在本仓库，但相关）

- `~/Projects/deploy/fleet.conf:102` 已启用 8181（`local-myds-deepseek-v4.1-flash-q2`，macstudio / type 15 / CTX=131072 / MAX_OUT=4096 / TEMP=0.7）。
  **该服务尚无运行日志 ⇒ 还没真跑过。** 在 §3.1（零拷贝）修好之前**不要点火**。
- `fleet.conf` 的 `TOP_K=40 / MIN_P=0 / REP_PEN=1.0` 对 myds **无效**（myds 不支持这些参数），上线前需在文档/配置里对齐预期（见 T3-4）。
- `~/Projects/deploy/start_fleet.sh` 启动时不 `cd`，依赖 myds 的 cwd 兜底逻辑（`src/myds_metal.m:110-118`）——T3-5 修掉绝对路径后需复核。

---

*报告结束。所有结论可复现；**唯一待定稿项**是「目标 GGUF 的真实张量名」（§8-1），该项需在目标模型上实测一次。*
