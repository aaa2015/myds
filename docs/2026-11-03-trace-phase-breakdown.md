# /trace 分阶段拆解：口径与实现（2026-11-03）

**背景**：页面上只有「排队 / 解析+渲染 / 进引擎前 / 引擎总 / prefill / decode / 未归因」七段。
对一次 43 秒的请求，这些只回答「哪一段大」，不回答「为什么大」。
本次把 **prefill、decode、未归因** 三条各自拆开，全部常开、零额外同步点。

## 一、口径（先读这一节，否则数字会读错）

| 字段 | 含义 | 怎么来的 |
|---|---|---|
| `墙钟` | 该阶段的真实耗时 | 已有的起止时间戳 |
| `GPU 忙` | GPU 真正在算的时间 | Σ(command buffer 的 `GPUEndTime - GPUStartTime`) |
| `阻塞等待` | CPU 卡在 `waitUntilCompleted` 的墙钟 | 只在那两个本来就阻塞的函数里取两次时间 |
| `宿主` | 提交 kernel / 采样 / 分词 | `墙钟 - 等待` |

**关键：`GPU 忙` 与 `阻塞等待` 会重叠。异步提交时 CPU 与 GPU 并行，
所以 `GPU 忙 + 等待 ≠ 墙钟`。页面分开列，不能相加去凑满。**

这不是估算：GPU 时间戳是 Metal 自己记录的，wait 是真实的阻塞区间。

## 二、每段怎么看

### prefill 内部

- `GPU 忙` 接近墙钟 ⇒ prefill 是算力受限，优化只能动 kernel/量化。
- `GPU 忙` 远小于墙钟 ⇒ 瓶颈在 CPU 提交或 Python 侧准备，改 kernel 没用。

### decode 内部

- `GPU 忙 ≈ 墙钟` ⇒ 纯 GPU 受限。2026-11-03 实测 997.8ms / 1010ms = 98.8%。
- `回调写出` 大 ⇒ 客户端慢或网络拥塞，**不是模型变慢**。这一条最容易误判。

### 未归因分解

`引擎总 - prefill - decode`，拆成：

- `分词`：tokenizer。长 prompt 时值得看。
- `KV 落盘加载`：磁盘快照 `load+apply`。
- `其余`：首字前的 head+采样、快照写入等。

## 三、常开为什么不影响性能

- GPU 忙时：只读 command buffer 完成时**本来就存在**的两个时间戳。
- wait：只在 `waitUntilCompleted` 前后各取一次 `mach_absolute_time`，那里本来就在阻塞。
- decode 每 token 约 2 次系统调用级开销，实测对 40 t/s 无影响。

**没有新增 commit、没有加锁、没有改变批处理方式。**
逐层计时仍需要 `MYDS_PROFILE_PREFILL=1`，因为那个开关会把 chunk 退化成串行，
测出来的数不代表生产口径 —— 所以页面继续显示「未测量」，不画平线冒充。

## 四、实现位置

| 文件 | 内容 |
|---|---|
| `src/myds_metal.m` | `g_ph_gpu_busy_s / g_ph_wait_s / g_ph_cbs`，`myds_metal_phase_take/reset` |
| `include/myds_internal.h` | `ctx->ph_*` 每请求归因字段 |
| `src/myds_engine.c` | 入口清零；prefill 后 take；generate 末尾 take；分词与 kvdsk 计时 |
| `src/myds_http_io.c` | 回调 `emit_s`（SSE 与缓冲两条路） |
| `src/myds_server.c` | `myds_context_reset_phase_detail` + 填入 trace 记录 |
| `src/myds_trace.c` | JSON `detail` 对象 + 页面三张表 |

## 五、坑（踩过的）

1. **异步 flush 的 command buffer 必须在 `wait_inflight_commands` 里记账。**
   只记 `wait_command_buffer` 时，decode 的 GPU 忙时只有 1.5 ms/token，
   而墙钟是 20 ms/token —— 差了 13 倍，差点据此得出「CPU 是瓶颈」的错误结论。
   `flush_batch` 提交的那批只会在 `wait_inflight_commands` 完成。

2. **未测量必须是 null，不能是 0。** `-1` 在 JSON 里输出 `null`，页面显示「—」。

3. **顶部瀑布图不能同时画「引擎总」和它的子项。**
   原来 7 段一起画，一条 1 秒的请求能画出 2 秒的宽度，比例全错。
   现在 `bar` 只用互不包含的段：queue / render / setup / prefill / decode / unattributed。

4. **`engine - prefill - decode` 归因到的「其余」包含首字前的 head+采样**，
   不要把整块当成"未知开销"去找 bug。
