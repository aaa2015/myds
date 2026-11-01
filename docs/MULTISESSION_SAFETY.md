# 多会话安全审计（2026-11-01）

来源：对 `dsh-superagent-prototype` 多会话工作负载的适配评估（见 antigravity
`implementation_plan.md`）逐条与本仓库源码核对，并**在生产进程上实测**。

本文只记录**核实过**的结论。未核实的推测不写进来。

---

## 一、结论速览

| # | 问题 | 核实结果 | 状态 |
|---|---|---|---|
| 1 | 多锚点切换时压缩 KV 串台 | ✅ **生产复现，静默错答** | **已修复并实测通过**（§三） |
| 2 | 锚点槽位释放泄漏 | ✅ 确认，数组 60 槽只释放 44 | 已修 |
| 3 | 磁盘锚点恢复槽位硬编码 | ✅ 确认 | 已修 |
| 3b | 同长度锚点按位置恢复会挑错槽 | ✅ 代码确认 | 已修（按槽恢复） |
| 4 | `previous_response_id` 只认最后一条 | ✅ 确认 | 待做 |
| 5 | 锚点要求全长匹配 | ✅ 确认 | 待做 |
| 6 | 队列严格 FIFO，无优先级/亲和 | ✅ 确认 | 待做 |
| 7 | 断连只在每 token 回调检测 | ✅ 确认 | 待做 |
| 8 | 模型先加载、后绑端口 | ✅ 确认 | 已修（三道门提前） |
| 9 | 未知 CLI 参数 / 未知 `MYDS_*` 静默忽略 | ✅ 确认 | 已修 |
| 10 | 残缺 DSML 块混进正文 | ✅ 确认（代码注释「绝不吞内容」） | 待做 |

---

## 二、P0-1：锚点跨会话串台（本次最严重的缺陷）

### 复现

两个系统提示词（ALPHA / BRAVO，分别 2742 / 2482 token），交替请求，问
「ALPHA record 0042 的 note 值」：

| 步骤 | 输出 | |
|---|---|---|
| A 首次 | `branch-ALPHA-42` | 正确 |
| B 首次 | `branch-BRAVO-42` | 正确 |
| **A 再次** | **`branch-BRAVO-42`** | **A 读到了 B 的状态** |
| A 冷启动参照 | `branch-ALPHA-42` | 正确 |

日志显示确实走了锚点：

```
captured anchor #0 at pos 2743 (slot 44)
captured anchor #1 at pos 2482 (slot 45)
prompt 2751 tok -> resume from ANCHOR #0 at 2742 (reuse 2742 tok, still to prefill 8)
```

**没有日志、没有报错，`finish_reason: "stop"`。** 就是错答案。

### 根因

`myds_v41_snapshot_store_slot` / `snapshot_restore` 保存与恢复的是：

* 40 层 `window`（末尾 128 token 的滑动窗口）
* `previous_kv[4]` / `previous_score[4]`（压缩器跨步状态）
* `engram->history`

而 `compressed[4]` / `index_cache[4]` 是**按位置索引的常驻缓存**，decode 时
确实在读（`myds_v41.c:4370 / 4419 / 4429 / 4752`）。两个会话交替 prefill 时，
`[L, anchor_pos)` 这段被后者的内容覆盖；恢复锚点只还了末尾 128 token，
前面那段是别人的。

### 为什么只有锚点出问题，环形快照没事

环形快照（位置 1024…11264）只在「新 prompt 与**上一条**请求的 `pcache_tokens`
有公共前缀」时才续跑。多会话交错时公共前缀通常只有 2 个 token，
`myds_v41_snapshot_best(2)` 返回 0 ⇒ 不续跑 ⇒ 安全。

锚点则跨请求持久存在，专门用于「不同请求、相同系统提示词」——正好是多会话场景。

### 临时处置（已上线）

生产 `myds.conf` 加 `MYDS_ANCHOR_SNAPSHOT=0`。代价是每次请求重新 prefill
系统提示词；收益是不会答错。

复测：`A(after B)` == `A(冷启动)`，sha 完全相同；B 正确回答「没有 ALPHA 记录」。

---

## 三、锚点隔离的正解（已实现）

### 结论：不需要额外内存，只要一条判据

`compressed[4]` / `index_cache[4]` 是**按位置索引的常驻数组**，不是会话隔离的。
所以「恢复锚点」能不能成立，等价于一个问题：

> 常驻数组在 `[0, anchor_len)` 上，是不是**这串 token** 产生的？

这个问题的答案引擎早就算好了：`L` = 本次 prompt 与 `pcache_tokens` 的公共前缀长度
（`pcache_tokens` 正是「产生当前常驻状态的那串 token」）。于是判据就是：

```c
if (best_len && best_len <= L) { /* 可用 */ }
```

**环形快照一直隐含这个约束**（`myds_v41_snapshot_best(L)` 只挑 ≤ L 的点），
只有锚点这条路漏了。加上之后两条路径的统一不变量是：

> **永不从超过公共前缀的位置续跑。**

实测日志（`L=13`，锚点 2743）：

```
prefix cache: ANCHOR #0 at 2743 NOT usable - the resident compressed KV only
              matches the first 13 tokens (another session overwrote it).
              Full prefill instead of reusing a foreign state.
```

### 附带修掉的第二个洞：同长度锚点会挑错槽

`myds_v41_snapshot_restore(g, p)` 是「取第一个 `snap_pos == p` 的槽」。
两套系统提示词**恰好同长**（实测 2742/2483 这种很常见）时，两个锚点槽的
`snap_pos` 相同 ⇒ 可能装上另一套的窗口环。新增
`myds_v41_snapshot_restore_slot(g, slot, p)`，engine 命中锚点时按 `best_i` 指定的
槽号恢复。

### 关于「压缩段快照池」

最初设想的方案（按锚点保存 `[L, anchor_len)` 的压缩 KV 段）**不做了**：
上面那条判据已经消除了错误，而段池只是在「前缀确实不同」时额外买速度，
那正是应该走全量 prefill 的场景。少一个几百 MiB 的池、少一条需要独立验证的
快路径，是更好的取舍。

### 实测（macstudio 8187，锚点恢复已重新打开）

系统提示词 14,713 token（超过环形容量 11264，只有这样锚点才会真正参与决策）：

| 步骤 | 输出 | 判定 |
|---|---|---|
| A 首次 | `branch-ALPHA-42` | ✓ |
| B 首次 | `branch-BRAVO-42` | ✓ |
| **A 再次**（B 已覆盖常驻压缩 KV） | **`branch-ALPHA-42`** | ✓ 修复前是 `branch-BRAVO-42` |
| A 第二轮（同系统提示词、不同问题） | `branch-ALPHA-7` | ✓ 内容正确 |

生产配置里 `MYDS_ANCHOR_SNAPSHOT=0` 已**撤除**，锚点恢复正常使用。

### 顺带发现：锚点大多被环形快照抢先

日志显示 `resume from snapshot at 14336` 会在锚点之前命中（`resume_from != 0`
时锚点分支不执行）。也就是说短/中等系统提示词下环形快照已经够用，锚点的
净收益比预期小。这不影响正确性，但如果将来要评估「锚点值不值得留」，
要看的是**环形容量覆盖不到的深度**（>11264）那一段。

## 四、其余待做项（按性价比排序）

### ~~高：`previous_response_id` 多会话热恢复~~ —— 已完成（见 §五）

`resp_id_is_live` 只承认 ring 里**最新一条**（`myds_server.c:441`），且
`resp_recover` 只在 `pcache_len == 0` 时触发，否则直接 400：

> `previous_response_id 'x' is not the most recent response (this server keeps one live state)`

多会话交错时这个 400 是必然的，不是偶发。

修法（两步，可分开上）：
1. `RESP_ID_SLOTS` 4 → 16，每槽存 `int32_t *tokens`（省掉文本重拼再分词，
   避免 BPE 边界合并差异）；
2. 命中非最新槽 ⇒ 走内存级前缀恢复，而不是 400。

### ~~中：断连剔除~~ —— 已完成（见 §七）

`client_disconnected()` 存在，但只在**每个 token 回调**里调用
（`myds_server.c:3350 / 3369`）。Prefill 阶段（数十秒到数百秒）没有任何调用点，
也没有 watchdog 线程 ⇒ 客户端早断开，GPU 仍跑完全程 prefill。

两处补：
* 出队拿到推理槽之后、进 GPU 之前，先探一次；
* prefill 每个 chunk 之间探一次（需要一个注册式的轻量探针，不能直接碰 socket）。

### 中：队列优先级 / 前缀亲和

严格 FIFO（`g_queue_head`），且排队前未解析任何元数据。后果：长 worker 的
prefill 会让 orchestrator 的路由请求排队超时（503）。

低风险的最小版本：**同优先级下，`previous_response_id` 等于刚完成请求的
排在前面**。这一条不需要新 header、不引入饥饿风险，且直接命中零开销续跑。

### ~~低：残缺 DSML 隔离~~ —— 已完成（见 §八）

`myds_server.c:4194-4202` 的注释写着「把没发出的正文补出去，绝不吞内容」——
在工具块解析失败时会把半截 DSML XML 当正文发给客户端。判据应是
`stop_reason == MYDS_STOP_LENGTH` 且已见到 `<|DSML|function_calls>` 起始标签。

---

## 五、本次已落地的改动

| 改动 | 验证 |
|---|---|
| `context_free` 循环上限 `V41_SNAP_SLOTS`(44) → `V41_SLOT_CAPACITY`(60) | 编译 + 全量测试 |
| 磁盘锚点恢复改为由调用方指定 seat 槽 | 编译 + 全量测试 |
| 未知 CLI 参数 ⇒ exit 2 | `tests/test_startup_gates.py` |
| 未知 `MYDS_*` 键 ⇒ exit 2（`MYDS_STRICT_CONFIG=0` 降级） | 同上 |
| 端口占用 ⇒ exit 3；模型加载前取每端口单实例锁 | 同上 + 生产实测 |
| 锚点加 `best_len <= L` 判据 | 14.7K 系统提示词交替请求：修复前答 B、修复后答 A |
| 锚点按槽恢复（`..._restore_slot`） | 同上 |
| 生产撤除 `MYDS_ANCHOR_SNAPSHOT=0` | 交替 + 同会话两轮均正确 |
| `previous_response_id` 多活（槽数可配，默认 16） | 旧二进制 400，新二进制 200 且答对 |

生产部署后：7 个基线 prompt 的 sha 与改动前**逐一相同**，说明正常路径零行为变化。

---

## 六、`previous_response_id` 多活（2026-11-01）

### 原行为与后果

`resp_id_is_live()` 只承认 ring 里**最新一条**，其他一律 400：

> `previous_response_id 'x' is not the most recent response (this server keeps one live state)`

多会话交错时这是**必然**失败，不是偶发：

```
A turn1 (id=resp_…_1)   → 200
B turn1 (id=resp_…_2)   → 200       ← 此刻 A 的 id 已经不是最新
A turn2 (prev=…_1)      → 400       ← 三秒前拿到的 id 就废了
```

### 改法

保留快路径，补一条恢复路径：

| 情况 | 走哪条 | 代价 |
|---|---|---|
| id 就是最新 且 pcache 还活着 | 零拷贝续跑（原行为，未改） | 0 |
| id 仍在表里但不是最新（或 pcache 已冷） | 累积文本 + 本次 suffix 重建完整输入，全量 prefill | 一次 prefill |
| id 从未见过 | 400 `unknown_previous_response_id` | — |

重建路径本来就有（原本只用于「服务重启后 pcache 为空」），这次只是把它的
触发条件从「pcache 必须为空」放宽到「id 仍在表里」。正确性由既有的
`tests/test_http_bytes.py` 与下面的实测覆盖。

**槽数**：4 → 可配（`MYDS_RESP_SLOTS`，默认 16，上限 64）。4 个槽在三个会话
交错一次之后就绕回来了。每个槽只是一份累积文本，16 个槽是几十 KB 的事。

**顺带修掉两个会在多活之后才暴露的错**：

1. 工具集校验原来拿「最近一次」的指纹比 —— 多活后 `prev_id` 很可能不是最新那条，
   会把 A 的续跑按 B 的工具集判定。改为按**祖先 id** 查指纹（`resp_id_tools_sig`）。
2. 登记新响应时同理，沿用的应是祖先的工具指纹。

还有一个竞态：`resp_id_saved_text()` 解锁后返回内部指针，另一个线程登记新响应时
会 free 它。新增 `resp_id_saved_text_copy()` 在锁内拷贝，新路径一律用它。

### 实测（macstudio 8187，同一个测试跑新旧两个二进制）

| 检查 | 旧二进制 | 新二进制 |
|---|---|---|
| A turn1 接受 | 200 | 200 |
| B turn1 接受 | 200 | 200 |
| **A turn2 不再 400** | **400 ✗** | **200 ✓** |
| **A turn2 记得自己的上下文** | 空 ✗ | `Teal` ✓ |
| **A turn2 没串到 B 的上下文** | （无输出） | `crimson` 未出现 ✓ |
| 未知 id 仍拒绝 | 400 ✓ | 400 ✓ |

复现脚本：`tests/previous_response_live.py`（需模型，不进 `make test`）。

---

## 七、prefill 期间断连剔除（2026-11-01）

### 原行为

`client_disconnected()` 存在，但只在**每个 token 的解码回调**里被调用。
prefill 阶段一个 token 都不会回调，于是：

> 客户端 abort / 超时重连 / 用户取消 ⇒ 服务端仍然把整段 prompt 算完。

实测（macstudio 8187，60K prompt，3 秒后断开）：

| | 旧二进制 | 新二进制 |
|---|---|---|
| 该请求的 wall | **>60 s（仍在跑）** | **3.0 s** |
| 后续请求 | **503 queue_timeout（等了 60 s）** | **0.7 s 内返回 200** |
| 日志 | 无任何取消痕迹 | `prefill: CANCELLED (planned 59990 tok)` |

一个没人等的请求足以把整台机器堵住一分钟以上。

### 实现：注册式探针

引擎不知道 socket 的存在，所以走回调：

```c
typedef bool (*myds_liveness_probe_t)(void *user_data);
void myds_context_set_liveness_probe(myds_context_t *ctx,
                                     myds_liveness_probe_t probe, void *user_data);
```

服务端注册的实现就是 `recv(MSG_PEEK | MSG_DONTWAIT)`，非阻塞、O(1)。
探针返回 false **等价于外部取消**：只置 `cancel_flag`，然后走既有的
`goto cancelled` 路径 —— 不新增一条需要独立验证的出口。

探针挂在四处（都是长循环）：

| 位置 | 粒度 | 理由 |
|---|---|---|
| 出队后、进 GPU 前 | 一次 | 排队期间客户端可能已经走了，此时应零成本丢弃 |
| engram 准备 | 每 64 token | prefill 里最先跑的一段，大 prompt 下本身就能到秒级 |
| embedding 拷贝 | 每 1024 token | 逐 token 调用 `recv` 会变成 syscall 热点 |
| chunk 主循环 | 每 chunk（默认 2048） | 真正干活的地方 |

### 两个踩到的坑（都只有真机才暴露）

1. **探针最初只挂在 macroblock 边界，等于从不触发。**
   60K prompt 自适应出的 `macroblock=60416` —— 整段就是一块，边界只出现一次。
   实测客户端断开后仍然把 60K 全算完。必须降到 chunk 粒度。
2. **取消时日志把「计划量」当「已完成量」打印。**
   `prefill: 19995 tok in 5.3s = 3762 t/s` 看着像跑完了，实际在第一个 chunk 就中止。
   现改为 `prefill: CANCELLED (planned N tok, cached M)` —— 取消时不报速率。

### 验证

- `tests/prefill_disconnect_live.py`：发 60K prompt，3 秒后挂断，检查取消日志、
  检查后续请求不被堵。
- 基线 7 个 prompt 的 sha 与改动前**逐一相同**（探针不改任何计算结果）。
- `make test` 全绿。

---

## 八、DSML 工具块隔离（2026-11-01）

### 原行为

解析失败时「整段当正文」：

```c
if (!myds_parse_completion_tool_calls(...)) {
    tool_content_len = ns.len; tool_calls = NULL; tool_count = 0;   /* 格式不合 ⇒ 整段当正文 */
}
```

被 `max_tokens` 截断时，模型正写到一半：

```
...<|DSML| calls>
<|DSML| invoke name="get_weather">
<|DSML| parameter name="city"
```

整块原样进了 `content`。上层任何解析工具调用的代码都会拿到垃圾。

（流式路径其实没漏 —— `sse_emit_bytes` 见到开标签就 `in_tool_block=true` 停了。
漏的是收尾那一步：把 `emitted..acc_len` 整段补发。）

### 改法：四态分类 + 隔离

```c
typedef enum { MYDS_DSML_NONE, MYDS_DSML_COMPLETE,
               MYDS_DSML_TRUNCATED, MYDS_DSML_MALFORMED } myds_dsml_state_t;
myds_dsml_state_t myds_dsml_classify(const char *text, size_t len, size_t *content_len);
```

| 状态 | 处置 |
|---|---|
| `NONE` | 全是正文 |
| `COMPLETE` | 正常解析成 tool_calls |
| `TRUNCATED` | **隔离**：正文只到开标签之前，块丢弃 + WARN，`finish_reason` 归 `"length"` |
| `MALFORMED` | 同上（归 `"length"` 是因为输出确实不完整，不能谎报 `"stop"`） |

`content_len` 一律只算开标签之前的部分（并削掉尾部空白），所以正文边界不依赖解析是否成功。

### 另一个必须记下来的发现

真机测试一开始「通过」了，但**是假通过**：我的泄漏判据只查 `｜DSML｜`，
而模型实际吐出的是 `<function_calls>`（不带全角竖线）。

日志确认 `has_tools=1` —— 工具定义确实渲染进了提示词，模板里也明确要求
`｜DSML｜calls`。模型仍然用了另一套同构方言。于是解析器认不出来，整块当正文发。

**这是另一件事（方言不匹配），不是截断问题。** 本次**不**去猜着兼容第二种方言：
多认一种写法要连带定义转义、嵌套、歧义规则，得单独设计和验证。
改为**让它在日志里现形**：

> `[WARN] tool-call markup found but this request's dialect is not |DSML|calls; it will be sent as plain content.`

### 验证

- `tests/test_http_bytes.c` 新增 6 个 C 断言（NONE / COMPLETE / TRUNCATED /
  MALFORMED / 空正文 / 空输入），进 `make test`，不需要模型。
- `tests/dsml_truncation_live.py`（需模型）**当前是 FAIL 的，而且这就是正确结果**：
  它如实报出 `<function_calls>` 泄漏，即上面那条方言问题。
- 基线 7 个 prompt 的 sha 不变。

---

## 九、更正：工具调用一直是能用的（2026-11-01 晚）

上一节我写「这台机器上的工具调用目前根本不工作」。**这个判断是错的**，必须更正。

### 错在哪

我只用了一个 `max_tokens=30` 的请求做判断 —— 那次确实被截断了，输出的是半截块。
把它当成「工具调用坏了」是把一次截断读成了系统性故障。

重新测：

| 测试 | 结果 |
|---|---|
| 20 个真实任务（中英文、单/多工具、读文件/跑命令/查天气） | **0 泄漏，全部 `finish=tool_calls`** |
| 「What's the weather in Beijing?」`max_tokens=512` | `tool_calls=1`，参数 `{"city":"Beijing"}` 正确 |
| 重启后第一个请求就是工具调用 | 正常 |

### 真实存在的是什么

有一个**措辞特定的**分支会让模型改用另一套方言，且**100% 可复现**：

```
Prompt A: "Check the weather in Beijing, Tokyo and Paris. Use the tool."
          -> |DSML| calls  -> tool_calls=3  ✓

Prompt B: "Use the get_weather tool to check the weather in San Francisco,
           Beijing and Tokyo. Call the tool for all three."
          -> <function_calls> （不带全角竖线） -> 原文泄漏进 content  ✗
```

两套结构完全同构，只是标签名差一个 `｜DSML｜` 前缀。25 个真实 prompt 里命中 1 个。

### 修法：认第二套方言，但只认「结尾块」

`dsml_dialect_pick()` 选方言，规则：

* 先找 `｜DSML｜calls`，有就用（特殊 token，正文里几乎不可能出现）；
* 否则看 `<function_calls>`，**必须满足**：闭合标签存在，且其后只剩空白。

第二条是安全边界：正文里讲解或引用这种标记时（后面还有正文），**不认**，原样保留。
DSML 方言不要求结尾，保持原行为。

另外 `<function_calls>` 方言的参数常常不写 `string="true|false"`（实测模型只写
`name=`），于是加了形状推断：能当 JSON 标量/容器解析的原样放，否则当字符串加引号。

### 验证

| 检查 | 结果 |
|---|---|
| Prompt B（原来必泄漏） | `finish=tool_calls`，`tool_calls=3`，content 干净 |
| A/B 措辞交替 4 轮 | 8/8 正确 |
| 20 个真实任务 | 0 泄漏 |
| 正文中间出现 `<function_calls>` | 不认，整段保留（C 断言 case 8） |
| 截断的 `<function_calls>` | `TRUNCATED`，正文只到标签前（case 9） |
| 基线 7 个 prompt sha | 逐一不变 |
| `make test` | 全绿 |

### 教训

**一个 `max_tokens=30` 的失败样本，不足以支撑「功能不可用」这种结论。**
要下这种判断，至少得跑一批真实任务、并且把「截断」和「解析不了」分开测。
这次是先下了结论、后来才补的对照组 —— 顺序反了。
