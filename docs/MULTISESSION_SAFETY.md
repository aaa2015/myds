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

### 高：`previous_response_id` 多会话热恢复

`resp_id_is_live` 只承认 ring 里**最新一条**（`myds_server.c:441`），且
`resp_recover` 只在 `pcache_len == 0` 时触发，否则直接 400：

> `previous_response_id 'x' is not the most recent response (this server keeps one live state)`

多会话交错时这个 400 是必然的，不是偶发。

修法（两步，可分开上）：
1. `RESP_ID_SLOTS` 4 → 16，每槽存 `int32_t *tokens`（省掉文本重拼再分词，
   避免 BPE 边界合并差异）；
2. 命中非最新槽 ⇒ 走内存级前缀恢复，而不是 400。

### 中：断连剔除

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

### 低：残缺 DSML 隔离

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

生产部署后：7 个基线 prompt 的 sha 与改动前**逐一相同**，说明正常路径零行为变化。
