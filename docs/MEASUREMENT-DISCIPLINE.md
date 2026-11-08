# myds 测量纪律

> **这份文件回答一个问题：怎么保证测出来的数字不是假的。**
>
> 每一条都来自一次真实翻车，标了账本节号。**不是建议，是判据。**
> 新建测量脚本或写任何 `t/s` 之前，先过一遍 §快速清单。

---

## 0. 快速清单（照做即可）

```sh
# 1. 有守卫：跑任何会动 8187 的东西
tools/with_prod_guard.sh <你的命令>

# 2. 有断言：请求成功了，而且真的做了全量 prefill
#    （不要用 curl -w 的时间当证据）
grep -ac "prefill: 32[0-9][0-9] tok in" server.log   # 必须 == 请求数

# 3. 两臂确实不同：打印两个臂的身份（metallib md5 / conf diff / 启动日志关键行）
shasum -a 256 bin/myds.metallib
diff conf_armA conf_armB

# 4. 0-ULP：只接受实测 sha 逐字节相同
#    空输出 != 相同（先断言非空）
```

**任何一条没做，数字就不能写进账本。**

---

## 1. 配置：`--config` 会无条件覆盖环境变量

**判据**：`src/myds_configfile.c` 的 `setenv(key, val, 1)` —— 配置文件是 `MYDS_*` 的唯一来源。
`env VAR=... ./myds --config x.conf` **一律无效，且不报错**。

⇒ 要 A/B 某个键，**必须改 conf 副本**。

**踩过**：§159.2。两组 sweep（`decode_split ∈ {0..40}`、`NR0 × HEAD_REFERENCE`）
数字全在噪声内，日志一字不变 —— 因为两"臂"其实是同一份配置。
后来改用 conf 副本才发现真实结论相反（关掉分段慢 6.3%，不是"持平"）。

**例外**：只有**不在 conf 里**的键才能用环境变量。`grep '^MYDS_XXX=' myfleet/myds.conf` 先确认。

---

## 2. 冷 prefill 的三个杀手

测「32K 冷 prefill」时，有三样东西会让它**悄悄变成热启动**，而且**都成功返回**。

### 2.1 另一个实例还在加载（OOM）

机器 192 GB，单实例 wired ≈ 150 GB。两个并存时请求失败：

```
myds_metal: command buffer failed: Insufficient Memory
V4.1 prefill failed at layer 31: HC FFN input
prefix cache invalidated: prefill returned -1
```

但 `curl -w %{time_total}` **照样打印一个很小的数**。
**踩过**：§167.6 —— 我看到"6.27 秒 prefill"，实际是失败。

**判据**：时间不是证据。必须断言 HTTP 200 **且**日志里有 `prefill: N tok in`。

### 2.2 `~/.cache/myds/responses.idx`

16 个槽的累积提示词，**跨进程存活**。另一臂发**完全相同的请求**时直接重放：

```
Completed ChatCompletions - prompt: 32220 tok, prefill: none (all 32219 cached)
```

**0.137 秒。**

**判据**：每臂独立的 `MYDS_RESP_STATE` 路径，并 `rm -rf ~/.cache/myds`。

### 2.3 `MYDS_KV_DSK`（最狠）

`myfleet/myds.conf` 里**是开着的**：

```
MYDS_KV_DSK=/Users/fangqingyuan/myds-kv
MYDS_KV_DSK_BUDGET_MB=102400
```

`kvdisk` 把 prompt-end KV 快照**落盘**，**跨进程、跨重启存活**。于是第一个请求就是：

```
prefix cache: DISK load+apply 34.280 ms, selected 32219 tokens
kvdisk: 命中位置 32219 (prompt-end) ⇒ 免掉 32219 个 token 的 prefill
```

而**它是 conf 键** ⇒ 环境变量无效（见 §1）。**只能改 conf 副本，指向每臂独立的空目录。**

**踩过**：§167.6 —— 连续三次跑出 0.137 秒的"prefill"，前两次用 resp cache 解释、
第三次才发现是 kvdisk。

### 2.4 固化断言

```sh
n_real=$(grep -ac "prefill: 322[0-9][0-9] tok in" "server_$tag.log")
if [ "${n_real:-0}" -lt 3 ]; then
    echo "!! 出现缓存重放，本轮数据无效"; return 1
fi
```

**没有这条，P-1 的结论会是「+3.4%」（假），而不是「−7.5%」（真）。**

---

## 3. A/B 前先证明两臂确实不同

两组相同配置当然给出相同数字。**数字接近是结果，不是证据。**

必须落在**可观测的结构差异**上：

| 手段 | 看什么 |
|---|---|
| 产物身份 | `shasum -a 256 bin/myds bin/myds.metallib` 两臂必须不同（且只差预期的那一项） |
| conf 差异 | `diff conf_a conf_b` 输出必须只有预期的一行 |
| 启动日志 | `MYDS_ATTN_WHICH=1` 看 `matrix_encoded=1`；knob 行；`CB/token` |
| 编译期 | kernel 变体名、模板参数 |

**踩过**：§26.10.1 第二次假阳性 —— 我在 macmini 编译、脚本在 macstudio 执行，
两臂 md5 都是 `3d2d66b8`，"0-ULP 通过""kprof −1.2%"全是同一文件跑两遍的噪声。

**踩过**：§161.2 —— `matrix` 编码失败会**静默回退 vec**。不做 `matrix_encoded=1` 的确认，
"两臂持平"就是假的。

---

## 4. 0-ULP 验收：只接受实测 sha

- **只认实测**。论证再漂亮也要跑。
- **空输出 / 缺失输出先断言非空**（§26.10.1 第一次假阳性：两个 500 错误都返回 0 字节，
  `cmp -s` 判定相同）。
- 建议 3 次，两臂各 3 次，全 6 个 sha 一致才算过。

### 4.1 但 0-ULP 通过 ≠ 有收益

**两次实例，同日**：

| 项 | 0-ULP | 收益 |
|---|---|---|
| D-1 分表异步 engram | 通过（6/6 同 sha） | **+0.14%** |
| P-1 `so4` 进寄存器 | 通过（6/6 同 sha） | **−7.5%** |

**正确性论证对，收益估计错** —— 这是常态，不是意外。

---

## 5. 上界估计：三个必查问题

提任何一个百分比之前，先回答：

### 5.1 这段时间在关键路径上吗？

`gpu_gap`（上一个 token GPU 结束 → 下一个 GPU 开始）**恰恰是不能与 GPU 重叠的那部分**。
把它当成"可回收时间"是错的。

**踩过**：§162.4 / §170.3 —— 我用 `699 µs ÷ 23630 = +3.0%` 推出 D-1 的上界。
实测 +0.14%。gap-prof 直接给出：engram 662.9 → 38.6，**encode1 67.9 → 670.4**。
时间没消失，只是换了桶。

**根因**：一个 token 开始时 GPU 是空闲的（上一 token 的 logits 已采样完、
`waitUntilCompleted` 已返回）。第一次 commit 之前没有任何 GPU 活在跑，**没有东西可藏**。

### 5.2 这条路径历史上试过几次？

**动手前先读账本 §0.4「已证否清单」。**

**踩过**：§169.5b —— 我推"MoE 瓶颈是 MLP，加 MLP 就能提速"。
真实根因是**每权重指令数**；加 NSG 实测 **+77.9%（慢一倍）**。
账本 13 次尝试全败，`DQ_COST_PROBE` 的控制变量实验早就证明了机理。

### 5.3 数字有没有量级错误？

**踩过**：§166.1 外部提案两处硬错 —— `DECODE_SPLIT=4`（实际 20）、
`48.9 GB/tok`（实际 7.555）。**结论对、依据错，仍然不能用。**

---

## 6. 口径：先核自己的折算，再去核别人

### 6.1 桶覆盖多少 token，由 `cbs` 判定

`dtags` 的 `cbs` 是 command buffer 计数。`DEC_PRE_ATTN` 的 1000 CBs
= **25 token × 40 层**（`MYDS_PHASE_SPLIT_DECODE=8` ⇒ 每 8 token 采一次）。

**踩过**：§169.1 —— 我按 `out_tokens=200` 折算，得出"DEC_ATTN 只占 2.5%"。
改除 25 后是 **4.44 ms**，**正是原始待办写的 4.45 ms**。
**原记录是对的，我差点把它"纠正"成错的。**

### 6.2 不能用 kprof 的串行耗时算带宽

kprof armed 会强制串行排程，造出假气泡。只能看 kernel 数与形状。

**踩过**：§143.3 / §159.6 —— 拿 kprof 串行 517 µs 比生产 trace 1454 µs，
相减得"0.94 ms 空转"。两个口径。

### 6.3 `gpu_gap` 的各分量

`gpu_gap ≠ wake + sample + engine + engram + encode1 + launch` 的简单解释 ——
`GPU 忙` 与 `阻塞等待` **会重叠**，不得相加。decode 各 tag 互斥（CB 总和 = decode_cbs），可相加。

---

## 7. 部署：成对，且回滚也成对

| 配对 | 后果 |
|---|---|
| 二进制 ↔ metallib | 新二进制配老 metallib **拒绝启动**（必检内核列表） |
| 二进制 ↔ conf | 旧二进制配新 conf **退出码 2 拒绝启动**（"这个键没有任何代码读取"） |
| 源码 ↔ 生成表 | 加了 knob 不跑 `tools/knob_inventory.py --write` ⇒ `check-knobs` 报 STALE |

**判据**：`make check-knob-pairing`（knob 集合指纹，只随"认哪些键"变，不随排版漂移）。

**不要用 `cp` 覆盖正在运行的可执行文件**（内核 SIGKILL，rc=137）—— 用 `install` 换 inode。

---

## 8. 生产守卫：会动 8187 的脚本必须能把它拉回来

**事实**：2026-11-08 07:25:04 生产被停掉后无人重启，**停摆 6 分 32 秒**。
根因是结构性的：6 个 shell 验收脚本都会 kill 生产，**没有一个负责恢复**；
另有两个 python 基准用 `pgrep -x myds` 杀**所有** myds 进程（与端口无关）。

| 做法 | 用法 |
|---|---|
| shell 脚本 | `. "$(dirname "$0")/prod_guard.sh"` |
| 任何命令 | `tools/with_prod_guard.sh <命令…>` |
| 只看不跑 | `tools/with_prod_guard.sh --dryrun <命令>` |

**禁止** `pgrep -x myds` 这类按进程名的杀法 —— 只杀自己端口的实例：
`lsof -nP -iTCP:<port> -sTCP:LISTEN -t`。

**判别捷径**：`lsof -nP -iTCP:8187 -sTCP:LISTEN -t` 空 ⇒ 生产已经躺了。

---

## 9. 退出码与崩溃：基线要先记

改退出路径之前，先记 `ls ~/Library/Logs/DiagnosticReports/myds-*.ips | wc -l`。

**事实**（§165）：退出期 SIGSEGV 有**两层**原因 ——

1. `myds_metal_cleanup()` 在 `src/*.c` 里一次都没被调用 ⇒ residency 心跳活到 `exit()`；
2. 即使停了心跳，`_MTLCommandQueue dealloc → _purgeDevice` 仍会崩（驱动层时序）。

修法：停心跳（**有界** 200 ms，不用 `FOREVER` —— 卡住时永久挂起比崩溃更难诊断）
→ 释放自己的内存 → `fflush(NULL); _exit(rc)`。

**必须 `_exit` 不能 `return`**：`return` 会跑 atexit 与 `__cxa_finalize`，把竞态请回来。

**A/B 判据**：同一二进制，两臂各 ≥6 次退出，比崩溃计数（33% → 0%）。
两个 knob 存在只为**复现**：`MYDS_RESIDENCY_EXIT_STOP`（默认 1）、
`MYDS_EXIT_METAL_TEARDOWN`（默认 0）。

---

## 10. 结论的措辞：不要把"没测"说成"没变"

**踩过**：§163.6 —— 今天 43.0 t/s、账本 43.97。我一开始想写"没有退化"，
但**口径未对齐**（21 tok vs ~30 tok prompt）。**只能说"未确认"。**

判定"退化"的门槛：

- 同口径的**前后各 ≥3 次**测量
- 差值**超过实测波动范围**（不是超过某个拍的阈值）
- 单次差异 <5% 一律当噪声

---

## 11. 落账：否掉一个方向的证据，比失败的尝试更容易丢

**事实**：同一天两次"对话里查清了但没落账"——§159 节号写重了、§170 那条
engram 实测漏了整整四小时。

**原因**：落账时按"我刚做了什么"写，而不是按"下了什么结论"写。

**规则**：

- 失败的尝试有提交记录，**自动留痕**；
- **"查清后决定不做"什么都不留** ⇒ 必须当场进账本
- 凡是用来**否掉一个方向**的实测，写完立刻落账，不要攒

---

## 12. 附：判据速查

| 问题 | 判据 | 出处 |
|---|---|---|
| 配置真生效了吗 | `myds --config x --check-config x` 报"不同 0" | §155.2 |
| 二进制与 conf 配对吗 | `make check-knob-pairing` | §156 |
| 真的做了全量 prefill 吗 | 日志 `prefill: N tok in` 行数 == 请求数 | §167.6 |
| 两臂真的不同吗 | 产物 sha + conf diff + 启动日志关键行 | §26.10.1 |
| 桶覆盖多少 token | `cbs`，不是 `out_tokens` | §169.1 |
| 时间在关键路径上吗 | `gpu_gap` 之类**不可**重叠 | §162.4 |
| 这条试过了吗 | 账本 §0.4 | §169.5b |
| 生产还在吗 | `lsof -nP -iTCP:8187 -sTCP:LISTEN -t` 非空 | §164 |
| 崩溃计数变了吗 | 改动前后各数十份数 | §165 |

---

## 相关文件

- 性能账本：`docs/2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md`
- ULP 开关：`docs/ULP_SWITCHES.md`
- 生产守卫：`tools/prod_guard.sh`、`tools/with_prod_guard.sh`
- 可复用测量装置：`tools/probe_p2_attn_variant.sh`、`tools/probe_p1_attn_reg.sh`、
  `tools/probe_engram_async.sh`、`tools/parse_scaling.py`、`tools/bench_engram_rows.c`
