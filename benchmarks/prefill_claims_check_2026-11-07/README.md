# prefill_claims_check_2026-11-07 —— 「prefill 掉了 20%」是怎么被否掉的

这个目录存在的目的：记录一次**被自己推翻**的结论，以及推翻它用的证据。
不是成功案例集，是失败复盘。

## 结论（先说）

**prefill 没有掉。台账 §105 的数字今天是可复现的。**

| 臂 | prompt | token | prefill | 与台账 §105（618.9）|
|---|---|---|---|---|
| 台账 §105（2026-10-27，`691bc26` 时代） | `p32k_ab__v0.txt` | 32,210 | 618.9 t/s | 基线 |
| `691bc26`（2026-10-27 构建） | **同一文件** | 32,225 | **615.2 t/s** | −0.6% |
| `b0b7b77`（今天 HEAD） | **同一文件** | 32,224 | **619.4 t/s** | +0.1% |

三者在 0.7% 以内，全部 `cached: 0`（真的跑了 prefill）。

## 我错在哪

我用**生成的英文随机词** prompt 测出 ~490 t/s，与台账的 618.9 相比，
判定「掉了 20%」，还据此去二分提交、怀疑代码回归。

同 token 数、不同语料：

| prompt | token | prefill（同一只二进制）|
|---|---|---|
| 英文随机词 | 38,980 | 496.4 |
| 台账中文语料 | 32,224 | **619.4** |

**差 25%，全在 prompt 内容。** 随机词表的 MoE 路由几乎均匀铺满所有专家、
没有局部性；真实语料路由集中、权重驻留命中率高。生成文本做基线会系统性低估。

所以错不在机器、不在代码，在**我拿一个不该比较的数去比较**。

## 顺带否掉的两个「嫌疑人」

怀疑「掉了 20%」时列过两个候选，都实测排除：

| 假设 | 实测 | 结论 |
|---|---|---|
| 10-27 → HEAD 代码回归 | 同一 38.9K prompt：`691bc26` **497.0** / HEAD **496.4** | 否 |
| `MYDS_CHUNK_KEY_MAX` 524288→1048576（`234bf98` 记录的首要候选） | **498.0**（512K） vs **498.9**（1M） | 否 |

`234bf98`（2026-10-31）当时测出 32K prefill −5.3%，把 `CHUNK_KEY_MAX` 列为首要嫌疑。
本轮实测它对 prefill 没有影响 —— 那 −5.3% 大概率同样是条件差异。

## 口径（要复用这份数据，先读这一段）

- 机器：macstudio，M2 Ultra 76 核 / 192 GiB **独占**，无其它 myds 进程。
- 配置：`conf/myds-bench.conf`（线上 `myds.conf` 原样，**仅注释掉 `MYDS_KV_DSK`**，
  避免每请求写 GB 级快照把落盘时间混进墙钟；凭证已删）。
- 请求：每轮先发一次 `X-Myds-Warmup: 1` 的 306 token 探针，再发目标 prompt，
  末尾追加唯一 nonce ⇒ 前缀缓存必然 miss（日志里 `cached: 0` 可验）。
- 指标：服务端自己的 `Completed` 日志行（`prefill: N tok in S s`），**不是 HTTP 墙钟**。
- 两臂都是各自 commit 的完整构建（**含各自的 metallib**）。所以这只回答
  「今天的东西和台账的一样快吗」，**不能**用来把差异归给某一行代码。

## 目录内容

```
README.md        本文
runs.jsonl       每次测量一行，由 extract.py 从 raw 日志生成，禁止手改
extract.py       raw/*.log → runs.jsonl；--check 校验两者一致
verify.py        把可比的 claim 与 runs.jsonl / 台账原始行对上
raw/*.log        服务端原始日志（逐行保留，含预热行与失败尝试）
scripts/measure_one.sh   实际用的测量脚本
conf/myds-bench.conf     实际用的配置（已脱敏）
```

## 怎么复核（从低到高）

| 层 | 命令 | 需要 | 耗时 |
|---|---|---|---|
| L1 表与日志一致 | `python3 extract.py --check` | Python 3 | <1s |
| L2 claim 有出处 | `python3 verify.py` | Python 3 | <1s |
| L3 全仓库门 | `make audit-ledger`（仓库根） | Python 3 | 秒级 |
| L4 重测 | `scripts/measure_one.sh <tag> <binary>` | M2/M3 Ultra + 362 GB 权重 | 每臂约 1.5 分钟 |

L1/L2 都不需要 GPU、不需要模型。

## 变成了什么规则

`benchmarks/prefill_claims.json` 登记了文档里出现的每个 `t/s` 字面量；
`comparable_to_ledger_105: true` 的条目必须指到台账语料文件、sha256 对得上 manifest，
否则 `make audit-ledger` 失败。**这个检查的自测里有一条，就是本文这个错误形状**
（「无语料却声称可比」）。

## 这份 archive 不能证明什么

- 不能证明 10-27 到今天的**每一条**路径都没变慢 —— 只覆盖 prefill 吞吐、单一 prompt 尺寸。
- decode 不在本目录范围内（另测过：24 token prompt、160 token 输出，两臂 42.4–42.6 t/s，
  与台账 §0.1 的 42.0 一致）。
- 两臂的 metallib 不同，无法把 0.6% 的差异归因到具体一层的编译器产物。
