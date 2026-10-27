# register_only_2026-10-27 — 账本 §105 的可复现数据包

这个目录存在的目的只有一个：让**别人（含另一个模型）能独立检验 §105 的数字**，
而不是读一段自称严谨的文字然后选择相信。

## 三层复现，代价从低到高

| 层次 | 命令 | 需要什么 | 耗时 |
|---|---|---|---|
| L1 算术 | `make audit-ledger` | Python 3 | 秒级（166 项）|
| L2 判定逻辑 | `make verify-register-only` | Python 3 | 秒级（36 项）|
| L3 实测 | `make bench-register-only` | M2/M3 Ultra 192 GiB + 362 GB 权重 + 免密 sudo | 约 60 分钟 |

**L1/L2 不需要 GPU、不需要模型、不需要本机是 Mac。**
这是刻意的：最低成本的检查也要能给出有意义的结论。

完整说明见 [`docs/REPRODUCE-register-only.md`](../../docs/REPRODUCE-register-only.md)。

## 目录内容

```
prompts/            31 个语料文件 + manifest.json
                    6 个 prefill 尺寸 + 1 个 A/B 尺寸 + 3 个 decode 场景，
                    各 3 个变体，另加 1 个专供"固定样本"的文件
                    变体存在的唯一原因：避开服务端 prefix cache
raw/                rig.jsonl / sweep2.jsonl / dec2.jsonl / smoke2.jsonl
                    账本 §105 每个数字的原始出处，逐行保留，未做任何汇总；
                    smoke2.jsonl 是 §106.5 单臂冒烟的四行原始记录
expected.json       期望值 + 容差 + 为什么这样定
runs/               你自己跑出来的结果放这里（不入库）
```

## 三个容易踩的坑（都实际踩过）

1. **语料变体不能复用。** 固定样本曾与第一个计时样本共用同一个文件，
   结果固定样本把缓存填满，第一个计时样本 0.17 s 返回、`cached != 0`，
   **根本没有发生 prefill**。现在 `ab_fixed.txt` 独立存在，脚本里还有一道
   「同一 boot 内 prompt 不得重复」的断言。
2. **遥测要在负载中采。** 请求结束后再采只能读到排空残值（0–2.31% 驻留、0–386 mW），
   与负载态（100%、104 W）差两个量级。
3. **别在小机器上跑。** 本基准钉住约 150 GB；16/64/128 GiB 的机器会因 wired 耗尽
   **内核 panic**，不是 OOM kill。脚本预检会硬拒绝。

## 语料是可审计的，不是一坨二进制

`tools/make_register_only_corpus.py` 是生成器；`--check` 模式重算全部 31 个文件
并与磁盘逐字节比对，再与 `manifest.json` 的 sha256 表核对。
`make corpus-check` 就是执行这一步。

改语料 → manifest 会变 → `corpus-check` 会红。这是设计意图。
