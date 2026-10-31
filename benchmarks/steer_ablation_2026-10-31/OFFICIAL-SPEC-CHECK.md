# 官方规格核对：为什么我们的 v3 不可用（账本 §133）

## 官方数据（HuggingFace 模型卡）

**GLP-27（v3，α=3.5）官方 ladder，thinking **on**：**

| α | weaponize32 拒答 | 备注 |
|---:|---:|---|
| stock | 20/32 | |
| 1.0 | 16/32 | |
| 2.0 | 8/32 | |
| 3.0 | 5/32 | |
| **3.5** | **4/32** | **operating point — clean outputs** |
| 4.0 | 3/32 | CVE-ID token 损坏出现 |
| 5.0 | 4/32 | regression |

3.5 处：refusal32 **6/32 拒答（stock 32/32）**、**zero garbled**、completion 中位 **1871 字符**。

**GLP-39（v2，α=0.75）：** refusal32 1/32 → **23/32 应答**；cyber32 12/32 → 29/32。

⇒ **官方意义上 v3 比 v2 强**（26/32 vs 23/32）。

## 我们的实测：v3 六次零干净

同 prompt、α=3.5、预算 4096：

| # | content | 判定 |
|---:|---:|---|
| 0 | 0 | EMPTY |
| 1 | 6,918 | 英文乱码（CJK 0.08）|
| 2 | 0 | EMPTY |
| 3 | 5,145 | 英文乱码（CJK 0.03）|
| 4 | 0 | EMPTY |
| 5 | 0 | EMPTY |

## 三个不合规点

**① Q2 量化不在验证范围。** 官方：「direction … calibrated on that exact checkpoint …
**not validated against other revisions or quants**」。方向来自 **FP8+MXFP4** 的 FFN writer SVD；
我们跑 **Q2**。writer 与 SVD 行不对齐 ⇒ 削的不是拒答方向，且 `(h·d)` 量级改变 ⇒
同样 α 的等效剂量更高。

**② 挂钩位置可能是 post-Engram。** 官方要求 `hc_fold_pre_engram`（**pre-Engram**）。
我们的残差钩在 `myds_v41_moe_pipeline_scoped(...)` 之后（`src/myds_v41.c:4565`），
而 `engram_kv` 在该调用内被消费 ⇒ **L0/L13 两层是 post-Engram 施加**。
官方要求不合规 reader「必须拒绝该文件」——我们没拒。

**③ 内核不确定度，官方自己也记录了。** GLP-39 卡片 no-op gate：
「alpha=0 … 29/32 … all three mismatches … **attributable to engine kernel nondeterminism
across boots**」。与 §126/§128 指向同一现象。

## 我上一节的错误

我写「α=3.5 是 350% 过减，建议降到 1.0」。**数学没错，结论错**：
官方在 3.5 验证干净；且 α=1.0 只有 16/32 应答，**3.5 才有 4/32**。降到 1.0 会砍掉大半投送。

## 正确做法

官方原文：**「re-run the ladder」**——α 不可跨模型/量化移植。

1. 在我们自己的 Q2 栈上重跑 ladder，每点 **≥6 次重复**，判据是「**干净率**」而非「是否不同」
2. 先修挂钩位置（移到 engram 之前，或 L0/L13 跳过）
3. 若低 α 仍不稳定 ⇒ 结论应是「Q2 不支持该控制向量」，而非继续调 α
