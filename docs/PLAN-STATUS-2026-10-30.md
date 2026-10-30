# 优化方案执行状态（2026-10-30 汇总）

对应 `implementation_plan.md` 的 P0–P3。**本文件只是索引；结论以账本对应章节为准。**

## P0 配置与路径可观测

| 项 | 状态 | 产物 | 账本 |
|---|---|---|---|
| P0-1 `--print-config` + knob 清单 | **完成** | `tools/knob_inventory.py`、`include/myds_knobs_generated.h`、`myds --print-config`、`make check-knobs` | §A.2 之后 |
| P0-2 `/debug/state` 端点 | **完成**（CPU 层验证；HTTP 层待窗口） | `src/myds_debugstate.c`、`myds --print-config` 同源的 knob 视图 | 待补 §116 |
| P0-3 ABI 交叉校验 | **跳过**（评估为包装，非新能力） | — | — |
| P0-4 独占窗口 | **完成并以真机验证** | `tools/run_register_only_service_window.py`（nice/stdio 保留 + 内存回收门）、`tools/gpu_window.py`（只读核对） | §108 后续 |
| P0-5 请求阶段计时 | **完成并已部署** | `MYDS_REQUEST_PHASES=1` | §111 → §113 |

**P0-1 上线当天抓到真缺陷**：生产 `myds.conf` 的 `MYDS_TOPK_THRESHOLD=1`
是死键（代码读 `MYDS_INDEXER_TOPK_THRESHOLD`，默认 false），自 2026-10-23 加入起从未生效。
同一错名也复制在 `tools/run_kprof_window.sh` ⇒ **过去的 kprof decode 分解是在该路径关闭下测的。**
已删键并留注释。

## P1 归因实验

| 实验 | 状态 | 结论 | 账本 |
|---|---|---|---|
| E3 capacity A/B | **完成** | `CHUNK_KEY_MAX` 1M→512K = **+0.02%**；ctx 影响 −0.39%（噪声内）⇒ §105.7 关账 | §111 |
| E2 热追加 | **完成** | 1.89 s 固定 + 2.68 ms/tok（1.56×）；MoE **每 chunk 恒定 ~1.1 s** | §112、§114 |
| E1 短 prompt | **完成** | 1.222 s 固定 + 2.005 ms/tok；复现 §105.6 | §115 |

## P2 预声明分支

| 分支 | 触发条件 | 结果 |
|---|---|---|
| B1 KV 缓冲按需增长 | E3 差 >2% | **未触发**（差 0.02%）⇒ 不立项 |
| B2 定点修（短 prompt） | E1 非 prefill 固定成本 ≥0.3 s | 达阈值（0.46 s）但**无对应可修项** ⇒ 不立项 |
| B3 定点修（热追加） | E2 超额 ≥2 s 且不在 FLOP 本体 | 超额在 prefill 内；主因是专家权重流 ⇒ **记为边界，不立项** |

## P3 采纳与记录

- 部署：`myfleet/sync_myds.sh`（指纹校验 + 临时目录重编译 + 原子替换 + `.deploy-prev`）
- 生产二进制：`bbe8aa66…` → **`227ae371…`**（`ca1a17d`），metallib/config 未变
- `make audit-ledger` 每次提交后 PASS；`make test` 全绿

## 净结果（诚实版）

**三个实验全部完成，零个采纳优化。** 这是正确结果，不是失败：

| 方向 | 结论 | 省下的工作 |
|---|---|---|
| §105.7 `CHUNK_KEY_MAX` | 无影响 | B1：1–2 天 |
| G1 热追加 | 专家权重流边界 | B3：1–2 天（且我先提的「补零对齐」被自己证否：4.04→7.37 s） |
| G2 短 prompt | 同一权重流边界 | B2：1–2 天 |

**真正的产出是三条可复核的边界 + 一套能把「付在哪一段」量出来的工具**，
而不是一个更快的数字。跨请求合批仍是唯一未排除的杠杆，其收益上界已写入
§114/§115（`1.22–1.89 s × (1 − 1/并发数)`），且不改数值路径。

## P0-2 说明

`GET /debug/state` 已实现：默认关闭（`MYDS_DEBUG_ENDPOINT=1` 才开）、**且仅回环**，
关闭时返回 404 而不是 403（无鉴权的服务不该确认隐藏端点的存在）。
内容：binary/metallib SHA256、knob 统计（known/set/unread）、
register-only 计数器、prefix cache 长度。

**三条硬约束写在代码注释里**：
1. 计数器访问器必须 `reset=false` —— 传 true 会 `atomic_exchange` 清零，
   直接毁掉账本里每次 A/B 依赖的臂分离（§105 的「requests 恒 2897、encoded 0 vs 2025」）。
   读操作永远不允许减去证据。
2. 只读：不插 command buffer、不 commit、不改驻留。
3. 实现放在独立的 `myds_debugstate.c`，由 `main.c` 通过函数指针注册。
   理由：`myds_server.c` 若直接依赖 knob 清单 / GPU 计数器 / Metal 路径，
   会波及所有 `include server.c` 的轻量测试 —— 今天已因此踩过一次。

**验证**：`tests/test_debug_state.c` 用系统 `shasum` 对拍手写 SHA256（不自证）、
检查 JSON 结构、确认缺失文件返回空而不是假零、确认回环判定的五个边界。
反向对照：给错哈希则 exit 1。

**未验证**：HTTP 层（404/200 分支的真实收发）。起第二个 myds 实例会占 ~300 GB
并触发 wired panic，所以只能在部署或窗口内验证。

## 仍未做

- §105.6 / §111 / §113 的制品身份已三次过期，任何引用旧 SHA 的对照实验都指向不存在的二进制
- 跨请求合批的重新评估（若重开，本文件的收益上界可直接用）
