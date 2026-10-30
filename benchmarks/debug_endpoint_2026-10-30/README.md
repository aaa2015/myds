# 2026-10-30：`/debug/state` 真机验证

账本 §116。

## 设计约束

| 约束 | 理由 |
|---|---|
| `MYDS_DEBUG_ENDPOINT=1` 才开 | 服务端无鉴权 |
| 仅回环 | 唯一访问控制 |
| 关闭返 **404** 不是 403 | 不该确认隐藏端点存在 |
| 计数器 `reset=false` | 传 true 会清零，毁掉 A/B 臂分离 |

## 实测（二进制 `6ce62f86…`）

| 场景 | 结果 |
|---|---|
| 默认关，本机 | **404**（`registered yes, switch off, loopback yes`） |
| 开关开，本机 | **200**，`build/knobs/counters` 齐全 |
| 开关开，从 macmini（`192.168.88.4`） | **404**（`switch on, loopback no`） |
| 同刻本机对照 | **200** |

第三条是关键：同一台服务器、同一时刻，局域网被拒、回环放行。
原始证据：`loopback-probe.jsonl`、`lan-probe.jsonl`、`server-log.txt`。

## ★ `unread: 1` 不是 bug

macstudio 的 `myfleet/myds.conf` 仍是旧版（`735e0dc7…`，63 键），第 109 行是
`MYDS_TOPK_THRESHOLD=1` —— 正是 P0-1 发现的死键。
**端点在独立路径上复现了同一个缺陷。**

## 两个环境问题

**1. macstudio 缺 Metal 工具链**：`xcodebuild -downloadComponent MetalToolchain` 未执行。
`sync_myds.sh` 在 metallib 步骤失败，阻塞任何 shader 变更的部署。
本次绕法：diff 无 `.metal` 改动 → 预置已部署 metallib 并哈希自证（`fe557933…` 前后一致）。

**2. 跨机工具链漂移**：macmini `clang-2100.3.34.2` vs macstudio `clang-2100.4.1.1`。
同源码产出 `0f6bdb41…` vs `6ce62f86…` ⇒ **§105.6「与 macmini 逐字节相同」不再成立**。
同机可复现仍成立，跨机不成立。

## 复现

```bash
# 必须在窗口内；脚本会拒绝在生产监听时启动
python3 tools/run_register_only_service_window.py \
  --root /tmp/myds-window-p02c --pid $(lsof -nP -t -a -iTCP:8187 -sTCP:LISTEN) \
  --cwd ~/Projects/myfleet --port 8187 --protect myds.conf \
  -- python3 tools/probe_debug_endpoint.py --out /tmp/p02.jsonl
```
