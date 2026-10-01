#!/bin/bash
# tools/low_kv_gate.sh — 低位区 KV 落盘的【逐位对齐门禁】（冷跑 vs 磁盘命中）
#
# 为什么必须做这一步：本仓库的硬规矩是「任何改数值路径的开关，接受前必须与全量 prefill
# 逐位对齐」（AGENTS.md「加速开关必须配输出与金标准逐位对齐的门禁」）。
# 低位区落盘把「状态从哪里来」换成了磁盘，正是这类开关。
#
# 做法（必须跨【重启】，否则内存快照会先命中，测不到磁盘路径）：
#   ① 冷跑：起实例（MYDS_KV_DSK=<空目录>），发提示词 P（temp=0）→ 输出 A，同时落盘
#   ② 重启：内存快照清空、磁盘还在
#   ③ 热跑：再发同一个 P → 日志必须出现 "resume from DISK snapshot" → 输出 B
#   ④ 断言 A == B（同种子 + temp=0 的贪心解码 ⇒ 文本相同即 token 相同；文本是 token 的
#      确定性函数，反例只可能来自「多个 token 映到同一文本」这种极端情形）
#
# ★ 需要模型常驻 155 GB ⇒ 跑之前必须把生产 8181 停掉。脚本会自己检查端口。
#
# 用法: tools/low_kv_gate.sh <myds_bin> <model.gguf> <cache_dir> [port] [ctx]
set -uo pipefail

BIN="${1:?用法: $0 <myds_bin> <model.gguf> <cache_dir> [port] [ctx]}"
MODEL="${2:?缺少 model.gguf}"
CACHE="${3:?缺少 cache_dir}"
PORT="${4:-8199}"
CTX="${5:-32768}"

PROMPT_FILE="${PROMPT_FILE:-/tmp/low_kv_gate_prompt.txt}"
LOG_A=/tmp/low_kv_gate_cold.log
LOG_B=/tmp/low_kv_gate_warm.log
OUT_A=/tmp/low_kv_gate_A.txt
OUT_B=/tmp/low_kv_gate_B.txt

fails=0
note() { printf '  %s\n' "$*"; }
fail() { printf '  ✗ %s\n' "$*"; fails=$((fails + 1)); }
ok()   { printf '  ✓ %s\n' "$*"; }

# ── 前置检查 ──
# 必须在【有模型的那台机器】上跑（模型 155 GB，常驻）。先看文件在不在。
if [ ! -f "$MODEL" ]; then
    echo "✗ 找不到模型文件: $MODEL" >&2
    echo "  这条门禁需要模型常驻，请在 macstudio 上跑。" >&2
    exit 2
fi
# 端口不能被占（那说明生产还在跑）
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "✗ 端口 $PORT 已被占用 —— 生产还在跑？先停掉再跑本门禁（它要独占 155 GB）。" >&2
    exit 2
fi

# ── 提示词：要足够长，让低位区那 11 个固定槽都有东西可存 ──
if [ ! -f "$PROMPT_FILE" ]; then
    {
        printf 'You are a careful assistant. Answer with only the requested value.\n'
        for i in $(seq 1 600); do printf 'context-%d w%d filler text for the constant prefix. ' "$i" "$((i % 97))"; done
        printf '\nRemember this codeword: CRIMSONOTTER9.\n'
        printf 'Question: reply with the codeword only.\n'
    } > "$PROMPT_FILE"
fi
note "提示词: $PROMPT_FILE（$(wc -c < "$PROMPT_FILE" | tr -d ' ') 字节）"

start_server() {   # $1=日志文件
    MYDS_PREFIX_CACHE=1 MYDS_KV_DSK="$CACHE" \
        "$BIN" -m "$MODEL" --port "$PORT" -c "$CTX" --alias low-kv-gate \
        > "$1" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 180); do
        if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then return 0; fi
        kill -0 "$SRV_PID" 2>/dev/null || { echo "✗ 服务进程已退出，看 $1" >&2; return 1; }
        sleep 2
    done
    echo "✗ 起服务超时（$1）" >&2
    return 1
}

ask() {   # $1=输出文件
    python3 - "$PORT" "$PROMPT_FILE" "$1" <<'PY'
import json, sys, urllib.request, pathlib
port, pf, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = pathlib.Path(pf).read_text()
body = {"model": "low-kv-gate",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "seed": 42, "max_tokens": 64, "stream": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.loads(r.read())
pathlib.Path(out).write_text(d["choices"][0]["message"]["content"])
print("  回答:", repr(d["choices"][0]["message"]["content"][:80]))
PY
}

echo "═══ ① 冷跑（空盘 ⇒ 全量 prefill，并落盘）═══"
rm -rf "$CACHE"; mkdir -p "$CACHE"
start_server "$LOG_A" || exit 1
note "服务已起（pid $SRV_PID）"
ask "$OUT_A" || { fail "冷跑请求失败"; kill "$SRV_PID" 2>/dev/null; exit 1; }
grep -c "kvdisk: 低位区落盘已启用" "$LOG_A" >/dev/null && ok "落盘已启用" || fail "日志里没有启用行"
grep -E "kvdisk: 槽|写盘失败" "$LOG_A" | head -3 | sed 's/^/    /'
ls -la "$CACHE" 2>/dev/null | tail -2 | sed 's/^/    /'
kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null

echo
echo "═══ ② 重启（内存快照清空，磁盘保留）═══"
start_server "$LOG_B" || exit 1
note "服务已起（pid $SRV_PID）"

echo
echo "═══ ③ 热跑（同一个提示词 ⇒ 应命中磁盘）═══"
ask "$OUT_B" || { fail "热跑请求失败"; kill "$SRV_PID" 2>/dev/null; exit 1; }
if grep -q "resume from DISK snapshot" "$LOG_B"; then
    ok "日志确认走了【磁盘】路径（不是内存快照，也不是全量）"
    grep -m1 "resume from DISK snapshot" "$LOG_B" | sed 's/^/    /'
else
    fail "没有出现 'resume from DISK snapshot' —— 磁盘路径根本没被走到，本次门禁无效"
    grep -E "prefix cache:" "$LOG_B" | head -3 | sed 's/^/    /'
fi
kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null

echo
echo "═══ ④ 逐位对齐 ═══"
if [ -s "$OUT_A" ] && cmp -s "$OUT_A" "$OUT_B"; then
    ok "冷跑与磁盘命中的输出【逐字节相同】（$(wc -c < "$OUT_A" | tr -d ' ') 字节）"
else
    fail "输出不同 —— 磁盘恢复的状态与全量 prefill 不等价，本特性【不能启用】"
    note "A: $(head -c 200 "$OUT_A")"
    note "B: $(head -c 200 "$OUT_B")"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "✅ 门禁通过（A == B，且走的确实是磁盘路径）。日志: $LOG_A / $LOG_B"
    exit 0
fi
echo "❌ 门禁未通过（$fails 处）。日志: $LOG_A / $LOG_B" >&2
exit 1
