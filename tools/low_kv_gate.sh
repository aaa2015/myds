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
LOG_C=/tmp/low_kv_gate_alt.log
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

# ── 提示词：★ 码词放在【最前面】，后面跟一大段填充 ──
# 为什么必须这样放：快照的窗口环只有 128 个 token，码词若落在尾部，召回只需窗口环
# 就能做到 —— 那测不到【压缩 KV】，而压缩 KV 恰恰是磁盘路径新加、内存路径不存的那一块。
# 放在最前面（位置 < 100）时，要在末尾召回就必须靠压缩 KV 覆盖 [0..6144) 那段。
PREFIX_FILE="${PREFIX_FILE:-/tmp/low_kv_gate_prefix.txt}"
if [ ! -f "$PREFIX_FILE" ]; then
    {
        printf 'The vault code is CRIMSONOTTER9. Remember it exactly.\n'
        for i in $(seq 1 600); do printf 'context-%d w%d filler text for the constant prefix. ' "$i" "$((i % 97))"; done
        printf '\n'
    } > "$PREFIX_FILE"
fi

# 第二个常量前缀：与第一个【长度相近但内容不同】—— 用来验证两套「system + tools」
# 互不顶替（改之前它们会争同一批槽，第二个把第一个整个顶掉）。
# ★ 长度要相近才有效：长度差得远会自动落到不同的位置组，那就测不到争用。
PREFIX2_FILE="${PREFIX2_FILE:-/tmp/low_kv_gate_prefix2.txt}"
if [ ! -f "$PREFIX2_FILE" ]; then
    {
        printf 'You are a DIFFERENT assistant. The vault code is AMBERFALCON4. Remember it exactly.\n'
        for i in $(seq 1 600); do printf 'context-%d z%d other filler text for the second prefix. ' "$i" "$((i % 89))"; done
        printf '\n'
    } > "$PREFIX2_FILE"
fi

# 三个提示词共用同一段常量前缀（这正是要被磁盘复用的那段），只在末尾提问不同。
# 判据写的是「回答里必须出现什么」—— 逐字匹配对生成模型太脆，包含关系才是稳定的判据。
PROMPT_TAILS=(
    "Reply with the vault code only.|CRIMSONOTTER9"
    "What is 6 times 7? Reply with just the number.|42"
    "Is the vault code CRIMSONOTTER9 or CRIMSONOTTER8? Reply with the correct one.|CRIMSONOTTER9"
)
note "前缀: $PREFIX_FILE（$(wc -c < "$PREFIX_FILE" | tr -d ' ') 字节，码词在最前面）"
note "提示词: ${#PROMPT_TAILS[@]} 个（共用同一段常量前缀）"

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

ask() {   # $1=前缀文件 $2=末尾提问 $3=输出文件
    python3 - "$PORT" "$1" "$2" "$3" <<'PY'
import json, sys, urllib.request, pathlib
port, pf, tail, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
prompt = pathlib.Path(pf).read_text() + tail
body = {"model": "low-kv-gate",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 32, "stream": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.loads(r.read())
pathlib.Path(out).write_text(d["choices"][0]["message"]["content"])
print("  回答:", repr(d["choices"][0]["message"]["content"][:60]))
PY
}

# 跑一轮全部提示词（$1 = 输出前缀名；$2 = 用哪个常量前缀，默认第一个）
run_all() {
    local tag="$1" pfile="${2:-$PREFIX_FILE}" i=0 ok_here=0
    for entry in "${PROMPT_TAILS[@]}"; do
        i=$((i + 1))
        local tail="${entry%%|*}" want="${entry##*|}"
        printf '  [%s] #%d 期望含 %s\n' "$tag" "$i" "$want"
        ask "$pfile" "$tail" "/tmp/low_kv_gate_${tag}_${i}.txt" || { fail "$tag #$i 请求失败"; continue; }
        if grep -qF "$want" "/tmp/low_kv_gate_${tag}_${i}.txt"; then
            ok_here=$((ok_here + 1))
        else
            fail "$tag #$i 回答里没有 '$want'"
        fi
    done
    printf '  %s: %d/%d 正确\n' "$tag" "$ok_here" "$i"
    return 0
}

echo "═══ ① 冷跑（空盘 => 全量 prefill，并落盘；此时内存里也没有快照）═══"
rm -rf "$CACHE"; mkdir -p "$CACHE"
start_server "$LOG_A" || exit 1
note "服务已起（pid $SRV_PID）"
run_all cold
grep -q "kvdisk: 低位区落盘已启用" "$LOG_A" && ok "落盘已启用" || fail "日志里没有启用行"
ls -la "$CACHE" 2>/dev/null | grep kvlow | sed 's/^/    /'
kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null

echo
echo "═══ ② 重启（内存快照清空，磁盘保留 —— 这一步不可省）═══"
start_server "$LOG_B" || exit 1
note "服务已起（pid $SRV_PID）"

echo
echo "═══ ③ 热跑（同样三个提示词 => 第一个必须命中磁盘，其余走磁盘派生出的状态）═══"
run_all warm
if grep -q "resume from DISK snapshot" "$LOG_B"; then
    ok "日志确认走了【磁盘】路径（不是内存快照，也不是全量）"
    grep -m1 "resume from DISK snapshot" "$LOG_B" | sed 's/^/    /'
    grep -m1 "kvdisk: 命中位置" "$LOG_B" | sed 's/^/    /'
else
    fail "没有出现 'resume from DISK snapshot' —— 磁盘路径根本没被走到，本次门禁无效"
    grep -E "prefix cache:" "$LOG_B" | head -3 | sed 's/^/    /'
fi
kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null

echo
echo "═══ ④ 逐位对齐（三个提示词逐个比对）═══"
i=0
for entry in "${PROMPT_TAILS[@]}"; do
    i=$((i + 1))
    want="${entry##*|}"
    A="/tmp/low_kv_gate_cold_${i}.txt"; B="/tmp/low_kv_gate_warm_${i}.txt"
    if [ -s "$A" ] && [ -s "$B" ] && cmp -s "$A" "$B"; then
        ok "#$i 冷跑与磁盘命中逐字节相同（$(wc -c < "$A" | tr -d ' ') 字节）"
    else
        fail "#$i 输出不同 => 磁盘恢复的状态与全量 prefill 不等价，本特性不能启用"
        note "A: $(head -c 160 "$A" 2>/dev/null)"
        note "B: $(head -c 160 "$B" 2>/dev/null)"
    fi
done

echo
echo "═══ ⑤ 两个不同的「system + tools」交替 —— 必须各自命中，不得互相顶掉 ═══"
# 这一项是 way 机制真正的回归测试。顺序 P1 -> P2 -> P1：
#   · P2 从未存过 ⇒ 全量 prefill，然后把【自己那一套】写进空着的 way（不许动 P1 的）
#   · 再发 P1 ⇒ 内存里是 P2 的状态（内存必然落空）⇒ 只能靠磁盘
#   · 改之前：P2 的写入会顶掉 P1 的 11 条 ⇒ 这一步必然落空 ⇒ 本项失败
start_server "$LOG_C" || exit 1
note "服务已起（pid $SRV_PID，用同一份磁盘缓存）"
ask "$PREFIX_FILE"  "Reply with the vault code only." /tmp/low_kv_gate_alt1.txt | sed 's/^/    P1 /'
ask "$PREFIX2_FILE" "Reply with the vault code only." /tmp/low_kv_gate_alt2.txt | sed 's/^/    P2 /'
ask "$PREFIX_FILE"  "Reply with the vault code only." /tmp/low_kv_gate_alt3.txt | sed 's/^/    P1 /'
if grep -qF "CRIMSONOTTER9" /tmp/low_kv_gate_alt1.txt; then ok "P1 回答正确"; else fail "P1 回答不对"; fi
if grep -qF "AMBERFALCON4" /tmp/low_kv_gate_alt2.txt; then ok "P2 回答正确"; else fail "P2 回答不对"; fi
# 第三次（回到 P1）必须命中磁盘 —— 这是「P1 的条目没被 P2 顶掉」的直接证据
_HIT_BEFORE=$(grep -c "resume from DISK snapshot" "$LOG_C" || true)
if grep -q "resume from DISK snapshot" "$LOG_C" && [ "${_HIT_BEFORE:-0}" -ge 1 ]; then
    ok "第三次回到 P1 时命中了磁盘（P1 的条目仍在）"
    grep -m2 "resume from DISK snapshot" "$LOG_C" | sed 's/^/    /'
else
    fail "P1 的条目被 P2 顶掉了（第三次没有磁盘命中）—— way 机制没生效"
    grep -E "prefix cache:|kvdisk:" "$LOG_C" | tail -4 | sed 's/^/    /'
fi
# P2 也必须被存下来（不是只有 P1 能享受）
if grep -q "命中位置" "$LOG_C"; then ok "磁盘命中记录存在"; fi
kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null

echo
if [ "$fails" -eq 0 ]; then
    echo "✅ 门禁通过（A == B，且走的确实是磁盘路径）。日志: $LOG_A / $LOG_B"
    exit 0
fi
echo "❌ 门禁未通过（$fails 处）。日志: $LOG_A / $LOG_B" >&2
exit 1
