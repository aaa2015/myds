#!/bin/zsh
# =================================================================
# run_ppl_protocol.sh — 按 llama.cpp perplexity 口径量一次 ppl
#
#   ./run_ppl_protocol.sh <tag> <model.gguf> <text.txt> [chunks] [context]
#
# 口径来源（acrogenesis recipe kit PROTOCOL.md + prepare_data.py）：
#   · 语料 = Salesforce/wikitext wikitext-2-raw-v1 的 test split
#   · 取 len(strip)>=300 的段落，random.Random(1017).shuffle()，取前 64 段，
#     以 "\n\n" 连接、末尾补 "\n"
#   · 每段 512 token，共 8 段；每段只计分 [256, 510] ⇒ 2,040 个计分位置
#
# 为什么必须停生产：模型 362 GB，192 GB 机器上再加载一份会触发交换。
# 恢复走 trap，任何退出路径都会执行。
# =================================================================
set -u

TAG="${1:?usage: run_ppl_protocol.sh <tag> <model.gguf> <text.txt> [chunks] [context]}"
MODEL="${2:?missing model}"
TEXT="${3:?missing text}"
CHUNKS="${4:-8}"
CTX="${5:-512}"
MYDS_DIR="$HOME/Projects/myds"
FLEET_DIR="$HOME/Projects/deploy"
OUT_DIR="$HOME/myds-eval"
PORT=8181
RESTORED=0

restore_production() {
    (( RESTORED )) && return
    RESTORED=1
    echo "── 恢复生产 ──"
    cd "$FLEET_DIR" || return
    env -u MYDS_PROFILE_FILE -u MYDS_PROF ./start_fleet.sh 2>&1 | tail -2
    for i in $(seq 1 150); do
        curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok \
            && { echo "HEALTH_OK $(date +%H:%M:%S)"; return; }
        sleep 2
    done
    echo "❌ 生产未恢复健康 —— 需要人工介入"
}
trap 'restore_production' EXIT INT TERM

PID="$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | head -1)"
if [[ -n "$PID" && "$PID" == <-> ]]; then
    echo "── 停生产 pid=$PID ──"
    kill "$PID"
    for i in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$PID" 2>/dev/null && { echo "  强制终止"; kill -9 "$PID"; sleep 1; }
else
    echo "── :$PORT 无监听 ──"
fi
sleep 5

cd "$MYDS_DIR" || { echo "❌ 找不到 $MYDS_DIR"; exit 1; }
echo "── 跑 ${CHUNKS}x${CTX} ──"
start=$(date +%s)
./bin/score "$MODEL" --text "$TEXT" --context "$CTX" --chunks "$CHUNKS" \
    > "$OUT_DIR/$TAG.json" 2> "$OUT_DIR/$TAG.err"
rc=$?
echo "  rc=$rc  用时 $(( $(date +%s) - start ))s"
cat "$OUT_DIR/$TAG.json"
tail -2 "$OUT_DIR/$TAG.err" | sed 's/^/    /'
echo "DONE tag=$TAG"
