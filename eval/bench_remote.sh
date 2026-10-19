#!/bin/zsh
# 在 macstudio 上直接跑，避免 ssh heredoc 与 zsh 展开冲突。
set -u
TAG="${1:?tag}"
MODEL="${2:?model}"
PORT=8199
MYDS_DIR="$HOME/Projects/myds"
OUT="$HOME/myds-eval/perf-$TAG.json"
LOG="$HOME/myds-eval/perf-$TAG.log"
PID=""

cleanup() {
    if [[ -n "$PID" && "$PID" == <-> ]]; then
        kill "$PID" 2>/dev/null
        for i in $(seq 1 40); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
        kill -9 "$PID" 2>/dev/null
    fi
}
trap 'cleanup' EXIT INT TERM

cd "$MYDS_DIR" || exit 1
echo "── 起 $TAG 于 :$PORT ──"
./bin/myds -m "$MODEL" --port $PORT --host 127.0.0.1 \
    --ctx-size 8192 --temp 0 --top-p 1.0 --max-tokens 4096 --thinking off \
    > "$LOG" 2>&1 &
PID=$!
echo "  pid=$PID"

for i in $(seq 1 240); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break
    kill -0 "$PID" 2>/dev/null || { echo "❌ 服务退出"; tail -5 "$LOG"; exit 1; }
    sleep 1
done
curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok || { echo "❌ 健康检查超时"; exit 1; }
echo "  就绪 $(date +%H:%M:%S)"

/usr/bin/python3 "$HOME/Projects/myds/eval/perf_client.py" "$PORT" 2 "$OUT"
echo "  写出 $OUT"
