#!/bin/bash
# §151 验收：indexer rows4 的端到端 A/B。
# 两臂只差 MYDS_INDEXER_SCORE_ROWS4（0 = 1-row smem 原版，1 = rows4）。
# 每个上下文长度各跑 3 轮，交替。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
BIN=/tmp/myds-new
D=$ROOT/acc151
rm -rf "$D"
mkdir -p "$D/metal"
cp /tmp/myds.metallib "$D/metal/myds.metallib"
cd "$D" || exit 1

PROD=$(pgrep -f "myfleet/myds.conf" | head -1)
[ -n "$PROD" ] && kill -TERM "$PROD"
for i in $(seq 1 90); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 3

run_arm() {
  local tag=$1 rows4=$2 ctx=$3 rep=$4
  local out="$D/${tag}_${ctx}_$rep"
  mkdir -p "$out"
  MYDS_INDEXER_SCORE_ROWS4=$rows4 MYDS_ATTN_WHICH=1 "$BIN" \
    --config "$HOME/Projects/myfleet/myds.conf" -m "$MODEL" \
    --port $PORT --host 127.0.0.1 --ctx-size 1048576 --alias bench --temp 0 --top-p 1.0 \
    --max-tokens 256 --thinking off --vision "$VISION" > "$out/server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 900); do curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -sf -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -H "X-Myds-Warmup: 1" -d @"$ROOT/W.json" -o /dev/null
  curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d @"$ROOT/$ctx.json" -o "$out/resp.json" -w "$tag $ctx r$rep %{time_total}\n"
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null
  sleep 4
}

for rep in 1 2 3; do
  run_arm ref 0 d2k  "$rep"
  run_arm r4  1 d2k  "$rep"
done
for rep in 1 2; do
  run_arm ref 0 d24k "$rep"
  run_arm r4  1 d24k "$rep"
done
echo ACC151_DONE
