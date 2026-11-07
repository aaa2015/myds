#!/bin/bash
# 判别：关机段错误是本次提交引入的，还是老二进制也有的既有问题？
# 做法：老二进制（a79b410a）跑一次请求，然后 SIGTERM，看是否产生崩溃报告。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
D=$ROOT/shutdown_test
rm -rf "$D"; mkdir -p "$D"
cd "$D" || exit 1

PROD=$(pgrep -f "myfleet/myds.conf" | head -1)
[ -n "$PROD" ] && kill -TERM "$PROD"
for i in $(seq 1 90); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 3

before=$(ls ~/Library/Logs/DiagnosticReports/ 2>/dev/null | wc -l | tr -d ' ')
echo "崩溃报告基线数: $before"

# 老二进制：macstudio 上 ~/Projects/myds/bin/myds（a79b410a）+ 同目录 metallib
"$HOME/Projects/myds/bin/myds" --config "$HOME/Projects/myfleet/myds.conf" -m "$MODEL" \
  --port $PORT --host 127.0.0.1 --ctx-size 1048576 --alias bench --temp 0 --top-p 1.0 \
  --max-tokens 64 --thinking off --vision "$VISION" > old.log 2>&1 &
pid=$!
for i in $(seq 1 600); do curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -H "X-Myds-Warmup: 1" -d @"$ROOT/W.json" -o /dev/null
# 请求发出去后立刻 SIGTERM，制造「有 inflight CB」的关机场景
curl -sf -m 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d @"$ROOT/d2k.json" -o /dev/null -w "old req %{time_total}\n" &
cpid=$!
sleep 0.4
kill -TERM $pid 2>/dev/null
wait $cpid 2>/dev/null
for i in $(seq 1 40); do kill -0 $pid 2>/dev/null || break; sleep 1; done
kill -9 $pid 2>/dev/null
sleep 3

after=$(ls ~/Library/Logs/DiagnosticReports/ 2>/dev/null | wc -l | tr -d ' ')
echo "崩溃报告现在: $after  (新增 $((after-before)))"
ls -lt ~/Library/Logs/DiagnosticReports/ | head -3
echo SHUTDOWN_TEST_DONE
