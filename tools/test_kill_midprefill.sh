#!/bin/bash
# 决定性判别：P0 的 engram 后台线程在「prefill 中途被 SIGTERM」时会不会崩？
# 生产二进制 a79b410a（含 P0）跑 32K prefill，10 秒后 SIGTERM，看是否产生崩溃报告。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
D=$ROOT/killmid
rm -rf "$D"; mkdir -p "$D"
cd "$D" || exit 1

PROD=$(pgrep -f "myfleet/myds.conf" | head -1)
[ -n "$PROD" ] && kill -TERM "$PROD"
for i in $(seq 1 90); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 3

before=$(ls ~/Library/Logs/DiagnosticReports/ 2>/dev/null | wc -l | tr -d ' ')
echo "基线崩溃报告: $before"

"$HOME/Projects/myds/bin/myds" --config "$HOME/Projects/myfleet/myds.conf" -m "$MODEL" \
  --port $PORT --host 127.0.0.1 --ctx-size 1048576 --alias bench --temp 0 --top-p 1.0 \
  --max-tokens 8 --thinking off --vision "$VISION" > kill.log 2>&1 &
pid=$!
for i in $(seq 1 600); do curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -H "X-Myds-Warmup: 1" -d @"$ROOT/W.json" -o /dev/null
# 32K冷 prefill 要 ~50s；10 秒后打断，此时 engram 后台线程正在跑
python3 - "$ROOT/p32k_v0.json" <<'PY'
import json, sys, random
d = json.load(open(sys.argv[1])); d["max_tokens"] = 8
d["messages"][0]["content"] += "\n\n[killmid %d]" % random.randint(10**8, 10**9)
json.dump(d, open("/tmp/km.json", "w"), ensure_ascii=False)
PY
curl -sf -m 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d @/tmp/km.json -o /dev/null -w "req %{time_total}\n" &
cpid=$!
sleep 10
echo "--- 10s 后 SIGTERM（prefill 应在进行中）---"
kill -TERM $pid 2>/dev/null
wait $cpid 2>/dev/null
for i in $(seq 1 40); do kill -0 $pid 2>/dev/null || break; sleep 1; done
kill -9 $pid 2>/dev/null
sleep 3

after=$(ls ~/Library/Logs/DiagnosticReports/ 2>/dev/null | wc -l | tr -d ' ')
echo "现在: $after  新增 $((after-before))"
ls -lt ~/Library/Logs/DiagnosticReports/ | head -3
echo "--- 日志尾部 ---"
tail -4 kill.log | cut -c1-130
echo KILLMID_DONE
