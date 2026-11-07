#!/bin/bash
# 2026-11-07：确认生产实际派发哪个 prefill attention kernel。
# 起因：P1 的成本探针打在 kernel_flash_attn_ext（heads）上，
# 跑了两轮冷启动才发现 MYDS_PREFILL_ATTN 未设 ⇒ variant=0 ⇒ use_matrix 恒 0，
# 生产根本不走那条分支。成功的派发一直没有日志，所以只能靠 MYDS_ATTN_WHICH=1。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
D=$ROOT/which
rm -rf "$D"
mkdir -p "$D/metal"
cp "$HOME/Projects/myds/bin/myds.metallib" "$D/metal/myds.metallib"
cd "$D" || exit 1

MYDS_ATTN_WHICH=1 /tmp/myds-which --config "$HOME/Projects/myfleet/myds.conf" -m "$MODEL" \
  --port $PORT --host 127.0.0.1 --ctx-size 1048576 --alias bench --temp 0 --top-p 1.0 \
  --max-tokens 1 --thinking off --vision "$VISION" > server.log 2>&1 &
PID=$!
for i in $(seq 1 900); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

python3 - "$ROOT/p32k_v0.json" <<'PY'
import json, sys, random
d = json.load(open(sys.argv[1]))
d["max_tokens"] = 1
d["messages"][0]["content"] += "\n\n[which %d]" % random.randint(10**8, 10**9)
json.dump(d, open("/tmp/w24k.json", "w"), ensure_ascii=False)
PY

curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" -d @/tmp/w24k.json -o /dev/null \
  -w "prefill done %{time_total}\n"

kill -TERM $PID 2>/dev/null
for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -9 $PID 2>/dev/null
echo WHICH_DONE
