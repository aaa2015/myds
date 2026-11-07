#!/bin/bash
# 一次测量：一只二进制 + 一份 prompt（带唯一 nonce 保证缓存必 miss）。
#
# 这是 2026-11-07 复盘时用的原脚本，只改了路径来源（ROOT/PORT 可覆盖）。
# 用法：measure_one.sh <tag> <binary> [prompt.json]
#   ROOT 下需要有：W.json（预热探针）、<prompt>.json、myds-bench.conf
#   prompt 默认 p32k_v0.json，即由本仓库
#   benchmarks/register_only_2026-10-27/prompts/p32k_ab__v0.txt 生成的那份。
set -u
ROOT=${ROOT:-$HOME/bench/ab}; PORT=${PORT:-8187}; TAG=$1; BIN=$2
cd "$ROOT" || { echo "no ROOT $ROOT"; exit 1; }
MODEL=${MODEL:-$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf}
VISION=${VISION:-$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf}
P=${3:-p32k_v0}
OUT=$ROOT/results/$TAG; mkdir -p "$OUT"
LOG=$OUT/server.log; : > "$LOG"
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "$TAG PORT_BUSY" >> $ROOT/p32.log; exit 3; fi
python3 - "$ROOT/$P.json" "$OUT/req.json" "$TAG" <<PY
import json,sys,random
d=json.load(open(sys.argv[1]))
d["messages"][0]["content"] += "\n\n[run %s %d]" % (sys.argv[3], random.randint(10**8,10**9))
json.dump(d, open(sys.argv[2],"w"), ensure_ascii=False)
PY
"$BIN" --config "$ROOT/myds-bench.conf" -m "$MODEL" --port $PORT --host 127.0.0.1 \
  --ctx-size 1048576 --alias bench --temp 0 --top-p 1.0 --max-tokens 8 --thinking off --vision "$VISION" >> "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 900); do curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
echo "$TAG health pid=$(lsof -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | awk NR==2{print }) mine=$PID" >> $ROOT/p32.log
curl -sf -m 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -H "X-Myds-Warmup: 1" -d @$ROOT/W.json -o /dev/null
curl -sf -m 2400 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @$OUT/req.json -o /dev/null -w "$TAG p32 total=%{time_total}\n" >> $ROOT/p32.log
kill -TERM $PID 2>/dev/null; for i in $(seq 1 90); do kill -0 $PID 2>/dev/null || break; sleep 1; done; kill -9 $PID 2>/dev/null
sleep 3
