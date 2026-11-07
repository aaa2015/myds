#!/bin/bash
# 生产级 A/B：旧二进制（adeb1f4e，rows4 关）vs 新二进制（e6d2a1a0，rows4 auto）
# 28.4K prompt + 200 token 输出，交替各 2 轮。这是 rows4 真正生效的唯一口径。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
D=$ROOT/prod_ab
rm -rf "$D"; mkdir -p "$D"
cd "$D" || exit 1

PROD=$(pgrep -f "myfleet/myds.conf" | head -1)
[ -n "$PROD" ] && kill -TERM "$PROD"
for i in $(seq 1 120); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 3

run_one() {
  local tag=$1 bin=$2 conf=$3 rep=$4
  local out="$D/${tag}_$rep"; mkdir -p "$out/metal"
  # metallib 必须放在 <cwd>/metal/ 下（服务按 cwd 相对路径找）
  cp "$(dirname "$bin")/metal/myds.metallib" "$out/metal/myds.metallib" || {
      echo "$tag r$rep NO_METALLIB"; return 1; }
  cd "$out" || return 1
  "$bin" --config "$conf" -m "$MODEL" --port $PORT --host 127.0.0.1 --ctx-size 1048576 \
    --alias bench --temp 0 --top-p 1.0 --max-tokens 200 --thinking off --vision "$VISION" \
    > server.log 2>&1 &
  local pid=$!; local ok=0
  for i in $(seq 1 600); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 1
  done
  if [ $ok = 1 ]; then
    python3 - "$ROOT/d24k.json" "$out/req.json" "$tag$rep" <<'PY'
import json, sys, random
d = json.load(open(sys.argv[1])); d["max_tokens"] = 200
d["messages"][0]["content"] += "\n\n[ab %s %d]" % (sys.argv[3], random.randint(10**8, 10**9))
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY
    curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" --data-binary @"$out/req.json" -o /dev/null \
      -w "$tag r$rep %{time_total}\n"
  else
    echo "$tag r$rep START_FAILED"
  fi
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null
  sleep 4
}

CONF_OFF="$HOME/bench/ab/conf-rows4off.conf"
CONF_AUTO="$HOME/bench/ab/conf-auto.conf"

for rep in 1 2; do
  run_one old /tmp/oldbin/myds          "$CONF_OFF"  "$rep"
  run_one new /tmp/dep3/myds            "$CONF_AUTO" "$rep"
done

for d in "$D"/old_1 "$D"/old_2 "$D"/new_1 "$D"/new_2; do
  [ -f "$d/server.log" ] || continue
  v=$(grep -oE "decode: 200 tok in [0-9.]+s = [0-9.]+ t/s" "$d/server.log" | tail -1)
  echo "$(basename "$d"): $v"
done
echo PROD_AB_DONE
