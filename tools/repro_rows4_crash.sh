#!/bin/bash
. "$(dirname "$0")/prod_guard.sh"   # §164：退出时自动恢复生产 8187
# §151 复现：rows4 在长上下文上段错误。
# 每次追加唯一 nonce，避免 KV 磁盘快照命中（否则 prefill 不跑，复现不出）。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
BIN=/tmp/myds-new
D=$ROOT/crash151
rm -rf "$D"; mkdir -p "$D/metal"
cp /tmp/myds.metallib "$D/metal/myds.metallib"
cd "$D" || exit 1

PROD=$(pgrep -f "myfleet/myds.conf" | head -1)
[ -n "$PROD" ] && kill -TERM "$PROD"
for i in $(seq 1 90); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 3

for arm in 0 1; do
  for ctx in d2k d24k; do
    for rep in 1 2; do
      tag="r${arm}_${ctx}_r${rep}"
      out="$D/$tag"; mkdir -p "$out"
      python3 - "$ROOT/$ctx.json" "$out/req.json" "$tag" <<'PY'
import json, sys, random
d = json.load(open(sys.argv[1]))
d["messages"][0]["content"] += "\n\n[%s %d]" % (sys.argv[3], random.randint(10**8, 10**9))
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY
      MYDS_INDEXER_SCORE_ROWS4=$arm MYDS_ATTN_WHICH=1 "$BIN" \
        --config "$HOME/Projects/myfleet/myds.conf" -m "$MODEL" \
        --port $PORT --host 127.0.0.1 --ctx-size 1048576 --alias bench --temp 0 --top-p 1.0 \
        --max-tokens 128 --thinking off --vision "$VISION" > "$out/server.log" 2>&1 &
      pid=$!
      ok=0
      for i in $(seq 1 600); do
        curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 1
      done
      if [ $ok = 1 ]; then
        curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
          -H "Content-Type: application/json" -H "X-Myds-Warmup: 1" -d @"$ROOT/W.json" -o /dev/null
        curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
          -H "Content-Type: application/json" -d @"$out/req.json" -o "$out/resp.json" \
          -w "$tag %{time_total}\n"
      else
        echo "$tag START_FAILED"
      fi
      if kill -0 $pid 2>/dev/null; then
        kill -TERM $pid 2>/dev/null
        for i in $(seq 1 40); do kill -0 $pid 2>/dev/null || break; sleep 1; done
        kill -9 $pid 2>/dev/null
        echo "$tag STILL_ALIVE_AFTER_REQUEST"
      else
        wait $pid 2>/dev/null
        echo "$tag DIED rc=$?"
      fi
      sleep 4
    done
  done
done
echo CRASH151_DONE
