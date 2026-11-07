#!/bin/bash
# §151 第 2 道门：同 prompt、temp 0，两臂逐字节比对。
# 与性能 A/B 分开：那边每次加 nonce 保证冷 prefill，会把 prompt 改掉。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
BIN=/tmp/myds-new
D=$ROOT/ulp151
rm -rf "$D"; mkdir -p "$D/metal"
cp /tmp/myds.metallib "$D/metal/myds.metallib"
cd "$D" || exit 1

PROD=$(pgrep -f "myfleet/myds.conf" | head -1)
[ -n "$PROD" ] && kill -TERM "$PROD"
for i in $(seq 1 90); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 3

for arm in 0 1; do
  for ctx in d2k d24k; do
    out="$D/a${arm}_${ctx}"; mkdir -p "$out"
    MYDS_INDEXER_SCORE_ROWS4=$arm "$BIN" \
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
        -H "Content-Type: application/json" -d @"$ROOT/$ctx.json" -o "$out/resp.json" \
        -w "a$arm $ctx %{time_total}\n"
    else
      echo "a$arm $ctx START_FAILED"
    fi
    kill -TERM $pid 2>/dev/null
    for i in $(seq 1 40); do kill -0 $pid 2>/dev/null || break; sleep 1; done
    kill -9 $pid 2>/dev/null
    sleep 4
  done
done

python3 - "$D" <<'PY'
import json, hashlib, sys, os
D = sys.argv[1]
for ctx in ("d2k", "d24k"):
    row = []
    for arm in ("0", "1"):
        p = os.path.join(D, "a%s_%s" % (arm, ctx), "resp.json")
        if not os.path.exists(p):
            row.append((arm, "MISSING", 0)); continue
        try:
            t = json.load(open(p))["choices"][0]["message"]["content"]
            row.append((arm, hashlib.sha256(t.encode()).hexdigest()[:20], len(t)))
        except Exception as e:
            row.append((arm, "ERR:%s" % e, 0))
    same = row[0][1] == row[1][1] and row[0][1] != "MISSING"
    print("%-5s ref=%s(%d) rows4=%s(%d)  %s" % (ctx, row[0][1], row[0][2], row[1][1], row[1][2],
                                                "IDENTICAL" if same else "DIFFERENT"))
PY
echo ULP151_DONE
