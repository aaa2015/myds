#!/bin/bash
. "$(dirname "$0")/prod_guard.sh"   # §164：退出时自动恢复生产 8187
# §152 补测 + §154 验收：rows4 的 auto 阈值。
#   auto（默认）: n_comp >= 8192 才用 rows4
#   ref        : 强制 0（原 1-row smem）
#   forced     : 强制 1（全开，用于确认真的会退化）
# 三臂各 2 轮，短(2.1K) 与长(28.4K) 上下文。同时验 0-ULP（同 prompt）。
set -u
ROOT=$HOME/bench/ab
PORT=8187
MODEL=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf
VISION=$HOME/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf
BIN=/tmp/dep2/myds
D=$ROOT/acc154
rm -rf "$D"; mkdir -p "$D/metal"
cp /tmp/dep2/myds.metallib "$D/metal/myds.metallib" 2>/dev/null || cp ~/Projects/myds/bin/myds.metallib "$D/metal/myds.metallib"
cd "$D" || exit 1

PROD=$(pgrep -f "myfleet/myds.conf" | head -1)
[ -n "$PROD" ] && kill -TERM "$PROD"
for i in $(seq 1 120); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 3

run_arm() {
  local tag=$1 mode=$2 ctx=$3 rep=$4 nonce=$5
  local out="$D/${tag}_${ctx}_$rep"; mkdir -p "$out"
  python3 - "$ROOT/$ctx.json" "$out/req.json" "$nonce" <<'PY'
import json, sys, random
d = json.load(open(sys.argv[1]))
if sys.argv[3] == "1":
    d["messages"][0]["content"] += "\n\n[x %d]" % random.randint(10**8, 10**9)
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY
  MYDS_INDEXER_SCORE_ROWS4=$mode MYDS_ATTN_WHICH=1 "$BIN" \
    --config "$HOME/Projects/myfleet/myds.conf" -m "$MODEL" \
    --port $PORT --host 127.0.0.1 --ctx-size 1048576 --alias bench --temp 0 --top-p 1.0 \
    --max-tokens 200 --thinking off --vision "$VISION" > "$out/server.log" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 600); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  if [ $ok = 1 ]; then
    curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" -H "X-Myds-Warmup: 1" -d @"$ROOT/W.json" -o /dev/null
    curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" --data-binary @"$out/req.json" -o "$out/resp.json" \
      -w "$tag $ctx r$rep %{time_total}\n"
  else
    echo "$tag $ctx r$rep START_FAILED"
  fi
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null
  sleep 3
}

# 性能：自动档 vs 强制关，短长各 2 轮（加 nonce 保证冷 prefill）
for rep in 1 2; do
  run_arm ref  auto d2k  "$rep" 1
  run_arm auto auto d2k  "$rep" 1
  run_arm ref  auto d24k "$rep" 1
  run_arm auto auto d24k "$rep" 1
done
# 0-ULP：同一 prompt（无 nonce），auto vs 强制关
run_arm ulp_ref  0    d2k  9 0
run_arm ulp_auto auto d2k  9 0
run_arm ulp_ref  0    d24k 9 0
run_arm ulp_auto auto d24k 9 0

python3 - "$D" <<'PY'
import json, hashlib, os, sys
D = sys.argv[1]
for ctx in ("d2k", "d24k"):
    hs = {}
    for arm in ("ulp_ref", "ulp_auto"):
        p = os.path.join(D, "%s_%s_9" % (arm, ctx), "resp.json")
        if not os.path.exists(p): hs[arm] = "MISSING"; continue
        try:
            t = json.load(open(p))["choices"][0]["message"]["content"]
            hs[arm] = hashlib.sha256(t.encode()).hexdigest()[:20]
        except Exception as e:
            hs[arm] = "ERR"
    print("ULP %-5s ref=%s auto=%s  %s" % (ctx, hs["ulp_ref"], hs["ulp_auto"],
          "IDENTICAL" if hs["ulp_ref"] == hs["ulp_auto"] != "MISSING" else "DIFFERENT"))
PY
echo ACC154_DONE
