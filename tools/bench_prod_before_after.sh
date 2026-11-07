#!/bin/bash
# 部署前后同口径对比：32K 冷 prefill + decode（各 3 次 decode、2 次 prefill）
# 用法: bench_prod.sh <tag>
set -u
ROOT=$HOME/bench/ab
PORT=8187
TAG=$1
D=$ROOT/prod_$TAG
rm -rf "$D"; mkdir -p "$D"
cd "$D" || exit 1

# 32K 冷 prefill 两次（唯一 nonce），decode 三次（同一 prompt，缓存命中后只测 decode）
for r in 1 2; do
  python3 - "$ROOT/p32k_v0.json" "$D/pf.json" "$TAG$r" <<'PY'
import json, sys, random
d = json.load(open(sys.argv[1])); d["max_tokens"] = 8
d["messages"][0]["content"] += "\n\n[%s %d]" % (sys.argv[3], random.randint(10**8, 10**9))
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY
  curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" --data-binary @"$D/pf.json" -o /dev/null \
    -w "$TAG prefill$r %{time_total}\n"
done

for r in 1 2 3; do
  curl -sf -m 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" --data-binary @"$ROOT/dec256.json" -o /dev/null \
    -w "$TAG decode$r %{time_total}\n"
done
echo "$TAG PROD_BENCH_DONE"
