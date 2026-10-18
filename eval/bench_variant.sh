#!/bin/zsh
# =================================================================
# bench_variant.sh — 量一个权重变体的 decode 性能
#
#   ./bench_variant.sh <tag> <model.gguf> [runs] [max_tokens]
#
# 测量口径（固定、可复现）：
#   · 直接起临时 myds，端口 8199，ctx 8192，不占生产端口
#   · 3 个不同 prompt（避免专家缓存预热造成的假加速）
#   · temperature=0，max_tokens 固定，流式请求
#   · 记录：TTFT、decode tok/s（首 token 到末 token）、总时长
#   · 每个 prompt 先跑一次丢弃（载入缓存），再跑一次计入
#
# 为什么不用 bench_phase：它测的是 prefill 网格，且要求 MYDS_PREFIX_CACHE=0；
# 这里要的是「服务实际跑起来后，用户感受到的 decode 速度」。
# =================================================================
set -u

TAG="${1:?usage: bench_variant.sh <tag> <model.gguf> [runs] [max_tokens]}"
MODEL="${2:?missing model}"
RUNS="${3:-2}"
MAXTOK="${4:-128}"
PORT=8199
MYDS_DIR="$HOME/Projects/myds"
OUT="$HOME/myds-eval/perf-$TAG.json"
LOGDIR="$HOME/myds-eval"
PID=""

cleanup() {
    if [[ -n "$PID" && "$PID" == <-> ]]; then
        kill "$PID" 2>/dev/null
        for i in $(seq 1 40); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
    fi
}
trap 'cleanup' EXIT INT TERM

# 生产必须已停（362 GB 模型，192 GB 机器装不下两份）
if lsof -nP -t -iTCP:8181 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "❌ 8181 仍在监听 —— 先停生产再测（内存不够两份模型）"
    exit 1
fi

cd "$MYDS_DIR" || exit 1
echo "── 起 $TAG 于 :$PORT ──"
./bin/myds -m "$MODEL" --port $PORT --host 127.0.0.1 \
    --ctx-size 8192 --temp 0 --top-p 1.0 --max-tokens 4096 --thinking off \
    > "$LOGDIR/perf-$TAG.log" 2>&1 &
PID=$!
echo "  pid=$PID"

for i in $(seq 1 240); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break
    kill -0 "$PID" 2>/dev/null || { echo "❌ 服务进程退出"; tail -5 "$LOGDIR/perf-$TAG.log"; exit 1; }
    sleep 1
done
curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok || { echo "❌ 健康检查超时"; exit 1; }
echo "  服务就绪"

PROMPTS=(
"The quick brown fox jumps over the lazy dog. Write a detailed technical explanation of how a compiler translates source code into machine instructions."
"Explain the difference between a hash table and a balanced binary search tree, including time complexity for lookup, insertion and deletion."
"Write a short story about a lighthouse keeper who discovers a message in a bottle."
)

python3 - "$TAG" "$RUNS" "$MAXTOK" "$OUT" "${PROMPTS[@]}" <<'PY'
import json, sys, time, urllib.request

tag, runs, maxtok, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
prompts = sys.argv[5:]
PORT = 8199
res = []

def one(prompt, maxtok):
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if ttft is None:
                    ttft = time.monotonic() - t0
                chunks += 1
    total = time.monotonic() - t0
    dec = total - (ttft or 0)
    return {
        "ttft_s": round(ttft, 3) if ttft else None,
        "decode_s": round(dec, 3),
        "chunks": chunks,
        "total_s": round(total, 3),
        "decode_tok_s": round(chunks / dec, 2) if dec > 0 and chunks else None,
    }

for pi, p in enumerate(prompts):
    one(p, 16)                       # 预热：让专家缓存热起来，结果丢弃
    for r in range(runs):
        m = one(p, maxtok)
        m.update({"prompt_idx": pi, "run": r})
        res.append(m)
        print(f"  p{pi} r{r}: ttft={m['ttft_s']}s decode={m['decode_tok_s']} tok/s chunks={m['chunks']}")

ok = [m for m in res if m["decode_tok_s"]]
if ok:
    import statistics as st
    d = sorted(m["decode_tok_s"] for m in ok)
    t = sorted(m["ttft_s"] for m in ok if m["ttft_s"])
    summary = {
        "tag": tag,
        "samples": len(ok),
        "decode_tok_s_median": st.median(d),
        "decode_tok_s_min": d[0],
        "decode_tok_s_max": d[-1],
        "ttft_s_median": st.median(t) if t else None,
    }
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "samples": res}, open(out, "w"), indent=2)
PY
echo "  写出 $OUT"
