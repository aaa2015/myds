#!/usr/bin/env python3
"""perf_client.py — 测 decode 速度（TTFT + tok/s），独立于 shell。

用法: perf_client.py <port> [runs] [out_json]

3 个不同 prompt，每个先预热一次丢弃，再测 runs 次。
专家缓存会让重复 prompt 显得更快，所以预热与测量都用同一批不同 prompt。
"""
import json
import statistics as st
import sys
import time
import urllib.request

port = int(sys.argv[1])
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = sys.argv[3] if len(sys.argv) > 3 else None
maxtok = 128

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. Write a detailed technical explanation of how a compiler translates source code into machine instructions.",
    "Explain the difference between a hash table and a balanced binary search tree, including time complexity for lookup, insertion and deletion.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
]


def one(prompt):
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=900) as r:
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


res = []
for pi, p in enumerate(PROMPTS):
    one(p)                                  # 预热，丢弃
    for r in range(runs):
        m = one(p)
        m.update({"prompt_idx": pi, "run": r})
        res.append(m)
        print(f"  p{pi} r{r}: ttft={m['ttft_s']}s decode={m['decode_tok_s']} tok/s chunks={m['chunks']}", flush=True)

ok = [m for m in res if m["decode_tok_s"]]
if ok:
    d = sorted(m["decode_tok_s"] for m in ok)
    t = sorted(m["ttft_s"] for m in ok if m["ttft_s"])
    summary = {
        "tag": sys.argv[3] if out else "run",
        "samples": len(ok),
        "decode_tok_s_median": st.median(d),
        "decode_tok_s_min": d[0],
        "decode_tok_s_max": d[-1],
        "ttft_s_median": st.median(t) if t else None,
    }
    print(json.dumps(summary, indent=2))
    if out:
        json.dump({"summary": summary, "samples": res}, open(out, "w"), indent=2)
else:
    print("  ❌ 没有有效样本")
    sys.exit(1)
