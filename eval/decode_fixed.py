#!/usr/bin/env python3
"""decode_fixed.py — 固定 128 token 的 decode 基准（撞上限，避免 EOS 抖动）。

用法: decode_fixed.py <port> [alias]

· max_tokens=128 且提示词设计为不会提前结束
· temperature=0
· 每个 prompt 预热一次丢弃，再测 3 次
· 只统计 chunks >= 120 的样本（撞上限才算有效）
"""
import json
import statistics as st
import sys
import time
import urllib.request

port = int(sys.argv[1])
alias = sys.argv[2] if len(sys.argv) > 2 else "local-myds-deepseek-v4.1-flash-q2"
N = 128

PROMPTS = [
    "Write a long detailed technical explanation of how a modern CPU pipeline works, covering fetch, decode, execute, memory access and writeback stages.",
    "Explain in detail the differences between B-trees and LSM-trees for storage engines, including write amplification and read paths.",
    "Describe the complete lifecycle of an HTTP request from DNS resolution through TLS handshake to the server returning a response.",
]


def once(prompt):
    body = json.dumps({
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": N, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    chunks = 0
    fin = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            ch = d.get("choices") or []
            if not ch:
                continue
            if ch[0].get("finish_reason"):
                fin = ch[0]["finish_reason"]
            delta = ch[0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if ttft is None:
                    ttft = time.monotonic() - t0
                chunks += 1
    total = time.monotonic() - t0
    dec = total - (ttft or 0)
    return {"ttft": ttft, "chunks": chunks, "finish": fin,
            "tok_s": (chunks / dec) if dec > 0 and chunks else None}


rows = []
for pi, p in enumerate(PROMPTS):
    once(p)                      # 预热
    for r in range(3):
        m = once(p)
        ok = (m["chunks"] or 0) >= 120
        rows.append((pi, m, ok))
        print(f"  p{pi} r{r}: {m['chunks']} chunks  {m['tok_s']:.2f} tok/s  "
              f"finish={m['finish']}  {'✓' if ok else '✗样本不足'}", flush=True)

good = [m["tok_s"] for _, m, ok in rows if ok]
if good:
    print(f"  ── 有效样本 {len(good)}/{len(rows)}  median={st.median(good):.2f} tok/s "
          f"({1000/st.median(good):.2f} ms/token) ──")
else:
    print("  ── 无有效样本 ──")
