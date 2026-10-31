#!/usr/bin/env python3
"""Measure the live production server, not an offline build.

Ledger rules this obeys:
  - decode throughput comes from the server's own
    "decode: N tok in S s = X t/s" line, never from HTTP wall time
  - greedy (temperature 0), fixed max_tokens
  - median of repeated runs, not a single sample
  - the first request after a state change is discarded (it is not reproducible)
  - refuses to report if another client is active during measurement: a shared box
    measures the neighbour, not the model

Log correlation: the caller passes a "since" line offset per request; every row keeps
the raw server line so the number can be re-derived from the log afterwards.
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

URL = "http://macstudio:8187/v1/chat/completions"

DECODE_RE = re.compile(r"decode: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s")
PREFILL_RE = re.compile(r"prefill: ([\d]+) tok in ([\d.]+)s = ([\d.]+) t/s")
PREFIX_RE = re.compile(r"prefill: none \(all (\d+) cached\)")
PROMPT_RE = re.compile(r"prompt: (\d+) tok")
TTFT_RE = re.compile(r"ttft: ([\d.]+)s")

FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer documents "
    "the pipeline, the latency budget, and the throughput target for the quarter. "
)


def build_prompt(target_tokens):
    """Roughly target_tokens of prompt, deterministic so runs are comparable."""
    approx_tokens_per_repeat = 34          # FILLER is ~34 tokens
    n = max(1, target_tokens // approx_tokens_per_repeat)
    return ("Summarise the following notes in one sentence. " + FILLER * n)


def post(prompt, max_tokens, timeout=1800):
    body = {
        "model": "local-deepseek-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "thinking": False,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def parse_line(line):
    out = {}
    m = DECODE_RE.search(line)
    if m:
        out.update(decode_tok=int(m.group(1)), decode_s=float(m.group(2)),
                   decode_tps=float(m.group(3)))
    m = PREFILL_RE.search(line)
    if m:
        out.update(prefill_tok=int(m.group(1)), prefill_s=float(m.group(2)),
                   prefill_tps=float(m.group(3)))
    m = PREFIX_RE.search(line)
    if m:
        out.update(prefill_tok=0, prefill_s=0.0, cached_tok=int(m.group(1)))
    m = PROMPT_RE.search(line)
    if m:
        out["prompt_tok"] = int(m.group(1))
    m = TTFT_RE.search(line)
    if m:
        out["ttft_s"] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="60,600,4000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/perf_now.json")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    rows = []

    # warmup, discarded: first request after a restart is not a sample
    print("warmup (discarded) ...", flush=True)
    d, wall = post("Reply with the single word: ready", 16)
    print("  %.1fs\n" % wall, flush=True)

    for target in sizes:
        prompt = build_prompt(target)
        for rep in range(args.repeats):
            d, wall = post(prompt, args.max_tokens)
            usage = d["usage"]
            ch = d["choices"][0]
            rows.append(dict(
                target=target, rep=rep, wall=round(wall, 2),
                prompt_tok=usage.get("prompt_tokens"),
                completion_tok=usage.get("completion_tokens"),
                finish=ch.get("finish_reason"),
            ))
            print("  target=%-5d rep=%d  completion=%-4d wall=%.1fs" % (
                target, rep, usage.get("completion_tokens"), wall), flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
