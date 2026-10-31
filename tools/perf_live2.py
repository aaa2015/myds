#!/usr/bin/env python3
"""Live-server performance measurement, with the two traps removed.

Trap 1: decode over 24 tokens is noise. The model self-stops on short prompts, so a
short reply measures start-up jitter rather than steady-state decoding. This forces
long generation so there are hundreds of decode steps to average.

Trap 2: prefix cache turns repeat runs into "prefill: none (all N cached)". Repeating a
prompt therefore does not measure prefill at all -- it measures the cache. Every prefill
sample gets a unique nonce so the cache cannot hit, and cached hits are reported
separately as their own number rather than silently mixed in.

decode/prefill values are read from the server's own log lines, per the ledger rule;
HTTP wall is kept only as a cross-check, never as the reported number.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request

URL = "http://macstudio:8187/v1/chat/completions"
LOG = "/Users/fangqingyuan/Projects/myfleet/logs/local-deepseek-flash.log"
HOST = "macstudio"

DECODE_RE = re.compile(r"decode: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s")
PREFILL_RE = re.compile(r"prefill: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s")
CACHED_RE = re.compile(r"prefill: none \(all (\d+) cached\)")
PROMPT_RE = re.compile(r"prompt: (\d+) tok")
TTFT_RE = re.compile(r"ttft: ([\d.]+)s")

FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineering team reviews "
    "the pipeline, the latency budget, and the throughput target for the quarter. "
)


def ssh(cmd):
    return subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True, check=True).stdout


def log_lines():
    return int(ssh("wc -l < %s" % LOG).strip())


def new_completed(mark):
    out = ssh("tail -n +%d %s" % (mark + 1, LOG))
    return [l for l in out.splitlines() if "Completed" in l]


def parse(line):
    r = {"raw": line.strip()}
    m = DECODE_RE.search(line)
    if m:
        r.update(decode_tok=int(m.group(1)), decode_s=float(m.group(2)), decode_tps=float(m.group(3)))
    m = PREFILL_RE.search(line)
    if m:
        r.update(prefill_tok=int(m.group(1)), prefill_s=float(m.group(2)), prefill_tps=float(m.group(3)))
    m = CACHED_RE.search(line)
    if m:
        r.update(prefill_tok=0, prefill_s=0.0, prefill_tps=None, cached_tok=int(m.group(1)))
    m = PROMPT_RE.search(line)
    if m:
        r["prompt_tok"] = int(m.group(1))
    m = TTFT_RE.search(line)
    if m:
        r["ttft_s"] = float(m.group(1))
    return r


def post(body, timeout=3600):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as f:
        d = json.loads(f.read())
    return d, time.time() - t0


def run_once(prompt, max_tokens):
    mark = log_lines()
    d, wall = post({"model": "local-deepseek-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0, "max_tokens": max_tokens, "thinking": False})
    time.sleep(0.4)                      # let the log line land
    lines = new_completed(mark)
    rec = parse(lines[-1]) if lines else {}
    rec.update(http_wall=round(wall, 2),
               completion_tok=d["usage"].get("completion_tokens"),
               finish=d["choices"][0].get("finish_reason"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-sizes", default="500,2000,8000")
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/perf_live2.json")
    args = ap.parse_args()

    res = {"decode": [], "prefill": [], "cached": []}

    print("warmup (discarded)", flush=True)
    run_once("Reply with the single word: ready", 16)

    # ---- decode: one long generation per repeat, same prompt (decode is steady-state,
    # and re-using the cached prefix isolates decode from prefill) ----
    print("\nDECODE  max_tokens=%d" % args.decode_tokens, flush=True)
    dec_prompt = ("Write a detailed technical explanation of how a modern CPU pipeline "
                  "works, covering fetch, decode, execute, memory hierarchy and hazards.")
    for i in range(args.repeats):
        r = run_once(dec_prompt, args.decode_tokens)
        res["decode"].append(r)
        print("  rep=%d decode=%s tok in %ss = %s t/s | finish=%s" % (
            i, r.get("decode_tok"), r.get("decode_s"), r.get("decode_tps"), r.get("finish")),
            flush=True)

    # ---- prefill: unique nonce every call so the cache cannot hit ----
    print("\nPREFILL (unique prompt each run)", flush=True)
    for target in [int(x) for x in args.prefill_sizes.split(",")]:
        n = max(1, target // 34)
        for i in range(args.repeats):
            nonce = "run%d-%d-%d" % (target, i, int(time.time() * 1000) % 100000)
            prompt = "Note id %s. Summarise the following in one sentence. " % nonce + FILLER * n
            r = run_once(prompt, 32)
            kind = "cached" if r.get("prefill_tps") is None else "prefill"
            res["prefill" if kind == "prefill" else "cached"].append(r)
            print("  target=%-6d rep=%d %-8s prompt_tok=%-6s prefill=%-8s t/s  decode=%s t/s" % (
                target, i, kind, r.get("prompt_tok"),
                r.get("prefill_tps"), r.get("decode_tps")), flush=True)

    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    dt = [r["decode_tps"] for r in res["decode"] if r.get("decode_tps")]
    if dt:
        print("decode      median %.1f t/s  (n=%d, min %.1f max %.1f)" % (
            statistics.median(dt), len(dt), min(dt), max(dt)))
    for target in [int(x) for x in args.prefill_sizes.split(",")]:
        g = [r for r in res["prefill"] if r.get("prompt_tok") and target * 0.7 <= r["prompt_tok"] <= target * 1.4]
        if g:
            v = [r["prefill_tps"] for r in g]
            print("prefill %-6d median %.1f t/s  (n=%d)" % (target, statistics.median(v), len(v)))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
