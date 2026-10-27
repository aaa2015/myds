#!/usr/bin/env python3
"""Prefill/decode sweep against a running myds server, using the frozen corpus.

Every prompt is a distinct corpus file, so `cached` must read 0 for every row;
anything else is recorded and excluded, because a cached prompt performed no
prefill and its t/s is meaningless.

Two measurements per size: server-side prefill (from myds' own log line) and
client-side wall. They are kept side by side so a reader can see whether the
two clocks agree instead of having to trust one.

Usage:
    python3 tools/bench_register_only_sweep.py --out /tmp/sweep_run.jsonl
    python3 tools/verify_register_only_bench.py --sweep /tmp/sweep_run.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "benchmarks" / "register_only_2026-10-27" / "prompts"

PREFILL_CASES = ["p0.2k", "p1.6k", "p6.5k", "p13k", "p26k", "p52k"]
# decode cases need enough generated tokens for the rate to mean anything; the
# earlier run used a prompt that asked for one word and produced 2 tokens.
DECODE_CASES = [("d_short", 256), ("d_8k", 160), ("d_32k", 160)]
REPS = 3


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def emit(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ask(base, alias, prompt, max_tokens, timeout=1800):
    payload = {"model": alias, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def server_line(logfile, offset):
    with open(logfile, "rb") as f:
        f.seek(offset)
        txt = f.read().decode("utf-8", "replace")
    best = None
    for line in txt.splitlines():
        if "Completed ChatCompletions" not in line:
            continue
        m = re.search(r"prompt: (\d+) tok, prefill: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s "
                      r"\(cached (\d+)\), .*?decode: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s", line)
        if m:
            best = dict(prompt_tok=int(m.group(1)), prefill_tok=int(m.group(2)),
                        prefill_s=float(m.group(3)), prefill_tps=float(m.group(4)),
                        cached=int(m.group(5)), decode_tok=int(m.group(6)),
                        decode_s=float(m.group(7)), decode_tps=float(m.group(8)))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8187")
    ap.add_argument("--alias", default="local-deepseek-flash")
    ap.add_argument("--log", default=str(Path.home() /
                                         "Projects/myfleet/logs/local-deepseek-flash.log"))
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()

    if not CORPUS.exists():
        sys.exit(f"corpus missing: {CORPUS}\nrun tools/make_register_only_corpus.py first")
    try:
        with urllib.request.urlopen(a.base + "/health", timeout=5) as r:
            if r.status != 200:
                sys.exit("server is not healthy")
    except Exception as e:
        sys.exit(f"cannot reach {a.base}: {e}")

    open(a.out, "w").close()
    log(f"sweeping {a.base}, {a.reps} reps per size")

    for case in PREFILL_CASES:
        for rep in range(a.reps):
            prompt = (CORPUS / f"{case}__v{rep}.txt").read_text()
            off = os.path.getsize(a.log)
            d, wall = ask(a.base, a.alias, prompt, 2)
            time.sleep(1.6)          # let the log line flush before reading it
            s = server_line(a.log, off)
            emit(a.out, dict(phase="sweep", case=case, rep=rep, wall_s=round(wall, 3),
                             usage_prompt=d["usage"]["prompt_tokens"],
                             usage_completion=d["usage"]["completion_tokens"], server=s))
            log(f"  {case} r{rep}: " + (f"prefill {s['prefill_s']}s = {s['prefill_tps']} t/s "
                                        f"cached={s['cached']}" if s else "NO SERVER LINE"))
            time.sleep(0.8)

    for case, mx in DECODE_CASES:
        for rep in range(a.reps):
            prompt = (CORPUS / f"{case}__v{rep}.txt").read_text()
            off = os.path.getsize(a.log)
            d, wall = ask(a.base, a.alias, prompt, mx)
            time.sleep(1.6)
            s = server_line(a.log, off)
            emit(a.out, dict(phase="sweep", case=case, rep=rep, wall_s=round(wall, 3),
                             usage_prompt=d["usage"]["prompt_tokens"],
                             usage_completion=d["usage"]["completion_tokens"], server=s))
            log(f"  {case} r{rep}: " + (f"prefill {s['prefill_tps']} t/s | decode "
                                        f"{s['decode_tok']} in {s['decode_s']}s = "
                                        f"{s['decode_tps']} t/s" if s else "NO SERVER LINE"))
            time.sleep(0.8)

    emit(a.out, dict(phase="done"))
    log("done")
    print(f"\nwrote {a.out}")
    print("now run: python3 tools/verify_register_only_bench.py --sweep " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
