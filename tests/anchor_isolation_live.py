#!/usr/bin/env python3
"""LIVE (needs the model): anchor isolation across interleaved sessions.

Not part of `make test` -- it needs a running server with the real V4.1 model.
Run against a live port:

    python3 tests/anchor_isolation_live.py seedA seedB backToA secondTurn

What it proves, in order:
  seedA   session A with a ~14.7K system prompt answers from A's records
  seedB   session B (different system prompt) answers from B's
  backToA A asks again AFTER B overwrote the resident compressed KV.
          Before the 2026-11-01 fix this returned B's answer
          ('branch-BRAVO-42') with finish_reason "stop" and no log line.
  secondTurn  same system prompt, different question -> still answers correctly

Why the system prompt must exceed the ring capacity (11264): with a shorter
one a ring snapshot always covers the anchor, so the anchor code never decides
anything and the bug stays hidden.
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8187/v1/chat/completions"


def filler(tag, n):
    return "\n".join(f"{tag} record {i:04d}: value={i*7%97} note=branch-{tag}-{i}" for i in range(n))


HEAD = "You are an assistant. Read the records below carefully.\n"
SYS_A = HEAD + filler("ALPHA", 700)      # ~14.7K tokens, beyond the 11264 ring
SYS_B = HEAD + filler("BRAVO", 700)
Q_NOTE = "What is the note value of ALPHA record 0042? Answer with only the note string."
Q_TWO = "What is the note value of ALPHA record 0007? Answer with only the note string."


def call(sysmsg, q, mt=24):
    body = json.dumps({"model": "local-deepseek-flash", "temperature": 0.0, "max_tokens": mt,
                       "messages": [{"role": "system", "content": sysmsg},
                                    {"role": "user", "content": q}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", d["choices"][0].get("finish_reason")


for phase in sys.argv[1:]:
    if phase == "seedA":
        c, f = call(SYS_A, Q_NOTE)
        print(f"A#1        {c!r} finish={f}")
    elif phase == "seedB":
        c, f = call(SYS_B, Q_NOTE)
        print(f"B#1        {c!r} finish={f}")
    elif phase == "backToA":
        c, f = call(SYS_A, Q_NOTE)
        ok = c.strip() == "branch-ALPHA-42"
        print(f"A#2        {c!r} finish={f}")
        print("ISOLATION:", "PASS" if ok else f"FAIL ({c!r})")
    elif phase == "secondTurn":
        c, f = call(SYS_A, Q_TWO)
        ok = c.strip() == "branch-ALPHA-7"
        print(f"A#3        {c!r} finish={f}")
        print("CONTENT:", "PASS" if ok else f"FAIL ({c!r})")
