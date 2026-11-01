#!/usr/bin/env python3
"""LIVE: previous_response_id must survive interleaved sessions.

Before the 2026-11-01 change the server kept exactly one live response, so this
sequence returned 400 on the third call -- deterministically, not occasionally:
A turn1 -> B turn1 -> A turn2 makes A's id stale the moment B finishes.

Needs the real model on a live port:
    python3 tests/previous_response_live.py
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get("MYDS_RESP_URL", "http://127.0.0.1:8187/v1/responses")
MODEL = os.environ.get("MYDS_RESP_MODEL", "local-deepseek-flash")


def post(payload, mt=64):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def text_of(resp):
    # Responses API: output[].content[].text ; tolerate a couple of shapes.
    out = []
    for item in resp.get("output") or []:
        for part in item.get("content") or []:
            t = part.get("text")
            if t:
                out.append(t)
    if not out and resp.get("output_text"):
        out.append(resp["output_text"])
    return "".join(out)


def turn(sid_label, input_text, prev=None):
    payload = {"model": MODEL, "input": input_text, "max_output_tokens": 64,
               "temperature": 0.0}
    if prev:
        payload["previous_response_id"] = prev
    status, resp = post(payload)
    rid = resp.get("id")
    print(f"  {sid_label:12s} status={status} id={rid} text={text_of(resp)[:60]!r}")
    return status, rid, text_of(resp)


failures = 0


def check(name, cond, detail=""):
    global failures
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} {detail}")
        failures += 1


print("=== interleaved sessions ===")
s1, a1, t1 = turn("A turn1", "My favourite colour is teal. Remember it. Reply with just OK.")
s2, b1, _ = turn("B turn1", "My favourite colour is crimson. Remember it. Reply with just OK.")

# The critical call: A continues using an id that is no longer the most recent.
s3, a2, t2 = turn("A turn2", "What is my favourite colour? One word.", prev=a1)
print()
check("A turn1 accepted", s1 == 200, f"status={s1}")
check("B turn1 accepted", s2 == 200, f"status={s2}")
check("A turn2 not rejected with 400", s3 != 400, f"status={s3}")
check("A turn2 recalls A's context", "teal" in t2.lower(), f"text={t2!r}")
check("A turn2 did NOT inherit B's context", "crimson" not in t2.lower(), f"text={t2!r}")

print()
print("=== unknown id still rejected ===")
s4, _, _ = turn("bogus", "hello", prev="resp_does_not_exist_12345")
check("unknown previous_response_id -> 400", s4 == 400, f"status={s4}")

print()
if failures:
    print(f"{failures} case(s) FAILED")
    sys.exit(1)
print("All previous_response_id cases passed.")
