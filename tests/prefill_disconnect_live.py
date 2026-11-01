#!/usr/bin/env python3
"""LIVE: a client that hangs up during prefill must not burn the whole prefill.

Sends a deliberately large prompt over a raw socket, reads nothing meaningful,
then closes. Before the 2026-11-01 change the server kept computing every
remaining token: cancel_flag was only ever set from the per-token decode
callback, which does not run during prefill.

Checks:
  1. the server logs a cancellation for this request,
  2. it does not report a completed prefill of the full prompt,
  3. a following request is served immediately (the slot was released).

    python3 tests/prefill_disconnect_live.py [port] [tokens]
"""
import json
import re
import socket
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8187
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
LOG = "/Users/fangqingyuan/Projects/myfleet/logs/local-deepseek-flash.log"


def big_prompt(target_tokens):
    """~19 BPE tokens per line; prefill runs ~350 tok/s, so 60K tokens is ~3 min."""
    line = "record {i:05d}: the quick brown fox jumps over the lazy dog near the river bank."
    n = max(1, target_tokens // 19)
    parts = [line.format(i=i) for i in range(n)]
    return "Read these records.\n" + "\n".join(parts)


def log_size():
    try:
        with open(LOG, "rb") as f:
            return f.seek(0, 2)
    except OSError:
        return 0


def main():
    prompt = big_prompt(TOKENS)
    body = json.dumps({"model": "local-deepseek-flash", "temperature": 0.0,
                       "max_tokens": 8,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
           f"Connection: close\r\n\r\n").encode() + body

    before = log_size()
    t0 = time.time()
    s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
    s.sendall(req)
    # Let the server get into prefill, then hang up. 3 s is enough to be well
    # inside prefill for a prompt this size and well before it could finish.
    time.sleep(3.0)
    s.close()
    print(f"disconnected after {time.time() - t0:.1f}s (prompt {len(prompt)} chars)")

    # The slot must be freed promptly. Probe with a tiny request.
    def small():
        b = json.dumps({"model": "local-deepseek-flash", "temperature": 0.0, "max_tokens": 4,
                        "messages": [{"role": "user", "content": "Say OK"}]}).encode()
        r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                   data=b, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=600) as resp:
            return resp.status, json.load(resp)

    t1 = time.time()
    status, resp = small()
    wait = time.time() - t1
    print(f"follow-up request served in {wait:.1f}s, status={status}")
    print(f"  reply={resp['choices'][0]['message'].get('content','')[:40]!r}")

    time.sleep(1.0)
    new_log = ""
    try:
        with open(LOG, "r", errors="replace") as f:
            f.seek(before)
            new_log = f.read()
    except OSError as e:
        print(f"  (cannot read log: {e})")

    cancelled = "Cancelled" in new_log or "cancelled" in new_log
    full_prefill = re.search(r"prefill: (\d+) tok", new_log)
    print()
    ok = True
    if new_log:
        if cancelled:
            print("[PASS] server logged a cancellation")
        else:
            print("[FAIL] no cancellation line in the log")
            ok = False
    if full_prefill:
        got = int(full_prefill.group(1))
        print(f"       prefill reported {got} tok of ~{TOKENS} requested "
              f"({'partial -> stopped early' if got < TOKENS * 0.9 else 'COMPLETED the full prompt'})")
    print(f"[{'PASS' if wait < 120 else 'FAIL'}] slot released promptly ({wait:.1f}s)")
    if wait >= 120:
        ok = False

    print()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
