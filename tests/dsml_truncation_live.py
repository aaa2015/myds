#!/usr/bin/env python3
"""LIVE: a truncated DSML tool block must not leak into `content`.

Before 2026-11-01, when the model started a <|DSML| calls> block and ran out of
max_tokens before closing it, the parser returned false and the server fell back
to "the whole thing is prose" -- so the client received raw XML in the message
body. Anything downstream that parses tool calls sees garbage.

Needs the real model:
    python3 tests/dsml_truncation_live.py [port] [max_tokens]
"""
import json
import os
import sys
import urllib.error
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8187
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 12
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = os.environ.get("MYDS_DSML_MODEL", "local-deepseek-flash")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

PROMPT = ("Use the get_weather tool to check the weather in San Francisco, "
          "Beijing and Tokyo. Call the tool for all three.")


def post(payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main():
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {'' if cond else detail}")
        if not cond:
            failures += 1

    for stream in (0, 1):
        print(f"=== stream={stream} ===")
        payload = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                   "tools": TOOLS, "temperature": 0.0, "max_tokens": MAXTOK,
                   "stream": bool(stream)}
        if stream:
            # SSE is not JSON; read the body as text instead of json.load().
            raw = json.dumps(payload).encode()
            req = urllib.request.Request(URL, data=raw,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    text = r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")
                print(f"  request failed: {e.code} {text[:200]}")
                failures += 1
                continue
            content = ""
            finish = None
            for line in text.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        content += delta["content"]
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        else:
            status, resp = post(payload)
            if status != 200:
                print(f"  request failed: {status} {resp}")
                failures += 1
                continue
            ch = (resp.get("choices") or [{}])[0]
            content = (ch.get("message") or {}).get("content") or ""
            finish = ch.get("finish_reason")

        # Any tool-call markup in the prose is a leak -- not just the DSML spelling.
        # The 1017 build has been observed emitting the plain <function_calls> form,
        # which the parser does not recognise at all (see the note in the docstring).
        leaked = any(m in content for m in
                     ("<|DSML|", "｜DSML｜", "<function_calls>", "<invoke ", "<parameter "))
        print(f"  finish_reason={finish} content={content[:120]!r}")
        check("no raw DSML markup in content", not leaked, f"content={content[:200]!r}")
        check("truncation is reported, not hidden",
              finish == "length" or finish == "tool_calls" or leaked is False,
              f"finish={finish}")

    print()
    print("OK" if failures == 0 else f"{failures} case(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
