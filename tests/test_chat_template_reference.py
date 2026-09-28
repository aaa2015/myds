#!/usr/bin/env python3
"""Compare text-only Chat/Responses rendering to the official V4.1 encoder.

Download the fixed reference, then run:
  python tests/test_chat_template_reference.py /path/to/encoding.py \
    --renderer bin/test_server_api --output /tmp/chat-template-reference.json
Reference URL:
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/517ef625df97ec57aadc91b67506a57c20fdc5bb/encoding/encoding.py
The test executes the supplied local reference Python module; no model or GPU.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoding", type=Path)
    parser.add_argument("--renderer", type=Path, default=Path("bin/test_server_api"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("deepseek41_released_encoding", args.encoding)
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    conversations = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "system", "content": ""}, {"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "One"}, {"role": "user", "content": "Two"}],
        [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello"}, {"role": "system", "content": "Keep going."}],
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        [{"role": "assistant", "content": "Leading assistant."}],
        [{"role": "system", "content": "Only a system."}],
    ]
    rng = random.Random(20260929)
    texts = ["", "你好🙂", " A\nB\n ", "int x = 1;\n", '"quoted" \\ and {braces}', "é e\u0301", "Two\n\nparagraphs"]
    for _ in range(100):
        conversations.append([{"role": rng.choice(["system", "user", "assistant"]),
                               "content": rng.choice(texts)} for _ in range(rng.randint(1, 10))])
    results = []
    for messages in conversations:
        expected = encoding.encode_messages(messages, "chat", drop_thinking=True)
        responses = [{"type": "message", "role": m["role"], "content": [
            {"type": "input_text" if m["role"] != "assistant" else "output_text", "text": m["content"][:len(m["content"]) // 2]},
            {"type": "text", "text": m["content"][len(m["content"]) // 2:]},
        ]} for m in messages]
        bodies = [{"messages": messages}, {"input": responses}]
        if messages[0]["role"] == "system" and len(messages) > 1:
            bodies.append({"instructions": messages[0]["content"], "input": responses[1:]})
        if len(messages) == 1 and messages[0]["role"] == "user":
            bodies += [{key: messages[0]["content"]} for key in ("input", "prompt")]
        for body in bodies:
            result = subprocess.run([str(args.renderer.resolve()), "--render", json.dumps(body)], capture_output=True, check=True)
            actual = result.stdout.decode("utf-8")
            passed = actual == expected
            results.append({"body": body, "expected_prompt": expected, "actual_prompt": actual, "pass": passed})
            if not passed:
                print("MISMATCH", json.dumps(body, ensure_ascii=False), repr(expected), repr(actual))
    report = {"reference": "DeepSeek-V4.1-Flash encoding.encode_messages(thinking_mode='chat')",
              "source_revision": "517ef625df97ec57aadc91b67506a57c20fdc5bb",
              "encoding_py_sha256": hashlib.sha256(args.encoding.read_bytes()).hexdigest(),
              "cases": len(results), "passed": sum(r["pass"] for r in results), "results": results}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['passed']}/{report['cases']} exact official chat-template matches")
    return int(report["passed"] != report["cases"])


if __name__ == "__main__":
    raise SystemExit(main())
