#!/usr/bin/env python3
"""Compile the real HTTP handler with a deterministic generator; verify its wire bytes.

This is a transport test, not a tokenizer/model quality oracle. The normal C unit
suites remain separate. --sanitize additionally enables AddressSanitizer and UBSan.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def verify(output: Path) -> None:
    raw = b"A\x00" + "中😀".encode() + b"\xff\xe4\xb8" + b"\x01" * 8192 + b"\xf0\x9f"
    expected = raw.decode("utf-8", errors="replace")
    for responses in (0, 1):
        for stream in (0, 1):
            for fail in (0, 1):
                wire = (output / f"myds-http-{responses}-{stream}-{fail}.txt").read_bytes()
                header, body = wire.split(b"\r\n\r\n", 1)
                if stream:
                    assert header.startswith(b"HTTP/1.1 200 OK\r\n")
                    events = [
                        json.loads(line[6:])
                        for line in body.decode("utf-8", errors="strict").splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    ]
                    if fail:
                        assert any("error" in event for event in events)
                        assert not any(event.get("type") == "response.completed" for event in events)
                    else:
                        if responses:
                            text = "".join(event.get("delta", "") for event in events)
                            assert events[-1]["response"]["model"] == 'test-"alias'
                            assert events[-1]["response"]["usage"]["output_tokens"] == 14
                        else:
                            text = "".join(
                                event["choices"][0].get("delta", {}).get("content", "")
                                for event in events
                            )
                        assert text == expected
                else:
                    obj = json.loads(body.decode("utf-8", errors="strict"))
                    content_length = next(
                        line.split(b": ", 1)[1]
                        for line in header.split(b"\r\n")
                        if line.startswith(b"Content-Length:")
                    )
                    assert len(body) == int(content_length)
                    if fail:
                        assert header.startswith(b"HTTP/1.1 500 Internal Server Error\r\n")
                        assert "error" in obj
                    else:
                        assert header.startswith(b"HTTP/1.1 200 OK\r\n")
                        text = (
                            obj["output"][0]["content"][0]["text"]
                            if responses else obj["choices"][0]["message"]["content"]
                        )
                        assert text == expected and obj["model"] == 'test-"alias'
                        assert obj["usage"]["output_tokens" if responses else "completion_tokens"] == 14
                print(
                    f"PASS {'Responses' if responses else 'Chat Completions'} / "
                    f"{'SSE' if stream else 'JSON'} / {'failure' if fail else 'success'}"
                )


def run(output: Path, sanitize: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="myds-http-build-") as build:
        binary = Path(build) / "test_http_bytes"
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-O1", "-g", "-Wall", "-Wextra", "-std=c11", "-Iinclude",
        ]
        if sanitize:
            command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        command += ["tests/test_http_bytes.c", "-lpthread", "-o", str(binary)]
        print("COMPILE", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run([str(binary), str(output.resolve())], cwd=ROOT, check=True)
    verify(output)
    print("8 production HTTP cases passed strict UTF-8/JSON parsing and content/usage/error checks.")
    print("C assertions also verify default 4096 vs explicit 7000 limits and configuration restoration.")
    print("Sanitizers:", "AddressSanitizer + UBSan" if sanitize else "disabled (ordinary integration run)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true", help="Enable AddressSanitizer and UBSan")
    parser.add_argument("--output-dir", type=Path, help="Keep raw HTTP responses in this directory")
    args = parser.parse_args()
    if args.output_dir:
        run(args.output_dir, args.sanitize)
    else:
        with tempfile.TemporaryDirectory(prefix="myds-http-output-") as output:
            run(Path(output), args.sanitize)


if __name__ == "__main__":
    main()
