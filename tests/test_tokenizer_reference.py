#!/usr/bin/env python3
"""Independent metadata-only differential check against Hugging Face tokenizers.

Usage: python tests/test_tokenizer_reference.py /path/to/tokenizer.json \
          --probe ./bin/test_tokenizer --output /tmp/tokenizer-reference.json
Install `tokenizers` into a separate virtual environment. The JSON is from
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/517ef625df97ec57aadc91b67506a57c20fdc5bb/tokenizer.json
No model weights are downloaded or evaluated. --gguf can check the deployed
GGUF's metadata instead of the temporary GGUF generated from the JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile

import tokenizers
from tokenizers import Tokenizer


def write_metadata(path, doc):
    vocab = doc["model"]["vocab"]
    tokens = [""] * (max(max(vocab.values()), max(t["id"] for t in doc["added_tokens"])) + 1)
    types = [1] * len(tokens)
    for text, idx in vocab.items():
        tokens[idx] = text
    for token in doc["added_tokens"]:
        tokens[token["id"]] = token["content"]
        types[token["id"]] = 3 if token["special"] else 4
    merges = [" ".join(m) if isinstance(m, list) else m for m in doc["model"]["merges"]]

    def string(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    fields = [
        ("tokenizer.ggml.model", 8, string("gpt2")),
        ("tokenizer.ggml.pre", 8, string("joyai-llm")),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, len(tokens)) + b"".join(map(string, tokens))),
        ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, len(merges)) + b"".join(map(string, merges))),
        ("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, len(types)) + struct.pack(f"<{len(types)}i", *types)),
        ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 0)),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 1)),
        ("tokenizer.ggml.padding_token_id", 4, struct.pack("<I", 2)),
    ]
    data = struct.pack("<IIQQ", 0x46554747, 3, 0, len(fields))
    for key, kind, value in fields:
        data += string(key) + struct.pack("<I", kind) + value
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("--probe", type=Path, default=Path("bin/test_tokenizer"))
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.tokenizer_json.read_bytes()
    reference = Tokenizer.from_file(str(args.tokenizer_json))
    corpus = [
        "", "a b", "The capital of France is", "What is the capital of France?",
        "中文", "你好，world!🙂", "中文English混合1234日本語",
        "def foo():\n    return 12345;\n", "abc1234def", "  a  b  ",
        "\t\r\n  Hello\r\nworld\n\n", "café e\u0301 Δοκιμή Привет ٣٤٥٦ Ⅻ²",
        "👩‍💻🏳️‍🌈 你好！。", "\u00a0\u2003\u3000x\u2028y", "\x01\x7f\u0080",
        "<｜begin▁of▁sentence｜><｜User｜>你好<｜Assistant｜>",
        "!hello 'world I'm we've CAN'T 9876543210",
        "a" * 4096,
    ]
    # Seeded mixed-script prompts expose pre-tokenizer boundary mistakes that
    # ASCII-only and encode/decode self-consistency tests cannot find.
    rng = random.Random(20260929)
    chunks = ["abc", "你", "中", "文", "é", "Δ", "12345", " ", "  ", "\n", "\r\n", "\t", "，", "!", "🙂", "\u0301", "٣", "\u00a0", "-foo"]
    corpus += ["".join(rng.choices(chunks, k=rng.randint(1, 24))) for _ in range(100)]
    results = []
    with tempfile.TemporaryDirectory(prefix="myds-tokenizer-") as directory:
        metadata = args.gguf or Path(directory) / "reference.gguf"
        if not args.gguf:
            write_metadata(metadata, json.loads(raw))
        for text in corpus:
            expected = reference.encode(text, add_special_tokens=False).ids
            result = subprocess.run([str(args.probe.resolve()), "--dump", str(metadata), text], check=True, capture_output=True, text=True)
            actual = json.loads(result.stdout)
            passed = expected == actual["ids"] and actual["decoded_hex"] == text.encode().hex()
            results.append({"text": text, "reference_ids": expected, "myds_ids": actual["ids"], "decoded_hex": actual["decoded_hex"], "pass": passed})
            if not passed:
                print("MISMATCH", repr(text), "\n HF", expected, "\nmyds", actual["ids"])
    report = {"reference": "Hugging Face tokenizers.Tokenizer", "tokenizers_version": tokenizers.__version__, "gguf_metadata_sha256": hashlib.sha256(args.gguf.read_bytes()).hexdigest() if args.gguf and args.gguf.stat().st_size < 20_000_000 else None, "source_revision": "517ef625df97ec57aadc91b67506a57c20fdc5bb", "tokenizer_json_sha256": hashlib.sha256(raw).hexdigest(), "cases": len(results), "passed": sum(r["pass"] for r in results), "results": results}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['passed']}/{report['cases']} independent token-ID and byte-output cases passed")
    return int(report["passed"] != report["cases"])


if __name__ == "__main__":
    raise SystemExit(main())
