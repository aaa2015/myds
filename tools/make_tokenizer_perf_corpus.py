#!/usr/bin/env python3
"""Generate deterministic, synthetic CPU tokenizer workloads (no user data)."""
import argparse
import hashlib
import json
from pathlib import Path


def corpus():
    english = (
        "The compressor reads rows from the table and folds them into the "
        "residual stream before the next sublayer. Check the result carefully.\n"
    )
    code = (
        "def stable_sort(rows):\n"
        "    # Preserve the input order for equal keys.\n"
        "    return sorted(rows, key=lambda row: (row['priority'], row['name']))\n"
        "assert stable_sort([{'priority': 12, 'name': 'alpha'}])[0]['priority'] == 12\n"
    )
    chinese = "请检查网络连接和配置文件，记录错误原因。处理完成后保留日志，确保下一次请求能够正常运行。\n"
    mixed = (
        "<｜User｜>请审查这段代码并解释边界条件。\n" + code
        + "café e\u0301 Δοκιμή Привет ٣٤٥٦ 👩‍💻\n<｜Assistant｜>"
    )

    def repeat(text, target):
        data = text.encode("utf-8")
        # Repeat complete units so no synthetic UTF-8 truncation enters timing.
        return data * max(1, target // len(data))

    return {
        "short_ascii.txt": b"Please explain how a stable sort works.",
        "short_mixed.txt": "你好，请解释 stable sort 的含义。🙂".encode("utf-8"),
        "english_256k.txt": repeat(english, 256 * 1024),
        "code_256k.txt": repeat(code, 256 * 1024),
        "chinese_128k.txt": repeat(chinese, 128 * 1024),
        "mixed_512k.txt": repeat(mixed, 512 * 1024),
        "long_piece.txt": b"a" * 16384 + b" short words 123 " + b"b" * 8192,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = corpus()
    manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    files["manifest.json"] = manifest_bytes
    if args.check:
        for name, data in files.items():
            path = args.out / name
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit(f"corpus mismatch: {path}")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (args.out / name).write_bytes(data)
    print(f"corpus {'verified' if args.check else 'written'}: {len(manifest)} files")


if __name__ == "__main__":
    main()
