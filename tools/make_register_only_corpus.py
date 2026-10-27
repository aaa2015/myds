#!/usr/bin/env python3
"""Generate the register-only benchmark corpus deterministically.

Why a generator and a frozen corpus both exist: the corpus files are what gets
measured, so they must be byte-stable. The generator is what lets a reader check
that the files are exactly what they claim to be (same inputs, same bytes, same
sha256), instead of trusting a binary blob.

Three variants per size exist for one reason: the myds server has a prefix cache,
and a repeated prompt would be served from cache (`cached != 0`) which means no
prefill happens at all. Distinct variants defeat the cache without needing any
config change on the server.

Usage:
    python3 tools/make_register_only_corpus.py            # (re)write the corpus
    python3 tools/make_register_only_corpus.py --check     # verify, do not write
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "register_only_2026-10-27" / "prompts"

# The filler is deliberately dull and repetitive: it must not make the model
# produce long outputs, and it must tokenize the same way on every run.
UNIT = "齿轮箱温度监测正常，润滑油位在刻度线之间。"
Q = "\n\n问题：上面反复提到的是什么设备部件？只回答一个词。"
TASK = ("请写一段约 400 字的设备维护说明，主题是齿轮箱巡检，分四个要点，语气专业，"
        "每个要点两到三句，开头写一行小标题，不要复述题目。")

# name -> (body_chars, tail).  body_chars=0 means no filler at all.
PREFILL = {
    "p0.2k": 900,
    "p1.6k": 1600,
    "p6.5k": 6500,
    "p13k": 13000,
    "p26k": 26000,
    "p52k": 52000,
    "p32k_ab": 52000,        # the size used by the A/B
}
DECODE = {
    "d_short": 0,
    "d_8k": 13000,
    "d_32k": 52000,
}
REPS = 3
TAG = "[r{rep}] "


def body(chars):
    if chars <= 0:
        return ""
    reps = chars // len(UNIT) + 1
    return (UNIT * reps)[:chars]


def render(kind, name, chars, rep):
    tag = TAG.format(rep=rep)
    if kind == "prefill":
        return tag + body(chars) + Q
    return tag + body(chars) + ("\n\n" if chars else "") + TASK


def render_fixed(chars):
    """The equality sample must be a *different* file from every timed variant.
    An earlier version reused variant 0 here, so the fixed sample warmed the cache
    for the first timed run and that run did no prefill at all (finished in 0.17 s
    with no server line). Distinct bytes make that impossible.
    """
    return "[fx] " + body(chars) + Q


def build():
    files = {}
    for name, chars in PREFILL.items():
        for rep in range(REPS):
            files[f"{name}__v{rep}.txt"] = render("prefill", name, chars, rep)
    files["ab_fixed.txt"] = render_fixed(PREFILL["p32k_ab"])
    for name, chars in DECODE.items():
        for rep in range(REPS):
            files[f"{name}__v{rep}.txt"] = render("decode", name, chars, rep)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify the corpus, do not write")
    a = ap.parse_args()

    files = build()
    manifest = {}
    bad = []

    for fname in sorted(files):
        data = files[fname].encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        manifest[fname] = dict(chars=len(files[fname]), bytes=len(data), sha256=digest)
        path = OUT / fname
        if a.check:
            if not path.exists():
                bad.append(f"missing {fname}")
                continue
            have = path.read_bytes()
            if have != data:
                bad.append(f"content drift {fname}: on-disk {hashlib.sha256(have).hexdigest()[:16]} "
                           f"!= built {digest[:16]}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    if a.check:
        man = OUT / "manifest.json"
        if not man.exists():
            bad.append("missing manifest.json")
        else:
            on_disk = json.loads(man.read_text())
            if on_disk != manifest:
                diffs = [k for k in set(on_disk) | set(manifest)
                         if on_disk.get(k) != manifest.get(k)]
                bad.append(f"manifest drift in {sorted(diffs)[:5]}")
        if bad:
            print("CORPUS CHECK FAILED", file=sys.stderr)
            for b in bad:
                print("  " + b, file=sys.stderr)
            return 1
        print(f"corpus OK: {len(files)} files match manifest")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum(m["bytes"] for m in manifest.values())
    print(f"wrote {len(files)} prompts ({total} bytes) to {OUT}")
    print(f"manifest sha256 = {hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
