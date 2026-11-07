#!/usr/bin/env python3
"""从 raw/*.log 重新推导 runs.jsonl，再与已提交的 runs.jsonl 比对。

这是本目录唯一的检查：**表里的数必须能从原始日志里重新算出来**。
不重新跑 GPU、不需要模型，秒级完成。

用法：python3 extract.py --check   （CI / make 用）
      python3 extract.py --write   （重写 runs.jsonl）
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "runs.jsonl"

# 服务端自己的 Completed 行：唯一可信的计时来源。
ROW = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:]+)[^\]]*\] \[INFO\] \[(?P<client>[\d.:]+)\] Completed [^-]+- "
    r"prompt: (?P<prompt>\d+) tok, prefill: (?P<pf_tok>\d+) tok in (?P<pf_s>[\d.]+)s "
    r"= (?P<tps>[\d.]+) t/s \(cached (?P<cached>\d+)\).*?"
    r"decode: (?P<dec_tok>\d+|n/a)"
)

# 每个 raw 日志属于哪一臂、用的哪份 prompt。二进制哈希单独校验。
ARMS = {
    "p32now_server.log":   ("HEAD b0b7b77", "p32k_ab__v0.txt", "e61b7b5c9252b1d3"),
    "p32oct27_server.log": ("691bc26 (2026-10-27)", "p32k_ab__v0.txt", "51c58fec367cbba7"),
    "now.log":             ("HEAD b0b7b77", "synthetic-random-words", "e61b7b5c9252b1d3"),
    "oct27.log":           ("691bc26 (2026-10-27)", "synthetic-random-words", "51c58fec367cbba7"),
    "ck512k.log":          ("HEAD b0b7b77", "synthetic-random-words", "e61b7b5c9252b1d3"),
    "ck1m.log":            ("HEAD b0b7b77", "synthetic-random-words", "e61b7b5c9252b1d3"),
    # 2026-11-07 §143：engram IO 线程 8 vs 16，跑的是台账语料 p32k_ab__v0.txt。
    # 同一二进制（HEAD f0fb04e 的父 a897cec 之后无 C 侧改动），只差一个环境变量。
    "io8_server.log":      ("f0fb04e MYDS_ENGRAM_IO_THREADS=8",  "p32k_ab__v0.txt", "7185c1735f462aef"),
    "io16_server.log":     ("f0fb04e MYDS_ENGRAM_IO_THREADS=16", "p32k_ab__v0.txt", "7185c1735f462aef"),
}

# 冒烟探针（X-Myds-Warmup）不是测量，剔掉。
WARMUP_PROMPT_MAX = 400


def parse():
    rows = []
    for name, (arm, prompt, bin_sha) in ARMS.items():
        p = RAW / name
        if not p.exists():
            sys.exit(f"MISSING RAW LOG: {p}")
        for line in p.read_text(errors="replace").splitlines():
            m = ROW.match(line)
            if not m:
                continue
            prompt_tok = int(m["prompt"])
            if prompt_tok <= WARMUP_PROMPT_MAX:
                continue          # 预热探针
            rows.append({
                "arm": arm,
                "binary_sha256_16": bin_sha,
                "prompt": prompt,
                "prompt_tok": prompt_tok,
                "prefill_tok": int(m["pf_tok"]),
                "prefill_s": float(m["pf_s"]),
                "tps": float(m["tps"]),
                "cached": int(m["cached"]),
                "ts": m["ts"],
                "log": name,
            })
    return rows


def render(rows):
    # 固定排序：臂 → prompt → 时间，便于人眼对照，也便于 diff。
    rows = sorted(rows, key=lambda r: (r["arm"], r["prompt"], r["ts"]))
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main():
    rows = parse()
    text = render(rows)
    if "--write" in sys.argv:
        OUT.write_text(text)
        print(f"wrote {len(rows)} rows -> {OUT.name}")
        return 0
    if not OUT.exists():
        sys.exit("runs.jsonl 不存在；先跑 --write")
    if OUT.read_text() != text:
        sys.exit("runs.jsonl 与 raw 日志不一致 —— 有人改了表没改日志，或反之。"
                 "跑 python3 extract.py --write 重生成，并确认这是有意的。")
    by_arm = {}
    for r in rows:
        by_arm.setdefault((r["arm"], r["prompt"]), []).append(r["tps"])
    print(f"{len(rows)} rows, 与 raw 日志一致")
    for (arm, prompt), v in sorted(by_arm.items()):
        print(f"  {arm:22} {prompt:24} n={len(v):2}  " +
              " ".join(f"{x:.1f}" for x in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
