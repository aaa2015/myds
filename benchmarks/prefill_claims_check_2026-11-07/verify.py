#!/usr/bin/env python3
"""把 prefill_claims.json 里可比的 claim 与 runs.jsonl 对上。

为什么单独一个检查：runs.jsonl 说「这一臂测到 619.4」，claim 说「619.4 可与台账比」，
两者必须指同一次测量。分开写就会各自漂移。

不需要 GPU、不需要模型。
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs.jsonl"
CLAIMS = ROOT / "benchmarks" / "prefill_claims.json"
# 台账 §105 自己的原始行。可比的 claim 要么来自本次测量，要么来自这里 ——
# 只接受这两处，别的地方出现的数就没有出处。
LEDGER_RAW = ROOT / "benchmarks" / "register_only_2026-10-27" / "raw" / "rig.jsonl"

TOL = 0.35   # t/s：日志只有一位小数，取 0.35 足以吃掉四舍五入又不掩盖真差异


def ledger_tps():
    """台账那次运行里所有未被缓存命中的 prefill t/s。"""
    if not LEDGER_RAW.exists():
        return []
    out = []
    for line in LEDGER_RAW.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "run" or r.get("cached") != 0:
            continue
        if r.get("tps") is not None:
            out.append(float(r["tps"]))
    return out


def main():
    fails = []
    runs = [json.loads(l) for l in RUNS.read_text().splitlines() if l.strip()]
    reg = json.loads(CLAIMS.read_text())
    ref = ledger_tps()

    corpus_rows = [r for r in runs if r["prompt"].endswith(".txt")]
    if not corpus_rows:
        fails.append("runs.jsonl 里没有一条用仓库语料的测量 —— 这份 archive 就没回答它该回答的问题")

    # 1. 每个「可比」claim 必须能在两处原始证据之一里找到
    for c in reg["claims"]:
        if not c.get("comparable_to_ledger_105"):
            continue
        hits = [r for r in corpus_rows if abs(r["tps"] - c["value"]) <= TOL]
        if hits:
            if c.get("prompt_file") and hits[0]["prompt"] != c["prompt_file"]:
                fails.append(f"claim {c['value']} 说用 {c['prompt_file']}，runs.jsonl 里那条用的是 "
                             f"{hits[0]['prompt']}")
        elif any(abs(v - c["value"]) <= TOL for v in ref):
            pass                      # 出自台账自己的原始行，合法
        else:
            fails.append(f"claim {c['value']} t/s（{c['where']}）在 runs.jsonl 与台账原始行里"
                         f"都找不到对应测量")

    # 2. 台账 §105 的数必须落在两臂的范围内 —— 否则「没掉」这句话站不住
    ledger = 618.9
    med = {}
    for arm in sorted({r["arm"] for r in corpus_rows}):
        v = [r["tps"] for r in corpus_rows if r["arm"] == arm]
        med[arm] = st.median(v)
        if abs(med[arm] - ledger) / ledger > 0.01:
            fails.append(f"{arm} 在台账语料上是 {med[arm]:.1f} t/s，与台账 {ledger} 相差超过 1%")

    # 3. 同一语料、两臂之间不得出现「回归」（差 >2%）
    arms = list(med)
    if len(arms) < 2:
        fails.append(f"只有 {len(arms)} 臂用了台账语料，无法做同语料对照")
    else:
        spread = (max(med.values()) - min(med.values())) / max(med.values())
        if spread > 0.02:
            fails.append(f"两臂在台账语料上相差 {spread*100:.1f}% —— 这才是回归，别放过")

    for arm, v in med.items():
        print(f"  {arm:22} 台账语料 median {v:.1f} t/s")
    print(f"  台账 §105 记录 {ledger} t/s")
    if fails:
        for f in fails:
            print("FAIL", f)
        return 1
    print("PASS —— 可比的 claim 都能在原始日志里找到，两臂与台账一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
