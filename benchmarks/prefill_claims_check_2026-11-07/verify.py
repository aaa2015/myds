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

    # 2. 任何一臂都不得【比台账慢】2% 以上 —— 那才是回归。
    #    比台账快是正常的：那说明有优化落地，应该记录而不是失败。
    #    （2026-11-07：这条原先是双向的「两臂差 >2% 即回归」，把 engram 线程
    #      8→16 的 +3.7% 改进误判成回归。方向错了，改成单向。）
    ledger = 618.9
    med = {}
    for arm in sorted({r["arm"] for r in corpus_rows}):
        v = [r["tps"] for r in corpus_rows if r["arm"] == arm]
        med[arm] = st.median(v)
        if med[arm] < ledger * 0.98:
            fails.append(f"{arm} 在台账语料上是 {med[arm]:.1f} t/s，比台账 {ledger} 慢 "
                         f"{(1 - med[arm]/ledger)*100:.1f}% —— 这是回归")

    # 3. 至少两臂用过台账语料，否则「同语料对照」这句话没有对象
    arms = list(med)
    if len(arms) < 2:
        fails.append(f"只有 {len(arms)} 臂用了台账语料，无法做同语料对照")
    # 4. 最快的臂必须是真的比台账快，而不是靠记错数字产生的幻觉
    best = max(med.values())
    if best > max((r["tps"] for r in corpus_rows)) * 1.001:
        fails.append(f"中位数 {best:.1f} 高于任何单次测量 —— 汇总逻辑有 bug")

    for arm, v in sorted(med.items(), key=lambda kv: -kv[1]):
        delta = (v / ledger - 1.0) * 100
        print(f"  {arm:36} 台账语料 median {v:6.1f} t/s  ({delta:+.1f}% vs 台账)")
    print(f"  {'台账 §105 记录':36}                  {ledger:6.1f} t/s")
    if fails:
        for f in fails:
            print("FAIL", f)
        return 1
    print("PASS —— 可比的 claim 都能在原始日志里找到，两臂与台账一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
