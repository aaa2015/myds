#!/usr/bin/env python3
"""conflict_findings.py — 找出「同一代码位置被不同臂给出不同定性」的冲突。

用法: conflict_findings.py <report_dir>

做法：
  1. 扫描四臂报告，按「文件:行号」收集每臂对该位置的**原文片段**（前后文各 300 字）
  2. 对多臂都提到的位置，抽取严重度标记（P0/P1/P2/高/中/低/严重/建议 等）
  3. 严重度不一致，或某臂说有问题而另一臂明确说正确/无问题 → 报为冲突
  4. 同时报告「仅一臂提到」的位置数（不可比部分）
"""
import glob
import os
import re
import sys
from collections import defaultdict

ARMS = [
    ("V4.1非", "local-myds-deepseek-v4.1-flash-q2"),
    ("V4.1思", "local-myds-deepseek-v4.1-flash-q2-think"),
    ("1017非", "local-myds-deepseek-v4.1-flash-1017-q2-q4"),
    ("1017思", "local-myds-deepseek-v4.1-flash-1017-q2-q4-think"),
]

REF = re.compile(r"([A-Za-z0-9_./-]+\.(?:c|h|m|py|sh|rs|go|awk))[:：](\d+)")

# 严重度词表 → 数值（越大越严重）
SEV = [
    (re.compile(r"\bP0\b|严重|致命|critical", re.I), 3),
    (re.compile(r"\bP1\b|高(?=危|优先)|\bhigh\b", re.I), 2),
    (re.compile(r"\bP2\b|中(?=等|危)|\bmedium\b", re.I), 1),
    (re.compile(r"\bP3\b|低(?=危|优先)|\blow\b|建议|改进", re.I), 0),
    (re.compile(r"无问题|不是问题|没有问题|正确|无需修改|false positive|误报", re.I), -1),
]


def sev_of(text):
    for rx, v in SEV:
        if rx.search(text):
            return v
    return None


def main():
    d = sys.argv[1]
    # (file, lineno) -> arm -> list of snippet
    hits = defaultdict(lambda: defaultdict(list))
    for label, alias in ARMS:
        for f in sorted(glob.glob(os.path.join(d, alias + "-run*.md"))):
            txt = open(f, errors="replace").read()
            for m in REF.finditer(txt):
                fn = m.group(1).split("/")[-1]  # 允许 bin/x.sh 与 x.sh 归一
                ln = int(m.group(2))
                s, e = max(0, m.start() - 300), min(len(txt), m.end() + 300)
                hits[(fn, ln)][label].append(txt[s:e])

    multi = {k: v for k, v in hits.items() if len(v) >= 2}
    single = {k: v for k, v in hits.items() if len(v) == 1}
    print(f"  位置总数 {len(hits)}   多臂提及 {len(multi)}   仅一臂 {len(single)}")
    print()

    conflicts = []
    for (fn, ln), arms in multi.items():
        sev = {}
        for a, snips in arms.items():
            vals = [sev_of(s) for s in snips]
            vals = [v for v in vals if v is not None]
            if vals:
                sev[a] = max(vals)
        if len(sev) >= 2 and len(set(sev.values())) > 1:
            conflicts.append((fn, ln, sev))

    print(f"  ── 定性冲突：{len(conflicts)} 处 ──")
    for fn, ln, sev in sorted(conflicts, key=lambda x: -max(x[2].values()))[:25]:
        desc = "  ".join(f"{a}={v:+d}" for a, v in sorted(sev.items()))
        print(f"    {fn}:{ln:<5} {desc}")

    print()
    print("  ── 多臂一致（严重度相同）示例 ──")
    same = 0
    for (fn, ln), arms in multi.items():
        sev = {}
        for a, snips in arms.items():
            vals = [sev_of(s) for s in snips if sev_of(s) is not None]
            if vals:
                sev[a] = max(vals)
        if len(sev) >= 3 and len(set(sev.values())) == 1:
            same += 1
            if same <= 8:
                print(f"    {fn}:{ln:<5} {'  '.join(f'{a}={v:+d}' for a, v in sorted(sev.items()))}")
    print(f"    （共 {same} 处三臂以上同严重度）")


if __name__ == "__main__":
    main()
