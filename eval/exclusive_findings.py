#!/usr/bin/env python3
"""exclusive_findings.py — 统计四次评审各自的独占发现。

用法: exclusive_findings.py <report_dir>

以「文件:行号」为发现的指纹：同一位置被多臂指出视为同一发现（不论措辞）。
输出：
  1. 每臂的唯一发现数、被其它臂共享的数目、独占的数目
  2. 两两重叠矩阵
  3. 每臂的独占清单（前若干条）
"""
import glob
import os
import re
import sys
from collections import defaultdict

ARMS = [
    ("V4.1 非思考", "local-myds-deepseek-v4.1-flash-q2"),
    ("V4.1 思考", "local-myds-deepseek-v4.1-flash-q2-think"),
    ("1017 非思考", "local-myds-deepseek-v4.1-flash-1017-q2-q4"),
    ("1017 思考", "local-myds-deepseek-v4.1-flash-1017-q2-q4-think"),
]

# 文件:行号 或 文件:起-止
REF = re.compile(r"([A-Za-z0-9_./-]+\.(?:c|h|m|py|sh|rs|go|awk))[:：](\d+)(?:\s*[-–~]\s*(\d+))?")


def key_of(m):
    f, a, b = m.group(1), int(m.group(2)), m.group(3)
    # 行号归并到最近 5 行，避免 231 与 233 算成两个发现
    return (f, a // 5 * 5)


def main():
    d = sys.argv[1]
    refs = {}
    for label, alias in ARMS:
        s = set()
        for f in sorted(glob.glob(os.path.join(d, alias + "-run*.md"))):
            txt = open(f, errors="replace").read()
            for m in REF.finditer(txt):
                s.add(key_of(m))
        refs[label] = s

    print("  ── 每臂规模 ──")
    for label, _ in ARMS:
        print(f"    {label:<14} 唯一位置 {len(refs[label]):3d}")

    print()
    print("  ── 独占与被共享 ──")
    print(f"    {'臂':<14}{'唯一':>6}{'共享':>6}{'独占':>6}{'独占率':>8}")
    excl = {}
    for label, _ in ARMS:
        mine = refs[label]
        others = set().union(*[refs[o] for o, _ in ARMS if o != label])
        e = mine - others
        excl[label] = e
        rate = 100 * len(e) / len(mine) if mine else 0
        print(f"    {label:<14}{len(mine):>6}{len(mine) - len(e):>6}{len(e):>6}{rate:>7.0f}%")

    print()
    print("  ── 两两重叠（共同指出同一位置）──")
    labels = [l for l, _ in ARMS]
    print("    " + " " * 14 + "".join(f"{l[:6]:>8}" for l in labels))
    for a in labels:
        row = f"    {a:<14}"
        for b in labels:
            row += f"{len(refs[a] & refs[b]):>8}"
        print(row)

    print()
    print("  ── 全臂共识（4 臂都指出）──")
    consensus = set.intersection(*[refs[l] for l in labels])
    print(f"    共 {len(consensus)} 处")
    for f, ln in sorted(consensus)[:10]:
        print(f"      {f}:{ln}")

    print()
    print("  ── 各臂独占清单（最多 8 条）──")
    for label, _ in ARMS:
        e = sorted(excl[label])
        print(f"    {label}（{len(e)} 条）:")
        for f, ln in e[:8]:
            print(f"      {f}:{ln}")


if __name__ == "__main__":
    main()
