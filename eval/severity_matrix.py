#!/usr/bin/env python3
"""severity_matrix.py — 从报告自身的编号条目里提取严重度，重建冲突矩阵。

上一版用「前后 300 字里第一个出现的严重度词」，会把相邻条目的"建议"误当成
该位置的定性。这一版改为**只认条目开头的标注**：

  C15｜开机等待超时后无条件启动（P2）      → P2
  **S4（中）SELinux 域白名单…**            → 中
  C12（低）service.sh 超时后…              → 低
  - **A-3，P1**                            → P1

即：在同一段落里，找「编号 + 严重度」紧邻出现的模式，再确认该段落提到目标行号。

用法: severity_matrix.py <report_dir>
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

# 条目头：编号（C12 / S4 / A-3 / E7 / P0 独立）后跟严重度
ITEM = re.compile(
    r"(?:\b[A-Z]{1,2}-?\d{1,2}\b[｜|]?)"          # 编号
    r"[^。\n]{0,60}?"
    r"(?:\bP0\b|\bP1\b|\bP2\b|\bP3\b|严重|高危|中危|低危|高优先|中优先|低优先|"
    r"\bHigh\b|\bMedium\b|\bLow\b|\[High\]|\[Med\]|\[Low\])",
    re.I,
)

LEVEL = [
    (re.compile(r"\bP0\b|严重|高危|高优先", re.I), 3),
    (re.compile(r"\bP1\b|\[High\]|\bHigh\b", re.I), 2),
    (re.compile(r"\bP2\b|中危|中优先|\[Med\]|\bMedium\b", re.I), 1),
    (re.compile(r"\bP3\b|低危|低优先|\[Low\]|\bLow\b", re.I), 0),
]

REF = re.compile(r"([A-Za-z0-9_./-]+\.(?:c|h|m|py|sh|rs|go|awk))[:：](\d+)")


def level_of(text):
    for rx, v in LEVEL:
        if rx.search(text):
            return v
    return None


def main():
    d = sys.argv[1]
    # (file,line) -> arm -> set(levels)
    table = defaultdict(lambda: defaultdict(set))

    for label, alias in ARMS:
        for f in sorted(glob.glob(os.path.join(d, alias + "-run*.md"))):
            txt = open(f, errors="replace").read()
            # 按段落切（空行分隔）
            for para in re.split(r"\n\s*\n", txt):
                refs = {(m.group(1).split("/")[-1], int(m.group(2))) for m in REF.finditer(para)}
                if not refs:
                    continue
                lv = None
                m = ITEM.search(para)
                if m:
                    lv = level_of(m.group(0))
                if lv is None:
                    # 段落里没有条目头，退回段落内最强的显式等级
                    lv = level_of(para)
                if lv is None:
                    continue
                for r in refs:
                    table[r][label].add(lv)

    multi = {k: v for k, v in table.items() if len(v) >= 2}
    conflict = {k: v for k, v in multi.items() if len({max(s) for s in v.values()}) > 1}

    print(f"  有明确等级标注且多臂提及的位置: {len(multi)}")
    print(f"  其中等级不一致: {len(conflict)}")
    print()
    print(f"  {'位置':<34}{'V4.1非':>8}{'V4.1思':>8}{'1017非':>8}{'1017思':>8}")
    names = ["V4.1非", "V4.1思", "1017非", "1017思"]
    for (fn, ln), arms in sorted(conflict.items(), key=lambda x: -max(max(s) for s in x[1].values())):
        row = f"  {fn}:{ln:<26}"
        for n in names:
            v = arms.get(n)
            row += f"{('P'+str(3-max(v)) if v else '—'):>8}" if v else f"{'—':>8}"
        print(row)


if __name__ == "__main__":
    main()
