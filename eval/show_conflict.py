#!/usr/bin/env python3
"""show_conflict.py — 导出指定「文件:行号」在四臂报告中的原文片段。

用法: show_conflict.py <report_dir> <file> <line> [window]

为了判断谁对，需要看每臂对该位置的**完整论述**，而不只是一句结论。
片段以该行引用为中心，向前后各展开 window 字符（默认 700）。
"""
import glob
import os
import re
import sys

ARMS = [
    ("V4.1 非思考", "local-myds-deepseek-v4.1-flash-q2"),
    ("V4.1 思考", "local-myds-deepseek-v4.1-flash-q2-think"),
    ("1017 非思考", "local-myds-deepseek-v4.1-flash-1017-q2-q4"),
    ("1017 思考", "local-myds-deepseek-v4.1-flash-1017-q2-q4-think"),
]


def main():
    d, fn, ln = sys.argv[1], sys.argv[2], int(sys.argv[3])
    win = int(sys.argv[4]) if len(sys.argv) > 4 else 700
    print(f"  ═══ {fn}:{ln} ═══")
    for label, alias in ARMS:
        found = False
        for f in sorted(glob.glob(os.path.join(d, alias + "-run*.md"))):
            txt = open(f, errors="replace").read()
            run = re.search(r"run(\d)", f).group(1)
            for m in re.finditer(re.escape(fn) + r"[:：]" + str(ln) + r"(?!\d)", txt):
                s, e = max(0, m.start() - win // 2), min(len(txt), m.end() + win)
                frag = txt[s:e].replace("\n", " ")
                frag = re.sub(r"\s{2,}", " ", frag)
                print(f"\n  ── {label} run{run} ──")
                print(f"     {frag}")
                found = True
                break
        if not found:
            print(f"\n  ── {label} ── （未提及）")


if __name__ == "__main__":
    main()
