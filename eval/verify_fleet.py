#!/usr/bin/env python3
"""verify_fleet.py — 核对 fleet.conf 每行 PATH 能否解析到可用的主模型。

对 E != 0 的行做完整核对；E == 0 的行只报告明显断裂（可选 --all）。
解析规则与 start_fleet.sh 保持一致：
  find <dir> -iname "*.gguf" 排除 mmproj/mtp/dspark/dflash/Vision -maxdepth 1 | sort | head -1
"""
import os
import sys

D = os.path.expanduser("~/.lmstudio/models")
EXCLUDE = ("mmproj", "mtp", "dspark", "dflash", "vision")


def pick(d):
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return None
    for n in names:
        low = n.lower()
        if not low.endswith(".gguf"):
            continue
        if any(x in low for x in EXCLUDE):
            continue
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
        return ("BROKEN", p)
    return None


def main():
    show_all = "--all" in sys.argv
    rows = []
    for line in open("fleet.conf"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = [x.strip() for x in s.split("|")]
        if len(f) < 18:
            continue
        rows.append(f)

    print(f"  数据行 {len(rows)}")
    bad = 0
    for f in rows:
        e, alias, port, path = f[0], f[2], f[3], f[15]
        if e == "0" and not show_all:
            continue
        d = os.path.join(D, path)
        r = pick(d)
        if r is None:
            print(f"  ❌ E={e} PORT={port} {alias}\n       PATH={path} → 目录缺失或无主模型")
            bad += 1
            continue
        if isinstance(r, tuple):
            print(f"  ❌ E={e} PORT={port} {alias}\n       PATH={path} → 断链 {os.path.basename(r[1])}")
            bad += 1
            continue
        vis = [n for n in sorted(os.listdir(d)) if "vision" in n.lower() and n.lower().endswith(".gguf")]
        vok = bool(vis) and os.path.exists(os.path.join(d, vis[0]))
        print(f"  ✅ E={e} PORT={port} {alias}")
        print(f"       lang: {os.path.basename(r)}")
        print(f"       vis : {vis[0] if vis else '无'}{'' if vok or not vis else ' (断链)'}")
    print()
    print(f"  问题 {bad} 处")


if __name__ == "__main__":
    main()
