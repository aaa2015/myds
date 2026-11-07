#!/usr/bin/env python3
"""从源码算出 knob 集合指纹，并与二进制自报的值比对。

为什么需要（2026-11-07 实测踩到）：二进制与 myds.conf 现在是【成对】的。
旧二进制不认识 `MYDS_INDEXER_ROWS4_MIN`，配新 conf 会拒绝启动（退出码 2）；
回滚时若只换二进制不换 conf，服务就起不来。我们踩了两轮 START_FAILED 才发现。

指纹 = 本 build 认识的【全部 knob 名字】按字节序排序后的 FNV-1a 64。
它只取决于 include/myds_knobs_generated.h，与编译器/SDK 无关，
所以两台机器上同一源码必然得到同一个值 —— 这正是它适合做配对判据的原因。

两种用法：

  # 1) 只算源码侧的期望值（不需要二进制）
  python3 tools/knob_fingerprint.py --expect

  # 2) 与某个二进制的自报值比对（部署/回滚前用）
  python3 tools/knob_fingerprint.py --check bin/myds
  python3 tools/knob_fingerprint.py --check bin/myds --conf /path/myds.conf

--conf 会额外检查：conf 里出现的每个 MYDS_* 键，二进制是否都认识
（等价于 `myds --print-config` 的 unread 那一半，但只看键名，不加载模型）。
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "include" / "myds_knobs_generated.h"

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def knob_names():
    """从生成的表里取出全部 knob 名字。"""
    text = GENERATED.read_text(encoding="utf-8")
    return re.findall(r'\{"(MYDS_[A-Z0-9_]+)"', text)


def fingerprint(names):
    """必须与 src/myds_knobs.c 的 myds_knobs_fingerprint 逐位一致：
    名字按字节序排序 → 逐个名字字节做 FNV-1a → 每个名字后混入 0x1F 分隔 →
    最后混入元素个数。"""
    h = FNV_OFFSET
    ordered = sorted(names, key=lambda s: s.encode())
    for name in ordered:
        for b in name.encode():
            h ^= b
            h = (h * FNV_PRIME) & MASK64
        h ^= 0x1F
        h = (h * FNV_PRIME) & MASK64
    h ^= len(ordered)
    h = (h * FNV_PRIME) & MASK64
    return h


def conf_keys(path):
    """myds.conf 的有效键（忽略注释与空行、只认 MYDS_ 前缀）。"""
    keys = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s and s.startswith("MYDS_"):
            keys.append(s.split("=", 1)[0].strip())
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", action="store_true", help="只打印源码侧期望指纹")
    ap.add_argument("--check", metavar="BINARY", help="与二进制的 --knob-fingerprint 比对")
    ap.add_argument("--conf", metavar="CONF", help="同时检查 conf 里的键是否都被该二进制认识")
    args = ap.parse_args()

    if not GENERATED.exists():
        sys.exit(f"缺少 {GENERATED}；先跑 python3 tools/knob_inventory.py --write")

    names = knob_names()
    if not names:
        sys.exit(f"{GENERATED} 里没有解析出任何 knob 名字 —— 格式变了？")
    want = fingerprint(names)
    want_s = "%016x" % want

    if args.expect or not args.check:
        print(f"knobs    : {len(names)}")
        print(f"expected : {want_s}")
        return 0

    # ── 与二进制比对 ────────────────────────────────────────────────
    try:
        out = subprocess.run([args.check, "--knob-fingerprint"],
                             capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        sys.exit(f"二进制不可执行：{args.check}")
    except subprocess.TimeoutExpired:
        sys.exit(f"{args.check} --knob-fingerprint 超时")
    got = out.stdout.strip()
    if out.returncode != 0 or not re.fullmatch(r"[0-9a-f]{16}", got):
        # 老二进制不认这个参数 —— 这本身就是「比当前源码旧」的最强证据，
        # 比指纹不等更直接。把它报成明确的不兼容，而不是「输出异常」。
        if "未知命令行参数" in out.stderr or "unknown" in out.stderr.lower():
            sys.exit(f"❌ {args.check} 不认识 --knob-fingerprint ⇒ 它比当前源码旧，\n"
                     f"   换二进制时必须同时换 conf（旧二进制不认新键会拒绝启动）。")
        sys.exit(f"{args.check} --knob-fingerprint 输出异常：rc={out.returncode} "
                 f"stdout={got!r} stderr={out.stderr.strip()[:200]!r}")

    print(f"knobs    : {len(names)}")
    print(f"expected : {want_s}   (源码)")
    print(f"binary   : {got}   ({args.check})")
    if got != want_s:
        sys.exit("❌ 指纹不一致：这个二进制不是由当前源码构建的，"
                 "或 knobs_generated.h 在构建后被改过。")
    print("✅ 二进制与源码的 knob 集合一致")

    # ── conf 兼容性 ─────────────────────────────────────────────────
    if args.conf:
        keys = conf_keys(args.conf)
        known = set(names)
        unknown = [k for k in keys if k not in known]
        print(f"\nconf     : {args.conf}（{len(keys)} 个 MYDS_* 键）")
        if unknown:
            print(f"❌ 其中 {len(unknown)} 个这个二进制不认识 —— 它会拒绝启动：")
            for k in unknown:
                print(f"   {k}")
            print("\n⇒ 换二进制时必须同时换 conf（或回滚 conf）。")
            return 1
        print("✅ conf 里的每个键这个二进制都认识")
    return 0


if __name__ == "__main__":
    sys.exit(main())
