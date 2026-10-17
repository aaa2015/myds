#!/usr/bin/env python3
"""kl_report.py — 比较两个 score --dump 目录的逐位置 KL。

用法:
  kl_report.py <ref_dir> <tgt_dir> [--json out.json]

两个目录必须来自【同一 token 序列】（同一 tokenizer、同一文本），
否则 pos-N 文件不对应，结果无意义。文件数不一致直接报错，不猜。

指标（每个位置）:
  KL(ref || tgt) = sum_v P_ref(v) * (logP_ref(v) - logP_tgt(v))
两个模型都只给 logits，所以统一用 log_softmax 归一。

汇总用【每位置 KL 的均值和中位数】。中位数抗离群：某个位置若模型分歧极大
（例如文本切换主题），均值会被单个位置拉高；两者都报，不二选一。
"""
import argparse
import json
import math
import os
import struct
import sys


def load_logits(path):
    with open(path, "rb") as f:
        b = f.read()
    n = len(b) // 4
    if n == 0 or len(b) % 4:
        raise ValueError(f"{path}: not a float32 array ({len(b)} bytes)")
    return struct.unpack("<%df" % n, b)


def log_softmax(v):
    mx = max(v)
    s = 0.0
    for x in v:
        s += math.exp(x - mx)
    lse = mx + math.log(s)
    return [x - lse for x in v]


def kl(p_log, q_log):
    """KL(P||Q) given log-probabilities."""
    tot = 0.0
    for lp, lq in zip(p_log, q_log):
        p = math.exp(lp)
        if p > 0.0:
            tot += p * (lp - lq)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("tgt_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ref_files = sorted(f for f in os.listdir(args.ref_dir) if f.endswith(".f32"))
    tgt_files = sorted(f for f in os.listdir(args.tgt_dir) if f.endswith(".f32"))
    if not ref_files:
        sys.exit(f"no .f32 files in {args.ref_dir}")
    if ref_files != tgt_files:
        only_r = sorted(set(ref_files) - set(tgt_files))
        only_t = sorted(set(tgt_files) - set(ref_files))
        sys.exit(f"position sets differ (ref-only={only_r[:3]} tgt-only={only_t[:3]}) "
                 f"— the two dumps are not from the same token sequence")

    kl_rt, kl_tr, top1_agree = [], [], 0
    for name in ref_files:
        a = load_logits(os.path.join(args.ref_dir, name))
        b = load_logits(os.path.join(args.tgt_dir, name))
        if len(a) != len(b):
            sys.exit(f"{name}: vocab size differs ({len(a)} vs {len(b)})")
        la, lb = log_softmax(a), log_softmax(b)
        kl_rt.append(kl(la, lb))
        kl_tr.append(kl(lb, la))
        if max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__):
            top1_agree += 1

    def stats(xs):
        n = len(xs)
        m = sum(xs) / n
        s = sorted(xs)
        med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        return m, med, max(xs)

    m_rt, med_rt, max_rt = stats(kl_rt)
    m_tr, med_tr, _ = stats(kl_tr)
    out = {
        "kind": "kl",
        "positions": len(ref_files),
        "kl_ref_tgt_mean": m_rt,
        "kl_ref_tgt_median": med_rt,
        "kl_ref_tgt_max": max_rt,
        "kl_tgt_ref_mean": m_tr,
        "kl_tgt_ref_median": med_tr,
        "top1_agreement": top1_agree / len(ref_files),
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
