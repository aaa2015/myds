#!/usr/bin/env python3
"""按 pos 分解 decode 的 tag 时间：分离「常数成本」与「随 pos 增长的成本」。

输入：~/bench/ab/results/scaling/trace_{p0,p2k,p8k,p20k}.json
输出：每个 tag 的 每-token-微秒 在四个 pos 上的值，以及对 prompt_tokens 的线性斜率。

用法：python3 tools/parse_scaling.py [--json]

口径：
  · dtags = [tag_id, gpu_seconds, cb_count]，gpu_seconds 是该 tag 上
    GPU 忙时之和（按提交归属，非互斥切片 —— 并行 CB 会重复计入）。
  · 每 token 微秒 = 该 tag 总秒 / out_tokens * 1e6。
  · 斜率单位为 µs / 1K tok 位置；截距（pos=0 外推）即常数部分。
  · 只有 slope 显著 > 0 的 tag 才是「随上下文增长」的，那才是 indexer/attention。
"""
import json, os, sys

NAMES = {0:"NONE",1:"ENGRAM",2:"EMBED",3:"NORM_MIX",4:"QKV",5:"QB",6:"ATTN",
7:"ATTN_OUT",8:"FFN_IN",9:"SHARED_ROUTE",10:"MOE",11:"HEAD",12:"OTHER",
13:"MOE_ROUTE",14:"MOE_GATEUP",15:"MOE_DOWN",16:"MOE_SUM",17:"ATTN_PROJ",
18:"ATTN_COMP",19:"ATTN_IDX",20:"ATTN_RING",21:"ATTN_CORE",22:"ATTN_TAIL",
23:"DEC_PRE_ATTN",24:"DEC_ATTN",25:"DEC_FFN",26:"DEC_ATTN_OUT"}

BASE = os.path.expanduser("~/bench/ab/results/scaling")
FILES = ["trace_p0.json", "trace_p2k.json", "trace_p8k.json", "trace_p20k.json"]


def load():
    out = []
    for f in FILES:
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        d = json.load(open(p))
        r = d["records"][0]
        out.append({
            "name": f.replace("trace_", "").replace(".json", ""),
            "prompt": r["prompt_tokens"],
            "out": r["out_tokens"],
            "tps": r["decode_tps"],
            "decode_s": r["phases"]["decode"],
            "dtags": r["detail"]["dtags"] or [],
            "dgpu": r["detail"]["decode_gpu"],
            "dwait": r["detail"]["decode_wait"],
            "cbs": r["detail"]["decode_cbs"],
        })
    return out


def linreg(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, my - slope * mx


def main():
    as_json = "--json" in sys.argv
    data = load()

    if not as_json:
        print("=== 样本 ===")
        print(f"{'trace':>10} {'prompt':>8} {'out':>5} {'tps':>7} {'decode_s':>9} "
              f"{'gpu_s':>8} {'wait_s':>8} {'cbs':>6}")
        for d in data:
            print(f"{d['name']:>10} {d['prompt']:8d} {d['out']:5d} {d['tps']:7.2f} "
                  f"{d['decode_s']:9.3f} {d['dgpu']:8.3f} {d['dwait']:8.3f} {d['cbs']:6d}")

    tags = sorted({t[0] for d in data for t in d["dtags"]})
    xs = [d["prompt"] for d in data]
    result = {}

    if not as_json:
        print("\n=== 每 token GPU 微秒（按 prompt 位置）===")
        head = f"{'tag':<14}" + "".join(f"{d['name']:>9}" for d in data) + f"{'斜率':>10} {'截距':>9}"
        print(head)
        print("-" * len(head))

    for tid in tags:
        cells = []
        for d in data:
            s = sum(t[1] for t in d["dtags"] if t[0] == tid)
            cells.append(s / d["out"] * 1e6 if d["out"] else 0.0)
        slope, icpt = linreg(xs, cells)
        nm = NAMES.get(tid, f"SEG{tid}")
        result[nm] = {"cells": cells, "slope_per_1k": slope * 1000,
                      "slope_per_token": slope, "intercept": icpt,
                      "share_at_p20k": cells[-1]}
        if not as_json:
            print(f"{nm:<14}" + "".join(f"{c:9.1f}" for c in cells)
                  + f"{slope*1000:10.3f} {icpt:9.1f}")

    # 关键：分解 p20k 的每 token 成本
    tot = result.get("HEAD", {}).get("cells", [0]*4)
    p20k_idx = len(data) - 1
    total_us = sum(v["cells"][p20k_idx] for v in result.values())

    if not as_json:
        print(f"\n=== 在 p20k 上分解每 token {total_us:.1f} µs ===")
        print(f"{'tag':<14} {'µs/tok':>9} {'占比':>7} {'斜率µs/K':>10}  判定")
        for nm, v in sorted(result.items(), key=lambda kv: -kv[1]["cells"][p20k_idx]):
            us = v["cells"][p20k_idx]
            if us < 0.05:
                continue
            share = us / total_us * 100 if total_us else 0
            slope = v["slope_per_1k"]
            if slope > 0.02:
                verdict = "★ 随 pos 增长"
            elif abs(slope) <= 0.02:
                verdict = "常数（与 pos 无关）"
            else:
                verdict = "随 pos 减小（异常）"
            print(f"{nm:<14} {us:9.1f} {share:6.1f}% {slope:10.3f}  {verdict}")

        # 常数 vs 增长 的整体拆分
        grow = sum(v["cells"][p20k_idx] for v in result.values() if v["slope_per_1k"] > 0.02)
        const = total_us - grow
        print(f"\n随 pos 增长部分: {grow:.1f} µs/tok ({grow/total_us*100:.1f}%)")
        print(f"常数部分:        {const:.1f} µs/tok ({const/total_us*100:.1f}%)")

    if as_json:
        print(json.dumps({"samples": data, "tags": result,
                          "total_us_p20k": total_us}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
