#!/usr/bin/env python3
"""Generate the ledger's §105 tables and benchmarks/.../expected.json from raw rows.

Design rule this script exists to enforce:

    A number that is copied is a number that can drift.
    So the raw JSONL is the only input, and every displayed figure is produced
    from it by this file. Nothing is transcribed by hand.

Markers in the ledger delimit the generated regions:

    <!-- BEGIN GENERATED:<name> -->
    ...
    <!-- END GENERATED:<name> -->

`--check` regenerates in memory and shows what a rebuild would change; exit 1 if the
committed ledger is stale. `--write` rewrites the regions in place.

Usage:
    python3 tools/gen_ledger_tables.py --check
    python3 tools/gen_ledger_tables.py --write
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "benchmarks" / "register_only_2026-10-27" / "raw"
BENCH = ROOT / "benchmarks" / "register_only_2026-10-27"
LEDGER = ROOT / "docs" / "2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md"

# Tolerances are a *judgement*, not a measurement: they cannot be derived from the
# rows. They live here, in one place, and the ledger prints them from here.
TOLERANCES = {
    "tps_pct": 2.0,
    "prompt_tok_pct": 2.0,
    "gain_pp": 1.5,
    "gpu_residency_min": 95.0,
    "gpu_power_w": [85.0, 125.0],
}


def load(name):
    p = RAW / name
    if not p.exists():
        sys.exit(f"MISSING RAW DATA: {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def ab_facts():
    rows = load("rig.jsonl")
    runs = [r for r in rows if r.get("kind") == "run"]
    nonce = [r for r in runs if r.get("sample") == "nonce"]
    fixed = [r for r in runs if r.get("sample") == "fixed"]
    ends = [r for r in rows if r.get("kind") == "boot_end" and r.get("delta")]

    per, cycles = {}, {}
    for arm in ("OFF", "ON"):
        v = sorted(r["tps"] for r in nonce if r["arm"] == arm)
        per[arm] = dict(n=len(v), min=v[0], max=v[-1], median=st.median(v),
                        range_pct=100 * (v[-1] - v[0]) / st.median(v),
                        sd=st.stdev(v) if len(v) > 1 else 0.0)
        for b in sorted({r["boot"] for r in nonce if r["arm"] == arm}):
            vv = [r["tps"] for r in nonce if r["arm"] == arm and r["boot"] == b]
            cycles[f"{arm}{b}"] = dict(vals=vv, median=st.median(vv))

    def spread(vals):
        return 100 * (max(vals) - min(vals)) / st.median(vals)

    off_cyc = [cycles[f"OFF{i}"]["median"] for i in (1, 2, 3)]
    on_cyc = [cycles[f"ON{i}"]["median"] for i in (1, 2, 3)]
    paired = [100 * (cycles[f"ON{i}"]["median"] / cycles[f"OFF{i}"]["median"] - 1)
              for i in (1, 2, 3)]

    gain = 100 * (per["ON"]["median"] / per["OFF"]["median"] - 1)
    diff = st.mean([r["tps"] for r in nonce if r["arm"] == "ON"]) - \
        st.mean([r["tps"] for r in nonce if r["arm"] == "OFF"])
    se = (per["OFF"]["sd"] ** 2 / per["OFF"]["n"] + per["ON"]["sd"] ** 2 / per["ON"]["n"]) ** 0.5

    shas = {a: sorted({r["sha"] for r in fixed if r["arm"] == a}) for a in ("OFF", "ON")}
    toks = sorted({r["prompt_tok"] for r in fixed if r.get("prompt_tok")})
    ctok = sorted({r["completion_tok"] for r in fixed})

    deltas = {a: [e["delta"] for e in ends if e["arm"] == a] for a in ("OFF", "ON")}

    wall = {}
    for arm in ("OFF", "ON"):
        pairs = [(r["wall_s"], r["prefill_s"]) for r in nonce
                 if r["arm"] == arm and r.get("prefill_s")]
        wall[arm] = dict(wall=st.median([p[0] for p in pairs]),
                         server=st.median([p[1] for p in pairs]),
                         overhead=st.median([p[0] - p[1] for p in pairs]))

    return dict(per=per, cycles=cycles, gain=gain, diff=diff, se=se,
                z=diff / se if se else 0,
                spread_off=spread(off_cyc), spread_on=spread(on_cyc),
                paired=paired, shas=shas, fixed_prompt_tok=toks, fixed_completion=ctok,
                deltas=deltas, wall=wall,
                cached_nonzero=sum(1 for r in runs if r.get("cached") != 0))


def sweep_facts():
    rows = [r for r in load("sweep2.jsonl") if r.get("phase") == "sweep" and r.get("server")]
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r["server"])
    return by


def dec_facts():
    by = {}
    for r in load("dec2.jsonl"):
        if r.get("server"):
            by.setdefault(r["case"], []).append(r["server"])
    return by


def gpu_facts():
    return {r["arm"]: r for r in load("sweep2.jsonl") if r.get("phase") == "gpu"}


def smoke_facts():
    p = RAW / "smoke2.jsonl"
    if not p.exists():
        return None
    runs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    runs = [r for r in runs if r.get("kind") == "run"]
    return dict(arms=sorted({r["arm"] for r in runs}),
                fixed=[r["tps"] for r in runs if r.get("sample") == "fixed"],
                nonce=sorted(r["tps"] for r in runs if r.get("sample") == "nonce"),
                cached=[r.get("cached") for r in runs],
                res=sorted({r["gpu_detail"]["res_median"] for r in runs if r.get("gpu_detail")}),
                power=sorted({round(r["gpu_detail"]["power_median_mw"] / 1000, 1)
                              for r in runs if r.get("gpu_detail")}))


def f1(x):
    return f"{x:.1f}"


def render_cycle_table(ab):
    out = ["| 周期 | 臂 | 三次 | 周期中位 |", "|---|---|---:|---:|"]
    for i in (1, 2, 3):
        for arm, label, tag in (("OFF", "关", "O"), ("ON", "开", "N")):
            c = ab["cycles"][f"{arm}{i}"]
            vals = " / ".join(f1(v) for v in c["vals"])
            out.append(f"| {tag}{i} | {label} | {vals} | {f1(c['median'])} |")
    return "\n".join(out)


def render_summary(ab):
    p = ab["per"]
    out = ["| 臂 | n | min | max | 中位 | range% | sd |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, label in (("OFF", "关"), ("ON", "开")):
        x = p[arm]
        out.append(f"| {label} | {x['n']} | {f1(x['min'])} | {f1(x['max'])} | "
                   f"**{f1(x['median'])}** | {x['range_pct']:.3f} | {x['sd']:.3f} |")
    return "\n".join(out)


def render_ab_conclusions(ab):
    p = ab["per"]
    return "\n".join([
        f"- **保守收益 = {f1(p['ON']['median'])} / {f1(p['OFF']['median'])} − 1 = "
        f"+{ab['gain']:.2f}%**（绝对 +{p['ON']['median'] - p['OFF']['median']:.1f} t/s）",
        f"- **控制漂移**：三个关周期中位 "
        f"{' / '.join(f1(ab['cycles'][f'OFF{i}']['median']) for i in (1,2,3))}，"
        f"跨度 **{ab['spread_off']:.3f}%**；三个开周期 "
        f"{' / '.join(f1(ab['cycles'][f'ON{i}']['median']) for i in (1,2,3))}，"
        f"跨度 **{ab['spread_on']:.3f}%**。**无任何周期主导结果。**",
        f"- **两个独立估计量**：合并中位比 **+{ab['gain']:.2f}%**；"
        f"逐周期配对 {' / '.join(f'+{x:.2f}%' for x in ab['paired'])}，"
        f"均值 **+{st.mean(ab['paired']):.2f}%**。两者相差 "
        f"{abs(st.mean(ab['paired']) - ab['gain']):.2f} 个百分点。",
        f"- **噪声下限**：关 sd {p['OFF']['sd']:.3f}、开 sd {p['ON']['sd']:.3f}（n={p['OFF']['n']}），"
        f"均值差 {ab['diff']:.2f} t/s，标准误 {ab['se']:.3f}，**z ≈ {ab['z']:.0f}**。"
        f"这度量的是「run-to-run 重复性」，不覆盖系统性误差——**可重复 ≠ 正确**。",
        f"- **计时交叉校验**：客户端 wall 中位 {ab['wall']['OFF']['wall']:.2f} s（关）/ "
        f"{ab['wall']['ON']['wall']:.2f} s（开），服务端自报 "
        f"{ab['wall']['OFF']['server']:.2f} / {ab['wall']['ON']['server']:.2f}，"
        f"开销 {ab['wall']['OFF']['overhead']:.2f} / {ab['wall']['ON']['overhead']:.2f} s。",
    ])


def render_equality(ab):
    t = ab["fixed_prompt_tok"]
    one = len(t) == 1
    rows = ["| 臂 | prompt tok | completion tok | sha256 前 16 |",
            "|---|---:|---:|---|"]
    for arm, label in (("OFF", "关"), ("ON", "开")):
        rows.append(f"| {label} | {','.join(str(x) for x in t) if t else '—'} | "
                    f"{','.join(str(x) for x in ab['fixed_completion'])} | "
                    f"`{ab['shas'][arm][0]}` |")
    verdict = ("**逐字节相同**" if ab["shas"]["OFF"] == ab["shas"]["ON"] and
               len(ab["shas"]["OFF"]) == 1 else "**不一致**")
    return "\n".join(rows) + f"\n\n两臂输出 {verdict}" + \
        (f"（prompt {t[0]} tok 一致）。" if one else f"（prompt tok 有 {len(t)} 种取值）。")


def render_separation(ab):
    d = ab["deltas"]
    rows = ["| 臂 | 周期 1 | 周期 2 | 周期 3 |", "|---|---:|---:|---:|"]
    for arm, label in (("OFF", "关"), ("ON", "开")):
        req = [x["requests"] for x in d[arm]]
        enc = [x["encoded"] for x in d[arm]]
        rows.append(f"| {label} `requests` | {' | '.join(str(x) for x in req)} |")
        rows.append(f"| {label} `encoded` | {' | '.join('**'+str(x)+'**' for x in enc)} |")
    fb = sorted({(x["fallbacks"], x["failures"]) for arm in d for x in d[arm]})
    enc_on = {x["encoded"] for x in d["ON"]}
    req_same = len({x["requests"] for arm in d for x in d[arm]}) == 1
    rate = 100 * list(enc_on)[0] / d["ON"][0]["requests"] if enc_on else 0
    rows.append(f"| 两臂 `fallbacks` / `failures` | "
                f"{' / '.join(f'{a} / {b}' for a, b in fb)} |")
    note = (f"\n\n两臂每周期 `requests` "
            f"{'完全相同' if req_same else '**不同**'}，唯一变量是 `encoded`。"
            f"开臂编码率 **{rate:.1f}%**。")
    return "\n".join(rows) + note


def render_gpu(g):
    rows = ["| 项 | 关 | 开 |", "|---|---:|---:|"]
    rows.append(f"| 活跃驻留 中位 / 最大 | **{g['OFF']['res_median']:.1f}% / "
                f"{g['OFF']['res_max']:.1f}%** | **{g['ON']['res_median']:.1f}% / "
                f"{g['ON']['res_max']:.1f}%** |")
    rows.append(f"| 出现过的最高频点 | {max(g['OFF']['mhz_mode'])} MHz | "
                f"{max(g['ON']['mhz_mode'])} MHz |")
    rows.append(f"| 功耗 中位 / 最大 | **{g['OFF']['power_median']/1000:.1f} W / "
                f"{g['OFF']['power_max']/1000:.1f} W** | **{g['ON']['power_median']/1000:.1f} W / "
                f"{g['ON']['power_max']/1000:.1f} W** |")
    rows.append(f"| 采样点数 | {g['OFF']['samples']} | {g['ON']['samples']} |")
    rows.append(f"| 本次 prefill 吞吐 | {g['OFF']['server']['prefill_tps']:.1f} t/s | "
                f"{g['ON']['server']['prefill_tps']:.1f} t/s |")
    return "\n".join(rows)


ORDER = ["p0.2k", "p1.6k", "p6.5k", "p13k", "p26k", "p52k"]
LABELS = {"p0.2k": "0.2K", "p1.6k": "1.6K", "p6.5k": "6.5K", "p13k": "13K",
          "p26k": "26K", "p52k": "52K"}


def render_sweep(sw):
    rows = ["| 语料 | prompt tok | 时延 | 中位 t/s | 范围 |", "|---|---:|---:|---:|---|"]
    for case in ORDER:
        v = sw[case]
        tp = sorted(x["prefill_tps"] for x in v)
        rows.append(f"| {LABELS[case]} | {v[0]['prompt_tok']} | "
                    f"{st.median([x['prefill_s'] for x in v]):.1f} s | "
                    f"**{st.median(tp):.1f}** | {tp[0]:.1f} – {tp[-1]:.1f} |")
    return "\n".join(rows)


def render_decode(sw, dec):
    rows = ["| 语料 | 上下文 tok | 生成 tok | 中位 t/s | 范围 | ms/token |",
            "|---|---:|---:|---:|---|---:|"]
    short = sw.get("d_short", [])
    if short:
        tp = sorted(x["decode_tps"] for x in short)
        ms = 1000 / st.median(tp)
        rows.append(f"| 短 | {short[0]['prompt_tok']} | {short[0]['decode_tok']} | "
                    f"**{st.median(tp):.1f}** | — | {ms:.2f} |")
    for case, label in (("d_8k", "8K"), ("d_32k", "32K")):
        v = dec.get(case, [])
        if not v:
            continue
        tp = sorted(x["decode_tps"] for x in v)
        ms = sorted(1000 / x for x in tp)
        rows.append(f"| {label} | {v[0]['prompt_tok']} | {v[0]['decode_tok']} | "
                    f"**{st.median(tp):.1f}** | {tp[0]:.1f} – {tp[-1]:.1f} | "
                    f"{ms[0]:.2f} – {ms[-1]:.2f} |")
    return "\n".join(rows)


def render_sweep_fit(sw):
    """Two-point fit over the two largest sizes, with residuals. Marked as inference
    in the ledger: a linear model is not a mechanism, it is a shape observation."""
    big = [c for c in ("p26k", "p52k") if c in sw]
    if len(big) < 2:
        return "_数据不足，未拟合。_"
    pts = [(sw[c][0]["prompt_tok"], st.median([x["prefill_s"] for x in sw[c]])) for c in big]
    (n1, t1), (n2, t2) = pts
    b = (t2 - t1) / (n2 - n1)
    a = t1 - n1 * b
    rows = [f"**[推断]** 以最大两点（{n1:,}→{t1:.1f} s、{n2:,}→{t2:.1f} s）做线性拟合："
            f"固定成本 ≈ **{a:.2f} s**、渐近 ≈ **{1/b:.0f} t/s**。回算："]
    res = []
    for c in ORDER:
        if c not in sw:
            continue
        tok = sw[c][0]["prompt_tok"]
        actual = st.median([x["prefill_s"] for x in sw[c]])
        pred = a + tok * b
        res.append(f"{tok:,} 预测 {pred:.2f} s（实测 {actual:.1f}，{100*(pred-actual)/actual:+.1f}%）")
    rows.append("、".join(res) + "。")
    rows.append("**线性模型只在大 n 区间成立，短 prompt 的固定成本另有来源，本轮未定位。**")
    return "\n".join(rows)


def render_tolerances():
    t = TOLERANCES
    rows = ["| 量 | 容差 | 依据 |", "|---|---|---|",
            f"| prefill / decode 吞吐 | **±{t['tps_pct']:.1f}%** | "
            f"单会话内实测离散 ~0.1%；跨会话需覆盖热漂移与页缓存状态 |",
            f"| A/B 收益 | **±{t['gain_pp']:.1f} 个百分点** | 是两个 ±{t['tps_pct']:.1f}% 量之差的组合，误差放大 |",
            f"| prompt token 数 | **±{t['prompt_tok_pct']:.1f}%** | 冻结语料的标签与历史跑批的 nonce 不同 |",
            "| 两臂输出 | **完全相等** | 等价性主张，不是性能量，没有容差 |",
            f"| GPU 活跃驻留 | **≥{t['gpu_residency_min']:.0f}%** | 低于此值时比较无效（一臂在等），不是「更快」 |",
            f"| GPU 功耗 | **{t['gpu_power_w'][0]:.0f}–{t['gpu_power_w'][1]:.0f} W** | "
            f"越界说明负载形态变了 |"]
    return "\n".join(rows)


def build_expected(ab, sw, dec, g, smoke):
    e = json.loads((BENCH / "expected.json").read_text())
    e["tolerances"] = dict(TOLERANCES, why=e["tolerances"].get("why", ""))
    e["ab"] = {
        "case": "p32k_ab",
        "arms": {a: {"median_tps": round(ab["per"][a]["median"], 1),
                     "min_tps": round(ab["per"][a]["min"], 1),
                     "max_tps": round(ab["per"][a]["max"], 1),
                     "n": ab["per"][a]["n"]} for a in ("OFF", "ON")},
        "gain_pct": round(ab["gain"], 2),
        "paired_gain_pct": round(st.mean(ab["paired"]), 2),
        "control_drift_pct": {"OFF": round(ab["spread_off"], 3),
                              "ON": round(ab["spread_on"], 3)},
        "requests_per_boot": ab["deltas"]["ON"][0]["requests"],
        "encoded_per_boot": {a: ab["deltas"][a][0]["encoded"] for a in ("OFF", "ON")},
        "encoding_rate_pct": round(100 * ab["deltas"]["ON"][0]["encoded"] /
                                   ab["deltas"]["ON"][0]["requests"], 1),
        "fixed_sample": {
            "prompt_tok_reference": ab["fixed_prompt_tok"][0] if ab["fixed_prompt_tok"] else None,
            "prompt_tok_tolerance_pct": TOLERANCES["prompt_tok_pct"],
            "completion_tok": ab["fixed_completion"][0] if ab["fixed_completion"] else None,
            "sha16": ab["shas"]["OFF"][0] if ab["shas"]["OFF"] else None,
            "claim": "the two arms must produce identical bytes on this sample",
        },
    }
    e["sweep"] = {c: {"prompt_tok_reference": sw[c][0]["prompt_tok"],
                      "median_prefill_tps": round(st.median([x["prefill_tps"] for x in sw[c]]), 1)}
                  for c in ORDER if c in sw}
    e["decode"] = {}
    for case in ("d_short",):
        if case in sw:
            e["decode"][case] = {"prompt_tok_reference": sw[case][0]["prompt_tok"],
                                 "decode_tokens": sw[case][0]["decode_tok"],
                                 "median_decode_tps": round(st.median(
                                     [x["decode_tps"] for x in sw[case]]), 1)}
    for case in ("d_8k", "d_32k"):
        if case in dec:
            e["decode"][case] = {"prompt_tok_reference": dec[case][0]["prompt_tok"],
                                 "decode_tokens": dec[case][0]["decode_tok"],
                                 "median_decode_tps": round(st.median(
                                     [x["decode_tps"] for x in dec[case]]), 1)}
    e["gpu"] = {
        "residency_median_pct": g["ON"]["res_median"],
        "frequency_peak_mhz": max(max(g["OFF"]["mhz_mode"]), max(g["ON"]["mhz_mode"])),
        "power_median_w": {a: round(g[a]["power_median"] / 1000, 1) for a in ("OFF", "ON")},
        "claim": e["gpu"].get("claim", ""),
    }
    if smoke:
        e["l3_smoke"] = dict(source="raw/smoke2.jsonl", arm=smoke["arms"][0],
                             fixed_tps=smoke["fixed"][0] if smoke["fixed"] else None,
                             nonce_tps=smoke["nonce"],
                             gpu_residency=smoke["res"][0] if smoke["res"] else None,
                             gpu_power_w=smoke["power"],
                             all_cached_zero=all(c == 0 for c in smoke["cached"]))
    return e


def regions(ab, sw, dec, g):
    return {
        "ab_cycles": render_cycle_table(ab),
        "ab_summary": render_summary(ab),
        "ab_conclusions": render_ab_conclusions(ab),
        "ab_equality": render_equality(ab),
        "ab_separation": render_separation(ab),
        "gpu": render_gpu(g),
        "sweep": render_sweep(sw),
        "decode": render_decode(sw, dec),
        "sweep_fit": render_sweep_fit(sw),
        "tolerances": render_tolerances(),
    }


def apply(text, regs):
    out = text
    for name, body in regs.items():
        begin = f"<!-- BEGIN GENERATED:{name} -->"
        end = f"<!-- END GENERATED:{name} -->"
        i, j = out.find(begin), out.find(end)
        if i < 0 or j < 0:
            sys.exit(f"marker missing: {name}")
        out = out[:i] + begin + "\n" + body + "\n" + out[j:]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (a.write or a.check):
        ap.error("give --write or --check")

    ab, sw, dec, g, smoke = ab_facts(), sweep_facts(), dec_facts(), gpu_facts(), smoke_facts()
    if ab["cached_nonzero"]:
        sys.exit(f"the raw A/B rows contain {ab['cached_nonzero']} cache hits; "
                 f"they did no prefill and the tables would be wrong")

    regs = regions(ab, sw, dec, g)
    now = LEDGER.read_text()
    want = apply(now, regs)
    exp_now = (BENCH / "expected.json").read_text()
    exp_want = json.dumps(build_expected(ab, sw, dec, g, smoke),
                          ensure_ascii=False, indent=2) + "\n"

    if a.check:
        stale_ledger = now != want
        stale_exp = exp_now != exp_want
        print(f"ledger tables: {'STALE' if stale_ledger else 'up to date'}")
        print(f"expected.json: {'STALE' if stale_exp else 'up to date'}")
        if stale_ledger or stale_exp:
            print("\nregenerate with: python3 tools/gen_ledger_tables.py --write")
            return 1
        return 0

    if now != want:
        LEDGER.write_text(want)
        print("ledger tables regenerated")
    if exp_now != exp_want:
        (BENCH / "expected.json").write_text(exp_want)
        print("expected.json regenerated")
    if now == want and exp_now == exp_want:
        print("nothing to do, already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
