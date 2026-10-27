#!/usr/bin/env python3
"""Independent audit: recompute every performance number in ledger §105 from the
raw JSONL, and diff it against what the ledger actually says.

This does not trust any earlier aggregation script. It re-derives:
  * A/B medians, min, max, range, per-cycle medians  (rig.jsonl)
  * gain% and absolute delta
  * 0-ULP claim (fixed-prompt rows: text sha, prompt tokens, completion tokens)
  * arm separation (encoded deltas per boot, requests equality)
  * sweep table (sweep2.jsonl)
  * decode table (sweep2.jsonl d_short + dec2.jsonl)
  * GPU evidence (sweep2.jsonl gpu rows)
then greps the ledger's §105 for each expected figure and reports MISSING/WRONG.

Exit code 1 if any figure in the ledger disagrees with the recomputation.
"""
import json, re, statistics as st, sys
from pathlib import Path

RAW = Path.home() / "Projects/myds/benchmarks/register_only_2026-10-27/raw"
LEDGER = Path.home() / "Projects/myds/docs/2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md"

# Fail loudly and specifically if the evidence is not on this machine. An audit that
# silently skips missing data is worse than no audit.
for p in (RAW / "rig.jsonl", RAW / "sweep2.jsonl", RAW / "dec2.jsonl", LEDGER):
    if not p.exists():
        sys.exit(f"MISSING EVIDENCE: {p}\n"
                 f"The §105 numbers cannot be checked without it. It ships with this "
                 f"repository under benchmarks/register_only_2026-10-27/raw/; if it is "
                 f"absent, the checkout is incomplete rather than the numbers wrong.")

problems, checks = [], []


def chk(name, ok, detail=""):
    checks.append((name, ok, detail))
    if not ok:
        problems.append(f"{name}: {detail}")


rig = [json.loads(l) for l in (RAW / "rig.jsonl").read_text().splitlines() if l.strip()]
sw = [json.loads(l) for l in (RAW / "sweep2.jsonl").read_text().splitlines() if l.strip()]
dc = [json.loads(l) for l in (RAW / "dec2.jsonl").read_text().splitlines() if l.strip()]

ledger = LEDGER.read_text()
sec = ledger[ledger.index("## §105"):]

# ---------- A/B ----------
runs = [r for r in rig if r.get("kind") == "run"]
nonce = [r for r in runs if r.get("sample") == "nonce"]
fixed = [r for r in runs if r.get("sample") == "fixed"]

chk("A/B nonce rows == 18", len(nonce) == 18, f"got {len(nonce)}")
chk("A/B fixed rows == 6", len(fixed) == 6, f"got {len(fixed)}")
chk("all runs cached==0", all(r.get("cached") == 0 for r in runs),
    f"non-zero: {[(r['arm'], r['boot'], r.get('cached')) for r in runs if r.get('cached') != 0]}")

per = {}
for arm in ("OFF", "ON"):
    v = sorted(r["tps"] for r in nonce if r["arm"] == arm)
    per[arm] = dict(n=len(v), min=v[0], max=v[-1], median=round(st.median(v), 2),
                    range_pct=round(100 * (v[-1] - v[0]) / st.median(v), 3))
    chk(f"{arm} n==9", len(v) == 9, f"got {len(v)}")

off, on = per["OFF"], per["ON"]
chk("OFF median 593.1", abs(off["median"] - 593.1) < 0.05, str(off["median"]))
chk("ON median 618.9", abs(on["median"] - 618.9) < 0.05, str(on["median"]))
chk("OFF range 0.135%", abs(off["range_pct"] - 0.135) < 0.005, str(off["range_pct"]))
chk("ON range 0.065%", abs(on["range_pct"] - 0.065) < 0.005, str(on["range_pct"]))
gain = 100 * (on["median"] / off["median"] - 1)
chk("gain 4.35%", abs(gain - 4.35) < 0.01, f"{gain:.4f}")
chk("abs delta 25.8", abs((on["median"] - off["median"]) - 25.8) < 0.05,
    f"{on['median'] - off['median']:.2f}")

# per-cycle medians
cyc = {}
for arm in ("OFF", "ON"):
    for b in sorted({r["boot"] for r in nonce if r["arm"] == arm}):
        v = [r["tps"] for r in nonce if r["arm"] == arm and r["boot"] == b]
        cyc[f"{arm}{b}"] = round(st.median(v), 1)
expect_cyc = {"OFF1": 593.2, "ON1": 618.8, "OFF2": 593.3, "ON2": 619.1,
              "OFF3": 592.9, "ON3": 618.8}
for k, want in expect_cyc.items():
    chk(f"cycle {k} == {want}", abs(cyc.get(k, 0) - want) < 0.05, str(cyc.get(k)))

# control drift = spread of per-arm cycle medians
def spread(vals):
    return round(100 * (max(vals) - min(vals)) / st.median(vals), 3)
off_cyc = [cyc[f"OFF{i}"] for i in (1, 2, 3)]
on_cyc = [cyc[f"ON{i}"] for i in (1, 2, 3)]
chk("OFF cycle spread 0.067%", abs(spread(off_cyc) - 0.067) < 0.01, f"{spread(off_cyc)}")
chk("ON cycle spread 0.048%", abs(spread(on_cyc) - 0.048) < 0.01, f"{spread(on_cyc)}")

# ---------- 0-ULP ----------
shas = {a: sorted({r["sha"] for r in fixed if r["arm"] == a}) for a in ("OFF", "ON")}
chk("fixed sha identical", shas["OFF"] == shas["ON"] and len(shas["OFF"]) == 1, str(shas))
chk("fixed sha == 0d99c08267aa96cb", shas["OFF"] == ["0d99c08267aa96cb"], str(shas["OFF"]))
ptok = sorted({r["prompt_tok"] for r in fixed})
ctok = sorted({r["completion_tok"] for r in fixed})
chk("fixed prompt_tok == [32210]", ptok == [32210], str(ptok))
chk("fixed completion_tok == [2]", ctok == [2], str(ctok))
texts = sorted({r["text"] for r in fixed})
chk("fixed text identical", len(texts) == 1, str(texts))

# ---------- arm separation ----------
ends = [r for r in rig if r.get("kind") == "boot_end"]
for arm in ("OFF", "ON"):
    d = [e["delta"] for e in ends if e["arm"] == arm]
    chk(f"{arm} has 3 boot_end", len(d) == 3, str(len(d)))
    chk(f"{arm} requests all 2897", all(x["requests"] == 2897 for x in d), str([x["requests"] for x in d]))
    enc = [x["encoded"] for x in d]
    want = [0, 0, 0] if arm == "OFF" else [2025, 2025, 2025]
    chk(f"{arm} encoded == {want}", enc == want, str(enc))
    chk(f"{arm} fallbacks/failures 0",
        all(x["fallbacks"] == 0 and x["failures"] == 0 for x in d), str(d))
chk("encoding rate 69.9%", abs(100 * 2025 / 2897 - 69.9) < 0.05, f"{100*2025/2897:.2f}")

# ---------- wall vs server ----------
for arm, want_wall, want_srv, want_oh in (("OFF", 54.47, 54.30, 0.16), ("ON", 52.21, 52.10, 0.11)):
    pairs = [(r["wall_s"], r["prefill_s"]) for r in nonce if r["arm"] == arm]
    w = round(st.median([p[0] for p in pairs]), 2)
    s = round(st.median([p[1] for p in pairs]), 2)
    o = round(st.median([p[0] - p[1] for p in pairs]), 2)
    chk(f"{arm} wall {want_wall}", abs(w - want_wall) < 0.01, str(w))
    chk(f"{arm} server {want_srv}", abs(s - want_srv) < 0.01, str(s))
    chk(f"{arm} overhead {want_oh}", abs(o - want_oh) < 0.02, str(o))

# ---------- GPU ----------
gpu = {r["arm"]: r for r in sw if r.get("phase") == "gpu"}
for arm in ("OFF", "ON"):
    g = gpu[arm]
    chk(f"gpu {arm} samples 58", g["samples"] == 58, str(g["samples"]))
    chk(f"gpu {arm} res_median 100.0", g["res_median"] == 100.0, str(g["res_median"]))
    chk(f"gpu {arm} 1398MHz present", 1398 in g["mhz_mode"], str(g["mhz_mode"]))
chk("gpu OFF power 104.2W", abs(gpu["OFF"]["power_median"] / 1000 - 104.2) < 0.05,
    str(gpu["OFF"]["power_median"] / 1000))
chk("gpu ON power 106.0W", abs(gpu["ON"]["power_median"] / 1000 - 106.0) < 0.05,
    str(gpu["ON"]["power_median"] / 1000))
chk("gpu OFF power_max 114.2W", abs(gpu["OFF"]["power_max"] / 1000 - 114.2) < 0.05,
    str(gpu["OFF"]["power_max"] / 1000))
chk("gpu ON power_max 114.6W", abs(gpu["ON"]["power_max"] / 1000 - 114.6) < 0.05,
    str(gpu["ON"]["power_max"] / 1000))

# ---------- sweep ----------
swr = [r for r in sw if r.get("phase") == "sweep" and r.get("server")]
by = {}
for r in swr:
    by.setdefault(r["case"], []).append(r["server"])
expect_sweep = {"p0.2k": (592, 246.8), "p1.6k": (1026, 319.0), "p6.5k": (4059, 453.6),
                "p13k": (8081, 539.5), "p26k": (16128, 592.7), "p52k": (32224, 616.4)}
for case, (tok, want) in expect_sweep.items():
    v = by.get(case, [])
    chk(f"sweep {case} n==3", len(v) == 3, str(len(v)))
    chk(f"sweep {case} tokens=={tok}", v and v[0]["prompt_tok"] == tok, str(v[0]["prompt_tok"]) if v else "none")
    m = round(st.median([x["prefill_tps"] for x in v]), 1) if v else None
    chk(f"sweep {case} median {want}", m == want, str(m))
chk("sweep all cached==0", all(x["cached"] == 0 for v in by.values() for x in v),
    str([(k, x["cached"]) for k, v in by.items() for x in v if x["cached"] != 0]))

# d_short decode
ds = by.get("d_short", [])
chk("d_short n==3", len(ds) == 3, str(len(ds)))
chk("d_short dtok==256", ds and ds[0]["decode_tok"] == 256, str(ds[0]["decode_tok"]) if ds else "none")
chk("d_short decode 42.0", round(st.median([x["decode_tps"] for x in ds]), 1) == 42.0,
    str(round(st.median([x["decode_tps"] for x in ds]), 1)) if ds else "none")

# ---------- dec2 ----------
dby = {}
for r in dc:
    if r.get("server"):
        dby.setdefault(r["case"], []).append(r["server"])
for case, tok, want in (("d_8k", 8111, 39.6), ("d_32k", 32254, 38.2)):
    v = dby.get(case, [])
    chk(f"dec2 {case} n==3", len(v) == 3, str(len(v)))
    chk(f"dec2 {case} tokens=={tok}", v and v[0]["prompt_tok"] == tok, str(v[0]["prompt_tok"]) if v else "none")
    chk(f"dec2 {case} dtok==160", v and v[0]["decode_tok"] == 160, str(v[0]["decode_tok"]) if v else "none")
    m = round(st.median([x["decode_tps"] for x in v]), 1) if v else None
    chk(f"dec2 {case} decode {want}", m == want, str(m))

# ms/token ranges
ms8 = sorted(round(1000 / x["decode_tps"], 2) for x in dby["d_8k"])
ms32 = sorted(round(1000 / x["decode_tps"], 2) for x in dby["d_32k"])
chk("d_8k ms range 25.13-25.51", (ms8[0], ms8[-1]) == (25.13, 25.51), str((ms8[0], ms8[-1])))
chk("d_32k ms range 26.11-26.25", (ms32[0], ms32[-1]) == (26.11, 26.25), str((ms32[0], ms32[-1])))

# paired estimator must agree with the pooled one
per_cycle = {}
for arm in ("OFF", "ON"):
    for b in (1, 2, 3):
        per_cycle[(arm, b)] = st.median([r["tps"] for r in nonce if r["arm"] == arm and r["boot"] == b])
paired = [100 * (per_cycle[("ON", b)] / per_cycle[("OFF", b)] - 1) for b in (1, 2, 3)]
chk("paired gains ~4.32/4.35/4.37",
    [round(x, 2) for x in paired] == [4.32, 4.35, 4.37], str([round(x, 2) for x in paired]))
chk("paired mean 4.34", abs(st.mean(paired) - 4.34) < 0.01, f"{st.mean(paired):.3f}")
chk("pooled and paired agree within 0.02pp", abs(st.mean(paired) - gain) < 0.02,
    f"paired {st.mean(paired):.3f} vs pooled {gain:.3f}")

# noise floor
sd_off = st.stdev([r["tps"] for r in nonce if r["arm"] == "OFF"])
sd_on = st.stdev([r["tps"] for r in nonce if r["arm"] == "ON"])
chk("sd_off 0.235", abs(sd_off - 0.235) < 0.005, f"{sd_off:.3f}")
chk("sd_on 0.169", abs(sd_on - 0.169) < 0.005, f"{sd_on:.3f}")
diff = st.mean([r["tps"] for r in nonce if r["arm"] == "ON"]) - \
       st.mean([r["tps"] for r in nonce if r["arm"] == "OFF"])
se = (sd_off ** 2 / 9 + sd_on ** 2 / 9) ** 0.5
chk("mean diff 25.76", abs(diff - 25.76) < 0.02, f"{diff:.3f}")
chk("sem 0.097", abs(se - 0.097) < 0.002, f"{se:.4f}")
chk("z about 267", abs(diff / se - 267) < 2, f"{diff/se:.1f}")
chk("ledger states repeatability != validity", "可重复 ≠ 正确" in sec, "missing caveat")

# ---------- post-run GPU snapshots: the corrected claim ----------
# The retracted draft said these were "always 0 MHz / 0 mW". They were not: they read
# drain residue. The ledger must list the actual distinct tuples and must not repeat
# the word "constant zero".
post = sorted({(r["gpu"].get("mhz"), r["gpu"].get("residency"), r["gpu"].get("power_mw"))
               for r in runs})
chk("post-run snapshots are not all zero", any(v != (0, 0.0, 0) for v in post), str(post))
chk("post-run residency max 2.31%", max(v[1] for v in post) == 2.31,
    str(sorted({v[1] for v in post})))
chk("post-run power max 386 mW", max(v[2] for v in post) == 386,
    str(sorted({v[2] for v in post})))
# the phrase may appear exactly once, and only inside the retraction itself
stale = [m.start() for m in re.finditer("恒为 0 mW", sec)]
chk("constant-zero claim only appears as a retraction",
    len(stale) <= 1 and (not stale or "旧写法" in sec[max(0, stale[0] - 80):stale[0] + 80]),
    f"occurrences={len(stale)}")
chk("ledger no longer asserts constant-zero readout", "恒读出 0 MHz" not in sec, "stale wording present")
for mw in ("48 mW", "386 mW", "270 mW", "203 mW", "256 mW"):
    chk(f"ledger lists post-run residue {mw}", mw in sec, "missing")

# ---------- linear fit on the two largest prefill points ----------
p26 = st.median([x["prefill_s"] for x in by["p26k"]])
p52 = st.median([x["prefill_s"] for x in by["p52k"]])
b = (p52 - p26) / (32224 - 16128)
a = p26 - 16128 * b
chk("fit fixed cost 2.05 s", abs(a - 2.05) < 0.02, f"{a:.4f}")
chk("fit asymptotic 641 t/s", abs(1 / b - 641.3) < 1.0, f"{1/b:.1f}")
for tok, actual, want_resid in ((8081, 15.0, -2.3), (4059, 8.9, -5.9), (1026, 3.2, 14.1), (592, 2.4, 23.9)):
    pred = a + tok * b
    resid = 100 * (pred - actual) / actual
    chk(f"fit residual {tok} -> {want_resid}%", abs(resid - want_resid) < 0.6, f"{resid:.2f}%")
chk("ledger has sweep 16,128 at 27.2 s", "| 16,128 | 27.2 s" in sec, "wrong time cell")
chk("ledger free of 27.3 s", "27.3 s" not in sec, "stale cell present")
chk("ledger marks the overhead model as inference", "**[推断]**" in sec, "no inference marker")
chk("ledger no longer claims 1.6-2.0 s overhead", "1.6–2.0 s" not in sec, "unsourced range present")

# the outlier must be described as a *false slowdown*, not a false gain
chk("outlier framed as -2.42%", "−2.42%" in sec, "wrong sign or value")
chk("ledger does not claim the outlier yields +2.4%", "~+2.4%" not in sec, "sign error present")
chk("outlier recomputation holds", abs(100 * (577.4 / 591.7 - 1) + 2.42) < 0.01,
    f"{100*(577.4/591.7-1):.2f}")

# decode delta must state both units, since ms and t/s move in opposite directions
chk("decode row gives ms delta", "+4.7%" in sec, "ms delta missing")
chk("decode row gives throughput delta", "−4.5%" in sec, "throughput delta missing")
chk("ms/throughput deltas are consistent",
    abs(100 * (23.81 / 22.74 - 1) - 4.71) < 0.01 and abs(100 * (22.74 / 23.81 - 1) + 4.49) < 0.01,
    f"{100*(23.81/22.74-1):.2f} / {100*(22.74/23.81-1):.2f}")

# RSS is a fluctuating observation, not a constant
chk("RSS stated as a range", "151–154 GB" in sec, "point value presented as constant")

# ---------- ledger cross-check ----------
must_appear = [
    ("593.1", "OFF median"), ("618.9", "ON median"), ("+4.35%", "gain"),
    ("0.135", "OFF range"), ("0.065", "ON range"), ("25.8", "abs delta"),
    ("0d99c08267aa96cb", "sha"), ("2,025", "encoded on"), ("2,897", "requests"),
    ("69.9", "encoding rate"), ("100.0", "gpu residency"),
    ("104.2", "power off"), ("106.0", "power on"),
    ("246.8", "sweep 592"), ("319.0", "sweep 1026"), ("453.6", "sweep 4059"),
    ("539.5", "sweep 8081"), ("592.7", "sweep 16128"), ("616.4", "sweep 32224"),
    ("42.0", "decode short"), ("39.6", "decode 8k"), ("38.2", "decode 32k"),
    ("23.81", "ms short"), ("25.13", "ms 8k low"), ("26.25", "ms 32k high"),
    ("32,210", "fixed prompt tokens"),
]
# numeric figures are compared with thousands separators stripped from both sides,
# because the ledger mixes "32,210" and "2897" styles. The value is what matters.
sec_norm = sec.replace(",", "")
for needle, label in must_appear:
    ok = needle in sec or needle.replace(",", "") in sec_norm
    chk(f"ledger has {label} ({needle})", ok, "not found in §105")

# retracted figures may appear only inside an explicit retraction sentence
for bad, label, marker in (("591.7", "old OFF median", "577.4"),
                           ("617.7", "old ON median", "old"),
                           ("4.39", "old gain", "旧"),
                           ("1,620", "old encoded", "旧")):
    idx = sec.find(bad)
    ok = bad not in sec and bad.replace(",", "") not in sec_norm
    if not ok and idx >= 0:
        ctx = sec[max(0, idx - 150):idx + 150]
        ok = marker in ctx
    chk(f"ledger free of retracted {label}", ok, "present outside a retraction context")

# ---------- §106: reproducibility section must agree with expected.json ----------
ledger_all = ledger
try:
    sec106 = ledger_all[ledger_all.index("## §106"):]
except ValueError:
    sec106 = ""
chk("ledger has a §106 reproducibility section", bool(sec106), "missing")

if sec106:
    expj = json.loads((Path.home() / "Projects/myds/benchmarks/register_only_2026-10-27/"
                       "expected.json").read_text())
    tol = expj["tolerances"]

    chk("106 quotes tps tolerance", f"±{tol['tps_pct']:.1f}%" in sec106,
        f"expected ±{tol['tps_pct']}%")
    chk("106 quotes gain tolerance", f"±{tol['gain_pp']:.1f} 个百分点" in sec106,
        f"expected ±{tol['gain_pp']}pp")
    chk("106 quotes prompt token tolerance", f"±{tol['prompt_tok_pct']:.1f}%" in sec106,
        f"expected ±{tol['prompt_tok_pct']}%")
    chk("106 quotes gpu residency min", f"≥{tol['gpu_residency_min']:.0f}%" in sec106,
        f"expected >= {tol['gpu_residency_min']}%")
    chk("106 quotes gpu power band",
        f"{tol['gpu_power_w'][0]:.0f}–{tol['gpu_power_w'][1]:.0f} W" in sec106,
        str(tol['gpu_power_w']))

    for name, needle in (("cached", "cached"), ("encoded", "encoded"),
                         ("fallbacks", "fallbacks")):
        chk(f"106 states the {name} invariant", needle in sec106, "not mentioned")

    for rel in ("benchmarks/register_only_2026-10-27/expected.json",
                "benchmarks/register_only_2026-10-27/README.md",
                "benchmarks/register_only_2026-10-27/raw/smoke2.jsonl",
                "docs/REPRODUCE-register-only.md"):
        chk(f"106 references exist: {rel}",
            (Path.home() / "Projects/myds" / rel).exists(), "missing file")

    smoke_rows = [json.loads(l) for l in
                  (Path.home() / "Projects/myds/benchmarks/register_only_2026-10-27/raw/"
                   "smoke2.jsonl").read_text().splitlines() if l.strip()]
    sruns = [r for r in smoke_rows if r.get("kind") == "run"]
    fx = [r["tps"] for r in sruns if r.get("sample") == "fixed"]
    nc = sorted(r["tps"] for r in sruns if r.get("sample") == "nonce")
    chk("106.5 smoke fixed tps matches data", any(abs(t - 592.2) < 0.05 for t in fx), str(fx))
    chk("106.5 smoke nonce tps matches data", nc == [592.8, 592.8], str(nc))
    chk("106.5 smoke fixed tps quoted", "592.2" in sec106, "missing")
    chk("106.5 smoke nonce tps quoted", "592.8 / 592.8" in sec106, "missing")
    chk("106 states the arithmetic check count", "147" in sec106, "missing")
    chk("106 lists exactly five explicit limits",
        sum(sec106.count(f"\n{i}. **不能") for i in range(1, 6)) == 5,
        "limit list changed")

print(f"checks: {len(checks)}  passed: {sum(1 for _, ok, _ in checks if ok)}  failed: {len(problems)}")
for n, ok, d in checks:
    if not ok:
        print(f"  FAIL {n} -> {d}")
print()
print("RECOMPUTED:")
print(json.dumps(dict(ab=per, gain_pct=round(gain, 3), abs_delta=round(on["median"] - off["median"], 2),
                      cycles=cyc, cycle_spread=dict(OFF=spread(off_cyc), ON=spread(on_cyc)),
                      fixed_sha=shas, sweep={k: round(st.median([x["prefill_tps"] for x in v]), 1)
                                             for k, v in by.items()},
                      dec2={k: round(st.median([x["decode_tps"] for x in v]), 1) for k, v in dby.items()}),
                 ensure_ascii=False, indent=1))
sys.exit(1 if problems else 0)
