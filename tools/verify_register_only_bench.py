#!/usr/bin/env python3
"""Compare a freshly measured run against the frozen expectations, and say PASS
or FAIL per claim rather than printing a table and letting the reader decide.

Two modes, both usable in CI:

    # against a real run
    python3 tools/verify_register_only_bench.py --ab /tmp/rig_run.jsonl \\
                                               --sweep /tmp/sweep_run.jsonl

    # self-test the verifier itself, no GPU needed
    python3 tools/verify_register_only_bench.py --historical

The historical mode validates this script against the archived JSONL that the
ledger's numbers came from. If the verifier can accept that data, it is at least
computing the right quantities; a verifier that has never been shown a passing
case is not a verifier.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "register_only_2026-10-27"
EXPECTED = BENCH / "expected.json"
HIST_AB = BENCH / "raw" / "rig.jsonl"
HIST_SWEEP = BENCH / "raw" / "sweep2.jsonl"
HIST_DEC = BENCH / "raw" / "dec2.jsonl"

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))


def near(got, want, pct):
    if want == 0:
        return got == 0
    return abs(got - want) <= abs(want) * pct / 100.0


def load(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"MISSING INPUT: {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def verify_ab(rows, exp, label=""):
    tol = exp["tolerances"]
    e = exp["ab"]
    print(f"\nA/B{(' — ' + label) if label else ''}")
    runs = [r for r in rows if r.get("kind") == "run"]
    nonce = [r for r in runs if r.get("sample") == "nonce"]
    fixed = [r for r in runs if r.get("sample") == "fixed"]
    if not runs:
        check("A/B rows present", False, "no rows with kind=run")
        return

    check("all rows reported cached=0", all(r.get("cached") == 0 for r in runs),
          str([(r["arm"], r["boot"], r.get("cached")) for r in runs if r.get("cached") != 0]))

    med = {}
    for arm in ("OFF", "ON"):
        v = sorted(r["tps"] for r in nonce if r["arm"] == arm)
        if len(v) < 6:
            check(f"{arm} has enough samples (>=6)", False, f"n={len(v)}")
            return
        med[arm] = st.median(v)
        want = e["arms"][arm]
        check(f"{arm} median t/s within {tol['tps_pct']}% of {want['median_tps']}",
              near(med[arm], want["median_tps"], tol["tps_pct"]),
              f"got {med[arm]:.1f}")
        check(f"{arm} spread is tight (<=1%)", (v[-1] - v[0]) / st.median(v) * 100 <= 1.0,
              f"{(v[-1]-v[0])/st.median(v)*100:.3f}%")

    gain = 100 * (med["ON"] / med["OFF"] - 1)
    check(f"gain within {tol['gain_pp']}pp of {e['gain_pct']}%",
          abs(gain - e["gain_pct"]) <= tol["gain_pp"], f"got {gain:.2f}%")
    check("gain is positive (the point of the experiment)", gain > 0, f"got {gain:.2f}%")

    # arm separation
    ends = [r for r in rows if r.get("kind") == "boot_end" and r.get("delta")]
    if ends:
        for arm in ("OFF", "ON"):
            d = [x["delta"] for x in ends if x["arm"] == arm]
            if not d:
                check(f"{arm} counter deltas present", False, "none")
                continue
            check(f"{arm} requests constant across boots",
                  len({x["requests"] for x in d}) == 1, str([x["requests"] for x in d]))
            want = e["encoded_per_boot"][arm]
            check(f"{arm} encoded ~ {want}",
                  all(abs(x["encoded"] - want) <= max(5, want * 0.05) for x in d),
                  str([x["encoded"] for x in d]))
            check(f"{arm} no fallbacks/failures",
                  all(x["fallbacks"] == 0 and x["failures"] == 0 for x in d), str(d))
    else:
        check("counter deltas present (needs sudo lldb)", False, "no boot_end rows with delta")

    # equality of the two arms on the same prompt
    if fixed:
        shas = {a: sorted({r["sha"] for r in fixed if r["arm"] == a}) for a in ("OFF", "ON")}
        check("both arms produced identical output bytes on the fixed prompt",
              shas["OFF"] == shas["ON"] and len(shas["OFF"]) == 1, str(shas))
        toks = sorted({r["prompt_tok"] for r in fixed if r.get("prompt_tok")})
        if toks:
            ref = e["fixed_sample"]["prompt_tok_reference"]
            check(f"fixed prompt tokenised near {ref}",
                  all(near(t, ref, e["fixed_sample"]["prompt_tok_tolerance_pct"]) for t in toks),
                  str(toks))
    else:
        check("fixed-prompt sample present", False, "none")

    # gpu, if sampled
    g = [r for r in runs if r.get("gpu_detail")]
    if g:
        for arm in ("OFF", "ON"):
            rows_a = [r["gpu_detail"] for r in g if r["arm"] == arm]
            if not rows_a:
                continue
            res = st.median([x["res_median"] for x in rows_a])
            check(f"{arm} GPU residency >= {tol['gpu_residency_min']}%",
                  res >= tol["gpu_residency_min"], f"{res:.1f}%")
            peak = max(max(x["mhz_modes"]) for x in rows_a)
            check(f"{arm} reached {exp['gpu']['frequency_peak_mhz']} MHz",
                  peak >= exp["gpu"]["frequency_peak_mhz"], f"{peak}")
            pw = st.median([x["power_median_mw"] for x in rows_a]) / 1000
            lo, hi = tol["gpu_power_w"]
            check(f"{arm} power median in {lo}-{hi} W", lo <= pw <= hi, f"{pw:.1f} W")


def verify_sweep(rows, exp, check_decode=True):
    tol = exp["tolerances"]
    print("\nSweep")
    sw = [r for r in rows if r.get("phase") == "sweep" and r.get("server")]
    if not sw:
        check("sweep rows present", False, "none")
        return
    check("all sweep rows cached=0", all(r["server"]["cached"] == 0 for r in sw),
          str([(r["case"], r["server"]["cached"]) for r in sw if r["server"]["cached"] != 0]))

    by = {}
    for r in sw:
        by.setdefault(r["case"], []).append(r["server"])

    for case, want in exp["sweep"].items():
        v = by.get(case, [])
        if len(v) < 2:
            check(f"{case} has >=2 reps", False, f"n={len(v)}")
            continue
        m = st.median([x["prefill_tps"] for x in v])
        check(f"{case} prefill median within {tol['tps_pct']}% of {want['median_prefill_tps']}",
              near(m, want["median_prefill_tps"], tol["tps_pct"]), f"got {m:.1f}")
        toks = sorted({x["prompt_tok"] for x in v})
        check(f"{case} prompt tokenised near {want['prompt_tok_reference']}",
              all(near(t, want["prompt_tok_reference"], tol["prompt_tok_pct"]) for t in toks),
              str(toks))

    if not check_decode:
        print("  [note] decode skipped in this file: the archived sweep generated only 2 "
              "tokens for d_8k/d_32k, so those rates are meaningless. dec2.jsonl replaces them.")
        return

    for case, want in exp["decode"].items():
        v = by.get(case, [])
        if not v:
            continue
        dtok = sorted({x["decode_tok"] for x in v})
        check(f"{case} generated about {want['decode_tokens']} tokens (else the rate is noise)",
              all(abs(t - want["decode_tokens"]) <= 16 for t in dtok), str(dtok))
        rates = [x["decode_tps"] for x in v if x["decode_tok"] > 8]
        if not rates:
            check(f"{case} has a usable decode rate", False, "no row generated >8 tokens")
            continue
        m = st.median(rates)
        check(f"{case} decode median within {tol['tps_pct']}% of {want['median_decode_tps']}",
              near(m, want["median_decode_tps"], tol["tps_pct"]), f"got {m:.1f}")


def verify_smoke(rows, exp):
    """The single-arm smoke run is not an A/B; it only shows the shipped harness
    works and produces the same order of numbers. Checking it keeps ledger 106.5
    from being another unsourced claim."""
    want = exp.get("l3_smoke")
    if not want:
        return
    print("\nL3 smoke (single arm, single boot)")
    runs = [r for r in rows if r.get("kind") == "run"]
    if not runs:
        check("smoke rows present", False, "none")
        return
    check("smoke arms are OFF only", {r["arm"] for r in runs} == {want["arm"]},
          str({r["arm"] for r in runs}))
    check("smoke rows all cached=0", all(r.get("cached") == 0 for r in runs),
          str([r.get("cached") for r in runs]))
    fx = [r for r in runs if r.get("sample") == "fixed"]
    check(f"smoke fixed tps ~ {want['fixed_tps']}",
          any(near(r["tps"], want["fixed_tps"], 1.0) for r in fx),
          str([r["tps"] for r in fx]))
    nc = sorted(r["tps"] for r in runs if r.get("sample") == "nonce")
    check("smoke nonce tps match",
          nc == sorted(want["nonce_tps"]), str(nc))
    g = [r["gpu_detail"] for r in runs if r.get("gpu_detail")]
    if g:
        check("smoke GPU residency 100%",
              all(x["res_median"] == want["gpu_residency"] for x in g),
              str([x["res_median"] for x in g]))
        pw = sorted(round(x["power_median_mw"] / 1000, 1) for x in g)
        check("smoke GPU power matches", pw == sorted(want["gpu_power_w"]), str(pw))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ab", help="A/B JSONL from bench_register_only_ab.py")
    ap.add_argument("--sweep", help="sweep JSONL from bench_register_only_sweep.py")
    ap.add_argument("--historical", action="store_true",
                    help="self-test against the archived data the ledger cites")
    ap.add_argument("--smoke", help="optional: a single-arm smoke JSONL to sanity-check")
    a = ap.parse_args()

    exp = json.loads(EXPECTED.read_text())

    if a.historical:
        print(f"self-test against archived data:\n  {HIST_AB}\n  {HIST_AB.parent / 'sweep2.jsonl'}")
        verify_ab(load(HIST_AB), exp)
        # the archived sweep records d_8k/d_32k with only 2 generated tokens, so its
        # decode columns are excluded here; dec2.jsonl is checked instead.
        verify_sweep(load(HIST_SWEEP), exp, check_decode=False)
        smoke = BENCH / "raw" / "smoke2.jsonl"
        if smoke.exists():
            verify_smoke(load(smoke), exp)
        # the end-to-end rerun by the shipped harness: same expectations, different run
        repro = BENCH / "raw" / "reproduce_run.jsonl"
        if repro.exists():
            print("\nA/B rerun by the shipped harness (raw/reproduce_run.jsonl)")
            verify_ab(load(repro), exp, label="rerun")
        # decode from dec2
        print("\nDecode (from dec2.jsonl)")
        dec = {}
        for r in load(HIST_DEC):
            if r.get("server"):
                dec.setdefault(r["case"], []).append(r["server"])
        for case, want in exp["decode"].items():
            v = dec.get(case, [])
            if not v:
                continue
            m = st.median([x["decode_tps"] for x in v])
            check(f"{case} decode median within {exp['tolerances']['tps_pct']}% of "
                  f"{want['median_decode_tps']}",
                  near(m, want["median_decode_tps"], exp["tolerances"]["tps_pct"]), f"got {m:.1f}")
    else:
        if not (a.ab or a.sweep):
            ap.error("give --ab and/or --sweep, or --historical")
        if a.ab:
            verify_ab(load(a.ab), exp)
        if a.sweep:
            verify_sweep(load(a.sweep), exp)
        if a.smoke:
            verify_smoke(load(a.smoke), exp)

    total = len(passes) + len(fails)
    print(f"\n{total} checks, {len(passes)} passed, {len(fails)} failed")
    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  - " + f)
        return 1
    print("VERDICT: PASS — the measured run is consistent with the recorded results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
