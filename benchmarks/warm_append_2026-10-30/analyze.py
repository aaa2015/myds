import json
import statistics
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
cold, warm = {}, {}
for r in rows:
    if r["kind"] == "cold" and r.get("completed"):
        cold[r["size"]] = r["completed"]
    elif r["kind"] == "warm" and r.get("completed"):
        warm.setdefault(r["size"], {})[r["append"]] = r["completed"]

sizes = [s for s in ("160k", "320k", "470k") if s in cold]

print("=== prefill rate: cold vs warm append ===")
print("{:6s} {:>10s} {:>8s} {:>9s} | {:>9s} {:>8s} | {:>9s} {:>8s}".format(
    "size", "cold tok", "cold s", "cold t/s", "warm+512", "t/s", "warm+2048", "t/s"))
for s in sizes:
    c = cold[s]
    w5 = warm.get(s, {}).get(512, {})
    w2 = warm.get(s, {}).get(2048, {})
    print("{:6s} {:10d} {:8.1f} {:9.1f} | {:9d} {:8.1f} | {:9d} {:8.1f}".format(
        s, c["prompt_tok"], c["prefill_s"], c["tps"],
        w5.get("prefill_tok", 0), w5.get("tps", 0.0),
        w2.get("prefill_tok", 0), w2.get("tps", 0.0)))

print()
print("=== two-point fit of the warm append cost (fixed + marginal) ===")
print("{:6s} {:>9s} {:>11s} {:>11s} {:>8s} {:>7s}".format(
    "size", "cached%", "fixed s", "ms/token", "cold ms", "ratio"))
fits = {}
for s in sizes:
    w5 = warm.get(s, {}).get(512)
    w2 = warm.get(s, {}).get(2048)
    if not (w5 and w2):
        continue
    d_tok = w2["prefill_tok"] - w5["prefill_tok"]
    d_s = w2["prefill_s"] - w5["prefill_s"]
    marginal_ms = 1000.0 * d_s / d_tok
    fixed = w5["prefill_s"] - w5["prefill_tok"] * marginal_ms / 1000.0
    cold_ms = 1000.0 * cold[s]["prefill_s"] / cold[s]["prefill_tok"]
    # cached counts tokens the server reused; prompt_tok is the whole request.
    # Dividing by (prompt - prefill) gave >100% because an earlier warm request
    # had already extended the context past the base length.
    reuse = 100.0 * w5["cached"] / max(1, w5["prompt_tok"])
    fits[s] = (fixed, marginal_ms, cold_ms)
    print("{:6s} {:8.1f}% {:11.2f} {:11.2f} {:8.2f} {:6.2f}x".format(
        s, reuse, fixed, marginal_ms, cold_ms, marginal_ms / cold_ms))

print()
print("=== implied cost of the A1 shape: 1983 appended tokens ===")
for s in sizes:
    if s not in fits:
        continue
    fixed, marginal_ms, _ = fits[s]
    print("  {:6s} predicted {:.1f}s for 1983 tokens".format(s, fixed + 1983 * marginal_ms / 1000.0))
print("  A1 (mask-skip screen) measured 11.74 s warm prefill at 98304 cached / 1983 recomputed")

print()
print("=== reuse hygiene ===")
for s in sizes:
    for a in (512, 2048):
        w = warm.get(s, {}).get(a)
        if not w:
            continue
        total = w["prompt_tok"]
        print("  {:6s} +{:4d}: reused {} of {} tokens ({:.1f}%), prefilled {} tok".format(
            s, a, w["cached"], total, 100.0 * w["cached"] / total, w["prefill_tok"]))
