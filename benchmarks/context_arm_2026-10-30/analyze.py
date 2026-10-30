import json
import statistics
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
arms = ["A", "B", "D", "C"]
labels = {"A": "ctx1M   key1M", "B": "ctx1M   key512K",
          "D": "ctx512K key512K", "C": "ctx512K key1M->clamped"}

data = {}
print("=== prefill t/s (all samples shown; n=2 per cell) ===")
print("{:4s} {:22s} {:4s} {:>8s} {:>8s} {:>8s} {:>7s}".format(
    "arm", "config", "size", "median", "min", "max", "spread"))
for arm in arms:
    for size in ("8k", "32k"):
        vals = [r["tps"] for r in rows if r["kind"] == "sample" and r["arm"] == arm and r["size"] == size]
        if not vals:
            continue
        data[(arm, size)] = vals
        med = statistics.median(vals)
        print("{:4s} {:22s} {:4s} {:8.1f} {:8.1f} {:8.1f} {:6.2f}%".format(
            arm, labels[arm], size, med, min(vals), max(vals), 100 * (max(vals) - min(vals)) / med))

print()
print("=== the two questions ===")
for size in ("8k", "32k"):
    a, b = statistics.median(data[("A", size)]), statistics.median(data[("B", size)])
    print("  {} CHUNK_KEY_MAX 1M->512K (ctx fixed at 1M):  {:+.2f}%".format(size, 100 * (b - a) / a))
for size in ("8k", "32k"):
    b, c = statistics.median(data[("B", size)]), statistics.median(data[("C", size)])
    print("  {} ctx 1M->512K (key fixed at 512K):          {:+.2f}%".format(size, 100 * (c - b) / b))
print("  ^ B vs C is also the A/A control: C's key_max is clamped to 512K, so C IS B.")

print()
print("=== noise floor ===")
spreads = sorted(100 * (max(v) - min(v)) / statistics.median(v) for v in data.values())
print("  per-cell spread: median {:.2f}%  max {:.2f}%".format(statistics.median(spreads), max(spreads)))
print("  the max comes from one cell with a single outlier (C 8k: 497.9 vs 536.6);")
print("  the other seven cells are all <=0.50%, so that is the usable floor.")
print("  -> effects below ~0.5% are indistinguishable from noise at n=2.")

print()
print("=== decode, server-side (wall included prefill; this does not) ===")
dec = {}
for line in open(sys.argv[2]):
    if "Completed ChatCompletions" not in line or "decode: 160 tok" not in line:
        continue
    part = line.split("decode: ", 1)[1]
    tokps = float(part.split("=")[1].split("t/s")[0].strip())
    dec.setdefault(len(dec), tokps)
if dec:
    vals = [v for v in dec.values()]
    print("  {} runs, {:.1f}-{:.1f} t/s = {:.2f}-{:.2f} ms/tok, median {:.2f} ms/tok".format(
        len(vals), min(vals), max(vals), 1000 / max(vals), 1000 / min(vals),
        statistics.median([1000 / v for v in vals])))

print()
print("=== integrity ===")
samples = [r for r in rows if r["kind"] == "sample"]
print("  prefill samples={}  all cached=0: {}".format(len(samples), all(r["cached"] == 0 for r in samples)))
print("  aborts: {}".format([r.get("error") for r in rows if r["kind"] in ("abort", "teardown_failure")] or "none"))
boots = [r for r in rows if r["kind"] == "boot"]
ok = all(r["ctx_in_argv"] == str(r["ctx_expected"]) and r["chunk_key_max_in_env"] == r["chunk_key_max_expected"] for r in boots)
print("  arm identity verified from live process: {} ({} boots)".format(ok, len(boots)))
