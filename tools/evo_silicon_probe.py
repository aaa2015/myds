#!/usr/bin/env python3
"""tools/evo_silicon_probe.py — Silicon Microscope for autonomous myds evolution.

Combines three physical observables into one machine-actionable JSON report
without violating docs/MEASUREMENT-DISCIPLINE.md:

  1. Hardware & Metal PSO ceilings (via bin/evo_pso_probe):
     - GPU family & Dynamic Caching (Apple8 M2 static regs vs Apple9+ M3/M4/M5)
     - Per-kernel staticThreadgroupMemoryLength -> max_tgs_per_core_by_smem
     - Per-kernel maxTotalThreadsPerThreadgroup -> register pressure / spill tier
     - Optional PSO diff (--diff-metallib) to catch smem / occupancy regressions
       at compile time before touching any full-model window.

  2. Production trace roofline (--trace-json, optional):
     - Enforces §6.1 / §169.1: dtags are normalized by sampled tokens derived
       from `cb_count / layers_for_tag`, NEVER raw `out_tokens` (which under-
       states phases by 8x when MYDS_PHASE_SPLIT_DECODE=8).
     - Computes effective GB/s and % of Apple Silicon peak memory bandwidth
       against V4.1-Flash-1017's analytical active byte floor.
     - Enforces §5.1 / §162.4: stamps `gpu_gap` with
       `overlappable_before_first_cb: false`.

  3. Serialized kernel share (--kprof-jsonl, optional):
     - Cross-joins kprof per-kernel net ms/tok with PSO smem & register tiers.
     - Enforces §6.2 / §143.3: stamps `serialized_only_no_bandwidth_claim: true`
       so an agent never subtracts serialized kprof from concurrent wall time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kprof_report  # noqa: E402


def categorize_kernel(name: str) -> str:
    cat = kprof_report.categorize(name)
    if cat == "other":
        low = name.lower()
        if any(tok in low for tok in ("topk", "radix", "idx_")):
            return "indexer"
    return cat

# Peak unified memory bandwidth (GB/s, decimal 1e9 B/s as quoted by Apple specs).
PEAK_BW_GBPS_TABLE: list[tuple[str, float]] = [
    ("m5 ultra", 1024.0),
    ("m4 ultra", 819.0),
    ("m3 ultra", 819.0),
    ("m2 ultra", 800.0),
    ("m1 ultra", 800.0),
    ("m5 max", 546.0),
    ("m4 max", 546.0),
    ("m3 max", 400.0),
    ("m2 max", 400.0),
    ("m4 pro", 273.0),
    ("m3 pro", 150.0),
    ("m2 pro", 200.0),
    ("m5", 153.0),
    ("m4", 120.0),
    ("m3", 100.0),
    ("m2", 100.0),
]

# DeepSeek-V4.1-Flash-1017 Q2-dense-q4-imx-cliecy analytical active read floor
# per single-token decode step (decimal GB, see ledger §0.2 & §166.1):
#   dense (MLA Q/K/V/Out + shared expert + LM head): 5.166 GB
#   routed MoE (6 of 384 experts, IQ2_XXS gate/up + Q2_K down): 2.389 GB @ Top-6
V41_DENSE_ACTIVE_GB = 5.166
V41_ROUTED_TOP6_GB = 2.389

# Trace tag IDs (see src/myds_trace.c & tools/parse_scaling.py) and how many
# layer CBs correspond to 1 sampled decode token under phase-split tracing.
DTAG_INFO: dict[int, tuple[str, int]] = {
    0: ("NONE", 1),
    1: ("ENGRAM", 2),
    2: ("EMBED", 1),
    3: ("NORM_MIX", 40),
    4: ("QKV", 40),
    5: ("QB", 40),
    6: ("ATTN", 40),
    7: ("ATTN_OUT", 40),
    8: ("FFN_IN", 40),
    9: ("SHARED_ROUTE", 40),
    10: ("MOE", 40),
    11: ("HEAD", 1),
    12: ("OTHER", 1),
    13: ("MOE_ROUTE", 40),
    14: ("MOE_GATEUP", 40),
    15: ("MOE_DOWN", 40),
    16: ("MOE_SUM", 40),
    17: ("ATTN_PROJ", 40),
    18: ("ATTN_COMP", 40),
    19: ("ATTN_IDX", 40),
    20: ("ATTN_RING", 40),
    21: ("ATTN_CORE", 40),
    22: ("ATTN_TAIL", 40),
    23: ("DEC_PRE_ATTN", 40),
    24: ("DEC_ATTN", 40),
    25: ("DEC_FFN", 40),
    26: ("DEC_ATTN_OUT", 40),
}


def infer_peak_gbps(gpu_name: str, override: float | None) -> tuple[float, str]:
    if override is not None and override > 0:
        return override, "cli_override"
    low = gpu_name.lower()
    for pat, bw in PEAK_BW_GBPS_TABLE:
        if pat in low:
            return bw, f"matched:{pat}"
    return 800.0, "fallback_default_800"


def run_pso_probe(
    metallib: Path,
    probe_bin: Path,
    kernel_filter: str | None = None,
    fc_specs: list[str] | None = None,
) -> dict[str, Any]:
    if not probe_bin.exists():
        raise RuntimeError(
            f"PSO probe binary missing at {probe_bin}; run `make bin/evo_pso_probe`"
        )
    cmd = [str(probe_bin), str(metallib)]
    if kernel_filter:
        cmd += ["--filter", kernel_filter]
    for spec in fc_specs or []:
        cmd += ["--fc", spec]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not proc.stdout.strip():
        raise RuntimeError(
            f"evo_pso_probe failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    data = json.loads(proc.stdout)
    for k in data.get("kernels", []):
        k["category"] = categorize_kernel(k["name"])
    return data


def diff_pso(base_pso: dict[str, Any], cand_pso: dict[str, Any]) -> dict[str, Any]:
    bmap = {k["name"]: k for k in base_pso.get("kernels", []) if k.get("compiled")}
    cmap = {k["name"]: k for k in cand_pso.get("kernels", []) if k.get("compiled")}
    added = sorted(set(cmap) - set(bmap))
    removed = sorted(set(bmap) - set(cmap))
    changed = []
    regressions = []
    for name in sorted(set(bmap) & set(cmap)):
        b, c = bmap[name], cmap[name]
        fields = (
            "static_tg_mem_bytes",
            "max_tgs_per_core_by_smem",
            "max_total_threads_per_tg",
            "register_pressure_tier",
        )
        if any(b.get(f) != c.get(f) for f in fields):
            delta = {
                "name": name,
                "category": c.get("category"),
                "static_tg_mem_bytes": [b["static_tg_mem_bytes"], c["static_tg_mem_bytes"]],
                "max_tgs_per_core_by_smem": [
                    b["max_tgs_per_core_by_smem"],
                    c["max_tgs_per_core_by_smem"],
                ],
                "max_total_threads_per_tg": [
                    b["max_total_threads_per_tg"],
                    c["max_total_threads_per_tg"],
                ],
            }
            changed.append(delta)
            if (
                c["max_tgs_per_core_by_smem"] < b["max_tgs_per_core_by_smem"]
                or c["max_total_threads_per_tg"] < b["max_total_threads_per_tg"]
            ):
                regressions.append(delta)
    return {
        "added_kernels": added,
        "removed_kernels": removed,
        "changed_kernels": changed,
        "occupancy_or_register_regressions": regressions,
    }


def analyze_trace_record(
    rec: dict[str, Any], peak_gbps: float, decode_topk: int
) -> dict[str, Any]:
    out_tok = int(rec.get("out_tokens") or 0)
    prompt_tok = int(rec.get("prompt_tokens") or 0)
    phases = rec.get("phases") or {}
    detail = rec.get("detail") or {}
    decode_s = float(phases.get("decode") or 0.0)
    prefill_s = float(phases.get("prefill") or 0.0)

    routed_gb = V41_ROUTED_TOP6_GB * (max(1, decode_topk) / 6.0)
    # Approximate KV + compressed latent + indexer read per decode step at prompt_tok:
    # compressed BF16 (512 * 2B * 20 unique KV layers = 20 KiB/tok) + indexer FP8/meta.
    ctx_gb = (prompt_tok * 24576.0) / 1e9
    active_gb_per_tok = V41_DENSE_ACTIVE_GB + routed_gb + ctx_gb

    ms_per_tok = (decode_s * 1000.0 / out_tok) if out_tok > 0 and decode_s > 0 else 0.0
    eff_gbps = (active_gb_per_tok / (decode_s / out_tok)) if ms_per_tok > 0 else 0.0
    roof_pct = (100.0 * eff_gbps / peak_gbps) if peak_gbps > 0 else 0.0

    # §6.1 / §169.1: normalize dtags by `cbs / layers_for_tag`, NEVER raw `out_tokens`
    # when phase-split sampling is active!
    dtags_raw = detail.get("dtags") or []
    dtag_rows = []
    for item in dtags_raw:
        if not isinstance(item, list) or len(item) < 3:
            continue
        tid, gpu_s, cbs = int(item[0]), float(item[1]), int(item[2])
        name, layers = DTAG_INFO.get(tid, (f"TAG_{tid}", 1))
        if cbs > 0 and layers > 0:
            sampled_tokens = cbs / float(layers)
            norm_basis = f"cbs/{layers} (§6.1)"
        elif out_tok > 0:
            sampled_tokens = float(out_tok)
            norm_basis = "out_tokens_fallback"
        else:
            sampled_tokens = 0.0
            norm_basis = "none"
        us_per_tok = (gpu_s * 1e6 / sampled_tokens) if sampled_tokens > 0 else 0.0
        dtag_rows.append(
            {
                "tag_id": tid,
                "name": name,
                "gpu_seconds": round(gpu_s, 6),
                "cbs": cbs,
                "sampled_tokens": round(sampled_tokens, 2),
                "us_per_token": round(us_per_tok, 2),
                "normalization": norm_basis,
            }
        )
    dtag_rows.sort(key=lambda r: -r["us_per_token"])

    decode_gpu_s = float(detail.get("decode_gpu") or 0.0)
    gpu_gap_us = (
        max(0.0, (decode_s - decode_gpu_s) * 1e6 / out_tok)
        if out_tok > 0 and decode_s > 0 and decode_gpu_s > 0
        else None
    )

    return {
        "prompt_tokens": prompt_tok,
        "out_tokens": out_tok,
        "prefill_s": round(prefill_s, 4),
        "prefill_tps": round(prompt_tok / prefill_s, 2) if prefill_s > 0 else None,
        "decode_s": round(decode_s, 4),
        "decode_tps": round(out_tok / decode_s, 2) if decode_s > 0 else None,
        "decode_ms_per_token": round(ms_per_tok, 3),
        "decode_roofline": {
            "decode_topk": decode_topk,
            "analytical_active_gb_per_token": round(active_gb_per_tok, 4),
            "dense_gb": V41_DENSE_ACTIVE_GB,
            "routed_moe_gb": round(routed_gb, 4),
            "context_kv_est_gb": round(ctx_gb, 4),
            "effective_gbps": round(eff_gbps, 2),
            "peak_gbps": peak_gbps,
            "roof_utilization_pct": round(roof_pct, 2),
            "theoretical_min_ms_per_token": round(
                1000.0 * active_gb_per_tok / peak_gbps, 3
            )
            if peak_gbps > 0
            else None,
        },
        "gpu_gap": {
            "estimated_us_per_token": round(gpu_gap_us, 2)
            if gpu_gap_us is not None
            else None,
            "overlappable_before_first_cb": False,
            "discipline_note": (
                "§5.1 / §162.4: GPU is idle at token start after waitUntilCompleted; "
                "async host work before CB#1 shifts time into encode1 instead of "
                "reducing token latency."
            ),
        },
        "dtags_per_token": dtag_rows,
    }


def join_kprof_with_pso(
    kprof_path: Path, pso_data: dict[str, Any], top_n: int
) -> dict[str, Any]:
    kernels, summary, run = kprof_report.load(str(kprof_path))
    tokens = summary["tokens"]
    floor_us = summary.get("floor_us", 0.0)
    cat_rows, rows = kprof_report.aggregate(kernels, tokens, floor_us)
    pso_map = {
        k["name"]: k for k in pso_data.get("kernels", []) if k.get("compiled")
    }
    enriched = []
    for r in rows[:top_n]:
        pk = pso_map.get(r["name"])
        item = dict(r)
        if pk:
            item["pso"] = {
                "static_tg_mem_bytes": pk["static_tg_mem_bytes"],
                "max_tgs_per_core_by_smem": pk["max_tgs_per_core_by_smem"],
                "max_total_threads_per_tg": pk["max_total_threads_per_tg"],
                "register_pressure_tier": pk["register_pressure_tier"],
            }
        enriched.append(item)
    return {
        "serialized_only_no_bandwidth_claim": True,
        "discipline_note": (
            "§6.2 / §143.3: kprof serializes encoders and removes concurrent "
            "overlap. Never subtract kprof ms/tok from production wall time or "
            "derive GB/s from kprof durations."
        ),
        "tokens": tokens,
        "floor_us": floor_us,
        "net_ms_per_token_serialized": round(
            sum(c["ms_per_token"] for c in cat_rows), 4
        ),
        "categories": cat_rows,
        "top_kernels": enriched,
        "run": run,
    }


def summarize_hot_pso_Watchlist(pso_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Surface V4.1 production hot-path kernels where smem or registers cap occupancy."""
    hot_pat = re.compile(
        r"indexer|flash_attn|moe|q4_k|iq2|hc_|engram|argsort|topk|rmsnorm",
        re.IGNORECASE,
    )
    constrained = []
    for k in pso_data.get("kernels", []):
        if not k.get("compiled") or not hot_pat.search(k["name"]):
            continue
        if k["max_tgs_per_core_by_smem"] <= 4 or k["max_total_threads_per_tg"] < 1024:
            constrained.append(
                {
                    "name": k["name"],
                    "category": k["category"],
                    "static_tg_mem_bytes": k["static_tg_mem_bytes"],
                    "max_tgs_per_core_by_smem": k["max_tgs_per_core_by_smem"],
                    "max_total_threads_per_tg": k["max_total_threads_per_tg"],
                    "register_pressure_tier": k["register_pressure_tier"],
                }
            )
    constrained.sort(
        key=lambda x: (x["max_tgs_per_core_by_smem"], x["max_total_threads_per_tg"], x["name"])
    )
    return constrained


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metallib", type=Path, default=ROOT / "bin" / "myds.metallib")
    ap.add_argument("--probe-bin", type=Path, default=ROOT / "bin" / "evo_pso_probe")
    ap.add_argument("--pso-json", type=Path, help="Read pre-captured evo_pso_probe JSON")
    ap.add_argument("--diff-metallib", type=Path, help="Baseline metallib to diff PSO limits against")
    ap.add_argument("--filter", help="Substring filter for kernel names")
    ap.add_argument("--fc", action="append", default=[], help="Function constant KEY=INT")
    ap.add_argument("--trace-json", type=Path, help="Saved /trace JSON payload to compute roofline")
    ap.add_argument("--trace-url", help="Live loopback /trace/api/requests URL to compute roofline")
    ap.add_argument("--debug-token", help="X-Debug-Token header for --trace-url")
    ap.add_argument("--kprof-jsonl", type=Path, help="Saved bench_kprof JSONL to cross-join with PSO")
    ap.add_argument("--decode-topk", type=int, default=3, help="Active routed experts in decode (3 or 6)")
    ap.add_argument("--peak-gbps", type=float, help="Override GPU theoretical memory bandwidth (GB/s)")
    ap.add_argument("--top", type=int, default=30, help="Max constrained/top kernels to include")
    ap.add_argument("--json", action="store_true", help="Emit full JSON to stdout")
    ap.add_argument("--out", type=Path, help="Write full JSON report to file")
    args = ap.parse_args()

    if args.pso_json:
        pso = json.loads(args.pso_json.read_text())
        for k in pso.get("kernels", []):
            k.setdefault("category", categorize_kernel(k["name"]))
    else:
        pso = run_pso_probe(args.metallib, args.probe_bin, args.filter, args.fc)

    dev = pso["device"]
    peak_gbps, bw_src = infer_peak_gbps(dev.get("gpu_name", ""), args.peak_gbps)

    report: dict[str, Any] = {
        "kind": "evo_silicon_probe",
        "hardware": {
            **dev,
            "peak_memory_bandwidth_gbps": peak_gbps,
            "peak_bandwidth_source": bw_src,
            "architecture_guidance": (
                "Apple9+ Dynamic Caching: threadgroup memory and registers share "
                "unified L1 cache dynamically; verify occupancy empirically."
                if dev.get("has_dynamic_caching")
                else "Apple8 static allocation: static_tg_mem_bytes hard-caps "
                "max_tgs_per_core_by_smem = floor(32768 / smem); <=4096 B unlocks 8 TGs/core."
            ),
        },
        "metallib": pso["metallib"],
        "occupancy_constrained_hot_kernels": summarize_hot_pso_Watchlist(pso)[: args.top],
    }

    if args.diff_metallib:
        base_pso = run_pso_probe(args.diff_metallib, args.probe_bin, args.filter, args.fc)
        report["pso_diff_vs_baseline"] = diff_pso(base_pso, pso)

    trace_payload = None
    if args.trace_json:
        trace_payload = json.loads(args.trace_json.read_text())
    elif args.trace_url:
        hdrs = {"X-Debug-Token": args.debug_token} if args.debug_token else {}
        req = urllib.request.Request(args.trace_url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            trace_payload = json.loads(resp.read().decode("utf-8"))
    if trace_payload is not None:
        records = trace_payload.get("records") or (
            [trace_payload] if "phases" in trace_payload else []
        )
        report["trace_roofline"] = [
            analyze_trace_record(r, peak_gbps, args.decode_topk) for r in records
        ]

    if args.kprof_jsonl:
        report["kprof_joined"] = join_kprof_with_pso(args.kprof_jsonl, pso, args.top)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    hw = report["hardware"]
    ml = report["metallib"]
    print(
        f"== Silicon Probe: {hw['gpu_name']} ({hw['gpu_cores']} cores, "
        f"{hw['unified_mem_gib']:.1f} GiB, family=Apple{hw['apple_gpu_family_max']}, "
        f"dynamic_caching={hw['has_dynamic_caching']}, peak={peak_gbps:.0f} GB/s)"
    )
    print(
        f"   metallib: {ml['compiled_count']}/{ml['matched_functions']} PSOs compiled "
        f"(failed={ml['failed_count']}, max_tg_mem={hw['max_tg_mem_bytes']} B)"
    )
    print(f"   guidance: {hw['architecture_guidance']}")

    if "pso_diff_vs_baseline" in report:
        d = report["pso_diff_vs_baseline"]
        print(
            f"\n== PSO Diff vs baseline: +{len(d['added_kernels'])} "
            f"-{len(d['removed_kernels'])} ~{len(d['changed_kernels'])} "
            f"(regressions={len(d['occupancy_or_register_regressions'])})"
        )
        for ch in d["changed_kernels"]:
            print(
                f"   {ch['name']}: smem {ch['static_tg_mem_bytes'][0]}->{ch['static_tg_mem_bytes'][1]} B, "
                f"max_tgs/core {ch['max_tgs_per_core_by_smem'][0]}->{ch['max_tgs_per_core_by_smem'][1]}, "
                f"max_threads {ch['max_total_threads_per_tg'][0]}->{ch['max_total_threads_per_tg'][1]}"
            )

    if report["occupancy_constrained_hot_kernels"]:
        print(f"\n== Occupancy/Register Constrained Hot Kernels (top {args.top})")
        print(f"   {'kernel':<50} {'cat':<14} {'smem_B':>7} {'tgs/core':>8} {'max_thr':>7}")
        for k in report["occupancy_constrained_hot_kernels"]:
            print(
                f"   {k['name'][:49]:<50} {k['category']:<14} "
                f"{k['static_tg_mem_bytes']:>7} {k['max_tgs_per_core_by_smem']:>8} "
                f"{k['max_total_threads_per_tg']:>7}"
            )

    for idx, tr in enumerate(report.get("trace_roofline", [])):
        rf = tr["decode_roofline"]
        print(
            f"\n== Trace[{idx}] prompt={tr['prompt_tokens']} out={tr['out_tokens']} "
            f"decode={tr['decode_tps']} t/s ({tr['decode_ms_per_token']} ms/tok)"
        )
        print(
            f"   Active read floor: {rf['analytical_active_gb_per_token']} GB/tok "
            f"-> effective {rf['effective_gbps']} GB/s = {rf['roof_utilization_pct']}% of {rf['peak_gbps']} GB/s roof "
            f"(min possible={rf['theoretical_min_ms_per_token']} ms/tok)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
