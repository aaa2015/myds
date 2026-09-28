#!/usr/bin/env python3
"""Compare native V4.1 traces to a pinned independent ds4 implementation.

Positions are zero-based teacher-forced input indices. Shape, missing stages,
nonfinite values and collapsed post-HC streams fail. No residual stream is
silently repeated or collapsed to make mismatching tensor shapes compare.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
from compare_f32 import compare, load

STAGES = ("embedding", "hc_attn_mix", "hc_attn_split", "hc_attn_pre",
          "hc_attn_input", "attn_norm", "q_a_raw", "q_a_norm", "q_b",
          "q_rope", "kv_raw", "kv_norm", "kv_rope", "kv_quant", "attn_out",
          "after_attn", "hc_ffn_mix", "hc_ffn_split", "hc_ffn_input",
          "ffn_norm", "route_logits", "route_probs", "route_ids", "route_weights",
          "shared_out", "routed_out", "ffnblock", "layer_out", "logits")
GLOBAL_STAGES = {"embedding", "logits"}
STREAM_STAGES = {"embedding", "after_attn", "layer_out"}


def indices(value):
    try:
        parts = value.split(",")
        if not parts or any(not p.isascii() or not p.isdecimal() for p in parts):
            raise ValueError
        result = [int(p) for p in parts]
        if len(set(result)) != len(result) or max(result) > 2**32 - 1:
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected unique nonnegative decimal indices separated by commas") from exc


def distinct_streams(values, streams):
    """Report every pair, so a duplicated stream cannot hide behind one outlier."""
    width = len(values) // streams
    pairs = []
    for first, second in itertools.combinations(range(streams), 2):
        a = values[first * width:(first + 1) * width]
        b = values[second * width:(second + 1) * width]
        errors = [abs(x - y) for x, y in zip(a, b)]
        pairs.append({"streams": [first, second], "distinct": a != b,
                      "max_abs_difference": max(errors),
                      "rmse": math.sqrt(math.fsum(x * x for x in errors) / width)})
    return {"pass": all(p["distinct"] for p in pairs), "pairs": pairs}


def compare_stage(reference, candidate, stage, pos, streams=4, atol=0, rtol=0):
    result = compare(reference, candidate, atol, rtol)
    if stage not in STREAM_STAGES:
        return result
    # A shape error is an actual architecture failure. Do not compare one native
    # vector four times and turn it into evidence for a four-stream HC graph.
    if len(reference) != len(candidate):
        result["expected_streams"] = streams
        return result
    if not reference or len(reference) % streams:
        return {"pass": False, "reason": "stream tensor is not divisible by stream count",
                "count": len(reference), "expected_streams": streams}
    if result.get("reason") == "nonfinite values":
        return result
    width = len(reference) // streams
    result["streams"] = [dict(stream=i, **compare(
        reference[i * width:(i + 1) * width], candidate[i * width:(i + 1) * width], atol, rtol))
        for i in range(streams)]
    # At every token, embedding initializes four identical streams. The first
    # attention input therefore cannot establish HC operation. Only post-HC
    # layer_out is required to contain pairwise-distinct streams at pos >= 1.
    if stage == "layer_out" and pos >= 1:
        result["reference_stream_diversity"] = distinct_streams(reference, streams)
        result["candidate_stream_diversity"] = distinct_streams(candidate, streams)
        result["pass"] = (result["pass"] and result["reference_stream_diversity"]["pass"]
                          and result["candidate_stream_diversity"]["pass"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positions", type=indices, default=[0])
    parser.add_argument("--layers", type=indices, default=[0])
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--tolerance-reason", help="Required when choosing any nonzero tolerance")
    parser.add_argument("--tokens-file", type=Path, help="Record the exact teacher-forced token file SHA256")
    args = parser.parse_args()
    if args.streams < 2:
        parser.error("stream count must be at least two")
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    if (args.atol or args.rtol) and not args.tolerance_reason:
        parser.error("nonzero tolerance requires --tolerance-reason; it must be chosen before inspecting a result")
    stages = args.stages.split(",")
    if len(set(stages)) != len(stages) or any(s not in STAGES for s in stages):
        parser.error("--stages must list distinct known stage names: " + ",".join(STAGES))
    captures = []
    for pos in args.positions:
        for layer in args.layers:
            results = {}
            for stage in stages:
                if layer != 0 and stage in GLOBAL_STAGES:
                    continue
                ref = args.directory / f"reference_{stage}-{layer}_pos{pos}.bin"
                candidate = args.directory / f"myds_{stage}-{layer}_pos{pos}.bin"
                try:
                    results[stage] = compare_stage(load(ref), load(candidate), stage, pos,
                                                   args.streams, args.atol, args.rtol)
                except (OSError, ValueError) as exc:
                    results[stage] = {"pass": False, "reason": str(exc)}
            if results:
                captures.append({"position": pos, "layer": layer, "stages": results,
                                 "pass": all(r["pass"] for r in results.values())})
    if not captures:
        parser.error("selection contains no stages (embedding/logits use global layer 0)")
    report = {"pass": all(c["pass"] for c in captures),
              "reference": "ds4 pinned commit 6f873ad7f5940ab8556abc41ae510853118ecc4a",
              "positions": args.positions, "layers": args.layers, "streams": args.streams,
              "atol": args.atol, "rtol": args.rtol,
              "tolerance_reason": args.tolerance_reason, "captures": captures}
    if args.tokens_file:
        import hashlib
        report["tokens_file"] = str(args.tokens_file)
        report["tokens_sha256"] = hashlib.sha256(args.tokens_file.read_bytes()).hexdigest()
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
