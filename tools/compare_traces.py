#!/usr/bin/env python3
"""Compare the fixed DeepSeek V4.1 first-token trace stages against ds4.

Shape or numerical mismatches are failures. Four embedding streams can be
compared individually; residual streams must not be silently collapsed.
"""
import argparse
import json
from pathlib import Path
from compare_f32 import compare, load

STAGES = ("embedding", "attn_norm", "q_a_raw", "q_a_norm", "q_b",
          "q_rope", "kv_norm", "kv_rope", "attn_out", "layer_out", "logits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {}
    for stage in STAGES:
        ref = args.directory / f"reference_{stage}-0_pos0.bin"
        candidate = args.directory / f"myds_{stage}-0_pos0.bin"
        try:
            reference, current = load(ref), load(candidate)
        except (OSError, ValueError) as exc:
            results[stage] = {"pass": False, "reason": str(exc)}
            continue
        if stage == "embedding" and len(reference) == 4 * len(current):
            streams = [compare(reference[i * len(current):(i + 1) * len(current)], current, 0, 0)
                       for i in range(4)]
            results[stage] = {"pass": all(s["pass"] for s in streams), "streams": streams}
        else:
            results[stage] = compare(reference, current, 0, 0)
    report = {"pass": all(stage["pass"] for stage in results.values()),
              "reference": "ds4 fixed commit, exact token 35 at position 0",
              "atol": 0, "rtol": 0, "stages": results}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
