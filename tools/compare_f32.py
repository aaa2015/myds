#!/usr/bin/env python3
"""Compare reference/candidate F32 dumps; a mismatch exits nonzero.

No same-engine baseline is implicitly treated as a correctness oracle.
Tolerances must be chosen before a test from the numerical contract.
"""
import argparse
import array
import json
import math
import sys
from pathlib import Path


def load(path):
    data = Path(path).read_bytes()
    if not data or len(data) % 4:
        raise ValueError(f"{path}: expected nonempty little-endian F32 data")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def compare(reference, candidate, atol, rtol):
    if len(reference) != len(candidate):
        return {"pass": False, "reason": "shape mismatch",
                "reference_size": len(reference), "candidate_size": len(candidate)}
    if any(not math.isfinite(x) for x in reference) or any(not math.isfinite(x) for x in candidate):
        return {"pass": False, "reason": "nonfinite values"}
    error = [abs(a - b) for a, b in zip(reference, candidate)]
    mismatches = [i for i, (a, e) in enumerate(zip(reference, error))
                  if e > atol + rtol * abs(a)]
    return {"pass": not mismatches, "count": len(error),
            "atol": atol, "rtol": rtol,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
            "max_abs_error": max(error),
            "rmse": math.sqrt(math.fsum(e * e for e in error) / len(error)),
            "reference_argmax": max(range(len(reference)), key=reference.__getitem__),
            "candidate_argmax": max(range(len(candidate)), key=candidate.__getitem__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    args = parser.parse_args()
    if not (math.isfinite(args.atol) and math.isfinite(args.rtol)) or min(args.atol, args.rtol) < 0:
        parser.error("tolerances must be finite and nonnegative")
    try:
        result = compare(load(args.reference), load(args.candidate), args.atol, args.rtol)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
