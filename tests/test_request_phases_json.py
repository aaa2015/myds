#!/usr/bin/env python3
"""Validate the request_phases JSON that the server emits.

A printf whose argument types drift from its format string compiles cleanly and
emits unparsable output. A parser then drops the line silently, and the run looks
as if the stage was never measured -- so "it compiled" is not evidence. This test
parses every case and asserts the null-vs-zero semantics that make the numbers
trustworthy.

Usage: python3 tests/test_request_phases_json.py ./bin/test_request_phases_json
"""
import json
import subprocess
import sys

KEYS = ["queue", "render", "setup", "engine", "prefill", "decode",
        "engine_unattributed", "prompt_tok", "cached_tok", "out_tok", "stream"]
TIME_KEYS = ["render", "setup", "engine", "prefill", "decode", "engine_unattributed"]


def check(name, condition, detail=""):
    return {"case": name, "passed": bool(condition), "detail": detail}


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "./bin/test_request_phases_json"
    lines = [line for line in subprocess.run([binary], capture_output=True, text=True,
                                             check=True).stdout.splitlines() if line.strip()]
    results = []
    parsed = []
    for index, line in enumerate(lines):
        try:
            parsed.append(json.loads(line))
            results.append(check(f"parses_case_{index}", True))
        except ValueError as exc:
            parsed.append(None)
            results.append(check(f"parses_case_{index}", False, f"{exc}: {line[:120]}"))

    if len(parsed) != 4 or any(item is None for item in parsed):
        results.append(check("all_cases_parse", False, f"{len(parsed)} parsed"))
        print(json.dumps({"results": results}, indent=2))
        return 1
    results.append(check("all_cases_parse", True))

    for index, item in enumerate(parsed):
        results.append(check(f"case_{index}_has_all_keys",
                             list(item) == KEYS, f"{list(item)}"))

    full, none, partial, negative = parsed

    # Timing fields are numbers when measured.
    results.append(check("full_all_times_numeric",
                         all(isinstance(full[key], (int, float)) for key in TIME_KEYS),
                         str({k: full[k] for k in TIME_KEYS})))
    results.append(check("full_sum_consistent",
                         abs((full["prefill"] + full["decode"] + full["engine_unattributed"])
                             - full["engine"]) < 1e-6,
                         f"{full['prefill']} + {full['decode']} + {full['engine_unattributed']}"
                         f" != {full['engine']}"))

    # Unmeasured stages must be null, never 0. A 0 here would read as
    # "measured, took no time" and would silently deflate any average.
    results.append(check("engine_absent_is_null",
                         none["engine"] is None and none["render"] is None
                         and none["prefill"] is None and none["decode"] is None,
                         str(none)))
    results.append(check("queue_always_numeric",
                         isinstance(none["queue"], (int, float)),
                         str(none["queue"])))
    results.append(check("stream_is_json_bool",
                         none["stream"] is True and full["stream"] is False,
                         f"{none['stream']} {full['stream']}"))

    # A partially measured request keeps what it knows and nulls the rest.
    results.append(check("partial_keeps_render",
                         isinstance(partial["render"], (int, float)),
                         str(partial["render"])))
    results.append(check("partial_nulls_engine",
                         partial["engine"] is None and partial["decode"] is None,
                         str(partial)))

    # Negative unattributed is informative; it must not be coerced to null or 0.
    results.append(check("negative_unattributed_survives",
                         isinstance(negative["engine_unattributed"], (int, float))
                         and negative["engine_unattributed"] < 0,
                         str(negative["engine_unattributed"])))

    passed = all(item["passed"] for item in results)
    print(json.dumps({"results": results, "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
