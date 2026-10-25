#!/usr/bin/env python3
"""Fixed-output online decode screening, using final usage token counts.

Usage: decode_fixed.py <port> [alias] [--tokens 128] [--jsonl PATH]
Every prompt gets one discarded warmup and three formal samples. A sample must
finish at its requested token limit, with final usage and [DONE]. HTTP exposes
visible text arrivals, not exact generated-token timing; native decode is the
authority for formal performance acceptance.
"""
import argparse
import json
import statistics as st

try:
    from .stream_bench import BenchmarkError, request_stream
except ImportError:
    from stream_bench import BenchmarkError, request_stream


PROMPTS = [
    "Write a long detailed technical explanation of how a modern CPU pipeline works, covering fetch, decode, execute, memory access and writeback stages.",
    "Explain in detail the differences between B-trees and LSM-trees for storage engines, including write amplification and read paths.",
    "Describe the complete lifecycle of an HTTP request from DNS resolution through TLS handshake to the server returning a response.",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("alias", nargs="?", default="local-myds-deepseek-v4.1-flash-q2")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--jsonl", help="append complete timing/usage records (including warmups and rejections)")
    args = parser.parse_args(argv)
    if args.tokens < 2:
        parser.error("--tokens must be at least 2 for decode")
    print("HTTP decode estimate=(usage.completion_tokens-1)/(last_text-first_text). "
          "UTF-8/reasoning buffering and invisible tokens can shift the estimate; native timing is authoritative.")
    good = []
    for pi, prompt in enumerate(PROMPTS):
        warmup_ok = True
        for run in range(-1, 3):
            if not warmup_ok:
                break
            label = "warmup" if run < 0 else "r%d" % run
            record = {"prompt_index": pi, "run": run, "warmup": run < 0,
                      "requested_tokens": args.tokens, "port": args.port, "alias": args.alias}
            try:
                row = request_stream(args.port, args.alias, prompt, args.tokens, timeout=900)
                record.update(row)
                rate = row["decode_est_tok_s"]
                if rate is None:
                    raise BenchmarkError("no measurable visible decode interval")
                record["accepted"] = True
            except BenchmarkError as exc:
                record.update(accepted=False, error=str(exc))
                print("  p%d %s: rejected: %s" % (pi, label, exc), flush=True)
                if run < 0:
                    warmup_ok = False
            else:
                if run >= 0:
                    good.append(rate)
                    print("  p%d %s: %d tokens / %d text chunks  %.2f est tok/s  finish=%s" %
                          (pi, label, row["output_tokens"], row["text_chunks"], rate, row["finish"]), flush=True)
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
    if good:
        median = st.median(good)
        print("  Valid samples %d/9; HTTP decode estimate median=%.2f tok/s (%.2f ms/interval)" %
              (len(good), median, 1000 / median))
    else:
        print("  No valid samples")
    return 0 if len(good) == 9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
