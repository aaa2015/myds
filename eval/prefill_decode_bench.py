#!/usr/bin/env python3
"""Online TTFT and decode screening, using final usage token counts.

Usage: prefill_decode_bench.py <port> [ctx_sizes...] [--decode-tokens 512] [--jsonl PATH]
Prefill requests use max_tokens=1 and a random leading nonce. tokens/TTFT is an
end-to-end estimate including cache/restore, first-token work and HTTP transport.
usage.prompt_tokens alone cannot prove a cache miss: inspect server evidence.
Decode timing from visible text cannot replace native generated-token timing.
"""
import argparse
import json
import os
import random
import statistics as st

try:
    from .stream_bench import BenchmarkError, request_stream
except ImportError:
    from stream_bench import BenchmarkError, request_stream


FILLER = ("The compressor reads rows from the table and folds them into the "
          "residual stream before the next sublayer begins its work. ")


def make_prompt(target_tokens, nonce):
    body = FILLER * (target_tokens // 14 + 2)
    return "[nonce %s] " % nonce + body[:target_tokens * 4]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("ctx_sizes", nargs="*", type=int)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--jsonl", help="append complete timing/usage records (including warmups and rejections)")
    args = parser.parse_args(argv)
    if args.decode_tokens < 2 or any(size < 1 for size in args.ctx_sizes):
        parser.error("context sizes must be positive and --decode-tokens must be at least 2")
    sizes = args.ctx_sizes or [512, 2048, 4096]
    alias = os.environ.get("MYDS_ALIAS", "local-myds-deepseek-v4.1-flash-1017-q2-q4")
    failed = False

    def call(prompt, tokens, phase, run, warmup=False):
        record = {"phase": phase, "run": run, "warmup": warmup,
                  "requested_tokens": tokens, "port": args.port, "alias": alias}
        try:
            row = request_stream(args.port, alias, prompt, tokens)
            record.update(row, accepted=True)
            return row
        except BenchmarkError as exc:
            record.update(accepted=False, error=str(exc))
            raise
        finally:
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")

    print("TTFT screen (max_tokens=1, random leading nonce); tokens/TTFT includes first-token and HTTP work.")
    print("Verify cache misses in server evidence; usage.prompt_tokens is full prompt length, not recomputed tokens.")
    print("  %7s %8s %9s %16s" % ("target", "tokens", "TTFT s", "tokens/TTFT"))
    prefill_rows = []
    for size in sizes:
        prompt = make_prompt(size, random.getrandbits(48))
        try:
            row = call(prompt, 1, "ttft", size)
            ptok, ttft = row["prompt_tokens"], row["ttft_s"]
            if ptok <= 0 or ttft <= 0:
                raise BenchmarkError("no measurable prompt/TTFT")
        except BenchmarkError as exc:
            failed = True
            print("  %7d rejected: %s" % (size, exc))
            continue
        rate = ptok / ttft
        prefill_rows.append((ptok, ttft, rate))
        print("  %7d %8d %9.2f %16.1f" % (size, ptok, ttft, rate))

    print("\nHTTP decode estimate (~1K input, %d output tokens; 1 warmup + 3 formal)." % args.decode_tokens)
    print("Estimate=(usage.completion_tokens-1)/(last_text-first_text); buffering/invisible tokens can shift the estimate.")
    dec_rows = []
    for run in range(-1, 3):
        prompt = make_prompt(1024, random.getrandbits(48))
        label = "warmup" if run < 0 else "run %d" % run
        try:
            row = call(prompt, args.decode_tokens, "decode", run, warmup=run < 0)
            rate = row["decode_est_tok_s"]
            if rate is None:
                raise BenchmarkError("no measurable visible decode interval")
        except BenchmarkError as exc:
            failed = True
            print("  %s rejected: %s" % (label, exc))
            if run < 0:
                break
            continue
        if run < 0:
            continue
        dec_rows.append(rate)
        print("  %s: TTFT %.2f s, %d tokens / %d text chunks, %.2f est tok/s, total %.2f s, finish=%s" %
              (label, row["ttft_s"], row["output_tokens"], row["text_chunks"], rate,
               row["total_s"], row["finish"]))
    if prefill_rows:
        print("\nTTFT summary:")
        for ptok, ttft, rate in prefill_rows:
            print("  %7d tokens %7.2f s %8.1f tokens/TTFT" % (ptok, ttft, rate))
    if dec_rows:
        print("HTTP decode estimate median: %.2f tok/s (range %.2f–%.2f; valid %d/3)" %
              (st.median(dec_rows), min(dec_rows), max(dec_rows), len(dec_rows)))
    return 1 if failed or len(dec_rows) != 3 else 0


if __name__ == "__main__":
    raise SystemExit(main())
