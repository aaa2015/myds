#!/usr/bin/env python3
"""Measure a running myds server without loading a second model process.

Each JSONL result contains the exact request, full response text, its SHA256,
usage and timestamped SSE events. TTFT and decode timing are client wall time.
The fixed greedy prompts permit before/after comparisons; output equality does
not establish model correctness or answer quality. The current server ignores
the seed field, which is immaterial to these temperature=0 requests.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request


PROMPTS = [
    (
        "recall64",
        "Repeat the following text verbatim: DeepSeek-V4.1-Flash is an "
        "open-source Mixture-of-Experts language model.",
        64,
    ),
    ("arithmetic64", "Calculate 127 * 43 step by step and provide the final answer.", 64),
    ("code64", "Write a concise C function that computes the square of an integer.", 64),
    (
        "long16",
        "Remember the codeword Amber-lantern-731.\n"
        + "".join(
            f"Record {i}: local inference should preserve every token while "
            "improving scheduling and memory reuse.\n"
            for i in range(30)
        )
        + "\nWhat is the codeword? Answer briefly.",
        16,
    ),
]


def benchmark(url, model, name, prompt, max_tokens, timeout, run):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    chunks, events, chunk_times = [], [], []
    usage = first = last = finish = None
    first_byte_time = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for raw_line in response:
            now = time.perf_counter()
            if first_byte_time is None:
                first_byte_time = now
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            events.append({"time_s": now - start, "line": line})
            if line == "data: [DONE]":
                done = True
                continue
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    if first is None:
                        first = now
                    last = now
                    chunks.append(text)
                    chunk_times.append(now)
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    if status != 200 or not done or usage is None or finish is None:
        raise RuntimeError(f"Incomplete SSE response for {name}: status={status}, usage={usage}, finish={finish}, done={done}")
    output = "".join(chunks)
    count = usage.get("completion_tokens", 0)
    decode_s = (last - first) if (first is not None and last is not None and last > first) else 0.0
    inter_chunk_ms = [
        round((chunk_times[i] - chunk_times[i - 1]) * 1000.0, 2)
        for i in range(1, len(chunk_times))
    ]
    ttft_s = first - start if first is not None else None
    
    # Calculate decode speed using usage token count vs event count alignment
    decode_tps = None
    if decode_s > 0:
        if count and count > 1:
            decode_tps = (count - 1) / decode_s
        elif len(chunks) > 1:
            decode_tps = (len(chunks) - 1) / decode_s

    return {
        "name": name,
        "run": run,
        "url": url,
        "status": status,
        "request": payload,
        "usage": usage,
        "finish_reason": finish,
        "ttft_s": ttft_s,
        "first_byte_s": first_byte_time - start if first_byte_time else None,
        "decode_s": decode_s,
        "decode_tps": round(decode_tps, 2) if decode_tps is not None else None,
        "wall_s": end - start,
        "content_events": len(chunks),
        "completion_tokens": count,
        "inter_chunk_latencies_ms": inter_chunk_ms,
        "text_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output": output,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8181", help="Server base URL, without /v1")
    parser.add_argument("--model", default="local-myds-deepseek-v4.1-flash-q2")
    parser.add_argument("--repeats", type=int, default=3, help="Sequential repetitions of all four fixed prompts")
    parser.add_argument("--output", help="Write full JSONL results here; default: stdout")
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP timeout in seconds")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")
    destination = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    results_by_prompt = {}
    try:
        for run in range(1, args.repeats + 1):
            for name, prompt, max_tokens in PROMPTS:
                result = benchmark(args.url, args.model, name, prompt, max_tokens, args.timeout, run)
                results_by_prompt.setdefault(name, []).append(result)
                print(json.dumps(result, ensure_ascii=False), file=destination, flush=True)
                print(
                    f"{name} run={run} tokens={result['completion_tokens']} events={result['content_events']} "
                    f"TTFT={result['ttft_s']:.3f}s decode={result['decode_tps']}t/s "
                    f"wall={result['wall_s']:.3f}s SHA256={result['text_sha256'][:16]}...",
                    file=sys.stderr,
                    flush=True,
                )
        
        # Summary table of independent medians
        import statistics
        print("\n" + "=" * 80, file=sys.stderr)
        print(f"BENCHMARK AGGREGATE SUMMARY ({args.repeats} runs, independent medians):", file=sys.stderr)
        print(f"{'Prompt':<14} | {'P-Tokens':<8} | {'D-Tokens':<8} | {'TTFT P50':<9} | {'TTFT P95':<9} | {'Decode TPS':<11} | {'Wall P50':<9} | {'Hash Consistent':<15}", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        for name, run_list in results_by_prompt.items():
            ttfts = [r["ttft_s"] for r in run_list if r["ttft_s"] is not None]
            tps_list = [r["decode_tps"] for r in run_list if r["decode_tps"] is not None]
            walls = [r["wall_s"] for r in run_list]
            p_tok = run_list[0]["usage"].get("prompt_tokens", 0)
            d_tok = run_list[0]["completion_tokens"]
            
            med_ttft = statistics.median(ttfts) if ttfts else 0.0
            idx_p95 = int(0.95 * (len(ttfts) - 1)) if ttfts else 0
            p95_ttft = sorted(ttfts)[idx_p95] if ttfts else 0.0
            
            med_tps = statistics.median(tps_list) if tps_list else 0.0
            med_wall = statistics.median(walls) if walls else 0.0
            hashes = set(r["text_sha256"] for r in run_list)
            consistent = "YES (deterministic)" if len(hashes) == 1 else f"NO ({len(hashes)} variants)"
            
            print(
                f"{name:<14} | {p_tok:<8} | {d_tok:<8} | {med_ttft:<9.4f} | {p95_ttft:<9.4f} | {med_tps:<11.2f} | {med_wall:<9.3f} | {consistent:<15}",
                file=sys.stderr
            )
        print("=" * 80 + "\n", file=sys.stderr)
    finally:
        if destination is not sys.stdout:
            destination.close()


if __name__ == "__main__":
    main()
