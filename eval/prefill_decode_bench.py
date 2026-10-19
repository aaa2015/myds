#!/usr/bin/env python3
"""prefill_decode_bench.py — 测在线服务的 prefill 与 decode 性能。

要点
----
· prefix cache 默认开启，重复的 prompt 会命中缓存 ⇒ 每次用【唯一的随机前缀】，
  并在结果里记录 usage.prompt_tokens 以确认真的走了全量 prefill。
· prefill 用 max_tokens=1 隔离：TTFT 主要是 prefill + 首 token。
  prefill_tok_s = prompt_tokens / (TTFT − 1 个 token 的 decode 时间)，后者可忽略不计。
· decode 用固定短 prompt + 长输出，测首 token 到末 token 的速率。

用法: prefill_decode_bench.py <port> [ctx_sizes...]
"""
import json
import random
import statistics as st
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
SIZES = [int(x) for x in sys.argv[2:]] or [512, 2048, 8192, 16384, 32768]
ALIAS = "local-myds-deepseek-v4.1-flash-q2"

# 一段约 8 token 的填充文本，重复拼接以构造不同长度
FILLER = ("The compressor reads rows from the table and folds them into the "
          "residual stream before the next sublayer begins its work. ")


def make_prompt(target_tokens, nonce):
    # 约 4 字符/token 的粗估，最后用 usage 校正误差
    body = (FILLER * (target_tokens // 14 + 2))
    return f"[nonce {nonce}] " + body[: target_tokens * 4]


def call(prompt, max_tokens, timeout=1800):
    body = json.dumps({
        "model": ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        # 流式默认不回 usage，prefill 的 prompt_tokens 只能靠它
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    chunks = 0
    usage = None
    last_fr = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or []
            if not choices:
                continue          # usage-only 尾巴，choices 为空
            fr = choices[0].get("finish_reason")
            if fr:
                last_fr = fr
            delta = choices[0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if ttft is None:
                    ttft = time.monotonic() - t0
                chunks += 1
    total = time.monotonic() - t0
    return {"ttft": ttft, "total": total, "chunks": chunks,
            "usage": usage, "finish": last_fr}


def main():
    print(f"  ── prefill（max_tokens=1，唯一前缀避免缓存）──")
    print(f"  {'目标':>7} {'实际tok':>8} {'TTFT s':>9} {'prefill tok/s':>14}")
    prefill_rows = []
    for size in SIZES:
        nonce = random.getrandbits(48)
        p = make_prompt(size, nonce)
        r = call(p, 1)
        ptok = (r["usage"] or {}).get("prompt_tokens")
        if not ptok or not r["ttft"]:
            print(f"  {size:>7} {str(ptok):>8} {'—':>9} {'失败':>14}")
            continue
        rate = ptok / r["ttft"]
        prefill_rows.append((ptok, r["ttft"], rate))
        print(f"  {size:>7} {ptok:>8} {r['ttft']:>9.2f} {rate:>14.1f}")

    print()
    print(f"  ── decode（约 1K 输入，输出 256 token）──")
    print(f"  {'run':>4} {'TTFT s':>8} {'chunks':>7} {'decode tok/s':>13} {'总时长 s':>10} {'finish':>10}")
    dec_rows = []
    for run in range(3):
        nonce = random.getrandbits(48)
        p = make_prompt(1024, nonce)
        r = call(p, 512)
        if not r["ttft"] or r["chunks"] < 64:
            print(f"  {run:>4} {'样本不足 chunks=%s finish=%s' % (r['chunks'], r['finish']):>8}")
            continue
        dec = r["total"] - r["ttft"]
        rate = r["chunks"] / dec
        dec_rows.append(rate)
        print(f"  {run:>4} {r['ttft']:>8.2f} {r['chunks']:>7} {rate:>13.2f} {r['total']:>10.2f} {str(r['finish']):>10}")

    print()
    if prefill_rows:
        print("  ── prefill 汇总 ──")
        for ptok, ttft, rate in prefill_rows:
            print(f"    {ptok:>7} tok  {ttft:>7.2f} s  {rate:>8.1f} tok/s")
    if dec_rows:
        print(f"  ── decode 中位数: {st.median(dec_rows):.2f} tok/s "
              f"(范围 {min(dec_rows):.2f}–{max(dec_rows):.2f}) ──")


if __name__ == "__main__":
    main()
