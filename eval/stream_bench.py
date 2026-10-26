"""HTTP streaming measurements for online performance screening.

SSE deltas are transport events, not tokens. Final usage supplies token counts.
HTTP exposes first visible text, which may follow multiple byte-BPE tokens due
to UTF-8/reasoning buffering. The decode rate is therefore an estimate; native
measurements are required for exact decode intervals.
"""
import http.client
import json
import time
import urllib.error
import urllib.request


class BenchmarkError(ValueError):
    """An incomplete response or an unsuitable measurement sample."""

    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}


def _token_count(usage, field):
    value = usage.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError("missing or invalid usage." + field)
    return value


def read_stream(response, requested_tokens, started_at, clock=None):
    """Require final usage, length finish, exact requested count, and [DONE].

    Estimate=(completion_tokens-1)/(last_text-first_text), excluding the usage
    and DONE tail. This assumes first visible text represents the first output
    token; usage can include invisible/special tokens, so this is not an exact
    decode measurement. Preserve all arrival times and response metadata.
    """
    if requested_tokens < 1:
        raise BenchmarkError("requested_tokens must be positive")
    clock = clock or time.monotonic
    first = last = None
    chunks = 0
    usage, finish, done = None, None, False
    ended_at = None

    def reject(message):
        raise BenchmarkError(message, {
            "started_at_s": started_at, "first_text_at_s": first,
            "last_text_at_s": last, "ended_at_s": ended_at,
            "text_chunks": chunks, "usage": usage, "finish": finish,
            "done": done, "requested_tokens": requested_tokens})

    try:
        for raw in response:
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                reject("invalid UTF-8 in SSE response: " + str(exc))
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            try:
                event = json.loads(payload)
            except (ValueError, TypeError) as exc:
                reject("invalid SSE JSON: " + str(exc))
            if not isinstance(event, dict) or event.get("error"):
                reject("server returned an error or invalid event")
            if event.get("usage") is not None:
                if not isinstance(event["usage"], dict):
                    reject("invalid usage object")
                if usage is not None and usage != event["usage"]:
                    reject("conflicting usage objects")
                usage = event["usage"]
            choices = event.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                reject("benchmark requires one output choice")
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                reject("invalid output choice")
            if choice.get("finish_reason") is not None:
                if finish is not None and finish != choice["finish_reason"]:
                    reject("conflicting finish reasons")
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                reject("invalid text delta")
            pieces = (delta.get("content"), delta.get("reasoning_content"))
            if any(piece is not None and not isinstance(piece, str) for piece in pieces):
                reject("invalid text delta")
            if any(pieces):
                arrival = clock()
                if first is None:
                    first = arrival
                last = arrival
                chunks += 1
    except (OSError, TimeoutError, http.client.IncompleteRead) as exc:
        reject("stream interrupted: " + str(exc))
    ended_at = clock()
    if not done:
        reject("truncated stream: missing [DONE]")
    if usage is None:
        reject("missing final usage; text chunks cannot count tokens")
    try:
        output_tokens = _token_count(usage, "completion_tokens")
        prompt_tokens = _token_count(usage, "prompt_tokens")
    except BenchmarkError as exc:
        reject(str(exc))
    if finish != "length" or output_tokens != requested_tokens:
        reject("short or invalid run: finish=%s, output=%s, requested=%s" %
               (finish, output_tokens, requested_tokens))
    if first is None:
        reject("no visible text in completed response")
    ttft, span, total = first - started_at, last - first, ended_at - started_at
    if ttft < 0 or span < 0 or total < ttft + span:
        reject("nonmonotonic arrival times")
    rate = (output_tokens - 1) / span if output_tokens > 1 and span > 0 else None
    return {"ttft_s": ttft, "total_s": total, "visible_decode_s": span,
            "started_at_s": started_at, "first_text_at_s": first,
            "last_text_at_s": last, "ended_at_s": ended_at,
            "text_chunks": chunks, "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens, "requested_tokens": requested_tokens,
            "decode_est_tok_s": rate, "finish": finish, "done": done, "usage": usage}


def request_stream(port, alias, prompt, max_tokens, timeout=1800):
    body = json.dumps({
        "model": alias, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=body, headers={"Content-Type": "application/json"})
    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return read_stream(response, max_tokens, started_at)
    except (OSError, TimeoutError, urllib.error.URLError, http.client.IncompleteRead) as exc:
        raise BenchmarkError("HTTP request failed: " + str(exc), {
            "started_at_s": started_at, "ended_at_s": time.monotonic(),
            "first_text_at_s": None, "last_text_at_s": None,
            "text_chunks": 0, "usage": None, "finish": None,
            "done": False, "requested_tokens": max_tokens}) from exc
