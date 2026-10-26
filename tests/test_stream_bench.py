#!/usr/bin/env python3
"""CPU-only SSE fixtures for the online performance screen's measurement unit."""
import contextlib
import http.client
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import decode_fixed
import prefill_decode_bench
import stream_bench


def event(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


def fixture(chunks=120, output=128, finish="length", usage=True, done=True):
    rows = [event({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]})
            for _ in range(chunks)]
    rows.append(event({"choices": [{"delta": {}, "finish_reason": finish}]}))
    if usage:
        rows.append(event({"choices": [], "usage": {"completion_tokens": output,
                                                   "prompt_tokens": 42}}))
    if done:
        rows.append(b"data: [DONE]\n\n")
    return rows


def measure(rows, chunks=120, requested=128):
    # First text at 1s, last at 2s; usage/DONE finish at 9s. The seven-second
    # transport tail must not become decode work.
    times = iter([1 + i / max(chunks - 1, 1) for i in range(chunks)] + [9.])
    return stream_bench.read_stream(rows, requested, 0., clock=lambda: next(times))


class StreamMeasurementTests(unittest.TestCase):
    def test_128_tokens_120_chunks_uses_usage_and_127_intervals(self):
        row = measure(fixture())
        self.assertEqual(row["output_tokens"], 128)
        self.assertEqual(row["text_chunks"], 120)
        self.assertEqual(row["decode_est_tok_s"], 127.)
        self.assertEqual(row["visible_decode_s"], 1.)
        self.assertEqual(row["total_s"], 9.)
        self.assertEqual([row[k] for k in ("started_at_s", "first_text_at_s", "last_text_at_s", "ended_at_s")],
                         [0., 1., 2., 9.])
        self.assertTrue(row["done"])

    def test_one_chunk_per_token_does_not_count_first_token_twice(self):
        row = measure(fixture(chunks=128), chunks=128)
        self.assertEqual(row["decode_est_tok_s"], 127.)

    def test_reasoning_and_empty_role_chunks_are_not_tokens(self):
        rows = fixture(chunks=2, output=5)
        rows[0] = event({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
        rows.insert(0, event({"choices": [{"delta": {"role": "assistant", "content": ""}}]}))
        row = measure(rows, chunks=2, requested=5)
        self.assertEqual(row["text_chunks"], 2)
        self.assertEqual(row["decode_est_tok_s"], 4.)

    def test_single_token_is_ttft_without_decode_rate(self):
        row = measure(fixture(chunks=1, output=1), chunks=1, requested=1)
        self.assertEqual(row["ttft_s"], 1.)
        self.assertIsNone(row["decode_est_tok_s"])

    def test_missing_usage_truncation_early_stop_and_wrong_count_rejected(self):
        for kwargs in ({"usage": False}, {"done": False}, {"finish": "stop"},
                       {"finish": "error"}, {"output": 127}, {"output": 129},
                       {"output": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(stream_bench.BenchmarkError):
                measure(fixture(**kwargs))

    def test_no_visible_output_or_malformed_json_rejected(self):
        with self.assertRaises(stream_bench.BenchmarkError):
            measure(fixture(chunks=0), chunks=0)
        with self.assertRaises(stream_bench.BenchmarkError):
            measure([b"data: invalid json\n"])

    def test_partial_stream_timeout_rejected(self):
        def interrupted():
            yield fixture()[0]
            raise TimeoutError("fixture timeout")
        with self.assertRaisesRegex(stream_bench.BenchmarkError, "interrupted") as caught:
            measure(interrupted())
        self.assertEqual(caught.exception.metadata["text_chunks"], 1)
        self.assertEqual(caught.exception.metadata["first_text_at_s"], 1.)
        self.assertFalse(caught.exception.metadata["done"])

    def test_rejected_truncation_retains_timing_usage_and_finish(self):
        with self.assertRaises(stream_bench.BenchmarkError) as caught:
            measure(fixture(done=False))
        self.assertEqual(caught.exception.metadata["usage"]["completion_tokens"], 128)
        self.assertEqual(caught.exception.metadata["finish"], "length")
        self.assertEqual(caught.exception.metadata["last_text_at_s"], 2.)
        self.assertEqual(caught.exception.metadata["ended_at_s"], 9.)
        self.assertFalse(caught.exception.metadata["done"])

    def test_http_timeout_before_first_text_rejected(self):
        with mock.patch.object(stream_bench.urllib.request, "urlopen", side_effect=TimeoutError("fixture")):
            with self.assertRaisesRegex(stream_bench.BenchmarkError, "HTTP request failed"):
                stream_bench.request_stream(8187, "fixture", "prompt", 128)

    def test_http_content_length_truncation_rejected(self):
        def interrupted():
            yield fixture()[0]
            raise http.client.IncompleteRead(b"partial", 20)
        with self.assertRaisesRegex(stream_bench.BenchmarkError, "interrupted"):
            measure(interrupted())

    def test_request_preserves_token_limit_and_requests_usage(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __iter__(self): return iter(fixture(chunks=2, output=17))
        with mock.patch.object(stream_bench.urllib.request, "urlopen", return_value=Response()) as opener:
            stream_bench.request_stream(8187, "fixture", "prompt", 17)
        body = json.loads(opener.call_args[0][0].data)
        self.assertEqual(body["max_tokens"], 17)
        self.assertTrue(body["stream_options"]["include_usage"])

    def test_cli_token_options_and_rejections_have_failure_exit(self):
        for mod, args, expected in ((decode_fixed, ["8187", "fixture", "--tokens", "17"], 17),
                                    (prefill_decode_bench, ["8187", "10", "--decode-tokens", "17"], 17)):
            calls = []
            def fail(*args, **kwargs):
                calls.append(args[3])
                raise stream_bench.BenchmarkError("fixture missing usage")
            with self.subTest(module=mod.__name__), mock.patch.object(mod, "request_stream", side_effect=fail):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(mod.main(args), 1)
            self.assertIn(expected, calls)

    def test_raw_jsonl_keeps_warmup_timing_usage_and_done(self):
        row = measure(fixture())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            with mock.patch.object(decode_fixed, "request_stream", return_value=row), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(decode_fixed.main(["8187", "fixture", "--jsonl", str(path)]), 0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 12)
        self.assertEqual(sum(record["warmup"] for record in records), 3)
        self.assertTrue(all(record["usage"]["completion_tokens"] == 128 and record["done"] for record in records))
        self.assertTrue(all(record["first_text_at_s"] == 1. for record in records))


if __name__ == "__main__":
    unittest.main()
