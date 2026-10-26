#!/usr/bin/env python3
"""CPU-only HTTP/parser/lifecycle tests; never invokes a model or Metal."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("register_http", ROOT / "tools/run_register_only_http_bridge.py")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def event(content=None, finish=None, usage=None):
    value = {"choices": [{"index": 0, "delta": {} if content is None else {"content": content}, "finish_reason": finish}]}
    if usage is not None:
        value["usage"] = usage
    return json.dumps(value, ensure_ascii=False).encode()


USAGE = {"prompt_tokens": 38404, "completion_tokens": 256, "total_tokens": 38660}


class FakeSocket:
    def __init__(self):
        self.closed = False

    def getsockname(self):
        return ("127.0.0.1", 54321)

    def settimeout(self, timeout):
        assert timeout > 0

    def shutdown(self, how):
        assert how == bridge.socket.SHUT_RDWR
        self.closed = True


class FakeResponse:
    status = 200

    def __init__(self, lines):
        self.lines = iter(lines)
        self.closed = False

    def getheader(self, name, default=None):
        return "text/event-stream"

    def getheaders(self):
        return [("Content-Type", "text/event-stream")]

    def readline(self):
        return next(self.lines, b"")

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.sock = FakeSocket()
        self.closed = False

    def connect(self):
        pass

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class FakeProcess:
    pid = 7654
    returncode = None

    def __init__(self):
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)

    def wait(self, timeout):
        self.returncode = -signal.SIGTERM


class HTTPBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="myds-http-host-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_stream_counts_usage_not_deltas_and_retains_exact_bytes(self):
        payloads = [event("你"), event("好\n"), event(finish="length", usage=USAGE), b"[DONE]"]
        lines = [part for value in payloads for part in (b"data: " + value + b"\n", b"\n")]
        connection = FakeConnection(FakeResponse(lines))
        with mock.patch.object(bridge.http.client, "HTTPConnection", return_value=connection):
            result = bridge.exchange(8198, b'{"stream":true}', self.root / "sample")
        self.assertEqual(result["usage"]["completion_tokens"], 256)
        self.assertEqual(result["visible_events_not_tokens"], 2)
        self.assertEqual(result["client_port"], 54321)
        self.assertEqual((self.root / "sample/output.utf8.bin").read_bytes(), "你好\n".encode())
        self.assertEqual((self.root / "sample/response.sse.bin").read_bytes(), b"".join(lines))
        self.assertTrue(connection.closed and connection.response.closed)

    def test_truncated_stream_preserves_partial_output_raw_and_failure(self):
        lines = [b"data: " + event("partial") + b"\n", b"\n"]
        connection = FakeConnection(FakeResponse(lines))
        with mock.patch.object(bridge.http.client, "HTTPConnection", return_value=connection), self.assertRaises(bridge.BridgeError):
            bridge.exchange(8198, b"{}", self.root / "failed")
        record = json.loads((self.root / "failed/observation.json").read_text())
        self.assertFalse(record["success"])
        self.assertIn("error", record)
        self.assertEqual((self.root / "failed/output.utf8.bin").read_bytes(), b"partial")
        self.assertEqual((self.root / "failed/response.sse.bin").read_bytes(), b"".join(lines))

    def test_interrupt_only_after_visible_decode_and_closes_socket(self):
        lines = [part for i in range(12) for part in (b"data: " + event(str(i)) + b"\n", b"\n")]
        connection = FakeConnection(FakeResponse(lines))
        with mock.patch.object(bridge.http.client, "HTTPConnection", return_value=connection):
            result = bridge.exchange(8198, b"{}", self.root / "cancel", interrupt_after_events=8)
        self.assertTrue(connection.sock.closed)
        self.assertTrue(result["intentionally_interrupted"])
        self.assertEqual(result["visible_events_not_tokens"], 8)
        self.assertFalse(result["done"])

    def test_n256_and_done_are_required_not_empty_output(self):
        state = bridge.StreamState(1)
        state.event(event(finish="length", usage=USAGE), 2)
        state.event(b"[DONE]", 3)
        with self.assertRaises(bridge.BridgeError):
            state.finish_check()
        for value in (True, 255, -1, "256"):
            with self.subTest(value=value), self.assertRaises(bridge.BridgeError):
                bridge.counts(dict(USAGE, completion_tokens=value))

    def test_logs_match_client_port_not_neighbor_request_or_name(self):
        prefix = b"[I] [127.0.0.1:54321] Completed ChatCompletions /v1/chat/completions - "
        tail = b", ttft: 1.0s, decode: 256 tok in 1.0s = 255.0 t/s, wall: 2.0s (infer 2.0s, queue 0.0s), finish_reason: length\n"
        cold = prefix + b"prompt: 38404 tok, prefill: 38403 tok in 80.0s = 480.0 t/s (cached 0)" + tail
        warm = prefix + b"prompt: 38404 tok, prefill: none (all 38404 cached)" + tail
        neighbor = cold.replace(b":54321]", b":55555]")
        self.assertIsNone(bridge.completed_record(neighbor, 54321))
        c = bridge.completed_record(neighbor + cold, 54321)
        w = bridge.completed_record(warm, 54321)
        bridge.check_completed(c, {"usage": USAGE}, "cold")
        bridge.check_completed(w, {"usage": USAGE}, "warm")
        with self.assertRaises(bridge.BridgeError):
            bridge.check_completed(w, {"usage": USAGE}, "cold")
        with self.assertRaises(bridge.BridgeError):
            bridge.completed_record(cold + cold, 54321)

    def test_unknown_process_is_never_signalled(self):
        with mock.patch.object(bridge, "inference_processes", return_value=[{"pid": 999, "command": "myds"}]), \
                mock.patch.object(bridge.os, "kill") as kill, self.assertRaises(bridge.BridgeError):
            bridge.exclusive(owned_pid=7654)
        kill.assert_not_called()

    def test_owned_cleanup_uses_only_sigterm(self):
        child = FakeProcess()
        self.assertTrue(bridge.terminate_owned(child)["terminated"])
        self.assertEqual(child.signals, [signal.SIGTERM])
        child = FakeProcess()
        with mock.patch.object(child, "wait", side_effect=subprocess.TimeoutExpired("owned", 30)), self.assertRaises(bridge.BridgeError):
            bridge.terminate_owned(child)
        self.assertEqual(child.signals, [signal.SIGTERM])

    def test_config_and_environment_do_not_access_diskcache_or_override_mode(self):
        config = self.root / "config"
        config.write_text("MYDS_PREFIX_CACHE=1\n")
        self.assertEqual(bridge.inspect_config(config), {"MYDS_PREFIX_CACHE": "1"})
        for text in ("MYDS_KV_DSK=/production\n", "MYDS_INDEXER_REGISTER_ONLY=1\n", "MYDS_KV_DSK_BUDGET_MB=9000\n", "MYDS_RESP_STATE=/production\n"):
            config.write_text(text)
            with self.assertRaises(bridge.BridgeError):
                bridge.inspect_config(config)
        with mock.patch.dict(os.environ, {"MYDS_KV_DSK": "/production", "MYDS_OTHER": "1"}):
            self.assertFalse(any(k.startswith("MYDS_") for k in bridge.clean_environment(None)))
            self.assertEqual({k: v for k, v in bridge.clean_environment(1).items() if k.startswith("MYDS_")}, {bridge.MODE: "1"})

    def test_loaded_library_path_must_be_actual_adjacent_library(self):
        library = self.root / "myds.metallib"
        library.write_bytes(b"test")
        line = f"loaded precompiled library from {library} (all required V4.1 kernels verified)"
        self.assertEqual(bridge.loaded_library(line, library), str(library.resolve()))
        for bad in ("source compilation completed", line.replace(str(library), "/wrong/myds.metallib"), line + "\n" + line):
            with self.assertRaises(bridge.BridgeError):
                bridge.loaded_library(bad, library)

    def test_request_preserves_prompt_and_sampling_with_only_transport_changes(self):
        request = {"model": "retained", "messages": [{"role": "user", "content": "你好"}],
                   "max_tokens": 256, "temperature": 0, "thinking": False, "seed": 42, "stream": False}
        path = self.root / "request.json"
        path.write_text(json.dumps(request))
        actual = json.loads(bridge.prepare_request(path))
        self.assertEqual(actual, dict(request, stream=True, stream_options={"include_usage": True}))

    def test_failed_preflight_keeps_summary_and_starts_no_service(self):
        stage = self.root / "stage"
        missing = self.root / "missing"
        argv = ["--stage-root", str(stage), "--gold-exe", str(missing), "--candidate-exe", str(missing),
                "--config", str(missing), "--model", str(missing), "--inputs-root", str(missing)]
        with mock.patch.object(bridge.subprocess, "Popen") as launch:
            self.assertEqual(bridge.main(argv), 1)
        launch.assert_not_called()
        self.assertFalse(json.loads((stage / "summary.json").read_text())["http_functional_bridge_passed"])
        with self.assertRaises(bridge.BridgeError):
            bridge.main(argv)

    def test_startup_failure_terminates_only_owned_child_and_retains_receipt(self):
        executable = self.root / "myds"
        executable.write_bytes(b"mock")
        (self.root / "myds.metallib").write_bytes(b"mock library")
        config = self.root / "config"
        config.write_text("MYDS_PREFIX_CACHE=1\n")
        args = types.SimpleNamespace(stage_root=self.root, config=config, model=self.root / "model", port=8198)
        child = FakeProcess()
        with mock.patch.object(bridge, "exclusive", return_value=[]), \
                mock.patch.object(bridge, "port_free"), \
                mock.patch.object(bridge.subprocess, "Popen", return_value=child) as popen, \
                mock.patch.object(bridge, "health", side_effect=bridge.BridgeError("fixture startup failure")), \
                self.assertRaises(bridge.BridgeError):
            bridge.run_arm("gold", executable, None, args, {}, {})
        self.assertEqual(child.signals, [signal.SIGTERM])
        result = json.loads((self.root / "gold/arm-result.json").read_text())
        self.assertFalse(result["passed"])
        self.assertTrue(result["cleanup"]["terminated"])
        child_env = popen.call_args.kwargs["env"]
        self.assertNotIn(bridge.MODE, child_env)
        self.assertEqual(child_env["MYDS_RESP_STATE"], str(self.root / "gold/responses.idx"))


if __name__ == "__main__":
    unittest.main()
