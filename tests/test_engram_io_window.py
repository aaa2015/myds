#!/usr/bin/env python3
"""Fault-inject CPU exclusive-window restoration; no SSH, services or GPU.

Production commands, artifact hashes and process/health results are mocked.
Assertions cover preflight rejection and restoration after an actual wrapper
control-flow failure, not target-machine deployment success.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import runpy
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/2026-10-05-engram-io/scripts/run-cpu-exclusive.py"
HEAD = "3fb95ae9a7b41cf86f85ed37652e16396c73225b"
HASHES = {"myds": "4d27bd6a0278aea7198a8f54fcd6f1f2544ee5ae87178a71772c963afe3fc5e0",
          "myds.metallib": "d6e6bb7990f5694f50d7c2eb00227f2ad7c8b468252f66aa5d447fd800236e20"}
REAL_SHA = hashlib.sha256


class Child:
    def __init__(self, rc):
        self.rc = rc
        self.pid = 123456

    def wait(self, timeout=None):
        return self.rc

    def poll(self):
        return self.rc


class CPUWindowTests(unittest.TestCase):
    def execute(self, mode):
        with tempfile.TemporaryDirectory(prefix="myds-cpu-window-tests-") as td:
            work = Path(td)
            source = work / "source"
            production = work / "production"
            fleet = work / "fleet"
            (source / "bin").mkdir(parents=True)
            (source / "tools").mkdir()
            (production / "bin").mkdir(parents=True)
            fleet.mkdir()
            for name in HASHES:
                (production / "bin" / name).write_bytes(("mock-" + name).encode())
            for name in ["bench_engram_io.c", "run_engram_io.py", "summarize_engram_io.py"]:
                (source / "tools" / name).write_bytes(b"mock tool")
            (source / "bin/bench_engram_io").write_bytes(b"mock benchmark")
            (fleet / "fleet.conf").write_text("3|15|local-myds-deepseek-v4.1-flash-q2|8181\n")
            model = work / "model.gguf"
            model.write_bytes(b"mock model")
            normal = work / "normal.tokens.bin"
            normal.write_bytes(struct.pack("<" + "i" * 1831, *([1] * 1831)))
            stress = work / "stress512.tokens.bin"
            rng = random.Random(2026100503)
            stress.write_bytes(struct.pack("<" + "i" * 512, *(rng.randrange(129280) for _ in range(512))))
            stress.with_suffix(".bin.json").write_text(json.dumps(dict(
                sha256=REAL_SHA(stress.read_bytes()).hexdigest(), seed=2026100503,
                vocab_size=129280, tokens=512)))
            calls = []
            term = []
            stopped = False

            def sha(data=b""):
                for name, expected in HASHES.items():
                    if data == ("mock-" + name).encode():
                        class FakeHash:
                            def hexdigest(self):
                                return expected
                        return FakeHash()
                return REAL_SHA(data)

            def run(args, **kwargs):
                args = [str(s) for s in args]
                calls.append(args)
                text = ""
                rc = 0
                if args[0] == "lsof":
                    if "-sTCP:ESTABLISHED" in args:
                        rc = 0 if mode == "connections" else 1
                    else:
                        text = "441\n"
                elif args[0] == "ps":
                    text = "~/Projects/myds/bin/myds --ctx-size 262144\n"
                elif args[0] == "git":
                    text = HEAD + "\n"
                elif args[0] == "pgrep":
                    rc = 1
                elif args[0] == "python3":
                    text = '{"1":{},"4":{},"8":{}}\n'
                elif args[0] in ("vm_stat", "sysctl"):
                    text = "mock memory state\n"
                else:
                    raise AssertionError("unexpected command " + repr(args))
                return subprocess.CompletedProcess(args, rc, text, "")

            def popen(args, **kwargs):
                args = [str(s) for s in args]
                calls.append(args)
                if args[0] == "/bin/zsh":
                    self.assertEqual(args[1], str(fleet / "start_fleet.sh"))
                    return Child(0)
                self.assertEqual(args[0], "python3")
                self.assertTrue(args[1].endswith("tools/run_engram_io.py"))
                if mode == "benchmark_failure":
                    return Child(7)
                output = Path(args[args.index("--output") + 1])
                output.mkdir(parents=True)
                (output / "summary.json").write_text(json.dumps(dict(by_workers={"1": {}, "4": {}, "8": {}})))
                return Child(0)

            def kill(pid, signum):
                nonlocal stopped
                term.append((pid, signum))
                self.assertEqual(pid, 441)
                if signum == signal.SIGTERM:
                    stopped = True
                elif signum == 0 and stopped:
                    raise ProcessLookupError()
                elif signum != 0:
                    raise AssertionError("unexpected signal")

            argv = [str(SCRIPT), "--root", str(work), "--production", str(production),
                    "--fleet", str(fleet), "--model", str(model)]
            exitcode = 0
            with patch.object(sys, "argv", argv), patch("subprocess.run", side_effect=run), \
                    patch("subprocess.Popen", side_effect=popen), patch("hashlib.sha256", side_effect=sha), \
                    patch("os.kill", side_effect=kill), patch("signal.signal"), \
                    patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(b'{"status":"ok"}')), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    runpy.run_path(str(SCRIPT), run_name="__main__")
                except SystemExit as exc:
                    exitcode = exc.code
            outputs = list((work / "results").glob("cpu-exclusive-*"))
            self.assertEqual(len(outputs), 1)
            records = [json.loads(line) for line in (outputs[0] / "window.jsonl").read_text().splitlines()]
            return exitcode, calls, term, records

    def test_active_connections_reject_before_stop(self):
        exitcode, calls, term, rows = self.execute("connections")
        self.assertEqual(exitcode, 1)
        self.assertFalse(term)
        self.assertFalse(any(call[0] == "/bin/zsh" for call in calls))
        self.assertFalse(any(r["kind"] == "production_stop" for r in rows))
        self.assertIn("active production connections", rows[-1]["error"])

    def test_failed_cpu_measurement_restores_production(self):
        exitcode, calls, term, rows = self.execute("benchmark_failure")
        self.assertEqual(exitcode, 1)
        self.assertIn((441, signal.SIGTERM), term)
        restored = [r for r in rows if r["kind"] == "restoration_health"]
        self.assertEqual(len(restored), 1)
        self.assertTrue(restored[0]["ready"] and restored[0]["unchanged"])
        self.assertEqual(restored[0]["hashes"], HASHES)
        self.assertIn("normal failed", rows[-1]["error"])

    def test_success_runs_two_cpu_scopes_and_restores(self):
        exitcode, calls, term, rows = self.execute("success")
        self.assertEqual(exitcode, 0)
        self.assertEqual([r["name"] for r in rows if r["kind"] == "measurement_validated"], ["normal", "stress512"])
        self.assertIsNone(rows[-1]["error"])
        self.assertTrue(next(r for r in rows if r["kind"] == "restoration_health")["unchanged"])
        bench_calls = [call for call in calls if call[0] == "python3" and call[1].endswith("tools/run_engram_io.py")]
        self.assertEqual(len(bench_calls), 2)
        self.assertTrue(all(call[call.index("--order") + 1] == "1,4,8,8,4,1" for call in bench_calls))
        self.assertFalse(any("bench_phase" in arg for call in bench_calls for arg in call))


if __name__ == "__main__":
    unittest.main()
