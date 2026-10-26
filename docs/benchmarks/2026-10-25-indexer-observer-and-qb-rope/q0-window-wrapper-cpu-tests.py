#!/usr/bin/env python3
"""CPU fixtures: native processes are always mocked, no model/device/network."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("q0_wrapper", ROOT / "q0-window-wrapper.py")
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)


class DummyProcess:
    pid = 7654321

    def __init__(self, returncode=0):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class Q0WindowCpuTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT / "fullmodel-q-protocol.json").read_text())
        self.p["artifacts"]["reader"] = dict(path="/frozen/reader.py", bytes=1, sha256="a" * 64)

    def test_strip_all_MYDS_then_add_only_mode_and_two_sinks(self):
        inherited = dict(PATH="/bin", MYDS_KPROF="1", MYDS_PLD="0", MYDS_FAKE="x", MYDS_INDEXER_K_RESIDENT="0")
        env = w.child_environment(inherited, 1, "/raw/logits", "/raw/state")
        self.assertEqual({k: v for k, v in env.items() if k.startswith("MYDS_")},
                         dict(MYDS_INDEXER_K_RESIDENT="1", MYDS_BENCH_LOGITS_DIR="/raw/logits", MYDS_BENCH_STATE_DIR="/raw/state"))
        self.assertEqual(inherited["MYDS_PLD"], "0")
        self.assertEqual(env["PATH"], "/bin")

    def test_exact_same_binary_argv_and_deployed_library_anchor(self):
        self.assertEqual([p["mode"] for p in self.p["providers"]], [0, 0, 1])
        self.assertEqual(len({p["binary"]["sha256"] for p in self.p["providers"]}), 1)
        for p in self.p["providers"]:
            cmd = w.command(self.p, p)
            self.assertEqual(cmd[1:3], ["--config", self.p["artifacts"]["config"]["path"]])
            self.assertEqual(cmd[-3:], ["1", "64", "327680"])
            self.assertNotIn("--dspark", cmd)
            self.assertEqual(p["expected_loaded_library_path"], p["library"]["path"])
        self.assertNotEqual(self.p["providers"][0]["library"]["sha256"], self.p["providers"][1]["library"]["sha256"])

    def test_complete_raw_inventory_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "state" / "run-0").mkdir(parents=True)
            (root / "logits").mkdir()
            for name, raw in (("stdout.jsonl", b"{}\n"), ("stderr.txt", b""),
                              ("state/run-0/a.bin", b"all bytes"), ("logits/run-0.f32", bytes(4))):
                (root / name).write_bytes(raw)
            inv = w.raw_inventory(root)
            self.assertEqual(inv["file_count"], 4)
            self.assertEqual(inv["files"]["state/run-0/a.bin"]["sha256"], w.sha(root / "state/run-0/a.bin"))
            (root / "state" / "alias.bin").symlink_to(root / "state/run-0/a.bin")
            with self.assertRaises(ValueError):
                w.raw_inventory(root)

    def test_artifact_and_model_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model"; p.write_bytes(b"prefix" + b"body")
            item = dict(path=str(p), bytes=p.stat().st_size, sha256=w.sha(p))
            w.verify_descriptor(item)
            st = p.stat()
            model = dict(path=str(p), bytes=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns,
                         prefix_bytes=6, prefix_sha256=__import__("hashlib").sha256(b"prefix").hexdigest())
            actual, prefix = w.model_state(dict(model=model))
            self.assertEqual(prefix, b"prefix")
            self.assertFalse(actual["whole_model_payload_sha256_verified"])
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError): w.verify_descriptor(item)
            with self.assertRaises(ValueError): w.model_state(dict(model=model))

    def collect(self, root, factory, process_rows=None):
        with mock.patch.object(w, "canonical_state", return_value={"artifact": "same"}), \
                mock.patch.object(w, "model_state", return_value=({"prefix": "same"}, b"prefix")), \
                mock.patch.object(w, "processes", side_effect=process_rows or [[], []]), \
                mock.patch.object(w.subprocess, "Popen", side_effect=factory):
            return w.collect_provider(self.p, self.p["providers"][0], root / "G_lib", "p" * 64, [])

    def test_success_is_collection_only_and_reader_is_bound(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_start(*args, **kwargs):
                kinds = ["indexer_metadata", "metadata", "load", "sample", "sample"]
                kwargs["stdout"].write(("\n".join(json.dumps(dict(kind=k)) for k in kinds) + "\n").encode())
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(len([k for k in kwargs["env"] if k.startswith("MYDS_")]), 3)
                return DummyProcess()
            result = self.collect(Path(d), fake_start)
            self.assertIsNone(result["abort_reason"])
            self.assertIsNone(result["qualification_passed"])
            self.assertFalse(result["performance_acceptance"])
            self.assertEqual(result["reader_sha256"], "a" * 64)
            self.assertTrue((Path(d) / "G_lib/raw-inventory.json").exists())

    def test_nonzero_and_bad_stdout_are_saved_as_collection_failures(self):
        for code, content in ((1, b"failure raw\n"), (0, b"not JSON\n")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as d:
                def fake_start(*args, **kwargs):
                    kwargs["stdout"].write(content)
                    return DummyProcess(code)
                result = self.collect(Path(d), fake_start)
                self.assertIsNotNone(result["abort_reason"])
                self.assertEqual((Path(d) / "G_lib/stdout.jsonl").read_bytes(), content)
                self.assertTrue((Path(d) / "G_lib/window.json").exists())

    def test_timeout_stops_only_mock_owned_process_and_preserves_all_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            proc = DummyProcess(None)
            def fake_stop(p):
                self.assertIs(p, proc)
                p.returncode = -15
            with mock.patch.object(w, "terminate_owned", side_effect=fake_stop) as stop, \
                    mock.patch.object(w.time, "monotonic", side_effect=[0, 601, 602, 603]):
                result = self.collect(Path(d), lambda *a, **kw: proc,
                                      [[], [dict(pid=proc.pid, comm="bench_indexer_fullmodel")], []])
            stop.assert_called_once()
            self.assertEqual(result["abort_reason"], "timeout")
            self.assertEqual(result["exit_code"], -15)
            self.assertTrue((Path(d) / "G_lib/raw-inventory.json").exists())

    def test_existing_output_root_refused_before_lock_or_native_start(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = dict(self.p, window_root=str(root), wrapper_sha256=w.sha(ROOT / "q0-window-wrapper.py"))
            protocol = root / "protocol.json"; protocol.write_text(json.dumps(p))
            with mock.patch.object(sys, "argv", ["wrapper", "--protocol", str(protocol)]), \
                    mock.patch.object(w.subprocess, "Popen", side_effect=AssertionError("native forbidden")):
                with self.assertRaises(ValueError): w.main()

    def test_interrupt_handler_unwinds_without_signalling_any_real_process(self):
        with self.assertRaises(InterruptedError):
            w.interrupted(15, None)

    def test_outer_window_stops_at_first_failed_provider_without_retry_or_next_stage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = dict(self.p, window_root=str(root / "Q0-window"), lock_path=str(root / "fixture.lock"),
                     wrapper_sha256=w.sha(ROOT / "q0-window-wrapper.py"))
            protocol = root / "protocol.json"; protocol.write_text(json.dumps(p))
            failed = [dict(exit_code=0, abort_reason=None), dict(exit_code=1, abort_reason="native nonzero exit")]
            with mock.patch.object(sys, "argv", ["wrapper", "--protocol", str(protocol)]), \
                    mock.patch.object(w, "verify_descriptor"), mock.patch.object(w, "canonical_state", return_value={"same": 1}), \
                    mock.patch.object(w, "model_state", return_value=({"same": 1}, b"prefix")), \
                    mock.patch.object(w, "processes", return_value=[]), mock.patch.object(w.signal, "signal"), \
                    mock.patch.object(w, "collect_provider", side_effect=failed) as collect, \
                    mock.patch.object(w.subprocess, "Popen", side_effect=AssertionError("native forbidden")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.main(), 2)
            self.assertEqual(collect.call_count, 2)
            result = json.loads((root / "Q0-window/window.json").read_text())
            self.assertFalse(result["execution_complete"])
            self.assertIsNone(result["qualification_passed"])
            self.assertFalse(result["performance_acceptance"])
            self.assertEqual(len(result["providers"]), 2)

    def test_three_zero_exits_still_require_external_reader(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = dict(self.p, window_root=str(root / "Q0-window"), lock_path=str(root / "fixture.lock"),
                     wrapper_sha256=w.sha(ROOT / "q0-window-wrapper.py"))
            protocol = root / "protocol.json"; protocol.write_text(json.dumps(p))
            with mock.patch.object(sys, "argv", ["wrapper", "--protocol", str(protocol)]), \
                    mock.patch.object(w, "verify_descriptor"), mock.patch.object(w, "canonical_state", return_value={"same": 1}), \
                    mock.patch.object(w, "model_state", return_value=({"same": 1}, b"prefix")), \
                    mock.patch.object(w, "processes", return_value=[]), mock.patch.object(w.signal, "signal"), \
                    mock.patch.object(w, "collect_provider", return_value=dict(exit_code=0, abort_reason=None)) as collect, \
                    mock.patch.object(w.subprocess, "Popen", side_effect=AssertionError("native forbidden")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.main(), 0)
            self.assertEqual(collect.call_count, 3)
            result = json.loads((root / "Q0-window/window.json").read_text())
            self.assertTrue(result["execution_complete"])
            self.assertTrue(result["external_reader_required"])
            self.assertIsNone(result["qualification_passed"])
            self.assertFalse(result["fullmodel_speedup_proven"])


if __name__ == "__main__":
    unittest.main()
