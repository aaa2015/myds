#!/usr/bin/env python3
"""CPU-only long-stage window fixtures. Every native launch is mocked."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("long_wrapper", ROOT / "fullmodel-long-window-wrapper.py")
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)


class DummyProcess:
    pid, returncode = 7654321, 0
    def poll(self): return self.returncode
    def wait(self, timeout=None): return self.returncode


class LongWindowCpuTests(unittest.TestCase):
    def protocol(self, stage):
        p = json.loads((ROOT / ("fullmodel-qlong-protocol.json" if stage == "Qlong" else "fullmodel-p-protocol.json")).read_text())
        p["artifacts"]["reader"] = dict(path="/frozen/reader.py", bytes=1, sha256="a" * 64)
        return p

    def test_full_helper_inverse_proof_restores_immutable_Q0(self):
        original = (ROOT / "q0-window-wrapper.py").read_text()
        actual = (ROOT / "fullmodel-long-window-wrapper.py").read_text()
        proof = json.loads((ROOT / "fullmodel-long-wrapper-mechanics-proof.json").read_text())
        restored = actual.split("def selected_workload(", 1)[0]
        for edit in reversed(proof["exact_mechanics_edits"]):
            self.assertEqual(restored.count(edit["after"]), edit["count"])
            restored = restored.replace(edit["after"], edit["before"])
        self.assertEqual(restored, original.split("def main():\n", 1)[0])
        self.assertEqual(proof["source_q0_wrapper_sha256"], hashlib.sha256(original.encode()).hexdigest())
        self.assertEqual(proof["new_wrapper_sha256"], hashlib.sha256(actual.encode()).hexdigest())

    def test_two_real_frozen_token_files_request_parameters_and_generation_counts(self):
        for stage in ("Qlong", "P"):
            p = self.protocol(stage)
            self.assertEqual([x["prompt_tokens"] for x in p["workloads"]], [29609, 38404])
            for work in p["workloads"]:
                for provider in p["providers"]:
                    selected = w.selected_workload(p, work, "b" * 64)
                    self.assertEqual(w.command(selected, provider)[-3:], ["1" if stage == "Qlong" else "3", "64", "327680"])
                    self.assertEqual(w.command(selected, provider)[5], work["artifacts"]["request"]["path"])
                    self.assertEqual(selected["request_contract"]["prompt_tokens"], work["prompt_tokens"])
            self.assertEqual(2 * 3 * (p["request_contract"]["repeats"] + 1), 12 if stage == "Qlong" else 24)
        p = self.protocol("P")
        self.assertEqual([x["name"] for x in p["providers"]], ["A1", "B", "A2"])
        self.assertEqual([x["mode"] for x in p["providers"]], [0, 1, 0])
        self.assertEqual(len({x["binary"]["sha256"] for x in p["providers"]}), 1)
        self.assertEqual(len({x["library"]["sha256"] for x in p["providers"]}), 1)

    def test_root_approved_P_formulas_are_prospective_and_exact(self):
        gate = self.protocol("P")["performance_gate"]
        self.assertEqual(gate["primary_min_saved_percent_strict"], 2.0)
        self.assertEqual(gate["max_abs_control_drift_percent_inclusive"], 1.0)
        self.assertEqual(gate["max_nonregression_percent_inclusive"], .5)
        self.assertEqual(gate["formal_runs"], [1, 2, 3])
        self.assertEqual(gate["nonregression_metrics"], ["ttft_s", "decode_ms_per_token"])
        self.assertEqual(gate["primary_saved_percent_formula"], "100*(1-B/min(A1,A2))")
        self.assertEqual(gate["nonregression_percent_formula"], "100*(B/min(A1,A2)-1)")

    def test_prerequisite_report_false_or_wrong_reader_protocol_cannot_launch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            for stage in ("Qlong", "P"):
                p = self.protocol(stage)
                row = dict(review_completed=True, qualification_passed=True, long_workload_qualification_completed=True,
                           protocol_sha256="p" * 64, reader_sha256="r" * 64)
                for mutation in (None, "qualification_passed", "protocol_sha256", "reader_sha256", "long_workload_qualification_completed"):
                    changed = dict(row)
                    if mutation is not None: changed[mutation] = False if mutation.endswith("passed") or mutation.endswith("completed") else "wrong"
                    path.write_text(json.dumps(changed))
                    p["prerequisite"] = dict(protocol_sha256="p" * 64, reader_sha256="r" * 64,
                        **({"report": dict(path=str(path), bytes=path.stat().st_size, sha256=w.sha(path))} if stage == "Qlong" else {"report_path": str(path)}))
                    if mutation is None or (stage == "Qlong" and mutation == "long_workload_qualification_completed"):
                        self.assertEqual(w.prerequisite(p)["sha256"], w.sha(path))
                    else:
                        with self.assertRaises(ValueError): w.prerequisite(p)

    def test_P_collect_retains_four_sample_records_and_never_qualifies_itself(self):
        p = self.protocol("P")
        selected = w.selected_workload(p, p["workloads"][0], "b" * 64)
        with tempfile.TemporaryDirectory() as d:
            def fake_start(*args, **kwargs):
                kinds = ["indexer_metadata", "metadata", "load"] + ["sample"] * 4
                kwargs["stdout"].write(("\n".join(json.dumps(dict(kind=k)) for k in kinds) + "\n").encode())
                return DummyProcess()
            with mock.patch.object(w, "canonical_state", return_value={"same": 1}), \
                    mock.patch.object(w, "model_state", return_value=({"same": 1}, b"prefix")), \
                    mock.patch.object(w, "processes", return_value=[]), mock.patch.object(w.subprocess, "Popen", side_effect=fake_start):
                result = w.collect_provider(selected, p["providers"][0], Path(d) / "A1", "x" * 64, [])
            self.assertIsNone(result["abort_reason"])
            self.assertIsNone(result["qualification_passed"])
            self.assertFalse(result["performance_acceptance"])
            self.assertEqual(result["stage"], "P")
            self.assertEqual(result["workload"], "business96")
            self.assertEqual(result["prerequisite_report_sha256"], "b" * 64)
            plan = json.loads((Path(d) / "A1/execution-plan.json").read_text())
            self.assertTrue(plan["adoption_cohort"])

    def run_outer(self, stage, fail_at=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = self.protocol(stage)
            p.update(window_root=str(root / "window"), lock_path=str(root / "fixture.lock"), wrapper_sha256=w.sha(ROOT / "fullmodel-long-window-wrapper.py"))
            protocol = root / "protocol.json"; protocol.write_text(json.dumps(p))
            calls = []
            def fake_collect(selected, provider, *args):
                calls.append((selected["workload_name"], provider["name"], selected["request_contract"]["repeats"]))
                failed = len(calls) == fail_at
                return dict(exit_code=1 if failed else 0, abort_reason="native nonzero exit" if failed else None)
            with mock.patch.object(sys, "argv", ["wrapper", "--protocol", str(protocol)]), \
                    mock.patch.object(w, "verify_descriptor"), mock.patch.object(w, "canonical_state", return_value={"same": 1}), \
                    mock.patch.object(w, "model_state", return_value=({"same": 1}, b"prefix")), \
                    mock.patch.object(w, "processes", return_value=[]), mock.patch.object(w.signal, "signal"), \
                    mock.patch.object(w.os, "access", return_value=True), \
                    mock.patch.object(w, "prerequisite", return_value=dict(path="/raw/q.json", bytes=1, sha256="b" * 64)), \
                    mock.patch.object(w, "collect_provider", side_effect=fake_collect), \
                    mock.patch.object(w.subprocess, "Popen", side_effect=AssertionError("native forbidden")), contextlib.redirect_stdout(io.StringIO()):
                code = w.main()
            return code, calls, json.loads((root / "window/window.json").read_text())

    def test_fixed_six_process_order_and_no_automatic_next_stage(self):
        for stage, names in (("Qlong", ["G_lib", "off", "on"]), ("P", ["A1", "B", "A2"])):
            code, calls, result = self.run_outer(stage)
            self.assertEqual(code, 0)
            self.assertEqual([(x[0], x[1]) for x in calls], [(work, name) for work in ["business96", "code38404"] for name in names])
            self.assertTrue(result["execution_complete"])
            self.assertTrue(result["no_automatic_next_stage"])
            self.assertIsNone(result["qualification_passed"])
            self.assertFalse(result["performance_acceptance"])

    def test_stop_on_failure_never_retries_or_enters_second_workload(self):
        code, calls, result = self.run_outer("Qlong", fail_at=2)
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result["execution_complete"])
        self.assertEqual(result["abort_reason"], "provider collection stopped: business96/off")


if __name__ == "__main__": unittest.main()
