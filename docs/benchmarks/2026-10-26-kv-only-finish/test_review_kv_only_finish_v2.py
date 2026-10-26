"""CPU-only independent-reader fixtures; no harness/builder import or execution."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("reader", HERE / "review_kv_only_finish_v2.py")
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
BUILD = Path("/tmp/myds-kv-only-finish-probe-final-v1-20261025")
PROTOCOL = Path("/tmp/myds-kv-only-finish-probe-20261025/component-correctness-protocol-v1.json")
HOST = Path("/tmp/myds-kv-only-finish-probe-host-only-v1-20261025")
HOST_STDOUT = Path("/tmp/myds-kv-only-finish-probe-20261025/host-only.stdout.jsonl")
GUARDS = [b"\xa5" * 64] * 2


def changed(data, word, value=0x3f800000):
    data = bytearray(data); struct.pack_into("<I", data, word * 4, value); return bytes(data)


def stdout_fixture():
    events = [dict(kind="case", case=i, layer=l, position=p, compressed=l >= 2,
                   args_sha256="0" * 64, args_bytes=136) for i, (l, p) in enumerate(r.CASES)]
    events.append(dict(kind="host_contract", case_count=114, eligible_layers=r.LAYERS,
                       native_groups=80, candidate_groups=80, args_bytes=136, no_gpu_device_created=False))
    events.extend([dict(kind="loaded_library")] * 2 + [dict(kind="pipeline")] * 3)
    for i in range(342):
        c, p = i // 3, i % 3
        events.append(dict(kind="gpu_completion", stage="phase1", case=c, provider=p,
                           ordinal=i + 1, completed=True, status=4, error=None))
        events.extend(dict(kind="parity", case=c, provider=p, role=role) for role in r.ROLES)
    return events


def args(raw=None, stdout=None, host=False):
    return argparse.Namespace(protocol=PROTOCOL, build_dir=BUILD,
        host_manifest=HOST / "artifact-manifest.json", host_stdout=HOST_STDOUT,
        path_map=[], check_host_preparation=host, raw_dir=raw, stdout=stdout,
        stderr=None, expected_device="CPU fixture synthetic device")


class ContractFixtures(unittest.TestCase):
    def test_all_cases_file_extents_and_wrap_positions(self):
        self.assertEqual((len(r.CASES), len(r.before_sizes()), len(r.after_sizes())), (114, 457, 3078))
        self.assertEqual(sum(x.endswith(".raw") and "guard" not in x for x in r.after_sizes()), 1026)
        self.assertEqual(sum("guard" in x for x in r.after_sizes()), 2052)
        self.assertEqual(len(r.LAYERS), 32)
        for pos in (127, 128, 16383, 16384, 65535, 65536, 327679):
            self.assertIn(pos, [p for _, p in r.CASES])

    def test_exact342_completion_order_and_every_three_roles(self):
        ev = stdout_fixture(); completions, parity = r.check_stream(ev)
        self.assertEqual((len(completions), len(parity)), (342, 1026))
        for name, edit in (
            ("missing last completion", lambda x: x.__delitem__(slice(-4, None))),
            ("duplicate ordinal", lambda x: x[124].update(ordinal=1)),
            ("bool ordinal", lambda x: x[120].update(ordinal=True)),
            ("wrong provider", lambda x: x[120].update(provider=1)),
            ("wrong case", lambda x: x[124].update(case=1)),
            ("unsignaled CB", lambda x: x[120].update(status=2)),
            ("status mismatch", lambda x: x[120].update(completed=False)),
            ("success with error", lambda x: x[120].update(error="GPU error")),
            ("wrong role order", lambda x: x[121].update(role="window")),
            ("timing event", lambda x: x.append(dict(kind="timing"))),
        ):
            with self.subTest(name=name), self.assertRaises(r.EvidenceError):
                bad = copy.deepcopy(ev); edit(bad); r.check_stream(bad)
        bad = copy.deepcopy(ev); bad[120].update(status=5, completed=False, error="fixture error")
        cs, _ = r.check_stream(bad)
        self.assertFalse(all(c["completed"] for c in cs))

    def test_host_preparation_cannot_be_gpu_evidence(self):
        ev = stdout_fixture()[:115]; ev[-1]["no_gpu_device_created"] = True
        self.assertEqual(r.check_stream(ev, host=True), ([], []))
        with self.assertRaises(r.EvidenceError): r.check_stream(ev)
        ev.append(dict(kind="gpu_completion"))
        with self.assertRaises(r.EvidenceError): r.check_stream(ev, host=True)

    def test_candidate_dummy_q_uses_before_not_native_gold_or_finite_oracle(self):
        before = struct.pack("<6I", 0x7fc00001, 0x7f800000, 0xff800000, 0x80000000, 1, 0x3f808000) + bytes(r.SIZES["dummy_q"] - 24)
        gold = changed(before, 0, 0x7fc00000)
        rec, ok = r.role_result((0, 0, 0), 2, "dummy_q", before, GUARDS, before, gold, bytes(2048))
        self.assertTrue(ok); self.assertFalse(rec["finite_bf16_required"])
        rec, ok = r.role_result((0, 0, 0), 2, "dummy_q", gold, GUARDS, before, gold, bytes(2048))
        self.assertFalse(ok); self.assertEqual((rec["mismatches"], rec["first_mismatch"]), (1, 0))
        self.assertFalse(r.role_result((0, 0, 0), 1, "dummy_q", before, GUARDS, before, gold, bytes(2048))[1])

    def test_all_noncurrent_window_cells_and_current_complete_kv(self):
        before = bytes(r.SIZES["window"]); kv = struct.pack("<I", 0x3f800000) * 512
        for pos in (0, 127, 128, 16383, 16384, 65535, 65536, 327679):
            start = (pos % 128) * 2048
            after = before[:start] + kv + before[start + 2048:]
            rec, ok = r.role_result((0, 0, pos), 2, "window", after, GUARDS, before, after, kv)
            self.assertTrue(ok)
            for slot in range(128):
                if slot == pos % 128: continue
                bad = changed(after, slot * 512)
                rec, ok = r.role_result((0, 0, pos), 2, "window", bad, GUARDS, before, bad, kv)
                self.assertFalse(ok); self.assertEqual(rec["untouched_window_mismatches"], 1)
            rec, ok = r.role_result((0, 0, pos), 2, "window", after, GUARDS, before, after, changed(kv, 511, 0))
            self.assertFalse(ok); self.assertFalse(rec["current_window_slot_equals_complete_kv"])

    def test_kv_and_window_reject_non_bf16_nan_inf_even_when_gold_same(self):
        for role in ("kv", "window"):
            for word in (1, 0x3f800001, 0x7f800000, 0xff800000, 0x7fc00000):
                with self.subTest(role=role, word=word):
                    before = bytes(r.SIZES[role]); body = changed(before, 0, word)
                    rec, ok = r.role_result((0, 0, 0), 0, role, body, GUARDS, before, body, body[:2048])
                    self.assertFalse(ok); self.assertFalse(rec["finite_bf16"])

    def test_both_guards_and_full_word_byte_mismatch(self):
        zero = bytes(2048)
        for side in (0, 1):
            bad = list(GUARDS); bad[side] = b"\xa5" * 63 + b"\xa4"
            self.assertFalse(r.role_result((0, 0, 0), 1, "kv", zero, bad, zero, zero, zero)[1])
        self.assertEqual(r.difference(zero, changed(zero, 511)), (1, 511))
        self.assertFalse(r.role_result((0, 0, 0), 1, "kv", changed(zero, 511), GUARDS, zero, zero, zero)[1])

    def test_all_manifest_descriptors_required_and_raw_sha_recomputed(self):
        rows = [dict(path=n, bytes=size or 1, sha256="0" * 64) for n, size in r.before_sizes().items()]
        self.assertEqual(len(r.indexed_files(dict(files=rows), r.before_sizes())), 457)
        for bad in (rows[:-1], rows + [rows[0]], rows + [dict(path="../escape", bytes=1, sha256="0" * 64)]):
            with self.assertRaises(r.EvidenceError): r.indexed_files(dict(files=bad), r.before_sizes())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.raw"; path.write_bytes(b"body")
            desc = dict(path="body.raw", bytes=4, sha256=r.digest(b"body"))
            self.assertEqual(r.raw(tmp, desc), b"body")
            path.write_bytes(b"evil")
            with self.assertRaises(r.EvidenceError): r.raw(tmp, desc)
            with self.assertRaises(r.EvidenceError): r.local_file(tmp, "../body.raw")

    def test_independent_source_extractor_and_no_builder_execution(self):
        code = b'kernel void test(int a) { const char *s="}"; /* } */ if(a) { // }\n a++; } }\n'
        self.assertEqual(r.kernel(code, "test"), code.rstrip())
        with self.assertRaises(r.EvidenceError): r.kernel(code[:-3], "test")
        data = b'PREAMBLE="p"\nSHADER_FILES=["a"]\nCOPY_KERNELS="c"\nraise Exception("must never execute")\n'
        self.assertEqual(r.literal_constants(data)["PREAMBLE"], "p")
        with self.assertRaises((ValueError, r.EvidenceError)): r.literal_constants(b'PREAMBLE=__import__("os").getcwd()\nSHADER_FILES=[]\nCOPY_KERNELS=""')

    def test_json_and_witness_types_are_not_coerced(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(r.EvidenceError): r.loads(text)
        self.assertFalse(r.same(dict(guards=1), dict(guards=True)))
        self.assertFalse(r.same([0], [False]))

    def test_loaded_path_allows_only_prebound_mac_tmp_alias_not_arbitrary_map(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="kv-reader-path-") as tmp:
            path = Path(tmp) / "payload"; path.write_bytes(b"fixture")
            a = args(); a.path_map = ["/private/tmp=/tmp"]
            review = r.Review(a)
            review.bound_path("/private" + str(path), path, r.digest(b"fixture"))
            a.path_map.append("/arbitrary=" + tmp)
            review = r.Review(a)
            with self.assertRaisesRegex(r.EvidenceError, "prebound path"):
                review.bound_path("/arbitrary/payload", path, r.digest(b"fixture"))
            with self.assertRaisesRegex(r.EvidenceError, "complete SHA"):
                review.bound_path(str(path), path, r.digest(b"other"))

    def test_frozen_host_preparation_full_sha_and_source_inverse(self):
        a = args(host=True); a.expected_device = None
        report = r.Review(a).run()
        self.assertTrue(report["preparation_valid"])
        self.assertTrue(report["complete_TU_source_verified"])
        self.assertTrue(report["original_KV_branch_bytes_verified"])
        self.assertEqual(report["preparation_input_files_verified"], 457)
        self.assertFalse(report["GPU_correctness_evaluated"])
        self.assertFalse(report["correctness_passed"])


class FullReaderFixture(unittest.TestCase):
    """CPU-created retained payload, explicitly not GPU evidence. Exercises the
    complete reader path, exact source/library binding and all3535 files."""
    def test_full_positive_and_updated_sha_false_parity_refused(self):
        host = r.read_json(HOST / "artifact-manifest.json")
        with tempfile.TemporaryDirectory(prefix="kv-only-cpu-fixture-") as tmp:
            root = Path(tmp); rawdir = root / "raw"; rawdir.mkdir(); stdout = root / "stdout.jsonl"
            manifest = copy.deepcopy(host); manifest.update(host_only=False, device_name="CPU fixture synthetic device", device_registry_id=1,
                timing_requested=False, gpu_command_buffers_submitted=342, all_complete_outputs_and_guards_pass=True,
                status="synthetic_correctness_complete_not_adoption", files=[], parity_records=[])
            events = [r.loads(line) for line in HOST_STDOUT.read_text().splitlines()]
            events[114]["no_gpu_device_created"] = False
            protocol = r.read_json(PROTOCOL)
            events.extend([dict(kind="loaded_library", provider_library=i, path=path, sha256=checksum, direct_url_load=True)
                for i, path, checksum in ((0, protocol["golden_metallib_path"], protocol["golden_metallib_sha256"]),
                    (1, str(BUILD / "probe.metallib"), protocol["probe_metallib_sha256"]))])
            events.extend(dict(kind="pipeline", symbol=symbol, groups=[80,1,1], threads=[32,1,1],
                static_tg_bytes=0, dynamic_tg_bytes=0, fallback=False) for symbol in (r.NATIVE, r.NATIVE, r.CANDIDATE))
            def save(name, data):
                (rawdir / name).write_bytes(data)
                manifest["files"].append(dict(path=name, bytes=len(data), sha256=r.digest(data)))
            for desc in host["files"]: save(desc["path"], (HOST / desc["path"]).read_bytes())
            for case, (layer, pos) in enumerate(r.CASES):
                before = {role: (HOST / f"case{case:03d}.{role}.before.raw").read_bytes() for role in r.ROLES}
                start = (pos % 128) * 2048
                gold = dict(before); gold["window"] = before["window"][:start] + before["kv"] + before["window"][start+2048:]
                # Gold Q intentionally differs from before in every case. Candidate
                # correctly retains Q before; no computed native-math oracle is claimed.
                gold["dummy_q"] = changed(before["dummy_q"], 0, 0x3f800000 if before["dummy_q"][:4] != struct.pack("<I",0x3f800000) else 0)
                for provider in range(3):
                    bodies = dict(gold)
                    if provider == 2: bodies["dummy_q"] = before["dummy_q"]
                    ordinal = case * 3 + provider + 1
                    events.append(dict(kind="gpu_completion", stage="phase1", case=case, provider=provider,
                        ordinal=ordinal, completed=True, status=4, error=None))
                    for role in r.ROLES:
                        stem = f"phase1.case{case:03d}.provider{provider}.{role}"
                        save(stem + ".raw", bodies[role]); save(stem + ".prefix-guard.raw", GUARDS[0]); save(stem + ".suffix-guard.raw", GUARDS[1])
                        rec, ok = r.role_result((case,layer,pos), provider, role, bodies[role], GUARDS, before[role], gold[role], bodies["kv"])
                        self.assertTrue(ok); manifest["parity_records"].append(rec); events.append(rec)
            def emit():
                (rawdir / "artifact-manifest.json").write_text(json.dumps(manifest))
                stdout.write_text("".join(json.dumps(e) + "\n" for e in events))
            emit(); report = r.Review(args(rawdir, stdout)).run()
            self.assertTrue(report["correctness_passed"])
            self.assertEqual((report["complete_command_buffers"], report["total_raw_files"]), (342, 3535))
            self.assertFalse(report["production_adoption"]); self.assertFalse(report["timing_evaluated"])
            # Updating raw+manifest SHA and advertised parity cannot conceal a
            # changed untouched window: independent semantic gate still refuses it.
            stem = "phase1.case000.provider2.window"; path = rawdir / (stem + ".raw")
            original_window = path.read_bytes()
            bad = changed(original_window, 512, 0x3f800000); path.write_bytes(bad)
            for desc in manifest["files"]:
                if desc["path"] == path.name: desc["sha256"] = r.digest(bad)
            emit()
            with self.assertRaisesRegex(r.EvidenceError, "independently recomputed full parity"):
                r.Review(args(rawdir, stdout)).run()
            # A fully disclosed numerical failure remains complete evidence,
            # distinct from malformed or lying evidence; it still fails gate.
            before0 = (HOST / "case000.window.before.raw").read_bytes()
            kv0 = (HOST / "case000.kv.before.raw").read_bytes()
            rec, passed = r.role_result((0,0,0), 2, "window", bad, GUARDS, before0, original_window, kv0)
            self.assertFalse(passed)
            manifest["parity_records"][8] = rec; events[131] = rec
            manifest.update(all_complete_outputs_and_guards_pass=False, status="failed")
            emit(); report = r.Review(args(rawdir, stdout)).run()
            self.assertTrue(report["review_completed"]); self.assertTrue(report["raw_integrity_passed"])
            self.assertFalse(report["numerical_passed"]); self.assertFalse(report["correctness_passed"])
            # Full GPU-error disclosure also remains complete failed evidence.
            path.write_bytes(original_window)
            for desc in manifest["files"]:
                if desc["path"] == path.name: desc["sha256"] = r.digest(original_window)
            rec, passed = r.role_result((0,0,0), 2, "window", original_window, GUARDS, before0, original_window, kv0)
            self.assertTrue(passed)
            manifest["parity_records"][8] = rec; events[131] = rec
            events[120].update(completed=False, status=5, error="CPU-fixture GPU error")
            emit(); report = r.Review(args(rawdir, stdout)).run()
            self.assertTrue(report["review_completed"]); self.assertTrue(report["numerical_passed"])
            self.assertFalse(report["GPU_completion_passed"]); self.assertFalse(report["correctness_passed"])


class ProducerSchemaRepairFixtures(unittest.TestCase):
    ROOT = Path("/tmp/myds-kv-only-finish-M4-screen-20261025")

    def actual_args(self):
        a = args(self.ROOT / "outputs", self.ROOT / "stdout.jsonl")
        a.stderr = self.ROOT / "stderr.txt"; a.expected_device = "Apple M4"
        return a

    def test_actual_stdout_producer_int0_1_and_complete_original_raw_pass(self):
        events = [r.loads(line) for line in (self.ROOT / "stdout.jsonl").read_text().splitlines()]
        records = [event for event in events if event["kind"] == "parity"]
        self.assertEqual(len(records), 1026)
        for record in records:
            self.assertIs(type(record["finite_bf16_required"]), int)
            self.assertEqual(record["finite_bf16_required"], int(record["role"] != "dummy_q"))
        producer = (BUILD / "source/tools/bench_kv_only_finish_probe.m").read_text()
        self.assertIn('@"finite_bf16_required":@(r!=0)', producer)
        report = r.Review(self.actual_args()).run()
        self.assertTrue(report["correctness_passed"])
        self.assertTrue(report["numerical_passed"])
        self.assertTrue(report["GPU_completion_passed"])
        self.assertFalse(report["repair_scope"]["numeric_guards_completion_provenance_gates_changed"])
        self.assertEqual(report["old_reader_sha256"], "db555ce6e7450e11c119785a9738f8e33338ca72918d7aeba8e57221a9e5c816")

    def test_bool_substitution_in_both_manifest_and_stdout_is_refused(self):
        # In-memory corruption only: immutable actual raw/stdout remain untouched.
        # Altering both declarations must not bypass independent recomputation.
        original = r.loads
        def parse_as_spoofed_bool(text):
            value = original(text)
            def spoof(node):
                if isinstance(node, dict):
                    for key, child in list(node.items()):
                        if key == "finite_bf16_required": node[key] = bool(child)
                        else: spoof(child)
                elif isinstance(node, list):
                    for child in node: spoof(child)
            spoof(value); return value
        with patch.object(r, "loads", parse_as_spoofed_bool):
            with self.assertRaisesRegex(r.EvidenceError, "independently recomputed full parity declaration"):
                r.Review(self.actual_args()).run()

    def test_declared_field_requires_exact_integer_not_boolean_or_float(self):
        for role in r.ROLES:
            zero = bytes(r.SIZES[role]); kv = bytes(2048)
            rec, _ = r.role_result((0,0,0), 0, role, zero, GUARDS, zero, zero, kv)
            self.assertIs(type(rec["finite_bf16_required"]), int)
            for value in (bool(rec["finite_bf16_required"]), float(rec["finite_bf16_required"]), 2, -1):
                spoof = dict(rec, finite_bf16_required=value)
                self.assertFalse(r.same(spoof, rec))

    def test_manual_actual_json_like_record_uses_int_declaration(self):
        # Literal record transcribed from retained actual stdout, independently
        # of role_result fixture generation. BOOL fields deliberately stay bool.
        expected = r.loads('''{"bytes":2048,"case":0,"compressed":false,
            "current_window_slot_equals_complete_kv":true,
            "expectation":"actual-golden-output-complete-bytes",
            "finite_bf16":true,"finite_bf16_required":1,"first_mismatch":null,
            "guards":true,"kind":"parity","layer":0,"mismatches":0,
            "position":0,"provider":0,"role":"kv",
            "sha256":"94ba56e42706fa8cd8a5dfabd745273c8961d8c51a29f25b3c97be9d4571839a",
            "stage":"phase1","untouched_window_mismatches":0}''')
        before = (self.ROOT / "outputs/case000.kv.before.raw").read_bytes()
        actual = (self.ROOT / "outputs/phase1.case000.provider0.kv.raw").read_bytes()
        computed, passed = r.role_result((0,0,0), 0, "kv", actual, GUARDS, before, actual, actual)
        self.assertTrue(passed); self.assertTrue(r.same(computed, expected))
        self.assertFalse(r.same(computed, dict(expected, finite_bf16_required=True)))


if __name__ == "__main__": unittest.main(verbosity=2)
