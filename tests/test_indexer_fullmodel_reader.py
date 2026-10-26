#!/usr/bin/env python3
"""Small CPU fixtures; never create a model, Metal device or subprocess."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import review_indexer_fullmodel as reader


def fixture(root, run=0):
    root.mkdir()
    scalars = [8, 42, 8192, 2, 42, 8192, 2, 2, 0, 1, 8, 8, 4096, 2048, 0, 3, 2048]
    lengths, positions = [2, 1, 0], [4, 0] + [0] * 86
    output, tokens, logits = b"\x00\xe4\xb8\xad", [17, 18], struct.pack("<fff", 0, 1, -2)
    # The schema is independently spelled out in the fixture, including one
    # published snapshot, an empty anchor slot and the ratio1 fourth owner.
    payloads = {"context-scalars.u64le": struct.pack("<17Q", *scalars),
                "engram-tail.u32le": struct.pack("<3I", 1, 2, 3),
                "snapshot-positions.u32le": struct.pack("<88I", *positions),
                "prefix-tokens.i32le": struct.pack("<2i", 1, 2),
                "anchor-lengths.u32le": struct.pack("<3I", *lengths),
                "tokens.i32le": struct.pack("<2i", *tokens), "output.bin": output,
                "anchor-00-tokens.i32le": struct.pack("<2i", 1, 2),
                "anchor-01-tokens.i32le": struct.pack("<i", 1), "anchor-02-tokens.i32le": b""}
    for layer in range(40):
        payloads[f"layer-{layer:02d}-window-active.f32"] = bytes(8 * 512 * 4)
        payloads[f"snapshot-00-layer-{layer:02d}-window-active.f32"] = bytes(4 * 512 * 4)
    for owner in range(4):
        rows = 8 if owner == 3 else 4
        payloads[f"owner-{owner}-compressed-active.bf16"] = bytes(rows * 1024)
        payloads[f"owner-{owner}-index-active.bf16"] = bytes(rows * 256)
        if owner < 3:
            for kind in ("kv", "score"):
                payloads[f"owner-{owner}-previous-{kind}.f32"] = bytes(512 * 4)
                payloads[f"snapshot-00-owner-{owner}-previous-{kind}.f32"] = bytes(512 * 4)
    for name, data in payloads.items():
        (root / name).write_bytes(data)
    metadata = dict(byte_order="little", fresh_context=True, ctx_pos=8, next_position=8,
        window_rows=8, layers=40, window_width=512, window_order="chronological_active_rows",
        compressed_row_bytes=1024, index_row_bytes=256, rng_state="000000000000002a",
        engram_position=8, engram_tail=[1, 2, 3], anchor_layout="per_slot_active_tokens",
        anchor_slots=3, anchor_cap=2048, anchor_n=2, scalar_order=list(reader.SCALARS))
    manifest = dict(schema_version=2, subject="prefill_indexer_k_resident", run=run,
        scope="fresh_context_full_prefill_final_state", metadata=metadata, limitations=reader.LIMITATIONS,
        files=[dict(path=name, bytes=len(data), fnv1a64_bytes=reader.fnv1a(data)) for name, data in payloads.items()],
        file_count=len(payloads), complete=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    logits_path = root.parent / (root.name + "-logits.f32")
    logits_path.write_bytes(logits)
    sample = dict(run=run, output_bytes=len(output), output_tokens=2, token_ids=tokens,
        token_ids_fnv1a64=reader.fnv1a(payloads["tokens.i32le"]), output_bytes_fnv1a64=reader.fnv1a(output),
        final_logits=dict(elements=3, finite=True, fnv1a64_bytes=reader.fnv1a(logits)))
    # The small fake vocabulary must nevertheless include the generated IDs.
    sample["token_ids"] = [1, 2]
    payloads["tokens.i32le"] = struct.pack("<2i", 1, 2)
    (root / "tokens.i32le").write_bytes(payloads["tokens.i32le"])
    sample["token_ids_fnv1a64"] = reader.fnv1a(payloads["tokens.i32le"])
    next(d for d in manifest["files"] if d["path"] == "tokens.i32le")["fnv1a64_bytes"] = sample["token_ids_fnv1a64"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest, sample, logits_path


def witness(mode=1):
    row = dict(api_available=True, mode=mode, adoption_valid=True, run=0,
        symbol_f32=reader.SYMBOLS[0] if mode else None, symbol_bf16=reader.SYMBOLS[1] if mode else None,
        first_callback_scope="prefill_plus_first_output_callback", counters_count_encoding_not_gpu_completion=True)
    first = dict(requests=20, encoded=10 if mode else 0, fallbacks=0, failures=0, f32_dispatches=0, bf16_dispatches=10 if mode else 0)
    for name, value in first.items():
        row["through_first_callback_" + name] = value
        row[name] = value + (4 if name == "requests" else 0)
        row["decode_" + name] = 4 if name == "requests" else 0
    return row


class FullmodelReaderTests(unittest.TestCase):
    def test_complete_schema2_multi_anchor_all_published_payloads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "state"
            manifest, sample, logits = fixture(root)
            result = reader.PayloadReader().read_export(root, logits, sample, 3)
            self.assertTrue(result["complete"])
            self.assertTrue(result["finite_logits"])
            self.assertEqual(len(result["files"]), manifest["file_count"] + 1)
            self.assertEqual(result["evidence"]["anchor-02-tokens.i32le"]["bytes"], 0)
            self.assertEqual(result["evidence"]["owner-3-compressed-active.bf16"]["bytes"], 8192)

    def test_missing_owner_even_with_self_consistent_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "state"; m, sample, logits = fixture(root)
            omitted = "owner-3-index-active.bf16"
            (root / omitted).unlink(); m["files"] = [x for x in m["files"] if x["path"] != omitted]; m["file_count"] -= 1
            (root / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(reader.EvidenceError): reader.PayloadReader().read_export(root, logits, sample, 3)

    def test_duplicate_extra_or_semantically_unknown_payload_metadata_rejected(self):
        for mutation in ("duplicate", "extra_file", "unknown_metadata", "missing_snapshot"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "state"; m, sample, logits = fixture(root)
                if mutation == "duplicate": m["files"].append(deepcopy(m["files"][0])); m["file_count"] += 1
                elif mutation == "extra_file": (root / "unmanifested.bin").write_bytes(b"")
                elif mutation == "unknown_metadata": m["metadata"]["new_semantic_field"] = 1
                else:
                    name = "snapshot-00-layer-39-window-active.f32"; (root / name).unlink()
                    m["files"] = [x for x in m["files"] if x["path"] != name]; m["file_count"] -= 1
                (root / "manifest.json").write_text(json.dumps(m))
                with self.assertRaises(reader.EvidenceError): reader.PayloadReader().read_export(root, logits, sample, 3)

    def test_raw_size_fnv_tokens_and_scalar_metadata_are_independent_checks(self):
        for mutation in ("short", "fnv", "tokens", "rng"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "state"; m, sample, logits = fixture(root)
                if mutation == "short": (root / "owner-0-compressed-active.bf16").write_bytes(b"x")
                elif mutation == "fnv": (root / "layer-00-window-active.f32").write_bytes(b"x" + bytes(8 * 512 * 4 - 1))
                elif mutation == "tokens": sample["token_ids"] = [2, 1]
                else: m["metadata"]["rng_state"] = "000000000000002b"
                (root / "manifest.json").write_text(json.dumps(m))
                with self.assertRaises(reader.EvidenceError): reader.PayloadReader().read_export(root, logits, sample, 3)

    def test_exact_bytes_distinguish_signed_zero_and_return_structured_failure(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            a.write_bytes(struct.pack("<f", 0.0)); b.write_bytes(struct.pack("<f", -0.0))
            left = dict(files={"final-logits.f32": a}, metadata=dict(rng_state="same"), finite_logits=True)
            right = dict(files={"final-logits.f32": b}, metadata=dict(rng_state="same"), finite_logits=True)
            diff = reader.compare_exports(left, right)
            self.assertFalse(diff["passed"])
            self.assertEqual(diff["files"]["final-logits.f32"]["first_mismatch_byte"], 3)
            self.assertEqual(diff["files"]["final-logits.f32"]["mismatched_bytes"], 1)
            right["metadata"]["rng_state"] = "different"
            self.assertFalse(reader.compare_exports(left, right)["metadata_equal"])

    def test_identical_nan_or_infinity_cannot_pass_cross_arm_comparison(self):
        for nonfinite in (float("nan"), float("inf")):
            with self.subTest(nonfinite=nonfinite), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "state"; _, sample, logits = fixture(root)
                data = struct.pack("<fff", nonfinite, 1, -2); logits.write_bytes(data)
                sample["final_logits"].update(finite=False, fnv1a64_bytes=reader.fnv1a(data))
                result = reader.PayloadReader().read_export(root, logits, sample, 3)
                self.assertFalse(result["finite_logits"])
                self.assertFalse(reader.compare_exports(result, result)["passed"])

    def test_actual_adoption_requests_can_exceed_encoded_but_decode_cannot(self):
        reader.provider_witness(witness(1), 1); reader.provider_witness(witness(0), 0)
        for mutation in ("decode", "late_only", "rewind", "symbol", "fallback"):
            with self.subTest(mutation=mutation):
                row = witness(1)
                if mutation == "decode":
                    for field in ("encoded", "bf16_dispatches"): row[field] += 1; row["decode_" + field] += 1
                elif mutation == "late_only":
                    for field in ("encoded", "bf16_dispatches"): row["through_first_callback_" + field] = 0; row["decode_" + field] = 10
                elif mutation == "rewind": row["requests"] = 1
                elif mutation == "symbol": row["symbol_bf16"] = "wrong_symbol"
                else: row["fallbacks"] = row["decode_fallbacks"] = 1
                with self.assertRaises(reader.EvidenceError): reader.provider_witness(row, 1)

    def test_off_cannot_hide_real_candidate_or_fake_an_unavailable_api(self):
        for mutation in ("encoded", "api", "counter_bool"):
            row = witness(0)
            if mutation == "encoded": row["encoded"] = row["decode_encoded"] = row["bf16_dispatches"] = row["decode_bf16_dispatches"] = 1
            elif mutation == "api": row["api_available"] = False
            else: row["requests"] = True
            with self.assertRaises(reader.EvidenceError): reader.provider_witness(row, 0)

    def test_gold_library_and_candidate_failures_are_distinct_without_performance_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, b = root / "a", root / "b"; a.write_bytes(b"same"); b.write_bytes(b"diff")
            same = dict(files={"final-logits.f32": a}, metadata=dict(rng_state="same"), finite_logits=True)
            other = dict(files={"final-logits.f32": b}, metadata=dict(rng_state="same"), finite_logits=True)
            for gold, off, on, bridge, candidate in ((same, other, other, False, True), (same, same, other, True, False)):
                exports = {name: {0: value, 1: value} for name, value in (("G_lib", gold), ("off", off), ("on", on))}
                result = reader.qualification_comparisons(exports)
                self.assertEqual(result["gold_library_bridge_passed"], bridge)
                self.assertEqual(result["candidate_reference_passed"], candidate)
                self.assertFalse(result["qualification_passed"])
                self.assertFalse(result["production_acceptance"])
                self.assertFalse(result["performance_acceptance"])
                self.assertFalse(result["deployed_executable_wire_bridge_completed"])

    def test_complete_but_different_state_layout_is_a_retained_numerical_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "payload"; path.write_bytes(b"x")
            left = dict(files={"final-logits.f32": path, "anchor-00-tokens.i32le": path}, metadata={}, finite_logits=True)
            right = dict(files={"final-logits.f32": path}, metadata={}, finite_logits=True)
            result = reader.compare_exports(left, right)
            self.assertFalse(result["passed"])
            self.assertFalse(result["complete_file_set_equal"])
            self.assertEqual(result["missing_state_files"], ["anchor-00-tokens.i32le"])
            self.assertTrue(result["files"]["final-logits.f32"]["equal"])

    def test_all_config_entries_plus_only_mode_and_two_sinks_are_required(self):
        raw = b"# comment\nMYDS_PREFIX_CACHE=1\nMYDS_PLD=1\nMYDS_METAL_CONCURRENT=1 # ordinary\n"
        settings, count = reader.config_settings(raw)
        expected = dict(settings, MYDS_INDEXER_K_RESIDENT="1", MYDS_BENCH_LOGITS_DIR="/raw/logits", MYDS_BENCH_STATE_DIR="/raw/state")
        indexer = dict(kind="indexer_metadata", schema_version=1, subject="prefill_indexer_k_resident", execute_requested=True,
            provider_adapter_available=True, fresh_context_per_sample=True, cache_configuration_preserved=True,
            config_load_override_env=True, config_file="/frozen.conf", config_loaded_entries=count, config_file_bytes=len(raw),
            config_file_fnv1a64=reader.fnv1a(raw), config_raw=raw.decode(), effective_configuration_count=len(expected), effective_configuration=expected)
        metadata = dict(environment=dict(expected, MYDS_UNSET_DEFAULT=None))
        result = reader.effective_configuration(indexer, metadata, raw, 1, "/raw/logits", "/raw/state", "/frozen.conf", 3)
        self.assertEqual(result["effective_entries"], 6)
        bad = deepcopy(indexer); bad["effective_configuration"]["MYDS_PLD"] = "0"
        with self.assertRaises(reader.EvidenceError): reader.effective_configuration(bad, metadata, raw, 1, "/raw/logits", "/raw/state", "/frozen.conf", 3)
        for key in ("MYDS_UNDECLARED_SELECTOR", "MYDS_KPROF"):
            bad = deepcopy(indexer); bad["effective_configuration"][key] = "1"; bad["effective_configuration_count"] += 1
            with self.assertRaises(reader.EvidenceError): reader.effective_configuration(bad, metadata, raw, 1, "/raw/logits", "/raw/state", "/frozen.conf", 3)

    def test_full_prefill_and_n_minus_one_callback_contract(self):
        row = dict(kind="sample", run=0, warmup=True, fresh_context=True, prompt_tokens=9, cached_tokens=0, prefill_tokens=8,
            output_tokens=2, decode_intervals=1, token_ids=[1, 2], prefill_s=1, wall_s=2, ttft_s=1.1, first_callback_s=1.1,
            last_callback_s=1.2, decode_s=.1, decode_ms_per_token=100, callback_intervals_ms=[100], generation_ok=True,
            stop_reason="length", stop_reason_code=1)
        reader.sample_contract(row, 0, 9, 2)
        for key, value in (("cached_tokens", 1), ("prefill_tokens", 9), ("decode_intervals", 2), ("callback_intervals_ms", [99]), ("stop_reason", "eos")):
            with self.subTest(key=key), self.assertRaises(reader.EvidenceError): reader.sample_contract(dict(row, **{key: value}), 0, 9, 2)

    def test_complete_two_sample_provider_raw_contract_and_no_stdout_filtering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "state").mkdir(); (root / "logits").mkdir()
            metadata = dict(kind="metadata", subject="prefill_indexer_k_resident", protocol_version=3, repeats=1, warmup_runs=1,
                max_new_tokens=2, max_seq_len=8192, temperature=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1, seed=42,
                qos="user_interactive", qos_result=0, prompt_renderer="parse_chat_prompt", clock="CLOCK_MONOTONIC",
                transport="direct_byte_callback", decode_basis="first_to_last_callback/(tokens-1)", dspark_model=None)
            loaded = dict(kind="load", native_v41=True, layers=40, dspark_loaded=False, fresh_context_per_sample=True,
                cache_configuration_preserved=True, vocab_size=3)
            records = [dict(kind="indexer_metadata"), metadata, loaded]
            for run in (0, 1):
                _, sample, logits = fixture(root / "state" / f"run-{run}", run)
                logits.rename(root / "logits" / f"run-{run}.f32")
                sample.update(kind="sample", warmup=run == 0, fresh_context=True, prompt_tokens=9, cached_tokens=0,
                    prefill_tokens=8, decode_intervals=1, prefill_s=1, wall_s=2, ttft_s=1.1, first_callback_s=1.1,
                    last_callback_s=1.2, decode_s=.1, decode_ms_per_token=100, callback_intervals_ms=[100],
                    generation_ok=True, stop_reason="length", stop_reason_code=1, sample_valid=True, output_equal_to_warmup=True,
                    gpu_indexer_provider=dict(witness(1), run=run))
                records.append(sample)
            stdout = root / "stdout.jsonl"; stdout.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            result = reader.read_run_exports(root, stdout, 1, 9, 2)
            self.assertTrue(result["within_provider_comparison"]["passed"])
            self.assertTrue(result["provider_adoption_passed"])
            stdout.write_text("unknown native line\n" + stdout.read_text())
            with self.assertRaises(reader.EvidenceError): reader.read_run_exports(root, stdout, 1, 9, 2)

    def test_native_source_proof_cannot_be_empty_or_truncated_prefix(self):
        data = b"\n".join((signature + 'void) { const char *x="}"; /* { */ if (x) { return 1; } return 0; }').encode()
                          for signature in sorted(reader.REFERENCE_SIGNATURES))
        proofs = {}
        for signature in reader.REFERENCE_SIGNATURES:
            function = reader.complete_c_function(data, signature)
            proofs[signature] = dict(byte_exact=True, bytes=len(function), sha256=reader.digest(function))
        reader.verify_reference_proofs(data, data, proofs)
        with self.assertRaises(reader.EvidenceError): reader.verify_reference_proofs(data, data, {})
        truncated = deepcopy(proofs); key = next(iter(truncated)); function = reader.complete_c_function(data, key)
        truncated[key].update(bytes=8, sha256=reader.digest(function[:8]))
        with self.assertRaises(reader.EvidenceError): reader.verify_reference_proofs(data, data, truncated)
        inventory = {"src/a.c": dict(bytes=1, sha256="a"), "include/a.h": dict(bytes=2, sha256="b"), "tools/driver.c": dict(bytes=3, sha256="c")}
        reader.verify_core_proofs({key: inventory[key] for key in ("src/a.c", "include/a.h")}, inventory)
        with self.assertRaises(reader.EvidenceError): reader.verify_core_proofs({}, inventory)

    def test_canonical_and_model_window_changes_are_never_qualified(self):
        with tempfile.TemporaryDirectory() as d:
            protocol = Path(d) / "protocol.json"
            model = dict(path="/frozen/model", bytes=2000000, inode=8, mtime_ns=9, prefix_bytes=1048576, prefix_sha256="prefix")
            canonical = dict(path="/frozen/myds", bytes=2, sha256="binary")
            p = dict(wrapper_sha256="wrapper", model=model, canonical_files={canonical["path"]: canonical})
            protocol.write_text(json.dumps(p))
            review = reader.Review(SimpleNamespace(path_map=[], protocol=protocol)); review.protocol = p
            actual_model = dict(model, canonical_path=model["path"], device=1, whole_model_payload_sha256_verified=False)
            actual_canonical = dict(canonical, canonical_path=canonical["path"], device=1, inode=1, mtime_ns=1)
            window = dict(stage="Q0", protocol_sha256=reader.file_sha(protocol), wrapper_sha256="wrapper", abort_reason=None,
                canonical_before={canonical["path"]: actual_canonical}, canonical_after={canonical["path"]: actual_canonical},
                canonical_unchanged=True, model_before=actual_model, model_after=actual_model, model_unchanged=True,
                recognized_before=[], recognized_after=[])
            review.check_window_common(window)
            for mutate in ("canonical", "model", "competitor", "protocol"):
                bad = deepcopy(window)
                if mutate == "canonical": bad["canonical_after"][canonical["path"]]["sha256"] = "changed"
                elif mutate == "model": bad["model_before"]["prefix_sha256"] = bad["model_after"]["prefix_sha256"] = "wrong"
                elif mutate == "competitor": bad["recognized_before"] = [dict(pid=123, comm="myds")]
                else: bad["protocol_sha256"] = "different"
                with self.subTest(mutate=mutate), self.assertRaises(reader.EvidenceError): review.check_window_common(bad)

    def test_full_raw_inventory_cannot_omit_or_substitute_payloads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "state").mkdir(); (root / "logits").mkdir()
            for name, raw in (("stdout.jsonl", b"{}\n"), ("stderr.txt", b""), ("logits/run-0.f32", b"all raw")):
                (root / name).write_bytes(raw)
            files = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=reader.file_sha(p)) for p in root.rglob("*") if p.is_file()}
            inventory = dict(schema_version=1, whole_payload_files_rehashed=True, tree_sha256=reader.inventory_digest(files),
                files=files, file_count=len(files), logical_bytes=sum(x["bytes"] for x in files.values()))
            path = root / "raw-inventory.json"; path.write_text(json.dumps(inventory))
            window = dict(raw_inventory_sha256=reader.file_sha(path), raw_tree_sha256=inventory["tree_sha256"])
            review = reader.Review(SimpleNamespace(path_map=[]))
            self.assertEqual(review.check_inventory(root, window)["files"], 3)
            (root / "logits/run-0.f32").write_bytes(b"new raw")
            with self.assertRaises(reader.EvidenceError): review.check_inventory(root, window)
            (root / "logits/run-0.f32").write_bytes(b"all raw"); (root / "state/unlisted.bin").write_bytes(b"x")
            with self.assertRaises(reader.EvidenceError): review.check_inventory(root, window)

    def test_evidence_failure_keeps_details_but_resets_every_eligibility(self):
        prior = dict(raw_inventory_files=123, qualification_passed=True, formal_performance_cohort_eligible=True,
            qualified_to_prepare_separate_long_protocol=True, production_acceptance=True)
        result = reader.evidence_failure(prior, reader.EvidenceError("wrong frozen reader SHA"))
        self.assertEqual(result["raw_inventory_files"], 123)
        self.assertEqual(result["evidence_error"], "wrong frozen reader SHA")
        for key in ("review_completed", "qualification_passed", "formal_performance_cohort_eligible", "qualified_to_prepare_separate_long_protocol",
                    "long_workload_qualification_completed", "production_acceptance", "production_adoption", "performance_acceptance", "deployed_executable_wire_bridge_completed"):
            self.assertIs(result[key], False)


if __name__ == "__main__":
    unittest.main()
