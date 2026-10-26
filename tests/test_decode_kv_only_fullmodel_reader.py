#!/usr/bin/env python3
"""CPU-only new-subject counter/no-spec/state/metric contract fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import struct
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import review_decode_kv_only_fullmodel as reader
import test_indexer_fullmodel_reader as old_fixtures


def telemetry(count):
    row={key:0 for key in reader.SPEC_COUNTERS}
    row.update(target_forward_calls=count,target_rows=count,output_tokens=count,cycle_count=count,
        timing_enabled=False,recorded_cycles=0,draft_ms=None,verify_ms=None,rollback_ms=None,snapshot_ms=None,pld_ms=None,
        cycle_ms=[],cycle_output_tokens=[])
    return row


def provider(mode,count):
    row=dict(api_available=True,mode=mode,run=0,adoption_valid=True,selected_symbol=reader.SYMBOL if mode else None,
        observed_callbacks=count,expected_calls_per_scalar_forward=32,no_spec_source_predicate_and_telemetry_verified=True,
        first_callback_scope="chunk_prefill_plus_last_prompt_scalar_forward_through_first_output_callback",
        decode_scope="N_minus_1_output_intervals_ordinary_scalar_decode",counters_count_encoding_not_gpu_completion=True)
    # Independent spelling of the producer's six integer totals/first/deltas.
    totals=dict(requests=32*count,eligible=32*count if mode else 0,encoded=32*count if mode else 0,
                references=0 if mode else 32*count,fallbacks=0,failures=0)
    first=dict(requests=32,eligible=32 if mode else 0,encoded=32 if mode else 0,references=0 if mode else 32,fallbacks=0,failures=0)
    for name in totals:
        row[name]=totals[name];row["through_first_callback_"+name]=first[name];row["decode_"+name]=totals[name]-first[name]
    return row


def raw_fixture(root,repeats=3,count=256,mode=1):
    (root/"state").mkdir();(root/"logits").mkdir()
    metadata=dict(kind="metadata",subject="decode_kv_only_finish",protocol_version=3,repeats=repeats,warmup_runs=1,max_new_tokens=count,
        max_seq_len=8192,temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1,seed=42,qos="user_interactive",qos_result=0,
        prompt_renderer="parse_chat_prompt",clock="CLOCK_MONOTONIC",transport="direct_byte_callback",decode_basis="first_to_last_callback/(tokens-1)",dspark_model=None)
    loaded=dict(kind="load",native_v41=True,layers=40,dspark_loaded=False,fresh_context_per_sample=True,cache_configuration_preserved=True,vocab_size=3)
    records=[dict(kind="decode_kv_metadata"),metadata,loaded]
    for run in range(repeats+1):
        target=root/"state"/f"run-{run}";m,sample,logits=old_fixtures.fixture(target,run)
        m.update(subject="decode_kv_only_finish",scope="fresh_context_final_state_after_prefill_and_decode")
        scalars=list(struct.unpack("<17Q",(target/"context-scalars.u64le").read_bytes()));scalars[3]=count
        (target/"context-scalars.u64le").write_bytes(struct.pack("<17Q",*scalars))
        tokens=[1,2]*(count//2);(target/"tokens.i32le").write_bytes(struct.pack("<"+"i"*count,*tokens))
        for item in m["files"]:
            data=(target/item["path"]).read_bytes();item.update(bytes=len(data),fnv1a64_bytes=reader.fnv1a(data))
        (target/"manifest.json").write_text(json.dumps(m));logits.rename(root/"logits"/f"run-{run}.f32")
        sample.update(kind="sample",subject="decode_kv_only_finish",warmup=run==0,fresh_context=True,prompt_tokens=9,cached_tokens=0,prefill_tokens=8,
            output_tokens=count,token_ids=tokens,token_ids_fnv1a64=reader.fnv1a((target/"tokens.i32le").read_bytes()),decode_intervals=count-1,
            prefill_s=1.000000001,wall_s=count+2,ttft_s=1.1,first_callback_s=1.1,last_callback_s=1.1+(count-1)/10,
            decode_s=(count-1)/10,decode_ms_per_token=100,callback_intervals_ms=[100]*(count-1),generation_ok=True,stop_reason="length",stop_reason_code=1,
            sample_valid=True,output_equal_to_warmup=True,gpu_decode_kv_only_provider=dict(provider(mode,count),run=run),decode_telemetry=telemetry(count))
        records.append(sample)
    stdout=root/"stdout.jsonl";stdout.write_text("\n".join(json.dumps(r)for r in records)+"\n")
    return stdout,records


class DecodeReaderTests(unittest.TestCase):
    def test_exact_new_decode_N64_and_N256_both_modes(self):
        for count in (64,256):
            for mode in (0,1):
                with self.subTest(count=count,mode=mode):
                    result=reader.provider_witness(provider(mode,count),mode,count,telemetry(count))
                    self.assertEqual(result["expected_first_calls"],32)
                    self.assertEqual(result["expected_total_calls"],count*32)
                    self.assertEqual(result["expected_decode_calls"],(count-1)*32)
        result=reader.provider_witness(provider(1,256),1,256,telemetry(256))
        self.assertEqual((result["counters"]["through_first_callback_encoded"],result["counters"]["decode_encoded"],result["counters"]["encoded"]),(32,8160,8192))

    def test_old_prefill_only_adoption_does_not_count_as_decode_adoption(self):
        row=provider(1,256)
        for name in ("eligible","encoded"):
            row[name]=32;row["decode_"+name]=0
        with self.assertRaises(reader.EvidenceError):reader.provider_witness(row,1,256,telemetry(256))
        row=provider(1,256)
        for name in ("eligible","encoded"):
            row["through_first_callback_"+name]=0;row["decode_"+name]=8192
        with self.assertRaises(reader.EvidenceError):reader.provider_witness(row,1,256,telemetry(256))

    def test_candidate_reference_fallback_failure_symbol_and_regression_fail_closed(self):
        for key,value in (("references",1),("fallbacks",1),("failures",1),("selected_symbol","wrong"),
                          ("decode_encoded",None),("through_first_callback_requests",31),("observed_callbacks",64),
                          ("no_spec_source_predicate_and_telemetry_verified",False),("encoded",True)):
            with self.subTest(key=key),self.assertRaises(reader.EvidenceError):
                reader.provider_witness(dict(provider(1,256),**{key:value}),1,256,telemetry(256))

    def test_mode0_must_reference_every_call_and_never_encode(self):
        for key,value in (("references",8191),("encoded",1),("eligible",1),("selected_symbol",reader.SYMBOL)):
            with self.subTest(key=key),self.assertRaises(reader.EvidenceError):
                reader.provider_witness(dict(provider(0,256),**{key:value}),0,256,telemetry(256))

    def test_no_spec_real_telemetry_cannot_be_replaced_by_true_declaration(self):
        for key,value in (("target_forward_calls",257),("target_rows",512),("output_tokens",64),("cycle_count",255),
                          ("batch2_calls",1),("draft_proposed",1),("pld_queries",1),("depth2_hits1",1),
                          ("timing_enabled",True),("cycle_ms",[1]),("recorded_cycles",False)):
            t=telemetry(256);t[key]=value
            with self.subTest(key=key),self.assertRaises(reader.EvidenceError):reader.provider_witness(provider(1,256),1,256,t)
        t=telemetry(256);t["unreviewed_counter"]=0
        with self.assertRaises(reader.EvidenceError):reader.no_spec_telemetry(t,256)

    def test_dynamicN_rejects_zero_one_or_actualN_mismatch(self):
        for n in (0,1):
            with self.assertRaises(reader.EvidenceError):reader.provider_witness(provider(1,n),1,n,telemetry(n))
        with self.assertRaises(reader.EvidenceError):reader.provider_witness(provider(1,64),1,256,telemetry(256))

    def test_schema2_subject_scope_adaptation_only_and_old_subject_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"state";manifest,sample,logits=old_fixtures.fixture(root)
            with self.assertRaises(reader.EvidenceError):reader.DecodePayloadReader().read_export(root,logits,sample,3)
            manifest["subject"]=reader.SUBJECT;manifest["scope"]=reader.STATE_SCOPE
            (root/"manifest.json").write_text(json.dumps(manifest))
            result=reader.DecodePayloadReader().read_export(root,logits,sample,3)
            self.assertTrue(result["complete"]);self.assertTrue(result["finite_logits"])
            self.assertEqual(len(result["files"]),manifest["file_count"]+1)
            with self.assertRaises(reader.EvidenceError):reader.core.PayloadReader().read_export(root,logits,sample,3)

    def test_decode_primary_strict2_and_inclusive1_half_boundaries(self):
        self.assertFalse(reader.metric_gate("100","98","100",True)["passed"])
        self.assertTrue(reader.metric_gate("100","97.999999999","101",True)["passed"])
        self.assertTrue(reader.metric_gate("100","100.5","101",False)["passed"])
        self.assertFalse(reader.metric_gate("100","100.500000001","101",False)["passed"])
        self.assertFalse(reader.metric_gate("100","96","101.000000001",True)["passed"])

    def test_decode_is_primary_not_prefill_and_warmup_never_a_formal_sample(self):
        rows={}
        for arm in ("A1","B","A2"):
            rows[arm]=[dict(run=i,warmup=i==0,exact_native_metrics=dict(
                decode_ms_per_token="999999" if i==0 and arm=="B" else "97" if arm=="B" else "100",
                ttft_s="1.005" if arm=="B" else "1",prefill_s="1.005" if arm=="B" else "1"))for i in range(4)]
        result=reader.performance_summary(rows)
        self.assertTrue(result["passed"]);self.assertEqual(result["metrics"]["decode_ms_per_token"]["b_median"],97)
        for sample in rows["B"][1:]:
            sample["exact_native_metrics"].update(decode_ms_per_token="98",prefill_s="0.1")
        self.assertFalse(reader.performance_summary(rows)["passed"])
        rows["A2"].pop()
        with self.assertRaises(reader.EvidenceError):reader.performance_summary(rows)

    def test_dynamic_Q_and_P_read_every_N256_payload_including_formal3(self):
        for repeats in (1,3):
            with self.subTest(repeats=repeats),tempfile.TemporaryDirectory()as d:
                root=Path(d);stdout,_=raw_fixture(root,repeats)
                with patch.object(reader.core,"read_run_exports",side_effect=AssertionError("prefill fixedQ helper forbidden")):
                    result=reader.read_run_exports(root,stdout,1,9,256,repeats)
                self.assertEqual(set(result["exports"]),set(range(repeats+1)))
                self.assertTrue(result["provider_adoption_passed"]);self.assertTrue(result["finite_logits_passed"])
                self.assertEqual(result["samples"][-1]["exact_native_metrics"]["prefill_s"],"1.000000001")
                self.assertEqual(result["samples"][-1]["provider"]["counters"]["decode_encoded"],8160)

    def test_last_formal_truthful_counter_failure_keeps_raw_and_cannot_pass(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);stdout,records=raw_fixture(root)
            records[-1]["gpu_decode_kv_only_provider"].update(failures=1,decode_failures=1,adoption_valid=False)
            records[-1]["sample_valid"]=False
            stdout.write_text("\n".join(json.dumps(r)for r in records)+"\n")
            result=reader.read_run_exports(root,stdout,1,9,256,3)
            self.assertEqual(set(result["exports"]),{0,1,2,3});self.assertFalse(result["provider_adoption_passed"])
            self.assertFalse(result["native_declared_valid"]);self.assertFalse(result["samples"][-1]["provider"]["passed"])
            self.assertTrue(result["samples"][-1]["raw_files"])

    def test_missing_duplicate_or_corrupt_last_formal_is_evidence_invalid(self):
        for mutation in ("missing3","duplicate2","short_raw"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as d:
                root=Path(d);stdout,records=raw_fixture(root)
                if mutation=="missing3":records.pop()
                elif mutation=="duplicate2":records[-1]["run"]=2
                else:(root/"state/run-3/owner-3-index-active.bf16").write_bytes(b"bad")
                stdout.write_text("\n".join(json.dumps(r)for r in records)+"\n")
                with self.assertRaises(reader.EvidenceError):reader.read_run_exports(root,stdout,1,9,256,3)

    def test_Q_gold_bridge_and_candidate_differences_are_separate_full_byte_failures(self):
        with tempfile.TemporaryDirectory()as d:
            a,b=Path(d)/"a",Path(d)/"b";a.write_bytes(struct.pack("<f",0.));b.write_bytes(struct.pack("<f",-0.))
            def e(path):return dict(files={"final-logits.f32":path},metadata={},finite_logits=True)
            for gold,off,on,gb,cb in ((a,b,b,False,True),(a,a,b,True,False)):
                exports={arm:{r:e(path)for r in (0,1)}for arm,path in (("G_lib",gold),("off",off),("on",on))}
                result=reader.raw_comparisons(exports,"Q")
                self.assertFalse(result["passed"]);self.assertEqual(result["gold_library_bridge_passed"],gb);self.assertEqual(result["candidate_reference_passed"],cb)
            exports={arm:{r:e(a)for r in range(4)}for arm in ("A1","B","A2")};exports["B"][3]=e(b)
            result=reader.raw_comparisons(exports,"P");self.assertFalse(result["passed"]);self.assertEqual(len(result["comparisons"]),11)

    def test_identical_nonfinite_final_logits_truthfully_declared_remain_numerical_failure(self):
        for value in (float("nan"),float("inf")):
            with self.subTest(value=value),tempfile.TemporaryDirectory()as d:
                root=Path(d);stdout,records=raw_fixture(root,1)
                data=struct.pack("<fff",value,1,-2)
                for run in (0,1):
                    (root/"logits"/f"run-{run}.f32").write_bytes(data)
                    records[run+3]["final_logits"].update(finite=False,fnv1a64_bytes=reader.fnv1a(data));records[run+3]["sample_valid"]=False
                stdout.write_text("\n".join(json.dumps(r)for r in records)+"\n")
                result=reader.read_run_exports(root,stdout,1,9,256,1)
                self.assertFalse(result["finite_logits_passed"]);self.assertFalse(reader.core.compare_exports(result["exports"][0],result["exports"][1])["passed"])

    def test_M4_component_or_truthful_failure_cannot_authorize_target_M2_Q(self):
        spec=dict(protocol_sha256="76a066e601e9788b24550d0014d21acdc3e4a285d6672c9696a0e19cfb046b00",
            reader_sha256="7f42684ec6f76e3921d8bb811205f2627c454bab95f06be8ef2eb4cc0f0e555f",device_name="Apple M2 Ultra",future_report_exact_descriptor_bound_in_prelaunch_plan=True)
        report=dict(review_completed=True,protocol_sha256=spec["protocol_sha256"],reader_sha256=spec["reader_sha256"],correctness_passed=True,numerical_passed=True,
            GPU_completion_passed=True,raw_integrity_passed=True,complete_run=True,complete_command_buffers=342,complete_cases=114,complete_role_outputs=1026,
            complete_guard_files=2052,total_raw_files=3535,device_name="Apple M2 Ultra",timing_evaluated=False,harness_failure=None,manifest_status="synthetic_correctness_complete_not_adoption")
        reader.prerequisite_gate("Q",report,spec,{},"bin")
        for key,val in (("device_name","Apple M4"),("complete_run",False),("numerical_passed",False),("GPU_completion_passed",False),("harness_failure","failed"),("total_raw_files",3534)):
            with self.subTest(key=key),self.assertRaises(reader.EvidenceError):reader.prerequisite_gate("Q",dict(report,**{key:val}),spec,{},"bin")

    def test_no_P_performance_after_numerical_failure_or_one_workload_failure(self):
        groups=[dict(workload="short",passed=True),dict(workload="business2",passed=True)]
        self.assertTrue(reader.stage_decision("P",True,groups)["native_decode_gate_passed"])
        for numerical,rows in ((False,groups),(True,groups[:1]),(True,[groups[0],dict(groups[1],passed=False)])):
            r=reader.stage_decision("P",numerical,rows);self.assertFalse(r["performance_acceptance"]);self.assertFalse(r["production_acceptance"])
        r=reader.evidence_failure(dict(native_decode_gate_passed=True,performance_acceptance=True,raw_logical_bytes=123),reader.EvidenceError("source SHA changed"))
        self.assertEqual(r["raw_logical_bytes"],123)
        for k in ("review_completed","native_decode_gate_passed","performance_acceptance","qualification_passed","formal_performance_cohort_eligible"):
            self.assertIs(r[k],False)

    def test_Q_P_fixed_order_rejects_duplicates_and_draft_initialize_rejects_before_artifacts(self):
        for stage in ("Q","P"):
            order=reader.stage_order(stage);self.assertEqual(reader.checked_order(order,stage),order)
            for bad in ([],order+[order[-1]],[order[0],order[0]]):
                with self.assertRaises(reader.EvidenceError):reader.checked_order(bad,stage)
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);proto=root/"draft.json";proto.write_text(json.dumps(dict(stage="Q",schema_version=1,status="draft_pending")))
            args=SimpleNamespace(protocol=proto,window_root=root/"none",private_build_manifest=root/"private",target_build_manifest=root/"target",path_map=[])
            with self.assertRaisesRegex(reader.EvidenceError,"final frozen"):reader.Review(args).initialize()

    def test_all_config_entries_and_four_extras_with_indexer_fixed_off(self):
        raw=b"MYDS_PREFIX_CACHE=1\nMYDS_PLD=1\nMYDS_METAL_CONCURRENT=1\nMYDS_BATCH2_SPEC=0\nMYDS_DSPARK=0\n"
        base=dict(MYDS_PREFIX_CACHE="1",MYDS_PLD="1",MYDS_METAL_CONCURRENT="1",MYDS_BATCH2_SPEC="0",MYDS_DSPARK="0")
        expected=dict(base,MYDS_DECODE_KV_ONLY_FINISH="1",MYDS_INDEXER_K_RESIDENT="0",MYDS_BENCH_LOGITS_DIR="/r/logits",MYDS_BENCH_STATE_DIR="/r/state")
        outer=dict(kind="decode_kv_metadata",schema_version=1,subject="decode_kv_only_finish",execute_requested=True,provider_adapter_available=True,
            fresh_context_per_sample=True,cache_configuration_preserved=True,config_load_override_env=True,config_file="/frozen.conf",config_loaded_entries=5,
            config_file_bytes=len(raw),config_file_fnv1a64=reader.fnv1a(raw),config_raw=raw.decode(),effective_configuration_count=9,effective_configuration=expected,
            runtime_mode=1,requested_symbol="kernel_myds_probe_kv_only_finish")
        metadata=dict(environment=dict(expected,MYDS_UNSET=None))
        self.assertEqual(reader.effective_configuration(outer,metadata,raw,1,"/r/logits","/r/state","/frozen.conf",expected_count=5)["effective_entries"],9)
        for key,val in (("MYDS_INDEXER_K_RESIDENT","1"),("MYDS_PLD","0"),("MYDS_SPEC_SERIAL_PROBE","1"),("MYDS_DECODE_METRICS","1")):
            bad=deepcopy(outer);bad["effective_configuration"][key]=val
            with self.subTest(key=key),self.assertRaises(reader.EvidenceError):reader.effective_configuration(bad,metadata,raw,1,"/r/logits","/r/state","/frozen.conf",expected_count=5)

    def test_P_requires_actual_completed_same_reader_source_inputs_Q_and_prelaunch_binding(self):
        protocol=dict(artifacts=dict(private_build_manifest=dict(sha256="source"),target_build_manifest=dict(sha256="target"),reader=dict(sha256="newreader"),
            qualification_protocol=dict(sha256="qprotocol")),workloads=["frozeninputs"],model=dict(prefix_sha256="model"))
        spec=dict(protocol_sha256="qprotocol",reader_sha256="newreader",future_report_exact_descriptor_bound_in_prelaunch_plan=True)
        report=dict(review_completed=True,protocol_sha256="qprotocol",reader_sha256="newreader",qualification_passed=True,N256_two_workload_qualification_completed=True,
            formal_performance_cohort_eligible=True,execution_complete=True,native_generation_count=12,providers=[{}]*6,binary_sha256="bin",private_build_manifest_sha256="source",
            target_build_manifest_sha256="target",workload_identities=["frozeninputs"],model_identity_scope=dict(prefix_sha256="model"),production_acceptance=False,
            subject="decode_kv_only_finish",stage="Q",core_reader_sha256=reader.CORE_SHA,
            workload_numerical_comparisons=[dict(workload=w,passed=True,gold_library_bridge_passed=True,candidate_reference_passed=True)for w in ("short","business2")])
        reader.prerequisite_gate("P",report,spec,protocol,"bin")
        for key,val in (("qualification_passed",False),("native_generation_count",6),("stage","P"),("subject","prefill_indexer_k_resident"),
                        ("core_reader_sha256","other"),("binary_sha256","other"),("workload_identities",["other"]),("N256_two_workload_qualification_completed",False)):
            with self.subTest(key=key),self.assertRaises(reader.EvidenceError):reader.prerequisite_gate("P",dict(report,**{key:val}),spec,protocol,"bin")
        bad=deepcopy(report);bad["workload_numerical_comparisons"][1]["candidate_reference_passed"]=False
        with self.assertRaises(reader.EvidenceError):reader.prerequisite_gate("P",bad,spec,protocol,"bin")
        with self.assertRaises(reader.EvidenceError):reader.prerequisite_gate("P",report,dict(spec,future_report_exact_descriptor_bound_in_prelaunch_plan=False),protocol,"bin")


if __name__=="__main__":unittest.main()
