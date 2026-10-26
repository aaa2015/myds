#!/usr/bin/env python3
"""CPU-only driver contracts: no model file, Metal device, service or GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / "tools/bench_output_a_fullmodel.c").read_text()
spec = importlib.util.spec_from_file_location("indexer_driver_prepare", ROOT / "tools/prepare_indexer_fullmodel.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def section(start, end):
    return BASE[BASE.index(start):BASE.index(end, BASE.index(start))]


HASH = section("static uint64_t hash_bytes(", "static void capture(")
READ = section("static char *read_request(", "static void json_string(")
JSON = section("static void json_string(", "static const char *stop_name(")
STATE_HELPERS = section("static bool state_file(", "static bool export_state(")

ADAPTER = r'''
static unsigned fixture_stage;
static int bench_indexer_adapter_mode(void) { return getenv("TEST_MODE") ? atoi(getenv("TEST_MODE")) : 1; }
static const char *bench_indexer_adapter_symbol(bool bf16) {
    if (getenv("TEST_NULL_SYMBOL") || (bf16 && getenv("TEST_NULL_BF16_SYMBOL"))) return NULL;
    if (getenv("TEST_EMPTY_SYMBOL")) return "";
    return bf16 ? "fixture_bf16" : "fixture_f32";
}
static void bench_indexer_adapter_counters(bench_indexer_provider_counters_t *out, bool reset) {
    if (reset) fixture_stage = 0;
    if (out) {
        memset(out, 0, sizeof(*out));
        out->requests = fixture_stage;
        unsigned encoded = fixture_stage ? 10 : 0;
        if (getenv("TEST_ENCODE_ONLY_AFTER_FIRST")) encoded = fixture_stage > 10 ? fixture_stage - 10 : 0;
        if (getenv("TEST_DECODE_ENCODING") || getenv("TEST_COUNTER_REWIND")) encoded = fixture_stage;
        out->encoded = out->bf16_dispatches = bench_indexer_adapter_mode() && !getenv("TEST_NO_ENCODING") ? encoded : 0;
        if (getenv("TEST_FALLBACK")) out->fallbacks = 1;
    }
}
'''

STATE_FIXTURE = r'''
#define _POSIX_C_SOURCE 200809L
#include "myds.h"
#include "myds_internal.h"
#include "myds_metal.h"
#include "myds_v41.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
enum { MAX_OUTPUT_TOKENS = 512, MAX_OUTPUT_BYTES = 4*1024*1024 };
typedef struct { unsigned char *raw_bytes; uint32_t count; int32_t ids[512]; uint64_t byte_count; } capture_t;
struct myds_metal_buffer { void *pointer; size_t size; };
static float backing[40u*128u*512u];
static struct myds_metal_buffer buffer = {backing, sizeof(backing)};
static bool published;
void *myds_metal_buffer_data(myds_metal_buffer_t *b) { return b->pointer; }
size_t myds_metal_buffer_size(const myds_metal_buffer_t *b) { return b->size; }
uint32_t myds_v41_snapshot_get_pos(const myds_v41_context_t *g,uint32_t s) { (void)g; return published && s == 0 ? 4 : 0; }
uint32_t myds_v41_snapshot_get_staged_pos(const myds_v41_context_t *g,uint32_t s) { (void)g;(void)s;return 0; }
void myds_v41_snapshot_get_engram_history(const myds_v41_context_t *g,uint32_t out[3]) { (void)g;out[0]=1;out[1]=2;out[2]=3; }
uint32_t myds_v41_snapshot_get_next_position(const myds_v41_context_t *g) { (void)g;return 8; }
uint32_t myds_v41_snapshot_get_engram_position(const myds_v41_context_t *g) { (void)g;return 8; }
bool myds_v41_snapshot_get_valid(const myds_v41_context_t *g) { (void)g;return true; }
bool myds_v41_low_kv_enabled(const myds_v41_context_t *g) { (void)g;return false; }
uint64_t myds_v41_snapshot_allocated_bytes(const myds_v41_context_t *g) { (void)g;return 0; }
#define BUFFER_GETTER(n) myds_metal_buffer_t *n(myds_v41_context_t *g,uint32_t i) { (void)g;(void)i;return &buffer; }
BUFFER_GETTER(myds_v41_snapshot_get_window)
BUFFER_GETTER(myds_v41_snapshot_get_previous_kv)
BUFFER_GETTER(myds_v41_snapshot_get_previous_score)
BUFFER_GETTER(myds_v41_snapshot_get_compressed)
BUFFER_GETTER(myds_v41_snapshot_get_index_cache)
BUFFER_GETTER(myds_v41_snapshot_get_snap_window)
#define SNAP_GETTER(n) myds_metal_buffer_t *n(myds_v41_context_t *g,uint32_t s,uint32_t o) { (void)g;(void)s;(void)o;return &buffer; }
SNAP_GETTER(myds_v41_snapshot_get_snap_prev_kv)
SNAP_GETTER(myds_v41_snapshot_get_snap_prev_score)
'''

STATE_MAIN = r'''
#include "indexer_fullmodel_state.inc"
int main(int argc,char **argv) {
    if (argc != 3) return 3;
    setenv("MYDS_BENCH_STATE_DIR",argv[1],1);
    int32_t anchor[2048], prefix[2]={9,10}, short_anchor[4]={101,102,103,104};
    for (unsigned i=0;i<2048;i++) anchor[i]=(int32_t)i;
    myds_context_t ctx={0};ctx.v41=(myds_v41_context_t *)(uintptr_t)1;
    ctx.cfg.max_seq_len=8192;ctx.pos=8;ctx.pcache_tokens=prefix;ctx.pcache_len=ctx.pcache_cap=2;
    ctx.anchor_slots=3;ctx.anchor_cap=4096;ctx.anchor_n=2;
    ctx.anchor_tokens[0]=anchor;ctx.anchor_len[0]=2048;
    ctx.anchor_tokens[1]=short_anchor;ctx.anchor_len[1]=4;
    capture_t captured={0};unsigned char text[]={0,0xe4,0xb8,0xad};
    captured.raw_bytes=text;captured.byte_count=sizeof(text);captured.count=2;captured.ids[0]=17;captured.ids[1]=18;
    if (!strcmp(argv[2],"overflow")) ctx.anchor_len[0]=4097;
    if (!strcmp(argv[2],"bad_slots")) ctx.anchor_slots=MYDS_ANCHOR_SLOTS_MAX+1;
    if (!strcmp(argv[2],"bad_pointer")) ctx.anchor_tokens[0]=NULL;
    if (!strcmp(argv[2],"bad_count")) ctx.anchor_n=1;
    if (!strcmp(argv[2],"prefix_overflow")) ctx.pcache_len=3;
    if (!strcmp(argv[2],"published")) published=true;
    return export_state(0,&ctx,&captured) ? 0 : 2;
}
'''


class DriverContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="myds-indexer-driver-cpu-")
        cls.root = Path(cls.temporary.name)
        stub = cls.root / "base_stub.inc"
        stub.write_text(HASH + READ + JSON + r'''
int main(int argc,char **argv) {
    (void)argc;(void)argv;
#ifdef MYDS_BENCH_INDEXER_ADAPTER_HEADER
    bench_indexer_before_sample();fixture_stage=getenv("TEST_COUNTER_REWIND")?20:10;bench_indexer_first_callback();fixture_stage=14;
    fputs("{\"kind\":\"fixture\"",stdout);bench_indexer_sample_json(stdout,1);fputs("}\n",stdout);
    return bench_indexer_provider_valid() ? 17 : 19;
#else
    fputs("BASE_MAIN_REACHED\n",stderr);return 17;
#endif
}
''')
        adapter = cls.root / "adapter.inc"
        adapter.write_text(ADAPTER)
        common = ["cc", "-std=c11", "-I" + str(ROOT / "include"), "-I" + str(ROOT / "tools")]
        for name, extra in (("cli", []), ("cli_adapter", ['-DMYDS_BENCH_INDEXER_ADAPTER_HEADER="' + str(adapter) + '"'])):
            command = common + ['-DMYDS_BENCH_INDEXER_BASE_INCLUDE="' + str(stub) + '"'] + extra
            command += [str(ROOT / "tools/bench_indexer_fullmodel.c"), str(ROOT / "src/myds_configfile.c"),
                        str(ROOT / "src/myds_log.c"), "-lpthread", "-o", str(cls.root / name)]
            compiled = subprocess.run(command, capture_output=True, text=True)
            if compiled.returncode:
                raise RuntimeError(compiled.stderr)
        source = cls.root / "state.c"
        source.write_text(STATE_FIXTURE + HASH + JSON + STATE_HELPERS + STATE_MAIN)
        compiled = subprocess.run(common + [str(source), "-o", str(cls.root / "state")], capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def environment(self, directory):
        env = {key: value for key, value in os.environ.items() if not key.startswith("MYDS_")}
        for name in ("logits", "state"):
            (directory / name).mkdir()
        env.update(MYDS_BENCH_LOGITS_DIR=str(directory / "logits"), MYDS_BENCH_STATE_DIR=str(directory / "state"),
                   MYDS_PREFIX_CACHE="0", MYDS_TEST_EXTRA="preserved inherited selector")
        return env

    def run_cli(self, directory, env, config_text="MYDS_PREFIX_CACHE=1\nMYDS_PLD=1\n", run=False, adapter=False):
        config = directory / "config"
        config.write_text(config_text)
        args = [str(self.root / ("cli_adapter" if adapter else "cli")), "--config", str(config),
                "--run" if run else "--preflight"]
        if run:
            args += ["/not-loaded.gguf", "/not-read.json"]
        return subprocess.run(args, env=env, capture_output=True, text=True)

    def test_checked_adapter_preserves_base_and_rejects_drift(self):
        adapted, proof = prepare.prepare(BASE)
        self.assertTrue(proof["reverse_restores_base_byte_for_byte"])
        self.assertIn('#include "indexer_fullmodel_state.inc"', adapted)
        self.assertNotIn("ctx->anchor_len>1024u", adapted)
        self.assertEqual((ROOT / "tools/bench_output_a_fullmodel.c").read_text(), BASE)
        for mutation in (BASE.replace("ctx->anchor_len>1024u", "ctx->anchor_len>2048u"),
                         BASE.replace("        c->first = t;", "        c->first = 0;")):
            with self.assertRaises(ValueError):
                prepare.prepare(mutation)

    def test_preflight_loads_config_before_base_and_records_all_effective_values(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temp:
            directory = Path(temp); env = self.environment(directory)
            result = self.run_cli(directory, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)
            self.assertEqual(metadata["subject"], "prefill_indexer_k_resident")
            self.assertEqual(metadata["effective_configuration"]["MYDS_PREFIX_CACHE"], "1")
            self.assertEqual(metadata["effective_configuration"]["MYDS_PLD"], "1")
            self.assertEqual(metadata["effective_configuration"]["MYDS_TEST_EXTRA"], "preserved inherited selector")
            self.assertEqual(metadata["effective_configuration_count"], len(metadata["effective_configuration"]))
            self.assertFalse(metadata["execute_requested"])
            self.assertNotIn("BASE_MAIN_REACHED", result.stderr)

    def test_missing_nonempty_and_aliased_evidence_roots_rejected(self):
        for case in ("missing", "nonempty", "alias"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=self.root) as temp:
                directory = Path(temp); env = self.environment(directory)
                if case == "missing": del env["MYDS_BENCH_STATE_DIR"]
                if case == "nonempty": (directory / "logits" / "stale.f32").write_bytes(b"x")
                if case == "alias": env["MYDS_BENCH_STATE_DIR"] = env["MYDS_BENCH_LOGITS_DIR"]
                result = self.run_cli(directory, env)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")

    def test_cache_flags_are_preserved_and_invalid_config_refused(self):
        for config in ("MYDS_PREFIX_CACHE=0\nMYDS_PLD=1\n", "MYDS_PREFIX_CACHE=1\nMYDS_PLD=0\n"):
            with self.subTest(config=config), tempfile.TemporaryDirectory(dir=self.root) as temp:
                directory = Path(temp); result = self.run_cli(directory, self.environment(directory), config)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("BASE_MAIN_REACHED", result.stderr)

    def test_run_without_witness_adapter_never_reaches_base(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temp:
            directory = Path(temp); result = self.run_cli(directory, self.environment(directory), run=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("witness adapter is unavailable", result.stderr)
            self.assertNotIn("BASE_MAIN_REACHED", result.stderr)

    def test_adapter_first_callback_and_decode_delta_and_fail_closed(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=self.root) as temp:
                directory = Path(temp); env = self.environment(directory)
                if failure: env["TEST_FALLBACK"] = "1"
                result = self.run_cli(directory, env, run=True, adapter=True)
                self.assertEqual(result.returncode, 19 if failure else 17, result.stderr)
                row = json.loads(result.stdout.splitlines()[-1])["gpu_indexer_provider"]
                self.assertEqual((row["requests"], row["through_first_callback_requests"], row["decode_requests"]), (14, 10, 4))
                self.assertEqual((row["encoded"], row["through_first_callback_encoded"], row["decode_encoded"]), (10, 10, 0))
                self.assertEqual(row["adoption_valid"], not failure)

    def test_decode_only_or_additional_decode_adoption_is_rejected(self):
        for flag, expected in (("TEST_ENCODE_ONLY_AFTER_FIRST", (4, 0, 4)),
                               ("TEST_DECODE_ENCODING", (14, 10, 4))):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory(dir=self.root) as temp:
                directory = Path(temp); env = self.environment(directory); env[flag] = "1"
                result = self.run_cli(directory, env, run=True, adapter=True)
                self.assertEqual(result.returncode, 19)
                row = json.loads(result.stdout.splitlines()[-1])["gpu_indexer_provider"]
                self.assertEqual((row["encoded"], row["through_first_callback_encoded"], row["decode_encoded"]), expected)
                self.assertFalse(row["adoption_valid"])

    def test_candidate_requires_nonempty_f32_and_bf16_symbols(self):
        for flag in ("TEST_NULL_SYMBOL", "TEST_EMPTY_SYMBOL", "TEST_NULL_BF16_SYMBOL"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory(dir=self.root) as temp:
                directory = Path(temp); env = self.environment(directory); env[flag] = "1"
                result = self.run_cli(directory, env, run=True, adapter=True)
                self.assertEqual(result.returncode, 19)
                row = json.loads(result.stdout.splitlines()[-1])["gpu_indexer_provider"]
                self.assertEqual((row["encoded"], row["through_first_callback_encoded"], row["decode_encoded"]), (10, 10, 0))
                self.assertFalse(row["adoption_valid"])

    def test_counter_regression_is_rejected_without_unsigned_decode_wrap(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temp:
            directory = Path(temp); env = self.environment(directory); env["TEST_COUNTER_REWIND"] = "1"
            result = self.run_cli(directory, env, run=True, adapter=True)
            self.assertEqual(result.returncode, 19)
            row = json.loads(result.stdout.splitlines()[-1])["gpu_indexer_provider"]
            self.assertIsNone(row["decode_encoded"])
            self.assertFalse(row["adoption_valid"])

    def test_control_has_zero_candidate_encodings_and_candidate_requires_real_adoption(self):
        for mode, no_encoding, expected in (("0", False, 17), ("1", True, 19), ("2", False, 19)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.root) as temp:
                directory = Path(temp); env = self.environment(directory); env["TEST_MODE"] = mode
                if no_encoding: env["TEST_NO_ENCODING"] = "1"
                result = self.run_cli(directory, env, run=True, adapter=True)
                self.assertEqual(result.returncode, expected)
                row = json.loads(result.stdout.splitlines()[-1])["gpu_indexer_provider"]
                self.assertEqual(row["adoption_valid"], expected == 17)
                if mode == "0": self.assertEqual(row["encoded"], 0)

    def test_multislot_anchor_above_1024_exports_actual_tokens_and_state(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temp:
            result = subprocess.run([str(self.root / "state"), temp, "published"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            directory = Path(temp) / "run-0"
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual((directory / "anchor-lengths.u32le").read_bytes(), struct.pack("<3I", 2048, 4, 0))
            self.assertEqual((directory / "anchor-00-tokens.i32le").read_bytes(), struct.pack("<2048i", *range(2048)))
            self.assertEqual((directory / "anchor-01-tokens.i32le").read_bytes(), struct.pack("<4i", 101, 102, 103, 104))
            self.assertEqual((directory / "anchor-02-tokens.i32le").read_bytes(), b"")
            self.assertEqual((directory / "output.bin").read_bytes(), b"\0\xe4\xb8\xad")
            self.assertEqual((directory / "tokens.i32le").read_bytes(), struct.pack("<2i", 17, 18))
            self.assertTrue((directory / "snapshot-00-layer-39-window-active.f32").exists())
            self.assertEqual(manifest["file_count"], len(manifest["files"]))

    def test_anchor_capacity_null_pointer_slot_and_prefix_bounds_refused_before_files(self):
        for mode in ("overflow", "bad_slots", "bad_pointer", "bad_count", "prefix_overflow"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.root) as temp:
                result = subprocess.run([str(self.root / "state"), temp, mode], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
