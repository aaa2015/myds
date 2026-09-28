#!/usr/bin/env python3
"""Small, no-model regression tests for native/reference diagnostic boundaries."""
import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compare_traces import compare_stage, indices


class StreamComparisonTests(unittest.TestCase):
    def test_repeated_embedding_is_valid_at_later_positions(self):
        values = [1.0, 2.0] * 4
        result = compare_stage(values, values, "embedding", 64)
        self.assertTrue(result["pass"])
        self.assertEqual(len(result["streams"]), 4)

    def test_collapsed_post_hc_fails_even_when_equal_to_reference(self):
        values = [1.0, 2.0] * 4
        result = compare_stage(values, values, "layer_out", 8)
        self.assertFalse(result["pass"])
        self.assertFalse(result["reference_stream_diversity"]["pass"])

    def test_every_stream_pair_must_differ(self):
        values = [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = compare_stage(values, values, "layer_out", 1)
        self.assertFalse(result["pass"])
        self.assertEqual(len(result["candidate_stream_diversity"]["pairs"]), 6)

    def test_distinct_streams_and_per_stream_errors(self):
        reference = list(map(float, range(8)))
        candidate = reference.copy()
        candidate[5] += 2
        result = compare_stage(reference, candidate, "layer_out", 129)
        self.assertFalse(result["pass"])
        self.assertEqual([s["max_abs_error"] for s in result["streams"]], [0, 0, 2, 0])
        self.assertTrue(result["candidate_stream_diversity"]["pass"])

    def test_shape_mismatch_never_repeats_candidate(self):
        result = compare_stage([1.0, 2.0] * 4, [1.0, 2.0], "embedding", 0)
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "shape mismatch")
        self.assertNotIn("streams", result)

    def test_nonfinite_rejected(self):
        for value in (float("inf"), float("nan")):
            result = compare_stage([value] * 4, [value] * 4, "layer_out", 8)
            self.assertFalse(result["pass"])
            self.assertEqual(result["reason"], "nonfinite values")

    def test_malformed_stream_shape_rejected(self):
        self.assertFalse(compare_stage([1.0] * 5, [1.0] * 5, "layer_out", 1)["pass"])

    def test_csv_indices(self):
        self.assertEqual(indices("1,8,64,129"), [1, 8, 64, 129])
        for value in ("1,", "1,1", "-1", "1,all", " 1", "4294967296"):
            with self.assertRaises(Exception):
                indices(value)

    def test_multi_position_multi_layer_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pos in (1, 8, 64, 129):
                for layer in (0, 1, 14):
                    for prefix in ("reference", "myds"):
                        (root / f"{prefix}_layer_out-{layer}_pos{pos}.bin").write_bytes(
                            array.array("f", range(8)).tobytes())
            cmd = [sys.executable, str(HERE.parent / "compare_traces.py"), str(root),
                   "--positions", "1,8,64,129", "--layers", "0,1,14", "--stages", "layer_out"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)["captures"]), 12)
            (root / "myds_layer_out-14_pos129.bin").unlink()
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(result.stdout)["captures"][-1]["pass"])

    def test_unexplained_tolerance_rejected(self):
        result = subprocess.run([sys.executable, str(HERE.parent / "compare_traces.py"),
                                 "/tmp", "--atol", "0.1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--tolerance-reason", result.stderr)


class ExactTokenInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source = cls.root / "input_test.c"
        source.write_text('''#include "trace_input.h"
int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "filter")) {
        printf("%d %d\\n", trace_filter_valid(argv[2]),
               trace_filter_matches(argv[2], (uint32_t)strtoul(argv[3], NULL, 10)));
        return 0;
    }
    trace_tokens tokens = {0};
    int ok = trace_tokens_parse(&tokens, argc, argv);
    for (size_t i = 0; i < tokens.count; i++) printf("%d ", tokens.ids[i]);
    free(tokens.ids);
    return ok ? 0 : 2;
}
''')
        cls.binary = cls.root / "input_test"
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(HERE),
                        str(source), "-o", str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_input(self, *args):
        return subprocess.run([str(self.binary), "model", *args], capture_output=True, text=True)

    def test_positional_exact_ids(self):
        result = self.run_input("35", "0", "2147483647")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.split(), ["35", "0", "2147483647"])

    def test_file_whitespace_comments_and_final_token(self):
        path = self.root / "tokens.txt"
        path.write_text("# test\n35 47\t671 # inline\n 129280\r\n0")
        result = self.run_input("--tokens-file", str(path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["35", "47", "671", "129280", "0"])

    def test_bad_file_and_ids_fail(self):
        path = self.root / "bad.txt"
        for data in (b"", b"# empty", b"35 -1", b"35 +1", b"35 1.5", b"35,47", b"35\x0047", b"2147483648", b"9" * 128):
            path.write_bytes(data)
            self.assertEqual(self.run_input("--tokens-file", str(path)).returncode, 2, data)
        for value in ("-1", "+1", "1,8", "2147483648", "abc", ""):
            self.assertEqual(self.run_input(value).returncode, 2, value)

    def test_filters_parse_complete_csv(self):
        cases = [("all", 500, "1 1"), ("", 8, "1 1"), ("1,8,64,129", 64, "1 1"),
                 ("1,8,64,129", 9, "1 0"), ("1,", 1, "0 0"), ("-1", 1, "0 0"),
                 ("4294967296", 0, "0 0"), ("0junk", 0, "0 0")]
        for value, target, expected in cases:
            result = subprocess.run([str(self.binary), "filter", value, str(target)],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
