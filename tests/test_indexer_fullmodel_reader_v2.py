#!/usr/bin/env python3
"""CPU regression and negative cases for the exact Q0 path-alias repair."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import review_indexer_fullmodel_v2 as reader
import test_indexer_fullmodel_reader as original_tests


class AllOriginalContracts(original_tests.FullmodelReaderTests):
    def setUp(self):
        self.previous_reader = original_tests.reader
        original_tests.reader = reader

    def tearDown(self):
        original_tests.reader = self.previous_reader


class AliasRepairTests(unittest.TestCase):
    def test_identical_path_and_only_macos_tmp_alias_pass_same_file_sha(self):
        with tempfile.TemporaryDirectory() as d:
            frozen = Path(d) / "myds.metallib"; frozen.write_bytes(b"frozen library")
            a, b = "/tmp/frozen/target/myds.metallib", "/private/tmp/frozen/target/myds.metallib"
            for actual, expected in ((a, b), (b, a), (b, b)):
                reader.loaded_library_identity(actual, expected, b, reader.file_sha(frozen), lambda _: frozen)

    def test_other_paths_cannot_be_made_valid_by_arbitrary_path_maps(self):
        with tempfile.TemporaryDirectory() as d:
            frozen = Path(d) / "myds.metallib"; frozen.write_bytes(b"frozen library")
            expected = "/private/tmp/frozen/target/myds.metallib"
            for actual in ("/var/tmp/frozen/target/myds.metallib", "/tmp/other/target/myds.metallib",
                           "/tmp/frozen/target/other.metallib", "/tmp/frozen/target/../target/myds.metallib"):
                with self.subTest(actual=actual), self.assertRaises(reader.EvidenceError):
                    reader.loaded_library_identity(actual, expected, expected, reader.file_sha(frozen), lambda _: frozen)

    def test_same_alias_never_waives_frozen_complete_library_sha(self):
        with tempfile.TemporaryDirectory() as d:
            frozen = Path(d) / "myds.metallib"; frozen.write_bytes(b"original library")
            expected_sha = reader.file_sha(frozen); frozen.write_bytes(b"changed library!")
            a, b = "/tmp/frozen/target/myds.metallib", "/private/tmp/frozen/target/myds.metallib"
            with self.assertRaises(reader.EvidenceError):
                reader.loaded_library_identity(a, b, b, expected_sha, lambda _: frozen)

    def test_same_name_and_identical_bytes_still_require_same_frozen_resolved_file(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"; a.mkdir(); b.mkdir()
            left, right = a / "myds.metallib", b / "myds.metallib"
            left.write_bytes(b"identical bytes"); right.write_bytes(b"identical bytes")
            actual, expected = "/tmp/frozen/target/myds.metallib", "/private/tmp/frozen/target/myds.metallib"
            def mapped(value):
                return left if value == actual else right
            with self.assertRaises(reader.EvidenceError):
                reader.loaded_library_identity(actual, expected, expected, reader.file_sha(right), mapped)


if __name__ == "__main__":
    unittest.main()
