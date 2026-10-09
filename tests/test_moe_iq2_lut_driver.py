#!/usr/bin/env python3
"""CPU-only failure artifact tests; no Metal executable is invoked."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('iq2_component_driver', Path(__file__).with_name('test_moe_iq2_lut.py'))
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class FailureEvidence(unittest.TestCase):
    def test_nonzero_exit_retains_both_logs_before_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory)/'stdout.log', Path(directory)/'stderr.log'
            result = subprocess.CompletedProcess(['fixture'], 7, 'before failure\n', 'INVALID_F32 gate index=256\n')
            with mock.patch.object(DRIVER.subprocess, 'run', return_value=result):
                with self.assertRaises(RuntimeError):
                    DRIVER.run(Path('/cpu-only-fixture'), [], None, 4, out, err)
            self.assertEqual(out.read_text(), result.stdout)
            self.assertEqual(err.read_text(), result.stderr)
            process = json.loads(out.with_name('stdout.log.process.json').read_text())
            self.assertEqual(process['returncode'], 7)
            self.assertFalse(process['timed_out'])

    def test_timeout_retains_partial_byte_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory)/'stdout.log', Path(directory)/'stderr.log'
            failure = subprocess.TimeoutExpired(['fixture'], 180, output=b'partial out\n', stderr=b'partial err\n')
            with mock.patch.object(DRIVER.subprocess, 'run', side_effect=failure):
                with self.assertRaises(RuntimeError):
                    DRIVER.run(Path('/cpu-only-fixture'), [], '1', 2, out, err)
            self.assertEqual(out.read_text(), 'partial out\n')
            self.assertEqual(err.read_text(), 'partial err\n')
            process = json.loads(out.with_name('stdout.log.process.json').read_text())
            self.assertTrue(process['timed_out'])
            self.assertIsNone(process['returncode'])

    def test_success_still_parses_complete_record_and_retains_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory)/'stdout.log', Path(directory)/'stderr.log'
            result = subprocess.CompletedProcess(['fixture'], 0, '{"kind":"complete","pass":true}\n', 'diagnostic\n')
            with mock.patch.object(DRIVER.subprocess, 'run', return_value=result):
                records, stdout, stderr = DRIVER.run(Path('/cpu-only-fixture'), [], '0', 4, out, err)
            self.assertEqual(records, [{'kind':'complete', 'pass':True}])
            self.assertEqual(out.read_text(), stdout)
            self.assertEqual(err.read_text(), stderr)

    def test_failure_without_explicit_artifacts_keeps_temp_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(DRIVER.tempfile, 'mkdtemp', return_value=directory), \
                 mock.patch.object(DRIVER, 'suite', side_effect=RuntimeError('component failure')), \
                 mock.patch('sys.argv', ['test_moe_iq2_lut.py', '--host-only']):
                with self.assertRaises(RuntimeError):
                    DRIVER.main()
            failure = json.loads((Path(directory)/'failure.json').read_text())
            self.assertFalse(failure['pass'])
            self.assertEqual(failure['retained_artifacts'], directory)


if __name__ == '__main__':
    unittest.main()
