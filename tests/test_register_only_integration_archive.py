#!/usr/bin/env python3
"""Tiny local CPU fixtures only; no remote, model, signal, or production reads."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('integration_archive', ROOT / 'tools/archive_register_only_integration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CORE = ROOT / 'docs/benchmarks/2026-10-26-register-only-and-narrow-audit/fullmodel-durable-evidence-v1.archiver.py'
if not CORE.exists():
    CORE = ROOT.parent / 'mydsdoc/benchmarks/2026-10-26-register-only-and-narrow-audit/fullmodel-durable-evidence-v1.archiver.py'


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='integration-archive-CPU-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / 'integration'; self.source.mkdir()
        self.window = self.source / 'service-window-v1'; self.window.mkdir(mode=0o700)
        failed = self.source / 'native-build-v1'; failed.mkdir()
        (failed / 'failed.stdout').write_text('retained failed build\n')
        (self.source / 'Q.stdout').write_text('tiny synthetic raw\n')
        self.secret = 'CPU_FIXTURE_FULL_ENV_MUST_STAY_PRIVATE'
        self.lease = dict(restored=True, restore_blocked=False, finished_epoch=1,
                          environment={'PRIVATE_SECRET': self.secret}, identities={})
        (self.window / 'lease.json').write_text(json.dumps(self.lease))
        (self.window / 'summary.json').write_text(json.dumps(dict(restored=True, restore_blocked=False)))
        for name in m.LIVE_NAMES:
            (self.window / name).write_text('frozen original restored log\n')
        self.out = self.base / 'private-archive'
        self.core = m.load_core(CORE)
        old_umask = os.umask(0o077)
        self.addCleanup(os.umask, old_umask)

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return m.execute(self.source, self.out, self.core)

    def test_exact_private_archive_with_frozen_logs_and_failed_build(self):
        report = self.execute()
        self.assertTrue(report['all_logical_bytes_SHA_and_modes_verified'])
        self.assertTrue(report['original_source_closure_stat_SHA_unchanged'])
        self.assertTrue(report['failed_native_build_v1_preserved'])
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in self.out.iterdir()))
        public = (self.out / 'public-receipt.json').read_text()
        self.assertNotIn(self.secret, public)
        self.assertNotIn('PRIVATE_SECRET', public)
        with tarfile.open(self.out / (m.STEM + '.tar.gz')) as archive:
            names = archive.getnames()
            prefix = 'origin/M2' + str(self.source)
            self.assertIn(prefix + '/native-build-v1/failed.stdout', names)
            for name in m.LIVE_NAMES:
                self.assertNotIn(prefix + '/service-window-v1/' + name, names)
                self.assertIn(prefix + '/final-service-record/' + name, names)
            private = archive.extractfile(prefix + '/final-service-record/lease.json').read().decode()
            self.assertIn(self.secret, private)
        for line in (self.out / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(self.core.digest(self.out / name), digest)

    def test_active_log_append_does_not_change_frozen_snapshot(self):
        original_execute = self.core.execute
        def append_then_archive(*args):
            for name in m.LIVE_NAMES:
                with (self.window / name).open('a') as stream:
                    stream.write('later service activity\n')
            return original_execute(*args)
        self.core.execute = append_then_archive
        self.execute()
        for name in m.LIVE_NAMES:
            self.assertEqual((self.source / 'final-service-record' / name).read_text(),
                             'frozen original restored log\n')
            self.assertIn('later service activity', (self.window / name).read_text())

    def test_unrestored_window_refused(self):
        (self.window / 'summary.json').write_text(json.dumps(dict(restored=False, restore_blocked=True)))
        with self.assertRaisesRegex(ValueError, 'restoration required'):
            self.execute()
        self.assertFalse((self.source / 'final-service-record').exists())
        self.assertFalse((self.out / (m.STEM + '.tar.gz')).exists())

    def test_existing_output_refused(self):
        self.out.mkdir(); original = self.out / 'keep'; original.write_text('unchanged')
        with self.assertRaisesRegex(ValueError, 'new archive'):
            self.execute()
        self.assertEqual(original.read_text(), 'unchanged')
        self.assertFalse((self.source / 'final-service-record').exists())


if __name__ == '__main__':
    unittest.main()
