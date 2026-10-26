#!/usr/bin/env python3
"""Private, post-restoration archive of the single register-only integration root.

Reuses the frozen fullmodel archiver without modifying it. No execution occurs
unless --execute-after-root-confirmed-native-http-ended-and-service-restored is
present. This flag records an already received authorization; it cannot grant it.
No network, service signals, models, or GPU calls are made by this tool.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time

SOURCE = Path('/private/tmp/myds-register-only-integration-20261026-v1')
BENCHMARKS = Path('/Users/fangqingyuan/Benchmarks')
CORE_SHA256 = '42cd97201e75688c08d0621b1ba903c8bdc7958dcf1b4e91fe196d6c12666f69'
STEM = 'fullmodel-evidence-v1'
LIVE_NAMES = ('restored.stdout', 'restored.stderr')


def need(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.chmod(path, 0o600)


def load_core(path):
    need(hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA256,
         'frozen fullmodel archive core SHA differs')
    spec = importlib.util.spec_from_file_location('frozen_fullmodel_archive_core', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def freeze_restoration(source, core):
    window = source / 'service-window-v1'
    summary = json.loads((window / 'summary.json').read_text())
    lease = json.loads((window / 'lease.json').read_text())
    need(summary.get('restored') is True and summary.get('restore_blocked') is False and
         lease.get('restored') is True and lease.get('restore_blocked') is False and
         lease.get('finished_epoch'), 'completed original service restoration required')
    frozen = source / 'final-service-record'
    frozen.mkdir(mode=0o700, exist_ok=False)
    records = []
    for name in ('lease.json', 'summary.json', *LIVE_NAMES):
        path = window / name
        before = core.signature(core.regular_stat(path))
        target = frozen / name
        # Live restored logs are bounded prefixes at this post-restoration instant.
        # An append after the measured boundary cannot force an unbounded read.
        with path.open('rb') as original, target.open('xb') as copy:
            need(core.signature(os.fstat(original.fileno())) == before,
                 'restoration source changed before open: ' + name)
            remaining = before['bytes']
            while remaining:
                block = original.read(min(1048576, remaining))
                need(block, 'restoration source shortened during snapshot: ' + name)
                copy.write(block)
                remaining -= len(block)
            copy.flush()
            os.fsync(copy.fileno())
            after = core.signature(os.fstat(original.fileno()))
        os.chmod(target, 0o600)
        if name in LIVE_NAMES:
            need(all(before[k] == after[k] for k in ('device', 'inode', 'mode')) and
                 after['bytes'] >= before['bytes'], 'restored log replaced or truncated')
        else:
            need(before == after and core.digest(path, before) == core.digest(target),
                 'final lease/summary changed during snapshot')
        records.append(dict(name=name, before=before, after=after,
                            frozen=core.descriptor(target),
                            bounded_live_log_prefix=name in LIVE_NAMES))
    write_new(frozen / 'capture.json', dict(kind='post_restoration_frozen_record',
        captured_epoch=time.time(), files=records, full_environment_is_private=True,
        live_log_prefix_boundary='Each restored log retains exactly its initial fstat byte count.'))
    return lease


def archive_spec(source, lease):
    window = source / 'service-window-v1'
    excluded = {window / name for name in LIVE_NAMES}
    roots, singles = [], []
    # Enumerating the two containers avoids teaching the frozen core an exclusion
    # language. All other subtrees, including failed native-build-v1, stay whole.
    for parent in (source, window):
        for path in sorted(parent.iterdir()):
            if path == window or path in excluded:
                continue
            need(not path.is_symlink(), 'symlink source refused: ' + str(path))
            (roots if path.is_dir() else singles).append(str(path))
    return dict(schema_version=1, kind='private_completed_register_only_integration',
        origin='M2', archive_stem=STEM, required_roots=roots, required_files=singles,
        execution_authorized_by_root_after_all_GPU_and_service_restore=True,
        native_activity_finished_and_no_more_runs_planned=True,
        service_restoration_record_frozen=True,
        forbidden_model_paths=[
            '/Users/fangqingyuan/hf-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf',
            '/Users/fangqingyuan/.lmstudio/models/antirez/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf'],
        canonical_paths=[path for path, item in lease['identities'].items() if item['exists']],
        excluded_active_files=[str(p) for p in sorted(excluded)],
        archive_is_private_and_contains_full_environment=True,
        archive_is_not_a_speed_or_success_claim=True)


def execute(source, out, core):
    need(not out.exists() and not out.is_symlink(), 'new archive directory required')
    need(not out.resolve().is_relative_to(source.resolve()), 'archive must be outside source')
    need((source / 'native-build-v1').is_dir(), 'failed native-build-v1 evidence required')
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    lease = freeze_restoration(source, core)
    spec = archive_spec(source, lease)
    spec_path = out / 'private-archive-spec.json'
    write_new(spec_path, spec)
    base_collect = core.collect

    def bounded_collect(settings, directory):
        files, dirs = base_collect(settings, directory)
        excluded = set(settings['excluded_active_files'])
        actual = set()
        for base, children, names in os.walk(source, followlinks=False):
            need(all(not (Path(base) / name).is_symlink() for name in children),
                 'symlink directory refused')
            actual.update(str(Path(base) / name) for name in names
                          if str(Path(base) / name) not in excluded)
        need(actual == {row['original_path'] for row in files},
             'single-root file coverage changed or incomplete')
        # Preserve the two explicitly split containers, including private modes.
        for parent in (source, source / 'service-window-v1'):
            value = parent.lstat()
            need(stat.S_ISDIR(value.st_mode), 'source container is not a real directory')
            dirs.append(dict(original_path=str(parent),
                archive_member='origin/M2' + parent.as_posix(), stat=core.signature(value)))
        return files, sorted(dirs, key=lambda row: row['archive_member'])

    core.collect = bounded_collect
    try:
        report = core.execute(spec, spec_path, out)
    finally:
        core.collect = base_collect
    # Inventory/spec/capture may describe private paths. Only this strict
    # allowlist receipt is intended to leave the private Benchmarks directory.
    checksums = {path.name: core.digest(path) for path in sorted(out.iterdir()) if path.is_file()}
    with (out / 'SHA256SUMS').open('x') as stream:
        for name, digest in checksums.items():
            stream.write(digest + '  ' + name + '\n')
    receipt = dict(kind='register_only_integration_private_archive_receipt',
        created_epoch=time.time(), archive=report['archive'], inventory=report['inventory'],
        checksums=core.descriptor(out / 'SHA256SUMS'),
        logical_file_count=report['logical_file_count'], logical_bytes=report['logical_bytes'],
        original_source_closure_stat_SHA_unchanged=report['original_source_closure_stat_SHA_unchanged'],
        all_logical_bytes_SHA_and_modes_verified=report['all_logical_bytes_SHA_and_modes_verified'],
        complete_gzip_CRC_and_length_verified=report['complete_gzip_CRC_and_length_verified'],
        canonical_unchanged=report['canonical_unchanged'],
        failed_native_build_v1_preserved=(source / 'native-build-v1').is_dir(),
        frozen_restoration_records_preserved=True, live_restored_logs_excluded=True,
        private_directory_mode='0700', private_artifact_mode='0600',
        full_environment_exported_in_public_receipt=False, GPU_work_submitted=False,
        services_modified=False, original_sources_deleted=False)
    write_new(out / 'public-receipt.json', receipt)
    need(stat.S_IMODE(out.stat().st_mode) == 0o700 and
         all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in out.iterdir()),
         'private archive artifact modes differ')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute-after-root-confirmed-native-http-ended-and-service-restored',
                        action='store_true')
    args = parser.parse_args()
    need(args.out.is_absolute() and args.out.is_relative_to(BENCHMARKS),
         'archive output must be a new directory under /Users/fangqingyuan/Benchmarks')
    if not args.execute_after_root_confirmed_native_http_ended_and_service_restored:
        print(json.dumps(dict(execution=False, source=str(SOURCE), output=str(args.out),
            instruction='Wait for root confirmation; no remote raw has been read by this plan.')))
        return
    receipt = execute(SOURCE, args.out, load_core(args.core))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
