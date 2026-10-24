#!/usr/bin/env python3
"""Verify and materialize a lossless MoE evidence archive using local hardlinks.

Only CPU file operations are performed. The existing archive verifier checks
every original member; this tool preserves its exact logical names and records
the local inode/SHA identity of duplicate payloads. No model payload is fetched.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def verifier():
    source = Path(__file__).with_name('archive_attention_multihead_evidence.py')
    spec = importlib.util.spec_from_file_location('existing_lossless_verifier', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def destination(root, name):
    path = PurePosixPath(name)
    need(not path.is_absolute() and str(path) == name and
         all(part not in ('', '.', '..') for part in path.parts), 'Unsafe logical archive path')
    target = root.joinpath(*path.parts)
    need(target.is_relative_to(root), 'Logical path escaped destination')
    return target


def materialize(archive_dir, output):
    archive_dir = archive_dir.resolve(strict=True)
    output = output.resolve()
    need(not output.exists() and not output.is_relative_to(archive_dir), 'New external output directory required')
    manifest = json.loads((archive_dir / 'raw-files.json').read_text())
    records = manifest['files']
    need(all(r['storage'] in ('tar_regular', 'tar_hardlink') for r in records),
         'This MoE materializer requires all raw files as physical archive members')
    existing = verifier()
    verification = existing.verify_archive(archive_dir / 'raw.tar.gz', manifest)
    output.mkdir(parents=True, exist_ok=False)
    by_path = {r['path']: r for r in records}
    emitted = set()
    with tarfile.open(archive_dir / 'raw.tar.gz', 'r|gz') as archive:
        for member in archive:
            name = member.name
            need(name in by_path and name not in emitted, 'Unexpected or repeated original member')
            record = by_path[name]
            target = destination(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.islnk():
                need(record['storage'] == 'tar_hardlink' and member.linkname == record['linkname'] and
                     member.linkname in emitted, 'Invalid forward or mismatched hardlink')
                os.link(destination(output, member.linkname), target)
            else:
                need(member.isfile() and record['storage'] == 'tar_regular' and member.size == record['bytes'],
                     'Only exact original regular files are materialized')
                stream = archive.extractfile(member)
                need(stream is not None, 'Missing original payload stream')
                digest = hashlib.sha256()
                count = 0
                with target.open('xb') as sink:
                    for block in iter(lambda: stream.read(1048576), b''):
                        digest.update(block)
                        count += len(block)
                        sink.write(block)
                need(count == record['bytes'] and digest.hexdigest() == record['sha256'],
                     'Materialized payload SHA differs from original')
                os.chmod(target, member.mode)
            emitted.add(name)
    need(emitted == set(by_path), 'Incomplete logical output coverage')
    inodes = {}
    content = {}
    mappings = []
    allocated = 0
    for record in records:
        target = destination(output, record['path'])
        need(target.is_file() and not target.is_symlink(), 'Original file not materialized')
        st = target.stat()
        key = (st.st_dev, st.st_ino)
        if key not in inodes:
            inodes[key] = sha(target)
            allocated += st.st_blocks * 512
        need(st.st_size == record['bytes'] and inodes[key] == record['sha256'],
             'Final local original bytes differ')
        identity = (record['bytes'], record['sha256'])
        if identity in content:
            need(content[identity] == key, 'Duplicate original bytes did not retain hardlink deduplication')
        else:
            content[identity] = key
        mappings.append(dict(path=record['path'], source_path=record['source_path'], bytes=st.st_size,
                             sha256=record['sha256'], device=st.st_dev, inode=st.st_ino))
    return dict(schema_version=1, scope='CPU_lossless_MoE_evidence_materialization',
                evidence_valid=True, source_data_modified=False, gpu_or_model_executed=False,
                archive_sha256=sha(archive_dir / 'raw.tar.gz'),
                raw_files_sha256=sha(archive_dir / 'raw-files.json'),
                archive_verification=verification, original_logical_files_verified=len(records),
                logical_bytes=sum(r['bytes'] for r in records), local_unique_inodes=len(inodes),
                local_allocated_bytes=allocated, deduplicated_logical_files=len(records)-len(inodes),
                weight_payload_downloaded=False, original_to_local_file_identity=mappings)


def self_test():
    existing = verifier()
    checks = []
    with tempfile.TemporaryDirectory(prefix='moe-evidence-materializer-CPU-') as tmp:
        base = Path(tmp)
        first, second = base/'first', base/'second'
        first.mkdir(); second.mkdir()
        (first/'a.bin').write_bytes(b'original CPU fixture bytes')
        (second/'copy.bin').write_bytes((first/'a.bin').read_bytes())
        (second/'empty.bin').write_bytes(b'')
        existing.create(existing.parse_windows(['first='+str(first),'second='+str(second)]), base/'archive', True)
        result = materialize(base/'archive', base/'local')
        need(result['original_logical_files_verified'] == 3 and result['local_unique_inodes'] == 2,
             'Positive hardlink control failed')
        checks.append('all_original_members_SHA_and_local_hardlink_positive')
        for name in ('../escape', '/absolute', 'raw/../escape'):
            try:
                destination(base/'local', name)
            except ValueError:
                checks.append('reject_path_'+name)
            else:
                raise AssertionError('Unsafe path accepted')
        source = base/'archive/raw-files.json'
        original = source.read_bytes()
        changed = json.loads(original)
        changed['files'][0]['sha256'] = '0'*64
        source.write_text(json.dumps(changed))
        try:
            materialize(base/'archive', base/'invalid')
        except ValueError:
            checks.append('reject_changed_original_payload_digest')
        else:
            raise AssertionError('Changed SHA accepted')
        source.write_bytes(original)
    return dict(schema_version=1, all_controls_pass=True, checks=checks,
                gpu_or_model_executed=False, production_evidence=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        need(args.archive_dir is None and args.out_dir is None, 'Separate CPU self-test mode required')
        result = self_test()
    else:
        need(args.archive_dir is not None and args.out_dir is not None, 'Archive and new output required')
        result = materialize(args.archive_dir, args.out_dir)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('original_to_local_file_identity','archive_verification')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
