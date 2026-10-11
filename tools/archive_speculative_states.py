#!/usr/bin/env python3
"""Store repeated fixed-state payloads once, with a reversible SHA256 map.

Reports/reviewer files stay separate. Unpack restores ordinary files (no links)
for the independent strict reviewer. This is archival, not a correctness gate.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pack(root, archive, directories):
    files, blobs = {}, {}
    for directory in directories:
        folder = root / directory
        if folder.is_symlink() or not folder.is_dir() or '/' in directory or directory in ('.', '..'):
            raise ValueError('Invalid raw directory')
        for path in sorted(folder.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError('Raw entries must be ordinary files')
            data = path.read_bytes()
            identity = digest(data)
            files[path.relative_to(root).as_posix()] = dict(sha256=identity, bytes=len(data))
            if identity in blobs and blobs[identity] != data:
                raise ValueError('SHA256 collision')
            blobs[identity] = data
    manifest = dict(schema_version=1, raw_files=len(files), unique_payloads=len(blobs), files=files)
    with tarfile.open(archive, 'x:gz') as tar:
        for name, data in [('manifest.json', (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode())] + [
            ('blobs/'+identity, data) for identity, data in sorted(blobs.items())]:
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o600
            tar.addfile(item, io.BytesIO(data))
    return manifest


def unpack(archive, root):
    if root.exists():
        raise ValueError('Destination must be new')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len({item.name for item in members}) != len(members) or any(
            not item.isfile() or item.size > 64 * 1024 * 1024 for item in members):
            raise ValueError('Invalid/duplicate archive entry')
        manifest = json.loads(tar.extractfile('manifest.json').read())
        if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
            raise ValueError('Invalid archive manifest')
        identities = {item['sha256'] for item in manifest['files'].values()}
        if set(item.name for item in members) != {'manifest.json'} | {'blobs/'+value for value in identities}:
            raise ValueError('Unreferenced/missing archive entry')
        if manifest['raw_files'] != len(manifest['files']) or manifest['unique_payloads'] != len(identities):
            raise ValueError('Incomplete manifest')
        blobs = {}
        for identity in identities:
            if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
                raise ValueError('Invalid SHA256')
            data = tar.extractfile('blobs/'+identity).read()
            if digest(data) != identity:
                raise ValueError('Original raw SHA256 differs')
            blobs[identity] = data
        for name, item in manifest['files'].items():
            path = PurePosixPath(name)
            if path.is_absolute() or len(path.parts) != 2 or any(p in ('.', '..') for p in path.parts):
                raise ValueError('Invalid raw path')
            if type(item['bytes']) is not int or len(blobs[item['sha256']]) != item['bytes']:
                raise ValueError('Original raw size differs')
        root.mkdir()
        for name, item in manifest['files'].items():
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(blobs[item['sha256']])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pack', 'unpack'))
    parser.add_argument('archive', type=Path)
    parser.add_argument('root', type=Path)
    parser.add_argument('directories', nargs='*')
    args = parser.parse_args()
    result = pack(args.root, args.archive, args.directories) if args.mode == 'pack' else unpack(args.archive, args.root)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, sort_keys=True))


if __name__ == '__main__':
    main()
