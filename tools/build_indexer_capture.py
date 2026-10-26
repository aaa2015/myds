#!/usr/bin/env python3
"""Build the compile-time real-prefill observer in a new immutable source snapshot.

Compiles and links only: no model, Metal device, inference or service operations.
The production metallib is supplied separately to the capture process. No probe
shader or changed production registry is linked by this builder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    root, out = a.source_root.resolve(strict=True), a.out_dir.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('New private output directory outside checkout required')
    # Freeze every compiler-visible source/header and capture include before
    # building. The manifest describes these exact dirty-tree bytes, not HEAD.
    inputs = [root / 'Makefile']
    for directory in ('src', 'include', 'third_party'):
        inputs += [f for f in (root / directory).rglob('*')
                   if f.is_file() and f.suffix in ('.c', '.h', '.inc', '.m')]
    for rel in ('tools/capture_indexer_k.c', 'tools/indexer_capture.inc',
                'tools/reference/trace_input.h', 'tools/build_indexer_capture.py'):
        inputs.append(root / rel)
    snapshot = out / 'source'
    identities = {}
    out.mkdir(parents=True)
    for f in sorted(set(inputs)):
        data = f.read_bytes()
        rel = f.relative_to(root)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        identities[str(rel)] = dict(bytes=len(data), sha256=sha(data))
    make = (snapshot / 'Makefile').read_text()
    section = make.split('SRCS_C = ', 1)[1].split('\nSRCS_M =', 1)[0]
    sources = re.findall(r'src/[A-Za-z0-9_]+\.c', section)
    sources = [f for f in sources if f != 'src/myds_server.c']
    if 'src/myds_v41.c' not in sources or 'src/myds_v41_gpu.c' not in sources:
        raise ValueError('Makefile source registry')
    env = dict(os.environ)
    env.setdefault('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer')
    obj = out / 'objects'
    obj.mkdir()
    common = ['xcrun', 'clang', '-O3', '-Wall', '-Wextra', '-mmacosx-version-min=26.0',
              '-I' + str(snapshot / 'include'), '-I' + str(snapshot / 'third_party')]
    commands, objects = [], []
    for rel in sources + ['tools/capture_indexer_k.c']:
        target = obj / (Path(rel).stem + '.o')
        macro = ['-DMYDS_INDEXER_CAPTURE'] if rel in ('src/myds_v41.c', 'src/myds_v41_gpu.c') else []
        commands.append(common + ['-std=c11'] + macro + ['-c', str(snapshot / rel), '-o', str(target)])
        objects.append(str(target))
    target = obj / 'myds_metal.o'
    commands.append(common + ['-fobjc-arc', '-Wno-deprecated-declarations', '-c',
                              str(snapshot / 'src/myds_metal.m'), '-o', str(target)])
    objects.append(str(target))
    binary = out / 'capture_indexer_k'
    commands.append(['xcrun', 'clang', '-mmacosx-version-min=26.0', *objects,
                     '-framework', 'Foundation', '-framework', 'Metal', '-lm', '-lpthread', '-o', str(binary)])
    normal = []
    for rel in ('src/myds_v41.c', 'src/myds_v41_gpu.c'):
        target = out / (Path(rel).stem + '-normal.pp.c')
        commands.append(common + ['-std=c11', '-E', '-P', str(snapshot / rel), '-o', str(target)])
        normal.append(target)
    head_query = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    identity = dict(schema_version=1, kind='indexer_capture_build', source_root=str(root),
                    head=head_query.stdout.strip() if head_query.returncode == 0 else None,
                    source_identity_kind='frozen_exact_file_bytes_not_HEAD_equivalence',
                    inputs=identities, commands=commands, compile_only_pass=False,
                    gpu_work_submitted=False, performance_from_capture=False, production_registered=False)
    manifest = out / 'build-manifest.json'
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    with (out / 'compile.log').open('w') as log:
        for cmd in commands:
            log.write(json.dumps(cmd) + '\n')
            log.flush()
            subprocess.run(cmd, env=env, stdout=log, stderr=log, check=True)
    for f in normal:
        data = f.read_bytes()
        if b'myds_indexer_capture' in data or b'indexer_capture_before' in data or b'indexer_capture_after' in data:
            raise ValueError('Observer survived normal preprocessing')
    symbols = subprocess.check_output(['nm', '-g', str(binary)], text=True)
    if '_myds_indexer_capture_begin' not in symbols or '_myds_indexer_capture_set_group' not in symbols:
        raise ValueError('Observer not linked')
    identity.update(compile_only_pass=True, binary=str(binary), binary_sha256=sha(binary.read_bytes()),
                    normal_preprocessor_observer_absent=True,
                    normal_preprocessed={f.name: sha(f.read_bytes()) for f in normal})
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(dict(build_manifest=str(manifest), binary=str(binary),
                         binary_sha256=identity['binary_sha256'], compile_only_pass=True,
                         normal_preprocessor_observer_absent=True, gpu_work_submitted=False), indent=2))


if __name__ == '__main__':
    main()
