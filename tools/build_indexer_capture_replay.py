#!/usr/bin/env python3
"""Freeze and CPU-compile real-indexer replay plus the actual production TopK API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


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
    out.mkdir(parents=True)
    snapshot = out / 'source'
    inputs = {}
    paths = [f for directory in ('src', 'include', 'third_party')
             for f in (root / directory).rglob('*')
             if f.is_file() and f.suffix in ('.h', '.inc', '.m', '.c')]
    paths += [root / name for name in ('tools/bench_indexer_k_capture.m', 'tools/bench_attention_multihead.m',
              'tools/build_indexer_capture_replay.py', 'tools/review_indexer_capture.py')]
    for f in sorted(set(paths)):
        data = f.read_bytes()
        rel = f.relative_to(root)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        inputs[str(rel)] = dict(bytes=len(data), sha256=sha(data))
    common = ['xcrun', 'clang', '-O3', '-Wall', '-Wextra', '-mmacosx-version-min=26.0',
              '-I' + str(snapshot / 'include'), '-I' + str(snapshot / 'third_party')]
    commands, objects = [], []
    for rel in ('src/myds_v41_gpu.c', 'src/myds_gguf.c', 'src/myds_log.c', 'src/myds_configfile.c',
                'src/myds_metal.m', 'tools/bench_indexer_k_capture.m'):
        target = out / (Path(rel).stem + '.o')
        flags = ['-fobjc-arc', '-Wno-deprecated-declarations'] if rel.endswith('.m') else ['-std=c11']
        commands.append(common + flags + ['-c', str(snapshot / rel), '-o', str(target)])
        objects.append(str(target))
    binary = out / 'bench_indexer_k_capture'
    commands.append(['xcrun', 'clang', '-mmacosx-version-min=26.0', *objects,
                     '-framework', 'Foundation', '-framework', 'Metal', '-lm', '-lpthread', '-o', str(binary)])
    env = dict(os.environ)
    env.setdefault('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer')
    identity = dict(schema_version=1, kind='real_indexer_replay_build', inputs=inputs, commands=commands,
                    compile_only_pass=False, gpu_work_submitted=False, production_registered=False,
                    topk_api='myds_v41_gpu_topk_causal', normal_runtime_object_observer_absent=True)
    manifest = out / 'build-manifest.json'
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    with (out / 'compile.log').open('w') as log:
        for command in commands:
            log.write(json.dumps(command) + '\n')
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
    identity.update(compile_only_pass=True, binary=str(binary), binary_sha256=sha(binary.read_bytes()))
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(manifest), binary=str(binary), binary_sha256=identity['binary_sha256'],
                         compile_only_pass=True, gpu_work_submitted=False), indent=2))


if __name__ == '__main__':
    main()
