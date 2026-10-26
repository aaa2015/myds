#!/usr/bin/env python3
"""Freeze and CPU-build the finite predecessor/observer diagnostic. No GPU run."""
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
    p.add_argument('--emit-only', action='store_true')
    a = p.parse_args()
    root, out = a.source_root.resolve(strict=True), a.out_dir.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('new private output outside source checkout required')
    out.mkdir(parents=True)
    snapshot = out / 'source'
    names = [f for directory in ('src', 'include', 'third_party') for f in (root / directory).rglob('*')
             if f.is_file() and f.suffix in ('.h', '.inc', '.m', '.c')]
    names += [root / n for n in ('tools/bench_indexer_stability.m', 'tools/bench_indexer_k_capture.m',
              'tools/bench_attention_multihead.m', 'tools/build_indexer_stability.py')]
    inputs = {}
    for f in sorted(set(names)):
        data = f.read_bytes(); rel = f.relative_to(root); target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        inputs[str(rel)] = dict(bytes=len(data), sha256=sha(data))
    frozen = (snapshot / 'tools/bench_indexer_k_capture.m').read_bytes()
    marker = b'int main(int argc,char **argv){@autoreleasepool {'
    if frozen.count(marker) != 1:
        raise ValueError('unique original replay main boundary required')
    helpers, removed = frozen.split(marker, 1)
    (snapshot / 'tools/indexer_replay_helpers.inc').write_bytes(helpers)
    commands, objects = [], []
    common = ['xcrun', 'clang', '-O3', '-Wall', '-Wextra', '-mmacosx-version-min=26.0',
              '-I' + str(snapshot / 'include'), '-I' + str(snapshot / 'third_party')]
    for rel in ('src/myds_v41_gpu.c', 'src/myds_gguf.c', 'src/myds_log.c', 'src/myds_configfile.c',
                'src/myds_metal.m', 'tools/bench_indexer_stability.m'):
        target = out / (Path(rel).stem + '.o')
        flags = ['-fobjc-arc', '-Wno-deprecated-declarations'] if rel.endswith('.m') else ['-std=c11']
        commands.append(common + flags + ['-c', str(snapshot / rel), '-o', str(target)])
        objects.append(str(target))
    binary = out / 'bench_indexer_stability'
    commands.append(['xcrun', 'clang', '-mmacosx-version-min=26.0', *objects,
                     '-framework', 'Foundation', '-framework', 'Metal', '-lm', '-lpthread', '-o', str(binary)])
    identity = dict(schema_version=1, kind='indexer_stability_diagnostic_build', inputs=inputs, commands=commands,
                    helpers_sha256=sha(helpers), helpers_bytes=len(helpers), removed_main_suffix_sha256=sha(marker + removed),
                    original_replay_file_unchanged=True, compile_only_pass=False, gpu_work_submitted=False,
                    production_registered=False, adoption_cohort=False, compilation_skipped=a.emit_only)
    manifest = out / 'build-manifest.json'
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    if not a.emit_only:
        env = dict(os.environ); env.setdefault('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer')
        with (out / 'compile.log').open('w') as log:
            for command in commands:
                log.write(json.dumps(command) + '\n'); log.flush()
                subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
        identity.update(compile_only_pass=True, binary=str(binary), binary_sha256=sha(binary.read_bytes()))
        manifest.write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(manifest), compile_only_pass=identity['compile_only_pass'],
                         gpu_work_submitted=False, compilation_skipped=a.emit_only)))


if __name__ == '__main__':
    main()
