#!/usr/bin/env python3
"""Tiny synthetic CPU-only diagnostic archive/slice rejection controls."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from test_bench_attention_multihead_host import FlashArgs


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else 'bin/bench_attention_multihead_diagnostic').resolve()
    args = FlashArgs(ne01=2, ne02=64, ne03=1, nb01=131072, nb02=2048, nb03=131072,
                     ne11=640, ne_12_2=1, ne_12_3=1, ns10=512, nb11=1024, nb12=655360, nb13=655360,
                     ns20=512, nb21=1024, nb22=655360, nb23=655360, ne31=2, ne32=1, ne33=1,
                     nb31=2048, nb32=4096, nb33=4096, ne1=64, ne2=2, ne3=1,
                     scale=1 / math.sqrt(512), nb_kvq=655360)
    with tempfile.TemporaryDirectory(prefix='myds-multihead-diagnostic-host-') as temp:
        root = Path(temp)
        component = root / 'component'
        component.mkdir()
        blobs = {
            'capture-case-000-args200': bytes(args),
            'capture-case-000-q.f32': struct.pack('<65536f', *([0.5] * 65536)),
            'capture-case-000-kv-unique.f16': struct.pack('<512H', *([0x3800] * 512)),
            'capture-case-000-kv-rows.u32': bytes(2 * 640 * 4),
            'capture-case-000-mask-compact.f16': struct.pack('<H', 0xfc00) + bytes(2 * 1280 - 2),
            'capture-case-000-sinks.f32': bytes(256),
            'parity-case-000-provider-0-output.f32': struct.pack('<65536f', *([0.25] * 65536)),
            'parity-case-000-provider-2-output.f32': struct.pack('<65536f', *([0.125] * 65536)),
        }

        def archive():
            records = []
            for name, data in blobs.items():
                (component / name).write_bytes(data)
                records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            manifest = dict(status='failed', files=records)
            (component / 'artifact-manifest.json').write_text(json.dumps(manifest))
            return manifest

        for name in ('production', 'probe', 'debug'):
            (root / (name + '.metallib')).write_bytes(b'CPU-control-never-a-Metal-library')

        def run(label, expected):
            out = root / label
            result = subprocess.run([str(binary), '--input-dir', str(component), '--case', '0',
                '--production-metallib', str(root / 'production.metallib'), '--probe-metallib', str(root / 'probe.metallib'),
                '--debug-metallib', str(root / 'debug.metallib'), '--out-dir', str(out), '--host-only'], capture_output=True, text=True)
            assert (result.returncode == 0) == expected, (label, result.stdout, result.stderr)
            manifest = json.loads((out / 'artifact-manifest.json').read_text())
            assert manifest['status'] == ('host_only' if expected else 'failed')
            if expected:
                complete = json.loads(result.stdout.splitlines()[-1])
                assert complete['host_only'] and not complete['metal_device_created']
                sliced = FlashArgs.from_buffer_copy((out / 'slice-args200').read_bytes())
                assert sliced.ne01 == sliced.ne31 == sliced.ne2 == 1
                assert sliced.nb32 == sliced.nb33 == 2048
                assert (out / 'slice-q.f32').read_bytes() == blobs['capture-case-000-q.f32'][:131072]
                mask = (out / 'slice-mask-padded.f16').read_bytes()
                assert mask[:1280] == blobs['capture-case-000-mask-compact.f16'][:1280]
                assert mask[1280:] == struct.pack('<H', 0xfc00) * ((2048 - 1280) // 2)
                assert (out / 'slice-kv.f16').read_bytes() == blobs['capture-case-000-kv-unique.f16'] * 640
            print(json.dumps(dict(control=label, passed=True, expected_success=expected, gpu_executed=False)))

        archive()
        run('valid-real-layout-slice', True)
        manifest = archive()
        manifest['files'][0]['sha256'] = '0' * 64
        (component / 'artifact-manifest.json').write_text(json.dumps(manifest))
        run('source-sha-rejected', False)
        archive()
        blobs['capture-case-000-kv-rows.u32'] = struct.pack('<I', 1) + bytes(2 * 640 * 4 - 4)
        archive()
        run('dictionary-id-rejected', False)
        blobs['capture-case-000-kv-rows.u32'] = bytes(2 * 640 * 4)
        blobs['capture-case-000-mask-compact.f16'] = struct.pack('<H', 0xfbff) + bytes(2 * 1280 - 2)
        archive()
        run('wrong-source-mask-rejected', False)


if __name__ == '__main__':
    main()
