#!/usr/bin/env python3
"""Compile private stage-dump clones from a frozen attention source checkout.

Only reads the frozen checkout. The generated shaders are diagnostic observers;
their full outputs must match the uninstrumented providers before stage values
can be used to explain the original failure. Never submits GPU work.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

REC, STRIDE = 1124, 20 * 1124 + 516


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(source, marker):
    at = source.index(marker)
    start = source.rfind('template<', 0, at)
    require(start >= 0, 'Missing template declaration')
    opening = source.index('{', at)
    depth = 1
    end = opening + 1
    while depth:
        if source[end] == '{':
            depth += 1
        elif source[end] == '}':
            depth -= 1
        end += 1
    return source[start:end]


def once(text, old, new):
    require(text.count(old) == 1, 'Diagnostic insertion anchor is not unique: ' + old[:70])
    return text.replace(old, new, 1)


def last(text, old, new):
    at = text.rfind(old)
    require(at >= 0, 'Missing sink diagnostic anchor')
    return text[:at] + text[at:].replace(old, new, 1)


def vector_store(field, expression, tail=False):
    base = f'(uint)iq2*{STRIDE}u + ' + ('22480u' if tail else f'(uint)(ic/32)*{REC}u + {field}u')
    return ('\n            ((device float4 *)((device float *)debug + ' + base + '))[ii*32 + tiisg] = ' + expression + ';\n')


def observer(body, candidate):
    old_name = 'kernel_myds_prefill_attention_multihead' if candidate else 'kernel_flash_attn_ext_vec'
    new_name = 'probe_mh_candidate_debug_impl' if candidate else 'probe_mh_reference_debug_impl'
    body = once(body, 'kernel void ' + old_name + '(', 'kernel void ' + new_name + '(')
    dst = 'device char * dst [[buffer(7)]],' if candidate else 'device       char * dst,'
    body = once(body, dst, dst + '\n        device char * debug [[buffer(8)]],')
    score = ('ss[tx] = fma(mqk[tx], args.scale, (float) sm[tx]);' if candidate else
             'ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);')
    base = f'((device float *)debug + (uint)iq2*{STRIDE}u + (uint)(ic/32)*{REC}u)'
    body = once(body, score, f'{base}[tiisg] = mqk[tx];\n                    ' + score + f'\n                    {base}[32u + tiisg] = ss[tiisg];')
    soft = 'ss[tiisg] = vs;'
    dump_soft = f'''{soft}
                {base}[64u + tiisg] = vs;
                const float debug_vs_sum = simd_sum(vs);
                if (tiisg == 0) {{
                    {base}[96] = M; {base}[97] = S;
                    {base}[98] = ms; {base}[99] = debug_vs_sum;
                }}'''
    body = once(body, soft, dump_soft)
    if candidate:
        add = 'so4[ii*NL] += lo[ii];'
        body = once(body, add, vector_store(100, 'lo[ii]') + add + vector_store(612, 'so4[ii*NL]'))
    else:
        merge = 'FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {\n                    if (NE > 1) {'
        body = once(body, merge, 'FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {' + vector_store(100, 'lo[ii]') + '\n                }\n\n                ' + merge)
        body = once(body, 'so4[ii*NL] += lo[ii];', 'so4[ii*NL] += lo[ii];' + vector_store(612, 'so4[ii*NL]'))
    update = 'S = S*ms + simd_sum(vs);'
    tail = f'((device float *)debug + (uint)iq2*{STRIDE}u + 22480u)'
    body = last(body, update, update + f'''
        const float debug_sink_vs_sum = simd_sum(vs);
        if (tiisg == 0) {{
            {tail}[512] = M; {tail}[513] = S;
            {tail}[514] = ms; {tail}[515] = debug_sink_vs_sum;
        }}''')
    body = last(body, 'so4[ii*NL] *= ms;', 'so4[ii*NL] *= ms;' + vector_store(0, 'so4[ii*NL]', tail=True))
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--emit-only', action='store_true')
    args = parser.parse_args()
    root, out = args.source_root.resolve(strict=True), args.out_dir.resolve()
    require(not out.exists() and out != root and not out.is_relative_to(root), 'New private output outside frozen source required')
    out.mkdir(parents=True)
    flash = (root / 'metal/flash_attn.metal').read_text()
    multi = (root / 'metal/myds_prefill_attention_multihead.metal').read_text()
    native = function(flash, 'kernel void kernel_flash_attn_ext_vec(')
    candidate = function(multi, 'kernel void kernel_myds_prefill_attention_multihead(')
    reference_debug, candidate_debug = observer(native, False), observer(candidate, True)
    types = ('half4, half4, half4, float, float, float4, float4, '
             'half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 512, 512, 1, 1, 32, true')
    wrappers = f'''
typedef decltype(probe_mh_reference_debug_impl<{types}>) probe_mh_ref_debug_t;
template [[host_name("probe_mh_ref_debug")]]
kernel probe_mh_ref_debug_t probe_mh_reference_debug_impl<{types}>;
typedef decltype(probe_mh_candidate_debug_impl<2>) probe_mh_h2_debug_t;
template [[host_name("probe_mh_h2_debug")]]
kernel probe_mh_h2_debug_t probe_mh_candidate_debug_impl<2>;
'''
    debug = '// Private stage observer; never a performance kernel.\n' + reference_debug + '\n' + candidate_debug + wrappers
    (out / 'probe.metal').write_text(debug)
    spec = importlib.util.spec_from_file_location('frozen_build', root / 'tools/build_metallib.py')
    build = importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    combined = build.PREAMBLE + '\n' + ''.join(
        f'\n#line 1 "{name}"\n' + (root / 'metal' / name).read_text() for name in build.SHADER_FILES)
    combined += build.COPY_KERNELS + '\n#line 1 "private-multihead-diagnostic.metal"\n' + debug
    (out / 'all.metal').write_text(combined)
    identity = dict(schema_version=1, gpu_work_submitted=False, source_root=str(root),
                    observer_changes_codegen_possible=True, required_full_output_observer_gates=True,
                    native_template_sha256=hashlib.sha256(native.encode()).hexdigest(),
                    candidate_template_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                    inputs={str(p.relative_to(root)): sha(p) for p in
                            [root / 'tools/build_metallib.py', *(root / 'metal' / n for n in build.SHADER_FILES)]},
                    trace_layout=dict(heads=64, blocks=20, block_floats=REC, head_floats=STRIDE,
                                      offsets=dict(mqk=0, score=32, softmax=64, stats=96, lo=100, acc=612),
                                      stats=['M', 'S', 'ms', 'sum_vs'], sink_acc_offset=22480,
                                      sink_stats_offset=22992, total_bytes=64 * STRIDE * 4),
                    probe_sha256=sha(out / 'probe.metal'), combined_sha256=sha(out / 'all.metal'))
    if not args.emit_only:
        commands = [
            ['xcrun', '-sdk', 'macosx', 'metal', '-mmacosx-version-min=26.0', '-O3', '-ffast-math',
             '-DDS4_METAL_HC_STABLE=1', '-DDS4_METAL_NORM_RSQRT_DISABLE=1', '-c', str(out / 'all.metal'), '-o', str(out / 'debug.air')],
            ['xcrun', '-sdk', 'macosx', 'metallib', str(out / 'debug.air'), '-o', str(out / 'debug.metallib')],
        ]
        env = dict(os.environ);env.setdefault('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer')
        with (out / 'build.log').open('w') as log:
            for command in commands:
                log.write(json.dumps(command) + '\n');log.flush();subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
        identity['commands'] = commands;identity['debug_metallib_sha256'] = sha(out / 'debug.metallib')
    (out / 'source-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(dict(built=not args.emit_only, out_dir=str(out), gpu_work_submitted=False)))


if __name__ == '__main__':
    main()
