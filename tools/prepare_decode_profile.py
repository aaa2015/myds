#!/usr/bin/env python3
"""Instrument an isolated source copy; never patch or run production.

Only wraps existing CB operations and host work. No GPU dispatch, encoder
boundary, barrier, commit or wait is added. Metal System Trace is optional and
is attached by bench_decode_profile after a steady-decode marker.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('src/myds_metal.m', 'src/myds_v41.c', 'src/myds_engine.c',
         'include/myds_metal.h', 'tools/bench_phase.c')


def replace(text, old, new, count=1):
    actual = text.count(old)
    if actual != count:
        raise ValueError(f'expected {count} anchors, got {actual}: {old[:100]!r}')
    return text.replace(old, new)


def replace_pattern(text, pattern, replacement, label):
    """Allow formatting changes while still requiring one semantic anchor."""
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f'expected 1 {label} anchor, got {len(matches)}')
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)


def prepare(tree):
    tree = tree.resolve()
    roots = (Path('/tmp').resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(p != tree and p in tree.parents for p in roots):
        raise ValueError('diagnostic tree must be a separate copy below a temporary root')
    if tree == ROOT or ROOT in tree.parents or tree in ROOT.parents:
        raise ValueError('diagnostic tree must be outside this checkout')
    if (tree / 'decode-diag-manifest.json').exists():
        raise ValueError('diagnostic already prepared')
    originals = {n: (tree / n).read_text() for n in FILES}
    changed = dict(originals)
    m = changed['src/myds_metal.m']
    counts = {name: m.count(anchor) for name, anchor in (
        ('created', '[g_queue commandBuffer]'), ('commit', '[cb commit];'),
        ('wait', '[cb waitUntilCompleted];'))}
    unretained_created = m.count('[g_queue commandBufferWithUnretainedReferences]')
    counts['created'] += unretained_created
    if not all(counts.values()):
        raise ValueError('missing existing CB operation anchors')
    m = m.replace('[g_queue commandBuffer]', 'rp_created([g_queue commandBuffer])')
    m = m.replace('[g_queue commandBufferWithUnretainedReferences]',
                  'rp_created([g_queue commandBufferWithUnretainedReferences])')
    m = m.replace('[cb commit];', 'rp_commit(cb);')
    m = m.replace('[cb waitUntilCompleted];', 'rp_wait(cb);')
    m = replace_pattern(m, r'^static[ \t]+int[ \t]+g_auto_barrier[ \t]*=[ \t]*1[ \t]*;[ \t]*$',
                        lambda match: match.group(0) + '\n#include "myds_decode_profile.inc"',
                        'g_auto_barrier declaration')
    m = replace(m, '    BOOL signaled = [g_shared_event waitUntilSignaledValue:val timeoutMS:60000];',
                '    const double rp_event_start = myds_remaining_tick();\n'
                '    BOOL signaled = [g_shared_event waitUntilSignaledValue:val timeoutMS:60000];\n'
                '    if (rp_event_start) rp_event_wait_s += rp_now() - rp_event_start;')
    m = replace(m, '    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];',
                '    rp_dispatch(name, threadgroups, threadsPerThreadgroup, false);\n'
                '    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];')
    m = replace(m, '    [enc dispatchThreads:threads threadsPerThreadgroup:threadsPerThreadgroup];',
                '    rp_dispatch(name, threads, threadsPerThreadgroup, true);\n'
                '    [enc dispatchThreads:threads threadsPerThreadgroup:threadsPerThreadgroup];')
    changed['src/myds_metal.m'] = m
    h = changed['include/myds_metal.h']
    h = replace(h, 'int  myds_metal_init(const char *shader_dir);',
                '/* Only present in an isolated diagnostic source copy. */\n'
                'void myds_remaining_request_begin(uint32_t run);\n'
                'void myds_remaining_request_end(void);\n'
                'double myds_remaining_tick(void);\n'
                'void myds_remaining_host(const char *name, double start, uint64_t bytes);\n'
                'void myds_remaining_context(const char *phase, uint32_t pos, uint32_t layer,\n'
                '    uint32_t rows, uint32_t kv_only, uint32_t moe_rows);\n'
                'void myds_decode_diag_stage(uint32_t pos, uint32_t layer, const char *stage);\n'
                'void myds_decode_diag_callback(uint32_t count);\n'
                'int  myds_metal_init(const char *shader_dir);')
    changed['include/myds_metal.h'] = h
    c = changed['src/myds_v41.c']
    implementation = 'static int32_t v41_forward_impl('
    if c.count(implementation) == 1:
        begin = c.index(implementation)
        end = c.index('\nint32_t myds_v41_forward(', begin)
    elif implementation in c:
        raise ValueError('duplicate v41_forward_impl anchor')
    else:
        # Older source kept the scalar implementation in the public entry.
        begin = c.index('int32_t myds_v41_forward(')
        end = c.index('\nint myds_v41_batch2_projection_probe(', begin)
    f = c[begin:end]
    f = replace(f, '    CHECK(myds_engram_prepare_token(g->engram,token,pos));',
                '    myds_remaining_context(need_logits ? "decode" : "prefill_scalar", pos, 0, 1, 0, 1);\n'
                '    double rp_host_start = myds_remaining_tick();\n'
                '    CHECK(myds_engram_prepare_token(g->engram,token,pos));\n'
                '    myds_remaining_host("engram_prepare", rp_host_start, 0);\n'
                '    rp_host_start = myds_remaining_tick();')
    f = replace(f, '    float *res=myds_metal_buffer_data(g->residual);',
                '    myds_remaining_host("engram_copy", rp_host_start, 2u*MYDS_ENGRAM_INPUT_DIM*4u);\n'
                '    rp_host_start = myds_remaining_tick();\n'
                '    float *res=myds_metal_buffer_data(g->residual);')
    f = replace_pattern(f,
        r'^[ \t]+(?:static int fuse_hc_norm_enabled = -1;|const bool fuse_hc_norm_enabled = v41_fuse_hc_sum_rmsnorm_enabled\(\);)[ \t]*$',
        lambda match: '    myds_remaining_host("embedding_expand", rp_host_start, 4u*V41_DIM*4u);\n' + match.group(0),
        'scalar HC norm mode declaration')
    f, stages = re.subn(r'(?m)^(\s*)phase="([^"]+)";',
                       lambda x: x.group(0) + '\n' + x[1] +
                       f'myds_decode_diag_stage(pos, il, "{x[2]}");', f)
    if stages < 8:
        raise ValueError('decode phase anchors changed')
    c = c[:begin] + f + c[end:]
    changed['src/myds_v41.c'] = c
    e = changed['src/myds_engine.c']
    e = replace(e, 'int32_t myds_sample_next_token(\n', 'static int32_t rp_sample_next_token(\n')
    e += '''\n/* Diagnostic host wrapper; original sampler is otherwise unchanged. */
int32_t myds_sample_next_token(const float *logits, uint32_t vocab_size,
    const myds_config_t *cfg, myds_context_t *ctx,
    const int32_t *prev_tokens, size_t n_prev_tokens) {
    const double start = myds_remaining_tick();
    const int32_t token = rp_sample_next_token(logits, vocab_size, cfg, ctx, prev_tokens, n_prev_tokens);
    myds_remaining_host("sampler", start, 0);
    return token;
}
'''
    changed['src/myds_engine.c'] = e
    b = changed['tools/bench_phase.c']
    b = replace(b, '    ++c->count;', '    ++c->count;\n    myds_decode_diag_callback(c->count);')
    b = replace(b, '        const double start = now_s();',
                '        myds_remaining_request_begin(run);\n        const double start = now_s();')
    b = replace(b, '        const double wall_s = now_s() - start;',
                '        const double wall_s = now_s() - start;\n        myds_remaining_request_end();')
    changed['tools/bench_phase.c'] = b
    inc = (ROOT / 'tools/decode_profile.inc').read_text()
    patch = ''.join(''.join(difflib.unified_diff(originals[n].splitlines(True), changed[n].splitlines(True),
                                               fromfile='a/'+n, tofile='b/'+n)) for n in FILES)
    # Validate every anchor before the first mutation.
    for n in FILES:
        (tree / n).write_text(changed[n])
    (tree / 'src/myds_decode_profile.inc').write_text(inc)
    (tree / 'decode-diag.patch').write_text(patch)
    digest = lambda data: hashlib.sha256(data.encode()).hexdigest()
    outputs = dict(changed, **{'src/myds_decode_profile.inc': inc})
    manifest = {'schema': 1, 'scope': 'isolated CB/host/dispatch-shape diagnostic; no per-kernel GPU timestamp',
                'input_sha256': {n: digest(v) for n, v in originals.items()},
                'output_sha256': {n: digest(v) for n, v in outputs.items()},
                'patch_sha256': digest(patch), 'operation_sites': counts,
                'unretained_creation_sites': unretained_created,
                'profile_env': 'MYDS_PROFILE_DECODE=cb', 'gpu_execution': False,
                'added_gpu_dispatches': 0, 'added_gpu_barriers': 0,
                'added_encoder_boundaries': 0, 'added_gpu_commits': 0, 'added_gpu_waits': 0}
    (tree / 'decode-diag-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.tree), indent=2))


if __name__ == '__main__':
    main()
