#!/usr/bin/env python3
"""Summarize bin/bench_kprof JSONL into a per-category / per-kernel ms/token table.

    tools/kprof_report.py run.jsonl [--top 25] [--diff baseline.jsonl] [--json out.json]

Numbers are isolated GPU time per kernel (serialized encoders) minus the
calibrated fixed per-encoder bracket cost (floor_us), divided by the number of
decode tokens in the capture window. They split the token budget; they are not
the unarmed token latency (overlap is removed, inter-encoder gaps are reported
separately).
"""
import argparse
import json
import re
import sys
from collections import defaultdict

# First match wins. Patterns run against the lower-cased kernel name.
CATEGORIES = [
    ('lm_head',        r'lm_head|output_head'),
    ('moe_router',     r'route|router|argsort|topk_threshold'),
    ('moe_routed',     r'fused_moe|moe_down|moe_sum|moe_pair|moe_map|moe_q2|gemm_grouped|mul_mv_id|mul_mm_id|moe_'),
    ('shared_expert',  r'shared'),
    ('indexer',        r'index'),
    ('attn_core',      r'flash_attn|attention|sdpa|attn_vec|packed8'),
    ('attn_out',       r'attn_out'),
    ('hc_residual',    r'hc_|_hc|v41_hc'),
    ('engram',         r'engram'),
    ('kv_rope_comp',   r'kv|rope|compress|pool'),
    ('norm',           r'rms|norm'),
    ('dense_gemv',     r'gemv|mul_mv|q8_0|q4_k|q4_K|dense'),
    ('copy_elementwise', r'copy|cpy|add|swiglu|bf16|concat|set_rows|get_rows|repeat|unary|glu|scale|softmax'),
]


def categorize(name):
    low = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat.lower(), low):
            return cat
    return 'other'


def load(path):
    kernels, summary, run = [], None, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            r = json.loads(line)
            kind = r.get('kind')
            if kind == 'kprof_kernel':
                kernels.append(r)
            elif kind == 'kprof_summary':
                summary = r
            elif kind == 'kprof_run':
                run = r
    if not summary or not kernels:
        raise SystemExit(f'{path}: no kprof_kernel/kprof_summary records')
    return kernels, summary, run


def aggregate(kernels, tokens, floor_us):
    """net = bracket time minus the calibrated per-encoder floor (clamped at 0)."""
    cats = defaultdict(lambda: {'us': 0.0, 'raw_us': 0.0, 'calls': 0})
    rows = []
    for k in kernels:
        cat = categorize(k['name'])
        net = max(0.0, k['sum_us'] - k['calls'] * floor_us)
        cats[cat]['us'] += net
        cats[cat]['raw_us'] += k['sum_us']
        cats[cat]['calls'] += k['calls']
        rows.append({'category': cat, 'name': k['name'], 'grid': k['grid'], 'threads': k['threads'],
                     'ms_per_token': net / 1000.0 / tokens,
                     'raw_ms_per_token': k['sum_us'] / 1000.0 / tokens,
                     'calls_per_token': k['calls'] / tokens,
                     'avg_us': net / max(1, k['calls']),
                     'min_us': k['min_us'], 'max_us': k['max_us'],
                     'merged': k['merged_encoders']})
    rows.sort(key=lambda r: -r['ms_per_token'])
    cat_rows = [{'category': c, 'ms_per_token': v['us'] / 1000.0 / tokens,
                 'raw_ms_per_token': v['raw_us'] / 1000.0 / tokens,
                 'calls_per_token': v['calls'] / tokens,
                 'avg_us': v['us'] / max(1, v['calls'])} for c, v in cats.items()]
    cat_rows.sort(key=lambda r: -r['ms_per_token'])
    return cat_rows, rows


def report(path, top):
    kernels, s, run = load(path)
    tokens = s['tokens']
    floor_us = s.get('floor_us', 0.0)
    cat_rows, rows = aggregate(kernels, tokens, floor_us)
    net = sum(r['ms_per_token'] for r in cat_rows)
    raw = s['busy_ms'] / tokens
    print(f'== {path}')
    if run:
        print(f"prompt_tokens={run['prompt_tokens']} warm={run['warm_tokens']} window={tokens} max_seq={run['max_seq']}")
    print(f"encoders/token={s['encoders'] / tokens:.1f}  encoder floor={floor_us:.1f} us")
    print(f"NET kernel time={net:.3f} ms/token  (raw bracket={raw:.3f}, floor removed={raw - net:.3f})  "
          f"inter-encoder gap(<1ms)={s['gap_lt1ms_ms'] / tokens:.3f}  idle(>=1ms)={s['idle_ge1ms_ms'] / tokens:.3f}  "
          f"tick={s['gpu_tick_ns']:.4f}ns")
    if floor_us == 0:
        print('WARNING: floor_us=0; times include the fixed encoder bracket cost and over-state small kernels')
    warn = [f'{k}={s[k]}' for k in ('bad_samples', 'untagged_encoders', 'agg_overflow', 'buffer_overflow') if s[k]]
    if warn:
        print('WARNING: ' + ', '.join(warn))
    if s['merged_encoders']:
        print(f"note: {s['merged_encoders']} encoders held >1 dispatch; their time is attributed to the first kernel")
    print(f"\n{'category':<18}{'net ms/tok':>11}{'%':>7}{'raw ms/tok':>11}{'calls/tok':>11}{'net avg us':>11}")
    for r in cat_rows:
        print(f"{r['category']:<18}{r['ms_per_token']:>11.3f}{100 * r['ms_per_token'] / max(net, 1e-12):>7.1f}"
              f"{r['raw_ms_per_token']:>11.3f}{r['calls_per_token']:>11.1f}{r['avg_us']:>11.1f}")
    print(f"\n{'kernel':<44}{'cat':<14}{'grid':<16}{'tg':<12}{'net ms/tok':>11}{'x/tok':>7}{'net avg us':>11}")
    for r in rows[:top]:
        g = 'x'.join(map(str, r['grid']))
        t = 'x'.join(map(str, r['threads']))
        print(f"{r['name'][:43]:<44}{r['category']:<14}{g:<16}{t:<12}{r['ms_per_token']:>11.3f}"
              f"{r['calls_per_token']:>7.1f}{r['avg_us']:>11.1f}")
    return {'summary': s, 'run': run, 'net_ms_per_token': net, 'categories': cat_rows, 'kernels': rows}


def diff(a, b):
    print('\n== diff (candidate - baseline), NET ms/token')
    ca = {r['category']: r['ms_per_token'] for r in a['categories']}
    cb = {r['category']: r['ms_per_token'] for r in b['categories']}
    print(f"{'category':<18}{'base':>9}{'cand':>9}{'delta':>9}")
    for c in sorted(set(ca) | set(cb), key=lambda c: -(abs(cb.get(c, 0) - ca.get(c, 0)))):
        print(f"{c:<18}{ca.get(c, 0):>9.3f}{cb.get(c, 0):>9.3f}{cb.get(c, 0) - ca.get(c, 0):>+9.3f}")
    ta, tb = a['net_ms_per_token'], b['net_ms_per_token']
    print(f"{'TOTAL net':<18}{ta:>9.3f}{tb:>9.3f}{tb - ta:>+9.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jsonl')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--diff', metavar='BASELINE', help='baseline JSONL; prints candidate-baseline per category')
    ap.add_argument('--json', metavar='OUT', help='write the aggregated result as JSON')
    args = ap.parse_args()
    cand = report(args.jsonl, args.top)
    result = {'candidate': cand}
    if args.diff:
        base = report(args.diff, args.top)
        diff(base, cand)
        result['baseline'] = base
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
