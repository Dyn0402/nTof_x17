#!/usr/bin/env python3
"""
summarize_scans.py — collect `run_bench.py` result jsons into one table.

    ../../.venv/bin/python mx_june_wft/bench/summarize_scans.py w        # window scan
    ../../.venv/bin/python mx_june_wft/bench/summarize_scans.py sens     # sensitivity
    ... --key sat_det3 --ref full32

With --ref, every row also shows the change against that row (the paired
comparison: all variants run on the same event subset).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

COLS = [('within5', 'within5 %', '{:7.2f}'), ('reco_far', 'far %', '{:6.2f}'),
        ('core_sigma', 'core mm', '{:8.3f}'), ('median_r', 'med mm', '{:7.3f}'),
        ('sigma_x', 'sigX °', '{:7.3f}'), ('bias_x', 'biasX °', '{:+8.3f}'),
        ('sigma_y', 'sigY °', '{:7.3f}'), ('bias_y', 'biasY °', '{:+8.3f}'),
        ('vspread_x', 'vspX', '{:5.1f}'), ('vspread_y', 'vspY', '{:5.1f}'),
        # the reference-selected metrics: unbiased by construction, unlike the
        # slope_reliable sigma/bias above (see RECO_BENCH §5)
        ('sig14_x', 's14X °', '{:7.3f}'), ('sig14_y', 's14Y °', '{:7.3f}'),
        ('comp14_x', 'cmpX °', '{:+8.3f}'), ('comp14_y', 'cmpY °', '{:+8.3f}'),
        ('s_per_plane', 's/fit', '{:6.2f}')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix', nargs='?', default='')
    ap.add_argument('--key', default='sat_det3')
    ap.add_argument('--ref', default=None, help='tag to difference against')
    ap.add_argument('--dir', default=None)
    args = ap.parse_args()

    if args.dir:
        d = args.dir
    else:
        from qa_config import get_config, setup_paths
        setup_paths()
        d = os.path.join(get_config(args.key).OUT_BASE, 'wft', 'bench')

    rows = {}
    for f in sorted(glob.glob(os.path.join(d, f'bench_{args.prefix}*.json'))):
        s = json.load(open(f))
        tag = os.path.basename(f)[len('bench_'):-len('.json')]
        rows[tag] = s
    if not rows:
        print(f'no results matching bench_{args.prefix}*.json in {d}')
        return

    ref = None
    for t, s in rows.items():
        if args.ref and t.startswith(args.ref):
            ref = s
    hdr = f'{"tag":26s} {"n":>5s} ' + ' '.join(f'{h:>8s}' for _, h, _ in COLS)
    print(hdr)
    print('-' * len(hdr))
    for t, s in rows.items():
        line = f'{t[:26]:26s} {s["n"]:5d} '
        line += ' '.join(f.format(s.get(k, float("nan"))).rjust(8)
                         for k, _, f in COLS)
        print(line)
        if ref is not None and s is not ref:
            dl = f'{"":26s} {"":5s} '
            dl += ' '.join(
                (f'{s.get(k, float("nan")) - ref.get(k, float("nan")):+.3f}'
                 if k in ref else '').rjust(8) for k, _, _ in COLS)
            print(dl)


if __name__ == '__main__':
    main()
