#!/usr/bin/env python3
"""Emit the markdown tables used in ../README.md (so they can be regenerated)."""
import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
OFF_RUNS = [224356, 224357, 224358, 224359, 224360, 224464, 224466]
EPOCH = {224356: '07-11', 224357: '07-11', 224358: '07-11', 224359: '07-11',
         224360: '07-11', 224464: '07-16', 224466: '07-16'}


def rows():
    out = []
    for r in csv.DictReader(open(DATA / 'per_channel_flash_timing.csv')):
        d = {k: (v if k == 'tree' else (float(v) if v != '' else float('nan')))
             for k, v in r.items()}
        out.append(d)
    return out


def f(x, n=1):
    return '—' if not np.isfinite(x) else f'{x:.{n}f}'


def main():
    R = rows()
    off = [r for r in R if int(r['run']) in OFF_RUNS]
    runs = sorted({int(r['run']) for r in off})
    walls = ['WALA', 'WALB', 'WALC', 'WALD']

    print('### Table 1 — per-channel flash arrival, C = t_channel − t_PKUP [ns]\n')
    hdr = '| channel | ' + ' | '.join(str(r) for r in runs) + ' | mean | run-to-run σ | per-bunch σ |'
    print(hdr)
    print('|' + '---|' * (len(runs) + 4))
    for w in walls:
        for ch in range(1, 9):
            sub = {int(r['run']): r for r in off if r['tree'] == w and int(r['ch']) == ch}
            if not sub:
                continue
            vals = [sub[r]['dt_mean'] if r in sub else float('nan') for r in runs]
            good = [v for v in vals if np.isfinite(v)]
            sig = np.nanmean([sub[r]['dt_sigma'] for r in sub])
            print(f'| {w} ch{ch} | ' + ' | '.join(f(v) for v in vals) +
                  f' | **{np.mean(good):.1f}** | {np.std(good):.2f} | {sig:.1f} |')
    print()

    print('### Table 2 — per-wall summary\n')
    print('| wall | C mean [ns] | channel spread (σ / range) | per-bunch σ [ns] | flash amp [ADC] | risetime [ns] |')
    print('|---|---|---|---|---|---|')
    for w in walls:
        sub = [r for r in off if r['tree'] == w]
        c = [np.mean([r['dt_mean'] for r in sub if int(r['ch']) == ch])
             for ch in sorted({int(r['ch']) for r in sub})]
        print(f'| {w} | **{np.mean(c):.1f}** | {np.std(c):.1f} / {max(c)-min(c):.1f} '
              f'| {np.mean([r["dt_sigma"] for r in sub]):.1f} '
              f'| {np.median([r["amp_med"] for r in sub]):.0f} '
              f'| {np.median([r["rise_med"] for r in sub]):.1f} |')
    allc = [r['dt_mean'] for r in off if r['tree'] in walls]
    print(f'| **all 32** | **{np.mean(allc):.1f}** | | | | |')
    print()

    print('### Table 3 — beam-intensity dependence (time walk)\n')
    print('| tree | Δt parasitic 4.1e12 | Δt dedicated 8.5e12 | walk | amp ratio |')
    print('|---|---|---|---|---|')
    for w in walls + ['PSSA', 'PSSB', 'PSSC', 'PSSD']:
        sub = [r for r in off if r['tree'] == w and np.isfinite(r['dt_lo']) and np.isfinite(r['dt_hi'])]
        if not sub:
            continue
        lo = np.mean([r['dt_lo'] for r in sub]); hi = np.mean([r['dt_hi'] for r in sub])
        ar = np.nanmean([r['amp_hi'] / r['amp_lo'] for r in sub])
        print(f'| {w} | {lo:.1f} | {hi:.1f} | **{hi-lo:+.1f}** | {ar:.2f} |')
    print()

    print('### Table 4 — plastics (PSS), same reference\n')
    print('| channel | C [ns] | per-bunch σ | usable bunches | flash amp |')
    print('|---|---|---|---|---|')
    for t in ['PSSA', 'PSSB', 'PSSC', 'PSSD']:
        for ch in (1, 2):
            sub = [r for r in off if r['tree'] == t and int(r['ch']) == ch]
            if not sub:
                continue
            print(f'| {t} ch{ch} | **{np.mean([r["dt_mean"] for r in sub]):.1f}** '
                  f'| {np.mean([r["dt_sigma"] for r in sub]):.1f} '
                  f'| {np.mean([r["frac_core"] for r in sub]):.0%} '
                  f'| {np.median([r["amp_med"] for r in sub]):.0f} |')
    print()

    print('### Table 5 — epoch comparison (07-11 vs 07-16), wall channels\n')
    e1 = [r['dt_mean'] for r in off if r['tree'] in walls and EPOCH[int(r['run'])] == '07-11']
    e2 = [r['dt_mean'] for r in off if r['tree'] in walls and EPOCH[int(r['run'])] == '07-16']
    if e1 and e2:
        print(f'| epoch | runs | mean C [ns] | n |')
        print('|---|---|---|---|')
        print(f"| 2026-07-11 | {[r for r in runs if EPOCH[r]=='07-11']} | {np.mean(e1):.1f} | {len(e1)} |")
        print(f"| 2026-07-16 | {[r for r in runs if EPOCH[r]=='07-16']} | {np.mean(e2):.1f} | {len(e2)} |")
        print(f"\ndifference: **{np.mean(e2)-np.mean(e1):+.2f} ns**")


if __name__ == '__main__':
    main()
