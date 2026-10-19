#!/usr/bin/env python3
"""What one plastic pulse looks like, with every PSA hit the PSA put on it.

Picks large isolated pulses out of a raw chunk and draws the recorded block with
the reconstructed hits marked, so the after-pulse tail can be read off a single
event rather than a statistic.

    python event_display.py <head_N.bin> [--det PSSB] [-n 6]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

REPROC = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
PRE_SAMPLES = 259
T_PHYS = 20_000.0
BR = ['segment', 'BunchNumber', 'detn', 'tof', 'amp_0', 'amp', 'fwhm', 'chi2',
      'pileup1', 'pileup2', 'pulseshape']
PLASTIC, WALL, MUTED, INK = '#0072B2', '#D55E00', '#6b7280', '#20242b'
SURFACE = '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.edgecolor': '#c9ccd1',
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#e6e8ea',
    'grid.linewidth': 0.7, 'axes.axisbelow': True, 'font.size': 9,
    'legend.frameon': False,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('--det', default='PSSB')
    ap.add_argument('-n', type=int, default=6)
    ap.add_argument('--min-amp', type=float, default=4000.0)
    ap.add_argument('--quiet', type=float, default=5000.0)
    ap.add_argument('-o', '--out', default='figures/event_display.png')
    args = ap.parse_args()

    seg = int(''.join(c for c in Path(args.raw).stem if c.isdigit()))
    with uproot.open(REPROC / f'run224572_{seg // 10 + 1:04d}.root') as fh:
        a = fh[args.det].arrays(BR, library='np')
    keep = (a['segment'] == seg) & (a['tof'] > T_PHYS)
    a = {k: v[keep] for k, v in a.items()}

    blocks, bunch = [], -1
    for _o, tag, _v, pay in iter_banks(args.raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC' or pay[0:4].decode('ascii', 'replace') != args.det:
            continue
        _d, chan, blks = parse_acqc(pay, with_samples=True)
        for start, s in blks:
            if start >= T_PHYS:
                blocks.append((bunch, chan, start - PRE_SAMPLES,
                               s.view('<i2').astype(np.float64)))

    grp = a['BunchNumber'].astype(np.int64) * 100 + a['detn']
    t = a['tof'].astype(np.float64)
    amp = a['amp_0'].astype(np.float64)
    o = np.lexsort((t, grp))
    grp, t, amp = grp[o], t[o], amp[o]
    a = {k: v[o] for k, v in a.items()}
    prev = np.full(t.size, np.inf)
    prev[1:] = np.where(grp[1:] == grp[:-1], t[1:] - t[:-1], np.inf)
    cand = np.flatnonzero((amp > args.min_amp) & (prev > args.quiet))

    picked = []
    for i in cand:
        b, c = int(grp[i] // 100), int(grp[i] % 100)
        for bunch, chan, t0, v in blocks:
            if bunch == b and chan == c and t0 <= t[i] < t0 + len(v):
                if len(v) > 500:
                    picked.append((i, t0, v))
                break
        if len(picked) >= args.n:
            break
    if not picked:
        print('nothing found')
        return 1

    ncol = 2
    nrow = int(np.ceil(len(picked) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 2.9 * nrow),
                             squeeze=False)
    for ax, (i, t0, v) in zip(axes.ravel(), picked):
        base = np.median(v[:200])
        y = -(v - base)
        x = np.arange(len(v)) + t0 - t[i]
        m = (x > -60) & (x < 620)
        ax.plot(x[m], y[m], '-', color=PLASTIC, lw=1.0)
        same = ((grp == grp[i]) & (a['tof'] >= t[i] - 60)
                & (a['tof'] <= t[i] + 620))
        hits = np.flatnonzero(same)
        for k in hits:
            d = a['tof'][k] - t[i]
            ax.axvline(d, color=WALL if abs(d) > 1e-6 else MUTED, lw=0.9,
                       alpha=0.85)
        ax.axhline(0, color=MUTED, lw=0.7, ls=':')
        ax.set_yscale('symlog', linthresh=100)
        ax.set_xlim(-60, 620)
        ax.set_xlabel('time from the primary pulse  [ns]')
        ax.set_ylabel('trace  [ADC, sign-flipped]')
        ax.set_title(f'bunch {int(grp[i] // 100)} ch {int(grp[i] % 100)},  '
                     f'peak {amp[i]:,.0f} ADC  |  {len(hits)} PSA hits on this '
                     f'one pulse', fontsize=9, loc='left')
    for ax in axes.ravel()[len(picked):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], color=PLASTIC, lw=1.4, label='raw trace'),
               plt.Line2D([], [], color=WALL, lw=1.2, label='PSA hit time')]
    axes.ravel()[0].legend(handles=handles, fontsize=8, loc='upper right')
    fig.suptitle(f'{args.det}: one zero-suppressed block, one physical pulse — '
                 'and the train of hits the PSA reports on its tail',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(args.out, dpi=150)
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
