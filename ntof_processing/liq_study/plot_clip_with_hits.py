#!/usr/bin/env python3
"""Draw a clipped raw block with the PSA hits of that same (det, segment, bunch)
overlaid, so it can be seen WHICH pulse carries the saturation flag.

The per-pulse check (`verify_satuflag.py`) finds that at physics time the nearest
flagged hit sits ~258 ns before the clipped sample, while an unflagged hit sits
~29 ns from it. Either the PSA time base is offset for zero-suppressed blocks, or
the flag is landing on the wrong pulse. Only the waveform decides.

    python plot_clip_with_hits.py <outdir> <reproc_dir> <raw_head.bin> <clips.txt>

Draws every physics-time clip found in that chunk.
"""
import sys
from pathlib import Path

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

BR = ['segment', 'BunchNumber', 'tof', 'amp', 'satuflag', 'chi2', 'pileup1']
PAD = 600            # ns drawn either side of the clip


def main():
    outdir, reproc, raw, clipfile = (Path(sys.argv[1]), Path(sys.argv[2]),
                                     sys.argv[3], Path(sys.argv[4]))
    outdir.mkdir(parents=True, exist_ok=True)
    want = []
    for line in clipfile.read_text().splitlines():
        p = line.split()
        if len(p) == 7 and p[6] == 'physics':
            want.append(dict(det=p[0], seg=int(p[1]), bunch=int(p[2]),
                             t=int(p[4]), n=int(p[5])))
    if not want:
        print('no physics-time clips in', clipfile)
        return 0

    found, bunch = [], -1
    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC':
            continue
        det, _c, blks = parse_acqc(pay, with_samples=True)
        for start, s in blks:
            for w in want:
                if w['det'] != det or w['bunch'] != bunch:
                    continue
                if start <= w['t'] < start + len(s):
                    found.append(dict(w, start=start,
                                      v=s.view('<i2').astype(np.int64)))
    print(f'{len(found)} of {len(want)} physics-time clips located in the raw chunk')
    if not found:
        return 0

    cache = {}
    fig, axes = plt.subplots(1, len(found), figsize=(6.2 * len(found), 4.8),
                             squeeze=False)
    for ax, b in zip(axes[0], found):
        part = b['seg'] // 10 + 1
        key = (part, b['det'])
        if key not in cache:
            cache[key] = uproot.open(reproc / f'run224572_{part:04d}.root'
                                     )[b['det']].arrays(BR, library='np')
        a = cache[key]
        m = (a['segment'] == b['seg']) & (a['BunchNumber'] == b['bunch'])
        tof, amp, sat = a['tof'][m], a['amp'][m], a['satuflag'][m]

        i0 = b['t'] - b['start']
        lo, up = max(0, i0 - PAD), min(len(b['v']), i0 + PAD)
        t = np.arange(lo, up) + b['start']
        ax.axhline(-32768, color='crimson', lw=1.2, label='rail')
        base = float(np.median(b['v'][:40]))
        ax.axhline(base, color='0.5', lw=0.9, ls=':', label='baseline')
        ax.plot(t, b['v'][lo:up], color='tab:blue', lw=1.2, label='raw (int16)')

        sel = (tof > t[0]) & (tof < t[-1])
        for k in np.flatnonzero(sel):
            col = 'crimson' if sat[k] else 'tab:green'
            ax.axvline(tof[k], color=col, lw=1.6 if sat[k] else 0.8,
                       alpha=0.9 if sat[k] else 0.5)
            if sat[k] or amp[k] > 5000:
                ax.annotate(f'{amp[k]:.0f}{" SAT" if sat[k] else ""}',
                            (tof[k], base), rotation=90, fontsize=6,
                            color=col, ha='right', va='top')
        ax.set_title(f'{b["det"]}  seg {b["seg"]} bunch {b["bunch"]}\n'
                     f'clip at {b["t"]} ns ({b["n"]} samples at rail); '
                     f'{int(sel.sum())} PSA hits drawn', fontsize=8)
        ax.set_xlabel('sample index in movie [ns]')
        ax.set_ylabel('sample value [ADC, signed]')
        ax.legend(fontsize=7, loc='lower right')

        near = np.flatnonzero(sel & (sat != 0))
        print(f'\n{b["det"]} seg {b["seg"]} bunch {b["bunch"]} clip at {b["t"]}:')
        for k in np.flatnonzero(sel):
            print(f'    tof {tof[k]:12.1f}  dt {tof[k] - b["t"]:+8.1f}  '
                  f'amp {amp[k]:9.0f}  satu {sat[k]}  pileup {a["pileup1"][m][k]}')
        if len(near) == 0:
            print('    -> no flagged hit inside the drawn window')
    fig.suptitle('Physics-time liquid clips with the PSA hits of the same bunch '
                 'overlaid (red = satuflag set)', fontsize=11)
    fig.tight_layout()
    p = outdir / f'clip_with_hits_{Path(raw).stem}.png'
    fig.savefig(p, dpi=130)
    print('\nwrote', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
